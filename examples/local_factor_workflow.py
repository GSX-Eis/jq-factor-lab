"""Synthetic CSV -> DuckDB -> factors -> composite -> labels -> JoinQuant bundle."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

FACTOR_SOURCE = '''from utils.factor_base import FactorBase

class TutorialFast(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(["close", "factor"], table_name="price_data_daily")

    def calculate_factor(self, data):
        prices = data.pivot(index="date", columns="code", values="close").sort_index()
        return prices.pct_change(5, fill_method=None)

class TutorialSlow(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(["close", "factor"], table_name="price_data_daily")

    def calculate_factor(self, data):
        prices = data.pivot(index="date", columns="code", values="close").sort_index()
        return prices.pct_change(10, fill_method=None)

class TutorialComposite(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(["close", "factor"], table_name="price_data_daily")

    def calculate_factor(self, data):
        prices = data.pivot(index="date", columns="code", values="close").sort_index()
        # The generated strategy computes at context.previous_date.  Keep this
        # formula contemporaneous so that strategy execution on t uses the
        # same t-1 inputs as the offline synthesis signal on t.
        return (0.5 * prices.pct_change(5, fill_method=None)
                + 0.5 * prices.pct_change(10, fill_method=None))
'''


def run_workflow(output_dir):
    import duckdb
    import numpy as np
    import pandas as pd
    from backtest_and_analysis.factor_loader import FactorLoader
    from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
    from data_pipeline.loaders.price_data_to_duckdb import DuckDBPriceImporter
    from factor_synthesis.factor_reader import FactorReader
    from factor_synthesis.pipeline.weighted_sum_method import run_weighted_sum_synthesis
    from scripts.build_jq_strategy import build_jq_strategy_main
    from utils.duckdb_results_writer import DuckDBResultsWriter

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    daily = root / 'csv' / 'daily'
    daily.mkdir(parents=True)
    dates = pd.bdate_range('2024-01-02', periods=80)
    rng = np.random.default_rng(42)
    codes = [f'DEMO{i:03d}' for i in range(12)]
    # Full importer schema; no external data or private configuration.
    for code in codes:
        close = 100 * np.exp(rng.normal(0, .01, len(dates)).cumsum())
        pd.DataFrame(dict(date=dates, open=close, high=close, low=close, close=close,
                          volume=1000., money=1000*close, factor=1., paused=0,
                          high_limit=close*1.1, low_limit=close*.9, pre_close=close)).to_csv(daily/f'{code}.csv', index=False)
    market = root/'market.duckdb'
    importer = DuckDBPriceImporter(str(market), export_parquet=False)
    importer.connect()
    try:
        importer.init_tables()
        files, successes, rows = importer.import_directory('daily', str(daily), 'price_data_daily')
        if files != len(codes) or successes != files or rows != len(dates)*len(codes):
            raise RuntimeError('Synthetic price import was incomplete')
    finally:
        importer.close()

    source = root/'user_factors.py'
    source.write_text(FACTOR_SOURCE)
    classes = {cls.__name__: cls for cls in FactorLoader.load_factor_class_from_file(str(source))}
    requirements = root/'requirements.json'
    requirements.write_text(json.dumps({'factors': {name: {'requirements': {'daily': {
        'min_days': 12, 'fields': ['close', 'factor']}}} for name in classes}}, indent=2))
    names = ['TutorialFast', 'TutorialSlow']
    store = root/'factors.duckdb'
    writer = DuckDBResultsWriter(str(store), export_parquet=False)
    local = {}
    for name, cls in classes.items():
        factor = cls(None, name, str(dates[0].date()), str(dates[-1].date()),
                     stock_pool=codes, duckdb_path=str(market), pit_enabled=True)
        try:
            local[name] = factor.get_factor_values(force_recalculate=True)
        finally:
            factor._duckdb_adapter.close()
        if name in names:
            writer.save_factor_values(local[name], name)
    reader = FactorReader(str(store))
    loaded = reader.get_multiple_factors(names, str(dates[0].date()), str(dates[-1].date()))
    features = pd.DataFrame({name: values.rename_axis(index='date', columns='stock_code').stack()
                             for name, values in loaded.items()}).dropna().sort_index()
    features.to_csv(root/'features.csv')

    captured = []
    context = SimpleNamespace(logger=logging.getLogger(__name__), config={
        'factor_groups': {'tutorial': {'factors': names}},
        'output_settings': {'composite_factor_prefix': 'tutorial', 'weights_output_dir': str(root/'weights')},
        'performance_evaluation': {'skip_alphalens_analysis': True}},
        direct_analyze_composite_factor=lambda values, *_a, **_kw: captured.append(values))
    run_weighted_sum_synthesis(context, 'equal', {'factor_group': 'tutorial',
        'weights': dict.fromkeys(names, .5), 'preprocess': dict.fromkeys(names, 'none')}, features)
    scores = captured[0].sort_index()
    # weighted_sum shifts the feature panel once.  A deployed strategy is
    # called on t with context.previous_date=t-1, which applies that same
    # delay without shifting the factor formula itself.
    deploy_values = (local['TutorialComposite'].shift(1)
                     .rename_axis(index='date', columns='stock_code').stack().sort_index())
    pd.testing.assert_series_equal(scores, deploy_values, check_names=False)
    scores.rename('score').to_csv(root/'scores.csv')
    writer.save_factor_values(scores.unstack('stock_code'), 'TutorialComposite')

    with duckdb.connect(str(market), read_only=True) as connection:
        prices = connection.execute('SELECT date, stock_code, close, factor FROM price_data_daily').fetchdf()
    prices['adjusted_close'] = prices.close * prices.factor
    price_series = prices.set_index(['date','stock_code']).adjusted_close
    price_series.index.names = ['date','asset']
    # Explicit synthetic market for this generated tutorial, not a real A-share calendar.
    from utils.trading_calendar import TradingCalendar
    calendar = TradingCalendar(dates, dates.min(), dates.max(), 'TUTORIAL_SYNTHETIC')
    labels = ForwardReturnsCalculator(trading_calendar=calendar).calculate_forward_returns(price_series, (1,5), execution_lag=0)
    labels.index.names = ['date','stock_code']
    labels.to_csv(root/'labels.csv')
    paired = scores.rename('score').to_frame().join(labels).dropna()
    if paired.empty:
        raise RuntimeError('Composite and price-derived labels have no common observations')
    paired.to_csv(root/'evaluation_rows.csv')

    bundle = root/'joinquant_strategy.py'
    status = build_jq_strategy_main(['--public', '--factor-files', str(source), '--factors', 'TutorialComposite',
        '--requirements', str(requirements), '--single-file', '--out', str(bundle)])
    if status != 0 or not bundle.is_file():
        raise RuntimeError('JoinQuant bundle generation failed')
    compile(bundle.read_text(), str(bundle), 'exec')

    # Execute the emitted bundle with a local get_price implementation, rather
    # than only compiling it.  This asserts the strategy's previous_date
    # convention against the offline synthesis date convention.
    market_prices = prices.rename(columns={'stock_code': 'code'}).copy()

    def mock_get_price(security, end_date, count=None, start_date=None, fields=None, **_kwargs):
        securities = list(security) if isinstance(security, (list, tuple)) else [security]
        frame = market_prices[market_prices.code.isin(securities)].copy()
        frame = frame[frame.date <= pd.Timestamp(end_date).normalize()]
        if start_date is not None:
            frame = frame[frame.date >= pd.Timestamp(start_date).normalize()]
        elif count is not None:
            frame = (frame.sort_values(['code', 'date'])
                     .groupby('code', group_keys=False).tail(int(count)))
        price_fields = [field for field in (fields or ['close', 'factor']) if field in frame]
        return frame[['date', 'code', *price_fields]].reset_index(drop=True)

    # Compatibility aliases are process-global; restore them after local replay.
    aliases = ('factor_base', 'utils.factor_base', 'jq_adapter', 'utils.jq_adapter')
    missing = object()
    saved_modules = {name: sys.modules.get(name, missing) for name in aliases}
    try:
        spec = importlib.util.spec_from_file_location('tutorial_joinquant_strategy', bundle)
        bundle_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bundle_module)
        bundle_module.set_external_get_price(mock_get_price)
        execution_date = pd.Timestamp(dates[-1])
        previous_date = pd.Timestamp(dates[-2])
        deployed = bundle_module.compute_factor_batch(
            {'TutorialComposite': bundle_module.TutorialComposite},
            bundle_module.FACTOR_REQUIREMENTS, bundle_module.FACTOR_DEPENDENCIES,
            codes, str(previous_date.date()), bundle_module.JQFactorAdapter,
            windows=bundle_module.FACTOR_REQUIRED_WINDOWS,
        )['TutorialComposite'].sort_index()
        offline = scores.xs(execution_date, level='date').sort_index()
        pd.testing.assert_series_equal(deployed, offline, check_names=False)
    finally:
        for name, original in saved_modules.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    summary = {'data': 'synthetic', 'imported_rows': rows, 'factor_rows': len(features),
               'score_rows': len(scores), 'evaluation_rows': len(paired),
               'label_definition': 'adjusted_close[t + period] / adjusted_close[t] - 1',
               'signal_delay_sessions': 1, 'execution_lag': 0,
               'bundle_previous_date': str(previous_date.date()),
               'bundle_execution_date': str(execution_date.date()),
               'synthesis': 'equal weights; no model training', 'bundle': str(bundle)}
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            result = run_workflow(args.output_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Workflow failed: {exc}; use a new writable output directory and inspect the named input\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
