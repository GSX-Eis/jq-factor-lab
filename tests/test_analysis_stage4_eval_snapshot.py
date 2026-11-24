"""Snapshot production evaluation adapters from the repository on PYTHONPATH."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SEED = 431
DAILY_HORIZONS = (1, 2)


def daily_snapshot(root: Path, output: Path) -> None:
    from backtest_and_analysis.alphalens_runner import AlphalensRunner
    from backtest_and_analysis.evaluation import EvaluationEngine
    from backtest_and_analysis.evaluation.cross_section import hac_p_value
    from backtest_and_analysis.evaluation.labels import label_daily_returns
    from backtest_and_analysis.evaluation.panel import EvaluationPanel
    from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
    from backtest_and_analysis.price_fetcher import PriceFetcher

    factors = pd.read_parquet(root / 'daily_factors.parquet')
    begin, end = factors.index[0].date().isoformat(), factors.index[-1].date().isoformat()
    fetcher = PriceFetcher(root / 'ashare.duckdb', 'price_data')
    try:
        prices = fetcher.fetch(factors.columns.tolist(), begin, end, future_periods=0)
    finally:
        fetcher.close()
    runner = AlphalensRunner(n_quantiles=3, periods=DAILY_HORIZONS)
    returns = ForwardReturnsCalculator().calculate_forward_returns(prices, DAILY_HORIZONS, runner.execution_lag)
    results = runner.run(factors, prices, returns)
    for name in ('ic', 'ir', 'quantile_returns', 'std_err'):
        assert not results[name].empty, name
        table = results[name].copy()
        table.attrs.clear()
        table.to_parquet(output / f'{name}.parquet')
    returns.to_parquet(output / 'daily_returns.parquet')
    long = factors.rename_axis(index='date', columns='asset').stack().rename('factor').to_frame()
    panel = EvaluationPanel.from_long(long.join(returns))
    close_clock = (panel.datetimes.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)).tz_convert('UTC')
    daily = label_daily_returns(prices, None, runner.execution_lag, DAILY_HORIZONS)
    exits, entries = {}, {}
    for horizon in DAILY_HORIZONS:
        days = daily.scheduled_exit_dates(horizon).reindex(panel.datetimes)
        clock = (pd.DatetimeIndex(days).tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)).tz_convert('UTC')
        key = f'{horizon}D'
        exits[key] = np.broadcast_to(clock.to_numpy(dtype='datetime64[ns]')[:, None], panel.factor.shape)
        endpoint = daily.endpoints(horizon).entry_date.unstack().reindex(index=panel.datetimes, columns=panel.assets)
        entry_calendar = pd.Series(close_clock, index=panel.datetimes)
        entries[key] = np.column_stack([
            endpoint[asset].map(entry_calendar).to_numpy(dtype='datetime64[ns]') for asset in panel.assets
        ])
    report = EvaluationEngine('cross_sectional').evaluate(
        panel, n_quantiles=3, split={'train_end': close_clock[8], 'oos_start': close_clock[10]},
        signal_available_at=close_clock, exit_timestamps=exits, hac_lag='auto',
    )
    metrics, slot_reference = {}, {}
    for name, split in report.splits.items():
        stats = split.statistics
        metrics[f'{name}_turnover'] = stats.turnover.tolist()
        metrics[f'{name}_dates'] = [
            (value.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)).tz_convert('UTC').isoformat().replace('+00:00', 'Z')
            for value in stats.datetimes
        ]
        slot_reference[name] = {}
        for horizon, row in stats.horizons.items():
            prefix = f'{name}_{horizon}'
            p_value, _ = hac_p_value(row.hac, row.ic_summary.mean)
            t_value = row.ic_summary.mean / np.sqrt(row.hac.variance) if row.hac.variance > 0 else None
            slot_reference[name][horizon] = dict(
                count=row.ic_summary.n_valid, mean=row.ic_summary.mean, icir=row.ic_summary.icir,
                buckets={str(q): value.mean for q, value in row.bucket_summary.items()},
                spread=row.spread_summary.mean, p_value=p_value, hac_t=t_value,
                hac_variance=row.hac.variance, purged=split.split_purged_count[horizon],
            )
            metrics[prefix] = dict(
                rank_ic=row.ic.values.tolist(), icir=row.ic_summary.icir,
                quantiles=row.bucket_returns.tolist(), long_short=row.spread.tolist(),
                pairs=row.ic.counts.tolist(), purged=split.split_purged_count[horizon], hac_t=t_value, hac_p=p_value,
                hac_ci=[row.hac.lower, row.hac.upper],
            )
    (output / 'daily_inference.json').write_text(json.dumps(metrics, sort_keys=True))
    (output / 'daily_slot_reference.json').write_text(json.dumps(slot_reference, sort_keys=True))
    time_series = EvaluationEngine('time_series').evaluate(
        panel, n_quantiles=3, split={'train_end': close_clock[8], 'oos_start': close_clock[10]},
        signal_available_at=close_clock, entry_timestamps=entries, exit_timestamps=exits,
        block_length=3, seed=SEED, n_boot=29, min_valid_replicates=10,
    )
    bootstrap = {
        f'{name}_{horizon}_{asset}': asdict(instrument.bootstrap)
        for name, split in time_series.splits.items()
        for horizon, result in split.statistics.horizons.items()
        for asset, instrument in result.instruments.items()
    }
    (output / 'daily_bootstrap.json').write_text(json.dumps(bootstrap, sort_keys=True))



def minute_snapshot(root: Path, output: Path) -> None:
    from tools.agent_harness.time_series_analysis import evaluate

    config = json.loads((root / 'minute_legacy.json').read_text())
    evaluate(config, output)


if __name__ == '__main__':
    import backtest_and_analysis

    assert Path(backtest_and_analysis.__file__).resolve().is_relative_to(Path.cwd())
    inputs, destination = map(Path, sys.argv[1:])
    daily_snapshot(inputs, destination)
    minute_snapshot(inputs, destination / 'minute')
    minute_snapshot(inputs / 'minute-cross', destination / 'minute-cross')
