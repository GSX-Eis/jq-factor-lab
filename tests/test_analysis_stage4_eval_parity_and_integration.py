"""Synthetic Stage 4 parity against master and production Harness round trips."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb

from market_db_samples import declare_market_db
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series.labels import LABEL_COLUMNS
from slots import build_slot_from_config
from test_analysis_stage4_eval_snapshot import DAILY_HORIZONS, SEED, daily_snapshot
from test_slot_futures import _populate
from utils import bar_evaluation_app
from utils.market_db.identity import bind_market_db_fingerprint, slot_market_db_identity
from utils.market_factor.identity import build_identity_record, compute_factor_identity_hash
from utils.market_factor_store import MarketFactorValueStore
from utils.session_calendar import SessionCalendar
from utils.slots import SlotConfig
from utils.time_axis import Frequency

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '680e50b5f79c37fd65f52ac40777488a76dbcce4'
DAYS = ('2026-01-05', '2026-01-30')
FUTURES_DAYS = ('2026-01-05', '2026-01-09')


def _native(provider, start, end, profile, **bar_options):
    return {
        'schema_version': 1, 'slot_id': provider, 'execution_profile': profile,
        'calendar': {'provider': provider, 'options': {'start': start, 'end': end}},
        'bar_feed': {'provider': provider, 'options': bar_options},
        **{field: {'provider': provider} for field in ('universe', 'aux_features', 'microstructure')},
    }


def _save_factors(root, slot, frequency, mode, frame, strategy):
    fingerprint = bind_market_db_fingerprint({'fixture_seed': SEED}, slot_market_db_identity(slot))
    digest, payload = compute_factor_identity_hash(
        factor_id='synthetic', factor_class='Synthetic', source_sha256='a' * 64, params={},
        frequency=frequency, session_view='ETH', analysis_mode=mode, bucket_strategy=strategy,
        calendar_hash=slot.calendar.calendar_hash, slot_id=slot.slot_id,
        slot_config_hash=slot.slot_config_hash, adjustment_policy='none',
        slot_data_fingerprint=fingerprint, code_version='fixture',
    )
    record = build_identity_record(factor_identity_hash=digest, payload=payload, code_version_source='test')
    MarketFactorValueStore(root).append_factor_values(
        frame.assign(factor_id='synthetic', factor_identity_hash=digest, frequency=frequency, session_view='ETH',
                     source_build_id='fixture-build', calendar_hash=slot.calendar.calendar_hash,
                     code_version='fixture', created_at=pd.Timestamp('2026-02-01T00:00:00Z'), batch_id='fixture'),
        identity_record=record, calendar=slot.calendar,
    )


def _evaluate_cli(root, name, command, config, db, *arguments):
    path = root / f'{name}.json'
    path.write_text(json.dumps(config))
    return _cli(root, name, command, '--config', str(path), '--results-db', str(db),
                '--output-dir', str(root / f'{name}-evaluation'), *arguments)


def _cli(root, name, command, *arguments):
    return _cli_result(root, name, command, *arguments)['metrics']


def _cli_result(root, name, command, *arguments, exit_code=0):
    run = subprocess.run(
        [sys.executable, '-m', 'tools.agent_harness.cli', command, '--output-root', str(root / 'runs'),
         '--run-id', name, *arguments],
        cwd=ROOT, capture_output=True, text=True, timeout=90,
    )
    assert run.returncode == exit_code, run.stdout + run.stderr
    result = json.loads(run.stdout)
    assert result['exit_code'] == exit_code
    return result


def _ashare(root):
    db = root / 'ashare.duckdb'
    store = root / 'ashare-factors'
    profile = dict(entry_price_field='close', exit_price_field='close', min_entry_offset_bars=0,
                   exit_adjustment=0, return_unit='decimal', gap_rule='endpoints_only',
                   boundary_policy='cross_session')
    config = _native('jq_ashare_local', *DAYS, profile)
    calendar = SessionCalendar.from_pmc(*DAYS)
    assets = [f'{i:06d}.XSHE' for i in range(1, 7)]
    days = calendar.sessions()
    rng = np.random.default_rng(SEED)
    prices = 100 * np.exp(rng.normal(0, .02, (len(days), len(assets))).cumsum(axis=0))
    prices[:3, 0] = [90., 100., 100.]
    prices[:3, 1] = [100., 110., 132.]
    factors = pd.DataFrame(rng.normal(size=prices.shape), index=days, columns=assets)
    factors.to_parquet(root / 'daily_factors.parquet')
    raw = pd.DataFrame(prices, index=days, columns=assets).rename_axis(index='date', columns='stock_code')
    raw = raw.stack().rename('close').reset_index()
    with duckdb.connect(str(db)) as conn:
        declare_market_db(conn, 'cn_equity')
        dates = pd.date_range(calendar.start, calendar.end)
        conn.from_df(pd.DataFrame({
            'calendar_id': calendar.calendar_id, 'date': dates.date, 'is_open': dates.isin(days),
        })).create('trading_calendar')
        conn.from_df(raw.assign(open=raw.close, high=raw.close + 1, low=raw.close - 1,
                                volume=100, money=100 * raw.close,
                                factor=np.where((raw.stock_code == assets[0]) & (raw.date >= days[2]), 2., 1.))).create('price_data_daily')
        slot = build_slot_from_config(SlotConfig.from_mapping(config), connection=conn)
        keys = slot.calendar.bar_grid('1d')
        values = keys.merge(pd.DataFrame({'instrument_id': assets}), how='cross')
        _save_factors(store, slot, '1d', 'cross_sectional',
                      values.assign(factor_value=factors.to_numpy().ravel()), 'session')
    return dict(slot=config, slot_db=str(db), factor_store_root=str(store), factor_id='synthetic', frequency='1d',
                start_date=DAYS[0], end_date=DAYS[1], horizons=list(DAILY_HORIZONS),
                analysis_mode='cross_sectional', n_quantiles=3, hac_lag='auto',
                split={'train_end': keys.timestamp_utc.iloc[8].isoformat(),
                       'oos_start': keys.timestamp_utc.iloc[10].isoformat()})


def _futures(root, frequency, mode="time_series"):
    db = root / f'futures-{frequency}.duckdb'
    store = root / f'futures-{frequency}-factors'
    expiries = ['2026-03'] if mode == 'time_series' else ['2026-03', '2026-04', '2026-05']
    with duckdb.connect(str(db), config={'threads': 1}) as conn:
        from slots.cn_futures import build_cn_futures_slot
        raw, calendar = _populate(conn, root='CU', exchange='SHFE', start=FUTURES_DAYS[0],
                                  end=FUTURES_DAYS[1], expiries=expiries, frequency=frequency)
        slot = build_cn_futures_slot(conn, product_root='CU', start=FUTURES_DAYS[0], end=FUTURES_DAYS[1])
        config = _native('cn_futures', *FUTURES_DAYS, asdict(slot.microstructure.execution_profile), product_root='CU')
        slot = build_slot_from_config(SlotConfig.from_mapping(config), connection=conn)
        keys = slot.calendar.bar_grid(frequency, bucket_strategy='segment_anchored')
        observations = keys.merge(pd.DataFrame({'instrument_id': sorted(raw.instrument_id.unique())}), how='cross')
        factors = np.random.default_rng(SEED).permutation(len(observations)) / 16.
        values = observations.assign(factor_value=factors)
        _save_factors(store, slot, frequency, mode, values,
                      'session' if frequency == '1d' else 'segment_anchored')
    options = dict(analysis_mode=mode, n_quantiles=2, horizons=[1, 2], trial_selection={})
    if mode == 'time_series':
        options['bootstrap'] = {'seed': SEED, 'n_boot': 29, 'min_valid_replicates': 10}
    else:
        options['hac_lag'] = 'auto'
    if frequency != '1d':
        csv_path = root / 'minute.csv'
        raw.merge(values, on=['timestamp_utc', 'session_id', 'instrument_id']).drop(
            columns='session_id').to_csv(csv_path, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
        options['split'] = {'train_end': keys.timestamp_utc.iloc[len(keys) // 3].isoformat(),
                            'oos_start': keys.timestamp_utc.iloc[len(keys) // 2].isoformat()}
        legacy = dict(options, bars_csv=str(csv_path), calendar=calendar.calendar_id,
                      start_date=FUTURES_DAYS[0], end_date=FUTURES_DAYS[1], bucket_strategy='segment_anchored',
                      factor_id='synthetic')
        (root / 'minute_legacy.json').write_text(json.dumps(legacy))
        options['boundary_policy'] = 'same_session'
    return dict(options, slot=config, slot_id='cn_futures', slot_db=str(db), factor_store_root=str(store),
                factor_id='synthetic', frequency=frequency, start_date=FUTURES_DAYS[0], end_date=FUTURES_DAYS[1])


def _declarative(root, frequency):
    start, end = '2026-01-05', '2026-03-13'
    config = dict(
        schema_version=1, slot_id='declarative',
        calendar={'provider': 'custom_sessions', 'options': {
            'calendar_id': 'synthetic', 'start': start, 'end': end,
            'custom_sessions': {'timezone': 'UTC', 'weekdays': list(range(5)),
                                'segments': [{'open': '09:00', 'close': '16:00'}]},
        }},
        bar_feed={'provider': 'csv_files', 'options': {
            'paths': ['custom.csv'], 'frequency': frequency, 'timestamp_anchor': 'bar_end',
            'source_timezone': 'UTC', 'volume_mode': 'interval', 'data_version': 'fixture',
            'column_mapping': {'instrument_id': 'instrument_id', 'time': 'timestamp',
                               **{name: name for name in ('open', 'high', 'low', 'close', 'volume')}},
        }},
        universe={'provider': 'static', 'options': {'instruments': {'X': {'asset_class': 'crypto', 'currency': 'USD'}}}},
        aux_features={'provider': 'none'}, microstructure={'provider': 'open'},
        execution_profile=dict(entry_price_field='open', exit_price_field='close', min_entry_offset_bars=1,
                               exit_adjustment=-1, return_unit='percent', gap_rule='require_complete_bars',
                               boundary_policy='cross_session'),
    )
    file = root / 'custom.csv'
    file.touch()
    slot = build_slot_from_config(SlotConfig.from_mapping(config), base_dir=root)
    keys = slot.calendar.bar_grid(frequency)
    from utils.time_axis import BarSchedule

    schedule = BarSchedule(slot.calendar, frequency)
    i = np.arange(len(keys))
    price = 100 + i / 10
    closes = price + np.sin(i * .77)
    frame = keys.assign(instrument_id='X', factor_value=np.cos(i * .63))
    source = frame.assign(time=keys.timestamp_utc.map(schedule.available_at), open=price,
                          high=price + 1, low=price - 1, close=closes, volume=1)
    source.drop(columns=['timestamp_utc', 'session_id', 'factor_value']).to_csv(
        file, index=False, date_format='%Y-%m-%dT%H:%M:%SZ')
    store = root / 'custom-factors'
    _save_factors(store, slot, frequency, 'time_series', frame, 'segment_anchored')
    manifest = root / 'slot.json'
    manifest.write_text(json.dumps(config))
    return dict(slot_config=str(manifest), factor_store_root=str(store), factor_id='synthetic',
                start_date=start, end_date=end, frequency=frequency, horizons=[1],
                analysis_mode='time_series', n_quantiles=2,
                bootstrap={'seed': SEED, 'n_boot': 19, 'min_valid_replicates': 5})


def _artifact(root, run, name='summary.json'):
    return root / f'{run}-evaluation' / name


def _compare(actual, expected, path, counts):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            _compare(actual[key], value, f'{path}.{key}', counts)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _compare(a, e, f'{path}[{i}]', counts)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if actual is None and np.isnan(expected):
            actual = np.nan
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True, err_msg=path)
        if np.isfinite(expected):
            counts['finite'] += 1
            counts['max_abs_diff'] = max(counts['max_abs_diff'], float(abs(actual - expected)))
        else:
            counts['nan'] += 1
    else:
        assert actual == expected, path
        counts['exact'] += 1




@pytest.mark.parametrize('frequency', ['1h', '1w'])
def test_stage4_declarative_full_frequency_roundtrip(tmp_path, frequency):
    db = tmp_path / 'custom-evaluation.duckdb'
    config = _declarative(tmp_path, frequency)
    metrics = _evaluate_cli(tmp_path, 'custom', 'time-series-analysis', config, db, '--slot-id', 'declarative')
    assert metrics['results_db']['status'] == 'inserted'
    assert metrics['exclusion_summary']['valid_count'] > 0
    found = _cli(tmp_path, 'read-custom', 'read-bar-evaluations', '--results-db', str(db),
                 '--slot-id', 'declarative')
    assert found['run_count'] == 1
    assert found['runs'][0]['frequency'] == Frequency.parse(frequency).canonical
    assert bar_evaluation_app.get_bar_evaluation_filter_options(db)['slot_ids'] == ['declarative']


def test_stage4_futures_daily_roundtrip(tmp_path):
    config = _futures(tmp_path, '1d')
    db = tmp_path / 'futures-evaluation.duckdb'
    metrics = _evaluate_cli(tmp_path, 'futures-daily', 'single-factor-analysis', config, db)
    assert metrics['results_db']['status'] == 'inserted'
    assert metrics['exclusion_summary']['valid_count'] > 0


@pytest.mark.parametrize('offset, expected', [
    (None, [0., .2]), (1, [0., .2]), (0, [100. / 90. - 1, .1]),
])
def test_ashare_harness_default_and_explicit_zero_entry_offset(tmp_path, offset, expected):
    config = _ashare(tmp_path)
    if offset is not None:
        config['entry_offset_bars'] = offset
    _evaluate_cli(tmp_path, 'ashare', 'single-factor-analysis', config, tmp_path / 'eval.duckdb')
    labels = pd.read_csv(_artifact(tmp_path, 'ashare', 'per_bar_labels.csv'))
    first = labels.loc[labels.signal_timestamp.eq(labels.signal_timestamp.min()) & labels.horizon_bars.eq(1)]
    np.testing.assert_allclose(first.return_pct.iloc[:2], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(first.entry_price, first.entry_return_price)
    np.testing.assert_array_equal(first.exit_price, first.exit_return_price)


@pytest.fixture
def legacy_daily_config(tmp_path):
    inputs = _ashare(tmp_path)
    assets = pd.read_parquet(tmp_path / 'daily_factors.parquet').columns.tolist()
    factor = tmp_path / 'fixture_factor.py'
    factor.write_text("""from pathlib import Path
import pandas as pd
from utils.factor_base import FactorBase

class FixtureFactor(FactorBase):
    def get_factor_values(self, force_recalculate=False):
        values = pd.read_parquet(Path(__file__).with_name("daily_factors.parquet"))
        return values.loc[:"2026-01-23"]
""")
    legacy_db = tmp_path / 'legacy-results.duckdb'
    cfg = tmp_path / 'legacy.json'
    cfg.write_text(json.dumps(dict(
        duckdb_path=inputs['slot_db'], results_duckdb_path=str(legacy_db), results_export_parquet=False,
        price_table='price_data_daily', start_date='2026-01-05', end_date='2026-01-23', code_list=assets,
        analysis_mode='cross_sectional', n_quantiles=3, ic_periods=list(DAILY_HORIZONS),
        factor_file=str(factor), use_dynamic_pools=False, max_workers=1, cache_dir=str(tmp_path / 'cache'),
    )))
    return cfg, legacy_db


def _legacy_daily_evaluation(result):
    artifact, = [item for item in result['artifacts'] if item['artifact_type'] == 'evaluation_summary']
    return Path(artifact['path'])


def test_legacy_daily_harness_ingests_without_slot_and_preserves_execution_result(tmp_path, legacy_daily_config):
    cfg, legacy_db = legacy_daily_config
    args = ('--config', str(cfg))
    plain = _cli_result(tmp_path, 'plain', 'single-factor-analysis', *args)
    query = 'SELECT code, ic, ir, "return" FROM factor_analysis_results ORDER BY code'
    with duckdb.connect(str(legacy_db), read_only=True) as connection:
        legacy_metrics = connection.execute(query).fetchall()
    assert legacy_metrics

    db = tmp_path / 'eval.duckdb'
    ingested = _cli_result(tmp_path, 'ingested', 'single-factor-analysis', *args, '--results-db', str(db))
    def execution(result):
        artifact, = [item for item in result['artifacts'] if item['artifact_type'] == 'execution_result']
        return json.loads(Path(artifact['path']).read_text())

    before, after = execution(plain), execution(ingested)
    for key in ('schema_version', 'command', 'config_sha256', 'execution', 'metrics'):
        assert after[key] == before[key]
    assert after['execution']['status'] == 'succeeded'
    parsed = {key: value for key, value in ingested['metrics'].items() if key != 'results_db'}
    assert parsed == plain['metrics']
    with duckdb.connect(str(legacy_db), read_only=True) as connection:
        assert connection.execute(query).fetchall() == legacy_metrics
    assert ingested['metrics']['results_db']['status'] == 'inserted'
    labels = pd.read_csv(_legacy_daily_evaluation(ingested).with_name('per_bar_labels.csv'))
    first = labels.loc[labels.signal_timestamp.eq(labels.signal_timestamp.min()) & labels.horizon_bars.eq(1)]
    np.testing.assert_allclose(first.return_pct.iloc[:2], [0., .2], rtol=0, atol=1e-12)
    retry = _cli(tmp_path, 'retry', 'single-factor-analysis', *args, '--results-db', str(db))
    assert retry['results_db']['status'] == 'noop'
    read = _cli(tmp_path, 'read', 'read-bar-evaluations', '--results-db', str(db), '--slot-id', 'jq_ashare_local')
    assert read['run_count'] == 1
    assert read['runs'][0]['frequency'] == '1d'
    for field in ('ic_by_horizon', 'quantile_stats', 'cross_sections', 'split_metrics'):
        assert read[field]


def test_legacy_daily_harness_ignores_empty_weekend_padding(tmp_path, legacy_daily_config):
    cfg, _ = legacy_daily_config
    db = tmp_path / 'eval.duckdb'
    args = ('--config', str(cfg))
    stored = ('--results-db', str(db))
    baseline = _cli_result(tmp_path, 'baseline', 'single-factor-analysis', *args, *stored)
    query = ('read-bar-evaluations', *stored, '--slot-id', 'jq_ashare_local')
    before = _cli(tmp_path, 'before', *query)

    panel = tmp_path / 'daily_factors.parquet'
    factors = pd.read_parquet(panel)
    factors.loc[pd.Timestamp('2026-01-10')] = np.nan
    factors.sort_index().to_parquet(panel)

    plain = _cli_result(tmp_path, 'padded-plain', 'single-factor-analysis', *args)
    padded = _cli_result(tmp_path, 'padded-ingested', 'single-factor-analysis', *args, *stored)
    receipt = padded['metrics'].pop('results_db')
    assert receipt['status'] == 'noop'
    assert receipt['identity_hash'] == baseline['metrics']['results_db']['identity_hash']
    assert padded['metrics'] == plain['metrics']
    assert _cli(tmp_path, 'after', *query) == before
    before_summary = _legacy_daily_evaluation(baseline)
    after_summary = _legacy_daily_evaluation(padded)
    assert json.loads(after_summary.read_text()) == json.loads(before_summary.read_text())
    for filename in ('per_bar_labels.csv', 'ic_by_horizon.csv', 'quantile_stats.csv'):
        assert after_summary.with_name(filename).read_bytes() == before_summary.with_name(filename).read_bytes()


def test_legacy_daily_harness_rejects_populated_weekend(tmp_path, legacy_daily_config):
    cfg, _ = legacy_daily_config
    panel = tmp_path / 'daily_factors.parquet'
    factors = pd.read_parquet(panel)
    factors.loc[pd.Timestamp('2026-01-10'), factors.columns[0]] = 1.0
    factors.sort_index().to_parquet(panel)
    db = tmp_path / 'eval.duckdb'
    result = _cli_result(tmp_path, 'populated-weekend', 'single-factor-analysis', '--config', str(cfg),
                         '--results-db', str(db), exit_code=3)
    assert 'Factor output contains values outside the Slot output grid' in result['stderr_tail']
    assert not db.exists()


def test_legacy_daily_harness_store_precondition_revokes_completion(tmp_path, legacy_daily_config):
    cfg, _ = legacy_daily_config
    db = tmp_path / 'incompatible.duckdb'
    with duckdb.connect(str(db)) as connection:
        connection.execute('CREATE TABLE bar_evaluation_runs (identity_hash VARCHAR PRIMARY KEY)')

    result = _cli_result(tmp_path, 'store-failed', 'single-factor-analysis', '--config', str(cfg),
                         '--results-db', str(db), exit_code=2)
    assert not result['ok']
    assert result['command']['mode'] == 'results_db'
    assert result['error']['code'] == 'bar_evaluation_store_precondition_failed'
    assert result['error']['details']['exception_type'] == 'BarEvaluationPreconditionError'
    assert 'schema' in result['error']['message'].lower()
    artifact, = [item for item in result['artifacts'] if item['artifact_type'] == 'execution_result']
    evaluation, = (Path(artifact['path']).parent / 'bar-evaluations').iterdir()
    assert (evaluation / 'per_bar_labels.csv').is_file()
    assert not (evaluation / 'summary.json').exists()
    assert not any(item['artifact_type'] == 'evaluation_summary' for item in result['artifacts'])
    with duckdb.connect(str(db), read_only=True) as connection:
        assert connection.execute('SHOW TABLES').fetchall() == [('bar_evaluation_runs',)]
        assert connection.execute('SELECT * FROM bar_evaluation_runs').fetchall() == []
