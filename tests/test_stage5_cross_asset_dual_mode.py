"""Stage 5: importer -> temporary DuckDB -> FactorBase -> frozen evaluators."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner
from backtest_and_analysis.time_series.contracts import validate_factor_frame, ContractValidationError
from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame, session_daily_frame
from data_pipeline.loaders.market_bar_importer import import_market_bars
from utils.factor_base import FactorBase
from utils.session_calendar import SessionCalendar
from test_time_series_cli_end_to_end import run_cli, read_outputs


class RollingFactor(FactorBase):
    def __init__(self, *args, window=3, within_session=True, warmup_sessions=0, warmup_bars=0, **kwargs):
        self.window = window
        self.within_session = within_session
        self.warmup_sessions = warmup_sessions
        self.warmup_bars = warmup_bars
        super().__init__(*args, **kwargs)

    def calculate_factor(self, data):
        values = self.session_rolling(data, 'close', self.window, within_session=self.within_session,
            warmup_sessions=self.warmup_sessions, warmup_bars=self.warmup_bars)
        return data[['timestamp_utc', 'instrument_id']].assign(factor_value=values)

    def _get_cache_inputs(self):
        return dict(window=self.window, within_session=self.within_session,
                    warmup_sessions=self.warmup_sessions, warmup_bars=self.warmup_bars)


@pytest.fixture
def conn(tmp_path):
    with duckdb.connect(str(tmp_path / 'synthetic.duckdb')) as connection:
        yield connection


def populate(conn, *, future=False, frequency='1min', instruments=1, days=2, missing=()):
    start, end = ('2026-12-01', '2026-12-02') if future else ('2026-09-01', '2026-09-15')
    calendar = SessionCalendar('NQ' if future else 'XSHG', start, end)
    sessions = calendar.sessions()[:days]
    frames, metadata, ids = [], [], []
    for i in range(instruments):
        instrument = 'NQZ26' if future else f'60000{i}.XSHG'
        ids.append(instrument)
        metadata.append(dict(instrument_id=instrument, asset_class='futures' if future else 'equity',
            exchange='CME' if future else 'XSHG', timezone='America/Chicago' if future else 'Asia/Shanghai',
            contract_expiry='2026-12' if future else None, price_adjustment='none',
            point_value=20. if future else 1., rth_reference_calendar='NYSE' if future else 'XSHG'))
        for day_index, day in enumerate(sessions):
            times = calendar.expected_bar_opens(day.strftime('%Y-%m-%d'), frequency)
            j = np.arange(len(times))
            opening = 100 + i * 3 + day_index * (i + 1) + .01 * j + .5 * np.sin(j / (7 + i))
            close = opening + .1 + .02 * (i + 1) * np.cos(j / 11)
            frame = pd.DataFrame(dict(timestamp_utc=times, instrument_id=instrument, frequency=frequency,
                open=opening, high=np.maximum(opening, close) + 1, low=np.minimum(opening, close) - 1,
                close=close, volume=1., source='SYNTHETIC', data_version='v1'))
            if day_index == 0:
                frame = frame.drop(index=list(missing))
            frames.append(frame)
    import_market_bars(conn, pd.concat(frames, ignore_index=True), pd.DataFrame(metadata))
    return calendar, ids, sessions[0].strftime('%Y-%m-%d'), sessions[-1].strftime('%Y-%m-%d')


def factor(conn, spec, **kwargs):
    calendar, ids, start, end = spec
    options = dict(stock_pool=ids, data_frequency='1min', analysis_mode='time_series',
                   session_calendar=calendar, data_version='v1', label_config={'horizons': [1, 5]})
    return RollingFactor(conn, 'rolling', start, end, **{**options, **kwargs})


def check_ts_oracle(result):
    for row in result.ic_by_horizon.query("scope == 'per_instrument'").itertuples():
        group = result.per_bar_labels.query('instrument_id == @row.instrument_id and horizon_bars == @row.horizon_bars')
        valid = group.loc[group.status.eq('valid')]
        expected = valid.factor_value.corr(valid.return_pct, method='spearman') if len(valid) >= 3 else np.nan
        sampled = group.sort_values('signal_timestamp').iloc[::row.horizon_bars]
        sampled = sampled.loc[sampled.status.eq('valid')]
        independent = sampled.factor_value.corr(sampled.return_pct, method='spearman') if len(sampled) >= 3 else np.nan
        assert row.spearman_ic == pytest.approx(expected, nan_ok=True)
        assert row.non_overlapping_ic == pytest.approx(independent, nan_ok=True)
        assert row.non_overlapping_count == len(sampled)
    assert result.quantile_stats.descriptive_only.all()
    assert result.exclusion_summary['total_count'] == len(result.per_bar_labels)


@pytest.mark.parametrize('frequency', ['1min', '5min'])
def test_scenario1_session_daily_cross_section_enters_unchanged_alphalens(conn, frequency):
    spec = populate(conn, frequency=frequency, instruments=8, days=8)
    item = factor(conn, spec, data_frequency=frequency, analysis_mode='cross_sectional', output_frequency='1d')
    daily = item.get_factor_values()
    pricing = item.get_session_daily_pricing()
    assert daily.index.name == 'date' and daily.columns.name == 'stock_code'
    assert pricing.index.names == ['asset', 'date']
    assert pricing.attrs['trading_calendar'].calendar_id == 'XSHG'
    assert len(daily) == 8
    result = AlphalensRunner(periods=(1, 2), n_quantiles=2, execution_lag=0).run(daily, pricing)
    prices = pricing.unstack('asset')
    for h in (1, 2):
        returns = prices.shift(-h) / prices - 1
        expected = daily.corrwith(returns, axis=1, method='spearman').dropna()
        pd.testing.assert_series_equal(result['ic'][f'{h}D'].dropna(), expected, check_names=False)
    assert not result['quantile_returns'].empty


@pytest.mark.parametrize('frequency', ['1min', '5min'])
def test_scenario2_minute_cross_section_and_cli(conn, tmp_path, frequency):
    spec = populate(conn, frequency=frequency, instruments=5)
    item = factor(conn, spec, data_frequency=frequency, analysis_mode='cross_sectional')
    frame = item.get_bar_factor_frame()
    validate_factor_frame(frame)
    result = item.evaluate_bar_factor(n_quantiles=2)
    sections = result.cross_sections
    for row in sections.itertuples():
        valid = result.per_bar_labels.query('signal_timestamp == @row.signal_timestamp and horizon_bars == @row.horizon_bars and status == "valid"')
        ic = valid.factor_value.corr(valid.return_pct, method='spearman') if len(valid) >= 3 else np.nan
        assert row.spearman_ic == pytest.approx(ic, nan_ok=True)
    for row in result.ic_by_horizon.itertuples():
        values = sections.loc[sections.horizon_bars.eq(row.horizon_bars), 'spearman_ic'].dropna()
        assert row.mean_ic == pytest.approx(values.mean())
        assert row.icir == pytest.approx(values.mean() / values.std())
    for q in result.quantile_stats.itertuples():
        means = []
        for _, group in result.per_bar_labels.query('horizon_bars == @q.horizon_bars and status == "valid"').groupby('signal_timestamp'):
            groups = pd.qcut(group.factor_value, 2, labels=False, duplicates='drop') + 1
            means.append(group.loc[groups.eq(q.quantile), 'return_pct'].mean())
        assert q.mean_return_pct == pytest.approx(np.mean(means))
    assert result.exclusion_summary['exclusion_counts']['factor_missing'] == 5 * 2 * 2 * 2
    path, factors = tmp_path / 'bars.csv', tmp_path / 'factors.csv'
    item.market_bars.to_csv(path, index=False)
    frame.to_csv(factors, index=False)
    payload = run_cli(tmp_path, dict(bars_csv=str(path), factors_csv=str(factors),
        analysis_mode='cross_sectional', calendar='XSHG', start_date=spec[2], end_date=spec[3],
        horizons=[1, 5], n_quantiles=2))
    assert payload['ok'], payload
    _, cli_ic, _, summary = read_outputs(payload)
    np.testing.assert_allclose(cli_ic.mean_ic, result.ic_by_horizon.mean_ic)
    assert summary['exclusion_summary'] == result.exclusion_summary


@pytest.mark.parametrize('frequency', ['1min', '5min'])
def test_scenario3_stock_time_series_lunch_missing_bars_and_session_reset(conn, frequency):
    spec = populate(conn, frequency=frequency, missing=(6,))
    item = factor(conn, spec, data_frequency=frequency)
    result = item.evaluate_bar_factor(n_quantiles=3)
    check_ts_oracle(result)
    reasons = result.exclusion_summary['exclusion_counts']
    assert reasons['factor_missing'] > 0 and reasons['missing_bar'] > 0
    grid = item.market_bar_grid
    long = item.get_bar_factor_frame()
    for _, day in grid.groupby('session_id'):
        values = long.set_index('timestamp_utc').factor_value.reindex(day.timestamp_utc)
        assert values.iloc[:2].isna().all()
    times = spec[0].expected_bar_opens(spec[2], frequency)
    missing_time = times[6]
    around = long.set_index('timestamp_utc').factor_value.reindex(times)
    assert around.iloc[6:9].isna().all()
    assert around.iloc[9] == pytest.approx(grid.set_index('timestamp_utc').close.loc[times[7:10]].mean())
    assert grid.loc[grid.timestamp_utc.eq(missing_time), 'close'].isna().all()
    # Lunch is a scheduled closure, so the first PM rolling window uses the last AM bars.
    pm = times.get_loc(pd.Timestamp(spec[2] + ' 05:00Z'))
    assert around.iloc[pm] == pytest.approx(grid.set_index('timestamp_utc').close.loc[times[pm-2:pm+1]].mean())


def test_scenario4_future_overnight_maintenance_and_daily_session_ownership(conn):
    spec = populate(conn, future=True)
    item = factor(conn, spec)
    result = item.evaluate_bar_factor()
    check_ts_oracle(result)
    grid = item.market_bar_grid
    first = grid.loc[grid.session_id.eq(spec[2])]
    assert first.timestamp_utc.dt.date.nunique() == 2
    assert not grid.timestamp_utc.dt.hour.eq(22).any()  # CME 16-17 CST maintenance.
    values = item.get_bar_factor_frame()
    daily = session_daily_frame(values, grid)
    assert list(daily.index.strftime('%Y-%m-%d')) == [spec[2], spec[3]]
    assert daily.iloc[0, 0] == values.set_index('timestamp_utc').factor_value.loc[first.timestamp_utc.iloc[-1]]
    assert result.exclusion_summary['exclusion_counts']['session_boundary'] > 0


def test_scenario5_eth_rth_multi_horizon_counts_windows_and_nonoverlap(conn):
    spec = populate(conn, future=True)
    results = {}
    for view, size in [('ETH', 1380), ('RTH', 390)]:
        item = factor(conn, spec, session_view=view, label_config={'horizons': [1, 5, 15, 60]})
        result = item.evaluate_bar_factor()
        check_ts_oracle(result)
        results[view] = result
        for row in result.ic_by_horizon.query("scope == 'per_instrument'").itertuples():
            assert row.total_count == 2 * size
            assert row.valid_count == 2 * (size - 2 - row.horizon_bars)
            assert row.exclusion_counts['factor_missing'] == 4
            assert row.exclusion_counts['session_boundary'] == (row.horizon_bars if view == 'ETH' else 2 * row.horizon_bars)
        values = item.get_bar_factor_frame().set_index('timestamp_utc').factor_value
        opening = spec[0].expected_bar_opens(spec[2], '1min', view)[0]
        assert values.loc[opening:opening + pd.Timedelta('1min')].isna().all()
    eth = results['ETH'].ic_by_horizon.query("scope == 'per_instrument'")
    rth = results['RTH'].ic_by_horizon.query("scope == 'per_instrument'")
    assert not np.allclose(eth.non_overlapping_ic, rth.non_overlapping_ic)


def test_cache_identity_every_context_dimension_and_roundtrip(conn, tmp_path):
    spec = populate(conn)
    item = factor(conn, spec, cache_dir=str(tmp_path / 'cache'))
    raw = item._fetch_market_bars()
    key = item._get_cache_key(raw)
    assert key is not None
    changes = dict(analysis_mode='cross_sectional', data_frequency='5min', output_frequency='1d',
        session_view='RTH', data_version='v2', source_revision='new', label_config={'horizons': [15]},
        calendar_hash='different', rth_reference_calendar='NYSE', window=5, all_stocks=['OTHER'],
        instrument_metadata=item.instrument_metadata.assign(point_value=2.))
    for name, replacement in changes.items():
        old = getattr(item, name)
        setattr(item, name, replacement)
        assert item._get_cache_key(raw) not in (None, key), name
        setattr(item, name, old)
    original = item.get_bar_factor_frame()
    # The calculation is skipped on a bound cache hit, while the read snapshot refreshes.
    item.calculate_factor = lambda _: pytest.fail('cache missed')
    pd.testing.assert_frame_equal(item.get_bar_factor_frame(), original)
    with pytest.raises(ValueError, match='data_version'):
        factor(conn, spec, data_version='v2').get_bar_factor_frame()


def test_warmup_counts_expected_sessions_and_slots_including_missing_whole_session(conn):
    spec = populate(conn, days=3, missing=(0, 1))
    # A whole missing trading session must not become invisible to rolling windows.
    second = spec[0].sessions()[1].strftime('%Y-%m-%d')
    conn.execute('DELETE FROM market_bars WHERE timestamp_utc >= ? AND timestamp_utc < ?',
        [pd.Timestamp(second, tz='UTC'), pd.Timestamp(second, tz='UTC') + pd.Timedelta('1d')])
    item = factor(conn, spec, within_session=False, warmup_sessions=1, warmup_bars=4)
    values = item.get_bar_factor_frame()
    grid = item.market_bar_grid
    joined = grid.merge(values, on=['timestamp_utc', 'instrument_id'])
    assert joined.loc[joined.session_id.eq(spec[2]), 'factor_value'].isna().all()
    assert joined.loc[joined.session_id.eq(second), 'factor_value'].isna().all()
    third = joined.loc[joined.session_id.eq(spec[3]), 'factor_value']
    assert third.iloc[:2].isna().all() and third.iloc[2:].notna().all()


def test_wide_long_conversion_preserves_missing_and_rejects_invalid_keys():
    wide = pd.DataFrame({'A': [np.nan, 2.], 'B': [3., 4.]},
        index=pd.date_range('2026-09-01 01:30Z', periods=2, freq='1min'))
    long = bar_factor_frame(wide)
    assert len(long) == 4 and long.factor_value.isna().sum() == 1
    pd.testing.assert_frame_equal(long.pivot(index='timestamp_utc', columns='instrument_id', values='factor_value'),
        wide.rename_axis(index='timestamp_utc', columns='instrument_id'), check_freq=False)
    with pytest.raises(ContractValidationError):
        bar_factor_frame(wide.tz_localize(None))
    with pytest.raises(ContractValidationError):
        bar_factor_frame(pd.concat([long, long.iloc[:1]]))


@pytest.mark.parametrize('options', [dict(data_frequency='1min'), dict(data_frequency='1d'),
    dict(data_frequency='5min', analysis_mode='bad'), dict(data_frequency='1min', analysis_mode='time_series', session_view='bad'),
    dict(data_frequency='1min', analysis_mode='time_series', output_frequency='daily'),
    dict(data_frequency='1min', analysis_mode='time_series', output_frequency='5min')])
def test_new_interface_rejects_implicit_or_inconsistent_context(options):
    with pytest.raises(ValueError):
        RollingFactor(None, 'invalid', '2026-09-01', '2026-09-02', stock_pool=['A'], **options)


def test_warmup_bars_is_independent_of_observed_row_count(conn):
    spec = populate(conn, missing=(0, 1))
    item = factor(conn, spec, window=1, warmup_bars=5)
    values = item.get_bar_factor_frame()
    assert values.factor_value.iloc[:5].isna().all()
    assert values.factor_value.iloc[5] == pytest.approx(item.market_bar_grid.close.iloc[5])


def test_missing_session_close_is_not_replaced_by_previous_price(conn):
    spec = populate(conn)
    last = spec[0].expected_bar_opens(spec[2], '1min')[-1]
    conn.execute('DELETE FROM market_bars WHERE timestamp_utc = ?', [last])
    item = factor(conn, spec, analysis_mode='cross_sectional', output_frequency='daily')
    assert np.isnan(item.get_factor_values().iloc[0, 0])
    assert np.isnan(item.get_session_daily_pricing().loc[(spec[1][0], pd.Timestamp(spec[2]))])


def test_auto_calendar_and_read_only_path_use_explicit_source(conn, tmp_path):
    spec = populate(conn, frequency='5min')
    item = factor(conn, spec, data_frequency='5m', session_calendar=None)
    frame = item.get_bar_factor_frame()
    assert len(frame) == 96
    item = RollingFactor(None, 'disk', spec[2], spec[3], stock_pool=spec[1],
        data_frequency='5min', analysis_mode='time_series', duckdb_path=str(tmp_path / 'synthetic.duckdb'))
    # DuckDB requires all simultaneous file connections to have the same read mode.
    conn.close()
    pd.testing.assert_frame_equal(item.get_bar_factor_frame(), frame)


def test_cloud_builder_omits_local_minute_dependencies():
    import ast
    from scripts.build_jq_strategy import lift_minimal_factor_base
    source = lift_minimal_factor_base()
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert 'utils.market_bar_reader' not in imports
    assert 'backtest_and_analysis.time_series.factor_bridge' not in imports
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FactorBase')
    bridge = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == 'get_bar_factor_frame')
    assert isinstance(bridge.body[0], ast.Raise)


@pytest.mark.parametrize('backed_by_path', [False, True])
def test_1d_alias_preserves_legacy_daily_reads_and_pit(conn, tmp_path, backed_by_path):
    from test_duckdb_only_factor_base import CloseSignal
    conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2026-09-01' AS date, "
                 "'A' AS stock_code, 10. AS close, 1. AS factor UNION ALL "
                 "SELECT DATE '2026-09-02', 'A', 12., 1.")
    kwargs = dict(stock_pool=['A'])
    connection = conn
    if backed_by_path:
        connection = None
        kwargs['duckdb_path'] = str(tmp_path / 'synthetic.duckdb')
        conn.close()
    items = [CloseSignal(connection, 'alias', '2026-09-01', '2026-09-02',
                         data_frequency=frequency, analysis_mode='cross_sectional', **kwargs)
             for frequency in ('daily', '1d')]
    try:
        pd.testing.assert_frame_equal(items[0].get_factor_values(), items[1].get_factor_values())
    finally:
        for item in items:
            if item._duckdb_adapter is not None:
                item._duckdb_adapter.close()


def test_mixed_asset_calendars_isolate_read_bounds(conn):
    stock_spec = populate(conn, days=2)
    # Include an equity row before the requested first session; CME's earlier
    # overnight UTC opening must not pull it into the stock calendar read.
    conn.execute("INSERT INTO market_bars SELECT instrument_id, timestamp_utc - INTERVAL '1 day', "
                 "frequency, open, high, low, close, volume, source, data_version, money, factor "
                 "FROM market_bars WHERE timestamp_utc = (SELECT min(timestamp_utc) FROM market_bars)")
    calendar = SessionCalendar('NQ', stock_spec[2], stock_spec[3])
    times = calendar.expected_bar_opens(stock_spec[2], '1min')
    bars = pd.DataFrame(dict(timestamp_utc=times, instrument_id='NQZ26', frequency='1min',
                            open=100., high=102., low=99., close=101., volume=1.,
                            source='SYNTHETIC', data_version='v1'))
    meta = pd.DataFrame([dict(instrument_id='NQZ26', asset_class='futures', exchange='CME',
        timezone='America/Chicago', contract_expiry='2026-12', price_adjustment='none',
        point_value=20., rth_reference_calendar='NYSE')])
    import_market_bars(conn, bars, meta)
    ids = [*stock_spec[1], 'NQZ26']
    calendars = {stock_spec[1][0]: stock_spec[0], 'NQZ26': calendar}
    item = factor(conn, stock_spec, stock_pool=ids, session_calendar=calendars)
    frame = item.get_bar_factor_frame()
    assert set(frame.instrument_id) == set(ids)
    assert item.market_bars.loc[item.market_bars.instrument_id.eq(stock_spec[1][0]), 'session_id'].min() == stock_spec[2]
    with pytest.raises(ContractValidationError, match='shared calendar'):
        item.get_session_daily_pricing()
