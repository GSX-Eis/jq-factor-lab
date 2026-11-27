"""Synthetic parity and PIT boundaries for the local JoinQuant MarketSlot."""

from datetime import date

import duckdb

from market_db_samples import declare_market_db
import numpy as np
import pandas as pd
import pytest

from slots.jq_ashare_local import UniverseDefinition, build_jq_ashare_local
from slots.jq_ashare_local.panel import ASharePanelCatalog, ASharePanelReader
from utils.bars.catalog import BarCatalog
from utils.data_access_config import load_data_access_config
from utils.duckdb_adapter import DuckDBAdapter
from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter
from utils.pit import apply_pit_adjustment, normalize_price_money
from utils.session_calendar import VERIFIED_XSHG
from utils.time_axis import Frequency


FIELDS = ['open', 'high', 'low', 'close', 'volume', 'money', 'factor', 'paused']
DATES = ('2026-01-06', '2026-01-07', '2026-01-08')


@pytest.fixture
def connection(tmp_path):
    with duckdb.connect(str(tmp_path / 'a-share.duckdb')) as conn:
        declare_market_db(conn, 'cn_equity')
        yield conn


def _prices(connection, table, timestamps):
    connection.execute(f'''CREATE TABLE {table} (stock_code VARCHAR, date TIMESTAMP,
        open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, money DOUBLE,
        factor DOUBLE, paused BOOLEAN, high_limit DOUBLE, low_limit DOUBLE)''')
    connection.executemany(f'INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
        ('A', timestamps[0], 100, 104, 98, 102, 12, 23, 2, False, 110, 90),
        ('A', timestamps[1], 200, 203, 191, 201, 0, 0, 4, True, 220, 180),
        ('A', timestamps[2], 300, 304, 292, 303, 16, 42, 8, False, 330, 270),
        ('B', timestamps[0], 50, 53, 49, 51, 9, 17, 3, False, 55, 45),
        ('B', timestamps[1], None, None, None, None, None, 0, None, True, None, None),
    ])


@pytest.mark.parametrize(('frequency', 'table', 'suffix', 'first', 'second', 'third', 'completion'), [
    ('1d', 'price_data_daily', '', '07:00', '07:00', '07:00', '07:00'),
    ('5m', 'price_data_5m', ' 09:35:00', '01:30', '01:30', '01:30', '01:35'),
])
def test_bar_views_match_adapter_and_hide_unavailable_factors(
    connection, frequency, table, suffix, first, second, third, completion,
):
    _prices(connection, table, [day + suffix for day in DATES])
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1])
    assert slot.calendar.calendar_id == VERIFIED_XSHG
    first, second, third = [pd.Timestamp(f'{day} {clock}Z')
                            for day, clock in zip(DATES, (first, second, third))]
    anchor = pd.Timestamp(f'{DATES[1]} {completion}Z')
    feed = slot.bar_feed
    parsed = Frequency.parse(frequency)
    tables = connection.execute('SHOW TABLES').fetchall()
    legacy = ASharePanelCatalog(connection).resolve(frequency, 'A')
    assert legacy.table_name == table
    with pytest.raises(LookupError, match='Missing frequency'):
        BarCatalog(connection).resolve(frequency, 'A')
    adapter = DuckDBAdapter(duckdb_path=connection.execute('PRAGMA database_list').fetchone()[2],
                            data_bindings={})
    adapter._conn = connection
    original = adapter._fetch_panel(
        load_data_access_config().dataset(table), ['A', 'B'], DATES[0], DATES[1], FIELDS)
    normalized = normalize_price_money(original, source='local', frequency='daily' if frequency == '1d' else '5m')
    expected = apply_pit_adjustment(normalized)
    actual = feed.read_signal_bars(['A', 'B'], parsed, start_utc=first, end_utc=third, as_of_utc=anchor)
    np.testing.assert_allclose(actual[FIELDS].to_numpy(dtype=float), expected[FIELDS].to_numpy(dtype=float),
                               rtol=0, atol=1e-12, equal_nan=True)
    assert actual.session_id.tolist() == [*DATES[:2], *DATES[:2]]
    assert actual.timestamp_utc.tolist() == [first, second, first, second]
    before = feed.read_signal_bars(['A'], parsed, start_utc=first, end_utc=third,
                                   as_of_utc=anchor - pd.Timedelta(microseconds=1))
    assert before.close.tolist() == [102.0]
    narrower = feed.read_signal_bars(['A'], parsed, start_utc=first, end_utc=first, as_of_utc=anchor)
    assert narrower.close.tolist() == [51.0]
    execution = feed.read_execution_bars(['A'], parsed, start_utc=first, end_utc=third)
    assert execution.close.tolist() == [102, 201, 303]
    assert execution.factor.tolist() == [2, 4, 8]
    assert execution.money.tolist() == [23, 0, 42]
    assert connection.execute('SHOW TABLES').fetchall() == tables


def test_calendar_preserves_close_lunch_and_supported_grid_frequencies(connection):
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1])
    day = DATES[0]
    assert slot.calendar.bar_grid('1d', day, day).timestamp_utc.tolist() == [pd.Timestamp(f'{day} 07:00Z')]
    assert len(slot.calendar.bar_grid('30min', day, day)) == 8
    grid = slot.calendar.bar_grid('5m', day, day)
    assert len(grid) == 48
    assert grid.timestamp_utc.iloc[23:25].tolist() == [
        pd.Timestamp(f'{day} 03:25Z'), pd.Timestamp(f'{day} 05:00Z')]


def test_universe_matches_existing_selector_and_blocks_future_membership(connection):
    connection.execute('''CREATE TABLE factor_data AS SELECT
        stock_code, CAST(date AS DATE) AS date, cap AS val_market_cap,
        false AS paused, false AS is_st, false AS is_new_stock
        FROM (VALUES ('A','2026-01-06',1), ('B','2026-01-06',2), ('C','2026-01-06',3),
                     ('A','2026-01-07',2), ('B','2026-01-07',3), ('C','2026-01-07',1),
                     ('A','2026-01-08',1), ('B','2026-01-08',2), ('C','2026-01-08',3))
        AS v(stock_code, date, cap)''')
    connection.execute('''CREATE TABLE index_constituents AS SELECT index_code, stock_code, CAST(date AS DATE) AS date
        FROM (VALUES ('000300.XSHG'), ('000905.XSHG'), ('8841431.WI')) AS i(index_code)
        CROSS JOIN (VALUES ('A','2026-01-06'), ('A','2026-01-07'), ('B','2026-01-07'), ('C','2026-01-07'),
                           ('C','2026-01-08')) AS v(stock_code, date)''')
    definition = UniverseDefinition((StockFilter('val_market_cap', '>=', 2),), sort_by='val_market_cap')
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1], universes={'custom': definition})
    universe = slot.universe
    manager = DynamicStockPoolManager(connection, price_table='price_data_daily', data_bindings={})
    day = pd.Timestamp(DATES[1]).to_pydatetime()
    anchor = pd.Timestamp('2026-01-07 07:00Z')
    for pool, code in [('csi300', '000300.XSHG'), ('csi500', '000905.XSHG'), ('wind_micro', '8841431.WI')]:
        filters = [StockFilter('paused', '==', 0), StockFilter('is_st', '==', 0), StockFilter('index_code', '==', code)]
        if pool == 'wind_micro':
            filters += [StockFilter('is_new_stock', '==', 0),
                        StockFilter('val_market_cap', 'percentile_between', (20, 80))]
        assert universe.resolve_universe(pool, as_of_utc=anchor) == manager._select_stock_pool(day, filters)
    assert universe.resolve_universe('custom', as_of_utc=anchor) == manager._select_stock_pool(
        day, list(definition.filters), sort_by=definition.sort_by)
    assert universe.resolve_universe('csi300', as_of_utc=anchor - pd.Timedelta(seconds=1)) == ['A']
    assert universe.resolve_universe('000300.XSHG', as_of_utc=anchor) == ['A', 'B', 'C']
    assert universe.resolve_active_contract('A', as_of_utc=anchor) == 'A'
    spec = universe.instrument_spec('A')
    assert (spec.asset_class, spec.currency, spec.lot_size, spec.contract_multiplier) == ('equity', 'CNY', 100, 1.)
    connection.execute("CREATE TABLE security_master AS SELECT 'A' stock_code, DATE '2020-01-02' start_date, 200 lot_size")
    spec = universe.instrument_spec('A')
    assert spec.listing_date == date(2020, 1, 2) and spec.lot_size == 200


def test_aux_reads_snapshot_publish_date_and_timestamp_without_lookahead(connection):
    connection.execute("CREATE TABLE factor_data AS SELECT 'A' stock_code, DATE '2026-01-06' date, 1.25 alpha")
    connection.execute("CREATE TABLE valuation_daily AS SELECT 'A' stock_code, DATE '2026-01-06' date, 2.5 pe")
    connection.execute('''CREATE TABLE earnings_forecast_events AS SELECT
        'A' AS stock_code, CAST(pub_date AS DATE) AS pub_date, DATE '2025-12-31' AS report_date, forecast
        FROM (VALUES ('2026-01-05', 10.), ('2026-01-06', 20.), ('2026-01-06', 21.),
                     ('2026-01-08', 999.)) v(pub_date, forecast)''')
    connection.execute('''CREATE TABLE raw_income_statement_events AS SELECT
        'A' code, DATE '2026-01-06' pub_date, 7.5 AS revenue''')
    connection.execute('''CREATE TABLE regime_cninfo_material_litigation_initial_events_v1 AS SELECT
        'A' issuer_stock_code, TIMESTAMP '2026-01-06 03:00' announcement_timestamp, true AS flag''')
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1])
    grid = pd.DataFrame({'instrument_id': ['A'] * 4, 'session_id': [DATES[0]] * 3 + [DATES[1]],
                         'timestamp_utc': pd.to_datetime([
                             '2026-01-06 02:59:59Z', '2026-01-06 03:00:00Z',
                             '2026-01-06 07:00:00Z', '2026-01-07 07:00:00Z'])})
    announcement = 'regime_cninfo_material_litigation_initial_events_v1.flag'
    forecast = 'earnings_forecast_events.forecast'
    financial = 'raw_income_statement_events.revenue'
    result = slot.read_aux_features(grid, features=['alpha', 'valuation_daily.pe', forecast, financial, announcement])
    assert (result.available_at_utc <= result.timestamp_utc).all()
    matches = result.loc[result[forecast].notna()]
    original = connection.execute(
        "SELECT forecast FROM earnings_forecast_events WHERE pub_date < '2026-01-07' ORDER BY pub_date, forecast"
    ).fetchnumpy()['forecast']
    expected = [*([original[0]] * 3), *original[1:]]
    np.testing.assert_allclose(matches[forecast].sort_values(), expected, rtol=0, atol=1e-12, equal_nan=True)
    assert matches.loc[matches.session_id.eq(DATES[1]), 'available_at_utc'].eq(pd.Timestamp('2026-01-06 16:00Z')).all()
    assert result.loc[result[financial].notna(), 'timestamp_utc'].tolist() == [grid.timestamp_utc.iloc[3]]
    assert result.loc[result.alpha.notna(), 'timestamp_utc'].tolist() == [grid.timestamp_utc.iloc[2]]
    assert result.loc[result[announcement].notna(), 'timestamp_utc'].tolist() == grid.timestamp_utc.iloc[1:].tolist()
    assert result['valuation_daily.pe'].dropna().tolist() == [2.5]
    assert slot.read_aux_features(grid, features=['dividend_events.bonus_ratio_rmb']).empty
    categorical = grid.assign(instrument_id=pd.Categorical(grid.instrument_id))
    pd.testing.assert_frame_equal(slot.read_aux_features(categorical, features=[forecast]),
                                  slot.read_aux_features(grid, features=[forecast]))


def test_microstructure_pause_priority_limits_st_and_t_plus_one(connection):
    _prices(connection, 'price_data_daily', DATES)
    connection.execute("UPDATE price_data_daily SET close=high_limit WHERE date='2026-01-06'")
    connection.execute("UPDATE price_data_daily SET close=low_limit WHERE date='2026-01-08'")
    connection.execute("CREATE TABLE all_a_paused_daily AS SELECT 'A' stock_code, DATE '2026-01-07' date, false paused")
    connection.execute("CREATE TABLE regime_all_a_paused_daily_v1 AS SELECT 'A' stock_code, DATE '2026-01-07' date, true paused")
    connection.execute("CREATE TABLE historical_st_daily AS SELECT 'B' stock_code, DATE '2026-01-06' date, true is_st")
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1], commission_bps=2., sell_tax_bps=5.)
    grid = slot.calendar.bar_grid('1d')
    micro = slot.microstructure
    assert micro.tradability_mask(grid, instrument_id='A').tolist() == [False, False, False]
    assert micro.tradability_mask(grid.assign(side='buy'), instrument_id='A').tolist() == [False, False, True]
    assert micro.tradability_mask(grid.assign(side='sell'), instrument_id='A').tolist() == [True, False, False]
    assert not micro.tradability_mask(grid.assign(side='sell', entry_session_id=grid.session_id),
                                      instrument_id='A').any()
    assert not micro.tradability_mask(grid, instrument_id='B').iloc[0]
    connection.execute('DROP TABLE regime_all_a_paused_daily_v1')
    assert micro.tradability_mask(grid.assign(side='buy'), instrument_id='A').tolist() == [False, True, True]
    connection.execute('DROP TABLE all_a_paused_daily')
    assert not micro.tradability_mask(grid.assign(side='buy'), instrument_id='A').iloc[1]
    profile = micro.execution_profile
    assert (profile.entry_price_field, profile.exit_price_field, profile.min_entry_offset_bars,
            profile.exit_adjustment, profile.return_unit, profile.gap_rule, profile.boundary_policy,
            profile.allow_short, profile.commission_bps, profile.sell_tax_bps) == (
                'close', 'close', 0, 0, 'decimal', 'endpoints_only', 'cross_session', False, 2., 5.)


def test_intraday_tradability_uses_minute_price_and_aligned_t_plus_one(connection):
    _prices(connection, 'price_data_daily', DATES)
    _prices(connection, 'price_data_5m', [day + ' 09:35:00' for day in DATES])
    connection.execute("UPDATE price_data_5m SET close=110 WHERE date='2026-01-06 09:35:00'")
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1])
    grid = slot.calendar.bar_grid('5m', DATES[0], DATES[0]).iloc[:1]
    assert not slot.microstructure.tradability_mask(grid.assign(side='buy'), instrument_id='A').iloc[0]
    assert slot.microstructure.tradability_mask(grid.assign(side='sell'), instrument_id='A').iloc[0]
    assert not slot.microstructure.tradability_mask(
        grid.assign(side='sell', entry_session_id=DATES[0]), instrument_id='A').iloc[0]


def test_compatibility_reader_keeps_old_daily_timestamp(connection):
    _prices(connection, 'price_data_daily', DATES)
    daily = ASharePanelReader(connection).read(['A'], '1d')
    assert daily.timestamp_utc.iloc[0] == pd.Timestamp('2026-01-05 16:00Z')
    assert daily.attrs['daily_timestamp_semantics'] == 'local_midnight_day_identifier'


def test_review_future_dividend_cancellation_preserves_implementation_history(connection):
    connection.execute("""CREATE TABLE dividend_events AS SELECT
        'A' stock_code, DATE '2026-01-05' implementation_pub_date,
        DATE '2026-01-06' a_ex_dividend_date, 'cash dividend' implementation_bonusnote,
        NULL::DATE bonus_cancel_pub_date""")
    slot = build_jq_ashare_local(connection, start=DATES[0], end='2026-01-12')
    grid = slot.calendar.bar_grid('1d').assign(instrument_id='A')
    before_cancel = grid.loc[grid.session_id.eq('2026-01-07')]
    note, cancel = ('dividend_events.implementation_bonusnote', 'dividend_events.bonus_cancel_pub_date')
    before = slot.read_aux_features(before_cancel, features=[note])
    assert before[note].tolist() == ['cash dividend']

    connection.execute("UPDATE dividend_events SET bonus_cancel_pub_date = DATE '2026-01-09'")
    pd.testing.assert_frame_equal(slot.read_aux_features(before_cancel, features=[note]), before)
    result = slot.read_aux_features(grid, features=[note, cancel])
    assert result.loc[result[note].notna(), 'session_id'].tolist() == grid.session_id.tolist()
    cancellation = result.loc[result[cancel].notna()]
    assert cancellation.session_id.tolist() == ['2026-01-12']
    assert cancellation.available_at_utc.tolist() == [pd.Timestamp('2026-01-09 16:00Z')]


def test_review_audit_opinion_is_visible_on_activation_day(connection):
    connection.execute("""CREATE TABLE audit_opinion_events AS SELECT
        'A' stock_code, DATE '2026-01-06' pub_date, DATE '2026-01-07' activation_date,
        'unqualified' opinion_type""")
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1])
    grid = slot.calendar.bar_grid('1d', DATES[0], DATES[1]).assign(instrument_id='A')
    feature = 'audit_opinion_events.opinion_type'
    result = slot.read_aux_features(grid, features=[feature])
    assert result.session_id.tolist() == [DATES[1]]
    assert result[feature].tolist() == ['unqualified']


def test_review_industry_uses_last_effective_snapshot_before_grid_start(connection):
    connection.execute("""CREATE TABLE industry_membership AS SELECT
        date::DATE AS date, stock_code, sw_l1_code FROM (VALUES
        ('2026-01-06', 'A', 'I'), ('2026-01-06', 'B', 'II'),
        ('2026-01-08', 'A', 'III')) AS v(date, stock_code, sw_l1_code)""")
    slot = build_jq_ashare_local(connection, start=DATES[1], end=DATES[-1])
    day_grid = slot.calendar.bar_grid('1d')
    minute_grid = slot.calendar.bar_grid('5m', DATES[-1], DATES[-1]).iloc[:1]
    grid = pd.concat([day_grid, minute_grid], ignore_index=True)
    feature = 'industry_membership.sw_l1_code'
    result = slot.read_aux_features(grid.assign(instrument_id='A'), features=[feature])
    assert result.sort_values('timestamp_utc')[feature].tolist() == ['I', 'I', 'III']
    assert result.sort_values('timestamp_utc').available_at_utc.tolist() == pd.to_datetime([
        '2026-01-06 07:00Z', '2026-01-06 07:00Z', '2026-01-08 07:00Z']).tolist()
    dropped = slot.read_aux_features(grid.assign(instrument_id='B'), features=[feature])
    assert dropped.sort_values('timestamp_utc')[feature].tolist() == ['II', 'II']


def test_review_tradability_requires_valid_prices_at_the_requested_frequency(connection):
    _prices(connection, 'price_data_daily', DATES)
    _prices(connection, 'price_data_5m', [day + ' 09:35:00' for day in DATES])
    connection.execute("DELETE FROM price_data_5m WHERE stock_code='B'")
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1])
    daily = slot.calendar.bar_grid('1d', DATES[0], DATES[0])
    minutes = slot.calendar.bar_grid('5m', DATES[0], DATES[0]).iloc[:1]
    micro = slot.microstructure
    assert micro.tradability_mask(daily, instrument_id='B').tolist() == [True]
    assert micro.tradability_mask(daily, instrument_id='absent').tolist() == [False]
    assert micro.tradability_mask(minutes, instrument_id='B').tolist() == [False]
    assert micro.tradability_mask(minutes, instrument_id='A').tolist() == [True]
    invalid = pd.concat([daily, daily, daily], ignore_index=True).assign(close=[0.0, np.inf, np.nan])
    assert micro.tradability_mask(invalid, instrument_id='A').tolist() == [False, False, False]


def test_review_signal_validates_only_minute_rows_visible_at_the_anchor(connection):
    labels = [day + ' 09:35:00' for day in DATES]
    _prices(connection, 'price_data_5m', labels)
    slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1])
    frequency = Frequency.parse('5m')
    start = pd.Timestamp(f'{DATES[0]} 01:30Z')
    end = pd.Timestamp(f'{DATES[0]} 07:00Z')
    anchor = pd.Timestamp(f'{DATES[0]} 01:35Z')
    before = slot.bar_feed.read_signal_bars(['A'], frequency, start_utc=start, end_utc=end, as_of_utc=anchor)
    connection.execute("""INSERT INTO price_data_5m SELECT * REPLACE (TIMESTAMP '2026-01-06 14:01' AS date)
        FROM price_data_5m WHERE stock_code='A' AND date=TIMESTAMP '2026-01-06 09:35'""")
    after = slot.bar_feed.read_signal_bars(['A'], frequency, start_utc=start, end_utc=end, as_of_utc=anchor)
    pd.testing.assert_frame_equal(after, before)
    assert after.close.tolist() == [102.0]
    with pytest.raises(ValueError, match='off-grid'):
        slot.bar_feed.read_signal_bars(['A'], frequency, start_utc=start, end_utc=end,
                                       as_of_utc=pd.Timestamp(f'{DATES[0]} 06:01Z'))


def _build_quarterly_financials(net_profit: float = 100.0, cfo: float = 20.0) -> pd.DataFrame:
    rows = []
    for end_dt in pd.date_range('2018-03-31', '2023-12-31', freq='QE'):
        pub_dt = end_dt + pd.Timedelta(days=45)
        rows.append({
            'statement_type': 'income',
            'stock_code': 'A',
            'pub_date': pub_dt,
            'start_date': pd.Timestamp(year=end_dt.year, month=1, day=1),
            'end_date': end_dt,
            'net_profit': net_profit,
            'net_operate_cash_flow': cfo,
            'total_liability': np.nan,
            'total_assets': np.nan,
        })
        rows.append({
            'statement_type': 'balance',
            'stock_code': 'A',
            'pub_date': pub_dt,
            'start_date': end_dt,
            'end_date': end_dt,
            'net_profit': np.nan,
            'net_operate_cash_flow': np.nan,
            'total_liability': 1000.0,
            'total_assets': 2000.0,
        })
    return pd.DataFrame(rows)


def test_quality_features_preserve_same_pub_date_dedup_parity():
    from slots.jq_ashare_local.aux import _quality_features_for_stock

    financials = _build_quarterly_financials(net_profit=100.0, cfo=20.0)
    conflict_end = pd.Timestamp('2020-12-31')
    financials = pd.concat([
        financials,
        pd.DataFrame([{
            'statement_type': 'income',
            'stock_code': 'A',
            'pub_date': conflict_end + pd.Timedelta(days=45),
            'start_date': pd.Timestamp('2020-01-01'),
            'end_date': conflict_end,
            'net_profit': 200.0,
            'net_operate_cash_flow': 40.0,
            'total_liability': np.nan,
            'total_assets': np.nan,
        }]),
    ], ignore_index=True).sort_values(['stock_code', 'pub_date', 'end_date'])
    dividends = pd.DataFrame([
        {
            'stock_code': 'A',
            'report_date': pd.Timestamp(f'{yr}-12-31'),
            'announcement_date': pd.Timestamp(f'{yr + 1}-04-30'),
            'bonus_amount_rmb': 0.005,
            'comparable_dps': 0.1,
        }
        for yr in (2020, 2021, 2022)
    ])
    valuation = pd.DataFrame({'stock_code': ['A'], 'date': [pd.Timestamp('2024-05-31')], 'market_cap': [10.0]})
    industry = pd.DataFrame({'stock_code': ['A'], 'date': [pd.Timestamp('2024-05-31')], 'sw_l1_name': ['Tech']})
    result = _quality_features_for_stock(pd.Timestamp('2024-06-01'), financials, dividends, valuation, industry)
    assert result is not None
    assert result['roe_mean'] == pytest.approx(0.10)
    assert result['roe_change'] == pytest.approx(0.0)
    assert result['cfo_liability'] == pytest.approx(0.02)


def test_quality_features_ignore_nat_dividend_report_date():
    from slots.jq_ashare_local.aux import _quality_features_for_stock

    financials = _build_quarterly_financials(net_profit=100.0, cfo=20.0)
    nat_only_div = pd.DataFrame([{
        'stock_code': 'A',
        'report_date': pd.NaT,
        'announcement_date': pd.Timestamp('2023-05-01'),
        'bonus_amount_rmb': 1.0,
        'comparable_dps': 0.1,
    }])
    valuation = pd.DataFrame({'stock_code': ['A'], 'date': [pd.Timestamp('2024-05-31')], 'market_cap': [10.0]})
    assert _quality_features_for_stock(
        pd.Timestamp('2024-06-01'), financials, nat_only_div, valuation, None,
    ) is None

    valid_div = pd.DataFrame([
        {
            'stock_code': 'A',
            'report_date': pd.Timestamp(f'{yr}-12-31'),
            'announcement_date': pd.Timestamp(f'{yr + 1}-04-30'),
            'bonus_amount_rmb': 0.005,
            'comparable_dps': 0.1,
        }
        for yr in (2020, 2021, 2022)
    ])
    with_nat_div = pd.concat([valid_div, nat_only_div], ignore_index=True)
    assert _quality_features_for_stock(
        pd.Timestamp('2024-06-01'), financials, with_nat_div, valuation, None,
    ) == _quality_features_for_stock(
        pd.Timestamp('2024-06-01'), financials, valid_div, valuation, None,
    )


def test_quality_features_exact_groupby_sum_preserves_payout_boundary(connection):
    from slots.jq_ashare_local.aux import AShareDividendQualityPlugin, _quality_features_for_stock

    financials = _build_quarterly_financials(net_profit=100000.0, cfo=20000.0)
    dividends = pd.DataFrame([
        {
            'stock_code': 'A',
            'report_date': pd.Timestamp(f'{yr}-12-31'),
            'announcement_date': pd.Timestamp(f'{yr + 1}-04-30'),
            'bonus_amount_rmb': 0.1,
            'comparable_dps': 0.01,
        }
        for yr in (2020, 2021, 2022)
        for _ in range(10)
    ])
    valuation = pd.DataFrame({'stock_code': ['A'], 'date': [pd.Timestamp('2024-05-31')], 'market_cap': [10.0]})
    result = _quality_features_for_stock(
        pd.Timestamp('2024-06-01'), financials, dividends, valuation, None,
    )
    assert result is not None
    assert result['payout_mean'] >= 0.10
    assert result['payout_mean'] == float(np.mean([0.1, 0.1, 0.1]))

    connection.register('fin_df', financials)
    connection.execute('CREATE TABLE dividend_quality_financial_events AS SELECT * FROM fin_df')
    div_table = dividends.rename(columns={'announcement_date': 'implementation_pub_date'}).assign(
        board_plan_pub_date=pd.NaT,
        shareholders_plan_pub_date=pd.NaT,
        bonus_type='年度分红',
        bonus_ratio_rmb=0.1,
        dividend_ratio=0.0,
        transfer_ratio=0.0,
    )
    connection.register('div_df', div_table)
    connection.execute('CREATE TABLE dividend_events AS SELECT * FROM div_df')
    connection.execute("CREATE TABLE valuation_daily AS SELECT 'A' stock_code, DATE '2024-06-01' date, 10.0 market_cap")
    connection.execute("CREATE TABLE industry_membership AS SELECT 'A' stock_code, DATE '2024-06-01' date, 'Tech' sw_l1_name")
    filtered, diagnostics = AShareDividendQualityPlugin(connection).filter_universe(
        {'2024-06-01': ['A']},
        config={'enabled': True, 'min_qualified_count': 1, 'min_score_percentile': 0.0},
    )
    assert filtered == {'2024-06-01': ['A']}
    assert diagnostics['eligible_counts']['2024-06-01'] == 1
