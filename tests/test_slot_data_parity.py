"""Stage 2 input parity checks on isolated reproducible market data."""
from pathlib import Path

import duckdb

from market_db_samples import declare_market_db
import numpy as np
import pandas as pd
import pytest

from test_slot_futures import _populate
from utils.bars.continuous import ContinuousBarSource
from utils.bars.futures_roll import build_roll_schedule
from utils.data_access_config import load_data_access_config
from utils.duckdb_adapter import DuckDBAdapter
from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter
from utils.pit import apply_pit_adjustment, normalize_price_money
from utils.time_axis import Frequency


DATES = ('2026-01-06', '2026-01-07', '2026-01-08')


def _same_numbers(actual, expected, fields, record_property, *, name):
    left, right = (frame[fields].to_numpy(dtype=float) for frame in (actual, expected))
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-12, equal_nan=True)
    finite = np.isfinite(left) & np.isfinite(right)
    record_property(f'{name}.matched_rows', len(left))
    record_property(f'{name}.compared_values', left.size)
    record_property(f'{name}.matched_nan_values', int(np.isnan(left).sum()))
    record_property(f'{name}.max_abs_error', float(np.max(np.abs(left[finite] - right[finite]), initial=0.)))


@pytest.mark.parametrize('frequency,table,tail,opening,closing', [
    ('1d', 'price_data_daily', '', '07:00', '07:00'),
    ('5m', 'price_data_5m', ' 09:35:00', '01:30', '01:35'),
])
def test_ashare_execution_and_pit_match_master_adapter(
    tmp_path, frequency, table, tail, opening, closing, record_property,
):
    from test_slot_jq_ashare_local import FIELDS, _prices

    from slots.jq_ashare_local import build_jq_ashare_local

    day1, day2, day3 = DATES
    path = tmp_path / 'ashare.duckdb'
    with duckdb.connect(str(path)) as connection:
        declare_market_db(connection, 'cn_equity')
        _prices(connection, table, [day + tail for day in DATES])
        slot = build_jq_ashare_local(connection, start=day1, end=day3)
        adapter = DuckDBAdapter(duckdb_path=str(path), data_bindings={})
        adapter._conn = connection
        anchor = pd.Timestamp(f'{day2} {closing}Z')
        lo, hi = (pd.Timestamp(f'{day} {opening}Z') for day in (day1, day3))
        parsed = Frequency.parse(frequency)
        raw = adapter._fetch_panel(load_data_access_config().dataset(table), ['A', 'B'], day1, day3, FIELDS)
        actual = slot.bar_feed.read_execution_bars(['A', 'B'], parsed, start_utc=lo, end_utc=hi)
        sessions = raw.date.dt.strftime('%Y-%m-%d')
        keys = pd.to_datetime(sessions + f' {opening}Z', utc=True)
        assert actual.instrument_id.tolist() == raw.code.tolist()
        assert actual.timestamp_utc.tolist() == keys.tolist()
        assert actual.session_id.tolist() == sessions.tolist()
        _same_numbers(actual, raw, FIELDS, record_property, name='execution')

        historical = raw.loc[sessions.le(day2)]
        expected = apply_pit_adjustment(normalize_price_money(historical, source='local',
                                            frequency='daily' if frequency == '1d' else frequency))
        actual = slot.bar_feed.read_signal_bars(['A', 'B'], parsed, start_utc=lo, end_utc=hi, as_of_utc=anchor)
        assert actual.timestamp_utc.tolist() == keys.loc[sessions.le(day2)].tolist()
        assert actual.instrument_id.tolist() == expected.code.tolist()
        _same_numbers(actual, expected, FIELDS, record_property, name='signal')


def test_ashare_universe_and_feature_clocks_match_master_queries(record_property):
    from slots.jq_ashare_local import build_jq_ashare_local

    with duckdb.connect() as connection:
        declare_market_db(connection, 'cn_equity')
        connection.execute("""CREATE TABLE factor_data AS SELECT stock_code, day::DATE date, cap val_market_cap,
             false paused, false is_st, false is_new_stock, alpha
             FROM (VALUES ('A','2026-01-06',1,0.25), ('B','2026-01-06',2,NULL), ('C','2026-01-06',3,0.75),
                          ('B','2026-01-07',2,1.25), ('C','2026-01-07',1,NULL)) v(stock_code,day,cap,alpha)""")
        connection.execute("""CREATE TABLE index_constituents AS SELECT index_code, stock_code, day::DATE date
            FROM (VALUES ('000300.XSHG'), ('000905.XSHG'), ('8841431.WI')) i(index_code)
            CROSS JOIN (VALUES ('A','2026-01-06'), ('B','2026-01-06'), ('C','2026-01-06'),
                               ('B','2026-01-07'), ('C','2026-01-07')) v(stock_code,day)""")
        connection.execute("""CREATE TABLE raw_income_statement_events AS
            SELECT code, published::DATE pub_date, revenue FROM
            (VALUES ('A','2026-01-05',7.25), ('A','2026-01-06',8.25), ('B','2026-01-06',NULL),
                    ('A','2026-01-08',999.)) v(code,published,revenue)""")
        slot = build_jq_ashare_local(connection, start=DATES[0], end=DATES[-1])
        manager = DynamicStockPoolManager(connection, price_table='price_data_daily', data_bindings={})
        for day in DATES[:2]:
            for key, code in [('csi300', '000300.XSHG'), ('csi500', '000905.XSHG'), ('wind_micro', '8841431.WI')]:
                filters = [StockFilter('index_code', '==', code), StockFilter('paused', '==', 0),
                           StockFilter('is_st', '==', 0)]
                limit = None
                if key == 'wind_micro':
                    filters += [StockFilter('is_new_stock', '==', 0),
                                StockFilter('val_market_cap', 'percentile_between', (20, 80))]
                    limit = 500
                expected = manager._select_stock_pool(pd.Timestamp(day).to_pydatetime(), filters, max_stocks=limit)
                assert slot.universe.resolve_universe(key, as_of_utc=pd.Timestamp(f'{day} 07:00Z')) == expected
        record_property('matched_universe_snapshots', 6)
        grid = slot.calendar.bar_grid('1d', DATES[0], DATES[1]).merge(
            pd.DataFrame({'instrument_id': ['A', 'B', 'C']}), how='cross')
        alpha = slot.read_aux_features(grid, features=['alpha']).sort_values(['session_id', 'instrument_id'])
        expected = connection.execute(
            'SELECT stock_code, date::VARCHAR session_id, alpha FROM factor_data ORDER BY date, stock_code').fetchdf()
        assert alpha.instrument_id.tolist() == expected.stock_code.tolist()
        assert alpha.session_id.tolist() == expected.session_id.tolist()
        assert alpha.timestamp_utc.tolist() == alpha.available_at_utc.tolist()
        _same_numbers(alpha, expected, ['alpha'], record_property, name='factor')

        field = 'raw_income_statement_events.revenue'
        actual = slot.read_aux_features(grid, features=[field]).sort_values(['session_id', 'instrument_id'])
        # The historical date-granular publication cutoff is pub_date < the grid day.
        expected = connection.execute('''SELECT instrument_id, session_id, pub_date,
              revenue AS "raw_income_statement_events.revenue"
            FROM grid JOIN raw_income_statement_events ON code = instrument_id
              AND pub_date = (SELECT max(pub_date) FROM raw_income_statement_events
                              WHERE code = instrument_id AND pub_date < session_id::DATE)
            ORDER BY session_id, instrument_id''').fetchdf()
        assert actual.instrument_id.tolist() == expected.instrument_id.tolist()
        assert actual.session_id.tolist() == expected.session_id.tolist()
        publication = pd.to_datetime(expected.pub_date).dt.tz_localize('Asia/Shanghai') + pd.Timedelta(days=1)
        assert actual.available_at_utc.tolist() == publication.dt.tz_convert('UTC').tolist()
        _same_numbers(actual, expected, [field], record_property, name='financial')


DOMESTIC = [('CU', 'SHFE'), ('SC', 'INE'), ('M', 'DCE'), ('SR', 'CZCE'), ('IF', 'CFFEX'), ('SI', 'GFEX')]
DOMESTIC_DATES = ('2026-01-05', '2026-01-06', '2026-01-09')
FUTURES_FIELDS = ['open', 'high', 'low', 'close', 'volume']


def _domestic_reference(connection, root, exchange):
    """Check the exact master reader against independently assembled OHLCV."""
    start, first, end = DOMESTIC_DATES
    raw, cal = _populate(connection, root=root, exchange=exchange, start=start, end=end,
                         expiries=['2026-02', '2026-03'])
    build_roll_schedule(connection, product_roots=[root], start=first, end=end, rule='max_oi')
    early, late = f'{root}2602.{exchange}', f'{root}2603.{exchange}'
    contract_ids = [early, late, late, late]
    dates = pd.date_range(first, end).strftime('%Y-%m-%d')
    roll = connection.execute('SELECT decision_session_id, to_contract FROM futures_roll_schedule '
                              'ORDER BY effective_session_id').fetchall()
    assert roll == list(zip(pd.date_range(start, '2026-01-08').strftime('%Y-%m-%d'), contract_ids))
    choices = pd.DataFrame({'session_id': dates, 'instrument_id': contract_ids})
    raw_selected = choices.merge(raw, on=['session_id', 'instrument_id'], validate='one_to_one')
    references = {}
    for adjustment, baseline_mode in [
        ('none', 'none'), ('panama_canal', 'diff_back_adjust'), ('ratio_back', 'ratio_back_adjust'),
    ]:
        reference = ContinuousBarSource(connection).read(root, '1d', first, end, adjust=baseline_mode)
        assert reference.contract_id.tolist() == contract_ids
        assert reference.session_id.tolist() == list(dates)
        expected = raw_selected[FUTURES_FIELDS].copy()
        prices = FUTURES_FIELDS[:4]
        if adjustment == 'ratio_back':
            expected.loc[0, prices] *= 123. / 103.
        elif adjustment == 'panama_canal':
            expected.loc[0, prices] += 20.
        np.testing.assert_allclose(reference[FUTURES_FIELDS], expected, rtol=0, atol=1e-12, equal_nan=True)
        references[adjustment] = reference
    return cal, references


@pytest.mark.parametrize('root,exchange', DOMESTIC)
def test_cn_futures_max_oi_matches_master_session_and_prices(root, exchange, record_property):
    from slots.cn_futures import build_cn_futures_slot

    start, first, end = DOMESTIC_DATES
    with duckdb.connect(config={'threads': 1}) as connection:
        cal, references = _domestic_reference(connection, root, exchange)
        lo, hi = cal.session_on(first).segments[0][0], cal.session_on(end).segments[-1][1]
        for adjustment, reference in references.items():
            slot = build_cn_futures_slot(connection, product_root=root, start=start, end=end, adjustment=adjustment)
            signal = slot.bar_feed.read_signal_bars([root], Frequency.parse('1d'),
                                                    start_utc=lo, end_utc=hi, as_of_utc=hi)
            assert signal.concrete_instrument_id.tolist() == reference.contract_id.tolist()
            pd.testing.assert_frame_equal(signal[['timestamp_utc', 'session_id']], cal.bar_grid('1d', first, end))
            _same_numbers(signal, reference, FUTURES_FIELDS, record_property, name=adjustment)


def test_overseas_roll_excludes_future_decisions_and_prices(record_property):
    from slots.global_futures import build_global_futures_slot

    start, first, end = '2026-03-05', '2026-03-06', '2026-03-10'
    with duckdb.connect(config={'threads': 1}) as connection:
        _, cal = _populate(connection, root='NQ', exchange='CME', start=start, end=end,
                           expiries=['2026-03-20', '2026-06-18'], calendar_id='CME Globex Equity',
                           reference='NYSE', frequency='30min')
        connection.execute("UPDATE market_bars_30min SET volume=600 WHERE instrument_id='NQ2603.CME' "
                           "AND timestamp_utc < '2026-03-06 22:00:00+00'")
        build_roll_schedule(connection, product_roots=['NQ'], start=first, end=end, rule='max_volume')
        slot = build_global_futures_slot(connection, product_root='NQ', start=start, end=end,
                                         adjustment='ratio_back', roll_policy='max_volume')
        lo, hi = cal.session_on(first).segments[0][0], cal.session_on(end).segments[-1][1]
        roll_open = cal.session_on(end).segments[0][0]
        anchor = roll_open - pd.Timedelta(nanoseconds=1)

        def signals(as_of):
            return slot.bar_feed.read_signal_bars(['NQ'], Frequency.parse('30min'),
                                                  start_utc=lo, end_utc=hi, as_of_utc=as_of)

        before = signals(anchor)
        assert set(before.concrete_instrument_id) == {'NQ2603.CME'}
        assert set(signals(hi).concrete_instrument_id) == {'NQ2603.CME', 'NQ2606.CME'}
        connection.execute('UPDATE market_bars_30min SET close=999 WHERE timestamp_utc>=?', [roll_open])
        connection.execute("DELETE FROM futures_roll_schedule WHERE effective_session_id=?", [end])
        pd.testing.assert_frame_equal(signals(anchor), before, check_exact=True)
        record_property('matched_rows', len(before))


def test_crypto_pit_excludes_future_bars_and_releases(tmp_path, record_property):
    from test_slot_declarative import EXAMPLE, FREQ, START, END, SYMBOL

    from slots import build_slot_from_config

    for filename in ('slot.yaml', 'bars.csv', 'funding.csv'):
        (tmp_path / filename).write_bytes((EXAMPLE / filename).read_bytes())
    slot = build_slot_from_config(tmp_path / 'slot.yaml')
    anchor = START + pd.Timedelta(hours=1)
    grid = slot.calendar.bar_grid(FREQ).iloc[:2].assign(instrument_id=SYMBOL)

    def view():
        signal = slot.bar_feed.read_signal_bars([SYMBOL], FREQ, start_utc=START, end_utc=END, as_of_utc=anchor)
        aux = slot.read_aux_features(grid, features=['funding_rate'])
        return signal, aux

    bars, releases = view()
    assert bars.close.tolist() == [103]
    assert releases.funding_rate.tolist() == [0.0001]
    assert releases.available_at_utc.tolist() == [anchor]
    (tmp_path / 'bars.csv').write_text((tmp_path / 'bars.csv').read_text().replace(',106,', ',107,'))
    (tmp_path / 'funding.csv').write_text((tmp_path / 'funding.csv').read_text().replace('0.0003', '999'))
    for before, after in zip((bars, releases), view()):
        pd.testing.assert_frame_equal(before, after, check_exact=True)
    record_property('matched_rows', len(bars) + len(releases))


def test_bars_core_has_no_legacy_or_shanghai_specifics(record_property):
    paths = sorted((Path(__file__).resolve().parents[1] / 'utils' / 'bars').glob('*.py'))
    assert paths
    for path in paths:
        contents = path.read_text(encoding='utf-8').lower()
        assert 'legacy' not in contents and 'asia/shanghai' not in contents, path.name
    record_property('scanned_modules', len(paths))
