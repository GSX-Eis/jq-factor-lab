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


def test_data_source_parity_csv_duckdb_file_and_borrowed_connection(tmp_path):
    """Same market data across CSV, DuckDB file, and borrowed connection yields identical factor, IC, and ranks."""
    from dataclasses import replace

    from backtest_and_analysis.alphalens_runner import AlphalensRunner
    from backtest_and_analysis.time_series.evaluator import evaluate_slot
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame
    from slots import build_slot_from_config
    from slots.jq_ashare_local import build_jq_ashare_local
    from utils.factor_base import FactorBase
    from utils.session_calendar import SessionCalendar
    from utils.slots import ExecutionProfile, ProviderConfig, SlotConfig
    from utils.trading_calendar import TradingCalendar

    dates = tuple(SessionCalendar('XSHG', '2026-01-05', '2026-01-23').sessions().strftime('%Y-%m-%d'))
    stocks = ['000001.XSHE', '000002.XSHE', '600000.XSHG', '600519.XSHG']
    # Include a paused day (paused=1, money=0 -> NaN) and a normal zero-turnover day (paused=0, money=0 -> 0.0).
    # Keep >= 50 valid points so AlphalensRunner runs full IC & quantile analysis too.
    money_by_stock = {
        '000001.XSHE': [100., 0., 200., 240., 260.] + [280. + 20. * i for i in range(10)],
        '000002.XSHE': [300., 340., 0., 120., 90.] + [110. + 15. * i for i in range(10)],
        '600000.XSHG': [200., 180., 260., 320., 380.] + [400. + 25. * i for i in range(10)],
        '600519.XSHG': [500., 540., 620., 700., 760.] + [800. + 30. * i for i in range(10)],
    }
    rows = []
    for day_idx, day in enumerate(dates):
        for stock_idx, code in enumerate(stocks):
            is_paused = int(code == '000001.XSHE' and day_idx == 1)
            money = float(money_by_stock[code][day_idx])
            vol = 0.0 if money == 0.0 else 100.0 + 10.0 * day_idx + 25.0 * stock_idx
            close = 10.0 * (stock_idx + 1) + 0.4 * day_idx + 0.15 * ((day_idx + stock_idx) % 3)
            rows.append((code, day, close - 0.1, close + 0.3, close - 0.3, close, vol, money, is_paused, 1.0))
    frame = pd.DataFrame(
        rows,
        columns=['stock_code', 'date', 'open', 'high', 'low', 'close', 'volume', 'money', 'paused', 'factor'],
    )

    class MoneyMeanParityFactor(FactorBase):
        def _get_raw_data_for_factor(self) -> pd.DataFrame:
            return self._fetch_raw_data(['money'], table_name='price_data')

        def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
            time_col, code_col = ('date', 'code') if 'date' in data else ('timestamp_utc', 'instrument_id')
            wide = data.pivot(index=time_col, columns=code_col, values='money').sort_index()
            return wide.rolling(2, min_periods=1).mean()

    csv_df = frame.assign(date=frame['date'] + ' 15:00:00')
    csv_df.to_csv(tmp_path / 'bars.csv', index=False)
    csv_cfg = SlotConfig(
        schema_version=1,
        slot_id='csv_parity_slot',
        market_domain='cn_equity',
        calendar=ProviderConfig('pmc', {'calendar_id': 'XSHG', 'start': dates[0], 'end': dates[-1]}),
        bar_feed=ProviderConfig('csv_files', {
            'paths': ['bars.csv'],
            'frequency': '1d',
            'timestamp_anchor': 'bar_end',
            'source_timezone': 'Asia/Shanghai',
            'volume_mode': 'interval',
            'source': 'csv_parity',
            'data_version': 'v1',
            'column_mapping': {
                'date': 'timestamp',
                'stock_code': 'instrument_id',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
            },
        }),
        universe=ProviderConfig('static', {
            'instruments': {
                code: {'asset_class': 'equity', 'currency': 'CNY', 'lot_size': 100, 'contract_multiplier': 1.0}
                for code in stocks
            },
            'universes': {'all': tuple(stocks)},
        }),
        aux_features=ProviderConfig('none', {}),
        microstructure=ProviderConfig('open', {}),
        execution_profile=ExecutionProfile('close', 'close', 0, 0, 'decimal', 'endpoints_only', 'cross_session', False),
    )
    csv_slot = build_slot_from_config(csv_cfg, base_dir=tmp_path)

    db_path = tmp_path / 'parity.duckdb'
    with duckdb.connect(str(db_path)) as writer:
        declare_market_db(writer, 'cn_equity')
        writer.register('source_rows', frame.assign(date=pd.to_datetime(frame['date'])))
        writer.execute('CREATE TABLE price_data_daily AS SELECT * FROM source_rows')

    csv_factor = MoneyMeanParityFactor(
        None, 'MoneyParityCSV', dates[0], dates[-1],
        stock_pool=stocks, slot=csv_slot, data_frequency='1d', analysis_mode='cross_sectional',
    )
    csv_raw_panel = csv_factor.get_factor_values()
    csv_eval = evaluate_slot(
        csv_slot,
        bar_factor_frame(csv_raw_panel),
        frequency='1d',
        start_date=dates[0],
        end_date=dates[-1],
        instruments=stocks,
        horizons=[1, 2],
        entry_offset_bars=1,
        n_quantiles=2,
        min_ic_samples=3,
        min_cross_section_size=3,
        analysis_mode='cross_sectional',
    )
    csv_panel = csv_raw_panel.copy()
    csv_panel.index = pd.DatetimeIndex(csv_panel.index.tz_convert('Asia/Shanghai').tz_localize(None).normalize(), name='date')
    csv_panel.columns.name = 'code'

    # Compare CSV against both DuckDB file (duckdb_path) and borrowed DuckDB connection (db_connection).
    duckdb_factor_daily = MoneyMeanParityFactor(
        None, 'MoneyParityFileDaily', dates[0], dates[-1],
        stock_pool=stocks, duckdb_path=str(db_path), data_frequency='daily',
    )
    duckdb_factor_1d = MoneyMeanParityFactor(
        None, 'MoneyParityFile1d', dates[0], dates[-1],
        stock_pool=stocks, duckdb_path=str(db_path), data_frequency='1d', analysis_mode='cross_sectional',
    )
    with duckdb.connect(str(db_path), read_only=True) as borrowed:
        borrowed_factor_daily = MoneyMeanParityFactor(
            borrowed, 'MoneyParityBorrowedDaily', dates[0], dates[-1],
            stock_pool=stocks, data_frequency='daily',
        )
        borrowed_factor_1d = MoneyMeanParityFactor(
            borrowed, 'MoneyParityBorrowed1d', dates[0], dates[-1],
            stock_pool=stocks, data_frequency='1d', analysis_mode='cross_sectional',
        )
        try:
            file_daily_panel = duckdb_factor_daily.get_factor_values()
            file_1d_panel = duckdb_factor_1d.get_factor_values()
            borrowed_daily_panel = borrowed_factor_daily.get_factor_values()
            borrowed_1d_panel = borrowed_factor_1d.get_factor_values()

            # P1-01 suspended day rolling mean & unpaused zero-turnover preservation across all paths.
            for panel in (csv_panel, file_daily_panel, file_1d_panel, borrowed_daily_panel, borrowed_1d_panel):
                np.testing.assert_allclose(
                    panel['000001.XSHE'].iloc[:5].to_numpy(dtype=float),
                    [100.0, 100.0, 200.0, 220.0, 250.0],
                    rtol=0, atol=1e-12,
                )
                np.testing.assert_allclose(
                    panel['000002.XSHE'].iloc[:5].to_numpy(dtype=float),
                    [300.0, 320.0, 170.0, 60.0, 105.0],
                    rtol=0, atol=1e-12,
                )

            # 1. Exact factor value and cross-sectional rank parity across CSV, DuckDB file, and borrowed connection.
            for other in (file_daily_panel, file_1d_panel, borrowed_daily_panel, borrowed_1d_panel):
                pd.testing.assert_frame_equal(
                    other, csv_panel, check_exact=False, rtol=0, atol=1e-12, check_names=False, check_freq=False,
                )
                pd.testing.assert_frame_equal(
                    other.rank(axis=1, ascending=True),
                    csv_panel.rank(axis=1, ascending=True),
                    check_exact=True,
                    check_names=False,
                    check_freq=False,
                )

            # 2. Unified Slot evaluation (IC & quantile stats) parity across CSV, DuckDB file, and borrowed connection.
            grid = csv_slot.calendar.bar_grid('1d', dates[0], dates[-1])
            keys = grid.groupby(pd.to_datetime(grid.session_id))['timestamp_utc'].max()
            file_eval_slot = replace(
                build_jq_ashare_local(duckdb_factor_daily._duckdb_adapter._get_connection(), start=dates[0], end=dates[-1]),
                slot_config_hash='ashare-file-v1',
            )
            borrowed_eval_slot = replace(
                build_jq_ashare_local(borrowed, start=dates[0], end=dates[-1]),
                slot_config_hash='ashare-borrowed-v1',
            )
            for eval_slot, panel in ((file_eval_slot, file_1d_panel), (borrowed_eval_slot, borrowed_1d_panel)):
                utc_panel = panel.set_axis(pd.DatetimeIndex(pd.to_datetime(panel.index).map(keys)))
                slot_eval = evaluate_slot(
                    eval_slot,
                    bar_factor_frame(utc_panel),
                    frequency='1d',
                    start_date=dates[0],
                    end_date=dates[-1],
                    instruments=stocks,
                    horizons=[1, 2],
                    entry_offset_bars=1,
                    n_quantiles=2,
                    min_ic_samples=3,
                    min_cross_section_size=3,
                    analysis_mode='cross_sectional',
                )
                pd.testing.assert_frame_equal(
                    slot_eval.ic_by_horizon[['horizon_bars', 'mean_ic', 'median_ic', 'icir']],
                    csv_eval.ic_by_horizon[['horizon_bars', 'mean_ic', 'median_ic', 'icir']],
                    check_exact=False, rtol=0, atol=1e-12,
                )
                pd.testing.assert_frame_equal(
                    slot_eval.cross_sections[['signal_timestamp', 'horizon_bars', 'valid_instruments', 'spearman_ic']],
                    csv_eval.cross_sections[['signal_timestamp', 'horizon_bars', 'valid_instruments', 'spearman_ic']],
                    check_exact=False, rtol=0, atol=1e-12,
                )
                pd.testing.assert_frame_equal(
                    slot_eval.quantile_stats[['horizon_bars', 'quantile', 'total_sample_count', 'mean_return_pct']],
                    csv_eval.quantile_stats[['horizon_bars', 'quantile', 'total_sample_count', 'mean_return_pct']],
                    check_exact=False, rtol=0, atol=1e-12,
                )

            # 3. Alphalens IC parity across DuckDB file and borrowed connection (plus CSV pricing).
            cal = duckdb_factor_daily.slot.calendar.default
            trading_cal = TradingCalendar(cal.sessions(), cal.start, cal.end, calendar_id=cal.calendar_id)
            file_pricing = duckdb_factor_daily.get_price_data('close').rename_axis(columns='asset').unstack().rename('close')
            borrowed_pricing = borrowed_factor_daily.get_price_data('close').rename_axis(columns='asset').unstack().rename('close')
            file_pricing.attrs['trading_calendar'] = trading_cal
            borrowed_pricing.attrs['trading_calendar'] = trading_cal
            runner = AlphalensRunner(periods=(1, 2), n_quantiles=2, execution_lag=1)
            al_file = runner.run(file_daily_panel, file_pricing)
            al_borrowed = runner.run(borrowed_daily_panel, borrowed_pricing)
            al_csv = runner.run(csv_panel, borrowed_pricing)
            pd.testing.assert_frame_equal(al_borrowed['ic'], al_file['ic'], check_exact=False, rtol=0, atol=1e-12)
            pd.testing.assert_frame_equal(al_csv['ic'], al_file['ic'], check_exact=False, rtol=0, atol=1e-12)
            pd.testing.assert_frame_equal(al_borrowed['quantile_returns'], al_file['quantile_returns'], check_exact=False, rtol=0, atol=1e-12)
            pd.testing.assert_frame_equal(al_csv['quantile_returns'], al_file['quantile_returns'], check_exact=False, rtol=0, atol=1e-12)
        finally:
            duckdb_factor_daily.close()
            duckdb_factor_1d.close()
