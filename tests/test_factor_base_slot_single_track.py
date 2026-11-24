"""Synthetic market slots and exact comparisons to the frozen factor engine."""
from dataclasses import replace
from pathlib import Path

import duckdb

from market_db_samples import declare_market_db
import numpy as np
import pandas as pd
import pytest

from slots import build_slot_from_config
from slots.cn_futures import build_cn_futures_slot
from slots.jq_ashare_local import build_jq_ashare_local
from test_factor_base_frequency_routing import baseline_base
from test_slot_futures import _populate
from utils.factor_base import FactorBase
from utils.slots import ProviderConfig, SlotRegistry, load_slot_config
from utils.time_axis import Frequency


START, END = '2026-01-05', '2026-01-07'
EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'declarative_crypto'


class TurnoverFactor(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(['close', 'volume'], table_name='price_data')

    def calculate_factor(self, data):
        time, instrument = ('date', 'code') if 'date' in data else ('timestamp_utc', 'instrument_id')
        return data.assign(value=data.close * data.volume).pivot(
            index=time, columns=instrument, values='value')

    def _get_cache_inputs(self):
        return {}


def _populate_cn_stock(connection):
    declare_market_db(connection, 'cn_equity')
    daily = pd.DataFrame({
        'date': pd.to_datetime(np.repeat(pd.date_range(START, END), 2)),
        'stock_code': ['A', 'B'] * 3,
        'close': [10., 20., 12., 22., 14., np.nan],
        'volume': [100., 200., 300., 400., 500., 600.],
        'factor': [1., 1., 2., 1., 2., 1.],
    })
    minute = daily.assign(date=daily.date + pd.Timedelta(hours=9, minutes=35))
    features = daily[['date', 'stock_code']].assign(val_market_cap=[.5, 1., 2., 4., 8., 16.])
    for table, rows in [('price_data_daily', daily), ('price_data_5m', minute), ('factor_data', features)]:
        connection.from_df(rows).create(table)


def _declarative_slot(tmp_path):
    config = load_slot_config(EXAMPLE / 'slot.yaml')
    pd.read_csv(EXAMPLE / 'bars.csv').to_csv(tmp_path / 'bars.csv', index=False)
    pd.read_csv(EXAMPLE / 'funding.csv').to_csv(tmp_path / 'funding.csv', index=False)
    return build_slot_from_config(config, base_dir=tmp_path)


def _assert_equal(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual.isna(), expected.isna())


def test_unchanged_turnover_formula_runs_on_three_real_slot_implementations(tmp_path):
    with duckdb.connect(config={'threads': 1}) as collection, duckdb.connect(config={'threads': 1}) as futures_db:
        _populate_cn_stock(collection)
        ashare_slot = build_jq_ashare_local(collection, start=START, end=END)
        ashare = TurnoverFactor(None, 'turnover', START, END, stock_pool=['A'], slot=ashare_slot)
        np.testing.assert_allclose(ashare.get_factor_values()['A'], [1000., 3600., 7000.],
                                   rtol=0, atol=1e-12)
        raw, calendar = _populate(futures_db, root='CU', exchange='SHFE', start=START,
                                  end=END, expiries=['2026-03'])
        futures_slot = build_cn_futures_slot(futures_db, product_root='CU', start=START, end=END)
        futures = TurnoverFactor(None, 'turnover', START, END, stock_pool=['CU2603.SHFE'],
                                 data_frequency=Frequency.parse('1d'), slot=futures_slot,
                                 analysis_mode='time_series')
        actual = futures.get_factor_values()['CU2603.SHFE']
        pd.testing.assert_index_equal(actual.index, pd.DatetimeIndex(
            calendar.bar_grid('1d').timestamp_utc, name='date'))
        np.testing.assert_allclose(actual, raw.close * raw.volume, rtol=0, atol=1e-12)
        np.testing.assert_allclose(futures.get_price_data()['CU2603.SHFE'], raw.close, rtol=0, atol=1e-12)
        assert futures.instrument_metadata.set_index('instrument_id').loc['CU2603.SHFE', 'point_value'] == 20.

    declarative = _declarative_slot(tmp_path)
    registry = SlotRegistry()
    registry.register(declarative)
    crypto = TurnoverFactor(None, 'turnover', '2026-01-03', '2026-01-03', stock_pool=['BTC-USD'],
                            slot_id=declarative.slot_id, slot_registry=registry, data_frequency='1h',
                            analysis_mode='time_series')
    actual = crypto.get_factor_values()['BTC-USD']
    np.testing.assert_allclose(actual, [103 * 12, 106 * 14] + [np.nan] * 22,
                               rtol=0, atol=1e-12, equal_nan=True)
    assert crypto.slot is declarative
    assert crypto.calendar_hash == {'BTC-USD': declarative.calendar.calendar_hash}
    assert crypto.market_bars.attrs['raw_coverage_by_key'][('BTC-USD', '60min')] == (
        pd.Timestamp('2026-01-03 00:00Z'), pd.Timestamp('2026-01-03 01:00Z'))


@pytest.mark.parametrize('frequency,features,mode', [
    ('daily', False, None), ('daily', True, None), ('5m', False, None), ('5m', True, None),
    ('daily', True, 'raw'), ('daily', True, 'factor'), ('daily', True, 'both'),
    ('5m', False, 'raw'), ('5m', False, 'factor'), ('5m', False, 'both'),
])
@pytest.mark.parametrize('backing', ['borrowed', 'read_only_path'])
def test_explicit_and_implicit_ashare_panels_match_before_slot_migration(tmp_path, frequency, features, mode, backing):
    frozen = baseline_base()

    def price_or_features(self):
        price = self._fetch_raw_data(['close', 'volume'], table_name='price_data')
        if features:
            aux = self._fetch_raw_data(['val_market_cap'])
            price = price.merge(aux.assign(date=aux.date.dt.normalize()),
                                left_on=[price.date.dt.normalize(), 'code'],
                                right_on=['date', 'code'], suffixes=('', '_daily')).drop(columns='date_daily')
        return price

    def formula(self, frame):
        value = frame['close'] + frame['volume'] * .01
        if features:
            value += frame.val_market_cap
        return frame.assign(value=value).pivot(index='date', columns='code', values='value')

    def factor_type(base):
        return type('Probe', (base,), {
            '_get_raw_data_for_factor': price_or_features, 'calculate_factor': formula,
            'STOCK_CHUNK_INDEPENDENT': True,
        })

    path = tmp_path / 'inputs.duckdb'
    with duckdb.connect(str(path), config={'threads': 1}) as writer:
        _populate_cn_stock(writer)
    with duckdb.connect(str(path), read_only=True) as connection:
        source = ({'duckdb_path': str(path)} if backing == 'read_only_path' else {})
        borrowed = None if source else connection
        pools = {'2026-01-05': ['A', 'B'], '2026-01-06': ['B'], '2026-01-07': ['A', 'B']}
        options = dict(stock_pool=['A', 'B'], data_frequency=frequency, data_bindings={}, **source)
        if mode is not None:
            options.update(use_dynamic_pool=True, dynamic_pool_mode=mode, stock_pools_by_period=pools)
        args = (borrowed, 'parity', START, END)
        before = factor_type(frozen)(*args, **options)
        implicit = factor_type(FactorBase)(*args, **options)
        native = build_jq_ashare_local(connection, start=START, end=END)
        explicit = factor_type(FactorBase)(*args, slot=native, **options)
        try:
            for actual in (implicit, explicit):
                for name in ('get_factor_values', 'get_factor_values_unfiltered', 'get_price_data',
                             'calculate_forward_returns'):
                    _assert_equal(getattr(actual, name)(), getattr(before, name)())
                _assert_equal(actual.get_factor_values_unfiltered_chunked(1),
                              before.get_factor_values_unfiltered_chunked(1))
                if mode is not None and frequency == 'daily':
                    _assert_equal(actual._get_factor_values_by_rebalance_pool(1, rebalance_only=False),
                                  before._get_factor_values_by_rebalance_pool(1, rebalance_only=False))
        finally:
            for item in (implicit, explicit, before):
                if item._duckdb_adapter is not None:
                    item._duckdb_adapter.close()
        assert connection.execute('SELECT 1').fetchone() == (1,)


def test_cache_key_binds_id_config_and_calendar(tmp_path):
    original = _declarative_slot(tmp_path)
    raw = pd.DataFrame({'date': pd.to_datetime([START]), 'code': ['BTC-USD'], 'close': [5.]})

    def key(slot, config_hash=None):
        factor = TurnoverFactor(None, 'key', '2026-01-03', '2026-01-03', stock_pool=['BTC-USD'],
                                slot=slot, slot_config_hash=config_hash, data_frequency='1h')
        return factor._get_cache_key(raw)

    reference = key(original)
    assert reference is not None
    assert key(original) == reference
    assert key(replace(original, slot_id='other-market')) != reference
    assert key(original, 'different-config') != reference
    config = load_slot_config(EXAMPLE / 'slot.yaml')
    earlier = replace(config, calendar=ProviderConfig(config.calendar.provider,
                      {**config.calendar.options, 'start': '2026-01-02'}))
    other_calendar = build_slot_from_config(earlier, base_dir=tmp_path)
    assert original.calendar.calendar_hash != other_calendar.calendar.calendar_hash
    assert key(original, 'same-config') != key(other_calendar, 'same-config')


def test_releasing_factor_preserves_borrowed_connection():
    with duckdb.connect() as connection:
        factor = TurnoverFactor(connection, 'borrowed', START, END, stock_pool=['A'])
        factor.close()
        assert connection.execute('SELECT 1').fetchone() == (1,)


def test_bar_bridge_has_no_ashare_reader_dependency():
    path = Path(__file__).resolve().parents[1] / 'utils' / 'factor_core' / 'bar_bridge.py'
    source = path.read_text()
    assert 'ASharePanelReader' not in source
    assert 'market_bar_importer' not in source


class AnchoredGridFeed:
    """A provider serving any frequency supported by its session clock."""

    def __init__(self, calendar):
        self.calendar = calendar

    def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc):
        grid = self.calendar.bar_grid(frequency)
        times = grid.loc[grid.timestamp_utc.between(start_utc, end_utc), 'timestamp_utc']
        prices = 10. + np.flatnonzero(grid.timestamp_utc.isin(times))
        return pd.DataFrame({'instrument_id': instruments[0], 'timestamp_utc': times,
                             'open': prices, 'high': prices, 'low': prices, 'close': prices, 'volume': 1.})

    def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
        assert as_of_utc >= end_utc
        frame = self.read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)
        return frame.assign(close=frame.close + as_of_utc.day)


@pytest.mark.parametrize('frequency', ['1m', '2h', '4h', '1w'])
def test_factor_base_accepts_provider_supported_frequencies(tmp_path, frequency):
    from utils.session_calendar import SessionCalendar

    calendar = SessionCalendar('XSHG', START, '2026-01-09')
    original = _declarative_slot(tmp_path)
    slot = replace(original, calendar=calendar, bar_feed=AnchoredGridFeed(calendar))
    factor = TurnoverFactor(None, 'frequency', START, '2026-01-09', stock_pool=['BTC-USD'],
                            slot=slot, data_frequency=frequency, analysis_mode='time_series')
    panel = factor.get_factor_values()
    grid = calendar.bar_grid(frequency)
    pd.testing.assert_index_equal(panel.index, pd.DatetimeIndex(grid.timestamp_utc, name='date'))
    assert panel.iloc[-1, 0] == 18. + len(grid)


def test_slot_rebalance_reanchors_signal_prices_at_each_pool_window(tmp_path):
    from utils.session_calendar import SessionCalendar

    calendar = SessionCalendar('XSHG', START, '2026-01-09')
    original = _declarative_slot(tmp_path)
    slot = replace(original, calendar=calendar, bar_feed=AnchoredGridFeed(calendar))
    factor = TurnoverFactor(
        None, 'replay', START, '2026-01-09', use_dynamic_pool=True,
        stock_pools_by_period={'2026-01-07': ['BTC-USD'], '2026-01-09': ['BTC-USD']},
        slot=slot, data_frequency='1d',
    )
    actual = factor._get_factor_values_by_rebalance_pool(1)
    assert actual['BTC-USD'].tolist() == [17., 21.]
    assert factor.end_date == '2026-01-09'
    assert factor.all_stocks == ['BTC-USD']


def test_declarative_aux_features_keep_publication_clock_and_named_membership(tmp_path):
    slot = _declarative_slot(tmp_path)
    factor = TurnoverFactor(None, 'aux', '2026-01-03', '2026-01-03',
                            slot=slot, data_frequency='1h', universe_id='spot')
    assert factor.all_stocks == ['BTC-USD']
    aux = factor._fetch_raw_data(['funding_rate'])
    assert aux.date.min() == pd.Timestamp('2026-01-03 01:00Z')
    assert aux.funding_rate.notna().all()


def test_path_backed_slot_reuses_owner_adapter_without_closing_it(tmp_path):
    path = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(path)) as writer:
        _populate_cn_stock(writer)
    owner = TurnoverFactor(None, 'owner', START, END, stock_pool=['A'], duckdb_path=str(path),
                           data_bindings={})
    borrower = TurnoverFactor(None, 'borrower', START, END, stock_pool=['A'], slot=owner.slot,
                              data_bindings={})
    try:
        _assert_equal(borrower.get_price_data(), owner.get_price_data())
        _assert_equal(borrower._fetch_raw_data(['val_market_cap']), owner._fetch_raw_data(['val_market_cap']))
        borrower.close()
        assert owner._duckdb_adapter._conn.execute('SELECT 1').fetchone() == (1,)
    finally:
        owner.close()
    assert owner._duckdb_adapter._conn is None


def test_named_universe_cannot_use_membership_published_after_session_open(tmp_path):
    from slots.declarative.source import FrameSource

    original = _declarative_slot(tmp_path)
    (tmp_path / 'pool.csv').write_text(
        'instrument_id,universe_id,available_at_utc\nBTC-USD,spot,2026-01-03T01:00:00Z\n',
        encoding='utf-8',
    )
    membership = FrameSource(ProviderConfig('csv_files', {'paths': ['pool.csv']}), tmp_path,
                             ('instrument_id', 'universe_id', 'available_at_utc'))
    slot = replace(original, universe=replace(original.universe, static={}, membership=membership))
    at_open = TurnoverFactor(None, 'pool', '2026-01-03', '2026-01-03', slot=slot,
                              data_frequency='1h', universe_id='spot')
    next_day = TurnoverFactor(None, 'pool', '2026-01-04', '2026-01-04', slot=slot,
                              data_frequency='1h', universe_id='spot')
    assert at_open.all_stocks == []
    assert next_day.all_stocks == ['BTC-USD']


def _anchored_factor(tmp_path, end='2026-01-09', frequency='1d', **options):
    from utils.session_calendar import SessionCalendar

    calendar = SessionCalendar('XSHG', START, '2026-01-09')
    slot = replace(_declarative_slot(tmp_path), calendar=calendar, bar_feed=AnchoredGridFeed(calendar),
                   slot_config_hash=None)
    return TurnoverFactor(None, 'anchored', START, end, slot=slot, data_frequency=frequency,
                          **{'stock_pool': ['BTC-USD'], 'analysis_mode': 'time_series', **options})


@pytest.mark.parametrize('analysis_mode', ['time_series', None])
def test_review_published_signal_survives_append(tmp_path, analysis_mode):
    short = _anchored_factor(tmp_path, end=START, analysis_mode=analysis_mode)
    long = _anchored_factor(tmp_path, analysis_mode=analysis_mode)
    published = short.get_factor_values()
    assert published.iloc[0, 0] == 15.
    pd.testing.assert_frame_equal(long.get_factor_values().loc[published.index], published)


def test_review_each_signal_reprices_its_rolling_history(tmp_path):
    class RollingMean(TurnoverFactor):
        def calculate_factor(self, data):
            return (data.pivot(index='timestamp_utc', columns='instrument_id', values='close')
                    .rolling(2).mean())

    base = _anchored_factor(tmp_path)
    factor = RollingMean(None, 'rolling', START, '2026-01-09', stock_pool=base.stock_pool,
                         slot=base.slot, data_frequency='1d', analysis_mode='time_series')
    np.testing.assert_allclose(factor.get_factor_values()['BTC-USD'],
                               [np.nan, 16.5, 18.5, 20.5, 22.5], equal_nan=True)


def test_review_evaluation_and_session_prices_use_execution_bars(tmp_path):
    class ShiftedOHLC(AnchoredGridFeed):
        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
            frame = self.read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)
            prices = ['open', 'high', 'low', 'close']
            frame[prices] += 9
            return frame

    base = _anchored_factor(tmp_path, end=START)
    slot = replace(base.slot, bar_feed=ShiftedOHLC(base.slot.calendar))
    factor = TurnoverFactor(None, 'evaluation', START, START, stock_pool=base.stock_pool,
                            slot=slot, data_frequency='1h', analysis_mode='time_series',
                            label_config={'horizons': (2,)})
    result = factor.evaluate_bar_factor().per_bar_labels.sort_values('signal_timestamp').iloc[0]
    assert result.factor_value == 19.
    assert (result.entry_price, result.exit_price) == (11., 12.)
    assert result.return_pct == pytest.approx(100 * (12 / 11 - 1))
    assert factor.get_session_daily_pricing().iloc[0] == 13.


def test_review_forward_returns_keep_scheduled_missing_bar(tmp_path):
    slot = _declarative_slot(tmp_path)
    path = tmp_path / 'bars.csv'
    rows = pd.read_csv(path)
    rows.loc[1, 'time'] = '2026-01-03T03:00:00Z'
    rows.to_csv(path, index=False)
    factor = TurnoverFactor(None, 'gap', '2026-01-03', '2026-01-03', stock_pool=['BTC-USD'],
                            slot=slot, data_frequency='1h', analysis_mode='time_series')
    values = factor.get_factor_values()
    returns = factor.calculate_forward_returns()
    pd.testing.assert_index_equal(returns.index, values.index)
    assert returns.iloc[:2, 0].isna().all()
    assert factor.calculate_forward_returns(2).iloc[0, 0] == pytest.approx(106 / 103 - 1)


def test_review_daily_bar_returns_use_factor_utc_keys(tmp_path):
    factor = _anchored_factor(tmp_path, frequency='1d')
    values = factor.get_factor_values()
    returns = factor.calculate_forward_returns()
    pd.testing.assert_index_equal(returns.index, values.index)
    assert returns.index[0] == pd.Timestamp('2026-01-05 07:00Z')
    assert returns.iloc[0, 0] == pytest.approx(11 / 10 - 1)


@pytest.mark.parametrize('name,boundary,expected_key', [
    ('XSHG', '2026-01-07', '2026-01-06 06:00Z'),
    ('CME Globex Equity', '2026-01-07', '2026-01-06 21:00Z'),
])
def test_review_intraday_rebalance_uses_session_boundary(tmp_path, name, boundary, expected_key):
    from utils.session_calendar import SessionCalendar

    class RecordingFeed(AnchoredGridFeed):
        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
            cutoffs.append((end_utc, as_of_utc))
            return super().read_signal_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc,
                                            as_of_utc=as_of_utc)

    cutoffs = []
    calendar = SessionCalendar(name, START, '2026-01-09')
    slot = replace(_declarative_slot(tmp_path), calendar=calendar, bar_feed=RecordingFeed(calendar))
    factor = TurnoverFactor(None, 'rebalance', START, '2026-01-09', use_dynamic_pool=True,
                            stock_pools_by_period={boundary: ['BTC-USD']},
                            slot=slot, data_frequency='1h')
    result = factor._get_factor_values_by_rebalance_pool(1)
    assert result.index.tolist() == [pd.Timestamp(expected_key)]
    grid = calendar.bar_grid('1h')
    position = grid.index[grid.timestamp_utc.eq(pd.Timestamp(expected_key))].item()
    assert result.iloc[0, 0] == 10. + position + 6
    interval = factor._get_factor_values_by_rebalance_pool(1, rebalance_only=False)
    first = grid.loc[grid.session_id.eq(boundary), 'timestamp_utc'].iloc[0]
    assert interval.index[0] == first
    assert (first, first + pd.Timedelta(hours=1)) in cutoffs


class ConstantAux:
    def read_aux_features(self, grid, *, features):
        return grid.assign(available_at_utc=grid.timestamp_utc, **dict.fromkeys(features, 42.))


@pytest.mark.parametrize('frequency', ['daily', '5m'])
def test_review_explicit_ashare_auxiliary_component_is_used(frequency):
    with duckdb.connect() as connection:
        _populate_cn_stock(connection)
        native = build_jq_ashare_local(connection, start=START, end=END)
        slot = replace(native, aux_features=ConstantAux())
        factor = TurnoverFactor(None, 'aux', START, END, stock_pool=['A'],
                                slot=slot, data_frequency=frequency)
        result = factor._fetch_raw_data(['val_market_cap'])
        assert len(result) > 0
        assert result.val_market_cap.eq(42.).all()
        assert result.date.dt.tz is None


def test_review_explicit_ashare_native_auxiliary_can_borrow_another_source():
    from slots.jq_ashare_local.aux import AShareAuxFeatures

    with duckdb.connect() as connection, duckdb.connect() as features:
        _populate_cn_stock(connection)
        _populate_cn_stock(features)
        features.execute('UPDATE factor_data SET val_market_cap = 42')
        native = build_jq_ashare_local(connection, start=START, end=END)
        slot = replace(native, aux_features=AShareAuxFeatures(features, native.calendar))
        factor = TurnoverFactor(None, 'aux', START, END, stock_pool=['A'], slot=slot)
        assert factor._fetch_raw_data(['val_market_cap']).val_market_cap.tolist() == [42.] * 3


def test_review_declarative_components_allow_custom_auxiliary_provider(tmp_path):
    native = _declarative_slot(tmp_path)
    slot = replace(native, aux_features=ConstantAux(), slot_config_hash=None)
    factor = TurnoverFactor(None, 'custom', '2026-01-03', '2026-01-03', stock_pool=['BTC-USD'],
                            slot=slot, data_frequency='1h', analysis_mode='time_series')
    assert factor.slot_config_hash is None
    assert factor.get_factor_values().iloc[0, 0] == 103 * 12
    assert factor._fetch_raw_data(['funding_rate']).funding_rate.eq(42.).all()



def test_review_signal_is_retained_when_its_execution_bar_is_missing(tmp_path):
    class FirstExecutionBarMissing(AnchoredGridFeed):
        def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc):
            frame = super().read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)
            return frame.iloc[1:]

        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
            return super().read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)

    base = _anchored_factor(tmp_path, end=START)
    slot = replace(base.slot, bar_feed=FirstExecutionBarMissing(base.slot.calendar))
    factor = TurnoverFactor(None, 'execution_gap', START, START, stock_pool=base.stock_pool,
                            slot=slot, data_frequency='1h', analysis_mode='time_series',
                            label_config={'horizons': (2,)})
    result = factor.evaluate_bar_factor().per_bar_labels.sort_values('signal_timestamp').iloc[0]
    assert result.factor_value == 10.
    assert (result.entry_price, result.exit_price) == (11., 12.)



@pytest.mark.parametrize('analysis_mode', ['time_series', None])
def test_review_causal_cache_reuses_target_values_only(tmp_path, monkeypatch, analysis_mode):
    factor = _anchored_factor(tmp_path, analysis_mode=analysis_mode, slot_config_hash='anchored-v1',
                              cache_dir=str(tmp_path / 'cache'))
    values = factor.get_factor_values()
    files = list((tmp_path / 'cache').glob('*.pkl'))
    assert len(files) == len(values)
    for path in files:
        assert len(pd.read_pickle(path)) == 1
    monkeypatch.setattr(factor, 'calculate_factor', lambda raw: pytest.fail('the cutoff should be cached'))
    pd.testing.assert_frame_equal(factor.get_factor_values(), values)


@pytest.mark.parametrize('rebalance_only,dates,values', [
    (True, ['2026-01-06'], [23.]),
    (False, ['2026-01-07', '2026-01-08', '2026-01-09'], [28., 33., 38.]),
])
def test_review_daily_session_rollup_in_pool_replay(tmp_path, rebalance_only, dates, values):
    factor = _anchored_factor(
        tmp_path, frequency='1h', analysis_mode='cross_sectional', output_frequency='1d',
        use_dynamic_pool=True, stock_pools_by_period={'2026-01-07': ['BTC-USD']},
    )
    actual = factor._get_factor_values_by_rebalance_pool(1, rebalance_only=rebalance_only)
    pd.testing.assert_index_equal(actual.index, pd.DatetimeIndex(dates, name='date'))
    assert actual['BTC-USD'].tolist() == values


@pytest.mark.parametrize('execution_missing', ['entire_instrument', 'same_bar_as_signal'])
def test_review_bar_labels_sample_the_observed_signal_keys(tmp_path, execution_missing):
    class SparseFeed(AnchoredGridFeed):
        def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc):
            frame = super().read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)
            if execution_missing == 'entire_instrument':
                return frame.iloc[:0]
            return frame.loc[frame.timestamp_utc.ne(gap)]

        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
            frame = super().read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)
            if execution_missing == 'same_bar_as_signal':
                frame = frame.loc[frame.timestamp_utc.ne(gap)]
            return frame

    gap = pd.Timestamp('2026-01-05 02:30Z')
    base = _anchored_factor(tmp_path, end=START)
    slot = replace(base.slot, bar_feed=SparseFeed(base.slot.calendar))
    factor = TurnoverFactor(None, 'signal_keys', START, START, stock_pool=base.stock_pool,
                            slot=slot, data_frequency='1h', analysis_mode='time_series',
                            label_config={'horizons': (1,)})
    labels = factor.evaluate_bar_factor().per_bar_labels.sort_values('signal_timestamp')
    expected = slot.calendar.bar_grid('1h', START, START).timestamp_utc
    if execution_missing == 'entire_instrument':
        assert labels.signal_timestamp.tolist() == expected.tolist()
        assert labels.factor_value.tolist() == [10., 11., 12., 13.]
        assert labels.exclusion_reason.eq('insufficient_history').all()
        assert labels.entry_price.isna().all()
    else:
        assert labels.signal_timestamp.tolist() == expected[expected.ne(gap)].tolist()
        assert labels.factor_value.tolist() == [10., 12., 13.]
        assert labels.iloc[0].exclusion_reason == 'missing_bar'
        assert labels.iloc[1].entry_price == 13.


def test_review_each_pit_window_exposes_its_contract_metadata(tmp_path):
    class RollingUniverse:
        def instrument_spec(self, instrument_id):
            points = {'front': 2., 'next': 3.}[instrument_id]
            return replace(original_spec, instrument_id=instrument_id, contract_multiplier=points)

        def resolve_active_contract(self, instrument_id, *, as_of_utc):
            return 'front' if as_of_utc < pd.Timestamp('2026-01-05 05:00Z') else 'next'

        def resolve_universe(self, universe_id, *, as_of_utc):
            return ['BTC-USD']

    class PerContract(TurnoverFactor):
        def calculate_factor(self, data):
            metadata = self.instrument_metadata.set_index('instrument_id')
            assert self.market_bar_grid.timestamp_utc.tolist() == data.timestamp_utc.tolist()
            assert self.market_bars.timestamp_utc.tolist() == data.loc[data.observed, 'timestamp_utc'].tolist()
            assert data.attrs['instrument_metadata'][0]['point_value'] == metadata.point_value.iloc[0]
            assert self.calendar_hash == {'BTC-USD': self.slot.calendar.calendar_hash}
            return super().calculate_factor(data) * metadata.point_value

    base = _anchored_factor(tmp_path, end=START)
    original_spec = base.slot.universe.instrument_spec('BTC-USD')
    slot = replace(base.slot, universe=RollingUniverse())
    factor = PerContract(None, 'points', START, START, stock_pool=base.stock_pool,
                         slot=slot, data_frequency='1h', analysis_mode='time_series')
    for _ in range(2):
        assert factor.get_factor_values()['BTC-USD'].tolist() == [30., 32., 51., 54.]
        assert len(factor.market_bar_grid) == 4


def test_review_merged_signal_coverage_retains_source_bounds(tmp_path):
    class DeclaredCoverageFeed(AnchoredGridFeed):
        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
            frame = super().read_signal_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc,
                                             as_of_utc=as_of_utc)
            frame = frame.loc[~frame.timestamp_utc.isin([first, last])]
            # Windows may carry disjoint coverage, even when the view yields no rows.
            bounds = (end_utc, end_utc)
            frame.attrs['raw_coverage_by_instrument'] = {instruments[0]: bounds}
            frame.attrs['raw_coverage_by_key'] = {(instruments[0], frequency.canonical): bounds}
            return frame

    first, second, third, last = pd.to_datetime([
        '2026-01-05 01:30Z', '2026-01-05 02:30Z', '2026-01-05 05:00Z', '2026-01-05 06:00Z',
    ])
    base = _anchored_factor(tmp_path, end=START)
    slot = replace(base.slot, bar_feed=DeclaredCoverageFeed(base.slot.calendar))
    factor = TurnoverFactor(None, 'raw_coverage', START, START, stock_pool=base.stock_pool, slot=slot,
                            data_frequency='1h', analysis_mode='time_series', session_view='RTH')
    factor.get_factor_values()
    assert factor.market_bars.timestamp_utc.tolist() == [second, third]
    for frame in (factor.market_bars, factor.market_bar_grid):
        assert frame.attrs['raw_coverage_by_key'][('BTC-USD', '60min')] == (first, last)
        assert frame.attrs['raw_coverage_by_instrument']['BTC-USD'] == (first, last)
        assert frame.attrs['raw_coverage_start_utc'] == first
        assert frame.attrs['raw_coverage_end_utc'] == last
