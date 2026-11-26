"""Source-aware empty-bar amounts, independent of factor and field projection."""
import duckdb
from market_db_samples import declare_market_db
import numpy as np
import pandas as pd
import pytest

from utils.factor_base import FactorBase
from utils.pit import apply_pit_adjustment, normalize_price_money


@pytest.mark.parametrize('source', ['local', 'joinquant'])
@pytest.mark.parametrize('frequency', ['daily', '5m', '1m'])
def test_explicit_paused_zero_only_is_missing(source, frequency):
    raw = pd.DataFrame({'money': [0., 0., 10., np.nan, 0.],
                        'paused': [1., 0., 0., 1., np.nan],
                        'factor': [1.] * 5, 'volume': [0.] * 5})
    before = raw.copy(deep=True)
    expected = raw.copy(deep=True)
    expected.loc[0, 'money'] = np.nan
    result = normalize_price_money(raw, source=source, frequency=frequency)
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    pd.testing.assert_frame_equal(raw, before, check_exact=True)
    pd.testing.assert_frame_equal(normalize_price_money(result, source=source, frequency=frequency), result)


def test_minute_zero_volume_without_paused_is_not_suspension():
    raw = pd.DataFrame({'money': [0], 'volume': [0], 'factor': [1.]})
    assert normalize_price_money(raw, source='local', frequency='5m') is raw


@pytest.mark.parametrize('source,frequency', [('unknown', 'daily'), ('local', 'weekly')])
def test_unknown_normalization_contract_is_rejected(source, frequency):
    with pytest.raises(ValueError, match='source or frequency'):
        normalize_price_money(pd.DataFrame({'money': [0.]}), source=source, frequency=frequency)


@pytest.mark.parametrize('fields', [['money'], ['money', 'close'], ['money', 'volume']])
def test_local_daily_price_boundary_reads_paused_before_projection(tmp_path, fields):
    path = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(path)) as conn:
        declare_market_db(conn, 'cn_equity')
        conn.execute("""CREATE TABLE price_data_daily AS SELECT * FROM (VALUES
            ('A', DATE '2022-03-07', 1., 10., 10., 100., 0),
            ('A', DATE '2022-03-08', 2., NULL, NULL, 0., 1),
            ('A', DATE '2022-03-09', 1., 10., 0., 0., 0)
        ) p(stock_code, date, factor, close, volume, money, paused)""")
    factor = FactorBase(None, 'MoneyProbe', '2022-03-07', '2022-03-09',
                        stock_pool=['A'], duckdb_path=str(path), pit_enabled=True)
    try:
        for _ in range(2):  # Cold and warm raw adapter cache.
            result = factor._fetch_raw_data_duckdb(fields, 'price_data_daily')
            assert 'paused' not in result
            adjusted = apply_pit_adjustment(result)
            assert adjusted.money.iloc[0] == 100.
            assert pd.isna(adjusted.money.iloc[1])
            assert adjusted.money.iloc[2] == 0.
            assert pd.isna(adjusted.factor.iloc[1])
    finally:
        factor._duckdb_adapter.close()



@pytest.mark.parametrize('frequency,table', [('daily', 'price_data_daily'), ('5m', 'price_data_5m')])
def test_money_only_chunk_anchor_ignores_paused_storage_zero(tmp_path, frequency, table):
    from utils.duckdb_adapter import DuckDBAdapter
    path = tmp_path / 'anchors.duckdb'
    with duckdb.connect(str(path)) as conn:
        conn.execute(f"""CREATE TABLE {table} AS SELECT * FROM (VALUES
            ('A', DATE '2022-03-07', 2., 100., 0),
            ('A', DATE '2022-03-08', 1., 0., 1)
        ) p(stock_code, date, factor, money, paused)""")
    adapter = DuckDBAdapter(duckdb_path=str(path))
    try:
        assert adapter.fetch_price_factor_ends(['A'], '2022-03-07', '2022-03-08',
                                               fields=['money', 'factor'], frequency=frequency) == {'A': 2.}
    finally:
        adapter.close()


class _RollingMoneyMeanFactor(FactorBase):
    def _get_raw_data_for_factor(self) -> pd.DataFrame:
        return self._fetch_raw_data(['money'], table_name='price_data')

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        wide = data.pivot(index='date', columns='code', values='money').sort_index()
        return wide.rolling(2, min_periods=1).mean()


@pytest.mark.parametrize('pit_enabled', [True, False])
@pytest.mark.parametrize('data_frequency', ['daily', '1d'])
def test_p101_borrowed_connection_matches_adapter_on_paused_money(tmp_path, pit_enabled, data_frequency):
    path = tmp_path / 'p101.duckdb'
    with duckdb.connect(str(path)) as conn:
        declare_market_db(conn, 'cn_equity')
        conn.execute("""CREATE TABLE price_data_daily AS SELECT * FROM (VALUES
            ('A', DATE '2022-03-07', 1., 10., 10., 100., 0),
            ('A', DATE '2022-03-08', 1., 10., 0., 0., 1),
            ('A', DATE '2022-03-09', 1., 10., 20., 200., 0)
        ) p(stock_code, date, factor, close, volume, money, paused)""")

    mode_kwargs = {'analysis_mode': 'cross_sectional'} if data_frequency == '1d' else {}
    adapter_factor = _RollingMoneyMeanFactor(
        None, 'P101Adapter', '2022-03-07', '2022-03-09',
        stock_pool=['A'], duckdb_path=str(path), data_frequency=data_frequency,
        pit_enabled=pit_enabled, **mode_kwargs,
    )
    with duckdb.connect(str(path), read_only=True) as borrowed:
        borrowed_factor = _RollingMoneyMeanFactor(
            borrowed, 'P101Borrowed', '2022-03-07', '2022-03-09',
            stock_pool=['A'], data_frequency=data_frequency,
            pit_enabled=pit_enabled, **mode_kwargs,
        )
        try:
            adapter_values = adapter_factor.get_factor_values()
            borrowed_values = borrowed_factor.get_factor_values()
            assert 'paused' not in adapter_factor.last_raw_data.columns
            assert 'paused' not in borrowed_factor.last_raw_data.columns
            expected = [100.0, 100.0, 200.0]
            np.testing.assert_allclose(adapter_values['A'].to_numpy(), expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(borrowed_values['A'].to_numpy(), expected, rtol=0, atol=1e-12)
            pd.testing.assert_frame_equal(borrowed_values, adapter_values, check_exact=True)
            assert borrowed.execute('SELECT 1').fetchone() == (1,)
        finally:
            adapter_factor.close()


@pytest.mark.parametrize('pit_enabled', [True, False])
def test_borrowed_connection_preserves_unpaused_zero_and_handles_missing_paused_column(tmp_path, pit_enabled):
    with duckdb.connect(':memory:') as conn:
        conn.execute("""CREATE TABLE price_data_daily AS SELECT * FROM (VALUES
            ('A', DATE '2022-03-07', 1., 10., 100., 0),
            ('A', DATE '2022-03-08', 1., 10., 0., 0),
            ('A', DATE '2022-03-09', 1., NULL, 0., 1)
        ) p(stock_code, date, factor, close, money, paused)""")
        factor = FactorBase(
            conn, 'ZeroMoneyProbe', '2022-03-07', '2022-03-09',
            stock_pool=['A'], pit_enabled=pit_enabled,
        )
        raw_money = factor._fetch_raw_data(['money'], table_name='price_data')
        assert 'paused' not in raw_money.columns
        np.testing.assert_allclose(raw_money['money'].to_numpy(dtype=float), [100., 0., np.nan], equal_nan=True)
        money_panel = factor.get_price_data('money')
        np.testing.assert_allclose(money_panel['A'].to_numpy(dtype=float), [100., 0., np.nan], equal_nan=True)

        explicit = factor._fetch_raw_data(['money', 'paused'], table_name='price_data')
        assert 'paused' in explicit.columns
        np.testing.assert_allclose(explicit['money'].to_numpy(dtype=float), [100., 0., np.nan], equal_nan=True)

    # Minimal table without a paused column must succeed on both borrowed connection and DuckDB file adapter.
    minimal_path = tmp_path / 'minimal.duckdb'
    with duckdb.connect(str(minimal_path)) as conn:
        declare_market_db(conn, 'cn_equity')
        conn.execute("""CREATE TABLE price_data_daily AS SELECT * FROM (VALUES
            ('A', DATE '2022-03-07', 1., 10., 100.),
            ('A', DATE '2022-03-08', 1., 10., 0.),
            ('A', DATE '2022-03-09', 1., 10., 200.)
        ) p(stock_code, date, factor, close, money)""")

    adapter_probe = FactorBase(
        None, 'NoPausedAdapter', '2022-03-07', '2022-03-09',
        stock_pool=['A'], duckdb_path=str(minimal_path), pit_enabled=pit_enabled,
    )
    with duckdb.connect(str(minimal_path), read_only=True) as borrowed:
        borrowed_probe = FactorBase(
            borrowed, 'NoPausedBorrowed', '2022-03-07', '2022-03-09',
            stock_pool=['A'], pit_enabled=pit_enabled,
        )
        try:
            adapter_raw = adapter_probe._fetch_raw_data(['money'], table_name='price_data')
            borrowed_raw = borrowed_probe._fetch_raw_data(['money'], table_name='price_data')
            pd.testing.assert_frame_equal(borrowed_raw, adapter_raw, check_exact=True)
            np.testing.assert_allclose(borrowed_raw['money'].to_numpy(dtype=float), [100., 0., 200.])
        finally:
            adapter_probe.close()


def test_declarative_csv_signal_bars_normalize_paused_money(tmp_path):
    from slots import build_slot_from_config
    from utils.slots import ExecutionProfile, ProviderConfig, SlotConfig
    from utils.time_axis import Frequency

    bars = pd.DataFrame({
        'date': ['2026-01-05 15:00:00', '2026-01-06 15:00:00', '2026-01-07 15:00:00'],
        'stock_code': ['000001.XSHE'] * 3,
        'open': [10., 10., 10.],
        'high': [10., 10., 10.],
        'low': [10., 10., 10.],
        'close': [10., 10., 10.],
        'volume': [100., 0., 0.],
        'money': [100., 0., 0.],
        'paused': [0, 1, 0],
        'factor': [1., 1., 1.],
    })
    bars.to_csv(tmp_path / 'bars.csv', index=False)
    cfg = SlotConfig(
        schema_version=1,
        slot_id='csv_ashare_probe',
        market_domain='cn_equity',
        calendar=ProviderConfig('pmc', {'calendar_id': 'XSHG', 'start': '2026-01-05', 'end': '2026-01-07'}),
        bar_feed=ProviderConfig('csv_files', {
            'paths': ['bars.csv'],
            'frequency': '1d',
            'timestamp_anchor': 'bar_end',
            'source_timezone': 'Asia/Shanghai',
            'volume_mode': 'interval',
            'source': 'csv_probe',
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
                '000001.XSHE': {'asset_class': 'equity', 'currency': 'CNY', 'lot_size': 100, 'contract_multiplier': 1.0},
            },
            'universes': {'all': ('000001.XSHE',)},
        }),
        aux_features=ProviderConfig('none', {}),
        microstructure=ProviderConfig('open', {}),
        execution_profile=ExecutionProfile('close', 'close', 0, 0, 'decimal', 'endpoints_only', 'cross_session', False),
    )
    slot = build_slot_from_config(cfg, base_dir=tmp_path)
    start = pd.Timestamp('2026-01-05 07:00Z')
    end = pd.Timestamp('2026-01-07 07:00Z')
    freq = Frequency.parse('1d')
    signal = slot.bar_feed.read_signal_bars(['000001.XSHE'], freq, start_utc=start, end_utc=end, as_of_utc=end)
    execution = slot.bar_feed.read_execution_bars(['000001.XSHE'], freq, start_utc=start, end_utc=end)
    assert 'paused' not in signal.columns
    assert 'paused' not in execution.columns
    np.testing.assert_allclose(signal['money'].to_numpy(dtype=float), [100., np.nan, 0.], equal_nan=True)
    np.testing.assert_allclose(execution['money'].to_numpy(dtype=float), [100., 0., 0.])
