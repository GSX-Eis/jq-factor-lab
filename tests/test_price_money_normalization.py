"""Source-aware empty-bar amounts, independent of factor and field projection."""
import duckdb
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
