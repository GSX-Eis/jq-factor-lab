"""Explicit sources retain precedence and borrowed handles retain their owner."""
from calendar_test_support import seed_calendar
import duckdb
import pandas as pd
import pytest
from utils.factor_base import FactorBase
from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter


def test_borrowed_connection_does_not_bypass_dual_source(tmp_path, monkeypatch):
    monkeypatch.delenv('QUANT_DATA_BINDINGS', raising=False)
    db = tmp_path / 'input.duckdb'
    folder = tmp_path / 'parquet' / 'index_constituents'
    folder.mkdir(parents=True)
    with duckdb.connect(str(db)) as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2024-01-31' date, 'A' stock_code, 10. AS close")
        conn.execute('CREATE TABLE price_data AS SELECT * FROM factor_data')
        conn.execute('CREATE TABLE index_constituents(date DATE, stock_code VARCHAR, index_code VARCHAR)')
    pd.DataFrame({'date': pd.to_datetime(['2024-02-01']), 'stock_code': ['A'],
        'index_code': ['I']}).to_parquet(folder / 'part.parquet', index=False)
    with duckdb.connect(str(db), read_only=True) as borrowed:
        factor = FactorBase(borrowed, 'PoolProbe', '2024-02-01', '2024-02-29',
            use_dynamic_pool=True, stock_filters=[StockFilter('index_code', '==', 'I')],
            rebalance_schedule='monthly', duckdb_path=str(db), parquet_dir=str(folder.parent),
            source_policy='duckdb_then_parquet_on_empty', data_bindings={})
        try:
            assert factor.stock_pools_by_period == {'2024-02-01': ['A']}
        finally:
            factor._duckdb_adapter.close()
            if factor.dynamic_pool_manager._duckdb_adapter:
                factor.dynamic_pool_manager._duckdb_adapter.close()
        assert borrowed.execute('SELECT 1').fetchone() == (1,)


def test_connection_only_still_uses_borrowed_handle():
    with duckdb.connect() as borrowed:
        manager = DynamicStockPoolManager(borrowed, data_bindings={})
        assert manager._duckdb_conn is borrowed
        assert manager._duckdb_adapter is None


def test_borrowed_handle_does_not_hide_invalid_explicit_policy(tmp_path):
    with duckdb.connect() as borrowed:
        with pytest.raises(ValueError, match='source_policy'):
            DynamicStockPoolManager(borrowed, source_policy='unknown', data_bindings={})
        assert borrowed.execute('SELECT 1').fetchone() == (1,)
