"""Schema failures must remain retryable and cannot masquerade as empty pools."""
from calendar_test_support import seed_calendar
from datetime import datetime

import duckdb
import pytest

from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter


@pytest.mark.parametrize('kind', ['fundamental', 'price'])
def test_missing_table_error_does_not_poison_cache_and_can_retry(kind):
    with duckdb.connect() as conn:
        manager = DynamicStockPoolManager(conn, fundamental_table='fund', price_table='prices', data_bindings={})
        table, getter, cache = (
            ('fund', manager._get_available_columns, '_columns_cache') if kind == 'fundamental'
            else ('prices', manager._get_price_columns, '_price_columns_cache')
        )
        with pytest.raises(RuntimeError, match=table) as error:
            getter()
        assert isinstance(error.value.__cause__, duckdb.CatalogException)
        assert getattr(manager, cache) is None
        conn.execute(f'CREATE TABLE {table} (date DATE, stock_code VARCHAR, value DOUBLE)')
        assert getter() == ['date', 'stock_code', 'value']


def test_closed_connection_error_retains_cause_and_does_not_cache():
    conn = duckdb.connect()
    manager = DynamicStockPoolManager(conn, data_bindings={})
    conn.close()
    with pytest.raises(RuntimeError, match='factor_data') as error:
        manager._get_available_columns()
    assert isinstance(error.value.__cause__, duckdb.ConnectionException)
    assert manager._columns_cache is None


@pytest.mark.parametrize('entry', ['monthly', 'weekly', 'specific'])
def test_required_schema_failure_propagates_and_same_pool_request_can_retry(entry, monkeypatch):
    with duckdb.connect() as conn:
        manager = DynamicStockPoolManager(conn, fundamental_table='fund', price_table='prices', data_bindings={})
        seed_calendar(conn)
        monkeypatch.setattr(manager, '_get_previous_data_date', lambda _: '2024-01-31')
        monkeypatch.setattr(manager, '_save_filter_log', lambda _: None)
        filters = [StockFilter('market_cap', '>', 5)]

        def run():
            if entry == 'monthly':
                return manager.get_stock_pools_for_period('2024-01-01','2024-01-31',filters,
                    schedule={'unit':'month','session':-1,'include_previous':False})['2024-01-31']
            if entry == 'weekly':
                return manager._select_stock_pool(datetime(2024, 1, 31), filters)
            return manager._select_stock_pool(datetime(2024, 1, 31), filters)

        with pytest.raises(RuntimeError, match='fund'):
            run()
        assert manager._pool_cache == {}
        conn.execute("CREATE TABLE fund AS SELECT DATE '2024-01-31' date, 'A' stock_code, 10. market_cap")
        # No price table is needed for this fundamental-only request.
        assert run() == ['A']
        assert manager._price_columns_cache is None


@pytest.mark.parametrize('sort_only', [False, True])
def test_required_price_schema_failure_is_not_skipped(sort_only, monkeypatch):
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE fund AS SELECT DATE '2024-01-31' date, 'A' stock_code, 10. market_cap")
        manager = DynamicStockPoolManager(conn, fundamental_table='fund', price_table='prices', data_bindings={})
        seed_calendar(conn)
        monkeypatch.setattr(manager, '_save_filter_log', lambda _: None)
        filters = [] if sort_only else [StockFilter('close', '>', 5)]
        sort_by = 'close' if sort_only else None
        with pytest.raises(RuntimeError, match='prices'):
            manager._select_stock_pool(datetime(2024, 1, 31), filters, sort_by=sort_by)
        assert manager._price_columns_cache is None
        assert manager._pool_cache == {}
        conn.execute("CREATE TABLE prices AS SELECT DATE '2024-01-31' date, 'A' stock_code, 10. AS close, 10. AS open")
        assert manager._select_stock_pool(datetime(2024, 1, 31), filters, sort_by=sort_by) == ['A']


def test_validate_filters_can_retry_after_missing_price_table_is_fixed():
    with duckdb.connect() as conn:
        conn.execute('CREATE TABLE fund (date DATE, stock_code VARCHAR, market_cap DOUBLE)')
        manager = DynamicStockPoolManager(conn, fundamental_table='fund', price_table='prices', data_bindings={})
        assert manager.validate_filters([StockFilter('market_cap', '>', 5)])
        filters = [StockFilter('close', '>', 5)]
        assert not manager.validate_filters(filters)
        assert manager._price_columns_cache is None
        # Validation now checks the mapped field actually used by the query.
        conn.execute('CREATE TABLE prices (date DATE, stock_code VARCHAR, close DOUBLE, open DOUBLE)')
        assert manager.validate_filters(filters)


def test_successful_schema_cache_is_reused(monkeypatch):
    with duckdb.connect() as conn:
        conn.execute('CREATE TABLE fund (date DATE, stock_code VARCHAR)')
        manager = DynamicStockPoolManager(conn, fundamental_table='fund', data_bindings={})
        assert manager._get_available_columns() == ['date', 'stock_code']
        def unexpected_query(_sql):
            pytest.fail('successful schema cache should avoid repeated DESCRIBE')
        monkeypatch.setattr(manager, '_execute_sql', unexpected_query)
        assert manager._get_available_columns() == ['date', 'stock_code']
