"""Committed imports invalidate dependent caches; failed imports change nothing."""
from calendar_test_support import seed_calendar
from datetime import datetime
import json
import subprocess
import sys

import duckdb
import pandas as pd
import pytest

from data_pipeline.loaders._transactions import atomic_import, import_transaction
from utils.data_revision import (
    advance_revision, connection_revision, database_revision,
    parquet_publication, parquet_revision,
)
from utils.duckdb_adapter import DuckDBAdapter
from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter


@atomic_import
def replace_member(conn, member, fail=False):
    conn.execute('UPDATE index_constituents SET stock_code=?', [member])
    if fail:
        raise ValueError('invalid import')


@pytest.fixture
def pool(monkeypatch):
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE index_constituents AS SELECT DATE '2024-01-31' date, 'A' stock_code, 'I' index_code")
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2024-01-31' date, code stock_code, 10. market_cap FROM (VALUES ('A'), ('B')) t(code)")
        manager = DynamicStockPoolManager(conn, data_bindings={})
        seed_calendar(conn)
        monkeypatch.setattr(manager, '_get_previous_data_date', lambda _: '2024-01-31')
        monkeypatch.setattr(manager, '_save_filter_log', lambda _: None)
        yield conn, manager


@pytest.mark.parametrize('entry', ['monthly', 'weekly', 'specific'])
def test_committed_import_invalidates_final_pool_and_membership(pool, entry):
    conn, manager = pool
    filters = [StockFilter('index_code', '==', 'I')]
    def read():
        if entry == 'monthly':
            return manager.get_stock_pools_for_period('2024-01-01','2024-01-31',filters,
                schedule={'unit':'month','session':-1,'include_previous':False})['2024-01-31']
        if entry == 'weekly':
            return manager.get_stock_pools_for_period('2024-01-31','2024-01-31',filters,
                rebalance_schedule='weekly')['2024-01-31']
        return manager._select_stock_pool(datetime(2024, 1, 31), filters)
    assert read() == ['A']
    assert manager._pool_cache
    before = connection_revision(conn)
    replace_member(conn, 'B')
    assert connection_revision(conn) != before
    assert read() == ['B']
    assert read() == ['B']


def test_failed_import_keeps_data_generation_and_successful_cache(pool):
    conn, manager = pool
    replace_member(conn, 'A')
    assert manager._get_index_constituents('2024-01-31', 'I') == {'A'}
    before = connection_revision(conn)
    cached = dict(manager._index_constituents_cache)
    with pytest.raises(ValueError, match='invalid import'):
        replace_member(conn, 'B', fail=True)
    assert connection_revision(conn) == before
    assert manager._get_index_constituents('2024-01-31', 'I') == {'A'}
    assert manager._index_constituents_cache == cached


def test_schema_refresh_and_borrowed_connection_not_closed(pool):
    conn, manager = pool
    assert manager._get_available_columns() == ['date', 'stock_code', 'market_cap']
    with import_transaction(conn):
        conn.execute('ALTER TABLE factor_data ADD COLUMN earnings DOUBLE')
    assert manager._get_available_columns()[-1] == 'earnings'
    manager.clear_cache()
    assert conn.execute('SELECT 1').fetchone() == (1,)


def test_revision_failure_rolls_back_data_too(pool, monkeypatch):
    conn, _ = pool
    import data_pipeline.loaders._transactions as transactions
    before = connection_revision(conn)
    def fail(_):
        raise RuntimeError('cannot publish generation')
    monkeypatch.setattr(transactions, 'advance_revision', fail)
    with pytest.raises(RuntimeError, match='generation'):
        replace_member(conn, 'B')
    assert connection_revision(conn) == before
    assert conn.execute('SELECT stock_code FROM index_constituents').fetchone() == ('A',)


def test_change_during_calculation_raises_and_discards_cache(pool, monkeypatch):
    conn, manager = pool
    original = manager._read_index_constituents
    def read_then_import(*args, **kwargs):
        result = original(*args, **kwargs)
        replace_member(conn, 'B')
        return result
    monkeypatch.setattr(manager, '_read_index_constituents', read_then_import)
    with pytest.raises(RuntimeError, match='changed during calculation'):
        manager._get_index_constituents('2024-01-31', 'I')
    assert not manager._index_constituents_cache
    monkeypatch.setattr(manager, '_read_index_constituents', original)
    assert manager._get_index_constituents('2024-01-31', 'I') == {'B'}


def test_adapter_refreshes_same_instance_after_import(tmp_path):
    path = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-01-31' date, 'A' stock_code, 10. AS close")
        adapter = DuckDBAdapter(duckdb_path=path, data_bindings={})
        # Borrow the existing writer only for this same-process transaction test.
        adapter._conn = conn
        def read():
            return adapter.fetch_price_data(['A'], '2024-01-31', '2024-01-31', ['close']).close.iloc[0]
        assert read() == 10
        with import_transaction(conn):
            conn.execute('UPDATE price_data_daily SET close=20')
        assert read() == 20
        adapter._conn = None


def test_committed_generation_visible_across_processes(tmp_path):
    path = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(path)) as conn:
        advance_revision(conn)
    before = database_revision(path)
    subprocess.run([sys.executable, '-c',
        'import duckdb,sys; from data_pipeline.loaders._transactions import import_transaction; '
        'c=duckdb.connect(sys.argv[1]);\nwith import_transaction(c):\n c.execute("CREATE TABLE prices(i INT)")\nc.close()',
        str(path)], check=True)
    assert database_revision(path) != before


def test_parquet_export_refreshes_live_reader_and_rejects_failed_export(tmp_path):
    root = tmp_path / 'lake'
    folder = root / 'price_data_daily' / 'year=2024' / 'month=1'
    folder.mkdir(parents=True)
    def write(value):
        pd.DataFrame({'date': pd.to_datetime(['2024-01-31']), 'stock_code': ['A'], 'close': [value]}).to_parquet(folder / 'data.parquet')
    with parquet_publication(root, 'price_data_daily'):
        write(10)
    adapter = DuckDBAdapter(parquet_dir=root, data_bindings={})
    try:
        def read():
            return adapter.fetch_price_data(['A'], '2024-01-31', '2024-01-31', ['close']).close.iloc[0]
        assert read() == 10
        with parquet_publication(root, 'price_data_daily'):
            write(20)
            with pytest.raises(RuntimeError, match='incomplete'):
                read()
        assert read() == 20
        with pytest.raises(ValueError):
            with parquet_publication(root, 'price_data_daily'):
                write(30)
                raise ValueError('export failed')
        with pytest.raises(RuntimeError, match='incomplete'):
            read()
        with parquet_publication(root, 'price_data_daily'):
            write(40)
        assert read() == 40
        (root / '.cache-revision.json').unlink()
        assert read() == 40
        write(50)
        assert read() == 50  # unmanaged sources do not reuse old application results
    finally:
        adapter.close()


def test_write_store_generation_and_fixed_binding(tmp_path):
    from utils.data_store import migrate_table, write_store
    source, store = tmp_path / 'source.duckdb', tmp_path / 'input.duckdb'
    with duckdb.connect(str(source)) as conn:
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-01-31' date, 'A' stock_code, 10. AS close")
    ref = migrate_table(source, store, 'price_data_daily', role='input', execute=True)['ref']
    before = database_revision(store)
    with pytest.raises(ValueError):
        with write_store(store) as conn:
            raise ValueError('rollback')
    assert database_revision(store) == before
    with duckdb.connect(str(source)) as conn:
        conn.execute('UPDATE price_data_daily SET close=20')
    migrate_table(source, store, 'price_data_daily', role='input', execute=True)
    assert database_revision(store) != before
    adapter = DuckDBAdapter(data_bindings={'input_db': str(store), 'inputs': {'price_data_daily': ref}})
    try:
        assert adapter.fetch_price_data(['A'], '2024-01-31', '2024-01-31', ['close']).close.iloc[0] == 10
    finally:
        adapter.close()


def test_actual_index_import_updates_generation_and_failed_replace_keeps_it(tmp_path):
    from data_pipeline.loaders.import_index_constituents_duckdb import DuckDBIndexConstituentsImporter
    path = tmp_path / 'input.csv'
    path.write_text('date,stock_code,index_code\n2024-01-31,A,I\n')
    importer = DuckDBIndexConstituentsImporter(str(tmp_path / 'input.duckdb'))
    try:
        importer.import_csv(str(path))
        manager = DynamicStockPoolManager(importer.conn, data_bindings={})
        assert manager._get_index_constituents('2024-01-31', 'I') == {'A'}
        before = connection_revision(importer.conn)
        path.write_text('date,stock_code,index_code\n2024-01-31,B,I\n')
        importer.import_csv(str(path), replace=True)
        assert connection_revision(importer.conn) != before
        assert manager._get_index_constituents('2024-01-31', 'I') == {'B'}
        before = connection_revision(importer.conn)
        path.write_text('date,stock_code,index_code\nbad,B,I\n')
        with pytest.raises(ValueError):
            importer.import_csv(str(path), replace=True)
        assert connection_revision(importer.conn) == before
        assert manager._get_index_constituents('2024-01-31', 'I') == {'B'}
    finally:
        importer.close()


def test_export_replaces_obsolete_partitions(tmp_path):
    from data_pipeline.loaders.price_data_to_duckdb import DuckDBPriceImporter
    importer = DuckDBPriceImporter(str(tmp_path / 'prices.duckdb'), export_parquet=True)
    importer.parquet_dir = tmp_path / 'lake'
    conn = importer.connect()
    reader = None
    try:
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-01-31' date, 'A' stock_code, 10. AS close")
        importer.export_to_parquet('price_data_daily', ['year', 'month'])
        reader = DuckDBAdapter(parquet_dir=importer.parquet_dir, data_bindings={})
        def read():
            return reader.fetch_price_data(['A'], '2024-01-01', '2024-03-31', ['close'])
        assert read().date.dt.month.tolist() == [1]
        with import_transaction(conn):
            conn.execute("UPDATE price_data_daily SET date=DATE '2024-03-01', close=20")
        importer.export_to_parquet('price_data_daily', ['year', 'month'])
        assert read().date.dt.month.tolist() == [3]
        assert read().close.tolist() == [20]
    finally:
        if reader is not None:
            reader.close()
        importer.close()


def test_harness_json_publication_keeps_staging_and_live_generations_separate(tmp_path):
    from utils.data_store import main as store_main
    stage, live = tmp_path / 'stage.duckdb', tmp_path / 'live.duckdb'
    with duckdb.connect(str(stage)) as conn, import_transaction(conn):
        conn.execute('CREATE TABLE prices(id INTEGER PRIMARY KEY, close DOUBLE)')
        conn.execute('INSERT INTO prices VALUES (1, 10)')
    stage_revision = database_revision(stage)
    assert not live.exists()
    request, response = tmp_path / 'request.json', tmp_path / 'response.json'
    request.write_text(json.dumps({'action': 'publish_input', 'path': str(live),
        'staging': str(stage), 'tables': ['prices'], 'parents': {},
        'metadata': {'source': 'synthetic', 'source_hash': 'one', 'loader_hash': 'test'}}))
    assert store_main(['--request', str(request), '--response', str(response)]) == 0
    assert json.loads(response.read_text())['ok']
    assert database_revision(stage) == stage_revision
    before = database_revision(live)
    with duckdb.connect(str(stage)) as conn, import_transaction(conn):
        conn.execute('UPDATE prices SET close=20')
    assert database_revision(live) == before
    # Reusing the same batch identity for different data must fail publication.
    with pytest.raises(ValueError):
        store_main(['--request', str(request), '--response', str(response)])
    assert database_revision(live) == before
    assert not response.exists()


def test_pool_reuses_manager_after_owned_connection_released_for_import(tmp_path):
    path = tmp_path / 'input.duckdb'
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE index_constituents AS SELECT DATE '2024-01-31' date, 'A' stock_code, 'I' index_code")
    manager = DynamicStockPoolManager(None, duckdb_path=str(path), data_bindings={})
    try:
        assert manager._get_index_constituents('2024-01-31', 'I') == {'A'}
        manager._duckdb_adapter.close()
        with duckdb.connect(str(path)) as conn:
            replace_member(conn, 'B')
        assert manager._get_index_constituents('2024-01-31', 'I') == {'B'}
        assert manager._execute_sql('SELECT stock_code FROM index_constituents').iloc[0, 0] == 'B'
    finally:
        manager._duckdb_adapter.close()
