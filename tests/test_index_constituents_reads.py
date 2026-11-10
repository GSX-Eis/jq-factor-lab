"""Index snapshots must be exact, configured and fail explicitly across local sources."""
import json

from calendar_test_support import seed_calendar
import duckdb
import pandas as pd
import pytest

from utils.data_store import migrate_table
from utils.duckdb_adapter import DuckDBAdapter
from utils.dynamic_stock_pool import DynamicStockPoolManager
from utils.data_access_config import load_data_access_config


def frame():
    return pd.DataFrame({
        'date': ['2026-01-05', '2026-01-05', '2026-01-05', '2026-01-06', '2026-01-05'],
        'stock_code': ['B', 'A', 'A', 'NEXT', 'OTHER'],
        'index_code': ["O'X"] * 4 + ['different'],
    })


@pytest.fixture(params=['duckdb', 'bound', 'parquet_flat', 'parquet_partitioned'])
def source(tmp_path, request):
    data = frame()
    db = tmp_path / 'source.duckdb'
    with duckdb.connect(str(db)) as conn:
        conn.execute('CREATE TABLE index_constituents AS SELECT * FROM data')
    if request.param == 'bound':
        store = tmp_path / 'input.duckdb'
        ref = migrate_table(db, store, 'index_constituents', role='input', execute=True)['ref']
        args = {'data_bindings': {'input_db': str(store), 'inputs': {'index_constituents': ref}}}
    elif request.param.startswith('parquet'):
        root = tmp_path / 'parquet'
        folder = root / 'index_constituents'
        if request.param == 'parquet_partitioned': folder = folder / 'year=2026' / 'month=1'
        folder.mkdir(parents=True)
        data.to_parquet(folder / 'data.parquet', index=False)
        args = {'parquet_dir': str(root), 'data_bindings': {}}
    else:
        args = {'duckdb_path': str(db), 'data_bindings': {}}
    reader = DuckDBAdapter(**args)
    yield reader, args
    reader.close()


def test_exact_snapshot_literal_parameters_sorted_and_unique(source):
    reader, _ = source
    assert reader.fetch_index_constituents("O'X", '2026-01-05') == ['A', 'B']
    assert reader.fetch_index_constituents("O'X", '2026-01-07') == []
    assert reader.fetch_index_constituents("' OR TRUE --", '2026-01-05') == []
    assert reader.fetch_index_constituents('different', '2026-01-05') == ['OTHER']


@pytest.mark.parametrize('index,date', [(None, '2026-01-05'), ('', '2026-01-05'), ('  ', '2026-01-05'),
    ('I', None), ('I', 'bad'), ('I', '2026-02-30'), ('I', "2026-01-05' OR TRUE --")])
def test_bad_requests_raise(source, index, date):
    reader, _ = source
    with pytest.raises(ValueError): reader.fetch_index_constituents(index, date)


def test_pool_uses_same_reader_and_empty_never_disables_index_filter(source):
    _, args = source
    manager = DynamicStockPoolManager(None, **args)
    try:
        assert manager._filter_by_index(['B', 'A', 'OTHER'], '2026-01-05', "O'X") == ['B', 'A']
        assert manager._filter_by_index(['B', 'A', 'OTHER'], '2026-01-07', "O'X") == []
        stocks = manager._get_index_constituents('2026-01-05', "O'X")
        stocks.clear()
        assert manager._get_index_constituents('2026-01-05', "O'X") == {'A', 'B'}
    finally:
        manager._duckdb_adapter.close()


@pytest.mark.parametrize('mode', ['duckdb', 'bound', 'parquet_missing', 'parquet_empty', 'parquet_broken'])
def test_bad_sources_are_errors_not_empty_snapshots(tmp_path, mode):
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)) as conn: conn.execute('CREATE TABLE marker AS SELECT 1 AS value')
    args = {'duckdb_path': str(db), 'data_bindings': {}}
    if mode == 'bound':
        store = tmp_path / 'bound.duckdb'
        ref = migrate_table(db, store, 'marker', role='input', execute=True)['ref']
        args = {'data_bindings': {'input_db': str(store), 'inputs': {'marker': ref}}}
    if mode.startswith('parquet'):
        args = {'parquet_dir': str(tmp_path), 'data_bindings': {}}
        if mode != 'parquet_missing':
            folder = tmp_path / 'index_constituents'; folder.mkdir()
            if mode == 'parquet_broken': (folder / 'data.parquet').write_bytes(b'broken')
    reader = DuckDBAdapter(**args)
    manager = DynamicStockPoolManager(None, **args)
    try:
        with pytest.raises((FileNotFoundError, duckdb.Error)): reader.fetch_index_constituents('I', '2026-01-05')
        with pytest.raises((FileNotFoundError, duckdb.Error)): manager._filter_by_index(['A'], '2026-01-05', 'I')
    finally:
        reader.close(); manager._duckdb_adapter.close()


@pytest.mark.parametrize('mode', ['duckdb', 'parquet', 'borrowed'])
def test_configurable_table_keys_and_directory(tmp_path, mode):
    config = load_data_access_config().as_dict()
    config['index_constituents'] = {'table': 'main.members', 'index_column': 'benchmark"id'}
    config['tables']['main.members'] = {'date_column': 'snapshot"date', 'stock_column': 'symbol',
        'parquet_subdir': 'membership', 'default_fields': None}
    path = tmp_path / 'config.json'; path.write_text(json.dumps(config))
    data = frame().rename(columns={'date': 'snapshot"date', 'stock_code': 'symbol', 'index_code': 'benchmark"id'})
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)) as conn: conn.execute('CREATE TABLE members AS SELECT * FROM data')
    conn = None
    if mode == 'parquet':
        folder = tmp_path / 'parquet' / 'membership'; folder.mkdir(parents=True)
        data.to_parquet(folder / 'data.parquet', index=False)
        args = {'parquet_dir': str(tmp_path / 'parquet')}
    elif mode == 'borrowed':
        conn = duckdb.connect(str(db)); args = {}
    else: args = {'duckdb_path': str(db)}
    manager = DynamicStockPoolManager(conn, **args, data_bindings={}, data_config_path=str(path))
    try:
        assert manager._get_index_constituents('2026-01-05', "O'X") == {'A', 'B'}
        assert manager._filter_by_index(['A'], '2026-01-07', "O'X") == []
        if mode != 'borrowed':
            assert manager._duckdb_adapter.fetch_index_constituents("O'X", '2026-01-05') == ['A', 'B']
    finally:
        if manager._duckdb_adapter: manager._duckdb_adapter.close()
        if conn: conn.close()


def test_explicit_manager_table_override_and_errors_do_not_poison_cache(tmp_path):
    with duckdb.connect() as conn:
        manager = DynamicStockPoolManager(conn, index_constituents_table='main.custom_members', data_bindings={})
        with pytest.raises(duckdb.Error): manager._get_index_constituents('2026-01-05', "O'X")
        data = frame()
        conn.execute('CREATE TABLE custom_members AS SELECT * FROM data')
        assert manager._get_index_constituents('2026-01-05', "O'X") == {'A', 'B'}
        assert manager._get_index_constituents('2026-01-07', "O'X") == set()
        conn.execute('INSERT INTO custom_members VALUES (?, ?, ?)', ['2026-01-07', 'NEW', "O'X"])
        assert manager._get_index_constituents('2026-01-07', "O'X") == {'NEW'}


@pytest.mark.parametrize('failure', ['missing_table', 'missing_column', 'success_empty', 'partial'])
def test_dual_source_fallback_only_after_successful_zero_rows(tmp_path, failure):
    data = frame()
    folder = tmp_path / 'parquet' / 'index_constituents'; folder.mkdir(parents=True)
    data.to_parquet(folder / 'data.parquet', index=False)
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)) as conn:
        if failure == 'missing_column': conn.execute('CREATE TABLE index_constituents AS SELECT date, stock_code FROM data')
        if failure == 'success_empty': conn.execute('CREATE TABLE index_constituents AS SELECT * FROM data WHERE FALSE')
        if failure == 'partial': conn.execute("CREATE TABLE index_constituents AS SELECT * FROM data WHERE stock_code='B'")
    reader = DuckDBAdapter(duckdb_path=str(db), parquet_dir=str(tmp_path / 'parquet'), data_bindings={},
        source_policy='duckdb_then_parquet_on_empty')
    try:
        if failure.startswith('missing'):
            with pytest.raises(duckdb.Error): reader.fetch_index_constituents("O'X", '2026-01-05')
        else:
            assert reader.fetch_index_constituents("O'X", '2026-01-05') == (['B'] if failure == 'partial' else ['A', 'B'])
    finally: reader.close()


@pytest.mark.parametrize('failure', ['table', 'key', 'typo'])
def test_invalid_index_config_is_rejected(tmp_path, failure):
    config = load_data_access_config().as_dict()
    if failure == 'table': config['index_constituents'] = {'table': 'absent', 'index_column': 'index_code'}
    if failure == 'key': config['index_constituents'] = {'table': 'factor_data', 'index_column': 'date'}
    if failure == 'typo': config['index_constituents'] = {'table': 'factor_data', 'index_field': 'index_code'}
    path = tmp_path / 'config.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError): load_data_access_config(path)


def test_index_config_optional_for_panel_only_users(tmp_path):
    config = load_data_access_config().as_dict(); config.pop('index_constituents', None)
    path = tmp_path / 'config.json'; path.write_text(json.dumps(config))
    reader = DuckDBAdapter(parquet_dir=str(tmp_path), data_bindings={}, data_config_path=str(path))
    try:
        assert reader.fetch_price_data([], '2026-01-05', '2026-01-05', ['close']).empty
        with pytest.raises(ValueError, match='index_constituents'):
            reader.fetch_index_constituents('I', '2026-01-05')
    finally: reader.close()


@pytest.mark.parametrize('failure', ['missing_index', 'invalid_index_filter', 'generation'])
def test_factor_initialization_does_not_turn_pool_errors_into_empty_results(failure, monkeypatch):
    from utils.factor_base import FactorBase
    from utils.dynamic_stock_pool import StockFilter
    with duckdb.connect() as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2024-01-31' AS date, 'A' AS stock_code, 10 AS close")
        conn.execute('CREATE TABLE price_data_daily AS SELECT * FROM factor_data')
        filters = [StockFilter('index_code', '!=' if failure == 'invalid_index_filter' else '==', 'I')]
        if failure == 'generation':
            conn.execute("CREATE TABLE index_constituents AS SELECT DATE '2024-01-31' AS date, 'A' AS stock_code, 'I' AS index_code")
            def fail(*args, **kwargs):
                raise RuntimeError('pool generation failed')
            monkeypatch.setattr(DynamicStockPoolManager, 'get_stock_pools_for_period', fail)
        with pytest.raises((ValueError, RuntimeError, duckdb.Error)):
            FactorBase(conn, 'StrictPool', '2024-02-01', '2024-02-29',
                       use_dynamic_pool=True, stock_filters=filters,
                       price_table='price_data_daily', data_bindings={})
        assert conn.execute('SELECT 7').fetchone() == (7,)


def test_failed_factor_construction_closes_owned_readers(tmp_path, monkeypatch):
    from utils.factor_base import FactorBase
    from utils.dynamic_stock_pool import StockFilter
    path = tmp_path / 'input.duckdb'
    with duckdb.connect(str(path)) as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2024-01-31' AS date, 'A' AS stock_code, 10 AS close")
        conn.execute('CREATE TABLE price_data_daily AS SELECT * FROM factor_data')
    readers = []
    original = DuckDBAdapter.__init__
    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        readers.append(self)
    monkeypatch.setattr(DuckDBAdapter, '__init__', capture)
    with pytest.raises(ValueError):
        FactorBase(None, 'StrictPool', '2024-02-01', '2024-02-29', use_dynamic_pool=True,
                   stock_filters=[StockFilter('index_code', '==', 'I')], price_table='price_data_daily',
                   duckdb_path=str(path), data_bindings={})
    assert len(readers) == 2
    assert all(reader._conn is None for reader in readers)
    # A leaked read-only reader would prevent opening the file in write mode.
    with duckdb.connect(str(path)) as conn:
        conn.execute('CREATE TABLE marker AS SELECT 1 AS value')


def test_factor_initialization_keeps_valid_empty_index_snapshot(monkeypatch):
    from utils.factor_base import FactorBase
    from utils.dynamic_stock_pool import StockFilter
    with duckdb.connect() as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2024-01-31' AS date, 'A' AS stock_code, 10 AS close")
        conn.execute('CREATE TABLE price_data_daily AS SELECT * FROM factor_data')
        conn.execute('CREATE TABLE index_constituents(date DATE, stock_code VARCHAR, index_code VARCHAR)')
        monkeypatch.setattr(DynamicStockPoolManager, '_save_filter_log', lambda *args: None)
        factor = FactorBase(conn, 'EmptyPool', '2024-02-01', '2024-02-29',
                           use_dynamic_pool=True, stock_filters=[StockFilter('index_code', '==', 'I')],
                           price_table='price_data_daily', data_bindings={})
        assert factor.stock_pools_by_period == {'2024-02-01': []}
        assert factor.all_stocks == []


@pytest.mark.parametrize('entry', ['monthly', 'weekly', 'specific_date'])
@pytest.mark.parametrize('snapshot', ['valid', 'empty', 'missing'])
def test_public_pool_entries_preserve_index_constraint_and_errors(entry, snapshot, monkeypatch):
    from datetime import datetime
    from utils.dynamic_stock_pool import StockFilter
    with duckdb.connect() as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE factor_data AS SELECT '2026-01-05' AS date, stock_code, 1 AS value "
                     "FROM (VALUES ('A'), ('B'), ('OTHER')) AS t(stock_code)")
        conn.execute("INSERT INTO factor_data SELECT '2026-01-30', stock_code, value FROM factor_data")
        if snapshot != 'missing':
            conn.execute('CREATE TABLE index_constituents(date VARCHAR, stock_code VARCHAR, index_code VARCHAR)')
            if snapshot == 'valid':
                conn.executemany('INSERT INTO index_constituents VALUES (?, ?, ?)', [(d, 'A', "O'X") for d in ['2026-01-05','2026-01-30']])
        manager = DynamicStockPoolManager(conn, data_bindings={})
        monkeypatch.setattr(manager, '_save_filter_log', lambda log: None)
        filters = [StockFilter('index_code', '==', "O'X")]
        def run():
            if entry == 'monthly':
                return manager.get_stock_pools_for_period('2026-01-01','2026-01-31',filters,
                    schedule={'unit':'month','session':-1,'include_previous':False})['2026-01-30']
            if entry == 'weekly':
                return manager.get_stock_pools_for_period('2026-01-05','2026-01-05',filters,
                    rebalance_schedule='weekly')['2026-01-05']
            return manager._select_stock_pool(datetime(2026, 1, 5), filters)
        if snapshot == 'missing':
            with pytest.raises(duckdb.Error): run()
        else:
            assert run() == (['A'] if snapshot == 'valid' else [])


@pytest.mark.parametrize('conditions', [[('!=', 'I')], [('==', None)], [('==', '')], [('==', 'I'), ('==', 'J')]])
def test_unsupported_index_filters_are_not_silently_skipped(conditions):
    from utils.dynamic_stock_pool import StockFilter
    with duckdb.connect() as conn:
        manager = DynamicStockPoolManager(conn, data_bindings={})
        with pytest.raises(ValueError):
            manager._has_index_filter([StockFilter('index_code', op, val) for op, val in conditions])


@pytest.mark.parametrize('index_code', [' I', 'I ', '\tI', 'I\n', '\u3000I\u3000'])
def test_index_code_surrounding_whitespace_is_rejected(index_code):
    from utils.dynamic_stock_pool import StockFilter
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE index_constituents AS SELECT '2026-01-05' AS date, "
                     "'A' AS stock_code, 'I' AS index_code")
        manager = DynamicStockPoolManager(conn, data_bindings={})
        assert manager._filter_by_index(['A'], '2026-01-05', 'I') == ['A']
        for request in (
            lambda: manager._has_index_filter([StockFilter('index_code', '==', index_code)]),
            lambda: manager._filter_by_index(['A'], '2026-01-05', index_code),
            lambda: manager._read_index_constituents(index_code, exists_only=True),
        ):
            with pytest.raises(ValueError, match='leading or trailing whitespace'):
                request()


def test_adapter_rejects_index_code_surrounding_whitespace(source):
    reader, _ = source
    with pytest.raises(ValueError, match='leading or trailing whitespace'):
        reader.fetch_index_constituents(" O'X ", '2026-01-05')


def test_importer_flat_parquet_export_is_readable(tmp_path):
    from data_pipeline.loaders.import_index_constituents_duckdb import DuckDBIndexConstituentsImporter
    csv = tmp_path / 'members.csv'; frame().to_csv(csv, index=False)
    root = tmp_path / 'parquet'
    importer = DuckDBIndexConstituentsImporter(str(tmp_path / 'input.duckdb'), str(root), export_parquet=True)
    try:
        importer.import_csv(str(csv))
        importer.export_to_parquet()
    finally: importer.close()
    assert (root / 'index_constituents' / 'data.parquet').is_file()
    reader = DuckDBAdapter(parquet_dir=str(root), data_bindings={})
    try: assert reader.fetch_index_constituents("O'X", '2026-01-05') == ['A', 'B']
    finally: reader.close()


def test_specific_date_limit_is_applied_after_index_filter():
    from datetime import datetime
    from utils.dynamic_stock_pool import StockFilter
    with duckdb.connect() as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE factor_data AS SELECT '2026-01-05' AS date, stock_code, value "
                     "FROM (VALUES ('OUTSIDE', 10), ('A', 2), ('B', 1)) AS t(stock_code, value)")
        conn.execute("CREATE TABLE index_constituents AS SELECT '2026-01-05' AS date, 'A' AS stock_code, 'I' AS index_code")
        manager = DynamicStockPoolManager(conn, data_bindings={})
        result = manager._select_stock_pool(datetime(2026, 1, 5),
            [StockFilter('index_code', '==', 'I')], max_stocks=1, sort_by='value')
        assert result == ['A']


def test_frozen_membership_versions_remain_distinct(tmp_path):
    source, store = tmp_path / 'source.duckdb', tmp_path / 'input.duckdb'
    refs = []
    for stock in ['OLD', 'NEW']:
        with duckdb.connect(str(source)) as conn:
            conn.execute("CREATE OR REPLACE TABLE index_constituents AS SELECT '2026-01-05' AS date, ? AS stock_code, 'I' AS index_code", [stock])
        refs.append(migrate_table(source, store, 'index_constituents', role='input', execute=True)['ref'])
    for ref, expected in zip(refs, ['OLD', 'NEW']):
        reader = DuckDBAdapter(data_bindings={'input_db': str(store), 'inputs': {'index_constituents': ref}})
        try: assert reader.fetch_index_constituents('I', '2026-01-05') == [expected]
        finally: reader.close()
