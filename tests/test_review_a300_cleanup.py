"""Approved A300 failure cleanup; all databases are temporary or in memory."""
import sys
from types import SimpleNamespace

import duckdb
import pytest

from backtest_and_analysis.factor_loader import FactorLoader
import utils.factor_base as base


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit, ValueError])
@pytest.mark.parametrize('serial', [False, True])
def test_module_failure_removes_partial_registration(tmp_path, error_type, serial):
    path = tmp_path / 'interrupted_factor.py'
    path.write_text('partial = 1\nraise ' + error_type.__name__ + "('original')\n")
    name = FactorLoader._module_name_for_path(path)
    old_module = SimpleNamespace(old=True)
    sys.modules[name] = old_module
    try:
        with pytest.raises(error_type, match='original'):
            FactorLoader._load_module_from_path(path, serial=serial)
        assert name not in sys.modules
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def local_env(monkeypatch):
    for key in ['JQ_RUNNING', 'QUANTOPIAN_RUNNING']:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('QUANT_DATA_BINDINGS', '{}')
    monkeypatch.setitem(sys.modules, 'jqdata', None)


class Reader:
    def __init__(self, failure=None):
        self.duckdb_path = 'temporary-source'
        self.data_bindings = {}
        self.closes = 0
        self.failure = failure

    def close(self):
        self.closes += 1
        if self.failure:
            raise self.failure


@pytest.mark.parametrize('where', ['pool', 'mkdir', 'adapter_metadata'])
@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit, PermissionError])
@pytest.mark.parametrize('close_failure', [False, True])
def test_constructor_preserves_primary_and_closes_owned(
    tmp_path, monkeypatch, local_env, where, error_type, close_failure
):
    primary = error_type('original construction failure')
    readers = [Reader(OSError('first close') if close_failure else None), Reader()]
    pools = {'2024-02-01': ['A']}

    def fail(*args, **kwargs):
        raise primary

    manager = SimpleNamespace(
        _duckdb_adapter=readers[1], validate_filters=lambda _: True,
        get_stock_pools_for_period=fail if where == 'pool' else lambda *a, **k: pools,
    )
    if where == 'adapter_metadata':
        class BadPath:
            def __str__(self):
                raise primary
        readers[0].duckdb_path = BadPath()
    monkeypatch.setattr(base, 'get_data_adapter', lambda **kw: readers[0])
    monkeypatch.setattr(base, 'DynamicStockPoolManager', lambda *a, **kw: manager)
    if where == 'mkdir':
        monkeypatch.setattr(base.os, 'makedirs', fail)
    with duckdb.connect(':memory:') as borrowed:
        with pytest.raises(error_type) as caught:
            base.FactorBase(
                borrowed, 'cleanup', '2024-02-01', '2024-02-29',
                duckdb_path='temporary-source', use_dynamic_pool=True,
                stock_filters=[object()], cache_dir=str(tmp_path / 'new-cache'),
            )
        assert caught.value is primary
        assert readers[0].closes == 1
        assert readers[1].closes == (0 if where == 'adapter_metadata' else 1)
        assert borrowed.execute('SELECT 7').fetchone() == (7,)


@pytest.mark.parametrize('where', ['interrupt', 'mkdir', 'close_failure'])
def test_actual_dynamic_pool_reader_closed(tmp_path, monkeypatch, local_env, where):
    from calendar_test_support import seed_calendar
    from utils.duckdb_adapter import DuckDBAdapter
    from utils.dynamic_stock_pool import StockFilter

    db = tmp_path / 'pool.duckdb'
    with duckdb.connect(str(db)) as con:
        seed_calendar(con)
        con.execute("CREATE TABLE factor_data AS SELECT DATE '2024-01-31' AS date, 'A' AS stock_code, 10 AS close")
        con.execute('CREATE TABLE price_data_daily AS SELECT * FROM factor_data')
        con.execute('CREATE TABLE index_constituents(date DATE, stock_code VARCHAR, index_code VARCHAR)')
    readers = []
    original_init = DuckDBAdapter.__init__
    original_close = DuckDBAdapter.close
    original_mkdir = base.os.makedirs
    cache = tmp_path / 'cache'
    primary = KeyboardInterrupt('original') if where == 'interrupt' else PermissionError('original')

    def capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        readers.append(self)

    def fail(*args, **kwargs):
        raise primary

    def mkdir(path, *args, **kwargs):
        if str(path) == str(cache):
            raise primary
        return original_mkdir(path, *args, **kwargs)

    def close(self):
        original_close(self)
        if where == 'close_failure' and self is readers[0]:
            raise OSError('secondary close failure')

    monkeypatch.setattr(DuckDBAdapter, '__init__', capture)
    monkeypatch.setattr(DuckDBAdapter, 'close', close)
    monkeypatch.setattr(base.DynamicStockPoolManager, '_save_filter_log', lambda *a: None)
    if where == 'interrupt':
        monkeypatch.setattr(base.DynamicStockPoolManager, 'get_stock_pools_for_period', fail)
    else:
        monkeypatch.setattr(base.os, 'makedirs', mkdir)
    try:
        with duckdb.connect(':memory:') as borrowed:
            with pytest.raises(type(primary)) as caught:
                base.FactorBase(
                    borrowed, 'real-reader', '2024-02-01', '2024-02-29',
                    duckdb_path=str(db), price_table='price_data_daily',
                    use_dynamic_pool=True, stock_filters=[StockFilter('index_code', '==', 'I')],
                    cache_dir=str(cache),
                )
            assert caught.value is primary
            assert len(readers) == 2
            assert all(reader._conn is None for reader in readers)
            assert borrowed.execute('SELECT 7').fetchone() == (7,)
    finally:
        for reader in readers:
            original_close(reader)
