"""Cached panels retain query identity and expire at the end of a read session."""
import duckdb
import pandas as pd
import pytest

from utils.duckdb_adapter import DuckDBAdapter
from utils.data_revision import parquet_publication


class CollidingName(str):
    """Force a hash collision without changing SQL identifiers or values."""

    def __hash__(self):
        return 42


@pytest.fixture(params=['duckdb', 'parquet'])
def panel_source(tmp_path, request):
    tables = ('price_data_daily', 'factor_data', 'custom_panel')
    path = tmp_path / ('prices.duckdb' if request.param == 'duckdb' else 'parquet')

    def write(value):
        frame = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-02', '2020-01-02']),
            'stock_code': ['A', 'B'], 'close': [value, value * 2],
            'volume': [100, 200],
        })
        if request.param == 'duckdb':
            with duckdb.connect(str(path)) as conn:
                for table in tables:
                    conn.execute(f'CREATE OR REPLACE TABLE {table} AS SELECT * FROM frame')
        else:
            for table in tables:
                folder = path / table / 'year=2020' / 'month=1'
                folder.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(folder / 'panel.parquet', index=False)

    write(10)
    kwargs = {'duckdb_path' if request.param == 'duckdb' else 'parquet_dir': str(path)}
    adapter = DuckDBAdapter(**kwargs, data_bindings={})
    yield adapter, write
    adapter.close()


def query(adapter, method='price', stocks=None, fields=None):
    args = (stocks or ['A'], '2020-01-02', '2020-01-02', fields or ['close'])
    if method == 'table':
        return adapter.fetch_table_data('custom_panel', *args)
    return getattr(adapter, f'fetch_{method}_data')(*args)


@pytest.mark.parametrize('method', ['price', 'factor', 'table'])
def test_colliding_stock_hashes_do_not_reuse_other_stock_data(panel_source, method):
    adapter, _ = panel_source
    a, b = CollidingName('A'), CollidingName('B')
    assert hash((a,)) == hash((b,))
    assert query(adapter, method, stocks=[a]).close.tolist() == [10]
    result = query(adapter, method, stocks=[b])
    assert result.code.tolist() == ['B']
    assert result.close.tolist() == [20]


def test_colliding_field_hashes_do_not_reuse_other_columns(panel_source):
    adapter, _ = panel_source
    close, volume = CollidingName('close'), CollidingName('volume')
    assert hash((close,)) == hash((volume,))
    assert query(adapter, fields=[close]).close.tolist() == [10]
    result = query(adapter, fields=[volume])
    assert result.columns.tolist() == ['date', 'code', 'volume']
    assert result.volume.tolist() == [100]


@pytest.mark.parametrize('method', ['price', 'factor', 'table'])
def test_close_discards_cached_data_before_source_update(panel_source, method):
    adapter, write = panel_source
    assert query(adapter, method).close.tolist() == [10]
    adapter.close()
    adapter.close()
    write(30)
    assert query(adapter, method).close.tolist() == [30]


def test_close_clears_cache_even_without_connection(panel_source):
    adapter, _ = panel_source
    key = adapter._build_cache_key('price_data_daily', ['A'], '2020-01-02', '2020-01-02', ['close'])
    adapter._cache.put(key, pd.DataFrame({'close': [-1]}))
    assert adapter._conn is None
    adapter.close()
    assert query(adapter).close.tolist() == [10]


def test_cache_hit_preserves_order_and_copies_results(panel_source, monkeypatch):
    adapter, _ = panel_source
    if adapter.parquet_dir:
        with parquet_publication(adapter.parquet_dir, 'price_data_daily'):
            pass
    first = query(adapter, stocks=['B', 'A'], fields=['close', 'volume'])
    assert first.code.tolist() == ['A', 'B']
    first.loc[:, 'close'] = -1
    original = adapter._query_from_duckdb if adapter.duckdb_path else adapter._query_from_parquet
    attr = '_query_from_duckdb' if adapter.duckdb_path else '_query_from_parquet'

    def unexpected_query(*args, **kwargs):
        pytest.fail('Equivalent stock order should hit the existing cache')

    monkeypatch.setattr(adapter, attr, unexpected_query)
    second = query(adapter, stocks=['A', 'B'], fields=['close', 'volume'])
    assert second.close.tolist() == [10, 20]
    second.loc[:, 'close'] = -2
    assert query(adapter, stocks=['B', 'A'], fields=['close', 'volume']).close.tolist() == [10, 20]
    monkeypatch.setattr(adapter, attr, original)
    reversed_fields = query(adapter, fields=['volume', 'close'])
    assert reversed_fields.columns.tolist() == ['date', 'code', 'volume', 'close']


def test_disabled_cache_and_close_support_new_data(panel_source):
    adapter, write = panel_source
    adapter.close()
    uncached = DuckDBAdapter(duckdb_path=adapter.duckdb_path, parquet_dir=adapter.parquet_dir,
                             use_cache=False, data_bindings={})
    try:
        assert query(uncached).close.tolist() == [10]
        uncached.close()
        write(40)
        assert query(uncached).close.tolist() == [40]
    finally:
        uncached.close()


def test_reopening_bound_cache_preserves_frozen_version(tmp_path):
    from utils.data_store import migrate_table

    source, store = tmp_path / 'source.duckdb', tmp_path / 'input.duckdb'
    refs = []
    for value in (10, 30):
        with duckdb.connect(str(source)) as conn:
            conn.execute("CREATE OR REPLACE TABLE price_data_daily AS SELECT "
                         "DATE '2020-01-02' AS date, 'A' AS stock_code, ? AS close", [value])
        refs.append(migrate_table(source, store, 'price_data_daily', role='input', execute=True)['ref'])
    readers = [DuckDBAdapter(data_bindings={
        'input_db': str(store), 'inputs': {'price_data_daily': ref},
    }) for ref in refs]
    try:
        for _ in range(2):
            assert query(readers[0]).close.tolist() == [10]
            assert query(readers[1]).close.tolist() == [30]
            for reader in readers:
                reader.close()
    finally:
        for reader in readers:
            reader.close()
