"""Price reads must preserve requested frequency and a stable empty schema."""
import duckdb
import pandas as pd
import pytest

from utils.duckdb_adapter import DuckDBAdapter
from utils.factor_base import FactorBase


DEFAULT_FIELDS = ['open', 'close', 'high', 'low', 'volume', 'money']


def write_prices(path):
    with duckdb.connect(str(path)) as conn:
        for table, value in [('price_data_daily', 1.0), ('price_data_5m', 5.0)]:
            conn.execute(f"CREATE TABLE {table} AS SELECT DATE '2026-01-05' AS date, "
                         "'A' AS stock_code, ? AS open, ? AS close, ? AS high, ? AS low, "
                         "100. AS volume, 200. AS money", [value] * 4)


@pytest.fixture(params=['duckdb', 'parquet'])
def reader(tmp_path, request):
    database = tmp_path / 'prices.duckdb'
    write_prices(database)
    if request.param == 'duckdb':
        adapter = DuckDBAdapter(duckdb_path=str(database), data_bindings={})
    else:
        root = tmp_path / 'parquet'
        with duckdb.connect(str(database)) as conn:
            for table in ('price_data_daily', 'price_data_5m'):
                folder = root / table / 'year=2026' / 'month=1'
                folder.mkdir(parents=True)
                conn.execute(f'COPY {table} TO ? (FORMAT PARQUET)', [str(folder / 'data.parquet')])
        from utils.data_revision import parquet_publication
        for table in ('price_data_daily', 'price_data_5m'):
            with parquet_publication(root, table):
                pass
        adapter = DuckDBAdapter(parquet_dir=str(root), data_bindings={})
    yield adapter
    adapter.close()


@pytest.mark.parametrize('frequency', ['1m', '5M', '', None])
@pytest.mark.parametrize('stocks', [[], ['A']])
def test_unsupported_frequency_never_reads_daily_data(reader, frequency, stocks):
    # Prime a daily cache entry as well: invalid requests must not hit it.
    reader.fetch_price_data(['A'], '2026-01-05', '2026-01-05', ['close'])
    with pytest.raises(ValueError, match='frequency'):
        reader.fetch_price_data(stocks, '2026-01-05', '2026-01-05', ['close'], frequency)


@pytest.mark.parametrize('fields,expected', [
    (None, ['date', 'code', *DEFAULT_FIELDS]),
    ([], ['date', 'code']),
    (['volume', 'close'], ['date', 'code', 'volume', 'close']),
    (['date', 'stock_code', 'close'], ['date', 'code', 'close']),
])
@pytest.mark.parametrize('stocks', [[], ['ABSENT']])
def test_empty_and_nonempty_prices_have_same_column_contract(reader, fields, expected, stocks):
    nonempty = reader.fetch_price_data(['A'], '2026-01-05', '2026-01-05', fields)
    empty = reader.fetch_price_data(stocks, '2026-01-05', '2026-01-05', fields)
    assert nonempty.columns.tolist() == empty.columns.tolist() == expected
    assert empty.empty
    assert pd.api.types.is_datetime64_any_dtype(empty.date)
    assert pd.api.types.is_datetime64_any_dtype(nonempty.date)
    assert isinstance(empty.index, pd.RangeIndex)


def test_valid_frequencies_and_cache_keep_values_and_column_order(reader, monkeypatch):
    assert reader.fetch_price_data(['A'], '2026-01-05', '2026-01-05', ['volume', 'close']).close.tolist() == [1.0]
    assert reader.fetch_price_data(['A'], '2026-01-05', '2026-01-05', ['volume', 'close'], '5m').close.tolist() == [5.0]
    def unexpected_query(*args, **kwargs):
        pytest.fail('A repeated valid request should use its cached result')
    monkeypatch.setattr(reader, '_query_from_duckdb', unexpected_query)
    monkeypatch.setattr(reader, '_query_from_parquet', unexpected_query)
    result = reader.fetch_price_data(['A'], '2026-01-05', '2026-01-05', ['volume', 'close'])
    assert result.columns.tolist() == ['date', 'code', 'volume', 'close']
    assert result.close.tolist() == [1.0]


def test_empty_stocks_do_not_open_connection(reader, monkeypatch):
    def unexpected_connection():
        pytest.fail('No connection is needed without stocks')
    monkeypatch.setattr(reader, '_get_connection', unexpected_connection)
    result = reader.fetch_price_data([], '2026-01-05', '2026-01-05', ['close'])
    assert result.columns.tolist() == ['date', 'code', 'close']


def test_missing_parquet_dataset_is_an_error(tmp_path):
    reader = DuckDBAdapter(parquet_dir=str(tmp_path), data_bindings={})
    try:
        with pytest.raises(FileNotFoundError):
            reader.fetch_price_data(['A'], '2026-01-05', '2026-01-05', ['close'])
    finally:
        reader.close()


@pytest.mark.parametrize('mode', ['path', 'borrowed'])
@pytest.mark.parametrize('frequency', ['1m', '5M', None])
def test_factor_base_does_not_coerce_invalid_local_frequency(tmp_path, mode, frequency):
    path = tmp_path / 'prices.duckdb'
    write_prices(path)
    conn = duckdb.connect(str(path)) if mode == 'borrowed' else None
    factor = FactorBase(conn, 'FrequencyCheck', '2026-01-05', '2026-01-05',
                        stock_pool=['A'], data_frequency=frequency, pit_enabled=False,
                        duckdb_path=str(path) if mode == 'path' else None, data_bindings={})
    try:
        with pytest.raises(ValueError, match='frequency'):
            factor.get_price_data()
        # Exercise the direct adapter route too, bypassing _fetch_raw_data.
        if mode == 'path':
            with pytest.raises(ValueError, match='frequency'):
                factor._fetch_raw_data_duckdb(['close'], 'price_data')
    finally:
        if factor._duckdb_adapter is not None:
            factor._duckdb_adapter.close()
        if conn is not None:
            conn.close()


def test_factor_base_preserves_cloud_minute_frequency_without_local_reads():
    factor = FactorBase(None, 'CloudFrequency', '2026-01-05', '2026-01-05',
                        stock_pool=['A'], data_frequency='15m', data_bindings={})
    assert factor.data_frequency == '15m'
