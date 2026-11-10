"""Daily query bounds and literal values must agree across local sources."""
import duckdb
import pandas as pd
import pytest

from utils.data_store import migrate_table, table_name as quote_table
from utils.duckdb_adapter import DuckDBAdapter


def make_adapter(root, mode, frame, table='panel', date_column='date'):
    root.mkdir(parents=True, exist_ok=True)
    if mode == 'parquet':
        dates = pd.to_datetime(frame[date_column])
        for (year, month), partition in frame.groupby([dates.dt.year, dates.dt.month]):
            folder = root / table / f'year={year}' / f'month={month}'
            folder.mkdir(parents=True, exist_ok=True)
            partition.to_parquet(folder / 'data.parquet', index=False)
        return DuckDBAdapter(parquet_dir=str(root), data_bindings={})
    source = root / 'source.duckdb'
    with duckdb.connect(str(source)) as conn:
        conn.execute(f'CREATE TABLE {quote_table(table)} AS SELECT * FROM frame')
    if mode == 'bound':
        store = root / 'input.duckdb'
        ref = migrate_table(source, store, table, role='input', execute=True)['ref']
        return DuckDBAdapter(data_bindings={'input_db': str(store), 'inputs': {table: ref}})
    return DuckDBAdapter(duckdb_path=str(source), data_bindings={})


@pytest.fixture(params=['duckdb', 'parquet', 'bound'])
def source_factory(tmp_path, request):
    readers = []

    def create(frame, **kwargs):
        reader = make_adapter(tmp_path / str(len(readers)), request.param, frame, **kwargs)
        readers.append(reader)
        return reader

    yield request.param, create
    for reader in readers:
        reader.close()


def sample_frame():
    return pd.DataFrame({
        'date': pd.to_datetime([
            '2025-12-30 23:59:59', '2025-12-31 00:00:00',
            '2025-12-31 14:55:00', '2026-01-01 23:59:59.999999',
            '2026-01-02 00:00:00',
        ], format='mixed'),
        'stock_code': ['A'] * 5,
        'value': list(range(5)),
    })


def test_all_tables_include_full_end_day_and_exclude_next_day(source_factory):
    _, create = source_factory
    reader = create(sample_frame())
    result = reader.fetch_table_data('panel', ['A'], '2025-12-31', '2026-01-01', ['value'])
    assert result.value.tolist() == [1, 2, 3]


@pytest.mark.parametrize('table', ['price_data_5m', 'main.price_data_5m'])
def test_schema_qualified_table_has_same_intraday_bounds(tmp_path, table):
    reader = make_adapter(tmp_path, 'duckdb', sample_frame(), table='price_data_5m')
    try:
        result = reader.fetch_table_data(table, ['A'], '2025-12-31', '2026-01-01', ['value'])
        assert result.value.tolist() == [1, 2, 3]
    finally:
        reader.close()


@pytest.mark.parametrize("storage", ["date", "iso_string", "timestamp_string"])
def test_date_columns_keep_inclusive_daily_bounds(source_factory, storage):
    _, create = source_factory
    frame = sample_frame()
    if storage == 'date':
        frame['date'] = frame.date.dt.date
    elif storage == 'iso_string':
        frame['date'] = frame.date.dt.strftime('%Y-%m-%d')
    else:
        frame['date'] = frame.date.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    reader = create(frame)
    result = reader.fetch_table_data('panel', ['A'], '2025-12-31', '2026-01-01', ['value'])
    assert result.value.tolist() == [1, 2, 3]


@pytest.mark.parametrize('stock', ["O'X", "A') OR TRUE --"])
def test_stock_values_are_literals_not_sql(source_factory, stock):
    _, create = source_factory
    frame = pd.DataFrame({'date': pd.to_datetime(['2025-12-31'] * 2),
                          'stock_code': [stock, 'OTHER'], 'value': [7, 99]})
    reader = create(frame)
    result = reader.fetch_table_data('panel', [stock], '2025-12-31', '2025-12-31', ['value'])
    assert result.code.tolist() == [stock]
    assert result.value.tolist() == [7]


def test_identifier_quotes_are_escaped(source_factory):
    mode, create = source_factory
    frame = pd.DataFrame({'trade"date': pd.to_datetime(['2025-12-31']),
                          'stock"code': ['A'], 'close"value': [7]})
    reader = create(frame, table='panel"quoted', date_column='trade"date')
    query = reader._query_from_parquet if mode == 'parquet' else reader._query_from_duckdb
    result = query('panel"quoted', ['A'], '2025-12-31', '2025-12-31',
                   ['stock"code', 'close"value', 'trade"date'], 'trade"date', 'stock"code')
    assert result.columns.tolist() == ['trade"date', 'stock"code', 'close"value']
    assert result['close"value'].tolist() == [7]


def test_invalid_field_is_an_error_not_empty_data(source_factory):
    _, create = source_factory
    reader = create(sample_frame())
    with pytest.raises(duckdb.Error):
        reader.fetch_table_data('panel', ['A'], '2025-12-31', '2026-01-01', ['missing'])


def test_invalid_date_is_an_error(source_factory):
    _, create = source_factory
    reader = create(sample_frame())
    with pytest.raises((duckdb.Error, ValueError)):
        reader.fetch_table_data('panel', ['A'], "2025-12-31' OR TRUE --", '2026-01-01', ['value'])


def test_empty_stocks_and_empty_matches_remain_empty(source_factory):
    mode, create = source_factory
    reader = create(sample_frame())
    query = reader._query_from_parquet if mode == 'parquet' else reader._query_from_duckdb
    assert query('panel', [], '2025-12-31', '2026-01-01', ['value']).empty
    result = reader.fetch_table_data('panel', ['ABSENT'], '2025-12-31', '2026-01-01', ['value'])
    assert result.empty
    assert result.columns.tolist() == ['date', 'code', 'value']
    with pytest.raises(ValueError, match='start_date'):
        reader.fetch_table_data('panel', ['A'], '2026-02-01', '2025-12-01', ['value'])


def test_parquet_path_with_apostrophe_is_a_literal(tmp_path):
    reader = make_adapter(tmp_path / "owner's data", 'parquet', sample_frame())
    try:
        result = reader.fetch_table_data('panel', ['A'], '2025-12-31', '2026-01-01', ['value'])
        assert result.value.tolist() == [1, 2, 3]
    finally:
        reader.close()


@pytest.mark.parametrize('failure', ['missing_table', 'missing_field', 'broken_parquet'])
def test_source_errors_do_not_trigger_successful_fallback(tmp_path, failure, monkeypatch):
    frame = sample_frame()
    parquet = make_adapter(tmp_path / 'parquet', 'parquet', frame)
    parquet.close()
    database = tmp_path / 'input.duckdb'
    with duckdb.connect(str(database)) as conn:
        if failure != 'missing_table':
            columns = 'date, stock_code' if failure == 'missing_field' else '*'
            conn.execute(f'CREATE TABLE panel AS SELECT {columns} FROM frame')
    if failure == 'broken_parquet':
        part = next((tmp_path / 'parquet').rglob('*.parquet'))
        part.write_bytes(b'not a parquet file')
        reader = DuckDBAdapter(parquet_dir=str(tmp_path / 'parquet'), data_bindings={})
    else:
        reader = DuckDBAdapter(duckdb_path=str(database), parquet_dir=str(tmp_path / 'parquet'), data_bindings={},
                               source_policy='duckdb_then_parquet_on_empty')
        def unexpected_fallback(*args, **kwargs):
            pytest.fail('A failed DuckDB query must not fall back to Parquet')
        monkeypatch.setattr(reader, '_query_from_parquet', unexpected_fallback)
    try:
        with pytest.raises(duckdb.Error):
            reader.fetch_table_data('panel', ['A'], '2025-12-31', '2026-01-01', ['value'])
    finally:
        reader.close()


def test_successful_empty_duckdb_query_can_still_fall_back(tmp_path):
    frame = sample_frame()
    parquet = make_adapter(tmp_path / 'parquet', 'parquet', frame)
    parquet.close()
    database = tmp_path / 'input.duckdb'
    with duckdb.connect(str(database)) as conn:
        conn.execute('CREATE TABLE panel AS SELECT * FROM frame WHERE FALSE')
    reader = DuckDBAdapter(duckdb_path=str(database), parquet_dir=str(tmp_path / 'parquet'), data_bindings={},
                               source_policy='duckdb_then_parquet_on_empty')
    try:
        result = reader.fetch_table_data('panel', ['A'], '2025-12-31', '2026-01-01', ['value'])
        assert result.value.tolist() == [1, 2, 3]
    finally:
        reader.close()
