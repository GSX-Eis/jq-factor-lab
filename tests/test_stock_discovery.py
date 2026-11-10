"""Discovery and panel reads must use the same configured daily source."""
import json

import duckdb
import pandas as pd
import pytest

from utils.data_access_config import load_data_access_config
from utils.data_store import migrate_table
from utils.duckdb_adapter import DuckDBAdapter


@pytest.fixture(params=['duckdb', 'bound', 'parquet'])
def reader(tmp_path, request):
    data = pd.DataFrame({'trade_date': pd.to_datetime(['2024-01-31 15:00', '2024-02-01 15:00', '2024-02-02 00:00']),
                         'symbol': ['A', 'A', 'NEXT'], 'close': [1., 2., 3.]})
    config = load_data_access_config().as_dict()
    config['price_tables']['daily'] = 'custom_prices'
    config['tables']['custom_prices'] = {'date_column': 'trade_date', 'stock_column': 'symbol',
        'parquet_subdir': 'custom', 'default_fields': ['close']}
    cfg = tmp_path / 'config.json'; cfg.write_text(json.dumps(config))
    source = tmp_path / 'input.duckdb'
    with duckdb.connect(str(source)) as conn:
        conn.execute('CREATE TABLE custom_prices AS SELECT * FROM data')
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-02-01' AS date, 'WRONG' AS stock_code, 9 AS close")
    args = {'duckdb_path': str(source), 'data_bindings': {}}
    if request.param == 'bound':
        store = tmp_path / 'bound.duckdb'
        ref = migrate_table(source, store, 'custom_prices', role='input', execute=True)['ref']
        args = {'data_bindings': {'input_db': str(store), 'inputs': {'custom_prices': ref}}}
    elif request.param == 'parquet':
        root = tmp_path / 'parquet'
        for (year, month), rows in data.groupby([data.trade_date.dt.year, data.trade_date.dt.month]):
            folder = root / 'custom' / f'year={year}' / f'month={month}'
            folder.mkdir(parents=True)
            rows.to_parquet(folder / 'data.parquet', index=False)
        args = {'parquet_dir': str(root), 'data_bindings': {}}
    adapter = DuckDBAdapter(**args, data_config_path=str(cfg))
    yield adapter
    adapter.close()


def test_discovery_uses_config_full_end_day_and_count_threshold(reader):
    assert reader.get_available_stocks('2024-01-31', '2024-02-01', 2) == ['A']
    stocks = reader.get_available_stocks('2024-02-01', '2024-02-01', 1)
    assert stocks == ['A']
    assert reader.fetch_price_data(stocks, '2024-02-01', '2024-02-01').close.tolist() == [2.]
    assert reader.get_available_stocks('2024-01-31', '2024-02-01', 3) == []
    assert reader.get_available_stocks('2024-03-01', '2024-03-02', 1) == []


@pytest.mark.parametrize('start,end,minimum', [
    ('bad', '2024-02-01', 1), ('2024-02-02', '2024-02-01', 1),
    ('2024-02-01', '2024-02-01', 0), ('2024-02-01', '2024-02-01', -1),
    ('2024-02-01', '2024-02-01', True), ('2024-02-01', '2024-02-01', 1.5),
    ('2024-02-01', '2024-02-01', '1 OR TRUE'),
])
def test_discovery_rejects_invalid_query_inputs(reader, start, end, minimum):
    with pytest.raises(ValueError):
        reader.get_available_stocks(start, end, minimum)


@pytest.mark.parametrize('primary', ['missing', 'empty', 'insufficient', 'enough'])
def test_discovery_fallback_only_when_primary_has_no_rows(tmp_path, primary):
    data = pd.DataFrame({'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
                         'stock_code': ['A', 'A'], 'close': [1., 2.]})
    folder = tmp_path / 'parquet' / 'price_data_daily' / 'year=2024' / 'month=1'
    folder.mkdir(parents=True); data.to_parquet(folder / 'data.parquet', index=False)
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)) as conn:
        if primary != 'missing':
            limit = {'empty': 0, 'insufficient': 1, 'enough': 2}[primary]
            conn.execute(f'CREATE TABLE price_data_daily AS SELECT * FROM data LIMIT {limit}')
    adapter = DuckDBAdapter(duckdb_path=str(db), parquet_dir=str(tmp_path / 'parquet'),
                           source_policy='duckdb_then_parquet_on_empty', data_bindings={})
    try:
        if primary == 'missing':
            with pytest.raises(duckdb.Error): adapter.get_available_stocks('2024-01-01', '2024-01-02', 2)
        else:
            assert adapter.get_available_stocks('2024-01-01', '2024-01-02', 2) == ([] if primary == 'insufficient' else ['A'])
    finally:
        adapter.close()


@pytest.mark.parametrize('kind', ['missing_directory', 'empty_directory', 'broken_file'])
def test_discovery_parquet_errors_propagate(tmp_path, kind):
    if kind != 'missing_directory':
        folder = tmp_path / 'price_data_daily'; folder.mkdir()
        if kind == 'broken_file': (folder / 'data.parquet').write_bytes(b'bad')
    adapter = DuckDBAdapter(parquet_dir=str(tmp_path), data_bindings={})
    try:
        with pytest.raises((FileNotFoundError, duckdb.Error)):
            adapter.get_available_stocks('2024-01-01', '2024-01-02', 1)
    finally:
        adapter.close()
