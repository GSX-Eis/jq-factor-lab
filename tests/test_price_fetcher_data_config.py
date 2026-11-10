from calendar_test_support import seed_calendar
"""Forward-return prices honor the same configured keys and daily table."""
import json
import duckdb
import pandas as pd
import pytest
from backtest_and_analysis.price_fetcher import PriceFetcher
from utils.data_access_config import load_data_access_config

@pytest.mark.parametrize('table',['price_data_daily','wanted'])
def test_configured_forward_prices_do_not_read_legacy_table(tmp_path,table):
    db=tmp_path/'input.duckdb'
    with duckdb.connect(str(db)) as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE wanted AS SELECT TIMESTAMP '2024-01-31 15:00:00' AS trade_date, 'A' AS symbol, 99. AS close, 1. AS open, 1. AS high, 1. AS low, 1. AS volume, 1. AS money")
        conn.execute("CREATE TABLE price_data_daily AS SELECT trade_date AS date, symbol AS stock_code, 1. AS close, open,high,low,volume,money FROM wanted")
    data=load_data_access_config().as_dict()
    data['price_tables']['daily']='wanted'
    data['tables']['wanted']={'date_column':'trade_date','stock_column':'symbol','parquet_subdir':'wanted','default_fields':['close']}
    path=tmp_path/'access.json';path.write_text(json.dumps(data))
    reader=PriceFetcher(duckdb_path=db,price_table=table,data_config_path=str(path))
    try:
        values=reader.fetch(['A'],'2024-01-31','2024-01-31',future_periods=0)
        assert values.tolist()==[99.]
        assert str(values.index[0][1])=='2024-01-31 00:00:00'
    finally:reader.close()

@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('alias', ['price_data', 'price_data_daily', 'wanted'])
def test_implicit_default_matches_explicit_config(tmp_path, monkeypatch, explicit, alias):
    from utils import data_access_config
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)) as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE wanted AS SELECT TIMESTAMP '2024-01-31 15:00:00' AS day, 'A' AS symbol, 99. AS close, 1. AS open, 1. AS high, 1. AS low, 1. AS volume, 1. AS money")
        conn.execute('CREATE TABLE price_data_daily AS SELECT day AS date, symbol AS stock_code, 1. AS close, open, high, low, volume, money FROM wanted')
        conn.execute('CREATE TABLE price_data AS SELECT * FROM price_data_daily')
    data = load_data_access_config().as_dict()
    data['price_tables']['daily'] = 'wanted'
    data['tables']['wanted'] = {'date_column': 'day', 'stock_column': 'symbol',
        'parquet_subdir': 'wanted', 'default_fields': ['close']}
    path = tmp_path / 'default.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(data_access_config, 'DEFAULT_CONFIG_PATH', path)
    reader = PriceFetcher(duckdb_path=db, price_table=alias, data_config_path=str(path) if explicit else None)
    try:
        assert reader.fetch(['A'], '2024-01-31', '2024-01-31', 0).tolist() == [99.]
    finally:
        reader.close()

@pytest.mark.parametrize('as_path', [False, True])
def test_single_database_path_parameter_reads_prices(tmp_path, monkeypatch, as_path):
    monkeypatch.delenv('QUANT_DATA_BINDINGS', raising=False)
    db = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(db)) as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-01-02' date, 'A' stock_code, 2. AS close, 1. AS open, 3. AS high, 1. AS low, 10. AS volume, 20. AS money")
    reader = PriceFetcher(duckdb_path=db if as_path else str(db), price_table='price_data_daily')
    try:
        assert reader.fetch(['A'], '2024-01-02', '2024-01-02', 0).tolist() == [2.]
    finally:
        reader.close()
        reader.close()


@pytest.mark.parametrize('path', [None, '', '   ', 'duckdb:////data/input.duckdb', 'sqlite:///input.db'])
def test_path_must_be_a_nonempty_filesystem_path(path):
    with pytest.raises(ValueError, match='filesystem path'):
        PriceFetcher(duckdb_path=path, price_table='price_data_daily')


@pytest.mark.parametrize('legacy', [{'db_conn_str': 'other.duckdb'}, {'db_backend': 'duckdb'}])
def test_removed_connection_options_are_not_silently_ignored(tmp_path, legacy):
    with pytest.raises(TypeError, match='unexpected keyword'):
        PriceFetcher(duckdb_path=tmp_path / 'missing.duckdb', price_table='price_data_daily', **legacy)
    assert not (tmp_path / 'missing.duckdb').exists()


@pytest.mark.parametrize('frequency', ['daily', '5m'])
@pytest.mark.parametrize('use_alias', [True, False])
def test_custom_price_alias_and_table_keep_frequency_semantics(tmp_path, frequency, use_alias):
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)) as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE quotes AS SELECT * FROM (VALUES (TIMESTAMP '2024-01-02 10:00:00', 'A', 2.), (TIMESTAMP '2024-01-02 15:00:00', 'A', 9.)) t(day, symbol, close)")
        for column in ('open', 'high', 'low', 'volume', 'money'):
            conn.execute(f'ALTER TABLE quotes ADD COLUMN {column} DOUBLE DEFAULT 1')
        # The alias is deliberately also a real table: the configured target must win.
        conn.execute('CREATE TABLE legacy_quotes AS SELECT *, 0 AS unused FROM quotes WHERE FALSE')
    data = load_data_access_config().as_dict()
    data['price_tables'][frequency] = 'quotes'
    data['price_aliases']['legacy_quotes'] = frequency
    data['tables']['quotes'] = dict(data['tables']['price_data_daily'],
        date_column='day', stock_column='symbol')
    path = tmp_path / 'access.json'
    path.write_text(json.dumps(data))
    reader = PriceFetcher(duckdb_path=db,
        price_table='legacy_quotes' if use_alias else 'quotes', data_config_path=str(path))
    try:
        values = reader.fetch(['A'], '2024-01-02', '2024-01-02', 0)
        assert values.tolist() == ([2., 9.] if frequency == 'daily' else [9.])
        assert all(values.index.get_level_values('date') == pd.Timestamp('2024-01-02'))
    finally:
        reader.close()
