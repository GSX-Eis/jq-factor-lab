"""Automatic analysis caches identify effective configs and frozen inputs."""
import json
import duckdb
import pandas as pd
import pytest
from calendar_test_support import seed_calendar
import run_analysis_main as runner
from utils import data_access_config
from utils.data_access_config import load_data_access_config
from utils.data_store import migrate_table
from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter


@pytest.fixture
def pool_store(tmp_path, monkeypatch):
    monkeypatch.delenv('QUANT_DATA_BINDINGS', raising=False)
    source = tmp_path / 'source.duckdb'
    with duckdb.connect(str(source)) as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2024-01-31' date, code stock_code, 1. AS close FROM (VALUES ('A'), ('B')) t(code)")
        conn.execute('CREATE TABLE price_data AS SELECT * FROM factor_data')
        conn.execute("CREATE TABLE index_constituents AS SELECT DATE '2024-02-01' date, stock_code, 'I' index_code FROM factor_data WHERE stock_code='A'")
        conn.execute("CREATE TABLE members_b AS SELECT DATE '2024-02-01' date, stock_code, 'I' index_code FROM factor_data WHERE stock_code='B'")
    config = {'duckdb_path': str(source), 'start_date': '2024-02-01', 'end_date': '2024-02-29',
        'pool_snapshot_path': str(tmp_path / 'pool.json')}
    return source, config


POOLS = [{'name': 'demo', 'filters': [StockFilter('index_code', '==', 'I')]}]


def read_pool(config):
    manager = DynamicStockPoolManager(None, duckdb_path=config['duckdb_path'])
    try:
        return {'demo': manager.get_stock_pools_for_period(config['start_date'], config['end_date'],
            POOLS[0]['filters'], rebalance_schedule='monthly')}
    finally:
        manager._duckdb_adapter.close()


def test_default_config_change_invalidates_automatic_pool_snapshot(pool_store, tmp_path, monkeypatch):
    source, config = pool_store
    data = load_data_access_config().as_dict()
    path = tmp_path / 'default.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(data_access_config, 'DEFAULT_CONFIG_PATH', path)
    first = read_pool(config)
    runner._save_pool_snapshot(config, POOLS, first)
    assert runner._load_pool_snapshot(config, POOLS) == first
    explicit = {**config, 'data_config_path': str(path)}
    assert runner._pool_snapshot_fingerprint(config, POOLS) == runner._pool_snapshot_fingerprint(explicit, POOLS)
    data['tables']['members_b'] = data['tables']['index_constituents']
    data['index_constituents']['table'] = 'members_b'
    path.write_text(json.dumps(data))
    assert read_pool(config) == {'demo': {'2024-02-01': ['B']}}
    assert runner._load_pool_snapshot(config, POOLS) is None


def test_frozen_version_change_invalidates_only_automatic_snapshot(pool_store, tmp_path, monkeypatch):
    source, config = pool_store
    store = tmp_path / 'input.duckdb'
    refs = {table: migrate_table(source, store, table, role='input', execute=True)['ref']
        for table in ['factor_data', 'price_data', 'index_constituents', 'trading_calendar']}
    with duckdb.connect(str(source)) as conn:
        conn.execute("UPDATE index_constituents SET stock_code='B'")
    revised = migrate_table(source, store, 'index_constituents', role='input', execute=True)['ref']
    config['duckdb_path'] = str(store)
    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'input_db': str(store), 'inputs': refs}))
    first = read_pool(config)
    runner._save_pool_snapshot(config, POOLS, first)
    assert runner._load_pool_snapshot(config, POOLS) == first
    frozen = {**config, 'pool_snapshot_read_only': True,
        'pool_snapshot_sha256': runner._sha256_file(config['pool_snapshot_path'])}
    before = store.stat()
    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'input_db': str(store),
        'inputs': {**refs, 'index_constituents': revised}}))
    assert read_pool(config) == {'demo': {'2024-02-01': ['B']}}
    assert store.stat().st_mtime_ns == before.st_mtime_ns
    assert runner._load_pool_snapshot(config, POOLS) is None
    assert runner._load_pool_snapshot(frozen, POOLS) == first


def test_legacy_filter_snapshot_is_invalidated_but_frozen_evidence_survives(pool_store, monkeypatch):
    _, config = pool_store
    first = read_pool(config)
    with monkeypatch.context() as legacy:
        legacy.setattr(runner, 'POOL_SNAPSHOT_CONTRACT', 'resolved-selection-v2')
        runner._save_pool_snapshot(config, POOLS, first)
    frozen = {**config, 'pool_snapshot_read_only': True,
              'pool_snapshot_sha256': runner._sha256_file(config['pool_snapshot_path'])}
    assert runner._load_pool_snapshot(config, POOLS) is None
    assert runner._load_pool_snapshot(frozen, POOLS) == first


def test_realization_receipt_requires_matching_effective_config(tmp_path, monkeypatch):
    class Cached:
        DATA_FREQUENCY = '5m'
        FACTOR_VALUE_CACHE_DIR = str(tmp_path)
    source_hash = runner._sha256_file(runner.inspect.getsourcefile(Cached))
    values_path = tmp_path / f'{source_hash}.parquet'
    pd.DataFrame({'A': [1.]}, index=pd.to_datetime(['2024-01-02'])).to_parquet(values_path)
    metadata = {'schema_version': 'quanta_alpha.factor_value_cache.v1', 'source_sha256': source_hash,
        'data_file': values_path.name, 'data_sha256': runner._sha256_file(values_path),
        'start_date': '2024-01-01', 'end_date': '2024-01-03'}
    receipt = tmp_path / f'{source_hash}.json'
    config = {'start_date': '2024-01-01', 'end_date': '2024-01-03'}
    receipt.write_text(json.dumps(metadata))
    assert runner._load_factor_value_cache(Cached, config) is None
    data = load_data_access_config().as_dict()
    from factor_mining.symbolic.quanta_alpha import _write_factor_value_cache
    written = _write_factor_value_cache(tmp_path, source_hash=source_hash,
        panel=pd.DataFrame({'date': pd.to_datetime(['2024-01-02']), 'code': ['A']}),
        prediction=pd.Series([1.]), panel_hash='synthetic', **config)
    assert written['data_access'] == data
    assert runner._load_factor_value_cache(Cached, config).iloc[0, 0] == 1.
    data['tables']['revised_intraday'] = dict(data['tables']['price_data_5m'])
    data['price_tables']['5m'] = 'revised_intraday'
    path = tmp_path / 'default.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(data_access_config, 'DEFAULT_CONFIG_PATH', path)
    assert runner._load_factor_value_cache(Cached, config) is None


def test_pool_snapshot_invalidated_even_if_database_stat_is_preserved(tmp_path, monkeypatch):
    import os
    from data_pipeline.loaders._transactions import import_transaction
    import run_analysis_main as runner
    monkeypatch.delenv('QUANT_DATA_BINDINGS', raising=False)
    path = tmp_path / 'market.duckdb'
    with duckdb.connect(str(path)) as conn, import_transaction(conn):
        conn.execute("CREATE TABLE prices AS SELECT 10 AS close")
    config = {'duckdb_path': str(path), 'pool_snapshot_path': str(tmp_path / 'pool.json')}
    pools = {'demo': {'2024-01': ['A']}}
    runner._save_pool_snapshot(config, [], pools)
    assert runner._load_pool_snapshot(config, []) == pools
    stat = path.stat()
    with duckdb.connect(str(path)) as conn, import_transaction(conn):
        conn.execute('UPDATE prices SET close=20')
    real_stat = os.stat
    # Hold the old stat identity fixed to isolate the committed generation.
    monkeypatch.setattr(os, "stat", lambda target, *a, **kw:
        stat if str(target) == str(path) else real_stat(target, *a, **kw))
    assert runner._load_pool_snapshot(config, []) is None


@pytest.mark.parametrize("change", ["missing_previous", "source_priority"])
def test_pool_policy_change_invalidates_automatic_snapshot(pool_store, tmp_path, monkeypatch, change):
    from utils import stock_pool_schedule
    _, config = pool_store
    policy = stock_pool_schedule.load_pool_policy()
    path = tmp_path / 'pool_policy.json'
    path.write_text(json.dumps(policy))
    monkeypatch.setattr(stock_pool_schedule, 'DEFAULT_CONFIG', path)
    first = read_pool(config)
    runner._save_pool_snapshot(config, POOLS, first)
    assert runner._load_pool_snapshot(config, POOLS) == first
    if change == 'missing_previous':
        policy['selection']['missing_previous'] = 'raise'
    else:
        policy['selection']['source_priority']['percentile'] = ['fundamental', 'price']
    path.write_text(json.dumps(policy))
    assert runner._load_pool_snapshot(config, POOLS) is None
