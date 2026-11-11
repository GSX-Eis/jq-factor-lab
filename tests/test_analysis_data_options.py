"""Exercise the private analysis entrypoints with real local readers."""
import json
from pathlib import Path
import duckdb
import pandas as pd
import pytest
import run_analysis_main as runner
from utils.factor_base import FactorBase
from utils.data_access_config import load_data_access_config
from utils.config_loader import load_and_parse_config
from backtest_and_analysis.factor_loader import FactorLoader


FACTOR = '''from utils.factor_base import FactorBase
class Probe(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(['close'], table_name='price_data')
    def calculate_factor(self, data):
        return data.pivot(index='date', columns='code', values='close')
'''

@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv('QUANT_DATA_BINDINGS',raising=False)
    db=tmp_path/'input.duckdb'
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-01-31' date, symbol AS stock_code, 1. AS close, 1. AS factor FROM (VALUES ('A'),('B'),('C'),('D'),('E')) t(symbol)")
        conn.execute("CREATE TABLE wanted_prices AS SELECT DATE '2024-01-31' trade_date, symbol, 99. AS close, 1. AS factor FROM (VALUES ('A'),('B'),('C'),('D'),('E')) t(symbol)")
    data=load_data_access_config().as_dict()
    data['price_tables']['daily']='wanted_prices'
    data['tables']['wanted_prices']={'date_column':'trade_date','stock_column':'symbol','parquet_subdir':'wanted_prices','default_fields':['close']}
    path=tmp_path/'access.json';path.write_text(json.dumps(data))
    factor=tmp_path/'probe.py';factor.write_text(FACTOR)
    return db,path,factor


@pytest.mark.parametrize('dual', [False,True,'fallback'])
def test_dynamic_analysis_reads_configured_dataset(setup,tmp_path,monkeypatch,dual):
    db,path,factor_file=setup
    config={'start_date':'2024-01-01','end_date':'2024-01-31','duckdb_path':str(db),
        'data_config_path':str(path),'pit_adjustment':{'enabled':False},'materialize_only':True}
    if dual:
        parquet=tmp_path/'parquet';parquet.mkdir()
        config.update(parquet_dir=str(parquet),source_policy='duckdb_then_parquet_on_empty')
        if dual == 'fallback':
            folder=parquet/'wanted_prices'/'year=2024'/'month=1';folder.mkdir(parents=True)
            with duckdb.connect(str(db)) as conn:
                conn.execute('COPY wanted_prices TO ? (FORMAT PARQUET)',[str(folder/'part.parquet')])
                conn.execute('DELETE FROM wanted_prices')
    captured={}
    def persist(**kwargs): captured['values']=kwargs['factor_df'];return True
    monkeypatch.setattr(runner,'_persist_materialized_factor_values',persist)
    cls=FactorLoader.load_factor_class_from_file(str(factor_file))[0]
    assert runner.analyze_factor_dynamic(cls,{},config,None,None,None,{'2024-01':list('ABCDE')})
    assert captured['values'].loc['2024-01-31','A']==99.


@pytest.mark.parametrize('directory', [False,True])
def test_static_loader_forwards_data_options(setup,directory):
    db,path,factor_file=setup
    load=FactorLoader.load_all_factors if directory else FactorLoader.load_factor_from_file
    factors=load(str(factor_file.parent if directory else factor_file),None,'2024-01-01','2024-01-31',['A'],None,
        duckdb_path=str(db), data_config_path=str(path),pit_enabled=False)
    assert len(factors)==1
    try: assert factors[0].get_factor_values().loc['2024-01-31','A']==99.
    finally: factors[0]._duckdb_adapter.close()


def test_dynamic_file_options_are_resolved_and_reach_analysis_config(setup,tmp_path):
    db,path,_=setup
    cfg=tmp_path/'analysis.json'
    cfg.write_text(json.dumps({'database':{'duckdb_path':str(db),'data_config_path':path.name,
         'source_policy':'duckdb_then_parquet_on_empty'},'dynamic_pools':{'demo':{'filters':[]}}}))
    database,analysis,pools=load_and_parse_config(str(cfg))
    assert analysis['data_config_path']==str(path)
    assert analysis['source_policy']=='duckdb_then_parquet_on_empty'
    assert database['data_config_path']==str(path)


def test_pool_snapshot_identity_includes_mapping_contents(setup):
    db,path,_=setup
    cfg={'duckdb_path':str(db),'data_config_path':str(path)}
    before=runner._pool_snapshot_fingerprint(cfg,[])
    data=json.loads(path.read_text()); data['price_tables']['daily']='price_data_daily';path.write_text(json.dumps(data))
    assert runner._pool_snapshot_fingerprint(cfg,[]) != before


def test_static_main_receives_custom_mapping(setup,tmp_path,monkeypatch):
    db,path,factor_file=setup
    captured={}; opened=[]
    original=runner.connect_read_only
    def connect(*args,**kwargs):
        conn=original(*args,**kwargs);opened.append(conn);return conn
    monkeypatch.setattr(runner,'connect_read_only',connect)
    monkeypatch.setattr(runner,'wait_for_db_connection',lambda **k:None)
    monkeypatch.setattr(runner,'wait_for_result_db_write_slot',lambda *a:None)
    monkeypatch.setattr(runner,'PriceFetcher',lambda *a,**k:None)
    monkeypatch.setattr(runner,'AlphalensRunner',lambda **k:None)
    monkeypatch.setattr(runner,'ResultWriter',lambda *a,**k:None)
    monkeypatch.setattr(runner,'MONTHLY_WRITER_AVAILABLE',False)
    def analyze(factor,*args):
        try: captured['values']=factor.get_factor_values();return True
        finally: runner._close_factor_duckdb_handles(factor,'probe')
    monkeypatch.setattr(runner,'analyze_factor_static',analyze)
    try:
        runner.main({'start_date':'2024-01-01','end_date':'2024-01-31','duckdb_path':str(db),
            'results_duckdb_path':str(tmp_path/'output.duckdb'),'price_table':'price_data_daily',
            'n_quantiles':1,'code_list':['A'],'factor_file':str(factor_file),'data_config_path':str(path),
            'use_precomputed_returns':False,'max_workers':1})
        assert captured['values'].loc['2024-01-31','A']==99.
    finally:
        for conn in opened:conn.close()


def test_new_data_options_bypass_old_realization_receipt(monkeypatch):
    class Cached:
        DATA_FREQUENCY='5m'
        FACTOR_VALUE_CACHE_DIR='unused'
    def unexpected(*a):raise AssertionError('old cache must not be consulted')
    monkeypatch.setattr(runner.inspect,'getsourcefile',unexpected)
    assert runner._load_factor_value_cache(Cached, {'data_config_path':'new.json'}) is None
    assert runner._load_factor_value_cache(Cached, {'source_policy':'duckdb_then_parquet_on_empty'}) is None


def test_dual_source_mutable_pool_cache_is_not_reused(tmp_path):
    path=tmp_path/'snapshot.json'
    cfg={'pool_snapshot_path':str(path),'source_policy':'duckdb_then_parquet_on_empty'}
    path.write_text(json.dumps({'fingerprint':runner._pool_snapshot_fingerprint(cfg,[]),'pools':{'demo':{'2024-01':['OLD']}}}))
    assert runner._load_pool_snapshot(cfg,[]) is None


@pytest.mark.parametrize('dual', [False, True])
def test_main_precomputed_pools_use_configured_index_source(setup,tmp_path,monkeypatch,dual):
    from calendar_test_support import seed_calendar
    db,path,factor_file=setup
    with duckdb.connect(str(db)) as conn:
        seed_calendar(conn)
        conn.execute('CREATE TABLE factor_data AS SELECT * FROM price_data_daily')
        conn.execute("CREATE TABLE members AS SELECT DATE '2024-02-01' AS day, 'A' AS security, 'I' AS benchmark")
    data=json.loads(path.read_text())
    data['tables']['members']={'date_column':'day','stock_column':'security','parquet_subdir':'members','default_fields':[]}
    data['index_constituents']={'table':'members','index_column':'benchmark'}
    path.write_text(json.dumps(data))
    captured={};opened=[]
    original=runner.connect_read_only
    def connect(*a,**k):
        conn=original(*a,**k);opened.append(conn);return conn
    monkeypatch.setattr(runner,'connect_read_only',connect)
    monkeypatch.setattr(runner,'wait_for_db_connection',lambda **k:None)
    monkeypatch.setattr(runner,'wait_for_result_db_write_slot',lambda *a:None)
    class Writer:
        using_duckdb=True
        def __init__(self,*a,**k):pass
    monkeypatch.setattr(runner,'ResultWriter',Writer)
    def analyze(*args):captured['pool']=args[-1];return True
    monkeypatch.setattr(runner,'analyze_factor_dynamic',analyze)
    config={'start_date':'2024-02-01','end_date':'2024-02-29','duckdb_path':str(db),
        'results_duckdb_path':str(tmp_path/'output.duckdb'),'price_table':'price_data_daily',
        'factor_file':str(factor_file),'data_config_path':str(path),'materialize_only':True,
        'max_workers':1,'use_dynamic_pools':True,'dynamic_pools':{'demo':{'rebalance_schedule':'monthly',
        'filters':[{'field':'index_code','operator':'==','value':'I'}]}}}
    if dual:
        # Force membership fallback; fundamental tables and calendar stay in DuckDB.
        folder=tmp_path/'parquet'/'members';folder.mkdir(parents=True)
        with duckdb.connect(str(db)) as conn:
            conn.execute('COPY members TO ? (FORMAT PARQUET)',[str(folder/'part.parquet')])
            conn.execute('DELETE FROM members')
        config.update(parquet_dir=str(tmp_path/'parquet'),source_policy='duckdb_then_parquet_on_empty')
    try:
        runner.main(config)
        assert captured['pool']=={'2024-02-01':['A']}
    finally:
        for conn in opened:conn.close()
