"""Explicit boundaries approved at the line A A220 checkpoint."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner
from data_pipeline.loaders.factor_data_to_duckdb import DuckDBFactorImporter
from factor_synthesis.vectorbt_adapter.factor_loader import FactorLoader

@pytest.mark.parametrize('stock,factor', [("owner's",'alpha'),('A',"owner's_factor"),("o'ne", "f'one")])
def test_daily_loader_quotes_are_values(stock,factor):
    with duckdb.connect() as c:
        c.execute('CREATE TABLE factor_values(date DATE,stock_code VARCHAR,factor_name VARCHAR,factor_value DOUBLE)')
        c.execute('INSERT INTO factor_values VALUES (?,?,?,?)',['2024-01-02',stock,factor,42.])
        loader=FactorLoader(existing_conn=c,shared_cache={})
        direct=loader.load_factors('2024-01-02',[stock],[factor])
        assert direct.iloc[0,0]==42.
        loader.clear_cache();loader.prefetch(['2024-01-02'],[stock],[factor])
        pd.testing.assert_frame_equal(direct,loader.load_factors('2024-01-02',[stock],[factor]),check_exact=True)

@pytest.mark.parametrize('case', ['zero_quantiles','bool_quantiles','float_quantiles','zero_period','empty_periods','negative_lag','factor_series','factor_multiindex','bad_factor_dates','bad_factor_values','flat_prices','flat_returns','bad_return_values'])
def test_explicit_analysis_input_errors_raise(case):
    dates=pd.bdate_range('2024-01-02',periods=10);assets=list('ABCDEF')
    factors=pd.DataFrame(np.tile(np.arange(6.),(10,1)),index=dates,columns=assets)
    pricing=(factors+100).stack();pricing.index.names=['date','asset']
    returns=pd.DataFrame({'1D':np.arange(60.)/100},index=pricing.index)
    kwargs={'periods':(1,), 'n_quantiles':2,'execution_lag':1}
    if case=='zero_quantiles':kwargs['n_quantiles']=0
    if case=='bool_quantiles':kwargs['n_quantiles']=True
    if case=='float_quantiles':kwargs['n_quantiles']=2.5
    if case=='zero_period':kwargs['periods']=(0,)
    if case=='empty_periods':kwargs['periods']=()
    if case=='negative_lag':kwargs['execution_lag']=-1
    if case=='factor_series':factors=factors.stack()
    if case=='factor_multiindex':factors.index=pd.MultiIndex.from_product([['bad'],dates])
    if case=='bad_factor_dates':factors.index=['bad-date']*10
    if case=='bad_factor_values':factors=factors.astype(object);factors.iloc[0,0]='invalid'
    if case=='flat_prices':pricing=pricing.reset_index(drop=True)
    if case=='flat_returns':returns=returns.reset_index(drop=True)
    if case=='bad_return_values':returns=returns.astype(object);returns.iloc[0,0]='invalid'
    with pytest.raises((ValueError,TypeError)):
        AlphalensRunner(**kwargs).run(factors,pricing,returns,max_loss=1.)

@pytest.mark.parametrize('schema',[None,{}, {'date':'VARCHAR'},{'date':'VARCHAR','stock_code':'VARCHAR'}])
def test_minimal_key_table(schema,tmp_path):
    imp=DuckDBFactorImporter(str(tmp_path/'minimal.duckdb'))
    try:
        imp.init_factor_table(schema)
        assert [(x[0],x[1],x[3]) for x in imp.conn.execute('DESCRIBE factor_data').fetchall()]==[('stock_code','VARCHAR','PRI'),('date','VARCHAR','PRI')]
    finally:imp.close()

@pytest.mark.parametrize('existing_main',[False,True])
def test_table_existence_uses_target_namespace(tmp_path,existing_main):
    imp=DuckDBFactorImporter(str(tmp_path/'namespace.duckdb'));imp.connect()
    try:
        imp.conn.execute('CREATE SCHEMA other');imp.conn.execute('CREATE TABLE other.factor_data(i INT)')
        if existing_main:
            imp.conn.execute('CREATE TABLE main.factor_data(keep_me INT)')
            imp.conn.execute('INSERT INTO main.factor_data VALUES (7)')
        imp.init_factor_table({'date':'VARCHAR','alpha':'DOUBLE'})
        if existing_main:assert imp.conn.execute('SELECT * FROM main.factor_data').fetchall()==[(7,)]
        else:assert [r[0] for r in imp.conn.execute('DESCRIBE main.factor_data').fetchall()]==['stock_code','date','alpha']
        assert imp.conn.execute('SELECT count(*) FROM other.factor_data').fetchone()==(0,)
    finally:imp.close()

def test_uppercase_csv_rejected_before_database_creation(tmp_path):
    source=tmp_path/'csv';source.mkdir();(source/'A.CSV').write_text('date,alpha\n2024-01-02,1\n')
    imp=DuckDBFactorImporter(str(tmp_path/'new.duckdb'))
    with pytest.raises(ValueError,match='No factor CSV'):
        imp.run_import(str(source))
    assert not imp.db_path.exists()
