"""A270 approved cache identity and SQL-value contracts, synthetic stores only."""
import duckdb
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from unittest.mock import patch
from calendar_test_support import seed_calendar
from factor_synthesis.core import factor_data_loader as m


@pytest.fixture
def store(tmp_path):
    c=duckdb.connect(':memory:'); seed_calendar(c)
    c.execute('CREATE TABLE factor_values(date VARCHAR,stock_code VARCHAR,factor_name VARCHAR,factor_value DOUBLE)')
    c.executemany('INSERT INTO factor_values VALUES (?,?,?,?)',[(d,'S',f,float(i+j)) for j,d in enumerate(['2024-01-01','2024-01-02','2024-01-03']) for i,f in enumerate(['A','B','a_b','c','a','b_c'],1)])
    c.execute('CREATE TABLE other_values AS SELECT date,stock_code,factor_name,factor_value+1000 AS factor_value FROM factor_values')
    bindings={'output_db':'fixture.duckdb','inputs':{},'outputs':{'factor_values':{'role':'output','table':'main.factor_values','version':'v1'},'other_values':{'role':'output','table':'main.other_values','version':'v2'}}}
    def call(names,table='factor_values',bound=True,cache=True):
        return m.load_factor_data(c,table,names,'2024-01-01','2024-01-03',config={'cache_settings':{'cache_dir':str(tmp_path),'use_parquet_cache':cache},'data_bindings':bindings if bound else {}},price_conn=c,price_table='prices')
    yield call,c,tmp_path
    c.close()


def test_bound_cache_distinguishes_selected_relation(store):
    call,c,p=store
    call(['A','B'])
    assert_frame_equal(call(['A','B'],'other_values'),call(['A','B'],'other_values',cache=False),check_exact=True)


def test_bound_name_list_is_structural(store):
    call,c,p=store
    call(['a_b','c'])
    assert_frame_equal(call(['a','b_c']),call(['a','b_c'],cache=False),check_exact=True)


@pytest.mark.parametrize('bound',[True,False])
def test_processed_cache_respects_order_while_raw_is_shared(store,bound):
    call,c,p=store
    call(['A','B'],bound=bound)
    with patch.object(m,'_read_factor_rows',side_effect=AssertionError('raw cache should hit')):
        reordered=call(['B','A'],bound=bound)
    assert_frame_equal(reordered,call(['B','A'],bound=bound,cache=False),check_exact=True)
    assert len([f for f in p.glob('*.parquet') if '_processed_' not in f.name])==1
    assert len([f for f in p.glob('*.parquet') if '_processed_' in f.name])==2


@pytest.mark.parametrize('fallback,pit',[(False,True),(False,False),(True,False)])
def test_quoted_stock_main_and_fallback(fallback,pit):
    c=duckdb.connect(':memory:');seed_calendar(c)
    try:
        c.execute('CREATE TABLE prices(date TIMESTAMP,stock_code VARCHAR,close DOUBLE,factor DOUBLE)')
        c.execute('CREATE TABLE factor_data(date TIMESTAMP,stock_code VARCHAR,close_price DOUBLE)')
        c.execute('INSERT INTO prices VALUES (?, ?,42,1)',['2024-01-02',"O'X"])
        c.execute('INSERT INTO factor_data VALUES (?, ?,42)',['2024-01-02',"O'X"])
        result=m.get_pricing_data(c,'missing' if fallback else 'prices',["O'X"],pd.DatetimeIndex(['2024-01-02']),future_periods=0,pit_enabled=pit)
        assert result.iloc[0]==42
        assert result.index.get_level_values('asset').tolist()==["O'X"]
    finally:c.close()
