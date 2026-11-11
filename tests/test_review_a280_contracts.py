"""Approved A280 boundaries, using only synthetic frames/in-memory databases."""
import builtins
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from factor_synthesis.core import factor_data_loader as loader
from factor_synthesis.core import factor_ops as ops
from factor_synthesis.core import feature_preprocessor as fp
from factor_synthesis.core.result_persistence import check_table_exists
from factor_synthesis.pipeline import preprocessing as pp
from calendar_test_support import seed_calendar
from utils.trading_calendar import load_trading_calendar


@pytest.fixture
def price_db():
    with duckdb.connect(':memory:') as c:
        seed_calendar(c,'2025-01-01','2025-01-31')
        c.execute('CREATE TABLE prices(date TIMESTAMP, stock_code VARCHAR, open DOUBLE, close DOUBLE, factor DOUBLE, natural_log_of_market_cap DOUBLE)')
        c.executemany('INSERT INTO prices VALUES (?,?,?,?,?,?)',[(d,"A'B",10.+i,10.+i,1.+i/10,3.+i) for i,d in enumerate(['2025-01-02','2025-01-03','2025-01-06'])])
        yield c


@pytest.mark.parametrize('name',['get_open_price_data','get_complete_price_data','get_factor_ratio_data','load_market_cap_data','calculate_forward_returns','raw_fallback'])
def test_stock_literal_quotes(price_db,name):
    if name=='load_market_cap_data':
        result=loader.load_market_cap_data(price_db,'prices','2025-01-02','2025-01-06',{"A'B"},trading_calendar=load_trading_calendar(price_db))
    elif name in {'calculate_forward_returns','raw_fallback'}:
        idx=pd.MultiIndex.from_product([pd.to_datetime(['2025-01-02']),["A'B"]],names=['date','stock_code'])
        result=ops.calculate_forward_returns(pd.DataFrame({'x':[1.]},index=idx),price_db,'missing' if name=='raw_fallback' else 'prices','prices',period=1,pit_enabled=name!='raw_fallback')
    else:result=getattr(loader,name)(price_db,'prices',["A'B"],'2025-01-02','2025-01-06')
    assert result is not None and not result.empty


@pytest.mark.parametrize('failure',['missing_table','duplicate_key','closed_connection'])
def test_market_cap_errors_propagate(price_db,failure):
    table='prices'
    if failure=='missing_table':table='missing'
    if failure=='duplicate_key':price_db.execute('INSERT INTO prices SELECT * FROM prices LIMIT 1')
    calendar=load_trading_calendar(price_db)
    if failure=='closed_connection':price_db.close()
    with pytest.raises(Exception):loader.load_market_cap_data(price_db,table,'2025-01-02','2025-01-06',trading_calendar=calendar)


@pytest.mark.parametrize('key',['feature_lag_trading_days','embargo_trading_days'])
@pytest.mark.parametrize('value',[True,1.9])
def test_label_parameters_not_truncated(key,value):
    with pytest.raises(ValueError):ops.resolve_label_definition({'mode':'next_rebalance_open',key:value})


def test_bare_table_ascii_case_and_real_errors():
    with duckdb.connect(':memory:') as c:
        c.execute('CREATE TABLE MiXeD(x INTEGER)')
        assert check_table_exists(c,'mixed')
        assert not check_table_exists(c,'main.mixed')
        assert not check_table_exists(c,'absent')
        c.execute('CREATE TABLE "Ä"(x INTEGER)')
        assert not check_table_exists(c,'ä')
    with pytest.raises(Exception):check_table_exists(c,'mixed')


def frame():
    idx=pd.MultiIndex.from_product([pd.to_datetime(['2025-01-02','2025-01-03']),range(12)],names=['date','stock_code'])
    return pd.DataFrame({'a':np.tile(np.arange(12.),2),'b':np.tile(np.arange(12.)[::-1],2)},index=idx)


@pytest.mark.parametrize('name',['cross_sectional_zscore','cross_sectional_winsorize_zscore','cross_sectional_standardize','cross_sectional_binning'])
def test_native_runtime_errors_propagate(name):
    with patch.object(pd.DataFrame,'groupby',side_effect=RuntimeError('injected group failure')), patch.object(pd.Series,'groupby',side_effect=RuntimeError('injected group failure')):
        with pytest.raises(RuntimeError,match='injected'):getattr(fp,name)(frame())


@pytest.mark.parametrize('mode,method',[('zscore','cross_sectional_zscore'),('winsorize_zscore','cross_sectional_winsorize_zscore'),('binning','cross_sectional_binning')])
def test_native_router_does_not_return_partial_processing(mode,method):
    with patch.object(fp,method,side_effect=[frame()['a']*2,RuntimeError('second factor failed')]):
        with pytest.raises(RuntimeError,match='second factor'):ops.apply_feature_preprocessing(frame(),{'ALL':mode})


@pytest.mark.parametrize('value',[0,-1,2.5,True,'5'])
def test_invalid_bin_count_rejected(value):
    with pytest.raises((ValueError,TypeError)):fp.cross_sectional_binning(frame(),n_bins=value)


@pytest.mark.parametrize('name',['cross_sectional_zscore','cross_sectional_winsorize_zscore'])
def test_bad_percentile_is_not_raw_fallback(name):
    with pytest.raises(ValueError):getattr(fp,name)(frame(),percentile=.5)


@pytest.mark.parametrize('fallback',[False,True])
@pytest.mark.parametrize('decay,half',[(-.5,0.),(np.nan,0.),(.9,np.nan),(.9,np.inf)])
def test_only_used_weight_parameters_validated(fallback,decay,half):
    original=builtins.__import__
    def unavailable(name,*args,**kwargs):
        if name=='factor_synthesis.qlib_adapter.processors':raise ImportError('controlled')
        return original(name,*args,**kwargs)
    with patch('builtins.__import__',side_effect=unavailable if fallback else original):
        with pytest.raises(ValueError):pp.compute_sample_weights(frame(),decay,half)


@pytest.mark.parametrize('length',[0,1,3])
def test_residual_single_date_and_empty_series(length):
    idx=pd.MultiIndex.from_product([pd.to_datetime(['2025-01-02']),range(3)],names=['date','stock_code'])[:length]
    y=pd.Series([-1.,-2.,3.][:length],idx,name='target');x=pd.Series([-1.,0.,1.][:length],idx)
    result=pp.residualize_target_by_factor(y,x,min_samples=3)
    expected=pd.Series(([1.,-2.,1.] if length==3 else [0.] if length else []),index=idx,name='target',dtype=float)
    assert_series_equal(result,expected,check_exact=True)
    assert isinstance(bool(result.notna().any()),bool)

@pytest.mark.parametrize('bad',[0.,-1.,np.inf,-np.inf])
def test_invalid_observed_ratio_factor(price_db,bad):
    price_db.execute("UPDATE prices SET stock_code='A'")
    price_db.execute("UPDATE prices SET factor=? WHERE date='2025-01-03'",[bad])
    with pytest.raises(ValueError):loader.get_factor_ratio_data(price_db,'prices',['A'],'2025-01-02','2025-01-06')

@pytest.mark.parametrize('bad',[0.,-1.,np.inf,-np.inf])
@pytest.mark.parametrize('entry',['scheduled','training','fallback'])
def test_invalid_observed_label_price(price_db,bad,entry):
    price_db.execute("UPDATE prices SET stock_code='A'")
    idx=pd.MultiIndex.from_tuples([(pd.Timestamp('2025-01-02'),'A')],names=['date','stock_code'])
    factors=pd.DataFrame({'x':[1.]},index=idx)
    if entry=='scheduled':
        prices=pd.DataFrame({'A':[bad,11.]},index=pd.to_datetime(['2025-01-02','2025-01-03']))
        with pytest.raises(ValueError):ops.calculate_scheduled_price_returns(factors,prices,[('2025-01-02','2025-01-03')])
    else:
        price_db.execute("UPDATE prices SET close=? WHERE date='2025-01-02'",[bad])
        with pytest.raises(ValueError):ops.calculate_forward_returns(factors,price_db,'missing' if entry=='fallback' else 'prices','prices',period=1,pit_enabled=entry=='training')

@pytest.mark.parametrize('kind',['short','bins_exceed_samples','constant','one_bin','all_missing','qcut_value_error'])
def test_binning_legal_rank_fallback(kind):
    x=frame()['a'].iloc[:12].copy();kwargs={}
    if kind=='short':x=x.iloc[:3]
    if kind=='bins_exceed_samples':kwargs['n_bins']=20
    if kind=='constant':x[:]=3.
    if kind=='one_bin':kwargs['n_bins']=1
    if kind=='all_missing':x[:]=np.nan
    if kind=='qcut_value_error':
        with patch.object(pd,'qcut',side_effect=ValueError('duplicate edges')):result=fp.cross_sectional_binning(x,**kwargs)
    else:result=fp.cross_sectional_binning(x,**kwargs)
    assert_series_equal(result,x.groupby(level='date').rank(pct=True),check_exact=True)

@pytest.mark.parametrize('decay',[np.nan,-1.,np.inf])
def test_half_life_ignores_unused_decay(decay):
    np.testing.assert_array_equal(pp.compute_sample_weights(frame(),decay,3.),pp.compute_sample_weights(frame(),.9,3.))
