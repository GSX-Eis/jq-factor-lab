"""Approved A270 contracts and separately approved NERE-194/197 metrics."""
from types import SimpleNamespace
from unittest.mock import Mock
import logging
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from factor_synthesis.factor_composer import FactorComposer, SynthesisConfig
from factor_synthesis.composite_factor_base import CompositeFactorBase


def composer(method='ridge', **kw):
    cfg=SynthesisConfig(method=method, training_window=10, min_observations=2, update_frequency=1,
                        scaler_type='none', outlier_threshold=0, ridge_cv_folds=2, **kw)
    dates=pd.date_range('2024-01-01',periods=6)
    x=pd.DataFrame(np.arange(18).reshape(6,3)+1.,index=dates,columns=list('ABC'))
    y=x**2/30
    frames={'x':x,'y':y,'missing':pd.DataFrame()}
    reader=SimpleNamespace(get_factor_values=lambda factor_name,**kw:frames[factor_name])
    return FactorComposer(cfg,reader),x,y


@pytest.mark.parametrize('name',['date','stock','target_return'])
def test_reserved_factor_rejected_before_read(name):
    c,_,y=composer(); c.factor_reader.get_factor_values=Mock(side_effect=AssertionError('read'))
    with pytest.raises(ValueError,match='reserved'):
        c.synthesize_factors(['x',name],y,'2024-01-01','2024-01-06',target_available_at=pd.DataFrame({col:y.index for col in y},index=y.index),trading_calendar=y.index)
    c.factor_reader.get_factor_values.assert_not_called()


def test_auto_pca_can_grow_on_later_window():
    c,_,_=composer('pca')
    rank1=np.array([[-3,-3],[-1,-1],[1,1],[3,3.]])
    rank2=np.array([[-1,-1],[-1,1],[1,-1],[1,1.]])
    c._train_model_window(rank1,np.arange(4.),dates=pd.date_range('2024-01-01',periods=4),available_at=pd.date_range('2024-01-01',periods=4)); assert c.model.n_components_==1
    c._train_model_window(rank2,np.arange(4.),dates=pd.date_range('2024-01-01',periods=4),available_at=pd.date_range('2024-01-01',periods=4)); assert c.model.n_components_==2


@pytest.mark.parametrize('names',[['missing','x','y'],['x','x','y']])
def test_importance_uses_actual_order(names):
    c,_,y=composer(); c.synthesize_factors(names,y,'2024-01-01','2024-01-06',target_available_at=pd.DataFrame({col:y.index for col in y},index=y.index),trading_calendar=y.index,composite_factor_name='one')
    assert c.get_feature_importance('one')==dict(zip(['x','y'],abs(c.model.coef_)))


@pytest.mark.parametrize('method',['ridge','pca'])
def test_diagnostics_belong_to_result_not_current_model(method):
    c,_,y=composer(method)
    c.synthesize_factors(['x','y'],y,'2024-01-01','2024-01-06',target_available_at=pd.DataFrame({col:y.index for col in y},index=y.index),trading_calendar=y.index,composite_factor_name='one')
    getter=c.get_feature_importance if method=='ridge' else c.get_explained_variance
    before=getter('one'); assert before
    c.synthesize_factors(['x'],y,'2024-01-01','2024-01-06',target_available_at=pd.DataFrame({col:y.index for col in y},index=y.index),trading_calendar=y.index,composite_factor_name='two')
    c.config.method='lightgbm' # old result retains its method too
    assert getter('one')==before


@pytest.mark.parametrize('method',['ridge','pca'])
def test_failed_training_does_not_borrow_old_snapshot(method):
    c,_,y=composer(method)
    c.synthesize_factors(['x','y'],y,'2024-01-01','2024-01-06',target_available_at=pd.DataFrame({col:y.index for col in y},index=y.index),trading_calendar=y.index,composite_factor_name='one')
    getter=c.get_feature_importance if method=='ridge' else c.get_explained_variance
    first=getter('one'); assert first
    c._train_model_window=lambda *args,**kwargs:None
    c.synthesize_factors(['x','y'],y,'2024-01-01','2024-01-06',target_available_at=pd.DataFrame({col:y.index for col in y},index=y.index),trading_calendar=y.index,composite_factor_name='failed')
    assert getter('failed')=={}
    assert getter('one')==first


@pytest.mark.parametrize('other',[[np.nan,3,2,1],[np.nan,3,2,np.nan]])
def test_input_correlations_align_keys_before_dropna(other):
    obj=object.__new__(CompositeFactorBase)
    obj.logger=logging.getLogger('test'); obj.input_factor_names=['x','y']; obj.start_date='2024-01-01';obj.end_date='2024-01-04';obj.all_stocks=['A']
    x=pd.DataFrame({'A':[1,2,3,np.nan]},index=pd.date_range('2024-01-01',periods=4))
    y=pd.DataFrame({'A':other},index=x.index)
    obj.factor_reader=SimpleNamespace(get_multiple_factors=lambda **kw:{'x':x,'y':y})
    expected=pd.DataFrame({'x':x['A'],'y':y['A']}).corr()
    assert_frame_equal(obj.get_input_factor_correlation(),expected,check_exact=True)


@pytest.mark.parametrize('rows',[
    [[2,1],[20,10]], # pooled positive, both days negative
    [[1,2],[20,10]], # std is sample std, no annualization
    [[1,1],[2,2]],   # constant cross sections excluded
    [[2,1],[np.nan,10]], # one valid day -> missing IR
])
def test_daily_ic_and_ir_contract(rows):
    c,_,_=composer();x=pd.DataFrame([[1,2],[10,20]],index=pd.date_range('2024-01-01',periods=2),columns=['A','B']);y=pd.DataFrame(rows,index=x.index,columns=x.columns)
    c.synthesis_results['test']={'factor_data':x}
    m=c.evaluate_performance('test',y)
    ics=[]
    for d in x.index:
        pair=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
        if len(pair)>=2 and all(pair.nunique()>1):ics.append(pair.iloc[:,0].corr(pair.iloc[:,1],method='spearman'))
    expected=float(np.mean(ics)) if ics else np.nan
    if np.isnan(expected): assert np.isnan(m['ic'])
    else: assert m['ic']==expected
    std=np.std(ics,ddof=1) if len(ics)>1 else np.nan
    if not std>0: assert np.isnan(m['ir'])
    else: assert m['ir']==expected/std
    assert m['ic_version']=='daily_cross_section_v1'
