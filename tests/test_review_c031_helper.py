"""Helper side contract; synthetic processor models the agreed line C signature."""
import importlib.util,os,sys,types
from unittest.mock import patch
import numpy as np,pandas as pd,pytest
from pandas.testing import assert_series_equal
from factor_synthesis.pipeline import preprocessing as current
helper=current

@pytest.mark.parametrize('eps',[1e-8,.1])
@pytest.mark.parametrize('rank,standardize',[(False,True),(False,False),(True,True)])
def test_helper_forwards_eps_and_preserves_policy(eps,rank,standardize):
    calls=[]
    class Processor:
        def __init__(self,*,cs_demean,cs_rank,cs_standardize,eps,fields_group):
            calls.append((cs_demean,cs_rank,cs_standardize,eps,fields_group));self.opts=(cs_demean,cs_rank,cs_standardize,eps)
        def __call__(self,df):
            # Use the existing fallback arithmetic as the independent helper oracle.
            series=df.LABEL0.rename_axis(index={'datetime':'date'})
            result=current._apply_target_cs_preprocessing_fallback(series,*self.opts,None)
            return result.rename_axis(index={'date':'datetime'}).to_frame('LABEL0')
    idx=pd.MultiIndex.from_tuples([(pd.Timestamp('2025-01-02'),'A'),(pd.Timestamp('2025-01-03'),'A'),(pd.Timestamp('2025-01-03'),'B')],names=['date','stock_code'])
    y=pd.Series([.2,.1,.3],index=idx,name='label')
    stub=types.ModuleType('factor_synthesis.qlib_adapter.processors');stub.TargetCSProcessor=Processor
    with patch.dict(sys.modules,{'factor_synthesis.qlib_adapter.processors':stub}):r=helper.apply_target_cs_preprocessing(y,False,rank,standardize,eps)
    expected=current._apply_target_cs_preprocessing_fallback(y,False,rank,standardize,eps,None).rename('LABEL0')
    if not (rank or standardize):expected=y
    assert_series_equal(r,expected,check_exact=True)
    if rank or standardize:assert calls==[(False,rank,standardize,eps,'label')]
    else:assert calls==[]
