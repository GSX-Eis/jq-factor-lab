"""NERE-213: average zero-based ties, unchanged bucket boundaries."""
import numpy as np
import pandas as pd
import pytest
from factor_synthesis.pipeline.sample_weighting import _rank_array, compute_return_based_weights

@pytest.mark.parametrize('values,expected',[([2.,2.,1.,3.],[1.5,1.5,0.,3.]),([0.,0.,0.,0.],[1.5]*4),([3.,1.,1.,3.,2.],[3.5,.5,.5,3.5,2.])])
def test_average_zero_based_ranks(values,expected):
    np.testing.assert_array_equal(_rank_array(np.array(values)),expected)

@pytest.mark.parametrize('cross_sectional',[False,True])
def test_identical_labels_have_uniform_weights(cross_sectional):
    idx=pd.MultiIndex.from_product([pd.to_datetime(['2025-01-02','2025-01-03']),list('ABCD')])
    cfg={'sample_weight_method':'quantile','sample_weight_config':{'cross_sectional':cross_sectional}}
    np.testing.assert_array_equal(compute_return_based_weights(np.zeros(8),idx,cfg),np.ones(8))

@pytest.mark.parametrize('order',[[3,2,1,0],[1,0,2,3],[2,0,3,1]])
def test_partial_ties_are_stock_permutation_invariant(order):
    y=np.array([1.,1.,2.,3.]);idx=pd.MultiIndex.from_product([[pd.Timestamp('2025-01-02')],list('ABCD')])
    cfg={'sample_weight_method':'quantile','sample_weight_config':{'top_pct':.2,'bottom_pct':.2}}
    got=compute_return_based_weights(y[order],idx[order],cfg)
    by_stock=pd.Series(got,index=idx[order]).sort_index()
    np.testing.assert_array_equal(by_stock.to_numpy(),[2/7,2/7,4/7,20/7])
