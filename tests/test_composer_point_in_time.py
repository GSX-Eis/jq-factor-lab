"""NERE-196: hand-computable timing, scheduling and purged validation contracts."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from factor_synthesis.factor_composer import FactorComposer, SynthesisConfig
from factor_synthesis.composite_factor_base import CompositeFactorBase


def fixture(method='ridge', **changes):
    cfg = dict(method=method, training_window=4, min_observations=2,
               update_frequency=1, outlier_threshold=0, scaler_type='none',
               ridge_cv_folds=2, ridge_n_alphas=1, lasso_cv_folds=2,
               lasso_n_alphas=1, lgb_n_jobs=1, lgb_min_child_samples=1)
    cfg.update(changes)
    dates = pd.DatetimeIndex(['2024-01-04','2024-01-05','2024-01-08',
                              '2024-01-10','2024-01-11','2024-01-15','2024-01-16'])
    x = pd.DataFrame(np.arange(14.).reshape(7,2), index=dates, columns=['A','B'])
    y = x / 2 + 1
    available = pd.DataFrame({c:dates for c in x.columns}, index=dates)
    reader = SimpleNamespace(get_factor_values=lambda **kw:x)
    return FactorComposer(SynthesisConfig(**cfg),reader), x, y, available


def run(c,x,y,a,name='result'):
    return c.synthesize_factors(['x'], y, str(x.index[0].date()), str(x.index[-1].date()),
        composite_factor_name=name, target_available_at=a, trading_calendar=x.index)


class MeanModel:
    def __init__(self,y):
        self.mean=float(np.mean(y)); self.coef_=np.ones(1)
    def predict(self,x): return np.full(len(x), self.mean)


def record_fits(c, failures=()):
    calls=[]
    def fit(x,y,*,dates,available_at):
        calls.append((x.copy(),y.copy(),pd.DatetimeIndex(dates),pd.DatetimeIndex(available_at)))
        if len(calls) in failures: return None
        c.model=MeanModel(y); c.scaler=None
        return c.model
    c._train_model_window=fit
    return calls


def test_required_timing_inputs_before_read():
    c,x,y,a=fixture(); c.factor_reader.get_factor_values=Mock(side_effect=AssertionError('read'))
    with pytest.raises(TypeError): c.synthesize_factors(['x'],y,'2024-01-04','2024-01-16')
    with pytest.raises(ValueError,match='calendar'):
        c.synthesize_factors(['x'],y,'2024-01-04','2024-01-16',target_available_at=a,trading_calendar=[])
    c.factor_reader.get_factor_values.assert_not_called()


def test_maturity_boundary_future_labels_and_missing_prediction_targets():
    c,x,y,a=fixture(); calls=record_fits(c)
    a.iloc[0]=x.index[2]            # equality accepted on Jan 8
    a.iloc[1,1]=x.index[3]          # not yet available on Jan 8
    y.iloc[2:]=np.nan              # still predict these feature rows
    out=run(c,x,y,a)
    assert out.index[0]==x.index[2]
    assert out.loc[x.index[2],'A']==np.mean([1,1.5,2])
    assert calls[0][2].nunique()==2 and len(calls[0][1])==3
    other,_,_,_=fixture(); record_fits(other)
    changed=y.copy();changed.iloc[1,1]=9999
    assert run(other,x,changed,a).loc[x.index[2],'A']==out.loc[x.index[2],'A']


def test_pca_ignores_labels_and_maturity_and_predicts_tail():
    c,x,y,a=fixture('pca'); before=run(c,x,y,a)
    other,_,_,_=fixture('pca')
    after=run(other,x,y*float('nan'),a.mask(a.notna()),'other')
    assert_frame_equal(before,after,check_exact=True)
    assert before.index[-1]==x.index[-1]


def test_composite_tracks_actual_exit_observation_without_changing_labels():
    _,x,_,_=fixture();x.iloc[3,1]=np.nan
    obj=object.__new__(CompositeFactorBase);obj.logger=logging.getLogger('timing')
    obj.synthesis_config=SynthesisConfig(prediction_window=2)
    obj.get_price_data=lambda **kw:x+1
    actual=obj._get_target_returns()
    assert_frame_equal(actual,((x+1).shift(-2)/(x+1)-1).iloc[:-2],check_exact=True)
    assert obj._target_available_at.iloc[0,0]==x.index[2]
    assert pd.isna(obj._target_available_at.iloc[1,1])
    assert obj._trading_calendar.equals(x.index)


def test_days_and_rows_are_separate_and_window_is_not_divided_by_five():
    c,x,y,a=fixture(training_window=3,min_observations=4);calls=record_fits(c)
    out=run(c,x,y,a)
    assert out.index[0]==x.index[2]
    assert list(calls[-1][2].unique())==list(x.index[-4:-1])
    assert len(calls[-1][1])==6


@pytest.mark.parametrize('frequency,expected',[ (1,5), (2,3), (20,1)])
def test_update_frequency_uses_calendar(frequency,expected):
    c,x,y,a=fixture(update_frequency=frequency);calls=record_fits(c)
    run(c,x,y,a);assert len(calls)==expected


def test_missing_feature_day_does_not_compress_clock():
    c,x,y,a=fixture(update_frequency=2);x.iloc[3]=np.nan
    calls=record_fits(c);out=run(c,x,y,a)
    assert x.index[3] not in out.index
    assert len(calls)==3
    history=c.synthesis_results['result']['training_history']
    assert [h['date'] for h in history if h['retrained']]==list(x.index[[2,4,6]])


def test_failed_refits_keep_successful_model_until_next_success(caplog):
    c,x,y,a=fixture(update_frequency=2);calls=record_fits(c,failures=(2,))
    with caplog.at_level(logging.WARNING): out=run(c,x,y,a)
    assert out.loc[x.index[4],'A']==out.loc[x.index[2],'A']
    assert out.loc[x.index[5],'A']==out.loc[x.index[2],'A']
    assert out.loc[x.index[6],'A']!=out.loc[x.index[2],'A']
    h=c.synthesis_results['result']['training_history']
    assert [r['carried_days'] for r in h if r['predicted']]==[0,1,2,3,0]
    assert '拟合失败，沿用' in caplog.text
    assert len(calls)==3


def test_initial_failure_has_no_prediction_or_unscheduled_retry():
    c,x,y,a=fixture(update_frequency=2);calls=record_fits(c,failures=(1,))
    out=run(c,x,y,a)
    assert out.index[0]==x.index[4] and len(calls)==3


@pytest.mark.parametrize('value',[0,-1,1.5,True])
def test_invalid_frequency_rejected(value):
    with pytest.raises(ValueError,match='update_frequency'): fixture(update_frequency=value)


def test_purged_folds_use_complete_dates_and_shared_equality_rule():
    from factor_synthesis.factor_composer import _labels_available
    c,x,y,a=fixture()
    dates=x.index.repeat(2); available=dates.to_numpy().copy()
    available[0]=x.index[3];available[1]=x.index[3]+pd.Timedelta(seconds=1)
    folds=c._validation_splits(dates,pd.DatetimeIndex(available))
    for train,val in folds:
        cutoff=dates[val].min()
        assert dates[train].max()<cutoff
        assert _labels_available(pd.DatetimeIndex(available)[train],cutoff).all()
        assert set(dates[train]).isdisjoint(set(dates[val]))
    train,val=folds[0]
    assert 0 in train and 1 not in train


def test_no_available_validation_fold_fails_without_mutating_pair():
    c,x,y,a=fixture(); old_model=c.model;old_scaler=c.scaler
    result=c._train_model_window(x.iloc[:3].values,y.iloc[:3,0].values,
        dates=x.index[:3],available_at=pd.DatetimeIndex([x.index[-1]]*3))
    assert result is None and c.model is old_model and c.scaler is old_scaler


def test_alpha_tie_uses_first_grid_item_and_refits_all_rows():
    c,x,y,a=fixture(ridge_n_alphas=3,ridge_alpha_range=(.3,30))
    X=np.zeros((8,1)); Y=np.arange(8.)*0
    dates=pd.date_range('2024-01-01',periods=4).repeat(2)
    model=c._train_model_window(X,Y,dates=dates,available_at=dates)
    assert model.alpha==np.logspace(np.log10(.3),np.log10(30),3)[0]
    assert model.n_features_in_==1 and model.intercept_==0


def test_lgb_validation_scaler_is_train_only_and_full_refit_uses_best_iteration(monkeypatch):
    import factor_synthesis.factor_composer as module
    records=[]
    class FakeLGB:
        def __init__(self,**params): self.params=params
        def get_params(self,deep=True): return self.params
        def set_params(self,**params): self.params.update(params);return self
        def fit(self,X,y,**kw):
            records.append((X.copy(),kw));self.best_iteration_=3
            return self
    monkeypatch.setattr(module.lgb,'LGBMRegressor',FakeLGB)
    c,x,y,a=fixture('lightgbm',scaler_type='standard',lgb_validation_fraction=.25)
    X=np.array([[0.],[2.],[0.],[2.],[0.],[2.],[100.],[102.]])
    dates=pd.date_range('2024-01-01',periods=4).repeat(2)
    model=c._train_model_window(X,np.arange(8.),dates=dates,available_at=dates)
    np.testing.assert_array_equal(records[0][0],np.array([[-1.],[1.]]*3))
    assert records[0][1]['eval_set'][0][0][0,0]==99
    assert len(records[1][0])==8 and model.params['n_estimators']==3
    assert c.scaler.mean_[0]==X.mean()


def test_auto_pca_dimension_can_expand_and_failed_fit_keeps_scaler_pair():
    c,_,_,_=fixture('pca',scaler_type='standard')
    dates=pd.date_range('2024-01-01',periods=4)
    c._train_model_window(np.array([[-3,-3],[-1,-1],[1,1],[3,3.]]),np.zeros(4),dates=dates,available_at=dates)
    assert c.model.n_components_==1
    c._train_model_window(np.array([[-1,-1],[-1,1],[1,-1],[1,1.]]),np.zeros(4),dates=dates,available_at=dates)
    assert c.model.n_components_==2
    pair=(c.model,c.scaler)
    assert c._train_model_window(np.array([[np.nan,1.]]),np.zeros(1),dates=dates[:1],available_at=dates[:1]) is None
    assert (c.model,c.scaler)==pair


@pytest.mark.parametrize('method',['ridge','lasso','pca','lightgbm'])
def test_real_models_produce_tail_predictions_without_observing_future_y(method):
    c,x,y,a=fixture(method,update_frequency=1,lgb_n_estimators=5)
    a.iloc[2:]=x.index[-1]+pd.Timedelta(days=1)
    # Three full historical dates are needed for the two-fold linear validation.
    a.iloc[2]=x.index[3]
    first=run(c,x,y,a)
    other,_,_,_=fixture(method,update_frequency=1,lgb_n_estimators=5)
    changed=y.copy();changed.iloc[3:]=-9999
    second=run(other,x,changed,a)
    assert not first.empty
    assert_frame_equal(first,second,check_exact=True)


def test_pca_empty_label_frame_and_legal_metadata_like_factor_name():
    c,x,_,_=fixture('pca')
    out=c.synthesize_factors(['available_at'],pd.DataFrame(),str(x.index[0].date()),
        str(x.index[-1].date()),target_available_at=pd.DataFrame(),trading_calendar=x.index)
    assert not out.empty


def test_base_exception_in_fit_does_not_install_candidate(monkeypatch):
    from sklearn.decomposition import PCA
    c,x,y,a=fixture('pca',scaler_type='standard');pair=(c.model,c.scaler)
    def interrupted(*args,**kwargs): raise KeyboardInterrupt('stop')
    monkeypatch.setattr(PCA,'fit',interrupted)
    with pytest.raises(KeyboardInterrupt):
        c._train_model_window(x.values,y.iloc[:,0].values,dates=x.index,available_at=x.index)
    assert (c.model,c.scaler)==pair


def test_auto_pca_expands_inside_calendar_windows():
    c,x,y,a=fixture('pca',training_window=2)
    second=x.copy()
    second.iloc[2:]=np.tile([1.,-1.],(5,1))
    c.factor_reader=SimpleNamespace(get_factor_values=lambda factor_name,**kw: {'a':x,'b':second}[factor_name])
    counts=[];original=c._train_model_window
    def fit(*args,**kwargs):
        model=original(*args,**kwargs)
        counts.append(model.n_components_)
        return model
    c._train_model_window=fit
    result=c.synthesize_factors(['a','b'],y,str(x.index[0].date()),str(x.index[-1].date()),
        target_available_at=a,trading_calendar=x.index)
    assert counts[0]==1 and 2 in counts[1:]
    assert result.index[0]==x.index[2]


def test_validation_targets_cannot_affect_training_fold_outlier_filter(monkeypatch):
    from sklearn.linear_model import Ridge
    seen=[];original=Ridge.fit
    def fit(self,X,y,*args,**kwargs):
        seen.append(X[:,0].tolist())
        return original(self,X,y,*args,**kwargs)
    monkeypatch.setattr(Ridge,'fit',fit)
    c,_,_,_=fixture(outlier_threshold=1)
    X=np.arange(8.).reshape(-1,1);y=np.array([0.,0.,0.,10.,1000.,1000.,1000.,1000.])
    dates=pd.date_range('2024-01-01',periods=4).repeat(2)
    assert c._train_model_window(X,y,dates=dates,available_at=dates) is not None
    # First train block: y=[0,0,0,10], mean2.5/std4.33 => remove 10.
    # The 1000 values in future validation blocks must not affect this filter.
    assert seen[0]==[0.,1.,2.]


def test_composite_diagnostics_expose_the_results_training_history():
    c,x,y,a=fixture(update_frequency=2);record_fits(c,failures=(2,))
    out=run(c,x,y,a)
    obj=object.__new__(CompositeFactorBase)
    obj._synthesis_completed=True;obj._synthesized_factor_df=out
    obj.synthesis_config=c.config;obj.input_factor_names=['x']
    obj.factor_name='result';obj.factor_composer=c
    diagnostic=obj.get_synthesis_diagnostics()
    assert diagnostic['training_history'][4]['carried_days']==2
    assert diagnostic['training_history'][4]['training_date']==x.index[2]
