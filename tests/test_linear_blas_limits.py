"""Coordinate-descent limits must preserve fits and restore caller BLAS state."""
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, Ridge
from threadpoolctl import threadpool_info, threadpool_limits

from factor_synthesis.qlib_adapter import qlib_linear_model as linear
from factor_synthesis.qlib_adapter.cv_metrics import CVMetricEvaluator


def blas_state():
    return {item['filepath']: item['num_threads'] for item in threadpool_info()
            if item['user_api'] == 'blas'}


@pytest.fixture(autouse=True)
def require_controllable_blas():
    if not blas_state():
        pytest.skip("This platform exposes no BLAS thread pool to threadpoolctl")


@pytest.mark.parametrize('estimator', [ElasticNet, ElasticNetCV, Lasso, LassoCV])
@pytest.mark.parametrize('fail', [False, True])
def test_limits_only_during_coordinate_descent_fit(monkeypatch, estimator, fail):
    model = estimator()
    X, y, weights = object(), object(), object()
    params = model.get_params()

    def observed(self, actual_X, actual_y, *, sample_weight):
        assert self is model and actual_X is X and actual_y is y
        assert sample_weight is weights
        assert self.get_params() == params
        assert blas_state() and set(blas_state().values()) == {1}
        if fail:
            raise ValueError('fit failed')
        return self

    monkeypatch.setattr(estimator, 'fit', observed)
    with threadpool_limits(limits=3, user_api='blas'):
        before = blas_state()
        if fail:
            with pytest.raises(ValueError, match='fit failed'):
                linear._fit_linear_estimator(model, X, y, sample_weight=weights)
        else:
            assert linear._fit_linear_estimator(model, X, y, sample_weight=weights) is model
        assert blas_state() == before


def test_other_estimator_keeps_callers_threads(monkeypatch):
    with threadpool_limits(limits=3, user_api='blas'):
        before = blas_state()
        def observed(self, X, y, *, sample_weight):
            assert blas_state() == before
            return self
        monkeypatch.setattr(Ridge, 'fit', observed)
        model = Ridge()
        assert linear._fit_linear_estimator(model, None, None) is model
        assert blas_state() == before


def test_weighted_fit_coefficients_predictions_and_iterations_are_unchanged():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(1600, 7))
    X[:, 1:] = X[:, [0]] * .8 + X[:, 1:] * .2
    y = X @ rng.normal(size=7) + rng.normal(size=len(X)) * .1
    weights = np.linspace(.5, 1.5, len(X))
    params = dict(alpha=.001, l1_ratio=.3, max_iter=15000, random_state=42)
    with threadpool_limits(limits=8, user_api='blas'):
        old = ElasticNet(**params).fit(X, y, sample_weight=weights)
        new = linear._fit_linear_estimator(ElasticNet(**params), X, y, sample_weight=weights)
        np.testing.assert_array_equal(new.coef_, old.coef_)
        assert new.intercept_ == old.intercept_
        assert new.n_iter_ == old.n_iter_
        np.testing.assert_array_equal(new.predict(X), old.predict(X))


def test_threaded_cv_restores_limits_and_matches_serial():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(1200, 5))
    y = X @ rng.normal(size=5) + rng.normal(size=len(X)) * .2
    index = pd.MultiIndex.from_product([pd.bdate_range('2020-01-01', periods=40), range(30)])
    splits = [(np.arange(600), np.arange(600, 900)), (np.arange(900), np.arange(900, 1200))]
    config = dict(model_type='elasticnet', param_search_metric='ic', cv_backend='threading',
                  lasso_max_iter=15000, cv_warm_start=True)
    logger = logging.getLogger(__name__)
    def score(jobs):
        cfg = dict(config, cv_n_jobs=jobs)
        wrapper = linear.QlibLinearWrapper(cfg, logger)
        return wrapper._score_param_grid_cv(X, y, index, None, splits,
                                           [(.01, .3), (.001, .3), (.01, .7), (.001, .7)],
                                           CVMetricEvaluator('ic', cfg, logger))
    with threadpool_limits(limits=3, user_api='blas'):
        before = blas_state()
        serial = score(1)
        parallel = score(4)
        assert parallel == serial
        assert blas_state() == before


def test_threaded_dispatch_failure_restores_limits(monkeypatch):
    def fail_dispatch(**kwargs):
        assert set(blas_state().values()) == {1}
        raise RuntimeError('dispatch failed')

    monkeypatch.setattr(linear, 'Parallel', fail_dispatch)
    wrapper = linear.QlibLinearWrapper(dict(model_type='elasticnet', cv_backend='threading', cv_n_jobs=2))
    index = pd.MultiIndex.from_product([pd.date_range('2020-01-01', periods=4), ['A']])
    with threadpool_limits(limits=3, user_api='blas'):
        before = blas_state()
        with pytest.raises(RuntimeError, match='dispatch failed'):
            wrapper._score_param_grid_cv(np.ones((4, 2)), np.ones(4), index, None,
                                        [(np.array([0, 1]), np.array([2, 3]))],
                                        [(.001, .3), (.001, .7)], None)
        assert blas_state() == before
