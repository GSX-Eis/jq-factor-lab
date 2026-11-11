"""Equivalence tests for the warm-start + joblib CV grid scorer.

These guard the optimisation in ``QlibLinearWrapper._score_param_grid_cv`` that
replaced the legacy per-(alpha, l1, fold) cold-start loops in
``_rerank_top_alphas`` and ``_fit_with_custom_cv``.

Contract:
1. Serial cold-start (cv_warm_start=false, cv_n_jobs=1) must reproduce the exact
   per-pair CV scores of an independent reference implementation.
2. Parallel cold-start (cv_n_jobs>1) must equal serial cold-start bit-for-bit.
3. Warm-start (cv_warm_start=true) must select the SAME best (alpha, l1_ratio)
   and produce numerically-equivalent scores (convexity => same optimum up to
   solver tolerance), including with a non-trivial ``sample_weight``.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_mod = importlib.import_module("factor_synthesis.qlib_adapter.qlib_linear_model")
QlibLinearWrapper = _mod.QlibLinearWrapper
CVMetricEvaluator = importlib.import_module(
    "factor_synthesis.qlib_adapter.cv_metrics"
).CVMetricEvaluator

LOGGER = logging.getLogger("test_warmstart")


def test_deflated_sharpe_uses_shared_custom_cv():
    assert "deflated_sharpe" in QlibLinearWrapper.CUSTOM_CV_METRICS


def test_custom_cv_fails_closed_without_finite_scores(monkeypatch):
    X, y, index = _make_panel()
    wrapper = QlibLinearWrapper(
        {
            **_config(warm_start=False, n_jobs=1),
            "_label_exit_dates": {d: d + pd.Timedelta(days=1) for d in index.get_level_values(0).unique()},
            "model_type": "ridge",
            "param_search_metric": "deflated_sharpe",
            "lasso_alpha_range": [0.001, 0.01],
            "lasso_cv_folds": 3,
            "factor_count_grid": None,
        },
        logger=LOGGER,
    )
    monkeypatch.setattr(wrapper, "_score_param_grid_cv", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="No finite deflated_sharpe CV scores"):
        wrapper._fit_with_custom_cv(X, y, index)


def test_refit_estimator_regularization_params_are_not_reported_as_defaults():
    model = ElasticNet(alpha=0.003, l1_ratio=0.7)
    read_param = _mod._fitted_regularization_param

    assert read_param(model, "alpha", 0.0) == 0.003
    assert read_param(model, "l1_ratio", 1.0) == 0.7


def _make_panel(n_dates: int = 40, n_stocks: int = 30, n_features: int = 6, seed: int = 7):
    """Synthetic panel with a MultiIndex (datetime, instrument) and a signal."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    stocks = [f"S{i:03d}" for i in range(n_stocks)]
    index = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])

    n = len(index)
    X = rng.normal(size=(n, n_features))
    true_beta = np.array([0.8, -0.5, 0.3, 0.0, 0.0, 0.1])[:n_features]
    y = X @ true_beta + rng.normal(scale=0.5, size=n)
    return X, y, index


def _config(warm_start: bool, n_jobs: int) -> dict:
    return {
        "model_type": "elasticnet",
        "lasso_max_iter": 20000,
        "positive": False,
        "random_state": 42,
        "l1_ratio_range": [0.3, 0.7],
        "param_search_metric": "ic",
        "top_k_size": 5,
        "target_period": 5,
        "cv_warm_start": warm_start,
        "cv_n_jobs": n_jobs,
        "cv_backend": "loky",
    }


def _score(config: dict, X, y, index, cv_splits, param_pairs):
    wrapper = QlibLinearWrapper(config, logger=LOGGER)
    wrapper.model_type = config["model_type"]
    evaluator = CVMetricEvaluator(
        config["param_search_metric"], config, LOGGER, n_trials=len(param_pairs)
    )
    return wrapper._score_param_grid_cv(
        X, y, index, None, cv_splits, param_pairs, evaluator
    )


def _reference_scores(config, X, y, index, cv_splits, param_pairs, sample_weight=None):
    """Independent cold-start reference (no warm start, no joblib)."""
    date_groups = index.get_level_values(0).values
    evaluator = CVMetricEvaluator(
        config["param_search_metric"], config, LOGGER, n_trials=len(param_pairs)
    )
    out: dict[tuple[float, float], float] = {}
    for alpha, l1 in param_pairs:
        scores = []
        for tr, va in cv_splits:
            model = ElasticNet(
                alpha=alpha, l1_ratio=l1, max_iter=config["lasso_max_iter"],
                tol=1e-4, positive=False, random_state=42,
            )
            sw = sample_weight[tr] if sample_weight is not None else None
            model.fit(X[tr], y[tr], sample_weight=sw)
            pred = model.predict(X[va])
            s = evaluator.evaluate(pred, y[va], date_groups[va])
            if not np.isnan(s) and s > float("-inf"):
                scores.append(s)
        if scores:
            out[(alpha, l1)] = float(np.mean(scores))
    return out


def _splits(index, n_splits: int = 4):
    dates = index.get_level_values(0)
    unique_dates = np.array(sorted(dates.unique()))
    folds = np.array_split(unique_dates, n_splits)
    splits = []
    for f in folds:
        val_mask = np.asarray(dates.isin(f))
        train_mask = ~val_mask
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return splits


def _best(scores: dict, param_pairs):
    best_key, best_val = None, -np.inf
    for pair in param_pairs:
        entry = scores.get(pair)
        if entry is None:
            continue
        val = entry["score"] if isinstance(entry, dict) else entry
        if val > best_val:
            best_val, best_key = val, pair
    return best_key


def test_cold_start_serial_matches_reference():
    X, y, index = _make_panel()
    cv_splits = _splits(index)
    param_pairs = [(a, l1) for a in [1e-3, 1e-2, 1e-1] for l1 in [0.3, 0.7]]

    cold = _score(_config(warm_start=False, n_jobs=1), X, y, index, cv_splits, param_pairs)
    ref = _reference_scores(
        _config(warm_start=False, n_jobs=1), X, y, index, cv_splits, param_pairs
    )

    assert set(cold.keys()) == set(ref.keys())
    for key in ref:
        np.testing.assert_allclose(cold[key]["score"], ref[key], rtol=1e-9, atol=1e-12)


def test_parallel_matches_serial_cold_start():
    X, y, index = _make_panel(seed=11)
    cv_splits = _splits(index)
    param_pairs = [(a, l1) for a in [1e-3, 1e-2, 1e-1] for l1 in [0.3, 0.7]]

    serial = _score(_config(warm_start=False, n_jobs=1), X, y, index, cv_splits, param_pairs)
    parallel = _score(_config(warm_start=False, n_jobs=4), X, y, index, cv_splits, param_pairs)

    assert set(serial.keys()) == set(parallel.keys())
    for key in serial:
        np.testing.assert_allclose(
            serial[key]["score"], parallel[key]["score"], rtol=1e-12, atol=1e-12
        )


def test_warm_start_selects_same_best_and_scores():
    X, y, index = _make_panel(seed=13)
    cv_splits = _splits(index)
    param_pairs = [(a, l1) for a in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1] for l1 in [0.3, 0.7]]

    cold = _score(_config(warm_start=False, n_jobs=1), X, y, index, cv_splits, param_pairs)
    warm = _score(_config(warm_start=True, n_jobs=1), X, y, index, cv_splits, param_pairs)

    assert _best(cold, param_pairs) == _best(warm, param_pairs)
    for key in cold:
        np.testing.assert_allclose(
            cold[key]["score"], warm[key]["score"], rtol=1e-4, atol=1e-5
        )


def test_warm_start_equivalence_with_sample_weight():
    X, y, index = _make_panel(seed=17)
    cv_splits = _splits(index)
    param_pairs = [(a, l1) for a in [1e-3, 1e-2, 1e-1] for l1 in [0.3, 0.7]]

    rng = np.random.default_rng(23)
    sample_weight = rng.uniform(0.5, 1.5, size=len(index))

    cfg_cold = _config(warm_start=False, n_jobs=1)
    cfg_warm = _config(warm_start=True, n_jobs=1)

    wrapper_cold = QlibLinearWrapper(cfg_cold, logger=LOGGER)
    wrapper_warm = QlibLinearWrapper(cfg_warm, logger=LOGGER)
    evaluator = CVMetricEvaluator(cfg_cold["param_search_metric"], cfg_cold, LOGGER,
                                  n_trials=len(param_pairs))

    cold = wrapper_cold._score_param_grid_cv(
        X, y, index, sample_weight, cv_splits, param_pairs, evaluator
    )
    warm = wrapper_warm._score_param_grid_cv(
        X, y, index, sample_weight, cv_splits, param_pairs, evaluator
    )
    ref = _reference_scores(cfg_cold, X, y, index, cv_splits, param_pairs, sample_weight)

    # Cold path reproduces the weighted reference exactly.
    for key in ref:
        np.testing.assert_allclose(cold[key]["score"], ref[key], rtol=1e-9, atol=1e-12)
    # Warm path selects the same best and stays within solver tolerance.
    assert _best(cold, param_pairs) == _best(warm, param_pairs)
    for key in cold:
        np.testing.assert_allclose(
            cold[key]["score"], warm[key]["score"], rtol=1e-4, atol=1e-5
        )
