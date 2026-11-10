# -*- coding: utf-8 -*-
"""
Regression tests for LightGBM custom objectives.

Guards the lightgbm>=4.0 argument-order bug: the native API calls the objective
as ``objective(preds, train_data)`` (Dataset is the *second* arg), but the old
code assumed Dataset-first and crashed with
``object of type 'Dataset' has no len()``.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from factor_synthesis.qlib_adapter.custom_objectives import QlibCustomObjectives

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class _FakeDataset:
    """Minimal stand-in exposing the lgb.Dataset getter surface used here."""

    def __init__(self, labels, weights=None, groups=None):
        self._labels, self._weights, self._groups = labels, weights, groups

    def get_label(self):
        return self._labels

    def get_weight(self):
        return self._weights

    def get_group(self):
        return self._groups


class TestUnpackObjectiveArgs(unittest.TestCase):
    def test_native_order_preds_then_dataset(self):
        # lightgbm>=4 native API: (preds, dataset)
        preds = np.arange(5.0)
        ds = _FakeDataset(labels=np.ones(5), weights=np.full(5, 2.0), groups=np.array([5]))
        p, lbl, w, g = QlibCustomObjectives._unpack_objective_args(preds, ds)
        np.testing.assert_array_equal(p, preds)
        np.testing.assert_array_equal(lbl, np.ones(5))
        np.testing.assert_array_equal(w, np.full(5, 2.0))
        np.testing.assert_array_equal(g, np.array([5]))

    def test_legacy_order_dataset_then_preds(self):
        # position-agnostic: also tolerate (dataset, preds)
        preds = np.arange(5.0)
        ds = _FakeDataset(labels=np.ones(5))
        p, lbl, _, _ = QlibCustomObjectives._unpack_objective_args(ds, preds)
        np.testing.assert_array_equal(p, preds)
        np.testing.assert_array_equal(lbl, np.ones(5))

    def test_sklearn_order_ytrue_then_ypred(self):
        # sklearn API: (y_true, y_pred) -> labels=arg0, preds=arg1
        p, lbl, w, g = QlibCustomObjectives._unpack_objective_args(
            np.array([1.0, 2.0]), np.array([3.0, 4.0])
        )
        np.testing.assert_array_equal(lbl, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(p, np.array([3.0, 4.0]))
        self.assertIsNone(w)
        self.assertIsNone(g)


@unittest.skipUnless(LIGHTGBM_AVAILABLE, "lightgbm not installed")
class TestLgbTrainEndToEnd(unittest.TestCase):
    """The exact regressed path: native lgb.train + topk_sharpe objective."""

    def _panel(self, n_dates=40, n_stocks=50, n_factors=8, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n_dates * n_stocks, n_factors))
        y = rng.standard_normal(n_dates * n_stocks)
        groups = np.full(n_dates, n_stocks)
        return x, y, groups

    def test_topk_sharpe_objective_trains_without_len_error(self):
        x, y, groups = self._panel()
        dtrain = lgb.Dataset(x, label=y, group=groups, weight=np.abs(y) + 0.1)
        obj = QlibCustomObjectives.get_lgb_objective("topk_sharpe")
        feval = QlibCustomObjectives.get_lgb_eval_metric("sharpe")
        model = lgb.train(
            {"objective": obj, "verbose": -1, "num_leaves": 7, "learning_rate": 0.05},
            dtrain,
            num_boost_round=10,
            valid_sets=[dtrain],
            valid_names=["t"],
            feval=feval,
        )
        pred = model.predict(x)
        self.assertEqual(pred.shape, (x.shape[0],))
        self.assertTrue(np.isfinite(pred).all())

    def test_cross_sectional_ic_objective_trains(self):
        x, y, groups = self._panel(seed=1)
        dtrain = lgb.Dataset(x, label=y, group=groups)
        obj = QlibCustomObjectives.get_lgb_objective("cross_sectional_ic")
        model = lgb.train(
            {"objective": obj, "verbose": -1, "num_leaves": 7},
            dtrain,
            num_boost_round=5,
        )
        self.assertTrue(np.isfinite(model.predict(x)).all())


if __name__ == "__main__":
    unittest.main()
