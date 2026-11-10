# -*- coding: utf-8 -*-
"""
Test suite for CV metric evaluator.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from factor_synthesis.qlib_adapter.cv_metrics import CVMetricEvaluator


def create_synthetic_cv_data(
    n_dates: int = 30,
    n_stocks: int = 20,
    seed: int = 42,
) -> tuple:
    """
    Create synthetic predictions/returns with positive long-short signal.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    stocks = [f"S{i:04d}" for i in range(n_stocks)]

    index = pd.MultiIndex.from_product(
        [dates, stocks], names=["datetime", "instrument"]
    )

    base = np.tile(np.linspace(-1, 1, n_stocks), n_dates)
    y_true = base + rng.normal(scale=0.05, size=len(index))
    y_pred = y_true + rng.normal(scale=0.02, size=len(index))

    return y_pred, y_true, index.get_level_values(0)


class TestCVMetricEvaluator(unittest.TestCase):
    """Test deflated Sharpe and GT-Score metrics."""

    def test_deflated_sharpe_penalizes_multiple_trials(self):
        y_pred, y_true, dates = create_synthetic_cv_data()
        cfg = {"top_k_size": 5, "target_period": 1}

        evaluator_low = CVMetricEvaluator("deflated_sharpe", cfg, n_trials=1)
        evaluator_high = CVMetricEvaluator("deflated_sharpe", cfg, n_trials=200)

        score_low = evaluator_low.evaluate(y_pred, y_true, dates)
        score_high = evaluator_high.evaluate(y_pred, y_true, dates)

        self.assertTrue(0.0 <= score_low <= 1.0)
        self.assertTrue(0.0 <= score_high <= 1.0)
        self.assertGreaterEqual(score_low, score_high)

    def test_gt_score_is_finite(self):
        y_pred, y_true, dates = create_synthetic_cv_data()
        cfg = {"top_k_size": 5, "target_period": 1}

        evaluator = CVMetricEvaluator("gt_score", cfg)
        score = evaluator.evaluate(y_pred, y_true, dates)

        self.assertTrue(np.isfinite(score))
        self.assertGreater(score, -1.0)


if __name__ == "__main__":
    unittest.main()
