# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from factor_synthesis.preselection.volatility_regime_strategy import (
    VolatilityBucketConsensusPreSelectionStrategy,
)


def _make_regime_data(seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    stocks = [f"S{i:03d}" for i in range(24)]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])

    date_pos = np.repeat(np.arange(len(dates)), len(stocks))
    scales = np.where(date_pos < 30, 0.4, np.where(date_pos < 60, 1.0, 2.2))
    y_values = rng.normal(0.0, scales, size=len(index))
    y = pd.Series(y_values, index=index, name="forward_return")

    high_mask = date_pos >= 60
    sign_flip = y_values.copy()
    sign_flip[high_mask] *= -1.0

    X = pd.DataFrame(
        {
            "stable_factor": y_values + rng.normal(0.0, 0.08, size=len(index)),
            "high_only_factor": np.where(
                high_mask,
                y_values + rng.normal(0.0, 0.08, size=len(index)),
                rng.normal(0.0, 1.0, size=len(index)),
            ),
            "sign_flip_factor": sign_flip + rng.normal(0.0, 0.08, size=len(index)),
            "noise_factor": rng.normal(0.0, 1.0, size=len(index)),
        },
        index=index,
    )
    return X, y


def test_regime_consensus_keeps_stable_and_rejects_single_bucket_factor():
    X, y = _make_regime_data()
    strategy = VolatilityBucketConsensusPreSelectionStrategy(
        {
            "enabled": True,
            "metrics": {
                "min_abs_ic": 0.05,
                "min_regimes": 2,
                "consensus_sign": "same",
            },
            "regime_quantiles": [0.33, 0.66],
            "fallback_top_n": 2,
            "min_bucket_dates": 5,
            "min_bucket_samples": 50,
        }
    )

    selected = strategy.select(X, y)

    assert "stable_factor" in selected
    assert "high_only_factor" not in selected


def test_consensus_sign_rejects_sign_flip():
    X, y = _make_regime_data()
    strategy = VolatilityBucketConsensusPreSelectionStrategy(
        {
            "enabled": True,
            "metrics": {
                "min_abs_ic": 0.05,
                "min_regimes": 2,
                "consensus_sign": "same",
            },
            "regime_quantiles": [0.33, 0.66],
            "fallback_top_n": 2,
            "min_bucket_dates": 5,
            "min_bucket_samples": 50,
        }
    )

    selected = strategy.select(X, y)

    assert "sign_flip_factor" not in selected
