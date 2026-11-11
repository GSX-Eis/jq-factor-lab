"""Tests for the MVO weighting scheme in the overlay path."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VectorBTBacktestRunner = importlib.import_module(
    "factor_synthesis.vectorbt_adapter.backtest_runner"
).VectorBTBacktestRunner


def _runner(mvo_extra=None, max_weight=None):
    mvo = {"risk_aversion": 8.0, "turnover_penalty": 0.0, "cov_lookback_days": 120,
           "cov_min_obs": 40, "style_dev_limit": None, "industry_dev_limit": None}
    mvo.update(mvo_extra or {})
    weighting = {"scheme": "mvo", "mvo": mvo}
    if max_weight is not None:
        weighting["max_weight"] = max_weight
    return VectorBTBacktestRunner(
        {"top_k_strategy": {"top_pct": 0.05, "holding_period": 1, "weighting": weighting}}
    )


def _price_df(vols, n_days=200, seed=3):
    """Wide close-price frame with per-stock daily vol given by ``vols``."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    data = {}
    for i, vol in enumerate(vols):
        rets = rng.normal(0.0003, vol, n_days)
        data[f"S{i}"] = 10.0 * np.cumprod(1 + rets)
    return pd.DataFrame(data, index=dates)


def test_mvo_config_defaults_parsed():
    runner = _runner()
    cfg = runner.overlay["weighting"]["mvo"]
    assert cfg["risk_aversion"] == 8.0
    assert cfg["cov_lookback_days"] == 120
    assert cfg["style_dev_limit"] is None
    assert runner.overlay_active is True  # scheme mvo activates the overlay


def test_shrunk_covariance_shape_and_psd():
    runner = _runner()
    price_df = _price_df([0.01, 0.02, 0.03])
    sigma = runner._shrunk_covariance(
        ["S0", "S1", "S2"], price_df, price_df.index[-1], runner.overlay["weighting"]["mvo"]
    )
    assert sigma is not None and sigma.shape == (3, 3)
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals.min() > 0  # PSD after shrinkage
    # annualized daily vol of ~2% -> variance order of magnitude check
    assert 0.001 < sigma[0, 0] < 1.0


def test_shrunk_covariance_requires_min_obs():
    runner = _runner()
    price_df = _price_df([0.01, 0.02], n_days=20)  # < cov_min_obs=40
    sigma = runner._shrunk_covariance(
        ["S0", "S1"], price_df, price_df.index[-1], runner.overlay["weighting"]["mvo"]
    )
    assert sigma is None


def test_mvo_prefers_high_alpha_and_respects_bounds():
    runner = _runner(max_weight=0.6)
    price_df = _price_df([0.02, 0.02, 0.02])
    scores = pd.Series({"S0": 3.0, "S1": 2.0, "S2": 1.0})
    weights = runner._mvo_weights(["S0", "S1", "S2"], scores, price_df, price_df.index[-1], {})
    assert weights is not None
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["S0"] >= weights["S1"] >= weights["S2"]
    assert all(w <= 0.6 + 1e-9 for w in weights.values())
    assert all(w >= -1e-12 for w in weights.values())


def test_mvo_risk_aversion_penalizes_volatile_name():
    runner = _runner({"risk_aversion": 50.0})
    # equal alpha, S1 has 5x the vol of S0/S2 -> should get the least weight
    price_df = _price_df([0.01, 0.05, 0.01])
    scores = pd.Series({"S0": 1.0, "S1": 1.0, "S2": 1.0})
    weights = runner._mvo_weights(["S0", "S1", "S2"], scores, price_df, price_df.index[-1], {})
    assert weights is not None
    assert weights["S1"] < weights["S0"]
    assert weights["S1"] < weights["S2"]


def test_mvo_turnover_penalty_anchors_to_previous_weights():
    price_df = _price_df([0.02, 0.02, 0.02])
    scores = pd.Series({"S0": 3.0, "S1": 2.0, "S2": 1.0})
    prev = {"S0": 0.2, "S1": 0.3, "S2": 0.5}
    free = _runner({"turnover_penalty": 0.0})._mvo_weights(
        ["S0", "S1", "S2"], scores, price_df, price_df.index[-1], prev
    )
    sticky = _runner({"turnover_penalty": 50.0})._mvo_weights(
        ["S0", "S1", "S2"], scores, price_df, price_df.index[-1], prev
    )
    assert free is not None and sticky is not None
    turnover_free = sum(abs(free[s] - prev[s]) for s in prev)
    turnover_sticky = sum(abs(sticky[s] - prev[s]) for s in prev)
    assert turnover_sticky < turnover_free
    assert abs(sticky["S2"] - 0.5) < 0.05  # heavy penalty pins weights near prev


def test_mvo_falls_back_to_signal_weights_on_short_history():
    runner = _runner()
    price_df = _price_df([0.02, 0.02], n_days=10)  # too short for covariance
    scores = pd.Series({"S0": 2.0, "S1": 1.0})
    weights = runner._compute_overlay_weights(
        ["S0", "S1"], scores, price_df, price_df.index[-1], {}
    )
    # signal fallback: rank weights 2:1 normalized
    assert abs(weights["S0"] - 2.0 / 3.0) < 1e-9
    assert abs(weights["S1"] - 1.0 / 3.0) < 1e-9


def test_mvo_style_constraint_limits_exposure_drift():
    class _FakeStyleConn:
        def __init__(self, frame):
            self._frame = frame

        def execute(self, sql):
            self._sql = sql
            return self

        def fetchdf(self):
            return self._frame.copy()

        def fetchall(self):
            return []

    price_df = _price_df([0.02, 0.02, 0.02, 0.02])
    stocks = ["S0", "S1", "S2", "S3"]
    # One style factor: S0 has an extreme exposure; unconstrained MVO would
    # overweight S0 (highest alpha).
    style = pd.DataFrame(
        {"stock_code": stocks, "date": ["2023-06-01"] * 4, "size": [3.0, 0.0, 0.0, 0.0]}
    )
    scores = pd.Series({"S0": 4.0, "S1": 1.0, "S2": 1.0, "S3": 1.0})

    unconstrained = _runner({"style_dev_limit": None})
    constrained = _runner({"style_dev_limit": 0.15})
    constrained.price_db_conn = _FakeStyleConn(style)

    w_free = unconstrained._mvo_weights(stocks, scores, price_df, price_df.index[-1], {})
    w_band = constrained._mvo_weights(stocks, scores, price_df, price_df.index[-1], {})
    assert w_free is not None and w_band is not None

    exposure = lambda w: sum(w[s] * x for s, x in zip(stocks, [3.0, 0.0, 0.0, 0.0]))
    eq_exposure = 3.0 / 4.0
    assert exposure(w_free) - eq_exposure > 0.15  # would drift without the band
    assert abs(exposure(w_band) - eq_exposure) <= 0.15 + 1e-6


def test_mvo_industry_constraint_limits_sector_drift():
    class _FakeIndustryConn:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, sql):
            self._sql = sql
            return self

        def fetchall(self):
            return list(self._rows)

        def fetchdf(self):
            return pd.DataFrame()

    price_df = _price_df([0.02, 0.02, 0.02, 0.02])
    stocks = ["S0", "S1", "S2", "S3"]
    scores = pd.Series({"S0": 4.0, "S1": 3.5, "S2": 1.0, "S3": 1.0})
    rows = [("S0", "801010"), ("S1", "801010"), ("S2", "801080"), ("S3", "801080")]

    runner = _runner({"industry_dev_limit": 0.1, "style_dev_limit": None})
    runner.price_db_conn = _FakeIndustryConn(rows)
    weights = runner._mvo_weights(stocks, scores, price_df, price_df.index[-1], {})
    assert weights is not None
    sector_801010 = weights["S0"] + weights["S1"]
    assert sector_801010 <= 0.5 + 0.1 + 1e-6  # equal-weight share 0.5 + band 0.1
