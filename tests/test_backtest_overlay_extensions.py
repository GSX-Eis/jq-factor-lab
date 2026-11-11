"""Tests for liquidity_cap application and regime_filter in the overlay path."""

from __future__ import annotations

import importlib
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VectorBTBacktestRunner = importlib.import_module(
    "factor_synthesis.vectorbt_adapter.backtest_runner"
).VectorBTBacktestRunner


class _FakeAdvConn:
    """Duckdb-like connection returning fixed ADV rows."""

    def __init__(self, adv_rows):
        self._adv_rows = adv_rows

    def execute(self, sql):
        return self

    def fetchall(self):
        return list(self._adv_rows)


class _FakeIndexConn:
    """Duckdb-like connection returning a fixed index_price_daily frame."""

    def __init__(self, frame):
        self._frame = frame

    def execute(self, sql):
        return self

    def fetchdf(self):
        return self._frame.copy()


def _runner(strategy_extra=None, holding_period=1):
    strategy = {"top_pct": 0.05, "holding_period": holding_period, "cost_rate": 0.003}
    strategy.update(strategy_extra or {})
    return VectorBTBacktestRunner({"top_k_strategy": strategy})


def test_overlay_defaults_keep_new_features_disabled():
    runner = _runner()
    assert runner.overlay_active is False
    assert runner.overlay["liquidity_cap"]["enabled"] is False
    assert runner.overlay["risk_overlay"]["regime_filter"]["enabled"] is False


def test_external_daily_signal_expires_and_keeps_legacy_opt_in(tmp_path):
    path = tmp_path / "signal.parquet"
    dates = pd.to_datetime(["2023-12-01", "2023-12-06"])
    signal = pd.DataFrame({"effective_date": dates[:1],
                           "observed_through": pd.to_datetime(["2023-11-30"]),
                           "valid_until": dates[:1], "regime_score": [-1.]})
    signal.to_parquet(path, index=False)
    external = {"enabled": True, "position_plugin_id": "one_sided_linear_v1",
                "signal_path": str(path), "signal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "require_valid_until": True, "missing_exposure": 1.0}
    runner = _runner({"risk_overlay": {"external_regime": external}})
    try:
        returns = pd.Series(0., index=dates)
        exposure = runner._compute_exposure_by_date(returns, list(dates), pd.DataFrame(index=dates), [])
        assert exposure[dates[0]] == .5
        assert exposure[dates[1]] == 1.0  # no unbounded ffill
        signal.drop(columns="valid_until").to_parquet(path, index=False)
        cfg = dict(runner.overlay["risk_overlay"]["external_regime"])
        cfg["signal_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="requires valid_until"):
            runner._load_external_regime_signal(cfg)
        cfg["require_valid_until"] = False
        assert len(runner._load_external_regime_signal(cfg)) == 1
    finally:
        runner.close()


def test_liquidity_cap_and_regime_activate_overlay():
    liq = _runner({"liquidity_cap": {"enabled": True}})
    assert liq.overlay_active is True
    regime = _runner({"risk_overlay": {"regime_filter": {"enabled": True}}})
    assert regime.overlay_active is True
    assert regime._risk_overlay_active(regime.overlay) is True


def test_apply_liquidity_cap_clips_and_redistributes():
    runner = _runner(
        {
            "liquidity_cap": {
                "enabled": True,
                "adv_lookback_days": 20,
                "max_participation": 0.1,
                "capital_base": 10_000_000,
            }
        }
    )
    # A is illiquid: cap = 0.1 * 1_000_000 * 1 / 10_000_000 = 0.01
    # B and C are deep enough to absorb the excess.
    runner.price_db_conn = _FakeAdvConn(
        [("A", 1_000_000.0), ("B", 500_000_000.0), ("C", 500_000_000.0)]
    )
    weights = {"A": 0.5, "B": 0.25, "C": 0.25}
    capped = runner._apply_liquidity_cap(weights, "2024-01-10", deployed_capital=10_000_000)
    assert capped["A"] == 1e-2 or abs(capped["A"] - 0.01) < 1e-9
    # Total weight preserved because B/C had headroom.
    assert abs(sum(capped.values()) - 1.0) < 1e-9
    assert capped["B"] > 0.25 and capped["C"] > 0.25


def test_apply_liquidity_cap_residual_goes_to_cash_when_all_capped():
    runner = _runner(
        {
            "liquidity_cap": {
                "enabled": True,
                "adv_lookback_days": 20,
                "max_participation": 0.1,
                "capital_base": 100_000_000,
            }
        }
    )
    # Both names capped at 0.1 * 2_000_000 / 100_000_000 = 0.002.
    runner.price_db_conn = _FakeAdvConn([("A", 2_000_000.0), ("B", 2_000_000.0)])
    capped = runner._apply_liquidity_cap({"A": 0.5, "B": 0.5}, "2024-01-10", deployed_capital=100_000_000)
    assert abs(capped["A"] - 0.002) < 1e-9
    assert abs(capped["B"] - 0.002) < 1e-9
    assert sum(capped.values()) < 1.0  # residual stays in cash


def test_apply_liquidity_cap_keeps_names_without_adv_data():
    runner = _runner({"liquidity_cap": {"enabled": True, "capital_base": 10_000_000}})
    runner.price_db_conn = _FakeAdvConn([("A", 1_000_000.0)])  # no row for B
    capped = runner._apply_liquidity_cap({"A": 0.5, "B": 0.5}, "2024-01-10", deployed_capital=10_000_000)
    assert capped["B"] >= 0.5  # untouched or grew via redistribution


def test_regime_spread_zscore_flags_microcap_crash():
    dates = pd.date_range("2022-01-03", periods=400, freq="B")
    rng = np.random.default_rng(7)
    bench = pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0002, 0.005, len(dates))), index=dates)
    micro_rets = rng.normal(0.0002, 0.005, len(dates))
    micro_rets[-15:] = -0.04  # micro-cap crash at the tail
    micro = pd.Series(100.0 * np.cumprod(1 + micro_rets), index=dates)
    frame = pd.concat(
        [
            pd.DataFrame({"date": dates, "index_code": "399101.XSHE", "close": micro.values}),
            pd.DataFrame({"date": dates, "index_code": "000300.XSHG", "close": bench.values}),
        ],
        ignore_index=True,
    )
    runner = _runner({"risk_overlay": {"regime_filter": {"enabled": True}}})
    runner.price_db_conn = _FakeIndexConn(frame)
    z = runner._regime_spread_zscore(runner.overlay["risk_overlay"]["regime_filter"])
    assert z is not None and len(z) > 0
    assert float(z.iloc[-1]) < -2.0  # crash shows up as a strongly negative z


def test_compute_exposure_applies_regime_derisk_and_exit():
    runner = _runner(
        {
            "risk_overlay": {
                "regime_filter": {
                    "enabled": True,
                    "derisk_z": -1.5,
                    "derisk_exposure": 0.5,
                    "exit_z": -2.5,
                }
            }
        }
    )
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    z = pd.Series([0.0, 0.0, -1.8, -1.8, -3.0, -3.0], index=dates)
    runner._regime_spread_zscore = lambda rf: z  # bypass DB
    returns = pd.Series(0.001, index=dates)
    rebalance_dates = [dates[1], dates[3], dates[5]]
    price_df = pd.DataFrame({"A": 1.0}, index=dates)
    exposure = runner._compute_exposure_by_date(returns, rebalance_dates, price_df, ["A"])
    norm = {pd.Timestamp(k).normalize(): v for k, v in exposure.items()}
    assert norm[dates[1].normalize()] == 1.0  # prior z = 0
    assert norm[dates[3].normalize()] == 0.5  # prior z = -1.8 -> derisk
    assert norm[dates[5].normalize()] == 0.0  # prior z = -3.0 -> exit


def test_regime_filter_skips_gracefully_without_connection():
    runner = _runner({"risk_overlay": {"regime_filter": {"enabled": True}}})
    runner.price_db_conn = None
    assert runner._regime_spread_zscore(runner.overlay["risk_overlay"]["regime_filter"]) is None
