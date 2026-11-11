"""Unit tests for portfolio-overlay logic in VectorBTBacktestRunner.

Covers turnover-control selection (banding hysteresis + tranche throttle),
risk-aware weighting (inverse-vol / signal / max-weight cap) and the
endogenous tail-risk exposure schedule (vol targeting / drawdown control).
These exercise the pure overlay helpers without running a full VectorBT
simulation so they stay fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner


def _runner(overlay: dict) -> VectorBTBacktestRunner:
    cfg = {"top_k_strategy": {"top_k": 5, **overlay}}
    return VectorBTBacktestRunner(cfg)


class TestOverlayConfig:
    def test_defaults_inactive(self):
        runner = _runner({})
        assert runner.overlay_active is False

    def test_banding_activates(self):
        runner = _runner({"banding": {"enabled": True}})
        assert runner.overlay_active is True

    def test_equal_weight_not_active(self):
        runner = _runner({"weighting": {"scheme": "equal"}})
        assert runner.overlay_active is False

    def test_inverse_vol_activates(self):
        runner = _runner({"weighting": {"scheme": "inverse_vol"}})
        assert runner.overlay_active is True

    def test_raw_score_activates(self):
        runner = _runner({"weighting": {"scheme": "raw_score"}})
        assert runner.overlay_active is True

    def test_max_weight_activates(self):
        runner = _runner({"weighting": {"max_weight": 0.2}})
        assert runner.overlay_active is True

    def test_single_tranche_not_active(self):
        runner = _runner({"tranches": {"enabled": True, "n_tranches": 1}})
        assert runner.overlay_active is False


class TestSelectionOverlays:
    def _ranked(self, n: int) -> pd.Series:
        return pd.Series(
            np.linspace(1.0, 0.0, n),
            index=[f"S{i:03d}" for i in range(n)],
        )

    def test_no_overlay_is_plain_topk(self):
        runner = _runner({})
        ranked = self._ranked(100)
        # overlay inactive -> helper still mirrors head(k) when called directly
        selected = runner._select_with_overlays(ranked, 5, set())
        assert selected == ["S000", "S001", "S002", "S003", "S004"]

    def test_banding_retains_held_in_sell_band(self):
        runner = _runner(
            {"banding": {"enabled": True, "buy_rank_pct": 0.04, "sell_rank_pct": 0.10}}
        )
        ranked = self._ranked(100)
        # A held name now ranked 7th (inside 10% sell band) should be retained
        held = {"S007"}
        selected = runner._select_with_overlays(ranked, 5, held)
        assert "S007" in selected
        assert len(selected) == 5

    def test_banding_fills_topk_from_empty_book(self):
        runner = _runner(
            {"banding": {"enabled": True, "buy_rank_pct": 0.04, "sell_rank_pct": 0.10}}
        )
        selected = runner._select_with_overlays(self._ranked(100), 5, set())
        assert selected == ["S000", "S001", "S002", "S003", "S004"]

    def test_banding_drops_held_outside_sell_band(self):
        runner = _runner(
            {"banding": {"enabled": True, "buy_rank_pct": 0.04, "sell_rank_pct": 0.10}}
        )
        ranked = self._ranked(100)
        held = {"S050"}  # ranked 50th, far outside 10% band
        selected = runner._select_with_overlays(ranked, 5, held)
        assert "S050" not in selected

    def test_tranche_throttles_new_entries(self):
        base = {"banding": {"enabled": True, "buy_rank_pct": 0.10, "sell_rank_pct": 0.20}}
        ranked = self._ranked(100)
        held = {"S050", "S051", "S052"}  # all stale, would all be replaced
        no_tranche = _runner(base)
        full = no_tranche._select_with_overlays(ranked, 5, held)
        new_full = [s for s in full if s not in held]

        tranche = _runner({**base, "tranches": {"enabled": True, "n_tranches": 5}})
        throttled = tranche._select_with_overlays(ranked, 5, held)
        new_throttled = [s for s in throttled if s not in held]
        # tranche throttle must not add more new names than the no-throttle case
        assert len(new_throttled) <= len(new_full)
        assert len(new_throttled) <= 1  # ceil(5/5) == 1

    def test_selection_diagnostic_keeps_actual_books(self):
        runner = _runner({})
        runner._record_selection_diagnostic(
            date=pd.Timestamp("2023-01-03"),
            pool_count=3,
            valid_score_count=3,
            tradeable_count=3,
            ranked=pd.Series([3.0, 2.0, 1.0], index=["C", "A", "B"]),
            selected=["C", "A"],
            previous_holdings={"A", "B"},
        )
        row = runner._selection_diagnostics[0]
        assert row["selected_stocks"] == ["A", "C"]
        assert row["previous_holdings"] == ["A", "B"]


class TestWeighting:
    def test_mvo_records_fallback_state(self, monkeypatch):
        runner = _runner({"weighting": {"scheme": "mvo", "max_weight": 0.2}})
        monkeypatch.setattr(runner, "_mvo_weights", lambda *args, **kwargs: None)
        selected = ["A", "B", "C", "D", "E"]

        weights = runner._compute_overlay_weights(
            selected,
            pd.Series([5, 4, 3, 2, 1], index=selected),
            pd.DataFrame(),
            "2024-01-02",
        )

        assert sum(weights.values()) == pytest.approx(1.0)
        assert runner._mvo_diagnostics == [
            {"date": "2024-01-02", "state": "fallback_signal"}
        ]

    def test_equal_weights_sum_to_one(self):
        runner = _runner({"weighting": {"scheme": "equal"}})
        w = runner._compute_overlay_weights(["A", "B", "C", "D"], pd.Series(dtype=float), pd.DataFrame(), None)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert all(abs(v - 0.25) < 1e-9 for v in w.values())

    def test_signal_weighting_monotonic(self):
        runner = _runner({"weighting": {"scheme": "signal"}})
        # selected is already best-first
        w = runner._compute_overlay_weights(["A", "B", "C"], pd.Series(dtype=float), pd.DataFrame(), None)
        assert w["A"] > w["B"] > w["C"]
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_raw_score_preserves_positive_magnitudes(self):
        runner = _runner({"weighting": {"scheme": "raw_score"}})
        w = runner._compute_overlay_weights(
            ["A", "B", "C"], pd.Series({"A": 2.0, "B": 6.0, "C": -1.0}), pd.DataFrame(), None
        )
        assert w == {"A": 0.25, "B": 0.75, "C": 0.0}

    def test_raw_score_falls_back_equal_when_no_positive_score(self):
        runner = _runner({"weighting": {"scheme": "raw_score"}})
        w = runner._compute_overlay_weights(
            ["A", "B"], pd.Series({"A": float("nan"), "B": -1.0}), pd.DataFrame(), None
        )
        assert w == {"A": 0.5, "B": 0.5}

    def test_inverse_vol_weighting(self):
        dates = pd.date_range("2023-01-01", periods=40, freq="B")
        # A: low vol (small wiggle), B: high vol (big wiggle)
        a = 100 + np.sin(np.arange(40)) * 0.1
        b = 100 + np.sin(np.arange(40)) * 5.0
        price = pd.DataFrame({"A": a, "B": b}, index=dates)
        runner = _runner({"weighting": {"scheme": "inverse_vol", "vol_lookback_days": 20}})
        w = runner._compute_overlay_weights(["A", "B"], pd.Series(dtype=float), price, dates[-1])
        assert w["A"] > w["B"]  # low-vol name gets more weight
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_max_weight_cap_redistributes(self):
        runner = _runner({"weighting": {"scheme": "signal", "max_weight": 0.3}})
        w = runner._compute_overlay_weights(["A", "B", "C", "D"], pd.Series(dtype=float), pd.DataFrame(), None)
        assert all(v <= 0.3 + 1e-9 for v in w.values())
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_max_weight_cap_infeasible_falls_back_equal(self):
        runner = _runner({"weighting": {"scheme": "signal"}})
        # cap * n < 1 -> infeasible -> equal weight
        capped = runner._apply_max_weight_cap({"A": 0.6, "B": 0.4}, 0.1)
        assert abs(capped["A"] - 0.5) < 1e-9
        assert abs(capped["B"] - 0.5) < 1e-9


class TestExposureSchedule:
    def _returns(self, vals) -> pd.Series:
        dates = pd.date_range("2023-01-01", periods=len(vals), freq="B")
        return pd.Series(vals, index=dates)

    def test_vol_targeting_scales_down_high_vol(self):
        runner = _runner(
            {
                "risk_overlay": {
                    "vol_targeting": {
                        "enabled": True,
                        "target_annual_vol": 0.10,
                        "lookback_days": 10,
                        "min_exposure": 0.1,
                        "max_exposure": 1.0,
                    }
                }
            }
        )
        # high realized vol -> exposure < 1
        rng = np.random.default_rng(0)
        rets = self._returns(rng.normal(0, 0.05, 60))
        reb_dates = list(rets.index[::10])
        exposure = runner._compute_exposure_by_date(rets, reb_dates, pd.DataFrame(), [])
        later = [v for d, v in exposure.items() if d > rets.index[15]]
        assert later
        assert all(v < 1.0 for v in later)

    def test_drawdown_control_derisks(self):
        runner = _runner(
            {
                "risk_overlay": {
                    "drawdown_control": {
                        "enabled": True,
                        "derisk_dd": -0.05,
                        "derisk_exposure": 0.5,
                        "exit_dd": -0.50,
                        "reentry_dd": -0.02,
                    }
                }
            }
        )
        # steady losses build a drawdown beyond -5%
        rets = self._returns([-0.01] * 40)
        reb_dates = list(rets.index[::5])
        exposure = runner._compute_exposure_by_date(rets, reb_dates, pd.DataFrame(), [])
        later = [v for d, v in exposure.items() if d > rets.index[20]]
        assert later
        assert all(v <= 0.5 + 1e-9 for v in later)

    def test_no_risk_overlay_returns_unit_exposure(self):
        runner = _runner({"banding": {"enabled": True}})
        rets = self._returns([0.001] * 30)
        reb_dates = list(rets.index[::5])
        exposure = runner._compute_exposure_by_date(rets, reb_dates, pd.DataFrame(), [])
        assert all(abs(v - 1.0) < 1e-9 for v in exposure.values())


class TestPartialRebalance:
    def test_activates_overlay(self):
        runner = _runner({"partial_rebalance": {"enabled": True, "trade_rate": 0.3}})
        assert runner.overlay_active is True
        assert runner.overlay["partial_rebalance"]["trade_rate"] == 0.3

    def test_blend_moves_partially_toward_target(self):
        runner = _runner(
            {"partial_rebalance": {"enabled": True, "trade_rate": 0.5, "min_weight": 0.0}}
        )
        prev = {"A": 0.5, "B": 0.5}
        target = {"B": 0.5, "C": 0.5}
        blended = runner._blend_partial_rebalance(target, prev)
        assert abs(blended["A"] - 0.25) < 1e-9
        assert abs(blended["B"] - 0.50) < 1e-9
        assert abs(blended["C"] - 0.25) < 1e-9
        assert abs(sum(blended.values()) - 1.0) < 1e-9

    def test_empty_book_jumps_to_target(self):
        runner = _runner({"partial_rebalance": {"enabled": True, "trade_rate": 0.3}})
        target = {"A": 0.5, "B": 0.5}
        assert runner._blend_partial_rebalance(target, {}) == target

    def test_dust_positions_liquidated_and_renormalized(self):
        runner = _runner(
            {"partial_rebalance": {"enabled": True, "trade_rate": 0.5, "min_weight": 0.05}}
        )
        prev = {"A": 0.04, "B": 0.96}  # A is decaying out
        target = {"B": 1.0}
        blended = runner._blend_partial_rebalance(target, prev)
        # A blends to 0.02 < min_weight -> dropped, B renormalized to 1.0
        assert "A" not in blended
        assert abs(blended["B"] - 1.0) < 1e-9

    def test_geometric_convergence_to_target(self):
        runner = _runner(
            {"partial_rebalance": {"enabled": True, "trade_rate": 0.5, "min_weight": 0.0}}
        )
        weights = {"A": 1.0}
        target = {"B": 1.0}
        for _ in range(12):
            weights = runner._blend_partial_rebalance(target, weights)
        assert weights.get("B", 0.0) > 0.999


class TestCrisisGuard:
    def _runner_cg(self, **cg_kwargs) -> VectorBTBacktestRunner:
        cg = {
            "enabled": True,
            "min_triggers": 2,
            "derisk_exposure": 0.3,
            "hard_exit_triggers": 3,
            "cooldown_days": 2,
            "signals": {
                # DB-backed signals off so unit tests stay offline.
                "limit_down": {"enabled": False},
                "turnover_collapse": {"enabled": False},
                "breadth": {"enabled": True, "loss_threshold": -0.05, "breadth_threshold": 0.4},
                "spread": {"enabled": False},
            },
        }
        cg.update(cg_kwargs)
        return _runner({"risk_overlay": {"crisis_guard": cg}})

    def test_activates_risk_overlay(self):
        runner = self._runner_cg()
        assert runner.overlay_active is True
        assert runner._risk_overlay_active(runner.overlay) is True

    def test_single_signal_below_min_triggers_keeps_full_exposure(self):
        runner = self._runner_cg(min_triggers=2)
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        # all 4 pool names crash every day -> breadth fires, but it is the
        # only enabled signal and min_triggers=2 -> never de-risk
        price = pd.DataFrame(
            {f"S{i}": 100 * (0.9 ** np.arange(len(dates))) for i in range(4)}, index=dates
        )
        rets = pd.Series(-0.02, index=dates)
        exposure = runner._compute_exposure_by_date(rets, list(dates[2:]), price, list(price.columns))
        assert all(abs(v - 1.0) < 1e-9 for v in exposure.values())

    def test_min_triggers_one_breadth_derisks_with_cooldown(self):
        runner = self._runner_cg(min_triggers=1, hard_exit_triggers=9, cooldown_days=2)
        dates = pd.date_range("2023-01-02", periods=12, freq="B")
        # crash for the first 4 days, flat afterwards
        path = [0.9] * 4 + [1.0] * 8
        levels = 100 * np.cumprod(path)
        price = pd.DataFrame({f"S{i}": levels for i in range(4)}, index=dates)
        rets = pd.Series(0.0, index=dates)
        reb = list(dates)
        exposure = runner._compute_exposure_by_date(rets, reb, price, list(price.columns))
        # day after a crash day -> derisk_exposure
        assert abs(exposure[dates[2]] - 0.3) < 1e-9
        # cooldown keeps the book de-risked right after the crash stops
        assert abs(exposure[dates[5]] - 0.3) < 1e-9
        assert abs(exposure[dates[6]] - 0.3) < 1e-9
        # exposure recovers once cooldown has elapsed
        assert abs(exposure[dates[-1]] - 1.0) < 1e-9

    def test_hard_exit_zeroes_exposure(self):
        runner = self._runner_cg(min_triggers=1, hard_exit_triggers=1, cooldown_days=0)
        dates = pd.date_range("2023-01-02", periods=6, freq="B")
        levels = 100 * np.cumprod([0.9] * 6)
        price = pd.DataFrame({f"S{i}": levels for i in range(4)}, index=dates)
        rets = pd.Series(0.0, index=dates)
        exposure = runner._compute_exposure_by_date(rets, list(dates[2:]), price, list(price.columns))
        assert all(v == 0.0 for v in exposure.values())
