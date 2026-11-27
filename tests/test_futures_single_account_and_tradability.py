"""End-to-end and counter-example tests for P0-4 (futures single-account closed loop)
and P1-1 (unified futures order tradability check layer).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import pytest

from factor_synthesis.vectorbt_adapter import (
    FuturesExecutionConfig,
    VectorBTBacktestRunner,
    futures_execution as fut_exec_mod,
)
from factor_synthesis.vectorbt_adapter.futures_execution import (
    simulate_futures_contract_portfolio,
)
from factor_synthesis.vectorbt_adapter.portfolio_policy import SharedPortfolioPolicy
from slots.cn_futures import build_cn_futures_slot
from utils.slots.contracts import ExecutionProfile, InstrumentSpec, MarketSlot
from utils.time_axis import Frequency

from test_futures_contract_execution import (
    _build_rolling_futures_db,
    _read_continuous_wide,
)


def _make_mock_futures_slot(
    *,
    dates: pd.DatetimeIndex,
    instruments: List[str],
    open_panel: Dict[str, List[float]],
    close_panel: Dict[str, List[float]],
    volume_panel: Optional[Dict[str, List[float]]] = None,
    contract_map: Optional[Dict[Tuple[int, str], str]] = None,
    specs: Optional[Dict[str, InstrumentSpec]] = None,
    tradability_fn: Optional[Callable[[pd.DataFrame, str], pd.Series]] = None,
    allow_short: bool = True,
    use_open: bool = True,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> MarketSlot:
    """Build a deterministic in-memory futures ``MarketSlot`` for closed-loop tests."""
    utc_dates = pd.DatetimeIndex(dates).tz_localize("UTC") if dates.tz is None else dates.tz_convert("UTC")
    spec_map = dict(specs or {})
    for inst in instruments:
        spec_map.setdefault(
            inst, InstrumentSpec(inst, "futures", "CNY", lot_size=1, contract_multiplier=10.0)
        )
    if contract_map:
        for c_id in set(contract_map.values()):
            spec_map.setdefault(
                c_id, InstrumentSpec(c_id, "futures", "CNY", lot_size=1, contract_multiplier=10.0)
            )

    class _MockCalendar:
        @property
        def calendar_hash(self) -> str:
            return "mock_fut_cal"

        @property
        def start(self) -> pd.Timestamp:
            return pd.Timestamp(dates[0]).normalize()

        @property
        def end(self) -> pd.Timestamp:
            return pd.Timestamp(dates[-1]).normalize()

        def sessions(self, start=None, end=None):
            return pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates])

        def align(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame

        def session_at(self, ts_utc, view="ETH", *, frequency=None):
            return None

        def bar_grid(self, freq: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "session_id": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
                    "timestamp_utc": utc_dates,
                }
            )

    class _MockUniverse:
        def resolve_universe(self, universe_id: str, *, as_of_utc) -> Tuple[str, ...]:
            return tuple(instruments)

        def resolve_active_contract(self, instrument_id: str, *, as_of_utc) -> str:
            if contract_map:
                ts = pd.Timestamp(as_of_utc)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                for pos, u in enumerate(utc_dates):
                    if u == ts and (pos, instrument_id) in contract_map:
                        return contract_map[(pos, instrument_id)]
            return instrument_id

        def instrument_spec(self, contract_id: str) -> InstrumentSpec:
            return spec_map[contract_id]

    class _MockBarFeed:
        def read_signal_bars(self, req_insts, frequency, *, start_utc, end_utc, as_of_utc) -> pd.DataFrame:
            return pd.DataFrame()

        def read_execution_bars(self, req_insts, frequency, *, start_utc, end_utc) -> pd.DataFrame:
            rows = []
            start_ts = pd.Timestamp(start_utc)
            end_ts = pd.Timestamp(end_utc)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            for pos, (d, utc_ts) in enumerate(zip(dates, utc_dates)):
                if utc_ts < start_ts or utc_ts > end_ts:
                    continue
                for inst in req_insts:
                    # Support reading either root instrument or concrete contract.
                    if inst in open_panel:
                        o_px = open_panel[inst][pos]
                        c_px = close_panel[inst][pos]
                        vol = volume_panel[inst][pos] if volume_panel else 1_000_000.0
                        c_id = contract_map.get((pos, inst), inst) if contract_map else inst
                    elif inst in spec_map:
                        # Concrete contract fallback lookup
                        parent = next(
                            (k[1] for k, v in (contract_map or {}).items() if v == inst),
                            inst,
                        )
                        if parent not in open_panel:
                            continue
                        o_px = open_panel[parent][pos]
                        c_px = close_panel[parent][pos]
                        vol = volume_panel[parent][pos] if volume_panel else 1_000_000.0
                        c_id = inst
                    else:
                        continue
                    rows.append(
                        {
                            "timestamp_utc": utc_ts,
                            "session_id": pd.Timestamp(d).strftime("%Y-%m-%d"),
                            "instrument_id": inst,
                            "concrete_instrument_id": c_id,
                            "open": o_px,
                            "close": c_px,
                            "volume": vol,
                            "money": vol * c_px,
                        }
                    )
            return pd.DataFrame(rows)

    class _MockAux:
        def read_aux_features(self, grid, *, features):
            return pd.DataFrame()

    class _MockMicro:
        @property
        def execution_profile(self) -> ExecutionProfile:
            return ExecutionProfile(
                entry_price_field="open" if use_open else "close",
                exit_price_field="open" if use_open else "close",
                min_entry_offset_bars=1,
                exit_adjustment=0,
                return_unit="decimal",
                gap_rule="endpoints_only",
                boundary_policy="cross_session",
                allow_short=allow_short,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )

        def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
            if tradability_fn is not None:
                return tradability_fn(grid, instrument_id)
            return pd.Series(True, index=grid.index, dtype=bool)

    return MarketSlot(
        slot_id="cn_futures",
        calendar=_MockCalendar(),
        universe=_MockUniverse(),
        bar_feed=_MockBarFeed(),
        aux_features=_MockAux(),
        microstructure=_MockMicro(),
    )


def test_p0_4_no_decode_helper_and_futures_skips_nominal_presim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0-4: _decode_runner_target_weights is gone and futures mode never runs nominal pre-sim."""
    assert not hasattr(fut_exec_mod, "_decode_runner_target_weights")

    db_path = tmp_path / "futures_single_acct.duckdb"
    roll_start, end, _, _ = _build_rolling_futures_db(db_path, jump_pct=0.10)
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(slot, "CU")

        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 1,
                    "rebalance_freq": "daily",
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.5,
                }
            },
            slot=slot,
            frequency="1d",
        )

        def _forbid_nominal_presim(*args, **kwargs):
            raise AssertionError("Futures mode must not execute nominal _compute_target_weights")

        monkeypatch.setattr(runner, "_compute_target_weights", _forbid_nominal_presim)
        res = runner.run_backtest(factor_wide, close_wide, open_df=open_wide, rebalance_freq="daily")
        runner.close()
    finally:
        conn.close()

    assert "error" not in res, res
    assert res["futures_execution"]["single_account_closed_loop"] is True
    assert res["futures_execution"]["tradability_checks_enabled"] is True
    assert res["futures_execution"]["roll_count"] == 1


def test_p0_4_lot_truncation_to_zero_feeds_back_into_partial_rebalance_and_strict_holdings() -> None:
    """Large contract multiplier flooring BIG to 0 lots updates partial_rebalance & strict_holdings."""
    dates = pd.bdate_range("2026-01-05", periods=4)
    # BIG: 4000 * 300 = 1,200,000 / lot -> floored to 0 lots under 100k init_cash!
    # SMALL: 100 * 10 = 1,000 / lot -> fills 50 lots (50% weight).
    open_p = {"BIG": [4000.0] * 4, "SMALL": [100.0] * 4}
    close_p = {"BIG": [4000.0] * 4, "SMALL": [100.0] * 4}
    specs = {
        "BIG": InstrumentSpec("BIG", "futures", "CNY", lot_size=1, contract_multiplier=300.0),
        "SMALL": InstrumentSpec("SMALL", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
    }
    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["BIG", "SMALL"],
        open_panel=open_p,
        close_panel=close_p,
        specs=specs,
        allow_short=False,
    )

    scores = pd.DataFrame(
        {"BIG": [2.0, 2.0, 2.0, 2.0], "SMALL": [1.0, 1.0, 1.0, 1.0]},
        index=dates,
    )
    close_df = pd.DataFrame(close_p, index=dates)
    open_df = pd.DataFrame(open_p, index=dates)

    # 1. Partial rebalance (trade_rate=0.5):
    # Bar 1 enters: BIG=0 lots (floored), SMALL=50 lots (50k / 100k = 0.5).
    # Bar 2 rebalances: real book ONLY holds SMALL, so prev_norm={"SMALL": 1.0}.
    # Blending 0.5 * {"SMALL": 1.0} + 0.5 * {"BIG": 0.5, "SMALL": 0.5} -> {"BIG": 0.25, "SMALL": 0.75}!
    # Therefore SMALL MUST increase to 75 lots on Bar 2 (dual-account would keep 50 lots!).
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 1.0,
                "lot_rounding": "floor",
                "allow_short": False,
                "partial_rebalance": {"enabled": True, "trade_rate": 0.5, "min_weight": 0.01},
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        close_df,
        open_df=open_df,
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    pf = res["portfolio"]
    assert list(pf.lots_history["BIG"]) == [0, 0, 0, 0]
    assert pf.lots_history.loc[dates[1], "SMALL"] == 50
    assert pf.lots_history.loc[dates[2], "SMALL"] == 75

    # 2. strict_max_actual_holdings=True with top_k=1:
    # Bar 1 wants BIG (floored to 0 lots). On Bar 2, scores flip to SMALL > BIG;
    # because BIG is actually 0 lots (NOT a phantom holding), SMALL enters cleanly!
    scores_flip = pd.DataFrame(
        {"BIG": [2.0, 1.0, 1.0, 1.0], "SMALL": [1.0, 2.0, 2.0, 2.0]},
        index=dates,
    )
    runner_strict = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 1.0,
                "lot_rounding": "floor",
                "allow_short": False,
                "strict_max_actual_holdings": True,
            }
        },
        slot=slot,
        frequency="1d",
    )
    res_strict = runner_strict.run_backtest(
        scores_flip,
        close_df,
        open_df=open_df,
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner_strict.close()
    assert "error" not in res_strict, res_strict
    assert res_strict["portfolio"].lots_history.loc[dates[1], "BIG"] == 0
    assert res_strict["portfolio"].lots_history.loc[dates[2], "SMALL"] == 100
    assert res_strict["execution_diagnostics"]["strict_position_buys_skipped"] == 0


def test_p0_4_margin_clipping_feeds_back_into_mvo_and_partial_rebalance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Margin clipping on bar 1 must feed real clipped weights to MVO & partial_rebalance on bar 2."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    open_p = {"A": [100.0] * 5, "B": [100.0] * 5}
    close_p = {"A": [100.0] * 5, "B": [100.0] * 5}
    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["A", "B"],
        open_panel=open_p,
        close_panel=close_p,
        allow_short=False,
    )
    scores = pd.DataFrame({"A": [2.0] * 5, "B": [1.0] * 5}, index=dates)
    close_df = pd.DataFrame(close_p, index=dates)
    open_df = pd.DataFrame(open_p, index=dates)

    # Margin rate = 0.80, leverage = 1.5 -> MVO falls back to signal weights (A=2/3, B=1/3) on <40 bars:
    # wants 100k A (100 lots) + 50k B (50 lots) = 150k notional.
    # Required margin = 120k > 100k equity -> margin-clipped!
    # Scale = 100k / 120k = 0.83333 -> floor(100 * 0.83333) = 83 lots A, floor(50 * 0.83333) = 41 lots B.
    # Real prev_weight_map at bar 2 is notional / (equity * leverage) (NOT 2/3, 1/3)!
    captured_prev_weights: List[Dict[str, float]] = []
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 0.80,
                "leverage": 1.5,
                "insufficient_margin_policy": "clip",
                "allow_short": False,
                "weighting": {"scheme": "mvo"},
            }
        },
        slot=slot,
        frequency="1d",
    )
    orig_mvo = runner._mvo_weights

    def _spy_mvo(selected, score_series, price_df, as_of_date, prev_weights):
        captured_prev_weights.append(dict(prev_weights))
        return orig_mvo(selected, score_series, price_df, as_of_date, prev_weights)

    monkeypatch.setattr(runner, "_mvo_weights", _spy_mvo)
    res = runner.run_backtest(
        scores,
        close_df,
        open_df=open_df,
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    assert res["futures_execution"]["margin_clipped_events"] >= 1
    assert res["portfolio"].lots_history.loc[dates[1], "A"] == 83
    assert res["portfolio"].lots_history.loc[dates[1], "B"] == 41
    assert len(captured_prev_weights) == 2
    assert captured_prev_weights[0] == {}
    assert captured_prev_weights[1]["A"] == pytest.approx((83 * 100.0 * 10.0) / (100_000.0 * 1.5), abs=1e-9)
    assert captured_prev_weights[1]["B"] == pytest.approx((41 * 100.0 * 10.0) / (100_000.0 * 1.5), abs=1e-9)


def test_p0_4_liquidation_and_roll_cost_feed_back_into_nav_and_liquidity_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Liquidation & roll costs update real equity passed to next rebalance and ADV liquidity_cap."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    # Bar 1 opens @ 100 -> 400 lots (4x lev, 400k notional, 80k initial margin).
    # Bar 1 closes @ 80 -> equity = 100k - 80k = 20k < maint_req (400*80*10*0.10 = 32k) -> LIQUIDATION!
    # Bar 2 opens @ 90 -> rebounds, but real account is liquidated (equity=20k, lots=0).
    # Bar 3 rebalances with partial_rebalance=0.5 & liquidity_cap:
    # must see equity=20k (NOT nominal 100k!) and empty book (so jumps 100% to target, 20k*4=80k -> 88 lots).
    open_p = {"FUT": [100.0, 100.0, 90.0, 90.0, 90.0]}
    close_p = {"FUT": [100.0, 80.0, 90.0, 90.0, 90.0]}
    vol_p = {"FUT": [1_000_000.0] * 5}
    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        volume_panel=vol_p,
        allow_short=False,
    )
    scores = pd.DataFrame({"FUT": [1.0] * 5}, index=dates)
    close_df = pd.DataFrame(close_p, index=dates)
    open_df = pd.DataFrame(open_p, index=dates)

    captured_capitals: List[float] = []
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 0.20,
                "maintenance_margin_rate": 0.10,
                "leverage": 4.0,
                "liquidation_policy": "liquidate",
                "allow_short": False,
                "partial_rebalance": {"enabled": True, "trade_rate": 0.5, "min_weight": 0.01},
                "liquidity_cap": {"enabled": True, "max_participation": 0.5, "adv_lookback_days": 1},
            }
        },
        slot=slot,
        frequency="1d",
    )
    orig_liq_caps = runner._liquidity_weight_caps

    def _spy_liq_caps(stocks, as_of_date_str, deployed_capital):
        captured_capitals.append(float(deployed_capital))
        return orig_liq_caps(stocks, as_of_date_str, deployed_capital)

    monkeypatch.setattr(runner, "_liquidity_weight_caps", _spy_liq_caps)
    res = runner.run_backtest(
        scores,
        close_df,
        open_df=open_df,
        rebalance_dates=[dates[1].to_pydatetime(), dates[3].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    pf = res["portfolio"]
    assert res["futures_execution"]["liquidation_count"] == 1
    assert pf.lots_history.loc[dates[1], "FUT"] == 0
    assert pf.lots_history.loc[dates[2], "FUT"] == 0
    # Bar 3 re-enters based on real post-liquidation equity = 20,000 * 4.0 = 80,000 -> floor(80k / 900) = 88 lots.
    assert pf.lots_history.loc[dates[3], "FUT"] == 88
    # _liquidity_weight_caps receives equity * leverage (2 calls per rebalance: _apply_liquidity_cap + dollar_caps).
    assert captured_capitals[0] == pytest.approx(400_000.0)
    assert captured_capitals[-1] == pytest.approx(80_000.0)


def test_p0_4_risk_overlay_two_pass_uses_real_futures_returns() -> None:
    """Two-pass risk_overlay computes drawdown/vol exposure from real futures pass-1 returns."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    # Bar 1 enters 400 lots @ 100 (4x lev); bar 1 close drops to 95 -> real drawdown = -20% (4x of -5%)!
    # Nominal 1x equity would only see -5% drawdown and miss derisk_dd=-0.10!
    open_p = {"FUT": [100.0, 100.0, 95.0, 95.0, 95.0]}
    close_p = {"FUT": [100.0, 95.0, 95.0, 95.0, 95.0]}
    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        allow_short=False,
    )
    scores = pd.DataFrame({"FUT": [1.0] * 5}, index=dates)
    close_df = pd.DataFrame(close_p, index=dates)
    open_df = pd.DataFrame(open_p, index=dates)

    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 0.20,
                "leverage": 4.0,
                "allow_short": False,
                "risk_overlay": {
                    "drawdown_control": {
                        "enabled": True,
                        "derisk_dd": -0.10,
                        "derisk_exposure": 0.5,
                        "exit_dd": -0.50,
                        "reentry_dd": -0.02,
                    }
                },
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        close_df,
        open_df=open_df,
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    # On bar 2, real pass-1 drawdown was -20% <= -10%, so exposure=0.5!
    assert runner._last_exposure_by_date[pd.Timestamp(dates[2])] == pytest.approx(0.5)
    # Equity at bar 2 open = 80,000; 0.5 exposure * 4.0 leverage = 160,000 / 950 = 168 lots!
    assert res["portfolio"].lots_history.loc[dates[2], "FUT"] == 168


def test_p1_1_roll_close_blocked_defers_roll_and_holds_old_contract(tmp_path: Path) -> None:
    """P1-1: Suspended old contract at roll bar defers roll and preserves old contract position."""
    db_path = tmp_path / "futures_roll_close_blocked.duckdb"
    roll_start, end, c1, c2 = _build_rolling_futures_db(db_path, jump_pct=0.10)

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        base_slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(base_slot, "CU")
        # Roll bar is index 3 (2026-01-09). Block C1 on 2026-01-09, unblock on 2026-01-12 (index 4).
        blocked_day = close_wide.index[3].strftime("%Y-%m-%d")

        class _SuspendedOldMicro:
            def __init__(self, inner):
                self._inner = inner
                self.execution_profile = inner.execution_profile
                self.portfolio_execution_profile = getattr(inner, "portfolio_execution_profile", None)

            def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
                base = self._inner.tradability_mask(grid, instrument_id=instrument_id)
                if instrument_id == c1:
                    return base & (grid["session_id"].astype(str) != blocked_day)
                return base

        slot = replace(base_slot, microstructure=_SuspendedOldMicro(base_slot.microstructure))
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.5,
                }
            },
            slot=slot,
            frequency="1d",
        )
        res = runner.run_backtest(
            factor_wide,
            close_wide,
            open_df=open_wide,
            rebalance_dates=[close_wide.index[1].to_pydatetime()],
        )
        runner.close()
    finally:
        conn.close()

    assert "error" not in res, res
    pf = res["portfolio"]
    diag = res["futures_execution"]
    # Entered on bar 1 (96 lots of C1 @ 104).
    lots_bar1 = int(pf.lots_history.iloc[1, 0])
    assert lots_bar1 > 0
    # Bar 3 (blocked roll day): C1 still held, C2 NOT opened, margin still occupied!
    assert pf.lots_history.iloc[3, 0] == lots_bar1
    assert pf.margin_used.iloc[3] > 0.0
    assert diag["roll_deferred_count"] == 1
    assert diag["roll_close_blocked_count"] == 1
    assert diag["open_deferred_rolls_at_end"] == 0
    assert diag["roll_count"] == 1
    # Roll completed on bar 4 (next tradable bar for C1).
    assert pd.Timestamp(diag["roll_events"][0]["timestamp"]) == close_wide.index[4]
    assert pd.Timestamp(diag["roll_events"][0]["deferred_from"]) == close_wide.index[3]
    assert diag["roll_events"][0]["deferred_bars"] == 1
    assert any(
        e["order_type"] == "ROLL_CLOSE" and e["contract"] == c1 and e["status"] == "deferred"
        for e in diag["blocked_order_events"]
    )
    logs = pf.logs.records_readable
    assert (logs["Result Status"] == "Rejected").any()


@pytest.mark.parametrize("direction", ["long", "short"])
def test_p1_1_roll_open_blocked_by_limit_up_releases_margin_and_respects_side(
    tmp_path: Path, direction: str
) -> None:
    """P1-1: Limit-up C2 blocks long ROLL_OPEN (releasing margin) while allowing short ROLL_OPEN."""
    db_path = tmp_path / f"futures_roll_open_{direction}.duckdb"
    roll_start, end, c1, c2 = _build_rolling_futures_db(db_path, jump_pct=0.10)

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        base_slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(
            base_slot, "CU", score_value=(1.0 if direction == "long" else -1.0)
        )
        roll_day = close_wide.index[3].strftime("%Y-%m-%d")

        class _LimitUpNewMicro:
            def __init__(self, inner):
                self._inner = inner
                self.execution_profile = inner.execution_profile
                self.portfolio_execution_profile = getattr(inner, "portfolio_execution_profile", None)

            def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
                base = self._inner.tradability_mask(grid, instrument_id=instrument_id)
                if instrument_id == c2 and "side" in grid.columns:
                    blocked = (grid["session_id"].astype(str) == roll_day) & (grid["side"] == "buy")
                    return base & ~blocked
                return base

        slot = replace(base_slot, microstructure=_LimitUpNewMicro(base_slot.microstructure))
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.5,
                }
            },
            slot=slot,
            frequency="1d",
        )
        res = runner.run_backtest(
            factor_wide,
            close_wide,
            open_df=open_wide,
            rebalance_dates=[close_wide.index[1].to_pydatetime()],
        )
        runner.close()
    finally:
        conn.close()

    assert "error" not in res, res
    pf = res["portfolio"]
    diag = res["futures_execution"]
    assert diag["roll_count"] == 1
    if direction == "long":
        # C1 closed, C2 buy blocked -> flat with margin released!
        assert pf.lots_history.iloc[3, 0] == 0
        assert pf.margin_used.iloc[3] == pytest.approx(0.0)
        assert diag["roll_open_blocked_count"] == 1
        assert diag["roll_events"][0]["opened_lots"] == 0
        assert diag["roll_events"][0]["open_blocked"] is True
        assert any(
            e["order_type"] == "ROLL_OPEN" and e["contract"] == c2 and e["side"] == "buy"
            for e in diag["blocked_order_events"]
        )
    else:
        # Short roll sells C2 (allowed at limit-up) -> opens short C2 normally!
        assert pf.lots_history.iloc[3, 0] < 0
        assert pf.margin_used.iloc[3] > 0.0
        assert diag["roll_open_blocked_count"] == 0
        assert diag["roll_events"][0]["open_blocked"] is False


def test_p1_1_liquidation_blocked_bar_defers_to_next_tradable_bar() -> None:
    """P1-1: Maintenance liquidation on non-tradable bar defers until next tradable bar."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    # Bar 0: enter 400 lots @ 100.
    # Bar 1: close drops to 80 (equity=20k < maint=32k), sell blocked on bar 1 & bar 2!
    # Bar 2: still blocked, close drops to 78 (equity=12k).
    # Bar 3: tradable at open=76 -> deferred LIQUIDATION executes at 76 (equity = 100k + 400*(76-100)*10 = 4k)!
    open_p = {"FUT": [100.0, 100.0, 80.0, 76.0, 90.0]}
    close_p = {"FUT": [100.0, 80.0, 78.0, 85.0, 90.0]}
    blocked_days = {dates[1].strftime("%Y-%m-%d"), dates[2].strftime("%Y-%m-%d")}

    def _mask_fn(grid: pd.DataFrame, instrument_id: str) -> pd.Series:
        is_blocked = grid["session_id"].astype(str).isin(blocked_days)
        if "side" in grid.columns:
            return ~(is_blocked & (grid["side"] == "sell"))
        return ~is_blocked

    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        tradability_fn=_mask_fn,
        allow_short=False,
    )
    target_weights = pd.DataFrame({"FUT": [1.0, np.nan, np.nan, np.nan, np.nan]}, index=dates)
    cfg = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=0.20,
        maintenance_margin_rate=0.10,
        leverage=4.0,
        liquidation_policy="liquidate",
    )
    pf, diag = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=pd.DataFrame(close_p, index=dates),
        open_aligned=pd.DataFrame(open_p, index=dates),
        use_open=True,
        slot=slot,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg,
        target_value_mode="target_weight",
    )

    # Bar 1 & Bar 2 could not liquidate: 400 lots remained open with margin occupied.
    assert pf.lots_history.loc[dates[0], "FUT"] == 400
    assert pf.lots_history.loc[dates[1], "FUT"] == 400
    assert pf.lots_history.loc[dates[2], "FUT"] == 400
    assert pf.margin_used.loc[dates[1]] > 0.0
    assert pf.margin_used.loc[dates[2]] > 0.0

    # Bar 3 is first tradable bar -> deferred liquidation executes at Bar 3 open=76.0!
    assert pf.lots_history.loc[dates[3], "FUT"] == 0
    assert pf.margin_used.loc[dates[3]] == pytest.approx(0.0)
    assert pf.value().iloc[-1] == pytest.approx(4_000.0)
    assert diag["liquidation_count"] == 1
    assert diag["liquidation_blocked_count"] == 2
    assert diag["liquidation_deferred_executed_count"] == 1
    assert diag["open_pending_liquidations_at_end"] == 0
    assert any(evt.get("status") == "deferred" for evt in diag["liquidation_events"])
    assert any(evt.get("status") == "deferred_executed" for evt in diag["liquidation_events"])
    assert any(
        e["order_type"] == "LIQUIDATION" and e["status"] == "deferred"
        for e in diag["blocked_order_events"]
    )
    liq_orders = pf.orders.records_readable
    liq_row = liq_orders.loc[liq_orders["Order Type"] == "LIQUIDATION"].iloc[0]
    assert pd.Timestamp(liq_row["Timestamp"]) == dates[3]
    assert liq_row["Price"] == pytest.approx(76.0)


def test_p1_1_rebalance_exit_and_entry_enforce_unified_tradability_layer() -> None:
    """P1-1: Strategy rebalance entries and exits pass through unified tradability_mask checks."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    # A: blocked for buy on bar 1; B: tradable on bar 1, blocked for sell on bar 2, tradable on bar 3.
    open_p = {"A": [100.0] * 5, "B": [100.0] * 5}
    close_p = {"A": [100.0] * 5, "B": [100.0] * 5}
    d1 = dates[1].strftime("%Y-%m-%d")
    d2 = dates[2].strftime("%Y-%m-%d")

    def _mask_fn(grid: pd.DataFrame, instrument_id: str) -> pd.Series:
        sids = grid["session_id"].astype(str)
        side = grid["side"] if "side" in grid.columns else pd.Series("", index=grid.index)
        if instrument_id == "A":
            return ~((sids == d1) & (side == "buy"))
        if instrument_id == "B":
            return ~((sids == d2) & (side == "sell"))
        return pd.Series(True, index=grid.index, dtype=bool)

    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["A", "B"],
        open_panel=open_p,
        close_panel=close_p,
        tradability_fn=_mask_fn,
        allow_short=False,
    )
    # Bar 1 wants A (blocked) or B? Make enable_tday_filter=False so policy tries to buy A -> unified validator rejects A!
    # Also on Bar 1 select B, on Bar 2 drop B -> exit blocked on Bar 2, retried & closed on Bar 3!
    scores = pd.DataFrame(
        {
            "A": [2.0, 2.0, 0.5, 0.5, 0.5],
            "B": [1.0, 1.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )
    # Use pools: bar 1 pool=["A", "B"] with top_k=2 -> tries A (buy blocked) & B (bought!).
    # bar 2 pool=["A"] -> wants to sell B (exit blocked on d2!), buys A on d2; bar 3 non-rebalance retries selling B!
    pools = {
        dates[1].strftime("%Y-%m-%d"): ["A", "B"],
        dates[2].strftime("%Y-%m-%d"): ["A"],
    }
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 1.0,
                "enable_tday_filter": False,
                "allow_short": False,
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates),
        dynamic_pools=pools,
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    pf = res["portfolio"]
    # Bar 1: A buy blocked (0 lots), B bought (50 lots).
    assert pf.lots_history.loc[dates[1], "A"] == 0
    assert pf.lots_history.loc[dates[1], "B"] == 50
    # Bar 2: B exit blocked (remains 50 lots), A bought.
    assert pf.lots_history.loc[dates[2], "B"] == 50
    # Bar 3: non-rebalance retry closes B when sell becomes tradable!
    assert pf.lots_history.loc[dates[3], "B"] == 0
    assert res["execution_diagnostics"]["blocked_exit_events"] == 1
    assert res["execution_diagnostics"]["completed_blocked_exits"] == 1
    assert res["futures_execution"]["blocked_order_count"] >= 2


def test_p1_1_cn_futures_exclude_delivery_month_blocks_rebalance_via_real_slot(
    tmp_path: Path,
) -> None:
    """Real cn_futures slot with exclude_delivery_month=True enforces tradability_mask in engine."""
    db_path = tmp_path / "futures_delivery_month.duckdb"
    roll_start, end, c1, c2 = _build_rolling_futures_db(db_path, jump_pct=0.10)
    # Mark c1's contract_month as 2026-01 (current session month) so real FuturesSlot
    # tradability_mask blocks c1 under exclude_delivery_month=True while allowing c2 (2026-04).
    with duckdb.connect(str(db_path)) as write_conn:
        write_conn.execute(
            "UPDATE instrument_metadata SET contract_month='2026-01' WHERE instrument_id=?",
            [c1],
        )

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_cn_futures_slot(
            conn, product_root="CU", start=roll_start, end=end, exclude_delivery_month=True
        )
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(slot, "CU")
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 1,
                    "rebalance_freq": "daily",
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.5,
                    "enable_tday_filter": False,
                }
            },
            slot=slot,
            frequency="1d",
        )
        res = runner.run_backtest(factor_wide, close_wide, open_df=open_wide, rebalance_freq="daily")
        runner.close()
    finally:
        conn.close()

    assert "error" not in res, res
    pf = res["portfolio"]
    diag = res["futures_execution"]
    assert diag["tradability_checks_enabled"] is True
    # Bars 1..2 have active contract c1 (delivery month 2026-01 -> blocked);
    # Bar 3+ rolls to c2 (2026-04 -> tradable -> opens!).
    assert pf.lots_history.iloc[1, 0] == 0
    assert pf.lots_history.iloc[2, 0] == 0
    assert pf.lots_history.iloc[3, 0] > 0
    assert diag["blocked_order_count"] >= 1
    assert any(
        e["contract"] == c1 and e["reason"] == "tradability_mask_blocked"
        for e in diag["blocked_order_events"]
    )


def test_p1_1_deferred_roll_with_exit_intent_completes_exit_on_non_rebalance_recovery() -> None:
    """Deferred roll overlapping a rebalance exit must complete the exit (no ROLL_OPEN) on recovery."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    open_p = {"A": [100.0] * 5, "B": [100.0] * 5}
    close_p = {"A": [100.0] * 5, "B": [100.0] * 5}
    contract_map = {
        (0, "A"): "A_C1",
        (1, "A"): "A_C1",
        (2, "A"): "A_C2",
        (3, "A"): "A_C2",
        (4, "A"): "A_C2",
        (0, "B"): "B_C1",
        (1, "B"): "B_C1",
        (2, "B"): "B_C1",
        (3, "B"): "B_C1",
        (4, "B"): "B_C1",
    }
    blocked_day = dates[2].strftime("%Y-%m-%d")

    def _mask_fn(grid: pd.DataFrame, instrument_id: str) -> pd.Series:
        if instrument_id == "A_C1":
            return grid["session_id"].astype(str) != blocked_day
        return pd.Series(True, index=grid.index, dtype=bool)

    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["A", "B"],
        open_panel=open_p,
        close_panel=close_p,
        contract_map=contract_map,
        tradability_fn=_mask_fn,
        allow_short=False,
    )
    # T-1 causal signal: bar 1 reads bar 0 (A>B), bar 2 reads bar 1 (B>A -> exit A).
    scores = pd.DataFrame(
        {
            "A": [2.0, 1.0, 1.0, 1.0, 1.0],
            "B": [1.0, 2.0, 2.0, 2.0, 2.0],
        },
        index=dates,
    )
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 0.30,
                "allow_short": False,
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates),
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res

    pf = res["portfolio"]
    diag = res["futures_execution"]
    exec_diag = res["execution_diagnostics"]

    assert pf.lots_history.loc[dates[1], "A"] == 100
    assert pf.lots_history.loc[dates[2], "A"] == 100
    # Bar 3 is non-rebalance recovery day: A_C1 closes and A_C2 must NOT be opened.
    assert pf.lots_history.loc[dates[3], "A"] == 0
    assert pf.lots_history.loc[dates[4], "A"] == 0
    assert pf.margin_used.loc[dates[3]] == pytest.approx(0.0)
    assert diag["roll_deferred_count"] == 1
    assert len(diag["roll_events"]) == 1
    assert diag["roll_events"][0]["closed_lots"] == 100
    assert diag["roll_events"][0]["opened_lots"] == 0
    orders = pf.orders.records_readable
    assert not ((orders["Column"] == "A") & (orders["Order Type"] == "ROLL_OPEN")).any()
    assert exec_diag["blocked_exit_events"] == 1
    assert exec_diag["completed_blocked_exits"] == 1
    assert exec_diag["open_blocked_exits_at_end"] == 0


def test_p1_1_deferred_roll_still_selected_on_rebalance_keeps_hold_and_rolls_later() -> None:
    """Deferred roll on a rebalance day must not trigger a false exit when still top-ranked."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    open_p = {"A": [100.0] * 5, "B": [100.0] * 5}
    close_p = {"A": [100.0] * 5, "B": [100.0] * 5}
    contract_map = {
        (0, "A"): "A_C1",
        (1, "A"): "A_C1",
        (2, "A"): "A_C2",
        (3, "A"): "A_C2",
        (4, "A"): "A_C2",
        (0, "B"): "B_C1",
        (1, "B"): "B_C1",
        (2, "B"): "B_C1",
        (3, "B"): "B_C1",
        (4, "B"): "B_C1",
    }
    blocked_day = dates[2].strftime("%Y-%m-%d")

    def _mask_fn(grid: pd.DataFrame, instrument_id: str) -> pd.Series:
        if instrument_id == "A_C1":
            return grid["session_id"].astype(str) != blocked_day
        return pd.Series(True, index=grid.index, dtype=bool)

    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["A", "B"],
        open_panel=open_p,
        close_panel=close_p,
        contract_map=contract_map,
        tradability_fn=_mask_fn,
        allow_short=False,
    )
    scores = pd.DataFrame({"A": [2.0] * 5, "B": [1.0] * 5}, index=dates)
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 0.30,
                "allow_short": False,
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates),
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res

    pf = res["portfolio"]
    diag = res["futures_execution"]
    exec_diag = res["execution_diagnostics"]

    assert pf.lots_history.loc[dates[1], "A"] == 100
    assert pf.lots_history.loc[dates[2], "A"] == 100
    # Bar 3: A was still selected -> no exit intent -> rolls cleanly from A_C1 to A_C2.
    assert pf.lots_history.loc[dates[3], "A"] == 100
    assert pf.lots_history.loc[dates[3], "B"] == 0
    assert exec_diag["blocked_exit_events"] == 0
    assert exec_diag["open_blocked_exits_at_end"] == 0
    assert len(diag["roll_events"]) == 1
    assert diag["roll_events"][0]["closed_lots"] == 100
    assert diag["roll_events"][0]["opened_lots"] == 100


def test_p0_4_matrix_mode_blocked_exit_retries_on_subsequent_tradable_bar() -> None:
    """Matrix mode (policy=None) retries deferred exits on subsequent NaN bars."""
    dates = pd.bdate_range("2026-01-05", periods=4)
    open_p = {"FUT": [100.0] * 4}
    close_p = {"FUT": [100.0] * 4}
    blocked_day = dates[1].strftime("%Y-%m-%d")

    def _mask_fn(grid: pd.DataFrame, instrument_id: str) -> pd.Series:
        is_blocked = grid["session_id"].astype(str) == blocked_day
        if "side" in grid.columns:
            return ~(is_blocked & (grid["side"] == "sell"))
        return ~is_blocked

    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        tradability_fn=_mask_fn,
        allow_short=False,
    )
    target_weights = pd.DataFrame({"FUT": [1.0, 0.0, np.nan, np.nan]}, index=dates)
    cfg = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=0.50,
        leverage=1.0,
    )
    pf, diag = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=pd.DataFrame(close_p, index=dates),
        open_aligned=pd.DataFrame(open_p, index=dates),
        use_open=True,
        slot=slot,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg,
        target_value_mode="target_weight",
    )

    assert pf.lots_history.loc[dates[0], "FUT"] == 100
    assert pf.lots_history.loc[dates[1], "FUT"] == 100
    assert pf.lots_history.loc[dates[2], "FUT"] == 0
    assert pf.lots_history.loc[dates[3], "FUT"] == 0
    assert diag["open_deferred_rolls_at_end"] == 0


def _build_clocked_futures_slot(
    tmp_path: Path,
    *,
    cal: Any,
    instruments: List[str],
    open_panel: Dict[str, List[float]],
    close_panel: Dict[str, List[float]],
    use_open: bool,
    halt_rows: Optional[List[Tuple[str, str, str]]] = None,
    recorded_queries: Optional[List[Dict[str, Any]]] = None,
) -> MarketSlot:
    """Build a clocked futures MarketSlot backed by DeclarativeMicrostructure."""
    from slots.declarative.slot import DeclarativeMicrostructure
    from slots.declarative.source import FrameSource
    from utils.slots import ProviderConfig

    grid_1d = cal.bar_grid("1d")
    session_dates = list(grid_1d["session_id"])
    utc_closes = pd.DatetimeIndex(grid_1d["timestamp_utc"]).tz_convert("UTC")
    spec_map = {
        inst: InstrumentSpec(inst, "futures", "USD", lot_size=1, contract_multiplier=10.0)
        for inst in instruments
    }

    class _ClockedUniverse:
        def resolve_universe(self, universe_id: str, *, as_of_utc) -> Tuple[str, ...]:
            return tuple(instruments)

        def resolve_active_contract(self, instrument_id: str, *, as_of_utc) -> str:
            return instrument_id

        def instrument_spec(self, contract_id: str) -> InstrumentSpec:
            return spec_map[contract_id]

    class _ClockedBarFeed:
        def read_signal_bars(self, req_insts, frequency, *, start_utc, end_utc, as_of_utc) -> pd.DataFrame:
            return pd.DataFrame()

        def read_execution_bars(self, req_insts, frequency, *, start_utc, end_utc) -> pd.DataFrame:
            start_ts = pd.Timestamp(start_utc)
            end_ts = pd.Timestamp(end_utc)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            rows = []
            for pos, (sid, utc_ts) in enumerate(zip(session_dates, utc_closes)):
                if utc_ts < start_ts or utc_ts > end_ts:
                    continue
                for inst in req_insts:
                    rows.append(
                        {
                            "timestamp_utc": utc_ts,
                            "session_id": sid,
                            "instrument_id": inst,
                            "concrete_instrument_id": inst,
                            "open": open_panel[inst][pos],
                            "close": close_panel[inst][pos],
                            "volume": 1_000_000.0,
                        }
                    )
            return pd.DataFrame(rows)

    class _EmptyAux:
        def read_aux_features(self, grid, *, features):
            return pd.DataFrame()

    source = None
    if halt_rows:
        csv_name = f"halts_{len(halt_rows)}_{abs(hash(tuple(halt_rows)))}.csv"
        csv_lines = ["instrument_id,available_at_utc,tradable"]
        for inst_id, ts_str, tradable_str in halt_rows:
            csv_lines.append(f"{inst_id},{ts_str},{tradable_str}")
        (tmp_path / csv_name).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        source = FrameSource(
            ProviderConfig("csv_files", {"paths": [csv_name]}),
            tmp_path,
            ("instrument_id", "available_at_utc"),
        )

    profile = ExecutionProfile(
        entry_price_field="open" if use_open else "close",
        exit_price_field="open" if use_open else "close",
        min_entry_offset_bars=1,
        exit_adjustment=0,
        return_unit="decimal",
        gap_rule="endpoints_only",
        boundary_policy="cross_session",
        allow_short=False,
    )
    inner_micro = DeclarativeMicrostructure(cal, profile, source)

    class _RecordingMicro:
        def __init__(self, inner: DeclarativeMicrostructure) -> None:
            self._inner = inner
            self.execution_profile = inner.execution_profile

        def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
            mask = self._inner.tradability_mask(grid, instrument_id=instrument_id)
            if recorded_queries is not None:
                for row_idx, row in grid.iterrows():
                    recorded_queries.append(
                        {
                            "instrument_id": instrument_id,
                            "timestamp_utc": pd.Timestamp(row["timestamp_utc"]),
                            "session_id": str(row["session_id"]),
                            "execution_price_field": row.get("execution_price_field"),
                            "side": row.get("side"),
                            "allowed": bool(mask.loc[row_idx]),
                        }
                    )
            return mask

    return MarketSlot(
        slot_id="global_futures",
        calendar=cal,
        universe=_ClockedUniverse(),
        bar_feed=_ClockedBarFeed(),
        aux_features=_EmptyAux(),
        microstructure=_RecordingMicro(inner_micro),
    )


def test_p1_2_open_execution_queries_open_timestamp_and_ignores_future_intraday_restriction(
    tmp_path: Path,
) -> None:
    """NERE-1395: Open orders query bar open timestamp and must not leak mid-session restrictions."""
    from utils.session_calendar import SessionCalendar

    cal = SessionCalendar(
        "SYNTH_FUT",
        "2026-01-05",
        "2026-01-09",
        custom_sessions={
            "timezone": "UTC",
            "weekdays": [0, 1, 2, 3, 4],
            "segments": [{"open": "00:00", "close": "16:00"}],
        },
    )
    dates = cal.sessions()
    open_p = {"FUT": [100.0, 102.0, 105.0, 107.0, 109.0]}
    close_p = {"FUT": [101.0, 104.0, 106.0, 108.0, 110.0]}
    open_df = pd.DataFrame(open_p, index=dates)
    close_df = pd.DataFrame(close_p, index=dates)
    target_weights = pd.DataFrame({"FUT": [np.nan, 1.0, np.nan, np.nan, np.nan]}, index=dates)
    cfg = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=0.50,
        leverage=1.0,
    )

    # 1. Baseline without mid-day restriction event (use_open=True).
    base_slot = _build_clocked_futures_slot(
        tmp_path,
        cal=cal,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        use_open=True,
        halt_rows=None,
    )
    pf_base, diag_base = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=close_df,
        open_aligned=open_df,
        use_open=True,
        slot=base_slot,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg,
        target_value_mode="target_weight",
    )

    # 2. Counter-example: 12:00Z tradable=False event on 2026-01-06 (restored 2026-01-07 00:00Z).
    # Open fill occurs at 00:00Z so it MUST remain identical to baseline.
    open_queries: List[Dict[str, Any]] = []
    open_slot_with_midday_halt = _build_clocked_futures_slot(
        tmp_path,
        cal=cal,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        use_open=True,
        halt_rows=[
            ("FUT", "2026-01-06T12:00:00Z", "false"),
            ("FUT", "2026-01-07T00:00:00Z", "true"),
        ],
        recorded_queries=open_queries,
    )
    pf_open, diag_open = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=close_df,
        open_aligned=open_df,
        use_open=True,
        slot=open_slot_with_midday_halt,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg,
        target_value_mode="target_weight",
    )

    pd.testing.assert_frame_equal(pf_open.lots_history, pf_base.lots_history)
    pd.testing.assert_frame_equal(pf_open.orders.records_readable, pf_base.orders.records_readable)
    pd.testing.assert_series_equal(pf_open.value(), pf_base.value())
    assert diag_open["blocked_order_count"] == 0
    assert pf_open.lots_history.loc[dates[1], "FUT"] > 0
    bar1_open_buy_queries = [
        q for q in open_queries if q["session_id"] == "2026-01-06" and q["side"] == "buy"
    ]
    assert len(bar1_open_buy_queries) >= 1
    assert bar1_open_buy_queries[0]["timestamp_utc"] == pd.Timestamp("2026-01-06T00:00:00Z")
    assert bar1_open_buy_queries[0]["execution_price_field"] == "open"
    assert bar1_open_buy_queries[0]["allowed"] is True

    # 3. Control A: use_open=False (close execution at 16:00Z) MUST be blocked by the 12:00Z event.
    close_queries: List[Dict[str, Any]] = []
    close_slot_with_midday_halt = _build_clocked_futures_slot(
        tmp_path,
        cal=cal,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        use_open=False,
        halt_rows=[
            ("FUT", "2026-01-06T12:00:00Z", "false"),
            ("FUT", "2026-01-07T00:00:00Z", "true"),
        ],
        recorded_queries=close_queries,
    )
    pf_close, diag_close = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=close_df,
        open_aligned=None,
        use_open=False,
        slot=close_slot_with_midday_halt,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg,
        target_value_mode="target_weight",
    )
    assert pf_close.lots_history.loc[dates[1], "FUT"] == 0
    assert diag_close["blocked_order_count"] >= 1
    assert any(
        e["reason"] == "tradability_mask_blocked" and e["side"] == "buy"
        for e in diag_close["blocked_order_events"]
    )
    bar1_close_buy_queries = [
        q for q in close_queries if q["session_id"] == "2026-01-06" and q["side"] == "buy"
    ]
    assert len(bar1_close_buy_queries) >= 1
    assert bar1_close_buy_queries[0]["timestamp_utc"] == pd.Timestamp("2026-01-06T16:00:00Z")
    assert bar1_close_buy_queries[0]["execution_price_field"] == "close"
    assert bar1_close_buy_queries[0]["allowed"] is False

    # 4. Control B: restriction effective at 00:00Z (at/before open) MUST block use_open=True buy.
    open_blocked_queries: List[Dict[str, Any]] = []
    open_slot_with_open_halt = _build_clocked_futures_slot(
        tmp_path,
        cal=cal,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        use_open=True,
        halt_rows=[
            ("FUT", "2026-01-06T00:00:00Z", "false"),
            ("FUT", "2026-01-07T00:00:00Z", "true"),
        ],
        recorded_queries=open_blocked_queries,
    )
    pf_open_blocked, diag_open_blocked = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=close_df,
        open_aligned=open_df,
        use_open=True,
        slot=open_slot_with_open_halt,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg,
        target_value_mode="target_weight",
    )
    assert pf_open_blocked.lots_history.loc[dates[1], "FUT"] == 0
    assert diag_open_blocked["blocked_order_count"] >= 1
    assert any(
        e["reason"] == "tradability_mask_blocked" and e["side"] == "buy"
        for e in diag_open_blocked["blocked_order_events"]
    )
    bar1_open_blocked = [
        q for q in open_blocked_queries if q["session_id"] == "2026-01-06" and q["side"] == "buy"
    ]
    assert bar1_open_blocked[0]["timestamp_utc"] == pd.Timestamp("2026-01-06T00:00:00Z")
    assert bar1_open_blocked[0]["execution_price_field"] == "open"
    assert bar1_open_blocked[0]["allowed"] is False


def test_p1_2_same_bar_open_and_close_mask_cache_do_not_collide(tmp_path: Path) -> None:
    """Open entry at 00:00Z and close liquidation check at 16:00Z on the same bar use distinct cache keys."""
    from utils.session_calendar import SessionCalendar

    cal = SessionCalendar(
        "SYNTH_FUT",
        "2026-01-05",
        "2026-01-08",
        custom_sessions={
            "timezone": "UTC",
            "weekdays": [0, 1, 2, 3, 4],
            "segments": [{"open": "00:00", "close": "16:00"}],
        },
    )
    dates = cal.sessions()
    # Bar 0: buy 400 lots @ 100.0 open (100k * 4x / (100 * 10) = 400).
    # Bar 1: open=100.0 (tradable at 00:00Z), close drops to 80.0 -> equity=20k < maint=32k,
    # while 12:00Z restriction blocks close liquidation at 16:00Z!
    # Also add a pre-check on Bar 1 with side="sell" at open? Even better: enter Bar 1 at open=100,
    # then close=80 triggers Step 4.7 close liquidation on Bar 1; and verify via direct validator
    # that querying open vs close on the same bar+side does not collide.
    open_p = {"FUT": [100.0, 100.0, 76.0, 90.0]}
    close_p = {"FUT": [100.0, 80.0, 85.0, 90.0]}
    queries: List[Dict[str, Any]] = []
    slot = _build_clocked_futures_slot(
        tmp_path,
        cal=cal,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        use_open=True,
        halt_rows=[
            ("FUT", "2026-01-06T12:00:00Z", "false"),
            ("FUT", "2026-01-07T00:00:00Z", "true"),
        ],
        recorded_queries=queries,
    )
    # Bar 0 enters 400 lots; Bar 1 reduces by 1 lot at open (sell @ 00:00Z, allowed & cached under open),
    # then close=80 triggers Step 4.7 liquidation (sell @ 16:00Z, MUST query close & be blocked!).
    target_weights = pd.DataFrame({"FUT": [1.0, 0.9975, np.nan, np.nan]}, index=dates)
    cfg = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=0.20,
        maintenance_margin_rate=0.10,
        leverage=4.0,
        liquidation_policy="liquidate",
    )
    pf, diag = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=pd.DataFrame(close_p, index=dates),
        open_aligned=pd.DataFrame(open_p, index=dates),
        use_open=True,
        slot=slot,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg,
        target_value_mode="target_weight",
    )

    bar1_sells = [q for q in queries if q["session_id"] == "2026-01-06" and q["side"] == "sell"]
    assert [(q["execution_price_field"], q["timestamp_utc"], q["allowed"]) for q in bar1_sells] == [
        ("open", pd.Timestamp("2026-01-06T00:00:00Z"), True),
        ("close", pd.Timestamp("2026-01-06T16:00:00Z"), False),
    ]
    assert pf.lots_history.loc[dates[1], "FUT"] == 399
    assert pf.lots_history.loc[dates[2], "FUT"] == 0
    assert diag["liquidation_blocked_count"] == 1
    assert diag["liquidation_deferred_executed_count"] == 1


@pytest.mark.parametrize("use_public_protocol", [False, True], ids=["SessionCalendar", "CalendarSlotProtocol"])
@pytest.mark.parametrize("frequency", ["1d", "1w", "1h"])
def test_p1_2_validator_aligns_execution_times_with_label_axis_across_frequencies(
    use_public_protocol: bool, frequency: str
) -> None:
    """_FuturesOrderValidator resolves open/close timestamps matching _LabelAxis across 1d/1w/intraday."""
    from backtest_and_analysis.evaluation.labels import _LabelAxis
    from utils.session_calendar import SessionCalendar

    cal = SessionCalendar(
        "SYNTH_MULTI",
        "2026-01-05",
        "2026-01-16",
        custom_sessions={
            "timezone": "UTC",
            "weekdays": [0, 1, 2, 3, 4],
            "segments": [
                {"open": "01:00", "close": "03:00"},
                {"open": "05:00", "close": "07:00"},
            ],
        },
    )

    if use_public_protocol:
        class _PublicCalendar:
            calendar_hash = cal.calendar_hash
            sessions = staticmethod(cal.sessions)
            session_at = staticmethod(cal.session_at)
            bar_grid = staticmethod(cal.bar_grid)

        active_cal: Any = _PublicCalendar()
    else:
        active_cal = cal

    freq_obj = Frequency.parse(frequency)
    grid = active_cal.bar_grid(
        freq_obj,
        "2026-01-05",
        "2026-01-16" if frequency != "1h" else "2026-01-05",
        view="ETH",
    )
    axis = _LabelAxis.from_slot_grid(active_cal, grid, freq_obj, "ETH")
    bar_utc_index = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
    session_ids = {
        (idx, "FUT"): str(sid) for idx, sid in enumerate(grid["session_id"].tolist())
    }

    seen: List[Tuple[pd.Timestamp, Optional[str]]] = []

    class _SpyMicro:
        execution_profile = ExecutionProfile(
            entry_price_field="open",
            exit_price_field="close",
            min_entry_offset_bars=1,
            exit_adjustment=0,
            return_unit="decimal",
            gap_rule="endpoints_only",
            boundary_policy="cross_session",
            allow_short=True,
        )

        def tradability_mask(self, query_grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
            del instrument_id
            seen.append(
                (
                    pd.Timestamp(query_grid["timestamp_utc"].iloc[0]),
                    query_grid["execution_price_field"].iloc[0],
                )
            )
            return pd.Series(True, index=query_grid.index, dtype=bool)

    class _NoopFeed:
        def read_signal_bars(self, *a, **kw):
            return pd.DataFrame()

        def read_execution_bars(self, *a, **kw):
            return pd.DataFrame()

    class _NoopUniverse:
        def instrument_spec(self, instrument_id: str) -> InstrumentSpec:
            return InstrumentSpec(instrument_id, "futures", "USD")

        def resolve_universe(self, universe_id: str, *, as_of_utc):
            return ("FUT",)

        def resolve_active_contract(self, instrument_id: str, *, as_of_utc) -> str:
            return instrument_id

    class _NoopAux:
        def read_aux_features(self, g, *, features):
            return pd.DataFrame()

    slot = MarketSlot(
        slot_id="global_futures",
        calendar=active_cal,
        bar_feed=_NoopFeed(),
        universe=_NoopUniverse(),
        aux_features=_NoopAux(),
        microstructure=_SpyMicro(),
    )
    validator = fut_exec_mod._FuturesOrderValidator(
        slot=slot,
        bar_utc_index=bar_utc_index,
        session_ids=session_ids,
        frequency=freq_obj,
        default_execution_price_field="open",
    )

    for field in ("open", "close"):
        seen.clear()
        expected_times = axis.execution_times(field)
        for bar_idx in range(len(bar_utc_index)):
            assert validator.is_mask_tradable(
                bar_idx=bar_idx,
                inst="FUT",
                contract="FUT",
                side="buy",
                price=100.0,
                execution_price_field=field,
            )
        assert [ts for ts, _ in seen] == list(expected_times)
        assert all(f == field for _, f in seen)


def _make_contract_priced_mock_slot(
    *,
    dates: pd.DatetimeIndex,
    instruments: List[str],
    open_panel: Dict[str, List[float]],
    close_panel: Dict[str, List[float]],
    contract_open_panel: Dict[str, List[float]],
    contract_close_panel: Dict[str, List[float]],
    contract_map: Dict[Tuple[int, str], str],
    tradability_fn: Optional[Callable[[pd.DataFrame, str], pd.Series]] = None,
    specs: Optional[Dict[str, InstrumentSpec]] = None,
    allow_short: bool = False,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> MarketSlot:
    """Wrap mock slot so querying a concrete contract returns its own prices (distinct from new active)."""
    base_slot = _make_mock_futures_slot(
        dates=dates,
        instruments=instruments,
        open_panel=open_panel,
        close_panel=close_panel,
        contract_map=contract_map,
        specs=specs,
        tradability_fn=tradability_fn,
        allow_short=allow_short,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    utc_dates = pd.DatetimeIndex(dates).tz_localize("UTC") if dates.tz is None else dates.tz_convert("UTC")

    class _ContractPricedBarFeed:
        def read_signal_bars(self, req_insts, frequency, *, start_utc, end_utc, as_of_utc) -> pd.DataFrame:
            return base_slot.bar_feed.read_signal_bars(
                req_insts, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
            )

        def read_execution_bars(self, req_insts, frequency, *, start_utc, end_utc) -> pd.DataFrame:
            rows = []
            start_ts = pd.Timestamp(start_utc)
            end_ts = pd.Timestamp(end_utc)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            for pos, (d, utc_ts) in enumerate(zip(dates, utc_dates)):
                if utc_ts < start_ts or utc_ts > end_ts:
                    continue
                for inst in req_insts:
                    if inst in open_panel:
                        o_px = open_panel[inst][pos]
                        c_px = close_panel[inst][pos]
                        c_id = contract_map.get((pos, inst), inst)
                    elif inst in contract_open_panel:
                        o_px = contract_open_panel[inst][pos]
                        c_px = contract_close_panel[inst][pos]
                        c_id = inst
                    else:
                        continue
                    rows.append(
                        {
                            "timestamp_utc": utc_ts,
                            "session_id": pd.Timestamp(d).strftime("%Y-%m-%d"),
                            "instrument_id": inst,
                            "concrete_instrument_id": c_id,
                            "open": o_px,
                            "close": c_px,
                            "volume": 1_000_000.0,
                            "money": 1_000_000.0 * c_px,
                        }
                    )
            return pd.DataFrame(rows)

    return replace(base_slot, bar_feed=_ContractPricedBarFeed())


@pytest.mark.parametrize("target_weight_basis", ["notional", "margin"])
def test_p1_3_deferred_roll_old_contract_valuation_preserves_weights_and_avoids_false_sell(
    monkeypatch: pytest.MonkeyPatch,
    target_weight_basis: str,
) -> None:
    """[一-3] Deferred roll with price spread (old=100, new=200) must value held old contract at 100 and not false-sell B."""
    dates = pd.bdate_range("2026-01-05", periods=5)
    # A rolls from A_C1 (100) to A_C2 (200) on bar 2, while A_C1 stays at 100 and is blocked from selling on bar 2.
    open_p = {
        "A": [100.0, 100.0, 200.0, 200.0, 200.0],
        "B": [100.0, 100.0, 100.0, 100.0, 100.0],
    }
    close_p = {
        "A": [100.0, 100.0, 200.0, 200.0, 200.0],
        "B": [100.0, 100.0, 100.0, 100.0, 100.0],
    }
    contract_open_p = {
        "A_C1": [100.0, 100.0, 100.0, 100.0, 100.0],
        "A_C2": [200.0, 200.0, 200.0, 200.0, 200.0],
        "B_C1": [100.0, 100.0, 100.0, 100.0, 100.0],
    }
    contract_close_p = {
        "A_C1": [100.0, 100.0, 100.0, 100.0, 100.0],
        "A_C2": [200.0, 200.0, 200.0, 200.0, 200.0],
        "B_C1": [100.0, 100.0, 100.0, 100.0, 100.0],
    }
    contract_map = {
        (0, "A"): "A_C1",
        (1, "A"): "A_C1",
        (2, "A"): "A_C2",
        (3, "A"): "A_C2",
        (4, "A"): "A_C2",
        (0, "B"): "B_C1",
        (1, "B"): "B_C1",
        (2, "B"): "B_C1",
        (3, "B"): "B_C1",
        (4, "B"): "B_C1",
    }
    blocked_day = dates[2].strftime("%Y-%m-%d")

    def _mask_fn(grid: pd.DataFrame, instrument_id: str) -> pd.Series:
        if instrument_id == "A_C1":
            return grid["session_id"].astype(str) != blocked_day
        return pd.Series(True, index=grid.index, dtype=bool)

    slot = _make_contract_priced_mock_slot(
        dates=dates,
        instruments=["A", "B"],
        open_panel=open_p,
        close_panel=close_p,
        contract_open_panel=contract_open_p,
        contract_close_panel=contract_close_p,
        contract_map=contract_map,
        tradability_fn=_mask_fn,
        allow_short=False,
    )

    observed_prev_weights: List[Dict[str, float]] = []
    orig_blend = VectorBTBacktestRunner._blend_partial_rebalance

    def _spy_blend(self, target_map: Dict[str, float], prev_weight_map: Dict[str, float]) -> Dict[str, float]:
        observed_prev_weights.append(dict(prev_weight_map))
        return orig_blend(self, target_map, prev_weight_map)

    monkeypatch.setattr(VectorBTBacktestRunner, "_blend_partial_rebalance", _spy_blend)

    margin_rate = 1.0 if target_weight_basis == "notional" else 0.2
    expected_lots = 50 if target_weight_basis == "notional" else 250
    scores = pd.DataFrame({"A": [2.0] * 5, "B": [1.0] * 5}, index=dates)
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": margin_rate,
                "target_weight_basis": target_weight_basis,
                "allow_short": False,
                "partial_rebalance": {"enabled": True, "trade_rate": 0.5, "min_weight": 0.01},
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates),
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res

    pf = res["portfolio"]
    diag = res["futures_execution"]
    # Bar 1 enters equal weight; Bar 2 sees deferred roll on A_C1 (100, NOT A_C2=200).
    assert len(observed_prev_weights) == 2
    assert observed_prev_weights[1] == pytest.approx({"A": 0.5, "B": 0.5})
    assert pf.value().loc[dates[2]] == pytest.approx(100_000.0)
    # Asset B must NOT be falsely sold on Bar 2 or subsequent bars.
    assert pf.lots_history.loc[dates[1], "B"] == expected_lots
    assert pf.lots_history.loc[dates[2], "B"] == expected_lots
    assert pf.lots_history.loc[dates[3], "B"] == expected_lots
    assert pf.lots_history.loc[dates[4], "B"] == expected_lots
    orders = pf.orders.records_readable
    assert not ((orders["Column"] == "B") & (orders["Side"] == "Sell")).any()
    # A stays on A_C1 on Bar 2 and rolls to A_C2 on Bar 3.
    assert pf.lots_history.loc[dates[2], "A"] == expected_lots
    assert diag["roll_deferred_count"] == 1
    assert diag["roll_count"] == 1


def test_p1_3_deferred_roll_old_contract_consistency_across_account_view_and_blocked_exit_log() -> None:
    """[一-3] Verify _FuturesAccountView, branch3 free budget, and blocked-exit logs all use old contract price."""
    dates = pd.bdate_range("2026-01-05", periods=4)
    # A_C1=100 on bar 1..2; on bar 2 A rolls to A_C2=200 while A_C1 is blocked at 100.
    # On bar 2, A and B are both dropped from top_k (C and D are selected):
    # B sells cleanly (frees 50k), A becomes a blocked exit at 100 (occupies 50k, NOT 100k),
    # so C and D split the remaining 50k budget (25 lots each, rather than 0 lots!).
    open_p = {
        "A": [100.0, 100.0, 200.0, 200.0],
        "B": [100.0, 100.0, 100.0, 100.0],
        "C": [100.0, 100.0, 100.0, 100.0],
        "D": [100.0, 100.0, 100.0, 100.0],
    }
    close_p = dict(open_p)
    contract_open_p = {
        "A_C1": [100.0, 100.0, 100.0, 100.0],
        "A_C2": [200.0, 200.0, 200.0, 200.0],
        "B_C1": [100.0, 100.0, 100.0, 100.0],
        "C_C1": [100.0, 100.0, 100.0, 100.0],
        "D_C1": [100.0, 100.0, 100.0, 100.0],
    }
    contract_close_p = dict(contract_open_p)
    contract_map = {
        (0, "A"): "A_C1",
        (1, "A"): "A_C1",
        (2, "A"): "A_C2",
        (3, "A"): "A_C2",
        (0, "B"): "B_C1",
        (1, "B"): "B_C1",
        (2, "B"): "B_C1",
        (3, "B"): "B_C1",
        (0, "C"): "C_C1",
        (1, "C"): "C_C1",
        (2, "C"): "C_C1",
        (3, "C"): "C_C1",
        (0, "D"): "D_C1",
        (1, "D"): "D_C1",
        (2, "D"): "D_C1",
        (3, "D"): "D_C1",
    }
    blocked_day = dates[2].strftime("%Y-%m-%d")

    def _mask_fn(grid: pd.DataFrame, instrument_id: str) -> pd.Series:
        if instrument_id == "A_C1":
            return grid["session_id"].astype(str) != blocked_day
        return pd.Series(True, index=grid.index, dtype=bool)

    slot = _make_contract_priced_mock_slot(
        dates=dates,
        instruments=["A", "B", "C", "D"],
        open_panel=open_p,
        close_panel=close_p,
        contract_open_panel=contract_open_p,
        contract_close_panel=contract_close_p,
        contract_map=contract_map,
        tradability_fn=_mask_fn,
        allow_short=False,
    )
    # Bar 1 (reads bar 0): A=4, B=3, C=2, D=1 -> top_k=2 selects A & B (50 lots each @ 100).
    # Bar 2 (reads bar 1): C=4, D=3, A=2, B=1 -> top_k=2 selects C & D (drops A & B!).
    scores = pd.DataFrame(
        {
            "A": [4.0, 2.0, 2.0, 2.0],
            "B": [3.0, 1.0, 1.0, 1.0],
            "C": [2.0, 4.0, 4.0, 4.0],
            "D": [1.0, 3.0, 3.0, 3.0],
        },
        index=dates,
    )
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "margin_rate": 1.0,
                "allow_short": False,
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates),
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res

    pf = res["portfolio"]
    # Bar 1: A=50, B=50, C=0, D=0.
    assert pf.lots_history.loc[dates[1], "A"] == 50
    assert pf.lots_history.loc[dates[1], "B"] == 50
    # Bar 2: B is sold (frees 50k), A is roll-blocked at 100 (occupies 50k, NOT 100k!),
    # so C and D must each enter at 25 lots (25k) rather than 0 lots!
    assert pf.lots_history.loc[dates[2], "A"] == 50
    assert pf.lots_history.loc[dates[2], "B"] == 0
    assert pf.lots_history.loc[dates[2], "C"] == 25
    assert pf.lots_history.loc[dates[2], "D"] == 25
    # Bar 3: A_C1 unblocks and finishes pending exit at 100.
    assert pf.lots_history.loc[dates[3], "A"] == 0
    # Blocked REBALANCE log entry for A_C1 must record Val Price=100.0 (not new contract 200.0).
    logs = pf.logs.records_readable
    reb_rejected = logs.loc[
        (logs["Column"] == "A")
        & (logs["Result Status"] == "Rejected")
        & (logs["Reason"] == "REBALANCE:tradability_mask_blocked")
    ]
    assert not reb_rejected.empty
    assert float(reb_rejected.iloc[0]["Val Price"]) == pytest.approx(100.0)
    assert res["execution_diagnostics"]["blocked_exit_notional"] == pytest.approx(50_000.0)


@pytest.mark.parametrize("lot_rounding", ["floor", "nearest"])
def test_margin_liquidity_cap_uses_each_instruments_rate(
    lot_rounding: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    dates = pd.bdate_range("2026-01-05", periods=4)
    prices = {inst: [100.0] * len(dates) for inst in ("A", "B")}
    price_df = pd.DataFrame(prices, index=dates)
    use_contract_override = lot_rounding == "nearest"
    cap = 100_600.0 if use_contract_override else 100_000.0
    margin_rates = {"A": 0.1, "B": 0.2}
    contract_map = None
    if use_contract_override:
        margin_rates.update({"A": 0.4, "A_C1": 0.1})
        contract_map = {(i, "A"): "A_C1" for i in range(len(dates))}
    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=list(prices),
        open_panel=prices,
        close_panel=prices,
        volume_panel={inst: [cap / 10.0] * len(dates) for inst in prices},
        contract_map=contract_map,
        allow_short=use_contract_override,
    )
    scores = pd.DataFrame({"A": 2.0, "B": 1.0}, index=dates)
    observed = []
    original_evaluate = SharedPortfolioPolicy.evaluate_bar

    def capture_decision(self, **kwargs):
        decision = original_evaluate(self, **kwargs)
        if decision.kind == "branch_overlay":
            observed.append((decision.target_weights, decision.target_values))
        return decision

    monkeypatch.setattr(SharedPortfolioPolicy, "evaluate_bar", capture_decision)
    lots = {}
    for default_rate in (0.2, 0.1):
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 2,
                    "score_source": "input_factor",
                    "holding_period": 1,
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "target_weight_basis": "margin",
                    "margin_rate": default_rate,
                    "margin_rates": margin_rates,
                    "allow_short": use_contract_override,
                    "lot_rounding": lot_rounding,
                    "liquidity_cap": {
                        "enabled": True,
                        "adv_lookback_days": 2,
                        "max_participation": 0.1,
                        "capital_base": 100_000.0 if use_contract_override else None,
                    },
                }
            },
            slot=slot,
            frequency="1d",
        )
        try:
            res = runner.run_backtest(
                scores,
                price_df,
                open_df=price_df,
                rebalance_dates=[dates[2].to_pydatetime()],
            )
        finally:
            runner.close()
        assert "error" not in res, res
        lots[default_rate] = res["portfolio"].lots_history.loc[dates[2]].to_dict()

    notionals = {
        rate: {inst: abs(count) * 100.0 * 10.0 for inst, count in by_inst.items()}
        for rate, by_inst in lots.items()
    }
    assert all(value <= cap for row in notionals.values() for value in row.values()), (lots, notionals)
    sign_b = -1 if use_contract_override else 1
    assert lots[0.2] == lots[0.1] == {"A": 100, "B": sign_b * 100}
    assert len(observed) == 2
    assert observed[0] == observed[1]
    weights, targets = observed[0]
    assert weights == pytest.approx({"A": cap * 0.1 / 100_000.0, "B": sign_b * cap * 0.2 / 100_000.0})
    assert targets == pytest.approx({"A": cap, "B": sign_b * cap})


def test_margin_liquidity_reallocation_charges_held_margin_for_blocked_exit() -> None:
    dates = pd.bdate_range("2026-01-05", periods=5)
    prices = {inst: [100.0] * len(dates) for inst in ("A", "B")}
    price_df = pd.DataFrame(prices, index=dates)
    blocked_date = dates[3].strftime("%Y-%m-%d")

    def tradable(grid: pd.DataFrame, instrument_id: str) -> pd.Series:
        if instrument_id == "A":
            return grid["session_id"].astype(str) != blocked_date
        return pd.Series(True, index=grid.index)

    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=list(prices),
        open_panel=prices,
        close_panel=prices,
        volume_panel={inst: [10_000.0] * len(dates) for inst in prices},
        tradability_fn=tradable,
        allow_short=False,
    )
    scores = pd.DataFrame({"A": [2, 2, 1, 1, 1], "B": [1, 1, 2, 2, 2]}, index=dates)
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "target_weight_basis": "margin",
                "margin_rate": 0.2,
                "margin_rates": {"A": 0.1, "B": 0.2},
                "allow_short": False,
                "liquidity_cap": {"enabled": True, "adv_lookback_days": 2, "max_participation": 0.1},
            }
        },
        slot=slot,
        frequency="1d",
    )
    try:
        res = runner.run_backtest(
            scores,
            price_df,
            open_df=price_df,
            rebalance_dates=[dates[2].to_pydatetime(), dates[3].to_pydatetime()],
        )
    finally:
        runner.close()
    assert "error" not in res, res
    lots = res["portfolio"].lots_history
    assert lots.loc[dates[2]].to_dict() == {"A": 100, "B": 0}
    # B's 20,000 margin budget has 10,000 left after A's blocked exit.
    assert lots.loc[dates[3]].to_dict() == {"A": 100, "B": 50}


def test_p1_5_branch_topk_margin_basis_respects_per_instrument_margin_rates() -> None:
    """[一-5] Standard Top-K under target_weight_basis='margin' allocates equal margin budgets using margin_rates."""
    dates = pd.bdate_range("2026-01-05", periods=4)
    open_p = {"A": [100.0] * 4, "B": [100.0] * 4}
    close_p = {"A": [100.0] * 4, "B": [100.0] * 4}
    specs = {
        "A": InstrumentSpec("A", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        "B": InstrumentSpec("B", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
    }
    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["A", "B"],
        open_panel=open_p,
        close_panel=close_p,
        specs=specs,
        allow_short=False,
    )
    scores = pd.DataFrame({"A": [2.0] * 4, "B": [1.0] * 4}, index=dates)
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "target_weight_basis": "margin",
                "margin_rate": 0.2,
                "margin_rates": {"A": 0.1, "B": 0.2},
                "allow_short": False,
                "cost_rate": 0.0,
                "buy_cost_rate": 0.0,
                "sell_cost_rate": 0.0,
                "slippage": 0.0,
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates),
        rebalance_dates=[dates[1].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res

    pf = res["portfolio"]
    # Bar 1: each gets 50,000 margin budget -> A: 50,000/(0.1*100*10)=500 lots; B: 50,000/(0.2*100*10)=250 lots.
    assert pf.lots_history.loc[dates[1], "A"] == 500
    assert pf.lots_history.loc[dates[1], "B"] == 250
    assert pf.margin_used.loc[dates[1]] == pytest.approx(100_000.0)
    assert pf.margin_utilization.loc[dates[1]] == pytest.approx(1.0)


def test_p1_5_branch_topk_margin_basis_rotation_exposure_leverage_and_close_costs() -> None:
    """[一-5] Rotation in standard Top-K margin basis handles held pos.margin_rate, close costs, leverage, and exposure."""
    dates = pd.bdate_range("2026-01-05", periods=4)
    open_p = {"A": [100.0] * 4, "B": [100.0] * 4, "C": [100.0] * 4}
    close_p = dict(open_p)
    specs = {
        inst: InstrumentSpec(inst, "futures", "CNY", lot_size=1, contract_multiplier=10.0)
        for inst in ("A", "B", "C")
    }
    slot = _make_mock_futures_slot(
        dates=dates,
        instruments=["A", "B", "C"],
        open_panel=open_p,
        close_panel=close_p,
        specs=specs,
        allow_short=False,
        commission_bps=10.0,
        slippage_bps=0.0,
    )
    # Bar 1 (reads bar 0): A=3, B=2, C=1 -> buy A (mr=0.1) & B (mr=0.2).
    # Bar 2 (reads bar 1): A=3, C=2, B=1 -> sell B, keep A, buy C (mr=0.25).
    scores = pd.DataFrame(
        {
            "A": [3.0, 3.0, 3.0, 3.0],
            "B": [2.0, 1.0, 1.0, 1.0],
            "C": [1.0, 2.0, 2.0, 2.0],
        },
        index=dates,
    )
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": 100_000.0,
                "execution_mode": "futures_contract",
                "target_weight_basis": "margin",
                "margin_rate": 0.2,
                "margin_rates": {"A": 0.1, "B": 0.2, "C": 0.25},
                "leverage": 0.8,
                "allow_short": False,
                "cost_rate": 0.001,
                "buy_cost_rate": 0.001,
                "sell_cost_rate": 0.001,
                "slippage": 0.0,
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates),
        rebalance_dates=[dates[1].to_pydatetime(), dates[2].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res

    pf = res["portfolio"]
    # Bar 1 (empty book): margin budget per slot = 100,000 * 0.8 / 2 = 40,000.
    # A (mr=0.1): 400 lots (notional 400,000, entry cost 400).
    # B (mr=0.2): 200 lots (notional 200,000, entry cost 200).
    # End of Bar 1 equity = 99,400.
    assert pf.lots_history.loc[dates[1], "A"] == 400
    assert pf.lots_history.loc[dates[1], "B"] == 200
    assert pf.value().loc[dates[1]] == pytest.approx(99_400.0)

    # Bar 2 (rotation): B close cost = 200,000 * 0.001 = 200 -> post_sell_equity = 99,200.
    # A used_margin = 400 * 100 * 10 * 0.1 = 40,000.
    # free_margin = 99,200 * 1.0 * 0.8 - 40,000 = 39,360.
    # C (mr=0.25): target_notional = 39,360 / 0.25 = 157,440 -> floor(157.44) = 157 lots!
    assert pf.lots_history.loc[dates[2], "A"] == 400
    assert pf.lots_history.loc[dates[2], "B"] == 0
    assert pf.lots_history.loc[dates[2], "C"] == 157
    assert pf.margin_used.loc[dates[2]] == pytest.approx(40_000.0 + 157 * 100.0 * 10.0 * 0.25)

    # Also verify direct _FuturesAccountView.branch3_buy_target_value exposure scaling on empty & held states.
    cfg = fut_exec_mod.FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=0.2,
        margin_rates={"A": 0.1, "A_C2": 0.3, "C": 0.25},
        leverage=0.8,
        target_weight_basis="margin",
    )
    validator = fut_exec_mod._FuturesOrderValidator(
        slot=None,
        bar_utc_index=pd.DatetimeIndex(dates, tz="UTC"),
        session_ids={},
        frequency=Frequency.parse("1d"),
        default_execution_price_field="open",
    )
    empty_view = fut_exec_mod._FuturesAccountView(
        bar_idx=0,
        equity=100_000.0,
        positions={},
        pending_rolls={},
        pending_liquidations=set(),
        active_contract={(0, "A"): "A_C1", (0, "C"): "C_C1"},
        missing_bar_insts=set(),
        exec_price_fn=lambda _b, _i: 100.0,
        last_contract_close={},
        validator=validator,
        config=cfg,
        get_spec_fn=lambda c: InstrumentSpec(c, "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        buy_cost_rate=0.001,
        sell_cost_rate=0.001,
        slippage=0.0,
    )
    assert empty_view.branch3_buy_target_value(
        executed_sells=set(),
        remaining_held=set(),
        effective_top_k=2,
        buy_count=2,
        exposure=0.5,
    ) == pytest.approx(100_000.0 * 0.5 * 0.8 / 2.0)

    held_view = fut_exec_mod._FuturesAccountView(
        bar_idx=1,
        equity=100_000.0,
        positions={
            "A": fut_exec_mod._OpenPosition(
                instrument_id="A",
                concrete_instrument_id="A_C1",
                lots=100,
                multiplier=10.0,
                margin_rate=0.1,
                avg_entry_price=100.0,
                last_mark_price=100.0,
            ),
            "B": fut_exec_mod._OpenPosition(
                instrument_id="B",
                concrete_instrument_id="B_C1",
                lots=100,
                multiplier=10.0,
                margin_rate=0.2,
                avg_entry_price=100.0,
                last_mark_price=100.0,
            ),
        },
        pending_rolls={},
        pending_liquidations=set(),
        active_contract={(1, "A"): "A_C2", (1, "B"): "B_C1", (1, "C"): "C_C1"},
        missing_bar_insts=set(),
        exec_price_fn=lambda _b, _i: 100.0,
        last_contract_close={},
        validator=validator,
        config=cfg,
        get_spec_fn=lambda c: InstrumentSpec(c, "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        buy_cost_rate=0.001,
        sell_cost_rate=0.001,
        slippage=0.0,
    )
    assert held_view.target_margin_rate("A") == 0.3
    assert held_view.target_margin_rate("B") == 0.2
    assert held_view.target_margin_rate("C") == 0.25
    assert held_view.held_budget_value("A") == pytest.approx(10_000.0)
    # B close cost = 100 * 100 * 10 * 0.001 = 100 -> post_sell_equity = 99,900.
    # A used_margin = 100 * 100 * 10 * 0.1 = 10,000.
    # free_margin(exposure=0.5, leverage=0.8) = 99,900 * 0.5 * 0.8 - 10,000 = 29,960.
    assert held_view.branch3_buy_target_value(
        executed_sells={"B"},
        remaining_held={"A"},
        effective_top_k=2,
        buy_count=1,
        exposure=0.5,
    ) == pytest.approx(29_960.0)
