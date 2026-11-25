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
