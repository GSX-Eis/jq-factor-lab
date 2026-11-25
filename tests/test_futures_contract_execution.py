"""Tests for nominal-mode continuous futures protection and contract-level futures execution."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from data_pipeline.loaders.market_bar_importer import import_market_bars
from factor_synthesis.vectorbt_adapter import (
    ContinuousFuturesNominalModeError,
    FuturesExecutionConfig,
    FuturesExecutionError,
    FuturesMarginError,
    VectorBTBacktestRunner,
)
from factor_synthesis.vectorbt_adapter.futures_execution import (
    parse_futures_execution_config,
    round_contract_lots,
    simulate_futures_contract_portfolio,
)
from market_db_samples import declare_futures_bars
from slots.cn_futures import build_cn_futures_slot
from utils.bars.futures_roll import build_roll_schedule
from utils.session_calendar import SessionCalendar
from utils.slots.contracts import ExecutionProfile, InstrumentSpec, MarketSlot
from utils.time_axis import BarSchedule, Frequency


def _build_rolling_futures_db(
    db_path: Path,
    *,
    root: str = "CU",
    exchange: str = "SHFE",
    point_value: float = 10.0,
    jump_pct: float = 0.10,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    c1_opens: list[float] | None = None,
    c1_closes: list[float] | None = None,
    c2_opens: list[float] | None = None,
    c2_closes: list[float] | None = None,
):
    """Create a synthetic futures DuckDB with deterministic roll from C1 to C2."""
    from utils.time_axis.futures import futures_calendar

    start, end = "2026-01-05", "2026-01-13"
    fcal = futures_calendar(root)
    cal = SessionCalendar(
        fcal["calendar_id"],
        start,
        end,
        rth_reference_calendar=fcal["rth_reference_calendar"],
    )
    sessions = cal.sessions()
    n = len(sessions)
    assert n == 7, f"Expected 7 sessions in {start}..{end}, got {n}"

    schedule = BarSchedule(cal, "1d", bucket_strategy="segment_anchored")
    slot_times = pd.concat(
        [schedule.for_session(day) for day in sessions], ignore_index=True
    )["start_utc"]

    c1 = f"{root}2603.{exchange}"
    c2 = f"{root}2604.{exchange}"
    contracts = [c1, c2]

    if c1_opens is None:
        c1_opens = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0]
    if c1_closes is None:
        c1_closes = [101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0]
    if c2_opens is None:
        c2_opens = [round(p * (1.0 + jump_pct), 6) for p in c1_opens]
    if c2_closes is None:
        c2_closes = [round(p * (1.0 + jump_pct), 6) for p in c1_closes]

    meta = pd.DataFrame(
        [
            dict(
                instrument_id=c,
                product_root=root,
                asset_class="futures",
                exchange=exchange,
                timezone=cal.timezone,
                contract_expiry="2026-03-16" if i == 0 else "2026-04-16",
                contract_month="2026-03" if i == 0 else "2026-04",
                price_adjustment="none",
                point_value=float(point_value),
                rth_reference_calendar=fcal["rth_reference_calendar"],
            )
            for i, c in enumerate(contracts)
        ]
    )

    # Session 0..2: C1 dominates OI (decision on session 2 close keeps C1 on session 3?
    # Wait: decision on session 2 (2026-01-07) -> effective session 3 (2026-01-08)!
    # Let's make C1 dominate sessions 0..2 and C2 dominate sessions 3..6 so roll effective on session 4 (2026-01-09).
    oi_c1 = [500.0, 500.0, 500.0, 100.0, 100.0, 100.0, 100.0]
    oi_c2 = [100.0, 100.0, 100.0, 500.0, 500.0, 500.0, 500.0]

    bars_list = []
    for idx, (c, opens, closes, ois) in enumerate(
        [(c1, c1_opens, c1_closes, oi_c1), (c2, c2_opens, c2_closes, oi_c2)]
    ):
        opens_arr = np.asarray(opens, dtype=float)
        closes_arr = np.asarray(closes, dtype=float)
        bars_list.append(
            pd.DataFrame(
                {
                    "timestamp_utc": slot_times,
                    "instrument_id": c,
                    "frequency": "1d",
                    "open": opens_arr,
                    "high": np.maximum(opens_arr, closes_arr) + 2.0,
                    "low": np.minimum(opens_arr, closes_arr) - 2.0,
                    "close": closes_arr,
                    "open_interest": ois,
                    "volume": 500.0,
                    "source": "SYNTH",
                    "data_version": "v1",
                }
            )
        )

    roll_start = sessions[1].strftime("%Y-%m-%d")
    conn = duckdb.connect(str(db_path))
    try:
        declare_futures_bars(conn, exchange)
        import_market_bars(
            conn,
            pd.concat(bars_list, ignore_index=True),
            meta,
            bucket_strategy="segment_anchored",
            legacy_compatibility=False,
        )
        build_roll_schedule(conn, product_roots=[root], start=roll_start, end=end)
    finally:
        conn.close()

    return roll_start, end, c1, c2


def _read_continuous_wide(slot: MarketSlot, root: str, score_value: float = 1.0):
    grid = slot.calendar.bar_grid("1d")
    exec_bars = slot.bar_feed.read_execution_bars(
        [root],
        Frequency.parse("1d"),
        start_utc=grid["timestamp_utc"].min(),
        end_utc=grid["timestamp_utc"].max(),
    )
    exec_bars = exec_bars.sort_values("timestamp_utc")
    idx = pd.DatetimeIndex(pd.to_datetime(exec_bars["session_id"]))
    close_wide = pd.DataFrame({root: exec_bars["close"].to_numpy(dtype=float)}, index=idx)
    open_wide = pd.DataFrame({root: exec_bars["open"].to_numpy(dtype=float)}, index=idx)
    factor_wide = pd.DataFrame(score_value, index=idx, columns=[root])
    return exec_bars, close_wide, open_wide, factor_wide


def test_nominal_weight_mode_rejects_rolling_continuous_futures(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_roll_guard.duckdb"
    roll_start, end, c1, c2 = _build_rolling_futures_db(db_path, jump_pct=0.10)

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        exec_bars, close_wide, open_wide, factor_wide = _read_continuous_wide(slot, "CU")
        assert exec_bars["concrete_instrument_id"].nunique() == 2

        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 1,
                    "rebalance_freq": "daily",
                }
            },
            slot=slot,
            frequency="1d",
        )
        assert runner.execution_mode == "nominal_weight"
        with pytest.raises(ContinuousFuturesNominalModeError, match=r"nominal_weight.*CU"):
            runner.run_backtest(factor_wide, close_wide, open_df=open_wide)

        # Direct _simulate_portfolio call is also protected.
        target_dummy = pd.DataFrame(100_000.0, index=close_wide.index, columns=["CU"])
        with pytest.raises(ContinuousFuturesNominalModeError):
            runner._simulate_portfolio(target_dummy, close_wide, open_wide, True)
        runner.close()
    finally:
        conn.close()


def test_nominal_weight_mode_allows_concrete_non_rolling_contracts(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_concrete.duckdb"
    roll_start, end, c1, c2 = _build_rolling_futures_db(db_path, jump_pct=0.10)
    contracts = [c1, c2]

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        grid = slot.calendar.bar_grid("1d")
        bars = slot.bar_feed.read_execution_bars(
            contracts,
            Frequency.parse("1d"),
            start_utc=grid["timestamp_utc"].min(),
            end_utc=grid["timestamp_utc"].max(),
        )
        bars["date"] = pd.to_datetime(bars["session_id"])
        close_wide = bars.pivot(index="date", columns="instrument_id", values="close").sort_index()
        open_wide = bars.pivot(index="date", columns="instrument_id", values="open").sort_index()
        factor_wide = pd.DataFrame(
            np.tile([1.0, -1.0], (len(close_wide), 1)),
            index=close_wide.index,
            columns=contracts,
        )
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 1,
                    "rebalance_freq": "daily",
                }
            },
            slot=slot,
            frequency="1d",
        )
        res = runner.run_backtest(factor_wide, close_wide, open_df=open_wide)
        runner.close()
    finally:
        conn.close()

    assert "error" not in res, res
    assert res["trade_count"] > 0


def test_nominal_weight_mode_rejects_mock_slot_resolve_active_contract_switch(tmp_path: Path) -> None:
    """Even if read_execution_bars omits concrete_instrument_id, resolve_active_contract switching fails fast."""
    db_path = tmp_path / "futures_mock_resolve.duckdb"
    roll_start, end, c1, c2 = _build_rolling_futures_db(db_path)

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        base_slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(base_slot, "CU")

        class _NoConcreteColFeed:
            def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
                return base_slot.bar_feed.read_signal_bars(
                    instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
                )

            def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc):
                df = base_slot.bar_feed.read_execution_bars(
                    instruments, frequency, start_utc=start_utc, end_utc=end_utc
                )
                return df.drop(columns=["concrete_instrument_id"])

        wrapped = replace(base_slot, bar_feed=_NoConcreteColFeed())
        runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 1, "score_source": "input_factor", "holding_period": 1}},
            slot=wrapped,
            frequency="1d",
        )
        with pytest.raises(ContinuousFuturesNominalModeError):
            runner.run_backtest(factor_wide, close_wide, open_df=open_wide)
        runner.close()
    finally:
        conn.close()


@pytest.mark.parametrize("direction", ["long", "short"])
@pytest.mark.parametrize("use_open", [True, False])
def test_continuous_futures_roll_manual_pnl_zero_gap_pollution(
    tmp_path: Path, direction: str, use_open: bool
) -> None:
    """Verify exact manual bar-by-bar PnL across a +10% (and +50%) roll jump gap."""
    # Construct two DBs differing ONLY in C2's absolute price level (+10% vs +50% jump,
    # keeping C2's post-roll point return identical or testing exact manual settlement).
    db_path = tmp_path / f"futures_{direction}_{use_open}.duckdb"
    c1_opens = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0]
    c1_closes = [101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0]
    # Huge +10% jump on C2, plus +2/bar move after roll.
    c2_opens = [110.0, 112.2, 114.4, 116.6, 118.8, 121.0, 123.2]
    c2_closes = [111.0, 113.2, 115.4, 117.6, 119.8, 122.0, 124.2]

    roll_start, end, c1, c2 = _build_rolling_futures_db(
        db_path,
        point_value=10.0,
        c1_opens=c1_opens,
        c1_closes=c1_closes,
        c2_opens=c2_opens,
        c2_closes=c2_closes,
    )

    score_val = 1.0 if direction == "long" else -1.0
    sign = 1 if direction == "long" else -1
    init_cash = 100_000.0
    cost_rate = 0.0005  # 5 bps
    slippage = 0.0002   # 2 bps
    fee_mult = cost_rate + slippage
    mult = 10.0

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        # Also create a ratio_back slot to verify signal bars with back-adjustment don't leak!
        slot_exec = build_cn_futures_slot(
            conn,
            product_root="CU",
            start=roll_start,
            end=end,
            commission_bps=5.0,
            slippage_bps=2.0,
        )
        slot_sig = build_cn_futures_slot(
            conn,
            product_root="CU",
            start=roll_start,
            end=end,
            adjustment="ratio_back",
            commission_bps=5.0,
            slippage_bps=2.0,
        )
        grid = slot_sig.calendar.bar_grid("1d")
        sig_bars = slot_sig.bar_feed.read_signal_bars(
            ["CU"],
            Frequency.parse("1d"),
            start_utc=grid["timestamp_utc"].min(),
            end_utc=grid["timestamp_utc"].max(),
            as_of_utc=grid["timestamp_utc"].max(),
        ).sort_values("timestamp_utc")
        idx = pd.DatetimeIndex(pd.to_datetime(sig_bars["session_id"]))
        close_sig = pd.DataFrame({"CU": sig_bars["close"].to_numpy(dtype=float)}, index=idx)
        open_sig = pd.DataFrame({"CU": sig_bars["open"].to_numpy(dtype=float)}, index=idx)
        factor_wide = pd.DataFrame(score_val, index=idx, columns=["CU"])

        if not use_open:
            open_arg = None

            class _CloseMicro:
                def __init__(self, inner):
                    self._inner = inner
                    self.execution_profile = replace(
                        inner.execution_profile,
                        entry_price_field="close",
                        exit_adjustment=0,
                    )

                def tradability_mask(self, g, *, instrument_id):
                    return self._inner.tradability_mask(g, instrument_id=instrument_id)

            slot_exec = replace(slot_exec, microstructure=_CloseMicro(slot_exec.microstructure))
        else:
            open_arg = open_sig

        # Single entry at bar 1 (bar 0 has no T-1 factor), then held across roll on bar 3.
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": init_cash,
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.5,
                }
            },
            slot=slot_exec,
            frequency="1d",
        )
        res = runner.run_backtest(
            factor_wide,
            close_sig,
            open_df=open_arg,
            rebalance_dates=[idx[1].to_pydatetime()],
        )
        runner.close()
    finally:
        conn.close()

    assert "error" not in res, res
    pf = res["portfolio"]
    diag = res["futures_execution"]

    # Sessions 1..6 (roll_start is session 1 = 2026-01-06).
    # In _build_rolling_futures_db:
    # session 1..3: C1, session 4..6: C2 -> roll occurs at bar index 3 (2026-01-09)!
    # Entry occurs at bar index 1 (first bar with a causal T-1 factor value).
    c1_o = c1_opens[1:]
    c1_c = c1_closes[1:]
    c2_o = c2_opens[1:]
    c2_c = c2_closes[1:]
    assert len(c1_o) == 6

    # Manual bar-by-bar settlement:
    entry_idx = 1
    roll_idx = 3
    entry_px_c1 = c1_o[entry_idx] if use_open else c1_c[entry_idx]
    lots_c1 = int(init_cash // (entry_px_c1 * mult)) * sign
    assert abs(lots_c1) > 0

    exit_px_c1 = c1_o[roll_idx] if use_open else c1_c[roll_idx]
    entry_px_c2 = c2_o[roll_idx] if use_open else c2_c[roll_idx]
    # roll_sizing="same_lots" keeps same lots across non-rebalance roll:
    lots_c2 = lots_c1

    open_cost_c1 = abs(lots_c1) * entry_px_c1 * mult * fee_mult
    close_cost_c1 = abs(lots_c1) * exit_px_c1 * mult * fee_mult
    open_cost_c2 = abs(lots_c2) * entry_px_c2 * mult * fee_mult

    expected_equity = []
    for t in range(len(c1_o)):
        if t < entry_idx:
            eq = init_cash
        elif t < roll_idx:
            eq = init_cash - open_cost_c1 + lots_c1 * (c1_c[t] - entry_px_c1) * mult
        else:
            pnl_c1 = lots_c1 * (exit_px_c1 - entry_px_c1) * mult
            pnl_c2 = lots_c2 * (c2_c[t] - entry_px_c2) * mult
            eq = init_cash - open_cost_c1 - close_cost_c1 - open_cost_c2 + pnl_c1 + pnl_c2
        expected_equity.append(eq)

    np.testing.assert_allclose(pf.value().to_numpy(dtype=float), expected_equity, rtol=0.0, atol=1e-8)
    assert diag["roll_count"] == 1
    assert diag["roll_events"][0]["from_contract"] == c1
    assert diag["roll_events"][0]["to_contract"] == c2
    assert diag["roll_cost_total"] == pytest.approx(close_cost_c1 + open_cost_c2, rel=0, abs=1e-8)
    assert diag["roll_events"][0]["price_gap"] == pytest.approx(entry_px_c2 - exit_px_c1, rel=0, abs=1e-8)

    # Verify order audit records contain concrete contracts and roll types.
    orders = pf.orders.records_readable
    assert list(orders["Order Type"]) == ["REBALANCE", "ROLL_CLOSE", "ROLL_OPEN"]
    assert list(orders["Concrete Instrument"]) == [c1, c1, c2]


def test_contract_multiplier_and_lot_size_floor_rounding(tmp_path: Path) -> None:
    """Verify contract_multiplier=300.0 & lot_size floor rounding prevents silent leverage."""
    db_path = tmp_path / "futures_if_lots.duckdb"
    # Price ~ 4000, multiplier = 300 -> 1 lot notional = 1,200,000.
    roll_start, end, c1, c2 = _build_rolling_futures_db(
        db_path,
        root="IF",
        exchange="CFFEX",
        point_value=300.0,
        c1_opens=[4000.0] * 7,
        c1_closes=[4050.0, 4050.0, 4050.0, 4050.0, 4050.0, 4050.0, 4050.0],
        c2_opens=[4100.0] * 7,
        c2_closes=[4150.0] * 7,
    )

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_cn_futures_slot(conn, product_root="IF", start=roll_start, end=end)
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(slot, "IF")

        # Case 1: init_cash = 1,000,000 < 1,200,000 -> floor to 0 lots!
        runner_zero = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": 1_000_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 1.0,
                    "lot_rounding": "floor",
                }
            },
            slot=slot,
            frequency="1d",
        )
        res_zero = runner_zero.run_backtest(
            factor_wide,
            close_wide,
            open_df=open_wide,
            rebalance_dates=[close_wide.index[1].to_pydatetime()],
        )
        runner_zero.close()
        assert "error" not in res_zero, res_zero
        np.testing.assert_allclose(res_zero["portfolio"].value().to_numpy(), 1_000_000.0)
        assert res_zero["futures_execution"]["max_abs_lots"] == 0
        assert res_zero["futures_execution"]["lot_floored_to_zero_events"] >= 1

        # Case 2: init_cash = 3,000,000 -> 3,000,000 / 1,200,000 = 2.5 -> floor to 2 lots!
        runner_two = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": 3_000_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 1.0,
                    "lot_rounding": "floor",
                }
            },
            slot=slot,
            frequency="1d",
        )
        res_two = runner_two.run_backtest(
            factor_wide,
            close_wide,
            open_df=open_wide,
            rebalance_dates=[close_wide.index[1].to_pydatetime()],
        )
        runner_two.close()
        assert "error" not in res_two, res_two
        assert res_two["futures_execution"]["max_abs_lots"] == 2
        assert res_two["futures_execution"]["contract_multipliers"][c1] == 300.0
        # Bar 0 is flat (no T-1 signal); bar 1 close PnL: 2 lots * (4050 - 4000) * 300 = +30,000.
        assert res_two["portfolio"].value().iloc[0] == pytest.approx(3_000_000.0)
        assert res_two["portfolio"].value().iloc[1] == pytest.approx(3_030_000.0)
    finally:
        conn.close()

    # Case 3: lot_size = 5 rounds down to integer multiples of 5.
    assert round_contract_lots(990_000.0, 100.0, 100.0, lot_size=5, lot_rounding="floor") == 95
    assert round_contract_lots(-490_000.0, 100.0, 100.0, lot_size=5, lot_rounding="floor") == -45
    assert round_contract_lots(49_000.0, 100.0, 100.0, lot_size=5, lot_rounding="floor") == 0


def test_margin_rates_leverage_amplification_and_diagnostics(tmp_path: Path) -> None:
    """Verify margin occupancy, leveraged PnL amplification, and per-instrument margin_rates."""
    db_path = tmp_path / "futures_margin_amp.duckdb"
    # Session 2 (bar index 1) opens at 100 and closes at 102 (+2 each day).
    roll_start, end, c1, c2 = _build_rolling_futures_db(
        db_path,
        point_value=10.0,
        c1_opens=[100.0, 100.0, 100.0, 102.0, 104.0, 106.0, 108.0],
        c1_closes=[100.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        c2_opens=[120.0, 120.0, 120.0, 122.0, 124.0, 126.0, 128.0],
        c2_closes=[120.0, 120.0, 122.0, 124.0, 126.0, 128.0, 130.0],
    )

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(slot, "CU")

        # 1x notional (margin_rate=0.5, leverage=1.0 -> sufficient margin headroom across roll)
        r1 = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.5,
                    "leverage": 1.0,
                }
            },
            slot=slot,
            frequency="1d",
        )
        res1 = r1.run_backtest(
            factor_wide,
            close_wide,
            open_df=open_wide,
            rebalance_dates=[close_wide.index[1].to_pydatetime()],
        )
        r1.close()

        # 4x leveraged (margin_rate=0.20, leverage=4.0 -> 80% margin utilization)
        r4 = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.25,
                    "margin_rates": {"CU": 0.20},
                    "leverage": 4.0,
                }
            },
            slot=slot,
            frequency="1d",
        )
        res4 = r4.run_backtest(
            factor_wide,
            close_wide,
            open_df=open_wide,
            rebalance_dates=[close_wide.index[1].to_pydatetime()],
        )
        r4.close()
    finally:
        conn.close()

    assert res1["futures_execution"]["max_abs_lots"] == 100
    assert res4["futures_execution"]["max_abs_lots"] == 400
    # 1x return vs 4x return amplification on exact integer lots:
    pnl_1x = res1["portfolio"].value().iloc[-1] - 100_000.0
    pnl_4x = res4["portfolio"].value().iloc[-1] - 100_000.0
    assert pnl_4x == pytest.approx(4.0 * pnl_1x, rel=0, abs=1e-8)

    # Margin used at bar 1 (entry bar) on 400 lots @ close=102 * mult=10 * margin_rate=0.20 = 81,600.
    pf4 = res4["portfolio"]
    assert pf4.margin_used.iloc[1] == pytest.approx(400 * 102.0 * 10.0 * 0.20)
    assert pf4.margin_utilization.iloc[1] == pytest.approx(
        (400 * 102.0 * 10.0 * 0.20) / pf4.value().iloc[1]
    )


def test_insufficient_margin_clip_and_raise_and_maintenance_liquidation() -> None:
    """Verify clip vs raise on insufficient initial margin and maintenance margin liquidation."""
    dates = pd.bdate_range("2026-01-05", periods=4)
    # Price drops sharply on bar 1 from 100 -> 75 (-25%), then recovers on bar 2 -> 90.
    close_df = pd.DataFrame({"FUT": [100.0, 75.0, 90.0, 95.0]}, index=dates)
    open_df = pd.DataFrame({"FUT": [100.0, 100.0, 75.0, 90.0]}, index=dates)
    target_weights = pd.DataFrame({"FUT": [1.0, np.nan, np.nan, np.nan]}, index=dates)
    specs = {"FUT": InstrumentSpec("FUT", "futures", "CNY", lot_size=1, contract_multiplier=10.0)}

    # 1. Initial margin insufficient: margin_rate=0.80, leverage=2.0 -> req 160k > 100k.
    cfg_clip = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=0.80,
        leverage=2.0,
        insufficient_margin_policy="clip",
    )
    pf_clip, diag_clip = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=close_df,
        open_aligned=open_df,
        use_open=True,
        slot=None,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg_clip,
        target_value_mode="target_weight",
        instrument_specs=specs,
    )
    # Max affordable lots = floor(100_000 / (100 * 10 * 0.8)) = 125 lots.
    assert diag_clip["margin_clipped_events"] == 1
    assert diag_clip["max_abs_lots"] == 125
    assert pf_clip.margin_utilization.iloc[0] <= 1.0 + 1e-9

    cfg_raise = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=0.80,
        leverage=2.0,
        insufficient_margin_policy="raise",
    )
    with pytest.raises(FuturesMarginError, match="Insufficient margin"):
        simulate_futures_contract_portfolio(
            target_weights=target_weights,
            price_aligned=close_df,
            open_aligned=open_df,
            use_open=True,
            slot=None,
            frequency="1d",
            init_cash=100_000.0,
            buy_cost_rate=0.0,
            sell_cost_rate=0.0,
            slippage=0.0,
            config=cfg_raise,
            target_value_mode="target_weight",
            instrument_specs=specs,
        )

    # 2. Maintenance margin liquidation: margin_rate=0.20, leverage=4.0 (400 lots @ 100 = 400k notional).
    # Bar 1 close drops to 75 -> PnL = 400 * (75 - 100) * 10 = -100,000? Let's use -20% (close=80):
    # At close=80: equity = 100_000 - 80_000 = 20_000; maint_req @ 0.10 = 400 * 80 * 10 * 0.10 = 32_000 -> BREACH!
    close_drop = pd.DataFrame({"FUT": [100.0, 80.0, 95.0, 95.0]}, index=dates)
    open_drop = pd.DataFrame({"FUT": [100.0, 100.0, 80.0, 95.0]}, index=dates)

    cfg_liq = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=0.20,
        maintenance_margin_rate=0.10,
        leverage=4.0,
        liquidation_policy="liquidate",
    )
    pf_liq, diag_liq = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=close_drop,
        open_aligned=open_drop,
        use_open=True,
        slot=None,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg_liq,
        target_value_mode="target_weight",
        instrument_specs=specs,
    )
    assert diag_liq["liquidation_count"] == 1
    assert pf_liq.lots_history.iloc[1, 0] == 0
    assert pf_liq.lots_history.iloc[2, 0] == 0
    # Equity frozen at 20,000 after liquidation (misses bar 2 rebound to 95).
    assert pf_liq.value().iloc[-1] == pytest.approx(20_000.0)

    # Record-only policy does not liquidate, so bar 2 rebound restores equity!
    cfg_rec = replace(cfg_liq, liquidation_policy="record")
    pf_rec, diag_rec = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=close_drop,
        open_aligned=open_drop,
        use_open=True,
        slot=None,
        frequency="1d",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg_rec,
        target_value_mode="target_weight",
        instrument_specs=specs,
    )
    assert diag_rec["liquidation_count"] == 1
    assert pf_rec.lots_history.iloc[2, 0] == 400
    assert pf_rec.value().iloc[-1] == pytest.approx(80_000.0)

    cfg_liq_raise = replace(cfg_liq, liquidation_policy="raise")
    with pytest.raises(FuturesMarginError, match="Maintenance margin breached"):
        simulate_futures_contract_portfolio(
            target_weights=target_weights,
            price_aligned=close_drop,
            open_aligned=open_drop,
            use_open=True,
            slot=None,
            frequency="1d",
            init_cash=100_000.0,
            buy_cost_rate=0.0,
            sell_cost_rate=0.0,
            slippage=0.0,
            config=cfg_liq_raise,
            target_value_mode="target_weight",
            instrument_specs=specs,
        )


def test_config_validation_rejects_invalid_futures_execution_params() -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        parse_futures_execution_config({"top_k_strategy": {"execution_mode": "unknown"}})
    with pytest.raises(ValueError, match="margin_rate"):
        parse_futures_execution_config({"top_k_strategy": {"margin_rate": 0.0}})
    with pytest.raises(ValueError, match="maintenance_margin_rate"):
        parse_futures_execution_config(
            {"top_k_strategy": {"margin_rate": 0.15, "maintenance_margin_rate": 0.20}}
        )
    with pytest.raises(ValueError, match="lot_rounding"):
        parse_futures_execution_config({"top_k_strategy": {"lot_rounding": "ceil"}})
    with pytest.raises(ValueError, match="insufficient_margin_policy"):
        parse_futures_execution_config({"top_k_strategy": {"insufficient_margin_policy": "ignore"}})
    with pytest.raises(ValueError, match="liquidation_policy"):
        parse_futures_execution_config({"top_k_strategy": {"liquidation_policy": "ignore"}})
    with pytest.raises(ValueError, match="missing_execution_bar_policy"):
        parse_futures_execution_config({"top_k_strategy": {"missing_execution_bar_policy": "fallback"}})
    with pytest.raises(ValueError, match="roll_close_fallback_policy"):
        parse_futures_execution_config({"top_k_strategy": {"roll_close_fallback_policy": "ignore"}})


def test_futures_contract_audit_artifacts_and_margin_basis(tmp_path: Path) -> None:
    """Verify target_weight_basis='margin' and audit/return-series artifact generation."""
    db_path = tmp_path / "futures_audit.duckdb"
    roll_start, end, c1, c2 = _build_rolling_futures_db(
        db_path,
        point_value=10.0,
        c1_opens=[100.0, 100.0, 100.0, 102.0, 104.0, 106.0, 108.0],
        c1_closes=[100.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        c2_opens=[120.0, 120.0, 120.0, 122.0, 124.0, 126.0, 128.0],
        c2_closes=[120.0, 120.0, 122.0, 124.0, 126.0, 128.0, 130.0],
    )
    diag_dir = tmp_path / "diag"
    ret_dir = tmp_path / "returns"

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
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.20,
                    "target_weight_basis": "margin",
                }
            },
            slot=slot,
            frequency="1d",
            diagnostics_output_dir=str(diag_dir),
            return_series_output_dir=str(ret_dir),
            return_series_trial_id="fut_margin_trial",
        )
        res = runner.run_backtest(
            factor_wide,
            close_wide,
            open_df=open_wide,
            rebalance_freq="daily",
        )
        runner.close()
    finally:
        conn.close()

    assert "error" not in res, res
    # Bar 0 is flat (no T-1 factor); bar 1 enters 500 lots @ 100*10 with margin_rate=0.20.
    assert res["portfolio"].lots_history.iloc[0, 0] == 0
    assert res["portfolio"].lots_history.iloc[1, 0] == 500
    assert res["return_series_artifact"] is not None
    assert Path(res["return_series_artifact"]["path"]).exists()
    assert res["portfolio_audit_artifacts"] is not None
    assert Path(res["portfolio_audit_artifacts"]["orders_path"]).exists()
    assert res["execution_diagnostics"]["effective_execution_mode"] == "futures_contract"
    assert res["futures_execution"]["roll_count"] == 1


def test_roll_gap_invariance_10pct_vs_50pct_and_roll_sizing_match_notional(tmp_path: Path) -> None:
    """Equity curves for +10% vs +50% roll jump gaps are identical when post-roll increments match."""
    c1_opens = [100.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    c1_closes = [100.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0]
    # C2 base=110 (+10%) vs C2 base=150 (+50%), identical +2/bar post-roll increments.
    c2_10_o = [104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0]
    c2_10_c = [105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0]
    c2_50_o = [p + 40.0 for p in c2_10_o]
    c2_50_c = [p + 40.0 for p in c2_10_c]

    equities: list[np.ndarray] = []
    for tag, c2_o, c2_c in [("10pct", c2_10_o, c2_10_c), ("50pct", c2_50_o, c2_50_c)]:
        db_path = tmp_path / f"futures_gap_{tag}.duckdb"
        roll_start, end, _, _ = _build_rolling_futures_db(
            db_path,
            point_value=10.0,
            c1_opens=c1_opens,
            c1_closes=c1_closes,
            c2_opens=c2_o,
            c2_closes=c2_c,
        )
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
            _, close_wide, open_wide, factor_wide = _read_continuous_wide(slot, "CU")
            runner = VectorBTBacktestRunner(
                {
                    "top_k_strategy": {
                        "top_k": 1,
                        "score_source": "input_factor",
                        "holding_period": 10,
                        "init_cash": 100_000.0,
                        "execution_mode": "futures_contract",
                        "margin_rate": 0.5,
                        "roll_sizing": "same_lots",
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
            assert "error" not in res, res
            equities.append(res["portfolio"].value().to_numpy(dtype=float))
        finally:
            conn.close()

    np.testing.assert_allclose(equities[0], equities[1], rtol=0.0, atol=1e-8)

    # Also verify roll_sizing="match_notional" rescales lots across the jump gap.
    db_path = tmp_path / "futures_match_notional.duckdb"
    roll_start, end, _, _ = _build_rolling_futures_db(
        db_path,
        point_value=10.0,
        c1_opens=c1_opens,
        c1_closes=c1_closes,
        c2_opens=c2_50_o,
        c2_closes=c2_50_c,
    )
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(slot, "CU")
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 1.0,
                    "roll_sizing": "match_notional",
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
        assert "error" not in res, res
        roll_evt = res["futures_execution"]["roll_events"][0]
        # Entry on bar 1 @ 102.0 -> 98 lots; at roll (bar 3), C1 @ 106.0 = 103,880 notional;
        # C2 @ 152.0 -> floor(103,880 / 1,520) = 68 lots.
        assert roll_evt["closed_lots"] == 98
        assert roll_evt["opened_lots"] == 68
    finally:
        conn.close()


def test_missing_execution_bar_never_falls_back_to_adjusted_prices() -> None:
    """Missing concrete execution bar must fail fast by default and hold at last real price when configured."""
    dates = pd.bdate_range("2026-01-05", periods=4)
    utc_dates = pd.DatetimeIndex(dates).tz_localize("UTC")
    # Signal/aligned prices are ratio_back-adjusted (~109.7) while physical C1 is 100 -> 101 -> 102 -> 103.
    adj_close = pd.DataFrame({"CU": [109.7, 110.797, 111.894, 112.991]}, index=dates)
    adj_open = adj_close.copy()

    class _MockCalendar:
        @property
        def calendar_hash(self) -> str:
            return "mock_cal"

        def sessions(self, start=None, end=None):
            return dates

        def session_at(self, ts_utc, view="ETH", *, frequency=None):
            return None

        def bar_grid(self, freq: str, start: str, end: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "session_id": [d.strftime("%Y-%m-%d") for d in dates],
                    "timestamp_utc": utc_dates,
                }
            )

    class _MockUniverse:
        def resolve_universe(self, universe_id: str, *, as_of_utc):
            return ("CU2603.SHFE",)

        def resolve_active_contract(self, instrument_id: str, *, as_of_utc) -> str:
            return "CU2603.SHFE"

        def instrument_spec(self, contract_id: str) -> InstrumentSpec:
            return InstrumentSpec(
                contract_id, "futures", "CNY", lot_size=1, contract_multiplier=1.0
            )

    class _MissingBarFeed:
        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc) -> pd.DataFrame:
            return pd.DataFrame()

        def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc) -> pd.DataFrame:
            rows = []
            # Drop bar 1 (2026-01-06) completely from execution feed!
            for idx, (d, utc_ts, px) in enumerate(zip(dates, utc_dates, [100.0, 101.0, 102.0, 103.0])):
                if idx == 1:
                    continue
                rows.append(
                    {
                        "timestamp_utc": utc_ts,
                        "session_id": d.strftime("%Y-%m-%d"),
                        "instrument_id": "CU",
                        "concrete_instrument_id": "CU2603.SHFE",
                        "open": px,
                        "close": px,
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
                entry_price_field="close",
                exit_price_field="close",
                min_entry_offset_bars=1,
                exit_adjustment=0,
                return_unit="decimal",
                gap_rule="endpoints_only",
                boundary_policy="same_session",
                allow_short=True,
            )

        def tradability_mask(self, grid, *, instrument_id: str) -> pd.Series:
            return pd.Series(True, index=grid.index, dtype=bool)

    mock_slot = MarketSlot(
        slot_id="cn_futures",
        calendar=_MockCalendar(),
        universe=_MockUniverse(),
        bar_feed=_MissingBarFeed(),
        aux_features=_MockAux(),
        microstructure=_MockMicro(),
    )

    hold_weights = pd.DataFrame({"CU": [1.0, np.nan, np.nan, np.nan]}, index=dates)
    rebal_weights = pd.DataFrame({"CU": [1.0, 0.5, np.nan, np.nan]}, index=dates)

    # 1. Default policy="raise" fails fast instead of falling back to adjusted price_aligned.
    cfg_raise = FuturesExecutionConfig(execution_mode="futures_contract", margin_rate=1.0)
    for w_df in (hold_weights, rebal_weights):
        with pytest.raises(FuturesExecutionError, match=r"Missing physical execution bar.*CU"):
            simulate_futures_contract_portfolio(
                target_weights=w_df,
                price_aligned=adj_close,
                open_aligned=adj_open,
                use_open=False,
                slot=mock_slot,
                frequency="1d",
                init_cash=10_000.0,
                buy_cost_rate=0.0,
                sell_cost_rate=0.0,
                slippage=0.0,
                config=cfg_raise,
                target_value_mode="target_weight",
            )

    # 2. Explicit policy="hold" does not trade and marks to last real price (no fake +882/+441 PnL).
    cfg_hold = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=1.0,
        missing_execution_bar_policy="hold",
    )
    for w_df in (hold_weights, rebal_weights):
        pf, diag = simulate_futures_contract_portfolio(
            target_weights=w_df,
            price_aligned=adj_close,
            open_aligned=adj_open,
            use_open=False,
            slot=mock_slot,
            frequency="1d",
            init_cash=10_000.0,
            buy_cost_rate=0.0,
            sell_cost_rate=0.0,
            slippage=0.0,
            config=cfg_hold,
            target_value_mode="target_weight",
        )
        np.testing.assert_allclose(
            pf.value().to_numpy(dtype=float),
            [10_000.0, 10_000.0, 10_200.0, 10_300.0],
            rtol=0.0,
            atol=1e-8,
        )
        assert list(pf.lots_history["CU"]) == [100, 100, 100, 100]
        assert diag["missing_execution_bar_count"] == 1


def test_missing_roll_close_bar_fails_fast_by_default_and_allows_explicit_last_close(
    tmp_path: Path,
) -> None:
    """Missing C_old bar at roll must fail fast by default instead of silently using prev close."""
    db_path = tmp_path / "futures_missing_roll_close.duckdb"
    roll_start, end, c1, c2 = _build_rolling_futures_db(db_path, jump_pct=0.10)

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        base_slot = build_cn_futures_slot(conn, product_root="CU", start=roll_start, end=end)
        _, close_wide, open_wide, factor_wide = _read_continuous_wide(base_slot, "CU")

        class _MissingOldRollBarFeed:
            def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
                return base_slot.bar_feed.read_signal_bars(
                    instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
                )

            def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc):
                if list(instruments) == [c1]:
                    raise RuntimeError(f"Missing roll exit bar for {c1}")
                return base_slot.bar_feed.read_execution_bars(
                    instruments, frequency, start_utc=start_utc, end_utc=end_utc
                )

        wrapped_slot = replace(base_slot, bar_feed=_MissingOldRollBarFeed())

        runner_raise = VectorBTBacktestRunner(
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
            slot=wrapped_slot,
            frequency="1d",
        )
        with pytest.raises(FuturesExecutionError, match=r"Missing physical roll exit bar.*CU"):
            runner_raise.run_backtest(
                factor_wide,
                close_wide,
                open_df=open_wide,
                rebalance_dates=[close_wide.index[1].to_pydatetime()],
            )
        runner_raise.close()

        runner_fallback = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 10,
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.5,
                    "roll_close_fallback_policy": "last_close",
                }
            },
            slot=wrapped_slot,
            frequency="1d",
        )
        res = runner_fallback.run_backtest(
            factor_wide,
            close_wide,
            open_df=open_wide,
            rebalance_dates=[close_wide.index[1].to_pydatetime()],
        )
        runner_fallback.close()
        assert "error" not in res, res
        assert res["futures_execution"]["roll_close_fallback_count"] == 1
        assert res["futures_execution"]["roll_events"][0]["close_price_fallback"] is True
    finally:
        conn.close()
