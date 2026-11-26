"""Regression and Release Gate contract tests for Stage 3 Line B (NERE-1430).

Covers:
1. P1-05: Unified execution timestamp & authoritative trading ``session_id`` resolution
   across ``1d``, ``1w``, intraday, and overnight China futures calendars.
2. P1-06: Held-contract valuation during multi-bar deferred rolls (numerical counterexample
   and boundary checks across ``1d`` and ``1w``).
3. P1-07: Per-instrument/contract ``margin_rates`` sizing across ``branch_topk``,
   ``branch_overlay``, and direct simulation paths (``1d`` and ``1w``).
4. Release Gate Execution Causality: perturbing post-execution prices, microstructure,
   signals, or active-contract schedules never changes earlier orders, fills, or positions.
5. Release Gate Position Accounting: bar-by-bar conservation of equity, realized/unrealized
   PnL, fees, margin_used, margin_utilization, ``asset_values``, and ``prev_weight_map``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import pytest

from data_pipeline.loaders.market_bar_importer import import_market_bars
from factor_synthesis.vectorbt_adapter import (
    FuturesExecutionConfig,
    FuturesExecutionError,
    VectorBTBacktestRunner,
    futures_execution as fut_exec_mod,
)
from factor_synthesis.vectorbt_adapter.futures_execution import (
    simulate_futures_contract_portfolio,
)
from factor_synthesis.vectorbt_adapter.portfolio_policy import SharedPortfolioPolicy
from market_db_samples import declare_futures_bars
from slots.cn_futures import build_cn_futures_slot
from slots.declarative.slot import DeclarativeMicrostructure
from slots.declarative.source import FrameSource, ProviderConfig
from utils.bars.futures_roll import build_roll_schedule
from utils.session_calendar import SessionCalendar
from utils.slots.contracts import ExecutionProfile, InstrumentSpec, MarketSlot
from utils.time_axis import BarSchedule, Frequency

from test_futures_single_account_and_tradability import (
    _make_contract_priced_mock_slot,
)


def _build_weekly_calendar_slot(
    tmp_path: Path,
    *,
    cal: SessionCalendar,
    instruments: List[str],
    open_panel: Dict[str, List[float]],
    close_panel: Dict[str, List[float]],
    use_open: bool,
    halt_rows: Optional[List[Tuple[str, str, str]]] = None,
    session_at_strict: bool = False,
    recorded_queries: Optional[List[Dict[str, Any]]] = None,
    contract_map: Optional[Dict[Tuple[int, str], str]] = None,
    contract_open_panel: Optional[Dict[str, List[float]]] = None,
    contract_close_panel: Optional[Dict[str, List[float]]] = None,
    specs: Optional[Dict[str, InstrumentSpec]] = None,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    allow_short: bool = False,
) -> Tuple[MarketSlot, pd.DataFrame]:
    """Build a ``1w`` MarketSlot backed by a real ``SessionCalendar``."""
    grid = cal.bar_grid("1w")
    utc_dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
    sids = grid["session_id"].astype(str).tolist()

    spec_map = dict(specs or {})
    for inst in instruments:
        spec_map.setdefault(
            inst, InstrumentSpec(inst, "futures", "CNY", lot_size=1, contract_multiplier=10.0)
        )
    if contract_map:
        for cid in set(contract_map.values()):
            spec_map.setdefault(
                cid, InstrumentSpec(cid, "futures", "CNY", lot_size=1, contract_multiplier=10.0)
            )

    class _WeeklyUniverse:
        def resolve_universe(self, universe_id: str, *, as_of_utc: Any) -> Tuple[str, ...]:
            del universe_id, as_of_utc
            return tuple(instruments)

        def resolve_active_contract(self, instrument_id: str, *, as_of_utc: Any) -> str:
            if contract_map:
                ts = pd.Timestamp(as_of_utc)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                for pos, bar_ts in enumerate(utc_dates):
                    if bar_ts == ts and (pos, instrument_id) in contract_map:
                        return contract_map[(pos, instrument_id)]
            return instrument_id

        def instrument_spec(self, contract_id: str) -> InstrumentSpec:
            return spec_map[contract_id]

    class _WeeklyBarFeed:
        def read_signal_bars(self, req_insts, frequency, *, start_utc, end_utc, as_of_utc):
            return pd.DataFrame()

        def read_execution_bars(self, req_insts, frequency, *, start_utc, end_utc):
            start_ts = pd.Timestamp(start_utc)
            end_ts = pd.Timestamp(end_utc)
            start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
            end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
            rows: List[Dict[str, Any]] = []
            for pos, (utc_ts, sid) in enumerate(zip(utc_dates, sids)):
                if utc_ts < start_ts or utc_ts > end_ts:
                    continue
                for inst in req_insts:
                    if inst in open_panel:
                        o_px = open_panel[inst][pos]
                        c_px = close_panel[inst][pos]
                        cid = contract_map.get((pos, inst), inst) if contract_map else inst
                    elif contract_open_panel and inst in contract_open_panel:
                        o_px = contract_open_panel[inst][pos]
                        c_px = contract_close_panel[inst][pos]  # type: ignore[index]
                        cid = inst
                    else:
                        continue
                    rows.append(
                        {
                            "timestamp_utc": utc_ts,
                            "session_id": sid,
                            "instrument_id": inst,
                            "concrete_instrument_id": cid,
                            "open": o_px,
                            "close": c_px,
                            "volume": 1_000_000.0,
                            "money": 1_000_000.0 * c_px,
                        }
                    )
            return pd.DataFrame(rows)

    class _EmptyAux:
        def read_aux_features(self, req_grid, *, features):
            return pd.DataFrame()

    source = None
    if halt_rows:
        csv_name = f"weekly_halts_{len(halt_rows)}_{abs(hash(tuple(halt_rows)))}.csv"
        lines = ["instrument_id,available_at_utc,tradable"]
        for inst_id, ts_str, tradable_str in halt_rows:
            lines.append(f"{inst_id},{ts_str},{tradable_str}")
        (tmp_path / csv_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        allow_short=allow_short,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    decl_micro = DeclarativeMicrostructure(cal, profile, source)

    class _WeeklyMicro:
        execution_profile = profile

        def tradability_mask(self, query_grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
            mask = decl_micro.tradability_mask(query_grid, instrument_id=instrument_id)
            if session_at_strict:
                # Mirror slots/futures/slot.py:338-341 contract:
                # cal.session_at(ts, view) or cal.session_at(ts, view, frequency='1d')
                # must match row.session_id.
                strict_vals = []
                for ts, sid in zip(query_grid["timestamp_utc"], query_grid["session_id"]):
                    sess = cal.session_at(ts, "ETH") or cal.session_at(ts, "ETH", frequency="1d")
                    strict_vals.append(sess is not None and sess.session_date == str(sid))
                mask = mask & pd.Series(strict_vals, index=query_grid.index, dtype=bool)
            if recorded_queries is not None:
                for row_idx, row in query_grid.iterrows():
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

    slot = MarketSlot(
        slot_id="cn_futures",
        calendar=cal,
        universe=_WeeklyUniverse(),
        bar_feed=_WeeklyBarFeed(),
        aux_features=_EmptyAux(),
        microstructure=_WeeklyMicro(),
    )
    return slot, grid


@pytest.mark.parametrize("session_at_strict", [False, True], ids=["DeclarativeMicro", "FuturesSessionAtMicro"])
@pytest.mark.parametrize("use_open", [True, False], ids=["open", "close"])
def test_p1_05_weekly_xshg_execution_matches_authoritative_session_id(
    tmp_path: Path,
    session_at_strict: bool,
    use_open: bool,
) -> None:
    """P1-05 counterexample: XSHG 2024-01-08..19 1w buys 100 lots instead of [0, 0] blocked orders."""
    cal = SessionCalendar("XSHG", "2024-01-08", "2024-01-19")
    open_p = {"FUT": [100.0, 100.0]}
    close_p = {"FUT": [102.0, 104.0]} if use_open else {"FUT": [100.0, 100.0]}
    queries: List[Dict[str, Any]] = []
    slot, grid = _build_weekly_calendar_slot(
        tmp_path,
        cal=cal,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        use_open=use_open,
        session_at_strict=session_at_strict,
        recorded_queries=queries,
    )
    idx = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
    assert list(grid["session_id"].astype(str)) == ["2024-01-12", "2024-01-19"]

    target_weights = pd.DataFrame({"FUT": [1.0, 1.0]}, index=idx)
    cfg = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=1.0,
        leverage=1.0,
    )
    pf, diag = simulate_futures_contract_portfolio(
        target_weights=target_weights,
        price_aligned=pd.DataFrame(close_p, index=idx),
        open_aligned=pd.DataFrame(open_p, index=idx),
        use_open=use_open,
        slot=slot,
        frequency="1w",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg,
        target_value_mode="target_weight",
    )

    assert diag["blocked_order_count"] == 0
    assert list(pf.lots_history["FUT"]) == [100, 100]

    buy_queries = [q for q in queries if q["side"] == "buy"]
    assert len(buy_queries) >= 1
    if use_open:
        assert buy_queries[0]["timestamp_utc"] == pd.Timestamp("2024-01-08 01:30:00+00:00")
        assert buy_queries[0]["session_id"] == "2024-01-08"
        assert buy_queries[0]["allowed"] is True
        if len(buy_queries) > 1:
            assert buy_queries[1]["timestamp_utc"] == pd.Timestamp("2024-01-15 01:30:00+00:00")
            assert buy_queries[1]["session_id"] == "2024-01-15"
    else:
        assert buy_queries[0]["timestamp_utc"] == pd.Timestamp("2024-01-12 07:00:00+00:00")
        assert buy_queries[0]["session_id"] == "2024-01-12"
        assert buy_queries[0]["allowed"] is True


@pytest.mark.parametrize("use_public_protocol", [False, True], ids=["SessionCalendar", "CalendarProtocol"])
@pytest.mark.parametrize("frequency", ["1d", "1w", "1h"])
def test_p1_05_validator_resolves_overnight_futures_session_id_without_utc_date_bug(
    use_public_protocol: bool,
    frequency: str,
) -> None:
    """Overnight futures sessions (21:00 CST = 13:00 UTC) must map to trading session_date, not UTC date."""
    # Night segment opens at 13:00 UTC (21:00 CST) on prior calendar day (open_day_offset=-1),
    # and day segment closes at 07:00 UTC (15:00 CST) on session_date.
    cal = SessionCalendar(
        "SYNTH_NIGHT_FUT",
        "2026-01-05",
        "2026-01-16",
        custom_sessions={
            "timezone": "Asia/Shanghai",
            "weekdays": [0, 1, 2, 3, 4],
            "segments": [
                {"open": "21:00", "close": "23:00", "open_day_offset": -1, "close_day_offset": -1},
                {"open": "09:00", "close": "15:00", "open_day_offset": 0, "close_day_offset": 0},
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
        "2026-01-16" if frequency != "1h" else "2026-01-06",
        view="ETH",
    )
    bar_utc_index = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
    session_ids = {(idx, "CU"): str(sid) for idx, sid in enumerate(grid["session_id"].tolist())}

    seen: List[Tuple[pd.Timestamp, str, str]] = []

    class _NightMicro:
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
            ts = pd.Timestamp(query_grid["timestamp_utc"].iloc[0])
            sid = str(query_grid["session_id"].iloc[0])
            field = str(query_grid["execution_price_field"].iloc[0])
            seen.append((ts, sid, field))
            return pd.Series(True, index=query_grid.index, dtype=bool)

    class _NoopFeed:
        def read_signal_bars(self, *a, **kw):
            return pd.DataFrame()

        def read_execution_bars(self, *a, **kw):
            return pd.DataFrame()

    class _NoopUniverse:
        def instrument_spec(self, instrument_id: str) -> InstrumentSpec:
            return InstrumentSpec(instrument_id, "futures", "CNY")

        def resolve_universe(self, universe_id: str, *, as_of_utc: Any):
            return ("CU",)

        def resolve_active_contract(self, instrument_id: str, *, as_of_utc: Any) -> str:
            return instrument_id

    class _NoopAux:
        def read_aux_features(self, g, *, features):
            return pd.DataFrame()

    slot = MarketSlot(
        slot_id="cn_futures",
        calendar=active_cal,
        bar_feed=_NoopFeed(),
        universe=_NoopUniverse(),
        aux_features=_NoopAux(),
        microstructure=_NightMicro(),
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
        for bar_idx in range(len(bar_utc_index)):
            assert validator.is_mask_tradable(
                bar_idx=bar_idx,
                inst="CU",
                contract="CU",
                side="buy",
                price=100.0,
                execution_price_field=field,
            )
        assert len(seen) == len(bar_utc_index)
        if frequency == "1w" and field == "open":
            # Week 1: Mon 2026-01-05 opens on 2026-01-04 13:00Z, session_id MUST be 2026-01-05 (not 2026-01-04 or 2026-01-09).
            # Week 2: Mon 2026-01-12 opens on 2026-01-11 13:00Z, session_id MUST be 2026-01-12.
            assert seen[0] == (pd.Timestamp("2026-01-04 13:00:00+00:00"), "2026-01-05", "open")
            assert seen[1] == (pd.Timestamp("2026-01-11 13:00:00+00:00"), "2026-01-12", "open")
        elif frequency == "1w" and field == "close":
            assert seen[0] == (pd.Timestamp("2026-01-09 07:00:00+00:00"), "2026-01-09", "close")
            assert seen[1] == (pd.Timestamp("2026-01-16 07:00:00+00:00"), "2026-01-16", "close")
        elif frequency == "1d" and field == "open":
            # First session 2026-01-05 opens on 2026-01-04 13:00Z -> session_id MUST be 2026-01-05!
            assert seen[0] == (pd.Timestamp("2026-01-04 13:00:00+00:00"), "2026-01-05", "open")
        elif frequency == "1h":
            # Night bar at 2026-01-04 13:00Z belongs to session 2026-01-05!
            expected_close = (
                pd.Timestamp("2026-01-04 14:00:00+00:00")
                if field == "close"
                else pd.Timestamp("2026-01-04 13:00:00+00:00")
            )
            assert seen[0] == (expected_close, "2026-01-05", field)


def test_p1_05_native_futures_slot_night_segment_end_and_first_day_coverage() -> None:
    """Native futures slot tradability_mask must accept night segment-end close on coverage start & interior."""
    from utils.time_axis.futures import futures_calendar

    # DCE (M) night segment runs 21:00..23:00 CST (13:00Z..15:00Z on the prior calendar day).
    # Slot covers 2026-01-06..2026-01-07 so 2026-01-06 is coverage first day and its 2nd 1h night bar
    # closes at 2026-01-05 15:00Z (23:00 CST on 2026-01-05, prior to coverage start!).
    fcal = futures_calendar("M")
    cal_import = SessionCalendar(
        fcal["calendar_id"],
        "2026-01-05",
        "2026-01-07",
        rth_reference_calendar=fcal["rth_reference_calendar"],
    )
    sched_import = BarSchedule(cal_import, "1h", bucket_strategy="segment_anchored")
    import_sessions = cal_import.sessions()
    slot_rows = pd.concat([sched_import.for_session(day) for day in import_sessions], ignore_index=True)

    cal = SessionCalendar(
        fcal["calendar_id"],
        "2026-01-06",
        "2026-01-07",
        rth_reference_calendar=fcal["rth_reference_calendar"],
    )
    sched_1h = BarSchedule(cal, "1h", bucket_strategy="segment_anchored")
    sessions = cal.sessions()
    assert list(sessions.strftime("%Y-%m-%d")) == ["2026-01-06", "2026-01-07"]
    c_alive = "M2603.DCE"
    c_expiring_0106 = "M2601.DCE"
    meta = pd.DataFrame(
        [
            dict(
                instrument_id=c_alive,
                product_root="M",
                asset_class="futures",
                exchange="DCE",
                timezone=cal.timezone,
                contract_expiry="2026-03-16",
                contract_month="2026-03",
                price_adjustment="none",
                point_value=10.0,
                rth_reference_calendar=fcal["rth_reference_calendar"],
            ),
            dict(
                instrument_id=c_expiring_0106,
                product_root="M",
                asset_class="futures",
                exchange="DCE",
                timezone=cal.timezone,
                contract_expiry="2026-01-06",
                contract_month="2026-01",
                price_adjustment="none",
                point_value=10.0,
                rth_reference_calendar=fcal["rth_reference_calendar"],
            ),
        ]
    )
    bars_list = []
    for cid, oi_val in [(c_alive, 500.0), (c_expiring_0106, 100.0)]:
        bars_list.append(
            pd.DataFrame(
                {
                    "timestamp_utc": slot_rows["start_utc"],
                    "instrument_id": cid,
                    "frequency": "1h",
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "open_interest": oi_val,
                    "volume": 500.0,
                    "source": "SYNTH",
                    "data_version": "v1",
                }
            )
        )

    with duckdb.connect(config={"threads": 1}) as conn:
        declare_futures_bars(conn, "DCE")
        import_market_bars(
            conn,
            pd.concat(bars_list, ignore_index=True),
            meta,
            bucket_strategy="segment_anchored",
            legacy_compatibility=False,
        )
        build_roll_schedule(conn, product_roots=["M"], start="2026-01-06", end="2026-01-07")
        slot = build_cn_futures_slot(conn, product_root="M", start="2026-01-06", end="2026-01-07")

        grid_1h = cal.bar_grid("1h", "2026-01-06", "2026-01-07", view="ETH", bucket_strategy="segment_anchored")
        bar_utc_index = pd.DatetimeIndex(grid_1h["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
        session_ids = {(idx, "M"): str(sid) for idx, sid in enumerate(grid_1h["session_id"].tolist())}
        validator = fut_exec_mod._FuturesOrderValidator(
            slot=slot,
            bar_utc_index=bar_utc_index,
            session_ids=session_ids,
            frequency=Frequency.parse("1h"),
            default_execution_price_field="close",
        )

        # 2nd 60-minute night bar on coverage first day (2026-01-06) closes at 2026-01-05 15:00Z (23:00 CST).
        s0_slots = sched_1h.for_session("2026-01-06")
        night_seg_end_0 = pd.Timestamp(s0_slots.iloc[1]["end_utc"])
        assert night_seg_end_0 == pd.Timestamp("2026-01-05 15:00:00+00:00")
        assert night_seg_end_0.tz_convert(cal.timezone).strftime("%Y-%m-%d") == "2026-01-05"
        assert validator.is_mask_tradable(
            bar_idx=1,
            inst="M",
            contract=c_alive,
            side="buy",
            price=101.0,
            execution_price_field="close",
        ) is True

        # Interior session (2026-01-07) night segment end: 2026-01-06 15:00Z.
        s1_start_idx = len(s0_slots)
        s1_slots = sched_1h.for_session("2026-01-07")
        night_seg_end_1 = pd.Timestamp(s1_slots.iloc[1]["end_utc"])
        assert night_seg_end_1 == pd.Timestamp("2026-01-06 15:00:00+00:00")
        assert validator.is_mask_tradable(
            bar_idx=s1_start_idx + 1,
            inst="M",
            contract=c_alive,
            side="buy",
            price=101.0,
            execution_price_field="close",
        ) is True

        # Direct native microstructure checks at night segment end:
        # - close on authoritative trading day 2026-01-06 -> True (coverage first day!)
        # - open at segment end -> False
        # - wrong session_id '2026-01-06' for 2026-01-07's night segment end -> False
        # - contract expiring ON 2026-01-06 checked at 2026-01-05 15:00Z -> False (expiry uses 2026-01-06, not 2026-01-05!)
        probe_grid = pd.DataFrame(
            {
                "instrument_id": [c_alive, c_alive, c_alive, c_expiring_0106],
                "timestamp_utc": pd.DatetimeIndex(
                    [night_seg_end_0, night_seg_end_0, night_seg_end_1, night_seg_end_0],
                    tz="UTC",
                ),
                "session_id": ["2026-01-06", "2026-01-06", "2026-01-06", "2026-01-06"],
                "execution_price_field": ["close", "open", "close", "close"],
            }
        )
        alive_mask = slot.microstructure.tradability_mask(probe_grid.iloc[:3], instrument_id=c_alive)
        assert alive_mask.tolist() == [True, False, False]
        expiring_mask = slot.microstructure.tradability_mask(probe_grid.iloc[3:4], instrument_id=c_expiring_0106)
        assert expiring_mask.tolist() == [False]

        # Full execution engine regression: 2nd 60-minute night bar on coverage first day (bar key 14:00Z,
        # execution endpoint 2026-01-05 15:00Z, session_id 2026-01-06) must bind M2603.DCE and execute
        # 100 lots @ 101.0 under both hold & raise.
        night_bar_ts = bar_utc_index[1]
        assert validator._resolve_execution_context(
            bar_idx=1, inst="M", execution_price_field="close"
        ) == (night_seg_end_0, "2026-01-06")
        target_weights = pd.DataFrame({"M": 1.0}, index=bar_utc_index)
        target_weights.iloc[0, 0] = 0.0
        price_aligned = pd.DataFrame({"M": 101.0}, index=bar_utc_index)
        open_aligned = pd.DataFrame({"M": 100.0}, index=bar_utc_index)
        for policy in ("hold", "raise"):
            cfg = FuturesExecutionConfig(
                execution_mode="futures_contract",
                margin_rate=1.0,
                leverage=1.0,
                missing_execution_bar_policy=policy,
            )
            pf, diag = simulate_futures_contract_portfolio(
                target_weights=target_weights,
                price_aligned=price_aligned,
                open_aligned=open_aligned,
                use_open=False,
                slot=slot,
                frequency="1h",
                init_cash=101_000.0,
                buy_cost_rate=0.0,
                sell_cost_rate=0.0,
                slippage=0.0,
                config=cfg,
                target_value_mode="target_weight",
            )
            assert diag["missing_execution_bar_count"] == 0
            orders = pf.orders.records_readable
            night_orders = orders.loc[
                pd.to_datetime(orders["Timestamp"]) == night_bar_ts
            ].reset_index(drop=True)
            assert list(night_orders["Concrete Instrument"]) == [c_alive]
            assert list(night_orders["Lots"]) == [100]
            assert float(night_orders["Price"].iloc[0]) == pytest.approx(101.0)
            assert pf.lots_history.loc[night_bar_ts, "M"] == 100


def test_p1_05_weekly_missing_friday_close_does_not_block_monday_open(tmp_path: Path) -> None:
    """Missing Friday close on 1w must never block Monday open execution (use_open=True)."""
    cal = SessionCalendar("XSHG", "2024-01-08", "2024-01-12")
    open_p = {"FUT": [100.0]}
    close_p = {"FUT": [float("nan")]}
    queries: List[Dict[str, Any]] = []
    slot, grid = _build_weekly_calendar_slot(
        tmp_path,
        cal=cal,
        instruments=["FUT"],
        open_panel=open_p,
        close_panel=close_p,
        use_open=True,
        session_at_strict=True,
        recorded_queries=queries,
    )
    idx = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
    w_df = pd.DataFrame({"FUT": [1.0]}, index=idx)

    # 1. Policy='hold': Monday open buys 100 lots @ 100.0; close missing only increments diagnostic counter.
    cfg_hold = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=1.0,
        leverage=1.0,
        missing_execution_bar_policy="hold",
    )
    pf_hold, diag_hold = simulate_futures_contract_portfolio(
        target_weights=w_df,
        price_aligned=pd.DataFrame({"FUT": [100.0]}, index=idx),
        open_aligned=pd.DataFrame(open_p, index=idx),
        use_open=True,
        slot=slot,
        frequency="1w",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg_hold,
        target_value_mode="target_weight",
    )
    assert list(pf_hold.lots_history["FUT"]) == [100]
    assert pf_hold.value().iloc[0] == pytest.approx(100_000.0)
    assert diag_hold["blocked_order_count"] == 0
    assert diag_hold["missing_execution_bar_count"] == 1

    # 2. Policy='raise': flat/zero target never touches missing close and succeeds.
    cfg_raise = FuturesExecutionConfig(
        execution_mode="futures_contract",
        margin_rate=1.0,
        leverage=1.0,
        missing_execution_bar_policy="raise",
    )
    pf_flat, diag_flat = simulate_futures_contract_portfolio(
        target_weights=pd.DataFrame({"FUT": [0.0]}, index=idx),
        price_aligned=pd.DataFrame({"FUT": [100.0]}, index=idx),
        open_aligned=pd.DataFrame(open_p, index=idx),
        use_open=True,
        slot=slot,
        frequency="1w",
        init_cash=100_000.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        config=cfg_raise,
        target_value_mode="target_weight",
    )
    assert list(pf_flat.lots_history["FUT"]) == [0]
    assert diag_flat["missing_execution_bar_count"] == 0

    # 3. Policy='raise' when held through Friday close: Monday open is validated & executed first,
    #    and missing close only raises at end-of-bar close valuation.
    queries.clear()
    with pytest.raises(FuturesExecutionError, match="Missing physical close mark bar"):
        simulate_futures_contract_portfolio(
            target_weights=w_df,
            price_aligned=pd.DataFrame({"FUT": [100.0]}, index=idx),
            open_aligned=pd.DataFrame(open_p, index=idx),
            use_open=True,
            slot=slot,
            frequency="1w",
            init_cash=100_000.0,
            buy_cost_rate=0.0,
            sell_cost_rate=0.0,
            slippage=0.0,
            config=cfg_raise,
            target_value_mode="target_weight",
        )
    assert [(q["execution_price_field"], q["session_id"]) for q in queries] == [
        ("open", "2024-01-08"),
        ("close", "2024-01-12"),
    ]


@pytest.mark.parametrize("frequency", ["1d", "1w"])
def test_p1_06_deferred_roll_held_contract_valuation_numerical_counterexample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frequency: str,
) -> None:
    """P1-06 counterexample (1d & 1w): equity 100k, A holds 500@100 (A_NEW=200), B holds 500@100 -> 50%/50%."""
    if frequency == "1d":
        dates = pd.bdate_range("2024-01-08", periods=6)
        slot_builder_cal = None
    else:
        slot_builder_cal = SessionCalendar("XSHG", "2024-01-08", "2024-02-23")
        grid = slot_builder_cal.bar_grid("1w")
        dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
        assert len(dates) == 6

    # Bar 1 enters A_OLD (500 lots @ 100, mult=1) and B_C1 (500 lots @ 100, mult=1).
    # Bars 2 & 3: A rolls to A_NEW (200) while A_OLD (100) is blocked for 2 consecutive bars!
    open_p = {"A": [100.0, 100.0, 200.0, 200.0, 200.0, 200.0], "B": [100.0] * 6}
    close_p = {"A": [100.0, 100.0, 200.0, 200.0, 200.0, 200.0], "B": [100.0] * 6}
    c_open = {"A_OLD": [100.0] * 6, "A_NEW": [200.0] * 6, "B_C1": [100.0] * 6}
    c_close = {"A_OLD": [100.0] * 6, "A_NEW": [200.0] * 6, "B_C1": [100.0] * 6}
    contract_map = {
        (0, "A"): "A_OLD",
        (1, "A"): "A_OLD",
        (2, "A"): "A_NEW",
        (3, "A"): "A_NEW",
        (4, "A"): "A_NEW",
        (5, "A"): "A_NEW",
        **{(i, "B"): "B_C1" for i in range(6)},
    }
    specs = {
        k: InstrumentSpec(k, "futures", "CNY", lot_size=1, contract_multiplier=1.0)
        for k in ("A", "B", "A_OLD", "A_NEW", "B_C1")
    }

    if frequency == "1d":
        blocked_sids = {dates[2].strftime("%Y-%m-%d"), dates[3].strftime("%Y-%m-%d")}

        def _mask_fn(qgrid: pd.DataFrame, instrument_id: str) -> pd.Series:
            if instrument_id == "A_OLD":
                return ~qgrid["session_id"].astype(str).isin(blocked_sids)
            return pd.Series(True, index=qgrid.index, dtype=bool)

        slot = _make_contract_priced_mock_slot(
            dates=dates,
            instruments=["A", "B"],
            open_panel=open_p,
            close_panel=close_p,
            contract_open_panel=c_open,
            contract_close_panel=c_close,
            contract_map=contract_map,
            tradability_fn=_mask_fn,
            specs=specs,
            allow_short=False,
        )
    else:
        assert slot_builder_cal is not None
        # For 1w use_open=True, bar 2 & bar 3 open on Mondays 2024-01-22 and 2024-01-29!
        halt_rows = [
            ("A_OLD", "2024-01-22T01:30:00Z", "false"),
            ("A_OLD", "2024-02-05T01:30:00Z", "true"),
        ]
        slot, _ = _build_weekly_calendar_slot(
            tmp_path,
            cal=slot_builder_cal,
            instruments=["A", "B"],
            open_panel=open_p,
            close_panel=close_p,
            use_open=True,
            halt_rows=halt_rows,
            session_at_strict=True,
            contract_map=contract_map,
            contract_open_panel=c_open,
            contract_close_panel=c_close,
            specs=specs,
            allow_short=False,
        )

    observed_prev_weights: List[Dict[str, float]] = []
    orig_blend = VectorBTBacktestRunner._blend_partial_rebalance

    def _spy_blend(self, target_map: Dict[str, float], prev_weight_map: Dict[str, float]) -> Dict[str, float]:
        observed_prev_weights.append(dict(prev_weight_map))
        return orig_blend(self, target_map, prev_weight_map)

    monkeypatch.setattr(VectorBTBacktestRunner, "_blend_partial_rebalance", _spy_blend)

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
                "partial_rebalance": {"enabled": True, "trade_rate": 0.5, "min_weight": 0.01},
            }
        },
        slot=slot,
        frequency=frequency,
    )
    scores = pd.DataFrame({"A": [2.0] * 6, "B": [1.0] * 6}, index=dates)
    rebal_dates = [dates[1].to_pydatetime(), dates[2].to_pydatetime(), dates[3].to_pydatetime()]
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates),
        rebalance_dates=rebal_dates,
    )
    runner.close()
    assert "error" not in res, res

    pf = res["portfolio"]
    diag = res["futures_execution"]
    # Bar 1 enters 500 lots A_OLD @ 100, 500 lots B_C1 @ 100.
    assert pf.lots_history.loc[dates[1], "A"] == 500
    assert pf.lots_history.loc[dates[1], "B"] == 500
    # Bars 2 & 3 (multi-bar deferred roll): A_OLD valued at 100 (NOT A_NEW=200),
    # so prev_weight_map == {'A': 0.5, 'B': 0.5} on BOTH deferred bars!
    assert len(observed_prev_weights) == 3
    assert observed_prev_weights[1] == pytest.approx({"A": 0.5, "B": 0.5})
    assert observed_prev_weights[2] == pytest.approx({"A": 0.5, "B": 0.5})
    assert list(pf.lots_history["B"]) == [0, 500, 500, 500, 500, 500]
    assert pf.lots_history.loc[dates[2], "A"] == 500
    assert pf.lots_history.loc[dates[3], "A"] == 500
    assert diag["roll_count"] == 1
    assert diag["roll_events"][0]["deferred_bars"] == 2
    assert pd.Timestamp(diag["roll_events"][0]["timestamp"]) == dates[4]


@pytest.mark.parametrize("frequency", ["1d", "1w"])
@pytest.mark.parametrize("branch_mode", ["branch_topk", "branch_overlay", "direct"])
def test_p1_07_per_instrument_margin_rates_across_branches_and_frequencies(
    tmp_path: Path,
    frequency: str,
    branch_mode: str,
) -> None:
    """P1-07 counterexample (1d & 1w): 100k, margin_rates={'A':0.1,'B':0.2} -> A 500 lots, B 250 lots."""
    if frequency == "1d":
        dates = pd.bdate_range("2024-01-08", periods=4)
    else:
        cal = SessionCalendar("XSHG", "2024-01-08", "2024-02-02")
        grid = cal.bar_grid("1w")
        dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
        assert len(dates) == 4

    open_p = {"A": [100.0] * 4, "B": [100.0] * 4}
    close_p = {"A": [100.0] * 4, "B": [100.0] * 4}
    specs = {
        "A": InstrumentSpec("A", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        "B": InstrumentSpec("B", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
    }

    if frequency == "1d":
        slot = _make_contract_priced_mock_slot(
            dates=dates,
            instruments=["A", "B"],
            open_panel=open_p,
            close_panel=close_p,
            contract_open_panel={},
            contract_close_panel={},
            contract_map={},
            specs=specs,
            allow_short=False,
        )
    else:
        slot, _ = _build_weekly_calendar_slot(
            tmp_path,
            cal=cal,
            instruments=["A", "B"],
            open_panel=open_p,
            close_panel=close_p,
            use_open=True,
            session_at_strict=True,
            specs=specs,
            allow_short=False,
        )

    if branch_mode == "direct":
        w_df = pd.DataFrame(
            {"A": [np.nan, 0.5, np.nan, np.nan], "B": [np.nan, 0.5, np.nan, np.nan]},
            index=dates,
        )
        cfg = FuturesExecutionConfig(
            execution_mode="futures_contract",
            target_weight_basis="margin",
            margin_rate=0.2,
            margin_rates={"A": 0.1, "B": 0.2},
            leverage=1.0,
        )
        pf, _ = simulate_futures_contract_portfolio(
            target_weights=w_df,
            price_aligned=pd.DataFrame(close_p, index=dates),
            open_aligned=pd.DataFrame(open_p, index=dates),
            use_open=True,
            slot=slot,
            frequency=frequency,
            init_cash=100_000.0,
            buy_cost_rate=0.0,
            sell_cost_rate=0.0,
            slippage=0.0,
            config=cfg,
            allow_short=False,
            target_value_mode="target_weight",
            instrument_specs=specs,
        )
    else:
        top_k_cfg: Dict[str, Any] = {
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
        if branch_mode == "branch_overlay":
            top_k_cfg["partial_rebalance"] = {"enabled": True, "trade_rate": 1.0, "min_weight": 0.01}
        runner = VectorBTBacktestRunner({"top_k_strategy": top_k_cfg}, slot=slot, frequency=frequency)
        scores = pd.DataFrame({"A": [2.0] * 4, "B": [1.0] * 4}, index=dates)
        res = runner.run_backtest(
            scores,
            pd.DataFrame(close_p, index=dates),
            open_df=pd.DataFrame(open_p, index=dates),
            rebalance_dates=[dates[1].to_pydatetime()],
        )
        runner.close()
        assert "error" not in res, res
        pf = res["portfolio"]
        if branch_mode == "branch_overlay":
            assert runner._last_target_weights.loc[dates[1], "A"] == pytest.approx(500_000.0)
            assert runner._last_target_weights.loc[dates[1], "B"] == pytest.approx(250_000.0)

    assert pf.lots_history.loc[dates[1], "A"] == 500
    assert pf.lots_history.loc[dates[1], "B"] == 250
    assert pf.margin_used.loc[dates[1]] == pytest.approx(100_000.0)
    assert pf.margin_utilization.loc[dates[1]] == pytest.approx(1.0)


@pytest.mark.parametrize("frequency", ["1d", "1w"])
@pytest.mark.parametrize("use_open", [True, False], ids=["open", "close"])
def test_release_gate_execution_causality(
    tmp_path: Path,
    frequency: str,
    use_open: bool,
) -> None:
    """Release Gate: post-execution prices, microstructure, signals & rolls never alter earlier orders/fills."""
    if frequency == "1d":
        cal = SessionCalendar("XSHG", "2024-01-08", "2024-01-12")
        grid = cal.bar_grid("1d")
        dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
        # 5 daily bars
        base_open = {"A": [100.0, 101.0, 102.0, 103.0, 104.0], "B": [100.0, 102.0, 104.0, 106.0, 108.0]}
        base_close = {"A": [100.5, 101.5, 102.5, 103.5, 104.5], "B": [101.0, 103.0, 105.0, 107.0, 109.0]}
        base_scores = pd.DataFrame(
            {"A": [2.0, 2.0, 1.0, 1.0, 1.0], "B": [1.0, 1.0, 2.0, 2.0, 2.0]},
            index=dates,
        )
        # Future perturbations strictly after bar 1 execution:
        # - For use_open=True: bar 1 close changes, plus bars 2..4 prices, signals, and mid-day halt at bar 1 12:00 CST (04:00Z)
        # - For use_open=False: bars 2..4 prices, signals, and halt at bar 2 open
        fut_open = {
            "A": [100.0, 101.0, 888.0, 777.0, 666.0],
            "B": [100.0, 102.0, 999.0, 888.0, 777.0],
        }
        fut_close = {
            "A": [100.5, 250.0 if use_open else 101.5, 889.0, 778.0, 667.0],
            "B": [101.0, 260.0 if use_open else 103.0, 998.0, 887.0, 776.0],
        }
        fut_scores = pd.DataFrame(
            {"A": [2.0, 2.0, 99.0, 99.0, 99.0], "B": [1.0, 1.0, -99.0, -99.0, -99.0]},
            index=dates,
        )
        halt_ts = "2024-01-09T04:00:00Z" if use_open else "2024-01-10T01:30:00Z"
        restore_ts = "2024-01-11T01:30:00Z"
        rebal_bars = [dates[1].to_pydatetime(), dates[3].to_pydatetime()]
        cutoff_pos = 1
    else:
        cal = SessionCalendar("XSHG", "2024-01-08", "2024-01-26")
        grid = cal.bar_grid("1w")
        dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
        assert len(dates) == 3
        # 3 weekly bars: W1 (Jan 08..12), W2 (Jan 15..19), W3 (Jan 22..26)
        base_open = {"A": [100.0, 104.0, 108.0], "B": [100.0, 102.0, 104.0]}
        base_close = {"A": [102.0, 106.0, 110.0], "B": [101.0, 103.0, 105.0]}
        base_scores = pd.DataFrame({"A": [2.0, 2.0, 1.0], "B": [1.0, 1.0, 2.0]}, index=dates)
        # Future perturbations strictly after W2 (bar 1) execution:
        # For use_open=True, W2 executes Mon 2024-01-15 01:30Z:
        # mutate Tue..Fri microstructure halt (2024-01-16T01:30:00Z), W2 close, and all of W3!
        # For use_open=False, W2 executes Fri 2024-01-19 07:00Z:
        # mutate W3 microstructure halt (2024-01-22T01:30:00Z) and all of W3!
        fut_open = {"A": [100.0, 104.0, 777.0], "B": [100.0, 102.0, 888.0]}
        fut_close = {
            "A": [102.0, 250.0 if use_open else 106.0, 779.0],
            "B": [101.0, 260.0 if use_open else 103.0, 889.0],
        }
        fut_scores = pd.DataFrame({"A": [2.0, 2.0, -50.0], "B": [1.0, 1.0, 50.0]}, index=dates)
        halt_ts = "2024-01-16T01:30:00Z" if use_open else "2024-01-22T01:30:00Z"
        restore_ts = "2024-01-24T01:30:00Z"
        rebal_bars = [dates[1].to_pydatetime(), dates[2].to_pydatetime()]
        cutoff_pos = 1

    def _run_case(
        open_p: Dict[str, List[float]],
        close_p: Dict[str, List[float]],
        scores_df: pd.DataFrame,
        halts: Optional[List[Tuple[str, str, str]]],
        extra_contract_future: Optional[Dict[Tuple[int, str], str]] = None,
    ):
        specs = {
            k: InstrumentSpec(k, "futures", "CNY", lot_size=1, contract_multiplier=10.0)
            for k in ("A", "B", "A_FUT")
        }
        if frequency == "1d":
            # Build daily slot backed by SessionCalendar + DeclarativeMicrostructure
            slot, _ = _build_weekly_calendar_slot(
                tmp_path,
                cal=cal,
                instruments=["A", "B"],
                open_panel=open_p,
                close_panel=close_p,
                use_open=use_open,
                halt_rows=halts,
                session_at_strict=True,
                contract_map=extra_contract_future,
                specs=specs,
            )
            # Replace BarFeed with 1d session_ids
            sids_1d = grid["session_id"].astype(str).tolist()

            class _DailyFeed:
                def read_signal_bars(self, *a, **kw):
                    return pd.DataFrame()

                def read_execution_bars(self, req_insts, freq, *, start_utc, end_utc):
                    rows = []
                    for pos, (utc_ts, sid) in enumerate(zip(dates, sids_1d)):
                        for inst in req_insts:
                            if inst not in open_p:
                                continue
                            cid = (extra_contract_future or {}).get((pos, inst), inst)
                            rows.append(
                                {
                                    "timestamp_utc": utc_ts,
                                    "session_id": sid,
                                    "instrument_id": inst,
                                    "concrete_instrument_id": cid,
                                    "open": open_p[inst][pos],
                                    "close": close_p[inst][pos],
                                    "volume": 1_000_000.0,
                                    "money": 1_000_000.0 * close_p[inst][pos],
                                }
                            )
                    return pd.DataFrame(rows)

            slot = replace(slot, bar_feed=_DailyFeed())
        else:
            slot, _ = _build_weekly_calendar_slot(
                tmp_path,
                cal=cal,
                instruments=["A", "B"],
                open_panel=open_p,
                close_panel=close_p,
                use_open=use_open,
                halt_rows=halts,
                session_at_strict=True,
                contract_map=extra_contract_future,
                specs=specs,
            )

        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "score_source": "input_factor",
                    "holding_period": 1,
                    "init_cash": 100_000.0,
                    "execution_mode": "futures_contract",
                    "margin_rate": 0.5,
                    "allow_short": False,
                }
            },
            slot=slot,
            frequency=frequency,
        )
        res = runner.run_backtest(
            scores_df,
            pd.DataFrame(close_p, index=dates),
            open_df=pd.DataFrame(open_p, index=dates) if use_open else None,
            rebalance_dates=rebal_bars,
        )
        runner.close()
        assert "error" not in res, res
        return res["portfolio"]

    pf_base = _run_case(base_open, base_close, base_scores, None, None)
    pf_mutated = _run_case(
        fut_open,
        fut_close,
        fut_scores,
        [("A", halt_ts, "false"), ("A", restore_ts, "true")],
        {(len(dates) - 1, "A"): "A_FUT"},
    )

    cutoff_ts = dates[cutoff_pos]
    orders_base = pf_base.orders.records_readable
    orders_mut = pf_mutated.orders.records_readable
    ord_base_prefix = orders_base.loc[pd.to_datetime(orders_base["Timestamp"]) <= cutoff_ts].reset_index(drop=True)
    ord_mut_prefix = orders_mut.loc[pd.to_datetime(orders_mut["Timestamp"]) <= cutoff_ts].reset_index(drop=True)
    assert not ord_base_prefix.empty
    pd.testing.assert_frame_equal(ord_base_prefix, ord_mut_prefix)

    # Lots through cutoff bar must be identical.
    pd.testing.assert_frame_equal(
        pf_base.lots_history.iloc[: cutoff_pos + 1],
        pf_mutated.lots_history.iloc[: cutoff_pos + 1],
    )
    if not use_open:
        # Under close execution, even end-of-bar equity & margin at cutoff are identical.
        pd.testing.assert_series_equal(
            pf_base.value().iloc[: cutoff_pos + 1],
            pf_mutated.value().iloc[: cutoff_pos + 1],
        )
    else:
        # Under open execution, all bars strictly before cutoff_pos have identical equity & margin.
        pd.testing.assert_series_equal(
            pf_base.value().iloc[:cutoff_pos],
            pf_mutated.value().iloc[:cutoff_pos],
        )


@pytest.mark.parametrize("omit_concrete_col", [True, False], ids=["fallback_resolve", "override_bar_end_col"])
def test_release_gate_weekly_open_midweek_roll_prefix_causality(
    tmp_path: Path,
    omit_concrete_col: bool,
) -> None:
    """Mid-week roll (2024-01-10 A_OLD -> A_NEW) must never alter Monday 2024-01-08 open contract or price."""
    cal = SessionCalendar("XSHG", "2024-01-08", "2024-01-19")
    grid = cal.bar_grid("1w")
    dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
    assert len(dates) == 2

    specs = {
        k: InstrumentSpec(k, "futures", "CNY", lot_size=1, contract_multiplier=10.0)
        for k in ("A", "A_OLD", "A_NEW")
    }
    c_open = {"A_OLD": [100.0, 102.0], "A_NEW": [200.0, 204.0]}
    c_close = {"A_OLD": [101.0, 103.0], "A_NEW": [202.0, 206.0]}

    def _run_weekly_open(roll_at_utc: Optional[pd.Timestamp]):
        class _MidweekRollUniverse:
            def resolve_universe(self, universe_id: str, *, as_of_utc: Any) -> Tuple[str, ...]:
                del universe_id, as_of_utc
                return ("A",)

            def resolve_active_contract(self, instrument_id: str, *, as_of_utc: Any) -> str:
                if instrument_id != "A":
                    return instrument_id
                ts = pd.Timestamp(as_of_utc)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                if roll_at_utc is not None and ts >= roll_at_utc:
                    return "A_NEW"
                return "A_OLD"

            def instrument_spec(self, contract_id: str) -> InstrumentSpec:
                return specs[contract_id]

        class _MidweekRollFeed:
            def read_signal_bars(self, *a, **kw):
                return pd.DataFrame()

            def read_execution_bars(self, req_insts, freq, *, start_utc, end_utc):
                del freq
                start_ts = pd.Timestamp(start_utc)
                end_ts = pd.Timestamp(end_utc)
                start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
                end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
                rows: List[Dict[str, Any]] = []
                for pos, (utc_ts, sid) in enumerate(zip(dates, grid["session_id"].astype(str))):
                    if utc_ts < start_ts or utc_ts > end_ts:
                        continue
                    for inst in req_insts:
                        if inst == "A":
                            # Continuous bar keyed at Friday close utc_ts: reflects A_NEW if rolled by Friday!
                            end_cid = "A_NEW" if (roll_at_utc is not None and utc_ts >= roll_at_utc) else "A_OLD"
                            row: Dict[str, Any] = {
                                "timestamp_utc": utc_ts,
                                "session_id": sid,
                                "instrument_id": "A",
                                "open": c_open[end_cid][pos],
                                "close": c_close[end_cid][pos],
                                "volume": 1_000_000.0,
                            }
                            if not omit_concrete_col:
                                row["concrete_instrument_id"] = end_cid
                            rows.append(row)
                        elif inst in c_open:
                            rows.append(
                                {
                                    "timestamp_utc": utc_ts,
                                    "session_id": sid,
                                    "instrument_id": inst,
                                    "concrete_instrument_id": inst,
                                    "open": c_open[inst][pos],
                                    "close": c_close[inst][pos],
                                    "volume": 1_000_000.0,
                                }
                            )
                return pd.DataFrame(rows)

        base_slot, _ = _build_weekly_calendar_slot(
            tmp_path,
            cal=cal,
            instruments=["A"],
            open_panel={"A": [100.0, 102.0]},
            close_panel={"A": [101.0, 103.0]},
            use_open=True,
            session_at_strict=True,
            specs=specs,
        )
        slot = replace(base_slot, universe=_MidweekRollUniverse(), bar_feed=_MidweekRollFeed())
        cfg = FuturesExecutionConfig(
            execution_mode="futures_contract",
            margin_rate=1.0,
            leverage=1.0,
        )
        return simulate_futures_contract_portfolio(
            target_weights=pd.DataFrame({"A": [1.0, 1.0]}, index=dates),
            price_aligned=pd.DataFrame({"A": [101.0, 103.0]}, index=dates),
            open_aligned=pd.DataFrame({"A": [100.0, 102.0]}, index=dates),
            use_open=True,
            slot=slot,
            frequency="1w",
            init_cash=100_000.0,
            buy_cost_rate=0.0,
            sell_cost_rate=0.0,
            slippage=0.0,
            config=cfg,
            target_value_mode="target_weight",
            instrument_specs=specs,
        )

    pf_no_roll, diag_no_roll = _run_weekly_open(None)
    pf_midweek_roll, diag_midweek_roll = _run_weekly_open(pd.Timestamp("2024-01-10 01:30:00+00:00"))

    w1_ts = dates[0]
    orders_base_w1 = pf_no_roll.orders.records_readable.loc[
        pd.to_datetime(pf_no_roll.orders.records_readable["Timestamp"]) == w1_ts
    ].reset_index(drop=True)
    orders_roll_w1 = pf_midweek_roll.orders.records_readable.loc[
        pd.to_datetime(pf_midweek_roll.orders.records_readable["Timestamp"]) == w1_ts
    ].reset_index(drop=True)

    # Week 1 Monday open MUST buy 100 lots of A_OLD @ 100.0 in BOTH runs!
    pd.testing.assert_frame_equal(orders_base_w1, orders_roll_w1)
    assert list(orders_roll_w1["Concrete Instrument"]) == ["A_OLD"]
    assert list(orders_roll_w1["Lots"]) == [100]
    assert float(orders_roll_w1["Price"].iloc[0]) == pytest.approx(100.0)
    assert pf_midweek_roll.lots_history.loc[w1_ts, "A"] == 100
    assert pf_midweek_roll.value().loc[w1_ts] == pytest.approx(pf_no_roll.value().loc[w1_ts])
    assert diag_no_roll["roll_count"] == 0
    # Week 2 Monday open sees A_NEW and executes ROLL_CLOSE(A_OLD) -> ROLL_OPEN(A_NEW).
    assert diag_midweek_roll["roll_count"] == 1
    assert diag_midweek_roll["roll_events"][0]["from_contract"] == "A_OLD"
    assert diag_midweek_roll["roll_events"][0]["to_contract"] == "A_NEW"


@pytest.mark.parametrize(
    "scenario",
    [
        "missing_physical_bar",
        "missing_endpoint_binding_raise",
        "missing_endpoint_binding_none",
    ],
)
def test_release_gate_weekly_open_midweek_roll_missing_endpoint_data(
    tmp_path: Path,
    scenario: str,
) -> None:
    """Missing endpoint physical bar or contract binding must trigger missing-data policy, not Friday roll price/contract."""
    cal = SessionCalendar("XSHG", "2024-01-08", "2024-01-19")
    grid = cal.bar_grid("1w")
    dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
    assert len(dates) == 2

    specs = {
        k: InstrumentSpec(k, "futures", "CNY", lot_size=1, contract_multiplier=10.0)
        for k in ("A", "A_OLD", "A_NEW")
    }
    roll_at_utc = pd.Timestamp("2024-01-10 01:30:00+00:00")
    c_open = {"A_OLD": [100.0, 102.0], "A_NEW": [200.0, 204.0]}
    c_close = {"A_OLD": [101.0, 103.0], "A_NEW": [202.0, 206.0]}

    class _MissingEndpointUniverse:
        def resolve_universe(self, universe_id: str, *, as_of_utc: Any) -> Tuple[str, ...]:
            del universe_id, as_of_utc
            return ("A",)

        def resolve_active_contract(self, instrument_id: str, *, as_of_utc: Any) -> Optional[str]:
            if instrument_id != "A":
                return instrument_id
            ts = pd.Timestamp(as_of_utc)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            if ts >= roll_at_utc:
                return "A_NEW"
            if scenario == "missing_endpoint_binding_raise":
                raise LookupError("missing active contract binding at Monday open")
            if scenario == "missing_endpoint_binding_none":
                return None
            return "A_OLD"

        def instrument_spec(self, contract_id: str) -> InstrumentSpec:
            return specs[contract_id]

    class _MissingEndpointFeed:
        def read_signal_bars(self, *a, **kw):
            return pd.DataFrame()

        def read_execution_bars(self, req_insts, freq, *, start_utc, end_utc):
            del freq
            start_ts = pd.Timestamp(start_utc)
            end_ts = pd.Timestamp(end_utc)
            start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
            end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
            rows: List[Dict[str, Any]] = []
            for pos, (utc_ts, sid) in enumerate(zip(dates, grid["session_id"].astype(str))):
                if utc_ts < start_ts or utc_ts > end_ts:
                    continue
                for inst in req_insts:
                    if inst == "A":
                        # Friday-keyed continuous bar sees A_NEW (200.0) because roll happened on 2024-01-10.
                        row: Dict[str, Any] = {
                            "timestamp_utc": utc_ts,
                            "session_id": sid,
                            "instrument_id": "A",
                            "open": c_open["A_NEW"][pos],
                            "close": c_close["A_NEW"][pos],
                            "volume": 1_000_000.0,
                        }
                        if scenario != "missing_physical_bar":
                            row["concrete_instrument_id"] = "A_NEW"
                        rows.append(row)
                    elif inst == "A_NEW":
                        rows.append(
                            {
                                "timestamp_utc": utc_ts,
                                "session_id": sid,
                                "instrument_id": inst,
                                "concrete_instrument_id": inst,
                                "open": c_open[inst][pos],
                                "close": c_close[inst][pos],
                                "volume": 1_000_000.0,
                            }
                        )
                    elif inst == "A_OLD" and scenario != "missing_physical_bar":
                        rows.append(
                            {
                                "timestamp_utc": utc_ts,
                                "session_id": sid,
                                "instrument_id": inst,
                                "concrete_instrument_id": inst,
                                "open": c_open[inst][pos],
                                "close": c_close[inst][pos],
                                "volume": 1_000_000.0,
                            }
                        )
            return pd.DataFrame(rows)

    base_slot, _ = _build_weekly_calendar_slot(
        tmp_path,
        cal=cal,
        instruments=["A"],
        open_panel={"A": [100.0, 102.0]},
        close_panel={"A": [101.0, 103.0]},
        use_open=True,
        session_at_strict=True,
        specs=specs,
    )
    slot = replace(base_slot, universe=_MissingEndpointUniverse(), bar_feed=_MissingEndpointFeed())

    def _run_with_policy(policy: str):
        cfg = FuturesExecutionConfig(
            execution_mode="futures_contract",
            margin_rate=1.0,
            leverage=1.0,
            missing_execution_bar_policy=policy,
        )
        return simulate_futures_contract_portfolio(
            target_weights=pd.DataFrame({"A": [1.0, 1.0]}, index=dates),
            price_aligned=pd.DataFrame({"A": [202.0, 206.0]}, index=dates),
            open_aligned=pd.DataFrame({"A": [200.0, 204.0]}, index=dates),
            use_open=True,
            slot=slot,
            frequency="1w",
            init_cash=100_000.0,
            buy_cost_rate=0.0,
            sell_cost_rate=0.0,
            slippage=0.0,
            config=cfg,
            target_value_mode="target_weight",
            instrument_specs=specs,
        )

    pf_hold, diag_hold = _run_with_policy("hold")
    w1_ts = dates[0]
    orders_w1 = pf_hold.orders.records_readable.loc[
        pd.to_datetime(pf_hold.orders.records_readable["Timestamp"]) == w1_ts
    ]
    assert orders_w1.empty
    assert pf_hold.lots_history.loc[w1_ts, "A"] == 0
    assert diag_hold["missing_execution_bar_count"] == 1

    with pytest.raises(FuturesExecutionError):
        _run_with_policy("raise")


@pytest.mark.parametrize("frequency", ["1d", "1w"])
@pytest.mark.parametrize("use_open", [True, False], ids=["open", "close"])
def test_release_gate_position_accounting_invariants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frequency: str,
    use_open: bool,
) -> None:
    """Release Gate: bar-by-bar equity, PnL/fees conservation, margin_used, asset_values & prev_weight_map."""
    if frequency == "1d":
        cal = SessionCalendar("XSHG", "2024-01-08", "2024-01-16")
        grid = cal.bar_grid("1d")
        dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
        # 7 bars:
        # Bar 0: flat (no T-1 signal)
        # Bar 1: enter A_C1 (mr=0.1) & B_C1 (mr=0.2)
        # Bar 2: A rolls to A_C2, A_C1 deferred (blocked on bar 2) -> A_C1 valued at A_C1 price!
        # Bar 3: A_C1 unblocks -> rolls into A_C2
        # Bar 4: rebalance replaces B with C (mr=0.25)
        # Bar 5..6: hold A_C2 & C_C1 with price moves
        assert len(dates) == 7
        open_p = {
            "A": [100.0, 100.0, 200.0, 204.0, 206.0, 208.0, 210.0],
            "B": [100.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
            "C": [50.0, 50.0, 51.0, 52.0, 54.0, 55.0, 56.0],
        }
        close_p = {
            "A": [100.0, 102.0, 202.0, 205.0, 207.0, 209.0, 212.0],
            "B": [100.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0],
            "C": [50.0, 50.0, 51.5, 53.0, 54.5, 55.5, 57.0],
        }
        c_open = {
            "A_C1": [100.0, 100.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "A_C2": [198.0, 199.0, 200.0, 204.0, 206.0, 208.0, 210.0],
            "B_C1": open_p["B"],
            "C_C1": open_p["C"],
        }
        c_close = {
            "A_C1": [100.0, 102.0, 103.5, 104.5, 105.5, 106.5, 107.5],
            "A_C2": [199.0, 201.0, 202.0, 205.0, 207.0, 209.0, 212.0],
            "B_C1": close_p["B"],
            "C_C1": close_p["C"],
        }
        halt_ts = "2024-01-10T01:30:00Z" if use_open else "2024-01-10T07:00:00Z"
        restore_ts = "2024-01-11T01:30:00Z"
        n = 7
    else:
        cal = SessionCalendar("XSHG", "2024-01-08", "2024-03-01")
        grid = cal.bar_grid("1w")
        dates = pd.DatetimeIndex(grid["timestamp_utc"]).tz_convert("UTC").as_unit("ns")
        assert len(dates) == 7
        open_p = {
            "A": [100.0, 100.0, 200.0, 204.0, 206.0, 208.0, 210.0],
            "B": [100.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
            "C": [50.0, 50.0, 51.0, 52.0, 54.0, 55.0, 56.0],
        }
        close_p = {
            "A": [100.0, 102.0, 202.0, 205.0, 207.0, 209.0, 212.0],
            "B": [100.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0],
            "C": [50.0, 50.0, 51.5, 53.0, 54.5, 55.5, 57.0],
        }
        c_open = {
            "A_C1": [100.0, 100.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "A_C2": [198.0, 199.0, 200.0, 204.0, 206.0, 208.0, 210.0],
            "B_C1": open_p["B"],
            "C_C1": open_p["C"],
        }
        c_close = {
            "A_C1": [100.0, 102.0, 103.5, 104.5, 105.5, 106.5, 107.5],
            "A_C2": [199.0, 201.0, 202.0, 205.0, 207.0, 209.0, 212.0],
            "B_C1": close_p["B"],
            "C_C1": close_p["C"],
        }
        # Bar 2 is week 2024-01-22..26
        halt_ts = "2024-01-22T01:30:00Z" if use_open else "2024-01-26T07:00:00Z"
        restore_ts = "2024-01-29T01:30:00Z"
        n = 7

    contract_map = {
        **{(i, "A"): ("A_C1" if i < 2 else "A_C2") for i in range(n)},
        **{(i, "B"): "B_C1" for i in range(n)},
        **{(i, "C"): "C_C1" for i in range(n)},
    }
    specs = {
        "A": InstrumentSpec("A", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        "B": InstrumentSpec("B", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        "C": InstrumentSpec("C", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        "A_C1": InstrumentSpec("A_C1", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        "A_C2": InstrumentSpec("A_C2", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        "B_C1": InstrumentSpec("B_C1", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
        "C_C1": InstrumentSpec("C_C1", "futures", "CNY", lot_size=1, contract_multiplier=10.0),
    }
    halt_rows = [("A_C1", halt_ts, "false"), ("A_C1", restore_ts, "true")]

    slot, _ = _build_weekly_calendar_slot(
        tmp_path,
        cal=cal,
        instruments=["A", "B", "C"],
        open_panel=open_p,
        close_panel=close_p,
        use_open=use_open,
        halt_rows=halt_rows,
        session_at_strict=True,
        contract_map=contract_map,
        contract_open_panel=c_open,
        contract_close_panel=c_close,
        specs=specs,
        commission_bps=10.0,
        slippage_bps=2.0,
        allow_short=False,
    )
    if frequency == "1d":
        sids = grid["session_id"].astype(str).tolist()

        class _DailyContractFeed:
            def read_signal_bars(self, *a, **kw):
                return pd.DataFrame()

            def read_execution_bars(self, req_insts, freq, *, start_utc, end_utc):
                rows = []
                for pos, (utc_ts, sid) in enumerate(zip(dates, sids)):
                    for inst in req_insts:
                        if inst in open_p:
                            o_px, c_px = open_p[inst][pos], close_p[inst][pos]
                            cid = contract_map[(pos, inst)]
                        elif inst in c_open:
                            o_px, c_px = c_open[inst][pos], c_close[inst][pos]
                            cid = inst
                        else:
                            continue
                        rows.append(
                            {
                                "timestamp_utc": utc_ts,
                                "session_id": sid,
                                "instrument_id": inst,
                                "concrete_instrument_id": cid,
                                "open": o_px,
                                "close": c_px,
                                "volume": 1_000_000.0,
                                "money": 1_000_000.0 * c_px,
                            }
                        )
                return pd.DataFrame(rows)

        class _DailyContractUniverse:
            def resolve_universe(self, universe_id: str, *, as_of_utc: Any) -> Tuple[str, ...]:
                del universe_id, as_of_utc
                return ("A", "B", "C")

            def resolve_active_contract(self, instrument_id: str, *, as_of_utc: Any) -> str:
                ts = pd.Timestamp(as_of_utc)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                if instrument_id == "A":
                    return "A_C2" if ts >= dates[2].normalize() else "A_C1"
                if instrument_id in ("B", "C"):
                    return f"{instrument_id}_C1"
                return instrument_id

            def instrument_spec(self, contract_id: str) -> InstrumentSpec:
                return specs[contract_id]

        slot = replace(slot, universe=_DailyContractUniverse(), bar_feed=_DailyContractFeed())

    margin_rates = {"A": 0.10, "B": 0.20, "C": 0.25}
    scores = pd.DataFrame(
        {
            "A": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "B": [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            "C": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        },
        index=dates,
    )
    init_cash = 100_000.0
    rebalance_bars = [
        dates[1].to_pydatetime(),
        dates[2].to_pydatetime(),
        dates[4].to_pydatetime(),
        dates[5].to_pydatetime(),
    ]
    observed_decisions: Dict[pd.Timestamp, Tuple[Dict[str, float], float]] = {}
    orig_eval = SharedPortfolioPolicy.evaluate_bar

    def _spy_eval(self, *, ctx, date, account, exposure_by_date=None):
        ts_key = pd.Timestamp(date)
        if ctx.norm_bar_fn(date) in ctx.rebalance_set:
            observed_decisions[ts_key] = (dict(account.prev_weight_map()), float(account.nav()))
        return orig_eval(self, ctx=ctx, date=date, account=account, exposure_by_date=exposure_by_date)

    monkeypatch.setattr(SharedPortfolioPolicy, "evaluate_bar", _spy_eval)

    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "holding_period": 1,
                "init_cash": init_cash,
                "execution_mode": "futures_contract",
                "target_weight_basis": "margin",
                "margin_rate": 0.20,
                "margin_rates": margin_rates,
                "leverage": 1.0,
                "roll_sizing": "match_notional",
                "allow_short": False,
                "cost_rate": 0.001,
                "buy_cost_rate": 0.001,
                "sell_cost_rate": 0.001,
                "slippage": 0.0002,
            }
        },
        slot=slot,
        frequency=frequency,
    )
    res = runner.run_backtest(
        scores,
        pd.DataFrame(close_p, index=dates),
        open_df=pd.DataFrame(open_p, index=dates) if use_open else None,
        rebalance_dates=rebalance_bars,
    )
    runner.close()
    assert "error" not in res, res

    pf = res["portfolio"]
    diag = res["futures_execution"]
    orders = pf.orders.records_readable
    trades = pf.trades.records_readable
    equity = pf.value()
    asset_vals = pf.asset_value()
    lots_hist = pf.lots_history
    margin_used = pf.margin_used
    margin_util = pf.margin_utilization

    # 1. Deferred roll occurred on bar 2 and completed on bar 3.
    assert diag["roll_deferred_count"] == 1
    assert diag["roll_count"] == 1
    assert diag["roll_events"][0]["from_contract"] == "A_C1"
    assert diag["roll_events"][0]["to_contract"] == "A_C2"

    # 2. Exact total PnL + fee conservation across all closed and open trades.
    assert equity.iloc[-1] - init_cash == pytest.approx(float(trades["PnL"].sum()), abs=1e-6)
    assert float(orders["Fees"].sum()) == pytest.approx(diag["total_transaction_cost"], abs=1e-8)
    assert float(trades["Fees"].sum()) == pytest.approx(diag["total_transaction_cost"], abs=1e-8)

    # 3. Independent shadow ledger: reconstruct bar-by-bar equity, cash/margin solvency,
    #    margin_used, margin_utilization (with independent equity denominator), and decision prev_weight_map.
    fee_rate = 0.001 + 0.0002
    mult = 10.0
    shadow_positions: Dict[str, Tuple[str, int, float]] = {}  # inst -> (cid, signed_lots, avg_entry_price)
    shadow_realized_pnl = 0.0
    shadow_fees_paid = 0.0
    exec_price_table = c_open if use_open else c_close
    rebal_ts_set = {pd.Timestamp(d) for d in rebalance_bars}

    def _shadow_equity_at(pos_idx: int, px_table: Dict[str, List[float]]) -> float:
        unrealized = sum(
            float(lots) * (px_table[cid][pos_idx] - avg_px) * mult
            for cid, lots, avg_px in shadow_positions.values()
        )
        return init_cash + shadow_realized_pnl + unrealized - shadow_fees_paid

    for pos, dt in enumerate(dates):
        bar_orders = orders.loc[pd.to_datetime(orders["Timestamp"]) == dt]
        # Verify decision-point prev_weight_map & nav BEFORE rebalance orders execute (after roll_close).
        pre_rebal_orders = bar_orders.loc[bar_orders["Order Type"] == "ROLL_CLOSE"]
        for _, row in pre_rebal_orders.iterrows():
            inst = str(row["Column"])
            fill_px = float(row["Price"])
            fill_lots = int(row["Lots"])
            expected_fee = float(fill_lots) * fill_px * mult * fee_rate
            assert float(row["Fees"]) == pytest.approx(expected_fee, abs=1e-8)
            shadow_fees_paid += expected_fee
            cid, cur_lots, avg_px = shadow_positions.pop(inst)
            assert cid == str(row["Concrete Instrument"])
            assert abs(cur_lots) == fill_lots
            shadow_realized_pnl += float(cur_lots) * (fill_px - avg_px) * mult

        if dt in rebal_ts_set:
            assert dt in observed_decisions
            obs_weights, obs_nav = observed_decisions[dt]
            decision_equity = _shadow_equity_at(pos, exec_price_table)
            assert obs_nav == pytest.approx(decision_equity, abs=1e-6)
            expected_weights: Dict[str, float] = {}
            for inst, (cid, lots, _avg_px) in sorted(shadow_positions.items()):
                notional = float(lots) * exec_price_table[cid][pos] * mult
                expected_weights[inst] = (notional * margin_rates[inst]) / decision_equity
            assert obs_weights == pytest.approx(expected_weights, abs=1e-9)
            if pos == 2:
                # Bar 2 deferred roll: A must still be valued at A_C1, NOT A_C2!
                assert "A" in shadow_positions and shadow_positions["A"][0] == "A_C1"

        non_roll_close_orders = bar_orders.loc[bar_orders["Order Type"] != "ROLL_CLOSE"]
        for _, row in non_roll_close_orders.iterrows():
            inst = str(row["Column"])
            cid = str(row["Concrete Instrument"])
            signed_delta = int(row["Signed Lots"])
            fill_px = float(row["Price"])
            expected_fee = float(abs(signed_delta)) * fill_px * mult * fee_rate
            assert float(row["Fees"]) == pytest.approx(expected_fee, abs=1e-8)
            shadow_fees_paid += expected_fee
            if inst not in shadow_positions:
                shadow_positions[inst] = (cid, signed_delta, fill_px)
            else:
                prev_cid, prev_lots, prev_avg = shadow_positions[inst]
                assert prev_cid == cid
                if prev_lots * signed_delta > 0:
                    new_lots = prev_lots + signed_delta
                    new_avg = (prev_avg * abs(prev_lots) + fill_px * abs(signed_delta)) / float(abs(new_lots))
                    shadow_positions[inst] = (cid, new_lots, new_avg)
                else:
                    close_abs = min(abs(prev_lots), abs(signed_delta))
                    sign_held = 1 if prev_lots > 0 else -1
                    shadow_realized_pnl += float(sign_held * close_abs) * (fill_px - prev_avg) * mult
                    rem_lots = prev_lots + signed_delta
                    if rem_lots == 0:
                        shadow_positions.pop(inst)
                    else:
                        shadow_positions[inst] = (cid, rem_lots, prev_avg)

        shadow_equity = _shadow_equity_at(pos, c_close)
        expected_margin = 0.0
        for inst in ("A", "B", "C"):
            lots = int(lots_hist.loc[dt, inst])
            if inst not in shadow_positions:
                assert lots == 0
                assert asset_vals.loc[dt, inst] == pytest.approx(0.0)
                continue
            cid, s_lots, _avg_px = shadow_positions[inst]
            assert lots == s_lots
            if pos == 2 and inst == "A":
                assert cid == "A_C1"  # Must still hold A_C1 during deferred roll!
            close_px = c_close[cid][pos]
            expected_notional = float(s_lots) * close_px * mult
            assert asset_vals.loc[dt, inst] == pytest.approx(expected_notional, abs=1e-8)
            expected_margin += abs(expected_notional) * margin_rates[inst]

        shadow_free_cash = shadow_equity - expected_margin
        assert shadow_free_cash >= -1e-8
        assert equity.loc[dt] == pytest.approx(shadow_equity, abs=1e-6)
        assert margin_used.loc[dt] == pytest.approx(expected_margin, abs=1e-8)
        assert margin_util.loc[dt] == pytest.approx(expected_margin / shadow_equity, abs=1e-10)
        assert margin_util.loc[dt] <= 1.0 + 1e-9
