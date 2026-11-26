"""Cross-frequency ADV/ABV aggregation and liquidity weight-cap unit consistency tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from factor_synthesis.vectorbt_adapter.slot_provider import (
    SlotDataProvider,
    apply_magnitude_caps,
    compute_liquidity_weight_caps,
)
from slots.declarative.slot import build_declarative_slot
from utils.slots import SlotConfig
from utils.time_axis import Frequency


def _build_declarative_market_slot(
    tmp_path: Path,
    *,
    frequency: str,
    daily_traded_values: Mapping[str, Mapping[str, float]],
    calendar_id: str = "crypto_utc",
    start: str = "2026-01-05",
    end: str = "2026-01-08",
    use_money_column: bool = True,
    price: float = 100.0,
    intraday_profile: str = "nonuniform",
    future_spike_date: str | None = None,
    custom_sessions_spec: Mapping[str, Any] | None = None,
    bar_value_overrides: Mapping[tuple[str, str], float] | None = None,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Build a real declarative Slot at ``1d``/``1h``/``30min``/``5min`` preserving exact session totals."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    is_custom = calendar_id == "crypto_utc" or custom_sessions_spec is not None
    default_crypto_sessions = {
        "timezone": "UTC",
        "weekdays": [0, 1, 2, 3, 4, 5, 6],
        "segments": [
            {
                "open": "00:00",
                "close": "00:00",
                "open_day_offset": 0,
                "close_day_offset": 1,
            }
        ],
    }
    sessions_payload = custom_sessions_spec or default_crypto_sessions
    cal_cfg = (
        {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": calendar_id,
                "start": start,
                "end": end,
                "custom_sessions": sessions_payload,
            },
        }
        if is_custom
        else {
            "provider": "pmc",
            "options": {"calendar_id": calendar_id, "start": start, "end": end},
        }
    )

    from utils.session_calendar import SessionCalendar
    from utils.time_axis import BarSchedule

    if is_custom:
        cal = SessionCalendar(
            calendar_id,
            start,
            end,
            custom_sessions=sessions_payload,
        )
    else:
        cal = SessionCalendar(calendar_id, start, end)

    freq_obj = Frequency.parse(frequency)
    schedule = BarSchedule(cal, freq_obj)
    grid = cal.bar_grid(frequency, start, end)
    symbols = sorted(daily_traded_values.keys())
    overrides_utc = {
        (sym, pd.Timestamp(ts_str).tz_convert("UTC") if pd.Timestamp(ts_str).tzinfo else pd.Timestamp(ts_str, tz="UTC")): float(val)
        for (sym, ts_str), val in (bar_value_overrides or {}).items()
    }

    rows = []
    for sym in symbols:
        sym_days = daily_traded_values[sym]
        for session_id, sess_grid in grid.groupby("session_id", sort=True):
            total_val = float(sym_days[str(session_id)])
            n_bars = len(sess_grid)
            if n_bars == 1 or intraday_profile == "uniform":
                weights = np.full(n_bars, 1.0 / n_bars, dtype=float)
            else:
                raw_w = np.arange(1, n_bars + 1, dtype=float)
                weights = raw_w / raw_w.sum()

            bar_starts = pd.to_datetime(sess_grid["timestamp_utc"], utc=True)
            if freq_obj.kind == "intraday":
                bar_ends = pd.to_datetime(schedule.for_session(str(session_id))["end_utc"], utc=True)
            else:
                bar_ends = bar_starts
            if future_spike_date is not None and str(session_id) == future_spike_date and n_bars > 1:
                # Place 90% of the day's value in the final bar to catch any look-ahead leak.
                weights = np.full(n_bars, 0.1 / (n_bars - 1), dtype=float)
                weights[-1] = 0.9

            for idx_bar, (bar_start, bar_end) in enumerate(
                zip(bar_starts, bar_ends)
            ):
                bar_val = overrides_utc.get(
                    (sym, pd.Timestamp(bar_start).tz_convert("UTC")),
                    total_val * float(weights[idx_bar]),
                )
                row: dict[str, Any] = {
                    "symbol": sym,
                    "time": pd.Timestamp(bar_end).isoformat(),
                    "open": float(price),
                    "high": float(price) * 1.01,
                    "low": float(price) * 0.99,
                    "close": float(price),
                    "volume": float(bar_val / price),
                }
                if use_money_column:
                    row["money"] = float(bar_val)
                rows.append(row)

    bars_csv = tmp_path / f"bars_{freq_obj.canonical}.csv"
    pd.DataFrame(rows).to_csv(bars_csv, index=False)

    col_map = {
        "symbol": "instrument_id",
        "time": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    if use_money_column:
        col_map["money"] = "money"

    slot_cfg = {
        "schema_version": 1,
        "slot_id": f"probe_{calendar_id}_{freq_obj.canonical}",
        "calendar": cal_cfg,
        "bar_feed": {
            "provider": "csv_files",
            "options": {
                "paths": [bars_csv.name],
                "frequency": freq_obj.canonical,
                "timestamp_anchor": "bar_end",
                "source_timezone": "UTC",
                "volume_mode": "interval",
                "data_version": "v1",
                "column_mapping": col_map,
            },
        },
        "universe": {
            "provider": "static",
            "options": {
                "instruments": {
                    sym: {
                        "asset_class": "crypto" if is_custom else "equity",
                        "currency": "USD" if is_custom else "CNY",
                    }
                    for sym in symbols
                },
                "universes": {"spot": symbols},
            },
        },
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {
            "entry_price_field": "open",
            "exit_price_field": "open",
            "min_entry_offset_bars": 1,
            "exit_adjustment": 0,
            "return_unit": "decimal",
            "gap_rule": "endpoints_only",
            "boundary_policy": "cross_session",
            "allow_short": False,
            "commission_bps": 0.0,
            "sell_tax_bps": 0.0,
            "slippage_bps": 0.0,
            "benchmark_instrument": None,
        },
    }
    slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
    bars = slot.bar_feed.read_execution_bars(
        symbols,
        freq_obj,
        start_utc=pd.Timestamp(grid["timestamp_utc"].min()).to_pydatetime(),
        end_utc=pd.Timestamp(grid["timestamp_utc"].max()).to_pydatetime(),
    )
    close_df = bars.pivot(index="timestamp_utc", columns="instrument_id", values="close").sort_index()
    open_df = bars.pivot(index="timestamp_utc", columns="instrument_id", values="open").sort_index()
    return slot, close_df, open_df


@pytest.mark.parametrize("use_money_column", [True, False])
def test_adv_session_aggregation_parity_across_1d_1h_5min(
    tmp_path: Path, use_money_column: bool
) -> None:
    """Identical daily totals split into 1d/1h/5min must yield identical ADV for both money and volume*close."""
    daily_totals = {
        "A": {
            "2026-01-05": 1_200_000.0,
            "2026-01-06": 2_400_000.0,
            "2026-01-07": 3_600_000.0,
            "2026-01-08": 4_800_000.0,
        },
        "B": {
            "2026-01-05": 600_000.0,
            "2026-01-06": 1_200_000.0,
            "2026-01-07": 1_800_000.0,
            "2026-01-08": 2_400_000.0,
        },
    }
    expected_adv_3d = {
        "A": (2_400_000.0 + 3_600_000.0 + 4_800_000.0) / 3.0,
        "B": (1_200_000.0 + 1_800_000.0 + 2_400_000.0) / 3.0,
    }

    for freq, as_of in (
        ("1d", "2026-01-08"),
        ("1h", "2026-01-09T00:00:00+00:00"),
        ("5min", "2026-01-09T00:00:00+00:00"),
    ):
        slot, _, _ = _build_declarative_market_slot(
            tmp_path / f"m_{int(use_money_column)}_{freq}",
            frequency=freq,
            daily_traded_values=daily_totals,
            use_money_column=use_money_column,
            intraday_profile="nonuniform",
        )
        provider = SlotDataProvider(slot=slot, frequency=freq)
        adv = provider.average_daily_traded_value(["A", "B"], as_of, 3, None, None)
        adv_compat = provider.average_daily_value(["A", "B"], as_of, 3, None, None)
        assert adv == pytest.approx(expected_adv_3d, rel=1e-9)
        assert adv_compat == pytest.approx(expected_adv_3d, rel=1e-9)

    # Opening timestamps of 2026-01-08's final bar (23:00Z / 23:55Z) precede session close (00:00Z next day)
    # so 2026-01-08 is still incomplete and excluded from completed-session ADV.
    for freq, unclosed_as_of in (
        ("1h", "2026-01-08T23:00:00+00:00"),
        ("5min", "2026-01-08T23:55:00+00:00"),
    ):
        slot_u, _, _ = _build_declarative_market_slot(
            tmp_path / f"u_{int(use_money_column)}_{freq}",
            frequency=freq,
            daily_traded_values=daily_totals,
            use_money_column=use_money_column,
            intraday_profile="nonuniform",
        )
        adv_incomplete = SlotDataProvider(slot=slot_u, frequency=freq).average_daily_traded_value(
            ["A", "B"], unclosed_as_of, 3, None, None
        )
        assert adv_incomplete == pytest.approx(
            {
                "A": (1_200_000.0 + 2_400_000.0 + 3_600_000.0) / 3.0,
                "B": (600_000.0 + 1_200_000.0 + 1_800_000.0) / 3.0,
            },
            rel=1e-9,
        )


def test_adv_causal_cutoff_and_incomplete_session_policy(tmp_path: Path) -> None:
    """Mid-session as_of_utc must never leak future bars and handles incomplete sessions consistently."""
    daily_totals = {
        "A": {
            "2026-01-05": 2_400_000.0,
            "2026-01-06": 4_800_000.0,
            "2026-01-07": 100_000_000.0,
        }
    }
    slot, _, _ = _build_declarative_market_slot(
        tmp_path,
        frequency="1h",
        daily_traded_values=daily_totals,
        start="2026-01-05",
        end="2026-01-07",
        intraday_profile="uniform",
        future_spike_date="2026-01-07",
    )
    provider = SlotDataProvider(slot=slot, frequency="1h")

    # At 2026-01-07T02:00:00Z, 2026-01-07 is incomplete (21:00..23:00 future spike must never appear).
    # Default policy excludes the unfinished session -> trailing 2 completed sessions (2.4M, 4.8M).
    adv_complete_only = provider.average_daily_traded_value(
        ["A"], "2026-01-07T02:00:00+00:00", 2, None, None
    )
    assert adv_complete_only["A"] == pytest.approx((2_400_000.0 + 4_800_000.0) / 2.0)

    # Naive timestamp with time-of-day is treated as UTC (not date-only).
    adv_naive_ts = provider.average_daily_traded_value(
        ["A"], "2026-01-07 02:00:00", 2, None, None
    )
    assert adv_naive_ts["A"] == pytest.approx(adv_complete_only["A"])

    # Explicit include_incomplete_session=True sums ONLY completed bars (available_at <= 02:00Z,
    # i.e. 00:00Z and 01:00Z = 2/23 of 10%), strictly excluding the unclosed 02:00Z bar and 23:00Z 90M spike.
    expected_partial_jan7 = 100_000_000.0 * (0.1 / 23.0) * 2.0
    adv_with_partial = provider.average_daily_traded_value(
        ["A"],
        "2026-01-07T02:00:00+00:00",
        2,
        None,
        None,
        include_incomplete_session=True,
    )
    assert adv_with_partial["A"] == pytest.approx((4_800_000.0 + expected_partial_jan7) / 2.0)


def test_adv_vs_abv_distinction_across_frequencies(tmp_path: Path) -> None:
    """ADV aggregates per-session sums over N days; ABV averages per-bar traded value over N bars."""
    daily_totals = {
        "A": {
            "2026-01-05": 2_400_000.0,  # 100k / 1h bar, 8,333.333... / 5min bar
            "2026-01-06": 2_400_000.0,
            "2026-01-07": 4_800_000.0,  # 200k / 1h bar, 16,666.666... / 5min bar
        }
    }
    slot_1d, _, _ = _build_declarative_market_slot(
        tmp_path / "d1", frequency="1d", daily_traded_values=daily_totals, start="2026-01-05", end="2026-01-07"
    )
    slot_1h, _, _ = _build_declarative_market_slot(
        tmp_path / "h1", frequency="1h", daily_traded_values=daily_totals, start="2026-01-05", end="2026-01-07", intraday_profile="uniform"
    )
    slot_5m, _, _ = _build_declarative_market_slot(
        tmp_path / "m5", frequency="5min", daily_traded_values=daily_totals, start="2026-01-05", end="2026-01-07", intraday_profile="uniform"
    )

    p_1d = SlotDataProvider(slot=slot_1d, frequency="1d")
    p_1h = SlotDataProvider(slot=slot_1h, frequency="1h")
    p_5m = SlotDataProvider(slot=slot_5m, frequency="5min")

    # On 1d, 1 bar == 1 session so ADV(2) == ABV(2).
    assert p_1d.average_daily_traded_value(["A"], "2026-01-07", 2, None, None)["A"] == pytest.approx(3_600_000.0)
    assert p_1d.average_bar_traded_value(["A"], "2026-01-07", 2, None, None)["A"] == pytest.approx(3_600_000.0)

    # On 1h, before 01-07 closes (23:00Z open), 01-07 is incomplete so ADV(2) = (2.4M + 2.4M) / 2 = 2.4M;
    # at 2026-01-08T00:00:00Z (01-07 close), ADV(2) = (2.4M + 4.8M) / 2 = 3.6M, ABV(2) = 200k/bar.
    assert p_1h.average_daily_traded_value(["A"], "2026-01-07T23:00:00+00:00", 2, None, None)["A"] == pytest.approx(2_400_000.0)
    assert p_1h.average_daily_traded_value(["A"], "2026-01-08T00:00:00+00:00", 2, None, None)["A"] == pytest.approx(3_600_000.0)
    assert p_1h.average_bar_traded_value(["A"], "2026-01-08T00:00:00+00:00", 2, None, None)["A"] == pytest.approx(200_000.0)

    # On 5min, before 01-07 closes (23:55Z open), 01-07 is incomplete so ADV(2) = 2.4M;
    # at 2026-01-08T00:00:00Z (01-07 close), ADV(2) = 3.6M, ABV(2) = 4.8M / 288 = 16,666.6667/bar.
    assert p_5m.average_daily_traded_value(["A"], "2026-01-07T23:55:00+00:00", 2, None, None)["A"] == pytest.approx(2_400_000.0)
    assert p_5m.average_daily_traded_value(["A"], "2026-01-08T00:00:00+00:00", 2, None, None)["A"] == pytest.approx(3_600_000.0)
    assert p_5m.average_bar_traded_value(["A"], "2026-01-08T00:00:00+00:00", 2, None, None)["A"] == pytest.approx(4_800_000.0 / 288.0)


def test_compute_liquidity_weight_caps_unit_alignment_and_invalid_inputs() -> None:
    """Verify ADV/ABV unit conversion matrix and fail-closed validation on invalid units/params."""
    adv = {"A": 2_400_000.0}
    abv_1h = {"A": 100_000.0}
    abv_5m = {"A": 2_400_000.0 / 288.0}

    # Equal 1-day holding horizon across 1d (1 bar), 1h (24 bars), 5min (288 bars)
    cap_1d = compute_liquidity_weight_caps(
        adv, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
        holding_period_unit="bars", bars_per_day=1.0, liquidity_basis="adv",
    )
    cap_1h = compute_liquidity_weight_caps(
        adv, ["A"], max_participation=0.1, holding_period=24, capital=10_000_000.0,
        holding_period_unit="bars", bars_per_day=24.0, liquidity_basis="adv",
    )
    cap_5m = compute_liquidity_weight_caps(
        adv, ["A"], max_participation=0.1, holding_period=288, capital=10_000_000.0,
        holding_period_unit="bars", bars_per_day=288.0, liquidity_basis="adv",
    )
    cap_days = compute_liquidity_weight_caps(
        adv, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
        holding_period_unit="days", liquidity_basis="adv",
    )
    assert cap_1d["A"] == pytest.approx(0.024)
    assert cap_1h["A"] == pytest.approx(0.024)
    assert cap_5m["A"] == pytest.approx(0.024)
    assert cap_days["A"] == pytest.approx(0.024)

    # Equal 1-hour holding horizon: ADV with bars vs ABV with bars
    cap_1h_adv = compute_liquidity_weight_caps(
        adv, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
        holding_period_unit="bars", bars_per_day=24.0, liquidity_basis="adv",
    )
    cap_5m_adv = compute_liquidity_weight_caps(
        adv, ["A"], max_participation=0.1, holding_period=12, capital=10_000_000.0,
        holding_period_unit="bars", bars_per_day=288.0, liquidity_basis="adv",
    )
    cap_1h_abv = compute_liquidity_weight_caps(
        abv_1h, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
        holding_period_unit="bars", liquidity_basis="abv",
    )
    cap_5m_abv = compute_liquidity_weight_caps(
        abv_5m, ["A"], max_participation=0.1, holding_period=12, capital=10_000_000.0,
        holding_period_unit="bars", liquidity_basis="abv",
    )
    assert cap_1h_adv["A"] == pytest.approx(0.001)
    assert cap_5m_adv["A"] == pytest.approx(0.001)
    assert cap_1h_abv["A"] == pytest.approx(0.001)
    assert cap_5m_abv["A"] == pytest.approx(0.001)

    # ABV with explicit days unit scales by bars_per_day
    cap_abv_days = compute_liquidity_weight_caps(
        abv_1h, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
        holding_period_unit="days", bars_per_day=24.0, liquidity_basis="abv",
    )
    assert cap_abv_days["A"] == pytest.approx(0.024)

    # Non-positive capital returns empty caps (preserving existing behavior)
    assert compute_liquidity_weight_caps(
        adv, ["A"], max_participation=0.1, holding_period=1, capital=0.0,
        holding_period_unit="days", liquidity_basis="adv",
    ) == {}

    # Negative / fail-closed cases
    with pytest.raises(ValueError, match="bars_per_day is required"):
        compute_liquidity_weight_caps(
            adv, ["A"], max_participation=0.1, holding_period=24, capital=10_000_000.0,
            holding_period_unit="bars", liquidity_basis="adv",
        )
    with pytest.raises(ValueError, match="bars_per_day is required"):
        compute_liquidity_weight_caps(
            abv_1h, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
            holding_period_unit="days", liquidity_basis="abv",
        )
    for bad_unit in ("hours", "minutes", "", None):
        with pytest.raises(ValueError, match="holding_period_unit"):
            compute_liquidity_weight_caps(
                adv, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
                holding_period_unit=bad_unit, bars_per_day=1.0,  # type: ignore[arg-type]
            )
    for bad_basis in ("volume", "turnover", "", None):
        with pytest.raises(ValueError, match="liquidity_basis"):
            compute_liquidity_weight_caps(
                adv, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
                liquidity_basis=bad_basis, holding_period_unit="days",  # type: ignore[arg-type]
            )
    for bad_hp in (0, -5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="holding_period"):
            compute_liquidity_weight_caps(
                adv, ["A"], max_participation=0.1, holding_period=bad_hp, capital=10_000_000.0,
                holding_period_unit="days",
            )
    for bad_bpd in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="bars_per_day"):
            compute_liquidity_weight_caps(
                adv, ["A"], max_participation=0.1, holding_period=1, capital=10_000_000.0,
                holding_period_unit="bars", bars_per_day=bad_bpd,
            )
    with pytest.raises(ValueError, match="max_participation"):
        compute_liquidity_weight_caps(
            adv, ["A"], max_participation=0.0, holding_period=1, capital=10_000_000.0,
            holding_period_unit="days",
        )


def test_runner_liquidity_weight_caps_cross_frequency_consistency(tmp_path: Path) -> None:
    """Runner _liquidity_weight_caps and end-to-end rebalance weights match across 1d, 1h, and 5min."""
    daily_totals = {
        sym: {
            day: val
            for day in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")
        }
        for sym, val in (("A", 1_000_000.0), ("B", 50_000_000.0), ("C", 50_000_000.0))
    }

    caps_by_freq: dict[str, dict[str, float]] = {}
    weights_by_freq: dict[str, dict[str, float]] = {}

    for freq, hp_bars, as_of_query, rebal_ts_str in (
        ("1d", 1, "2026-01-07", "2026-01-08"),
        ("1h", 24, "2026-01-08T00:00:00+00:00", "2026-01-08T00:00:00+00:00"),
        ("5min", 288, "2026-01-08T00:00:00+00:00", "2026-01-08T00:00:00+00:00"),
    ):
        slot, close_df, open_df = _build_declarative_market_slot(
            tmp_path / f"run_{freq}",
            frequency=freq,
            daily_traded_values=daily_totals,
            start="2026-01-05",
            end="2026-01-08",
            intraday_profile="uniform",
        )
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 3,
                    "holding_period": hp_bars,
                    "score_source": "input_factor",
                    "cost_rate": 0.0,
                    "init_cash": 10_000_000.0,
                    "enable_tday_filter": False,
                    "liquidity_cap": {
                        "enabled": True,
                        "adv_lookback_days": 3,
                        "max_participation": 0.1,
                    },
                }
            },
            slot=slot,
            frequency=freq,
        )
        try:
            caps_by_freq[freq] = runner._liquidity_weight_caps(
                ["A", "B", "C"], as_of_query, deployed_capital=10_000_000.0
            )
            scores = pd.DataFrame(
                {"A": 3.0, "B": 2.0, "C": 1.0},
                index=close_df.index,
            )
            if freq == "1d":
                daily_close = close_df.copy()
                daily_close.index = pd.DatetimeIndex(
                    [pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).normalize() for ts in daily_close.index]
                )
                daily_open = open_df.copy()
                daily_open.index = daily_close.index
                daily_scores = scores.copy()
                daily_scores.index = daily_close.index
                rebal_dt = pd.Timestamp(rebal_ts_str).to_pydatetime()
                tw = runner._compute_target_weights(
                    daily_scores, daily_close, {}, [rebal_dt], open_df=daily_open
                )
                row = tw.loc[pd.Timestamp(rebal_ts_str)]
            else:
                rebal_dt = pd.Timestamp(rebal_ts_str).to_pydatetime()
                tw = runner._compute_target_weights(
                    scores, close_df, {}, [rebal_dt], open_df=open_df
                )
                row = tw.loc[pd.Timestamp(rebal_ts_str)]
            weights_by_freq[freq] = {s: float(row[s]) for s in ("A", "B", "C")}
        finally:
            runner.close()

    # Exact cap for A: 0.1 * 1_000_000 * 1 / 10_000_000 = 0.01 across all frequencies
    for freq in ("1d", "1h", "5min"):
        assert caps_by_freq[freq]["A"] == pytest.approx(0.01)
        assert caps_by_freq[freq]["B"] == pytest.approx(0.5)
        assert caps_by_freq[freq]["C"] == pytest.approx(0.5)
        # Equal weight 1/3 (~3.333M) for A is clipped to 1% (100k), excess redistributed to B and C
        assert weights_by_freq[freq]["A"] == pytest.approx(100_000.0)
        assert weights_by_freq[freq]["B"] == pytest.approx(4_950_000.0)
        assert weights_by_freq[freq]["C"] == pytest.approx(4_950_000.0)


def test_runner_xshg_intraday_30min_5min_and_abv_config_and_invalid_cases(tmp_path: Path) -> None:
    """Verify XSHG 1d/30min/5min ADV+cap parity, explicit ABV mode, and invalid config errors."""
    daily_totals = {
        "000001.XSHE": {
            "2026-01-05": 4_800_000.0,
            "2026-01-06": 4_800_000.0,
        },
        "000002.XSHE": {
            "2026-01-05": 9_600_000.0,
            "2026-01-06": 9_600_000.0,
        },
    }
    caps_xshg: dict[str, dict[str, float]] = {}
    for freq, hp_bars, as_of in (
        ("1d", 1, "2026-01-06"),
        ("30min", 8, "2026-01-06T07:00:00+00:00"),
        ("5min", 48, "2026-01-06T07:00:00+00:00"),
    ):
        slot, _, _ = _build_declarative_market_slot(
            tmp_path / f"xshg_{freq}",
            frequency=freq,
            daily_traded_values=daily_totals,
            calendar_id="XSHG",
            start="2026-01-05",
            end="2026-01-06",
            intraday_profile="uniform",
        )
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "holding_period": hp_bars,
                    "liquidity_cap": {
                        "enabled": True,
                        "adv_lookback_days": 2,
                        "max_participation": 0.1,
                    },
                }
            },
            slot=slot,
            frequency=freq,
        )
        try:
            caps_xshg[freq] = runner._liquidity_weight_caps(
                ["000001.XSHE", "000002.XSHE"], as_of, deployed_capital=10_000_000.0
            )
        finally:
            runner.close()

    for freq in ("1d", "30min", "5min"):
        assert caps_xshg[freq]["000001.XSHE"] == pytest.approx(0.048)
        assert caps_xshg[freq]["000002.XSHE"] == pytest.approx(0.096)

    # Explicit ABV mode on 5min (48 bars/day => 1 bar = 100k for 000001.XSHE) with holding_period=6 bars
    slot_5m, _, _ = _build_declarative_market_slot(
        tmp_path / "xshg_abv",
        frequency="5min",
        daily_traded_values=daily_totals,
        calendar_id="XSHG",
        start="2026-01-05",
        end="2026-01-06",
        intraday_profile="uniform",
    )
    runner_abv = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "holding_period": 6,
                "liquidity_cap": {
                    "enabled": True,
                    "liquidity_basis": "abv",
                    "holding_period_unit": "bars",
                    "abv_lookback_bars": 4,
                    "max_participation": 0.1,
                },
            }
        },
        slot=slot_5m,
        frequency="5min",
    )
    try:
        caps_abv = runner_abv._liquidity_weight_caps(
            ["000001.XSHE"], "2026-01-06T07:00:00+00:00", deployed_capital=10_000_000.0
        )
        # 6 / 48 day * 4.8M/day * 0.1 / 10M == 6 * 100k * 0.1 / 10M == 0.006
        assert caps_abv["000001.XSHE"] == pytest.approx(0.006)

        with pytest.raises(ValueError, match="lookback_days must be positive"):
            runner_abv._slot_provider.average_daily_traded_value(
                ["000001.XSHE"], "2026-01-06T07:00:00+00:00", 0, None, None
            )
        with pytest.raises(ValueError, match="lookback_bars must be positive"):
            runner_abv._slot_provider.average_bar_traded_value(
                ["000001.XSHE"], "2026-01-06T07:00:00+00:00", -2, None, None
            )
    finally:
        runner_abv.close()

    # Invalid holding_period_unit / liquidity_basis in runner config fails closed
    for bad_cfg in (
        {"enabled": True, "holding_period_unit": "hours"},
        {"enabled": True, "liquidity_basis": "invalid"},
        {"enabled": True, "liquidity_basis": "abv", "abv_lookback_bars": 0},
    ):
        bad_runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"holding_period": 6, "liquidity_cap": bad_cfg}},
            slot=slot_5m,
            frequency="5min",
        )
        try:
            with pytest.raises(ValueError):
                bad_runner._liquidity_weight_caps(
                    ["000001.XSHE"], "2026-01-06T07:00:00+00:00", deployed_capital=10_000_000.0
                )
        finally:
            bad_runner.close()


def test_weekly_1w_adv_normalization_and_cap_parity_with_1d(tmp_path: Path) -> None:
    """Weekly (1w) bars normalize weekly traded value back to per-session ADV without double-counting horizon."""
    # 1. crypto_utc: 3 full 7-day weeks, 7M per week == 1M per day -> hp=1 bar (7 days) cap = 0.07 (NOT 0.49).
    weekly_crypto = {
        "A": {
            "2026-01-11": 7_000_000.0,
            "2026-01-18": 7_000_000.0,
            "2026-01-25": 7_000_000.0,
        }
    }
    daily_crypto = {
        "A": {
            day.strftime("%Y-%m-%d"): 1_000_000.0
            for day in pd.date_range("2026-01-05", "2026-01-25", freq="D")
        }
    }
    slot_1w, _, _ = _build_declarative_market_slot(
        tmp_path / "crypto_1w",
        frequency="1w",
        daily_traded_values=weekly_crypto,
        calendar_id="crypto_utc",
        start="2026-01-05",
        end="2026-01-25",
    )
    slot_1d, _, _ = _build_declarative_market_slot(
        tmp_path / "crypto_1d",
        frequency="1d",
        daily_traded_values=daily_crypto,
        calendar_id="crypto_utc",
        start="2026-01-05",
        end="2026-01-25",
    )

    provider_1w = SlotDataProvider(slot=slot_1w, frequency="1w")
    assert provider_1w.bars_per_session() == pytest.approx(1.0 / 7.0)
    assert provider_1w.average_daily_traded_value(
        ["A"], "2026-01-26T00:00:00+00:00", 3, None, None
    )["A"] == pytest.approx(1_000_000.0)
    assert provider_1w.average_bar_traded_value(
        ["A"], "2026-01-26T00:00:00+00:00", 3, None, None
    )["A"] == pytest.approx(7_000_000.0)

    runner_1w = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "holding_period": 1,
                "liquidity_cap": {
                    "enabled": True,
                    "adv_lookback_days": 3,
                    "max_participation": 0.1,
                },
            }
        },
        slot=slot_1w,
        frequency="1w",
    )
    runner_1d = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "holding_period": 7,
                "liquidity_cap": {
                    "enabled": True,
                    "adv_lookback_days": 3,
                    "max_participation": 0.1,
                },
            }
        },
        slot=slot_1d,
        frequency="1d",
    )
    try:
        cap_1w = runner_1w._liquidity_weight_caps(
            ["A"], "2026-01-26T00:00:00+00:00", deployed_capital=10_000_000.0
        )
        cap_1d = runner_1d._liquidity_weight_caps(
            ["A"], "2026-01-25", deployed_capital=10_000_000.0
        )
        assert cap_1w["A"] == pytest.approx(0.07)
        assert cap_1w["A"] == pytest.approx(cap_1d["A"])
    finally:
        runner_1w.close()
        runner_1d.close()

    # Cross-mode parity on 1w: adv+days(7), abv+bars(1), abv+days(7) all equal 0.07.
    for hp_val, liq_extra in (
        (7, {"holding_period_unit": "days", "liquidity_basis": "adv"}),
        (1, {"holding_period_unit": "bars", "liquidity_basis": "abv", "abv_lookback_bars": 2}),
        (7, {"holding_period_unit": "days", "liquidity_basis": "abv", "abv_lookback_bars": 2}),
    ):
        r = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "holding_period": hp_val,
                    "liquidity_cap": {
                        "enabled": True,
                        "adv_lookback_days": 3,
                        "max_participation": 0.1,
                        **liq_extra,
                    },
                }
            },
            slot=slot_1w,
            frequency="1w",
        )
        try:
            assert r._liquidity_weight_caps(
                ["A"], "2026-01-26T00:00:00+00:00", deployed_capital=10_000_000.0
            )["A"] == pytest.approx(0.07)
        finally:
            r.close()

    # 2. XSHG: 5-day weeks + partial boundary robustness (starts Wed 2026-01-07 -> partial edge week excluded by 1w grid)
    # 5M per week == 1M per trading day -> 1w(hp=1) matches 1d(hp=5) == 0.05 (NOT 0.25).
    weekly_xshg = {
        "000001.XSHE": {
            "2026-01-09": 5_000_000.0,
            "2026-01-16": 5_000_000.0,
            "2026-01-23": 10_000_000.0,
        }
    }
    slot_xshg_1w, _, _ = _build_declarative_market_slot(
        tmp_path / "xshg_1w",
        frequency="1w",
        daily_traded_values=weekly_xshg,
        calendar_id="XSHG",
        start="2026-01-05",
        end="2026-01-23",
    )
    provider_xshg_1w = SlotDataProvider(slot=slot_xshg_1w, frequency="1w")
    assert provider_xshg_1w.bars_per_session() == pytest.approx(0.2)
    # Trailing 10 sessions = 5 days from 2026-01-23 (2M/day) + 5 days from 2026-01-16 (1M/day) = 1.5M/day.
    assert provider_xshg_1w.average_daily_traded_value(
        ["000001.XSHE"], "2026-01-23T07:00:00+00:00", 10, None, None
    )["000001.XSHE"] == pytest.approx(1_500_000.0)
    # Trailing 5 sessions = only 2026-01-23 week (2M/day) -> cap = 0.1 * 2M * 5 / 10M = 0.10.
    runner_xshg_1w = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "holding_period": 1,
                "liquidity_cap": {
                    "enabled": True,
                    "adv_lookback_days": 5,
                    "max_participation": 0.1,
                },
            }
        },
        slot=slot_xshg_1w,
        frequency="1w",
    )
    try:
        assert runner_xshg_1w._liquidity_weight_caps(
            ["000001.XSHE"], "2026-01-23T07:00:00+00:00", deployed_capital=10_000_000.0
        )["000001.XSHE"] == pytest.approx(0.10)
    finally:
        runner_xshg_1w.close()

    # Partial boundary weeks on calendar must not distort bars_per_session for 1w.
    slot_partial_cal, _, _ = _build_declarative_market_slot(
        tmp_path / "crypto_partial_edge",
        frequency="1w",
        daily_traded_values={"A": {"2026-01-18": 7_000_000.0}},
        calendar_id="crypto_utc",
        start="2026-01-07",
        end="2026-01-20",
    )
    assert SlotDataProvider(slot=slot_partial_cal, frequency="1w").bars_per_session() == pytest.approx(
        1.0 / 7.0
    )


def test_intraday_bar_open_vs_close_and_unclosed_spike_anti_lookahead(tmp_path: Path) -> None:
    """An unclosed bar's traded-value spike must never leak into ADV or ABV before BarSchedule.available_at()."""
    daily_totals = {
        "A": {
            "2026-01-05": 2_400_000.0,  # 100k / 1h bar
            "2026-01-06": 2_400_000.0,  # 100k / 1h bar
            "2026-01-07": 2_400_000.0,  # 100k / 1h bar unless overridden
        }
    }
    spike_1h_mid = 99_900_000.0
    spike_1h_eod = 88_800_000.0
    slot_1h, _, _ = _build_declarative_market_slot(
        tmp_path / "spike_1h",
        frequency="1h",
        daily_traded_values=daily_totals,
        start="2026-01-05",
        end="2026-01-07",
        intraday_profile="uniform",
        bar_value_overrides={
            ("A", "2026-01-07T02:00:00+00:00"): spike_1h_mid,
            ("A", "2026-01-07T23:00:00+00:00"): spike_1h_eod,
        },
    )
    p_1h = SlotDataProvider(slot=slot_1h, frequency="1h")

    # At 02:00:00Z (open of 02:00Z bar) and 02:59:59Z (1s before close), 02:00Z spike is invisible.
    for as_of_before in ("2026-01-07T02:00:00+00:00", "2026-01-07T02:59:59+00:00"):
        assert p_1h.average_bar_traded_value(["A"], as_of_before, 2, None, None)["A"] == pytest.approx(100_000.0)
        assert p_1h.average_daily_traded_value(["A"], as_of_before, 2, None, None)["A"] == pytest.approx(2_400_000.0)
        adv_partial = p_1h.average_daily_traded_value(
            ["A"], as_of_before, 2, None, None, include_incomplete_session=True
        )["A"]
        assert adv_partial == pytest.approx((2_400_000.0 + 200_000.0) / 2.0)

    # At 03:00:00Z (exact available_at of 02:00Z bar), the 02:00Z bar becomes visible.
    assert p_1h.average_bar_traded_value(["A"], "2026-01-07T03:00:00+00:00", 2, None, None)["A"] == pytest.approx(
        (100_000.0 + spike_1h_mid) / 2.0
    )
    assert p_1h.average_daily_traded_value(
        ["A"], "2026-01-07T03:00:00+00:00", 2, None, None, include_incomplete_session=True
    )["A"] == pytest.approx((2_400_000.0 + (200_000.0 + spike_1h_mid)) / 2.0)

    # At 23:00:00Z and 23:59:59Z, the final 23:00Z bar has not closed so 01-07 is incomplete and 23:00Z spike never leaks.
    for as_of_eod_open in ("2026-01-07T23:00:00+00:00", "2026-01-07T23:59:59+00:00"):
        assert p_1h.average_daily_traded_value(["A"], as_of_eod_open, 3, None, None)["A"] == pytest.approx(2_400_000.0)
        assert p_1h.average_bar_traded_value(["A"], as_of_eod_open, 2, None, None)["A"] == pytest.approx(100_000.0)

    # At 2026-01-08T00:00:00Z (session close), 01-07 is complete with all 24 bars including 23:00Z.
    expected_jan7_total = 22 * 100_000.0 + spike_1h_mid + spike_1h_eod
    assert p_1h.average_daily_traded_value(["A"], "2026-01-08T00:00:00+00:00", 3, None, None)["A"] == pytest.approx(
        (2_400_000.0 + 2_400_000.0 + expected_jan7_total) / 3.0
    )
    assert p_1h.average_bar_traded_value(["A"], "2026-01-08T00:00:00+00:00", 2, None, None)["A"] == pytest.approx(
        (100_000.0 + spike_1h_eod) / 2.0
    )


def test_overnight_night_session_boundary_adv_and_abv(tmp_path: Path) -> None:
    """Night-session bars completed before the overnight pause are visible to ABV/partial ADV while session stays incomplete."""
    night_day_sessions = {
        "timezone": "Asia/Shanghai",
        "weekdays": [0, 1, 2, 3, 4],
        "segments": [
            {"open": "21:00", "close": "23:00", "open_day_offset": -1, "close_day_offset": -1},
            {"open": "09:00", "close": "15:00", "open_day_offset": 0, "close_day_offset": 0},
        ],
    }
    # 8 hourly bars per session: 2 night bars (21:00-23:00 CST = 13:00-15:00Z prev day) + 6 day bars (09:00-15:00 CST = 01:00-07:00Z)
    daily_totals = {
        "A": {
            "2026-01-07": 800_000.0,  # 100k / 1h bar
            "2026-01-08": 1_600_000.0,  # 200k / 1h bar
        }
    }
    slot, _, _ = _build_declarative_market_slot(
        tmp_path / "night_1h",
        frequency="1h",
        daily_traded_values=daily_totals,
        calendar_id="night_cny",
        start="2026-01-07",
        end="2026-01-08",
        intraday_profile="uniform",
        custom_sessions_spec=night_day_sessions,
    )
    provider = SlotDataProvider(slot=slot, frequency="1h")

    # At 2026-01-07T15:30:00Z (23:30 CST on Jan 7, during overnight pause after Jan 8 night segment closed at 15:00Z):
    # - Jan 8 night bars (13:00Z and 14:00Z, available_at 14:00Z & 15:00Z) belong to session_id='2026-01-08'
    #   and MUST be visible to ABV and partial ADV even though UTC/local calendar date is still 2026-01-07.
    # - Session 2026-01-08 closes at 2026-01-08T07:00:00Z so default ADV excludes incomplete 2026-01-08.
    as_of_pause = "2026-01-07T15:30:00+00:00"
    assert provider.average_bar_traded_value(["A"], as_of_pause, 2, None, None)["A"] == pytest.approx(200_000.0)
    assert provider.average_daily_traded_value(["A"], as_of_pause, 2, None, None)["A"] == pytest.approx(800_000.0)
    assert provider.average_daily_traded_value(
        ["A"], as_of_pause, 2, None, None, include_incomplete_session=True
    )["A"] == pytest.approx((800_000.0 + 400_000.0) / 2.0)

    # At 2026-01-08T06:59:59Z, 2026-01-08 is still incomplete; at 2026-01-08T07:00:00Z, it is complete.
    assert provider.average_daily_traded_value(["A"], "2026-01-08T06:59:59+00:00", 2, None, None)["A"] == pytest.approx(
        800_000.0
    )
    assert provider.average_daily_traded_value(["A"], "2026-01-08T07:00:00+00:00", 2, None, None)["A"] == pytest.approx(
        (800_000.0 + 1_600_000.0) / 2.0
    )


def test_daily_and_weekly_timestamp_cutoffs_respect_available_at(tmp_path: Path) -> None:
    """Timestamped queries on 1d and 1w bars must withhold the active bar until session/week close."""
    daily_totals = {
        "A": {
            "2026-01-05": 1_000_000.0,
            "2026-01-06": 2_000_000.0,
            "2026-01-07": 9_000_000.0,
        }
    }
    slot_1d, _, _ = _build_declarative_market_slot(
        tmp_path / "d1_ts",
        frequency="1d",
        daily_traded_values=daily_totals,
        start="2026-01-05",
        end="2026-01-07",
    )
    p_1d = SlotDataProvider(slot=slot_1d, frequency="1d")
    assert p_1d.average_daily_traded_value(["A"], "2026-01-07T23:59:59+00:00", 2, None, None)["A"] == pytest.approx(
        1_500_000.0
    )
    assert p_1d.average_daily_traded_value(["A"], "2026-01-08T00:00:00+00:00", 2, None, None)["A"] == pytest.approx(
        5_500_000.0
    )

    weekly_totals = {
        "A": {
            "2026-01-11": 7_000_000.0,
            "2026-01-18": 14_000_000.0,
        }
    }
    slot_1w, _, _ = _build_declarative_market_slot(
        tmp_path / "w1_ts",
        frequency="1w",
        daily_traded_values=weekly_totals,
        start="2026-01-05",
        end="2026-01-18",
    )
    p_1w = SlotDataProvider(slot=slot_1w, frequency="1w")
    # 1 second before week 2026-01-18 closes, only week 2026-01-11 (1M/day) is visible.
    assert p_1w.average_daily_traded_value(["A"], "2026-01-18T23:59:59+00:00", 14, None, None)["A"] == pytest.approx(
        1_000_000.0
    )
    assert p_1w.average_daily_traded_value(["A"], "2026-01-19T00:00:00+00:00", 14, None, None)["A"] == pytest.approx(
        1_500_000.0
    )


def test_runner_rebalance_decision_uses_available_at_and_never_leaks_current_bar(tmp_path: Path) -> None:
    """At execution bar open, runner sees the just-closed prior session/bar and ignores current-bar spikes."""
    # Distinct session values: 01-07=1M must be visible at 01-08T00:00Z rebalance (not dropped as incomplete),
    # while 01-08=100M (current bar/session) must never leak into ADV caps.
    daily_totals = {
        "A": {
            "2026-01-05": 3_000_000.0,
            "2026-01-06": 3_000_000.0,
            "2026-01-07": 1_000_000.0,
            "2026-01-08": 100_000_000.0,
        },
        "B": {day: 50_000_000.0 for day in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")},
        "C": {day: 50_000_000.0 for day in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")},
    }
    weights_by_freq: dict[str, dict[str, float]] = {}
    for freq, hp_bars, rebal_ts_str in (
        ("1d", 1, "2026-01-08"),
        ("1h", 24, "2026-01-08T00:00:00+00:00"),
        ("5min", 288, "2026-01-08T00:00:00+00:00"),
    ):
        slot, close_df, open_df = _build_declarative_market_slot(
            tmp_path / f"e2e_{freq}",
            frequency=freq,
            daily_traded_values=daily_totals,
            start="2026-01-05",
            end="2026-01-08",
            intraday_profile="uniform",
        )
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 3,
                    "holding_period": hp_bars,
                    "score_source": "input_factor",
                    "cost_rate": 0.0,
                    "init_cash": 10_000_000.0,
                    "enable_tday_filter": False,
                    "liquidity_cap": {
                        "enabled": True,
                        "adv_lookback_days": 1,
                        "max_participation": 0.1,
                    },
                }
            },
            slot=slot,
            frequency=freq,
        )
        try:
            scores = pd.DataFrame({"A": 3.0, "B": 2.0, "C": 1.0}, index=close_df.index)
            if freq == "1d":
                daily_close = close_df.copy()
                daily_close.index = pd.DatetimeIndex(
                    [pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).normalize() for ts in daily_close.index]
                )
                daily_open = open_df.copy()
                daily_open.index = daily_close.index
                daily_scores = scores.copy()
                daily_scores.index = daily_close.index
                tw = runner._compute_target_weights(
                    daily_scores, daily_close, {}, [pd.Timestamp(rebal_ts_str).to_pydatetime()], open_df=daily_open
                )
                row = tw.loc[pd.Timestamp(rebal_ts_str)]
            else:
                tw = runner._compute_target_weights(
                    scores, close_df, {}, [pd.Timestamp(rebal_ts_str).to_pydatetime()], open_df=open_df
                )
                row = tw.loc[pd.Timestamp(rebal_ts_str)]
            weights_by_freq[freq] = {s: float(row[s]) for s in ("A", "B", "C")}
        finally:
            runner.close()

    # Cap for A from 2026-01-07 ADV (1M) = 0.1 * 1M = 100k across 1d, 1h, and 5min.
    for freq in ("1d", "1h", "5min"):
        assert weights_by_freq[freq]["A"] == pytest.approx(100_000.0)
        assert weights_by_freq[freq]["B"] == pytest.approx(4_950_000.0)
        assert weights_by_freq[freq]["C"] == pytest.approx(4_950_000.0)

    # Mid-session ABV rebalance at 2026-01-07T02:00:00Z: current bar (02:00Z) has 100M spike,
    # while completed bars (00:00Z, 01:00Z) are 100k -> cap = 0.1 * 100k * 1 = 10k (spike does NOT leak).
    abv_totals = {
        "A": {"2026-01-07": 2_400_000.0},
        "B": {"2026-01-07": 2_400_000_000.0},
        "C": {"2026-01-07": 2_400_000_000.0},
    }
    slot_abv, close_abv, open_abv = _build_declarative_market_slot(
        tmp_path / "e2e_abv_1h",
        frequency="1h",
        daily_traded_values=abv_totals,
        start="2026-01-07",
        end="2026-01-07",
        intraday_profile="uniform",
        bar_value_overrides={("A", "2026-01-07T02:00:00+00:00"): 100_000_000.0},
    )
    runner_abv = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 3,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "init_cash": 10_000_000.0,
                "enable_tday_filter": False,
                "liquidity_cap": {
                    "enabled": True,
                    "liquidity_basis": "abv",
                    "holding_period_unit": "bars",
                    "abv_lookback_bars": 2,
                    "max_participation": 0.1,
                },
            }
        },
        slot=slot_abv,
        frequency="1h",
    )
    try:
        rebal_mid = pd.Timestamp("2026-01-07T02:00:00+00:00")
        scores_abv = pd.DataFrame({"A": 3.0, "B": 2.0, "C": 1.0}, index=close_abv.index)
        tw_abv = runner_abv._compute_target_weights(
            scores_abv, close_abv, {}, [rebal_mid.to_pydatetime()], open_df=open_abv
        )
        assert float(tw_abv.loc[rebal_mid, "A"]) == pytest.approx(10_000.0)
        assert float(tw_abv.loc[rebal_mid, "B"]) == pytest.approx(4_995_000.0)
        assert float(tw_abv.loc[rebal_mid, "C"]) == pytest.approx(4_995_000.0)
    finally:
        runner_abv.close()


@pytest.mark.parametrize("use_money_column", [True, False])
def test_zero_volume_preserved_vs_missing_nan_ignored_across_daily_and_hourly(
    tmp_path: Path, use_money_column: bool
) -> None:
    """Real 0-volume observations stay in ADV/ABV windows (2400,2400,0,0 -> 0.0), while NaN/invalid observations are ignored."""
    from dataclasses import replace

    # 1. Daily (1d): sequence [2400, 2400, 0, 0] vs all-zero vs NaN-missing vs negative/inf invalid
    days = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")
    daily_totals = {
        "A_zero": {"2026-01-05": 2400.0, "2026-01-06": 2400.0, "2026-01-07": 0.0, "2026-01-08": 0.0},
        "B_all_zero": {d: 0.0 for d in days},
        "C_missing": {"2026-01-05": 2400.0, "2026-01-06": 2400.0, "2026-01-07": 1.0, "2026-01-08": 1.0},
        "D_all_missing": {d: 1.0 for d in days},
        "E_invalid": {"2026-01-05": 2400.0, "2026-01-06": 2400.0, "2026-01-07": 1.0, "2026-01-08": 1.0},
    }
    base_1d, _, _ = _build_declarative_market_slot(
        tmp_path / f"z1d_{int(use_money_column)}",
        frequency="1d",
        daily_traded_values=daily_totals,
        start="2026-01-05",
        end="2026-01-08",
        use_money_column=use_money_column,
    )
    orig_feed_1d = base_1d.bar_feed

    def _mutate_1d(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        c_tail = out["instrument_id"].eq("C_missing") & out["session_id"].isin(["2026-01-07", "2026-01-08"])
        d_all = out["instrument_id"].eq("D_all_missing")
        e_tail_neg = out["instrument_id"].eq("E_invalid") & out["session_id"].eq("2026-01-07")
        e_tail_inf = out["instrument_id"].eq("E_invalid") & out["session_id"].eq("2026-01-08")
        if use_money_column:
            out.loc[c_tail | d_all, ["money", "volume"]] = np.nan
            out.loc[e_tail_neg, "money"] = -100.0
            out.loc[e_tail_inf, "money"] = np.inf
        else:
            out.loc[c_tail | d_all, "volume"] = np.nan
            out.loc[e_tail_neg, "volume"] = -100.0
            out.loc[e_tail_inf, "volume"] = np.inf
        return out

    slot_1d = replace(
        base_1d,
        bar_feed=type(
            "_ZeroMutFeed1d",
            (),
            {
                "read_signal_bars": lambda _s, inst, freq, *, start_utc, end_utc, as_of_utc: orig_feed_1d.read_signal_bars(
                    inst, freq, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
                ),
                "read_execution_bars": lambda _s, inst, freq, *, start_utc, end_utc: _mutate_1d(
                    orig_feed_1d.read_execution_bars(inst, freq, start_utc=start_utc, end_utc=end_utc)
                ),
            },
        )(),
    )
    p_1d = SlotDataProvider(slot=slot_1d, frequency="1d")
    names = ["A_zero", "B_all_zero", "C_missing", "D_all_missing", "E_invalid"]
    for _ in range(2):  # verify both uncached compute and cached retrieval preserve 0.0 vs None
        adv_1d = p_1d.average_daily_traded_value(names, "2026-01-08", 2, None, None)
        abv_1d = p_1d.average_bar_traded_value(names, "2026-01-08", 2, None, None)
        assert adv_1d["A_zero"] == pytest.approx(0.0)
        assert adv_1d["B_all_zero"] == pytest.approx(0.0)
        assert adv_1d["C_missing"] == pytest.approx(2400.0)
        assert adv_1d["E_invalid"] == pytest.approx(2400.0)
        assert "D_all_missing" not in adv_1d

        assert abv_1d["A_zero"] == pytest.approx(0.0)
        assert abv_1d["B_all_zero"] == pytest.approx(0.0)
        assert abv_1d["C_missing"] == pytest.approx(2400.0)
        assert abv_1d["E_invalid"] == pytest.approx(2400.0)
        assert "D_all_missing" not in abv_1d

    # 2. Hourly (1h): 2400/day = 100/bar on first 2 days, then 0/bar on last 2 days vs NaN on last 2 days
    base_1h, _, _ = _build_declarative_market_slot(
        tmp_path / f"z1h_{int(use_money_column)}",
        frequency="1h",
        daily_traded_values=daily_totals,
        start="2026-01-05",
        end="2026-01-08",
        use_money_column=use_money_column,
        intraday_profile="uniform",
    )
    orig_feed_1h = base_1h.bar_feed

    def _mutate_1h(df: pd.DataFrame) -> pd.DataFrame:
        return _mutate_1d(df)

    slot_1h = replace(
        base_1h,
        bar_feed=type(
            "_ZeroMutFeed1h",
            (),
            {
                "schedule": orig_feed_1h.schedule,
                "read_signal_bars": lambda _s, inst, freq, *, start_utc, end_utc, as_of_utc: orig_feed_1h.read_signal_bars(
                    inst, freq, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
                ),
                "read_execution_bars": lambda _s, inst, freq, *, start_utc, end_utc: _mutate_1h(
                    orig_feed_1h.read_execution_bars(inst, freq, start_utc=start_utc, end_utc=end_utc)
                ),
            },
        )(),
    )
    p_1h = SlotDataProvider(slot=slot_1h, frequency="1h")
    as_of_1h = "2026-01-09T00:00:00+00:00"
    for _ in range(2):
        adv_1h = p_1h.average_daily_traded_value(names, as_of_1h, 2, None, None)
        abv_1h = p_1h.average_bar_traded_value(names, as_of_1h, 2, None, None)
        assert adv_1h["A_zero"] == pytest.approx(0.0)
        assert adv_1h["B_all_zero"] == pytest.approx(0.0)
        assert adv_1h["C_missing"] == pytest.approx(2400.0)
        assert "D_all_missing" not in adv_1h

        assert abv_1h["A_zero"] == pytest.approx(0.0)
        assert abv_1h["B_all_zero"] == pytest.approx(0.0)
        assert abv_1h["C_missing"] == pytest.approx(100.0)
        assert "D_all_missing" not in abv_1h

    # 3. Weekly (1w): trailing zero week yields 0.0 while missing week is ignored
    weekly_totals = {
        "A_zero": {"2026-01-11": 7000.0, "2026-01-18": 0.0},
        "C_missing": {"2026-01-11": 7000.0, "2026-01-18": 1.0},
    }
    base_1w, _, _ = _build_declarative_market_slot(
        tmp_path / f"z1w_{int(use_money_column)}",
        frequency="1w",
        daily_traded_values=weekly_totals,
        start="2026-01-05",
        end="2026-01-18",
        use_money_column=use_money_column,
    )
    orig_feed_1w = base_1w.bar_feed

    def _mutate_1w(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        c_tail = out["instrument_id"].eq("C_missing") & out["session_id"].eq("2026-01-18")
        if use_money_column:
            out.loc[c_tail, ["money", "volume"]] = np.nan
        else:
            out.loc[c_tail, "volume"] = np.nan
        return out

    slot_1w = replace(
        base_1w,
        bar_feed=type(
            "_ZeroMutFeed1w",
            (),
            {
                "schedule": orig_feed_1w.schedule,
                "read_signal_bars": lambda _s, inst, freq, *, start_utc, end_utc, as_of_utc: orig_feed_1w.read_signal_bars(
                    inst, freq, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
                ),
                "read_execution_bars": lambda _s, inst, freq, *, start_utc, end_utc: _mutate_1w(
                    orig_feed_1w.read_execution_bars(inst, freq, start_utc=start_utc, end_utc=end_utc)
                ),
            },
        )(),
    )
    p_1w = SlotDataProvider(slot=slot_1w, frequency="1w")
    adv_1w = p_1w.average_daily_traded_value(["A_zero", "C_missing"], "2026-01-19T00:00:00+00:00", 7, None, None)
    abv_1w = p_1w.average_bar_traded_value(["A_zero", "C_missing"], "2026-01-19T00:00:00+00:00", 1, None, None)
    assert adv_1w["A_zero"] == pytest.approx(0.0)
    assert adv_1w["C_missing"] == pytest.approx(1000.0)
    assert abv_1w["A_zero"] == pytest.approx(0.0)
    assert abv_1w["C_missing"] == pytest.approx(7000.0)


@pytest.mark.parametrize("pretrade_fixed_targets", [True, False])
def test_zero_liquidity_cap_zero_penetrates_to_final_target_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pretrade_fixed_targets: bool
) -> None:
    """Explicit cap=0.0 from zero-volume history must force final target weights to 0.0 across single/multi/blocked paths."""
    # Unit checks for explicit cap=0.0 generation and magnitude capping
    caps_zero = compute_liquidity_weight_caps(
        {"A": 0.0},
        ["A", "Missing"],
        max_participation=0.1,
        holding_period=1,
        capital=1_000_000.0,
        holding_period_unit="days",
    )
    assert caps_zero == {"A": 0.0}
    assert apply_magnitude_caps({"A": 1.0}, {"A": 0.0}) == {"A": 0.0}
    assert apply_magnitude_caps({"A": 1e-10}, {"A": 0.0}, tol=1e-9) == {"A": 0.0}
    assert apply_magnitude_caps({"A": -0.5, "B": 0.5}, {"A": 0.0, "B": 0.0}) == {"A": 0.0, "B": 0.0}
    assert apply_magnitude_caps({"A": 0.5, "B": 0.5}, {"A": 0.0, "B": 0.8}) == {"A": 0.0, "B": 0.8}

    # 1. Single-name all-zero history (1d & 1h): final target weight == 0.0, zero orders, 100% cash
    for freq, hp_bars, rebal_str in (
        ("1d", 1, "2026-01-07"),
        ("1h", 24, "2026-01-07T00:00:00+00:00"),
    ):
        slot_single, close_s, open_s = _build_declarative_market_slot(
            tmp_path / f"single_{freq}_{int(pretrade_fixed_targets)}",
            frequency=freq,
            daily_traded_values={"A": {"2026-01-05": 0.0, "2026-01-06": 0.0, "2026-01-07": 0.0}},
            start="2026-01-05",
            end="2026-01-07",
            use_money_column=False,
            intraday_profile="uniform",
        )
        if freq == "1d":
            idx = pd.DatetimeIndex(slot_single.calendar.sessions())
            close_s.index = idx
            open_s.index = idx
        scores_s = pd.DataFrame({"A": 1.0}, index=close_s.index)
        runner_s = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 1,
                    "holding_period": hp_bars,
                    "score_source": "input_factor",
                    "cost_rate": 0.0,
                    "init_cash": 100_000.0,
                    "enable_tday_filter": False,
                    "pretrade_fixed_targets": pretrade_fixed_targets,
                    "liquidity_cap": {"enabled": True, "adv_lookback_days": 2, "max_participation": 0.1},
                }
            },
            slot=slot_single,
            frequency=freq,
        )
        try:
            assert runner_s._apply_liquidity_cap({"A": 1.0}, "2026-01-06" if freq == "1d" else rebal_str, 100_000.0) == {
                "A": 0.0
            }
            res_s = runner_s.run_backtest(
                scores_s, close_s, open_df=open_s, rebalance_dates=[pd.Timestamp(rebal_str).to_pydatetime()]
            )
            assert "error" not in res_s, res_s
            tw_s = runner_s._last_target_weights
            assert float(tw_s.loc[pd.Timestamp(rebal_str), "A"]) == pytest.approx(0.0)
            assert len(res_s["portfolio"].orders.records_readable) == 0
            assert (res_s["portfolio"].asset_value(group_by=False).abs() < 1e-12).all().all()
        finally:
            runner_s.close()

    # 2. Multi-name all-zero history (1d): both A and B drop to 0.0
    days_4 = ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08")
    slot_all_zero, close_az, open_az = _build_declarative_market_slot(
        tmp_path / f"all_zero_{int(pretrade_fixed_targets)}",
        frequency="1d",
        daily_traded_values={"A": {d: 0.0 for d in days_4}, "B": {d: 0.0 for d in days_4}},
        start="2026-01-05",
        end="2026-01-08",
        use_money_column=False,
    )
    idx_az = pd.DatetimeIndex(slot_all_zero.calendar.sessions())
    close_az.index = idx_az
    open_az.index = idx_az
    scores_az = pd.DataFrame({"A": 2.0, "B": 1.0}, index=idx_az)
    runner_az = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "init_cash": 100_000.0,
                "enable_tday_filter": False,
                "pretrade_fixed_targets": pretrade_fixed_targets,
                "liquidity_cap": {"enabled": True, "adv_lookback_days": 2, "max_participation": 0.1},
            }
        },
        slot=slot_all_zero,
        frequency="1d",
    )
    try:
        res_az = runner_az.run_backtest(
            scores_az, close_az, open_df=open_az, rebalance_dates=[pd.Timestamp("2026-01-08").to_pydatetime()]
        )
        assert "error" not in res_az, res_az
        tw_az = runner_az._last_target_weights
        assert float(tw_az.loc[pd.Timestamp("2026-01-08"), "A"]) == pytest.approx(0.0)
        assert float(tw_az.loc[pd.Timestamp("2026-01-08"), "B"]) == pytest.approx(0.0)
        assert len(res_az["portfolio"].orders.records_readable) == 0
    finally:
        runner_az.close()

    # 3. Multi-name partial-zero + buy-block reallocation (1h):
    #    A has ADV=0 on 01-07/01-08 (previously liquid on 01-05/01-06 so held on 01-07, then liquidated to 0.0 on 01-08),
    #    B has cap=40k (40%), C is buy-blocked on 01-08. Under pretrade_fixed_targets=False, B must stay <=40k and A must stay 0.0.
    slot_part, close_p, open_p = _build_declarative_market_slot(
        tmp_path / f"part_zero_{int(pretrade_fixed_targets)}",
        frequency="1h",
        daily_traded_values={
            "A": {"2026-01-05": 4_000_000.0, "2026-01-06": 4_000_000.0, "2026-01-07": 0.0, "2026-01-08": 0.0},
            "B": {d: 400_000.0 for d in days_4},  # cap = 0.1 * 400k * 1 = 40k
            "C": {d: 10_000_000.0 for d in days_4},
        },
        start="2026-01-05",
        end="2026-01-08",
        use_money_column=False,
        intraday_profile="uniform",
    )
    t_rebal_1 = pd.Timestamp("2026-01-07T00:00:00+00:00")
    t_rebal_2 = pd.Timestamp("2026-01-08T00:00:00+00:00")
    scores_p = pd.DataFrame({"A": 3.0, "B": 2.0, "C": 1.0}, index=close_p.index)
    runner_p = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 3,
                "holding_period": 24,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "init_cash": 100_000.0,
                "enable_tday_filter": False,
                "pretrade_fixed_targets": pretrade_fixed_targets,
                "liquidity_cap": {"enabled": True, "adv_lookback_days": 1, "max_participation": 0.1},
            }
        },
        slot=slot_part,
        frequency="1h",
    )

    def _block_c_day2(date_str: str, stocks: list[str], side: str = "both", require_non_st: bool = True) -> list[str]:
        ts = pd.Timestamp(date_str).tz_convert("UTC") if pd.Timestamp(date_str).tzinfo else pd.Timestamp(date_str, tz="UTC")
        if ts == t_rebal_2 and side == "buy":
            return [s for s in stocks if s != "C"]
        return list(stocks)

    monkeypatch.setattr(runner_p, "_filter_tradeable_stocks", _block_c_day2)
    try:
        res_p = runner_p.run_backtest(
            scores_p,
            close_p,
            open_df=open_p,
            rebalance_dates=[t_rebal_1.to_pydatetime(), t_rebal_2.to_pydatetime()],
        )
        assert "error" not in res_p, res_p
        tw_p = runner_p._last_target_weights
        # At t_rebal_1 (using 01-06 ADV): A liquid -> 1/3 (~33.3k), B capped 40k, C gets remainder
        assert float(tw_p.loc[t_rebal_1, "A"]) > 0.0
        # At t_rebal_2 (using 01-07 ADV=0 for A): A must be liquidated to 0.0; C is buy-blocked; B stays <= 40k cap
        assert float(tw_p.loc[t_rebal_2, "A"]) == pytest.approx(0.0)
        assert float(tw_p.loc[t_rebal_2, "B"]) == pytest.approx(40_000.0)
        assert float(tw_p.loc[t_rebal_2, "C"]) == pytest.approx(0.0) or pd.isna(tw_p.loc[t_rebal_2, "C"])
    finally:
        runner_p.close()


def test_futures_ohlcv_fallback_applies_contract_multiplier_without_double_counting(tmp_path: Path) -> None:
    """Futures OHLCV fallback (volume * close) multiplies by contract_multiplier without double-counting real money."""
    from dataclasses import replace

    import duckdb

    from data_pipeline.loaders.market_bar_importer import import_market_bars
    from market_db_samples import declare_futures_bars
    from slots.cn_futures import build_cn_futures_slot
    from utils.bars.futures_roll import build_roll_schedule
    from utils.session_calendar import SessionCalendar
    from utils.time_axis import BarSchedule
    from utils.time_axis.futures import futures_calendar

    start, end = "2026-01-05", "2026-01-09"
    root, exchange = "CU", "SHFE"
    fcal = futures_calendar(root)
    cal = SessionCalendar(fcal["calendar_id"], start, end, rth_reference_calendar=fcal["rth_reference_calendar"])
    sessions = cal.sessions()
    schedule = BarSchedule(cal, "1d", bucket_strategy="segment_anchored")
    slot_times = pd.concat([schedule.for_session(day) for day in sessions], ignore_index=True)["start_utc"]
    c1, c2 = "CU2603.SHFE", "CU2604.SHFE"

    def _build_futures_db(db_file: Path, *, mult_c1: float, mult_c2: float, closes_c1: list[float], closes_c2: list[float]) -> str:
        meta = pd.DataFrame(
            [
                dict(
                    instrument_id=c1,
                    product_root=root,
                    asset_class="futures",
                    exchange=exchange,
                    timezone=cal.timezone,
                    contract_expiry="2026-03-16",
                    contract_month="2026-03",
                    price_adjustment="none",
                    point_value=float(mult_c1),
                    rth_reference_calendar=fcal["rth_reference_calendar"],
                ),
                dict(
                    instrument_id=c2,
                    product_root=root,
                    asset_class="futures",
                    exchange=exchange,
                    timezone=cal.timezone,
                    contract_expiry="2026-04-16",
                    contract_month="2026-04",
                    price_adjustment="none",
                    point_value=float(mult_c2),
                    rth_reference_calendar=fcal["rth_reference_calendar"],
                ),
            ]
        )
        # C1 leads sessions 0..2 (effective 01-06..01-08), C2 leads sessions 3..4 (effective 01-09)
        oi_c1 = [500.0, 500.0, 500.0, 100.0, 100.0]
        oi_c2 = [100.0, 100.0, 100.0, 500.0, 500.0]
        bars = []
        for contract, closes, ois in ((c1, closes_c1, oi_c1), (c2, closes_c2, oi_c2)):
            c_arr = np.asarray(closes, dtype=float)
            bars.append(
                pd.DataFrame(
                    {
                        "timestamp_utc": slot_times,
                        "instrument_id": contract,
                        "frequency": "1d",
                        "open": c_arr - 1.0,
                        "high": c_arr + 1.0,
                        "low": c_arr - 2.0,
                        "close": c_arr,
                        "open_interest": ois,
                        "volume": 500.0,
                        "source": "SYNTH",
                        "data_version": "v1",
                    }
                )
            )
        conn = duckdb.connect(str(db_file))
        try:
            declare_futures_bars(conn, exchange)
            import_market_bars(
                conn, pd.concat(bars, ignore_index=True), meta, bucket_strategy="segment_anchored", legacy_compatibility=False
            )
            roll_start = sessions[1].strftime("%Y-%m-%d")
            build_roll_schedule(conn, product_roots=[root], start=roll_start, end=end)
            return roll_start
        finally:
            conn.close()

    # 1. Native & continuous futures: mult=10, trailing 2 sessions close=[111, 113], volume=500 -> ADV = 560,000
    db_same = tmp_path / "futures_same_mult.duckdb"
    roll_start = _build_futures_db(
        db_same,
        mult_c1=10.0,
        mult_c2=10.0,
        closes_c1=[105.0, 107.0, 109.0, 111.0, 113.0],
        closes_c2=[105.0, 107.0, 109.0, 111.0, 113.0],
    )
    conn_same = duckdb.connect(str(db_same), read_only=True)
    try:
        slot = build_cn_futures_slot(conn_same, product_root=root, start=roll_start, end=end)
        orig_feed = slot.bar_feed

        def _wrap_money(mode: str):
            def _read_exec(insts, freq, *, start_utc, end_utc):
                df = orig_feed.read_execution_bars(insts, freq, start_utc=start_utc, end_utc=end_utc).copy()
                if mode == "full":
                    df["money"] = df["volume"] * df["close"] * 10.0
                elif mode == "partial":
                    # First bar has real money, last bar falls back to volume * close * multiplier
                    df["money"] = df["volume"] * df["close"] * 10.0
                    last_sid = df["session_id"].max()
                    df.loc[df["session_id"].eq(last_sid), "money"] = np.nan
                return df

            return replace(
                slot,
                bar_feed=type(
                    "_MoneyFeed",
                    (),
                    {
                        "read_signal_bars": lambda _s, insts, freq, *, start_utc, end_utc, as_of_utc: orig_feed.read_signal_bars(
                            insts, freq, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
                        ),
                        "read_execution_bars": lambda _s, insts, freq, *, start_utc, end_utc: _read_exec(
                            insts, freq, start_utc=start_utc, end_utc=end_utc
                        ),
                    },
                )(),
            )

        # Native concrete contract (CU2604.SHFE) and continuous root (CU)
        for target_inst in (c2, root):
            # Missing money column -> fallback volume * close * 10 == 560,000
            p_nomoney = SlotDataProvider(slot=slot, frequency="1d")
            assert p_nomoney.average_daily_traded_value([target_inst], end, 2, None, None)[target_inst] == pytest.approx(560_000.0)
            assert p_nomoney.average_bar_traded_value([target_inst], end, 2, None, None)[target_inst] == pytest.approx(560_000.0)

            # Full money column -> exact 560,000 (never double-multiplied to 5,600,000)
            p_full = SlotDataProvider(slot=_wrap_money("full"), frequency="1d")
            assert p_full.average_daily_traded_value([target_inst], end, 2, None, None)[target_inst] == pytest.approx(560_000.0)
            assert p_full.average_bar_traded_value([target_inst], end, 2, None, None)[target_inst] == pytest.approx(560_000.0)

            # Partial money column -> rows with money keep money, rows without money use volume * close * 10
            p_partial = SlotDataProvider(slot=_wrap_money("partial"), frequency="1d")
            assert p_partial.average_daily_traded_value([target_inst], end, 2, None, None)[target_inst] == pytest.approx(560_000.0)
            assert p_partial.average_bar_traded_value([target_inst], end, 2, None, None)[target_inst] == pytest.approx(560_000.0)
    finally:
        conn_same.close()

    # 2. Continuous futures rolling across contracts with different multipliers (10 -> 20)
    db_diff = tmp_path / "futures_diff_mult.duckdb"
    roll_start_diff = _build_futures_db(
        db_diff,
        mult_c1=10.0,
        mult_c2=20.0,
        closes_c1=[100.0] * 5,
        closes_c2=[100.0] * 5,
    )
    conn_diff = duckdb.connect(str(db_diff), read_only=True)
    try:
        slot_diff = build_cn_futures_slot(conn_diff, product_root=root, start=roll_start_diff, end=end)
        # Session 2026-01-08 uses CU2603.SHFE (mult=10 -> 500*100*10 = 500,000),
        # Session 2026-01-09 uses CU2604.SHFE (mult=20 -> 500*100*20 = 1,000,000) -> ADV(2) = 750,000
        p_diff = SlotDataProvider(slot=slot_diff, frequency="1d")
        assert p_diff.average_daily_traded_value([root], end, 2, None, None)[root] == pytest.approx(750_000.0)
        assert p_diff.average_bar_traded_value([root], end, 2, None, None)[root] == pytest.approx(750_000.0)

        # Also test slot.roll_schedule resolution when concrete_instrument_id is absent, and root-spec fallback
        orig_diff_feed = slot_diff.bar_feed

        class _RollSchedSlot:
            calendar = slot_diff.calendar
            universe = slot_diff.universe
            aux_features = slot_diff.aux_features
            microstructure = slot_diff.microstructure
            slot_id = slot_diff.slot_id
            roll_schedule = {root: {"2026-01-08": c1, "2026-01-09": c2}}
            bar_feed = type(
                "_NoConcreteFeed",
                (),
                {
                    "read_signal_bars": lambda _s, insts, freq, *, start_utc, end_utc, as_of_utc: orig_diff_feed.read_signal_bars(
                        insts, freq, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
                    ),
                    "read_execution_bars": lambda _s, insts, freq, *, start_utc, end_utc: orig_diff_feed.read_execution_bars(
                        insts, freq, start_utc=start_utc, end_utc=end_utc
                    ).drop(columns=["concrete_instrument_id"]),
                },
            )()

        p_sched = SlotDataProvider(slot=_RollSchedSlot(), frequency="1d")  # type: ignore[arg-type]
        assert p_sched.average_daily_traded_value([root], end, 2, None, None)[root] == pytest.approx(750_000.0)

        # Root-only fallback when neither concrete_instrument_id nor roll_schedule exists
        from utils.slots.contracts import InstrumentSpec

        class _RootSpecUniverse:
            def instrument_spec(self, instrument_id: str) -> InstrumentSpec:
                return InstrumentSpec(instrument_id, "futures", "CNY", contract_multiplier=10.0)

            def resolve_universe(self, universe_id: str, *, as_of_utc):
                return (root,)

            def resolve_active_contract(self, instrument_id: str, *, as_of_utc):
                return instrument_id

        slot_root = replace(slot_diff, universe=_RootSpecUniverse(), bar_feed=_RollSchedSlot.bar_feed)
        p_root = SlotDataProvider(slot=slot_root, frequency="1d")
        assert p_root.average_daily_traded_value([root], end, 2, None, None)[root] == pytest.approx(500_000.0)
    finally:
        conn_diff.close()
