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
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Build a real declarative Slot at ``1d``/``1h``/``30min``/``5min`` preserving exact session totals."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    is_crypto = calendar_id == "crypto_utc"
    cal_cfg = (
        {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "crypto_utc",
                "start": start,
                "end": end,
                "custom_sessions": {
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
                },
            },
        }
        if is_crypto
        else {
            "provider": "pmc",
            "options": {"calendar_id": calendar_id, "start": start, "end": end},
        }
    )

    from utils.session_calendar import SessionCalendar

    if is_crypto:
        cal = SessionCalendar(
            "crypto_utc",
            start,
            end,
            custom_sessions=cal_cfg["options"]["custom_sessions"],
        )
    else:
        cal = SessionCalendar(calendar_id, start, end)

    freq_obj = Frequency.parse(frequency)
    grid = cal.bar_grid(frequency, start, end)
    symbols = sorted(daily_traded_values.keys())

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
            bar_ends = (
                bar_starts + freq_obj.duration
                if freq_obj.kind == "intraday"
                else bar_starts
            )
            if future_spike_date is not None and str(session_id) == future_spike_date and n_bars > 1:
                # Place 90% of the day's value in the final bar to catch any look-ahead leak.
                weights = np.full(n_bars, 0.1 / (n_bars - 1), dtype=float)
                weights[-1] = 0.9

            for idx_bar, (bar_start, bar_end) in enumerate(
                zip(sess_grid["timestamp_utc"], bar_ends)
            ):
                bar_val = total_val * float(weights[idx_bar])
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
                        "asset_class": "crypto" if is_crypto else "equity",
                        "currency": "USD" if is_crypto else "CNY",
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
            "exit_price_field": "close",
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
        ("1h", "2026-01-08T23:00:00+00:00"),
        ("5min", "2026-01-08T23:55:00+00:00"),
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

    # Explicit include_incomplete_session=True sums ONLY bars <= 02:00Z (3/23 of 10% = 10M * 3/23)
    # and strictly excludes the 23:00Z 90M future spike.
    expected_partial_jan7 = 100_000_000.0 * (0.1 / 23.0) * 3.0
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

    # On 1h, ADV(2 days) = (2.4M + 4.8M) / 2 = 3.6M, while ABV(2 bars at end of 01-07) = 200k/bar.
    assert p_1h.average_daily_traded_value(["A"], "2026-01-07T23:00:00+00:00", 2, None, None)["A"] == pytest.approx(3_600_000.0)
    assert p_1h.average_bar_traded_value(["A"], "2026-01-07T23:00:00+00:00", 2, None, None)["A"] == pytest.approx(200_000.0)

    # On 5min, ADV(2 days) = 3.6M, while ABV(2 bars at end of 01-07) = 4.8M / 288 = 16,666.6667/bar.
    assert p_5m.average_daily_traded_value(["A"], "2026-01-07T23:55:00+00:00", 2, None, None)["A"] == pytest.approx(3_600_000.0)
    assert p_5m.average_bar_traded_value(["A"], "2026-01-07T23:55:00+00:00", 2, None, None)["A"] == pytest.approx(4_800_000.0 / 288.0)


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
        ("1h", 24, "2026-01-07T23:00:00+00:00", "2026-01-08T00:00:00+00:00"),
        ("5min", 288, "2026-01-07T23:55:00+00:00", "2026-01-08T00:00:00+00:00"),
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
        ("5min", 48, "2026-01-06T06:55:00+00:00"),
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
            ["000001.XSHE"], "2026-01-06T06:55:00+00:00", deployed_capital=10_000_000.0
        )
        # 6 / 48 day * 4.8M/day * 0.1 / 10M == 6 * 100k * 0.1 / 10M == 0.006
        assert caps_abv["000001.XSHE"] == pytest.approx(0.006)

        with pytest.raises(ValueError, match="lookback_days must be positive"):
            runner_abv._slot_provider.average_daily_traded_value(
                ["000001.XSHE"], "2026-01-06T06:55:00+00:00", 0, None, None
            )
        with pytest.raises(ValueError, match="lookback_bars must be positive"):
            runner_abv._slot_provider.average_bar_traded_value(
                ["000001.XSHE"], "2026-01-06T06:55:00+00:00", -2, None, None
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
                    ["000001.XSHE"], "2026-01-06T06:55:00+00:00", deployed_capital=10_000_000.0
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
