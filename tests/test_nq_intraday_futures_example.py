"""End-to-end tests for the declarative NQ 1min intraday futures example (ETH & RTH)."""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series import evaluate_slot
from slots import build_slot_from_config
from utils.time_axis import Frequency

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "nq_intraday_futures"


def _read_cli_outputs(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    assert payload["ok"] is True, payload
    assert payload["metrics"]["causal_gate"]["status"] == "passed"
    items = {item["artifact_type"]: Path(item["path"]) for item in payload["artifacts"]}
    for kind in ("per_bar_labels", "ic_by_horizon", "quantile_stats", "evaluation_summary"):
        item = next(entry for entry in payload["artifacts"] if entry["artifact_type"] == kind)
        assert hashlib.sha256(items[kind].read_bytes()).hexdigest() == item["sha256"]
    summary = json.loads(items["evaluation_summary"].read_text(encoding="utf-8"))
    labels = pd.read_csv(items["per_bar_labels"])
    labels["signal_timestamp"] = pd.to_datetime(labels["signal_timestamp"], utc=True)
    return (
        labels,
        pd.read_csv(items["ic_by_horizon"]),
        pd.read_csv(items["quantile_stats"]),
        summary,
    )


@pytest.mark.parametrize(
    ("manifest_name", "expected_view", "expected_slot_id"),
    [
        ("slot_eth.yaml", "ETH", "nq_intraday_eth"),
        ("slot_rth.yaml", "RTH", "nq_intraday_rth"),
    ],
)
def test_nq_intraday_slot_configs_load_with_expected_metadata_and_calendar(
    manifest_name: str,
    expected_view: str,
    expected_slot_id: str,
) -> None:
    slot = build_slot_from_config(EXAMPLE_DIR / manifest_name)
    assert slot.slot_id == expected_slot_id
    assert slot.bar_feed.view == expected_view
    assert slot.microstructure.view == expected_view
    assert slot.bar_feed.frequency.canonical == "1min"
    assert slot.bar_feed.spec.timestamp_anchor == "bar_start"
    assert slot.bar_feed.spec.source_timezone == "UTC"
    assert slot.bar_feed.spec.volume_mode == "interval"
    assert slot.calendar.calendar_id == "CME Globex Equity"
    assert slot.calendar.rth_reference_calendar == "NYSE"
    assert slot.calendar.sessions().strftime("%Y-%m-%d").tolist() == ["2026-12-01", "2026-12-02"]

    spec = slot.universe.instrument_spec("NQZ26")
    assert spec.asset_class == "futures"
    assert spec.currency == "USD"
    assert spec.contract_multiplier == pytest.approx(20.0)
    assert spec.expiry_date == date(2026, 12, 18)

    profile = slot.microstructure.execution_profile
    assert (profile.entry_price_field, profile.exit_price_field) == ("open", "close")
    assert (profile.min_entry_offset_bars, profile.exit_adjustment) == (1, -1)
    assert (profile.return_unit, profile.gap_rule, profile.boundary_policy, profile.allow_short) == (
        "decimal",
        "endpoints_only",
        "same_session",
        True,
    )

    bars = slot.bar_feed.load(instruments=["NQZ26"])
    assert not bars.empty
    assert set(bars.instrument_id) == {"NQZ26"}
    assert bars.frequency.eq("1min").all()
    if expected_view == "ETH":
        assert len(bars) == 30
        assert pd.Timestamp("2026-12-01T04:00:00Z") in set(bars.timestamp_utc)
        assert pd.Timestamp("2026-12-01T21:00:00Z") in set(bars.timestamp_utc)
    else:
        assert len(bars) == 21
        assert pd.Timestamp("2026-12-01T04:00:00Z") not in set(bars.timestamp_utc)
        assert pd.Timestamp("2026-12-01T21:00:00Z") not in set(bars.timestamp_utc)


@pytest.mark.parametrize("view", ["ETH", "RTH"])
def test_nq_intraday_evaluate_slot_and_cli_hand_computed_oracles(tmp_path: Path, view: str) -> None:
    suffix = view.lower()
    slot = build_slot_from_config(EXAMPLE_DIR / f"slot_{suffix}.yaml")
    bars = slot.bar_feed.read_execution_bars(
        ["NQZ26"],
        Frequency.parse("1m"),
        start_utc=pd.Timestamp("2026-11-30T23:00:00Z"),
        end_utc=pd.Timestamp("2026-12-02T22:00:00Z"),
    )
    factors = bars[["timestamp_utc", "instrument_id"]].assign(factor_value=bars["close"] - bars["open"])

    direct = evaluate_slot(
        slot,
        factors,
        frequency="1m",
        start_date="2026-12-01",
        end_date="2026-12-02",
        instruments=["NQZ26"],
        horizons=[1, 5, 15, 60],
        entry_offset_bars=1,
        analysis_mode="time_series",
        n_quantiles=5,
        min_ic_samples=3,
        apply_tradability_mask=True,
    )

    out_dir = tmp_path / f"eval_{suffix}"
    harness_root = tmp_path / f"harness_{suffix}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.agent_harness.cli",
            "time-series-analysis",
            "--config",
            f"examples/nq_intraday_futures/eval_{suffix}.json",
            "--output-dir",
            str(out_dir),
            "--output-root",
            str(harness_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    labels, ic_df, quantiles_df, summary = _read_cli_outputs(json.loads(proc.stdout))

    assert summary["analysis_mode"] == "time_series"
    assert summary["frequency"] == "1min"
    assert summary["return_unit"] == "decimal"
    assert summary["calendar_identity"]["view"] == view
    assert summary["calendar_identity"]["boundary_policy"] == "same_session"

    # CLI (full-grid FactorBase) and direct evaluate_slot (observed signal keys) must agree on all valid metrics.
    direct_ic = direct.ic_by_horizon.query("scope == 'per_instrument'").set_index("horizon_bars")
    cli_ic = ic_df.query("scope == 'per_instrument'").set_index("horizon_bars")
    for h in (1, 5, 15, 60):
        assert cli_ic.loc[h, "valid_count"] == direct_ic.loc[h, "valid_count"]
        assert cli_ic.loc[h, "spearman_ic"] == pytest.approx(direct_ic.loc[h, "spearman_ic"], nan_ok=True)
        assert cli_ic.loc[h, "ic_status"] == direct_ic.loc[h, "ic_status"]
        assert json.loads(cli_ic.loc[h, "exclusion_counts"])["session_boundary"] == direct_ic.loc[
            h, "exclusion_counts"
        ]["session_boundary"]

    # 1. Valid block: exact entry/exit timestamps, decimal returns, Spearman IC = 1.0, and quantiles.
    by_h_ts = labels.set_index(["horizon_bars", "signal_timestamp"])
    for idx in range(10):
        sig_ts = pd.Timestamp(f"2026-12-01T14:{30 + idx:02d}:00Z")
        row = by_h_ts.loc[(1, sig_ts)]
        assert row["status"] == "valid"
        assert pd.isna(row["exclusion_reason"])
        assert pd.Timestamp(row["entry_timestamp"]) == pd.Timestamp(f"2026-12-01T14:{31 + idx:02d}:00Z")
        assert pd.Timestamp(row["exit_timestamp"]) == pd.Timestamp(f"2026-12-01T14:{32 + idx:02d}:00Z")
        assert row["entry_price"] == pytest.approx(20000.0)
        assert row["exit_price"] == pytest.approx(20000.0 + 20.0 * (idx + 2))
        assert row["return_pct"] == pytest.approx((idx + 2) * 0.001)

    for idx in range(6):
        sig_ts = pd.Timestamp(f"2026-12-01T14:{30 + idx:02d}:00Z")
        row5 = by_h_ts.loc[(5, sig_ts)]
        assert row5["status"] == "valid"
        assert pd.Timestamp(row5["entry_timestamp"]) == pd.Timestamp(f"2026-12-01T14:{31 + idx:02d}:00Z")
        assert pd.Timestamp(row5["exit_timestamp"]) == pd.Timestamp(f"2026-12-01T14:{36 + idx:02d}:00Z")
        assert row5["return_pct"] == pytest.approx((idx + 6) * 0.001)

    assert cli_ic.loc[1, "valid_count"] == (21 if view == "ETH" else 15)
    assert cli_ic.loc[1, "spearman_ic"] == pytest.approx(1.0)
    assert cli_ic.loc[1, "non_overlapping_ic"] == pytest.approx(1.0)
    assert cli_ic.loc[5, "valid_count"] == 6
    assert cli_ic.loc[5, "spearman_ic"] == pytest.approx(1.0)
    assert cli_ic.loc[15, "valid_count"] == 0
    assert cli_ic.loc[15, "ic_status"] == "insufficient_samples"

    q1 = quantiles_df.loc[quantiles_df["horizon_bars"].eq(1)].sort_values("quantile")
    assert q1["quantile"].tolist() == [1, 2, 3, 4, 5]
    if view == "RTH":
        assert q1["sample_count"].tolist() == [3, 3, 3, 3, 3]
        assert q1["mean_return_pct"].tolist() == pytest.approx([0.003, 0.006, 0.009, 0.0125, 0.059 / 3])
        assert q1["median_return_pct"].tolist() == pytest.approx([0.003, 0.006, 0.009, 0.0125, 0.017])
        assert q1["positive_ratio"].eq(1.0).all()
    else:
        assert q1["sample_count"].tolist() == [5, 4, 4, 4, 4]
        assert q1["mean_return_pct"].tolist() == pytest.approx([0.00295, 0.0075, 0.011875, 0.0175, 0.02325])

    # 2. Near-close same_session boundary: RTH truncates at 21:00Z (16:00 ET), ETH continues until 22:00Z (17:00 ET).
    rth_edge_h1 = by_h_ts.loc[(1, pd.Timestamp("2026-12-01T20:59:00Z"))]
    if view == "RTH":
        assert rth_edge_h1["exclusion_reason"] == "session_boundary"
        assert np.isnan(rth_edge_h1["return_pct"])
        for ts_str in ("2026-12-01T20:57:00Z", "2026-12-01T20:58:00Z", "2026-12-01T20:59:00Z"):
            ts = pd.Timestamp(ts_str)
            assert by_h_ts.loc[(5, ts), "exclusion_reason"] == "session_boundary"
            assert by_h_ts.loc[(60, ts), "exclusion_reason"] == "session_boundary"
        assert cli_ic.loc[60, "valid_count"] == 0
    else:
        assert rth_edge_h1["status"] == "valid"
        assert rth_edge_h1["return_pct"] == pytest.approx(0.018)
        assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T21:58:00Z")), "status"] == "valid"
        assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T21:58:00Z")), "return_pct"] == pytest.approx(0.022)
        assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T21:59:00Z")), "exclusion_reason"] == "session_boundary"
        for ts_str in ("2026-12-01T21:57:00Z", "2026-12-01T21:58:00Z", "2026-12-01T21:59:00Z"):
            ts = pd.Timestamp(ts_str)
            assert by_h_ts.loc[(5, ts), "exclusion_reason"] == "session_boundary"
            assert by_h_ts.loc[(60, ts), "exclusion_reason"] == "session_boundary"
        for idx, ts_str in enumerate(("2026-12-01T20:57:00Z", "2026-12-01T20:58:00Z", "2026-12-01T20:59:00Z")):
            row60 = by_h_ts.loc[(60, pd.Timestamp(ts_str))]
            assert row60["status"] == "valid"
            assert row60["return_pct"] == pytest.approx(0.020 + idx * 0.001)
        assert cli_ic.loc[60, "valid_count"] == 3
        assert cli_ic.loc[60, "spearman_ic"] == pytest.approx(1.0)

    # 3. Missing bar (15:02Z omitted): grid slot is factor_missing; adjacent signals record missing_bar without shifting.
    assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T15:00:00Z")), "status"] == "valid"
    assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T15:00:00Z")), "return_pct"] == pytest.approx(0.0125)
    assert by_h_ts.loc[(5, pd.Timestamp("2026-12-01T15:00:00Z")), "exclusion_reason"] == "missing_bar"
    assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T15:01:00Z")), "exclusion_reason"] == "missing_bar"
    assert np.isnan(by_h_ts.loc[(1, pd.Timestamp("2026-12-01T15:01:00Z")), "return_pct"])
    assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T15:02:00Z")), "exclusion_reason"] == "factor_missing"
    assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T15:03:00Z")), "status"] == "valid"
    assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T15:03:00Z")), "return_pct"] == pytest.approx(0.014)
    assert by_h_ts.loc[(1, pd.Timestamp("2026-12-01T15:04:00Z")), "exclusion_reason"] == "missing_bar"
    assert by_h_ts.loc[(1, pd.Timestamp("2026-12-02T15:35:00Z")), "exclusion_reason"] == "insufficient_history"
