"""Lightweight contract tests for the minute-bar scale benchmark tool."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tools.benchmarks.minute_bar_scale_benchmark import (
    SCHEMA_VERSION,
    BenchmarkConfig,
    _compare_fast_vs_pit,
    render_markdown_report,
    run_benchmark,
)


def test_benchmark_config_validation() -> None:
    with pytest.raises(ValueError, match="date range"):
        BenchmarkConfig(start_date="2026-01-10", end_date="2026-01-05")
    with pytest.raises(ValueError, match="intraday"):
        BenchmarkConfig(frequency="1d")
    with pytest.raises(ValueError, match="calendar_mode"):
        BenchmarkConfig(calendar_mode="unknown")
    with pytest.raises(ValueError, match="source_format"):
        BenchmarkConfig(source_format="json")
    with pytest.raises(ValueError, match="num_instruments"):
        BenchmarkConfig(num_instruments=0)
    with pytest.raises(ValueError, match="num_factors"):
        BenchmarkConfig(num_factors=0)
    with pytest.raises(ValueError, match="incremental_count"):
        BenchmarkConfig(num_factors=4, incremental_count=5)


def test_small_scale_benchmark_contract_and_schema(tmp_path: Path) -> None:
    work_dir = tmp_path / "bench_work"
    cfg = BenchmarkConfig(
        start_date="2026-01-05",
        end_date="2026-01-06",
        frequency="60min",
        calendar_mode="crypto_24x7",
        source_format="parquet",
        num_instruments=2,
        num_factors=6,
        incremental_count=2,
        pit_sample_cutoffs=4,
        pit_sample_factors=4,
        with_gaps_and_nans=True,
        enable_cache=True,
        warm_pass=True,
        enable_tracemalloc=True,
        work_dir=work_dir,
        cleanup_work_dir=True,
    )
    report = run_benchmark(cfg)

    assert report["schema_version"] == SCHEMA_VERSION
    assert set(report) >= {
        "schema_version",
        "timestamp_utc",
        "machine",
        "config",
        "dataset_stats",
        "stage_timings_seconds",
        "factor_summary",
        "throughput",
        "memory",
        "io",
        "storage",
        "pit_consistency",
    }

    ds = report["dataset_stats"]
    assert ds["total_sessions"] == 2
    assert ds["cutoffs_per_instrument"] == 48
    assert ds["total_scheduled_grid_rows"] == 96
    assert ds["injected_gap_rows"] > 0
    assert ds["injected_nan_rows"] > 0
    assert ds["dataset_bytes"] > 0

    fs = report["factor_summary"]
    assert fs["total_factors"] == 6
    assert fs["vectorized_count"] == 4
    assert fs["incremental_count"] == 2
    assert fs["pit_replay_fallback_count"] == 0
    assert fs["cold_cache_hits"] == 0
    assert fs["warm_cache_hits"] == 6

    st = report["stage_timings_seconds"]
    for key in (
        "data_generation_seconds",
        "cold_batch_compute_seconds",
        "warm_cache_hit_seconds",
        "pit_replay_sample_seconds",
        "total_benchmark_seconds",
    ):
        assert st[key] >= 0.0

    io = report["io"]
    assert io["cold_read_signal_bars_calls"] == 6 * 2
    assert io["warm_read_signal_bars_calls"] == 0
    assert io["pit_sample_read_signal_bars_calls"] == 4 * 4 * 2

    stor = report["storage"]
    assert stor["cache_pkl_count"] == 6
    assert stor["cache_slotmeta_count"] == 6
    assert stor["cache_file_count"] == 12
    assert stor["cache_total_bytes"] > 0

    mem = report["memory"]
    assert mem["peak_rss_bytes"] > 0
    assert mem["tracemalloc_peak_bytes"] > 0

    pit = report["pit_consistency"]
    assert pit["passed"] is True
    assert pit["factors_sampled"] == 4
    assert pit["cutoffs_sampled"] == 4
    assert pit["total_compared_points"] == 4 * 4 * 2
    assert pit["total_nan_mismatch_count"] == 0
    assert pit["overall_max_abs_diff"] <= 1e-12
    sampled_modes = {row["fast_mode"] for row in pit["factors_checked"]}
    assert sampled_modes == {"vectorized", "incremental"}

    assert list(work_dir.glob("run_*")) == []

    md = render_markdown_report(report)
    assert "# Minute-Bar Scale Benchmark Report" in md
    assert "Sampled `pit_replay` Consistency Verification" in md
    assert "`PASSED`" in md


def test_benchmark_cli_and_report_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    json_out = tmp_path / "out" / "report.json"
    md_out = tmp_path / "out" / "report.md"
    work_dir = tmp_path / "cli_work"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.benchmarks.minute_bar_scale_benchmark",
            "--start-date",
            "2026-01-05",
            "--end-date",
            "2026-01-06",
            "--frequency",
            "60min",
            "--calendar-mode",
            "trading_session",
            "--source-format",
            "csv",
            "--num-instruments",
            "2",
            "--num-factors",
            "4",
            "--incremental-count",
            "1",
            "--pit-sample-cutoffs",
            "3",
            "--pit-sample-factors",
            "2",
            "--no-tracemalloc",
            "--work-dir",
            str(work_dir),
            "--json-output",
            str(json_out),
            "--md-output",
            str(md_out),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json_out.is_file()
    assert md_out.is_file()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["factor_summary"]["vectorized_count"] == 3
    assert payload["factor_summary"]["incremental_count"] == 1
    assert payload["pit_consistency"]["passed"] is True
    assert "PASSED" in md_out.read_text(encoding="utf-8")


def test_compare_fast_vs_pit_detects_value_and_nan_mismatch() -> None:
    ts = pd.to_datetime(["2026-01-05T10:00:00Z", "2026-01-05T11:00:00Z"], utc=True)
    panel = pd.DataFrame({"SYM-01": [1.0, 2.0]}, index=ts)
    panel.index.name = "date"
    sampled_targets = pd.DataFrame(
        {"timestamp_utc": [ts[0], ts[1]], "instrument_id": ["SYM-01", "SYM-01"]}
    )

    # Exact match passes
    pit_ok = pd.DataFrame(
        {"timestamp_utc": [ts[0], ts[1]], "instrument_id": ["SYM-01", "SYM-01"], "factor_value": [1.0, 2.0]}
    )
    ok_res = _compare_fast_vs_pit(panel, pit_ok, sampled_targets, rtol=1e-10, atol=1e-12)
    assert ok_res["passed"] is True
    assert ok_res["nan_mismatch_count"] == 0

    # Numeric drift fails
    pit_drift = pd.DataFrame(
        {"timestamp_utc": [ts[0], ts[1]], "instrument_id": ["SYM-01", "SYM-01"], "factor_value": [1.0, 2.01]}
    )
    drift_res = _compare_fast_vs_pit(panel, pit_drift, sampled_targets, rtol=1e-10, atol=1e-12)
    assert drift_res["passed"] is False
    assert drift_res["max_abs_diff"] == pytest.approx(0.01)

    # NaN mismatch fails
    pit_nan = pd.DataFrame(
        {"timestamp_utc": [ts[0], ts[1]], "instrument_id": ["SYM-01", "SYM-01"], "factor_value": [1.0, np.nan]}
    )
    nan_res = _compare_fast_vs_pit(panel, pit_nan, sampled_targets, rtol=1e-10, atol=1e-12)
    assert nan_res["passed"] is False
    assert nan_res["nan_mismatch_count"] == 1
