"""Run the kernel benchmark on tiny fixtures and enforce frozen sample parity."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.benchmarks import eval_engine_kernels as benchmark


@pytest.mark.perf_gate
def test_eval_engine_kernels_perf_smoke(tmp_path: Path) -> None:
    output = tmp_path / "kernels.json"
    subprocess.run(
        [sys.executable, "-m", "tools.benchmarks.eval_engine_kernels",
         "--smoke", "--repeats", "1", "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["seed"] == 791
    assert report["machine"].keys() >= {"cpu", "python", "numpy", "pandas", "scipy"}
    assert report["wall_measurement"] and report["memory_measurement"]
    assert report["passed"]
    expected = {
        "daily_cross_section": (12, 12, [1]),
        "minute_cross_section": (24, 5, [1]),
        "nq_time_series": (96, 1, [1, 5, 15, 60]),
    }
    assert report["scenarios"].keys() == expected.keys()
    for name, (time_points, assets, horizons) in expected.items():
        scenario = report["scenarios"][name]
        assert scenario["input_size"] == {
            "rows": time_points * assets, "time_points": time_points, "assets": assets,
            "horizons": len(horizons), "label_rows": time_points * assets * len(horizons),
        }
        method = scenario["methods"]["numpy_kernels"]
        assert method["status"] == "complete"
        assert method["median_wall_seconds"] > 0
        attempt, = method["attempts"]
        assert attempt["peak_rss_bytes"] >= attempt["setup_peak_rss_bytes"] > 0
        assert scenario["quantile_profile"]["quantile_peak_allocated_bytes"] > 0
        assert [row["holding_period"] for row in attempt["per_horizon"]] == horizons
        for actual, reference in zip(attempt["per_horizon"], scenario["reference_per_horizon"], strict=True):
            assert actual["valid_pairs"] == reference["valid_pairs"] > 0
            assert abs(actual["mean_ic"] - reference["mean_ic"]) <= 1e-12


@pytest.mark.parametrize("lost_pair", [False, True])
def test_eval_engine_perf_gate_frozen_multihorizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost_pair: bool,
) -> None:
    name = "nq_time_series"
    evaluator, horizons = benchmark.legacy.SCENARIOS[name]
    monkeypatch.setattr(benchmark.legacy, "SCENARIOS", {name: (evaluator, horizons)})
    monkeypatch.setattr(benchmark.legacy, "_cpu_name", lambda: "test CPU")
    counts = dict(zip(horizons, (94, 90, 80, 35), strict=True))
    setup = {"input_size": {"rows": 96}, "input_metadata": {}, "prepared_valid_pairs": counts}
    reference = [
        {"holding_period": h, "valid_pairs": count, "ic_groups": 1, "mean_ic": 0.5}
        for h, count in counts.items()
    ]
    frozen = tmp_path / "baseline.json"
    frozen.write_text(json.dumps({
        "seed": 791, "smoke": False, "machine": {},
        "scenarios": {name: {**setup, "methods": {
            method: {
                "median_wall_seconds": 4.0,
                "attempts": [{"peak_rss_bytes": 1024, "per_horizon": reference}],
            }
            for method in (evaluator, benchmark.legacy.REFERENCE)
        }}},
    }), encoding="utf-8")

    actual = [row.copy() for row in reference]
    if lost_pair:
        actual[1]["valid_pairs"] -= 1

    def attempt(scenario, method, smoke, timeout, *, prepare):
        if method == benchmark.QUANTILE_PROFILE:
            return setup, {"status": "complete", "quantile_peak_allocated_bytes": 512}
        return setup, {"status": "complete", "wall_seconds": 2.0, "per_horizon": actual}

    monkeypatch.setattr(benchmark.legacy, "_attempt", attempt)
    report = benchmark.run_benchmark(smoke=False, repeats=1, timeout=10, baseline_path=frozen)
    assert report["passed"] is not lost_pair
    failures = report["scenarios"][name]["gate"]["failures"]
    if lost_pair:
        assert failures == ["attempt 1: holding period 5: valid_pairs 89 != 90"]
    else:
        assert not failures
