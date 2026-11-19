"""Exercise each benchmark path with tiny, in-memory inputs."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_eval_engine_perf_baseline_smoke(tmp_path: Path) -> None:
    output = tmp_path / "baseline.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "tools.benchmarks.eval_engine_baseline",
         "--smoke", "--repeats", "1", "--output", str(output)],
        cwd=root, check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["seed"] == 791
    assert report["machine"].keys() >= {"cpu", "python", "numpy", "pandas"}
    assert report["memory_measurement"]
    expected = {
        "daily_cross_section": ("AlphalensRunner", 12, 12, [1]),
        "minute_cross_section": ("BarCrossSectionalEvaluator", 24, 5, [1]),
        "nq_time_series": ("TimeSeriesEvaluator", 96, 1, [1, 5, 15, 60]),
    }
    assert report["scenarios"].keys() == expected.keys()
    for name, (evaluator, time_points, assets, horizons) in expected.items():
        scenario = report["scenarios"][name]
        assert scenario["input_size"] == {
            "rows": time_points * assets,
            "time_points": time_points,
            "assets": assets,
            "horizons": len(horizons),
            "label_rows": time_points * assets * len(horizons),
        }
        assert scenario["methods"].keys() == {evaluator, "scipy_spearman_reference"}
        for method in scenario["methods"].values():
            assert method["status"] == "complete"
            assert "median_wall_seconds" in method
            assert len(method["attempts"]) == 1
            run = method["attempts"][0]
            assert run.keys() >= {"wall_seconds", "peak_rss_bytes"}
            assert [entry["holding_period"] for entry in run["per_horizon"]] == horizons

        legacy = scenario["methods"][evaluator]["attempts"][0]["per_horizon"]
        reference = scenario["methods"]["scipy_spearman_reference"]["attempts"][0]["per_horizon"]
        for actual, target in zip(legacy, reference, strict=True):
            assert 0 < actual["valid_pairs"] == target["valid_pairs"] < scenario["input_size"]["rows"]
            assert actual["ic_groups"] == target["ic_groups"] > 0
            assert actual["mean_ic"] == pytest.approx(target["mean_ic"], rel=0, abs=1e-12)
        if name == "nq_time_series":
            assert [entry["valid_pairs"] for entry in legacy] == [94, 90, 80, 35]


def test_daily_benchmark_rejects_pair_loss_even_when_ic_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    from backtest_and_analysis.alphalens_runner import AlphalensRunner, MaxLossExceededError
    from tools.benchmarks import eval_engine_baseline as benchmark

    prepare = AlphalensRunner._prepare_factor_data_with_precomputed_returns

    def drop_one_pair(self, factor_series: pd.Series, forward_returns: pd.DataFrame) -> pd.DataFrame:
        cleaned = prepare(self, factor_series, forward_returns)
        return cleaned.drop(index=(pd.Timestamp("2020-01-08"), "SYN0002"))

    monkeypatch.setattr(AlphalensRunner, "_prepare_factor_data_with_precomputed_returns", drop_one_pair)
    setup, evaluate = benchmark._prepare("daily_cross_section", "AlphalensRunner", smoke=True)
    assert setup["prepared_valid_pairs"] == {1: 142}
    with pytest.raises(MaxLossExceededError, match="retained=141/142"):
        evaluate()


def test_preparation_timeout_preserves_completed_benchmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.benchmarks import eval_engine_baseline as benchmark

    output = tmp_path / "baseline.json"
    original_attempt = benchmark._attempt

    def attempt(
        scenario: str, method: str, smoke: bool, timeout: float,
    ) -> tuple[dict | None, dict]:
        if scenario == "nq_time_series" and method == "TimeSeriesEvaluator":
            with monkeypatch.context() as patch:
                patch.setattr(benchmark.Connection, "poll", lambda self, timeout: False)
                return original_attempt(scenario, method, smoke, timeout)
        return original_attempt(scenario, method, smoke, timeout)

    monkeypatch.setattr(benchmark, "_attempt", attempt)
    monkeypatch.setattr(sys, "argv", [
        "eval_engine_baseline", "--smoke", "--repeats", "1", "--output", str(output),
    ])
    benchmark.main()

    scenarios = json.loads(output.read_text(encoding="utf-8"))["scenarios"]
    for name in ("daily_cross_section", "minute_cross_section"):
        for method in scenarios[name]["methods"].values():
            (run,) = method["attempts"]
            assert run["status"] == "complete"
            assert run["per_horizon"]
    nq = scenarios["nq_time_series"]
    method = nq["methods"]["TimeSeriesEvaluator"]
    assert method["status"] == "timeout"
    assert method["median_wall_seconds"] is None
    (run,) = method["attempts"]
    assert run == {
        "status": "timeout", "timeout_stage": "preparation", "completed_stage": None,
        "wall_seconds": None, "peak_rss_bytes": None, "setup_peak_rss_bytes": None,
        "per_horizon": [],
    }
    assert nq["methods"][benchmark.REFERENCE]["status"] == "complete"
    assert nq["prepared_valid_pairs"] == {"1": 94, "5": 90, "15": 80, "60": 35}
