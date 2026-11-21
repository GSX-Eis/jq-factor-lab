"""Gate the NumPy kernels against the frozen S1-C evaluation benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import tracemalloc

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation.kernels import bucket_sums, quantile_buckets, spearman
from tools.benchmarks import eval_engine_baseline as legacy
from utils.local_outputs import write_json

KERNELS = "numpy_kernels"
QUANTILE_PROFILE = "quantile_profile"
QUANTILES = 5
REPORTS = Path("reports/NERE-791")
BASELINE = REPORTS / "perf_baseline_20261118.json"


def _prepare_kernels(
    scenario: str, method: str, smoke: bool,
) -> tuple[dict, Callable[[], dict]]:
    _, horizons = legacy.SCENARIOS[scenario]
    rng = np.random.default_rng(legacy.SEED)
    time_series = scenario == "nq_time_series"
    axis = 0 if time_series else 1
    daily = scenario == "daily_cross_section"
    if daily:
        factor, _, returns = legacy._daily_inputs(rng, smoke)
        timestamps, assets = factor.shape
        label_rows = len(returns)
        counts = {1: int(np.isfinite(factor.to_numpy()).sum())}
        metadata = {"calendar": "synthetic weekdays", "holding_period_unit": "sessions"}

        def panel_arrays() -> tuple[np.ndarray, np.ndarray]:
            keys = pd.MultiIndex.from_product(factor.axes)
            labels = returns["1D"].reindex(keys).to_numpy().reshape(timestamps, assets, 1)
            return factor.to_numpy(), labels
    else:
        labels, timestamps, assets, metadata = legacy._bar_inputs(rng, time_series, smoke, horizons)
        label_rows = len(labels)
        metadata["holding_period_unit"] = "1min bars"
        valid = labels.factor_value.notna() & labels.return_pct.notna()
        counts = {int(h): int(n) for h, n in labels.loc[valid, "horizon_bars"].value_counts().items()}

        def panel_arrays() -> tuple[np.ndarray, np.ndarray]:
            # _bar_inputs supplies a dense grid ordered by instrument, timestamp, then horizon.
            shape = (assets, timestamps, len(horizons))
            factors = labels["factor_value"].to_numpy().reshape(shape)
            returns = labels["return_pct"].to_numpy().reshape(shape)
            return factors[:, :, 0].T, returns.transpose(1, 0, 2)

    def evaluate() -> dict:
        start = perf_counter()
        factors, returns = panel_arrays()
        stages = {"panel": perf_counter() - start}
        if method == QUANTILE_PROFILE:
            tracemalloc.start()
            try:
                quantile_buckets(factors, QUANTILES, axis=axis)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return {"per_horizon": [], "quantile_peak_allocated_bytes": peak}

        start = perf_counter()
        metrics = []
        for index, horizon in enumerate(horizons):
            result = spearman(factors, returns[:, :, index], axis=axis, min_samples=2 if daily else 3)
            values = result.values[np.isfinite(result.values)]
            metrics.append(legacy._metric(
                horizon, int(result.counts.sum()), values.mean() if values.size else np.nan, int(values.size),
            ))
        stages["spearman"] = perf_counter() - start
        if not time_series:
            start = perf_counter()
            quantiles = quantile_buckets(factors, QUANTILES, axis=axis)
            stages["quantile_buckets"] = perf_counter() - start
            start = perf_counter()
            for index in range(len(horizons)):
                bucket_sums(returns[:, :, index], quantiles.labels, QUANTILES, axis=axis)
            stages["bucket_sums"] = perf_counter() - start
        return {"per_horizon": metrics, "stage_seconds": stages}

    return {
        "input_size": {
            "rows": timestamps * assets, "time_points": timestamps, "assets": assets,
            "horizons": len(horizons), "label_rows": label_rows,
        },
        "input_metadata": metadata,
        "prepared_valid_pairs": counts,
    }, evaluate


def _validate_metrics(actual: list[dict], reference: list[dict], horizons: tuple[int, ...]) -> None:
    if any(tuple(row["holding_period"] for row in rows) != horizons for rows in (actual, reference)):
        raise ValueError(f"Expected per-horizon metrics for {horizons}")
    for measured, expected in zip(actual, reference, strict=True):
        horizon = measured["holding_period"]
        for field in ("valid_pairs", "ic_groups"):
            if measured[field] != expected[field] or expected[field] <= 0:
                raise ValueError(f"holding period {horizon}: {field} {measured[field]} != {expected[field]}")
        value, target = measured["mean_ic"], expected["mean_ic"]
        if value is None or target is None or not np.isfinite(value - target) or abs(value - target) > 1e-12:
            raise ValueError(f"holding period {horizon}: mean_ic {value} differs from {target} by more than 1e-12")


def run_benchmark(*, smoke: bool, repeats: int, timeout: float, baseline_path: Path) -> dict:
    report = legacy._report_metadata(smoke, repeats, timeout)
    report["wall_measurement"] += (
        "; includes panel assembly and Spearman in all scenarios, plus quantile and bucket sums in cross-section"
    )
    report["quantile_memory_measurement"] = (
        "Peak bytes traced by tracemalloc during five-quantile assignment in a separate fresh worker; "
        "excludes existing inputs and is not process RSS. NQ profiles descriptive full-sample quantiles."
    )
    limits = {"minute_cross_section": 1.0, "nq_time_series": 3.0}
    if smoke:
        report["baseline"] = {"source": "live SciPy reference on smoke inputs"}
    else:
        raw = baseline_path.read_bytes()
        frozen = json.loads(raw)
        if frozen["seed"] != legacy.SEED or frozen["smoke"]:
            raise ValueError("Expected full-size S1-C baseline with the frozen seed")
        report["baseline"] = {
            "source": str(baseline_path), "sha256": sha256(raw).hexdigest(), "machine": frozen["machine"],
        }

    for name, (evaluator, horizons) in legacy.SCENARIOS.items():
        if smoke:
            limit = None
            expected_setup, reference_attempt = legacy._attempt(name, legacy.REFERENCE, True, timeout)
            reference = reference_attempt["per_horizon"]
            legacy_summary = None
        else:
            scenario = frozen["scenarios"][name]
            expected_setup = {key: scenario[key] for key in (
                "input_size", "input_metadata", "prepared_valid_pairs",
            )}
            expected_setup["prepared_valid_pairs"] = {
                int(horizon): count for horizon, count in scenario["prepared_valid_pairs"].items()
            }
            methods = scenario["methods"]
            reference = methods[legacy.REFERENCE]["attempts"][0]["per_horizon"]
            old_median = methods[evaluator]["median_wall_seconds"]
            if old_median is None or not np.isfinite(old_median) or old_median <= 0:
                raise ValueError(f"{name}: S1-C legacy median required")
            legacy_summary = {
                method: {
                    "median_wall_seconds": methods[method]["median_wall_seconds"],
                    "peak_rss_bytes": max(a["peak_rss_bytes"] for a in methods[method]["attempts"]),
                }
                for method in (evaluator, legacy.REFERENCE)
            }
            limit = old_median if name == "daily_cross_section" else limits[name]

        failures = []
        attempts = []
        observed_setup = None
        for repeat in range(repeats):
            print(f"{name} / {KERNELS} / {repeat + 1} of {repeats}", flush=True)
            setup, attempt = legacy._attempt(name, KERNELS, smoke, timeout, prepare=_prepare_kernels)
            attempts.append(attempt)
            if attempt["status"] != "complete":
                failures.append(f"attempt {repeat + 1} timed out during {attempt['timeout_stage']}")
                break
            observed_setup = setup
            if setup != expected_setup:
                failures.append(
                    f"attempt {repeat + 1} inputs differ from S1-C: got {setup!r}, expected {expected_setup!r}"
                )
            try:
                _validate_metrics(attempt["per_horizon"], reference, horizons)
            except ValueError as exc:
                failures.append(f"attempt {repeat + 1}: {exc}")

        complete = len(attempts) == repeats and attempts[-1]["status"] == "complete"
        elapsed = median(a["wall_seconds"] for a in attempts) if complete else None
        if limit is not None and elapsed is not None and elapsed > limit:
            failures.append(f"median {elapsed:.6f}s exceeds {limit:.6f}s")

        print(f"{name} / {QUANTILE_PROFILE}", flush=True)
        profile_setup, profile = legacy._attempt(name, QUANTILE_PROFILE, smoke, timeout, prepare=_prepare_kernels)
        if profile["status"] != "complete":
            failures.append(f"quantile profile timed out during {profile['timeout_stage']}")
        elif profile_setup != expected_setup:
            failures.append(
                f"quantile profile inputs differ from S1-C: got {profile_setup!r}, expected {expected_setup!r}"
            )

        report["scenarios"][name] = {
            **(observed_setup or profile_setup or dict.fromkeys(("input_size", "input_metadata", "prepared_valid_pairs"))),
            "reference_per_horizon": reference,
            "baseline_methods": legacy_summary,
            "methods": {KERNELS: {
                "status": "complete" if complete else "timeout",
                "median_wall_seconds": elapsed,
                "attempts": attempts,
            }},
            "quantile_profile": profile,
            "gate": {"limit_seconds": limit, "passed": not failures, "failures": failures},
        }

    report["passed"] = all(scenario["gate"]["passed"] for scenario in report["scenarios"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Compare tiny inputs against live SciPy without timing gates")
    parser.add_argument("--baseline", type=Path, default=BASELINE, help="Path to the frozen full-size S1-C JSON")
    parser.add_argument("--repeats", type=int, default=3, help="Fresh timing workers per scenario")
    parser.add_argument("--timeout", type=float, default=180, help="Worker evaluation deadline in seconds")
    parser.add_argument("--output", type=Path, help="Override the dated JSON report path")
    args = parser.parse_args()
    if args.repeats < 1 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--repeats must be positive and --timeout must be finite and positive")

    report = run_benchmark(
        smoke=args.smoke, repeats=args.repeats, timeout=args.timeout, baseline_path=args.baseline,
    )
    name = "perf_kernels_smoke" if args.smoke else "perf_kernels"
    output = args.output or REPORTS / f"{name}_{date.today():%Y%m%d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(output, flush=True)
    if not report["passed"]:
        for scenario, result in report["scenarios"].items():
            for failure in result["gate"]["failures"]:
                print(f"{scenario}: {failure}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
