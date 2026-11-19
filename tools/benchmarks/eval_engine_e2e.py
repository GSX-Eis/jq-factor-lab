"""Time the switched evaluator adapters on the S1-C labeled workloads."""

import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from tools.benchmarks import eval_engine_baseline as legacy
from tools.benchmarks.eval_engine_kernels import BASELINE, REPORTS, _validate_metrics
from utils.local_outputs import write_json


def run_benchmark(baseline: Path, repeats: int, timeout: float) -> dict:
    raw = baseline.read_bytes()
    frozen = json.loads(raw)
    if frozen["smoke"] or frozen["seed"] != legacy.SEED:
        raise ValueError("Expected the full-size frozen S1-C baseline")
    report = legacy._report_metadata(False, repeats, timeout)
    report["baseline"] = {"source": str(baseline), "sha256": sha256(raw).hexdigest()}
    report["wall_measurement"] += " Includes adapter panel assembly, statistics and legacy result formatting."
    for name, (adapter, horizons) in legacy.SCENARIOS.items():
        previous = frozen["scenarios"][name]
        old_median = previous["methods"][adapter]["median_wall_seconds"]
        limit = {"daily_cross_section": old_median, "minute_cross_section": 1.0, "nq_time_series": 3.0}[name]
        expected = previous["methods"][legacy.REFERENCE]["attempts"][0]["per_horizon"]
        attempts, failures = [], []
        for number in range(1, repeats + 1):
            print(f"{name} / {adapter} / {number} of {repeats}", flush=True)
            setup, attempt = legacy._attempt(name, adapter, False, timeout)
            attempts.append(attempt)
            if attempt["status"] != "complete":
                failures.append(f"Attempt {number} timed out during {attempt['timeout_stage']}")
                continue
            print(f"  {attempt['wall_seconds']:.6f}s {attempt['per_horizon']}", flush=True)
            for key in ("input_size", "input_metadata", "prepared_valid_pairs"):
                actual = setup[key]
                if key == "prepared_valid_pairs":
                    actual = {str(h): count for h, count in actual.items()}
                if actual != previous[key]:
                    failures.append(f"Attempt {number} differs from frozen S1-C {key}")
            try:
                _validate_metrics(attempt["per_horizon"], expected, horizons)
            except ValueError as exc:
                failures.append(f"Attempt {number}: {exc}")

        elapsed = median(a["wall_seconds"] for a in attempts) if all(
            a["status"] == "complete" for a in attempts
        ) else None
        if elapsed is not None and elapsed > limit:
            failures.append(f"Median {elapsed:.6f}s exceeds {limit:.6f}s")
        report["scenarios"][name] = {
            "input_size": previous["input_size"], "input_metadata": previous["input_metadata"],
            "reference_per_horizon": expected, "adapter": adapter,
            "baseline_median_wall_seconds": old_median, "median_wall_seconds": elapsed, "attempts": attempts,
            "gate": {"limit_seconds": limit, "passed": not failures, "failures": failures},
        }
        print(f"  median={elapsed} limit={limit} failures={failures}", flush=True)

    report["passed"] = all(result["gate"]["passed"] for result in report["scenarios"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=REPORTS / f"perf_s6_e2e_{date.today():%Y%m%d}.json")
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.timeout < float("inf"):
        parser.error("--repeats must be positive and --timeout must be finite and positive")
    report = run_benchmark(args.baseline, args.repeats, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(args.output, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
