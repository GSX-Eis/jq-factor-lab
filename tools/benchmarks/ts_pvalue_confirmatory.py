"""Run NERE-1279 confirmatory calibration, nested production checks and timing."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from math import sqrt
import os
from pathlib import Path
import subprocess
from time import perf_counter, process_time

import numpy as np
from scipy.stats import binomtest, norm

from tools.benchmarks.ts_pvalue_checkpoint import bind_checkpoint, calculation_binding
from tools.benchmarks.ts_pvalue_dense_draws import warp_statistic
from tools.benchmarks.ts_pvalue_features import feature_sample, innovations
from tools.benchmarks.ts_pvalue_nested_draws import nested_p_value
from tools.benchmarks.ts_pvalue_recalibration import Chunk, _git
from tools.benchmarks.ts_pvalue_recalibration_rules import apply_holm, check_scope, power_summary, size_summary
from utils.local_outputs import write_json


_CONFIRMATORY_MODULES = (
    "tools.benchmarks.ts_pvalue_recalibration",
    "tools.benchmarks.ts_pvalue_features",
    "tools.benchmarks.ts_pvalue_nested_draws",
    "tools.benchmarks.ts_pvalue_recalibration_rules",
    "utils.local_outputs",
)


@dataclass(frozen=True)
class Job:
    chunk: Chunk
    feature: str
    rho: float
    parameters: dict
    nested: bool
    n_boot: int
    min_valid: int


def _simulate(job: Job) -> tuple[np.ndarray, float]:
    chunk = job.chunk
    start = process_time()
    result = np.empty(chunk.stop - chunk.start)
    seeds = np.random.SeedSequence([chunk.seed, chunk.index, chunk.phase, chunk.start]).spawn(2)
    data_rng, boot_rng = map(np.random.default_rng, seeds)
    for offset, seed in enumerate(boot_rng.bit_generator.random_raw(len(result))):
        x, y = feature_sample(chunk.scenario, job.feature, job.rho, job.parameters, chunk.size, chunk.horizon,
                              data_rng)
        seed = int(seed)
        if job.nested:
            result[offset] = nested_p_value(x, y, chunk.horizon, seed, job.n_boot, job.min_valid)
        else:
            result[offset] = warp_statistic(x, y, chunk.horizon, seed, pivot=chunk.phase == 0)
    return result, process_time() - start


def _jobs(design: dict, mode: str, pilot: bool, size_override: int | None = None):
    plan = design[mode]
    seed = design["pilot"][mode]["seed"] if pilot else plan["seed"]
    if pilot and size_override == 4096:
        seed = design["pilot"]["probe_4096_seed"]
    if mode == "nested":
        cells = [(design["scenarios"][scenario], size, horizon) for scenario, size, horizon in plan["cells"]]
    else:
        cells = list(product(design["scenarios"].values(), [size_override] if size_override else design["grid_sizes"],
                             design["horizons"]))
    jobs = []
    for i, (spec, size, horizon) in enumerate(cells):
        if mode == "nested":
            stage_sizes = (plan["repetitions_per_cell"],)
        else:
            counts = plan["effect" if spec["rho"] else "null"]
            stage_sizes = (counts["critical"], counts["evaluation"])
        for phase, full_count in enumerate(stage_sizes):
            count = design["pilot"][mode]["per_stage_per_cell"] if pilot else full_count
            batch = count if pilot else plan["chunk_size"]
            for start in range(0, count, batch):
                chunk = Chunk(spec["name"], size, horizon, seed, i, phase, start, min(start + batch, count))
                jobs.append(Job(chunk, spec["feature"], spec["rho"], design["features"].get(spec["feature"], {}),
                                mode == "nested", design["nested"]["n_boot"], design["nested"]["min_valid"]))
    return cells, jobs


def _warm_worker(nested: bool, garch_parameters: dict) -> None:
    grid = np.arange(512)
    x, y = np.sin(grid / 3), np.cos(grid / 5)
    if nested:
        nested_p_value(x, y, 1, 1, 8, 8)
    else:
        warp_statistic(x, y, 1, 1, pivot=True)
        innovations("garch", np.random.default_rng(1), 32, garch_parameters)


def _pilot(design: dict, mode: str, workers: int, size_override: int | None = None) -> dict:
    cells, jobs = _jobs(design, mode, True, size_override)
    _, full = _jobs(design, mode, False, size_override)
    full_counts = {}
    for job in full:
        chunk = job.chunk
        key = chunk.index, chunk.phase
        full_counts[key] = full_counts.get(key, 0) + chunk.stop - chunk.start
    with ProcessPoolExecutor(max_workers=workers, initializer=_warm_worker,
                             initargs=(mode == "nested", design["features"]["garch"])) as pool:
        list(pool.map(int, range(workers)))
        start = perf_counter()
        times = [seconds for _, seconds in pool.map(_simulate, jobs)]
        wall = perf_counter() - start
    cpu = sum(times)
    throughput = min(workers, cpu / wall)
    by_size = {}
    for job, seconds in zip(jobs, times, strict=True):
        c = job.chunk
        estimated_cpu = seconds * full_counts[c.index, c.phase] / (c.stop - c.start)
        group = by_size.setdefault(c.size, dict(pilot_cpu_seconds=0.0, estimated_wall_seconds=0.0))
        group["pilot_cpu_seconds"] += seconds
        group["estimated_wall_seconds"] += estimated_cpu / throughput
    return dict(cells=len(cells), samples_per_cell=design["pilot"][mode]["per_stage_per_cell"],
                wall_seconds=wall, cpu_seconds=cpu, effective_cores=throughput, by_size=by_size,
                estimated_wall_seconds=sum(row["estimated_wall_seconds"] for row in by_size.values()))


def _run(design: dict, mode: str, workers: int, directory: Path, binding: dict):
    cells, jobs = _jobs(design, mode, False)
    bind_checkpoint(directory, calculation_binding(binding, __file__, _CONFIRMATORY_MODULES))
    paths = [directory / f"{j.chunk.index}-{j.chunk.phase}-{j.chunk.start}-{j.chunk.stop}.npy" for j in jobs]
    missing = [(job, path) for job, path in zip(jobs, paths, strict=True) if not path.is_file()]
    print(f"{mode}: {len(jobs) - len(missing)}/{len(jobs)} chunks already present", flush=True)
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_simulate, job): path for job, path in missing}
            for count, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                pending = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                with pending.open("wb") as stream:
                    np.save(stream, future.result()[0], allow_pickle=False)
                pending.replace(path)
                if count % 25 == 0 or count == len(missing):
                    print(f"{mode}: {count}/{len(missing)} new chunks complete", flush=True)
    grouped = {}
    for job, path in zip(jobs, paths, strict=True):
        chunk = job.chunk
        values = np.load(path, allow_pickle=False)
        if values.shape != (chunk.stop - chunk.start,) or values.dtype != np.float64:
            raise ValueError(f"Invalid checkpoint {path}")
        grouped.setdefault((chunk.index, chunk.phase), []).append(values)
    return cells, {key: np.concatenate(parts) for key, parts in grouped.items()}


def _warp_rows(design: dict, cells: list, draws: dict) -> list[dict]:
    rows = []
    for i, (spec, size, horizon) in enumerate(cells):
        critical, observed = draws[i, 0], draws[i, 1]
        sign = int(np.sign(spec["rho"]))
        invalid_critical = int(np.count_nonzero(~np.isfinite(critical)))
        invalid = int(np.count_nonzero(~np.isfinite(observed)))
        if invalid or invalid_critical:
            raise ValueError(f"Unestimable draws in {spec['name']}, T={size}, h={horizon}: {invalid_critical}, {invalid}")
        thresholds = [
            power_summary(critical, observed, alpha, sign) if sign else size_summary(critical, np.abs(observed), alpha)
            for alpha in design["alphas"]
        ]
        rows.append(dict(scenario=spec["name"], feature=spec["feature"], sign=sign, grid_size=size, horizon=horizon,
                         unestimable=invalid, unestimable_critical=invalid_critical, thresholds=thresholds))
    apply_holm(rows)
    return rows


def _difference(rate: float, reference: float, variance: float, margin: float) -> dict:
    delta = rate - reference
    width = norm.ppf(0.995) * sqrt(variance)
    return dict(difference=delta, ci99=[delta - width, delta + width], equivalence_margin=margin,
                equivalent=bool(abs(delta) + width <= margin))


def _nested_rows(design: dict, cells: list, draws: dict, reference: list[dict]) -> dict:
    rule = design["nested"]
    alpha = rule["alpha"]
    rows = []
    for i, (spec, size, horizon) in enumerate(cells):
        p = draws[i, 0]
        invalid = int(np.count_nonzero(~np.isfinite(p)))
        count = int(np.count_nonzero(p <= alpha))
        rate = count / len(p)
        upper = binomtest(count, len(p), alternative="less").proportion_ci(0.99, method="exact").high
        warp_cell = next(row for row in reference if (row["scenario"], row["grid_size"], row["horizon"])
                         == (spec["name"], size, horizon))
        warp = next(row for row in warp_cell["thresholds"] if row["alpha"] == alpha)
        variance = rate * (1 - rate) / len(p) + warp["evaluation_variance"] + warp["critical_variance"]
        difference = _difference(rate, warp["rejection_rate"], variance, rule["cell_equivalence_margin"])
        rows.append(dict(scenario=spec["name"], grid_size=size, horizon=horizon, alpha=alpha, rejected=count,
                         repetitions=len(p), unestimable=invalid, rejection_rate=rate, upper99=upper,
                         warp_rate=warp["rejection_rate"], difference_variance=variance,
                         passes=not invalid and upper <= 1.5 * alpha and difference["equivalent"]) | difference)
    n = len(rows)
    pooled = _difference(sum(row["rejection_rate"] for row in rows) / n, sum(row["warp_rate"] for row in rows) / n,
                         sum(row["difference_variance"] for row in rows) / n**2, rule["pooled_equivalence_margin"])
    return dict(cells=rows, pooled=pooled, passes=all(row["passes"] for row in rows) and pooled["equivalent"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", choices=["pilot", "warp", "nested"], required=True)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    path = args.design.resolve().relative_to(Path.cwd())
    prereg = _git("log", "-1", "--format=%H", "--", str(path))
    if not prereg or subprocess.check_output(["git", "show", f"{prereg}:{path}"]) != path.read_bytes():
        parser.error("Commit the exact preregistration before running")
    diagnostic = json.loads(args.diagnostic.read_text())
    if not diagnostic["route_a"]:
        parser.error("Step 0 did not authorize route A")
    design = json.loads(path.read_text())
    binding = calculation_binding(
        dict(preregistration_commit=prereg, design_sha256=sha256(path.read_bytes()).hexdigest()),
        __file__, _CONFIRMATORY_MODULES,
    )
    commit = _git("rev-parse", "HEAD")
    if args.mode == "pilot":
        warp = _pilot(design, "warp", args.workers)
        probe = _pilot(design, "warp", args.workers, 4096)
        nested = _pilot(design, "nested", args.workers)
        total = warp["estimated_wall_seconds"] + nested["estimated_wall_seconds"]
        pilot = dict(warp=warp, probe_4096=probe, nested=nested, total_estimated_wall_seconds=total,
                     proceed=total <= design["pilot"]["wall_limit_seconds"])
        write_json(args.workspace / "pilot.json", binding | pilot | dict(computation_commit=commit, workers=args.workers))
        print(json.dumps(pilot, indent=2), flush=True)
        return
    pilot = json.loads((args.workspace / "pilot.json").read_text())
    if not pilot["proceed"] or any(pilot.get(key) != value for key, value in binding.items()):
        parser.error("A matching, under-budget timing pilot is required")
    if args.mode == "nested":
        warp = json.loads((args.workspace / "warp.json").read_text())
        if any(warp.get(key) != value for key, value in binding.items()):
            parser.error("Nested comparison requires the matching warp-speed result")
    start = perf_counter()
    cells, draws = _run(design, args.mode, args.workers, args.workspace / f"{args.mode}-cells", binding)
    if args.mode == "warp":
        result = dict(cells=_warp_rows(design, cells, draws))
    else:
        nested = _nested_rows(design, cells, draws, warp["cells"])
        result = dict(nested=nested, decision=check_scope(warp["cells"], nested["passes"]))
    write_json(args.workspace / f"{args.mode}.json", binding | result | dict(computation_commit=commit,
               workers=args.workers, seconds=perf_counter() - start))


if __name__ == "__main__":
    main()
