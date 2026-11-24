"""NERE-1279 local synthetic calibration with per-chunk restart checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from tools.benchmarks.ts_pvalue_checkpoint import bind_checkpoint, calculation_binding
from tools.benchmarks.ts_pvalue_dense_draws import legacy_sample, warp_statistic


@dataclass(frozen=True)
class Chunk:
    scenario: str
    size: int
    horizon: int
    seed: int
    index: int
    phase: int
    start: int
    stop: int


def _warp_chunk(job: Chunk) -> np.ndarray:
    result = np.empty(job.stop - job.start)
    for index, repetition in enumerate(range(job.start, job.stop)):
        seeds = np.random.SeedSequence([job.seed, job.index, job.phase, repetition]).spawn(2)
        x, y = legacy_sample(job.scenario, job.size, job.horizon, np.random.default_rng(seeds[0]))
        boot_seed = int(seeds[1].generate_state(1, dtype=np.uint64)[0])
        result[index] = warp_statistic(x, y, job.horizon, boot_seed, pivot=job.phase == 0)
    return result


def _run_chunks(jobs: list[Chunk], directory: Path, workers: int, binding: dict) -> dict:
    bind_checkpoint(directory, calculation_binding(binding, __file__))
    paths = {job: directory / f"{job.index}-{job.phase}-{job.start}-{job.stop}.npy" for job in jobs}
    missing = [job for job, path in paths.items() if not path.exists()]
    print(f"{len(jobs) - len(missing)}/{len(jobs)} chunks already present", flush=True)
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_warp_chunk, job): job for job in missing}
            for done, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                path = paths[job]
                pending = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                with pending.open("wb") as stream:
                    np.save(stream, future.result(), allow_pickle=False)
                pending.replace(path)
                if done % 25 == 0 or done == len(missing):
                    print(f"{done}/{len(missing)} new chunks complete", flush=True)
    draws = {}
    for job, path in paths.items():
        values = np.load(path, allow_pickle=False)
        if values.shape != (job.stop - job.start,) or values.dtype != np.float64:
            raise ValueError(f"Invalid checkpoint {path}")
        draws.setdefault((job.index, job.phase), []).append(values)
    return {key: np.concatenate(values) for key, values in draws.items()}


def _write_report(output: Path, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _diagnostic(design: dict, checkpoint: Path, binding: dict, workers: int) -> dict:
    jobs = []
    size, seed, alpha = design["grid_size"], design["seed"], design["alpha"]
    for index, (scenario, horizon) in enumerate(design["cells"]):
        for phase, key in enumerate(("critical_repetitions_per_cell", "evaluation_repetitions_per_cell")):
            repetitions = design[key]
            jobs.extend(
                Chunk(scenario, size, horizon, seed, index, phase, start, min(start + 20000, repetitions))
                for start in range(0, repetitions, 20000)
            )
    draws = _run_chunks(jobs, checkpoint, workers, binding)
    cells = []
    for index, (scenario, horizon) in enumerate(design["cells"]):
        critical, observed = draws[index, 0], draws[index, 1]
        finite = critical[np.isfinite(critical)]
        cutoff = float(np.quantile(finite, 1 - alpha, method="inverted_cdf")) if finite.size else None
        rejected = int(np.count_nonzero(np.abs(observed) > cutoff)) if cutoff is not None else 0
        cells.append(dict(scenario=scenario, grid_size=size, horizon=horizon, critical_value=cutoff,
                          rejected=rejected, rejection_rate=rejected / len(observed),
                          unestimable_critical=len(critical) - len(finite),
                          unestimable=int(np.count_nonzero(~np.isfinite(observed)))))
    total = sum(cell["rejected"] for cell in cells)
    available = all(not (cell["unestimable"] or cell["unestimable_critical"]) for cell in cells)
    return dict(cells=cells, rejected=total, rejection_rate=total / design["decision"]["pooled_denominator"],
                route_a=available and total <= design["decision"]["route_a_if_rejections_at_most"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    path = args.design.resolve().relative_to(Path.cwd())
    prereg = _git("log", "-1", "--format=%H", "--", str(path))
    if not prereg or subprocess.check_output(["git", "show", f"{prereg}:{path}"]) != path.read_bytes():
        parser.error("Commit the exact preregistration before running")
    binding = calculation_binding(
        dict(preregistration_commit=prereg, design_sha256=sha256(path.read_bytes()).hexdigest()), __file__,
    )
    design = json.loads(path.read_text())
    start = perf_counter()
    result = _diagnostic(design, args.checkpoint, binding, args.workers)
    _write_report(args.output, binding | dict(computation_commit=_git("rev-parse", "HEAD"),
                  design=design, workers=args.workers, seconds=perf_counter() - start) | result)


if __name__ == "__main__":
    main()
