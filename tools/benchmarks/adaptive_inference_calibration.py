"""Run the frozen NERE-936 independent synthetic calibration matrix."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
import json
from math import asin, isfinite, pi, sqrt
import os
from pathlib import Path
import platform
from typing import Literal

import numpy as np
import scipy

from backtest_and_analysis.evaluation.calibration import (
    _CoverageCounts,
    _adaptive_calibration_observation,
    _adaptive_domain_mismatches,
    _candidate_adaptive_domain,
    _gaussian_series,
    adjust_pvalues,
    coverage_passes,
)
from backtest_and_analysis.evaluation.cross_section import bartlett_hac, hac_p_value
from backtest_and_analysis.evaluation.kernels import spearman
from backtest_and_analysis.evaluation.time_series import _bootstrap_spearman
from backtest_and_analysis.evaluation.types import (
    BootstrapInterval, DEFAULT_BOOTSTRAP_REPLICATES, DEFAULT_MIN_VALID_REPLICATES, HACInterval,
)
from utils.local_outputs import write_json

ALPHA = .05
N_CANDIDATES = 6
GENERALIZED = "generalized_ar_heteroskedastic"


@dataclass(frozen=True)
class Design:
    scenario: str
    factor_ar: float = 0.
    return_ar: float = 0.
    horizon: int = 1
    gap: int = 0
    heteroskedastic: bool = False
    pearson_rho: float = 0.
    fixed_lags: tuple[int, ...] = (0,)
    n_times: int = 512
    n_assets: int = 32

    def sample(
        self, rng: np.random.Generator, n_alternatives: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times, assets = self.n_times, self.n_assets
        size = times + self.horizon - 1
        if self.pearson_rho:
            factor = _gaussian_series(rng, times, assets, 0.)
            noise = _gaussian_series(rng, size, assets, 0.)
            p = self.pearson_rho
            returns = p * factor + sqrt(1 - p**2) * noise
        else:
            returns = _gaussian_series(rng, size, assets, self.return_ar)
            factor = _gaussian_series(rng, times, assets, self.factor_ar)
            if n_alternatives is not None:
                common = factor
                factor = np.stack([
                    sqrt(.65) * common + sqrt(.35) * _gaussian_series(rng, times, assets, self.factor_ar)
                    for _ in range(N_CANDIDATES)
                ])
                if n_alternatives:
                    signal = _gaussian_series(rng, size, assets, 0.)
                    factor[:n_alternatives] += .5 * signal[:times]
                    returns += .35 * signal

        if self.heteroskedastic:
            scale = np.where(np.arange(size) // 32 % 2 == 0, 1., 4.)
            factor *= scale[:times, None]
            returns *= scale[:, None]
        held = np.lib.stride_tricks.sliding_window_view(returns, self.horizon, axis=0).sum(axis=-1)
        valid = np.broadcast_to((np.arange(times) % 17 < 17 - self.gap)[:, None], held.shape)
        return factor, held, valid


CORE = (
    Design("iid"),
    Design("strong_ar", .8, .8, fixed_lags=(0, 16)),
    Design("overlap", .8, horizon=5, fixed_lags=(4, 16)),
    Design("gaps", gap=3),
    Design("heteroskedastic", heteroskedastic=True),
    Design("combined", .8, .8, horizon=5, gap=3, heteroskedastic=True, fixed_lags=(16,)),
    Design("positive_effect", pearson_rho=.35),
    Design("negative_effect", pearson_rho=-.35),
)


@dataclass(frozen=True)
class Profile:
    seed: int
    repetitions: int
    fixed_boot: int
    adaptive_boot: int
    adaptive_min_valid: int


QUICK = Profile(93673, 3, 19, 19, 18)
FORMAL = Profile(93620261120, 500, 399, DEFAULT_BOOTSTRAP_REPLICATES, DEFAULT_MIN_VALID_REPLICATES)
S5_QUICK = Profile(93679, 3, 19, 19, 18)
S5_FORMAL = Profile(93620261125, 500, 399, 999, 900)


@dataclass(frozen=True)
class Cell:
    section: int
    index: int
    design: Design
    n_alternatives: int | None = None

    @property
    def cell_id(self) -> str:
        return f"{self.section}:{self.index:03d}"

    def rng(self, profile: Profile, repetition: int, stream: int) -> np.random.Generator:
        seed = np.random.SeedSequence(profile.seed, spawn_key=(self.section, self.index, repetition, stream))
        return np.random.Generator(np.random.PCG64(seed))


def _matrix(*, s5: bool = False) -> list[Cell]:
    matrix = [Cell(0, index, design) for index, design in enumerate(CORE)]
    matrix.extend(Cell(1, index, replace(design, n_assets=1)) for index, design in enumerate(CORE))
    historical_times = (128, 256, 512, 1024)
    grids = (
        ((2, (256, 512, 1024), (32, 64)), (3, (512, 1024), (1,))) if s5
        else ((2, historical_times, (16, 32, 64)), (3, historical_times, (1,)))
    )
    for section, grid_sizes, widths in grids:
        designs = product(grid_sizes, widths, (1, 5, 20), (0, 3, 5))
        for index, (times, width, h, gap) in enumerate(designs):
            matrix.append(Cell(section, index, Design(
                GENERALIZED, .8, .8, horizon=h, gap=gap, heteroskedastic=True,
                fixed_lags=(max(16, h - 1),), n_times=times, n_assets=width,
            )))
    for index, (design, alternatives) in enumerate(product((CORE[0], CORE[5]), (0, 2))):
        matrix.append(Cell(4, index, design, alternatives))
    return matrix


def _truth(design: Design, cross_section: bool) -> float:
    rho = 6 / pi * asin(design.pearson_rho / 2)
    if not cross_section:
        return rho
    tau = 2 / pi * asin(design.pearson_rho)
    return ((design.n_assets - 2) * rho + 3 * tau) / (design.n_assets + 1)


def _interval_reason(interval: HACInterval | BootstrapInterval) -> str | None:
    if interval.reason is not None:
        return interval.reason
    if not isfinite(interval.lower) or not isfinite(interval.upper) or interval.lower > interval.upper:
        return "invalid_interval_endpoints"
    return None


def _coverage(cell: Cell, profile: Profile, *, s5: bool = False) -> dict:
    design = cell.design
    cross_section = cell.section in (0, 2)
    version = "cs_hac_adaptive_v1" if cross_section else "ts_boot_adaptive_v1"
    fixed_windows = tuple(lag + int(not cross_section) for lag in design.fixed_lags)
    truth = _truth(design, cross_section)
    domain = _candidate_adaptive_domain(
        cross_section=cross_section, scenario=design.scenario, horizon=design.horizon, gap=design.gap, s5=s5,
    )
    windows = ("auto", *fixed_windows)
    counts = {window: _CoverageCounts() for window in windows}
    unavailable = {window: Counter() for window in windows}
    reverse = Counter()
    selected_windows = Counter()
    mismatches = Counter()
    in_domain_count = in_domain_covered = 0
    for repetition in range(profile.repetitions):
        x, y, valid = design.sample(cell.rng(profile, repetition, 0))
        ic = spearman(x, y, valid=valid, axis=int(cross_section), min_samples=3)
        finite = np.isfinite(ic.values)
        point = float(ic.values[finite].mean()) if finite.any() else float("nan")
        widths = np.where(finite, ic.counts, 0) if cross_section else None
        for index, window in enumerate(windows):
            adaptive = window == "auto"
            if cross_section:
                interval = bartlett_hac(
                    ic.values, lag=window, min_lag=design.horizon - 1,
                    alpha=ALPHA, calibration_scenario=design.scenario,
                )
            else:
                seed = int(cell.rng(profile, repetition, index + 1).integers(0, 2**63, dtype=np.int64))
                interval = _bootstrap_spearman(
                    x[:, 0], y[:, 0], valid[:, 0], point=ic, min_samples=3,
                    block_length=window, min_block_length=design.horizon, seed=seed,
                    n_boot=profile.adaptive_boot if adaptive else profile.fixed_boot,
                    min_valid_replicates=profile.adaptive_min_valid if adaptive else profile.fixed_boot,
                    alpha=ALPHA, calibration_scenario=design.scenario, intervals_verified=True,
                )
            reason = _interval_reason(interval)
            if reason is None:
                counts[window].add(point, interval.lower, interval.upper, truth)
                reverse[window] += int((truth > 0 and interval.upper < 0) or (truth < 0 and interval.lower > 0))
            else:
                unavailable[window][reason] += 1
                counts[window].add(point, float("nan"), float("nan"), truth)
            if adaptive:
                if reason is None:
                    selected_windows[interval.lag if cross_section else interval.block_length] += 1
                observation = _adaptive_calibration_observation(
                    interval, point_estimate=point, horizon=design.horizon, ic_method="spearman",
                    cross_section_sizes=widths, intervals_verified=True,
                )
                failures = ("observation_unavailable",) if observation is None else _adaptive_domain_mismatches(
                    domain, observation,
                )
                mismatches.update(failures)
                if not failures:
                    in_domain_count += 1
                    in_domain_covered += int(interval.lower <= truth <= interval.upper)

    repetitions = profile.repetitions
    estimates = {}
    for window, count in counts.items():
        within = coverage_passes(count.covered, repetitions)
        directions_ok = not truth or (
            20 * count.point_sign_correct >= 19 * repetitions and 20 * count.correct_direction >= 19 * repetitions
        )
        estimates[window] = dict(
            window=window, **asdict(count), coverage=count.covered / repetitions,
            unavailable_count=sum(unavailable[window].values()), unavailable_reasons=dict(unavailable[window]),
            reverse_direction=reverse[window], within_tolerance=within,
            direction_passes=bool(directions_ok) if truth else None,
            interval_passes=bool(within and not unavailable[window]),
        )
    adaptive = estimates.pop("auto")
    adaptive.update(
        method_version=version, selected_windows=dict(sorted(selected_windows.items())),
        domain_id=domain.domain_id, in_domain_count=in_domain_count, in_domain_covered=in_domain_covered,
        out_of_domain_reasons=dict(sorted(mismatches.items())),
        domain_passes=in_domain_count == repetitions and (not s5 or in_domain_covered == adaptive["covered"]),
    )
    return dict(
        cell_id=cell.cell_id, seed_spawn_prefix=[cell.section, cell.index], repetitions=repetitions,
        **asdict(design), true_ic=truth, method="hac" if cross_section else "bootstrap",
        fixed=list(estimates.values()), adaptive=adaptive, passes=(
            adaptive["interval_passes"] and adaptive["domain_passes"] and adaptive["direction_passes"] is not False
        ),
    )


def _family(cell: Cell, profile: Profile, *, s5: bool = False) -> dict:
    design = cell.design
    alternatives = cell.n_alternatives
    assert alternatives is not None
    repetitions = profile.repetitions
    fixed_lag = design.fixed_lags[-1]
    threshold = ALPHA + 3 * sqrt(ALPHA * (1 - ALPHA) / repetitions)
    results = {}
    windows = (fixed_lag, "auto")
    metrics = {window: {method: Counter() for method in ("raw", "holm", "bh")} for window in windows}
    unavailable = {window: Counter() for window in windows}
    unavailable_families = Counter()
    domain = None
    if s5:
        domain = _candidate_adaptive_domain(
            cross_section=True, scenario=design.scenario, horizon=design.horizon, gap=design.gap, s5=True,
        )
    mismatches = Counter()
    in_domain_count = 0
    for repetition in range(repetitions):
        factors, returns, valid = design.sample(cell.rng(profile, repetition, 0), alternatives)
        ic = [spearman(x, returns, valid=valid, axis=1, min_samples=3) for x in factors]
        for window in windows:
            p_values = []
            family_mismatches = set()
            for candidate in ic:
                values = candidate.values
                interval = bartlett_hac(
                    values, lag=window, min_lag=design.horizon - 1, alpha=ALPHA,
                    calibration_scenario=design.scenario,
                )
                finite = values[np.isfinite(values)]
                point = float(finite.mean()) if finite.size else float("nan")
                p, _ = hac_p_value(interval, point)
                reason = _interval_reason(interval)
                if reason is not None or p is None or not isfinite(p) or not 0 <= p <= 1:
                    unavailable[window][reason or "invalid_p_value"] += 1
                else:
                    p_values.append(p)
                if window == "auto" and domain is not None:
                    observation = _adaptive_calibration_observation(
                        interval, point_estimate=point, horizon=design.horizon, ic_method="spearman",
                        cross_section_sizes=np.where(np.isfinite(values), candidate.counts, 0),
                    )
                    failures = ("observation_unavailable",) if observation is None else _adaptive_domain_mismatches(
                        domain, observation,
                    )
                    family_mismatches.update(failures)
            if window == "auto" and domain is not None:
                mismatches.update(family_mismatches)
                in_domain_count += int(not family_mismatches)
            if len(p_values) != N_CANDIDATES:
                unavailable_families[window] += 1
                continue
            for method, counts in metrics[window].items():
                adjusted = np.asarray(p_values) if method == "raw" else adjust_pvalues(p_values, method=method)
                rejects = adjusted <= ALPHA
                false = int(rejects[alternatives:].sum())
                counts["false_positive_families"] += int(false > 0)
                counts["sum_fdp"] += false / max(1, int(rejects.sum()))
                counts["true_rejections"] += int(rejects[:alternatives].sum())

    for window, methods in metrics.items():
        rates = {}
        for method, counts in methods.items():
            fwer = counts["false_positive_families"] / repetitions
            fdr = counts["sum_fdp"] / repetitions
            rates[method] = dict(
                false_positive_families=counts["false_positive_families"], sum_fdp=counts["sum_fdp"],
                true_rejections=counts["true_rejections"], fwer=fwer, fdr=fdr,
                power=counts["true_rejections"] / (alternatives * repetitions) if alternatives else None,
                fwer_passes=fwer <= threshold and not unavailable_families[window],
                fdr_passes=fdr <= threshold and not unavailable_families[window],
            )
        results[window] = dict(
            window=window, unavailable_families=unavailable_families[window],
            unavailable_intervals=sum(unavailable[window].values()),
            unavailable_reasons=dict(unavailable[window]), methods=rates,
        )
    adaptive = results["auto"]
    adaptive["method_version"] = "cs_hac_adaptive_v1"
    required_metrics = [("holm", "fwer_passes"), ("holm", "fdr_passes"), ("bh", "fdr_passes")]
    if not s5 or alternatives == 0:
        required_metrics.append(("bh", "fwer_passes"))
    if domain is not None:
        adaptive.update(
            domain_id=domain.domain_id, in_domain_count=in_domain_count,
            out_of_domain_reasons=dict(sorted(mismatches.items())), domain_passes=in_domain_count == repetitions,
        )
    return dict(
        cell_id=cell.cell_id, seed_spawn_prefix=[cell.section, cell.index], repetitions=repetitions,
        **asdict(design), n_alternatives=alternatives, n_null_candidates=N_CANDIDATES - alternatives,
        fixed=results[fixed_lag], adaptive=adaptive,
        passes=adaptive.get("domain_passes", True)
        and all(adaptive["methods"][method][metric] for method, metric in required_metrics),
    )


def _run_cell(cell: Cell, profile: Profile, s5: bool) -> dict:
    run = _family if cell.section == 4 else _coverage
    return run(cell, profile, s5=s5)


def _read_s5_checkpoint(path: Path, cell: Cell, profile: Profile, fingerprint: str) -> dict:
    try:
        row = json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"Invalid checkpoint JSON for frozen cell {cell.cell_id}") from exc
    expected = dict(
        asdict(cell.design), cell_id=cell.cell_id, seed_spawn_prefix=[cell.section, cell.index],
        repetitions=profile.repetitions, n_alternatives=cell.n_alternatives,
    )
    expected["fixed_lags"] = list(cell.design.fixed_lags)
    if (
        not isinstance(row, dict) or row.pop("run_fingerprint", None) != fingerprint
        or any(row.get(field) != value for field, value in expected.items())
        or not isinstance(row.get("passes"), bool)
    ):
        raise ValueError(f"Checkpoint data does not match frozen cell {cell.cell_id}")
    return row


def run_benchmark(
    *, quick: bool, output: Path, workers: int, resume: bool = False, profile_name: Literal["s4", "s5"] = "s4",
) -> dict:
    s5 = profile_name == "s5"
    profile = (S5_QUICK if quick else S5_FORMAL) if s5 else (QUICK if quick else FORMAL)
    matrix = _matrix(s5=s5)
    checkpoints = output.with_suffix(".cells")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Calibration report already exists at {output}")
    metadata = dict(
        profile="quick" if quick else "registered", **asdict(profile),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        source_revision=os.getenv("QUANTATION_CODE_VERSION"),
    )
    fingerprint = None
    if s5:
        metadata["benchmark_profile"] = profile_name
        metadata["matrix"] = [asdict(cell) for cell in matrix]
        # Compare the serialized representation after a process restart.
        metadata = json.loads(json.dumps(metadata))
        fingerprint = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    manifest = checkpoints / "profile.json"
    if resume:
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError("Checkpoint profile does not match the frozen run")
        if s5 and {path.stem for path in checkpoints.glob("*.json")} - {"profile", *(c.cell_id for c in matrix)}:
            raise ValueError("Checkpoint directory contains cells outside the frozen matrix")
    else:
        checkpoints.mkdir()
        write_json(manifest, metadata)
    pending = []
    retained = {}
    for cell in matrix:
        path = checkpoints / f"{cell.cell_id}.json"
        if not path.exists():
            pending.append(cell)
        elif fingerprint is not None:
            retained[cell.cell_id] = _read_s5_checkpoint(path, cell, profile, fingerprint)
    # Start the costly Bootstrap cells first, longest grids first, to reduce the idle tail.
    pending.sort(key=lambda cell: (cell.section in (1, 3), cell.design.n_times), reverse=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_cell, cell, profile, s5): cell for cell in pending}
        for completed, future in enumerate(as_completed(futures), start=len(matrix) - len(pending) + 1):
            cell = futures[future]
            row = future.result()
            if fingerprint is not None:
                row = dict(row, run_fingerprint=fingerprint)
            write_json(checkpoints / f"{cell.cell_id}.json", row)
            if completed % 10 == 0 or completed == len(matrix):
                print(f"Completed {completed}/{len(matrix)} cells", flush=True)

    cells = []
    for cell in matrix:
        path = checkpoints / f"{cell.cell_id}.json"
        if cell.cell_id in retained:
            row = retained[cell.cell_id]
        elif fingerprint is not None:
            row = _read_s5_checkpoint(path, cell, profile, fingerprint)
        else:
            row = json.loads(path.read_text())
        cells.append(row)
    tolerance = 3 * sqrt(ALPHA * (1 - ALPHA) / profile.repetitions)
    report = dict(
        synthetic=True, **metadata, nominal_coverage=1 - ALPHA, coverage_tolerance=tolerance,
        error_rate_upper_bound=ALPHA + tolerance, coverage=cells[:-4], multiplicity=cells[-4:],
        release_passes=None if quick else all(cell["passes"] for cell in cells),
    )
    if s5:
        del report["matrix"]  # The final rows already include every cell's design.
        by_method = {"hac": [], "bootstrap": []}
        for cell, row in zip(matrix, cells, strict=True):
            method = "bootstrap" if cell.section in (1, 3) else "hac"
            by_method[method].append(row["passes"])
        hac_passes = all(by_method["hac"])
        bootstrap_passes = all(by_method["bootstrap"])
        report.update(
            hac_release_passes=None if quick else hac_passes,
            bootstrap_release_passes=None if quick else bootstrap_passes,
            release_passes=None if quick else hac_passes and bootstrap_passes,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("s4", "s5"), default="s4")
    parser.add_argument("--quick", action="store_true", help="Run the preregistered smoke profile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume only interrupted cells on the frozen source")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = run_benchmark(quick=args.quick, output=args.output, workers=args.workers, resume=args.resume,
                           profile_name=args.profile)
    write_json(args.output, report)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
