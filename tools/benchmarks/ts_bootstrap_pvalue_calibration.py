"""NERE-1269 synthetic bootstrap P-value calibration and leased timing pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from math import sqrt
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.stats import binomtest

from backtest_and_analysis.evaluation.calibration import _SCENARIOS, _gaussian_series
from backtest_and_analysis.evaluation.kernels import spearman
from backtest_and_analysis.evaluation.time_series import _adaptive_bootstrap_inputs, _bootstrap_spearman


def _sample(scenario: str, size: int, horizon: int, rng: np.random.Generator):
    spec = _SCENARIOS[scenario]
    factor = _gaussian_series(rng, size, 1, spec.factor_ar).ravel()
    returns = _gaussian_series(rng, size + horizon - 1, 1, spec.return_ar).ravel()
    if horizon > 1:
        returns = np.lib.stride_tricks.sliding_window_view(returns, horizon).sum(axis=1)
        variance = horizon + 2 * sum((horizon - k) * spec.return_ar**k for k in range(1, horizon))
        returns /= sqrt(variance)
    rho = spec.pearson_rho
    return factor, rho * factor + sqrt(1 - rho**2) * returns


def _cell(job: tuple) -> dict:
    index, (scenario, size, horizon), mode, design = job
    spec = _SCENARIOS[scenario]
    plan = design[mode]
    alphas = np.array(design["alphas"])
    valid = np.ones(size, dtype=bool)
    result = dict(scenario=scenario, grid_size=size, horizon=horizon)
    start = perf_counter()

    def inputs(phase: int, repetition: int):
        data_seed, boot_seed = np.random.SeedSequence([plan["seed"], index, phase, repetition]).spawn(2)
        x, y = _sample(scenario, size, horizon, np.random.default_rng(data_seed))
        # Dense continuous scenarios always have an estimable point IC. The shared
        # bootstrap kernel still checks the original rank influence and each draw.
        point = spearman(x[:, None], y[:, None], valid=valid[:, None], axis=0, min_samples=4)
        return x, y, dict(
            point=point, min_samples=4, min_block_length=horizon, p_value=True,
            seed=int(boot_seed.generate_state(1, dtype=np.uint64)[0]),
            alpha=0.05, calibration_scenario=scenario,
        )

    counts = np.zeros(len(alphas), dtype=int)
    signed = counts.copy()
    invalid = 0
    critical = None
    invalid_critical = 0
    if mode == "warp_speed":
        def draw(phase: int, repetition: int):
            x, y, settings = inputs(phase, repetition)
            interval, z, draws = _adaptive_bootstrap_inputs(
                x, y, valid, n_boot=1 if phase == 0 else 0, min_valid_replicates=2,
                intervals_verified=True, historical_v1=False, **settings,
            )
            if interval.reason is not None:
                return np.nan
            return abs(next(draws, np.nan)) if phase == 0 else z / interval.fisher_z_standard_error

        pivots = np.array([draw(0, i) for i in range(plan["critical_repetitions_per_cell"])])
        usable = np.isfinite(pivots)
        invalid_critical = int((~usable).sum())
        if usable.any():
            critical = np.quantile(pivots[usable], 1 - alphas, method="inverted_cdf")
        repetitions = plan["evaluation_repetitions_per_cell"]
        for i in range(repetitions):
            observed = draw(1, i)
            if critical is None or not np.isfinite(observed):
                invalid += 1
                continue
            rejects = abs(observed) > critical
            counts += rejects
            if observed * spec.pearson_rho > 0:
                signed += rejects
    else:
        key = "effect_repetitions_per_cell" if spec.pearson_rho else "null_repetitions_per_cell"
        repetitions = plan["repetitions_per_cell"] if mode == "pilot" else plan[key]
        for i in range(repetitions):
            x, y, settings = inputs(0, i)
            interval = _bootstrap_spearman(
                x, y, valid, n_boot=design["n_boot"], min_valid_replicates=design["min_valid_replicates"],
                **settings,
            )
            if mode != "pilot":
                if interval.p_value is None:
                    invalid += 1
                    continue
                rejects = interval.p_value <= alphas
                counts += rejects
                if settings["point"].values[0] * spec.pearson_rho > 0:
                    signed += rejects

    result.update(repetitions=repetitions, seconds=perf_counter() - start)
    if mode == "pilot":
        return result
    rows = []
    for j, alpha in enumerate(alphas):
        rejected = int(counts[j])
        rate = rejected / repetitions
        ci = binomtest(rejected, repetitions).proportion_ci(design["confidence"], method="exact")
        passes = (
            signed[j] / repetitions >= design["power_floor"] if spec.pearson_rho else
            ci.low <= alpha <= ci.high and rate <= design["size_cap_multiplier"] * alpha
        )
        rows.append(dict(alpha=float(alpha), rejected=rejected, rejection_rate=rate, ci99=[ci.low, ci.high],
                         direction_correct_rejections=int(signed[j]) if spec.pearson_rho else None,
                         critical_value=float(critical[j]) if critical is not None else None,
                         passes=bool(passes and invalid == 0 and invalid_critical == 0)))
    return result | dict(null=spec.pearson_rho == 0, unestimable=invalid,
                         unestimable_critical=invalid_critical, thresholds=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--preregistration-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--mode", choices=["pilot", "nested", "warp_speed"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    jobs = [(idx, cell, args.mode, design) for idx, cell in enumerate(
        product(design["scenarios"], design["grid_sizes"], design["horizons"]),
    )]
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cells = list(executor.map(_cell, jobs))
    report = dict(
        preregistration_commit=args.preregistration_commit, code_commit=args.code_commit,
        mode=args.mode, workers=args.workers, design=design, seconds=perf_counter() - start,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, cells=cells,
    )
    if args.mode != "pilot":
        report["all_cells_pass"] = all(row["passes"] for cell in cells for row in cell["thresholds"])
        report["release_passes"] = args.mode == "nested" and report["all_cells_pass"]
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{args.mode}: {report['seconds']:.1f}s, {len(cells)} cells", flush=True)


if __name__ == "__main__":
    main()
