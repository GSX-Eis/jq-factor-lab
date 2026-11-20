"""Run the fixed NERE-917 synthetic coverage and correlated-family benchmark."""

import argparse
from collections import Counter
from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path

import numpy as np

from backtest_and_analysis.evaluation.calibration import (
    _gaussian_series, adjust_pvalues, coverage_passes,
)
from backtest_and_analysis.evaluation.cross_section import bartlett_hac
from backtest_and_analysis.evaluation.kernels import spearman
from backtest_and_analysis.evaluation.time_series import _bootstrap_spearman
from utils.local_outputs import write_json


ALPHA = .05
N_TIMES = 512
N_ASSETS = 32
N_CANDIDATES = 6


@dataclass(frozen=True)
class Design:
    factor_ar: float = 0.
    return_ar: float = 0.
    horizon: int = 1
    gaps: bool = False
    heteroskedastic: bool = False
    lags: tuple[int, ...] = (0,)

    def sample(self, rng: np.random.Generator, n_assets: int, n_alternatives: int = 0,
               *, family: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.arange(N_TIMES)
        valid = (rows % 17 < 14) if self.gaps else np.ones(N_TIMES, dtype=bool)
        size = N_TIMES + self.horizon - 1
        scale = np.where(np.arange(size) // 32 % 2 == 0, 1., 4.) if self.heteroskedastic else np.ones(size)
        returns = _gaussian_series(rng, size, n_assets, self.return_ar)
        if family:
            common = _gaussian_series(rng, N_TIMES, n_assets, self.factor_ar)
            factors = np.stack([
                sqrt(.65) * common + sqrt(.35) * _gaussian_series(rng, N_TIMES, n_assets, self.factor_ar)
                for _ in range(N_CANDIDATES)
            ])
            if n_alternatives:
                signal = _gaussian_series(rng, size, n_assets, 0.)
                factors[:n_alternatives] += .5 * signal[:N_TIMES]
                returns += .35 * signal
        else:
            factors = _gaussian_series(rng, N_TIMES, n_assets, self.factor_ar)
        factors *= scale[:N_TIMES, None]
        returns *= scale[:, None]
        returns = np.lib.stride_tricks.sliding_window_view(returns, self.horizon, axis=0).sum(axis=-1)
        return factors, returns, np.broadcast_to(valid[:, None], returns.shape)


DESIGNS = {
    "iid": Design(),
    "strong_ar": Design(.8, .8, lags=(0, 16)),
    "overlap": Design(.8, horizon=5, lags=(4, 16)),
    "gaps": Design(gaps=True),
    "heteroskedastic": Design(heteroskedastic=True),
    "combined": Design(.8, .8, horizon=5, gaps=True, heteroskedastic=True, lags=(16,)),
}


def _coverage(design: Design, lag: int, cross_section: bool, repetitions: int,
              n_boot: int, stream: np.random.SeedSequence) -> dict:
    data_seed, bootstrap_seed = stream.spawn(2)
    rng, bootstrap_rng = np.random.default_rng(data_seed), np.random.default_rng(bootstrap_seed)
    unavailable = Counter()
    covered = 0
    for _ in range(repetitions):
        x, y, valid = design.sample(rng, N_ASSETS if cross_section else 1)
        ic = spearman(x, y, valid=valid, axis=int(cross_section), min_samples=3)
        if cross_section:
            interval = bartlett_hac(ic.values, lag=lag, alpha=ALPHA)
        else:
            interval = _bootstrap_spearman(
                x[:, 0], y[:, 0], valid[:, 0], point=ic, min_samples=3,
                block_length=lag + 1, seed=int(bootstrap_rng.integers(2**63)), n_boot=n_boot,
                min_valid_replicates=n_boot, alpha=ALPHA,
            )
        if interval.reason is not None:
            unavailable[interval.reason] += 1
        else:
            covered += interval.lower <= 0 <= interval.upper

    within_tolerance = coverage_passes(covered, repetitions)
    return dict(
        seed_spawn_key=list(stream.spawn_key), covered=covered, coverage=covered / repetitions,
        unavailable_count=sum(unavailable.values()), unavailable_reasons=dict(unavailable),
        within_tolerance=within_tolerance, passes=within_tolerance and not unavailable,
    )


def _family(design: Design, n_alternatives: int, repetitions: int,
            stream: np.random.SeedSequence) -> dict:
    lag = design.lags[-1]
    rng = np.random.default_rng(stream)
    unavailable_reasons = Counter()
    unavailable_families = 0
    metrics = {method: Counter() for method in ("raw", "holm", "bh")}
    for _ in range(repetitions):
        factors, returns, valid = design.sample(rng, N_ASSETS, n_alternatives, family=True)
        p_values = []
        for factor in factors:
            ic = spearman(factor, returns, valid=valid, axis=1, min_samples=3).values
            interval = bartlett_hac(ic, lag=lag, alpha=ALPHA)
            if interval.reason is not None:
                unavailable_reasons[interval.reason] += 1
            else:
                mean = float(ic[np.isfinite(ic)].mean())
                p_values.append(erfc(abs(mean) / sqrt(2 * interval.variance)))
        if len(p_values) != N_CANDIDATES:
            unavailable_families += 1
            continue
        for method, counts in metrics.items():
            p = np.asarray(p_values) if method == "raw" else adjust_pvalues(p_values, method=method)
            rejects = p <= ALPHA
            false = int(rejects[n_alternatives:].sum())
            counts["false_positive_families"] += int(false > 0)
            counts["sum_fdp"] += false / max(1, int(rejects.sum()))
            counts["true_rejections"] += int(rejects[:n_alternatives].sum())

    threshold = ALPHA + _tolerance(repetitions)
    methods = {}
    for method, counts in metrics.items():
        fwer = counts["false_positive_families"] / repetitions
        fdr = counts["sum_fdp"] / repetitions
        methods[method] = dict(
            fwer=fwer, fdr=fdr, fwer_passes=fwer <= threshold and not unavailable_families,
            fdr_passes=fdr <= threshold and not unavailable_families,
            false_positive_families=counts["false_positive_families"],
            power=counts["true_rejections"] / (repetitions * n_alternatives) if n_alternatives else None,
        )
    return dict(
        seed_spawn_key=list(stream.spawn_key), lag=lag, n_alternatives=n_alternatives,
        n_null_candidates=N_CANDIDATES - n_alternatives, unavailable_families=unavailable_families,
        unavailable_intervals=sum(unavailable_reasons.values()),
        unavailable_reasons=dict(unavailable_reasons), methods=methods,
    )


def _tolerance(repetitions: int) -> float:
    return 3 * sqrt(ALPHA * (1 - ALPHA) / repetitions)


def run_benchmark(*, quick: bool = False) -> dict:
    seed, repetitions, n_boot = (91773, 3, 19) if quick else (91720261120, 500, 399)
    matrix = [(name, design, lag, cs) for name, design in DESIGNS.items()
              for lag in design.lags for cs in (True, False)]
    families = [(name, alternatives) for name in ("iid", "combined") for alternatives in (0, 2)]
    streams = iter(np.random.SeedSequence(seed).spawn(len(matrix) + len(families)))
    coverage = [
        dict(scenario=name, method="hac" if cs else "mbb", window=lag if cs else lag + 1,
             n_observed_grid_slots=N_TIMES - (N_TIMES // 17 * 3 if design.gaps else 0),
             **_coverage(design, lag, cs, repetitions, n_boot, next(streams)))
        for name, design, lag, cs in matrix
    ]
    multiplicity = [
        dict(scenario=name, **_family(DESIGNS[name], alt, repetitions, next(streams)))
        for name, alt in families
    ]
    return dict(
        synthetic=True, profile="quick" if quick else "registered", seed=seed,
        repetitions=repetitions, n_times=N_TIMES, n_cross_section_assets=N_ASSETS, n_boot=n_boot,
        nominal_coverage=1 - ALPHA, coverage_tolerance=_tolerance(repetitions),
        error_rate_upper_bound=ALPHA + _tolerance(repetitions),
        dependence_assumptions=dict(
            raw="No multiplicity control.",
            holm="Arbitrary candidate dependence with valid marginal p-values.",
            bh="Independent or suitable positive dependence (PRDS); not established here for two-sided tests.",
        ),
        coverage=coverage, multiplicity=multiplicity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the independent smoke-test profile")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_benchmark(quick=args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
