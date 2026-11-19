"""Synthetic interval coverage and selection across a registered family of trials."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import asin, pi, sqrt
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from .cross_section import bartlett_hac
from .kernels import _positive_int, spearman
from .time_series import _bootstrap_spearman
from .types import FloatArray

_ALPHA = 0.05
_NOMINAL_COVERAGE = 1 - _ALPHA
_TOLERANCE_MCSE = 3


@dataclass(frozen=True)
class _Scenario:
    horizon: int = 1
    factor_ar: float = 0.0
    return_ar: float = 0.0
    pearson_rho: float = 0.0


_SCENARIOS = {
    "independent_null": _Scenario(),
    "ar1_null": _Scenario(factor_ar=0.8, return_ar=0.8),
    "overlap_null": _Scenario(horizon=5, factor_ar=0.8),
    "positive_effect": _Scenario(pearson_rho=0.35),
    "negative_effect": _Scenario(pearson_rho=-0.35),
}


def _gaussian_series(
    rng: np.random.Generator, n_times: int, n_assets: int, ar: float,
) -> FloatArray:
    values = rng.standard_normal((n_times, n_assets))
    if ar:
        values[1:] *= sqrt(1 - ar**2)
        for i in range(1, n_times):
            values[i] += ar * values[i - 1]
    return values


def synthetic_sample(
    scenario: str, rng: np.random.Generator, *, n_times: int, n_assets: int,
) -> tuple[FloatArray, FloatArray]:
    """Generate paired Gaussian factors and forward returns on a dense signal grid."""
    if scenario not in _SCENARIOS:
        raise ValueError(f"Unknown calibration scenario {scenario!r}")
    n_times = _positive_int(n_times, "n_times")
    n_assets = _positive_int(n_assets, "n_assets")
    spec = _SCENARIOS[scenario]
    factor = _gaussian_series(rng, n_times, n_assets, spec.factor_ar)
    returns = _gaussian_series(rng, n_times + spec.horizon - 1, n_assets, spec.return_ar)
    if spec.horizon > 1:
        returns = np.lib.stride_tricks.sliding_window_view(returns, spec.horizon, axis=0).sum(axis=-1)
    if spec.pearson_rho:
        rho = spec.pearson_rho
        returns = rho * factor + sqrt(1 - rho**2) * returns
    return factor, returns


def _coverage_mcse(repetitions: int) -> float:
    return sqrt(_NOMINAL_COVERAGE * _ALPHA / repetitions)


def coverage_passes(covered: int, repetitions: int) -> bool:
    """Apply the predeclared per-cell tolerance to the count of covering intervals."""
    repetitions = _positive_int(repetitions, "repetitions")
    covered = _positive_int(covered, "covered", minimum=0)
    if covered > repetitions:
        raise ValueError("covered must be <= repetitions")
    # Square the frozen 95% ± 3 MCSE inequality to keep both boundaries exact.
    return (20 * covered - 19 * repetitions)**2 <= 171 * repetitions


@dataclass
class _CoverageCounts:
    covered: int = 0
    point_sign_correct: int = 0
    correct_direction: int = 0
    rejects_zero: int = 0

    def add(self, point: float, lower: float, upper: float, truth: float) -> None:
        self.covered += lower <= truth <= upper
        self.point_sign_correct += (truth > 0 and point > 0) or (truth < 0 and point < 0)
        self.correct_direction += (truth > 0 and lower > 0) or (truth < 0 and upper < 0)
        self.rejects_zero += lower > 0 or upper < 0


def calibrate_coverage(
    *, seed: int = 791823, cross_section_repetitions: int = 2000,
    time_series_repetitions: int = 500, n_times: int = 512, n_assets: int = 32,
    n_boot: int = 399,
) -> dict:
    """Run the fixed synthetic scenarios through the S3-A and S3-B interval estimators."""
    seed = _positive_int(seed, "seed", minimum=0)
    n_times = _positive_int(n_times, "n_times", minimum=18)
    n_assets = _positive_int(n_assets, "n_assets", minimum=3)
    n_boot = _positive_int(n_boot, "n_boot", minimum=2)
    repetitions = {
        "cross_section_hac": _positive_int(cross_section_repetitions, "cross_section_repetitions"),
        "time_series_mbb": _positive_int(time_series_repetitions, "time_series_repetitions"),
    }
    cells = []
    seed_streams = iter(np.random.SeedSequence(seed).spawn(len(repetitions) * len(_SCENARIOS)))
    for method, m in repetitions.items():
        cross_section = method == "cross_section_hac"
        mcse = _coverage_mcse(m)
        for scenario, spec in _SCENARIOS.items():
            child = next(seed_streams)
            rng = np.random.default_rng(child)
            minimum = spec.horizon - int(cross_section)
            dependent = bool(spec.factor_ar or spec.return_ar or spec.horizon > 1)
            recommended = max(minimum, 16 if cross_section else 17) if dependent else minimum
            windows = tuple(dict.fromkeys((minimum, recommended)))
            rho = spec.pearson_rho
            half_angle = asin(rho / 2)
            if cross_section:
                # The HAC target is the expected finite-cross-section sample Spearman.
                truth = 6 * (asin(rho) + (n_assets - 2) * half_angle) / (pi * (n_assets + 1))
            else:
                truth = 6 * half_angle / pi

            counts = {window: _CoverageCounts() for window in windows}
            unavailable = {window: Counter() for window in windows}
            valid = np.ones(n_times, dtype=bool)
            for _ in range(m):
                x, y = synthetic_sample(
                    scenario, rng, n_times=n_times, n_assets=n_assets if cross_section else 1,
                )
                ic = spearman(x, y, axis=1 if cross_section else 0, min_samples=3)
                point = float(ic.values.mean())
                draw_seed = 0 if cross_section else int(rng.integers(2**63))
                for window in windows:
                    if cross_section:
                        interval = bartlett_hac(ic.values, lag=window, alpha=_ALPHA)
                    else:
                        interval = _bootstrap_spearman(
                            x[:, 0], y[:, 0], valid, point=ic, min_samples=3,
                            block_length=window, seed=draw_seed, n_boot=n_boot,
                            min_valid_replicates=n_boot, alpha=_ALPHA,
                        )
                    if interval.reason is not None:
                        unavailable[window][interval.reason] += 1
                    counts[window].add(point, interval.lower, interval.upper, truth)

            for window, count in counts.items():
                within_tolerance = coverage_passes(count.covered, m)
                cells.append({
                    "method": method,
                    "scenario": scenario,
                    **asdict(spec),
                    "seed_spawn_key": list(child.spawn_key),
                    "M": m,
                    "window": window,
                    "minimum_window": minimum,
                    "recommended_window": recommended,
                    "true_ic": truth,
                    "covered": count.covered,
                    "unavailable_intervals": dict(unavailable[window]),
                    "coverage": count.covered / m,
                    "mcse_at_nominal": mcse,
                    "tolerance": _TOLERANCE_MCSE * mcse,
                    "within_tolerance": within_tolerance,
                    "passes": within_tolerance and not unavailable[window],
                    "point_sign_correct_rate": count.point_sign_correct / m if truth else None,
                    "correct_direction_rate": count.correct_direction / m if truth else None,
                    "reject_zero_rate": count.rejects_zero / m,
                })
    return {
        "seed": seed,
        "n_times": n_times,
        "n_cross_section_assets": n_assets,
        "n_boot": n_boot,
        "nominal_coverage": _NOMINAL_COVERAGE,
        "tolerance_mcse_multiple": _TOLERANCE_MCSE,
        "cells": cells,
    }


Correction = Literal["holm", "bh"]
TrialKey = tuple[str, int]


def adjust_pvalues(p_values: npt.ArrayLike, *, method: Correction) -> FloatArray:
    """Return adjusted p-values in input order for Holm FWER or BH FDR control."""
    if method not in ("holm", "bh"):
        raise ValueError("method must be 'holm' or 'bh'")
    if np.ma.is_masked(p_values):
        raise ValueError("p_values must not contain masked values")
    values = np.asarray(p_values)
    if (values.ndim != 1 or not values.size or values.dtype.kind not in "iuf"
            or not np.all((0 <= values) & (values <= 1))):
        raise ValueError("p_values must be a nonempty vector of finite values in [0, 1]")
    values = values.astype(float, copy=False)
    order = np.argsort(values, kind="stable")
    sorted_p = values[order]
    if method == "holm":
        adjusted = np.maximum.accumulate(sorted_p * np.arange(values.size, 0, -1))
    else:
        # Scale the input's exact integer ratio so only the final division rounds.
        scaled = np.empty_like(sorted_p)
        for index, p_value in enumerate(sorted_p):
            numerator, denominator = p_value.as_integer_ratio()
            scaled[index] = (numerator * values.size) / (denominator * (index + 1))
        adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1)
    return result


@dataclass(frozen=True)
class Trial:
    statistic: float | None
    p_value: float | None
    n_effective: int
    interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        _positive_int(self.n_effective, "n_effective", minimum=0)
        if self.statistic is not None and not np.isfinite(self.statistic):
            raise ValueError("statistic must be finite or None")
        if self.p_value is not None and not 0 <= self.p_value <= 1:
            raise ValueError("p_value must be in [0, 1] or None")
        if self.interval is not None:
            if (not isinstance(self.interval, tuple) or len(self.interval) != 2
                    or not np.isfinite(self.interval).all() or self.interval[0] > self.interval[1]):
                raise ValueError("interval must be a pair of ordered finite bounds")
        if self.p_value is not None or self.interval is not None:
            if self.statistic is None or not self.n_effective:
                raise ValueError("inference requires a statistic and at least one effective sample")


class AdjustedTrial(NamedTuple):
    key: TrialKey
    trial: Trial
    adjusted_p_value: float
    significant: bool


@dataclass(frozen=True)
class TrialSelection:
    best: TrialKey
    method: Correction
    alpha: float
    trials: tuple[AdjustedTrial, ...]

    @property
    def attempt_count(self) -> int:
        return len(self.trials)


class TrialLedger:
    """Register the complete candidate family before recording its evaluated trials."""

    def __init__(self, candidates: Iterable[TrialKey]) -> None:
        self._candidates = tuple(candidates)
        if not self._candidates:
            raise ValueError("Register at least one candidate")
        for key in self._candidates:
            if (not isinstance(key, tuple) or len(key) != 2
                    or not isinstance(key[0], str) or not key[0].strip()):
                raise ValueError("Each candidate must be a (factor, horizon) pair")
            _positive_int(key[1], "horizon")
        self._expected = frozenset(self._candidates)
        if len(self._expected) != len(self._candidates):
            raise ValueError("Candidate keys must be unique")
        self._trials: dict[TrialKey, Trial] = {}

    def record(self, key: TrialKey, trial: Trial) -> None:
        if key not in self._expected:
            raise ValueError(f"Unregistered candidate {key!r}")
        if key in self._trials:
            raise ValueError(f"Trial already recorded for {key!r}")
        if not isinstance(trial, Trial):
            raise TypeError("trial must be a Trial")
        self._trials[key] = trial

    @property
    def records(self) -> dict[TrialKey, Trial]:
        """Snapshot recorded attempts, including trials with unavailable inference."""
        return self._trials.copy()

    def select_best(self, *, method: Correction, alpha: float = 0.05) -> TrialSelection:
        """Return the full adjusted family and the strongest candidate by p-value."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        missing = self._expected - self._trials.keys()
        if missing:
            raise ValueError(f"Missing trial records for {sorted(missing)!r}")

        p_values = []
        for key in self._candidates:
            p_value = self._trials[key].p_value
            if p_value is None:
                raise ValueError(f"Missing raw p-value for {key!r}")
            p_values.append(p_value)
        adjusted = adjust_pvalues(p_values, method=method)
        trials = tuple(
            AdjustedTrial(key, self._trials[key], float(p), bool(p <= alpha))
            for key, p in zip(self._candidates, adjusted)
        )
        best = self._candidates[int(np.argmin(p_values))]
        return TrialSelection(best, method, alpha, trials)
