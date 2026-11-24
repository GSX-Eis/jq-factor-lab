"""Offline dense-sample acceleration for the NERE-1279 calibration.

The production implementation remains the reference for observed statistics and
is used directly by the nested benchmark. This module only accelerates the
single resample per trial in the warp-speed experiment.
"""
from math import sqrt

import numpy as np
from scipy.signal import lfilter

from backtest_and_analysis.evaluation.calibration import _SCENARIOS
from backtest_and_analysis.evaluation.time_series import (
    _circular_block_indices,
    _fisher_v2_studentization,
    _ppw_block_length,
)
from utils.numba_compat import njit


def gaussian_series(rng: np.random.Generator, size: int, ar: float) -> np.ndarray:
    """Evaluate the original Gaussian AR recurrence in SciPy's filter loop."""
    values = rng.standard_normal((size, 1)).ravel()
    if ar and size > 1:
        values[1:] *= sqrt(1 - ar**2)
        values[1:] = lfilter([1.0], [1.0, -ar], values[1:], zi=[ar * values[0]])[0]
    return values


def legacy_sample(scenario: str, size: int, horizon: int, rng: np.random.Generator):
    spec = _SCENARIOS[scenario]
    x = gaussian_series(rng, size, spec.factor_ar)
    y = gaussian_series(rng, size + horizon - 1, spec.return_ar)
    if horizon > 1:
        y = np.lib.stride_tricks.sliding_window_view(y, horizon).sum(axis=1)
        variance = horizon + 2 * sum((horizon - lag) * spec.return_ar**lag for lag in range(1, horizon))
        y /= sqrt(variance)
    rho = spec.pearson_rho
    return x, rho * x + sqrt(1 - rho**2) * y


@njit(cache=True)
def _margin(groups: np.ndarray, rows: np.ndarray):
    n = len(rows)
    counts = np.bincount(groups[rows], minlength=n)
    centered = (np.cumsum(counts) - (counts + n) / 2) / n
    cdf = centered[groups[rows]]
    return cdf, np.sqrt(np.mean(cdf**2))


@njit(cache=True)
def _rank_adjustment(groups: np.ndarray, cdf: np.ndarray, weights: np.ndarray):
    totals = np.bincount(groups, weights)
    suffix = np.cumsum(totals[::-1])[::-1] - 0.5 * totals
    return suffix[groups] - np.dot(weights, cdf + 0.5)


@njit(cache=True)
def _scores(x_groups: np.ndarray, y_groups: np.ndarray, rows: np.ndarray):
    fx, sx = _margin(x_groups, rows)
    fy, sy = _margin(y_groups, rows)
    n = len(rows)
    if sx == 0 or sy == 0:
        return np.nan, fx, fy
    u, v = fx / sx, fy / sy
    score = u * v
    rho = min(1.0, max(-1.0, score.mean()))
    if np.array_equal(u, v):
        rho = 1.0
    elif np.array_equal(u, -v):
        rho = -1.0
    influence = score - rho * (u**2 + v**2) / 2
    influence += _rank_adjustment(x_groups[rows], fx, v - rho * u) / (n * sx)
    influence += _rank_adjustment(y_groups[rows], fy, u - rho * v) / (n * sy)
    return rho, score - rho, influence


def warp_statistic(x: np.ndarray, y: np.ndarray, horizon: int, seed: int, *, pivot: bool) -> float:
    size = len(x)
    rows = np.arange(size)
    valid = np.ones(size, dtype=bool)
    x_groups = np.unique(x, return_inverse=True)[1].reshape(-1)
    y_groups = np.unique(y, return_inverse=True)[1].reshape(-1)
    rho, scores, influence = _scores(x_groups, y_groups, rows)
    if not np.isfinite(rho):
        return np.nan
    length = _ppw_block_length(scores, size, horizon, min(size // 4, int(np.ceil(3 * sqrt(size)))))
    z, se, ratio = _fisher_v2_studentization(rho, influence, valid, length)
    if not np.isfinite(se):
        return np.nan
    if not pivot:
        return z / se
    sampled = _circular_block_indices(np.random.default_rng(seed), size, np.arange(length))
    rho_star, _, influence_star = _scores(x_groups, y_groups, sampled)
    z_star, se_star, _ = _fisher_v2_studentization(rho_star, influence_star, valid, length, residual_ratio=ratio)
    return abs((z_star - z) / se_star)
