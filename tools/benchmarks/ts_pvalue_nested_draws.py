"""Dense batching of the production resampling stream for the offline nested study."""
from math import sqrt

import numpy as np

from backtest_and_analysis.evaluation.time_series import (
    _fisher_v2_studentization,
    _ppw_block_length,
    _spearman_scores,
)
from tools.benchmarks.ts_pvalue_dense_draws import _scores
from utils.numba_compat import njit


# Numba cannot invalidate this caller's disk cache when the imported _scores changes.
@njit(cache=False)
def _nested_pivots(x_groups: np.ndarray, y_groups: np.ndarray, starts: np.ndarray,
                   length: int, z: float, ratio: float) -> np.ndarray:
    n = len(x_groups)
    q = n // length
    rows = np.empty(n, dtype=np.int64)
    block_sums = np.empty(q)
    pivots = np.full(len(starts), np.nan)
    for i in range(len(starts)):
        for block, start in enumerate(starts[i]):
            offset = block * length
            for j in range(min(length, n - offset)):
                rows[offset + j] = (start + j) % n
        rho, _, influence = _scores(x_groups, y_groups, rows)
        if not np.isfinite(rho):
            continue
        rho = min(1 - 1e-6, max(-1 + 1e-6, rho))
        scores = (influence - influence.mean()) / (1 - rho**2)
        v0 = np.dot(scores, scores) / n**2
        if not np.isfinite(v0) or v0 <= 1e-24 / n:
            continue
        for block in range(q):
            block_sums[block] = scores[block * length:(block + 1) * length].sum()
        variance = (q + ratio) * block_sums.var() * q / ((q - 1) * n**2)
        if np.isfinite(variance) and variance > 1e-12 * v0:
            pivots[i] = (np.arctanh(rho) - z) / sqrt(variance)
    return pivots


def nested_draws(x: np.ndarray, y: np.ndarray, horizon: int, seed: int, n_boot: int) -> tuple[float, np.ndarray]:
    n = len(x)
    original = _spearman_scores(x, y)
    if original is None:
        return np.nan, np.empty(0)
    rho, scores, influence = original
    length = _ppw_block_length(scores, n, horizon, min(n // 4, int(np.ceil(min(3 * sqrt(n), n / 3)))))
    z, se, ratio = _fisher_v2_studentization(rho, influence, np.ones(n, dtype=bool), length)
    if not np.isfinite(se) or n / length < 4:
        return np.nan, np.empty(0)
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, n, size=(n_boot, (n + length - 1) // length), dtype=np.int64)
    x_groups = np.unique(x, return_inverse=True)[1].reshape(-1)
    y_groups = np.unique(y, return_inverse=True)[1].reshape(-1)
    pivots = _nested_pivots(x_groups, y_groups, starts, length, z, ratio)
    return z / se, pivots[np.isfinite(pivots)]


def nested_p_value(x: np.ndarray, y: np.ndarray, horizon: int, seed: int, n_boot: int, min_valid: int) -> float:
    observed, pivots = nested_draws(x, y, horizon, seed, n_boot)
    if not np.isfinite(observed) or len(pivots) < min_valid or 10 * len(pivots) < 9 * n_boot:
        return np.nan
    return (1 + np.count_nonzero(np.abs(pivots) >= abs(observed))) / (1 + len(pivots))
