"""Dense synthetic return features for NERE-1279. The legacy scenarios are frozen."""
from math import sqrt

import numpy as np

from tools.benchmarks.ts_pvalue_dense_draws import legacy_sample
from utils.numba_compat import njit


@njit(cache=True)
def _garch(innovations: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    variance = omega / (1 - alpha - beta)
    path = np.empty_like(innovations)
    for i, innovation in enumerate(innovations):
        path[i] = sqrt(variance) * innovation
        variance = omega + alpha * path[i] ** 2 + beta * variance
    return path


def innovations(feature: str, rng: np.random.Generator, size: int, params: dict) -> np.ndarray:
    if feature == "student_t":
        df = params["df"]
        return rng.standard_t(df, size) * sqrt((df - 2) / df)
    if feature == "garch":
        burn = params["burn_in"]
        return _garch(rng.standard_normal(size + burn), params["omega"], params["alpha"], params["beta"])[burn:]
    if feature == "jumps":
        lam, sd = params["intensity"], params["jump_std"]
        # Conditional on count k, the sum of k independent N(0, sd^2) jump sizes is N(0, k*sd^2).
        counts = rng.poisson(lam, size)
        return (rng.standard_normal(size) + rng.standard_normal(size) * sd * np.sqrt(counts)) / sqrt(1 + lam * sd**2)
    raise ValueError(f"Unknown feature {feature!r}")


def feature_sample(scenario: str, feature: str, rho: float, params: dict, size: int, horizon: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if feature == "gaussian":
        return legacy_sample(scenario, size, horizon, rng)
    x = rng.standard_normal(size)
    y = innovations(feature, rng, size + horizon - 1, params)
    if horizon > 1:
        y = np.lib.stride_tricks.sliding_window_view(y, horizon).sum(axis=1) / sqrt(horizon)
    return x, rho * x + sqrt(1 - rho**2) * y
