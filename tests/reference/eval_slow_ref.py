"""Independent row-wise references for the S2 evaluation primitives."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata


def average_rank(
    factor: np.ndarray, returns: np.ndarray, valid: np.ndarray, axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.full(factor.shape, np.nan), np.full(returns.shape, np.nan)
    for index, (x, y, mask) in enumerate(zip(
        np.moveaxis(factor, axis, -1), np.moveaxis(returns, axis, -1), np.moveaxis(valid, axis, -1),
    )):
        keep = mask & np.isfinite(x) & np.isfinite(y)
        np.moveaxis(ranks[0], axis, -1)[index, keep] = rankdata(x[keep])
        np.moveaxis(ranks[1], axis, -1)[index, keep] = rankdata(y[keep])
    return ranks


def correlation(
    factor: np.ndarray, returns: np.ndarray, valid: np.ndarray, axis: int,
    min_samples: int, method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = np.moveaxis(factor, axis, -1)
    y_rows = np.moveaxis(returns, axis, -1)
    mask_rows = np.moveaxis(valid, axis, -1)
    result = np.full(len(x_rows), np.nan)
    counts = np.zeros(len(x_rows), dtype=int)
    reasons = np.full(len(x_rows), None, dtype=object)
    for i, (x, y, mask) in enumerate(zip(x_rows, y_rows, mask_rows)):
        keep = mask & np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        counts[i] = len(x)
        if counts[i] < min_samples:
            reasons[i] = "insufficient_samples" if axis in (0, -2) else "insufficient_cross_section"
        elif len(set(x)) == 1:
            reasons[i] = "constant_factor"
        elif len(set(y)) == 1:
            reasons[i] = "constant_return"
        elif method == "spearman":
            # Rank separately so SciPy never coerces mixed signed/unsigned inputs together.
            result[i] = pearsonr(rankdata(x), rankdata(y))[0]
        else:
            # Shift integer data exactly before SciPy's floating-point calculation.
            if np.issubdtype(x.dtype, np.integer):
                x = np.array([int(value) - int(x[0]) for value in x], dtype=float)
            if np.issubdtype(y.dtype, np.integer):
                y = np.array([int(value) - int(y[0]) for value in y], dtype=float)
            result[i] = pearsonr(x, y)[0]
    return result, counts, reasons


def quantile_buckets(
    factor: np.ndarray, valid: np.ndarray, axis: int, n_quantiles: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = np.moveaxis(factor, axis, -1)
    mask_rows = np.moveaxis(valid, axis, -1)
    labels = np.zeros(x_rows.shape, dtype=int)
    edges = np.full((len(x_rows), n_quantiles + 1), np.nan)
    n_bins = np.zeros(len(x_rows), dtype=int)
    for i, (x, mask) in enumerate(zip(x_rows, mask_rows)):
        keep = mask & np.isfinite(x)
        if keep.any():
            codes, boundaries = pd.qcut(x[keep], n_quantiles, labels=False, retbins=True, duplicates="drop")
            labels[i, keep] = np.where(np.isnan(codes), 0, codes + 1)
            edges[i, :len(boundaries)] = boundaries
            n_bins[i] = len(boundaries) - 1
    return np.moveaxis(labels, -1, axis), np.moveaxis(edges, -1, axis), n_bins


def bucket_sums(
    values: np.ndarray, labels: np.ndarray, valid: np.ndarray, axis: int, n_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
    value_rows = np.moveaxis(values, axis, -1)
    label_rows = np.moveaxis(labels, axis, -1)
    mask_rows = np.moveaxis(valid, axis, -1)
    sums = np.zeros((len(value_rows), n_buckets))
    counts = np.zeros((len(value_rows), n_buckets), dtype=int)
    for i, (row, buckets, mask) in enumerate(zip(value_rows, label_rows, mask_rows)):
        for bucket in range(1, n_buckets + 1):
            observations = row[mask & (buckets == bucket) & np.isfinite(row)]
            sums[i, bucket - 1] = sum(observations)
            counts[i, bucket - 1] = len(observations)
    return np.moveaxis(sums, -1, axis), np.moveaxis(counts, -1, axis)
