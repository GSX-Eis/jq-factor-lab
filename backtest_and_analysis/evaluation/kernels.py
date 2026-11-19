"""NumPy kernels for two-dimensional datetime-by-asset panels."""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

_Float = npt.NDArray[np.floating]
_Int = npt.NDArray[np.intp]
_Bool = npt.NDArray[np.bool_]


class CorrelationResult(NamedTuple):
    values: _Float
    counts: _Int
    reasons: npt.NDArray[np.object_]


class QuantileResult(NamedTuple):
    labels: _Int
    edges: _Float
    n_bins: _Int


class BucketResult(NamedTuple):
    sums: _Float
    counts: _Int


def _rows(values: npt.ArrayLike, axis: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional panel")
    if axis in (1, -1):
        return array
    if axis in (0, -2):
        return array.T
    raise ValueError("axis must identify datetime (0) or asset (1)")


def _real_rows(values: npt.ArrayLike, axis: int) -> np.ndarray:
    rows = _rows(values, axis)
    if rows.dtype.kind not in "biuf":
        raise ValueError("Expected a real numeric panel")
    return rows.astype(np.int64) if rows.dtype.kind == "b" else rows


def _eligible(finite: _Bool, valid: npt.ArrayLike | None, axis: int) -> _Bool:
    if valid is not None:
        mask = _rows(valid, axis)
        if mask.shape != finite.shape or mask.dtype.kind != "b":
            raise ValueError("valid must be a boolean mask with the panel shape")
        finite &= mask
    return finite


def _pairs(
    factor: npt.ArrayLike, returns: npt.ArrayLike, valid: npt.ArrayLike | None, axis: int,
) -> tuple[np.ndarray, np.ndarray, _Bool]:
    x, y = _real_rows(factor, axis), _real_rows(returns, axis)
    if x.shape != y.shape:
        raise ValueError("factor and returns must have the same panel shape")
    paired = _eligible(np.isfinite(x) & np.isfinite(y), valid, axis)
    return x, y, paired


def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _rank_rows(rows: np.ndarray, paired: _Bool) -> _Float:
    width = rows.shape[1]
    ranks = np.empty(rows.shape, dtype=np.float64)
    if not width:
        return ranks

    if rows.dtype.kind in "iu":
        order = np.lexsort((rows, ~paired), axis=1)
    else:
        order = np.argsort(np.where(paired, rows, np.inf), axis=1)
    sorted_rows = np.take_along_axis(rows, order, axis=1)
    ties = sorted_rows[:, 1:] == sorted_rows[:, :-1]
    ties &= np.arange(1, width) < paired.sum(axis=1)[:, None]
    np.put_along_axis(ranks, order, np.arange(1, width + 1)[None, :], axis=1)
    tied_rows, left = np.nonzero(ties)
    if left.size:
        # Runs of adjacent equal values share the mean of their endpoint ranks.
        begins = np.r_[0, np.flatnonzero((np.diff(tied_rows) != 0) | (np.diff(left) != 1)) + 1]
        ends = np.r_[begins[1:] - 1, len(left) - 1]
        averages = np.repeat((left[begins] + left[ends] + 3) * .5, ends - begins + 1)
        ranks[tied_rows, order[tied_rows, left]] = averages
        ranks[tied_rows, order[tied_rows, left + 1]] = averages
    ranks[~paired] = np.nan
    return ranks


def average_rank(
    factor: npt.ArrayLike, returns: npt.ArrayLike, *, axis: int = 1,
    valid: npt.ArrayLike | None = None,
) -> tuple[_Float, _Float]:
    """Rank both panels on their shared finite, eligible observations."""
    x, y, paired = _pairs(factor, returns, valid, axis)
    return _rows(_rank_rows(x, paired), axis), _rows(_rank_rows(y, paired), axis)


def _center(rows: np.ndarray, paired: _Bool, counts: _Int) -> tuple[_Float, _Bool]:
    if rows.dtype.kind in "iu":
        lo = rows.min(axis=1, where=paired, initial=np.iinfo(rows.dtype).max)
        # Unsigned subtraction represents the full nonnegative difference before rounding.
        shifted = (rows.astype(np.uint64) - lo.astype(np.uint64)[:, None]).astype(np.float64)
        constant = ~np.any(paired & (shifted != 0), axis=1)
    else:
        rows = rows.astype(np.result_type(rows.dtype, np.float64), copy=False)
        lo = rows.min(axis=1, where=paired, initial=np.inf)
        hi = rows.max(axis=1, where=paired, initial=-np.inf)
        constant = lo == hi
        lo = np.where(counts, lo, 0)
        hi = np.where(counts, hi, 0)

        # Scale by a power of two and shift before averaging to preserve large-offset differences.
        _, exp = np.frexp(np.maximum(np.abs(lo), np.abs(hi)))
        shifted = np.ldexp(np.where(paired, rows, 0.0), -exp[:, None])
        shifted -= np.ldexp(lo, -exp)[:, None]
    shifted[~paired] = 0.0
    mean = shifted.sum(axis=1) / np.maximum(counts, 1)
    shifted -= mean[:, None]
    shifted[~paired] = 0.0
    return shifted, constant


def _integer_significand(mantissa: _Float, precision: int) -> np.ndarray:
    if precision <= 63:
        return np.ldexp(mantissa, precision).astype(np.int64).astype(object)

    # Extended floats can need two integer limbs to represent the significand.
    head = np.ldexp(mantissa, 62)
    upper = head.astype(np.int64)
    lower = np.ldexp(head - upper, precision - 62).astype(np.int64)
    return (upper.astype(object) << (precision - 62)) + lower.astype(object)


def _integer_moment_rows(rows: np.ndarray, paired: _Bool, counts: _Int) -> tuple[np.ndarray, _Int]:
    rows = np.where(paired, rows, 0)
    if rows.dtype.kind in "iu":
        return rows.astype(object), np.zeros_like(counts)

    rows = rows.astype(np.result_type(rows.dtype, np.float64), copy=False)
    mantissa, exponent = np.frexp(rows)
    nonzero = rows != 0
    limits = np.iinfo(exponent.dtype)
    low = exponent.min(axis=1, where=nonzero, initial=limits.max)
    high = exponent.max(axis=1, where=nonzero, initial=limits.min)
    precision = np.finfo(rows.dtype).nmant + 1
    shift = np.where(nonzero, exponent - low[:, None], 0)
    coefficients = _integer_significand(mantissa, precision)
    # Match the power-of-two scaling in _center: scaled rows are integers / 2**scale.
    return coefficients << shift.astype(object), high - low + precision


def _pearson_after_cancellation(
    x: np.ndarray, y: np.ndarray, paired: _Bool, counts: _Int, norm_product: _Float,
) -> _Float:
    # Batch the near-zero rows. Arbitrary-precision integer moments keep exact
    # cancellation and small signed differences independent of summation order.
    left, scale_x = _integer_moment_rows(x, paired, counts)
    right, scale_y = _integer_moment_rows(y, paired, counts)
    numerator = counts.astype(object) * (left * right).sum(axis=1) - left.sum(axis=1) * right.sum(axis=1)
    nonzero = numerator != 0
    values = np.zeros(len(counts))
    if nonzero.any():
        mantissa, exponent = np.frexp(norm_product[nonzero] * counts[nonzero])
        precision = np.finfo(mantissa.dtype).nmant + 1
        shift = scale_x[nonzero] + scale_y[nonzero] + exponent - precision
        numerator = numerator[nonzero] << np.maximum(-shift, 0).astype(object)
        denominator = _integer_significand(mantissa, precision) << np.maximum(shift, 0).astype(object)
        values[nonzero] = numerator / denominator
    return values


def _correlate(
    x: np.ndarray, y: np.ndarray, paired: _Bool, axis: int, min_samples: int,
    *, refine_pearson: bool = False, legacy_rank_covariance: bool = False, ranked: bool = False,
) -> CorrelationResult:
    counts = paired.sum(axis=1)
    if ranked:
        # Pairwise average ranks always have mean (n + 1) / 2, even with ties.
        center = (counts[:, None] + 1) * .5
        centered_x = np.subtract(x, center, out=x, where=paired)
        centered_y = np.subtract(y, center, out=y, where=paired)
        centered_x[~paired] = 0
        centered_y[~paired] = 0
    else:
        centered_x, constant_x = _center(x, paired, counts)
        centered_y, constant_y = _center(y, paired, counts)
    insufficient = counts < min_samples

    numerator = np.einsum("ij,ij->i", centered_x, centered_y)
    sum_sq_x = np.einsum("ij,ij->i", centered_x, centered_x)
    sum_sq_y = np.einsum("ij,ij->i", centered_y, centered_y)
    if ranked:
        constant_x, constant_y = sum_sq_x == 0, sum_sq_y == 0
    reasons = np.full(len(counts), None, dtype=object)
    reasons[constant_y] = "constant_return"
    reasons[constant_x] = "constant_factor"
    reasons[insufficient] = "insufficient_samples" if axis in (0, -2) else "insufficient_cross_section"
    # One square root preserves unit correlation for identical centered rows.
    norm_product = np.sqrt(sum_sq_x * sum_sq_y)
    values = np.full(len(counts), np.nan)
    estimable = ~(insufficient | constant_x | constant_y)
    if legacy_rank_covariance:
        # Legacy SciPy Spearman reads np.corrcoef's [1, 0] element.
        inverse_ddof = 1.0 / (counts[estimable] - 1)
        values[estimable] = (numerator[estimable] * inverse_ddof
                             / np.sqrt(sum_sq_y[estimable] * inverse_ddof)
                             / np.sqrt(sum_sq_x[estimable] * inverse_ddof))
    else:
        np.divide(numerator, norm_product, out=values, where=estimable)
        two_pairs = estimable & (counts == 2)
        values[two_pairs] = np.copysign(1.0, numerator[two_pairs])
    if refine_pearson:
        near_unit = (counts > 2) & (np.abs(values) > .99)
        # For unit vectors the squared distance to the aligned vector is 2(1-|r|).
        sign = np.sign(values[near_unit])
        left = centered_x[near_unit] / np.sqrt(sum_sq_x[near_unit, None])
        right = centered_y[near_unit] / np.sqrt(sum_sq_y[near_unit, None])
        residual = left - sign[:, None] * right
        values[near_unit] = sign * (1 - .5 * np.einsum("ij,ij->i", residual, residual))

        # Only recompute rows where accumulation error can change the sign.
        rounding_bound = np.finfo(np.float64).eps * (counts + 4)
        near_zero = np.abs(values) <= rounding_bound
        if near_zero.any():
            values[near_zero] = _pearson_after_cancellation(
                x[near_zero], y[near_zero], paired[near_zero], counts[near_zero], norm_product[near_zero],
            )
    return CorrelationResult(np.clip(values, -1.0, 1.0), counts, reasons)


def pearson(
    factor: npt.ArrayLike, returns: npt.ArrayLike, *, axis: int = 1,
    valid: npt.ArrayLike | None = None, min_samples: int = 2,
) -> CorrelationResult:
    """Compute Pearson correlation for each slice of paired observations."""
    minimum = _positive_int(min_samples, "min_samples", 2)
    x, y, paired = _pairs(factor, returns, valid, axis)
    return _correlate(x, y, paired, axis, minimum, refine_pearson=True)


def _compact_average_ranks(values: np.ndarray) -> _Float:
    """Rank a vector whose invalid entries have already been removed."""
    order = np.argsort(values)
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    sorted_values = values[order]
    ties = np.flatnonzero(sorted_values[1:] == sorted_values[:-1])
    if ties.size:
        starts = np.r_[0, np.flatnonzero(np.diff(ties) > 1) + 1]
        ends = np.r_[starts[1:] - 1, len(ties) - 1]
        averages = np.repeat((ties[starts] + ties[ends] + 3) * .5, ends - starts + 1)
        ranks[order[ties]] = averages
        ranks[order[ties + 1]] = averages
    return ranks[None, :]


def spearman(
    factor: npt.ArrayLike, returns: npt.ArrayLike, *, axis: int = 1,
    valid: npt.ArrayLike | None = None, min_samples: int = 2,
    _legacy_covariance: bool = False,
) -> CorrelationResult:
    """Compute Pearson correlation of the two pairwise average ranks."""
    minimum = _positive_int(min_samples, "min_samples", 2)
    x, y, paired = _pairs(factor, returns, valid, axis)
    if len(x) == 1 and x.shape[1] >= 65_536:
        # In long single-series calls, use direct vector sorting/scattering
        # and avoid the general axis index arrays for masked cells.
        x_rank = _compact_average_ranks(x[paired])
        y_rank = _compact_average_ranks(y[paired])
        return _correlate(
            x_rank, y_rank, np.ones(x_rank.shape, dtype=bool), axis, minimum,
            legacy_rank_covariance=_legacy_covariance, ranked=True,
        )
    return _correlate(
        _rank_rows(x, paired), _rank_rows(y, paired), paired, axis, minimum,
        legacy_rank_covariance=_legacy_covariance, ranked=True,
    )


def _left_search(
    edges: _Float, keys: _Float, low: int | _Int, high: _Int,
) -> _Int:
    """Batched searchsorted with explicit bounds and NaNs ordered last."""
    row_shape = (len(edges),) + (1,) * (keys.ndim - 1)
    row_ids = np.arange(len(edges)).reshape(row_shape)
    capacity = edges.shape[1]
    for _ in range(capacity.bit_length()):
        mid = (low + high) // 2
        advance = (low < high) & (edges[row_ids, np.minimum(mid, capacity - 1)] < keys)
        high = np.where(advance, high, mid)
        low = np.where(advance, mid + 1, low)
    return low


def _search_unordered_edges(edges: _Float, keys: _Float, counts: _Int, eligible: _Bool) -> _Int:
    # On unordered edges, NumPy's bounds depend on the previous key and insertion index.
    width = keys.shape[1]
    previous = np.full(keys.shape, -1, dtype=np.intp)
    observed = np.where(eligible[:, :-1], np.arange(width - 1), -1)
    previous[:, 1:] = np.maximum.accumulate(observed, axis=1)
    prior_key = np.take_along_axis(keys, np.maximum(previous, 0), axis=1)
    increasing = (previous >= 0) & (prior_key < keys)
    states = np.arange(edges.shape[1] + 1)
    limit = counts[:, None, None]
    low = np.where(increasing[:, :, None], np.minimum(states, limit), 0)
    reset_high = increasing | (previous < 0)
    high = np.where(reset_high[:, :, None], limit, np.minimum(states + 1, limit))

    # Compose each key's transition in parallel prefixes, including identity for masked keys.
    transitions = _left_search(edges, keys[:, :, None], low, high)
    np.copyto(transitions, states, where=~eligible[:, :, None])
    for bit in range((width - 1).bit_length()):
        step = 1 << bit
        transitions[:, step:] = np.take_along_axis(transitions[:, step:], transitions[:, :-step], axis=2)
    return transitions[:, :, 0]


def quantile_buckets(
    factor: npt.ArrayLike, n_quantiles: int, *, axis: int = 1,
    valid: npt.ArrayLike | None = None,
) -> QuantileResult:
    """Assign right-closed qcut bins to finite, eligible factor values."""
    n_quantiles = _positive_int(n_quantiles, "n_quantiles")
    rows = _real_rows(factor, axis)
    eligible = _eligible(np.isfinite(rows), valid, axis)
    labels = np.zeros(rows.shape, dtype=np.intp)
    edges = np.full((len(rows), n_quantiles + 1), np.nan, dtype=np.result_type(rows.dtype, np.float64))
    counts = eligible.sum(axis=1)
    n_bins = np.zeros_like(counts)
    if eligible.any():
        # Interpolate only within the finite prefix, using the same arithmetic as NumPy quantile.
        sentinel = np.iinfo(rows.dtype).max if rows.dtype.kind in "iu" else np.inf
        last = np.maximum(counts - 1, 0)[:, None]
        # pandas converts quantiles to percentiles before NumPy normalizes them.
        percentiles = np.linspace(0.0, 1.0, n_quantiles + 1) * 100.0
        positions = last * (percentiles / 100.0)
        lower = positions.astype(np.intp)
        upper = np.minimum(lower + 1, last)
        ordered = np.where(eligible, rows, sentinel)
        if len(rows) == 1 and 2 * (n_quantiles + 1) < rows.shape[1]:
            # Long time-series need only the order statistics at bucket edges.
            ordered.partition(np.unique(np.r_[lower[0], upper[0]]), axis=1)
        else:
            ordered.sort(axis=1)
        left = np.take_along_axis(ordered, lower, axis=1)
        right = np.take_along_axis(ordered, upper, axis=1)
        left[counts == 0] = 0.0
        right[counts == 0] = 0.0
        weight = positions - lower
        with np.errstate(over="ignore", invalid="ignore"):
            gap = right - left
            raw_edges = np.where(weight < 0.5, left + gap * weight, right - gap * (1 - weight))

        # qcut keeps distinct edges in first-occurrence order, even after overflow.
        unordered = (np.isnan(raw_edges).any(axis=1)
                     | (raw_edges[:, 1:] < raw_edges[:, :-1]).any(axis=1))
        keep = np.ones(raw_edges.shape, dtype=bool)
        if n_quantiles > 1:
            keep[:, 1:] = raw_edges[:, 1:] != raw_edges[:, :-1]
            if unordered.any():
                order = np.argsort(raw_edges[unordered], kind="stable", axis=1)
                sorted_edges = np.take_along_axis(raw_edges[unordered], order, axis=1)
                prior, following = sorted_edges[:, :-1], sorted_edges[:, 1:]
                distinct = np.ones(sorted_edges.shape, dtype=bool)
                distinct[:, 1:] = (following != prior) & ~(np.isnan(following) & np.isnan(prior))
                deduplicated = np.empty_like(distinct)
                np.put_along_axis(deduplicated, order, distinct, axis=1)
                keep[unordered] = deduplicated
        keep[counts == 0] = False
        edge_count = keep.sum(axis=1)
        n_bins = np.maximum(edge_count - 1, 0)
        edge_rows, _ = np.nonzero(keep)
        edges[edge_rows, (np.cumsum(keep, axis=1) - 1)[keep]] = raw_edges[keep]

        keys = rows.astype(edges.dtype, copy=False)
        if len(rows) == 1 and not unordered[0]:
            ids = np.searchsorted(edges[0, :edge_count[0]], keys)
        elif n_quantiles <= 8:
            ids = np.zeros(keys.shape, dtype=np.intp)
            for boundary in edges.T:
                ids += boundary[:, None] < keys
        else:
            ids = _left_search(edges, keys, 0, edge_count[:, None])
        if unordered.any():
            ids[unordered] = _search_unordered_edges(
                edges[unordered], keys[unordered], edge_count[unordered], eligible[unordered],
            )
        ids[keys == edges[:, :1]] = 1
        labels = np.where(eligible & (ids > 0) & (ids < edge_count[:, None]), ids, 0)
    return QuantileResult(_rows(labels, axis), _rows(edges, axis), n_bins)


def bucket_sums(
    values: npt.ArrayLike, labels: npt.ArrayLike, n_buckets: int, *, axis: int = 1,
    valid: npt.ArrayLike | None = None,
) -> BucketResult:
    """Sum and count finite, eligible values by slice and positive bucket number."""
    n_buckets = _positive_int(n_buckets, "n_buckets")
    rows, buckets = _real_rows(values, axis), _rows(labels, axis)
    if buckets.shape != rows.shape or buckets.dtype.kind not in "iu":
        raise ValueError("labels must be an integer array with the panel shape")
    if np.any(buckets < 0) or np.any(buckets > n_buckets):
        raise ValueError("labels must be in [0, n_buckets]")

    include = _eligible(np.isfinite(rows) & (buckets != 0), valid, axis)
    # Route unused cells to bin zero. Row-major traversal retains the
    # accumulation order of the finite samples in every occupied bucket.
    ids = np.zeros(buckets.shape, dtype=np.intp)
    offsets = np.arange(len(rows), dtype=np.intp)[:, None] * n_buckets
    np.add(buckets.astype(np.intp, copy=False), offsets, out=ids, where=include)
    flat_ids = ids.ravel()
    shape = (len(rows), n_buckets)
    size = len(rows) * n_buckets + 1
    weights = np.where(include, rows, 0).astype(np.float64, copy=False)
    sums = np.bincount(flat_ids, weights=weights.ravel(), minlength=size)[1:].astype(np.float64, copy=False)
    counts = np.bincount(flat_ids, minlength=size)[1:]
    return BucketResult(_rows(sums.reshape(shape), axis), _rows(counts.reshape(shape), axis))
