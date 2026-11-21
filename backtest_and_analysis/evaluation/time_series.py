"""Instrument-level statistics on an expected signal grid for one split."""

from collections.abc import Hashable, Mapping, Set
from dataclasses import replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import kernels
from .calibration import resolve_interval_calibration
from .cross_section import _directional_accuracy, _directional_masks, _time_mean
from .kernels import CorrelationResult, QuantileResult
from .panel import EvaluationPanel
from .quantiles import QuantileFit
from .types import (
    BOOTSTRAP_V1,
    BOOTSTRAP_V2,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_MIN_VALID_REPLICATES,
    BootstrapInterval,
    DirectionalAccuracy,
    FloatArray,
    ICMethod,
    InstrumentICSummary,
    ReturnDistribution,
    TimeSeriesHorizon,
    TimeSeriesInstrument,
    TimeSeriesResult,
)


def _ppw_block_length(scores: FloatArray, n: int, minimum: int, maximum: int) -> int:
    """Select the CBB block length from PPW flat-top grid autocovariances."""
    size = len(scores)
    variance = float(np.dot(scores, scores) / size)
    if n < 11 or variance <= 1e-12:
        return minimum

    k_n = max(5, int(np.ceil(np.log10(n))))
    max_lag = min(size - 1, int(np.ceil(np.sqrt(n))) + k_n)
    threshold = 2 * np.sqrt(np.log10(n) / n)
    covariance = np.array([np.dot(scores[k:], scores[:-k]) / size for k in range(1, max_lag + 1)])
    quiet = np.abs(covariance) / variance < threshold
    bandwidth = max_lag
    for a in range(max_lag - k_n + 1):
        if quiet[a:a + k_n].all():
            bandwidth = min(max_lag, 2 * max(1, a))
            break
    lags = np.arange(1, bandwidth + 1)
    windowed = np.minimum(1, 2 * (1 - lags / bandwidth)) * covariance[:bandwidth]
    g = float(2 * np.dot(lags, windowed))
    spectral_sum = float(variance + 2 * windowed.sum())
    if spectral_sum <= 1e-12 * variance:
        return maximum
    optimal = np.cbrt(1.5 * size) * (abs(g) / spectral_sum)**(2 / 3)
    return int(np.ceil(np.clip(optimal, minimum, maximum)))


def _circular_block_indices(
    rng: np.random.Generator, size: int, offsets: npt.NDArray[np.intp],
) -> npt.NDArray[np.intp]:
    length = len(offsets)
    starts = rng.integers(0, size, size=(size + length - 1) // length, dtype=np.int64)
    return ((starts[:, None] + offsets) % size).ravel()[:size]


def _rank_margin(values: FloatArray) -> tuple[FloatArray, float, npt.NDArray[np.intp]]:
    """Return centered midrank CDF values, their scale and sorted tie-group IDs."""
    _, groups, counts = np.unique(values, return_inverse=True, return_counts=True)
    centered = (np.cumsum(counts) - (counts + len(values)) / 2)[groups] / len(values)
    return centered, float(np.sqrt(np.mean(centered**2))), groups


def _rank_influence(groups: npt.NDArray[np.intp], centered_cdf: FloatArray, weights: FloatArray) -> FloatArray:
    # H gives each tied group half weight and every group above it full weight.
    totals = np.bincount(groups, weights=weights)
    suffix = np.cumsum(totals[::-1])[::-1] - 0.5 * totals
    return suffix[groups] - np.dot(weights, centered_cdf + 0.5)


def _spearman_scores(x: FloatArray, y: FloatArray) -> tuple[float, FloatArray, FloatArray] | None:
    fx, sx, x_groups = _rank_margin(x)
    fy, sy, y_groups = _rank_margin(y)
    if sx == 0 or sy == 0:
        return None
    u, v = fx / sx, fy / sy
    score = u * v
    rho = float(np.clip(score.mean(), -1, 1))
    # Exact rank agreement has zero influence even if the mean rounds below one.
    if np.array_equal(u, v):
        rho = 1.0
    elif np.array_equal(u, -v):
        rho = -1.0
    influence = score - rho * (u**2 + v**2) / 2
    n = len(x)
    influence += _rank_influence(x_groups, fx, v - rho * u) / (n * sx)
    influence += _rank_influence(y_groups, fy, u - rho * v) / (n * sy)
    return rho, score - rho, influence


def _fisher_grid_scores(
    rho: float, influence: FloatArray, valid: npt.NDArray[np.bool_],
) -> tuple[float, FloatArray, float]:
    clipped = float(np.clip(rho, -1 + 1e-6, 1 - 1e-6))
    n = len(influence)
    scores = np.zeros(len(valid))
    scores[valid] = (influence - influence.mean()) / (1 - clipped**2)
    return clipped, scores, float(np.dot(scores, scores) / n**2)


def _fisher_studentization(
    rho: float, influence: FloatArray, valid: npt.NDArray[np.bool_], block_length: int,
) -> tuple[float, float]:
    """Historical v1 circular-window studentization."""
    n, size = len(influence), len(valid)
    clipped, scores, v0 = _fisher_grid_scores(rho, influence, valid)
    if not np.isfinite(v0) or v0 <= 1e-24 / n:
        return np.nan, np.nan
    prefix = np.r_[0.0, np.cumsum(np.r_[scores, scores[:block_length - 1]])]
    q, r = divmod(size, block_length)
    variance_sum = 0.0
    for count, width in ((q, block_length), (1, r)):
        if width:
            sums = prefix[width:width + size] - prefix[:size]
            variance_sum += count * np.dot(sums, sums)
    variance = variance_sum / (size * n**2)
    if not np.isfinite(variance) or variance <= 1e-12 * v0:
        return np.nan, np.nan
    return float(np.arctanh(clipped)), float(np.sqrt(variance))


def _flat_top_variance(
    scores: FloatArray, n: int, length: int, remainder: int, v0: float,
) -> tuple[float, float]:
    """Return the original variance and frozen residual-to-full block variance ratio."""
    size = len(scores)
    prefix = np.r_[0.0, np.cumsum(np.r_[scores, scores[:length - 1]])]

    def window_energy(width: int) -> float:
        sums = prefix[width:width + size] - prefix[:size]
        return float(np.dot(sums, sums) / size)

    floor = 1e-12 * v0 * n**2 / size
    full = window_energy(length)
    ratio = 0.0
    if remainder:
        if not np.isfinite(full) or full / length <= floor:
            return np.nan, np.nan
        ratio = window_energy(remainder) / full
        if not np.isfinite(ratio) or ratio < 0:
            return np.nan, np.nan
    k = length // 2
    spectrum = np.nan
    if k:
        spectrum = (2 * window_energy(2 * k) / (2 * k) - window_energy(k) / k) / (1 - 3 * k / size)
    if not np.isfinite(spectrum) or spectrum <= floor:
        spectrum = (full / length) / (1 - length / size)
    return size * spectrum / n**2, ratio


def _independent_block_variance(scores: FloatArray, n: int, length: int, ratio: float) -> float:
    q = len(scores) // length
    sums = scores[:q * length].reshape(q, length).sum(axis=1)
    return float((q + ratio) * np.var(sums, ddof=1) / n**2)


def _fisher_v2_studentization(
    rho: float, influence: FloatArray, valid: npt.NDArray[np.bool_], length: int,
    *, residual_ratio: float | None = None,
) -> tuple[float, float, float]:
    n = len(influence)
    clipped, scores, v0 = _fisher_grid_scores(rho, influence, valid)
    if not np.isfinite(v0) or v0 <= 1e-24 / n:
        return np.nan, np.nan, np.nan
    if residual_ratio is None:
        variance, residual_ratio = _flat_top_variance(scores, n, length, len(valid) % length, v0)
    else:
        variance = _independent_block_variance(scores, n, length, residual_ratio)
    if not np.isfinite(variance) or variance <= 1e-12 * v0:
        return np.nan, np.nan, np.nan
    return float(np.arctanh(clipped)), float(np.sqrt(variance)), residual_ratio


def _missing_pattern(valid: npt.NDArray[np.bool_]) -> str:
    if valid.all():
        return "none"
    cycle = np.arange(len(valid)) % 17
    for gaps in (3, 5):
        if np.array_equal(valid, cycle < 17 - gaps):
            return f"periodic_17_{gaps}"
    return "other"


def _adaptive_bootstrap_spearman(
    factor: FloatArray, returns: FloatArray, valid: npt.NDArray[np.bool_],
    *, point: CorrelationResult, min_samples: int, min_block_length: int, seed: int,
    n_boot: int, min_valid_replicates: int, alpha: float, calibration_scenario: str | None,
    intervals_verified: bool, historical_v1: bool,
) -> BootstrapInterval:
    valid = valid & np.isfinite(factor) & np.isfinite(returns)
    size, n = len(valid), int(valid.sum())
    min_pairs = max(4, min_samples)
    version, selector, resampler, interval_method = BOOTSTRAP_V1 if historical_v1 else BOOTSTRAP_V2
    result = BootstrapInterval(
        size, 0, seed, n_boot if intervals_verified else 0, 0, min_valid_replicates, alpha,
        np.nan, np.nan, point.reasons[0] if intervals_verified else "holding_intervals_unavailable",
        n_valid_pairs=n, method_version=version, missing_pattern=_missing_pattern(valid),
        calibration_scenario=calibration_scenario, block_length_method=selector,
        resample_method=resampler, interval_method=interval_method,
        min_block_length=min_block_length, min_pairs_per_replicate=min_pairs,
    )
    if result.reason is not None:
        return result
    if n < min_pairs:
        return replace(result, reason="insufficient_samples")
    maximum = min(n // 4, int(np.ceil(min(3 * np.sqrt(size), size / 3))))
    if min_block_length > maximum:
        return replace(result, reason="insufficient_bootstrap_grid")

    original = _spearman_scores(factor[valid], returns[valid])
    if original is None:
        return replace(result, reason="degenerate_bootstrap_standard_error")
    rho, scores, influence = original
    grid_scores = np.zeros(size)
    grid_scores[valid] = scores
    length = _ppw_block_length(grid_scores, n, min_block_length, maximum)
    result = replace(result, block_length=length, effective_blocks=n / length)
    if historical_v1:
        fisher_z, standard_error = _fisher_studentization(rho, influence, valid, length)
    else:
        fisher_z, standard_error, residual_ratio = _fisher_v2_studentization(rho, influence, valid, length)
    if not np.isfinite(standard_error):
        return replace(result, reason="degenerate_bootstrap_standard_error")

    rng = np.random.Generator(np.random.PCG64(seed))
    offsets = np.arange(length)
    pivots = []
    for _ in range(n_boot):
        rows = _circular_block_indices(rng, size, offsets)
        paired = valid[rows]
        if paired.sum() < min_pairs:
            continue
        paired_rows = rows[paired]
        sample = _spearman_scores(factor[paired_rows], returns[paired_rows])
        if sample is None:
            continue
        rho_star, _, influence_star = sample
        if historical_v1:
            fisher_star, se_star = _fisher_studentization(rho_star, influence_star, paired, length)
        else:
            fisher_star, se_star, _ = _fisher_v2_studentization(
                rho_star, influence_star, paired, length, residual_ratio=residual_ratio,
            )
        pivot = (fisher_star - fisher_z) / se_star
        if np.isfinite(pivot):
            pivots.append(pivot)
    result = replace(result, valid_replicates=len(pivots), fisher_z_standard_error=standard_error)
    if len(pivots) < min_valid_replicates:
        return replace(result, reason="insufficient_bootstrap_replicates")
    quantiles = np.quantile(pivots, [1 - alpha / 2, alpha / 2], method="linear")
    inverted = fisher_z - quantiles * standard_error
    lower, upper = np.tanh(inverted)
    if not np.isfinite(inverted).all() or not np.isfinite([lower, upper]).all() or lower > upper:
        return replace(result, reason="degenerate_bootstrap_standard_error")
    return replace(result, lower=float(lower), upper=float(upper))


def _bootstrap_spearman(
    factor: FloatArray, returns: FloatArray, valid: npt.NDArray[np.bool_],
    *, point: CorrelationResult, min_samples: int, block_length: int | Literal["auto"] = "auto",
    min_block_length: int = 1, seed: int = 0, n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    min_valid_replicates: int = DEFAULT_MIN_VALID_REPLICATES, alpha: float = 0.05,
    calibration_scenario: str | None = None, intervals_verified: bool = True,
    historical_v1: bool = False,
) -> BootstrapInterval:
    if isinstance(block_length, str) and block_length == "auto":
        if not 0 < alpha < 1 or alpha / 2 <= 0:
            raise ValueError("alpha must be in (0, 1) with a positive representable half-tail")
        return _adaptive_bootstrap_spearman(
            factor, returns, valid, point=point, min_samples=kernels._positive_int(min_samples, "min_samples", 2),
            min_block_length=kernels._positive_int(min_block_length, "min_block_length"),
            seed=kernels._positive_int(seed, "seed", 0), n_boot=kernels._positive_int(n_boot, "n_boot"),
            min_valid_replicates=kernels._positive_int(min_valid_replicates, "min_valid_replicates", 2),
            alpha=alpha, calibration_scenario=calibration_scenario, intervals_verified=intervals_verified,
            historical_v1=historical_v1,
        )
    block_length = kernels._positive_int(block_length, "block_length")
    size = len(factor)
    if not intervals_verified:
        return BootstrapInterval(
            size, block_length, seed, 0, 0, min_valid_replicates, alpha,
            np.nan, np.nan, "holding_intervals_unavailable",
        )
    lower = upper = np.nan
    n_valid = 0
    reason = point.reasons[0]
    if reason is None and block_length >= size:
        reason = "insufficient_bootstrap_grid"
    if reason is None:
        rng = np.random.default_rng(seed)
        n_blocks = (size + block_length - 1) // block_length
        offsets = np.arange(block_length)
        replicates = np.empty(n_boot)
        for draw in range(n_boot):
            starts = rng.integers(size - block_length + 1, size=n_blocks)
            rows = (starts[:, None] + offsets).ravel()[:size]
            replicates[draw] = kernels.spearman(
                factor[rows, None], returns[rows, None], valid=valid[rows, None],
                axis=0, min_samples=min_samples,
            ).values[0]
        estimates = replicates[np.isfinite(replicates)]
        n_valid = len(estimates)
        if n_valid < min_valid_replicates:
            reason = "insufficient_bootstrap_replicates"
        else:
            lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2], method="linear")
    return BootstrapInterval(
        size, block_length, seed, n_boot, n_valid, min_valid_replicates,
        alpha, float(lower), float(upper), reason,
        n_valid_pairs=int(point.counts[0]),
    )


def _distribution(returns: FloatArray, empty_reason: str, *, location_only: bool = False) -> ReturnDistribution:
    if location_only:
        if not returns.size:
            return ReturnDistribution(n_samples=0, reason=empty_reason)
        return ReturnDistribution(n_samples=len(returns), mean=float(returns.mean()),
                                  p50=float(np.percentile(returns, 50, method="linear")))
    summary = _time_mean(returns, empty_reason)
    if not summary.n_times:
        return ReturnDistribution(n_samples=0, reason=summary.reason)
    p5, p25, p50, p75, p95 = np.percentile(returns, [5, 25, 50, 75, 95], method="linear")
    return ReturnDistribution(
        n_samples=summary.n_times,
        mean=summary.mean,
        standard_error=summary.standard_error,
        p5=float(p5), p25=float(p25), p50=float(p50), p75=float(p75), p95=float(p95),
        mean_le_p5=float(returns[returns <= p5].mean()),
        mean_le_p25=float(returns[returns <= p25].mean()),
        mean_ge_p75=float(returns[returns >= p75].mean()),
        mean_ge_p95=float(returns[returns >= p95].mean()),
        reason=summary.reason,
    )


def _direction(factor: FloatArray, returns: FloatArray, direction: int) -> DirectionalAccuracy:
    long, short = _directional_masks(factor, direction)
    active = long | short
    up = returns > 0
    hits = (long & up) | (short & (returns < 0))
    return _directional_accuracy(hits.sum(keepdims=True), active.sum(keepdims=True),
                                 (active & up).sum(keepdims=True))


def _summarize_ics(ics: list[CorrelationResult]) -> InstrumentICSummary:
    values = np.array([ic.values[0] for ic in ics])
    estimates = values[np.isfinite(values)]
    if not len(estimates):
        return InstrumentICSummary(len(ics), 0, np.nan, np.nan, "no_valid_instruments")
    return InstrumentICSummary(
        len(ics), len(estimates), float(estimates.mean()), float(np.median(estimates)), None,
    )


def summarize_time_series_horizon(
    instruments: dict[Hashable, TimeSeriesInstrument], offsets: range,
) -> TimeSeriesHorizon:
    """Summarize per-instrument ICs with equal weights on each available phase."""
    return TimeSeriesHorizon(
        instruments,
        _summarize_ics([result.ic for result in instruments.values()]),
        {
            offset: _summarize_ics([result.non_overlapping_ic[offset] for result in instruments.values()])
            for offset in offsets
        },
    )


def _validate_bootstrap_options(
    *, block_length: int | Literal["auto"] = "auto", seed: int = 0,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES, min_valid_replicates: int = DEFAULT_MIN_VALID_REPLICATES,
    alpha: float = 0.05,
) -> tuple[int | Literal["auto"], int, int, int]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    auto_block = isinstance(block_length, str) and block_length == "auto"
    if auto_block:
        if alpha / 2 <= 0:
            raise ValueError("alpha must have a positive representable half-tail")
    else:
        block_length = kernels._positive_int(block_length, "block_length")
    seed = kernels._positive_int(seed, "seed", 0)
    n_boot = kernels._positive_int(n_boot, "n_boot")
    min_valid_replicates = kernels._positive_int(min_valid_replicates, "min_valid_replicates", 2)

    return block_length, seed, n_boot, min_valid_replicates


def _validate_block_length_for_horizon(block_length: int | Literal["auto"], horizon: int) -> None:
    if block_length != "auto" and block_length < horizon:
        raise ValueError(f"block_length must be >= horizon ({horizon})")


def evaluate_time_series(
    panel: EvaluationPanel,
    holding_intervals: Mapping[int, tuple[npt.ArrayLike, npt.ArrayLike]] | None,
    *,
    block_length: int | Literal["auto"] = "auto",
    seed: int = 0,
    n_boot: int = DEFAULT_BOOTSTRAP_REPLICATES,
    min_valid_replicates: int = DEFAULT_MIN_VALID_REPLICATES,
    alpha: float = 0.05,
    min_samples: int = 3,
    ic_method: ICMethod = "spearman",
    n_quantiles: int = 5,
    direction: int = 1,
    signal_valid: npt.ArrayLike | None = None,
    expected_grid: npt.ArrayLike | None = None,
    bucket_labels: npt.ArrayLike | None = None,
    quantile_fit: QuantileFit | None = None,
    calibration_scenario: str | None = None,
    all_non_overlapping_phases: bool = True,
    return_diagnostics: bool = True,
    _point_estimates: Mapping[Hashable, TimeSeriesInstrument] | None = None,
    _reuse_phase_zero: Set[Hashable] = frozenset(),
) -> TimeSeriesResult:
    """Evaluate each instrument in a single frequency and split, retaining grid gaps.

    Without intervals the result is descriptive: phase ICs are positional,
    and no bootstrap interval is produced.
    """
    if not panel.datetimes.is_monotonic_increasing:
        raise ValueError("panel.datetimes must be in chronological order")
    if not isinstance(all_non_overlapping_phases, bool):
        raise ValueError("all_non_overlapping_phases must be a boolean")
    if not isinstance(return_diagnostics, bool):
        raise ValueError("return_diagnostics must be a boolean")
    if ic_method not in ("spearman", "pearson"):
        raise ValueError("ic_method must be 'spearman' or 'pearson'")
    kernels._validate_direction(direction)
    min_samples = kernels._positive_int(min_samples, "min_samples", 2)
    n_quantiles = kernels._positive_int(n_quantiles, "n_quantiles", 2)
    block_length, seed, n_boot, min_valid_replicates = _validate_bootstrap_options(
        block_length=block_length, seed=seed, n_boot=n_boot, min_valid_replicates=min_valid_replicates, alpha=alpha)
    auto_block = block_length == "auto"
    if holding_intervals is not None and holding_intervals.keys() != panel.returns.keys():
        raise ValueError("holding_intervals must contain the same horizons as the panel")

    factor = panel.factor
    eligible = np.isfinite(factor)
    for name, mask in (("signal_valid", signal_valid), ("expected_grid", expected_grid)):
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != factor.shape or mask.dtype.kind != "b":
                raise ValueError(f"{name} must be a boolean mask with the panel shape")
            eligible &= mask
    grid = None if expected_grid is None else np.asarray(expected_grid)

    directional_signal = factor
    if quantile_fit is not None:
        if bucket_labels is not None:
            raise ValueError("quantile_fit and bucket_labels are mutually exclusive")
        quantiles = quantile_fit.assign(factor, panel.assets, eligible)
        directional_signal = quantile_fit.directional_signal(factor)
        direction = quantile_fit.direction
    elif bucket_labels is None:
        quantiles = kernels.quantile_buckets(factor, n_quantiles, axis=0, valid=eligible)
    else:
        labels = np.asarray(bucket_labels)
        if (labels.shape != factor.shape or labels.dtype.kind not in "iu"
                or np.any((labels < 0) | (labels > n_quantiles))):
            raise ValueError("bucket_labels must be integer labels in [0, n_quantiles] with the panel shape")
        quantiles = QuantileResult(
            np.where(eligible, labels, 0).astype(np.intp),
            np.full((n_quantiles + 1, factor.shape[1]), np.nan),
            np.full(factor.shape[1], n_quantiles, dtype=np.intp),
        )
    unformed = quantiles.n_bins < 2
    quantiles.labels[:, unformed] = 0
    quantile_reasons = np.where(unformed, "unable_to_form_quantiles", None)

    correlate = kernels.spearman if ic_method == "spearman" else kernels.pearson
    horizons = {}
    for key, returns in panel.returns.items():
        horizon = kernels._positive_int(key, "horizon")
        _validate_block_length_for_horizon(block_length, horizon)
        if holding_intervals is not None:
            entry, exit = (np.asarray(timestamps) for timestamps in holding_intervals[horizon])
            if any(a.shape != factor.shape or a.dtype.kind != "M" for a in (entry, exit)):
                raise ValueError("entry_timestamp and exit_timestamp must be datetime64 arrays with the panel shape")

        valid = panel.valid[horizon] & eligible & np.isfinite(returns)
        offsets = range(horizon if all_non_overlapping_phases else 1)
        # Verify every scheduled holding interval even when only phase zero is reported.
        checked_offsets = range(horizon) if holding_intervals is not None else offsets
        instruments = {}
        for column, asset in enumerate(panel.assets):
            rows = slice(None) if grid is None or grid[:, column].all() else grid[:, column]
            x, y = factor[rows, column], returns[rows, column]
            signal = directional_signal[rows, column]
            pairs = valid[rows, column]
            if holding_intervals is not None:
                entries, exits = entry[rows, column], exit[rows, column]
                if np.any(pairs & (np.isnat(entries) | np.isnat(exits) | (entries >= exits))):
                    raise ValueError(f"Expected entry_timestamp < exit_timestamp for {asset!r}, horizon {horizon}")

            reference = None if _point_estimates is None else _point_estimates.get(asset)
            if reference is None:
                spearman = kernels.spearman(
                    x[:, None], y[:, None], valid=pairs[:, None], axis=0, min_samples=min_samples,
                )
                ic = spearman if ic_method == "spearman" else correlate(
                    x[:, None], y[:, None], valid=pairs[:, None], axis=0, min_samples=min_samples,
                )
            else:
                ic, spearman = reference.ic, reference.spearman_ic
            sampled = {}
            for offset in checked_offsets:
                phase = slice(offset, None, horizon)
                phase_pairs = pairs[phase]
                if holding_intervals is not None:
                    phase_entries, phase_exits = entries[phase][phase_pairs], exits[phase][phase_pairs]
                    order = np.argsort(phase_entries)
                    if np.any(phase_entries[order][1:] < phase_exits[order][:-1]):
                        raise AssertionError(
                            f"Overlapping holding intervals for {asset!r}, horizon {horizon}, offset {offset}",
                        )
                if offset not in offsets:
                    continue
                if offset == 0 and reference is not None and asset in _reuse_phase_zero:
                    sampled[offset] = reference.non_overlapping_ic[0]
                    continue
                sampled[offset] = ic if horizon == 1 else correlate(
                    x[phase, None], y[phase, None], valid=phase_pairs[:, None],
                    axis=0, min_samples=min_samples,
                )

            if holding_intervals is None and not auto_block:
                bootstrap = BootstrapInterval(
                    len(x), block_length, seed, 0, 0, min_valid_replicates, alpha,
                    np.nan, np.nan, "holding_intervals_unavailable",
                )
            else:
                bootstrap = _bootstrap_spearman(
                    x, y, pairs, point=spearman, min_samples=min_samples, block_length=block_length,
                    seed=seed, n_boot=n_boot, min_valid_replicates=min_valid_replicates, alpha=alpha,
                    min_block_length=horizon, calibration_scenario=calibration_scenario,
                    intervals_verified=holding_intervals is not None,
                )
            if calibration_scenario is not None and not auto_block:
                status, evidence = resolve_interval_calibration(
                    scenario=calibration_scenario, method="time_series_mbb", window=bootstrap.block_length,
                    sample_size=bootstrap.grid_size, nominal_coverage=bootstrap.confidence_level,
                )
                bootstrap = replace(bootstrap, calibration_status=status, calibration_evidence=evidence)
            distributions, directions = {}, {}
            if not unformed[column]:
                labels = quantiles.labels[rows, column]
                for bucket in range(1, quantiles.n_bins[column] + 1):
                    members = pairs & (labels == bucket)
                    distributions[bucket] = _distribution(y[members], "empty_bucket",
                                                          location_only=not return_diagnostics)
                    if return_diagnostics:
                        directions[bucket] = _direction(signal[members], y[members], direction)
            instruments[asset] = TimeSeriesInstrument(
                ic=ic,
                spearman_ic=spearman,
                bootstrap=bootstrap,
                returns=(_distribution(y[pairs], "insufficient_samples") if return_diagnostics else
                         ReturnDistribution(n_samples=int(ic.counts[0]), reason="not_requested")),
                returns_by_bucket=distributions,
                direction=(_direction(signal[pairs], y[pairs], direction) if return_diagnostics else
                           DirectionalAccuracy(0, 0, 0, np.nan, np.nan, np.nan, "not_requested")),
                direction_by_bucket=directions,
                non_overlapping_ic=sampled,
            )
        horizons[horizon] = summarize_time_series_horizon(instruments, offsets)
    return TimeSeriesResult(
        panel.datetimes, panel.assets, min_samples, ic_method, direction, quantiles, quantile_reasons, horizons,
        holding_intervals is not None,
    )
