"""Time-weighted statistics on a fixed datetime-by-asset signal grid."""

from collections.abc import Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from math import ceil, erfc, isfinite, nextafter, sqrt
import re
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.stats import t

from . import kernels
from .calibration import resolve_interval_calibration
from .panel import EvaluationPanel
from .types import (
    CrossSectionHorizon,
    CrossSectionResult,
    DirectionalAccuracy,
    FloatArray,
    HACInterval,
    ICSummary,
    ReturnMode,
    TimeMean,
)


_PARALLEL_STATISTIC_CELLS = 1_000_000
_HACLag = int | Literal["auto"]


def _daily_horizon_steps(horizon: Hashable) -> Hashable:
    if isinstance(horizon, str) and re.fullmatch(r"[0-9]+[dD]", horizon):
        return int(horizon[:-1])
    return horizon


def _ic_summary(values: FloatArray) -> ICSummary:
    n = values.size
    if not n:
        return ICSummary(0, np.nan, np.nan, np.nan, np.nan, np.nan, "no_valid_cross_sections")

    mean = float(values.mean())
    median = float(np.median(values))
    positive = float(np.count_nonzero(values > 0) / n)
    if n == 1:
        return ICSummary(n, mean, median, positive, np.nan, np.nan, "insufficient_valid_cross_sections")

    std = float((values - values[0]).std(ddof=1))
    if std == 0:
        return ICSummary(n, mean, median, positive, std, np.nan, "zero_ic_std")
    return ICSummary(n, mean, median, positive, std, mean / std, None)


def bartlett_hac(
    ic: npt.ArrayLike, *, lag: _HACLag = "auto", min_lag: int = 0,
    alpha: float = 0.05, calibration_scenario: str | None = None,
) -> HACInterval:
    """Estimate the IC mean interval, with nonfinite ICs occupying their grid slots."""
    adaptive = isinstance(lag, str) and lag == "auto"
    if adaptive:
        min_lag = kernels._positive_int(min_lag, "min_lag", minimum=0)
    else:
        lag = kernels._positive_int(lag, "lag", minimum=0)
    if not 0 < alpha / 2 < 0.5:
        raise ValueError("alpha must be between 0 and 1 with a representable alpha / 2")
    values = np.asarray(ic, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("ic must be a one-dimensional sequence on the signal grid")
    valid = np.isfinite(values)
    if adaptive:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            return _adaptive_hac(values, valid, min_lag, alpha, calibration_scenario)
    n = int(valid.sum())
    grid_size = values.size

    pair_counts = np.zeros(lag, dtype=np.intp)
    if n >= 2:
        for offset in range(1, min(lag + 1, grid_size)):
            pair_counts[offset - 1] = np.count_nonzero(valid[offset:] & valid[:-offset])
    variance = lower = upper = np.nan
    if n < 2:
        reason = "insufficient_valid_cross_sections"
    elif lag >= grid_size:
        reason = "insufficient_hac_grid"
    else:
        observed = values[valid]
        # Center relative to an observed IC to preserve exact zeros for a constant series.
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = observed - observed[0]
            shifted_mean = shifted.mean()
            scores = np.zeros(grid_size)
            scores[valid] = shifted - shifted_mean
            variance = float((
                np.dot(scores, scores)
                + 2 * sum(
                    (1 - offset / (lag + 1)) * np.dot(scores[offset:], scores[:-offset])
                    for offset in range(1, lag + 1)
                )
            ) / n**2)
        reason = "degenerate_hac_variance"
        if np.isfinite(variance) and variance > 0:
            mean = float(observed[0] + shifted_mean)
            half_width = -NormalDist().inv_cdf(alpha / 2) * sqrt(variance)
            lower, upper = mean - half_width, mean + half_width
            reason = None
        else:
            variance = np.nan

    status, evidence = resolve_interval_calibration(
        scenario=calibration_scenario, method="cross_section_hac", window=lag,
        sample_size=grid_size, nominal_coverage=1 - alpha,
    )
    return HACInterval(
        n, grid_size, lag, alpha, pair_counts, variance, lower, upper, reason,
        calibration_status=status, calibration_evidence=evidence,
    )


def _grid_lag_products(values: FloatArray, lag: int) -> FloatArray:
    """Sum products at offsets 0 through lag on the original grid."""
    if lag <= 64:
        return np.array([values @ values, *(
            values[k:] @ values[:-k] for k in range(1, lag + 1)
        )])
    center = len(values) - 1
    return signal.correlate(values, values, method="fft")[center:center + lag + 1]


def _sliding_sums(values: FloatArray, width: int) -> FloatArray:
    """Sum all windows intersecting the grid, using zero extension at both ends."""
    if width == 1:
        return values
    prefix = np.cumsum(np.pad(values, (width, width - 1)))
    return prefix[width:] - prefix[:-width]


def _adaptive_hac(
    values: FloatArray, valid: npt.NDArray[np.bool_], min_lag: int, alpha: float,
    calibration_scenario: str | None,
) -> HACInterval:
    slots = np.flatnonzero(valid)
    n, grid_size = slots.size, values.size
    pattern = "none" if n == grid_size else "other"
    if pattern == "other":
        phases = np.arange(grid_size) % 17
        for gap in (3, 5):
            if np.array_equal(valid, phases < 17 - gap):
                pattern = f"periodic_17_{gap}"
                break

    interval = HACInterval(
        n, grid_size, 0, alpha, np.zeros(0, dtype=np.intp), np.nan, np.nan, np.nan,
        "degenerate_hac_variance", method_version="cs_hac_adaptive_v1", missing_pattern=pattern,
        calibration_scenario=calibration_scenario, bandwidth_method="nw94_prewhitened", min_lag=min_lag,
    )
    max_lag = (grid_size - 1) // 4
    if n < 3:
        return replace(interval, reason="insufficient_valid_cross_sections")
    if min_lag > max_lag:
        return replace(interval, reason="insufficient_hac_grid")

    observed = values[valid]
    shifted = observed - observed[0]
    residuals = shifted - shifted.mean()
    energy = residuals @ residuals
    mean = float(observed.mean())
    if not isfinite(energy) or energy <= 0 or not isfinite(mean):
        return interval

    distances = np.diff(slots)
    adjacent = distances == 1
    previous = residuals[:-1][adjacent]
    previous_energy = previous @ previous
    phi = 0.0
    if previous.size >= 2 and previous_energy > 1e-12 * energy:
        phi = float(np.clip((previous @ residuals[1:][adjacent]) / previous_energy, -.97, .97))
    interval = replace(interval, prewhitening_ar1=phi)

    transitions = phi ** distances
    scales = np.r_[1., np.sqrt(1 - transitions**2)]
    whitened = np.r_[residuals[0], (residuals[1:] - transitions * residuals[:-1]) / scales[1:]]
    # Solve A.T @ weights = 1 by backward recurrence on the observed slots.
    recurrence = np.ones(n)
    for i in range(n - 2, -1, -1):
        recurrence[i] = 1 + transitions[i] * recurrence[i + 1]
    weights = scales * recurrence

    scores = np.zeros(grid_size)
    scores[valid] = weights * whitened
    # Use one denominator for the grid autocovariances, including across missing slots.
    pilot = min(grid_size - 1, max(1, min_lag, ceil(4 * (grid_size / 100) ** (2 / 9))))
    gamma = _grid_lag_products(scores, pilot) / grid_size
    s0 = gamma[0] + 2 * gamma[1:].sum()
    s1 = 2 * (np.arange(1, pilot + 1) @ gamma[1:])
    if not np.isfinite([s0, s1, gamma[0]]).all():
        return interval
    lag = max_lag
    if s0 > 1e-12 * gamma[0] > 0:
        bandwidth = 1.1447 * np.cbrt(grid_size * np.square(s1 / s0))
        lag = max(min_lag, ceil(np.clip(bandwidth, 1, max_lag + 1)) - 1)
    width = lag + 1
    counts = np.rint(_grid_lag_products(valid.astype(float), lag)[1:]).astype(np.intp)
    interval = replace(interval, lag=lag, pair_counts=counts)

    windows = _sliding_sums(scores, width)
    q = (windows @ windows) / width
    score_energy = scores @ scores
    if not isfinite(q) or not isfinite(score_energy) or q <= 1e-12 * score_energy:
        return interval

    def apply_h(vector: FloatArray) -> FloatArray:
        grid = np.zeros(grid_size)
        grid[valid] = weights * vector
        smoothed = _sliding_sums(_sliding_sums(grid, width), width)
        return weights * smoothed[slots + lag] / width

    # Correct the two traces for the rank-two update induced by estimating the mean.
    ones_whitened = np.r_[1., (1 - transitions) / scales[1:]]
    d = apply_h(ones_whitened) / n
    u = np.column_stack((d, weights))
    c = np.array([[0., -1.], [-1., (ones_whitened @ d) / n]])
    update = c @ (u.T @ u)
    weight_energy = weights @ weights
    trace = weight_energy + np.trace(update)
    squared_weights = np.zeros(grid_size)
    squared_weights[valid] = weights**2
    products = _grid_lag_products(squared_weights, lag)
    kernel = 1 - np.arange(1, width) / width
    h_squared = products[0] + 2 * (kernel**2 @ products[1:])
    hu = np.column_stack((apply_h(d), apply_h(weights)))
    trace_b2 = (
        h_squared + 2 * np.trace(c @ (u.T @ hu))
        + np.trace(update @ update)
    )
    if not np.isfinite([trace, trace_b2]).all() or min(trace, trace_b2) <= 0:
        return interval
    variance = float((q / n**2) * (weight_energy / trace))
    moment_df = trace**2 / trace_b2
    if not isfinite(moment_df):
        return interval
    df = float(min(n - 1, moment_df))
    interval = replace(interval, effective_df=df)
    if not isfinite(variance) or variance <= 0 or df < 1:
        return interval
    critical = float(t.isf(alpha / 2, df))
    half_width = critical * sqrt(variance)
    lower, upper = mean - half_width, mean + half_width
    if not np.isfinite([lower, upper]).all():
        return interval
    return replace(
        interval, variance=variance, lower=lower, upper=upper, reason=None, critical_value=critical,
    )


def student_t_hac_p_value(
    point_estimate: float, variance: float, effective_df: float, alpha: float, lower: float, upper: float,
) -> float:
    """Compute the tail probability with the interval's floating-point decision at alpha."""
    if lower == 0.0 or upper == 0.0:
        return alpha
    raw_p = float(2 * t.sf(abs(point_estimate) / sqrt(variance), effective_df))
    if lower > 0.0 or upper < 0.0:
        return min(raw_p, nextafter(alpha, 0.0))
    return max(raw_p, nextafter(alpha, 1.0))


def hac_p_value(interval: HACInterval, point_estimate: float) -> tuple[float | None, str | None]:
    """Return the two-sided zero-mean test associated with an estimable HAC interval."""
    if (interval.reason is not None or not isfinite(point_estimate)
            or not isfinite(interval.variance) or interval.variance <= 0):
        return None, None
    if interval.bandwidth_method == "nw94_prewhitened":
        if interval.effective_df is None or not isfinite(interval.effective_df) or interval.effective_df < 1:
            return None, None
        return student_t_hac_p_value(
            point_estimate, interval.variance, interval.effective_df, interval.alpha, interval.lower, interval.upper,
        ), "two_sided_student_t_hac"
    return erfc(abs(point_estimate) / sqrt(interval.variance) / sqrt(2)), "two_sided_normal_hac"


def _time_mean(values: FloatArray, empty_reason: str) -> TimeMean:
    observed = values[np.isfinite(values)]
    n = observed.size
    if not n:
        return TimeMean(0, np.nan, np.nan, empty_reason)
    mean = float(observed.mean())
    if n == 1:
        return TimeMean(n, mean, np.nan, "insufficient_time_points")
    std = 0.0 if np.all(observed == observed[0]) else float(observed.std(ddof=1))
    return TimeMean(n, mean, std / sqrt(n), None)


def _divide(sums: npt.ArrayLike, counts: npt.NDArray[np.intp]) -> FloatArray:
    return np.divide(sums, counts, out=np.full(counts.shape, np.nan), where=counts > 0)


def _directional_masks(
    factor: npt.NDArray[np.integer | np.floating], direction: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    # Negating a signed integer's minimum value can overflow.
    positive, negative = factor > 0, factor < 0
    return (positive, negative) if direction == 1 else (negative, positive)


def _directional_accuracy(
    hits: npt.ArrayLike, counts: npt.NDArray[np.intp], ups: npt.ArrayLike,
) -> DirectionalAccuracy:
    denominator = int(counts.sum())
    if not denominator:
        return DirectionalAccuracy(0, 0, 0, np.nan, np.nan, np.nan, "no_directional_samples")

    active = counts > 0
    accuracy = float((np.asarray(hits)[active] / counts[active]).mean())
    baseline = float((np.asarray(ups)[active] / counts[active]).mean())
    return DirectionalAccuracy(
        int(np.sum(hits)), denominator, int(np.sum(ups)), accuracy, baseline, accuracy - baseline, None,
    )


def _signal_diagnostics(
    factor: FloatArray, eligible: npt.NDArray[np.bool_], labels: npt.NDArray[np.intp],
    n_quantiles: int, period: int,
) -> tuple[FloatArray, FloatArray]:
    autocorrelation = np.full(len(factor), np.nan)
    turnover = np.full((len(factor), n_quantiles), np.nan)
    if period < len(factor):
        ranks, _ = kernels.average_rank(factor, factor, valid=eligible)
        autocorrelation[period:] = kernels.pearson(ranks[period:], ranks[:-period]).values
        for column in range(n_quantiles):
            members = labels == column + 1
            current, previous = members[period:], members[:-period]
            counts = current.sum(axis=1)
            turnover[period:, column] = _divide(
                (current & ~previous).sum(axis=1), np.where(previous.any(axis=1), counts, 0),
            )
    return autocorrelation, turnover


def evaluate_cross_section(
    panel: EvaluationPanel,
    *,
    hac_lag: _HACLag | Mapping[Hashable, _HACLag] | None = None,
    hac_min_lag: Mapping[Hashable, int] | None = None,
    alpha: float = 0.05,
    frequency: Literal["daily", "bar"] = "daily",
    min_samples: int | None = None,
    ic_method: Literal["spearman", "pearson"] = "spearman",
    n_quantiles: int = 5,
    return_mode: ReturnMode = "raw",
    low_quantile: int = 1,
    high_quantile: int | None = None,
    direction: int = 1,
    diagnostic_period: int = 1,
    signal_valid: npt.NDArray[np.bool_] | None = None,
    bucket_labels: npt.ArrayLike | None = None,
    bucket_standard_errors: bool = False,
    calibration_scenario: str | None = None,
    _legacy_ic_rounding: bool = False,
    signal_diagnostics: bool = True,
    return_diagnostics: bool = True,
    _point_ic: kernels.CorrelationResult | None = None,
    _point_hac: HACInterval | None = None,
) -> CrossSectionResult:
    """Evaluate all horizons using one signal grid and one set of factor buckets."""
    if not panel.datetimes.is_monotonic_increasing:
        raise ValueError("panel.datetimes must be in chronological order")
    if not isinstance(signal_diagnostics, bool):
        raise ValueError("signal_diagnostics must be a boolean")
    if not isinstance(return_diagnostics, bool):
        raise ValueError("return_diagnostics must be a boolean")
    if frequency not in ("daily", "bar"):
        raise ValueError("frequency must be 'daily' or 'bar'")
    if hac_lag is None:
        hac_lag = 0 if _legacy_ic_rounding else "auto"
    elif _legacy_ic_rounding and (
        hac_lag == "auto" or (isinstance(hac_lag, Mapping) and "auto" in hac_lag.values())
    ):
        raise ValueError('hac_lag="auto" is incompatible with _legacy_ic_rounding')
    if ic_method not in ("spearman", "pearson"):
        raise ValueError("ic_method must be 'spearman' or 'pearson'")
    if return_mode not in ("raw", "cross_section_excess"):
        raise ValueError("return_mode must be 'raw' or 'cross_section_excess'")
    if isinstance(direction, (bool, np.bool_)) or direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if min_samples is None:
        min_samples = 2 if frequency == "daily" else 3
    min_samples = kernels._positive_int(min_samples, "min_samples", minimum=2)
    n_quantiles = kernels._positive_int(n_quantiles, "n_quantiles", minimum=2)
    diagnostic_period = kernels._positive_int(diagnostic_period, "diagnostic_period")
    low_quantile = kernels._positive_int(low_quantile, "low_quantile")
    high_quantile = (n_quantiles if high_quantile is None
                     else kernels._positive_int(high_quantile, "high_quantile"))
    if not low_quantile < high_quantile <= n_quantiles:
        raise ValueError("Expected 1 <= low_quantile < high_quantile <= n_quantiles")

    factor = panel.factor
    eligible = np.isfinite(factor)
    if signal_valid is not None:
        signal_valid = np.asarray(signal_valid)
        if signal_valid.shape != factor.shape or signal_valid.dtype.kind != "b":
            raise ValueError("signal_valid must be a boolean mask with the panel shape")
        eligible &= signal_valid

    correlate = kernels.spearman if ic_method == "spearman" else kernels.pearson
    if ic_method == "spearman" and _legacy_ic_rounding:
        correlate = partial(kernels.spearman, _legacy_covariance=True)
    single_horizon_ic = _point_ic
    if bucket_labels is None:
        if (single_horizon_ic is None and frequency == "bar" and len(panel.returns) == 1 and len(factor) >= 2
                and factor.size >= _PARALLEL_STATISTIC_CELLS):
            horizon, returns = next(iter(panel.returns.items()))
            paired = panel.valid[horizon] & eligible & np.isfinite(returns)
            midpoint = len(factor) // 2
            sections = (slice(None, midpoint), slice(midpoint, None))
            with ThreadPoolExecutor(max_workers=2) as executor:
                ic_futures = [
                    executor.submit(correlate, factor[section], returns[section],
                                    valid=paired[section], min_samples=min_samples)
                    for section in sections
                ]
                quantiles = kernels.quantile_buckets(factor, n_quantiles, valid=eligible)
                single_horizon_ic = kernels.CorrelationResult(*(
                    np.concatenate(parts) for parts in zip(*(future.result() for future in ic_futures), strict=True)
                ))
        else:
            quantiles = kernels.quantile_buckets(factor, n_quantiles, valid=eligible)
        unformed = quantiles.n_bins < 2
        quantiles.labels[unformed] = 0
    else:
        labels = np.asarray(bucket_labels)
        if (labels.shape != factor.shape or labels.dtype.kind not in "iu"
                or np.any((labels < 0) | (labels > n_quantiles))):
            raise ValueError("bucket_labels must be integer labels in [0, n_quantiles] with the panel shape")
        labels = np.where(eligible, labels, 0).astype(np.intp)
        quantiles = kernels.QuantileResult(
            labels, np.full((len(factor), n_quantiles + 1), np.nan), labels.max(axis=1, initial=0),
        )
        unformed = ~np.any(labels > 0, axis=1)
    quantile_reasons = np.where(unformed, "unable_to_form_quantiles", None)
    labels = quantiles.labels
    if signal_diagnostics:
        autocorrelation, turnover = _signal_diagnostics(factor, eligible, labels, n_quantiles, diagnostic_period)
    else:
        autocorrelation = np.full(len(factor), np.nan)
        turnover = np.full((len(factor), n_quantiles), np.nan)
    if return_diagnostics:
        long, short = _directional_masks(factor, direction)
    horizons = {}
    for horizon, returns in panel.returns.items():
        valid = panel.valid[horizon] & eligible & np.isfinite(returns)
        ic = (single_horizon_ic if single_horizon_ic is not None else
              correlate(factor, returns, valid=valid, min_samples=min_samples))
        ic_summary = _ic_summary(ic.values[np.isfinite(ic.values)])
        lag = hac_lag[horizon] if isinstance(hac_lag, Mapping) else hac_lag
        min_lag = 0
        adaptive = isinstance(lag, str) and lag == "auto"
        if adaptive:
            if hac_min_lag is not None and horizon in hac_min_lag:
                min_lag = hac_min_lag[horizon]
            else:
                steps = _daily_horizon_steps(horizon) if frequency == "daily" else horizon
                min_lag = kernels._positive_int(steps, f"horizon {horizon!r}") - 1
        hac = _point_hac if _point_hac is not None else bartlett_hac(
            ic.values, lag=lag, min_lag=min_lag, alpha=alpha,
            calibration_scenario=(
                calibration_scenario if adaptive or (ic_method == "spearman" and not _legacy_ic_rounding)
                else None
            ),
        )

        buckets = kernels.bucket_sums(returns, labels, n_quantiles, valid=valid)
        bucket_returns = _divide(buckets.sums, buckets.counts)
        if return_mode == "cross_section_excess":
            benchmark = _divide(np.sum(returns, axis=1, where=valid), valid.sum(axis=1))
            bucket_returns -= benchmark[:, None]
        standard_errors = None
        if bucket_standard_errors:
            row, col = np.nonzero(valid & (labels > 0))
            bucket = labels[row, col] - 1
            residual = returns[row, col]
            if return_mode == "cross_section_excess":
                residual -= benchmark[row]
            residual -= bucket_returns[row, bucket]
            squared = np.bincount(
                row * n_quantiles + bucket, weights=residual**2, minlength=buckets.counts.size,
            ).reshape(buckets.counts.shape)
            standard_errors = np.sqrt(_divide(squared, buckets.counts * (buckets.counts - 1)))
        if not return_diagnostics:
            bucket_summary, by_bucket = {}, {}
            spread = np.full(len(factor), np.nan)
            spread_summary = TimeMean(0, np.nan, np.nan, "not_requested")
            overall = DirectionalAccuracy(0, 0, 0, np.nan, np.nan, np.nan, "not_requested")
        else:
            bucket_summary = {
                q: _time_mean(bucket_returns[:, q - 1], "empty_bucket")
                for q in range(1, n_quantiles + 1)
            }
            spread = bucket_returns[:, high_quantile - 1] - bucket_returns[:, low_quantile - 1]
            spread_summary = _time_mean(spread, "no_paired_bucket_returns")

            directional = valid & (long | short)
            up = returns > 0
            hits = (long & up) | (short & (returns < 0))
            overall = _directional_accuracy(
                (hits & directional).sum(axis=1), directional.sum(axis=1), (up & directional).sum(axis=1),
            )
            bucket_hits = kernels.bucket_sums(hits, labels, n_quantiles, valid=directional)
            bucket_ups = kernels.bucket_sums(up, labels, n_quantiles, valid=directional)
            by_bucket = {
                q: _directional_accuracy(bucket_hits.sums[:, q - 1], bucket_hits.counts[:, q - 1],
                                         bucket_ups.sums[:, q - 1])
                for q in range(1, n_quantiles + 1)
            }
        horizons[horizon] = CrossSectionHorizon(
            ic=ic,
            ic_summary=ic_summary,
            hac=hac,
            bucket_returns=bucket_returns,
            bucket_counts=buckets.counts,
            bucket_standard_errors=standard_errors,
            bucket_summary=bucket_summary,
            spread=spread,
            spread_summary=spread_summary,
            direction=overall,
            direction_by_bucket=by_bucket,
        )

    return CrossSectionResult(
        datetimes=panel.datetimes,
        assets=panel.assets,
        min_samples=min_samples,
        ic_method=ic_method,
        return_mode=return_mode,
        direction=direction,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        diagnostic_period=diagnostic_period,
        quantiles=quantiles,
        quantile_reasons=quantile_reasons,
        rank_autocorrelation=autocorrelation,
        turnover=turnover,
        horizons=horizons,
    )
