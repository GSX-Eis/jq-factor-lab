"""Instrument-level statistics on an expected signal grid for one split."""

from collections.abc import Hashable, Mapping
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from . import kernels
from .calibration import resolve_interval_calibration
from .cross_section import _directional_accuracy, _directional_masks, _time_mean
from .kernels import CorrelationResult, QuantileResult
from .panel import EvaluationPanel
from .quantiles import QuantileFit
from .types import (
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


def _bootstrap_spearman(
    factor: FloatArray, returns: FloatArray, valid: npt.NDArray[np.bool_],
    *, point: CorrelationResult, min_samples: int, block_length: int,
    seed: int, n_boot: int, min_valid_replicates: int, alpha: float,
) -> BootstrapInterval:
    size = len(factor)
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


def evaluate_time_series(
    panel: EvaluationPanel,
    holding_intervals: Mapping[int, tuple[npt.ArrayLike, npt.ArrayLike]] | None,
    *,
    block_length: int,
    seed: int,
    n_boot: int,
    min_valid_replicates: int,
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
    _bar_v1: bool = False,
) -> TimeSeriesResult:
    """Evaluate each instrument in a single frequency and split, retaining grid gaps.

    Without intervals the result is descriptive: phase ICs are positional,
    and no bootstrap interval is produced.
    """
    if not panel.datetimes.is_monotonic_increasing:
        raise ValueError("panel.datetimes must be in chronological order")
    if ic_method not in ("spearman", "pearson"):
        raise ValueError("ic_method must be 'spearman' or 'pearson'")
    if isinstance(direction, (bool, np.bool_)) or direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    min_samples = kernels._positive_int(min_samples, "min_samples", 2)
    n_quantiles = kernels._positive_int(n_quantiles, "n_quantiles", 2)
    block_length = kernels._positive_int(block_length, "block_length")
    seed = kernels._positive_int(seed, "seed", 0)
    n_boot = kernels._positive_int(n_boot, "n_boot")
    min_valid_replicates = kernels._positive_int(min_valid_replicates, "min_valid_replicates", 2)
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
        if block_length < horizon:
            raise ValueError(f"block_length must be >= horizon ({horizon})")
        if holding_intervals is not None:
            entry, exit = (np.asarray(timestamps) for timestamps in holding_intervals[horizon])
            if any(a.shape != factor.shape or a.dtype.kind != "M" for a in (entry, exit)):
                raise ValueError("entry_timestamp and exit_timestamp must be datetime64 arrays with the panel shape")

        valid = panel.valid[horizon] & eligible & np.isfinite(returns)
        # Bar v1 publishes phase zero. Verified intervals still require every phase.
        offsets = range(1 if _bar_v1 and holding_intervals is None else horizon)
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

            spearman = kernels.spearman(
                x[:, None], y[:, None], valid=pairs[:, None], axis=0, min_samples=min_samples,
            )
            ic = spearman if ic_method == "spearman" else correlate(
                x[:, None], y[:, None], valid=pairs[:, None], axis=0, min_samples=min_samples,
            )
            sampled = {}
            for offset in offsets:
                phase = slice(offset, None, horizon)
                phase_pairs = pairs[phase]
                if holding_intervals is not None:
                    phase_entries, phase_exits = entries[phase][phase_pairs], exits[phase][phase_pairs]
                    order = np.argsort(phase_entries)
                    if np.any(phase_entries[order][1:] < phase_exits[order][:-1]):
                        raise AssertionError(
                            f"Overlapping holding intervals for {asset!r}, horizon {horizon}, offset {offset}",
                        )
                sampled[offset] = ic if horizon == 1 else correlate(
                    x[phase, None], y[phase, None], valid=phase_pairs[:, None],
                    axis=0, min_samples=min_samples,
                )

            bootstrap = BootstrapInterval(
                len(x), block_length, seed, 0, 0, min_valid_replicates, alpha,
                np.nan, np.nan, "holding_intervals_unavailable",
            ) if holding_intervals is None else _bootstrap_spearman(
                x, y, pairs, point=spearman, min_samples=min_samples, block_length=block_length,
                seed=seed, n_boot=n_boot, min_valid_replicates=min_valid_replicates, alpha=alpha,
            )
            if calibration_scenario is not None:
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
                    distributions[bucket] = _distribution(y[members], "empty_bucket", location_only=_bar_v1)
                    if not _bar_v1:
                        directions[bucket] = _direction(signal[members], y[members], direction)
            instruments[asset] = TimeSeriesInstrument(
                ic=ic,
                spearman_ic=spearman,
                bootstrap=bootstrap,
                returns=(ReturnDistribution(n_samples=int(ic.counts[0]), reason="not_requested")
                         if _bar_v1 else _distribution(y[pairs], "insufficient_samples")),
                returns_by_bucket=distributions,
                direction=(DirectionalAccuracy(0, 0, 0, np.nan, np.nan, np.nan, "not_requested")
                           if _bar_v1 else _direction(signal[pairs], y[pairs], direction)),
                direction_by_bucket=directions,
                non_overlapping_ic=sampled,
            )
        horizons[horizon] = summarize_time_series_horizon(instruments, offsets)
    return TimeSeriesResult(
        panel.datetimes, panel.assets, min_samples, ic_method, direction, quantiles, quantile_reasons, horizons,
        holding_intervals is not None,
    )
