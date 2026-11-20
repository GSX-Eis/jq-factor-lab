"""Single entry point for daily and Bar panel statistics."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import kernels
from .cross_section import _daily_horizon_steps, evaluate_cross_section
from .kernels import CorrelationResult, _positive_int
from .panel import EvaluationPanel
from .quantiles import QuantileFit, QuantileScheme, Standardization
from .split import SplitConfig, SplitEvaluation, evaluate_splits
from .time_series import evaluate_time_series
from .types import (
    DEFAULT_BOOTSTRAP_REPLICATES, DEFAULT_MIN_VALID_REPLICATES, CrossSectionResult, HACInterval,
    ReturnMode, TimeSeriesInstrument, TimeSeriesResult,
)

AnalysisMode = Literal["cross_sectional", "time_series"]
Frequency = Literal["daily", "bar"]
_Value = TypeVar("_Value")


def _normalize_daily_horizons(values: Mapping[Hashable, _Value]) -> dict[int, _Value]:
    by_step: dict[int, _Value] = {}
    for horizon, value in values.items():
        steps = _positive_int(_daily_horizon_steps(horizon), f"horizon {horizon!r}")
        if steps in by_step:
            raise ValueError(f"Duplicate daily horizon after normalization: {horizon!r} ({steps} steps)")
        by_step[steps] = value
    return by_step


@dataclass(frozen=True)
class DescriptiveSample:
    """The same Bar labels on their observed start grid, with valid-label buckets."""

    panel: EvaluationPanel
    observed: npt.NDArray[np.bool_]
    inference_times: pd.DatetimeIndex | None = None


def _eligible_signals(
    panel: EvaluationPanel, signal_valid: npt.ArrayLike | None, expected_grid: npt.ArrayLike | None,
) -> npt.NDArray[np.bool_]:
    eligible = np.isfinite(panel.factor)
    for name, mask in (("signal_valid", signal_valid), ("expected_grid", expected_grid)):
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype.kind != "b" or mask.shape != eligible.shape:
                raise ValueError(f"{name} must be a boolean mask with the panel shape")
            eligible &= mask
    return eligible


def _same_values(left: np.ndarray, right: np.ndarray) -> npt.NDArray[np.bool_]:
    if left.dtype.kind != right.dtype.kind:
        # Comparing a wide integer with a float must not round the integer.
        return left.astype(object) == right.astype(object)
    return left == right


@dataclass(frozen=True)
class _TimeSeriesReference:
    statistics: TimeSeriesResult
    eligible: npt.NDArray[np.bool_]
    grid: npt.ArrayLike | None


def _matching_time_series_points(
    sample: DescriptiveSample, valid: npt.NDArray[np.bool_], horizon: int,
    source: EvaluationPanel, reference: _TimeSeriesReference | None,
) -> tuple[dict[Hashable, TimeSeriesInstrument], set[Hashable]]:
    points, phase_zero = {}, set()
    if (reference is None or reference.statistics.ic_method != "spearman"
            or horizon not in reference.statistics.horizons):
        return points, phase_zero

    panel = sample.panel
    source_grid = None if reference.grid is None else np.asarray(reference.grid)
    for col, asset in enumerate(panel.assets):
        if asset not in source.assets:
            continue
        src = source.assets.get_loc(asset)
        source_return = source.returns[horizon][:, src]
        source_pairs = source.valid[horizon][:, src] & reference.eligible[:, src] & np.isfinite(source_return)
        observed_pairs = valid[:, col]
        if (not source.datetimes[source_pairs].equals(panel.datetimes[observed_pairs])
                or not _same_values(source.factor[source_pairs, src], panel.factor[observed_pairs, col]).all()
                or not _same_values(source_return[source_pairs], panel.returns[horizon][observed_pairs, col]).all()):
            continue

        points[asset] = reference.statistics.horizons[horizon].instruments[asset]
        source_rows = slice(None) if source_grid is None else source_grid[:, src]
        observed_rows = sample.observed[:, col]
        source_phase = source.datetimes[source_rows][::horizon][source_pairs[source_rows][::horizon]]
        observed_phase = panel.datetimes[observed_rows][::horizon][observed_pairs[observed_rows][::horizon]]
        if source_phase.equals(observed_phase):
            phase_zero.add(asset)
    return points, phase_zero


def _observed_section_statistics(
    sample: DescriptiveSample, valid: npt.NDArray[np.bool_], horizon: int, source: EvaluationPanel,
    splits: Mapping[str, SplitEvaluation], min_samples: int, calibration_scenario: str | None,
) -> tuple[CorrelationResult | None, HACInterval | None]:
    if sample.inference_times is None:
        return None, None
    panel = sample.panel
    source_positions = source.datetimes.get_indexer(sample.inference_times)
    pending = np.ones(len(panel.datetimes), dtype=bool)
    values = np.empty(len(pending), dtype=float)
    counts = np.empty(len(pending), dtype=np.intp)
    reasons = np.empty(len(pending), dtype=object)
    for evaluation in splits.values():
        reference = evaluation.statistics
        if (not isinstance(reference, CrossSectionResult) or reference.ic_method != "spearman"
                or reference.min_samples != min_samples or horizon not in reference.horizons
                or not panel.assets.equals(reference.assets)):
            continue
        positions = reference.datetimes.get_indexer(sample.inference_times)
        rows = np.flatnonzero(pending & (positions >= 0) & (source_positions >= 0))
        # Purged sections can have fewer pairs than the full-sample table.
        rows = rows[np.all(valid[rows] == evaluation.valid[horizon][positions[rows]], axis=1)]
        src = source_positions[rows]
        same_values = (_same_values(panel.factor[rows], source.factor[src])
                       & _same_values(panel.returns[horizon][rows], source.returns[horizon][src]))
        rows = rows[np.all(~valid[rows] | same_values, axis=1)]
        result = reference.horizons[horizon]
        ic, hac = result.ic, result.hac
        if (rows.size == len(pending) and reference.datetimes.equals(sample.inference_times)
                and hac.bandwidth_method == "fixed" and hac.method_version is None
                and hac.lag == horizon - 1 and hac.alpha == 0.05 and calibration_scenario is None):
            return ic, hac
        matched = positions[rows]
        values[rows], counts[rows], reasons[rows] = ic.values[matched], ic.counts[matched], ic.reasons[matched]
        pending[rows] = False
    if pending.all():
        return None, None
    if pending.any():
        values[pending], counts[pending], reasons[pending] = kernels.spearman(
            panel.factor[pending], panel.returns[horizon][pending], valid=valid[pending], min_samples=min_samples,
        )
    return CorrelationResult(values, counts, reasons), None


@dataclass(frozen=True)
class EngineResult:
    analysis_mode: AnalysisMode
    frequency: Frequency
    return_mode: ReturnMode
    splits: dict[str, SplitEvaluation]
    descriptive: dict[int, TimeSeriesResult | CrossSectionResult] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationEngine:
    analysis_mode: AnalysisMode
    frequency: Frequency = "daily"

    def __post_init__(self) -> None:
        if self.analysis_mode not in ("cross_sectional", "time_series"):
            raise ValueError("analysis_mode must be 'cross_sectional' or 'time_series'")
        if self.frequency not in ("daily", "bar"):
            raise ValueError("frequency must be 'daily' or 'bar'")

    def evaluate(
        self,
        panel: EvaluationPanel,
        *,
        split: SplitConfig | Mapping[str, str | pd.Timestamp] | None = None,
        quantile_scheme: QuantileScheme | None = None,
        quantile_fit: QuantileFit | None = None,
        return_mode: ReturnMode = "raw",
        signal_available_at: pd.DatetimeIndex | None = None,
        entry_timestamps: Mapping[Hashable, npt.ArrayLike] | None = None,
        exit_timestamps: Mapping[Hashable, npt.ArrayLike] | None = None,
        signal_valid: npt.ArrayLike | None = None,
        expected_grid: npt.ArrayLike | None = None,
        n_quantiles: int = 5,
        direction: int = 1,
        standardization: Standardization | None = None,
        calibration_scenario: str | None = None,
        all_non_overlapping_phases: bool = True,
        signal_diagnostics: bool = True,
        return_diagnostics: bool = True,
        descriptive_samples: Mapping[int, DescriptiveSample] | None = None,
        **statistics_options: Any,
    ) -> EngineResult:
        """Evaluate one labeled panel on its original return scale.

        Pass the signal clock and scheduled exits for a split. Time-series
        inference also needs entries. Daily time-series inputs accept labels
        like "5D" or integer grid steps and return integer horizon keys.
        Forward mode-specific options to the S3 statistics layer, such as
        `hac_lag` or the bootstrap settings. Declare calibration_scenario
        only when it applies to the intervals being evaluated.
        Bar callers can request descriptive tables on the observed, valid-label
        sample alongside inference on the scheduled signal grid.
        """
        cross_sectional = self.analysis_mode == "cross_sectional"
        for name, value in (
            ("all_non_overlapping_phases", all_non_overlapping_phases),
            ("signal_diagnostics", signal_diagnostics),
            ("return_diagnostics", return_diagnostics),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if cross_sectional:
            if not all_non_overlapping_phases:
                raise ValueError("all_non_overlapping_phases requires time_series analysis")
            statistics_options["signal_diagnostics"] = signal_diagnostics
        else:
            if not signal_diagnostics:
                raise ValueError("signal_diagnostics requires cross_sectional analysis")
            statistics_options["all_non_overlapping_phases"] = all_non_overlapping_phases
        statistics_options["return_diagnostics"] = return_diagnostics
        if descriptive_samples is not None and self.frequency != "bar":
            raise ValueError("descriptive_samples requires Bar evaluation")
        if return_mode not in ("raw", "cross_section_excess"):
            raise ValueError("return_mode must be 'raw' or 'cross_section_excess'")
        if not cross_sectional and return_mode != "raw":
            raise ValueError("cross_section_excess requires cross_sectional analysis")
        if quantile_fit is not None and (cross_sectional or split is not None):
            raise ValueError("quantile_fit requires full-sample time_series analysis")
        scheme: QuantileScheme | None = (
            "per_cross_section" if cross_sectional else
            "fitted_on_train" if split is not None or quantile_fit is not None else None
        )
        if quantile_scheme is not None and quantile_scheme != scheme:
            raise ValueError("Use per_cross_section for cross_sectional analysis or "
                             "fitted_on_train for time_series with a split or frozen quantile_fit")
        if standardization is not None and (cross_sectional or split is None):
            raise ValueError("standardization requires a time_series train/oos split")

        if cross_sectional:
            legacy = bool(statistics_options.get("_legacy_ic_rounding"))
            requested_lag = statistics_options.pop("hac_lag", None)
            if requested_lag is None:
                requested_lag = 0 if legacy else "auto"
            adaptive = isinstance(requested_lag, str) and requested_lag == "auto"
            if adaptive and legacy:
                raise ValueError('hac_lag="auto" is incompatible with _legacy_ic_rounding')
            minimum_lag = 0 if adaptive else _positive_int(requested_lag, "hac_lag", minimum=0)
            lags: dict[Hashable, int] = {}
            for horizon in panel.returns:
                steps = _daily_horizon_steps(horizon) if self.frequency == "daily" else horizon
                if legacy and not isinstance(steps, (int, np.integer)):
                    # Legacy daily dictionaries can include extra label keys and expose no HAC results.
                    lags[horizon] = minimum_lag
                else:
                    steps = _positive_int(steps, f"horizon {horizon!r}")
                    lags[horizon] = max(minimum_lag, steps - 1)
            statistics_options.update(
                frequency=self.frequency, return_mode=return_mode, hac_lag="auto" if adaptive else lags,
            )
            if adaptive:
                statistics_options["hac_min_lag"] = lags
        else:
            if self.frequency == "daily":
                returns = _normalize_daily_horizons(panel.returns)
                valid = _normalize_daily_horizons(panel.valid)
                if valid.keys() != returns.keys():
                    raise ValueError("Valid masks must contain the same horizons as the panel returns")
                panel = replace(panel, returns=returns, valid=valid, _columns=pd.Index(["factor", *returns]))
                if entry_timestamps is not None:
                    entry_timestamps = _normalize_daily_horizons(entry_timestamps)
                if exit_timestamps is not None:
                    exit_timestamps = _normalize_daily_horizons(exit_timestamps)
            statistics_options.setdefault("block_length", "auto")
            statistics_options.setdefault("seed", 0)
            statistics_options.setdefault("n_boot", DEFAULT_BOOTSTRAP_REPLICATES)
            statistics_options.setdefault("min_valid_replicates", DEFAULT_MIN_VALID_REPLICATES)

        if calibration_scenario is not None:
            statistics_options["calibration_scenario"] = calibration_scenario
        if split is not None:
            if signal_available_at is None or exit_timestamps is None:
                raise ValueError("A split requires signal_available_at and exit_timestamps")
            splits = evaluate_splits(
                panel, split, mode="cross_section" if cross_sectional else "time_series",
                signal_available_at=signal_available_at, exit_timestamps=exit_timestamps,
                entry_timestamps=entry_timestamps, signal_valid=signal_valid, expected_grid=expected_grid,
                n_quantiles=n_quantiles, direction=direction, standardization=standardization,
                **statistics_options,
            )
            descriptive = self._describe(
                descriptive_samples, n_quantiles, statistics_options.get("min_samples", 3), panel,
                {} if statistics_options.get("_legacy_ic_rounding") else splits, calibration_scenario, None,
            )
            return EngineResult(self.analysis_mode, self.frequency, return_mode, splits, descriptive)

        eligible = _eligible_signals(panel, signal_valid, expected_grid)
        reference = None

        if cross_sectional:
            statistics = evaluate_cross_section(
                panel, signal_valid=eligible, n_quantiles=n_quantiles, direction=direction,
                **statistics_options,
            )
        else:
            if (entry_timestamps is None) != (exit_timestamps is None):
                raise ValueError("Provide both entry_timestamps and exit_timestamps for time_series")
            intervals = None
            if entry_timestamps is not None and exit_timestamps is not None:
                if entry_timestamps.keys() != panel.returns.keys() or exit_timestamps.keys() != panel.returns.keys():
                    raise ValueError("Entry and exit timestamps must contain the same horizons as the panel")
                intervals = {h: (entry_timestamps[h], exit_timestamps[h]) for h in panel.returns}
            statistics = evaluate_time_series(
                panel, intervals, signal_valid=eligible, expected_grid=expected_grid,
                n_quantiles=n_quantiles, direction=direction, quantile_fit=quantile_fit, **statistics_options,
            )
            reference = _TimeSeriesReference(statistics, eligible, expected_grid)
        valid = {h: panel.valid[h] & eligible & np.isfinite(values) for h, values in panel.returns.items()}
        splits = {"full": SplitEvaluation(
            statistics, valid, dict.fromkeys(panel.returns, 0), scheme,
            None if quantile_fit is None else quantile_fit.identity(),
        )}
        descriptive = self._describe(
            descriptive_samples, n_quantiles, statistics_options.get("min_samples", 3), panel,
            {} if statistics_options.get("_legacy_ic_rounding") else splits, calibration_scenario, reference,
        )
        return EngineResult(self.analysis_mode, self.frequency, return_mode, splits, descriptive)

    def _describe(
        self, samples: Mapping[int, DescriptiveSample] | None, n_quantiles: int, min_samples: int,
        source: EvaluationPanel, splits: Mapping[str, SplitEvaluation], calibration_scenario: str | None,
        reference: _TimeSeriesReference | None,
    ) -> dict[int, TimeSeriesResult | CrossSectionResult]:
        results = {}
        for horizon, sample in (samples or {}).items():
            panel = sample.panel
            observed = sample.observed if self.analysis_mode == "time_series" else None
            valid = (panel.valid[horizon] & _eligible_signals(panel, None, observed)
                     & np.isfinite(panel.returns[horizon]))
            if self.analysis_mode == "time_series":
                points, phase_zero = _matching_time_series_points(sample, valid, horizon, source, reference)
                results[horizon] = evaluate_time_series(
                    panel, None, block_length=horizon,
                    signal_valid=panel.valid[horizon], expected_grid=sample.observed, min_samples=min_samples,
                    n_quantiles=n_quantiles, all_non_overlapping_phases=False, return_diagnostics=False,
                    _point_estimates=points, _reuse_phase_zero=phase_zero,
                )
            else:
                point, hac = _observed_section_statistics(
                    sample, valid, horizon, source, splits, min_samples, calibration_scenario,
                )
                results[horizon] = evaluate_cross_section(
                    panel, frequency="bar", hac_lag=horizon - 1, alpha=0.05,
                    signal_valid=panel.valid[horizon], min_samples=min_samples,
                    n_quantiles=n_quantiles, signal_diagnostics=False, return_diagnostics=False,
                    _point_ic=point, _point_hac=hac,
                )
        return results
