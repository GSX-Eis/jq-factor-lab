"""Single entry point for daily and Bar panel statistics."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
import re
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from .cross_section import evaluate_cross_section
from .kernels import _positive_int
from .panel import EvaluationPanel
from .quantiles import QuantileFit, QuantileScheme, Standardization
from .split import SplitConfig, SplitEvaluation, evaluate_splits
from .time_series import evaluate_time_series
from .types import ReturnMode

AnalysisMode = Literal["cross_sectional", "time_series"]
Frequency = Literal["daily", "bar"]
_Value = TypeVar("_Value")


def _daily_horizon_steps(horizon: Hashable) -> Hashable:
    if isinstance(horizon, str) and re.fullmatch(r"[0-9]+[dD]", horizon):
        return int(horizon[:-1])
    return horizon


def _normalize_daily_horizons(values: Mapping[Hashable, _Value]) -> dict[int, _Value]:
    by_step: dict[int, _Value] = {}
    for horizon, value in values.items():
        steps = _positive_int(_daily_horizon_steps(horizon), f"horizon {horizon!r}")
        if steps in by_step:
            raise ValueError(f"Duplicate daily horizon after normalization: {horizon!r} ({steps} steps)")
        by_step[steps] = value
    return by_step


@dataclass(frozen=True)
class EngineResult:
    analysis_mode: AnalysisMode
    frequency: Frequency
    return_mode: ReturnMode
    splits: dict[str, SplitEvaluation]


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
        **statistics_options: Any,
    ) -> EngineResult:
        """Evaluate one labeled panel on its original return scale.

        Pass the signal clock and scheduled exits for a split. Time-series
        inference also needs entries. Daily time-series inputs accept labels
        like "5D" or integer grid steps and return integer horizon keys.
        Forward mode-specific options to the S3 statistics layer, such as
        `hac_lag` or the bootstrap settings. Declare calibration_scenario
        only when it applies to the intervals being evaluated.
        """
        cross_sectional = self.analysis_mode == "cross_sectional"
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
            minimum_lag = _positive_int(statistics_options.pop("hac_lag", 0), "hac_lag", minimum=0)
            lags = {}
            for horizon in panel.returns:
                steps = _daily_horizon_steps(horizon) if self.frequency == "daily" else horizon
                if statistics_options.get("_legacy_ic_rounding") and not isinstance(steps, (int, np.integer)):
                    # Legacy daily dictionaries can include extra label keys and expose no HAC results.
                    lags[horizon] = minimum_lag
                else:
                    steps = _positive_int(steps, f"horizon {horizon!r}")
                    lags[horizon] = max(minimum_lag, steps - 1)
            statistics_options.update(frequency=self.frequency, return_mode=return_mode, hac_lag=lags)
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
            statistics_options.setdefault("block_length", max(panel.returns))
            statistics_options.setdefault("seed", 0)
            statistics_options.setdefault("n_boot", 999)
            statistics_options.setdefault("min_valid_replicates", 900)

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
            return EngineResult(self.analysis_mode, self.frequency, return_mode, splits)

        eligible = np.isfinite(panel.factor)
        for name, mask in (("signal_valid", signal_valid), ("expected_grid", expected_grid)):
            if mask is not None:
                mask = np.asarray(mask)
                if mask.dtype.kind != "b" or mask.shape != eligible.shape:
                    raise ValueError(f"{name} must be a boolean mask with the panel shape")
                eligible &= mask

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
        valid = {h: panel.valid[h] & eligible & np.isfinite(values) for h, values in panel.returns.items()}
        splits = {"full": SplitEvaluation(
            statistics, valid, dict.fromkeys(panel.returns, 0), scheme,
            None if quantile_fit is None else quantile_fit.identity(),
        )}
        return EngineResult(self.analysis_mode, self.frequency, return_mode, splits)
