"""Point-in-time split evaluation and higher-timeframe feature alignment."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .cross_section import evaluate_cross_section
from .panel import EvaluationPanel
from .quantiles import QuantileScheme, Standardization, fit_quantiles
from .time_series import evaluate_time_series
from .types import CrossSectionResult, TimeSeriesResult


@dataclass(frozen=True)
class SplitConfig:
    train_end: str | pd.Timestamp
    oos_start: str | pd.Timestamp


@dataclass(frozen=True)
class SplitEvaluation:
    statistics: CrossSectionResult | TimeSeriesResult
    valid: dict[Hashable, npt.NDArray[np.bool_]]
    split_purged_count: dict[Hashable, int]
    quantile_scheme: QuantileScheme | None
    fit_identity: dict | None


def _timestamps(values: npt.ArrayLike, name: str) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(values)
    if times.hasnans or times.tz is None:
        raise ValueError(f"{name} must contain non-missing timezone-aware timestamps")
    return times


def align_released_features(
    features: pd.DataFrame, signal_available_at: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align one feature stream by effective release, then by most recent Bar close."""
    ends = _timestamps(features["high_tf_bar_end"], "high_tf_bar_end").as_unit("ns").tz_convert("UTC")
    published = _timestamps(features["signal_available_at"], "feature signal_available_at")
    published = published.as_unit("ns").tz_convert("UTC")
    signals = _timestamps(signal_available_at, "signal_available_at")
    if not ends.is_unique:
        raise ValueError("Expected unique high_tf_bar_end values within each feature stream")

    release = ends.where(ends >= published, published)
    order = np.lexsort((ends.asi8, release.asi8))
    sorted_ends = ends.asi8[order]
    # A late release of an older Bar must not replace a newer Bar already known.
    latest = sorted_ends == np.maximum.accumulate(sorted_ends)
    order = order[latest]
    positions = release.take(order).searchsorted(signals.as_unit("ns"), side="right") - 1
    aligned = features.iloc[order].reset_index(drop=True).reindex(positions)
    return aligned.set_axis(signals.rename("signal_available_at"), axis=0)


def _validate_split_order(train_end: pd.Timestamp, oos_start: pd.Timestamp) -> None:
    if train_end >= oos_start:
        raise ValueError("Expected train_end < oos_start")


def evaluate_splits(
    panel: EvaluationPanel,
    split: SplitConfig | Mapping[str, str | pd.Timestamp],
    *,
    mode: Literal["cross_section", "time_series"],
    signal_available_at: pd.DatetimeIndex,
    exit_timestamps: Mapping[Hashable, npt.ArrayLike],
    entry_timestamps: Mapping[Hashable, npt.ArrayLike] | None = None,
    signal_valid: npt.ArrayLike | None = None,
    expected_grid: npt.ArrayLike | None = None,
    n_quantiles: int = 5,
    direction: int = 1,
    standardization: Standardization | None = None,
    **statistics_options: Any,
) -> dict[str, SplitEvaluation]:
    """Evaluate the two signal-time windows with horizon-specific exit purging.

    The timestamp arrays are aligned to the panel. Scheduled exits come from the
    label calendar independently of observed returns. All datetime64 trade
    arrays encode UTC when the signal clock is timezone-aware.
    """
    if mode not in ("cross_section", "time_series"):
        raise ValueError("mode must be 'cross_section' or 'time_series'")
    if mode == "cross_section" and standardization is not None:
        raise ValueError("standardization requires the fitted_on_train scheme")
    if exit_timestamps.keys() != panel.returns.keys():
        raise ValueError("exit_timestamps must contain the same horizons as the panel")
    if mode == "time_series" and (
        entry_timestamps is None or entry_timestamps.keys() != panel.returns.keys()
    ):
        raise ValueError("Time-series entry_timestamps must contain the same horizons as the panel")

    if isinstance(split, Mapping):
        split = SplitConfig(**split)
    times = _timestamps(signal_available_at, "signal_available_at")
    train_end, oos_start = pd.Timestamp(split.train_end), pd.Timestamp(split.oos_start)
    if pd.isna(train_end) or pd.isna(oos_start):
        raise ValueError("train_end and oos_start must be valid timestamps")
    if (not panel.datetimes.is_monotonic_increasing or not times.is_unique
            or not times.is_monotonic_increasing or len(times) != len(panel.datetimes)):
        raise ValueError("signal_available_at must be strictly increasing and aligned to panel rows")
    if train_end.tzinfo is not None and oos_start.tzinfo is not None:
        split_axis = times
    elif (train_end.tzinfo is None and oos_start.tzinfo is None
          and train_end == train_end.normalize() and oos_start == oos_start.normalize()
          and panel.datetimes.tz is None and panel.datetimes.equals(panel.datetimes.normalize())):
        split_axis = panel.datetimes
    else:
        raise ValueError("Use timezone-aware split bounds or daily trading date keys")
    _validate_split_order(train_end, oos_start)
    train_rows, oos_rows = split_axis <= train_end, split_axis >= oos_start
    eligible = np.isfinite(panel.factor)
    grid = None
    for name, mask in (("signal_valid", signal_valid), ("expected_grid", expected_grid)):
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != panel.factor.shape or mask.dtype.kind != "b":
                raise ValueError(f"{name} must be a boolean mask with the panel shape")
            eligible &= mask
            if name == "expected_grid":
                grid = mask

    scheduled_oos = oos_rows if grid is None else oos_rows & grid.any(axis=1)
    if not train_rows.any() or not scheduled_oos.any():
        raise ValueError("Both train and oos require at least one scheduled signal")
    first_oos = times[scheduled_oos][0].tz_convert("UTC").tz_localize(None).to_datetime64()

    fit = None
    scheme: QuantileScheme = "per_cross_section"
    if mode == "time_series":
        scheme = "fitted_on_train"
        fit = fit_quantiles(
            panel.factor[train_rows], panel.assets, eligible[train_rows],
            n_quantiles=n_quantiles, direction=direction, standardization=standardization,
        )
    results = {}
    for name, rows in (("train", train_rows), ("oos", oos_rows)):
        datetimes = panel.datetimes[rows]
        signal_mask = eligible[rows]
        returns, valid, intervals, counts = {}, {}, {}, {}
        for horizon, outcome in panel.returns.items():
            full_exit = np.asarray(exit_timestamps[horizon])
            if full_exit.shape != panel.factor.shape or full_exit.dtype.kind != "M":
                raise ValueError(f"Expected a panel-shaped datetime64 exit array for {horizon!r}")
            exits = full_exit[rows]
            pairs = panel.valid[horizon][rows] & signal_mask
            if np.any(pairs & np.isnat(exits)):
                raise ValueError(f"Missing exit timestamp for an eligible label at horizon {horizon!r}")
            purged = signal_mask & (exits > first_oos) if name == "train" else np.zeros_like(pairs)
            counts[horizon] = int(purged.sum())
            returns[horizon] = np.where(purged, np.nan, outcome[rows])
            valid[horizon] = pairs & ~purged
            if entry_timestamps is not None and mode == "time_series":
                entry = np.asarray(entry_timestamps[horizon])
                if entry.shape != panel.factor.shape or entry.dtype.kind != "M":
                    raise ValueError(f"Expected a panel-shaped datetime64 entry array for {horizon!r}")
                intervals[horizon] = (entry[rows], exits)

        current = replace(
            panel, datetimes=datetimes, factor=panel.factor[rows], returns=returns, valid=valid,
            _index=panel._index[panel._index.get_level_values(0).isin(datetimes)],
        )
        if mode == "cross_section":
            statistics = evaluate_cross_section(
                current, signal_valid=signal_mask, n_quantiles=n_quantiles, direction=direction,
                **statistics_options,
            )
        else:
            statistics = evaluate_time_series(
                current, intervals, signal_valid=signal_mask, quantile_fit=fit,
                expected_grid=None if grid is None else grid[rows], **statistics_options,
            )
        results[name] = SplitEvaluation(statistics, valid, counts, scheme, None if fit is None else fit.identity())
    return results
