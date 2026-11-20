"""Check label coverage, Spearman points and split clocks without rerunning evaluation."""
from dataclasses import dataclass, replace
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation.kernels import spearman
from backtest_and_analysis.time_series.evaluator import (
    COUNT_COLUMNS, _PreparedLabels, _counts, _observed_panel, _prepare,
)
from backtest_and_analysis.time_series.labels import LABEL_COLUMNS

from .grid_validation import LabelGrid
from .normalization import _format_utc_iso
from .schema import BarEvaluationIntegrityError

StreamKey = tuple[str, int, str | None]


@dataclass
class SampleMinimum:
    """Feasible effective sample minimum where a Python export omits config."""

    lower: int = 2
    upper: int | None = None

    @classmethod
    def from_summary(cls, summary: dict[str, Any], time_series: bool) -> "SampleMinimum":
        field = "min_ic_samples" if time_series else "min_cross_section_size"
        config = summary.get("config") or {}
        harness_config = "analysis_mode" in config and (
            config.get("bars_csv") is not None or config.get("continuous_futures") is not None
        )
        if field not in config and not harness_config:
            return cls()
        value = config.get(field, 3)  # The Harness worker resolves omitted options to 3.
        if type(value) is not int or value < 2:
            raise BarEvaluationIntegrityError(f"summary.json config.{field} must be an integer >= 2")
        return cls(value, value)

    def observe(self, count: int, insufficient: bool, field: str) -> None:
        if insufficient:
            self.lower = max(self.lower, count + 1)
        else:
            self.upper = count if self.upper is None else min(self.upper, count)
        if self.upper is not None and self.lower > self.upper:
            raise BarEvaluationIntegrityError(f"{field} is inconsistent with the effective sample minimum")

    def require_sections(self, widths: np.ndarray, n_sections: int, field: str) -> None:
        if n_sections > len(widths):
            raise BarEvaluationIntegrityError(f"{field} has more valid sections than per_bar_labels.csv")
        if self.upper == self.lower:
            if np.count_nonzero(widths >= self.lower) != n_sections:
                raise BarEvaluationIntegrityError(f"{field} count disagrees with the effective sample minimum")
            return
        # Finite correlations at the kernel minimum survive monotonically by width.
        widths = np.sort(widths)[::-1]
        if n_sections:
            self.observe(int(widths[n_sections - 1]), False, field)
        if n_sections < len(widths):
            self.observe(int(widths[n_sections]), True, field)


def check_point(claim: Any, actual: float, field: str) -> None:
    matches = (
        claim is None if not isfinite(actual)
        else type(claim) in (int, float) and isclose(claim, actual, rel_tol=1e-12, abs_tol=1e-12)
    )
    if not matches:
        raise BarEvaluationIntegrityError(f"{field} point estimate disagrees with per_bar_labels.csv")


def _check_descriptive_points(
    prepared: _PreparedLabels, summary: dict[str, Any], time_series: bool, minimum: SampleMinimum,
) -> None:
    ic_rows = summary["ic_by_horizon"]
    instrument_rows = {
        (row["frequency"], row["horizon_bars"], row["instrument_id"]): row
        for row in ic_rows if time_series and row["scope"] == "per_instrument"
    }
    sections = {}
    for row in summary["cross_sections"]:
        if (not isinstance(row, dict) or not isinstance(row.get("frequency"), str)
                or type(row.get("horizon_bars")) is not int or not isinstance(row.get("signal_timestamp"), str)):
            raise BarEvaluationIntegrityError("cross_sections must contain valid frequency, horizon and timestamp keys")
        sections[row["frequency"], row["horizon_bars"], row["signal_timestamp"]] = row
    if len(sections) != len(summary["cross_sections"]):
        raise BarEvaluationIntegrityError("cross_sections contains duplicate keys")

    points: dict[StreamKey, float] = {}
    for group in prepared.groups(by_instrument=time_series):
        freq, horizon = group.frequency, group.horizon
        if time_series:
            key = (freq, horizon, group.assets[group.instruments[0]])
            row = instrument_rows[key]
            result = spearman(group.factor[:, None], group.returns[:, None],
                              valid=(group.reasons < 0)[:, None], axis=0, min_samples=2)
            count = int(result.counts[0])
            minimum.observe(count, row["ic_status"] == "insufficient_samples", "ic_by_horizon.ic_status")
            points[key] = result.values[0] if count >= minimum.lower else np.nan
        else:
            panel = _observed_panel(group).panel
            result = spearman(panel.factor, panel.returns[horizon], valid=panel.valid[horizon], min_samples=2)
            valid = []
            for time, value, width in zip(panel.datetimes, result.values, result.counts, strict=True):
                row = sections.pop((freq, horizon, _format_utc_iso(time)), None)
                if row is None or row.get("valid_instruments") != width:
                    raise BarEvaluationIntegrityError("cross_sections keys or widths disagree with per_bar_labels.csv")
                minimum.observe(int(width), row.get("cs_status") == "insufficient_cross_section",
                                "cross_sections.cs_status")
                point = value if width >= minimum.lower else np.nan
                check_point(row.get("spearman_ic"), point, "cross_sections.spearman_ic")
                if isfinite(point):
                    valid.append(point)
            points[freq, horizon, None] = np.mean(valid) if valid else np.nan
    if sections:
        raise BarEvaluationIntegrityError("cross_sections contains timestamps absent from per_bar_labels.csv")

    for row in ic_rows:
        frequency, horizon = row["frequency"], row["horizon_bars"]
        if time_series and row["scope"] == "multi_instrument_summary":
            valid = [value for (freq, h, _), value in points.items() if (freq, h) == (frequency, horizon)
                     and isfinite(value)]
            check_point(row["mean_spearman_ic"], np.mean(valid) if valid else np.nan, "ic_by_horizon.mean_spearman_ic")
            check_point(row["median_spearman_ic"], np.median(valid) if valid else np.nan, "ic_by_horizon.median_spearman_ic")
        else:
            field = "spearman_ic" if time_series else "mean_ic"
            key = (frequency, horizon, row.get("instrument_id"))
            if key not in points:
                raise BarEvaluationIntegrityError("ic_by_horizon keys disagree with per_bar_labels.csv")
            check_point(row[field], points[key], f"ic_by_horizon.{field}")


def check_labels(
    path: Path, summary: dict[str, Any],
) -> tuple[dict[StreamKey, pd.DataFrame], SampleMinimum, pd.Index, pd.Index]:
    try:
        labels = pd.read_csv(path, dtype={"instrument_id": "string", "frequency": "string"},
                             keep_default_na=False, na_values=[""], float_precision="round_trip",
                             dtype_backend="numpy_nullable")
        if list(labels.columns) != [*LABEL_COLUMNS, "factor_value"]:
            raise ValueError("unexpected label columns")
        for field in ("signal_timestamp", "signal_available_at"):
            if not labels[field].astype("string").str.endswith("Z", na=False).all():
                raise ValueError(f"{field} must use UTC timestamps")
            labels[field] = pd.to_datetime(labels[field], format="ISO8601", utc=True)
        prepared = _prepare(labels, None)
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        raise BarEvaluationIntegrityError(f"Invalid per_bar_labels.csv: {exc}") from exc
    if _counts(prepared.reasons) != summary["exclusion_summary"]:
        raise BarEvaluationIntegrityError("per_bar_labels.csv exclusion counts disagree with summary.json")

    time_series = summary["analysis_mode"] == "time_series"
    coverage = {
        (g.frequency, g.horizon, g.assets[g.instruments[0]] if time_series else None): _counts(g.reasons)
        for g in prepared.groups(by_instrument=time_series)
    }
    ic_coverage = {
        (r["frequency"], r["horizon_bars"], r.get("instrument_id")): {k: r[k] for k in COUNT_COLUMNS}
        for r in summary["ic_by_horizon"] if not time_series or r.get("scope") == "per_instrument"
    }
    if coverage != ic_coverage:
        raise BarEvaluationIntegrityError("per_bar_labels.csv stream coverage disagrees with ic_by_horizon")
    minimum = SampleMinimum.from_summary(summary, time_series)
    _check_descriptive_points(prepared, summary, time_series, minimum)

    if not summary.get("splits"):
        return {}, minimum, prepared.assets, prepared.frequencies
    columns = ["frequency", "horizon_bars"]
    if time_series:
        columns.append("instrument_id")
    streams = {key if time_series else (*key, None): group for key, group in labels.groupby(columns)}
    return streams, minimum, prepared.assets, prepared.frequencies


def check_split_labels(
    labels: pd.DataFrame, *, grid: LabelGrid, time_series: bool, split: str, start: pd.Timestamp, end: pd.Timestamp,
    train_end: pd.Timestamp | None, oos_start: pd.Timestamp | None, oos_grid_start: pd.Timestamp | None,
    grid_size: int, minimum: SampleMinimum, claimed_count: int, point_reason: str | None, field: str,
) -> tuple[float, int, np.ndarray | None, np.ndarray | None]:
    """Bind a split to the observed clocks and return its IC and section data."""
    available = labels["signal_available_at"]
    clock = labels["signal_timestamp"] if time_series else available
    in_window = pd.Series(True, index=labels.index)
    if split == "train":
        in_window = available <= train_end
    elif split == "oos":
        in_window = available >= oos_start

    observed_clock = pd.DatetimeIndex(clock[in_window].unique())
    observed_size = len(observed_clock)
    minimum_grid_size = observed_size + sum(endpoint not in observed_clock for endpoint in {start, end})
    if (not clock.min() <= start <= end <= clock.max()
            or (split != "oos" and start != clock.min())
            or (split != "train" and end != clock.max())
            or (split == "train" and (end > train_end or (time_series and end == train_end)))
            or (split == "oos" and not time_series and start < oos_start)
            or not np.array_equal(clock.between(start, end), in_window)
            or grid_size < minimum_grid_size):
        raise BarEvaluationIntegrityError(
            f"{field}.grid_start_utc/grid_end_utc or grid_size disagrees with the split label clock"
        )
    expected = grid.check(time_series=time_series, split=split, start=start, end=end, train_end=train_end,
                          oos_start=oos_start, grid_size=grid_size, observed=observed_clock, field=field)

    pairs = labels.loc[in_window & labels["status"].eq("valid")]
    if split == "train":
        # On a single instrument's grid, Bar starts order its valid exits exactly as ends do.
        exit_col = "exit_bar_timestamp" if time_series else "exit_timestamp"
        try:
            exits = pd.to_datetime(pairs[exit_col], format="ISO8601", utc=True)
        except (ValueError, TypeError) as exc:
            raise BarEvaluationIntegrityError(f"Invalid per_bar_labels.csv {exit_col}: {exc}") from exc
        if exits.isna().any():
            raise BarEvaluationIntegrityError(f"Missing per_bar_labels.csv {exit_col} for a valid pair")
        pairs = pairs.loc[exits <= oos_grid_start]

    if pairs.empty:
        if time_series:
            minimum.observe(0, point_reason == "insufficient_samples", field)
            return np.nan, 0, None, None
        if expected is None:
            return np.nan, 0, np.empty(0, dtype=int), None
        return np.nan, 0, np.zeros(grid_size, dtype=int), np.full(grid_size, np.nan)
    prepared = _prepare(pairs, None)
    if time_series:
        point = spearman(prepared.factor[:, None], prepared.returns[:, None], axis=0, min_samples=2)
        count = int(point.counts[0])
        minimum.observe(count, point_reason == "insufficient_samples", field)
        return (point.values[0] if count >= minimum.lower else np.nan), count, None, None

    # Cross-sectional inference groups by when each signal becomes available.
    times = pd.DatetimeIndex(pairs["signal_available_at"])
    if prepared.order is not None:
        times = times.take(prepared.order)
    group = replace(next(prepared.groups()), times=times)
    panel = _observed_panel(group).panel
    point = spearman(panel.factor, panel.returns[group.horizon], valid=panel.valid[group.horizon],
                     min_samples=2)
    finite = np.isfinite(point.values)
    minimum.require_sections(point.counts[finite], claimed_count, field)
    selected = point.counts >= minimum.lower
    values = np.where(selected, point.values, np.nan)
    valid = values[np.isfinite(values)]
    mean = valid.mean() if valid.size else np.nan
    if expected is None:
        return mean, len(valid), point.counts, None
    positions = expected.get_indexer(panel.datetimes)
    if (positions < 0).any():
        raise BarEvaluationIntegrityError(f"{field} label sections fall outside the verified split grid")
    sizes = np.zeros(grid_size, dtype=int)
    aligned = np.full(grid_size, np.nan)
    sizes[positions], aligned[positions] = point.counts, values
    return mean, len(valid), sizes, aligned
