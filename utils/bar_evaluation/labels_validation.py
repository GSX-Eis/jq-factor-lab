"""Check label coverage, Spearman points and split clocks without rerunning evaluation."""
from dataclasses import asdict, dataclass, replace
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation.cross_section import _divide, _time_mean
from backtest_and_analysis.evaluation.kernels import bucket_sums, quantile_buckets, spearman
from backtest_and_analysis.evaluation.time_series import _distribution, _missing_pattern
from backtest_and_analysis.time_series.contracts import _numeric_column
from backtest_and_analysis.time_series.evaluator import (
    SLOT_RETURN_COLUMNS,
    COUNT_COLUMNS, _PreparedLabels, _counts, _observed_panel, _prepare,
)
from backtest_and_analysis.time_series.labels import LABEL_COLUMNS
from utils.time_axis import Frequency

from .grid_validation import LabelGrid, build_label_grid, matching_calendars, resolve_bucket_strategy
from .normalization import _array, _coerce_optional_int, _count, _format_utc_iso, _metric, _object, _require
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


def _check_label_frequency(summary: dict[str, Any], observed: pd.Index, *, explicit_contract: bool) -> None:
    declared = summary.get("frequency")
    parsed = Frequency.parse(declared)
    _require(declared == "bar" or parsed.is_valid, "frequency", "bar or a supported trading frequency")
    if explicit_contract:
        _require(all(Frequency.parse(value).is_valid for value in observed),
                 "frequency", "supported trading frequencies in per_bar_labels.csv")
    canonical_observed = {Frequency.parse(value).canonical or value for value in observed}
    identity = summary["data_identity"]
    canonical_inputs = set()
    if "frequencies" in identity:
        field = "data_identity.frequencies"
        for value in _array(identity["frequencies"], field):
            try:
                Frequency.contract_duration(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise BarEvaluationIntegrityError(f"summary.json {field} contains an invalid frequency") from exc
            input_frequency = Frequency.parse(value)
            _require(input_frequency.is_valid or not explicit_contract, field, "supported trading frequencies")
            canonical_inputs.add(input_frequency.canonical or value)
        _require(canonical_observed <= canonical_inputs, field, "coverage of per_bar_labels.csv frequencies")
    if parsed.is_valid:
        _require(canonical_observed | canonical_inputs == {parsed.canonical},
                 "frequency", "the declared trading frequency in the labels or frozen input axes")


def _check_label_return_unit(labels: pd.DataFrame, unit: str) -> None:
    prices = SLOT_RETURN_COLUMNS if set(SLOT_RETURN_COLUMNS) <= set(labels) else ("entry_price", "exit_price")
    fields = [*prices, "return_pct"]
    valid = labels.loc[labels["status"].eq("valid"), fields]
    if valid.empty:
        return
    entry, exit_, recorded = (_numeric_column(valid[name]) for name in fields)
    if not (np.isfinite(entry).all() and np.isfinite(exit_).all() and (entry > 0).all() and (exit_ > 0).all()):
        raise ValueError("valid labels require finite positive entry_price and exit_price to verify return_unit")
    with np.errstate(over="ignore"):
        expected = (exit_ / entry - 1) * (100 if unit == "percent" else 1)
    if not np.isclose(recorded, expected, rtol=1e-12, atol=1e-12).all():
        raise ValueError(f"return_pct disagrees with the declared return_unit={unit} and the execution prices")


def _stream_key(row: dict[str, Any]) -> StreamKey:
    return row["frequency"], row["horizon_bars"], row.get("instrument_id")


def _section_bucket_returns(factor: np.ndarray, returns: np.ndarray, n_quantiles: int,
                            eligible: np.ndarray, valid: np.ndarray) -> np.ndarray:
    quantiles = quantile_buckets(factor, n_quantiles, valid=eligible)
    quantiles.labels[quantiles.n_bins < 2] = 0
    buckets = bucket_sums(returns, quantiles.labels, n_quantiles, valid=valid)
    return _divide(buckets.sums, buckets.counts)


def _check_descriptive_returns(prepared: _PreparedLabels, summary: dict[str, Any], *,
                               time_series: bool, minimum: SampleMinimum) -> dict[StreamKey, int]:
    counts: dict[StreamKey, int] = {}
    axes = ([row for row in summary["ic_by_horizon"] if row["scope"] == "per_instrument"]
            if time_series else summary["cross_sections"])
    for row in axes:
        key = _stream_key(row)
        count = _count(_coerce_optional_int(row.get("requested_quantiles"), "requested_quantiles"),
                       "requested_quantiles", minimum=2)
        _require(counts.setdefault(key, count) == count, "requested_quantiles", "one bin count per stream")
    claims = {}
    for row in summary["quantile_stats"]:
        number = _count(_coerce_optional_int(row["quantile"], "quantile_stats.quantile"),
                        "quantile_stats.quantile", minimum=1)
        key = (*_stream_key(row), number)
        _require(key not in claims, "quantile_stats", "unique stream and quantile keys")
        claims[key] = row

    for group in prepared.groups(by_instrument=time_series):
        instrument = group.assets[group.instruments[0]] if time_series else None
        key = group.frequency, group.horizon, instrument
        n_quantiles = counts[key]
        if time_series:
            valid = (group.reasons < 0)[:, None]
            quantiles = quantile_buckets(group.factor[:, None], n_quantiles, axis=0, valid=valid)
            bins = quantiles.labels[:, 0]
            buckets = [group.returns[bins == bucket] for bucket in np.unique(bins[bins > 0])]
        else:
            panel = _observed_panel(group).panel
            valid = panel.valid[group.horizon]
            returns = _section_bucket_returns(panel.factor, panel.returns[group.horizon], n_quantiles, valid, valid)
            occupied = np.isfinite(returns)
            usable = (valid.sum(axis=1) >= minimum.lower) & (occupied.sum(axis=1) >= 2)
            numbers = np.where(occupied & usable[:, None], occupied.cumsum(axis=1), 0)
            buckets = [returns[numbers == number] for number in range(1, int(numbers.max(initial=0)) + 1)]
        if len(buckets) < 2:
            continue
        for number, values in enumerate(buckets, 1):
            field = f"quantile_stats[{key}, {number}]"
            row = _object(claims.pop((*key, number), None), field)
            check_point(row.get("mean_return_pct"), float(values.mean()), f"{field}.mean_return_pct")
            check_point(row.get("median_return_pct"), float(np.median(values)), f"{field}.median_return_pct")
    _require(not claims, "quantile_stats", "buckets derived from the labeled returns")
    return counts


def _check_return_summary(claim: Any, expected: dict[str, Any], field: str) -> None:
    report = _object(claim, field, *expected)
    for name, value in expected.items():
        if name in ("n_samples", "n_times", "reason"):
            _require(type(report[name]) is type(value) and report[name] == value, f"{field}.{name}",
                     "the value derived from the labeled returns")
        else:
            check_point(report[name], value, f"{field}.{name}")


def _check_split_returns(window: pd.DataFrame, pairs: pd.DataFrame, *,
                         row: dict[str, Any], n_quantiles: int, fit: dict[str, Any] | None,
                         time_series: bool, field: str) -> None:
    prepared = _prepare(window, None, minimum_offset=0)
    kept = window.index.isin(pairs.index)
    eligible = window["factor_value"].notna().to_numpy()
    if prepared.order is not None:
        kept = kept[prepared.order]
        eligible = eligible[prepared.order]
    buckets = _object(row.get("buckets"), f"{field}.buckets")
    if time_series:
        factor, returns = prepared.factor, prepared.returns
        if fit is None:
            quantiles = quantile_buckets(factor[:, None], n_quantiles, axis=0, valid=eligible[:, None])
            bins, n_bins = quantiles.labels[:, 0], int(quantiles.n_bins[0])
        else:
            instruments = _array(fit["instruments"], f"{field}.fit_identity.instruments")
            _require(len(instruments) == 1 and fit["n_quantiles"] == n_quantiles,
                     f"{field}.fit_identity", "one instrument and the requested quantile count")
            instrument = _object(instruments[0], f"{field}.fit_identity.instruments[]", "asset", "training_edges")
            _require(instrument["asset"] == row["instrument_id"], f"{field}.fit_identity", "the split instrument")
            edges = _array(instrument["training_edges"], f"{field}.fit_identity.training_edges")
            _require(len(edges) <= n_quantiles + 1
                     and all(_metric(edge, f"{field}.fit_identity.training_edges") is not None for edge in edges)
                     and all(left < right for left, right in zip(edges, edges[1:])),
                     f"{field}.fit_identity.training_edges", "finite strictly increasing boundaries")
            n_bins = len(edges) - 1
            bins = np.searchsorted(edges[1:-1], factor, side="left") + 1
        _check_return_summary(row.get("return_distribution"), asdict(_distribution(returns[kept], "insufficient_samples")),
                              f"{field}.return_distribution")
        bucket_ids = range(1, n_bins + 1) if n_bins >= 2 else range(0)
        _require(set(buckets) == {str(bucket) for bucket in bucket_ids}, f"{field}.buckets", "the fitted buckets")
        for bucket in bucket_ids:
            name = f"{field}.buckets.{bucket}.return_distribution"
            claim = _object(buckets[str(bucket)], name, "return_distribution")["return_distribution"]
            _check_return_summary(claim, asdict(_distribution(returns[kept & (bins == bucket)], "empty_bucket")), name)
        return

    times = pd.DatetimeIndex(window["signal_available_at"])
    if prepared.order is not None:
        times = times.take(prepared.order)
    group = replace(next(prepared.groups()), times=times)
    observed = _observed_panel(group)
    panel = observed.panel
    valid = np.zeros_like(panel.valid[group.horizon])
    positions = observed.row_codes, observed.asset_codes
    valid[positions] = kept
    signals = np.zeros_like(valid)
    signals[positions] = eligible
    # The observed panel masks unpaired factors on sparse grids, but split buckets
    # are formed from the signal-time universe before returns or purging are known.
    panel.factor[positions] = group.factor
    returns = _section_bucket_returns(panel.factor, panel.returns[group.horizon], n_quantiles, signals, valid)
    _require(set(buckets) == {str(bucket) for bucket in range(1, n_quantiles + 1)},
             f"{field}.buckets", "the cross-sectional buckets")
    for bucket, values in enumerate(returns.T, 1):
        _check_return_summary(buckets[str(bucket)], asdict(_time_mean(values, "empty_bucket")),
                              f"{field}.buckets.{bucket}")
    _check_return_summary(row.get("spread"), asdict(_time_mean(returns[:, -1] - returns[:, 0],
                                                               "no_paired_bucket_returns")), f"{field}.spread")


def _parse_utc_column(series: pd.Series, *, require_z: bool = False, field_name: str = "timestamp") -> pd.Series:
    """Parse an ISO8601 UTC string column, deduplicating repeated grid timestamps."""
    n = len(series)
    if n >= 64:
        codes, uniques = pd.factorize(series, sort=False)
        if len(uniques) * 2 <= n:
            if require_z:
                if (codes < 0).any() or not pd.Series(uniques, dtype="string").str.endswith("Z", na=False).all():
                    raise ValueError(f"{field_name} must use UTC timestamps")
            if len(uniques) == 0:
                return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
            parsed = pd.to_datetime(uniques, format="ISO8601", utc=True)
            if (codes < 0).any():
                safe = np.where(codes >= 0, codes, 0)
                return pd.Series(parsed.take(safe), index=series.index).where(codes >= 0, pd.NaT)
            return pd.Series(parsed.take(codes), index=series.index)
    if require_z and not series.astype("string").str.endswith("Z", na=False).all():
        raise ValueError(f"{field_name} must use UTC timestamps")
    return pd.to_datetime(series, format="ISO8601", utc=True)


def check_labels(
    path: Path, summary: dict[str, Any], *, minimum_offset: int, explicit_offset: int | None,
    explicit_contract: bool,
) -> tuple[dict[StreamKey, pd.DataFrame], SampleMinimum, pd.Index, pd.Index, dict[StreamKey, int]]:
    try:
        labels = pd.read_csv(path, dtype={"instrument_id": "string", "frequency": "string"},
                             keep_default_na=False, na_values=[""], float_precision="round_trip",
                             dtype_backend="numpy_nullable")
        columns = [*LABEL_COLUMNS, "factor_value"]
        accepted = [columns]
        if "slot_id" in summary["data_identity"]:
            accepted.append([*columns, *SLOT_RETURN_COLUMNS])
        if list(labels.columns) not in accepted:
            raise ValueError("unexpected label columns")
        for field in ("signal_timestamp", "signal_available_at"):
            labels[field] = _parse_utc_column(labels[field], require_z=True, field_name=field)
        if explicit_contract:
            offset = labels["entry_offset_bars"]
            if not labels.empty and (not pd.api.types.is_integer_dtype(offset.dtype) or offset.hasnans
                                     or (offset < minimum_offset).any()
                                     or (explicit_offset is not None and (offset != explicit_offset).any())):
                raise ValueError("entry_offset_bars must match the effective execution configuration")
            prepared = _prepare(labels, None, minimum_offset=minimum_offset)
            _check_label_return_unit(labels, summary["return_unit"])
        else:
            prepared = _prepare(labels, None)
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        raise BarEvaluationIntegrityError(f"Invalid per_bar_labels.csv: {exc}") from exc
    _check_label_frequency(summary, prepared.frequencies, explicit_contract=explicit_contract)
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
    quantile_counts = (_check_descriptive_returns(prepared, summary, time_series=time_series, minimum=minimum)
                       if explicit_contract else {})

    if not summary.get("splits"):
        if explicit_contract and (calendars := matching_calendars(summary)) and not labels.empty:
            strategy = resolve_bucket_strategy(summary, calendars, instruments=prepared.assets,
                                               frequencies=prepared.frequencies)
            for _, group in labels.groupby(["frequency", "horizon_bars"]):
                build_label_grid(summary, group, calendars, bucket_strategy=strategy)
        return {}, minimum, prepared.assets, prepared.frequencies, quantile_counts
    columns = ["frequency", "horizon_bars"]
    if time_series:
        columns.append("instrument_id")
    streams = {key if time_series else (*key, None): group for key, group in labels.groupby(columns)}
    return streams, minimum, prepared.assets, prepared.frequencies, quantile_counts


def check_split_labels(
    labels: pd.DataFrame, *, grid: LabelGrid, time_series: bool, split: str, start: pd.Timestamp, end: pd.Timestamp,
    train_end: pd.Timestamp | None, oos_start: pd.Timestamp | None, oos_grid_start: pd.Timestamp | None,
    grid_size: int, minimum: SampleMinimum, claimed_count: int, point_reason: str | None, field: str,
    bootstrap_v2: bool, row: dict[str, Any], n_quantiles: int | None, fit: dict[str, Any] | None,
) -> tuple[float, int, np.ndarray | None, np.ndarray | None, str | None]:
    """Bind a split to the label clock and derive applicable HAC or Bootstrap audit data."""
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
    verified_session = grid.scheduled and grid.session_or_week
    if (not clock.min() <= start <= end <= clock.max()
            or (split != "oos" and start != clock.min())
            or (split != "train" and end != clock.max())
            or (split == "train" and (end > train_end or (time_series and not verified_session and end == train_end)))
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
        exit_col = "exit_timestamp"
        cutoff = oos_grid_start
        if time_series and verified_session:
            # A daily or weekly signal may be keyed by its opening but becomes available at close.
            starts, ends = grid.slots[0]
            position = starts.get_indexer([oos_grid_start])[0]
            if position < 0:
                raise BarEvaluationIntegrityError(f"{field} OOS grid_start_utc disagrees with the Bar schedule")
            cutoff = ends[position]
        elif time_series:
            # On a legacy intraday grid, Bar starts order its valid exits exactly as ends do.
            exit_col = "exit_bar_timestamp"
        try:
            exits = _parse_utc_column(pairs[exit_col])
        except (ValueError, TypeError) as exc:
            raise BarEvaluationIntegrityError(f"Invalid per_bar_labels.csv {exit_col}: {exc}") from exc
        if exits.isna().any():
            raise BarEvaluationIntegrityError(f"Missing per_bar_labels.csv {exit_col} for a valid pair")
        pairs = pairs.loc[exits <= cutoff]

    if n_quantiles is not None:
        _check_split_returns(labels.loc[in_window], pairs, row=row, n_quantiles=n_quantiles, fit=fit,
                             time_series=time_series, field=field)

    pair_pattern = None
    if bootstrap_v2 and expected is not None:
        pair_mask = expected.isin(pairs["signal_timestamp"])
        if pair_mask.sum() != len(pairs):
            raise BarEvaluationIntegrityError(f"{field} label pairs fall outside the verified split grid")
        pair_pattern = _missing_pattern(pair_mask)

    if pairs.empty:
        if time_series:
            minimum.observe(0, point_reason == "insufficient_samples", field)
            return np.nan, 0, None, None, pair_pattern
        if expected is None:
            return np.nan, 0, np.empty(0, dtype=int), None, None
        return np.nan, 0, np.zeros(grid_size, dtype=int), np.full(grid_size, np.nan), None
    prepared = _prepare(pairs, None, minimum_offset=0)
    if time_series:
        point = spearman(prepared.factor[:, None], prepared.returns[:, None], axis=0, min_samples=2)
        count = int(point.counts[0])
        minimum.observe(count, point_reason == "insufficient_samples", field)
        return (point.values[0] if count >= minimum.lower else np.nan), count, None, None, pair_pattern

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
        return mean, len(valid), point.counts, None, None
    positions = expected.get_indexer(panel.datetimes)
    if (positions < 0).any():
        raise BarEvaluationIntegrityError(f"{field} label sections fall outside the verified split grid")
    sizes = np.zeros(grid_size, dtype=int)
    aligned = np.full(grid_size, np.nan)
    sizes[positions], aligned[positions] = point.counts, values
    return mean, len(valid), sizes, aligned, None
