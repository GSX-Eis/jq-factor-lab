"""Explicit time-series and same-timestamp cross-sectional Bar evaluation.

IC uses average ranks. Full-sample time-series quantiles are descriptive;
overlapping returns never imply unadjusted statistical significance.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation import (
    EvaluationEngine, IntervalUsability, evaluate_interval_usability,
)
from backtest_and_analysis.evaluation.engine import DescriptiveSample, _validate_hac_lag
from backtest_and_analysis.evaluation.kernels import _validate_direction
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.quantiles import Standardization, _validate_standardization
from backtest_and_analysis.evaluation.split import SplitConfig, SplitEvaluation
from backtest_and_analysis.evaluation.time_series import (
    _validate_block_length_for_horizon, _validate_bootstrap_options,
)
from backtest_and_analysis.evaluation.types import CrossSectionResult, TimeSeriesResult

from .contracts import (
    EXCLUSION_PRIORITY, ContractValidationError, DuplicateKeyError,
    _duration, _frame_columns, _mask_unavailable_factor_values, _numeric_column, _text_column,
    _validate_aligned_factor_frame, validate_bar_frame,
)
from .inference import (
    BarTrialFamily, _SelectionWindow, _bar_panel, _record_trials, _selection_window, build_bar_splits,
)
from .labels import BarReturnLabeler, LABEL_COLUMNS

if TYPE_CHECKING:
    from utils.session_calendar import SessionCalendar


RESULT_SCHEMA_VERSION = 'bar_evaluation.v2'
LABEL_KEY = ['instrument_id', 'frequency', 'horizon_bars', 'signal_timestamp']
EXCLUSION_COLUMNS = [reason.value for reason in EXCLUSION_PRIORITY]
_PARALLEL_LABEL_ROWS = 1_000_000
COUNT_COLUMNS = ['total_count', 'valid_count', 'excluded_count', 'exclusion_counts',
                 'nonzero_exclusion_counts']
QUANTILE_META = ['requested_quantiles', 'actual_quantiles', 'quantile_status',
                 'quantile_scope', 'descriptive_only']
QUANTILE_VALUES = ['mean_return_pct', 'median_return_pct', 'positive_ratio']
TS_QUANTILE_COLUMNS = ['instrument_id', 'frequency', 'horizon_bars', 'quantile',
                       *QUANTILE_META, 'sample_count', *QUANTILE_VALUES]
CS_QUANTILE_COLUMNS = ['frequency', 'horizon_bars', 'quantile', 'requested_quantiles',
                       'min_actual_quantiles', 'max_actual_quantiles', 'quantile_status',
                       'quantile_scope', 'descriptive_only', 'cross_section_count',
                       'total_sample_count', *QUANTILE_VALUES]
TS_IC_COLUMNS = ['scope', 'instrument_id', 'frequency', 'horizon_bars', *COUNT_COLUMNS,
                 'spearman_ic', 'ic_status', 'non_overlapping_ic', 'non_overlapping_count',
                 'non_overlapping_status', *QUANTILE_META, 'mean_spearman_ic',
                 'median_spearman_ic', 'mean_non_overlapping_ic', 'median_non_overlapping_ic',
                 'instrument_count', 'valid_instrument_count', 'valid_non_overlapping_instrument_count']
CS_IC_COLUMNS = ['frequency', 'horizon_bars', *COUNT_COLUMNS, 'mean_ic', 'median_ic',
                 'std_ic', 'icir', 'positive_ic_ratio', 'total_cross_sections',
                 'valid_cross_sections', 'excluded_cross_sections',
                 'insufficient_cross_section_count', 'constant_factor_cross_section_count',
                 'constant_return_cross_section_count', 'mean_valid_instruments',
                 'median_valid_instruments', 'ic_status']
CS_COLUMNS = ['frequency', 'horizon_bars', 'signal_timestamp', *COUNT_COLUMNS,
              'valid_instruments', 'spearman_ic', 'cs_status', *QUANTILE_META]


ICInferenceKey = tuple[str, int, str | None]


@dataclass
class EvaluationResult:
    """Standard Bar tables and optional calendar-grid inference from one evaluation."""
    analysis_mode: str
    per_bar_labels: pd.DataFrame
    ic_by_horizon: pd.DataFrame
    quantile_stats: pd.DataFrame
    exclusion_summary: dict[str, Any]
    cross_sections: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CS_COLUMNS))
    data_identity: dict[str, Any] = field(default_factory=dict)
    calendar_identity: dict[str, Any] = field(default_factory=dict)
    ic_inference: dict[ICInferenceKey, IntervalUsability] = field(default_factory=dict)
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)
    trial_ledger: dict[str, Any] | None = None


def _mode(value: str | None, expected: str | None = None) -> str:
    if not isinstance(value, str) or value not in ('time_series', 'cross_sectional') or (
            expected is not None and value != expected):
        raise ContractValidationError('analysis_mode must explicitly specify '
                                      + (expected or 'time_series or cross_sectional'))
    return value


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
        raise ContractValidationError(f'{name} must be an integer >= 2')
    return int(value)


@dataclass(frozen=True)
class _LabelGroup:
    frequency: str
    horizon: int
    times: pd.DatetimeIndex
    instruments: np.ndarray
    assets: pd.Index
    factor: np.ndarray
    returns: np.ndarray
    reasons: np.ndarray


@dataclass(frozen=True)
class _PreparedLabels:
    frame: pd.DataFrame
    assets: pd.Index
    frequencies: pd.Index
    order: np.ndarray | None
    times: pd.DatetimeIndex
    keys: tuple[np.ndarray, np.ndarray, np.ndarray]
    factor: np.ndarray
    returns: np.ndarray
    reasons: np.ndarray

    def canonical_frame(self) -> pd.DataFrame:
        columns = (*LABEL_COLUMNS, 'factor_value')
        frame = self.frame
        if tuple(frame.columns) != columns:
            frame = frame.loc[:, list(columns)]
        frame = frame.copy() if self.order is None else frame.take(self.order)
        frame.index = pd.RangeIndex(len(frame))
        return frame

    def groups(self, *, by_instrument: bool = False) -> Iterator[_LabelGroup]:
        instruments, frequencies, horizons = self.keys
        if not len(horizons):
            return
        changes = np.flatnonzero((instruments[1:] != instruments[:-1])
                                 | (frequencies[1:] != frequencies[:-1])
                                 | (horizons[1:] != horizons[:-1])) + 1
        boundaries = np.r_[0, changes, len(horizons)]
        spans: dict[tuple[int, int, int], list[slice]] = {}
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            key = (int(frequencies[start]), int(horizons[start]), int(instruments[start]) if by_instrument else -1)
            spans.setdefault(key, []).append(slice(start, end))
        for (frequency, horizon, _), pieces in sorted(spans.items()):
            if len(spans) == 1:
                take = slice(None)
            elif len(pieces) == 1:
                take = pieces[0]
            else:
                take = np.concatenate([np.arange(piece.start, piece.stop) for piece in pieces])
            yield _LabelGroup(self.frequencies[frequency], horizon, self.times[take],
                              instruments[take], self.assets, self.factor[take],
                              self.returns[take], self.reasons[take])

    def identity(self) -> dict[str, Any]:
        return dict(instruments=sorted(self.assets.tolist()), frequencies=sorted(self.frequencies.tolist()),
                    sources=[], data_versions=[], raw_coverage_start_utc=None,
                    raw_coverage_end_utc=None)


def _contract_codes(column: pd.Series, allowed: list[str], message: str) -> np.ndarray:
    try:
        codes, values = pd.factorize(column.to_numpy(copy=False), size_hint=len(allowed))
        lookup = pd.Index(allowed).get_indexer(values)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(message) from exc
    if (lookup < 0).any():
        raise ContractValidationError(message)
    # A missing input has code -1 and indexes the last element.
    return np.r_[lookup.astype(np.int8), np.int8(-1)][codes]


def _canonical_order(keys: tuple[np.ndarray, ...]) -> np.ndarray | None:
    # Ignore constant keys when ordering; an interleaved timestamp × horizon
    # rectangle can be transposed without sorting millions of equal keys.
    active = tuple(key for key in keys if len(key) and not np.all(key == key[0]))
    size = len(keys[0])
    if not active:
        if size > 1:
            raise DuplicateKeyError(f'duplicate label keys: {LABEL_KEY}')
        return None
    if len(active) == 2 and active[0] is keys[2]:
        horizons, timestamps = active
        stride = int(np.argmax(timestamps != timestamps[0]))
        if 1 < stride < size and size % stride == 0:
            grid = timestamps.reshape(-1, stride)
            pattern = horizons[:stride]
            if (np.all(grid[:, 1:] == grid[:, :1]) and np.all(grid[1:, 0] > grid[:-1, 0])
                    and np.all(pattern[1:] > pattern[:-1])
                    and np.all(horizons.reshape(-1, stride) == pattern)):
                return np.arange(size).reshape(-1, stride).T.ravel()
    same = np.ones(max(0, size - 1), dtype=bool)
    descending = np.zeros_like(same)
    for key in active:
        descending |= same & (key[1:] < key[:-1])
        same &= key[1:] == key[:-1]
    if same.any():
        raise DuplicateKeyError(f'duplicate label keys: {LABEL_KEY}')
    if not descending.any():
        return None
    order = np.lexsort(active[::-1])
    same.fill(True)
    for key in active:
        ordered = key[order]
        same &= ordered[1:] == ordered[:-1]
    if same.any():
        raise DuplicateKeyError(f'duplicate label keys: {LABEL_KEY}')
    return order


def _attach_factor_values(labels: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    values = _validate_aligned_factor_frame(factors)[['timestamp_utc', 'instrument_id', 'factor_value']]
    return labels.reset_index(drop=True).drop(columns='factor_value', errors='ignore').merge(
        values.rename(columns={'timestamp_utc': 'signal_timestamp'}),
        on=['signal_timestamp', 'instrument_id'], how='left', validate='many_to_one')


def _prepare(labels: pd.DataFrame, factors: pd.DataFrame | None) -> _PreparedLabels:
    _frame_columns(labels, LABEL_COLUMNS)
    frame = labels
    for name in ('signal_timestamp', 'signal_available_at'):
        times = frame[name]
        if (not isinstance(times.dtype, pd.DatetimeTZDtype)
                or str(times.dt.tz) not in ('UTC', 'Etc/UTC') or times.isna().any()):
            raise ContractValidationError(f'{name} must have UTC datetime dtype and no NaT')
    text_keys = []
    for name in ('instrument_id', 'frequency'):
        try:
            codes, values = pd.factorize(frame[name].to_numpy(copy=False), sort=True, size_hint=64)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(f'{name} must contain nonempty strings') from exc
        if (codes < 0).any():
            raise ContractValidationError(f'{name} must contain nonempty strings')
        _text_column(pd.Series(values, name=name))
        if len(values) == 1:
            codes = np.broadcast_to(np.int8(0), codes.shape)
        text_keys.append((codes, pd.Index(values, name=name)))
    if factors is not None:
        frame = _attach_factor_values(frame, factors)
    _frame_columns(frame, ('factor_value',))
    # The labeler's empty contract table has object value columns; normalize only empties.
    if frame.empty:
        frame = frame.copy()
        for name in ('factor_value', 'return_pct'):
            frame[name] = frame[name].astype('float64')
    (instrument_codes, assets), (frequency_codes, frequencies) = text_keys
    for frequency in frequencies:
        _duration(frequency)
    for name in ('horizon_bars', 'entry_offset_bars'):
        column = frame[name]
        if pd.api.types.is_integer_dtype(column.dtype):
            invalid = column.hasnans or (not column.empty and column.min() < 1)
        else:
            invalid = any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1
                          for v in column)
        if invalid:
            raise ContractValidationError(f'{name} must contain positive integers')
    horizons = frame.horizon_bars.to_numpy()
    keys = (instrument_codes, frequency_codes, horizons, frame.signal_timestamp.array.asi8)
    order = _canonical_order(keys)
    status_codes = _contract_codes(frame.status, ['excluded', 'valid'], 'status must be valid or excluded')
    if (status_codes < 0).any():
        raise ContractValidationError('status must be valid or excluded')
    valid = status_codes == 1
    reason_codes = _contract_codes(
        frame.exclusion_reason, EXCLUSION_COLUMNS, 'status and exclusion_reason disagree',
    )
    if np.any(valid != (reason_codes < 0)):
        raise ContractValidationError('status and exclusion_reason disagree')
    values = _numeric_column(frame.factor_value)
    returns = _numeric_column(frame.return_pct)
    if np.isinf(values).any() or np.any(valid & ~np.isfinite(values)):
        raise ContractValidationError('valid labels require finite factor_value; excluded may have NaN')
    if np.any(valid & ~np.isfinite(returns)):
        raise ContractValidationError('valid labels require finite return_pct')
    dtype = frame.factor_value.dtype
    if pd.api.types.is_integer_dtype(dtype) and len(frame):
        native = frame.factor_value.to_numpy(dtype=getattr(dtype, 'numpy_dtype', dtype), na_value=0)
        if int(native.min()) < -(2**53) or int(native.max()) > 2**53:
            values = native
    times = pd.DatetimeIndex(frame.signal_timestamp)
    if order is not None:
        if len(assets) > 1:
            instrument_codes = instrument_codes[order]
        if len(frequencies) > 1:
            frequency_codes = frequency_codes[order]
        times, horizons = times.take(order), horizons[order]
        reason_codes, values, returns = reason_codes[order], values[order], returns[order]
    return _PreparedLabels(frame, assets, frequencies, order, times, (instrument_codes, frequency_codes, horizons),
                           values, returns, reason_codes)


def _counts(reasons: np.ndarray) -> dict[str, Any]:
    counts = dict(zip(EXCLUSION_COLUMNS, np.bincount(reasons[reasons >= 0],
                                                     minlength=len(EXCLUSION_COLUMNS)).tolist(), strict=True))
    return _count_summary(len(reasons), counts)


def _count_summary(total: int, counts: dict[str, int]) -> dict[str, Any]:
    excluded = sum(counts.values())
    return dict(total_count=total, valid_count=total - excluded, excluded_count=excluded,
                exclusion_counts=counts, nonzero_exclusion_counts={k: v for k, v in counts.items() if v})


@dataclass(frozen=True)
class _ObservedLabels:
    panel: EvaluationPanel
    observed: np.ndarray
    row_codes: np.ndarray
    asset_codes: np.ndarray


def _observed_panel(group: _LabelGroup) -> _ObservedLabels:
    times = group.times
    starts = np.r_[0, np.flatnonzero(group.instruments[1:] != group.instruments[:-1]) + 1]
    assets = group.assets.take(group.instruments[starts])
    widths = np.diff(starts, append=len(times))
    width = widths[0]
    dense = len(assets) == 1 or (
        np.all(widths == width) and np.all(times.asi8.reshape(-1, width) == times.asi8[:width])
    )
    if len(assets) == 1:
        row = np.arange(width, dtype=np.int32 if width <= np.iinfo(np.int32).max else np.int64)
        col = np.broadcast_to(np.int8(0), row.shape)
    else:
        col = np.repeat(np.arange(len(assets)), widths)
        if dense:
            row = np.tile(np.arange(width), len(assets))
            times = times[:width]
        else:
            row, times = times.factorize(sort=True)
    index = pd.MultiIndex(levels=[times, assets], codes=[row, col],
                          names=['signal_timestamp', 'instrument_id'], verify_integrity=False)
    shape = (len(times), len(assets))
    valid = group.reasons < 0
    values = group.factor
    if dense:
        paired = np.ascontiguousarray(valid.reshape(len(assets), -1).T)
        factor = np.ascontiguousarray(values.reshape(len(assets), -1).T)
        returns = np.ascontiguousarray(group.returns.reshape(len(assets), -1).T)
        observed = np.ones(shape, dtype=bool)
    else:
        factor = np.zeros(shape, dtype=values.dtype)
        returns = np.full(shape, np.nan)
        factor[row, col] = values
        returns[row, col] = group.returns
        observed = np.zeros(shape, dtype=bool)
        observed[row, col] = True
        paired = np.zeros(shape, dtype=bool)
        paired[row, col] = valid
        if factor.dtype.kind not in 'iu':
            factor[~paired] = np.nan
        returns[~paired] = np.nan
    panel = EvaluationPanel(
        times, assets, factor, {group.horizon: returns}, {group.horizon: paired},
        index, pd.Index(['factor', group.horizon]),
    )
    return _ObservedLabels(panel, observed, row, col)


_TableStatistics = tuple[_LabelGroup, _ObservedLabels, TimeSeriesResult | CrossSectionResult]


def _quantile_meta(count: int, n_bins: int, occupied: int, requested: int, constant: bool) -> dict[str, Any]:
    actual = max(occupied, int(n_bins == 1)) if count >= 2 else 0
    if count < 2:
        status = 'insufficient_samples'
    elif n_bins == 0 and constant:
        status = 'constant_factor'
    elif actual < 2:
        status = 'unable_to_form_quantiles'
    else:
        status = 'reduced_quantiles' if actual < requested else 'ok'
    return dict(requested_quantiles=requested, actual_quantiles=actual, quantile_status=status,
                quantile_scope='full_sample_descriptive_only', descriptive_only=True)


def _data_identity(bars: pd.DataFrame | None, labels: pd.DataFrame) -> dict[str, Any]:
    frame = bars if bars is not None else labels
    return dict(instruments=sorted(frame.instrument_id.unique().tolist()),
                frequencies=sorted(frame.frequency.unique().tolist()),
                sources=sorted(frame.source.unique().tolist()) if 'source' in frame else [],
                data_versions=sorted(frame.data_version.unique().tolist()) if 'data_version' in frame else [],
                raw_coverage_start_utc=(bars.attrs.get('raw_coverage_start_utc', bars.timestamp_utc.min())
                                        if bars is not None else None),
                raw_coverage_end_utc=(bars.attrs.get('raw_coverage_end_utc', bars.timestamp_utc.max())
                                      if bars is not None else None))


class TimeSeriesEvaluator:
    """Compute each instrument/frequency/horizon independently, then aggregate ICs."""

    def __init__(self, *, analysis_mode: str | None = None, min_ic_samples: int = 3,
                 n_quantiles: int = 5) -> None:
        self.analysis_mode = _mode(analysis_mode, 'time_series')
        self.min_ic_samples = _integer(min_ic_samples, 'min_ic_samples')
        self.n_quantiles = _integer(n_quantiles, 'n_quantiles')

    def evaluate(self, per_bar_labels: pd.DataFrame,
                 factors: pd.DataFrame | None = None) -> EvaluationResult:
        prepared = _prepare(per_bar_labels, factors)
        return self._result(prepared, dict(per_bar_labels.attrs.get('calendar_identity', {})))

    def _result(self, prepared: _PreparedLabels, calendar_identity: dict[str, Any],
                tables: Iterable[_TableStatistics] | None = None) -> EvaluationResult:
        rows, quantiles = [], []
        ic_inference = {}
        groups = list(prepared.groups()) if tables is None else []
        if tables is not None:
            evaluated = [self._group_rows(*table) for table in tables]
            canonical_frame = prepared.canonical_frame()
        elif len(groups) > 1 and len(prepared.frame) >= _PARALLEL_LABEL_ROWS:
            with ThreadPoolExecutor(max_workers=min(4, len(groups))) as executor:
                pending = executor.map(self._evaluate_group, groups)
                # The table can be materialized while independent array kernels run.
                canonical_frame = prepared.canonical_frame()
                evaluated = list(pending)
        else:
            evaluated = [self._evaluate_group(group) for group in groups]
            canonical_frame = prepared.canonical_frame()
        for group_rows, group_quantiles, inference in evaluated:
            rows.extend(group_rows)
            quantiles.extend(group_quantiles)
            ic_inference.update(inference)
        rows.sort(key=lambda row: (row['instrument_id'], row['frequency'], row['horizon_bars']))
        quantiles.sort(key=lambda row: (row['instrument_id'], row['frequency'], row['horizon_bars'], row['quantile']))
        details = pd.DataFrame(rows, columns=TS_IC_COLUMNS)
        for (frequency, horizon), group in details.groupby(['frequency', 'horizon_bars'], sort=True):
            full = group.spearman_ic.dropna()
            sampled = group.non_overlapping_ic.dropna()
            counts = _count_summary(int(group.total_count.sum()), {
                name: sum(counts[name] for counts in group.exclusion_counts) for name in EXCLUSION_COLUMNS
            })
            rows.append(dict(scope='multi_instrument_summary', instrument_id='__ALL__',
                             frequency=frequency, horizon_bars=int(horizon), **counts,
                             mean_spearman_ic=float(full.mean()), median_spearman_ic=float(full.median()),
                             mean_non_overlapping_ic=float(sampled.mean()),
                             median_non_overlapping_ic=float(sampled.median()),
                             instrument_count=len(group), valid_instrument_count=len(full),
                             valid_non_overlapping_instrument_count=len(sampled),
                             ic_status='ok' if len(full) else 'no_valid_instruments'))
        return EvaluationResult(self.analysis_mode, canonical_frame, pd.DataFrame(rows, columns=TS_IC_COLUMNS),
                                pd.DataFrame(quantiles, columns=TS_QUANTILE_COLUMNS), _counts(prepared.reasons),
                                data_identity=prepared.identity(),
                                calendar_identity=calendar_identity,
                                ic_inference=ic_inference)

    def _evaluate_group(self, group: _LabelGroup) -> tuple[
        list[dict], list[dict], dict[ICInferenceKey, IntervalUsability],
    ]:
        observed = _observed_panel(group)
        panel = observed.panel
        statistics = EvaluationEngine('time_series', frequency='bar').evaluate(
            panel, expected_grid=observed.observed, signal_valid=panel.valid[group.horizon],
            min_samples=self.min_ic_samples, n_quantiles=self.n_quantiles, block_length=group.horizon,
            all_non_overlapping_phases=False, return_diagnostics=False,
        ).splits['full'].statistics
        return self._group_rows(group, observed, statistics)

    def _group_rows(self, group: _LabelGroup, observed: _ObservedLabels, statistics: TimeSeriesResult) -> tuple[
        list[dict], list[dict], dict[ICInferenceKey, IntervalUsability],
    ]:
        rows, quantiles, ic_inference = [], [], {}
        horizon = group.horizon
        panel, asset_codes = observed.panel, observed.asset_codes
        offsets = np.r_[0, np.flatnonzero(asset_codes[1:] != asset_codes[:-1]) + 1, len(asset_codes)]
        for column, instrument in enumerate(panel.assets):
            identity = dict(instrument_id=instrument, frequency=group.frequency, horizon_bars=horizon)
            instrument_stats = statistics.horizons[horizon].instruments[instrument]
            ic = instrument_stats.ic
            ic_inference[group.frequency, horizon, instrument] = evaluate_interval_usability(
                instrument_stats.bootstrap, point_estimate=ic.values[0], point_reason=ic.reasons[0],
            )
            phase = instrument_stats.non_overlapping_ic[0]
            buckets = [(bucket, stats) for bucket, stats in instrument_stats.returns_by_bucket.items()
                       if stats.n_samples]
            n_bins = statistics.quantiles.n_bins[column]
            asset_slice = slice(offsets[column], offsets[column + 1])
            constant = False
            if n_bins == 0 and ic.counts[0] >= 2:
                values = group.factor[asset_slice][group.reasons[asset_slice] < 0]
                constant = bool(np.all(values == values[0]))
            meta = _quantile_meta(ic.counts[0], n_bins, len(buckets), self.n_quantiles, constant)
            asset_counts = _counts(group.reasons[asset_slice])
            rows.append(dict(scope='per_instrument', **identity, **asset_counts,
                             spearman_ic=ic.values[0], ic_status=ic.reasons[0] or 'ok',
                             non_overlapping_ic=phase.values[0], non_overlapping_count=int(phase.counts[0]),
                             non_overlapping_status=phase.reasons[0] or 'ok', **meta))
            if meta['actual_quantiles'] >= 2:
                for number, (bucket, stats) in enumerate(buckets, 1):
                    members = statistics.quantiles.labels[:, column] == bucket
                    returns = panel.returns[horizon][members, column]
                    quantiles.append(dict(**identity, **meta, quantile=number, sample_count=stats.n_samples,
                                          mean_return_pct=stats.mean, median_return_pct=stats.p50,
                                          positive_ratio=float((returns > 0).mean())))
        return rows, quantiles, ic_inference


class BarCrossSectionalEvaluator:
    """Rank instruments within each timestamp, then aggregate equally over time."""

    def __init__(self, *, analysis_mode: str | None = None, n_quantiles: int = 5,
                 min_cross_section_size: int = 3) -> None:
        self.analysis_mode = _mode(analysis_mode, 'cross_sectional')
        self.n_quantiles = _integer(n_quantiles, 'n_quantiles')
        self.min_cross_section_size = _integer(min_cross_section_size, 'min_cross_section_size')

    def evaluate(self, per_bar_labels: pd.DataFrame,
                 factors: pd.DataFrame | None = None) -> EvaluationResult:
        prepared = _prepare(per_bar_labels, factors)
        return self._result(prepared, dict(per_bar_labels.attrs.get('calendar_identity', {})),
                            self._evaluate_groups(prepared))

    def _evaluate_groups(self, prepared: _PreparedLabels) -> Iterator[_TableStatistics]:
        engine = EvaluationEngine('cross_sectional', frequency='bar')
        for group in prepared.groups():
            observed = _observed_panel(group)
            panel = observed.panel
            statistics = engine.evaluate(
                panel, signal_valid=panel.valid[group.horizon],
                min_samples=self.min_cross_section_size, n_quantiles=self.n_quantiles, hac_lag=0,
                signal_diagnostics=False, return_diagnostics=False,
            ).splits['full'].statistics
            yield group, observed, statistics

    def _result(self, prepared: _PreparedLabels, calendar_identity: dict[str, Any],
                tables: Iterable[_TableStatistics]) -> EvaluationResult:
        sections, rows, aggregates = [], [], []
        ic_inference = {}
        for group, observed, statistics in tables:
            frequency, horizon = group.frequency, group.horizon
            panel, row_codes = observed.panel, observed.row_codes
            result = statistics.horizons[horizon]
            counts = result.ic.counts
            reasons = pd.Series(result.ic.reasons, index=panel.datetimes)
            cs_status = reasons.fillna("ok").replace({"insufficient_samples": "insufficient_cross_section"})
            statuses = cs_status.value_counts()
            n_bins = statistics.quantiles.n_bins
            occupied = result.bucket_counts > 0
            enough = counts >= self.min_cross_section_size
            actual = np.where(enough, np.maximum(occupied.sum(axis=1), n_bins == 1), 0)
            quantile_status = np.select(
                [~enough, (n_bins == 0) & reasons.eq("constant_factor"), actual < 2,
                 actual < self.n_quantiles],
                ["insufficient_samples", "constant_factor", "unable_to_form_quantiles", "reduced_quantiles"],
                default="ok",
            )
            total = np.bincount(row_codes, minlength=len(panel.datetimes))
            excluded = group.reasons >= 0
            reason_codes = group.reasons[excluded]
            by_reason = pd.DataFrame(
                np.bincount(row_codes[excluded] * len(EXCLUSION_COLUMNS) + reason_codes,
                            minlength=len(total) * len(EXCLUSION_COLUMNS)).reshape(len(total), -1),
                columns=EXCLUSION_COLUMNS,
            ).to_dict("records")
            sections.append(pd.DataFrame({
                "frequency": frequency, "horizon_bars": horizon, "signal_timestamp": panel.datetimes,
                "total_count": total, "valid_count": counts, "excluded_count": total - counts,
                "exclusion_counts": by_reason,
                "nonzero_exclusion_counts": [{k: v for k, v in row.items() if v} for row in by_reason],
                "valid_instruments": counts, "spearman_ic": result.ic.values, "cs_status": cs_status.to_numpy(),
                "requested_quantiles": self.n_quantiles, "actual_quantiles": actual,
                "quantile_status": quantile_status, "quantile_scope": "cross_sectional_time_aggregated",
                "descriptive_only": False,
            }, columns=CS_COLUMNS))
            # Group corresponding occupied buckets directly on the time axis.
            bucket_numbers = np.where(occupied & (actual >= 2)[:, None], occupied.cumsum(axis=1), 0)
            for number in range(1, int(bucket_numbers.max(initial=0)) + 1):
                row, bucket = np.nonzero(bucket_numbers == number)
                returns = pd.Series(result.bucket_returns[row, bucket], copy=False)
                quantile_counts = actual[row]
                aggregates.append(dict(
                    frequency=frequency, horizon_bars=horizon, quantile=number,
                    cross_section_count=len(row), total_sample_count=int(result.bucket_counts[row, bucket].sum()),
                    mean_return_pct=float(returns.mean()), median_return_pct=float(returns.median()),
                    positive_ratio=float((returns > 0).mean()), requested_quantiles=self.n_quantiles,
                    min_actual_quantiles=int(quantile_counts.min()), max_actual_quantiles=int(quantile_counts.max()),
                    quantile_scope='cross_sectional_time_aggregated', descriptive_only=False,
                    quantile_status='ok' if np.all(quantile_counts == self.n_quantiles) else 'reduced_quantiles',
                ))
            summary = result.ic_summary
            ic_inference[frequency, horizon, None] = evaluate_interval_usability(
                result.hac, point_estimate=summary.mean,
            )
            status = ('ok' if summary.n_valid else
                      statuses.index[0] if len(statuses) == 1 else 'no_valid_cross_sections')
            rows.append(dict(frequency=frequency, horizon_bars=horizon, **_counts(group.reasons),
                             mean_ic=summary.mean, median_ic=summary.median, std_ic=summary.sample_std,
                             icir=summary.icir, positive_ic_ratio=summary.positive_fraction,
                             total_cross_sections=len(counts), valid_cross_sections=summary.n_valid,
                             excluded_cross_sections=len(counts) - summary.n_valid,
                             insufficient_cross_section_count=int(statuses.get('insufficient_cross_section', 0)),
                             constant_factor_cross_section_count=int(statuses.get('constant_factor', 0)),
                             constant_return_cross_section_count=int(statuses.get('constant_return', 0)),
                             mean_valid_instruments=float(counts.mean()),
                             median_valid_instruments=float(np.median(counts)), ic_status=status))
        section_frame = pd.concat(sections, ignore_index=True) if sections else pd.DataFrame(columns=CS_COLUMNS)
        return EvaluationResult(self.analysis_mode, prepared.canonical_frame(), pd.DataFrame(rows, columns=CS_IC_COLUMNS),
                                pd.DataFrame(aggregates, columns=CS_QUANTILE_COLUMNS), _counts(prepared.reasons),
                                cross_sections=section_frame, data_identity=prepared.identity(),
                                calendar_identity=calendar_identity,
                                ic_inference=ic_inference)


def _evaluate_bar_groups(
    prepared: _PreparedLabels, evaluator: TimeSeriesEvaluator | BarCrossSectionalEvaluator, *,
    labeler: BarReturnLabeler, window: _SelectionWindow, options: dict[str, Any],
    family: BarTrialFamily | None, factor_id: str,
) -> EvaluationResult:
    selection_split, split = window
    if family is not None:
        family._validate_evaluation_window(window)
    if split is not None and prepared.frame.empty:
        raise ContractValidationError('Both train and oos require at least one scheduled signal')

    mode = evaluator.analysis_mode
    time_series = mode == 'time_series'
    engine = EvaluationEngine(mode, 'bar')
    groups: dict[tuple[str, str | None], list[_LabelGroup]] = {}
    for group in prepared.groups(by_instrument=time_series):
        instrument = group.assets[group.instruments[0]] if time_series else None
        groups.setdefault((group.frequency, instrument), []).append(group)
    tables: list[_TableStatistics] = []

    def evaluate_groups() -> Iterator[tuple[str, str | None, dict[str, SplitEvaluation]]]:
        keys = ['frequency', 'instrument_id'] if time_series else 'frequency'
        for key, labels in prepared.frame.groupby(keys):
            frequency, instrument = key if time_series else (key, None)
            panel, expected, signal_valid, clock, entries, exits = _bar_panel(
                labels, labeler, frequency, scheduled_exits=split is not None, by_availability=not time_series,
            )
            observed = [(group, _observed_panel(group)) for group in groups[frequency, instrument]]
            # Start-time sections and close-time sections share point IC only when
            # the clocks define a one-to-one mapping, including truncated session Bars.
            close_by_start = None
            if not time_series:
                clocks = labels[['signal_timestamp', 'signal_available_at']].drop_duplicates()
                if clocks.signal_timestamp.is_unique and clocks.signal_available_at.is_unique:
                    close_by_start = clocks.set_index('signal_timestamp').signal_available_at
            samples = {
                group.horizon: DescriptiveSample(
                    table.panel, table.observed,
                    None if close_by_start is None else pd.DatetimeIndex(
                        close_by_start.reindex(table.panel.datetimes)),
                )
                for group, table in observed
            }
            evaluation = engine.evaluate(
                panel, split=split, expected_grid=expected, signal_valid=signal_valid,
                signal_available_at=clock, entry_timestamps=entries, exit_timestamps=exits,
                descriptive_samples=samples,
                **options,
            )
            tables.extend((group, table, evaluation.descriptive[group.horizon]) for group, table in observed)
            yield frequency, instrument, evaluation.splits

    reports = build_bar_splits(evaluate_groups(), horizons=labeler.horizons, mode=mode, split=split)
    result = evaluator._result(prepared, dict(prepared.frame.attrs.get('calendar_identity', {})), tables)
    result.splits = reports
    if family is not None:
        _record_trials(family, reports[selection_split]['horizons'],
                       keys=tuple((factor_id, h) for h in labeler.horizons), mode=mode, window=window)
        result.trial_ledger = family.report()
    return result


def _calendar_spec(calendar: SessionCalendar) -> dict[str, Any]:
    return dict(
        calendar_id=calendar.calendar_id, calendar_hash=calendar.calendar_hash,
        schedule=dict(start_date=calendar.start.date().isoformat(), end_date=calendar.end.date().isoformat(),
                      rth_reference_calendar=calendar.rth_reference_calendar),
    )


def _calendar_mapping_hash(calendars: Mapping[str, Mapping[str, Any]]) -> str:
    hashes = {spec['calendar_hash'] for spec in calendars.values()}
    if len(hashes) == 1:
        return next(iter(hashes))
    identities = {key: {name: spec[name] for name in ('calendar_id', 'calendar_hash')}
                  for key, spec in calendars.items()}
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


def _prepare_bar_evaluation(
    labeler: BarReturnLabeler | None, *, analysis_mode: str | None, n_quantiles: int,
    min_ic_samples: int, min_cross_section_size: int,
    split: SplitConfig | Mapping[str, str | pd.Timestamp] | None = None,
    bootstrap: Mapping[str, Any] | None = None, hac_lag: int | Literal['auto'] | None = None,
    direction: int | None = None, standardization: Standardization | None = None,
    calibration_scenario: str | None = None, trial_selection: Mapping[str, Any] | None = None,
    factor_id: str = 'factor_value', trial_family: BarTrialFamily | None = None,
    diagnostics: bool = False,
) -> tuple[TimeSeriesEvaluator | BarCrossSectionalEvaluator, _SelectionWindow, BarTrialFamily | None]:
    mode = _mode(analysis_mode)
    evaluator = (TimeSeriesEvaluator(analysis_mode=mode, n_quantiles=n_quantiles,
                                     min_ic_samples=min_ic_samples) if mode == 'time_series' else
                 BarCrossSectionalEvaluator(analysis_mode=mode, n_quantiles=n_quantiles,
                                             min_cross_section_size=min_cross_section_size))
    if not isinstance(labeler, BarReturnLabeler):
        raise ContractValidationError('labeler must be a BarReturnLabeler')
    if not isinstance(diagnostics, bool) or (diagnostics and mode != 'time_series'):
        raise ContractValidationError('diagnostics must be a boolean and requires time_series when enabled')
    for name, value, supported in (
        ('bootstrap', bootstrap, {'block_length', 'n_boot', 'seed', 'min_valid_replicates', 'alpha'}),
        ('trial_selection', trial_selection, {'method', 'alpha'}),
    ):
        if value is not None and (not isinstance(value, Mapping) or value.keys() - supported):
            raise ContractValidationError(f'{name} must be a mapping with keys from {sorted(supported)}')
    if bootstrap is not None and mode != 'time_series':
        raise ContractValidationError('bootstrap requires time_series analysis')
    if hac_lag is not None and mode != 'cross_sectional':
        raise ContractValidationError('hac_lag requires cross_sectional analysis')
    if standardization is not None and (mode != 'time_series' or split is None):
        raise ContractValidationError('standardization requires a time_series train/oos split')
    if isinstance(split, Mapping):
        split = SplitConfig(**split)
    if split is not None and not isinstance(split, SplitConfig):
        raise ContractValidationError('split must specify train_end and oos_start')
    window = _selection_window(split)
    _validate_standardization(standardization)
    if direction is not None:
        _validate_direction(direction)
    if bootstrap is not None:
        block_length, _, _, _ = _validate_bootstrap_options(**bootstrap)
        for horizon in labeler.horizons:
            _validate_block_length_for_horizon(block_length, horizon)
    if hac_lag is not None:
        _validate_hac_lag(hac_lag)
    if trial_family is not None:
        if not isinstance(trial_family, BarTrialFamily):
            raise ContractValidationError('trial_family must be a BarTrialFamily')
        if trial_selection is not None:
            raise ContractValidationError('Configure method and alpha on trial_family for a shared selection')
        trial_family._validate_registration((factor_id, h) for h in labeler.horizons)
    family = trial_family
    if family is None and any(value is not None for value in (
        split, bootstrap, hac_lag, direction, standardization, calibration_scenario, trial_selection,
    )):
        family = BarTrialFamily(((factor_id, h) for h in labeler.horizons), **(trial_selection or {}))
    return evaluator, window, family


def evaluate_bars(bars: pd.DataFrame, factors: pd.DataFrame | None = None,
                  labeler: BarReturnLabeler | None = None, *, analysis_mode: str | None = None,
                  n_quantiles: int = 5, min_ic_samples: int = 3,
                  min_cross_section_size: int = 3,
                  split: SplitConfig | Mapping[str, str | pd.Timestamp] | None = None,
                  bootstrap: Mapping[str, Any] | None = None,
                  hac_lag: int | Literal['auto'] | None = None, direction: int | None = None,
                  standardization: Standardization | None = None,
                  calibration_scenario: str | None = None,
                  trial_selection: Mapping[str, Any] | None = None,
                  factor_id: str = 'factor_value',
                  trial_family: BarTrialFamily | None = None,
                  diagnostics: bool = False) -> EvaluationResult:
    """Label and evaluate raw Bars; preserve raw coverage and calendar provenance.

    For precomputed labels use the matching evaluator's ``evaluate`` method.
    ``analysis_mode`` is mandatory even for a one-instrument input. Inference
    options or diagnostics add calendar-grid reports alongside the descriptive
    tables. Split selection uses training evidence. Calendar-grid inference
    defaults to automatic HAC lags or Bootstrap block lengths by horizon.
    Pass an integer to use a fixed window.
    Pass a preregistered trial_family to record this factor's configured horizons
    in a shared selection. The first successful evaluation binds the family's
    selection window. Each result carries a snapshot of family progress. Enable
    diagnostics to retain time-series statistics when no inference options are set.
    """
    evaluator, window, family = _prepare_bar_evaluation(
        labeler, analysis_mode=analysis_mode, n_quantiles=n_quantiles, min_ic_samples=min_ic_samples,
        min_cross_section_size=min_cross_section_size, split=split, bootstrap=bootstrap,
        hac_lag=hac_lag, direction=direction, standardization=standardization,
        calibration_scenario=calibration_scenario, trial_selection=trial_selection,
        factor_id=factor_id, trial_family=trial_family, diagnostics=diagnostics,
    )
    mode = evaluator.analysis_mode
    _, split = window
    frame = validate_bar_frame(bars)
    labeling = labeler
    if split is not None and not labeler.include_scheduled_exit:
        labeling = copy(labeler)
        labeling.include_scheduled_exit = True
    labels = labeling.label(frame, factors)
    if factors is None:
        values = frame[['timestamp_utc', 'instrument_id', 'frequency', 'factor_value']].rename(
            columns={'timestamp_utc': 'signal_timestamp'})
        if 'is_complete' in frame:
            incomplete = ~frame.is_complete.eq(True).to_numpy(dtype=bool, na_value=False)
            values['factor_value'] = _mask_unavailable_factor_values(values.factor_value, incomplete)
        labels = labels.merge(values, on=['signal_timestamp', 'instrument_id', 'frequency'],
                              how='left', validate='many_to_one')
    if family is not None or diagnostics:
        options = dict(n_quantiles=n_quantiles, direction=1 if direction is None else direction,
                       standardization=standardization, calibration_scenario=calibration_scenario)
        if mode == 'time_series':
            options.update(min_samples=min_ic_samples, **(bootstrap or {}))
        else:
            options.update(min_samples=min_cross_section_size, hac_lag=hac_lag,
                           signal_diagnostics=False)
        result = _evaluate_bar_groups(
            _prepare(labels, factors), evaluator, labeler=labeler, window=window, family=family,
            factor_id=factor_id, options=options,
        )
    else:
        result = evaluator.evaluate(labels, factors)
    result.data_identity = _data_identity(frame, labels)
    if isinstance(labeler.calendar, Mapping):
        calendars = {key: _calendar_spec(cal) for key, cal in sorted(labeler.calendar.items())}
        result.calendar_identity = dict(calendar_id='per_instrument', calendar_hash=_calendar_mapping_hash(calendars),
                                        calendars=calendars)
    else:
        result.calendar_identity = _calendar_spec(labeler.calendar)
    result.calendar_identity.update(view=labeler.view, boundary_policy=labeler.boundary_policy,
                                    bucket_strategy=labeler.bucket_strategy)
    return result
