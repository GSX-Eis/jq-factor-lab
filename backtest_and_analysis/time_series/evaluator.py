"""Explicit time-series and same-timestamp cross-sectional Bar evaluation.

IC uses average ranks. Full-sample time-series quantiles are descriptive;
overlapping returns never imply unadjusted statistical significance.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    EXCLUSION_PRIORITY, ContractValidationError, DuplicateKeyError,
    _duration, _frame_columns, _numeric_column, _text_column,
    validate_bar_frame, validate_factor_frame,
)
from .labels import BarReturnLabeler, LABEL_COLUMNS


RESULT_SCHEMA_VERSION = 'bar_evaluation.v1'
LABEL_KEY = ['instrument_id', 'frequency', 'horizon_bars', 'signal_timestamp']
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


@dataclass
class EvaluationResult:
    analysis_mode: str
    per_bar_labels: pd.DataFrame
    ic_by_horizon: pd.DataFrame
    quantile_stats: pd.DataFrame
    exclusion_summary: dict[str, Any]
    cross_sections: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CS_COLUMNS))
    data_identity: dict[str, Any] = field(default_factory=dict)
    calendar_identity: dict[str, Any] = field(default_factory=dict)


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


def _prepare(labels: pd.DataFrame, factors: pd.DataFrame | None) -> pd.DataFrame:
    _frame_columns(labels, LABEL_COLUMNS)
    frame = labels.copy().reset_index(drop=True)
    for name in ('signal_timestamp', 'signal_available_at'):
        times = frame[name]
        if (not isinstance(times.dtype, pd.DatetimeTZDtype)
                or str(times.dt.tz) not in ('UTC', 'Etc/UTC') or times.isna().any()):
            raise ContractValidationError(f'{name} must have UTC datetime dtype and no NaT')
    for name in ('instrument_id', 'frequency'):
        _text_column(frame[name])
    if factors is not None:
        values = validate_factor_frame(factors)[['timestamp_utc', 'instrument_id', 'factor_value']]
        frame = frame.drop(columns='factor_value', errors='ignore').merge(
            values.rename(columns={'timestamp_utc': 'signal_timestamp'}),
            on=['signal_timestamp', 'instrument_id'], how='left', validate='many_to_one')
    _frame_columns(frame, ('factor_value',))
    # The labeler's empty contract table has object value columns; normalize only empties.
    if frame.empty:
        for name in ('factor_value', 'return_pct'):
            frame[name] = frame[name].astype('float64')
    for frequency in frame.frequency.unique():
        _duration(frequency)
    for name in ('horizon_bars', 'entry_offset_bars'):
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1
               for v in frame[name]):
            raise ContractValidationError(f'{name} must contain positive integers')
    if frame.duplicated(LABEL_KEY).any():
        raise DuplicateKeyError(f'duplicate label keys: {LABEL_KEY}')
    if not frame.status.isin(['valid', 'excluded']).all():
        raise ContractValidationError('status must be valid or excluded')
    valid = frame.status.eq('valid')
    if (frame.loc[valid, 'exclusion_reason'].notna().any()
            or not frame.loc[~valid, 'exclusion_reason'].isin(EXCLUSION_PRIORITY).all()):
        raise ContractValidationError('status and exclusion_reason disagree')
    values = _numeric_column(frame.factor_value)
    returns = _numeric_column(frame.return_pct)
    if np.isinf(values).any() or not np.isfinite(values[valid]).all():
        raise ContractValidationError('valid labels require finite factor_value; excluded may have NaN')
    if not np.isfinite(returns[valid]).all():
        raise ContractValidationError('valid labels require finite return_pct')
    return frame.loc[:, [*LABEL_COLUMNS, 'factor_value']].sort_values(
        LABEL_KEY, kind='stable').reset_index(drop=True)


def _counts(frame: pd.DataFrame) -> dict[str, Any]:
    valid = int(frame.status.eq('valid').sum())
    reasons = frame.exclusion_reason.value_counts()
    counts = {reason.value: int(reasons.get(reason, 0)) for reason in EXCLUSION_PRIORITY}
    return dict(total_count=len(frame), valid_count=valid, excluded_count=len(frame) - valid,
                exclusion_counts=counts, nonzero_exclusion_counts={k: v for k, v in counts.items() if v})


def _ic(valid: pd.DataFrame, minimum: int, insufficient: str = 'insufficient_samples') -> tuple[float, str]:
    if len(valid) < minimum:
        return np.nan, insufficient
    if valid.factor_value.nunique() <= 1:
        return np.nan, 'constant_factor'
    if valid.return_pct.nunique() <= 1:
        return np.nan, 'constant_return'
    # Pearson correlation of average ranks is Spearman, without p-values or SciPy warnings.
    return float(valid.factor_value.rank(method='average').corr(
        valid.return_pct.rank(method='average'))), 'ok'


def _quantiles(valid: pd.DataFrame, requested: int, *, cross_sectional: bool = False,
               minimum: int = 2) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = dict(requested_quantiles=requested, actual_quantiles=0,
                quantile_status='insufficient_samples',
                quantile_scope=('cross_sectional_time_aggregated' if cross_sectional
                                else 'full_sample_descriptive_only'),
                descriptive_only=not cross_sectional)
    if len(valid) < minimum:
        return meta, []
    if valid.factor_value.nunique() <= 1:
        meta['quantile_status'] = 'constant_factor'
        return meta, []
    bins = pd.qcut(valid.factor_value, q=requested, labels=False, duplicates='drop')
    actual = int(bins.nunique())
    meta.update(actual_quantiles=actual, quantile_status=(
        'unable_to_form_quantiles' if actual < 2 else
        'reduced_quantiles' if actual < requested else 'ok'))
    rows = []
    if actual >= 2:
        # Empty intervals are not actual groups. Renumber occupied bins consecutively.
        for quantile, (_, group) in enumerate(valid.groupby(bins, sort=True), 1):
            returns = group.return_pct
            rows.append(dict(quantile=quantile, **meta, sample_count=len(group),
                             mean_return_pct=float(returns.mean()),
                             median_return_pct=float(returns.median()),
                             positive_ratio=float(returns.gt(0).mean())))
    return meta, rows


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
        frame = _prepare(per_bar_labels, factors)
        rows, quantiles = [], []
        for (instrument, frequency, horizon), group in frame.groupby(
                ['instrument_id', 'frequency', 'horizon_bars'], sort=True):
            identity = dict(instrument_id=instrument, frequency=frequency, horizon_bars=int(horizon))
            valid = group.loc[group.status.eq('valid')]
            ic, status = _ic(valid, self.min_ic_samples)
            sampled = group.sort_values('signal_timestamp', kind='stable').iloc[::int(horizon)]
            sampled = sampled.loc[sampled.status.eq('valid')]
            nonoverlap, nonoverlap_status = _ic(sampled, self.min_ic_samples)
            meta, groups = _quantiles(valid, self.n_quantiles)
            rows.append(dict(scope='per_instrument', **identity, **_counts(group),
                             spearman_ic=ic, ic_status=status, non_overlapping_ic=nonoverlap,
                             non_overlapping_count=len(sampled), non_overlapping_status=nonoverlap_status,
                             **meta))
            quantiles.extend(dict(**identity, **q) for q in groups)
        details = pd.DataFrame(rows, columns=TS_IC_COLUMNS)
        for (frequency, horizon), group in details.groupby(['frequency', 'horizon_bars'], sort=True):
            candidates = frame.loc[frame.frequency.eq(frequency) & frame.horizon_bars.eq(horizon)]
            full = group.spearman_ic.dropna()
            sampled = group.non_overlapping_ic.dropna()
            rows.append(dict(scope='multi_instrument_summary', instrument_id='__ALL__',
                             frequency=frequency, horizon_bars=int(horizon), **_counts(candidates),
                             mean_spearman_ic=float(full.mean()), median_spearman_ic=float(full.median()),
                             mean_non_overlapping_ic=float(sampled.mean()),
                             median_non_overlapping_ic=float(sampled.median()),
                             instrument_count=len(group), valid_instrument_count=len(full),
                             valid_non_overlapping_instrument_count=len(sampled),
                             ic_status='ok' if len(full) else 'no_valid_instruments'))
        return EvaluationResult(self.analysis_mode, frame, pd.DataFrame(rows, columns=TS_IC_COLUMNS),
                                pd.DataFrame(quantiles, columns=TS_QUANTILE_COLUMNS), _counts(frame),
                                data_identity=_data_identity(None, frame),
                                calendar_identity=dict(per_bar_labels.attrs.get('calendar_identity', {})))


class BarCrossSectionalEvaluator:
    """Rank instruments within each timestamp, then aggregate equally over time."""

    def __init__(self, *, analysis_mode: str | None = None, n_quantiles: int = 5,
                 min_cross_section_size: int = 3) -> None:
        self.analysis_mode = _mode(analysis_mode, 'cross_sectional')
        self.n_quantiles = _integer(n_quantiles, 'n_quantiles')
        self.min_cross_section_size = _integer(min_cross_section_size, 'min_cross_section_size')

    def evaluate(self, per_bar_labels: pd.DataFrame,
                 factors: pd.DataFrame | None = None) -> EvaluationResult:
        frame = _prepare(per_bar_labels, factors)
        sections, quantiles, rows, aggregates = [], [], [], []
        for (frequency, horizon, timestamp), group in frame.groupby(
                ['frequency', 'horizon_bars', 'signal_timestamp'], sort=True):
            identity = dict(frequency=frequency, horizon_bars=int(horizon), signal_timestamp=timestamp)
            valid = group.loc[group.status.eq('valid')]
            ic, status = _ic(valid, self.min_cross_section_size, 'insufficient_cross_section')
            meta, groups = _quantiles(valid, self.n_quantiles, cross_sectional=True,
                                      minimum=self.min_cross_section_size)
            sections.append(dict(**identity, **_counts(group), valid_instruments=len(valid),
                                 spearman_ic=ic, cs_status=status, **meta))
            quantiles.extend(dict(**identity, **q) for q in groups)
        section_frame = pd.DataFrame(sections, columns=CS_COLUMNS)
        for (frequency, horizon), group in section_frame.groupby(['frequency', 'horizon_bars'], sort=True):
            candidates = frame.loc[frame.frequency.eq(frequency) & frame.horizon_bars.eq(horizon)]
            values = group.spearman_ic.dropna()
            std = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
            statuses = group.cs_status.value_counts()
            status = 'ok' if len(values) else (
                group.cs_status.iloc[0] if len(statuses) == 1 else 'no_valid_cross_sections')
            rows.append(dict(frequency=frequency, horizon_bars=int(horizon), **_counts(candidates),
                             mean_ic=float(values.mean()), median_ic=float(values.median()), std_ic=std,
                             icir=float(values.mean() / std) if std > 0 else np.nan,
                             positive_ic_ratio=float(values.gt(0).mean()) if len(values) else np.nan,
                             total_cross_sections=len(group), valid_cross_sections=len(values),
                             excluded_cross_sections=len(group) - len(values),
                             insufficient_cross_section_count=int(statuses.get('insufficient_cross_section', 0)),
                             constant_factor_cross_section_count=int(statuses.get('constant_factor', 0)),
                             constant_return_cross_section_count=int(statuses.get('constant_return', 0)),
                             mean_valid_instruments=float(group.valid_instruments.mean()),
                             median_valid_instruments=float(group.valid_instruments.median()), ic_status=status))
        qframe = pd.DataFrame(quantiles)
        if not qframe.empty:
            for (frequency, horizon, quantile), group in qframe.groupby(
                    ['frequency', 'horizon_bars', 'quantile'], sort=True):
                # Equal timestamp weight, independent of each section's instrument count.
                returns = group.mean_return_pct
                aggregates.append(dict(frequency=frequency, horizon_bars=int(horizon), quantile=int(quantile),
                    cross_section_count=len(group), total_sample_count=int(group.sample_count.sum()),
                    mean_return_pct=float(returns.mean()), median_return_pct=float(returns.median()),
                    positive_ratio=float(returns.gt(0).mean()), requested_quantiles=self.n_quantiles,
                    min_actual_quantiles=int(group.actual_quantiles.min()),
                    max_actual_quantiles=int(group.actual_quantiles.max()),
                    quantile_scope='cross_sectional_time_aggregated', descriptive_only=False,
                    quantile_status='ok' if group.actual_quantiles.eq(self.n_quantiles).all() else 'reduced_quantiles'))
        return EvaluationResult(self.analysis_mode, frame, pd.DataFrame(rows, columns=CS_IC_COLUMNS),
                                pd.DataFrame(aggregates, columns=CS_QUANTILE_COLUMNS), _counts(frame),
                                cross_sections=section_frame, data_identity=_data_identity(None, frame),
                                calendar_identity=dict(per_bar_labels.attrs.get('calendar_identity', {})))


def evaluate_bars(bars: pd.DataFrame, factors: pd.DataFrame | None = None,
                  labeler: BarReturnLabeler | None = None, *, analysis_mode: str | None = None,
                  n_quantiles: int = 5, min_ic_samples: int = 3,
                  min_cross_section_size: int = 3) -> EvaluationResult:
    """Label and evaluate raw Bars; preserve raw coverage and calendar provenance.

    For precomputed labels use the matching evaluator's ``evaluate`` method.
    ``analysis_mode`` is mandatory even for a one-instrument input.
    """
    mode = _mode(analysis_mode)
    evaluator = (TimeSeriesEvaluator(analysis_mode=mode, n_quantiles=n_quantiles,
                                     min_ic_samples=min_ic_samples) if mode == 'time_series' else
                 BarCrossSectionalEvaluator(analysis_mode=mode, n_quantiles=n_quantiles,
                                             min_cross_section_size=min_cross_section_size))
    if not isinstance(labeler, BarReturnLabeler):
        raise ContractValidationError('labeler must be a BarReturnLabeler')
    frame = validate_bar_frame(bars)
    labels = labeler.label(frame, factors)
    if factors is None:
        values = frame[['timestamp_utc', 'instrument_id', 'frequency', 'factor_value']].rename(
            columns={'timestamp_utc': 'signal_timestamp'})
        labels = labels.merge(values, on=['signal_timestamp', 'instrument_id', 'frequency'],
                              how='left', validate='many_to_one')
    result = evaluator.evaluate(labels, factors)
    result.data_identity = _data_identity(frame, labels)
    if isinstance(labeler.calendar, Mapping):
        calendars = {key: dict(calendar_id=cal.calendar_id, calendar_hash=cal.calendar_hash)
                     for key, cal in sorted(labeler.calendar.items())}
        hashes = {cal['calendar_hash'] for cal in calendars.values()}
        digest = (next(iter(hashes)) if len(hashes) == 1 else
                  hashlib.sha256(json.dumps(calendars, sort_keys=True).encode()).hexdigest())
        result.calendar_identity = dict(calendar_id='per_instrument', calendar_hash=digest, calendars=calendars)
    else:
        result.calendar_identity = dict(calendar_id=labeler.calendar.calendar_id,
                                        calendar_hash=labeler.calendar.calendar_hash)
    result.calendar_identity.update(view=labeler.view, boundary_policy=labeler.boundary_policy)
    return result
