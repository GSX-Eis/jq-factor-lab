"""Calendar-grid inputs and the evaluation.v2 Bar result contract."""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import asdict, dataclass
from math import erfc, sqrt
from numbers import Real
from threading import Lock
from typing import Any, Literal

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation import EvaluationEngine, evaluate_interval_usability
from backtest_and_analysis.evaluation.calibration import Correction, Trial, TrialKey, TrialLedger
from backtest_and_analysis.evaluation.kernels import CorrelationResult
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.split import SplitConfig
from backtest_and_analysis.evaluation.time_series import summarize_time_series_horizon
from backtest_and_analysis.evaluation.types import CrossSectionResult, TimeSeriesInstrument, TimeSeriesResult

from .contracts import ContractValidationError, _numeric_column
from .labels import BarReturnLabeler, _grid

EVALUATION_V2_SCHEMA_VERSION = 'evaluation.v2'
EVALUATION_V2_FILENAME = 'evaluation_v2.json'
_SelectionWindow = tuple[Literal['full', 'train'], SplitConfig | None]


def _selection_window(split: SplitConfig | None) -> _SelectionWindow:
    if split is None:
        return 'full', None
    train_end, oos_start = pd.Timestamp(split.train_end), pd.Timestamp(split.oos_start)
    if train_end.tzinfo is None or oos_start.tzinfo is None:
        raise ContractValidationError('Use timezone-aware split bounds for Bar evaluation')
    return 'train', SplitConfig(train_end.tz_convert('UTC'), oos_start.tz_convert('UTC'))


@dataclass(frozen=True)
class BarEvaluationV2:
    """Inference and candidate evidence carried alongside the frozen Bar tables."""

    splits: dict[str, dict[str, Any]]
    trial_ledger: dict[str, Any]
    schema_version: str = EVALUATION_V2_SCHEMA_VERSION
    frequency: str = 'bar'
    return_unit: str = 'percent'


class BarTrialFamily:
    """One preregistered factor/horizon family, reusable across factor evaluations."""

    def __init__(self, candidates: Iterable[TrialKey], *, method: Correction = 'holm',
                 alpha: float = 0.05) -> None:
        if method not in ('holm', 'bh'):
            raise ContractValidationError("trial_selection.method must be 'holm' or 'bh'")
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha < 1:
            raise ContractValidationError('trial_selection.alpha must be between 0 and 1')
        self.candidates = tuple(candidates)
        self._ledger = TrialLedger(self.candidates)
        self.method = method
        self.alpha = float(alpha)
        self._reasons: dict[TrialKey, str | None] = {}
        self._window: _SelectionWindow | None = None
        self._lock = Lock()

    def record(self, key: TrialKey, trial: Trial, *, reason: str | None = None) -> None:
        """Record external trials in a family with no evaluation window metadata."""
        self._record_trials({key: (trial, reason)})

    def _validate_registration(self, keys: Iterable[TrialKey]) -> None:
        registered = set(self.candidates)
        for key in keys:
            if key not in registered:
                raise ContractValidationError(f'Unregistered candidate {key!r}')

    def _require_same_window(self, window: _SelectionWindow | None, *, has_records: bool) -> None:
        if has_records and self._window != window:
            expected = self._window or 'manual (unspecified)'
            actual = window or 'manual (unspecified)'
            raise ContractValidationError(
                f'Trial family selection window mismatch: expected {expected!r}, got {actual!r}',
            )

    def _validate_evaluation_window(self, window: _SelectionWindow) -> None:
        with self._lock:
            self._require_same_window(window, has_records=bool(self._ledger.records))

    def _record_trials(self, trials: Mapping[TrialKey, tuple[Trial, str | None]], *,
                       window: _SelectionWindow | None = None) -> None:
        self._validate_registration(trials)
        with self._lock:
            records = self._ledger.records
            self._require_same_window(window, has_records=bool(records))
            for key, (trial, _) in trials.items():
                if key in records:
                    raise ContractValidationError(f'Trial already recorded for {key!r}')
                if not isinstance(trial, Trial):
                    raise TypeError('trial must be a Trial')
            for key, (trial, reason) in trials.items():
                self._ledger.record(key, trial)
                self._reasons[key] = (reason or 'raw_p_value_unavailable') if trial.p_value is None else None
            if trials and not records:
                self._window = window

    def report(self) -> dict[str, Any]:
        """Snapshot the registered family and adjust once all raw p values are available."""
        with self._lock:
            records = self._ledger.records
            pending = [key for key in self.candidates if key not in records]
            missing = [key for key in self.candidates if key in records and records[key].p_value is None]
            selection = None if pending or missing else self._ledger.select_best(method=self.method, alpha=self.alpha)
            reasons = self._reasons.copy()
            window = self._window
        adjusted = {row.key: row.adjusted_p_value for row in selection.trials} if selection else {}
        return dict(
            method=self.method, alpha=self.alpha, attempt_count=len(self.candidates),
            selection_complete=selection is not None, best=selection.best if selection else None,
            pending_candidates=pending, missing_p_value_candidates=missing,
            **({'split': window[0]} if window is not None else {}),
            candidates=[
                dict(factor_id=key[0], horizon=key[1],
                     **(asdict(records[key]) if key in records else
                        dict(statistic=None, p_value=None, n_effective=None, interval=None)),
                     reason=reasons.get(key, 'not_evaluated'), adjusted_p_value=adjusted.get(key))
                for key in self.candidates
            ],
        )


def _bar_panel(labels: pd.DataFrame, labeler: BarReturnLabeler, frequency: str, *,
               scheduled_exits: bool = False, by_availability: bool = False) -> tuple[
                   EvaluationPanel, np.ndarray, np.ndarray, pd.DatetimeIndex,
                   dict[int, np.ndarray], dict[int, np.ndarray],
               ]:
    """Use Bar starts for single-instrument phases and availability for cross-sections."""
    streams = {}
    grid = pd.DatetimeIndex([], tz='UTC')
    for instrument, group in labels.groupby('instrument_id', sort=True):
        calendar = labeler.calendar[instrument] if isinstance(labeler.calendar, Mapping) else labeler.calendar
        first, last = group.signal_timestamp.min(), group.signal_timestamp.max()
        starts, ends, _ = _grid(calendar, frequency, first, last, labeler.view, labeler.bucket_strategy)
        keep = (starts >= first) & (starts <= last)
        starts, ends = starts[keep], ends[keep]
        signals = group.drop_duplicates('signal_timestamp').set_index('signal_timestamp')
        positions = starts.get_indexer(signals.index)
        if (positions < 0).any() or not ends.take(positions).equals(pd.DatetimeIndex(signals.signal_available_at)):
            raise ContractValidationError('Bar signal_available_at does not match the calendar grid')
        axis = ends if by_availability else starts
        values = signals.factor_value
        if by_availability:
            values = values.set_axis(ends.take(positions))
        streams[instrument] = (group, values, axis, ends)
        grid = grid.union(axis)

    assets = pd.Index(list(streams))
    shape = (len(grid), len(assets))
    factors = []
    expected = np.zeros(shape, dtype=bool)
    signal_valid = np.zeros(shape, dtype=bool)
    available = np.full(len(grid), np.iinfo('int64').min, dtype='int64')
    returns = {int(h): np.full(shape, np.nan) for h in sorted(labels.horizon_bars.unique())}
    valid = {h: np.zeros(shape, dtype=bool) for h in returns}
    entries = {h: np.full(shape, np.datetime64('NaT', 'ns')) for h in returns}
    exits = {h: values.copy() for h, values in entries.items()}
    exit_field = 'scheduled_exit_timestamp' if scheduled_exits else 'exit_timestamp'
    row_field = 'signal_available_at' if by_availability else 'signal_timestamp'
    for col, (group, values, axis, ends) in enumerate(streams.values()):
        rows = grid.get_indexer(axis)
        expected[rows, col] = True
        available[rows] = ends.as_unit('ns').asi8
        signal_valid[:, col] = values.notna().reindex(grid, fill_value=False).to_numpy()
        if pd.api.types.is_integer_dtype(values.dtype):
            dtype = getattr(values.dtype, 'numpy_dtype', values.dtype)
            factors.append(values.reindex(grid, fill_value=0).to_numpy(dtype=dtype, na_value=0))
        else:
            factors.append(_numeric_column(values.reindex(grid)))
        for h, rows_at_h in group.groupby('horizon_bars'):
            rows = grid.get_indexer(rows_at_h[row_field])
            returns[h][rows, col] = rows_at_h.return_pct.to_numpy(dtype=float)
            valid[h][rows, col] = rows_at_h.status.eq('valid').to_numpy(dtype=bool)
            entries[h][rows, col] = rows_at_h.entry_timestamp.to_numpy(dtype='datetime64[ns]')
            exits[h][rows, col] = rows_at_h[exit_field].to_numpy(dtype='datetime64[ns]')

    panel = EvaluationPanel(grid, assets, np.column_stack(factors), returns, valid,
                            pd.MultiIndex.from_product([grid, assets]), pd.Index(['factor', *returns]))
    return panel, expected, signal_valid, pd.to_datetime(available, utc=True), entries, exits


def _ic(result: CorrelationResult) -> dict[str, Any]:
    return dict(sample_count=int(result.counts[0]), spearman_ic=float(result.values[0]),
                reason=result.reasons[0])


def _time_series_rows(stats: TimeSeriesResult, instrument: str, frequency: str) -> list[dict[str, Any]]:
    rows = []
    for h, horizon in stats.horizons.items():
        row = horizon.instruments[instrument]
        ic = _ic(row.spearman_ic)
        usability = evaluate_interval_usability(
            row.bootstrap, point_estimate=ic['spearman_ic'], point_reason=ic['reason'],
        )
        rows.append(dict(
            instrument_id=instrument, frequency=frequency, horizon_bars=h,
            grid_start_utc=stats.datetimes[0], grid_end_utc=stats.datetimes[-1],
            intervals_verified=stats.intervals_verified, ic=ic,
            bootstrap=asdict(row.bootstrap), interval_usability=asdict(usability),
            direction=asdict(row.direction), return_distribution=asdict(row.returns),
            quantile_reason=stats.quantile_reasons[0],
            buckets={q: dict(direction=asdict(row.direction_by_bucket[q]), return_distribution=asdict(dist))
                     for q, dist in row.returns_by_bucket.items()},
            non_overlapping_ic={offset: _ic(value) for offset, value in row.non_overlapping_ic.items()},
        ))
    return rows


def _cross_section_rows(stats: CrossSectionResult, frequency: str) -> list[dict[str, Any]]:
    rows = []
    for h, horizon in stats.horizons.items():
        summary, hac = horizon.ic_summary, horizon.hac
        interval = asdict(hac)
        interval['pair_counts'] = hac.pair_counts.tolist()
        usability = evaluate_interval_usability(hac, point_estimate=summary.mean)
        # Use the HAC variance in the same two-sided normal test underlying the interval.
        p_value = erfc(abs(summary.mean) / sqrt(hac.variance) / sqrt(2)) if hac.reason is None else None
        rows.append(dict(
            frequency=frequency, horizon_bars=h, grid_start_utc=stats.datetimes[0],
            grid_end_utc=stats.datetimes[-1], ic=asdict(summary), hac=interval,
            interval_usability=asdict(usability), p_value=p_value,
            p_value_method='two_sided_normal_hac' if p_value is not None else None,
            buckets={q: asdict(value) for q, value in horizon.bucket_summary.items()},
            spread=asdict(horizon.spread_summary), direction=asdict(horizon.direction),
            direction_by_bucket={q: asdict(value) for q, value in horizon.direction_by_bucket.items()},
        ))
    return rows


def _record_trials(family: BarTrialFamily, rows: list[dict[str, Any]], *,
                   keys: tuple[TrialKey, ...], mode: str, window: _SelectionWindow) -> None:
    by_horizon: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_horizon.setdefault(row['horizon_bars'], []).append(row)
    trials = {}
    for key in keys:
        matches = by_horizon.get(key[1], [])
        if len(matches) != 1:
            reason = 'multiple_evaluation_streams' if matches else 'no_evaluation_stream'
            trials[key] = (Trial(None, None, 0), reason)
            continue
        row = matches[0]
        ic = row['ic']
        if mode == 'time_series':
            statistic, count, interval = ic['spearman_ic'], ic['sample_count'], row['bootstrap']
        else:
            statistic, count, interval = ic['mean'], ic['n_valid'], row['hac']
        bounds = interval['lower'], interval['upper']
        trials[key] = (Trial(
            statistic if np.isfinite(statistic) else None, row.get('p_value'), count,
            bounds if np.isfinite(bounds).all() else None,
        ), interval['reason'] or ic['reason'])
    family._record_trials(trials, window=window)


def evaluate_bar_splits(labels: pd.DataFrame, labeler: BarReturnLabeler, *,
                        engine: EvaluationEngine, split: SplitConfig | None,
                        family: BarTrialFamily, factor_id: str,
                        options: dict[str, Any]) -> BarEvaluationV2:
    """Adapt Bar labels to the unified engine and retain split-level evidence."""
    window = _selection_window(split)
    family._validate_evaluation_window(window)
    selection_split, split = window
    if split is not None and labels.empty:
        raise ContractValidationError('Both train and oos require at least one scheduled signal')
    time_series = engine.analysis_mode == 'time_series'
    scheme = 'per_cross_section'
    if time_series:
        scheme = 'fitted_on_train' if split else 'full_sample_descriptive_only'
    splits = {
        name: dict(train_end=pd.Timestamp(split.train_end) if split else None,
                   oos_start=pd.Timestamp(split.oos_start) if split else None, quantile_scheme=scheme,
                   fit_identity={} if time_series and split else None,
                   split_purged_count=dict.fromkeys(labeler.horizons, 0), horizons=[])
        for name in (('train', 'oos') if split else ('full',))
    }
    instrument_stats: dict[str, dict[tuple[str, int], dict[Hashable, TimeSeriesInstrument]]] = {
        name: {} for name in splits
    }
    keys = ['frequency', 'instrument_id'] if time_series else 'frequency'
    for key, group in labels.groupby(keys):
        frequency, instrument = key if time_series else (key, None)
        panel, expected, signal_valid, clock, entries, exits = _bar_panel(
            group, labeler, frequency, scheduled_exits=split is not None, by_availability=not time_series,
        )
        results = engine.evaluate(
            panel, split=split, expected_grid=expected, signal_valid=signal_valid,
            signal_available_at=clock, entry_timestamps=entries, exit_timestamps=exits, **options,
        )
        for name, evaluation in results.splits.items():
            report = splits[name]
            if evaluation.fit_identity is not None:
                report['fit_identity'].setdefault(frequency, {})[instrument] = evaluation.fit_identity
            stats = evaluation.statistics
            if isinstance(stats, TimeSeriesResult):
                for h, horizon in stats.horizons.items():
                    instrument_stats[name].setdefault((frequency, h), {}).update(horizon.instruments)
                rows = _time_series_rows(stats, instrument, frequency)
            else:
                rows = _cross_section_rows(stats, frequency)
            for row in rows:
                h = row['horizon_bars']
                row['split_purged_count'] = evaluation.split_purged_count[h]
                report['split_purged_count'][h] += row['split_purged_count']
                report['horizons'].append(row)

    if time_series:
        for name, report in splits.items():
            summaries = []
            for (frequency, h), instruments in sorted(instrument_stats[name].items()):
                summary = summarize_time_series_horizon(instruments, range(h))
                summaries.append(dict(
                    frequency=frequency, horizon_bars=h, ic_summary=asdict(summary.ic_summary),
                    non_overlapping_ic_summary={offset: asdict(ic)
                                                for offset, ic in summary.non_overlapping_ic_summary.items()},
                ))
            report['instrument_ic_summaries'] = summaries

    _record_trials(family, splits[selection_split]['horizons'],
                   keys=tuple((factor_id, h) for h in labeler.horizons), mode=engine.analysis_mode, window=window)
    return BarEvaluationV2(splits, family.report())
