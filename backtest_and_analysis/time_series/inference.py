"""Calendar-grid inputs, inference reports and trial families for Bar evaluation."""
from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import asdict
from numbers import Integral, Real
from threading import Lock
from typing import Any, Literal

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation import evaluate_interval_usability
from backtest_and_analysis.evaluation.calibration import (
    REGISTERED_CALIBRATIONS, Correction, Trial, TrialKey, TrialLedger, _registered_adaptive_evidence,
    _bootstrap_observation_shape, _finite_real,
)
from backtest_and_analysis.evaluation.cross_section import hac_p_value
from backtest_and_analysis.evaluation.kernels import CorrelationResult
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.split import SplitConfig, SplitEvaluation, _validate_split_order
from backtest_and_analysis.evaluation.time_series import summarize_time_series_horizon
from backtest_and_analysis.evaluation.types import (
    BOOTSTRAP_FIXED, BOOTSTRAP_V1, BOOTSTRAP_V2,
    AdaptiveCalibrationObservation, CalibrationEvidence, CrossSectionResult, IntervalUsability,
    TimeSeriesInstrument, TimeSeriesResult, interval_usability_payload,
)

from .contracts import ContractValidationError, _numeric_column
from .labels import BarReturnLabeler, _grid

_SelectionWindow = tuple[Literal['full', 'train'], SplitConfig | None]


def _selection_window(split: SplitConfig | None) -> _SelectionWindow:
    if split is None:
        return 'full', None
    train_end, oos_start = pd.Timestamp(split.train_end), pd.Timestamp(split.oos_start)
    if train_end.tzinfo is None or oos_start.tzinfo is None:
        raise ContractValidationError('Use timezone-aware split bounds for Bar evaluation')
    _validate_split_order(train_end, oos_start)
    return 'train', SplitConfig(train_end.tz_convert('UTC'), oos_start.tz_convert('UTC'))


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
        adjusted = {row.key: row for row in selection.trials} if selection else {}
        inference_adjusted = selection.inference_adjusted_p_values if selection else {}
        candidates = []
        for key in self.candidates:
            trial, row = records.get(key), adjusted.get(key)
            trial_payload = asdict(trial) if trial is not None else dict(
                statistic=None, p_value=None, n_effective=None, interval=None, interval_usability=None,
                p_value_method=None,
            )
            if trial is not None and trial.interval_usability is not None:
                trial_payload["interval_usability"] = interval_usability_payload(trial.interval_usability)
            candidates.append(dict(
                factor_id=key[0], horizon=key[1],
                **trial_payload,
                n_observations=trial.n_observations if trial is not None else None,
                n_dependence_adjusted=None,
                reason=reasons.get(key, 'not_evaluated'),
                adjusted_p_value=row.adjusted_p_value if row else None,
                inference_adjusted_p_value=inference_adjusted.get(key),
                significant=row.significant if row else False,
            ))
        return dict(
            method=self.method, alpha=self.alpha, attempt_count=len(self.candidates),
            selection_complete=selection is not None, best=selection.best if selection else None,
            pending_candidates=pending, missing_p_value_candidates=missing,
            **({'split': window[0]} if window is not None else {}),
            candidates=candidates,
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
            row.bootstrap, point_estimate=ic['spearman_ic'], point_reason=ic['reason'], horizon=h,
            ic_method='spearman', intervals_verified=stats.intervals_verified,
        )
        rows.append(dict(
            instrument_id=instrument, frequency=frequency, horizon_bars=h,
            grid_start_utc=stats.datetimes[0], grid_end_utc=stats.datetimes[-1],
            intervals_verified=stats.intervals_verified, ic=ic,
            bootstrap=asdict(row.bootstrap), interval_usability=interval_usability_payload(usability),
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
        sizes = horizon.ic.counts
        if hac.bandwidth_method != 'fixed':
            sizes = np.where(np.isfinite(horizon.ic.values), sizes, 0)
        usability = evaluate_interval_usability(
            hac, point_estimate=summary.mean, horizon=h,
            cross_section_sizes=sizes, ic_method=stats.ic_method,
        )
        p_value, p_value_method = hac_p_value(hac, summary.mean)
        rows.append(dict(
            frequency=frequency, horizon_bars=h, grid_start_utc=stats.datetimes[0],
            grid_end_utc=stats.datetimes[-1], ic=asdict(summary), hac=interval,
            interval_usability=interval_usability_payload(usability), p_value=p_value,
            p_value_method=p_value_method,
            buckets={q: asdict(value) for q, value in horizon.bucket_summary.items()},
            spread=asdict(horizon.spread_summary), direction=asdict(horizon.direction),
            direction_by_bucket={q: asdict(value) for q, value in horizon.direction_by_bucket.items()},
        ))
    return rows


def _hac_observation_matches(interval: Mapping[str, Any], observation: AdaptiveCalibrationObservation,
                             *, horizon: int, count: int) -> bool:
    expected = {
        'method_version': observation.method_version,
        'calibration_scenario': observation.scenario,
        'grid_size': observation.grid_size,
        'n_valid': observation.n_valid,
        'lag': observation.window,
        'min_lag': observation.structural_minimum,
        'variance': observation.hac_variance,
        'effective_df': observation.effective_df,
        'alpha': observation.alpha,
        'missing_pattern': observation.missing_pattern,
        'bandwidth_method': 'nw94_prewhitened',
    }
    required = (*expected.values(), observation.cross_section_width_min, observation.cross_section_width_max)
    return (
        None not in required and 'reason' in interval and interval['reason'] is None
        and observation.horizon == horizon and observation.n_valid == count
        and all(
            field in interval and not isinstance(interval[field], (bool, np.bool_)) and interval[field] == value
            for field, value in expected.items()
        )
    )


def _bootstrap_observation_matches(
    interval: Mapping[str, Any], observation: AdaptiveCalibrationObservation, *,
    horizon: int, count: int, intervals_verified: bool,
) -> bool:
    o = observation
    tags = BOOTSTRAP_V2 if o.method_version == BOOTSTRAP_V2[0] else BOOTSTRAP_V1
    tag_fields = ('method_version', 'block_length_method', 'resample_method', 'interval_method')
    integers = {
        'grid_size': o.grid_size, 'n_valid_pairs': o.n_valid, 'block_length': o.window,
        'min_block_length': o.structural_minimum, 'n_boot': o.n_boot,
        'min_valid_replicates': o.min_valid_replicates, 'valid_replicates': o.valid_replicates,
        'min_pairs_per_replicate': o.min_pairs_per_replicate,
    }
    reals = {'effective_blocks': o.effective_blocks, 'alpha': o.alpha}
    se = interval.get('fisher_z_standard_error')
    if tags == BOOTSTRAP_V2:
        reals['fisher_z_standard_error'] = o.fisher_z_standard_error
    lower, upper = interval.get('lower'), interval.get('upper')
    return (
        intervals_verified is True and _bootstrap_observation_shape(o)
        and tuple(interval.get(key) for key in tag_fields) == tags and o.method_version == tags[0]
        and all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) for v in (horizon, count))
        and o.horizon == horizon and o.n_valid == count
        and 'reason' in interval and interval['reason'] is None
        and interval.get('calibration_scenario') == o.scenario and interval.get('missing_pattern') == o.missing_pattern
        and all(
            isinstance(interval.get(field), Integral) and not isinstance(interval[field], (bool, np.bool_))
            and interval[field] == expected for field, expected in integers.items()
        )
        and all(_finite_real(interval.get(field)) and interval[field] == expected for field, expected in reals.items())
        and _finite_real(se) and se > 0
        and _finite_real(lower) and _finite_real(upper) and -1 <= lower <= upper <= 1
    )


def _read_trial_usability(row: Mapping[str, Any], interval: Mapping[str, Any], usability: Mapping[str, Any], *,
                          mode: str, count: int, has_bounds: bool, point_usable: bool) -> IntervalUsability:
    status = usability.get('calibration_status')
    reason = usability.get('reason')
    descriptive = usability.get('descriptive_stats_usable')
    significant = usability.get('significance_interpretation_usable')
    verified_horizon = usability.get('verified_horizon')
    horizon = row['horizon_bars']
    horizon_matches = (
        all(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            for value in (horizon, verified_horizon))
        and horizon > 0 and verified_horizon == horizon
    )
    declaration_valid = (
        'reason' in usability and (reason is None or isinstance(reason, str))
        and status in ('uncalibrated', 'calibrated_pass', 'calibrated_fail')
        and isinstance(descriptive, bool) and isinstance(significant, bool)
        and (verified_horizon is None or horizon_matches)
        and (not significant or (status == 'calibrated_pass' and reason is None and descriptive and horizon_matches))
    )
    payload = usability.get('adaptive_observation')
    observation = None
    if payload is not None:
        try:
            observation = AdaptiveCalibrationObservation(**payload)
        except TypeError:
            pass
    time_series = mode == 'time_series'
    adaptive = payload is not None or interval.get('method_version') is not None
    if time_series:
        adaptive = adaptive or tuple(interval.get(field, default) for field, default in zip(
            ('method_version', 'block_length_method', 'resample_method', 'interval_method'), BOOTSTRAP_FIXED,
        )) != BOOTSTRAP_FIXED
        statistic = row.get('ic', {}).get('spearman_ic')
        observation_matches = observation is not None and _bootstrap_observation_matches(
            interval, observation, horizon=horizon, count=count, intervals_verified=row.get('intervals_verified'),
        ) and point_usable and _finite_real(statistic) and -1 <= statistic <= 1
    else:
        adaptive = (adaptive or interval.get('bandwidth_method', 'fixed') != 'fixed'
                    or row.get('p_value_method') == 'two_sided_student_t_hac')
        observation_matches = observation is not None and _hac_observation_matches(
            interval, observation, horizon=horizon, count=count,
        )
    invalid_observation = adaptive and not observation_matches
    evidence_payload = usability.get('calibration_evidence')
    evidence = None
    if isinstance(evidence_payload, Mapping):
        try:
            evidence = CalibrationEvidence(**evidence_payload)
        except TypeError:
            pass
    claims_calibration = evidence_payload is not None or status != 'uncalibrated'
    registered = _registered_adaptive_evidence(observation) if observation_matches and adaptive else None
    if time_series and registered is not None and isinstance(evidence_payload, Mapping):
        evidence_counts_valid = all(
            isinstance(evidence_payload.get(key), Integral) and not isinstance(evidence_payload[key], (bool, np.bool_))
            for key in ('window', 'sample_size', 'bootstrap_replicates')
        )
    else:
        evidence_counts_valid = True
    inconsistent_bootstrap = time_series and adaptive and (
        not evidence_counts_valid
        or status != ('calibrated_pass' if registered is not None else 'uncalibrated')
        or evidence_payload != (asdict(registered) if registered is not None else None)
        or verified_horizon != (horizon if registered is not None else None)
        or reason is not None or descriptive is not True or significant is not (registered is not None)
        or row.get('p_value') is not None or row.get('p_value_method') is not None
        or interval.get('calibration_status', 'uncalibrated') != 'uncalibrated'
        or interval.get('calibration_evidence') is not None
    )
    if not declaration_valid or invalid_observation or inconsistent_bootstrap or (claims_calibration and (
        evidence is None or status == 'uncalibrated' or evidence.passes is not (status == 'calibrated_pass')
        or not {'lower', 'upper', 'reason'} <= interval.keys() or (significant and not has_bounds)
        or (
            (not horizon_matches or evidence != registered) if adaptive
            else evidence not in REGISTERED_CALIBRATIONS
        )
    )):
        return IntervalUsability(
            reason=interval.get('reason') or (reason if isinstance(reason, str) else None),
            calibration_status='uncalibrated',
            calibration_evidence=None,
            descriptive_stats_usable=point_usable if time_series and adaptive else (
                descriptive if isinstance(descriptive, bool) else point_usable
            ),
            significance_interpretation_usable=False,
        )
    return IntervalUsability(
        reason=reason, calibration_status=status, calibration_evidence=evidence,
        descriptive_stats_usable=descriptive, significance_interpretation_usable=significant,
        verified_horizon=verified_horizon, adaptive_observation=observation,
    )


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
            statistic, count, interval = ic['spearman_ic'], ic.get('sample_count'), row['bootstrap']
        else:
            statistic, count, interval = ic['mean'], ic['n_valid'], row['hac']
        lower, upper = interval.get('lower'), interval.get('upper')
        bounds = (lower, upper) if all(
            isinstance(bound, Real) and not isinstance(bound, bool) and np.isfinite(bound)
            for bound in (lower, upper)
        ) and lower <= upper else None
        statistic = statistic if statistic is not None and np.isfinite(statistic) else None
        declared = row.get('interval_usability')
        declared = declared if isinstance(declared, Mapping) else {}
        # A cross-sectional summary reason can explain missing ICIR while the mean is defined.
        point_usable = statistic is not None and (mode != 'time_series' or ic['reason'] is None)
        usability = _read_trial_usability(
            row, interval, declared, mode=mode, count=count, has_bounds=bounds is not None,
            point_usable=point_usable,
        )
        trial_reason = interval.get('reason') or ic['reason']
        if mode == 'time_series':
            paired_count = interval.get('n_valid_pairs')
            valid_count = isinstance(count, Integral) and not isinstance(count, (bool, np.bool_)) and count >= 0
            valid_paired_count = (
                isinstance(paired_count, Integral) and not isinstance(paired_count, (bool, np.bool_))
                and paired_count >= 0
            )
            if (
                not valid_count or (count == 0 and bounds is not None)
                or (paired_count is not None and (not valid_paired_count or count != paired_count))
            ):
                # A point estimate can remain descriptive without inferring a count from a single field.
                payload = declared.get('adaptive_observation')
                observed_count = payload.get('n_valid') if isinstance(payload, Mapping) else None
                grid_size = interval.get('grid_size')
                corroborated = (
                    valid_paired_count
                    and all(isinstance(n, Integral) and not isinstance(n, (bool, np.bool_))
                            for n in (observed_count, grid_size))
                    and 0 < paired_count == observed_count <= grid_size
                )
                count = int(paired_count) if corroborated else 0
                bounds = None
                trial_reason = 'inconsistent_sample_count'
                usability = IntervalUsability(
                    reason=trial_reason, calibration_status='uncalibrated', calibration_evidence=None,
                    descriptive_stats_usable=point_usable, significance_interpretation_usable=False,
                )
        elif count == 0:
            # Corroborate the count for the ledger only after checking the conflict.
            payload = declared.get('adaptive_observation')
            observed_count = payload.get('n_valid') if isinstance(payload, Mapping) else None
            hac_count = interval.get('n_valid')
            if hac_count == observed_count and all(
                isinstance(n, Integral) and not isinstance(n, bool) and n > 0
                for n in (hac_count, observed_count)
            ):
                count = int(hac_count)
        trials[key] = (Trial(
            statistic, None if mode == 'time_series' else row.get('p_value'), count,
            bounds,
            usability,
            p_value_method=None if mode == 'time_series' else row.get('p_value_method'),
        ), trial_reason)
    family._record_trials(trials, window=window)


def build_bar_splits(
    groups: Iterable[tuple[str, str | None, dict[str, SplitEvaluation]]],
    *, horizons: tuple[int, ...], mode: str, split: SplitConfig | None,
) -> dict[str, dict[str, Any]]:
    """Assemble split reports from existing engine statistics."""
    time_series = mode == 'time_series'
    scheme = 'per_cross_section'
    if time_series:
        scheme = 'fitted_on_train' if split else 'full_sample_descriptive_only'
    splits = {
        name: dict(train_end=pd.Timestamp(split.train_end) if split else None,
                   oos_start=pd.Timestamp(split.oos_start) if split else None, quantile_scheme=scheme,
                   fit_identity={} if time_series and split else None,
                   split_purged_count=dict.fromkeys(horizons, 0), horizons=[])
        for name in (('train', 'oos') if split else ('full',))
    }
    instrument_stats: dict[str, dict[tuple[str, int], dict[Hashable, TimeSeriesInstrument]]] = {
        name: {} for name in splits
    }
    for frequency, instrument, evaluations in groups:
        for name, evaluation in evaluations.items():
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

    return splits
