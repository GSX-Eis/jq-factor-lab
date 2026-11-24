"""Calendar-aligned forward Bar labels shared by both analysis modes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation.labels import ExecutionConvention, UnifiedReturnLabeler, _LabelAxis

from .contracts import (
    ContractValidationError, LabelRecord, _numeric_column, _validate_aligned_factor_frame,
    validate_bar_frame,
)
from .csv_reader import _check_view
from utils.time_axis import BarSchedule

if TYPE_CHECKING:
    from utils.session_calendar import SessionCalendar, View
    from utils.continuous_bars import ContinuousBarSource


_BAR = ExecutionConvention.OPEN_TO_CLOSE.value
LABEL_COLUMNS = tuple(LabelRecord.__dataclass_fields__)
TIME_OUTPUTS = ('entry_timestamp', 'exit_bar_timestamp', 'exit_timestamp')
VALUE_OUTPUTS = ('entry_price', 'exit_price', 'price_change_points', 'return_pct')


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ContractValidationError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _grid(calendar: SessionCalendar, frequency: str, start: pd.Timestamp,
          end: pd.Timestamp, view: View, bucket_strategy: str | None
          ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray]:
    # Include whole edge sessions, including overnight ownership on adjacent dates.
    days = calendar.sessions()
    owners = [calendar.session_at(t) for t in (start, end)]
    bounds = [pd.Timestamp(s.session_date) if s else t.tz_convert(calendar.timezone).tz_localize(None).normalize()
              for s,t in zip(owners, (start,end))]
    days = days[(days >= min(bounds)) & (days <= max(bounds))]
    grids, ends, owners = [], [], []
    schedule = BarSchedule(calendar, frequency, view=view, bucket_strategy=bucket_strategy)
    for day in days:
        session_date = day.strftime('%Y-%m-%d')
        slots = schedule.for_session(session_date)
        grid = pd.DatetimeIndex(slots.start_utc)
        grids.append(grid)
        ends.append(pd.DatetimeIndex(slots.end_utc))
        owners.append(np.repeat(session_date, len(grid)))
    if not grids:
        return pd.DatetimeIndex([], tz='UTC'), pd.DatetimeIndex([], tz='UTC'), np.array([], dtype=object)
    return grids[0].append(grids[1:]), ends[0].append(ends[1:]), np.concatenate(owners)


def _normalize_bar_horizons(horizons: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(horizons, Integral):
        horizons = (horizons,)
    if not isinstance(horizons, Sequence) or isinstance(horizons, (str, bytes)) or not horizons:
        raise ContractValidationError('horizons must be a positive integer or nonempty sequence')
    periods = tuple(_positive_integer(h, 'horizons') for h in horizons)
    if len(set(periods)) != len(periods):
        raise ContractValidationError('horizons must not contain duplicates')
    return periods


class BarReturnLabeler:
    """Label observed signals on expected Bar grids, independently per series.

    ``label`` returns the per_bar_labels long table. Computation loops only over
    instrument/frequency groups, calendar sessions and horizons, never signals.
    ``label_records`` additionally serializes rows to the frozen record contract.
    """

    def __init__(self, calendar: SessionCalendar | Mapping[str, SessionCalendar], *,
                 entry_offset_bars: int = 1, horizons: Sequence[int] | int = (1, 5, 15, 60),
                 view: View = 'ETH', boundary_policy: str = 'same_session',
                 bucket_strategy: str | None = None,
                 continuous_source: ContinuousBarSource | None = None,
                 include_scheduled_exit: bool = False, factor_keys_only: bool = False) -> None:
        # The calendar imports contracts through this package during initialization.
        from utils.session_calendar import SessionCalendar

        self.entry_offset_bars = _positive_integer(entry_offset_bars, 'entry_offset_bars', _BAR.min_entry_offset)
        self.horizons = _normalize_bar_horizons(horizons)
        if view not in ('ETH', 'RTH'):
            raise ContractValidationError('view must be ETH or RTH')
        if boundary_policy != 'same_session' and not (
                boundary_policy == 'cross_session' and continuous_source is not None):
            raise ContractValidationError('boundary_policy must be same_session; '
                                          'cross_session requires continuous_source')
        if not isinstance(calendar, SessionCalendar):
            if not isinstance(calendar, Mapping) or any(
                    not isinstance(key, str) or not isinstance(value, SessionCalendar)
                    for key, value in calendar.items()):
                raise ContractValidationError('calendar must be a SessionCalendar or instrument mapping')
            calendar = dict(calendar)
        self.calendar = calendar
        self.view = view
        self.bucket_strategy = bucket_strategy
        self.boundary_policy = boundary_policy
        self.continuous_source = continuous_source
        self.include_scheduled_exit = include_scheduled_exit
        self.factor_keys_only = factor_keys_only

    def label(self, bars: pd.DataFrame, factors: pd.DataFrame | None = None, *,
              signal_timestamps: Sequence | None = None,
              _incomplete_signals: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        """Return all selected candidates, with one primary reason per exclusion.

        Factors join on (timestamp_utc, instrument_id); alternatively factor_value
        can be supplied on each Bar. Raw coverage metadata survives view filtering.
        Set factor_keys_only when the Bar frame contains execution grid placeholders
        and the supplied factors carry exactly the observed signal keys.
        The input frames and their attrs are never modified.
        """
        if self.factor_keys_only and factors is None:
            raise ContractValidationError('factor_keys_only requires aligned factors')
        frame = validate_bar_frame(bars).reset_index(drop=True)
        if 'contract_id' in frame:
            if self.continuous_source is None:
                raise ContractValidationError('Continuous labels require continuous_source for raw contract prices')
            return self.continuous_source.label(frame, self, factors, signal_timestamps=signal_timestamps)
        factor_index = None
        if factors is not None:
            factor_index = _validate_aligned_factor_frame(factors).set_index(['timestamp_utc', 'instrument_id'])
        elif 'factor_value' not in frame:
            raise ContractValidationError('Supply factors or a factor_value Bar column')
        elif np.isinf(_numeric_column(frame.factor_value)).any():
            raise ContractValidationError('factor_value must not be infinite (NaN is allowed)')
        signals = None
        if signal_timestamps is not None:
            try:
                signals = pd.DatetimeIndex(signal_timestamps)
            except (TypeError, ValueError) as exc:
                raise ContractValidationError('signal_timestamps must contain UTC timestamps') from exc
            if signals.empty:
                signals = pd.DatetimeIndex([], tz='UTC')
            elif str(signals.tz) not in ('UTC', 'Etc/UTC') or signals.isna().any():
                raise ContractValidationError('signal_timestamps must contain UTC timestamps')
        coverage = frame.attrs.get('raw_coverage_by_key')
        if coverage is None:
            if 'session_view' in frame.attrs or 'calendar_hash' in frame.attrs:
                raise ContractValidationError('Filtered bars require raw_coverage_by_key; use raw bars')
            coverage = frame.groupby(['instrument_id', 'frequency']).timestamp_utc.max().to_dict()
        else:
            coverage = {key: bounds[1] for key, bounds in coverage.items()}
        tables = []
        for (instrument, frequency), group in frame.groupby(['instrument_id', 'frequency'], sort=True):
            calendar = self.calendar.get(instrument) if isinstance(self.calendar, Mapping) else self.calendar
            if calendar is None:
                raise ContractValidationError(f'No calendar for instrument_id {instrument}')
            _check_view(group, calendar, self.view)
            key = (instrument, frequency)
            no_execution = self.factor_keys_only and 'is_complete' in group and not group.is_complete.eq(True).any()
            if key not in coverage and not no_execution:
                raise ContractValidationError(f'Missing raw coverage for {key}')
            tables.extend(self._label_group(group, factor_index, signals, calendar,
                                            coverage.get(key, pd.NaT), instrument, frequency,
                                            _incomplete_signals))
        scheduled = ('scheduled_exit_timestamp',) if self.include_scheduled_exit else ()
        columns = (*LABEL_COLUMNS, *scheduled)
        if not tables:
            result = pd.DataFrame(columns=columns)
            for name in ('signal_timestamp', 'signal_available_at', *TIME_OUTPUTS, *scheduled):
                result[name] = pd.Series(dtype='datetime64[ns, UTC]')
            return result
        return pd.concat(tables, ignore_index=True).loc[:, columns].sort_values(
            ['instrument_id', 'frequency', 'signal_timestamp', 'horizon_bars'],
            kind='stable').reset_index(drop=True)

    def _label_group(self, group: pd.DataFrame, factors: pd.DataFrame | None,
                     signals: pd.DatetimeIndex | None, calendar: SessionCalendar,
                     coverage_end: pd.Timestamp, instrument: str, frequency: str,
                     incomplete_signals: pd.DatetimeIndex | None) -> list[pd.DataFrame]:
        first, last = group.timestamp_utc.min(), group.timestamp_utc.max()
        calendar.session_at(first)
        calendar.session_at(last)
        grid_end = last if pd.isna(coverage_end) else max(last, coverage_end)
        eth, _, _ = _grid(calendar, frequency, first, grid_end, 'ETH', self.bucket_strategy)
        grid, ends, owners = _grid(calendar, frequency, first, grid_end, self.view, self.bucket_strategy)
        if eth.empty or grid.empty:
            return []
        times = pd.DatetimeIndex(group.timestamp_utc)
        positions = grid.get_indexer(times)
        off_grid = (eth.get_indexer(times) < 0) & (positions < 0)
        if off_grid.any() and any(calendar.session_at(t, 'ETH') is not None for t in times[off_grid]):
            raise ContractValidationError('timestamp_utc is not on the expected Bar start grid')
        selected = positions >= 0
        if signals is not None:
            selected &= times.isin(signals)
        if self.factor_keys_only:
            signal_keys = pd.MultiIndex.from_frame(group[['timestamp_utc', 'instrument_id']])
            selected &= signal_keys.isin(factors.index)
        candidates = group.loc[selected]
        if candidates.empty:
            return []
        positions = positions[selected]
        signal_times = pd.DatetimeIndex(candidates.timestamp_utc)
        # RTH can start inside an ETH bucket (e.g. CME 60m at 14:30).
        eth_positions = np.maximum(eth.searchsorted(signal_times, side='right') - 1, 0)
        if factors is None:
            factor_missing = candidates.factor_value.isna().to_numpy()
        else:
            keys = pd.MultiIndex.from_arrays([signal_times, candidates.instrument_id],
                                              names=['timestamp_utc', 'instrument_id'])
            factor_missing = factors.reindex(keys).factor_value.isna().to_numpy()
        observed = group.set_index('timestamp_utc').reindex(grid)
        # An already aligned external signal does not need the current Bar prices.
        incomplete_signal = False
        if factors is None and incomplete_signals is not None:
            incomplete_signal = signal_times.isin(incomplete_signals)
        return UnifiedReturnLabeler._from_convention(
            ExecutionConvention.OPEN_TO_CLOSE, self.boundary_policy,
        )._label_windows(
            observed, _LabelAxis(grid, grid, ends, owners), positions, eth, eth_positions, coverage_end,
            instrument, frequency, self.entry_offset_bars, self.horizons, factor_missing,
            incomplete_signal=incomplete_signal, require_signal_bar=factors is None,
            include_scheduled_exit=self.include_scheduled_exit,
        )

    def label_records(self, bars: pd.DataFrame, factors: pd.DataFrame | None = None, *,
                      signal_timestamps: Sequence | None = None) -> list[LabelRecord]:
        """Serialize the computed table; NaT/NaN output fields become None."""
        table = self.label(bars, factors, signal_timestamps=signal_timestamps).loc[:, LABEL_COLUMNS]
        records = table.astype(object).where(table.notna(), None).to_dict('records')
        result = []
        schedules = {}
        for record in records:
            instrument, frequency = record['instrument_id'], record['frequency']
            key = (instrument, frequency)
            if key not in schedules:
                calendar = self.calendar.get(instrument) if isinstance(self.calendar, Mapping) else self.calendar
                schedules[key] = BarSchedule(calendar, frequency, view=self.view,
                                             bucket_strategy=self.bucket_strategy)
            result.append(LabelRecord.from_schedule(schedules[key], **record))
        return result
