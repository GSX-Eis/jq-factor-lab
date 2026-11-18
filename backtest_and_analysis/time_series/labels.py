"""Calendar-aligned forward Bar labels shared by both analysis modes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .contracts import (
    ContractValidationError, LabelRecord, _numeric_column,
    invalid_price_mask, primary_exclusion_reasons, validate_bar_frame,
    validate_factor_frame,
)
from .csv_reader import _check_view
from utils.time_axis import BarSchedule

if TYPE_CHECKING:
    from utils.session_calendar import SessionCalendar, View
    from utils.continuous_bars import ContinuousBarSource


LABEL_COLUMNS = tuple(LabelRecord.__dataclass_fields__)
TIME_OUTPUTS = ('entry_timestamp', 'exit_bar_timestamp', 'exit_timestamp')
VALUE_OUTPUTS = ('entry_price', 'exit_price', 'price_change_points', 'return_pct')


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ContractValidationError(f'{name} must be an integer >= 1')
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


def _window_counts(flags: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    prefix = np.concatenate(([0], np.cumsum(flags, dtype=np.int64)))
    return prefix[np.minimum(ends, len(flags))] - prefix[np.minimum(starts, len(flags))]


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
                 continuous_source: ContinuousBarSource | None = None) -> None:
        # The calendar imports contracts through this package during initialization.
        from utils.session_calendar import SessionCalendar

        self.entry_offset_bars = _positive_integer(entry_offset_bars, 'entry_offset_bars')
        if isinstance(horizons, Integral):
            horizons = (horizons,)
        if not isinstance(horizons, Sequence) or isinstance(horizons, (str, bytes)) or not horizons:
            raise ContractValidationError('horizons must be a positive integer or nonempty sequence')
        self.horizons = tuple(_positive_integer(h, 'horizons') for h in horizons)
        if len(set(self.horizons)) != len(self.horizons):
            raise ContractValidationError('horizons must not contain duplicates')
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

    def label(self, bars: pd.DataFrame, factors: pd.DataFrame | None = None, *,
              signal_timestamps: Sequence | None = None) -> pd.DataFrame:
        """Return all selected candidates, with one primary reason per exclusion.

        Factors join on (timestamp_utc, instrument_id); alternatively factor_value
        can be supplied on each Bar. Raw coverage metadata survives view filtering.
        The input frames and their attrs are never modified.
        """
        frame = validate_bar_frame(bars).reset_index(drop=True)
        if 'contract_id' in frame:
            if self.continuous_source is None:
                raise ContractValidationError('Continuous labels require continuous_source for raw contract prices')
            return self.continuous_source.label(frame, self, factors, signal_timestamps=signal_timestamps)
        factor_index = None
        if factors is not None:
            factor_index = validate_factor_frame(factors).set_index(['timestamp_utc', 'instrument_id'])
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
            if (instrument, frequency) not in coverage:
                raise ContractValidationError(f'Missing raw coverage for {(instrument, frequency)}')
            tables.extend(self._label_group(group, factor_index, signals, calendar,
                                            coverage[(instrument, frequency)], instrument, frequency))
        if not tables:
            result = pd.DataFrame(columns=LABEL_COLUMNS)
            for name in ('signal_timestamp', 'signal_available_at', *TIME_OUTPUTS):
                result[name] = pd.Series(dtype='datetime64[ns, UTC]')
            return result
        return pd.concat(tables, ignore_index=True).loc[:, LABEL_COLUMNS].sort_values(
            ['instrument_id', 'frequency', 'signal_timestamp', 'horizon_bars'],
            kind='stable').reset_index(drop=True)

    def _label_group(self, group: pd.DataFrame, factors: pd.DataFrame | None,
                     signals: pd.DatetimeIndex | None, calendar: SessionCalendar,
                     coverage_end: pd.Timestamp, instrument: str, frequency: str) -> list[pd.DataFrame]:
        first, last = group.timestamp_utc.min(), group.timestamp_utc.max()
        calendar.session_at(first)
        calendar.session_at(last)
        eth, _, _ = _grid(calendar, frequency, first, max(last, coverage_end),
                                  'ETH', self.bucket_strategy)
        grid, ends, owners = _grid(calendar, frequency, first, max(last, coverage_end),
                                     self.view, self.bucket_strategy)
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
        missing = observed.instrument_id.isna().to_numpy()
        if 'is_complete' in observed:
            missing |= ~observed.is_complete.fillna(False).to_numpy(dtype=bool)
        invalid = invalid_price_mask(observed).to_numpy()
        entries = positions + self.entry_offset_bars
        tables = []
        for horizon in self.horizons:
            exits = entries + horizon - 1
            eth_exits = eth_positions + self.entry_offset_bars + horizon - 1
            boundary = ((exits >= len(grid))
                        | ((self.boundary_policy == 'same_session')
                           & (owners[positions] != owners[np.minimum(exits, len(grid) - 1)])))
            insufficient = np.where(boundary,
                (eth_exits >= len(eth)) | (eth.take(np.minimum(eth_exits, len(eth)-1)) > coverage_end),
                grid.take(np.minimum(exits, len(grid)-1)) > coverage_end)
            reasons = primary_exclusion_reasons(pd.DataFrame({
                'factor_missing': factor_missing,
                'insufficient_history': insufficient,
                'session_boundary': boundary,
                'missing_bar': ((_window_counts(missing, positions + 1, exits + 1) > 0)
                                | missing[positions]),
                'invalid_price': _window_counts(invalid, entries, exits + 1) > 0,
            }))
            valid = reasons.isna().to_numpy()
            table = pd.DataFrame({
                'instrument_id': instrument, 'frequency': frequency,
                'entry_offset_bars': self.entry_offset_bars, 'horizon_bars': horizon,
                'signal_timestamp': signal_times, 'signal_available_at': ends.take(positions),
                'entry_timestamp': grid.take(np.minimum(entries, len(grid) - 1)),
                'exit_bar_timestamp': grid.take(np.minimum(exits, len(grid) - 1)),
                'exit_timestamp': ends.take(np.minimum(exits, len(grid) - 1)),
                'status': np.where(valid, 'valid', 'excluded'), 'exclusion_reason': reasons,
            })
            table.loc[~valid, list(TIME_OUTPUTS)] = pd.NaT
            table.loc[:, list(VALUE_OUTPUTS)] = np.nan
            opening = observed.open.to_numpy(dtype=float, na_value=np.nan)[entries[valid]]
            close = observed.close.to_numpy(dtype=float, na_value=np.nan)[exits[valid]]
            table.loc[valid, 'entry_price'] = opening
            table.loc[valid, 'exit_price'] = close
            table.loc[valid, 'price_change_points'] = close - opening
            table.loc[valid, 'return_pct'] = (close / opening - 1) * 100
            tables.append(table)
        return tables

    def label_records(self, bars: pd.DataFrame, factors: pd.DataFrame | None = None, *,
                      signal_timestamps: Sequence | None = None) -> list[LabelRecord]:
        """Serialize the computed table; NaT/NaN output fields become None."""
        table = self.label(bars, factors, signal_timestamps=signal_timestamps)
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
