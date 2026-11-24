"""Bar boundaries in trading time, without consulting observed prices."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .frequency import Frequency

if TYPE_CHECKING:
    from utils.session_calendar import Session, SessionCalendar, View


class BarSchedule:
    def __init__(self, calendar: SessionCalendar, frequency: str | Frequency, *,
                 view: View = 'ETH', bucket_strategy: str | None = None) -> None:
        self.frequency = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
        if not self.frequency.is_valid:
            raise ValueError(f'Invalid frequency: {self.frequency.raw!r}')
        calendar._view(view)
        if bucket_strategy is None:
            if calendar.is_domestic_futures and self.frequency.kind != 'week':
                raise ValueError('Domestic futures require an explicit bucket_strategy')
            bucket_strategy = 'segment_anchored'
        if bucket_strategy not in ('segment_anchored', 'session_clock', 'session'):
            raise ValueError('bucket_strategy must be segment_anchored, session_clock or session')
        if bucket_strategy == 'session' and self.frequency.kind != 'session':
            raise ValueError('session bucket_strategy requires session frequency (1d)')
        self.calendar, self.view = calendar, view
        self.bucket_strategy = 'session' if self.frequency.kind == 'session' else bucket_strategy
        self._ends: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}

    def end_at(self, timestamp_utc: pd.Timestamp) -> pd.Timestamp:
        """Return the scheduled end for a Bar's opening timestamp."""
        session = self.calendar.session_at(timestamp_utc, self.view)
        if session is None:
            raise ValueError('timestamp_utc is outside the Bar schedule')
        day = session.session_date
        if self.frequency.kind == 'week':
            week = self.calendar._week_on(day, self.view)
            if not week:
                raise ValueError('timestamp_utc is outside a covered trading week')
            day = week[-1].session_date
        if day not in self._ends:
            slots = self.for_session(day)
            self._ends[day] = dict(zip(slots.start_utc, slots.end_utc))
        if timestamp_utc not in self._ends[day]:
            raise ValueError('timestamp_utc is off the Bar schedule grid')
        return self._ends[day][timestamp_utc]

    def available_at(self, grid_timestamp_utc: pd.Timestamp) -> pd.Timestamp:
        """Return availability for a timestamp emitted by grid()."""
        if self.frequency.kind == 'intraday':
            return self.end_at(grid_timestamp_utc)
        session = self.calendar.session_at(grid_timestamp_utc, self.view, frequency=self.frequency)
        if session is None:
            raise ValueError('grid_timestamp_utc is off the Bar schedule grid')
        return session.segments[-1][1]

    def for_session(self, session: Session | str) -> pd.DataFrame:
        found = self.calendar.session_on(session, self.view)
        segments = () if found is None else found.segments
        starts, ends = pd.DatetimeIndex([], tz='UTC'), pd.DatetimeIndex([], tz='UTC')
        partial = np.array([], dtype=bool)
        if segments and self.frequency.kind == 'week':
            week = self.calendar._week_on(found.session_date, self.view)
            segments = (week[0].segments[0], week[-1].segments[-1]) if week and found == week[-1] else ()
        if segments:
            if self.frequency.kind == 'intraday' and self.bucket_strategy == 'segment_anchored':
                width = self.frequency.duration
                grids = [pd.DatetimeIndex([a]) if width >= b - a
                         else pd.date_range(a, b, freq=width, inclusive='left') for a, b in segments]
                starts = grids[0].append(grids[1:])
                ends = pd.DatetimeIndex([t + min(width, b - t) for grid, (_,b) in zip(grids, segments)
                                         for t in grid])
                partial = (ends - starts) < width
            elif self.frequency.kind == 'intraday':
                width = self.frequency.duration.value
                lengths = np.array([(b-a).value for a,b in segments], dtype=np.int64)
                offsets = np.concatenate(([0], lengths.cumsum()))
                left = np.arange(0, offsets[-1], width, dtype=np.int64)
                right = left + np.minimum(width, offsets[-1] - left)
                # A boundary at a pause closes the prior bucket before the pause;
                # the next bucket starts after it.
                li = np.searchsorted(offsets[1:], left, side='right')
                ri = np.searchsorted(offsets[1:], right, side='left')
                starts = pd.DatetimeIndex([segments[i][0] + pd.Timedelta(int(t-offsets[i]), 'ns')
                                           for i,t in zip(li,left)])
                ends = pd.DatetimeIndex([segments[i][0] + pd.Timedelta(int(t-offsets[i]), 'ns')
                                         for i,t in zip(ri,right)])
                partial = right-left < width
            else:
                starts = pd.DatetimeIndex([segments[0][0]])
                ends = pd.DatetimeIndex([segments[-1][1]])
                partial = np.array([False])
        return pd.DataFrame(dict(session_id=[found.session_date]*len(starts) if found else [],
                                 slot=np.arange(len(starts)), start_utc=starts, end_utc=ends,
                                 available_at_utc=ends, is_partial=partial))

    def session_slots(self) -> pd.DataFrame:
        """Collect physical Bar slots over the calendar's declared coverage."""
        days = self.calendar.sessions().strftime('%Y-%m-%d')
        slots = [self.for_session(day) for day in days]
        return pd.concat(slots, ignore_index=True) if slots else pd.DataFrame()

    def grid(self, start: str | pd.Timestamp | None = None,
             end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """Return intraday opens and daily/weekly anchors keyed by session date."""
        keys = []
        for day in self.calendar.sessions(start, end):
            session = self.calendar.session_on(day, self.view)
            if session is None:
                continue
            slots = self.for_session(session)
            if slots.empty:
                continue
            timestamps = slots.start_utc if self.frequency.kind == 'intraday' else self.calendar._session_anchor(session)
            keys.append(slots[['session_id']].assign(timestamp_utc=timestamps))
        columns = ['timestamp_utc', 'session_id']
        if not keys:
            return pd.DataFrame({'timestamp_utc': pd.DatetimeIndex([], tz='UTC'),
                                 'session_id': pd.Series(dtype=object)})
        return pd.concat(keys, ignore_index=True)[columns]


@dataclass(frozen=True)
class Lookback:
    sessions: int | None = None
    bars: int | None = None

    def __post_init__(self) -> None:
        if (self.sessions is None) == (self.bars is None):
            raise ValueError('Specify exactly one of sessions or bars')
        value = self.sessions if self.sessions is not None else self.bars
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError('Lookback must be a positive integer')

    def to_bars(self, schedule: BarSchedule, *, end_session: str) -> int:
        """Count backwards including end_session; never estimate a sessions/day ratio."""
        schedule.calendar.session_on(end_session, schedule.view)
        if self.bars is not None:
            return int(self.bars)
        days = schedule.calendar.sessions()
        days = days[days <= pd.Timestamp(end_session)]
        if len(days) < self.sessions:
            raise ValueError('Lookback exceeds calendar coverage')
        return sum(len(schedule.for_session(day.strftime('%Y-%m-%d'))) for day in days[-self.sessions:])
