"""Bar boundaries in trading time, without consulting observed prices."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .frequency import Frequency

if TYPE_CHECKING:
    from utils.session_calendar import Session, SessionCalendar


class BarSchedule:
    def __init__(self, calendar: SessionCalendar, frequency: str | Frequency, *,
                 view: str = 'ETH', bucket_strategy: str | None = None) -> None:
        self.frequency = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
        if not self.frequency.is_valid:
            raise ValueError(f'Invalid frequency: {self.frequency.raw!r}')
        calendar._view(view)
        if bucket_strategy is None:
            if calendar.is_domestic_futures:
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
        """Look up actual availability without rebuilding a Session per label."""
        session = self.calendar.session_at(timestamp_utc, self.view)
        if session is None:
            raise ValueError('timestamp_utc is outside the Bar schedule')
        day = session.session_date
        if day not in self._ends:
            slots = self.for_session(session)
            self._ends[day] = dict(zip(slots.start_utc, slots.end_utc))
        if timestamp_utc not in self._ends[day]:
            raise ValueError('timestamp_utc is off the Bar schedule grid')
        return self._ends[day][timestamp_utc]

    def for_session(self, session: Session | str) -> pd.DataFrame:
        found = self.calendar.session_on(session, self.view)
        segments = () if found is None else found.segments
        starts, ends = pd.DatetimeIndex([], tz='UTC'), pd.DatetimeIndex([], tz='UTC')
        partial = np.array([], dtype=bool)
        if segments:
            if self.bucket_strategy == 'session':
                starts = pd.DatetimeIndex([segments[0][0]])
                ends = pd.DatetimeIndex([segments[-1][1]])
                partial = np.array([False])
            elif self.bucket_strategy == 'segment_anchored':
                width = self.frequency.duration
                grids = [pd.date_range(a, b, freq=width, inclusive='left') for a,b in segments]
                starts = grids[0].append(grids[1:])
                ends = pd.DatetimeIndex([min(t + width, b) for grid, (_,b) in zip(grids, segments)
                                         for t in grid])
                partial = (ends - starts) < width
            else:
                width = self.frequency.duration.value
                lengths = np.array([(b-a).value for a,b in segments], dtype=np.int64)
                offsets = np.concatenate(([0], lengths.cumsum()))
                left = np.arange(0, offsets[-1], width, dtype=np.int64)
                right = np.minimum(left + width, offsets[-1])
                # A boundary at a pause closes the prior bucket before the pause;
                # the next bucket starts after it.
                li = np.searchsorted(offsets[1:], left, side='right')
                ri = np.searchsorted(offsets[1:], right, side='left')
                starts = pd.DatetimeIndex([segments[i][0] + pd.Timedelta(int(t-offsets[i]), 'ns')
                                           for i,t in zip(li,left)])
                ends = pd.DatetimeIndex([segments[i][0] + pd.Timedelta(int(t-offsets[i]), 'ns')
                                         for i,t in zip(ri,right)])
                partial = right-left < width
        return pd.DataFrame(dict(session_id=[found.session_date]*len(starts) if found else [],
                                 slot=np.arange(len(starts)), start_utc=starts, end_utc=ends,
                                 available_at_utc=ends, is_partial=partial))


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
