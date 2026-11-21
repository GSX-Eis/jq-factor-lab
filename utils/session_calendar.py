"""PMC session snapshots, corrected by versioned evidence; never infer from prices."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import pandas_market_calendars as pmc

from backtest_and_analysis.time_series.contracts import _utc
from utils.time_axis import BarSchedule

View = Literal['ETH', 'RTH']
Segment = tuple[pd.Timestamp, pd.Timestamp]
DEFAULT_OVERRIDES = Path(__file__).resolve().parents[1] / 'config/calendar_overrides.json'


class CalendarCoverageError(ValueError):
    """Requested dates/history are outside the declared calendar coverage."""


def _date(value: str | pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('Session date must be a finite, normalized, timezone-naive date')
    return day


@dataclass(frozen=True)
class Session:
    session_date: str
    segments: tuple[Segment, ...]


def _segments(row: pd.Series, open_close_map: dict) -> tuple[Segment, ...]:
    opening, closing = row['market_open'], row['market_close']
    cuts = []
    pending = None
    for name in row.index:
        if name in ('market_open', 'market_close'):
            continue
        is_open = open_close_map.get(name)
        if name.startswith('interruption_start_'):
            is_open = False
        elif name.startswith('interruption_end_'):
            is_open = True
        if is_open is False:
            if pending is not None:
                raise ValueError('Unpaired PMC session boundary')
            pending = name
        elif is_open is True:
            if pending is None:
                raise ValueError('Unpaired PMC session boundary')
            if pd.notna(row[pending]) or pd.notna(row[name]):
                cuts.append((row[pending], row[name]))
            pending = None
    if pending is not None:
        raise ValueError('Unpaired PMC session boundary')
    segments = [(opening, closing)]
    for left, right in cuts:
        if pd.isna(left) or pd.isna(right) or right < left:
            raise ValueError('Invalid PMC interruption')
        if left == right:
            continue
        segments = [(a, b) for start, end in segments
                    for a, b in ((start, min(end, left)), (max(start, right), end)) if a < b]
    return tuple(segments)


def _schedule(name: str, start: pd.Timestamp, end: pd.Timestamp,
              overrides: dict) -> tuple[dict[str, tuple[Segment, ...]], str]:
    calendar = pmc.get_calendar(name)
    frame = calendar.schedule(start, end, interruptions=True)
    sessions = {day.strftime('%Y-%m-%d'): _segments(row, calendar.open_close_map)
                for day, row in frame.iterrows()}
    corrections = overrides.get(name, {})
    for kind in ('remove_sessions', 'add_sessions', 'special_sessions'):
        entries = corrections.get(kind, [])
        seen = set()
        for entry in entries:
            day = _date(entry['session_date']).strftime('%Y-%m-%d')
            if day in seen or not isinstance(entry.get('source'), str) or not entry['source'].strip():
                raise ValueError('Each calendar correction needs a unique date and verification source')
            seen.add(day)
            if not start <= pd.Timestamp(day) <= end:
                continue
            if kind == 'remove_sessions':
                sessions.pop(day, None)
                continue
            segments = []
            for left, right in entry['segments']:
                a, b = pd.Timestamp(left), pd.Timestamp(right)
                _utc(a, 'segment_open_utc')
                _utc(b, 'segment_close_utc')
                if a >= b or (segments and a < segments[-1][1]):
                    raise ValueError('Correction segments must be ordered, disjoint and nonempty')
                segments.append((a, b))
            if not segments:
                raise ValueError('Open sessions require segments')
            sessions[day] = tuple(segments)
    return dict(sorted(sessions.items())), str(calendar.tz)


class SessionCalendar:
    """Declared session-date coverage and UTC segments for ETH and derived RTH.

    NQ/ES use CME Globex Equity and NYSE by default. Other PMC calendar names
    are accepted; callers declare their coverage explicitly. XSHG defaults to
    the locally cross-checked 2016-01-04..2026-09-30 interval.
    """

    def __init__(self, calendar_id: str = 'XSHG', start_date: str | None = None,
                 end_date: str | None = None, *, rth_reference_calendar: str | None = None,
                 overrides_path: str | Path = DEFAULT_OVERRIDES) -> None:
        aliases = {'NQ': 'CME Globex Equity', 'ES': 'CME Globex Equity',
                   'CMEGlobex_Equity': 'CME Globex Equity'}
        self.calendar_id = aliases.get(calendar_id, calendar_id)
        calendar = pmc.get_calendar(self.calendar_id)
        self.is_domestic_futures = calendar.__class__.__module__.endswith('.china_futures')
        if self.calendar_id == 'XSHG':
            start_date = start_date or '2016-01-04'
            end_date = end_date or '2026-09-30'
        if start_date is None or end_date is None:
            raise ValueError('Declare start_date and end_date for calendar coverage')
        self.start, self.end = _date(start_date), _date(end_date)
        if self.start > self.end:
            raise ValueError('Calendar start must not follow end')
        default_reference = self.calendar_id
        if self.calendar_id in ('CME Globex Equity', 'CME_Equity'):
            default_reference = 'NYSE'
        elif self.is_domestic_futures and not calendar.name.startswith(('CFFEX', 'GFEX')):
            default_reference = calendar.name.split('_')[0] + '_Day'
        self.rth_reference_calendar = rth_reference_calendar or default_reference
        raw = Path(overrides_path).read_bytes()
        spec = json.loads(raw)
        if spec.get('schema_version') != 1 or not isinstance(spec.get('calendars'), dict):
            raise ValueError('Calendar overrides require schema_version 1 and calendars')
        overrides = spec['calendars']
        # Padding identifies overnight timestamps belonging to uncovered sessions.
        eth, self.timezone = _schedule(self.calendar_id, self.start - pd.Timedelta(days=7),
                                       self.end + pd.Timedelta(days=7), overrides)
        reference, _ = _schedule(self.rth_reference_calendar, self.start - pd.Timedelta(days=7),
                                 self.end + pd.Timedelta(days=7), overrides)
        reference_segments = sorted(s for segments in reference.values() for s in segments)
        reference_ends = [b for _, b in reference_segments]
        reference_opens = [a for a, _ in reference_segments]
        rth = {}
        for day, segments in eth.items():
            rth[day] = tuple((max(a, c), min(b, d)) for a, b in segments
                             for c, d in reference_segments[bisect_right(reference_ends, a):
                                                            bisect_left(reference_opens, b)]
                             if max(a, c) < min(b, d))
        self._sessions: dict[View, dict[str, Session]] = {}
        self._intervals: dict[View, list[tuple[pd.Timestamp, pd.Timestamp, Session]]] = {}
        self._opens: dict[View, list[pd.Timestamp]] = {}
        for view, schedule in (('ETH', eth), ('RTH', rth)):
            self._sessions[view] = {day: Session(day, segments) for day, segments in schedule.items()
                                    if self.start <= pd.Timestamp(day) <= self.end}
            intervals = sorted((a, b, Session(day, segments)) for day, segments in schedule.items()
                               for a, b in segments)
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError('Calendar sessions must not overlap')
            self._intervals[view] = intervals
            self._opens[view] = [a for a, _, _ in intervals]
        payload = {'calendar': self.calendar_id, 'reference': self.rth_reference_calendar,
                   'start': str(self.start), 'end': str(self.end), 'timezone': self.timezone,
                   'pmc': pmc.__version__, 'exchange_calendars': version('exchange-calendars'),
                   'tzdata': version('tzdata'), 'overrides_sha256': hashlib.sha256(raw).hexdigest(),
                   'sessions': {v: [(s.session_date, [(a.isoformat(), b.isoformat())
                                                    for a, b in s.segments])
                                    for s in days.values()] for v, days in self._sessions.items()}}
        self._calendar_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def calendar_hash(self) -> str:
        return self._calendar_hash

    def _view(self, view: View) -> None:
        if view not in ('ETH', 'RTH'):
            raise ValueError('view must be ETH or RTH')

    def _validate_date(self, day: pd.Timestamp) -> None:
        if not self.start <= day <= self.end:
            raise CalendarCoverageError(f'Calendar covers {self.start.date()}..{self.end.date()}; requested {day.date()}')

    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(list(self._sessions['ETH']), dtype='datetime64[ns]')

    def session_at(self, ts_utc: datetime, view: View = 'ETH') -> Session | None:
        _utc(ts_utc, 'ts_utc')
        self._view(view)
        ts = pd.Timestamp(ts_utc)
        # Use ETH ownership even for RTH queries in the overnight edge of coverage.
        intervals = self._intervals['ETH']
        pos = bisect_right(self._opens['ETH'], ts) - 1
        owner = intervals[pos][2] if pos >= 0 and ts < intervals[pos][1] else None
        day = pd.Timestamp(owner.session_date) if owner else ts.tz_convert(self.timezone).tz_localize(None).normalize()
        self._validate_date(day)
        intervals = self._intervals[view]
        pos = bisect_right(self._opens[view], ts) - 1
        return intervals[pos][2] if pos >= 0 and ts < intervals[pos][1] else None

    def expected_bar_opens(self, session: Session | str, frequency: str,
                           view: View = 'ETH', *, bucket_strategy: str | None = None) -> pd.DatetimeIndex:
        slots = BarSchedule(self, frequency, view=view, bucket_strategy=bucket_strategy).for_session(session)
        return pd.DatetimeIndex(slots.start_utc)

    def session_on(self, session: Session | str, view: View = 'ETH') -> Session | None:
        """Return the selected view's segments on a covered session date."""
        self._view(view)
        day = _date(session.session_date if isinstance(session, Session) else session)
        self._validate_date(day)
        return self._sessions[view].get(day.strftime('%Y-%m-%d'))

    def expected_bar_opens_in_range(self, start_utc: datetime, end_utc: datetime,
                                    frequency: str, view: View = 'ETH', *,
                                    bucket_strategy: str | None = None) -> pd.DatetimeIndex:
        _utc(start_utc, 'start_utc')
        _utc(end_utc, 'end_utc')
        self._view(view)
        schedule = BarSchedule(self, frequency, view=view, bucket_strategy=bucket_strategy)
        start, end = pd.Timestamp(start_utc), pd.Timestamp(end_utc)
        if start > end:
            raise ValueError('Range start must not follow end')
        self.session_at(start)
        if start < end:
            self.session_at(end - pd.Timedelta(1, 'ns'))
        result = pd.DatetimeIndex([], tz='UTC')
        for session in self._sessions[view].values():
            if any(a < end and b > start for a, b in session.segments):
                grid = pd.DatetimeIndex(schedule.for_session(session).start_utc)
                result = result.append(grid[(grid >= start) & (grid < end)])
        return result
