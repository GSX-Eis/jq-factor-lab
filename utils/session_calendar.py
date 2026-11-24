"""PMC session snapshots, corrected by versioned evidence; never infer from prices."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, time
from functools import cached_property
import hashlib
from importlib.metadata import version
import json
from numbers import Integral
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import exchange_calendars
import numpy as np
import pandas as pd
import pandas_market_calendars as pmc

from backtest_and_analysis.time_series.contracts import _utc
from utils.time_axis import BarSchedule, Frequency

View = Literal['ETH', 'RTH']
Segment = tuple[pd.Timestamp, pd.Timestamp]
DEFAULT_OVERRIDES = Path(__file__).resolve().parents[1] / 'config/calendar_overrides.json'
DEFAULT_CONFIG = DEFAULT_OVERRIDES.with_name('trading_calendar.json')
VERIFIED_XSHG = 'LOCAL_A_SHARE_XSHG_CROSSCHECKED'


class CalendarCoverageError(ValueError):
    """Requested dates/history are outside the declared calendar coverage."""


def _date(value: str | pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('Session date must be a finite, normalized, timezone-naive date')
    return day


def _day(value: str | pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None:
        raise ValueError('Calendar dates must be finite, timezone-naive local market dates')
    return day.normalize()


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
    if calendar.name == 'XSGO' and 'market_close' not in calendar.market_times:
        # PMC's mirror omits XSGO's dynamic close_times property.
        frame = exchange_calendars.get_calendar('XSGO', start=start, end=end).schedule.rename(
            columns={'open': 'market_open', 'close': 'market_close'})
    else:
        frame = calendar.schedule(start, end, interruptions=True)
        # PMC extends discontinued boundaries using their last historical time.
        for boundary, cutoff in calendar.discontinued_market_times.items():
            if boundary in frame.columns:
                frame.loc[frame.index >= pd.Timestamp(cutoff), boundary] = pd.NaT
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


def _wall_time(value: str) -> time:
    if isinstance(value, str):
        try:
            parsed = time.fromisoformat(value)
            if value in (parsed.isoformat(timespec='minutes'), parsed.isoformat(timespec='seconds')):
                return parsed
        except ValueError:
            pass
    raise ValueError('Local times must be HH:MM or HH:MM:SS')


def _local_instant(day: pd.Timestamp, clock: time, timezone: ZoneInfo) -> pd.Timestamp:
    local = pd.Timestamp(datetime.combine(day, clock))
    localized = local.tz_localize(timezone, ambiguous='NaT', nonexistent='NaT')
    if pd.isna(localized):
        raise ValueError(f'Ambiguous or nonexistent local session time: {local} in {timezone}')
    return localized.tz_convert('UTC').as_unit('ns')


def _session_template(value: object) -> tuple[tuple[time, time, int, int], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('An open custom session requires segments')
    segments = []
    for item in value:
        if (not isinstance(item, Mapping) or not {'open', 'close'} <= item.keys()
                or item.keys() - {'open', 'close', 'open_day_offset', 'close_day_offset'}):
            raise ValueError('Custom segments require open and close local times')
        opening, closing = _wall_time(item['open']), _wall_time(item['close'])
        left = item.get('open_day_offset', -1 if closing <= opening else 0)
        right = item.get('close_day_offset', 0)
        if any(type(offset) is not int or offset not in (-1, 0, 1) for offset in (left, right)):
            raise ValueError('Custom segment day offsets must be -1, 0 or 1')
        if (left, opening) >= (right, closing):
            raise ValueError('Custom segment open must precede close')
        if segments and (left, opening) < (segments[-1][3], segments[-1][1]):
            raise ValueError('Custom segments must be ordered and disjoint')
        segments.append((opening, closing, left, right))
    return tuple(segments)


def _custom_schedule(spec: Mapping[str, object], start: pd.Timestamp, end: pd.Timestamp
                     ) -> tuple[dict[str, tuple[Segment, ...]], str]:
    if (not isinstance(spec, Mapping) or not {'timezone', 'segments'} <= spec.keys()
            or spec.keys() - {'timezone', 'segments', 'weekdays', 'dates', 'overrides'}):
        raise ValueError('custom_sessions requires timezone, segments and weekdays or dates')
    try:
        timezone = ZoneInfo(spec['timezone'])
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError('custom_sessions requires an IANA timezone') from exc
    weekdays, dates = spec.get('weekdays', ()), spec.get('dates', ())
    if not isinstance(weekdays, (list, tuple)) or any(type(n) is not int or n not in range(7) for n in weekdays):
        raise ValueError('Custom weekdays must be integers 0 (Monday) through 6 (Sunday)')
    if not isinstance(dates, (list, tuple)) or any(not isinstance(day, str) for day in dates):
        raise ValueError('Custom dates must be a list of date strings')
    if not weekdays and not dates:
        raise ValueError('Declare custom weekdays or dates')
    dates = {_date(day) for day in dates}
    default = _session_template(spec['segments'])
    raw_overrides = spec.get('overrides', {})
    if not isinstance(raw_overrides, Mapping) or any(not isinstance(day, str) for day in raw_overrides):
        raise ValueError('Custom overrides must map dates to segments or null (closed)')
    overrides = {_date(day): () if value is None else _session_template(value)
                 for day, value in raw_overrides.items()}
    if len(overrides) != len(raw_overrides):
        raise ValueError('Custom overrides contain duplicate session dates')
    result = {}
    for day in pd.date_range(start, end):
        template = overrides.get(day, default if day.weekday() in weekdays or day in dates else ())
        if template:
            segments = tuple((_local_instant(day + pd.Timedelta(days=left), opening, timezone),
                              _local_instant(day + pd.Timedelta(days=right), closing, timezone))
                             for opening, closing, left, right in template)
            result[day.strftime('%Y-%m-%d')] = segments
    return result, timezone.key


class SessionCalendar:
    """Declared session-date coverage and UTC segments for ETH and derived RTH.

    NQ/ES resolve to CME Globex Equity with an NYSE RTH reference.
    Callers declare the calendar and coverage explicitly.
    """

    def __init__(self, calendar_id: str, start_date: str | pd.Timestamp | None = None,
                 end_date: str | pd.Timestamp | None = None, *, rth_reference_calendar: str | None = None,
                 overrides_path: str | Path = DEFAULT_OVERRIDES,
                 custom_sessions: Mapping[str, object] | None = None, daily_anchor: str = 'close') -> None:
        aliases = {'NQ': 'CME Globex Equity', 'ES': 'CME Globex Equity',
                   'CMEGlobex_Equity': 'CME Globex Equity'}
        self.calendar_id = aliases.get(calendar_id, calendar_id) if custom_sessions is None else calendar_id
        calendar = pmc.get_calendar(self.calendar_id) if custom_sessions is None else None
        self.is_domestic_futures = calendar is not None and calendar.__class__.__module__.endswith('.china_futures')
        if start_date is None or end_date is None:
            raise ValueError('Declare start_date and end_date for calendar coverage')
        self.start, self.end = _date(start_date), _date(end_date)
        if self.start > self.end:
            raise ValueError('Calendar start must not follow end')
        default_reference = self.calendar_id
        if calendar is not None and self.calendar_id in ('CME Globex Equity', 'CME_Equity'):
            default_reference = 'NYSE'
        elif self.is_domestic_futures and not calendar.name.startswith(('CFFEX', 'GFEX')):
            default_reference = calendar.name.split('_')[0] + '_Day'
        self.rth_reference_calendar = rth_reference_calendar or default_reference
        raw = Path(overrides_path).read_bytes()
        spec = json.loads(raw)
        if spec.get('schema_version') != 1 or not isinstance(spec.get('calendars'), dict):
            raise ValueError('Calendar overrides require schema_version 1 and calendars')
        overrides = spec['calendars']
        # Padding identifies uncovered overnight owners and incomplete weeks.
        left, right = self.start - pd.Timedelta(days=7), self.end + pd.Timedelta(days=7)
        if custom_sessions is None:
            eth, self.timezone = _schedule(self.calendar_id, left, right, overrides)
        else:
            eth, self.timezone = _custom_schedule(custom_sessions, left, right)
        reference = eth
        if self.rth_reference_calendar != self.calendar_id:
            reference, _ = _schedule(self.rth_reference_calendar, left, right, overrides)
        self._set_schedule(eth, reference, daily_anchor)
        payload = {'calendar': self.calendar_id, 'reference': self.rth_reference_calendar,
                   'start': str(self.start), 'end': str(self.end), 'timezone': self.timezone,
                   'pmc': pmc.__version__, 'exchange_calendars': version('exchange-calendars'),
                   'tzdata': version('tzdata'), 'overrides_sha256': hashlib.sha256(raw).hexdigest(),
                   'sessions': {v: [(item.session_date, [(a.isoformat(), b.isoformat())
                                                       for a, b in item.segments])
                                    for item in days.values()] for v, days in self._sessions.items()}}
        if daily_anchor != 'close':
            payload['daily_anchor'] = daily_anchor
        if custom_sessions is not None:
            payload['custom_sessions'] = [(day, [(a.isoformat(), b.isoformat()) for a, b in segments])
                                          for day, segments in eth.items()]
        self._calendar_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _set_schedule(self, eth: dict[str, tuple[Segment, ...]], reference: dict[str, tuple[Segment, ...]],
                      daily_anchor: str) -> None:
        self._daily_anchor = daily_anchor if daily_anchor in ('open', 'close') else _wall_time(daily_anchor)
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
        self._padded_sessions: dict[View, dict[str, Session]] = {}
        self._intervals: dict[View, list[tuple[pd.Timestamp, pd.Timestamp, Session]]] = {}
        self._opens: dict[View, list[pd.Timestamp]] = {}
        for view, schedule in (('ETH', eth), ('RTH', rth)):
            padded = {day: Session(day, segments) for day, segments in schedule.items()}
            self._padded_sessions[view] = padded
            self._sessions[view] = {day: item for day, item in padded.items()
                                    if self.start <= pd.Timestamp(day) <= self.end}
            intervals = sorted(((a, b, item) for item in padded.values() for a, b in item.segments),
                               key=lambda interval: interval[:2])
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError('Calendar sessions must not overlap')
            self._intervals[view] = intervals
            self._opens[view] = [a for a, _, _ in intervals]
        self._set_dates(self._sessions['ETH'], self.start, self.end, self.calendar_id, allow_empty=True)

    def _set_dates(self, sessions: Iterable[str | pd.Timestamp], start: str | pd.Timestamp,
                   end: str | pd.Timestamp, calendar_id: str, *, allow_empty: bool = False) -> None:
        self.start, self.end = _day(start), _day(end)
        days = pd.DatetimeIndex([_day(day) for day in sessions], dtype='datetime64[ns]')
        if (self.start > self.end or (days.empty and not allow_empty)
                or days.has_duplicates or not days.is_monotonic_increasing):
            raise ValueError('Calendar needs nonempty, sorted, distinct sessions and ordered coverage')
        if days.min() < self.start or days.max() > self.end:
            raise ValueError('Sessions extend outside declared coverage')
        self._session_dates = days
        self.calendar_id = calendar_id
        self.fingerprint = hashlib.sha256(json.dumps([
            calendar_id, str(self.start), str(self.end), days.strftime('%Y-%m-%d').tolist()
        ]).encode()).hexdigest()

    @classmethod
    def from_sessions(cls, sessions: Iterable[str | pd.Timestamp], start: str | pd.Timestamp,
                      end: str | pd.Timestamp, calendar_id: str = 'explicit') -> SessionCalendar:
        """Build a verified date snapshot when no trading clock was supplied."""
        calendar = cls.__new__(cls)
        calendar._set_dates(sessions, start, end, calendar_id)
        return calendar

    @classmethod
    def from_pmc(cls, start: str | pd.Timestamp = '2016-01-04', end: str | pd.Timestamp = '2026-09-30', *,
                 calendar_id: str = VERIFIED_XSHG, template_calendar: str = 'XSHG',
                 overrides_path: str | Path | None = None) -> SessionCalendar:
        """Build the cross-checked XSHG dates and their session clock."""
        options = {} if overrides_path is None else {'overrides_path': overrides_path}
        calendar = cls(template_calendar, start, end, **options)
        calendar._set_dates(calendar._session_dates, calendar.start, calendar.end, calendar_id)
        return calendar

    @classmethod
    def from_verified_sessions(
        cls,
        sessions: Iterable[str | pd.Timestamp],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        calendar_id: str = VERIFIED_XSHG,
        template_calendar: str = 'XSHG',
        overrides_path: str | Path = DEFAULT_OVERRIDES,
    ) -> SessionCalendar:
        """Build a SessionCalendar clock restricted to verified DB sessions and coverage bounds."""
        start_day, end_day = _day(start), _day(end)
        verified_days = pd.DatetimeIndex([_day(day) for day in sessions], dtype='datetime64[ns]')
        if (
            start_day > end_day
            or verified_days.empty
            or verified_days.has_duplicates
            or not verified_days.is_monotonic_increasing
            or verified_days.min() < start_day
            or verified_days.max() > end_day
        ):
            raise ValueError('Verified sessions must be nonempty, sorted, distinct, and within coverage bounds')

        raw = Path(overrides_path).read_bytes()
        spec = json.loads(raw)
        if spec.get('schema_version') != 1 or not isinstance(spec.get('calendars'), dict):
            raise ValueError('Calendar overrides require schema_version 1 and calendars')
        left, right = start_day - pd.Timedelta(days=7), end_day + pd.Timedelta(days=7)
        eth, tz_name = _schedule(template_calendar, left, right, spec['calendars'])

        verified_strs = set(verified_days.strftime('%Y-%m-%d'))
        covered_strs = set(pd.date_range(start_day, end_day).strftime('%Y-%m-%d'))
        for day_str in list(eth):
            if day_str in covered_strs and day_str not in verified_strs:
                eth.pop(day_str, None)
        tz_info = ZoneInfo(tz_name)
        for day in verified_days:
            day_str = day.strftime('%Y-%m-%d')
            if day_str not in eth or not eth[day_str]:
                if template_calendar == 'XSHG':
                    eth[day_str] = (
                        (_local_instant(day, time(9, 30), tz_info), _local_instant(day, time(11, 30), tz_info)),
                        (_local_instant(day, time(13, 0), tz_info), _local_instant(day, time(15, 0), tz_info)),
                    )
                else:
                    raise ValueError(f'Verified session {day_str} missing clock segments for {template_calendar!r}')
        eth = dict(sorted(eth.items()))

        calendar = cls.__new__(cls)
        calendar.calendar_id = calendar_id
        calendar.rth_reference_calendar = calendar_id
        calendar.timezone = tz_name
        calendar.is_domestic_futures = False
        calendar.start, calendar.end = start_day, end_day
        calendar._set_schedule(eth, eth, 'close')
        calendar._set_dates(verified_days, start_day, end_day, calendar_id)
        payload = {
            'calendar': calendar_id,
            'reference': calendar_id,
            'start': str(start_day),
            'end': str(end_day),
            'timezone': tz_name,
            'overrides_sha256': hashlib.sha256(raw).hexdigest(),
            'sessions': {
                view: [
                    (item.session_date, [(a.isoformat(), b.isoformat()) for a, b in item.segments])
                    for item in days.values()
                ]
                for view, days in calendar._sessions.items()
            },
        }
        calendar._calendar_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return calendar

    def _require_clock(self) -> None:
        if not hasattr(self, '_intervals'):
            raise ValueError('Calendar contains verified dates only. Supply a session clock for UTC queries.')

    def validate_coverage(self, start: str | pd.Timestamp,
                          end: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        start, end = _day(start), _day(end)
        if start > end:
            raise ValueError('Calendar start must not follow end')
        if start < self.start or end > self.end:
            raise CalendarCoverageError(
                f'Calendar {self.calendar_id} covers {self.start.date()}..{self.end.date()}; '
                f'requested {start.date()}..{end.date()}. Import verified coverage first.'
            )
        return start, end

    def sessions(self, start: str | pd.Timestamp | None = None,
                 end: str | pd.Timestamp | None = None) -> pd.DatetimeIndex:
        start, end = self.validate_coverage(
            self.start if start is None else start, self.end if end is None else end)
        return self._session_dates[(self._session_dates >= start) & (self._session_dates <= end)].copy()

    def _window(self, anchor: str | pd.Timestamp, count: int,
                include_anchor: bool, previous: bool) -> pd.DatetimeIndex:
        if not isinstance(include_anchor, bool):
            raise ValueError('include_anchor must be a boolean')
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError('Session count must be a non-negative integer')
        anchor, _ = self.validate_coverage(anchor, anchor)
        if count == 0:
            return self._session_dates[:0].copy()
        if previous:
            stop = self._session_dates.searchsorted(anchor, side='right' if include_anchor else 'left')
            result = self._session_dates[max(0, stop-count):stop]
        else:
            start = self._session_dates.searchsorted(anchor, side='left' if include_anchor else 'right')
            result = self._session_dates[start:start+count]
        if len(result) != count:
            raise CalendarCoverageError(f'Calendar has fewer than {count} sessions '
                                        f'{"before" if previous else "after"} {anchor.date()}')
        return result.copy()

    def previous_sessions(self, anchor: str | pd.Timestamp, count: int, *,
                          include_anchor: bool = False) -> pd.DatetimeIndex:
        return self._window(anchor, count, include_anchor, True)

    def next_sessions(self, anchor: str | pd.Timestamp, count: int, *,
                      include_anchor: bool = False) -> pd.DatetimeIndex:
        return self._window(anchor, count, include_anchor, False)

    def align(self, frame: pd.DataFrame | pd.Series, start: str | pd.Timestamp | None = None,
              end: str | pd.Timestamp | None = None) -> pd.DataFrame | pd.Series:
        """Restore missing market rows as NaN; reject non-session observations."""
        if frame.empty:
            return frame.copy()
        dates = pd.DatetimeIndex(frame.index)
        if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()) or dates.has_duplicates:
            raise ValueError('Daily panel index must contain distinct normalized local dates')
        sessions = self.sessions(dates.min() if start is None else start,
                                 dates.max() if end is None else end)
        if not dates.isin(self._session_dates).all():
            raise ValueError('Panel contains observations on non-session dates')
        frame = frame.set_axis(dates, axis=0, copy=False)
        return frame.reindex(sessions.rename(frame.index.name))

    @property
    def calendar_hash(self) -> str:
        self._require_clock()
        return self._calendar_hash

    def _view(self, view: View) -> None:
        if view not in ('ETH', 'RTH'):
            raise ValueError('view must be ETH or RTH')
        self._require_clock()

    def _validate_date(self, day: pd.Timestamp) -> None:
        if not self.start <= day <= self.end:
            raise CalendarCoverageError(f'Calendar covers {self.start.date()}..{self.end.date()}; requested {day.date()}')

    def session_at(self, ts_utc: datetime, view: View = 'ETH', *,
                   frequency: str | Frequency | None = None) -> Session | None:
        """Resolve interval ownership or an exact daily/weekly grid anchor."""
        _utc(ts_utc, 'ts_utc')
        self._view(view)
        ts = pd.Timestamp(ts_utc)
        if frequency is not None:
            freq = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
            if not freq.is_valid:
                raise ValueError(f'Invalid frequency: {freq.raw!r}')
            if freq.kind in ('session', 'week'):
                item = self._session_at_anchor(ts, view)
                if item is None:
                    # Retain coverage errors for queries outside the grid's declared scope.
                    self.session_at(ts, view)
                elif freq.kind == 'week':
                    week = self._week_on(item.session_date, view)
                    if not week or item != week[-1]:
                        return None
                return item
        # Use ETH ownership even for RTH queries in the overnight edge of coverage.
        intervals = self._intervals['ETH']
        pos = bisect_right(self._opens['ETH'], ts) - 1
        owner = intervals[pos][2] if pos >= 0 and ts < intervals[pos][1] else None
        day = pd.Timestamp(owner.session_date) if owner else ts.tz_convert(self.timezone).tz_localize(None).normalize()
        self._validate_date(day)
        intervals = self._intervals[view]
        pos = bisect_right(self._opens[view], ts) - 1
        return intervals[pos][2] if pos >= 0 and ts < intervals[pos][1] else None

    @cached_property
    def _eth_session_index(self) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray, np.ndarray]:
        intervals = self._intervals['ETH']
        opens = pd.DatetimeIndex(self._opens['ETH'], tz='UTC')
        closes = pd.DatetimeIndex([close for _, close, _ in intervals], tz='UTC')
        labels = np.array([session.session_date for _, _, session in intervals], dtype=object)
        covered = np.array([label in self._sessions['ETH'] for label in labels], dtype=bool)
        return opens, closes, labels, covered

    def session_dates_at(self, timestamps: pd.DatetimeIndex) -> pd.Index:
        """Resolve ETH ownership in input order for validated UTC Bar starts."""
        self._view('ETH')
        opens, closes, labels, covered = self._eth_session_index
        dates = np.full(len(timestamps), None, dtype=object)
        owned = np.zeros(len(timestamps), dtype=bool)
        outside = owned.copy()
        if len(opens):
            positions = opens.searchsorted(timestamps, side='right') - 1
            owned = (positions >= 0) & (timestamps < closes.take(positions))
            dates[owned] = labels[positions[owned]]
            outside = owned & ~covered[positions]
        local_days = timestamps[~owned].tz_convert(self.timezone).tz_localize(None).normalize()
        outside[~owned] = (local_days < self.start) | (local_days > self.end)
        if outside.any():
            # Keep the scalar coverage error and report the first offending timestamp.
            self.session_at(timestamps[outside.argmax()])
        return pd.Index(dates, dtype=object)

    def _session_at_anchor(self, ts: pd.Timestamp, view: View) -> Session | None:
        if isinstance(self._daily_anchor, time):
            item = self._sessions[view].get(ts.tz_convert(self.timezone).strftime('%Y-%m-%d'))
        else:
            # At a shared boundary, the close belongs to the previous interval.
            seek = bisect_left if self._daily_anchor == 'close' else bisect_right
            pos = seek(self._opens[view], ts) - 1
            item = self._intervals[view][pos][2] if pos >= 0 else None
        if item is None or not item.segments or item.session_date not in self._sessions[view]:
            return None
        try:
            return item if self._session_anchor(item) == ts else None
        except ValueError:
            # The declared wall time may be outside this Session or invalid under DST.
            return None

    def _session_anchor(self, session: Session) -> pd.Timestamp:
        if self._daily_anchor == 'close':
            return session.segments[-1][1]
        if self._daily_anchor == 'open':
            return session.segments[0][0]
        anchor = _local_instant(pd.Timestamp(session.session_date), self._daily_anchor, ZoneInfo(self.timezone))
        if not (any(left <= anchor < right for left, right in session.segments)
                or anchor == session.segments[-1][1]):
            raise ValueError(f'Daily anchor is outside the trading session {session.session_date}')
        return anchor

    def _week_on(self, session_id: str, view: View) -> list[Session]:
        day = pd.Timestamp(session_id)
        monday = day - pd.Timedelta(days=day.weekday())
        schedule = self._padded_sessions[view]
        week = [schedule[label] for label in pd.date_range(monday, periods=7).strftime('%Y-%m-%d')
                if label in schedule and schedule[label].segments]
        if any(item.session_date not in self._sessions[view] for item in week):
            return []
        return week

    def bar_grid(self, frequency: str | Frequency, start: str | pd.Timestamp | None = None,
                 end: str | pd.Timestamp | None = None, *, view: View = 'ETH',
                 bucket_strategy: str | None = None) -> pd.DataFrame:
        """Return UTC grid keys within the inclusive range of session labels."""
        parsed = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
        if parsed.kind == 'session' and bucket_strategy is None:
            bucket_strategy = 'session'
        return BarSchedule(self, parsed, view=view, bucket_strategy=bucket_strategy).grid(start, end)

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
            if schedule.frequency.kind == 'week' or any(a < end and b > start for a, b in session.segments):
                grid = pd.DatetimeIndex(schedule.for_session(session).start_utc)
                result = result.append(grid[(grid >= start) & (grid < end)])
        return result


class TradingCalendar(SessionCalendar):
    """Legacy date-only constructor and PMC snapshot entry point."""

    def __init__(self, sessions: Iterable[str | pd.Timestamp], start: str | pd.Timestamp,
                 end: str | pd.Timestamp, calendar_id: str = 'explicit') -> None:
        self._set_dates(sessions, start, end, calendar_id)

    def __setstate__(self, state: dict) -> None:
        # Price panels can carry pickled calendars in their pandas attrs.
        if '_sessions' in state:
            state['_session_dates'] = state.pop('_sessions')
        self.__dict__.update(state)

    @classmethod
    def from_pmc(cls, start: str = '2016-01-04', end: str = '2026-09-30', *,
                 calendar_id: str = VERIFIED_XSHG,
                 overrides_path: str | Path | None = None) -> TradingCalendar:
        core = SessionCalendar.from_pmc(start, end, calendar_id=calendar_id, overrides_path=overrides_path)
        return cls(core.sessions(), start, end, calendar_id)


def load_trading_calendar(connection=None, *, duckdb_path=None, bindings=None,
                          config_path: str | Path | None = None,
                          require_bound_view: bool = False) -> TradingCalendar:
    """Read the configured calendar table without deriving dates from prices.

    A supplied connection is borrowed; its creator is responsible for version
    binding. Extra duckdb_path/bindings and environment bindings are not resolved
    in this mode. Otherwise, connect_input opens and binds an owned connection
    using an explicit path or bound input_db, and this function closes it.
    For a frozen input connection, require_bound_view confines lookup to the
    current catalog where connect_input creates the exact version views.

    Coverage is inferred from the first and last stored dates. Interior gaps
    raise; missing edge rows may shrink coverage without a load-time error.
    The original import receipt's coverage boundaries are not checked here.
    """
    from utils.data_store import binding_contract, connect_input, quote, table_name
    spec = json.loads(Path(config_path or DEFAULT_CONFIG).read_text())
    if set(spec) != {'table', 'calendar_id'} or not all(isinstance(v, str) and v for v in spec.values()):
        raise ValueError('Calendar config requires table and calendar_id')
    owned = connection is None
    if owned:
        contract = binding_contract(bindings)
        path = duckdb_path or contract.get('input_db')
        if not path:
            raise ValueError('Supply a calendar connection, duckdb_path, or bound input_db')
        connection = connect_input(path, bindings=contract)
    try:
        table = table_name(spec["table"])
        if require_bound_view:
            catalog = connection.execute('SELECT current_database()').fetchone()[0]
            table = f'{quote(catalog)}.{table}'
        rows = connection.execute(
            f'SELECT date, is_open FROM {table} '
            'WHERE calendar_id = ? ORDER BY date', [spec['calendar_id']]).fetchdf()
    finally:
        if owned:
            connection.close()
    if rows.empty or rows.isna().any().any():
        raise ValueError('Verified calendar is empty or contains null values')
    dates = pd.DatetimeIndex(pd.to_datetime(rows.date)).as_unit('ns')
    expected = pd.date_range(dates.min(), dates.max())
    if not dates.equals(dates.normalize()) or not dates.equals(expected) or rows.is_open.dtype != bool:
        raise ValueError('Calendar must contain one DATE/BOOLEAN row per covered natural date')
    return TradingCalendar(dates[rows.is_open.to_numpy()], dates.min(), dates.max(), spec['calendar_id'])
