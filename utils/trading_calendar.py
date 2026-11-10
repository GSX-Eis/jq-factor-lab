"""Verified daily market sessions. Missing coverage is an error, never a weekday guess."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path

import pandas as pd

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / 'config' / 'trading_calendar.json'


class CalendarCoverageError(ValueError):
    """Requested dates/history are outside the verified calendar."""


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None:
        raise ValueError('Calendar dates must be finite, timezone-naive local market dates')
    return day.normalize()


class TradingCalendar:
    """An immutable snapshot of sessions AND the verified natural-date interval.

    Closed dates are known only inside coverage. This is a market calendar,
    not a guarantee of any individual security's observations or intraday bars.
    """

    def __init__(self, sessions, start, end, calendar_id='explicit'):
        self.start, self.end = _day(start), _day(end)
        days = pd.DatetimeIndex([_day(d) for d in sessions])
        if self.start > self.end or days.empty or days.has_duplicates or not days.is_monotonic_increasing:
            raise ValueError('Calendar needs nonempty, sorted, distinct sessions and ordered coverage')
        if days.min() < self.start or days.max() > self.end:
            raise ValueError('Sessions extend outside declared coverage')
        self._sessions = days
        self.calendar_id = calendar_id
        self.fingerprint = hashlib.sha256(json.dumps([
            calendar_id, str(self.start), str(self.end), days.strftime('%Y-%m-%d').tolist()
        ]).encode()).hexdigest()

    def validate_coverage(self, start, end):
        start, end = _day(start), _day(end)
        if start > end:
            raise ValueError('Calendar start must not follow end')
        if start < self.start or end > self.end:
            raise CalendarCoverageError(
                f'Calendar {self.calendar_id} covers {self.start.date()}..{self.end.date()}; '
                f'requested {start.date()}..{end.date()}. Import verified coverage first.'
            )
        return start, end

    def sessions(self, start=None, end=None):
        start, end = self.validate_coverage(
            self.start if start is None else start, self.end if end is None else end)
        return self._sessions[(self._sessions >= start) & (self._sessions <= end)].copy()

    def _window(self, anchor, count, include_anchor, previous):
        if not isinstance(include_anchor, bool):
            raise ValueError('include_anchor must be a boolean')
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError('Session count must be a non-negative integer')
        anchor, _ = self.validate_coverage(anchor, anchor)
        if count == 0:
            return self._sessions[:0].copy()
        if previous:
            stop = self._sessions.searchsorted(anchor, side='right' if include_anchor else 'left')
            result = self._sessions[max(0, stop-count):stop]
        else:
            start = self._sessions.searchsorted(anchor, side='left' if include_anchor else 'right')
            result = self._sessions[start:start+count]
        if len(result) != count:
            raise CalendarCoverageError(f'Calendar has fewer than {count} sessions '
                                        f'{"before" if previous else "after"} {anchor.date()}')
        return result.copy()

    def previous_sessions(self, anchor, count, *, include_anchor=False):
        return self._window(anchor, count, include_anchor, True)

    def next_sessions(self, anchor, count, *, include_anchor=False):
        return self._window(anchor, count, include_anchor, False)

    def align(self, frame, start=None, end=None):
        """Restore missing market rows as NaN; reject non-session observations."""
        if frame.empty:
            return frame.copy()
        dates = pd.DatetimeIndex(frame.index)
        if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()) or dates.has_duplicates:
            raise ValueError('Daily panel index must contain distinct normalized local dates')
        sessions = self.sessions(dates.min() if start is None else start,
                                 dates.max() if end is None else end)
        if not dates.isin(self._sessions).all():
            raise ValueError('Panel contains observations on non-session dates')
        frame = frame.set_axis(dates, axis=0, copy=False)
        return frame.reindex(sessions.rename(frame.index.name))


def load_trading_calendar(connection=None, *, duckdb_path=None, bindings=None, config_path=None):
    """Read the configured calendar table without deriving dates from prices.

    A supplied connection is borrowed; its creator is responsible for version
    binding. Extra duckdb_path/bindings and environment bindings are not resolved
    in this mode. Otherwise, connect_input opens and binds an owned connection
    using an explicit path or bound input_db, and this function closes it.

    Coverage is inferred from the first and last stored dates. Interior gaps
    raise; missing edge rows may shrink coverage without a load-time error.
    The original import receipt's coverage boundaries are not checked here.
    """
    from utils.data_store import binding_contract, connect_input, table_name
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
        rows = connection.execute(
            f'SELECT date, is_open FROM {table_name(spec["table"])} '
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
