"""Session quality evidence and one aggregated warning on every overlapping read."""
from __future__ import annotations

import logging
import warnings

import duckdb
import pandas as pd

from utils.bar_catalog import table_exists
from utils.time_axis import BarSchedule

LOGGER = logging.getLogger(__name__)
_UNSCANNED: set[tuple[str, str]] = set()


class BarDataQualityWarning(UserWarning):
    pass


def write_quality(connection: duckdb.DuckDBPyConnection, frequency: str, instrument: str,
                  session_id: str, slots: pd.DataFrame, bars: pd.DataFrame) -> None:
    connection.execute('''CREATE TABLE IF NOT EXISTS bar_quality (
        frequency VARCHAR, instrument_id VARCHAR, session_id VARCHAR,
        n_expected_bars BIGINT, n_actual_bars BIGINT, status VARCHAR,
        incomplete_buckets BIGINT, start_utc TIMESTAMPTZ, end_utc TIMESTAMPTZ,
        PRIMARY KEY (frequency, instrument_id, session_id))''')
    present = bars.loc[bars.timestamp_utc.isin(slots.start_utc)]
    incomplete = (int((~present.is_complete.fillna(False)).sum())
                  if 'is_complete' in present else 0)
    status = ('missing_session' if present.empty else
              'partial' if len(present) < len(slots) or incomplete else 'complete')
    connection.execute('INSERT OR REPLACE INTO bar_quality VALUES (?,?,?,?,?,?,?,?,?)',
        [frequency, instrument, session_id, len(slots), len(present), status,
         incomplete + (len(slots)-len(present) if not present.empty else 0),
         slots.start_utc.min(), slots.end_utc.max()])


def scan_quality(connection: duckdb.DuckDBPyConnection, instrument: str,
                 schedule: BarSchedule, bars: pd.DataFrame) -> int:
    for day in schedule.calendar.sessions():
        session_id = day.strftime('%Y-%m-%d')
        slots = schedule.for_session(session_id)
        if not slots.empty:
            write_quality(connection, schedule.frequency.canonical, instrument, session_id, slots, bars)
    return len(schedule.calendar.sessions())


def warn_quality(connection: duckdb.DuckDBPyConnection, frequency: str,
                 requests: list[tuple]) -> None:
    messages = []
    for instrument, entry, start, end in requests:
        if entry.quality_status == 'unscanned':
            databases = connection.execute('PRAGMA database_list').fetchall()
            key = (str(tuple(row[2] or f'memory:{id(connection)}' for row in databases)), entry.table_name)
            if key not in _UNSCANNED:
                LOGGER.info('Bar table %s quality is unscanned', entry.table_name)
                _UNSCANNED.add(key)
            continue
        if not table_exists(connection, 'bar_quality'):
            continue
        rows = connection.execute('SELECT session_id, status, incomplete_buckets FROM bar_quality '
            "WHERE frequency=? AND instrument_id=? AND status <> 'complete' "
            'AND start_utc <= ? AND end_utc > ? ORDER BY session_id',
            [frequency, instrument, end, start]).fetchall()
        if rows:
            missing = sum(status == 'missing_session' for _, status, _ in rows)
            partial = sum(count for _, _, count in rows)
            messages.append(f'frequency={frequency} instrument={instrument} missing_sessions={missing} '
                            f'incomplete_buckets={partial} examples={[row[0] for row in rows[:3]]}')
    if messages:
        message = '; '.join(messages)
        LOGGER.warning(message)
        warnings.warn(message, BarDataQualityWarning, stacklevel=3)
