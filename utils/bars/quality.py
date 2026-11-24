"""Session quality evidence and one aggregated warning on every overlapping read."""
from __future__ import annotations

import logging
from uuid import uuid4
import warnings

import duckdb
import pandas as pd

from .catalog import table_exists
from utils.time_axis import BarSchedule

# Keep existing logger filters and handlers working.
LOGGER = logging.getLogger('utils.bar_quality')
_UNSCANNED: set[tuple[str, str]] = set()
_QUALITY_COLUMNS = (
    'frequency', 'instrument_id', 'session_id', 'n_expected_bars', 'n_actual_bars',
    'status', 'incomplete_buckets', 'start_utc', 'end_utc',
)


class BarDataQualityWarning(UserWarning):
    pass


def write_quality(connection: duckdb.DuckDBPyConnection, frequency: str, instrument: str,
                  session_id: str, slots: pd.DataFrame, bars: pd.DataFrame) -> None:
    _create_quality_table(connection)
    connection.execute('INSERT OR REPLACE INTO bar_quality VALUES (?,?,?,?,?,?,?,?,?)',
                       quality_row(frequency, instrument, session_id, slots, bars))


def quality_row(frequency: str, instrument: str, session_id: str,
                slots: pd.DataFrame, bars: pd.DataFrame) -> tuple:
    present = bars.loc[bars.timestamp_utc.isin(slots.start_utc)]
    incomplete = (int((~present.is_complete.fillna(False)).sum())
                  if 'is_complete' in present else 0)
    status = ('missing_session' if present.empty else
              'partial' if len(present) < len(slots) or incomplete else 'complete')
    return (frequency, instrument, session_id, len(slots), len(present), status,
            incomplete + (len(slots)-len(present) if not present.empty else 0),
            slots.start_utc.min(), slots.end_utc.max())


def _create_quality_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute('''CREATE TABLE IF NOT EXISTS bar_quality (
        frequency VARCHAR, instrument_id VARCHAR, session_id VARCHAR,
        n_expected_bars BIGINT, n_actual_bars BIGINT, status VARCHAR,
        incomplete_buckets BIGINT, start_utc TIMESTAMPTZ, end_utc TIMESTAMPTZ,
        PRIMARY KEY (frequency, instrument_id, session_id))''')


def write_quality_rows(connection: duckdb.DuckDBPyConnection,
                       rows: pd.DataFrame | list[tuple]) -> None:
    if len(rows) == 0:
        return
    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame.from_records(rows, columns=_QUALITY_COLUMNS)
    _create_quality_table(connection)
    batch_name = f'_bar_quality_{uuid4().hex}'
    batch = rows.loc[:, list(_QUALITY_COLUMNS)].copy()
    # Match scalar binding by dropping submicrosecond precision before the scan.
    for name in ('start_utc', 'end_utc'):
        batch[name] = batch[name].dt.as_unit('us')
    connection.register(batch_name, batch)
    try:
        connection.execute(f'INSERT OR REPLACE INTO bar_quality SELECT * FROM {batch_name}')
    finally:
        connection.unregister(batch_name)


def scan_quality(connection: duckdb.DuckDBPyConnection, instrument: str,
                 schedule: BarSchedule, bars: pd.DataFrame, *,
                 grid: pd.DataFrame | None = None) -> int:
    slots = schedule.session_slots() if grid is None else grid
    if not slots.empty:
        rows = slots.groupby('session_id', sort=False).agg(
            n_expected_bars=('start_utc', 'size'), start_utc=('start_utc', 'min'), end_utc=('end_utc', 'max'),
        )
        matched = bars.loc[bars.timestamp_utc.isin(slots.start_utc)]
        observed = pd.DataFrame(dict(
            timestamp_utc=pd.to_datetime(matched.timestamp_utc, utc=True),
            incomplete=(~matched.is_complete.fillna(False) if 'is_complete' in matched else False),
        ))
        # Match each observed row at most once per Session, as the scalar isin did.
        present = slots[['session_id', 'start_utc']].drop_duplicates().merge(
            observed, left_on='start_utc', right_on='timestamp_utc',
        )
        counts = present.groupby('session_id', sort=False).agg(
            n_actual_bars=('incomplete', 'size'), incomplete=('incomplete', 'sum'),
        )
        actual = counts.n_actual_bars.reindex(rows.index, fill_value=0)
        incomplete = counts.incomplete.reindex(rows.index, fill_value=0)
        missing = actual.eq(0)
        gap = rows.n_expected_bars - actual
        rows['n_actual_bars'] = actual
        rows['status'] = 'complete'
        rows.loc[gap.gt(0) | incomplete.ne(0), 'status'] = 'partial'
        rows.loc[missing, 'status'] = 'missing_session'
        rows['incomplete_buckets'] = (incomplete + gap).where(~missing, 0)
        rows['frequency'] = schedule.frequency.canonical
        rows['instrument_id'] = instrument
        write_quality_rows(connection, rows.reset_index())
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
