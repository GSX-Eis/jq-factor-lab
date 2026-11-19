"""Offline Bar construction and quality audit on an explicitly owned connection."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Sequence
import uuid

import duckdb
import pandas as pd

from data_pipeline.loaders._transactions import import_transaction
from data_pipeline.loaders.market_bar_importer import calendar_name, store_instrument_metadata
from .catalog import (BarCatalog, CatalogEntry, create_bar_table, identifier,
                      register_entry, table_exists)
from .quality import scan_quality, write_quality
from .source import BarSource, _instrument_ids
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule, Frequency


def code_version() -> str:
    from utils.code_version import resolve_code_version
    return resolve_code_version()[0]


def instrument_calendar(metadata: dict, start: str, end: str) -> SessionCalendar:
    return SessionCalendar(calendar_name(metadata), start, end,
                           rth_reference_calendar=metadata['rth_reference_calendar'])


def slot_mapping(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    boundaries = set(source.start_utc) | set(source.end_utc)
    if not (set(target.start_utc) | set(target.end_utc)) <= boundaries:
        raise ValueError('Target boundaries must fall on source boundaries')
    rows = []
    for slot in target.itertuples():
        children = source.loc[source.start_utc.ge(slot.start_utc) & source.end_utc.le(slot.end_utc)]
        if children.empty:
            raise ValueError('Target boundaries do not contain source slots')
        rows.extend((row.start_utc, slot.start_utc, len(children)) for row in children.itertuples())
    if len(rows) != len(source):
        raise ValueError('Target boundaries split or omit source slots')
    return pd.DataFrame(rows, columns=['source_start', 'target_start', 'n_expected_bars'])


def _aggregate(connection: duckdb.DuckDBPyConnection, raw: pd.DataFrame,
               mapping: pd.DataFrame, instrument: str, target: str,
               build_id: str, version: int) -> pd.DataFrame:
    # Only the schedule mapping is generated in Python; all OHLCV reductions are SQL.
    if 'is_complete' not in raw:
        raw = raw.assign(is_complete=True)
    for column in ('factor', 'money', 'open_interest'):
        if column not in raw:
            raw = raw.assign(**{column: float('nan')})
    factors = raw.factor.dropna().unique()
    if len(factors) > 1:
        raise ValueError('Source factor must be constant within a Session')
    connection.register('_bar_source_input', raw)
    connection.register('_bar_slot_map', mapping)
    try:
        return connection.execute('''SELECT ? AS instrument_id, m.target_start AS timestamp_utc,
            ? AS frequency, coalesce(first(s.open ORDER BY s.timestamp_utc), 'NaN'::DOUBLE) AS open,
            coalesce(max(s.high), 'NaN'::DOUBLE) AS high, coalesce(min(s.low), 'NaN'::DOUBLE) AS low,
            coalesce(last(s.close ORDER BY s.timestamp_utc), 'NaN'::DOUBLE) AS close,
            sum(s.volume) AS volume, 'offline_build' AS source, ? AS data_version,
            sum(s.money) AS money, CAST(? AS DOUBLE) AS factor,
            count(*) AS n_source_bars, first(m.n_expected_bars) AS n_expected_bars,
            count(*)=first(m.n_expected_bars) AND bool_and(coalesce(s.is_complete, true)) AS is_complete,
            last(s.open_interest ORDER BY s.timestamp_utc) AS open_interest
            FROM _bar_source_input s JOIN _bar_slot_map m ON s.timestamp_utc=m.source_start
            GROUP BY m.target_start ORDER BY m.target_start''',
            [instrument, target, f'{build_id}:v{version}', float(factors[0]) if len(factors) else None]).fetchdf()
    finally:
        connection.unregister('_bar_source_input')
        connection.unregister('_bar_slot_map')


def _metadata(reader: BarSource, instruments: list[str], frequency: str) -> pd.DataFrame:
    frames = []
    catalog = BarCatalog(reader.connection)
    for instrument in instruments:
        exists = table_exists(reader.connection, 'instrument_metadata') and reader.connection.execute(
            'SELECT count(*) FROM instrument_metadata WHERE instrument_id=?', [instrument]).fetchone()[0]
        if exists:
            frames.append(reader.get_instrument_metadata([instrument]))
        else:
            entry = catalog.resolve(frequency, instrument)
            if entry.timestamp_semantics == 'UTC_bar_start':
                raise ValueError(f'Missing instrument metadata: {instrument}')
            frames.append(reader._legacy_metadata([instrument], 'Asia/Shanghai'))
    return pd.concat(frames, ignore_index=True)


def build_bars(connection: duckdb.DuckDBPyConnection, *, source: str,
               targets: Sequence[str], instruments: Sequence[str], start: str, end: str,
               bucket_strategy: str | None = None, check_only: bool = False) -> dict:
    ids = _instrument_ids(instruments)
    source_frequency = Frequency.parse(source)
    if not source_frequency.is_valid:
        raise ValueError(f'Invalid source frequency: {source!r}')
    if isinstance(targets, str) or not targets:
        raise ValueError('targets must be a nonempty sequence')
    parsed_targets = [Frequency.parse(target) for target in targets]
    if any(not target.is_valid for target in parsed_targets):
        raise ValueError('Invalid target frequency')
    if len({target.canonical for target in parsed_targets}) != len(targets):
        raise ValueError('Duplicate target frequency')
    if any(target.duration <= source_frequency.duration for target in parsed_targets):
        raise ValueError('Target frequency must be coarser than source frequency')
    # SessionCalendar validates date-only, ordered bounds.
    reader, catalog = BarSource(connection), BarCatalog(connection)
    metadata = _metadata(reader, ids, source_frequency.canonical)
    revision = code_version()
    jobs, comparisons = [], []
    for meta in metadata.to_dict('records'):
        instrument = meta['instrument_id']
        entry = catalog.resolve(source_frequency.canonical, instrument, asset_class=meta['asset_class'])
        cal = instrument_calendar(meta, start, end)
        # Validate explicit domestic policy even on a closed range.
        BarSchedule(cal, source_frequency, bucket_strategy=bucket_strategy)
        session_inputs = {}
        if len(cal.sessions()):
            first = cal.session_on(cal.sessions()[0].strftime('%Y-%m-%d')).segments[0][0]
            last = cal.session_on(cal.sessions()[-1].strftime('%Y-%m-%d')).segments[-1][1]
            source_bars = reader.read_bars([instrument], source_frequency.canonical,
                start_utc=first, end_utc=last-pd.Timedelta(nanoseconds=1), source_table=entry.table_name)
            for day in cal.sessions():
                session_id = day.strftime('%Y-%m-%d')
                session_cal = instrument_calendar(meta, session_id, session_id)
                source_schedule = BarSchedule(session_cal, source_frequency,
                                               bucket_strategy=entry.bucket_strategy or bucket_strategy)
                slots = source_schedule.for_session(session_id)
                raw = source_bars.loc[source_bars.timestamp_utc.ge(slots.start_utc.min()) &
                                      source_bars.timestamp_utc.lt(slots.end_utc.max())].copy()
                session_inputs[session_id] = (session_cal, slots, raw)
        for target in parsed_targets:
            target_entry = None
            try:
                target_entry = catalog.resolve(target.canonical, instrument, asset_class=meta['asset_class'])
            except LookupError:
                pass
            if target_entry and target_entry.kind == 'native' and not check_only:
                raise ValueError(f'Cannot derive native frequency={target.canonical}; use --check-only')
            for day in cal.sessions():
                session_id = day.strftime('%Y-%m-%d')
                # A session hash is independent of the requested batch range.
                session_cal, src_slots, raw = session_inputs[session_id]
                dst_schedule = BarSchedule(session_cal, target, bucket_strategy=bucket_strategy)
                dst_slots = dst_schedule.for_session(session_id)
                mapping = slot_mapping(src_slots, dst_slots)
                if not raw.timestamp_utc.isin(src_slots.start_utc).all():
                    raise ValueError('Source contains off-grid bars')
                digest = hashlib.sha256(pd.util.hash_pandas_object(raw, index=False).values.tobytes()
                    + pd.util.hash_pandas_object(mapping, index=False).values.tobytes()
                    + (session_cal.calendar_hash + revision + entry.table_name +
                       dst_schedule.bucket_strategy).encode()).hexdigest()
                old = None
                if table_exists(connection, 'bar_build_log'):
                    old = connection.execute('SELECT source_hash, version FROM bar_build_log '
                        'WHERE frequency=? AND instrument_id=? AND session_id=?',
                        [target.canonical, instrument, session_id]).fetchone()
                if not check_only and old and old[0] == digest:
                    continue
                version = old[1]+1 if old else 1
                build_id = uuid.uuid4().hex
                aggregated = _aggregate(connection, raw, mapping, instrument, target.canonical, build_id, version)
                if check_only:
                    if target_entry is not None:
                        # Legacy daily dates are local midnight labels, not executable starts.
                        if target_entry.timestamp_semantics == 'local_midnight_day_identifier_pit':
                            bound = day.tz_localize('Asia/Shanghai').tz_convert('UTC')
                            native = reader.read_bars([instrument], target.canonical, start_utc=bound,
                                end_utc=bound, source_table=target_entry.table_name)
                        else:
                            native = reader.read_bars([instrument], target.canonical,
                                start_utc=dst_slots.start_utc.min(),
                                end_utc=dst_slots.end_utc.max()-pd.Timedelta(nanoseconds=1),
                                source_table=target_entry.table_name)
                    else:
                        native = pd.DataFrame()
                    comparisons.append(dict(instrument_id=instrument, frequency=target.canonical,
                        session_id=session_id, missing_days=[session_id] if native.empty or aggregated.empty else [],
                        close_difference=(None if native.empty or aggregated.empty else
                                          float(aggregated.close.iloc[-1]-native.close.iloc[-1])),
                        volume_difference=(None if native.empty or aggregated.empty else
                                           float(aggregated.volume.sum()-native.volume.sum()))))
                else:
                    jobs.append((meta, entry, dst_schedule, dst_slots, aggregated, digest, version, build_id))
    if check_only or not jobs:
        return dict(rebuilt_sessions=0, comparisons=comparisons, check_only=check_only)
    with import_transaction(connection):
        store_instrument_metadata(connection, metadata)
        connection.execute('''CREATE TABLE IF NOT EXISTS bar_build_log (
            frequency VARCHAR, instrument_id VARCHAR, session_id VARCHAR, source_hash VARCHAR,
            version BIGINT, build_id VARCHAR, source_frequency VARCHAR, source_version VARCHAR,
            bucket_strategy VARCHAR, calendar_hash VARCHAR, code_version VARCHAR,
            start_utc TIMESTAMPTZ, end_utc TIMESTAMPTZ,
            PRIMARY KEY (frequency, instrument_id, session_id))''')
        for meta, entry, schedule, slots, frame, digest, version, build_id in jobs:
            frequency, instrument = schedule.frequency.canonical, meta['instrument_id']
            session_id = slots.session_id.iloc[0]
            name = create_bar_table(connection, frequency)
            connection.execute(f'DELETE FROM {identifier(name)} WHERE instrument_id=? '
                'AND timestamp_utc>=? AND timestamp_utc<?',
                [instrument, slots.start_utc.min(), slots.end_utc.max()])
            connection.from_df(frame).insert_into(name)
            connection.execute('INSERT OR REPLACE INTO bar_build_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                [frequency, instrument, session_id, digest, version, build_id, source_frequency.canonical,
                 digest, schedule.bucket_strategy, schedule.calendar.calendar_hash, revision,
                 slots.start_utc.min(), slots.end_utc.max()])
            write_quality(connection, frequency, instrument, session_id, slots, frame)
            register_entry(connection, CatalogEntry(frequency, name, kind='derived', instrument_id=instrument,
                source_frequency=source_frequency.canonical, source_version=digest,
                bucket_strategy=schedule.bucket_strategy, calendar_hash=schedule.calendar.calendar_hash,
                build_id=build_id, code_version=revision, coverage_start=session_id, coverage_end=session_id,
                quality_status='scanned'))
    return dict(rebuilt_sessions=len(jobs), comparisons=[], check_only=False)


def audit_bars(connection: duckdb.DuckDBPyConnection, *, instruments: Sequence[str] | None = None,
               start: str | None = None, end: str | None = None,
               bucket_strategy: str | None = None) -> dict:
    """Audit existing native/derived/legacy tables without changing their schemas or rows."""
    catalog, reader = BarCatalog(connection), BarSource(connection)
    plans = []
    for entry in catalog.entries:
        if not table_exists(connection, entry.table_name):
            continue
        legacy = entry.timestamp_semantics != 'UTC_bar_start'
        column = 'stock_code' if legacy else 'instrument_id'
        ids = [row[0] for row in connection.execute(
            f'SELECT DISTINCT {column} FROM {identifier(entry.table_name)}').fetchall()]
        if entry.instrument_id != '*':
            ids = [value for value in ids if value == entry.instrument_id]
        if instruments is not None:
            ids = [value for value in ids if value in _instrument_ids(instruments)]
        for instrument in ids:
            if any(item[0].table_name == entry.table_name and item[1] == instrument for item in plans):
                continue
            metadata = (reader._legacy_metadata([instrument], 'Asia/Shanghai') if legacy
                        else reader.get_instrument_metadata([instrument]))
            raw = reader.read_bars([instrument], entry.frequency, source_table=entry.table_name)
            if raw.empty:
                continue
            meta = metadata.iloc[0].to_dict()
            padded = instrument_calendar(meta,
                (raw.timestamp_utc.min().tz_convert(meta['timezone'])-pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
                (raw.timestamp_utc.max().tz_convert(meta['timezone'])+pd.Timedelta(days=7)).strftime('%Y-%m-%d'))
            if entry.frequency == '1d' and legacy:
                first = raw.timestamp_utc.min().tz_convert(meta['timezone']).strftime('%Y-%m-%d')
                last = raw.timestamp_utc.max().tz_convert(meta['timezone']).strftime('%Y-%m-%d')
            else:
                owners = [padded.session_at(ts) for ts in raw.timestamp_utc]
                if any(owner is None for owner in owners):
                    raise ValueError('Cannot audit off-session bars')
                first, last = min(owner.session_date for owner in owners), max(owner.session_date for owner in owners)
            cal = instrument_calendar(meta, start or first, end or last)
            schedule = BarSchedule(cal, entry.frequency, bucket_strategy=bucket_strategy or entry.bucket_strategy)
            if entry.frequency == '1d' and legacy:
                # Quality uses executable session envelopes, retaining the price table's date identifiers.
                dates = raw.timestamp_utc.dt.tz_convert(meta['timezone']).dt.tz_localize(None).dt.normalize()
                selected = dates.between(cal.start, cal.end)
                raw, dates = raw.loc[selected].copy(), dates.loc[selected]
                nontrading = dates.loc[~dates.isin(cal.sessions())]
                if not nontrading.empty:
                    raise ValueError(f'Cannot audit daily bars on non-trading dates for {instrument}: '
                                     f'{sorted(nontrading.dt.strftime("%Y-%m-%d").unique())}')
                starts = {day: schedule.for_session(day.strftime('%Y-%m-%d')).start_utc.iloc[0]
                          for day in cal.sessions()}
                raw['timestamp_utc'] = dates.map(starts)
            plans.append((entry, instrument, schedule, raw))
    count = 0
    if not plans:
        return dict(audited_sessions=0)
    with import_transaction(connection):
        for entry, instrument, schedule, raw in plans:
            count += scan_quality(connection, instrument, schedule, raw)
            register_entry(connection, replace(entry, instrument_id=instrument, quality_status='scanned',
                bucket_strategy=schedule.bucket_strategy, calendar_hash=schedule.calendar.calendar_hash,
                coverage_start=schedule.calendar.start.strftime('%Y-%m-%d'),
                coverage_end=schedule.calendar.end.strftime('%Y-%m-%d')))
    return dict(audited_sessions=count)
