"""Catalog-backed, read-only direct Bar access. No aggregation or frequency fallback."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from backtest_and_analysis.time_series.contracts import _utc
from utils.bar_catalog import BarCatalog, identifier, table_exists
from utils.bar_quality import warn_quality
from utils.market_bar_reader import MarketBarReader, _instrument_ids
from utils.time_axis import Frequency
from utils.data_access_config import DataAccessConfig

if TYPE_CHECKING:
    from utils.duckdb_adapter import DuckDBAdapter


class MissingBarDataError(ValueError):
    pass


class BarSource(MarketBarReader):
    @classmethod
    def read_panel(cls, instruments: Sequence[str], frequency: str | None, start: str,
                   end: str, fields: Sequence[str], *, config: DataAccessConfig,
                   connection: duckdb.DuckDBPyConnection | None = None,
                   adapter: DuckDBAdapter | None = None, date_field: str = 'date',
                   code_field: str = 'stock_code', additional_condition: str = '') -> pd.DataFrame:
        """Catalog entry for legacy panels, retaining their exact SQL/PIT inputs.

        The adapter continues to own Parquet selection, revision caches and SQL.
        UTC Bar conversion is only used for datasets stored as canonical Bars.
        """
        from utils.data_access_config import Dataset
        from utils.data_store import quote, table_name

        parsed = Frequency.parse(frequency)
        if not parsed.is_valid:
            raise MissingBarDataError(f'Invalid frequency={frequency!r}')
        if adapter is None and not isinstance(connection, duckdb.DuckDBPyConnection):
            raise TypeError(f'Supply a DuckDB connection or adapter for frequency={frequency!r}')
        if pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError('start_date must not be after end_date')
        if not isinstance(instruments, (list, tuple)) or any(
                not isinstance(value, str) or not value for value in instruments):
            raise ValueError('instruments must be a list or tuple of nonempty stock codes')
        adapter_panel = (adapter is not None and date_field == 'date'
                         and code_field == 'stock_code' and not additional_condition)
        catalog = BarCatalog(config=config.as_dict())
        try:
            entry = catalog.resolve(frequency, '*')
        except LookupError:
            entry = None
        dataset = config.dataset(entry.table_name if entry is not None else parsed.table_name)
        if date_field == 'date':
            date_field = dataset.date_column
        if code_field == 'stock_code':
            code_field = dataset.stock_column
        if not isinstance(fields, (list, tuple)):
            raise ValueError('fields must be a list or tuple of column names')
        excluded = {date_field, code_field, 'date', 'stock_code', 'code'}
        selected = Dataset(dataset.table, date_field, code_field).fields(
            list(dict.fromkeys(field for field in fields if field not in excluded)))
        if not instruments:
            frame = pd.DataFrame(columns=['date', 'code', *selected])
            frame['date'] = pd.to_datetime(frame['date'])
            if adapter is not None:
                frame.attrs['_sorted_by_code_date'] = True
            return frame

        conn = adapter._get_connection() if adapter is not None else connection
        # Parquet declarations describe file datasets, not in-memory SQL tables.
        if adapter is None or adapter.parquet_dir is None:
            catalog = BarCatalog(conn, config=config.as_dict())
            try:
                entry = catalog.resolve(frequency, '*')
            except LookupError:
                entry = None
        if entry is None or entry.timestamp_semantics == 'UTC_bar_start':
            with cls(conn, config=config.as_dict()) as reader:
                bars = reader.read(instruments, frequency, start, end)
            panel = bars.rename(columns={'timestamp_utc': 'date', 'instrument_id': 'code'})
            panel['date'] = panel.date.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
            return panel[['date', 'code', *selected]].sort_values(
                ['code', 'date'], kind='mergesort', ignore_index=True)
        dataset = config.dataset(entry.table_name)
        if adapter_panel:
            frame = adapter._fetch_panel(dataset, instruments, start, end, fields)
        else:
            columns = [f'{quote(date_field)} AS date', f'{quote(code_field)} AS code',
                       *[quote(field) for field in selected]]
            placeholders = ', '.join('?' for _ in instruments)
            sql = f"""SELECT {', '.join(columns)} FROM {table_name(dataset.table)}
                WHERE CAST({quote(date_field)} AS TIMESTAMP) >= ?
                  AND CAST({quote(date_field)} AS TIMESTAMP) < ?
                  AND {quote(code_field)} IN ({placeholders})
                  {additional_condition}
                ORDER BY {quote(code_field)}, {quote(date_field)}"""
            frame = conn.execute(sql, [pd.Timestamp(start).to_pydatetime(),
                (pd.Timestamp(end) + pd.Timedelta(days=1)).to_pydatetime(), *instruments]).fetchdf()
            frame['date'] = pd.to_datetime(frame['date'])
        # Preserve panel attrs, dtypes, ordering and timestamps for old cache keys.
        persisted = (BarCatalog(conn, config=config.as_dict())
                     if table_exists(conn, entry.table_name) else catalog)
        warn_quality(conn, parsed.canonical, [(instrument,
            persisted.resolve(frequency, instrument),
            pd.Timestamp(start, tz='Asia/Shanghai').tz_convert('UTC'),
            (pd.Timestamp(end, tz='Asia/Shanghai') + pd.Timedelta(days=1)).tz_convert('UTC'))
            for instrument in instruments])
        return frame

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None, *,
                 duckdb_path: str | Path | None = None, config: Mapping | str | Path | None = None,
                 legacy_snapshot: bool = False) -> None:
        super().__init__(connection, duckdb_path=duckdb_path)
        self.config = config
        self.legacy_snapshot = legacy_snapshot

    def _missing(self, frequency: str, instrument: str, catalog: BarCatalog,
                 asset_class: str, reason: str) -> MissingBarDataError:
        available = sorted({entry.frequency for entry in catalog.available(instrument, asset_class=asset_class)})
        return MissingBarDataError(
            f'Missing frequency={frequency} instrument={instrument}: {reason}; '
            f'available frequencies={available}; build: python -m tools.agent_harness.cli '
            f'build-bars --db <path> --source <available-frequency> --targets {frequency} '
            f'--instruments {instrument} --start <session> --end <session> '
            '[--bucket-strategy <strategy>]')

    def read(self, instruments: Sequence[str], frequency: str, start=None, end=None,
             view: str = 'ETH') -> pd.DataFrame:
        ids = _instrument_ids(instruments)
        parsed = Frequency.parse(frequency)
        if not parsed.is_valid:
            raise MissingBarDataError(f'Invalid frequency={frequency!r}')
        if view not in ('ETH', 'RTH'):
            raise ValueError('view must be ETH or RTH')
        frequency = parsed.canonical
        catalog = BarCatalog(self.connection, config=self.config)
        frames, entries, quality_bounds, build_ids = [], [], [], {}
        for instrument in ids:
            asset_class = 'equity'
            if table_exists(self.connection, 'instrument_metadata'):
                row = self.connection.execute('SELECT asset_class FROM instrument_metadata WHERE instrument_id=?',
                                              [instrument]).fetchone()
                if row:
                    asset_class = row[0]
            try:
                entry = catalog.resolve(frequency, instrument, asset_class=asset_class)
            except LookupError:
                entry = None
            snapshot = (self.legacy_snapshot and (entry is None or entry.kind != 'derived')
                        and table_exists(self.connection, 'market_bars')
                        and self.connection.execute('SELECT EXISTS(SELECT 1 FROM market_bars '
                            'WHERE instrument_id=? AND frequency=?)', [instrument, frequency]).fetchone()[0])
            if entry is None:
                if not snapshot:
                    raise self._missing(frequency, instrument, catalog, asset_class, 'no catalog entry')
                from utils.bar_catalog import CatalogEntry
                entry = CatalogEntry(frequency, 'market_bars', dataset=asset_class, instrument_id=instrument)
            entries.append(entry)
            legacy = not snapshot and entry.timestamp_semantics != 'UTC_bar_start'
            key = 'stock_code' if legacy else 'instrument_id'
            time = 'date' if legacy else 'timestamp_utc'
            source_table = 'market_bars' if snapshot else entry.table_name
            bounds = self.connection.execute(f'SELECT min({time}), max({time}) FROM '
                f'{identifier(source_table)} WHERE {key}=?' + (' AND frequency=?' if snapshot else ''),
                [instrument, frequency] if snapshot else [instrument]).fetchone()
            if bounds[0] is None and not entry.coverage_start:
                raise self._missing(frequency, instrument, catalog, asset_class, 'no instrument coverage')
            def utc_bound(value, upper: bool):
                if value is None:
                    return None
                timestamp = pd.Timestamp(value)
                if timestamp.tzinfo is None:
                    if timestamp != timestamp.normalize():
                        raise ValueError('start/end must be UTC timestamps or session dates')
                    timestamp = timestamp.tz_localize('Asia/Shanghai' if legacy else 'UTC')
                    if upper:
                        timestamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
                else:
                    _utc(timestamp, 'start/end')
                return timestamp.tz_convert('UTC')
            lower, upper = utc_bound(start, False), utc_bound(end, True)
            # Date bounds refer to session labels, including the preceding night.
            date_request = lambda value: value is not None and pd.Timestamp(value).tzinfo is None
            sessions = []
            if not legacy:
                from data_pipeline.loaders.market_bar_importer import calendar_name
                from utils.session_calendar import SessionCalendar
                meta = self.get_instrument_metadata([instrument]).iloc[0].to_dict()
                start_day = str(start)[:10] if date_request(start) else entry.coverage_start
                end_day = str(end)[:10] if date_request(end) else entry.coverage_end
                if ((date_request(start) and entry.coverage_end and start_day > entry.coverage_end)
                        or (date_request(end) and entry.coverage_start and end_day < entry.coverage_start)):
                    raise self._missing(frequency, instrument, catalog, asset_class, 'insufficient coverage')
                if start_day and end_day:
                    cal = SessionCalendar(calendar_name(meta), start_day, end_day,
                                          rth_reference_calendar=meta['rth_reference_calendar'])
                    sessions = [cal.session_on(day.strftime('%Y-%m-%d')) for day in cal.sessions()]
                    if sessions:
                        if date_request(start):
                            lower = sessions[0].segments[0][0]
                        if date_request(end):
                            upper = sessions[-1].segments[-1][1] - pd.Timedelta(nanoseconds=1)
            # Audited coverage is a session envelope; interior holes remain quality evidence.
            if (entry.quality_status == 'scanned' and table_exists(self.connection, 'bar_quality')
                    and entry.timestamp_semantics != 'local_midnight_day_identifier_pit'):
                coverage = self.connection.execute('SELECT min(start_utc), max(end_utc) FROM bar_quality '
                    'WHERE frequency=? AND instrument_id=?', [frequency, instrument]).fetchone()
                cstart, cend = [pd.Timestamp(value).tz_convert('UTC') for value in coverage]
                cend -= pd.Timedelta(nanoseconds=1)
            else:
                shift = parsed.duration if entry.timestamp_semantics == 'Asia/Shanghai_bar_end' else pd.Timedelta(0)
                cstart, cend = [pd.Timestamp(value).tz_localize('Asia/Shanghai').tz_convert('UTC') - shift
                                if legacy else pd.Timestamp(value).tz_convert('UTC') for value in bounds]
            lower = cstart if lower is None else lower
            upper = cend if upper is None else upper
            if lower > upper:
                raise ValueError('start must not follow end')
            # Unscanned local-date requests check dates, rather than requiring full-day bars.
            outside_start, outside_end = lower < cstart, upper > cend
            if legacy:
                first_day, last_day = cstart.tz_convert('Asia/Shanghai').date(), cend.tz_convert('Asia/Shanghai').date()
                if date_request(start):
                    outside_start = pd.Timestamp(start).date() < first_day
                if date_request(end):
                    outside_end = pd.Timestamp(end).date() > last_day
            if not snapshot and (outside_start or outside_end):
                raise self._missing(frequency, instrument, catalog, asset_class, 'insufficient coverage')
            if not snapshot and not legacy and entry.quality_status == 'scanned':
                scanned = {row[0] for row in self.connection.execute(
                    'SELECT session_id FROM bar_quality WHERE frequency=? AND instrument_id=? '
                    'AND start_utc<=? AND end_utc>?', [frequency, instrument, upper, lower]).fetchall()}
                required = {session.session_date for session in sessions
                            if session.segments[0][0] <= upper and session.segments[-1][1] > lower}
                if not required <= scanned:
                    raise self._missing(frequency, instrument, catalog, asset_class,
                                        'unbuilt or unaudited Session coverage')
            frame = self.read_bars([instrument], frequency, start_utc=lower, end_utc=upper,
                                   source_table=source_table, view=view)
            frames.append(frame)
            quality_bounds.append((lower, upper))
            if table_exists(self.connection, 'bar_build_log'):
                rows = self.connection.execute('SELECT session_id, build_id FROM bar_build_log '
                    'WHERE frequency=? AND instrument_id=? AND start_utc<=? AND end_utc>?',
                    [frequency, instrument, upper, lower]).fetchall()
                build_ids[instrument] = dict(rows)
        # Clear per-frame attrs before concat: dict equality may contain arrays.
        hashes = {instrument: frame.attrs.get('calendar_hash') for instrument, frame in zip(ids, frames)}
        metadata = [row for frame in frames for row in frame.attrs.get('instrument_metadata', [])]
        coverage = {name: {key: bounds for frame in frames
                           for key, bounds in frame.attrs.get(name, {}).items()}
                    for name in ('raw_coverage_by_instrument', 'raw_coverage_by_key')}
        for name, aggregate in [('raw_coverage_start_utc', min), ('raw_coverage_end_utc', max)]:
            bounds = [frame.attrs[name] for frame in frames if pd.notna(frame.attrs.get(name))]
            coverage[name] = aggregate(bounds) if bounds else pd.NaT
        has_completeness = any('is_complete' in frame for frame in frames)
        for frame in frames:
            # Stage-5 snapshots have no bucket flags; their present native Bars
            # keep the prior implicit-complete contract when mixed with new tables.
            if has_completeness and 'is_complete' not in frame:
                frame['is_complete'] = True
            frame.attrs = {}
        result = pd.concat(frames, ignore_index=True)
        result.attrs.update(coverage)
        result.attrs.update(session_view=view, calendar_hash=hashes, build_ids=build_ids)
        if frequency == '1d' and all(entry.timestamp_semantics == 'local_midnight_day_identifier_pit'
                                   for entry in entries):
            result.attrs['daily_timestamp_semantics'] = 'local_midnight_day_identifier'
        if metadata:
            result.attrs['instrument_metadata'] = metadata
        # Each instrument retains its own bounds; emit at most one warning for the request.
        warn_quality(self.connection, frequency, [
            (instrument, entry, lower, upper)
            for instrument, entry, (lower, upper) in zip(ids, entries, quality_bounds)])
        return result
