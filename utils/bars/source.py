"""Read-only Bar access, calendar views and explicit legacy conversion."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from backtest_and_analysis.time_series.contracts import (
    ContractValidationError, _duration, _text, _utc, validate_bar_frame,
)
from backtest_and_analysis.time_series.csv_reader import (
    BarAvailability, _record_raw_coverage, filter_bars_by_view,
)
from data_pipeline.loaders.market_bar_importer import (
    ExternalBarSourceSpec, adapt_external_bars, calendar_name, calendars_for_bars,
    validate_instrument_metadata,
)
from utils.data_access_config import DataAccessConfig
from utils.session_calendar import SessionCalendar
from utils.time_axis import Frequency

from .catalog import BarCatalog, CatalogEntry, identifier, table_exists
from .quality import warn_quality

if TYPE_CHECKING:
    from utils.duckdb_adapter import DuckDBAdapter

# JoinQuant A-share five-minute compatibility: local, right-labelled intervals.
LEGACY_5M_TIMEZONE = 'Asia/Shanghai'
LEGACY_5M_TIMESTAMP_ANCHOR = 'bar_end'


def _instrument_ids(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ContractValidationError('instrument_ids must be a nonempty sequence')
    for value in values:
        _text(value, 'instrument_id')
    return list(dict.fromkeys(values))


def _bounds(start_utc, end_utc) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    values = []
    for name, value in [('start_utc', start_utc), ('end_utc', end_utc)]:
        if value is not None:
            value = pd.Timestamp(value)
            _utc(value, name)
        values.append(value)
    start, end = values
    if start is not None and end is not None and start > end:
        raise ContractValidationError('start_utc must not follow end_utc')
    return start, end


class MissingBarDataError(ValueError):
    pass


class BarSource:
    """Borrow a connection, or own a read-only file connection (context manager).

    Bounds are inclusive UTC Bar starts. Legacy daily rows use local midnight
    as their day identifier, frequency 1d, and are returned without intraday view
    filtering; they are a compatibility panel, not executable intraday labels.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None, *,
                 duckdb_path: str | Path | None = None, config: Mapping | str | Path | None = None,
                 legacy_snapshot: bool = False) -> None:
        if (connection is None) == (duckdb_path is None):
            raise ValueError('Supply exactly one of connection or duckdb_path')
        self._owns_connection = connection is None
        self.connection = connection if connection is not None else duckdb.connect(
            str(duckdb_path), read_only=True)
        self.config = config
        self.legacy_snapshot = legacy_snapshot

    def close(self) -> None:
        if self._owns_connection:
            self.connection.close()

    def __enter__(self) -> BarSource:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def get_instrument_metadata(self, instrument_ids: Sequence[str]) -> pd.DataFrame:
        ids = _instrument_ids(instrument_ids)
        placeholders = ','.join('?' for _ in ids)
        metadata = self.connection.execute(
            f'SELECT * FROM instrument_metadata WHERE instrument_id IN ({placeholders}) '
            'ORDER BY instrument_id', ids).fetchdf()
        missing = set(ids) - set(metadata.instrument_id)
        if missing:
            raise ContractValidationError(f'Missing instrument metadata: {sorted(missing)}')
        return validate_instrument_metadata(metadata)

    def get_calendars(self, bars: pd.DataFrame) -> dict[str, SessionCalendar]:
        """Construct instrument calendars over the unfiltered Bar coverage."""
        if bars.empty:
            return {}
        snapshot = bars.attrs.get('instrument_metadata')
        metadata = (validate_instrument_metadata(pd.DataFrame(snapshot)) if snapshot is not None
                    else self.get_instrument_metadata(list(bars.instrument_id.unique())))
        return calendars_for_bars(metadata, bars)

    def read_bars(self, instrument_ids: Sequence[str], frequency: str, *,
                  start_utc=None, end_utc=None, view: str = 'ETH',
                  calendar: SessionCalendar | Mapping[str, SessionCalendar] | None = None,
                  source_table: str = 'market_bars',
                  legacy_tz: str = LEGACY_5M_TIMEZONE,
                  legacy_timestamp_anchor: str = LEGACY_5M_TIMESTAMP_ANCHOR,
                  data_version: str = 'legacy_unversioned') -> pd.DataFrame:
        """Read and validate Bars, preserving raw coverage before ETH/RTH selection."""
        ids = _instrument_ids(instrument_ids)
        _duration(frequency)
        start, end = _bounds(start_utc, end_utc)
        if view not in ('ETH', 'RTH'):
            raise ContractValidationError('view must be ETH or RTH')
        parsed = Frequency.parse(frequency)
        is_physical = parsed.is_valid and source_table == parsed.table_name
        if source_table != 'market_bars' and not is_physical:
            expected = {'price_data_5m': '5min', 'price_data_daily': '1d'}
            if source_table not in expected or frequency != expected[source_table]:
                raise ContractValidationError('legacy source_table and frequency must match 5min or 1d')
            bars = self.read_legacy_price_table(
                source_table, ids, start_utc=start, end_utc=end, legacy_tz=legacy_tz,
                legacy_timestamp_anchor=legacy_timestamp_anchor, data_version=data_version)
            if source_table == 'price_data_daily':
                return bars
            if calendar is None:
                calendar = calendars_for_bars(self._legacy_metadata(ids, legacy_tz), bars)
        else:
            placeholders = ','.join('?' for _ in ids)
            query = f'SELECT * FROM {source_table} WHERE instrument_id IN ({placeholders}) AND frequency = ?'
            parameters = [*ids, frequency]
            for bound, operator in [(start, '>='), (end, '<=')]:
                if bound is not None:
                    query += f' AND timestamp_utc {operator} ?'
                    parameters.append(bound)
            bars = self.connection.execute(query, parameters).fetchdf()
            bars['timestamp_utc'] = pd.to_datetime(bars.timestamp_utc, utc=True).astype('datetime64[ns, UTC]')
            bars = validate_bar_frame(bars).reset_index(drop=True)
            # Resolve all requested metadata even when this range has no bars.
            metadata = self.get_instrument_metadata(ids)
            if calendar is None:
                calendar = calendars_for_bars(metadata, bars)
        _record_raw_coverage(bars)
        if isinstance(calendar, SessionCalendar):
            return filter_bars_by_view(bars, calendar, view=view)
        if not isinstance(calendar, Mapping):
            raise ContractValidationError('calendar must be a SessionCalendar or instrument mapping')
        selected, hashes = [], {}
        for instrument, group in bars.groupby('instrument_id', sort=True):
            cal = calendar.get(instrument)
            if not isinstance(cal, SessionCalendar):
                raise ContractValidationError(f'Missing calendar for {instrument}')
            selected.append(filter_bars_by_view(group, cal, view=view))
            hashes[instrument] = cal.calendar_hash
        if selected:
            result = pd.concat(selected, ignore_index=True)
            result = validate_bar_frame(result).reset_index(drop=True)
        else:
            result = bars.copy()
        result.attrs = dict(bars.attrs)
        unique_hashes = set(hashes.values())
        result.attrs.update(session_view=view, calendar_hash=(
            next(iter(unique_hashes)) if len(unique_hashes) == 1 else hashes))
        return result

    @staticmethod
    def _legacy_metadata(ids: list[str], timezone: str) -> pd.DataFrame:
        return pd.DataFrame([dict(instrument_id=instrument, asset_class='equity',
                                  exchange='XSHG', timezone=timezone, contract_expiry=None,
                                  price_adjustment='split_dividend_factor', point_value=1.0,
                                  rth_reference_calendar='XSHG') for instrument in ids])

    def read_legacy_price_table(self, source_table: str, instrument_ids: Sequence[str], *,
                                start_utc=None, end_utc=None,
                                legacy_tz: str = LEGACY_5M_TIMEZONE,
                                legacy_timestamp_anchor: str = LEGACY_5M_TIMESTAMP_ANCHOR,
                                data_version: str = 'legacy_unversioned') -> pd.DataFrame:
        """Query old tables without DDL, migration, adjustment or writes.

        The 5m preset is Asia/Shanghai + bar_end. Override both semantics for
        another legacy source. Daily dates remain day identifiers at midnight.
        Supply a frozen data_version when using legacy results in research.
        """
        if source_table not in ('price_data_5m', 'price_data_daily'):
            raise ContractValidationError('Unsupported legacy source_table')
        ids = _instrument_ids(instrument_ids)
        start, end = _bounds(start_utc, end_utc)
        daily = source_table == 'price_data_daily'
        frequency = '1d' if daily else '5min'
        spec = ExternalBarSourceSpec(
            timestamp_anchor='bar_start' if daily else legacy_timestamp_anchor,
            source_timezone=legacy_tz, volume_mode='interval',
            column_mapping={'date': 'timestamp', 'stock_code': 'instrument_id',
                            **{c: c for c in ('open', 'high', 'low', 'close', 'volume')}})
        placeholders = ','.join('?' for _ in ids)
        # Names are allowlisted; values are bound. Optional legacy fields survive.
        query = f'SELECT * FROM {source_table} WHERE stock_code IN ({placeholders})'
        parameters = list(ids)
        shift = pd.Timedelta(0) if spec.timestamp_anchor == 'bar_start' else _duration(frequency)
        for bound, operator in [(start, '>='), (end, '<=')]:
            if bound is not None:
                query += f' AND date {operator} ?'
                parameters.append((bound + shift).tz_convert(legacy_tz).tz_localize(None))
        raw = self.connection.execute(query, parameters).fetchdf()
        metadata = self._legacy_metadata(ids, legacy_tz)
        bars = adapt_external_bars(raw, spec, metadata=metadata,
                                   frequency=frequency, source=f'legacy_{source_table}',
                                   data_version=data_version)
        bars.attrs['instrument_metadata'] = metadata.to_dict('records')
        if daily:
            bars.attrs['daily_timestamp_semantics'] = 'local_midnight_day_identifier'
        _record_raw_coverage(bars)
        return bars

    def build_availability(self, bars: pd.DataFrame, view: str = 'ETH', *,
                           calendar: SessionCalendar | Mapping[str, SessionCalendar] | None = None,
                           bucket_strategy: str | None = None
                           ) -> BarAvailability | dict[str, BarAvailability]:
        """Return one availability snapshot, or instrument snapshots across markets."""
        calendars = calendar if calendar is not None else self.get_calendars(bars)
        if isinstance(calendars, SessionCalendar):
            return BarAvailability(bars, calendars, view=view, bucket_strategy=bucket_strategy)
        snapshots = {instrument: BarAvailability(group, calendars[instrument], view=view,
                                                  bucket_strategy=bucket_strategy)
                     for instrument, group in bars.groupby('instrument_id')}
        return next(iter(snapshots.values())) if len(snapshots) == 1 else snapshots


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


MarketBarReader = BarSource


def read_market_bars(instrument_ids: Sequence[str], frequency: str, *,
                     connection: duckdb.DuckDBPyConnection | None = None,
                     duckdb_path: str | Path | None = None, **kwargs) -> pd.DataFrame:
    """Convenience read with deterministic ownership of file connections."""
    with BarSource(connection, duckdb_path=duckdb_path) as reader:
        return reader.read_bars(instrument_ids, frequency, **kwargs)
