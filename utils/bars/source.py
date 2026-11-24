"""Read-only Bar access, calendar views."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import duckdb
import pandas as pd

from backtest_and_analysis.time_series.contracts import (
    ContractValidationError, _duration, _text, _utc, validate_bar_frame,
)
from backtest_and_analysis.time_series.csv_reader import (
    BarAvailability, _record_raw_coverage, filter_bars_by_view,
)
from data_pipeline.loaders.market_bar_importer import (
    calendars_for_bars, instrument_calendar,
    validate_instrument_metadata,
)
from utils.session_calendar import SessionCalendar
from utils.time_axis import Frequency

from .catalog import BarCatalog, CatalogEntry, identifier, table_exists
from .quality import warn_quality

def _instrument_ids(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ContractValidationError('instrument_ids must be a nonempty sequence')
    for value in values:
        _text(value, 'instrument_id')
    return list(dict.fromkeys(values))


def _utc_bound(value, upper: bool, timezone: str = 'UTC') -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        if timestamp != timestamp.normalize():
            raise ValueError('start/end must be UTC timestamps or session dates')
        timestamp = timestamp.tz_localize(timezone)
        if upper:
            timestamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    else:
        _utc(timestamp, 'start/end')
    return timestamp.tz_convert('UTC')


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
    """Read canonical Bars with inclusive UTC starts using a read-only or borrowed connection."""

    _default_asset_class = 'market'

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None, *,
                 duckdb_path: str | Path | None = None, config: Mapping | str | Path | None = None) -> None:
        if (connection is None) == (duckdb_path is None):
            raise ValueError('Supply exactly one of connection or duckdb_path')
        self._owns_connection = connection is None
        self.connection = connection if connection is not None else duckdb.connect(
            str(duckdb_path), read_only=True)
        self.config = config

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
                  source_table: str = 'market_bars') -> pd.DataFrame:
        """Read and validate Bars, preserving raw coverage before ETH/RTH selection."""
        ids = _instrument_ids(instrument_ids)
        _duration(frequency)
        start, end = _bounds(start_utc, end_utc)
        if view not in ('ETH', 'RTH'):
            raise ContractValidationError('view must be ETH or RTH')
        parsed = Frequency.parse(frequency)
        if source_table != 'market_bars' and (not parsed.is_valid or source_table != parsed.table_name):
            raise ContractValidationError(f'Unsupported Bar source_table: {source_table!r}')
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
        return self._filter_view(bars, calendar, view)

    @staticmethod
    def _filter_view(bars: pd.DataFrame, calendar, view: str) -> pd.DataFrame:
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


    def _missing(self, frequency: str, instrument: str, catalog: BarCatalog,
                 asset_class: str, reason: str) -> MissingBarDataError:
        available = sorted({entry.frequency for entry in catalog.available(instrument, asset_class=asset_class)})
        return MissingBarDataError(
            f'Missing frequency={frequency} instrument={instrument}: {reason}; '
            f'available frequencies={available}; build: python -m tools.agent_harness.cli '
            f'build-bars --db <path> --source <available-frequency> --targets {frequency} '
            f'--instruments {instrument} --start <session> --end <session> '
            '[--bucket-strategy <strategy>]')

    @property
    def catalog(self) -> BarCatalog:
        return BarCatalog(self.connection)

    def catalog_instruments(self, entry: CatalogEntry) -> list[str]:
        return [row[0] for row in self.connection.execute(
            f'SELECT DISTINCT instrument_id FROM {identifier(entry.table_name)} WHERE frequency=?',
            [entry.frequency]).fetchall()]

    def catalog_metadata(self, instrument: str, frequency: str, *,
                         entry: CatalogEntry | None = None) -> pd.DataFrame:
        return self.get_instrument_metadata([instrument])

    def read_catalog_bars(self, entry: CatalogEntry, instrument: str, *, start_utc=None,
                          end_utc=None, calendar: SessionCalendar | None = None) -> pd.DataFrame:
        return self.read_bars([instrument], entry.frequency, start_utc=start_utc, end_utc=end_utc,
                              source_table=entry.table_name, calendar=calendar)

    def audit_catalog_bars(self, entry: CatalogEntry, instrument: str, metadata: dict,
                           start: str | None, end: str | None
                           ) -> tuple[pd.DataFrame, SessionCalendar] | None:
        raw = self.read_catalog_bars(entry, instrument)
        if raw.empty:
            return None
        timezone = metadata['timezone']
        left = raw.timestamp_utc.min().tz_convert(timezone) - pd.Timedelta(days=7)
        right = raw.timestamp_utc.max().tz_convert(timezone) + pd.Timedelta(days=7)
        padded = instrument_calendar(metadata, left.strftime('%Y-%m-%d'), right.strftime('%Y-%m-%d'))
        owners = [padded.session_at(timestamp) for timestamp in raw.timestamp_utc]
        if any(owner is None for owner in owners):
            raise ValueError('Cannot audit off-session bars')
        first = min(owner.session_date for owner in owners)
        last = max(owner.session_date for owner in owners)
        calendar = instrument_calendar(metadata, start or first, end or last)
        return raw, calendar

    def _resolve_entry(self, catalog: BarCatalog, frequency: str, instrument: str,
                       asset_class: str) -> CatalogEntry:
        try:
            return catalog.resolve(frequency, instrument, asset_class=asset_class)
        except LookupError:
            raise self._missing(frequency, instrument, catalog, asset_class, 'no catalog entry') from None

    def _quality_bounds(self, frequency: str, instrument: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        bounds = self.connection.execute('SELECT min(start_utc), max(end_utc) FROM bar_quality '
            'WHERE frequency=? AND instrument_id=?', [frequency, instrument]).fetchone()
        start, end = [pd.Timestamp(value).tz_convert('UTC') for value in bounds]
        return start, end - pd.Timedelta(nanoseconds=1)

    def _read_entry(self, entry: CatalogEntry, instrument: str, start, end, view: str,
                    catalog: BarCatalog, asset_class: str, *, require_coverage: bool = True
                    ) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        frequency = entry.frequency
        bounds = self.connection.execute(
            f'SELECT min(timestamp_utc), max(timestamp_utc) FROM {identifier(entry.table_name)} '
            'WHERE instrument_id=? AND frequency=?', [instrument, frequency]).fetchone()
        if bounds[0] is None and not entry.coverage_start:
            raise self._missing(frequency, instrument, catalog, asset_class, 'no instrument coverage')
        lower, upper = _utc_bound(start, False), _utc_bound(end, True)
        start_is_date = start is not None and pd.Timestamp(start).tzinfo is None
        end_is_date = end is not None and pd.Timestamp(end).tzinfo is None
        start_day = str(start)[:10] if start_is_date else entry.coverage_start
        end_day = str(end)[:10] if end_is_date else entry.coverage_end
        sessions = []
        if ((start_is_date and entry.coverage_end and start_day > entry.coverage_end)
                or (end_is_date and entry.coverage_start and end_day < entry.coverage_start)):
            raise self._missing(frequency, instrument, catalog, asset_class, 'insufficient coverage')
        if start_day and end_day:
            meta = self.get_instrument_metadata([instrument]).iloc[0].to_dict()
            cal = instrument_calendar(meta, start_day, end_day)
            sessions = [cal.session_on(day) for day in cal.sessions()]
            if sessions:
                if start_is_date:
                    lower = sessions[0].segments[0][0]
                if end_is_date:
                    upper = sessions[-1].segments[-1][1] - pd.Timedelta(nanoseconds=1)
        scanned_quality = entry.quality_status == 'scanned' and table_exists(self.connection, 'bar_quality')
        if scanned_quality:
            cstart, cend = self._quality_bounds(frequency, instrument)
        else:
            cstart, cend = [pd.Timestamp(value).tz_convert('UTC') for value in bounds]
        lower = cstart if lower is None else lower
        upper = cend if upper is None else upper
        if lower > upper:
            raise ValueError('start must not follow end')
        if require_coverage and (lower < cstart or upper > cend):
            raise self._missing(frequency, instrument, catalog, asset_class, 'insufficient coverage')
        if require_coverage and scanned_quality:
            scanned = {row[0] for row in self.connection.execute(
                'SELECT session_id FROM bar_quality WHERE frequency=? AND instrument_id=? '
                'AND start_utc<=? AND end_utc>?', [frequency, instrument, upper, lower]).fetchall()}
            required = {session.session_date for session in sessions
                        if session.segments[0][0] <= upper and session.segments[-1][1] > lower}
            if not required <= scanned:
                raise self._missing(frequency, instrument, catalog, asset_class,
                                    'unbuilt or unaudited Session coverage')
        frame = self.read_bars([instrument], frequency, start_utc=lower, end_utc=upper,
                               source_table=entry.table_name, view=view)
        return frame, lower, upper

    def read(self, instruments: Sequence[str], frequency: str, start=None, end=None,
             view: str = 'ETH') -> pd.DataFrame:
        ids = _instrument_ids(instruments)
        parsed = Frequency.parse(frequency)
        if not parsed.is_valid:
            raise MissingBarDataError(f'Invalid frequency={frequency!r}')
        if view not in ('ETH', 'RTH'):
            raise ValueError('view must be ETH or RTH')
        frequency = parsed.canonical
        catalog = self.catalog
        frames, entries, quality_bounds, build_ids = [], [], [], {}
        for instrument in ids:
            asset_class = self._default_asset_class
            if table_exists(self.connection, 'instrument_metadata'):
                row = self.connection.execute('SELECT asset_class FROM instrument_metadata WHERE instrument_id=?',
                                              [instrument]).fetchone()
                if row:
                    asset_class = row[0]
            entry = self._resolve_entry(catalog, frequency, instrument, asset_class)
            entries.append(entry)
            frame, lower, upper = self._read_entry(entry, instrument, start, end, view, catalog, asset_class)
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
        daily_semantics = {frame.attrs.get('daily_timestamp_semantics') for frame in frames}
        for frame in frames:
            # Present native Bars without a completeness field are implicitly complete.
            if has_completeness and 'is_complete' not in frame:
                frame['is_complete'] = True
            frame.attrs = {}
        result = pd.concat(frames, ignore_index=True)
        result.attrs.update(coverage)
        result.attrs.update(session_view=view, calendar_hash=hashes, build_ids=build_ids)
        if len(daily_semantics) == 1 and None not in daily_semantics:
            result.attrs['daily_timestamp_semantics'] = next(iter(daily_semantics))
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
