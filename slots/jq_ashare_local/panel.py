"""Transitional readers for historical A-share panels and snapshots."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from backtest_and_analysis.time_series.contracts import ContractValidationError, _duration
from backtest_and_analysis.time_series.csv_reader import _record_raw_coverage
from data_pipeline.loaders.market_bar_importer import ExternalBarSourceSpec, adapt_external_bars, calendars_for_bars
from utils.bars.catalog import BarCatalog, CatalogEntry, identifier, table_exists
from utils.bars.quality import warn_quality
from utils.bars.source import BarSource, MissingBarDataError, _bounds, _instrument_ids, _utc_bound
from utils.data_access_config import DataAccessConfig, load_data_access_config
from utils.market_db import MarketDbProfile
from utils.market_db.storage import _require_same_hash
from utils.market_db.validation import (
    calendar_key, market_db_calendar_bounds, require_market_db_profile,
    validate_market_db_calendar, validate_market_db_provider,
)
from utils.session_calendar import SessionCalendar
from utils.time_axis import Frequency

if TYPE_CHECKING:
    from utils.duckdb_adapter import DuckDBAdapter

TIMEZONE = 'Asia/Shanghai'
LEGACY_CALENDAR_ID = 'XSHG'
LEGACY_5M_TIMEZONE = TIMEZONE
LEGACY_5M_TIMESTAMP_ANCHOR = 'bar_end'
_DAILY_SEMANTICS = 'local_midnight_day_identifier_pit'
_INTRADAY_SEMANTICS = 'Asia/Shanghai_bar_end'


def require_ashare_panel_profile(
    connection: duckdb.DuckDBPyConnection, *, expected: MarketDbProfile | None = None,
) -> MarketDbProfile:
    profile = require_market_db_profile(connection)
    validate_market_db_provider(profile, 'jq_ashare_local', domain='cn_equity')
    validate_market_db_calendar(profile, LEGACY_CALENDAR_ID, TIMEZONE)
    if expected is not None:
        _require_same_hash(profile, expected, 'Factor database disagrees with its Slot')
    return profile


class ASharePanelCatalog(BarCatalog):
    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None, *,
                 config: Mapping | None = None) -> None:
        prices = (load_data_access_config().price_tables if config is None else config.get('price_tables', {}))
        entries = []
        for frequency, name in prices.items():
            parsed = Frequency.parse(frequency)
            if not parsed.is_valid:
                raise ValueError(f'Invalid configured frequency: {frequency!r}')
            entries.append(CatalogEntry(parsed.canonical, name, dataset='equity',
                timestamp_semantics=_DAILY_SEMANTICS if parsed.kind == 'session' else _INTRADAY_SEMANTICS))
        super().__init__(connection, entries=entries)

    def _accepts(self, entry: CatalogEntry) -> bool:
        return entry.timestamp_semantics in ('UTC_bar_start', _DAILY_SEMANTICS, _INTRADAY_SEMANTICS)

    def available(self, instrument: str, *, asset_class: str = 'equity',
                  dataset: str | None = None) -> list[CatalogEntry]:
        return [entry for entry in super().available(instrument, asset_class=asset_class, dataset=dataset)
                if entry.timestamp_semantics == 'UTC_bar_start' or asset_class == 'equity']

    def resolve(self, frequency: str, instrument: str, *, asset_class: str = 'equity',
                dataset: str | None = None) -> CatalogEntry:
        return super().resolve(frequency, instrument, asset_class=asset_class, dataset=dataset)


class ASharePanelReader(BarSource):
    _default_asset_class = 'equity'

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None, *,
                 duckdb_path: str | Path | None = None,
                 config: Mapping | None = None, legacy_snapshot: bool = False,
                 market_db_profile: MarketDbProfile | None = None) -> None:
        super().__init__(connection, duckdb_path=duckdb_path, config=config)
        self.legacy_snapshot = legacy_snapshot
        self.market_db_profile = market_db_profile
        self.timezone = market_db_profile.timezone if market_db_profile is not None else TIMEZONE
        self.calendar_id = (calendar_key(market_db_profile.calendar_id, market_domain=market_db_profile.market_domain)
                            if market_db_profile is not None else LEGACY_CALENDAR_ID)

    def _panel_calendar(self, start, end) -> SessionCalendar:
        if self.market_db_profile is not None:
            start, end = market_db_calendar_bounds(self.market_db_profile, start, end)
        return SessionCalendar(self.calendar_id, start, end)

    def legacy_instrument_metadata(self, instruments: Sequence[str], *, timezone: str | None = None) -> pd.DataFrame:
        profile = self.market_db_profile
        metadata = self._legacy_metadata(list(instruments), timezone or self.timezone, calendar_id=self.calendar_id)
        if profile is not None:
            validate_market_db_provider(profile, 'jq_ashare_local', domain='cn_equity')
            metadata['calendar_id'] = self.calendar_id
            metadata['exchange'] = [
                instrument.rsplit('.', 1)[1] if '.' in instrument else self.calendar_id
                for instrument in instruments
            ]
        return metadata

    @property
    def catalog(self) -> ASharePanelCatalog:
        return ASharePanelCatalog(self.connection, config=self.config)

    def catalog_instruments(self, entry: CatalogEntry) -> list[str]:
        if entry.timestamp_semantics == 'UTC_bar_start':
            return super().catalog_instruments(entry)
        return [row[0] for row in self.connection.execute(
            f'SELECT DISTINCT stock_code FROM {identifier(entry.table_name)}').fetchall()]

    def catalog_metadata(self, instrument: str, frequency: str, *,
                         entry: CatalogEntry | None = None) -> pd.DataFrame:
        if entry is not None and entry.timestamp_semantics != 'UTC_bar_start':
            return self.legacy_instrument_metadata([instrument])
        if table_exists(self.connection, 'instrument_metadata') and self.connection.execute(
                'SELECT EXISTS(SELECT 1 FROM instrument_metadata WHERE instrument_id=?)',
                [instrument]).fetchone()[0]:
            return self.get_instrument_metadata([instrument])
        entry = self.catalog.resolve(frequency, instrument)
        if entry.timestamp_semantics == 'UTC_bar_start':
            raise ValueError(f'Missing instrument metadata: {instrument}')
        return self.legacy_instrument_metadata([instrument])

    def read_catalog_bars(self, entry: CatalogEntry, instrument: str, *, start_utc=None,
                          end_utc=None, calendar: SessionCalendar | None = None) -> pd.DataFrame:
        if entry.timestamp_semantics != _DAILY_SEMANTICS:
            return super().read_catalog_bars(entry, instrument, start_utc=start_utc, end_utc=end_utc,
                                             calendar=calendar)
        # Builders consume session-open labels for daily bars.
        local_bounds = [(bound.tz_convert(self.timezone).normalize().tz_convert('UTC')
                         if bound is not None else None) for bound in (start_utc, end_utc)]
        bars = self.read_legacy_price_table(entry.table_name, [instrument], start_utc=local_bounds[0],
                                            end_utc=local_bounds[1], frequency='1d')
        if bars.empty:
            return bars
        days = bars.timestamp_utc.dt.tz_convert(self.timezone).dt.tz_localize(None).dt.normalize()
        if calendar is None:
            calendar = self._panel_calendar(days.min(), days.max())
        return self._daily_bars_on_calendar(bars, days, calendar, instrument)

    def audit_catalog_bars(self, entry: CatalogEntry, instrument: str, metadata: dict,
                           start: str | None, end: str | None
                           ) -> tuple[pd.DataFrame, SessionCalendar] | None:
        if entry.timestamp_semantics != _DAILY_SEMANTICS:
            return super().audit_catalog_bars(entry, instrument, metadata, start, end)
        bars = self.read_legacy_price_table(entry.table_name, [instrument], frequency='1d')
        if bars.empty:
            return None
        days = bars.timestamp_utc.dt.tz_convert(self.timezone).dt.tz_localize(None).dt.normalize()
        calendar = self._panel_calendar(start or days.min(), end or days.max())
        # An audit only validates source labels within its declared session range.
        selected = days.between(calendar.start, calendar.end)
        return self._daily_bars_on_calendar(bars.loc[selected], days.loc[selected], calendar, instrument), calendar

    @staticmethod
    def _daily_bars_on_calendar(bars: pd.DataFrame, days: pd.Series, calendar: SessionCalendar,
                                instrument: str) -> pd.DataFrame:
        invalid = days.loc[~days.isin(calendar.sessions())]
        if not invalid.empty:
            raise ValueError(f'Cannot audit daily bars on non-trading dates for {instrument}: '
                             f'{sorted(invalid.dt.strftime("%Y-%m-%d").unique())}')
        starts = {day: calendar.session_on(day).segments[0][0] for day in days.unique()}
        return bars.assign(timestamp_utc=pd.to_datetime(days.map(starts), utc=True))

    def _resolve_entry(self, catalog: BarCatalog, frequency: str, instrument: str,
                       asset_class: str) -> CatalogEntry:
        try:
            entry = super()._resolve_entry(catalog, frequency, instrument, asset_class)
        except MissingBarDataError:
            entry = None
        if (self.legacy_snapshot and (entry is None or entry.kind != 'derived')
                and table_exists(self.connection, 'market_bars')
                and self.connection.execute('SELECT EXISTS(SELECT 1 FROM market_bars '
                    'WHERE instrument_id=? AND frequency=?)', [instrument, frequency]).fetchone()[0]):
            return (CatalogEntry(frequency, 'market_bars', dataset=asset_class, instrument_id=instrument)
                    if entry is None else replace(entry, table_name='market_bars',
                                                 timestamp_semantics='UTC_bar_start'))
        if entry is None:
            raise self._missing(frequency, instrument, catalog, asset_class, 'no catalog entry')
        return entry

    def _read_entry(self, entry: CatalogEntry, instrument: str, start, end, view: str,
                    catalog: BarCatalog, asset_class: str, *, require_coverage: bool = True
                    ) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        if entry.timestamp_semantics == 'UTC_bar_start':
            return super()._read_entry(entry, instrument, start, end, view, catalog, asset_class,
                                       require_coverage=entry.table_name != 'market_bars')
        bounds = self.connection.execute(f'SELECT min(date), max(date) FROM {identifier(entry.table_name)} '
                                         'WHERE stock_code=?', [instrument]).fetchone()
        if bounds[0] is None and not entry.coverage_start:
            raise self._missing(entry.frequency, instrument, catalog, asset_class, 'no instrument coverage')
        shift = Frequency.parse(entry.frequency).duration if entry.timestamp_semantics == _INTRADAY_SEMANTICS else pd.Timedelta(0)
        if (entry.timestamp_semantics == _INTRADAY_SEMANTICS and entry.quality_status == 'scanned'
                and table_exists(self.connection, 'bar_quality')):
            cstart, cend = self._quality_bounds(entry.frequency, instrument)
        else:
            cstart, cend = [pd.Timestamp(value).tz_localize(TIMEZONE).tz_convert('UTC') - shift for value in bounds]
        lower, upper = _utc_bound(start, False, TIMEZONE), _utc_bound(end, True, TIMEZONE)
        lower = cstart if lower is None else lower
        upper = cend if upper is None else upper
        if lower > upper:
            raise ValueError('start must not follow end')
        outside_start, outside_end = lower < cstart, upper > cend
        if start is not None and pd.Timestamp(start).tzinfo is None:
            outside_start = pd.Timestamp(start).date() < cstart.tz_convert(TIMEZONE).date()
        if end is not None and pd.Timestamp(end).tzinfo is None:
            outside_end = pd.Timestamp(end).date() > cend.tz_convert(TIMEZONE).date()
        if outside_start or outside_end:
            raise self._missing(entry.frequency, instrument, catalog, asset_class, 'insufficient coverage')
        return self.read_bars([instrument], entry.frequency, start_utc=lower, end_utc=upper,
                              source_table=entry.table_name, view=view), lower, upper

    def read_bars(self, instrument_ids: Sequence[str], frequency: str, *, start_utc=None, end_utc=None,
                  view: str = 'ETH', calendar=None, source_table: str = 'market_bars',
                  legacy_tz: str | None = None,
                  legacy_timestamp_anchor: str = LEGACY_5M_TIMESTAMP_ANCHOR,
                  data_version: str = 'legacy_unversioned') -> pd.DataFrame:
        parsed = Frequency.parse(frequency)
        if source_table == 'market_bars' or (parsed.is_valid and source_table == parsed.table_name):
            return super().read_bars(instrument_ids, frequency, start_utc=start_utc, end_utc=end_utc,
                                     view=view, calendar=calendar, source_table=source_table)
        expected = {'price_data_5m': '5min', 'price_data_daily': '1d'}
        for key, table in (load_data_access_config().price_tables if self.config is None
                           else self.config.get('price_tables', {})).items():
            expected[table] = Frequency.parse(key).canonical
        if expected.get(source_table) != frequency or frequency not in ('5min', '1d'):
            raise ContractValidationError('A-share source_table and frequency must match 5min or 1d')
        if view not in ('ETH', 'RTH'):
            raise ContractValidationError('view must be ETH or RTH')
        bars = self.read_legacy_price_table(source_table, instrument_ids, start_utc=start_utc,
            end_utc=end_utc, legacy_tz=legacy_tz, legacy_timestamp_anchor=legacy_timestamp_anchor,
            data_version=data_version, frequency=frequency)
        if frequency == '1d':
            return bars
        if calendar is None:
            calendar = calendars_for_bars(pd.DataFrame(bars.attrs['instrument_metadata']), bars)
        return self._filter_view(bars, calendar, view)

    @staticmethod
    def _legacy_metadata(ids: list[str], timezone: str, *, calendar_id: str = LEGACY_CALENDAR_ID) -> pd.DataFrame:
        return pd.DataFrame([dict(instrument_id=instrument, asset_class='equity',
                                  exchange=calendar_id,
                                  timezone=timezone, contract_expiry=None,
                                  price_adjustment='split_dividend_factor', point_value=1.0,
                                  rth_reference_calendar=calendar_id) for instrument in ids])

    def read_legacy_price_table(self, source_table: str, instrument_ids: Sequence[str], *,
                                start_utc=None, end_utc=None,
                                legacy_tz: str | None = None,
                                legacy_timestamp_anchor: str = LEGACY_5M_TIMESTAMP_ANCHOR,
                                data_version: str = 'legacy_unversioned',
                                frequency: str | None = None) -> pd.DataFrame:
        """Query old tables without DDL, migration, adjustment or writes.

        The 5m preset is Asia/Shanghai + bar_end. Override both semantics for
        another legacy source. Daily dates remain day identifiers at midnight.
        Supply a frozen data_version when using legacy results in research.
        """
        if frequency is None:
            frequency = {'price_data_5m': '5min', 'price_data_daily': '1d'}.get(source_table)
        if frequency not in ('5min', '1d'):
            raise ContractValidationError('Unsupported A-share source_table or frequency')
        legacy_tz = self.timezone if legacy_tz is None else legacy_tz
        ids = _instrument_ids(instrument_ids)
        start, end = _bounds(start_utc, end_utc)
        daily = frequency == '1d'
        spec = ExternalBarSourceSpec(
            timestamp_anchor='bar_start' if daily else legacy_timestamp_anchor,
            source_timezone=legacy_tz, volume_mode='interval',
            column_mapping={'date': 'timestamp', 'stock_code': 'instrument_id',
                            **{c: c for c in ('open', 'high', 'low', 'close', 'volume')}})
        placeholders = ','.join('?' for _ in ids)
        # Names are allowlisted; values are bound. Optional legacy fields survive.
        query = f'SELECT * FROM {identifier(source_table)} WHERE stock_code IN ({placeholders})'
        parameters = list(ids)
        shift = pd.Timedelta(0) if spec.timestamp_anchor == 'bar_start' else _duration(frequency)
        for bound, operator in [(start, '>='), (end, '<=')]:
            if bound is not None:
                query += f' AND date {operator} ?'
                parameters.append((bound + shift).tz_convert(legacy_tz).tz_localize(None))
        raw = self.connection.execute(query, parameters).fetchdf()
        metadata = self.legacy_instrument_metadata(ids, timezone=legacy_tz)
        bars = adapt_external_bars(raw, spec, metadata=metadata,
                                   frequency=frequency, source=f'legacy_{source_table}',
                                   data_version=data_version)
        bars.attrs['instrument_metadata'] = metadata.to_dict('records')
        if daily:
            bars.attrs['daily_timestamp_semantics'] = 'local_midnight_day_identifier'
        _record_raw_coverage(bars)
        return bars

    @classmethod
    def _source_columns(cls, conn: duckdb.DuckDBPyConnection, table: str,
                        adapter: DuckDBAdapter | None = None,
                        dataset=None) -> set[str]:
        from utils.data_store import table_name

        def parquet_columns() -> set[str]:
            if adapter is None or adapter.parquet_dir is None or dataset is None:
                return set()
            parquet_path = adapter.parquet_dir / dataset.parquet_subdir
            if not parquet_path.is_dir():
                return set()
            return {row[0] for row in conn.execute(
                'DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning=true)',
                [str(parquet_path / '**' / '*.parquet')],
            ).fetchall()}

        try:
            if adapter is not None and not adapter.duckdb_path and adapter.parquet_dir is not None and dataset is not None:
                return parquet_columns()
            return {row[0] for row in conn.execute(f'DESCRIBE {table_name(table)}').fetchall()}
        except duckdb.Error:
            try:
                return parquet_columns()
            except duckdb.Error:
                return set()

    @classmethod
    def read_panel(cls, instruments: Sequence[str], frequency: str | None, start: str,
                   end: str, fields: Sequence[str], *, config: DataAccessConfig,
                   connection: duckdb.DuckDBPyConnection | None = None,
                   adapter: DuckDBAdapter | None = None, date_field: str = 'date',
                   code_field: str = 'stock_code', additional_condition: str = '',
                   market_db_profile: MarketDbProfile | None = None,
                   check_profile: Callable[[duckdb.DuckDBPyConnection], MarketDbProfile] | None = None,
                   optional_fields: Sequence[str] = (),
                   ) -> pd.DataFrame:
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
        catalog = ASharePanelCatalog(config=config.as_dict())
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
        if check_profile is not None:
            market_db_profile = check_profile(conn)
        # Parquet declarations describe file datasets, not in-memory SQL tables.
        if adapter is None or adapter.parquet_dir is None:
            catalog = ASharePanelCatalog(conn, config=config.as_dict())
            try:
                entry = catalog.resolve(frequency, '*')
            except LookupError:
                entry = None
        if entry is None or entry.timestamp_semantics == 'UTC_bar_start':
            with cls(conn, config=config.as_dict(), market_db_profile=market_db_profile) as reader:
                bars = reader.read(instruments, frequency, start, end)
            panel = bars.rename(columns={'timestamp_utc': 'date', 'instrument_id': 'code'})
            panel['date'] = panel.date.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
            final_selected = list(selected)
            for opt in optional_fields:
                if opt in panel.columns and opt not in final_selected and opt not in excluded:
                    final_selected.append(opt)
            return panel[['date', 'code', *final_selected]].sort_values(
                ['code', 'date'], kind='mergesort', ignore_index=True)
        dataset = config.dataset(entry.table_name)
        extra_optional = []
        if optional_fields:
            available_cols = cls._source_columns(conn, dataset.table, adapter=adapter if adapter_panel else None, dataset=dataset)
            for opt in optional_fields:
                if opt in available_cols and opt not in selected and opt not in excluded:
                    selected = (*selected, opt)
                    extra_optional.append(opt)
        if adapter_panel:
            query_fields = [*fields, *extra_optional] if extra_optional else fields
            frame = adapter._fetch_panel(dataset, instruments, start, end, query_fields)
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
        persisted = (ASharePanelCatalog(conn, config=config.as_dict())
                     if table_exists(conn, entry.table_name) else catalog)
        warn_quality(conn, parsed.canonical, [(instrument,
            persisted.resolve(frequency, instrument),
            pd.Timestamp(start, tz=TIMEZONE).tz_convert('UTC'),
            (pd.Timestamp(end, tz=TIMEZONE) + pd.Timedelta(days=1)).tz_convert('UTC'))
            for instrument in instruments])
        return frame
