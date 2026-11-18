"""Read-only minute-Bar access, calendar views and explicit legacy conversion."""
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
    ExternalBarSourceSpec, adapt_external_bars, calendars_for_bars,
    validate_instrument_metadata,
)
from utils.session_calendar import SessionCalendar
from utils.time_axis import Frequency

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


class MarketBarReader:
    """Borrow a connection, or own a read-only file connection (context manager).

    Bounds are inclusive UTC Bar starts. Legacy daily rows use local midnight
    as their day identifier, frequency 1d, and are returned without intraday view
    filtering; they are a compatibility panel, not executable intraday labels.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None, *,
                 duckdb_path: str | Path | None = None) -> None:
        if (connection is None) == (duckdb_path is None):
            raise ValueError('Supply exactly one of connection or duckdb_path')
        self._owns_connection = connection is None
        self.connection = connection if connection is not None else duckdb.connect(
            str(duckdb_path), read_only=True)

    def close(self) -> None:
        if self._owns_connection:
            self.connection.close()

    def __enter__(self) -> MarketBarReader:
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


def read_market_bars(instrument_ids: Sequence[str], frequency: str, *,
                     connection: duckdb.DuckDBPyConnection | None = None,
                     duckdb_path: str | Path | None = None, **kwargs) -> pd.DataFrame:
    """Convenience read with deterministic ownership of file connections."""
    with MarketBarReader(connection, duckdb_path=duckdb_path) as reader:
        return reader.read_bars(instrument_ids, frequency, **kwargs)
