"""Project historical DuckDB prices onto the A-share session grid."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import duckdb
import pandas as pd

from utils.bars.catalog import identifier
from utils.bars.source import _bounds, _instrument_ids
from utils.pit import apply_pit_adjustment, normalize_price_money
from utils.session_calendar import SessionCalendar
from utils.time_axis import Frequency

from .panel import TIMEZONE


PRICE_TABLES = {'1d': 'price_data_daily', '5min': 'price_data_5m'}
_FIELDS = ('open', 'high', 'low', 'close', 'volume', 'money', 'factor',
           'pre_close', 'avg', 'high_limit', 'low_limit', 'paused')
_REQUIRED_FIELDS = {'open', 'high', 'low', 'close', 'volume', 'factor'}
BAR_KEYS = ['instrument_id', 'timestamp_utc', 'session_id']


def table_columns(connection: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    return {name: dtype for name, dtype, *_ in
            connection.execute(f'DESCRIBE {identifier(table)}').fetchall()}


def local_day(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.tz_convert(TIMEZONE).tz_localize(None).normalize()


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame({'instrument_id': pd.Series(dtype=object),
                         'timestamp_utc': pd.DatetimeIndex([], tz='UTC'),
                         'session_id': pd.Series(dtype=object),
                         **{field: pd.Series(dtype=float) for field in _FIELDS}})


def read_analysis_price_rows(
    connection: duckdb.DuckDBPyConnection, instruments: Sequence[str], start: date | str, end: date | str, *,
    table: str, date_column: str, instrument_column: str,
) -> pd.DataFrame:
    """Read the materialized A-share analysis price scale and configured aliases."""
    from utils.data_store import quote, table_name

    day, asset = quote(date_column), quote(instrument_column)
    return connection.execute(
        f"""
        SELECT {asset} AS stock_code, {day} AS date, open, high, low, close, volume, money
        FROM {table_name(table)}
        WHERE {asset} = ANY(?) AND CAST({day} AS DATE) BETWEEN ? AND ?
        ORDER BY stock_code, date
        """,
        [list(instruments), start, end],
    ).fetchdf()


class AShareBarFeed:
    def __init__(self, connection: duckdb.DuckDBPyConnection, calendar: SessionCalendar) -> None:
        self.connection = connection
        self.calendar = calendar

    @staticmethod
    def _frequency(frequency: Frequency) -> Frequency:
        parsed = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
        if parsed.canonical not in PRICE_TABLES:
            raise ValueError('jq_ashare_local bar feed supports 1d and 5min')
        return parsed

    def _query(self, ids: list[str], frequency: Frequency, start: pd.Timestamp,
               end: pd.Timestamp) -> pd.DataFrame:
        if start > end:
            return _empty_bars()
        daily = frequency.canonical == '1d'
        table = PRICE_TABLES[frequency.canonical]
        day_start, day_end = local_day(start), local_day(end)
        grid = self.calendar.bar_grid(frequency, day_start, day_end)
        if grid.empty:
            return _empty_bars()
        columns = table_columns(self.connection, table)
        missing = _REQUIRED_FIELDS - columns.keys()
        if missing:
            raise ValueError(f'{table} is missing price columns {sorted(missing)}')
        # DuckDB projects the source directly. NaN OHLCV and optional fields survive.
        fields = ', '.join(f"CAST({identifier(field) if field in columns else 'NULL'} AS DOUBLE) AS {identifier(field)}"
                           for field in _FIELDS)
        source_date = 'CAST(date AS TIMESTAMP_NS)' if columns.get('date') == 'VARCHAR' else 'date'
        if daily:
            bounds = f'{source_date} >= ? AND {source_date} < ?'
            parameters = [day_start, day_end + pd.Timedelta(days=1)]
        else:
            # The source stores local bar ends. Narrow the scan before validating grid keys.
            bounds = f'{source_date} BETWEEN CAST(? AS TIMESTAMP_NS) AND CAST(? AS TIMESTAMP_NS)'
            parameters = [(bound + frequency.duration).tz_convert(TIMEZONE).tz_localize(None).isoformat()
                          for bound in (start, end)]
        panel = self.connection.execute(
            f'SELECT stock_code AS instrument_id, CAST({source_date} AS TIMESTAMP_NS) AS date, {fields} '
            f'FROM {identifier(table)} WHERE stock_code IN (SELECT unnest(?)) '
            f'AND {bounds} ORDER BY stock_code, date', [ids, *parameters]).fetchdf()
        if daily:
            panel['session_id'] = panel.date.dt.strftime('%Y-%m-%d')
            panel = panel.merge(grid, how='left', on='session_id', validate='many_to_one')
        else:
            panel['timestamp_utc'] = (panel.date.dt.tz_localize(TIMEZONE).dt.tz_convert('UTC')
                                      - frequency.duration).astype('datetime64[ns, UTC]')
            panel = panel.merge(grid, how='left', on='timestamp_utc', validate='many_to_one')
        if panel[BAR_KEYS].isna().any().any() or panel.duplicated(BAR_KEYS).any():
            raise ValueError(f'{table} contains off-grid or duplicate price rows')
        result = panel.loc[panel.timestamp_utc.between(start, end), [*BAR_KEYS, *_FIELDS]].reset_index(drop=True)
        result.attrs['sources'] = [table]
        return result

    def read_execution_bars(self, instruments: Sequence[str], frequency: Frequency, *,
                            start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        ids = _instrument_ids(instruments)
        start, end = _bounds(start_utc, end_utc)
        if start is None or end is None:
            raise ValueError('start_utc and end_utc are required')
        return self._query(ids, self._frequency(frequency), start, end)

    def read_signal_bars(self, instruments: Sequence[str], frequency: Frequency, *,
                         start_utc: datetime, end_utc: datetime, as_of_utc: datetime) -> pd.DataFrame:
        ids = _instrument_ids(instruments)
        start, end = _bounds(start_utc, end_utc)
        as_of, _ = _bounds(as_of_utc, None)
        if start is None or end is None or as_of is None:
            raise ValueError('start_utc, end_utc and as_of_utc are required')
        parsed = self._frequency(frequency)
        daily = parsed.canonical == '1d'
        # Intraday grid keys are starts, but the source publishes completed bars at their ends.
        latest = as_of if daily else as_of - parsed.duration
        # Include the anchor even if the requested output window ends earlier.
        panel = self._query(ids, parsed, start, latest).rename(
            columns={'instrument_id': 'code', 'timestamp_utc': 'date'})
        panel = normalize_price_money(panel, source='local', frequency='daily' if daily else '5m')
        adjusted = apply_pit_adjustment(panel).rename(columns={'code': 'instrument_id', 'date': 'timestamp_utc'})
        return adjusted.loc[adjusted.timestamp_utc.le(end)].reset_index(drop=True)


class AShareDailyEvaluationFeed(AShareBarFeed):
    """Daily evaluation uses the historical materialized price scale.

    The native storage factor describes the separate PIT signal view. It must
    not be applied again to the historical analysis prices.
    """

    def __init__(self, feed: AShareBarFeed, *, pricing: pd.Series | None = None,
                 source_table: str = PRICE_TABLES['1d']) -> None:
        super().__init__(feed.connection, feed.calendar)
        self.pricing = pricing
        self.source_table = source_table

    def read_execution_bars(self, instruments: Sequence[str], frequency: Frequency, *,
                            start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        if frequency.canonical != '1d':
            raise ValueError('A-share daily evaluation feed requires 1d')
        if self.pricing is None:
            return super().read_execution_bars(
                instruments, frequency, start_utc=start_utc, end_utc=end_utc,
            ).rename(columns={'factor': 'pit_factor'})

        grid = self.calendar.bar_grid(frequency, local_day(pd.Timestamp(start_utc)), local_day(pd.Timestamp(end_utc)))
        snapshot = self.pricing.rename('close').reset_index().rename(columns={'asset': 'instrument_id'})
        snapshot = snapshot.loc[snapshot.instrument_id.isin(instruments)]
        snapshot['session_id'] = snapshot.pop('date').dt.strftime('%Y-%m-%d')
        bars = snapshot.merge(grid, on='session_id', validate='many_to_one')
        bars = bars.loc[bars.timestamp_utc.between(start_utc, end_utc)]
        bars = bars.assign(open=float('nan'), high=float('nan'), low=float('nan'))
        bars.attrs['sources'] = [self.source_table]
        return bars
