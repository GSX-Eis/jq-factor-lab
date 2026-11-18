"""Explicit external minute-Bar normalization and atomic DuckDB imports."""
from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pandas_market_calendars as pmc

from backtest_and_analysis.time_series.contracts import (
    BAR_COLUMNS, PRICE_COLUMNS, ContractValidationError, _duration, _numeric_column, _text,
    validate_bar_frame,
)
from data_pipeline.loaders._transactions import import_transaction
from utils.session_calendar import SessionCalendar
from utils.time_axis.futures import futures_calendar

METADATA_COLUMNS = ('instrument_id', 'asset_class', 'exchange', 'timezone',
                    'contract_expiry', 'price_adjustment', 'point_value',
                    'rth_reference_calendar')
OPTIONAL_METADATA_COLUMNS = ('calendar_id', 'product_root', 'contract_month')
ALL_METADATA_COLUMNS = (*METADATA_COLUMNS, *OPTIONAL_METADATA_COLUMNS)
STORED_BAR_COLUMNS = (*BAR_COLUMNS, 'money', 'factor')
CONVERSION_MARKER = 'external_bar_conversion'


def calendar_name(metadata: Mapping) -> str:
    """Resolve supported exchange aliases; other exchanges use their PMC name."""
    explicit = metadata.get('calendar_id')
    if isinstance(explicit, str) and explicit:
        return explicit
    if metadata.get('asset_class') == 'futures' and metadata['exchange'] in (
            'SHFE', 'INE', 'DCE', 'CZCE', 'CFFEX', 'GFEX'):
        return futures_calendar(metadata.get('product_root'))['calendar_id']
    return {'CME': 'CME Globex Equity', 'XSHE': 'XSHG'}.get(
        metadata['exchange'], metadata['exchange'])


def validate_instrument_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Validate concrete instruments before any database changes."""
    if not isinstance(metadata, pd.DataFrame) or not metadata.columns.is_unique:
        raise ContractValidationError('metadata must be a DataFrame with unique columns')
    frame = metadata.copy()
    for name in OPTIONAL_METADATA_COLUMNS:
        if name not in frame:
            frame[name] = None
        frame[name] = frame[name].astype(object).where(frame[name].notna(), None)
    if 'asset_class' not in frame:
        raise ContractValidationError('missing metadata columns: asset_class')
    if 'contract_expiry' not in frame:
        frame['contract_expiry'] = None
    if 'point_value' not in frame:
        frame['point_value'] = [1.0 if kind == 'equity' else None
                                for kind in frame.get('asset_class', [])]
    missing = set(METADATA_COLUMNS) - set(frame)
    if missing:
        raise ContractValidationError(f'missing metadata columns: {sorted(missing)}')
    if frame.instrument_id.duplicated().any():
        raise ContractValidationError('duplicate instrument_metadata keys')
    for row in frame.to_dict('records'):
        for name in OPTIONAL_METADATA_COLUMNS:
            if row[name] is not None:
                _text(row[name], name)
        if row['product_root'] is not None:
            futures_calendar(row['product_root'])
        if row['contract_month'] is not None:
            if re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', row['contract_month']) is None:
                raise ContractValidationError('contract_month must be YYYY-MM or null')
        for name in ('instrument_id', 'asset_class', 'exchange', 'timezone',
                     'price_adjustment', 'rth_reference_calendar'):
            _text(row[name], name)
        try:
            ZoneInfo(row['timezone'])
            for name in (calendar_name(row), row['rth_reference_calendar']):
                if name not in pmc.get_calendar_names():
                    raise ValueError(f'Unknown calendar: {name}')
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ContractValidationError(f'Unknown timezone or calendar: {exc}') from exc
        point = row['point_value']
        if (isinstance(point, (bool, np.bool_)) or not isinstance(point, Real)
                or not np.isfinite(point) or point <= 0):
            raise ContractValidationError('point_value must be a positive finite real number')
        expiry = row['contract_expiry']
        if expiry is not None and not isinstance(expiry, str):
            raise ContractValidationError('contract_expiry must be a string or None')
        code = row['instrument_id'].upper()
        if 'CONT' in code or 'MAIN' in code:
            raise ContractValidationError('continuous/main contracts are unsupported')
        if row['asset_class'] == 'futures':
            if (not expiry or re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?', expiry) is None
                    or re.search(r'(888|000)(?:\D*)$', code)):
                raise ContractValidationError('futures require a single concrete contract_expiry')
            try:
                pd.Timestamp(expiry)
            except ValueError as exc:
                raise ContractValidationError('Invalid contract_expiry') from exc
            if row['price_adjustment'] != 'none':
                raise ContractValidationError('futures price_adjustment must be none')
    return frame.loc[:, ALL_METADATA_COLUMNS].reset_index(drop=True)


def calendars_for_bars(metadata: pd.DataFrame, bars: pd.DataFrame) -> dict[str, SessionCalendar]:
    """Bind calendars to data dates, padding overnight ownership and label exits."""
    calendars = {}
    cache = {}
    for row in metadata.to_dict('records'):
        times = bars.loc[bars.instrument_id == row['instrument_id'], 'timestamp_utc']
        bounds = bars.attrs.get('raw_coverage_by_instrument', {}).get(row['instrument_id'])
        if bounds is not None:
            times = pd.Series(pd.DatetimeIndex(bounds))
        if times.empty:
            continue
        local = times.dt.tz_convert(row['timezone'])
        start = (local.min() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        end = (local.max() + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        key = (calendar_name(row), row['rth_reference_calendar'], start, end)
        if key not in cache:
            cache[key] = SessionCalendar(key[0], start, end, rth_reference_calendar=key[1])
        calendars[row['instrument_id']] = cache[key]
    return calendars


def _validate_factors(bars: pd.DataFrame, metadata: pd.DataFrame) -> None:
    missing = set(bars.instrument_id) - set(metadata.instrument_id)
    if missing:
        raise ContractValidationError(f'Missing instrument metadata: {sorted(missing)}')
    for name in ('money', 'factor', 'open_interest'):
        if name in bars and not bars[name].isna().all():
            values = _numeric_column(bars[name])
            if np.isinf(values).any():
                raise ContractValidationError(f'{name} must be finite or null')
            if name == 'open_interest' and (values < 0).any():
                raise ContractValidationError('open_interest must be nonnegative or null')
    if 'factor' in bars:
        futures = set(metadata.loc[metadata.asset_class == 'futures', 'instrument_id'])
        factors = bars.loc[bars.instrument_id.isin(futures), 'factor'].dropna()
        if (factors != 1.0).any():
            raise ContractValidationError('futures factor must be null or 1.0; adjustment unsupported')


@dataclass(frozen=True)
class ExternalBarSourceSpec:
    """No implicit timestamp, timezone, volume or mapping semantics."""

    timestamp_anchor: str | None = None
    source_timezone: str | None = None
    volume_mode: str | None = None
    column_mapping: Mapping[str, str] | None = None
    conversion_applied: bool = False

    def __post_init__(self) -> None:
        if self.timestamp_anchor not in ('bar_start', 'bar_end'):
            raise ContractValidationError('timestamp_anchor must explicitly be bar_start or bar_end')
        if not isinstance(self.source_timezone, str) or not self.source_timezone:
            raise ContractValidationError('source_timezone is required')
        try:
            ZoneInfo(self.source_timezone)
        except (KeyError, ValueError) as exc:
            raise ContractValidationError('source_timezone must be an IANA timezone') from exc
        if self.volume_mode not in ('interval', 'cumulative_session'):
            raise ContractValidationError('volume_mode must explicitly be interval or cumulative_session')
        if (not isinstance(self.column_mapping, Mapping) or not self.column_mapping
                or any(not isinstance(k, str) or not isinstance(v, str)
                       for k, v in self.column_mapping.items())):
            raise ContractValidationError('column_mapping is required and must map column names')
        if self.conversion_applied:
            raise ContractValidationError('external bars already converted')


def _external_timestamps(values: pd.Series, timezone: str) -> pd.DatetimeIndex:
    converted = []
    zone = ZoneInfo(timezone)
    for value in values:
        try:
            if not isinstance(value, (str, datetime)):
                raise ValueError('timestamp must be a datetime or timestamp string')
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                raise ValueError('timestamp must not be missing')
            if ts.tzinfo is None:
                if timezone in ('UTC', 'Etc/UTC'):
                    raise ValueError('UTC source requires explicitly aware UTC timestamps')
                ts = ts.tz_localize(timezone, ambiguous='raise', nonexistent='raise')
            elif ts.utcoffset() != ts.tz_convert(zone).utcoffset():
                raise ValueError('timestamp offset conflicts with source_timezone')
            converted.append(ts.tz_convert('UTC'))
        except Exception as exc:
            raise ContractValidationError(f'Invalid timestamp for source_timezone {timezone}: {exc}') from exc
    return pd.DatetimeIndex(converted, tz='UTC').astype('datetime64[ns, UTC]')


def adapt_external_bars(data: pd.DataFrame | str | Path, spec: ExternalBarSourceSpec, *,
                        metadata: pd.DataFrame, frequency: str, source: str,
                        data_version: str, instrument_id: str | None = None,
                        calendar: SessionCalendar | Mapping[str, SessionCalendar] | None = None,
                        bucket_strategy: str | None = None
                        ) -> pd.DataFrame:
    """Normalize IQFeed or another explicitly mapped source exactly once."""
    if not isinstance(spec, ExternalBarSourceSpec):
        raise ContractValidationError('spec must be an ExternalBarSourceSpec')
    # Recheck mutable mapping contents at the consumption boundary.
    spec.__post_init__()
    duration = _duration(frequency)
    metadata = validate_instrument_metadata(metadata)
    if not isinstance(data, pd.DataFrame):
        with Path(data).open(encoding='utf-8-sig', newline='') as handle:
            header = next(csv.reader(handle), [])
        if len(header) != len(set(header)):
            raise ContractValidationError('source column names must be unique')
        text_columns = {name: 'string' for name, target in spec.column_mapping.items()
                        if target in ('instrument_id', 'timestamp')}
        data = pd.read_csv(data, dtype=text_columns)
    if data.attrs.get(CONVERSION_MARKER) or set(BAR_COLUMNS).issubset(data.columns):
        raise ContractValidationError('external bars already converted')
    if not data.columns.is_unique or set(spec.column_mapping) - set(data):
        raise ContractValidationError('column_mapping references missing or duplicate source columns')
    frame = data.rename(columns=dict(spec.column_mapping)).copy()
    required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(spec.column_mapping.values()) or not frame.columns.is_unique:
        raise ContractValidationError('column_mapping must cover required columns without duplicates')
    if instrument_id is not None:
        _text(instrument_id, 'instrument_id')
        if 'instrument_id' in frame and not frame.instrument_id.eq(instrument_id).all():
            raise ContractValidationError('instrument_id conflicts with mapped source instrument')
        frame['instrument_id'] = instrument_id
    elif 'instrument_id' not in spec.column_mapping.values():
        raise ContractValidationError('instrument_id must be supplied or explicitly mapped')
    frame['timestamp_utc'] = _external_timestamps(frame.pop('timestamp'), spec.source_timezone)
    if spec.timestamp_anchor == 'bar_end':
        frame['timestamp_utc'] -= duration
    frame = frame.assign(frequency=frequency, source=source, data_version=data_version)
    # Validate cumulative values and unique keys before differencing can hide errors.
    frame = validate_bar_frame(frame).reset_index(drop=True)
    _validate_factors(frame, metadata)
    if spec.volume_mode == 'cumulative_session':
        calendars = calendar or calendars_for_bars(metadata, frame)
        sessions = []
        for row in frame.itertuples():
            cal = calendars.get(row.instrument_id) if isinstance(calendars, Mapping) else calendars
            if not isinstance(cal, SessionCalendar):
                raise ContractValidationError(f'Missing calendar for {row.instrument_id}')
            session = cal.session_at(row.timestamp_utc)
            if session is None:
                raise ContractValidationError('cumulative_session volume requires an open session')
            sessions.append(session.session_date)
        groups = frame.groupby([frame.instrument_id, pd.Series(sessions, index=frame.index)])
        for (instrument, day), group in groups:
            cal = calendars.get(instrument) if isinstance(calendars, Mapping) else calendars
            grid = cal.expected_bar_opens(day, frequency, bucket_strategy=bucket_strategy)
            positions = grid.get_indexer(group.timestamp_utc)
            if (positions < 0).any() or (np.diff(positions) != 1).any():
                raise ContractValidationError('cumulative_session volume has off-grid or missing bars')
        difference = groups.volume.diff().fillna(frame.volume)
        if (difference < -1e-9).any():
            raise ContractValidationError('cumulative_session volume decreased within session')
        frame['volume'] = difference.clip(lower=0)
    frame = validate_bar_frame(frame).reset_index(drop=True)
    frame.attrs.update({CONVERSION_MARKER: True, 'timestamp_anchor': 'bar_start',
                        'source_timezone': 'UTC', 'volume_mode': 'interval'})
    return frame


def store_instrument_metadata(connection: duckdb.DuckDBPyConnection, metadata: pd.DataFrame) -> None:
    """Persist validated metadata inside the caller's Bar writer transaction."""
    metadata = validate_instrument_metadata(metadata)
    connection.execute('''CREATE TABLE IF NOT EXISTS instrument_metadata (
        instrument_id VARCHAR PRIMARY KEY, asset_class VARCHAR NOT NULL,
        exchange VARCHAR NOT NULL, timezone VARCHAR NOT NULL, contract_expiry VARCHAR,
        price_adjustment VARCHAR NOT NULL, point_value DOUBLE NOT NULL,
        rth_reference_calendar VARCHAR NOT NULL)''')
    for name in OPTIONAL_METADATA_COLUMNS:
        connection.execute(f'ALTER TABLE instrument_metadata ADD COLUMN IF NOT EXISTS {name} VARCHAR')
    for row in metadata.itertuples(index=False, name=None):
        existing = connection.execute(
            f'SELECT {", ".join(ALL_METADATA_COLUMNS)} FROM instrument_metadata WHERE instrument_id = ?',
            [row[0]]).fetchone()
        if existing is not None and existing != row:
            raise ContractValidationError(f'Conflicting metadata for {row[0]}')
        if existing is None:
            columns = ', '.join(ALL_METADATA_COLUMNS)
            placeholders = ', '.join('?' for _ in ALL_METADATA_COLUMNS)
            connection.execute(f'INSERT INTO instrument_metadata ({columns}) VALUES ({placeholders})', row)


def import_market_bars(connection: duckdb.DuckDBPyConnection, bars: pd.DataFrame,
                       metadata: pd.DataFrame, *, bucket_strategy: str | None = None,
                       derive: tuple[str, ...] = (), legacy_compatibility: bool = True) -> int:
    """Insert metadata and bars atomically; duplicate keys fail without replacement.

    The caller owns the explicit connection and its database destination. This
    function neither opens a default database nor nests an existing transaction.
    Existing metadata must match exactly, so old bars cannot change interpretation.
    New callers set legacy_compatibility=False. Old domestic calls without a
    policy retain only market_bars and do not create frequency/quality entries.
    """
    if not isinstance(legacy_compatibility, bool):
        raise ContractValidationError('legacy_compatibility must be a boolean')
    metadata = validate_instrument_metadata(metadata)
    frame = validate_bar_frame(bars).reset_index(drop=True)
    if derive and len(frame.frequency.unique()) != 1:
        raise ContractValidationError('derive requires a single source frequency')
    _validate_factors(frame, metadata)
    for name in ('money', 'factor', 'open_interest'):
        if name not in frame:
            frame[name] = np.nan
    from utils.bar_catalog import CatalogEntry, create_bar_table, register_entry
    from utils.bar_quality import scan_quality
    from utils.time_axis import BarSchedule, Frequency
    from utils.bar_builder import instrument_calendar, code_version

    frame['_legacy_frequency'] = frame.frequency
    frame['frequency'] = frame.frequency.map(lambda value: Frequency.parse(value).canonical)
    if frame.frequency.isna().any():
        raise ContractValidationError('Invalid frequency for physical Bar table')
    frame = validate_bar_frame(frame).reset_index(drop=True)
    calendars = calendars_for_bars(metadata, frame)
    plans = []
    revision = code_version()
    for (instrument, frequency), group in frame.groupby(['instrument_id', 'frequency']):
        cal = calendars[instrument]
        if cal.is_domestic_futures and bucket_strategy is None and legacy_compatibility and not derive:
            # Preserve the stage-5/6.1 raw import entry without inventing a policy.
            continue
        # Calendar construction, grid validation and policy checks precede any DDL.
        BarSchedule(cal, frequency, bucket_strategy=bucket_strategy)
        owners = [cal.session_at(ts) for ts in group.timestamp_utc]
        if any(owner is None for owner in owners):
            raise ContractValidationError('Imported bars must belong to an open Session')
        first, last = min(owner.session_date for owner in owners), max(owner.session_date for owner in owners)
        meta = metadata.loc[metadata.instrument_id.eq(instrument)].iloc[0].to_dict()
        cal = instrument_calendar(meta, first, last)
        schedule = BarSchedule(cal, frequency, bucket_strategy=bucket_strategy)
        grid = pd.concat([schedule.for_session(day.strftime('%Y-%m-%d')) for day in cal.sessions()])
        if not group.timestamp_utc.isin(grid.start_utc).all():
            raise ContractValidationError('Imported timestamp is off the Bar schedule grid')
        plans.append((instrument, frequency, schedule, first, last))
    with import_transaction(connection):
        store_instrument_metadata(connection, metadata)
        if legacy_compatibility:
            connection.execute('''CREATE TABLE IF NOT EXISTS market_bars (
                instrument_id VARCHAR NOT NULL, timestamp_utc TIMESTAMPTZ NOT NULL,
                frequency VARCHAR NOT NULL, open DOUBLE NOT NULL, high DOUBLE NOT NULL,
                low DOUBLE NOT NULL, close DOUBLE NOT NULL, volume DOUBLE NOT NULL,
                source VARCHAR NOT NULL, data_version VARCHAR NOT NULL, money DOUBLE, factor DOUBLE,
                PRIMARY KEY (instrument_id, timestamp_utc, frequency))''')
        stored = frame.loc[:, STORED_BAR_COLUMNS].assign(frequency=frame['_legacy_frequency'])
        # A relation avoids registering a persistent caller-visible temporary view.
        # pandas NaN normally becomes SQL NULL; preserve bad-price evidence in
        # the NOT NULL price columns as IEEE NaN for downstream invalid_price.
        projection = ', '.join(
            f'coalesce("{name}", CAST(\'NaN\' AS DOUBLE)) AS "{name}"'
            if name in PRICE_COLUMNS else f'"{name}"' for name in STORED_BAR_COLUMNS)
        if legacy_compatibility:
            connection.from_df(stored).project(projection).insert_into('market_bars')
        # Stage-5 callers mutate market_bars directly. Keep that compatibility
        # snapshot; stage-6 consumers read only the per-frequency physical tables.
        for instrument, frequency, schedule, first, last in plans:
            name = create_bar_table(connection, frequency)
            incoming = frame.loc[frame.instrument_id.eq(instrument) & frame.frequency.eq(frequency),
                                 [*STORED_BAR_COLUMNS, 'open_interest']].assign(
                                     n_source_bars=1, n_expected_bars=1, is_complete=True)
            # Migration appends columns, so insertion must follow the physical order.
            incoming = incoming[[row[0] for row in connection.execute(f'DESCRIBE {name}').fetchall()]]
            projection = ', '.join(
                f'coalesce("{column}", CAST(\'NaN\' AS DOUBLE)) AS "{column}"'
                if column in PRICE_COLUMNS else f'"{column}"' for column in incoming)
            connection.from_df(incoming).project(projection).insert_into(name)
            # Re-audit whole affected sessions so incremental imports can close gaps.
            observed = connection.execute(f'SELECT * FROM {name} WHERE instrument_id=?',
                                          [instrument]).fetchdf()
            observed['timestamp_utc'] = pd.to_datetime(observed.timestamp_utc, utc=True)
            scan_quality(connection, instrument, schedule, observed)
            register_entry(connection, CatalogEntry(frequency, name, instrument_id=instrument,
                source_version=','.join(sorted(set(incoming.data_version))), bucket_strategy=schedule.bucket_strategy,
                calendar_hash=schedule.calendar.calendar_hash, code_version=revision,
                coverage_start=first, coverage_end=last, quality_status='scanned'))
    if derive:
        from utils.bar_builder import build_bars
        for instrument, frequency, _, first, last in plans:
            build_bars(connection, source=frequency, targets=derive, instruments=[instrument],
                       start=first, end=last, bucket_strategy=bucket_strategy)
    return len(frame)
