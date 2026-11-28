"""Explicit external minute-Bar normalization and atomic DuckDB imports."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import csv
from dataclasses import dataclass
from datetime import datetime
import json
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
from utils.market_db.storage import _require_same_hash
from utils.market_db.validation import (
    market_db_calendar_bounds, require_market_db_profile, validate_market_db_frequencies,
    validate_market_db_metadata, validate_market_db_provider,
)
from utils.session_calendar import SessionCalendar, VERIFIED_XSHG
from utils.time_axis.futures import futures_calendar
from utils.time_axis import BarSchedule, Frequency

METADATA_COLUMNS = ('instrument_id', 'asset_class', 'exchange', 'timezone',
                    'contract_expiry', 'price_adjustment', 'point_value',
                    'rth_reference_calendar')
OPTIONAL_METADATA_COLUMNS = ('calendar_id', 'product_root', 'contract_month', 'custom_sessions')
ALL_METADATA_COLUMNS = (*METADATA_COLUMNS, *OPTIONAL_METADATA_COLUMNS)
STORED_BAR_COLUMNS = (*BAR_COLUMNS, 'money', 'factor')
CONVERSION_MARKER = 'external_bar_conversion'


def calendar_name(metadata: Mapping) -> str:
    """Resolve supported exchange aliases; other exchanges use their PMC name."""
    explicit = metadata.get('calendar_id')
    if isinstance(explicit, str) and explicit:
        return 'XSHG' if explicit == VERIFIED_XSHG else explicit
    if metadata.get('asset_class') == 'futures':
        root = metadata.get('product_root')
        if metadata['exchange'] in ('SHFE', 'INE', 'DCE', 'CZCE', 'CFFEX', 'GFEX'):
            return futures_calendar(root)['calendar_id']
        if isinstance(root, str):
            raise ValueError(f'Explicit calendar_id is required for futures product_root={root}')
    return {'CME': 'CME Globex Equity', 'XSHE': 'XSHG', 'SSE': 'XSHG', 'SZSE': 'XSHG'}.get(
        metadata['exchange'], metadata['exchange'])


def instrument_calendar(metadata: Mapping, start: str, end: str) -> SessionCalendar:
    template = metadata.get('custom_sessions')
    return SessionCalendar(calendar_name(metadata), start, end,
                           rth_reference_calendar=metadata['rth_reference_calendar'],
                           custom_sessions=json.loads(template) if template is not None else None)


def _calendar_key(metadata: Mapping, start: str, end: str) -> tuple:
    return (calendar_name(metadata), metadata['rth_reference_calendar'],
            metadata.get('custom_sessions'), start, end)


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
        if row['contract_month'] is not None:
            if re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', row['contract_month']) is None:
                raise ContractValidationError('contract_month must be YYYY-MM or null')
        for name in ('instrument_id', 'asset_class', 'exchange', 'timezone',
                     'price_adjustment', 'rth_reference_calendar'):
            _text(row[name], name)
        try:
            ZoneInfo(row['timezone'])
            name, reference = calendar_name(row), row['rth_reference_calendar']
            if row['custom_sessions'] is None:
                names = (name, reference)
            else:
                if row['calendar_id'] is None:
                    raise ValueError('custom_sessions requires an explicit calendar_id')
                template = json.loads(row['custom_sessions'])
                if not isinstance(template, dict) or template.get('timezone') != row['timezone']:
                    raise ValueError('custom_sessions timezone must match instrument timezone')
                names = () if name == reference else (reference,)
            for name in names:
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
    observed = bars.groupby('instrument_id').timestamp_utc.agg(['min', 'max'])
    raw_coverage = bars.attrs.get('raw_coverage_by_instrument', {})
    for row in metadata.to_dict('records'):
        instrument = row['instrument_id']
        bounds = raw_coverage.get(instrument)
        if bounds is None and instrument in observed.index:
            bounds = observed.loc[instrument]
        if bounds is None or len(bounds) == 0:
            continue
        local = pd.DatetimeIndex(bounds).tz_convert(row['timezone'])
        start = (local.min() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        end = (local.max() + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        key = _calendar_key(row, start, end)
        if key not in cache:
            cache[key] = instrument_calendar(row, start, end)
        calendars[instrument] = cache[key]
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
        if self.volume_mode not in ('interval', 'cumulative', 'cumulative_session'):
            raise ContractValidationError('volume_mode must explicitly be interval or cumulative')
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
                        bucket_strategy: str | None = None,
                        schedule: BarSchedule | None = None,
                        through_utc: datetime | None = None,
                        ) -> pd.DataFrame:
    """Normalize mapped source data, optionally against the declared Bar schedule."""
    if not isinstance(spec, ExternalBarSourceSpec):
        raise ContractValidationError('spec must be an ExternalBarSourceSpec')
    # Recheck mutable mapping contents at the consumption boundary.
    spec.__post_init__()
    duration = _duration(frequency)
    if through_utc is not None and schedule is None:
        raise ContractValidationError('A Bar schedule is required to truncate signal inputs')
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
        if schedule is not None:
            if through_utc is not None:
                frame = frame.loc[frame.timestamp_utc.le(through_utc)].copy()
            starts_by_end = {}
            for ts in frame.timestamp_utc.unique():
                if ts in starts_by_end:
                    continue
                session = schedule.calendar.session_at(ts - pd.Timedelta(nanoseconds=1), schedule.view)
                if session is None:
                    raise ContractValidationError('bar_end must belong to an open Session')
                slots = schedule.for_session(session)
                starts_by_end.update(zip(slots.end_utc, slots.start_utc))
                if ts not in starts_by_end:
                    raise ContractValidationError('bar_end is off the Bar schedule grid')
            frame['timestamp_utc'] = pd.DatetimeIndex(
                frame.timestamp_utc.map(starts_by_end), tz='UTC').astype('datetime64[ns, UTC]')
        else:
            frame['timestamp_utc'] -= duration
    elif schedule is not None:
        if through_utc is not None:
            frame = frame.loc[frame.timestamp_utc.lt(through_utc)]
        ends = {ts: schedule.end_at(ts) for ts in frame.timestamp_utc.unique()}
        if through_utc is not None:
            frame = frame.loc[frame.timestamp_utc.isin(ts for ts, end in ends.items() if end <= through_utc)]
    frame = frame.assign(frequency=frequency, source=source, data_version=data_version)
    # Validate cumulative values and unique keys before differencing can hide errors.
    frame = validate_bar_frame(frame).reset_index(drop=True)
    _validate_factors(frame, metadata)
    if spec.volume_mode != 'interval':
        view = schedule.view if schedule is not None else 'ETH'
        calendars = calendar or calendars_for_bars(metadata, frame)
        sessions = []
        for row in frame.itertuples():
            cal = calendars.get(row.instrument_id) if isinstance(calendars, Mapping) else calendars
            if not isinstance(cal, SessionCalendar):
                raise ContractValidationError(f'Missing calendar for {row.instrument_id}')
            session = cal.session_at(row.timestamp_utc, view)
            if session is None:
                raise ContractValidationError('cumulative_session volume requires an open session')
            sessions.append(session.session_date)
        groups = frame.groupby([frame.instrument_id, pd.Series(sessions, index=frame.index)])
        for (instrument, day), group in groups:
            cal = calendars.get(instrument) if isinstance(calendars, Mapping) else calendars
            grid = cal.expected_bar_opens(day, frequency, view=view, bucket_strategy=bucket_strategy)
            positions = grid.get_indexer(group.timestamp_utc)
            if ((positions < 0).any() or (np.diff(positions) != 1).any()
                    or (spec.volume_mode == 'cumulative' and positions[0] != 0)):
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
                       derive: tuple[str, ...] = (), legacy_compatibility: bool = True,
                       slot_provider: str | None = None, check_only: bool = False,
                       _transaction_open: bool = False) -> int:
    """Insert metadata and bars atomically; duplicate keys fail without replacement.

    The caller owns the explicit connection and its database destination. This
    function neither opens a default database nor nests an existing transaction.
    Existing metadata must match exactly, so old bars cannot change interpretation.
    New callers set legacy_compatibility=False. Old domestic calls without a
    policy retain only market_bars and do not create frequency/quality entries.
    """
    profile = require_market_db_profile(connection, write=True)
    if slot_provider is not None:
        validate_market_db_provider(profile, slot_provider)
    if not isinstance(legacy_compatibility, bool):
        raise ContractValidationError('legacy_compatibility must be a boolean')
    metadata = validate_instrument_metadata(metadata)
    frame = validate_bar_frame(bars).reset_index(drop=True)
    if derive and len(frame.frequency.unique()) != 1:
        raise ContractValidationError('derive requires a single source frequency')
    validate_market_db_metadata(profile, metadata, calendar_name=calendar_name)
    validate_market_db_frequencies(profile, (*frame.frequency.unique(), *derive))
    if derive:
        source = Frequency.parse(frame.frequency.iloc[0])
        targets = [Frequency.parse(target) for target in derive]
        if len({target.canonical for target in targets}) != len(targets):
            raise ContractValidationError('Duplicate target frequency')
        if any(target.duration <= source.duration for target in targets):
            raise ContractValidationError('Target frequency must be coarser than source frequency')
    if 'exchange' in frame:
        pairs = frame[['instrument_id', 'exchange']].drop_duplicates()
        for instrument, exchange in pairs.itertuples(index=False, name=None):
            if not isinstance(exchange, str) or profile.exchange_aliases.get(exchange, exchange) != instrument.rsplit('.', 1)[1]:
                raise ContractValidationError(f'Bar exchange {exchange!r} disagrees with {instrument!r}')
    if 'asset_class' in frame:
        assets = frame.instrument_id.map(metadata.set_index('instrument_id').asset_class)
        if not frame.asset_class.eq(assets).fillna(False).all():
            raise ContractValidationError('Bar asset_class disagrees with instrument_metadata')
    _validate_factors(frame, metadata)
    for name in ('money', 'factor', 'open_interest'):
        if name not in frame:
            frame[name] = np.nan
    from utils.bar_catalog import CatalogEntry, create_bar_table, register_entry
    from utils.bar_quality import scan_quality
    from utils.bar_builder import code_version

    frame['_legacy_frequency'] = frame.frequency
    canonical = {value: Frequency.parse(value).canonical for value in frame.frequency.unique()}
    frame['frequency'] = frame.frequency.map(canonical)
    if frame.frequency.isna().any():
        raise ContractValidationError('Invalid frequency for physical Bar table')
    frame = validate_bar_frame(frame).reset_index(drop=True)
    calendars = calendars_for_bars(metadata, frame)
    metadata_by_id = metadata.set_index('instrument_id', drop=False).to_dict('index')
    scoped_calendars = {}
    grids = {}
    plans = []
    revision = code_version()
    for (instrument, frequency), group in frame.groupby(['instrument_id', 'frequency']):
        cal = calendars[instrument]
        owners = cal.session_dates_at(pd.DatetimeIndex(group.timestamp_utc))
        if owners.hasnans:
            raise ContractValidationError('Imported bars must belong to an open Session')
        first, last = owners.min(), owners.max()
        market_db_calendar_bounds(profile, first, last)
        if cal.timezone != profile.timezone:
            raise ContractValidationError(f'Calendar timezone {cal.timezone!r} disagrees with {profile.timezone!r}')
        if cal.is_domestic_futures and bucket_strategy is None and legacy_compatibility and not derive:
            # Preserve the stage-5/6.1 raw import entry without inventing a policy.
            continue
        meta = metadata_by_id[instrument]
        key = _calendar_key(meta, first, last)
        if key not in scoped_calendars:
            scoped_calendars[key] = instrument_calendar(meta, first, last)
        cal = scoped_calendars[key]
        schedule = BarSchedule(cal, frequency, bucket_strategy=bucket_strategy)
        grid_key = (cal.calendar_hash, frequency, schedule.bucket_strategy)
        if grid_key not in grids:
            grids[grid_key] = schedule.session_slots()
        grid = grids[grid_key]
        if not group.timestamp_utc.isin(grid.start_utc).all():
            raise ContractValidationError('Imported timestamp is off the Bar schedule grid')
        plans.append((instrument, frequency, schedule, first, last, group.index, grid))
    if check_only:
        return len(frame)
    with nullcontext() if _transaction_open else import_transaction(connection):
        _require_same_hash(require_market_db_profile(connection, write=True), profile, 'Import database changed')
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
        for instrument, frequency, schedule, first, last, indexes, grid in plans:
            name = create_bar_table(connection, frequency)
            incoming = frame.loc[indexes, [*STORED_BAR_COLUMNS, 'open_interest']].assign(
                n_source_bars=1, n_expected_bars=1, is_complete=True)
            # Migration appends columns, so insertion must follow the physical order.
            incoming = incoming[[row[0] for row in connection.execute(f'DESCRIBE {name}').fetchall()]]
            projection = ', '.join(
                f'coalesce("{column}", CAST(\'NaN\' AS DOUBLE)) AS "{column}"'
                if column in PRICE_COLUMNS else f'"{column}"' for column in incoming)
            connection.from_df(incoming).project(projection).insert_into(name)
            # Re-audit whole affected sessions so incremental imports can close gaps.
            observed = connection.execute(
                f'SELECT timestamp_utc, is_complete FROM {name} WHERE instrument_id=? '
                'AND timestamp_utc BETWEEN ? AND ?',
                [instrument, grid.start_utc.min(), grid.start_utc.max()]).fetchdf()
            observed['timestamp_utc'] = pd.to_datetime(observed.timestamp_utc, utc=True)
            scan_quality(connection, instrument, schedule, observed, grid=grid)
            register_entry(connection, CatalogEntry(frequency, name, instrument_id=instrument,
                source_version=','.join(sorted(set(incoming.data_version))), bucket_strategy=schedule.bucket_strategy,
                calendar_hash=schedule.calendar.calendar_hash, code_version=revision,
                coverage_start=first, coverage_end=last, quality_status='scanned'))
        if derive:
            from utils.bar_builder import build_bars
            for instrument, frequency, _, first, last, _, _ in plans:
                build_bars(connection, source=frequency, targets=derive, instruments=[instrument],
                           start=first, end=last, bucket_strategy=bucket_strategy,
                           _transaction_open=True)
    return len(frame)
