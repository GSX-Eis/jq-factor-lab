"""MarketSlot components assembled from one versioned SlotConfig."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
import json
from pathlib import Path

import pandas as pd

from backtest_and_analysis.time_series.contracts import BAR_COLUMNS, _utc
from data_pipeline.loaders.market_bar_importer import (
    ExternalBarSourceSpec, _external_timestamps, adapt_external_bars,
    instrument_calendar, validate_instrument_metadata,
)
from utils.bars.source import _bounds, _instrument_ids
from utils.market_db import MarketDbProfile, MarketDbProfileError, open_market_db
from utils.market_db.errors import MARKET_DB_PROFILE_INVALID
from utils.market_db.validation import (
    market_db_calendar_bounds, validate_market_db_calendar, validate_market_db_execution_rules,
    validate_market_db_frequencies, validate_market_db_provider,
)
from utils.session_calendar import SessionCalendar
from utils.slots import ExecutionProfile, InstrumentSpec, MarketSlot, SlotConfig
from utils.time_axis import BarSchedule, Frequency

from .source import FrameSource


_KEYS = ['instrument_id', 'timestamp_utc', 'session_id']
_AVAILABLE = 'available_at_utc'


def _calendar(config: SlotConfig, profile: MarketDbProfile | None = None) -> tuple[SessionCalendar, str | None]:
    provider = config.calendar
    options = provider.options
    required = ('calendar_id', 'start', 'end') if profile is None else ('calendar_id',)
    for field in required:
        if not isinstance(options.get(field), str) or not options[field].strip():
            raise ValueError(f'calendar requires explicit {field}')
    name = options['calendar_id']
    if profile is None:
        start, end = options['start'], options['end']
    else:
        validate_market_db_calendar(profile, name, options.get('timezone', profile.timezone))
        start, end = market_db_calendar_bounds(profile, options.get('start'), options.get('end'))
        if provider.provider == 'custom_sessions' and profile.calendar_source != 'custom_sessions':
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, 'Database does not declare custom_sessions')
    reference = options.get('rth_reference_calendar', name)
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError('rth_reference_calendar must be a calendar ID')
    if provider.provider == 'pmc' and 'custom_sessions' not in options:
        custom = None
    elif provider.provider == 'custom_sessions' and isinstance(options.get('custom_sessions'), Mapping):
        custom = json.dumps(options['custom_sessions'], default=dict, sort_keys=True, separators=(',', ':'))
    else:
        raise ValueError('calendar requires pmc or a custom_sessions provider with a session template')
    try:
        calendar = instrument_calendar(
            dict(calendar_id=name, rth_reference_calendar=reference, custom_sessions=custom),
            start, end,
        )
    except RuntimeError as exc:
        raise ValueError(f'Cannot build calendar {name!r} with RTH reference {reference!r}') from exc
    if profile is not None:
        validate_market_db_calendar(profile, calendar.calendar_id, calendar.timezone)
    return calendar, custom


def _specs(options: Mapping) -> dict[str, InstrumentSpec]:
    instruments = options.get('instruments')
    if not isinstance(instruments, Mapping) or not instruments:
        raise ValueError('universe requires an instruments mapping of instrument IDs to specs')
    specs = {}
    for instrument, spec in instruments.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f'Expected a specification for {instrument}')
        spec = dict(spec)
        try:
            for field in ('listing_date', 'expiry_date'):
                if spec.get(field) is not None:
                    spec[field] = date.fromisoformat(spec[field])
            specs[instrument] = InstrumentSpec(instrument_id=instrument, **spec)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid instrument specification for {instrument}: {exc}') from exc
    return specs


@dataclass(frozen=True)
class DeclarativeUniverse:
    specs: Mapping[str, InstrumentSpec]
    static: Mapping[str, tuple[str, ...]]
    membership: FrameSource | None

    def instrument_spec(self, instrument_id: str) -> InstrumentSpec:
        return self.specs[instrument_id]

    def resolve_active_contract(self, instrument_id: str, *, as_of_utc: datetime) -> str:
        _utc(as_of_utc, 'as_of_utc')
        return self.instrument_spec(instrument_id).instrument_id

    def resolve_universe(self, universe_id: str, *, as_of_utc: datetime) -> Sequence[str]:
        _utc(as_of_utc, 'as_of_utc')
        if universe_id in self.static:
            return self.static[universe_id]
        if self.membership is None:
            raise KeyError(f'Unknown universe: {universe_id!r}')
        data = self.membership.read_mapped()
        _validate_events(data, ('universe_id', 'instrument_id'), self.specs)
        rows = data.loc[data.universe_id.eq(universe_id)].copy()
        if rows.empty:
            raise KeyError(f'Unknown universe: {universe_id!r}')
        rows = rows.loc[rows[_AVAILABLE].le(as_of_utc)]
        return tuple(sorted(rows.loc[rows[_AVAILABLE].eq(rows[_AVAILABLE].max()), 'instrument_id']))


def _validate_events(data: pd.DataFrame, keys: tuple[str, ...], specs: Mapping | None = None) -> None:
    missing = {*keys, _AVAILABLE} - set(data)
    if missing:
        raise ValueError(f'Missing event columns: {sorted(missing)}')
    for key in keys:
        if any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in data[key]):
            raise ValueError(f'{key} must contain nonempty IDs without surrounding whitespace')
    if specs is not None and (unknown := set(data.instrument_id) - set(specs)):
        raise ValueError(f'Unknown member instruments: {sorted(unknown)}')
    data[_AVAILABLE] = _external_timestamps(data[_AVAILABLE], 'UTC')
    if data.duplicated([*keys, _AVAILABLE]).any():
        raise ValueError(f'Duplicate event keys: {(*keys, _AVAILABLE)}')


def _align_events(grid: pd.DataFrame, events: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return pd.merge_asof(
        grid.assign(instrument_id=grid.instrument_id.astype(object),
                    timestamp_utc=grid.timestamp_utc.astype('datetime64[ns, UTC]')).sort_values('timestamp_utc'),
        events[['instrument_id', _AVAILABLE, *columns]].assign(
            instrument_id=events.instrument_id.astype(object)).sort_values(_AVAILABLE),
        by='instrument_id', left_on='timestamp_utc', right_on=_AVAILABLE, direction='backward',
    )


@dataclass(frozen=True)
class DeclarativeAuxFeatures:
    source: FrameSource | None

    def read_aux_features(self, grid: pd.DataFrame, *, features: Sequence[str]) -> pd.DataFrame:
        if (not isinstance(features, Sequence) or isinstance(features, (str, bytes))
                or any(not isinstance(name, str) or not name for name in features)):
            raise ValueError('features must be a sequence of nonempty field names')
        columns = list(dict.fromkeys(features))
        if set(columns) & {*_KEYS, _AVAILABLE}:
            raise ValueError('Feature names must differ from the grid and availability columns')
        if self.source is None or not columns:
            return grid.loc[:, _KEYS].iloc[:0].assign(
                available_at_utc=pd.DatetimeIndex([], tz='UTC'), **{column: [] for column in columns})
        data = self.source.read_mapped()
        _validate_events(data, ('instrument_id',))
        if missing := set(columns) - set(data):
            raise ValueError(f'Unknown auxiliary features: {sorted(missing)}')
        aligned = _align_events(grid[_KEYS], data, columns)
        return aligned.loc[aligned[_AVAILABLE].notna()].reset_index(drop=True)


@dataclass(frozen=True)
class DeclarativeMicrostructure:
    calendar: SessionCalendar
    execution_profile: ExecutionProfile
    source: FrameSource | None

    def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
        def in_trading_segment(ts: pd.Timestamp, session_id: str, price_field: str) -> bool:
            _utc(ts, 'timestamp_utc')
            session = self.calendar.session_on(session_id)
            if session is None:
                return False
            # Grid callers without an execution field also use the session's daily close.
            if price_field == '' and ts == session.segments[-1][1]:
                return True
            return any(left <= ts < right or (price_field == 'close' and ts == right)
                       for left, right in session.segments)

        price_fields = grid.get('execution_price_field', pd.Series('', index=grid.index))
        result = pd.Series([in_trading_segment(ts, day, field)
                            for ts, day, field in zip(grid.timestamp_utc, grid.session_id, price_fields)],
                           index=grid.index, name='tradable', dtype=bool)
        if self.source is not None and not grid.empty:
            data = self.source.read_mapped()
            _validate_events(data, ('instrument_id',))
            if 'tradable' not in data or not pd.api.types.is_bool_dtype(data.tradable) or data.tradable.isna().any():
                raise ValueError('Microstructure source requires a non-null boolean tradable column')
            keys = grid[['timestamp_utc']].assign(instrument_id=instrument_id, position=range(len(grid)))
            mask = _align_events(keys, data, ['tradable']).sort_values('position').tradable
            result &= mask.astype('boolean').fillna(True).to_numpy(dtype=bool)
        return result


class DeclarativeBarFeed:
    def __init__(self, config: SlotConfig, calendar: SessionCalendar, specs: Mapping[str, InstrumentSpec],
                 custom_sessions: str | None, source: FrameSource) -> None:
        options = config.bar_feed.options
        volume = options.get('volume_mode')
        self.spec = ExternalBarSourceSpec(
            timestamp_anchor=options.get('timestamp_anchor'), source_timezone=options.get('source_timezone'),
            volume_mode='cumulative' if volume == 'cumulative_session' else volume,
            column_mapping=options.get('column_mapping'),
        )
        self.source = source
        self.frequency = Frequency.parse(options.get('frequency'))
        if not self.frequency.is_valid:
            raise ValueError('bar_feed requires an intraday, 1d or 1w frequency')
        if source.market_db_profile is not None:
            validate_market_db_frequencies(source.market_db_profile, (self.frequency.canonical,))
        self.calendar = calendar
        self.source_name = options.get('source', config.slot_id)
        self.data_version = options.get('data_version')
        if (not isinstance(self.source_name, str) or not self.source_name.strip()
                or not isinstance(self.data_version, str) or not self.data_version.strip()):
            raise ValueError('bar_feed source and data_version must be nonempty strings')
        self.instrument_id = options.get('instrument_id')
        if self.instrument_id is not None and (
                not isinstance(self.instrument_id, str) or self.instrument_id not in specs):
            raise ValueError('bar_feed instrument_id must have an instrument specification')
        self.bucket_strategy = options.get('bucket_strategy')
        self.schedule = BarSchedule(calendar, self.frequency, bucket_strategy=self.bucket_strategy)
        self.metadata = validate_instrument_metadata(pd.DataFrame([
            dict(instrument_id=spec.instrument_id, asset_class=spec.asset_class,
                 exchange=options.get('exchange', spec.instrument_id.rsplit('.', 1)[1]
                     if source.market_db_profile is not None and '.' in spec.instrument_id else calendar.calendar_id),
                 timezone=calendar.timezone,
                 calendar_id=calendar.calendar_id, rth_reference_calendar=calendar.rth_reference_calendar,
                 custom_sessions=custom_sessions, price_adjustment='none', point_value=spec.contract_multiplier,
                 contract_expiry=spec.expiry_date.isoformat() if spec.expiry_date is not None else None)
            for spec in specs.values()
        ]))

    def load(self, *, instruments: Sequence[str] | None = None, as_of_utc: datetime | None = None,
             bucket_strategy: str | None = None) -> pd.DataFrame:
        data = self.source.read()
        if instruments is not None:
            instrument_column = next((src for src, dest in self.spec.column_mapping.items()
                                      if dest == 'instrument_id'), None)
            if instrument_column is not None:
                data = data.loc[data[instrument_column].isin(instruments)]
            elif self.instrument_id not in instruments:
                data = data.iloc[:0]
        schedule = self.schedule if bucket_strategy is None else BarSchedule(
            self.calendar, self.frequency, bucket_strategy=bucket_strategy)
        return adapt_external_bars(
            data, self.spec, metadata=self.metadata, calendar=self.calendar, frequency=self.frequency.canonical,
            source=self.source_name, data_version=self.data_version, instrument_id=self.instrument_id,
            bucket_strategy=schedule.bucket_strategy, schedule=schedule, through_utc=as_of_utc,
        )

    def _read(self, instruments: Sequence[str], frequency: Frequency, *, start_utc: datetime,
              end_utc: datetime, as_of_utc: datetime | None = None) -> pd.DataFrame:
        if frequency.canonical != self.frequency.canonical:
            raise ValueError(f'Declarative feed serves {self.frequency.canonical}, got {frequency.raw!r}')
        start, end = _bounds(start_utc, end_utc)
        if start is None or end is None:
            raise ValueError('Declarative bar reads require start_utc and end_utc')
        ids = _instrument_ids(instruments)
        bars = self.load(instruments=ids, as_of_utc=as_of_utc)
        # Only the signal path needs completion, but both views validate the grid.
        if self.frequency.kind == 'intraday':
            bars = bars.loc[bars.timestamp_utc.between(start, end)].copy()
        ends = {ts: self.schedule.end_at(ts) for ts in bars.timestamp_utc.unique()}
        end_times = pd.to_datetime(bars.timestamp_utc.map(ends), utc=True)
        days = bars.timestamp_utc if self.frequency.kind == 'intraday' else end_times - pd.Timedelta(1, 'ns')
        owners = [self.calendar.session_at(ts).session_date for ts in days]
        bars = bars.assign(session_id=pd.Series(owners, index=bars.index, dtype=object))
        if self.frequency.kind != 'intraday':
            keys = self.calendar.bar_grid(self.frequency).set_index('session_id').timestamp_utc
            bars = bars.assign(timestamp_utc=bars.session_id.map(keys))
        mask = bars.timestamp_utc.between(start, end)
        extras = [field for field in ('money', 'factor', 'open_interest') if field in bars]
        return bars.loc[mask, [*BAR_COLUMNS, 'session_id', *extras]].reset_index(drop=True)

    def read_signal_bars(self, instruments: Sequence[str], frequency: Frequency, *, start_utc: datetime,
                         end_utc: datetime, as_of_utc: datetime) -> pd.DataFrame:
        _utc(as_of_utc, 'as_of_utc')
        return self._read(instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc)

    def read_execution_bars(self, instruments: Sequence[str], frequency: Frequency, *, start_utc: datetime,
                            end_utc: datetime) -> pd.DataFrame:
        return self._read(instruments, frequency, start_utc=start_utc, end_utc=end_utc)


def build_declarative_slot(config: SlotConfig, *, base_dir: Path) -> MarketSlot:
    source = FrameSource(config.bar_feed, base_dir, ('instrument_id', 'timestamp'))
    membership = (None if config.universe.provider == 'static' else
                  FrameSource(config.universe, base_dir, ('instrument_id', 'universe_id', _AVAILABLE)))
    aux = (None if config.aux_features.provider == 'none' else
           FrameSource(config.aux_features, base_dir, ('instrument_id', _AVAILABLE)))
    micro = (None if config.microstructure.provider == 'open' else
             FrameSource(config.microstructure, base_dir, ('instrument_id', _AVAILABLE)))
    databases = {frame.database_path for frame in (source, membership, aux, micro)
                 if frame is not None and frame.provider.provider == 'duckdb_table'}
    profile = None
    if len(databases) > 1:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, 'A Slot can bind only one market database')
    if databases:
        with open_market_db(next(iter(databases))) as (_, profile, _):
            validate_market_db_provider(profile, config.bar_feed.provider, domain=config.market_domain)
        source, membership, aux, micro = (
            replace(frame, market_db_profile=profile) if frame is not None else None
            for frame in (source, membership, aux, micro)
        )
        st_filter = False
        if aux is not None:
            aux_schema = aux.read_mapped(schema_only=True)
            st_filter = 'is_st' in aux_schema
            if st_filter:
                _validate_events(aux_schema, ('instrument_id',))
        validate_market_db_execution_rules(
            profile, config.bar_feed.provider, st_filter=st_filter, price_limit=False, t_plus=0,
        )
    calendar, custom = _calendar(config, profile)
    options = config.universe.options
    specs = _specs(options)
    static = options.get('universes', {})
    if (not isinstance(static, Mapping) or any(not isinstance(ids, tuple) or
            any(not isinstance(id_, str) or id_ not in specs for id_ in ids) or len(set(ids)) != len(ids)
            for ids in static.values())):
        raise ValueError('universes must map names to lists of distinct configured instruments')
    return MarketSlot(
        config.slot_id, calendar, DeclarativeBarFeed(config, calendar, specs, custom, source),
        DeclarativeUniverse(specs, static, membership), DeclarativeAuxFeatures(aux),
        DeclarativeMicrostructure(calendar, config.execution_profile, micro),
        slot_config_hash=config.slot_config_hash, adjustment_policy="none", market_db_profile=profile,
    )
