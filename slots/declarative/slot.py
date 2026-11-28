"""MarketSlot components assembled from one versioned SlotConfig."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
import json
from pathlib import Path

import pandas as pd

from backtest_and_analysis.time_series.contracts import BAR_COLUMNS, ContractValidationError, _utc
from data_pipeline.loaders.market_bar_importer import (
    CONVERSION_MARKER, ExternalBarSourceSpec, _external_timestamps, adapt_external_bars,
    instrument_calendar, validate_instrument_metadata,
)
from utils.bars.source import _bounds, _instrument_ids
from utils.market_db import MarketDbProfile, MarketDbProfileError, open_market_db
from utils.market_db.errors import MARKET_DB_PROFILE_INVALID
from utils.market_db.validation import (
    market_db_calendar_bounds, validate_market_db_calendar, validate_market_db_execution_rules,
    validate_market_db_frequencies, validate_market_db_provider,
)
from utils.pit import normalize_price_money
from utils.session_calendar import SessionCalendar, View
from utils.slots import ExecutionProfile, InstrumentSpec, MarketSlot, SlotConfig
from utils.time_axis import BarSchedule, Frequency

from .source import FrameSource, ReadRequest


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
    if 'rth_reference_calendar' in options:
        reference = options['rth_reference_calendar']
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError('rth_reference_calendar must be a calendar ID')
    else:
        reference = None
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
        data = self.membership.read_mapped(ReadRequest(columns=('universe_id', 'instrument_id', _AVAILABLE)))
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
        if grid.empty:
            request = ReadRequest(columns=('instrument_id', _AVAILABLE), optional_columns=tuple(columns))
        else:
            instruments = tuple(dict.fromkeys(grid.instrument_id.astype(str)))
            max_ts = pd.Timestamp(grid.timestamp_utc.max()) if 'timestamp_utc' in grid else pd.NaT
            as_of = (
                max_ts.tz_convert('UTC').to_pydatetime()
                if not pd.isna(max_ts) and max_ts.tzinfo is not None else None
            )
            request = ReadRequest(
                instruments=instruments,
                as_of_utc=as_of,
                columns=('instrument_id', _AVAILABLE),
                optional_columns=tuple(columns),
            )
        data = self.source.read_mapped(request)
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
    view: View = 'ETH'

    def __post_init__(self) -> None:
        if not isinstance(self.view, str):
            raise ValueError('view must be ETH or RTH')
        self.calendar._view(self.view)

    def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
        def in_trading_segment(ts: pd.Timestamp, session_id: str, price_field: str) -> bool:
            _utc(ts, 'timestamp_utc')
            session = self.calendar.session_on(session_id, self.view)
            if session is None or not session.segments:
                return False
            # Grid callers without an execution field also use the session's daily close.
            if price_field == '' and ts == session.segments[-1][1]:
                return True
            return any(left <= ts < right or (price_field == 'close' and ts == right)
                       for left, right in session.segments)

        raw_fields = grid.get('execution_price_field', pd.Series('', index=grid.index))
        price_fields = ['' if pd.isna(v) else str(v) for v in raw_fields]
        result = pd.Series([in_trading_segment(ts, day, field)
                            for ts, day, field in zip(grid.timestamp_utc, grid.session_id, price_fields)],
                           index=grid.index, name='tradable', dtype=bool)
        if self.source is not None and not grid.empty:
            max_ts = pd.Timestamp(grid.timestamp_utc.max()) if 'timestamp_utc' in grid else pd.NaT
            as_of = (
                max_ts.tz_convert('UTC').to_pydatetime()
                if not pd.isna(max_ts) and max_ts.tzinfo is not None else None
            )
            push_inst = (instrument_id,) if isinstance(instrument_id, str) and instrument_id.strip() else None
            data = self.source.read_mapped(ReadRequest(
                instruments=push_inst,
                as_of_utc=as_of,
                columns=('instrument_id', _AVAILABLE, 'tradable'),
            ))
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
        view = options['view'] if 'view' in options else 'ETH'
        if not isinstance(view, str):
            raise ValueError('view must be ETH or RTH')
        calendar._view(view)
        if source.market_db_profile is not None and view not in source.market_db_profile.session_views:
            raise MarketDbProfileError(
                MARKET_DB_PROFILE_INVALID,
                f'Session view {view!r} is not allowed by {source.market_db_profile.db_id}',
            )
        self.view: View = view
        if 'source_view' in options:
            source_view = options['source_view']
            if not isinstance(source_view, str) or source_view not in ('ETH', 'RTH'):
                raise ValueError('source_view must be ETH or RTH')
            calendar._view(source_view)
            if source.market_db_profile is not None and source_view not in source.market_db_profile.session_views:
                raise MarketDbProfileError(
                    MARKET_DB_PROFILE_INVALID,
                    f'Session view {source_view!r} is not allowed by {source.market_db_profile.db_id}',
                )
            if view == 'ETH' and source_view == 'RTH':
                raise ValueError("view='ETH' cannot read a source declared with source_view='RTH'")
        else:
            source_view = None
        self.source_view: View | None = source_view
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
        self.schedule = BarSchedule(calendar, self.frequency, view=self.view, bucket_strategy=self.bucket_strategy)
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

    def _load_data(self, *, instruments: Sequence[str] | None = None,
                   start_utc: datetime | None = None, end_utc: datetime | None = None,
                   as_of_utc: datetime | None = None,
                   include_paused: bool = False) -> pd.DataFrame:
        required_columns = tuple(dict.fromkeys(self.spec.column_mapping.values()))
        optional_candidates = (
            ('money', 'factor', 'open_interest', 'paused')
            if include_paused
            else ('money', 'factor', 'open_interest')
        )
        optional_columns = tuple(
            field for field in optional_candidates
            if field not in required_columns and field not in self.spec.column_mapping
        )
        request = ReadRequest(
            instruments=instruments,
            start_utc=start_utc,
            end_utc=end_utc,
            as_of_utc=as_of_utc,
            columns=required_columns,
            optional_columns=optional_columns,
        )
        return self.source.read(request)

    def _rth_eth_overlap_bounds(
        self, start: pd.Timestamp, end: pd.Timestamp, session_days: pd.DatetimeIndex,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Expand an RTH intraday pushdown window to cover overlapping ETH bar anchors."""
        rth_days = [
            day for day in session_days
            if (session := self.calendar.session_on(day, 'RTH')) is not None
            and any(b >= start and a <= end for a, b in session.segments)
        ]
        if not rth_days:
            return start, end

        target_start: pd.Timestamp | None = None
        target_end: pd.Timestamp | None = None
        for day in rth_days:
            slots = self.schedule.for_session(day)
            matched = slots.loc[slots.start_utc.between(start, end)]
            if not matched.empty:
                target_start = matched.start_utc.min()
                break
        for day in reversed(rth_days):
            slots = self.schedule.for_session(day)
            matched = slots.loc[slots.start_utc.between(start, end)]
            if not matched.empty:
                target_end = matched.end_utc.max()
                break
        if target_start is None or target_end is None:
            target_start = start
            target_end = end + pd.Timedelta(nanoseconds=1)

        eth_schedule = BarSchedule(
            self.calendar, self.frequency, view='ETH', bucket_strategy=self.schedule.bucket_strategy,
        )
        eth_days = [
            day for day in session_days
            if (session := self.calendar.session_on(day, 'ETH')) is not None
            and any(b > target_start and a < target_end for a, b in session.segments)
        ]
        if not eth_days:
            return start, end
        is_bar_end = self.spec.timestamp_anchor == 'bar_end'
        push_start, push_end = start, end
        for day in dict.fromkeys((eth_days[0], eth_days[-1])):
            slots = eth_schedule.for_session(day)
            overlapping = slots.loc[(slots.end_utc > target_start) & (slots.start_utc < target_end)]
            if overlapping.empty:
                continue
            anchors = overlapping.end_utc if is_bar_end else overlapping.start_utc
            push_start = min(push_start, anchors.min())
            push_end = max(push_end, anchors.max())
        return push_start, push_end

    def _pushdown_window(self, start: pd.Timestamp, end: pd.Timestamp, *,
                         grid: pd.DataFrame | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.frequency.kind != 'intraday':
            bar_grid = self.calendar.bar_grid(self.frequency, view=self.view) if grid is None else grid
            matched = bar_grid.loc[bar_grid.timestamp_utc.between(start, end)]
            if matched.empty:
                return start, end
            push_start = self.schedule.for_session(matched.session_id.iloc[0]).start_utc.min()
            push_end = self.schedule.for_session(matched.session_id.iloc[-1]).end_utc.max()
            return push_start, push_end

        push_start, push_end = start, end
        need_rth_overlap = self.view == 'RTH' and self.source_view != 'RTH'
        if not need_rth_overlap and self.spec.volume_mode == 'interval' and self.spec.timestamp_anchor != 'bar_end':
            return push_start, push_end

        tz = self.calendar.timezone
        d0 = max(self.calendar.start, (start.tz_convert(tz) - pd.Timedelta(days=15)).tz_localize(None).normalize())
        d1 = min(self.calendar.end, (end.tz_convert(tz) + pd.Timedelta(days=15)).tz_localize(None).normalize())
        if d0 <= d1:
            session_days = self.calendar.sessions(d0, d1)
            if need_rth_overlap:
                push_start, push_end = self._rth_eth_overlap_bounds(start, end, session_days)
            if self.spec.volume_mode != 'interval':
                # Cumulative ETH sources read under view='RTH' must still begin at the ETH session open.
                vol_view: View = self.source_view or 'ETH'
                for day in session_days:
                    session = self.calendar.session_on(day, vol_view)
                    if session is not None and session.segments and session.segments[-1][1] >= start and session.segments[0][0] <= end:
                        push_start = min(push_start, session.segments[0][0])
                        break
            if self.spec.timestamp_anchor == 'bar_end':
                for day in reversed(session_days):
                    session = self.calendar.session_on(day, self.schedule.view)
                    if session is not None and session.segments and session.segments[0][0] <= end and session.segments[-1][1] >= start:
                        push_end = max(push_end, session.segments[-1][1])
                        break
        return push_start, push_end

    def _active_external_timestamps(
        self, data: pd.DataFrame, *, as_of_utc: datetime | None, require_all_columns: bool = True,
    ) -> pd.DatetimeIndex | None:
        ts_col = next((src for src, dst in self.spec.column_mapping.items() if dst == 'timestamp'), None)
        if (data.empty or data.attrs.get(CONVERSION_MARKER) or set(BAR_COLUMNS).issubset(data.columns)
                or not data.columns.is_unique or ts_col is None or ts_col not in data
                or (require_all_columns and set(self.spec.column_mapping) - set(data))):
            return None
        timestamps = _external_timestamps(data[ts_col], self.spec.source_timezone)
        if as_of_utc is not None:
            keep = timestamps <= as_of_utc if self.spec.timestamp_anchor == 'bar_end' else timestamps < as_of_utc
            timestamps = timestamps[keep]
        return timestamps.unique()

    def _classify_rth_projection_timestamps(
        self, unique_ts: pd.DatetimeIndex, rth_schedule: BarSchedule, eth_schedule: BarSchedule,
    ) -> tuple[bool, bool, bool]:
        """Classify intraday timestamps under view='RTH' as (has_rth_only, has_eth_only, has_ambiguous)."""
        rth_slots_by_day: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
        eth_slots_by_day: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
        rth_segments_by_day: dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]] = {}
        is_bar_end = self.spec.timestamp_anchor == 'bar_end'

        def _slots(
            cache: dict[str, dict[pd.Timestamp, pd.Timestamp]], sched: BarSchedule, day: str,
        ) -> dict[pd.Timestamp, pd.Timestamp]:
            if day not in cache:
                frame = sched.for_session(day)
                keys = frame.end_utc if is_bar_end else frame.start_utc
                vals = frame.start_utc if is_bar_end else frame.end_utc
                cache[day] = dict(zip(keys, vals))
            return cache[day]

        has_rth_only = False
        has_eth_only = False
        has_ambiguous = False
        for ts in unique_ts:
            probe = ts - pd.Timedelta(nanoseconds=1) if is_bar_end else ts
            rth_session = self.calendar.session_at(probe, 'RTH')
            eth_session = self.calendar.session_at(probe, 'ETH')
            rth_slots = _slots(rth_slots_by_day, rth_schedule, rth_session.session_date) if rth_session is not None else {}
            eth_slots = _slots(eth_slots_by_day, eth_schedule, eth_session.session_date) if eth_session is not None else {}
            in_rth = ts in rth_slots
            in_eth = ts in eth_slots

            if not in_rth and not in_eth:
                if is_bar_end:
                    if eth_session is None and rth_session is None:
                        raise ContractValidationError('bar_end must belong to an open Session')
                    raise ContractValidationError('bar_end is off the Bar schedule grid')
                if eth_session is None and rth_session is None:
                    raise ValueError('timestamp_utc is outside the Bar schedule')
                raise ValueError('timestamp_utc is off the Bar schedule grid')

            if in_rth and in_eth:
                rth_pair = (rth_slots[ts], ts) if is_bar_end else (ts, rth_slots[ts])
                eth_pair = (eth_slots[ts], ts) if is_bar_end else (ts, eth_slots[ts])
                if rth_pair != eth_pair:
                    has_ambiguous = True
                continue

            if in_rth:
                has_rth_only = True
                continue

            day = eth_session.session_date
            bar_start, bar_end = (eth_slots[ts], ts) if is_bar_end else (ts, eth_slots[ts])
            if day not in rth_segments_by_day:
                rth_day = self.calendar.session_on(day, 'RTH')
                rth_segments_by_day[day] = () if rth_day is None else rth_day.segments
            if any(max(bar_start, left) < min(bar_end, right) for left, right in rth_segments_by_day[day]):
                if is_bar_end:
                    raise ContractValidationError('bar_end overlaps RTH session with misaligned boundaries')
                raise ValueError('Bar overlaps RTH session with misaligned boundaries')
            has_eth_only = True

        return has_rth_only, has_eth_only, has_ambiguous

    def _resolve_intraday_rth_source_view(
        self,
        data: pd.DataFrame,
        rth_schedule: BarSchedule,
        eth_schedule: BarSchedule,
        *,
        as_of_utc: datetime | None,
        instruments: Sequence[str] | None,
        allow_source_probe: bool,
    ) -> View | None:
        unique_ts = self._active_external_timestamps(data, as_of_utc=as_of_utc)
        if unique_ts is None or unique_ts.empty:
            return None
        if self.source_view is not None:
            return self.source_view

        has_rth_only, has_eth_only, has_ambiguous = self._classify_rth_projection_timestamps(
            unique_ts, rth_schedule, eth_schedule,
        )
        if has_rth_only and has_eth_only:
            raise ContractValidationError('Source mixes incompatible RTH and ETH bar schedules')
        if has_eth_only:
            return 'ETH'
        if has_rth_only or not has_ambiguous:
            return 'RTH'

        # Ambiguous anchors (e.g., 60m bar_end at RTH close) cannot be classified from a pushed-down slice.
        if allow_source_probe:
            inst_col = next((src for src, dst in self.spec.column_mapping.items() if dst == 'instrument_id'), None)
            probe_cols = ('instrument_id', 'timestamp') if inst_col is not None else ('timestamp',)
            probe_data = self.source.read(ReadRequest(
                instruments=instruments if inst_col is not None else None,
                as_of_utc=as_of_utc,
                columns=probe_cols,
            ))
            probe_ts = self._active_external_timestamps(probe_data, as_of_utc=as_of_utc, require_all_columns=False)
            if probe_ts is not None and not probe_ts.empty:
                p_rth_only, p_eth_only, _ = self._classify_rth_projection_timestamps(
                    probe_ts, rth_schedule, eth_schedule,
                )
                if p_rth_only and p_eth_only:
                    raise ContractValidationError('Source mixes incompatible RTH and ETH bar schedules')
                if p_eth_only:
                    return 'ETH'
                if p_rth_only:
                    return 'RTH'

        raise ContractValidationError(
            'Ambiguous bar timestamps between ETH and RTH schedules; specify bar_feed.options.source_view'
        )

    def _project_eth_bars_to_rth(
        self, bars: pd.DataFrame, rth_schedule: BarSchedule, eth_schedule: BarSchedule,
    ) -> pd.DataFrame:
        rth_pairs_by_day: dict[str, set[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        rth_segments_by_day: dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]] = {}
        is_bar_end = self.spec.timestamp_anchor == 'bar_end'
        keep_starts = set()

        for ts in bars.timestamp_utc.unique():
            eth_end = eth_schedule.end_at(ts)
            eth_session = self.calendar.session_at(ts, 'ETH')
            day = eth_session.session_date
            if day not in rth_segments_by_day:
                rth_day = self.calendar.session_on(day, 'RTH')
                rth_segments_by_day[day] = () if rth_day is None else rth_day.segments
            overlaps_rth = any(max(ts, left) < min(eth_end, right) for left, right in rth_segments_by_day[day])
            if not overlaps_rth:
                continue
            if day not in rth_pairs_by_day:
                slots = rth_schedule.for_session(day)
                rth_pairs_by_day[day] = set(zip(slots.start_utc, slots.end_utc))
            if (ts, eth_end) not in rth_pairs_by_day[day]:
                if is_bar_end:
                    raise ContractValidationError('bar_end overlaps RTH session with misaligned boundaries')
                raise ValueError('Bar overlaps RTH session with misaligned boundaries')
            keep_starts.add(ts)

        attrs = dict(bars.attrs)
        projected = bars.loc[bars.timestamp_utc.isin(keep_starts)].reset_index(drop=True)
        projected.attrs.update(attrs)
        return projected

    def _adapt_bars(self, data: pd.DataFrame, *, as_of_utc: datetime | None = None,
                    bucket_strategy: str | None = None,
                    instruments: Sequence[str] | None = None,
                    allow_source_probe: bool = False) -> pd.DataFrame:
        schedule = self.schedule if bucket_strategy is None else BarSchedule(
            self.calendar, self.frequency, view=self.view, bucket_strategy=bucket_strategy)
        if self.view == 'RTH' and self.frequency.kind == 'intraday' and not data.empty:
            eth_schedule = BarSchedule(
                self.calendar, self.frequency, view='ETH', bucket_strategy=schedule.bucket_strategy)
            target_source_view = self._resolve_intraday_rth_source_view(
                data, schedule, eth_schedule,
                as_of_utc=as_of_utc, instruments=instruments, allow_source_probe=allow_source_probe,
            )
            if target_source_view == 'ETH':
                bars = adapt_external_bars(
                    data, self.spec, metadata=self.metadata, calendar=self.calendar,
                    frequency=self.frequency.canonical, source=self.source_name,
                    data_version=self.data_version, instrument_id=self.instrument_id,
                    bucket_strategy=eth_schedule.bucket_strategy, schedule=eth_schedule, through_utc=as_of_utc,
                )
                return self._project_eth_bars_to_rth(bars, schedule, eth_schedule)
        return adapt_external_bars(
            data, self.spec, metadata=self.metadata, calendar=self.calendar, frequency=self.frequency.canonical,
            source=self.source_name, data_version=self.data_version, instrument_id=self.instrument_id,
            bucket_strategy=schedule.bucket_strategy, schedule=schedule, through_utc=as_of_utc,
        )

    def load(self, *, instruments: Sequence[str] | None = None, as_of_utc: datetime | None = None,
             bucket_strategy: str | None = None) -> pd.DataFrame:
        data = self._load_data(instruments=instruments, as_of_utc=as_of_utc)
        return self._adapt_bars(data, as_of_utc=as_of_utc, bucket_strategy=bucket_strategy)

    def _read(self, instruments: Sequence[str], frequency: Frequency, *, start_utc: datetime,
              end_utc: datetime, as_of_utc: datetime | None = None,
              normalize_money: bool = False) -> pd.DataFrame:
        if frequency.canonical != self.frequency.canonical:
            raise ValueError(f'Declarative feed serves {self.frequency.canonical}, got {frequency.raw!r}')
        start, end = _bounds(start_utc, end_utc)
        if start is None or end is None:
            raise ValueError('Declarative bar reads require start_utc and end_utc')
        ids = _instrument_ids(instruments)
        grid = self.calendar.bar_grid(self.frequency, view=self.view) if self.frequency.kind != 'intraday' else None
        push_start, push_end = self._pushdown_window(start, end, grid=grid)
        data = self._load_data(
            instruments=ids, start_utc=push_start, end_utc=push_end, as_of_utc=as_of_utc,
            include_paused=normalize_money,
        )
        bars = self._adapt_bars(data, as_of_utc=as_of_utc, instruments=ids, allow_source_probe=True)
        # Only the signal path needs completion, but both views validate the grid.
        if self.frequency.kind == 'intraday':
            bars = bars.loc[bars.timestamp_utc.between(start, end)].copy()
        ends = {ts: self.schedule.end_at(ts) for ts in bars.timestamp_utc.unique()}
        end_times = pd.to_datetime(bars.timestamp_utc.map(ends), utc=True)
        days = bars.timestamp_utc if self.frequency.kind == 'intraday' else end_times - pd.Timedelta(1, 'ns')
        owners = [self.calendar.session_at(ts, self.view).session_date for ts in days]
        bars = bars.assign(session_id=pd.Series(owners, index=bars.index, dtype=object))
        if grid is not None:
            keys = grid.set_index('session_id').timestamp_utc
            bars = bars.assign(timestamp_utc=bars.session_id.map(keys))
        if normalize_money and 'money' in bars and 'paused' in bars:
            money_frequency = (
                'daily' if self.frequency.kind == 'session'
                else '1m' if self.frequency.canonical == '1min'
                else '5m' if self.frequency.canonical == '5min'
                else None
            )
            if money_frequency is not None:
                bars = normalize_price_money(bars, source='local', frequency=money_frequency)
        mask = bars.timestamp_utc.between(start, end)
        extras = [field for field in ('money', 'factor', 'open_interest') if field in bars]
        return bars.loc[mask, [*BAR_COLUMNS, 'session_id', *extras]].reset_index(drop=True)

    def read_signal_bars(self, instruments: Sequence[str], frequency: Frequency, *, start_utc: datetime,
                         end_utc: datetime, as_of_utc: datetime) -> pd.DataFrame:
        _utc(as_of_utc, 'as_of_utc')
        return self._read(
            instruments, frequency, start_utc=start_utc, end_utc=end_utc,
            as_of_utc=as_of_utc, normalize_money=True,
        )

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
    feed = DeclarativeBarFeed(config, calendar, specs, custom, source)
    return MarketSlot(
        config.slot_id, calendar, feed,
        DeclarativeUniverse(specs, static, membership), DeclarativeAuxFeatures(aux),
        DeclarativeMicrostructure(calendar, config.execution_profile, micro, view=feed.view),
        slot_config_hash=config.slot_config_hash, adjustment_policy="none", market_db_profile=profile,
    )
