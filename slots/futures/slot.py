"""Read-only futures slot over concrete Bars and a materialized roll policy."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from datetime import datetime
import json
from typing import TypedDict

import duckdb
import pandas as pd

from backtest_and_analysis.evaluation.labels import ExecutionConvention
from backtest_and_analysis.time_series.contracts import BAR_COLUMNS, PRICE_COLUMNS, _utc
from utils.bars.builder import calendar_name, instrument_calendar
from utils.bars.catalog import BarCatalog, table_exists
from utils.bars.continuous import apply_roll_adjustments
from utils.bars.futures_roll import MissingSessionCloseError, product_metadata, session_close
from utils.bars.source import BarSource, MissingBarDataError, _instrument_ids
from utils.market_db import MarketDbProfile, MarketDbProfileError
from utils.market_db.errors import MARKET_DB_PROFILE_INVALID
from utils.market_db.validation import (
    market_db_calendar_bounds, require_market_db_profile, validate_market_db_calendar,
    validate_market_db_execution_rules, validate_market_db_metadata, validate_market_db_provider,
)
from utils.session_calendar import Session, View
from utils.slots.contracts import ExecutionProfile, InstrumentSpec, MarketSlot, PortfolioExecutionProfile
from utils.time_axis import BarSchedule, Frequency

from .policy import DaysBeforeExpiry, RollPolicy, expiry_date, resolve_roll_policy


_ADJUSTMENTS = {'none': 'none', 'ratio_back': 'ratio_back_adjust',
                'panama_canal': 'diff_back_adjust'}


class FuturesOptions(TypedDict, total=False):
    roll_policy: str | RollPolicy
    adjustment: str
    view: View
    exclude_delivery_month: bool
    commission_bps: float
    slippage_bps: float
    calendar_id: str
    timezone: str


class _FuturesData:
    def __init__(self, connection: duckdb.DuckDBPyConnection, *, product_root: str,
                 start: str | None, end: str | None, roll_policy: str | RollPolicy, adjustment: str,
                 view: View, currency: str, exclude_delivery_month: bool,
                 commission_bps: float, slippage_bps: float, profile: MarketDbProfile,
                 calendar_id: str | None, timezone: str | None) -> None:
        if adjustment not in _ADJUSTMENTS:
            raise ValueError(f'adjustment must be one of {tuple(_ADJUSTMENTS)}')
        if not isinstance(exclude_delivery_month, bool):
            raise ValueError('exclude_delivery_month must be a boolean')
        self.connection = connection
        self.root = product_root
        self.policy = resolve_roll_policy(roll_policy)
        self.expiry_lead = self.policy.days if isinstance(self.policy, DaysBeforeExpiry) else None
        self.adjustment = _ADJUSTMENTS[adjustment]
        self.reader = BarSource(connection)
        self.catalog = BarCatalog(connection)
        metadata = product_metadata(connection, product_root)
        self.profile = profile
        validate_market_db_metadata(profile, metadata, calendar_name=calendar_name)
        if view not in profile.session_views:
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                       f'Session view {view!r} is not allowed by {profile.db_id}')
        start, end = market_db_calendar_bounds(profile, start, end)
        self.metadata = metadata.set_index('instrument_id', drop=False).to_dict('index')
        if product_root in self.metadata:
            raise ValueError(f'Product root {product_root} must differ from its concrete contract IDs')
        self.calendar = instrument_calendar(metadata.iloc[0].to_dict(), start, end)
        validate_market_db_calendar(profile, calendar_id if calendar_id is not None else self.calendar.calendar_id,
                                    timezone if timezone is not None else self.calendar.timezone,
                                    expected_calendar=self.calendar.calendar_id)
        validate_market_db_calendar(profile, self.calendar.calendar_id, self.calendar.timezone,
                                    expected_calendar=self.calendar.calendar_id)
        self.calendar._view(view)
        self.view = view
        self.specs = {
            contract: InstrumentSpec(contract, 'futures', currency,
                                     contract_multiplier=row['point_value'],
                                     expiry_date=expiry_date(row['contract_expiry']).date())
            for contract, row in self.metadata.items()
        }
        self.exclude_delivery_month = exclude_delivery_month
        rules = ExecutionConvention.OPEN_TO_CLOSE.value
        self.execution_profile = ExecutionProfile(
            rules.entry_price_field, rules.exit_price_field, rules.min_entry_offset,
            rules.exit_adjustment, rules.unit, rules.gap_rule, 'cross_session', allow_short=True,
            commission_bps=commission_bps, slippage_bps=slippage_bps,
        )
        self.portfolio_execution_profile = PortfolioExecutionProfile(
            rules.entry_price_field, rules.entry_price_field, rules.min_entry_offset,
            0, rules.unit, rules.gap_rule, 'cross_session', allow_short=True,
            commission_bps=commission_bps, slippage_bps=slippage_bps,
        )
        days = self.calendar.sessions().strftime('%Y-%m-%d')
        self.sessions = tuple(self.calendar.session_on(day) for day in days)
        self.opens = tuple(session.segments[0][0] for session in self.sessions)
        self.closes = tuple(session.segments[-1][1] for session in self.sessions)

    def _plans(self, first: str, last: str) -> dict[str, dict]:
        if not table_exists(self.connection, 'futures_roll_schedule'):
            raise MissingBarDataError(f'Missing roll schedule for {self.root}; run build-roll-schedule')
        rows = self.connection.execute(
            'SELECT * FROM futures_roll_schedule WHERE product_root=? '
            'AND effective_session_id BETWEEN ? AND ? ORDER BY effective_session_id',
            [self.root, first, last]).fetchdf()
        return {row['effective_session_id']: row for row in rows.to_dict('records')}

    def _binding(self, session_id: str, opened_at: pd.Timestamp, plans: dict[str, dict]) -> dict:
        plan = plans.get(session_id)
        if plan is None:
            raise MissingBarDataError(f'Missing roll schedule for {self.root} Session={session_id}')
        if plan['rule'] != self.policy.name or (self.expiry_lead is not None and
                json.loads(plan['rule_params']).get('days_before_expiry') != self.expiry_lead):
            raise ValueError(f'Roll schedule must use {self.policy.name} for {self.root} Session={session_id}')
        decision = plan['decision_session_id']
        if not isinstance(decision, str) or decision >= session_id:
            raise ValueError(f'Invalid decision Session for {self.root} Session={session_id}')
        if pd.Timestamp(decision) < self.calendar.start:
            meta = next(iter(self.metadata.values()))
            market_db_calendar_bounds(self.profile, decision, decision)
            decision_calendar = instrument_calendar(meta, decision, decision)
        else:
            decision_calendar = self.calendar
        decision_session = decision_calendar.session_on(decision)
        if decision_session is None or decision_session.segments[-1][1] > opened_at:
            raise ValueError(f'Decision Session must close before effective open: {self.root} {session_id}')
        for contract in (plan['from_contract'], plan['to_contract']):
            if pd.notna(contract) and contract not in self.metadata:
                raise MissingBarDataError(f'Missing contract={contract} for {self.root} Session={session_id}')
        return plan

    def _read(self, instruments: Sequence[str], frequency: str | Frequency, *,
              start_utc: datetime, end_utc: datetime, as_of_utc: datetime | None = None) -> pd.DataFrame:
        ids = _instrument_ids(instruments)
        freq = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
        if not freq.is_valid or freq.kind == 'week':
            raise ValueError('Futures slot frequency must be intraday or 1d')
        _utc(start_utc, 'start_utc')
        _utc(end_utc, 'end_utc')
        if start_utc > end_utc:
            raise ValueError('start_utc must not follow end_utc')
        start, end = pd.Timestamp(start_utc), pd.Timestamp(end_utc)
        for instrument in ids:
            if instrument != self.root:
                self.instrument_spec(instrument)
        self.calendar.session_at(start, frequency=freq)
        if as_of_utc is not None:
            _utc(as_of_utc, 'as_of_utc')
            self.calendar.session_at(as_of_utc, frequency=freq)
            end = min(end, pd.Timestamp(as_of_utc))
        if end < start:
            return self._empty_bars()
        self.calendar.session_at(end, frequency=freq)
        daily = freq.kind == 'session'
        key_column = 'end_utc' if daily else 'start_utc'
        first = bisect_left(self.closes, start) if daily else bisect_right(self.closes, start)
        stop = bisect_right(self.closes if daily else self.opens, end)
        sessions = self.sessions[first:stop]
        if not sessions:
            return self._empty_bars()

        adjust = self.adjustment if as_of_utc is not None else 'none'
        bindings = {}
        if self.root in ids:
            anchor_stop = stop if adjust == 'none' else bisect_right(self.opens, as_of_utc)
            effective = self.sessions[first:anchor_stop]
            plans = self._plans(effective[0].session_date, effective[-1].session_date)
            bindings = {session.session_date: self._binding(session.session_date, opened_at, plans)
                        for session, opened_at in zip(effective, self.opens[first:anchor_stop])}

        entries = {}
        frames = []
        for session in sessions:
            day = session.session_date
            for instrument in ids:
                plan = bindings.get(day) if instrument == self.root else None
                contract = plan['to_contract'] if plan is not None else instrument
                if contract not in entries:
                    try:
                        entry = self.catalog.resolve(freq.canonical, contract, asset_class='futures')
                    except LookupError as exc:
                        raise MissingBarDataError(str(exc)) from exc
                    if entry.table_name != freq.table_name or entry.timestamp_semantics != 'UTC_bar_start':
                        raise MissingBarDataError(f'Missing physical frequency={freq.canonical} for {contract}')
                    entries[contract] = entry
                entry = entries[contract]
                schedule = BarSchedule(self.calendar, freq, bucket_strategy=entry.bucket_strategy)
                grid = schedule.for_session(session)
                if self.view == 'RTH':
                    regular = BarSchedule(self.calendar, freq, view='RTH',
                                          bucket_strategy=entry.bucket_strategy).for_session(session)
                    # ETH data can be projected only when the stored bucket has the same RTH boundaries.
                    grid = grid.merge(regular[['start_utc', 'end_utc']], on=['start_utc', 'end_utc'])
                grid = grid.loc[grid[key_column].between(start, end)]
                if as_of_utc is not None:
                    grid = grid.loc[grid.available_at_utc.le(as_of_utc)]
                if grid.empty:
                    continue
                raw = self.reader.read_bars([contract], freq.canonical,
                    start_utc=grid.start_utc.min(), end_utc=grid.start_utc.max(),
                    source_table=entry.table_name, calendar=self.calendar)
                raw.attrs = {}
                keys = grid[['start_utc', 'available_at_utc']].rename(
                    columns={'start_utc': 'timestamp_utc'}).assign(public_key=grid[key_column])
                frame = raw.merge(keys, on='timestamp_utc')
                if frame.empty:
                    raise MissingBarDataError(f'Missing frequency={freq.canonical} Bars for {contract} Session={day}')
                frame['timestamp_utc'] = frame.pop('public_key')
                frames.append(frame.assign(instrument_id=instrument, session_id=day,
                                           concrete_instrument_id=contract))
        if not frames:
            return self._empty_bars()
        result = pd.concat(frames, ignore_index=True)
        if adjust != 'none' and self.root in ids:
            mask = result.instrument_id.eq(self.root)
            if mask.any():
                continuous = result.loc[mask].copy()
                first_bar = continuous.session_id.min()
                events = [plan for day, plan in bindings.items() if day > first_bar]
                apply_roll_adjustments(self.connection, continuous, events, self.metadata, adjust)
                result.loc[mask, list(PRICE_COLUMNS)] = continuous[list(PRICE_COLUMNS)]
        return result.sort_values(['timestamp_utc', 'instrument_id']).reset_index(drop=True)

    @staticmethod
    def _empty_bars() -> pd.DataFrame:
        return pd.DataFrame(columns=[*BAR_COLUMNS, 'session_id', 'concrete_instrument_id',
                                    'available_at_utc']).astype({
            'timestamp_utc': 'datetime64[ns, UTC]', 'available_at_utc': 'datetime64[ns, UTC]',
        })

    def read_signal_bars(self, instruments: Sequence[str], frequency: Frequency, *,
                         start_utc: datetime, end_utc: datetime, as_of_utc: datetime) -> pd.DataFrame:
        _utc(as_of_utc, 'as_of_utc')
        return self._read(instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc)

    def read_execution_bars(self, instruments: Sequence[str], frequency: Frequency, *,
                            start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        return self._read(instruments, frequency, start_utc=start_utc, end_utc=end_utc)

    def instrument_spec(self, instrument_id: str) -> InstrumentSpec:
        if instrument_id not in self.specs:
            raise ValueError(f'Unknown concrete futures instrument {instrument_id!r} for {self.root}')
        return self.specs[instrument_id]

    def _session_date(self, as_of_utc: datetime) -> str:
        _utc(as_of_utc, 'as_of_utc')
        session = self.calendar.session_at(as_of_utc)
        return (session.session_date if session else
                pd.Timestamp(as_of_utc).tz_convert(self.calendar.timezone).strftime('%Y-%m-%d'))

    def resolve_universe(self, universe_id: str, *, as_of_utc: datetime) -> Sequence[str]:
        if universe_id != self.root:
            raise ValueError(f'Unknown futures universe {universe_id!r}; expected {self.root}')
        day = pd.Timestamp(self._session_date(as_of_utc)).date()
        return tuple(contract for contract, spec in self.specs.items() if spec.expiry_date >= day)

    def _resolve_active_contract(
        self, instrument_id: str, *, as_of_utc: datetime, session_date: str | None = None
    ) -> str:
        resolved_session = session_date if session_date is not None else self._session_date(as_of_utc)
        day = pd.Timestamp(resolved_session).date()
        contract = instrument_id
        if instrument_id == self.root:
            position = bisect_right(self.opens, as_of_utc) - 1
            if position < 0:
                raise MissingBarDataError(f'No effective roll Session for {self.root} at {as_of_utc}')
            effective = self.sessions[position].session_date
            contract = self._binding(effective, self.opens[position],
                                     self._plans(effective, effective))['to_contract']
        if self.instrument_spec(contract).expiry_date < day:
            raise MissingBarDataError(f'Expired futures contract {contract} at {as_of_utc}')
        return contract

    def resolve_active_contract(
        self, instrument_id: str, *, as_of_utc: datetime, session_date: str | None = None
    ) -> str:
        return self._resolve_active_contract(instrument_id, as_of_utc=as_of_utc, session_date=session_date)

    def _session_before_coverage(self) -> Session | None:
        metadata = next(iter(self.metadata.values()))
        first, _ = market_db_calendar_bounds(self.profile)
        end = self.calendar.start - pd.Timedelta(days=1)
        span = 7
        while end >= first:
            start = max(first, self.calendar.start - pd.Timedelta(days=span))
            history = instrument_calendar(metadata, start, end)
            sessions = history.sessions()
            if len(sessions):
                return history.session_on(sessions[-1])
            if start == first:
                break
            span *= 2
        return None

    def read_aux_features(self, grid: pd.DataFrame, *, features: Sequence[str]) -> pd.DataFrame:
        allowed = {'days_to_expiry', 'open_interest'}
        if (isinstance(features, str) or not features or len(set(features)) != len(features)
                or not set(features) <= allowed):
            raise ValueError(f'Futures features must be a nonempty subset of {sorted(allowed)}')
        data = grid[['instrument_id', 'timestamp_utc', 'session_id']].copy()
        interest = {}
        available, values = [], {name: [] for name in features}
        previous = None
        previous_resolved = False
        for row in data.itertuples():
            ts = row.timestamp_utc
            spec = self.instrument_spec(row.instrument_id)
            session_id = self._session_date(ts)
            if str(row.session_id) != session_id:
                raise ValueError(f'Futures feature Session mismatch for {row.instrument_id} at {ts}')
            observed_at = ts
            if 'days_to_expiry' in values:
                values['days_to_expiry'].append((spec.expiry_date - pd.Timestamp(session_id).date()).days)
            if 'open_interest' in values:
                # Carry the last scheduled close observation only after that close.
                pos = bisect_right(self.closes, ts) - 1
                if pos < 0 and not previous_resolved:
                    previous = self._session_before_coverage()
                    previous_resolved = True
                session = self.sessions[pos] if pos >= 0 else previous
                value = float('nan')
                if session is not None and (closed_at := session.segments[-1][1]) <= ts:
                    key = (row.instrument_id, session.session_date)
                    if key not in interest:
                        try:
                            interest[key] = session_close(self.connection, self.metadata[row.instrument_id],
                                                          key[1], 'open_interest')['value']
                        except MissingSessionCloseError:
                            interest[key] = float('nan')
                    value = interest[key]
                    if 'days_to_expiry' not in values:
                        observed_at = closed_at
                values['open_interest'].append(value)
            available.append(observed_at)
        return data.assign(**values, available_at_utc=pd.to_datetime(available, utc=True))

    def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
        def eligible(ts: pd.Timestamp, session_id: str, price_field: str) -> bool:
            _utc(ts, 'timestamp_utc')
            ts_norm = pd.Timestamp(ts)
            session = self.calendar.session_on(str(session_id), self.view)
            if session is None or not session.segments:
                return False
            if price_field == 'close':
                in_session = any(left <= ts_norm < right or ts_norm == right
                                 for left, right in session.segments)
            elif price_field == 'open':
                in_session = any(left <= ts_norm < right for left, right in session.segments)
            else:
                in_session = (any(left <= ts_norm < right for left, right in session.segments)
                              or ts_norm == session.segments[-1][1])
            if not in_session:
                return False
            try:
                contract = self._resolve_active_contract(
                    instrument_id, as_of_utc=ts_norm, session_date=session.session_date
                )
            except MissingBarDataError:
                return False
            day = pd.Timestamp(session.session_date).date()
            expiry = self.instrument_spec(contract).expiry_date
            delivery_month = self.metadata[contract]['contract_month']
            return day < expiry and (not self.exclude_delivery_month or
                                     session.session_date[:7] != delivery_month)

        raw_fields = grid.get('execution_price_field', pd.Series('', index=grid.index))
        price_fields = ['' if pd.isna(v) else str(v) for v in raw_fields]
        return pd.Series(
            [eligible(ts, day, field)
             for ts, day, field in zip(grid.timestamp_utc, grid.session_id, price_fields)],
            index=grid.index, dtype=bool, name='tradable')


def build_futures_slot(connection: duckdb.DuckDBPyConnection, *, provider: str, slot_id: str,
                       product_root: str, start: str | None, end: str | None, currency: str,
                       roll_policy: str | RollPolicy = 'max_oi', adjustment: str = 'none',
                       view: View = 'ETH', exclude_delivery_month: bool = False,
                       commission_bps: float = 0., slippage_bps: float = 0.,
                       calendar_id: str | None = None, timezone: str | None = None) -> MarketSlot:
    """Bind one futures product and its physical data to the shared calendar and policy."""
    profile = require_market_db_profile(connection)
    validate_market_db_provider(profile, provider, domain=provider)
    validate_market_db_execution_rules(profile, provider, st_filter=False, price_limit=False, t_plus=0)
    data = _FuturesData(connection, product_root=product_root, start=start, end=end,
                        roll_policy=roll_policy, adjustment=adjustment, view=view,
                        currency=currency, exclude_delivery_month=exclude_delivery_month,
                        commission_bps=commission_bps, slippage_bps=slippage_bps, profile=profile,
                        calendar_id=calendar_id, timezone=timezone)
    return MarketSlot(slot_id, data.calendar, data, data, data, data,
                      adjustment_policy=data.adjustment, market_db_profile=profile)
