"""Market input contracts shared by computation, evaluation and portfolio execution."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import pandas as pd

from utils.session_calendar import Session, View
from utils.time_axis import Frequency


def _name(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f'{field} must be a nonempty string without surrounding whitespace')


@dataclass(frozen=True)
class InstrumentSpec:
    """Static instrument metadata with optional market-local lifecycle dates."""

    instrument_id: str
    asset_class: str
    currency: str
    lot_size: int = 1
    contract_multiplier: float = 1.0
    listing_date: date | None = None
    expiry_date: date | None = None

    def __post_init__(self) -> None:
        for field in ('instrument_id', 'asset_class', 'currency'):
            _name(getattr(self, field), field)
        if isinstance(self.lot_size, bool) or not isinstance(self.lot_size, Integral) or self.lot_size < 1:
            raise ValueError('lot_size must be a positive integer')
        multiplier = self.contract_multiplier
        if (isinstance(multiplier, bool) or not isinstance(multiplier, Real)
                or not isfinite(multiplier) or multiplier <= 0):
            raise ValueError('contract_multiplier must be finite and positive')
        for field in ('listing_date', 'expiry_date'):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, date) or isinstance(value, datetime)):
                raise ValueError(f'{field} must be a date or None')
        if self.listing_date is not None and self.expiry_date is not None and self.expiry_date < self.listing_date:
            raise ValueError('expiry_date must be on or after listing_date')


@dataclass(frozen=True)
class ExecutionProfile:
    """Entry is signal slot + entry offset, exit is entry slot + horizon + adjustment."""

    entry_price_field: Literal['open', 'close']
    exit_price_field: Literal['open', 'close']
    min_entry_offset_bars: int
    exit_adjustment: int
    return_unit: Literal['decimal', 'percent']
    gap_rule: Literal['endpoints_only', 'require_complete_bars']
    boundary_policy: Literal['same_session', 'cross_session']
    allow_short: bool = False
    commission_bps: float = 0.0
    sell_tax_bps: float = 0.0
    slippage_bps: float = 0.0
    benchmark_instrument: str | None = None

    def __post_init__(self) -> None:
        for field, allowed in (
            ('entry_price_field', ('open', 'close')),
            ('exit_price_field', ('open', 'close')),
            ('return_unit', ('decimal', 'percent')),
            ('gap_rule', ('endpoints_only', 'require_complete_bars')),
            ('boundary_policy', ('same_session', 'cross_session')),
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f'{field} must be one of {allowed}')
        for field, minimum in (('min_entry_offset_bars', 0), ('exit_adjustment', -1)):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{field} must be an integer >= {minimum}')
            object.__setattr__(self, field, int(value))
        if not isinstance(self.allow_short, bool):
            raise ValueError('allow_short must be a boolean')
        for field in ('commission_bps', 'sell_tax_bps', 'slippage_bps'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f'{field} must be finite and nonnegative')
            object.__setattr__(self, field, float(value) if value else 0.0)
        if self.benchmark_instrument is not None:
            _name(self.benchmark_instrument, 'benchmark_instrument')


@runtime_checkable
class CalendarSlot(Protocol):
    @property
    def calendar_hash(self) -> str: ...

    def sessions(self, start: str | pd.Timestamp | None = None,
                 end: str | pd.Timestamp | None = None) -> pd.DatetimeIndex: ...

    def session_at(self, ts_utc: datetime, view: View = 'ETH', *,
                   frequency: str | Frequency | None = None) -> Session | None: ...

    def bar_grid(self, frequency: str | Frequency, start: str | pd.Timestamp | None = None,
                 end: str | pd.Timestamp | None = None, *, view: View = 'ETH',
                 bucket_strategy: str | None = None) -> pd.DataFrame: ...


@runtime_checkable
class BarFeedSlot(Protocol):
    def read_signal_bars(self, instruments: Sequence[str], frequency: Frequency, *,
                         start_utc: datetime, end_utc: datetime, as_of_utc: datetime) -> pd.DataFrame:
        """Use data available as of the UTC anchor, including adjustment inputs."""
        ...

    def read_execution_bars(self, instruments: Sequence[str], frequency: Frequency, *,
                            start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        """Return actual execution prices and quantities on the requested UTC interval."""
        ...


@runtime_checkable
class UniverseSlot(Protocol):
    def instrument_spec(self, instrument_id: str) -> InstrumentSpec: ...

    def resolve_universe(self, universe_id: str, *, as_of_utc: datetime) -> Sequence[str]: ...

    def resolve_active_contract(self, instrument_id: str, *, as_of_utc: datetime) -> str:
        """Resolve the executable contract or return the instrument itself if it has no roll."""
        ...


@runtime_checkable
class AuxFeatureSlot(Protocol):
    def read_aux_features(self, grid: pd.DataFrame, *, features: Sequence[str]) -> pd.DataFrame:
        """Align (instrument_id, timestamp_utc, session_id) rows with available_at_utc <= timestamp_utc."""
        ...


@runtime_checkable
class MicrostructureSlot(Protocol):
    @property
    def execution_profile(self) -> ExecutionProfile: ...

    def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str) -> pd.Series:
        """Return a boolean mask indexed like the input grid for the requested instrument."""
        ...


_GRID_KEYS = ['instrument_id', 'timestamp_utc', 'session_id']


def _require_utc_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame:
        raise ValueError(f'Auxiliary feature data requires {column}')
    values = frame[column]
    if (not isinstance(values.dtype, pd.DatetimeTZDtype)
            or str(values.dtype.tz) not in ('UTC', 'Etc/UTC') or values.isna().any()):
        raise ValueError(f'{column} must contain non-null UTC timestamps')


def _grid_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('Auxiliary feature data must be a DataFrame with unique column names')
    _require_utc_column(frame, 'timestamp_utc')
    if 'session_id' not in frame or frame.session_id.isna().any():
        raise ValueError('Auxiliary feature data requires non-null session_id')
    instruments = frame.get('instrument_id')
    if instruments is None or any(
            not isinstance(value, str) or not value or value != value.strip() for value in instruments):
        raise ValueError('Auxiliary feature data requires nonempty instrument_id strings without surrounding whitespace')
    return pd.MultiIndex.from_frame(frame[_GRID_KEYS])


if TYPE_CHECKING:
    from utils.market_db import MarketDbProfile


@dataclass(frozen=True)
class MarketSlot:
    slot_id: str
    calendar: CalendarSlot
    bar_feed: BarFeedSlot
    universe: UniverseSlot
    aux_features: AuxFeatureSlot
    microstructure: MicrostructureSlot
    slot_config_hash: str | None = None
    adjustment_policy: str | None = None
    slot_data_fingerprint: str | Mapping[str, object] | None = None
    market_db_profile: MarketDbProfile | None = None

    def __post_init__(self) -> None:
        _name(self.slot_id, 'slot_id')
        for field, protocol in (
            ('calendar', CalendarSlot), ('bar_feed', BarFeedSlot), ('universe', UniverseSlot),
            ('aux_features', AuxFeatureSlot), ('microstructure', MicrostructureSlot),
        ):
            component = getattr(self, field)
            if not isinstance(component, protocol) or any(
                    not callable(getattr(component, method, None))
                    for method, member in vars(protocol).items()
                    if not method.startswith('_') and callable(member)):
                raise TypeError(f'{field} must implement {protocol.__name__}')
        if not isinstance(self.microstructure.execution_profile, ExecutionProfile):
            raise TypeError('microstructure.execution_profile must be an ExecutionProfile')

        from utils.market_db.identity import bind_market_db_fingerprint, slot_market_db_identity

        identity = slot_market_db_identity(self)
        if identity:
            object.__setattr__(self, 'slot_data_fingerprint',
                               bind_market_db_fingerprint(self.slot_data_fingerprint, identity))

    def read_aux_features(self, grid: pd.DataFrame, *, features: Sequence[str]) -> pd.DataFrame:
        """Enforce instrument, grid and PIT membership before exposing adapter output."""
        keys = _grid_keys(grid)
        if not keys.is_unique:
            raise ValueError('Auxiliary feature grid keys must be distinct')
        result = self.aux_features.read_aux_features(grid, features=features)
        result_keys = _grid_keys(result)
        _require_utc_column(result, 'available_at_utc')
        if not result_keys.isin(keys).all():
            raise ValueError('Auxiliary feature rows must belong to the requested grid')
        if (result.available_at_utc > result.timestamp_utc).any():
            raise ValueError('Auxiliary features must satisfy available_at_utc <= timestamp_utc')
        return result
