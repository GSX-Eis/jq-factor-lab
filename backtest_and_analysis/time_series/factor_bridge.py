"""Session-grid operations shared by minute FactorBase implementations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from utils.market_db.identity import market_db_identity, slot_market_db_identity
from utils.slots import MarketSlot

from .contracts import (
    ContractValidationError, _mask_unavailable_factor_values, _validate_aligned_factor_frame,
    validate_bar_frame, validate_factor_frame,
)
from utils.time_axis import BarSchedule


def continuous_factor_input(source, configuration: Mapping) -> pd.DataFrame:
    """Read a configured futures factor input with no implicit price adjustment."""
    from utils.continuous_bars import ADJUSTMENTS

    if not isinstance(configuration, Mapping):
        raise ContractValidationError('continuous_futures must be a configuration object')
    if configuration.get('adjust') not in ADJUSTMENTS:
        raise ContractValidationError(f'continuous_futures requires explicit adjust: {ADJUSTMENTS}')
    required = ('product_root', 'frequency', 'start', 'end')
    for name in required:
        if not isinstance(configuration.get(name), str) or not configuration[name].strip():
            raise ContractValidationError(f'continuous_futures requires {name}')
    return source.read(*(configuration[name] for name in required),
                       view=configuration.get('view', 'ETH'), adjust=configuration['adjust'])


def bar_factor_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Convert a UTC wide panel without dropping NaNs, or validate a long panel."""
    if not isinstance(values, pd.DataFrame):
        raise ContractValidationError('factor output must be a DataFrame')
    keys = ['timestamp_utc', 'instrument_id', 'factor_value']
    if set(keys).issubset(values.columns):
        return _validate_aligned_factor_frame(values).loc[:, keys].reset_index(drop=True)
    if not isinstance(values.index, pd.DatetimeIndex) or not values.columns.is_unique:
        raise ContractValidationError('wide factor output needs a UTC DatetimeIndex and unique instruments')
    long = values.rename_axis(index='timestamp_utc', columns='instrument_id').reset_index().melt(
        id_vars='timestamp_utc', var_name='instrument_id', value_name='factor_value')
    return validate_factor_frame(long).reset_index(drop=True)


def incomplete_bar_keys(grid: pd.DataFrame) -> tuple[tuple[int, str], ...]:
    """Capture the unavailable keys from the grid that produced a panel."""
    if 'is_complete' not in grid:
        return ()
    incomplete = ~grid.is_complete.eq(True).fillna(False)
    keys = grid.loc[incomplete, ['timestamp_utc', 'instrument_id']]
    return tuple((timestamp.value, instrument) for timestamp, instrument in keys.itertuples(index=False, name=None))


def mask_incomplete_bars(
    values: pd.DataFrame, keys: Sequence[tuple[int, str]],
) -> pd.DataFrame:
    if not keys:
        return values
    times = values.timestamp_utc.dt.as_unit('ns').astype('int64')
    unavailable = pd.MultiIndex.from_arrays([times, values.instrument_id]).isin(
        [(timestamp, instrument) for timestamp, instrument in keys],
    )
    return values.assign(factor_value=_mask_unavailable_factor_values(values.factor_value, unavailable))


def session_grid(bars: pd.DataFrame, calendars: dict, frequency: str, *,
                 view: str = 'ETH', start_date: str, end_date: str,
                 bucket_strategy: str | None = None) -> pd.DataFrame:
    """Expose every scheduled slot, including entire missing sessions, as NaN."""
    bars = validate_bar_frame(bars)
    grids = []
    for instrument, calendar in calendars.items():
        schedule = BarSchedule(calendar, frequency, view=view, bucket_strategy=bucket_strategy)
        days = calendar.sessions()
        for day in days[(days >= pd.Timestamp(start_date)) & (days <= pd.Timestamp(end_date))]:
            session_id = day.strftime('%Y-%m-%d')
            times = pd.DatetimeIndex(schedule.for_session(session_id).start_utc)
            grids.append(pd.DataFrame(dict(timestamp_utc=times, instrument_id=instrument,
                                           session_id=session_id)))
    if not grids:
        return bars.iloc[:0].assign(session_id=pd.Series(dtype=str), observed=pd.Series(dtype=bool))
    grid = pd.concat(grids, ignore_index=True)
    result = grid.merge(bars.drop(columns='session_id', errors='ignore'),
                        on=['timestamp_utc', 'instrument_id'], how='left', validate='one_to_one',
                        indicator=True)
    if not bars[['timestamp_utc', 'instrument_id']].merge(
            grid, on=['timestamp_utc', 'instrument_id'], how='left').session_id.notna().all():
        raise ContractValidationError('bars contain timestamps outside the requested Session grid')
    result['observed'] = result.pop('_merge').eq('both')
    result['frequency'] = frequency
    result.attrs = dict(bars.attrs)
    return result.sort_values(['instrument_id', 'timestamp_utc']).reset_index(drop=True)


def session_rolling(data: pd.DataFrame, column: str, window: int, *,
                    min_periods: int | None = None, within_session: bool = True,
                    warmup_sessions: int = 0, warmup_bars: int = 0,
                    aggregation: str = 'mean') -> pd.Series:
    """Roll an expected grid; lunch/maintenance closures are not missing slots.

    Pass the grid returned by FactorBase, never compressed observed-only Bars.
    Warmup counts scheduled slots and sessions per instrument, including gaps.
    """
    for name, value, minimum in [('window', window, 1), ('warmup_sessions', warmup_sessions, 0),
                                  ('warmup_bars', warmup_bars, 0)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ContractValidationError(f'{name} must be an integer >= {minimum}')
    if not {'timestamp_utc', 'instrument_id', 'session_id', 'observed', column}.issubset(data):
        raise ContractValidationError('session_rolling requires an expected Session grid')
    if aggregation not in ('mean', 'sum', 'std', 'min', 'max', 'median'):
        raise ContractValidationError('unsupported rolling aggregation')
    ordered = data.sort_values(['instrument_id', 'timestamp_utc'])
    groups = ['instrument_id', 'session_id'] if within_session else ['instrument_id']
    result = ordered.groupby(groups, sort=False)[column].transform(
        lambda series: series.rolling(window, min_periods=min_periods).agg(aggregation))
    slots = ordered.groupby('instrument_id').cumcount()
    sessions = ordered.groupby('instrument_id')['session_id'].transform(
        lambda series: pd.factorize(series, sort=False)[0])
    result.loc[(slots < warmup_bars) | (sessions < warmup_sessions) | ~ordered.observed] = np.nan
    return result.reindex(data.index)


def session_daily_frame(values: pd.DataFrame, grid: pd.DataFrame, *,
                        aggregation: str = 'last') -> pd.DataFrame:
    """Aggregate by calendar session date; 'last' preserves a missing close slot."""
    if aggregation not in ('last', 'mean', 'sum', 'min', 'max', 'median'):
        raise ContractValidationError('unsupported Session aggregation')
    factors = bar_factor_frame(values)
    keyed = grid[['timestamp_utc', 'instrument_id', 'session_id']].merge(
        factors, on=['timestamp_utc', 'instrument_id'], how='left', validate='one_to_one')
    grouped = keyed.sort_values('timestamp_utc').groupby(['session_id', 'instrument_id']).factor_value
    if aggregation == 'last':
        daily = grouped.agg(lambda series: series.iloc[-1])
    elif aggregation == 'sum':
        daily = grouped.sum(min_count=1)
    else:
        daily = grouped.agg(aggregation)
    wide = daily.unstack('instrument_id')
    wide.index = pd.DatetimeIndex(wide.index, name='date')
    wide.columns.name = 'stock_code'
    return wide


def session_daily_pricing(grid: pd.DataFrame, calendars: dict) -> pd.Series:
    """Session-close pricing for daily Alphalens, bound to one TradingCalendar."""
    from utils.trading_calendar import TradingCalendar

    identities = {calendar.calendar_hash for calendar in calendars.values()}
    if len(identities) != 1:
        raise ContractValidationError('daily cross-sectional pricing requires one shared calendar')
    values = grid[['timestamp_utc', 'instrument_id', 'close']].rename(columns={'close': 'factor_value'})
    daily = session_daily_frame(values, grid)
    pricing = daily.rename_axis(columns='asset').unstack().rename('close')
    calendar = next(iter(calendars.values()))
    days = calendar.sessions()
    pricing.attrs['trading_calendar'] = TradingCalendar(days, calendar.start, calendar.end,
                                                       calendar_id=calendar.calendar_id)
    return pricing



def read_slot_factor_values(
    store_root: str | Path, slot: MarketSlot, *, factor_id: str, frequency: str, start_date: str, end_date: str,
    factor_identity_hash: str | None = None, instruments: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Read one committed identity bound to the requested Slot and its calendar."""
    from utils.market_factor_store import MarketFactorValueStore

    store = MarketFactorValueStore(store_root)
    selection = dict(factor_id=factor_id, frequency=frequency, slot_id=slot.slot_id,
                     factor_identity_hash=factor_identity_hash)
    identities = store.list_identities(**selection)
    matches = identities.loc[identities.slot_config_hash.eq(slot.slot_config_hash)]
    market_db = slot_market_db_identity(slot)
    if not matches.empty:
        matches = matches.loc[matches.slot_data_fingerprint.map(lambda value: market_db_identity(value) == market_db)]
    if len(matches) != 1:
        raise ValueError('Specify one committed factor identity matching the Slot ID, config, market database and frequency')
    identity = matches.iloc[0]
    view = getattr(slot.bar_feed, 'view', 'ETH')
    if identity.session_view != view:
        raise ValueError('Factor identity session_view does not match the Slot')
    selection['factor_identity_hash'] = identity.factor_identity_hash
    values = store.read_factor_values(**selection, session_view=view, instruments=instruments,
                                      session_start=start_date, session_end=end_date)
    if values.empty:
        raise ValueError('No committed factor values match the evaluation window and instruments')
    if not values.calendar_hash.eq(slot.calendar.calendar_hash).all():
        raise ValueError('Factor values calendar_hash does not match the Slot')
    return values, identity
