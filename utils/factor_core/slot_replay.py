"""Replay explicit Slot factors over session-effective pool memberships."""
from typing import Callable

import numpy as np
import pandas as pd


_AVAILABLE = 'signal_available_at'


def slot_clock(self, bucket_strategy: Callable) -> pd.DataFrame:
    """Expected Bar keys with their completion times on the selected session clock."""
    from utils.session_calendar import SessionCalendar
    from utils.time_axis import BarSchedule

    slot, frequency, view = self.slot, self._frequency_route.frequency, self.session_view
    calendar = slot.calendar
    frames = []
    for instrument in dict.fromkeys(self.all_stocks):
        strategy = bucket_strategy(slot.bar_feed, frequency, instrument)
        grid = calendar.bar_grid(frequency, self.start_date, self.end_date, view=view, bucket_strategy=strategy)
        clock = getattr(calendar, 'default', calendar)
        schedule = (BarSchedule(clock, frequency, view=view, bucket_strategy=strategy)
                    if isinstance(clock, SessionCalendar) else None)
        if _AVAILABLE not in grid:
            if schedule is not None:
                ends = [schedule.available_at(ts) for ts in grid.timestamp_utc]
            else:
                ends = [_protocol_bar_end(calendar, frequency, view, strategy, ts)
                        for ts in grid.timestamp_utc]
            grid = grid.assign(**{_AVAILABLE: pd.DatetimeIndex(ends, tz='UTC')})
        frames.append(grid.assign(instrument_id=instrument))
    if not frames:
        return pd.DataFrame({'timestamp_utc': pd.DatetimeIndex([], tz='UTC'),
                             _AVAILABLE: pd.DatetimeIndex([], tz='UTC'),
                             'session_id': pd.Series(dtype=str), 'instrument_id': pd.Series(dtype=str)})
    return pd.concat(frames, ignore_index=True).sort_values(['instrument_id', 'timestamp_utc'])


def _protocol_bar_end(calendar, frequency, view, strategy, ts):
    """Find the completion boundary from CalendarSlot session segments."""
    session = calendar.session_at(ts, view, frequency=frequency)
    if frequency.kind != 'intraday':
        return session.segments[-1][1]
    remaining = frequency.duration
    for start, end in session.segments:
        left = max(start, ts)
        if left >= end:
            continue
        if strategy != 'session_clock' or remaining <= end - left:
            return min(left + remaining, end)
        remaining -= end - left
    return session.segments[-1][1]


def replay_slot_pools(self, calculate: Callable, to_panel: Callable,
                      window_bars: int, warmup_calendar_days: int, use_last_available: bool,
                      combine_duplicate_dates: bool, rebalance_only: bool) -> pd.DataFrame:
    pools = self._get_stock_pool_intervals()
    if not self.use_dynamic_pool or not pools:
        return self.get_factor_values(force_recalculate=True)

    start = pd.Timestamp(getattr(self, '_factor_start_date', self.start_date)).normalize()
    end = pd.Timestamp(getattr(self, '_factor_end_date', self.end_date)).normalize()
    original_start, original_stocks = self.start_date, self.all_stocks
    outputs = []
    try:
        self.use_dynamic_pool = False
        if warmup_calendar_days > 0:
            self.start_date = (pd.Timestamp(original_start) - pd.Timedelta(days=warmup_calendar_days)).strftime('%Y-%m-%d')
        for pool_start, pool_end, stocks in pools:
            if not stocks or pool_start > pd.Timestamp(self.end_date):
                continue
            if rebalance_only and pool_start <= start:
                continue
            first_day = max(pool_start, start)
            limit = end + pd.Timedelta(days=1) if pool_end is None else min(
                pool_end, end + pd.Timedelta(days=1))
            if not rebalance_only and first_day >= limit:
                continue
            self.all_stocks = list(dict.fromkeys(stocks))
            clock = self._slot_clock()
            session_days = pd.to_datetime(clock.session_id)
            prior = clock.loc[session_days.lt(first_day), 'timestamp_utc'].drop_duplicates().sort_values()
            if len(prior) < window_bars:
                continue
            window = clock.loc[clock.timestamp_utc.ge(prior.iloc[-window_bars])
                               & session_days.lt(first_day if rebalance_only else limit)]
            target_days = pd.to_datetime(window.session_id)
            if rebalance_only:
                in_range = target_days.between(start, end)
                targets = window.loc[in_range if use_last_available else
                                     in_range & window.timestamp_utc.eq(prior.iloc[-1])]
            else:
                targets = window.loc[target_days.ge(first_day)]

            values = calculate(self, window, targets=targets, force_recalculate=True)
            keyed = targets.merge(values, on=['timestamp_utc', 'instrument_id'], validate='one_to_one')
            if keyed.empty:
                continue
            panel = to_panel(self, keyed).replace([np.inf, -np.inf], np.nan).dropna(how='all')
            outputs.append(panel.tail(1) if rebalance_only else panel)
    finally:
        self.start_date, self.all_stocks = original_start, original_stocks
        self.use_dynamic_pool = True

    if not outputs:
        return pd.DataFrame()
    result = pd.concat(outputs)
    if combine_duplicate_dates:
        return result.groupby(level=0, sort=True).first()
    return result.loc[~result.index.duplicated(keep='last')].sort_index()
