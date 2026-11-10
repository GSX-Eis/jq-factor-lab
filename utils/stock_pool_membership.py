"""Shared membership rules for effective-date pools and bounded daily panels."""

from collections.abc import Mapping
from datetime import datetime
from numbers import Number

import numpy as np
import pandas as pd


MEMBERSHIP_CONTRACT = 'effective-date-membership-v1'


class StockPoolMembership:
    """Validate once; resolve [effective date, next effective date) membership."""

    def __init__(self, pools):
        if not isinstance(pools, Mapping) or not pools:
            raise ValueError('Stock pool history must be a nonempty mapping; use [] for an explicitly empty period')
        periods = []
        for key, stocks in pools.items():
            if not isinstance(key, str):
                raise ValueError(f'Invalid effective stock pool date {key!r}: expected YYYY-MM-DD')
            try:
                date = datetime.strptime(key, '%Y-%m-%d')
            except ValueError as exc:
                raise ValueError(f'Invalid effective stock pool date {key!r}: expected YYYY-MM-DD') from exc
            if date.strftime('%Y-%m-%d') != key:
                raise ValueError(f'Invalid effective stock pool date {key!r}: use zero-padded YYYY-MM-DD')
            if not isinstance(stocks, (list, tuple)) or any(not isinstance(s, str) or not s.strip() for s in stocks):
                raise ValueError(f'Stock pool {key!r} must contain a list or tuple of nonempty stock codes')
            if any(s != s.strip() for s in stocks):
                raise ValueError(f'Stock pool {key!r} contains stock codes with surrounding whitespace')
            if len(set(stocks)) != len(stocks):
                raise ValueError(f'Stock pool {key!r} contains duplicate stock codes')
            periods.append((pd.Timestamp(date), tuple(stocks)))
        periods.sort(key=lambda item: item[0])
        self.effective_dates = pd.DatetimeIndex([date for date, _ in periods])
        self.members = tuple(stocks for _, stocks in periods)

    def _resolve(self, dates):
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.Index(dates)
            if any(isinstance(value, (Number, np.bool_)) for value in dates):
                raise ValueError('Membership dates must be dates, not numeric values')
        dates = pd.DatetimeIndex(pd.to_datetime(dates, errors='raise'))
        if dates.hasnans or dates.tz is not None:
            raise ValueError('Membership dates must be valid timezone-naive local dates')
        dates = dates.normalize()
        return dates, self.effective_dates.searchsorted(dates, side='right') - 1

    def filter_wide(self, frame):
        """Preserve all axes; mask outside membership without a full-size boolean panel."""
        if not frame.columns.is_unique:
            raise ValueError('Stock columns must be unique')
        _, positions = self._resolve(frame.index)
        result = frame.copy()
        for position in np.unique(positions):
            rows = np.flatnonzero(positions == position)
            outside = (np.ones(len(frame.columns), dtype=bool) if position < 0
                       else ~frame.columns.isin(self.members[position]))
            result.iloc[rows, np.flatnonzero(outside)] = np.nan
        return result

    def coverage(self, dates):
        """Describe expected membership without filling missing observations."""
        dates, positions = self._resolve(dates)
        return pd.DataFrame({
            'has_pool': positions >= 0,
            'member_count': [len(self.members[p]) if p >= 0 else 0 for p in positions],
        }, index=dates)

    def daily_frame(self, dates):
        """Materialize only requested dates, with one row per (date, stock_code)."""
        dates, positions = self._resolve(dates)
        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError('Daily membership dates must be sorted and unique')
        rows = [(date, stock) for date, position in zip(dates, positions) if position >= 0
                for stock in self.members[position]]
        return pd.DataFrame({
            'date': pd.Series([date for date, _ in rows], dtype='datetime64[ns]'),
            'stock_code': pd.Series([stock for _, stock in rows], dtype='string'),
        })
