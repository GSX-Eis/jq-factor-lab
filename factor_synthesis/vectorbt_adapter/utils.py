"""
VectorBT Adapter Utility Functions

Helper functions for rebalance date computation and trading calendar management.
"""

from datetime import datetime, timedelta
from typing import List, Set

import pandas as pd
import numpy as np


def build_trading_calendar(price_df: pd.DataFrame, *, calendar=None) -> pd.DatetimeIndex:
    """Validate observations against the independent calendar and expose sessions."""
    from utils.trading_calendar import load_trading_calendar
    calendar = calendar if calendar is not None else load_trading_calendar()
    return calendar.align(price_df).index


def compute_rebalance_dates(
    trading_calendar: pd.DatetimeIndex,
    start_date: datetime,
    end_date: datetime,
    rebalance_freq: str = 'weekly',  # 'daily', 'weekly', 'monthly'
    weekday: int = 4,  # Friday = 4 (legacy, use nth_trading_day instead)
    nth_trading_day: int = 1,  # 1 = first trading day, 2 = second, etc.
    holding_period: int = 1,  # For weekly: rebalance every N weeks
) -> List[datetime]:
    """
    Compute rebalance dates based on trading calendar.
    
    Args:
        trading_calendar: Available trading days
        start_date: Backtest start date
        end_date: Backtest end date
        rebalance_freq: Rebalancing frequency ('daily', 'weekly', 'monthly')
        weekday: (Legacy) Target weekday for weekly rebalancing (0=Mon, 4=Fri)
        nth_trading_day: For weekly/monthly, select nth trading day (1=first, 2=second, etc.)
        holding_period: For weekly mode, rebalance every N weeks (default=1, every week)
        
    Returns:
        List of rebalance dates (as datetime objects)
    """
    if trading_calendar.empty:
        return []
    
    # Filter to backtest period
    mask = (trading_calendar >= start_date) & (trading_calendar <= end_date)
    period_dates = trading_calendar[mask]
    
    if period_dates.empty:
        return []
    
    rebalance_dates: List[datetime] = []
    
    if rebalance_freq == 'daily':
        rebalance_dates = [d.to_pydatetime() for d in period_dates]
        
    elif rebalance_freq == 'weekly':
        # Group by (year, week_number) and select nth trading day
        week_groups: dict = {}
        for date in period_dates:
            week_id = (date.isocalendar()[0], date.isocalendar()[1])
            if week_id not in week_groups:
                week_groups[week_id] = []
            week_groups[week_id].append(date.to_pydatetime())
        
        # Select nth trading day every holding_period weeks
        for i, week_id in enumerate(sorted(week_groups.keys())):
            if i % holding_period == 0:  # Every holding_period weeks
                week_dates = week_groups[week_id]
                # nth_trading_day is 1-indexed
                idx = min(nth_trading_day - 1, len(week_dates) - 1)
                if idx >= 0:
                    rebalance_dates.append(week_dates[idx])
                    
    elif rebalance_freq == 'monthly':
        # Group by (year, month) and select nth trading day
        month_groups: dict = {}
        for date in period_dates:
            month_id = (date.year, date.month)
            if month_id not in month_groups:
                month_groups[month_id] = []
            month_groups[month_id].append(date.to_pydatetime())
        
        # Select nth trading day from each month
        for month_id in sorted(month_groups.keys()):
            month_dates = month_groups[month_id]
            idx = min(nth_trading_day - 1, len(month_dates) - 1)
            if idx >= 0:
                rebalance_dates.append(month_dates[idx])
    
    return rebalance_dates



def filter_stocks_by_pool(
    stocks: List[str],
    date: datetime,
    dynamic_pools: dict,
) -> Set[str]:
    """
    Get valid stocks for a given date based on dynamic pool configuration.
    
    Args:
        stocks: Full list of stocks
        date: Target date
        dynamic_pools: Dict mapping date strings to stock lists
        
    Returns:
        Set of valid stock codes for the date
    """
    if not dynamic_pools:
        return set(stocks)
    
    date_str = date.strftime('%Y-%m-%d')
    
    # Find the most recent pool date <= target date
    pool_dates = sorted([pd.to_datetime(d) for d in dynamic_pools.keys()])
    
    applicable_date = None
    for pd_date in pool_dates:
        if pd_date <= pd.to_datetime(date):
            applicable_date = pd_date
    
    if applicable_date is None:
        return set()
    
    pool_date_str = applicable_date.strftime('%Y-%m-%d')
    pool_stocks = dynamic_pools.get(pool_date_str, [])
    
    return set(pool_stocks) & set(stocks)
