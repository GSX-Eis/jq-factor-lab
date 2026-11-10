"""Hybrid data-fetch helper for factors that need a short recent intraday (5m)
window plus a long daily baseline window.

Why this exists
---------------
Some intraday factors compute a *genuinely intraday* signal over a short,
recent window (e.g. a 20-trading-day high-frequency beta) but also need *long
daily baselines* (e.g. a 120-trading-day volatility baseline). Fetching the
whole long window at 5-minute resolution pulls ~48x more rows than necessary
(48 bars per trading day) and is the main reason these factors exceed the
3000-bar data window.

This helper fetches:
  * recent 5-minute bars only for the short intraday window, and
  * wide daily rows for the long baseline window,

by temporarily overriding the factor's ``start_date`` and ``data_frequency``,
mirroring the established override pattern already used inside individual
factors using the same fetch interface. It deliberately
lives outside ``FactorBase`` to keep the base class' blast radius unchanged.

The explicit-window helpers operate on the ``FactorBase`` fetch surface:
``start_date``, ``end_date``, ``data_frequency`` and ``_fetch_raw_data(...)``.
``history_start`` and ``fetch_hybrid_price_data`` additionally require a
non-null ``_duckdb_adapter`` whose ``trading_calendar()`` returns a verified
``TradingCalendar``. The fetch surface alone does not provide this calendar.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, List, Protocol, Sequence, Tuple

import pandas as pd

from utils.trading_calendar import TradingCalendar

__all__ = [
    "trading_days_to_calendar_days",
    "history_start",
    "override_fetch_scope",
    "fetch_recent_5m",
    "fetch_wide_daily",
    "fetch_hybrid_price_data",
]

# A-share trading calendar has ~244 trading days per 365 calendar days.
_TRADING_DAYS_PER_YEAR: float = 244.0
_CALENDAR_DAYS_PER_YEAR: float = 365.0


class _Fetchable(Protocol):
    """Fetch surface for helpers receiving an already resolved start date."""

    start_date: str
    end_date: str
    data_frequency: str

    def _fetch_raw_data(self, fields: List[str], table_name: str = ...) -> pd.DataFrame: ...


class _CalendarProvider(Protocol):
    """Adapter surface needed to resolve a verified historical window."""

    def trading_calendar(self) -> TradingCalendar: ...


class _CalendarFetchable(_Fetchable, Protocol):
    """Fetch surface with a configured, non-null calendar provider."""

    @property
    def _duckdb_adapter(self) -> _CalendarProvider: ...


def trading_days_to_calendar_days(trading_days: int, slack: float = 0.15) -> int:
    """Legacy estimate for standalone factor formulas; not a verified calendar.

    The price tables are indexed by calendar date, so a rolling window measured
    in trading days must be widened to span weekends/holidays, plus a small
    ``slack`` as extra margin. This cannot guarantee session coverage.
    Shared local fetch paths use ``history_start`` instead.

    Args:
        trading_days: Window length expressed in trading days.
        slack: Extra fractional margin applied on top of the calendar span.

    Returns:
        Calendar-day buffer (``>= 0``); ``0`` when ``trading_days <= 0``.
    """
    if trading_days <= 0:
        return 0
    calendar_span = trading_days * (_CALENDAR_DAYS_PER_YEAR / _TRADING_DAYS_PER_YEAR)
    return int(math.ceil(calendar_span * (1.0 + float(slack))))


def history_start(
    factor: _CalendarFetchable,
    anchor: str | pd.Timestamp,
    trading_days: int,
    *,
    slack: float = 0.,
    min_calendar_days: int = 0,
) -> str:
    """Resolve warmup through ``factor._duckdb_adapter.trading_calendar()``.

    Request ceil(trading_days * (1 + slack)) sessions strictly before anchor,
    then extend further if needed to satisfy the min_calendar_days floor.
    Even zero trading_days can extend history when that floor is positive.
    Insufficient calendar coverage raises; security/bar completeness is not
    guaranteed by the calendar.
    """
    if not math.isfinite(slack) or slack < 0 or min_calendar_days < 0:
        raise ValueError('Warmup slack and calendar-day floor must be non-negative')
    if isinstance(trading_days, bool) or not isinstance(trading_days, int) or trading_days < 0:
        raise ValueError('Warmup trading_days must be a non-negative integer')
    calendar = factor._duckdb_adapter.trading_calendar()
    anchor = pd.Timestamp(anchor).normalize()
    days = calendar.previous_sessions(anchor, math.ceil(trading_days * (1 + slack)))
    start = days[0] if len(days) else anchor
    start = min(start, anchor - pd.Timedelta(days=min_calendar_days))
    calendar.validate_coverage(start, anchor)
    return start.strftime('%Y-%m-%d')


@contextmanager
def override_fetch_scope(
    factor: _Fetchable,
    *,
    start_date: str | None = None,
    data_frequency: str | None = None,
) -> Iterator[None]:
    """Temporarily override a factor's fetch scope, then restore it.

    The DuckDB fetch path in ``FactorBase`` keys off
    ``self.start_date``/``self.end_date``/``self.data_frequency``. Overriding
    these around a single ``_fetch_raw_data`` call lets one factor read a
    different frequency and/or date range for part of its inputs without
    permanently mutating its configured analysis scope.
    """
    original_start = factor.start_date
    original_frequency = factor.data_frequency
    try:
        if start_date is not None:
            factor.start_date = start_date
        if data_frequency is not None:
            factor.data_frequency = data_frequency
        yield
    finally:
        factor.start_date = original_start
        factor.data_frequency = original_frequency


def fetch_recent_5m(
    factor: _Fetchable,
    fields: Sequence[str],
    recent_start_date: str,
    table_name: str = "price_data",
) -> pd.DataFrame:
    """Fetch 5-minute bars for the recent window ``[recent_start_date, end_date]``."""
    with override_fetch_scope(factor, start_date=recent_start_date, data_frequency="5m"):
        return factor._fetch_raw_data(fields=list(fields), table_name=table_name)


def fetch_wide_daily(
    factor: _Fetchable,
    fields: Sequence[str],
    long_start_date: str,
    table_name: str = "price_data",
) -> pd.DataFrame:
    """Fetch daily rows for the long window ``[long_start_date, end_date]``."""
    with override_fetch_scope(factor, start_date=long_start_date, data_frequency="daily"):
        return factor._fetch_raw_data(fields=list(fields), table_name=table_name)


def fetch_hybrid_price_data(
    factor: _CalendarFetchable,
    *,
    anchor_start_date: str | pd.Timestamp,
    recent_trading_days: int,
    long_trading_days: int,
    intraday_fields: Sequence[str],
    daily_fields: Sequence[str],
    table_name: str = "price_data",
    recent_slack: float = 0.15,
    long_slack: float = 0.15,
    min_recent_calendar_days: int = 10,
    min_long_calendar_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch a short recent 5m window and a long daily baseline window.

    Each start date uses ceil(trading_days * (1 + slack)) verified sessions
    strictly before ``anchor_start_date``, then extends further if needed to
    satisfy its minimum natural-day span. Both reads end at factor.end_date:
    the lookbacks control warmup, not the total number of returned sessions.
    Missing calendar coverage raises rather than shortening the window.

    Args:
        factor: Fetch surface with a non-null _duckdb_adapter whose
            trading_calendar() returns a verified TradingCalendar.
        anchor_start_date: The factor's real start date (before extension).
        recent_trading_days: Lookback (trading days) of the intraday 5m signal.
        long_trading_days: Lookback (trading days) of the daily baseline.
        intraday_fields: Fields to request from the 5m table.
        daily_fields: Fields to request from the daily table.
        table_name: Logical price table name (resolves to 5m/daily physically).
        recent_slack: Extra fraction of 5m warmup trading days, rounded up.
        long_slack: Extra fraction of daily warmup trading days, rounded up.
        min_recent_calendar_days: Minimum natural-day span before the anchor
            for 5m warmup; may extend beyond the session-based start.
        min_long_calendar_days: Minimum natural-day span before the anchor
            for daily warmup; may extend beyond the session-based start.

    Returns:
        Tuple ``(intraday_5m_df, daily_df)``.
    """
    anchor = pd.to_datetime(anchor_start_date).normalize()

    recent_start = history_start(factor, anchor, recent_trading_days,
                                 slack=recent_slack, min_calendar_days=min_recent_calendar_days)
    long_start = history_start(factor, anchor, long_trading_days,
                               slack=long_slack, min_calendar_days=min_long_calendar_days)

    intraday_df = fetch_recent_5m(factor, intraday_fields, recent_start, table_name)
    daily_df = fetch_wide_daily(factor, daily_fields, long_start, table_name)
    return intraday_df, daily_df
