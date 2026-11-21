"""Legacy XSHG window declarations, measured on a complete reference Session."""
from functools import lru_cache

from utils.session_calendar import SessionCalendar
from .schedule import BarSchedule, Lookback


@lru_cache(maxsize=16)
def stock_bars_per_session(frequency: str) -> int:
    calendar = SessionCalendar('XSHG', '2026-01-05', '2026-01-05')
    return Lookback(sessions=1).to_bars(BarSchedule(calendar, frequency), end_session='2026-01-05')


def requirement_sessions(*, min_bars: int | None = None, min_days: int | None = None) -> int:
    lookback = Lookback(bars=min_bars) if min_bars is not None else Lookback(sessions=min_days)
    if lookback.sessions is not None:
        return lookback.sessions
    width = stock_bars_per_session('5m')
    return (lookback.bars + width - 1) // width
