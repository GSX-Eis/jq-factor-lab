"""Explicit synthetic calendars for unit tests; never a production holiday source."""
import pandas as pd
from utils.trading_calendar import TradingCalendar


def synthetic_calendar(start='2020-01-01', end='2027-12-31'):
    return TradingCalendar(pd.bdate_range(start,end), start, end, 'SYNTHETIC_TEST_WEEKDAYS')


def seed_calendar(conn, start='2020-01-01', end='2027-12-31'):
    calendar=synthetic_calendar(start,end)
    days=pd.date_range(start,end)
    frame=pd.DataFrame({'calendar_id':'LOCAL_A_SHARE_XSHG_CROSSCHECKED',
                        'date':days, 'is_open':days.isin(calendar.sessions())})
    conn.register('_test_calendar_rows',frame)
    conn.execute('CREATE TABLE trading_calendar AS SELECT calendar_id, CAST(date AS DATE) date, is_open FROM _test_calendar_rows')
