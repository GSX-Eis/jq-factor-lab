"""NERE-202 approved inclusive end-day, retaining PIT and raw fallback rules."""
import duckdb
import pandas as pd
import pytest
from calendar_test_support import seed_calendar
from factor_synthesis.core.factor_data_loader import get_pricing_data


@pytest.mark.parametrize('future',[0,1])
@pytest.mark.parametrize('mode',['pit','raw','fallback'])
def test_end_day_includes_intraday_prices_but_not_next_midnight(future,mode):
    with duckdb.connect(':memory:') as c:
        seed_calendar(c)
        c.execute('create table prices(date timestamp,stock_code varchar,close double,factor double)')
        c.executemany('insert into prices values (?,?,?,?)',[
            ('2024-01-02 00:00:00',"S'1",10,1),('2024-01-02 15:00:00',"S'1",20,2),
            ('2024-01-03 00:00:00',"S'1",30,3),('2024-01-03 15:00:00',"S'1",40,4),
            ('2024-01-04 00:00:00',"S'1",100,10)])
        c.execute('create table factor_data as select date,stock_code,close close_price from prices')
        actual=get_pricing_data(c,'missing' if mode=='fallback' else 'prices',["S'1"],
            pd.DatetimeIndex(['2024-01-02']),pit_enabled=mode=='pit',future_periods=future)
        assert actual.iloc[-1]==(20 if future==0 else 40)
        assert len(actual)==future+1
        if future and mode=='pit': assert actual.iloc[0]==10
        assert actual.index.get_level_values('date').max()==pd.Timestamp('2024-01-02')+pd.Timedelta(days=future)
