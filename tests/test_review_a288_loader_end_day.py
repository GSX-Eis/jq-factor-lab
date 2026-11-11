"""NERE-202 approved open/ratio upper bounds, memory-only prices."""
import duckdb
import pandas as pd
import pytest
from factor_synthesis.core import factor_data_loader as loader

@pytest.fixture
def prices():
    with duckdb.connect(':memory:') as c:
        c.execute('CREATE TABLE prices(date TIMESTAMP, stock_code VARCHAR, open DOUBLE, factor DOUBLE)')
        c.executemany('INSERT INTO prices VALUES (?,?,?,?)',[(d,'A',v,f) for d,v,f in [('2025-01-02 09:30',10.,.5),('2025-01-02 15:00',20.,.5),('2025-01-03 09:30',30.,1.),('2025-01-03 23:59:59',40.,1.),('2025-01-04 00:00',999.,2.)]])
        yield c

@pytest.mark.parametrize('pit',[False,True])
@pytest.mark.parametrize('end',['2025-01-03',pd.Timestamp('2025-01-03')])
def test_open_whole_end_day_without_next_midnight(prices,pit,end):
    actual=loader.get_open_price_data(prices,'prices',['A'],'2025-01-02',end,pit_enabled=pit)
    assert actual.index.tolist()==pd.to_datetime(['2025-01-02 09:30','2025-01-02 15:00','2025-01-03 09:30','2025-01-03 23:59:59'],format='mixed').tolist()
    assert actual.A.tolist()==([5.,10.,30.,40.] if pit else [10.,20.,30.,40.])

@pytest.mark.parametrize('end',['2025-01-03',pd.Timestamp('2025-01-03')])
def test_ratio_whole_end_day_keeps_last_daily_factor(prices,end):
    actual=loader.get_factor_ratio_data(prices,'prices',['A'],'2025-01-02',end)
    assert actual.index.tolist()==pd.to_datetime(['2025-01-02','2025-01-03']).tolist()
    assert actual.A.tolist()==[1.,2.]
