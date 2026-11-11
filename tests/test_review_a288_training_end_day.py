"""NERE-202 approved training-label upper bound; no historical data."""
import duckdb
import pandas as pd
import pytest
from factor_synthesis.core.factor_ops import calculate_forward_returns
from calendar_test_support import seed_calendar

@pytest.mark.parametrize('route',['main_raw','fallback','pit'])
def test_training_includes_final_session_minute_close(route):
    with duckdb.connect(':memory:') as c:
        seed_calendar(c,'2025-01-01','2025-01-10')
        c.execute('CREATE TABLE prices(date TIMESTAMP, stock_code VARCHAR, close DOUBLE, factor DOUBLE)')
        c.executemany('INSERT INTO prices VALUES (?,?,?,?)',[(d,'A',v,f) for d,v,f in [('2025-01-02 09:30',10.,.5),('2025-01-02 15:00',20.,.5),('2025-01-03 09:30',30.,1.),('2025-01-03 23:59:59',40.,1.),('2025-01-04 00:00',999.,2.)]])
        idx=pd.MultiIndex.from_tuples([(pd.Timestamp('2025-01-02'),'A')],names=['date','stock_code'])
        actual=calculate_forward_returns(pd.DataFrame({'x':[1.]},index=idx),c,'missing' if route=='fallback' else 'prices','prices',period=1,pit_enabled=route=='pit')
        assert actual.index.equals(idx)
        assert actual.tolist()==([3.] if route=='pit' else [1.])
