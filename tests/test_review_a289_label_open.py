"""NERE-210: isolate label daily-open selection from execution prices."""
import duckdb
import pandas as pd
import pytest
from factor_synthesis.core.factor_data_loader import get_open_price_data, PITValidationError
from factor_synthesis.core.factor_ops import calculate_scheduled_price_returns


def fixture_conn(daily=False, first_missing=False):
    c=duckdb.connect(':memory:')
    c.execute('CREATE TABLE price_data_5m(date TIMESTAMP, stock_code VARCHAR, open DOUBLE, factor DOUBLE)')
    c.executemany('INSERT INTO price_data_5m VALUES (?,?,?,?)',[
        ('2025-01-02 15:00','A',20.,1.),('2025-01-02 09:30','A',None if first_missing else 10.,1.),
        ('2025-01-08 15:00','A',80.,1.),('2025-01-08 09:30','A',30.,1.)])
    if daily:
        c.execute('CREATE TABLE price_data_daily(date DATE, stock_code VARCHAR, open DOUBLE, factor DOUBLE)')
        c.executemany('INSERT INTO price_data_daily VALUES (?,?,?,?)',[('2025-01-02','A',12.,1.),('2025-01-08','A',48.,1.)])
    return c


@pytest.mark.parametrize('pit',[False,True])
@pytest.mark.parametrize('daily',[False,True])
def test_label_prefers_daily_or_original_first_minute(daily,pit):
    with fixture_conn(daily) as c:
        got=get_open_price_data(c,'price_data_5m',['A'],'2025-01-02','2025-01-08',pit_enabled=pit,for_label=True)
        assert got.index.tolist()==pd.to_datetime(['2025-01-02','2025-01-08']).tolist()
        assert got.A.tolist()==([12.,48.] if daily else [10.,30.])
        idx=pd.MultiIndex.from_tuples([(pd.Timestamp('2025-01-02'),'A')],names=['date','stock_code'])
        label=calculate_scheduled_price_returns(pd.DataFrame({'x':[1.]},index=idx),got,[('2025-01-02','2025-01-08')])
        assert label.tolist()==([3.] if daily else [2.])


@pytest.mark.parametrize('pit',[False,True])
def test_first_missing_bar_is_not_replaced(pit):
    with fixture_conn(first_missing=True) as c:
        got=get_open_price_data(c,'price_data_5m',['A'],'2025-01-02','2025-01-08',pit_enabled=pit,for_label=True)
        assert pd.isna(got.loc['2025-01-02','A'])
        assert got.loc['2025-01-08','A']==30.
        idx=pd.MultiIndex.from_tuples([(pd.Timestamp('2025-01-02'),'A')],names=['date','stock_code'])
        assert calculate_scheduled_price_returns(pd.DataFrame({'x':[1.]},index=idx),got,[('2025-01-02','2025-01-08')]).empty


@pytest.mark.parametrize('pit',[False,True])
def test_daily_missing_never_borrows_minute_price(pit):
    with fixture_conn(daily=True) as c:
        c.execute("UPDATE price_data_daily SET open=NULL WHERE date='2025-01-02'")
        got=get_open_price_data(c,'price_data_5m',['A'],'2025-01-02','2025-01-08',pit_enabled=pit,for_label=True)
        assert pd.isna(got.loc['2025-01-02','A'])
        assert got.loc['2025-01-08','A']==48.


@pytest.mark.parametrize('pit',[False,True])
def test_first_bar_selection_is_per_stock_not_frame_row(pit):
    with fixture_conn() as c:
        c.executemany('INSERT INTO price_data_5m VALUES (?,?,?,?)',[
            ('2025-01-02 09:35','B',5.,1.),('2025-01-02 15:00','B',99.,1.),
            ('2025-01-08 09:35','B',10.,1.),('2025-01-08 15:00','B',999.,1.)])
        got=get_open_price_data(c,'price_data_5m',['A','B'],'2025-01-02','2025-01-08',pit_enabled=pit,for_label=True)
        assert got.A.tolist()==[10.,30.] and got.B.tolist()==[5.,10.]


def test_label_pit_rejects_invalid_observed_first_factor():
    with fixture_conn() as c:
        c.execute("UPDATE price_data_5m SET factor=0 WHERE date='2025-01-02 09:30'")
        with pytest.raises(PITValidationError):
            get_open_price_data(c,'price_data_5m',['A'],'2025-01-02','2025-01-08',for_label=True)


def test_pit_uses_selected_first_observations_only():
    with fixture_conn() as c:
        c.execute("UPDATE price_data_5m SET factor=0.5 WHERE date='2025-01-02 09:30'")
        c.execute("UPDATE price_data_5m SET factor=99 WHERE date='2025-01-08 15:00'")
        got=get_open_price_data(c,'price_data_5m',['A'],'2025-01-02','2025-01-08',for_label=True)
        assert got.A.tolist()==[5.,30.]


def test_missing_first_open_does_not_validate_its_unobserved_factor():
    with fixture_conn(first_missing=True) as c:
        c.execute("UPDATE price_data_5m SET factor=0 WHERE date='2025-01-02 09:30'")
        got=get_open_price_data(c,'price_data_5m',['A'],'2025-01-02','2025-01-08',for_label=True)
        assert pd.isna(got.loc['2025-01-02','A']) and got.loc['2025-01-08','A']==30.
