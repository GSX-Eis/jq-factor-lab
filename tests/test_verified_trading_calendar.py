import hashlib
import json

import duckdb
import pandas as pd
import pytest

from utils.trading_calendar import TradingCalendar, CalendarCoverageError, load_trading_calendar
from scripts.import_trading_calendar import import_calendar
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from factor_synthesis.core.factor_data_loader import process_raw_factor_df


@pytest.fixture
def calendar():
    # Explicit fixture: Oct 1-7 are closed; Oct 8/9 are open. No external calls.
    return TradingCalendar(['2024-09-30', '2024-10-08', '2024-10-09'], '2024-09-30', '2024-10-09')


def seed_calendar(conn, calendar):
    days = pd.date_range(calendar.start, calendar.end)
    frame = pd.DataFrame({'calendar_id':'LOCAL_A_SHARE_XSHG_CROSSCHECKED',
                          'date':days, 'is_open':days.isin(calendar.sessions())})
    conn.register('seed_calendar_rows', frame)
    conn.execute('CREATE TABLE trading_calendar AS SELECT calendar_id, CAST(date AS DATE) date, is_open FROM seed_calendar_rows')


def test_holiday_windows_and_bounds(calendar):
    assert calendar.previous_sessions('2024-10-08', 1)[0] == pd.Timestamp('2024-09-30')
    assert calendar.next_sessions('2024-09-30', 1)[0] == pd.Timestamp('2024-10-08')
    assert calendar.sessions('2024-10-01', '2024-10-07').empty
    assert calendar.previous_sessions('2024-10-08', 1, include_anchor=True)[0] == pd.Timestamp('2024-10-08')
    for count in [-1, True, 1.5]:
        with pytest.raises(ValueError): calendar.next_sessions('2024-09-30', count)
    with pytest.raises(CalendarCoverageError): calendar.previous_sessions('2024-09-30', 1)
    with pytest.raises(CalendarCoverageError): calendar.sessions('2024-09-29','2024-10-09')
    with pytest.raises(CalendarCoverageError): calendar.next_sessions('2024-10-09',1)


def test_dense_database_validation(calendar):
    with duckdb.connect() as conn:
        seed_calendar(conn,calendar)
        assert load_trading_calendar(conn).sessions().equals(calendar.sessions())
        conn.execute("DELETE FROM trading_calendar WHERE date='2024-10-03'")
        with pytest.raises(ValueError, match='one DATE/BOOLEAN'): load_trading_calendar(conn)


def test_missing_market_day_does_not_shorten_label(calendar):
    pricing = pd.Series([10.,12.], index=pd.MultiIndex.from_product([
        ['A'], pd.to_datetime(['2024-09-30','2024-10-09'])], names=['asset','date']))
    result = ForwardReturnsCalculator(trading_calendar=calendar).calculate_forward_returns(pricing, (1,2),0)
    assert pd.isna(result.loc[(pd.Timestamp('2024-09-30'),'A'),'1D'])
    assert result.loc[(pd.Timestamp('2024-09-30'),'A'),'2D'] == pytest.approx(.2)


def test_complete_prices_same_results_and_ipo_missing(calendar):
    wide = pd.DataFrame({'A':[10.,11.,12.], 'IPO':[None,20.,21.]}, index=calendar.sessions())
    pricing = wide.rename_axis(index='date', columns='asset').stack()
    result = ForwardReturnsCalculator(trading_calendar=calendar).calculate_forward_returns(pricing,(1,),0)
    expected = (wide.shift(-1)/wide-1).stack().to_frame('1D')
    expected.index.names=['date','asset']
    pd.testing.assert_frame_equal(result,expected)


def test_factor_lag_uses_calendar_even_when_price_table_missing(calendar):
    raw = pd.DataFrame({'date':pd.to_datetime(['2024-09-30','2024-10-09']),
        'stock_code':['A','A'],'factor_name':['x','x'],'factor_value':[10.,30.]})
    with duckdb.connect() as conn:
        seed_calendar(conn,calendar)
        result=process_raw_factor_df(raw,['x'],'2024-09-30','2024-10-09',price_conn=conn,price_table='missing')
        assert result.loc[(pd.Timestamp('2024-10-08'),'A'),'x']==10.
        assert (pd.Timestamp('2024-10-09'),'A') not in result.index
        conn.execute('DROP TABLE trading_calendar')
        with pytest.raises(duckdb.CatalogException):
            process_raw_factor_df(raw,['x'],'2024-09-30','2024-10-09',price_conn=conn)


def test_verified_import_version_and_roundtrip(tmp_path,calendar,monkeypatch):
    from utils.local_outputs import managed_output
    monkeypatch.setattr('scripts.import_trading_calendar.managed_output',
                        lambda _root, purpose: managed_output(tmp_path, purpose))
    csv=tmp_path/'days.csv'; pd.DataFrame({'date':calendar.sessions().strftime('%Y-%m-%d')}).to_csv(csv,index=False)
    receipt=tmp_path/'verified.json'
    receipt.write_text(json.dumps({'status':'verified','differences':{'local_only':[],'online_only':[]},
        'calendar_id':'LOCAL_A_SHARE_XSHG_CROSSCHECKED','start_date':'2024-09-30','end_date':'2024-10-09',
        'trading_days':3,'files':{csv.name:hashlib.sha256(csv.read_bytes()).hexdigest()}}))
    db=tmp_path/'input.duckdb'
    first=import_calendar(db,csv,receipt)
    assert first['ok']
    assert import_calendar(db,csv,receipt)==first
    assert load_trading_calendar(duckdb_path=db,bindings={}).sessions().equals(calendar.sessions())
    with duckdb.connect(str(db),read_only=True) as conn:
        head=conn.execute("SELECT version FROM meta.input_heads WHERE target_table='main.trading_calendar'").fetchone()[0]
    bindings={'input_db':str(db),'inputs':{'trading_calendar':{'role':'input','table':'main.trading_calendar','version':head}}}
    assert load_trading_calendar(duckdb_path=db,bindings=bindings).sessions().equals(calendar.sessions())
    csv.write_text(csv.read_text()+'2024-10-10\n')
    with pytest.raises(ValueError,match='hash'): import_calendar(db,csv,receipt)




def test_price_fetcher_uses_holiday_calendar_and_rejects_future_overflow(tmp_path, calendar):
    from backtest_and_analysis.price_fetcher import PriceFetcher
    db=tmp_path/'prices.duckdb'
    with duckdb.connect(str(db)) as conn:
        seed_calendar(conn,calendar)
        conn.execute("CREATE TABLE price_data_daily AS SELECT d::DATE date, 'A' stock_code, 10. AS close, 10. AS open, 10. AS high, 10. AS low, 1. AS volume, 1. AS money FROM (VALUES ('2024-09-30'),('2024-10-08'),('2024-10-09')) t(d)")
    reader=PriceFetcher(db,'price_data_daily')
    try:
        prices=reader.fetch(['A'],'2024-09-30','2024-09-30',1)
        assert prices.index.get_level_values('date').tolist()==list(pd.to_datetime(['2024-09-30','2024-10-08']))
        with pytest.raises(CalendarCoverageError): reader.fetch(['A'],'2024-10-09','2024-10-09',1)
    finally: reader.close()


def test_synthesis_training_labels_share_sessions_with_and_without_pit(calendar):
    from factor_synthesis.core.factor_ops import calculate_forward_returns
    factors=pd.DataFrame({'x':[1.]}, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2024-09-30'),'A')], names=['date','stock_code']))
    with duckdb.connect() as conn:
        seed_calendar(conn,calendar)
        conn.execute("CREATE TABLE prices AS SELECT d::DATE date, 'A' stock_code, p AS close, 1. AS factor FROM (VALUES ('2024-09-30',10.),('2024-10-09',12.)) t(d,p)")
        for pit in (False,True):
            one=calculate_forward_returns(factors,conn,'prices','unused',1,pit_enabled=pit)
            two=calculate_forward_returns(factors,conn,'prices','unused',2,pit_enabled=pit)
            assert one.empty
            assert two.iloc[0]==pytest.approx(.2)


def test_warmup_resolves_exact_sessions_not_ratio(calendar):
    from types import SimpleNamespace
    from utils.hybrid_fetch import history_start
    factor=SimpleNamespace(_duckdb_adapter=SimpleNamespace(trading_calendar=lambda:calendar))
    assert history_start(factor,'2024-10-08',1)=='2024-09-30'
    with pytest.raises(CalendarCoverageError): history_start(factor,'2024-10-08',2)


def test_missing_calendar_is_not_hidden_by_alphalens(monkeypatch):
    from backtest_and_analysis.alphalens_runner import AlphalensRunner
    monkeypatch.delenv('QUANT_DATA_BINDINGS',raising=False)
    pricing=pd.Series([10.],index=pd.MultiIndex.from_tuples(
        [('A',pd.Timestamp('2024-01-02'))],names=['asset','date']))
    with pytest.raises(ValueError,match='Supply a calendar'):
        AlphalensRunner().run(pd.DataFrame({'A':[1.]},index=pd.to_datetime(['2024-01-02'])),pricing)


def test_label_cache_depends_on_calendar_content(tmp_path):
    closed=TradingCalendar(['2024-09-30','2024-10-09'],'2024-09-30','2024-10-09')
    opened=TradingCalendar(['2024-09-30','2024-10-08','2024-10-09'],'2024-09-30','2024-10-09')
    prices=pd.Series([10.,12.],index=pd.MultiIndex.from_product([
        ['A'],pd.to_datetime(['2024-09-30','2024-10-09'])],names=['asset','date']))
    calc=ForwardReturnsCalculator(str(tmp_path),trading_calendar=closed)
    first=calc.calculate_or_load_forward_returns(prices,['A'],'2024-09-30','2024-10-09',(1,),0)
    calc.trading_calendar=opened
    with pytest.raises(ValueError, match='未能计算出任何前瞻收益率'):
        calc.calculate_or_load_forward_returns(prices,['A'],'2024-09-30','2024-10-09',(1,),0)
    assert len(first)==1


@pytest.mark.parametrize('method', ['previous_sessions', 'next_sessions'])
@pytest.mark.parametrize('count', [0, 1])
@pytest.mark.parametrize('include', ['False', 'True', 0, 1, 1.0, None, []])
def test_session_window_requires_boolean_include_anchor(calendar, method, count, include):
    with pytest.raises(ValueError, match='include_anchor must be a boolean'):
        getattr(calendar, method)('2024-10-08', count, include_anchor=include)


@pytest.mark.parametrize('method,include,expected', [
    ('previous_sessions', False, '2024-09-30'),
    ('previous_sessions', True, '2024-10-08'),
    ('next_sessions', False, '2024-10-09'),
    ('next_sessions', True, '2024-10-08'),
])
def test_session_window_keeps_boolean_anchor_semantics(calendar, method, include, expected):
    result = getattr(calendar, method)('2024-10-08', 1, include_anchor=include)
    assert result.tolist() == [pd.Timestamp(expected)]
    assert getattr(calendar, method)('2024-10-08', 0, include_anchor=include).empty


@pytest.mark.parametrize('index_kind', ['string', 'datetime', 'date'])
@pytest.mark.parametrize('as_series', [False, True])
@pytest.mark.parametrize('bounds', [(None, None), ('2024-10-08', '2024-10-09')])
def test_align_uses_validated_dates_without_changing_input(calendar, index_kind, as_series, bounds):
    dates = pd.DatetimeIndex(['2024-10-09', '2024-09-30'], name='date')
    index = dates.strftime('%Y-%m-%d') if index_kind == 'string' else (
        pd.Index(dates.date, name='date') if index_kind == 'date' else dates)
    frame = pd.Series([12., 10.], index=index, name='A')
    if not as_series: frame = frame.to_frame()
    frame.attrs['source'] = 'test-input'
    original = frame.copy(deep=True)
    expected = pd.Series([10., float('nan'), 12.], index=calendar.sessions().rename('date'), name='A')
    if bounds[0] is not None: expected = expected.loc[bounds[0]:bounds[1]]
    if not as_series: expected = expected.to_frame()
    expected.attrs['source'] = 'test-input'
    result = calendar.align(frame, *bounds)
    compare = pd.testing.assert_series_equal if as_series else pd.testing.assert_frame_equal
    compare(result, expected)
    compare(frame, original)
    assert result.attrs == original.attrs
    result.iloc[-1] = 999.
    compare(frame, original)


@pytest.mark.parametrize('index', [
    ['2024-09-30', 'bad-date'],
    ['2024-09-30', None],
    ['2024-09-30', '2024-09-30'],
    ['2024-09-30', '2024-10-01'],
    ['2024-09-30 12:00', '2024-10-09'],
    pd.DatetimeIndex(['2024-09-30', '2024-10-09'], tz='Asia/Shanghai'),
])
def test_align_still_rejects_invalid_daily_observations(calendar, index):
    with pytest.raises((ValueError, TypeError)):
        calendar.align(pd.DataFrame({'A': [10., 12.]}, index=index))
