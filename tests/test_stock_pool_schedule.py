"""Unified schedules must preserve the ordered output of the pre-refactor weekly path."""
from datetime import datetime
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter


def seed_pool(conn):
    days = pd.date_range('2025-12-01', '2026-04-30')
    closed = pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-15', '2026-01-16'])
    calendar = pd.DataFrame({'calendar_id': 'LOCAL_A_SHARE_XSHG_CROSSCHECKED',
        'date': days, 'is_open': (days.dayofweek < 5) & ~days.isin(closed)})
    conn.execute('CREATE TABLE trading_calendar AS SELECT * FROM calendar')
    rows = []
    for n, day in enumerate(days[calendar.is_open]):
        for i, code in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
            rows.append({'date':day,'stock_code':code,'val_market_cap':float((i+n)%6+1),
                         'is_st':int((n+i)%11==0),'is_new_stock':int(i==5),
                         'paused':int((n+i)%13==0),'close':float(10+i),
                         'open':float(6+i),'high_limit':11.,'low_limit':5.,'volume':float(i+1)})
    fund = pd.DataFrame(rows)
    conn.execute('CREATE TABLE factor_data AS SELECT * FROM fund')
    conn.execute('CREATE TABLE price_data_daily AS SELECT * FROM fund')
    members = fund.loc[(fund.val_market_cap % 3)!=0,['date','stock_code']].assign(index_code='I')
    conn.execute('CREATE TABLE index_constituents AS SELECT * FROM members')


def scenarios():
    return {
        'all': ([], None, False, None),
        'state': ([StockFilter('is_st','==',0),StockFilter('paused','==',0),StockFilter('close','<','high_limit')],None,False,None),
        'fundamental': ([StockFilter('is_new_stock','==',0),StockFilter('val_market_cap','between',(2,5))],'val_market_cap',True,3),
        'percentile': ([StockFilter('index_code','==','I'),StockFilter('val_market_cap','percentile_between',(20,80))],'val_market_cap',False,2),
        'empty': ([StockFilter('val_market_cap','>',100)],None,False,None),
    }


@pytest.fixture
def pool():
    with duckdb.connect() as conn:
        seed_pool(conn)
        yield DynamicStockPoolManager(conn,price_table='price_data_daily',data_bindings={})


# Recorded before the refactor; includes a short week and partial boundary weeks.
WEEKLY_EXPECTED = {'all:2026-01-05:2026-01-30': {'2026-01-07': ['A', 'B', 'C', 'D', 'E', 'F'],
                               '2026-01-14': ['A', 'B', 'C', 'D', 'E', 'F'],
                               '2026-01-21': ['A', 'B', 'C', 'D', 'E', 'F'],
                               '2026-01-28': ['A', 'B', 'C', 'D', 'E', 'F']},
 'all:2026-01-08:2026-01-20': {'2026-01-09': ['A', 'B', 'C', 'D', 'E', 'F'],
                               '2026-01-14': ['A', 'B', 'C', 'D', 'E', 'F'],
                               '2026-01-20': ['A', 'B', 'C', 'D', 'E', 'F']},
 'state:2026-01-05:2026-01-30': {'2026-01-07': ['A', 'C', 'D', 'E'],
                                 '2026-01-14': ['A', 'B', 'C', 'E'],
                                 '2026-01-21': ['B', 'C', 'D', 'E'],
                                 '2026-01-28': ['A', 'C', 'D', 'E']},
 'state:2026-01-08:2026-01-20': {'2026-01-09': ['A', 'B', 'C', 'D', 'E'],
                                 '2026-01-14': ['A', 'B', 'C', 'E'],
                                 '2026-01-20': ['A', 'C', 'D', 'E']},
 'fundamental:2026-01-05:2026-01-30': {'2026-01-07': ['B', 'C', 'D'],
                                       '2026-01-14': ['C', 'D', 'E'],
                                       '2026-01-21': ['A', 'B', 'C'],
                                       '2026-01-28': ['A', 'B', 'C']},
 'fundamental:2026-01-08:2026-01-20': {'2026-01-09': ['A', 'B', 'C'],
                                       '2026-01-14': ['C', 'D', 'E'],
                                       '2026-01-20': ['A', 'B', 'C']},
 'percentile:2026-01-05:2026-01-30': {'2026-01-07': ['D', 'C'],
                                      '2026-01-14': ['E', 'D'],
                                      '2026-01-21': ['B', 'A'],
                                      '2026-01-28': ['C', 'B']},
 'percentile:2026-01-08:2026-01-20': {'2026-01-09': ['B', 'A'],
                                      '2026-01-14': ['E', 'D'],
                                      '2026-01-20': ['C', 'B']},
 'empty:2026-01-05:2026-01-30': {'2026-01-07': [], '2026-01-14': [], '2026-01-21': [], '2026-01-28': []},
 'empty:2026-01-08:2026-01-20': {'2026-01-09': [], '2026-01-14': [], '2026-01-20': []}}


@pytest.mark.parametrize('start,end', [('2026-01-05','2026-01-30'),('2026-01-08','2026-01-20')])
@pytest.mark.parametrize('case', list(scenarios()))
def test_existing_weekly_order_and_dates_are_unchanged(pool, start, end, case):
    filters,sort_by,ascending,limit=scenarios()[case]
    actual=pool.get_stock_pools_for_period(start,end,filters,sort_by=sort_by,ascending=ascending,
                                          max_stocks=limit,rebalance_schedule='weekly')
    assert actual == WEEKLY_EXPECTED[f'{case}:{start}:{end}']


def dates(pool, start, end, spec):
    return list(pool.get_stock_pools_for_period(start,end,[],schedule=spec))


def test_complete_week_uses_full_period_then_adds_previous_effective_pool(pool):
    assert dates(pool,'2026-01-08','2026-01-20',{'unit':'week'}) == ['2026-01-07','2026-01-14']
    assert dates(pool,'2026-01-08','2026-01-20',{'unit':'week','include_previous':False}) == ['2026-01-14']


def test_monthly_and_quarterly_use_actual_dates(pool):
    assert dates(pool,'2026-01-01','2026-04-30',{'unit':'month','include_previous':False}) == [
        '2026-01-05','2026-02-02','2026-03-02','2026-04-01']
    assert dates(pool,'2026-01-01','2026-04-30',{'unit':'month','interval':3,'anchor':'2026-01-01','include_previous':False}) == ['2026-01-05','2026-04-01']
    assert dates(pool,'2026-01-01','2026-02-28',{'unit':'month','session':-1,'include_previous':False}) == ['2026-01-30','2026-02-27']


def test_monthly_query_in_middle_keeps_the_effective_pool(pool):
    assert dates(pool,'2026-02-10','2026-03-10',{'unit':'month'}) == ['2026-02-02','2026-03-02']


def test_interval_anchor_independent_of_query_start_and_year(pool):
    spec={'unit':'week','interval':2,'anchor':'2025-12-01','include_previous':False}
    full=dates(pool,'2025-12-01','2026-02-27',spec)
    partial=dates(pool,'2026-01-08','2026-02-27',spec)
    assert partial == [d for d in full if d >= '2026-01-08']


def test_frequency_does_not_change_selection_on_the_same_date(pool):
    filters,sort,asc,limit=scenarios()['percentile']
    kwargs=dict(filters=filters,sort_by=sort,ascending=asc,max_stocks=limit)
    weekly=pool.get_stock_pools_for_period('2026-01-01','2026-01-31',schedule={'unit':'week','include_previous':False},**kwargs)
    monthly=pool.get_stock_pools_for_period('2026-01-01','2026-01-31',schedule={'unit':'month','session':8,'include_previous':False},**kwargs)
    assert monthly == {'2026-01-14':weekly['2026-01-14']}


@pytest.mark.parametrize('spec', [
    {'unit':'day'}, {'unit':'week','interval':0}, {'unit':'week','interval':True},
    {'unit':'month','session':0}, {'unit':'week','interval':2},
    {'unit':'month','months':[0]}, {'unit':'week','unexpected':1},
    {'unit':'week','anchor':'NaT'}, {'unit':'week','boundary':'query_window','include_previous':True},
])
def test_invalid_schedules_fail(pool,spec):
    with pytest.raises(ValueError): dates(pool,'2026-01-01','2026-01-31',spec)


def test_short_week_policies(pool):
    spec={'unit':'week','session':4,'include_previous':False}
    assert dates(pool,'2026-01-12','2026-01-18',spec) == ['2026-01-14']
    assert dates(pool,'2026-01-12','2026-01-18',dict(spec,short_period='skip')) == []
    with pytest.raises(ValueError,match='fewer'):
        dates(pool,'2026-01-12','2026-01-18',dict(spec,short_period='raise'))


def test_explicit_schedule_requires_full_period_coverage():
    from utils.stock_pool_schedule import generate_rebalance_dates, resolve_schedule, load_pool_policy
    from utils.trading_calendar import TradingCalendar, CalendarCoverageError
    cal=TradingCalendar(['2026-01-07','2026-01-08','2026-01-09'],'2026-01-07','2026-01-09')
    spec=resolve_schedule(load_pool_policy(),{'unit':'week','include_previous':False})
    with pytest.raises(CalendarCoverageError): generate_rebalance_dates(cal,'2026-01-07','2026-01-09',spec)


@pytest.mark.parametrize('session', [4, -4])
def test_previous_effective_pool_cannot_cross_a_rejected_short_week(pool, session):
    spec = {'unit': 'week', 'session': session, 'short_period': 'raise'}
    with pytest.raises(ValueError, match='2026-01-12/2026-01-18.*fewer'):
        dates(pool, '2026-01-19', '2026-01-19', spec)


@pytest.mark.parametrize('policy, expected', [
    ('last_session', ['2026-01-14']),
    ('skip', ['2026-01-08']),
])
def test_previous_effective_pool_keeps_other_short_week_policies(pool, policy, expected):
    spec = {'unit': 'week', 'session': 4, 'short_period': policy}
    assert dates(pool, '2026-01-19', '2026-01-20', spec) == expected


@pytest.mark.parametrize('start,end,overrides,expected', [
    ('2026-01-19', '2026-01-20', {'include_previous': False}, []),
    ('2026-01-26', '2026-01-27', {}, ['2026-01-22']),
    ('2026-01-22', '2026-01-22', {}, ['2026-01-22']),
    ('2026-01-19', '2026-01-20', {'interval': 2, 'anchor': '2026-01-05'}, ['2026-01-08']),
])
def test_unrelated_short_weeks_do_not_block_effective_pool(pool, start, end, overrides, expected):
    spec = {'unit': 'week', 'session': 4, 'short_period': 'raise', **overrides}
    assert dates(pool, start, end, spec) == expected


@pytest.mark.parametrize('unit,start,end,closed,session,query', [
    ('week', '2026-01-12', '2026-01-25', ['2026-01-15', '2026-01-16'], 4, '2026-01-19'),
    ('month', '2026-01-01', '2026-03-31', ['2026-02-02'], 20, '2026-03-02'),
])
def test_previous_short_period_is_rejected_for_week_and_month(unit, start, end, closed, session, query):
    from utils.stock_pool_schedule import generate_rebalance_dates, resolve_schedule, load_pool_policy
    from utils.trading_calendar import TradingCalendar
    # Synthetic sessions only; the weekly case has no earlier valid pool.
    sessions = pd.bdate_range(start, end).difference(pd.to_datetime(closed))
    calendar = TradingCalendar(sessions, start, end)
    spec = resolve_schedule(load_pool_policy(), {
        'unit': unit, 'session': session, 'short_period': 'raise',
    })
    with pytest.raises(ValueError, match='fewer'):
        generate_rebalance_dates(calendar, query, query, spec)


@pytest.mark.parametrize('condition', [
    StockFilter('absent','>',1), StockFilter('val_market_cap','between',(5,1)),
    StockFilter('val_market_cap','percentile_between',(20,101)),
    StockFilter('val_market_cap','percentile_bottom',float('nan')),
    StockFilter('val_market_cap','>',float('inf')), StockFilter('val_market_cap','unsupported',1),
])
def test_bad_selection_never_releases_unfiltered_pool(pool,condition):
    with pytest.raises(ValueError): pool._select_stock_pool(datetime(2026,1,14),[condition])
    assert pool._pool_cache == {}


@pytest.mark.parametrize('operator', ['in', 'not_in'])
@pytest.mark.parametrize('value', ['AB', b'AB', {'AB': 1}, None, 2, [], ()])
def test_membership_requires_nonempty_explicit_collection(pool, operator, value, monkeypatch):
    condition = StockFilter('stock_code', operator, value)
    with pytest.raises(ValueError, match='nonempty'):
        condition.to_sql_condition()
    def unexpected_query(sql):
        pytest.fail('Invalid membership reached a data query')
    monkeypatch.setattr(pool, '_execute_sql', unexpected_query)
    with pytest.raises(ValueError, match='nonempty'):
        pool._select_stock_pool(datetime(2026, 1, 14), [condition])


@pytest.mark.parametrize('operator', ['==', '!='])
@pytest.mark.parametrize('value', [2, -1, 'yes', '2', None, [], {'false': 0}, float('nan'),
    0.0, 1.0, np.float32(0), np.float64(1)])
def test_invalid_boolean_comparisons_fail_before_query(pool, operator, value, monkeypatch):
    condition = StockFilter('paused', operator, value)
    with pytest.raises(ValueError, match='Boolean'):
        condition.to_sql_condition()
    def unexpected_query(sql):
        pytest.fail('Invalid boolean reached a data query')
    monkeypatch.setattr(pool, '_execute_sql', unexpected_query)
    with pytest.raises(ValueError):
        pool._select_stock_pool(datetime(2026, 1, 14), [condition])


@pytest.mark.parametrize('operator', ['==', '!='])
@pytest.mark.parametrize('value,truth', [(0, False), ('0', False), (False, False),
    ('false', False), ('False', False), (np.int64(0), False), (1, True), ('1', True),
    (True, True), ('true', True), ('True', True), (np.bool_(True), True)])
def test_boolean_compatibility_preserves_recognized_rows(pool, operator, value, truth):
    wanted = truth if operator == '==' else not truth
    condition = StockFilter('paused', operator, value)
    expected = pool._duckdb_conn.execute(
        "SELECT stock_code FROM factor_data WHERE date='2026-01-14' AND paused=? ORDER BY stock_code",
        [int(wanted)]).fetchall()
    assert pool._select_stock_pool(datetime(2026, 1, 14), [condition]) == [row[0] for row in expected]
    # Also preserve the imported string representations, including NULL exclusion.
    rows = pool._duckdb_conn.execute(
        "SELECT paused FROM (VALUES ('0'),('1'),('false'),('true'),('False'),('True'),(NULL)) t(paused) WHERE "
        + condition.to_sql_condition()).fetchall()
    assert {row[0] for row in rows} == ({'1', 'true', 'True'} if wanted else {'0', 'false', 'False'})


@pytest.mark.parametrize('container', [list, tuple, set, frozenset])
@pytest.mark.parametrize('operator,expected', [('in', ['AB']), ('not_in', ['A', 'B'])])
def test_membership_preserves_whole_codes(pool, container, operator, expected):
    condition = StockFilter('stock_code', operator, container(['AB']))
    rows = pool._duckdb_conn.execute(
        "SELECT stock_code FROM (VALUES ('A'),('B'),('AB')) t(stock_code) WHERE "
        + condition.to_sql_condition() + ' ORDER BY stock_code').fetchall()
    assert [row[0] for row in rows] == expected


@pytest.mark.parametrize('operator', ['in', 'not_in', 'like', 'not_like'])
def test_string_literals_preserve_quotes_and_like_patterns(operator):
    values = ["O'Reilly", "O'Reader", 'Other', None]
    value = ["O'Reilly"] if operator in {'in', 'not_in'} else "O'Re%"
    expected = {'in': ["O'Reilly"], 'not_in': ["O'Reader", 'Other'],
                'like': ["O'Reader", "O'Reilly"], 'not_like': ['Other']}[operator]
    with duckdb.connect() as conn:
        conn.execute('CREATE TABLE labels(label VARCHAR)')
        conn.executemany('INSERT INTO labels VALUES (?)', [(v,) for v in values])
        condition = StockFilter('label', operator, value)
        rows = conn.execute('SELECT label FROM labels WHERE ' + condition.to_sql_condition() + ' ORDER BY label').fetchall()
    assert [row[0] for row in rows] == expected


@pytest.mark.parametrize('operator', ['starts_with', 'not_starts_with'])
@pytest.mark.parametrize('prefix', ['A_', 'A%', 'A!', "O'", 'A\\', ['A_', "O'"], ['A%', 'A!'], ''])
def test_prefixes_match_literal_text(operator, prefix):
    values = ['A_1', 'AB1', 'A%1', 'A!1', 'A\\1', "O'Reilly", 'Other', '', None]
    prefixes = prefix if isinstance(prefix, list) else [prefix]
    expected = sorted(v for v in values if v is not None and
                      (v.startswith(tuple(prefixes)) == (operator == 'starts_with')))
    with duckdb.connect() as conn:
        conn.execute('CREATE TABLE labels(label VARCHAR)')
        conn.executemany('INSERT INTO labels VALUES (?)', [(v,) for v in values])
        condition = StockFilter('label', operator, prefix)
        rows = conn.execute('SELECT label FROM labels WHERE ' + condition.to_sql_condition() + ' ORDER BY label').fetchall()
    assert [row[0] for row in rows] == expected


@pytest.mark.parametrize('operator', ['starts_with', 'not_starts_with'])
@pytest.mark.parametrize('prefix', [[], None, 3, ['A', None], ('A', 'B')])
def test_invalid_prefixes_fail_before_query(pool, operator, prefix, monkeypatch):
    def unexpected_query(sql):
        pytest.fail('Invalid prefix reached a data query')
    monkeypatch.setattr(pool, '_execute_sql', unexpected_query)
    with pytest.raises(ValueError, match='prefix'):
        pool._select_stock_pool(datetime(2026, 1, 14), [StockFilter('stock_code', operator, prefix)])


def test_quoted_stock_codes_survive_filter_percentile_and_sort():
    with duckdb.connect() as conn:
        conn.execute('CREATE TABLE factor_data(date DATE, stock_code VARCHAR, val_rank DOUBLE)')
        for day in ['2026-01-13', '2026-01-14']:
            conn.executemany('INSERT INTO factor_data VALUES (?, ?, ?)',
                             [(day, "O'Reilly", 1.), (day, 'A', 2.), (day, 'B', 3.)])
        pool = DynamicStockPoolManager(conn, data_bindings={})
        result = pool._select_stock_pool(datetime(2026, 1, 14), [
            StockFilter('stock_code', 'in', ["O'Reilly", 'A']),
            StockFilter('val_rank', 'percentile_bottom', 100)], sort_by='val_rank', ascending=True)
        assert result == ["O'Reilly", 'A']


def test_query_failure_propagates_and_retry_recovers(pool,monkeypatch):
    original=pool._execute_sql
    def fail(sql):
        if 'AND val_market_cap' in sql: raise duckdb.InvalidInputException('broken query')
        return original(sql)
    monkeypatch.setattr(pool,'_execute_sql',fail)
    filters=[StockFilter('val_market_cap','>',2)]
    with pytest.raises(duckdb.InvalidInputException,match='broken query'):
        pool._select_stock_pool(datetime(2026,1,14),filters)
    assert not pool._pool_cache
    monkeypatch.setattr(pool,'_execute_sql',original)
    assert pool._select_stock_pool(datetime(2026,1,14),filters)


def test_price_only_sort_uses_the_price_table(pool):
    pool._duckdb_conn.execute('ALTER TABLE factor_data DROP COLUMN volume')
    actual=pool._select_stock_pool(datetime(2026,1,14),[],sort_by='volume',ascending=False,max_stocks=2)
    assert actual == ['F','E']


def test_unified_diagnostics_preserve_selection_and_explain_sources(pool, tmp_path, monkeypatch):
    filters = [StockFilter('close', '<', 'high_limit'),
               StockFilter('val_market_cap', 'percentile_bottom', 50)]
    date = datetime(2026, 1, 14)
    monkeypatch.setenv('QUANT_DEBUG_OUTPUTS', '0')
    expected = pool._select_stock_pool(date, filters)
    pool.clear_cache()
    monkeypatch.setenv('QUANT_DEBUG_OUTPUTS', '1')
    monkeypatch.setenv('QUANT_DIAGNOSTICS_DIR', str(tmp_path))
    assert pool._select_stock_pool(date, filters) == expected
    log = (tmp_path / 'pool_filter_log.txt').read_text()
    assert 'TRADING_DATE: 2026-01-14 (FUNDAMENTAL_DATE: 2026-01-13)' in log
    assert 'FILTER: open < high_limit [state@2026-01-14]' in log
    assert 'ORIGINAL_FILTER: close < high_limit' in log
    assert 'TABLE: factor_data' in log
    assert 'THRESHOLD: 3.0 LOW: None HIGH: None' in log
    assert f'FINAL_STOCKS: {len(expected)}' in log


def test_diagnostic_removals_are_explicitly_samples(pool, tmp_path, monkeypatch):
    monkeypatch.setenv('QUANT_DEBUG_OUTPUTS', '1')
    monkeypatch.setenv('QUANT_DIAGNOSTICS_DIR', str(tmp_path))
    pool._save_filter_log({
        'trading_date': '2026-01-14', 'fundamental_date': '2026-01-09',
        'initial_count': 30, 'final_count': 5,
        'filter_steps': [{'filter': 'paused == 0', 'original_filter': 'paused == 0',
            'table_used': 'factor_data', 'type': 'state', 'date_used': '2026-01-14',
            'before': 30, 'after': 5, 'removed': 25,
            'removed_samples': [f'S{i:02}' for i in range(20)]}],
    })
    log = (tmp_path / 'pool_filter_log.txt').read_text()
    assert 'REMOVED_SAMPLES (20 of 25): S00,S01,' in log
    assert 'FUNDAMENTAL_DATE: 2026-01-09' in log
    assert 'T-1:' not in log


def test_diagnostic_write_failure_does_not_change_selection(pool, monkeypatch, caplog):
    date = datetime(2026, 1, 14)
    monkeypatch.setenv('QUANT_DEBUG_OUTPUTS', '0')
    expected = pool._select_stock_pool(date, [])
    pool.clear_cache()
    monkeypatch.setenv('QUANT_DEBUG_OUTPUTS', '1')
    def fail(*args, **kwargs):
        raise OSError('diagnostics unavailable')
    monkeypatch.setattr('utils.output_paths.open_diagnostic', fail)
    assert pool._select_stock_pool(date, []) == expected
    assert 'Failed to save filter log: diagnostics unavailable' in caplog.text


def test_missing_percentile_values_raise(pool):
    pool._duckdb_conn.execute('UPDATE factor_data SET val_market_cap=NULL')
    with pytest.raises(ValueError,match='No values'):
        pool._select_stock_pool(datetime(2026,1,14),[StockFilter('val_market_cap','percentile_bottom',20)])


def test_month_lag_is_not_silently_used_as_a_frequency(pool):
    with pytest.raises(ValueError,match='lag is retired'):
        pool.get_stock_pools_for_period('2026-01-01','2026-01-31',[],rebalance_schedule='monthly',lag_months=1)


@pytest.mark.parametrize('date_type', ['DATE', 'TIMESTAMP'])
def test_previous_snapshot_returns_iso_date_and_selects_same_rows(date_type):
    with duckdb.connect() as conn:
        conn.execute(f'CREATE TABLE factor_data(date {date_type}, stock_code VARCHAR, val_rank DOUBLE)')
        manager = DynamicStockPoolManager(conn, data_bindings={})
        assert manager._get_previous_data_date(datetime(2026, 1, 14)) is None
        conn.executemany('INSERT INTO factor_data VALUES (?, ?, ?)', [
            ('2026-01-12', 'OLD', 1.), ('2026-01-13', 'A', 2.),
            ('2026-01-13', 'B', 3.), ('2026-01-14', 'TODAY', 1.)])
        assert manager._get_previous_data_date(datetime(2026, 1, 14)) == '2026-01-13'
        assert manager._get_previous_data_date('2026-01-14') == '2026-01-13'
        assert manager._select_stock_pool(datetime(2026, 1, 14), [
            StockFilter('val_rank', '>=', 2)], sort_by='val_rank', ascending=True) == ['A', 'B']


def test_policy_can_reject_same_day_fundamental_fallback(tmp_path):
    from utils.stock_pool_schedule import load_pool_policy
    spec=load_pool_policy();spec['selection']['missing_previous']='raise'
    path=tmp_path/'policy.json';path.write_text(json.dumps(spec))
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2026-01-14' date,'A' stock_code,1 val_market_cap")
        manager=DynamicStockPoolManager(conn,data_bindings={},pool_config_path=path)
        with pytest.raises(ValueError,match='No fundamental snapshot'):
            manager._select_stock_pool(datetime(2026,1,14),[])


def test_factor_base_passes_explicit_schedule(pool):
    from utils.factor_base import FactorBase
    factor=FactorBase(pool._duckdb_conn,'ScheduleProbe','2026-01-01','2026-02-28',
        use_dynamic_pool=True,stock_filters=[StockFilter('index_code','==','I')],
        price_table='price_data_daily',data_bindings={},pit_enabled=False,
        pool_schedule={'unit':'month','include_previous':False})
    assert list(factor.stock_pools_by_period) == ['2026-01-05','2026-02-02']


def test_synthesis_config_passes_explicit_schedule(pool):
    from factor_synthesis.core.pool_manager import compute_dynamic_pools_for_topk, _create_pool_manager
    _create_pool_manager.cache_clear()
    config={'stock_pool':{'use_dynamic_pools':True,'schedule':{'unit':'month','include_previous':False},
            'filters':[{'field':'index_code','operator':'==','value':'I'}]}}
    # Keep the public call below explicit so configuration forwarding is exercised.
    result=compute_dynamic_pools_for_topk(pool._duckdb_conn,config,'factor_data','price_data_daily','duckdb',
                                         '2026-01-01','2026-02-28')
    assert list(result) == ['2026-01-05','2026-02-02']
    _create_pool_manager.cache_clear()


def make_split_source_pool(conn, *, fund_open=False, fund_limit=True, original_close=True):
    """Different values expose a source switch even when all filters pass."""
    fund = pd.DataFrame([
        {'date': pd.Timestamp(day), 'stock_code': code, 'volume': volume,
         **({'close': 99.0} if original_close else {}),
         **({'open': opening} if fund_open else {}),
         **({'high_limit': 15.0} if fund_limit else {})}
        for day in ['2026-01-13', '2026-01-14']
        for code, volume, opening in [('A', 1., 30.), ('B', 2., 5.)]
    ])
    prices = pd.DataFrame([
        {'date': pd.Timestamp('2026-01-14'), 'stock_code': code,
         'open': opening, 'volume': volume, 'high_limit': 15., 'price_rank': rank}
        for code, opening, volume, rank in [('A', 10., 20., 1.), ('B', 20., 10., 2.)]
    ])
    conn.execute('CREATE TABLE factor_data AS SELECT * FROM fund')
    conn.execute('CREATE TABLE price_data_daily AS SELECT * FROM prices')
    return DynamicStockPoolManager(conn, price_table='price_data_daily', data_bindings={})


@pytest.mark.parametrize('options,expected', [
    ({}, ['A']),
    ({'original_close': False}, ['A']),
    ({'fund_open': True}, ['B']),
    ({'fund_open': True, 'fund_limit': False}, ['A']),
])
def test_state_alias_resolves_complete_condition_before_source_selection(options, expected):
    with duckdb.connect() as conn:
        manager = make_split_source_pool(conn, **options)
        condition = StockFilter('close', '<', 'high_limit')
        assert manager.validate_filters([condition])
        assert manager._select_stock_pool(datetime(2026, 1, 14), [condition]) == expected
        assert condition.field == 'close' and condition.value == 'high_limit'


def test_missing_rhs_field_fails_validation_and_selection_even_with_empty_candidates():
    with duckdb.connect() as conn:
        manager = make_split_source_pool(conn)
        filters = [StockFilter('volume', '>', 1000), StockFilter('close', '<', 'absent_limit')]
        assert not manager.validate_filters(filters)
        with pytest.raises(ValueError, match='absent_limit'):
            manager._select_stock_pool(datetime(2026, 1, 14), filters)
        assert not manager._pool_cache


@pytest.mark.parametrize('extra,sort_by', [
    ([], None),
    ([StockFilter('high_limit', '>', 0)], None),
    ([], 'price_rank'),
])
def test_percentile_source_does_not_depend_on_other_filters_or_sort(extra, sort_by):
    with duckdb.connect() as conn:
        manager = make_split_source_pool(conn, fund_limit=False)
        filters = [StockFilter('volume', 'percentile_bottom', 50), *extra]
        assert manager._select_stock_pool(datetime(2026, 1, 14), filters, sort_by=sort_by) == ['B']


def test_percentile_recomputes_with_smaller_candidates_but_keeps_source():
    with duckdb.connect() as conn:
        manager = make_split_source_pool(conn)
        filters = [StockFilter('volume', 'percentile_bottom', 50), StockFilter('stock_code', 'in', ['A'])]
        assert manager._select_stock_pool(datetime(2026, 1, 14), filters) == ['A']


def test_source_plan_logs_actual_fields_table_and_date(monkeypatch):
    with duckdb.connect() as conn:
        manager = make_split_source_pool(conn)
        logs = []
        monkeypatch.setattr(manager, '_save_filter_log', logs.append)
        manager._select_stock_pool(datetime(2026, 1, 14), [StockFilter('close', '<', 'high_limit')])
        step = logs[-1]['filter_steps'][0]
        assert step['filter'] == 'open < high_limit'
        assert step['original_filter'] == 'close < high_limit'
        assert step['table_used'] == 'price_data_daily'
        assert step['date_used'] == '2026-01-14'


def test_missing_rows_do_not_trigger_source_fallback():
    with duckdb.connect() as conn:
        manager = make_split_source_pool(conn)
        conn.execute('DELETE FROM price_data_daily')
        with pytest.raises(ValueError, match='No values'):
            manager._select_stock_pool(datetime(2026, 1, 14), [StockFilter('volume', 'percentile_bottom', 50)])


def test_percentile_falls_back_only_when_field_is_absent_from_price_schema():
    with duckdb.connect() as conn:
        manager = make_split_source_pool(conn)
        conn.execute('ALTER TABLE price_data_daily DROP COLUMN volume')
        assert manager._select_stock_pool(datetime(2026, 1, 14), [StockFilter('volume', 'percentile_bottom', 50)]) == ['A']


def test_explicit_source_priority_is_stable_and_part_of_cache_identity(tmp_path):
    from utils.stock_pool_schedule import load_pool_policy
    policy = load_pool_policy()
    policy['selection']['source_priority']['percentile'] = ['fundamental', 'price']
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy))
    with duckdb.connect() as conn:
        make_split_source_pool(conn)
        manager = DynamicStockPoolManager(conn, price_table='price_data_daily', data_bindings={}, pool_config_path=path)
        # Keep a fixed revision to exercise the selection cache rather than its invalidation guard.
        manager.cache_revision = lambda: ('fixed-source',)
        filters = [StockFilter('volume', 'percentile_bottom', 50)]
        assert manager._select_stock_pool(datetime(2026, 1, 14), filters, sort_by='price_rank') == ['A']
        manager.pool_policy['selection']['source_priority']['percentile'] = ['price', 'fundamental']
        assert manager._select_stock_pool(datetime(2026, 1, 14), filters, sort_by='price_rank') == ['B']


def test_older_policy_file_inherits_source_rules_without_rewriting_it(tmp_path):
    from utils.stock_pool_schedule import load_pool_policy
    policy = load_pool_policy()
    expected = policy['selection'].pop('source_priority')
    path = tmp_path / 'old_policy.json'
    path.write_text(json.dumps(policy))
    before = path.read_bytes()
    assert load_pool_policy(path)['selection']['source_priority'] == expected
    assert path.read_bytes() == before


@pytest.mark.parametrize('priority', [None, {}, {'state': ['fundamental']},
    {'state': ['fundamental'], 'percentile': [], 'sort': ['fundamental']},
    {'state': ['fundamental'], 'percentile': ['price', 'price'], 'sort': ['fundamental']},
    {'state': ['fundamental'], 'percentile': ['unknown'], 'sort': ['fundamental']},
])
def test_invalid_source_policy_fails_before_reading_data(tmp_path, priority):
    from utils.stock_pool_schedule import load_pool_policy
    policy = load_pool_policy()
    policy['selection']['source_priority'] = priority
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='source_priority'):
        load_pool_policy(path)


def test_sql_source_failure_is_not_hidden_by_another_source(monkeypatch):
    with duckdb.connect() as conn:
        manager = make_split_source_pool(conn)
        original = manager._execute_sql
        def fail(sql):
            if 'DESCRIBE SELECT * FROM price_data_daily' in sql:
                raise duckdb.IOException('source offline')
            return original(sql)
        monkeypatch.setattr(manager, '_execute_sql', fail)
        with pytest.raises(RuntimeError, match='price_data_daily') as error:
            manager._select_stock_pool(datetime(2026, 1, 14), [StockFilter('volume', 'percentile_bottom', 50)])
        assert isinstance(error.value.__cause__, duckdb.IOException)
        assert not manager._pool_cache


@pytest.mark.parametrize('legacy', ['weekly', 'monthly', 'quarterly', 'report_based'])
@pytest.mark.parametrize('unit', ['wek', '', None, 1, [], {}])
def test_legacy_schedule_rejects_invalid_final_unit(legacy, unit):
    from utils.stock_pool_schedule import load_pool_policy, resolve_schedule
    policy = load_pool_policy()
    name = 'monthly' if legacy == 'quarterly' else legacy
    policy['schedules'][name]['unit'] = unit
    with pytest.raises(ValueError, match=r'schedule.unit'):
        resolve_schedule(policy, legacy=legacy)


def test_legacy_schedule_rejects_missing_unit_before_calendar_query(pool, monkeypatch):
    del pool.pool_policy['schedules']['weekly']['unit']
    def unexpected(): pytest.fail('Invalid schedule must fail before calendar/data queries')
    monkeypatch.setattr(pool, '_trading_calendar', unexpected)
    with pytest.raises(ValueError, match=r'schedule.unit'):
        pool.get_stock_pools_for_period('2026-01-01', '2026-01-31', [], rebalance_schedule='weekly')


@pytest.mark.parametrize('unit,template', [('week', 'weekly'), ('month', 'monthly')])
def test_explicit_schedule_validates_after_overriding_template(unit, template):
    from copy import deepcopy
    from utils.stock_pool_schedule import load_pool_policy, resolve_schedule
    policy = load_pool_policy()
    expected = resolve_schedule(policy, {'unit': unit})
    policy['schedules'][template]['unit'] = 'typo'
    before = deepcopy(policy)
    assert resolve_schedule(policy, {'unit': unit}) == expected
    assert policy == before
