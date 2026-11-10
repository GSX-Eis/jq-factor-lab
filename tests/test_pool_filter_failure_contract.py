"""Synthesis must not treat failed membership reads as an unrestricted universe."""
from calendar_test_support import seed_calendar
import duckdb
import pandas as pd
import pytest
from factor_synthesis.core import pool_manager as module


@pytest.fixture
def source():
    module._create_pool_manager.cache_clear()
    with duckdb.connect() as conn:
        seed_calendar(conn)
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2024-01-31' date, 'A' stock_code, 10. AS close")
        conn.execute('CREATE TABLE price_data AS SELECT * FROM factor_data')
        yield conn
    module._create_pool_manager.cache_clear()


def panel():
    return pd.DataFrame({'signal':[1., 2.]}, index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2024-02-01'), c) for c in ['A','B']],names=['date','stock_code']))


def apply(conn, config=None):
    config = config if config is not None else {'stock_pool':{'use_dynamic_pools':True,
        'rebalance_schedule':'monthly','filters':[{'field':'index_code','operator':'==','value':'I'}]}}
    return module.apply_dynamic_stock_pool_filter(panel(),conn,config,'factor_data','price_data','duckdb',
                                                  '2024-02-01','2024-02-29')


def test_missing_index_source_propagates(source):
    with pytest.raises(duckdb.CatalogException):
        apply(source)
    assert source.execute('SELECT 1').fetchone() == (1,)


@pytest.mark.parametrize('members', [[], ['A']])
def test_successful_membership_read_still_filters(source, members):
    source.execute('CREATE TABLE index_constituents(date DATE, stock_code VARCHAR, index_code VARCHAR)')
    for code in members:
        source.execute("INSERT INTO index_constituents VALUES ('2024-02-01', ?, 'I')", [code])
    result=apply(source)
    assert result.index.get_level_values('stock_code').tolist() == members


def test_empty_generated_mapping_does_not_release_original_panel(source, monkeypatch):
    monkeypatch.setattr(module.DynamicStockPoolManager,'get_stock_pools_for_period',lambda *a,**k:{})
    assert apply(source).empty


def test_disabled_filter_keeps_input(source):
    pd.testing.assert_frame_equal(apply(source,{}),panel())


def test_basic_filter_construction_error_propagates(source, monkeypatch):
    def fail(*a,**k): raise RuntimeError('basic filter failure')
    monkeypatch.setattr(module.DynamicStockPoolManager,'add_basic_filters',fail)
    with pytest.raises(RuntimeError,match='basic filter failure'):
        apply(source, {'stock_pool':{'use_dynamic_pools':True,'include_basic_filters':True}})


@pytest.mark.parametrize('entry', ['universe','predict','topk'])
def test_generation_error_propagates_through_all_synthesis_entries(source, monkeypatch, entry):
    def fail(*args, **kwargs):
        raise ValueError('invalid pool condition')
    monkeypatch.setattr(module.DynamicStockPoolManager, 'get_stock_pools_for_period', fail)
    config={'stock_pool':{'use_dynamic_pools':True,'filters':[]}}
    args=(source,config,'factor_data','price_data','duckdb')
    with pytest.raises(ValueError,match='invalid pool condition'):
        if entry=='universe': module.get_all_pool_stocks(*args,'2024-02-01','2024-02-29')
        elif entry=='predict': module.compute_predict_period_dynamic_pools(*args,pd.Timestamp('2024-01-31'),28)
        else: module.compute_dynamic_pools_for_topk(*args,'2024-02-01','2024-02-29')


@pytest.mark.parametrize('pools', [
    {'2026-2-01': ['A'], '2026-10-01': ['B']},
    {'2026-02': ['A'], '2026-10': ['B']},
    {'2026-02': ['A'], '2026-10-01': ['B']},
])
def test_long_filter_sorts_parsed_dates_and_preserves_month_keys(pools):
    from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_long_df
    frame = panel().rename(index={pd.Timestamp('2024-02-01'): pd.Timestamp('2026-03-01')})
    result = apply_date_level_pool_filter_to_long_df(frame, pools)
    pd.testing.assert_frame_equal(result, frame.iloc[:1])


@pytest.mark.parametrize('pools', [None, {}, {'2024-02-01': None},
    {'2024-02-01': 'A'}, {'2024-02-01': ['A', 'A']},
    {'2024-02-01': [1]}, {'2024-02-01': ['']}, {'NaT': ['A']},
    {'2024-02': ['A'], '2024-02-01': ['B']}])
def test_long_filter_rejects_ambiguous_or_missing_history(pools):
    from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_long_df
    with pytest.raises(ValueError):
        apply_date_level_pool_filter_to_long_df(panel(), pools)


@pytest.mark.parametrize('names', [['date', 'ticker'], ['day', 'stock_code']])
def test_long_filter_requires_both_named_index_levels(names):
    from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_long_df
    with pytest.raises(ValueError, match='MultiIndex'):
        apply_date_level_pool_filter_to_long_df(panel().rename_axis(names), {'2024-02-01': ['A']})


def test_long_filter_preserves_values_order_duplicates_and_explicit_empty_period():
    from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_long_df
    frame = pd.DataFrame({'signal': [4., 1., 2., 3., 5., 6.]}, index=pd.MultiIndex.from_tuples([
        ('2026-01-08', 'B'), ('2026-01-04', 'A'), ('2026-01-05', 'A'),
        ('2026-01-05', 'A'), ('2026-01-07', 'A'), ('2026-02-01', 'B'),
    ], names=['date', 'stock_code']))
    before = frame.copy()
    result = apply_date_level_pool_filter_to_long_df(frame, {
        '2026-01-08': ['B'], '2026-01-05': ['A'], '2026-01-07': [],
    })
    pd.testing.assert_frame_equal(result, frame.iloc[[0, 2, 3, 5]])
    pd.testing.assert_frame_equal(frame, before)
    assert apply_date_level_pool_filter_to_long_df(frame, {'2026-01-01': []}).empty


@pytest.mark.parametrize('date', [pd.NaT, pd.Timestamp('2024-02-01', tz='UTC')])
def test_long_filter_rejects_invalid_panel_dates(date):
    from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_long_df
    frame = panel()
    frame.index = pd.MultiIndex.from_tuples([(date, code) for code in ['A', 'B']],
                                          names=['date', 'stock_code'])
    with pytest.raises(ValueError, match='Panel dates'):
        apply_date_level_pool_filter_to_long_df(frame, {'2024-02-01': ['A']})


@pytest.mark.parametrize('start,end,expected', [
    (None, None, ('2026-03-02', '2026-03-06')),
    ('2026-02-01', None, ('2026-02-01', '2026-03-06')),
    (None, '2026-03-31', ('2026-03-02', '2026-03-31')),
    ('2026-02-01', '2026-03-31', ('2026-02-01', '2026-03-31')),
])
def test_filter_date_bounds_preserve_each_explicit_endpoint(monkeypatch, start, end, expected):
    frame = pd.DataFrame({'signal': [1., 2., 3.]}, index=pd.MultiIndex.from_tuples([
        ('2026-03-06', 'A'), ('2026-03-02', 'A'), ('2026-03-02', 'B'),
    ], names=['date', 'stock_code']))
    before = frame.copy()
    calls = []

    class Manager:
        def get_stock_pools_for_period(self, **kwargs):
            calls.append((kwargs['start_date'], kwargs['end_date']))
            return {'2026-02-01': ['A']}

    monkeypatch.setattr(module, '_create_pool_manager', lambda **kwargs: Manager())
    monkeypatch.setattr(module, 'build_stock_filters', lambda **kwargs: [])
    result = module.apply_dynamic_stock_pool_filter(
        frame, None, {'stock_pool': {'use_dynamic_pools': True}},
        'factor_data', 'price_data', 'duckdb', start, end,
    )
    assert calls == [expected]
    pd.testing.assert_frame_equal(result, frame.iloc[:2])
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('config', [
    None, [], {'filters': None}, {'filters': {}}, {'filters': 'bad'},
    {'filters': [None]}, {'filters': ['bad']}, {'filters': [{}]},
    {'filters': [{'field': 'market_cap', 'value': 1.5}]},
    {'filters': [{'operator': '<', 'value': 1.5}]},
    {'filters': [{'field': '', 'operator': '<', 'value': 1.5}]},
    {'filters': [{'field': ' ', 'operator': '<', 'value': 1.5}]},
    {'filters': [{'field': 1, 'operator': '<', 'value': 1.5}]},
    {'filters': [{'field': 'market_cap', 'operator': None, 'value': 1.5}]},
    {'filters': [{'field': 'market_cap', 'operator': '', 'value': 1.5}]},
    {'filters': [{'field': 'market_cap', 'operator': ' ', 'value': 1.5}]},
    {'filters': [{'field': 'market_cap', 'operator': 1, 'value': 1.5}]},
])
def test_filter_builder_rejects_malformed_config_before_manager_creation(config, monkeypatch):
    def unexpected_manager(*args, **kwargs):
        pytest.fail('Malformed custom filters must fail before initializing a manager')
    monkeypatch.setattr(module, 'DynamicStockPoolManager', unexpected_manager)
    if isinstance(config, dict):
        config = {**config, 'include_basic_filters': True}
    with pytest.raises(ValueError, match='stock_pool'):
        module.build_stock_filters(config, None, 'factor_data', 'price_data', 'duckdb')


@pytest.mark.parametrize('include_basic,enabled', [(True, True), (True, False), (False, True)])
def test_filter_builder_preserves_order_and_basic_switches(include_basic, enabled, monkeypatch):
    from copy import deepcopy
    config = {'include_basic_filters': enabled, 'filters': [
        {'field': 'market_cap', 'operator': 'percentile_bottom', 'value': 20, 'description': 'first'},
        {'field': 'close', 'operator': 'percentile_top', 'value': 30, 'description': 'second'},
    ]}
    before = deepcopy(config)
    calls = []
    class Manager:
        def __init__(self, *args, **kwargs): calls.append(1)
        def add_basic_filters(self): return [module.StockFilter('is_st', '==', 0)]
    monkeypatch.setattr(module, 'DynamicStockPoolManager', Manager)
    filters = module.build_stock_filters(config, None, 'factor_data', 'price_data', 'duckdb', include_basic=include_basic)
    expected = [('market_cap', 'percentile_bottom', 20), ('close', 'percentile_top', 30)]
    if include_basic and enabled: expected.insert(0, ('is_st', '==', 0))
    assert [(f.field, f.operator, f.value) for f in filters] == expected
    assert [f.description for f in filters[-2:]] == ['first', 'second']
    assert len(calls) == int(include_basic and enabled)
    assert config == before


def test_missing_custom_filter_cannot_expand_selected_universe(source):
    with pytest.raises(ValueError, match='operator'):
        apply(source, {'stock_pool': {'use_dynamic_pools': True, 'rebalance_schedule': 'monthly',
            'filters': [{'field': 'close', 'value': 1.5}]}})


@pytest.mark.parametrize('skip_rules', [set(), {1}, {'01-16:02-01'}])
def test_universe_keeps_cross_month_members_before_calendar_filter(monkeypatch, skip_rules):
    pools = {'2026-01-28': ['A'], '2026-02-05': ['B']}
    class Manager:
        def get_stock_pools_for_period(self, **kwargs): return pools
    monkeypatch.setattr(module, '_create_pool_manager', lambda **kwargs: Manager())
    monkeypatch.setattr(module, 'build_stock_filters', lambda **kwargs: [])
    config = {'stock_pool': {'use_dynamic_pools': True},
              'calendar_filters': {'skip_months': list(skip_rules)}}
    stocks = module.get_all_pool_stocks(None, config, 'factor_data', 'price_data', 'duckdb',
                                       '2026-01-28', '2026-02-06')
    assert stocks == {'A', 'B'}
    frame = pd.DataFrame({'signal': [1., 2., 3.]}, index=pd.MultiIndex.from_tuples([
        (pd.Timestamp('2026-01-29'), 'A'), (pd.Timestamp('2026-02-02'), 'A'),
        (pd.Timestamp('2026-02-06'), 'B'),
    ], names=['date', 'stock_code']))
    from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_long_df
    members = apply_date_level_pool_filter_to_long_df(frame, pools)
    pd.testing.assert_frame_equal(members, frame)
    result = module.apply_skip_months(members, skip_rules)
    pd.testing.assert_frame_equal(result, frame if not skip_rules else frame.iloc[1:])


@pytest.mark.parametrize('enabled,pools,expected', [
    (False, {}, None), (True, {}, None), (True, {'2026-01-05': []}, set()),
])
def test_universe_cleanup_preserves_disabled_and_empty_contracts(monkeypatch, enabled, pools, expected):
    class Manager:
        def get_stock_pools_for_period(self, **kwargs): return pools
    monkeypatch.setattr(module, '_create_pool_manager', lambda **kwargs: Manager())
    monkeypatch.setattr(module, 'build_stock_filters', lambda **kwargs: [])
    result = module.get_all_pool_stocks(None, {'stock_pool': {'use_dynamic_pools': enabled}},
                                       'factor_data', 'price_data', 'duckdb', '2026-01-01', '2026-02-01')
    assert result == expected


@pytest.mark.parametrize('filters', [None, [{'field': 'market_cap', 'value': 1.5}],
                                      [{'operator': '<', 'value': 1.5}]])
def test_prediction_pool_rejects_malformed_filters_before_query(monkeypatch, filters):
    def unexpected(**kwargs): pytest.fail('Invalid filters must fail before getting a manager')
    monkeypatch.setattr(module, '_create_pool_manager', unexpected)
    with pytest.raises(ValueError, match='stock_pool.filters'):
        module.compute_predict_period_dynamic_pools(None,
            {'stock_pool': {'use_dynamic_pools': True, 'filters': filters}},
            'factor_data', 'price_data', 'duckdb', pd.Timestamp('2026-01-31'), 28)


def test_prediction_pool_rejects_missing_effective_periods(monkeypatch):
    from calendar_test_support import synthetic_calendar
    class Manager:
        def _trading_calendar(self): return synthetic_calendar()
        def get_stock_pools_for_period(self, **kwargs): return {}
    monkeypatch.setattr(module, '_create_pool_manager', lambda **kwargs: Manager())
    with pytest.raises(ValueError, match='2026-02-01.*2026-03-01'):
        module.compute_predict_period_dynamic_pools(None,
            {'stock_pool': {'use_dynamic_pools': True}}, 'factor_data', 'price_data',
            'duckdb', pd.Timestamp('2026-01-31'), 28)


@pytest.mark.parametrize('train_end,days,expected_dates', [
    ('2026-01-31', 28, ('2026-02-01', '2026-03-01')),
    ('2024-02-28', 1, ('2024-02-29', '2024-03-01')),
    ('2026-02-01', 0, ('2026-02-02', '2026-02-02')),
])
def test_prediction_pool_keeps_date_convention_custom_filters_and_empty_period(monkeypatch, train_end, days, expected_dates):
    from factor_synthesis.pipeline.lasso_method import _filter_by_pool_date_range
    import logging
    calls = []
    class Manager:
        def get_stock_pools_for_period(self, **kwargs):
            calls.append(kwargs)
            return {expected_dates[0]: []}
    monkeypatch.setattr(module, '_create_pool_manager', lambda **kwargs: Manager())
    def unexpected(*args, **kwargs): pytest.fail('Prediction must not add basic filters')
    monkeypatch.setattr(module.DynamicStockPoolManager, 'add_basic_filters', unexpected)
    config = {'stock_pool': {'use_dynamic_pools': True, 'include_basic_filters': True,
                            'filters': [{'field': 'market_cap', 'operator': '<', 'value': 1.5}]}}
    result = module.compute_predict_period_dynamic_pools(None, config, 'factor_data',
        'price_data', 'duckdb', pd.Timestamp(train_end), days)
    assert result == {expected_dates[0]: set()}
    assert result  # Explicit empty period still activates downstream membership filtering.
    call = calls[0]
    assert (call['start_date'], call['end_date']) == expected_dates
    assert call['rebalance_schedule'] == 'weekly'
    assert [(f.field, f.operator, f.value) for f in call['filters']] == [('market_cap', '<', 1.5)]
    index = pd.MultiIndex.from_tuples([(pd.Timestamp(expected_dates[0]), 'A')], names=['date', 'stock_code'])
    assert _filter_by_pool_date_range(index, result, logging.getLogger(__name__)) == []


def test_prediction_pool_disabled_stays_unrestricted(monkeypatch):
    def unexpected(**kwargs): pytest.fail('Disabled filter must not query pools')
    monkeypatch.setattr(module, '_create_pool_manager', unexpected)
    assert module.compute_predict_period_dynamic_pools(None, {}, 'factor_data', 'price_data',
        'duckdb', pd.Timestamp('2026-01-31'), 28) is None



def test_prediction_pool_holiday_only_final_window_is_explicitly_empty(source):
    # The existing lightgbm smoke configuration ends on Saturday 2023-07-01.
    result = module.compute_predict_period_dynamic_pools(source,
        {'stock_pool': {'use_dynamic_pools': True, 'filters': []}},
        'factor_data', 'price_data', 'duckdb', pd.Timestamp('2023-06-30'), 0)
    assert result == {'2023-07-01': set()}
    # Calendar data, rather than weekday arithmetic, also controls weekday holidays.
    source.execute("UPDATE trading_calendar SET is_open=false WHERE date=DATE '2023-07-03'")
    result = module.compute_predict_period_dynamic_pools(source,
        {'stock_pool': {'use_dynamic_pools': True, 'filters': []}},
        'factor_data', 'price_data', 'duckdb', pd.Timestamp('2023-07-02'), 0)
    assert result == {'2023-07-03': set()}


def test_topk_pool_missing_periods_with_sessions_raises(source, monkeypatch):
    monkeypatch.setattr(module.DynamicStockPoolManager, 'get_stock_pools_for_period', lambda *a, **k: {})
    with pytest.raises(ValueError, match='2024-02-01.*2024-02-29'):
        module.compute_dynamic_pools_for_topk(source,
            {'stock_pool': {'use_dynamic_pools': True}}, 'factor_data', 'price_data',
            'duckdb', '2024-02-01', '2024-02-29')


def test_topk_pool_holiday_only_window_has_explicit_empty_members(source):
    for date in ['2023-07-01', '2023-07-03']:
        source.execute('UPDATE trading_calendar SET is_open=false WHERE date=?', [date])
        result = module.compute_dynamic_pools_for_topk(source,
            {'stock_pool': {'use_dynamic_pools': True, 'rebalance_schedule': 'weekly'}}, 'factor_data', 'price_data',
            'duckdb', date, date)
        assert result == {date: []}


@pytest.mark.parametrize('pools', [{'2024-02-01': ['B', 'A']}, {'2024-02-01': []}])
def test_topk_pool_preserves_nonempty_history_and_order(source, monkeypatch, pools):
    monkeypatch.setattr(module.DynamicStockPoolManager, 'get_stock_pools_for_period', lambda *a, **k: pools)
    result = module.compute_dynamic_pools_for_topk(source,
        {'stock_pool': {'use_dynamic_pools': True}}, 'factor_data', 'price_data',
        'duckdb', '2024-02-01', '2024-02-29')
    assert result is pools


def test_topk_disabled_keeps_unrestricted_contract(monkeypatch):
    def unexpected(**kwargs): pytest.fail('Disabled pool must not query')
    monkeypatch.setattr(module, '_create_pool_manager', unexpected)
    assert module.compute_dynamic_pools_for_topk(None, {}, 'factor_data', 'price_data',
        'duckdb', '2024-02-01', '2024-02-29') == {}
