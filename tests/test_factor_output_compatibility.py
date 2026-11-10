"""Require DataFrame outputs and preserve their supported date/stock identities."""
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from utils.factor_base import FactorBase


@pytest.mark.parametrize('with_close', [False, True])
def test_pit_valid_integer_money_preserves_existing_dtype(with_close):
    from utils.pit import apply_pit_adjustment

    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02']), 'code': ['A'],
                        'factor': [1.], 'money': [100]})
    if with_close:
        raw['close'] = 10.
    expected = raw.money if with_close else raw.money.astype(float)
    pd.testing.assert_series_equal(apply_pit_adjustment(raw).money, expected, check_exact=True)


@pytest.mark.parametrize('field', ['high_limit', 'low_limit'])
def test_pit_limit_price_with_money_preserves_endpoint(field):
    from utils.pit import apply_pit_adjustment

    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-03']),
                        'code': ['A', 'A'], 'factor': [1., 2.],
                        field: [20., 20.], 'money': [100., np.nan]})
    original = raw.copy(deep=True)
    expected = raw.copy(deep=True)
    expected[field] = [10., 20.]
    expected.attrs['_sorted_by_code_date'] = True
    pd.testing.assert_frame_equal(apply_pit_adjustment(raw), expected, check_exact=True)
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


@pytest.mark.parametrize('factor', [0., np.nan, -1., np.inf, -np.inf])
@pytest.mark.parametrize('companion', ['close', 'volume'])
def test_pit_money_normalization_is_independent_of_field_projection(factor, companion):
    from utils.pit import apply_pit_adjustment

    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02 09:35', '2024-01-02 09:40']),
                        'code': ['A', 'A'], 'factor': [1., factor],
                        'money': [10., 0.], companion: [10., np.nan]})
    original = raw.copy(deep=True)
    wide = apply_pit_adjustment(raw)
    narrow = apply_pit_adjustment(raw.drop(columns=companion))
    pd.testing.assert_series_equal(wide.money, narrow.money, check_exact=True)
    assert wide.money.mean() == 10.
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


def evaluate(output, *, duplicate=False):
    raw = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-02', '2024-01-02']),
        'code': ['A', 'A' if duplicate else 'B'],
    }, index=[8, 3])
    context = SimpleNamespace(
        cache_dir=None, pit_enabled=False, use_dynamic_pool=False,
        logger=logging.getLogger('factor-output-compatibility'),
        _get_raw_data_for_factor=lambda: raw,
        calculate_factor=lambda data: output,
    )
    return FactorBase.get_factor_values(context)


@pytest.mark.parametrize('entry', ['get_factor_values', 'get_factor_values_unfiltered'])
@pytest.mark.parametrize('output', [
    [100., 200.], (100., 200.), np.array([100., 200.]),
    np.array([[100., 200.]]), np.array([[100.], [200.]]),
    pd.Series([100., 200.], index=['B', 'A']),
    pd.Series([200., 100.], index=pd.MultiIndex.from_tuples([
        (pd.Timestamp('2024-01-02'), 'B'), (pd.Timestamp('2024-01-02'), 'A'),
    ], names=['date', 'code'])),
    7., np.array(7.), pd.Series([1.]), [1.], None, {},
])
def test_non_dataframe_outputs_raise_instead_of_positional_conversion(entry, output):
    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02'] * 2), 'code': ['A', 'B']}, index=[8, 3])
    with pytest.raises(ValueError, match='因子输出必须是日期×股票 DataFrame'):
        evaluate_frame(output, raw, entry)


@pytest.mark.parametrize('entry', ['get_factor_values', 'get_factor_values_unfiltered'])
def test_explicit_empty_dataframe_remains_no_result(entry):
    raw, _ = panel_data()
    assert evaluate_frame(pd.DataFrame(), raw, entry).empty


@pytest.mark.parametrize('entry', ['get_factor_values', 'get_factor_values_unfiltered'])
def test_keyed_output_order_is_preserved_in_both_entries(entry):
    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02'] * 2), 'code': ['A', 'B']})
    output = pd.DataFrame({'date': pd.to_datetime(['2024-01-02'] * 2),
                           'code': ['B', 'A'], 'factor': [200., 100.]})
    assert evaluate_frame(output, raw, entry).iloc[0].to_dict() == {'A': 100., 'B': 200.}


def test_valid_dataframe_is_returned_without_rebuilding():
    frame = pd.DataFrame({'A': [1.], 'B': [2.]},
                         index=pd.DatetimeIndex(['2024-01-02'], name='date'))
    assert evaluate(frame) is frame


def evaluate_frame(output, raw, entry='get_factor_values'):
    context = SimpleNamespace(
        cache_dir=None, pit_enabled=False, use_dynamic_pool=False,
        dynamic_pool_mode='factor', all_stocks=['A', 'B'],
        logger=logging.getLogger('factor-dataframe-output'),
        _get_raw_data_for_factor=lambda: raw,
        calculate_factor=lambda data: output,
    )
    return getattr(FactorBase, entry)(context)


def panel_data():
    dates = pd.to_datetime(['2024-01-02', '2024-01-03'])
    raw = pd.DataFrame({'date': dates.tolist() * 2, 'code': ['A', 'A', 'B', 'B']})
    wide = pd.DataFrame({'A': [10., 20.], 'B': [100., 200.]}, index=dates.rename('date'))
    return raw, wide


@pytest.mark.parametrize('entry', ['get_factor_values', 'get_factor_values_unfiltered'])
@pytest.mark.parametrize('form', ['unnamed', 'string_dates', 'sparse', 'long', 'single_column'])
def test_dataframe_conversion_preserves_date_stock_keys(entry, form):
    raw, expected = panel_data()
    if form == 'long':
        output = raw.assign(factor=[10., 20., 100., 200.]).iloc[::-1]
        expected.columns.name = 'code'
    elif form == 'single_column':
        output = pd.DataFrame({'value': [10., 20., 100., 200.]}, index=raw.index)
        expected.columns.name = 'code'
    else:
        if form == 'sparse':
            raw = raw.iloc[:3]
            expected.iloc[1, 1] = np.nan
        output = expected.rename_axis(None)
        if form == 'string_dates':
            output.index = output.index.strftime('%Y-%m-%d')
    before = output.copy(deep=True)
    actual = evaluate_frame(output, raw, entry)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(output, before)


@pytest.mark.parametrize('entry', ['get_factor_values', 'get_factor_values_unfiltered'])
@pytest.mark.parametrize('invalid', [
    'anonymous_matrix', 'wrong_single_column_index', 'duplicate_dates',
    'duplicate_codes', 'non_string_codes', 'bad_dates', 'missing_date',
    'long_duplicate_keys', 'long_extra_column',
])
def test_ambiguous_or_invalid_dataframe_raises(entry, invalid):
    raw, wide = panel_data()
    if invalid == 'anonymous_matrix':
        output = pd.DataFrame(wide.to_numpy())
    elif invalid == 'wrong_single_column_index':
        output = pd.DataFrame({'value': [10., 20., 100., 200.]}, index=[3, 2, 1, 0])
    elif invalid == 'duplicate_dates':
        output = wide.set_axis([wide.index[0]] * 2)
    elif invalid == 'duplicate_codes':
        output = wide.set_axis(['A', 'A'], axis=1)
    elif invalid == 'non_string_codes':
        output = wide.set_axis([1, 2], axis=1)
    elif invalid == 'bad_dates':
        output = wide.set_axis(['bad', 'dates'])
    elif invalid == 'missing_date':
        output = wide.set_axis([wide.index[0], pd.NaT])
    else:
        output = raw.assign(factor=[10., 20., 100., 200.])
        if invalid == 'long_duplicate_keys':
            output = pd.concat([output, output.iloc[:1]])
        else:
            output['extra'] = 0
    with pytest.raises(ValueError):
        evaluate_frame(output, raw, entry)


def test_composite_producer_returns_wide_frame():
    from factor_synthesis.composite_factor_base import CompositeFactorBase
    raw, expected = panel_data()
    raw['composite_factor'] = [10., 20., 100., 200.]
    context = SimpleNamespace(logger=logging.getLogger('composite-output'))
    actual = CompositeFactorBase.calculate_factor(context, raw.iloc[::-1])
    expected.columns.name = 'code'
    pd.testing.assert_frame_equal(actual, expected)


def test_single_stock_wide_frame_is_not_mistaken_for_positional_values():
    raw, expected = panel_data()
    output = expected[['A']].iloc[::-1].rename_axis(None)
    actual = evaluate_frame(output, raw)
    pd.testing.assert_frame_equal(actual, output.rename_axis('date'))


def test_numeric_dates_are_not_silently_interpreted_as_epoch_nanoseconds():
    raw, wide = panel_data()
    with pytest.raises(ValueError):
        evaluate_frame(wide.set_axis(pd.Index([0, 1], dtype=object)), raw)
    with pytest.raises(ValueError):
        evaluate_frame(raw.assign(date=[0, 1, 2, 3], factor=1.), raw)


def test_unfiltered_restores_pool_state_after_invalid_output():
    raw, wide = panel_data()
    context = SimpleNamespace(
        all_stocks=['A', 'B'], use_dynamic_pool=True, dynamic_pool_mode='both',
        pit_enabled=False, logger=logging.getLogger('state-restoration'),
        _get_raw_data_for_factor=lambda: raw,
        calculate_factor=lambda _: pd.DataFrame(wide.to_numpy()),
    )
    with pytest.raises(ValueError):
        FactorBase.get_factor_values_unfiltered(context)
    assert context.all_stocks == ['A', 'B']
    assert context.use_dynamic_pool is True
    assert context.dynamic_pool_mode == 'both'


@pytest.mark.parametrize('long_output', [False, True])
def test_rebalance_window_accepts_keyed_outputs(long_output):
    factor = object.__new__(FactorBase)
    factor.use_dynamic_pool = True
    factor.monthly_stock_pools = {'2024-01-04': ['A', 'B']}
    factor.start_date = '2024-01-01'
    factor.end_date = '2024-01-04'
    factor.all_stocks = ['A', 'B']
    factor.dynamic_pool_mode = 'factor'
    factor.pit_enabled = False
    factor.logger = logging.getLogger('rebalance-output')
    raw, expected = panel_data()
    factor._get_raw_data_for_factor = lambda: raw
    def calculate(data):
        if long_output:
            return raw.assign(factor=[10., 20., 100., 200.]).iloc[::-1]
        return expected.rename_axis(None)
    factor.calculate_factor = calculate
    actual = factor._get_factor_values_by_rebalance_pool(window_bars=2)
    assert actual.loc[pd.Timestamp('2024-01-03'), ['A', 'B']].tolist() == [20., 200.]


@pytest.mark.parametrize('mode', ['factor', 'both'])
@pytest.mark.parametrize('cache_dir', [None, 'enabled'])
def test_pool_filter_failure_propagates_without_saving_result(mode, cache_dir):
    raw, wide = panel_data()
    error = RuntimeError('pool filter failed')
    saved = []

    def fail_filter(values, **kwargs):
        raise error

    context = SimpleNamespace(
        cache_dir=cache_dir, pit_enabled=False, use_dynamic_pool=True,
        stock_pools_by_period={'2024-01': ['A']}, dynamic_pool_mode=mode,
        logger=logging.getLogger('pool-failure'),
        _get_raw_data_for_factor=lambda: raw,
        _get_cache_key=lambda data: 'key', _load_from_cache=lambda key: None,
        calculate_factor=lambda data: wide,
        _apply_dynamic_pool_to_factor_df=fail_filter,
        _save_to_cache=lambda values, key: saved.append((values, key)),
    )
    with pytest.raises(RuntimeError) as caught:
        FactorBase.get_factor_values(context)
    assert caught.value is error
    assert saved == []


@pytest.mark.parametrize('pools', [
    {'2024-01-02': ['A'], '2024-02-01': ['B']},
    {'2024-01': ['A'], '2024-02': ['B']},
])
def test_result_pool_filter_preserves_daily_rows_and_empty_rows(pools):
    wide = pd.DataFrame(
        {'A': [1., 2., 3., 4., np.nan], 'B': [10., 20., 30., 40., np.nan]},
        index=pd.DatetimeIndex([
            '2023-12-29', '2024-01-02', '2024-01-03', '2024-02-01', '2024-02-02',
        ], name='date'),
    )
    original = wide.copy()
    context = SimpleNamespace(
        cache_dir=None, pit_enabled=False, use_dynamic_pool=True,
        stock_pools_by_period=pools, dynamic_pool_mode='factor',
        logger=logging.getLogger('pool-daily-output'),
        _get_raw_data_for_factor=lambda: wide.reset_index().melt(
            id_vars='date', var_name='code', value_name='input_value',
        ),
        calculate_factor=lambda data: wide,
    )
    context._apply_dynamic_pool_to_factor_df = (
        lambda values: FactorBase._apply_dynamic_pool_to_factor_df(context, values)
    )
    actual = FactorBase.get_factor_values(context)
    expected = pd.DataFrame(
        {'A': [np.nan, 2., 3., np.nan, np.nan],
         'B': [np.nan, np.nan, np.nan, 40., np.nan]},
        index=wide.index,
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(wide, original)


def test_pool_filter_invalid_date_raises_instead_of_returning_unfiltered_values():
    frame = pd.DataFrame({'A': [1.], 'B': [2.]}, index=['2024-13-02'])
    context = SimpleNamespace(
        use_dynamic_pool=True, stock_pools_by_period={'2024-01': ['A']},
        logger=logging.getLogger('pool-invalid-date'),
    )
    with pytest.raises(ValueError):
        FactorBase._apply_dynamic_pool_to_factor_df(context, frame)
    assert frame.index.tolist() == ['2024-13-02']


def test_pool_filter_converts_string_dates_without_mutating_input():
    frame = pd.DataFrame({'A': [1.], 'B': [2.]}, index=['2024-01-02'])
    original = frame.copy()
    context = SimpleNamespace(
        use_dynamic_pool=True, stock_pools_by_period={'2024-01': ['A']},
        logger=logging.getLogger('pool-string-date'),
    )
    actual = FactorBase._apply_dynamic_pool_to_factor_df(context, frame)
    expected = pd.DataFrame(
        {'A': [1.], 'B': [np.nan]}, index=pd.DatetimeIndex(['2024-01-02']),
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('pools', [
    {'2024-01': ['A'], '2024-02-01': ['B']},
    {'2024-13': ['A']}, {'2024-02-30': []}, {'2024-1': ['A']},
    {'2024-1-02': ['A']}, {'bad': []}, {202401: ['A']},
])
@pytest.mark.parametrize('entry', [
    '_filter_data_by_pools', '_apply_dynamic_pool_to_factor_df', 'get_stock_pool_for_date',
])
def test_pool_entries_reject_invalid_or_mixed_keys(pools, entry):
    context = SimpleNamespace(use_dynamic_pool=True, stock_pools_by_period=pools)
    raw, wide = panel_data()
    argument = {'_filter_data_by_pools': raw, '_apply_dynamic_pool_to_factor_df': wide,
                'get_stock_pool_for_date': '2024-01-02'}[entry]
    with pytest.raises(ValueError, match='[Ss]tock pool'):
        getattr(FactorBase, entry)(context, argument)


@pytest.mark.parametrize('pools,expected_codes', [
    ({'2024-03': ['B'], '2024-01': ['A']}, ['A', None, 'B', None]),
    ({'2024-03-01': ['B'], '2024-01-01': ['A']}, ['A', 'A', 'B', 'B']),
    ({'2024-03-01': [], '2024-01-01': ['A']}, ['A', 'A', None, None]),
])
def test_pool_entries_agree_on_gaps_boundaries_and_final_period(pools, expected_codes):
    dates = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-01', '2024-04-01'], name='date')
    wide = pd.DataFrame({'A': [1.] * 4, 'B': [2.] * 4}, index=dates)
    raw = wide.reset_index().melt(id_vars='date', var_name='code', value_name='value')
    context = SimpleNamespace(
        use_dynamic_pool=True, stock_pools_by_period=pools,
        logger=logging.getLogger('pool-intervals'),
    )
    filtered = FactorBase._apply_dynamic_pool_to_factor_df(context, wide)
    filtered_raw = FactorBase._filter_data_by_pools(context, raw)
    for date, code in zip(dates, expected_codes):
        expected = [code] if code else []
        assert filtered.loc[date].dropna().index.tolist() == expected
        assert filtered_raw.loc[filtered_raw.date == date, 'code'].tolist() == expected
        assert FactorBase.get_stock_pool_for_date(context, str(date.date())) == expected
    pd.testing.assert_index_equal(filtered.index, dates)


def test_pool_legacy_attribute_alias_shares_one_mapping():
    factor = FactorBase.__new__(FactorBase)
    pools = {'2024-01': ['A']}
    factor.monthly_stock_pools = pools
    assert factor.stock_pools_by_period is pools
    factor.stock_pools_by_period = {'2024-02': ['B']}
    assert factor.monthly_stock_pools is factor.stock_pools_by_period
    factor.monthly_stock_pools['2024-03'] = ['C']
    assert factor.stock_pools_by_period['2024-03'] == ['C']
    assert 'monthly_stock_pools' not in vars(factor)


@pytest.mark.parametrize('parameter', ['monthly_stock_pools', 'stock_pools_by_period'])
@pytest.mark.parametrize('filters', [None, ['unused']])
def test_pool_constructor_accepts_legacy_and_canonical_keyword(parameter, filters, monkeypatch):
    import utils.factor_base as module
    def unexpected_manager(*args, **kwargs):
        raise AssertionError('precomputed pools must not create a selector')
    monkeypatch.setattr(module, 'DynamicStockPoolManager', unexpected_manager)
    monkeypatch.delenv('QUANT_DATA_BINDINGS', raising=False)
    pools = {'2024-01': ['A']}
    factor = FactorBase(
        None, 'PoolTest', '2024-01-01', '2024-01-31',
        use_dynamic_pool=True, pit_enabled=False, stock_filters=filters,
        **{parameter: pools},
    )
    assert factor.stock_pools_by_period is pools
    assert factor.monthly_stock_pools is pools
    assert factor.all_stocks == ['A']
    assert factor.dynamic_pool_manager is None


def test_pool_constructor_rejects_conflicting_keywords_and_bad_keys():
    with pytest.raises(ValueError, match='not both'):
        FactorBase(None, 'PoolTest', '2024-01-01', '2024-01-31',
                   monthly_stock_pools={}, stock_pools_by_period={})
    with pytest.raises(ValueError, match='Invalid stock pool key'):
        FactorBase(None, 'PoolTest', '2024-01-01', '2024-01-31',
                   stock_pools_by_period={'2024-13': []})


def test_month_pool_uses_local_calendar_for_timezone_aware_inputs():
    dates = pd.DatetimeIndex(['2024-01-31 23:59', '2024-02-01 00:00'], tz='Asia/Shanghai', name='date')
    wide = pd.DataFrame({'A': [1., 2.]}, index=dates)
    raw = wide.reset_index().assign(code='A')
    context = SimpleNamespace(
        use_dynamic_pool=True, stock_pools_by_period={'2024-01': ['A']},
        logger=logging.getLogger('pool-timezone'),
    )
    actual = FactorBase._apply_dynamic_pool_to_factor_df(context, wide)
    pd.testing.assert_frame_equal(actual, pd.DataFrame({'A': [1., np.nan]}, index=dates))
    pd.testing.assert_frame_equal(
        FactorBase._filter_data_by_pools(context, raw), raw.iloc[:1].reset_index(drop=True),
    )


@pytest.mark.parametrize('parameter', ['monthly_stock_pools', 'stock_pools_by_period'])
def test_jq_adapter_accepts_both_pool_keywords(parameter):
    from utils.jq_factor_adapter import JQFactorAdapter

    class PoolSignal(FactorBase):
        def __init__(self, **kwargs):
            super().__init__(pit_enabled=False, **kwargs)

        def calculate_factor(self, data):
            assert self.monthly_stock_pools is self.stock_pools_by_period
            return self._apply_dynamic_pool_to_factor_df(
                data.pivot(index='date', columns='code', values='close'),
            )

    data = pd.DataFrame({'date': pd.to_datetime(['2024-01-02'] * 2),
                         'code': ['A', 'B'], 'close': [1., 2.]})
    adapter = JQFactorAdapter({}, freq='daily', window=1)
    actual = adapter.run_factor(PoolSignal, data, **{parameter: {'2024-01': ['A']}})
    assert actual['A'].tolist() == [1.]
    assert actual['B'].isna().all()


def test_fixed_pool_uses_shared_filters_without_truncating_warmup(monkeypatch):
    import utils.factor_base as module
    def unexpected_manager(*args, **kwargs):
        raise AssertionError('fixed pools must not create a selector')
    monkeypatch.setattr(module, 'DynamicStockPoolManager', unexpected_manager)
    factor = FactorBase(None, 'FixedPool', '2024-01-03', '2024-01-04',
                        stock_pool=['A'], pit_enabled=False)
    dates = pd.DatetimeIndex(['2024-01-01', '2024-01-03', '2024-01-05'], name='date')
    wide = pd.DataFrame({'A': [1., 2., 3.], 'B': [4., 5., 6.]}, index=dates)
    raw = wide.reset_index().melt(id_vars='date', var_name='code', value_name='close')
    actual = factor._apply_dynamic_pool_to_factor_df(wide)
    expected = wide.copy()
    expected['B'] = np.nan
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(factor._filter_data_by_pools(raw), raw[raw.code == 'A'].reset_index(drop=True))
    for date in dates:
        assert factor.get_stock_pool_for_date(str(date.date())) == ['A']
    assert factor.use_dynamic_pool is False
    assert factor.stock_pools_by_period is None
    assert factor.dynamic_pool_manager is None


def test_fixed_pool_keeps_daily_calculation_path():
    class DailySignal(FactorBase):
        def _get_raw_data_for_factor(self):
            return pd.DataFrame({'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
                                 'code': ['A'] * 3, 'close': [10., 20., 30.]})

        def calculate_factor(self, data):
            return data.pivot(index='date', columns='code', values='close').rolling(2).mean()

    factor = DailySignal(None, 'DailySignal', '2024-01-02', '2024-01-03',
                         stock_pool=['A'], pit_enabled=False)
    expected = factor.get_factor_values()
    actual = factor._get_factor_values_by_rebalance_pool(window_bars=2)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc[pd.Timestamp('2024-01-02'), 'A'] == 15.
    assert actual.loc[pd.Timestamp('2024-01-03'), 'A'] == 25.
    assert factor.use_dynamic_pool is False


def test_temporarily_disabled_period_pool_does_not_apply_fixed_fallback():
    factor = FactorBase(None, 'Precomputed', '2024-01-01', '2024-01-31',
                        stock_pool=['A'], use_dynamic_pool=True,
                        stock_pools_by_period={'2024-01': ['A', 'B']}, pit_enabled=False)
    raw, wide = panel_data()
    factor.use_dynamic_pool = False
    assert factor._filter_data_by_pools(raw) is raw
    assert factor._apply_dynamic_pool_to_factor_df(wide) is wide
    assert factor.get_stock_pool_for_date('2024-01-02') == ['A']


def test_generated_pool_still_uses_selector_with_original_arguments(monkeypatch):
    import utils.factor_base as module
    calls = []
    pools = {'2024-01': ['A', 'B']}
    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return pools
    manager = SimpleNamespace(validate_filters=lambda filters: True, get_stock_pools_for_period=generate)
    monkeypatch.setattr(module, 'DynamicStockPoolManager', lambda *args, **kwargs: manager)
    filters = ['test']
    factor = FactorBase(None, 'Generated', '2024-01-01', '2024-01-31',
                        use_dynamic_pool=True, stock_filters=filters, pit_enabled=False)
    assert factor.dynamic_pool_manager is manager
    assert factor.stock_pools_by_period is pools
    assert calls == [(('2024-01-01', '2024-01-31', filters, None, None, False),
                      {'rebalance_period_months': 1, 'rebalance_schedule': 'quarterly', 'schedule': None})]
    precomputed = FactorBase(None, 'PrecomputedWithFilters', '2024-01-01', '2024-01-31',
                             use_dynamic_pool=True, stock_filters=filters,
                             stock_pools_by_period=pools, pit_enabled=False)
    assert precomputed.dynamic_pool_manager is None
    assert precomputed.stock_pools_by_period is pools
    assert len(calls) == 1  # The supplied history is not regenerated during initialization.
    with pytest.raises(ValueError, match='stock_filters'):
        FactorBase(None, 'MissingFilters', '2024-01-01', '2024-01-31',
                   use_dynamic_pool=True, pit_enabled=False)


@pytest.mark.parametrize('field', ['open', 'high', 'low', 'close', 'pre_close', 'avg', 'high_limit', 'low_limit', 'volume'])
@pytest.mark.parametrize('entry', ['shared', 'base', 'jq'])
def test_pit_requires_factor_for_adjusted_fields(field, entry):
    from utils.pit import apply_pit_adjustment
    from utils.jq_factor_adapter import JQFactorAdapter
    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02']), 'code': ['A'], field: [10.]})
    context = SimpleNamespace(pit_enabled=True, pit_factor_column='custom_factor', logger=logging.getLogger('pit-contract'))
    with pytest.raises(ValueError, match='pit_factor_column_missing'):
        if entry == 'shared': apply_pit_adjustment(raw, 'custom_factor')
        elif entry == 'base': FactorBase._apply_pit_adjustment(context, raw)
        else: JQFactorAdapter._apply_dynamic_pit(None, raw)


@pytest.mark.parametrize('field', ['val_market_cap', 'money'])
def test_pit_preserves_inputs_without_adjusted_fields(field):
    from utils.pit import apply_pit_adjustment
    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02']), 'code': ['A'], field: [10.]})
    assert apply_pit_adjustment(raw) is raw


def test_pit_custom_factor_and_disabled_or_empty_inputs():
    from utils.pit import apply_pit_adjustment
    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-03']), 'code': ['A', 'A'],
                        'close': [10., 12.], 'volume': [100., 200.], 'custom_factor': [1., 2.]})
    result = apply_pit_adjustment(raw, 'custom_factor')
    assert result['close'].tolist() == [5., 12.]
    assert result['volume'].tolist() == [200., 200.]
    missing = raw.drop(columns='custom_factor')
    context = SimpleNamespace(pit_enabled=False)
    assert FactorBase._apply_pit_adjustment(context, missing) is missing
    empty = missing.iloc[:0]
    assert apply_pit_adjustment(empty) is empty


@pytest.mark.parametrize('case', ['success', 'empty', 'fetch_error', 'pit_error', 'calculate_error', 'invalid_output', 'none_output'])
def test_unfiltered_records_attempted_stage_times_on_every_exit(case, monkeypatch):
    import itertools
    from utils import factor_base
    clock = itertools.count()
    monkeypatch.setattr(factor_base.time, 'perf_counter', lambda: float(next(clock)))
    raw = pd.DataFrame({'date': pd.to_datetime(['2024-01-02']), 'code': ['A'], 'close': [10.], 'factor': [1.]})
    failure = RuntimeError('original failure')
    def fetch():
        if case == 'fetch_error': raise failure
        return raw.iloc[:0] if case == 'empty' else raw
    def pit(data):
        if case == 'pit_error': raise failure
        return data
    def calculate(data):
        if case == 'calculate_error': raise failure
        if case == 'none_output': return None
        if case == 'invalid_output': return pd.DataFrame([[1., 2.]], columns=['A', 'A'])
        return data.pivot(index='date', columns='code', values='close')
    context = SimpleNamespace(all_stocks=['A'], use_dynamic_pool=True, dynamic_pool_mode='both',
        pit_enabled=True, logger=logging.getLogger('timings'), _get_raw_data_for_factor=fetch,
        _apply_pit_adjustment=pit, calculate_factor=calculate)
    if case.endswith('_error'):
        with pytest.raises(RuntimeError) as caught: FactorBase.get_factor_values_unfiltered(context)
        assert caught.value is failure
    elif case in {'invalid_output', 'none_output'}:
        with pytest.raises(ValueError): FactorBase.get_factor_values_unfiltered(context)
    else:
        result = FactorBase.get_factor_values_unfiltered(context)
        assert result.empty == (case == 'empty')
        if case == 'success': assert result.iloc[0, 0] == 10.
    metrics = context.last_performance_metrics
    expected = {'raw_fetch_seconds', 'total_seconds'}
    if case not in {'fetch_error', 'empty'}: expected.add('pit_seconds')
    if case not in {'fetch_error', 'empty', 'pit_error'}: expected.add('calculate_seconds')
    assert {k for k in metrics if k.endswith('_seconds')} == expected
    assert all(metrics[k] > 0 for k in expected)
    assert context.use_dynamic_pool is True and context.dynamic_pool_mode == 'both'
    assert context.all_stocks == ['A']


class _PoolBoundaryProbe(FactorBase):
    def _get_raw_data_for_factor(self):
        self.fetch_calls += 1
        return self.raw.loc[self.raw.code.isin(self.all_stocks)].copy()

    def calculate_factor(self, data):
        self.calculation_windows.append(data.copy())
        return data.pivot(index='date', columns='code', values='close')


def _pool_boundary_probe(pools, start, pit_enabled=False):
    factor = _PoolBoundaryProbe(None, 'boundaries', start, '2024-01-12',
        use_dynamic_pool=True, stock_pools_by_period=pools,
        pit_enabled=pit_enabled, data_bindings={})
    factor.raw = pd.DataFrame([
        (day, code, value, 1.)
        for day in pd.bdate_range('2024-01-01', '2024-01-12')
        for code, value in [('A', 10.), ('B', 20.)]
    ], columns=['date', 'code', 'close', 'factor'])
    factor.fetch_calls = 0
    factor.calculation_windows = []
    return factor


@pytest.mark.parametrize('pit_enabled', [False, True])
@pytest.mark.parametrize('pools,start,a_days,b_days', [
    ({'2024-01-02': ['A'], '2024-01-05': [], '2024-01-10': ['B']}, '2024-01-02', [2, 3, 4], [10, 11, 12]),
    ({'2024-01-02': ['A'], '2024-01-05': [], '2024-01-10': ['B']}, '2024-01-08', [], [10, 11, 12]),
    ({'2024-01-02': ['A'], '2024-01-05': [], '2024-01-10': ['B']}, '2024-01-05', [], [10, 11, 12]),
    ({'2024-01-02': ['A'], '2024-01-05': []}, '2024-01-02', [2, 3, 4], []),
    ({'2024-01-02': ['A'], '2024-01-05': []}, '2024-01-08', [], []),
    ({'2024-01-02': [], '2024-01-05': ['B']}, '2024-01-02', [], [5, 8, 9, 10, 11, 12]),
    ({'2024-01-02': ['A'], '2024-01-05': [], '2024-01-08': [], '2024-01-10': ['B']}, '2024-01-02', [2, 3, 4], [10, 11, 12]),
    ({'2024-01-02': [], '2024-01-05': []}, '2024-01-02', [], []),
    ({'2024-01-02': ['A'], '2024-01-05': ['B']}, '2024-01-02', [2, 3, 4], [5, 8, 9, 10, 11, 12]),
])
def test_daily_rebalance_preserves_empty_pool_boundaries(pools, start, a_days, b_days, pit_enabled):
    factor = _pool_boundary_probe(pools, start, pit_enabled)
    original_stocks = factor.all_stocks
    original_mode = factor.dynamic_pool_mode
    actual = factor._get_factor_values_by_rebalance_pool(window_bars=1, rebalance_only=False)
    expected = {(pd.Timestamp(f'2024-01-{day:02d}'), code): value
                for code, days, value in [('A', a_days, 10.), ('B', b_days, 20.)]
                for day in days}
    assert actual.stack().to_dict() == expected
    assert factor.all_stocks is original_stocks and factor.use_dynamic_pool is True
    assert factor.dynamic_pool_mode == original_mode and factor.start_date == start
    if not expected:
        assert factor.fetch_calls == 0 and factor.calculation_windows == []
    for window in factor.calculation_windows:
        if window.code.eq('A').all() and pools.get('2024-01-05') == []:
            assert window.date.max() < pd.Timestamp('2024-01-05')


@pytest.mark.parametrize('pit_enabled', [False, True])
def test_rebalance_single_signal_unchanged_by_empty_period(pit_enabled):
    pools = {'2024-01-03': ['A'], '2024-01-05': [], '2024-01-10': ['B']}
    factor = _pool_boundary_probe(pools, '2024-01-02', pit_enabled)
    baseline = _pool_boundary_probe({key: stocks for key, stocks in pools.items() if stocks},
        '2024-01-02', pit_enabled)
    actual = factor._get_factor_values_by_rebalance_pool(window_bars=1)
    expected = baseline._get_factor_values_by_rebalance_pool(window_bars=1)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.stack().to_dict() == {
        (pd.Timestamp('2024-01-02'), 'A'): 10.,
        (pd.Timestamp('2024-01-09'), 'B'): 20.,
    }
    assert len(factor.calculation_windows) == 2


@pytest.mark.parametrize('rebalance_only', [False, True])
@pytest.mark.parametrize('field,bad_value', [
    ('date', 'bad-date'), ('date', None), ('date', pd.NaT),
    ('date', float('nan')), ('date', pd.NA),
    ('code', None), ('code', float('nan')), ('code', pd.NA),
    ('code', ''), ('code', '   '),
])
def test_rebalance_rejects_invalid_raw_keys(field, bad_value, rebalance_only):
    factor = _pool_boundary_probe({'2024-01-04': ['A']}, '2024-01-01')
    raw = pd.DataFrame({'date': ['2024-01-02', '2024-01-03'],
                        'code': ['A', 'A'], 'close': [10., 20.]})
    raw.loc[1, field] = bad_value
    original_raw = raw.copy(deep=True)
    factor._get_raw_data_for_factor = lambda: raw
    original_stocks = factor.all_stocks
    original_mode = factor.dynamic_pool_mode
    with pytest.raises(ValueError, match=f'rebalance raw data.*{field}'):
        factor._get_factor_values_by_rebalance_pool(
            window_bars=1, warmup_calendar_days=10, rebalance_only=rebalance_only)
    assert factor.calculation_windows == []
    pd.testing.assert_frame_equal(raw, original_raw)
    assert factor.all_stocks is original_stocks
    assert factor.start_date == '2024-01-01'
    assert factor.use_dynamic_pool is True and factor.dynamic_pool_mode == original_mode


@pytest.mark.parametrize('rebalance_only', [False, True])
@pytest.mark.parametrize('pit_enabled', [False, True])
def test_rebalance_valid_string_keys_preserve_results_and_input(rebalance_only, pit_enabled):
    pools = {'2024-01-03': ['A'], '2024-01-08': ['B']}
    factor = _pool_boundary_probe(pools, '2024-01-02', pit_enabled)
    expected = factor._get_factor_values_by_rebalance_pool(1, rebalance_only=rebalance_only)
    raw = factor.raw.iloc[::-1].copy()
    raw['date'] = raw.date.dt.strftime('%Y-%m-%d')
    original_raw = raw.copy(deep=True)
    factor._get_raw_data_for_factor = lambda: raw
    actual = factor._get_factor_values_by_rebalance_pool(1, rebalance_only=rebalance_only)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(raw, original_raw)


@pytest.mark.parametrize('rebalance_only', [False, True])
@pytest.mark.parametrize('output', [None, pd.Series(dtype=float), pd.Series([10.]), []])
def test_rebalance_rejects_non_dataframe_outputs(output, rebalance_only):
    factor = _pool_boundary_probe({'2024-01-04': ['A']}, '2024-01-02')
    factor.calculate_factor = lambda data: output
    original_stocks = factor.all_stocks
    original_mode = factor.dynamic_pool_mode
    with pytest.raises(ValueError, match='因子输出必须是日期×股票 DataFrame'):
        factor._get_factor_values_by_rebalance_pool(
            2, warmup_calendar_days=10, rebalance_only=rebalance_only)
    assert factor.all_stocks is original_stocks
    assert factor.start_date == '2024-01-02'
    assert factor.use_dynamic_pool is True and factor.dynamic_pool_mode == original_mode


@pytest.mark.parametrize('rebalance_only', [False, True])
@pytest.mark.parametrize('combine_duplicate_dates', [False, True])
def test_rebalance_rejects_multiple_outputs_on_one_day(rebalance_only, combine_duplicate_dates):
    factor = _pool_boundary_probe({'2024-01-04': ['A']}, '2024-01-02')
    output = pd.DataFrame({'A': [10., 20.]}, index=pd.DatetimeIndex(
        ['2024-01-04 09:35', '2024-01-04 09:40'], name='date'))
    original = output.copy(deep=True)
    factor.calculate_factor = lambda data: output
    original_stocks = factor.all_stocks
    original_mode = factor.dynamic_pool_mode
    with pytest.raises(ValueError, match='因子输出按日期归一化后必须唯一'):
        factor._get_factor_values_by_rebalance_pool(2, rebalance_only=rebalance_only,
            combine_duplicate_dates=combine_duplicate_dates)
    pd.testing.assert_frame_equal(output, original)
    assert factor.all_stocks is original_stocks
    assert factor.start_date == '2024-01-02'
    assert factor.use_dynamic_pool is True and factor.dynamic_pool_mode == original_mode


@pytest.mark.parametrize('rebalance_only', [False, True])
@pytest.mark.parametrize('kind', ['empty', 'daily', 'one_timestamp_per_day'])
def test_rebalance_daily_output_contract_preserves_values(kind, rebalance_only):
    factor = _pool_boundary_probe({'2024-01-04': ['A']}, '2024-01-02')
    if kind == 'empty':
        output = pd.DataFrame()
    else:
        dates = pd.DatetimeIndex(['2024-01-03', '2024-01-04', '2024-01-05'], name='date')
        if kind == 'one_timestamp_per_day':
            dates += pd.Timedelta(hours=15)
        output = pd.DataFrame({'A': [10., 20., 30.]}, index=dates)
    original = output.copy(deep=True)
    factor.calculate_factor = lambda data: output
    actual = factor._get_factor_values_by_rebalance_pool(2, rebalance_only=rebalance_only)
    expected = {} if kind == 'empty' else (
        {(pd.Timestamp('2024-01-03'), 'A'): 10.} if rebalance_only else
        {(pd.Timestamp('2024-01-04'), 'A'): 20., (pd.Timestamp('2024-01-05'), 'A'): 30.})
    assert actual.stack().to_dict() == expected
    pd.testing.assert_frame_equal(output, original)
