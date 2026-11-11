"""Synthetic regressions for the local adapter changes authorized in NERE-209."""
import sys
import types
import numpy as np
import pandas as pd
import pytest
from utils.jq_factor_adapter import rolling_money_flow_sum


def test_prehistory_and_exact_rolling_sum_work_together(monkeypatch):
    import utils.jq_factor_adapter as module

    requests = []

    def money_flow(securities, **kwargs):
        requests.append(kwargs)
        if 'count' in kwargs:
            assert kwargs['count'] == 2
            assert str(kwargs['end_date']) == '2022-01-02'
            return pd.DataFrame({'date': ['2022-01-01', '2022-01-02'],
                                 'sec_code': ['A', 'A'], 'net_amount_main': [1e16, 1.]})
        return pd.DataFrame({'date': ['2022-01-03'], 'sec_code': ['A'],
                             'net_amount_main': [-1e16]})

    monkeypatch.setitem(sys.modules, 'jqdata', types.SimpleNamespace(get_money_flow=money_flow))
    monkeypatch.setattr(module, 'CLOUD_API_MAPPING', {'get_money_flow': {
        'FLOW': {'field': 'net_amount_main', 'rolling_days': 3, 'min_periods': 3}}})
    result = module.JQFactorAdapter({}, strict=True)._fetch_money_flow(
        ['A'], '2022-01-03', '2022-01-03', ['FLOW'])
    expected = pd.DataFrame({'date': pd.to_datetime(['2022-01-03']), 'code': ['A'], 'FLOW': [1.]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected, check_exact=True)
    assert len(requests) == 2

def test_sum_preserves_exact_cancellation_ties_and_missing_rules():
    a = pd.Series([1e16, 1., -1e16] + [0.] * 17, name='flow')
    b = pd.Series([1.] + [0.] * 19, name='flow')
    values = pd.Series([rolling_money_flow_sum(a, 20, 20).iloc[-1],
                        rolling_money_flow_sum(b, 20, 20).iloc[-1]])
    assert values.tolist() == [1., 1.]
    assert values.rank(method='average', pct=True).tolist() == [.75, .75]
    observed = pd.Series([1., np.nan, 2., np.inf, 4.], index=[8, 2, 5, 1, 9], name='flow')
    expected = pd.Series([np.nan, np.nan, 3., np.nan, 6.], index=observed.index, name='flow')
    pd.testing.assert_series_equal(rolling_money_flow_sum(observed, 3, 2), expected, check_exact=True)


def test_sum_depends_only_on_finite_values_in_the_current_window():
    values = pd.Series(np.random.default_rng(91).normal(size=50) * 1e5)
    prefix = pd.Series([1e16, 1., -1e16, np.nan, -1e15])
    original = rolling_money_flow_sum(values, 20, 16)
    prefixed = rolling_money_flow_sum(pd.concat([prefix, values], ignore_index=True), 20, 16)
    np.testing.assert_array_equal(original.iloc[19:], prefixed.iloc[len(prefix) + 19:])
    pd.testing.assert_series_equal(original.iloc[:30], rolling_money_flow_sum(values.iloc[:30], 20, 16), check_exact=True)
    empty_minimum = rolling_money_flow_sum(pd.Series([np.nan, np.inf]), 2, 0)
    assert empty_minimum.tolist() == [0., 0.]


@pytest.mark.parametrize('window,minimum', [(0, 1), (2, -1), (2, 3), (True, 1), (2, True)])
def test_invalid_window_contract_is_rejected(window, minimum):
    with pytest.raises(ValueError, match='money_flow_rolling_contract_invalid'):
        rolling_money_flow_sum(pd.Series([1., 2.]), window, minimum)


@pytest.mark.parametrize('pit', [False, True])
def test_large_price_fetch_preserves_api_batches_fields_and_pit(monkeypatch, pit):
    import utils.jq_factor_adapter as module

    monkeypatch.setattr(module, 'EXTERNAL_PIT_MODE', None)
    days = pd.bdate_range('2021-01-04', periods=120)
    offsets = np.r_[np.arange(575, 695, 5), np.arange(785, 905, 5)]
    times = np.repeat(days.to_numpy(), 48) + np.tile(offsets, len(days)).astype('timedelta64[m]')
    stocks = [f'S{index:02}' for index in reversed(range(36))]
    raw = pd.DataFrame(dict(date=np.repeat(times, len(stocks)),
        code=np.tile(stocks, len(times)), open=10., close=11.))
    raw.loc[raw.index[::53], 'close'] = np.nan
    calls = []

    def make_adapter():
        def api(**kwargs):
            calls.append(tuple(kwargs['security']))
            return (raw.loc[raw.code.isin(kwargs['security']), ['date', 'code'] + kwargs['fields']]
                    .sort_values(['date', 'code']).reset_index(drop=True))

        adapter = module.JQFactorAdapter({}, freq='5m', window=5760,
            get_price_func=api, strict=True, pit_mode='research')
        adapter._probed = True
        adapter.pit_enabled = pit

        def factors(securities, start, end):
            dates = np.repeat(days.to_numpy(), len(securities))
            codes = np.tile(securities, len(days))
            return pd.DataFrame(dict(date=dates, code=codes,
                factor=(np.asarray([int(code[1:]) + 1 for code in codes])
                        * np.where(pd.DatetimeIndex(dates).day > 15, 2., 1.))))

        adapter._fetch_factor_separately = factors
        return adapter

    fields = ['open', 'close', 'factor'] if pit else ['open', 'close']
    original = make_adapter()._fetch_price_unbatched(stocks, '2022-01-04', fields)
    original_calls = list(calls)
    calls.clear()
    bounded = make_adapter().fetch_price(stocks, '2022-01-04', fields)
    pd.testing.assert_frame_equal(original, bounded, check_exact=True)
    assert calls == original_calls
    assert len(bounded) == len(raw) and not bounded.duplicated(['date', 'code']).any()


@pytest.mark.parametrize('frequency,expanding,fields', [
    ('daily', False, ['close']), ('5m', True, ['close']), ('5m', False, ['custom_field'])])
def test_unrelated_price_fetch_paths_keep_the_original_call(frequency, expanding, fields):
    from utils.jq_factor_adapter import JQFactorAdapter, bounded_price_fetch

    adapter = JQFactorAdapter({}, freq=frequency, window=10000)
    adapter.use_expanding_window = expanding
    stocks = ['A', 'B'] * 20
    sentinel = object()
    calls = []

    def fetch_one(securities, end, requested):
        calls.append((securities, end, requested))
        return sentinel

    assert bounded_price_fetch(adapter, stocks, '2022-01-04', fields, fetch_one) is sentinel
    assert calls == [(stocks, '2022-01-04', fields)]


def test_factor_field_aliases_deduplicate_api_requests_and_preserve_missing(monkeypatch):
    import utils.jq_factor_adapter as adapter_module

    requested = []
    dates = pd.to_datetime(['2022-01-03', '2022-01-04'])
    stocks = [f'S{i:02}' for i in range(27)]
    def get_factors(securities, factors, **kwargs):
        requested.append((list(securities), list(factors)))
        return {name: pd.DataFrame({stock: [float(i + 1), np.nan] for i, stock in enumerate(securities)}, index=dates)
                for name in factors}

    config = {'get_factor_values': ['beta'], 'field_contracts': {
        'FACTOR_CNE5_BETA': {'cloud': {'api': 'get_factor_values', 'field': 'beta'}},
        'FACTOR_CNE5_SIZE': {'cloud': {'api': 'get_factor_values', 'field': 'size'}},
    }}
    monkeypatch.setattr(adapter_module, 'CLOUD_API_MAPPING', config)
    monkeypatch.setattr(adapter_module, 'EXTERNAL_GET_FACTOR_VALUES', get_factors)
    result = adapter_module.JQFactorAdapter({}, strict=True)._fetch_extra_factors(
        stocks, dates[0], dates[-1], ['FACTOR_CNE5_BETA', 'FACTOR_CNE5_SIZE', 'beta'])
    assert [len(securities) for securities, _ in requested] == [25, 2]
    assert all(fields == ['beta', 'size'] for _, fields in requested)
    assert len(result) == 54 and not result.duplicated(['date', 'code']).any()
    assert result.loc[result.date.eq(dates[-1]), ['FACTOR_CNE5_BETA', 'FACTOR_CNE5_SIZE', 'beta']].isna().all().all()
    pd.testing.assert_series_equal(result.FACTOR_CNE5_BETA, result.beta, check_names=False)


