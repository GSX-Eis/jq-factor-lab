import numpy as np
import pandas as pd
import pytest

from scripts.alignment.static_code_review import FactorCodeReviewer
from utils.jq_factor_adapter import rolling_money_flow_sum


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


@pytest.mark.parametrize('body,expected', [
    ('''table_name = getattr(self, "fundamental_table", "factor_data")
        fields = ["net_amount_main"]
        return self._fetch_raw_data(fields=fields, table_name=table_name)''', 'SKIP'),
    ('''fields = ["close"]
        return self._fetch_raw_data(fields=fields, table_name="price_data")''', 'ISSUE'),
    ('''fields = ["close", "factor"]
        return self._fetch_raw_data(fields=fields, table_name="price_data")''', 'OK'),
    ('''table_name = getattr(self, "fundamental_table", "factor_data")
        fields = ["net_amount_main"]
        flow = self._fetch_raw_data(fields=fields, table_name=table_name)
        return self._fetch_raw_data(fields=["close"], table_name="price_data_daily")''', 'ISSUE'),
    ('''table_name = getattr(self, "fundamental_table", "factor_data")
        fields = ["net_amount_main"]
        flow = self._fetch_raw_data(fields=fields, table_name=table_name)
        return self._fetch_raw_data(fields=["close"], table_name=self.price_table)''', 'ISSUE'),
    ('''table_name = getattr(self, "fundamental_table", "factor_data")
        fields = ["net_amount_main"]
        flow = self._fetch_raw_data(fields=fields, table_name=table_name)
        return get_price(fields=["close"])''', 'ISSUE'),
    ('''# table_name = getattr(self, "fundamental_table", "factor_data")
        fields = ["close"]
        return self._fetch_raw_data(fields=fields, table_name=unknown())''', 'ISSUE'),
    ('''table_name = getattr(self, "fundamental_table", "factor_data")
        fields = ["net_amount_main"]
        flow = self._fetch_raw_data(fields=fields, table_name=table_name)
        return self._fetch_raw_data(fields=["close"], table_name=unknown())''', 'ISSUE'),
    ('''table_name = getattr(self, "fundamental_table", "factor_data")
        fields = ["close"]
        table_name = unknown()
        return self._fetch_raw_data(fields=fields, table_name=table_name)''', 'ISSUE'),
])
def test_price_pit_check_distinguishes_fundamental_and_mixed_reads(tmp_path, body, expected):
    path = tmp_path / 'test_factor.py'
    path.write_text('class ExampleFactor:\n    def _get_raw_data_for_factor(self):\n        ' + body + '\n')
    assert FactorCodeReviewer(path).check_1_factor_field().status == expected
