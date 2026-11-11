"""Dynamic-pool as-of boundaries, ordering and duplicate preservation."""
import logging

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.pipeline.lasso_method import _filter_by_pool_date_range

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize('pool_type', [list, set, tuple])
def test_asof_boundaries_and_original_order(pool_type):
    tuples = [('2020-01-09', 'C'), ('2020-01-03', 'A'), ('2020-01-01', 'A'),
              ('2020-01-02', 'A'), ('2020-01-06', 'A'), ('2020-01-07', 'C'),
              ('2020-01-03', 'A'), ('2020-01-08', 'C')]
    index = pd.MultiIndex.from_tuples(tuples, names=['date', 'stock_code'])
    pools = {'2020-01-08': pool_type(['C']), '2020-01-02': pool_type(['A']),
             '2020-01-06': None, '2020-01-07': pool_type([])}
    assert _filter_by_pool_date_range(index, pools, LOGGER) == [tuples[i] for i in [0, 1, 3, 6, 7]]


def test_empty_pools_and_empty_index():
    index = pd.MultiIndex.from_tuples([('2020-01-01', 'A'), ('2020-01-01', 'A')])
    assert _filter_by_pool_date_range(index, {}, LOGGER) == list(index)
    assert _filter_by_pool_date_range(index[:0], {'2020-01-01': ['A']}, LOGGER) == []


def test_mixed_date_formats_and_last_equivalent_pool_key_wins():
    index = pd.MultiIndex.from_tuples([('2020/01/02', 'B'), ('Jan 3 2020', 'B'),
                                      ('2020-01-02', 'A')])
    pools = {'2020-01-02': ['A'], pd.Timestamp('2020-01-02'): ['B']}
    assert _filter_by_pool_date_range(index, pools, LOGGER) == list(index[:2])


def test_intraday_timezone_boundaries_are_not_normalized():
    times = pd.to_datetime(['2020-01-02 09:00Z', '2020-01-02 12:00Z', '2020-01-02 15:00Z'])
    index = pd.MultiIndex.from_product([times, ['A', 'B']])
    pools = {times[0]: ['A'], times[1]: ['B']}
    assert _filter_by_pool_date_range(index, pools, LOGGER) == [index[0], index[3], index[5]]


def test_random_unsorted_duplicate_indices_match_asof_reference():
    rng = np.random.default_rng(17)
    dates = pd.bdate_range('2020-01-01', periods=50)
    pool_dates = dates[3::5]
    pools = {date: set(rng.choice(20, 8, replace=False)) for date in pool_dates}
    index = pd.MultiIndex.from_tuples([(dates[int(rng.integers(50))], int(rng.integers(20)))
                                      for _ in range(2000)])
    expected = []
    for value in index:
        applicable = [date for date in pool_dates if date <= value[0]]
        if applicable and value[1] in pools[applicable[-1]]:
            expected.append(value)
    assert _filter_by_pool_date_range(index, pools, LOGGER) == expected
