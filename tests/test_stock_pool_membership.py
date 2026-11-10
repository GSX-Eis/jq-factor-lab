"""Wide panels and SQL membership share effective-date and empty-period semantics."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from utils.stock_pool_membership import StockPoolMembership


def test_mask_preserves_axes_history_and_explicit_empty_period():
    dates = pd.date_range('2026-01-04', periods=5)
    frame = pd.DataFrame({'A': range(5), 'B': range(5, 10), 'C': np.nan}, index=dates)
    before = frame.copy()
    pools = {'2026-01-08': ['B'], '2026-01-05': ['A', 'C'], '2026-01-07': []}
    result = StockPoolMembership(pools).filter_wide(frame)
    expected = pd.DataFrame({'A': [np.nan, 1, 2, np.nan, np.nan],
                             'B': [np.nan, np.nan, np.nan, np.nan, 9], 'C': np.nan}, index=dates)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(frame, before)
    # The last pool continues and masks independently of input row order.
    later = frame.iloc[[4, 1]].rename(index={dates[4]: pd.Timestamp('2026-02-01')})
    filtered = StockPoolMembership(pools).filter_wide(later)
    assert filtered.index.equals(later.index)
    assert filtered.iloc[0]['B'] == 9
    assert filtered.iloc[1]['A'] == 1


def test_sql_membership_and_wide_mask_have_the_same_observations():
    dates = pd.date_range('2026-01-04', periods=5)
    prices = pd.DataFrame({'A': [1., 2., 3., 4., 5.], 'B': [5., np.nan, 3., 2., 1.]}, index=dates)
    membership = StockPoolMembership({'2026-01-05': ['A', 'B'], '2026-01-07': [], '2026-01-08': ['B']})
    daily = membership.daily_frame(dates)
    assert not daily.duplicated(['date', 'stock_code']).any()
    panel = prices.rename_axis(index='date', columns='stock_code').stack(future_stack=True).rename('value').reset_index()
    with duckdb.connect() as conn:
        conn.register('membership', daily)
        conn.register('panel', panel)
        observed = conn.execute('SELECT date, avg(value) AS value FROM panel JOIN membership USING(date,stock_code) GROUP BY date ORDER BY date').fetchdf()
    actual = observed.set_index('date')['value'].reindex(dates)
    expected = membership.filter_wide(prices).mean(axis=1)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


@pytest.mark.parametrize('pools', [None, {}, {'2026-1-05': ['A']}, {'2026-01': ['A']},
    {'2026-02-30': ['A']}, {pd.Timestamp('2026-01-05'): ['A']},
    {'2026-01-05': 'A'}, {'2026-01-05': None}, {'2026-01-05': ['A', 'A']},
    {'2026-01-05': ['']}, {'2026-01-05': [1]}])
def test_invalid_history_is_rejected_even_for_empty_panels(pools):
    with pytest.raises(ValueError):
        StockPoolMembership(pools).filter_wide(pd.DataFrame())


@pytest.mark.parametrize('dates', [pd.to_datetime(['2026-01-06', '2026-01-05']),
    pd.to_datetime(['2026-01-05', '2026-01-05']), pd.DatetimeIndex([pd.NaT]),
    pd.date_range('2026-01-05', periods=2, tz='UTC')])
def test_sql_membership_rejects_invalid_or_duplicate_days(dates):
    with pytest.raises(ValueError):
        StockPoolMembership({'2026-01-05': ['A']}).daily_frame(dates)


@pytest.mark.parametrize('dates', [pd.DatetimeIndex([]), pd.date_range('2025-01-01', periods=2)])
def test_uncovered_batch_is_empty_and_can_be_registered_with_duckdb(dates):
    daily = StockPoolMembership({'2026-01-05': ['A']}).daily_frame(dates)
    assert daily.empty
    with duckdb.connect() as conn:
        conn.register('membership', daily)
        assert conn.execute("SELECT count(*) FROM membership WHERE date=DATE '2025-01-01' AND stock_code='A'").fetchone() == (0,)


def test_explicit_empty_membership_keeps_wide_columns():
    frame = pd.DataFrame({'A': [1.]}, index=pd.to_datetime(['2026-01-05']))
    pools = StockPoolMembership({'2026-01-05': []})
    result = pools.filter_wide(frame)
    assert result.shape == frame.shape
    assert result.isna().all().all()
    assert pools.daily_frame(frame.index).empty


def test_duplicate_stock_columns_are_rejected():
    frame = pd.DataFrame([[1., 2.]], columns=['A', 'A'], index=pd.to_datetime(['2026-01-05']))
    with pytest.raises(ValueError, match='unique'):
        StockPoolMembership({'2026-01-05': ['A']}).filter_wide(frame)


@pytest.mark.parametrize('stocks', [
    [' A'], ['A '], ['\tA'], ['A\n'], ['A', ' A'], ('A', 'B\u00a0'),
])
def test_stock_codes_with_surrounding_whitespace_are_rejected(stocks):
    with pytest.raises(ValueError, match='whitespace'):
        StockPoolMembership({'2026-01-05': stocks})


@pytest.mark.parametrize('dates', [
    [20260105], [20260105.0], np.array([20260105], dtype=np.int64),
    pd.Index([20260105]), pd.Series([20260105]),
    [pd.Timestamp('2026-01-05'), 20260106], [True], [np.bool_(False)],
])
@pytest.mark.parametrize('method', ['filter_wide', 'coverage', 'daily_frame'])
def test_numeric_dates_raise_in_all_membership_consumers(dates, method):
    membership = StockPoolMembership({'2026-01-05': ['A']})
    value = pd.DataFrame({'A': np.ones(len(dates))}, index=dates) if method == 'filter_wide' else dates
    with pytest.raises(ValueError, match='numeric'):
        getattr(membership, method)(value)


@pytest.mark.parametrize('kind', ['datetime', 'strings', 'python_dates', 'numpy_dates', 'intraday'])
def test_valid_date_inputs_preserve_membership_and_original_panel(kind):
    dates = pd.to_datetime(['2026-01-04', '2026-01-05', '2026-01-07', '2026-01-08'])
    variants = {
        'datetime': dates,
        'strings': dates.strftime('%Y-%m-%d').tolist(),
        'python_dates': dates.date.tolist(),
        'numpy_dates': dates.to_numpy(),
        'intraday': dates + pd.Timedelta(hours=15),
    }
    membership = StockPoolMembership({'2026-01-08': ['B'], '2026-01-05': ['A'], '2026-01-07': []})
    frame = pd.DataFrame({'A': [1., 2., 3., 4.], 'B': [5., 6., 7., 8.]}, index=variants[kind])
    before = frame.copy(deep=True)
    expected = pd.DataFrame({'A': [np.nan, 2., np.nan, np.nan], 'B': [np.nan, np.nan, np.nan, 8.]}, index=frame.index)
    pd.testing.assert_frame_equal(membership.filter_wide(frame), expected)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(membership.coverage(variants[kind]), membership.coverage(dates))
    pd.testing.assert_frame_equal(membership.daily_frame(variants[kind]), membership.daily_frame(dates))
