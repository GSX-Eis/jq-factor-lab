"""Input and mutation contracts for shared numeric factor helpers."""
import numpy as np
import pandas as pd
import pytest

from utils.factor_base import FactorBase


@pytest.mark.parametrize('window', [0, -1, 1.5, 2.0, True, np.bool_(True), None])
def test_rolling_rejects_invalid_window(window):
    with pytest.raises(ValueError, match='positive integer'):
        FactorBase.vectorized_rolling_window(np.arange(4.), window, np.mean)


@pytest.mark.parametrize('values', [np.array(2.), np.ones((3, 2))])
def test_rolling_rejects_non_vector_input(values):
    with pytest.raises(ValueError, match='one-dimensional'):
        FactorBase.vectorized_rolling_window(values, 2, np.mean)


@pytest.mark.parametrize('values', [np.arange(7.), np.arange(7.)[::-1], np.arange(12.)[::2]])
def test_rolling_matches_explicit_windows(values):
    actual = FactorBase.vectorized_rolling_window(values, np.int64(3), np.mean)
    expected = np.full(len(values), np.nan)
    expected[2:] = [np.mean(values[i-2:i+1]) for i in range(2, len(values))]
    np.testing.assert_array_equal(actual, expected)


def test_rolling_callback_cannot_mutate_input():
    data = np.arange(5.)
    before = data.copy()
    def mutate(window):
        window[0] = 999.
        return window.mean()
    with pytest.raises(ValueError, match='read-only'):
        FactorBase.vectorized_rolling_window(data, 2, mutate)
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize('count', [0, 2])
def test_short_history_returns_nan_without_calling_callback(count):
    def unexpected(window):
        raise AssertionError('no complete window')
    result = FactorBase.vectorized_rolling_window(np.arange(count), 3, unexpected)
    assert result.shape == (count,)
    assert np.isnan(result).all()


def test_sorting_rechecks_order_after_marker_is_inherited():
    original = pd.DataFrame({'code': ['B', 'A', 'A'], 'date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-02']), 'value': [1.1234567890123, 2., 3.]})
    ordered = FactorBase.optimize_dataframe_operations(None, original)
    reordered = ordered.iloc[::-1].copy()
    assert reordered.attrs['_sorted_by_code_date']
    untouched = reordered.copy(deep=True)
    actual = FactorBase.optimize_dataframe_operations(None, reordered)
    pd.testing.assert_frame_equal(actual, ordered, check_exact=True)
    pd.testing.assert_frame_equal(reordered, untouched, check_exact=True)


def test_sorting_preserves_tie_order_and_numeric_precision():
    data = pd.DataFrame({'code': ['A', 'A'], 'date': pd.to_datetime(['2024-01-02']*2), 'value': [1.1234567890123, 1.1234567890124]}, index=[8, 3])
    actual = FactorBase.optimize_dataframe_operations(None, data)
    pd.testing.assert_frame_equal(actual, data.reset_index(drop=True), check_exact=True)


def test_rolling_does_not_require_new_numpy_window_api(monkeypatch):
    monkeypatch.delattr(np.lib.stride_tricks, 'sliding_window_view', raising=False)
    result = FactorBase.vectorized_rolling_window(np.array([1., 2., 3.]), 2, np.mean)
    np.testing.assert_array_equal(result, np.array([np.nan, 1.5, 2.5]))
