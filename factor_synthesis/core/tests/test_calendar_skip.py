"""Tests for calendar skip rules (whole months + MM-DD:MM-DD windows)."""

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.core.calendar_skip import (
    date_in_skip,
    describe_skip_rules,
    parse_skip_rules,
    skip_mask,
)
from factor_synthesis.core.pool_manager import apply_skip_months


def test_parse_backward_compatible_months():
    assert parse_skip_rules([1, 4, 12]) == {1, 4, 12}
    assert parse_skip_rules(["1", "4"]) == {1, 4}


def test_parse_windows_and_mixed():
    rules = parse_skip_rules(["01-16:02-05", "4-16:4-30", 12])
    assert rules == {"01-16:02-05", "04-16:04-30", 12}


@pytest.mark.parametrize('item', [0, 13, -1, "13-01:01-05", "garbage", None,
    True, False, 1.9, 1.0, np.float64(2), np.bool_(True), [], {},
    "02-30:03-01", "04-31:05-01", "01-01:02-30", "00-01:01-05", ""])
def test_parse_rejects_invalid_entry(item):
    with pytest.raises(ValueError, match=r"skip_months\[1\]"):
        parse_skip_rules([4, item])


@pytest.mark.parametrize('raw', ["12", "", b"12", {}, {1: 2}, 12, 0, False, 0.0])
def test_parse_rejects_invalid_container(raw):
    with pytest.raises(TypeError, match='skip_months'):
        parse_skip_rules(raw)


@pytest.mark.parametrize('raw', [None, [], (), set(), frozenset()])
def test_parse_empty_rules(raw):
    assert parse_skip_rules(raw) == set()


def test_parse_preserves_iterables_and_leap_day():
    for raw in [[np.int64(12), " 4 ", "2-29:3-1", "12-20:1-10", 12],
                (12, 4, "02-29:03-01", "12-20:01-10"),
                iter([12, 4, "02-29:03-01", "12-20:01-10"])]:
        assert parse_skip_rules(raw) == {12, 4, "02-29:03-01", "12-20:01-10"}
    dates = pd.to_datetime(['2023-02-28', '2023-03-01', '2024-02-29'])
    assert list(skip_mask(dates, parse_skip_rules(['02-29:02-29']))) == [False, False, True]


def test_skip_mask_window_inclusive_bounds():
    rules = parse_skip_rules(["01-16:02-05", "04-16:04-30"])
    dates = pd.to_datetime(
        ["2024-01-15", "2024-01-16", "2024-02-05", "2024-02-06", "2024-04-15", "2024-04-16", "2024-04-30", "2024-05-01"]
    )
    assert list(skip_mask(dates, rules)) == [False, True, True, False, False, True, True, False]


def test_skip_mask_whole_month_matches_legacy():
    rules = parse_skip_rules([1, 4, 12])
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    expected = dates.month.isin([1, 4, 12])
    assert (skip_mask(dates, rules) == np.asarray(expected)).all()


def test_skip_mask_year_wrap_window():
    rules = parse_skip_rules(["12-20:01-10"])
    dates = pd.to_datetime(["2023-12-19", "2023-12-20", "2024-01-10", "2024-01-11", "2024-06-15"])
    assert list(skip_mask(dates, rules)) == [False, True, True, False, False]


def test_date_in_skip_and_describe():
    rules = parse_skip_rules([12, "04-16:04-30"])
    assert date_in_skip(pd.Timestamp("2024-04-20"), rules)
    assert not date_in_skip(pd.Timestamp("2024-04-10"), rules)
    assert describe_skip_rules(rules) == "[12, 04-16:04-30]"
    assert describe_skip_rules(set()) == "none"


def test_apply_skip_months_with_windows():
    idx = pd.date_range("2024-04-01", "2024-05-10", freq="D")
    series = pd.Series(1.0, index=idx)
    out = apply_skip_months(series, parse_skip_rules(["04-16:04-30"]))
    assert out.index.min() == pd.Timestamp("2024-04-01")
    assert not any((d.month == 4 and d.day >= 16) for d in out.index)
    assert pd.Timestamp("2024-05-01") in out.index


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


@pytest.mark.parametrize('index', [
    pd.Index(['2026-01-05', 'bad-date']),
    pd.DatetimeIndex(['2026-01-05', None]),
    pd.Index([20260105, 20260205]),
    pd.Index(['2026-01-05', 20260205], dtype=object),
    pd.Index([1.0, 2.0]),
    pd.Index([True, False]),
    pd.MultiIndex.from_tuples([('2026-01-05', 'A')], names=['day', 'stock_code']),
    pd.MultiIndex.from_tuples([(20260105, 'A')], names=['date', 'stock_code']),
    pd.MultiIndex.from_tuples([(None, 'A')], names=['date', 'stock_code']),
])
def test_apply_skip_rejects_invalid_dates(index):
    with pytest.raises((ValueError, TypeError)):
        apply_skip_months(pd.Series(1., index=index), {1})


@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('frame', [False, True])
@pytest.mark.parametrize('date_kind', ['datetime', 'string', 'date', 'tz'])
def test_apply_skip_preserves_valid_rows_and_input(multi, frame, date_kind):
    dates = pd.DatetimeIndex(['2026-02-05', '2026-01-05', '2026-02-05'])
    if date_kind == 'string': dates = dates.strftime('%Y-%m-%d')
    elif date_kind == 'date': dates = pd.Index(dates.date)
    elif date_kind == 'tz': dates = dates.tz_localize('Asia/Shanghai')
    index = pd.MultiIndex.from_arrays([dates, ['B', 'A', 'B']], names=['date', 'stock_code']) if multi else dates
    obj = pd.Series([3., 1., 2.], index=index, name='signal')
    if frame: obj = obj.to_frame()
    original = obj.copy(deep=True)
    result = apply_skip_months(obj, {1})
    compare = pd.testing.assert_frame_equal if frame else pd.testing.assert_series_equal
    compare(result, original.iloc[[0, 2]])
    compare(obj, original)


def test_apply_skip_noop_and_all_removed():
    assert apply_skip_months(None, {1}) is None
    empty = pd.DataFrame()
    assert apply_skip_months(empty, {1}) is empty
    obj = pd.Series([1.], index=pd.to_datetime(['2026-01-05']))
    assert apply_skip_months(obj, set()) is obj
    assert apply_skip_months(obj, {1}).empty


def test_apply_skip_rejects_unsupported_object():
    with pytest.raises(TypeError, match='DataFrame|Series'):
        apply_skip_months([1, 2], {1})
