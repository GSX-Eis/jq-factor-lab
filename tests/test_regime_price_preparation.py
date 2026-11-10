import duckdb
import pandas as pd
import pytest
from factor_synthesis.core.factor_data_loader import get_complete_price_data
from factor_synthesis.core.factor_data_loader import get_open_price_data, get_pricing_data
from calendar_test_support import seed_calendar


@pytest.mark.parametrize('reader', ['pricing', 'open', 'complete'])
@pytest.mark.parametrize('pit_enabled', [False, True])
def test_price_entrypoints_use_explicit_pit_switch(reader, pit_enabled):
    source = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-02', '2023-01-03']),
        'stock_code': ['A', 'A'],
        'open': [100., 50.],
        'close': [100., 50.],
        'factor': [1., 2.],
    })
    with duckdb.connect() as conn:
        conn.register('prices', source)
        seed_calendar(conn, '2023-01-02', '2023-01-03')
        if reader == 'pricing':
            # One future session includes the split date in the query window.
            result = get_pricing_data(conn, 'prices', ['A'],
                                      pd.DatetimeIndex(['2023-01-02']),
                                      pit_enabled=pit_enabled, future_periods=1)
            values = result.xs('A', level='asset').tolist()
        else:
            loader = get_open_price_data if reader == 'open' else get_complete_price_data
            result = loader(conn, 'prices', ['A'], '2023-01-02', '2023-01-03',
                            pit_enabled=pit_enabled)
            values = result['A'].tolist()
    assert values == ([50., 50.] if pit_enabled else [100., 50.])


@pytest.mark.parametrize("intraday", [False, True])
@pytest.mark.parametrize("pit", [False, True])
def test_price_preparation_preserves_daily_and_intraday_values(monkeypatch, intraday, pit):
    source = pd.DataFrame(
        {
            "stock_code": ["a", "a", "b", "b"],
            "date": ["2023-01-02", "2023-01-03", "2023-01-02", "2023-01-03"],
            "close": [10.0, 12.0, 20.0, 22.0],
            "factor": [1.0, 2.0, 1.0, 1.0],
        }
    )
    if intraday:
        early = source.iloc[[0]].copy()
        early["date"] = "2023-01-02 09:30:00"
        early["close"] = 9.0
        source["date"] = pd.to_datetime(source["date"]) + pd.Timedelta(hours=15)
        early["date"] = pd.to_datetime(early["date"])
        source = pd.concat([early, source], ignore_index=True).sort_values(["stock_code", "date"])
    source["date"] = pd.to_datetime(source["date"])
    with duckdb.connect() as conn:
        conn.register("prices", source)
        result = get_complete_price_data(conn, "prices", ["a", "b"], "2023-01-02", "2023-01-03", pit_enabled=pit)
    expected = pd.DataFrame(
        {"a": [5.0 if pit else 10.0, 12.0], "b": [20.0, 22.0]},
        index=pd.DatetimeIndex(["2023-01-02", "2023-01-03"], name="date"),
    )
    expected.columns.name = "stock_code"
    pd.testing.assert_frame_equal(result, expected)


def test_daily_price_detection_does_not_materialize_group_objects(monkeypatch):
    from pandas.core.groupby.generic import DataFrameGroupBy

    source = pd.DataFrame({"stock_code": ["a", "a"], "date": ["2023-01-02", "2023-01-03"], "close": [10.0, 12.0]})
    monkeypatch.setattr(DataFrameGroupBy, "__len__", lambda _self: pytest.fail("unneeded group object materialization"))
    source["date"] = pd.to_datetime(source["date"])
    with duckdb.connect() as conn:
        conn.register("prices", source)
        result = get_complete_price_data(conn, "prices", ["a"], "2023-01-02", "2023-01-03", pit_enabled=False)
    assert result.shape == (2, 1)


@pytest.mark.parametrize(
    "rows",
    [[], [("a", None)], [(None, None), (None, None)], [("a", "2023-01-02"), ("a", "2023-01-02"), (None, "2023-01-02")]],
)
def test_duplicate_detection_matches_old_null_and_empty_semantics(rows):
    frame = pd.DataFrame(rows, columns=["stock_code", "date_only"])
    frame["date_only"] = pd.to_datetime(frame["date_only"]).dt.date
    assert bool(frame.duplicated(["stock_code", "date_only"]).any()) == (
        len(frame) > len(frame.groupby(["stock_code", "date_only"]))
    )
