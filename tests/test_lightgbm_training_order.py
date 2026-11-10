from __future__ import annotations

import pandas as pd

from factor_synthesis.pipeline.lightgbm_method import _prepare_lgb_data


def test_prepare_lgb_data_sorts_date_and_instrument() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-03"), "000003.XSHE"),
            (pd.Timestamp("2024-01-02"), "000002.XSHE"),
            (pd.Timestamp("2024-01-02"), "000001.XSHE"),
            (pd.Timestamp("2024-01-03"), "000001.XSHE"),
        ],
        names=["date", "stock_code"],
    )
    features = pd.DataFrame({"factor": [3.0, 2.0, 1.0, 4.0]}, index=index)
    target = pd.Series([0.3, 0.2, 0.1, 0.4], index=index)

    sorted_features, sorted_target, groups = _prepare_lgb_data(
        features, target, {"objective": "regression"}
    )

    expected = index.sort_values()
    assert sorted_features.index.equals(expected)
    assert sorted_target.index.equals(expected)
    assert groups is None
