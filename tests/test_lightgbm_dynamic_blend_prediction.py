from __future__ import annotations

import numpy as np
import pandas as pd

from factor_synthesis.pipeline.lightgbm_method import _predict_lightgbm_in_sample


class _RawModel:
    def __init__(self) -> None:
        self.seen_values = None

    def predict(self, values):
        self.seen_values = values.copy()
        return values[:, 0] * 2.0 + values[:, 1]


class _WrapperWithDatasetPredict:
    feature_names = ["b", "a"]

    def __init__(self) -> None:
        self.model = _RawModel()
        self.qlib_model = None

    def predict(self, dataset, segment="test"):
        raise AssertionError("Dataset-only wrapper predict should not be used here")


def test_predict_lightgbm_in_sample_uses_raw_model_with_feature_order() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["000001", "000002"]],
        names=["date", "stock_code"],
    )
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        },
        index=index,
    )
    wrapper = _WrapperWithDatasetPredict()

    predictions = _predict_lightgbm_in_sample(wrapper, X)

    np.testing.assert_allclose(wrapper.model.seen_values, X[["b", "a"]].values)
    np.testing.assert_allclose(predictions.values, np.array([21.0, 42.0, 63.0, 84.0]))
    assert predictions.index.equals(index)
