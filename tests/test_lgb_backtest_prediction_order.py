from __future__ import annotations

import numpy as np
import pandas as pd

from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner


class _Loader:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def load_factors(self, _date: str, stocks: list[str], factors: list[str]) -> pd.DataFrame:
        self.requested = list(stocks)
        values = {stock: float(int(stock[-1])) for stock in stocks}
        return pd.DataFrame({factors[0]: pd.Series(values)})


class _Booster:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return values[:, 0]


def test_lgb_prediction_canonicalizes_dynamic_pool_stock_order() -> None:
    runner = VectorBTBacktestRunner({"top_k_strategy": {}, "calendar_filters": {}})
    loader = _Loader()
    runner._factor_loader = loader
    runner._get_model_for_date = lambda _date: {
        "feature_names": ["factor"],
        "standardization_method": {"factor": "none"},
        "postprocess_cs_standardize": False,
    }
    runner._load_lgb_booster_for_date = lambda _date: _Booster()

    score, _, _ = runner._compute_lgb_prediction(
        "2024-01-02", ["S3", "S1", "S2", "S1"]
    )

    assert loader.requested == ["S1", "S2", "S3"]
    assert score.index.tolist() == ["S1", "S2", "S3"]


def test_shared_synthesis_score_cache_reuses_exact_result() -> None:
    shared_cache = {}
    expected_score = pd.Series([1.25, -0.5], index=["S1", "S2"])
    expected_processed = pd.DataFrame({"FactorA": [0.5, -0.5]}, index=["S1", "S2"])
    expected_raw = pd.DataFrame({"FactorA": [10.0, 20.0]}, index=["S1", "S2"])
    calls = 0

    first = VectorBTBacktestRunner({}, synthesis_score_cache=shared_cache)
    second = VectorBTBacktestRunner({}, synthesis_score_cache=shared_cache)

    def compute_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return expected_score, expected_processed, expected_raw

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("shared score cache was not used")

    first._compute_synthesis_score = compute_once
    second._compute_synthesis_score = fail_if_called

    actual_first = first._get_synthesis_score("2024-01-02", ["S1", "S2"])
    pd.testing.assert_series_equal(actual_first[0], expected_score, check_exact=True)
    pd.testing.assert_frame_equal(actual_first[1], expected_processed, check_exact=True)
    pd.testing.assert_frame_equal(actual_first[2], expected_raw, check_exact=True)
    actual_first[0].iloc[0] = 99.0
    actual_second = second._get_synthesis_score("2024-01-02", ["S1", "S2"])

    assert calls == 1
    pd.testing.assert_series_equal(actual_second[0], expected_score, check_exact=True)
    pd.testing.assert_frame_equal(
        actual_second[1], expected_processed, check_exact=True
    )
    pd.testing.assert_frame_equal(actual_second[2], expected_raw, check_exact=True)
