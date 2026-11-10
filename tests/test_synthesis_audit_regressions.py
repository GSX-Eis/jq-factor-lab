"""Regression coverage for the round-two synthesis audit fixes."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from factor_synthesis.diagnostics.multicollinearity_checker import compute_condition_number
from factor_synthesis.factor_composer import FactorComposer, SynthesisConfig
from factor_synthesis.pipeline.dynamic_blend import (
    DynamicBlendConfig,
    blend_predictions,
    compute_dynamic_blend_ratio,
    compute_rolling_ic_stats,
)


def test_factor_composer_uses_forward_return_at_signal_date_including_friday() -> None:
    composer = FactorComposer(SynthesisConfig(method="ridge", prediction_window=1), factor_reader=None)
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    factors = pd.DataFrame({"date": dates, "stock": ["A", "A"], "factor": [1.0, 2.0]})
    labels = pd.DataFrame({"A": [7 / 9, 9 / 16]}, index=dates)

    aligned = composer._align_target_returns(factors, labels)

    assert aligned.target_return.tolist() == pytest.approx([7 / 9, 9 / 16])


def test_factor_composer_evaluation_does_not_shift_forward_labels_again() -> None:
    composer = FactorComposer(SynthesisConfig(method="ridge", prediction_window=1), factor_reader=None)
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    composer.synthesis_results["composite"] = {
        "factor_data": pd.DataFrame({"A": [1.0, 2.0]}, index=dates)
    }

    metrics = composer.evaluate_performance("composite", pd.DataFrame({"A": [0.1, 0.2]}, index=dates))

    assert metrics["mse"] == pytest.approx(2.025)


@pytest.mark.parametrize(
    ("use_icir", "correlation_aware"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_dynamic_blend_preserves_negative_rank_ic_and_favors_positive_signal(
    use_icir: bool, correlation_aware: bool
) -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    stocks = [f"S{i:02d}" for i in range(10)]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])
    forward_returns = pd.Series(np.tile(np.arange(10, dtype=float), 2), index=index)
    ml_predictions = forward_returns.copy()
    prior_composite = -forward_returns
    config = DynamicBlendConfig(
        base_ratio=0.5,
        max_delta=0.3,
        lookback_window=2,
        min_samples=2,
        use_icir=use_icir,
        correlation_aware=correlation_aware,
    )

    prior_ic, _, _ = compute_rolling_ic_stats(prior_composite, forward_returns, window=2, min_periods=2)
    result = compute_dynamic_blend_ratio(ml_predictions, prior_composite, forward_returns, config)
    blended = blend_predictions(ml_predictions, prior_composite, result.blend_ratio)

    assert prior_ic == pytest.approx(-1.0)
    assert result.ml_ic == pytest.approx(1.0)
    assert result.prior_ic == pytest.approx(-1.0)
    assert result.blend_ratio < config.base_ratio
    assert not np.allclose(blended.to_numpy(), 0.0)


def test_forward_returns_are_order_independent_and_validate_horizons() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    index = pd.MultiIndex.from_product([["A", "B"], dates], names=["asset", "date"])
    pricing = pd.Series([10, 11, 12, 20, 22, 24], index=index, dtype=float)
    from utils.trading_calendar import TradingCalendar
    calculator = ForwardReturnsCalculator(trading_calendar=TradingCalendar(dates, dates.min(), dates.max()))

    expected = calculator.calculate_forward_returns(pricing, periods=(1,), execution_lag=0)
    actual = calculator.calculate_forward_returns(pricing.sample(frac=1.0, random_state=7), periods=(1,), execution_lag=0)

    pd.testing.assert_frame_equal(actual, expected)
    for bad_periods, bad_lag, message in [((0,), 0, "periods"), ((-1,), 0, "periods"), ((1,), -1, "execution_lag")]:
        with pytest.raises(ValueError, match=message):
            calculator.calculate_forward_returns(pricing, periods=bad_periods, execution_lag=bad_lag)


def test_forward_returns_reject_duplicate_price_keys() -> None:
    index = pd.MultiIndex.from_tuples(
        [("A", "2024-01-02"), ("A", "2024-01-02")], names=["asset", "date"]
    )
    with pytest.raises(ValueError, match="重复"):
        ForwardReturnsCalculator().calculate_forward_returns(pd.Series([10.0, 11.0], index=index), periods=(1,))


def test_forward_returns_reject_null_asset_keys() -> None:
    index = pd.MultiIndex.from_tuples(
        [("A", "2024-01-02"), (None, "2024-01-03")], names=["asset", "date"]
    )
    with pytest.raises(ValueError, match="空的 asset"):
        ForwardReturnsCalculator().calculate_forward_returns(pd.Series([10.0, 11.0], index=index), periods=(1,))


def test_rank_deficient_condition_number_is_infinite() -> None:
    values = np.arange(20, dtype=float)
    factor_df = pd.DataFrame({"left": values, "right": values})

    condition_number = compute_condition_number(factor_df, ["left", "right"], 100, logging.getLogger(__name__))

    assert np.isinf(condition_number)
