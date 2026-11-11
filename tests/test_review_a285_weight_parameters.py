"""A285: reject invalid active weight parameters before date-group dispatch."""
import numpy as np
import pandas as pd
import pytest

from factor_synthesis.pipeline.sample_weighting import compute_return_based_weights


INVALID_PARAMETERS = (
    [("return_magnitude", "cap", v) for v in (-0.1, -np.inf, np.nan)]
    + [("return_magnitude", "epsilon", v) for v in (-0.01, -np.inf, np.inf, np.nan)]
    + [("quantile", k, v) for k in ("top_pct", "bottom_pct")
       for v in (-0.1, 1.1, -np.inf, np.inf, np.nan)]
    + [("quantile", k, v) for k in ("top_weight", "mid_weight", "bottom_weight")
       for v in (-0.1, -np.inf, np.inf, np.nan)]
    + [("softmax", "temperature", np.nan)]
    + [("asymmetric_linear", k, v) for k in ("positive_scale", "negative_scale", "epsilon")
       for v in (-0.1, -np.inf, np.inf, np.nan)]
)


@pytest.mark.parametrize("method,key,value", INVALID_PARAMETERS)
@pytest.mark.parametrize("cross_sectional", [True, False])
def test_active_parameters_rejected_even_on_singleton_dates(method, key, value, cross_sectional):
    # Every date has one row, so strategy-local checks would otherwise be bypassed.
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(["2026-01-02", "2026-01-05"]), ["A", "B"]],
        names=["date", "stock_code"],
    )
    config = {"sample_weight_method": method,
              "sample_weight_config": {key: value, "cross_sectional": cross_sectional}}
    with pytest.raises(ValueError, match=key):
        compute_return_based_weights(np.array([0.1, -0.2]), index, config)


@pytest.mark.parametrize("method", [None, "none", "time_decay", "unknown"])
def test_disabled_or_unknown_strategy_does_not_validate_unused_parameters(method):
    assert compute_return_based_weights(
        None, None, {"sample_weight_method": method,
                     "sample_weight_config": {"epsilon": np.nan, "cap": -1}}
    ) is None


@pytest.mark.parametrize("method,config", [
    ("return_magnitude", {"cap": 0.0, "epsilon": 0.0}),
    ("quantile", {"top_weight": 0.0, "mid_weight": 0.0, "bottom_weight": 0.0}),
    ("asymmetric_linear", {"positive_scale": 0.0, "negative_scale": 0.0, "epsilon": 0.0}),
])
def test_legal_zero_weights_are_not_replaced(method, config):
    index = pd.MultiIndex.from_product([[pd.Timestamp("2026-01-02")], ["A", "B", "C"]])
    with np.errstate(invalid="ignore"):
        weights = compute_return_based_weights(
            np.array([-0.2, 0.0, 0.2]), index,
            {"sample_weight_method": method, "sample_weight_config": config},
        )
    np.testing.assert_array_equal(weights, np.zeros(3))


@pytest.mark.parametrize("method", ["return_magnitude", "quantile", "softmax", "asymmetric_linear"])
def test_inactive_strategy_parameters_are_ignored(method):
    index = pd.MultiIndex.from_product([[pd.Timestamp("2026-01-02")], ["A", "B", "C"]])
    y = np.array([-0.2, 0.0, 0.2])
    unused = {"return_magnitude": {"temperature": np.nan},
              "quantile": {"cap": -1.0, "epsilon": np.nan},
              "softmax": {"positive_scale": np.nan, "top_pct": np.nan},
              "asymmetric_linear": {"cap": np.nan, "temperature": np.nan}}
    expected = compute_return_based_weights(y, index, {"sample_weight_method": method})
    result = compute_return_based_weights(y, index, {"sample_weight_method": method,
                                                    "sample_weight_config": unused[method]})
    np.testing.assert_array_equal(result, expected)
