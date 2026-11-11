import numpy as np
import pandas as pd

from factor_mining.symbolic.objective import rank_ic_by_date
from factor_mining.symbolic.operator_runtime import runtime_namespace


def test_rank_ic_by_date_accepts_object_numeric_predictions():
    dates = pd.Series(pd.to_datetime(["2021-01-01"] * 4))
    pred = pd.Series([0, 1, 0, 1], dtype=object)
    label = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = rank_ic_by_date(pred, label, dates)

    assert len(result) == 1
    assert np.isfinite(result.iloc[0])


def test_numeric_negation_accepts_boolean_expression_output():
    values = pd.Series([True, False], dtype=object)

    result = runtime_namespace({}, pd.Series([0, 0]), pd.Series(["a", "b"]))[
        "neg"
    ](values)

    assert result.tolist() == [-1.0, 0.0]
