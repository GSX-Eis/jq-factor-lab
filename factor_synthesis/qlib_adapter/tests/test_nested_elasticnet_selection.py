import logging

import numpy as np
import pandas as pd

from factor_synthesis.preselection.stability_selection_strategy import (
    StabilitySelectionPreStrategy,
)
from factor_synthesis.qlib_adapter.qlib_linear_model import QlibLinearWrapper
from run_factor_synthesis import _lasso_outer_fold_diagnostics


class _NegMSE:
    def evaluate(self, prediction, target, _dates):
        return -float(np.mean((prediction - target) ** 2))


def test_contiguous_stability_blocks_keep_whole_dates():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=10), ["a", "b"]],
        names=["date", "stock"],
    )
    X = pd.DataFrame({"f": np.arange(len(index))}, index=index)
    strategy = StabilitySelectionPreStrategy(
        {
            "sampling_mode": "contiguous_time_blocks",
            "parallel": {"enabled": False},
        }
    )
    sampled = []

    def capture(_X, _y, indices, *_args):
        sampled.append(indices)
        return ["f"], {}

    strategy._run_single_bootstrap = capture
    strategy._run_stability_selection(
        X,
        pd.Series(np.arange(len(index)), index=index),
        "ic",
        n_bootstrap=3,
        subsample_ratio=0.4,
        selection_mode="top_k",
        per_sample_top_k=1,
        per_sample_top_pct=1.0,
        per_sample_metric_threshold=0.0,
    )

    assert [len(indices) for indices in sampled] == [8, 8, 8]
    assert all(
        len(pd.Index(index[indices].get_level_values("date")).unique()) == 4
        for indices in sampled
    )


def test_inner_cv_scores_regularization_and_factor_count_together():
    rng = np.random.default_rng(7)
    dates = np.repeat(pd.date_range("2024-01-01", periods=12), 3)
    index = pd.MultiIndex.from_arrays(
        [dates, np.tile(["a", "b", "c"], 12)],
        names=["datetime", "instrument"],
    )
    X = rng.normal(size=(len(index), 3))
    y = X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.01, size=len(index))
    wrapper = QlibLinearWrapper(
        {
            "model_type": "elasticnet",
            "cv_n_jobs": 1,
            "cv_warm_start": False,
            "lasso_max_iter": 2000,
        },
        logging.getLogger(__name__),
    )
    scores = wrapper._score_param_grid_cv(
        X,
        y,
        index,
        None,
        [(np.arange(0, 24), np.arange(24, 36))],
        [(0.001, 0.5)],
        _NegMSE(),
        factor_counts=[1, 2],
    )

    assert set(scores) == {(0.001, 0.5, 1), (0.001, 0.5, 2)}


def test_outer_diagnostics_report_worst_untouched_fold(tmp_path):
    path = tmp_path / "returns.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6),
            "net_return": [0.02, 0.01, 0.0, -0.03, -0.02, 0.0],
        }
    ).to_csv(path, index=False)

    result = _lasso_outer_fold_diagnostics(
        [
            {"validation_start": "2024-01-01", "validation_end": "2024-01-03"},
            {"validation_start": "2024-01-04", "validation_end": "2024-01-06"},
        ],
        [
            {
                "annual_return": -0.1,
                "max_drawdown": -0.05,
                "annual_turnover": 8.0,
                "return_series_artifact": {"path": str(path)},
            }
        ],
    )

    assert result["outer_fold_count"] == 2
    assert result["worst_outer_fold"]["fold"] == 2
    assert result["annual_turnover"] == 8.0
