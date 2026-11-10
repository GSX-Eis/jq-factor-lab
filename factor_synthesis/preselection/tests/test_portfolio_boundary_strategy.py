import logging

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.pipeline.lasso_method import _resolve_inner_train_valid_dates
from factor_synthesis.preselection.portfolio_boundary_strategy import (
    PortfolioBoundaryReplacementStrategy,
)


def _panel() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.to_datetime(["2023-01-02", "2023-01-09", "2023-01-16"])
    stocks = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])
    baseline = pd.Series([4.0, 3.0, 2.0, 1.0] * 3, index=index)
    stable = [1.0, 2.0, 4.0, 3.0] * 3
    switching = [1.0, 2.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0] + [1.0, 2.0, 4.0, 3.0]
    X = pd.DataFrame(
        {
            "stable": stable,
            "switching": switching,
            "baseline_like": baseline.to_numpy(),
        },
        index=index,
    )
    returns = pd.Series([0.0, 0.0, 0.1, 0.1] * 3, index=index)
    return X, returns, baseline


def _config(**overrides):
    config = {
        "enabled": True,
        "baseline_factor": "DemoBaseline",
        "top_k": 2,
        "sleeve_count": 3,
        "max_factors_per_sleeve": 1,
        "sleeve_similarity_metric": "spearman_rank",
        "max_intra_sleeve_similarity": 0.8,
        "replacement_score": "net_topk_replacement_vs_baseline",
        "turnover_cost_rate": 0.03,
        "holding_period": 1,
        "min_fold_observations": 3,
        "selection_fraction": 1 / 3,
        "fallback_on_missing_baseline": False,
    }
    config.update(overrides)
    return config


def test_exact_replacement_score_and_deterministic_bounded_sleeves():
    X, returns, baseline = _panel()
    strategy = PortfolioBoundaryReplacementStrategy(_config())

    # Inverted baseline ties with stable; deterministic name ordering breaks the tie.
    assert strategy.select(X, returns, baseline_scores=baseline) == ["baseline_like"]
    assert strategy.select(X, returns, baseline_scores=baseline) == ["baseline_like"]

    report = strategy.get_performance_report().set_index(["record_type", "name"])
    assert report.loc[("factor", "baseline_like"), "direction"] == -1
    stable = report.loc[("factor", "stable")]
    switching = report.loc[("factor", "switching")]
    assert stable["gross_replacement_delta"] == pytest.approx(0.1)
    assert stable["incremental_turnover"] == pytest.approx(0.0)
    assert stable["net_contribution"] == pytest.approx(0.1)
    assert switching["gross_replacement_delta"] == pytest.approx(2 / 30)
    assert switching["incremental_turnover"] == pytest.approx(2 / 3)
    assert switching["net_contribution"] == pytest.approx(2 / 30 - 0.03 * 2 / 3)
    sleeves = report.loc["sleeve"]
    assert len(sleeves) == 3
    assert all(len(members) == 1 for members in sleeves["members"])


def test_cutoff_helper_and_missing_baseline_policy():
    dates = pd.bdate_range("2023-01-02", periods=12)
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B"]], names=["date", "stock_code"]
    )
    X = pd.DataFrame({"factor": np.arange(len(index))}, index=index)
    train_end, valid_start, valid_end = _resolve_inner_train_valid_dates(
        X,
        {"lasso_validation_fraction": 0.2, "target_period": 2,
         "_label_exit_dates": {
             str(day.date()): str((day + pd.offsets.BDay(2)).date())
             for day in dates
         }},
        logging.getLogger(__name__),
    )
    assert (train_end, valid_start, valid_end) == (
        "2023-01-10",
        "2023-01-13",
        "2023-01-17",
    )

    panel, returns, _ = _panel()
    with pytest.raises(ValueError, match="baseline scores unavailable"):
        PortfolioBoundaryReplacementStrategy(_config()).select(panel, returns)
    fallback = PortfolioBoundaryReplacementStrategy(
        _config(fallback_on_missing_baseline=True, selection_fraction=0.5)
    )
    assert fallback.select(panel, returns) == ["stable", "switching"]
    assert (
        fallback.get_performance_report().attrs["fallback_reason"]
        == "baseline scores unavailable"
    )


def test_minimum_counts_complete_input_dates_not_weekly_rebalances():
    dates = pd.bdate_range("2023-01-02", periods=60)
    stocks = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product(
        [dates, stocks], names=["date", "stock_code"]
    )
    X = pd.DataFrame({"stable": [1.0, 2.0, 4.0, 3.0] * len(dates)}, index=index)
    returns = pd.Series([0.0, 0.0, 0.1, 0.1] * len(dates), index=index)
    baseline = pd.Series([4.0, 3.0, 2.0, 1.0] * len(dates), index=index)
    config = _config(
        min_fold_observations=60,
        sleeve_count=1,
        selection_fraction=1.0,
    )

    strategy = PortfolioBoundaryReplacementStrategy(config)
    assert strategy.select(X, returns, baseline_scores=baseline) == ["stable"]
    report = strategy.get_performance_report()
    factor = report[report["record_type"] == "factor"].iloc[0]
    assert report.attrs["input_date_minimum"] == 60
    assert factor["complete_input_dates"] == 60
    assert 2 <= factor["rebalance_observations"] < 60

    too_short = index.get_level_values("date") < dates[-1]
    with pytest.raises(ValueError, match="insufficient complete input dates"):
        PortfolioBoundaryReplacementStrategy(config).select(
            X.loc[too_short],
            returns.loc[too_short],
            baseline_scores=baseline.loc[too_short],
        )
