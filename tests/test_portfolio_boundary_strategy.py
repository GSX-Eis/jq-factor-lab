from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.preselection import portfolio_boundary_strategy as strategy_module
from factor_synthesis.preselection.portfolio_boundary_strategy import (
    PortfolioBoundaryReplacementStrategy,
)


def test_factor_direction_is_learned_before_sleeve_packing() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=3), ["a", "b"]],
        names=["date", "instrument"],
    )
    X = pd.DataFrame(
        {"positive": [1, 2, 1, 2, 1, 2], "negative": [2, 1, 2, 1, 2, 1]}, index=index
    )
    y = pd.Series(0.0, index=index)
    baseline = pd.Series(0.0, index=index)
    strategy = PortfolioBoundaryReplacementStrategy(
        {
            "baseline_factor": "Factor1",
            "sleeve_count": 1,
            "max_factors_per_sleeve": 2,
            "max_intra_sleeve_similarity": 1.0,
            "min_fold_observations": 2,
            "selection_fraction": 1.0,
            "fallback_on_missing_baseline": False,
        }
    )

    def score(signal: pd.Series, *_args: pd.Series) -> dict[str, float | int]:
        contribution = (
            1.0 if np.all(signal.to_numpy()[::2] < signal.to_numpy()[1::2]) else -1.0
        )
        return {
            "gross_replacement_delta": contribution,
            "candidate_turnover": 0.0,
            "baseline_turnover": 0.0,
            "incremental_turnover": 0.0,
            "net_contribution": contribution,
            "rebalance_observations": 2,
            "complete_input_dates": 3,
        }

    strategy._score_signal = score  # type: ignore[method-assign]

    assert strategy.select(X, y, baseline_scores=baseline) == ["negative", "positive"]
    report = strategy.get_performance_report().set_index("name")
    assert report.loc["negative", "direction"] == -1
    assert report.loc["positive", "direction"] == 1


def test_no_positive_contribution_returns_empty_selection() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=2), ["a", "b"]],
        names=["date", "instrument"],
    )
    X = pd.DataFrame({"negative": [1, 2, 2, 1]}, index=index)
    y = pd.Series(0.0, index=index)
    strategy = PortfolioBoundaryReplacementStrategy(
        {
            "baseline_factor": "Factor1",
            "min_fold_observations": 2,
            "fallback_on_missing_baseline": False,
        }
    )
    strategy._score_signal = lambda *_args: {  # type: ignore[method-assign]
        "gross_replacement_delta": -1.0,
        "candidate_turnover": 0.0,
        "baseline_turnover": 0.0,
        "incremental_turnover": 0.0,
        "net_contribution": -1.0,
        "rebalance_observations": 2,
        "complete_input_dates": 2,
    }

    assert strategy.select(X, y, baseline_scores=y) == []


def test_mandatory_factor_is_kept_outside_top_sleeve() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=2), ["a", "b"]],
        names=["date", "instrument"],
    )
    X = pd.DataFrame(
        {"mandatory": [1, 2, 1, 2], "optional": [2, 1, 2, 1]}, index=index
    )
    y = pd.Series(0.0, index=index)
    strategy = PortfolioBoundaryReplacementStrategy(
        {
            "baseline_factor": "Factor1",
            "sleeve_count": 2,
            "max_factors_per_sleeve": 1,
            "min_fold_observations": 2,
            "selection_fraction": 0.5,
            "mandatory_factors": ["mandatory"],
        }
    )

    def score(signal: pd.Series, *_args: pd.Series) -> dict[str, float | int]:
        contribution = 2.0 if abs(signal.iloc[0]) == 1.0 else -1.0
        return {
            "gross_replacement_delta": contribution,
            "candidate_turnover": 0.0,
            "baseline_turnover": 0.0,
            "incremental_turnover": 0.0,
            "net_contribution": contribution,
            "rebalance_observations": 2,
            "complete_input_dates": 2,
        }

    strategy._score_signal = score  # type: ignore[method-assign]

    assert strategy.select(X, y, baseline_scores=y) == ["optional", "mandatory"]


def test_missing_parent_history_can_return_empty_selection() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=2), ["a", "b"]],
        names=["date", "instrument"],
    )
    X = pd.DataFrame({"factor": [1, 2, 1, 2]}, index=index)
    y = pd.Series(0.0, index=index)
    strategy = PortfolioBoundaryReplacementStrategy(
        {
            "baseline_factor": "parent",
            "empty_on_missing_baseline": True,
        }
    )

    assert strategy.select(X, y, baseline_scores=None) == []
    assert strategy.get_performance_report().attrs["fallback_reason"] == (
        "baseline scores unavailable"
    )


def test_residual_gamma_scores_the_deployed_parent_blend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=2), ["a", "b"]],
        names=["date", "instrument"],
    )
    signal = pd.Series([1.0, 2.0, 1.0, 2.0], index=index)
    returns = pd.Series(0.0, index=index)
    baseline = pd.Series([0.2, -0.2, 0.3, -0.3], index=index)
    seen: list[np.ndarray] = []
    dates = pd.DatetimeIndex(index.get_level_values("date").unique())
    monkeypatch.setattr(
        strategy_module, "prepare_top10_context", lambda *_args, **_kwargs: None
    )

    def utility(values, *_args, **_kwargs):
        seen.append(np.asarray(values))
        return 0.0, pd.Series([0.0, 0.0], index=dates), 0.0

    monkeypatch.setattr(strategy_module, "top10_net_utility", utility)
    strategy = PortfolioBoundaryReplacementStrategy(
        {
            "baseline_factor": "parent",
            "residual_gamma": 0.1,
            "min_fold_observations": 2,
        }
    )

    strategy._score_signal(signal, returns, baseline)

    residual = signal.groupby(level="date").transform(
        lambda values: (values - values.mean()) / (values.std() + 1e-8)
    )
    np.testing.assert_allclose(seen[0], (baseline + 0.1 * residual).to_numpy())
    np.testing.assert_allclose(seen[1], baseline.to_numpy())
