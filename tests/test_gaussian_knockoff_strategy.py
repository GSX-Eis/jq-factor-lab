import numpy as np
import pandas as pd

from factor_synthesis.preselection.gaussian_knockoff_strategy import (
    GaussianKnockoffPreSelectionStrategy,
    _knockoff_plus_threshold,
)
from factor_synthesis.preselection.time_series_knockoff_strategy import (
    TimeSeriesKnockoffPreSelectionStrategy,
)


def test_knockoff_plus_threshold_controls_empty_and_selected_cases() -> None:
    statistics = np.array([3.0, 2.0, 1.0, -0.2])
    assert _knockoff_plus_threshold(statistics, 0.5) == 1.0
    assert np.isinf(_knockoff_plus_threshold(statistics, 0.2))


def test_gaussian_knockoff_records_empty_selection_diagnostics() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=24, freq="B")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "code"])
    X = pd.DataFrame(rng.normal(size=(len(index), 6)), index=index)
    y = pd.Series(rng.normal(size=len(index)), index=index)

    selector = GaussianKnockoffPreSelectionStrategy(
        {"fdr": 0.01, "cv_folds": 3, "purge_dates": 1, "random_seed": 3}
    )
    selector.select(X, y)

    summary = selector.get_performance_report().attrs
    assert summary["strategy"] == "gaussian_knockoff"
    assert summary["positive_count"] + summary["negative_count"] + summary["zero_count"] == 6
    assert "threshold" in summary
    assert summary["lasso_alpha"] > 0


def test_time_series_knockoff_aggregates_subsample_e_values(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "code"])
    X = pd.DataFrame(np.arange(len(index) * 2).reshape(len(index), 2), index=index)
    y = pd.Series(np.arange(len(index)), index=index)

    def fake_select(self, X, y):
        selected = [X.columns[0]]
        self._performance_report = pd.DataFrame(
            {
                "factor_name": X.columns,
                "knockoff_statistic": [1.0, -1.0],
                "selected": [True, False],
            }
        )
        self._performance_report.attrs.update(
            {"negative_count": 0, "selected_count": 1}
        )
        return selected

    monkeypatch.setattr(GaussianKnockoffPreSelectionStrategy, "select", fake_select)
    selector = TimeSeriesKnockoffPreSelectionStrategy(
        {"subsampling_q": 1, "subsample_fdr": 0.1, "fdr": 1.0}
    )

    assert selector.select(X, y) == [0]
    report = selector.get_performance_report()
    assert report.attrs["group_count"] == 2
    assert len(report.attrs["groups"][0]["factor_statistics"]) == 2
    assert report.loc[report["factor_name"] == 0, "subsample_selected_count"].item() == 2
