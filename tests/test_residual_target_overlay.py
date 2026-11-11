import hashlib

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.pipeline.lasso_method import (
    IncompleteResearchWindowError,
    _maybe_residualize_training_target,
    _run_blend_only_mode,
    _require_complete_research_window,
    _target_residual_baseline,
    run_lasso_synthesis,
)
from factor_synthesis.pipeline.preprocessing import residualize_target_by_factor
from factor_synthesis.qlib_adapter.cv_metrics import CVMetricEvaluator
from factor_synthesis.qlib_adapter.qlib_linear_model import QlibLinearWrapper


def test_residual_target_removes_daily_size_exposure():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    stocks = [f"S{i:02d}" for i in range(12)]
    index = pd.MultiIndex.from_product(
        [dates, stocks], names=["date", "stock_code"]
    )
    baseline = pd.Series(np.tile(np.arange(12, dtype=float), 2), index=index)
    idiosyncratic = pd.Series(
        np.tile(np.array([-1.0, 1.0] * 6), 2), index=index
    )
    target = 0.03 * baseline + idiosyncratic

    residual = residualize_target_by_factor(target, baseline)

    for date in dates:
        day = residual.xs(date, level="date")
        size = baseline.xs(date, level="date")
        assert day.mean() == pytest.approx(0.0)
        assert day.corr(size) == pytest.approx(0.0, abs=1e-12)


def test_training_target_ablation_keeps_baseline_for_selection_without_residualizing():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), [f"S{i:02d}" for i in range(12)]],
        names=["date", "stock_code"],
    )
    baseline = pd.Series(np.tile(np.arange(12, dtype=float), 2), index=index)
    target = 0.02 * baseline + pd.Series(
        np.tile(np.array([-1.0, 1.0] * 6), 2), index=index
    )

    unchanged, applied = _maybe_residualize_training_target(
        target,
        baseline,
        {"residualize_training_target": False, "min_samples": 10},
    )
    residual, residual_applied = _maybe_residualize_training_target(
        target,
        baseline,
        {"residualize_training_target": True, "min_samples": 10},
    )

    pd.testing.assert_series_equal(unchanged, target)
    assert applied is False
    assert residual_applied is True
    for date in pd.to_datetime(["2025-01-02", "2025-01-03"]):
        assert residual.xs(date, level="date").corr(
            baseline.xs(date, level="date")
        ) == pytest.approx(0.0, abs=1e-12)


def test_composite_anchor_can_be_the_residualization_baseline():
    class Synthesizer:
        @staticmethod
        def _apply_feature_preprocessing(frame, _config):
            return frame

    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2023-01-03")], ["A", "B"]], names=["date", "stock_code"]
    )
    factors = pd.DataFrame({"dividend": [2.0, 4.0], "low_vol": [6.0, 2.0]}, index=index)
    baseline = _target_residual_baseline(
        Synthesizer(),
        factors,
        index,
        {
            "blend_weights": {"dividend": 0.5, "low_vol": 0.5},
            "target_residualization": {
                "enabled": True,
                "baseline_factors": ["dividend", "low_vol"],
            },
        },
    )

    np.testing.assert_array_equal(baseline.to_numpy(), [4.0, 3.0])


def test_custom_cv_can_score_dsr_on_raw_residual_returns():
    captured = []

    class Evaluator:
        def evaluate(self, _prediction, realized, _dates):
            captured.append(realized.copy())
            return float(np.mean(realized))

    wrapper = QlibLinearWrapper(
        {
            "model_type": "elasticnet",
            "lasso_max_iter": 1000,
            "cv_n_jobs": 1,
        }
    )
    dates = pd.date_range("2025-01-02", periods=20, freq="B")
    index = pd.MultiIndex.from_arrays(
        [dates, ["A"] * len(dates)], names=["datetime", "instrument"]
    )
    raw_metric = np.arange(20, dtype=float)

    wrapper._score_param_grid_cv(
        np.column_stack([np.arange(20), np.arange(20) ** 2]),
        np.linspace(-1.0, 1.0, 20),
        index,
        None,
        [(np.arange(10), np.arange(10, 20))],
        [(0.01, 0.5)],
        Evaluator(),
        metric_y=raw_metric,
    )

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], raw_metric[10:])


def test_dsr_is_finite_for_skewed_short_validation_fold():
    evaluator = CVMetricEvaluator(
        "deflated_sharpe",
        {"top_k_size": 1, "target_period": 5},
        n_trials=200,
    )
    dates = np.repeat(pd.date_range("2024-01-01", periods=20), 2)
    predictions = np.tile([1.0, -1.0], 20)
    long_short = np.array([0.01] * 19 + [1.0])
    returns = np.column_stack([long_short, np.zeros(20)]).reshape(-1)

    score = evaluator.evaluate(predictions, returns, dates)

    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_net_topk_dsr_penalizes_switching_and_counts_rebalances():
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    panel_dates = np.repeat(dates, 2)
    instruments = np.tile(["A", "B"], len(dates))
    realized = np.repeat(np.linspace(0.002, 0.012, len(dates)), 2)
    stable = np.tile([1.0, 0.0], len(dates))
    switching = np.concatenate(
        [[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0] for index in range(len(dates))]
    )
    evaluator = CVMetricEvaluator(
        "net_topk_dsr",
        {
            "top_k_size": 1,
            "target_period": 5,
            "cv_rebalance_stride": 5,
            "cv_cost_rate": 0.003,
        },
        n_trials=2,
    )

    assert evaluator.effective_period_count(panel_dates) == 8
    assert evaluator.evaluate(stable, realized, panel_dates, instruments) > evaluator.evaluate(
        switching, realized, panel_dates, instruments
    )


def test_pooled_oof_gamma_scoring_and_threshold_fallback(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=40, freq="5B")
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B"]], names=["datetime", "instrument"]
    )
    X = np.tile([[1.0], [-1.0]], (len(dates), 1))
    y = X[:, 0].copy()
    baseline = np.zeros(len(index))
    captured_periods = []

    class Evaluator:
        def evaluate(self, prediction, _realized, score_dates):
            captured_periods.append(pd.Index(score_dates).nunique())
            return float(np.mean(np.abs(prediction)))

    config = {
        "model_type": "elasticnet",
        "lasso_max_iter": 1000,
        "cv_n_jobs": 1,
        "param_search_metric": "deflated_sharpe",
        "pooled_oof_dsr": True,
        "residual_gamma_grid": [0.0, 0.5],
    }
    wrapper = QlibLinearWrapper(config)
    scores = wrapper._score_param_grid_cv(
        X,
        y,
        index,
        None,
        [(np.arange(len(index) // 2), np.arange(len(index) // 2, len(index)))],
        [(0.01, 0.5)],
        Evaluator(),
        metric_y=y,
        baseline_y=baseline,
    )

    assert captured_periods == [20, 20]
    assert scores[(0.01, 0.5, 0.5)]["score"] > scores[(0.01, 0.5, 0.0)]["score"]

    fallback_config = {
        **config,
        "lasso_alpha_range": [0.01],
        "l1_ratio_range": [0.5],
        "lasso_cv_folds": 2,
        "cv_min_oof_periods": 30,
        "cv_dsr_min_threshold": 0.8,
    }
    fallback_config["_label_exit_dates"] = {d: d + pd.Timedelta(days=7) for d in dates}
    fallback_wrapper = QlibLinearWrapper(fallback_config)
    monkeypatch.setattr(
        fallback_wrapper,
        "_score_param_grid_cv",
        lambda *args, **kwargs: {
            (0.01, 0.5, 0.0): {"score": 0.70, "n_folds": 2, "oof_periods": 35},
            (0.01, 0.5, 0.5): {"score": 0.79, "n_folds": 2, "oof_periods": 35},
        },
    )
    model_info = fallback_wrapper._fit_with_custom_cv(
        X, y, index, metric_y=y, baseline_y=baseline
    )

    assert model_info["selected_residual_gamma"] == 0.0
    assert model_info["selected_blend_ratio"] == 1.0
    assert model_info["cv_fallback_reason"] == "dsr_below_threshold"

    monkeypatch.setattr(
        fallback_wrapper,
        "_score_param_grid_cv",
        lambda *args, **kwargs: {
            (0.01, 0.5, 0.0): {"score": 0.60, "n_folds": 2, "oof_periods": 25},
            (0.01, 0.5, 0.5): {"score": 0.75, "n_folds": 2, "oof_periods": 25},
        },
    )
    model_info = fallback_wrapper._fit_with_custom_cv(
        X, y, index, metric_y=y, baseline_y=baseline
    )
    assert model_info["cv_score"] == 0.75
    assert model_info["pooled_oof_periods"] == 25
    assert model_info["selected_residual_gamma"] == 0.0
    assert model_info["cv_fallback_reason"] == "insufficient_oof_history"
    with pytest.raises(IncompleteResearchWindowError, match="insufficient_oof_history"):
        _require_complete_research_window(
            {"require_complete_research_windows": True},
            model_info["cv_fallback_reason"],
        )


def test_fixed_positive_gamma_counts_only_the_actual_parameter_grid(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=40, freq="5B")
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B"]], names=["datetime", "instrument"]
    )
    X = np.tile([[1.0], [-1.0]], (len(dates), 1))
    y = X[:, 0].copy()
    baseline = np.zeros(len(index))
    wrapper = QlibLinearWrapper(
        {
            "model_type": "elasticnet",
            "_label_exit_dates": {d: d + pd.Timedelta(days=7) for d in dates},
            "lasso_alpha_range": [0.01],
            "l1_ratio_range": [0.5],
            "lasso_cv_folds": 2,
            "lasso_max_iter": 1000,
            "cv_n_jobs": 1,
            "param_search_metric": "deflated_sharpe",
            "pooled_oof_dsr": True,
            "residual_gamma_grid": [0.1],
            "cv_min_oof_periods": 0,
        }
    )
    monkeypatch.setattr(
        wrapper,
        "_score_param_grid_cv",
        lambda *args, **kwargs: {
            (0.01, 0.5, 0.1): {"score": 0.7, "n_folds": 2, "oof_periods": 25}
        },
    )

    model_info = wrapper._fit_with_custom_cv(
        X, y, index, metric_y=y, baseline_y=baseline
    )

    assert model_info["cv_candidate_count"] == 1
    assert model_info["selected_residual_gamma"] == pytest.approx(0.1)


def test_pure_size_fallback_uses_the_complete_dynamic_pool():
    class Synthesizer:
        config = {"rolling_window": {"predict_window_days": 10}}

        @staticmethod
        def _apply_feature_preprocessing(frame, _config):
            return frame

    date = pd.Timestamp("2025-01-08")
    index = pd.MultiIndex.from_product(
        [[date], ["A", "B"]], names=["date", "stock_code"]
    )
    factor_df = pd.DataFrame({"Factor1": [1.0, -1.0]}, index=index)

    result, model_info = _run_blend_only_mode(
        Synthesizer(),
        factor_df,
        "2025-01-07",
        {
            "blend_with_original_ratio": 1.0,
            "blend_weights": {"Factor1": 1.0},
            "_training_factor_names": ["ResidualFactor"],
        },
        {date: ["A", "B"]},
        __import__("logging").getLogger("test"),
    )

    assert list(result.index) == list(index)
    assert model_info["validation_start"] == "2025-01-08"
    assert model_info["coefficients"] == {"ResidualFactor": 0.0}


def test_parent_baseline_is_the_exact_residual_target_and_fallback(tmp_path):
    class Synthesizer:
        config = {"rolling_window": {"predict_window_days": 10}}

        @staticmethod
        def _apply_feature_preprocessing(frame, _config):
            return frame

    date = pd.Timestamp("2025-01-08")
    parent_index = pd.MultiIndex.from_product(
        [[date], ["A", "B"]], names=["date", "stock_code"]
    )
    parent = pd.Series([0.7, -0.2], index=parent_index, name="composite_value")
    path = tmp_path / "composite_cache_parent.parquet"
    parent.to_frame().to_parquet(path)
    config = {
        "blend_with_original_ratio": 1.0,
        "blend_weights": {},
        "parent_baseline": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "target_residualization": {"enabled": True, "parent_baseline": True},
        "_training_factor_names": ["ResidualFactor"],
    }
    child_index = pd.MultiIndex.from_product(
        [[date], ["A", "B", "C"]], names=["date", "stock_code"]
    )
    factor_df = pd.DataFrame({"ResidualFactor": [1.0, 2.0, 3.0]}, index=child_index)

    baseline = _target_residual_baseline(Synthesizer(), factor_df, child_index, config)
    result, model_info = _run_blend_only_mode(
        Synthesizer(), factor_df, "2025-01-07", config, None, __import__("logging").getLogger("test")
    )

    pd.testing.assert_series_equal(baseline.dropna(), parent)
    pd.testing.assert_series_equal(result, parent)
    assert model_info["blend_only"] is True


def test_research_window_fails_closed_when_training_history_is_missing():
    class Synthesizer:
        config = {"rolling_window": {"predict_window_days": 10}}
        logger = __import__("logging").getLogger("test")

        @staticmethod
        def _apply_feature_preprocessing(frame, _config):
            return frame

    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-08")], ["A", "B"]],
        names=["date", "stock_code"],
    )
    factors = pd.DataFrame(
        {"ResidualFactor": [1.0, -1.0], "Factor1": [2.0, -2.0]}, index=index
    )
    config = {
        "_training_factor_names": ["ResidualFactor"],
        "blend_weights": {"Factor1": 1.0},
        "require_complete_research_windows": True,
    }

    with pytest.raises(IncompleteResearchWindowError, match="training_data_unavailable"):
        run_lasso_synthesis(
            Synthesizer(),
            factors,
            pd.Series([0.01, -0.01], index=index),
            "2024-01-01",
            "2025-01-01",
            config,
        )
