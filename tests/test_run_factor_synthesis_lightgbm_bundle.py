from __future__ import annotations

import pandas as pd
import pytest

import run_factor_synthesis


def test_prediction_window_stops_at_analysis_end() -> None:
    start, end, days = run_factor_synthesis._bounded_prediction_window(
        "2023-12-01", 179, "2023-12-31"
    )

    assert start == pd.Timestamp("2023-12-02")
    assert end == pd.Timestamp("2023-12-31")
    assert days == 29


def test_build_lasso_selection_record_marks_grid_boundaries() -> None:
    record = run_factor_synthesis._build_lasso_selection_record(
        {
            "train_start": "2023-01-01",
            "train_end": "2024-01-01",
            "validation_start": "2024-01-02",
            "alpha": 0.0005,
            "l1_ratio": 0.9,
            "cv_score": 0.12,
            "cv_metric": "deflated_sharpe",
            "cv_candidate_count": 200,
            "cv_folds": 5,
            "alpha_grid_min": 0.0005,
            "alpha_grid_max": 0.005,
            "l1_ratio_grid": [0.3, 0.5, 0.7, 0.9],
        },
        {
            "model_type": "elasticnet",
            "sample_weight_decay": 0.998,
            "neutralization": "none",
            "target_period": 5,
        },
    )

    assert record["alpha_at_lower_bound"] is True
    assert record["alpha_at_upper_bound"] is False
    assert record["alpha_in_grid"] is True
    assert record["alpha_invalid"] is False
    assert record["l1_at_upper_bound"] is True
    assert record["cv_score"] == 0.12
    assert record["cv_candidate_count"] == 200


def test_lasso_selection_record_keeps_pre_blend_residual_sleeve() -> None:
    record = run_factor_synthesis._build_lasso_selection_record(
        {
            "alpha": 0.01,
            "l1_ratio": 0.9,
            "residual_sleeve_n_selected": 1,
            "residual_sleeve_selected_features": ["ResidualA"],
            "residual_sleeve_coefficients": {"ResidualA": 0.2},
        },
        {"lasso_alpha_range": [0.01], "l1_ratio_range": [0.9]},
    )

    assert record["residual_sleeve_n_selected"] == 1
    assert record["residual_sleeve_selected_features"] == ["ResidualA"]
    assert record["residual_sleeve_coefficients"] == {"ResidualA": 0.2}


def test_build_lightgbm_bundle_payload_derives_model_file_from_model_path(tmp_path) -> None:
    model_path = tmp_path / "lgb_model_2024-01-31.txt"
    model_path.write_text("tree\nleaf\n", encoding="utf-8")

    model_info = {
        "train_start": "2023-01-01",
        "train_end": "2024-01-31",
        "predict_start": "2024-02-01",
        "predict_end": "2024-02-29",
        "model_path": str(model_path),
        "feature_names": ["alpha_1", "alpha_2"],
    }

    index_models, index_data, bundle_data = run_factor_synthesis._build_lightgbm_bundle_payload(
        [model_info],
        synthesis_name="lightgbm_demo",
        factor_group="demo_group",
        experiment="LightGBM",
        created_at="2026-04-22 10:00:00",
    )

    assert index_models[0]["model_file"] == model_path.name
    assert model_info["model_file"] == model_path.name
    assert index_data["models"][0]["model_file"] == model_path.name
    assert bundle_data["models"] == {model_path.name: "tree\nleaf\n"}


def test_build_lightgbm_bundle_payload_raises_for_missing_model_file(tmp_path) -> None:
    missing_model_path = tmp_path / "missing_model.txt"

    with pytest.raises(FileNotFoundError, match="缺少模型文件"):
        run_factor_synthesis._build_lightgbm_bundle_payload(
            [{"model_path": str(missing_model_path)}],
            synthesis_name="lightgbm_demo",
            factor_group="demo_group",
            experiment="LightGBM",
            created_at="2026-04-22 10:00:00",
        )


def test_build_algorithm_model_evidence_keeps_linear_window_details() -> None:
    evidence = run_factor_synthesis._build_algorithm_model_evidence(
        method="lasso",
        synthesis_name="elasticnet_demo",
        model_infos=[
            {
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "validation_start": "2023-01-01",
                "validation_end": "2023-03-31",
                "predict_start": "2023-01-01",
                "predict_end": "2023-03-31",
                "coefficients": {"alpha_1": 0.4, "alpha_2": -0.1},
                "r2_score": 0.12,
                "validation_r2": -0.03,
                "alpha": 0.001,
                "l1_ratio": 0.7,
                "cv_score": 0.21,
                "cv_metric": "deflated_sharpe",
                "cv_candidate_count": 20,
                "n_selected": 2,
                "n_features": 2,
                "training_window_audit": {
                    "missing_rate_by_factor": {"alpha_1": 0.0, "alpha_2": 0.02},
                    "removed_factors": [],
                    "survival_ratio": 1.0,
                },
            }
        ],
        diagnostics={
            "outer_folds": [
                {
                    "fold": 1,
                    "start": "2023-01-01",
                    "end": "2023-03-31",
                    "observations": 60,
                    "annual_return": 0.1,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.04,
                }
            ],
            "annual_turnover": 4.0,
        },
    )

    window = evidence["windows"][0]
    assert window["fit"]["train_r2"] == 0.12
    assert window["fit"]["validation_r2"] == -0.03
    assert window["factors"][0] == {
        "factor_id": "alpha_1",
        "attribution": 0.4,
        "abs_rank": 1,
        "selected": True,
        "sign": 1,
        "missing_rate": 0.0,
        "removed": False,
    }
    assert window["outer_fold"]["observations"] == 60
    assert window["outer_fold"]["turnover"] is None
    assert window["outer_fold"]["realized_cost"] is None
    assert evidence["summary"]["factors"]["alpha_1"]["selection_frequency"] == 1.0


def test_build_algorithm_model_evidence_uses_importance_without_fake_r2() -> None:
    evidence = run_factor_synthesis._build_algorithm_model_evidence(
        method="lightgbm",
        synthesis_name="lgb_demo",
        model_infos=[
            {
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "predict_start": "2023-01-01",
                "predict_end": "2023-03-31",
                "feature_importance": {"alpha_1": 7.0, "alpha_2": 0.0},
                "best_iteration": 12,
                "objective": "lambdarank",
            }
        ],
        diagnostics={},
    )

    window = evidence["windows"][0]
    assert window["attribution_kind"] == "feature_importance"
    assert window["fit"]["train_r2"] is None
    assert window["fit"]["validation_r2"] is None
    assert window["factors"][0]["sign"] is None
    assert window["factors"][0]["factor_id"] == "alpha_1"
    assert window["factors"][0]["attribution"] == 7.0


def test_build_algorithm_model_evidence_keeps_removed_factor_without_coefficient() -> None:
    evidence = run_factor_synthesis._build_algorithm_model_evidence(
        method="lasso",
        synthesis_name="removed",
        model_infos=[
            {
                "coefficients": {"F1": 0.2},
                "training_window_audit": {
                    "missing_rate_by_factor": {"F1": 0.0, "F2": 0.9},
                    "removed_factors": ["F2"],
                },
            }
        ],
        diagnostics={},
    )

    removed = next(item for item in evidence["windows"][0]["factors"] if item["factor_id"] == "F2")
    assert removed["attribution"] is None
    assert removed["sign"] is None
    assert removed["removed"] is True
