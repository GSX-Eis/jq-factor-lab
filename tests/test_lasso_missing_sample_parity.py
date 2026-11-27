import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.core.factor_ops import apply_feature_preprocessing
from factor_synthesis.pipeline.lasso_method import _blend_with_original, run_lasso_synthesis
from factor_synthesis.qlib_adapter import QlibLinearWrapper
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from run_factor_synthesis import _build_lasso_bundle_model
from utils.strategy_kernel import linear_signal, rank_scores


def _synthesizer() -> SimpleNamespace:
    return SimpleNamespace(
        config={"rolling_window": {"predict_window_days": 10, "min_train_samples": 2}},
        logger=logging.getLogger(__name__),
        _apply_feature_preprocessing=apply_feature_preprocessing,
    )


def _bundle_replay(
    tmp_path: Path,
    raw: pd.DataFrame,
    stocks: list[str],
    day: pd.Timestamp,
    model_info: dict,
    config: dict,
) -> pd.Series:
    model = _build_lasso_bundle_model(
        train_end=model_info["train_end"],
        weights={key: value for key, value in model_info["coefficients"].items() if value != 0},
        model_info=model_info,
        synthesis_config=config,
        parent_baseline={},
    )
    assert model is not None
    bundle_path = tmp_path / "lasso_model_bundle_test.json"
    bundle_path.write_text(json.dumps({"index": {"models": [model]}}), encoding="utf-8")
    runner = VectorBTBacktestRunner({}, bundle_path=str(bundle_path))
    runner.factor_db_conn = object()
    runner._factor_loader = SimpleNamespace(
        load_factors=lambda _day, codes, names: raw.reindex(index=codes, columns=names),
    )
    replay, _, _ = runner._compute_synthesis_score(day.strftime("%Y-%m-%d"), stocks)
    expected, _ = linear_signal(raw.reindex(stocks), model)
    pd.testing.assert_series_equal(replay, expected, check_exact=True)
    return replay


def test_blend_only_excludes_all_missing_dynamic_pool_stock(tmp_path: Path) -> None:
    day = pd.Timestamp("2025-01-08")
    stocks = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([[day], stocks], names=["date", "stock_code"])
    factors = pd.DataFrame({"f": [1.0, 2.0, 10.0, np.nan]}, index=index)
    config = {
        "blend_with_original_ratio": 1.0,
        "blend_weights": {"f": 1.0},
        "preprocess": {"f": "zscore"},
        "standardize": True,
        "postprocess_cs_standardize": False,
    }

    local, model = run_lasso_synthesis(
        _synthesizer(), factors, pd.Series(np.nan, index=index), "2025-01-01", "2025-01-07",
        config, daily_pools={day: stocks},
    )
    replay = _bundle_replay(tmp_path, factors.xs(day), stocks, day, model, config)

    assert model["cv_fallback_reason"] == "training_data_unavailable"
    assert rank_scores(replay)[:2] == ["C", "B"]
    assert rank_scores(local.xs(day))[:2] == rank_scores(replay)[:2]
    pd.testing.assert_series_equal(local.xs(day), replay, check_exact=True)


def test_blend_only_fills_partial_inputs_and_skips_missing_pool_members(tmp_path: Path) -> None:
    day = pd.Timestamp("2025-01-08")
    stocks = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([[day], stocks[:3]], names=["date", "stock_code"])
    factors = pd.DataFrame({"f": [-2.0, np.nan, np.nan], "g": [np.nan, 8.0, np.nan]}, index=index)
    config = {
        "blend_with_original_ratio": 0.5,
        "blend_weights": {"f": 2.0, "g": 1.0},
        "preprocess": {"ALL": "zscore"},
        "standardize": False,
    }
    local, model = run_lasso_synthesis(
        _synthesizer(), factors, pd.Series(np.nan, index=index), "2025-01-01", "2025-01-07",
        config, daily_pools={day: stocks},
    )
    replay = _bundle_replay(tmp_path, factors.xs(day), stocks, day, model, config)
    local = local.xs(day)

    assert local.to_dict() == {"A": -2.0, "B": 4.0}
    pd.testing.assert_series_equal(local, replay, check_exact=True)
    assert rank_scores(local)[:2] == rank_scores(replay)[:2] == ["B", "A"]


@pytest.mark.parametrize(
    ("ratio", "expected", "top_two"),
    [
        (0.0, {"A": 20.0, "B": -4.0, "F": -2.0}, ["A", "F"]),
        (0.5, {"A": 10.0, "B": -2.0, "C": 4.0, "D": 3.0, "F": 0.0}, ["A", "C"]),
        (1.0, {"A": 0.0, "B": 0.0, "C": 8.0, "D": 6.0, "F": 2.0}, ["C", "D"]),
    ],
)
def test_fitted_lasso_uses_active_weight_and_blend_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ratio: float,
    expected: dict[str, float], top_two: list[str],
) -> None:
    day = pd.Timestamp("2025-01-08")
    stocks = list("ABCDEFG")
    train_dates = pd.date_range("2025-01-01", "2025-01-06")
    names = ["date", "stock_code"]
    train_index = pd.MultiIndex.from_product([train_dates, stocks], names=names)
    predict_index = pd.MultiIndex.from_product([[day], stocks], names=names)
    columns = ["m1", "m2", "inactive", "b1", "b2"]
    train = pd.DataFrame(1.0, index=train_index, columns=columns)
    predict = pd.DataFrame(
        [
            [10, np.nan, 1, np.nan, np.nan],
            [np.nan, -4, 1, np.nan, np.nan],
            [np.nan, np.nan, 1, np.nan, 8],
            [np.nan, np.nan, 1, 6, np.nan],
            [np.nan, np.nan, 1, np.nan, np.nan],
            [np.nan, -2, 1, 2, np.nan],
            [100, 100, 1, 100, 100],
        ],
        index=predict_index, columns=columns, dtype=float,
    )
    def _fake_fit(self, dataset, sample_weight):
        self.feature_names = ["m1", "m2", "inactive"]
        self.model = SimpleNamespace(
            predict=lambda values: np.asarray(values, dtype=float) @ np.array([2.0, 1.0, 0.0])
        )
        return {"coefficients": {"m1": 2.0, "m2": 1.0, "inactive": 0.0}}

    monkeypatch.setattr(QlibLinearWrapper, "fit", _fake_fit)
    config = {
        "_training_factor_names": ["m1", "m2", "inactive"],
        "_label_exit_dates": {d: d + pd.Timedelta(days=1) for d in train_dates},
        "blend_with_original_ratio": ratio,
        "blend_weights": {"b1": 1.0, "b2": 1.0},
        "preprocess": {"ALL": "none"},
    }

    local, info = run_lasso_synthesis(
        _synthesizer(), pd.concat([train, predict]), pd.Series(1.0, index=train_index),
        "2025-01-01", "2025-01-07", config, daily_pools={day: stocks[:-1]},
    )
    replay = _bundle_replay(tmp_path, predict.xs(day), stocks[:-1], day, info, config)
    local = local.xs(day)

    assert local.to_dict() == expected
    pd.testing.assert_series_equal(local.rename(replay.name), replay, check_exact=True)
    assert rank_scores(local)[:2] == rank_scores(replay)[:2] == top_two


def test_blend_with_original_filters_all_missing_and_preserves_union() -> None:
    day = pd.Timestamp("2025-01-08")
    stocks = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([[day], stocks], names=["date", "stock_code"])
    factors = pd.DataFrame(
        {
            "f1": [1.0, 2.0, np.nan, np.nan],
            "f2": [np.nan, 4.0, 6.0, np.nan],
        },
        index=index,
    )
    logger = logging.getLogger(__name__)

    # Without active model predictions (all NaN), 100% blend excludes D while keeping partial A/B/C.
    pure_blend = _blend_with_original(
        pd.Series(np.nan, index=index),
        factors,
        {"f1": 1.0, "f2": 1.0},
        1.0,
        logger,
    )
    assert list(pure_blend.xs(day).index) == ["A", "B", "C"]
    assert "D" not in pure_blend.xs(day).index

    # With active model predictions, 100% blend retains model-eligible rows (D gets 0.0).
    model_union_at_one = _blend_with_original(
        pd.Series(0.0, index=index),
        factors,
        {"f1": 1.0, "f2": 1.0},
        1.0,
        logger,
    )
    assert list(model_union_at_one.xs(day).index) == ["A", "B", "C", "D"]
    assert model_union_at_one.xs(day).loc["D"] == 0.0

    # Partial blend uses union of valid model predictions and valid blend factors; D stays excluded.
    predictions = pd.Series([10.0, np.nan, np.nan, np.nan], index=index)
    mixed = _blend_with_original(
        predictions,
        pd.DataFrame({"f1": [np.nan, 2.0, 6.0, np.nan]}, index=index),
        {"f1": 1.0},
        0.5,
        logger,
    )
    assert list(mixed.xs(day).index) == ["A", "B", "C"]
    assert mixed.xs(day).loc["A"] == pytest.approx(5.0)
    assert mixed.xs(day).loc["B"] < mixed.xs(day).loc["C"]


def test_fitted_lasso_overlapping_blend_factor_uses_same_effective_preprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    day = pd.Timestamp("2025-01-08")
    stocks = list("ABCDEFG")
    train_dates = pd.date_range("2025-01-01", "2025-01-06")
    names = ["date", "stock_code"]
    train_index = pd.MultiIndex.from_product([train_dates, stocks], names=names)
    predict_index = pd.MultiIndex.from_product([[day], stocks], names=names)
    columns = ["m1", "m2", "inactive", "b1", "b2"]
    train = pd.DataFrame(1.0, index=train_index, columns=columns)
    predict = pd.DataFrame(
        [
            [10, np.nan, 1, np.nan, np.nan],
            [np.nan, -4, 1, np.nan, np.nan],
            [np.nan, np.nan, 1, np.nan, 8],
            [np.nan, np.nan, 1, 6, np.nan],
            [np.nan, np.nan, 1, np.nan, np.nan],
            [np.nan, -2, 1, 2, np.nan],
            [100, 100, 1, 100, 100],
        ],
        index=predict_index, columns=columns, dtype=float,
    )

    def _fake_fit(self, dataset, sample_weight):
        self.feature_names = ["m1", "m2", "inactive"]
        self.model = SimpleNamespace(
            predict=lambda values: np.asarray(values, dtype=float) @ np.array([2.0, 1.0, 0.0]),
            intercept_=0.0,
        )
        return {"coefficients": {"m1": 2.0, "m2": 1.0, "inactive": 0.0}}

    monkeypatch.setattr(QlibLinearWrapper, "fit", _fake_fit)
    config = {
        "_training_factor_names": ["m1", "m2", "inactive"],
        "_label_exit_dates": {d: d + pd.Timedelta(days=1) for d in train_dates},
        "blend_with_original_ratio": 0.5,
        "blend_weights": {"m1": 1.0},
        "preprocess": {},
    }

    local, info = run_lasso_synthesis(
        _synthesizer(), pd.concat([train, predict]), pd.Series(1.0, index=train_index),
        "2025-01-01", "2025-01-07", config, daily_pools={day: stocks[:-1]},
    )
    replay = _bundle_replay(tmp_path, predict.xs(day), stocks[:-1], day, info, config)
    local = local.xs(day)

    assert local.to_dict() == {"A": 15.0, "B": -2.0, "F": -1.0}
    pd.testing.assert_series_equal(local.rename(replay.name), replay, check_exact=True)
    assert rank_scores(local)[:2] == rank_scores(replay)[:2] == ["A", "F"]


@pytest.mark.parametrize("postprocess_cs", [False, True])
def test_fitted_lasso_nonzero_intercept_matches_bundle_replay_with_disjoint_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, postprocess_cs: bool,
) -> None:
    day = pd.Timestamp("2025-01-08")
    stocks = list("ABCDEFG")
    train_dates = pd.date_range("2025-01-01", "2025-01-06")
    names = ["date", "stock_code"]
    train_index = pd.MultiIndex.from_product([train_dates, stocks], names=names)
    predict_index = pd.MultiIndex.from_product([[day], stocks], names=names)
    columns = ["m1", "m2", "inactive", "b1", "b2"]
    train = pd.DataFrame(1.0, index=train_index, columns=columns)
    predict = pd.DataFrame(
        [
            [3.5, np.nan, 1, np.nan, np.nan],
            [np.nan, -4, 1, np.nan, np.nan],
            [np.nan, np.nan, 1, np.nan, 8],
            [np.nan, np.nan, 1, 6, np.nan],
            [np.nan, np.nan, 1, np.nan, np.nan],
            [np.nan, -2, 1, 2, np.nan],
            [100, 100, 1, 100, 100],
        ],
        index=predict_index, columns=columns, dtype=float,
    )

    def _fake_fit(self, dataset, sample_weight):
        self.feature_names = ["m1", "m2", "inactive"]
        self.model = SimpleNamespace(
            predict=lambda values: np.asarray(values, dtype=float) @ np.array([2.0, 1.0, 0.0]) + 2.0
        )
        return {"coefficients": {"m1": 2.0, "m2": 1.0, "inactive": 0.0}}

    monkeypatch.setattr(QlibLinearWrapper, "fit", _fake_fit)
    config = {
        "_training_factor_names": ["m1", "m2", "inactive"],
        "_label_exit_dates": {d: d + pd.Timedelta(days=1) for d in train_dates},
        "blend_with_original_ratio": 0.5,
        "blend_weights": {"b1": 1.0, "b2": 1.0},
        "preprocess": {"ALL": "none"},
        "postprocess_cs_standardize": postprocess_cs,
    }

    local, info = run_lasso_synthesis(
        _synthesizer(), pd.concat([train, predict]), pd.Series(1.0, index=train_index),
        "2025-01-01", "2025-01-07", config, daily_pools={day: stocks[:-1]},
    )
    replay = _bundle_replay(tmp_path, predict.xs(day), stocks[:-1], day, info, config)
    local = local.xs(day)

    assert info["intercept"] == pytest.approx(2.0)
    if not postprocess_cs:
        assert local.to_dict() == {"A": 4.5, "B": -1.0, "C": 4.0, "D": 3.0, "F": 1.0}
        assert rank_scores(local)[:2] == rank_scores(replay)[:2] == ["A", "C"]
    pd.testing.assert_series_equal(local.rename(replay.name), replay, check_exact=True)
    assert rank_scores(local) == rank_scores(replay)

    # Legacy bundle dict omitting "intercept" remains backward compatible (intercept=0.0).
    legacy_score, _ = linear_signal(
        predict.xs(day).reindex(stocks[:-1]),
        {
            "weights": {"m1": 2.0, "m2": 1.0},
            "standardize": True,
            "standardization_method": {"ALL": "none"},
            "postprocess_cs_standardize": False,
            "blend_with_original_ratio": 0.5,
            "blend_weights": {"b1": 1.0, "b2": 1.0},
        },
    )
    assert rank_scores(legacy_score)[:2] == ["C", "A"]


def test_real_qlib_fit_nonzero_intercept_and_distribution_prefilter_parity(
    tmp_path: Path,
) -> None:
    day = pd.Timestamp("2025-01-20")
    stocks = list("ABCDEF")
    train_dates = pd.bdate_range("2025-01-01", "2025-01-16")
    names = ["date", "stock_code"]
    train_index = pd.MultiIndex.from_product([train_dates, stocks], names=names)
    predict_index = pd.MultiIndex.from_product([[day], stocks], names=names)

    rng = np.random.default_rng(1471)
    train_m1 = rng.normal(0.0, 1.0, len(train_index))
    train = pd.DataFrame({"m1": train_m1, "b1": rng.normal(0.0, 1.0, len(train_index))}, index=train_index)
    # Positive mean label + target_cs_demean=False guarantees a non-zero fitted intercept.
    y = pd.Series(0.5 + 0.3 * train_m1, index=train_index, dtype=float)
    predict = pd.DataFrame(
        [
            [0.2, np.nan],
            [1.5, np.nan],
            [np.nan, 4.0],
            [np.nan, 1.0],
            [np.nan, np.nan],
            [-0.5, 2.0],
        ],
        index=predict_index,
        columns=["m1", "b1"],
        dtype=float,
    )
    config = {
        "_training_factor_names": ["m1"],
        "_label_exit_dates": {d: d + pd.Timedelta(days=1) for d in train_dates},
        "target_cs_demean": False,
        "lasso_alpha_range": [0.001],
        "l1_ratio_range": [1.0],
        "lasso_cv_folds": 2,
        "cv_n_jobs": 1,
        "random_state": 42,
        "distribution_prefilter": {"enabled": True, "min_samples_per_date": 4},
        "blend_with_original_ratio": 0.5,
        "blend_weights": {"m1": 0.4, "b1": 0.6},
        "postprocess_cs_standardize": False,
    }

    local, info = run_lasso_synthesis(
        _synthesizer(),
        pd.concat([train, predict]),
        y,
        "2025-01-01",
        "2025-01-17",
        config,
        daily_pools={day: stocks},
    )
    replay = _bundle_replay(tmp_path, predict.xs(day), stocks, day, info, config)
    local = local.xs(day)

    assert abs(info["intercept"]) > 0.1
    assert "E" not in local.index and "E" not in replay.index
    assert list(local.index) == list(replay.index) == ["A", "B", "C", "D", "F"]
    pd.testing.assert_series_equal(local.rename(replay.name), replay, check_exact=False, atol=1e-12, rtol=0)
    assert rank_scores(local) == rank_scores(replay)


def test_overlapping_ts_preprocess_uses_same_history_mask_as_bundle_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dates = pd.date_range("2025-01-01", "2025-01-06")
    predict_day = pd.Timestamp("2025-01-08")
    stocks = ["A", "B", "C"]
    names = ["date", "stock_code"]
    train_index = pd.MultiIndex.from_product([train_dates, stocks], names=names)
    predict_index = pd.MultiIndex.from_product([[predict_day], stocks], names=names)

    train_m1 = np.repeat(np.arange(1.0, 7.0), len(stocks))
    train = pd.DataFrame({"m1": train_m1, "b1": 1.0}, index=train_index, dtype=float)
    train.loc[(pd.Timestamp("2025-01-06"), "A"), "m1"] = np.nan
    predict = pd.DataFrame(
        {"m1": [9.0, 7.0, np.nan], "b1": [np.nan, 2.0, 3.0]},
        index=predict_index,
        dtype=float,
    )
    panel = pd.concat([train, predict])

    def _fake_fit(self, dataset, sample_weight):
        self.feature_names = ["m1"]
        self.model = SimpleNamespace(
            predict=lambda values: np.asarray(values, dtype=float) @ np.array([1.0]),
            intercept_=0.0,
        )
        return {"coefficients": {"m1": 1.0}, "intercept": 0.0}

    monkeypatch.setattr(QlibLinearWrapper, "fit", _fake_fit)
    config = {
        "_training_factor_names": ["m1"],
        "_label_exit_dates": {d: d + pd.Timedelta(days=1) for d in train_dates},
        "data_quality": {"max_factor_missing_rate": 0.2},
        "blend_with_original_ratio": 0.5,
        "blend_weights": {"m1": 1.0, "b1": 1.0},
        "preprocess": {"m1": "time_series_zscore", "b1": "none", "_ts_window": 2},
        "postprocess_cs_standardize": False,
    }

    local, info = run_lasso_synthesis(
        _synthesizer(),
        panel,
        pd.Series(1.0, index=train_index),
        "2025-01-01",
        "2025-01-07",
        config,
        daily_pools={predict_day: stocks},
    )
    model = _build_lasso_bundle_model(
        train_end=info["train_end"],
        weights={k: v for k, v in info["coefficients"].items() if v != 0},
        model_info=info,
        synthesis_config=config,
        parent_baseline={},
    )
    assert model is not None
    kernel_replay, _ = linear_signal(
        predict.xs(predict_day), model, history=panel, asof=predict_day,
    )
    local_day = local.xs(predict_day)

    assert list(local_day.index) == list(kernel_replay.index) == ["B", "C"]
    assert "A" not in local_day.index
    assert local_day.to_dict() == pytest.approx({"B": 4.0, "C": 1.5})
    pd.testing.assert_series_equal(local_day.rename(kernel_replay.name), kernel_replay, check_exact=True)
    assert rank_scores(local_day)[:1] == rank_scores(kernel_replay)[:1] == ["B"]
