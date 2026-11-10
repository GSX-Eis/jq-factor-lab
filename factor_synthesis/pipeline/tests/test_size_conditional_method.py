import inspect
import json
import math

import numpy as np
import pandas as pd
import pytest

import factor_synthesis.pipeline.size_conditional_method as size_conditional_method
from factor_synthesis.pipeline.size_conditional_method import (
    FIXED_BUCKET_EDGES,
    FIXED_GATE_CANDIDATES,
    _fit_elasticnet,
    _atomic_json,
    build_effect_coded_features,
    daily_size_percentile,
    gate_mask,
    load_or_build_size_conditional_snapshot,
    prepare_elasticnet_path,
    prepare_top10_context,
    select_inner_policy,
    size_conditional_snapshot_identity,
    solve_elasticnet_path,
    top10_net_utility,
    validate_size_conditional_config,
)


def _valid_config():
    return {
        "target_period": 5,
        "lasso_alpha_range": [0.001, 0.01],
        "n_alphas": 2,
        "l1_ratio_range": [0.5, 0.9],
        "size_conditioning": {
            "bucket_edges": list(FIXED_BUCKET_EDGES),
            "gate_candidates": [list(gate) for gate in FIXED_GATE_CANDIDATES],
            "interaction_shrinkage_grid": [1, 4],
            "inner_folds": 3,
            "embargo_days": 5,
            "stability_threshold": 0.6,
        },
    }


def _top_k():
    return {
        "enabled": True,
        "top_k": 10,
        "holding_period": 1,
        "cost_rate": 0.001,
        "enable_tday_filter": True,
    }


def test_config_percentiles_boundaries_and_effect_coding():
    normalized = validate_size_conditional_config(_valid_config(), _top_k())
    assert len(normalized["alpha_grid"]) == 2
    bad = _valid_config()
    bad["size_conditioning"]["embargo_days"] = 4
    with pytest.raises(ValueError, match="embargo"):
        validate_size_conditional_config(bad, _top_k())

    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], ["c", "a", "b"]],
        names=["date", "stock_code"],
    )
    cap = pd.Series([3.0, 1.0, 2.0], index=index)
    percentile = daily_size_percentile(cap, index)
    assert percentile.tolist() == pytest.approx([100.0, 100 / 3, 200 / 3])

    boundary = pd.Series([0.0, 20.0, 99.9, 100.0])
    assert gate_mask(boundary, (0.0, 20.0)).tolist() == [True, False, False, False]
    assert gate_mask(boundary, (20.0, 100.0)).tolist() == [False, True, True, True]

    five_index = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-01", periods=5), ["A"]],
        names=["date", "stock_code"],
    )
    factors = pd.DataFrame({"factor": np.arange(1, 6, dtype=float)}, index=five_index)
    percentiles = pd.Series([10, 30, 50, 70, 90], index=five_index)
    matrix, names, metadata = build_effect_coded_features(
        factors, percentiles, FIXED_BUCKET_EDGES, 2.0
    )
    assert names == [
        "factor",
        "factor::size_dev_0_20",
        "factor::size_dev_20_40",
        "factor::size_dev_40_60",
        "factor::size_dev_60_80",
    ]
    assert matrix[:, 1:].tolist() == [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [-2.5, -2.5, -2.5, -2.5],
    ]
    assert metadata["coefficient_rescale"] == 0.5


def test_nested_selection_recovers_signal_gate_and_purges_dates(monkeypatch):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=64)
    stocks = [f"S{value:02d}" for value in range(25)]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])
    percentile_by_stock = np.tile(np.arange(1, 26) * 4.0, len(dates))
    signal = rng.normal(size=len(index))
    noise = rng.normal(scale=0.02, size=len(index))
    low = percentile_by_stock < 40
    returns = np.where(low, 0.04 * signal + noise, noise)
    factors = pd.DataFrame(
        {"signal": signal, "unsupported": rng.normal(size=len(index))}, index=index
    )
    target = pd.Series(returns, index=index)
    percentiles = pd.Series(percentile_by_stock, index=index)
    normalized = {
        "bucket_edges": FIXED_BUCKET_EDGES,
        "gate_candidates": ((0.0, 40.0), (40.0, 100.0)),
        "interaction_shrinkage_grid": (1.0, 20.0),
        "alpha_grid": (0.001, 0.01),
        "l1_ratio_grid": (0.9,),
        "inner_folds": 3,
        "embargo_days": 5,
        "stability_threshold": 0.6,
        "target_period": 5,
        "top_k": 3,
        "cost_rate": 0.001,
        "max_iter": 5000,
        "tol": 1e-5,
    }
    build_calls = 0
    original_build = size_conditional_method.build_effect_coded_features

    def counted_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    monkeypatch.setattr(
        size_conditional_method, "build_effect_coded_features", counted_build
    )
    result = select_inner_policy(
        factors, target, target, percentiles, None, normalized,
        # Synthetic fixture declares a five-business-session forward horizon.
        label_context={"_label_exit_dates": {d: d + pd.offsets.BDay(5) for d in dates}},
    )
    assert result["selected_policy"]["gate"] == [0.0, 40.0]
    assert len(result["trials"]) == 8
    # 2 gates * 2 shrinkages * 3 folds * (train + validation), followed by
    # one selected-policy stability refit per fold. Alpha/l1 candidates reuse
    # those matrices instead of rebuilding them.
    assert build_calls == 27
    assert result["performance"]["validation_context_prepare_count"] == 6
    assert result["performance"]["gram_prepare_count"] == 12
    assert result["performance"]["path_solve_count"] == 24
    assert len(result["inner_folds"]) == 3
    assert all(
        pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"])
        for fold in result["inner_folds"]
    )
    assert "outer" not in inspect.signature(select_inner_policy).parameters


def test_inner_selection_rejects_constant_intercept_only_policy():
    rng = np.random.default_rng(29)
    dates = pd.bdate_range("2024-01-02", periods=64)
    stocks = [f"S{value:02d}" for value in range(20)]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])
    factors = pd.DataFrame({"signal": rng.normal(size=len(index))}, index=index)
    target = pd.Series(rng.normal(size=len(index)), index=index)
    percentiles = pd.Series(np.tile(np.arange(1, 21) * 5.0, len(dates)), index=index)
    normalized = {
        "bucket_edges": FIXED_BUCKET_EDGES,
        "gate_candidates": ((0.0, 100.0),),
        "interaction_shrinkage_grid": (1.0,),
        "alpha_grid": (100.0,),
        "l1_ratio_grid": (0.9,),
        "inner_folds": 3,
        "embargo_days": 5,
        "stability_threshold": 0.6,
        "target_period": 5,
        "top_k": 3,
        "cost_rate": 0.001,
        "max_iter": 5000,
        "tol": 1e-5,
    }
    with pytest.raises(ValueError, match=r"all_zero_inner_fit=1/1"):
        select_inner_policy(
        factors, target, target, percentiles, None, normalized,
        # Synthetic fixture declares a five-business-session forward horizon.
        label_context={"_label_exit_dates": {d: d + pd.offsets.BDay(5) for d in dates}},
    )


def test_inner_selection_reports_empty_stability_without_invalidating_policy(
    monkeypatch,
):
    rng = np.random.default_rng(29)
    dates = pd.bdate_range("2024-01-02", periods=64)
    stocks = [f"S{value:02d}" for value in range(20)]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])
    signal = rng.normal(size=len(index))
    factors = pd.DataFrame({"signal": signal}, index=index)
    target = pd.Series(signal + rng.normal(scale=0.01, size=len(index)), index=index)
    percentiles = pd.Series(np.tile(np.arange(1, 21) * 5.0, len(dates)), index=index)
    normalized = {
        "bucket_edges": FIXED_BUCKET_EDGES,
        "gate_candidates": ((0.0, 100.0),),
        "interaction_shrinkage_grid": (1.0,),
        "alpha_grid": (0.001,),
        "l1_ratio_grid": (0.9,),
        "inner_folds": 3,
        "embargo_days": 5,
        "stability_threshold": 0.6,
        "target_period": 5,
        "top_k": 3,
        "cost_rate": 0.001,
        "max_iter": 5000,
        "tol": 1e-5,
    }
    fit_calls = 0

    class AlternatingSupport:
        def __init__(self, width, position):
            self.coef_ = np.zeros(width)
            self.coef_[position] = 1.0

    def alternating_fit(X, *args, **kwargs):
        nonlocal fit_calls
        model = AlternatingSupport(X.shape[1], fit_calls % X.shape[1])
        fit_calls += 1
        return model

    monkeypatch.setattr(size_conditional_method, "_fit_elasticnet", alternating_fit)
    result = select_inner_policy(
        factors, target, target, percentiles, None, normalized,
        # Synthetic fixture declares a five-business-session forward horizon.
        label_context={"_label_exit_dates": {d: d + pd.offsets.BDay(5) for d in dates}},
    )
    assert result["stable_support"] == []
    assert result["stability_status"] == "no_stable_support"


def test_stronger_shrinkage_suppresses_unsupported_interaction():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=40)
    stocks = [f"S{value:02d}" for value in range(20)]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])
    factor = rng.normal(size=len(index))
    frame = pd.DataFrame({"factor": factor}, index=index)
    percentiles = pd.Series(np.tile(np.arange(1, 21) * 5.0, len(dates)), index=index)
    bucket = np.tile(np.arange(1, 21) * 5.0, len(dates))
    weak_deviation = np.where(bucket < 20, 0.08 * factor, 0.0)
    weak_deviation += np.where(bucket >= 80, -0.08 * factor, 0.0)
    target = factor + weak_deviation + rng.normal(scale=0.02, size=len(index))

    interaction_norms = []
    recovered_deviations = []
    for shrinkage in (1.0, 50.0):
        matrix, _, _ = build_effect_coded_features(
            frame, percentiles, FIXED_BUCKET_EDGES, shrinkage
        )
        model = _fit_elasticnet(
            matrix,
            target,
            alpha=0.0005,
            l1_ratio=0.9,
            sample_weight=None,
            max_iter=5000,
            tol=1e-6,
        )
        deviations = model.coef_[1:] / shrinkage
        recovered_deviations.append(deviations)
        interaction_norms.append(np.abs(deviations).sum())
    assert recovered_deviations[0][0] > 0
    assert -recovered_deviations[0].sum() < 0
    assert interaction_norms[1] < interaction_norms[0]


def test_prepared_top10_preserves_ties_turnover_and_returns():
    dates = pd.to_datetime(["2025-01-06"] * 4 + ["2025-01-13"] * 4 + ["2025-01-20"] * 4)
    index = pd.MultiIndex.from_arrays(
        [dates, ["A", "B", "C", "D"] * 3], names=["date", "stock_code"]
    )
    prediction = np.array([3, 2, 2, 1, 1, 3, 2, 2, 2, 1, 3, 2], dtype=float)
    raw_returns = np.array(
        [0.01, 0.03, 0.08, 0.09, 0.01, 0.04, 0.02, 0.08, -0.01, 0, 0.01, 0.08]
    )
    context = prepare_top10_context(raw_returns, index, top_k=2)
    sharpe, weekly, turnover = top10_net_utility(
        prediction,
        raw_returns,
        index,
        top_k=2,
        cost_rate=0.001,
        target_period=5,
        prepared_context=context,
    )
    expected = pd.Series(
        [0.0195, 0.0295, -0.0005],
        index=pd.to_datetime(["2025-01-06", "2025-01-13", "2025-01-20"]),
        name="inner_weekly_net_return",
    )
    pd.testing.assert_series_equal(weekly, expected)
    assert turnover == pytest.approx(0.5)
    assert sharpe == pytest.approx(expected.mean() / expected.std(ddof=1) * np.sqrt(52))


@pytest.mark.parametrize("weighted", [False, True])
def test_elasticnet_path_matches_independent_fits(weighted):
    rng = np.random.default_rng(23)
    X = rng.normal(size=(180, 9))
    y = X @ rng.normal(size=9) + rng.normal(scale=0.2, size=180)
    weights = rng.uniform(0.2, 2.0, size=len(y)) if weighted else None
    prepared = prepare_elasticnet_path(X, y, weights)
    for alpha in [0.001, 0.01, 0.1]:
        solved = solve_elasticnet_path(
            prepared,
            alphas=[alpha],
            l1_ratio=0.7,
            max_iter=10000,
            tol=1e-4,
        )
        reference = _fit_elasticnet(
            X,
            y,
            alpha=float(alpha),
            l1_ratio=0.7,
            sample_weight=weights,
            max_iter=10000,
            tol=1e-4,
        )
        np.testing.assert_allclose(
            solved["coefficients"][0], reference.coef_, atol=1e-12, rtol=0
        )
        assert solved["intercepts"][0] == pytest.approx(reference.intercept_, abs=1e-12)
        np.testing.assert_allclose(
            X @ solved["coefficients"][0] + solved["intercepts"][0],
            reference.predict(X),
            atol=1e-12,
            rtol=0,
        )


def test_prepared_snapshot_hits_and_source_change_invalidates(tmp_path):
    source = tmp_path / "source.duckdb"
    import duckdb
    from data_pipeline.loaders._transactions import import_transaction
    with duckdb.connect(str(source)) as conn, import_transaction(conn):
        conn.execute("CREATE TABLE prices (value DOUBLE)")
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], ["A", "B"]],
        names=["date", "stock_code"],
    )
    panel = {
        "train_factors": pd.DataFrame({"factor": [1.0, 2.0]}, index=index),
        "predict_factors": pd.DataFrame({"factor": [1.0, 2.0]}, index=index),
        "raw_returns": pd.Series([0.01, 0.02], index=index),
        "fit_target": pd.Series([-0.5, 0.5], index=index),
        "train_percentiles": pd.Series([50.0, 100.0], index=index),
        "predict_percentiles": pd.Series([50.0, 100.0], index=index),
        "label_context": {"_label_exit_dates": {"2025-01-02": "2025-01-09"}},
    }
    calls = 0

    def builder():
        nonlocal calls
        calls += 1
        return panel

    identity = size_conditional_snapshot_identity(
        {"stock_pool": {"id": "wind"}, "calendar_filters": {"skip": []}},
        {"target_period": 5, "preprocess": {"ALL": "rank_zscore"},
         "_label_exit_dates": {"2025-01-02": "2025-01-09"}},
        ["factor"],
        "2024-01-01",
        "2025-01-01",
        [str(source)],
    )
    first, first_info = load_or_build_size_conditional_snapshot(
        tmp_path, identity, builder, enabled=True
    )
    second, second_info = load_or_build_size_conditional_snapshot(
        tmp_path, identity, builder, enabled=True
    )
    assert first_info["status"] == "miss"
    assert second_info["status"] == "hit"
    assert calls == 1
    pd.testing.assert_frame_equal(first["train_factors"], second["train_factors"])

    with duckdb.connect(str(source)) as conn, import_transaction(conn):
        conn.execute("INSERT INTO prices VALUES (2)")
    changed_identity = size_conditional_snapshot_identity(
        {"stock_pool": {"id": "wind"}, "calendar_filters": {"skip": []}},
        {"target_period": 5, "preprocess": {"ALL": "rank_zscore"},
         "_label_exit_dates": {"2025-01-02": "2025-01-09"}},
        ["factor"],
        "2024-01-01",
        "2025-01-01",
        [str(source)],
    )
    _, changed_info = load_or_build_size_conditional_snapshot(
        tmp_path, changed_identity, builder, enabled=True
    )
    assert changed_info["status"] == "miss"
    assert calls == 2


def test_diagnostic_json_replaces_non_finite_values_with_null(tmp_path):
    path = tmp_path / "diagnostic.json"
    _atomic_json(path, {"rejected": -math.inf, "nested": [math.nan, 1.0]})
    assert json.loads(path.read_text()) == {"rejected": None, "nested": [None, 1.0]}
