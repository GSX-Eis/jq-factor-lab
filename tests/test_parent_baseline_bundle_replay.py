from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from run_factor_synthesis import (
    _build_lasso_bundle_model,
    _merge_replay_scores_into_parent_cache,
    _materialize_parent_baseline_contract,
)


def _write_parent_cache(path, values=(10.0, 20.0, 30.0)):
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2023-01-03")], ["A", "B", "C"]],
        names=["date", "stock_code"],
    )
    pd.DataFrame({"composite_value": values}, index=index).to_parquet(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_model(parent_baseline, *, weights=None, ratio=1.0):
    return _build_lasso_bundle_model(
        train_end="2023-01-02",
        weights=weights or {},
        model_info={
            "validation_start": "2023-01-02",
            "selected_blend_ratio": ratio,
            "selected_residual_gamma": 0.0 if ratio == 1.0 else 1.0,
            "preprocess_config": {"ALL": "zscore"},
        },
        synthesis_config={
            "blend_with_original_ratio": 1.0,
            "blend_weights": {},
            "postprocess_cs_standardize": True,
        },
        parent_baseline=parent_baseline,
    )


def _runner(tmp_path, model):
    bundle_path = tmp_path / "lasso_model_bundle_test.json"
    bundle_path.write_text(
        json.dumps({"index": {"models": [model]}}), encoding="utf-8"
    )
    return VectorBTBacktestRunner({}, bundle_path=str(bundle_path))


def test_parent_baseline_is_materialized_next_to_bundle(tmp_path):
    source = tmp_path / "source_parent.parquet"
    digest = _write_parent_cache(source)
    output = tmp_path / "weights"
    output.mkdir()

    contract = _materialize_parent_baseline_contract(
        {
            "parent_baseline": {
                "parent_winner_id": "root.C001",
                "parent_run_id": "root-run",
                "path": str(source),
                "sha256": digest,
            }
        },
        output,
    )

    copied = output / contract["path"]
    assert copied.exists()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest
    assert contract["parent_winner_id"] == "root.C001"
    assert not copied.is_symlink()


def test_replay_scores_fill_window_boundary_and_override_training_cache(tmp_path):
    cache_path = tmp_path / "composite_cache_test.parquet"
    cache_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2023-01-03"), "A"), (pd.Timestamp("2023-01-03"), "B")],
        names=["date", "stock_code"],
    )
    pd.DataFrame(
        {"composite_value": [1.0, 2.0]}, index=cache_index
    ).to_parquet(cache_path)
    snapshot_path = tmp_path / "factor_values_snapshot.parquet"
    pd.DataFrame(
        {
            "factor_date": ["2023-01-03", "2023-01-04"],
            "stock_code": ["A", "C"],
            "composite_score": [10.0, 30.0],
        }
    ).to_parquet(snapshot_path, index=False)

    _merge_replay_scores_into_parent_cache(cache_path, snapshot_path)

    merged = pd.read_parquet(cache_path)["composite_value"]
    assert merged.loc[(pd.Timestamp("2023-01-03"), "A")] == 10.0
    assert merged.loc[(pd.Timestamp("2023-01-03"), "B")] == 2.0
    assert merged.loc[(pd.Timestamp("2023-01-04"), "C")] == 30.0


def test_zero_gamma_parent_fallback_is_kept_as_bundle_model(tmp_path):
    parent = tmp_path / "parent.parquet"
    digest = _write_parent_cache(parent)
    contract = {"path": parent.name, "sha256": digest}

    model = _bundle_model(contract)

    assert model is not None
    assert model["blend_only"] is True
    assert model["weights"] == {}
    assert model["parent_baseline"] == contract
    assert model["blend_with_original_ratio"] == pytest.approx(1.0)


def test_parent_only_bundle_replay_exactly_inherits_parent_scores(tmp_path):
    parent = tmp_path / "parent.parquet"
    digest = _write_parent_cache(parent)
    runner = _runner(tmp_path, _bundle_model({"path": parent.name, "sha256": digest}))

    score, processed, raw = runner._compute_synthesis_score(
        "2023-01-03", ["A", "B", "C"]
    )

    pd.testing.assert_series_equal(
        score,
        pd.Series(
            [10.0, 20.0, 30.0],
            index=pd.Index(["A", "B", "C"], name="stock_code"),
            name="composite_value",
        ),
    )
    assert processed is None
    assert raw is None


def test_active_leaf_bundle_replay_blends_parent_and_residual(tmp_path):
    parent = tmp_path / "parent.parquet"
    digest = _write_parent_cache(parent)
    model = _bundle_model(
        {"path": parent.name, "sha256": digest},
        weights={"Residual": 1.0},
        ratio=0.5,
    )
    runner = _runner(tmp_path, model)

    class Loader:
        @staticmethod
        def load_factors(_date, stocks, factors):
            assert factors == ["Residual"]
            return pd.DataFrame({"Residual": [1.0, 2.0, 3.0]}, index=stocks)

    runner.factor_db_conn = object()
    runner._factor_loader = Loader()
    score, _, _ = runner._compute_synthesis_score("2023-01-03", ["A", "B", "C"])

    expected_residual = pd.Series([-1.0, 0.0, 1.0], index=["A", "B", "C"])
    expected = expected_residual * 0.5 + pd.Series(
        [10.0, 20.0, 30.0], index=["A", "B", "C"]
    ) * 0.5
    pd.testing.assert_series_equal(score, expected)


def test_active_leaf_replay_keeps_parent_stock_when_all_residual_inputs_are_missing(
    tmp_path,
):
    parent = tmp_path / "parent.parquet"
    digest = _write_parent_cache(parent)
    model = _bundle_model(
        {"path": parent.name, "sha256": digest},
        weights={"Residual": 1.0},
        ratio=0.5,
    )
    runner = _runner(tmp_path, model)

    class Loader:
        @staticmethod
        def load_factors(_date, stocks, _factors):
            return pd.DataFrame({"Residual": [1.0, 2.0, float("nan")]}, index=stocks)

    runner.factor_db_conn = object()
    runner._factor_loader = Loader()
    score, _, _ = runner._compute_synthesis_score("2023-01-03", ["A", "B", "C"])

    assert score.index.tolist() == ["A", "B", "C"]
    assert pd.notna(score.loc["C"])


def test_parent_bundle_replay_fails_closed_on_hash_mismatch(tmp_path):
    parent = tmp_path / "parent.parquet"
    _write_parent_cache(parent)
    runner = _runner(
        tmp_path,
        _bundle_model({"path": parent.name, "sha256": "0" * 64}),
    )

    with pytest.raises(ValueError, match="parent baseline hash mismatch"):
        runner._compute_synthesis_score("2023-01-03", ["A", "B", "C"])


def test_parent_required_bundle_rejects_an_incomplete_model_set(tmp_path):
    parent = tmp_path / "parent.parquet"
    digest = _write_parent_cache(parent)
    model = _bundle_model({"path": parent.name, "sha256": digest})
    bundle_path = tmp_path / "lasso_model_bundle_incomplete.json"
    bundle_path.write_text(
        json.dumps(
            {
                "index": {
                    "requires_parent_baseline": True,
                    "expected_model_count": 2,
                    "models": [model],
                }
            }
        ),
        encoding="utf-8",
    )
    runner = VectorBTBacktestRunner({}, bundle_path=str(bundle_path))

    with pytest.raises(ValueError, match="incomplete parent bundle"):
        runner._load_bundle()
