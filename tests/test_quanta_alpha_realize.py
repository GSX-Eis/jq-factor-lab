from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from factor_mining.symbolic.compiler import compile_factor_file
from factor_mining.symbolic.config import AnalysisContext
from factor_mining.symbolic.data_loader import build_training_panel
from factor_mining.symbolic.expression import (
    ExpressionNode,
    evaluate_expression,
    expression_complexity,
    serialize_expression,
)
from factor_mining.symbolic.feature_space import (
    FeatureSpec,
    build_feature_space,
    build_operator_registry,
)
from factor_mining.symbolic.quanta_alpha import (
    QuantaAlphaEvidenceError,
    calibrate_funnel_top_k,
    realize_proposals,
)
from factor_mining.symbolic.structural import structural_gate
from factor_mining.symbolic.search_base import SearchCandidate
from tools.agent_harness.cli import main as harness_main


def _feature(name: str) -> ExpressionNode:
    return ExpressionNode("feature", name)


def _window(value: int) -> ExpressionNode:
    return ExpressionNode("window", value)


def _op(name: str, *children: ExpressionNode) -> ExpressionNode:
    return ExpressionNode("operator", name, children)


def _config() -> dict:
    return {
        "search": {
            "dsl": "full_alpha",
            "windows": [2, 5, 10, 20, 60],
            "constants": {"enabled": True, "values": [-1, 0.5, 1, 2]},
        },
        "features": {
            "daily_price_fields": ["close", "volume"],
            "intraday_5m_fields": [],
            "factor_data_fields": [],
        },
        "label": {"targets": [{"name": "ret_lag1_5", "start": 2, "end": 6}]},
        "objective": {"primary": "rank_ic_stability"},
        "output": {},
    }


def _panel() -> pd.DataFrame:
    rows = []
    dates = list(pd.date_range("2018-01-02", periods=8, freq="D")) + list(
        pd.date_range("2023-01-03", periods=6, freq="D")
    )
    for date_index, date in enumerate(dates):
        for code_index, code in enumerate(("a", "b", "c", "d")):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "label": float(code_index - 1.5),
                    "daily_close": 10.0 + date_index + code_index,
                    "daily_volume": 100.0
                    + date_index * (code_index + 1)
                    + (3 - code_index) * 4,
                }
            )
    return pd.DataFrame(rows)


def test_campaign_panel_hash_ignores_requested_feature_subset() -> None:
    import factor_mining.symbolic.quanta_alpha as realization

    base = _panel().assign(
        label__ret_1_5=lambda frame: frame["label"],
        anchor__existing=lambda frame: frame["daily_close"],
    )
    first = base.assign(intraday_feature_a=1.0)
    second = base.assign(intraday_feature_b=2.0)

    assert realization._panel_hash(first, None) == realization._panel_hash(second, None)
    changed = second.copy()
    changed.loc[0, "label"] += 0.01
    assert realization._panel_hash(first, None) != realization._panel_hash(
        changed, None
    )


def test_where_treats_missing_condition_as_false() -> None:
    from factor_mining.symbolic.operator_runtime import runtime_namespace

    condition = pd.Series([True, pd.NA, False], dtype="boolean")
    runtime = runtime_namespace({}, pd.Series(range(3)), pd.Series(["a"] * 3))
    result = runtime["where"](condition, pd.Series([1.0, 1.0, 1.0]), 0.0)

    assert result.tolist() == [1.0, 0.0, 0.0]


def _large_panel() -> pd.DataFrame:
    rows = []
    dates = list(pd.bdate_range("2021-01-04", periods=300)) + list(
        pd.bdate_range("2023-01-03", periods=150)
    )
    for date_index, date in enumerate(dates):
        for code_index, code in enumerate(("a", "b", "c", "d")):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "label": float(code_index - 1.5),
                    "daily_close": 10.0 + date_index + code_index,
                    "daily_volume": 100.0
                    + date_index * (code_index + 1)
                    + (3 - code_index) * 4,
                }
            )
    return pd.DataFrame(rows)


def _proposal(proposal_id: str, expression: ExpressionNode) -> dict:
    hypothesis = f"hypothesis {proposal_id}"
    semantic = f"semantic {proposal_id}"
    expression_payload = serialize_expression(expression)

    def hash_payload(value):
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    return {
        "proposal_id": proposal_id,
        "trajectory_id": "trajectory-1",
        "hypothesis": hypothesis,
        "semantic_description": semantic,
        "expression": expression_payload,
        "expression_hash": hash_payload(expression_payload),
        "consistency_report": {
            "reviewer": "independent-test-reviewer",
            "severity": "pass",
            "hypothesis_hash": hash_payload(hypothesis),
            "semantic_hash": hash_payload(semantic),
            "expression_hash": hash_payload(expression_payload),
            "rewrite_count": 0,
        },
    }


def _frozen_campaign_assets(tmp_path: Path) -> dict:
    snapshot = tmp_path / "pool_snapshot.json"
    snapshot.write_text(json.dumps({"pools": {"wind_micro": {
        "2018-01-01": ["a", "b", "c", "d"]}}}), encoding="utf-8")
    settings = {
        "factor_history_start_date": "2018-01-01",
        "pool_snapshot_path": str(snapshot),
        "pool_snapshot_read_only": True,
        "pool_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    analysis = {
        "dynamic_pools": {"wind_micro": {"enabled": True}},
        "analysis_period": {"start_date": "2018-01-01", "end_date": "2023-12-31"},
        "analysis_settings": settings,
    }
    paths = {}
    for key in ("analysis_config", "production_analysis_config"):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(analysis), encoding="utf-8")
        paths[key] = str(path)
    return {**paths, "data_snapshot_hash": "d" * 64, "research_start": "2018-01-01"}


def _write_input(
    tmp_path: Path, *, profile: str, top_k: int | None = None
) -> tuple[Path, Path]:
    panel_path = tmp_path / "panel.pkl"
    _panel().to_pickle(panel_path)
    proposals = [
        _proposal("close_rank", _op("cs_rank", _feature("daily_close"))),
        _proposal("close_z", _op("cs_zscore", _feature("daily_close"))),
        _proposal("close_mean", _op("ts_mean", _feature("daily_close"), _window(2))),
    ]
    payload = {
        **_frozen_campaign_assets(tmp_path),
        "profile": profile,
        "symbolic_config": _config(),
        "panel_path": str(panel_path),
        "windows": {
            "train": {"start": "2018-01-01", "end": "2022-12-31", "sample_dates": 2},
            "validation": {
                "start": "2023-01-01",
                "end": "2023-12-31",
                "sample_dates": 2,
            },
        },
        "proposals": proposals,
        "zoo_expressions": [
            {
                "proposal_id": "prior-volume-mean",
                "expression": serialize_expression(
                    _op("ts_mean", _feature("daily_volume"), _window(2))
                ),
            }
        ],
        "output_dir": str(tmp_path / "compiled"),
    }
    if top_k is not None:
        payload["top_k"] = top_k
    input_path = tmp_path / f"{profile}.json"
    output_path = tmp_path / f"{profile}_result.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    return input_path, output_path


def test_table7_registry_is_exportable_and_has_lookback_metadata() -> None:
    registry = build_operator_registry(_config())
    expected = {
        "cs_mean",
        "ts_quantile",
        "ts_covariance",
        "ts_pctchange",
        "ts_var",
        "ts_product",
        "prod",
        "sma",
        "wma",
        "count",
        "sum_if",
        "regbeta",
        "regresi",
        "decaylinear",
        "rsi",
        "macd",
        "bb_upper",
        "where",
    }
    assert expected <= set(registry)
    assert registry["delay"].lookback == "lag"
    assert registry["macd"].lookback == "rolling"
    assert registry["macd"].nan_behavior == "full_window"
    assert registry["cs_rank"].nan_behavior == "ignore_within_group"
    assert registry["where"].nan_behavior == "numpy_boolean"
    assert all(spec.nan_behavior for spec in registry.values())


def test_structural_gate_rejects_free_args_bad_types_and_large_subtree() -> None:
    feature_space = build_feature_space(_config())
    free_args = _op(
        "add",
        ExpressionNode("constant", 1.0),
        ExpressionNode("constant", 2.0),
    )
    assert (
        "free_argument_ratio_exceeded"
        in structural_gate(free_args, feature_space)["reasons"]
    )

    bad_condition = _op(
        "where",
        _feature("daily_close"),
        _feature("daily_close"),
        _feature("daily_volume"),
    )
    assert any(
        reason.startswith("invalid_ast:")
        for reason in structural_gate(bad_condition, feature_space)["reasons"]
    )

    shared = _feature("daily_close")
    for _ in range(5):
        shared = _op("cs_rank", shared)
    candidate = _op("add", shared, _feature("daily_volume"))
    report = structural_gate(
        candidate, feature_space, factor_zoo=[shared], max_common_subtree=4
    )
    assert "largest_common_subtree_exceeded" in report["reasons"]


def test_realize_profiles_use_full_parity_and_sampled_funnel(tmp_path: Path) -> None:
    parity_input, parity_output = _write_input(tmp_path, profile="paper_v3_parity")
    parity = realize_proposals(parity_input, parity_output)
    assert parity["windows"]["train"]["sample_dates"] == 8
    assert parity["windows"]["validation"]["sample_dates"] == 6
    assert len(parity["selected_proposal_ids"]) == 3
    assert all(row["source"]["sha256"] for row in parity["proposals"])
    assert all(row["funnel_metrics"]["train_dates"] == 8 for row in parity["proposals"])
    assert all(
        row["funnel_metrics"]["validation_dates"] == 6 for row in parity["proposals"]
    )
    proposal_ids = {row["proposal_id"] for row in parity["proposals"]}
    assert all(
        set(row["output_correlations"])
        == proposal_ids.union({"prior-volume-mean"}) - {row["proposal_id"]}
        for row in parity["proposals"]
    )
    assert (
        parity["output_hash"]
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in parity.items() if key != "output_hash"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )

    funnel_input, funnel_output = _write_input(
        tmp_path, profile="paper_v3_funnel", top_k=2
    )
    funnel = realize_proposals(funnel_input, funnel_output)
    assert funnel["windows"]["train"]["sample_dates"] == 2
    assert funnel["windows"]["validation"]["sample_dates"] == 2
    assert len(funnel["selected_proposal_ids"]) == 2


def test_parity_emits_prospective_funnel_metrics_and_ranks_by_them(
    tmp_path: Path,
) -> None:
    input_path, output_path = _write_input(tmp_path, profile="paper_v3_parity")
    _large_panel().to_pickle(tmp_path / "panel.pkl")
    result = realize_proposals(input_path, output_path)

    assert result["windows"]["train"]["sample_dates"] == 300
    assert result["windows"]["validation"]["sample_dates"] == 150
    assert all(
        row["funnel_metrics"]["train_dates"] == 252
        and row["funnel_metrics"]["validation_dates"] == 126
        for row in result["proposals"]
    )


def test_parity_allows_explicit_smaller_shadow_funnel(tmp_path: Path) -> None:
    import factor_mining.symbolic.quanta_alpha as realization

    input_path, output_path = _write_input(tmp_path, profile="paper_v3_parity")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["shadow_funnel"] = {"train_sample_dates": 3, "validation_sample_dates": 2}
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    result = realize_proposals(input_path, output_path)

    assert result["windows"]["shadow_funnel"] == {
        "train_sample_dates": 3,
        "validation_sample_dates": 2,
    }
    assert all(
        row["funnel_metrics"]["train_dates"] == 3
        and row["funnel_metrics"]["validation_dates"] == 2
        for row in result["proposals"]
    )
    assert all(
        row["fast_metrics"]["max_abs_output_correlation"]
        == max(abs(value) for value in row["output_correlations"].values())
        for row in result["proposals"]
    )

    full_winner = {
        "expression_hash": "a",
        "funnel_gate": {"passed": True},
        "fast_metrics": {"sampled_marginal_rank_ic": 0.9},
        "funnel_metrics": {
            "sampled_marginal_rank_ic": 0.1,
            "validation_rank_ic": 0.1,
            "max_abs_output_correlation": 0.1,
            "complexity": 1,
        },
    }
    funnel_winner = {
        "expression_hash": "b",
        "funnel_gate": {"passed": True},
        "fast_metrics": {"sampled_marginal_rank_ic": 0.2},
        "funnel_metrics": {
            "sampled_marginal_rank_ic": 0.8,
            "validation_rank_ic": 0.8,
            "max_abs_output_correlation": 0.1,
            "complexity": 1,
        },
    }
    ranked = sorted(
        [full_winner, funnel_winner],
        key=realization._prospective_rank_key,
    )
    assert ranked[0] is funnel_winner


def test_fast_gate_freezes_train_direction_and_rejects_only_sign_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import factor_mining.symbolic.quanta_alpha as realization

    input_path, output_path = _write_input(tmp_path, profile="paper_v3_parity")
    metrics = iter(
        [
            (-0.06, -0.04),
            (-0.05, -0.03),
            (-0.06, 0.04),
            (-0.05, 0.03),
            (0.06, 0.04),
            (0.05, 0.03),
        ]
    )

    def screen(*_args):
        train_ic, validation_ic = next(metrics)
        return {
            "train_rank_ic": train_ic,
            "validation_rank_ic": validation_ic,
            "sampled_marginal_rank_ic": validation_ic / 2,
            "coverage": 1.0,
            "variance": 1.0,
            "anchor_count": 0,
            "max_abs_anchor_correlation": None,
            "max_abs_output_correlation": None,
            "train_dates": 100,
            "validation_dates": 20,
        }

    monkeypatch.setattr(realization, "_screen_metrics", screen)
    result = realize_proposals(input_path, output_path)
    by_id = {row["proposal_id"]: row for row in result["proposals"]}

    assert by_id["close_rank"]["eligible"] is True
    assert by_id["close_rank"]["fast_metrics"]["frozen_direction"] == -1
    assert (
        by_id["close_rank"]["fast_metrics"]["direction_adjusted_validation_rank_ic"]
        == 0.04
    )
    assert by_id["close_z"]["eligible"] is False
    assert by_id["close_z"]["fast_gate"]["reasons"] == [
        "train_validation_direction_mismatch"
    ]
    assert by_id["close_mean"]["eligible"] is True


def test_funnel_calibration_requires_recall_and_thirty_percent_reduction() -> None:
    rows = (
        [
            {"proposal_id": f"a{rank}", "rank": rank, "production_pass": rank == 1}
            for rank in (1, 2, 3)
        ]
        + [
            {"proposal_id": f"b{rank}", "rank": rank, "production_pass": rank == 2}
            for rank in (1, 2, 3)
        ]
        + [
            {
                "proposal_id": "structural-reject",
                "rank": 1,
                "eligible": False,
                "production_pass": False,
            }
        ]
    )
    calibrated = calibrate_funnel_top_k(rows)
    assert calibrated == {
        "enabled": True,
        "top_k": 2,
        "pass_recall": 1.0,
        "reduction": 1 / 3,
        "eligible_count": 6,
        "reason": "calibrated",
    }

    rows[5]["production_pass"] = True
    disabled = calibrate_funnel_top_k(rows)
    assert disabled["enabled"] is False
    assert disabled["best_top_k"] == 3
    assert disabled["pass_recall"] == 1.0
    assert disabled["reduction"] == 0.0


def test_quanta_alpha_realize_harness_writes_frozen_artifact(tmp_path: Path) -> None:
    input_path, output_path = _write_input(tmp_path, profile="paper_v3_parity")
    exit_code = harness_main(
        [
            "quanta-alpha-realize",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--output-root",
            str(tmp_path / "harness"),
            "--run-id",
            "realize-test",
        ]
    )
    assert exit_code == 0
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["schema_version"]
        == "quanta_alpha_realization_v1"
    )
    assert (
        tmp_path / "harness/quanta-alpha-realize/realize-test/summary.json"
    ).is_file()


def test_factor_data_override_reaches_feature_space_and_panel_builder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import factor_mining.symbolic.quanta_alpha as realization

    captured: dict[str, list[str]] = {}

    def fake_build_training_panel(config, _context, **_kwargs):
        assert _kwargs["drop_missing_labels"] is False
        captured["fields"] = list(config["features"]["factor_data_fields"])
        return _panel().assign(fd_roe_ttm=1.0, fd_roic_ttm=2.0)

    monkeypatch.setattr(
        realization,
        "load_analysis_context",
        lambda _path: AnalysisContext(
            config_path="fixture.json",
            raw_config={},
            database_config={},
            analysis_config={},
            pool_configs={},
            runtime_config={},
        ),
    )
    monkeypatch.setattr(realization, "build_training_panel", fake_build_training_panel)
    config, panel, panel_source = realization._load_config_and_panel(
        {
            "symbolic_config": _config(),
            "analysis_config": "frozen-analysis.json",
            "factor_data_fields": ["roe_ttm", "roic_ttm", "roe_ttm"],
        },
        tmp_path / "input.json",
    )

    assert captured["fields"] == ["roe_ttm", "roic_ttm"]
    assert {"fd_roe_ttm", "fd_roic_ttm"} <= set(build_feature_space(config).features)
    assert {"fd_roe_ttm", "fd_roic_ttm"} <= set(panel)
    assert panel_source is None


def test_realization_loads_only_features_referenced_by_the_trajectory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import factor_mining.symbolic.quanta_alpha as realization

    captured: list[dict] = []
    monkeypatch.setattr(
        realization,
        "load_analysis_context",
        lambda _path: AnalysisContext(
            config_path="fixture.json",
            raw_config={},
            database_config={},
            analysis_config={},
            pool_configs={},
            runtime_config={},
        ),
    )
    monkeypatch.setattr(
        realization,
        "build_training_panel",
        lambda config, _context, **_kwargs: (
            captured.append(config["features"].copy()) or _panel()
        ),
    )
    for feature in ("daily_close", "daily_volume"):
        realization._load_config_and_panel(
            {
                "symbolic_config": _config(),
                "analysis_config": "frozen-analysis.json",
                "proposals": [{"expression": serialize_expression(_feature(feature))}],
            },
            tmp_path / "input.json",
        )

    assert [item["daily_price_fields"] for item in captured] == [["close"], ["volume"]]


def test_intraday_feature_cache_computes_only_missing_specs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import factor_mining.symbolic.data_loader as loader

    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-02 09:35", "2023-01-02 09:40", "2023-01-03 09:35"]
            ),
            "code": ["a", "a", "a"],
            "volume": [1.0, 2.0, 3.0],
            "money": [10.0, 30.0, 20.0],
        }
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        loader,
        "load_intraday_5m_panel",
        lambda _context, fields, _stocks: calls.append(list(fields)) or raw,
    )
    database = tmp_path / "prices.duckdb"
    database.write_bytes(b"snapshot")
    context = AnalysisContext(
        config_path="fixture.json",
        raw_config={"pit_adjustment": {"enabled": False}},
        database_config={},
        analysis_config={},
        pool_configs={},
        runtime_config={
            "cache_dir": str(tmp_path / "cache"),
            "intraday_cache_dir": str(tmp_path / "shared-intraday-cache"),
            "duckdb_path": str(database),
            "data_snapshot_hash": "frozen-source-hash",
            "start_date": "2023-01-01",
            "end_date": "2023-01-03",
        },
    )
    volume = FeatureSpec("intraday_sum_volume", "price_5m", "volume", "5m", "sum")
    money = FeatureSpec("intraday_range_money", "price_5m", "money", "5m", "range")

    loader._load_cached_intraday_feature_panel(context, {volume.name: volume}, ["a"])
    loader._load_cached_intraday_feature_panel(
        context,
        {volume.name: volume, money.name: money},
        ["a"],
    )
    loader._load_cached_intraday_feature_panel(
        context,
        {volume.name: volume, money.name: money},
        ["a"],
    )

    assert calls == [["volume"], ["money"]]
    assert (
        len(
            list(
                (tmp_path / "shared-intraday-cache/intraday_features").glob("*.parquet")
            )
        )
        == 2
    )
    assert not (tmp_path / "cache/intraday_features").exists()


def test_requested_anchor_evidence_must_be_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import factor_mining.symbolic.quanta_alpha as realization

    monkeypatch.setattr(
        realization,
        "load_analysis_context",
        lambda _path: AnalysisContext(
            config_path="fixture.json",
            raw_config={},
            database_config={},
            analysis_config={},
            pool_configs={},
            runtime_config={},
        ),
    )
    monkeypatch.setattr(
        realization, "build_training_panel", lambda *_args, **_kwargs: _panel()
    )

    with pytest.raises(
        QuantaAlphaEvidenceError, match="missing requested anchor evidence"
    ):
        realization._load_config_and_panel(
            {
                "symbolic_config": _config(),
                "analysis_config": "frozen-analysis.json",
                "anchors": ["ExistingFactor_wind_micro"],
                "anchor_db_path": "frozen-anchors.duckdb",
            },
            tmp_path / "input.json",
        )


def test_compiler_matches_ast_evaluator_on_same_daily_fixture(tmp_path: Path) -> None:
    config = _config()
    feature_space = build_feature_space(config)
    expression = _op("ts_mean", _feature("daily_close"), _window(2))
    candidate = SearchCandidate(
        expression=expression,
        train_metrics={},
        valid_metrics={},
        score=0.0,
        complexity=expression_complexity(expression),
    )
    factor_path = compile_factor_file(
        candidate,
        config,
        feature_space,
        tmp_path / "compiler_parity.py",
        factor_value_cache_dir=tmp_path / "factor_value_cache",
    )
    spec = importlib.util.spec_from_file_location(
        "compiler_parity_fixture", factor_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factor_cls = module.CompilerParity
    assert factor_cls.FACTOR_VALUE_CACHE_DIR == str(
        (tmp_path / "factor_value_cache").resolve()
    )

    panel = _panel().loc[lambda frame: frame["date"].dt.year == 2023].copy()
    raw = panel[["date", "code"]].copy()
    raw["close"] = panel["daily_close"]
    raw["factor"] = 1.0
    expected = evaluate_expression(expression, panel).rename("factor")
    expected_frame = pd.DataFrame(
        {"date": panel["date"], "code": panel["code"], "factor": expected}
    ).pivot(index="date", columns="code", values="factor")

    factor = factor_cls(
        None,
        "compiler_parity",
        "2023-01-01",
        "2023-12-31",
        sorted(panel["code"].unique()),
        pit_enabled=False,
    )
    actual = factor.calculate_factor(raw)
    pd.testing.assert_frame_equal(actual, expected_frame, check_names=False)


def test_panel_contract_keeps_pit_and_intraday_features_point_in_time(
    monkeypatch,
) -> None:
    import factor_mining.symbolic.data_loader as loader

    dates = pd.date_range("2023-01-02", periods=3, freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "code": ["a"] * 3,
            "daily_close": [10.0, 11.0, 13.0],
        }
    )
    # factor_data is an upstream-materialized daily PIT snapshot: the value is
    # absent before its release date and present from release onward.
    pit = pd.DataFrame(
        {
            "date": dates,
            "code": ["a"] * 3,
            "fd_metric": [float("nan"), 7.0, 7.0],
        }
    )
    intraday_rows = []
    for day_index, date in enumerate(dates):
        for minute, value in (
            ("09:35", 100.0 + day_index),
            ("14:55", 200.0 + day_index),
        ):
            intraday_rows.append(
                {
                    "date": pd.Timestamp(f"{date.date()} {minute}"),
                    "code": "a",
                    "close": value,
                }
            )
    intraday = pd.DataFrame(intraday_rows)

    monkeypatch.setattr(loader, "resolve_universe", lambda _context: ["a"])
    monkeypatch.setattr(loader, "load_daily_price_panel", lambda *_args: daily)
    monkeypatch.setattr(loader, "load_factor_data_panel", lambda *_args: pit)
    monkeypatch.setattr(loader, "load_intraday_5m_panel", lambda *_args: intraday)
    config = {
        **_config(),
        "features": {
            "daily_price_fields": ["close"],
            "factor_data_fields": ["metric"],
            "intraday_5m_fields": ["close"],
            "intraday_aggregators": ["last"],
        },
        "label": {"target_period": 1},
    }
    context = AnalysisContext(
        config_path="fixture.json",
        raw_config={},
        database_config={},
        analysis_config={},
        pool_configs={},
        runtime_config={"start_date": "2023-01-02", "end_date": "2023-01-04"},
    )
    panel = build_training_panel(config, context)

    assert pd.isna(panel.loc[panel["date"] == dates[0], "fd_metric"]).all()
    assert panel.loc[panel["date"] == dates[1], "fd_metric"].item() == 7.0
    assert panel.loc[panel["date"] == dates[0], "intraday_last_close"].item() == 200.0
    assert panel.loc[panel["date"] == dates[1], "intraday_last_close"].item() == 201.0
    assert panel.loc[panel["date"] == dates[0], "daily_close"].item() == 10.0
    assert panel.loc[panel["date"] == dates[0], "label"].item() == pytest.approx(0.1)


def test_symbolic_universe_prefers_frozen_pool_snapshot(tmp_path: Path) -> None:
    import factor_mining.symbolic.data_loader as loader

    snapshot = tmp_path / "pool_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "pools": {
                    "wind_micro": {
                        "2023-01-02": ["a", "b"],
                        "2023-01-09": ["b", "c"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    context = AnalysisContext(
        config_path="fixture.json",
        raw_config={},
        database_config={},
        analysis_config={},
        pool_configs={"wind_micro": {}},
        runtime_config={
            "pool_snapshot_path": str(snapshot),
            "code_list": ["fixture-only"],
        },
    )

    assert loader.resolve_universe(context) == ["a", "b", "c"]


def test_future_label_availability_does_not_select_ast_rows(tmp_path, monkeypatch):
    import factor_mining.symbolic.data_loader as loader
    import factor_mining.symbolic.quanta_alpha as realization

    dates = pd.date_range("2023-01-02", periods=5, freq="D")
    daily = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "daily_close": 10.0 + day + stock,
                "daily_volume": volume + day * (stock + 1),
            }
            for day, date in enumerate(dates)
            for stock, (code, volume) in enumerate(
                (("a", 10.0), ("b", 30.0), ("c", 100.0))
            )
        ]
    )
    context = AnalysisContext(
        config_path="fixture.json",
        raw_config={},
        database_config={},
        analysis_config={},
        pool_configs={},
        runtime_config={},
    )
    monkeypatch.setattr(realization, "load_analysis_context", lambda _path: context)
    monkeypatch.setattr(loader, "resolve_universe", lambda _context: ["a", "b", "c"])
    monkeypatch.setattr(loader, "load_daily_price_panel", lambda *_args: daily.copy())
    monkeypatch.setattr(loader, "load_factor_data_panel", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(
        loader, "_load_cached_intraday_feature_panel", lambda *_args: pd.DataFrame()
    )
    monkeypatch.setattr(loader, "_pool_snapshot", lambda _context: None)
    monkeypatch.setattr(loader, "binding_contract", lambda: {})
    config = {**_config(), "label": {"target_period": 2}}
    payload = {"symbolic_config": config, "analysis_config": "fixture.json"}
    _, before, _ = realization._load_config_and_panel(payload, tmp_path / "input.json")
    # Only a future close becomes unavailable; historical features are identical.
    daily.loc[(daily["date"] == dates[2]) & (daily["code"] == "b"), "daily_close"] = (
        float("nan")
    )
    _, after, _ = realization._load_config_and_panel(payload, tmp_path / "input.json")
    assert (
        before.loc[(before["date"] == dates[0]) & (before["code"] == "b"), "label"]
        .notna()
        .all()
    )
    assert (
        after.loc[(after["date"] == dates[0]) & (after["code"] == "b"), "label"]
        .isna()
        .all()
    )
    for expression in (
        _op("cs_demean", _feature("daily_volume")),
        _op("ts_mean", _feature("daily_volume"), _window(2)),
    ):

        def historical_values(panel):
            values = panel[["date", "code"]].copy()
            values["factor"] = evaluate_expression(expression, panel)
            return values.loc[values["date"] < dates[2]].set_index(["date", "code"])[
                "factor"
            ]

        pd.testing.assert_series_equal(
            historical_values(before), historical_values(after)
        )
    assert len(after) == len(daily)
    # Legacy training callers retain their default and cannot share the new cache.
    legacy = loader.build_training_panel(config, context)
    assert legacy["label"].notna().all()
    context.runtime_config["cache_dir"] = str(tmp_path / "cache")
    assert loader._panel_cache_path(
        config, context, ["a", "b", "c"], []
    ) != loader._panel_cache_path(
        config, context, ["a", "b", "c"], [], drop_missing_labels=False
    )
