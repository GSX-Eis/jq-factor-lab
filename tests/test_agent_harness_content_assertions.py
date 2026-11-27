from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tools.agent_harness import runner
from tools.agent_harness.run_result import SCHEMA_VERSION


def bound(tmp_path, action, metrics, *, completed=1, **context):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "command": action,
        "run_id": "run",
        "execution_id": "attempt",
        "config_sha256": "a" * 64,
        "execution": {
            "status": "succeeded", "expected": completed, "completed": completed,
            "failed": 0, "errors": [],
        },
        "metrics": metrics,
        "artifacts": [],
    }), encoding="utf-8")
    return runner._bound_result_metrics(
        path, action=action, run_id="run", execution_id="attempt",
        source_sha256="a" * 64, **context,
    )


@pytest.mark.parametrize("action", [
    "factor-synthesis", "single-factor-analysis", "export-strategy",
    "verify-alignment", "monitor-refresh", "time-series-analysis", "other",
])
@pytest.mark.parametrize("metrics", [{}, {"items": []}, {"note": "done"}, {"metric": None}, {"metric": float("nan")}])
def test_success_with_empty_or_invalid_metrics_is_rejected(tmp_path, action, metrics):
    with pytest.raises(ValueError):
        bound(tmp_path, action, metrics)


def synthesis_item(name, factor_name):
    return dict(synthesis_name=name, factor_name=factor_name, status="succeeded", overall_ic=0.0)


@pytest.mark.parametrize("top_k", [False, True])
def test_synthesis_requires_all_configured_outputs_and_pca_components(tmp_path, top_k):
    from tools.agent_harness.run_result import expected_backtest_trials

    config = {
        "synthesis_methods": {"weighted": {"method": "weighted_sum"}, "principal": {"method": "pca"}},
        "output_settings": {"composite_factor_prefix": "combo"},
    }
    rows = [
        synthesis_item("weighted", "combo_weighted"),
        synthesis_item("principal", "combo_principal_PC1"),
        synthesis_item("principal", "combo_principal_PC2"),
    ]
    if top_k:
        # PCA trials need no deployment bundle. Keep the scope of this check
        # on synthesis content after the complete top-k trial ledger passes.
        del config["synthesis_methods"]["weighted"]
        rows.pop(0)
        config["performance_evaluation"] = {"top_k_strategy": {"enabled": True}}
    runs = [
        dict(trial, annual_return=0.1, sharpe_ratio=1.0, max_drawdown=-0.1)
        for trial in expected_backtest_trials(config)
    ]
    metrics = dict(ic_summaries=rows, backtest_runs=runs)
    assert bound(tmp_path, "factor-synthesis", metrics, completed=len(rows), config_payload=config)
    with pytest.raises(ValueError, match="coverage mismatch"):
        bound(tmp_path, "factor-synthesis", dict(metrics, ic_summaries=rows[:-1]),
              completed=len(rows) - 1, config_payload=config)
    with pytest.raises(ValueError, match="completed count mismatch"):
        bound(tmp_path, "factor-synthesis", metrics, completed=len(rows) + 1, config_payload=config)
    with pytest.raises(ValueError, match="duplicate synthesis item"):
        bound(tmp_path, "factor-synthesis", dict(metrics, ic_summaries=[*rows, rows[0]]),
              completed=len(rows) + 1, config_payload=config)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", dict(metrics, ic_summaries=[
            dict(rows[0], overall_ic=float("nan"), overall_ir=None, periods={"1D": {"mean_ic": float("inf")}}),
            *rows[1:],
        ]), completed=len(rows), config_payload=config)


def test_synthesis_rejects_composite_only_on_normal_daily_ic_request(tmp_path):
    item = dict(synthesis_name="weighted", factor_name="weighted", status="succeeded",
                evaluation_mode="composite_only", finite_score_count=50)
    daily_cfg = {
        "synthesis_methods": {"weighted": {"method": "weighted_sum"}},
        "performance_evaluation": {"skip_alphalens_analysis": False},
        "frequency": "1d",
    }
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [item]}, config_payload=daily_cfg)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [item]},
              config_payload={"synthesis_methods": {"weighted": {"method": "weighted_sum"}}})
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {
            "synthesis_diagnostics": [dict(item, skip_reason="skip_alphalens_analysis")],
        }, config_payload=daily_cfg)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [item]},
              config_payload={
                  "synthesis_methods": {"weighted": {"method": "weighted_sum"}},
                  "synthesis_reuse": {"enabled": True},
              })
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {
            "synthesis_diagnostics": [dict(item, skip_reason="single_instrument", stock_count=1)],
        }, config_payload=daily_cfg)


def test_synthesis_allows_composite_only_for_justified_skip_paths_and_cli_overrides(tmp_path):
    item = dict(synthesis_name="weighted", factor_name="weighted", status="succeeded",
                evaluation_mode="composite_only", finite_score_count=50)
    base_cfg = {"synthesis_methods": {"weighted": {"method": "weighted_sum"}}}

    from slots.jq_ashare_local import daily_slot_config

    local_db_cfg = {
        **base_cfg,
        "database": {"local_db": {"duckdb_path": str(tmp_path / "local.duckdb")}},
    }

    assert bound(tmp_path, "factor-synthesis", {
        "synthesis_diagnostics": [dict(item, skip_reason="skip_alphalens_analysis")],
    }, config_payload={**local_db_cfg, "performance_evaluation": {"skip_alphalens_analysis": True}})

    assert bound(tmp_path, "factor-synthesis", {
        "synthesis_diagnostics": [dict(item, skip_reason="non_daily_frequency", frequency="1h")],
    }, config_payload={**local_db_cfg, "frequency": "1h"})

    slot_payload = daily_slot_config().as_dict()
    slot_payload["bar_feed"]["options"]["frequency"] = "1h"
    slot_file = tmp_path / "slot_1h.json"
    slot_file.write_text(json.dumps(slot_payload), encoding="utf-8")
    assert bound(tmp_path, "factor-synthesis", {
        "synthesis_diagnostics": [dict(item, skip_reason="non_daily_frequency", frequency="60min")],
    }, config_payload=local_db_cfg, slot_config=str(slot_file))

    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {
            "synthesis_diagnostics": [dict(item, skip_reason="non_daily_frequency", frequency="1h")],
        }, config_payload=local_db_cfg, slot_config=str(slot_file), frequency="1d")

    assert bound(tmp_path, "factor-synthesis", {
        "synthesis_diagnostics": [dict(item, skip_reason="non_daily_frequency", frequency="5min")],
    }, config_payload={**local_db_cfg, "frequency": "1d"}, frequency="5min")

    lasso_dir = tmp_path / "reuse_store"
    lasso_dir.mkdir()
    reuse_bundle = lasso_dir / "lasso_model_bundle_lasso_a.json"
    reuse_cache = lasso_dir / "composite_cache_lasso_a.parquet"
    reuse_bundle.write_text(json.dumps({
        "index": {
            "synthesis_name": "lasso_a",
            "models": [{
                "train_end": "2023-01-01",
                "feature_names": ["f1", "f2"],
                "weights": {"f1": 0.6, "f2": 0.4},
                "blend_only": False,
            }],
        },
    }), encoding="utf-8")
    import pandas as pd
    cache_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2023-01-03"), "000001.XSHE"),
            (pd.Timestamp("2023-01-03"), "000002.XSHE"),
            (pd.Timestamp("2023-01-04"), "000001.XSHE"),
        ],
        names=["date", "stock_code"],
    )
    pd.DataFrame({"composite_value": [0.1, 0.2, 0.3]}, index=cache_index).to_parquet(reuse_cache)
    import hashlib
    bundle_sha = hashlib.sha256(reuse_bundle.read_bytes()).hexdigest()
    cache_sha = hashlib.sha256(reuse_cache.read_bytes()).hexdigest()

    lasso_cfg = {
        "synthesis_methods": {"lasso_a": {"method": "lasso"}},
        "synthesis_reuse": {"enabled": True, "source_dir": str(lasso_dir)},
        "database": {"local_db": {"duckdb_path": str(tmp_path / "local.duckdb")}},
    }
    lasso_item = dict(
        synthesis_name="lasso_a", factor_name="lasso_a", status="succeeded",
        evaluation_mode="composite_only", skip_reason="synthesis_reuse", finite_score_count=3,
        stock_count=2, reuse_bundle_sha256=bundle_sha, reuse_cache_sha256=cache_sha,
    )
    assert bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [lasso_item]}, config_payload=lasso_cfg)

    # Reuse fallback when bundle/cache missing, hash mismatches, or label_mode disallows reuse.
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [lasso_item]},
              config_payload={**lasso_cfg, "label_definition": {"mode": "next_rebalance_open"}})
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {
            "synthesis_diagnostics": [dict(lasso_item, reuse_cache_sha256="0" * 64)],
        }, config_payload=lasso_cfg)
    reuse_cache.unlink()
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [lasso_item]}, config_payload=lasso_cfg)

    assert bound(tmp_path, "factor-synthesis", {
        "synthesis_diagnostics": [dict(item, skip_reason="no_local_db")],
    }, config_payload=base_cfg, slot_config="jq_ashare_local")

    # Single-instrument requires independent request scope or SHA-256 verified composite_scores artifact.
    assert bound(tmp_path, "factor-synthesis", {
        "synthesis_diagnostics": [dict(
            item, skip_reason="single_instrument", stock_count=1, instruments=["IF2403"],
        )],
    }, config_payload={**local_db_cfg, "code_list": ["IF2403"]})
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {
            "synthesis_diagnostics": [dict(
                item, skip_reason="single_instrument", stock_count=1, instruments=["IF2403"],
            )],
        }, config_payload=local_db_cfg)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {
            "synthesis_diagnostics": [dict(
                item, skip_reason="single_instrument", stock_count=1, instruments=["IF2403"],
            )],
        }, config_payload={**local_db_cfg, "code_list": ["IC2403"]})

    bad_item = dict(item, finite_score_count=0)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [bad_item]},
              config_payload={**local_db_cfg, "performance_evaluation": {"skip_alphalens_analysis": True}})


def test_synthesis_single_instrument_artifact_and_db_bindings_resolution(tmp_path, monkeypatch):
    from tools.agent_harness import runner
    from tools.agent_harness.run_result import config_sha256, execution_result

    cfg_path = tmp_path / "synth.json"
    cfg_path.write_text(json.dumps({
        "synthesis_methods": {"weighted": {"method": "weighted_sum"}},
        "database": {"local_db": {}},
    }), encoding="utf-8")

    # QUANT_DATA_BINDINGS populates local_db.duckdb_path -> ResultWriter exists -> no_local_db rejected.
    bindings_payload = json.dumps({
        "input_db": str(tmp_path / "input.duckdb"),
        "output_db": str(tmp_path / "output.duckdb"),
    })
    no_db_diag = dict(
        synthesis_name="weighted", factor_name="weighted", status="succeeded",
        evaluation_mode="composite_only", skip_reason="no_local_db", finite_score_count=10,
    )
    cfg_payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [no_db_diag]},
                 config_payload=cfg_payload, slot_config="jq_ashare_local")
    monkeypatch.setenv("QUANT_DATA_BINDINGS", bindings_payload)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", {"synthesis_diagnostics": [no_db_diag]},
              config_payload=cfg_payload, slot_config="jq_ashare_local")
    monkeypatch.delenv("QUANT_DATA_BINDINGS", raising=False)

    # Verified composite_scores artifact authorizes single_instrument when request omits code_list.
    res_path = tmp_path / "attempt" / "result.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    scores_csv = res_path.parent / "composite_scores_weighted_weighted.csv"
    scores_csv.write_text("date,stock_code,composite_value\n2023-01-03,IF2403,0.5\n2023-01-04,IF2403,-0.2\n",
                          encoding="utf-8")
    with execution_result(res_path, command="factor-synthesis", run_id="run",
                          execution_id="attempt", config_path=cfg_path) as res:
        res["execution"].update(expected=1, completed=1)
        res["metrics"]["synthesis_diagnostics"] = [dict(
            synthesis_name="weighted", factor_name="weighted", status="succeeded",
            evaluation_mode="composite_only", skip_reason="single_instrument",
            stock_count=1, instruments=["IF2403"], finite_score_count=2,
        )]
        res["artifacts"].append({
            "artifact_type": "composite_scores",
            "synthesis_name": "weighted",
            "factor_name": "weighted",
            "path": str(scores_csv),
            "sha256": config_sha256(scores_csv),
        })
    metrics = runner._bound_result_metrics(
        res_path, action="factor-synthesis", run_id="run", execution_id="attempt",
        source_sha256=config_sha256(cfg_path), config_payload=cfg_payload,
    )
    assert metrics["execution"]["completed"] == 1

    # Tampered artifact with two instruments or mismatched score count fails even if hash matches.
    scores_csv.write_text("date,stock_code,composite_value\n2023-01-03,IF2403,0.5\n2023-01-04,IC2403,-0.2\n",
                          encoding="utf-8")
    with execution_result(res_path, command="factor-synthesis", run_id="run",
                          execution_id="attempt", config_path=cfg_path) as res:
        res["execution"].update(expected=1, completed=1)
        res["metrics"]["synthesis_diagnostics"] = [dict(
            synthesis_name="weighted", factor_name="weighted", status="succeeded",
            evaluation_mode="composite_only", skip_reason="single_instrument",
            stock_count=1, instruments=["IF2403"], finite_score_count=2,
        )]
        res["artifacts"].append({
            "artifact_type": "composite_scores",
            "synthesis_name": "weighted",
            "factor_name": "weighted",
            "path": str(scores_csv),
            "sha256": config_sha256(scores_csv),
        })
    with pytest.raises(ValueError, match="invalid synthesis item"):
        runner._bound_result_metrics(
            res_path, action="factor-synthesis", run_id="run", execution_id="attempt",
            source_sha256=config_sha256(cfg_path), config_payload=cfg_payload,
        )

    # Producer single-instrument path emits composite_scores artifact verified by the gate.
    from unittest.mock import Mock
    import pandas as pd
    from run_factor_synthesis import FactorSynthesizer

    synth = object.__new__(FactorSynthesizer)
    synth.config = {
        "synthesis_methods": {"weighted": {"method": "weighted_sum"}},
        "performance_evaluation": {"skip_alphalens_analysis": False},
    }
    synth.logger = Mock()
    prod_res_path = tmp_path / "prod_attempt" / "result.json"
    synth.execution_dir = prod_res_path.parent
    synth.frequency = "1d"
    synth.result_writer = object()
    synth._execution_metrics = {"synthesis_diagnostics": []}
    synth._execution_artifacts = []
    synth._resolve_evaluation_execution_lag = lambda lag: 1
    dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
    synth._get_pricing_data = lambda *a, **kw: pd.Series(
        [100.0, 101.0],
        index=pd.MultiIndex.from_tuples([("IF2403", d) for d in dates], names=["asset", "date"]),
    )
    series = pd.Series(
        [0.4, -0.1],
        index=pd.MultiIndex.from_tuples([(d, "IF2403") for d in dates], names=["date", "stock_code"]),
    )
    synth.direct_analyze_composite_factor(series, "weighted", synthesis_name="weighted")
    with execution_result(prod_res_path, command="factor-synthesis", run_id="run",
                          execution_id="prod", config_path=cfg_path) as res:
        res["execution"].update(expected=1, completed=1)
        res["metrics"]["synthesis_diagnostics"] = synth._execution_metrics["synthesis_diagnostics"]
        res["artifacts"].extend(synth._execution_artifacts)
    prod_metrics = runner._bound_result_metrics(
        prod_res_path, action="factor-synthesis", run_id="run", execution_id="prod",
        source_sha256=config_sha256(cfg_path), config_payload=synth.config,
    )
    assert prod_metrics["execution"]["completed"] == 1


def test_synthesis_handler_forwards_slot_and_frequency_cli_overrides(tmp_path, monkeypatch):
    from slots.jq_ashare_local import daily_slot_config
    from tools.agent_harness import cli as harness_cli
    from tools.agent_harness.run_result import execution_result

    cfg_path = tmp_path / "synth.json"
    cfg_path.write_text(json.dumps({
        "synthesis_methods": {"demo": {"method": "weighted_sum"}},
        "database": {"local_db": {"duckdb_path": str(tmp_path / "local.duckdb")}},
    }), encoding="utf-8")
    slot_payload = daily_slot_config().as_dict()
    slot_payload["bar_feed"]["options"]["frequency"] = "5min"
    slot_file = tmp_path / "slot.json"
    slot_file.write_text(json.dumps(slot_payload), encoding="utf-8")
    monkeypatch.setattr(harness_cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(harness_cli, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")

    def fake_run_subprocess(**kwargs):
        result_path = kwargs["command"][kwargs["command"].index("--result-out") + 1]
        execution_id = kwargs["command"][kwargs["command"].index("--execution-id") + 1]
        with execution_result(result_path, command="factor-synthesis", run_id="run",
                              execution_id=execution_id, config_path=cfg_path) as res:
            res["execution"].update(expected=1, completed=1)
            res["metrics"]["synthesis_diagnostics"] = [dict(
                synthesis_name="demo", factor_name="demo", status="succeeded",
                evaluation_mode="composite_only", skip_reason="non_daily_frequency",
                frequency="5min", finite_score_count=10,
            )]
        metrics = kwargs["success_metrics_builder"]("", "")
        return harness_cli.HarnessResult(
            action="factor-synthesis", ok=True, exit_code=0, status="succeeded", reason="ok",
            mode="subprocess", output_dir=kwargs["output_dir"], command=kwargs["command"],
            metrics=metrics, artifacts=[], warnings=[], run_id="run",
        )

    monkeypatch.setattr(harness_cli, "_run_subprocess", fake_run_subprocess)
    args = harness_cli.build_parser().parse_args([
        "factor-synthesis", "--config", str(cfg_path), "--slot-config", str(slot_file), "--run-id", "run",
    ])
    assert harness_cli._handle_factor_synthesis(args).ok

    args_daily = harness_cli.build_parser().parse_args([
        "factor-synthesis", "--config", str(cfg_path), "--slot-config", str(slot_file),
        "--frequency", "1d", "--run-id", "run",
    ])
    with pytest.raises(ValueError, match="invalid synthesis item"):
        harness_cli._handle_factor_synthesis(args_daily)


def test_synthesis_reuse_verifies_bundle_cache_structure_calendar_and_dynamic_pool_counts(tmp_path, monkeypatch):
    import hashlib
    from unittest.mock import Mock
    import numpy as np
    import pandas as pd
    import factor_synthesis.core.pool_manager as synth_pool_manager
    from run_factor_synthesis import FactorSynthesizer, _build_lasso_bundle_model

    store = tmp_path / "reuse_store"
    store.mkdir()
    bundle_path = store / "lasso_model_bundle_lasso_a.json"
    cache_path = store / "composite_cache_lasso_a.parquet"
    cfg = {
        "synthesis_methods": {"lasso_a": {"method": "lasso"}},
        "synthesis_reuse": {"enabled": True, "source_dir": str(store)},
        "database": {"local_db": {"duckdb_path": str(tmp_path / "local.duckdb")}},
    }

    def write_bundle(payload: dict) -> str:
        bundle_path.write_text(json.dumps(payload), encoding="utf-8")
        return hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    def write_cache(df: pd.DataFrame) -> str:
        df.to_parquet(cache_path)
        return hashlib.sha256(cache_path.read_bytes()).hexdigest()

    def diag(bundle_sha: str, cache_sha: str, *, finite: int, stocks: int = 2) -> dict:
        return {
            "synthesis_diagnostics": [dict(
                synthesis_name="lasso_a", factor_name="lasso_a", status="succeeded",
                evaluation_mode="composite_only", skip_reason="synthesis_reuse",
                finite_score_count=finite, stock_count=stocks,
                reuse_bundle_sha256=bundle_sha, reuse_cache_sha256=cache_sha,
            )]
        }

    def make_model(*, weights=None, blend_weights=None, blend_ratio=0.0, parent_baseline=None):
        return _build_lasso_bundle_model(
            train_end="2023-01-01",
            weights=weights if weights is not None else {"f1": 1.0},
            model_info=None,
            synthesis_config={
                "blend_with_original_ratio": blend_ratio,
                "blend_weights": blend_weights or {},
                "postprocess_cs_standardize": True,
            },
            parent_baseline=parent_baseline or {},
        )

    valid_bundle_sha = write_bundle({
        "index": {
            "synthesis_name": "lasso_a",
            "models": [make_model()],
        },
    })

    # 1. RangeIndex, missing named date/stock_code levels, non-numeric, or all-NaN caches fail even with valid hashes.
    for bad_df, claimed_finite in (
        (pd.DataFrame({"composite_value": [0.1, 0.2]}), 2),
        (
            pd.DataFrame(
                {"composite_value": [0.1, 0.2]},
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp("2023-01-03"), "A"), (pd.Timestamp("2023-01-04"), "B")],
                    names=["dt", "symbol"],
                ),
            ),
            2,
        ),
        (
            pd.DataFrame(
                {"composite_value": ["bad", "text"]},
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp("2023-01-03"), "A"), (pd.Timestamp("2023-01-04"), "B")],
                    names=["date", "stock_code"],
                ),
            ),
            2,
        ),
        (
            pd.DataFrame(
                {"composite_value": [np.nan, np.nan]},
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp("2023-01-03"), "A"), (pd.Timestamp("2023-01-04"), "B")],
                    names=["date", "stock_code"],
                ),
            ),
            2,
        ),
    ):
        bad_cache_sha = write_cache(bad_df)
        with pytest.raises(ValueError, match="invalid synthesis item"):
            bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, bad_cache_sha, finite=claimed_finite), config_payload=cfg)

    # 2. Valid MultiIndex(date, stock_code) cache with Jan (2 finite) + Feb (1 finite) scores + 1 NaN.
    valid_idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2023-01-03"), "000001.XSHE"),
            (pd.Timestamp("2023-01-03"), "000002.XSHE"),
            (pd.Timestamp("2023-02-01"), "000001.XSHE"),
            (pd.Timestamp("2023-02-01"), "000002.XSHE"),
        ],
        names=["date", "stock_code"],
    )
    valid_cache_sha = write_cache(pd.DataFrame({"composite_value": [0.5, -0.3, 0.8, np.nan]}, index=valid_idx))
    assert bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, valid_cache_sha, finite=3, stocks=2), config_payload=cfg)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, valid_cache_sha, finite=12, stocks=2), config_payload=cfg)

    # 3. Calendar skip filtering + synthesis stock_pool.use_dynamic_pools expansion (NOT analysis top-level dynamic_pools!).
    skip_cfg = {
        **cfg,
        "analysis_period": {"start_date": "2023-01-01", "end_date": "2023-02-28"},
        "calendar_filters": {"skip_months": [1], "apply_to_evaluation": True},
    }
    assert bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, valid_cache_sha, finite=1, stocks=2), config_payload=skip_cfg)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, valid_cache_sha, finite=3, stocks=2), config_payload=skip_cfg)

    # Disabled stock_pool (even if analysis-only top-level dynamic_pools exists!) cannot expand stock_count beyond cache.
    disabled_pool_cfg = {
        **skip_cfg,
        "stock_pool": {"use_dynamic_pools": False},
        "dynamic_pools": {"pool_a": {"enabled": True, "filters": []}},
    }
    assert bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, valid_cache_sha, finite=1, stocks=2), config_payload=disabled_pool_cfg)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, valid_cache_sha, finite=1, stocks=4), config_payload=disabled_pool_cfg)

    # Enabled stock_pool.use_dynamic_pools=True allows NaN universe expansion beyond cache stocks (4 >= 2) while rejecting < cache.
    dyn_cfg = {**skip_cfg, "stock_pool": {"use_dynamic_pools": True, "filters": []}}
    assert bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, valid_cache_sha, finite=1, stocks=4), config_payload=dyn_cfg)
    with pytest.raises(ValueError, match="invalid synthesis item"):
        bound(tmp_path, "factor-synthesis", diag(valid_bundle_sha, valid_cache_sha, finite=1, stocks=1), config_payload=dyn_cfg)

    # 4. Bundle model contract: enforce feature_names / finite numeric weights / blend_only rules matching StrategyBundle.
    pb_file = store / "parent_ref.parquet"
    pb_file.write_bytes(b"parent")
    pb_sha = hashlib.sha256(pb_file.read_bytes()).hexdigest()
    assert bound(
        tmp_path,
        "factor-synthesis",
        diag(
            write_bundle({"index": {"models": [make_model(weights={}, blend_weights={"B1": 1.0}, blend_ratio=0.5)]}}),
            valid_cache_sha, finite=3, stocks=2,
        ),
        config_payload=cfg,
    )
    assert bound(
        tmp_path,
        "factor-synthesis",
        diag(
            write_bundle({
                "index": {
                    "requires_parent_baseline": True,
                    "expected_model_count": 1,
                    "models": [make_model(weights={}, blend_ratio=1.0, parent_baseline={"path": "parent_ref.parquet", "sha256": pb_sha})],
                },
            }),
            valid_cache_sha, finite=3, stocks=2,
        ),
        config_payload=cfg,
    )
    for bad_bundle in (
        {"index": {"models": [{"train_end": "2023-01-01"}]}},
        {"index": {"models": [{"train_end": "2023-01-01", "weights": {"f1": 1.0}}]}},
        {"index": {"models": [{"train_end": "2023-01-01", "weights": {"f1": 1.0}, "feature_names": []}]}},
        {"index": {"models": [{"train_end": "2023-01-01", "weights": {"f1": "bad"}, "feature_names": ["f1"]}]}},
        {"index": {"models": [{"train_end": "2023-01-01", "weights": {}, "blend_only": False, "blend_weights": {"B1": 1.0}}]}},
        {"index": {"models": [{"train_end": "2023-01-01", "weights": {}, "blend_only": True, "blend_weights": {"B1": "bad"}}]}},
        {"index": {"models": [{"train_end": "2023-01-01", "weights": {}, "blend_only": True, "blend_weights": {}}]}},
        {"index": {"requires_parent_baseline": True, "expected_model_count": 2, "models": [make_model()]}},
        {"index": {"requires_parent_baseline": True, "expected_model_count": 1, "models": [make_model(parent_baseline={"path": "missing.parquet", "sha256": "a" * 64})]}},
    ):
        bad_bundle_sha = write_bundle(bad_bundle)
        with pytest.raises(ValueError, match="invalid synthesis item"):
            bound(tmp_path, "factor-synthesis", diag(bad_bundle_sha, valid_cache_sha, finite=3, stocks=2), config_payload=cfg)

    # 5. Real _maybe_reuse_synthesis -> real compute_dynamic_pools_for_topk -> real _record_bundle_synthesis -> gate.
    bundle_sha = write_bundle({
        "index": {
            "synthesis_name": "lasso_a",
            "factor_group": "g1",
            "expected_model_count": 1,
            "models": [make_model()],
        },
    })

    class _FakePoolMgr:
        def get_stock_pools_for_period(self, **_kw):
            return {"2023-02-01": ["000001.XSHE", "000002.XSHE", "000003.XSHE", "000004.XSHE"]}

    monkeypatch.setattr(synth_pool_manager, "build_stock_filters", lambda **_kw: [])
    monkeypatch.setattr(synth_pool_manager, "_create_pool_manager", lambda **_kw: _FakePoolMgr())

    synth = object.__new__(FactorSynthesizer)
    synth.config = {
        **dyn_cfg,
        "output_settings": {"weights_output_dir": str(store), "composite_factor_prefix": "LASSO"},
        "factor_groups": {"g1": {"factors": ["f1"]}},
        "performance_evaluation": {"skip_alphalens_analysis": False, "top_k_strategy": {"enabled": False}},
    }
    synth.logger = Mock()
    synth.frequency = "1d"
    synth.db_backend = "duckdb"
    synth.fundamental_table = "factor_data"
    synth.price_table = "price_data"
    synth.result_writer = object()
    synth.skip_months = {1}
    synth.skip_months_evaluation = True
    synth._get_price_conn = lambda: None
    synth._apply_skip_months = lambda obj: synth_pool_manager.apply_skip_months(obj, synth.skip_months)
    synth._resolve_evaluation_execution_lag = lambda lag: 1
    synth._resolve_reuse_market_factor_inputs = lambda **kw: {}
    synth._get_pricing_data = lambda stocks, dates, **kw: pd.Series(
        [10.0] * len(stocks),
        index=pd.MultiIndex.from_product([stocks, [pd.Timestamp("2023-02-01")]], names=["asset", "date"]),
    )
    synth._execution_metrics = {"synthesis_diagnostics": []}
    assert synth._maybe_reuse_synthesis("lasso_a", {"factor_group": "g1", "method": "lasso"}, "lasso") is True
    assert synth._last_bundle_synthesis["synthesis_name"] == "lasso_a"
    assert synth._execution_metrics["synthesis_diagnostics"][0]["reuse_bundle_sha256"] == bundle_sha
    assert synth._execution_metrics["synthesis_diagnostics"][0]["stock_count"] == 4
    assert synth._execution_metrics["synthesis_diagnostics"][0]["finite_score_count"] == 1
    assert bound(tmp_path, "factor-synthesis", synth._execution_metrics, config_payload=synth.config)

    # When stock_pool.use_dynamic_pools=False (even with top-level dynamic_pools!), real pool helper skips expansion -> stock_count=2.
    synth.config = {
        **disabled_pool_cfg,
        "output_settings": {"weights_output_dir": str(store), "composite_factor_prefix": "LASSO"},
        "factor_groups": {"g1": {"factors": ["f1"]}},
        "performance_evaluation": {"skip_alphalens_analysis": False, "top_k_strategy": {"enabled": False}},
    }
    synth._execution_metrics = {"synthesis_diagnostics": []}
    assert synth._maybe_reuse_synthesis("lasso_a", {"factor_group": "g1", "method": "lasso"}, "lasso") is True
    assert synth._execution_metrics["synthesis_diagnostics"][0]["stock_count"] == 2
    assert bound(tmp_path, "factor-synthesis", synth._execution_metrics, config_payload=synth.config)


def _valid_stat_summary(mode="evaluation"):
    summary = {"ic": {"1D": {"sample_count": 5, "mean": 0.04}}}
    if mode == "evaluation":
        summary["ir"] = {"ir": {"sample_count": 1, "mean": 1.2}}
    return summary


@pytest.mark.parametrize("mode,pools", [
    ("evaluation", [None]), ("evaluation", ["small", "broad"]),
    ("materialize_only", ["small"]), ("mining_only", ["small"]),
])
def test_analysis_binds_mode_factor_pool_coverage_and_count(tmp_path, mode, pools):
    source = str(tmp_path / "factors.py")
    config = {"factor_class": "Factor"}
    dynamic = pools[0] is not None
    if dynamic:
        config = {"dynamic_pools": {name: {} for name in pools}, "analysis_settings": config}
        config["dynamic_pools"]["disabled"] = {"enabled": False}
    context = dict(config_payload=config, factor=source, use_dynamic=dynamic,
                   materialize_only=mode == "materialize_only", mining_only=mode == "mining_only")
    if mode == "materialize_only":
        evidence_fields = {"materialized_rows": 5}
    else:
        summary = _valid_stat_summary(mode)
        total = sum(entry["sample_count"] for tbl in summary.values() for entry in tbl.values())
        evidence_fields = {"stat_summary": summary, "finite_stat_count": total}
    rows = [
        dict(factor_name="Factor", factor_source=source, pool=pool, mode=mode,
             status="succeeded", **evidence_fields)
        for pool in pools
    ]
    metrics = {"analysis_items": rows}
    assert bound(tmp_path, "single-factor-analysis", metrics, completed=len(rows), **context)
    with pytest.raises(ValueError, match="completed count mismatch"):
        bound(tmp_path, "single-factor-analysis", metrics, completed=len(rows) + 1, **context)
    if len(rows) > 1:
        with pytest.raises(ValueError, match="pool coverage"):
            bound(tmp_path, "single-factor-analysis", {"analysis_items": rows[:1]}, **context)
        assert bound(tmp_path, "single-factor-analysis", {"analysis_items": rows[:1]},
                     pools=pools[:1], **context)
    patches = [
        {"status": "done"}, {"status": "failed"}, {"factor_name": "Other"},
        {"factor_source": str(tmp_path / "different.py")},
        {"pool": "wrong"}, {"mode": "unsupported"},
    ]
    if mode == "materialize_only":
        patches.extend([{"materialized_rows": float("inf")}, {"materialized_rows": 0}])
    else:
        patches.extend([
            {"stat_summary": None},
            {"stat_summary": {}},
            {"stat_summary": {"ic": {}}},
            {"stat_summary": {"ic": {"1D": {"sample_count": 0, "mean": 0.1}}}},
            {"stat_summary": {"ic": {"1D": {"sample_count": 5, "mean": float("nan")}}}},
            {"stat_summary": {"ic": {"1D": {"sample_count": 5, "mean": None}}}},
            {"stat_summary": {"ic": {"1D": {"sample_count": 5, "mean": "0.1"}}}},
            {"finite_stat_count": 999},
        ])
        if mode == "mining_only":
            patches.append({"stat_summary": {"quantile_returns": {"1D": {"sample_count": 5, "mean": 0.1}}}})
    for patch in patches:
        bad = [dict(rows[0], **patch), *rows[1:]]
        with pytest.raises(ValueError, match="analysis"):
            bound(tmp_path, "single-factor-analysis", {"analysis_items": bad},
                  completed=len(rows), **context)


def test_analysis_requires_stat_summary_not_just_count_and_enforces_default_factor_dir(tmp_path, monkeypatch):
    from backtest_and_analysis.config import CONFIG
    from tools.agent_harness import cli as harness_cli
    from tools.agent_harness.run_result import execution_result

    default_dir = tmp_path / "default_factors"
    default_dir.mkdir()
    valid_factor = default_dir / "alpha.py"
    valid_factor.write_text("# factor\n", encoding="utf-8")
    outside_factor = tmp_path / "outside.py"
    outside_factor.write_text("# outside\n", encoding="utf-8")
    monkeypatch.setitem(CONFIG, "factor_dir", str(default_dir))

    count_only = dict(
        factor_name="Alpha", factor_source=str(valid_factor.resolve()), pool=None,
        mode="evaluation", status="succeeded", finite_stat_count=5,
    )
    with pytest.raises(ValueError, match="missing stat_summary"):
        bound(tmp_path, "single-factor-analysis", {"analysis_items": [count_only]},
              config_payload={}, factor=str(valid_factor))

    outside_item = dict(count_only, factor_source=str(outside_factor.resolve()),
                        stat_summary=_valid_stat_summary("evaluation"), finite_stat_count=6)
    with pytest.raises(ValueError, match="invalid analysis item"):
        bound(tmp_path, "single-factor-analysis", {"analysis_items": [outside_item]}, config_payload={})
    with pytest.raises(ValueError, match="invalid analysis item"):
        bound(tmp_path, "single-factor-analysis", {"analysis_items": [outside_item]},
              config_payload={"start_date": "2023-01-01"})
    with pytest.raises(ValueError, match="invalid analysis item"):
        bound(tmp_path, "single-factor-analysis", {"analysis_items": [outside_item]},
              config_payload={"database": {"duckdb_path": "in.duckdb"},
                              "analysis_settings": {"start_date": "2023-01-01"}})

    valid_item = dict(outside_item, factor_source=str(valid_factor.resolve()))
    assert bound(tmp_path, "single-factor-analysis", {"analysis_items": [valid_item]}, config_payload={})
    assert bound(tmp_path, "single-factor-analysis", {"analysis_items": [valid_item]},
                 config_payload={"start_date": "2023-01-01"})

    # Verify _handle_single_factor_analysis without --config uses default CONFIG factor_dir.
    default_cfg_file = tmp_path / "backtest_and_analysis" / "config.py"
    default_cfg_file.parent.mkdir()
    default_cfg_file.write_text("CONFIG = {}\n", encoding="utf-8")
    monkeypatch.setattr(harness_cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(harness_cli, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")

    def run_with_item(item):
        def fake_sub(**kwargs):
            rpath = kwargs["command"][kwargs["command"].index("--result-out") + 1]
            eid = kwargs["command"][kwargs["command"].index("--execution-id") + 1]
            with execution_result(rpath, command="single-factor-analysis", run_id="run",
                                  execution_id=eid, config_path=default_cfg_file) as res:
                res["execution"].update(expected=1, completed=1)
                res["metrics"]["analysis_items"] = [item]
            return kwargs["success_metrics_builder"]("", "")

        monkeypatch.setattr(harness_cli, "_run_subprocess", fake_sub)
        args = harness_cli.build_parser().parse_args(["single-factor-analysis", "--run-id", "run"])
        return harness_cli._handle_single_factor_analysis(args)

    assert run_with_item(valid_item)["execution"]["completed"] == 1
    with pytest.raises(ValueError, match="invalid analysis item"):
        run_with_item(outside_item)


def test_analysis_producer_records_finite_stat_summary_and_rejects_all_nan():
    import numpy as np
    import pandas as pd

    from run_analysis_main import _record_analysis_statistics

    evidence = {}
    _record_analysis_statistics(evidence, {
        "factor_name": "Alpha",
        "ic": pd.DataFrame({"1D": [0.1, np.nan, 0.3], "5D": [0.2, 0.4, np.inf]}),
        "ir": pd.DataFrame({"ir": [1.5]}),
    })
    assert evidence["finite_stat_count"] == 5
    assert evidence["stat_summary"]["ic"]["1D"] == {"sample_count": 2, "mean": pytest.approx(0.2)}
    assert evidence["stat_summary"]["ic"]["5D"] == {"sample_count": 2, "mean": pytest.approx(0.3)}
    assert evidence["stat_summary"]["ir"]["ir"] == {"sample_count": 1, "mean": pytest.approx(1.5)}

    with pytest.raises(RuntimeError, match="No finite analysis statistics"):
        _record_analysis_statistics({}, {
            "factor_name": "Alpha",
            "ic": pd.DataFrame({"1D": [np.nan, np.inf]}),
            "ir": pd.DataFrame(),
        })


def test_analysis_producer_and_handler_share_effective_config_semantics(tmp_path, monkeypatch):
    import run_analysis_main as analysis_producer
    from backtest_and_analysis.config import CONFIG
    from tools.agent_harness import cli as harness_cli
    from tools.agent_harness.run_result import execution_result

    default_dir = tmp_path / "default_factors"
    exp_dir = tmp_path / "factors_exp"
    default_dir.mkdir()
    exp_dir.mkdir()
    exp_factor = exp_dir / "exp_alpha.py"
    exp_factor.write_text("# exp\n", encoding="utf-8")
    default_factor = default_dir / "default_alpha.py"
    default_factor.write_text("# default\n", encoding="utf-8")
    monkeypatch.setitem(CONFIG, "factor_dir", str(default_dir))

    # Traditional config with analysis_period must stay traditional and honor explicit factor_dir.
    flat_cfg_path = tmp_path / "flat_with_period.json"
    flat_cfg_path.write_text(json.dumps({
        "factor_dir": str(exp_dir),
        "analysis_period": {"start_date": "2021-01-01", "end_date": "2022-01-01"},
        "duckdb_path": str(tmp_path / "in.duckdb"),
        "results_duckdb_path": str(tmp_path / "out.duckdb"),
    }), encoding="utf-8")
    # Nested database config without factor_dir inherits default CONFIG["factor_dir"].
    nested_cfg_path = tmp_path / "nested_partial.json"
    nested_cfg_path.write_text(json.dumps({
        "database": {"duckdb_path": str(tmp_path / "in.duckdb")},
        "analysis_period": {"start_date": "2021-01-01", "end_date": "2022-01-01"},
        "results_database": {"duckdb_path": str(tmp_path / "out.duckdb")},
    }), encoding="utf-8")

    monkeypatch.setattr(harness_cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(harness_cli, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")
    default_cfg_file = tmp_path / "backtest_and_analysis" / "config.py"
    default_cfg_file.parent.mkdir(exist_ok=True)
    default_cfg_file.write_text("CONFIG = {}\n", encoding="utf-8")

    def capture_producer_config(cfg_path):
        captured: dict[str, object] = {}

        def fake_main(cfg, *, execution=None, metrics=None):
            captured.update(cfg)
            execution.update(expected=1, completed=1, failed=0, errors=[])
            src = str(exp_factor.resolve()) if cfg["factor_dir"] == str(exp_dir) else str(default_factor.resolve())
            metrics["analysis_items"] = [dict(
                factor_name="Alpha", factor_source=src, pool=None, mode="evaluation",
                status="succeeded", stat_summary=_valid_stat_summary("evaluation"), finite_stat_count=6,
            )]

        monkeypatch.setattr(analysis_producer, "main", fake_main)
        result_out = tmp_path / f"producer_{Path(cfg_path).stem if cfg_path else 'default'}" / "result.json"
        argv = ["run_analysis_main.py", "--result-out", str(result_out), "--run-id", "run", "--execution-id", "attempt"]
        if cfg_path is not None:
            argv.extend(["--config", str(cfg_path)])
        monkeypatch.setattr(sys, "argv", argv)
        analysis_producer.cli_main()
        return captured, result_out

    for cfg_path, expected_dir, valid_src, bad_src in (
        (flat_cfg_path, str(exp_dir), str(exp_factor.resolve()), str(default_factor.resolve())),
        (nested_cfg_path, str(default_dir), str(default_factor.resolve()), str(exp_factor.resolve())),
        (None, str(default_dir), str(default_factor.resolve()), str(exp_factor.resolve())),
    ):
        producer_cfg, _ = capture_producer_config(cfg_path)
        assert producer_cfg["factor_dir"] == expected_dir

        def run_handler_with_source(source_path):
            def fake_sub(**kwargs):
                rpath = kwargs["command"][kwargs["command"].index("--result-out") + 1]
                eid = kwargs["command"][kwargs["command"].index("--execution-id") + 1]
                cpath = cfg_path or default_cfg_file
                with execution_result(rpath, command="single-factor-analysis", run_id="run",
                                      execution_id=eid, config_path=cpath) as res:
                    res["execution"].update(expected=1, completed=1)
                    res["metrics"]["analysis_items"] = [dict(
                        factor_name="Alpha", factor_source=source_path, pool=None, mode="evaluation",
                        status="succeeded", stat_summary=_valid_stat_summary("evaluation"), finite_stat_count=6,
                    )]
                return kwargs["success_metrics_builder"]("", "")

            monkeypatch.setattr(harness_cli, "_run_subprocess", fake_sub)
            argv = ["single-factor-analysis", "--run-id", "run"]
            if cfg_path is not None:
                argv.extend(["--config", str(cfg_path)])
            return harness_cli._handle_single_factor_analysis(harness_cli.build_parser().parse_args(argv))

        assert run_handler_with_source(valid_src)["execution"]["completed"] == 1
        with pytest.raises(ValueError, match="invalid analysis item"):
            run_handler_with_source(bad_src)


def test_analysis_dynamic_pool_producer_handler_gate_preserves_custom_constraints_and_cli_filtering(tmp_path, monkeypatch):
    import run_analysis_main as analysis_producer
    from backtest_and_analysis.config import CONFIG
    from tools.agent_harness import cli as harness_cli
    from tools.agent_harness.result_content import resolve_effective_analysis_config
    from tools.agent_harness.run_result import execution_result

    default_dir = tmp_path / "default_factors"
    exp_dir = tmp_path / "factors_exp"
    default_dir.mkdir()
    exp_dir.mkdir()
    default_factor = default_dir / "default_alpha.py"
    default_factor.write_text("# default\n", encoding="utf-8")
    exp_factor = exp_dir / "example_alpha.py"
    exp_factor.write_text("# example\n", encoding="utf-8")
    monkeypatch.setitem(CONFIG, "factor_dir", str(default_dir))
    monkeypatch.setattr(harness_cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(harness_cli, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")

    in_db = str(tmp_path / "in.duckdb")
    out_db = str(tmp_path / "out.duckdb")

    # Case 1: Reviewer scenario - dynamic pool + analysis_settings(factor_dir, factor_class, mining_only=True)
    raw_mining = {
        "dynamic_pools": {
            "pool_a": {"enabled": True, "filters": []},
            "pool_disabled": {"enabled": False, "filters": []},
        },
        "analysis_settings": {
            "factor_dir": str(exp_dir),
            "factor_file": str(exp_factor),
            "factor_class": "ExampleFactor",
            "mining_only": True,
        },
        "analysis_period": {"start_date": "2022-01-01", "end_date": "2023-12-31"},
        "database": {"duckdb_path": in_db},
        "results_database": {"duckdb_path": out_db},
    }
    mining_cfg_path = tmp_path / "dynamic_mining.json"
    mining_cfg_path.write_text(json.dumps(raw_mining), encoding="utf-8")

    # Verify in-memory resolver parity directly (with and without factor_file, matching Reviewer check).
    raw_without_file = json.loads(json.dumps(raw_mining))
    del raw_without_file["analysis_settings"]["factor_file"]
    for label, resolved in (
        ("producer", resolve_effective_analysis_config(raw_without_file, config_path=mining_cfg_path, filter_pools=True, parse_pool_filters=True)),
        ("handler", resolve_effective_analysis_config(raw_without_file, config_path=mining_cfg_path, filter_pools=True)),
    ):
        assert resolved["factor_dir"] == str(exp_dir), label
        assert resolved["factor_class"] == "ExampleFactor", label
        assert resolved["mining_only"] is True, label
        assert set(resolved["dynamic_pools"]) == {"pool_a"}, label

    captured: dict[str, object] = {}

    def fake_main(cfg, *, execution=None, metrics=None):
        captured.clear()
        captured.update(cfg)
        execution.update(expected=1, completed=1, failed=0, errors=[])
        if metrics is not None:
            metrics["analysis_items"] = [dict(
                factor_name="ExampleFactor", factor_source=str(exp_factor.resolve()),
                pool="pool_a", mode="mining_only", status="succeeded",
                stat_summary=_valid_stat_summary("mining_only"), finite_stat_count=5,
            )]

    monkeypatch.setattr(analysis_producer, "main", fake_main)
    prod_out = tmp_path / "prod_mining" / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "run_analysis_main.py", "--config", str(mining_cfg_path),
        "--result-out", str(prod_out), "--run-id", "run", "--execution-id", "attempt",
    ])
    analysis_producer.cli_main()
    assert captured["factor_dir"] == str(exp_dir)
    assert captured["factor_class"] == "ExampleFactor"
    assert captured["mining_only"] is True
    assert set(captured["dynamic_pools"]) == {"pool_a"}

    def run_dynamic_handler(cfg_file: Path, items: list[dict], *, extra_argv: list[str] = ()):
        def fake_sub(**kwargs):
            rpath = kwargs["command"][kwargs["command"].index("--result-out") + 1]
            eid = kwargs["command"][kwargs["command"].index("--execution-id") + 1]
            with execution_result(rpath, command="single-factor-analysis", run_id="run",
                                  execution_id=eid, config_path=cfg_file) as res:
                res["execution"].update(expected=len(items), completed=len(items))
                res["metrics"]["analysis_items"] = items
            return kwargs["success_metrics_builder"]("", "")

        monkeypatch.setattr(harness_cli, "_run_subprocess", fake_sub)
        args = harness_cli.build_parser().parse_args([
            "single-factor-analysis", "--config", str(cfg_file), "--run-id", "run", *extra_argv,
        ])
        return harness_cli._handle_single_factor_analysis(args)

    valid_mining_item = dict(
        factor_name="ExampleFactor", factor_source=str(exp_factor.resolve()), pool="pool_a",
        mode="mining_only", status="succeeded", stat_summary=_valid_stat_summary("mining_only"), finite_stat_count=5,
    )
    assert run_dynamic_handler(mining_cfg_path, [valid_mining_item])["execution"]["completed"] == 1

    # Gate still enforces custom factor_dir/factor_file, factor_class, mining_only, and enabled dynamic pool!
    for bad_patch in (
        {"factor_source": str(default_factor.resolve())},
        {"factor_name": "WrongFactor"},
        {"mode": "evaluation", "stat_summary": _valid_stat_summary("evaluation"), "finite_stat_count": 6},
        {"pool": "pool_disabled"},
    ):
        with pytest.raises(ValueError, match="invalid analysis item|pool coverage"):
            run_dynamic_handler(mining_cfg_path, [dict(valid_mining_item, **bad_patch)])

    # Case 2: Dynamic pool evaluation with custom factor_dir (no factor_file) + multiple pools + CLI --pool filtering.
    raw_multi = {
        "dynamic_pools": {
            "pool_a": {"enabled": True, "filters": []},
            "pool_b": {"enabled": True, "filters": []},
            "pool_disabled": {"enabled": False, "filters": []},
        },
        "analysis_settings": {"factor_dir": str(exp_dir)},
        "database": {"duckdb_path": in_db},
        "results_database": {"duckdb_path": out_db},
    }
    multi_cfg_path = tmp_path / "dynamic_multi.json"
    multi_cfg_path.write_text(json.dumps(raw_multi), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "run_analysis_main.py", "--config", str(multi_cfg_path), "--pools", "pool_a",
        "--mining-only", "--factor", str(exp_factor),
        "--result-out", str(tmp_path / "prod_filtered" / "result.json"), "--run-id", "run", "--execution-id", "attempt",
    ])
    analysis_producer.cli_main()
    assert set(captured["dynamic_pools"]) == {"pool_a"}
    assert captured["mining_only"] is True

    eval_pool_a = dict(
        factor_name="ExampleFactor", factor_source=str(exp_factor.resolve()), pool="pool_a",
        mode="evaluation", status="succeeded", stat_summary=_valid_stat_summary("evaluation"), finite_stat_count=6,
    )
    assert run_dynamic_handler(multi_cfg_path, [eval_pool_a], extra_argv=["--pool", "pool_a"])["execution"]["completed"] == 1
    assert run_dynamic_handler(
        multi_cfg_path, [valid_mining_item], extra_argv=["--pool", "pool_a", "--mining-only", "--factor", str(exp_factor)]
    )["execution"]["completed"] == 1
    with pytest.raises(ValueError, match="invalid analysis item"):
        run_dynamic_handler(
            multi_cfg_path,
            [dict(eval_pool_a, factor_source=str(default_factor.resolve()))],
            extra_argv=["--pool", "pool_a"],
        )
    with pytest.raises(ValueError, match="unavailable dynamic pools"):
        run_dynamic_handler(multi_cfg_path, [dict(eval_pool_a, pool="pool_disabled")], extra_argv=["--pool", "pool_disabled"])


def test_alignment_counts_distinct_passing_checks_and_binds_request(tmp_path):
    target, digest = "signal_export", "b" * 64
    checks = [{"name": "manifest", "passed": True}, {"name": "positions", "passed": True}]
    report = dict(target_id=target, bundle_hash=digest, passed=True, checks=checks)
    metrics = dict(target_id=target, bundle_hash=digest, alignment_report=report)
    context = {"config_payload": {"target": target, "bundle_hash": digest}}
    assert bound(tmp_path, "verify-target-alignment", metrics, completed=2, **context)
    with pytest.raises(ValueError, match="completed count mismatch"):
        bound(tmp_path, "verify-alignment", metrics, completed=3, **context)
    for patch in ({"passed": "true"}, {"checks": []}, {"checks": [checks[0], checks[0]]},
                  {"checks": [{"name": "manifest", "passed": False}]}, {"bundle_hash": "c" * 64}):
        with pytest.raises(ValueError, match="alignment"):
            bound(tmp_path, "verify-alignment", dict(metrics, alignment_report=dict(report, **patch)),
                  completed=2, **context)
    with pytest.raises(ValueError, match="target_id or bundle_hash"):
        bound(tmp_path, "verify-alignment", dict(metrics, bundle_hash="c" * 64), completed=2, **context)


def test_export_requires_compatible_target_and_a_valid_manifest(tmp_path):
    from targets.contracts import CompatibilityReport, DeployArtifactEntry, DeployArtifactManifest

    target, digest = "signal_export", "b" * 64
    compatibility = CompatibilityReport(target, True, digest, "slot").to_dict()
    manifest = DeployArtifactManifest(
        target_id=target, bundle_hash=digest, slot_id="slot", strategy_id="demo", frequency="1d",
        created_at_utc="2026-11-27T00:00:00Z",
        artifacts=(DeployArtifactEntry("positions.jsonl", "c" * 64, "positions", 12),),
    ).to_dict()
    metrics = dict(target_id=target, bundle_hash=digest, compatibility=compatibility, deploy_manifest=manifest)
    assert bound(tmp_path, "export-strategy", metrics)
    with pytest.raises(ValueError, match="completed count mismatch"):
        bound(tmp_path, "export-strategy", metrics, completed=2)
    for patch in ({"compatibility": dict(compatibility, compatible=False)},
                  {"compatibility": dict(compatibility, slot_id="wrong")},
                  {"deploy_manifest": dict(manifest, manifest_hash="d" * 64)}, {"deploy_manifest": {}}):
        with pytest.raises(ValueError, match="compatibility|manifest|Manifest"):
            bound(tmp_path, "export-strategy", dict(metrics, **patch))


def test_generic_bound_action_needs_structured_items_and_matching_completion(tmp_path):
    item = {"item_id": "item", "status": "succeeded", "metrics": {"summary": {"count": 0}}}
    assert bound(tmp_path, "other", {"items": [item]})
    with pytest.raises(ValueError, match="completed count mismatch"):
        bound(tmp_path, "other", {"items": [item]}, completed=2)
    for values in ({}, {"note": "ok"}, {"estimate": None}, {"estimate": float("inf")}, {"passed": True}):
        with pytest.raises(ValueError, match="invalid other result item"):
            bound(tmp_path, "other", {"items": [dict(item, metrics=values)]})


def test_monitor_and_bar_success_require_action_specific_counts(tmp_path):
    monitor = {"backend": "duckdb", "available": True, "table_count": 0, "factor_count": 0}
    assert bound(tmp_path, "monitor-refresh", monitor)
    with pytest.raises(ValueError, match="valid counts"):
        bound(tmp_path, "monitor-refresh", dict(monitor, factor_count=float("nan")))
    totals = dict(total_count=5, valid_count=0, excluded_count=5)
    bar = dict(
        result_schema_version="bar_evaluation.v2", analysis_mode="time_series",
        result_identity={"identity_hash": "b" * 64}, exclusion_summary=totals,
        ic_by_horizon=[dict(totals, horizon_bars=1, ic_status="insufficient_samples", spearman_ic=None)],
        quantile_stats=[],
    )
    assert bound(tmp_path, "time-series-analysis", bar)
    with pytest.raises(ValueError, match="no label counts"):
        bound(tmp_path, "time-series-analysis", dict(
            bar, exclusion_summary=dict(total_count=0, valid_count=0, excluded_count=0),
        ))
    with pytest.raises(ValueError, match="ic_by_horizon"):
        bound(tmp_path, "time-series-analysis", dict(bar, ic_by_horizon=[]))
    with pytest.raises(ValueError, match="finite estimate"):
        bound(tmp_path, "time-series-analysis", dict(bar, ic_by_horizon=[
            dict(bar["ic_by_horizon"][0], ic_status="ok", spearman_ic=float("nan")),
        ]))


def test_pca_producer_counts_components_and_reports_missing_output():
    from unittest.mock import Mock

    from run_factor_synthesis import FactorSynthesizer

    synth = object.__new__(FactorSynthesizer)
    synth.config = {"synthesis_methods": {"principal": {"method": "pca", "n_components": 2}}}
    synth.logger = Mock()
    synth._run_local_reranker_profile = Mock()
    synth._emit_strategy_bundle = Mock()
    execution, metrics = {}, {}
    first = synthesis_item("principal", "principal_PC1")
    second = synthesis_item("principal", "principal_PC2")

    def produce_one(*_):
        metrics["ic_summaries"].append(first)

    def produce_both(*_):
        metrics["ic_summaries"].extend([first, second])

    synth.rolling_window_synthesis = produce_one
    with pytest.raises(RuntimeError, match="PCA output count mismatch"):
        synth.run_synthesis(execution=execution, metrics=metrics)
    assert (execution["expected"], execution["completed"], execution["failed"]) == (2, 0, 2)
    synth.rolling_window_synthesis = produce_both
    synth.run_synthesis(execution=execution, metrics=metrics)
    assert (execution["expected"], execution["completed"], execution["failed"]) == (2, 2, 0)
