"""Shared StrategyBundle test fixture builders."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from factor_synthesis import (
    BundleArtifactRef,
    StrategyBundle,
    StrategyFactorSpec,
)
from factor_synthesis.strategy_bundle import sha256_file
from utils.slots import ExecutionProfile


def _make_sample_strategy_bundle(tmp_path: Path) -> tuple[StrategyBundle, dict[str, Path]]:
    model_file = tmp_path / "src_weights.json"
    model_file.write_text(
        json.dumps(
            {
                "index": {
                    "models": [
                        {
                            "train_end": "1900-01-01",
                            "weights": {"alpha": 0.7, "beta": -0.3},
                            "feature_names": ["alpha", "beta"],
                        }
                    ]
                },
                "metadata": {"method": "weighted_sum"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    returns_file = tmp_path / "src_returns.csv"
    returns_file.write_text("date,net_return\n2023-01-04,0.01\n", encoding="utf-8")
    nav_file = tmp_path / "src_nav.csv"
    nav_file.write_text("date,nav\n2023-01-04,1.01\n", encoding="utf-8")
    exec_file = tmp_path / "src_exec.json"
    exec_file.write_text(
        json.dumps(
            {
                "schema_version": "execution_result.v1",
                "command": "factor-synthesis",
                "run_id": "run-1",
                "execution_id": "exec-1",
                "config_sha256": "c" * 64,
                "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
                "metrics": {},
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    bundle = StrategyBundle(
        strategy_id="demo_strategy",
        slot_id="jq_ashare_local",
        slot_config_hash="1" * 64,
        calendar_hash="2" * 64,
        slot_data_fingerprint={"snapshot": "v1"},
        frequency="1d",
        factors=(
            StrategyFactorSpec(
                factor_id="alpha",
                factor_identity_hash="3" * 64,
                frequency="1d",
                preprocessing="zscore",
                aux_features_required=("factor_data.natural_log_of_market_cap",),
                source_module="factors.alpha",
            ),
        ),
        synthesis_method="weighted_sum",
        model_artifacts=(
            BundleArtifactRef(
                path="models/weights.json",
                sha256=sha256_file(model_file),
                artifact_type="model_bundle",
                size_bytes=model_file.stat().st_size,
            ),
        ),
        preprocessing_params={"preprocess": {"alpha": "zscore"}},
        random_seed=42,
        rebalance_config={"rebalance_freq": "weekly", "top_k": 5},
        execution_profile=asdict(
            ExecutionProfile(
                "close", "close", 0, 0, "decimal", "endpoints_only", "cross_session",
                allow_short=False, commission_bps=3.0, sell_tax_bps=10.0, slippage_bps=0.0,
                benchmark_instrument="000300.XSHG",
            )
        ),
        universe_filter={"use_dynamic_pools": False},
        allow_short=False,
        benchmark_instrument="000300.XSHG",
        performance_summary={
            "annual_return": 0.15,
            "sharpe_ratio": 1.4,
            "max_drawdown": -0.08,
            "annual_turnover": 2.1,
        },
        backtest_returns=BundleArtifactRef(
            path="backtest/returns.csv",
            sha256=sha256_file(returns_file),
            artifact_type="backtest_returns",
            size_bytes=returns_file.stat().st_size,
        ),
        backtest_nav=BundleArtifactRef(
            path="backtest/nav.csv",
            sha256=sha256_file(nav_file),
            artifact_type="backtest_nav",
            size_bytes=nav_file.stat().st_size,
        ),
        execution_result_ref="provenance/execution_result.json",
        execution_result_sha256=sha256_file(exec_file),
    )
    sources = {
        "models/weights.json": model_file,
        "backtest/returns.csv": returns_file,
        "backtest/nav.csv": nav_file,
        "provenance/execution_result.json": exec_file,
    }
    return bundle, sources
