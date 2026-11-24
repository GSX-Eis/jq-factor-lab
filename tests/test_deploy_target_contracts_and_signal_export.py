"""Stage 6-1 tests: DeployTarget contracts, TargetRegistry, and SignalExportTarget."""
from __future__ import annotations

import json
import os
import re
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from factor_synthesis import (
    BundleArtifactRef,
    StrategyBundle,
    StrategyFactorSpec,
    verify_strategy_bundle,
    write_strategy_bundle,
)
from factor_synthesis.strategy_bundle import sha256_file
from targets import (
    DEPLOY_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    DEPLOY_MANIFEST_FILENAME,
    TARGET_ALIGNMENT_MISMATCH,
    TARGET_ARTIFACT_INTEGRITY_FAILED,
    TARGET_AUX_FEATURE_UNSUPPORTED,
    TARGET_BUNDLE_INVALID,
    TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
    TARGET_SIGNAL_SOURCE_MISSING,
    TARGET_SLOT_INCOMPATIBLE,
    AlignmentCheckItem,
    AlignmentReport,
    CompatibilityIssue,
    CompatibilityReport,
    DeployArtifactEntry,
    DeployArtifactManifest,
    DeployTarget,
    DeployTargetError,
    SignalExportTarget,
    StrategyBundleSource,
    TargetRegistry,
    build_default_target_registry,
    compute_deploy_manifest_hash,
    resolve_verified_bundle,
    verify_deploy_manifest,
    write_deploy_manifest,
)
from utils.slots import ExecutionProfile

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_bundle(
    tmp_path: Path,
    *,
    slot_id: str,
    frequency: str,
    allow_short: bool,
    benchmark_instrument: str | None,
    strategy_id: str = "demo_strategy",
    aux_features: tuple[str, ...] = (),
    rebalance_config: dict | None = None,
    execution_status: str = "succeeded",
    extra_model_artifacts: tuple[tuple[BundleArtifactRef, Path], ...] = (),
) -> tuple[StrategyBundle, dict[str, Path]]:
    src = tmp_path / f"src_{strategy_id}_{slot_id}_{execution_status}"
    src.mkdir(parents=True, exist_ok=True)
    model_file = src / "weights.json"
    model_file.write_text(
        json.dumps(
            {
                "index": {
                    "models": [
                        {
                            "train_end": "2023-02-24",
                            "weights": {"alpha": 0.6, "beta": 0.4},
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
    returns_file = src / "returns.csv"
    returns_file.write_text("date,net_return\n2023-03-01,0.01\n", encoding="utf-8")
    nav_file = src / "nav.csv"
    nav_file.write_text("date,nav\n2023-03-01,1.01\n", encoding="utf-8")
    exec_file = src / "execution_result.json"
    exec_file.write_text(
        json.dumps(
            {
                "schema_version": "execution_result.v1",
                "command": "factor-synthesis",
                "run_id": f"run-{strategy_id}",
                "execution_id": f"exec-{strategy_id}",
                "config_sha256": "c" * 64,
                "execution": {
                    "expected": 1,
                    "completed": 1 if execution_status == "succeeded" else 0,
                    "failed": 0 if execution_status == "succeeded" else 1,
                    "errors": [] if execution_status == "succeeded" else [{"item": "bt", "error": "boom"}],
                    "status": execution_status,
                },
                "metrics": {},
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    profile = ExecutionProfile(
        entry_price_field="close",
        exit_price_field="close",
        min_entry_offset_bars=0,
        exit_adjustment=0,
        return_unit="decimal",
        gap_rule="endpoints_only",
        boundary_policy="cross_session",
        allow_short=allow_short,
        commission_bps=2.5,
        sell_tax_bps=10.0 if slot_id == "jq_ashare_local" else 0.0,
        slippage_bps=1.0,
        benchmark_instrument=benchmark_instrument,
    )
    model_refs = [
        BundleArtifactRef(
            path="models/weights.json",
            sha256=sha256_file(model_file),
            artifact_type="model_bundle",
            size_bytes=model_file.stat().st_size,
        ),
        *(ref for ref, _ in extra_model_artifacts),
    ]
    bundle = StrategyBundle(
        strategy_id=strategy_id,
        slot_id=slot_id,
        slot_config_hash="1" * 64,
        calendar_hash="2" * 64,
        slot_data_fingerprint={"snapshot": slot_id},
        frequency=frequency,
        factors=(
            StrategyFactorSpec(
                factor_id="alpha",
                factor_identity_hash="3" * 64,
                frequency=frequency,
                preprocessing="zscore",
                aux_features_required=aux_features,
                source_module="factors.alpha",
            ),
        ),
        synthesis_method="weighted_sum",
        model_artifacts=tuple(model_refs),
        preprocessing_params={"preprocess": {"alpha": "zscore"}},
        random_seed=42,
        rebalance_config=(
            rebalance_config
            if rebalance_config is not None
            else {"rebalance_freq": "weekly", "selection_mode": "top_k", "top_k": 5}
        ),
        execution_profile=asdict(profile),
        universe_filter={"use_dynamic_pools": False},
        allow_short=allow_short,
        benchmark_instrument=benchmark_instrument,
        performance_summary={
            "annual_return": 0.14,
            "sharpe_ratio": 1.35,
            "max_drawdown": -0.07,
            "annual_turnover": 1.8,
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
        **{ref.path: src_path for ref, src_path in extra_model_artifacts},
    }
    return bundle, sources


def test_deploy_target_contracts_immutability_errors_and_registry(tmp_path: Path) -> None:
    expected_codes = {
        TARGET_BUNDLE_INVALID,
        TARGET_SLOT_INCOMPATIBLE,
        TARGET_AUX_FEATURE_UNSUPPORTED,
        TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
        TARGET_SIGNAL_SOURCE_MISSING,
        TARGET_ARTIFACT_INTEGRITY_FAILED,
        TARGET_ALIGNMENT_MISMATCH,
    }
    assert len(expected_codes) == 7

    err = DeployTargetError("bad bundle", code=TARGET_BUNDLE_INVALID)
    assert isinstance(err, ValueError)
    assert err.code == TARGET_BUNDLE_INVALID
    assert TARGET_BUNDLE_INVALID in str(err)

    issue = CompatibilityIssue(code=TARGET_SLOT_INCOMPATIBLE, message="unsupported slot", field="slot_id")
    compat = CompatibilityReport(
        target_id="signal_export",
        compatible=False,
        bundle_hash="a" * 64,
        slot_id="custom",
        issues=(issue,),
        warnings=("advisory",),
    )
    with pytest.raises((FrozenInstanceError, TypeError)):
        compat.compatible = True  # type: ignore[misc]
    with pytest.raises(DeployTargetError, match="compatible=True"):
        CompatibilityReport(
            target_id="signal_export",
            compatible=True,
            bundle_hash="a" * 64,
            slot_id="custom",
            issues=(issue,),
        )

    artifact_file = tmp_path / "positions.jsonl"
    artifact_file.write_text('{"target_weight":1.0}\n', encoding="utf-8")
    entry = DeployArtifactEntry(
        path="positions.jsonl",
        sha256=sha256_file(artifact_file),
        artifact_type="target_positions_jsonl",
        size_bytes=artifact_file.stat().st_size,
    )
    manifest = DeployArtifactManifest(
        target_id="signal_export",
        strategy_id="strat_1",
        bundle_hash="b" * 64,
        slot_id="declarative_custom",
        frequency="1h",
        created_at_utc="2026-11-23T00:00:00Z",
        artifacts=(entry,),
        metadata={"nested": {"count": 1}},
    )
    assert manifest.schema_version == DEPLOY_ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert len(manifest.manifest_hash) == 64
    with pytest.raises((FrozenInstanceError, TypeError)):
        manifest.target_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.metadata["nested"] = {}  # type: ignore[index]
    with pytest.raises(DeployTargetError, match="non-finite"):
        DeployArtifactManifest(
            target_id="signal_export",
            strategy_id="strat_1",
            bundle_hash="b" * 64,
            slot_id="declarative_custom",
            frequency="1h",
            created_at_utc="2026-11-23T00:00:00Z",
            artifacts=(entry,),
            metadata={"bad": float("nan")},
        )

    manifest_path = write_deploy_manifest(manifest, tmp_path)
    assert manifest_path == (tmp_path / DEPLOY_MANIFEST_FILENAME).resolve()
    verified_manifest = verify_deploy_manifest(tmp_path)
    assert verified_manifest.manifest_hash == manifest.manifest_hash

    check = AlignmentCheckItem(name="parity", passed=True, reason="ok")
    report = AlignmentReport(
        target_id="signal_export",
        passed=True,
        bundle_hash="b" * 64,
        checks=(check,),
        metrics={"max_diff": 0.0},
        max_abs_diff=0.0,
    )
    with pytest.raises((FrozenInstanceError, TypeError)):
        report.passed = False  # type: ignore[misc]
    with pytest.raises(DeployTargetError, match="max_abs_diff"):
        AlignmentReport(
            target_id="signal_export",
            passed=True,
            bundle_hash="b" * 64,
            max_abs_diff=float("inf"),
        )

    target = SignalExportTarget()
    assert isinstance(target, DeployTarget)

    registry = TargetRegistry()
    assert registry.list_targets() == ()
    registry.register(target)
    assert registry.list_targets() == ("signal_export",)
    assert registry.get("signal_export") is target
    with pytest.raises(ValueError, match="already registered"):
        registry.register(target)
    with pytest.raises(KeyError, match="Unknown DeployTarget"):
        registry.get("missing_target")
    with pytest.raises(TypeError, match="DeployTarget"):
        registry.register(SimpleNamespace(target_id="bad"))  # type: ignore[arg-type]

    default_reg = build_default_target_registry()
    assert default_reg.list_targets() == ("jq_cloud", "signal_export")
    assert isinstance(default_reg.get("signal_export"), SignalExportTarget)


def test_signal_export_end_to_end_across_three_market_slots(tmp_path: Path) -> None:
    target = build_default_target_registry().get("signal_export")

    # 1. jq_ashare_local (allow_short=False, long-only + optional fields)
    ashare_bundle, ashare_sources = _make_bundle(
        tmp_path,
        slot_id="jq_ashare_local",
        frequency="1d",
        allow_short=False,
        benchmark_instrument="000300.XSHG",
        strategy_id="ashare_ls_off",
        aux_features=("factor_data.natural_log_of_market_cap",),
    )
    ashare_bundle_dir = tmp_path / "ashare_bundle"
    write_strategy_bundle(ashare_bundle, ashare_bundle_dir, source_files=ashare_sources)
    assert target.check_compatibility(ashare_bundle).compatible is True

    ashare_positions = pd.DataFrame(
        [
            {
                "timestamp_utc": "2023-03-02T07:00:00Z",
                "session_id": "2023-03-02",
                "instrument_id": "600519.XSHG",
                "target_weight": 0.4,
                "signal_score": 0.85,
                "reference_price": 1750.0,
                "lot_size": 100,
                "contract_multiplier": 1.0,
            },
            {
                "timestamp_utc": "2023-03-01T07:00:00Z",
                "session_id": "2023-03-01",
                "instrument_id": "000001.XSHE",
                "target_weight": 0.6,
                "signal_score": 1.25,
                "reference_price": 13.5,
                "lot_size": 100,
                "contract_multiplier": 1.0,
            },
            {
                "timestamp_utc": "2023-03-01T07:00:00Z",
                "session_id": "2023-03-01",
                "instrument_id": "600519.XSHG",
                "target_weight": 0.4,
                "signal_score": 0.90,
                "reference_price": 1740.0,
                "lot_size": 100,
                "contract_multiplier": 1.0,
            },
            {
                "timestamp_utc": "2023-03-02T07:00:00Z",
                "session_id": "2023-03-02",
                "instrument_id": "000001.XSHE",
                "target_weight": 0.0,
                "signal_score": -0.10,
                "reference_price": 13.2,
                "lot_size": 100,
                "contract_multiplier": 1.0,
            },
        ]
    )
    ashare_out = tmp_path / "out_ashare"
    ashare_manifest = target.export(
        ashare_bundle,
        ashare_out,
        target_positions=ashare_positions,
        bundle_dir=ashare_bundle_dir,
    )
    assert verify_deploy_manifest(ashare_out).manifest_hash == ashare_manifest.manifest_hash
    ashare_report = target.verify_alignment(
        ashare_bundle,
        ashare_out,
        reference_positions=ashare_positions,
        bundle_dir=ashare_bundle_dir,
    )
    assert ashare_report.passed is True
    assert ashare_report.max_abs_diff == 0.0
    assert ashare_report.error_code == ""

    ashare_orders = [
        json.loads(line)
        for line in (ashare_out / "trade_orders.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(o["session_id"], o["instrument_id"], o["action"], o["delta_weight"]) for o in ashare_orders] == [
        ("2023-03-01", "000001.XSHE", "BUY", 0.6),
        ("2023-03-01", "600519.XSHG", "BUY", 0.4),
        ("2023-03-02", "000001.XSHE", "SELL", -0.6),
        ("2023-03-02", "600519.XSHG", "HOLD", 0.0),
    ]

    # 2. cn_futures (allow_short=True + futures contract codes + contract_multiplier)
    futures_bundle, futures_sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="cn_fut_ls",
    )
    futures_bundle_dir = tmp_path / "futures_bundle"
    write_strategy_bundle(futures_bundle, futures_bundle_dir, source_files=futures_sources)
    futures_source = StrategyBundleSource.from_dir(futures_bundle_dir)
    assert target.check_compatibility(futures_source).compatible is True
    futures_positions = [
        {
            "timestamp_utc": pd.Timestamp("2024-03-01T07:00:00Z"),
            "session_id": "2024-03-01",
            "instrument_id": "RB2405.XSGE",
            "target_weight": 0.5,
            "contract_multiplier": 10.0,
        },
        {
            "timestamp_utc": pd.Timestamp("2024-03-01T07:00:00Z"),
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": -0.5,
            "contract_multiplier": 300.0,
        },
        {
            "timestamp_utc": pd.Timestamp("2024-03-04T07:00:00Z"),
            "session_id": "2024-03-04",
            "instrument_id": "IF2406.CCFX",
            "target_weight": -0.2,
            "contract_multiplier": 300.0,
        },
        {
            "timestamp_utc": pd.Timestamp("2024-03-04T07:00:00Z"),
            "session_id": "2024-03-04",
            "instrument_id": "RB2405.XSGE",
            "target_weight": 0.0,
            "contract_multiplier": 10.0,
        },
    ]
    futures_out = tmp_path / "out_futures"
    futures_manifest = target.export(
        futures_source,
        futures_out,
        target_positions=futures_positions,
    )
    assert target.verify_alignment(futures_source, futures_manifest).passed is True
    futures_jsonl = [
        json.loads(line)
        for line in (futures_out / "target_positions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(r["instrument_id"], r["side"]) for r in futures_jsonl] == [
        ("IF2406.CCFX", "SHORT"),
        ("RB2405.XSGE", "LONG"),
        ("IF2406.CCFX", "SHORT"),
        ("RB2405.XSGE", "FLAT"),
    ]

    # 3. declarative_custom (Crypto 24/7 long/short, 1h frequency, current_positions -> SHORT/COVER)
    crypto_bundle, crypto_sources = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="1h",
        allow_short=True,
        benchmark_instrument=None,
        strategy_id="crypto_24x7",
        rebalance_config={"rebalance_freq": "1h", "selection_mode": "top_pct", "top_pct": 0.25, "top_k": None},
    )
    crypto_bundle_dir = tmp_path / "crypto_bundle"
    write_strategy_bundle(crypto_bundle, crypto_bundle_dir, source_files=crypto_sources)
    assert target.check_compatibility(crypto_bundle).compatible is True
    crypto_positions = pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp("2026-01-01T09:00:00+08:00"),
                "session_id": "2026-01-01",
                "instrument_id": "BTC-USDT",
                "target_weight": -0.2,
            },
            {
                "timestamp_utc": pd.Timestamp("2026-01-01T09:00:00+08:00"),
                "session_id": "2026-01-01",
                "instrument_id": "ETH-USDT",
                "target_weight": -0.4,
            },
            {
                "timestamp_utc": pd.Timestamp("2026-01-01T02:00:00Z"),
                "session_id": "2026-01-01",
                "instrument_id": "BTC-USDT",
                "target_weight": 0.3,
            },
        ]
    )
    crypto_out = tmp_path / "out_crypto"
    crypto_manifest = target.export(
        crypto_bundle_dir,
        crypto_out,
        target_positions=crypto_positions,
        current_positions={"BTC-USDT": -0.5, "SOL-USDT": 0.2},
    )
    crypto_report = target.verify_alignment(
        crypto_bundle_dir,
        crypto_out / DEPLOY_MANIFEST_FILENAME,
        reference_positions=crypto_positions,
    )
    assert crypto_report.passed is True
    assert crypto_report.max_abs_diff == 0.0

    crypto_orders = [
        json.loads(line)
        for line in (crypto_out / "trade_orders.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [
        (o["timestamp_utc"], o["instrument_id"], o["prev_weight"], o["target_weight"], o["action"])
        for o in crypto_orders
    ] == [
        ("2026-01-01T01:00:00Z", "BTC-USDT", -0.5, -0.2, "COVER"),
        ("2026-01-01T01:00:00Z", "ETH-USDT", 0.0, -0.4, "SHORT"),
        ("2026-01-01T01:00:00Z", "SOL-USDT", 0.2, 0.0, "SELL"),
        ("2026-01-01T02:00:00Z", "BTC-USDT", -0.2, 0.3, "BUY"),
        ("2026-01-01T02:00:00Z", "ETH-USDT", -0.4, 0.0, "COVER"),
    ]
    for order in crypto_orders:
        assert order["delta_weight"] == order["target_weight"] - order["prev_weight"]
    assert crypto_manifest.metadata["initial_positions"] == {"BTC-USDT": -0.5, "SOL-USDT": 0.2}


def test_signal_export_reference_alignment_tolerance_and_single_format(tmp_path: Path) -> None:
    target = SignalExportTarget()
    bundle, sources = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="15m",
        allow_short=True,
        benchmark_instrument=None,
        strategy_id="tolerance_test",
    )
    bundle_dir = tmp_path / "tolerance_bundle"
    write_strategy_bundle(bundle, bundle_dir, source_files=sources)
    positions = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-02-01T00:15:00Z",
                "session_id": "2026-02-01",
                "instrument_id": "BTC-USDT",
                "target_weight": 0.5,
            }
        ]
    )
    out_dir = tmp_path / "jsonl_only"
    manifest = target.export(
        bundle,
        out_dir,
        target_positions=positions,
        bundle_dir=bundle_dir,
        formats=("jsonl",),
    )
    assert [entry.path for entry in manifest.artifacts] == [
        "target_positions.jsonl",
        "trade_orders.jsonl",
    ]
    assert not (out_dir / "target_positions.parquet").exists()

    within_tol = positions.assign(target_weight=0.5 + 5e-13)
    ok_report = target.verify_alignment(
        bundle, out_dir, reference_positions=within_tol, bundle_dir=bundle_dir
    )
    assert ok_report.passed is True
    assert ok_report.max_abs_diff == pytest.approx(5e-13, abs=1e-16)

    outside_tol = positions.assign(target_weight=0.5 + 2e-12)
    bad_report = target.verify_alignment(
        bundle, out_dir, reference_positions=outside_tol, bundle_dir=bundle_dir
    )
    assert bad_report.passed is False
    assert bad_report.error_code == TARGET_ALIGNMENT_MISMATCH
    assert bad_report.max_abs_diff == pytest.approx(2e-12, abs=1e-16)


def test_signal_export_fail_closed_boundaries_and_tampering(tmp_path: Path) -> None:
    target = SignalExportTarget()
    long_only_bundle, sources = _make_bundle(
        tmp_path,
        slot_id="jq_ashare_local",
        frequency="1d",
        allow_short=False,
        benchmark_instrument="000300.XSHG",
        strategy_id="fail_closed",
    )
    valid_bundle_dir = tmp_path / "valid_bundle"
    write_strategy_bundle(long_only_bundle, valid_bundle_dir, source_files=sources)
    valid_pos = pd.DataFrame(
        [
            {
                "timestamp_utc": "2023-03-01T07:00:00Z",
                "session_id": "2023-03-01",
                "instrument_id": "000001.XSHE",
                "target_weight": 0.5,
            }
        ]
    )

    # 1. check_compatibility rejects invalid rebalance_config & corrupted execution_profile
    bad_rebal = replace(long_only_bundle, rebalance_config={}, bundle_hash="")
    compat_bad_rebal = target.check_compatibility(bad_rebal)
    assert compat_bad_rebal.compatible is False
    assert compat_bad_rebal.issues[0].code == TARGET_BUNDLE_INVALID

    corrupted_bundle = replace(long_only_bundle)
    object.__setattr__(corrupted_bundle, "execution_profile", {"allow_short": False})
    compat_bad_exec = target.check_compatibility(corrupted_bundle)
    assert compat_bad_exec.compatible is False
    assert compat_bad_exec.issues[0].code == TARGET_BUNDLE_INVALID

    # 2. Missing or empty target_positions fails closed with TARGET_SIGNAL_SOURCE_MISSING
    with pytest.raises(DeployTargetError, match=TARGET_SIGNAL_SOURCE_MISSING):
        target.export(
            long_only_bundle,
            tmp_path / "missing_src",
            target_positions=None,
            bundle_dir=valid_bundle_dir,
        )
    with pytest.raises(DeployTargetError, match=TARGET_SIGNAL_SOURCE_MISSING):
        target.export(
            long_only_bundle,
            tmp_path / "empty_src",
            target_positions=valid_pos.iloc[0:0],
            bundle_dir=valid_bundle_dir,
        )

    # 3. Negative target_weight / negative current_positions rejected when allow_short=False
    with pytest.raises(DeployTargetError, match=TARGET_EXECUTION_CONTRACT_UNSUPPORTED):
        target.export(
            long_only_bundle,
            tmp_path / "neg_weight",
            target_positions=valid_pos.assign(target_weight=-0.2),
            bundle_dir=valid_bundle_dir,
        )
    with pytest.raises(DeployTargetError, match=TARGET_EXECUTION_CONTRACT_UNSUPPORTED):
        target.export(
            long_only_bundle,
            tmp_path / "neg_current",
            target_positions=valid_pos,
            current_positions={"000001.XSHE": -0.1},
            bundle_dir=valid_bundle_dir,
        )

    # 4. NaN, Inf, naive timestamps, and duplicate primary keys rejected
    for bad_df in (
        valid_pos.assign(target_weight=float("nan")),
        valid_pos.assign(target_weight=float("inf")),
        valid_pos.assign(timestamp_utc="2023-03-01 07:00:00"),
        pd.concat([valid_pos, valid_pos], ignore_index=True),
    ):
        with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
            target.export(
                long_only_bundle,
                tmp_path / "bad_rows",
                target_positions=bad_df,
                bundle_dir=valid_bundle_dir,
            )

    # 5. Tampered bundle_dir rejected by verify_strategy_bundle via resolve_verified_bundle
    tampered_bundle_dir = tmp_path / "tampered_bundle"
    write_strategy_bundle(long_only_bundle, tampered_bundle_dir, source_files=sources)
    (tampered_bundle_dir / "models/weights.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        resolve_verified_bundle(long_only_bundle, bundle_dir=tampered_bundle_dir)
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(
            long_only_bundle,
            tmp_path / "from_tampered_bundle",
            target_positions=valid_pos,
            bundle_dir=tampered_bundle_dir,
        )

    # 6. Artifact tampering, manifest tampering, path traversal, symlink escape, and bundle_hash mismatch
    export_dir = tmp_path / "tamper_export"
    manifest = target.export(
        long_only_bundle,
        export_dir,
        target_positions=valid_pos,
        bundle_dir=valid_bundle_dir,
    )

    # 6a. Other bundle_hash mismatch
    other_bundle, other_sources = _make_bundle(
        tmp_path,
        slot_id="jq_ashare_local",
        frequency="1d",
        allow_short=False,
        benchmark_instrument="000300.XSHG",
        strategy_id="other_strategy",
    )
    other_bundle_dir = tmp_path / "other_bundle"
    write_strategy_bundle(other_bundle, other_bundle_dir, source_files=other_sources)
    mismatch_report = target.verify_alignment(
        other_bundle, export_dir, bundle_dir=other_bundle_dir
    )
    assert mismatch_report.passed is False
    assert mismatch_report.error_code == TARGET_BUNDLE_INVALID

    # 6b. Tamper exported file content -> SHA-256 mismatch
    jsonl_path = export_dir / "target_positions.jsonl"
    orig_jsonl = jsonl_path.read_bytes()
    jsonl_path.write_text('{"tampered": 1}\n', encoding="utf-8")
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        verify_deploy_manifest(export_dir)
    tampered_file_report = target.verify_alignment(
        long_only_bundle, export_dir, bundle_dir=valid_bundle_dir
    )
    assert tampered_file_report.passed is False
    assert tampered_file_report.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED
    jsonl_path.write_bytes(orig_jsonl)

    # 6c. Tamper deploy_manifest.json -> manifest_hash mismatch
    manifest_path = export_dir / DEPLOY_MANIFEST_FILENAME
    orig_manifest_bytes = manifest_path.read_bytes()
    manifest_payload = json.loads(orig_manifest_bytes.decode("utf-8"))
    manifest_payload["strategy_id"] = "tampered_id"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        verify_deploy_manifest(export_dir)
    tampered_manifest_report = target.verify_alignment(
        long_only_bundle, export_dir, bundle_dir=valid_bundle_dir
    )
    assert tampered_manifest_report.passed is False
    assert tampered_manifest_report.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED
    manifest_path.write_bytes(orig_manifest_bytes)

    # 6d. Path traversal rejected at entry construction and in raw JSON
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        DeployArtifactEntry(
            path="../escape.jsonl",
            sha256="a" * 64,
            artifact_type="target_positions_jsonl",
            size_bytes=10,
        )

    # 6e. Symlink escape rejected fail-closed
    outside_file = tmp_path / "outside_secret.jsonl"
    outside_file.write_bytes(orig_jsonl)
    jsonl_path.unlink()
    jsonl_path.symlink_to(outside_file)
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        verify_deploy_manifest(export_dir)
    symlink_report = target.verify_alignment(
        long_only_bundle, export_dir, bundle_dir=valid_bundle_dir
    )
    assert symlink_report.passed is False
    assert symlink_report.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED


def test_in_memory_bundle_cannot_bypass_gate_and_failed_exec_or_missing_files_fail(
    tmp_path: Path,
) -> None:
    target = SignalExportTarget()
    positions = pd.DataFrame(
        [
            {
                "timestamp_utc": "2024-03-01T07:00:00Z",
                "session_id": "2024-03-01",
                "instrument_id": "RB2405.XSGE",
                "target_weight": 0.5,
            }
        ]
    )

    # Failed execution_result.json with valid SHA-256 and valid bundle_hash
    failed_bundle, failed_sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="failed_exec",
        execution_status="failed",
    )
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(failed_bundle, tmp_path / "out_mem_failed", target_positions=positions)

    # Missing referenced file on disk -> write_strategy_bundle / manual manifest fails gate
    missing_dir = tmp_path / "missing_file_bundle"
    valid_bundle, valid_sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="missing_ref",
    )
    write_strategy_bundle(valid_bundle, missing_dir, source_files=valid_sources)
    (missing_dir / "provenance/execution_result.json").unlink()
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(valid_bundle, tmp_path / "out_missing_dir", target_positions=positions, bundle_dir=missing_dir)

    # Also write failed execution_result bundle manually and verify directory gate rejects it
    failed_dir = tmp_path / "failed_exec_dir"
    failed_dir.mkdir(parents=True, exist_ok=True)
    for rel, src in failed_sources.items():
        dst = failed_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    (failed_dir / "manifest.json").write_text(json.dumps(failed_bundle.to_dict()), encoding="utf-8")
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(failed_bundle, tmp_path / "out_dir_failed", target_positions=positions, bundle_dir=failed_dir)

    # Raw in-memory bundle without bundle_dir is rejected by verify_alignment too
    good_dir = tmp_path / "good_gate_bundle"
    write_strategy_bundle(valid_bundle, good_dir, source_files=valid_sources)
    good_out = tmp_path / "good_out"
    target.export(valid_bundle, good_out, target_positions=positions, bundle_dir=good_dir)
    unverified_align = target.verify_alignment(valid_bundle, good_out)
    assert unverified_align.passed is False
    assert unverified_align.error_code == TARGET_BUNDLE_INVALID

    # Even a StrategyBundle returned by verify_strategy_bundle() cannot omit bundle_dir via implicit global state
    verified_obj = verify_strategy_bundle(good_dir)
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(verified_obj, tmp_path / "out_bare_verified", target_positions=positions)
    bare_verified_align = target.verify_alignment(verified_obj, good_out)
    assert bare_verified_align.passed is False
    assert bare_verified_align.error_code == TARGET_BUNDLE_INVALID

    # Explicit StrategyBundleSource carries directory without global state and re-runs S5 gate on every read
    explicit_src = StrategyBundleSource.from_dir(good_dir)
    out_from_source = tmp_path / "out_from_source"
    target.export(explicit_src, out_from_source, target_positions=positions)
    assert target.verify_alignment(explicit_src, out_from_source).passed is True

    # Tampering on disk after constructing StrategyBundleSource is still caught because read re-runs S5 gate
    tamper_after_src_dir = tmp_path / "tamper_after_src"
    write_strategy_bundle(valid_bundle, tamper_after_src_dir, source_files=valid_sources)
    stale_src = StrategyBundleSource.from_dir(tamper_after_src_dir)
    (tamper_after_src_dir / "models/weights.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(stale_src, tmp_path / "out_stale_src", target_positions=positions)
    stale_align = target.verify_alignment(stale_src, out_from_source)
    assert stale_align.passed is False
    assert stale_align.error_code == TARGET_BUNDLE_INVALID


def test_signal_export_from_verified_bundle_dir_bound_signal_and_failures(tmp_path: Path) -> None:
    target = SignalExportTarget()
    sig_src = tmp_path / "sig_src"
    sig_src.mkdir(parents=True, exist_ok=True)
    sig_df = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-03-02T07:00:00Z",
                "session_id": "2026-03-02",
                "instrument_id": "BTC-USDT",
                "target_weight": 0.4,
                "signal_score": 1.1,
            },
            {
                "timestamp_utc": "2026-03-03T07:00:00Z",
                "session_id": "2026-03-03",
                "instrument_id": "ETH-USDT",
                "target_weight": -0.3,
                "signal_score": -0.8,
            },
        ]
    )
    sig_parquet = sig_src / "target_positions.parquet"
    sig_df.to_parquet(sig_parquet, index=False)
    sig_ref = BundleArtifactRef(
        path="signals/target_positions.parquet",
        sha256=sha256_file(sig_parquet),
        artifact_type="target_positions_parquet",
        size_bytes=sig_parquet.stat().st_size,
    )
    bundle, sources = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="1d",
        allow_short=True,
        benchmark_instrument=None,
        strategy_id="dir_signal_parquet",
        extra_model_artifacts=((sig_ref, sig_parquet),),
    )
    bdir = tmp_path / "dir_signal_bundle"
    write_strategy_bundle(bundle, bdir, source_files=sources)

    # Export and verify alignment using ONLY verified bundle_dir (omitting target_positions & reference_positions)
    out_dir = tmp_path / "dir_signal_out"
    manifest = target.export(bdir, out_dir)
    assert manifest.metadata["signal_source"] == "bundle_artifact"
    assert manifest.metadata["signal_artifact_path"] == "signals/target_positions.parquet"
    report = target.verify_alignment(bdir, out_dir)
    assert report.passed is True
    assert report.max_abs_diff == 0.0

    # Also test JSONL bound signal artifact
    sig_jsonl = sig_src / "target_positions.jsonl"
    sig_jsonl.write_text(
        "\n".join(json.dumps(row) for row in sig_df.to_dict(orient="records")) + "\n",
        encoding="utf-8",
    )
    jsonl_ref = BundleArtifactRef(
        path="signals/target_positions.jsonl",
        sha256=sha256_file(sig_jsonl),
        artifact_type="target_positions_jsonl",
        size_bytes=sig_jsonl.stat().st_size,
    )
    j_bundle, j_sources = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="1d",
        allow_short=True,
        benchmark_instrument=None,
        strategy_id="dir_signal_jsonl",
        extra_model_artifacts=((jsonl_ref, sig_jsonl),),
    )
    j_bdir = tmp_path / "dir_signal_jsonl_bundle"
    write_strategy_bundle(j_bundle, j_bdir, source_files=j_sources)
    j_out = tmp_path / "dir_signal_jsonl_out"
    j_source = StrategyBundleSource.from_dir(j_bdir)
    target.export(j_source, j_out)
    assert target.verify_alignment(j_source, j_out).passed is True

    # Failure 1: Tampered bound signal file -> TARGET_BUNDLE_INVALID
    (bdir / "signals/target_positions.parquet").write_bytes(b"corrupted-not-parquet")
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(bdir, tmp_path / "tampered_dir_signal_out")

    # Failure 2: Corrupted signal content with matching SHA-256 -> TARGET_ARTIFACT_INTEGRITY_FAILED
    bad_sig = sig_src / "bad.parquet"
    bad_sig.write_bytes(b"not-a-valid-parquet-payload")
    bad_sig_ref = BundleArtifactRef(
        path="signals/target_positions.parquet",
        sha256=sha256_file(bad_sig),
        artifact_type="target_positions_parquet",
        size_bytes=bad_sig.stat().st_size,
    )
    bad_bundle, bad_sources = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="1d",
        allow_short=True,
        benchmark_instrument=None,
        strategy_id="bad_dir_signal_content",
        extra_model_artifacts=((bad_sig_ref, bad_sig),),
    )
    bad_bdir = tmp_path / "bad_sig_content_bundle"
    write_strategy_bundle(bad_bundle, bad_bdir, source_files=bad_sources)
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        target.export(bad_bdir, tmp_path / "bad_sig_content_out")

    # Failure 3: Ambiguous multiple signal artifacts -> TARGET_BUNDLE_INVALID
    amb_bundle, amb_sources = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="1d",
        allow_short=True,
        benchmark_instrument=None,
        strategy_id="ambiguous_dir_signals",
        extra_model_artifacts=((sig_ref, sig_parquet), (jsonl_ref, sig_jsonl)),
    )
    amb_bdir = tmp_path / "ambiguous_sig_bundle"
    write_strategy_bundle(amb_bundle, amb_bdir, source_files=amb_sources)
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(amb_bdir, tmp_path / "ambiguous_sig_out")

    # Failure 4: Verified bundle_dir with no bound signal artifact and no target_positions -> TARGET_SIGNAL_SOURCE_MISSING
    no_sig_bundle, no_sig_sources = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="1d",
        allow_short=True,
        benchmark_instrument=None,
        strategy_id="no_dir_signal",
    )
    no_sig_bdir = tmp_path / "no_sig_bundle"
    write_strategy_bundle(no_sig_bundle, no_sig_bdir, source_files=no_sig_sources)
    with pytest.raises(DeployTargetError, match=TARGET_SIGNAL_SOURCE_MISSING):
        target.export(no_sig_bdir, tmp_path / "no_sig_out")


def test_rebalance_config_preflight_validation_and_cross_market_support(tmp_path: Path) -> None:
    target = SignalExportTarget()
    base_bundle, _ = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="1d",
        allow_short=True,
        benchmark_instrument=None,
        strategy_id="rebal_checks",
    )

    # Combined invalid fields from review: invalid rebalance_freq, mismatched selection_mode/top_k/top_pct, nth_trading_day=0, skip_months=[13]
    invalid_cfg = {
        "rebalance_freq": "not_a_frequency",
        "rebalance_schedule": "weekly",
        "holding_period": 5,
        "nth_trading_day": 0,
        "selection_mode": "top_pct",
        "top_k": 5,
        "skip_months": [13],
    }
    bad_bundle = replace(base_bundle, rebalance_config=invalid_cfg, bundle_hash="")
    bad_report = target.check_compatibility(bad_bundle)
    assert bad_report.compatible is False
    bad_fields = {issue.field for issue in bad_report.issues}
    assert all(issue.code == TARGET_BUNDLE_INVALID for issue in bad_report.issues)
    assert {
        "rebalance_config.rebalance_freq",
        "rebalance_config.nth_trading_day",
        "rebalance_config.top_pct",
        "rebalance_config.top_k",
        "rebalance_config.skip_months",
    }.issubset(bad_fields)

    # Additional invalid edge cases: top_k mode with top_pct set, invalid selection_mode, holding_period=0, bad schedule
    for extra_bad in (
        {"rebalance_freq": "weekly", "selection_mode": "top_k", "top_k": 5, "top_pct": 0.2},
        {"rebalance_freq": "weekly", "selection_mode": "unknown_mode", "top_k": 5},
        {"rebalance_freq": "weekly", "rebalance_schedule": "bogus", "selection_mode": "top_k", "top_k": 5},
        {"rebalance_freq": "weekly", "holding_period": 0, "selection_mode": "top_k", "top_k": 5},
    ):
        rep = target.check_compatibility(replace(base_bundle, rebalance_config=extra_bad, bundle_hash=""))
        assert rep.compatible is False
        assert all(issue.code == TARGET_BUNDLE_INVALID for issue in rep.issues)

    # Valid S5 and cross-market configs must pass
    for valid_cfg in (
        {
            "rebalance_freq": "weekly",
            "rebalance_schedule": "weekly",
            "holding_period": 5,
            "nth_trading_day": 3,
            "selection_mode": "top_k",
            "top_k": 5,
            "top_pct": None,
            "skip_months": ["01-16:02-05", 12],
            "synthesis_name": "ws",
            "trial_id": "t1",
        },
        {
            "rebalance_freq": "1h",
            "holding_period": 1,
            "nth_trading_day": 1,
            "selection_mode": "top_pct",
            "top_pct": 0.25,
            "top_k": None,
            "skip_months": [],
        },
        {
            "rebalance_freq": "monthly",
            "rebalance_schedule": "quarterly",
            "top_k": 10,
        },
    ):
        ok_rep = target.check_compatibility(replace(base_bundle, rebalance_config=valid_cfg, bundle_hash=""))
        assert ok_rep.compatible is True, ok_rep.issues


def test_failed_reexport_preserves_prior_publication_and_uses_exclusive_staging(
    tmp_path: Path,
) -> None:
    target = SignalExportTarget()
    bundle, sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="reexport_safety",
    )
    bdir = tmp_path / "reexport_bundle"
    write_strategy_bundle(bundle, bdir, source_files=sources)

    out_dir = tmp_path / "reexport_out"
    pos_v1 = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.2,
        }
    ]
    pos_v2 = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.5,
        }
    ]

    manifest_v1 = target.export(bdir, out_dir, target_positions=pos_v1)
    assert target.verify_alignment(bdir, out_dir, reference_positions=pos_v1).passed is True
    v1_snapshot = {
        name: (out_dir / name).read_bytes()
        for name in (
            "target_positions.parquet",
            "target_positions.jsonl",
            "trade_orders.jsonl",
            DEPLOY_MANIFEST_FILENAME,
        )
    }

    # Plant fixed-name temp symlinks to verify exclusive temp staging does not follow/reuse them
    decoy_target = tmp_path / "decoy_secret.txt"
    decoy_target.write_text("do-not-overwrite", encoding="utf-8")
    (out_dir / f".{DEPLOY_MANIFEST_FILENAME}.tmp").symlink_to(decoy_target)
    (out_dir / ".target_positions.parquet.tmp").symlink_to(decoy_target)

    # Failed re-export with invalid created_at_utc must not corrupt v1 artifacts or manifest
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        target.export(
            bdir,
            out_dir,
            target_positions=pos_v2,
            created_at_utc="not-a-utc-timestamp",
        )

    for name, expected_bytes in v1_snapshot.items():
        assert (out_dir / name).read_bytes() == expected_bytes
    assert decoy_target.read_text(encoding="utf-8") == "do-not-overwrite"
    still_ok = target.verify_alignment(bdir, out_dir, reference_positions=pos_v1)
    assert still_ok.passed is True
    assert verify_deploy_manifest(out_dir).manifest_hash == manifest_v1.manifest_hash

    # Valid re-export updates cleanly
    manifest_v2 = target.export(bdir, out_dir, target_positions=pos_v2, created_at_utc="2026-11-23T01:00:00Z")
    assert manifest_v2.manifest_hash != manifest_v1.manifest_hash
    assert decoy_target.read_text(encoding="utf-8") == "do-not-overwrite"
    assert target.verify_alignment(bdir, out_dir, reference_positions=pos_v2).passed is True


def _resign_manifest_dir(
    out_dir: Path,
    *,
    bundle: StrategyBundle,
    artifacts: tuple[DeployArtifactEntry, ...],
    metadata: dict,
) -> DeployArtifactManifest:
    """Helper for tests constructing SHA-valid synthetic manifests to test verifier edge cases."""
    manifest = DeployArtifactManifest(
        target_id="signal_export",
        strategy_id=bundle.strategy_id,
        bundle_hash=bundle.bundle_hash,
        slot_id=bundle.slot_id,
        frequency=bundle.frequency,
        created_at_utc="2026-11-23T00:00:00Z",
        artifacts=artifacts,
        metadata=metadata,
    )
    write_deploy_manifest(manifest, out_dir)
    return manifest


def test_verifier_rejects_incomplete_contradictory_and_malformed_artifacts_without_raw_exceptions(
    tmp_path: Path,
) -> None:
    target = SignalExportTarget()
    bundle, sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="verifier_edge",
    )
    bdir = tmp_path / "verifier_bundle"
    write_strategy_bundle(bundle, bdir, source_files=sources)

    positions = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.4,
        },
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "RB2405.XSGE",
            "target_weight": 0.6,
        },
    ]

    # 1. Manifest declares formats=['parquet', 'jsonl'] but lists only parquet artifact
    incomplete_dir = tmp_path / "incomplete_formats"
    base_manifest = target.export(bdir, incomplete_dir, target_positions=positions)
    only_parquet_and_orders = tuple(
        entry for entry in base_manifest.artifacts if entry.artifact_type != "target_positions_jsonl"
    )
    _resign_manifest_dir(
        incomplete_dir,
        bundle=bundle,
        artifacts=only_parquet_and_orders,
        metadata=dict(base_manifest.metadata),
    )
    rep_incomplete = target.verify_alignment(bdir, incomplete_dir)
    assert rep_incomplete.passed is False
    assert rep_incomplete.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    # 2. Duplicate artifact_type in manifest
    dup_type_dir = tmp_path / "dup_type"
    target.export(bdir, dup_type_dir, target_positions=positions, formats=("parquet",))
    parquet_copy = dup_type_dir / "target_positions_copy.parquet"
    parquet_copy.write_bytes((dup_type_dir / "target_positions.parquet").read_bytes())
    dup_artifacts = (
        DeployArtifactEntry(
            path="target_positions.parquet",
            sha256=sha256_file(dup_type_dir / "target_positions.parquet"),
            artifact_type="target_positions_parquet",
            size_bytes=(dup_type_dir / "target_positions.parquet").stat().st_size,
        ),
        DeployArtifactEntry(
            path="target_positions_copy.parquet",
            sha256=sha256_file(parquet_copy),
            artifact_type="target_positions_parquet",
            size_bytes=parquet_copy.stat().st_size,
        ),
        DeployArtifactEntry(
            path="trade_orders.jsonl",
            sha256=sha256_file(dup_type_dir / "trade_orders.jsonl"),
            artifact_type="trade_orders_jsonl",
            size_bytes=(dup_type_dir / "trade_orders.jsonl").stat().st_size,
        ),
    )
    _resign_manifest_dir(
        dup_type_dir,
        bundle=bundle,
        artifacts=dup_artifacts,
        metadata={**dict(base_manifest.metadata), "formats": ["parquet"]},
    )
    rep_dup_type = target.verify_alignment(bdir, dup_type_dir)
    assert rep_dup_type.passed is False
    assert rep_dup_type.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    # 3. Single-format SHA-valid artifacts with conflicting session_id at the same timestamp_utc
    conflict_dir = tmp_path / "session_conflict"
    m_single = target.export(bdir, conflict_dir, target_positions=positions, formats=("jsonl",))
    jsonl_lines = [
        json.loads(line)
        for line in (conflict_dir / "target_positions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    jsonl_lines[1]["session_id"] = "2024-03-01-OTHER"
    (conflict_dir / "target_positions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in jsonl_lines) + "\n",
        encoding="utf-8",
    )
    _resign_manifest_dir(
        conflict_dir,
        bundle=bundle,
        artifacts=(
            DeployArtifactEntry(
                path="target_positions.jsonl",
                sha256=sha256_file(conflict_dir / "target_positions.jsonl"),
                artifact_type="target_positions_jsonl",
                size_bytes=(conflict_dir / "target_positions.jsonl").stat().st_size,
            ),
            DeployArtifactEntry(
                path="trade_orders.jsonl",
                sha256=sha256_file(conflict_dir / "trade_orders.jsonl"),
                artifact_type="trade_orders_jsonl",
                size_bytes=(conflict_dir / "trade_orders.jsonl").stat().st_size,
            ),
        ),
        metadata=dict(m_single.metadata),
    )
    rep_conflict = target.verify_alignment(bdir, conflict_dir)
    assert rep_conflict.passed is False
    assert rep_conflict.error_code == TARGET_ALIGNMENT_MISMATCH

    # 4. Null weight in initial_positions -> returns failed AlignmentReport (no TypeError)
    null_init_dir = tmp_path / "null_initial_positions"
    m_init = target.export(bdir, null_init_dir, target_positions=positions, formats=("jsonl",))
    _resign_manifest_dir(
        null_init_dir,
        bundle=bundle,
        artifacts=m_init.artifacts,
        metadata={**dict(m_init.metadata), "initial_positions": {"IF2406.CCFX": None}},
    )
    rep_null_init = target.verify_alignment(bdir, null_init_dir)
    assert rep_null_init.passed is False
    assert rep_null_init.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    _resign_manifest_dir(
        null_init_dir,
        bundle=bundle,
        artifacts=m_init.artifacts,
        metadata={**dict(m_init.metadata), "initial_positions": ["bad"]},
    )
    rep_list_init = target.verify_alignment(bdir, null_init_dir)
    assert rep_list_init.passed is False
    assert rep_list_init.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    # 5. Corrupt Parquet bytes with valid SHA-256 -> returns failed AlignmentReport (no ArrowInvalid/OSError)
    bad_pq_dir = tmp_path / "corrupt_parquet"
    m_pq = target.export(bdir, bad_pq_dir, target_positions=positions, formats=("parquet",))
    (bad_pq_dir / "target_positions.parquet").write_bytes(b"corrupt-parquet-bytes")
    _resign_manifest_dir(
        bad_pq_dir,
        bundle=bundle,
        artifacts=(
            DeployArtifactEntry(
                path="target_positions.parquet",
                sha256=sha256_file(bad_pq_dir / "target_positions.parquet"),
                artifact_type="target_positions_parquet",
                size_bytes=(bad_pq_dir / "target_positions.parquet").stat().st_size,
            ),
            DeployArtifactEntry(
                path="trade_orders.jsonl",
                sha256=sha256_file(bad_pq_dir / "trade_orders.jsonl"),
                artifact_type="trade_orders_jsonl",
                size_bytes=(bad_pq_dir / "trade_orders.jsonl").stat().st_size,
            ),
        ),
        metadata=dict(m_pq.metadata),
    )
    rep_bad_pq = target.verify_alignment(bdir, bad_pq_dir)
    assert rep_bad_pq.passed is False
    assert rep_bad_pq.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    # 6. Malformed order field types (target_weight=null) with valid SHA-256 -> returns failed AlignmentReport
    bad_order_dir = tmp_path / "bad_order_field"
    m_ord = target.export(bdir, bad_order_dir, target_positions=positions, formats=("jsonl",))
    ord_lines = [
        json.loads(line)
        for line in (bad_order_dir / "trade_orders.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ord_lines[0]["target_weight"] = None
    (bad_order_dir / "trade_orders.jsonl").write_text(
        "\n".join(json.dumps(r) for r in ord_lines) + "\n",
        encoding="utf-8",
    )
    _resign_manifest_dir(
        bad_order_dir,
        bundle=bundle,
        artifacts=(
            DeployArtifactEntry(
                path="target_positions.jsonl",
                sha256=sha256_file(bad_order_dir / "target_positions.jsonl"),
                artifact_type="target_positions_jsonl",
                size_bytes=(bad_order_dir / "target_positions.jsonl").stat().st_size,
            ),
            DeployArtifactEntry(
                path="trade_orders.jsonl",
                sha256=sha256_file(bad_order_dir / "trade_orders.jsonl"),
                artifact_type="trade_orders_jsonl",
                size_bytes=(bad_order_dir / "trade_orders.jsonl").stat().st_size,
            ),
        ),
        metadata=dict(m_ord.metadata),
    )
    rep_bad_ord = target.verify_alignment(bdir, bad_order_dir)
    assert rep_bad_ord.passed is False
    assert rep_bad_ord.error_code == TARGET_ALIGNMENT_MISMATCH


def test_review_regressions_concurrent_reexport_overlap_signal_selection_overflow_and_duplicate_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SignalExportTarget()
    bundle, sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="review_round2_regressions",
    )
    bdir = tmp_path / "reg_bundle"
    write_strategy_bundle(bundle, bdir, source_files=sources)

    # 1a. Concurrent / interleaved re-export is exclusively rejected before modifying published files
    out_dir = tmp_path / "concurrent_out"
    pos_01 = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.1,
        }
    ]
    pos_02 = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.2,
        }
    ]
    pos_05 = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.5,
        }
    ]
    target.export(bdir, out_dir, target_positions=pos_01)
    real_replace = os.replace
    interleaved_errors: list[str] = []
    triggered = {"done": False}

    def _interleaving_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        real_replace(src, dst)
        if not triggered["done"] and Path(dst).name == "target_positions.parquet":
            triggered["done"] = True
            try:
                target.export(bdir, out_dir, target_positions=pos_05)
            except DeployTargetError as exc:
                interleaved_errors.append(exc.code)

    with monkeypatch.context() as mctx:
        mctx.setattr("targets.signal_export.exporter.os.replace", _interleaving_replace)
        manifest_02 = target.export(bdir, out_dir, target_positions=pos_02)

    assert triggered["done"] is True
    assert interleaved_errors == [TARGET_ARTIFACT_INTEGRITY_FAILED]
    assert verify_deploy_manifest(out_dir).manifest_hash == manifest_02.manifest_hash
    assert target.verify_alignment(bdir, out_dir, reference_positions=pos_02).passed is True

    # 1b. Failure during final verify_deploy_manifest(root) rolls back completely (including removing newly added format files)
    single_out = tmp_path / "rollback_single_out"
    m_v1 = target.export(bdir, single_out, target_positions=pos_01, formats=("jsonl",))
    v1_jsonl_bytes = (single_out / "target_positions.jsonl").read_bytes()
    v1_orders_bytes = (single_out / "trade_orders.jsonl").read_bytes()
    v1_manifest_bytes = (single_out / DEPLOY_MANIFEST_FILENAME).read_bytes()

    real_verify_manifest = verify_deploy_manifest
    call_count = {"n": 0}

    def _fail_final_root_verify(artifact, *, output_dir=None):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        # exporter calls write_deploy_manifest (which calls verify_deploy_manifest on stage_dir) first,
        # then calls verify_deploy_manifest(root) directly inside publish rollback block.
        if Path(artifact).resolve() == single_out.resolve():
            raise DeployTargetError("injected final verification failure", code=TARGET_ARTIFACT_INTEGRITY_FAILED)
        return real_verify_manifest(artifact, output_dir=output_dir)

    with monkeypatch.context() as mctx:
        mctx.setattr("targets.signal_export.exporter.verify_deploy_manifest", _fail_final_root_verify)
        with pytest.raises(DeployTargetError, match="injected final verification failure"):
            target.export(bdir, single_out, target_positions=pos_05, formats=("parquet", "jsonl"))

    assert not (single_out / "target_positions.parquet").exists()
    assert (single_out / "target_positions.jsonl").read_bytes() == v1_jsonl_bytes
    assert (single_out / "trade_orders.jsonl").read_bytes() == v1_orders_bytes
    assert (single_out / DEPLOY_MANIFEST_FILENAME).read_bytes() == v1_manifest_bytes
    assert verify_deploy_manifest(single_out).manifest_hash == m_v1.manifest_hash
    assert target.verify_alignment(bdir, single_out, reference_positions=pos_01).passed is True

    # 2. Output directory overlapping input StrategyBundle manifest or referenced files is rejected without mutating bundle
    sig_root_file = tmp_path / "root_signal.jsonl"
    sig_root_file.write_text(
        json.dumps(
            {
                "timestamp_utc": "2024-03-01T07:00:00Z",
                "session_id": "2024-03-01",
                "instrument_id": "IF2406.CCFX",
                "target_weight": 0.3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    root_sig_ref = BundleArtifactRef(
        path="target_positions.jsonl",
        sha256=sha256_file(sig_root_file),
        artifact_type="target_positions_jsonl",
        size_bytes=sig_root_file.stat().st_size,
    )
    overlap_bundle, overlap_sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="overlap_root",
        extra_model_artifacts=((root_sig_ref, sig_root_file),),
    )
    overlap_bdir = tmp_path / "overlap_bundle"
    write_strategy_bundle(overlap_bundle, overlap_bdir, source_files=overlap_sources)
    orig_sig_sha = sha256_file(overlap_bdir / "target_positions.jsonl")

    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        target.export(overlap_bdir, overlap_bdir, target_positions=pos_05)
    assert sha256_file(overlap_bdir / "target_positions.jsonl") == orig_sig_sha
    assert verify_strategy_bundle(overlap_bdir).bundle_hash == overlap_bundle.bundle_hash

    # Also test subdirectory collision when output_dir == bundle_dir / "signals"
    sub_pq = tmp_path / "sub_sig.parquet"
    pd.DataFrame(pos_01).to_parquet(sub_pq, index=False)
    sub_pq_ref = BundleArtifactRef(
        path="signals/target_positions.parquet",
        sha256=sha256_file(sub_pq),
        artifact_type="target_positions_parquet",
        size_bytes=sub_pq.stat().st_size,
    )
    sub_bundle, sub_sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="overlap_sub",
        extra_model_artifacts=((sub_pq_ref, sub_pq),),
    )
    sub_bdir = tmp_path / "overlap_sub_bundle"
    write_strategy_bundle(sub_bundle, sub_bdir, source_files=sub_sources)
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        target.export(sub_bdir, sub_bdir / "signals", target_positions=pos_05, formats=("jsonl",))
    assert verify_strategy_bundle(sub_bdir).bundle_hash == sub_bundle.bundle_hash

    # 3. Explicit target_positions export + verify_alignment ignores unused unparseable bundle signal artifact
    bad_bound_sig = tmp_path / "unparseable_signal.parquet"
    bad_bound_sig.write_bytes(b"corrupt-parquet-bytes-matching-sha")
    bad_bound_ref = BundleArtifactRef(
        path="signals/target_positions.parquet",
        sha256=sha256_file(bad_bound_sig),
        artifact_type="target_positions_parquet",
        size_bytes=bad_bound_sig.stat().st_size,
    )
    unparseable_bundle, unparseable_sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="unused_bad_bundle_sig",
        extra_model_artifacts=((bad_bound_ref, bad_bound_sig),),
    )
    unparseable_bdir = tmp_path / "unused_bad_sig_bundle"
    write_strategy_bundle(unparseable_bundle, unparseable_bdir, source_files=unparseable_sources)
    explicit_out = tmp_path / "explicit_over_bad_bundle_sig"
    explicit_manifest = target.export(unparseable_bdir, explicit_out, target_positions=pos_02)
    assert explicit_manifest.metadata["signal_source"] == "explicit_target_positions"
    explicit_report = target.verify_alignment(unparseable_bdir, explicit_out)
    assert explicit_report.passed is True

    # 4. Huge ints exceeding float range in top_pct and manifest initial_positions return reports (no OverflowError)
    huge_int = 10**400
    huge_pct_bundle = replace(
        bundle,
        rebalance_config={"rebalance_freq": "weekly", "selection_mode": "top_pct", "top_pct": huge_int},
        bundle_hash="",
    )
    huge_pct_report = target.check_compatibility(huge_pct_bundle)
    assert huge_pct_report.compatible is False
    assert huge_pct_report.issues[0].code == TARGET_BUNDLE_INVALID

    overflow_init_dir = tmp_path / "overflow_init_positions"
    m_ovf = target.export(bdir, overflow_init_dir, target_positions=pos_01, formats=("jsonl",))
    _resign_manifest_dir(
        overflow_init_dir,
        bundle=bundle,
        artifacts=m_ovf.artifacts,
        metadata={**dict(m_ovf.metadata), "initial_positions": {"IF2406.CCFX": huge_int}},
    )
    rep_ovf_init = target.verify_alignment(bdir, overflow_init_dir)
    assert rep_ovf_init.passed is False
    assert rep_ovf_init.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    # 5. Duplicate declared formats rejected in both export and verify_alignment
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(bdir, tmp_path / "dup_fmt_export", target_positions=pos_01, formats=("jsonl", "jsonl"))

    dup_fmt_dir = tmp_path / "dup_fmt_verify"
    m_dup_fmt = target.export(bdir, dup_fmt_dir, target_positions=pos_01, formats=("jsonl",))
    _resign_manifest_dir(
        dup_fmt_dir,
        bundle=bundle,
        artifacts=m_dup_fmt.artifacts,
        metadata={**dict(m_dup_fmt.metadata), "formats": ["jsonl", "jsonl"]},
    )
    rep_dup_fmt = target.verify_alignment(bdir, dup_fmt_dir)
    assert rep_dup_fmt.passed is False
    assert rep_dup_fmt.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED


def test_review_regressions_managed_dir_conflict_position_overflow_and_signal_source_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SignalExportTarget()
    bundle, sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="review_round3_regressions",
    )
    bdir = tmp_path / "r3_bundle"
    write_strategy_bundle(bundle, bdir, source_files=sources)

    pos_01 = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.1,
        }
    ]
    pos_02 = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.2,
        }
    ]

    # 1a. Pre-existing directory occupying a managed filename (e.g., target_positions.parquet)
    #     is rejected before publishing, preserves the original directory, and leaves no valid manifest.
    dir_occupied_out = tmp_path / "dir_occupied_out"
    blocking_dir = dir_occupied_out / "target_positions.parquet"
    blocking_dir.mkdir(parents=True)
    sentinel_file = blocking_dir / "keep.txt"
    sentinel_file.write_text("keep-me", encoding="utf-8")

    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        target.export(bdir, dir_occupied_out, target_positions=pos_01, formats=("jsonl",))

    assert blocking_dir.is_dir()
    assert sentinel_file.read_text(encoding="utf-8") == "keep-me"
    assert not (dir_occupied_out / "target_positions.jsonl").exists()
    assert not (dir_occupied_out / "trade_orders.jsonl").exists()
    assert not (dir_occupied_out / DEPLOY_MANIFEST_FILENAME).exists()
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        verify_deploy_manifest(dir_occupied_out)

    # 1b. Mid-publish failure rolls back only paths modified by the failed export and keeps prior state intact.
    partial_out = tmp_path / "partial_publish_out"
    m_prior = target.export(bdir, partial_out, target_positions=pos_01, formats=("jsonl",))
    prior_jsonl = (partial_out / "target_positions.jsonl").read_bytes()
    prior_orders = (partial_out / "trade_orders.jsonl").read_bytes()
    prior_manifest = (partial_out / DEPLOY_MANIFEST_FILENAME).read_bytes()

    real_replace = os.replace

    def _fail_on_orders_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(dst).name == "trade_orders.jsonl":
            raise OSError("injected mid-publish replace failure")
        real_replace(src, dst)

    with monkeypatch.context() as mctx:
        mctx.setattr("targets.signal_export.exporter.os.replace", _fail_on_orders_replace)
        with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
            target.export(bdir, partial_out, target_positions=pos_02, formats=("parquet", "jsonl"))

    assert not (partial_out / "target_positions.parquet").exists()
    assert (partial_out / "target_positions.jsonl").read_bytes() == prior_jsonl
    assert (partial_out / "trade_orders.jsonl").read_bytes() == prior_orders
    assert (partial_out / DEPLOY_MANIFEST_FILENAME).read_bytes() == prior_manifest
    assert verify_deploy_manifest(partial_out).manifest_hash == m_prior.manifest_hash

    # 2a. Record with target_weight=10**400 triggers stable DeployTargetError / AlignmentReport (no OverflowError).
    overflow_records = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 10**400,
        }
    ]
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(bdir, tmp_path / "overflow_record_export", target_positions=overflow_records)

    valid_out = tmp_path / "valid_for_overflow_ref"
    target.export(bdir, valid_out, target_positions=pos_01)
    rep_overflow_ref = target.verify_alignment(bdir, valid_out, reference_positions=overflow_records)
    assert rep_overflow_ref.passed is False
    assert rep_overflow_ref.error_code == TARGET_ALIGNMENT_MISMATCH

    # 2b. Extreme finite positions (1e308 vs -1e308) producing inf diff return TARGET_ALIGNMENT_MISMATCH report.
    pos_max = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 1e308,
        }
    ]
    pos_min = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": -1e308,
        }
    ]
    extreme_out = tmp_path / "extreme_finite_out"
    target.export(bdir, extreme_out, target_positions=pos_max)
    rep_inf_diff = target.verify_alignment(bdir, extreme_out, reference_positions=pos_min)
    assert rep_inf_diff.passed is False
    assert rep_inf_diff.error_code == TARGET_ALIGNMENT_MISMATCH
    assert rep_inf_diff.max_abs_diff is None
    assert rep_inf_diff.metrics.get("reference_max_abs_diff_overflow") is True

    # 3. Manifest signal_source enum & mutual-exclusion checks are enforced even when reference_positions is given.
    unknown_src_dir = tmp_path / "unknown_signal_source"
    m_src = target.export(bdir, unknown_src_dir, target_positions=pos_01, formats=("jsonl",))
    _resign_manifest_dir(
        unknown_src_dir,
        bundle=bundle,
        artifacts=m_src.artifacts,
        metadata={**dict(m_src.metadata), "signal_source": "unknown_source"},
    )
    rep_unknown_no_ref = target.verify_alignment(bdir, unknown_src_dir)
    assert rep_unknown_no_ref.passed is False
    assert rep_unknown_no_ref.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED
    rep_unknown_with_ref = target.verify_alignment(bdir, unknown_src_dir, reference_positions=pos_01)
    assert rep_unknown_with_ref.passed is False
    assert rep_unknown_with_ref.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    conflict_src_dir = tmp_path / "conflict_signal_source"
    m_conf = target.export(bdir, conflict_src_dir, target_positions=pos_01, formats=("jsonl",))
    _resign_manifest_dir(
        conflict_src_dir,
        bundle=bundle,
        artifacts=m_conf.artifacts,
        metadata={
            **dict(m_conf.metadata),
            "signal_source": "explicit_target_positions",
            "signal_artifact_path": "signals/target_positions.parquet",
        },
    )
    rep_conf_no_ref = target.verify_alignment(bdir, conflict_src_dir)
    assert rep_conf_no_ref.passed is False
    assert rep_conf_no_ref.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED
    rep_conf_with_ref = target.verify_alignment(bdir, conflict_src_dir, reference_positions=pos_01)
    assert rep_conf_with_ref.passed is False
    assert rep_conf_with_ref.error_code == TARGET_ARTIFACT_INTEGRITY_FAILED

    # Also verify explicit_target_positions with reference_positions still skips unchosen corrupt bundle signal
    bad_bound_sig = tmp_path / "r3_unparseable_signal.parquet"
    bad_bound_sig.write_bytes(b"corrupt-parquet-bytes-matching-sha")
    bad_bound_ref = BundleArtifactRef(
        path="signals/target_positions.parquet",
        sha256=sha256_file(bad_bound_sig),
        artifact_type="target_positions_parquet",
        size_bytes=bad_bound_sig.stat().st_size,
    )
    unparseable_bundle, unparseable_sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="r3_unused_bad_bundle_sig",
        extra_model_artifacts=((bad_bound_ref, bad_bound_sig),),
    )
    unparseable_bdir = tmp_path / "r3_unused_bad_sig_bundle"
    write_strategy_bundle(unparseable_bundle, unparseable_bdir, source_files=unparseable_sources)
    explicit_out = tmp_path / "r3_explicit_over_bad_sig"
    target.export(unparseable_bdir, explicit_out, target_positions=pos_02)
    assert target.verify_alignment(unparseable_bdir, explicit_out, reference_positions=pos_02).passed is True

    # And bundle_artifact manifest with mismatched signal_artifact_path is rejected even with reference_positions
    bad_path_dir = tmp_path / "bad_bundle_sig_path"
    m_bp = target.export(unparseable_bdir, bad_path_dir, target_positions=pos_02, formats=("jsonl",))
    _resign_manifest_dir(
        bad_path_dir,
        bundle=unparseable_bundle,
        artifacts=m_bp.artifacts,
        metadata={
            **dict(m_bp.metadata),
            "signal_source": "bundle_artifact",
            "signal_artifact_path": "signals/wrong.parquet",
        },
    )
    rep_bp_with_ref = target.verify_alignment(
        unparseable_bdir, bad_path_dir, reference_positions=pos_02
    )
    assert rep_bp_with_ref.passed is False
    assert rep_bp_with_ref.error_code == TARGET_BUNDLE_INVALID


def test_review_r4_regressions_manifest_rollback_backup_lot_size_overflow_and_cyclic_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SignalExportTarget()
    bundle, sources = _make_bundle(
        tmp_path,
        slot_id="cn_futures",
        frequency="1d",
        allow_short=True,
        benchmark_instrument="IF2406.CCFX",
        strategy_id="review_round4_regressions",
    )
    bdir = tmp_path / "r4_bundle"
    write_strategy_bundle(bundle, bdir, source_files=sources)

    pos_01 = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.1,
        }
    ]

    # 1. When post-publish verification fails AND restoring prior deploy_manifest.json fails,
    #    the failed new manifest is withdrawn (cannot pass verify_deploy_manifest) and the old
    #    manifest backup is preserved for recovery.
    rb_out = tmp_path / "r4_manifest_rollback_fail"
    m_v1 = target.export(
        bdir,
        rb_out,
        target_positions=pos_01,
        formats=("jsonl",),
        created_at_utc="2024-03-01T08:00:00Z",
    )
    v1_manifest_bytes = (rb_out / DEPLOY_MANIFEST_FILENAME).read_bytes()
    assert not list(rb_out.glob(".signal_export_backup_*"))

    real_replace = os.replace

    def _fail_manifest_restore(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        src_p = Path(src)
        dst_p = Path(dst)
        if src_p.parent.name.startswith(".signal_export_backup_") and dst_p.name == DEPLOY_MANIFEST_FILENAME:
            raise OSError("injected manifest rollback replace failure")
        real_replace(src, dst)

    def _fail_post_verify(_artifact: object, **_kwargs: object) -> object:
        raise DeployTargetError(
            "injected post-publish verify failure",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )

    with monkeypatch.context() as mctx:
        mctx.setattr("targets.signal_export.exporter.os.replace", _fail_manifest_restore)
        mctx.setattr("targets.signal_export.exporter.verify_deploy_manifest", _fail_post_verify)
        with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
            target.export(
                bdir,
                rb_out,
                target_positions=pos_01,
                formats=("jsonl",),
                created_at_utc="2024-03-01T09:00:00Z",
            )

    assert not (rb_out / DEPLOY_MANIFEST_FILENAME).exists()
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        verify_deploy_manifest(rb_out)

    backup_dirs = sorted(rb_out.glob(".signal_export_backup_*"))
    assert len(backup_dirs) == 1
    preserved_manifest = backup_dirs[0] / DEPLOY_MANIFEST_FILENAME
    assert preserved_manifest.is_file()
    assert preserved_manifest.read_bytes() == v1_manifest_bytes

    # Manual recovery using the preserved backup restores the prior v1 manifest cleanly.
    real_replace(preserved_manifest, rb_out / DEPLOY_MANIFEST_FILENAME)
    assert verify_deploy_manifest(rb_out).manifest_hash == m_v1.manifest_hash

    # 2. Huge lot_size values (10**400, 10**20, 2**63, float 1e20) are rejected before pandas/PyArrow
    #    conversion: export raises TARGET_BUNDLE_INVALID, alignment returns TARGET_ALIGNMENT_MISMATCH.
    valid_lot_out = tmp_path / "r4_valid_lot_out"
    max_valid_lot = (1 << 63) - 1
    pos_max_lot = [
        {
            "timestamp_utc": "2024-03-01T07:00:00Z",
            "session_id": "2024-03-01",
            "instrument_id": "IF2406.CCFX",
            "target_weight": 0.1,
            "lot_size": max_valid_lot,
        }
    ]
    target.export(bdir, valid_lot_out, target_positions=pos_max_lot)
    assert target.verify_alignment(bdir, valid_lot_out, reference_positions=pos_max_lot).passed is True

    for bad_lot in (10**400, 10**20, 1 << 63, 1e20):
        bad_lot_rows = [
            {
                "timestamp_utc": "2024-03-01T07:00:00Z",
                "session_id": "2024-03-01",
                "instrument_id": "IF2406.CCFX",
                "target_weight": 0.1,
                "lot_size": bad_lot,
            }
        ]
        with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
            target.export(
                bdir,
                tmp_path / f"bad_lot_export_{type(bad_lot).__name__}_{abs(hash(bad_lot))}",
                target_positions=bad_lot_rows,
            )
        rep_bad_lot = target.verify_alignment(
            bdir,
            valid_lot_out,
            reference_positions=bad_lot_rows,
        )
        assert rep_bad_lot.passed is False
        assert rep_bad_lot.error_code == TARGET_ALIGNMENT_MISMATCH

    # Also ensure an oversized lot_size in a re-signed exported JSONL artifact returns TARGET_ALIGNMENT_MISMATCH.
    jsonl_lot_dir = tmp_path / "r4_tampered_lot_jsonl"
    m_lot = target.export(bdir, jsonl_lot_dir, target_positions=pos_max_lot, formats=("jsonl",))
    jsonl_path = jsonl_lot_dir / "target_positions.jsonl"
    row_obj = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    row_obj["lot_size"] = 10**20
    jsonl_path.write_text(json.dumps(row_obj, separators=(",", ":")) + "\n", encoding="utf-8")
    resigned_artifacts = tuple(
        DeployArtifactEntry(
            path=entry.path,
            sha256=sha256_file(jsonl_lot_dir / entry.path),
            artifact_type=entry.artifact_type,
            size_bytes=(jsonl_lot_dir / entry.path).stat().st_size,
        )
        for entry in m_lot.artifacts
    )
    _resign_manifest_dir(
        jsonl_lot_dir,
        bundle=bundle,
        artifacts=resigned_artifacts,
        metadata=dict(m_lot.metadata),
    )
    rep_tampered_lot = target.verify_alignment(bdir, jsonl_lot_dir)
    assert rep_tampered_lot.passed is False
    assert rep_tampered_lot.error_code == TARGET_ALIGNMENT_MISMATCH

    # 3. Cyclic symlinks on managed filenames (including stale unchosen format target_positions.parquet)
    #    converge to TARGET_ARTIFACT_INTEGRITY_FAILED instead of leaking RuntimeError.
    cyclic_out = tmp_path / "r4_cyclic_symlink_out"
    cyclic_out.mkdir(parents=True)
    cyclic_pq = cyclic_out / "target_positions.parquet"
    os.symlink("target_positions.parquet", cyclic_pq)

    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        target.export(bdir, cyclic_out, target_positions=pos_01, formats=("jsonl",))

    assert cyclic_pq.is_symlink()
    assert not (cyclic_out / "target_positions.jsonl").exists()
    assert not (cyclic_out / "trade_orders.jsonl").exists()
    assert not (cyclic_out / DEPLOY_MANIFEST_FILENAME).exists()

    loop_manifest_dir = tmp_path / "r4_cyclic_manifest_dir"
    loop_manifest_dir.mkdir(parents=True)
    os.symlink("deploy_manifest.json", loop_manifest_dir / DEPLOY_MANIFEST_FILENAME)
    with pytest.raises(DeployTargetError, match=TARGET_ARTIFACT_INTEGRITY_FAILED):
        verify_deploy_manifest(loop_manifest_dir)


@pytest.mark.parametrize("relative_path", ["manifest.json", "models/weights.json"])
def test_bundle_gate_rejects_internal_symlink_loop(tmp_path: Path, relative_path: str) -> None:
    bundle, sources = _make_bundle(
        tmp_path,
        slot_id="declarative_custom",
        frequency="1d",
        allow_short=True,
        benchmark_instrument=None,
    )
    bundle_dir = tmp_path / "bundle"
    write_strategy_bundle(bundle, bundle_dir, source_files=sources)
    source = StrategyBundleSource.from_dir(bundle_dir)
    target = SignalExportTarget()
    positions = [
        {
            "timestamp_utc": "2023-03-01T00:00:00Z",
            "session_id": "2023-03-01",
            "instrument_id": "BTC-USD",
            "target_weight": 0.1,
        }
    ]
    output_dir = tmp_path / "output"
    target.export(source, output_dir, target_positions=positions, formats=("jsonl",))

    loop = bundle_dir / relative_path
    loop.unlink()
    loop.symlink_to(loop.name)

    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        StrategyBundleSource.from_dir(bundle_dir)
    with pytest.raises(DeployTargetError, match=TARGET_BUNDLE_INVALID):
        target.export(source, output_dir, target_positions=positions, formats=("jsonl",))
    report = target.verify_alignment(bundle_dir, output_dir)
    assert report.passed is False
    assert report.error_code == TARGET_BUNDLE_INVALID
    assert report.checks[0].name == "bundle_valid"


def test_targets_core_and_signal_export_contain_no_jq_ashare_or_private_paths() -> None:
    forbidden_patterns = (
        re.compile(r"\bjqdata\b"),
        re.compile(r"\bjqfactor\b"),
        re.compile(r"jq_api_upload_run"),
        re.compile(r"manual_cloud_deploy"),
        re.compile(r"\.XSHG\b"),
        re.compile(r"\.XSHE\b"),
        re.compile(r"000300"),
    )
    target_files = [
        REPO_ROOT / "targets" / "__init__.py",
        REPO_ROOT / "targets" / "contracts.py",
        REPO_ROOT / "targets" / "registry.py",
        *sorted((REPO_ROOT / "targets" / "signal_export").rglob("*.py")),
    ]
    assert all(p.is_file() for p in target_files)
    for py_file in target_files:
        text = py_file.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert not pattern.search(text), (
                f"Forbidden pattern {pattern.pattern!r} found in {py_file.relative_to(REPO_ROOT)}"
            )
