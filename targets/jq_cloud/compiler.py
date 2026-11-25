"""Build an offline JoinQuant candidate, probe and content-addressed manifest."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from factor_synthesis.strategy_bundle import StrategyBundle, sha256_file
from scripts.alignment.prepare_cloud_dumper import render_bundle_probe
from scripts.alignment.virtual_cloud_check import inspect_candidate_source
from scripts.build.assembler import compose_bundle_entry_block
from scripts.build_jq_strategy import (
    build_dependency_map,
    compile_single_file_strategy,
    lift_minimal_factor_base,
    select_and_prepare_factors,
)
from targets.contracts import (
    TARGET_ARTIFACT_INTEGRITY_FAILED,
    TARGET_BUNDLE_INVALID,
    DeployArtifactEntry,
    DeployArtifactManifest,
    DeployTargetError,
    resolve_path_checked,
    resolve_verified_bundle_with_dir,
    utc_now_iso,
    verify_deploy_manifest,
    write_deploy_manifest,
)
from targets.jq_cloud.compatibility import (
    BARS_PER_DAY,
    REPO_ROOT,
    TARGET_ID,
    BundleInput,
    FactorFiles,
    collect_compilation_inputs,
)

STRATEGY_FILENAME = "jq_strategy_candidate.py"
DUMPER_FILENAME = "cloud_factor_dumper.py"
REQUIREMENTS_FILENAME = "factor_requirements_slice.json"
ARTIFACT_TYPES = {
    STRATEGY_FILENAME: "jq_strategy_candidate",
    DUMPER_FILENAME: "cloud_factor_dumper",
    REQUIREMENTS_FILENAME: "factor_requirements_slice",
}
RUNTIME_DIR = REPO_ROOT / "scripts/build/cloud_runtime"
RUNTIME_NAMES = ("factor_base.py.txt", "jq_factor_adapter.py.txt")


def frozen_runtime_hashes() -> dict[str, str]:
    return {name: sha256_file(RUNTIME_DIR / name) for name in RUNTIME_NAMES}


def bundle_binding(bundle: StrategyBundle) -> dict[str, str]:
    return {
        "bundle_hash": bundle.bundle_hash,
        "slot_id": bundle.slot_id,
        "strategy_id": bundle.strategy_id,
        "synthesis_method": bundle.synthesis_method,
    }


def _requirement_slice(
    bundle: StrategyBundle,
    specs: dict[str, dict[str, Any]],
    factor_plans: dict[str, tuple[str, dict[str, Any], list[str]]],
) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for factor in bundle.factors:
        spec = specs[factor.factor_id]
        name = spec["class_name"]
        freq, entry, fields = factor_plans[factor.factor_id]
        plan = dict(entry.get("requirements", {}).get(freq, {}))
        if "min_bars" not in plan:
            days = plan.get("min_days")
            if days is not None:
                if type(days) is not int or days <= 0:
                    raise ValueError(f"{factor.factor_id}: min_days must be a positive integer")
                plan["min_bars"] = days * BARS_PER_DAY[freq]
            else:
                plan["min_bars"] = spec.get("required_window")
        slack = plan.get("slack_ratio", 0)
        if (not fields or type(plan["min_bars"]) is not int or plan["min_bars"] <= 0
            or type(slack) not in (int, float) or not 0 <= slack < float("inf")):
            raise ValueError(f"{factor.factor_id}: requires explicit input fields and a positive lookback window")
        if freq == "daily":
            plan.setdefault("min_days", plan["min_bars"])
        plan["fields"] = fields
        runtime = entry.get("runtime", {})
        window = runtime.get("window", plan["min_bars"])
        if type(window) is not int or window <= 0:
            raise ValueError(f"{factor.factor_id}: invalid runtime.window")
        factor_plan = dict(entry, requirements={freq: plan})
        if bundle.frequency != "1d":
            factor_plan["runtime"] = dict(runtime, output_date="bar")
        if sliced.setdefault(name, factor_plan) != factor_plan:
            raise ValueError(f"{name}: different requirement plans for the same factor class")
    return sliced


def export_bundle(
    source: BundleInput,
    output_dir: str | Path,
    *,
    bundle_dir: str | Path | None = None,
    factor_files: FactorFiles | None = None,
    requirements_path: str | Path | None = None,
    cloud_mapping_path: str | Path | None = None,
    generate_dumper: bool = True,
    created_at_utc: str | None = None,
) -> DeployArtifactManifest:
    bundle, verified_dir = resolve_verified_bundle_with_dir(source, bundle_dir=bundle_dir)
    assert verified_dir is not None
    issues, inputs = collect_compilation_inputs(
        bundle, verified_dir, factor_files=factor_files,
        cloud_mapping_path=cloud_mapping_path,
        requirements_path=requirements_path,
    )
    if issues:
        issue = issues[0]
        raise DeployTargetError(issue.message, code=issue.code)
    assert inputs is not None

    root = resolve_path_checked(
        Path(output_dir).expanduser(), label="output_dir",
        code=TARGET_ARTIFACT_INTEGRITY_FAILED,
    )
    protected = (
        verified_dir,
        RUNTIME_DIR,
        REPO_ROOT / "factors",
        REPO_ROOT / "backtest_and_analysis/evaluation/calibration_data",
        REPO_ROOT / "state",
        Path("/Volumes"),
    )
    if any(root.is_relative_to(path.resolve()) for path in protected):
        raise DeployTargetError(
            f"Output directory overlaps a protected input: {root}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise DeployTargetError(
            f"JoinQuant export requires an empty output directory: {root}",
            code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        )

    try:
        specs = inputs.specs
        requirements = _requirement_slice(bundle, specs, inputs.factor_plans)
        classes = {spec["class_name"]: spec for spec in specs.values()}
        prepared = select_and_prepare_factors(list(classes.values()), list(classes))
        runtime_hashes = frozen_runtime_hashes()
        metadata = {
            **bundle_binding(bundle),
            "frequency": bundle.frequency,
            "models": inputs.models,
            "preprocessing_params": bundle.to_dict()["preprocessing_params"],
            "selection": bundle.to_dict()["rebalance_config"],
            "factor_classes": {fid: spec["class_name"] for fid, spec in specs.items()},
        }
        candidate = compile_single_file_strategy(
            {
                "factor_base_code": lift_minimal_factor_base(),
                "factor_blocks": [item["code"] for item in prepared],
                "dependencies": build_dependency_map(prepared, requirements),
                "cloud_mapping": inputs.mapping,
            },
            prepared,
            requirements,
            entry_block=compose_bundle_entry_block(metadata),
            keep_logs=True,
        )
        for check in inspect_candidate_source(candidate, STRATEGY_FILENAME):
            if not check["passed"]:
                raise ValueError(check["reason"])
        if runtime_hashes != frozen_runtime_hashes():
            raise ValueError("Frozen runtime sources changed during compilation")
    except (ValueError, TypeError, KeyError, AttributeError, OSError, SyntaxError, RuntimeError) as exc:
        if isinstance(exc, DeployTargetError):
            raise
        raise DeployTargetError(f"JoinQuant compilation failed: {exc}", code=TARGET_BUNDLE_INVALID) from exc

    payloads = {
        STRATEGY_FILENAME: candidate,
        REQUIREMENTS_FILENAME: json.dumps({"version": "2", "factors": requirements}, indent=2) + "\n",
    }
    if generate_dumper:
        payloads[DUMPER_FILENAME] = render_bundle_probe(candidate)

    # Stage files beside the destination. Publishing the directory binds the
    # manifest and every artifact in one rename, including on retry/concurrent export.
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".jq-cloud-", dir=root.parent) as temporary:
            stage = Path(temporary)
            entries = []
            for filename, contents in payloads.items():
                path = stage / filename
                path.write_text(contents, encoding="utf-8")
                entries.append(DeployArtifactEntry(
                    filename, sha256_file(path), ARTIFACT_TYPES[filename], path.stat().st_size,
                ))
            manifest = DeployArtifactManifest(
                target_id=TARGET_ID,
                strategy_id=bundle.strategy_id,
                bundle_hash=bundle.bundle_hash,
                slot_id=bundle.slot_id,
                frequency=bundle.frequency,
                created_at_utc=utc_now_iso() if created_at_utc is None else created_at_utc,
                artifacts=tuple(entries),
                metadata={
                    "auto_deploy_blocked": True,
                    "human_approval_required": True,
                    "frozen_cloud_runtime_hashes": runtime_hashes,
                },
            )
            write_deploy_manifest(manifest, stage)
            stage.replace(root)
    except OSError as exc:
        raise DeployTargetError(
            f"Cannot publish JoinQuant artifacts: {exc}", code=TARGET_ARTIFACT_INTEGRITY_FAILED,
        ) from exc
    return verify_deploy_manifest(root)
