"""Verify the manifest and replay JoinQuant candidate signals on supplied inputs."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.alignment.virtual_cloud_check import execute_bundle_candidate, inspect_bundle_candidate
from targets.contracts import (
    TARGET_ALIGNMENT_MISMATCH,
    TARGET_ARTIFACT_INTEGRITY_FAILED,
    TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
    TARGET_SIGNAL_SOURCE_MISSING,
    AlignmentCheckItem,
    AlignmentReport,
    DeployArtifactManifest,
    DeployTargetError,
    resolve_contained_file,
    resolve_verified_bundle_with_dir,
    verify_deploy_manifest,
)
from targets.jq_cloud.compatibility import SLOT_ID, TARGET_ID, BundleInput, load_replay_models
from targets.jq_cloud.compiler import (
    ARTIFACT_TYPES,
    DUMPER_FILENAME,
    REQUIREMENTS_FILENAME,
    STRATEGY_FILENAME,
    bundle_binding,
    frozen_runtime_hashes,
)

ALIGNMENT_REPORT_FILENAME = "alignment_report.json"
ATOL = 1e-12


def _read_artifact(root: Path, name: str, hashes: dict[str, str]) -> str:
    path = resolve_contained_file(root, name, f"artifacts[{name!r}]")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != hashes[name]:
        raise ValueError(f"{name}: SHA-256 changed before reading artifact")
    return contents.decode("utf-8")


def _literal(source: str, name: str) -> Any:
    assignments = [
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError(f"Candidate requires one {name} assignment")
    return ast.literal_eval(assignments[0])


def _virtual_price_api(frames: Mapping[str, pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Expose price tables through get_price so the candidate owns all bar validation."""
    if not isinstance(frames, Mapping) or not frames:
        raise ValueError("bars_by_frequency must map JoinQuant frequencies to DataFrames")
    for frequency, frame in frames.items():
        if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
            or not {"date", "code"}.issubset(frame.columns)):
            raise ValueError(f"Virtual bars for {frequency} require unique columns including date and code")

    def get_price(*, security, end_date, frequency, fields=None, count=None, start_date=None, **kwargs):
        frame = frames[frequency]
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        securities = [security] if isinstance(security, str) else security
        eligible = frame["code"].isin(securities) & (dates <= pd.Timestamp(end_date))
        if start_date is not None:
            eligible &= dates >= pd.Timestamp(start_date)
        data = frame.loc[eligible].assign(date=dates.loc[eligible]).sort_values("date", kind="mergesort")
        if count is not None:
            data = data.groupby("code", sort=False).tail(count)
        return data if fields is None else data[["date", "code", *fields]]

    return get_price


def _replay_api(inputs: Mapping[str, Any]) -> dict[str, Any]:
    stocks = inputs.get("stocks")
    if (not isinstance(stocks, (list, tuple)) or not stocks
        or any(not isinstance(stock, str) or not stock for stock in stocks)
        or len(set(stocks)) != len(stocks)):
        raise ValueError("virtual_cloud_inputs.stocks requires unique stock codes")
    asof = inputs.get("asof")
    if not isinstance(asof, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?)?", asof
    ):
        raise ValueError("virtual_cloud_inputs.asof requires an ISO date or timestamp")
    try:
        if pd.isna(pd.Timestamp(asof)):
            raise ValueError
    except (ValueError, OverflowError) as exc:
        raise ValueError("virtual_cloud_inputs.asof requires an ISO date or timestamp") from exc
    api = dict(inputs.get("api", {}))
    if inputs.get("bars_by_frequency") is not None:
        api["get_price"] = _virtual_price_api(inputs["bars_by_frequency"])
    if not callable(api.get("get_price")):
        raise ValueError("Supply bars_by_frequency or an offline get_price API")
    return api


def _compare(
    name: str, observed: pd.DataFrame | pd.Series, expected: pd.DataFrame | pd.Series
) -> tuple[AlignmentCheckItem, float | None, int]:
    axes_match = type(expected) is type(observed) and all(
        actual.is_unique and reference.is_unique
        and len(actual) == len(reference) and actual.difference(reference).empty
        for actual, reference in zip(observed.axes, expected.axes)
    )
    if not axes_match or not observed.size:
        return AlignmentCheckItem(
            name, False, "Missing, empty, duplicated or different alignment keys", TARGET_ALIGNMENT_MISMATCH,
        ), None, 0
    reference = expected.reindex_like(observed)
    actual_values = observed.to_numpy(dtype=float, na_value=np.nan)
    expected_values = reference.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(actual_values).any() or np.isinf(expected_values).any():
        return AlignmentCheckItem(name, False, "Infinite values", TARGET_ALIGNMENT_MISMATCH), None, observed.size
    with np.errstate(over="ignore", invalid="ignore"):
        matches = np.isclose(actual_values, expected_values, rtol=0.0, atol=ATOL, equal_nan=True)
        finite = np.isfinite(actual_values) & np.isfinite(expected_values)
        diff = float(np.max(np.abs(actual_values[finite] - expected_values[finite]), initial=0.0))
    passed = bool(matches.all())
    reason = "" if passed else f"{int((~matches).sum())} of {observed.size} values differ"
    return AlignmentCheckItem(name, passed, reason, "" if passed else TARGET_ALIGNMENT_MISMATCH), (
        diff if np.isfinite(diff) else None
    ), observed.size


def _write_report(report: AlignmentReport, root: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(report.to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.close()
            os.replace(temporary, root / ALIGNMENT_REPORT_FILENAME)
        finally:
            temporary.unlink(missing_ok=True)


def verify_bundle_alignment(
    source: BundleInput,
    artifact: DeployArtifactManifest | str | Path,
    *,
    local_factor_values: pd.DataFrame | None = None,
    local_positions: pd.Series | Mapping[str, float] | None = None,
    virtual_cloud_inputs: Mapping[str, Any] | None = None,
    trusted_manifest_hash: str | None = None,
    requirements_path: str | Path | None = None,
    bundle_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_report: bool = False,
) -> AlignmentReport:
    """Replay only when the caller independently trusts the export's manifest hash."""
    checks: list[AlignmentCheckItem] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {"rtol": 0, "atol": ATOL}
    bundle_hash = ""
    root = None
    max_abs_diff = None
    stage = "bundle_valid"
    code = TARGET_ARTIFACT_INTEGRITY_FAILED

    try:
        bundle, verified_dir = resolve_verified_bundle_with_dir(source, bundle_dir=bundle_dir)
        assert verified_dir is not None
        bundle_hash = bundle.bundle_hash
        checks.append(AlignmentCheckItem(stage, True))
        stage = "manifest_integrity"
        manifest = verify_deploy_manifest(artifact, output_dir=output_dir)
        artifact_root = manifest._base_dir
        hashes = {item.path: item.sha256 for item in manifest.artifacts}
        required = {STRATEGY_FILENAME, REQUIREMENTS_FILENAME}
        if not required <= hashes.keys() <= ARTIFACT_TYPES.keys() or any(
            ARTIFACT_TYPES[item.path] != item.artifact_type for item in manifest.artifacts
        ):
            raise ValueError("JoinQuant manifest has an invalid artifact set")
        checks.append(AlignmentCheckItem(stage, True))
        stage = "bundle_binding"
        if (manifest.target_id != TARGET_ID or bundle.slot_id != SLOT_ID
            or any(getattr(manifest, key) != getattr(bundle, key) for key in
                   ("bundle_hash", "slot_id", "strategy_id", "frequency"))):
            raise ValueError("JoinQuant manifest does not match the StrategyBundle")
        if (manifest.metadata.get("auto_deploy_blocked") is not True
            or manifest.metadata.get("human_approval_required") is not True
            or manifest.metadata.get("frozen_cloud_runtime_hashes") != frozen_runtime_hashes()):
            raise ValueError("Missing approval requirements or inconsistent frozen runtime hashes")
        root = artifact_root
        checks.append(AlignmentCheckItem(stage, True))

        stage = "trusted_compilation"
        if not isinstance(trusted_manifest_hash, str) or trusted_manifest_hash != manifest.manifest_hash:
            raise ValueError("Supply the independently recorded manifest hash from a trusted JoinQuant export")
        checks.append(AlignmentCheckItem(stage, True))

        stage = "candidate_binding"
        candidate = _read_artifact(root, STRATEGY_FILENAME, hashes)
        requirements = _read_artifact(root, REQUIREMENTS_FILENAME, hashes)
        requirement_payload = json.loads(requirements)
        if requirements_path is not None:
            explicit = Path(requirements_path).expanduser().read_bytes()
            if hashlib.sha256(explicit).hexdigest() != hashes[REQUIREMENTS_FILENAME]:
                raise ValueError("requirements_path does not match the manifest-bound slice")
        metadata = _literal(candidate, "STRATEGY_BUNDLE_METADATA")
        factor_classes = metadata["factor_classes"]
        if (any(metadata.get(key) != value for key, value in bundle_binding(bundle).items())
            or metadata.get("frequency") != bundle.frequency
            or metadata.get("selection") != bundle.to_dict()["rebalance_config"]
            or metadata.get("preprocessing_params") != bundle.to_dict()["preprocessing_params"]
            or metadata.get("models") != load_replay_models(bundle, verified_dir)
            or set(factor_classes) != {spec.factor_id for spec in bundle.factors}
            or set(factor_classes.values()) != set(requirement_payload["factors"])
            or _literal(candidate, "FACTOR_REQUIREMENTS") != requirement_payload):
            raise ValueError("Candidate metadata or requirements do not match the verified bundle")
        if DUMPER_FILENAME in hashes:
            from scripts.alignment.prepare_cloud_dumper import render_bundle_probe

            if _read_artifact(root, DUMPER_FILENAME, hashes) != render_bundle_probe(candidate):
                raise ValueError("Cloud probe is not bound to the deployed factor runtime")
        checks.append(AlignmentCheckItem(stage, True))

        stage = "cloud_preflight"
        code = TARGET_EXECUTION_CONTRACT_UNSUPPORTED
        static_checks, warnings = inspect_bundle_candidate(
            candidate, STRATEGY_FILENAME, sorted(set(factor_classes.values())), requirement_payload,
            strategy_frequency=bundle.frequency,
        )
        checks.extend(
            AlignmentCheckItem(**item, code="" if item["passed"] else code)
            for item in static_checks
        )
        if all(item["passed"] for item in static_checks):
            stage = "reference_inputs"
            code = TARGET_SIGNAL_SOURCE_MISSING
            if (not isinstance(virtual_cloud_inputs, Mapping) or not isinstance(local_factor_values, pd.DataFrame)
                or local_factor_values.empty or not isinstance(local_positions, (pd.Series, Mapping))
                or len(local_positions) == 0):
                raise ValueError("Supply virtual_cloud_inputs, local_factor_values and local_positions")
            api = _replay_api(virtual_cloud_inputs)
            expected_positions = pd.Series(local_positions, dtype=float) if isinstance(local_positions, Mapping) else local_positions
            checks.append(AlignmentCheckItem(stage, True))
            stage = "virtual_cloud_load"
            code = TARGET_ALIGNMENT_MISMATCH
            namespace = execute_bundle_candidate(candidate, STRATEGY_FILENAME, api)
            checks.append(AlignmentCheckItem(stage, True))
            stage = "virtual_cloud_replay"
            factors, _, positions = namespace["compute_bundle_signal"](
                list(virtual_cloud_inputs["stocks"]), virtual_cloud_inputs["asof"],
                virtual_cloud_inputs.get("pool_history"),
            )
            checks.append(AlignmentCheckItem(stage, True))
            stage = "reference_comparison"
            for name, observed, expected in (
                ("factor_values", factors, local_factor_values),
                ("target_positions", pd.Series(positions, dtype=float), expected_positions),
            ):
                check, diff, count = _compare(name, observed, expected)
                checks.append(check)
                metrics[f"{name}_count"] = count
                metrics[f"{name}_max_abs_diff"] = diff
                if diff is not None:
                    max_abs_diff = max(diff, max_abs_diff or 0.0)
    except Exception as exc:
        checks.append(AlignmentCheckItem(stage, False, str(exc), exc.code if isinstance(exc, DeployTargetError) else code))

    failure = next((check for check in checks if not check.passed), None)
    report = AlignmentReport(
        TARGET_ID, failure is None, bundle_hash, tuple(checks), tuple(warnings),
        metrics, max_abs_diff, failure.code if failure else "",
    )
    if write_report and root is not None:
        _write_report(report, root)
    return report
