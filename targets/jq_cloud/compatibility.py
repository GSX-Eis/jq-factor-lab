"""Resolve a verified bundle's inputs for the frozen JoinQuant compiler."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from factor_synthesis.strategy_bundle import StrategyBundle, StrategyFactorSpec
from scripts.alignment.virtual_cloud_check import _event_loader_check
from scripts.build_jq_strategy import (
    discover_factor_files,
    extract_factor_specs,
    load_cloud_api_mapping,
    load_factor_requirements,
)
from targets.contracts import (
    TARGET_AUX_FEATURE_UNSUPPORTED,
    TARGET_BUNDLE_INVALID,
    TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
    TARGET_SLOT_INCOMPATIBLE,
    CompatibilityIssue,
    CompatibilityReport,
    DeployTargetError,
    StrategyBundleSource,
    resolve_contained_file,
    resolve_verified_bundle_with_dir,
)
from utils.factor_runtime import JQ_BARS_PER_DAY as BARS_PER_DAY
from utils.time_axis import Frequency

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ID = "jq_cloud"
SLOT_ID = "jq_ashare_local"
# Explicitly supported by the frozen adapter's native industry loader.
CLOUD_EVENT_FIELDS = {"sw_l1_code", "sw_l1_name"}

BundleInput = StrategyBundle | StrategyBundleSource | str | Path
FactorFiles = Mapping[str, str | Path] | Sequence[str | Path]


@dataclass(frozen=True)
class CompilationInputs:
    specs: dict[str, dict[str, Any]]
    mapping: dict[str, Any]
    models: list[dict[str, Any]]
    factor_plans: dict[str, tuple[str, dict[str, Any], list[str]]]


def cloud_frequency(frequency: str | None) -> str | None:
    parsed = Frequency.parse(frequency)
    mapped = "daily" if parsed.canonical == "1d" else parsed.joinquant
    return mapped if mapped in BARS_PER_DAY else None


def mapped_field(name: str, frequency: str, mapping: dict[str, Any]) -> str | None:
    table, _, field = name.rpartition(".")
    contracts = mapping.get("field_contracts", {})
    industry = frequency == "daily" and field in CLOUD_EVENT_FIELDS
    logical_table = contracts.get(field, {}).get("local", {}).get("table")
    if table and table != logical_table and not (industry and table == "industry_membership"):
        matches = [
            key for key, spec in contracts.items()
            if spec.get("local", {}).get("table") == table
            and spec["local"].get("column") == field
        ]
        if field not in matches:
            if len(matches) != 1:
                return None
            field = matches[0]
    contract = contracts.get(field, {})
    if contract and frequency not in contract.get("frequency", ()):
        return None
    supported = (
        field in mapping.get("get_price", ())
        or field in mapping.get("get_valuation", {})
        or field in mapping.get("get_factor_values", ())
        or field in mapping.get("get_money_flow", {})
        or contract.get("cloud", {}).get("api") == "get_factor_values"
        or industry
    )
    return field if supported else None


def _factor_input_plan(
    factor: StrategyFactorSpec, spec: dict[str, Any], registry: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    name = spec["class_name"]
    entry = registry.get(name) or registry.get(factor.factor_id) or {}
    plans = entry.get("requirements") or {}
    # Preserve the shared runtime's input priority for daily output from minute bars.
    frequency = next((key for key in ("1m", "5m", "15m", "30m", "60m", "daily") if key in plans), None)
    output = cloud_frequency(factor.frequency)
    frequency = frequency or cloud_frequency(spec.get("frequency")) or output
    if frequency is None or (plans and frequency not in plans) or (output != "daily" and frequency != output):
        raise DeployTargetError(
            f"{factor.factor_id}: unsupported JoinQuant input {frequency!r} for output {output!r}",
            code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
        )
    return frequency, entry


def _candidate_specs(path: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Factor source file is missing: {path}")
    source = path.read_text(encoding="utf-8")
    if not any(re.search(r"^class\s+" + re.escape(name) + r"\b", source, re.M) for name in names):
        return []
    return [spec for spec in extract_factor_specs(str(path)) if spec["class_name"] in names]


def locate_factor_sources(
    bundle: StrategyBundle, factor_files: FactorFiles | None
) -> dict[str, dict[str, Any]]:
    """Match a factor ID or a pinned Python class without importing factor modules."""
    if isinstance(factor_files, (str, bytes, Path)):
        raise ValueError("factor_files must be a mapping or a sequence of paths")
    explicit = factor_files if isinstance(factor_files, Mapping) else {}
    paths = [] if isinstance(factor_files, Mapping) else [
        Path(path).expanduser() for path in (factor_files or ())
    ]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Factor source file is missing: {path}")
    discovered = None
    result = {}
    for factor in bundle.factors:
        module_parts = (factor.source_module or "").replace(":", ".").split(".")
        name_hint = module_parts[-1]
        names = tuple(dict.fromkeys((factor.factor_id, name_hint))) if name_hint else (factor.factor_id,)
        pinned = None
        if len(module_parts) > 1 and all(part.isidentifier() for part in module_parts):
            for parts in (module_parts, module_parts[:-1]):
                module_path = REPO_ROOT.joinpath(*parts).with_suffix(".py")
                if module_path.is_file():
                    pinned = module_path
                    break
            if pinned is None and factor.factor_id not in explicit and not paths:
                raise ValueError(f"{factor.factor_id}: source_module {factor.source_module!r} cannot be located")

        if factor.factor_id in explicit:
            search = [Path(explicit[factor.factor_id]).expanduser()]
        elif pinned is not None:
            search = [*paths, pinned]
        else:
            search = paths
        matches = [
            spec
            for path in dict.fromkeys(p.resolve() for p in search)
            for spec in _candidate_specs(path, names)
        ]
        if not matches and pinned is None and factor.factor_id not in explicit:
            if discovered is None:
                discovered = [Path(path) for path in discover_factor_files(str(REPO_ROOT / "factors"))]
            matches = [
                spec for path in discovered for spec in _candidate_specs(path, names)
            ]
            active = [spec for spec in matches if "archived" not in Path(spec["module_file"]).parts]
            matches = active or matches
        if len(matches) != 1:
            problem = "missing" if not matches else "ambiguous"
            raise ValueError(f"{factor.factor_id}: {problem} factor source for {names!r}")
        spec = matches[0]
        capability = _event_loader_check(Path(spec["module_file"]).read_text(encoding="utf-8"))
        if not capability["passed"]:
            raise DeployTargetError(
                f"{factor.factor_id}: {capability['reason']}",
                code=TARGET_AUX_FEATURE_UNSUPPORTED,
            )
        result[factor.factor_id] = spec
    class_sources = {}
    for spec in result.values():
        name, path = spec["class_name"], spec["module_file"]
        if class_sources.setdefault(name, path) != path:
            raise ValueError(f"Duplicate class {name!r} from different source files")
    return result


def load_replay_models(bundle: StrategyBundle, bundle_dir: Path) -> list[dict[str, Any]]:
    refs = [ref for ref in bundle.model_artifacts if ref.artifact_type == "model_bundle"]
    if len(refs) != 1:
        raise ValueError("weighted_sum requires one verified model_bundle")
    path = resolve_contained_file(bundle_dir, refs[0].path, "model_bundle")
    payload = json.loads(path.read_text(encoding="utf-8"))
    factors = {item.factor_id for item in bundle.factors}
    models = payload["index"]["models"]
    allowed_methods = {
        "none", "zscore", "rank", "rank_zscore", "winsorize_zscore", "raw_zscore", "gauss_rank", "binning"
    }
    dates = set()
    for model in models:
        train_end = model.get("train_end")
        if not isinstance(train_end, str) or date.fromisoformat(train_end).isoformat() != train_end:
            raise ValueError("weighted_sum models require an ISO train_end date")
        if train_end in dates:
            raise ValueError(f"Duplicate weighted_sum train_end {train_end}")
        dates.add(train_end)
        weights = model["weights"]
        missing = set(weights) - (factors & set(model["feature_names"]))
        if missing:
            raise ValueError(f"Model weights reference unavailable factor IDs: {sorted(missing)}")
        if model.get("parent_baseline") or model.get("blend_only") or model.get("blend_with_original_ratio", 0):
            raise ValueError("JoinQuant weighted_sum replay requires a standalone linear model")
        methods = model.get("standardization_method", {})
        if not isinstance(methods, dict):
            raise ValueError("standardization_method must be a mapping")
        for name in {*weights, "ALL"} & methods.keys():
            if methods[name] not in allowed_methods:
                raise ValueError(f"{name}: unsupported JoinQuant preprocessing {methods[name]!r}")
        percentile = methods.get("_winsorize_percentile", 0.025)
        if type(percentile) not in (float, int) or not 0 <= percentile < 0.5:
            raise ValueError("Invalid _winsorize_percentile")
        bins = methods.get("_binning_n_bins", 5)
        if type(bins) is not int or bins < 2:
            raise ValueError("Invalid _binning_n_bins")
        for flag in ("standardize", "postprocess_cs_standardize"):
            if flag in model and type(model[flag]) is not bool:
                raise ValueError(f"{flag} must be boolean")
    return models


def _execution_issue(bundle: StrategyBundle) -> str | None:
    if bundle.execution_profile.get("allow_short") is not False or bundle.allow_short:
        return "JoinQuant requires allow_short=False"
    if cloud_frequency(bundle.frequency) is None:
        return f"Unsupported JoinQuant frequency {bundle.frequency!r}"
    if bundle.synthesis_method != "weighted_sum":
        return f"Unsupported JoinQuant synthesis method {bundle.synthesis_method!r}"
    top_k = bundle.rebalance_config.get("top_k")
    top_pct = bundle.rebalance_config.get("top_pct")
    if top_k is not None and (type(top_k) is not int or top_k <= 0):
        return "top_k must be a positive integer"
    if top_pct is not None and (type(top_pct) not in (float, int) or not 0 < top_pct <= 1):
        return "top_pct must be a fraction in (0, 1]"
    mode = bundle.rebalance_config.get("selection_mode")
    if (mode not in (None, "top_k", "top_pct", "default")
        or (mode == "top_k" and top_k is None)
        or (mode == "top_pct" and (top_pct is None or top_k is not None))):
        return f"Unsupported JoinQuant selection contract {mode!r}"
    return None


def collect_compilation_inputs(
    bundle: StrategyBundle,
    verified_dir: Path | None,
    *,
    factor_files: FactorFiles | None,
    cloud_mapping_path: str | Path | None,
    requirements_path: str | Path | None,
) -> tuple[tuple[CompatibilityIssue, ...], CompilationInputs | None]:
    if bundle.slot_id != SLOT_ID:
        issue = CompatibilityIssue(TARGET_SLOT_INCOMPATIBLE, f"JoinQuant cannot compile slot {bundle.slot_id!r}", "slot_id")
        return (issue,), None
    if execution_issue := _execution_issue(bundle):
        return (CompatibilityIssue(TARGET_EXECUTION_CONTRACT_UNSUPPORTED, execution_issue),), None
    try:
        mapping = load_cloud_api_mapping(REPO_ROOT, cloud_mapping_path)
        neutralization = bundle.preprocessing_params.get("neutralization") or {}
        if neutralization.get("enabled"):
            raise DeployTargetError(
                "JoinQuant linear replay requires frozen factors without synthesis-time neutralization",
                code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
            )
        specs = locate_factor_sources(bundle, factor_files)
        registry = load_factor_requirements(REPO_ROOT, requirements_path)
        factor_plans = {}
        issues = []
        for factor in bundle.factors:
            if factor.frequency != bundle.frequency:
                raise DeployTargetError(
                    f"{factor.factor_id}: output {factor.frequency!r} must match bundle frequency {bundle.frequency!r}",
                    code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
                )
            spec = specs[factor.factor_id]
            frequency, entry = _factor_input_plan(factor, spec, registry)
            plan = entry.get("requirements", {}).get(frequency, {})
            required = set(plan.get("fields") or spec["dependencies"])
            required.update(factor.aux_features_required)
            required.difference_update(("date", "code", "stock_code"))
            mapped = {name: mapped_field(name, frequency, mapping) for name in required}
            missing = sorted(name for name, field in mapped.items() if field is None)
            if missing:
                issues.append(CompatibilityIssue(
                    TARGET_AUX_FEATURE_UNSUPPORTED,
                    f"{factor.factor_id}: unmapped JoinQuant input fields at {frequency}: {', '.join(missing)}",
                    "factors.aux_features_required",
                ))
            fields = sorted({field for field in mapped.values() if field is not None})
            factor_plans[factor.factor_id] = frequency, entry, fields
        if issues:
            return tuple(issues), None
        if sum(ref.artifact_type == "model_bundle" for ref in bundle.model_artifacts) != 1:
            raise ValueError("weighted_sum requires one model_bundle")
        models = [] if verified_dir is None else load_replay_models(bundle, verified_dir)
    except DeployTargetError as exc:
        return (CompatibilityIssue(exc.code, str(exc)),), None
    except (OSError, ValueError, SyntaxError, TypeError, KeyError, AttributeError) as exc:
        return (CompatibilityIssue(TARGET_BUNDLE_INVALID, str(exc)),), None
    return (), CompilationInputs(specs, mapping, models, factor_plans)


def check_bundle_compatibility(
    source: BundleInput,
    *,
    bundle_dir: str | Path | None = None,
    factor_files: FactorFiles | None = None,
    cloud_mapping_path: str | Path | None = None,
    requirements_path: str | Path | None = None,
) -> CompatibilityReport:
    try:
        bundle, verified_dir = resolve_verified_bundle_with_dir(
            source, bundle_dir=bundle_dir, require_verified_source=False,
        )
    except DeployTargetError as exc:
        return CompatibilityReport(TARGET_ID, False, "", "", (CompatibilityIssue(exc.code, str(exc)),))
    issues, _ = collect_compilation_inputs(
        bundle, verified_dir, factor_files=factor_files, cloud_mapping_path=cloud_mapping_path,
        requirements_path=requirements_path,
    )
    return CompatibilityReport(TARGET_ID, not issues, bundle.bundle_hash, bundle.slot_id, issues)
