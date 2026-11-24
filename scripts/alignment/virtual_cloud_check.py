#!/usr/bin/env python3
"""Fast local compatibility gate for a built JoinQuant factor strategy.

This is deliberately a compatibility emulator, not a replacement for a
real JoinQuant run.  It catches failures that are deterministic locally:
unsupported Python 3.6 syntax, missing requirements, leaked project imports,
and bundles which cannot even load with the small JQ module surface available
to generated strategies.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import math
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List


def _check(name: str, passed: bool, reason: str = "") -> Dict[str, Any]:
    return {"name": name, "passed": passed, "reason": reason}


def _parse_py36(source: str, label: str) -> Dict[str, Any]:
    if "from __future__ import annotations" in source:
        return _check(
            "python36_syntax", False, f"{label}: future annotations require Python 3.7+"
        )
    try:
        ast.parse(source, filename=label, feature_version=(3, 6))
    except SyntaxError as exc:
        return _check(
            "python36_syntax", False, f"{label}: {exc.msg} (line {exc.lineno})"
        )
    return _check("python36_syntax", True)


def _external_imports(source: str, label: str, *, strict: bool = False) -> Dict[str, Any]:
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as exc:
        return _check("bundle_imports", False, f"{label}: {exc.msg}")
    leaked: List[str] = []
    local_roots = ("factors", "utils", "config") if strict else ("factors",)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            optional_runtime_import = node.level and module in {
                "dynamic_stock_pool",
                "duckdb_adapter",
            }
            if optional_runtime_import:
                continue
            if (
                node.level
                or module in local_roots
                or module.startswith(("factors.", "utils.", "config."))
            ):
                leaked.append(
                    f"line {node.lineno}: from {'.' * node.level}{module} import ..."
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in local_roots or alias.name.startswith(
                    ("factors.", "utils.", "config.")
                ):
                    leaked.append(f"line {node.lineno}: import {alias.name}")
    return _check("bundle_imports", not leaked, "; ".join(leaked[:5]))


def _load_bundle(source: str, label: str) -> Dict[str, Any]:
    """Execute definitions only, with the minimal import surface JQ code expects."""
    jqdata = types.ModuleType("jqdata")
    jqfactor = types.ModuleType("jqfactor")
    jqdata.get_valuation = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    jqfactor.get_factor_values = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    previous = {name: sys.modules.get(name) for name in ("jqdata", "jqfactor")}
    sys.modules["jqdata"] = jqdata
    sys.modules["jqfactor"] = jqfactor
    try:
        namespace: Dict[str, Any] = {"__name__": "__alignment_virtual_cloud__"}
        exec(compile(source, label, "exec"), namespace, namespace)
    except Exception as exc:
        return _check("virtual_cloud_load", False, f"{type(exc).__name__}: {exc}")
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return _check("virtual_cloud_load", True)


def _load_requirement_payload(path: Path, factor_class: str) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        # [ALIGNMENT-FIX] Preflight must validate the disposable, target-only
        # dumper that real-cloud execution uploads, rather than its stale
        # shared registry.  Its generated assignment is JSON-shaped.
        whole_req = re.search(r"^REQ\s*=\s*(\{.*\})\s*$", text, re.MULTILINE)
        if whole_req:
            try:
                payload = ast.literal_eval(whole_req.group(1))
                return payload if isinstance(payload, dict) else {}
            except (SyntaxError, ValueError):
                pass
        pattern = r"ReQ_DO_NOT_MATCH"
        pattern = (
            r'REQ\["factors"\]\["' + re.escape(factor_class) + r'"\]\s*=\s*(\{.*\})\s*#'
        )
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            return {}
        try:
            return {"factors": {factor_class: json.loads(match.group(1))}}
        except json.JSONDecodeError:
            return {}


def _requirements_check(path: Path, factor_class: str) -> Dict[str, Any]:
    payload = _load_requirement_payload(path, factor_class)
    factors = payload.get("factors")
    return _check_requirements_payload(payload, factor_class)


def _check_requirements_payload(payload: Dict[str, Any], factor_class: str) -> Dict[str, Any]:
    factors = payload.get("factors")
    if not isinstance(factors, dict) or not isinstance(factors.get(factor_class), dict):
        return _check(
            "requirements", False, f"cloud_requirement_missing:{factor_class}"
        )
    requirements = factors[factor_class].get("requirements")
    if not isinstance(requirements, dict) or not requirements:
        return _check("requirements", False, f"cloud_requirement_empty:{factor_class}")
    return _check("requirements", True)


def _factor_class_check(source: str, factor_class: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _check("factor_class", False, exc.msg)
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if factor_class not in classes:
        return _check("factor_class", False, f"factor_class_missing:{factor_class}")
    return _check("factor_class", True)


def _event_loader_check(source: str) -> Dict[str, Any]:
    """Fail fast only on an explicit unsupported-cloud declaration.

    SQL text alone is not enough: several factors have a valid cloud branch.
    Explicit metadata such as ``requires injected event_loader`` is a stable
    declaration that the current generated strategy cannot satisfy.
    """
    limitation = re.search(
        r"cloud_(?:limitation|event_loader).{0,240}(?:requires?\s+injected|no\s+DuckDB|local data)",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if limitation:
        return _check("cloud_event_capability", False, "cloud_event_loader_required")
    return _check("cloud_event_capability", True)


def _legacy_pandas_check(source: str) -> Dict[str, Any]:
    if re.search(r"(?:pd|pandas)\.MultiIndex\.from_frame\s*\(", source):
        return _check(
            "legacy_pandas_api", False, "jq_pandas_multiindex_from_frame_unsupported"
        )
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _check("legacy_pandas_api", True)
    multi_column_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, (ast.List, ast.Tuple, ast.ListComp))
        and (isinstance(node.value, ast.ListComp) or len(node.value.elts) > 1)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        target = node.targets[0] if len(node.targets) == 1 else None
        selected = target.slice if isinstance(target, ast.Subscript) else None
        multi_column_target = (
            isinstance(selected, (ast.List, ast.Tuple)) and len(selected.elts) > 1
        ) or (isinstance(selected, ast.Name) and selected.id in multi_column_names)
        if not multi_column_target:
            continue
        if (
            isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "ffill"
            and any(
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Attribute)
                and part.func.attr == "groupby"
                for part in ast.walk(node.value)
            )
        ):
            return _check(
                "legacy_pandas_api",
                False,
                "jq_pandas_groupby_multicolumn_ffill_unsupported",
            )
    return _check("legacy_pandas_api", True)


def _jq_runtime_contract(
    path: Path, factor_class: str
) -> tuple[Dict[str, Any], List[str]]:
    return _check_runtime_payload(_load_requirement_payload(path, factor_class), factor_class)


def _check_runtime_payload(
    payload: Dict[str, Any], factor_class: str, *, use_runtime_window: bool = False,
    include_session_prefix: bool = False,
) -> tuple[Dict[str, Any], List[str]]:
    factor = (payload.get("factors") or {}).get(factor_class) or {}
    requirements = factor.get("requirements") or {}
    # [ALIGNMENT-FIX] The generated JQ adapter derives ``paused`` from
    # ``volume`` for historical bars, so it is not an unsupported raw request.
    unsupported_history_fields = {"avg", "high_limit", "low_limit", "pre_close"}
    failures: List[str] = []
    warnings: List[str] = []
    for frequency, config in requirements.items():
        if not isinstance(config, dict):
            continue
        raw_window = config.get("min_bars", config.get("min_days", 1))
        try:
            planned_window = math.ceil(
                float(raw_window) * (1 + float(config.get("slack_ratio", 0)))
            )
            if use_runtime_window:
                planned_window = factor.get("runtime", {}).get("window", planned_window)
        except (TypeError, ValueError):
            continue
        invalid = sorted(set(config.get("fields") or ()) & unsupported_history_fields)
        if planned_window > 1 and invalid:
            failures.append(
                "jq_get_price_history_fields_unsupported:"
                + str(frequency)
                + ":"
                + ",".join(invalid)
            )
        if frequency != "daily":
            requested_window = planned_window
            if use_runtime_window:
                # Import only for the bundle preflight. The standalone legacy CLI
                # also runs directly from scripts/alignment without the repo on sys.path.
                from utils.factor_runtime import (
                    JQ_BARS_PER_DAY, MAX_JQ_MINUTE_REQUEST_BARS, jq_minute_request_count,
                )

                requested_window = jq_minute_request_count(
                    planned_window, JQ_BARS_PER_DAY[frequency],
                    include_session_prefix=include_session_prefix,
                )
                limit = MAX_JQ_MINUTE_REQUEST_BARS
            else:
                limit = 5000
            if requested_window > limit:
                warnings.append(f"cloud_runtime_budget_risk:{frequency}:bars={requested_window}")
    return _check("jq_api_contract", not failures, "; ".join(failures)), warnings


def inspect_candidate_source(source: str, label: str) -> list[Dict[str, Any]]:
    return [
        _parse_py36(source, label),
        _external_imports(source, label, strict=True),
    ]


def inspect_bundle_candidate(
    source: str, label: str, factor_classes: list[str], requirements: Dict[str, Any],
    *, strategy_frequency: str = "1d",
) -> tuple[list[Dict[str, Any]], list[str]]:
    """Run the static compatibility gates before executing a multi-factor candidate."""
    checks = [
        *inspect_candidate_source(source, label),
        _legacy_pandas_check(source),
        _event_loader_check(source),
    ]
    warnings = []
    for name in factor_classes:
        api_check, risks = _check_runtime_payload(
            requirements, name, use_runtime_window=True,
            include_session_prefix=strategy_frequency != "1d",
        )
        for check in (
            _factor_class_check(source, name),
            _check_requirements_payload(requirements, name),
            api_check,
            _check("cloud_runtime_budget", not risks, "; ".join(risks)),
        ):
            checks.append(dict(check, name=check["name"] + ":" + name))
        warnings.extend(f"{name}:{risk}" for risk in risks)
    return checks, warnings


def execute_bundle_candidate(
    source: str, label: str, api: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Load a generated candidate with per-execution virtual JoinQuant modules."""
    api = dict(api or {})
    modules = {}
    for name in ("jqdata", "jqfactor"):
        module = types.ModuleType(name)
        vars(module).update(api)
        module.__all__ = list(api)
        modules[name] = module

    def virtual_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.partition(".")[0] in modules:
            if name in modules:
                return modules[name]
            raise ImportError(f"Virtual cloud module unavailable: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    namespace = dict(api, __name__="__alignment_virtual_cloud__")
    namespace["__builtins__"] = dict(vars(builtins), __import__=virtual_import)
    exec(compile(source, label, "exec"), namespace, namespace)
    return namespace


def run_check(
    *, factor: Path, factor_class: str, bundle: Path, requirements: Path
) -> Dict[str, Any]:
    source = factor.read_text(encoding="utf-8")
    jq_contract, warnings = _jq_runtime_contract(requirements, factor_class)
    checks = [
        _parse_py36(source, str(factor)),
        _factor_class_check(source, factor_class),
        _requirements_check(requirements, factor_class),
        _event_loader_check(source),
        jq_contract,
    ]
    if bundle.exists():
        bundled = bundle.read_text(encoding="utf-8")
        checks.extend(
            [
                _parse_py36(bundled, str(bundle)),
                _external_imports(bundled, str(bundle)),
                _legacy_pandas_check(bundled),
                _load_bundle(bundled, str(bundle)),
            ]
        )
    else:
        checks.append(_check("bundle_exists", False, f"bundle_missing:{bundle}"))
    failures = [item["reason"] for item in checks if not item["passed"]]
    return {
        "status": "PASS" if not failures else "FAIL",
        "failed_stage": "virtual_cloud" if failures else "",
        "error_code": failures[0] if failures else "",
        "repairable": bool(failures),
        "checks": checks,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local virtual JoinQuant compatibility gate"
    )
    parser.add_argument("--factor", required=True)
    parser.add_argument("--factor-class", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--requirements", default="config/factor_requirements.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_check(
        factor=Path(args.factor),
        factor_class=args.factor_class,
        bundle=Path(args.bundle),
        requirements=Path(args.requirements),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
