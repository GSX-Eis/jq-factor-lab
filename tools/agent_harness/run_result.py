"""Producer/consumer contract for one research subprocess invocation."""

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

SCHEMA_VERSION = "execution_result.v1"
DEFAULT_TOP_K = 5


def resolve_candidate_selection(*, top_pct=None, top_k=None, default_top_k=DEFAULT_TOP_K):
    """Resolve canonical (selection_mode, top_pct, top_k) with priority top_k > top_pct > default_top_k."""
    if top_k is not None:
        return "top_k", None, top_k
    if top_pct is not None:
        return "top_pct", top_pct, None
    return "top_k", None, default_top_k


def resolve_backtest_candidates(top_k_cfg, *, default_top_k=DEFAULT_TOP_K):
    """Resolve candidate selection tuples before expansion using fixed-count priority."""
    cfg = top_k_cfg if isinstance(top_k_cfg, Mapping) else {}
    top_k_raw = cfg.get("top_k")
    if top_k_raw is not None:
        return [
            resolve_candidate_selection(
                top_pct=None, top_k=top_k_raw, default_top_k=default_top_k
            )
        ]
    top_pct_raw = cfg.get("top_pct", None)
    if top_pct_raw is None:
        return [
            resolve_candidate_selection(
                top_pct=None, top_k=None, default_top_k=default_top_k
            )
        ]
    if isinstance(top_pct_raw, list):
        return [
            resolve_candidate_selection(
                top_pct=pct, top_k=None, default_top_k=default_top_k
            )
            for pct in top_pct_raw
        ]
    return [
        resolve_candidate_selection(
            top_pct=top_pct_raw, top_k=None, default_top_k=default_top_k
        )
    ]


def config_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def _json_values(value):
    if isinstance(value, dict):
        return {str(key): _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_values(value.item())
    return value


_HEX_CHARS = frozenset("0123456789abcdef")


def build_backtest_trial_id(
    *,
    config_path=None,
    factor_name,
    selection_mode,
    top_pct,
    top_k,
    synthesis_name=None,
):
    """Build deterministic backtest trial_id shared by Synthesizer and Harness."""
    _ = config_path  # Kept in signature for caller compatibility; identity depends on config fields.
    payload = {
        "factor_name": factor_name,
        "selection_mode": selection_mode,
        "top_pct": top_pct,
        "top_k": top_k,
    }
    if synthesis_name:
        payload["synthesis_name"] = synthesis_name
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    selector = top_pct if selection_mode == "top_pct" else top_k
    raw = f"{factor_name}_{selection_mode}_{selector}_{digest}"
    safe = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw
    )
    return safe.strip("._") or f"trial_{digest}"


def expected_synthesis_items(config_payload: Mapping) -> dict[tuple[str, str], str]:
    """Map each configured (method name, output factor name) to its algorithm."""
    methods = config_payload.get("synthesis_methods") or {}
    if not isinstance(methods, Mapping):
        raise ValueError("synthesis_methods must be a mapping")
    prefix = str(
        config_payload.get("output_settings", {}).get("composite_factor_prefix") or ""
    ).strip()
    expected = {}
    for name, settings in methods.items():
        if not isinstance(settings, Mapping):
            raise ValueError(f"invalid synthesis configuration for {name!r}")
        method = str(settings.get("method") or "lasso").strip().lower()
        base = f"{prefix}_{name}" if prefix else name
        if method == "pca":
            count = settings.get("n_components", 2)
            if type(count) is not int or count < 1:
                raise ValueError(f"invalid PCA component count for {name!r}")
            for index in range(1, count + 1):
                expected[name, f"{base}_PC{index}"] = method
        else:
            expected[name, base] = method
    return expected


def expected_backtest_trials(config_payload, *, config_path=None):
    """Return expected VectorBT backtest candidates for a synthesis config."""
    if not isinstance(config_payload, Mapping):
        return []
    top_k_cfg = (
        config_payload.get("performance_evaluation", {}).get("top_k_strategy", {})
    )
    if not isinstance(top_k_cfg, Mapping) or not top_k_cfg.get("enabled", False):
        return []

    candidates = resolve_backtest_candidates(top_k_cfg)
    return [
        {
            "synthesis_name": name,
            "method": method,
            "factor_name": factor_name,
            "selection_mode": mode,
            "top_pct": pct,
            "top_k": top_k,
            "trial_id": build_backtest_trial_id(
                config_path=config_path,
                factor_name=factor_name,
                selection_mode=mode,
                top_pct=pct,
                top_k=top_k,
                synthesis_name=name,
            ),
        }
        for (name, factor_name), method in expected_synthesis_items(config_payload).items()
        for mode, pct, top_k in candidates
    ]


def _selection_pct_matches(actual, expected) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return False
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    actual_f = float(actual)
    expected_f = float(expected)
    return (
        math.isfinite(actual_f)
        and math.isfinite(expected_f)
        and actual_f == expected_f
    )


def _selection_top_k_matches(actual, expected) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool) or not isinstance(expected, int):
        return False
    if isinstance(actual, bool) or not isinstance(actual, int):
        return False
    return actual == expected


def _validate_trial_selection_fields(
    mapping,
    *,
    expected,
    context,
    require_fields=True,
    check_synthesis_name=True,
):
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{context} must be a mapping")
    if check_synthesis_name and (
        require_fields
        or "synthesis_name" in mapping
        or mapping.get("synthesis_name") is not None
    ):
        sname = mapping.get("synthesis_name")
        if (
            not isinstance(sname, str)
            or not sname.strip()
            or sname.strip() != expected["synthesis_name"]
        ):
            raise ValueError(
                f"{context} synthesis_name mismatch: expected {expected['synthesis_name']!r}, got {sname!r}"
            )
    if (
        require_fields
        or "selection_mode" in mapping
        or mapping.get("selection_mode") is not None
    ):
        mode = mapping.get("selection_mode")
        if (
            not isinstance(mode, str)
            or not mode.strip()
            or mode.strip() != expected["selection_mode"]
        ):
            raise ValueError(
                f"{context} selection_mode mismatch: expected {expected['selection_mode']!r}, got {mode!r}"
            )
    if require_fields or "top_pct" in mapping:
        if "top_pct" not in mapping or not _selection_pct_matches(
            mapping.get("top_pct"), expected["top_pct"]
        ):
            raise ValueError(
                f"{context} top_pct mismatch: expected {expected['top_pct']!r}, got {mapping.get('top_pct')!r}"
            )
    if require_fields or "top_k" in mapping:
        if "top_k" not in mapping or not _selection_top_k_matches(
            mapping.get("top_k"), expected["top_k"]
        ):
            raise ValueError(
                f"{context} top_k mismatch: expected {expected['top_k']!r}, got {mapping.get('top_k')!r}"
            )


def _is_sha256_hex(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX_CHARS for ch in value)
    )


def validate_execution_result_payload(
    payload,
    *,
    expected_identity=None,
    require_succeeded=False,
    require_identity=True,
):
    """Validate execution_result.v1 schema, run identity, and execution counters."""
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("execution result identity mismatch")
    if require_identity:
        for key in ("command", "run_id", "execution_id"):
            val = payload.get(key)
            if not isinstance(val, str) or not val.strip() or val != val.strip():
                raise ValueError(f"execution result identity mismatch: invalid {key}")
        if not _is_sha256_hex(payload.get("config_sha256")):
            raise ValueError("execution result identity mismatch: invalid config_sha256")
    if expected_identity is not None and any(
        payload.get(k) != v for k, v in expected_identity.items()
    ):
        raise ValueError("execution result identity mismatch")
    execution = payload.get("execution", {})
    if (
        not isinstance(execution, dict)
        or any(
            type(execution.get(k)) is not int or execution[k] < 0
            for k in ("expected", "completed", "failed")
        )
        or not isinstance(execution.get("errors"), list)
        or execution.get("status") not in ("succeeded", "failed")
        or not isinstance(payload.get("metrics"), dict)
        or not isinstance(payload.get("artifacts"), list)
        or execution["completed"] + execution["failed"] > execution["expected"]
    ):
        raise ValueError("invalid execution result shape")
    if require_succeeded and execution["status"] != "succeeded":
        raise ValueError("execution result status is not succeeded")
    if execution["status"] == "succeeded" and (
        execution["expected"] < 1
        or execution["expected"] != execution["completed"]
        or execution["failed"]
        or execution["errors"]
    ):
        raise ValueError("execution result claims success with incomplete work")
    causal_gate = payload["metrics"].get("causal_gate")
    if causal_gate is not None:
        from tools.agent_harness.causal_gate import validate_causal_gate_evidence

        validate_causal_gate_evidence(causal_gate)
    return payload


@contextmanager
def execution_result(path, *, command, run_id, execution_id, config_path):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "run_id": run_id,
        "execution_id": execution_id,
        "config_sha256": config_sha256(config_path),
        "execution": {"expected": 0, "completed": 0, "failed": 0, "errors": []},
        "metrics": {},
        "artifacts": [],
    }
    try:
        yield payload
        execution = payload["execution"]
        if (execution["expected"] < 1 or execution["completed"] != execution["expected"]
                or execution["failed"] or execution["errors"]):
            raise RuntimeError("execution incomplete")
        if config_sha256(config_path) != payload["config_sha256"]:
            raise RuntimeError("config changed during execution")
        execution["status"] = "succeeded"
    except BaseException as exc:
        payload["execution"]["status"] = "failed"
        payload["execution"]["errors"].append({"item": command, "error": str(exc)})
        raise
    finally:
        if path:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            exec_dir = destination.resolve().parent
            # Snapshot only values produced in this invocation; never read shared old files.
            for name, value in (
                ("benchmark_reports", payload["metrics"].get("benchmark_reports")),
                ("algorithm_model_evidence", payload["metrics"].get("model_evidence")),
            ):
                if value:
                    artifact = destination.parent / f"{name}.json"
                    if name == "algorithm_model_evidence":
                        value = {"version": "1.0", "syntheses": value}
                    artifact.write_text(json.dumps(_json_values(value), ensure_ascii=False,
                                                   allow_nan=False, default=str), encoding="utf-8")
                    payload["artifacts"].append({"artifact_type": name, "path": str(artifact.resolve()),
                                                 "sha256": config_sha256(artifact)})
            bundle_candidates = []
            if "strategy_bundles" in payload["metrics"] and payload["metrics"]["strategy_bundles"] is not None:
                raw_bundles = payload["metrics"]["strategy_bundles"]
                if not isinstance(raw_bundles, list) or not raw_bundles:
                    raise ValueError("strategy_bundles metric must be a non-empty list")
                bundle_candidates.extend(raw_bundles)
            if "strategy_bundle" in payload["metrics"] and payload["metrics"]["strategy_bundle"] is not None:
                primary_bundle = payload["metrics"]["strategy_bundle"]
                if not isinstance(primary_bundle, dict):
                    raise ValueError("strategy_bundle metric must be a mapping")
                bundle_candidates.append(primary_bundle)
            if payload["execution"].get("status") == "succeeded":
                seen_bundle_dirs = {}
                for sb in bundle_candidates:
                    if not (
                        isinstance(sb, dict)
                        and isinstance(sb.get("manifest_path"), str)
                        and sb["manifest_path"].strip()
                        and isinstance(sb.get("bundle_hash"), str)
                        and sb["bundle_hash"].strip()
                        and isinstance(sb.get("bundle_dir"), str)
                        and sb["bundle_dir"].strip()
                    ):
                        raise ValueError("strategy_bundle descriptor missing manifest_path, bundle_dir, or bundle_hash")
                    manifest_path = Path(str(sb["manifest_path"])).resolve()
                    bundle_dir = Path(str(sb["bundle_dir"])).resolve()
                    if not manifest_path.is_relative_to(exec_dir) or not manifest_path.is_file():
                        raise ValueError("strategy_bundle manifest must reside inside execution directory")
                    if not bundle_dir.is_relative_to(exec_dir) or bundle_dir == exec_dir or not bundle_dir.is_dir():
                        raise ValueError("strategy_bundle directory must reside inside execution directory")
                    key = str(bundle_dir)
                    pct_key = (
                        float(sb.get("top_pct"))
                        if isinstance(sb.get("top_pct"), (int, float))
                        and not isinstance(sb.get("top_pct"), bool)
                        and math.isfinite(float(sb.get("top_pct")))
                        else repr(sb.get("top_pct"))
                    )
                    descriptor_key = (
                        str(manifest_path),
                        str(sb["bundle_hash"]),
                        str(sb.get("synthesis_name") or "").strip(),
                        str(sb.get("trial_id") or "").strip(),
                        str(sb.get("selection_mode") or "").strip(),
                        pct_key,
                        repr(sb.get("top_k")),
                    )
                    if key in seen_bundle_dirs:
                        if seen_bundle_dirs[key] != descriptor_key:
                            raise ValueError("conflicting strategy_bundle descriptors for same bundle_dir")
                        continue
                    seen_bundle_dirs[key] = descriptor_key
                    entry = {
                        "artifact_type": "strategy_bundle",
                        "path": str(manifest_path),
                        "sha256": config_sha256(manifest_path),
                        "bundle_dir": key,
                        "bundle_hash": str(sb["bundle_hash"]),
                        "schema_version": str(sb.get("schema_version", "strategy_bundle.v1")),
                    }
                    if sb.get("strategy_id"):
                        entry["strategy_id"] = str(sb["strategy_id"])
                    if sb.get("trial_id"):
                        entry["trial_id"] = str(sb["trial_id"])
                    if sb.get("synthesis_name"):
                        entry["synthesis_name"] = str(sb["synthesis_name"])
                    if sb.get("selection_mode"):
                        entry["selection_mode"] = str(sb["selection_mode"])
                    if "top_pct" in sb:
                        entry["top_pct"] = sb["top_pct"]
                    if "top_k" in sb:
                        entry["top_k"] = sb["top_k"]
                    payload["artifacts"].append(entry)
            encoded = json.dumps(_json_values(payload), ensure_ascii=False, indent=2,
                                 allow_nan=False, default=str) + "\n"
            staged = NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent,
                prefix=".execution-result-", suffix=".tmp", delete=False,
            )
            temporary = Path(staged.name)
            try:
                with staged:
                    staged.write(encoded)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)


def read_execution_result(path, *, command, run_id, execution_id, source_sha256):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION, "command": command, "run_id": run_id,
        "execution_id": execution_id, "config_sha256": source_sha256,
    }
    validate_execution_result_payload(
        payload, expected_identity=expected, require_identity=False
    )
    exec_dir = Path(path).resolve().parent
    verified_bundles_by_dir = {}
    verified_bundles_by_trial_id = {}
    for artifact in payload["artifacts"]:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not artifact["path"].strip()
        ):
            raise ValueError("execution artifact missing or hash-mismatched")
        artifact_path = Path(artifact["path"]).resolve()
        if (
            not artifact_path.is_relative_to(exec_dir)
            or artifact_path == exec_dir
            or not artifact_path.is_file()
            or config_sha256(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError("execution artifact missing or hash-mismatched")
        if artifact.get("artifact_type") == "strategy_bundle":
            from factor_synthesis.strategy_bundle import verify_strategy_bundle

            raw_bundle_dir = artifact.get("bundle_dir")
            if not isinstance(raw_bundle_dir, str) or not raw_bundle_dir.strip():
                raise ValueError("strategy_bundle artifact missing bundle_dir")
            bundle_dir = Path(raw_bundle_dir).resolve()
            if not bundle_dir.is_relative_to(exec_dir) or bundle_dir == exec_dir:
                raise ValueError("strategy_bundle directory escapes execution directory")
            verified = verify_strategy_bundle(bundle_dir)
            if verified.bundle_hash != artifact.get("bundle_hash"):
                raise ValueError("strategy_bundle bundle_hash mismatch")
            if artifact.get("schema_version") != verified.schema_version:
                raise ValueError("strategy_bundle schema_version mismatch")
            inner_manifest = (bundle_dir / "manifest.json").resolve()
            if config_sha256(artifact_path) != config_sha256(inner_manifest):
                raise ValueError("strategy_bundle registered manifest differs from verified manifest")
            embedded_exec = json.loads(
                (bundle_dir / verified.execution_result_ref).read_text(encoding="utf-8")
            )
            if any(
                embedded_exec.get(k) != payload.get(k)
                for k in ("schema_version", "command", "run_id", "execution_id", "config_sha256")
            ):
                raise ValueError("strategy_bundle embedded execution_result identity mismatch")
            if str(bundle_dir) in verified_bundles_by_dir:
                raise ValueError("duplicate strategy_bundle artifact for bundle_dir")
            rebal = verified.rebalance_config
            manifest_trial_id = str(rebal.get("trial_id") or "").strip()
            manifest_synthesis_name = str(rebal.get("synthesis_name") or "").strip()
            manifest_selection_mode = str(rebal.get("selection_mode") or "").strip()
            has_manifest_top_pct = "top_pct" in rebal
            manifest_top_pct = rebal.get("top_pct")
            has_manifest_top_k = "top_k" in rebal
            manifest_top_k = rebal.get("top_k")
            if artifact.get("trial_id") is not None:
                art_trial_id = str(artifact.get("trial_id") or "").strip()
                if not art_trial_id or art_trial_id != manifest_trial_id:
                    raise ValueError(
                        "strategy_bundle artifact trial_id mismatch with manifest rebalance_config.trial_id"
                    )
            if artifact.get("synthesis_name") is not None:
                art_sname = str(artifact.get("synthesis_name") or "").strip()
                if not art_sname or art_sname != manifest_synthesis_name:
                    raise ValueError(
                        "strategy_bundle artifact synthesis_name mismatch with manifest rebalance_config.synthesis_name"
                    )
            if artifact.get("selection_mode") is not None:
                art_mode = str(artifact.get("selection_mode") or "").strip()
                if not art_mode or art_mode != manifest_selection_mode:
                    raise ValueError(
                        "strategy_bundle artifact selection_mode mismatch with manifest rebalance_config.selection_mode"
                    )
            if "top_pct" in artifact:
                if has_manifest_top_pct:
                    if not _selection_pct_matches(artifact.get("top_pct"), manifest_top_pct):
                        raise ValueError(
                            "strategy_bundle artifact top_pct mismatch with manifest rebalance_config.top_pct"
                        )
                elif artifact.get("top_pct") is not None:
                    raise ValueError(
                        "strategy_bundle artifact top_pct mismatch with manifest rebalance_config.top_pct"
                    )
            if "top_k" in artifact:
                if has_manifest_top_k:
                    if not _selection_top_k_matches(artifact.get("top_k"), manifest_top_k):
                        raise ValueError(
                            "strategy_bundle artifact top_k mismatch with manifest rebalance_config.top_k"
                        )
                elif artifact.get("top_k") is not None:
                    raise ValueError(
                        "strategy_bundle artifact top_k mismatch with manifest rebalance_config.top_k"
                    )
            if manifest_trial_id:
                if manifest_trial_id in verified_bundles_by_trial_id:
                    raise ValueError(
                        f"duplicate strategy_bundle artifact for trial_id {manifest_trial_id!r}"
                    )
                verified_bundles_by_trial_id[manifest_trial_id] = str(bundle_dir)
            verified_bundles_by_dir[str(bundle_dir)] = {
                "bundle_dir": str(bundle_dir),
                "manifest_path": str(artifact_path),
                "manifest_sha256": config_sha256(artifact_path),
                "bundle_hash": verified.bundle_hash,
                "schema_version": verified.schema_version,
                "trial_id": manifest_trial_id,
                "synthesis_name": manifest_synthesis_name,
                "selection_mode": manifest_selection_mode,
                "has_top_pct": has_manifest_top_pct,
                "top_pct": manifest_top_pct,
                "has_top_k": has_manifest_top_k,
                "top_k": manifest_top_k,
                "strategy_id": verified.strategy_id,
            }

    if payload["execution"]["status"] == "succeeded":
        expected_bundles = []
        has_list = (
            "strategy_bundles" in payload["metrics"]
            and payload["metrics"]["strategy_bundles"] is not None
        )
        raw_bundles = []
        if has_list:
            raw_bundles = payload["metrics"]["strategy_bundles"]
            if not isinstance(raw_bundles, list) or not raw_bundles:
                raise ValueError("strategy_bundles metric must be a non-empty list")
            expected_bundles.extend(raw_bundles)
        primary_sb = None
        if (
            "strategy_bundle" in payload["metrics"]
            and payload["metrics"]["strategy_bundle"] is not None
        ):
            primary_sb = payload["metrics"]["strategy_bundle"]
            if not isinstance(primary_sb, dict):
                raise ValueError("strategy_bundle metric must be a mapping")
            expected_bundles.append(primary_sb)

        for sb in expected_bundles:
            if (
                not isinstance(sb, dict)
                or not isinstance(sb.get("bundle_dir"), str)
                or not sb["bundle_dir"].strip()
                or not isinstance(sb.get("manifest_path"), str)
                or not sb["manifest_path"].strip()
                or not isinstance(sb.get("bundle_hash"), str)
                or not sb["bundle_hash"].strip()
            ):
                raise ValueError("strategy_bundle metric entry is incomplete")
            bdir = str(Path(str(sb["bundle_dir"])).resolve())
            mpath = Path(str(sb["manifest_path"])).resolve()
            if not mpath.is_relative_to(exec_dir) or not mpath.is_file():
                raise ValueError("strategy_bundle metric manifest_path missing or escapes execution directory")
            reg = verified_bundles_by_dir.get(bdir)
            if reg is None or reg["bundle_hash"] != sb["bundle_hash"]:
                raise ValueError("strategy_bundle metric does not match verified artifact")
            if str(mpath) != reg["manifest_path"] or config_sha256(mpath) != reg["manifest_sha256"]:
                raise ValueError("strategy_bundle metric manifest_path does not match verified artifact")
            if sb.get("schema_version") is not None and sb.get("schema_version") != reg["schema_version"]:
                raise ValueError("strategy_bundle metric schema_version mismatch")
            sb_trial_id = str(sb.get("trial_id") or "").strip()
            if sb.get("trial_id") is not None and not sb_trial_id:
                raise ValueError("strategy_bundle metric has empty trial_id")
            if (sb_trial_id or reg["trial_id"]) and sb_trial_id != reg["trial_id"]:
                raise ValueError(
                    "strategy_bundle metric trial_id does not match verified manifest rebalance_config.trial_id"
                )
            sb_sname = str(sb.get("synthesis_name") or "").strip()
            if sb.get("synthesis_name") is not None and not sb_sname:
                raise ValueError("strategy_bundle metric has empty synthesis_name")
            if (sb_sname or reg["synthesis_name"]) and sb_sname != reg["synthesis_name"]:
                raise ValueError(
                    "strategy_bundle metric synthesis_name does not match verified manifest"
                )
            if "selection_mode" in sb and sb.get("selection_mode") is not None:
                sb_mode = sb.get("selection_mode")
                if not isinstance(sb_mode, str) or not sb_mode.strip():
                    raise ValueError("strategy_bundle metric has empty selection_mode")
                if sb_mode.strip() != reg["selection_mode"]:
                    raise ValueError(
                        "strategy_bundle metric selection_mode does not match verified manifest"
                    )
            elif reg["selection_mode"]:
                raise ValueError(
                    "strategy_bundle metric missing selection_mode present in verified manifest"
                )
            if "top_pct" in sb:
                if reg["has_top_pct"]:
                    if not _selection_pct_matches(sb.get("top_pct"), reg["top_pct"]):
                        raise ValueError(
                            "strategy_bundle metric top_pct does not match verified manifest"
                        )
                elif sb.get("top_pct") is not None:
                    raise ValueError(
                        "strategy_bundle metric top_pct does not match verified manifest"
                    )
            elif reg["has_top_pct"] and reg["top_pct"] is not None:
                raise ValueError(
                    "strategy_bundle metric missing top_pct present in verified manifest"
                )
            if "top_k" in sb:
                if reg["has_top_k"]:
                    if not _selection_top_k_matches(sb.get("top_k"), reg["top_k"]):
                        raise ValueError(
                            "strategy_bundle metric top_k does not match verified manifest"
                        )
                elif sb.get("top_k") is not None:
                    raise ValueError(
                        "strategy_bundle metric top_k does not match verified manifest"
                    )
            elif reg["has_top_k"] and reg["top_k"] is not None and reg["selection_mode"] == "top_k":
                raise ValueError(
                    "strategy_bundle metric missing top_k present in verified manifest"
                )

        if has_list:
            seen_list_dirs = set()
            seen_list_trials = set()
            for sb in raw_bundles:
                bdir = str(Path(str(sb["bundle_dir"])).resolve())
                if bdir in seen_list_dirs:
                    raise ValueError(f"duplicate bundle_dir in strategy_bundles: {bdir!r}")
                seen_list_dirs.add(bdir)
                sb_trial_id = str(sb.get("trial_id") or "").strip()
                if sb_trial_id:
                    if sb_trial_id in seen_list_trials:
                        raise ValueError(f"duplicate trial_id in strategy_bundles: {sb_trial_id!r}")
                    seen_list_trials.add(sb_trial_id)
            if len(seen_list_dirs) != len(verified_bundles_by_dir):
                raise ValueError("strategy_bundles list does not match registered strategy_bundle artifacts")

        if has_list and primary_sb is not None:
            def _bundle_descriptor_key(sb):
                raw_pct = sb.get("top_pct")
                norm_pct = None if raw_pct is None else float(raw_pct)
                return (
                    str(Path(str(sb["bundle_dir"])).resolve()),
                    str(Path(str(sb["manifest_path"])).resolve()),
                    str(sb["bundle_hash"]),
                    str(sb.get("trial_id") or "").strip(),
                    str(sb.get("synthesis_name") or "").strip(),
                    str(sb.get("selection_mode") or "").strip(),
                    norm_pct,
                    sb.get("top_k"),
                )

            listed_keys = {_bundle_descriptor_key(sb) for sb in raw_bundles}
            primary_key = _bundle_descriptor_key(primary_sb)
            if primary_key not in listed_keys:
                raise ValueError(
                    "primary strategy_bundle metric does not match strategy_bundles candidate list"
                )
    return payload
