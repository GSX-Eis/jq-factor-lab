"""Producer/consumer contract for one research subprocess invocation."""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = "execution_result.v1"


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
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(_json_values(payload), ensure_ascii=False, indent=2,
                           allow_nan=False, default=str) + "\n", encoding="utf-8"
            )
            temporary.replace(destination)


def read_execution_result(path, *, command, run_id, execution_id, source_sha256):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": SCHEMA_VERSION, "command": command, "run_id": run_id,
        "execution_id": execution_id, "config_sha256": source_sha256,
    }
    if not isinstance(payload, dict) or any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("execution result identity mismatch")
    execution = payload.get("execution", {})
    if (not isinstance(execution, dict)
            or any(type(execution.get(k)) is not int or execution[k] < 0
                   for k in ("expected", "completed", "failed"))
            or not isinstance(execution.get("errors"), list)
            or execution.get("status") not in ("succeeded", "failed")
            or not isinstance(payload.get("metrics"), dict)
            or not isinstance(payload.get("artifacts"), list)
            or execution["completed"] + execution["failed"] > execution["expected"]):
        raise ValueError("invalid execution result shape")
    if execution["status"] == "succeeded" and (
        execution["expected"] < 1 or execution["expected"] != execution["completed"]
        or execution["failed"] or execution["errors"]
    ):
        raise ValueError("execution result claims success with incomplete work")
    for artifact in payload["artifacts"]:
        if (not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str)
                or Path(artifact["path"]).resolve().parent != Path(path).resolve().parent
                or config_sha256(artifact["path"]) != artifact.get("sha256")):
            raise ValueError("execution artifact missing or hash-mismatched")
    return payload
