"""One comparison contract for research snapshots and execution ledgers."""

import json
import math
import html
from pathlib import Path

STAGES = ("inputs", "factor", "preprocess", "score", "targets", "orders", "cash", "nav")
LAYERS = {
    "inputs": ("inputs",),
    "factor": ("factor",),
    "signal": ("preprocess", "score", "targets"),
    "execution": ("orders", "cash", "nav"),
}


def read_records(path, fingerprint):
    records = {}
    for line in Path(path).read_text().splitlines():
        if "ALIGNMENT_SNAPSHOT|" in line:
            line = line.split("ALIGNMENT_SNAPSHOT|", 1)[1]
        elif not line.startswith("{"):
            continue
        line = html.unescape(line)
        item, consumed = json.JSONDecoder().raw_decode(line)
        if line[consumed:].strip() not in ("", r"\n"):
            raise ValueError("snapshot_trailing_content")
        if item.get("fingerprint") != fingerprint:
            raise ValueError("snapshot_version_mismatch")
        values = (
            item["value"].items()
            if item["key"].startswith("__batch__:")
            else ((item["key"], item["value"]),)
        )
        for raw_key, value in values:
            key = (item["date"], item["stage"], raw_key)
            if item["stage"] not in STAGES or key in records:
                raise ValueError("snapshot_duplicate_or_invalid_stage: " + str(key))
            records[key] = value
    if not records:
        raise ValueError("snapshot_empty")
    return records


def _equal(a, b, atol, rtol):
    if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (
            math.isfinite(a)
            and math.isfinite(b)
            and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k], atol, rtol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y, atol, rtol) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def compare_records(local, cloud, expected_dates=None, tolerances=None):
    stages = {}
    tolerances = tolerances or {}
    for stage in STAGES:
        keys = {k for k in local.keys() | cloud.keys() if k[1] == stage}
        problems = []
        for day in (expected_dates or {}).get(stage, []):
            if not any(k[0] == day for k in keys):
                problems.append(
                    {
                        "date": day,
                        "stage": stage,
                        "key": "*",
                        "reason": "data_missing_both",
                    }
                )
        for key in sorted(keys):
            reason = ""
            if key not in local or key not in cloud:
                reason = (
                    "data_missing_local" if key not in local else "data_missing_cloud"
                )
            elif not _equal(
                local[key],
                cloud[key],
                float(tolerances.get(stage, {}).get("atol", 1e-6)),
                float(tolerances.get(stage, {}).get("rtol", 1e-6)),
            ):
                reason = "numerical_mismatch"
            if reason:
                problems.append(
                    {
                        "date": key[0],
                        "stage": stage,
                        "key": key[2],
                        "reason": reason,
                        "local": local.get(key),
                        "cloud": cloud.get(key),
                    }
                )
        problems.sort(key=lambda p: (p["date"], p["key"]))
        stages[stage] = {
            "status": "FAIL" if problems else ("PASS" if keys else "NOT_VERIFIED"),
            "compared": len(keys),
            "difference_count": len(problems),
            "first_difference": problems[0] if problems else None,
        }
    layers = {}
    for layer, members in LAYERS.items():
        statuses = [stages[s]["status"] for s in members]
        layers[layer] = {
            "status": "FAIL"
            if "FAIL" in statuses
            else ("NOT_VERIFIED" if "NOT_VERIFIED" in statuses else "PASS"),
            "stages": {s: stages[s] for s in members},
        }
    differences = [
        v["first_difference"] for v in stages.values() if v["first_difference"]
    ]
    first = (
        min(differences, key=lambda v: (v["date"], STAGES.index(v["stage"])))
        if differences
        else None
    )
    return {
        "status": "PASS"
        if all(v["status"] == "PASS" for v in layers.values())
        else "FAIL",
        "layers": layers,
        "first_difference": first,
        "failed_stage": first["stage"]
        if first
        else next((s for s in STAGES if stages[s]["status"] != "PASS"), ""),
        "publication_status": "pending_approval",
    }
