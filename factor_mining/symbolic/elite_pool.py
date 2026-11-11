"""Persistent cross-run elite expression pool for warm-start seeding.

Each symbolic-mining run discovers a hall of fame, but that knowledge is thrown
away when the run ends, so every run starts its policy search from scratch. The
elite pool persists the best expressions (as serialized AST trees, never fragile
strings) across runs so a later run can warm-start its GP population from known
good structures -- the lever that lifted OOS rank IC in warm-start GP studies
(arXiv 2412.00896): restricting the search to proven structures mitigates the
overfitting that pure from-scratch search suffers on weak short-horizon signals.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .artifacts import write_json
from .expression import (
    ExpressionNode,
    canonical_expression_hash,
    deserialize_expression,
    referenced_features,
    serialize_expression,
)


def _read_pool(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _entry_score(entry: dict[str, Any]) -> float:
    try:
        return float(entry.get("score"))
    except (TypeError, ValueError):
        return float("-inf")


def update_elite_pool(
    pool_path: str | Path,
    candidates: Iterable[Any],
    *,
    run_id: str,
    collect_top_k: int = 12,
    max_pool_size: int = 200,
) -> dict[str, Any]:
    """Merge a run's top hall-of-fame candidates into the persistent pool.

    Candidates are keyed by canonical expression hash (best score wins), sorted
    by score descending, and capped at ``max_pool_size``. The score is the
    candidate's validation-window selection score (OOS is not known at search
    time); structurally strong seeds are what the warm-start GP needs.
    """
    path = Path(pool_path)
    by_hash: dict[str, dict[str, Any]] = {}
    for entry in _read_pool(path):
        key = str(entry.get("expression_hash") or "")
        if key:
            by_hash[key] = entry

    added = 0
    updated = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for candidate in list(candidates)[: max(0, int(collect_top_k))]:
        expr = getattr(candidate, "expression", None)
        if expr is None:
            continue
        try:
            key = canonical_expression_hash(expr)
        except Exception:
            continue
        score = float(getattr(candidate, "score", float("-inf")))
        prior = by_hash.get(key)
        if prior is not None and _entry_score(prior) >= score:
            continue
        entry: dict[str, Any] = {
            "expression_hash": key,
            "expression": expr.to_string(),
            "expression_ast": serialize_expression(expr),
            "score": score,
            "complexity": int(getattr(candidate, "complexity", 0) or 0),
            "run_id": str(run_id),
            "updated_at": now,
        }
        valid_metrics = getattr(candidate, "valid_metrics", None)
        if isinstance(valid_metrics, dict):
            try:
                entry["valid_rank_ic_mean"] = float(valid_metrics.get("rank_ic_mean"))
            except (TypeError, ValueError):
                pass
        added += int(prior is None)
        updated += int(prior is not None)
        by_hash[key] = entry

    merged = sorted(by_hash.values(), key=_entry_score, reverse=True)[
        : max(1, int(max_pool_size))
    ]
    write_json(path, {"updated_at": now, "size": len(merged), "entries": merged})
    return {
        "pool_path": str(path),
        "size": len(merged),
        "added": added,
        "updated": updated,
    }


def load_elite_expressions(
    pool_path: str | Path,
    *,
    max_seeds: int = 24,
    available_features: set[str] | None = None,
) -> list[ExpressionNode]:
    """Load the top pooled expressions as AST trees for warm-start seeding.

    Entries whose referenced features are not all present in
    ``available_features`` are skipped: the pool accumulates across feature
    presets, so an intraday/fundamental seed must not be injected into a run
    whose panel lacks those columns.
    """
    entries = sorted(_read_pool(Path(pool_path)), key=_entry_score, reverse=True)
    out: list[ExpressionNode] = []
    limit = max(0, int(max_seeds))
    for entry in entries:
        if len(out) >= limit:
            break
        ast = entry.get("expression_ast")
        if not isinstance(ast, dict):
            continue
        try:
            expr = deserialize_expression(ast)
        except Exception:
            continue
        if available_features is not None and not referenced_features(expr).issubset(
            available_features
        ):
            continue
        out.append(expr)
    return out
