"""Read-only reconciliation of ledger records, staging and published shards."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .ledger import (
    _read_jsonl,
)


def verify_store(
    root: Path, parquet_root: Path, identities_path: Path, batches_path: Path,
) -> dict[str, Any]:
    """Read-only consistency check reconciling disk shards and staging with JSONL ledger."""
    if not root.exists():
        return {
            "ok": False,
            "root": str(root),
            "missing_root": True,
            "identity_count": 0,
            "batch_count": 0,
            "committed_shard_count": 0,
            "total_committed_rows": 0,
            "missing_shards": [],
            "uncommitted_shards": [],
            "staging_leftovers": [],
            "corrupt_ledger_lines": [],
            "row_count_mismatches": [],
            "unregistered_identities": [],
        }

    identities_list, id_corrupt, id_trailing = _read_jsonl(identities_path)
    batches_list, batch_corrupt, batch_trailing = _read_jsonl(batches_path)
    corrupt_lines = id_corrupt + id_trailing + batch_corrupt + batch_trailing

    identities = {str(rec["factor_identity_hash"]): rec for rec in identities_list if "factor_identity_hash" in rec}
    committed_batches = [rec for rec in batches_list if rec.get("status") == "committed"]

    committed_paths: set[str] = set()
    missing_shards: list[str] = []
    row_count_mismatches: list[dict[str, Any]] = []
    unregistered_identities: list[str] = []
    total_committed_rows = 0

    for batch in committed_batches:
        fhash = str(batch.get("factor_identity_hash", ""))
        if fhash not in identities and fhash not in unregistered_identities:
            unregistered_identities.append(fhash)
        for shard in batch.get("shards", []):
            rel = str(shard["part_path"])
            committed_paths.add(rel)
            expected_rows = int(shard["row_count"])
            total_committed_rows += expected_rows
            full_path = parquet_root / rel
            if not full_path.is_file():
                missing_shards.append(rel)
            else:
                try:
                    actual_rows = pq.read_metadata(str(full_path)).num_rows
                    if actual_rows != expected_rows:
                        row_count_mismatches.append(
                            {"part_path": rel, "expected_rows": expected_rows, "actual_rows": actual_rows}
                        )
                except Exception as exc:
                    row_count_mismatches.append(
                        {"part_path": rel, "expected_rows": expected_rows, "error": str(exc)}
                    )

    uncommitted_shards: list[str] = []
    if parquet_root.is_dir():
        for path in sorted(parquet_root.rglob("*.parquet")):
            rel = str(path.relative_to(parquet_root))
            if rel not in committed_paths:
                uncommitted_shards.append(rel)

    staging_leftovers = sorted(
        str(p.relative_to(root))
        for p in root.iterdir()
        if p.name.startswith(".staging")
    )

    ok = not (
        corrupt_lines
        or missing_shards
        or uncommitted_shards
        or staging_leftovers
        or row_count_mismatches
        or unregistered_identities
    )
    return {
        "ok": ok,
        "root": str(root),
        "missing_root": False,
        "identity_count": len(identities),
        "batch_count": len(committed_batches),
        "committed_shard_count": len(committed_paths),
        "total_committed_rows": total_committed_rows,
        "missing_shards": missing_shards,
        "uncommitted_shards": uncommitted_shards,
        "staging_leftovers": staging_leftovers,
        "corrupt_ledger_lines": corrupt_lines,
        "row_count_mismatches": row_count_mismatches,
        "unregistered_identities": unregistered_identities,
    }
