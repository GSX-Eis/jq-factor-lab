"""Read-only reconciliation of ledger records, staging and published shards."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .identity import _validate_identity_record
from .ledger import (
    _identity_filters,
    _matches_filters,
    _read_jsonl,
)
from .schema import MARKET_FACTOR_IDENTITY_SCHEMA_VERSION, _shard_ownership_fields, _shard_ownership_mismatches


def verify_store(
    root: Path, parquet_root: Path, identities_path: Path, batches_path: Path,
    *, factor_id: str | None = None, frequency: str | None = None, slot_id: str | None = None,
    factor_identity_hash: str | None = None,
) -> dict[str, Any]:
    """Read-only consistency check reconciling disk shards and staging with JSONL ledger."""
    filters = _identity_filters(factor_id, frequency, slot_id, factor_identity_hash)
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
            "shard_metadata_mismatches": [],
            "unregistered_identities": [],
            "invalid_identities": [],
            "invalid_batches": [],
        }

    batches_list, batch_corrupt, batch_trailing = _read_jsonl(batches_path)
    identities_list, id_corrupt, id_trailing = _read_jsonl(identities_path)
    corrupt_lines = id_corrupt + id_trailing + batch_corrupt + batch_trailing

    identities = {str(rec["factor_identity_hash"]): rec for rec in identities_list if "factor_identity_hash" in rec}
    selected_identities = set()
    invalid_identities: list[dict[str, Any]] = []
    for record in identities_list:
        if not _matches_filters(record, filters):
            continue
        fhash = record.get("factor_identity_hash")
        try:
            _validate_identity_record(record)
            if fhash in selected_identities:
                raise ValueError("Duplicate identity ledger entry")
        except (ValueError, TypeError, KeyError) as exc:
            invalid_identities.append({"factor_identity_hash": fhash, "error": str(exc)})
        if isinstance(fhash, str):
            selected_identities.add(fhash)

    committed_paths: set[str] = set()
    missing_shards: list[str] = []
    row_count_mismatches: list[dict[str, Any]] = []
    shard_metadata_mismatches: list[dict[str, Any]] = []
    unregistered_identities: list[str] = []
    invalid_batches: list[dict[str, Any]] = []
    total_committed_rows = 0
    batch_count = 0

    for batch in batches_list:
        if batch.get("status") != "committed":
            continue
        fhash = str(batch.get("factor_identity_hash", ""))
        identity = identities.get(fhash)
        if fhash not in selected_identities and not _matches_filters(batch, filters):
            continue
        batch_count += 1
        batch_id = batch.get("batch_id")
        schema_version = identity.get("source_schema_version") if identity is not None else None
        expected_fields = _shard_ownership_fields(fhash, batch_id, schema_version)
        if identity is None and fhash not in unregistered_identities:
            unregistered_identities.append(fhash)
        if identity is not None:
            expected = {
                "factor_id": identity.get("factor_id"),
                "frequency": identity.get("output_frequency") or identity.get("frequency"),
                "session_view": identity.get("session_view"),
            }
            if identity.get("source_schema_version") == MARKET_FACTOR_IDENTITY_SCHEMA_VERSION:
                expected["slot_id"] = identity.get("slot_id")
            for field, value in expected.items():
                if batch.get(field) != value:
                    invalid_batches.append({"batch_id": batch_id, "error": f"{field} does not match identity"})
        shards = batch.get("shards")
        if not isinstance(shards, list) or not shards:
            invalid_batches.append({"batch_id": batch_id, "error": "Committed batch requires shards"})
            continue
        batch_rows = 0
        for shard in shards:
            try:
                rel = shard["part_path"]
                part = Path(rel)
                prefix = (f"factor_id={batch['factor_id']}", f"frequency={batch['frequency']}",
                          f"year_month={shard['year_month']}")
                if len(part.parts) != 4 or part.parts[:3] != prefix or part.suffix != ".parquet":
                    raise ValueError("Shard path does not match the Hive partition")
                expected_rows = shard["row_count"]
                if type(expected_rows) is not int or expected_rows < 1:
                    raise ValueError("Shard row_count must be a positive integer")
                if rel in committed_paths:
                    raise ValueError(f"Duplicate committed shard path {rel}")
            except (ValueError, TypeError, KeyError) as exc:
                invalid_batches.append({"batch_id": batch_id, "error": str(exc)})
                continue
            committed_paths.add(rel)
            batch_rows += expected_rows
            total_committed_rows += expected_rows
            full_path = parquet_root / rel
            if not full_path.is_file():
                missing_shards.append(rel)
            else:
                try:
                    with pq.ParquetFile(full_path) as file:
                        actual_rows = file.metadata.num_rows
                        if actual_rows != expected_rows:
                            row_count_mismatches.append(
                                {"part_path": rel, "expected_rows": expected_rows, "actual_rows": actual_rows}
                            )
                        mismatches = _shard_ownership_mismatches(file, expected_fields)
                        if mismatches:
                            shard_metadata_mismatches.append({"part_path": rel, "fields": mismatches})
                except Exception as exc:
                    row_count_mismatches.append(
                        {"part_path": rel, "expected_rows": expected_rows, "error": str(exc)}
                    )
        if batch.get("row_count") != batch_rows:
            invalid_batches.append({"batch_id": batch_id, "error": "Batch row_count does not match its shards"})

    # Files without a commit record and staging directories have no reliable slot attribution.
    uncommitted_shards: list[str] = []
    if not filters and parquet_root.is_dir():
        for path in sorted(parquet_root.rglob("*.parquet")):
            rel = str(path.relative_to(parquet_root))
            if rel not in committed_paths:
                uncommitted_shards.append(rel)

    staging_leftovers = sorted(
        str(p.relative_to(root))
        for p in root.iterdir()
        if p.name.startswith(".staging")
    ) if not filters else []

    ok = not (
        corrupt_lines
        or missing_shards
        or uncommitted_shards
        or staging_leftovers
        or row_count_mismatches
        or shard_metadata_mismatches
        or unregistered_identities
        or invalid_identities
        or invalid_batches
    )
    return {
        "ok": ok,
        "root": str(root),
        "missing_root": False,
        "identity_count": len(selected_identities),
        "batch_count": batch_count,
        "committed_shard_count": len(committed_paths),
        "total_committed_rows": total_committed_rows,
        "missing_shards": missing_shards,
        "uncommitted_shards": uncommitted_shards,
        "staging_leftovers": staging_leftovers,
        "corrupt_ledger_lines": corrupt_lines,
        "row_count_mismatches": row_count_mismatches,
        "shard_metadata_mismatches": shard_metadata_mismatches,
        "unregistered_identities": unregistered_identities,
        "invalid_identities": invalid_identities,
        "invalid_batches": invalid_batches,
    }
