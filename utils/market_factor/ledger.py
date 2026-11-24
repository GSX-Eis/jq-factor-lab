"""Append-only JSONL ledgers and their POSIX writer lock."""
from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from utils.time_axis import Frequency

from .schema import (
    _canonical_json,
    _require_nonempty_str,
    validate_factor_id,
    validate_sha256_hex,
)


def _identity_filters(
    factor_id: str | None,
    frequency: str | None,
    slot_id: str | None,
    factor_identity_hash: str | None,
) -> dict[str, str]:
    filters = {}
    if factor_id is not None:
        filters["factor_id"] = validate_factor_id(factor_id)
    if frequency is not None:
        parsed = Frequency.parse(frequency)
        if parsed.canonical is None:
            raise ValueError(f"Invalid frequency: {frequency!r}")
        filters["frequency"] = parsed.canonical
    if slot_id is not None:
        filters["slot_id"] = _require_nonempty_str(slot_id, "slot_id")
    if factor_identity_hash is not None:
        filters["factor_identity_hash"] = validate_sha256_hex(factor_identity_hash, "factor_identity_hash")
    return filters


def _matches_filters(record: Mapping[str, Any], filters: Mapping[str, str]) -> bool:
    output_fields = {"frequency": record.get("output_frequency") or record.get("frequency")}
    return all(output_fields.get(field, record.get(field)) == value for field, value in filters.items())


def _trim_unterminated_trailing_line(path: Path) -> None:
    """Under writer lock, trim an uncommitted unterminated trailing line before appending."""
    if not path.is_file():
        return
    fd = os.open(str(path), os.O_RDWR)
    try:
        size = os.lseek(fd, 0, os.SEEK_END)
        if size <= 0:
            return
        os.lseek(fd, size - 1, os.SEEK_SET)
        if os.read(fd, 1) == b"\n":
            return
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, size)
        last_nl = raw.rfind(b"\n")
        trim_to = 0 if last_nl == -1 else last_nl + 1
        os.ftruncate(fd, trim_to)
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_jsonl_line(path: Path, record: Mapping[str, Any]) -> None:
    _trim_unterminated_trailing_line(path)
    line = (_canonical_json(dict(record)) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = 0
        while written < len(line):
            n = os.write(fd, line[written:])
            if n <= 0:
                raise OSError(f"Short write appending to {path}")
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read complete JSONL records without writer lock; separate mid-file corrupt vs trailing partial lines."""
    if not path.is_file():
        return [], [], []
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    corrupt: list[dict[str, Any]] = []
    trailing_incomplete: list[dict[str, Any]] = []
    pos = 0
    line_no = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        line_no += 1
        if nl == -1:
            trailing_incomplete.append({"file": str(path), "line": line_no, "reason": "unterminated_line"})
            break
        raw_line = raw[pos : nl + 1]
        pos = nl + 1
        text = raw_line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                corrupt.append({"file": str(path), "line": line_no, "reason": "non_object_json"})
                continue
            records.append(obj)
        except json.JSONDecodeError as exc:
            corrupt.append({"file": str(path), "line": line_no, "reason": f"json_error: {exc}"})
    return records, corrupt, trailing_incomplete


@contextmanager
def _write_lock(root: Path, parquet_root: Path, lock_path: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    parquet_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _load_committed_state(identities_path: Path, batches_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    # Writers register an identity before committing its first batch.
    batches_list, batch_corrupt, _ = _read_jsonl(batches_path)
    identities_list, id_corrupt, _ = _read_jsonl(identities_path)
    if id_corrupt or batch_corrupt:
        raise ValueError(f"Corrupt ledger entries detected: {id_corrupt + batch_corrupt}")
    identities = {str(rec["factor_identity_hash"]): rec for rec in identities_list}
    committed_batches = [rec for rec in batches_list if rec.get("status") == "committed"]
    return identities, committed_batches
