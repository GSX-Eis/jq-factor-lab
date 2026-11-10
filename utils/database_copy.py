"""Resumable physical copies of closed, checkpointed DuckDB snapshots."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import duckdb


def clone_database(source: str | Path, destination: str | Path, *, execute: bool = False) -> dict:
    """Keep partial blocks on failure; publish only a byte-verified database file.

    rsync owns block reuse and transfer recovery. A DuckDB read lock prevents
    concurrent writers, and a source fingerprint prevents resuming a new snapshot.
    This copies the whole database, including indexes, without SQL row insertion.
    """
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination:
        raise ValueError("Source and destination must differ")
    partial = Path(str(destination) + ".partial")
    receipt = Path(str(destination) + ".copy.json")
    if not execute:
        return dict(
            ok=True,
            execute=False,
            source=str(source),
            destination=str(destination),
            bytes=source.stat().st_size,
            partial_bytes=partial.stat().st_size if partial.exists() else 0,
        )
    rsync = shutil.which("rsync")
    if not rsync:
        raise RuntimeError("Resumable physical copies require rsync")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save(state):
        temporary = Path(str(receipt) + ".tmp")
        with temporary.open("w") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(receipt)

    def sha256(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    with Path(str(destination) + ".write.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if Path(str(source) + ".wal").exists():
            raise ValueError("Source has a WAL; checkpoint it before physical copying")
        with duckdb.connect(str(source), read_only=True):
            if Path(str(source) + ".wal").exists():
                raise ValueError("Source has a WAL; checkpoint it before physical copying")
            stat = source.stat()
            identity = dict(
                source=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns, device=stat.st_dev, inode=stat.st_ino
            )
            state = json.loads(receipt.read_text()) if receipt.exists() else None
            if state and state["source_snapshot"] != identity:
                raise ValueError("Source snapshot changed; refusing to reuse partial copy")
            if destination.exists():
                if state and state.get("sha256") == sha256(destination) == sha256(source):
                    state.update(phase="complete", ok=True)
                    save(state)
                    return state
                raise ValueError("Destination already exists; refusing to overwrite it")
            if state and state["phase"] == "complete":
                raise ValueError("Completed copy was moved or removed; inspect its adoption before copying again")
            if partial.exists() and state is None:
                raise ValueError("Partial copy has no snapshot receipt")
            remaining = max(0, stat.st_size - (partial.stat().st_size if partial.exists() else 0))
            if shutil.disk_usage(destination.parent).free < remaining + 1024**3:
                raise OSError("Insufficient space for the remaining copy plus 1 GiB reserve")
            transfer_complete = (
                state
                and state["phase"] in {"verifying", "verified"}
                and partial.exists()
                and partial.stat().st_size == stat.st_size
            )
            if not transfer_complete:
                state = dict(source_snapshot=identity, destination=str(destination), phase="copying", ok=False)
                save(state)
                # ponytail: one local transfer at a time; rsync's block checksums handle resume.
                result = subprocess.run(
                    [
                        rsync,
                        "--partial",
                        "--inplace",
                        "--no-whole-file",
                        *(["--checksum"] if partial.exists() else []),
                        "--stats",
                        str(source),
                        str(partial),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    check=False,
                )
                if result.returncode:
                    raise RuntimeError("Physical copy interrupted; rerun the same manifest: " + result.stderr[-2000:])
                state.update(transfer=result.stdout)
            state.update(phase="verifying")
            save(state)
            with partial.open("rb") as handle:
                os.fsync(handle.fileno())
            checksum = sha256(source)
            if partial.stat().st_size != stat.st_size or sha256(partial) != checksum:
                state.update(phase="copying")  # A retry must repair blocks, not only rehash them.
                save(state)
                raise ValueError("Physical copy checksum mismatch; partial file was not published")
            current = source.stat()
            if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError("Source changed during copy")
            with duckdb.connect(str(partial), read_only=True) as copied:
                tables = copied.execute("SHOW ALL TABLES").fetchall()
            state.update(phase="verified", sha256=checksum, table_count=len(tables))
            save(state)
            partial.replace(destination)
            state.update(phase="complete", ok=True)
            save(state)
            return state
