"""Owned local outputs: durable work/reports, disposable scratch, leased caches."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from numbers import Integral, Real
from pathlib import Path
from typing import Any


def json_compatible(payload: Any) -> Any:
    """Recursively encode non-finite numeric evidence as explicit JSON nulls."""
    if isinstance(payload, bool) or payload is None or isinstance(payload, str):
        return payload
    if isinstance(payload, Integral):
        return int(payload)
    if isinstance(payload, Real):
        value = float(payload)
        return value if math.isfinite(value) else None
    if isinstance(payload, Mapping):
        return {key: json_compatible(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_compatible(value) for value in payload]
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    """Atomically persist strict JSON while leaving in-memory research values untouched."""
    temporary = Path(path).with_name(f".{Path(path).name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                json_compatible(payload),
                ensure_ascii=False,
                indent=2,
                default=str,
                allow_nan=False,
            )
            + "\n"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


MARKER = ".managed-output.json"
VERSION = "local_output.v2"
KINDS = {"temporary": "tmp", "cache": "cache"}


def output_root(task_root: str | Path | None = None) -> Path:
    """Explicit caller root, then runner binding, then this installed checkout."""
    return Path(
        task_root
        or os.environ.get("FACTORPILOT_ROOT")
        or Path(__file__).resolve().parents[1]
    ).resolve()


def output_directory(
    task_root: Path, category: str, name: str = "", *, create: bool = True
) -> Path:
    if category not in {"work", "reports", "tmp", "cache"}:
        raise ValueError(f"unsupported output category: {category}")
    if name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", name):
        raise ValueError(
            "output identity must be 1–160 letters, digits, dots, underscores or hyphens"
        )
    root = output_root(task_root)
    path = root
    for part in (category, *([name] if name else [])):
        path /= part
        if path.is_symlink():
            raise ValueError(f"output directory cannot be a symlink: {path}")
        if path.exists() and not path.is_dir():
            raise ValueError(f"output directory is not a directory: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def output_lock(
    root: Path, *, exclusive: bool = False, create: bool = True, metadata: bool = False
) -> Iterator[int | None]:
    # Category locks protect legacy users and directory creation/removal;
    # v2 producers hold an inherited entry lock during computation.
    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
    try:
        fd = os.open(
            root / (".metadata.lock" if metadata else ".gc.lock"), flags, 0o600
        )
    except FileNotFoundError:
        yield None
        return
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
            raise ValueError("invalid output lease file")
        try:
            mode = fcntl.LOCK_EX if exclusive or metadata else fcntl.LOCK_SH
            fcntl.flock(fd, mode | (fcntl.LOCK_NB if exclusive and not metadata else 0))
        except BlockingIOError:
            yield None
            return
        # Closing, rather than LOCK_UN, keeps an inherited child lease alive.
        yield fd
    finally:
        os.close(fd)


def read_output_marker(path: Path, kind: str) -> dict:
    marker = path / MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("unmanaged")
    value = json.loads(marker.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema_version") not in {"local_output.v1", VERSION}
        or value.get("kind") != kind
        or value.get("identity") != path.name
        or value.get("state") not in {"running", "completed", "failed"}
        or value.get("disposable") is not True
    ):
        raise ValueError("invalid ownership marker")
    cleanup = value.get("cleanup")
    if cleanup is not None and (
        not isinstance(cleanup, dict)
        or not isinstance(cleanup.get("id"), str)
        or not re.fullmatch(r"[0-9a-f]{32}", cleanup["id"])
        or type(cleanup.get("bytes_before")) is not int
        or cleanup["bytes_before"] < 0
    ):
        raise ValueError("invalid cleanup marker")
    return value


@contextmanager
def managed_output(
    task_root: str | Path | None, purpose: str, *, kind: str = "temporary"
) -> Iterator[tuple[Path, int]]:
    """Lease scratch/cache bytes, never frozen inputs or resumable research state.

    Subprocess owners must pass the returned fd in pass_fds. Crashed/incomplete
    temporary operations remain uncollectable until their outcome is known.
    """
    if kind not in KINDS:
        raise ValueError("only temporary and cache outputs can be managed")
    task_root = output_root(task_root)
    root = output_directory(task_root, KINDS[kind])
    identity = f"{purpose}-{uuid.uuid4().hex[:12]}" if kind == "temporary" else purpose
    path = output_directory(task_root, KINDS[kind], identity, create=False)
    # Upgrade only when legacy readers are idle; old collectors reject v2.
    with output_lock(root, exclusive=True) as upgrade_fd:
        if upgrade_fd is not None and path.exists():
            previous = read_output_marker(path, kind)
            if previous["schema_version"] != VERSION and not previous.get("cleanup"):
                previous["schema_version"] = VERSION
                write_json(path / MARKER, previous)
    with ExitStack() as leases:
        category = ExitStack()
        leases.callback(category.close)
        fd = category.enter_context(output_lock(root))
        with output_lock(root, metadata=True):
            if path.exists():
                previous = read_output_marker(path, kind)
                if previous.get("cleanup"):
                    raise ValueError(
                        "output cleanup is incomplete; resume state gc before reuse"
                    )
            else:
                previous = {}
            path.mkdir(exist_ok=True)
            now = time.time()
            marker = {
                "schema_version": previous.get("schema_version", VERSION),
                "identity": identity,
                "kind": kind,
                "purpose": purpose,
                "disposable": True,
                "created_at": previous.get("created_at", now),
                "last_used": now,
                "state": "running" if kind == "temporary" else "completed",
            }
            if marker["schema_version"] == VERSION:
                fd = leases.enter_context(output_lock(path))
            write_json(path / MARKER, marker)
        if fd is None:
            raise RuntimeError("could not acquire output lease")
        entry_root = path if marker["schema_version"] == VERSION else root
        if entry_root == path:
            category.close()
        try:
            yield path, fd
        except BaseException:
            with output_lock(entry_root, metadata=True):
                marker.update(
                    state="failed", finished_at=time.time(), last_used=time.time()
                )
                write_json(path / MARKER, marker)
            raise
        else:
            with output_lock(entry_root, metadata=True):
                marker.update(
                    state="completed", finished_at=time.time(), last_used=time.time()
                )
                write_json(path / MARKER, marker)
    if kind == "temporary":
        with output_lock(root, exclusive=True) as fd:
            if fd is not None and path.exists():
                if path.is_symlink():
                    raise ValueError("temporary output changed into a symlink")
                read_output_marker(path, kind)
                with output_lock(path, exclusive=True) as entry_fd:
                    if entry_fd is not None:
                        shutil.rmtree(path)


@contextmanager
def cache_lease(path):
    """Protect existing managed caches even when a library is called directly."""
    if path:
        current = Path(path).expanduser().resolve()
        for candidate in (current, *current.parents):
            if candidate.parent.name == "cache" and (candidate / MARKER).exists():
                with managed_output(
                    candidate.parent.parent, candidate.name, kind="cache"
                ):
                    yield
                return
    # Explicit legacy caches remain unmanaged until their owner is established.
    yield
