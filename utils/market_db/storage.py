"""Read and maintain a DuckDB market profile and its read-only sidecar."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import struct
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import duckdb

from utils.data_store import _FROZEN_INPUT_PATH_VARIABLE, quote
from utils.local_outputs import reserve_output_directory, write_json

from .errors import (
    MARKET_DB_ALREADY_INITIALIZED,
    MARKET_DB_ATOMIC_PUBLISH_UNSUPPORTED,
    MARKET_DB_BINDING_MISMATCH,
    MARKET_DB_DOMAIN_MISMATCH,
    MARKET_DB_PROFILE_HASH_MISMATCH,
    MARKET_DB_PROFILE_INVALID,
    MARKET_DB_PROFILE_MISSING,
    MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH,
    MarketDbProfileError,
)
from .profile import MarketDbProfile, parse_profile_json

ProfileSource = Literal["table", "sidecar"]

_COLUMNS = {
    "schema_version": "INTEGER",
    "db_id": "VARCHAR",
    "market_domain": "VARCHAR",
    "allowed_slot_providers": "VARCHAR[]",
    "asset_classes": "VARCHAR[]",
    "timezone": "VARCHAR",
    "calendar_id": "VARCHAR",
    "calendar_source": "VARCHAR",
    "calendar_coverage": "VARCHAR",
    "exchanges": "VARCHAR[]",
    "exchange_aliases": "JSON",
    "instrument_id_pattern": "VARCHAR",
    "currency": "VARCHAR",
    "frequencies": "VARCHAR[]",
    "session_views": "VARCHAR[]",
    "market_rules": "JSON",
    "default_execution_profile": "VARCHAR",
    "created_at_utc": "VARCHAR",
    "profile_hash": "VARCHAR",
}
_COLUMN_NAMES = ", ".join(map(quote, _COLUMNS))


def market_db_sidecar_path(db_path: str | Path) -> Path:
    path = Path(db_path).expanduser().resolve()
    return path.with_name(f"{path.name}.market_profile.json")


def _connect_existing(path: Path, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if not path.is_file():
        raise MarketDbProfileError(MARKET_DB_PROFILE_MISSING, f"Database file does not exist: {path}")
    try:
        return duckdb.connect(str(path), read_only=read_only)
    except duckdb.Error as exc:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Cannot open market database {path}: {exc}") from exc


def _profile_table(connection: duckdb.DuckDBPyConnection, catalog: str) -> MarketDbProfile | None:
    try:
        table = connection.execute(
            """SELECT table_type FROM information_schema.tables
               WHERE table_catalog = ? AND table_schema = 'main' AND lower(table_name) = 'market_db_profile'""",
            [catalog],
        ).fetchone()
        if table is None:
            return None
        if table[0] != "BASE TABLE":
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, "market_db_profile must be a table")
        rows = connection.execute(
            f"SELECT singleton_guard, {_COLUMN_NAMES} FROM {quote(catalog)}.main.market_db_profile LIMIT 2"
        ).fetchall()
        if len(rows) != 1 or type(rows[0][0]) is not int or rows[0][0] != 1:
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, "market_db_profile must have exactly one row with guard 1")
        document = dict(zip(_COLUMNS, rows[0][1:]))
        for name in ("exchange_aliases", "market_rules"):
            if not isinstance(document[name], str):
                raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Invalid {name} in market_db_profile")
            document[name] = parse_profile_json(document[name])
        return MarketDbProfile.from_mapping(document, require_hash=True)
    except duckdb.Error as exc:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Cannot read market_db_profile: {exc}") from exc


def _file_fingerprint(path: Path) -> dict[str, int | str]:
    def identity(info: os.stat_result) -> tuple[int, int, int, int]:
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            header_hash = hashlib.sha256(handle.read(64 * 1024)).hexdigest()
            unchanged = identity(before) == identity(os.fstat(handle.fileno())) == identity(path.stat())
            if unchanged and not os.path.lexists(f"{path}.wal"):
                return {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns, "header_sha256": header_hash}
    except OSError as exc:
        raise MarketDbProfileError(MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH, f"Cannot fingerprint {path}: {exc}") from exc
    raise MarketDbProfileError(
        MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH, f"Database changed during fingerprinting or has a pending WAL: {path}"
    )


def _read_sidecar(path: Path) -> tuple[MarketDbProfile, dict[str, int | str]] | None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if not path.is_symlink():
            return None
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Broken sidecar link: {path}") from None
    except (OSError, UnicodeError) as exc:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Cannot read sidecar {path}: {exc}") from exc
    document = parse_profile_json(content)
    if not isinstance(document, dict):
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Sidecar must be a JSON object: {path}")
    fingerprint = document.pop("db_fingerprint", None)
    profile = MarketDbProfile.from_mapping(document, require_hash=True)
    if (not isinstance(fingerprint, dict) or fingerprint.keys() != {"size_bytes", "mtime_ns", "header_sha256"}
            or type(fingerprint["size_bytes"]) is not int or fingerprint["size_bytes"] < 0
            or type(fingerprint["mtime_ns"]) is not int or not isinstance(fingerprint["header_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", fingerprint["header_sha256"])):
        raise MarketDbProfileError(MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH, f"Invalid sidecar fingerprint: {path}")
    return profile, fingerprint


def _require_same_hash(found: MarketDbProfile, expected: MarketDbProfile, label: str) -> None:
    if found.profile_hash != expected.profile_hash:
        raise MarketDbProfileError(
            MARKET_DB_PROFILE_HASH_MISMATCH, f"{label}: {found.profile_hash} != {expected.profile_hash}"
        )


def _current_catalog(connection: duckdb.DuckDBPyConnection) -> tuple[str, str | None, str | None]:
    return connection.execute(
        f"SELECT database_name, path, getvariable('{_FROZEN_INPUT_PATH_VARIABLE}') "
        "FROM duckdb_databases() WHERE database_name = current_database()"
    ).fetchone()


def is_file_backed_market_db(connection: duckdb.DuckDBPyConnection) -> bool:
    """Recognize direct file catalogs and frozen inputs opened through connect_input."""
    _, filename, frozen_path = _current_catalog(connection)
    return bool(filename or frozen_path)


def read_market_db_profile(
    connection: duckdb.DuckDBPyConnection, *, write: bool = False,
) -> tuple[MarketDbProfile, ProfileSource]:
    """Read the physical profile belonging to this database or frozen input."""
    catalog, filename, frozen_path = _current_catalog(connection)
    if frozen_path is not None:
        if write:
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, "A frozen input database is read-only")
        attached = connection.execute(
            "SELECT database_name, path, readonly FROM duckdb_databases() WHERE path IS NOT NULL AND NOT internal"
        ).fetchall()
        if len(attached) != 1 or not attached[0][2] or attached[0][1] != frozen_path:
            raise MarketDbProfileError(MARKET_DB_BINDING_MISMATCH,
                                       "Frozen input must match exactly one read-only physical database")
        catalog, filename, _ = attached[0]
    table = _profile_table(connection, catalog)
    path = Path(filename).resolve() if filename else None
    sidecar = _read_sidecar(market_db_sidecar_path(path)) if path is not None else None
    if sidecar is not None:
        profile, fingerprint = sidecar
        if table is not None:
            _require_same_hash(table, profile, "Table and sidecar disagree")
        if _file_fingerprint(path) != fingerprint:
            raise MarketDbProfileError(MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH, f"Sidecar fingerprint disagrees with {path}")
        if write:
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, "A database with a sidecar must be opened read-only")
        if table is None:
            return profile, "sidecar"
    if table is not None:
        return table, "table"
    raise MarketDbProfileError(MARKET_DB_PROFILE_MISSING, f"No market profile for {filename or catalog}")


def _insert_profile(connection: duckdb.DuckDBPyConnection, table: str, profile: MarketDbProfile) -> None:
    document = profile.as_dict()
    for name in ("exchange_aliases", "market_rules"):
        document[name] = json.dumps(document[name], allow_nan=False)
    placeholders = ", ".join("?" for _ in _COLUMNS)
    connection.execute(
        f"INSERT INTO {table} (singleton_guard, {_COLUMN_NAMES}) VALUES (1, {placeholders})",
        [document[name] for name in _COLUMNS],
    )


def _require_new_destination(path: Path) -> None:
    if os.path.lexists(path) or os.path.lexists(market_db_sidecar_path(path)) or os.path.lexists(f"{path}.wal"):
        raise MarketDbProfileError(MARKET_DB_ALREADY_INITIALIZED, f"Destination or companion file already exists: {path}")


def _require_darwin_exclusive_rename(libc: ctypes.CDLL, directory: Path) -> None:
    # Darwin sys/attr.h: attrlist requests volume capabilities; the reply
    # contains a length followed by four capability words and four validity words.
    volume_attributes = 0x80000000 | 0x00020000  # ATTR_VOL_INFO | ATTR_VOL_CAPABILITIES
    attributes = ctypes.create_string_buffer(struct.pack("=HHIIIII", 5, 0, 0, volume_attributes, 0, 0, 0))
    reply = (ctypes.c_uint32 * 9)()
    result = libc.getattrlist(
        os.fsencode(directory), attributes, reply, ctypes.c_size_t(ctypes.sizeof(reply)), ctypes.c_ulong(0)
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), directory)
    exclusive = 0x00080000  # VOL_CAP_INT_RENAME_EXCL, in the interface capability word
    if not (reply[2] & reply[6] & exclusive):
        raise OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP), directory)


def _rename_no_replace(staged: Path, target: Path) -> None:
    """Atomically publish a staged file if its destination name is unused."""
    libc = ctypes.CDLL(None, use_errno=True)
    source, destination = os.fsencode(staged), os.fsencode(target)
    try:
        if sys.platform == "darwin":
            result = libc.renamex_np(source, destination, ctypes.c_uint(4))  # RENAME_EXCL
        elif sys.platform.startswith("linux"):
            at_fdcwd = -100
            result = libc.renameat2(at_fdcwd, source, at_fdcwd, destination, ctypes.c_uint(1))  # RENAME_NOREPLACE
        else:
            raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS), target)
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS), target) from exc
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


@contextmanager
def _report_init_publish_errors(path: Path) -> Iterator[None]:
    try:
        yield
    except FileExistsError as exc:
        raise MarketDbProfileError(MARKET_DB_ALREADY_INITIALIZED, f"Destination already exists: {path}") from exc
    except OSError as exc:
        if exc.errno not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        raise MarketDbProfileError(
            MARKET_DB_ATOMIC_PUBLISH_UNSUPPORTED,
            f"Destination requires a filesystem that supports exclusive rename: {path.parent}. "
            "Use a directory on a volume with this capability, such as internal APFS on macOS.",
        ) from exc


def _create_profile_table(connection: duckdb.DuckDBPyConnection, profile: MarketDbProfile) -> None:
    columns = ", ".join(f"{quote(name)} {sql_type} NOT NULL" for name, sql_type in _COLUMNS.items())
    connection.execute(
        f"""CREATE TABLE main.market_db_profile (
            singleton_guard INTEGER PRIMARY KEY CHECK (singleton_guard = 1),
            {columns}
        )"""
    )
    _insert_profile(connection, "main.market_db_profile", profile)


def init_market_db(db_path: str | Path, profile: MarketDbProfile, *, dry_run: bool = False) -> None:
    """Publish a complete database using an exclusive filesystem rename."""
    target = Path(db_path).expanduser()
    path = target.parent.resolve() / target.name
    _require_new_destination(path)
    if not path.parent.is_dir():
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Destination directory does not exist: {path.parent}")
    if sys.platform == "darwin":
        with _report_init_publish_errors(path):
            _require_darwin_exclusive_rename(ctypes.CDLL(None, use_errno=True), path.parent)
    if dry_run:
        return
    with reserve_output_directory(path.parent, overwrite=True), TemporaryDirectory(
        prefix=".market-db-", dir=path.parent
    ) as directory:
        _require_new_destination(path)
        staged = Path(directory) / "market_data.duckdb"
        with duckdb.connect(str(staged)) as connection:
            _create_profile_table(connection, profile)
        _require_new_destination(path)
        with _report_init_publish_errors(path):
            _rename_no_replace(staged, path)


def adopt_market_db_sidecar(db_path: str | Path, profile: MarketDbProfile, *, dry_run: bool = False) -> Path:
    """Fingerprint an existing database and write only its companion JSON file."""
    path = Path(db_path).expanduser().resolve()
    sidecar_path = market_db_sidecar_path(path)
    with _connect_existing(path) as connection, (
        nullcontext() if dry_run else reserve_output_directory(path.parent, overwrite=True)
    ):
        catalog = connection.execute("SELECT current_database()").fetchone()[0]
        table = _profile_table(connection, catalog)
        if table is not None:
            _require_same_hash(table, profile, "Table and proposed sidecar disagree")
        previous = _read_sidecar(sidecar_path)
        if previous is not None:
            _require_same_hash(previous[0], profile, "Existing and proposed sidecar profiles disagree")
        fingerprint = _file_fingerprint(path)
        if not dry_run:
            write_json(sidecar_path, {**profile.as_dict(), "db_fingerprint": fingerprint})
    return sidecar_path


@contextmanager
def _migration_transaction(connection: duckdb.DuckDBPyConnection) -> Iterator[None]:
    connection.execute("BEGIN TRANSACTION")
    try:
        yield
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _migration_source(connection: duckdb.DuckDBPyConnection, path: Path, profile: MarketDbProfile) -> MarketDbProfile:
    previous, source = read_market_db_profile(connection)
    if source == "sidecar" or os.path.lexists(market_db_sidecar_path(path)):
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, "Profile migration requires a table without a sidecar")
    if previous.market_domain != profile.market_domain:
        raise MarketDbProfileError(MARKET_DB_DOMAIN_MISMATCH, "Profile migration must preserve market_domain")
    if previous.db_id != profile.db_id:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, "Profile migration must preserve db_id")
    return previous


def migrate_market_db_profile(
    db_path: str | Path, profile: MarketDbProfile, *, dry_run: bool = False
) -> MarketDbProfile:
    """Archive and replace a table profile in a single transaction."""
    path = Path(db_path).expanduser().resolve()
    with _connect_existing(path) as connection:
        previous = _migration_source(connection, path, profile)
    if dry_run or previous.profile_hash == profile.profile_hash:
        return previous
    with _connect_existing(path, read_only=False) as connection, _migration_transaction(connection):
        previous = _migration_source(connection, path, profile)
        if previous.profile_hash != profile.profile_hash:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS main.market_db_profile_history (
                    migrated_at_utc TIMESTAMPTZ NOT NULL,
                    profile_hash VARCHAR NOT NULL,
                    previous_profile JSON NOT NULL
                )"""
            )
            connection.execute(
                "INSERT INTO main.market_db_profile_history VALUES (current_timestamp, ?, ?)",
                [previous.profile_hash, json.dumps(previous.as_dict(), allow_nan=False)],
            )
            connection.execute("DELETE FROM main.market_db_profile")
            _insert_profile(connection, "main.market_db_profile", profile)
    return previous
