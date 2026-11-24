"""Resolve a physical market database and validate its identity on open."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb

from utils.data_store import binding_contract

from .errors import (
    MARKET_DB_BINDING_MISMATCH,
    MARKET_DB_BINDING_MISSING,
    MARKET_DB_PROFILE_INVALID,
    MarketDbProfileError,
)
from .profile import MarketDbProfile, is_market_db_id, load_market_db_profile_yaml
from .storage import ProfileSource, _connect_existing, _require_same_hash, read_market_db_profile


@dataclass(frozen=True)
class CrossMarketAttachSpec:
    """Future cross-market inputs use read-only ATTACH and as-of alignment on available_at_utc."""

    db_ids: tuple[str, ...]


def resolve_market_db_path(db_id_or_path: str | Path, data_bindings: dict | None = None) -> Path:
    """Resolve a binding ID, a Path, or a string with an explicit path marker."""
    if isinstance(db_id_or_path, Path):
        return db_id_or_path.expanduser().resolve()
    if not isinstance(db_id_or_path, str) or not db_id_or_path.strip() or "\x00" in db_id_or_path:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, "Expected a market db_id or local database path")
    if any(character in db_id_or_path for character in "/\\.") or db_id_or_path.startswith("~"):
        return Path(db_id_or_path).expanduser().resolve()
    if not is_market_db_id(db_id_or_path):
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Invalid market db_id: {db_id_or_path!r}")
    path = binding_contract(data_bindings, include_market_dbs=True).get("market_dbs", {}).get(db_id_or_path)
    if path is None:
        raise MarketDbProfileError(MARKET_DB_BINDING_MISSING, f"No market_dbs binding for {db_id_or_path!r}")
    return Path(path).expanduser().resolve()


@contextmanager
def open_market_db(
    db_id_or_path: str | Path,
    *,
    read_only: bool = True,
    data_bindings: dict | None = None,
    expected_profile_yaml: str | Path | None = None,
) -> Iterator[tuple[duckdb.DuckDBPyConnection, MarketDbProfile, ProfileSource]]:
    """Open and validate a single market database. Sidecar databases are read-only."""
    path = resolve_market_db_path(db_id_or_path, data_bindings)
    expected = load_market_db_profile_yaml(expected_profile_yaml) if expected_profile_yaml is not None else None

    def checked_profile(connection: duckdb.DuckDBPyConnection) -> tuple[MarketDbProfile, ProfileSource]:
        profile, source = read_market_db_profile(connection, write=not read_only)
        if is_market_db_id(db_id_or_path) and profile.db_id != db_id_or_path:
            raise MarketDbProfileError(
                MARKET_DB_BINDING_MISMATCH, f"Binding {db_id_or_path!r} points to database {profile.db_id!r}"
            )
        if expected is not None:
            _require_same_hash(profile, expected, "Database and expected profile disagree")
        return profile, source

    with _connect_existing(path) as connection:
        profile, source = checked_profile(connection)
        if read_only:
            yield connection, profile, source
            return
    with _connect_existing(path, read_only=False) as connection:
        profile, source = checked_profile(connection)
        yield connection, profile, source
