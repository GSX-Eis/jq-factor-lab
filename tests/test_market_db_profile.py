from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import duckdb
import pytest

from utils.data_store import resolve_connection_binding, resolve_database_config, validate_binding_contract
from utils.market_db import (
    MarketDbProfile,
    MarketDbProfileError,
    adopt_market_db_sidecar,
    init_market_db,
    load_market_db_profile_yaml,
    market_db_sidecar_path,
    migrate_market_db_profile,
    open_market_db,
    read_market_db_profile,
    resolve_market_db_path,
)
from utils.market_db import storage
from utils.market_db.errors import (
    MARKET_DB_ALREADY_INITIALIZED,
    MARKET_DB_ATOMIC_PUBLISH_UNSUPPORTED,
    MARKET_DB_BINDING_MISMATCH,
    MARKET_DB_BINDING_MISSING,
    MARKET_DB_DOMAIN_MISMATCH,
    MARKET_DB_PROFILE_HASH_MISMATCH,
    MARKET_DB_PROFILE_INVALID,
    MARKET_DB_PROFILE_MISSING,
    MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH,
)

TEMPLATES = Path(__file__).resolve().parents[1] / "config/market_databases"


@pytest.fixture
def profile():
    return load_market_db_profile_yaml(TEMPLATES / "cn_equity.yaml")


@pytest.fixture
def unprofiled_db(tmp_path):
    path = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE fixture AS SELECT 42 AS value")
    return path


@pytest.mark.parametrize(("domain", "provider", "instrument"), [
    ("cn_equity", "jq_ashare_local", "000001.XSHE"),
    ("cn_futures", "cn_futures", "IF2612.CFFEX"),
    ("global_futures", "global_futures", "ESZ26.CME"),
    ("us_equity", "duckdb_table", "BRK.B.XNYS"),
])
def test_market_db_yaml_templates(domain, provider, instrument):
    profile = load_market_db_profile_yaml(TEMPLATES / f"{domain}.yaml")
    assert profile.market_domain == domain
    assert profile.allowed_slot_providers == (provider,)
    assert re.fullmatch(profile.instrument_id_pattern, instrument)
    assert re.fullmatch(r"[0-9a-f]{64}", profile.profile_hash)


def test_profile_hash_is_deterministic_and_content_is_immutable(tmp_path, profile):
    document = profile.as_dict()
    document["market_rules"] = dict(reversed(list(document["market_rules"].items())))
    document["created_at_utc"] = "2026-11-24T12:30:00+00:00"
    declaration = tmp_path / "profile.json"
    declaration.write_text(json.dumps(dict(reversed(list(document.items())))), encoding="utf-8")
    loaded = load_market_db_profile_yaml(declaration)
    assert loaded.profile_hash == profile.profile_hash
    assert hashlib.sha256(profile.canonical_json().encode("utf-8")).hexdigest() == profile.profile_hash
    assert replace(profile, currency="EUR", profile_hash="").profile_hash != profile.profile_hash

    with pytest.raises(FrozenInstanceError):
        profile.db_id = "other"
    with pytest.raises(TypeError):
        profile.exchange_aliases["SSE"] = "XSHE"
    with pytest.raises(TypeError):
        profile.market_rules["bars_per_session"]["5m"] = 1
    document["market_rules"]["bars_per_session"]["5m"] = 1
    assert profile.market_rules["bars_per_session"]["5m"] == 48


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", True),
    ("db_id", "../legacy"),
    ("market_domain", "crypto"),
    ("allowed_slot_providers", "jq_ashare_local"),
    ("asset_classes", ["equity", "equity"]),
    ("timezone", "Not/AZone"),
    ("calendar_coverage", "2026-01-01..2025-01-01"),
    ("calendar_coverage", "2026-02-30..2026-03-01"),
    ("exchange_aliases", {"SSE": "XNYS"}),
    ("instrument_id_pattern", "["),
    ("frequencies", ["2m"]),
    ("frequencies", ["1d", "daily"]),
    ("session_views", ["regular"]),
    ("market_rules", {"st_filter": True, "price_limit": True, "t_plus": True}),
    ("market_rules", {"st_filter": True, "price_limit": True, "t_plus": 1, "weight": float("inf")}),
    ("market_rules", {"st_filter": True, "price_limit": True, "t_plus": 1, "bars_per_session": {"15m": 16}}),
    ("created_at_utc", "2026-01-01T00:00:00+08:00"),
    ("unknown", True),
])
def test_invalid_profile_fields_fail_closed(profile, field, value):
    document = profile.as_dict()
    document[field] = value
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_PROFILE_INVALID):
        MarketDbProfile.from_mapping(document)


def test_missing_fields_and_duplicate_yaml_keys_are_invalid(tmp_path, profile):
    document = profile.as_dict()
    del document["calendar_id"]
    with pytest.raises(MarketDbProfileError, match="missing=.*calendar_id"):
        MarketDbProfile.from_mapping(document)
    declaration = tmp_path / "duplicate.yaml"
    declaration.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(MarketDbProfileError, match="Duplicate key"):
        load_market_db_profile_yaml(declaration)


def test_init_profile_table_and_singleton_constraint(tmp_path, profile, monkeypatch):
    db = tmp_path / "table.duckdb"

    def links_unavailable(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    monkeypatch.setattr(os, "link", links_unavailable)
    init_market_db(db, profile)
    with open_market_db(db, read_only=False) as (conn, stored, source):
        assert stored == profile and source == "table"
        with pytest.raises(duckdb.ConstraintException):
            conn.execute("INSERT INTO market_db_profile SELECT * FROM market_db_profile")
        with pytest.raises(duckdb.ConstraintException):
            conn.execute("UPDATE market_db_profile SET singleton_guard = 2")
        conn.execute("CREATE TEMP TABLE market_db_profile AS SELECT 'shadow' AS db_id")
        assert read_market_db_profile(conn) == (profile, "table")
    with pytest.raises(duckdb.ConnectionException):
        conn.execute("SELECT 1")
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_ALREADY_INITIALIZED):
        init_market_db(db, profile)


def test_init_cleans_up_failed_publication(tmp_path, profile, monkeypatch):
    db = tmp_path / "failed.duckdb"

    def failed_rename(staged, target):
        raise OSError("injected publication failure")

    monkeypatch.setattr(storage, "_rename_no_replace", failed_rename)
    with pytest.raises(OSError, match="injected publication failure"):
        init_market_db(db, profile)
    assert not db.exists()
    assert not list(tmp_path.iterdir())


def test_init_refuses_filesystem_without_exclusive_rename(tmp_path, profile, monkeypatch):
    db = tmp_path / "unsupported.duckdb"

    def unavailable_rename(staged, target):
        raise OSError(errno.ENOTSUP, "exclusive rename unavailable")

    monkeypatch.setattr(storage, "_rename_no_replace", unavailable_rename)
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_ATOMIC_PUBLISH_UNSUPPORTED):
        init_market_db(db, profile)
    assert not db.exists()
    assert not list(tmp_path.iterdir())


def test_init_can_retry_after_process_exits_at_publication(tmp_path, profile):
    db = tmp_path / "interrupted.duckdb"
    child = subprocess.run([sys.executable, "-c", """
import os
import sys
from pathlib import Path
from utils.market_db import init_market_db, load_market_db_profile_yaml, storage

db = Path(sys.argv[1])

def interrupt_publish(staged, target):
    assert target == db
    os._exit(73)

storage._rename_no_replace = interrupt_publish
init_market_db(db, load_market_db_profile_yaml(sys.argv[2]))
""", str(db), str(TEMPLATES / "cn_equity.yaml")], timeout=30)
    assert child.returncode == 73
    assert not db.exists()
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_PROFILE_MISSING):
        with open_market_db(db):
            pytest.fail("Unpublished database should be absent")
    init_market_db(db, profile)
    with open_market_db(db) as (_, loaded, source):
        assert (loaded, source) == (profile, "table")


def test_init_preserves_destination_created_at_publication(tmp_path, profile, monkeypatch):
    db = tmp_path / "competing.duckdb"
    publish = storage._rename_no_replace

    def create_competing_database(staged, target):
        assert target == db
        with duckdb.connect(str(db)) as connection:
            connection.execute("CREATE TABLE competing_payload AS SELECT 42 AS sentinel")
        publish(staged, target)

    monkeypatch.setattr(storage, "_rename_no_replace", create_competing_database)
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_ALREADY_INITIALIZED):
        init_market_db(db, profile)
    with duckdb.connect(str(db), read_only=True) as connection:
        assert connection.execute("SHOW TABLES").fetchall() == [("competing_payload",)]
        assert connection.execute("SELECT sentinel FROM competing_payload").fetchall() == [(42,)]


@pytest.mark.parametrize(("sql", "code"), [
    ("UPDATE market_db_profile SET currency = 'EUR'", MARKET_DB_PROFILE_HASH_MISMATCH),
    ("UPDATE market_db_profile SET profile_hash = repeat('0', 64)", MARKET_DB_PROFILE_HASH_MISMATCH),
    ("UPDATE market_db_profile SET timezone = 'Unknown/Timezone'", MARKET_DB_PROFILE_INVALID),
    ("DELETE FROM market_db_profile", MARKET_DB_PROFILE_INVALID),
    ("CREATE OR REPLACE TABLE market_db_profile AS SELECT * FROM market_db_profile "
     "UNION ALL SELECT * FROM market_db_profile", MARKET_DB_PROFILE_INVALID),
])
def test_tampered_table_is_not_allowed_to_fall_back_to_sidecar(tmp_path, profile, sql, code):
    db = tmp_path / "tampered.duckdb"
    init_market_db(db, profile)
    adopt_market_db_sidecar(db, profile)
    with duckdb.connect(str(db)) as conn:
        conn.execute(sql)
    with pytest.raises(MarketDbProfileError, match=code):
        with open_market_db(db):
            pytest.fail("Damaged profile should never be yielded")


def test_migration_archives_the_entire_old_profile_and_rolls_back(tmp_path, profile, monkeypatch):
    db = tmp_path / "migrate.duckdb"
    init_market_db(db, profile)
    revised = replace(profile, default_execution_profile="open_to_open", profile_hash="")
    assert migrate_market_db_profile(db, revised) == profile
    assert migrate_market_db_profile(db, revised) == revised
    with open_market_db(db) as (conn, current, source):
        assert (current, source) == (revised, "table")
        migrated_at, old_hash, old_json = conn.execute(
            "SELECT migrated_at_utc, profile_hash, previous_profile FROM market_db_profile_history"
        ).fetchone()
        assert migrated_at is not None and old_hash == profile.profile_hash
        assert json.loads(old_json) == profile.as_dict()
        assert conn.execute("SELECT count(*) FROM market_db_profile_history").fetchone() == (1,)

    def fail_insert(*args):
        raise RuntimeError("injected insert failure")

    monkeypatch.setattr(storage, "_insert_profile", fail_insert)
    with pytest.raises(RuntimeError, match="injected insert failure"):
        migrate_market_db_profile(db, profile)
    with open_market_db(db) as (conn, current, _):
        assert current == revised
        assert conn.execute("SELECT count(*) FROM market_db_profile_history").fetchone() == (1,)


def test_migration_cannot_change_the_domain_or_database_id(tmp_path, profile):
    db = tmp_path / "identity.duckdb"
    init_market_db(db, profile)
    custom = replace(profile, market_domain="custom:test_market", profile_hash="")
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_DOMAIN_MISMATCH):
        migrate_market_db_profile(db, custom)
    other_id = replace(profile, db_id="another_db", profile_hash="")
    with pytest.raises(MarketDbProfileError, match="preserve db_id"):
        migrate_market_db_profile(db, other_id)


def test_sidecar_leaves_legacy_db_unchanged_and_has_exact_fingerprint(unprofiled_db, profile):
    def snapshot():
        stat = unprofiled_db.stat()
        return stat.st_size, stat.st_mtime_ns, hashlib.sha256(unprofiled_db.read_bytes()).hexdigest()

    original = snapshot()
    sidecar = adopt_market_db_sidecar(unprofiled_db, profile)
    assert sidecar == market_db_sidecar_path(unprofiled_db)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload.pop("db_fingerprint") == {
        "size_bytes": original[0],
        "mtime_ns": original[1],
        "header_sha256": hashlib.sha256(unprofiled_db.read_bytes()[:65536]).hexdigest(),
    }
    assert payload == profile.as_dict()
    with open_market_db(unprofiled_db) as (conn, loaded, source):
        assert (loaded, source) == (profile, "sidecar")
        assert conn.execute("SELECT value FROM fixture").fetchone() == (42,)
    with pytest.raises(MarketDbProfileError, match="opened read-only"):
        with open_market_db(unprofiled_db, read_only=False):
            pytest.fail("Sidecar database should be read-only")
    with pytest.raises(MarketDbProfileError, match="migration requires a table"):
        migrate_market_db_profile(unprofiled_db, profile)
    assert snapshot() == original


def test_changed_database_invalidates_sidecar(unprofiled_db, profile):
    adopt_market_db_sidecar(unprofiled_db, profile)
    stat = unprofiled_db.stat()
    os.utime(unprofiled_db, ns=(stat.st_atime_ns, stat.st_mtime_ns - 1_000_000_000))
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH):
        with open_market_db(unprofiled_db):
            pytest.fail("Stale sidecar should never be yielded")
    adopt_market_db_sidecar(unprofiled_db, profile)
    with open_market_db(unprofiled_db) as (_, loaded, _):
        assert loaded == profile


def test_concurrent_adoption_preserves_the_first_published_profile(unprofiled_db, profile):
    alternative = replace(profile, currency="EUR", profile_hash="")

    def adopt(candidate):
        try:
            adopt_market_db_sidecar(unprofiled_db, candidate)
        except MarketDbProfileError as exc:
            assert exc.code == MARKET_DB_PROFILE_HASH_MISMATCH
            return None
        return candidate

    with ThreadPoolExecutor(max_workers=2) as executor:
        published = [candidate for candidate in executor.map(adopt, (profile, alternative)) if candidate]
    assert len(published) == 1
    with open_market_db(unprofiled_db) as (_, loaded, _):
        assert loaded == published[0]


def test_dual_profile_sources_must_agree(tmp_path, profile):
    db = tmp_path / "dual.duckdb"
    init_market_db(db, profile)
    sidecar = adopt_market_db_sidecar(db, profile)
    with open_market_db(db) as (_, loaded, source):
        assert (loaded, source) == (profile, "table")
    revised = replace(profile, currency="EUR", profile_hash="")
    with pytest.raises(MarketDbProfileError, match="without a sidecar"):
        migrate_market_db_profile(db, revised)
    with pytest.raises(MarketDbProfileError, match="Table and proposed sidecar disagree"):
        adopt_market_db_sidecar(db, revised)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.write_text(json.dumps(profile.as_dict()), encoding="utf-8")
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_SIDECAR_FINGERPRINT_MISMATCH):
        with open_market_db(db):
            pytest.fail("Malformed sidecar should never be accepted")
    payload.update(revised.as_dict())
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MarketDbProfileError, match="Table and sidecar disagree"):
        with open_market_db(db):
            pytest.fail("Conflicting sidecar should never be accepted")


def test_missing_profile_fails_closed_and_closes_the_connection(unprofiled_db, profile):
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_PROFILE_MISSING):
        with open_market_db(unprofiled_db):
            pytest.fail("Missing profile should never be accepted")
    # The failed read handle must be closed for DuckDB to allow a write connection.
    with duckdb.connect(str(unprofiled_db)) as conn:
        assert conn.execute("SELECT value FROM fixture").fetchone() == (42,)
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_PROFILE_MISSING):
        migrate_market_db_profile(unprofiled_db, profile)


def test_market_db_bindings_and_expected_declaration(tmp_path, monkeypatch, profile):
    db = tmp_path / "bound.duckdb"
    init_market_db(db, profile)
    bindings = {"input_db": "input.duckdb", "output_db": "output.duckdb",
                "market_dbs": {profile.db_id: str(db), "wrong_id": str(db)}}
    assert validate_binding_contract(bindings) == bindings
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(bindings))
    assert resolve_market_db_path(profile.db_id) == db
    assert resolve_market_db_path(db, {}) == db
    with open_market_db(profile.db_id, expected_profile_yaml=TEMPLATES / "cn_equity.yaml") as (_, stored, _):
        assert stored == profile
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_BINDING_MISSING):
        resolve_market_db_path(profile.db_id, {})
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_BINDING_MISMATCH):
        with open_market_db("wrong_id"):
            pytest.fail("Binding ID must match the physical database declaration")
    with pytest.raises(MarketDbProfileError, match="Database and expected profile disagree"):
        with open_market_db(profile.db_id, expected_profile_yaml=TEMPLATES / "us_equity.yaml"):
            pytest.fail("Expected declaration must match the physical database")


@pytest.mark.parametrize("value", [None, [], {"../escape": "ok.duckdb"}, {"valid": "  "}, {"valid": 42}])
def test_malformed_market_db_bindings_are_rejected(value):
    with pytest.raises(MarketDbProfileError, match="market_dbs must map"):
        validate_binding_contract({"market_dbs": value})


@pytest.mark.parametrize("with_frozen_bindings", [False, True])
def test_market_mapping_coexists_with_legacy_research(tmp_path, unprofiled_db, profile, monkeypatch, with_frozen_bindings):
    from factor_synthesis.vectorbt_adapter.factor_loader import FactorLoader
    from utils.duckdb_adapter import get_data_adapter
    from utils.factor_base import FactorBase
    from utils.market_factor import MarketFactorValueStore

    db = tmp_path / "bound.duckdb"
    init_market_db(db, profile)
    market_mapping = {"market_dbs": {profile.db_id: str(db)}}
    frozen = {"input_db": str(unprofiled_db), "output_db": str(tmp_path / "output.duckdb"),
              "inputs": {}, "outputs": {}} if with_frozen_bindings else {}
    contract = {**market_mapping, **frozen}
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(contract))
    with open_market_db(profile.db_id) as (_, actual, kind):
        assert (actual, kind) == (profile, "table")
    config = {"database": {"duckdb_path": "legacy.duckdb"}}
    resolved = resolve_database_config(config)
    assert resolve_connection_binding(str(unprofiled_db), contract) == (str(unprofiled_db), frozen)
    store = MarketFactorValueStore.beside_results_database(unprofiled_db)
    store.identities_path.parent.mkdir()
    store.identities_path.touch()
    loader = FactorLoader(str(unprofiled_db))
    if with_frozen_bindings:
        assert loader._market_store is None
        assert resolved["database"]["duckdb_path"] == str(unprofiled_db)
        assert resolved["data_bindings"] == frozen
        with pytest.raises(ValueError, match="no frozen binding"):
            resolve_connection_binding(db, contract)
    else:
        assert resolved is config
        assert get_data_adapter() is None
        assert loader.cache_revision()[0] == "market_store"
        with duckdb.connect(":memory:") as conn:
            factor = FactorBase(conn, "legacy", "2026-01-05", "2026-01-06", stock_pool=["000001.XSHE"])
            factor.close()
