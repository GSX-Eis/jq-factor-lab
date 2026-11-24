from __future__ import annotations

import errno
import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest

from tools.agent_harness import cli, runner
from utils.market_db import load_market_db_profile_yaml, market_db_sidecar_path, open_market_db
from utils.market_db import storage
from utils.market_db.errors import MARKET_DB_ATOMIC_PUBLISH_UNSUPPORTED

TEMPLATE = Path(__file__).resolve().parents[1] / "config/market_databases/us_equity.yaml"


@pytest.fixture
def run_market_db(tmp_path, capsys):
    def run(*args, code=0, json_out=None):
        output = tmp_path / "out"
        result_file = tmp_path / "result.json" if json_out is None else json_out
        if json_out is None:
            result_file.unlink(missing_ok=True)
        result = cli.main([
            "market-db", *map(str, args), "--run-id", "s7-1", "--output-root", str(output),
            "--json-out", str(result_file),
        ])
        payload = json.loads(capsys.readouterr().out)
        assert result == payload["exit_code"] == code
        assert payload["command"]["name"] == "market-db"
        assert payload["command"]["run_id"] == "s7-1"
        assert Path(payload["summary"]["summary_path"]).is_relative_to(output)
        if json_out is None and result_file.exists():
            assert payload == json.loads(result_file.read_text(encoding="utf-8"))
            summary = Path(payload["summary"]["summary_path"])
            assert json.loads(summary.read_text(encoding="utf-8")) == payload
            assert json.loads((output / "market-db/s7-1/summary.json").read_text(encoding="utf-8")) == payload
        elif code == 0:
            pytest.fail("Successful command did not write its Harness summaries")
        return payload

    return run


def test_cli_init_show_and_audited_migration(tmp_path, monkeypatch, run_market_db):
    db = tmp_path / "market.duckdb"
    expected = load_market_db_profile_yaml(TEMPLATE)
    init = run_market_db("init", "--db", db, "--profile", TEMPLATE)
    assert init["metrics"]["source_kind"] == "table"
    assert init["metrics"]["profile_hash"] == expected.profile_hash
    second = run_market_db("init", "--db", db, "--profile", TEMPLATE, code=2)
    assert second["error"]["code"] == "MARKET_DB_ALREADY_INITIALIZED"

    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"market_dbs": {expected.db_id: str(db)}}))
    shown = run_market_db("show", "--market-db", expected.db_id)
    assert shown["metrics"]["profile"] == expected.as_dict()
    assert shown["metrics"]["validation_status"] == "passed"

    updated = tmp_path / "updated.json"
    payload = expected.as_dict()
    del payload["profile_hash"]
    payload["default_execution_profile"] = "open_to_open"
    updated.write_text(json.dumps(payload), encoding="utf-8")
    migrated = run_market_db("migrate-profile", "--db", db, "--profile", updated)
    assert migrated["metrics"]["previous_profile_hash"] == expected.profile_hash
    assert migrated["metrics"]["profile_changed"] is True
    assert migrated["metrics"]["profile_hash"] != expected.profile_hash
    with open_market_db(db) as (conn, current, _):
        assert current.default_execution_profile == "open_to_open"
        assert conn.execute("SELECT profile_hash FROM market_db_profile_history").fetchone() == (expected.profile_hash,)


def test_cli_adopt_and_show_legacy_database(tmp_path, run_market_db):
    db = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE keep_me (id INTEGER)")
    before = db.read_bytes(), db.stat().st_mtime_ns
    missing = run_market_db("show", "--db", db, code=2)
    assert missing["error"]["code"] == "MARKET_DB_PROFILE_MISSING"
    adopted = run_market_db("adopt", "--db", db, "--profile", TEMPLATE, "--sidecar")
    assert adopted["metrics"]["sidecar_path"] == str(market_db_sidecar_path(db))
    shown = run_market_db("show", "--db", db)
    assert shown["metrics"]["source_kind"] == "sidecar"
    assert shown["metrics"]["profile_hash"] == adopted["metrics"]["profile_hash"]
    assert (db.read_bytes(), db.stat().st_mtime_ns) == before


def test_cli_dry_run_checks_all_operations_without_writing_the_database(tmp_path, run_market_db):
    db = tmp_path / "market.duckdb"
    plan = run_market_db("init", "--db", db, "--profile", TEMPLATE, "--dry-run")
    assert plan["summary"]["status"] == "dry-run"
    assert not db.exists()
    run_market_db("init", "--db", db, "--profile", TEMPLATE)
    before = db.read_bytes(), db.stat().st_mtime_ns

    updated = tmp_path / "updated.json"
    profile = load_market_db_profile_yaml(TEMPLATE)
    payload = profile.as_dict()
    del payload["profile_hash"]
    payload["default_execution_profile"] = "open_to_open"
    updated.write_text(json.dumps(payload), encoding="utf-8")

    run_market_db("migrate-profile", "--db", db, "--profile", updated, "--dry-run")
    run_market_db("adopt", "--db", db, "--profile", TEMPLATE, "--sidecar", "--dry-run")
    shown = run_market_db("show", "--db", db, "--dry-run")
    assert shown["metrics"]["profile_hash"] == profile.profile_hash
    assert not market_db_sidecar_path(db).exists()
    assert (db.read_bytes(), db.stat().st_mtime_ns) == before
    with duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'market_db_profile_history'"
        ).fetchone() == (0,)


@pytest.mark.parametrize("dry_run", [False, True])
def test_init_rejects_known_unsupported_mac_volume_before_staging(tmp_path, monkeypatch, run_market_db, dry_run):
    target_dir = tmp_path / "volume"
    target_dir.mkdir()
    db = target_dir / "market.duckdb"

    def unsupported_volume(libc, directory):
        assert directory == target_dir
        raise OSError(errno.ENOTSUP, "exclusive rename unavailable")

    def fail_if_staged(*args, **kwargs):
        pytest.fail("Unsupported volume must be rejected before staging")

    monkeypatch.setattr(storage, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(storage, "_require_darwin_exclusive_rename", unsupported_volume)
    monkeypatch.setattr(storage, "TemporaryDirectory", fail_if_staged)
    flags = ["--dry-run"] if dry_run else []
    result = run_market_db("init", "--db", db, "--profile", TEMPLATE, *flags, code=2)
    assert result["error"]["code"] == MARKET_DB_ATOMIC_PUBLISH_UNSUPPORTED
    assert "APFS" in result["metrics"]["message"]
    assert result["metrics"]["operation_committed"] is False
    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert not list(target_dir.iterdir())


@pytest.mark.parametrize("command", [
    ["adopt", "--db", "legacy.duckdb", "--profile", "a.yaml"],
    ["show"],
    ["show", "--db", "example.duckdb", "--market-db", "other"],
])
def test_cli_rejects_ambiguous_or_unsafe_arguments(command):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["market-db", *command])
    assert exc.value.code == 2


def test_cli_result_cannot_replace_the_database(tmp_path, run_market_db):
    db = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE keep_me (id INTEGER)")
    before = db.read_bytes(), db.stat().st_mtime_ns
    result = run_market_db("adopt", "--db", db, "--profile", TEMPLATE, "--sidecar", json_out=db, code=2)
    assert result["error"]["code"] == "MARKET_DB_PROFILE_INVALID"
    assert (db.read_bytes(), db.stat().st_mtime_ns) == before
    assert not market_db_sidecar_path(db).exists()


@pytest.mark.parametrize("invalid_destination", ["directory", "file_parent", "blocked_summary"])
def test_cli_rejects_invalid_output_before_initializing(tmp_path, capsys, invalid_destination):
    db, output, result_file = (tmp_path / name for name in ("new.duckdb", "out", "out.json"))
    if invalid_destination == "directory":
        result_file.mkdir()
    elif invalid_destination == "file_parent":
        result_file.touch()
        result_file /= "report.json"
    else:
        (output / "market-db/run/summary.json").mkdir(parents=True)
    code = cli.main(["market-db", "init", "--db", str(db), "--profile", str(TEMPLATE), "--run-id", "run",
                     "--output-root", str(output), "--json-out", str(result_file)])
    payload = json.loads(capsys.readouterr().out)
    assert code == payload["exit_code"] == 2
    assert payload["metrics"]["operation_committed"] is False
    assert not db.exists()


@pytest.mark.parametrize("fail_summary", [False, True])
def test_cli_reports_commit_if_report_write_fails(tmp_path, monkeypatch, run_market_db, fail_summary):
    db, destination = tmp_path / "committed.duckdb", tmp_path / "failed.json"
    latest = tmp_path / "out/market-db/s7-1/summary.json"
    failing_target = latest if fail_summary else destination
    surviving_target = destination if fail_summary else latest
    write_json = runner._write_json

    def fail_requested_output(path, payload):
        if path == failing_target:
            raise OSError("injected report write failure")
        write_json(path, payload)

    monkeypatch.setattr(runner, "_write_json", fail_requested_output)
    payload = run_market_db("init", "--db", db, "--profile", TEMPLATE, code=3, json_out=destination)
    assert payload["error"]["code"] == "MARKET_DB_OUTPUT_WRITE_FAILED"
    assert payload["metrics"]["operation_committed"] is True
    assert json.loads(Path(payload["summary"]["summary_path"]).read_text()) == payload
    assert json.loads(surviving_target.read_text()) == payload
    with open_market_db(db) as (_, installed, _):
        assert installed == load_market_db_profile_yaml(TEMPLATE)


@pytest.mark.parametrize("binding_state", ["valid", "missing_file", "invalid_database", "unbound"])
def test_cli_show_by_id_reports_identity_without_binding_path(tmp_path, monkeypatch, run_market_db, binding_state):
    db = tmp_path / "binding-secret-location.duckdb"
    profile = load_market_db_profile_yaml(TEMPLATE)
    expected_codes = {
        "missing_file": "MARKET_DB_PROFILE_MISSING",
        "invalid_database": "MARKET_DB_PROFILE_INVALID",
        "unbound": "MARKET_DB_BINDING_MISSING",
    }
    if binding_state == "valid":
        run_market_db("init", "--db", db, "--profile", TEMPLATE)
    elif binding_state == "invalid_database":
        db.write_text("invalid DuckDB")
    mapping = {} if binding_state == "unbound" else {profile.db_id: str(db)}
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"market_dbs": mapping}))
    payload = run_market_db("show", "--market-db", profile.db_id, code=0 if binding_state == "valid" else 2)
    assert payload["metrics"]["db_id"] == profile.db_id
    if binding_state == "valid":
        assert payload["metrics"]["profile_hash"] == profile.profile_hash
    else:
        assert payload["error"]["code"] == expected_codes[binding_state]
    for path in (tmp_path / "result.json", Path(payload["summary"]["summary_path"]),
                 tmp_path / "out/market-db/s7-1/summary.json"):
        assert path.is_file()
        assert db.name not in path.read_text()
    assert db.name not in json.dumps(payload)
