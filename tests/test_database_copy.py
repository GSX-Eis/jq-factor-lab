import json
import subprocess
from pathlib import Path

import duckdb
import pytest
from utils import database_copy


def test_physical_copy_resumes_and_rejects_changed_source(tmp_path, monkeypatch):
    source, target = tmp_path / "source.duckdb", tmp_path / "target.duckdb"
    with duckdb.connect(str(source)) as conn:
        conn.execute("CREATE TABLE prices (id INTEGER PRIMARY KEY, price DOUBLE)")
        conn.execute("INSERT INTO prices SELECT range, range / 10.0 FROM range(10000)")
    original = source.read_bytes()
    run = subprocess.run

    def interrupted(command, **kwargs):
        Path(command[-1]).write_bytes(original[: len(original) // 2])
        return subprocess.CompletedProcess(command, 1, "", "simulated power interruption")

    monkeypatch.setattr(subprocess, "run", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        database_copy.clone_database(source, target, execute=True)
    assert not target.exists()
    partial = Path(str(target) + ".partial")
    assert partial.stat().st_size == len(original) // 2
    assert database_copy.clone_database(source, target)["partial_bytes"] == len(original) // 2

    def resumed(command, **kwargs):
        assert partial.stat().st_size == len(original) // 2
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", resumed)
    result = database_copy.clone_database(source, target, execute=True)
    assert result["ok"] and result["phase"] == "complete"
    assert target.read_bytes() == original == source.read_bytes()
    assert not partial.exists()
    with duckdb.connect(str(target), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM prices").fetchone()[0] == 10000
        assert (
            conn.execute("SELECT count(*) FROM duckdb_constraints() WHERE constraint_type='PRIMARY KEY'").fetchone()[0]
            == 1
        )
    receipt = Path(str(target) + ".copy.json")
    receipt.write_text(json.dumps({**result, "phase": "verified", "ok": False}))
    resumed = database_copy.clone_database(source, target, execute=True)
    assert resumed["sha256"] == result["sha256"] and resumed["phase"] == "complete"
    with duckdb.connect(str(source)) as conn:
        conn.execute("INSERT INTO prices VALUES (10001,1)")
    with pytest.raises(ValueError, match="snapshot changed"):
        database_copy.clone_database(source, target, execute=True)
    assert json.loads(Path(str(target) + ".copy.json").read_text())["phase"] == "complete"


def test_physical_copy_verification_prevents_publishing_corruption(tmp_path, monkeypatch):
    source, target = tmp_path / "source.duckdb", tmp_path / "target.duckdb"
    with duckdb.connect(str(source)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")

    def corrupted(command, **kwargs):
        data = bytearray(source.read_bytes())
        data[-1] ^= 1
        Path(command[-1]).write_bytes(data)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", corrupted)
    with pytest.raises(ValueError, match="checksum mismatch"):
        database_copy.clone_database(source, target, execute=True)
    assert not target.exists()


def test_cloned_table_registration_never_falls_back_to_row_copy(tmp_path):
    from utils.data_store import migrate_table

    source, target = tmp_path / "source.duckdb", tmp_path / "target.duckdb"
    with duckdb.connect(str(source)) as conn:
        conn.execute("CREATE TABLE prices (id INTEGER PRIMARY KEY, price DOUBLE)")
        conn.execute("INSERT INTO prices VALUES (1,10)")
    database_copy.clone_database(source, target, execute=True)
    registered = migrate_table(source, target, "prices", role="input", execute=True, verify_existing=True)
    assert registered["physical_table"] == "main.prices"
    with duckdb.connect(str(source)) as conn:
        conn.execute("UPDATE prices SET price=20")
    with pytest.raises(ValueError, match="differs"):
        migrate_table(source, target, "prices", role="input", execute=True, verify_existing=True)
    with duckdb.connect(str(target), read_only=True) as conn:
        assert conn.execute("SELECT price FROM prices").fetchone()[0] == 10
        assert conn.execute("SELECT count(*) FROM meta.datasets").fetchone()[0] == 1


def test_resume_after_transfer_does_not_transfer_again(tmp_path, monkeypatch):
    source, target = tmp_path / "source.duckdb", tmp_path / "target.duckdb"
    with duckdb.connect(str(source)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER)")
    file_digest = database_copy.hashlib.file_digest

    def interrupted(*args):
        raise RuntimeError("interrupted during verification")

    monkeypatch.setattr(database_copy.hashlib, "file_digest", interrupted)
    with pytest.raises(RuntimeError, match="verification"):
        database_copy.clone_database(source, target, execute=True)
    assert json.loads(Path(str(target) + ".copy.json").read_text())["phase"] == "verifying"
    monkeypatch.setattr(database_copy.hashlib, "file_digest", file_digest)

    def must_not_retransfer(*args, **kwargs):
        raise AssertionError("Already completed transfer was restarted")

    monkeypatch.setattr(subprocess, "run", must_not_retransfer)
    assert database_copy.clone_database(source, target, execute=True)["ok"]
