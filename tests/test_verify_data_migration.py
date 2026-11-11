import json
from pathlib import Path

import duckdb
from scripts.verify_data_migration import verify
from utils.data_store import migrate_table, publish_attachment, publish_document

from scripts import verify_data_migration


def test_target_only_audit_after_source_cleanup_still_detects_changed_data(tmp_path):
    import pytest

    old, inputs, outputs = [tmp_path / name for name in ('old.duckdb', 'input.duckdb', 'output.duckdb')]
    with duckdb.connect(str(old)) as conn:
        conn.execute('CREATE TABLE prices(value INTEGER PRIMARY KEY); INSERT INTO prices VALUES (1)')
    for target, role in ((inputs, 'input'), (outputs, 'output')):
        migrate_table(old, target, 'prices', role=role, execute=True)
    manifest, report = tmp_path / 'manifest.json', tmp_path / 'report.json'
    manifest.write_text(json.dumps({'tables': [
        {'source': str(old), 'table': 'prices', 'role': role} for role in ('input', 'output')]}))
    assert verify(inputs, outputs, manifest, report)['ok']
    old.unlink()
    with pytest.raises(FileNotFoundError):
        verify(inputs, outputs, manifest, report)
    assert verify(inputs, outputs, manifest, report, target_only=True)['ok']
    body = json.loads(report.read_text())
    assert body['scope'] == 'registered_targets_and_frozen_files'
    assert not body['deletion_authorized']
    assert not any(key.startswith(('source:', 'legacy-output:', 'input-contract:')) for key in body['checks'])
    with duckdb.connect(str(inputs)) as conn:
        conn.execute('UPDATE prices SET value=2')
    assert not verify(inputs, outputs, manifest, report, target_only=True)['ok']
    import subprocess
    import sys

    command = subprocess.run(
        [sys.executable, verify_data_migration.__file__, "--input-db", str(inputs),
         "--output-db", str(outputs), "--manifest", str(manifest), "--report", str(report),
         "--target-only"], capture_output=True, text=True,
    )
    assert command.returncode == 1, command.stderr
    assert json.loads(command.stdout.splitlines()[-1])["ok"] is False


def test_recheck_detects_changed_data_and_attachment_without_deleting(
    tmp_path, monkeypatch
):
    old, inputs, outputs = [
        tmp_path / name for name in ("old.duckdb", "input.duckdb", "output.duckdb")
    ]
    manifest, report = tmp_path / "manifest.json", tmp_path / "report.json"
    with duckdb.connect(str(old)) as conn:
        conn.execute("CREATE TABLE prices(code VARCHAR,date DATE,close DOUBLE)")
        conn.execute(
            "INSERT INTO prices VALUES ('a','2019-01-02',1.5),('b','2019-01-02',NULL)"
        )
    migrate_table(old, inputs, "prices", role="input", execute=True)
    migrate_table(old, outputs, "prices", role="output", execute=True)
    publish_document(outputs, {"value": 1}, metadata={"run_id": "fixture"})
    attachment = tmp_path / "source.txt"
    attachment.write_text("immutable")
    attachment_ref = publish_attachment(outputs, attachment, metadata={"run_id": "fixture"})
    with duckdb.connect(str(outputs), read_only=True) as conn:
        frozen_attachment = Path(conn.execute(
            "SELECT original_file FROM meta.artifacts WHERE artifact_id=?", [attachment_ref.version]
        ).fetchone()[0])
    manifest.write_text(
        json.dumps(
            {
                "tables": [
                    {"source": str(old), "table": "prices", "role": role}
                    for role in ["input", "output"]
                ]
            }
        )
    )
    assert verify(inputs, outputs, manifest, report)["ok"]
    assert verify(inputs, outputs, manifest, report)["ok"]
    audited = []
    original_audit = verify_data_migration.audit_relation

    def track_audit(conn, relation):
        audited.append(relation)
        return original_audit(conn, relation)

    monkeypatch.setattr(verify_data_migration, "audit_relation", track_audit)
    publish_document(outputs, {"value": 2}, metadata={"run_id": "fixture2"})
    assert verify(inputs, outputs, manifest, report)["ok"]
    # Only the output table and its legacy source are rescanned, never input prices.
    assert len(audited) == 2
    assert json.loads(report.read_text())["reused_checks"] == 3
    audited.clear()
    assert verify(inputs, outputs, manifest, report)["ok"]
    assert not audited
    # Manifest formatting/order does not change content dependencies; coverage is checked afresh.
    specification = json.loads(manifest.read_text())
    specification["tables"].reverse()
    manifest.write_text(json.dumps(specification, indent=2))
    assert verify(inputs, outputs, manifest, report)["ok"]
    assert not audited
    # A failed table scan remains retryable while completed scans keep their receipt.
    publish_document(outputs, {"value": 3}, metadata={"run_id": "fixture3"})
    fail_once = True

    def interrupted_audit(conn, relation):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("simulated scan interruption")
        return track_audit(conn, relation)

    monkeypatch.setattr(verify_data_migration, "audit_relation", interrupted_audit)
    assert not verify(inputs, outputs, manifest, report)["ok"]
    audited.clear()
    assert verify(inputs, outputs, manifest, report)["ok"]
    assert len(audited) == 1
    audited.clear()
    attachment.write_text("changed")
    assert verify(inputs, outputs, manifest, report)["ok"]  # Original is no longer authoritative.
    frozen_attachment.write_text("changed")
    assert not verify(inputs, outputs, manifest, report)["ok"]
    assert not audited
    frozen_attachment.write_text("immutable")
    assert verify(inputs, outputs, manifest, report)["ok"]
    with duckdb.connect(str(inputs)) as conn:
        conn.execute("UPDATE prices SET close=2 WHERE code='a'")
    assert not verify(inputs, outputs, manifest, report)["ok"]
    assert len(audited) == 2  # Input checkpoint change invalidates input/source checks.
    assert not json.loads(report.read_text())["deletion_authorized"]
    assert old.exists() and attachment.exists()


def test_recheck_rejects_changed_frozen_source_policy(tmp_path):
    old, inputs, outputs = [
        tmp_path / name for name in ("old.duckdb", "input.duckdb", "output.duckdb")
    ]
    with duckdb.connect(str(old)) as conn:
        conn.execute("CREATE TABLE prices(value INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO prices VALUES (1)")
    migrate_table(old, inputs, "prices", role="input", execute=True, follow_head=False)
    publish_document(outputs, {"value": 1}, metadata={"run_id": "fixture"})
    manifest, report = tmp_path / "manifest.json", tmp_path / "report.json"
    manifest.write_text(
        json.dumps(
            {
                "tables": [
                    {
                        "source": str(old),
                        "table": "prices",
                        "role": "input",
                        "follow_head": False,
                    }
                ]
            }
        )
    )
    assert verify(inputs, outputs, manifest, report)["ok"]
    with duckdb.connect(str(inputs)) as conn:
        conn.execute("UPDATE meta.datasets SET follow_head=TRUE")
    assert not verify(inputs, outputs, manifest, report)["ok"]
    result = json.loads(report.read_text())
    assert all(check["ok"] for check in result["checks"].values())
    assert result["input_policy_mismatches"] == [
        {
            "source": str(old),
            "table": "main.prices",
            "expected": False,
            "actual": [True],
        }
    ]
    with duckdb.connect(str(inputs)) as conn:
        conn.execute("UPDATE meta.datasets SET follow_head=FALSE")
        conn.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM prices")
    assert not verify(inputs, outputs, manifest, report)["ok"]
    result = json.loads(report.read_text())
    assert not result["input_policy_mismatches"]
    failed = [key for key, check in result["checks"].items() if not check["ok"]]
    assert failed == ["input-contract:" + str(old) + ":main.prices"]


def test_archive_migration_retains_original_table_and_deduplicates_sources(tmp_path):
    from utils.data_store import DataRef, read_reference

    sources = [tmp_path / name for name in ("old.duckdb", "frozen.duckdb")]
    inputs, outputs = tmp_path / "input.duckdb", tmp_path / "output.duckdb"
    entries, refs = [], []
    for index, source in enumerate(sources):
        with duckdb.connect(str(source)) as conn:
            conn.execute("CREATE TABLE price_backup(code VARCHAR NOT NULL, date DATE, close DOUBLE)")
            conn.execute("INSERT INTO price_backup VALUES ('a','2019-01-02',1.5),('a','2019-01-02',1.5),('b','2019-01-03',NULL)")
        result = migrate_table(source, inputs, "price_backup", role="input", target="archive.price_backup", execute=True, follow_head=False, chunked=not bool(index))
        refs.append(result["ref"])
        entries.append({"source": str(source), "table": "price_backup", "target": "archive.price_backup", "role": "input", "follow_head": False})
    assert refs[0] == refs[1]
    with duckdb.connect(str(inputs), read_only=True) as conn:
        assert conn.execute("SELECT DISTINCT source_table,target_table,physical_table,follow_head FROM meta.datasets").fetchall() == [("main.price_backup", "archive.price_backup", "archive.price_backup", False)]
        assert conn.execute("SELECT count(*) FROM archive.price_backup").fetchone() == (3,)
        assert conn.execute("SELECT count(*) FROM meta.datasets").fetchone() == (2,)
    publish_document(outputs, {}, metadata={"run_id": "archive-check"})
    manifest, report = tmp_path / "manifest.json", tmp_path / "report.json"
    manifest.write_text(json.dumps({"tables": entries}))
    assert verify(inputs, outputs, manifest, report)["ok"]
    with duckdb.connect(str(inputs)) as conn:
        conn.execute("UPDATE meta.datasets SET source_table='main.missing'")
    assert not verify(inputs, outputs, manifest, report)["ok"]
    assert json.loads(report.read_text())["missing_manifest_entries"] == entries
    for source in sources:
        source.unlink()
    assert len(read_reference(inputs, DataRef(**refs[0]))) == 3


def test_partitioned_parquet_migration_freezes_files_and_recovers_publication(tmp_path, monkeypatch):
    import pandas as pd
    import pytest
    import utils.data_store as store

    source = tmp_path / "factor_values"
    source.mkdir()
    values = pd.DataFrame({"date": pd.to_datetime(["2019-01-02 00:00:00.123456789"]), "stock_code": ["a"], "factor_name": ["same"], "factor_value": [1.5]})
    original = source / "part.parquet"
    values.to_parquet(original, index=False)
    old, inputs, outputs = [tmp_path / name for name in ("old.duckdb", "input.duckdb", "output.duckdb")]
    with duckdb.connect(str(old)) as conn:
        conn.execute("CREATE TABLE prices(value INTEGER); INSERT INTO prices VALUES (1)")
    migrate_table(old, inputs, "prices", role="input", execute=True)
    publish_document(outputs, {}, metadata={"run_id": "existing"})
    incompatible = source / "incompatible.parquet"
    values.assign(date=values.date.astype(str)).to_parquet(incompatible, index=False)
    with pytest.raises(ValueError, match="different schemas"):
        migrate_table(source, outputs, "factor_values", role="output", source_format="parquet", execute=True)
    incompatible.unlink()
    publish = store.publish_relation
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise RuntimeError("interrupted publication")
        patch.setattr(store, "publish_relation", fail)
        with pytest.raises(RuntimeError, match="interrupted publication"):
            migrate_table(source, outputs, "factor_values", role="output", source_format="parquet", execute=True)
    with duckdb.connect(str(outputs), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM meta.artifacts").fetchone() == (1,)
    result = migrate_table(source, outputs, "factor_values", role="output", source_format="parquet", execute=True)
    assert migrate_table(source, outputs, "factor_values", role="output", source_format="parquet", execute=True)["ref"] == result["ref"]
    assert store.publish_relation == publish
    with duckdb.connect(str(outputs), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM duckdb_views() WHERE view_name='_legacy_parquet'").fetchone() == (0,)
    manifest, report = tmp_path / "manifest.json", tmp_path / "report.json"
    manifest.write_text(json.dumps({"tables": [
        {"source": str(old), "table": "prices", "role": "input"},
        {"source": str(source), "table": "factor_values", "role": "output", "source_format": "parquet"},
    ]}))
    assert verify(inputs, outputs, manifest, report)["ok"]
    values.to_parquet(source / "added.parquet", index=False)
    assert not verify(inputs, outputs, manifest, report)["ok"]
    (source / "added.parquet").unlink()
    assert verify(inputs, outputs, manifest, report)["ok"]
    with duckdb.connect(str(outputs), read_only=True) as conn:
        metadata = json.loads(conn.execute("SELECT metadata_json FROM meta.artifacts WHERE artifact_id=?", [result["ref"]["version"]]).fetchone()[0])
    original.unlink()
    actual = store.read_reference(outputs, store.DataRef(**result["ref"]))
    assert actual.date.iloc[0] == values.date.iloc[0]
    frozen = Path(metadata["legacy_files"][0]["file"])
    assert pd.read_parquet(frozen).equals(values)
    frozen.write_bytes(b"corrupt")
    assert not verify(inputs, outputs, manifest, report)["ok"]
    assert not json.loads(report.read_text())["checks"]["frozen-parts:" + result["ref"]["version"]]["ok"]
