import hashlib
import json
from pathlib import Path

import duckdb
import pytest
from utils import data_store as store


@pytest.mark.parametrize("execute", [False, True])
def test_adopt_document_without_original_file_is_rejected(tmp_path, execute):
    output = tmp_path / "output.duckdb"
    ref = store.publish_document(output, {"value": 7}, metadata={})
    with pytest.raises(ValueError, match="Artifact original_file must identify a regular file"):
        store.adopt_artifact_files(output, [ref.version], execute=execute)


@pytest.mark.parametrize("valid_first", [False, True])
@pytest.mark.parametrize("stale_checkpoint", [False, True])
def test_publish_batch_checkpoint_includes_missing_file(tmp_path, valid_first, stale_checkpoint):
    source = tmp_path / "valid.json"
    source.write_text('{"value": 1}')
    checkpoint = tmp_path / "progress.json"
    if stale_checkpoint:
        checkpoint.write_text('[{"file":"old","ok":true}]')
    missing = {"file": str(tmp_path / "missing.json"), "metadata": {"run_id": "toy"}}
    items = ([{"file": str(source), "metadata": {"run_id": "toy"}}] if valid_first else []) + [missing]
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(json.dumps({"action": "publish_batch", "path": str(tmp_path / "output.duckdb"),
                                   "items": items, "checkpoint": str(checkpoint)}))
    assert store.main(["--request", str(request), "--response", str(response)]) == 1
    result = json.loads(response.read_text())
    assert result["ok"] is False
    assert [row["file"] for row in result["results"]] == [row["file"] for row in items]
    assert [row["ok"] for row in result["results"]] == ([True] if valid_first else []) + [False]
    assert result["results"][-1]["error"] == "frozen artifact missing"
    assert checkpoint.read_text() == store.canonical(result["results"]) + "\n"


def test_custody_preserves_versions_and_retries_without_originals(tmp_path, monkeypatch):
    output = tmp_path / "new" / "output.duckdb"
    output.parent.mkdir()
    document, values, binary = [tmp_path / name for name in ("metrics.json", "values.csv", "model.bin")]
    document.write_text('{"score": 0.5}')
    values.write_text("date,value\n2019-01-02,2\n")
    binary.write_bytes(b"\x00\xff")
    # Simulate the historical publisher's external file references.
    with monkeypatch.context() as patch:
        patch.setattr(store, "freeze_artifact_file", lambda path, file: Path(file).resolve())
        refs = [
            store.publish_document(output, {"score": 0.5}, metadata={"run_id": "a"}, original_file=str(document)),
            store.publish_artifact(output, "features.values", file=values, metadata={"run_id": "a"}),
            store.publish_attachment(output, binary, metadata={"run_id": "a"}),
        ]
    ids = [ref.version for ref in refs]

    def catalog():
        with duckdb.connect(str(output), read_only=True) as conn:
            return conn.execute("SELECT * REPLACE (CAST(published_at AS VARCHAR) AS published_at) FROM meta.artifacts ORDER BY artifact_id").fetchall()

    before = catalog()
    entry = {"action": "adopt_files", "source": str(output), "artifact_ids": ids}
    assert store.migrate_tables([entry])["ok"]
    assert not (output.parent / "artifacts").exists()
    values.write_text("date,value\n2019-01-02,999\n")
    with pytest.raises(ValueError, match="differs from published"):
        store.migrate_tables([entry], execute=True)
    assert catalog() == before  # Failed batches never publish partial location updates.
    values.write_text("date,value\n2019-01-02,2\n")
    result = store.migrate_tables([entry], execute=True)
    assert result["results"][0]["artifacts"] == 3
    after = catalog()
    assert [[*row[:6], *row[7:]] for row in before] == [[*row[:6], *row[7:]] for row in after]
    for source in (document, values, binary):
        source.unlink()
    assert store.migrate_tables([entry], execute=True)["ok"]
    assert catalog() == after
    assert store.read_document(output, refs[0]) == {"score": 0.5}
    assert store.read_reference(output, refs[1])["value"].tolist() == [2]
    for row in result["results"][0]["files"]:
        assert Path(row["frozen_file"]).is_file()
    frozen = Path(next(row[6] for row in after if row[0] == refs[0].version))
    frozen.write_text('{"score": 0.5} ')  # Even semantically equal JSON violates frozen bytes.
    with pytest.raises(ValueError, match="content changed"):
        store.migrate_tables([entry], execute=True)
    assert catalog() == after


def test_new_publication_owns_files_and_rejects_mismatched_document(tmp_path):
    output = tmp_path / "output.duckdb"
    document = tmp_path / "document.json"
    document.write_text('{"value": 1}')
    with pytest.raises(ValueError, match="differs from its frozen file"):
        store.publish_document(output, {"value": 2}, metadata={}, original_file=str(document))
    assert not output.exists()
    ref = store.publish_document(output, json.loads(document.read_text()), metadata={}, original_file=str(document))
    document.unlink()
    with duckdb.connect(str(output), read_only=True) as conn:
        frozen = Path(store.resolve_artifact(conn, ref)["original_file"])
    assert frozen.read_text() == '{"value": 1}'
    assert store.read_document(output, ref) == {"value": 1}


