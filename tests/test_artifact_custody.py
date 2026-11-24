import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pytest
from utils import data_store as store
from utils import duckdb_results_writer as writer


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
    if not valid_first:
        assert not (tmp_path / "output.duckdb").exists()
        assert not (tmp_path / "artifacts").exists()


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




def test_frozen_batch_rejects_modified_source_during_freeze_without_leaving_copy(tmp_path, monkeypatch):
    output, source, request, response = [
        tmp_path / name for name in ("output.duckdb", "spec.json", "request.json", "response.json")
    ]
    source.write_text('{"value": 1}')
    item = {
        "file": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metadata": {"legacy_source": str(source)},
    }
    request.write_text(json.dumps({"action": "publish_batch", "path": str(output), "items": [item]}))
    real_freeze = store.freeze_artifact_file

    def swap_before_freeze(path, file, **kwargs):
        Path(file).write_text('{"value": 2}')
        return real_freeze(path, file, **kwargs)

    monkeypatch.setattr(store, "freeze_artifact_file", swap_before_freeze)
    assert store.main(["--request", str(request), "--response", str(response)]) == 1
    assert not output.exists()
    assert not (output.parent / "artifacts").exists()
    failed = json.loads(response.read_text())
    assert failed["ok"] is False
    assert failed["results"][0]["file"] == str(source)
    assert failed["results"][0]["ok"] is False
    assert failed["results"][0]["error"] == "Frozen import file does not match the declared checksum"


def test_frozen_batch_rejects_invalid_json_before_freezing_copy(tmp_path):
    output, source, request, response = [
        tmp_path / name for name in ("output.duckdb", "broken.json", "request.json", "response.json")
    ]
    source.write_text("{")
    item = {
        "file": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metadata": {"run_id": "toy"},
    }
    request.write_text(json.dumps({"action": "publish_batch", "path": str(output), "items": [item]}))
    assert store.main(["--request", str(request), "--response", str(response)]) == 1
    assert not output.exists()
    assert not (output.parent / "artifacts").exists()
    failed = json.loads(response.read_text())
    assert failed["ok"] is False
    assert failed["results"][0]["file"] == str(source)
    assert failed["results"][0]["ok"] is False
    assert failed["results"][0]["error"]


@pytest.mark.parametrize("target_kind", ["sqlite", "corrupt_duckdb"])
def test_publish_batch_aborts_on_target_store_failure_without_checkpoint_or_frozen_copies(tmp_path, target_kind):
    import sqlite3

    output = tmp_path / ("legacy.sqlite" if target_kind == "sqlite" else "corrupt.duckdb")
    if target_kind == "sqlite":
        conn = sqlite3.connect(str(output))
        conn.execute("CREATE TABLE legacy (id INTEGER)")
        conn.commit()
        conn.close()
        expected_exc = ValueError
    else:
        output.write_bytes(b"not a valid duckdb file")
        expected_exc = duckdb.Error

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text('{"value": 1}')
    second.write_text('{"value": 2}')
    items = [
        {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": {"run_id": path.stem},
        }
        for path in (first, second)
    ]
    checkpoint, request, response = tmp_path / "progress.json", tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "action": "publish_batch",
                "path": str(output),
                "items": items,
                "checkpoint": str(checkpoint),
            }
        )
    )
    with pytest.raises(expected_exc):
        store.main(["--request", str(request), "--response", str(response)])
    assert not response.exists()
    assert not checkpoint.exists()
    assert not (output.parent / "artifacts").exists()


def test_frozen_batch_rejects_explicit_null_sha256_before_publication(tmp_path):
    output, source, checkpoint, request, response = [
        tmp_path / name for name in ("output.duckdb", "spec.json", "progress.json", "request.json", "response.json")
    ]
    source.write_text('{"value": 1}')
    args = ["--request", str(request), "--response", str(response)]
    request.write_text(
        json.dumps(
            {
                "action": "publish_batch",
                "path": str(output),
                "items": [{"file": str(source), "sha256": None, "metadata": {"run_id": "toy"}}],
                "checkpoint": str(checkpoint),
            }
        )
    )
    assert store.main(args) == 1
    assert not output.exists()
    assert not (output.parent / "artifacts").exists()
    failed = json.loads(response.read_text())
    assert failed["ok"] is False
    assert failed["results"] == [
        {
            "file": str(source),
            "ok": False,
            "error": "Frozen import file does not match the declared checksum",
        }
    ]
    assert checkpoint.read_text() == store.canonical(failed["results"]) + "\n"

    # Omitting sha256 preserves compatibility publication.
    request.write_text(
        json.dumps(
            {
                "action": "publish_batch",
                "path": str(output),
                "items": [{"file": str(source), "metadata": {"run_id": "toy"}}],
                "checkpoint": str(checkpoint),
            }
        )
    )
    assert store.main(args) == 0
    assert output.is_file()
    succeeded = json.loads(response.read_text())
    assert succeeded["ok"] is True
    assert succeeded["results"][0]["file"] == str(source)


@pytest.mark.parametrize("bad_first", [False, True])
def test_publish_batch_aborts_before_freezing_when_new_store_lock_is_directory(tmp_path, bad_first):
    output, first, source, checkpoint, request, response = [
        tmp_path / name
        for name in ("output.duckdb", "first.json", "spec.json", "progress.json", "request.json", "response.json")
    ]
    first.write_text('{"value": 0}')
    source.write_text('{"value": 1}')
    (tmp_path / "output.duckdb.write.lock").mkdir()
    items = ([{"file": str(first), "sha256": "0" * 64, "metadata": {"run_id": "bad"}}] if bad_first else []) + [
        {
            "file": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "metadata": {"run_id": "toy"},
        }
    ]
    request.write_text(
        json.dumps(
            {
                "action": "publish_batch",
                "path": str(output),
                "items": items,
                "checkpoint": str(checkpoint),
            }
        )
    )
    with pytest.raises(IsADirectoryError):
        store.main(["--request", str(request), "--response", str(response)])
    assert not output.exists()
    assert not response.exists()
    assert not checkpoint.exists()
    assert not (tmp_path / "artifacts").exists()


def test_publish_batch_aborts_on_corrupt_store_before_writing_earlier_item_checkpoint(tmp_path):
    output, first, second, checkpoint, request, response = [
        tmp_path / name
        for name in ("corrupt.duckdb", "first.json", "second.json", "progress.json", "request.json", "response.json")
    ]
    output.write_bytes(b"not a valid duckdb file")
    first.write_text('{"value": 1}')
    second.write_text('{"value": 2}')
    request.write_text(
        json.dumps(
            {
                "action": "publish_batch",
                "path": str(output),
                "items": [
                    {"file": str(first), "sha256": "0" * 64, "metadata": {"run_id": "first"}},
                    {
                        "file": str(second),
                        "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                        "metadata": {"run_id": "second"},
                    },
                ],
                "checkpoint": str(checkpoint),
            }
        )
    )
    with pytest.raises(duckdb.IOException):
        store.main(["--request", str(request), "--response", str(response)])
    assert not response.exists()
    assert not checkpoint.exists()
    assert not (tmp_path / "artifacts").exists()


def _write_batch_request(request_path: Path, output_path: Path, source_path: Path, run_id: str) -> str:
    sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    request_path.write_text(
        json.dumps(
            {
                "action": "publish_batch",
                "path": str(output_path),
                "items": [{"file": str(source_path), "sha256": sha, "metadata": {"run_id": run_id}}],
            }
        )
    )
    return sha


def _run_publish_batch_subprocess(request_path: Path, response_path: Path) -> subprocess.Popen[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    return subprocess.Popen(
        [sys.executable, "-m", "utils.data_store", "--request", str(request_path), "--response", str(response_path)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_publish_batch_failure_preserves_store_created_by_concurrent_writer_before_lock(tmp_path, monkeypatch):
    output = tmp_path / "output.duckdb"
    source_a, source_b = tmp_path / "a.json", tmp_path / "b.json"
    source_a.write_text('{"owner": "A"}')
    source_b.write_text('{"owner": "B"}')
    req_a, resp_a = tmp_path / "req_a.json", tmp_path / "resp_a.json"
    req_b, resp_b = tmp_path / "req_b.json", tmp_path / "resp_b.json"
    _write_batch_request(req_a, output, source_a, "writer-a")
    _write_batch_request(req_b, output, source_b, "writer-b")

    real_acquire = writer.acquire_exclusive_write_conn
    b_refs = []

    @contextmanager
    def acquire_after_b_commits(path):
        # Writer B publishes after A checked target existence, before A acquires the write lock.
        proc_b = _run_publish_batch_subprocess(req_b, resp_b)
        stdout, stderr = proc_b.communicate(timeout=30)
        assert proc_b.returncode == 0, f"writer B failed: stdout={stdout!r} stderr={stderr!r}"
        b_payload = json.loads(resp_b.read_text())
        assert b_payload["ok"] is True
        b_ref = store.DataRef(**b_payload["results"][0]["ref"])
        assert store.read_document(output, b_ref) == {"owner": "B"}
        b_refs.append(b_ref)

        source_a.write_text('{"owner": "A-tampered"}')
        with real_acquire(path) as session:
            yield session

    monkeypatch.setattr(writer, "acquire_exclusive_write_conn", acquire_after_b_commits)
    assert store.main(["--request", str(req_a), "--response", str(resp_a)]) == 1
    assert json.loads(resp_a.read_text())["ok"] is False
    assert output.is_file()
    assert store.read_document(output, b_refs[0]) == {"owner": "B"}


def test_publish_batch_failure_cleans_up_before_unlock_and_preserves_writer_b_after_lock(tmp_path, monkeypatch):
    output = tmp_path / "output.duckdb"
    source_a, source_b = tmp_path / "a.json", tmp_path / "b.json"
    source_a.write_text('{"owner": "A"}')
    source_b.write_text('{"owner": "B"}')
    req_a, resp_a = tmp_path / "req_a.json", tmp_path / "resp_a.json"
    req_b, resp_b = tmp_path / "req_b.json", tmp_path / "resp_b.json"
    sha_a = _write_batch_request(req_a, output, source_a, "writer-a")
    _write_batch_request(req_b, output, source_b, "writer-b")

    real_acquire = writer.acquire_exclusive_write_conn
    b_refs = []

    @contextmanager
    def acquire_while_b_waits_for_unlock(path):
        with real_acquire(path) as session:
            # Start writer B while A holds the write lock; B must block on the same lock file.
            proc_b = _run_publish_batch_subprocess(req_b, resp_b)
            time.sleep(0.15)
            assert proc_b.poll() is None
            source_a.write_text('{"owner": "A-tampered"}')
            yield session

        # Immediately after A releases the write lock (and before A's main returns),
        # B acquires the preserved lock, publishes to a clean target, and commits.
        stdout, stderr = proc_b.communicate(timeout=30)
        assert proc_b.returncode == 0, f"writer B failed: stdout={stdout!r} stderr={stderr!r}"
        b_payload = json.loads(resp_b.read_text())
        assert b_payload["ok"] is True
        b_ref = store.DataRef(**b_payload["results"][0]["ref"])
        assert store.read_document(output, b_ref) == {"owner": "B"}
        b_refs.append(b_ref)

    monkeypatch.setattr(writer, "acquire_exclusive_write_conn", acquire_while_b_waits_for_unlock)
    assert store.main(["--request", str(req_a), "--response", str(resp_a)]) == 1
    assert json.loads(resp_a.read_text())["ok"] is False
    assert output.is_file()
    assert (tmp_path / "output.duckdb.write.lock").is_file()
    assert store.read_document(output, b_refs[0]) == {"owner": "B"}
    assert not (tmp_path / "artifacts" / sha_a).exists()


def test_publish_batch_dangling_symlink_defers_store_creation_on_invalid_checksum(tmp_path):
    absent = tmp_path / "absent.duckdb"
    output = tmp_path / "output.duckdb"
    output.symlink_to(absent)
    source = tmp_path / "spec.json"
    source.write_text('{"value": 1}')
    checkpoint, request, response = tmp_path / "progress.json", tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "action": "publish_batch",
                "path": str(output),
                "items": [{"file": str(source), "sha256": None, "metadata": {"run_id": "toy"}}],
                "checkpoint": str(checkpoint),
            }
        )
    )
    assert store.main(["--request", str(request), "--response", str(response)]) == 1
    assert not absent.exists()
    assert output.is_symlink()
    assert not output.exists()
    assert not (tmp_path / "artifacts").exists()
    failed = json.loads(response.read_text())
    assert failed["ok"] is False
    assert failed["results"] == [
        {
            "file": str(source),
            "ok": False,
            "error": "Frozen import file does not match the declared checksum",
        }
    ]
    assert checkpoint.read_text() == store.canonical(failed["results"]) + "\n"


@pytest.mark.parametrize("pause_at", ["batch_entry", "shared_write_conn"])
def test_publish_batch_recovers_when_empty_store_disappears_during_sqlite_header_check(
    tmp_path, monkeypatch, pause_at
):
    output = tmp_path / "output.duckdb"
    source_a, source_b = tmp_path / "a.json", tmp_path / "b.json"
    source_a.write_text('{"owner": "A"}')
    source_b.write_text('{"owner": "B"}')
    req_a, resp_a = tmp_path / "req_a.json", tmp_path / "resp_a.json"
    req_b, resp_b = tmp_path / "req_b.json", tmp_path / "resp_b.json"
    sha_a = _write_batch_request(req_a, output, source_a, "writer-a")
    _write_batch_request(req_b, output, source_b, "writer-b")

    a_created_empty_store = threading.Event()
    b_paused_before_open = threading.Event()
    a_finished_cleanup = threading.Event()
    a_tid: list[int | None] = [None]
    b_tid: list[int | None] = [None]
    lock_inodes: dict[str, int] = {}
    results: dict[str, int] = {}
    errors: list[BaseException] = []
    b_rb_open_count = 0
    target_open_index = 1 if pause_at == "batch_entry" else 2

    real_acquire = writer.acquire_exclusive_write_conn
    real_path_open = Path.open

    @contextmanager
    def coordinated_acquire(path, retry_interval_sec=1):
        tid = threading.get_ident()
        with real_acquire(path, retry_interval_sec=retry_interval_sec) as session:
            if tid == a_tid[0]:
                lock_inodes["a"] = session.lock_path.stat().st_ino
                a_created_empty_store.set()
                assert b_paused_before_open.wait(timeout=10)
                source_a.write_text('{"owner": "A-tampered"}')
            elif tid == b_tid[0]:
                lock_inodes["b"] = session.lock_path.stat().st_ino
            yield session

    def paused_path_open(self, mode="r", *args, **kwargs):
        nonlocal b_rb_open_count
        if (
            threading.get_ident() == b_tid[0]
            and mode == "rb"
            and self.expanduser().resolve() == output.resolve()
        ):
            b_rb_open_count += 1
            if b_rb_open_count == target_open_index:
                b_paused_before_open.set()
                assert a_finished_cleanup.wait(timeout=10)
        return real_path_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(writer, "acquire_exclusive_write_conn", coordinated_acquire)
    monkeypatch.setattr(Path, "open", paused_path_open)

    def run_a():
        a_tid[0] = threading.get_ident()
        try:
            results["a"] = store.main(["--request", str(req_a), "--response", str(resp_a)])
        except BaseException as exc:
            errors.append(exc)
        finally:
            a_finished_cleanup.set()

    def run_b():
        b_tid[0] = threading.get_ident()
        try:
            assert a_created_empty_store.wait(timeout=10)
            results["b"] = store.main(["--request", str(req_b), "--response", str(resp_b)])
        except BaseException as exc:
            errors.append(exc)
            b_paused_before_open.set()

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=20)
    thread_b.join(timeout=20)

    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert not errors
    assert results == {"a": 1, "b": 0}
    assert json.loads(resp_a.read_text())["ok"] is False
    b_payload = json.loads(resp_b.read_text())
    assert b_payload["ok"] is True
    b_ref = store.DataRef(**b_payload["results"][0]["ref"])
    assert output.is_file()
    lock_path = tmp_path / "output.duckdb.write.lock"
    assert lock_path.is_file()
    assert lock_inodes == {"a": lock_inodes["a"], "b": lock_inodes["a"]}
    assert store.read_document(output, b_ref) == {"owner": "B"}
    assert not (tmp_path / "artifacts" / sha_a).exists()


def test_publish_batch_rejects_sqlite_store_created_before_write_lock_without_freezing(tmp_path, monkeypatch):
    import fcntl
    import sqlite3

    output = tmp_path / "output.duckdb"
    source = tmp_path / "spec.json"
    source.write_text('{"value": 1}')
    checkpoint, request, response = tmp_path / "progress.json", tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "action": "publish_batch",
                "path": str(output),
                "items": [
                    {
                        "file": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "metadata": {"run_id": "toy"},
                    }
                ],
                "checkpoint": str(checkpoint),
            }
        )
    )

    real_flock = fcntl.flock
    injected = False

    def create_sqlite_before_lock(fd, operation):
        nonlocal injected
        if not injected and (operation & fcntl.LOCK_EX):
            injected = True
            conn = sqlite3.connect(str(output))
            conn.execute("CREATE TABLE legacy (id INTEGER)")
            conn.commit()
            conn.close()
        return real_flock(fd, operation)

    monkeypatch.setattr(writer.fcntl, "flock", create_sqlite_before_lock)
    with pytest.raises(ValueError, match="SQLite database files are unsupported"):
        store.main(["--request", str(request), "--response", str(response)])

    assert output.read_bytes()[:16] == b"SQLite format 3\x00"
    assert not response.exists()
    assert not checkpoint.exists()
    assert not (tmp_path / "artifacts").exists()
