import json

from scripts.alignment.evidence import archive_previous


def test_retry_moves_large_inputs_without_copying_and_keeps_process_history(tmp_path):
    output = tmp_path / "cycle"
    raw = output / "inputs/day/factor.parquet"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw input" * 100)
    inode = raw.stat().st_ino
    (output / "alignment_manifest.json").write_text(json.dumps({"fingerprint": "old"}))
    (output / "comparison.json").write_text('{"status":"FAIL"}')
    archive_previous(output, "old")
    assert raw.exists() and not (output / "history").exists()
    archive_previous(output, "new")
    history = next((output / "history").iterdir())
    assert not raw.exists()
    assert (history / "inputs/day/factor.parquet").stat().st_ino == inode
    assert json.loads((history / "alignment_manifest.json").read_text())["fingerprint"] == "old"
    assert (history / "comparison.json").is_file()
    assert json.loads((history / "retained_paths.json").read_text())[str(output / "inputs")] == str(history / "inputs")
    archive_previous(output, "new")
    assert len(list((output / "history").iterdir())) == 1
