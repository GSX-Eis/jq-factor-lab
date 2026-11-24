"""Full-cycle Scheme B identities, calendar anchors and Harness storage."""
from __future__ import annotations

import json
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Event

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from tools.agent_harness.cli import main as harness_main
from utils import market_factor_store
from utils.market_factor_store import (
    MARKET_FACTOR_IDENTITY_SCHEMA_VERSION,
    FactorValueConflictError,
    MarketFactorValueStore,
    build_identity_record,
    compute_factor_identity_hash,
)
from utils.session_calendar import SessionCalendar
from utils.slots import load_slot_config
from utils.time_axis import Frequency


def _identity_options(calendar: SessionCalendar, frequency: str) -> dict:
    return dict(
        factor_id="alpha",
        factor_class="tests.Alpha",
        source_sha256="a" * 64,
        params={},
        frequency=frequency,
        session_view="ETH",
        analysis_mode="cross_sectional",
        bucket_strategy="session" if frequency == "1d" else "segment_anchored",
        calendar_hash=calendar.calendar_hash,
        slot_id="test_market",
        slot_config_hash="b" * 64,
        adjustment_policy="none",
        slot_data_fingerprint={"duckdb_version": "v1", "roll": {"version": 1}},
        code_version="test-code",
    )


def _batch(calendar: SessionCalendar, frequency: str, **identity_overrides: object) -> tuple[pd.DataFrame, dict]:
    options = _identity_options(calendar, frequency) | identity_overrides
    digest, payload = compute_factor_identity_hash(**options)
    record = build_identity_record(factor_identity_hash=digest, payload=payload, code_version_source="test")
    keys = calendar.bar_grid(frequency).iloc[:4]
    frame = keys.assign(
        factor_id=options["factor_id"],
        factor_identity_hash=digest,
        frequency=frequency,
        session_view="ETH",
        instrument_id="X",
        factor_value=np.arange(len(keys), dtype=float),
        source_build_id="test-build",
        calendar_hash=calendar.calendar_hash,
        code_version="test-code",
        created_at=pd.Timestamp("2026-02-01", tz="UTC"),
        batch_id="test-batch",
    )
    return frame, record


@pytest.fixture
def calendar() -> SessionCalendar:
    return SessionCalendar("XSHG", "2026-01-05", "2026-01-16", rth_reference_calendar="XSHG")


@pytest.mark.parametrize("frequency", ["1m", "5m", "15m", "1h", "2h", "4h", "1d", "1w"])
def test_full_cycle_roundtrip(
    tmp_path: Path, calendar: SessionCalendar, frequency: str,
) -> None:
    store = MarketFactorValueStore(tmp_path)
    frame, record = _batch(calendar, frequency)
    receipt = store.append_factor_values(frame, identity_record=record, calendar=calendar)
    canonical = Frequency.parse(frequency).canonical
    assert receipt.inserted == len(frame) > 0
    assert f"factor_id=alpha/frequency={canonical}/year_month=2026-01/" in receipt.shards_written[0]

    filters = dict(slot_id="test_market", factor_id="alpha", frequency=frequency,
                   factor_identity_hash=record["factor_identity_hash"])
    assert store.list_identities(**filters)["slot_id"].tolist() == ["test_market"]
    pd.testing.assert_series_equal(store.read_factor_values(**filters).timestamp_utc, frame.timestamp_utc)
    assert store.read_factor_wide(**filters).iloc[:, 0].tolist() == frame.factor_value.tolist()
    assert store.read_factor_values(slot_id="other_market").empty
    assert store.verify_store(**filters)["total_committed_rows"] == len(frame)
    assert store.verify_store(slot_id="other_market")["batch_count"] == 0
    with duckdb.connect() as conn:
        store.create_view(conn)
        assert conn.execute("SELECT DISTINCT frequency FROM market_factor_values").fetchall() == [(canonical,)]


@pytest.mark.parametrize("frequency", ["1d", "1w"])
def test_daily_and_weekly_require_scheduled_anchor_and_session(
    tmp_path: Path, calendar: SessionCalendar, frequency: str,
) -> None:
    store = MarketFactorValueStore(tmp_path)
    frame, record = _batch(calendar, frequency)
    off_anchor = frame.assign(timestamp_utc=frame.timestamp_utc - pd.Timedelta(minutes=1))
    wrong_session = frame.assign(session_id="2026-01-06")
    for bad in (off_anchor, wrong_session):
        with pytest.raises(ValueError, match="session_id mismatch"):
            store.append_factor_values(bad, identity_record=record, calendar=calendar)
    if frequency == "1w":
        monday = calendar.bar_grid("1d").iloc[0]
        with pytest.raises(ValueError, match="session_id mismatch"):
            store.append_factor_values(
                frame.head(1).assign(timestamp_utc=monday.timestamp_utc, session_id=monday.session_id),
                identity_record=record, calendar=calendar,
            )


def test_v2_identity_hash_binds_slot_and_normalizes_metadata(calendar: SessionCalendar) -> None:
    options = _identity_options(calendar, "5m")
    options["calendar_hash"] = {"X": calendar.calendar_hash, "Y": "d" * 64}
    digest, payload = compute_factor_identity_hash(**options)
    assert payload["schema_version"] == MARKET_FACTOR_IDENTITY_SCHEMA_VERSION == "market_factor_identity.v2"
    reordered = options | {
        "frequency": "5min",
        "calendar_hash": {"Y": "d" * 64, "X": calendar.calendar_hash},
        "slot_data_fingerprint": {"roll": {"version": 1}, "duckdb_version": "v1"},
    }
    assert compute_factor_identity_hash(**reordered)[0] == digest
    for change in (
        {"slot_id": "other_market"},
        {"slot_config_hash": "e" * 64},
        {"calendar_hash": {"X": calendar.calendar_hash, "Y": "f" * 64}},
        {"adjustment_policy": "dynamic_pit"},
        {"slot_data_fingerprint": {"duckdb_version": "v2"}},
    ):
        assert compute_factor_identity_hash(**(options | change))[0] != digest
    for alias, canonical in (("ratio_back", "ratio_back_adjust"), ("panama_canal", "diff_back_adjust")):
        assert compute_factor_identity_hash(**(options | {"adjustment_policy": alias}))[0] == (
            compute_factor_identity_hash(**(options | {"adjustment_policy": canonical}))[0]
        )


def test_weekly_bitwise_idempotency_and_conflicts(tmp_path: Path, calendar: SessionCalendar) -> None:
    store = MarketFactorValueStore(tmp_path)
    frame, record = _batch(calendar, "1w")
    frame["factor_value"] = [np.nan, 1.25]
    assert store.append_factor_values(frame, identity_record=record, calendar=calendar).inserted == 2
    alternate_nan = np.array([0x7FF8000000000042], dtype=np.uint64).view(np.float64)[0]
    repeated = frame.assign(factor_value=[alternate_nan, 1.25])
    assert store.append_factor_values(repeated, calendar=calendar).skipped == 2
    for change in ([1, 1.25], [np.nan, np.nextafter(1.25, np.inf)]):
        with pytest.raises(FactorValueConflictError):
            store.append_factor_values(frame.assign(factor_value=change), calendar=calendar)
    assert store.verify_store()["committed_shard_count"] == 1


def test_uncommitted_weekly_shards_are_invisible_to_slot_reads(
    tmp_path: Path, calendar: SessionCalendar, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MarketFactorValueStore(tmp_path)
    frame, record = _batch(calendar, "1w")
    store.append_factor_values(frame.head(1), identity_record=record, calendar=calendar)
    original_append = market_factor_store._append_jsonl_line
    before_commit, release = Event(), Event()

    def pause_before_commit(path: Path, row: dict) -> None:
        if path == store.batches_path:
            before_commit.set()
            assert release.wait(15), "Writer was not released"
        original_append(path, row)

    monkeypatch.setattr(market_factor_store, "_append_jsonl_line", pause_before_commit)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(store.append_factor_values, frame.tail(1), calendar=calendar)
        try:
            assert before_commit.wait(15), "Writer did not reach the batch commit"
            assert len(list(store.parquet_root.rglob("*.parquet"))) == 2
            assert len(store.read_factor_values(slot_id="test_market", frequency="1w")) == 1
        finally:
            release.set()
        assert pending.result(timeout=15).inserted == 1
    assert len(store.read_factor_values(slot_id="test_market", frequency="1w")) == 2


def test_historical_v1_is_readable_and_read_only(tmp_path: Path, calendar: SessionCalendar) -> None:
    store = MarketFactorValueStore(tmp_path)
    frame, record = _batch(calendar, "5m")
    receipt = store.append_factor_values(frame, identity_record=record, calendar=calendar)

    # Construct the historical identity and ledger fields independently of the v2 writer.
    payload = json.loads(record["identity_payload_json"])
    slot_fields = ("slot_id", "slot_config_hash", "adjustment_policy", "slot_data_fingerprint")
    for field in slot_fields:
        del payload[field]
        del record[field]
    payload["schema_version"] = "market_factor_identity.v1"
    payload["source_identity"] = {"source_build_id": "test-build"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = sha256(encoded.encode()).hexdigest()
    old_options = {field: value for field, value in _identity_options(calendar, "5m").items() if field not in slot_fields}
    assert compute_factor_identity_hash(
        **old_options, source_build_id="test-build", schema_version="market_factor_identity.v1",
    )[0] == digest
    record.update(factor_identity_hash=digest, identity_payload_json=encoded,
                  source_schema_version=payload["schema_version"])
    batch = json.loads(store.batches_path.read_bytes())
    batch.pop("slot_id")
    batch["factor_identity_hash"] = digest
    old_path = store.parquet_root / receipt.shards_written[0]
    legacy_path = old_path.with_name(f"part-{digest[:12]}-historical.parquet")
    shard = pq.ParquetFile(old_path).read()
    shard = shard.set_column(
        shard.schema.get_field_index("factor_identity_hash"), "factor_identity_hash",
        pa.array([digest] * len(frame)),
    )
    pq.write_table(shard, legacy_path)
    old_path.unlink()
    batch["shards"][0]["part_path"] = str(legacy_path.relative_to(store.parquet_root))
    store.identities_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    store.batches_path.write_text(json.dumps(batch) + "\n", encoding="utf-8")
    paths = (store.identities_path, store.batches_path, legacy_path)
    historical_bytes = [path.read_bytes() for path in paths]

    assert store.list_identities(factor_identity_hash=digest)["slot_id"].isna().all()
    assert store.read_factor_values(factor_identity_hash=digest).factor_value.tolist() == frame.factor_value.tolist()
    assert store.verify_store(factor_identity_hash=digest)["ok"]
    assert store.read_factor_values(slot_id="test_market").empty
    replay = frame.assign(factor_identity_hash=digest)
    with pytest.raises(ValueError, match="read-only"):
        store.append_factor_values(replay, calendar=calendar)
    with pytest.raises(ValueError, match="read-only"):
        store.append_factor_values(replay, identity_record=record, calendar=calendar)
    assert [path.read_bytes() for path in paths] == historical_bytes

    store.append_factor_values(frame, identity_record=_batch(calendar, "5m")[1], calendar=calendar)
    assert store.verify_store()["identity_count"] == 2
    assert store.verify_store()["ok"]
    assert len(store.read_factor_values()) == 2 * len(frame)
    assert all(path.read_bytes().startswith(previous) for path, previous in zip(paths, historical_bytes))


def test_filtered_audit_validates_identity_and_only_checks_matching_shards(
    tmp_path: Path, calendar: SessionCalendar,
) -> None:
    store = MarketFactorValueStore(tmp_path)
    frame, record = _batch(calendar, "1d")
    store.append_factor_values(frame, identity_record=record, calendar=calendar)
    other, other_record = _batch(calendar, "1d", slot_id="other_market")
    other_receipt = store.append_factor_values(other, identity_record=other_record, calendar=calendar)
    assert store.read_factor_wide().shape == (len(frame), 2)

    (store.parquet_root / other_receipt.shards_written[0]).unlink()
    assert store.verify_store(slot_id="test_market")["ok"]
    assert store.verify_store(slot_id="other_market")["missing_shards"] == list(other_receipt.shards_written)

    records = [json.loads(line) for line in store.identities_path.read_text().splitlines()]
    records[0]["slot_config_hash"] = "c" * 64
    store.identities_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    report = store.verify_store(factor_identity_hash=record["factor_identity_hash"])
    assert not report["ok"]
    assert report["invalid_identities"][0]["factor_identity_hash"] == record["factor_identity_hash"]
    assert "slot_config_hash" in report["invalid_identities"][0]["error"]


@pytest.mark.parametrize("mode, frequency", [("explicit", "2h"), ("json", "4h"), ("yaml", "1w")])
def test_harness_v2_slot_roundtrip(
    tmp_path: Path, calendar: SessionCalendar, capsys: pytest.CaptureFixture[str], mode: str, frequency: str,
) -> None:
    root = tmp_path / "store"
    input_path = tmp_path / "factor.csv"
    if mode == "explicit":
        calendar_flags = ["--calendar", "XSHG", "--calendar-start", "2026-01-05", "--calendar-end", "2026-01-16"]
        metadata_flags = ["--slot-id", "test_market", "--slot-config-hash", "b" * 64, "--adjustment-policy", "none"]
        slot_config_hash = "b" * 64
    else:
        example = Path(__file__).resolve().parents[1] / "examples/declarative_crypto/slot.yaml"
        document = yaml.safe_load(example.read_text(encoding="utf-8"))
        document["slot_id"] = "test_market"
        document["calendar"] = {
            "provider": "pmc",
            "options": {"start": "2026-01-05", "end": "2026-01-16", "calendar_id": "XSHG"},
        }
        path = tmp_path / f"slot.{mode}"
        path.write_text(json.dumps(document) if mode == "json" else yaml.safe_dump(document), encoding="utf-8")
        calendar_flags = ["--slot-config", str(path)]
        metadata_flags = []
        slot_config_hash = load_slot_config(path).slot_config_hash

    keys = calendar.bar_grid(frequency).head(2).assign(instrument_id="X", factor_value=[1.25, np.nan])
    keys.to_csv(input_path, index=False)

    def cli(command: str, *flags: str, expected: int = 0) -> dict:
        rc = harness_main([
            command, "--factor-values-store", str(root), "--json", "--output-root", str(tmp_path / "out"), *flags,
        ])
        payload = json.loads(capsys.readouterr().out)
        assert rc == payload["exit_code"] == expected, payload
        return payload["metrics"]

    receipt = cli(
        "write-factor-values", "--input", str(input_path), "--factor-id", "alpha", "--frequency", frequency,
        "--analysis-mode", "time_series", "--slot-data-fingerprint", '{"duckdb":"v1"}',
        "--factor-source-sha256", "a" * 64,
        *calendar_flags, *metadata_flags,
    )
    assert receipt["inserted"] == 2
    identity = receipt["identity_record"]
    assert identity["source_schema_version"] == "market_factor_identity.v2"
    assert identity["slot_config_hash"] == slot_config_hash
    assert identity["slot_data_fingerprint"] == {"duckdb": "v1"}
    assert identity["slot_id"] == "test_market"
    filters = ["--slot-id", "test_market", "--factor-id", "alpha", "--frequency", frequency,
               "--factor-identity-hash", receipt["factor_identity_hash"]]
    output_csv = tmp_path / "export.csv"
    read = cli("read-factor-values", *filters, "--list-identities", "--limit", "2", "--out", str(output_csv))
    assert read["row_count"] == 2
    assert read["identity_count"] == 1
    assert [row["session_id"] for row in read["preview"]] == keys.session_id.tolist()
    assert read["preview"][1]["factor_value"] is None
    assert cli("verify-factor-values", *filters)["total_committed_rows"] == 2
    assert cli("verify-factor-values", "--slot-id", "unknown")["batch_count"] == 0

    modified = pd.read_csv(output_csv)
    modified.loc[0, "factor_value"] = 58.0
    modified.to_csv(output_csv, index=False)
    cli("write-factor-values", "--input", str(output_csv), "--slot-id", "other_market", *calendar_flags, expected=2)
    assert cli("write-factor-values", "--input", str(output_csv), *calendar_flags, expected=4)["conflict_count"] == 1


def _run_storage_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, *flags: str,
) -> dict:
    rc = harness_main([
        command, "--factor-values-store", str(tmp_path / "store"),
        "--output-root", str(tmp_path / "out"), *flags,
    ])
    result = json.loads(capsys.readouterr().out)
    assert rc == result["exit_code"]
    return result


@pytest.fixture
def weekly_csv_flags() -> tuple[str, ...]:
    return (
        "--factor-id", "alpha", "--frequency", "1w", "--analysis-mode", "time_series",
        "--calendar", "XSHG", "--calendar-start", "2026-01-05", "--calendar-end", "2026-01-16",
        "--slot-id", "test_market", "--slot-config-hash", "b" * 64,
        "--adjustment-policy", "none", "--slot-data-fingerprint", "bars-v1",
    )


def test_harness_slot_csv_reuses_definition_across_files(
    tmp_path: Path, calendar: SessionCalendar, capsys: pytest.CaptureFixture[str],
    weekly_csv_flags: tuple[str, ...],
) -> None:
    keys = calendar.bar_grid("1w").assign(instrument_id="X")
    for row, value, filename in (
        (0, 1.25, "first.csv"),
        (1, 9.0, "next.csv"),
        (0, np.nextafter(1.25, np.inf), "conflict.csv"),
    ):
        keys.iloc[[row]].assign(factor_value=value).to_csv(tmp_path / filename, index=False)

    def write(filename: str) -> dict:
        return _run_storage_cli(
            tmp_path, capsys, "write-factor-values", "--input", str(tmp_path / filename), *weekly_csv_flags,
            "--factor-source-sha256", "a" * 64,
        )

    missing_digest = _run_storage_cli(
        tmp_path, capsys, "write-factor-values", "--input", str(tmp_path / "first.csv"), *weekly_csv_flags,
    )
    assert missing_digest["exit_code"] == 2
    assert "--factor-source-sha256" in missing_digest["metrics"]["message"]
    first, appended, conflict = (write(filename) for filename in ("first.csv", "next.csv", "conflict.csv"))
    identity_hash = first["metrics"]["factor_identity_hash"]
    assert first["exit_code"] == appended["exit_code"] == 0
    assert conflict["exit_code"] == 4, conflict
    assert appended["metrics"]["factor_identity_hash"] == conflict["metrics"]["factor_identity_hash"] == identity_hash
    stored = MarketFactorValueStore(tmp_path / "store").read_factor_values(slot_id="test_market")
    assert stored["factor_value"].tolist() == [1.25, 9.0]


def test_harness_full_csv_preserves_float64_on_replay(
    tmp_path: Path, calendar: SessionCalendar, capsys: pytest.CaptureFixture[str],
    weekly_csv_flags: tuple[str, ...],
) -> None:
    input_path, output = tmp_path / "input.csv", tmp_path / "export.csv"
    calendar.bar_grid("1w").head(1).assign(instrument_id="X", factor_value=0.1).to_csv(input_path, index=False)
    write = _run_storage_cli(
        tmp_path, capsys, "write-factor-values", "--input", str(input_path), *weekly_csv_flags,
        "--factor-source-sha256", "a" * 64,
    )
    assert write["exit_code"] == 0, write
    read = _run_storage_cli(tmp_path, capsys, "read-factor-values", "--slot-id", "test_market", "--out", str(output))
    assert read["exit_code"] == 0, read

    def replay() -> dict:
        return _run_storage_cli(
            tmp_path, capsys, "write-factor-values", "--input", str(output),
            "--calendar", "XSHG", "--calendar-start", "2026-01-05", "--calendar-end", "2026-01-16",
        )

    original = replay()
    assert original["exit_code"] == 0 and original["metrics"]["skipped"] == 1
    exported = output.read_text(encoding="utf-8")
    assert ",0.1," in exported
    output.write_text(exported.replace(",0.1,", ",0.10000000000000002,"), encoding="utf-8")
    conflict = replay()
    assert conflict["exit_code"] == 4, conflict
    assert conflict["metrics"]["examples"][0]["incoming"] == np.nextafter(0.1, np.inf)


@pytest.mark.parametrize("corruption", ["swapped_slots", "wrong_batch"])
def test_shard_content_must_match_its_identity_and_batch(
    tmp_path: Path, calendar: SessionCalendar, corruption: str,
) -> None:
    store = MarketFactorValueStore(tmp_path)
    a, a_record = _batch(calendar, "1w", slot_id="slot_a")
    b, b_record = _batch(calendar, "1w", slot_id="slot_b")
    a = a.head(1).assign(factor_value=1.25, batch_id="a")
    b = b.head(1).assign(factor_value=9.0, batch_id="b")
    a_receipt = store.append_factor_values(a, identity_record=a_record, calendar=calendar)
    b_receipt = store.append_factor_values(b, identity_record=b_record, calendar=calendar)
    a_path = store.parquet_root / a_receipt.shards_written[0]
    b_path = store.parquet_root / b_receipt.shards_written[0]
    if corruption == "swapped_slots":
        a_shard, b_shard = a_path.read_bytes(), b_path.read_bytes()
        a_path.write_bytes(b_shard)
        b_path.write_bytes(a_shard)
    else:
        shard = pq.ParquetFile(b_path).read()
        shard = shard.set_column(shard.schema.get_field_index("batch_id"), "batch_id", pa.array(["wrong"]))
        pq.write_table(shard, b_path)

    full_audit = store.verify_store()
    slot_audit = store.verify_store(slot_id="slot_b")
    scoped_read = store.read_factor_values(slot_id="slot_b")
    assert not full_audit["ok"] and not slot_audit["ok"], (full_audit, slot_audit, scoped_read)
    expected_fields = ["batch_id", "factor_identity_hash"] if corruption == "swapped_slots" else ["batch_id"]
    assert slot_audit["shard_metadata_mismatches"] == [
        {"part_path": b_receipt.shards_written[0], "fields": expected_fields},
    ]
    assert scoped_read.empty
    expected_values = [] if corruption == "swapped_slots" else [1.25]
    assert store.read_factor_values()["factor_value"].tolist() == expected_values
    with duckdb.connect() as conn:
        store.create_view(conn)
        assert conn.execute("SELECT factor_value FROM market_factor_values").fetchnumpy()["factor_value"].tolist() == (
            expected_values
        )


def test_shard_rows_record_the_committed_batch_id(tmp_path: Path, calendar: SessionCalendar) -> None:
    store = MarketFactorValueStore(tmp_path)
    frame, record = _batch(calendar, "1w")
    frame["batch_id"] = ["original_a", "original_b"]
    receipt = store.append_factor_values(frame, identity_record=record, calendar=calendar)
    assert store.read_factor_values()["batch_id"].tolist() == [receipt.batch_id] * 2
    assert store.verify_store()["ok"]


def test_stage2_v1_mixed_batch_shard_survives_upgrade(tmp_path: Path) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "market_factor_v1_mixed_batch.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    store = MarketFactorValueStore(tmp_path)
    original_files = {tmp_path / path: b64decode(contents) for path, contents in fixture["files_base64"].items()}
    for path, contents in original_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    rows = store.read_factor_values()
    assert list(zip(rows["batch_id"], rows["factor_value"])) == [("a", 1.25), ("b", 9.0)]
    with duckdb.connect() as conn:
        store.create_view(conn)
        assert conn.execute(
            "SELECT batch_id, factor_value FROM market_factor_values ORDER BY timestamp_utc"
        ).fetchall() == [("a", 1.25), ("b", 9.0)]
    assert store.verify_store()["ok"]
    assert store.verify_store(factor_id="legacy_mixed")["ok"]
    assert all(path.read_bytes() == contents for path, contents in original_files.items())

    shard_path = next(path for path in original_files if path.suffix == ".parquet")
    table = pq.ParquetFile(shard_path).read()
    identity_col = table.schema.get_field_index("factor_identity_hash")
    table = table.set_column(
        identity_col, "factor_identity_hash", pa.array([rows["factor_identity_hash"].iloc[0], "f" * 64]),
    )
    pq.write_table(table, shard_path)
    assert store.read_factor_values()["batch_id"].tolist() == ["a"]
    assert store.verify_store(factor_id="legacy_mixed")["shard_metadata_mismatches"] == [
        {"part_path": str(shard_path.relative_to(store.parquet_root)), "fields": ["factor_identity_hash"]},
    ]


@pytest.mark.parametrize("field, wrong_value", [("batch_id", "z"), ("factor_identity_hash", "f" * 64)])
def test_v2_replay_rejects_candidate_with_wrong_shard_ownership(
    tmp_path: Path, calendar: SessionCalendar, field: str, wrong_value: str,
) -> None:
    store = MarketFactorValueStore(tmp_path)
    frame, record = _batch(calendar, "1w")
    receipt = store.append_factor_values(frame, identity_record=record, calendar=calendar)
    shard_path = store.parquet_root / receipt.shards_written[0]
    table = pq.ParquetFile(shard_path).read()
    table = table.set_column(table.schema.get_field_index(field), field, pa.array([wrong_value] * len(frame)))
    pq.write_table(table, shard_path)
    ledger_before = store.batches_path.read_bytes()
    shard_before = shard_path.read_bytes()
    assert store.read_factor_values().empty
    assert not store.verify_store()["ok"]

    with pytest.raises(ValueError, match=field) as error:
        store.append_factor_values(frame, identity_record=record, calendar=calendar)
    assert str(shard_path.relative_to(store.parquet_root)) in str(error.value)
    assert store.batches_path.read_bytes() == ledger_before
    assert shard_path.read_bytes() == shard_before
    assert list(store.parquet_root.rglob("*.parquet")) == [shard_path]
