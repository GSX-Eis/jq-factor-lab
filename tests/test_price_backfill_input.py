import csv
import hashlib
import json

import duckdb
import pytest

from data_pipeline.loaders.import_price_backfill_duckdb import (
    import_partition,
    RAW_FIELDS,
)
from utils.data_store import DataRef, migrate_table, read_reference
from utils.input_store import publish_input_batch


def test_monthly_backfill_keeps_checks_and_resumes_only_its_own_staging(
    tmp_path, monkeypatch
):
    source, live, staging = (
        tmp_path / name for name in ("old.duckdb", "input.duckdb", "staging.duckdb")
    )
    columns = [
        "stock_code",
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "money",
        "factor",
    ]
    with duckdb.connect(str(source)) as conn:
        conn.execute(
            "CREATE TABLE price_data_5m(stock_code VARCHAR, date TIMESTAMP, open DOUBLE, close DOUBLE, high DOUBLE, low DOUBLE, volume BIGINT, money DOUBLE, factor DOUBLE, PRIMARY KEY(stock_code,date))"
        )
        conn.execute(
            "INSERT INTO price_data_5m VALUES ('a','2019-01-02 10:00:00',10,10,10,10,100,1000,1),('b','2019-01-03 10:00:00',NULL,12,12,12,100,1000,1),('outside','2019-02-01 10:00:00',20,20,20,20,100,1000,1)"
        )
    original = migrate_table(source, live, "price_data_5m", role="input", execute=True)[
        "ref"
    ]
    source.unlink()
    bindings = {"input_db": str(live), "inputs": {"price_data_5m": original}}
    monkeypatch.setenv("QUANT_INPUT_PARENT_BINDINGS", json.dumps(bindings))
    monkeypatch.setenv("QUANT_INPUT_BATCH_STAGING", "1")
    partition = tmp_path / "csv" / "2019-01"
    partition.mkdir(parents=True)
    data = partition / "prices.csv"
    with data.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(
            [
                ["a", "2019-01-02 10:00:00", 11, 11, 11, 11, 100, 1000, 2],
                ["b", "2019-01-03 10:00:00", 12, 12, 12, 12, 100, 1000, 2],
                ["c", "2019-01-04 10:00:00", 13, 13, 13, 13, 100, 1000, 2],
            ]
        )
    manifest = {
        "rows": 3,
        "sha256": {"prices.csv": hashlib.sha256(data.read_bytes()).hexdigest()},
    }
    (partition / "manifest.json").write_text(json.dumps(manifest))
    # Legacy success is unrelated to the new database and must never skip this batch.
    (partition / "imported.json").write_text('{"legacy": true}')
    with pytest.raises(ValueError, match="existing/source conflicts"):
        import_partition(staging, partition, defer_factors=True)
    (partition / "replace_conflicts.json").write_text(
        json.dumps({"partition": "2019-01", "replace_fields": RAW_FIELDS})
    )
    result = import_partition(staging, partition, defer_factors=True)
    assert (
        result["inserted"]
        == result["repaired"]
        == result["replaced_conflict_rows"]
        == 1
    )
    assert import_partition(staging, partition, defer_factors=True) == result
    assert (partition / "imported.json").read_text() == '{"legacy": true}'
    assert not (partition / "existing_conflicts_before_replace.parquet").exists()
    with duckdb.connect(str(staging), read_only=True) as conn:
        assert conn.execute(
            "SELECT stock_code, open, factor FROM price_data_5m ORDER BY stock_code"
        ).fetchall() == [("a", 11, 1), ("b", 12, 1), ("c", 13, 2)]
    published = publish_input_batch(
        live,
        staging,
        tables=["price_data_5m"],
        metadata={
            "source": str(partition),
            "source_hash": manifest["sha256"]["prices.csv"],
            "loader_hash": "loader",
        },
        parents=bindings["inputs"],
    )["results"][0]
    assert published["changed_rows"] == 3 and published["audit"]["rows"] == 4
    assert read_reference(live, DataRef(**original)).sort_values("stock_code")[
        "close"
    ].tolist() == [10, 12, 20]
    assert read_reference(live, DataRef(**published["ref"])).sort_values("stock_code")[
        "close"
    ].tolist() == [11, 12, 13, 20]
    # A receipt left behind after losing a partition's staged rows cannot report success.
    with duckdb.connect(str(staging)) as conn:
        conn.execute("DELETE FROM price_data_5m WHERE stock_code='c'")
    with pytest.raises(ValueError, match="staging no longer matches"):
        import_partition(staging, partition, defer_factors=True)
