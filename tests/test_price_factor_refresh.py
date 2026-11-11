import hashlib
import json

import duckdb
import pandas as pd
import pytest

from data_pipeline.loaders.import_price_factor_refresh_duckdb import stage_refresh
from utils.data_store import DataRef, migrate_table, read_reference
from utils.input_store import publish_input_batch, publish_price_factors


def test_daily_factor_patch_preserves_intraday_history_and_future_imports(
    tmp_path, monkeypatch
):
    old, live, stage = (
        tmp_path / name for name in ("old.duckdb", "input.duckdb", "stage.duckdb")
    )
    source = tmp_path / "factors"
    year = source / "2019"
    year.mkdir(parents=True)
    data = year / "factors.csv"
    data.write_text(
        "stock_code,date,factor\na,2019-01-02,0.5\nb,2019-01-02,\nc,2019-01-02,0.7\n"
    )
    (year / "manifest.json").write_text(
        json.dumps(
            {
                "reference_date": "2019-12-31",
                "sha256": {
                    "factors.csv": hashlib.sha256(data.read_bytes()).hexdigest()
                },
            }
        )
    )
    plan = {
        "start": "2019-01-01",
        "end": "2019-12-31",
        "partitions": {
            "2019-01": [
                {
                    "stock_code": code,
                    "start": "2019-01-02",
                    "end": "2019-01-02",
                    "expected_rows": count,
                }
                for code, count in [("a", 2), ("b", 1)]
            ]
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    (source / "applied.json").write_text(
        '{"status":"complete"}'
    )  # old files cannot skip a new database batch
    with duckdb.connect(str(old)) as conn:
        conn.execute(
            "CREATE TABLE price_data_5m(stock_code VARCHAR,date TIMESTAMP,open DOUBLE,close DOUBLE,factor DOUBLE,PRIMARY KEY(stock_code,date))"
        )
        conn.execute(
            "INSERT INTO price_data_5m VALUES ('a','2019-01-02 09:35:00',10,11,1),('a','2019-01-02 09:40:00',12,13,2),('b','2019-01-02 09:35:00',14,15,3),('outside','2018-12-31 09:35:00',16,17,4)"
        )
    original = migrate_table(old, live, "price_data_5m", role="input", execute=True)[
        "ref"
    ]
    old.unlink()
    bindings = {"input_db": str(live), "inputs": {"price_data_5m": original}}
    monkeypatch.setenv("QUANT_INPUT_PARENT_BINDINGS", json.dumps(bindings))
    monkeypatch.setenv("QUANT_INPUT_BATCH_STAGING", "1")
    monkeypatch.setenv("QUANT_INPUT_BATCH_REPLACE", "0")
    report = stage_refresh(stage, source, plan_path)
    assert report["custody"]["expected_rows"] == 3
    metadata = {
        "source": str(source),
        "source_hash": "factors",
        "loader_hash": "loader",
    }
    args = dict(tables=["price_data_5m"], metadata=metadata, parents=bindings["inputs"])
    published = publish_price_factors(live, stage, **args)
    result = published["results"][0]
    assert result["changed_rows"] == 2 and result["patch_rows"] == 2
    assert (
        result["valid_price_bars_without_fresh_factor"] == 1
        and not result["research_ready"]
    )
    assert (
        result["status"] == "complete_with_source_gaps"
        and result["backup_ref"] == original
    )

    def frame(ref):
        return (
            read_reference(live, DataRef(**ref))
            .sort_values(["stock_code", "date"])
            .reset_index(drop=True)
        )

    before, after = frame(original), frame(result["ref"])
    assert before.factor.tolist() == [1, 2, 3, 4] and after.factor.tolist() == [
        0.5,
        0.5,
        3,
        4,
    ]
    pd.testing.assert_frame_equal(
        before.drop(columns="factor"), after.drop(columns="factor")
    )
    with duckdb.connect(str(live), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM price_data_5m").fetchone() == (4,)
        assert conn.execute(
            "SELECT table_type FROM information_schema.tables WHERE table_name='price_data_5m'"
        ).fetchone() == ("VIEW",)
        physical = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='input_versions' AND table_type='BASE TABLE'"
        ).fetchall()
        assert len(physical) == 1 and physical[0][0].startswith("factor_patch_")
    full = tmp_path / "next.duckdb"
    with duckdb.connect(str(full)) as conn:
        conn.register("one", after.iloc[:1])
        conn.execute("CREATE TABLE price_data_5m AS SELECT * FROM one")
        conn.execute("UPDATE price_data_5m SET close=20,factor=0.8")
    imported = publish_input_batch(
        live,
        full,
        tables=["price_data_5m"],
        metadata={**metadata, "source_hash": "new-bars"},
        parents={"price_data_5m": result["ref"]},
    )["results"][0]
    assert frame(imported["ref"]).factor.tolist() == [0.8, 0.5, 3, 4]
    pd.testing.assert_frame_equal(after, frame(result["ref"]))
    assert publish_price_factors(live, stage, **args) == published
    with duckdb.connect(str(live), read_only=True) as conn:
        assert conn.execute("SELECT version FROM meta.input_heads").fetchone() == (
            imported["ref"]["version"],
        )
    with duckdb.connect(str(stage)) as conn:
        conn.execute("UPDATE price_data_5m SET factor=.9 WHERE stock_code='a'")
    with pytest.raises(ValueError, match="changed after publication"):
        publish_price_factors(live, stage, **args)
    plan["partitions"]["2019-01"][0]["expected_rows"] = 3
    plan_path.write_text(json.dumps(plan))
    (tmp_path / "2019-01").mkdir()
    (tmp_path / "2019-01/imported.json").write_text("{}")
    with pytest.raises(ValueError, match="custody is incomplete"):
        stage_refresh(tmp_path / "bad-custody.duckdb", source, plan_path)
    assert not old.exists()
