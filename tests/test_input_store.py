import duckdb
import pytest

from utils.data_store import DataRef, connect_input, migrate_table, read_reference
from utils.input_store import publish_input_batch


def test_input_updates_keep_old_versions_and_commit_batch_atomically(tmp_path):
    source, live, stage = (
        tmp_path / name for name in ("old.duckdb", "input.duckdb", "stage.duckdb")
    )
    with duckdb.connect(str(source)) as conn:
        conn.execute(
            "CREATE TABLE prices (stock_code VARCHAR, date DATE, close DOUBLE, PRIMARY KEY(stock_code,date))"
        )
        conn.execute(
            "INSERT INTO prices VALUES ('a','2019-01-02',1),('b','2019-01-02',2)"
        )
    original = migrate_table(source, live, "prices", role="input", execute=True)["ref"]
    with duckdb.connect(str(stage)) as conn:
        conn.execute(
            "CREATE TABLE prices (stock_code VARCHAR, date DATE, close DOUBLE, PRIMARY KEY(stock_code,date))"
        )
        conn.execute(
            "INSERT INTO prices VALUES ('a','2019-01-02',10),('c','2019-01-02',3)"
        )
    metadata = {
        "source": str(tmp_path / "download"),
        "source_hash": "one",
        "loader_hash": "loader",
    }
    args = dict(tables=["prices"], metadata=metadata, parents={"prices": original})
    published = publish_input_batch(live, stage, **args)["results"][0]
    assert published["audit"]["rows"] == 3 and published["replaced_rows"] == 1
    assert read_reference(live, DataRef(**original)).sort_values(
        "stock_code"
    ).close.tolist() == [1, 2]
    assert read_reference(live, DataRef(**published["ref"])).sort_values(
        "stock_code"
    ).close.tolist() == [10, 2, 3]
    assert publish_input_batch(live, stage, **args)["results"][0] == published
    with duckdb.connect(str(live), read_only=True) as conn:
        assert conn.execute(
            "SELECT close FROM prices ORDER BY stock_code"
        ).fetchall() == [(10,), (2,), (3,)]
        assert (
            conn.execute("SELECT count(*) FROM meta.input_batches").fetchone()[0] == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM duckdb_constraints() WHERE constraint_type='PRIMARY KEY' "
                "AND table_name LIKE '__input_base_%'"
            ).fetchone()[0]
            == 1
        )
    with duckdb.connect(str(stage)) as conn:
        conn.execute("UPDATE prices SET close=11 WHERE stock_code='a'")
    with pytest.raises(ValueError, match="changed after publication"):
        publish_input_batch(live, stage, **args)
    with pytest.raises(ValueError, match="head changed"):
        publish_input_batch(
            live, stage, **{**args, "metadata": {**metadata, "source_hash": "two"}}
        )
    with duckdb.connect(str(stage)) as conn:
        conn.execute(
            "CREATE TABLE broken (id INTEGER); INSERT INTO broken VALUES (1),(1)"
        )
    with pytest.raises(ValueError, match="Duplicate keys"):
        publish_input_batch(
            live,
            stage,
            tables=["prices", "broken"],
            metadata={**metadata, "source_hash": "two"},
            parents={"prices": published["ref"]},
        )
    with duckdb.connect(str(live), read_only=True) as conn:
        assert conn.execute(
            "SELECT close FROM prices ORDER BY stock_code"
        ).fetchall() == [(10,), (2,), (3,)]
        assert (
            conn.execute("SELECT count(*) FROM meta.input_batches").fetchone()[0] == 1
        )
    second = publish_input_batch(
        live,
        stage,
        tables=["prices"],
        metadata={**metadata, "source_hash": "two"},
        parents={"prices": published["ref"]},
    )["results"][0]
    assert second["changed_rows"] == 1
    assert read_reference(live, DataRef(**published["ref"])).sort_values(
        "stock_code"
    ).close.tolist() == [10, 2, 3]
    assert read_reference(live, DataRef(**second["ref"])).sort_values(
        "stock_code"
    ).close.tolist() == [11, 2, 3]
    bindings = {"input_db": str(live), "inputs": {"prices": second["ref"]}}
    with connect_input(live, bindings=bindings) as conn:
        assert conn.execute(
            "SELECT close FROM prices ORDER BY stock_code"
        ).fetchall() == [(11,), (2,), (3,)]
    # Retrying an older committed batch cannot roll the current head backwards.
    with duckdb.connect(str(stage)) as conn:
        conn.execute("UPDATE prices SET close=10 WHERE stock_code='a'")
    assert publish_input_batch(live, stage, **args)["results"][0] == published
    with duckdb.connect(str(live), read_only=True) as conn:
        assert (
            conn.execute(
                "SELECT version FROM meta.input_heads WHERE target_table='main.prices'"
            ).fetchone()[0]
            == second["ref"]["version"]
        )


def test_snapshot_import_and_unique_index_identity(tmp_path):
    source, live, stage = (
        tmp_path / name for name in ("old.duckdb", "input.duckdb", "stage.duckdb")
    )
    for path in (source, stage):
        with duckdb.connect(str(path)) as conn:
            conn.execute(
                "CREATE TABLE events (id INTEGER, value DOUBLE); CREATE UNIQUE INDEX events_id ON events(id)"
            )
    with duckdb.connect(str(source)) as conn:
        conn.execute("INSERT INTO events VALUES (1,1),(2,2)")
    original = migrate_table(source, live, "events", role="input", execute=True)["ref"]
    with duckdb.connect(str(stage)) as conn:
        conn.execute("INSERT INTO events VALUES (1,10)")
    metadata = {
        "source": str(tmp_path / "csv"),
        "source_hash": "one",
        "loader_hash": "loader",
    }
    updated = publish_input_batch(
        live, stage, tables=["events"], metadata=metadata, parents={"events": original}
    )["results"][0]
    assert read_reference(live, DataRef(**updated["ref"])).sort_values(
        "id"
    ).value.tolist() == [10, 2]
    replaced = publish_input_batch(
        live,
        stage,
        tables=["events"],
        metadata={**metadata, "source_hash": "two"},
        parents={"events": updated["ref"]},
        replace=True,
    )["results"][0]
    assert read_reference(live, DataRef(**replaced["ref"])).value.tolist() == [10]
    assert read_reference(live, DataRef(**original)).sort_values(
        "id"
    ).value.tolist() == [1, 2]


@pytest.mark.parametrize("replace", ["false", 0, None])
@pytest.mark.parametrize("existing", [True, False])
def test_publish_input_batch_rejects_nonboolean_replace_without_writes(tmp_path, replace, existing):
    source, live, stage = (tmp_path / name for name in ("source.duckdb", "live.duckdb", "stage.duckdb"))
    with duckdb.connect(str(source)) as conn:
        conn.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, value INTEGER)")
        conn.execute("INSERT INTO events VALUES (1,10),(2,20)")
    parents = {}
    before = {}
    if existing:
        parents["events"] = migrate_table(source, live, "events", role="input", execute=True)["ref"]
        with duckdb.connect(str(live), read_only=True) as conn:
            before = {
                name: conn.execute(f'SELECT * FROM meta."{name}" ORDER BY ALL').fetchall()
                for name, in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='meta'").fetchall()
            }
    with duckdb.connect(str(stage)) as conn:
        conn.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, value INTEGER)")
        conn.execute("INSERT INTO events VALUES (1,11)")
    with pytest.raises(ValueError, match="replace must be a bool"):
        publish_input_batch(
            live, stage, tables=["events"], parents=parents, replace=replace,
            metadata={"source": str(source), "source_hash": "fixed", "loader_hash": "fixed"},
        )
    if existing:
        with duckdb.connect(str(live), read_only=True) as conn:
            assert conn.execute("SELECT * FROM events ORDER BY id").fetchall() == [(1,10),(2,20)]
            after = {
                name: conn.execute(f'SELECT * FROM meta."{name}" ORDER BY ALL').fetchall()
                for name, in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='meta'").fetchall()
            }
        assert before == after
    else:
        assert not live.exists()
