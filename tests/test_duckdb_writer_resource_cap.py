"""Native synthetic regression for the shared writer's resource ceiling."""

import fcntl
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from utils import duckdb_results_writer as module
from utils.data_store import _catalog

PUBLICATION_PROOF = {}


def test_writer_session_caps_atomic_publication_without_changing_hashes(tmp_path):
    context = {
        "run_id": "synthetic-cap-test",
        "pool_id": "synthetic",
        "source_hash": "same-source",
        "config_hash": "same-config",
        "binding_hash": "same-binding",
        "input_versions": {},
        "created_at": "2020-01-01T00:00:00+00:00",
    }
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    frame = pd.DataFrame({"A": [1.25, 2.5], "B": [3.75, 4.0]}, index=dates)
    analysis = {
        "ic": pd.DataFrame({"5D": [0.1, -0.2]}, index=dates),
        "ir": pd.DataFrame({"ir": [0.3]}, index=["5D"]),
        "quantile_returns": pd.DataFrame(),
        "pool_type": "synthetic",
        "factor_statistics": {"stock_count": 2},
    }

    def publish(path, capped):
        writer = module.DuckDBResultsWriter(
            str(path),
            export_parquet=False,
            publication_metadata=context,
            data_bindings={"output_db": str(path)},
        )
        manager = writer.acquire_write_session() if capped else duckdb.connect(str(path))
        with manager as session:
            conn = session.conn if capped else session
            if capped:
                assert conn.execute("SELECT current_setting('threads')").fetchone() == (2,)
                # DuckDB displays decimal 3 GB as a binary GiB string.
                assert conn.execute("SELECT current_setting('memory_limit')").fetchone()[0].replace(" ", "") == "2.7GiB"
            else:
                conn.execute("SET threads=4")
            _catalog(conn)
            conn.execute("BEGIN TRANSACTION")
            try:
                assert writer.save_factor_values(frame, "synthetic", skip_parquet=True, duckdb_conn=conn) == 4
                assert writer.write_factor_analysis_results("synthetic", analysis, duckdb_conn=conn) == 2
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            metadata = conn.execute(
                "SELECT artifact_id,target_table,content_hash,row_count,schema_json,metadata_json "
                "FROM meta.artifacts ORDER BY artifact_id"
            ).fetchall()
            values = conn.execute("SELECT * FROM factor_values ORDER BY date,stock_code").fetchall()
            results = conn.execute("SELECT * FROM factor_analysis_results ORDER BY date,code").fetchall()
        return metadata, values, results

    original = publish(tmp_path / "uncapped.duckdb", False)
    capped = publish(tmp_path / "capped.duckdb", True)
    assert capped == original  # Includes rows, metadata, content hash and artifact ID.
    assert len(capped[0]) == 2
    PUBLICATION_PROOF.update(
        {
            "rows_and_hashes_equal": True,
            "factor_rows": len(capped[1]),
            "analysis_rows": len(capped[2]),
            "artifacts": [{"id": row[0], "table": row[1], "content_hash": row[2], "rows": row[3]} for row in capped[0]],
        }
    )


def test_capped_existing_connection_publication_still_rolls_back(tmp_path):
    path = tmp_path / "rollback.duckdb"
    writer = module.DuckDBResultsWriter(
        str(path),
        export_parquet=False,
        publication_metadata={
            "run_id": "synthetic",
            "pool_id": "synthetic",
            "source_hash": "same-source",
            "config_hash": "same-config",
            "input_versions": {},
            "created_at": "2020-01-01T00:00:00+00:00",
        },
        data_bindings={"output_db": str(path)},
    )
    frame = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2020-01-02"]))
    with writer.acquire_write_session() as session:
        _catalog(session.conn)
        session.conn.execute("BEGIN TRANSACTION")
        try:
            assert writer.save_factor_values(frame, "synthetic", skip_parquet=True, duckdb_conn=session.conn) == 1
            raise RuntimeError("injected after first publication")
        except RuntimeError:
            session.conn.execute("ROLLBACK")
    with duckdb.connect(str(path), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM meta.artifacts").fetchone() == (0,)
        assert conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='factor_values'"
        ).fetchone() == (0,)


@pytest.mark.parametrize("failed_setting", ["SET memory_limit='3GB'", "SET threads=2"])
def test_setting_failure_preserves_connection_and_lock_cleanup(tmp_path, monkeypatch, failed_setting):
    class Connection:
        closed = False
        statements = []

        def execute(self, statement):
            self.statements.append(statement)
            if statement == failed_setting:
                # Even a lock-like SET failure must not enter the connect retry loop.
                raise RuntimeError("Could not set lock on file: injected resource-setting failure")
            return self

        def close(self):
            self.closed = True

    connection = Connection()
    connections = []

    def connect(*_args, **_kwargs):
        connections.append(connection)
        return connection

    monkeypatch.setattr(module.duckdb, "connect", connect)
    monkeypatch.setattr(module.time, "sleep", lambda _: pytest.fail("SET failure must not retry"))
    path = tmp_path / "failure.duckdb"
    with pytest.raises(RuntimeError, match="resource-setting failure"), module.acquire_exclusive_write_conn(path):
        pytest.fail("A connection with unapplied resource limits must not escape")
    assert connection.closed
    assert connections == [connection]
    assert (
        connection.statements == ["SET memory_limit='3GB'", "SET threads=2"][: 1 if "memory" in failed_setting else 2]
    )
    with Path(str(path) + ".write.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
