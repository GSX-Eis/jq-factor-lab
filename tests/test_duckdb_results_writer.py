from __future__ import annotations

import multiprocessing
import inspect
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from backtest_and_analysis.result_writer import ResultWriter
from utils.duckdb_results_writer import (
    ARROW_AVAILABLE,
    DuckDBResultsWriter,
    acquire_exclusive_write_conn,
    pq,
)


def test_write_lock_default_retry_is_short_enough_for_parallel_workers() -> None:
    assert inspect.signature(acquire_exclusive_write_conn).parameters["retry_interval_sec"].default == 1


def test_read_only_connection_retries_duckdb_lock(monkeypatch) -> None:
    from utils import duckdb_results_writer

    expected = duckdb_results_writer.duckdb.connect()
    attempts = iter(
        [
            RuntimeError("Could not set lock on file"),
            RuntimeError("Conflicting lock is held"),
            expected,
        ]
    )

    def fake_connect(_path: str, *, read_only: bool):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        assert read_only is True
        return result

    monkeypatch.setattr(duckdb_results_writer.duckdb, "connect", fake_connect)
    monkeypatch.setattr(duckdb_results_writer.time, "sleep", lambda _seconds: None)

    assert duckdb_results_writer.connect_read_only_with_retry("demo.duckdb") is expected
    expected.close()


def _factor_df(values: list[float]) -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "000001.XSHE": values,
            "000002.XSHE": [value * 10 for value in values],
        },
        index=dates,
    )


def _analysis_results(ic_values: list[float], ir_value: float = 0.5) -> dict:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return {
        "ic": pd.DataFrame({"1D": ic_values}, index=dates),
        "ir": pd.DataFrame({"ir": [ir_value]}, index=["1D"]),
        "quantile_returns": pd.DataFrame(),
        "pool_type": "static",
        "factor_statistics": {"stock_count": 2},
    }


def test_quantile_rows_do_not_duplicate_ic_or_ir(tmp_path: Path) -> None:
    writer = DuckDBResultsWriter(
        duckdb_path=str(tmp_path / "analysis_results.duckdb"),
        parquet_dir=str(tmp_path / "parquet"),
        export_parquet=False,
    )
    results = _analysis_results([0.1, 0.2])
    index = pd.MultiIndex.from_product(
        [[1, 2], pd.to_datetime(["2024-01-02", "2024-01-03"])],
        names=["factor_quantile", "date"],
    )
    results["quantile_returns"] = pd.DataFrame({"1D": [0.01, 0.02, 0.03, 0.04]}, index=index)

    writer.write_factor_analysis_results("factor_alpha", results)

    conn = writer._get_connection(read_only=True)
    try:
        aggregate_count, quantile_ic_count = conn.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE quantile IS NULL AND ic IS NOT NULL),
                COUNT(*) FILTER (WHERE quantile IS NOT NULL AND ic IS NOT NULL)
            FROM factor_analysis_results
            WHERE factor_name = 'factor_alpha'
            """
        ).fetchone()
    finally:
        conn.close()

    assert aggregate_count == 2
    assert quantile_ic_count == 0


def _hold_write_lock(db_path: str, ready_queue, release_queue) -> None:
    with acquire_exclusive_write_conn(db_path, retry_interval_sec=0.05):
        ready_queue.put("locked")
        release_queue.get(timeout=10)


class _FailingConnection:
    def __init__(self, conn, fail_sql_fragment: str):
        self._conn = conn
        self._fail_sql_fragment = fail_sql_fragment

    def execute(self, sql, *args, **kwargs):
        normalized = " ".join(str(sql).split())
        if self._fail_sql_fragment in normalized:
            raise RuntimeError("injected monthly insert failure")
        return self._conn.execute(sql, *args, **kwargs)

    def register(self, *args, **kwargs):
        return self._conn.register(*args, **kwargs)

    def unregister(self, *args, **kwargs):
        return self._conn.unregister(*args, **kwargs)




@pytest.mark.skipif(not ARROW_AVAILABLE, reason="PyArrow is required for parquet export tests")
def test_parquet_export_atomically_replaces_complete_factor(tmp_path: Path, monkeypatch) -> None:
    writer = DuckDBResultsWriter(
        duckdb_path=str(tmp_path / "analysis_results.duckdb"),
        parquet_dir=str(tmp_path / "parquet"),
    )
    writer.export_factor_values_to_parquet(_factor_df([1.0, 2.0]), "factor_alpha")

    replacement = pd.DataFrame(
        {"000001.XSHE": [9.0]},
        index=pd.to_datetime(["2024-02-01"]),
    )
    writer.export_factor_values_to_parquet(replacement, "factor_alpha")
    factor_dir = tmp_path / "parquet" / "factor_values" / "factor_alpha"
    files = writer._list_visible_parquet_files(factor_dir)
    assert sum(pq.ParquetFile(path).metadata.num_rows for path in files) == 1
    assert not (factor_dir / "year=2024" / "month=1").exists()

    def fail_export(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pq, "write_to_dataset", fail_export)
    with pytest.raises(RuntimeError, match="boom"):
        writer.export_factor_values_to_parquet(_factor_df([3.0, 4.0]), "factor_alpha")
    files = writer._list_visible_parquet_files(factor_dir)
    assert sum(pq.ParquetFile(path).metadata.num_rows for path in files) == 1




def test_acquire_exclusive_write_conn_waits_for_other_process(tmp_path: Path) -> None:
    db_path = tmp_path / "analysis_results.duckdb"
    ready_queue = multiprocessing.Queue()
    release_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_hold_write_lock,
        args=(str(db_path), ready_queue, release_queue),
    )
    proc.start()

    try:
        # Spawn/import time is not the lock-wait behavior measured below.
        assert ready_queue.get(timeout=60) == "locked"
        releaser = threading.Timer(0.3, lambda: release_queue.put("release"))
        releaser.start()
        started = time.monotonic()
        with acquire_exclusive_write_conn(db_path, retry_interval_sec=0.05) as session:
            elapsed = time.monotonic() - started
        releaser.cancel()
    finally:
        if proc.is_alive():
            release_queue.put("release")
        proc.join(timeout=10)
        if proc.exitcode is None:
            proc.kill()
            proc.join(timeout=10)

    assert proc.exitcode == 0
    assert elapsed >= 0.2
    assert session.retry_count >= 1
    assert session.wait_seconds >= 0.2


def test_init_database_rejects_legacy_output_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "analysis_results.duckdb"
    import duckdb

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE factor_analysis_results (
                id INTEGER,
                factor_name VARCHAR NOT NULL,
                date DATE NOT NULL,
                code VARCHAR NOT NULL,
                period INTEGER NOT NULL,
                quantile INTEGER,
                ic DOUBLE,
                ir DOUBLE,
                return DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="legacy schema"):
        DuckDBResultsWriter(
            duckdb_path=str(db_path),
            parquet_dir=str(tmp_path / "parquet"),
        )


def test_external_transaction_rollback_keeps_tables_consistent(tmp_path: Path) -> None:
    writer = DuckDBResultsWriter(
        duckdb_path=str(tmp_path / "analysis_results.duckdb"),
        parquet_dir=str(tmp_path / "parquet"),
        export_parquet=False,
    )
    factor_name = "factor_theta"
    original_factor_df = _factor_df([1.0, 2.0])
    updated_factor_df = _factor_df([10.0, 20.0])
    original_results = _analysis_results([0.1, 0.2], ir_value=0.3)
    updated_results = _analysis_results([0.9, 0.8], ir_value=1.3)

    writer.save_factor_values(original_factor_df, factor_name, skip_parquet=True)
    writer.write_factor_analysis_results(factor_name, original_results)
    writer.write_monthly_pool_results(factor_name, original_results)

    with writer.acquire_write_session(retry_interval_sec=0.05) as session:
        conn = session.conn
        failing_conn = _FailingConnection(
            conn,
            f"INSERT INTO {writer.monthly_output_table}",
        )
        conn.execute("BEGIN TRANSACTION")
        try:
            writer.save_factor_values(
                updated_factor_df,
                factor_name,
                duckdb_conn=failing_conn,
                allow_fallback=False,
                skip_parquet=True,
            )
            writer.write_factor_analysis_results(
                factor_name,
                updated_results,
                duckdb_conn=failing_conn,
                allow_fallback=False,
            )
            with pytest.raises(RuntimeError, match="injected monthly insert failure"):
                writer.write_monthly_pool_results(
                    factor_name,
                    updated_results,
                    duckdb_conn=failing_conn,
                    allow_fallback=False,
                )
        finally:
            conn.execute("ROLLBACK")

    conn = writer._get_connection(read_only=True)
    try:
        factor_value_sum = conn.execute(
            "SELECT SUM(factor_value) FROM factor_values WHERE factor_name = ?",
            [factor_name],
        ).fetchone()[0]
        analysis_ic_sum = conn.execute(
            "SELECT SUM(ic) FROM factor_analysis_results WHERE factor_name = ?",
            [factor_name],
        ).fetchone()[0]
        monthly_ic_sum = conn.execute(
            "SELECT SUM(monthly_ic) FROM monthly_pool_results WHERE factor_name = ?",
            [factor_name],
        ).fetchone()[0]
    finally:
        conn.close()

    assert factor_value_sum == pytest.approx(33.0)
    assert analysis_ic_sum == pytest.approx(0.3)
    assert monthly_ic_sum == pytest.approx(0.15)


def test_result_writer_auto_mode_does_not_fallback_mid_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = ResultWriter(
        db_conn_str=f"sqlite:///{tmp_path / 'fallback.sqlite'}",
        output_table="factor_analysis_results",
        db_backend="auto",
        duckdb_path=str(tmp_path / "analysis_results.duckdb"),
        parquet_dir=str(tmp_path / "parquet"),
        export_parquet=False,
    )

    assert writer.using_duckdb is True

    def _boom(*args, **kwargs):
        raise RuntimeError("forced duckdb failure")

    monkeypatch.setattr(writer._duckdb_writer, "write_factor_analysis_results", _boom)

    with pytest.raises(RuntimeError, match="forced duckdb failure"):
        writer.write("factor_auto", _analysis_results([0.1, 0.2]))


def test_factor_values_indexes_stay_disabled(tmp_path: Path) -> None:
    writer = DuckDBResultsWriter(
        duckdb_path=str(tmp_path / "analysis_results.duckdb"),
        parquet_dir=str(tmp_path / "parquet"),
    )
    writer.save_factor_values(_factor_df([7.0, 8.0]), "factor_delta", skip_parquet=True)

    conn = writer._get_connection(read_only=True)
    try:
        indexes = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'factor_values' ORDER BY index_name"
        ).fetchall()
    finally:
        conn.close()

    assert indexes == []


def test_save_factor_values_date_range_replaces_only_range_and_rejects_outliers(
    tmp_path: Path,
) -> None:
    writer = DuckDBResultsWriter(
        duckdb_path=str(tmp_path / "analysis_results.duckdb"),
        parquet_dir=str(tmp_path / "parquet"),
        export_parquet=False,
    )
    initial = pd.DataFrame(
        {"000001.XSHE": [1.0, 2.0, 3.0, 4.0]},
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        ),
    )
    writer.save_factor_values(initial, "factor_range", skip_parquet=True)

    replacement = pd.DataFrame(
        {"000001.XSHE": [20.0, 30.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    writer.save_factor_values(
        replacement,
        "factor_range",
        skip_parquet=True,
        date_range=("2024-01-02", "2024-01-03"),
    )

    conn = writer._get_connection(read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT CAST(date AS VARCHAR), factor_value
            FROM factor_values
            WHERE factor_name = 'factor_range'
            ORDER BY date
            """
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        ("2024-01-01", 1.0),
        ("2024-01-02", 20.0),
        ("2024-01-03", 30.0),
        ("2024-01-04", 4.0),
    ]

    out_of_range = pd.DataFrame(
        {"000001.XSHE": [99.0]},
        index=pd.to_datetime(["2024-01-04"]),
    )
    with pytest.raises(ValueError, match="outside requested date range"):
        writer.save_factor_values(
            out_of_range,
            "factor_range",
            skip_parquet=True,
            date_range=("2024-01-02", "2024-01-03"),
        )

    for no_values in (
        pd.DataFrame(
            columns=["000001.XSHE"],
            index=pd.DatetimeIndex([], dtype="datetime64[ns]"),
        ),
        pd.DataFrame(
            {"000001.XSHE": [float("nan"), float("nan")]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
    ):
        with pytest.raises(ValueError, match="requires at least one finite factor value"):
            writer.save_factor_values(
                no_values,
                "factor_range",
                skip_parquet=True,
                date_range=("2024-01-02", "2024-01-03"),
            )

    conn = writer._get_connection(read_only=True)
    try:
        unchanged = conn.execute(
            "SELECT SUM(factor_value) FROM factor_values WHERE factor_name = 'factor_range'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert unchanged == pytest.approx(55.0)

    writer.save_factor_values(
        pd.DataFrame(
            {"000001.XSHE": [60.0]},
            index=pd.to_datetime(["2024-02-01"]),
        ),
        "factor_range",
        skip_parquet=True,
    )
    conn = writer._get_connection(read_only=True)
    try:
        full_rewrite_rows = conn.execute(
            """
            SELECT CAST(date AS VARCHAR), factor_value
            FROM factor_values
            WHERE factor_name = 'factor_range'
            """
        ).fetchall()
    finally:
        conn.close()
    assert full_rewrite_rows == [("2024-02-01", 60.0)]


def test_date_range_mode_rejects_removed_sqlite_backend_and_corruption_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    date_range = ("2024-01-02", "2024-01-03")
    with pytest.raises(ValueError, match="SQLite result storage"):
        ResultWriter(
            db_conn_str=f"sqlite:///{tmp_path / 'fallback.sqlite'}",
            output_table="factor_analysis_results",
            db_backend="sqlite",
        )

    duckdb_writer = DuckDBResultsWriter(
        duckdb_path=str(tmp_path / "analysis_results.duckdb"),
        parquet_dir=str(tmp_path / "parquet"),
        export_parquet=False,
    )

    def corrupt(*args, **kwargs):
        raise RuntimeError(
            "IO Error: Corrupt database file: computed checksum 1 "
            "does not match stored checksum 0 in block at location 0"
        )

    monkeypatch.setattr(duckdb_writer, "_count_factor_rows", corrupt)

    with pytest.raises(RuntimeError, match="Corrupt database file"):
        duckdb_writer.save_factor_values(
            _factor_df([1.0, 2.0]),
            "factor_range",
            skip_parquet=True,
            date_range=date_range,
        )

def test_symbolic_read_waits_for_writer_release(monkeypatch):
    import select
    import subprocess
    import sys
    import duckdb
    import factor_mining.symbolic.data_loader as loader
    from utils.local_outputs import managed_output

    monkeypatch.delenv("QUANT_DATA_BINDINGS", raising=False)
    with managed_output(None, "symbolic-reader-lock-test") as (scratch, lease):
        path = scratch / "synthetic.duckdb"
        with duckdb.connect(str(path)) as conn:
            conn.execute("SET threads=1")
            conn.execute("CREATE TABLE synthetic AS SELECT 7 AS value")
        writer_code = """import duckdb,sys,time
c=duckdb.connect(sys.argv[1]); c.execute("SET threads=1")
c.execute("UPDATE synthetic SET value=9")
print("READY",flush=True); time.sleep(0.75); c.close()
"""
        writer = subprocess.Popen(
            [sys.executable, "-c", writer_code, str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            pass_fds=(lease,),
        )
        try:
            assert select.select([writer.stdout], [], [], 10)[0]
            assert writer.stdout.readline().strip() == "READY"
            with loader._connect_data_read_only(path) as conn:
                conn.execute("SET threads=1")
                assert conn.execute("SELECT value FROM synthetic").fetchone() == (9,)
        finally:
            stdout, stderr = writer.communicate(timeout=10)
        assert writer.returncode == 0, (stdout, stderr)


def test_symbolic_read_propagates_non_lock_error(monkeypatch):
    import factor_mining.symbolic.data_loader as loader
    from utils import data_store, duckdb_results_writer

    def broken(_path):
        raise ValueError("Input version missing or ambiguous")

    def unexpected_sleep(_seconds):
        raise AssertionError("A provenance failure must not retry")

    monkeypatch.setattr(data_store, "connect_read_only", broken)
    monkeypatch.setattr(duckdb_results_writer.time, "sleep", unexpected_sleep)
    with pytest.raises(ValueError, match="Input version missing or ambiguous"):
        loader._connect_data_read_only("synthetic.duckdb")
