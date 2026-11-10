"""Connections are published only after configuration succeeds."""

import duckdb
import pytest

from utils.duckdb_adapter import DuckDBAdapter
from utils.data_store import migrate_table


@pytest.fixture(params=["file", "bound", "parquet"])
def adapter(tmp_path, monkeypatch, request):
    for name in ("QUANT_DATA_STORE_MEMORY_LIMIT", "QUANT_DATA_STORE_THREADS", "QUANTATION_DUCKDB_MEMORY_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    if request.param == "parquet":
        reader = DuckDBAdapter(parquet_dir=str(tmp_path), data_bindings={})
    else:
        source = tmp_path / "source.duckdb"
        with duckdb.connect(str(source)) as conn:
            conn.execute("CREATE TABLE marker AS SELECT 7 AS value")
        binding = {}
        if request.param == "bound":
            target = tmp_path / "input.duckdb"
            ref = migrate_table(source, target, "marker", role="input", execute=True)["ref"]
            source.unlink()
            source = target
            binding = {"input_db": str(target), "inputs": {"marker": ref}}
        reader = DuckDBAdapter(duckdb_path=str(source), data_bindings=binding)
    yield reader
    reader.close()


@pytest.mark.parametrize("variable,invalid,fixed", [
    ("QUANT_DATA_STORE_MEMORY_LIMIT", "invalid-memory-limit", "64MiB"),
    ("QUANT_DATA_STORE_THREADS", "not-an-integer", "1"),
])
def test_configuration_failure_closes_connections_and_allows_retry(adapter, monkeypatch, variable, invalid, fixed):
    opened = []
    real_connect = duckdb.connect

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(duckdb, "connect", connect)
    monkeypatch.setenv(variable, invalid)
    try:
        with pytest.raises((duckdb.Error, ValueError)):
            adapter.execute_sql("SELECT 1")
        assert adapter._conn is None
        assert opened
        for conn in opened:
            with pytest.raises(duckdb.ConnectionException):
                conn.execute("SELECT 1")
        monkeypatch.setenv(variable, fixed)
        assert adapter.execute_sql("SELECT 7 AS value").value.tolist() == [7]
    finally:
        adapter.close()


@pytest.mark.parametrize("custom", [False, True])
def test_all_sources_share_resource_budget(adapter, monkeypatch, custom):
    # The retired adapter-only variable must not override the shared budget.
    monkeypatch.setenv("QUANTATION_DUCKDB_MEMORY_LIMIT", "invalid-old-value")
    if custom:
        monkeypatch.setenv("QUANT_DATA_STORE_MEMORY_LIMIT", "64MiB")
        monkeypatch.setenv("QUANT_DATA_STORE_THREADS", "1")
    settings = adapter.execute_sql(
        "SELECT current_setting('memory_limit') AS memory, current_setting('threads') AS threads"
    ).iloc[0]
    assert settings.memory.replace(" ", "") == ("64.0MiB" if custom else "2.7GiB")
    assert settings.threads == (1 if custom else 2)


def test_connection_reuse_close_and_reopen(adapter, monkeypatch):
    monkeypatch.setenv("QUANT_DATA_STORE_THREADS", "1")
    first = adapter._get_connection()
    monkeypatch.setenv("QUANT_DATA_STORE_THREADS", "2")
    assert adapter._get_connection() is first
    assert first.execute("SELECT current_setting('threads')").fetchone() == (1,)
    cache = adapter._cache
    adapter.close()
    adapter.close()
    assert adapter._conn is None
    assert adapter._cache is cache
    with pytest.raises(duckdb.ConnectionException):
        first.execute("SELECT 1")
    reopened = adapter._get_connection()
    assert reopened is not first
    assert reopened.execute("SELECT current_setting('threads')").fetchone() == (2,)


@pytest.mark.parametrize("sql", ["SELECT * FROM missing_table", "SELECT FROM"])
def test_sql_errors_propagate_and_connection_remains_usable(adapter, sql):
    with pytest.raises(duckdb.Error):
        adapter.execute_sql(sql)
    assert adapter.execute_sql("SELECT 7 AS value").value.tolist() == [7]


@pytest.mark.parametrize("sql,params", [
    ("SELECT ? AS stock_code", ["A'; DROP TABLE marker; --"]),
    ("SELECT $stock AS stock_code", {"stock": "A'; DROP TABLE marker; --"}),
])
def test_sql_parameters_are_literal_values_and_columns_are_unchanged(adapter, sql, params):
    result = adapter.execute_sql(sql, params)
    assert list(result.columns) == ["stock_code"]
    assert result.stock_code.tolist() == ["A'; DROP TABLE marker; --"]


def test_sql_empty_result_preserves_schema_and_bad_parameters_raise(adapter):
    result = adapter.execute_sql("SELECT ? AS stock_code WHERE FALSE", params=["A"])
    assert result.empty
    assert list(result.columns) == ["stock_code"]
    with pytest.raises(duckdb.Error):
        adapter.execute_sql("SELECT ? AS value", [])


def test_sql_session_commands_remain_supported_and_reads_are_uncached(adapter):
    assert not adapter.execute_sql("PRAGMA database_list").empty
    adapter.execute_sql("SET threads=1")
    assert adapter.execute_sql("SELECT current_setting('threads') AS value").value.tolist() == [1]
    adapter.execute_sql("SET threads=2")
    assert adapter.execute_sql("SELECT current_setting('threads') AS value").value.tolist() == [2]
