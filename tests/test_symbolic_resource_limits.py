"""Symbolic reads must retain the shared store budget."""
import duckdb

from factor_mining.symbolic import data_loader


def test_symbolic_spill_configuration_preserves_connection_budget(tmp_path, monkeypatch):
    path = tmp_path / "input.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE marker AS SELECT 1 AS value")
    monkeypatch.setenv("QUANT_DATA_STORE_MEMORY_LIMIT", "64MiB")
    monkeypatch.setenv("QUANT_DATA_STORE_THREADS", "1")
    monkeypatch.setenv("QUANT_DATA_BINDINGS", "{}")
    monkeypatch.setattr(data_loader, "_DUCKDB_TEMP_DIR", "")
    with data_loader._connect_data_read_only(path) as conn:
        before = conn.execute("SELECT current_setting('memory_limit'), current_setting('threads')").fetchone()
        data_loader._configure_duckdb_spill(conn)
        assert conn.execute("SELECT current_setting('memory_limit'), current_setting('threads')").fetchone() == before
