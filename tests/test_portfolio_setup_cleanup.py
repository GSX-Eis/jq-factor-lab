"""A failed connection setup must release the acquired database handle."""
import sys

import duckdb
import pytest

from data_pipeline.loaders import portfolio_data_to_duckdb as loader


@pytest.mark.parametrize("statement", ["SET threads TO 4", "SET memory_limit = '4GB'"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_setup_failure_closes_connection_and_preserves_exception(monkeypatch, tmp_path, statement, error_type):
    conn = duckdb.connect(str(tmp_path / "input.duckdb"))
    conn.execute("CREATE TABLE sentinel AS SELECT 42 AS value")
    failure = error_type("setup interrupted")
    close_calls = []
    imported = []

    class FailingSetup:
        def execute(self, sql):
            if sql == statement:
                raise failure
            return conn.execute(sql)

        def close(self):
            close_calls.append(True)
            conn.close()

    monkeypatch.setattr(loader.duckdb, "connect", lambda *a, **k: FailingSetup())
    monkeypatch.setattr(loader, "import_dataset", lambda *a, **k: imported.append(True))
    monkeypatch.setattr(sys, "argv", ["loader", "--dataset", "security_master", "--db", str(tmp_path / "input.duckdb")])
    try:
        with pytest.raises(error_type) as caught:
            loader.main()
        assert caught.value is failure
        assert imported == []
        assert close_calls == [True]
        with pytest.raises(duckdb.ConnectionException):
            conn.execute("SELECT * FROM sentinel")
    finally:
        conn.close()
