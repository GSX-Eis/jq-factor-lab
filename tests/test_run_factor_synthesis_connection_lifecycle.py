from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

import run_factor_synthesis


class _ClosableConn:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ExecuteResult:
    def fetchone(self):
        return [0]


class _SessionConn:
    def execute(self, sql, params=None):
        return _ExecuteResult()


class _RecordingWriter:
    def __init__(self, owner, events: list[str]) -> None:
        self.owner = owner
        self.events = events

    @contextmanager
    def duckdb_write_session(self, retry_interval_sec: int = 30):
        self.events.append(f"session:price_conn_is_none={self.owner.price_conn is None}")
        yield SimpleNamespace(conn=_SessionConn())

    def write(self, factor_name: str, results: dict, *, duckdb_conn=None):
        self.events.append(f"write:{factor_name}:{duckdb_conn is not None}")


def _make_synth() -> run_factor_synthesis.FactorSynthesizer:
    synth = run_factor_synthesis.FactorSynthesizer.__new__(run_factor_synthesis.FactorSynthesizer)
    synth.logger = run_factor_synthesis.logging.getLogger("test_run_factor_synthesis_connection_lifecycle")
    synth.factor_table = "factor_values"
    synth.price_table = "price_data_daily"
    synth.fundamental_table = "factor_data"
    synth.config = {
        "database": {
            "local_db": {
                "results_table": "factor_analysis_results",
                "duckdb_path": "/tmp/local.duckdb",
            }
        }
    }
    synth.factor_duckdb_path = "/tmp/factor.duckdb"
    synth.price_duckdb_path = "/tmp/price.duckdb"
    synth.local_duckdb_path = "/tmp/local.duckdb"
    synth.local_conn = None
    synth.factor_conn = None
    synth.price_conn = None
    synth._conn_paths = {
        "local_conn": synth.local_duckdb_path,
        "factor_conn": synth.factor_duckdb_path,
        "price_conn": synth.price_duckdb_path,
    }
    synth.pit_config = {"enabled": True}
    return synth


def test_load_factor_data_uses_temporary_factor_reader(monkeypatch) -> None:
    synth = _make_synth()
    connect_calls: list[_ClosableConn] = []

    def fake_connect(path: str, read_only: bool = False):
        conn = _ClosableConn(f"{path}|read_only={read_only}")
        connect_calls.append(conn)
        return conn

    monkeypatch.setattr(run_factor_synthesis, "connect_read_only_with_retry", lambda path: fake_connect(path, read_only=True))

    def fake_core_load_factor_data(**kwargs):
        factor_conn = kwargs["factor_conn"]
        price_conn = kwargs["price_conn"]
        assert factor_conn is connect_calls[0]
        assert price_conn is connect_calls[1]
        assert factor_conn.closed is False
        return pd.DataFrame({"alpha": [1.0]})

    monkeypatch.setattr(run_factor_synthesis, "core_load_factor_data", fake_core_load_factor_data)

    result = synth.load_factor_data(["alpha"], "2024-01-01", "2024-01-31")

    assert not result.empty
    assert connect_calls[0].closed is True
    assert synth.price_conn is connect_calls[1]
    assert synth.price_conn.closed is False


def test_release_result_db_readers_closes_shared_price_reader() -> None:
    synth = _make_synth()
    synth.price_duckdb_path = synth.local_duckdb_path
    price_conn = _ClosableConn("price")
    synth.price_conn = price_conn
    synth._conn_paths["price_conn"] = synth.local_duckdb_path

    synth._release_result_db_readers()

    assert price_conn.closed is True
    assert synth.price_conn is None


def test_save_analysis_results_releases_shared_reader_before_write(monkeypatch) -> None:
    synth = _make_synth()
    synth.price_duckdb_path = synth.local_duckdb_path
    price_conn = _ClosableConn("price")
    synth.price_conn = price_conn
    synth._conn_paths["price_conn"] = synth.local_duckdb_path

    events: list[str] = []
    synth.result_writer = _RecordingWriter(synth, events)

    monkeypatch.setattr(run_factor_synthesis, "MONTHLY_WRITER_AVAILABLE", False)

    ok = synth._save_analysis_results(
        "demo_factor",
        {
            "ic": pd.DataFrame(
                {"1D": [0.1, 0.2]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            )
        },
    )

    assert ok is True
    assert price_conn.closed is True
    assert events == [
        "session:price_conn_is_none=True",
        "session:price_conn_is_none=True",
        "write:demo_factor:True",
    ]


def test_run_synthesis_fails_closed_for_size_conditional_method(monkeypatch) -> None:
    synth = _make_synth()
    synth.config["synthesis_methods"] = {
        "conditional": {"method": "size_conditional_elasticnet"}
    }

    def fail(*_args, **_kwargs):
        raise ValueError("no aligned rows")

    monkeypatch.setattr(synth, "rolling_window_synthesis", fail)

    with pytest.raises(ValueError, match="no aligned rows"):
        synth.run_synthesis()
