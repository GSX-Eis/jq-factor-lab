"""Source selection must never silently discard an explicit adapter argument."""

import json

import duckdb
import pytest

from utils.duckdb_adapter import DuckDBAdapter, get_data_adapter
from utils.data_store import migrate_table


@pytest.fixture(autouse=True)
def isolated_binding_environment(monkeypatch):
    monkeypatch.delenv("QUANT_DATA_BINDINGS", raising=False)


@pytest.mark.parametrize("create", [get_data_adapter, DuckDBAdapter])
@pytest.mark.parametrize("from_environment", [False, True])
def test_binding_rejects_parquet_instead_of_ignoring_it(tmp_path, monkeypatch, create, from_environment):
    database = tmp_path / "input.duckdb"
    with duckdb.connect(str(database)):
        pass
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    contract = {"input_db": str(database)}
    if from_environment:
        monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(contract))
    with pytest.raises(ValueError, match="parquet_dir.*data_bindings"):
        create(parquet_dir=str(parquet), data_bindings=None if from_environment else contract)


@pytest.mark.parametrize("create", [get_data_adapter, DuckDBAdapter])
@pytest.mark.parametrize("from_environment", [False, True])
def test_binding_rejects_unmapped_database_path(tmp_path, monkeypatch, create, from_environment):
    requested, bound = tmp_path / "requested.duckdb", tmp_path / "input.duckdb"
    for path in (requested, bound):
        with duckdb.connect(str(path)):
            pass
    contract = {"input_db": str(bound)}
    if from_environment:
        monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(contract))
    with pytest.raises(ValueError, match="Database path has no frozen binding"):
        create(duckdb_path=str(requested), data_bindings=None if from_environment else contract)


@pytest.mark.parametrize("create", [get_data_adapter, DuckDBAdapter])
def test_empty_binding_reads_explicit_database_despite_environment(tmp_path, monkeypatch, create):
    requested = tmp_path / "requested.duckdb"
    with duckdb.connect(str(requested)) as conn:
        conn.execute("CREATE TABLE marker AS SELECT 7 AS value")
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"input_db": str(tmp_path / "missing.duckdb")}))
    adapter = create(duckdb_path=str(requested), data_bindings={})
    try:
        assert adapter.execute_sql("SELECT value FROM marker").value.tolist() == [7]
    finally:
        adapter.close()


@pytest.fixture
def versioned_inputs(tmp_path):
    old, current, store = (tmp_path / name for name in ("old.duckdb", "current.duckdb", "input.duckdb"))
    refs = []
    for path, value in ((old, 1), (current, 2)):
        with duckdb.connect(str(path)) as conn:
            conn.execute("CREATE TABLE marker AS SELECT ? AS value", [value])
        refs.append(migrate_table(path, store, "marker", role="input", execute=True)["ref"])
        path.unlink()
    return old, store, {
        "input_db": str(store),
        "inputs": {"marker": refs[1]},
    }, {"marker": refs[0]}


@pytest.mark.parametrize("create", [get_data_adapter, DuckDBAdapter])
@pytest.mark.parametrize("source", ["bound_only", "same_path", "frozen_version"])
def test_binding_preserves_selected_version(versioned_inputs, monkeypatch, create, source):
    old, store, contract, frozen_inputs = versioned_inputs
    if source == "frozen_version":
        contract = {**contract, "inputs": frozen_inputs}
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(contract))
    requested = None if source == "bound_only" else str(store)
    adapter = create(duckdb_path=requested)
    try:
        expected = 1 if source == "frozen_version" else 2
        assert adapter.execute_sql("SELECT value FROM marker").value.tolist() == [expected]
    finally:
        adapter.close()


@pytest.mark.parametrize("create", [get_data_adapter, DuckDBAdapter])
def test_explicit_binding_wins_over_environment(versioned_inputs, monkeypatch, create):
    _, _, contract, _ = versioned_inputs
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"input_db": "/missing/environment.duckdb"}))
    adapter = create(data_bindings=contract)
    try:
        assert adapter.execute_sql("SELECT value FROM marker").value.tolist() == [2]
    finally:
        adapter.close()


@pytest.mark.parametrize("create", [get_data_adapter, DuckDBAdapter])
def test_empty_binding_allows_parquet_read(tmp_path, monkeypatch, create):
    parquet = tmp_path / "parquet"
    partition = parquet / "price_data_daily" / "year=2026" / "month=1"
    partition.mkdir(parents=True)
    with duckdb.connect() as conn:
        conn.execute(
            "COPY (SELECT DATE '2026-01-05' AS date, 'A' AS stock_code, 7.0 AS close) TO ? (FORMAT PARQUET)",
            [str(partition / "part.parquet")],
        )
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"input_db": str(tmp_path / "missing.duckdb")}))
    adapter = create(parquet_dir=str(parquet), data_bindings={})
    try:
        result = adapter.fetch_price_data(["A"], "2026-01-05", "2026-01-05", ["close"])
        assert result.close.tolist() == [7.0]
    finally:
        adapter.close()


@pytest.mark.parametrize("create", [get_data_adapter, DuckDBAdapter])
def test_binding_rejects_output_store(tmp_path, create):
    input_db, output_db, old = (tmp_path / name for name in ("input.duckdb", "output.duckdb", "old.duckdb"))
    contract = {
        "input_db": str(input_db), "output_db": str(output_db),
    }
    with pytest.raises(ValueError, match="must select the input store"):
        create(duckdb_path=str(output_db), data_bindings=contract)


def test_factory_returns_none_for_no_source_even_when_environment_binding_is_disabled(monkeypatch):
    assert get_data_adapter() is None
    monkeypatch.setenv("QUANT_DATA_BINDINGS", '{"input_db": "/missing/environment.duckdb"}')
    assert get_data_adapter(data_bindings={}) is None


@pytest.mark.parametrize("create", [get_data_adapter, DuckDBAdapter])
def test_missing_bound_database_fails_without_creating_it(tmp_path, create):
    missing = tmp_path / "missing.duckdb"
    with pytest.raises(FileNotFoundError):
        create(data_bindings={"input_db": str(missing)})
    assert not missing.exists()
