"""Retired aliases fail before a reader can silently select another snapshot."""
import json

import pytest

from factor_synthesis.factor_reader import FactorReader
from utils.duckdb_adapter import get_data_adapter


@pytest.mark.parametrize("environment", [False, True])
def test_legacy_contract_rejected_even_with_canonical_path(tmp_path, monkeypatch, environment):
    store = tmp_path / "input.duckdb"
    contract = {"input_db": str(store), "sources": {str(tmp_path / "old.duckdb"): {"role": "input"}}}
    if environment:
        monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps(contract))
    with pytest.raises(ValueError, match="Retired.*sources"):
        get_data_adapter(duckdb_path=str(store), data_bindings=None if environment else contract)
    assert not store.exists()


@pytest.mark.parametrize("parameter", ["database_path", "results_database_path"])
def test_factor_reader_never_replaces_unmatched_path(tmp_path, parameter):
    store = tmp_path / "output.duckdb"
    kwargs = {"database_path": str(store), "results_database_path": str(store)}
    kwargs[parameter] = str(tmp_path / "unknown.duckdb")
    with pytest.raises(ValueError, match="Database path has no frozen binding"):
        FactorReader(**kwargs, data_bindings={"output_db": str(store)})
    assert not store.exists()
