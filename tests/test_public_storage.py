import duckdb
import pandas as pd
import pytest

from utils.data_store import DataRef, migrate_table, read_reference, publish_artifact


def test_versioned_input_is_readable_after_source_changes(tmp_path):
    source, destination = tmp_path / "source.duckdb", tmp_path / "input.duckdb"
    with duckdb.connect(str(source)) as connection:
        connection.execute("CREATE TABLE prices AS SELECT 1 AS value")
    record = migrate_table(source, destination, "prices", role="input", execute=True)
    with duckdb.connect(str(source)) as connection:
        connection.execute("UPDATE prices SET value=2")
    frozen = read_reference(destination, DataRef(**record["ref"]))
    assert frozen.value.tolist() == [1]


def test_label_access_requires_explicit_permission(tmp_path):
    destination = tmp_path / "output.duckdb"
    ref = publish_artifact(destination, "labels", frame=pd.DataFrame({"value": [1., 2.]}),
                           metadata={"run_id": "synthetic-storage-test"}, access_class="label")
    with pytest.raises(PermissionError):
        read_reference(destination, ref)
    assert read_reference(destination, ref, allow_labels=True).value.tolist() == [1., 2.]
