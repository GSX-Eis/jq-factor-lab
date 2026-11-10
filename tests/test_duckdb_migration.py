import duckdb
import pytest

from utils.data_store import DataRef, audit_relation, connect_input, migrate_table, read_reference


def test_duckdb_output_migration_keeps_versioned_values(tmp_path):
    first, second, output = (tmp_path / name for name in ("first.duckdb", "second.duckdb", "output.duckdb"))
    for source, pool in ((first, "a"), (second, "b")):
        with duckdb.connect(str(source)) as conn:
            conn.execute(
                "CREATE TABLE monthly_pool_results "
                "(factor_name VARCHAR, pool_type VARCHAR, ic_count BIGINT, payload BLOB)"
            )
            conn.execute(
                "INSERT INTO monthly_pool_results VALUES ('same', ?, ?, ?)",
                [pool, 2**63 - 1, b"\x00\xff"],
            )

    first_result = migrate_table(first, output, "monthly_pool_results", role="output", execute=True)
    second_result = migrate_table(second, output, "monthly_pool_results", role="output", execute=True)

    assert first_result["ref"] != second_result["ref"]
    assert read_reference(output, DataRef(**second_result["ref"]), allow_labels=True).pool_type.tolist() == ["b"]
    with duckdb.connect(str(output), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM meta.artifacts").fetchone()[0] == 2


def test_legacy_sqlite_header_is_rejected_before_duckdb_opens_it(tmp_path):
    legacy = tmp_path / "legacy.db"
    legacy.write_bytes(b"SQLite format 3\x00" + b"\x00" * 128)

    with pytest.raises(ValueError, match="SQLite database files are unsupported"):
        connect_input(legacy)
    with pytest.raises(ValueError, match="SQLite database files are unsupported"):
        migrate_table(legacy, tmp_path / "output.duckdb", "t", role="input", source_format="sqlite")
    with pytest.raises(ValueError, match="SQLite database files are unsupported"):
        migrate_table(legacy, tmp_path / "output.duckdb", "t", role="input")


def test_empty_version_audit_avoids_broken_date_statistics():
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE t AS SELECT 'a' AS version,DATE '2019-01-02' AS date,NULL::TIMESTAMP AS created_at")

        class EmptyDateEngine:
            def execute(self, sql):
                if sql.startswith("SELECT CAST(min("):
                    raise duckdb.InternalException("empty filtered MIN/MAX statistics regression")
                return conn.execute(sql)

        audit = audit_relation(EmptyDateEngine(), "(SELECT * FROM t WHERE version='missing')")
        assert audit["rows"] == 0 and audit["sum_hash"] == audit["xor_hash"] == "0"
        assert audit["dates"] == {"date": [None, None], "created_at": [None, None]}
        assert set(audit["nulls"].values()) == {0}
        assert audit_relation(conn, "t")["dates"]["date"] == ["2019-01-02", "2019-01-02"]
