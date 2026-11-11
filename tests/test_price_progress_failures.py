"""Progress failures must not turn failed imports into successful commits."""

import duckdb
import pytest
from pandas.testing import assert_frame_equal

from data_pipeline.loaders.price_data_to_duckdb import DuckDBPriceImporter


@pytest.fixture
def importer(tmp_path):
    owner = DuckDBPriceImporter(str(tmp_path / "input.duckdb"))
    owner.connect()
    owner.init_tables()
    try:
        yield owner
    finally:
        owner.close()


def csv(directory, name="A"):
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.csv").write_text(
        "date,open,close,high,low,volume,money\n"
        "2024-01-02,10,12,13,9,100,1100\n"
    )


@pytest.mark.parametrize("operation", ["read", "write"])
def test_progress_sql_failure_propagates(importer, operation):
    if operation == "read":
        importer.conn.execute("DROP TABLE import_progress")
        with pytest.raises(duckdb.CatalogException):
            importer.get_imported_files()
    else:
        importer.conn.execute("ALTER TABLE import_progress RENAME COLUMN import_date TO bad_date")
        with pytest.raises(duckdb.BinderException):
            importer.mark_file_imported("daily/A.csv", 1)


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_progress_read_preserves_cancellation(importer, error):
    real = importer.conn

    class InterruptedRead:
        def execute(self, sql):
            if "SELECT file_name FROM import_progress" in sql:
                raise error("cancel progress read")
            return real.execute(sql)

    importer.conn = InterruptedRead()
    try:
        with pytest.raises(error, match="cancel progress read"):
            importer.get_imported_files()
    finally:
        importer.conn = real


@pytest.mark.parametrize("operation", ["read", "write"])
def test_directory_progress_failure_rolls_back_rows_and_revision(importer, tmp_path, operation):
    original = tmp_path / "original"
    csv(original)
    assert importer.import_directory("daily", str(original), "price_data_daily") == (1, 1, 1)
    incoming = tmp_path / "incoming"
    csv(incoming, "B")
    bad_column = "imported" if operation == "read" else "import_date"
    importer.conn.execute(f"ALTER TABLE import_progress RENAME COLUMN {bad_column} TO broken_column")
    queries = [
        "SELECT * FROM price_data_daily ORDER BY stock_code,date",
        "SELECT * FROM import_progress ORDER BY file_name",
        "SELECT * FROM meta.cache_revision",
    ]
    before = [importer.conn.execute(sql).df() for sql in queries]
    expected = duckdb.BinderException if operation == "read" else RuntimeError
    with pytest.raises(expected):
        importer.import_directory("daily", str(incoming), "price_data_daily")
    for sql, frame in zip(queries, before):
        assert_frame_equal(frame, importer.conn.execute(sql).df(), check_exact=True)
    # The transaction is closed and the same owner can recover after schema repair.
    importer.conn.execute(f"ALTER TABLE import_progress RENAME COLUMN broken_column TO {bad_column}")
    assert importer.import_directory("daily", str(incoming), "price_data_daily") == (1, 1, 1)
    assert importer.get_imported_files() == {"daily/A.csv", "daily/B.csv"}
