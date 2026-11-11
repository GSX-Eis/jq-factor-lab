"""Reject ambiguous raw column identities before DuckDB silently renames them."""
import pytest
from pandas.testing import assert_frame_equal

from data_pipeline.loaders.price_data_to_duckdb import DuckDBPriceImporter


@pytest.mark.parametrize("frequency", ["daily", "5m"])
@pytest.mark.parametrize("duplicate", ["date", "open", "factor", "extra"])
def test_duplicate_price_header_rejected_before_mutation(tmp_path, frequency, duplicate):
    owner = DuckDBPriceImporter(str(tmp_path / "input.duckdb"))
    owner.connect()
    owner.init_tables()
    directory = tmp_path / "source"
    directory.mkdir()
    date = "2024-01-02" + (" 09:35:00" if frequency == "5m" else "")
    headers = ["date", "open", "close", "high", "low", "volume", "money", "factor", "extra"]
    values = [date, "10", "12", "13", "9", "100", "1100", "0.5", "first"]
    seed = directory / "A.csv"
    seed.write_text(",".join(headers) + "\n" + ",".join(values) + "\n", encoding="utf-8-sig")
    table = "price_data_5m" if frequency == "5m" else "price_data_daily"
    try:
        assert owner.import_directory(frequency, str(directory), table) == (1, 1, 1)
        queries = [f"SELECT * FROM {table} ORDER BY stock_code,date",
                   "SELECT * FROM import_progress ORDER BY file_name",
                   "SELECT * FROM meta.cache_revision"]
        before = [owner.conn.execute(sql).df() for sql in queries]
        second = {"date": "2099-01-01", "open": "99", "factor": "9", "extra": "second"}[duplicate]
        bad = directory / "B.csv"
        bad.write_text(",".join(headers + [duplicate]) + "\n" + ",".join(values + [second]) + "\n")
        with pytest.raises(ValueError, match=rf"B\.csv.*duplicate column.*{duplicate}"):
            owner.import_directory(frequency, str(directory), table)
        for sql, old in zip(queries, before):
            assert_frame_equal(old, owner.conn.execute(sql).df(), check_exact=True)
        assert owner.get_imported_files() == {f"{frequency}/A.csv"}
    finally:
        owner.close()
