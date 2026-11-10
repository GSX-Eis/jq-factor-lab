"""Regression coverage for atomic price and factor CSV imports."""

import sys
from pathlib import Path

import duckdb
import pytest


from data_pipeline.loaders.factor_data_to_duckdb import (
    DuckDBFactorImporter,
    main as factor_main,
)
from data_pipeline.loaders.price_data_to_duckdb import (
    DuckDBPriceImporter,
    main as price_main,
)


PRICE_HEADER = "date,open,close,high,low,volume,money\n"
PRICE_ROW = "2024-01-02,10,10,11,9,100,1000\n"


def _write_price_csv(path: Path, *, valid: bool) -> None:
    if valid:
        path.write_text(PRICE_HEADER + PRICE_ROW, encoding="utf-8")
    else:
        path.write_text(
            "date,open,close,high,low,volume\n2024-01-02,10,10,11,9,100\n",
            encoding="utf-8",
        )


def _write_factor_csv(path: Path, *, valid: bool) -> None:
    if valid:
        path.write_text("date,alpha\n2024-01-02,1.25\n", encoding="utf-8")
    else:
        path.write_text("alpha\n1.25\n", encoding="utf-8")


def _write_price_type_error(path: Path) -> None:
    path.write_text(
        PRICE_HEADER + "2024-01-02,not-a-number,10,11,9,100,1000\n",
        encoding="utf-8",
    )


def _write_factor_type_error(path: Path) -> None:
    path.write_text("date,alpha\n2024-01-02,not-a-number\n", encoding="utf-8")


def test_price_header_validation_rolls_back_all_batches_and_resumes(tmp_path: Path) -> None:
    """A malformed 33rd+ file cannot be masked by union_by_name or progress rows."""
    source = tmp_path / "price" / "daily"
    source.mkdir(parents=True)
    for number in range(1, 34):
        _write_price_csv(source / f"{number:06d}.XSHE.csv", valid=True)
    bad = source / "ZZ_BAD.csv"
    _write_price_csv(bad, valid=False)
    database = tmp_path / "price.duckdb"

    failed = DuckDBPriceImporter(db_path=str(database))
    with pytest.raises(ValueError, match="ZZ_BAD.csv missing money"):
        failed.run_import(data_dir_daily=str(source), data_dir_5m=str(tmp_path / "price" / "5m"))
    failed.close()

    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM price_data_daily").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM import_progress").fetchone() == (0,)

    _write_price_csv(bad, valid=True)
    repaired = DuckDBPriceImporter(db_path=str(database))
    stats = repaired.run_import(
        data_dir_daily=str(source), data_dir_5m=str(tmp_path / "price" / "5m")
    )
    assert stats["daily"] == {"total": 34, "success": 34, "records": 34}
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM price_data_daily").fetchone() == (34,)
        assert connection.execute("SELECT COUNT(*) FROM import_progress").fetchone() == (34,)


def test_factor_header_validation_rolls_back_all_batches_and_resumes(tmp_path: Path) -> None:
    """Every factor CSV must carry its own date column across a 32-file boundary."""
    source = tmp_path / "factor"
    source.mkdir()
    for number in range(1, 34):
        _write_factor_csv(source / f"{number:06d}.XSHE.csv", valid=True)
    bad = source / "ZZ_BAD.csv"
    _write_factor_csv(bad, valid=False)
    database = tmp_path / "factor.duckdb"

    failed = DuckDBFactorImporter(db_path=str(database))
    with pytest.raises(ValueError, match="ZZ_BAD.csv missing date"):
        failed.run_import(str(source))
    failed.close()

    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'factor_data'"
        ).fetchone() == (0,)

    _write_factor_csv(bad, valid=True)
    repaired = DuckDBFactorImporter(db_path=str(database))
    stats = repaired.run_import(str(source))
    assert stats["total_files"] == 34
    assert stats["successful_files"] == 34
    assert stats["total_records"] == 34
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM factor_data").fetchone() == (34,)


def test_price_late_type_error_rolls_back_batches_and_progress(tmp_path: Path) -> None:
    """A bad 34th record rolls back earlier batch data and recovery state."""
    database = tmp_path / "price.duckdb"
    baseline_source = tmp_path / "baseline-price" / "daily"
    baseline_source.mkdir(parents=True)
    _write_price_csv(baseline_source / "EXISTING.XSHE.csv", valid=True)
    DuckDBPriceImporter(db_path=str(database)).run_import(
        data_dir_daily=str(baseline_source), data_dir_5m=str(tmp_path / "baseline-price" / "5m")
    )

    source = tmp_path / "bad-price" / "daily"
    source.mkdir(parents=True)
    for number in range(1, 34):
        _write_price_csv(source / f"{number:06d}.XSHE.csv", valid=True)
    _write_price_type_error(source / "ZZ_BAD.csv")

    failed = DuckDBPriceImporter(db_path=str(database))
    with pytest.raises(RuntimeError, match="rolled back"):
        failed.run_import(data_dir_daily=str(source), data_dir_5m=str(tmp_path / "bad-price" / "5m"))
    failed.close()

    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            "SELECT stock_code FROM price_data_daily ORDER BY stock_code"
        ).fetchall() == [("EXISTING.XSHE",)]
        assert connection.execute(
            "SELECT file_name FROM import_progress ORDER BY file_name"
        ).fetchall() == [("daily/EXISTING.XSHE.csv",)]


def test_factor_late_type_error_rolls_back_batches(tmp_path: Path) -> None:
    """A bad 34th factor value cannot leave the first 32 rows committed."""
    database = tmp_path / "factor.duckdb"
    baseline_source = tmp_path / "baseline-factor"
    baseline_source.mkdir()
    _write_factor_csv(baseline_source / "EXISTING.XSHE.csv", valid=True)
    DuckDBFactorImporter(db_path=str(database)).run_import(str(baseline_source))

    source = tmp_path / "bad-factor"
    source.mkdir()
    for number in range(1, 34):
        _write_factor_csv(source / f"{number:06d}.XSHE.csv", valid=True)
    _write_factor_type_error(source / "ZZ_BAD.csv")

    failed = DuckDBFactorImporter(db_path=str(database))
    with pytest.raises(Exception, match="not-a-number|Conversion Error"):
        failed.run_import(str(source))
    failed.close()

    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            "SELECT stock_code, alpha FROM factor_data ORDER BY stock_code"
        ).fetchall() == [("EXISTING.XSHE", 1.25)]


def test_price_import_accepts_a_source_path_containing_an_apostrophe(tmp_path: Path) -> None:
    """Batch CSV paths are SQL-escaped before they reach DuckDB's reader."""
    source = tmp_path / "owner's-price" / "daily"
    source.mkdir(parents=True)
    _write_price_csv(source / "000001.XSHE.csv", valid=True)
    database = tmp_path / "quoted-path.duckdb"

    importer = DuckDBPriceImporter(db_path=str(database))
    stats = importer.run_import(
        data_dir_daily=str(source), data_dir_5m=str(tmp_path / "owner's-price" / "5m")
    )
    assert stats["daily"] == {"total": 1, "success": 1, "records": 1}
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM price_data_daily").fetchone() == (1,)


def test_factor_import_quotes_source_factor_column_names(tmp_path: Path) -> None:
    """Source factor labels remain valid identifiers in CREATE and INSERT SQL."""
    source = tmp_path / "factor-columns"
    source.mkdir()
    (source / "000001.XSHE.csv").write_text(
        'date,alpha"score\n2024-01-02,1.25\n', encoding="utf-8"
    )
    database = tmp_path / "factor-columns.duckdb"

    importer = DuckDBFactorImporter(db_path=str(database))
    stats = importer.run_import(str(source))
    assert stats["total_records"] == 1
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            'SELECT "alpha""score" FROM factor_data'
        ).fetchone() == (1.25,)


@pytest.mark.parametrize("missing", [True, False])
def test_price_cli_rejects_missing_and_empty_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    """The command must not report success when neither frequency has input files."""
    source = tmp_path / "missing" if missing else tmp_path / "empty"
    if not missing:
        (source / "daily").mkdir(parents=True)
        (source / "5m").mkdir()
    database = tmp_path / "price-empty.duckdb"
    monkeypatch.setattr(
        sys,
        "argv",
        ["price-import", "--db", str(database), "--data-dir", str(source)],
    )

    assert price_main() == 1
    assert not database.exists()


@pytest.mark.parametrize("missing", [True, False])
def test_factor_cli_rejects_missing_and_empty_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    """Empty factor directories fail before a database or success summary is created."""
    source = tmp_path / "missing" if missing else tmp_path / "empty"
    if not missing:
        source.mkdir()
    database = tmp_path / "factor-empty.duckdb"
    monkeypatch.setattr(
        sys,
        "argv",
        ["factor-import", "--db", str(database), "--csv-dir", str(source)],
    )

    assert factor_main() == 1
    assert not database.exists()


@pytest.mark.parametrize("kind", ["price", "factor"])
def test_same_importer_can_retry_after_a_failed_operation(tmp_path, kind):
    source = tmp_path / kind
    source.mkdir()
    csv = source / "DEMO.csv"
    if kind == "price":
        importer = DuckDBPriceImporter(str(tmp_path / "retry.duckdb"))
        importer.connect()
        importer.init_tables()
        invoke = lambda: importer.import_directory("daily", str(source), "price_data_daily")
        csv.write_text(PRICE_HEADER + PRICE_ROW.replace(",1000", ",bad-number"))
        valid = PRICE_HEADER + PRICE_ROW
    else:
        importer = DuckDBFactorImporter(str(tmp_path / "retry.duckdb"))
        importer.connect()
        importer.conn.execute("CREATE TABLE factor_data (stock_code VARCHAR, date VARCHAR, alpha DOUBLE)")
        invoke = lambda: importer.import_csv_files(str(source))
        csv.write_text("date,alpha\n2024-01-02,bad-number\n")
        valid = "date,alpha\n2024-01-02,1.25\n"
    try:
        with pytest.raises((RuntimeError, duckdb.Error)):
            invoke()
        csv.write_text(valid)
        assert invoke()[2] == 1
        assert importer.import_failed is False
    finally:
        importer.close()


@pytest.mark.parametrize("failure", ["invalid_header", "interrupted", "export", "connect"])
def test_factor_run_import_closes_owned_connection_on_failure(tmp_path, monkeypatch, failure):
    source = tmp_path / "source"
    source.mkdir()
    csv = source / "A.csv"
    csv.write_text("date,alpha\n2024-01-02,1.25\n")
    parquet = tmp_path / "parquet"
    importer = DuckDBFactorImporter(str(tmp_path / "run.duckdb"), str(parquet), failure == "export")
    opened = []
    connect = importer.connect

    def track_connect():
        conn = connect()
        opened.append(conn)
        if failure == "connect":
            raise RuntimeError("synthetic configuration failure after connect")
        return conn

    monkeypatch.setattr(importer, "connect", track_connect)
    expected = RuntimeError
    if failure == "invalid_header":
        csv.write_text("alpha\n1.25\n")
        expected = ValueError
    elif failure == "interrupted":
        def interrupt(_):
            raise KeyboardInterrupt()
        monkeypatch.setattr(importer, "_validate_csv_headers", interrupt)
        expected = KeyboardInterrupt
    elif failure == "export":
        (parquet / "factor_data").write_text("an existing file blocks directory creation")
        expected = FileExistsError
    try:
        with pytest.raises(expected):
            importer.run_import(str(source))
        assert importer.conn is None
        assert len(opened) == 1
        with pytest.raises(duckdb.ConnectionException, match="closed"):
            opened[0].execute("SELECT 1")
        with duckdb.connect(str(importer.db_path), read_only=True) as check:
            if failure == "export":
                # The CSV commit precedes export; cleanup must not undo committed data.
                assert check.execute("SELECT alpha FROM factor_data").fetchall() == [(1.25,)]
            else:
                assert check.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall() == []
    finally:
        importer.close()


@pytest.mark.parametrize("duplicate", ["date", "alpha", 'alpha"score'])
@pytest.mark.parametrize("existing", [False, True])
def test_factor_rejects_duplicate_source_headers_before_schema_changes(tmp_path, duplicate, existing):
    import csv
    from utils.data_revision import advance_revision, connection_revision

    source = tmp_path / "duplicate-header"
    source.mkdir()
    columns = ["date", "alpha", duplicate]
    if duplicate == 'alpha"score':
        columns = ["date", duplicate, duplicate]
    with (source / "A.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(["2024-01-02", 1, "2024-01-03" if duplicate == "date" else 2])

    importer = DuckDBFactorImporter(str(tmp_path / "duplicate.duckdb"))
    importer.connect()
    try:
        if existing:
            importer.conn.execute("CREATE TABLE factor_data(stock_code VARCHAR,date VARCHAR,alpha DOUBLE)")
            importer.conn.execute("INSERT INTO factor_data VALUES ('OLD','2024-01-01',9)")
            advance_revision(importer.conn)
        before = connection_revision(importer.conn)
        with pytest.raises(ValueError, match="A.csv duplicate column"):
            importer.import_csv_files(str(source))
        assert connection_revision(importer.conn) == before
        if existing:
            assert importer.conn.execute("SELECT * FROM factor_data").fetchall() == [("OLD", "2024-01-01", 9.)]
            assert [r[0] for r in importer.conn.execute("DESCRIBE factor_data").fetchall()] == ["stock_code", "date", "alpha"]
        else:
            assert importer.conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall() == []
    finally:
        importer.close()
