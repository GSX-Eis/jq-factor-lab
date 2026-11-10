"""Offline ingestion contracts, with synthetic files and disposable databases."""
import ast
import json
from pathlib import Path

import duckdb
import pytest

from data_pipeline.loaders.factor_data_to_duckdb import DuckDBFactorImporter
from data_pipeline.loaders.import_index_constituents_duckdb import DuckDBIndexConstituentsImporter
from data_pipeline.loaders.import_raw_financial_statements_duckdb import import_statement
from data_pipeline.loaders.portfolio_data_to_duckdb import DATASETS, import_dataset
from utils.factor_base import FactorBase


def test_online_imports_stay_in_cloud_runtime(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for name in ("utils", "data_pipeline", "factor_synthesis"):
        for path in (root / name).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                           [item.name for item in node.names] if isinstance(node, ast.Import) else [])
                assert not any(m and m.split(".")[0] == "jqdatasdk" for m in modules)
                if path.name not in {"jq_factor_adapter.py", "factor_base.py"}:
                    assert not any(m and m.split(".")[0] in {"jqdata", "jqfactor"} for m in modules)
    monkeypatch.setenv("JQ_RUNNING", "1")
    assert FactorBase._is_cloud_environment(None) is True


def test_index_import_upserts_and_bad_replace_rolls_back(tmp_path):
    source = tmp_path / "index's.csv"
    source.write_text("date,stock_code,index_code\n2027-01-04,DEMO001,INDEX001\n")
    importer = DuckDBIndexConstituentsImporter(str(tmp_path / "index.duckdb"))
    try:
        importer.import_csv(str(source))
        importer.import_csv(str(source))
        assert importer.conn.execute("select count(*) from index_constituents").fetchone()[0] == 1
        before = importer.conn.execute("select * from index_constituents").fetchall()
        source.write_text("wrong_column\ninvalid\n")
        with pytest.raises(duckdb.Error):
            importer.import_csv(str(source), replace=True)
        assert importer.conn.execute("select * from index_constituents").fetchall() == before
    finally:
        importer.close()


def test_factor_import_upserts_and_invalid_file_rolls_back(tmp_path):
    source = tmp_path / "factor's"
    source.mkdir()
    csv = source / "DEMO001.csv"
    csv.write_text("date,signal\n2027-01-04,1.5\n")
    importer = DuckDBFactorImporter(str(tmp_path / "factors.duckdb"))
    try:
        importer.import_csv_files(str(source))
        csv.write_text("date,signal\n2027-01-04,2.5\n")
        importer.import_csv_files(str(source))
        assert importer.conn.execute("select stock_code,signal from factor_data").fetchall() == [("DEMO001", 2.5)]
        before = importer.conn.execute("select * from factor_data").fetchall()
        csv.write_text("wrong_column\ninvalid\n")
        with pytest.raises(ValueError, match="DEMO001.csv missing date"):
            importer.import_csv_files(str(source))
        assert importer.conn.execute("select * from factor_data").fetchall() == before
    finally:
        importer.close()


def test_portfolio_import_upserts_and_replace_is_atomic(tmp_path):
    source = tmp_path / "local's"
    source.mkdir()
    csv = source / "security_master.csv"
    csv.write_text("stock_code,display_name,name,start_date,end_date,security_type\nDEMO001,Demo,Demo,2027-01-04,2099-01-01,stock\n")
    with duckdb.connect() as conn:
        import_dataset(conn, DATASETS["security_master"], source, False)
        import_dataset(conn, DATASETS["security_master"], source, False)
        before = conn.execute("select * from security_master").fetchall()
        assert len(before) == 1
        csv.write_text("wrong_column\ninvalid\n")
        with pytest.raises(duckdb.Error):
            import_dataset(conn, DATASETS["security_master"], source, True)
        assert conn.execute("select * from security_master").fetchall() == before


def test_raw_financial_import_has_no_private_date_window(tmp_path):
    (tmp_path / "income").mkdir()
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"income": {"columns": ["id", "code", "pub_date", "end_date", "revenue"]}}))
    csv = tmp_path / "income/raw_income_statement_2027.csv"
    csv.write_text("id,code,pub_date,end_date,revenue\n1,DEMO001,2010-01-04,2009-12-31,10\n2,DEMO001,2027-01-04,2026-12-31,20\n")
    with duckdb.connect() as conn:
        import_statement(conn, "income", tmp_path, schema)
        import_statement(conn, "income", tmp_path, schema)
        assert conn.execute("select id,revenue from raw_income_statement_events order by id").fetchall() == [(1, 10.0), (2, 20.0)]


def test_factor_column_names_are_quoted(tmp_path):
    import csv
    source = tmp_path / "inputs"
    source.mkdir()
    with (source / "DEMO001.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["date", 'quoted"signal'])
        writer.writerow(["2027-01-04", "2.5"])
    importer = DuckDBFactorImporter(str(tmp_path / "quoted.duckdb"))
    try:
        importer.import_csv_files(str(source))
        assert importer.conn.execute('SELECT "quoted""signal" FROM factor_data').fetchall() == [(2.5,)]
    finally:
        importer.close()


def test_price_resume_and_failed_batch_preserve_existing_data(tmp_path):
    from data_pipeline.loaders.price_data_to_duckdb import DuckDBPriceImporter
    source = tmp_path / "daily"
    source.mkdir()
    (source / "DEMO001.csv").write_text(
        "date,open,close,high,low,volume,money,paused,high_limit,low_limit,pre_close,factor\n"
        "2027-01-04,10,10,11,9,100,1000,0,11,9,10,1\n"
    )
    importer = DuckDBPriceImporter(str(tmp_path / "prices.duckdb"))
    try:
        importer.connect()
        importer.init_tables()
        assert importer.import_directory("daily", str(source), "price_data_daily")[2] == 1
        assert importer.import_directory("daily", str(source), "price_data_daily")[2] == 0
        before = importer.conn.execute("select * from price_data_daily").fetchall()
        (source / "DEMO002.csv").write_text("date,close\n2027-01-04,wrong\n")
        with pytest.raises(ValueError, match="DEMO002.csv missing"):
            importer.import_directory("daily", str(source), "price_data_daily")
        assert importer.conn.execute("select * from price_data_daily").fetchall() == before
        assert importer.conn.execute("select count(*) from import_progress").fetchone()[0] == 1
    finally:
        importer.close()
