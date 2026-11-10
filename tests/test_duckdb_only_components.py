"""Focused synthetic coverage for the DuckDB-only persistence boundary."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from backtest_and_analysis.price_fetcher import PriceFetcher
from backtest_and_analysis.result_writer import ResultWriter
from factor_synthesis.factor_reader import FactorReader
from utils.duckdb_results_writer import DuckDBResultsWriter


def _results(dates, *, quantiles: bool) -> dict:
    result = {
        "ic": pd.DataFrame({"1D": [0.1] * len(dates), "3D": [0.2] * len(dates)}, index=pd.to_datetime(dates)),
        "ir": pd.DataFrame({"ir": [0.3, 0.4]}, index=["1D", "3D"]),
        "factor_statistics": {"stock_count": 2},
    }
    if quantiles:
        index = pd.MultiIndex.from_product([[1, 2], pd.to_datetime(dates)], names=["factor_quantile", "date"])
        result["quantile_returns"] = pd.DataFrame({"1D": [0.01] * len(index)}, index=index)
    else:
        result["quantile_returns"] = pd.DataFrame()
    return result


def test_result_writer_explicit_date_range_preserves_other_dates_and_removes_stale_quantiles(tmp_path: Path) -> None:
    path = tmp_path / "results.duckdb"
    writer = ResultWriter(str(path), "factor_analysis_results", export_parquet=False)
    writer.write("alpha", _results(["2024-01-02", "2024-01-03"], quantiles=True))
    writer.write("alpha", _results(["2024-01-03"], quantiles=False), date_range=("2024-01-03", "2024-01-03"))
    with duckdb.connect(str(path), read_only=True) as conn:
        by_date = conn.execute("""
            SELECT CAST(date AS VARCHAR), count(*) AS rows, count(*) FILTER (WHERE quantile IS NOT NULL) AS quantiles
            FROM factor_analysis_results WHERE factor_name='alpha' GROUP BY 1 ORDER BY 1
        """).fetchall()
    assert by_date == [("2024-01-02", 4, 2), ("2024-01-03", 2, 0)]


def test_factor_reader_construction_is_read_only_and_missing_store_fails(tmp_path: Path) -> None:
    path = tmp_path / "factors.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE factor_values(date DATE, stock_code VARCHAR, factor_name VARCHAR, factor_value DOUBLE)")
        conn.execute("INSERT INTO factor_values VALUES ('2024-01-02', '000001.XSHE', 'alpha', 1.0)")
    reader = FactorReader(str(path))
    assert reader.get_factor_values("alpha", "2024-01-02", "2024-01-02").iloc[0, 0] == 1.0
    with duckdb.connect(str(path), read_only=True) as conn:
        assert {row[0] for row in conn.execute("SHOW TABLES").fetchall()} == {"factor_values"}
    with pytest.raises(Exception):
        FactorReader(str(tmp_path / "missing.duckdb"))


def test_price_fetcher_keeps_timestamped_explicit_end_date(tmp_path: Path) -> None:
    path = tmp_path / "prices.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("""CREATE TABLE price_data_daily(date TIMESTAMP, stock_code VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, money DOUBLE)""")
        conn.execute("INSERT INTO price_data_daily VALUES ('2024-01-02 14:55:00', '000001.XSHE', 1, 1, 1, 2, 1, 1)")
    fetcher = PriceFetcher(duckdb_path=path, price_table="price_data")
    frame = fetcher._fetch_from_duckdb(["000001.XSHE"], "2024-01-02", "2024-01-02")
    fetcher.close()
    assert len(frame) == 1


def test_explicit_sqlite_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SQLite result storage"):
        ResultWriter(str(tmp_path / "old.sqlite"), "factor_analysis_results", db_backend="sqlite")


def test_writer_rejects_legacy_sqlite_file_before_opening_connection(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    path.write_bytes(b"SQLite format 3\x00" + b"\x00" * 512)
    with pytest.raises(ValueError, match="SQLite database files are unsupported"):
        DuckDBResultsWriter(str(path), export_parquet=False)
