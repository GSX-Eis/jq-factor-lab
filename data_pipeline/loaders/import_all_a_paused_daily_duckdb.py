#!/usr/bin/env python3
"""Import immutable all-A daily suspension flags into an isolated DuckDB table."""

import argparse
from pathlib import Path

import duckdb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.loaders._transactions import import_transaction

DEFAULT_DB = "./data/duckdb/factor_database.duckdb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    source = Path(args.source)
    if not list(source.glob("paused_daily_*.csv")):
        raise SystemExit(f"no paused_daily_*.csv under {source}")
    pattern = str(source / "paused_daily_*.csv").replace("'", "''")
    with duckdb.connect(args.db) as connection, import_transaction(connection):
        connection.execute("""
            CREATE TABLE IF NOT EXISTS regime_all_a_paused_daily_v1 (
                run_id VARCHAR, partition_year INTEGER, date DATE, stock_code VARCHAR,
                paused UTINYINT, source_snapshot_date DATE,
                PRIMARY KEY (date, stock_code)
            )
        """)
        before = connection.execute("SELECT COUNT(*) FROM regime_all_a_paused_daily_v1").fetchone()[0]
        connection.execute(f"""
            CREATE OR REPLACE TEMP TABLE _paused_stage AS
            SELECT CAST(run_id AS VARCHAR) run_id, TRY_CAST(partition_year AS INTEGER) partition_year,
                   TRY_CAST(date AS DATE) date, CAST(stock_code AS VARCHAR) stock_code,
                   TRY_CAST(paused AS UTINYINT) paused,
                   TRY_CAST(source_snapshot_date AS DATE) source_snapshot_date
            FROM read_csv_auto('{pattern}', header=true, union_by_name=true)
        """)
        bad = connection.execute("""
            SELECT COUNT(*) FROM _paused_stage
            WHERE date IS NULL OR stock_code IS NULL OR paused NOT IN (0, 1)
               OR source_snapshot_date IS NULL
        """).fetchone()[0]
        duplicates = connection.execute("""
            SELECT COUNT(*) FROM (
                SELECT date, stock_code FROM _paused_stage GROUP BY 1, 2 HAVING COUNT(*) != 1
            )
        """).fetchone()[0]
        if bad or duplicates:
            raise ValueError(f"invalid paused staging rows: bad={bad} duplicate_keys={duplicates}")
        staged = connection.execute("SELECT COUNT(*) FROM _paused_stage").fetchone()[0]
        connection.execute("INSERT OR REPLACE INTO regime_all_a_paused_daily_v1 SELECT * FROM _paused_stage")
        after = connection.execute("SELECT COUNT(*) FROM regime_all_a_paused_daily_v1").fetchone()[0]
        print(f"[OK] regime_all_a_paused_daily_v1: staged={staged} before={before} after={after} (+{after - before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
