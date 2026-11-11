#!/usr/bin/env python3
"""Upsert validated supplemental industry partitions without replacing data with nulls."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.loaders._transactions import import_transaction


DEFAULT_DB = "./data/duckdb/factor_database.duckdb"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    source = Path(args.source)
    files = sorted(source.glob("industry_membership_*.csv*"))
    if not files:
        raise SystemExit(f"[ERROR] 未找到分区: {source}/industry_membership_*.csv*")
    source_sql = str(source / "industry_membership_*.csv*").replace("'", "''")

    with duckdb.connect(args.db) as conn, import_transaction(conn):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS industry_membership (
                date DATE NOT NULL,
                stock_code VARCHAR NOT NULL,
                sw_l1_code VARCHAR,
                sw_l1_name VARCHAR,
                sw_l2_code VARCHAR,
                sw_l2_name VARCHAR,
                zjw_code VARCHAR,
                zjw_name VARCHAR,
                PRIMARY KEY (date, stock_code)
            )
            """
        )
        before = conn.execute("SELECT COUNT(*) FROM industry_membership").fetchone()[0]
        conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE _industry_stage AS
            SELECT DISTINCT
                TRY_CAST(date AS DATE) AS date,
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(sw_l1_code AS VARCHAR) AS sw_l1_code,
                CAST(sw_l1_name AS VARCHAR) AS sw_l1_name,
                CAST(sw_l2_code AS VARCHAR) AS sw_l2_code,
                CAST(sw_l2_name AS VARCHAR) AS sw_l2_name,
                CAST(zjw_code AS VARCHAR) AS zjw_code,
                CAST(zjw_name AS VARCHAR) AS zjw_name
            FROM read_csv_auto('{source_sql}', header=true, union_by_name=true)
            WHERE TRY_CAST(date AS DATE) IS NOT NULL
              AND stock_code IS NOT NULL
              AND COALESCE(
                    CAST(sw_l1_code AS VARCHAR),
                    CAST(sw_l2_code AS VARCHAR),
                    CAST(zjw_code AS VARCHAR)
                  ) IS NOT NULL
            """
        )
        staged = conn.execute("SELECT COUNT(*) FROM _industry_stage").fetchone()[0]
        conn.execute(
            """
            INSERT OR REPLACE INTO industry_membership
            SELECT date, stock_code, sw_l1_code, sw_l1_name,
                   sw_l2_code, sw_l2_name, zjw_code, zjw_name
            FROM _industry_stage
            """
        )
        after = conn.execute("SELECT COUNT(*) FROM industry_membership").fetchone()[0]
        print(
            f"[OK] industry_membership supplemental: staged={staged} "
            f"before={before} after={after} (+{after - before})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
