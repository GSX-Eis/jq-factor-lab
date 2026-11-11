#!/usr/bin/env python3
"""Import frozen CNINFO equity-incentive plan draft events."""

import argparse
from pathlib import Path

import duckdb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.loaders._transactions import import_transaction

DEFAULT_DB = "./data/duckdb/factor_database.duckdb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    source = Path(args.source) / "events.csv"
    if not source.is_file():
        raise SystemExit(f"missing {source}")
    pattern = str(source).replace("'", "''")
    with duckdb.connect(args.db) as connection, import_transaction(connection):
        connection.execute("""
            CREATE TABLE IF NOT EXISTS regime_cninfo_equity_incentive_plan_draft_events_v1 (
              announcement_id VARCHAR PRIMARY KEY, issuer_stock_code VARCHAR,
              announcement_timestamp TIMESTAMP, title VARCHAR, initial_plan_flag BOOLEAN,
              board_approval_verified BOOLEAN, board_approval_excerpt_sha256 VARCHAR,
              official_url VARCHAR, content_path VARCHAR, content_sha256 VARCHAR
            )
        """)
        connection.execute(f"""
            CREATE OR REPLACE TEMP TABLE _cninfo_incentive_stage AS
            SELECT announcement_id,issuer_stock_code,
              CAST(to_timestamp(TRY_CAST(announcement_timestamp_ms AS DOUBLE)/1000) AS TIMESTAMP)
                AS announcement_timestamp,
              title,TRY_CAST(initial_plan_flag AS BOOLEAN) initial_plan_flag,
              TRY_CAST(board_approval_verified AS BOOLEAN) board_approval_verified,
              board_approval_excerpt_sha256,official_url,content_path,content_sha256
            FROM read_csv_auto('{pattern}',header=true,all_varchar=true)
        """)
        bad = connection.execute("""
            SELECT count(*) FROM _cninfo_incentive_stage
            WHERE announcement_id IS NULL OR announcement_timestamp IS NULL
              OR issuer_stock_code NOT LIKE '______.%' OR NOT initial_plan_flag
              OR NOT board_approval_verified OR length(board_approval_excerpt_sha256)!=64
              OR official_url NOT LIKE 'https://static.cninfo.com.cn/%'
              OR length(content_sha256)!=64
        """).fetchone()[0]
        duplicates = connection.execute("""
            SELECT count(*) FROM (
              SELECT announcement_id FROM _cninfo_incentive_stage
              GROUP BY 1 HAVING count(*)!=1
            )
        """).fetchone()[0]
        if bad or duplicates:
            raise ValueError(f"invalid CNINFO incentive rows: bad={bad} duplicate_ids={duplicates}")
        before = connection.execute(
            "SELECT count(*) FROM regime_cninfo_equity_incentive_plan_draft_events_v1"
        ).fetchone()[0]
        staged = connection.execute("SELECT count(*) FROM _cninfo_incentive_stage").fetchone()[0]
        connection.execute("""
            INSERT OR REPLACE INTO regime_cninfo_equity_incentive_plan_draft_events_v1
            SELECT * FROM _cninfo_incentive_stage
        """)
        after = connection.execute(
            "SELECT count(*) FROM regime_cninfo_equity_incentive_plan_draft_events_v1"
        ).fetchone()[0]
        print(
            "[OK] regime_cninfo_equity_incentive_plan_draft_events_v1: "
            f"staged={staged} before={before} after={after}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
