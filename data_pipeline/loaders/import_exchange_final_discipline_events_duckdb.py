#!/usr/bin/env python3
"""Import frozen exchange listed-issuer discipline events into an isolated table."""

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
            CREATE TABLE IF NOT EXISTS regime_exchange_final_discipline_events_v1 (
              exchange VARCHAR,decision_id VARCHAR,issuer_stock_code VARCHAR,
              issuer_role VARCHAR,sanction_type VARCHAR,publication_timestamp TIMESTAMP,
              publication_precision VARCHAR,title VARCHAR,official_url VARCHAR,
              content_path VARCHAR,content_sha256 VARCHAR,
              PRIMARY KEY (exchange,decision_id,issuer_stock_code)
            )
        """)
        connection.execute(f"""
            CREATE OR REPLACE TEMP TABLE _exchange_discipline_stage AS
            SELECT exchange,decision_id,issuer_stock_code,issuer_role,sanction_type,
              TRY_CAST(publication_timestamp AS TIMESTAMP) publication_timestamp,
              publication_precision,title,official_url,content_path,content_sha256
            FROM read_csv_auto('{pattern}',header=true,all_varchar=true)
        """)
        bad = connection.execute("""
            SELECT count(*) FROM _exchange_discipline_stage
            WHERE exchange NOT IN ('SSE','SZSE') OR issuer_role!='listed_issuer'
              OR sanction_type NOT IN ('公开谴责','通报批评') OR publication_timestamp IS NULL
              OR issuer_stock_code NOT LIKE '______.%' OR length(content_sha256)!=64
        """).fetchone()[0]
        duplicates = connection.execute("""
            SELECT count(*) FROM (
              SELECT exchange,decision_id,issuer_stock_code FROM _exchange_discipline_stage
              GROUP BY 1,2,3 HAVING count(*)!=1
            )
        """).fetchone()[0]
        if bad or duplicates:
            raise ValueError(f"invalid exchange event rows: bad={bad} duplicate_keys={duplicates}")
        before = connection.execute("SELECT count(*) FROM regime_exchange_final_discipline_events_v1").fetchone()[0]
        staged = connection.execute("SELECT count(*) FROM _exchange_discipline_stage").fetchone()[0]
        connection.execute("INSERT OR REPLACE INTO regime_exchange_final_discipline_events_v1 SELECT * FROM _exchange_discipline_stage")
        after = connection.execute("SELECT count(*) FROM regime_exchange_final_discipline_events_v1").fetchone()[0]
        print(f"[OK] regime_exchange_final_discipline_events_v1: staged={staged} before={before} after={after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
