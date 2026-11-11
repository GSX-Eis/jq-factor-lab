#!/usr/bin/env python3
"""Import frozen CSRC listed-issuer penalty events into an isolated table."""

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
    source = Path(args.source) / "issuer_penalty_events.csv"
    if not source.is_file():
        raise SystemExit(f"missing {source}")
    pattern = str(source).replace("'", "''")
    with duckdb.connect(args.db) as connection, import_transaction(connection):
        connection.execute("""
            CREATE TABLE IF NOT EXISTS regime_csrc_issuer_penalty_events_v1 (
                publication_timestamp TIMESTAMP, publication_date DATE,
                decision_document_date DATE, decision_number VARCHAR,
                issuer_stock_code VARCHAR, issuer_alias VARCHAR,
                subject_role VARCHAR, final_decision_category VARCHAR,
                title VARCHAR, official_url VARCHAR, content_sha256 VARCHAR,
                PRIMARY KEY (publication_timestamp, issuer_stock_code)
            )
        """)
        connection.execute(f"""
            CREATE OR REPLACE TEMP TABLE _csrc_penalty_stage AS
            SELECT TRY_CAST(publication_timestamp AS TIMESTAMP) publication_timestamp,
              TRY_CAST(publication_date AS DATE) publication_date,
              TRY_CAST(decision_document_date AS DATE) decision_document_date,
              CAST(decision_number AS VARCHAR) decision_number,
              CAST(issuer_stock_code AS VARCHAR) issuer_stock_code,
              CAST(issuer_alias AS VARCHAR) issuer_alias,
              CAST(subject_role AS VARCHAR) subject_role,
              CAST(final_decision_category AS VARCHAR) final_decision_category,
              CAST(title AS VARCHAR) title,CAST(official_url AS VARCHAR) official_url,
              CAST(content_sha256 AS VARCHAR) content_sha256
            FROM read_csv_auto('{pattern}', header=true, all_varchar=true)
        """)
        bad = connection.execute("""
            SELECT count(*) FROM _csrc_penalty_stage
            WHERE publication_timestamp IS NULL OR publication_date IS NULL
              OR issuer_stock_code NOT LIKE '______.%' OR subject_role!='listed_issuer'
              OR length(content_sha256)!=64 OR official_url NOT LIKE 'https://www.csrc.gov.cn/%'
        """).fetchone()[0]
        duplicates = connection.execute("""
            SELECT count(*) FROM (
              SELECT publication_timestamp,issuer_stock_code FROM _csrc_penalty_stage
              GROUP BY 1,2 HAVING count(*)!=1
            )
        """).fetchone()[0]
        if bad or duplicates:
            raise ValueError(f"invalid CSRC event rows: bad={bad} duplicate_keys={duplicates}")
        before = connection.execute("SELECT count(*) FROM regime_csrc_issuer_penalty_events_v1").fetchone()[0]
        staged = connection.execute("SELECT count(*) FROM _csrc_penalty_stage").fetchone()[0]
        connection.execute("INSERT OR REPLACE INTO regime_csrc_issuer_penalty_events_v1 SELECT * FROM _csrc_penalty_stage")
        after = connection.execute("SELECT count(*) FROM regime_csrc_issuer_penalty_events_v1").fetchone()[0]
        print(f"[OK] regime_csrc_issuer_penalty_events_v1: staged={staged} before={before} after={after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
