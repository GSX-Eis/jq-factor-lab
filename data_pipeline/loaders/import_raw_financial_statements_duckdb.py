#!/usr/bin/env python3
"""Import complete raw JoinQuant financial-statement partitions into DuckDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.data_revision import advance_revision


STATEMENTS = ("income", "balance", "cashflow")
TEXT_COLUMNS = {"company_name", "code", "a_code", "b_code", "h_code", "source"}
DATE_COLUMNS = {"pub_date", "start_date", "end_date", "report_date"}
INTEGER_COLUMNS = {"id", "company_id", "report_type", "source_id"}


def _quoted(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _type(column: str) -> str:
    if column in TEXT_COLUMNS:
        return "VARCHAR"
    if column in DATE_COLUMNS:
        return "DATE"
    if column in INTEGER_COLUMNS:
        return "BIGINT"
    return "DOUBLE"


def import_statement(
    conn: duckdb.DuckDBPyConnection,
    statement: str,
    source: Path,
    schema_contract: Path,
) -> dict[str, object]:
    if statement not in STATEMENTS:
        raise ValueError("Unsupported statement: " + statement)
    columns = list(json.loads(schema_contract.read_text())[statement]["columns"])
    pattern = source / statement / f"raw_{statement}_statement_????.csv"
    matches = sorted((source / statement).glob(f"raw_{statement}_statement_????.csv"))
    if not matches:
        raise FileNotFoundError("no raw financial partitions: %s" % pattern)
    table = f"raw_{statement}_statement_events"
    definitions = [f"{_quoted(column)} {_type(column)}" for column in columns]
    definitions.append('PRIMARY KEY ("id")')
    source_sql = str(pattern).replace("'", "''")
    casts = ", ".join(
        f"try_cast({_quoted(column)} as {_type(column)}) as {_quoted(column)}"
        for column in columns
    )
    conn.execute("begin")
    try:
        conn.execute(f"create table if not exists {_quoted(table)} ({', '.join(definitions)})")
        actual_schema = {
            row[0]: row[1]
            for row in conn.execute(f"describe {_quoted(table)}").fetchall()
        }
        expected_schema = {column: _type(column) for column in columns}
        if actual_schema != expected_schema:
            raise ValueError(
                f"Schema mismatch for {table}: expected {expected_schema}, found {actual_schema}"
            )
        target_columns = ", ".join(_quoted(column) for column in columns)
        changed = conn.execute(
            f"""insert or replace into {_quoted(table)} ({target_columns})
                select {casts}
                from read_csv('{source_sql}', header=true, all_varchar=true,
                    union_by_name=true, strict_mode=false, null_padding=true,
                    ignore_errors=false, parallel=false, max_line_size=10000000)
                where try_cast("id" as bigint) is not null
                  and try_cast("pub_date" as date) is not null
                  and "code" is not null"""
        ).fetchone()[0]
        advance_revision(conn)
        conn.execute(f"create index if not exists idx_raw_{statement}_pub_date on {_quoted(table)}(pub_date)")
        conn.execute(f"create index if not exists idx_raw_{statement}_code on {_quoted(table)}(code, end_date)")
        rows, stocks, date_min, date_max = conn.execute(
            f"select count(*), count(distinct code), min(pub_date), max(pub_date) from {_quoted(table)}"
        ).fetchone()
        conn.execute("commit")
    except BaseException:
        conn.execute("rollback")
        raise
    return {
        "statement": statement,
        "table": table,
        "files": len(matches),
        "imported_rows": changed,
        "table_rows": rows,
        "stocks": stocks,
        "pub_date_min": str(date_min),
        "pub_date_max": str(date_max),
        "columns": len(columns),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statement", choices=(*STATEMENTS, "all"), default="all")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--schema-contract", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    statements = STATEMENTS if args.statement == "all" else (args.statement,)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(args.db)) as conn:
        conn.execute("set threads=4")
        conn.execute("set memory_limit='8GB'")
        results = [import_statement(conn, statement, args.source, args.schema_contract) for statement in statements]
    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
