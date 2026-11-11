"""Validate fixed-reference daily factors and stage a versioned intraday factor refresh."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import duckdb
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.data_revision import advance_revision

from utils.input_store import attach_input_parent

SOURCE = Path(
    "./data/import/all_a_5m_backfill_2016_2026/factor_refresh"
)


def verify_bar_custody(conn, parent, plan):
    """Check actual frozen bar counts in every planned stock/month; old files are not custody."""
    start, end = (
        dt.date.fromisoformat(plan["start"]),
        dt.date.fromisoformat(plan["end"]),
    )
    if start > end or not plan.get("partitions"):
        raise ValueError("Invalid or empty bar custody plan")
    rows = []
    for partition, entries in plan["partitions"].items():
        month = dt.date.fromisoformat(partition + "-01")
        for entry in entries:
            first, last = (
                dt.date.fromisoformat(entry["start"]),
                dt.date.fromisoformat(entry["end"]),
            )
            expected = entry["expected_rows"]
            if (
                not entry["stock_code"]
                or not isinstance(expected, int)
                or expected <= 0
                or not start <= first <= last <= end
                or first.replace(day=1) != month
                or last.replace(day=1) != month
            ):
                raise ValueError("Invalid stock/month bar custody entry")
            rows.append((partition, month, entry["stock_code"], first, last, expected))
    if not rows or len({(r[0], r[2]) for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate stock/month bar custody entries")
    conn.execute(
        "CREATE TEMP TABLE refresh_bar_plan(partition VARCHAR,month DATE,stock_code VARCHAR,start_date DATE,end_date DATE,expected_rows BIGINT)"
    )
    conn.executemany("INSERT INTO refresh_bar_plan VALUES (?,?,?,?,?,?)", rows)
    missing = conn.execute(
        f"""SELECT e.partition,e.stock_code,e.expected_rows,count(p.date) AS actual_rows
        FROM refresh_bar_plan e LEFT JOIN
        (SELECT stock_code,date FROM {parent} WHERE date>=? AND date<CAST(? AS DATE)+INTERVAL 1 DAY) p ON p.stock_code=e.stock_code
        AND date_trunc('month',p.date)=e.month AND p.date>=e.start_date AND p.date<e.end_date+INTERVAL 1 DAY
        GROUP BY e.partition,e.stock_code,e.expected_rows HAVING count(p.date)<>e.expected_rows""",
        [start, end],
    ).fetchall()
    if missing:
        raise ValueError(
            f"Frozen bar custody is incomplete: {len(missing)} stock/month entries; sample={missing[:5]}"
        )
    return {
        "ok": True,
        "stock_month_entries": len(rows),
        "expected_rows": sum(r[5] for r in rows),
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def stage_refresh(db, source, plan_path):
    parents = json.loads(os.environ.get("QUANT_INPUT_PARENT_BINDINGS", "{}"))
    if (
        os.environ.get("QUANT_INPUT_BATCH_STAGING") != "1"
        or not parents
        or os.environ.get("QUANT_INPUT_BATCH_REPLACE") != "0"
    ):
        raise ValueError(
            "Factor refresh requires the harness versioned input staging publisher"
        )
    plan = json.loads(Path(plan_path).read_text())
    start, end = (
        dt.date.fromisoformat(plan["start"]),
        dt.date.fromisoformat(plan["end"]),
    )
    files = []
    source_hashes = {}
    for year in range(start.year, end.year + 1):
        directory = Path(source) / str(year)
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["reference_date"] != plan["end"]
            or "factors.csv" not in manifest["sha256"]
        ):
            raise ValueError(
                "Factor partition has a different adjustment reference date"
            )
        for name, expected in manifest["sha256"].items():
            file = directory / name
            if file.resolve().parent != directory.resolve():
                raise ValueError(
                    "Factor manifest references a file outside its partition"
                )
            with file.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError(f"Factor partition hash mismatch: {file}")
        files.append(str(directory / "factors.csv"))
        source_hashes[str(year)] = manifest["sha256"]["factors.csv"]
    with duckdb.connect(str(db)) as conn:
        conn.execute("SET threads=2")
        conn.execute("SET memory_limit='3GB'")
        parent = attach_input_parent(conn, "price_data_5m", bindings=parents)
        custody = verify_bar_custody(conn, parent, plan)
        conn.execute(
            "CREATE TEMP TABLE fresh AS SELECT stock_code,CAST(date AS DATE) AS trade_date,CAST(factor AS DOUBLE) AS factor FROM read_csv(?,header=true,all_varchar=true)",
            [files],
        )
        if not conn.execute("SELECT 1 FROM fresh LIMIT 1").fetchone():
            raise ValueError("Empty daily factor refresh")
        if conn.execute(
            "SELECT 1 FROM fresh WHERE stock_code IS NULL OR trade_date IS NULL OR trade_date<? OR trade_date>? OR (factor IS NOT NULL AND (NOT isfinite(factor) OR factor<=0)) LIMIT 1",
            [start, end],
        ).fetchone():
            raise ValueError("Invalid daily factor key, range or value")
        if conn.execute(
            "SELECT 1 FROM fresh GROUP BY stock_code,trade_date HAVING count(*)>1 LIMIT 1"
        ).fetchone():
            raise ValueError("Duplicate daily factor keys")
        conn.execute("BEGIN")
        try:
            conn.execute(
                "CREATE OR REPLACE TABLE main.price_data_5m AS SELECT * FROM fresh"
            )
            conn.execute("CREATE SCHEMA IF NOT EXISTS meta")
            conn.execute(
                "CREATE OR REPLACE TABLE meta.factor_refresh(start_date DATE,end_date DATE,reference_date DATE,parent_version VARCHAR,custody_json VARCHAR)"
            )
            conn.execute(
                "INSERT INTO meta.factor_refresh VALUES (?,?,?,?,?)",
                [
                    start,
                    end,
                    end,
                    parents["inputs"]["price_data_5m"]["version"],
                    json.dumps(custody),
                ],
            )
            advance_revision(conn)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return {
        "ok": True,
        "daily_partitions": len(files),
        "source_hashes": source_hashes,
        "custody": custody,
        "parent_ref": parents["inputs"]["price_data_5m"],
        "staging_only": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--plan", type=Path, default=SOURCE.parent / "plan.json")
    args = parser.parse_args()
    print(json.dumps(stage_refresh(args.db, args.source, args.plan)))
