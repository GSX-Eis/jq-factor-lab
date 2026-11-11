#!/usr/bin/env python3
"""import_dividend_events_duckdb.py - 除权除息事件 CSV -> DuckDB (append/upsert).

新建 `dividend_events` 附加表 (不触碰任何既有表 -> 隔离安全)。
PK (stock_code, report_date, a_ex_dividend_date)，`INSERT OR REPLACE` 幂等。
列对齐 finance.STK_XR_XD 真实 schema (a_xr_date 已重命名为 a_ex_dividend_date)。
下游因子层用它计算 TTM 股息率 / 连续分红年数 / 分红稳定性。

Usage:
    python data_pipeline/loaders/import_dividend_events_duckdb.py \
        --csv ./data/import/dividend_lowvol/dividend/dividend_events.csv \
        --db  ./data/duckdb/factor_database.duckdb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.loaders._transactions import import_transaction

DEFAULT_DB = "./data/duckdb/factor_database.duckdb"

DDL = """
CREATE TABLE IF NOT EXISTS dividend_events (
    stock_code              VARCHAR,
    company_name            VARCHAR,
    report_date             DATE,
    bonus_type              VARCHAR,
    plan_progress           VARCHAR,
    board_plan_pub_date     DATE,
    board_plan_bonusnote    VARCHAR,
    shareholders_plan_pub_date DATE,
    shareholders_plan_bonusnote VARCHAR,
    a_registration_date     DATE,
    a_ex_dividend_date      DATE,
    a_bonus_date            DATE,
    bonus_ratio_rmb         DOUBLE,
    dividend_ratio          DOUBLE,
    transfer_ratio          DOUBLE,
    bonus_amount_rmb        DOUBLE,
    implementation_pub_date DATE,
    implementation_bonusnote VARCHAR,
    bonus_cancel_pub_date   DATE,
    PRIMARY KEY (stock_code, report_date, a_ex_dividend_date)
);
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="导入除权除息事件 (append/upsert)")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="单个 dividend_events CSV 路径")
    source.add_argument("--source", help="包含 dividend_events_*.csv 分区的目录")
    ap.add_argument("--db", default=DEFAULT_DB, help="目标 DuckDB 路径")
    args = ap.parse_args()

    if args.csv:
        csv = Path(args.csv)
        if not csv.is_file():
            raise SystemExit(f"[ERROR] CSV 不存在: {csv}")
        csv_sql = str(csv).replace("'", "''")
    else:
        source_dir = Path(args.source)
        files = sorted(source_dir.glob("dividend_events_*.csv"))
        if not files:
            raise SystemExit(f"[ERROR] 未找到分区: {source_dir}/dividend_events_*.csv")
        csv_sql = str(source_dir / "dividend_events_*.csv").replace("'", "''")

    with duckdb.connect(args.db) as con, import_transaction(con):
        con.execute(DDL)
        con.execute("ALTER TABLE dividend_events ADD COLUMN IF NOT EXISTS board_plan_pub_date DATE")
        con.execute("ALTER TABLE dividend_events ADD COLUMN IF NOT EXISTS board_plan_bonusnote VARCHAR")
        con.execute("ALTER TABLE dividend_events ADD COLUMN IF NOT EXISTS shareholders_plan_pub_date DATE")
        con.execute("ALTER TABLE dividend_events ADD COLUMN IF NOT EXISTS shareholders_plan_bonusnote VARCHAR")
        con.execute("ALTER TABLE dividend_events ADD COLUMN IF NOT EXISTS implementation_bonusnote VARCHAR")
        con.execute("ALTER TABLE dividend_events ADD COLUMN IF NOT EXISTS bonus_cancel_pub_date DATE")
        before = con.execute("SELECT COUNT(*) FROM dividend_events").fetchone()[0]

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE _div_raw AS
            SELECT *
            FROM read_csv_auto('{csv_sql}', header=True, union_by_name=True)
            """
        )
        con.execute("ALTER TABLE _div_raw ADD COLUMN IF NOT EXISTS board_plan_pub_date DATE")
        con.execute("ALTER TABLE _div_raw ADD COLUMN IF NOT EXISTS board_plan_bonusnote VARCHAR")
        con.execute("ALTER TABLE _div_raw ADD COLUMN IF NOT EXISTS shareholders_plan_pub_date DATE")
        con.execute("ALTER TABLE _div_raw ADD COLUMN IF NOT EXISTS shareholders_plan_bonusnote VARCHAR")
        con.execute("ALTER TABLE _div_raw ADD COLUMN IF NOT EXISTS implementation_bonusnote VARCHAR")
        con.execute("ALTER TABLE _div_raw ADD COLUMN IF NOT EXISTS bonus_cancel_pub_date DATE")
        con.execute("ALTER TABLE _div_raw ADD COLUMN IF NOT EXISTS source_record_id BIGINT")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _div_stage AS
            SELECT
                TRY_CAST(source_record_id AS BIGINT)              AS source_record_id,
                CAST(stock_code AS VARCHAR)                   AS stock_code,
                CAST(company_name AS VARCHAR)                 AS company_name,
                TRY_CAST(report_date AS DATE)                 AS report_date,
                CAST(bonus_type AS VARCHAR)                   AS bonus_type,
                CAST(plan_progress AS VARCHAR)                AS plan_progress,
                TRY_CAST(board_plan_pub_date AS DATE)         AS board_plan_pub_date,
                CAST(board_plan_bonusnote AS VARCHAR)         AS board_plan_bonusnote,
                TRY_CAST(shareholders_plan_pub_date AS DATE)  AS shareholders_plan_pub_date,
                CAST(shareholders_plan_bonusnote AS VARCHAR)  AS shareholders_plan_bonusnote,
                TRY_CAST(a_registration_date AS DATE)         AS a_registration_date,
                TRY_CAST(a_ex_dividend_date AS DATE)          AS a_ex_dividend_date,
                TRY_CAST(a_bonus_date AS DATE)                AS a_bonus_date,
                TRY_CAST(bonus_ratio_rmb AS DOUBLE)           AS bonus_ratio_rmb,
                TRY_CAST(dividend_ratio AS DOUBLE)            AS dividend_ratio,
                TRY_CAST(transfer_ratio AS DOUBLE)            AS transfer_ratio,
                TRY_CAST(bonus_amount_rmb AS DOUBLE)          AS bonus_amount_rmb,
                TRY_CAST(implementation_pub_date AS DATE)     AS implementation_pub_date
                , CAST(implementation_bonusnote AS VARCHAR)   AS implementation_bonusnote
                , TRY_CAST(bonus_cancel_pub_date AS DATE)     AS bonus_cancel_pub_date
            FROM _div_raw
            """
        )
        # 同 PK 去重 (保留最新实施公告日的一条)
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE _div_dedup AS
            SELECT stock_code, company_name, report_date, bonus_type, plan_progress,
                   board_plan_pub_date, board_plan_bonusnote,
                   shareholders_plan_pub_date, shareholders_plan_bonusnote,
                   a_registration_date, a_ex_dividend_date, a_bonus_date,
                   bonus_ratio_rmb, dividend_ratio, transfer_ratio,
                   bonus_amount_rmb, implementation_pub_date, implementation_bonusnote,
                   bonus_cancel_pub_date
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY stock_code, report_date, a_ex_dividend_date
                    ORDER BY implementation_pub_date DESC NULLS LAST,
                             source_record_id DESC NULLS LAST
                ) AS rn
                FROM _div_stage
            )
            WHERE rn = 1
              AND stock_code IS NOT NULL
              AND report_date IS NOT NULL
              AND a_ex_dividend_date IS NOT NULL
            """
        )
        staged = con.execute("SELECT COUNT(*) FROM _div_dedup").fetchone()[0]
        con.execute(
            """
            INSERT OR REPLACE INTO dividend_events (
                stock_code, company_name, report_date, bonus_type, plan_progress,
                board_plan_pub_date, board_plan_bonusnote,
                shareholders_plan_pub_date, shareholders_plan_bonusnote,
                a_registration_date, a_ex_dividend_date, a_bonus_date,
                bonus_ratio_rmb, dividend_ratio, transfer_ratio,
                bonus_amount_rmb, implementation_pub_date, implementation_bonusnote,
                bonus_cancel_pub_date
            )
            SELECT
                stock_code, company_name, report_date, bonus_type, plan_progress,
                board_plan_pub_date, board_plan_bonusnote,
                shareholders_plan_pub_date, shareholders_plan_bonusnote,
                a_registration_date, a_ex_dividend_date, a_bonus_date,
                bonus_ratio_rmb, dividend_ratio, transfer_ratio,
                bonus_amount_rmb, implementation_pub_date, implementation_bonusnote,
                bonus_cancel_pub_date
            FROM _div_dedup
            """
        )
        after = con.execute("SELECT COUNT(*) FROM dividend_events").fetchone()[0]
        stocks = con.execute(
            "SELECT COUNT(DISTINCT stock_code) FROM dividend_events"
        ).fetchone()[0]
        print(
            f"[OK] dividend_events: staged={staged} before={before} "
            f"after={after} (+{after - before}) distinct_stocks={stocks}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
