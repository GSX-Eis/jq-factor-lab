#!/usr/bin/env python3
"""
portfolio_data_to_duckdb.py - Import portfolio-optimization datasets to DuckDB

Imports the 25 dataset specifications registered in DATASETS, including security
master, daily factors and prices, ownership, financial and corporate events.
Each specification defines its CSV pattern, schema, import SQL and summary.
Tables use the key or whole-row deduplication contract defined in that SQL.
Each import_dataset call commits independently; --dataset all is not one
transaction. DEFAULT_SOURCES supplies the built-in source paths; a dataset
without a default source must be invoked on its own with --source.

Usage:
    python -m data_pipeline.loaders.portfolio_data_to_duckdb --dataset valuation \
        --source ./data/import/valuation/ \
        --db ./data/duckdb/factor_database.duckdb

    # Import everything with default source roots:
    python -m data_pipeline.loaders.portfolio_data_to_duckdb --dataset all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.loaders._transactions import atomic_import

DEFAULT_DB_PATH = Path("./data/duckdb/factor_database.duckdb")
DEFAULT_SOURCE_ROOT = Path("./data/import")

VALUATION_FIELDS = [
    "capitalization",
    "circulating_cap",
    "market_cap",
    "circulating_market_cap",
    "turnover_ratio",
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "pcf_ratio",
]

STYLE_FACTORS = [
    "size",
    "beta",
    "momentum",
    "residual_volatility",
    "non_linear_size",
    "book_to_price_ratio",
    "liquidity",
    "earnings_yield",
    "growth",
    "leverage",
]

MONEY_FLOW_FIELDS = [
    "change_pct",
    "net_amount_main",
    "net_pct_main",
    "net_amount_xl",
    "net_pct_xl",
    "net_amount_l",
    "net_pct_l",
    "net_amount_m",
    "net_pct_m",
    "net_amount_s",
    "net_pct_s",
]

NORTHBOUND_HOLD_FIELDS = [
    "share_number",
    "share_ratio",
    "market_cap",
    "change",
    "change_ratio",
]

EARNINGS_FORECAST_FIELDS = [
    "forecast_type",
    "forecast_abstract",
    "forecast_content",
    "profit_min",
    "profit_max",
    "profit_last",
    "change_min",
    "change_max",
]

SHAREHOLDER_STRUCTURE_FIELDS = [
    "pub_date",
    "report_type",
    "holder_num",
    "holder_avg_pct",
    "holder_avg_number",
    "top10_share_ratio",
    "floating_top10_share_ratio",
    "institution_top10_share_ratio",
]

CAPITAL_CHANGE_FIELDS = [
    "share_total",
    "share_non_trade",
    "share_limited",
    "share_trade_total",
    "share_rmb",
    "share_b",
    "share_h",
    "share_management",
]

INDUSTRY_COLUMNS = [
    "sw_l1_code",
    "sw_l1_name",
    "sw_l2_code",
    "sw_l2_name",
    "zjw_code",
    "zjw_name",
]

MARGIN_TRADING_FIELDS = [
    "fin_value",
    "fin_buy_value",
    "fin_refund_value",
    "sec_value",
    "sec_sell_value",
    "sec_refund_value",
    "fin_sec_value",
]

# Event tables without database primary keys use their individual SQL contracts:
# billboard/frozen/share_change compare whole rows; pledge compares six natural-key columns.
ROBUST_CSV_OPTS = (
    "header=true, delim=',', quote='\"', escape='\"', strict_mode=false, "
    "null_padding=true, parallel=false, max_line_size=10000000, ignore_errors=false"
)


class LoaderError(Exception):
    """Raised when a dataset import cannot proceed."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    table: str
    csv_glob: str
    create_sql: str
    insert_sql: str
    index_sql: tuple[str, ...]
    summary_sql: str
    migration_sql: tuple[str, ...] = ()


def _doubles(columns: list[str]) -> str:
    return ",\n                ".join(f"{c} DOUBLE" for c in columns)


def _casts(columns: list[str]) -> str:
    return ",\n                ".join(f"CAST({c} AS DOUBLE) AS {c}" for c in columns)


DATASETS: dict[str, DatasetSpec] = {
    "security_master": DatasetSpec(
        name="security_master",
        table="security_master",
        csv_glob="security_master.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS security_master (
                stock_code VARCHAR PRIMARY KEY,
                display_name VARCHAR,
                name VARCHAR,
                start_date DATE,
                end_date DATE,
                security_type VARCHAR
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO security_master (
                "stock_code", "display_name", "name", "start_date", "end_date", "security_type"
            )
            SELECT
                CAST(stock_code AS VARCHAR),
                CAST(display_name AS VARCHAR),
                CAST(name AS VARCHAR),
                TRY_CAST(start_date AS DATE),
                TRY_CAST(end_date AS DATE),
                CAST(security_type AS VARCHAR)
            FROM read_csv('{source}', header=true, auto_detect=true, ignore_errors=false)
            WHERE stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_security_master_start_date ON security_master(start_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(start_date), MAX(start_date), COUNT(DISTINCT stock_code)
            FROM security_master
        """,
    ),
    "historical_st": DatasetSpec(
        name="historical_st",
        table="historical_st_daily",
        csv_glob="historical_st_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS historical_st_daily (
                stock_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                is_st BOOLEAN,
                PRIMARY KEY (stock_code, date)
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO historical_st_daily (
                "stock_code", "date", "is_st"
            )
            SELECT
                CAST(stock_code AS VARCHAR),
                TRY_CAST(date AS DATE),
                TRY_CAST(is_st AS BOOLEAN)
            FROM read_csv('{source}', header=true, auto_detect=true, ignore_errors=false)
            WHERE stock_code IS NOT NULL AND TRY_CAST(date AS DATE) IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_historical_st_date ON historical_st_daily(date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM historical_st_daily
        """,
    ),
    "shareholder_top10": DatasetSpec(
        name="shareholder_top10",
        table="shareholder_top10_events",
        csv_glob="shareholder_top10_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS shareholder_top10_events (
                stock_code VARCHAR NOT NULL,
                pub_date DATE,
                end_date DATE,
                report_type INTEGER,
                holder_scope VARCHAR,
                holder_rank INTEGER,
                shareholder_name VARCHAR,
                shareholder_class_id INTEGER,
                share_number DOUBLE,
                share_ratio DOUBLE
            )
        """,
        insert_sql=f"""
            INSERT INTO shareholder_top10_events (
                "stock_code", "pub_date", "end_date", "report_type", "holder_scope",
                "holder_rank", "shareholder_name", "shareholder_class_id", "share_number",
                "share_ratio"
            )
            SELECT * FROM (
                SELECT DISTINCT
                    CAST(stock_code AS VARCHAR),
                    TRY_CAST(pub_date AS DATE),
                    TRY_CAST(end_date AS DATE),
                    TRY_CAST(report_type AS INTEGER),
                    CAST(holder_scope AS VARCHAR),
                    TRY_CAST(holder_rank AS INTEGER),
                    CAST(shareholder_name AS VARCHAR),
                    TRY_CAST(shareholder_class_id AS INTEGER),
                    TRY_CAST(share_number AS DOUBLE),
                    TRY_CAST(share_ratio AS DOUBLE)
                FROM read_csv('{{source}}', {ROBUST_CSV_OPTS})
                WHERE stock_code IS NOT NULL AND TRY_CAST(pub_date AS DATE) IS NOT NULL
                EXCEPT SELECT "stock_code", "pub_date", "end_date", "report_type", "holder_scope", "holder_rank", "shareholder_name", "shareholder_class_id", "share_number", "share_ratio" FROM shareholder_top10_events
            )
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_shareholder_top10_pub_date ON shareholder_top10_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_shareholder_top10_stock ON shareholder_top10_events(stock_code)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM shareholder_top10_events
        """,
    ),
    "valuation": DatasetSpec(
        name="valuation",
        table="valuation_daily",
        csv_glob="valuation_*.csv",
        create_sql=f"""
            CREATE TABLE IF NOT EXISTS valuation_daily (
                stock_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                {_doubles(VALUATION_FIELDS)},
                PRIMARY KEY (stock_code, date)
            )
        """,
        insert_sql=f"""
            INSERT OR REPLACE INTO valuation_daily (
                "stock_code", "date", "capitalization", "circulating_cap", "market_cap",
                "circulating_market_cap", "turnover_ratio", "pe_ratio", "pb_ratio", "ps_ratio",
                "pcf_ratio"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(date AS DATE) AS date,
                {_casts(VALUATION_FIELDS)}
            FROM read_csv('{{source}}', header=true, auto_detect=true, ignore_errors=false)
            WHERE date IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_valuation_date ON valuation_daily(date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM valuation_daily
        """,
    ),
    "style": DatasetSpec(
        name="style",
        table="style_factor_exposure",
        csv_glob="style_factors_*.csv",
        create_sql=f"""
            CREATE TABLE IF NOT EXISTS style_factor_exposure (
                stock_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                {_doubles(STYLE_FACTORS)},
                PRIMARY KEY (stock_code, date)
            )
        """,
        insert_sql=f"""
            INSERT OR REPLACE INTO style_factor_exposure (
                "stock_code", "date", "size", "beta", "momentum", "residual_volatility",
                "non_linear_size", "book_to_price_ratio", "liquidity", "earnings_yield",
                "growth", "leverage"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(date AS DATE) AS date,
                {_casts(STYLE_FACTORS)}
            FROM read_csv('{{source}}', header=true, auto_detect=true, ignore_errors=false)
            WHERE date IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_style_date ON style_factor_exposure(date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM style_factor_exposure
        """,
    ),
    "industry": DatasetSpec(
        name="industry",
        table="industry_membership",
        csv_glob="industry_membership.csv",
        create_sql="""
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
        """,
        insert_sql="""
            INSERT OR REPLACE INTO industry_membership (
                "date", "stock_code", "sw_l1_code", "sw_l1_name", "sw_l2_code", "sw_l2_name",
                "zjw_code", "zjw_name"
            )
            SELECT
                CAST(date AS DATE) AS date,
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(sw_l1_code AS VARCHAR) AS sw_l1_code,
                CAST(sw_l1_name AS VARCHAR) AS sw_l1_name,
                CAST(sw_l2_code AS VARCHAR) AS sw_l2_code,
                CAST(sw_l2_name AS VARCHAR) AS sw_l2_name,
                CAST(zjw_code AS VARCHAR) AS zjw_code,
                CAST(zjw_name AS VARCHAR) AS zjw_name
            FROM read_csv('{source}', header=true, auto_detect=true, ignore_errors=false,
                          types={'sw_l1_code': 'VARCHAR', 'sw_l2_code': 'VARCHAR', 'zjw_code': 'VARCHAR'})
            WHERE date IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_industry_stock ON industry_membership(stock_code)",
            "CREATE INDEX IF NOT EXISTS idx_industry_sw_l1 ON industry_membership(sw_l1_code)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM industry_membership
        """,
    ),
    "index_price": DatasetSpec(
        name="index_price",
        table="index_price_daily",
        csv_glob="index_price_daily.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS index_price_daily (
                index_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                index_name VARCHAR,
                open DOUBLE,
                close DOUBLE,
                high DOUBLE,
                low DOUBLE,
                volume DOUBLE,
                money DOUBLE,
                PRIMARY KEY (index_code, date)
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO index_price_daily (
                "index_code", "date", "index_name", "open", "close", "high", "low", "volume",
                "money"
            )
            SELECT
                CAST(index_code AS VARCHAR) AS index_code,
                CAST(date AS DATE) AS date,
                CAST(index_name AS VARCHAR) AS index_name,
                CAST(open AS DOUBLE) AS open,
                CAST(close AS DOUBLE) AS close,
                CAST(high AS DOUBLE) AS high,
                CAST(low AS DOUBLE) AS low,
                CAST(volume AS DOUBLE) AS volume,
                CAST(money AS DOUBLE) AS money
            FROM read_csv('{source}', header=true, auto_detect=true, ignore_errors=false)
            WHERE date IS NOT NULL AND index_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_index_price_date ON index_price_daily(date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT index_code)
            FROM index_price_daily
        """,
    ),
    "money_flow": DatasetSpec(
        name="money_flow",
        table="money_flow_daily",
        csv_glob="money_flow_*.csv",
        create_sql=f"""
            CREATE TABLE IF NOT EXISTS money_flow_daily (
                stock_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                {_doubles(MONEY_FLOW_FIELDS)},
                PRIMARY KEY (stock_code, date)
            )
        """,
        insert_sql=f"""
            INSERT OR REPLACE INTO money_flow_daily (
                "stock_code", "date", "change_pct", "net_amount_main", "net_pct_main",
                "net_amount_xl", "net_pct_xl", "net_amount_l", "net_pct_l", "net_amount_m",
                "net_pct_m", "net_amount_s", "net_pct_s"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(date AS DATE) AS date,
                {_casts(MONEY_FLOW_FIELDS)}
            FROM read_csv('{{source}}', header=true, auto_detect=true, ignore_errors=false)
            WHERE date IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_money_flow_date ON money_flow_daily(date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM money_flow_daily
        """,
    ),
    "northbound_holding": DatasetSpec(
        name="northbound_holding",
        table="northbound_holding_daily",
        csv_glob="northbound_holding_*.csv",
        create_sql=f"""
            CREATE TABLE IF NOT EXISTS northbound_holding_daily (
                stock_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                {_doubles(NORTHBOUND_HOLD_FIELDS)},
                PRIMARY KEY (stock_code, date)
            )
        """,
        insert_sql=f"""
            INSERT OR REPLACE INTO northbound_holding_daily (
                "stock_code", "date", "share_number", "share_ratio", "market_cap", "change",
                "change_ratio"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(date AS DATE) AS date,
                {_casts(NORTHBOUND_HOLD_FIELDS)}
            FROM read_csv('{{source}}', header=true, auto_detect=true, ignore_errors=false)
            WHERE date IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_northbound_holding_date ON northbound_holding_daily(date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM northbound_holding_daily
        """,
    ),
    "earnings_forecast": DatasetSpec(
        name="earnings_forecast",
        table="earnings_forecast_events",
        csv_glob="earnings_forecast_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS earnings_forecast_events (
                stock_code VARCHAR NOT NULL,
                pub_date DATE NOT NULL,
                report_date DATE NOT NULL,
                forecast_type INTEGER,
                forecast_abstract VARCHAR,
                forecast_content VARCHAR,
                profit_min DOUBLE,
                profit_max DOUBLE,
                profit_last DOUBLE,
                change_min DOUBLE,
                change_max DOUBLE,
                PRIMARY KEY (stock_code, pub_date, report_date)
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO earnings_forecast_events (
                "stock_code", "pub_date", "report_date", "forecast_type", "forecast_abstract",
                "forecast_content", "profit_min", "profit_max", "profit_last", "change_min",
                "change_max"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                TRY_CAST(pub_date AS DATE) AS pub_date,
                TRY_CAST(report_date AS DATE) AS report_date,
                TRY_CAST(forecast_type AS INTEGER) AS forecast_type,
                CAST(forecast_abstract AS VARCHAR) AS forecast_abstract,
                CAST(forecast_content AS VARCHAR) AS forecast_content,
                CAST(profit_min AS DOUBLE) AS profit_min,
                CAST(profit_max AS DOUBLE) AS profit_max,
                CAST(profit_last AS DOUBLE) AS profit_last,
                CAST(change_min AS DOUBLE) AS change_min,
                CAST(change_max AS DOUBLE) AS change_max
            FROM read_csv(
                '{source}',
                header=true,
                delim=',',
                quote='"',
                escape='"',
                strict_mode=false,
                null_padding=true,
                parallel=false,
                max_line_size=10000000,
                ignore_errors=false
            )
            WHERE TRY_CAST(pub_date AS DATE) IS NOT NULL
              AND TRY_CAST(report_date AS DATE) IS NOT NULL
              AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_earnings_forecast_pub_date ON earnings_forecast_events(pub_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM earnings_forecast_events
        """,
    ),
    "dividend_quality_financials": DatasetSpec(
        name="dividend_quality_financials",
        table="dividend_quality_financial_events",
        csv_glob="financial_events_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS dividend_quality_financial_events (
                statement_type VARCHAR NOT NULL,
                source_record_id BIGINT NOT NULL,
                stock_code VARCHAR NOT NULL,
                pub_date DATE NOT NULL,
                start_date DATE,
                end_date DATE NOT NULL,
                report_date DATE,
                report_type INTEGER,
                source_id VARCHAR,
                net_profit DOUBLE,
                net_operate_cash_flow DOUBLE,
                total_liability DOUBLE,
                total_assets DOUBLE,
                operating_revenue DOUBLE,
                total_operating_revenue DOUBLE,
                operating_profit DOUBLE,
                total_profit DOUBLE,
                income_net_profit DOUBLE,
                PRIMARY KEY (statement_type, source_record_id)
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO dividend_quality_financial_events (
                "statement_type", "source_record_id", "stock_code", "pub_date", "start_date",
                "end_date", "report_date", "report_type", "source_id", "net_profit",
                "net_operate_cash_flow", "total_liability", "total_assets", "operating_revenue",
                "total_operating_revenue", "operating_profit", "total_profit",
                "income_net_profit"
            )
            SELECT
                CAST(statement_type AS VARCHAR), TRY_CAST(source_record_id AS BIGINT),
                CAST(stock_code AS VARCHAR), TRY_CAST(pub_date AS DATE), TRY_CAST(start_date AS DATE),
                TRY_CAST(end_date AS DATE), TRY_CAST(report_date AS DATE), TRY_CAST(report_type AS INTEGER),
                CAST(source_id AS VARCHAR), TRY_CAST(net_profit AS DOUBLE),
                TRY_CAST(net_operate_cash_flow AS DOUBLE), TRY_CAST(total_liability AS DOUBLE),
                TRY_CAST(total_assets AS DOUBLE), TRY_CAST(operating_revenue AS DOUBLE),
                TRY_CAST(total_operating_revenue AS DOUBLE), TRY_CAST(operating_profit AS DOUBLE),
                TRY_CAST(total_profit AS DOUBLE), TRY_CAST(income_net_profit AS DOUBLE)
            FROM read_csv('{source}', header=true, auto_detect=true, union_by_name=true, ignore_errors=false)
            WHERE statement_type IS NOT NULL AND TRY_CAST(source_record_id AS BIGINT) IS NOT NULL
              AND stock_code IS NOT NULL AND TRY_CAST(pub_date AS DATE) IS NOT NULL
              AND TRY_CAST(end_date AS DATE) IS NOT NULL
              AND TRY_CAST(pub_date AS DATE) IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_dividend_quality_financial_pub_date ON dividend_quality_financial_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_dividend_quality_financial_stock ON dividend_quality_financial_events(stock_code, end_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM dividend_quality_financial_events
        """,
        migration_sql=(
            "ALTER TABLE dividend_quality_financial_events ADD COLUMN IF NOT EXISTS operating_revenue DOUBLE",
            "ALTER TABLE dividend_quality_financial_events ADD COLUMN IF NOT EXISTS total_operating_revenue DOUBLE",
            "ALTER TABLE dividend_quality_financial_events ADD COLUMN IF NOT EXISTS operating_profit DOUBLE",
            "ALTER TABLE dividend_quality_financial_events ADD COLUMN IF NOT EXISTS total_profit DOUBLE",
            "ALTER TABLE dividend_quality_financial_events ADD COLUMN IF NOT EXISTS income_net_profit DOUBLE",
        ),
    ),
    "earned_equity_financial_events": DatasetSpec(
        name="earned_equity_financial_events",
        table="earned_equity_financial_events",
        csv_glob="earned_equity_financial_events_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS earned_equity_financial_events (
                run_id VARCHAR NOT NULL,
                partition_year INTEGER NOT NULL,
                source_record_id BIGINT NOT NULL,
                stock_code VARCHAR NOT NULL,
                report_end_date DATE NOT NULL,
                pub_date DATE NOT NULL,
                report_type INTEGER,
                retained_earnings DOUBLE,
                total_owners_equity DOUBLE,
                paid_in_capital DOUBLE,
                capital_reserve DOUBLE,
                surplus_reserve DOUBLE,
                parent_company_owner_equity DOUBLE,
                minority_interests DOUBLE,
                PRIMARY KEY (stock_code, report_end_date, pub_date)
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO earned_equity_financial_events (
                "run_id", "partition_year", "source_record_id", "stock_code", "report_end_date",
                "pub_date", "report_type", "retained_earnings", "total_owners_equity",
                "paid_in_capital", "capital_reserve", "surplus_reserve",
                "parent_company_owner_equity", "minority_interests"
            )
            SELECT
                CAST(run_id AS VARCHAR), TRY_CAST(partition_year AS INTEGER),
                TRY_CAST(source_record_id AS BIGINT), CAST(stock_code AS VARCHAR),
                TRY_CAST(report_end_date AS DATE), TRY_CAST(pub_date AS DATE),
                TRY_CAST(report_type AS INTEGER), TRY_CAST(retained_earnings AS DOUBLE),
                TRY_CAST(total_owners_equity AS DOUBLE), TRY_CAST(paid_in_capital AS DOUBLE),
                TRY_CAST(capital_reserve AS DOUBLE), TRY_CAST(surplus_reserve AS DOUBLE),
                TRY_CAST(parent_company_owner_equity AS DOUBLE), TRY_CAST(minority_interests AS DOUBLE)
            FROM read_csv('{source}', header=true, auto_detect=true, union_by_name=true, ignore_errors=false)
            WHERE run_id IS NOT NULL AND TRY_CAST(partition_year AS INTEGER) IS NOT NULL
              AND TRY_CAST(source_record_id AS BIGINT) IS NOT NULL AND stock_code IS NOT NULL
              AND TRY_CAST(report_end_date AS DATE) IS NOT NULL AND TRY_CAST(pub_date AS DATE) IS NOT NULL
              AND COALESCE(
                    TRY_CAST(total_owners_equity AS DOUBLE),
                    TRY_CAST(parent_company_owner_equity AS DOUBLE)
                  ) IS NOT NULL
              AND TRY_CAST(pub_date AS DATE) IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_earned_equity_pub_date "
            "ON earned_equity_financial_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_earned_equity_stock "
            "ON earned_equity_financial_events(stock_code, report_end_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM earned_equity_financial_events
        """,
    ),
    "statutory_cash_claim_events": DatasetSpec(
        name="statutory_cash_claim_events",
        table="statutory_cash_claim_events",
        csv_glob="statutory_cash_claim_events_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS statutory_cash_claim_events (
                run_id VARCHAR NOT NULL,
                partition_year INTEGER NOT NULL,
                source_record_id BIGINT NOT NULL,
                stock_code VARCHAR NOT NULL,
                report_end_date DATE NOT NULL,
                pub_date DATE NOT NULL,
                report_type INTEGER,
                taxs_payable DOUBLE,
                salaries_payable DOUBLE,
                interest_payable DOUBLE,
                dividend_payable DOUBLE,
                total_assets DOUBLE NOT NULL,
                deferred_tax_assets DOUBLE,
                bill_and_account_payable DOUBLE,
                lease_liability DOUBLE,
                PRIMARY KEY (stock_code, report_end_date, pub_date)
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO statutory_cash_claim_events (
                run_id, partition_year, source_record_id, stock_code,
                report_end_date, pub_date, report_type, taxs_payable,
                salaries_payable, interest_payable, dividend_payable,
                total_assets, deferred_tax_assets, bill_and_account_payable,
                lease_liability
            )
            SELECT
                CAST(run_id AS VARCHAR), TRY_CAST(partition_year AS INTEGER),
                TRY_CAST(source_record_id AS BIGINT), CAST(stock_code AS VARCHAR),
                TRY_CAST(report_end_date AS DATE), TRY_CAST(pub_date AS DATE),
                TRY_CAST(report_type AS INTEGER), TRY_CAST(taxs_payable AS DOUBLE),
                TRY_CAST(salaries_payable AS DOUBLE), TRY_CAST(interest_payable AS DOUBLE),
                TRY_CAST(dividend_payable AS DOUBLE), TRY_CAST(total_assets AS DOUBLE),
                TRY_CAST(deferred_tax_assets AS DOUBLE),
                TRY_CAST(bill_and_account_payable AS DOUBLE),
                TRY_CAST(lease_liability AS DOUBLE)
            FROM read_csv('{source}', header=true, auto_detect=true, union_by_name=true, ignore_errors=false)
            WHERE run_id IS NOT NULL AND TRY_CAST(partition_year AS INTEGER) IS NOT NULL
              AND TRY_CAST(source_record_id AS BIGINT) IS NOT NULL AND stock_code IS NOT NULL
              AND TRY_CAST(report_end_date AS DATE) IS NOT NULL AND TRY_CAST(pub_date AS DATE) IS NOT NULL
              AND TRY_CAST(total_assets AS DOUBLE) IS NOT NULL
              AND TRY_CAST(pub_date AS DATE) IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_statutory_cash_claim_pub_date ON statutory_cash_claim_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_statutory_cash_claim_stock ON statutory_cash_claim_events(stock_code, report_end_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM statutory_cash_claim_events
        """,
        migration_sql=(
            "ALTER TABLE statutory_cash_claim_events "
            "ADD COLUMN IF NOT EXISTS deferred_tax_assets DOUBLE",
            "ALTER TABLE statutory_cash_claim_events "
            "ADD COLUMN IF NOT EXISTS bill_and_account_payable DOUBLE",
            "ALTER TABLE statutory_cash_claim_events "
            "ADD COLUMN IF NOT EXISTS lease_liability DOUBLE",
        ),
    ),
    "fund_portfolio_stock_events": DatasetSpec(
        name="fund_portfolio_stock_events",
        table="fund_portfolio_stock_events",
        csv_glob="fund_portfolio_stock_events_????.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS fund_portfolio_stock_events (
                run_id VARCHAR NOT NULL,
                partition_year INTEGER NOT NULL,
                source_record_id BIGINT PRIMARY KEY,
                fund_code VARCHAR NOT NULL,
                stock_code VARCHAR NOT NULL,
                report_end_date DATE NOT NULL,
                pub_date DATE NOT NULL,
                report_type_id INTEGER,
                report_type VARCHAR,
                rank INTEGER,
                shares DOUBLE,
                market_cap DOUBLE,
                proportion DOUBLE
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO fund_portfolio_stock_events (
                "run_id", "partition_year", "source_record_id", "fund_code", "stock_code",
                "report_end_date", "pub_date", "report_type_id", "report_type", "rank",
                "shares", "market_cap", "proportion"
            )
            SELECT
                CAST(run_id AS VARCHAR), TRY_CAST(partition_year AS INTEGER),
                TRY_CAST(source_record_id AS BIGINT), CAST(fund_code AS VARCHAR),
                CAST(stock_code AS VARCHAR), TRY_CAST(report_end_date AS DATE),
                TRY_CAST(pub_date AS DATE), TRY_CAST(report_type_id AS INTEGER),
                CAST(report_type AS VARCHAR), TRY_CAST(rank AS INTEGER),
                TRY_CAST(shares AS DOUBLE), TRY_CAST(market_cap AS DOUBLE),
                TRY_CAST(proportion AS DOUBLE)
            FROM read_csv('{source}', header=true, auto_detect=true, union_by_name=true, ignore_errors=false)
            WHERE run_id IS NOT NULL AND TRY_CAST(partition_year AS INTEGER) IS NOT NULL
              AND TRY_CAST(source_record_id AS BIGINT) IS NOT NULL
              AND fund_code IS NOT NULL AND stock_code IS NOT NULL
              AND TRY_CAST(report_end_date AS DATE) IS NOT NULL
              AND TRY_CAST(pub_date AS DATE) IS NOT NULL
              AND TRY_CAST(pub_date AS DATE) IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_fund_portfolio_pub_date "
            "ON fund_portfolio_stock_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_fund_portfolio_stock "
            "ON fund_portfolio_stock_events(stock_code, report_end_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM fund_portfolio_stock_events
        """,
    ),
    "listing_compliance_status_events": DatasetSpec(
        name="listing_compliance_status_events",
        table="listing_compliance_status_events",
        csv_glob="listing_compliance_status_events_????.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS listing_compliance_status_events (
                id BIGINT PRIMARY KEY,
                code VARCHAR NOT NULL,
                pub_date DATE NOT NULL,
                change_date DATE NOT NULL,
                change_type VARCHAR NOT NULL,
                change_reason VARCHAR,
                public_status VARCHAR NOT NULL
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO listing_compliance_status_events (
                "id", "code", "pub_date", "change_date", "change_type", "change_reason",
                "public_status"
            )
            SELECT
                TRY_CAST(id AS BIGINT), CAST(code AS VARCHAR),
                TRY_CAST(pub_date AS DATE), TRY_CAST(change_date AS DATE),
                CAST(change_type AS VARCHAR), CAST(change_reason AS VARCHAR),
                CAST(public_status AS VARCHAR)
            FROM read_csv('{source}', header=true, auto_detect=true, union_by_name=true, ignore_errors=false)
            WHERE TRY_CAST(id AS BIGINT) IS NOT NULL AND code IS NOT NULL
              AND TRY_CAST(pub_date AS DATE) IS NOT NULL
              AND TRY_CAST(change_date AS DATE) IS NOT NULL AND change_type IS NOT NULL
              AND public_status IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_listing_compliance_status_pub_date ON listing_compliance_status_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_listing_compliance_status_code ON listing_compliance_status_events(code, change_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT code)
            FROM listing_compliance_status_events
        """,
    ),
    "audit_opinion_events": DatasetSpec(
        name="audit_opinion_events",
        table="audit_opinion_events",
        csv_glob="audit_opinion_events_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS audit_opinion_events (
                source_record_id BIGINT PRIMARY KEY,
                stock_code VARCHAR NOT NULL,
                pub_date DATE NOT NULL,
                activation_date DATE NOT NULL,
                report_date DATE,
                report_type INTEGER,
                accounting_firm VARCHAR,
                accountant VARCHAR,
                opinion_type_id INTEGER,
                opinion_type VARCHAR
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO audit_opinion_events (
                "source_record_id", "stock_code", "pub_date", "activation_date", "report_date",
                "report_type", "accounting_firm", "accountant", "opinion_type_id",
                "opinion_type"
            )
            SELECT
                TRY_CAST(id AS BIGINT),
                CAST(code AS VARCHAR),
                TRY_CAST(pub_date AS DATE),
                (SELECT MIN(date) FROM price_data_daily WHERE date > TRY_CAST(pub_date AS DATE)),
                TRY_CAST(end_date AS DATE),
                TRY_CAST(report_type AS INTEGER),
                CAST(accounting_firm AS VARCHAR),
                CAST(accountant AS VARCHAR),
                TRY_CAST(opinion_type_id AS INTEGER),
                CAST(opinion_type AS VARCHAR)
            FROM read_csv('{source}', header=true, auto_detect=true, ignore_errors=false)
            WHERE TRY_CAST(id AS BIGINT) IS NOT NULL
              AND code IS NOT NULL
              AND TRY_CAST(pub_date AS DATE) IS NOT NULL
              AND (SELECT MIN(date) FROM price_data_daily WHERE date > TRY_CAST(pub_date AS DATE)) IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_audit_opinion_pub_date ON audit_opinion_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_audit_opinion_activation_date ON audit_opinion_events(activation_date)",
            "CREATE INDEX IF NOT EXISTS idx_audit_opinion_stock ON audit_opinion_events(stock_code)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM audit_opinion_events
        """,
    ),
    "shareholder_structure": DatasetSpec(
        name="shareholder_structure",
        table="shareholder_structure",
        csv_glob="shareholder_structure_*.csv",
        create_sql=f"""
            CREATE TABLE IF NOT EXISTS shareholder_structure (
                stock_code VARCHAR NOT NULL,
                end_date DATE NOT NULL,
                pub_date DATE,
                report_type INTEGER,
                holder_num DOUBLE,
                {_doubles(SHAREHOLDER_STRUCTURE_FIELDS[3:])},
                PRIMARY KEY (stock_code, end_date)
            )
        """,
        insert_sql=f"""
            INSERT OR REPLACE INTO shareholder_structure (
                "stock_code", "end_date", "pub_date", "report_type", "holder_num",
                "holder_avg_pct", "holder_avg_number", "top10_share_ratio",
                "floating_top10_share_ratio", "institution_top10_share_ratio"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(end_date AS DATE) AS end_date,
                TRY_CAST(pub_date AS DATE) AS pub_date,
                TRY_CAST(report_type AS INTEGER) AS report_type,
                CAST(holder_num AS DOUBLE) AS holder_num,
                {_casts(SHAREHOLDER_STRUCTURE_FIELDS[3:])}
            FROM read_csv('{{source}}', header=true, auto_detect=true, ignore_errors=false)
            WHERE end_date IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_shareholder_structure_end_date ON shareholder_structure(end_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(end_date), MAX(end_date), COUNT(DISTINCT stock_code)
            FROM shareholder_structure
        """,
    ),
    "capital_change": DatasetSpec(
        name="capital_change",
        table="capital_change_events",
        csv_glob="capital_change_*.csv",
        create_sql=f"""
            CREATE TABLE IF NOT EXISTS capital_change_events (
                stock_code VARCHAR NOT NULL,
                change_date DATE NOT NULL,
                pub_date DATE,
                change_reason_id VARCHAR,
                change_reason VARCHAR,
                {_doubles(CAPITAL_CHANGE_FIELDS)},
                PRIMARY KEY (stock_code, change_date)
            )
        """,
        insert_sql=f"""
            INSERT OR REPLACE INTO capital_change_events (
                "stock_code", "change_date", "pub_date", "change_reason_id", "change_reason",
                "share_total", "share_non_trade", "share_limited", "share_trade_total",
                "share_rmb", "share_b", "share_h", "share_management"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                TRY_CAST(change_date AS DATE) AS change_date,
                TRY_CAST(pub_date AS DATE) AS pub_date,
                CAST(change_reason_id AS VARCHAR) AS change_reason_id,
                CAST(change_reason AS VARCHAR) AS change_reason,
                {_casts(CAPITAL_CHANGE_FIELDS)}
            FROM read_csv('{{source}}', header=true, auto_detect=true, ignore_errors=false)
            WHERE TRY_CAST(change_date AS DATE) IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_capital_change_date ON capital_change_events(change_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(change_date), MAX(change_date), COUNT(DISTINCT stock_code)
            FROM capital_change_events
        """,
    ),
    "margin_trading": DatasetSpec(
        name="margin_trading",
        table="margin_trading_daily",
        csv_glob="margin_trading_*.csv",
        create_sql=f"""
            CREATE TABLE IF NOT EXISTS margin_trading_daily (
                stock_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                {_doubles(MARGIN_TRADING_FIELDS)},
                PRIMARY KEY (stock_code, date)
            )
        """,
        insert_sql=f"""
            INSERT OR REPLACE INTO margin_trading_daily (
                "stock_code", "date", "fin_value", "fin_buy_value", "fin_refund_value",
                "sec_value", "sec_sell_value", "sec_refund_value", "fin_sec_value"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(date AS DATE) AS date,
                {_casts(MARGIN_TRADING_FIELDS)}
            FROM read_csv('{{source}}', header=true, auto_detect=true, ignore_errors=false)
            WHERE date IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_margin_trading_date ON margin_trading_daily(date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM margin_trading_daily
        """,
    ),
    "billboard": DatasetSpec(
        name="billboard",
        table="billboard_events",
        csv_glob="billboard_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS billboard_events (
                stock_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                direction VARCHAR,
                rank INTEGER,
                abnormal_code VARCHAR,
                abnormal_name VARCHAR,
                sales_depart_name VARCHAR,
                buy_value DOUBLE,
                buy_rate DOUBLE,
                sell_value DOUBLE,
                sell_rate DOUBLE,
                total_value DOUBLE,
                net_value DOUBLE,
                amount DOUBLE
            )
        """,
        insert_sql=f"""
            INSERT INTO billboard_events (
                "stock_code", "date", "direction", "rank", "abnormal_code", "abnormal_name",
                "sales_depart_name", "buy_value", "buy_rate", "sell_value", "sell_rate",
                "total_value", "net_value", "amount"
            )
            SELECT * FROM (
                SELECT DISTINCT
                    CAST(stock_code AS VARCHAR) AS stock_code,
                    TRY_CAST(date AS DATE) AS date,
                    CAST(direction AS VARCHAR) AS direction,
                    TRY_CAST(rank AS INTEGER) AS rank,
                    CAST(abnormal_code AS VARCHAR) AS abnormal_code,
                    CAST(abnormal_name AS VARCHAR) AS abnormal_name,
                    CAST(sales_depart_name AS VARCHAR) AS sales_depart_name,
                    TRY_CAST(buy_value AS DOUBLE) AS buy_value,
                    TRY_CAST(buy_rate AS DOUBLE) AS buy_rate,
                    TRY_CAST(sell_value AS DOUBLE) AS sell_value,
                    TRY_CAST(sell_rate AS DOUBLE) AS sell_rate,
                    TRY_CAST(total_value AS DOUBLE) AS total_value,
                    TRY_CAST(net_value AS DOUBLE) AS net_value,
                    TRY_CAST(amount AS DOUBLE) AS amount
                FROM read_csv('{{source}}', {ROBUST_CSV_OPTS})
                WHERE TRY_CAST(date AS DATE) IS NOT NULL AND stock_code IS NOT NULL
                EXCEPT
                SELECT "stock_code", "date", "direction", "rank", "abnormal_code", "abnormal_name", "sales_depart_name", "buy_value", "buy_rate", "sell_value", "sell_rate", "total_value", "net_value", "amount" FROM billboard_events
            )
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_billboard_date ON billboard_events(date)",
            "CREATE INDEX IF NOT EXISTS idx_billboard_stock ON billboard_events(stock_code)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM billboard_events
        """,
    ),
    "holder_num": DatasetSpec(
        name="holder_num",
        table="holder_num",
        csv_glob="holder_num_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS holder_num (
                stock_code VARCHAR NOT NULL,
                end_date DATE NOT NULL,
                pub_date DATE,
                share_holders DOUBLE,
                a_share_holders DOUBLE,
                b_share_holders DOUBLE,
                h_share_holders DOUBLE,
                PRIMARY KEY (stock_code, end_date)
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO holder_num (
                "stock_code", "end_date", "pub_date", "share_holders", "a_share_holders",
                "b_share_holders", "h_share_holders"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                TRY_CAST(end_date AS DATE) AS end_date,
                TRY_CAST(pub_date AS DATE) AS pub_date,
                TRY_CAST(share_holders AS DOUBLE) AS share_holders,
                TRY_CAST(a_share_holders AS DOUBLE) AS a_share_holders,
                TRY_CAST(b_share_holders AS DOUBLE) AS b_share_holders,
                TRY_CAST(h_share_holders AS DOUBLE) AS h_share_holders
            FROM read_csv('{source}', header=true, auto_detect=true, ignore_errors=false)
            WHERE TRY_CAST(end_date AS DATE) IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_holder_num_end_date ON holder_num(end_date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(end_date), MAX(end_date), COUNT(DISTINCT stock_code)
            FROM holder_num
        """,
    ),
    "locked_shares": DatasetSpec(
        name="locked_shares",
        table="locked_shares_events",
        csv_glob="locked_shares_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS locked_shares_events (
                stock_code VARCHAR NOT NULL,
                date DATE NOT NULL,
                num DOUBLE,
                rate1 DOUBLE,
                rate2 DOUBLE,
                PRIMARY KEY (stock_code, date)
            )
        """,
        insert_sql="""
            INSERT OR REPLACE INTO locked_shares_events (
                "stock_code", "date", "num", "rate1", "rate2"
            )
            SELECT
                CAST(stock_code AS VARCHAR) AS stock_code,
                TRY_CAST(date AS DATE) AS date,
                TRY_CAST(num AS DOUBLE) AS num,
                TRY_CAST(rate1 AS DOUBLE) AS rate1,
                TRY_CAST(rate2 AS DOUBLE) AS rate2
            FROM read_csv('{source}', header=true, auto_detect=true, ignore_errors=false)
            WHERE TRY_CAST(date AS DATE) IS NOT NULL AND stock_code IS NOT NULL
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_locked_shares_date ON locked_shares_events(date)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT stock_code)
            FROM locked_shares_events
        """,
    ),
    "shares_pledge": DatasetSpec(
        name="shares_pledge",
        table="shares_pledge_events",
        csv_glob="shares_pledge_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS shares_pledge_events (
                stock_code VARCHAR NOT NULL,
                pub_date DATE NOT NULL,
                pledgor VARCHAR,
                pledgee VARCHAR,
                pledge_item VARCHAR,
                pledge_nature VARCHAR,
                pledge_number DOUBLE,
                pledge_total_ratio DOUBLE,
                start_date DATE,
                end_date DATE,
                unpledged_date DATE,
                unpledged_number DOUBLE,
                is_buy_back VARCHAR
            )
        """,
        insert_sql=f"""
            INSERT INTO shares_pledge_events (
                "stock_code", "pub_date", "pledgor", "pledgee", "pledge_item", "pledge_nature",
                "pledge_number", "pledge_total_ratio", "start_date", "end_date",
                "unpledged_date", "unpledged_number", "is_buy_back"
            )
            SELECT stock_code, pub_date, pledgor, pledgee, pledge_item, pledge_nature,
                   pledge_number, pledge_total_ratio, start_date, end_date,
                   unpledged_date, unpledged_number, is_buy_back
            FROM (
                SELECT
                    CAST(stock_code AS VARCHAR) AS stock_code,
                    TRY_CAST(pub_date AS DATE) AS pub_date,
                    CAST(pledgor AS VARCHAR) AS pledgor,
                    CAST(pledgee AS VARCHAR) AS pledgee,
                    CAST(pledge_item AS VARCHAR) AS pledge_item,
                    CAST(pledge_nature AS VARCHAR) AS pledge_nature,
                    TRY_CAST(pledge_number AS DOUBLE) AS pledge_number,
                    TRY_CAST(pledge_total_ratio AS DOUBLE) AS pledge_total_ratio,
                    TRY_CAST(start_date AS DATE) AS start_date,
                    TRY_CAST(end_date AS DATE) AS end_date,
                    TRY_CAST(unpledged_date AS DATE) AS unpledged_date,
                    TRY_CAST(unpledged_number AS DOUBLE) AS unpledged_number,
                    CAST(is_buy_back AS VARCHAR) AS is_buy_back,
                    ROW_NUMBER() OVER (
                        PARTITION BY stock_code, pub_date, pledgor, pledgee, pledge_number, start_date
                    ) AS natural_key_rank
                FROM read_csv('{{source}}', {ROBUST_CSV_OPTS})
                WHERE TRY_CAST(pub_date AS DATE) IS NOT NULL AND stock_code IS NOT NULL
            ) AS incoming
            WHERE incoming.natural_key_rank = 1
              AND NOT EXISTS (
                  SELECT 1
                  FROM shares_pledge_events AS existing
                  WHERE existing.stock_code IS NOT DISTINCT FROM incoming.stock_code
                    AND existing.pub_date IS NOT DISTINCT FROM incoming.pub_date
                    AND existing.pledgor IS NOT DISTINCT FROM incoming.pledgor
                    AND existing.pledgee IS NOT DISTINCT FROM incoming.pledgee
                    AND existing.pledge_number IS NOT DISTINCT FROM incoming.pledge_number
                    AND existing.start_date IS NOT DISTINCT FROM incoming.start_date
              )
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_shares_pledge_pub_date ON shares_pledge_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_shares_pledge_stock ON shares_pledge_events(stock_code)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM shares_pledge_events
        """,
    ),
    "shares_frozen": DatasetSpec(
        name="shares_frozen",
        table="shares_frozen_events",
        csv_glob="shares_frozen_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS shares_frozen_events (
                stock_code VARCHAR NOT NULL,
                pub_date DATE NOT NULL,
                frozen_person VARCHAR,
                frozen_reason VARCHAR,
                frozen_share_nature VARCHAR,
                frozen_number DOUBLE,
                frozen_total_ratio DOUBLE,
                freeze_applicant VARCHAR,
                freeze_executor VARCHAR,
                start_date DATE,
                end_date DATE,
                unfrozen_date DATE,
                unfrozen_number DOUBLE
            )
        """,
        insert_sql=f"""
            INSERT INTO shares_frozen_events (
                "stock_code", "pub_date", "frozen_person", "frozen_reason",
                "frozen_share_nature", "frozen_number", "frozen_total_ratio",
                "freeze_applicant", "freeze_executor", "start_date", "end_date",
                "unfrozen_date", "unfrozen_number"
            )
            SELECT * FROM (
                SELECT DISTINCT
                    CAST(stock_code AS VARCHAR) AS stock_code,
                    TRY_CAST(pub_date AS DATE) AS pub_date,
                    CAST(frozen_person AS VARCHAR) AS frozen_person,
                    CAST(frozen_reason AS VARCHAR) AS frozen_reason,
                    CAST(frozen_share_nature AS VARCHAR) AS frozen_share_nature,
                    TRY_CAST(frozen_number AS DOUBLE) AS frozen_number,
                    TRY_CAST(frozen_total_ratio AS DOUBLE) AS frozen_total_ratio,
                    CAST(freeze_applicant AS VARCHAR) AS freeze_applicant,
                    CAST(freeze_executor AS VARCHAR) AS freeze_executor,
                    TRY_CAST(start_date AS DATE) AS start_date,
                    TRY_CAST(end_date AS DATE) AS end_date,
                    TRY_CAST(unfrozen_date AS DATE) AS unfrozen_date,
                    TRY_CAST(unfrozen_number AS DOUBLE) AS unfrozen_number
                FROM read_csv('{{source}}', {ROBUST_CSV_OPTS})
                WHERE TRY_CAST(pub_date AS DATE) IS NOT NULL AND stock_code IS NOT NULL
                EXCEPT
                SELECT "stock_code", "pub_date", "frozen_person", "frozen_reason", "frozen_share_nature", "frozen_number", "frozen_total_ratio", "freeze_applicant", "freeze_executor", "start_date", "end_date", "unfrozen_date", "unfrozen_number" FROM shares_frozen_events
            )
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_shares_frozen_pub_date ON shares_frozen_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_shares_frozen_stock ON shares_frozen_events(stock_code)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM shares_frozen_events
        """,
    ),
    "shareholder_share_change": DatasetSpec(
        name="shareholder_share_change",
        table="shareholder_share_change_events",
        csv_glob="shareholder_share_change_*.csv",
        create_sql="""
            CREATE TABLE IF NOT EXISTS shareholder_share_change_events (
                stock_code VARCHAR NOT NULL,
                pub_date DATE NOT NULL,
                end_date DATE,
                type INTEGER,
                shareholder_name VARCHAR,
                change_number DOUBLE,
                change_ratio DOUBLE,
                price_ceiling DOUBLE,
                after_change_ratio DOUBLE
            )
        """,
        insert_sql=f"""
            INSERT INTO shareholder_share_change_events (
                "stock_code", "pub_date", "end_date", "type", "shareholder_name",
                "change_number", "change_ratio", "price_ceiling", "after_change_ratio"
            )
            SELECT * FROM (
                SELECT DISTINCT
                    CAST(stock_code AS VARCHAR) AS stock_code,
                    TRY_CAST(pub_date AS DATE) AS pub_date,
                    TRY_CAST(end_date AS DATE) AS end_date,
                    TRY_CAST(type AS INTEGER) AS type,
                    CAST(shareholder_name AS VARCHAR) AS shareholder_name,
                    TRY_CAST(change_number AS DOUBLE) AS change_number,
                    TRY_CAST(change_ratio AS DOUBLE) AS change_ratio,
                    TRY_CAST(price_ceiling AS DOUBLE) AS price_ceiling,
                    TRY_CAST(after_change_ratio AS DOUBLE) AS after_change_ratio
                FROM read_csv('{{source}}', {ROBUST_CSV_OPTS})
                WHERE TRY_CAST(pub_date AS DATE) IS NOT NULL AND stock_code IS NOT NULL
                EXCEPT
                SELECT "stock_code", "pub_date", "end_date", "type", "shareholder_name", "change_number", "change_ratio", "price_ceiling", "after_change_ratio" FROM shareholder_share_change_events
            )
        """,
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_shareholder_share_change_pub_date "
            "ON shareholder_share_change_events(pub_date)",
            "CREATE INDEX IF NOT EXISTS idx_shareholder_share_change_stock "
            "ON shareholder_share_change_events(stock_code)",
        ),
        summary_sql="""
            SELECT COUNT(*), MIN(pub_date), MAX(pub_date), COUNT(DISTINCT stock_code)
            FROM shareholder_share_change_events
        """,
    ),
}

DEFAULT_SOURCES: dict[str, Path] = {
    "security_master": DEFAULT_SOURCE_ROOT / "security_master",
    "historical_st": DEFAULT_SOURCE_ROOT / "historical_st",
    "shareholder_top10": DEFAULT_SOURCE_ROOT / "shareholder_top10",
    "valuation": DEFAULT_SOURCE_ROOT / "valuation",
    "style": DEFAULT_SOURCE_ROOT / "style_factors",
    "industry": DEFAULT_SOURCE_ROOT / "industry",
    "index_price": DEFAULT_SOURCE_ROOT / "index_price",
    "money_flow": DEFAULT_SOURCE_ROOT / "money_flow",
    "northbound_holding": DEFAULT_SOURCE_ROOT / "northbound_holding",
    "earnings_forecast": DEFAULT_SOURCE_ROOT / "earnings_forecast",
    "dividend_quality_financials": DEFAULT_SOURCE_ROOT / "dividend_quality_financials",
    "earned_equity_financial_events": DEFAULT_SOURCE_ROOT / "dividend_lowvol_wide/earned_equity",
    "statutory_cash_claim_events": DEFAULT_SOURCE_ROOT / "dividend_lowvol_wide/statutory_cash_claims",
    "fund_portfolio_stock_events": DEFAULT_SOURCE_ROOT / "dividend_lowvol_wide/fund_portfolio_stock",
    "listing_compliance_status_events": DEFAULT_SOURCE_ROOT / "dividend_lowvol_wide/listing_compliance_status",
    "audit_opinion_events": DEFAULT_SOURCE_ROOT / "audit_opinion_events",
    "shareholder_structure": DEFAULT_SOURCE_ROOT / "shareholder_structure",
    "capital_change": DEFAULT_SOURCE_ROOT / "capital_change",
    "margin_trading": DEFAULT_SOURCE_ROOT / "margin_trading",
    "billboard": DEFAULT_SOURCE_ROOT / "billboard",
    "holder_num": DEFAULT_SOURCE_ROOT / "holder_num",
    "locked_shares": DEFAULT_SOURCE_ROOT / "locked_shares",
    "shares_pledge": DEFAULT_SOURCE_ROOT / "shares_pledge",
    "shares_frozen": DEFAULT_SOURCE_ROOT / "shares_frozen",
    "shareholder_share_change": DEFAULT_SOURCE_ROOT / "shareholder_share_change",
}


@atomic_import
def import_dataset(
    conn: duckdb.DuckDBPyConnection,
    spec: DatasetSpec,
    source_dir: Path,
    replace: bool,
) -> None:
    matches = sorted(source_dir.glob(spec.csv_glob))
    if not matches:
        raise LoaderError(f"no files matching '{spec.csv_glob}' in {source_dir}")

    source_pattern = str(source_dir / spec.csv_glob)
    print(f"\n[INFO] === {spec.name} -> {spec.table} ===")
    print(f"[INFO] source: {source_pattern} ({len(matches)} files)")

    start = time.time()
    conn.execute(spec.create_sql)
    if spec.name == 'shares_pledge' and os.environ.get('QUANT_INPUT_BATCH_STAGING') == '1':
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from utils.input_store import seed_input_rows

        parents = json.loads(os.environ.get('QUANT_INPUT_PARENT_BINDINGS', '{}'))
        conn.execute(f'''CREATE OR REPLACE TEMP TABLE pledge_keys AS SELECT
            CAST(stock_code AS VARCHAR) AS stock_code, TRY_CAST(pub_date AS DATE) AS pub_date,
            CAST(pledgor AS VARCHAR) AS pledgor, CAST(pledgee AS VARCHAR) AS pledgee,
            TRY_CAST(pledge_number AS DOUBLE) AS pledge_number, TRY_CAST(start_date AS DATE) AS start_date
            FROM read_csv(?, {ROBUST_CSV_OPTS})''', [source_pattern])
        seed_input_rows(conn, spec.table, 'pledge_keys',
                        ['stock_code','pub_date','pledgor','pledgee','pledge_number','start_date'], bindings=parents)
    for stmt in spec.migration_sql:
        conn.execute(stmt)
    if replace:
        print(f"[INFO] clearing existing rows in {spec.table}")
        conn.execute(f"DELETE FROM {spec.table}")

    conn.execute(spec.insert_sql.replace("{source}", source_pattern.replace("'", "''")))
    for stmt in spec.index_sql:
        conn.execute(stmt)

    count, dmin, dmax, n_entities = conn.execute(spec.summary_sql).fetchone()
    elapsed = time.time() - start
    print(f"[INFO] {spec.table}: {count:,} rows, {dmin} ~ {dmax}, "
          f"{n_entities} distinct entities ({elapsed:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Import portfolio-optimization datasets to DuckDB")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[*DATASETS.keys(), "all"],
        help="Dataset to import (or 'all')",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Directory containing the dataset CSVs (default: per-dataset dir under "
             f"{DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to DuckDB database file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing table rows before import (default: upsert)",
    )
    args = parser.parse_args()

    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    if args.source is not None and len(names) > 1:
        parser.error("--source can only be used with a single --dataset")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.db))
    try:
        conn.execute("SET threads TO 4")
        conn.execute("SET memory_limit = '4GB'")
        for name in names:
            spec = DATASETS[name]
            source_dir = args.source if args.source is not None else DEFAULT_SOURCES[name]
            import_dataset(conn, spec, source_dir, replace=args.replace)
    finally:
        conn.close()

    print(f"\n[INFO] done. database: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
