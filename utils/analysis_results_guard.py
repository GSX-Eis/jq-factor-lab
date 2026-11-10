from __future__ import annotations

import re
from typing import Any

LEGACY_POOLED_RESULTS_ERROR = (
    "legacy pooled analysis results are unsupported; run reset-analysis-results and rebuild"
)

_LEGACY_FACTOR_SUFFIX_RE = re.compile(r".+_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_rules$")
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _duckdb_tables(conn: Any) -> set[str]:
    rows = conn.execute("SHOW TABLES").fetchall()
    return {str(row[0]).translate(_ASCII_LOWER) for row in rows}


def _duckdb_columns(conn: Any, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]).translate(_ASCII_LOWER) for row in rows}


def _has_legacy_factor_names_duckdb(conn: Any) -> bool:
    for table_name in ("factor_analysis_results", "factor_values", "factor_monthly_results", "monthly_pool_results"):
        tables = _duckdb_tables(conn)
        if table_name not in tables:
            continue
        cursor = conn.execute(f"SELECT factor_name FROM {table_name}")
        while rows := cursor.fetchmany(10000):
            for row in rows:
                value = str(row[0]) if row and row[0] is not None else ""
                if _LEGACY_FACTOR_SUFFIX_RE.fullmatch(value):
                    return True
    return False


def assert_no_dynamic_pool_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        return
    if config.get("use_dynamic_pools"):
        raise RuntimeError(LEGACY_POOLED_RESULTS_ERROR)
    if "dynamic_pools" in config:
        raise RuntimeError(LEGACY_POOLED_RESULTS_ERROR)


def assert_no_legacy_pooled_results_duckdb(conn: Any) -> None:
    tables = _duckdb_tables(conn)
    if "monthly_pool_results" in tables:
        raise RuntimeError(LEGACY_POOLED_RESULTS_ERROR)
    for table_name in ("factor_analysis_results", "factor_values", "factor_monthly_results"):
        if table_name not in tables:
            continue
        if "pool_type" in _duckdb_columns(conn, table_name):
            raise RuntimeError(LEGACY_POOLED_RESULTS_ERROR)
    if _has_legacy_factor_names_duckdb(conn):
        raise RuntimeError(LEGACY_POOLED_RESULTS_ERROR)
