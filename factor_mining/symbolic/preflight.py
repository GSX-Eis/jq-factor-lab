"""Fail-fast data availability preflight for symbolic mining runs.

Cheap, read-only checks that run BEFORE any panel construction or search so a
detached data volume / missing table / broken weekly-pool schema aborts the
run immediately with a structured ``data_preflight_failed`` error instead of
burning a config-search trial and a run bundle on a guaranteed failure (the
2026-06-03 incident burnt 39 runs against an unmounted DuckDB volume).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from utils.data_store import binding_contract

try:  # pragma: no cover - environment dependent
    import duckdb
    from utils.data_store import connect_read_only as _connect_data_read_only
except ImportError:  # pragma: no cover - environment dependent
    duckdb = None

from .config import AnalysisContext

DATA_PREFLIGHT_ERROR_CODE = "data_preflight_failed"

# Pool filter fields that do not live in the fundamental table.
_NON_FUNDAMENTAL_POOL_FIELDS = {"index_code"}


class DataPreflightError(RuntimeError):
    """Mining data backend unreachable or structurally broken."""

    error_code = DATA_PREFLIGHT_ERROR_CODE


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _pool_filter_fields(raw_config: Dict[str, Any]) -> List[str]:
    """Collect filter fields referenced by enabled dynamic pools."""
    pools = raw_config.get("dynamic_pools")
    if not isinstance(pools, dict):
        return []
    fields: List[str] = []
    for pool in pools.values():
        if not isinstance(pool, dict) or not pool.get("enabled", True):
            continue
        for item in pool.get("filters") or []:
            if isinstance(item, dict):
                field = str(item.get("field") or "").strip()
                if field and field not in fields:
                    fields.append(field)
    return fields


def _probe_table(conn: Any, table: str) -> List[str]:
    """``SELECT * FROM table LIMIT 1``; require one row, return its columns."""
    try:
        cursor = conn.execute(f"SELECT * FROM {_quote_ident(table)} LIMIT 1")
        rows = cursor.fetchall()
        columns = [str(desc[0]) for desc in cursor.description or []]
    except Exception as exc:
        raise DataPreflightError(f"table probe failed for {table!r}: {exc}") from exc
    if not rows:
        raise DataPreflightError(f"table {table!r} returned no rows (empty backend?)")
    return columns


def run_data_preflight(analysis_context: AnalysisContext) -> Dict[str, Any]:
    """Validate the data backend is reachable and the weekly pool is queryable.

    Raises :class:`DataPreflightError` on the first failed check. Returns a
    summary of the performed checks for progress/summary logging.
    """
    runtime = analysis_context.runtime_config
    backend = str(runtime.get("db_backend") or "auto").lower()
    contract = binding_contract()
    duckdb_path = str(runtime.get("duckdb_path") or contract.get("input_db") or "").strip()
    price_table = str(runtime.get("price_table") or "price_data_daily")
    fundamental_table = str(runtime.get("fundamental_table") or "factor_data")
    pool_fields = _pool_filter_fields(analysis_context.raw_config)

    summary: Dict[str, Any] = {
        "ok": True,
        "backend": backend,
        "duckdb_path": duckdb_path,
        "price_table": price_table,
        "fundamental_table": fundamental_table,
        "pool_filter_fields": pool_fields,
        "checks": [],
    }
    if backend == "sqlite":
        raise DataPreflightError("SQLite backend is no longer supported; configure duckdb_path")
    if backend not in {"auto", "duckdb"}:
        raise DataPreflightError("db_backend must be 'duckdb' or 'auto'")
    if not duckdb_path:
        raise DataPreflightError("duckdb_path is required")
    if not contract and not Path(duckdb_path).expanduser().is_file():
        raise DataPreflightError(
            f"duckdb_path not reachable: {duckdb_path} (is the data volume mounted?)"
        )
    summary["backend"] = "duckdb"
    summary["checks"].append("frozen_binding" if contract else "duckdb_path_exists")
    if duckdb is None:
        raise DataPreflightError("duckdb module is required")
    try:
        conn = _connect_data_read_only(duckdb_path)
    except Exception as exc:
        raise DataPreflightError(f"duckdb connect failed for {duckdb_path}: {exc}") from exc
    try:
        price_columns = _probe_table(conn, price_table)
        summary["checks"].append(f"price_table_probe:{price_table}")
        fundamental_columns = _probe_table(conn, fundamental_table)
        summary["checks"].append(f"fundamental_table_probe:{fundamental_table}")
        required = [field for field in pool_fields if field not in _NON_FUNDAMENTAL_POOL_FIELDS]
        resolvable = set(fundamental_columns) | set(price_columns)
        missing = sorted(set(required) - resolvable)
        if missing:
            raise DataPreflightError(
                "weekly pool filter fields missing from "
                f"{fundamental_table!r} and {price_table!r}: {missing}"
            )
        if required:
            summary["checks"].append("pool_filter_fields_present")
        if "index_code" in pool_fields:
            _probe_table(conn, "index_constituents")
            summary["checks"].append("index_constituents_probe")
    finally:
        conn.close()
    return summary
