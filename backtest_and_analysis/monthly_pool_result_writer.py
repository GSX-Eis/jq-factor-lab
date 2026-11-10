"""DuckDB-only monthly pool result writer."""

from __future__ import annotations

from typing import Any, Optional

from backtest_and_analysis.result_writer import _require_duckdb_path
from utils.duckdb_results_writer import DuckDBResultsWriter


class MonthlyPoolResultWriter:
    """Persist monthly pool metrics in DuckDB without a fallback backend."""

    def __init__(
        self,
        db_conn_str: str | None = None,
        output_table: str = "monthly_pool_results",
        db_backend: str = "duckdb",
        duckdb_path: Optional[str] = None,
        parquet_dir: Optional[str] = None,
    ) -> None:
        path = _require_duckdb_path(db_backend, duckdb_path, db_conn_str)
        self.db_backend = "duckdb"
        self.duckdb_path = path
        self.parquet_dir = parquet_dir
        self.output_table = output_table
        self._duckdb_writer = DuckDBResultsWriter(
            duckdb_path=path,
            parquet_dir=parquet_dir,
            monthly_output_table=output_table,
            export_parquet=False,
        )

    @property
    def using_duckdb(self) -> bool:
        return True

    def write(
        self,
        factor_name: str,
        results: dict[str, Any],
        *,
        duckdb_conn: Any = None,
        allow_fallback: Optional[bool] = None,
    ) -> int:
        return self._duckdb_writer.write_monthly_pool_results(
            factor_name, results, duckdb_conn=duckdb_conn, allow_fallback=allow_fallback
        )
