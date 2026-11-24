"""Write analysis metrics to DuckDB and factor snapshots to Scheme B."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

import pandas as pd

from utils.duckdb_results_writer import DuckDBResultsWriter
from utils.market_factor import FactorWriteReceipt, MarketFactorValueStore, save_factor_values

if TYPE_CHECKING:
    from utils.factor_base import FactorBase


def _require_duckdb_path(
    db_backend: str | None,
    duckdb_path: Optional[str],
    db_conn_str: Optional[str] = None,
) -> str:
    """Resolve DuckDB storage while rejecting removed SQLite connection strings."""
    if db_backend not in (None, "duckdb", "auto"):
        raise ValueError(
            "SQLite result storage is no longer supported; migrate the result store to DuckDB "
            "and pass duckdb_path."
        )
    if duckdb_path:
        return duckdb_path
    if db_conn_str and db_conn_str.startswith("duckdb:///"):
        return db_conn_str.removeprefix("duckdb:///")
    if db_conn_str and "://" not in db_conn_str:
        return db_conn_str
    if db_conn_str and db_conn_str.startswith("sqlite"):
        raise ValueError(
            "SQLite result storage is no longer supported; pass a DuckDB path or duckdb:/// URI."
        )
    if not duckdb_path:
        raise ValueError("DuckDB result storage requires duckdb_path")
    raise AssertionError("unreachable")


class ResultWriter:
    """Persist analysis metrics in DuckDB and factor values in Scheme B."""

    def __init__(
        self,
        db_conn_str: str | None = None,
        output_table: str = "factor_analysis_results",
        db_backend: str = "duckdb",
        duckdb_path: Optional[str] = None,
        parquet_dir: Optional[str] = None,
        export_parquet: bool = True,
        *,
        factor_values_store: str | Path | None = None,
    ) -> None:
        path = _require_duckdb_path(db_backend, duckdb_path, db_conn_str)
        self.db_backend = "duckdb"
        self.duckdb_path = path
        self.parquet_dir = parquet_dir
        self.export_parquet = export_parquet
        self.output_table = output_table
        self.factor_values_store = (
            MarketFactorValueStore(factor_values_store) if factor_values_store is not None
            else MarketFactorValueStore.beside_results_database(path)
        )
        self._duckdb_writer = DuckDBResultsWriter(
            duckdb_path=path,
            parquet_dir=parquet_dir,
            output_table=output_table,
            export_parquet=export_parquet,
        )

    @property
    def using_duckdb(self) -> bool:
        return True

    @contextmanager
    def duckdb_write_session(self, retry_interval_sec: int = 30) -> Iterator[Any]:
        with self._duckdb_writer.acquire_write_session(retry_interval_sec=retry_interval_sec) as session:
            yield session

    def write_market_factor_values(
        self, factor: FactorBase, factor_df: pd.DataFrame, factor_name: str,
    ) -> FactorWriteReceipt:
        return save_factor_values(factor, self.factor_values_store, factor_values=factor_df, factor_id=factor_name)

    def write(
        self,
        factor_name: str,
        results: dict[str, Any],
        *,
        duckdb_conn: Any = None,
        allow_fallback: Optional[bool] = None,
        date_range: Optional[tuple[str, str]] = None,
    ) -> int:
        return self._duckdb_writer.write_factor_analysis_results(
            factor_name, results, duckdb_conn=duckdb_conn, allow_fallback=allow_fallback,
            date_range=date_range,
        )

    def save_factor_values(
        self,
        factor_df: pd.DataFrame,
        factor_name: str,
        skip_parquet: bool = False,
        *,
        duckdb_conn: Any = None,
        allow_fallback: Optional[bool] = None,
        date_range: Optional[tuple[str, str]] = None,
        allow_legacy_factor_values_write: bool = False,
    ) -> int:
        return self._duckdb_writer.save_factor_values(
            factor_df,
            factor_name,
            skip_parquet=skip_parquet,
            duckdb_conn=duckdb_conn,
            allow_fallback=allow_fallback,
            date_range=date_range,
            allow_legacy_factor_values_write=allow_legacy_factor_values_write,
        )
