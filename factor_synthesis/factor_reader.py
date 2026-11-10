"""DuckDB factor-value reader and composite-factor writer."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.duckdb_results_writer import DuckDBResultsWriter, connect_read_only_with_retry
from utils.analysis_results_guard import _duckdb_tables


class FactorReader:
    """Read factor values and save composites in one DuckDB output store."""

    def __init__(
        self,
        database_path: str,
        results_database_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        data_bindings: Optional[dict] = None,
        publication_metadata: Optional[dict] = None,
    ) -> None:
        from utils.data_store import binding_contract, resolve_connection_binding

        self.data_bindings = binding_contract(data_bindings)
        if self.data_bindings and not self.data_bindings.get("output_db"):
            raise ValueError("Bound factor reads require output_db")
        self.publication_metadata = publication_metadata
        self.logger = logger or logging.getLogger(__name__)
        self.database_path = database_path
        self.results_database_path = results_database_path or database_path
        self._read_bindings: dict[bool, dict] = {}
        for results, path in ((False, self.database_path), (True, self.results_database_path)):
            selected = self.data_bindings
            if selected:
                requested = str(Path(path).expanduser().resolve())
                path, selected = resolve_connection_binding(requested, selected, allow_labels=True)
                if Path(path).resolve() != Path(self.data_bindings["output_db"]).resolve():
                    raise ValueError("FactorReader source must select the output store")
            if results:
                self.results_database_path = path
            else:
                self.database_path = path
            self._read_bindings[results] = selected

        # Reads must be side-effect free.  Create a writer only at an explicit
        # save operation, after the caller has chosen an output store.
        self._writer: Optional[DuckDBResultsWriter] = None
        self._test_connection()

    def _test_connection(self) -> None:
        with self.get_connection() as conn:
            tables = _duckdb_tables(conn)
            if "factor_values" not in tables:
                raise ValueError("DuckDB factor store has no factor_values table")

    @contextmanager
    def get_connection(self, results_db: bool = False):
        path = self.results_database_path if results_db else self.database_path
        bindings = self._read_bindings[results_db]
        if bindings:
            from utils.data_store import connect_output

            with connect_output(path, bindings=bindings, allow_labels=True) as conn:
                yield conn
            return
        conn = connect_read_only_with_retry(path, bindings=bindings)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _fetchdf(conn: Any, query: str, params: list[Any]) -> pd.DataFrame:
        return conn.execute(query, params).fetchdf()

    def get_available_factors(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[str]:
        clauses, params = ["1=1"], []
        if start_date:
            clauses.append("date >= ?"); params.append(start_date)
        if end_date:
            clauses.append("date <= ?"); params.append(end_date)
        with self.get_connection() as conn:
            frame = self._fetchdf(conn, "SELECT DISTINCT factor_name FROM factor_values WHERE " + " AND ".join(clauses) + " ORDER BY factor_name", params)
        return frame["factor_name"].tolist()

    def get_factor_values(
        self,
        factor_name: str,
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
        pivot_format: bool = True,
    ) -> pd.DataFrame:
        query = """
            SELECT date, stock_code, factor_value FROM factor_values
            WHERE factor_name = ? AND date >= ? AND date <= ?
        """
        params: list[Any] = [factor_name, start_date, end_date]
        if stock_pool:
            query += " AND stock_code = ANY(?)"; params.append(stock_pool)
        query += " ORDER BY date, stock_code"
        with self.get_connection() as conn:
            frame = self._fetchdf(conn, query, params)
        if frame.empty:
            return pd.DataFrame()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.pivot(index="date", columns="stock_code", values="factor_value") if pivot_format else frame

    def get_multiple_factors(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        if not factor_names:
            return {}
        query = """
            SELECT date, stock_code, factor_name, factor_value FROM factor_values
            WHERE factor_name = ANY(?) AND date >= ? AND date <= ?
        """
        params: list[Any] = [factor_names, start_date, end_date]
        if stock_pool:
            query += " AND stock_code = ANY(?)"; params.append(stock_pool)
        query += " ORDER BY factor_name, date, stock_code"
        with self.get_connection() as conn:
            frame = self._fetchdf(conn, query, params)
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"])
        return {
            name: (frame.loc[frame["factor_name"] == name].pivot(index="date", columns="stock_code", values="factor_value")
                   if not frame.empty and (frame["factor_name"] == name).any() else pd.DataFrame())
            for name in factor_names
        }

    def save_composite_factor(
        self,
        composite_factor_df: pd.DataFrame,
        factor_name: str,
        overwrite: bool = True,
        batch_size: int = 1000,
    ) -> bool:
        del batch_size
        if composite_factor_df.empty:
            return False
        if not overwrite:
            with self.get_connection(results_db=True) as conn:
                exists = conn.execute("SELECT 1 FROM factor_values WHERE factor_name = ? LIMIT 1", [factor_name]).fetchone()
            if exists:
                raise ValueError(f"Factor already exists and overwrite=False: {factor_name}")
        if self._writer is None:
            self._writer = DuckDBResultsWriter(
                self.results_database_path,
                publication_metadata=self.publication_metadata,
                data_bindings=self.data_bindings,
            )
        return self._writer.save_factor_values(composite_factor_df, factor_name) > 0

    def check_data_quality(self, factor_name: str, start_date: str, end_date: str, stock_pool: Optional[List[str]] = None) -> Dict[str, Any]:
        frame = self.get_factor_values(factor_name, start_date, end_date, stock_pool)
        if frame.empty:
            return {"error": "No data available"}
        values = frame.to_numpy().ravel(); valid = values[~pd.isna(values)]
        total = frame.size; non_null = int(frame.count().sum())
        mean, std = (float(np.mean(valid)), float(np.std(valid))) if len(valid) else (0.0, 0.0)
        outlier_ratio = float(np.mean(np.abs(valid - mean) > 3 * std)) if len(valid) and std else 0.0
        return {
            "data_shape": frame.shape, "total_cells": total, "non_null_cells": non_null,
            "null_cells": total - non_null, "coverage_ratio": non_null / total if total else 0,
            "outlier_ratio": outlier_ratio,
            "value_stats": {"mean": mean, "std": std, "min": float(np.min(valid)) if len(valid) else None, "max": float(np.max(valid)) if len(valid) else None},
        }
