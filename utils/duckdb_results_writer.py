#!/usr/bin/env python3
"""
duckdb_results_writer.py - DuckDB adapter for writing factor analysis results

This module provides high-performance writing of factor analysis results to DuckDB,
with optional Parquet export for factor_values. It mirrors the functionality of
ResultWriter and MonthlyPoolResultWriter while providing better write performance.

Key Features:
    - Write factor_analysis_results, factor_values, factor_monthly_results and
      monthly_pool_results to DuckDB
    - Parquet export with year/month partitioning for factor_values
    - Thread-safe operations using DuckDB's built-in concurrency

Usage:
    from utils.duckdb_results_writer import DuckDBResultsWriter
    
    writer = DuckDBResultsWriter(
        duckdb_path='./analysis_results.duckdb',
        parquet_dir='./analysis_results_parquet/'
    )
    
    writer.write_factor_analysis_results(factor_name, results)
    writer.save_factor_values(factor_df, factor_name)
"""

import os
import time
import fcntl
import shutil
import logging
import re
import threading
from numbers import Real
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Iterator
from backtest_and_analysis.forward_returns_calculator import newey_west_mean_test

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

logger = logging.getLogger(__name__)


class CommittedParquetExportError(RuntimeError):
    """DuckDB committed successfully, but its subsequent Parquet export failed."""

    def __init__(self, factor_name: str):
        self.factor_name = factor_name
        self.database_committed = True
        super().__init__(
            f"DuckDB data for {factor_name!r} is already committed; Parquet export failed. "
            "Do not assume rollback or use the old Parquet as current data. "
            "Retry only the export using the current committed factor snapshot."
        )


def _is_duckdb_lock_conflict(exc: Exception) -> bool:
    message = str(exc)
    return "Could not set lock on file" in message or "Conflicting lock is held" in message


def connect_read_only_with_retry(
    duckdb_path: Union[str, Path],
    retry_interval_sec: float = 1,
    *,
    bindings: Optional[dict] = None,
):
    """Open a read-only DuckDB connection after transient writers release the file."""
    while True:
        try:
            from utils.data_store import connect_read_only
            if bindings is None:
                return connect_read_only(duckdb_path)
            return connect_read_only(duckdb_path, bindings=bindings)
        except Exception as exc:
            if not _is_duckdb_lock_conflict(exc):
                raise
            logger.warning("DuckDB read blocked by writer; retrying: %s", duckdb_path)
            time.sleep(retry_interval_sec)


@dataclass(frozen=True)
class DuckDBWriteSession:
    conn: Any
    retry_count: int
    wait_seconds: float
    lock_path: Path
    last_error: str = ""


@contextmanager
def acquire_exclusive_write_conn(
    duckdb_path: Union[str, Path],
    retry_interval_sec: float = 1,
) -> Iterator[DuckDBWriteSession]:
    """Acquire an inter-process write lock and an exclusive DuckDB write connection."""
    if not DUCKDB_AVAILABLE:
        raise ImportError("duckdb is required. Install with: pip install duckdb")

    db_path = Path(duckdb_path).expanduser().resolve()
    # DuckDB may load extensions while probing an existing file.  Reject legacy
    # SQLite stores before opening the path so a write operation cannot treat
    # one as a candidate DuckDB database.
    from utils.data_store import _reject_sqlite_file
    _reject_sqlite_file(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{db_path}.write.lock")
    started = time.monotonic()
    retry_count = 0
    last_error = ""
    conn = None

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                retry_count += 1
                time.sleep(retry_interval_sec)

        try:
            while True:
                try:
                    conn = duckdb.connect(str(db_path), read_only=False)
                    break
                except Exception as exc:
                    if not _is_duckdb_lock_conflict(exc):
                        raise
                    retry_count += 1
                    last_error = str(exc)
                    time.sleep(retry_interval_sec)

            conn.execute("SET memory_limit='3GB'")
            conn.execute("SET threads=2")
            yield DuckDBWriteSession(
                conn=conn,
                retry_count=retry_count,
                wait_seconds=round(time.monotonic() - started, 3),
                lock_path=lock_path,
                last_error=last_error,
            )
        finally:
            if conn is not None:
                conn.close()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class DuckDBResultsWriter:
    """
    High-performance results writer using DuckDB.
    
    This class provides optimized writing of factor analysis results to DuckDB,
    with optional Parquet export for factor_values.
    """
    
    # Class-level lock for thread safety
    _write_lock = threading.Lock()
    
    def __init__(
        self,
        duckdb_path: str,
        parquet_dir: Optional[str] = None,
        output_table: str = 'factor_analysis_results',
        factor_values_table: str = 'factor_values',
        monthly_factor_output_table: Optional[str] = 'factor_monthly_results',
        monthly_output_table: Optional[str] = None,
        export_parquet: bool = True,
        publication_metadata: Optional[dict] = None,
        data_bindings: Optional[dict] = None,
    ):
        """
        Initialize the DuckDB results writer.
        
        Args:
            duckdb_path: Path to DuckDB database file
            parquet_dir: Directory for Parquet export (optional)
            output_table: Name of factor analysis results table
            factor_values_table: Name of factor values table
            monthly_factor_output_table: Name of base-name monthly result table
            monthly_output_table: Optional name of monthly pool results table. It
                is created only for an explicit pool writer, not every result store.
            export_parquet: Whether to export factor_values to Parquet
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("duckdb is required. Install with: pip install duckdb")
        
        from utils.data_store import binding_contract, output_context

        self.publication_metadata = output_context(publication_metadata)
        self.duckdb_path = Path(binding_contract(data_bindings).get("output_db", duckdb_path) if self.publication_metadata else duckdb_path)
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        self.output_table = output_table
        self.factor_values_table = factor_values_table
        self.monthly_factor_output_table = monthly_factor_output_table
        self.monthly_output_table = monthly_output_table
        self.export_parquet = export_parquet and ARROW_AVAILABLE
        
        # Ensure directories exist
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        if self.parquet_dir:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and create tables
        if not self.publication_metadata:
            self._init_database()
        
    def _publish_frame(self, frame, table, factor_name, conn=None):
        from utils.data_store import _catalog, publish_relation, write_store

        frame = frame.copy()
        if "created_at" in frame:
            frame["created_at"] = pd.Timestamp(self.publication_metadata["created_at"]).tz_localize(None)
        metadata = {**self.publication_metadata, "factor_name": factor_name, "schema_version": "factor_results.v1"}

        def publish(connection):
            _catalog(connection)
            connection.register("_factor_publication", frame)
            try:
                from utils.data_store import quote
                integer_columns = {"period", "quantile", "period_days", "ic_count", "stock_count"}
                text_columns = {"stock_code", "factor_name", "code", "pool_type", "year_month", "raw_factor_name", "pool_id", "run_id"}
                projections = []
                for column in frame:
                    dtype = ("DATE" if column == "date" else "TIMESTAMP" if column == "created_at"
                             else "INTEGER" if column in integer_columns else "VARCHAR" if column in text_columns
                             else "DOUBLE")
                    projections.append(f"CAST({quote(column)} AS {dtype}) AS {quote(column)}")
                connection.execute("CREATE OR REPLACE TEMP VIEW _typed_factor_publication AS SELECT "
                                   + ", ".join(projections) + " FROM _factor_publication")
                publish_relation(connection, "_typed_factor_publication", table, metadata=metadata,
                                 access_class="feature" if table == self.factor_values_table else "evaluation")
                return len(frame)
            finally:
                connection.unregister("_factor_publication")

        if conn is not None:
            return publish(conn)
        with write_store(self.duckdb_path) as connection:
            return publish(connection)

    def _get_connection(self, read_only: bool = False):
        """Get a DuckDB connection"""
        from utils.data_store import _reject_sqlite_file
        _reject_sqlite_file(self.duckdb_path)
        return duckdb.connect(str(self.duckdb_path), read_only=read_only)

    def acquire_write_session(self, retry_interval_sec: float = 1):
        """Return a context manager that yields an exclusive write session."""
        return acquire_exclusive_write_conn(self.duckdb_path, retry_interval_sec=retry_interval_sec)

    def _count_factor_rows(self, conn, table_name: str, factor_name: str) -> int:
        """Count rows for one factor to force a small table-specific health probe."""
        return int(
            conn.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE factor_name = ?",
                [factor_name],
            ).fetchone()[0]
        )

    def _factor_values_parquet_root(self) -> Optional[Path]:
        """Return the root directory containing factor_values parquet backups."""
        if not self.parquet_dir:
            return None
        return self.parquet_dir / "factor_values"

    def _list_visible_parquet_files(self, root: Path) -> List[Path]:
        """List parquet files while ignoring macOS metadata files like ._*.parquet."""
        files: List[Path] = []
        for path in root.rglob("*.parquet"):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            files.append(path)
        return sorted(files)

    def _create_factor_values_table(self, conn) -> None:
        """Create a clean factor_values table."""
        table_name = self.factor_values_table
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date DATE NOT NULL,
                stock_code VARCHAR NOT NULL,
                factor_name VARCHAR NOT NULL,
                factor_value DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, stock_code, factor_name)
            )
        """)



    
    def _init_database(self):
        """Initialize the result database schema without runtime migration side effects."""
        start_time = time.time()

        with self._write_lock:
            with self.acquire_write_session() as session:
                conn = session.conn

                if conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='artifacts'").fetchone()[0]:
                    raise ValueError("Versioned output writers require publication_metadata")
                existing_tables = set()
                try:
                    result = conn.execute("SHOW TABLES").fetchall()
                    existing_tables = {row[0] for row in result}
                    logger.debug(
                        "Found %d existing tables in %.2fs",
                        len(existing_tables),
                        time.time() - start_time,
                    )
                except Exception as exc:
                    logger.debug("SHOW TABLES failed during initialization: %s", exc)

                if self.output_table not in existing_tables:
                    conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.output_table} (
                            factor_name VARCHAR NOT NULL,
                            date DATE NOT NULL,
                            code VARCHAR NOT NULL,
                            period INTEGER NOT NULL,
                            quantile INTEGER,
                            ic DOUBLE,
                            ir DOUBLE,
                            return DOUBLE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (factor_name, date, code)
                        )
                    """)
                else:
                    col_result = conn.execute(f"PRAGMA table_info({self.output_table})").fetchall()
                    existing_cols = {row[1]: row for row in col_result}
                    if 'id' in existing_cols:
                        raise RuntimeError(
                            f"Table {self.output_table} still uses legacy schema with 'id' column. "
                            "Runtime auto-migration is disabled; migrate the table explicitly before writing."
                        )

                if self.factor_values_table not in existing_tables:
                    self._create_factor_values_table(conn)

                if self.monthly_factor_output_table and self.monthly_factor_output_table not in existing_tables:
                    conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.monthly_factor_output_table} (
                            factor_name VARCHAR NOT NULL,
                            year_month VARCHAR(7) NOT NULL,
                            period_days INTEGER NOT NULL,
                            monthly_ic DOUBLE,
                            monthly_ir DOUBLE,
                            ic_std DOUBLE,
                            ic_count INTEGER,
                            ic_t_stat DOUBLE,
                            ic_p_value DOUBLE,
                            monthly_return DOUBLE,
                            long_short_return DOUBLE,
                            stock_count INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (factor_name, year_month, period_days)
                        )
                    """)

                if self.monthly_output_table and self.monthly_output_table not in existing_tables:
                    conn.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.monthly_output_table} (
                            factor_name VARCHAR NOT NULL,
                            pool_type VARCHAR NOT NULL,
                            year_month VARCHAR(7) NOT NULL,
                            period_days INTEGER NOT NULL,
                            monthly_ic DOUBLE,
                            monthly_ir DOUBLE,
                            ic_std DOUBLE,
                            ic_count INTEGER,
                            ic_t_stat DOUBLE,
                            ic_p_value DOUBLE,
                            monthly_return DOUBLE,
                            long_short_return DOUBLE,
                            stock_count INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (factor_name, pool_type, year_month, period_days)
                        )
                    """)

                existing_indexes = set()
                try:
                    result = conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
                    existing_indexes = {row[0] for row in result}
                except Exception as exc:
                    logger.debug("duckdb_indexes() lookup failed during initialization: %s", exc)

                # factor_values indexes are intentionally disabled on the external
                # analysis-results store because reconnecting after index creation
                # repeatedly yields unreadable checksum blocks.
                index_defs = [
                    (f"idx_{self.output_table}_factor", self.output_table, "factor_name"),
                    (f"idx_{self.output_table}_date", self.output_table, "date"),
                ]

                for idx_name, table_name, cols in index_defs:
                    if idx_name in existing_indexes:
                        continue
                    try:
                        conn.execute(f"CREATE INDEX {idx_name} ON {table_name}({cols})")
                    except Exception as exc:
                        logger.debug("Index %s creation skipped: %s", idx_name, exc)

                elapsed = time.time() - start_time
                logger.info(
                    "DuckDB database initialized at %s (%.2fs, wait_seconds=%.3f, lock_retries=%d)",
                    self.duckdb_path,
                    elapsed,
                    session.wait_seconds,
                    session.retry_count,
                )
    
    def _parse_period(self, period_str) -> int:
        """Parse a positive trading-session count: integer, '5', or '5D'."""
        if isinstance(period_str, str) and re.fullmatch(r'[0-9]+D?', period_str):
            period_str = int(period_str.removesuffix('D'))
        if (
            isinstance(period_str, (bool, np.bool_))
            or not isinstance(period_str, (int, np.integer))
            or period_str <= 0
        ):
            raise ValueError(f"Invalid period {period_str!r}: expected a positive integer or label such as '5D'")
        return int(period_str)
    
    def _normalize_ic_data(self, ic_data: pd.DataFrame) -> pd.DataFrame:
        """Validate daily IC keys/values for both daily writes and monthly stats.

        Return a sorted copy with normalized local dates and integer periods.
        Missing IC values remain missing; conflicting keys are never collapsed.
        """
        if not isinstance(ic_data, pd.DataFrame):
            raise ValueError("IC data must be a DataFrame")
        frame = ic_data.copy()
        if frame.empty:
            return frame
        if not isinstance(frame.index, pd.DatetimeIndex):
            if isinstance(frame.index, pd.MultiIndex) or any(isinstance(v, Real) for v in frame.index):
                raise ValueError("IC index must contain daily dates")
            frame.index = pd.to_datetime(frame.index)
        dates = frame.index
        if dates.tz is not None or dates.hasnans or dates.has_duplicates or not dates.equals(dates.normalize()):
            raise ValueError("IC index must contain distinct normalized local dates")
        periods = [self._parse_period(col) for col in frame.columns]
        if len(set(periods)) != len(periods):
            raise ValueError("IC columns must identify distinct periods")
        for col in frame.columns:
            observed = frame[col].dropna()
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                   or not np.isfinite(v) or not -1 <= v <= 1 for v in observed):
                raise ValueError(f"Invalid IC values in {col!r}: expected finite real values in [-1, 1] or missing")
            frame[col] = frame[col].where(frame[col].notna(), np.nan).astype(float)
        frame.columns = periods
        return frame.sort_index()

    def _prepare_factor_values_frame(self, factor_df: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """Validate daily factor keys/values and return nonmissing DOUBLE rows.

        Empty snapshots are rejected, never interpreted as a successful clear.
        Missing cells are omitted; all observed values must be finite real numbers.
        """
        if (not isinstance(factor_name, str) or not factor_name.strip()
                or factor_name.startswith('.') or any(c in factor_name for c in ('/', '\\', '\x00'))):
            raise ValueError("factor_name must be a nonempty, visible single path component")
        if not isinstance(factor_df, pd.DataFrame):
            raise ValueError("Factor values must be a DataFrame")
        if isinstance(factor_df.index, pd.MultiIndex) or any(isinstance(v, Real) for v in factor_df.index):
            raise ValueError("Factor dates must be daily dates, not numeric or multi-level keys")
        dates = pd.DatetimeIndex(pd.to_datetime(factor_df.index))
        if (dates.tz is not None or dates.hasnans or dates.has_duplicates
                or not dates.equals(dates.normalize())):
            raise ValueError("Factor dates must be unique normalized local dates")
        if (isinstance(factor_df.columns, pd.MultiIndex) or not factor_df.columns.is_unique
                or any(not isinstance(v, str) or not v.strip() for v in factor_df.columns)):
            raise ValueError("Stock codes must be unique nonempty strings")
        for column in factor_df.columns:
            observed = factor_df[column].dropna()
            if pd.api.types.is_bool_dtype(observed.dtype) or pd.api.types.is_complex_dtype(observed.dtype):
                raise ValueError("Factor values must be finite real numbers or missing")
            if not pd.api.types.is_numeric_dtype(observed.dtype) and any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in observed
            ):
                raise ValueError("Factor values must be finite real numbers or missing")
        values = factor_df.where(factor_df.notna(), np.nan).to_numpy(dtype=float, na_value=np.nan).ravel(order='F')
        if np.isinf(values).any():
            raise ValueError("Factor values must be finite real numbers or missing")
        present = ~np.isnan(values)
        if not present.any():
            raise ValueError("Factor snapshot requires at least one finite factor value; existing data was not replaced")
        # Column-major order matches melt, without collisions with index/column names.
        factor_long = pd.DataFrame({
            'date': np.tile(dates.date, len(factor_df.columns))[present],
            'stock_code': np.repeat(factor_df.columns.to_numpy(), len(dates))[present],
            'factor_name': factor_name,
            'factor_value': values[present],
        })
        factor_long['created_at'] = datetime.now()
        return factor_long[['date', 'stock_code', 'factor_name', 'factor_value', 'created_at']]

    def export_factor_values_to_parquet(
        self, factor_df: pd.DataFrame, factor_name: str, *,
        write_session: Optional[DuckDBWriteSession] = None,
    ) -> int:
        """Export under the database writer lock; disabled export returns 0.

        For ordinary tables with saved factor rows, reject snapshots differing
        from the current database (ignoring created_at). Export-only factors
        without saved rows remain supported. Versioned stores have no implicit
        latest version: the caller explicitly selects the snapshot to export.
        A supplied session must belong to this DB and its transaction must have
        committed. Its owner retains the lock and reports the commit state.
        Empty/all-missing snapshots raise; no database rows are changed here.
        """
        if not self.export_parquet or not self.parquet_dir:
            return 0

        factor_long = self._prepare_factor_values_frame(factor_df, factor_name)

        def export(session):
            expected_lock = Path(f"{self.duckdb_path.expanduser().resolve()}.write.lock")
            if session.lock_path != expected_lock:
                raise ValueError("Write session belongs to another database")
            conn = session.conn
            # Separate autocommit statements have different transaction IDs.
            # Never publish files from rows that their owner could still roll back.
            first_tx = conn.execute("SELECT current_transaction_id()").fetchone()[0]
            if conn.execute("SELECT current_transaction_id()").fetchone()[0] == first_tx:
                raise ValueError("Commit the DuckDB transaction before exporting Parquet")
            if not self.publication_metadata and self._count_factor_rows(conn, self.factor_values_table, factor_name):
                conn.register('_factor_export_snapshot', factor_long)
                try:
                    differs = conn.execute(f"""
                        WITH current_values AS (
                            SELECT date, stock_code, factor_value FROM {self.factor_values_table}
                            WHERE factor_name = ?
                        ), requested AS (
                            SELECT CAST(date AS DATE) AS date, stock_code, factor_value
                            FROM _factor_export_snapshot
                        )
                        SELECT EXISTS (
                            (SELECT * FROM current_values EXCEPT ALL SELECT * FROM requested)
                            UNION ALL
                            (SELECT * FROM requested EXCEPT ALL SELECT * FROM current_values)
                        )
                    """, [factor_name]).fetchone()[0]
                    if differs:
                        raise ValueError("Parquet snapshot differs from current DuckDB factor values; reload the committed snapshot")
                finally:
                    conn.unregister('_factor_export_snapshot')
            self._export_factor_values_to_parquet(factor_long, factor_name)
            return len(factor_long)

        if write_session is not None:
            return export(write_session)
        with self.acquire_write_session() as session:
            return export(session)
    
    def write_factor_analysis_results(
        self,
        factor_name: str,
        results: Dict[str, Any],
        *,
        duckdb_conn: Any = None,
        allow_fallback: Optional[bool] = None,
        date_range: Optional[tuple[str, str]] = None,
    ) -> int:
        """
        Write factor analysis results to DuckDB.
        
        Optimized for large datasets with:
        - Vectorized operations instead of nested loops
        - DataFrame merge/join instead of iterrows()
        - DuckDB register() for efficient bulk insert
        
        Args:
            factor_name: Name of the factor
            results: Dictionary containing IC, IR, quantile_returns data
        """
        import time
        start_time = time.time()
        
        _ = allow_fallback
        range_start = range_end = None
        if date_range is not None:
            if not isinstance(date_range, (tuple, list)) or len(date_range) != 2 or any(isinstance(v, Real) for v in date_range):
                raise ValueError("date_range must contain two daily dates")
            bounds = pd.DatetimeIndex(pd.to_datetime(date_range))
            if bounds.tz is not None or bounds.hasnans or not bounds.equals(bounds.normalize()):
                raise ValueError("date_range must contain normalized local dates")
            range_start, range_end = bounds.date
            if range_start > range_end:
                raise ValueError("date_range start must not be after end")
        logger.info(f"Writing DuckDB results for factor: {factor_name}")
        
        # Process IC data
        t1 = time.time()
        ic_df = results.get('ic')
        if ic_df is None:
            logger.warning(f"No IC data found for factor {factor_name}")
            return 0
        ic_df = self._normalize_ic_data(ic_df)
        if ic_df.empty:
            return 0
        ic_periods = set(ic_df.columns)
        ic_df = ic_df.rename_axis('date').reset_index()

        # Melt to long format
        ic_long = ic_df.melt(
            id_vars=['date'],
            var_name='period',
            value_name='ic'
        )
        logger.debug(f"  IC processing took {time.time()-t1:.2f}s, shape: {ic_long.shape}")
        
        # Process quantile returns
        t2 = time.time()
        qr_df = results.get('quantile_returns', pd.DataFrame())
        qr_long = self._process_quantile_returns(qr_df).dropna(subset=['return'])
        logger.debug(f"  Quantile returns processing took {time.time()-t2:.2f}s, shape: {qr_long.shape}")
        
        # Get IR data (vectorized)
        t3 = time.time()
        ir_data = {}
        ir_df = results.get('ir')
        if ir_df is not None:
            if not isinstance(ir_df, pd.DataFrame):
                raise ValueError("IR data must be a DataFrame")
            if not ir_df.empty:
                if ir_df.columns.has_duplicates or 'ir' not in ir_df.columns:
                    raise ValueError("IR data requires an unambiguous 'ir' column")
                ir_periods = [self._parse_period(period) for period in ir_df.index]
                if len(set(ir_periods)) != len(ir_periods) or not set(ir_periods).issubset(ic_periods):
                    raise ValueError("IR periods must be distinct and present in IC data")
                observed = ir_df['ir'].dropna()
                if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                       or not np.isfinite(v) for v in observed):
                    raise ValueError("IR values must be finite real numbers or missing")
                ir_data = {period: float(value) if pd.notna(value) else None
                           for period, value in zip(ir_periods, ir_df['ir'])}
        logger.debug(f"  IR processing took {time.time()-t3:.2f}s")
        
        # 🚀 OPTIMIZED: Use vectorized operations instead of nested loops
        t4 = time.time()
        now = datetime.now()
        
        # Step 1: Create aggregate records using groupby (vectorized)
        # This replaces the nested loop: for date in dates: for period in periods
        ic_long_clean = ic_long.dropna(subset=['ic']).copy()
        ic_long_clean['ic'] = ic_long_clean['ic'].astype(float)
        
        # Group by date and period, then merge with IR data
        aggregate_records = []
        
        if not ic_long_clean.empty:
            # Keys are already unique; do not silently discard conflicting ICs.
            grouped = ic_long_clean.sort_values(['date', 'period']).reset_index(drop=True)
            
            # Add IR values (vectorized merge)
            grouped['ir'] = grouped['period'].map(ir_data)
            grouped['ir'] = grouped['ir'].where(grouped['ir'].notna() & ~pd.isna(grouped['ir']), None)
            
            # Calculate average returns using groupby (vectorized)
            if not qr_long.empty:
                qr_agg = qr_long.groupby(['date', 'period'])['return'].mean().reset_index()
                qr_agg = qr_agg.rename(columns={'return': 'avg_return'})
                grouped = grouped.merge(qr_agg, on=['date', 'period'], how='left')
            else:
                grouped['avg_return'] = None
            
            # Build aggregate records (vectorized)
            aggregate_records = grouped.to_dict('records')
            for rec in aggregate_records:
                rec['factor_name'] = factor_name
                rec['code'] = f"{factor_name}_{rec['date'].strftime('%Y%m%d')}_{rec['period']}D"
                rec['quantile'] = None
                rec['return'] = rec.pop('avg_return', None)
                rec['created_at'] = now
                rec['ic'] = float(rec['ic']) if rec['ic'] is not None else None
                rec['ir'] = float(rec['ir']) if rec['ir'] is not None else None
        
        # Step 2: Create quantile records using merge (vectorized, replaces iterrows)
        quantile_records = []
        if not qr_long.empty:
            qr_with_ic = qr_long.copy()

            # Add IR values (vectorized)
            qr_with_ic['ic'] = None
            qr_with_ic['ir'] = None
            
            # Build quantile records (vectorized)
            quantile_records = qr_with_ic.to_dict('records')
            for rec in quantile_records:
                rec['factor_name'] = factor_name
                rec['code'] = f"{factor_name}_{rec['date'].strftime('%Y%m%d')}_{rec['period']}D_Q{rec['quantile']}"
                rec['quantile'] = int(rec['quantile']) if not pd.isna(rec['quantile']) else None
                rec['ic'] = float(rec['ic']) if rec['ic'] is not None and not pd.isna(rec['ic']) else None
                rec['ir'] = float(rec['ir']) if rec['ir'] is not None else None
                rec['return'] = float(rec['return']) if rec['return'] is not None and not pd.isna(rec['return']) else None
                rec['created_at'] = now
        
        # Combine all records
        result_rows = aggregate_records + quantile_records
        logger.debug(f"  Record building took {time.time()-t4:.2f}s, total records: {len(result_rows)}")
        
        if not result_rows:
            logger.warning(f"No data to write for factor {factor_name}")
            return 0
        
        # Convert to DataFrame and write to DuckDB
        t5 = time.time()
        result_df = pd.DataFrame(result_rows)
        
        # Convert date to proper format (vectorized)
        result_df['date'] = pd.to_datetime(result_df['date']).dt.date
        if date_range is not None and (
            (result_df['date'] < range_start).any() or (result_df['date'] > range_end).any()
        ):
            raise ValueError("analysis result contains dates outside requested date range")
        
        # Ensure proper column order
        expected_cols = ['factor_name', 'date', 'code', 'period', 'quantile', 'ic', 'ir', 'return', 'created_at']
        result_df = result_df[[col for col in expected_cols if col in result_df.columns]]
        logger.debug(f"  DataFrame conversion took {time.time()-t5:.2f}s")
        
        if self.publication_metadata:
            return self._publish_frame(result_df, self.output_table, factor_name, duckdb_conn)

        # Write to DuckDB.
        t6 = time.time()
        def _write_with_conn(conn: Any) -> int:
            self._count_factor_rows(conn, self.output_table, factor_name)
            conn.register('result_df', result_df)
            try:
                t_delete = time.time()
                if date_range is None:
                    conn.execute(f"DELETE FROM {self.output_table} WHERE factor_name = ?", [factor_name])
                else:
                    conn.execute(
                        f"DELETE FROM {self.output_table} WHERE factor_name = ? AND date BETWEEN ? AND ?",
                        [factor_name, range_start, range_end],
                    )
                logger.debug(f"  DELETE took {time.time()-t_delete:.2f}s")

                t_insert = time.time()
                conn.execute(f"""
                    INSERT INTO {self.output_table}
                    (factor_name, date, code, period, quantile, ic, ir, return, created_at)
                    SELECT * FROM result_df
                """)
                logger.debug(f"  INSERT took {time.time()-t_insert:.2f}s")
                logger.debug(f"  Total DuckDB write took {time.time()-t6:.2f}s")
                total_time = time.time() - start_time
                logger.info(f"✅ Wrote {len(result_df):,} IC/IR records for {factor_name} in {total_time:.2f}s")
                return len(result_df)
            finally:
                try:
                    conn.unregister('result_df')
                except Exception:
                    pass

        if duckdb_conn is not None:
            return _write_with_conn(duckdb_conn)

        with self._write_lock:
            with self.acquire_write_session() as session:
                conn = session.conn
                conn.execute("BEGIN TRANSACTION")
                try:
                    row_count = _write_with_conn(conn)
                    conn.execute("COMMIT")
                    return row_count
                except Exception as e:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    logger.error(f"DuckDB write failed: {e}")
                    raise e
    
    def _process_quantile_returns(self, qr_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize long/wide quantile panels, retaining missing return cells.

        Return distinct (date, quantile, period) rows with integer keys. Daily
        writers may drop missing returns; monthly summaries need their groups.
        """
        columns = ['date', 'quantile', 'period', 'return']
        if not isinstance(qr_df, pd.DataFrame):
            raise ValueError("Quantile returns must be a DataFrame")
        if qr_df.empty:
            return pd.DataFrame(columns=columns)
        if qr_df.columns.has_duplicates:
            raise ValueError("Quantile return columns must be distinct")

        parts = []
        if isinstance(qr_df.columns, pd.MultiIndex):
            if qr_df.columns.nlevels != 2 or isinstance(qr_df.index, pd.MultiIndex):
                raise ValueError("Wide quantile returns require date rows and (period, quantile) columns")
            for period, quantile in qr_df.columns:
                parts.append(pd.DataFrame({
                    'date': qr_df.index, 'quantile': quantile,
                    'period': self._parse_period(period), 'return': qr_df[(period, quantile)].to_numpy(),
                }))
        else:
            frame = qr_df if {'date', 'factor_quantile'}.issubset(qr_df.columns) else qr_df.reset_index()
            if not {'date', 'factor_quantile'}.issubset(frame.columns):
                raise ValueError("Long quantile returns require date and factor_quantile keys")
            for period in frame.columns:
                if period not in ('date', 'factor_quantile'):
                    part = frame[['date', 'factor_quantile', period]].rename(
                        columns={'factor_quantile': 'quantile', period: 'return'}).copy()
                    part['period'] = self._parse_period(period)
                    parts.append(part)
        if not parts:
            raise ValueError("Quantile returns require at least one period")
        result = pd.concat(parts, ignore_index=True)[columns]
        if any(isinstance(v, Real) for v in result['date']):
            raise ValueError("Quantile return dates must be daily dates, not numbers")
        dates = pd.DatetimeIndex(pd.to_datetime(result['date']))
        if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
            raise ValueError("Quantile returns require normalized local dates")
        result['date'] = dates
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or v <= 0 for v in result['quantile']):
            raise ValueError("Quantiles must be positive integers")
        if result.duplicated(['date', 'quantile', 'period']).any():
            raise ValueError("Duplicate quantile return keys after period normalization")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
               or not np.isfinite(v) for v in result['return'].dropna()):
            raise ValueError("Quantile returns must be finite real values or missing")
        result['return'] = result['return'].where(result['return'].notna(), np.nan).astype(float)
        return result

    def save_factor_values(
        self,
        factor_df: pd.DataFrame,
        factor_name: str,
        skip_parquet: bool = False,
        *,
        duckdb_conn: Any = None,
        allow_fallback: Optional[bool] = None,
        date_range: Optional[tuple[str, str]] = None,
    ) -> int:
        """
        Save factor values to DuckDB and optionally export to Parquet.
        
        Validate before deleting/replacing rows. Empty/all-missing input raises.
        In ordinary table mode, own connections commit DuckDB first, then export.
        Export failure raises CommittedParquetExportError; it cannot roll back
        that commit. Borrowed connections leave commit/rollback and export to
        their owner. date_range replaces only that range and disables export.
        Versioned mode publishes the supplied snapshot via _publish_frame;
        it does not merge an earlier artifact or automatically export Parquet.
        
        Args:
            factor_df: DataFrame with index=dates, columns=stock_codes
            factor_name: Name of the factor
            skip_parquet: If True, skip Parquet export for faster writes
        """
        import time
        start_time = time.time()
        
        _ = allow_fallback
        # Validate the exact snapshot before inspecting ranges or changing storage.
        factor_long = self._prepare_factor_values_frame(factor_df, factor_name)
        range_start = range_end = None
        if date_range is not None:
            if not skip_parquet:
                raise ValueError("date-range factor writes do not support Parquet export")
            if not isinstance(date_range, (tuple, list)) or len(date_range) != 2 or any(isinstance(v, Real) for v in date_range):
                raise ValueError("date_range must contain two daily dates")
            bounds = pd.DatetimeIndex(pd.to_datetime(date_range))
            if bounds.tz is not None or bounds.hasnans or not bounds.equals(bounds.normalize()):
                raise ValueError("date_range must contain normalized local dates")
            range_start, range_end = bounds.date
            if range_start > range_end:
                raise ValueError("date_range start must not be after end")
            frame_dates = pd.DatetimeIndex(pd.to_datetime(factor_df.index))
            if (
                frame_dates.isna().any()
                or (frame_dates.date < range_start).any()
                or (frame_dates.date > range_end).any()
            ):
                raise ValueError(
                    f"factor frame contains dates outside requested date range "
                    f"[{range_start}, {range_end}]"
                )

        original_shape = factor_df.shape
        logger.info(f"Saving factor values for {factor_name} to DuckDB (shape: {original_shape})")
        
        if self.publication_metadata:
            return self._publish_frame(factor_long, self.factor_values_table, factor_name, duckdb_conn)

        def _write_with_conn(conn: Any) -> int:
            self._count_factor_rows(conn, self.factor_values_table, factor_name)
            conn.register('factor_long_df', factor_long)
            try:
                t_delete = time.time()
                if date_range is None:
                    conn.execute(f"""
                        DELETE FROM {self.factor_values_table}
                        WHERE factor_name = ?
                    """, [factor_name])
                else:
                    conn.execute(f"""
                        DELETE FROM {self.factor_values_table}
                        WHERE factor_name = ? AND date BETWEEN ? AND ?
                    """, [factor_name, range_start, range_end])
                logger.debug(f"  DELETE took {time.time()-t_delete:.2f}s")

                t_insert = time.time()
                conn.execute(f"""
                    INSERT INTO {self.factor_values_table}
                        (date, stock_code, factor_name, factor_value, created_at)
                    SELECT * FROM factor_long_df
                """)
                logger.debug(f"  INSERT took {time.time()-t_insert:.2f}s")
                total_time = time.time() - start_time
                logger.info(f"✅ Saved {len(factor_long):,} factor values for {factor_name} in {total_time:.2f}s")
                return len(factor_long)
            finally:
                try:
                    conn.unregister('factor_long_df')
                except Exception:
                    pass

        if duckdb_conn is not None:
            row_count = _write_with_conn(duckdb_conn)
            return row_count

        with self._write_lock:
            with self.acquire_write_session() as session:
                conn = session.conn
                conn.execute("BEGIN TRANSACTION")
                try:
                    row_count = _write_with_conn(conn)
                    conn.execute("COMMIT")
                except Exception as e:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    logger.error(f"DuckDB write failed: {e}")
                    raise e

                if self.export_parquet and self.parquet_dir and not skip_parquet:
                    t4 = time.time()
                    try:
                        self._export_factor_values_to_parquet(factor_long, factor_name)
                    except Exception as exc:
                        raise CommittedParquetExportError(factor_name) from exc
                    logger.debug(f"  Parquet export took {time.time()-t4:.2f}s")
                elif skip_parquet:
                    logger.debug("  Parquet export skipped (skip_parquet=True)")

        return row_count
    
    def _export_factor_values_to_parquet(self, factor_long: pd.DataFrame, factor_name: str):
        """Atomically replace one factor's complete partitioned Parquet export."""
        if not ARROW_AVAILABLE:
            raise RuntimeError("PyArrow is required for Parquet export")

        parquet_root = self._factor_values_parquet_root()
        if parquet_root is None:
            raise RuntimeError("Parquet output directory is not configured")
        parquet_root.mkdir(parents=True, exist_ok=True)
        with (parquet_root / f".{factor_name}.export.lock").open("a+") as export_lock:
            fcntl.flock(export_lock.fileno(), fcntl.LOCK_EX)
            factor_parquet_dir = parquet_root / factor_name
            token = f"{os.getpid()}-{time.time_ns()}"
            staging_dir = parquet_root / f".{factor_name}.staging-{token}"
            backup_dir = parquet_root / f".{factor_name}.backup-{token}"
            staging_dir.mkdir()

            try:
                factor_long = factor_long.copy()
                factor_long['date'] = pd.to_datetime(factor_long['date'])
                factor_long['year'] = factor_long['date'].dt.year
                factor_long['month'] = factor_long['date'].dt.month
                table = pa.Table.from_pandas(factor_long, preserve_index=False)
                pq.write_to_dataset(
                    table,
                    root_path=str(staging_dir),
                    partition_cols=['year', 'month'],
                    existing_data_behavior='overwrite_or_ignore',
                )

                for metadata_file in staging_dir.rglob("._*.parquet"):
                    metadata_file.unlink()

                files = self._list_visible_parquet_files(staging_dir)
                exported_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
                if not files or exported_rows != len(factor_long):
                    raise RuntimeError(
                        f"Parquet validation failed for {factor_name}: "
                        f"expected {len(factor_long):,} rows, found {exported_rows:,}"
                    )

                if factor_parquet_dir.exists():
                    factor_parquet_dir.rename(backup_dir)
                staging_dir.rename(factor_parquet_dir)
                logger.info(f"Exported factor values for {factor_name} to Parquet at {factor_parquet_dir}")
            except BaseException:
                if not factor_parquet_dir.exists() and backup_dir.exists():
                    backup_dir.rename(factor_parquet_dir)
                raise
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)

            if backup_dir.exists():
                shutil.rmtree(backup_dir)

    def write_monthly_pool_results(
        self,
        factor_name: str,
        results: Dict[str, Any],
        *,
        duckdb_conn: Any = None,
        allow_fallback: Optional[bool] = None,
    ) -> int:
        """
        Write monthly pool results to DuckDB.
        
        Args:
            factor_name: Name of the factor
            results: Dictionary containing monthly IC/IR metrics
        """
        _ = allow_fallback
        # Normal analysis/result stores do not create a legacy pool table at
        # construction.  Direct use of this explicit pool API creates it lazily.
        if self.monthly_output_table is None:
            self.monthly_output_table = 'monthly_pool_results'
            self._init_database()
        logger.info(f"Writing monthly pool results for factor: {factor_name}")
        
        pool_type = results.get('pool_type', 'unknown')
        factor_stats = results.get('factor_statistics', {})
        stock_count = factor_stats.get('stock_count', 0)
        
        ic_data = results.get('ic')
        if ic_data is None:
            logger.warning(f"No IC data found for factor {factor_name}")
            return 0
        
        # Calculate monthly metrics
        monthly_metrics = self._calculate_monthly_metrics(ic_data)
        
        if monthly_metrics.empty:
            logger.warning(f"No monthly metrics calculated for factor {factor_name}")
            return 0
        
        # Add metadata
        monthly_metrics['factor_name'] = factor_name
        monthly_metrics['pool_type'] = pool_type
        monthly_metrics['stock_count'] = stock_count
        monthly_metrics['created_at'] = datetime.now()
        
        # Calculate monthly returns
        quantile_returns = results.get('quantile_returns', pd.DataFrame())
        monthly_returns_dict = self._calculate_monthly_returns(quantile_returns)
        
        # Add return data
        for idx, row in monthly_metrics.iterrows():
            year_month = row['year_month']
            period_days = row['period_days']
            
            period_key = f"{period_days}D"
            if period_key in monthly_returns_dict:
                returns_data = monthly_returns_dict[period_key]
                if year_month in returns_data.get('monthly_return', {}).index:
                    monthly_metrics.loc[idx, 'monthly_return'] = returns_data['monthly_return'][year_month]
                    monthly_metrics.loc[idx, 'long_short_return'] = returns_data['long_short_return'][year_month]
        
        # Ensure proper column order
        columns = ['factor_name', 'pool_type', 'year_month', 'period_days', 
                   'monthly_ic', 'monthly_ir', 'ic_std', 'ic_count',
                   'ic_t_stat', 'ic_p_value', 'monthly_return', 
                   'long_short_return', 'stock_count', 'created_at']
        
        for col in columns:
            if col not in monthly_metrics.columns:
                monthly_metrics[col] = None
        
        monthly_metrics = monthly_metrics[columns]
        if self.publication_metadata:
            return self._publish_frame(monthly_metrics, self.monthly_output_table, factor_name, duckdb_conn)
        
        def _write_with_conn(conn: Any) -> int:
            self._count_factor_rows(conn, self.monthly_output_table, factor_name)
            conn.register('monthly_metrics_df', monthly_metrics)
            try:
                conn.execute(f"""
                    DELETE FROM {self.monthly_output_table}
                    WHERE factor_name = ?
                """, [factor_name])
                conn.execute(f"""
                    INSERT INTO {self.monthly_output_table} ({', '.join(columns)})
                    SELECT * FROM monthly_metrics_df
                """)
                logger.info(f"Successfully wrote {len(monthly_metrics)} monthly records for {factor_name}")
                return len(monthly_metrics)
            finally:
                try:
                    conn.unregister('monthly_metrics_df')
                except Exception:
                    pass

        if duckdb_conn is not None:
            return _write_with_conn(duckdb_conn)

        with self._write_lock:
            with self.acquire_write_session() as session:
                conn = session.conn
                conn.execute("BEGIN TRANSACTION")
                try:
                    row_count = _write_with_conn(conn)
                    conn.execute("COMMIT")
                    return row_count
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
    
    def _calculate_monthly_metrics(self, ic_data: pd.DataFrame) -> pd.DataFrame:
        """Summarize valid ICs; HAC requires a complete verified session sequence.

        Calendar snapshot comes from ic_data.attrs['trading_calendar']. Missing
        ICs/sessions or unavailable calendar coverage leave t/p unset. Coverage
        is checked within each month's supplied date span, not a full month.
        """
        from utils.trading_calendar import CalendarCoverageError, TradingCalendar

        ic_data = self._normalize_ic_data(ic_data)
        if ic_data.empty:
            return pd.DataFrame()
        calendar = ic_data.attrs.get('trading_calendar')
        if calendar is not None and not isinstance(calendar, TradingCalendar):
            raise ValueError("IC trading_calendar must be a TradingCalendar snapshot")
        results = []

        for period_days in ic_data.columns:
            ic_series = ic_data[period_days]
            monthly_groups = ic_series.groupby(ic_series.index.to_period('M'))

            for month, monthly_group in monthly_groups:
                group = monthly_group.dropna()
                if group.empty:
                    continue
                monthly_ic = group.mean()
                ic_count = len(group)
                is_constant = group.eq(group.iloc[0]).all()
                ic_std = 0.0 if ic_count > 1 and is_constant else group.std()
                monthly_ir = monthly_ic / ic_std if ic_std > 0 else None
                ic_t_stat = ic_p_value = None
                sessions = None
                if calendar is not None:
                    try:
                        sessions = calendar.sessions(monthly_group.index.min(), monthly_group.index.max())
                    except CalendarCoverageError:
                        pass  # Descriptive statistics remain available outside coverage.
                if sessions is not None and not monthly_group.index.isin(sessions).all():
                    raise ValueError("IC contains observations on non-session dates")
                if sessions is not None and group.index.equals(sessions):
                    ic_t_stat, ic_p_value = newey_west_mean_test(group.values, max_lag=period_days - 1)
                else:
                    logger.warning("IC HAC unavailable for %s/%s: missing observations or unverified calendar coverage", month, period_days)

                results.append({
                    'year_month': str(month),
                    'period_days': period_days,
                    'monthly_ic': monthly_ic,
                    'monthly_ir': monthly_ir,
                    'ic_std': ic_std,
                    'ic_count': ic_count,
                    'ic_t_stat': ic_t_stat,
                    'ic_p_value': ic_p_value
                })
        
        return pd.DataFrame(results)
    
    def _calculate_monthly_returns(self, quantile_returns: pd.DataFrame) -> Dict:
        """Month-wise means of forward returns, not compounded strategy returns.

        Keep the observed group universe per period. A missing group makes the
        equal-group mean unavailable; missing either extreme makes the spread
        unavailable. Groups absent from the entire input cannot be inferred.
        """
        frame = self._process_quantile_returns(quantile_returns)
        if frame.empty:
            return {}
        frame['year_month'] = frame['date'].dt.to_period('M').astype(str)
        monthly_returns = {}
        for period, group in frame.groupby('period'):
            period_returns = group.groupby(['year_month', 'quantile'])['return'].mean().unstack()
            if len(period_returns.columns) < 2:
                continue
            highest_q = period_returns.columns.max()
            lowest_q = period_returns.columns.min()
            monthly_returns[f"{period}D"] = {
                'monthly_return': period_returns.mean(axis=1, skipna=False),
                'long_short_return': period_returns[highest_q] - period_returns[lowest_q],
            }
        return monthly_returns

if __name__ == "__main__":
    # Test the writer
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DuckDB results writer")
    parser.add_argument("--db", default="./test_results.duckdb")
    parser.add_argument("--parquet", default="./test_parquet/")
    
    args = parser.parse_args()
    
    writer = DuckDBResultsWriter(
        duckdb_path=args.db,
        parquet_dir=args.parquet
    )
    
    # Create test data
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    factor_df = pd.DataFrame(
        np.random.randn(10, 5),
        index=dates,
        columns=['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    )
    
    writer.save_factor_values(factor_df, 'test_factor')
    print("Test completed successfully!")
