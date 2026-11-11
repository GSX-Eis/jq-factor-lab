"""
Factor Loader for VectorBT Backtest

Loads raw factor values from DuckDB and provides preprocessing functions
for cross-sectional normalization, matching JQ Local strategy logic.
"""

from utils.strategy_kernel import preprocess_series, bin_series
import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from utils.duckdb_results_writer import connect_read_only_with_retry

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None


def cross_sectional_raw_zscore(series):
    return preprocess_series(series, "raw_zscore")


def cross_sectional_zscore(series):
    return preprocess_series(series, "zscore")


def cross_sectional_rank_zscore(series):
    return preprocess_series(series, "rank_zscore")


def cross_sectional_binning(series, n_bins=5):
    return bin_series(series, n_bins)


def cross_sectional_winsorize_zscore(series, percentile=0.025):
    return preprocess_series(series, "winsorize_zscore", percentile=percentile)


from utils.data_revision import connection_revision, revision_cached


class FactorLoader:
    """
    Loads factor values from DuckDB database.

    Matches JQ Local: LocalFactorLoader.get_factor_data()
    """

    def __init__(
        self,
        duckdb_path: Optional[str] = None,
        factor_table: str = "factor_values",
        logger: Optional[logging.Logger] = None,
        existing_conn: Optional["duckdb.DuckDBPyConnection"] = None,
        shared_cache: Optional[Dict[tuple, pd.DataFrame]] = None,
    ):
        """
        Initialize factor loader.

        Args:
            duckdb_path: Path to DuckDB database file (can be None if existing_conn provided)
            factor_table: Name of factor values table
            logger: Optional logger instance
            existing_conn: Optional existing DuckDB connection to reuse
            shared_cache: Optional dict for memoising ``load_factors`` results across
                loader instances. When provided, identical (date, stock_list,
                factor_names) queries return a cached copy instead of re-hitting the
                database. Intended to be shared across a multi-trial backtest sweep
                (e.g. a top_pct grid) where the raw factor reads are identical for
                every trial. When None, caching is disabled (default).
        """
        self.duckdb_path = duckdb_path
        self.factor_table = factor_table
        self.logger = logger or logging.getLogger(__name__)

        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._owns_conn = True  # Whether we own the connection (should close it)
        # Cross-trial read cache (opt-in via shared_cache injection).
        self._load_cache: Optional[Dict[tuple, pd.DataFrame]] = shared_cache
        self._prefetched = {}
        self._prefetch_dates: Set[str] = set()
        self._prefetch_stocks: Set[str] = set()
        self._prefetch_factors: Set[str] = set()

        if existing_conn is not None:
            self._conn = existing_conn
            self._owns_conn = False  # Don't close external connection
        elif duckdb_path is None:
            raise ValueError("Either duckdb_path or existing_conn must be provided")

    def _get_conn(self) -> "duckdb.DuckDBPyConnection":
        """Get or create DuckDB connection (read-only)."""
        if self._conn is None:
            if not DUCKDB_AVAILABLE:
                raise ImportError(
                    "duckdb is required. Install with: pip install duckdb"
                )
            self._conn = connect_read_only_with_retry(self.duckdb_path)
            self._owns_conn = True
        return self._conn

    def cache_revision(self):
        conn = self._get_conn()
        revision = connection_revision(conn)
        # Independent in-memory databases cannot share a cache namespace.
        memory_id = id(conn) if any(not row[1] for row in revision) else None
        return self.factor_table, revision, memory_id

    def clear_cache(self):
        if self._load_cache is not None:
            self._load_cache.clear()
        self._prefetched.clear()
        self._prefetch_dates.clear()
        self._prefetch_stocks.clear()
        self._prefetch_factors.clear()

    def close(self):
        """Close connection if we own it."""
        if self._conn is not None and self._owns_conn:
            self._conn.close()
            self._conn = None

    @revision_cached
    def prefetch(self, dates: List[str], stock_list: List[str], factor_names: List[str]) -> None:
        """Read frozen raw observations once; preprocessing still uses each exact pool."""
        dates, stocks, factors = sorted(set(dates)), sorted(set(stock_list)), sorted(set(factor_names))
        if not dates or not stocks or not factors:
            return
        clauses = [
            f"{column} IN ({','.join('?' for _ in values)})"
            for column, values in [("date", dates), ("stock_code", stocks), ("factor_name", factors)]
        ]
        frame = self._get_conn().execute(
            f"SELECT date, stock_code, factor_name, factor_value FROM {self.factor_table} "
            f"WHERE {' AND '.join(clauses)}", dates + stocks + factors,
        ).fetchdf()
        groups = {
            date: group
            for date, group in frame.groupby(pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d'))
        }
        self._prefetched = groups
        self._prefetch_dates, self._prefetch_stocks, self._prefetch_factors = set(dates), set(stocks), set(factors)

    @revision_cached
    def load_factors(
        self, date_str: str, stock_list: List[str], factor_names: List[str]
    ) -> pd.DataFrame:
        """
        Load factor values for a specific date and stock list.

        Matches JQ Local: LocalFactorLoader.get_factor_data() with strict mode (no ffill).

        Args:
            date_str: Date string in 'YYYY-MM-DD' format
            stock_list: List of stock codes to query
            factor_names: List of factor names to fetch

        Returns:
            DataFrame with index=stock_code, columns=factor_names
            Missing values are NaN (not filled with 0 like JQ Local)
        """
        if not stock_list or not factor_names:
            return pd.DataFrame(index=stock_list, columns=factor_names)

        # Cross-trial memoisation: raw factor reads are identical across e.g. a
        # top_pct sweep. Key on the exact query inputs; return a defensive copy so
        # no caller can mutate the cached frame. Stock order is preserved in the
        # key (stock_list drives the final reindex), so we must not sort it.
        cache_key: Optional[tuple] = None
        if self._load_cache is not None:
            cache_key = (self._cache_revision, date_str, tuple(stock_list), tuple(factor_names))
            cached = self._load_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

        conn = self._get_conn()
        try:
            if (
                date_str in self._prefetch_dates
                and set(stock_list) <= self._prefetch_stocks
                and set(factor_names) <= self._prefetch_factors
            ):
                df = self._prefetched.get(date_str, pd.DataFrame())
                if not df.empty:
                    df = df[df['stock_code'].isin(stock_list) & df['factor_name'].isin(factor_names)]
            else:
                stocks_sql = ','.join('?' for _ in stock_list)
                factors_sql = ','.join('?' for _ in factor_names)
                df = conn.execute(
                    f"SELECT date, stock_code, factor_name, factor_value FROM {self.factor_table} "
                    f"WHERE date = ? AND factor_name IN ({factors_sql}) "
                    f"AND stock_code IN ({stocks_sql})",
                    [date_str, *factor_names, *stock_list],
                ).fetchdf()

            if df.empty:
                self.logger.debug(f"[FactorLoader] No data for {date_str}")
                empty = pd.DataFrame(index=stock_list, columns=factor_names)
                if cache_key is not None:
                    self._load_cache[cache_key] = empty.copy()
                return empty

            # Pivot to wide format: index=stock_code, columns=factor_name
            result = df.pivot(
                index="stock_code", columns="factor_name", values="factor_value"
            )

            # Ensure all requested factors are present
            for factor in factor_names:
                if factor not in result.columns:
                    result[factor] = np.nan

            # Reorder columns to match requested order
            result = result[factor_names]

            # Reindex to include all requested stocks (fill missing with NaN)
            result = result.reindex(stock_list)

            if cache_key is not None:
                self._load_cache[cache_key] = result.copy()

            return result

        except Exception as e:
            self.logger.error(f"[FactorLoader] Failed to load factors: {e}")
            # Query/reshape failures are not missing observations.
            raise
