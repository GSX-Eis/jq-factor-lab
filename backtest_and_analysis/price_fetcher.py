"""DuckDB-only price loading for analysis and forward-return construction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from utils.duckdb_results_writer import connect_read_only_with_retry

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Read prices from DuckDB only; parquet_dir is retained but unused.

    duckdb_path is the sole database argument and accepts a filesystem path.
    data_config_path (or the bundled default) maps price aliases and key columns.
    Custom configs must declare price_aliases explicitly ({} for no aliases).
    Other table names select the matching configured dataset directly.
    """

    def __init__(
        self,
        duckdb_path: str | Path,
        price_table: str,
        *,
        parquet_dir: Optional[str] = None,
        data_config_path: Optional[str] = None,
    ) -> None:
        self._duckdb_conn = None
        del parquet_dir  # Forward-return prices retain their DuckDB-only contract.
        if (not isinstance(duckdb_path, (str, Path))
                or not str(duckdb_path).strip() or '://' in str(duckdb_path)):
            raise ValueError("duckdb_path must be a nonempty filesystem path, not a connection URI")
        self.duckdb_path = str(duckdb_path)
        from utils.data_access_config import load_data_access_config
        config = load_data_access_config(data_config_path)
        dataset, frequency = config.resolve_price(price_table)
        price_table = dataset.table
        self.date_column, self.stock_column = dataset.date_column, dataset.stock_column
        self.is_daily_table = frequency == 'daily'
        self.price_table = price_table
        from utils.data_store import table_name
        conn = connect_read_only_with_retry(self.duckdb_path)
        try:
            conn.execute(f'SELECT 1 FROM {table_name(price_table)} LIMIT 0')
        except Exception as exc:
            conn.close()
            raise ValueError(f"DuckDB price table is not readable: {price_table}") from exc
        self._duckdb_conn = conn

    def close(self) -> None:
        if getattr(self, "_duckdb_conn", None) is not None:
            self._duckdb_conn.close()
            self._duckdb_conn = None

    def __del__(self) -> None:
        self.close()

    def fetch(
        self,
        code_list: List[str],
        start_date: str,
        end_date: str,
        future_periods: int = 20,
    ) -> pd.Series:
        if not code_list:
            return pd.Series(dtype=float, name="close")
        from utils.trading_calendar import load_trading_calendar
        calendar = load_trading_calendar(self._duckdb_conn)
        calendar.validate_coverage(start_date, end_date)
        query_start = pd.Timestamp(start_date).normalize()
        future = calendar.next_sessions(end_date, future_periods)
        query_end = future[-1] if len(future) else pd.Timestamp(end_date).normalize()
        df = self._fetch_from_duckdb(code_list, query_start.date(), query_end.date())
        if df.empty:
            raise ValueError("No price data found for the specified parameters")
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        if self.is_daily_table:
            df["date"] = df["date"].dt.normalize()
            daily_prices = df.set_index(["stock_code", "date"])["close"]
        else:
            # Stable sort makes the last close deterministic if the source has duplicate timestamps.
            df = df.sort_values(["stock_code", "date"])
            daily_prices = df.groupby(["stock_code", df["date"].dt.normalize()])["close"].last()
        daily_prices.index = pd.MultiIndex.from_tuples(daily_prices.index.to_list(), names=["asset", "date"])
        daily_prices.attrs['trading_calendar'] = calendar
        return daily_prices

    def _fetch_from_duckdb(self, code_list: List[str], start_date, end_date) -> pd.DataFrame:
        if self._duckdb_conn is None:
            raise RuntimeError("PriceFetcher is closed")
        from utils.data_store import quote, table_name
        date, stock = quote(self.date_column), quote(self.stock_column)
        # CAST keeps timestamped intraday rows from losing the explicit end date.
        query = f"""
            SELECT {stock} AS stock_code, {date} AS date, open, high, low, close, volume, money
            FROM {table_name(self.price_table)}
            WHERE {stock} = ANY(?)
              AND CAST({date} AS DATE) BETWEEN ? AND ?
            ORDER BY stock_code, date
        """
        return self._duckdb_conn.execute(query, [code_list, start_date, end_date]).fetchdf()
