"""DuckDB-only price loading for analysis and forward-return construction."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from utils.duckdb_results_writer import connect_read_only_with_retry

if TYPE_CHECKING:
    from utils.slots import MarketSlot, SlotConfig

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

    @cached_property
    def _analysis_slot(self) -> MarketSlot:
        from slots.jq_ashare_local import build_jq_ashare_local

        return build_jq_ashare_local(self._duckdb_conn)

    def daily_evaluation_slot(self, pricing: pd.Series, config: SlotConfig) -> MarketSlot:
        from dataclasses import replace

        from slots import build_slot_from_config
        from slots.jq_ashare_local.feed import AShareDailyEvaluationFeed

        slot = build_slot_from_config(config, connection=self._duckdb_conn)
        feed = AShareDailyEvaluationFeed(slot.bar_feed, pricing=pricing, source_table=self.price_table)
        return replace(slot, bar_feed=feed)

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

        slot = self._analysis_slot
        calendar = load_trading_calendar(self._duckdb_conn)
        calendar.validate_coverage(start_date, end_date)
        query_start = pd.Timestamp(start_date).normalize()
        future = calendar.next_sessions(end_date, future_periods)
        query_end = future[-1] if len(future) else pd.Timestamp(end_date).normalize()
        slot.calendar.validate_coverage(query_start, query_end)
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
        daily_prices.attrs['execution_profile'] = slot.microstructure.execution_profile
        return daily_prices

    def _fetch_from_duckdb(self, code_list: List[str], start_date, end_date) -> pd.DataFrame:
        if self._duckdb_conn is None:
            raise RuntimeError("PriceFetcher is closed")
        from slots.jq_ashare_local.feed import read_analysis_price_rows

        return read_analysis_price_rows(
            self._duckdb_conn, code_list, start_date, end_date,
            table=self.price_table, date_column=self.date_column, instrument_column=self.stock_column,
        )
