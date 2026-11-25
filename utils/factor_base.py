import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pickle as pickle
import tempfile as tempfile
import logging
from typing import List, Optional, Callable, Dict, Any as Any, Union as Union
import time as time
import gc as gc

from utils.factor_core import (
    bar_bridge as _bar_bridge,
    cache as _cache,
    execution as _execution,
    legacy_storage as _legacy_storage,
    slot_binding as _slot_binding,
    slot_execution_mode as _slot_execution_mode,
    slot_fast_path as _slot_fast_path,
    slot_replay as _slot_replay,
)

try:
    from .dynamic_stock_pool import DynamicStockPoolManager, StockFilter
except ImportError:
    class DynamicStockPoolManager:
        def __init__(self, *args, **kwargs):
            pass
    class StockFilter:
        def __init__(self, *args, **kwargs):
            pass

# Local storage uses DuckDB; cloud bundles fetch data through the JoinQuant API.
try:
    from .duckdb_adapter import DuckDBAdapter as DuckDBAdapter, get_data_adapter
    DUCKDB_ADAPTER_AVAILABLE = True
except ImportError:
    DUCKDB_ADAPTER_AVAILABLE = False
    DuckDBAdapter = None
    get_data_adapter = None

from utils.pit import apply_pit_adjustment
from utils.slots import MarketSlot as _MarketSlot, SlotRegistry as _SlotRegistry


def _load_factor_requirements() -> dict:
    """Load factor requirements from the bundle or local configuration."""
    if "FACTOR_REQUIREMENTS" in globals():
        return FACTOR_REQUIREMENTS.get("factors", {})  # noqa: F821 - supplied by the bundle
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'config', 'factor_requirements.json'
    )
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return json.load(f).get('factors', {})


def _get_pit_window_from_config(factor_name: str, data_frequency: str = "daily") -> int:
    entry = _load_factor_requirements().get(factor_name)
    if not entry:
        raise ValueError("Missing requirements for factor: " + factor_name)
    requirements = entry.get("requirements", {})
    if data_frequency == "5m" and "5m" in requirements:
        return max(1, (requirements["5m"].get("min_bars", 240) + 47) // 48)
    if "daily" in requirements:
        return requirements["daily"].get("min_days", 120)
    if "5m" in requirements:
        return max(1, (requirements["5m"].get("min_bars", 240) + 47) // 48)
    raise ValueError("Missing daily or 5m requirements for factor: " + factor_name)


class FactorBase:
    """Base lifecycle and calculation contract for daily and Bar factors."""

    SLOT_EXECUTION_MODE: str = 'auto'
    CAUSAL_VECTORIZED: Optional[bool] = None
    INCREMENTAL_ENGINE = None

    def __init__(
        self,
        db_connection,
        factor_name: str,
        start_date: str,
        end_date: str,
        stock_pool: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        use_dynamic_pool: bool = False,
        stock_filters: Optional[List[StockFilter]] = None,
        max_stocks_per_month: Optional[int] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        rebalance_period_months: int = 1,
        rebalance_schedule: str = 'quarterly',
        monthly_stock_pools: Optional[Dict[str, List[str]]] = None,
        data_frequency: str = 'daily',
        fundamental_table: str = 'factor_data',
        price_table: str = 'price_data',
        dynamic_pool_mode: str = 'factor',
        db_backend: str = 'duckdb',
        duckdb_path: Optional[str] = None,
        parquet_dir: Optional[str] = None,
        use_duckdb_cache: bool = True,
        duckdb_cache_size_mb: int = 2000,
        pit_enabled: bool = True,  # 默认启用动态复权
        pit_factor_column: str = 'factor',  # 复权因子列名
        pit_window: int = None,  # PIT 调整窗口大小（交易日），None 则使用因子的 window 属性或默认 120
        data_bindings: Optional[dict] = None,
        stock_pools_by_period: Optional[Dict[str, List[str]]] = None,
        data_config_path: Optional[str] = None,
        source_policy: Optional[str] = None,
        calendar_duckdb_path: Optional[str] = None,
        pool_schedule: Optional[dict] = None,
        analysis_mode: Optional[str] = None,
        session_view: str = 'ETH',
        output_frequency: Optional[str] = None,
        label_config: Optional[dict] = None,
        data_version: Optional[str] = None,
        session_calendar=None,
        source_revision: Optional[str] = None,
        *,
        slot: _MarketSlot | None = None,
        slot_id: str | None = None,
        slot_registry: _SlotRegistry | None = None,
        slot_config_hash: str | None = None,
        universe_id: str | None = None,
        slot_execution_mode: Optional[str] = None,
    ):
        """Configure the factor, its input sources and stock universe."""
        if monthly_stock_pools is not None and stock_pools_by_period is not None:
            raise ValueError("Use only stock_pools_by_period or monthly_stock_pools, not both")
        if stock_pools_by_period is None:
            stock_pools_by_period = monthly_stock_pools
        if stock_pools_by_period is not None:
            FactorBase._parse_stock_pool_intervals(stock_pools_by_period)

        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("日期格式错误，请使用'YYYY-MM-DD'格式")

        self.logger = logging.getLogger(f"Factor.{factor_name}")

        if db_connection is not None:
            import duckdb
            if not isinstance(db_connection, duckdb.DuckDBPyConnection):
                raise TypeError("db_connection must be a DuckDB connection")
        selected = _slot_binding.resolve_slot(slot, slot_id, slot_registry)
        self._ashare_panel = _slot_binding.uses_ashare_panel(selected)
        if selected is not None and self._ashare_panel:
            db_connection = selected.bar_feed.connection if db_connection is None else db_connection
        if slot_config_hash is not None and (not isinstance(slot_config_hash, str) or not slot_config_hash.strip()):
            raise ValueError('slot_config_hash must be a nonempty string')
        self.universe_id = universe_id
        self.slot_execution_mode = _slot_execution_mode.validate_slot_execution_mode(
            slot_execution_mode if slot_execution_mode is not None else type(self).SLOT_EXECUTION_MODE
        )
        self.last_slot_execution_mode: Optional[str] = None
        self.db_connection = db_connection
        self.factor_name = factor_name
        self.start_date = start_date
        self.end_date = end_date
        self.stock_pool = stock_pool
        self.cache_dir = cache_dir
        self.use_dynamic_pool = use_dynamic_pool
        self.stock_filters = stock_filters
        self.max_stocks_per_month = max_stocks_per_month
        self.sort_by = sort_by
        self.ascending = ascending
        self.rebalance_period_months = rebalance_period_months
        self.rebalance_schedule = rebalance_schedule
        self.pool_schedule = pool_schedule
        self.fundamental_table = fundamental_table
        self.price_table = price_table
        _bar_bridge.initialize_frequency_route(
            self, data_frequency, analysis_mode, session_view, output_frequency,
            label_config, data_version, session_calendar, source_revision,
            slot=selected, legacy_panel=self._ashare_panel,
        )
        # 动态股票池应用位置：'raw'（仅原始数据阶段）、'factor'（仅结果阶段）、'both'
        allowed_modes = ('raw', 'factor', 'both')
        if dynamic_pool_mode not in allowed_modes:
            self.logger.warning(
                "无效 dynamic_pool_mode '%s'，将回退为 'factor'", dynamic_pool_mode
            )
            dynamic_pool_mode = 'factor'
        self.dynamic_pool_mode = dynamic_pool_mode

        if db_backend not in ('duckdb', 'auto'):
            raise ValueError("Only the duckdb backend is supported")
        db_backend = 'duckdb'
        self.db_backend = db_backend
        self.duckdb_path = duckdb_path
        self.parquet_dir = parquet_dir
        self.use_duckdb_cache = use_duckdb_cache
        self.duckdb_cache_size_mb = duckdb_cache_size_mb
        self.data_bindings = data_bindings
        self.calendar_duckdb_path = calendar_duckdb_path
        self.data_config_path = data_config_path
        self.source_policy = source_policy
        self._local_data_config = None
        self._duckdb_adapter = None

        self.dynamic_pool_manager = None
        try:
            if self._ashare_panel:
                _legacy_storage.initialize_configured_adapter(self, duckdb_path, parquet_dir)
            self.slot = _slot_binding.bind_ashare_slot(self, selected) if self._ashare_panel else selected
            self.slot_id = self.slot.slot_id
            self.slot_config_hash = slot_config_hash or _slot_binding.config_identity(self.slot)
            self.pit_enabled = pit_enabled
            self.pit_factor_column = pit_factor_column
            self.pit_window = pit_window
            _legacy_storage.initialize_stock_pool(self, FactorBase, DynamicStockPoolManager, stock_pools_by_period)

            if self.cache_dir and not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

            pool_info = f"动态股票池({len(self.all_stocks)}只股票)" if use_dynamic_pool else f"固定股票池({len(stock_pool) if stock_pool else 0}只股票)"
            self.logger.info(f"初始化因子 {factor_name}，时间范围: {start_date} 到 {end_date}，{pool_info}")
        except BaseException:
            try:
                FactorBase.close(self)
            except BaseException:
                self.logger.warning("Failed to close owned reader after construction failure", exc_info=True)
            raise

    def close(self) -> None:
        """Release owned adapters. Borrowed connections and slots stay open."""
        pool_reader = getattr(self.dynamic_pool_manager, '_duckdb_adapter', None)
        try:
            if pool_reader is not None and pool_reader is not self._duckdb_adapter:
                pool_reader.close()
        finally:
            if self._duckdb_adapter is not None:
                self._duckdb_adapter.close()

    def _init_duckdb_adapter(self):
        return _legacy_storage._init_duckdb_adapter(self, DUCKDB_ADAPTER_AVAILABLE, get_data_adapter)

    def _get_local_data_config(self):
        return _legacy_storage._get_local_data_config(self)

    def _is_cloud_environment(self) -> bool:
        """Detect the JoinQuant environment before initializing a local reader."""
        try:
            import jqdata  # noqa: F401 - import availability identifies the cloud runtime
            return True
        except ImportError:
            pass

        if os.environ.get('JQ_RUNNING') or os.environ.get('QUANTOPIAN_RUNNING'):
            return True

        bound = getattr(self, "data_bindings", None)
        if bound is None:
            bound = os.environ.get("QUANT_DATA_BINDINGS", "{}") not in ("", "{}", "null")
        if bound:
            return False  # Missing legacy paths cannot turn an explicit local binding into cloud mode.

        return False

    def _apply_pit_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        return _slot_binding.adjust_panel(self, df)

    @property
    def monthly_stock_pools(self):
        """Compatibility alias; both names refer to the same pool mapping."""
        return self.stock_pools_by_period

    @monthly_stock_pools.setter
    def monthly_stock_pools(self, value):
        self.stock_pools_by_period = value

    @staticmethod
    def _parse_stock_pool_intervals(stock_pools_by_period, date_timezone=None):
        return _legacy_storage._parse_stock_pool_intervals(stock_pools_by_period, date_timezone)

    def _get_stock_pool_intervals(self, date_timezone=None):
        return _legacy_storage._get_stock_pool_intervals(self, FactorBase, date_timezone)

    def _filter_data_by_pools(self, df: pd.DataFrame) -> pd.DataFrame:
        return _legacy_storage._filter_data_by_pools(self, FactorBase, df)

    def _filter_data_by_monthly_pools(self, df: pd.DataFrame) -> pd.DataFrame:
        return _legacy_storage._filter_data_by_monthly_pools(self, df)

    def _apply_dynamic_pool_to_factor_df(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        return _legacy_storage._apply_dynamic_pool_to_factor_df(self, FactorBase, factor_df)

    def _fetch_raw_data(self, fields: List[str], date_field: str = 'date',
                        code_field: str = 'stock_code', table_name: str = 'factor_data',
                        additional_condition: str = "") -> pd.DataFrame:
        return _legacy_storage._fetch_raw_data(
            self, _bar_bridge.read_slot_panel, fields, date_field, code_field, table_name, additional_condition)

    def _fetch_raw_data_duckdb(self, fields: List[str], table_name: str = 'factor_data') -> pd.DataFrame:
        if self._duckdb_adapter is None:
            return None
        return self._fetch_raw_data(fields, table_name=table_name)

    def _slot_clock(self) -> pd.DataFrame:
        return _slot_replay.slot_clock(self, _bar_bridge._bucket_strategy)

    def _fetch_market_bars(self) -> pd.DataFrame:
        return _bar_bridge._fetch_market_bars(self)

    def session_rolling(self, data: pd.DataFrame, column: str, window: int, **kwargs) -> pd.Series:
        """Roll over the expected Session grid, preserving gaps."""
        return _bar_bridge.session_rolling(self, data, column, window, **kwargs)

    def _get_market_factor_values(self, force_recalculate: bool = False) -> pd.DataFrame:
        return _bar_bridge._get_market_factor_values(self, force_recalculate)

    @staticmethod
    def _slot_execution_bindings() -> tuple:
        return (FactorBase, StockFilter, apply_pit_adjustment, _cache, _slot_execution_mode, _slot_fast_path)

    def get_incremental_engine(self):
        """Return an optional built-in incremental engine spec for non-A-share Slots."""
        return getattr(self, 'INCREMENTAL_ENGINE', None)

    def init_incremental_state(self):
        """Initialize causal incremental state for step-by-step Bar updates."""
        raise NotImplementedError("增量执行因子需实现 init_incremental_state 和 step_incremental")

    def step_incremental(self, state, bar_slice: pd.DataFrame):
        """Advance causal state by one cutoff slice and return (new_state, step_values)."""
        raise NotImplementedError("增量执行因子需实现 init_incremental_state 和 step_incremental")

    save_factor_values = _slot_binding.save_factor_values
    save_bar_factor_values = save_factor_values

    def get_bar_factor_frame(self, force_recalculate: bool = False) -> pd.DataFrame:
        """Return UTC (timestamp_utc, instrument_id, factor_value), retaining NaNs."""
        return _bar_bridge.get_bar_factor_frame(self, force_recalculate)

    def get_session_daily_pricing(self) -> pd.Series:
        """Return MultiIndex(asset, date) prices bound to the Session calendar."""
        return _bar_bridge.get_session_daily_pricing(self)

    def evaluate_bar_factor(self, **evaluator_options):
        """Label and evaluate a Bar factor using the same read snapshot."""
        return _bar_bridge.evaluate_bar_factor(self, **evaluator_options)

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a dates-by-stocks DataFrame, or an empty DataFrame for no result."""
        raise NotImplementedError("子类必须实现calculate_factor方法")

    def _get_raw_data_for_factor(self) -> pd.DataFrame:
        """Return the raw input DataFrame required by the concrete factor."""
        raise NotImplementedError("子类必须实现_get_raw_data_for_factor方法")

    @staticmethod
    def _normalize_factor_dataframe(values: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
        return _execution._normalize_factor_dataframe(values, raw_data)

    def get_factor_values(self, force_recalculate: bool = False) -> pd.DataFrame:
        """Compute or load the factor values for the configured stock pools."""
        return _execution.get_factor_values(self, FactorBase, force_recalculate)

    def get_factor_values_unfiltered(self, force_recalculate: bool = False) -> pd.DataFrame:
        """Compute over the current stock universe without period membership filtering."""
        return _execution.get_factor_values_unfiltered(self, FactorBase, force_recalculate)

    def get_factor_values_unfiltered_chunked(self, chunk_size: int) -> pd.DataFrame:
        """Compute over stock chunks when the factor proves stock independence."""
        return _execution.get_factor_values_unfiltered_chunked(self, FactorBase, chunk_size)

    def _get_factor_values_from_prepared_stock_chunks(
        self,
        chunk_size: int,
        prepare_chunk: Callable[[pd.DataFrame], pd.DataFrame],
        calculate_prepared: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        mode: str = "two_stage_intraday",
        min_total_rows: int = 0,
    ) -> pd.DataFrame:
        return _execution._get_factor_values_from_prepared_stock_chunks(
            self, chunk_size, prepare_chunk, calculate_prepared, mode=mode, min_total_rows=min_total_rows,
        )

    def _get_factor_values_by_rebalance_pool(
        self,
        window_bars: int,
        warmup_calendar_days: int = 0,
        use_last_available: bool = False,
        combine_duplicate_dates: bool = False,
        rebalance_only: bool = True,
    ) -> pd.DataFrame:
        if not getattr(self, '_ashare_panel', True):
            return _slot_replay.replay_slot_pools(
                self, _bar_bridge.causal_slot_values, _bar_bridge.panel_slot_values,
                window_bars, warmup_calendar_days, use_last_available, combine_duplicate_dates, rebalance_only,
            )
        return _execution._get_factor_values_by_rebalance_pool(
            self, FactorBase, window_bars, warmup_calendar_days, use_last_available,
            combine_duplicate_dates, rebalance_only,
        )

    def get_price_data(self, price_type: str = 'close') -> pd.DataFrame:
        """Read execution prices from the factor's market input."""
        if not self._ashare_panel or self._frequency_route.bar_grid:
            return _bar_bridge.execution_panel(self, price_type)
        return _legacy_storage.get_price_data(self, price_type)

    def calculate_forward_returns(self, period: int = 1) -> pd.DataFrame:
        """Compute forward returns from unadjusted close prices."""
        if not self._ashare_panel or self._frequency_route.bar_grid:
            return _bar_bridge.execution_panel(self, 'close', period)
        return _legacy_storage.calculate_forward_returns(self, period)

    def _get_cache_inputs(self) -> Optional[dict]:
        """Declare all JSON parameters and input DataFrames, or disable caching with None."""
        return _cache._get_cache_inputs(self)

    def _get_cache_key(self, raw_data: Optional[pd.DataFrame] = None, *, captured_inputs: dict | None = None) -> Optional[str]:
        return _cache._get_cache_key(self, FactorBase, StockFilter, apply_pit_adjustment, raw_data, captured_inputs=captured_inputs)

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        return _cache._save_to_cache(self, data, cache_key)

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        return _cache._load_from_cache(self, cache_key)

    def get_stock_pool_for_date(self, date: str) -> List[str]:
        """Return the pool covering a date, or an empty list for a missing period."""
        return _legacy_storage.get_stock_pool_for_date(self, FactorBase, date)

    @staticmethod
    def vectorized_rolling_window(data: np.ndarray, window_size: int, func: Callable):
        """Apply a scalar callback to read-only windows of a one-dimensional array."""
        return _execution.vectorized_rolling_window(data, window_size, func)

    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by stock and date while retaining the input precision."""
        return _execution.optimize_dataframe_operations(self, df)
