"""Local DuckDB reads, PIT adjustment and stock-pool membership for FactorBase."""
from datetime import datetime
import inspect
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def initialize_configured_adapter(
    self, duckdb_path: Optional[str], parquet_dir: Optional[str],
) -> None:
    if self._frequency_route.bar_grid:
        return
    bindings = self.data_bindings
    if bindings or (bindings is None and os.environ.get('QUANT_DATA_BINDINGS', '{}') not in ('', '{}', 'null')):
        from utils.data_store import binding_contract

        bindings = binding_contract(bindings)
    if self.duckdb_path or self.parquet_dir or bindings:
        self._duckdb_adapter = self._init_duckdb_adapter()
        if self._duckdb_adapter:
            self.duckdb_path = str(self._duckdb_adapter.duckdb_path) if self._duckdb_adapter.duckdb_path else None
            self.data_bindings = self._duckdb_adapter.data_bindings
            self.logger.info(f"DuckDB 后端已启用 (路径: {duckdb_path or parquet_dir})")
        elif not self._is_cloud_environment():
            raise RuntimeError("Configured DuckDB source could not be initialized")


def _init_duckdb_adapter(self, adapter_available: bool, adapter_factory: Callable):
    """初始化 DuckDB 适配器 (仅用于本地分析)

    此方法仅在本地环境中使用。云端部署会自动跳过，
    因为云端使用 JoinQuant API 获取数据。

    Returns:
        DuckDBAdapter 实例，如果初始化失败则返回 None
    """
    # 云端环境检测 - 跳过 DuckDB
    if self._is_cloud_environment():
        return None

    if not adapter_available:
        raise RuntimeError("DuckDB adapter is required for local database sources")

    # Configured local sources fail explicitly; empty data never selects another backend.
    return adapter_factory(
        duckdb_path=self.duckdb_path,
        parquet_dir=self.parquet_dir,
        use_cache=self.use_duckdb_cache and self.data_frequency != '5m',
        cache_size_mb=self.duckdb_cache_size_mb,
        data_bindings=self.data_bindings,
        data_config_path=self.data_config_path,
        source_policy=self.source_policy,
        calendar_duckdb_path=self.calendar_duckdb_path,
    )


def _get_local_data_config(self):
    """Load only for local reads; the cloud API does not need local JSON."""
    if self._duckdb_adapter is not None:
        return self._duckdb_adapter.data_config
    if self._local_data_config is None:
        from utils.data_access_config import load_data_access_config
        self._local_data_config = load_data_access_config(self.data_config_path)
    return self._local_data_config


def _resolve_universe(self, day) -> list[str]:
    grid = self.slot.calendar.bar_grid('1d', end=day, view=self.session_view, bucket_strategy='session')
    if grid.empty:
        return []
    anchor = grid.timestamp_utc.iloc[-1]
    session = self.slot.calendar.session_at(anchor, self.session_view, frequency='1d')
    on_requested_day = pd.Timestamp(session.session_date) == pd.Timestamp(day)
    as_of = session.segments[0][0] if on_requested_day else session.segments[-1][1]
    return list(self.slot.universe.resolve_universe(self.universe_id, as_of_utc=as_of))


def initialize_stock_pool(self, factor_base_cls: type, manager_factory: Callable,
                          stock_pools_by_period: Optional[Dict[str, List[str]]]) -> None:
    use_dynamic_pool, stock_pool = self.use_dynamic_pool, self.stock_pool
    start_date, end_date = self.start_date, self.end_date
    pool_schedule = self.pool_schedule
    rebalance_schedule, rebalance_period_months = self.rebalance_schedule, self.rebalance_period_months
    if use_dynamic_pool and stock_pools_by_period is None:
        if self.universe_id is None:
            reader = getattr(self.slot.universe, 'period_pools', None)
            if reader is None:
                raise ValueError('Dynamic pools require universe_id or stock_pools_by_period for this MarketSlot')
            stock_pools_by_period = reader(self, manager_factory)
        else:
            from utils.stock_pool_schedule import generate_rebalance_dates, load_pool_policy, resolve_schedule

            spec = resolve_schedule(load_pool_policy(), pool_schedule, legacy=rebalance_schedule,
                                    months=rebalance_period_months)
            dates = generate_rebalance_dates(self.slot.calendar, start_date, end_date, spec)
            stock_pools_by_period = {
                day.strftime('%Y-%m-%d'): _resolve_universe(self, day) for day in dates
            }
    elif not use_dynamic_pool and not stock_pool:
        if self.universe_id is None:
            raise ValueError("不使用动态股票池时必须提供固定的stock_pool")
        stock_pool = _resolve_universe(self, start_date)

    if self.use_dynamic_pool:
        self.stock_pools_by_period = stock_pools_by_period
        factor_base_cls._parse_stock_pool_intervals(self.stock_pools_by_period)
        self.all_stocks = list({stock for stocks in self.stock_pools_by_period.values() for stock in stocks})
        self.logger.info(f"动态股票池包含 {len(self.all_stocks)} 只股票，覆盖 {len(self.stock_pools_by_period)} 个期间")
    else:
        self.stock_pools_by_period = None
        self.stock_pool = self.all_stocks = stock_pool


def _parse_stock_pool_intervals(stock_pools_by_period, date_timezone=None):
    """Validate all keys and return (inclusive start, exclusive end, stocks)."""
    parsed_periods = []
    key_kind = None
    for period_key, pool_stocks in stock_pools_by_period.items():
        if not isinstance(period_key, str) or len(period_key) not in (7, 10):
            raise ValueError(f"Invalid stock pool key {period_key!r}: expected YYYY-MM or YYYY-MM-DD")
        current_kind = 'month' if len(period_key) == 7 else 'effective_date'
        date_format = '%Y-%m' if current_kind == 'month' else '%Y-%m-%d'
        try:
            parsed_date = datetime.strptime(period_key, date_format)
        except ValueError as exc:
            raise ValueError(f"Invalid stock pool key {period_key!r}") from exc
        if parsed_date.strftime(date_format) != period_key:
            raise ValueError(f"Invalid stock pool key {period_key!r}: use zero-padded dates")
        if key_kind is not None and current_kind != key_kind:
            raise ValueError("Stock pool keys cannot mix YYYY-MM and YYYY-MM-DD")
        key_kind = current_kind
        parsed_periods.append((pd.Timestamp(parsed_date), pool_stocks))
    parsed_periods.sort(key=lambda period: period[0])
    pool_intervals = []
    for index, (start_date, pool_stocks) in enumerate(parsed_periods):
        if key_kind == 'month':
            # Month keys refer to the input's local calendar, as year/month comparisons did.
            start_date = start_date.tz_localize(date_timezone)
            end_date = start_date + pd.offsets.MonthBegin(1)
        else:
            end_date = parsed_periods[index + 1][0] if index + 1 < len(parsed_periods) else None
        pool_intervals.append((start_date, end_date, pool_stocks))
    return pool_intervals


def _get_stock_pool_intervals(self, factor_base_cls: type, date_timezone=None):
    """Resolve fixed/period pools; None preserves an intentionally disabled filter."""
    if self.use_dynamic_pool:
        if not self.stock_pools_by_period:
            return None
        return factor_base_cls._parse_stock_pool_intervals(
            self.stock_pools_by_period, date_timezone=date_timezone,
        )
    # Unfiltered/PIT replay temporarily disables a configured period pool.
    if getattr(self, 'stock_pools_by_period', None) is not None:
        return None
    stocks = getattr(self, 'stock_pool', None)
    # No date bounds: fixed membership also covers warmup history.
    return [(None, None, stocks)] if stocks else None


def _filter_data_by_pools(self, factor_base_cls: type, df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw rows by the same pool intervals used for factor results."""
    pool_intervals = factor_base_cls._get_stock_pool_intervals(self)
    if pool_intervals is None:
        return df
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='raise')
    if df['date'].dt.tz is not None:
        pool_intervals = factor_base_cls._get_stock_pool_intervals(self, df['date'].dt.tz)

    filtered_rows = []
    for start_date, end_date, pool_stocks in pool_intervals:
        if not pool_stocks:
            continue
        dates_in_period = pd.Series(True, index=df.index)
        if start_date is not None:
            dates_in_period &= df['date'] >= start_date
        if end_date is not None:
            dates_in_period &= (df['date'] < end_date)
        period_rows = df[dates_in_period & df['code'].isin(pool_stocks)]
        if not period_rows.empty:
            filtered_rows.append(period_rows)
    if filtered_rows:
        result_df = pd.concat(filtered_rows, ignore_index=True)
        self.logger.debug(f"动态股票池过滤后数据形状: {result_df.shape}")
        return result_df
    return pd.DataFrame(columns=df.columns)


def _filter_data_by_monthly_pools(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Deprecated: Use _filter_data_by_pools instead.
    Maintained for backward compatibility with existing factor implementations.
    """
    return self._filter_data_by_pools(df)


def _apply_dynamic_pool_to_factor_df(self, factor_base_cls: type, factor_df: pd.DataFrame) -> pd.DataFrame:
    """Keep all input dates and columns; mask stocks outside each pool interval."""
    pool_intervals = factor_base_cls._get_stock_pool_intervals(self)
    if pool_intervals is None:
        return factor_df
    if not factor_df.empty and not isinstance(factor_df.index, pd.DatetimeIndex):
        factor_df = factor_df.copy()
        factor_df.index = pd.to_datetime(factor_df.index, errors='raise')
    dates = factor_df.index
    if getattr(dates, 'tz', None) is not None:
        if getattr(self, '_ashare_panel', True):
            pool_intervals = factor_base_cls._get_stock_pool_intervals(self, dates.tz)
        else:
            sessions = (self.slot.calendar.session_at(date, self.session_view,
                        frequency=self._frequency_route.frequency) for date in dates)
            dates = pd.to_datetime([session.session_date if session else None for session in sessions])
    if factor_df.empty:
        return factor_df

    pool_membership_mask = pd.DataFrame(
        False, index=factor_df.index, columns=factor_df.columns,
    )
    for start_date, end_date, pool_stocks in pool_intervals:
        if not pool_stocks:
            continue
        dates_in_period = np.ones(len(factor_df.index), dtype=bool)
        if start_date is not None:
            dates_in_period &= dates >= start_date
        if end_date is not None:
            dates_in_period &= dates < end_date
        available_stocks = factor_df.columns.intersection(pool_stocks)
        pool_membership_mask.loc[dates_in_period, available_stocks] = True
    return factor_df.where(pool_membership_mask)


def _fetch_raw_data(self, read_slot_panel: Callable, fields: List[str], date_field: str = 'date',
                    code_field: str = 'stock_code', table_name: str = 'factor_data',
                    additional_condition: str = "", *, optional_fields: Sequence[str] = ()) -> pd.DataFrame:
    """Retain the FactorBase panel contract while letting the slot supply its rows."""
    price = table_name.startswith('price_data')
    component = self.slot.bar_feed if price else self.slot.aux_features
    read_panel = getattr(component, 'read_factor_panel', None)
    if read_panel is not None:
        kwargs = {'optional_fields': optional_fields} if optional_fields else {}
        frame = read_panel(self, fields, date_field, code_field, table_name, additional_condition, **kwargs)
    else:
        if additional_condition or date_field != 'date' or code_field != 'stock_code':
            raise ValueError('SQL panel options require the historical A-share slot')
        frame = read_slot_panel(self, fields, table_name, price)
    if read_panel is not None and self.use_dynamic_pool and self.dynamic_pool_mode in ('raw', 'both'):
        frame = self._filter_data_by_pools(frame)
    return frame


def get_price_data(self, price_type: str = 'close', *, adjusted: bool = True) -> pd.DataFrame:
    """Read the historical price panel, factor-adjusted once unless ``adjusted=False``."""
    from utils.pit import (
        ADJUSTABLE_PRICE_FIELDS,
        PRICE_ADJUSTED_ATTR,
        PRICE_ADJUSTMENT_MODE_ATTR,
    )
    from utils.price_adjustment import adjust_prices_by_factor

    adjustable = price_type in (*ADJUSTABLE_PRICE_FIELDS, 'volume')
    if adjusted and adjustable and not self.pit_enabled:
        params = inspect.signature(self._fetch_raw_data).parameters
        accepts_optional = 'optional_fields' in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        frame = (
            self._fetch_raw_data([price_type], table_name='price_data', optional_fields=(self.pit_factor_column,))
            if accepts_optional
            else self._fetch_raw_data([price_type], table_name='price_data')
        )
    else:
        frame = self._fetch_raw_data([price_type], table_name='price_data')
    if frame.empty:
        return pd.DataFrame()
    if self.use_dynamic_pool:
        frame = self._filter_data_by_pools(frame)
        if frame.empty:
            return pd.DataFrame()
    if adjusted and adjustable and not frame.attrs.get(PRICE_ADJUSTED_ATTR):
        if self.pit_enabled:
            frame = self._apply_pit_adjustment(frame)
        elif self.pit_factor_column in frame.columns:
            frame = adjust_prices_by_factor(
                frame, factor_column=self.pit_factor_column, require_factor=False,
            )
    wide = frame.pivot(index='date', columns='code', values=price_type).sort_index()
    wide.attrs[PRICE_ADJUSTED_ATTR] = bool(frame.attrs.get(PRICE_ADJUSTED_ATTR, False))
    if PRICE_ADJUSTMENT_MODE_ATTR in frame.attrs:
        wide.attrs[PRICE_ADJUSTMENT_MODE_ATTR] = frame.attrs[PRICE_ADJUSTMENT_MODE_ATTR]
    return wide


def calculate_forward_returns(self, period: int = 1) -> pd.DataFrame:
    """Calculate forward returns from factor-adjusted close prices.

    Args:
        period: Number of days for the forward-looking period (default: 1)

    Returns:
        DataFrame with forward returns, rows are dates and columns are stock codes
    """
    # 获取复权收盘价 (若 get_price_data 已完成 PIT/factor 复权则不会重复乘 factor)
    price_df = self.get_price_data(price_type='close')

    # 计算前瞻收益率 (future_price / current_price - 1)
    forward_return = price_df.shift(-period) / price_df - 1

    self.logger.info(f"成功计算前瞻收益，周期: {period}天")
    return forward_return


def get_stock_pool_for_date(self, factor_base_cls: type, date: str) -> List[str]:
    """Return the pool covering a date; missing months do not carry forward."""
    pool_intervals = factor_base_cls._get_stock_pool_intervals(self)
    if pool_intervals is None or pool_intervals[0][0] is None:
        return self.stock_pool or []
    target_date = pd.to_datetime(date, errors='raise')
    if target_date.tz is not None:
        pool_intervals = factor_base_cls._get_stock_pool_intervals(self, target_date.tz)
    for start_date, end_date, pool_stocks in pool_intervals:
        if target_date >= start_date and (end_date is None or target_date < end_date):
            return pool_stocks
    return []
