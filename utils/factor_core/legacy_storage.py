"""Local DuckDB reads, PIT adjustment and stock-pool membership for FactorBase."""
from datetime import datetime
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def initialize_configured_adapter(
    self, duckdb_path: Optional[str], parquet_dir: Optional[str],
) -> None:
    if not self._frequency_route.bar_grid and (
        self.duckdb_path or self.parquet_dir or self.data_bindings
        or os.environ.get('QUANT_DATA_BINDINGS', '{}') not in ('', '{}', 'null')
    ):
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


def _apply_pit_adjustment(self, apply_pit_adjustment: Callable, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Dynamic PIT adjustment by replacing price columns.

    PIT Formula:
        adjusted_price[i] = price[i] × (factor[i] / factor_end)

    Where factor_end is the factor of the last date in the data window per stock.
    This ensures:
    - On the calculation date (window end), price remains unchanged
    - Historical prices are adjusted relative to the calculation date

    This matches JoinQuant's dynamic PIT behavior where each query window
    uses its own last date's factor as factor_end.

    Mathematical properties:
        - At calculation date T: adjusted_price[T] = price[T] (factor[T]/factor[T] = 1)
        - For historical date i: adjusted_price[i] = price[i] × (factor[i]/factor[T])
        - Return calculation: adjusted[T]/adjusted[i] = price[T]×factor[T] / (price[i]×factor[i])
          This correctly accounts for stock splits/dividends between i and T.

    Args:
        df: DataFrame containing raw prices and factor column

    Returns:
        DataFrame with price/volume columns replaced by pit-adjusted values
    """
    if not self.pit_enabled or df.empty:
        return df

    factor_col = self.pit_factor_column
    return apply_pit_adjustment(df, factor_col)


def initialize_stock_pool(
    self,
    factor_base_cls: type,
    manager_factory: Callable,
    db_connection,
    use_dynamic_pool: bool,
    stock_pools_by_period: Optional[Dict[str, List[str]]],
    stock_pool: Optional[List[str]],
    stock_filters: Optional[list],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    parquet_dir: Optional[str],
    use_duckdb_cache: bool,
    duckdb_cache_size_mb: int,
    start_date: str,
    end_date: str,
    max_stocks_per_month: Optional[int],
    sort_by: Optional[str],
    ascending: bool,
    rebalance_period_months: int,
    rebalance_schedule: str,
    pool_schedule: Optional[dict],
) -> None:
    # Only rule-generated pools need a selector; supplied histories are authoritative.
    if use_dynamic_pool and stock_pools_by_period is None:
        if not stock_filters:
            raise ValueError("使用动态股票池时必须提供stock_filters")
        self.dynamic_pool_manager = manager_factory(
            db_connection,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=self.duckdb_path,
            parquet_dir=parquet_dir,
            use_duckdb_cache=use_duckdb_cache,
            duckdb_cache_size_mb=duckdb_cache_size_mb,
            data_bindings=self.data_bindings,
            source_policy=self.source_policy,
            data_config_path=self.data_config_path,
        )
        if not self.dynamic_pool_manager.validate_filters(stock_filters):
            raise ValueError("股票筛选条件或数据源验证失败，请检查字段、指数条件和数据源")
    elif not use_dynamic_pool and not stock_pool:
        raise ValueError("不使用动态股票池时必须提供固定的stock_pool")

    if self.use_dynamic_pool:
        self.stock_pools_by_period = (
            stock_pools_by_period if stock_pools_by_period is not None
            else self.dynamic_pool_manager.get_stock_pools_for_period(
                start_date, end_date, stock_filters, max_stocks_per_month, sort_by, ascending,
                rebalance_period_months=rebalance_period_months,
                rebalance_schedule=rebalance_schedule,
                schedule=pool_schedule,
            )
        )
        factor_base_cls._parse_stock_pool_intervals(self.stock_pools_by_period)
        self.all_stocks = list({stock for stocks in self.stock_pools_by_period.values() for stock in stocks})
        self.logger.info(f"动态股票池包含 {len(self.all_stocks)} 只股票，覆盖 {len(self.stock_pools_by_period)} 个期间")
    else:
        self.stock_pools_by_period = None
        self.all_stocks = stock_pool


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
    if getattr(factor_df.index, 'tz', None) is not None:
        pool_intervals = factor_base_cls._get_stock_pool_intervals(self, factor_df.index.tz)
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
            dates_in_period &= factor_df.index >= start_date
        if end_date is not None:
            dates_in_period &= factor_df.index < end_date
        available_stocks = factor_df.columns.intersection(pool_stocks)
        pool_membership_mask.loc[dates_in_period, available_stocks] = True
    return factor_df.where(pool_membership_mask)


def _fetch_raw_data(self, fields: List[str], date_field: str = 'date',
                    code_field: str = 'stock_code', table_name: str = 'factor_data',
                    additional_condition: str = "") -> pd.DataFrame:
    """Read the configured DuckDB source; dates include the entire final day."""
    is_price = table_name.startswith('price_data')
    if self._duckdb_adapter is not None and date_field == 'date' and code_field == 'stock_code' and not additional_condition:
        return self._fetch_raw_data_duckdb(fields, table_name)
    if is_price:
        from utils.bar_source import BarSource
        if not isinstance(fields, (list, tuple)):
            raise ValueError('fields must be a list or tuple of column names')
        selected = list(fields)
        if self.pit_enabled and self.pit_factor_column not in selected:
            selected.append(self.pit_factor_column)
        frame = BarSource.read_panel(self.all_stocks, self.data_frequency,
            self.start_date, self.end_date, selected, config=self._get_local_data_config(),
            connection=self.db_connection, adapter=self._duckdb_adapter,
            date_field=date_field, code_field=code_field, additional_condition=additional_condition)
        if self.use_dynamic_pool and self.dynamic_pool_mode in ('raw', 'both'):
            frame = self._filter_data_by_pools(frame)
        return frame
    connection = self._duckdb_adapter._get_connection() if self._duckdb_adapter is not None else self.db_connection
    import duckdb
    if not isinstance(connection, duckdb.DuckDBPyConnection):
        raise TypeError("Supply a DuckDB connection, duckdb_path, Parquet directory, or data binding")

    def quote(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    config = self._get_local_data_config()
    dataset = config.dataset(
        config.factor_table if table_name == 'factor_data' else table_name)
    physical_table = dataset.table
    if date_field == 'date':
        date_field = dataset.date_column
    if code_field == 'stock_code':
        code_field = dataset.stock_column
    if not isinstance(fields, (list, tuple)):
        raise ValueError('fields must be a list or tuple of column names')
    excluded = {date_field, code_field, 'date', 'stock_code', 'code'}
    selected = list(dict.fromkeys(f for f in fields if f not in excluded))
    from utils.data_access_config import Dataset
    from utils.data_store import table_name as quote_table
    selected = list(Dataset(physical_table, date_field, code_field).fields(selected))
    if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
        raise ValueError('start_date must not be after end_date')
    if not self.all_stocks:
        frame = pd.DataFrame(columns=['date', 'code', *selected])
        frame['date'] = pd.to_datetime(frame['date'])
        return frame
    columns = [f'{quote(date_field)} AS date', f'{quote(code_field)} AS code'] + [quote(f) for f in selected]
    placeholders = ', '.join('?' for _ in self.all_stocks)
    end_exclusive = (pd.Timestamp(self.end_date) + pd.Timedelta(days=1)).to_pydatetime()
    sql = f"""SELECT {', '.join(columns)} FROM {quote_table(physical_table)}
        WHERE CAST({quote(date_field)} AS TIMESTAMP) >= ?
          AND CAST({quote(date_field)} AS TIMESTAMP) < ?
          AND {quote(code_field)} IN ({placeholders})
          {additional_condition}
        ORDER BY {quote(code_field)}, {quote(date_field)}"""
    frame = connection.execute(sql, [pd.Timestamp(self.start_date).to_pydatetime(), end_exclusive, *self.all_stocks]).fetchdf()
    frame['date'] = pd.to_datetime(frame['date'])
    if self.use_dynamic_pool and self.dynamic_pool_mode in ('raw', 'both'):
        frame = self._filter_data_by_pools(frame)
    return frame


def _fetch_raw_data_duckdb(
    self,
    fields: List[str],
    table_name: str = 'factor_data'
) -> pd.DataFrame:
    """使用 DuckDB 适配器获取原始数据 (高性能路径)

    此方法使用 DuckDB 进行数据查询。
    仅在本地分析时使用，云端部署会自动跳过。

    Args:
        fields: 需要查询的字段列表
        table_name: 数据表名

    Returns:
        包含查询结果的 DataFrame
    """
    if self._duckdb_adapter is None:
        return None

    # 确定实际表名和数据频率
    # Fix: Recognize all price table variants (price_data, price_data_daily, price_data_5m)
    is_price_table = table_name.startswith('price_data')

    # 如果启用 PIT 动态复权，确保查询包含 factor 列
    query_fields = list(fields)
    if is_price_table and self.pit_enabled and self.pit_factor_column not in query_fields:
        query_fields.append(self.pit_factor_column)
    if is_price_table and self.data_frequency in ('daily', '1d') and 'money' in query_fields and 'paused' not in query_fields:
        query_fields.append('paused')

    if is_price_table:
        # 价格数据查询
        from utils.bar_source import BarSource
        df = BarSource.read_panel(self.all_stocks, self.data_frequency,
            self.start_date, self.end_date, query_fields, config=self._get_local_data_config(),
            adapter=self._duckdb_adapter)
        from utils.pit import normalize_price_money
        frequency = self._frequency_route.frequency
        money_frequency = 'daily' if frequency.kind == 'session' else '1m' if frequency.minutes == 1 else '5m'
        df = normalize_price_money(df, source='local', frequency=money_frequency)
        if 'paused' not in fields and 'paused' in df:
            df = df.drop(columns='paused')
    else:
        fetch = (
            self._duckdb_adapter.fetch_factor_data
            if table_name == 'factor_data'
            else self._duckdb_adapter.fetch_table_data
        )
        df = fetch(
            stocks=self.all_stocks,
            start_date=self.start_date,
            end_date=self.end_date,
            fields=query_fields,
            **({} if table_name == 'factor_data' else {'table_name': table_name}),
        )

    # 如果配置为在原始数据阶段应用动态股票池过滤
    if self.use_dynamic_pool and self.dynamic_pool_mode in ('raw', 'both'):
        df = self._filter_data_by_pools(df)

    return df


def get_price_data(self, price_type: str = 'close') -> pd.DataFrame:
    """Read unadjusted prices from the same DuckDB source as the factor."""
    frame = self._fetch_raw_data([price_type], table_name='price_data')
    if frame.empty:
        return pd.DataFrame()
    if self.use_dynamic_pool:
        frame = self._filter_data_by_pools(frame)
    return frame.pivot(index='date', columns='code', values=price_type).sort_index()


def calculate_forward_returns(self, period: int = 1) -> pd.DataFrame:
    """Calculate forward returns for a specific period

    Args:
        period: Number of days for the forward-looking period (default: 1)

    Returns:
        DataFrame with forward returns, rows are dates and columns are stock codes
    """
    # 获取收盘价
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
