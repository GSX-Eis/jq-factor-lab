import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import tempfile
import logging
from typing import List, Optional, Callable, Dict, Any, Union
import time
import gc

# Import after other imports to avoid circular dependency
try:
    from .dynamic_stock_pool import DynamicStockPoolManager, StockFilter
except ImportError:
    # If import fails, define dummy classes
    class DynamicStockPoolManager:
        def __init__(self, *args, **kwargs):
            pass
    class StockFilter:
        def __init__(self, *args, **kwargs):
            pass

# Local storage uses DuckDB; cloud bundles fetch data through the JoinQuant API.
try:
    from .duckdb_adapter import DuckDBAdapter, get_data_adapter
    DUCKDB_ADAPTER_AVAILABLE = True
except ImportError:
    DUCKDB_ADAPTER_AVAILABLE = False
    DuckDBAdapter = None
    get_data_adapter = None

# PIT 动态前复权实现:
# _apply_pit_adjustment 统一处理: adjusted_price = price × (factor / factor_end)
# factor_end = 每只股票在数据窗口中的最后一个 factor 值


from utils.pit import apply_pit_adjustment


# ==================== PIT 窗口配置读取 ====================
def _load_factor_requirements() -> dict:
    """加载因子配置文件
    
    Returns:
        配置字典，如果文件不存在返回空字典
    """
    if "FACTOR_REQUIREMENTS" in globals():
        return FACTOR_REQUIREMENTS.get("factors", {})
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


# 单参数函数也需要处理不同类型
class FactorBase:
    """因子存储与计算基类
    
    主要职责是在给定的时间区间和股票池范围内，从数据库获取必要的原始数据，
    通过用户自定义的逻辑计算出因子值，并将结果转化为Alphalens所需的格式。
    """
    
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
        # DuckDB configuration (new)
        db_backend: str = 'duckdb',
        duckdb_path: Optional[str] = None,
        parquet_dir: Optional[str] = None,
        use_duckdb_cache: bool = True,
        duckdb_cache_size_mb: int = 2000,
        # PIT dynamic adjustment configuration (new)
        pit_enabled: bool = True,  # 默认启用动态复权
        pit_factor_column: str = 'factor',  # 复权因子列名
        pit_window: int = None,  # PIT 调整窗口大小（交易日），None 则使用因子的 window 属性或默认 120
        data_bindings: Optional[dict] = None,
        stock_pools_by_period: Optional[Dict[str, List[str]]] = None,
        data_config_path: Optional[str] = None,
        source_policy: Optional[str] = None,
        calendar_duckdb_path: Optional[str] = None,
        pool_schedule: Optional[dict] = None,
    ):
        """初始化因子基类
        
        Args:
            db_connection: 数据库连接对象
            factor_name: 因子名称，用于标识和构建缓存键
            start_date: 起始日期，格式为'YYYY-MM-DD'
            end_date: 截止日期，格式为'YYYY-MM-DD'
            stock_pool: 固定股票代码列表（当use_dynamic_pool=False时使用）
            cache_dir: 可选的本地缓存目录
            use_dynamic_pool: 是否使用动态股票池
            stock_filters: 仅在未提供预计算池、需要按规则生成股票池时使用
            max_stocks_per_month: 每月最大股票数量
            sort_by: 排序字段
            ascending: 排序方向
            rebalance_period_months: 调仓周期（月），默认为1
            rebalance_schedule: 调仓计划，'quarterly' 或 'report_based'
            monthly_stock_pools: stock_pools_by_period 的旧参数名，保留兼容
            stock_pools_by_period: 预计算股票池，键统一为 YYYY-MM 或 YYYY-MM-DD
            fundamental_table: 基础数据表名
            price_table: 价格数据表名
        """
        if monthly_stock_pools is not None and stock_pools_by_period is not None:
            raise ValueError("Use only stock_pools_by_period or monthly_stock_pools, not both")
        if stock_pools_by_period is None:
            stock_pools_by_period = monthly_stock_pools
        if stock_pools_by_period is not None:
            FactorBase._parse_stock_pool_intervals(stock_pools_by_period)

        # 验证日期格式
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("日期格式错误，请使用'YYYY-MM-DD'格式")
        
        # 创建日志 - 移到前面，确保在需要时可用
        self.logger = logging.getLogger(f"Factor.{factor_name}")
        
        # 保存实例属性
        if db_connection is not None:
            import duckdb
            if not isinstance(db_connection, duckdb.DuckDBPyConnection):
                raise TypeError("db_connection must be a DuckDB connection")
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
        self.data_frequency = data_frequency
        # 动态股票池应用位置：'raw'（仅原始数据阶段）、'factor'（仅结果阶段）、'both'
        allowed_modes = ('raw', 'factor', 'both')
        if dynamic_pool_mode not in allowed_modes:
            self.logger.warning(
                "无效 dynamic_pool_mode '%s'，将回退为 'factor'", dynamic_pool_mode
            )
            dynamic_pool_mode = 'factor'
        self.dynamic_pool_mode = dynamic_pool_mode
        
        # ==================== DuckDB 后端初始化 ====================
        # 仅用于本地分析加速，云端部署会自动跳过
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
            # 初始化 DuckDB 适配器 (如果配置为使用)
            if self.duckdb_path or self.parquet_dir or self.data_bindings or os.environ.get('QUANT_DATA_BINDINGS', '{}') not in ('', '{}', 'null'):
                self._duckdb_adapter = self._init_duckdb_adapter()
                if self._duckdb_adapter:
                    self.duckdb_path = str(self._duckdb_adapter.duckdb_path) if self._duckdb_adapter.duckdb_path else None
                    self.data_bindings = self._duckdb_adapter.data_bindings
                    self.logger.info(f"DuckDB 后端已启用 (路径: {duckdb_path or parquet_dir})")
                elif not self._is_cloud_environment():
                    raise RuntimeError("Configured DuckDB source could not be initialized")
        
            # PIT 动态前复权配置
            self.pit_enabled = pit_enabled
            self.pit_factor_column = pit_factor_column
            self.pit_window = pit_window
        
            if pit_enabled:
                self.logger.info("PIT 动态前复权已启用")
        
            # Only rule-generated pools need a selector; supplied histories are authoritative.
            if use_dynamic_pool and stock_pools_by_period is None:
                if not stock_filters:
                    raise ValueError("使用动态股票池时必须提供stock_filters")
                self.dynamic_pool_manager = DynamicStockPoolManager(
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
                FactorBase._parse_stock_pool_intervals(self.stock_pools_by_period)
                self.all_stocks = list({stock for stocks in self.stock_pools_by_period.values() for stock in stocks})
                self.logger.info(f"动态股票池包含 {len(self.all_stocks)} 只股票，覆盖 {len(self.stock_pools_by_period)} 个期间")
            else:
                self.stock_pools_by_period = None
                self.all_stocks = stock_pool

            # 如果设置了缓存目录，确保它存在
            if self.cache_dir and not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

            pool_info = f"动态股票池({len(self.all_stocks)}只股票)" if use_dynamic_pool else f"固定股票池({len(stock_pool) if stock_pool else 0}只股票)"
            self.logger.info(f"初始化因子 {factor_name}，时间范围: {start_date} 到 {end_date}，{pool_info}")
        except BaseException:
            # Release owned readers without replacing the original construction failure.
            for reader in (self._duckdb_adapter, getattr(self.dynamic_pool_manager, '_duckdb_adapter', None)):
                if reader is not None:
                    try:
                        reader.close()
                    except BaseException:
                        self.logger.warning("Failed to close owned reader after construction failure", exc_info=True)
            raise

    def _init_duckdb_adapter(self):
        """初始化 DuckDB 适配器 (仅用于本地分析)
        
        此方法仅在本地环境中使用。云端部署会自动跳过，
        因为云端使用 JoinQuant API 获取数据。
        
        Returns:
            DuckDBAdapter 实例，如果初始化失败则返回 None
        """
        # 云端环境检测 - 跳过 DuckDB
        if self._is_cloud_environment():
            return None
            
        if not DUCKDB_ADAPTER_AVAILABLE:
            raise RuntimeError("DuckDB adapter is required for local database sources")
            
        # Configured local sources fail explicitly; empty data never selects another backend.
        return get_data_adapter(
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
    
    def _is_cloud_environment(self) -> bool:
        """检测是否在云端环境运行 (如 JoinQuant)
        
        云端环境的特征:
        1. 没有本地 DuckDB 文件
        2. 存在特定的云端环境变量或模块
        
        Returns:
            True 如果在云端环境
        """
        # 检查是否有 JoinQuant 特有的模块
        try:
            import jqdata
            return True
        except ImportError:
            pass
            
        # 检查环境变量
        if os.environ.get('JQ_RUNNING') or os.environ.get('QUANTOPIAN_RUNNING'):
            return True

        bound = getattr(self, "data_bindings", None)
        if bound is None:
            bound = os.environ.get("QUANT_DATA_BINDINGS", "{}") not in ("", "{}", "null")
        if bound:
            return False  # Missing legacy paths cannot turn an explicit local binding into cloud mode.
            
        return False
    
    def _apply_pit_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
    @property
    def monthly_stock_pools(self):
        """Compatibility alias; both names refer to the same pool mapping."""
        return self.stock_pools_by_period

    @monthly_stock_pools.setter
    def monthly_stock_pools(self, value):
        self.stock_pools_by_period = value

    @staticmethod
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

    def _get_stock_pool_intervals(self, date_timezone=None):
        """Resolve fixed/period pools; None preserves an intentionally disabled filter."""
        if self.use_dynamic_pool:
            if not self.stock_pools_by_period:
                return None
            return FactorBase._parse_stock_pool_intervals(
                self.stock_pools_by_period, date_timezone=date_timezone,
            )
        # Unfiltered/PIT replay temporarily disables a configured period pool.
        if getattr(self, 'stock_pools_by_period', None) is not None:
            return None
        stocks = getattr(self, 'stock_pool', None)
        # No date bounds: fixed membership also covers warmup history.
        return [(None, None, stocks)] if stocks else None

    def _filter_data_by_pools(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter raw rows by the same pool intervals used for factor results."""
        pool_intervals = FactorBase._get_stock_pool_intervals(self)
        if pool_intervals is None:
            return df
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='raise')
        if df['date'].dt.tz is not None:
            pool_intervals = FactorBase._get_stock_pool_intervals(self, df['date'].dt.tz)

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

    def _apply_dynamic_pool_to_factor_df(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Keep all input dates and columns; mask stocks outside each pool interval."""
        pool_intervals = FactorBase._get_stock_pool_intervals(self)
        if pool_intervals is None:
            return factor_df
        if not factor_df.empty and not isinstance(factor_df.index, pd.DatetimeIndex):
            factor_df = factor_df.copy()
            factor_df.index = pd.to_datetime(factor_df.index, errors='raise')
        if getattr(factor_df.index, 'tz', None) is not None:
            pool_intervals = FactorBase._get_stock_pool_intervals(self, factor_df.index.tz)
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
        connection = self._duckdb_adapter._get_connection() if self._duckdb_adapter is not None else self.db_connection
        import duckdb
        if not isinstance(connection, duckdb.DuckDBPyConnection):
            raise TypeError("Supply a DuckDB connection, duckdb_path, Parquet directory, or data binding")
        quote = lambda value: '"' + value.replace('"', '""') + '"'
        config = self._get_local_data_config()
        dataset = config.price(self.data_frequency) if is_price else config.dataset(
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
        if is_price and self.pit_enabled and self.pit_factor_column not in selected:
            selected.append(self.pit_factor_column)
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
        if is_price_table and self.data_frequency == 'daily' and 'money' in query_fields and 'paused' not in query_fields:
            query_fields.append('paused')
        
        if is_price_table:
            # 价格数据查询
            df = self._duckdb_adapter.fetch_price_data(
                stocks=self.all_stocks,
                start_date=self.start_date,
                end_date=self.end_date,
                fields=query_fields,
                frequency=self.data_frequency
            )
            from utils.pit import normalize_price_money
            df = normalize_price_money(df, source='local', frequency=self.data_frequency)
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
    
    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算因子值的接口方法
        
        这是一个需要在子类中重写的方法，用于实现具体的因子计算逻辑。
        
        Args:
            raw_data: 包含原始数据的DataFrame
            
        Returns:
            返回日期×股票的数值 DataFrame，日期索引名称为 date，股票代码为字符串。
            无结果时显式返回空 DataFrame；None、Series、数组及标量不是合法输出。
            
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError("子类必须实现calculate_factor方法")
    
    def _get_raw_data_for_factor(self) -> pd.DataFrame:
        """获取用于计算因子的原始数据
        
        子类应该重写这个方法来获取特定因子所需的数据
        
        Returns:
            用于计算因子的原始数据DataFrame
        
        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError("子类必须实现_get_raw_data_for_factor方法")
    
    @staticmethod
    def _normalize_factor_dataframe(values: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Preserve keyed outputs; only legacy single-column outputs may use raw row order.

        Invalid or ambiguous nonempty tables raise ValueError instead of losing
        date/stock labels or masquerading as an empty calculation.
        """
        if not isinstance(values, pd.DataFrame):
            raise ValueError("因子输出必须是日期×股票 DataFrame")
        if values.empty:
            return values
        if not values.columns.is_unique:
            raise ValueError("因子输出列标签重复")

        if {'date', 'code', 'factor'}.issubset(values.columns):
            if len(values.columns) != 3:
                raise ValueError("因子长表只能包含 date、code、factor 三列")
            long = values.copy()
            if pd.api.types.is_numeric_dtype(long['date'].dtype):
                raise ValueError("因子长表日期不能是数值行号")
            long['date'] = pd.to_datetime(long['date'], errors='raise')
            if long[['date', 'code']].isna().any().any():
                raise ValueError("因子长表包含空日期或股票代码")
            values = long.pivot(index='date', columns='code', values='factor')
        elif values.index.inferred_type in {'integer', 'floating', 'mixed-integer', 'boolean', 'complex'}:
            # The old N×1 form is positional only when its row index is unchanged.
            if (values.index.name == 'date' or values.shape[1] != 1
                    or not values.index.is_unique or not values.index.equals(raw_data.index)):
                raise ValueError("无法识别因子 DataFrame：需要日期索引，或与原始行索引一致的单列值")
            values = pd.DataFrame({
                'date': pd.to_datetime(raw_data['date'], errors='raise'),
                'code': raw_data['code'],
                'factor': values.iloc[:, 0].to_numpy(),
            }).pivot(index='date', columns='code', values='factor')

        if isinstance(values.index, pd.MultiIndex):
            raise ValueError("因子宽表需要单层日期索引")
        if not all(isinstance(code, str) and code for code in values.columns):
            raise ValueError("因子宽表的股票代码必须是非空字符串")
        dates = pd.to_datetime(values.index, errors='raise')
        if dates.hasnans or not dates.is_unique:
            raise ValueError("因子宽表日期必须非空且唯一")
        if not isinstance(values.index, pd.DatetimeIndex) or values.index.name != 'date':
            values = values.copy(deep=False)
            values.index = dates.rename('date')
        return values

    def get_factor_values(self, force_recalculate: bool = False) -> pd.DataFrame:
        """获取因子值
        
        Args:
            force_recalculate: 是否强制重新计算
            
        Returns:
            因子值DataFrame，行为日期，列为股票代码
        """
        # Read first: mutable files and borrowed connections have no immutable
        # version token. Cache calculations only after identifying actual inputs.
        raw_data = self._get_raw_data_for_factor()
        cache_key = self._get_cache_key(raw_data) if self.cache_dir else None
        if not force_recalculate and cache_key:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                self.logger.info(f"从缓存加载因子 {self.factor_name}")
                return cached_result

        # 暴露最近一次用于计算的原始输入数据，便于外部诊断/对比（例如 jq_program/verify_data.py）
        # 不影响正常计算逻辑，仅作为调试接口。
        try:
            self.last_raw_data = raw_data
        except Exception:
            # 属性写入失败不应影响因子计算
            pass
        if raw_data.empty:
            self.logger.warning("原始数据为空，返回空DataFrame")
            return pd.DataFrame()
        
        # PIT 动态前复权
        if self.pit_enabled:
            raw_data = self._apply_pit_adjustment(raw_data)
        
        # 计算因子值
        factor_values = self.calculate_factor(raw_data)
        
        factor_values = FactorBase._normalize_factor_dataframe(factor_values, raw_data)
        if factor_values.empty:
            self.logger.warning("因子计算返回空DataFrame")
            return pd.DataFrame()

        # 如果使用动态股票池且配置为在结果阶段应用股票池过滤，
        # 则在因子结果上按每期股票池做截面过滤。
        if (
            self.use_dynamic_pool
            and self.stock_pools_by_period
            and self.dynamic_pool_mode in ('factor', 'both')
        ):
            factor_values = self._apply_dynamic_pool_to_factor_df(factor_values)
        
        # 如果设置了缓存目录，保存计算结果
        if self.cache_dir:
            try:
                self._save_to_cache(factor_values, cache_key)
            except Exception as e:
                self.logger.warning(f"保存缓存失败: {e}")
            
        return factor_values
    
    def get_factor_values_unfiltered(self, force_recalculate: bool = False) -> pd.DataFrame:
        """在当前股票范围内计算因子值，跳过按日期的动态股票池过滤。

        保持当前 all_stocks：通常为动态池各期成员的并集或固定股票列表，
        分块调用时为当前分块的股票。不会扩展为全市场股票。读取和计算仍
        遵循既定数据窗口，可保留成员入池前或离池后的观测。

        本基础实现每次取数并计算，不读写磁盘因子结果缓存。

        Args:
            force_recalculate: 保留调用接口；本基础实现不使用此参数，
                True/False 均执行计算，也不强制刷新底层取数缓存。

        Returns:
            日期×股票的因子值 DataFrame，不按动态池成员有效期过滤；
            实际行列取决于当前股票范围、可用数据及因子计算结果。
        """
        started = time.perf_counter()
        self.last_performance_metrics = {}
        # 保存原有的 all_stocks，稍后恢复
        original_all_stocks = self.all_stocks
        original_use_dynamic_pool = self.use_dynamic_pool
        original_dynamic_pool_mode = self.dynamic_pool_mode
        
        try:
            # 临时禁用动态股票池过滤
            self.use_dynamic_pool = False
            self.dynamic_pool_mode = 'factor'  # 不在原始数据阶段过滤
            
            # 保持当前股票范围，包括固定池或分块调用传入的股票列表
            self.logger.info(f"get_factor_values_unfiltered: 使用当前股票范围 ({len(self.all_stocks)} 只股票)")
            
            # 在当前 all_stocks 范围内获取原始数据，不扩展股票列表
            fetch_started = time.perf_counter()
            try:
                raw_data = self._get_raw_data_for_factor()
            finally:
                self.last_performance_metrics['raw_fetch_seconds'] = time.perf_counter() - fetch_started
            self.last_performance_metrics.update({
                'raw_rows': len(raw_data),
                'raw_fields': list(raw_data.columns),
                'raw_memory_mb': float(raw_data.memory_usage(deep=True).sum() / 1024**2),
            })
            
            if raw_data.empty:
                self.logger.warning("原始数据为空，返回空DataFrame")
                return pd.DataFrame()
            
            # PIT 动态前复权
            if self.pit_enabled:
                pit_started = time.perf_counter()
                try:
                    raw_data = self._apply_pit_adjustment(raw_data)
                finally:
                    self.last_performance_metrics['pit_seconds'] = time.perf_counter() - pit_started
            
            # 计算因子值
            calculate_started = time.perf_counter()
            try:
                factor_values = self.calculate_factor(raw_data)
            finally:
                self.last_performance_metrics['calculate_seconds'] = time.perf_counter() - calculate_started
            
            factor_values = FactorBase._normalize_factor_dataframe(factor_values, raw_data)
            if factor_values.empty:
                self.logger.warning("因子计算返回空DataFrame")
                return pd.DataFrame()

            self.logger.info(f"get_factor_values_unfiltered 返回因子形状: {factor_values.shape} (未应用动态股票池过滤)")
            return factor_values
            
        finally:
            # 恢复原有设置
            self.all_stocks = original_all_stocks
            self.use_dynamic_pool = original_use_dynamic_pool
            self.dynamic_pool_mode = original_dynamic_pool_mode
            self.last_performance_metrics['total_seconds'] = time.perf_counter() - started

    def get_factor_values_unfiltered_chunked(self, chunk_size: int) -> pd.DataFrame:
        """Stock chunks require an explicit independence contract and a live-data check.

        The concrete class must define STOCK_CHUNK_INDEPENDENT = True itself:
        changing the input stock set must not change any retained stock's values,
        dates, or availability. Inherited declarations do not opt subclasses in.
        Unknown dependencies and overridden entry points use a full calculation.
        """
        stocks = sorted(set(self.all_stocks or []))
        if self.data_frequency != '5m' or chunk_size <= 0 or len(stocks) <= chunk_size:
            return self.get_factor_values_unfiltered()
        if (self.__class__.__dict__.get('STOCK_CHUNK_INDEPENDENT') is not True
                or self.__class__.get_factor_values_unfiltered is not FactorBase.get_factor_values_unfiltered):
            self.last_performance_metrics = {}
            result = self.get_factor_values_unfiltered()
            self.last_performance_metrics['stock_chunk_mode'] = 'fallback_full'
            return result

        original_stocks = self.all_stocks
        started = time.perf_counter()
        probe_count = min(len(stocks), chunk_size * 2)
        probe_stocks = stocks[:probe_count]
        try:
            try:
                self.all_stocks = probe_stocks
                probe_full = self.get_factor_values_unfiltered(force_recalculate=True)
                if probe_full.empty:
                    raise ValueError("股票探针无有效截面，无法证明股票分块等价")
                probe_chunks = []
                for offset in range(0, probe_count, chunk_size):
                    self.all_stocks = probe_stocks[offset:offset + chunk_size]
                    probe_chunks.append(self.get_factor_values_unfiltered(force_recalculate=True))
                probe_chunked = pd.concat(probe_chunks, axis=1).sort_index().sort_index(axis=1)
                pd.testing.assert_frame_equal(
                    probe_full.sort_index().sort_index(axis=1),
                    probe_chunked,
                    check_exact=True,
                    check_dtype=False,
                )
            except (AssertionError, ValueError) as exc:
                self.logger.warning("股票分块等价检查失败，回退完整计算: %s", exc)
                self.all_stocks = original_stocks
                result = self.get_factor_values_unfiltered(force_recalculate=True)
                self.last_performance_metrics['stock_chunk_mode'] = 'fallback_full'
                return result

            results = [probe_full]
            for offset in range(probe_count, len(stocks), chunk_size):
                self.all_stocks = stocks[offset:offset + chunk_size]
                results.append(self.get_factor_values_unfiltered(force_recalculate=True))
            result = pd.concat(results, axis=1).sort_index().sort_index(axis=1)
            self.last_performance_metrics.update({
                'stock_chunk_mode': 'verified',
                'stock_chunk_size': chunk_size,
                'stock_chunk_seconds': time.perf_counter() - started,
            })
            return result
        finally:
            self.all_stocks = original_stocks

    def _get_factor_values_from_prepared_stock_chunks(
        self,
        chunk_size: int,
        prepare_chunk: Callable[[pd.DataFrame], pd.DataFrame],
        calculate_prepared: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        mode: str = "two_stage_intraday",
        min_total_rows: int = 0,
    ) -> pd.DataFrame:
        """Reduce stock chunks, then calculate once if total input rows suffice.

        min_total_rows applies to all raw rows, never to an individual chunk.
        The full-calculation entry must enforce the same admission threshold.
        """
        stocks = sorted(set(self.all_stocks or []))
        if chunk_size <= 0 or len(stocks) <= chunk_size:
            return self.get_factor_values_unfiltered()

        original_stocks = self.all_stocks
        original_use_dynamic_pool = self.use_dynamic_pool
        original_dynamic_pool_mode = self.dynamic_pool_mode
        prepared_parts = []
        raw_rows = 0
        prepared_rows = 0
        peak_raw_memory_mb = 0.0
        fetch_seconds = 0.0
        prepare_seconds = 0.0
        pit_seconds = 0.0
        calculate_seconds = 0.0
        started = time.perf_counter()
        self.last_performance_metrics = {}

        try:
            self.use_dynamic_pool = False
            self.dynamic_pool_mode = "factor"
            for offset in range(0, len(stocks), chunk_size):
                self.all_stocks = stocks[offset:offset + chunk_size]
                fetch_started = time.perf_counter()
                try:
                    raw_data = self._get_raw_data_for_factor()
                finally:
                    fetch_seconds += time.perf_counter() - fetch_started
                if raw_data.empty:
                    continue

                raw_rows += len(raw_data)
                peak_raw_memory_mb = max(
                    peak_raw_memory_mb,
                    float(raw_data.memory_usage(deep=True).sum() / 1024**2),
                )
                if self.pit_enabled:
                    pit_started = time.perf_counter()
                    try:
                        raw_data = self._apply_pit_adjustment(raw_data)
                    finally:
                        pit_seconds += time.perf_counter() - pit_started

                prepare_started = time.perf_counter()
                try:
                    prepared = prepare_chunk(raw_data)
                finally:
                    prepare_seconds += time.perf_counter() - prepare_started
                if not prepared.empty:
                    prepared_rows += len(prepared)
                    prepared_parts.append(prepared)
                del raw_data, prepared
                gc.collect()

            if raw_rows < min_total_rows or not prepared_parts:
                return pd.DataFrame()
            prepared_all = pd.concat(prepared_parts, ignore_index=True)
            self.all_stocks = original_stocks
            calculate_started = time.perf_counter()
            try:
                return calculate_prepared(prepared_all)
            finally:
                calculate_seconds = time.perf_counter() - calculate_started
        finally:
            self.all_stocks = original_stocks
            self.use_dynamic_pool = original_use_dynamic_pool
            self.dynamic_pool_mode = original_dynamic_pool_mode
            self.last_performance_metrics = {
                "stock_chunk_mode": mode,
                "stock_chunk_size": chunk_size,
                "raw_rows": raw_rows,
                "prepared_rows": prepared_rows,
                "peak_chunk_raw_memory_mb": peak_raw_memory_mb,
                "raw_fetch_seconds": fetch_seconds,
                "pit_seconds": pit_seconds,
                "prepare_seconds": prepare_seconds,
                "calculate_seconds": calculate_seconds,
                "total_seconds": time.perf_counter() - started,
            }

    def _get_factor_values_by_rebalance_pool(
        self,
        window_bars: int,
        warmup_calendar_days: int = 0,
        use_last_available: bool = False,
        combine_duplicate_dates: bool = False,
        rebalance_only: bool = True,
    ) -> pd.DataFrame:
        """Replay each PIT pool; default to the single T-1 row used by cloud rebalancing.

        IMPORTANT: ``rebalance_only=True`` intentionally emits about one row per
        weekly/monthly pool key.  It is correct for cloud/local rebalance
        alignment, but not for daily factor validation.  Daily factors must pass
        ``rebalance_only=False`` to retain every factor date in ``[key, next_key)``.
        """
        if not self.use_dynamic_pool or not self.stock_pools_by_period:
            return self.get_factor_values(force_recalculate=True)

        original_start_date = self.start_date
        original_all_stocks = self.all_stocks
        original_use_dynamic_pool = self.use_dynamic_pool
        original_dynamic_pool_mode = self.dynamic_pool_mode
        result_start = pd.to_datetime(
            getattr(self, "_factor_start_date", original_start_date)
        ).normalize()
        result_end = pd.to_datetime(
            getattr(self, "_factor_end_date", self.end_date)
        ).normalize()
        pool_key_end = pd.to_datetime(self.end_date).normalize()

        try:
            pool_periods = []
            for rebalance_key in sorted(self.stock_pools_by_period):
                rebalance_date = pd.to_datetime(rebalance_key, errors="coerce")
                if pd.isna(rebalance_date):
                    continue
                rebalance_date = rebalance_date.normalize()
                if rebalance_date > pool_key_end:
                    continue
                if rebalance_only and rebalance_date <= result_start:
                    continue
                pool_stocks = [
                    code
                    for code in (self.stock_pools_by_period.get(rebalance_key) or [])
                    if code
                ]
                # Empty pools still terminate the preceding membership interval.
                pool_periods.append((rebalance_date, pool_stocks))

            if not rebalance_only:
                carry_in = [
                    period for period in pool_periods if period[0] <= result_start
                ]
                pool_periods = carry_in[-1:] + [
                    period for period in pool_periods if period[0] > result_start
                ]

            all_pool_stocks = sorted(
                {
                    code
                    for _, stocks in pool_periods
                    for code in stocks
                }
            )
            if not all_pool_stocks:
                return pd.DataFrame()

            self.all_stocks = all_pool_stocks
            self.use_dynamic_pool = False
            self.dynamic_pool_mode = "factor"
            if warmup_calendar_days > 0:
                self.start_date = (
                    pd.to_datetime(original_start_date)
                    - pd.Timedelta(days=int(warmup_calendar_days))
                ).strftime("%Y-%m-%d")

            raw_data = self._get_raw_data_for_factor()
            if raw_data.empty:
                return pd.DataFrame()

            raw_data = raw_data.copy()
            raw_data["date"] = pd.to_datetime(raw_data["date"], errors="coerce")
            if raw_data["date"].isna().any():
                raise ValueError("rebalance raw data contains missing or invalid date values")
            if raw_data["code"].isna().any():
                raise ValueError("rebalance raw data contains missing code values")
            raw_data["code"] = raw_data["code"].astype(str)
            if raw_data["code"].str.strip().eq("").any():
                raise ValueError("rebalance raw data contains blank code values")
            raw_data = raw_data.sort_values(["date", "code"], kind="mergesort")
            available_bars = pd.DatetimeIndex(
                sorted(raw_data["date"].unique())
            )
            output_rows = []
            cumulative_pool_stocks = set()

            for period_index, (rebalance_date, pool_stocks) in enumerate(
                pool_periods
            ):
                if not pool_stocks:
                    continue
                cumulative_pool_stocks.update(pool_stocks)
                self.all_stocks = sorted(cumulative_pool_stocks)

                period_start = (
                    rebalance_date
                    if rebalance_only
                    else max(rebalance_date, result_start)
                )
                prior_bars = available_bars[available_bars < period_start]
                if len(prior_bars) < window_bars:
                    continue

                factor_date = prior_bars[-1].normalize()
                if rebalance_only and (
                    factor_date < result_start or factor_date > result_end
                ):
                    continue
                start_bar = prior_bars[-window_bars]
                period_end = (
                    pool_periods[period_index + 1][0]
                    if period_index + 1 < len(pool_periods)
                    else result_end + pd.Timedelta(days=1)
                )
                window_data = raw_data[
                    (raw_data["date"] >= start_bar)
                    & (
                        (raw_data["date"] < rebalance_date)
                        if rebalance_only
                        else (raw_data["date"] < period_end)
                    )
                    & raw_data["code"].isin(pool_stocks)
                ].copy()
                if window_data.empty:
                    continue
                if self.pit_enabled:
                    window_data = self._apply_pit_adjustment(window_data)

                factor_window = self.calculate_factor(window_data)
                factor_window = FactorBase._normalize_factor_dataframe(factor_window, window_data)
                if factor_window.empty:
                    continue
                output_dates = factor_window.index.normalize()
                if not output_dates.is_unique:
                    raise ValueError("因子输出按日期归一化后必须唯一：同一天不能有多行")
                factor_window = factor_window.set_axis(output_dates, axis=0)
                if not rebalance_only:
                    period_output = factor_window.loc[
                        (factor_window.index >= period_start)
                        & (factor_window.index < period_end)
                        & (factor_window.index <= result_end),
                        factor_window.columns.intersection(pool_stocks),
                    ]
                    if not period_output.empty:
                        output_rows.append(period_output.reindex(columns=pool_stocks))
                    continue

                output_date = factor_date
                if output_date not in factor_window.index:
                    if not use_last_available:
                        continue
                    available_outputs = factor_window.index[
                        factor_window.index <= factor_date
                    ]
                    if len(available_outputs) == 0:
                        continue
                    output_date = available_outputs.max()

                row = factor_window.loc[output_date].reindex(pool_stocks)
                row.name = output_date
                output_rows.append(row)

            if not output_rows:
                return pd.DataFrame()

            result = (
                pd.DataFrame(output_rows)
                if rebalance_only
                else pd.concat(output_rows, axis=0)
            )
            if combine_duplicate_dates:
                result = result.groupby(level=0, sort=True).first()
            else:
                result = result[~result.index.duplicated(keep="last")].sort_index()
            result = result.replace([np.inf, -np.inf], np.nan).dropna(how="all")
            return result[
                (result.index >= result_start)
                & (result.index <= result_end)
            ]
        finally:
            self.start_date = original_start_date
            self.all_stocks = original_all_stocks
            self.use_dynamic_pool = original_use_dynamic_pool
            self.dynamic_pool_mode = original_dynamic_pool_mode
    


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
    
    
    def _get_cache_inputs(self) -> Optional[dict]:
        """Declare every factor-specific parameter and additional data input.

        Implement on the concrete factor class to enable calculation caching.
        Values must be JSON parameters or top-level DataFrames. Read additional
        data before returning and calculate from that same snapshot. None means
        the dependencies cannot be identified completely, so caching is disabled.
        """
        return None

    def _get_cache_key(self, raw_data: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Hash declared inputs; never infer configuration from runtime state."""
        import hashlib
        import inspect
        from pathlib import Path

        if raw_data is None:
            return None
        # A child can add parameters/dependencies absent from its parent's hook.
        if "_get_cache_inputs" not in vars(type(self)):
            self.logger.warning("Cache disabled: %s must declare _get_cache_inputs", type(self).__name__)
            return None
        # Dependency read failures must propagate, even if an old cache exists.
        inputs = self._get_cache_inputs()
        if inputs is None:
            self.logger.warning("Cache disabled: _get_cache_inputs did not identify all inputs")
            return None

        source_revision = None
        if self._duckdb_adapter is not None:
            source_revision = self._duckdb_adapter.cache_revision()
            if source_revision is None:
                return None  # Unmanaged Parquet must not reuse derived disk caches.
        elif self.db_connection is not None:
            from utils.data_revision import connection_revision
            source_revision = connection_revision(self.db_connection)

        def frame_digest(frame):
            # Include values, ordering, dtypes, categorical metadata, axes and attrs.
            digest = hashlib.sha256()

            class HashWriter:
                def write(self, data):
                    digest.update(data)
                    return len(data)

            # Stream into the digest instead of retaining the entire pickle too.
            pickle.dump(frame, HashWriter(), protocol=4)
            return digest.hexdigest()

        try:
            if not isinstance(inputs, dict) or any(not isinstance(key, str) for key in inputs):
                raise TypeError("_get_cache_inputs must return a dictionary with string keys")
            declared = {}
            for key, value in inputs.items():
                if isinstance(value, pd.DataFrame):
                    declared[key] = ["dataframe", frame_digest(value)]
                else:
                    encoded = json.dumps(value, sort_keys=True, allow_nan=False)
                    normalized = json.loads(encoded)
                    if normalized != value:
                        raise TypeError("Cache input " + key + " must use JSON values without coercion")
                    declared[key] = ["json", normalized]
            fields = (
                "factor_name", "start_date", "end_date", "stock_pool", "all_stocks",
                "use_dynamic_pool", "stock_pools_by_period", "dynamic_pool_mode",
                "max_stocks_per_month", "sort_by", "ascending",
                "rebalance_period_months", "rebalance_schedule", "data_frequency",
                "fundamental_table", "price_table", "db_backend", "data_bindings",
                "pit_enabled", "pit_factor_column", "pit_window",
            )
            parameters = {name: getattr(self, name) for name in fields}
            parameters["pool_schedule"] = getattr(self, "pool_schedule", None)
            if self._duckdb_adapter is not None or self.db_connection is not None:
                parameters["data_access"] = self._get_local_data_config().as_dict()
                parameters["source_policy"] = self.source_policy
            for name in ("duckdb_path", "parquet_dir"):
                value = getattr(self, name)
                parameters[name] = str(Path(value).resolve()) if value is not None else None
            filters = []
            for condition in self.stock_filters or []:
                if type(condition) is not StockFilter:
                    raise TypeError("stock_filters must contain StockFilter instances")
                value = condition.value
                if condition.operator in {"in", "not_in"} and isinstance(value, (set, frozenset)):
                    value = sorted(value, key=lambda member: json.dumps(member, sort_keys=True, allow_nan=False))
                filters.append({"field": condition.field, "operator": condition.operator,
                                "value": value})
            parameters["stock_filters"] = filters
            sources = []
            implementations = [cls for cls in type(self).__mro__ if cls is not object]
            implementations.append(apply_pit_adjustment)
            for implementation in implementations:
                source_file = inspect.getsourcefile(implementation)
                if not source_file:
                    raise ValueError("Cannot identify source for " + implementation.__qualname__)
                sources.append(hashlib.sha256(Path(source_file).read_bytes()).hexdigest())
            identity = json.dumps({"schema": 3, "class": type(self).__qualname__,
                                   "source": sources, "parameters": parameters,
                                   "runtime": [pd.__version__, np.__version__],
                                   "inputs": declared, "source_revision": source_revision,
                                   "raw_data": frame_digest(raw_data)},
                                  sort_keys=True, allow_nan=False)
            return "factor-v3-" + hashlib.sha256(identity.encode()).hexdigest()
        except (TypeError, ValueError, OSError, pickle.PicklingError) as exc:
            self.logger.warning("Cache disabled: %s", exc)
            return None

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """完整写入临时文件后原子替换缓存，失败时保留原文件。
        
        Args:
            data: 要缓存的数据
            cache_key: 缓存键
        """
        if not self.cache_dir or not cache_key:
            return
            
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with tempfile.TemporaryDirectory(prefix=".factor-cache-", dir=self.cache_dir) as temporary_dir:
                temporary_path = os.path.join(temporary_dir, "data.pkl")
                with open(temporary_path, 'wb') as f:
                    pickle.dump(data, f)
                os.replace(temporary_path, cache_path)
            self.logger.debug(f"数据已缓存到: {cache_path}")
        except Exception as e:
            self.logger.warning(f"缓存数据失败: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            已规范化的因子宽表；缓存不存在、不可读或结构无效时返回None
        """
        if not self.cache_dir or not cache_key:
            return None
            
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("缓存结果必须是非空因子 DataFrame")
            if (not isinstance(data.index, pd.DatetimeIndex) or data.index.name != 'date'
                    or data.index.hasnans or not data.index.is_unique):
                raise ValueError("缓存结果必须使用名为 date 的非空且唯一的日期索引")
            if (not data.columns.is_unique
                    or not all(isinstance(code, str) and code for code in data.columns)):
                raise ValueError("缓存结果股票列必须是唯一的非空字符串")
            self.logger.debug(f"从缓存加载数据: {cache_path}")
            return data
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {e}")
            return None

    def get_stock_pool_for_date(self, date: str) -> List[str]:
        """Return the pool covering a date; missing months do not carry forward."""
        pool_intervals = FactorBase._get_stock_pool_intervals(self)
        if pool_intervals is None or pool_intervals[0][0] is None:
            return self.stock_pool or []
        target_date = pd.to_datetime(date, errors='raise')
        if target_date.tz is not None:
            pool_intervals = FactorBase._get_stock_pool_intervals(self, target_date.tz)
        for start_date, end_date, pool_stocks in pool_intervals:
            if target_date >= start_date and (end_date is None or target_date < end_date):
                return pool_stocks
        return []
    
    @staticmethod
    def vectorized_rolling_window(data: np.ndarray, window_size: int, func: Callable):
        """Apply a scalar callback to read-only windows of a one-dimensional array.

        The window must be a positive integer. Incomplete leading windows
        produce NaN; callback exceptions propagate to the caller.
        """
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("data must be one-dimensional")
        if (
            isinstance(window_size, (bool, np.bool_))
            or not isinstance(window_size, (int, np.integer))
            or window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer")
        if len(data) < window_size:
            return np.full(len(data), np.nan)

        # Keep the NumPy API supported by the existing JoinQuant bundle.
        shape = (len(data) - window_size + 1, window_size)
        strides = (data.strides[0], data.strides[0])
        rolled = np.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides, writeable=False
        )
        results = np.array([func(window) for window in rolled])
        full_results = np.full(len(data), np.nan)
        full_results[window_size - 1:] = results
        return full_results

    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame操作
        
        Args:
            df: 输入DataFrame
            
        Returns:
            优化后的DataFrame
        """
        # Preserve input precision; pandas downcast rules differ across runtimes.
        # 不设置MultiIndex，保持原有的列结构
        # attrs can survive reordering; always establish the actual row order.
        if 'date' in df.columns and 'code' in df.columns:
            df = df.sort_values(['code', 'date'], kind='mergesort').reset_index(drop=True)
            df.attrs['_sorted_by_code_date'] = True
        
        return df
