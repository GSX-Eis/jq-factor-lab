import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import re
from numbers import Integral, Real
from typing import List, Dict, Optional, Callable, Any, Tuple, Union, Mapping
import duckdb

# DuckDB 高性能后端 (可选)
try:
    from .duckdb_adapter import DuckDBAdapter, get_data_adapter
    DUCKDB_ADAPTER_AVAILABLE = True
except ImportError:
    DUCKDB_ADAPTER_AVAILABLE = False
    get_data_adapter = None

from utils.data_revision import connection_revision, revision_cached


def _sql_string_literal(value) -> str:
    """Quote a string value, not a SQL identifier or a comparison expression."""
    return "'" + str(value).replace("'", "''") + "'"


class StockFilter:
    """Individual stock filter condition"""
    
    # Percentile-based operators that require special handling
    _PERCENTILE_OPERATORS = {'percentile_bottom', 'percentile_top', 'percentile_between'}
    
    def __init__(self, field: str, operator: str, value: Any, description: str = ""):
        """
        Initialize a stock filter
        
        Args:
            field: Database field name (e.g., 'market_cap', 'roa', 'pe_ratio')
            operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=', 'between', 'in', 'not_in',
                      'percentile_bottom', 'percentile_top', 'percentile_between')
            value: Filter value or tuple for 'between'/'percentile_between' operator
            description: Human readable description of the filter
        """
        self.field = field
        self.operator = operator
        self.value = value
        self.description = description or f"{field} {operator} {value}"
    
    # Fields that may be stored as strings 'true'/'false' in imported inputs.
    _BOOLEAN_FIELDS = {'is_new_stock', 'is_st', 'paused', 'is_delisted', 'is_suspended'}
    
    def _to_boolean_condition(self, value, *, negate=False) -> str:
        """Convert boolean comparison to work with both string and integer storage.
        
        Use CAST to VARCHAR so boolean-like source columns compare consistently.
        """
        if not isinstance(value, (str, Integral, np.bool_)):
            raise ValueError(f'Boolean comparison for {self.field!r} requires bool, integer 0/1 or a recognized string; floats are not supported')
        if value in (0, '0', False, 'false', 'False'):
            truth = False
        elif value in (1, '1', True, 'true', 'True'):
            truth = True
        else:
            raise ValueError(f'Boolean comparison for {self.field!r} requires a recognized true/false value')
        if negate:
            truth = not truth
        literals = "'true', '1', 'True'" if truth else "'false', '0', 'False'"
        return f"(CAST({self.field} AS VARCHAR) IN ({literals}))"
    
    def to_sql_condition(self) -> str:
        """Convert filter to SQL WHERE condition"""
        # Special handling for boolean fields (DuckDB stores as 'true'/'false' strings)
        if self.field in self._BOOLEAN_FIELDS and self.operator in ('==', '!='):
            return self._to_boolean_condition(self.value, negate=self.operator == '!=')
        
        if self.operator == '>':
            return f"{self.field} > {self.value}"
        elif self.operator == '<':
            return f"{self.field} < {self.value}"
        elif self.operator == '>=':
            return f"{self.field} >= {self.value}"
        elif self.operator == '<=':
            return f"{self.field} <= {self.value}"
        elif self.operator == '==':
            return f"{self.field} = {self.value}"
        elif self.operator == '!=':
            return f"{self.field} != {self.value}"
        elif self.operator == 'between':
            return f"{self.field} BETWEEN {self.value[0]} AND {self.value[1]}"
        elif self.operator in {'in', 'not_in'}:
            if not isinstance(self.value, (list, tuple, set, frozenset)) or not self.value:
                raise ValueError(f'{self.operator} requires a nonempty list, tuple, set or frozenset')
            values_str = ', '.join(_sql_string_literal(v) for v in self.value)
            operator = 'IN' if self.operator == 'in' else 'NOT IN'
            return f"{self.field} {operator} ({values_str})"
        elif self.operator in {'like', 'not_like'}:
            operator = 'LIKE' if self.operator == 'like' else 'NOT LIKE'
            return f"{self.field} {operator} {_sql_string_literal(self.value)}"
        elif self.operator in {'starts_with', 'not_starts_with'}:
            prefixes = self.value if isinstance(self.value, list) else [self.value]
            if not prefixes or not all(isinstance(prefix, str) for prefix in prefixes):
                raise ValueError(f'{self.operator} requires a string prefix or a nonempty list of string prefixes')
            operator = 'LIKE' if self.operator == 'starts_with' else 'NOT LIKE'
            joiner = ' OR ' if self.operator == 'starts_with' else ' AND '
            conditions = []
            for prefix in prefixes:
                pattern = prefix.replace('!', '!!').replace('%', '!%').replace('_', '!_') + '%'
                conditions.append(f"{self.field} {operator} {_sql_string_literal(pattern)} ESCAPE '!'")
            return f"({joiner.join(conditions)})" if isinstance(self.value, list) else conditions[0]
        elif self.operator in self._PERCENTILE_OPERATORS:
            # Percentile operators require special handling by DynamicStockPoolManager
            # Validation discards this placeholder; execution dispatches via is_percentile_filter().
            return f"__PERCENTILE_FILTER__{self.field}__{self.operator}__{self.value}__"
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
    
    def is_percentile_filter(self) -> bool:
        """Check if this filter is a percentile-based filter requiring special handling."""
        return self.operator in self._PERCENTILE_OPERATORS
    
    def __str__(self):
        return self.description
    
    def __repr__(self):
        return f"StockFilter(field='{self.field}', operator='{self.operator}', value={self.value})"


def apply_date_level_pool_filter_to_wide_df(
    factor_df: pd.DataFrame, 
    monthly_stock_pools: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    对宽表格式因子值应用日期级别的动态股票池过滤。
    
    对于每个日期，只保留当时在股票池中的股票的因子值，
    其他不在池中的股票该日期的因子值设为 NaN。
    
    Args:
        factor_df: 未过滤的因子值 DataFrame (index=date, columns=stock_code)
        monthly_stock_pools: 月度/周度股票池 {period_key: [stock_codes]}
        
    Returns:
        过滤后的因子值 DataFrame (不在池中的 cell 为 NaN)
        
    Example:
        >>> monthly_pools = {'2023-01-01': ['A', 'B'], '2023-02-01': ['B', 'C']}
        >>> filtered = apply_date_level_pool_filter_to_wide_df(factor_df, monthly_pools)
        # 2023-01 期间只保留 A, B 的值，C 设为 NaN
        # 2023-02 期间只保留 B, C 的值，A 设为 NaN
    """
    if factor_df.empty or not monthly_stock_pools:
        return factor_df.copy()
    
    # 创建因子值副本
    result_df = factor_df.copy()
    
    # 将股票池键解析为日期并排序
    pool_periods = sorted(monthly_stock_pools.keys())
    pool_dates = [pd.to_datetime(p) for p in pool_periods]
    
    # 预先构建日期到股票池的映射，避免重复查找
    date_to_pool: Dict[str, set] = {}
    for pool_key in pool_periods:
        date_to_pool[pool_key] = set(monthly_stock_pools[pool_key] or [])
    
    # 为每个唯一的因子日期确定适用的股票池
    unique_dates = pd.to_datetime(result_df.index).unique()
    
    # 创建日期范围到股票池的映射 (优化性能)
    date_pool_mapping = {}
    for idx_date in unique_dates:
        # 找到该日期对应的调仓周期（最后一个 <= idx_date 的周期）
        applicable_pool_key = None
        for pool_key, pool_dt in zip(pool_periods, pool_dates):
            if pool_dt <= idx_date:
                applicable_pool_key = pool_key
            else:
                break
        date_pool_mapping[idx_date] = applicable_pool_key
    
    # 应用过滤：使用向量化操作优化性能
    for idx_date in result_df.index:
        idx_date_normalized = pd.to_datetime(idx_date)
        applicable_pool_key = date_pool_mapping.get(idx_date_normalized)
        
        if applicable_pool_key is None:
            # 日期早于所有股票池周期，所有值设为 NaN
            result_df.loc[idx_date, :] = np.nan
        else:
            # 获取该周期的股票池
            pool_stocks = date_to_pool[applicable_pool_key]
            
            # 使用向量化操作：设置不在池中的股票为 NaN
            cols_to_nan = [col for col in result_df.columns if col not in pool_stocks]
            if cols_to_nan:
                result_df.loc[idx_date, cols_to_nan] = np.nan
    
    # 删除全为 NaN 的列（整个期间都不在任何股票池中的股票）
    result_df = result_df.dropna(axis=1, how='all')
    
    return result_df


def apply_date_level_pool_filter_to_long_df(
    df: pd.DataFrame, 
    monthly_stock_pools: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    对长表格式因子值应用日期级别的动态股票池过滤。
    
    对于每个日期，只保留当时在股票池中的股票的行，
    其他不在池中的行将被丢弃。
    
    Args:
        df: MultiIndex (date, stock_code) 格式的 DataFrame
        monthly_stock_pools: 月度/周度股票池 {period_key: [stock_codes]}
        
    Returns:
        过滤后的 DataFrame (只保留在池中的行)
        
    Example:
        >>> monthly_pools = {'2023-01-01': ['A', 'B'], '2023-02-01': ['B', 'C']}
        >>> filtered = apply_date_level_pool_filter_to_long_df(df, monthly_pools)
        # 2023-01 期间只保留 A, B 的行
        # 2023-02 期间只保留 B, C 的行
    """
    if not isinstance(monthly_stock_pools, Mapping) or not monthly_stock_pools:
        raise ValueError("Stock pool history must be a nonempty mapping; use [] for an empty period")
    
    # 按实际日期排序，保留历史 YYYY-MM 和可解析日期键。
    pool_dates = pd.DatetimeIndex([pd.to_datetime(key) for key in monthly_stock_pools])
    if pool_dates.hasnans or pool_dates.tz is not None or not pool_dates.is_unique:
        raise ValueError("Stock pool dates must be valid, timezone-naive and unique")
    periods = sorted(zip(pool_dates, monthly_stock_pools))
    pool_dates, pool_periods = zip(*periods)

    date_to_pool: Dict[str, set] = {}
    for pool_key in pool_periods:
        members = monthly_stock_pools[pool_key]
        if not isinstance(members, (list, tuple)) or any(
            not isinstance(stock, str) or not stock.strip() for stock in members
        ):
            raise ValueError(f"Stock pool {pool_key!r} must contain a list or tuple of stock codes")
        if len(set(members)) != len(members):
            raise ValueError(f"Stock pool {pool_key!r} contains duplicate stock codes")
        date_to_pool[pool_key] = set(members)

    if df.empty:
        return df.copy()

    # 获取日期和股票索引
    idx = df.index
    if isinstance(idx, pd.MultiIndex) and {'date', 'stock_code'}.issubset(idx.names):
        dates = pd.to_datetime(idx.get_level_values('date'))
        if dates.hasnans or dates.tz is not None:
            raise ValueError("Panel dates must be valid and timezone-naive")
        stocks = idx.get_level_values('stock_code').astype(str)
    else:
        raise ValueError("DataFrame索引必须是MultiIndex (date, stock_code)")
    
    # 为每个唯一日期确定适用的股票池键
    # 使用 <= 匹配：找到最后一个 rebalance_date <= factor_date（即当前生效的池）
    # 与 apply_date_level_pool_filter_to_wide_df 逻辑保持一致，避免前视偏差
    unique_dates = dates.unique()
    date_pool_mapping = {}
    for idx_date in unique_dates:
        # 找到该日期对应的调仓周期（最后一个 <= idx_date 的周期）
        applicable_pool_key = None
        for pool_key, pool_dt in zip(pool_periods, pool_dates):
            if pool_dt <= idx_date:
                applicable_pool_key = pool_key
            else:
                break
        date_pool_mapping[idx_date] = applicable_pool_key
    
    # 按原行序构造成员掩码
    def is_in_pool(date, stock):
        pool_key = date_pool_mapping.get(date)
        if pool_key is None:
            return False
        return stock in date_to_pool[pool_key]
    
    mask = np.fromiter(
        (is_in_pool(d, s) for d, s in zip(dates, stocks)),
        dtype=bool,
        count=len(stocks)
    )
    
    return df[mask]

class DynamicStockPoolManager:
    """Generate date-effective stock pools with one schedule-independent selector."""
    
    def __init__(self, db_connection, fundamental_table: str = "factor_data", price_table: str = "price_data",
                 index_constituents_table: Optional[str] = None,
                 # DuckDB 高性能后端配置
                 db_backend: str = 'duckdb',
                 duckdb_path: Optional[str] = None,
                 parquet_dir: Optional[str] = None,
                 use_duckdb_cache: bool = True,
                 duckdb_cache_size_mb: int = 2000,
                 data_bindings: Optional[dict] = None,
                 source_policy: Optional[str] = None,
                 data_config_path: Optional[str] = None,
                 pool_config_path: Optional[str] = None):
        """
        Initialize dynamic stock pool manager
        
        Args:
            db_connection: Optional direct DuckDB connection for local callers.
            fundamental_table: Name of the table containing fundamental data (default: "factor_data")
            price_table: Name of the table containing price data (default: "price_data")
            index_constituents_table: Optional explicit table override; defaults to data_config_path.
            data_config_path: Local dataset definitions, including index keys and directory.
            pool_config_path: Schedule defaults and selection timing rules; defaults to config/stock_pool.json.
            db_backend: 数据库后端；仅支持 'duckdb'，旧配置中的 'auto' 会归一化为 'duckdb'
            duckdb_path: DuckDB 数据库路径
            parquet_dir: Parquet 数据目录路径
            use_duckdb_cache: 是否使用 DuckDB 缓存
            duckdb_cache_size_mb: DuckDB 缓存大小 (MB)
        """
        if db_backend == 'auto':
            db_backend = 'duckdb'
        if db_backend != 'duckdb':
            raise ValueError("Only DuckDB is supported; set db_backend to 'duckdb'")
        if db_connection is not None and not isinstance(db_connection, duckdb.DuckDBPyConnection):
            raise TypeError("db_connection must be a DuckDBPyConnection")
        self.db_connection = db_connection
        self.fundamental_table = fundamental_table
        self.price_table = price_table
        self.index_constituents_table = index_constituents_table
        self.logger = logging.getLogger("DynamicStockPool")
        
        # Cache for monthly stock pools
        self._pool_cache = {}
        from utils.stock_pool_schedule import load_pool_policy
        self.pool_policy = load_pool_policy(pool_config_path)
        # Cache for table columns
        self._columns_cache = None
        self._price_columns_cache = None
        self._index_constituents_cache = {}
        
        self.db_backend = db_backend
        self.duckdb_path = duckdb_path
        self.parquet_dir = parquet_dir
        self.source_policy = source_policy
        self.data_config_path = data_config_path
        from utils.data_store import binding_contract
        self.data_bindings = binding_contract(data_bindings)
        self._duckdb_adapter = None
        self._duckdb_conn = None
        self._init_duckdb(
            duckdb_path, parquet_dir, use_duckdb_cache, duckdb_cache_size_mb, db_connection
        )

        from utils.data_access_config import load_data_access_config
        self.data_config = (self._duckdb_adapter.data_config if self._duckdb_adapter is not None
                            else load_data_access_config(data_config_path))
        if self.index_constituents_table is None and self.data_config.index_constituents is not None:
            self.index_constituents_table = self.data_config.index_constituents.table

    def _init_duckdb(self, duckdb_path: Optional[str], parquet_dir: Optional[str],
                     use_duckdb_cache: bool, cache_size_mb: int, direct_connection) -> None:
        """Initialize the configured DuckDB source without a backend fallback."""
        if (direct_connection is not None and not self.data_bindings
                and duckdb_path is None and parquet_dir is None and self.source_policy is None):
            self._duckdb_conn = direct_connection
            return
        if not get_data_adapter:
            raise ImportError("DuckDB adapter is unavailable")
        self._duckdb_adapter = get_data_adapter(
            duckdb_path=duckdb_path,
            parquet_dir=parquet_dir,
            use_cache=use_duckdb_cache,
            cache_size_mb=cache_size_mb,
            data_bindings=self.data_bindings,
            source_policy=self.source_policy,
            data_config_path=self.data_config_path,
        )
        if self._duckdb_adapter is None:
            return
        self.duckdb_path = str(self._duckdb_adapter.duckdb_path) if self._duckdb_adapter.duckdb_path else None
        self.data_bindings = self._duckdb_adapter.data_bindings
        self._duckdb_conn = self._duckdb_adapter._get_connection()
    
    def cache_revision(self):
        if self._duckdb_adapter is not None:
            revision = self._duckdb_adapter.cache_revision()
            self._duckdb_conn = self._duckdb_adapter._get_connection()
            return revision
        if self._duckdb_conn is None:
            raise RuntimeError('A local data source is required')
        try:
            return connection_revision(self._duckdb_conn)
        except duckdb.Error as exc:
            raise RuntimeError(f'Cannot validate cache for {self.fundamental_table!r}: {exc}') from exc

    def clear_cache(self):
        """Invalidate both source caches and all derived stock pools."""
        self._pool_cache.clear()
        self._index_constituents_cache.clear()
        self._columns_cache = None
        self._price_columns_cache = None
        if self._duckdb_adapter is not None:
            self._duckdb_adapter.clear_cache()

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL against the configured DuckDB source.
        
        Args:
            sql: SQL 查询语句
            
        Returns:
            查询结果 DataFrame
        """
        if self._duckdb_conn is None:
            raise RuntimeError("A DuckDB connection, path, Parquet directory, or input binding is required")
        return self._duckdb_conn.execute(sql).fetchdf()

    @revision_cached
    def _get_available_columns(self) -> List[str]:
        """Cache successful schema reads until the next committed import."""
        if self._columns_cache is not None:
            return self._columns_cache
        self._columns_cache = self._get_table_columns(self.fundamental_table)
        return self._columns_cache

    @revision_cached
    def _get_price_columns(self) -> List[str]:
        """Cache the price schema only after a successful read."""
        if self._price_columns_cache is not None:
            return self._price_columns_cache
        self._price_columns_cache = self._get_table_columns(self.price_table)
        return self._price_columns_cache

    def _get_table_columns(self, table_name: str) -> List[str]:
        """Read a required table's schema; failures must not become cached empty lists."""
        try:
            df = self._execute_sql(f"DESCRIBE SELECT * FROM {table_name}")
        except Exception as exc:
            raise RuntimeError(f"Could not read columns for table {table_name!r}: {exc}") from exc
        if df.empty or 'column_name' not in df.columns:
            raise RuntimeError(f"Invalid schema description for table {table_name!r}")
        columns = df['column_name'].tolist()
        self.logger.debug(f"Available columns in '{table_name}': {sorted(columns)}")
        return columns

    @staticmethod
    def _condition_fields(condition: StockFilter) -> set:
        """Resolve simple column comparisons, without guessing arbitrary SQL expressions."""
        required = {condition.field}
        if condition.operator in {'>', '<', '>=', '<=', '==', '!='} and isinstance(condition.value, str):
            value = condition.value
            boolean_literal = condition.field in StockFilter._BOOLEAN_FIELDS and value in {
                'true', 'false', 'True', 'False', '0', '1'
            }
            if boolean_literal or re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
                pass
            elif re.fullmatch(r"'(?:[^']|'')*'", value):
                pass  # Explicit SQL string literal, not a column reference.
            elif value.isidentifier():
                required.add(value)
            else:
                raise ValueError(f'Comparison value {value!r} must be a number, column name, or quoted literal')
        return required

    def _resolve_selection(self, filters: List[StockFilter], sort_by: Optional[str] = None):
        """Fix each condition's effective fields, source and snapshot before querying rows."""
        fund_columns = set(self._get_available_columns())
        columns = {'fundamental': fund_columns}
        tables = {'fundamental': self.fundamental_table, 'price': self.price_table}
        priorities = self.pool_policy['selection']['source_priority']

        def choose_source(required, priority):
            missing = {}
            for source in priority:
                if source not in columns:
                    columns[source] = set(self._get_price_columns())
                absent = required - columns[source]
                if not absent:
                    return source
                missing[tables[source]] = sorted(absent)
            raise ValueError(f'No single stock pool source supplies fields {sorted(required)}; missing: {missing}')

        plans = []
        for original in filters:
            if original.field == 'index_code':
                continue
            percentile = original.is_percentile_filter()
            fundamental = self._is_fundamental_field(original.field) and (percentile or original.field in fund_columns)
            kind = 'percentile' if percentile else ('fundamental' if fundamental else 'state')
            field = (self.pool_policy['selection']['state_field_aliases'].get(original.field, original.field)
                     if kind == 'state' else original.field)
            condition = StockFilter(field, original.operator, original.value, original.description)
            required = self._condition_fields(condition)
            priority = ['fundamental'] if fundamental else priorities[kind]
            source = choose_source(required, priority)
            plans.append({'original': original, 'condition': condition, 'kind': kind,
                          'table': tables[source], 'required_fields': required,
                          'use_prior': fundamental or (percentile and source == 'fundamental')})
        sort_table = tables[choose_source({sort_by}, priorities['sort'])] if sort_by else None
        return plans, sort_table

    def _read_index_constituents(self, index_code=None, date=None, *, exists_only=False):
        if self._duckdb_adapter is not None:
            return self._duckdb_adapter._read_index_constituents(
                index_code, date, table_name=self.index_constituents_table, exists_only=exists_only)
        if self._duckdb_conn is None:
            raise RuntimeError('A local data source is required for index membership')
        from utils.duckdb_adapter import _query_index_constituents
        dataset, column = self.data_config.index_dataset(self.index_constituents_table)
        return _query_index_constituents(self._duckdb_conn, dataset, column, index_code, date,
                                         exists_only=exists_only)

    @revision_cached
    def _get_index_constituents(self, target_date: str, index_code: str) -> set:
        """Return exact-date membership; never cache query failures or empty results."""
        from utils.duckdb_adapter import _validate_index_request
        _validate_index_request(index_code, target_date)
        dataset, column = self.data_config.index_dataset(self.index_constituents_table)
        cache_key = (dataset, column, target_date, index_code)
        if cache_key in self._index_constituents_cache:
            return set(self._index_constituents_cache[cache_key])
        constituents = set(self._read_index_constituents(index_code, target_date))
        if constituents:
            self._index_constituents_cache[cache_key] = frozenset(constituents)
        return constituents

    def _has_index_filter(self, filters: List[StockFilter]) -> Tuple[bool, Optional[str]]:
        """
        Check if filters contain an index_code filter.
        
        Returns:
            Tuple of (has_filter, index_code). If has_filter is True, index_code is the value.
        """
        matches = [condition for condition in filters if condition.field == 'index_code']
        if not matches:
            return False, None
        if len(matches) != 1 or matches[0].operator != '==':
            raise ValueError('Use exactly one index_code == filter')
        value = matches[0].value
        from utils.duckdb_adapter import _validate_index_code
        _validate_index_code(value)
        return True, value
    
    def _filter_by_index(self, stock_list: List[str], target_date: str, index_code: str) -> List[str]:
        """
        Filter stock list to only include stocks in the specified index.
        
        Args:
            stock_list: List of stock codes to filter
            target_date: Date for index constituents lookup
            index_code: Index code
            
        Returns:
            Filtered list of stock codes
        """
        constituents = self._get_index_constituents(target_date, index_code)
        filtered = [s for s in stock_list if s in constituents]
        self.logger.debug(f"Index filter: {len(stock_list)} -> {len(filtered)} stocks (index: {index_code})")
        return filtered
    
    def add_basic_filters(self) -> List[StockFilter]:
        """Add common basic filters for stock selection"""
        return [
            StockFilter("is_st", "==", 0, "Exclude ST stocks"),
            StockFilter("is_delisted", "==", 0, "Exclude delisted stocks"),
        ]
    
    def _save_filter_log(self, filter_log: dict) -> None:
        """Write the unified selection log when debug output and its directory are bound."""
        from utils.output_paths import open_diagnostic
        if os.environ.get("QUANT_DEBUG_OUTPUTS") != "1":
            return
        try:
            with open_diagnostic("pool_filter_log.txt", "a", enabled=True) as f:
                f.write(f"\n{'='*70}\n")
                
                f.write(f"TRADING_DATE: {filter_log['trading_date']} (FUNDAMENTAL_DATE: {filter_log['fundamental_date']})\n")
                
                f.write(f"INITIAL_STOCKS: {filter_log.get('initial_count', 0)}\n")
                
                for step in filter_log.get('filter_steps', []):
                    removed_list = step['removed_samples']
                    removed_samples = ','.join(removed_list) if removed_list else 'none'
                    
                    filter_type = step.get('type', '')
                    date_used = step.get('date_used', '')
                    type_info = f" [{filter_type}@{date_used}]" if filter_type else ""
                    
                    f.write(f"FILTER: {step.get('filter', 'unknown')}{type_info}\n")
                    f.write(f"  ORIGINAL_FILTER: {step['original_filter']}\n")
                    f.write(f"  TABLE: {step['table_used']}\n")
                    if step['type'] == 'percentile':
                        f.write(f"  THRESHOLD: {step['threshold']} LOW: {step['threshold_low']} HIGH: {step['threshold_high']}\n")
                    f.write(f"  {step.get('before', 0)} -> {step.get('after', 0)} (removed {step.get('removed', 0)})\n")
                    if step.get('removed', 0) > 0:
                        f.write(f"  REMOVED_SAMPLES ({len(removed_list)} of {step['removed']}): {removed_samples}\n")
                
                f.write(f"FINAL_STOCKS: {filter_log.get('final_count', 0)}\n")
        except Exception as e:
            self.logger.warning(f"Failed to save filter log: {e}")
    
    def _trading_calendar(self):
        from utils.trading_calendar import load_trading_calendar
        return load_trading_calendar(self._duckdb_conn)

    def _is_fundamental_field(self, field: str) -> bool:
        """Use the configured prior-snapshot field rules for every frequency."""
        policy = self.pool_policy['selection']
        return field in policy['fundamental_fields'] or field.startswith(tuple(policy['fundamental_prefixes']))

    def _get_previous_data_date(self, reference_date: Union[str, datetime]) -> Optional[str]:
        """
        Get last available date in fundamental_table strictly before reference_date.
        This is the latest table snapshot, not necessarily the previous trading session.
        Returns date string 'YYYY-MM-DD' or None.
        """
        if isinstance(reference_date, str):
            ref_str = reference_date
        else:
            ref_str = reference_date.strftime('%Y-%m-%d')

        sql = f"""
        SELECT MAX(date) AS date
        FROM {self.fundamental_table}
        WHERE date < '{ref_str}'
        """
        df = self._execute_sql(sql)
        if not df.empty and pd.notna(df.iloc[0, 0]):
            return pd.Timestamp(df.iloc[0, 0]).strftime('%Y-%m-%d')
        return None

    def _apply_percentile_filter(
        self, 
        date_str: str, 
        current_stocks: set,
        filter_obj: StockFilter,
        table: str = None
    ) -> Tuple[set, Dict[str, Any]]:
        """
        Apply percentile-based filter by:
        1. Query field values for all stocks on this date
        2. Calculate percentile thresholds
        3. Filter stocks within the percentile range
        
        Args:
            date_str: Date string in 'YYYY-MM-DD' format
            current_stocks: Set of stock codes to filter
            filter_obj: StockFilter with percentile operator
            table: Table to query (defaults to fundamental_table)
            
        Returns:
            Tuple of (filtered_stocks_set, filter_info_dict)
            filter_info_dict contains: threshold, threshold_low, threshold_high
        """
        if table is None:
            table = self.fundamental_table
            
        field = filter_obj.field
        operator = filter_obj.operator
        value = filter_obj.value
        
        filter_info = {
            'threshold': None,
            'threshold_low': None,
            'threshold_high': None
        }
        
        if not current_stocks:
            return current_stocks, filter_info
        
        # Fetch field values for all stocks on this date
        stocks_str = ', '.join(_sql_string_literal(stock) for stock in current_stocks)
        sql = f"""
            SELECT stock_code, {field} 
            FROM {table}
            WHERE date = '{date_str}' 
            AND stock_code IN ({stocks_str})
            AND {field} IS NOT NULL
        """
        
        df = self._execute_sql(sql)
        if df.empty:
            raise ValueError(f"No values available for percentile field {field!r} on {date_str}")
        if not np.isfinite(pd.to_numeric(df[field], errors='raise')).all():
            raise ValueError(f"Non-finite percentile values for {field!r} on {date_str}")
        
        # Calculate percentile thresholds based on operator
        if operator == 'percentile_bottom':
            # Select bottom X% (smallest values)
            threshold = np.percentile(df[field].dropna(), value)
            passed_df = df[df[field] <= threshold]
            filter_info['threshold'] = threshold
            self.logger.debug(f"Percentile bottom {value}%: threshold={threshold:.2f}, passed={len(passed_df)}/{len(df)}")
            
        elif operator == 'percentile_top':
            # Select top X% (largest values)
            threshold = np.percentile(df[field].dropna(), 100 - value)
            passed_df = df[df[field] >= threshold]
            filter_info['threshold'] = threshold
            self.logger.debug(f"Percentile top {value}%: threshold={threshold:.2f}, passed={len(passed_df)}/{len(df)}")
            
        elif operator == 'percentile_between':
            # Select stocks between X% and Y% percentiles
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"percentile_between requires [low_pct, high_pct], got {value}")
            
            low_pct, high_pct = value
            low_thresh = np.percentile(df[field].dropna(), low_pct)
            high_thresh = np.percentile(df[field].dropna(), high_pct)
            passed_df = df[(df[field] >= low_thresh) & (df[field] <= high_thresh)]
            filter_info['threshold_low'] = low_thresh
            filter_info['threshold_high'] = high_thresh
            self.logger.debug(f"Percentile between {low_pct}%-{high_pct}%: thresholds=[{low_thresh:.2f}, {high_thresh:.2f}], passed={len(passed_df)}/{len(df)}")
        else:
            raise ValueError(f"Unknown percentile operator: {operator}")
        
        passed_stocks = set(passed_df['stock_code'].tolist())
        return passed_stocks, filter_info

    def get_weekly_run_dates(self, start_date: datetime, end_date: datetime, target_day_index: int = 3) -> List[datetime]:
        """Existing alignment callers retain query-window weekly date semantics."""
        if isinstance(target_day_index, bool) or not isinstance(target_day_index, int) or target_day_index < 1:
            raise ValueError('target_day_index must be a positive integer')
        from utils.stock_pool_schedule import resolve_schedule, generate_rebalance_dates
        spec = resolve_schedule(self.pool_policy, legacy='weekly')
        spec['session'] = target_day_index
        return generate_rebalance_dates(self._trading_calendar(), start_date, end_date, spec).to_pydatetime().tolist()

    def _validate_selection(self, filters, max_stocks=None):
        if max_stocks is not None and (isinstance(max_stocks, bool) or not isinstance(max_stocks, int) or max_stocks < 1):
            raise ValueError('max_stocks must be a positive integer or None')
        self._has_index_filter(filters)
        for condition in filters:
            if not isinstance(condition.field, str) or not condition.field:
                raise ValueError('Filter field must be a nonempty string')
            if condition.field == 'index_code':
                continue
            value = condition.value
            if condition.operator in {'between', 'percentile_between'}:
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise ValueError(f'{condition.operator} requires two bounds')
                if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in value) or value[0] > value[1]:
                    raise ValueError('Filter bounds must be finite numbers in ascending order')
            if condition.is_percentile_filter():
                values = value if condition.operator == 'percentile_between' else [value]
                if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or not 0 <= v <= 100 for v in values):
                    raise ValueError('Percentiles must be finite numbers in [0, 100]')
            elif isinstance(value, Real) and not np.isfinite(value):
                raise ValueError('Filter value must be finite')
            condition.to_sql_condition()

    @revision_cached
    def _select_stock_pool(self, trading_date: datetime, filters: List[StockFilter],
                           max_stocks: Optional[int] = None, sort_by: Optional[str] = None,
                           ascending: bool = False) -> List[str]:
        """Select on one effective date; schedule-independent, using the former weekly rules.

        The prior fundamental snapshot is the latest available date strictly before T.
        The explicit default same-day fallback preserves the existing weekly contract.
        Status/index membership use T; configured state aliases preserve open-price checks.
        """
        self._validate_selection(filters, max_stocks)
        trading_date_str = trading_date.strftime('%Y-%m-%d')
        cache_key = (trading_date_str, tuple((f.field, f.operator, repr(f.value)) for f in filters),
                     max_stocks, sort_by, bool(ascending), repr(self.pool_policy['selection']))
        if cache_key in self._pool_cache:
            return list(self._pool_cache[cache_key])
        plans, sort_table = self._resolve_selection(filters, sort_by)
        fund_date = self._get_previous_data_date(trading_date)
        if fund_date is None:
            if self.pool_policy['selection']['missing_previous'] == 'raise':
                raise ValueError(f'No fundamental snapshot before {trading_date_str}')
            fund_date = trading_date_str
        base = self._execute_sql(f"SELECT DISTINCT stock_code FROM {self.fundamental_table} WHERE date = '{fund_date}'")
        if base.empty:
            raise ValueError(f'No candidate snapshot on {fund_date}')
        current = set(base.stock_code.tolist())
        log = {'trading_date': trading_date_str, 'fundamental_date': fund_date,
               'initial_count': len(current), 'filter_steps': []}
        has_index, index_code = self._has_index_filter(filters)

        def record(condition, date, kind, before, *, table=None, original=None):
            original = original or condition
            log['filter_steps'].append({'filter': f'{condition.field} {condition.operator} {condition.value}',
                'original_filter': f'{original.field} {original.operator} {original.value}',
                'table_used': table, 'date_used': date, 'type': kind,
                'before': len(before), 'after': len(current),
                'removed': len(before-current), 'removed_samples': sorted(before-current)[:20]})

        for kind in ('fundamental', 'state'):
            for plan in (p for p in plans if p['kind'] == kind):
                if not current:
                    break
                condition, table = plan['condition'], plan['table']
                date = fund_date if plan['use_prior'] else trading_date_str
                sql_condition = condition.to_sql_condition()
                stocks = ', '.join(_sql_string_literal(stock) for stock in current)
                frame = self._execute_sql(f"SELECT DISTINCT stock_code FROM {table} WHERE date = '{date}' "
                                          f"AND stock_code IN ({stocks}) AND {sql_condition}")
                before = current
                current = set(frame.stock_code.tolist())
                record(condition, date, kind, before, table=table, original=plan['original'])
        if has_index and current:
            before = current
            current = set(self._filter_by_index(sorted(current), trading_date_str, index_code))
            record(StockFilter('index_code', '==', index_code), trading_date_str, 'index', before,
                   table=self.index_constituents_table)
        for plan in (p for p in plans if p['kind'] == 'percentile'):
            if not current:
                break
            condition, table = plan['condition'], plan['table']
            date = fund_date if plan['use_prior'] else trading_date_str
            before = current
            current, info = self._apply_percentile_filter(date, current, condition, table)
            record(condition, date, 'percentile', before, table=table, original=plan['original'])
            log['filter_steps'][-1].update({k: info[k] for k in ('threshold', 'threshold_low', 'threshold_high')})
        candidates = sorted(current)
        if sort_by and candidates:
            table = sort_table
            date = fund_date if self._is_fundamental_field(sort_by) else trading_date_str
            stocks = ', '.join(_sql_string_literal(stock) for stock in candidates)
            frame = self._execute_sql(f"SELECT stock_code, {sort_by} FROM {table} WHERE date = '{date}' "
                                      f"AND stock_code IN ({stocks})")
            if frame.empty:
                raise ValueError(f'No sort values for {sort_by!r} on {date}')
            frame = frame.drop_duplicates(subset=['stock_code'])
            candidates = frame.sort_values(by=sort_by, ascending=ascending).stock_code.tolist()
        if max_stocks is not None:
            candidates = candidates[:max_stocks]
        log['final_count'] = len(candidates)
        self._save_filter_log(log)
        self._pool_cache[cache_key] = tuple(candidates)
        return candidates

    @revision_cached
    def get_stock_pools_for_period(self, start_date: str, end_date: str,
                                  filters: List[StockFilter], max_stocks: int = None,
                                  sort_by: str = None, ascending: bool = False,
                                  lag_months: Optional[int] = None,
                                  rebalance_period_months: int = 1,
                                  rebalance_schedule: str = 'quarterly', *,
                                  schedule: Optional[dict] = None) -> Dict[str, List[str]]:
        """Return effective-date pools using one selector for every rebalance schedule.

        Existing weekly configuration retains its query-window dates. Explicit schedule
        uses complete calendar periods and, by default, includes the pool effective at
        the range start. Monthly/quarterly now use the same snapshots as weekly; the
        historical month-lag snapshot rule has been retired.
        """
        from utils.stock_pool_schedule import resolve_schedule, generate_rebalance_dates
        self._validate_selection(filters, max_stocks)
        if lag_months not in (None, 0) and not (schedule is None and rebalance_schedule == 'weekly'):
            raise ValueError('Monthly snapshot lag is retired; use schedule and the common selection policy')
        spec = resolve_schedule(self.pool_policy, schedule, legacy=rebalance_schedule, months=rebalance_period_months)
        dates = generate_rebalance_dates(self._trading_calendar(), start_date, end_date, spec)
        return {date.strftime('%Y-%m-%d'): self._select_stock_pool(date.to_pydatetime(), filters,
                max_stocks=max_stocks, sort_by=sort_by, ascending=ascending) for date in dates}



    def validate_filters(self, filters: List[StockFilter]) -> bool:
        """Validate that all filter fields exist in the database (fundamental, price, or index_constituents table)"""
        try:
            self._validate_selection(filters)
            self._resolve_selection(filters)
            for condition in filters:
                if condition.field == 'index_code':
                    self._read_index_constituents(condition.value, exists_only=True)
            return True
        except Exception as e:
            self.logger.warning(f"Could not validate filters: {e}")
            return False
    
