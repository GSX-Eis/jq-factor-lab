"""
Pool Manager Module

Dynamic stock pool management utilities for factor synthesis.
Handles stock pool filtering, computation, and skip-month logic.

Extracted from FactorSynthesizer for modularity and reuse.

Author: QuantationAI
Date: 2026-01-12
"""

import logging
from datetime import timedelta
from functools import lru_cache
from numbers import Number
from typing import Any, Dict, List, Optional, Set, Union, Mapping

import pandas as pd

from utils.dynamic_stock_pool import (
    DynamicStockPoolManager,
    StockFilter,
    apply_date_level_pool_filter_to_long_df,
)

from .calendar_skip import SkipRules, describe_skip_rules, skip_mask

logger = logging.getLogger(__name__)


# =============================================================================
# Filter Construction Utilities
# =============================================================================


def build_stock_filters(
    stock_pool_cfg: Dict[str, Any],
    price_conn: Any,
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    duckdb_path: Optional[str] = None,
    include_basic: bool = True,
) -> List[StockFilter]:
    """
    根据配置构建 StockFilter 列表
    
    Args:
        stock_pool_cfg: stock_pool 配置节
        price_conn: 价格数据库连接
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端类型
        duckdb_path: DuckDB 文件路径
        include_basic: 是否包含基础过滤器
        
    Returns:
        StockFilter 对象列表
    """
    if not isinstance(stock_pool_cfg, Mapping):
        raise ValueError("stock_pool must be a mapping")
    configured_filters = stock_pool_cfg.get('filters', [])
    if not isinstance(configured_filters, (list, tuple)):
        raise ValueError("stock_pool.filters must be a list or tuple")

    filters: List[StockFilter] = []
    for index, condition in enumerate(configured_filters):
        if not isinstance(condition, Mapping):
            raise ValueError(f"stock_pool.filters[{index}] must be a mapping")
        field = condition.get('field')
        operator = condition.get('operator')
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"stock_pool.filters[{index}].field must be a nonempty string")
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError(f"stock_pool.filters[{index}].operator must be a nonempty string")
        filters.append(StockFilter(field, operator, condition.get('value'), condition.get('description', '')))

    # Validate custom entries before initializing a manager; basic filters still run first.
    if include_basic and stock_pool_cfg.get('include_basic_filters', False):
        tmp_manager = DynamicStockPoolManager(
            price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
        )
        filters = tmp_manager.add_basic_filters() + filters

    return filters


@lru_cache(maxsize=8)
def _create_pool_manager(
    price_conn: Any,
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    duckdb_path: Optional[str] = None,
) -> DynamicStockPoolManager:
    """Internal helper to create DynamicStockPoolManager with common parameters."""
    return DynamicStockPoolManager(
        price_conn,
        fundamental_table=fundamental_table,
        price_table=price_table,
        db_backend=db_backend,
        duckdb_path=duckdb_path,
    )


# =============================================================================
# Dynamic Stock Pool Filtering
# =============================================================================


def apply_dynamic_stock_pool_filter(
    df: pd.DataFrame,
    price_conn: Any,
    config: Dict[str, Any],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    custom_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    按配置的调仓周期，可选地应用动态股票池过滤。

    Args:
        df: MultiIndex (date, stock_code) 的 DataFrame
        price_conn: 价格数据库连接
        config: 完整配置字典
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端
        start_date: 股票池生成开始日期；缺省时取面板最早日期
        end_date: 股票池生成结束日期；缺省时取面板最晚日期
        custom_logger: 可选自定义 logger

    Returns:
        过滤后的 DataFrame

    启用过滤后，数据源、日期和选股错误直接传播；合法空池返回空面板。
    """
    log = custom_logger or logger

    stock_pool_cfg = config.get('stock_pool', {})
    use_dynamic = bool(stock_pool_cfg.get('use_dynamic_pools', False))
    if not use_dynamic:
        return df

    if df.empty:
        return df

    # 计算过滤所需的时间范围
    if start_date is None or end_date is None:
        idx_dates = df.index.get_level_values('date')
        # 兼容字符串日期
        idx_dates = pd.to_datetime(idx_dates, errors='raise')
        period_start = start_date if start_date is not None else idx_dates.min().strftime('%Y-%m-%d')
        period_end = end_date if end_date is not None else idx_dates.max().strftime('%Y-%m-%d')
    else:
        period_start, period_end = start_date, end_date

    # 构建过滤器
    duckdb_path = config.get('database', {}).get('price_db', {}).get('duckdb_path')
    filters = build_stock_filters(
        stock_pool_cfg=stock_pool_cfg,
        price_conn=price_conn,
        fundamental_table=fundamental_table,
        price_table=price_table,
        db_backend=db_backend,
        duckdb_path=duckdb_path,
        include_basic=True,
    )

    manager = _create_pool_manager(
        price_conn=price_conn,
        fundamental_table=fundamental_table,
        price_table=price_table,
        db_backend=db_backend,
        duckdb_path=duckdb_path,
    )
    stock_pools = manager.get_stock_pools_for_period(
        start_date=period_start,
        end_date=period_end,
        filters=filters,
        max_stocks=stock_pool_cfg.get('max_stocks'),
        sort_by=stock_pool_cfg.get('sort_by'),
        ascending=bool(stock_pool_cfg.get('ascending', False)),
        rebalance_period_months=int(stock_pool_cfg.get('rebalance_period_months', 1)),
        rebalance_schedule=stock_pool_cfg.get('rebalance_schedule', 'quarterly'),
        schedule=stock_pool_cfg.get('schedule'),
    )

    if not stock_pools:
        log.warning("动态股票池结果为空，返回空面板")
        return df.iloc[:0].copy()

    # 使用共享工具函数进行日期级别动态股票池过滤
    total = len(df)
    filtered_df = apply_date_level_pool_filter_to_long_df(df, stock_pools)
    kept = len(filtered_df)

    log.info(f"应用动态股票池过滤: 保留 {kept}/{total} 行 ({kept/total:.2%})")
    if kept == 0:
        log.warning("动态股票池过滤后无数据，请检查过滤条件与时间范围")

    return filtered_df


def get_all_pool_stocks(
    price_conn: Any,
    config: Dict[str, Any],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    start_date: str,
    end_date: str,
    custom_logger: Optional[logging.Logger] = None,
) -> Optional[Set[str]]:
    """
    预计算所有期间股票池的并集，用于 SQL 级别过滤以减少内存占用。
    日期排除由训练、评估或回测阶段处理，不在此处缩小读取范围。
    
    Args:
        price_conn: 价格数据库连接
        config: 完整配置字典
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端
        start_date: 开始日期
        end_date: 结束日期
        custom_logger: 可选自定义 logger
        
    Returns:
        所有池内股票代码的集合，如果未启用动态池则返回 None
    """
    log = custom_logger or logger
    
    stock_pool_cfg = config.get('stock_pool', {})
    if not stock_pool_cfg.get('use_dynamic_pools', False):
        return None
    
    try:
        # 构建过滤器
        duckdb_path = config.get('database', {}).get('price_db', {}).get('duckdb_path')
        filters = build_stock_filters(
            stock_pool_cfg=stock_pool_cfg,
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
            include_basic=True,
        )
        
        manager = _create_pool_manager(
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
        )
        
        monthly_pools = manager.get_stock_pools_for_period(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            max_stocks=stock_pool_cfg.get('max_stocks'),
            sort_by=stock_pool_cfg.get('sort_by'),
            ascending=bool(stock_pool_cfg.get('ascending', False)),
            rebalance_period_months=int(stock_pool_cfg.get('rebalance_period_months', 1)),
            rebalance_schedule=stock_pool_cfg.get('rebalance_schedule', 'quarterly'),
            schedule=stock_pool_cfg.get('schedule'),
        )
        
        if not monthly_pools:
            log.warning("预计算动态股票池为空")
            return None
        
        # 计算所有期间股票的并集
        all_stocks: Set[str] = set()
        for stocks in monthly_pools.values():
            if stocks:
                all_stocks.update(stocks)
        
        # Debug: 显示日期范围和样本池信息
        pool_dates = sorted(monthly_pools.keys())
        sample_counts = [len(monthly_pools[d]) for d in pool_dates[:3]]
        log.info(f"[内存优化] 预计算动态股票池: {len(monthly_pools)} 个调仓期，共 {len(all_stocks)} 只唯一股票")
        log.info(f"[内存优化] 池日期范围: {pool_dates[0]} ~ {pool_dates[-1]}, 请求范围: {start_date} ~ {end_date}")
        log.info(f"[内存优化] 前3个池股票数: {sample_counts}")
        
        return all_stocks
        
    except Exception as e:
        log.error(f"预计算动态股票池失败: {e}")
        raise


def compute_predict_period_dynamic_pools(
    price_conn: Any,
    config: Dict[str, Any],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    train_end,
    predict_window_days: int,
    method_name: str = "synthesis",
    custom_logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Set[str]]]:
    """
    计算预测期的动态股票池。
    
    供 rolling_window_synthesis 中的 LASSO/LightGBM/SizeConditionalElasticNet 共用。
    
    Args:
        price_conn: 价格数据库连接
        config: 完整配置字典
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端
        train_end: 训练结束日期 (datetime)
        predict_window_days: 相对预测开始日的自然日结束偏移量，沿用上下游两端包含约定
        method_name: 方法名称，用于日志
        custom_logger: 可选自定义 logger
        
    Returns:
        动态股票池字典 {date_str: set(stock_codes)}，未启用时返回 None。
        有交易日但没有生效期时抛错；纯休市窗口返回显式空池。
        某个生效期明确为空成员仍合法。
    """
    log = custom_logger or logger
    
    predict_start = train_end + timedelta(days=1)
    predict_end = predict_start + timedelta(days=predict_window_days)
    
    stock_pool_cfg = config.get('stock_pool', {})
    if not stock_pool_cfg.get('use_dynamic_pools', False):
        return None
    
    try:
        duckdb_path = config.get('database', {}).get('price_db', {}).get('duckdb_path')
        
        # Preserve the prediction policy: validate custom conditions without basic filters.
        filters = build_stock_filters(
            stock_pool_cfg=stock_pool_cfg,
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
            include_basic=False,
        )

        manager = _create_pool_manager(
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
        )
        
        raw_pools = manager.get_stock_pools_for_period(
            start_date=predict_start.strftime('%Y-%m-%d'),
            end_date=predict_end.strftime('%Y-%m-%d'),
            filters=filters,
            max_stocks=stock_pool_cfg.get('max_stocks'),
            sort_by=stock_pool_cfg.get('sort_by'),
            ascending=bool(stock_pool_cfg.get('ascending', False)),
            rebalance_period_months=int(stock_pool_cfg.get('rebalance_period_months', 1)),
            rebalance_schedule=stock_pool_cfg.get('rebalance_schedule', 'weekly'),
            schedule=stock_pool_cfg.get('schedule'),
        )
        
        if not raw_pools:
            # A holiday-only final window has no predictions; keep it explicitly empty.
            if manager._trading_calendar().sessions(predict_start, predict_end).empty:
                return {predict_start.strftime('%Y-%m-%d'): set()}
            raise ValueError(
                f"Dynamic prediction pool has no effective periods: "
                f"{predict_start:%Y-%m-%d} ~ {predict_end:%Y-%m-%d}"
            )
        return {k: set(v) for k, v in raw_pools.items()}
        
    except Exception as e:
        log.warning(f"[动态池] {method_name}预测期池计算失败: {e}")
        raise


def compute_dynamic_pools_for_topk(
    price_conn: Any,
    config: Dict[str, Any],
    fundamental_table: str,
    price_table: str,
    db_backend: str,
    start_date: str,
    end_date: str,
    custom_logger: Optional[logging.Logger] = None,
) -> Dict[str, List[str]]:
    """
    为 Top-K 策略评估计算动态股票池
    
    与 get_all_pool_stocks 使用相同的配置参数，确保合成和回测阶段使用一致的股票池。
    
    Args:
        price_conn: 价格数据库连接
        config: 完整配置字典
        fundamental_table: 基础数据表名
        price_table: 价格表名
        db_backend: 数据库后端
        start_date: 开始日期
        end_date: 结束日期
        custom_logger: 可选自定义 logger
        
    Returns:
        {date_str: [stock_codes]} 按生效日期的候选池，未启用时返回 {}。
        有交易日但没有生效期时抛错；纯休市窗口返回显式空池。
        跳月规则由回测器单独处理，不删除本函数的池历史。
    """
    log = custom_logger or logger
    
    stock_pool_cfg = config.get('stock_pool', {})
    use_dynamic = stock_pool_cfg.get('use_dynamic_pools', False)
    
    if not use_dynamic:
        log.info("[Top-K动态池] 配置未启用动态股票池，跳过")
        return {}
    
    try:
        # 构建过滤器 - 与 get_all_pool_stocks 保持一致
        duckdb_path = config.get('database', {}).get('price_db', {}).get('duckdb_path')
        filters = build_stock_filters(
            stock_pool_cfg=stock_pool_cfg,
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
            include_basic=True,
        )
        
        # 从配置读取所有参数
        rebalance_schedule = stock_pool_cfg.get('rebalance_schedule', 'quarterly')
        rebalance_period_months = int(stock_pool_cfg.get('rebalance_period_months', 1))
        max_stocks = stock_pool_cfg.get('max_stocks', None)
        sort_by = stock_pool_cfg.get('sort_by')
        ascending = bool(stock_pool_cfg.get('ascending', False))
        
        # 创建 DynamicStockPoolManager
        manager = _create_pool_manager(
            price_conn=price_conn,
            fundamental_table=fundamental_table,
            price_table=price_table,
            db_backend=db_backend,
            duckdb_path=duckdb_path,
        )
        
        pools = manager.get_stock_pools_for_period(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            max_stocks=max_stocks,
            sort_by=sort_by,
            ascending=ascending,
            rebalance_period_months=rebalance_period_months,
            rebalance_schedule=rebalance_schedule,
            schedule=stock_pool_cfg.get('schedule'),
        )
        
        if not pools:
            # Keep an enabled empty window distinct from disabled membership filtering.
            if manager._trading_calendar().sessions(start_date, end_date).empty:
                return {pd.Timestamp(start_date).strftime('%Y-%m-%d'): []}
            raise ValueError(
                f"Dynamic Top-K pool has no effective periods: {start_date} ~ {end_date}"
            )
        
        return pools
        
    except Exception as e:
        log.error(f"[Top-K动态池] 计算失败: {e}")
        raise


# =============================================================================
# Skip Months Filtering
# =============================================================================


def apply_skip_months(
    obj: Union[pd.DataFrame, pd.Series, None],
    skip_months: Optional[SkipRules],
    custom_logger: Optional[logging.Logger] = None,
) -> Union[pd.DataFrame, pd.Series, None]:
    """
    按配置跳过指定月份/日期窗口的数据。
    
    支持：
    - index 为 MultiIndex(date, stock_code) 的 DataFrame/Series
    - index 为 DatetimeIndex 或可转换为日期的单层索引
    日期缺失、数字索引或无法解析的日期会报错；不在失败后返回未过滤数据。
    
    Args:
        obj: DataFrame 或 Series
        skip_months: 跳过规则集合：整月 (1-12) 或 "MM-DD:MM-DD" 窗口字符串
        custom_logger: 可选自定义 logger
        
    Returns:
        过滤后的对象
    """
    log = custom_logger or logger
    
    if obj is None:
        return obj
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        raise TypeError("apply_skip_months expects a DataFrame or Series")
    if not skip_months or obj.empty:
        return obj

    idx = obj.index
    if isinstance(idx, pd.MultiIndex):
        if idx.names.count('date') != 1:
            raise ValueError("Calendar filtering requires exactly one 'date' index level")
        idx = idx.get_level_values('date')
    # Numbers would otherwise be interpreted as nanoseconds since the Unix epoch.
    if not isinstance(idx, pd.DatetimeIndex) and any(isinstance(value, Number) for value in idx):
        raise ValueError("Calendar filtering does not accept numeric dates")
    dates = pd.DatetimeIndex(pd.to_datetime(idx))
    if dates.hasnans:
        raise ValueError("Calendar filtering requires dates without NaT or missing values")

    mask = ~skip_mask(dates, skip_months)
    keep_count = int(mask.sum())
    total = len(mask)
    if keep_count < total:
        log.info(
            f"应用skip_months过滤: 保留 {keep_count}/{total} 行 ({keep_count/total:.2%})，"
            f"跳过规则={describe_skip_rules(skip_months)}"
        )
    return obj.iloc[mask]
