"""
Factor Operations Module

Factor calculation and transformation utilities for factor synthesis.
Includes forward returns calculation, market cap neutralization, and preprocessing.

Extracted from FactorSynthesizer for modularity and reuse.

Author: QuantationAI
Date: 2026-01-12
Updated: 2026-01-13 - Added PIT adjustment support
"""

import logging
from datetime import timedelta
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Quality Filters
# =============================================================================




# =============================================================================
# Forward Returns Calculation
# =============================================================================


LABEL_MODE_CLOSE_TO_CLOSE = "close_to_close"
LABEL_MODE_NEXT_REBALANCE_CLOSE = "next_rebalance_close"
LABEL_MODE_NEXT_REBALANCE_OPEN = "next_rebalance_open"


def resolve_label_definition(label_definition: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Resolve and validate the executable supervised-label contract.

    Missing configuration deliberately preserves the historical close-to-close
    label.  Unknown explicit values fail closed instead of silently changing a
    governed experiment's target.  Rebalance labels use the factor loader's
    existing invariant that row T contains raw T-1 features; conflicting
    declarations fail instead of becoming inert documentation.
    """
    if label_definition is None:
        return {"mode": LABEL_MODE_CLOSE_TO_CLOSE}
    if not isinstance(label_definition, Mapping):
        raise ValueError("label_definition must be a mapping")
    mode = str(label_definition.get("mode") or LABEL_MODE_CLOSE_TO_CLOSE).strip().lower()
    allowed = {
        LABEL_MODE_CLOSE_TO_CLOSE,
        LABEL_MODE_NEXT_REBALANCE_CLOSE,
        LABEL_MODE_NEXT_REBALANCE_OPEN,
    }
    if mode not in allowed:
        raise ValueError(f"unsupported label_definition.mode={mode!r}; allowed={sorted(allowed)}")
    if mode == LABEL_MODE_CLOSE_TO_CLOSE:
        return {"mode": mode}

    expected_prices = {
        LABEL_MODE_NEXT_REBALANCE_CLOSE: ("close", "next_rebalance_close"),
        LABEL_MODE_NEXT_REBALANCE_OPEN: ("open", "next_rebalance_open"),
    }
    expected_entry, expected_exit = expected_prices[mode]
    for key, default in (("feature_lag_trading_days", 1), ("embargo_trading_days", 5)):
        value = label_definition.get(key, default)
        if isinstance(value, (bool, np.bool_)) or (
            not isinstance(value, str) and value != int(value)
        ):
            raise ValueError(f"{key} must be an integer, not bool or fractional")
    lag = int(label_definition.get("feature_lag_trading_days", 1))
    entry_price = str(label_definition.get("entry_price", expected_entry)).strip().lower()
    exit_price = str(label_definition.get("exit_price", expected_exit)).strip().lower()
    embargo = int(label_definition.get("embargo_trading_days", 5))
    if lag != 1:
        raise ValueError("rebalance label requires feature_lag_trading_days=1")
    if entry_price != expected_entry:
        raise ValueError(f"{mode} requires entry_price={expected_entry!r}")
    if exit_price != expected_exit:
        raise ValueError(f"{mode} requires exit_price={expected_exit!r}")
    if embargo < 1:
        raise ValueError("label_definition.embargo_trading_days must be positive")
    return {
        "mode": mode,
        "feature_lag_trading_days": lag,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "embargo_trading_days": embargo,
    }




def calculate_scheduled_price_returns(
    factor_df: pd.DataFrame,
    price_df: pd.DataFrame,
    entry_exit_dates: Sequence[tuple[Any, Any]],
    custom_logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """Calculate labels for explicit entry/exit events using one price field."""
    log = custom_logger or logger
    if factor_df.empty or price_df is None or price_df.empty:
        return pd.Series(dtype=float, name="forward_return")

    pairs = [
        (pd.Timestamp(entry).normalize(), pd.Timestamp(exit_).normalize())
        for entry, exit_ in entry_exit_dates
    ]
    pairs = sorted(set(pairs))
    if not pairs:
        raise ValueError("scheduled label requires at least one entry/exit pair")
    if any(exit_ <= entry for entry, exit_ in pairs):
        raise ValueError("scheduled label exits must be after entries")

    observed = price_df.notna()
    if (observed & ((price_df <= 0) | ~np.isfinite(price_df))).to_numpy().any():
        raise ValueError("Observed label prices must be positive and finite")
    prices = price_df.copy()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index)).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    entry_dates = pd.DatetimeIndex([entry for entry, _ in pairs])
    exit_dates = pd.DatetimeIndex([exit_ for _, exit_ in pairs])
    entry_prices = prices.reindex(entry_dates)
    exit_prices = prices.reindex(exit_dates)
    exit_prices.index = entry_dates
    returns_wide = exit_prices / entry_prices - 1.0

    returns_long = returns_wide.stack()
    returns_long.index.names = ["date", "stock_code"]
    result = returns_long.reindex(factor_df.index).dropna()
    result.name = "forward_return"
    result.attrs["label_exit_dates"] = {
        entry.strftime("%Y-%m-%d"): exit_.strftime("%Y-%m-%d")
        for entry, exit_ in pairs
    }
    log.info("成功计算 scheduled price 标签，调仓区间=%d，数据点=%d", len(pairs), len(result))
    return result




def calculate_forward_returns(
    factor_df: pd.DataFrame,
    price_conn: Any,
    price_table: str,
    fundamental_table: str,
    period: int = 5,
    custom_logger: Optional[logging.Logger] = None,
    pit_enabled: bool = True,
) -> pd.Series:
    """
    计算前瞻收益率
    
    Uses the same factor_end PIT adjustment as get_pricing_data to ensure
    training and evaluation forward returns are computed consistently.
    
    Args:
        factor_df: 因子 DataFrame，MultiIndex (date, stock_code)
        price_conn: 价格数据库连接
        price_table: 价格表名
        fundamental_table: 基础数据表名 (fallback)
        period: 前瞻期天数
        custom_logger: Optional custom logger
        pit_enabled: 是否启用 PIT 复权（使用 factor_end 方式，与 get_pricing_data 一致）
        
    Returns:
        前瞻收益率 Series，MultiIndex (date, stock_code)
    """
    from .factor_data_loader import PITValidationError, validate_pit_price_rows
    log = custom_logger or logger
    
    # [ALIGNMENT] Logic update: Removed execution_lag support. 
    # Always calculates returns starting from signal date T (Trade at T).
    
    # 获取日期和股票范围
    dates = factor_df.index.get_level_values('date').unique()
    stocks = factor_df.index.get_level_values('stock_code').unique()
    
    # 扩展日期范围获取价格数据
    from utils.trading_calendar import load_trading_calendar
    from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
    ForwardReturnsCalculator._validate_horizons((period,), 0)
    calendar = load_trading_calendar(price_conn)
    calendar.validate_coverage(dates.min(), dates.max())
    extended_start = dates.min()
    extended_end = calendar.next_sessions(dates.max(), period)[-1]
    end_exclusive = (extended_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    stocks_str = "', '".join(stock.replace("'", "''") for stock in stocks)
    
    # 根据是否启用 PIT 决定查询的列
    if pit_enabled:
        select_columns = "date, stock_code, close, factor"
        log.info("[PIT] 前瞻收益率计算启用 factor_end 复权")
    else:
        select_columns = "date, stock_code, close"
    
    # 首先尝试从价格数据库的price_data表获取价格
    sql_price = f"""
    SELECT {select_columns}
    FROM {price_table}
    WHERE stock_code IN ('{stocks_str}')
    AND date >= '{extended_start.strftime('%Y-%m-%d')}' AND date < '{end_exclusive}'
    ORDER BY stock_code, date
    """
    
    try:
        price_df = price_conn.execute(sql_price).fetchdf()
        if price_df.empty and not pit_enabled:
            raise ValueError("price_data表中未找到数据")
    except Exception as e:
        if pit_enabled:
            raise PITValidationError(f"PIT training-price query failed for {price_table}: {e}") from e
        # 如果price_data表没有数据，从factor_data表获取
        log.info(f"{price_table}表无数据 ({e})，尝试从{fundamental_table}表获取价格数据")
        # 仅显式关闭 PIT 时允许从无 factor 的价格表回退。
        sql_factor = f"""
        SELECT date, stock_code, close
        FROM {fundamental_table}
        WHERE stock_code IN ('{stocks_str}')
        AND date >= '{extended_start.strftime('%Y-%m-%d')}' AND date < '{end_exclusive}'
        ORDER BY stock_code, date
        """
        price_df = price_conn.execute(sql_factor).fetchdf()
    
    if price_df.empty:
        log.error("未找到价格数据")
        return pd.Series()
    
    observed = price_df['close'].notna()
    if (observed & ((price_df['close'] <= 0) | ~np.isfinite(price_df['close']))).any():
        raise ValueError("Observed label prices must be positive and finite")

    # 检查是否有 factor 列
    has_factor_column = 'factor' in price_df.columns
    if pit_enabled:
        validate_pit_price_rows(price_df, 'close')
        price_df.loc[price_df['close'].isna(), 'factor'] = float('nan')
    
    # 处理日期格式，可能是带时间戳的5分钟数据
    price_df['datetime'] = pd.to_datetime(price_df['date'])
    price_df['date_only'] = price_df['datetime'].dt.date
    
    # 如果是5分钟数据，需要聚合为日度数据（取每日最后一个收盘价）
    if len(price_df) > len(price_df.groupby(['stock_code', 'date_only'])):
        log.info("检测到高频数据，聚合为日度数据进行前瞻收益率计算")
        agg_dict = {'close': 'last'}
        if has_factor_column:
            agg_dict['factor'] = 'last'
        price_df = price_df.groupby(['stock_code', 'date_only']).agg(agg_dict).reset_index()
        price_df['date'] = pd.to_datetime(price_df['date_only'])
    else:
        price_df['date'] = price_df['datetime']
    
    close_wide = price_df.pivot(index='date', columns='stock_code', values='close').sort_index()
    if pit_enabled:
        from .factor_data_loader import apply_pit_adjustment_factor_end_df
        factor_wide = price_df.pivot(index='date', columns='stock_code', values='factor').sort_index()
        close_wide = apply_pit_adjustment_factor_end_df(close_wide, factor_wide, log)
    close_wide = calendar.align(close_wide, extended_start, extended_end)
    forward_returns_long = (close_wide.shift(-period) / close_wide - 1).stack()
    forward_returns_long.index.names = ['date', 'stock_code']
    forward_return_series = forward_returns_long.reindex(factor_df.index).dropna()
    forward_return_series.name = 'forward_return'
    # This is the calendar used by align/shift above, not the sparse factor index.
    forward_return_series.attrs['label_exit_dates'] = {
        entry.strftime('%Y-%m-%d'): calendar.next_sessions(entry, period)[-1].strftime('%Y-%m-%d')
        for entry in forward_return_series.index.get_level_values('date').unique()
    }

    pit_status = "启用 (factor_end)" if (pit_enabled and has_factor_column) else "禁用"
    log.info(f"成功计算前瞻收益率 (PIT {pit_status})，数据点: {len(forward_return_series)}")
    
    return forward_return_series


# =============================================================================
# Market Cap Neutralization
# =============================================================================




# =============================================================================
# Feature Preprocessing
# =============================================================================

# Native 预处理方法（不通过 Qlib Processor）
# 注意: zscore 已改为 Native 实现 (去极值 + Z-Score)，遵循标准量化预处理流程
_NATIVE_PREPROCESS_METHODS = {'binning', 'winsorize_zscore', 'zscore'}

# Qlib Processor 支持的方法
# raw_zscore: 原始 Z-Score（不去极值），适用于已处理过的数据
_QLIB_PREPROCESS_METHODS = {'raw_zscore', 'rank_zscore', 'gauss_rank', 'rank'}


def apply_feature_preprocessing(
    df: pd.DataFrame,
    preprocess_cfg: Dict[str, Any],
    custom_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    根据配置对特征逐因子进行按日截面预处理。
    
    支持的模式：
    - "zscore":           按日截面去极值 + Z-Score 标准化（标准量化预处理流程）
    - "raw_zscore":       按日截面原始 Z-Score（不去极值，适用于已处理数据）
    - "rank_zscore":      按日截面 rank 后再做 Z-Score
    - "gauss_rank":       Gauss Rank 标准化 (Rank + Inverse Normal Transform)
    - "rank":             按日截面百分位 rank 映射到 [0,1]
    - "binning":          按日截面等频分箱 [0,1]（适用于事件驱动型因子）
    - "winsorize_zscore": 按日截面缩尾后 Z-Score（适用于边界情况因子）
    - "none" / 缺省:      不做处理
    
    配置示例:
    ```python
    preprocess_cfg = {
        "ALL": "rank_zscore",           # 默认方法
        "DemoSignal1": "zscore",            # 特定因子覆盖
        "EventFactor": "binning",       # 事件驱动型因子
        "_binning_n_bins": 5,           # 分箱参数（可选）
        "_winsorize_percentile": 0.025, # 缩尾参数（可选）
    }
    ```
    
    Args:
        df: 因子数据 DataFrame，MultiIndex (date, stock_code)
        preprocess_cfg: 因子名 -> 预处理方式的映射，支持特殊键:
                       - "ALL": 默认预处理方法
                       - "_binning_n_bins": 分箱数（默认 5）
                       - "_winsorize_percentile": 缩尾百分位（默认 0.025）
        custom_logger: Optional custom logger
        
    Returns:
        预处理后的 DataFrame
    """
    log = custom_logger or logger
    
    if df.empty or not preprocess_cfg:
        return df
    
    # 提取参数
    default_mode = preprocess_cfg.get("ALL", "none")
    binning_n_bins = preprocess_cfg.get("_binning_n_bins", 5)
    winsorize_percentile = preprocess_cfg.get("_winsorize_percentile", 0.025)
    
    # 记录使用的预处理模式
    if default_mode and default_mode != "none":
        log.info(f"[预处理] 默认模式: {default_mode}")
    
    # 分类因子：哪些用 Native 方法，哪些用 Qlib Processor
    native_factors: Dict[str, str] = {}  # factor_name -> method
    qlib_factors: Dict[str, str] = {}    # factor_name -> method
    
    for col in df.columns:
        mode = preprocess_cfg.get(col, default_mode)
        
        if mode in _NATIVE_PREPROCESS_METHODS:
            native_factors[col] = mode
            if mode != default_mode:
                log.debug(f"[预处理] 因子 {col} 使用 Native 方法: {mode}")
        elif mode in _QLIB_PREPROCESS_METHODS:
            qlib_factors[col] = mode
            if mode != default_mode:
                log.debug(f"[预处理] 因子 {col} 使用 Qlib 方法: {mode}")
        elif mode and mode != "none":
            # 未知方法，尝试交给 Qlib 处理
            qlib_factors[col] = mode
            log.warning(f"[预处理] 因子 {col} 使用未知方法 '{mode}'，将尝试 Qlib 处理")
    
    # 统计
    native_count = len(native_factors)
    qlib_count = len(qlib_factors)
    skip_count = len(df.columns) - native_count - qlib_count
    
    if native_count > 0 or qlib_count > 0:
        log.info(
            f"[预处理] 因子分配: Native={native_count}, Qlib={qlib_count}, Skip={skip_count}"
        )
    
    result_df = df.copy()
    
    # === 1. 处理 Native 方法（zscore, binning, winsorize_zscore）===
    if native_factors:
        from .feature_preprocessor import (
            cross_sectional_binning,
            cross_sectional_winsorize_zscore,
            cross_sectional_zscore,
        )
        
        for factor_name, method in native_factors.items():
            try:
                if method == 'zscore':
                    result_df[factor_name] = cross_sectional_zscore(
                        result_df[factor_name], percentile=winsorize_percentile
                    )
                elif method == 'binning':
                    result_df[factor_name] = cross_sectional_binning(
                        result_df[factor_name], n_bins=binning_n_bins
                    )
                elif method == 'winsorize_zscore':
                    result_df[factor_name] = cross_sectional_winsorize_zscore(
                        result_df[factor_name], percentile=winsorize_percentile
                    )
            except Exception as e:
                log.warning(f"[预处理] 因子 {factor_name} Native 处理失败: {e}")
                raise
    
    # === 2. 处理 Qlib 方法（raw_zscore, rank_zscore, gauss_rank, rank）===
    if qlib_factors:
        try:
            from factor_synthesis.qlib_adapter.processors import PerFactorPreprocessor
            
            # 提取需要 Qlib 处理的列
            qlib_cols = list(qlib_factors.keys())
            df_qlib = result_df[qlib_cols].copy()
            
            # 构建 Qlib 配置
            qlib_preprocess_cfg = {"ALL": default_mode}
            qlib_preprocess_cfg.update(qlib_factors)
            
            # 转换索引格式以兼容 Qlib Processor (需要 'datetime' 级别)
            original_index_names = df_qlib.index.names
            if 'date' in original_index_names:
                df_qlib.index = df_qlib.index.rename(
                    ['datetime' if n == 'date' else n for n in original_index_names]
                )
            
            # 使用 Qlib Processor 进行预处理
            processor = PerFactorPreprocessor(qlib_preprocess_cfg, fields_group="feature")
            df_qlib_processed = processor(df_qlib)
            
            # 恢复原始索引名称
            df_qlib_processed.index = df_qlib_processed.index.rename(original_index_names)
            
            # 将处理结果合并回 result_df
            for col in qlib_cols:
                result_df[col] = df_qlib_processed[col]
                
        except ImportError as e:
            log.error(
                f"[预处理] Qlib processor 不可用: {e}. "
                f"将跳过 {len(qlib_factors)} 个因子的 Qlib 预处理"
            )
        except Exception as e:
            log.error(f"[预处理] Qlib 处理失败: {e}")
    
    return result_df
