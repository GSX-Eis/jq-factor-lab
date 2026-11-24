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
from utils.market_db.identity import slot_market_db_identity
from utils.market_db.guards import require_single_market_db
from utils.slots import MarketSlot

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
    pit_enabled: bool | None = None,
    *,
    slot: MarketSlot | None = None,
    frequency: str = "1d",
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
    from .factor_data_loader import _adjust_equity_prices, _prepare_price_rows, _price_rows, _pricing_window
    from ..factor_reader import resolve_market_input
    from backtest_and_analysis.evaluation.labels import ExecutionConvention, validate_daily_horizons

    log = custom_logger or logger
    dates = factor_df.index.get_level_values("date").unique()
    stocks = factor_df.index.get_level_values("stock_code").unique().tolist()
    if not len(dates):
        return pd.Series(dtype=float)
    _, frequency = resolve_market_input(slot, None, frequency)
    (period,) = validate_daily_horizons((period,), 0)
    calendar, extended_start, extended_end = _pricing_window(slot, price_conn, frequency, dates, period)
    pit_enabled = _adjust_equity_prices(slot, pit_enabled)
    price_df = _price_rows(
        price_conn, price_table, stocks, extended_start, extended_end, "close",
        slot=slot, frequency=frequency, pit_enabled=pit_enabled, fallback=(fundamental_table, "close"),
    )
    if price_df.empty:
        return pd.Series(dtype=float)
    observed = price_df.close.notna()
    if (observed & ((price_df.close <= 0) | ~np.isfinite(price_df.close))).any():
        raise ValueError("Observed label prices must be positive and finite")
    price_df = _prepare_price_rows(
        price_df, "close", pit_enabled=pit_enabled, legacy_daily=slot is None,
    )

    close_wide = price_df.pivot(index='date', columns='stock_code', values='close').sort_index()
    if pit_enabled:
        from .factor_data_loader import apply_pit_adjustment_factor_end_df
        factor_wide = price_df.pivot(index='date', columns='stock_code', values='factor').sort_index()
        close_wide = apply_pit_adjustment_factor_end_df(close_wide, factor_wide, log)
    availability = None
    if frequency == "1d":
        sessions = calendar.sessions(extended_start, extended_end)
        if not close_wide.index.isin(sessions).all():
            raise ValueError("Panel contains observations on non-session dates")
        close_wide = close_wide.reindex(sessions.rename("date"))
    else:
        from .pool_manager import market_bar_grid, synthesis_bar_schedule

        grid = market_bar_grid(slot, frequency, extended_start, extended_end)
        close_wide = close_wide.reindex(pd.DatetimeIndex(grid.timestamp_utc, name="date"))
        schedule = synthesis_bar_schedule(slot, frequency)
        availability = [schedule.available_at(timestamp) for timestamp in close_wide.index]
    rule = ExecutionConvention.CLOSE_TO_CLOSE.value
    offset = rule.exit_offset(period)
    forward_returns = rule.returns(close_wide, close_wide.shift(-offset)).stack()
    forward_returns.index.names = ["date", "stock_code"]
    result = forward_returns.reindex(factor_df.index).dropna().rename("forward_return")
    exits = dict(zip(close_wide.index, (close_wide.index if availability is None else availability)[offset:]))
    result.attrs["label_exit_dates"] = {
        entry.strftime("%Y-%m-%d") if frequency == "1d" else entry.isoformat():
        exits[entry].strftime("%Y-%m-%d") if frequency == "1d" else exits[entry].isoformat()
        for entry in result.index.get_level_values("date").unique()
    }
    log.info("Calculated %d forward returns", len(result))
    return result


def neutralize_market_features(
    frame: pd.DataFrame, config: Mapping[str, Any], *, slot: MarketSlot | None,
    price_conn=None, frequency: str = "1d", start_date: str, end_date: str,
    lag_exposures: bool = True,
) -> pd.DataFrame:
    """Residualize each cross-section using exposures available at that clock."""
    from .pool_manager import INDUSTRY_FEATURE, MARKET_CAP_FEATURE, aligned_aux_feature, legacy_aux_slot

    require_single_market_db(frame, slot_market_db_identity(slot))
    missing_policy = config.get("on_missing_aux", "skip")
    if missing_policy not in {"skip", "error"}:
        raise ValueError("neutralization.on_missing_aux must be 'skip' or 'error'")
    features = [
        key for key, option in ((MARKET_CAP_FEATURE, "market_cap"), (INDUSTRY_FEATURE, "industry"))
        if config.get(option, True)
    ]
    columns = frame.columns.difference(config.get("skip_factors") or [])
    if not features or not len(columns):
        return frame
    if slot is None and price_conn is not None:
        slot = legacy_aux_slot(price_conn)
    instruments = frame.index.get_level_values("stock_code").unique().tolist()
    exposures = {}
    for key in features:
        values = (aligned_aux_feature(slot, key, instruments, start_date, end_date,
                                      frequency=frequency, shift=1 if lag_exposures else 0)
                  if slot is not None else pd.Series(dtype=float))
        if values.empty:
            if missing_policy == "error":
                raise ValueError(f"MarketSlot does not supply neutralization feature {key!r}")
        else:
            exposures[key] = values
    if not exposures:
        return frame

    result = frame.copy()
    aligned = pd.DataFrame(exposures).reindex(frame.index)
    if MARKET_CAP_FEATURE in aligned:
        aligned[MARKET_CAP_FEATURE] = aligned[MARKET_CAP_FEATURE].where(
            np.isfinite(aligned[MARKET_CAP_FEATURE])
        )
    for _, group in aligned.groupby(level="date", sort=False):
        if missing_policy == "skip":
            group = group.dropna(axis=1, how="all")
        group = group.dropna()
        if group.empty or group.shape[1] == 0 or len(group) < 5:
            continue
        predictors = [np.ones(len(group))]
        if MARKET_CAP_FEATURE in group:
            predictors.append(group[MARKET_CAP_FEATURE].to_numpy(dtype=float))
        if INDUSTRY_FEATURE in group:
            predictors.append(pd.get_dummies(group[INDUSTRY_FEATURE], drop_first=True, dtype=float).to_numpy())
        design = np.column_stack(predictors)
        for column in columns:
            target = frame.loc[group.index, column]
            valid = np.isfinite(target.to_numpy())
            if valid.sum() < max(5, design.shape[1] + 1):
                continue
            x, y = design[valid], target.to_numpy()[valid]
            coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            result.loc[target.index[valid], column] = y - x @ coefficients
    return result


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
    *,
    history: pd.DataFrame | None = None,
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
    
    require_single_market_db(df, history)

    # 提取参数
    default_mode = preprocess_cfg.get("ALL", "none")
    binning_n_bins = preprocess_cfg.get("_binning_n_bins", 5)
    winsorize_percentile = preprocess_cfg.get("_winsorize_percentile", 0.025)
    
    # 记录使用的预处理模式
    if default_mode and default_mode != "none":
        log.info(f"[预处理] 默认模式: {default_mode}")
    
    from .feature_preprocessor import TIME_SERIES_METHODS, time_series_standardize

    rolling_methods = {
        col: preprocess_cfg.get(col, default_mode) for col in df
        if preprocess_cfg.get(col, default_mode) in TIME_SERIES_METHODS
    }
    # 分类因子：哪些用 Native 方法，哪些用 Qlib Processor
    native_factors: Dict[str, str] = {}  # factor_name -> method
    qlib_factors: Dict[str, str] = {}    # factor_name -> method
    
    for col in df.columns:
        mode = preprocess_cfg.get(col, default_mode)
        
        if col in rolling_methods:
            continue
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
    for col, mode in rolling_methods.items():
        source = df[col]
        if history is not None:
            dates = history.index.get_level_values("date")
            previous = (dates <= source.index.get_level_values("date").max()) & ~history.index.isin(source.index)
            source = pd.concat([history.loc[previous, col], source])
        result_df[col] = time_series_standardize(
            source, mode, window=preprocess_cfg.get("_ts_window", 20),
            min_periods=preprocess_cfg.get("_ts_min_periods"),
        ).reindex(df.index)
    
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
