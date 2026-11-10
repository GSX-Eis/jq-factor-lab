# -*- coding: utf-8 -*-
"""
Shared preprocessing utilities for synthesis methods.

Uses Qlib-style processors internally for consistent preprocessing logic.
"""

from typing import Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd


def residualize_target_by_factor(
    target: pd.Series,
    baseline: pd.Series,
    *,
    min_samples: int = 10,
) -> pd.Series:
    """Remove the same-date linear baseline exposure from a return target."""
    aligned = pd.concat(
        [
            pd.to_numeric(target, errors="coerce").rename("target"),
            pd.to_numeric(baseline, errors="coerce").rename("baseline"),
        ],
        axis=1,
    )

    def residualize(group: pd.DataFrame) -> pd.Series:
        valid = group.dropna()
        output = pd.Series(np.nan, index=group.index, dtype=float)
        if len(valid) < min_samples:
            output.loc[valid.index] = valid["target"] - valid["target"].mean()
            return output
        centered_x = valid["baseline"] - valid["baseline"].mean()
        denominator = float(centered_x.pow(2).sum())
        if denominator <= 1e-12:
            output.loc[valid.index] = valid["target"] - valid["target"].mean()
            return output
        centered_y = valid["target"] - valid["target"].mean()
        beta = float((centered_x * centered_y).sum() / denominator)
        output.loc[valid.index] = centered_y - beta * centered_x
        return output

    level = "date" if "date" in aligned.index.names else 0
    groups = [residualize(group) for _, group in aligned.groupby(level=level, sort=False)]
    residual = pd.concat(groups) if groups else pd.Series(index=target.index, dtype=float)
    residual.name = target.name
    return residual.reindex(target.index)


def apply_target_cs_preprocessing(
    y: pd.Series,
    cs_demean: bool = True,
    cs_rank: bool = False,
    cs_standardize: bool = False,
    eps: float = 1e-8,
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """
    Apply cross-sectional preprocessing to target variable.
    
    Uses Qlib TargetCSProcessor logic internally.
    
    Args:
        y: Target Series with MultiIndex (date, stock_code)
        cs_demean: Whether to demean cross-sectionally
        cs_rank: Whether to rank cross-sectionally (then demean)
        cs_standardize: Whether to standardize cross-sectionally
        eps: Small value to avoid division by zero
        logger: Optional logger
        
    Returns:
        Preprocessed target Series
    """
    try:
        from factor_synthesis.qlib_adapter.processors import TargetCSProcessor
        
        if not (cs_demean or cs_rank or cs_standardize):
            return y
        
        # Convert to DataFrame for processor
        y_df = pd.DataFrame({'LABEL0': y})
        
        # Rename index for Qlib compatibility
        original_names = y_df.index.names
        if 'date' in original_names:
            y_df.index = y_df.index.rename(['datetime' if n == 'date' else n for n in original_names])
        
        # Apply processor
        processor = TargetCSProcessor(
            cs_demean=cs_demean,
            cs_rank=cs_rank,
            cs_standardize=cs_standardize,
            eps=eps,
            fields_group="label"
        )
        
        y_processed = processor(y_df)['LABEL0']
        
        # Restore index names
        y_processed.index = y_processed.index.rename(original_names)
        
        if logger:
            logger.info(
                "[目标处理] 使用Qlib Processor: 去均值=%s, 标准化=%s, Rank化=%s",
                cs_demean, cs_standardize, cs_rank
            )
        
        return y_processed
        
    except ImportError:
        # Fallback to manual implementation
        if logger:
            logger.warning("[目标处理] Qlib processor不可用，使用回退实现")
        return _apply_target_cs_preprocessing_fallback(
            y, cs_demean, cs_rank, cs_standardize, eps, logger
        )


def _apply_target_cs_preprocessing_fallback(
    y: pd.Series,
    cs_demean: bool,
    cs_rank: bool,
    cs_standardize: bool,
    eps: float,
    logger: Optional[logging.Logger],
) -> pd.Series:
    """Fallback implementation without Qlib dependency."""
    result = y.copy()
    
    if cs_rank:
        y_rank = y.groupby(level="date").rank(pct=True)
        result = y_rank - y_rank.groupby(level="date").transform("mean")
        if logger:
            logger.info("[目标处理] 已执行Rank变换并按日去均值")
    else:
        if cs_demean:
            result = y - y.groupby(level="date").transform("mean")
        if cs_standardize:
            cs_std = result.groupby(level="date").transform("std")
            result = result / (cs_std.replace(0, np.nan).fillna(eps))
    
    return result


def compute_sample_weights(
    df: pd.DataFrame,
    weight_decay: float = 1.0,
    half_life: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """
    Compute time-decay sample weights.
    
    Uses Qlib SampleWeightProcessor logic internally.
    
    Args:
        df: DataFrame with MultiIndex (date, stock_code)
        weight_decay: Exponential decay factor (1.0 = no decay)
        half_life: Half-life for decay (0.0 = use weight_decay instead)
        logger: Optional logger
        
    Returns:
        Sample weights array, or None if no decay
    """
    if not np.isfinite(half_life):
        raise ValueError("half_life must be finite")
    if half_life <= 0:
        if weight_decay >= 1.0:
            return None
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Active weight_decay must be nonnegative and finite")
    
    try:
        from factor_synthesis.qlib_adapter.processors import SampleWeightProcessor
        
        # Rename index for Qlib compatibility
        df_qlib = df.copy()
        original_names = df_qlib.index.names
        if 'date' in original_names:
            df_qlib.index = df_qlib.index.rename(['datetime' if n == 'date' else n for n in original_names])
        
        processor = SampleWeightProcessor(
            weight_decay=weight_decay,
            half_life=half_life
        )
        processor(df_qlib)
        weights = processor.get_sample_weights()
        
        if logger:
            logger.info(
                "[样本权重] 使用Qlib Processor: decay=%.4f, half_life=%.1f",
                weight_decay, half_life
            )
        
        return weights
        
    except ImportError:
        # Fallback to manual implementation
        if logger:
            logger.warning("[样本权重] Qlib processor不可用，使用回退实现")
        return _compute_sample_weights_fallback(df, weight_decay, half_life, logger)


def _compute_sample_weights_fallback(
    df: pd.DataFrame,
    weight_decay: float,
    half_life: float,
    logger: Optional[logging.Logger],
) -> np.ndarray:
    """Fallback implementation without Qlib dependency."""
    dates = df.index.get_level_values("date")
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    days_diff = (max_date - dates).days.values.astype(float)
    
    if half_life > 0:
        weights = np.exp(-np.log(2) * days_diff / half_life)
    else:
        weights = np.power(weight_decay, days_diff)
    
    # Normalize
    weights = weights / weights.mean()
    
    if logger:
        logger.info("[样本权重] 时间衰减权重计算完成")
    
    return weights






