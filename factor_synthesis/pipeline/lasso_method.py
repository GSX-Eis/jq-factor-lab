# -*- coding: utf-8 -*-
"""
LASSO/ElasticNet Factor Synthesis - Qlib Adapter Only

This module provides LASSO/ElasticNet based factor synthesis using the Qlib adapter.
All model training and prediction is delegated to QlibLinearWrapper.
"""

import hashlib
from functools import lru_cache
from pathlib import Path
from datetime import timedelta
from typing import Any, Dict, Mapping, Tuple, Optional

from bisect import bisect_right

import numpy as np
import pandas as pd

# Qlib Adapter (required)
from factor_synthesis.qlib_adapter import DataConverter, QlibLinearWrapper

# Dynamic blend module
from .dynamic_blend import (
    DynamicBlendConfig,
    compute_prior_composite,
    compute_dynamic_blend_ratio,
)

QLIB_ADAPTER_AVAILABLE = True


class InsufficientSurvivingFactorsError(RuntimeError):
    """Fail closed when a training window loses too many configured factors."""

    def __init__(self, audit: Dict[str, Any]):
        self.audit = audit
        super().__init__(
            "training-window factor survival below threshold: "
            f"{audit['surviving_factor_count']}/{audit['configured_factor_count']} "
            f"< required {audit['min_required_factor_count']}"
        )


class IncompleteResearchWindowError(RuntimeError):
    """Fail closed when a research window cannot evaluate its residual sleeve."""


def _require_complete_research_window(config: Mapping[str, Any], reason: Any) -> None:
    if config.get("require_complete_research_windows") and reason in {
        "training_data_unavailable",
        "no_surviving_features",
        "parent_baseline_history_unavailable",
        "insufficient_training_samples",
        "prediction_features_unavailable",
        "insufficient_oof_history",
    }:
        raise IncompleteResearchWindowError(f"incomplete research window: {reason}")


@lru_cache(maxsize=8)
def _load_parent_baseline(path_value: str, expected_sha256: str) -> pd.Series:
    path = Path(path_value).expanduser().resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"parent baseline hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if "composite_value" not in frame or not isinstance(frame.index, pd.MultiIndex):
        raise ValueError(f"invalid parent baseline cache: {path}")
    baseline = pd.to_numeric(frame["composite_value"], errors="coerce")
    if baseline.index.duplicated().any() or baseline.isna().any() or not np.isfinite(baseline).all():
        raise ValueError(f"invalid parent baseline values: {path}")
    return baseline.sort_index()


def _parent_baseline_scores(config: Mapping[str, Any], index=None) -> Optional[pd.Series]:
    contract = config.get("parent_baseline")
    if not isinstance(contract, Mapping):
        return None
    path = str(contract.get("path", "")).strip()
    digest = str(contract.get("sha256", "")).strip()
    if not path or not digest:
        raise ValueError("parent_baseline requires path and sha256")
    baseline = _load_parent_baseline(path, digest)
    return baseline if index is None else baseline.reindex(index)


def _target_residual_baseline(synthesizer, factor_df, index, config):
    parent_baseline = _parent_baseline_scores(config, index)
    if parent_baseline is not None:
        return parent_baseline
    target_cfg = config.get("target_residualization", {})
    configured = target_cfg.get("baseline_factors")
    factors = [str(item) for item in configured] if isinstance(configured, list) else []
    if not factors:
        single = str(target_cfg.get("baseline_factor", "")).strip()
        factors = [single] if single else []
    if not factors or any(factor not in factor_df.columns for factor in factors):
        raise ValueError(f"target_residualization baseline factors unavailable: {factors!r}")
    weights = target_cfg.get("weights") or config.get("blend_weights", {})
    if set(weights) != set(factors):
        raise ValueError("target_residualization weights must exactly cover baseline_factors")
    if not all(np.isfinite(float(weights[factor])) for factor in factors):
        raise ValueError("target_residualization weights must be finite")
    preprocess = config.get("preprocess", {})
    baseline_cfg = {
        factor: target_cfg.get("preprocess", preprocess.get(factor, "zscore"))
        for factor in factors
    }
    processed = synthesizer._apply_feature_preprocessing(
        factor_df[factors].reindex(index), baseline_cfg
    )
    return sum((processed[factor] * float(weights[factor]) for factor in factors), pd.Series(0.0, index=index))


def _maybe_residualize_training_target(
    target: pd.Series,
    baseline: pd.Series,
    config: Mapping[str, Any],
) -> tuple[pd.Series, bool]:
    """Keep a pooled-selection baseline while optionally residualizing the fit target."""
    if not config.get("residualize_training_target", True):
        return target, False
    from .preprocessing import residualize_target_by_factor

    return (
        residualize_target_by_factor(
            target,
            baseline,
            min_samples=int(config.get("min_samples", 10)),
        ),
        True,
    )


def _validate_preselection_exclusivity(config: Dict) -> None:
    """检查预筛选配置互斥性
    
    确保barra_preselection、pre_selection和hierarchical_clustering
    三种预筛选方法同时只能启用一个。
    
    Args:
        config: 合成配置字典
        
    Raises:
        ValueError: 如果多个预筛选方法同时启用
    """
    enabled_methods = []
    
    if config.get('barra_preselection', {}).get('enabled', False):
        enabled_methods.append('barra_preselection')
    if config.get('pre_selection', {}).get('enabled', False):
        enabled_methods.append('pre_selection')
    if config.get('hierarchical_clustering', {}).get('enabled', False):
        enabled_methods.append('hierarchical_clustering')
    
    if len(enabled_methods) > 1:
        raise ValueError(
            f"❌ 预筛选方法互斥错误: 同时启用了 {enabled_methods}。\n"
            f"请确保只启用以下方法之一: barra_preselection, pre_selection, hierarchical_clustering"
        )


def _compute_embargo_cutoff(
    factor_df_dates: pd.DatetimeIndex,
    train_end_dt: pd.Timestamp,
    target_period: int,
) -> pd.Timestamp:
    """
    Compute the embargo cutoff: the latest training date such that its
    forward return (T + target_period trading days) does NOT extend past
    train_end_dt.

    Uses the actual dates present in factor_df as trading calendar.
    """
    all_dates = pd.to_datetime(factor_df_dates.unique()).sort_values()
    if len(all_dates) == 0:
        return train_end_dt
    pos = all_dates.searchsorted(train_end_dt, side='right') - 1
    if pos < 0:
        return train_end_dt
    embargo_pos = pos - target_period
    if embargo_pos < 0:
        return all_dates[0]
    return all_dates[embargo_pos]


def _resolve_embargo_trading_days(config: Dict[str, Any]) -> int:
    """Use the label horizon when it differs from the legacy target period."""
    return max(1, int(config.get("_label_embargo_days", config.get("target_period", 5))))


def _resolve_label_exit_dates(config: Mapping[str, Any]) -> dict[pd.Timestamp, pd.Timestamp]:
    """Return the explicit entry-to-exit event map for rebalance labels."""
    raw = config.get("_label_exit_dates")
    if not isinstance(raw, Mapping):
        return {}
    return {
        pd.Timestamp(entry).normalize(): pd.Timestamp(exit_).normalize()
        for entry, exit_ in raw.items()
    }


def _filter_index_by_label_exit(
    index: pd.MultiIndex,
    exit_dates: Mapping[pd.Timestamp, pd.Timestamp],
    boundary: pd.Timestamp,
) -> pd.MultiIndex:
    """Keep only labels whose realized exit does not cross a split boundary."""
    if not exit_dates:
        return index
    normalized_boundary = pd.Timestamp(boundary).normalize()
    keep = [
        exit_dates.get(pd.Timestamp(date).normalize(), pd.Timestamp.max.normalize())
        <= normalized_boundary
        for date in index.get_level_values("date")
    ]
    return index[np.asarray(keep, dtype=bool)]


def _label_dates_exiting_before_boundary(
    dates: pd.DatetimeIndex,
    exit_dates: Mapping[pd.Timestamp, pd.Timestamp],
    boundary: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Return training entries whose labels end strictly before validation."""
    normalized_boundary = pd.Timestamp(boundary).normalize()
    return [
        pd.Timestamp(date).normalize()
        for date in dates
        if exit_dates.get(pd.Timestamp(date).normalize(), pd.Timestamp.max.normalize())
        < normalized_boundary
    ]


def _resolve_inner_train_valid_dates(
    X: pd.DataFrame, config: Mapping[str, Any], logger
) -> tuple[str, str, str]:
    """Display bounds only; callers must also apply the exact eligible keys."""
    train, valid = _resolve_inner_train_valid_indices(X, config, logger)
    return (
        train.get_level_values("date").max().strftime("%Y-%m-%d"),
        valid.get_level_values("date").min().strftime("%Y-%m-%d"),
        valid.get_level_values("date").max().strftime("%Y-%m-%d"),
    )


def _strict_label_exit_index(index, config, boundary):
    """Resolve producer metadata on actual rows, and reject an empty safe set."""
    from factor_synthesis.qlib_adapter.panel_purged_kfold import (
        LabelIsolationError, resolve_label_exit_dates,
    )
    exits = resolve_label_exit_dates(
        index, config.get("_label_exit_dates"),
        trading_calendar=config.get("_label_trading_calendar"),
        label_period=config.get("_label_effective_period"),
        calendar_source=config.get("_label_calendar_source"),
    )
    boundary = pd.Timestamp(boundary).normalize()
    if pd.isna(boundary) or boundary.tz is not None:
        raise LabelIsolationError("invalid label isolation boundary")
    eligible = index[(exits < boundary).to_numpy()]
    if not len(eligible):
        raise LabelIsolationError(
            f"no training labels exit before {boundary.date()} (rows={len(index)})"
        )
    return eligible


def _outer_prediction_index(synthesizer, factor_df, train_end, daily_pools, logger):
    """Use the same prediction window and pool keys as the eventual output."""
    start = pd.Timestamp(train_end) + timedelta(days=1)
    end = start + timedelta(days=synthesizer.config["rolling_window"]["predict_window_days"])
    dates = factor_df.index.get_level_values("date")
    index = factor_df.index[(dates >= start) & (dates <= end)]
    if daily_pools and len(index):
        keys = _filter_by_pool_date_range(index, daily_pools, logger)
        index = index[index.isin(keys)]
    return index


def _resolve_inner_train_valid_indices(
    X: pd.DataFrame, config: Mapping[str, Any], logger, *, fraction_key="lasso_validation_fraction"
) -> tuple[pd.MultiIndex, pd.MultiIndex]:
    """Return strict train/validation keys without filling noncontiguous holes."""
    from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
    train_dates = X.index.get_level_values("date").unique().sort_values()
    n_dates = len(train_dates)
    validation_fraction = config.get(fraction_key, 0.2)
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise LabelIsolationError("validation fraction must be finite and between zero and one")
    split_idx = int(n_dates * (1.0 - validation_fraction))
    if not 0 < split_idx < n_dates:
        raise LabelIsolationError(f"insufficient dates for inner split: dates={n_dates}")
    boundary = train_dates[split_idx]
    train = _strict_label_exit_index(X.index, config, boundary)
    valid = X.index[X.index.get_level_values("date") >= boundary]
    logger.info("[LabelIsolation] boundary=%s train=%d valid=%d", boundary, len(train), len(valid))
    return train, valid


def _filter_by_missing_rate(
    df: pd.DataFrame,
    max_missing_rate: float,
    logger,
    *,
    min_surviving_ratio: float = 0.0,
    configured_factor_count: Optional[int] = None,
    configured_factor_names: Optional[list[str]] = None,
    return_audit: bool = False,
) -> Tuple[pd.DataFrame, list] | Tuple[pd.DataFrame, list, Dict[str, Any]]:
    """
    Filter out factors with missing rate exceeding threshold within the training window.
    
    Args:
        df: Factor DataFrame with columns as factor names
        max_missing_rate: Maximum allowed missing rate (e.g., 0.05 for 5%)
        logger: Logger instance
        
    Returns:
        Filtered DataFrame and list of removed factor names
    """
    if not np.isfinite(max_missing_rate):
        raise ValueError("max_missing_rate must be finite")
    if configured_factor_names:
        df = df.reindex(columns=list(dict.fromkeys([*df.columns, *configured_factor_names])))

    if df.empty or max_missing_rate <= 0:
        audit = {
            "configured_factor_count": int(configured_factor_count or len(df.columns)),
            "surviving_factor_count": int(len(df.columns)),
            "survival_ratio": 1.0 if len(df.columns) else 0.0,
            "min_surviving_ratio": float(min_surviving_ratio),
            "min_required_factor_count": int(np.ceil((configured_factor_count or len(df.columns)) * min_surviving_ratio)),
            "max_missing_rate": float(max_missing_rate),
            "missing_rate_by_factor": {},
            "removed_factors": [],
        }
        if min_surviving_ratio > 0 and audit["surviving_factor_count"] < audit["min_required_factor_count"]:
            raise InsufficientSurvivingFactorsError(audit)
        return (df, [], audit) if return_audit else (df, [])
    
    total_rows = len(df)
    removed_factors = []
    kept_factors = []
    
    missing_rate_by_factor: Dict[str, float] = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / total_rows
        missing_rate_by_factor[str(col)] = float(missing_rate)
        
        if missing_rate > max_missing_rate:
            removed_factors.append((col, missing_rate))
            logger.warning(
                "[数据质量] 训练窗口移除因子 %s: 缺失率 %.2f%% > 阈值 %.2f%%",
                col, missing_rate * 100, max_missing_rate * 100
            )
        else:
            kept_factors.append(col)
    
    if removed_factors:
        logger.info(
            "[数据质量] 训练窗口: 移除 %d 个高缺失率因子，保留 %d 个",
            len(removed_factors), len(kept_factors)
        )
    
    configured_count = int(configured_factor_count or len(df.columns))
    surviving_count = len(kept_factors)
    min_required = int(np.ceil(configured_count * min_surviving_ratio))
    audit: Dict[str, Any] = {
        "configured_factor_count": configured_count,
        "surviving_factor_count": surviving_count,
        "survival_ratio": surviving_count / configured_count if configured_count else 0.0,
        "min_surviving_ratio": float(min_surviving_ratio),
        "min_required_factor_count": min_required,
        "max_missing_rate": float(max_missing_rate),
        "missing_rate_by_factor": missing_rate_by_factor,
        "removed_factors": [str(name) for name, _ in removed_factors],
    }
    if min_surviving_ratio > 0 and surviving_count < min_required:
        raise InsufficientSurvivingFactorsError(audit)
    return (df[kept_factors], [f[0] for f in removed_factors], audit) if return_audit else (
        df[kept_factors], [f[0] for f in removed_factors]
    )

def _filter_by_pool_date_range(
    index: pd.MultiIndex,
    daily_pools: Dict,
    logger
) -> list:
    """
    Filter MultiIndex by pool membership using date range matching.
    
    For weekly/monthly pools, uses the most recent rebalance date's pool
    for all subsequent dates until the next rebalance.
    
    Args:
        index: MultiIndex of (date, stock_code)
        daily_pools: Dict mapping rebalance dates to stock sets
        logger: Logger instance
        
    Returns:
        List of valid (date, stock_code) tuples
    """
    pool_dates = []
    pool_date_to_key = {}
    
    for k in daily_pools.keys():
        dt = pd.to_datetime(k)
        pool_dates.append(dt)
        pool_date_to_key[dt] = k
    
    pool_dates = sorted(pool_dates)
    
    if not pool_dates:
        logger.warning("[动态池] 无调仓日期，返回全部索引")
        return list(index)
    
    # Convert/resolve each distinct input date once. Scalar conversion keeps
    # mixed date-string formats compatible with the original per-row path.
    pools_by_date = {}
    pools_by_rebalance = {}
    before_first = set()
    for value in index.get_level_values(0).unique():
        date = pd.to_datetime(value)
        pos = bisect_right(pool_dates, date)
        if pos == 0:
            before_first.add(value)
            pools_by_date[value] = set()
            continue
        applicable_date = pool_dates[pos - 1]
        if applicable_date not in pools_by_rebalance:
            pool = daily_pools.get(pool_date_to_key[applicable_date])
            pools_by_rebalance[applicable_date] = set(pool) if pool else set()
        pools_by_date[value] = pools_by_rebalance[applicable_date]

    # Iterate the original index to preserve ordering, duplicates and tuple values.
    valid_indices = [idx for idx in index if idx[1] in pools_by_date[idx[0]]]
    if before_first:
        miss_count = int(index.get_level_values(0).isin(before_first).sum())
        logger.debug("[动态池] %d 个样本在首个调仓日之前，已跳过", miss_count)
    logger.info("[动态池] %d 个调仓日期，过滤 %d -> %d 行",
                len(pool_dates), len(index), len(valid_indices))

    return valid_indices


def _blend_with_original(
    predictions: pd.Series,
    factor_df: pd.DataFrame,
    weights: Dict[str, float],
    blend_ratio: float,
    logger,
    daily_pools: Optional[Dict] = None,
    original_composite: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Blend model predictions with weighted original factor composite.
    
    Args:
        predictions: Model predictions
        factor_df: Original factor data
        weights: Factor weights for blending
        blend_ratio: Ratio of original factors (0-1)
        logger: Logger instance
        daily_pools: Optional dynamic pool dict
        
    Returns:
        Blended predictions
    """
    if blend_ratio <= 0 or (not weights and original_composite is None):
        return predictions

    if original_composite is not None:
        parent = original_composite.reindex(predictions.index)
        valid = parent.notna()
        if not valid.any():
            raise ValueError("parent baseline has no prediction-period overlap")
        return predictions.loc[valid] * (1 - blend_ratio) + parent.loc[valid] * blend_ratio

    try:
        original_composite = pd.Series(0.0, index=predictions.index)
        
        # [FIX] Cloud Alignment: Do NOT use daily_pools for blend factor zscore!
        # Cloud (advanced_small_cap_factors_lasso.py lines 1373-1379) calculates
        # zscore using valid_stocks (= predictions.index), not the full pool.
        
        for factor_name, weight in weights.items():
            if factor_name in factor_df.columns:
                factor_vals = factor_df[factor_name].reindex(predictions.index)
                
                # [CLOUD ALIGNED] Simple cross-sectional zscore on predictions.index
                factor_vals = factor_vals.groupby(level="date").transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
                
                # [FIX] Fill NaN with 0 (mean value for z-scored data) to preserve stocks
                nan_count = factor_vals.isna().sum()
                if nan_count > 0:
                    logger.debug(
                        "[混合] 因子 %s 有 %d 个 NaN 值，填充为 0 (均值)",
                        factor_name, nan_count
                    )
                    factor_vals = factor_vals.fillna(0.0)
                
                original_composite += factor_vals * weight

        result = predictions * (1 - blend_ratio) + original_composite * blend_ratio
        logger.info("[混合] Cloud-aligned blend_ratio=%.2f, 权重: %s", blend_ratio, weights)
        return result
    except Exception as e:
        logger.exception("[混合] 失败: %s", e)
        raise


def _run_blend_only_mode(
    synthesizer,
    factor_df: pd.DataFrame,
    train_end: str,
    config: dict,
    daily_pools: Optional[Dict],
    logger,
    preprocess_config: Optional[Dict] = None,
) -> Tuple[pd.Series, Dict]:
    """
    Run blend-only mode when no LASSO features are selected.
    
    Uses only blend_weights (e.g., DemoSignal1) to generate predictions.
    This ensures the window is not skipped when blend_ratio > 0.
    
    Args:
        synthesizer: FactorSynthesizer instance
        factor_df: Full factor DataFrame
        train_end: Training end date (model reference date)
        config: Synthesis configuration
        daily_pools: Optional dynamic pool dict
        logger: Logger instance
        preprocess_config: Optional preprocessing configuration used during training
        
    Returns:
        composite_series: Pure blend-weighted composite
        model_info: Model information dict with empty weights but valid blend_weights
    """
    blend_ratio = float(config.get("blend_with_original_ratio", 0.0))
    blend_weights = config.get("blend_weights", {})
    
    # Prepare prediction period data
    predict_start = pd.to_datetime(train_end) + timedelta(days=1)
    predict_end = predict_start + timedelta(
        days=synthesizer.config["rolling_window"]["predict_window_days"]
    )
    
    parent_composite = _parent_baseline_scores(config)
    if parent_composite is not None:
        dates = pd.to_datetime(parent_composite.index.get_level_values("date"))
        parent_composite = parent_composite.loc[(dates >= predict_start) & (dates <= predict_end)]
        predict_factors = parent_composite.rename("parent_baseline").to_frame()
    else:
        predict_mask = (factor_df.index.get_level_values("date") >= predict_start) & (
            factor_df.index.get_level_values("date") <= predict_end
        )
        predict_factors = factor_df.loc[predict_mask]
    
    if predict_factors.empty:
        logger.warning("[Blend-Only] 预测期无数据")
        return pd.Series(dtype=float), {}
    
    # Apply dynamic pool filter AND ensure all pool stocks are included
    if parent_composite is None and daily_pools and not predict_factors.empty:
        # Get all (date, stock) pairs that SHOULD be in the prediction
        from bisect import bisect_right
        
        pool_dates = sorted([pd.to_datetime(k) for k in daily_pools.keys()])
        pool_date_to_key = {pd.to_datetime(k): k for k in daily_pools.keys()}
        
        # Build complete index based on pool membership
        all_pool_indices = []
        existing_factor_dates = predict_factors.index.get_level_values('date').unique()
        
        for date in existing_factor_dates:
            date_dt = pd.to_datetime(date)
            # Find applicable pool
            pos = bisect_right(pool_dates, date_dt)
            if pos == 0:
                continue
            applicable_date = pool_dates[pos - 1]
            original_key = pool_date_to_key[applicable_date]
            pool = daily_pools.get(original_key)
            
            if pool:
                for stock in pool:
                    all_pool_indices.append((date, stock))
        
        if all_pool_indices:
            # Create a complete MultiIndex covering all pool stocks
            complete_index = pd.MultiIndex.from_tuples(all_pool_indices, names=['date', 'stock_code'])
            
            # Reindex factor data to include all pool stocks (fill missing with NaN, then fillna(0))
            input_row_count = len(predict_factors)
            predict_factors = predict_factors.reindex(complete_index)
            
            logger.info(
                "[Blend-Only] 动态池扩展: 因子数据 %d 行 -> 完整池 %d 行 (%d 行因子全缺失)",
                input_row_count,
                len(complete_index),
                len(complete_index) - len(predict_factors.dropna(how='all'))
            )
        else:
            # Fallback: filter existing indices
            valid_indices = _filter_by_pool_date_range(
                predict_factors.index, daily_pools, logger
            )
            predict_factors = predict_factors.loc[valid_indices]
    
    if predict_factors.empty:
        logger.warning("[Blend-Only] 过滤后预测期无数据")
        return pd.Series(dtype=float), {}

    if parent_composite is not None:
        composite_series = parent_composite
    else:
        methods = preprocess_config or config.get("preprocess", {})
        blend_preprocess = {
            factor: methods.get(factor, "zscore") for factor in blend_weights
        }
        processed_blend = synthesizer._apply_feature_preprocessing(
            predict_factors[list(blend_weights)],
            blend_preprocess,
        )
        predict_factors = predict_factors.copy()
        predict_factors[list(blend_weights)] = processed_blend

        model_prediction = pd.Series(0.0, index=predict_factors.index)
        composite_series = _blend_with_original(
            model_prediction, predict_factors, blend_weights, blend_ratio, logger, daily_pools
        )
    
    # [NOTE] No post-process standardization here!
    # Reason: We use "standardize first, then blend" pattern.
    # - Model prediction is already zscore'd (zeros in blend-only mode)
    # - Blend factors are already zscore'd inside _blend_with_original
    # - Applying zscore AFTER blend would distort the blend_ratio semantics
    #   (e.g., zscore(0.6 * DemoSignal1) == zscore(DemoSignal1), losing the 0.6 weight meaning)
    
    logger.info(
        "[Blend-Only] 先标准化后混合模式: blend_ratio=%.2f, 保持权重语义", blend_ratio
    )
    
    # Build model info with empty weights but valid metadata
    training_factor_names = list(config.get("_training_factor_names", []))
    model_info = {
        "train_end": train_end,
        "validation_start": predict_start.strftime("%Y-%m-%d"),
        "validation_end": predict_end.strftime("%Y-%m-%d"),
        "weights": {},  # Empty weights - pure blend mode
        "coefficients": {name: 0.0 for name in training_factor_names},
        "feature_names": [],
        "n_features": len(training_factor_names),
        "n_selected": 0,
        "r2_score": 0.0,
        "alpha": 0.0,
        "blend_only": True,  # Flag to indicate blend-only mode
        "standardize": config.get("standardize", True),
        "standardization_method": config.get("preprocess", {}),
        "postprocess_cs_standardize": config.get("postprocess_cs_standardize", True),
        "blend_with_original_ratio": blend_ratio,
        "blend_weights": blend_weights,
        # Add preprocessing config for bundle export (from distribution analysis)
        "preprocess_config": preprocess_config if preprocess_config else config.get("preprocess", {}),
        # Blend info for consistency with regular flow
        "blend_info": {
            "static_blend_ratio": blend_ratio,
            "dynamic_blend_enabled": False,
            "blend_ratio_used": blend_ratio,
            "blend_only_mode": True,
        },
    }
    
    logger.info(
        "[Blend-Only] 使用纯 blend_weights 模式生成合成因子, 预测样本数: %d",
        len(composite_series)
    )
    
    return composite_series, model_info


def _run_lasso_with_qlib_adapter(
    synthesizer,
    X: pd.DataFrame,
    y: pd.Series,
    predict_factors: pd.DataFrame,
    train_start: str,
    train_end: str,
    config: dict,
    logger,
    sample_weight: Optional[np.ndarray] = None,
    y_raw: Optional[pd.Series] = None,
    baseline_scores: Optional[pd.Series] = None,
) -> Tuple[pd.Series, Dict]:
    """
    Run LASSO/ElasticNet training and prediction using Qlib adapter.
    
    Args:
        synthesizer: FactorSynthesizer instance
        X: Preprocessed training features (MultiIndex: date, stock_code)
        y: Preprocessed training labels
        predict_factors: Prediction period factor data
        train_start/end: Training period dates
        config: Synthesis configuration
        logger: Logger instance
        sample_weight: Optional sample weights
        
    Returns:
        composite_series: Prediction Series
        model_info: Model information dict
    """
    n_outer_samples = len(X)
    train_keys, valid_keys = _resolve_inner_train_valid_indices(X, config, logger)
    keep = X.index.isin(train_keys) | X.index.isin(valid_keys)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)[keep]
    X = X.loc[keep]
    y = y.reindex(X.index)
    train_end_adj = train_keys.get_level_values("date").max().strftime("%Y-%m-%d")
    valid_start = valid_keys.get_level_values("date").min().strftime("%Y-%m-%d")
    valid_end = valid_keys.get_level_values("date").max().strftime("%Y-%m-%d")
    
    # Prediction period dates
    predict_dates = predict_factors.index.get_level_values("date").unique().sort_values()
    test_start = predict_dates[0].strftime("%Y-%m-%d") if len(predict_dates) > 0 else valid_end
    test_end = predict_dates[-1].strftime("%Y-%m-%d") if len(predict_dates) > 0 else valid_end
    
    logger.info(
        "[Qlib Adapter] 创建数据集: train=%s~%s, valid=%s~%s, test=%s~%s",
        train_start, train_end_adj, valid_start, valid_end, test_start, test_end
    )
    
    # Merge training and prediction data
    all_factors = pd.concat([X, predict_factors], axis=0)
    all_factors = all_factors[~all_factors.index.duplicated(keep='first')]
    
    neutralization_cfg = config.get("neutralization", {})
    if neutralization_cfg.get("enabled", False):
        logger.info("[中性化] 使用加载阶段的预计算中性化输入；跳过训练阶段市值回归")
    
    # Labels
    all_labels = y.reindex(all_factors.index)
    
    # Create Qlib-compatible dataset
    dataset = DataConverter.create_dataset_from_factors(
        factor_df=all_factors,
        forward_returns=all_labels,
        train_start=train_start,
        train_end=train_end_adj,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
        config=config,
        logger=logger
    )
    
    # Attach raw returns for return-based sample weighting
    # (weights must be computed from raw returns, not preprocessed labels)
    # Must convert to Qlib index format (datetime, instrument) to match dataset
    if y_raw is not None:
        raw_aligned = y_raw.reindex(all_factors.index).dropna()
        if not raw_aligned.empty:
            qlib_idx = DataConverter.convert_index_to_qlib_format(raw_aligned.index)
            dataset.raw_returns = pd.Series(
                raw_aligned.values, index=qlib_idx, name="raw_returns"
            )
    if baseline_scores is not None:
        baseline_aligned = baseline_scores.reindex(all_factors.index).dropna()
        if not baseline_aligned.empty:
            qlib_idx = DataConverter.convert_index_to_qlib_format(
                baseline_aligned.index
            )
            dataset.baseline_scores = pd.Series(
                baseline_aligned.values, index=qlib_idx, name="baseline_scores"
            )
    
    # Create and train model
    model = QlibLinearWrapper(config, logger)
    model_info = model.fit(dataset, sample_weight=sample_weight)
    
    # Predict
    predictions = model.predict(dataset, segment="test")
    
    # Convert back to original format
    composite_series = DataConverter.convert_predictions_to_series(
        predictions, predict_factors.index
    )
    composite_series.name = "lasso_composite"
    
    # Add model info
    model_info.update({
        "train_start": train_start,
        "train_end": train_end,
        "n_train_samples": n_outer_samples,
        "label_isolation_version": "strict_label_exit_v2",
        "validation_start": test_start,
        "validation_end": str(
            pd.to_datetime(predict_factors.index.get_level_values(0)).max().date()
        ),
        "qlib_adapter": True,
    })
    
    return composite_series, model_info


def run_lasso_synthesis(
    synthesizer,
    factor_df: pd.DataFrame,
    forward_returns: pd.Series,
    train_start: str,
    train_end: str,
    config: dict,
    daily_pools: Optional[Dict] = None,
) -> Tuple[pd.Series, Dict]:
    """
    LASSO / ElasticNet based factor synthesis using Qlib adapter.
    
    Args:
        synthesizer: FactorSynthesizer instance
        factor_df: Factor data DataFrame
        forward_returns: Forward returns Series
        train_start: Training start date
        train_end: Training end date
        config: Synthesis configuration
        daily_pools: Optional dynamic pool dict
        
    Returns:
        composite_series: Synthesized factor Series
        model_info: Model information dict
    """
    logger = synthesizer.logger
    DataConverter.convert_index_to_qlib_format(factor_df.index)
    DataConverter.convert_index_to_qlib_format(forward_returns.index)

    def baseline_fallback(reason, preprocess_config=None, audit=None):
        _require_complete_research_window(config, reason)
        if not (config.get("parent_baseline") or config.get("blend_weights")):
            return pd.Series(dtype=float), {}
        composite, info = _run_blend_only_mode(
            synthesizer,
            factor_df,
            train_end,
            config,
            daily_pools,
            logger,
            preprocess_config,
        )
        if info:
            info.update(
                train_start=train_start,
                selected_residual_gamma=0.0,
                cv_fallback_reason=reason,
            )
            if audit is not None:
                info["training_window_audit"] = audit
        return composite, info

    # Training window (with embargo to avoid look-ahead bias)
    # Purge by the configured label horizon in trading days. This equals
    # target_period for legacy labels and may be longer for rebalance labels.
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)

    # Actual prediction membership defines label availability, not sparse history.
    prediction_index = _outer_prediction_index(
        synthesizer, factor_df, train_end, daily_pools, logger
    )
    if not len(prediction_index):
        return pd.Series(dtype=float), {}
    prediction_boundary = prediction_index.get_level_values("date").min()
    train_end_adj = train_end_dt

    dates = factor_df.index.get_level_values("date")
    train_mask = (dates >= train_start_dt) & (dates <= train_end_adj)

    train_factors = factor_df.loc[train_mask]
    
    # [DEBUG] 诊断第一个训练窗口数据问题
    if train_factors.empty:
        all_dates = pd.to_datetime(dates.unique()).sort_values()
        logger.warning(
            "[DEBUG] train_factors 为空! "
            f"factor_df 日期范围: {all_dates.min().date()} ~ {all_dates.max().date()}, "
            f"请求范围: {train_start} ~ {train_end_adj.strftime('%Y-%m-%d')}"
        )
    else:
        train_dates = train_factors.index.get_level_values('date').unique()
        logger.info(
            f"[DEBUG] train_factors: {len(train_factors)} 行, "
            f"日期范围: {train_dates.min().date()} ~ {train_dates.max().date()}"
        )
    
    # [关键] 仅使用训练因子列进行 LASSO 训练（排除仅用于混合的因子）
    training_factor_names = config.get('_training_factor_names')
    if training_factor_names:
        available_training_cols = [c for c in training_factor_names if c in train_factors.columns]
        excluded_cols = set(train_factors.columns) - set(available_training_cols)
        if excluded_cols:
            logger.info(
                "[LASSO训练] 排除 %d 个仅用于混合的因子: %s",
                len(excluded_cols), list(excluded_cols)
            )
        train_factors = train_factors[available_training_cols]
        logger.info("[LASSO训练] 使用 %d 个训练因子", len(available_training_cols))
    
    # Align target
    common_index = train_factors.index.intersection(forward_returns.index)
    if len(common_index):
        common_index = _strict_label_exit_index(common_index, config, prediction_boundary)
    X_raw = train_factors.loc[common_index]
    y = forward_returns.loc[common_index]
    metric_returns = y.copy()
    baseline_scores: Optional[pd.Series] = None

    if len(X_raw) == 0 or len(y) == 0:
        # [DEBUG] 详细诊断无有效数据的原因
        logger.warning("训练期 %s 到 %s 无有效数据", train_start, train_end)
        logger.warning(
            f"[DEBUG] train_factors: {len(train_factors)} 行, "
            f"forward_returns: {len(forward_returns)} 行, "
            f"common_index: {len(common_index)} 行"
        )
        if len(train_factors) > 0:
            tf_dates = train_factors.index.get_level_values('date').unique()
            logger.warning(f"[DEBUG] train_factors 日期: {tf_dates.min().date()} ~ {tf_dates.max().date()}")
        if len(forward_returns) > 0:
            fr_dates = forward_returns.index.get_level_values('date').unique()
            logger.warning(f"[DEBUG] forward_returns 日期: {fr_dates.min().date()} ~ {fr_dates.max().date()}")
        return baseline_fallback("training_data_unavailable")

    # [数据质量] 按训练窗口过滤高缺失率因子
    data_quality_cfg = config.get('data_quality', {})
    max_missing_rate = data_quality_cfg.get('max_factor_missing_rate', 0.05)
    min_surviving_ratio = float(data_quality_cfg.get('min_surviving_factor_ratio', 0.0))
    configured_factor_count = len(training_factor_names or X_raw.columns)
    training_window_audit: Dict[str, Any] = {
        "configured_factor_count": configured_factor_count,
        "surviving_factor_count": len(X_raw.columns),
        "survival_ratio": len(X_raw.columns) / configured_factor_count if configured_factor_count else 0.0,
        "min_surviving_ratio": min_surviving_ratio,
        "min_required_factor_count": int(np.ceil(configured_factor_count * min_surviving_ratio)),
        "max_missing_rate": float(max_missing_rate),
        "missing_rate_by_factor": {},
        "removed_factors": [],
    }
    X_raw, removed_factors, training_window_audit = _filter_by_missing_rate(
        X_raw,
        max_missing_rate,
        logger,
        min_surviving_ratio=min_surviving_ratio,
        configured_factor_count=configured_factor_count,
        configured_factor_names=training_factor_names,
        return_audit=True,
    )
    if X_raw.empty:
        logger.warning("训练窗口 %s~%s: 过滤后无因子剩余", train_start, train_end)
        return baseline_fallback("no_surviving_features", audit=training_window_audit)

    # NOTE: neutralization.enabled loads precomputed market-cap-neutralized inputs.

    # =========================================================================
    # Phase 1: 分布分析 + 预处理配置（在筛选之前！）
    # =========================================================================
    dist_cfg = config.get('distribution_prefilter', {})
    if dist_cfg.get('enabled', False):
        from factor_synthesis.preselection.distribution_analyzer import (
            DistributionAnalyzer,
            DistributionConfig,
        )
        
        logger.info("[分布分析] 启用偏度/峰度分类预处理")
        
        analyzer = DistributionAnalyzer(
            config=DistributionConfig(
                skew_normal_threshold=dist_cfg.get('skew_normal_threshold', 2.0),
                kurtosis_normal_threshold=dist_cfg.get('kurtosis_normal_threshold', 6.0),
                skew_event_threshold=dist_cfg.get('skew_event_threshold', 3.0),
                kurtosis_event_threshold=dist_cfg.get('kurtosis_event_threshold', 10.0),
                binning_n_bins=dist_cfg.get('binning_n_bins', 5),
                winsorize_percentile=dist_cfg.get('winsorize_percentile', 0.025),
                min_samples_per_date=dist_cfg.get('min_samples_per_date', 10),
                max_sample_dates=dist_cfg.get('max_sample_dates', 100),
                use_vectorized=dist_cfg.get('use_vectorized', True),
            ),
            custom_logger=logger,
        )
        
        # 分析因子分布
        dist_report = analyzer.analyze(X_raw)
        logger.info("[分布配置] distribution_config_version=sampling_controls_v2")
        
        # 获取 Config 中的显式覆盖配置
        config_preprocess = config.get('preprocess', {}).copy()
        config_overrides = {k: v for k, v in config_preprocess.items() if k != 'ALL'}
        
        # 构建最终预处理配置: 分布分析默认 + Config覆盖
        # 线性模型下 event_driven 因子用 winsorize_zscore 保留连续信号
        final_preprocess_cfg = analyzer.build_preprocess_config(
            dist_report.results,
            config_overrides,
            model_type=config.get("model_type", "lasso"),
        )
        
        # 添加分箱/缩尾参数
        final_preprocess_cfg['_binning_n_bins'] = dist_cfg.get('binning_n_bins', 5)
        final_preprocess_cfg['_winsorize_percentile'] = dist_cfg.get('winsorize_percentile', 0.025)
        
        logger.info(
            f"[分布分析] 预处理配置: {dist_report.category_counts}"
        )
    else:
        # 未启用分布分析，使用原 Config 配置
        final_preprocess_cfg = config.get('preprocess', config.get('preprocessing', {}))
    
    # =========================================================================
    # Phase 2: 统一预处理（在筛选之前！）
    # =========================================================================
    if isinstance(final_preprocess_cfg, dict) and final_preprocess_cfg:
        X_preprocessed = synthesizer._apply_feature_preprocessing(X_raw, final_preprocess_cfg)
        logger.info(f"[预处理] 已对 {len(X_raw.columns)} 个因子执行预处理（筛选前）")
    else:
        X_preprocessed = X_raw.copy()

    target_residualization = config.get("target_residualization", {})
    target_residualization_applied = False
    if target_residualization.get("enabled", False):
        configured = target_residualization.get("baseline_factors")
        baseline_factors = [str(item) for item in configured] if isinstance(configured, list) else []
        if not baseline_factors:
            baseline_factors = [str(target_residualization.get("baseline_factor", "")).strip()]
        if any(factor in X_preprocessed.columns for factor in baseline_factors):
            raise ValueError(
                "target_residualization baseline factors must be excluded from "
                "penalized training features"
            )
        baseline = _target_residual_baseline(synthesizer, factor_df, y.index, config)
        baseline_scores = baseline
        y, target_residualization_applied = _maybe_residualize_training_target(
            y, baseline, target_residualization
        )
        if target_residualization_applied:
            logger.info(
                "[目标残差化] 已移除 %s 的同日截面暴露",
                "父代复合信号" if _parent_baseline_scores(config) is not None else baseline_factors,
            )
            if not y.notna().any():
                return baseline_fallback(
                    "parent_baseline_history_unavailable",
                    final_preprocess_cfg,
                    training_window_audit,
                )
        else:
            logger.info("[目标残差化诊断] 保留原始训练目标，仅将基线用于组合 DSR 评分")

    # =========================================================================
    # Phase 3: 预筛选（接收预处理后的因子！）
    # =========================================================================
    from factor_synthesis.preselection import get_preselection_strategy
    
    # [NEW] 预筛选方法互斥检查
    _validate_preselection_exclusivity(config)
    
    # 检查三种预筛选方法
    barra_cfg = config.get('barra_preselection', {})
    pre_sel_cfg = config.get('pre_selection', {})
    hierarchical_cfg = config.get('hierarchical_clustering', {})
    preselection_diagnostics: Dict[str, Any] = {}
    
    if barra_cfg.get('enabled', False):
        # Barra 分组预筛选 - 忽略传统 pre_selection 和 factor_groups
        from factor_synthesis.preselection.barra_grouped_strategy import BarraGroupedPreSelector
        
        config_path = barra_cfg.get('config_path', './config/barra_groups.json')
        group_overrides = barra_cfg.get('group_overrides', {})
        cross_group_cfg = barra_cfg.get('cross_group_correlation_filter', {})
        
        logger.info("[Barra预筛选] 启用分组因子筛选（接收预处理后数据）")
        
        selector = BarraGroupedPreSelector(
            config_path=config_path,
            group_overrides=group_overrides,
            cross_group_config=cross_group_cfg,
            logger=logger
        )
        # [CHANGED] 传入预处理后的因子 X_preprocessed
        selected_features = selector.select(X_preprocessed, y)
        
        # 记录分组报告 (debug级别)
        group_report = selector.get_group_report()
        if not group_report.empty:
            logger.debug(f"[Barra预筛选] 分组筛选报告:\n{group_report.to_string()}")
        
    elif hierarchical_cfg.get('enabled', False):
        # 层次聚类预筛选 - 激进降维到15-20个因子
        logger.info("[层次聚类] 启用层次聚类预筛选（接收预处理后数据）")
        if hierarchical_cfg.get("nested_distance_thresholds"):
            # The Qlib model must see the complete reservoir so every candidate
            # tree can be rebuilt on the training side of each purged fold.
            selected_features = list(X_preprocessed.columns)
            preselection_diagnostics = {
                "strategy": "nested_hierarchical_clustering",
                "candidate_thresholds": list(
                    hierarchical_cfg["nested_distance_thresholds"]
                ),
                "selection_scope": "inner_train_only",
            }
        else:
            strategy = get_preselection_strategy(
                'hierarchical_clustering', hierarchical_cfg, logger
            )
            selected_features = strategy.select(X_preprocessed, y)

            logger.info(f"[层次聚类] 筛选完成，保留 {len(selected_features)} 个代表因子")
            stability_cfg = hierarchical_cfg.get('stability_selection', {})
            if stability_cfg.get('enabled', False) and selected_features:
                stability = get_preselection_strategy(
                    'stability_selection', stability_cfg, logger
                )
                selected_features = stability.select(
                    X_preprocessed[selected_features], y
                )
                logger.info(
                    "[连续时间块稳定筛选] 聚类代表因子中保留 %d 个",
                    len(selected_features),
                )
        
    elif pre_sel_cfg.get('enabled', False):
        # 传统预筛选 (IC/IR/dCor)
        strategy_name = pre_sel_cfg.get('strategy', 'ic_ir')
        strategy = get_preselection_strategy(strategy_name, pre_sel_cfg, logger)
        if strategy_name == "portfolio_boundary_replacement":
            inner_keys, _ = _resolve_inner_train_valid_indices(
                X_preprocessed, config, logger
            )
            inner_X = X_preprocessed.loc[inner_keys]
            inner_returns = metric_returns.reindex(inner_X.index)
            if baseline_scores is None:
                baseline_factor = str(pre_sel_cfg.get("baseline_factor", "")).strip()
                if baseline_factor not in factor_df.columns:
                    baseline_for_selection = None
                else:
                    preprocess = final_preprocess_cfg
                    preprocess_rule = (
                        preprocess.get(baseline_factor, "zscore")
                        if isinstance(preprocess, dict)
                        else "zscore"
                    )
                    baseline_frame = synthesizer._apply_feature_preprocessing(
                        factor_df[[baseline_factor]].reindex(inner_X.index),
                        {baseline_factor: preprocess_rule},
                    )
                    baseline_for_selection = baseline_frame[baseline_factor]
            else:
                baseline_for_selection = baseline_scores.reindex(inner_X.index)
            selected_features = strategy.select(
                inner_X,
                inner_returns,
                baseline_scores=baseline_for_selection,
            )
        else:
            # Existing strategies retain their original outer-window inputs.
            selected_features = strategy.select(X_preprocessed, y)
        report = strategy.get_performance_report()
        preselection_diagnostics = {
            "strategy": strategy_name,
            "selected_features": list(selected_features),
            "summary": dict(report.attrs),
            "factor_statistics": report.to_dict(orient="records"),
        }
    else:
        selected_features = list(X_preprocessed.columns)
        
    # [FIX] 当无特征被选中时，检查是否可以使用 blend_weights 进行纯混合
    blend_ratio = float(config.get("blend_with_original_ratio", 0.0))
    blend_weights = config.get("blend_weights", {})
    
    if not selected_features:
        if blend_ratio > 0 and (blend_weights or config.get("parent_baseline")):
            logger.warning("预筛选后无有效因子，将使用纯 blend_weights 模式 (blend_ratio=%.2f)", blend_ratio)
            composite_series, model_info = baseline_fallback(
                "preselection_empty",
                final_preprocess_cfg,
                training_window_audit,
            )
            if model_info:
                model_info["preselection_diagnostics"] = preselection_diagnostics
            return composite_series, model_info
        else:
            logger.warning("预筛选后无有效因子且无 blend_weights，跳过本窗口")
            return pd.Series(dtype=float), {}

    # =========================================================================
    # Phase 4: 筛选后的因子准备（无二次预处理！）
    # =========================================================================
    # [CHANGED] 使用已预处理的因子，不再进行二次预处理
    X_selected = X_preprocessed[selected_features]

    # Cross-sectional processing of the target (using Qlib Processor)
    y_raw: Optional[pd.Series] = metric_returns.reindex(y.index)
    try:
        from .preprocessing import apply_target_cs_preprocessing
        
        target_cs_demean = bool(config.get("target_cs_demean", True))
        target_cs_standardize = bool(config.get("target_cs_standardize", False))
        target_cs_rank = bool(config.get("target_cs_rank", False))
        eps = float(config.get("target_cs_eps", 1e-8))

        if target_cs_demean or target_cs_standardize or target_cs_rank:
            y_mean_before = float(y.mean())
            y_std_before = float(y.std())

            y = apply_target_cs_preprocessing(
                y,
                cs_demean=target_cs_demean,
                cs_rank=target_cs_rank,
                cs_standardize=target_cs_standardize,
                eps=eps,
                logger=logger,
            )

            logger.info(
                "[目标处理] y统计变化: 均值 %.6f → %.6f, 标准差 %.6f → %.6f",
                y_mean_before,
                float(y.mean()),
                y_std_before,
                float(y.std()),
            )
    except Exception as e:
        logger.exception("[目标处理] 按日截面处理失败: %s", e)
        raise

    # [REMOVED] rank_before_standardize 已在 Phase 2 统一预处理中处理
    # 不再需要单独的 rank 步骤
    X = X_selected

    # Drop-NaN: LASSO (sklearn) requires complete data
    # Drop rows where any feature or target is NaN
    initial_len = len(X)
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    dropped = initial_len - len(X)
    if dropped > 0:
        logger.info(
            "[Drop-NaN] 训练数据: 丢弃 %d 行含缺失值的样本 (%.2f%%)",
            dropped, dropped / initial_len * 100
        )

    if len(X) < synthesizer.config["rolling_window"]["min_train_samples"]:
        logger.warning("训练样本不足: %d", len(X))
        return baseline_fallback(
            "insufficient_training_samples",
            final_preprocess_cfg,
            training_window_audit,
        )

    # Time-decay sample weights (using Qlib Processor)
    from .preprocessing import compute_sample_weights
    
    weight_decay = config.get("sample_weight_decay", 1.0)
    half_life = config.get("sample_weight_half_life", 0.0)
    sample_weight = compute_sample_weights(
        X, weight_decay=weight_decay, half_life=half_life, logger=logger
    )


    # Prepare prediction period data
    logger.info("=" * 60)
    logger.info("[LASSO] 使用 Qlib 适配器模式")
    logger.info("=" * 60)
    
    try:
        predict_start = pd.to_datetime(train_end) + timedelta(days=1)
        predict_end = predict_start + timedelta(
            days=synthesizer.config["rolling_window"]["predict_window_days"]
        )
        
        predict_mask = (factor_df.index.get_level_values("date") >= predict_start) & (
            factor_df.index.get_level_values("date") <= predict_end
        )
        predict_factors_raw = factor_df.loc[predict_mask]
        
        # [DEBUG] Check predict_factors_raw before filter
        if '2023-03-14' in predict_factors_raw.index.get_level_values('date').strftime('%Y-%m-%d').values:
            date_mask = predict_factors_raw.index.get_level_values('date').strftime('%Y-%m-%d') == '2023-03-14'
            stocks_before = set(predict_factors_raw.loc[date_mask].index.get_level_values('stock_code'))
            logger.info(f"[DEBUG] 2023-03-14 动态池过滤前: {len(stocks_before)} 只股票")
            logger.info(f"[DEBUG] 002087.XSHE in predict_factors_raw: {'002087.XSHE' in stocks_before}")
        
        if selected_features:
            predict_factors_raw = predict_factors_raw[selected_features]
        
        # Apply dynamic pool filter
        if daily_pools and not predict_factors_raw.empty:
            valid_indices = _filter_by_pool_date_range(
                predict_factors_raw.index, daily_pools, logger
            )
            predict_factors_raw = predict_factors_raw.loc[valid_indices]
            
            # [DEBUG] Check predict_factors_raw after filter
            if '2023-03-14' in predict_factors_raw.index.get_level_values('date').strftime('%Y-%m-%d').values:
                date_mask = predict_factors_raw.index.get_level_values('date').strftime('%Y-%m-%d') == '2023-03-14'
                stocks_after = set(predict_factors_raw.loc[date_mask].index.get_level_values('stock_code'))
                logger.info(f"[DEBUG] 2023-03-14 动态池过滤后: {len(stocks_after)} 只股票")
                logger.info(f"[DEBUG] 002087.XSHE after filter: {'002087.XSHE' in stocks_after}")
            else:
                logger.info("[DEBUG] 2023-03-14 不在过滤后的数据中!")

        
        # NOTE: neutralization.enabled loads precomputed market-cap-neutralized inputs.
        
        # =========================================================================
        # 预测期预处理：使用与训练期相同的预处理配置 final_preprocess_cfg
        # =========================================================================
        if isinstance(final_preprocess_cfg, dict) and final_preprocess_cfg:
            predict_factors_preprocessed = synthesizer._apply_feature_preprocessing(
                predict_factors_raw, final_preprocess_cfg
            )
            logger.debug(f"[预测期预处理] 已对 {len(predict_factors_raw.columns)} 个因子执行预处理")
        else:
            predict_factors_preprocessed = predict_factors_raw.copy()
        
        # [REMOVED] rank_before_standardize 已在上面统一预处理中处理
        predict_factors_processed = predict_factors_preprocessed
        
        # [Modified] Identify rows with NaN -> Fill with 0 instead of Drop
        nan_mask = predict_factors_processed.isna().any(axis=1)
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            logger.info(
                "[Fill-NaN] 预测数据: %d 行含缺失值 (%.2f%%), 填充为 0 (均值) 以保留样本",
                nan_count, nan_count / len(predict_factors_processed) * 100
            )
            # Fill NaNs with 0.0 (Mean value for Z-scored data)
            predict_factors = predict_factors_processed.fillna(0.0)
        else:
            predict_factors = predict_factors_processed
        
        if len(predict_factors) == 0:
            logger.warning("[Qlib] 预测期无数据")
            return baseline_fallback(
                "prediction_features_unavailable",
                final_preprocess_cfg,
                training_window_audit,
            )
        
        # Use Qlib adapter
        composite_series, model_info = _run_lasso_with_qlib_adapter(
            synthesizer=synthesizer,
            X=X,
            y=y,
            predict_factors=predict_factors,
            train_start=train_start,
            train_end=train_end,
            config=config,
            logger=logger,
            sample_weight=sample_weight,
            y_raw=y_raw,
            baseline_scores=baseline_scores,
        )
        model_info["training_window_audit"] = training_window_audit
        model_info["preselection_diagnostics"] = preselection_diagnostics
        model_info["target_residualization_applied"] = target_residualization_applied
        model_info["selection_baseline_attached"] = baseline_scores is not None
        _require_complete_research_window(config, model_info.get("cv_fallback_reason"))

        if model_info.get("selected_residual_gamma") == 0.0:
            selection_info = model_info
            composite_series, model_info = _run_blend_only_mode(
                synthesizer=synthesizer,
                factor_df=factor_df,
                train_end=train_end,
                config=config,
                daily_pools=daily_pools,
                logger=logger,
                preprocess_config=final_preprocess_cfg,
            )
            model_info["residual_sleeve_coefficients"] = dict(
                selection_info.get("coefficients", {})
            )
            model_info["residual_sleeve_selected_features"] = list(
                selection_info.get("selected_features", [])
            )
            model_info["residual_sleeve_n_selected"] = int(
                selection_info.get("n_selected", 0)
            )
            for key in (
                "train_start",
                "validation_start",
                "validation_end",
                "alpha",
                "l1_ratio",
                "r2_score",
                "cv_score",
                "cv_metric",
                "cv_results",
                "cv_candidate_count",
                "cv_factor_count",
                "factor_count_grid",
                "cv_folds",
                "selected_residual_gamma",
                "selected_blend_ratio",
                "pooled_oof_periods",
                "cv_dsr_min_threshold",
                "cv_fallback_reason",
                "alpha_grid_min",
                "alpha_grid_max",
                "l1_ratio_grid",
                "nested_hierarchical_cv",
                "preselection_diagnostics",
                "target_residualization_applied",
                "selection_baseline_attached",
            ):
                model_info[key] = selection_info.get(key)
            model_info["coefficients"] = {
                name: 0.0 for name in selection_info.get("feature_names", [])
            }
            model_info["n_features"] = selection_info.get("n_features", 0)
            model_info["n_selected"] = 0
            model_info["training_window_audit"] = training_window_audit
            logger.info(
                "[PooledOOF] 证据不足或纯父代胜出，预测期直接继承父代信号"
            )
            return composite_series, model_info
        
        # Post-processing
        if config.get("postprocess_cs_standardize", False):
            try:
                composite_series = composite_series.groupby(level="date").transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
                logger.info("[后处理] 应用截面标准化")
            except Exception as e:
                logger.exception("[后处理] 截面标准化失败: %s", e)
                raise
        
        # Blend with original factors
        configured_blend_ratio = float(config.get("blend_with_original_ratio", 0.0))
        static_blend_ratio = float(
            model_info.get("selected_blend_ratio") or configured_blend_ratio
        )
        blend_weights = config.get("blend_weights", {})
        dynamic_blend_cfg = config.get("dynamic_blend", {})
        parent_composite = _parent_baseline_scores(config, composite_series.index)
        
        # Initialize blend info for model_info
        blend_info = {
            "static_blend_ratio": configured_blend_ratio,
            "dynamic_blend_enabled": False,
            "blend_ratio_used": static_blend_ratio,
            "selected_residual_gamma": model_info.get("selected_residual_gamma"),
        }
        
        if (static_blend_ratio > 0 or dynamic_blend_cfg.get("enabled", False)) and (
            blend_weights or parent_composite is not None
        ):
            blend_preprocess = {
                factor: config.get("preprocess", {}).get(factor, "zscore")
                for factor in blend_weights
            }
            factors_for_blend = (
                pd.DataFrame(index=composite_series.index)
                if parent_composite is not None
                else synthesizer._apply_feature_preprocessing(
                    factor_df.loc[predict_mask, list(blend_weights)],
                    blend_preprocess,
                )
            )
            
            # Compute dynamic blend ratio if enabled
            if dynamic_blend_cfg.get("enabled", False):
                try:
                    # Create DynamicBlendConfig from dict
                    dyn_config = DynamicBlendConfig(
                        enabled=True,
                        base_ratio=dynamic_blend_cfg.get("base_ratio", 0.5),
                        max_delta=dynamic_blend_cfg.get("max_delta", 0.3),
                        bias=dynamic_blend_cfg.get("bias", 0.0),
                        lookback_window=dynamic_blend_cfg.get("lookback_window", 60),
                        use_icir=dynamic_blend_cfg.get("use_icir", False),
                        smoothing_scale=dynamic_blend_cfg.get("smoothing_scale", 0.1),
                        min_samples=dynamic_blend_cfg.get("min_samples", 20),
                        correlation_aware=dynamic_blend_cfg.get("correlation_aware", False),
                        shrinkage=dynamic_blend_cfg.get("shrinkage", 0.5),
                        topk_ic_pct=dynamic_blend_cfg.get("topk_ic_pct", 0.0),
                    )
                    
                    # Compute prior composite from training data for IC calculation
                    # [FIX] Use original factor_df for blend factors (not X_preprocessed)
                    # X_preprocessed only contains training factors after filtering/preselection
                    # Blend-only factors need to be accessed from the full factor_df
                    if parent_composite is not None:
                        prior_composite_train = _parent_baseline_scores(config, y.index)
                    else:
                        train_factor_df = synthesizer._apply_feature_preprocessing(
                            factor_df.loc[train_mask, list(blend_weights)],
                            blend_preprocess,
                        )
                        prior_composite_train = compute_prior_composite(
                            factor_df=train_factor_df,
                            blend_weights=blend_weights,
                            index=y.index,
                            logger=logger,
                        )
                    
                    # For ML predictions on training data, we need to get the in-sample predictions
                    # However, since we only have out-of-sample predictions (composite_series),
                    # we compute IC on training data using the model coefficients applied to training features
                    # This approximates the ML performance without look-ahead bias
                    coefficients = model_info.get("coefficients", {})
                    if coefficients:
                        # Compute in-sample ML predictions from coefficients
                        # [CHANGED] Use preprocessed features for consistency with training
                        ml_train_pred = pd.Series(0.0, index=y.index)
                        for feat, coef in coefficients.items():
                            if feat in X_preprocessed.columns and abs(coef) > 1e-10:
                                ml_train_pred += X_preprocessed[feat].reindex(y.index).fillna(0.0) * coef
                        
                        # Compute dynamic blend ratio
                        blend_result = compute_dynamic_blend_ratio(
                            ml_predictions=ml_train_pred,
                            prior_composite=prior_composite_train,
                            forward_returns=y,
                            config=dyn_config,
                            logger=logger,
                        )
                        blend_info["sample_contract_version"] = blend_result.sample_contract_version
                        blend_info["diagnostic_version"] = blend_result.diagnostic_version
                        
                        if blend_result.computation_successful:
                            blend_ratio = blend_result.blend_ratio
                            blend_info.update({
                                "dynamic_blend_enabled": True,
                                "blend_ratio_used": blend_ratio,
                                "ml_ic": blend_result.ml_ic,
                                "ml_ic_std": blend_result.ml_ic_std,
                                "prior_ic": blend_result.prior_ic,
                                "prior_ic_std": blend_result.prior_ic_std,
                                "adjustment": blend_result.adjustment,
                                "n_samples": blend_result.n_samples,
                                "signal_correlation": blend_result.signal_correlation,
                                "raw_optimal_ratio": blend_result.raw_optimal_ratio,
                            })
                        else:
                            logger.warning(
                                "[DynamicBlend] Falling back to static ratio: %s",
                                blend_result.fallback_reason
                            )
                            blend_ratio = static_blend_ratio
                    else:
                        logger.warning(
                            "[DynamicBlend] No coefficients available, using static ratio"
                        )
                        blend_ratio = static_blend_ratio
                        
                except Exception as e:
                    logger.exception("[DynamicBlend] Failed: %s", e)
                    raise
            else:
                blend_ratio = static_blend_ratio
            
            # Apply blending with the computed ratio
            if blend_ratio > 0:
                composite_series = _blend_with_original(
                    composite_series,
                    factors_for_blend,
                    blend_weights,
                    blend_ratio,
                    logger,
                    daily_pools,
                    original_composite=parent_composite,
                )
                blend_info["blend_ratio_used"] = blend_ratio
        
        # Add blend info to model_info
        model_info["blend_info"] = blend_info
        model_info["blend_with_original_ratio"] = blend_info["blend_ratio_used"]
        model_info["training_window_audit"] = training_window_audit
        
        # Add preprocessing config to model_info for bundle export
        # This ensures the actual preprocessing methods (from distribution analysis) are saved
        model_info["preprocess_config"] = final_preprocess_cfg
        if dist_cfg.get('enabled', False):
            model_info["distribution_config_version"] = "sampling_controls_v2"
        
        score_key = "pinball_score" if config.get("model_type", "lasso") == "quantile_elasticnet" else "R²"
        logger.info(
            "LASSO回归完成，选择特征数: %d, %s: %.4f, alpha: %.6f, blend_ratio: %.3f",
            model_info.get("n_selected", 0),
            score_key,
            model_info.get("r2_score", 0.0),
            model_info.get("alpha", 0.0),
            blend_info["blend_ratio_used"],
        )
        
        return composite_series, model_info
        
    except (InsufficientSurvivingFactorsError, IncompleteResearchWindowError):
        raise
    except Exception as e:
        logger.exception("LASSO回归失败: %s", e)
        raise
