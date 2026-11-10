# -*- coding: utf-8 -*-
"""
LightGBM Factor Synthesis - Qlib Adapter Only

This module provides LightGBM-based factor synthesis using the Qlib adapter.
All model training and prediction is delegated to QlibLGBWrapper.

[ALIGNED WITH LASSO] This module is kept in sync with lasso_method.py.
Input data flow, preprocessing, embargo, NaN handling, and blending
must match lasso_method.py to ensure comparable results.

Shared pipeline (identical to LASSO):
    1. Embargo cutoff (_compute_embargo_cutoff)
    2. Train mask + filter _training_factor_names
    3. Align target (common_index)
    4. Missing rate filter (_filter_by_missing_rate)
    5. Phase 1: Distribution analysis
    6. Phase 2: Feature preprocessing (_apply_feature_preprocessing)
    7. Phase 3: Preselection (barra / hierarchical_clustering / ic_ir)
    8. Phase 4: X = X_preprocessed[selected_features]
    9. Drop NaN, target CS preprocessing
    10. Predict: features -> pool_filter -> preprocess -> fillna(0)
    11. Qlib adapter: merge -> dataset -> train -> predict
    12. Post-process (cs_standardize) -> blend

Intentional LightGBM-specific differences (NOT alignment bugs):
    - distribution_prefilter uses model_type="lightgbm" (event -> binning, not winsorize_zscore)
    - _create_interaction_features: optional product-term feature engineering
    - target_winsorize: optional extreme clipping before training
    - _prepare_lgb_data: date-sort + ranking label transform
    - sample_weight_ic_based: optional IC-weighted training samples
    - QlibLGBWrapper recomputes weights internally (aligned with QlibLinearWrapper)
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Qlib Adapter (required)
from factor_synthesis.qlib_adapter import DataConverter, QlibLGBWrapper

# Dynamic blend module
from .dynamic_blend import (
    DynamicBlendConfig,
    compute_prior_composite,
    compute_dynamic_blend_ratio,
)

# [ALIGNED] Import shared utilities from lasso_method to avoid duplication
from .lasso_method import (
    _strict_label_exit_index,
    _outer_prediction_index,
    _resolve_inner_train_valid_indices,
    _filter_by_missing_rate,
    InsufficientSurvivingFactorsError,
    _filter_by_pool_date_range as _lasso_filter_by_pool_date_range,
    _blend_with_original as _lasso_blend_with_original,
    _run_blend_only_mode,
)

QLIB_ADAPTER_AVAILABLE = True

RANKING_OBJECTIVES = {"lambdarank", "rank_xendcg"}


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


# =============================================================================
# Shared Utility Functions
# =============================================================================


def _create_interaction_features(X: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Create factor interaction features (product terms).

    Args:
        X: Original factor data
        config: Config dict containing feature_interaction flag

    Returns:
        DataFrame with interaction features added
    """
    if not config.get("feature_interaction", False):
        return X

    X = X.copy()
    cols = X.columns.tolist()
    for i, col1 in enumerate(cols):
        for col2 in cols[i + 1:]:
            interaction_col = f"{col1}_x_{col2}"
            X[interaction_col] = X[col1] * X[col2]

    return X


# [ALIGNED] Use _filter_by_pool_date_range from lasso_method.py (imported above)
_filter_by_pool_date_range = _lasso_filter_by_pool_date_range


def _winsorize_target(
    y: pd.Series, lower: float = 0.01, upper: float = 0.99
) -> pd.Series:
    """
    Winsorize target variable to clip extreme values.

    Args:
        y: Target variable (returns)
        lower: Lower quantile
        upper: Upper quantile

    Returns:
        Winsorized target variable
    """
    lower_bound = y.quantile(lower)
    upper_bound = y.quantile(upper)
    return y.clip(lower_bound, upper_bound)


def _compute_ic_based_weights(
    X: pd.DataFrame, y: pd.Series, logger
) -> Optional[np.ndarray]:
    """
    Compute sample weights based on factor IC.
    High IC dates get higher weights.

    Args:
        X: Factor data
        y: Target variable
        logger: Logger instance

    Returns:
        Sample weight array, or None for empty input; calculation errors propagate
    """
    if len(X) == 0:
        return None
    try:
        dates = X.index.get_level_values("date").unique()
        weights = np.ones(len(X))

        for date in dates:
            mask = X.index.get_level_values("date") == date
            if mask.sum() < 10:
                continue

            X_daily = X.loc[mask]
            y_daily = y.loc[mask]

            daily_ic = X_daily.corrwith(y_daily).mean()

            if not np.isnan(daily_ic):
                weights[mask] = 1 + max(0, daily_ic * 10)

        weights = weights / weights.mean()
        logger.info(
            "[IC加权] 权重范围: [%.3f, %.3f], 均值=%.3f",
            weights.min(),
            weights.max(),
            weights.mean(),
        )
        return weights
    except Exception as e:
        logger.exception("[IC加权] 计算失败: %s", e)
        raise


# [ALIGNED] Use _blend_with_original from lasso_method.py (cloud-aligned, verified correct)
# LASSO version uses simple groupby zscore + fillna(0) - no pool-aware logic needed
# because predictions.index already reflects the pool-filtered stock universe.
_blend_with_original = _lasso_blend_with_original


def _predict_lightgbm_in_sample(model: object, X: pd.DataFrame) -> pd.Series:
    """Predict on already-prepared LightGBM feature rows without rebuilding a Qlib dataset."""
    feature_names = getattr(model, "feature_names", None)
    X_input = X
    if feature_names:
        ordered_features = list(feature_names)
        missing = [name for name in ordered_features if name not in X.columns]
        if missing:
            raise ValueError(f"LightGBM model is missing input features: {missing}")
        X_input = X.loc[:, ordered_features]

    raw_model = None
    qlib_model = getattr(model, "qlib_model", None)
    if qlib_model is not None:
        raw_model = getattr(qlib_model, "model", None)
    if raw_model is None:
        raw_model = getattr(model, "model", None)
    if raw_model is None or not hasattr(raw_model, "predict"):
        raise ValueError("LightGBM raw model is not available for in-sample prediction")

    pred_values = np.asarray(raw_model.predict(X_input.values)).reshape(-1)
    if len(pred_values) != len(X_input):
        raise ValueError(
            f"LightGBM prediction length mismatch: {len(pred_values)} != {len(X_input)}"
        )
    return pd.Series(pred_values, index=X.index, name="lightgbm_train_prediction")


def _prepare_lgb_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
    """
    Prepare LightGBM ranking / regression data (sorting, grouping, label transform).
    
    Args:
        X: Factor data
        y: Target variable
        config: Configuration dict
        
    Returns:
        X_sorted, y_sorted, groups
    """
    data = X.copy()
    data["__target__"] = y
    # DuckDB does not guarantee row order without ORDER BY.  Sorting only the
    # date level leaves the within-date instrument order unstable, which makes
    # LightGBM bagging select different rows across otherwise identical runs.
    # The full MultiIndex sort is part of the experiment reproducibility
    # contract: identical point-in-time inputs must produce identical models.
    data = data.sort_index()

    X_sorted = data.drop(columns=["__target__"])
    y_sorted = data["__target__"]

    objective = config.get("objective", "regression")
    groups = None

    if objective in RANKING_OBJECTIVES:
        groups = X_sorted.groupby(level="date").size().values

        label_type = config.get("label_type", "raw")

        if label_type == "rank_pct":
            y_rank = y_sorted.groupby(level="date").rank(pct=True)
            y_rank = y_rank.fillna(0.0)
            y_sorted = (y_rank * 4.99).astype(int).clip(lower=0, upper=4)
        elif label_type == "binning":
            def _bin_labels(x: pd.Series) -> pd.Series:
                return pd.qcut(x, 5, labels=False, duplicates="drop")

            y_sorted = y_sorted.groupby(level="date").transform(_bin_labels)
        elif label_type == "topk_vs_rest":
            top_k = int(config.get("label_top_k", 4))

            def _topk_binary(x: pd.Series) -> pd.Series:
                if x.empty:
                    return x
                ranks = x.rank(ascending=False, method="first")
                return (ranks <= top_k).astype(int)

            y_sorted = y_sorted.groupby(level="date").transform(_topk_binary)

    return X_sorted, y_sorted, groups


def _run_lightgbm_with_qlib_adapter(
    synthesizer,
    X_final: pd.DataFrame,
    y_final: pd.Series,
    predict_factors: pd.DataFrame,
    train_start: str,
    train_end: str,
    config: dict,
    logger,
    sample_weight: Optional[np.ndarray] = None,
    y_raw: Optional[pd.Series] = None,
) -> Tuple[pd.Series, Dict]:
    """
    Run LightGBM training and prediction using Qlib adapter.
    
    [ALIGNED WITH LASSO] Matches _run_lasso_with_qlib_adapter structure:
    - Embargo gap between train and valid periods
    - Sample weight threading
    - Precomputed neutralized inputs are loaded before training when enabled
    
    Args:
        synthesizer: FactorSynthesizer instance
        X_final: Preprocessed training features (MultiIndex: date, stock_code)
        y_final: Preprocessed training labels
        predict_factors: Prediction period factor data
        train_start/end: Training period dates
        config: Synthesis configuration
        logger: Logger instance
        sample_weight: Optional sample weights
        
    Returns:
        composite_series: Prediction Series
        model_info: Model information dict
        model: Trained model object (for saving/in-sample prediction)
    """
    n_outer_samples = len(X_final)
    train_keys, valid_keys = _resolve_inner_train_valid_indices(
        X_final, config, logger, fraction_key="lgb_validation_fraction"
    )
    keep = X_final.index.isin(train_keys) | X_final.index.isin(valid_keys)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)[keep]
    X_final = X_final.loc[keep]
    y_final = y_final.reindex(X_final.index)
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
    all_factors = pd.concat([X_final, predict_factors], axis=0)
    all_factors = all_factors[~all_factors.index.duplicated(keep='first')]
    
    neutralization_cfg = config.get("neutralization", {})
    if neutralization_cfg.get("enabled", False):
        logger.info("[中性化] 使用加载阶段的预计算中性化输入；跳过训练阶段市值回归")
    
    # Labels (only training period has labels)
    all_labels = y_final.reindex(all_factors.index)
    
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
    
    # [ALIGNED WITH LASSO] Create and train model with sample weights
    model = QlibLGBWrapper(config, logger)
    model_info = model.fit(dataset, sample_weight=sample_weight)
    
    # Predict
    predictions = model.predict(dataset, segment="test")
    
    # Convert back to original format
    composite_series = DataConverter.convert_predictions_to_series(
        predictions, predict_factors.index
    )
    composite_series.name = "lightgbm_composite"
    
    # Add model info
    # [ALIGNED WITH LASSO] Include validation_start for bundle export consistency
    model_info.update({
        "train_start": train_start,
        "train_end": train_end,
        "n_train_samples": n_outer_samples,
        "label_isolation_version": "strict_label_exit_v2",
        "predict_start": test_start,
        "predict_end": test_end,
        "validation_start": test_start,  # [ALIGNED] Same key as LASSO for bundle index
        "qlib_adapter": True,
    })
    
    return composite_series, model_info, model


def run_lightgbm_synthesis(
    synthesizer,
    factor_df: pd.DataFrame,
    forward_returns: pd.Series,
    train_start: str,
    train_end: str,
    config: dict,
    output_dir: Optional[str] = None,
    daily_pools: Optional[Dict] = None,
) -> Tuple[pd.Series, Dict]:
    """
    LightGBM gradient-boosting based factor synthesis using Qlib adapter.
    
    Args:
        synthesizer: FactorSynthesizer instance
        factor_df: Factor data DataFrame
        forward_returns: Forward returns Series
        train_start: Training start date
        train_end: Training end date
        config: Synthesis configuration
        output_dir: Optional output directory for model saving
        daily_pools: Optional dynamic pool dict
        
    Returns:
        composite_series: Synthesized factor Series
        model_info: Model information dict
    """
    logger = synthesizer.logger
    DataConverter.convert_index_to_qlib_format(factor_df.index)
    DataConverter.convert_index_to_qlib_format(forward_returns.index)

    # [ALIGNED WITH LASSO] Training window with embargo using trading days
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

    # [DEBUG] 诊断训练窗口数据问题
    if train_factors.empty:
        all_dates = pd.to_datetime(dates.unique()).sort_values()
        logger.warning(
            "[DEBUG] train_factors 为空! "
            f"factor_df 日期范围: {all_dates.min().date()} ~ {all_dates.max().date()}, "
            f"请求范围: {train_start} ~ {train_end_adj.strftime('%Y-%m-%d')}"
        )
    else:
        train_dates_range = train_factors.index.get_level_values('date').unique()
        logger.info(
            f"[DEBUG] train_factors: {len(train_factors)} 行, "
            f"日期范围: {train_dates_range.min().date()} ~ {train_dates_range.max().date()}"
        )

    # [ALIGNED WITH LASSO] 仅使用训练因子列（排除仅用于混合的因子）
    training_factor_names = config.get('_training_factor_names')
    if training_factor_names:
        available_training_cols = [c for c in training_factor_names if c in train_factors.columns]
        excluded_cols = set(train_factors.columns) - set(available_training_cols)
        if excluded_cols:
            logger.info(
                "[LightGBM训练] 排除 %d 个仅用于混合的因子: %s",
                len(excluded_cols), list(excluded_cols)
            )
        train_factors = train_factors[available_training_cols]
        logger.info("[LightGBM训练] 使用 %d 个训练因子", len(available_training_cols))

    # Align target
    common_index = train_factors.index.intersection(forward_returns.index)
    if len(common_index):
        common_index = _strict_label_exit_index(common_index, config, prediction_boundary)
    X_raw = train_factors.loc[common_index]
    y = forward_returns.loc[common_index]

    if len(X_raw) == 0 or len(y) == 0:
        # [DEBUG] 详细诊断无有效数据的原因
        logger.warning("训练期 %s 到 %s 无有效数据", train_start, train_end)
        logger.warning(
            f"[DEBUG] train_factors: {len(train_factors)} 行, "
            f"forward_returns: {len(forward_returns)} 行, "
            f"common_index: {len(common_index)} 行"
        )
        return pd.Series(dtype=float), {}

    # [ALIGNED WITH LASSO] 按训练窗口过滤高缺失率因子
    data_quality_cfg = config.get('data_quality', {})
    max_missing_rate = data_quality_cfg.get('max_factor_missing_rate', 0.05)
    min_surviving_ratio = float(data_quality_cfg.get('min_surviving_factor_ratio', 0.0))
    configured_factor_count = len(training_factor_names or X_raw.columns)
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
        return pd.Series(dtype=float), {}

    X_train_data = X_raw

    # =========================================================================
    # Phase 1: 分布分析 + 预处理配置（在筛选之前！）
    # (Aligned with lasso_method.py pipeline)
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
        dist_report = analyzer.analyze(X_train_data)
        logger.info("[分布配置] distribution_config_version=sampling_controls_v2")
        
        # 获取 Config 中的显式覆盖配置
        config_preprocess = config.get('preprocess', {}).copy()
        config_overrides = {k: v for k, v in config_preprocess.items() if k != 'ALL'}
        
        # 构建最终预处理配置: 分布分析默认 + Config覆盖
        # [INTENTIONAL DIFF vs LASSO] model_type="lightgbm" makes event_driven factors
        # use "binning" instead of "winsorize_zscore" (trees handle discrete bins well)
        final_preprocess_cfg = analyzer.build_preprocess_config(
            dist_report.results,
            config_overrides,
            model_type=config.get("model_type", "lightgbm"),
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
        if not isinstance(final_preprocess_cfg, dict):
            final_preprocess_cfg = {}

    # =========================================================================
    # Phase 2: 统一预处理（在筛选之前！）
    # [ALIGNED WITH LASSO] Always preprocess if config exists (no standardize guard)
    # =========================================================================
    if isinstance(final_preprocess_cfg, dict) and final_preprocess_cfg:
        X_preprocessed = synthesizer._apply_feature_preprocessing(X_train_data, final_preprocess_cfg)
        logger.info(f"[预处理] 已对 {len(X_train_data.columns)} 个因子执行预处理（筛选前）")
    else:
        X_preprocessed = X_train_data.copy()

    # =========================================================================
    # Phase 3: 预筛选（接收预处理后的因子！）
    # (Ported from lasso_method.py for architecture alignment)
    # =========================================================================
    from factor_synthesis.preselection import get_preselection_strategy
    
    # [NEW] 预筛选方法互斥检查
    _validate_preselection_exclusivity(config)
    
    # 检查三种预筛选方法
    barra_cfg = config.get('barra_preselection', {})
    pre_sel_cfg = config.get('pre_selection', {})
    hierarchical_cfg = config.get('hierarchical_clustering', {})
    
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
        
        strategy = get_preselection_strategy('hierarchical_clustering', hierarchical_cfg, logger)
        selected_features = strategy.select(X_preprocessed, y)
        
        logger.info(f"[层次聚类] 筛选完成，保留 {len(selected_features)} 个代表因子")
        
    elif pre_sel_cfg.get('enabled', False):
        # 传统预筛选 (IC/IR/dCor)
        strategy_name = pre_sel_cfg.get('strategy', 'ic_ir')
        strategy = get_preselection_strategy(strategy_name, pre_sel_cfg, logger)
        # [CHANGED] 传入预处理后的因子 X_preprocessed
        selected_features = strategy.select(X_preprocessed, y)
    else:
        selected_features = list(X_preprocessed.columns)
    
    # [FIX] 当无特征被选中时，检查是否可以使用 blend_weights 进行纯混合
    blend_ratio = float(config.get("blend_with_original_ratio", 0.0))
    blend_weights = config.get("blend_weights", {})
    
    # [ALIGNED WITH LASSO] 当无特征被选中时，使用 blend_weights 进行纯混合
    if not selected_features:
        if blend_ratio > 0 and blend_weights:
            logger.warning("预筛选后无有效因子，将使用纯 blend_weights 模式 (blend_ratio=%.2f)", blend_ratio)
            return _run_blend_only_mode(
                synthesizer=synthesizer,
                factor_df=factor_df,
                train_end=train_end,
                config=config,
                daily_pools=daily_pools,
                logger=logger,
                preprocess_config=final_preprocess_cfg,
            )
        else:
            logger.warning("预筛选后无有效因子且无 blend_weights，跳过本窗口")
        return pd.Series(dtype=float), {}

    # =========================================================================
    # Phase 4: 筛选后的因子准备（无二次预处理！）
    # =========================================================================
    # [CHANGED] 使用已预处理的因子
    X_selected = X_preprocessed[selected_features]

    # [INTENTIONAL DIFF vs LASSO] Feature interaction: tree models benefit from
    # explicit product-term features. LASSO does not have this step.
    if config.get("feature_interaction", False):
        n_before = X_selected.shape[1]
        X_selected = _create_interaction_features(X_selected, config)
        logger.info(
            "[特征交互] 添加交互特征: %d -> %d 列", n_before, X_selected.shape[1]
        )

    # [REMOVED] Preprocessing now happens in Phase 2
    X = X_selected

    objective = config.get("objective", "regression")
    is_ranking = objective in RANKING_OBJECTIVES

    # [INTENTIONAL DIFF vs LASSO] Target winsorize: prevents gradient explosion
    # in GBDT. LASSO does not need this step (linear model is less sensitive).
    # Applied BEFORE Drop-NaN to use full cross-section for quantile computation.
    if config.get("target_winsorize", False):
        winsorize_lower = float(config.get("target_winsorize_lower", 0.01))
        winsorize_upper = float(config.get("target_winsorize_upper", 0.99))
        y = _winsorize_target(y, winsorize_lower, winsorize_upper)
        logger.info(
            "[Winsorize] 目标变量截断 [%.2f%%, %.2f%%]",
            winsorize_lower * 100,
            winsorize_upper * 100,
        )

    # [ALIGNED WITH LASSO] Target CS preprocessing BEFORE Drop-NaN
    # LASSO computes cross-sectional mean/std using ALL stocks per date
    # (including those with NaN features), then drops NaN.
    # Previously this was AFTER Drop-NaN, causing slightly different target
    # values due to different cross-sectional universes.
    y_raw: Optional[pd.Series] = None  # Raw returns for sample weighting
    if not is_ranking:
        try:
            from .preprocessing import apply_target_cs_preprocessing
            
            target_cs_demean = bool(config.get("target_cs_demean", True))
            target_cs_standardize = bool(config.get("target_cs_standardize", False))
            target_cs_rank = bool(config.get("target_cs_rank", False))
            eps = float(config.get("target_cs_eps", 1e-8))

            if target_cs_demean or target_cs_standardize or target_cs_rank:
                y_raw = y.copy()  # Save raw returns before preprocessing
                y = apply_target_cs_preprocessing(
                    y,
                    cs_demean=target_cs_demean,
                    cs_rank=target_cs_rank,
                    cs_standardize=target_cs_standardize,
                    eps=eps,
                    logger=logger,
                )
        except Exception as e:
            logger.exception("[目标处理] 按日截面处理失败: %s", e)
            raise

    # [ALIGNED WITH LASSO] Drop-NaN: drop rows where any feature or target is NaN
    # Moved AFTER target CS preprocessing to match LASSO pipeline order.
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
        return pd.Series(dtype=float), {}

    # [INTENTIONAL DIFF vs LASSO] _prepare_lgb_data: sorts data by date and
    # transforms labels for ranking objectives. LASSO skips this step.
    X_final, y_final, groups = _prepare_lgb_data(X, y, config)

    # Sample weights (using Qlib Processor)
    # NOTE: These weights are computed here but QlibLGBWrapper.fit() will
    # RECOMPUTE them internally to match the actual train segment after
    # train/valid split (aligned with QlibLinearWrapper behavior).
    from .preprocessing import compute_sample_weights
    
    half_life = float(config.get("sample_weight_half_life", 0.0))
    weight_decay = float(config.get("sample_weight_decay", 1.0))
    try:
        sample_weight_full = compute_sample_weights(
            X_final, weight_decay=weight_decay, half_life=half_life, logger=logger
        )
    except Exception as e:
        logger.exception("[样本权重] 构造样本权重失败: %s", e)
        raise

    # [INTENTIONAL DIFF vs LASSO] IC-based sample weights: experimental
    # feature unique to LightGBM that up-weights dates with higher factor IC.
    if config.get("sample_weight_ic_based", False):
        ic_weights = _compute_ic_based_weights(X_final, y_final, logger)
        if ic_weights is not None:
            if sample_weight_full is not None:
                sample_weight_full = sample_weight_full * ic_weights
                sample_weight_full = sample_weight_full / sample_weight_full.mean()
                logger.info("[样本权重] IC权重与时间权重组合")
            else:
                sample_weight_full = ic_weights

    # =========================================================================
    # Qlib Adapter - Main Training Path
    # =========================================================================
    logger.info("=" * 60)
    logger.info("[LightGBM] 使用 Qlib 适配器模式")
    logger.info("=" * 60)
    
    try:
        # Prepare prediction period data
        predict_start = pd.to_datetime(train_end) + timedelta(days=1)
        predict_end = predict_start + timedelta(
            days=synthesizer.config["rolling_window"]["predict_window_days"]
        )
        
        # [FIX] Use factor_df directly - it's already T-1 aligned via shift(1) in process_raw_factor_df()
        # Previous code did an explicit T-1 lookup on already-shifted data, causing double T-1 shift (T-2 bug)
        # Now aligned with Lasso method behavior
        predict_mask = (factor_df.index.get_level_values("date") >= predict_start) & (
            factor_df.index.get_level_values("date") <= predict_end
        )
        predict_factors_raw = factor_df.loc[predict_mask]
        
        if predict_factors_raw.empty:
            logger.warning("[Qlib] 预测期无有效因子数据")
            return pd.Series(dtype=float), {}
        
        # [ALIGNED WITH LASSO] Select features FIRST, then apply pool filter
        # This matches lasso_method.py prediction pipeline order:
        #   1. Column selection (selected_features)
        #   2. Row filtering (_filter_by_pool_date_range)
        if selected_features:
            predict_factors_raw = predict_factors_raw[selected_features]
        
        if daily_pools and not predict_factors_raw.empty:
            valid_indices = _filter_by_pool_date_range(
                predict_factors_raw.index, daily_pools, logger
            )
            predict_factors_raw = predict_factors_raw.loc[valid_indices]
            
            if predict_factors_raw.empty:
                logger.warning("[Qlib] 动态池过滤后预测期无数据")
                return pd.Series(dtype=float), {}
        
        # =========================================================================
        # 预测期预处理：使用与训练期相同的预处理配置 final_preprocess_cfg
        # [ALIGNED WITH LASSO] Always preprocess if config exists (no standardize guard)
        # =========================================================================
        if isinstance(final_preprocess_cfg, dict) and final_preprocess_cfg:
            predict_factors_processed = synthesizer._apply_feature_preprocessing(
                predict_factors_raw, final_preprocess_cfg
            )
            logger.debug(f"[预测期预处理] 已对 {len(predict_factors_raw.columns)} 个因子执行预处理")
        else:
            predict_factors_processed = predict_factors_raw.copy()
        
        # [FIX] Feature interaction AFTER preprocessing to match training order
        # Training: Phase 2 preprocess -> Phase 4 select -> interaction
        # Prediction must follow: select -> preprocess -> interaction
        if config.get("feature_interaction", False):
            predict_factors_processed = _create_interaction_features(predict_factors_processed, config)
        
        # [ALIGNED WITH LASSO] Fill NaN with 0 (mean of z-scored data) to preserve all stocks
        # LASSO uses fillna(0.0) to ensure all pool stocks are present in predictions
        nan_mask = predict_factors_processed.isna().any(axis=1)
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            logger.info(
                "[Fill-NaN] 预测数据: %d 行含缺失值 (%.2f%%), 填充为 0 (均值) 以保留样本",
                nan_count, nan_count / len(predict_factors_processed) * 100
            )
            predict_factors = predict_factors_processed.fillna(0.0)
        else:
            predict_factors = predict_factors_processed
        
        if len(predict_factors) == 0:
            logger.warning("[Qlib] 预测期无有效数据")
            return pd.Series(dtype=float), {}
        
        # [ALIGNED WITH LASSO] Use Qlib adapter
        # Note: predict_factors already has the correct columns (selected_features + interaction)
        # Do NOT re-filter with selected_features here as that would drop interaction columns
        composite_series, model_info, model = _run_lightgbm_with_qlib_adapter(
            synthesizer=synthesizer,
            X_final=X_final,
            y_final=y_final,
            predict_factors=predict_factors,
            train_start=train_start,
            train_end=train_end,
            config=config,
            logger=logger,
            sample_weight=sample_weight_full,
            y_raw=y_raw,
        )
        
        # Save model if output_dir is provided
        if output_dir and model is not None:
            import os
            model_filename = f"lgb_model_{train_end}.txt"
            model_path = os.path.join(output_dir, model_filename)
            try:
                model.save_model(model_path)
                model_info["model_path"] = model_path
                model_info["model_file"] = os.path.basename(model_path)
                logger.info(f"[LightGBM] 模型已保存: {model_path}")
            except Exception as e:
                logger.exception(f"[LightGBM] 保存模型失败: {e}")
                raise
        
        # [ALIGNED WITH LASSO] Post-processing: cross-sectional standardization
        if config.get("postprocess_cs_standardize", False):
            try:
                composite_series = composite_series.groupby(level="date").transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
                logger.info("[后处理] 应用截面标准化")
            except Exception as e:
                logger.exception("[后处理] 截面标准化失败: %s", e)
                raise
        
        # [ALIGNED WITH LASSO] Blend with original factors AFTER standardization
        static_blend_ratio = float(config.get("blend_with_original_ratio", 0.0))
        blend_weights = config.get("blend_weights", {})
        dynamic_blend_cfg = config.get("dynamic_blend", {})
        
        # Initialize blend info for model_info
        blend_info = {
            "static_blend_ratio": static_blend_ratio,
            "dynamic_blend_enabled": False,
            "blend_ratio_used": static_blend_ratio,
        }
        
        if (static_blend_ratio > 0 or dynamic_blend_cfg.get("enabled", False)) and blend_weights:
            # [ALIGNED WITH LASSO] Use factor_df.loc[predict_mask] for blend factors
            factors_for_blend = factor_df.loc[predict_mask]
            
            # Compute dynamic blend ratio if enabled
            if dynamic_blend_cfg.get("enabled", False):
                try:
                    # [ALIGNED WITH LASSO] Create DynamicBlendConfig with all fields
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
                    if dyn_config.topk_ic_pct > 0:
                        blend_info["topk_ic_version"] = "configured_fraction_v2"
                    
                    # For LightGBM, we compute IC using training data (y_final)
                    # and model predictions on training features
                    if model is not None and hasattr(model, 'predict') and len(X_final) > 0:
                        try:
                            # Get in-sample predictions from LightGBM
                            ml_train_pred = _predict_lightgbm_in_sample(model, X_final)
                            
                            # [ALIGNED WITH LASSO] Use original factor_df for blend factors
                            train_factor_df = factor_df.loc[train_mask]
                            prior_composite_train = compute_prior_composite(
                                factor_df=train_factor_df,
                                blend_weights=blend_weights,
                                index=y_final.index,
                                logger=logger,
                            )
                            
                            # Compute dynamic blend ratio
                            blend_result = compute_dynamic_blend_ratio(
                                ml_predictions=ml_train_pred,
                                prior_composite=prior_composite_train,
                                forward_returns=y_final,
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
                        except Exception as e:
                            logger.exception(f"[DynamicBlend] Computation failed: {e}")
                            raise
                    else:
                        logger.warning("[DynamicBlend] Model not available, using static ratio")
                        blend_ratio = static_blend_ratio
                        
                except Exception as e:
                    logger.exception(f"[DynamicBlend] Failed: {e}")
                    raise
            else:
                blend_ratio = static_blend_ratio
            
            # Apply blending with the computed ratio
            if blend_ratio > 0:
                composite_series = _blend_with_original(
                    composite_series, factors_for_blend, blend_weights, blend_ratio, logger, daily_pools
                )
        
        # Add blend info to model_info
        model_info["blend_info"] = blend_info
        
        # [ALIGNED WITH LASSO] Add model info with preprocessing config for bundle export
        # These fields are read by run_factor_synthesis.py for LightGBM bundle/index export.
        # Previously missing, causing bundle to have wrong defaults (standardize=False, etc.)
        model_info["blend_with_original_ratio"] = blend_info["blend_ratio_used"]
        model_info["blend_weights"] = blend_weights
        model_info["preprocess_config"] = final_preprocess_cfg
        if dist_cfg.get('enabled', False):
            model_info["distribution_config_version"] = "sampling_controls_v2"
        model_info["standardize"] = config.get("standardize", True)
        model_info["standardization_method"] = final_preprocess_cfg
        model_info["postprocess_cs_standardize"] = config.get("postprocess_cs_standardize", False)
        model_info["training_window_audit"] = training_window_audit
        
        logger.info(
            "LightGBM训练完成 (%s), 最佳迭代: %s",
            config.get("objective", "regression"),
            model_info.get("best_iteration", 0),
        )
        
        return composite_series, model_info
        
    except InsufficientSurvivingFactorsError:
        raise
    except Exception as e:
        logger.error("LightGBM训练失败: %s", e)
        import traceback
        logger.debug(traceback.format_exc())
        raise
