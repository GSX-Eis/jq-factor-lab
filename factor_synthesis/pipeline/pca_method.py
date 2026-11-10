import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Core utilities
from factor_synthesis.core.feature_preprocessor import cross_sectional_standardize
from .lasso_method import _outer_prediction_index, _strict_label_exit_index


def run_pca_synthesis(
    synthesizer,
    factor_df: pd.DataFrame,
    forward_returns: pd.Series,
    train_start: str,
    train_end: str,
    config: dict,
) -> Tuple[List[pd.Series], Dict]:
    """
    PCA-based factor synthesis, extracted from the original
    `FactorSynthesizer.pca_synthesis` method.
    """
    logger = synthesizer.logger

    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)

    train_mask = (factor_df.index.get_level_values("date") >= train_start_dt) & (
        factor_df.index.get_level_values("date") <= train_end_dt
    )

    train_factors = factor_df.loc[train_mask]

    if len(train_factors) == 0:
        logger.warning("训练期 %s 到 %s 无有效数据", train_start, train_end)
        return [], {}

    predict_index = _outer_prediction_index(
        synthesizer, factor_df, train_end_dt, None, logger
    )
    if len(predict_index) == 0:
        logger.warning("预测期无数据")
        return [], {}
    boundary = predict_index.get_level_values("date").min()
    train_factors = train_factors.loc[
        _strict_label_exit_index(train_factors.index, config, boundary)
    ]

    # Optional: factor pre-selection using forward returns
    if not forward_returns.empty:
        common_index = train_factors.index.intersection(forward_returns.index)
        X_filter = train_factors.loc[common_index]
        y_filter = forward_returns.loc[common_index]

        # Pre-selection based on IC/IR/t-stat (using pluggable strategy)
        from factor_synthesis.preselection import get_preselection_strategy
        
        pre_sel_cfg = config.get('pre_selection', {})
        if pre_sel_cfg.get('enabled', False):
            strategy_name = pre_sel_cfg.get('strategy', 'ic_ir')
            strategy = get_preselection_strategy(strategy_name, pre_sel_cfg, logger)
            selected_features = strategy.select(X_filter, y_filter)
        else:
            selected_features = list(X_filter.columns)
            
        if not selected_features:
            logger.warning("预筛选后无有效因子，跳过本窗口")
            return [], {}

        train_factors = train_factors[selected_features]
    else:
        selected_features = list(train_factors.columns)

    # Per-factor preprocessing (zscore / rank_zscore / rank)
    # Support both "preprocessing" (new) and "preprocess" (legacy) config keys
    preprocess_cfg = config.get("preprocessing", config.get("preprocess", {}))
    if isinstance(preprocess_cfg, dict) and preprocess_cfg:
        train_factors = synthesizer._apply_feature_preprocessing(train_factors, preprocess_cfg)

    train_factors = train_factors.fillna(0.0)

    # Standardization
    standardization_method = config.get("standardization_method", "global")
    scaler: Optional[StandardScaler] = None

    if standardization_method == "cross_sectional":
        logger.info("使用截面标准化 (Cross-Sectional Standardization)")
        X_scaled = cross_sectional_standardize(train_factors).values
    elif config.get("standardize", True):
        logger.info("使用全局时间序列标准化 (Global Standardization)")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(train_factors)
    else:
        X_scaled = train_factors.values

    n_components = config.get("n_components", min(3, len(train_factors.columns)))
    pca = PCA(n_components=n_components, random_state=42)

    try:
        pca.fit(X_scaled)

        predict_factors = factor_df.loc[predict_index]

        if selected_features:
            predict_factors = predict_factors[selected_features]

        if isinstance(preprocess_cfg, dict) and preprocess_cfg:
            predict_factors = synthesizer._apply_feature_preprocessing(
                predict_factors, preprocess_cfg
            )

        predict_factors = predict_factors.fillna(0.0)

        if len(predict_factors) == 0:
            logger.warning("预测期无数据")
            return [], {}

        if standardization_method == "cross_sectional":
            X_predict_scaled = cross_sectional_standardize(
                predict_factors
            ).values
        elif scaler is not None:
            X_predict_scaled = scaler.transform(predict_factors)
        else:
            X_predict_scaled = predict_factors.values

        components = pca.transform(X_predict_scaled)

        component_series_list: List[pd.Series] = []
        for i in range(n_components):
            component_series = pd.Series(
                components[:, i],
                index=predict_factors.index,
                name=f"pca_component_{i+1}",
            )
            component_series_list.append(component_series)

        model_info: Dict = {
            "method": "pca",
            "label_isolation_version": "strict_label_exit_v2",
            "preprocessing_version": "shared_config_v2",
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "feature_names": train_factors.columns.tolist(),
            "train_start": train_start,
            "train_end": train_end,
            "n_train_samples": len(train_factors),
        }

        total_variance = float(sum(pca.explained_variance_ratio_))
        logger.info(
            "PCA降维完成，主成分数: %d，累计方差解释率: %.3f",
            n_components,
            total_variance,
        )
        return component_series_list, model_info

    except Exception as e:
        logger.error("PCA降维失败: %s", e)
        raise


