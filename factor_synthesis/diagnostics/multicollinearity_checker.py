"""
多重共线性检查器

用于检测LASSO选中因子的多重共线性问题。

核心功能：
1. VIF (Variance Inflation Factor) - 衡量单个因子的共线性程度
2. 条件数 (Condition Number) - 衡量协方差矩阵的整体共线性
3. 相关矩阵计算 - 可视化因子间相关性

Author: QuantationAI
Date: 2026-02-07
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def compute_vif_metrics(
    factor_df: pd.DataFrame,
    selected_factors: List[str],
    max_samples: int,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    计算选中因子的方差膨胀因子（VIF）
    
    VIF定义:
        VIF_i = 1 / (1 - R²_i)
    
    其中R²_i是将第i个因子对其他因子回归的决定系数。
    VIF越大表示该因子与其他因子的共线性越强。
    
    性能优化：
    - 如果样本数 > max_samples，使用分层采样（每个日期采样相同比例）
    - 使用sklearn LinearRegression加速回归计算
    
    Args:
        factor_df: 因子数据DataFrame (MultiIndex: date, stock_code)
        selected_factors: 选中的因子名称列表
        max_samples: VIF计算的最大样本数（防止OOM）
        logger: 日志记录器
        
    Returns:
        VIF字典 {factor: vif_value}
    """
    if len(selected_factors) < 2:
        logger.warning("选中因子少于2个，无法计算VIF")
        return {}
    
    # 提取选中因子数据
    try:
        X = factor_df[selected_factors].copy()
    except KeyError as e:
        logger.exception(f"因子数据缺少列: {e}")
        raise
    
    # 删除缺失值
    X_clean = X.dropna()
    n_samples = len(X_clean)
    
    if n_samples == 0:
        logger.warning("删除缺失值后无有效样本")
        return {}
    
    logger.debug(f"VIF计算: {n_samples}个样本, {len(selected_factors)}个因子")
    
    # 分层采样（如果样本过多）
    if n_samples > max_samples:
        logger.info(f"样本数({n_samples})超过阈值({max_samples})，使用分层采样")
        X_clean = _stratified_sample(X_clean, max_samples, logger)
        n_samples = len(X_clean)
        logger.info(f"采样后样本数: {n_samples}")
    
    # 计算VIF
    vif_dict = {}
    
    for i, target_factor in enumerate(selected_factors):
        # 其他因子作为自变量
        predictor_factors = [f for f in selected_factors if f != target_factor]
        
        if not predictor_factors:
            vif_dict[target_factor] = 1.0
            continue
        
        try:
            X_predictors = X_clean[predictor_factors].values
            y_target = X_clean[target_factor].values
            
            # 标准化（提升数值稳定性）
            X_mean = np.mean(X_predictors, axis=0)
            X_std = np.std(X_predictors, axis=0)
            X_std[X_std < 1e-10] = 1.0  # 避免除零
            X_norm = (X_predictors - X_mean) / X_std
            
            y_mean = np.mean(y_target)
            y_std = np.std(y_target)
            if y_std < 1e-10:
                logger.warning(f"因子{target_factor}方差接近零，跳过VIF计算")
                vif_dict[target_factor] = np.nan
                continue
            y_norm = (y_target - y_mean) / y_std
            
            # 线性回归
            model = LinearRegression(fit_intercept=True)
            model.fit(X_norm, y_norm)
            r_squared = model.score(X_norm, y_norm)
            
            # 计算VIF
            if r_squared >= 0.9999:
                # 完全共线性
                vif = np.inf
                logger.warning(f"因子{target_factor}完全共线（R²≈1），VIF=∞")
            else:
                vif = 1.0 / (1.0 - r_squared)
            
            vif_dict[target_factor] = float(vif)
            logger.debug(f"因子{target_factor}: R²={r_squared:.4f}, VIF={vif:.2f}")
            
        except Exception as e:
            logger.exception(f"计算因子{target_factor}的VIF时出错: {e}")
            raise
    
    # 统计信息
    valid_vifs = [v for v in vif_dict.values() if not np.isnan(v) and not np.isinf(v)]
    if valid_vifs:
        vif_mean = float(np.mean(valid_vifs))
        vif_max = float(np.max(valid_vifs))
        logger.info(f"VIF统计: 均值={vif_mean:.2f}, 最大值={vif_max:.2f}")
    
    return vif_dict


def compute_condition_number(
    factor_df: pd.DataFrame,
    selected_factors: List[str],
    max_samples: int,
    logger: logging.Logger
) -> float:
    """
    计算选中因子协方差矩阵的条件数
    
    条件数定义:
        κ(X'X) = λ_max / λ_min
    
    其中λ_max和λ_min是协方差矩阵的最大和最小特征值。
    条件数越大表示矩阵越病态，存在严重共线性。
    
    Args:
        factor_df: 因子数据DataFrame
        selected_factors: 选中的因子名称列表
        max_samples: 最大样本数（防止OOM）
        logger: 日志记录器
        
    Returns:
        条件数（float）
    """
    if len(selected_factors) < 2:
        logger.warning("选中因子少于2个，无法计算条件数")
        return np.nan
    
    # 提取选中因子数据
    try:
        X = factor_df[selected_factors].copy()
    except KeyError as e:
        logger.exception(f"因子数据缺少列: {e}")
        raise
    
    # 删除缺失值
    X_clean = X.dropna()
    n_samples = len(X_clean)
    
    if n_samples == 0:
        logger.warning("删除缺失值后无有效样本")
        return np.nan
    
    # 采样
    if n_samples > max_samples:
        logger.info(f"样本数({n_samples})超过阈值，使用分层采样")
        X_clean = _stratified_sample(X_clean, max_samples, logger)
        n_samples = len(X_clean)
    
    logger.debug(f"条件数计算: {n_samples}个样本, {len(selected_factors)}个因子")
    
    try:
        X_values = X_clean.values
        
        # 标准化
        X_mean = np.mean(X_values, axis=0)
        X_std = np.std(X_values, axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_norm = (X_values - X_mean) / X_std
        
        # 计算协方差矩阵 X'X
        cov_matrix = np.dot(X_norm.T, X_norm) / n_samples
        
        # A rank-deficient covariance matrix is singular even if filtering
        # zero eigenvalues would leave one positive value. Detect that before
        # deriving the ratio so duplicated factors cannot report cond=1.
        rank = np.linalg.matrix_rank(cov_matrix)
        if rank < cov_matrix.shape[0]:
            logger.warning(
                f"协方差矩阵秩亏（rank={rank}/{cov_matrix.shape[0]}），条件数=∞"
            )
            return np.inf

        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        lambda_max = float(np.max(eigenvalues))
        lambda_min = float(np.min(eigenvalues))
        
        if lambda_min < 1e-10:
            cond_number = np.inf
            logger.warning(f"最小特征值接近零({lambda_min:.2e})，矩阵奇异")
        else:
            cond_number = lambda_max / lambda_min
        
        logger.debug(f"特征值范围: [{lambda_min:.2e}, {lambda_max:.2e}]")
        logger.info(f"条件数: {cond_number:.2f}")
        
        return float(cond_number)
        
    except np.linalg.LinAlgError as e:
        logger.exception(f"特征值计算失败: {e}")
        raise
    except Exception as e:
        logger.exception(f"条件数计算异常: {e}")
        raise


def compute_factor_correlation_matrix(
    factor_df: pd.DataFrame,
    selected_factors: List[str],
    logger: logging.Logger
) -> pd.DataFrame:
    """
    计算选中因子的相关矩阵
    
    使用Spearman相关系数（对异常值更鲁棒）。
    
    Args:
        factor_df: 因子数据DataFrame
        selected_factors: 选中的因子名称列表
        logger: 日志记录器
        
    Returns:
        相关矩阵DataFrame
    """
    if len(selected_factors) < 2:
        logger.warning("选中因子少于2个，无法计算相关矩阵")
        return pd.DataFrame()
    
    try:
        X = factor_df[selected_factors].copy()
        X_clean = X.dropna()
        
        if len(X_clean) == 0:
            logger.warning("删除缺失值后无有效样本")
            return pd.DataFrame()
        
        # Spearman相关系数
        corr_matrix = X_clean.corr(method='spearman')
        
        # 统计高相关因子对
        high_corr_pairs = []
        for i in range(len(selected_factors)):
            for j in range(i + 1, len(selected_factors)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append((selected_factors[i], selected_factors[j], corr))
        
        if high_corr_pairs:
            logger.info(f"检测到{len(high_corr_pairs)}对高相关因子（|ρ|>0.8）")
            for f1, f2, corr in high_corr_pairs[:5]:
                logger.info(f"  {f1} <-> {f2}: ρ={corr:.3f}")
        
        return corr_matrix
        
    except Exception as e:
        logger.exception(f"相关矩阵计算失败: {e}")
        raise


def _stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    按日期分层采样
    
    确保每个日期采样相同比例的股票，保持时序分布。
    
    Args:
        df: MultiIndex DataFrame (date, stock_code)
        target_size: 目标样本数
        logger: 日志记录器
        
    Returns:
        采样后的DataFrame
    """
    if not isinstance(df.index, pd.MultiIndex):
        # 非MultiIndex，使用简单随机采样
        return df.sample(n=min(target_size, len(df)), random_state=42)
    
    # 按日期分组
    dates = df.index.get_level_values(0).unique()
    n_dates = len(dates)
    
    if n_dates == 0:
        return df
    
    # 每个日期的目标样本数
    samples_per_date = max(1, target_size // n_dates)
    
    sampled_dfs = []
    for date in dates:
        date_df = df.loc[date]
        n_available = len(date_df)
        n_sample = min(samples_per_date, n_available)
        
        if n_sample > 0:
            sampled = date_df.sample(n=n_sample, random_state=42)
            sampled_dfs.append(sampled)
    
    result = pd.concat(sampled_dfs)
    logger.debug(f"分层采样: {len(df)} -> {len(result)} 样本")
    
    return result
