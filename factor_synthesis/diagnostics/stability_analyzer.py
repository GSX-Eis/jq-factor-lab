"""
稳定性分析器

用于分析LASSO因子选择在滚动窗口间的稳定性。

核心功能：
1. Jaccard相似度 - 衡量窗口间选中因子集合的重叠度
2. 权重变异系数 - 衡量同一因子在不同窗口的权重波动
3. 零/非零切换检测 - 识别频繁进出模型的因子

Author: QuantationAI
Date: 2026-02-07
"""

import logging
from numbers import Integral
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np

STABILITY_SWITCH_VERSION = "window_count_v2"


def compute_jaccard_stability(
    weight_rows: List[pd.Series],
    logger: logging.Logger
) -> Tuple[float, List[float]]:
    """
    计算相邻滚动窗口之间的Jaccard相似度
    
    Jaccard相似度定义:
        J(A, B) = |A ∩ B| / |A ∪ B|
    
    其中A和B是相邻窗口中LASSO选中的因子集合（权重非零）
    
    Args:
        weight_rows: 每个窗口的LASSO权重Series列表
        logger: 日志记录器
        
    Returns:
        Tuple[平均Jaccard相似度, Jaccard相似度序列]
        
    Example:
        Window 1: [f1, f2, f3] (非零权重因子)
        Window 2: [f2, f3, f4]
        Jaccard = |{f2, f3}| / |{f1, f2, f3, f4}| = 2/4 = 0.5
    """
    if len(weight_rows) < 2:
        logger.warning("窗口数量少于2，无法计算Jaccard相似度")
        return np.nan, []
    
    jaccard_series = []
    selected_sets = []  # 存储每个窗口的选中因子集合
    
    # 提取每个窗口的选中因子集合
    for i, weights in enumerate(weight_rows):
        selected = set(weights[weights != 0].index)
        selected_sets.append(selected)
        logger.debug(f"窗口{i}: 选中{len(selected)}个因子")
    
    # 计算相邻窗口的Jaccard相似度
    for i in range(len(selected_sets) - 1):
        set_a = selected_sets[i]
        set_b = selected_sets[i + 1]
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        if union == 0:
            jaccard = 0.0
        else:
            jaccard = intersection / union
        
        jaccard_series.append(jaccard)
        logger.debug(f"窗口{i} vs {i+1}: Jaccard={jaccard:.3f} (交集={intersection}, 并集={union})")
    
    # 计算平均值
    jaccard_mean = float(np.mean(jaccard_series))
    
    # 统计信息
    jaccard_std = float(np.std(jaccard_series))
    jaccard_min = float(np.min(jaccard_series))
    jaccard_max = float(np.max(jaccard_series))
    
    logger.info(f"Jaccard相似度统计: 均值={jaccard_mean:.3f}, 标准差={jaccard_std:.3f}, 范围=[{jaccard_min:.3f}, {jaccard_max:.3f}]")
    
    # 找出相似度最低的窗口对（潜在问题区域）
    if jaccard_series:
        min_idx = int(np.argmin(jaccard_series))
        logger.info(f"最低相似度: 窗口{min_idx} vs {min_idx+1} = {jaccard_series[min_idx]:.3f}")
        
        # 输出该窗口对的详细信息
        set_a = selected_sets[min_idx]
        set_b = selected_sets[min_idx + 1]
        only_in_a = set_a - set_b
        only_in_b = set_b - set_a
        logger.info(f"  仅在窗口{min_idx}: {list(only_in_a)[:5]}")
        logger.info(f"  仅在窗口{min_idx+1}: {list(only_in_b)[:5]}")
    
    return jaccard_mean, jaccard_series


def compute_weight_variability(
    weight_rows: List[pd.Series],
    factor_names: List[str],
    logger: logging.Logger
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    计算每个因子的权重变异系数（CV）和零/非零切换次数
    
    变异系数定义:
        CV = std(weights) / mean(|weights|)
    
    仅对非零权重计算CV，避免大量零权重影响结果。
    
    Args:
        weight_rows: 每个窗口的LASSO权重Series列表
        factor_names: 训练因子名称列表
        logger: 日志记录器
        
    Returns:
        Tuple[权重CV字典 {factor: cv}, 零/非零切换次数字典 {factor: switches}]
    """
    # 构造权重矩阵: 行=窗口, 列=因子
    weight_matrix = pd.DataFrame(
        [weights.reindex(factor_names, fill_value=0.0) for weights in weight_rows],
        columns=factor_names
    )
    
    logger.debug(f"权重矩阵形状: {weight_matrix.shape}")
    
    weight_cv = {}
    switches = {}
    
    for factor in factor_names:
        weights = weight_matrix[factor].values
        
        # 计算零/非零切换次数
        is_nonzero = (weights != 0).astype(int)
        switch_count = int(np.sum(np.abs(np.diff(is_nonzero))))
        switches[factor] = switch_count
        
        # 提取非零权重
        nonzero_weights = weights[weights != 0]
        
        if len(nonzero_weights) == 0:
            # 该因子从未被选中
            weight_cv[factor] = np.nan
        elif len(nonzero_weights) == 1:
            # 只有一个非零权重，无法计算CV
            weight_cv[factor] = 0.0
        else:
            # 计算CV
            mean_abs_weight = np.mean(np.abs(nonzero_weights))
            std_weight = np.std(nonzero_weights)
            
            if mean_abs_weight < 1e-10:
                cv = np.nan
            else:
                cv = std_weight / mean_abs_weight
            
            weight_cv[factor] = float(cv)
    
    # 统计信息
    valid_cvs = [cv for cv in weight_cv.values() if not np.isnan(cv)]
    if valid_cvs:
        cv_mean = float(np.mean(valid_cvs))
        cv_median = float(np.median(valid_cvs))
        cv_max = float(np.max(valid_cvs))
        logger.info(f"权重CV统计: 均值={cv_mean:.3f}, 中位数={cv_median:.3f}, 最大值={cv_max:.3f}")
    
    # 切换次数统计
    switch_counts = list(switches.values())
    if switch_counts:
        switch_mean = float(np.mean(switch_counts))
        switch_max = int(np.max(switch_counts))
        logger.info(f"零/非零切换统计: 均值={switch_mean:.2f}, 最大值={switch_max}")
    
    return weight_cv, switches


def identify_unstable_factors(
    weight_cv: Dict[str, float],
    switches: Dict[str, int],
    cv_threshold: float,
    logger: logging.Logger,
    *,
    n_windows: int,
) -> List[str]:
    """
    识别不稳定因子（高CV或频繁切换）
    
    不稳定因子的判断标准：
    1. CV > cv_threshold
    2. 切换次数 > 窗口数的50% (频繁进出模型)
    
    Args:
        weight_cv: 权重变异系数字典
        switches: 零/非零切换次数字典
        cv_threshold: CV阈值
        logger: 日志记录器
        n_windows: 实际传入的权重窗口总数（不是相邻边界数或最大切换次数）
        
    Returns:
        不稳定因子名称列表
    """
    if isinstance(n_windows, bool) or not isinstance(n_windows, Integral) or n_windows < 0:
        raise ValueError("n_windows must be a non-negative integer")
    unstable = []
    
    # 基于CV识别
    for factor, cv in weight_cv.items():
        if not np.isnan(cv) and cv > cv_threshold:
            unstable.append(factor)
            logger.debug(f"高CV因子: {factor} (CV={cv:.3f})")
    
    # 基于切换次数识别（切换超过50%窗口数）
    if switches:
        switch_threshold = n_windows * 0.5
        
        for factor, switch_count in switches.items():
            if switch_count > switch_threshold and factor not in unstable:
                unstable.append(factor)
                logger.debug(f"频繁切换因子: {factor} (切换{switch_count}次)")
    
    if unstable:
        logger.info(f"识别到{len(unstable)}个不稳定因子")
    
    return unstable
