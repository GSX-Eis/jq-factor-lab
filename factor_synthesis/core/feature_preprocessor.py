"""
Feature Preprocessor Module

Cross-sectional preprocessing utilities for factor data.
Extracted from FactorSynthesizer for modularity and reuse.

Supported preprocessing methods:
- cross_sectional_standardize: 原始 Z-Score 标准化 (无去极值，用于 raw_zscore)
- cross_sectional_zscore: 标准 Z-Score (去极值 + Z-Score，遵循量化标准流程)
- cross_sectional_binning: 等频分箱 (适用于事件驱动型因子)
- cross_sectional_winsorize_zscore: 缩尾后 Z-Score (适用于边界情况)

Author: QuantationAI
Date: 2026-01-12
Updated: 2026-02-04 - Added binning and winsorize_zscore
Updated: 2026-02-11 - Added cross_sectional_zscore (winsorize + zscore)
"""

from utils.strategy_kernel import preprocess_series
import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)




def cross_sectional_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """对索引为(date, stock_code)的特征做逐日横截面标准化 (Z-Score)。

    内部使用与 Qlib PerFactorPreprocessor 相同的逻辑。

    Args:
        df: 行索引为MultiIndex(date, stock_code)的DataFrame

    Returns:
        标准化后的DataFrame，NaN填充为0
    """
    try:
        if df.empty:
            return df

        # 使用与 PerFactorPreprocessor._zscore 相同的实现
        def zscore(x):
            m = x.mean()
            std = x.std()
            if std and std > 1e-8:
                return (x - m) / std
            return x - m

        standardized = df.groupby(level="date").transform(zscore)
        return standardized.fillna(0)
    except Exception as e:
        logger.error(f"横截面标准化失败: {e}")
        raise


def cross_sectional_binning(
    data: Union[pd.DataFrame, pd.Series],
    n_bins: int = 5,
    min_samples: int = 10,
) -> Union[pd.DataFrame, pd.Series]:
    """
    横截面等频分箱：将连续因子转换为序数编码 [0, 1]

    适用于事件驱动型因子（高偏度/峰度），保留尾部信息，避免 rank 破坏稀疏信号。

    分箱逻辑:
    1. 对每个日期的截面数据进行等频分箱 (pd.qcut)
    2. 将分箱结果归一化到 [0, 1] 范围
    3. 若样本数不足或分箱失败，回退到百分位 rank

    Args:
        data: 因子数据
              - DataFrame: 行索引为 MultiIndex(date, stock_code)
              - Series: 索引为 MultiIndex(date, stock_code)
        n_bins: 分箱数量，默认 5
        min_samples: 每日最少样本数，低于此值使用 rank 替代

    Returns:
        分箱后的数据，格式与输入相同，值域 [0, 1]
    """

    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Expected a DataFrame or Series")

    def _bin_group(x: pd.Series) -> pd.Series:
        """对单个截面分组进行分箱"""
        clean = x.dropna()
        n_valid = len(clean)

        # 样本数不足
        if n_valid < min_samples or n_valid < n_bins:
            return x.rank(pct=True)

        try:
            # 等频分箱
            binned = pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
            # 获取实际分箱数（可能因 duplicates='drop' 而减少）
            actual_bins = binned.max()
            if actual_bins is not None and actual_bins > 0:
                # 归一化到 [0, 1]
                return binned / actual_bins
            else:
                # 分箱失败，回退到 rank
                return x.rank(pct=True)
        except (ValueError, TypeError):
            # qcut 失败（如所有值相同），回退到 rank
            return x.rank(pct=True)

    try:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return data
            # 对每列分别处理
            result = data.copy()
            for col in result.columns:
                result[col] = result[col].groupby(level="date").transform(_bin_group)
            return result
        elif isinstance(data, pd.Series):
            if data.empty:
                return data
            return data.groupby(level="date").transform(_bin_group)
        else:
            logger.warning(f"[分箱] 不支持的数据类型: {type(data)}")
            return data
    except Exception as e:
        logger.error(f"横截面分箱失败: {e}")
        raise


def cross_sectional_winsorize_zscore(
    data: Union[pd.DataFrame, pd.Series],
    percentile: float = 0.025,
    min_samples: int = 10,
) -> Union[pd.DataFrame, pd.Series]:
    """
    横截面 Winsorize + Z-Score：先缩尾再标准化

    适用于边界情况因子（中等偏度/峰度），在保留信息量的同时压制极端值影响。

    处理逻辑:
    1. 对每个日期的截面数据计算分位数边界
    2. 将超出边界的值截断 (winsorize)
    3. 对截断后的数据进行 Z-Score 标准化

    Args:
        data: 因子数据
              - DataFrame: 行索引为 MultiIndex(date, stock_code)
              - Series: 索引为 MultiIndex(date, stock_code)
        percentile: 缩尾百分位，默认 0.025（双侧各 2.5%）
        min_samples: 每日最少样本数，低于此值返回去均值后的结果

    Returns:
        缩尾标准化后的数据，格式与输入相同
    """

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Expected a DataFrame or Series")

    def _winsorize_zscore(x):
        return preprocess_series(
            x, "winsorize_zscore", percentile, min_samples, zero_degenerate=False
        )

    try:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return data
            # 对每列分别处理
            result = data.copy()
            for col in result.columns:
                result[col] = (
                    result[col].groupby(level="date").transform(_winsorize_zscore)
                )
            return result
        elif isinstance(data, pd.Series):
            if data.empty:
                return data
            return data.groupby(level="date").transform(_winsorize_zscore)
        else:
            logger.warning(f"[缩尾标准化] 不支持的数据类型: {type(data)}")
            return data
    except Exception as e:
        logger.error(f"横截面缩尾标准化失败: {e}")
        raise


def cross_sectional_zscore(
    data: Union[pd.DataFrame, pd.Series],
    percentile: float = 0.025,
    min_samples: int = 10,
) -> Union[pd.DataFrame, pd.Series]:
    """
    标准 Z-Score 预处理：去极值 + Z-Score 标准化

    遵循标准量化因子预处理流程：
    1. Winsorize: 对每个日期的截面数据按分位数截断极端值
    2. Z-Score: 对截断后的数据进行 (x - mean) / std 标准化

    与 cross_sectional_winsorize_zscore 逻辑一致，作为配置中 "zscore" 的默认实现，
    确保所有使用 zscore 的因子都经过去极值处理。
    若需要不去极值的原始 Z-Score，请使用 "raw_zscore"（对应 cross_sectional_standardize）。

    Args:
        data: 因子数据
              - DataFrame: 行索引为 MultiIndex(date, stock_code)
              - Series: 索引为 MultiIndex(date, stock_code)
        percentile: 缩尾百分位，默认 0.025（双侧各 2.5%）
        min_samples: 每日最少样本数，低于此值返回去均值后的结果

    Returns:
        去极值 + Z-Score 标准化后的数据，格式与输入相同
    """

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Expected a DataFrame or Series")

    def _zscore_with_winsorize(x):
        return preprocess_series(
            x, "zscore", percentile, min_samples, zero_degenerate=False
        )

    try:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return data
            result = data.copy()
            for col in result.columns:
                result[col] = (
                    result[col].groupby(level="date").transform(_zscore_with_winsorize)
                )
            return result
        elif isinstance(data, pd.Series):
            if data.empty:
                return data
            return data.groupby(level="date").transform(_zscore_with_winsorize)
        else:
            logger.warning(f"[Z-Score] 不支持的数据类型: {type(data)}")
            return data
    except Exception as e:
        logger.error(f"横截面 Z-Score (去极值+标准化) 失败: {e}")
        raise


