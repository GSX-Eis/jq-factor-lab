# -*- coding: utf-8 -*-
"""
Qlib适配器模块

提供将现有DuckDB数据格式转换为Qlib DatasetH格式的工具，
以及封装Qlib模型的包装器类。

**核心组件 - Qlib Processors (默认启用)**:
所有预处理逻辑现在使用 Qlib-style Processors 实现，无需额外配置:
- PerFactorPreprocessor: zscore, rank, rank_zscore, gauss_rank
- TargetCSProcessor: cs_demean, cs_rank, cs_standardize
- SampleWeightProcessor: 时间衰减样本权重

使用方式:
    >>> from factor_synthesis.qlib_adapter import PerFactorPreprocessor, TargetCSProcessor
    >>> 
    >>> # 特征预处理
    >>> processor = PerFactorPreprocessor({"ALL": "gauss_rank"})
    >>> X_processed = processor(X)
    >>> 
    >>> # 目标预处理
    >>> target_processor = TargetCSProcessor(cs_demean=True, cs_rank=True)
    >>> y_processed = target_processor(y_df)

或使用 pipeline/preprocessing.py 中的便捷函数:
    >>> from factor_synthesis.pipeline import apply_target_cs_preprocessing
    >>> y = apply_target_cs_preprocessing(y, cs_demean=True, cs_rank=True)
"""

# 导出适配器组件
from .data_converter import DataConverter
from .qlib_lgb_model import QlibLGBWrapper
from .qlib_linear_model import QlibLinearWrapper
from .quantile_elasticnet import QuantileElasticNet, QuantileElasticNetCV

# 导出Qlib处理器组件
from .processors import (
    PerFactorPreprocessor,
    TargetCSProcessor,
    SampleWeightProcessor,
)

# 导出自定义目标函数和CV指标
from .custom_objectives import QlibCustomObjectives
from .cv_metrics import CVMetricEvaluator

__all__ = [
    # Core components
    'DataConverter',
    'QlibLGBWrapper',
    'QlibLinearWrapper',
    # Quantile ElasticNet
    'QuantileElasticNet',
    'QuantileElasticNetCV',
    # Custom Qlib Processors
    'PerFactorPreprocessor',
    'TargetCSProcessor',
    'SampleWeightProcessor',
    # Custom objectives
    'QlibCustomObjectives',
    # CV metrics
    'CVMetricEvaluator',
]


