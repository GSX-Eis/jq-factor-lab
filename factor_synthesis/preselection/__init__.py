# -*- coding: utf-8 -*-
"""
因子预筛选模块

提供可扩展的因子预筛选策略框架，支持通过配置文件动态选择策略。

Usage:
    >>> from factor_synthesis.preselection import get_preselection_strategy
    >>> 
    >>> config = {
    ...     "enabled": True,
    ...     "strategy": "ic_ir",
    ...     "metrics": {"min_abs_ic": 0.02}
    ... }
    >>> strategy = get_preselection_strategy("ic_ir", config)
    >>> selected_factors = strategy.select(X, y)

Available Strategies:
    - "ic_ir": 基于 Rank IC / IR / t-stat 阈值筛选
    - "mi": 基于 Mutual Information 筛选
    - "dcor": 基于 Distance Correlation 筛选 (固定阈值)
    - "dcor_cv": 基于 Distance Correlation + 时序交叉验证动态阈值筛选
    - "stability_selection": 基于 Stability Selection 的因子筛选 (Meinshausen & Bühlmann 2010)
    - "barra_grouped": Barra分组预筛选 (按风格组分别筛选后合并)
    - "volatility_bucket_consensus": 波动率分桶一致性预筛选
    
Adding New Strategies:
    1. 创建新文件 (如 mutual_info_strategy.py)
    2. 继承 PreSelectionStrategy 基类
    3. 使用 @register_strategy 装饰器注册
    4. 在此 __init__.py 中导入
"""

from .base import PreSelectionStrategy
from .registry import (
    register_strategy,
    get_preselection_strategy,
    list_strategies,
    get_strategy_class,
)

# 导入策略以触发注册
from .ic_ir_strategy import ICIRPreSelectionStrategy
from .mi_strategy import MIPreSelectionStrategy
from .dcor_strategy import DCorPreSelectionStrategy
from .dcor_cv_strategy import DCorCVPreSelectionStrategy
from .stability_selection_strategy import StabilitySelectionPreStrategy
from .barra_grouped_strategy import BarraGroupedPreSelector, BarraGroupedPreSelectionStrategy
from .hierarchical_clustering_strategy import HierarchicalClusteringStrategy
from .volatility_regime_strategy import VolatilityBucketConsensusPreSelectionStrategy
from .gaussian_knockoff_strategy import GaussianKnockoffPreSelectionStrategy
from .time_series_knockoff_strategy import TimeSeriesKnockoffPreSelectionStrategy
from .portfolio_boundary_strategy import PortfolioBoundaryReplacementStrategy

__all__ = [
    # 基类
    'PreSelectionStrategy',
    # 注册表函数
    'register_strategy',
    'get_preselection_strategy',
    'list_strategies',
    'get_strategy_class',
    # 具体策略
    'ICIRPreSelectionStrategy',
    'MIPreSelectionStrategy',
    'DCorPreSelectionStrategy',
    'DCorCVPreSelectionStrategy',
    'StabilitySelectionPreStrategy',
    'BarraGroupedPreSelector',
    'BarraGroupedPreSelectionStrategy',
    'HierarchicalClusteringStrategy',
    'VolatilityBucketConsensusPreSelectionStrategy',
    'GaussianKnockoffPreSelectionStrategy',
    'TimeSeriesKnockoffPreSelectionStrategy',
    'PortfolioBoundaryReplacementStrategy',
]
