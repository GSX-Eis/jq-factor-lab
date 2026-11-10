"""
因子权重策略模块

提供多种持仓权重计算方法，替代固定 top_k 等权选股。

用法:
    from factor_synthesis.weighting import get_weight_strategy, ScoreWeightStrategy
    
    # 使用工厂函数
    strategy = get_weight_strategy('score', {
        'top_pct': 0.2,
        'max_weight': 0.3
    })
    weights = strategy.calculate_weights(scores)
    
    # 或直接实例化
    strategy = ScoreWeightStrategy(top_pct=0.2)
    weights = strategy.calculate_weights(scores)

可用策略:
    - equal: 等权 (固定 top_k)
    - score: 分数加权 (无边界，平滑过渡)
    - rank: 排名加权 (对极端值鲁棒)
    - softmax: Softmax 加权 (可调温度)
"""

from .weight_strategy import (
    WeightStrategy,
    EqualWeightStrategy,
    ScoreWeightStrategy,
    RankWeightStrategy,
    SoftmaxWeightStrategy,
    get_weight_strategy,
    list_strategies,
    WEIGHT_STRATEGIES,
)

__all__ = [
    'WeightStrategy',
    'EqualWeightStrategy',
    'ScoreWeightStrategy',
    'RankWeightStrategy',
    'SoftmaxWeightStrategy',
    'get_weight_strategy',
    'list_strategies',
    'WEIGHT_STRATEGIES',
]
