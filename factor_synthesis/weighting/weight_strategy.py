"""
因子权重策略模块

支持多种持仓权重计算方法：
- equal: 等权 (现有方法)
- score: 分数加权
- rank: 排名加权
- volatility_inverse: 波动率倒数加权 (TODO)

用法示例:
    from factor_synthesis.weighting import get_weight_strategy
    
    strategy = get_weight_strategy('score', {
        'top_pct': 0.2,
        'max_weight': 0.3,
        'min_weight': 0.02
    })
    weights = strategy.calculate_weights(scores)
"""

from abc import ABC, abstractmethod
from decimal import Decimal, ROUND_FLOOR
import pandas as pd
import numpy as np
from typing import Dict, Optional, List


def _select_scores(scores: pd.Series, top_pct: float) -> pd.Series:
    if not np.isfinite(top_pct) or not 0 < top_pct <= 1:
        raise ValueError("top_pct must be in (0, 1]")
    if not scores.index.is_unique or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Scores must have unique stock keys and finite values")
    count = int((Decimal(len(scores)) * Decimal(str(top_pct))).to_integral_value(rounding=ROUND_FLOOR))
    return scores.sort_values(ascending=False, kind='stable').head(max(1, count))


def _bounded_weights(raw: pd.Series, min_weight: float, max_weight: float) -> pd.Series:
    """Reserve the minimum, then distribute remaining capital with a hard cap.

    Capped names leave the allocation pool. If all remaining scores are zero,
    distribute the residual equally. Infeasible bounds raise instead of silently
    overriding the requested portfolio limits.
    """
    n = len(raw)
    if not (np.isfinite(min_weight) and np.isfinite(max_weight)
            and 0 <= min_weight <= max_weight <= 1):
        raise ValueError("Weights require 0 <= min_weight <= max_weight <= 1")
    if n * min_weight > 1 + 1e-12 or n * max_weight < 1 - 1e-12:
        raise ValueError(f"Infeasible weight bounds for {n} selected stocks: "
                         f"min_weight={min_weight}, max_weight={max_weight}")
    values = raw.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Raw weights must be finite and nonnegative")
    if values.max() > 0:
        values = values / values.max()
    weights = np.full(n, min_weight, dtype=float)
    free = np.ones(n, dtype=bool)
    remaining = 1.0 - weights.sum()
    for _ in range(n + 1):
        if remaining <= 1e-12:
            break
        preferences = values[free]
        if preferences.sum() == 0:
            preferences = np.ones(free.sum())
        allocation = remaining * preferences / preferences.sum()
        capacity = max_weight - weights[free]
        weights[free] += np.minimum(allocation, capacity)
        free[free] = allocation < capacity
        remaining = 1.0 - weights.sum()
    if not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Could not allocate weights within requested bounds")
    result = pd.Series(weights, index=raw.index)
    return result[result > 0].sort_values(ascending=False)


class WeightStrategy(ABC):
    """权重策略基类"""
    
    @abstractmethod
    def calculate_weights(
        self,
        scores: pd.Series,
        **kwargs
    ) -> pd.Series:
        """
        计算持仓权重
        
        Args:
            scores: 因子分数，index 为股票代码
            **kwargs: 策略特定参数
            
        Returns:
            权重 Series，index 为股票代码，values 为权重（和为1）
        """
        pass
    
    def get_selected_stocks(self, scores: pd.Series, **kwargs) -> List[str]:
        """获取被选中的股票列表（权重 > 0 的股票）"""
        weights = self.calculate_weights(scores, **kwargs)
        return weights[weights > 0].index.tolist()


class EqualWeightStrategy(WeightStrategy):
    """
    等权策略 (现有方法)
    
    选取分数最高的 top_k 只股票，每只等权。
    """
    
    def __init__(self, top_k: int = 4, **kwargs):
        self.top_k = top_k
    
    def calculate_weights(self, scores: pd.Series, **kwargs) -> pd.Series:
        if scores.empty:
            return pd.Series(dtype=float)
        
        sorted_stocks = scores.sort_values(ascending=False).head(self.top_k)
        n = len(sorted_stocks)
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n, index=sorted_stocks.index)


class ScoreWeightStrategy(WeightStrategy):
    """
    分数加权策略
    
    特点：
    - 分数越高，权重越大
    - 无硬性 top_k 边界，权重平滑过渡
    - 通过 top_pct 控制有效持仓范围
    
    数学公式:
        w_i = max(0, s_i - threshold) / sum_j(max(0, s_j - threshold))
    
    其中 threshold 为排名前 top_pct 的最低分数
    """
    
    def __init__(
        self,
        top_pct: float = 0.2,          # 前 20% 股票有权重
        max_weight: float = 0.3,       # 单只股票最大权重 30%
        min_weight: float = 0.02,      # 单只股票最小权重 2%
        weight_power: float = 1.0,     # 权重幂次 (1=线性, 2=平方强化头部)
        **kwargs
    ):
        self.top_pct = top_pct
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.weight_power = weight_power
    
    def calculate_weights(self, scores: pd.Series, **kwargs) -> pd.Series:
        if scores.empty:
            return pd.Series(dtype=float)
        
        selected = _select_scores(scores, self.top_pct)
        if not np.isfinite(self.weight_power) or self.weight_power <= 0:
            raise ValueError("weight_power must be positive and finite")
        # Scale before subtraction so opposite finite float64 extremes cannot overflow.
        scale = selected.abs().max()
        normalized = selected / scale if scale > 0 else selected
        excess = (normalized - normalized.min()).clip(lower=0)
        if excess.max() > 0:
            excess = excess / excess.max()
        raw = excess ** self.weight_power
        return _bounded_weights(raw, self.min_weight, self.max_weight)


class RankWeightStrategy(WeightStrategy):
    """
    排名加权策略
    
    特点：
    - 基于排名百分位，而非原始分数
    - 对极端值更鲁棒
    - 权重分布更均匀
    
    数学公式:
        w_i = (1 / rank_i) / sum_j(1 / rank_j) for all i in top_pct
    """
    
    def __init__(
        self,
        top_pct: float = 0.2,
        max_weight: float = 0.3,
        min_weight: float = 0.02,
        **kwargs
    ):
        self.top_pct = top_pct
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def calculate_weights(self, scores: pd.Series, **kwargs) -> pd.Series:
        if scores.empty:
            return pd.Series(dtype=float)
        
        selected = _select_scores(scores, self.top_pct)
        raw = 1.0 / selected.rank(ascending=False)
        return _bounded_weights(raw, self.min_weight, self.max_weight)


class SoftmaxWeightStrategy(WeightStrategy):
    """
    Softmax 加权策略
    
    特点：
    - 使用 softmax 函数将分数转换为权重
    - temperature 参数控制集中度（越小越集中）
    
    数学公式:
        w_i = exp(s_i / T) / sum_j(exp(s_j / T))
    """
    
    def __init__(
        self,
        temperature: float = 1.0,      # 温度参数
        top_pct: float = 0.2,          # 只对前 top_pct 应用 softmax
        max_weight: float = 0.3,
        min_weight: float = 0.02,
        **kwargs
    ):
        self.temperature = temperature
        self.top_pct = top_pct
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def calculate_weights(self, scores: pd.Series, **kwargs) -> pd.Series:
        if scores.empty:
            return pd.Series(dtype=float)
        
        selected = _select_scores(scores, self.top_pct)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        raw = np.exp((selected - selected.max()) / self.temperature)
        return _bounded_weights(raw, self.min_weight, self.max_weight)


# 策略注册表
WEIGHT_STRATEGIES = {
    'equal': EqualWeightStrategy,
    'score': ScoreWeightStrategy,
    'score_weighted': ScoreWeightStrategy,  # Alias for config compatibility
    'rank': RankWeightStrategy,
    'softmax': SoftmaxWeightStrategy,
}


def get_weight_strategy(name: str, config: dict = None) -> WeightStrategy:
    """
    工厂函数：根据名称和配置创建权重策略
    
    Args:
        name: 策略名称 ('equal', 'score', 'rank', 'softmax')
        config: 策略配置参数字典
        
    Returns:
        WeightStrategy 实例
        
    Example:
        strategy = get_weight_strategy('score', {
            'top_pct': 0.2,
            'max_weight': 0.3
        })
    """
    if config is None:
        config = {}
    
    if name not in WEIGHT_STRATEGIES:
        available = list(WEIGHT_STRATEGIES.keys())
        raise ValueError(f"Unknown weight strategy: '{name}'. Available: {available}")
    
    return WEIGHT_STRATEGIES[name](**config)


def list_strategies() -> list:
    """列出所有可用的权重策略"""
    return list(WEIGHT_STRATEGIES.keys())
