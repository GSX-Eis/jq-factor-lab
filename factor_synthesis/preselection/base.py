# -*- coding: utf-8 -*-
"""
因子预筛选策略基类

提供抽象接口，所有预筛选策略必须继承此基类。

Performance Optimizations (2026-02-04):
- PreSelectionCache: Session-level caching for correlation matrices and dCor results
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import pandas as pd
import numpy as np
import logging


# =============================================================================
# Preselection Cache (Performance Optimization)
# =============================================================================

class PreSelectionCache:
    """
    预筛选会话级缓存
    
    缓存相关性矩阵和 dCor 结果，避免重复计算。
    一个 rolling window 训练期间内应共享同一个缓存实例。
    
    Usage:
        >>> cache = PreSelectionCache()
        >>> 
        >>> # Get or compute correlation matrix
        >>> corr_matrix = cache.get_correlation_matrix(X, cols=['f1', 'f2', 'f3'])
        >>> 
        >>> # Get or compute dCor value
        >>> dcor_val = cache.get_dcor(X['demo_signal1'], y, compute_fn=_compute_dcor)
        >>> 
        >>> # Clear cache at end of session
        >>> cache.clear()
    """
    
    def __init__(self, max_corr_cache: int = 16, max_dcor_cache: int = 1024):
        """
        初始化缓存
        
        Args:
            max_corr_cache: 最大相关性矩阵缓存数量
            max_dcor_cache: 最大 dCor 结果缓存数量
        """
        self._corr_cache: Dict[str, np.ndarray] = {}
        self._corr_cols_cache: Dict[str, List[str]] = {}  # Store column order
        self._dcor_cache: Dict[str, float] = {}
        self._max_corr_cache = max_corr_cache
        self._max_dcor_cache = max_dcor_cache
        self._stats = {'corr_hits': 0, 'corr_misses': 0, 'dcor_hits': 0, 'dcor_misses': 0}
    
    def _hash_cols(self, cols: List[str]) -> str:
        """生成列名列表的哈希键"""
        # Sort to ensure consistent ordering
        sorted_cols = tuple(sorted(cols))
        return hashlib.md5(str(sorted_cols).encode()).hexdigest()[:16]
    
    def _hash_series(self, s: pd.Series) -> str:
        """Hash every value and index entry, including dtype/name metadata."""
        metadata = repr((s.name, str(s.dtype), s.index.names)).encode()
        values = pd.util.hash_pandas_object(s, index=True).values.tobytes()
        return hashlib.sha256(metadata + values).hexdigest()
    
    def get_correlation_matrix(
        self, 
        X: pd.DataFrame, 
        cols: Optional[List[str]] = None,
        method: str = 'pearson'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        获取或计算相关性矩阵（带缓存）
        
        Args:
            X: 因子数据 DataFrame
            cols: 要计算相关性的列，None 表示所有列
            method: 相关性计算方法 ('pearson' or 'spearman')
            
        Returns:
            (correlation_matrix, column_list) 元组
            - correlation_matrix: np.ndarray, 绝对值相关性矩阵
            - column_list: List[str], 矩阵对应的列顺序
        """
        if cols is None:
            cols = list(X.columns)
        
        valid_cols = sorted([c for c in cols if c in X.columns])
        frame = X[valid_cols]
        metadata = repr((tuple(valid_cols), tuple(map(str, frame.dtypes)), frame.index.names, method)).encode()
        values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        cache_key = 'corr_v2_' + self._hash_cols(cols) + '_' + hashlib.sha256(metadata + values).hexdigest()
        
        if cache_key in self._corr_cache:
            self._stats['corr_hits'] += 1
            return self._corr_cache[cache_key], self._corr_cols_cache[cache_key]
        
        self._stats['corr_misses'] += 1
        
        # Use a stable column order for the cached matrix.
        if not valid_cols:
            return np.array([[]]), []
        
        corr_matrix = frame.corr(method=method).abs().values
        
        # [FIX] 四舍五入消除浮点误差
        corr_matrix = np.round(corr_matrix, decimals=8)
        
        # LRU-style eviction if cache is full
        if len(self._corr_cache) >= self._max_corr_cache:
            # Remove oldest entry
            oldest_key = next(iter(self._corr_cache))
            del self._corr_cache[oldest_key]
            del self._corr_cols_cache[oldest_key]
        
        self._corr_cache[cache_key] = corr_matrix
        self._corr_cols_cache[cache_key] = valid_cols
        
        return corr_matrix, valid_cols
    
    def get_dcor(
        self,
        x: pd.Series,
        y: pd.Series,
        compute_fn: Optional[callable] = None,
        raise_on_error: bool = False,
    ) -> Optional[float]:
        """
        获取或计算 dCor 值（带缓存）
        
        Args:
            x: 因子值 Series
            y: 收益率 Series
            compute_fn: dCor 计算函数，签名为 (x_array, y_array) -> float
                       如果为 None，则仅检查缓存
            raise_on_error: strict 调用方显式开启异常传播；样本不足仍返回 None
            
        Returns:
            dCor 值，如果无法计算则返回 None
        """
        x_key = self._hash_series(x)
        y_key = self._hash_series(y)
        cache_key = f"dcor_v2_{x_key}_{y_key}"
        
        if cache_key in self._dcor_cache:
            self._stats['dcor_hits'] += 1
            return self._dcor_cache[cache_key]
        
        if compute_fn is None:
            return None
        
        self._stats['dcor_misses'] += 1
        
        # Compute dCor
        try:
            x_clean = x.dropna()
            y_clean = y.dropna()
            common_idx = x_clean.index.intersection(y_clean.index)
            
            if len(common_idx) < 10:
                return None
            
            dcor_value = compute_fn(x_clean.loc[common_idx].values, y_clean.loc[common_idx].values)
            
            # LRU-style eviction if cache is full
            if len(self._dcor_cache) >= self._max_dcor_cache:
                oldest_key = next(iter(self._dcor_cache))
                del self._dcor_cache[oldest_key]
            
            self._dcor_cache[cache_key] = dcor_value
            return dcor_value
            
        except LabelIsolationError:
            raise
        except Exception:
            if raise_on_error:
                raise
            return None
    
    def clear(self) -> None:
        """清空所有缓存"""
        self._corr_cache.clear()
        self._corr_cols_cache.clear()
        self._dcor_cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return self._stats.copy()
    

# Global cache instance for sharing across strategies in one session
_global_cache: Optional[PreSelectionCache] = None


def get_global_cache() -> PreSelectionCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = PreSelectionCache()
    return _global_cache


def reset_global_cache() -> None:
    """重置全局缓存（在新的训练窗口开始时调用）"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = PreSelectionCache()


# =============================================================================
# Base Strategy Class
# =============================================================================

class PreSelectionStrategy(ABC):
    """因子预筛选策略基类
    
    所有预筛选策略必须:
    1. 继承此类
    2. 定义 `name` 类属性（用于配置匹配）
    3. 实现 `select` 方法
    
    Example:
        >>> @register_strategy
        ... class MyStrategy(PreSelectionStrategy):
        ...     name = "my_strategy"
        ...     def select(self, X, y):
        ...         return list(X.columns)[:5]
    """
    
    name: str = "base"  # 策略名称，子类必须覆盖
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        初始化策略
        
        Args:
            config: 策略配置字典，来自 synthesis_config['pre_selection']
            logger: 日志记录器，若为None则创建默认logger
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._performance_report: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        执行因子筛选
        
        Args:
            X: 因子数据 DataFrame
               - Index: MultiIndex (date, stock_code)
               - Columns: 因子名称
            y: 前瞻收益率 Series
               - Index: MultiIndex (date, stock_code)
               
        Returns:
            List[str]: 保留的因子名称列表
            
        Raises:
            ValueError: 如果输入数据格式不正确
        """
        pass
    
    def get_performance_report(self) -> pd.DataFrame:
        """
        获取筛选过程中计算的性能指标报告
        
        Returns:
            pd.DataFrame: 包含各因子性能指标的DataFrame，
                         如果未计算则返回空DataFrame
        """
        if self._performance_report is not None:
            return self._performance_report
        return pd.DataFrame()
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        验证输入数据格式
        
        Args:
            X: 因子数据
            y: 前瞻收益率
            
        Raises:
            ValueError: 如果数据格式不正确
        """
        if X.empty:
            raise ValueError("因子数据 X 为空")
        if y.empty:
            raise ValueError("前瞻收益率 y 为空")
        
        # 检查MultiIndex
        if not isinstance(X.index, pd.MultiIndex):
            self.logger.warning("因子数据 X 不是 MultiIndex，可能影响筛选结果")
        if not isinstance(y.index, pd.MultiIndex):
            self.logger.warning("前瞻收益率 y 不是 MultiIndex，可能影响筛选结果")
