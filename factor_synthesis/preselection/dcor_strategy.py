# -*- coding: utf-8 -*-
"""
dCor (Distance Correlation) 预筛选策略

基于距离相关系数 (Distance Correlation) 筛选因子的策略。
ISIS-Lasso 迭代筛选法的第一步。
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import dcor
from joblib import Parallel, delayed

from .base import PreSelectionStrategy, get_global_cache
from .registry import register_strategy
from .dcor_cv_strategy import _compute_single_dcor


@register_strategy
class DCorPreSelectionStrategy(PreSelectionStrategy):
    """基于距离相关系数 (dCor) 的筛选策略
    
    配置示例:
    ```json
    {
        "pre_selection": {
            "enabled": true,
            "strategy": "dcor",
            "metrics": {
                "min_abs_dcor": 0.05,
                "max_correlation": 0.5
            },
            "fallback_top_n": 3
        }
    }
    ```
    
    筛选逻辑:
    1. 对齐后按整个窗口的有效样本计算每个因子与收益率的 dCor
    2. 将窗口 dCor 存入兼容字段 dcor_mean（不做逐日平均）
    3. 根据 max_correlation 进行贪婪相关性筛选 (优先去除冗余因子)：
       - 按 dCor 降序排列
       - 依次加入因子，若新因子与已选因子的最大相关性超过阈值，则丢弃
    4. 根据配置的阈值 (min_abs_dcor) 筛选
    5. 若筛选后无因子剩余，强制保留 dCor 最大的 Top N 因子
    """
    
    name = "dcor"
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        执行基于 dCor 的因子筛选
        
        Args:
            X: 因子数据 (MultiIndex: date, stock_code)
            y: 前瞻收益率 (MultiIndex: date, stock_code)
            
        Returns:
            保留的因子名称列表
        """
        self._validate_inputs(X, y)
        
        # 检查是否启用
        if not self.config.get('enabled', True):
            self.logger.info("[dCor策略] 未启用，返回所有因子")
            return list(X.columns)
        
        self.logger.info("执行因子预筛选 (dCor Strategy)...")
        
        # 计算性能指标
        perf_df = self._calculate_performance(X, y)
        perf_df.attrs['correlation_filter_version'] = 'input_priority_v2'
        perf_df.attrs['fallback_correlation_version'] = 'cap_first_v2'
        perf_df.attrs['threshold_order_version'] = 'input_priority_v2'
        self._performance_report = perf_df
        
        if perf_df.empty:
            self.logger.warning("无法计算因子 dCor 指标，跳过筛选，保留所有因子")
            return list(X.columns)
        
        # 计算当前训练期所有因子的 dCor 均值并日志记录 (用于监测整体信号强度)
        avg_dcor = perf_df['dcor_mean'].mean()
        self.logger.info(f"当前训练期所有因子 dCor 均值: {avg_dcor:.4f} (基于 {len(perf_df)} 个因子)")
        
        # 获取筛选阈值
        metrics = self.config.get('metrics', {})
        initial_count = len(X.columns)
        
        # 所有因子先按 dCor 降序排列
        sorted_factors_by_dcor = perf_df.sort_values('dcor_mean', ascending=False)['factor_name'].tolist()
        
        # Step 1: 相关性筛选 (Greedy Drop) - 先去除冗余因子
        selected_factors = sorted_factors_by_dcor
        if 'max_correlation' in metrics and len(sorted_factors_by_dcor) > 1:
            max_corr = metrics['max_correlation']
            self.logger.info(f"[步骤1] 执行相关性筛选 (阈值={max_corr})...")
            
            selected_factors = self._filter_by_correlation(X, sorted_factors_by_dcor, max_corr)
            after_corr_count = len(selected_factors)
            self.logger.info(f"[步骤1] 相关性筛选: {initial_count} -> {after_corr_count} 个因子")
        else:
            after_corr_count = initial_count
        
        # Step 2: dCor 阈值筛选
        if 'min_abs_dcor' in metrics:
            threshold = metrics['min_abs_dcor']
            self.logger.info(f"[步骤2] 执行 dCor 阈值筛选 (min_abs_dcor >= {threshold})...")
            
            # 从相关性筛选后的因子中，再用 dCor 阈值筛选
            perf_selected = perf_df[perf_df['factor_name'].isin(selected_factors)]
            mask = perf_selected['dcor_mean'] >= threshold
            qualified = set(perf_selected[mask]['factor_name'])
            selected_factors = [factor for factor in selected_factors if factor in qualified]
            
            final_count = len(selected_factors)
            self.logger.info(f"[步骤2] dCor 阈值筛选: {after_corr_count} -> {final_count} 个因子")
        else:
            final_count = after_corr_count
        
        self.logger.info(f"因子预筛选完成: {initial_count} -> {after_corr_count} (Corr) -> {final_count} (dCor) | 最终保留率: {final_count/initial_count:.1%}")
        
        # 安全保底
        if final_count == 0:
            fallback_n = self.config.get('fallback_top_n', 3)
            self.logger.warning(
                f"⚠️ 预筛选后无剩余因子！尝试保留 dCor 最大的 Top {fallback_n} 因子"
            )
            top_n = perf_df.nlargest(fallback_n, 'dcor_mean')['factor_name'].tolist()
            if 'max_correlation' in metrics and len(top_n) > 1:
                top_n = self._filter_by_correlation(X[top_n], top_n, metrics['max_correlation'])
            if len(top_n) < fallback_n:
                self.logger.warning(f"相关上限优先，保底缺额 {fallback_n - len(top_n)} 个因子")
            return top_n
        
        return selected_factors
    
    def _calculate_performance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """并行计算特征的 dCor 指标"""
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        if X_aligned.empty:
            return pd.DataFrame()
        
        # 获取并行配置
        parallel_cfg = self.config.get('parallel', {})
        parallel_enabled = parallel_cfg.get('enabled', True)
        n_jobs = parallel_cfg.get('n_jobs', -1)
        
        # 预提取数据为 numpy 数组
        y_values = y_aligned.values
        columns = list(X_aligned.columns)
        
        if parallel_enabled and len(columns) > 1:
            # 并行计算
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_compute_single_dcor)(col, X_aligned[col].values, y_values)
                for col in columns
            )
            performance_records = [r for r in results if r is not None]
        else:
            # 串行计算 (fallback)
            performance_records = []
            for col in columns:
                result = _compute_single_dcor(col, X_aligned[col].values, y_values)
                if result is not None:
                    performance_records.append(result)
        
        return pd.DataFrame(performance_records)

    def _filter_by_correlation(self, X: pd.DataFrame, sorted_factors: List[str], threshold: float) -> List[str]:
        """
        向量化贪婪相关性筛选
        
        使用 NumPy 向量化操作替代内层 Python 循环。
        Performance: Uses cached correlation matrix for repeated calls.
        """
        if not sorted_factors:
            return []
            
        try:
            valid_cols = [c for c in sorted_factors if c in X.columns]
            if len(valid_cols) < 2:
                return valid_cols
            
            # 计算相关性矩阵 (with caching)
            cache = get_global_cache()
            corr_matrix, cached_cols = cache.get_correlation_matrix(X, valid_cols)
            col_to_idx = {col: i for i, col in enumerate(cached_cols)}
            
            # 使用 NumPy 向量化的贪婪筛选
            kept_indices = []
            
            for factor in valid_cols:
                i = col_to_idx[factor]
                if not kept_indices:
                    kept_indices.append(i)
                    continue
                
                # 向量化: 检查当前因子与所有已保留因子的相关性
                kept_arr = np.array(kept_indices)
                max_corr_with_kept = corr_matrix[i, kept_arr].max()
                
                if max_corr_with_kept <= threshold:
                    kept_indices.append(i)
            
            return [cached_cols[i] for i in kept_indices]
            
        except LabelIsolationError:
            raise
        except Exception as e:
            self.logger.error(f"相关性筛选失败: {e}")
            raise
