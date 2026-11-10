# -*- coding: utf-8 -*-
"""
Mutual Information (MI) 预筛选策略

基于 Mutual Information (互信息) 筛选因子的策略。
互信息能捕捉非线性关系，对于挖掘复杂信号可能比线性相关系数更有效。
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .base import PreSelectionStrategy, get_global_cache
from .registry import register_strategy


@register_strategy
class MIPreSelectionStrategy(PreSelectionStrategy):
    """基于 Mutual Information (互信息) 的筛选策略
    
    配置示例:
    ```json
    {
        "pre_selection": {
            "enabled": true,
            "strategy": "mi",
            "metrics": {
                "min_mi": 0.03,
                "max_correlation": 0.5
            },
            "fallback_top_n": 10
        }
    }
    ```
    
    筛选逻辑:
    1. 计算每个因子与目标收益率的 Mutual Information Score。
    2. 根据 min_mi 阈值进行初步筛选。
    3. 根据 max_correlation 进行贪婪相关性去重。
    4. 若筛选后无因子剩余，在 MI Top N 候选内按相关上限保底，允许数量不足。
    """
    
    name = "mi"
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        执行基于 MI 的因子筛选
        """
        self._performance_report = None
        self._validate_inputs(X, y)
        
        if not self.config.get('enabled', True):
            return list(X.columns)
        
        self.logger.info("执行因子预筛选 (MI Strategy)...")
        
        # 对齐数据
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        if X_aligned.empty:
            return list(X.columns)
        
        # 处理缺失值 (MI 不允许 NaN)
        # 简单处理：全量 dropna
        combined = X_aligned.copy()
        combined['__target__'] = y_aligned
        combined = combined.dropna()
        
        if combined.empty:
            self.logger.warning("对齐并清除缺失值后无数据，跳过筛选")
            return list(X.columns)
            
        X_clean = combined.drop(columns=['__target__'])
        y_clean = combined['__target__']
        
        # 计算 MI Score
        # [OPT] 样本量大时 MI 计算较慢，可以采样
        n_samples = len(X_clean)
        if n_samples > 20000:
            self.logger.info(f"样本数 ({n_samples}) 较大，进行下采样以加速 MI 计算...")
            indices = np.random.default_rng(42).choice(n_samples, 20000, replace=False)
            X_clean = X_clean.iloc[indices]
            y_clean = y_clean.iloc[indices]
            
        try:
            mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            perf_df = pd.DataFrame({
                'factor_name': X_clean.columns,
                'mi_score': mi_scores
            })
            self._performance_report = perf_df
            perf_df.attrs['report_lifecycle_version'] = 'per_call_v2'
            perf_df.attrs['correlation_filter_version'] = 'input_priority_v2'
            if n_samples > 20000:
                perf_df.attrs['sampling_version'] = 'local_rng42_v2'
            
            # 筛选
            metrics = self.config.get('metrics', {})
            min_mi = metrics.get('min_mi', 0.0)
            
            selected_mask = perf_df['mi_score'] >= min_mi
            selected_factors = perf_df[selected_mask]['factor_name'].tolist()
            
            mid_count = len(selected_factors)
            
            # 相关性筛选
            if 'max_correlation' in metrics and mid_count > 1:
                max_corr = metrics['max_correlation']
                # 按 MI 降序排列
                sorted_factors = perf_df.loc[selected_mask].sort_values('mi_score', ascending=False)['factor_name'].tolist()
                selected_factors = self._filter_by_correlation(X_aligned[selected_factors], sorted_factors, max_corr)
                
            final_count = len(selected_factors)
            self.logger.info(f"MI 预筛选完成: {len(X.columns)} -> {mid_count} (MI) -> {final_count} (Corr)")
            
            if final_count == 0:
                fallback_n = self.config.get('fallback_top_n', 5)
                selected_factors = perf_df.nlargest(fallback_n, 'mi_score')['factor_name'].tolist()
                if 'max_correlation' in metrics:
                    selected_factors = self._filter_by_correlation(
                        X_aligned[selected_factors], selected_factors,
                        metrics['max_correlation'],
                    )
                perf_df.attrs['fallback_correlation_version'] = 'cap_first_v2'
                self.logger.warning(
                    "筛选后无剩余，MI Top %s候选按相关上限保底: %s，缺额=%s",
                    fallback_n, selected_factors, max(0, fallback_n - len(selected_factors)),
                )
                
            return selected_factors
            
        except LabelIsolationError:
            raise
        except Exception as e:
            self.logger.error(f"MI 计算失败: {e}")
            raise

    def _filter_by_correlation(self, X: pd.DataFrame, sorted_factors: List[str], threshold: float) -> List[str]:
        """贪婪相关性筛选 (with caching)"""
        if not sorted_factors:
            return []
        kept_factors = []
        try:
            valid_cols = [c for c in sorted_factors if c in X.columns]
            if len(valid_cols) < 2:
                return valid_cols
            
            # Use cached correlation matrix
            cache = get_global_cache()
            corr_matrix_arr, cached_cols = cache.get_correlation_matrix(X, valid_cols)
            col_to_idx = {col: i for i, col in enumerate(cached_cols)}
            
            for f in valid_cols:
                is_correlated = False
                f_idx = col_to_idx[f]
                for kept in kept_factors:
                    kept_idx = col_to_idx[kept]
                    if corr_matrix_arr[f_idx, kept_idx] > threshold:
                        is_correlated = True
                        break
                if not is_correlated:
                    kept_factors.append(f)
            return kept_factors
        except LabelIsolationError:
            raise
        except Exception:
            raise
