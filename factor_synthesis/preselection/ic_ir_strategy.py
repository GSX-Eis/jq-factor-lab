# -*- coding: utf-8 -*-
"""
IC/IR 预筛选策略

基于 Rank IC、IR、t-stat 阈值筛选因子的策略。
迁移自 run_factor_synthesis.py 中的 _calculate_feature_performance 和 _filter_training_features。
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from .base import PreSelectionStrategy, get_global_cache
from .registry import register_strategy


@register_strategy
class ICIRPreSelectionStrategy(PreSelectionStrategy):
    """基于 Rank IC / IR / t-stat 的硬阈值筛选策略
    
    配置示例:
    ```json
    {
        "pre_selection": {
            "enabled": true,
            "strategy": "ic_ir",
            "metrics": {
                "min_abs_ic": 0.02,
                "min_abs_ir": 0.1,
                "min_abs_t_stat": 2.0,
                "max_correlation": 0.7
            },
            "fallback_top_n": 3
        }
    }
    ```
    
    筛选逻辑:
    1. 计算每个因子的每日 Rank IC (Spearman相关系数)
    2. 聚合得到 IC Mean, IC Std, IR, t-stat
    3. 根据配置的阈值 (min_abs_ic, min_abs_ir, min_abs_t_stat) 筛选
    4. (新) 根据 max_correlation 进行贪婪筛选：
       - 按 Abs(IC) 降序排列
       - 依次加入因子，若新因子与已选因子的最大相关性超过阈值，则丢弃
    5. 若筛选后无因子剩余，强制保留 |IC| 最大的 Top N 因子
    """
    
    name = "ic_ir"
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        执行基于 IC/IR 的因子筛选
        
        Args:
            X: 因子数据 (MultiIndex: date, stock_code)
            y: 前瞻收益率 (MultiIndex: date, stock_code)
            
        Returns:
            保留的因子名称列表
        """
        self._validate_inputs(X, y)
        
        # 检查是否启用
        if not self.config.get('enabled', True):
            self.logger.info("[IC/IR策略] 未启用，返回所有因子")
            return list(X.columns)
        
        self.logger.info("执行因子预筛选 (IC/IR Strategy)...")
        
        # 计算性能指标
        perf_df = self._calculate_performance(X, y)
        perf_df.attrs['correlation_filter_version'] = 'input_priority_v2'
        perf_df.attrs['fallback_correlation_version'] = 'cap_first_v2'
        self._performance_report = perf_df
        
        if perf_df.empty:
            self.logger.warning("无法计算因子性能指标，跳过筛选，保留所有因子")
            return list(X.columns)
        
        # 获取筛选阈值
        metrics = self.config.get('metrics', {})
        initial_count = len(X.columns)
        
        # 构建筛选掩码（默认保留所有）
        mask = pd.Series(True, index=perf_df.index)
        
        # 逐项应用过滤条件
        if 'min_abs_ic' in metrics:
            threshold = metrics['min_abs_ic']
            mask &= (perf_df['abs_ic'] >= threshold)
            self.logger.debug(f"筛选条件 min_abs_ic >= {threshold}")
            
        if 'min_abs_ir' in metrics:
            threshold = metrics['min_abs_ir']
            mask &= (perf_df['abs_ir'] >= threshold)
            self.logger.debug(f"筛选条件 min_abs_ir >= {threshold}")
            
        if 'min_abs_t_stat' in metrics:
            threshold = metrics['min_abs_t_stat']
            mask &= (perf_df['abs_t_stat'] >= threshold)
            self.logger.debug(f"筛选条件 min_abs_t_stat >= {threshold}")
        
        selected_factors = perf_df[mask]['factor_name'].tolist()
        mid_count = len(selected_factors)
        
        # [NEW] 相关性筛选 (Greedy Drop)
        if 'max_correlation' in metrics and mid_count > 1:
            max_corr = metrics['max_correlation']
            self.logger.info(f"执行相关性筛选 (Threshold={max_corr})...")
            
            # 仅保留通过初步筛选的因子数据
            X_selected = X[selected_factors]
            
            # 需要按IC绝对值排序，优先保留IC高的
            # perf_df 已经有了，提取 selected_factors 对应的 abs_ic 并排序
            sorted_factors = perf_df.loc[mask].sort_values('abs_ic', ascending=False)['factor_name'].tolist()
            
            selected_factors = self._filter_by_correlation(X_selected, sorted_factors, max_corr)
            
        final_count = len(selected_factors)
        
        self.logger.info(f"因子预筛选完成: {initial_count} -> {mid_count} (IC/IR) -> {final_count} (Corr) | 最终保留率: {final_count/initial_count:.1%}")
        
        # 安全保底：若筛选后无因子，强制保留 Top N
        if final_count == 0:
            fallback_n = self.config.get('fallback_top_n', 3)
            self.logger.warning(
                f"⚠️ 预筛选后无剩余因子！尝试保留 Rank IC 绝对值最大的 Top {fallback_n} 因子"
            )
            top_n = perf_df.nlargest(fallback_n, 'abs_ic')['factor_name'].tolist()
            if 'max_correlation' in metrics and len(top_n) > 1:
                top_n = self._filter_by_correlation(X[top_n], top_n, metrics['max_correlation'])
            if len(top_n) < fallback_n:
                self.logger.warning(f"相关上限优先，保底缺额 {fallback_n - len(top_n)} 个因子")
            return top_n
        
        return selected_factors
    
    def _calculate_performance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        计算特征的性能指标 (IC, IR, t-stat)
        
        [OPT] 自动选择 GPU 或 CPU 实现
        
        Args:
            X: 因子数据 (MultiIndex: date, stock_code)
            y: 前瞻收益率 (MultiIndex: date, stock_code)
            
        Returns:
            包含性能指标的 DataFrame
        """
        # 确保索引对齐
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        if X_aligned.empty:
            return pd.DataFrame()
        
        # [OPT] 自动选择 GPU 或 CPU
        # GPU 适用于大数据集 (>5000 样本), 否则 CPU 更快
        n_samples = len(X_aligned)
        use_gpu = n_samples > 5000 and self._is_mps_available()
        
        if use_gpu:
            self.logger.info(f"[IC计算] 使用 MPS GPU 加速 (样本数: {n_samples})")
            return self._calculate_performance_gpu(X_aligned, y_aligned)
        else:
            self.logger.debug(f"[IC计算] 使用 CPU (样本数: {n_samples})")
            return self._calculate_performance_cpu(X_aligned, y_aligned)
    
    def _is_mps_available(self) -> bool:
        """检查 MPS 是否可用"""
        try:
            import torch
            return (
                hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_available() and
                torch.backends.mps.is_built()
            )
        except ImportError:
            return False
    
    def _calculate_performance_gpu(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        GPU 加速的 IC 计算
        
        使用 PyTorch MPS 批量计算每日 Spearman IC
        """
        if '__target__' in X.columns:
            raise ValueError("Reserved factor name: __target__")
        import torch
        
        device = torch.device('mps')
        factor_cols = list(X.columns)
        n_factors = len(factor_cols)
        
        # Rank only the common complete sample shared by all factors and target.
        complete = X.notna().all(axis=1) & y.notna()
        X, y = X.loc[complete], y.loc[complete]
        if X.empty:
            report = pd.DataFrame()
            report.attrs['ic_normalization'] = 'complete_case_spearman_v3'
            return report

        # 预计算 Rank (Spearman = Pearson on Ranks)
        X_ranked = X.groupby(level='date').rank(pct=True)
        y_ranked = y.groupby(level='date').rank(pct=True)
        
        # 按日期分组
        dates = X_ranked.index.get_level_values('date').unique().sort_values()
        n_days = len(dates)
        
        # 构建 3D 张量 (n_days, max_stocks, n_factors) 和 target
        max_stocks = X_ranked.groupby(level='date').size().max()
        
        X_tensor = torch.zeros((n_days, max_stocks, n_factors), dtype=torch.float32, device=device)
        y_tensor = torch.zeros((n_days, max_stocks), dtype=torch.float32, device=device)
        mask = torch.zeros((n_days, max_stocks), dtype=torch.bool, device=device)
        
        # 填充张量
        for i, date in enumerate(dates):
            x_day = X_ranked.xs(date, level='date').values
            y_day = y_ranked.xs(date, level='date').values
            n_stocks = len(x_day)
            
            # 处理 NaN
            valid_mask = ~(np.isnan(x_day).any(axis=1) | np.isnan(y_day))
            x_day_clean = np.nan_to_num(x_day, nan=0.0)
            y_day_clean = np.nan_to_num(y_day, nan=0.0)
            
            X_tensor[i, :n_stocks, :] = torch.tensor(x_day_clean, dtype=torch.float32, device=device)
            y_tensor[i, :n_stocks] = torch.tensor(y_day_clean.flatten(), dtype=torch.float32, device=device)
            mask[i, :n_stocks] = torch.tensor(valid_mask, dtype=torch.bool, device=device)

        # 批量计算 Pearson (on ranks = Spearman)
        # 使用 masked 操作
        mask_float = mask.float().unsqueeze(-1)  # (n_days, max_stocks, 1)
        n_valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (n_days, 1)
        
        # Masked mean
        X_mean = (X_tensor * mask_float).sum(dim=1, keepdim=True) / n_valid.unsqueeze(-1)  # (n_days, 1, n_factors)
        y_mean = (y_tensor * mask.float()).sum(dim=1, keepdim=True) / n_valid  # (n_days, 1)
        
        # Centered values
        X_centered = (X_tensor - X_mean) * mask_float  # (n_days, max_stocks, n_factors)
        y_centered = (y_tensor - y_mean) * mask.float()  # (n_days, max_stocks)
        
        # Covariance: sum(X_centered * y_centered) / n
        cov = (X_centered * y_centered.unsqueeze(-1)).sum(dim=1) / n_valid  # (n_days, n_factors)
        
        # Std
        X_std = torch.sqrt((X_centered ** 2).sum(dim=1) / n_valid)  # (n_days, n_factors)
        y_std = torch.sqrt((y_centered ** 2).sum(dim=1, keepdim=True) / n_valid)  # (n_days, 1)
        
        # Correlation
        daily_ic = cov / (X_std.clamp(min=1e-8) * y_std.clamp(min=1e-8))
        valid_day = (n_valid >= 3) & (X_std > 1e-10) & (y_std > 1e-10)
        daily_ic = daily_ic.masked_fill(~valid_day, float('nan'))
        
        # 转回 CPU numpy
        daily_ic_np = daily_ic.cpu().numpy()
        
        # 聚合统计
        performance_records = []
        for j, col in enumerate(factor_cols):
            ic_series = daily_ic_np[:, j]
            valid_ics = ic_series[~np.isnan(ic_series)]
            
            if len(valid_ics) == 0:
                continue
            
            mean_ic = float(valid_ics.mean())
            std_ic = float(valid_ics.std(ddof=1)) if len(valid_ics) > 1 else float('nan')
            n_days_valid = len(valid_ics)
            
            ir = mean_ic / std_ic if std_ic > 1e-6 else 0.0
            t_stat = mean_ic / (std_ic / np.sqrt(n_days_valid)) if std_ic > 1e-6 and n_days_valid > 0 else 0.0
            
            performance_records.append({
                'factor_name': col,
                'ic_mean': mean_ic,
                'ic_std': std_ic,
                'abs_ic': abs(mean_ic),
                'ir': ir,
                'abs_ir': abs(ir),
                't_stat': t_stat,
                'abs_t_stat': abs(t_stat),
                'n_days': n_days_valid
            })
        
        report = pd.DataFrame(performance_records)
        report.attrs['ic_normalization'] = 'complete_case_spearman_v3'
        return report
    
    def _calculate_performance_cpu(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        CPU 版本的 IC 计算 (向量化实现)
        """
        if '__target__' in X.columns:
            raise ValueError("Reserved factor name: __target__")
        # Rank only the common complete sample shared by all factors and target.
        complete = X.notna().all(axis=1) & y.notna()
        X, y = X.loc[complete], y.loc[complete]
        if X.empty:
            report = pd.DataFrame()
            report.attrs['ic_normalization'] = 'complete_case_spearman_v3'
            return report

        # [OPT] 一次性预计算所有因子和目标的 Rank (Spearman = Pearson on Ranks)
        X_ranked = X.groupby(level='date').rank(pct=True)
        y_ranked = y.groupby(level='date').rank(pct=True)
        
        # 合并用于批量计算
        combined = X_ranked.copy()
        combined['__target__'] = y_ranked
        
        # 删除包含 NaN 的行
        combined = combined.dropna()
        
        if combined.empty:
            return pd.DataFrame()
        
        factor_cols = [c for c in combined.columns if c != '__target__']
        
        # [OPT] 向量化按日计算 Pearson (on ranks = Spearman)
        def _batch_corr(group: pd.DataFrame) -> pd.Series:
            """批量计算一天内所有因子与目标的相关系数"""
            if len(group) < 3:
                return pd.Series(np.nan, index=factor_cols)
            
            target = group['__target__']
            target_std = target.std()
            if target_std < 1e-10:
                return pd.Series(np.nan, index=factor_cols)
            
            target_centered = target - target.mean()
            
            result = {}
            for col in factor_cols:
                factor = group[col]
                factor_std = factor.std()
                if factor_std < 1e-10:
                    result[col] = np.nan
                else:
                    factor_centered = factor - factor.mean()
                    # pandas std() uses ddof=1; covariance must use the same n-1.
                    corr = (factor_centered * target_centered).sum() / ((len(group) - 1) * factor_std * target_std)
                    result[col] = corr
            
            return pd.Series(result)
        
        # 计算每日 IC 矩阵 (n_days x n_factors)
        daily_ic_df = combined.groupby(level='date').apply(_batch_corr)
        
        # 聚合统计
        performance_records = []
        for col in factor_cols:
            daily_ic = daily_ic_df[col].dropna()
            if daily_ic.empty:
                continue
                
            mean_ic = float(daily_ic.mean())
            std_ic = float(daily_ic.std())
            n_days = len(daily_ic)
            
            ir = mean_ic / std_ic if std_ic > 1e-6 else 0.0
            t_stat = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic > 1e-6 and n_days > 0 else 0.0
            
            performance_records.append({
                'factor_name': col,
                'ic_mean': mean_ic,
                'ic_std': std_ic,
                'abs_ic': abs(mean_ic),
                'ir': ir,
                'abs_ir': abs(ir),
                't_stat': t_stat,
                'abs_t_stat': abs(t_stat),
                'n_days': n_days
            })
        
        report = pd.DataFrame(performance_records)
        report.attrs['ic_normalization'] = 'complete_case_spearman_v3'
        return report
    
    def _filter_by_correlation(self, X: pd.DataFrame, sorted_factors: List[str], threshold: float) -> List[str]:
        """
        贪婪相关性筛选：
        1. 接受已按 IC 排序的因子列表
        2. 依次检查新因子与已保留因子的相关性
        3. 若 max(corr) < threshold，则保留
        
        Args:
            X: 因子数据 (包含 sorted_factors 列)
            sorted_factors: 按优先级排序的因子名列表
            threshold: 相关性阈值 (0~1)
            
        Returns:
            筛选后的因子名列表
        """
        if not sorted_factors:
            return []
            
        kept_factors = []
        
        # 只计算相关性矩阵一次 (Spearman or Pearson? 通常因子值本身的相关性用Pearson即可，RankIC用Spearman)
        # 这里使用 Pearson 对因子值本身做相关性检查，因为线性模型主要受共线性影响
        # 也可以用 rank 后的相关性
        # [OPT] 计算全量相关性矩阵可能较慢，但因子数通常不多 (100以内)
        
        # 提取数据并计算相关矩阵
        # 处理 NaN: dropna? fillna(0)? usually dropping rows with any nan is safe for corr
        # or use min_periods
        try:
            # 确保列存在
            valid_cols = [c for c in sorted_factors if c in X.columns]
            if len(valid_cols) < 2:
                return valid_cols
                
            # 计算相关矩阵 (with caching)
            cache = get_global_cache()
            corr_matrix_arr, cached_cols = cache.get_correlation_matrix(X, valid_cols)
            col_to_idx = {col: i for i, col in enumerate(cached_cols)}
            
            for f in valid_cols:
                is_correlated = False
                f_idx = col_to_idx[f]
                for kept in kept_factors:
                    # 检查与已保留因子的相关性
                    kept_idx = col_to_idx[kept]
                    if corr_matrix_arr[f_idx, kept_idx] > threshold:
                        is_correlated = True
                        break
                
                if not is_correlated:
                    kept_factors.append(f)
                    
            if len(kept_factors) < len(valid_cols):
                self.logger.info(f"相关性筛选剔除: {len(valid_cols) - len(kept_factors)} 个因子 (从 {len(valid_cols)} 到 {len(kept_factors)})")
                
            return kept_factors
            
        except LabelIsolationError:
            raise
        except Exception as e:
            self.logger.error(f"相关性筛选失败: {e}")
            raise

