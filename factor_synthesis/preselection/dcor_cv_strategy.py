# -*- coding: utf-8 -*-
"""
dCor CV (Cross-Validation) 预筛选策略

基于时序交叉验证动态选择最优 dCor 阈值的策略。
在每个训练窗口内，通过 CV 寻找最优阈值，避免人工固定阈值的局限性。
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import dcor
from scipy.stats import spearmanr
from joblib import Parallel, delayed

from .base import PreSelectionStrategy, get_global_cache
from .registry import register_strategy


def _compute_single_dcor(
    col: str, 
    factor_values: np.ndarray, 
    y_values: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    计算单个因子的 dCor (供并行调用)
    
    Args:
        col: 因子名称
        factor_values: 因子值数组
        y_values: 收益率数组
        
    Returns:
        包含 factor_name, dcor_mean, n_samples 的字典，或 None
    """
    try:
        # 处理 NaN
        valid_mask = ~(np.isnan(factor_values) | np.isnan(y_values))
        factor_clean = factor_values[valid_mask]
        y_clean = y_values[valid_mask]
        
        if len(factor_clean) < 10:
            return None
        
        dcor_value = dcor.distance_correlation(
            factor_clean, 
            y_clean, 
            method='AVL'
        )
        
        return {
            'factor_name': col,
            'dcor_mean': dcor_value,
            'n_samples': len(factor_clean)
        }
    except LabelIsolationError:
        raise
    except Exception:
        raise


@register_strategy
class DCorCVPreSelectionStrategy(PreSelectionStrategy):
    """基于时序交叉验证的 dCor 动态阈值筛选策略
    
    **数据边界：每折预选只使用该折训练子集**
    整个 CV 过程都在训练窗口内进行:
    - 训练窗口被划分为 CV train/val 折
    - CV train 折用于计算 dCor 并筛选因子
    - CV val 折用于评估选出因子的性能
    - 最终阈值在整个训练窗口上应用
    - 外部验证集/测试集完全不参与阈值选择
    - 标签跨期的 purge/gap 由调用方另行处理，不在此保证范围内
    
    配置示例:
    ```json
    {
        "pre_selection": {
            "enabled": true,
            "strategy": "dcor_cv",
            "threshold_search_space": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10],
            "cv_folds": 3,
            "cv_metric": "mean_dcor",
            "cv_val_ratio": 0.15,
            "cv_initial_train_ratio": 0.5,
            "cv_holdout_ratio": 0.1,
            "min_abs_metric": 0.05,
            "max_correlation": 0.5,
            "fallback_top_n": 3
        }
    }
    ```
    
    CV 比例配置说明:
    - cv_val_ratio: 每折验证集占总数据的比例 (默认 0.1，即 10%)
    - cv_initial_train_ratio: 第一折训练集的初始比例 (默认 0.6，即 60%)
    - cv_holdout_ratio: 预留不参与 CV 的数据比例 (默认 0.1，即最后 10%)
    
    支持的 cv_metric:
    - "mean_ic": 选出因子组合的平均 Spearman IC (越大越好)
    - "ir": IC 的信息比率 IC_mean / IC_std (越大越好)
    - "composite_ic": IC * sqrt(n_days) (越大越好)
    - "mean_dcor": 选出因子的平均 dCor (越大越好)
    - "composite_dcor": dCor * sqrt(n_factors) (越大越好)
    
    筛选逻辑:
    1. 计算所有因子的 dCor (用于排序)
    2. 相关性筛选 (先去除冗余因子)
    3. 在全部候选因子上进行 CV 优化，独立于全窗口预选:
       a. 时序分割训练数据为 K 折 (expanding window, 时间严格有序)
       b. 对每个候选阈值:
          - 在每个 fold 的 CV-train 子集计算 dCor、排序、去相关及阈值筛选
          - 在 CV-val 子集评估选出因子的性能 (IC 或 dCor)
          - 跨 fold 平均得到该阈值的 CV score
       c. 选择 CV score 最高的阈值
    4. 用 CV 优化的最优阈值在完整训练数据上筛选因子
    """
    
    name = "dcor_cv"
    
    # 默认阈值搜索空间
    DEFAULT_THRESHOLD_SPACE = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        执行基于 CV 的 dCor 动态阈值因子筛选
        
        Args:
            X: 因子数据 (MultiIndex: date, stock_code)
            y: 前瞻收益率 (MultiIndex: date, stock_code)
            
        Returns:
            保留的因子名称列表
        """
        self._cv_results = {}
        self._validate_inputs(X, y)
        
        # 检查是否启用
        if not self.config.get('enabled', True):
            self.logger.info("[dCor-CV策略] 未启用，返回所有因子")
            return list(X.columns)
        
        self.logger.debug("执行因子预筛选 (dCor-CV Strategy)...")
        
        # 获取配置
        threshold_space = self.config.get(
            'threshold_search_space', 
            self.DEFAULT_THRESHOLD_SPACE
        )
        cv_folds = self.config.get('cv_folds', 3)
        cv_metric = self.config.get('cv_metric', 'mean_ic')
        if cv_metric not in ('mean_ic', 'ir', 'composite_ic', 'mean_dcor', 'composite_dcor'):
            raise ValueError(f'Unsupported cv_metric: {cv_metric!r}')
        fallback_threshold = self.config.get('min_abs_metric', 0.05)
        
        # 索引对齐
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        if X_aligned.empty:
            self.logger.warning("数据为空，返回所有因子")
            return list(X.columns)
        
        initial_count = len(X.columns)
        
        # Step 1: 计算所有因子的 dCor (用于排序)
        perf_df = self._calculate_dcor_performance(X_aligned, y_aligned)
        perf_df.attrs['correlation_filter_version'] = 'input_priority_v2'
        perf_df.attrs['cv_preselection_version'] = 'fold_train_only_v2'
        perf_df.attrs['cv_split_version'] = 'holdout_bounded_v2'
        perf_df.attrs['fallback_correlation_version'] = 'cap_first_v2'
        perf_df.attrs['threshold_order_version'] = 'input_priority_v2'
        perf_df.attrs['cv_diagnostics_version'] = 'cv_diagnostics_v2'
        self._performance_report = perf_df
        
        if perf_df.empty:
            self.logger.warning("无法计算因子 dCor 指标，跳过筛选，保留所有因子")
            return list(X.columns)
        
        avg_dcor = perf_df['dcor_mean'].mean()
        self.logger.debug(f"全量因子 dCor 均值: {avg_dcor:.4f} (共 {len(perf_df)} 个)")
        
        # 所有因子按 dCor 降序排列
        sorted_factors_by_dcor = perf_df.sort_values('dcor_mean', ascending=False)['factor_name'].tolist()
        
        # Step 2: 相关性筛选 (先去除冗余因子)
        selected_factors = sorted_factors_by_dcor
        metrics = self.config.get('metrics', {})
        
        if 'max_correlation' in metrics and len(sorted_factors_by_dcor) > 1:
            max_corr = metrics['max_correlation']
            selected_factors = self._filter_by_correlation(
                X_aligned, sorted_factors_by_dcor, max_corr
            )
            after_corr_count = len(selected_factors)
            self.logger.debug(f"相关性筛选: {initial_count} -> {after_corr_count}")
        else:
            after_corr_count = initial_count
        
        # Step 3: 每折独立预选，不能先用全窗口标签或相关性裁剪候选列
        best_threshold, cv_results = self._find_optimal_threshold_cv(
            X_aligned, y_aligned,
            threshold_space=threshold_space,
            n_folds=cv_folds,
            metric=cv_metric
        )
        
        if best_threshold is None:
            self.logger.warning(f"CV 未能找到有效阈值，使用 fallback: {fallback_threshold}")
            best_threshold = fallback_threshold
        
        self._cv_results.update(
            version='cv_diagnostics_v2', best_threshold=best_threshold,
            threshold_scores=dict(cv_results), metric=cv_metric,
            folds=self._cv_results.get('folds', []),
        )

        # Step 4: 应用 CV 优化的 dCor 阈值
        perf_selected = perf_df[perf_df['factor_name'].isin(selected_factors)]
        mask = perf_selected['dcor_mean'] >= best_threshold
        qualified = set(perf_selected[mask]['factor_name'])
        selected_factors = [factor for factor in selected_factors if factor in qualified]
        
        final_count = len(selected_factors)
        
        # 单行汇总日志
        self.logger.info(
            f"[dCor-CV] {initial_count} -> {after_corr_count} (corr) -> {final_count} "
            f"(thr={best_threshold:.3f}, cv={cv_results.get(best_threshold, 0):.3f})"
        )
        
        # 安全保底
        if final_count == 0:
            fallback_n = self.config.get('fallback_top_n', 3)
            self.logger.warning(
                f"预筛选后无剩余因子！从 dCor 最大的 Top {fallback_n} 候选中保底"
            )
            top_n = perf_df.nlargest(fallback_n, 'dcor_mean')['factor_name'].tolist()
            if 'max_correlation' in metrics and len(top_n) > 1:
                top_n = self._filter_by_correlation(
                    X_aligned, top_n, metrics['max_correlation']
                )
            if len(top_n) < fallback_n:
                self.logger.warning(
                    f"[dCor-CV] 外层保底缺额: {fallback_n - len(top_n)}；相关上限优先"
                )
            return top_n
        
        return selected_factors
    
    def _find_optimal_threshold_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold_space: List[float],
        n_folds: int,
        metric: str
    ) -> Tuple[Optional[float], Dict[float, float]]:
        """
        通过时序交叉验证寻找最优 dCor 阈值
        
        **注意**: 此方法在训练窗口内部进行 CV，不使用任何外部数据。
        每折排序和去相关只使用该折 train；标签跨期的 purge/gap 另行处理。
        
        Args:
            X: 因子数据 (训练窗口内)
            y: 前瞻收益率 (训练窗口内)
            threshold_space: 候选阈值列表
            n_folds: CV 折数
            metric: 评估指标:
                - "mean_ic": Spearman IC 均值
                - "ir": IC 信息比率
                - "composite_ic": IC * sqrt(n)
                - "mean_dcor": dCor 均值
                - "composite_dcor": dCor * sqrt(n)
            
        Returns:
            (最优阈值, {阈值: CV得分} 字典)
        """
        # 获取所有日期
        dates = X.index.get_level_values(0).unique().sort_values()
        n_dates = len(dates)
        
        if n_dates < n_folds + 1:
            self.logger.warning(
                f"日期数 ({n_dates}) 不足以进行 {n_folds} 折 CV，"
                f"使用 fallback"
            )
            return None, {}
        
        # 获取 CV 比例配置
        cv_val_ratio = self.config.get('cv_val_ratio', 0.1)
        cv_initial_train_ratio = self.config.get('cv_initial_train_ratio', 0.6)
        cv_holdout_ratio = self.config.get('cv_holdout_ratio', 0.1)
        
        # 生成 expanding window CV splits
        cv_splits = self._generate_expanding_cv_splits(
            dates, n_folds,
            val_ratio=cv_val_ratio,
            initial_train_ratio=cv_initial_train_ratio,
            holdout_ratio=cv_holdout_ratio
        )
        
        if not cv_splits:
            self.logger.warning("无法生成有效的 CV splits")
            return None, {}
        
        self.logger.debug(
            f"[CV] 生成 {len(cv_splits)} 个时序 CV 折 (expanding window)"
        )
        
        self._cv_results = {'folds': [
            {'train_count': len(train), 'validation_count': len(val),
             'train_dates': [str(d) for d in train],
             'validation_dates': [str(d) for d in val]}
            for train, val in cv_splits
        ]}

        # 对每个候选阈值评估
        threshold_scores: Dict[float, float] = {}
        
        for threshold in threshold_space:
            fold_scores = []
            
            for fold_idx, (train_dates, val_dates) in enumerate(cv_splits):
                try:
                    # 提取 train 和 val 数据
                    X_train = X.loc[X.index.get_level_values(0).isin(train_dates)]
                    y_train = y.loc[y.index.get_level_values(0).isin(train_dates)]
                    X_val = X.loc[X.index.get_level_values(0).isin(val_dates)]
                    y_val = y.loc[y.index.get_level_values(0).isin(val_dates)]
                    
                    if X_train.empty or X_val.empty:
                        continue
                    
                    # 在 train 上计算 dCor 并筛选因子
                    train_dcor = self._calculate_dcor_performance(X_train, y_train)
                    
                    if train_dcor.empty:
                        continue
                    
                    sorted_factors = train_dcor.sort_values(
                        'dcor_mean', ascending=False
                    )['factor_name'].tolist()
                    metrics = self.config.get('metrics', {})
                    retained = sorted_factors
                    if 'max_correlation' in metrics and len(retained) > 1:
                        retained = self._filter_by_correlation(
                            X_train, retained, metrics['max_correlation']
                        )
                    selected_mask = (
                        train_dcor['factor_name'].isin(retained)
                        & (train_dcor['dcor_mean'] >= threshold)
                    )
                    selected_factors = train_dcor[selected_mask]['factor_name'].tolist()
                    
                    # 如果没有因子通过阈值，取 top 3
                    if not selected_factors:
                        selected_factors = train_dcor.nlargest(
                            3, 'dcor_mean'
                        )['factor_name'].tolist()
                        if 'max_correlation' in metrics and len(selected_factors) > 1:
                            selected_factors = self._filter_by_correlation(
                                X_train, selected_factors, metrics['max_correlation']
                            )
                        if len(selected_factors) < 3:
                            self.logger.warning(
                                f"[CV] Fold {fold_idx} Top 3 保底缺额: "
                                f"{3 - len(selected_factors)}；相关上限优先"
                            )
                    
                    # 在 val 上评估选出因子的性能
                    score = self._evaluate_factors_on_validation(
                        X_val[selected_factors], y_val, metric
                    )
                    
                    if not np.isnan(score):
                        fold_scores.append(score)
                        
                except LabelIsolationError:
                    raise
                except Exception as e:
                    self.logger.debug(
                        f"[CV] Fold {fold_idx} 阈值 {threshold} 失败: {e}"
                    )
                    raise
            
            # 计算跨 fold 平均得分
            if fold_scores:
                avg_score = np.mean(fold_scores)
                threshold_scores[threshold] = avg_score
                self.logger.debug(
                    f"[CV] 阈值 {threshold:.4f}: 平均 {metric} = {avg_score:.4f} "
                    f"(来自 {len(fold_scores)} 个有效折)"
                )
        
        if not threshold_scores:
            return None, {}
        
        # 选择得分最高的阈值
        best_threshold = max(threshold_scores, key=threshold_scores.get)
        
        return best_threshold, threshold_scores
    
    def _generate_expanding_cv_splits(
        self,
        dates: pd.DatetimeIndex,
        n_folds: int,
        val_ratio: float = 0.1,
        initial_train_ratio: float = 0.6,
        holdout_ratio: float = 0.1
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        生成 expanding window 时序 CV splits
        
        使用 expanding window 而非 sliding window，以获得更稳定的估计。
        
        例如 n_folds=3, val_ratio=0.15, initial_train_ratio=0.5:
          Fold 1: Train [0:50%], Val [50%:65%]
          Fold 2: Train [0:60%], Val [60%:75%]
          Fold 3: Train [0:70%], Val [70%:85%]
          
        Args:
            dates: 排序后的日期索引
            n_folds: 折数
            val_ratio: 每折验证集占总数据的比例 (默认 0.1)
            initial_train_ratio: 第一折训练集的初始比例 (默认 0.6)
            holdout_ratio: 预留不参与 CV 的数据比例 (默认 0.1)
            
        Returns:
            [(train_dates, val_dates), ...] 列表
        """
        n_dates = len(dates)
        
        # 预留最后 holdout_ratio 作为 holdout (不参与 CV)
        effective_end = int(n_dates * (1.0 - holdout_ratio))
        
        # 每折的 validation 大小
        val_size = max(int(n_dates * val_ratio), 5)  # 至少 5 天
        
        # 初始 train 结束位置
        initial_train_end = int(n_dates * initial_train_ratio)
        
        # 计算步长
        remaining_for_cv = effective_end - initial_train_end - val_size
        if remaining_for_cv <= 0 or n_folds <= 1:
            # 只能做一折
            train_dates = dates[:initial_train_end]
            val_end = min(initial_train_end + val_size, effective_end)
            val_dates = dates[initial_train_end:val_end]
            if len(train_dates) == 0 or len(val_dates) == 0:
                return []
            return [(train_dates, val_dates)]
        
        step_size = remaining_for_cv // (n_folds - 1) if n_folds > 1 else remaining_for_cv
        step_size = max(step_size, 1)
        
        splits = []
        for i in range(n_folds):
            train_end = initial_train_end + i * step_size
            val_start = train_end
            val_end = min(val_start + val_size, effective_end)
            
            if val_end <= val_start:
                break
            
            train_dates = dates[:train_end]
            val_dates = dates[val_start:val_end]
            
            if len(train_dates) > 0 and len(val_dates) > 0:
                splits.append((train_dates, val_dates))
        
        return splits
    
    def _calculate_dcor_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        并行计算每个因子的 dCor 指标
        
        Args:
            X: 因子数据
            y: 前瞻收益率
            
        Returns:
            DataFrame with columns: [factor_name, dcor_mean, n_samples]
        """
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        if X_aligned.empty:
            return pd.DataFrame()
        
        # 获取并行配置
        parallel_cfg = self.config.get('parallel', {})
        parallel_enabled = parallel_cfg.get('enabled', True)
        n_jobs = parallel_cfg.get('n_jobs', -1)
        
        # 预提取数据为 numpy 数组，避免在并行中传递 DataFrame
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
    
    def _evaluate_factors_on_validation(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str
    ) -> float:
        """
        在验证集上评估选出因子的性能
        
        注意: 此验证集是训练窗口内部的 CV 验证折，不是外部测试集。
        本函数只使用传入的验证折；内层预选的数据边界取决于上游 select。
        
        Args:
            X_val: CV验证折因子数据 (训练窗口内的子集)
            y_val: CV验证折收益率 (训练窗口内的子集)
            metric: 评估指标:
                - "mean_ic": 选出因子组合的平均 Spearman IC
                - "ir": IC 的信息比率 (IC_mean / IC_std)
                - "composite_ic": IC * sqrt(n_days)
                - "mean_dcor": 选出因子的平均 dCor
                - "composite_dcor": dCor 均值 * sqrt(有效因子数)
            
        Returns:
            评估得分 (越大越好)
        """
        if X_val.empty:
            return np.nan
        
        common_index = X_val.index.intersection(y_val.index)
        X_aligned = X_val.loc[common_index]
        y_aligned = y_val.loc[common_index]
        
        if X_aligned.empty:
            return np.nan
        
        # dCor 系列指标: 直接计算选出因子的 dCor
        if metric in ('mean_dcor', 'composite_dcor'):
            return self._compute_dcor_metric(X_aligned, y_aligned, metric)
        
        # IC 系列指标: 计算每日 IC
        return self._compute_ic_metric(X_aligned, y_aligned, metric)
    
    def _compute_dcor_metric(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str
    ) -> float:
        """
        计算 dCor 系列评估指标
        
        Args:
            X: 因子数据
            y: 收益率
            metric: "mean_dcor" 或 "composite_dcor"
            
        Returns:
            dCor 指标值
        """
        dcor_values = []
        
        for col in X.columns:
            try:
                factor_data = X[col]
                df = pd.DataFrame({'factor': factor_data, 'y': y}).dropna()
                
                if len(df) < 10:
                    continue
                
                dcor_value = dcor.distance_correlation(
                    df['factor'].values,
                    df['y'].values,
                    method='AVL'
                )
                
                if not np.isnan(dcor_value):
                    dcor_values.append(dcor_value)
                    
            except LabelIsolationError:
                raise
            except Exception:
                raise
        
        if not dcor_values:
            return np.nan
        
        dcor_array = np.array(dcor_values)
        
        if metric == 'mean_dcor':
            return np.mean(dcor_array)
        elif metric == 'composite_dcor':
            # dCor * sqrt(n_factors) - 考虑因子数量
            return np.mean(dcor_array) * np.sqrt(len(dcor_array))
        else:
            return np.mean(dcor_array)
    
    def _compute_ic_metric(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str
    ) -> float:
        """
        计算 IC 系列评估指标
        
        Args:
            X: 因子数据
            y: 收益率
            metric: "mean_ic", "ir", 或 "composite_ic"
            
        Returns:
            IC 指标值
        """
        daily_ics = []
        dates = X.index.get_level_values(0).unique()
        
        for date in dates:
            try:
                X_day = X.xs(date, level=0)
                y_day = y.xs(date, level=0)
                
                # 所有因子值取均值作为组合因子
                combined_factor = X_day.mean(axis=1)
                
                # 计算 Spearman IC
                valid_mask = ~(combined_factor.isna() | y_day.isna())
                if valid_mask.sum() < 10:
                    continue
                
                ic, _ = spearmanr(
                    combined_factor[valid_mask],
                    y_day[valid_mask]
                )
                
                if not np.isnan(ic):
                    daily_ics.append(ic)
                    
            except LabelIsolationError:
                raise
            except Exception:
                raise
        
        if not daily_ics:
            return np.nan
        
        ic_array = np.array(daily_ics)
        
        if metric == 'mean_ic':
            return np.abs(np.mean(ic_array))
        elif metric == 'ir':
            ic_std = np.std(ic_array)
            if ic_std < 1e-8:
                return np.abs(np.mean(ic_array))
            return np.abs(np.mean(ic_array)) / ic_std
        elif metric == 'composite_ic':
            # IC * sqrt(n_days) - 考虑样本量
            return np.abs(np.mean(ic_array)) * np.sqrt(len(ic_array))
        else:
            return np.abs(np.mean(ic_array))
    
    def _filter_by_correlation(
        self,
        X: pd.DataFrame,
        sorted_factors: List[str],
        threshold: float
    ) -> List[str]:
        """
        向量化贪婪相关性筛选 (with caching)
        
        使用 NumPy 向量化操作替代内层 Python 循环，
        将 O(n^2) 的 Python 循环开销降低为 O(n) 的 NumPy 操作。
        
        Args:
            X: 因子数据
            sorted_factors: 按 dCor 降序排列的因子列表
            threshold: 相关性阈值
            
        Returns:
            筛选后的因子列表
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
                    # 第一个因子总是保留
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
    
    def get_cv_results(self) -> Dict[str, Any]:
        """
        获取 CV 结果详情
        
        Returns:
            包含 CV 过程详细信息的字典
        """
        return getattr(self, '_cv_results', {})
