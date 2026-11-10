# -*- coding: utf-8 -*-
"""
Barra分组预筛选策略

基于Barra多因子模型分类体系，在每个风格组内部进行因子筛选，
然后汇总各组选出的因子用于Lasso合成。

支持:
- 按Barra风格分组 (Momentum, Size, Value, Quality, Growth, Volatility, Liquidity, Sentiment)
- 每组独立的筛选策略配置 (dcor/ic_ir)
- 每组独立的相关性阈值
- 每组最小保留因子数 (top_n_per_group)
- 跨组最终相关性过滤
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from .base import PreSelectionStrategy, reset_global_cache
from .registry import register_strategy, get_preselection_strategy


class BarraGroupedPreSelector:
    """Barra分组因子预筛选器
    
    不直接继承PreSelectionStrategy，而是组合使用现有策略。
    这样可以更灵活地处理分组逻辑。
    
    配置示例 (barra_groups.json):
    ```json
    {
        "version": "1.0",
        "default_strategy": {
            "method": "dcor",
            "min_abs_metric": 0.03,
            "max_correlation": 0.6,
            "top_n_per_group": 3
        },
        "groups": {
            "momentum": {
                "enabled": true,
                "strategy": {"method": "dcor", "min_abs_metric": 0.04},
                "factors": ["DemoMomentum", ...]
            },
            ...
        }
    }
    ```
    
    使用方式:
    ```python
    selector = BarraGroupedPreSelector(
        config_path="./config/barra_groups.json",
        group_overrides={"momentum": True, "size": True, "quality": False}
    )
    selected_factors = selector.select(X, y)
    report = selector.get_group_report()
    ```
    """
    
    def __init__(
        self, 
        config_path: str, 
        group_overrides: Optional[Dict[str, bool]] = None,
        cross_group_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化Barra分组预筛选器
        
        Args:
            config_path: barra_groups.json 配置文件路径
            group_overrides: 组启用状态覆盖字典，如 {"momentum": True, "quality": False}
            cross_group_config: 跨组过滤配置，如 {"enabled": True, "max_correlation": 0.7}
            logger: 日志记录器
        """
        self.config_path = config_path
        self.group_overrides = group_overrides or {}
        self.cross_group_config = cross_group_config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # 加载配置
        self.config = self._load_config()
        
        # 存储各组筛选结果
        self._group_reports: Dict[str, pd.DataFrame] = {}
        self._selection_summary: List[Dict[str, Any]] = []
    
    def _load_config(self) -> Dict[str, Any]:
        """加载并验证Barra分组配置"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Barra配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证必需字段
        if not isinstance(config, dict):
            raise ValueError("Barra配置必须是 JSON 对象")
        if 'groups' not in config:
            raise ValueError("Barra配置缺少 'groups' 字段")
        if not isinstance(config['groups'], dict):
            raise ValueError("Barra配置 'groups' 必须是 JSON 对象")
        
        return config
    
    def _is_group_enabled(self, group_name: str) -> bool:
        """检查组是否启用 (考虑覆盖配置)"""
        # 首先检查覆盖配置
        if group_name in self.group_overrides:
            enabled = self.group_overrides[group_name]
        else:
            # 否则使用配置文件中的设置
            group_cfg = self.config.get('groups', {}).get(group_name, {})
            enabled = group_cfg.get('enabled', True)
        if not isinstance(enabled, bool):
            raise ValueError(f"组 '{group_name}' 的 enabled 必须是布尔值")
        return enabled
    
    def _get_group_strategy_config(self, group_name: str) -> Dict[str, Any]:
        """获取组的策略配置 (合并默认配置)
        
        支持的策略方法:
        - "dcor": 使用固定 dCor 阈值
        - "dcor_cv": 使用时序交叉验证动态选择 dCor 阈值
        - "stability_selection": 使用 Stability Selection 算法
        - "ic_ir": 使用 IC/IR 阈值
        
        兼容旧配置:
        - 若 method="dcor" 且 threshold_selection="cv"，自动切换到 dcor_cv
        """
        default_cfg = self.config.get('default_strategy', {})
        group_cfg = self.config.get('groups', {}).get(group_name, {})
        group_strategy = group_cfg.get('strategy', {})
        
        # 合并：默认 -> 组级别策略
        merged = {**default_cfg, **group_strategy}
        
        # 检查阈值选择模式 (兼容旧配置)
        threshold_selection = merged.get('threshold_selection', 'fixed')
        method = merged.get('method', 'dcor')
        
        # 如果使用 CV 模式且原方法是 dcor，则切换到 dcor_cv 策略
        if threshold_selection == 'cv' and method == 'dcor':
            strategy_name = 'dcor_cv'
        else:
            strategy_name = method
        
        # 转换为 pre_selection 格式
        result = {
            'enabled': True,
            'strategy': strategy_name,
            'metrics': {},
            'fallback_top_n': merged.get('fallback_top_n', merged.get('top_n_per_group', 2))
        }
        
        # 根据策略类型设置对应的配置
        if strategy_name == 'stability_selection':
            # Stability Selection 特有配置
            if 'base_metric' in merged:
                result['base_metric'] = merged['base_metric']
            if 'n_bootstrap' in merged:
                result['n_bootstrap'] = merged['n_bootstrap']
            if 'subsample_ratio' in merged:
                result['subsample_ratio'] = merged['subsample_ratio']
            if 'selection_threshold' in merged:
                result['selection_threshold'] = merged['selection_threshold']
            if 'selection_mode' in merged:
                result['selection_mode'] = merged['selection_mode']
            if 'per_sample_top_k' in merged:
                result['per_sample_top_k'] = merged['per_sample_top_k']
            # 百分比选择模式
            if 'per_sample_top_pct' in merged:
                result['per_sample_top_pct'] = merged['per_sample_top_pct']
            # 支持新旧配置名
            if 'per_sample_metric_threshold' in merged:
                result['per_sample_metric_threshold'] = merged['per_sample_metric_threshold']
            elif 'per_sample_dcor_threshold' in merged:
                result['per_sample_metric_threshold'] = merged['per_sample_dcor_threshold']
            if 'random_seed' in merged:
                result['random_seed'] = merged['random_seed']
            if 'max_correlation' in merged:
                result['metrics']['max_correlation'] = merged['max_correlation']
            # 并行配置
            if 'parallel' in merged:
                result['parallel'] = merged['parallel']
                
        elif strategy_name == 'dcor_cv':
            # CV 模式特有配置
            if 'threshold_search_space' in merged:
                result['threshold_search_space'] = merged['threshold_search_space']
            if 'cv_folds' in merged:
                result['cv_folds'] = merged['cv_folds']
            if 'cv_metric' in merged:
                result['cv_metric'] = merged['cv_metric']
            # CV 比例配置
            if 'cv_val_ratio' in merged:
                result['cv_val_ratio'] = merged['cv_val_ratio']
            if 'cv_initial_train_ratio' in merged:
                result['cv_initial_train_ratio'] = merged['cv_initial_train_ratio']
            if 'cv_holdout_ratio' in merged:
                result['cv_holdout_ratio'] = merged['cv_holdout_ratio']
            if 'min_abs_metric' in merged:
                result['min_abs_metric'] = merged['min_abs_metric']  # fallback
            if 'max_correlation' in merged:
                result['metrics']['max_correlation'] = merged['max_correlation']
                
        elif method == 'dcor':
            if 'min_abs_metric' in merged:
                result['metrics']['min_abs_dcor'] = merged['min_abs_metric']
            if 'max_correlation' in merged:
                result['metrics']['max_correlation'] = merged['max_correlation']
                
        elif method == 'ic_ir':
            if 'min_abs_ic' in merged:
                result['metrics']['min_abs_ic'] = merged['min_abs_ic']
            if 'min_abs_ir' in merged:
                result['metrics']['min_abs_ir'] = merged['min_abs_ir']
            if 'min_abs_metric' in merged:
                # 兼容：min_abs_metric 作为 min_abs_ic 的别名
                if 'min_abs_ic' in merged and merged['min_abs_ic'] != merged['min_abs_metric']:
                    self.logger.warning(
                        f"[{group_name}] min_abs_ic={merged['min_abs_ic']} 与 "
                        f"min_abs_metric={merged['min_abs_metric']} 冲突；保留 min_abs_metric 优先"
                    )
                result['metrics']['min_abs_ic'] = merged['min_abs_metric']
            if 'max_correlation' in merged:
                result['metrics']['max_correlation'] = merged['max_correlation']
        
        return result, merged.get('top_n_per_group', 3)
    
    def _get_group_factors(self, group_name: str) -> List[str]:
        """获取组内的因子列表"""
        group_cfg = self.config.get('groups', {}).get(group_name, {})
        return group_cfg.get('factors', [])
    
    def _process_single_group(
        self,
        group_name: str,
        group_cfg: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        available_factors: set,
    ) -> Tuple[str, List[str], pd.DataFrame, Dict[str, Any]]:
        """
        处理单个 Barra 组的筛选 (可并行调用)
        
        Args:
            group_name: 组名
            group_cfg: 组配置
            X: 完整因子数据
            y: 收益率
            available_factors: 可用因子集合
            
        Returns:
            (group_name, selected_factors, perf_df, summary)
        """
        # 检查组是否启用
        if not self._is_group_enabled(group_name):
            return (group_name, [], pd.DataFrame(), {
                'group': group_name,
                'configured': 0,
                'available': 0,
                'selected': 0,
                'status': 'disabled'
            })
        
        # 获取组内因子
        group_factors = self._get_group_factors(group_name)
        
        # 过滤出实际存在的因子
        existing_factors = [f for f in group_factors if f in available_factors]
        
        if not existing_factors:
            return (group_name, [], pd.DataFrame(), {
                'group': group_name,
                'configured': len(group_factors),
                'available': 0,
                'selected': 0,
                'status': 'no_factors'
            })
        
        # 获取组策略配置
        strategy_cfg, top_n_per_group = self._get_group_strategy_config(group_name)
        
        # 提取组内因子数据
        X_group = X[existing_factors]
        
        # 执行组内筛选
        selected_in_group, perf_df = self._select_within_group(
            X_group, y, group_name, strategy_cfg, top_n_per_group
        )
        
        # 添加组标签
        if not perf_df.empty:
            perf_df = perf_df.copy()
            perf_df['group'] = group_name
        
        summary = {
            'group': group_name,
            'configured': len(group_factors),
            'available': len(existing_factors),
            'selected': len(selected_in_group),
            'factors': selected_in_group,
            'status': 'ok' if selected_in_group else 'empty'
        }
        
        return (group_name, selected_in_group, perf_df, summary)
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        执行Barra分组预筛选
        
        Performance: 支持组级并行处理 (parallel_groups=True)
        
        Args:
            X: 因子数据 (MultiIndex: date, stock_code)
            y: 前瞻收益率 (MultiIndex: date, stock_code)
            
        Returns:
            所有组选中因子的合并列表
        """
        self.logger.debug("执行 Barra 分组因子预筛选")
        
        available_factors = set(X.columns)
        all_selected: List[str] = []
        self._selection_summary = []
        self._group_reports = {}
        
        groups = self.config.get('groups', {})
        
        # === Check parallel configuration ===
        parallel_cfg = self.config.get('parallel_groups', {})
        parallel_enabled = parallel_cfg.get('enabled', False)
        n_jobs = parallel_cfg.get('n_jobs', -1)
        
        enabled_groups = [
            (name, cfg) for name, cfg in groups.items() 
            if self._is_group_enabled(name)
        ]
        
        if parallel_enabled and len(enabled_groups) > 1:
            # === Parallel group processing ===
            self.logger.info(f"[Barra] 并行处理 {len(enabled_groups)} 个组 (n_jobs={n_jobs})")
            
            # Reset cache before parallel execution to avoid stale data
            reset_global_cache()
            
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(self._process_single_group)(
                    name, cfg, X, y, available_factors
                )
                for name, cfg in enabled_groups
            )
            
            # Collect results
            for group_name, selected, perf_df, summary in results:
                if summary['status'] == 'disabled':
                    self.logger.info(f"[{group_name}] 已禁用，跳过")
                    continue
                elif summary['status'] == 'no_factors':
                    self.logger.warning(f"[{group_name}] 无可用因子")
                
                if not perf_df.empty:
                    self._group_reports[group_name] = perf_df
                
                self._selection_summary.append(summary)
                all_selected.extend(selected)
                
                self.logger.debug(
                    f"[{group_name}] 筛选结果: {summary['available']} -> {len(selected)} 个因子"
                )
        else:
            # === Sequential processing (original code) ===
            for group_name, group_cfg in groups.items():
                # 检查组是否启用
                if not self._is_group_enabled(group_name):
                    self.logger.info(f"[{group_name}] 已禁用，跳过")
                    continue
                
                # 获取组内因子
                group_factors = self._get_group_factors(group_name)
                
                # 过滤出实际存在的因子
                existing_factors = [f for f in group_factors if f in available_factors]
                
                if not existing_factors:
                    self.logger.warning(f"[{group_name}] 无可用因子 (配置: {len(group_factors)}, 实际: 0)")
                    self._selection_summary.append({
                        'group': group_name,
                        'configured': len(group_factors),
                        'available': 0,
                        'selected': 0,
                        'status': 'no_factors'
                    })
                    continue
                
                self.logger.debug(f"[{group_name}] 可用因子: {len(existing_factors)}/{len(group_factors)}")
                
                # 获取组策略配置
                strategy_cfg, top_n_per_group = self._get_group_strategy_config(group_name)
                
                # 提取组内因子数据
                X_group = X[existing_factors]
                
                # 执行组内筛选
                selected_in_group, perf_df = self._select_within_group(
                    X_group, y, group_name, strategy_cfg, top_n_per_group
                )
                
                # 存储报告
                if not perf_df.empty:
                    perf_df['group'] = group_name
                    self._group_reports[group_name] = perf_df
                
                # 记录摘要
                self._selection_summary.append({
                    'group': group_name,
                    'configured': len(group_factors),
                    'available': len(existing_factors),
                    'selected': len(selected_in_group),
                    'factors': selected_in_group,
                    'status': 'ok' if selected_in_group else 'empty'
                })
                
                all_selected.extend(selected_in_group)
                
                self.logger.debug(
                    f"[{group_name}] 筛选结果: {len(existing_factors)} -> {len(selected_in_group)} 个因子"
                )
        
        # 去重 (理论上组间不应有重复，但以防万一)
        all_selected = list(dict.fromkeys(all_selected))
        
        self.logger.debug(f"分组筛选汇总: 共选出 {len(all_selected)} 个因子")
        
        # 跨组相关性过滤 (可选)
        cross_cfg = self.cross_group_config or self.config.get('cross_group_settings', {}).get('final_correlation_filter', {})
        if cross_cfg.get('enabled', False) and len(all_selected) > 1:
            max_corr = cross_cfg.get('max_correlation', 0.7)
            self.logger.info(f"执行跨组相关性过滤 (阈值: {max_corr})...")
            
            all_selected = self._filter_cross_group_correlation(
                X[all_selected], all_selected, max_corr
            )
            
            self.logger.info(f"跨组相关性过滤后: {len(all_selected)} 个因子")
        
        # 检查最小/最大因子数限制
        cross_settings = self.config.get('cross_group_settings', {})
        min_total = cross_settings.get('min_total_factors', 5)
        max_total = cross_settings.get('max_total_factors', 100)
        
        if len(all_selected) < min_total:
            self.logger.warning(
                f"选中因子数 ({len(all_selected)}) 少于最小限制 ({min_total})，"
                f"将从各组补充因子"
            )
            all_selected = self._supplement_factors(X, y, all_selected, min_total)
        
        if len(all_selected) > max_total:
            self.logger.warning(
                f"选中因子数 ({len(all_selected)}) 超过最大限制 ({max_total})，"
                f"将截断"
            )
            all_selected = all_selected[:max_total]
        
        self.logger.debug(f"Barra 分组预筛选完成: 最终选出 {len(all_selected)} 个因子")
        
        return all_selected
    
    def _select_within_group(
        self, 
        X_group: pd.DataFrame, 
        y: pd.Series,
        group_name: str,
        strategy_cfg: Dict[str, Any],
        top_n_per_group: int
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        在组内执行因子筛选
        
        Args:
            X_group: 组内因子数据
            y: 前瞻收益率
            group_name: 组名称
            strategy_cfg: 策略配置
            top_n_per_group: 组内最少保留因子数
            
        Returns:
            (选中因子列表, 性能报告DataFrame)
        """
        strategy_name = strategy_cfg.get('strategy', 'dcor')
        
        try:
            # 获取策略实例
            strategy = get_preselection_strategy(strategy_name, strategy_cfg, self.logger)
            
            # 执行筛选
            selected = strategy.select(X_group, y)
            
            # 获取性能报告
            perf_df = strategy.get_performance_report()
            
            # 确保至少保留 top_n_per_group 个因子
            if len(selected) < top_n_per_group and not perf_df.empty:
                # 从性能报告中选取 top_n
                # 根据策略类型确定排序列
                if strategy_name == 'stability_selection':
                    metric_col = 'selection_probability'
                elif strategy_name in ('dcor', 'dcor_cv'):
                    metric_col = 'dcor_mean'
                else:
                    metric_col = 'abs_ic'
                
                if metric_col in perf_df.columns:
                    top_factors = perf_df.nlargest(top_n_per_group, metric_col)['factor_name'].tolist()
                    
                    # 合并 (保留已选 + 补充)
                    for f in top_factors:
                        if f not in selected:
                            selected.append(f)
                        if len(selected) >= top_n_per_group:
                            break
                    
                    self.logger.info(
                        f"  [{group_name}] 补充因子至 top_n={top_n_per_group}: {len(selected)} 个"
                    )
            
            return selected, perf_df
            
        except LabelIsolationError:
            raise
        except Exception as e:
            self.logger.error(f"[{group_name}] 组内筛选失败: {e}")
            # 筛选失败不能伪装为有效因子选择。
            raise
    
    def _filter_cross_group_correlation(
        self, 
        X: pd.DataFrame, 
        factors: List[str], 
        threshold: float
    ) -> List[str]:
        """
        跨组相关性过滤
        
        按各组报告中的指标值降序排列，然后贪婪筛选
        """
        if not factors or len(factors) < 2:
            return factors
        
        try:
            # 构建因子优先级 (从各组报告中汇总)
            factor_scores = {}
            for group_name, report in self._group_reports.items():
                if report.empty:
                    continue
                
                # 选择适当的指标列
                if 'selection_probability' in report.columns:
                    metric_col = 'selection_probability'
                elif 'dcor_mean' in report.columns:
                    metric_col = 'dcor_mean'
                elif 'abs_ic' in report.columns:
                    metric_col = 'abs_ic'
                else:
                    continue
                
                for _, row in report.iterrows():
                    fname = row['factor_name']
                    if fname in factors:
                        # 使用绝对值
                        factor_scores[fname] = abs(row[metric_col])
            
            # 未在报告中的因子给予中等分数
            median_score = np.median(list(factor_scores.values())) if factor_scores else 0.05
            for f in factors:
                if f not in factor_scores:
                    factor_scores[f] = median_score
            
            # 按分数降序排列
            sorted_factors = sorted(factors, key=lambda f: factor_scores.get(f, 0), reverse=True)
            
            # 计算相关性矩阵
            corr_matrix = X[sorted_factors].corr().abs()
            
            # 贪婪筛选
            kept = []
            for f in sorted_factors:
                is_correlated = False
                for kept_f in kept:
                    if corr_matrix.loc[f, kept_f] > threshold:
                        is_correlated = True
                        break
                
                if not is_correlated:
                    kept.append(f)
            
            dropped_count = len(sorted_factors) - len(kept)
            if dropped_count > 0:
                self.logger.info(f"跨组相关性筛选剔除: {dropped_count} 个因子")
            
            return kept
            
        except LabelIsolationError:
            raise
        except Exception as e:
            self.logger.error(f"跨组相关性过滤失败: {e}")
            raise
    
    def _supplement_factors(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        current: List[str], 
        min_count: int
    ) -> List[str]:
        """
        当选中因子不足时，从各组补充因子
        """
        if len(current) >= min_count:
            return current
        
        needed = min_count - len(current)
        current_set = set(current)
        supplements = []
        cross_cfg = self.cross_group_config or self.config.get('cross_group_settings', {}).get('final_correlation_filter', {})
        corr = X.corr().abs() if cross_cfg.get('enabled', False) else None
        
        # 从各组报告中按指标值补充
        for group_name, report in self._group_reports.items():
            if report.empty or len(supplements) >= needed:
                continue
            
            # 选择指标列
            if 'selection_probability' in report.columns:
                metric_col = 'selection_probability'
            elif 'dcor_mean' in report.columns:
                metric_col = 'dcor_mean'
            elif 'abs_ic' in report.columns:
                metric_col = 'abs_ic'
            else:
                continue
            
            # 获取未选中的因子并按指标排序
            candidates = report[~report['factor_name'].isin(current_set)]
            if candidates.empty:
                continue
            
            candidate_count = len(candidates) if corr is not None else needed - len(supplements)
            top_candidates = candidates.nlargest(candidate_count, metric_col)['factor_name'].tolist()
            
            for f in top_candidates:
                if f not in current_set:
                    if corr is not None and any(
                        corr.loc[f, kept] > cross_cfg.get('max_correlation', 0.7)
                        for kept in current + supplements
                    ):
                        continue
                    supplements.append(f)
                    current_set.add(f)
                    if len(supplements) >= needed:
                        break
        
        if supplements:
            self.logger.info(f"补充 {len(supplements)} 个因子: {supplements}")
        if len(supplements) < needed:
            self.logger.warning(f"补足缺额 {needed - len(supplements)} 个因子；保持已启用的相关上限")
        
        return current + supplements
    
    def get_group_report(self) -> pd.DataFrame:
        """
        获取分组筛选汇总报告
        
        Returns:
            包含各组筛选统计的DataFrame
        """
        report = pd.DataFrame(self._selection_summary)
        report.attrs['fallback_correlation_version'] = 'cap_first_v2'
        return report
    
    def get_detailed_reports(self) -> Dict[str, pd.DataFrame]:
        """
        获取各组的详细性能报告
        
        Returns:
            {组名: 性能报告DataFrame}
        """
        return self._group_reports.copy()


# 注册为 PreSelectionStrategy (可选，用于通过统一接口调用)
@register_strategy
class BarraGroupedPreSelectionStrategy(PreSelectionStrategy):
    """Barra分组预筛选策略的封装，用于通过registry统一调用
    
    配置示例:
    ```json
    {
        "pre_selection": {
            "enabled": true,
            "strategy": "barra_grouped",
            "config_path": "./config/barra_groups.json",
            "group_overrides": {
                "momentum": true,
                "quality": false
            },
            "cross_group_correlation_filter": {
                "enabled": true,
                "max_correlation": 0.7
            }
        }
    }
    ```
    """
    
    name = "barra_grouped"
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """执行Barra分组预筛选"""
        self._validate_inputs(X, y)
        
        if not self.config.get('enabled', True):
            self.logger.info("[Barra分组策略] 未启用，返回所有因子")
            return list(X.columns)
        
        config_path = self.config.get('config_path', './config/barra_groups.json')
        group_overrides = self.config.get('group_overrides', {})
        cross_group_cfg = self.config.get('cross_group_correlation_filter', {})
        
        selector = BarraGroupedPreSelector(
            config_path=config_path,
            group_overrides=group_overrides,
            cross_group_config=cross_group_cfg,
            logger=self.logger
        )
        
        selected = selector.select(X, y)
        
        # 存储报告
        self._performance_report = selector.get_group_report()
        
        return selected
