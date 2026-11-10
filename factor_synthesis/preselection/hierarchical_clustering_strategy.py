# -*- coding: utf-8 -*-
"""
层次聚类预筛选策略

基于Spearman相关性的层次聚类实现激进的因子降维,将70个因子聚类到10-15个簇,
每个簇选择1个代表因子(基于dCor/IC/IR指标),最终输出15-20个因子用于LASSO合成。

理论依据: Bailey & López de Prado (2014) - 参数搜索组合数越多,Sharpe过拟合越严重

**新特性 (v2.0)**:
- 支持在聚类前使用可配置的预筛选策略（stability_selection, dcor_cv, ic_ir等）
- 组合模式设计，参考 Barra 分组策略的架构
- 向后兼容：未配置 pre_filter 时保持原有行为

配置示例:

基础配置（向后兼容）:
```json
{
    "hierarchical_clustering": {
        "enabled": true,
        "n_clusters": 12,
        "linkage_method": "ward",
        "correlation_method": "spearman",
        "distance_threshold": null,
        "selection_metric": "dcor",
        "min_metric_threshold": 0.03,
        "max_factors": 20,
        "fallback_top_n": 15
    }
}
```

高级配置（使用 Stability Selection 预筛选）:
```json
{
    "hierarchical_clustering": {
        "enabled": true,
        "n_clusters": 12,
        
        "pre_filter": {
            "enabled": true,
            "strategy": "stability_selection",
            "n_bootstrap": 30,
            "subsample_ratio": 0.5,
            "selection_threshold": 0.6,
            "selection_mode": "top_pct",
            "per_sample_top_pct": 0.3,
            "base_metric": "dcor",
            "max_correlation": 0.5,
            "parallel": {
                "enabled": true,
                "n_jobs": -1
            }
        },
        
        "linkage_method": "ward",
        "selection_metric": "dcor",
        "min_metric_threshold": 0.03,
        "max_factors": 20,
        "fallback_top_n": 15
    }
}
```

支持的预筛选策略:
- `stability_selection`: 使用 Bootstrap + 稳定性选择算法
- `dcor_cv`: 使用时序交叉验证动态选择 dCor 阈值
- `dcor`: 使用固定 dCor 阈值
- `ic_ir`: 使用 IC/IR 阈值
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from typing import List, Dict, Any, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

import dcor

# scipy聚类库
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .base import PreSelectionStrategy, get_global_cache
from .registry import register_strategy, get_preselection_strategy


def _compute_dcor_value(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """计算单个因子的dCor值(与PreSelectionCache签名对齐)"""
    return dcor.distance_correlation(x_values, y_values, method='AVL')


def select_nested_fold_representatives(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[float, List[Dict[str, Any]]]:
    """Fit each frozen clustering cell on the training side of every CV fold."""
    raw_cells = config.get("nested_distance_thresholds")
    if not isinstance(raw_cells, list) or not 3 <= len(raw_cells) <= 6:
        raise ValueError("nested_distance_thresholds must contain 3 to 6 cells")
    cells = [float(value) for value in raw_cells]
    if cells != sorted(set(cells)) or not all(0.0 < value < 1.0 for value in cells):
        raise ValueError("nested_distance_thresholds must be unique and increasing")
    if not X.index.equals(y.index):
        raise ValueError("nested clustering requires exactly aligned X/y indices")

    dates = pd.DatetimeIndex(X.index.get_level_values(0))
    result: Dict[float, List[Dict[str, Any]]] = {}
    for threshold in cells:
        folds: List[Dict[str, Any]] = []
        for fold_number, (train_idx, valid_idx) in enumerate(cv_splits):
            train_idx = np.asarray(train_idx, dtype=int)
            valid_idx = np.asarray(valid_idx, dtype=int)
            train_end = pd.Timestamp(dates[train_idx].max())
            validation_start = pd.Timestamp(dates[valid_idx].min())
            if train_end >= validation_start:
                raise ValueError("nested clustering fold is not strictly chronological")
            cell_config = dict(config)
            cell_config.pop("nested_distance_thresholds", None)
            cell_config["distance_threshold"] = threshold
            cell_config["bypass_global_cache"] = True
            cell_config["strict_no_fallback"] = True
            selected = HierarchicalClusteringStrategy(cell_config, logger).select(
                X.iloc[train_idx], y.iloc[train_idx]
            )
            if not selected:
                raise ValueError(
                    f"nested clustering selected no features for threshold={threshold}"
                )
            folds.append(
                {
                    "fold": fold_number,
                    "train_end": str(train_end.date()),
                    "validation_start": str(validation_start.date()),
                    "selected_features": list(selected),
                }
            )
        result[threshold] = folds
    return result


@register_strategy
class HierarchicalClusteringStrategy(PreSelectionStrategy):
    """层次聚类预筛选策略 (v2.0 - 支持可配置预筛选)
    
    核心流程:
    0. [可选] 预筛选: 使用配置的策略（如 stability_selection）进行初步筛选
    1. 计算因子间Spearman相关性矩阵
    2. 转换为距离矩阵: distance = 1 - |correlation|
    3. 执行层次聚类(Ward链接法)
    4. 切分为n_clusters个簇
    5. 计算每个簇内因子的dCor/IC/IR指标
    6. 每簇选择指标最优的1个代表因子
    7. 返回15-20个代表因子列表
    
    **新特性**:
    - 支持组合模式: 在聚类前可选择使用任意已注册的预筛选策略
    - 配置驱动: 通过 pre_filter 配置块灵活控制预筛选行为
    - 向后兼容: 未配置 pre_filter 时，使用原有的阈值过滤逻辑
    - 增强日志: 区分预筛选和聚类阶段，记录耗时统计
    
    降级策略:
    - 因子数<10: 跳过聚类,按指标直接排序取TopK
    - 预筛选失败: 回退到默认阈值过滤
    - 相关性计算失败: 降级到简单IC筛选
    - dCor计算失败: 降级到IC/IR
    - IC/IR计算失败: 降级到简单相关性排序
    """
    
    name = "hierarchical_clustering"
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """初始化层次聚类策略
        
        Args:
            config: 策略配置字典
            logger: 日志记录器
        """
        super().__init__(config, logger)
        
        # 解析配置参数
        self.n_clusters = config.get('n_clusters', 12)
        self.linkage_method = config.get('linkage_method', 'ward')
        self.correlation_method = config.get('correlation_method', 'spearman')
        self.distance_threshold = config.get('distance_threshold', None)
        self.selection_metric = config.get('selection_metric', 'dcor')
        self.min_metric_threshold = config.get('min_metric_threshold', 0.03)
        self.max_factors = config.get('max_factors', 20)
        self.fallback_top_n = config.get('fallback_top_n', 15)
        self.bypass_global_cache = bool(config.get('bypass_global_cache', False))
        self.strict_no_fallback = bool(config.get('strict_no_fallback', False))
        
        # 并行配置
        self.parallel_cfg = config.get('parallel', {})
        self.parallel_enabled = self.parallel_cfg.get('enabled', True)
        self.n_jobs = self.parallel_cfg.get('n_jobs', -1)
        
        # 解析预筛选配置
        self.pre_filter_enabled, self.pre_filter_strategy_name, self.pre_filter_config = \
            self._get_pre_filter_config()
        
        # 配置验证
        self._validate_config()
        
        log_msg = (
            f"[层次聚类] 初始化完成 - 目标簇数={self.n_clusters}, "
            f"链接方法={self.linkage_method}, 选择指标={self.selection_metric}"
        )
        if self.pre_filter_enabled:
            log_msg += f", 预筛选策略={self.pre_filter_strategy_name}"
        self.logger.info(log_msg)
    
    def _validate_config(self) -> None:
        """验证配置参数合法性"""
        if ((isinstance(self.n_clusters, (float, np.floating))
                and not np.isfinite(self.n_clusters))
                or self.n_clusters < 1
                or self.n_clusters != int(self.n_clusters)):
            raise ValueError(f"n_clusters必须为有限正整数, 当前值: {self.n_clusters}")
        
        if self.linkage_method not in ['ward', 'complete', 'average', 'single']:
            raise ValueError(
                f"不支持的链接方法: {self.linkage_method}, "
                f"支持: ward, complete, average, single"
            )
        
        if self.correlation_method not in ['spearman', 'pearson']:
            raise ValueError(
                f"不支持的相关性方法: {self.correlation_method}, "
                f"支持: spearman, pearson"
            )
        
        if self.selection_metric not in ['dcor', 'ir', 'icir', 'ic']:
            raise ValueError(
                f"不支持的选择指标: {self.selection_metric}, "
                f"支持: dcor, ir, icir, ic"
            )

    def _reject_fallback_if_strict(self, reason: str) -> None:
        if self.strict_no_fallback:
            raise ValueError(
                f"strict nested clustering forbids fallback: {reason}"
            )
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """执行层次聚类因子筛选（支持可配置预筛选）
        
        **工作流程**:
        1. [可选] 预筛选阶段: 使用配置的策略（如 stability_selection）进行初步筛选
        2. 聚类阶段: 对预筛选后的因子计算距离矩阵并执行层次聚类
        3. 簇内选择: 每簇选择 metric 最优的1个代表因子
        
        **向后兼容性**:
        - 如果未配置 pre_filter 或 pre_filter.enabled=false，将使用原有的
          _prefilter_by_metric 方法进行阈值过滤
        
        Args:
            X: 因子数据 (MultiIndex: date, stock_code)
            y: 前瞻收益率 (MultiIndex: date, stock_code)
            
        Returns:
            选中的代表因子列表(15-20个)
        """
        import time
        
        self._performance_report = pd.DataFrame()
        self._performance_report.attrs['distance_gate_version'] = 'distance_independent_v2'
        self._performance_report.attrs['count_limit_version'] = 'all_exits_v2'
        self._performance_report.attrs['metric_sample_version'] = 'pairwise10_valid_days_v2'
        if 'sampling_mode' in (self.config.get('pre_filter') or {}):
            self._performance_report.attrs['pre_filter_config_version'] = 'sampling_mode_v2'
        if self.strict_no_fallback:
            self._performance_report.attrs['strict_error_version'] = 'propagate_v2'
        self._validate_inputs(X, y)
        
        if not self.config.get('enabled', True):
            self.logger.info("[层次聚类] 未启用,返回所有因子")
            return list(X.columns)
        
        n_factors = len(X.columns)
        self.logger.info(
            f"[层次聚类] 开始执行 - 输入因子数={n_factors}, "
            f"{'距离阈值=' + str(self.distance_threshold) if self.distance_threshold is not None else '目标簇数=' + str(self.n_clusters)}, "
            f"{'不限制输出数量' if self.distance_threshold is not None else '最大输出=' + str(self.max_factors)}"
        )
        
        # 降级策略: 因子数不足时跳过聚类
        if n_factors < 10:
            self._reject_fallback_if_strict("fewer than 10 factors")
            self.logger.warning(
                f"[层次聚类] 因子数不足10个({n_factors}), 跳过聚类, "
                f"降级到按{self.selection_metric}排序取Top{self.fallback_top_n}"
            )
            return self._fallback_simple_selection(X, y)
        
        try:
            # Step 0: 可配置的预筛选阶段（NEW）
            stage_start = time.time()
            
            if self.pre_filter_enabled and self.pre_filter_strategy_name:
                self.logger.info(
                    f"[预筛选-{self.pre_filter_strategy_name}] 开始执行可配置预筛选..."
                )
                qualified_factors = self._execute_pre_filter(
                    X, y, 
                    self.pre_filter_strategy_name, 
                    self.pre_filter_config
                )
                
                # 如果预筛选失败或结果为空，回退到阈值过滤
                if not qualified_factors:
                    self._reject_fallback_if_strict("configured pre-filter returned empty")
                    self.logger.warning(
                        f"[预筛选-{self.pre_filter_strategy_name}] 无因子通过，"
                        f"回退到阈值过滤"
                    )
                    qualified_factors = self._prefilter_by_metric(X, y)
                else:
                    self.logger.info(
                        f"[预筛选-{self.pre_filter_strategy_name}] 完成 - "
                        f"{n_factors} -> {len(qualified_factors)} "
                        f"(降维比例={(1-len(qualified_factors)/n_factors)*100:.1f}%, "
                        f"耗时={time.time()-stage_start:.2f}s)"
                    )
            else:
                # 回退到默认阈值过滤
                self.logger.debug("[预筛选] 使用默认阈值过滤方式")
                qualified_factors = self._prefilter_by_metric(X, y)
            
            # 验证预筛选结果
            if len(qualified_factors) == 0:
                self._reject_fallback_if_strict("metric pre-filter returned empty")
                self.logger.warning(
                    f"[预筛选] 无因子通过阈值({self.min_metric_threshold}), "
                    f"降级到Top{self.fallback_top_n}"
                )
                return self._fallback_simple_selection(X, y)
            
            prefilter_count = len(qualified_factors)
            
            # 如果预筛选后因子数不足,直接返回
            if self.distance_threshold is None and prefilter_count <= self.n_clusters:
                self._reject_fallback_if_strict(
                    "candidate count does not exceed configured cluster count"
                )
                self.logger.info(
                    f"[预筛选] 剩余因子数({prefilter_count})≤目标簇数({self.n_clusters}), "
                    f"无需聚类,直接返回"
                )
                if len(qualified_factors) > self.max_factors:
                    metrics_df = self._calculate_metrics_for_factors(X, y, qualified_factors)
                    if metrics_df.empty:
                        return qualified_factors[:self.max_factors]
                    metric_col = self._get_metric_column(metrics_df)
                    return metrics_df.sort_values(
                        [metric_col, 'factor_name'], ascending=[False, True]
                    ).head(self.max_factors)['factor_name'].tolist()
                return qualified_factors
            
            # Step 1: 对预筛选后的因子计算距离矩阵
            clustering_start = time.time()
            self.logger.info(
                f"[层次聚类] 开始聚类阶段 - 输入={prefilter_count}因子"
            )
            
            X_qualified = X[qualified_factors]
            distance_matrix, factor_names = self._compute_correlation_distance(X_qualified)
            
            # Step 2: 执行层次聚类
            cluster_map = self._perform_clustering(distance_matrix, factor_names)
            
            # 日志: 簇大小分布
            self._log_cluster_stats(cluster_map)
            
            # Step 3: 簇内选择代表因子(不再做阈值过滤,因为已预筛选)
            selected_factors = self._select_representative_factors_no_threshold(
                cluster_map, X, y
            )
            
            # Step 4: 质量检查
            final_count = len(selected_factors)
            reduction_ratio = (1 - final_count / n_factors) * 100
            
            clustering_time = time.time() - clustering_start
            total_time = time.time() - stage_start
            
            self.logger.info(
                f"[层次聚类] 聚类阶段完成 - 耗时={clustering_time:.2f}s"
            )
            self.logger.info(
                f"[层次聚类] 全流程完成 - 输出因子数={final_count}, "
                f"降维比例={reduction_ratio:.1f}% ({n_factors}→{final_count}), "
                f"总耗时={total_time:.2f}s"
            )
            
            # 如果筛选结果过少,保底处理
            if final_count == 0:
                self._reject_fallback_if_strict("clustering selected no representatives")
                self.logger.warning(
                    f"[层次聚类] 无因子通过筛选,降级到Top{self.fallback_top_n}"
                )
                return self._fallback_simple_selection(X, y)
            
            return selected_factors
            
        except LabelIsolationError:
            raise
        except Exception as e:
            if self.strict_no_fallback:
                if isinstance(e, ValueError) and str(e).startswith(
                    "strict nested clustering forbids fallback:"
                ):
                    raise
                raise ValueError(
                    f"strict nested clustering forbids fallback: clustering failed: {e}"
                ) from e
            self.logger.error(f"[层次聚类] 执行失败: {e}, 降级到简单筛选")
            import traceback
            self.logger.debug(traceback.format_exc())
            return self._fallback_simple_selection(X, y)
    
    
    def _prefilter_by_metric(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> List[str]:
        """预筛选: 移除低质量因子
        
        在聚类前先计算所有因子的metric,过滤掉低于阈值的因子。
        这样可以避免低质量因子参与聚类并影响簇划分。
        
        Args:
            X: 因子数据
            y: 前瞻收益率
            
        Returns:
            通过阈值的因子列表
        """
        all_factors = list(X.columns)
        
        self.logger.info(
            f"[预筛选] 开始计算所有因子的{self.selection_metric}指标..."
        )
        
        metrics_df = self._calculate_metrics_for_factors(X, y, all_factors)
        
        if metrics_df.empty:
            self.logger.warning("[预筛选] 指标计算失败,返回空列表")
            return []
        
        # 获取排序用的指标列
        metric_col = self._get_metric_column(metrics_df)
        
        # 阈值过滤
        mask = metrics_df[metric_col] >= self.min_metric_threshold
        qualified_factors = metrics_df[mask]['factor_name'].tolist()
        
        # 统计信息
        total = len(metrics_df)
        passed = len(qualified_factors)
        metric_series = metrics_df[metric_col].astype(float)
        
        self.logger.info(
            f"[预筛选] {metric_col}统计: "
            f"min={metric_series.min():.4f}, max={metric_series.max():.4f}, "
            f"mean={metric_series.mean():.4f}, median={metric_series.median():.4f}"
        )
        self.logger.info(
            f"[预筛选] 阈值={self.min_metric_threshold}, "
            f"通过={passed}/{total} ({passed/total*100:.1f}%)"
        )
        
        # 记录Top10因子(用于诊断)
        top_factors = metrics_df.nlargest(10, metric_col)
        self.logger.debug("[预筛选] Top10因子:")
        for idx, row in top_factors.iterrows():
            self.logger.debug(
                f"  {row['factor_name']}: {metric_col}={row[metric_col]:.4f}"
            )
        
        return qualified_factors
    
    def _compute_correlation_distance(
        self, 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """计算因子间距离矩阵（确定性版本）
        
        距离定义: distance = 1 - |correlation|
        
        使用稳定列序确保相关性计算的可重复性；不修改全局随机状态。
        
        Args:
            X: 因子数据
            
        Returns:
            (distance_matrix, factor_names): 距离矩阵和对应的因子名列表
        """
        # [FIX] 按列名排序确保列顺序稳定
        factor_names = sorted(X.columns.tolist())
        X_sorted = X[factor_names].copy()
        
        if self.bypass_global_cache:
            cached_factors = factor_names
            corr_matrix = X_sorted.corr(method=self.correlation_method).abs().values
        else:
            cache = get_global_cache()
            corr_matrix, cached_factors = cache.get_correlation_matrix(
                X_sorted, cols=factor_names, method=self.correlation_method
            )
        
        if corr_matrix.size == 0:
            raise ValueError("相关性矩阵计算失败")
        
        # [FIX] 四舍五入到较低精度（消除浮点数误差累积）
        corr_matrix = np.round(corr_matrix, decimals=8)
        
        # 转换为距离: 1 - |corr|
        # 绝对值确保负相关也被视为相似(都是强信号)
        distance_matrix = 1.0 - np.abs(corr_matrix)
        
        # 确保对角线为0(自己与自己距离为0)
        np.fill_diagonal(distance_matrix, 0.0)
        
        # 确保距离非负且对称
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        
        # [FIX] 最终四舍五入确保稳定性
        distance_matrix = np.round(distance_matrix, decimals=8)
        
        self.logger.debug(
            f"[距离矩阵-稳定版] 形状={distance_matrix.shape}, "
            f"范围=[{distance_matrix.min():.3f}, {distance_matrix.max():.3f}]"
        )
        
        return distance_matrix, cached_factors
    
    def _perform_clustering(
        self, 
        distance_matrix: np.ndarray, 
        factor_names: List[str]
    ) -> Dict[int, List[str]]:
        """执行层次聚类
        
        Args:
            distance_matrix: 距离矩阵(方阵)
            factor_names: 因子名列表
            
        Returns:
            簇映射字典 {cluster_id: [factor_names]}
        """
        # 转换为压缩距离向量(scipy要求的格式)
        condensed_distance = squareform(distance_matrix, checks=False)
        
        # 执行层次聚类
        self.logger.debug(f"[聚类] 使用{self.linkage_method}链接法进行聚类...")
        linkage_matrix = linkage(condensed_distance, method=self.linkage_method)
        
        # 切分簇
        if self.distance_threshold is not None:
            # 按距离阈值切分
            labels = fcluster(linkage_matrix, t=self.distance_threshold, criterion='distance')
            self.logger.info(f"[聚类] 按距离阈值={self.distance_threshold}切分")
        else:
            # 按簇数切分
            labels = fcluster(linkage_matrix, t=self.n_clusters, criterion='maxclust')
            self.logger.info(f"[聚类] 按簇数={self.n_clusters}切分")
        
        # 构建簇映射
        cluster_map = defaultdict(list)
        for factor_name, cluster_id in zip(factor_names, labels):
            cluster_map[int(cluster_id)].append(factor_name)
        
        return dict(cluster_map)
    
    def _log_cluster_stats(self, cluster_map: Dict[int, List[str]]) -> None:
        """记录簇统计信息"""
        n_clusters = len(cluster_map)
        cluster_sizes = [len(factors) for factors in cluster_map.values()]
        
        self.logger.info(
            f"[簇统计] 总簇数={n_clusters}, "
            f"簇大小: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
            f"avg={np.mean(cluster_sizes):.1f}, median={np.median(cluster_sizes):.0f}"
        )
        
        # 详细簇分布(DEBUG级别)
        for cluster_id, factors in sorted(cluster_map.items()):
            self.logger.debug(
                f"  簇{cluster_id}: {len(factors)}个因子 - {factors[:3]}..."
                if len(factors) > 3 else 
                f"  簇{cluster_id}: {len(factors)}个因子 - {factors}"
            )
    
    def _select_representative_factors_no_threshold(
        self, 
        cluster_map: Dict[int, List[str]],
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[str]:
        """从每个簇中选择代表因子(不做阈值过滤)
        
        因为已经在预筛选阶段过滤了低质量因子,所以这里不再做阈值检查,
        每个簇直接选择metric最优的1个因子。
        
        Args:
            cluster_map: 簇映射
            X: 因子数据
            y: 前瞻收益率
            
        Returns:
            代表因子列表
        """
        representatives = []
        cluster_contributions = []
        
        for cluster_id, factors_in_cluster in sorted(cluster_map.items()):
            # 计算簇内因子的指标
            try:
                metrics_df = self._calculate_metrics_for_factors(X, y, factors_in_cluster)
                
                if metrics_df.empty:
                    self.logger.warning(f"[簇{cluster_id}] 指标计算失败,跳过")
                    continue
                
                # [FIX] 按选择指标排序，使用因子名作为二级排序键确保稳定性
                metric_col = self._get_metric_column(metrics_df)
                metrics_df = metrics_df.sort_values(
                    [metric_col, 'factor_name'],
                    ascending=[False, True]  # 指标降序，因子名升序（字典序）
                )
                
                # 选择Top1(不做阈值检查)
                best_factor = metrics_df.iloc[0]
                best_value = best_factor[metric_col]
                
                representatives.append(best_factor['factor_name'])
                cluster_contributions.append({
                    'cluster_id': cluster_id,
                    'factor': best_factor['factor_name'],
                    'metric': best_value,
                    'cluster_size': len(factors_in_cluster)
                })
                
                self.logger.debug(
                    f"[簇{cluster_id}] 选择 {best_factor['factor_name']} "
                    f"({self.selection_metric}={best_value:.4f}, "
                    f"簇大小={len(factors_in_cluster)})"
                )
                    
            except LabelIsolationError:
                raise
            except Exception as e:
                if self.strict_no_fallback:
                    raise
                self.logger.warning(f"[簇{cluster_id}] 代表因子选择失败: {e}")
                continue
        
        # 限制最大输出数量（distance_threshold 模式下不截断，尊重聚类自然结果）
        if self.distance_threshold is not None:
            self.logger.info(
                f"[簇代表] distance_threshold模式,不截断,保留全部{len(representatives)}个因子"
            )
        elif len(representatives) > self.max_factors:
            self.logger.info(
                f"[簇代表] 选出{len(representatives)}个因子,截断到{self.max_factors}个"
            )
            # [FIX] 按指标排序,使用因子名作为二级排序键确保稳定性
            sorted_contribs = sorted(
                cluster_contributions, 
                key=lambda x: (-x['metric'], x['factor']),
            )
            representatives = [c['factor'] for c in sorted_contribs[:self.max_factors]]
        
        # 日志: 簇贡献统计
        if cluster_contributions:
            self._log_cluster_contributions(cluster_contributions)
        
        return representatives
    
    def _calculate_metrics_for_factors(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        factors: List[str]
    ) -> pd.DataFrame:
        """计算指定因子的dCor/IC/IR指标
        
        优先计算dCor,失败则降级到IC/IR
        
        Args:
            X: 因子数据
            y: 前瞻收益率
            factors: 待计算的因子列表
            
        Returns:
            包含指标的DataFrame: columns=[factor_name, dcor, ic, ir, icir]
        """
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index, factors]
        y_aligned = y.loc[common_index]
        
        if X_aligned.empty:
            return pd.DataFrame()
        
        self.logger.debug(
            f"[指标计算] 对齐后样本数={len(X_aligned)}, 因子数={len(factors)}"
        )
        
        results = []
        
        for factor in factors:
            try:
                valid = X_aligned[factor].notna() & y_aligned.notna()
                x_values = X_aligned.loc[valid, factor]
                y_values = y_aligned.loc[valid]
                
                # 至少需要10个样本
                if len(x_values) < 10:
                    continue
                
                # 初始化指标
                metrics = {'factor_name': factor}
                
                # 计算dCor(如果需要)
                if self.selection_metric in ['dcor', 'icir']:
                    try:
                        if self.bypass_global_cache:
                            dcor_val = _compute_dcor_value(
                                x_values.to_numpy(), y_values.to_numpy()
                            )
                        else:
                            cache = get_global_cache()
                            dcor_val = cache.get_dcor(
                                x_values, y_values,
                                compute_fn=_compute_dcor_value,
                                raise_on_error=self.strict_no_fallback
                            )
                        if dcor_val is None:
                            continue
                        metrics['dcor'] = dcor_val
                    except LabelIsolationError:
                        raise
                    except Exception:
                        if self.strict_no_fallback:
                            raise
                        metrics['dcor'] = 0.0
                
                # 计算IC/IR
                if self.selection_metric in ['ic', 'ir', 'icir']:
                    try:
                        ic_values = self._calculate_daily_ic(X_aligned[[factor]], y_aligned)
                        if ic_values.empty:
                            continue
                        ic_mean = ic_values.mean()
                        ic_std = ic_values.std()
                        ir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
                        
                        metrics['ic'] = abs(ic_mean)
                        metrics['ir'] = abs(ir)
                        metrics['icir'] = abs(ic_mean) * abs(ir)  # 综合指标
                    except LabelIsolationError:
                        raise
                    except Exception:
                        if self.strict_no_fallback:
                            raise
                        metrics['ic'] = 0.0
                        metrics['ir'] = 0.0
                        metrics['icir'] = 0.0
                
                results.append(metrics)
                
            except LabelIsolationError:
                raise
            except Exception as e:
                if self.strict_no_fallback:
                    raise
                self.logger.debug(f"[指标计算] 因子{factor}失败: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # 诊断日志: dCor统计
        if self.selection_metric in ['dcor', 'icir'] and 'dcor' in results_df.columns:
            dcor_series = results_df['dcor'].astype(float)
            self.logger.debug(
                f"[指标计算] dCor统计: count={len(dcor_series)}, "
                f"min={dcor_series.min():.4f}, max={dcor_series.max():.4f}, "
                f"mean={dcor_series.mean():.4f}"
            )
            if (dcor_series <= 0).all():
                self.logger.warning("[指标计算] dCor全部为0或无效,可能存在对齐或计算问题")
        
        # 诊断日志: IC/IR统计
        if self.selection_metric in ['ic', 'ir', 'icir'] and 'ic' in results_df.columns:
            ic_series = results_df['ic'].astype(float)
            ir_series = results_df['ir'].astype(float) if 'ir' in results_df.columns else None
            self.logger.debug(
                f"[指标计算] IC统计: count={len(ic_series)}, "
                f"min={ic_series.min():.4f}, max={ic_series.max():.4f}, "
                f"mean={ic_series.mean():.4f}"
            )
            if ir_series is not None:
                self.logger.debug(
                    f"[指标计算] IR统计: count={len(ir_series)}, "
                    f"min={ir_series.min():.4f}, max={ir_series.max():.4f}, "
                    f"mean={ir_series.mean():.4f}"
                )
        
        return results_df
    
    def _calculate_daily_ic(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> pd.Series:
        """计算每日Rank IC(Spearman相关系数)
        
        复用ic_ir_strategy的逻辑
        
        Args:
            X: 单因子数据(单列DataFrame)
            y: 前瞻收益率
            
        Returns:
            每日IC序列
        """
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X的索引必须是MultiIndex(date, stock_code)")
        
        dates = X.index.get_level_values('date').unique()
        ic_list = []
        
        for date in dates:
            try:
                x_date = X.loc[date].iloc[:, 0]  # 取第一列
                y_date = y.loc[date]
                
                # 对齐索引
                common_stocks = x_date.index.intersection(y_date.index)
                if len(common_stocks) < 10:
                    continue
                
                x_values = x_date.loc[common_stocks]
                y_values = y_date.loc[common_stocks]
                valid = x_values.notna() & y_values.notna()
                x_values = x_values.loc[valid]
                y_values = y_values.loc[valid]
                if len(x_values) < 10 or x_values.nunique() <= 1 or y_values.nunique() <= 1:
                    continue
                
                # 计算Spearman相关性
                ic = x_values.corr(y_values, method='spearman')
                
                if not np.isnan(ic):
                    ic_list.append(ic)
                    
            except LabelIsolationError:
                raise
            except Exception:
                if self.strict_no_fallback:
                    raise
                continue
        
        return pd.Series(ic_list, dtype=float)
    
    def _get_metric_column(self, metrics_df: pd.DataFrame) -> str:
        """获取排序用的指标列名"""
        metric_map = {
            'dcor': 'dcor',
            'ic': 'ic',
            'ir': 'ir',
            'icir': 'icir'
        }
        
        col = metric_map.get(self.selection_metric, 'dcor')
        
        if col not in metrics_df.columns:
            if self.strict_no_fallback:
                raise ValueError(f"strict nested clustering requires metric column: {col}")
            # 降级: dcor -> ic -> ir -> icir
            fallback_order = ['dcor', 'ic', 'ir', 'icir']
            for fb_col in fallback_order:
                if fb_col in metrics_df.columns:
                    self.logger.warning(
                        f"[指标选择] {col}列不存在,降级到{fb_col}"
                    )
                    return fb_col
            raise ValueError(f"无可用指标列: {metrics_df.columns.tolist()}")
        
        return col
    
    def _log_cluster_contributions(
        self, 
        contributions: List[Dict[str, Any]]
    ) -> None:
        """记录簇贡献统计"""
        total = len(contributions)
        avg_metric = np.mean([c['metric'] for c in contributions])
        
        self.logger.info(
            f"[簇贡献] 总代表因子={total}, "
            f"平均{self.selection_metric}={avg_metric:.4f}"
        )
        
        # Top5簇(按指标排序)
        top_contribs = sorted(contributions, key=lambda x: x['metric'], reverse=True)[:5]
        for rank, c in enumerate(top_contribs, 1):
            self.logger.debug(
                f"  Top{rank}: 簇{c['cluster_id']} - {c['factor']} "
                f"({self.selection_metric}={c['metric']:.4f}, size={c['cluster_size']})"
            )
    
    def _fallback_simple_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> List[str]:
        """降级处理: 按指标简单排序取TopK
        
        Args:
            X: 因子数据
            y: 前瞻收益率
            
        Returns:
            TopK因子列表
        """
        limit = (self.fallback_top_n if self.distance_threshold is not None
                 else min(self.fallback_top_n, self.max_factors))
        try:
            all_factors = list(X.columns)
            metrics_df = self._calculate_metrics_for_factors(X, y, all_factors)
            
            if metrics_df.empty:
                # 完全失败,返回前N个因子
                self.logger.warning("[降级] 指标计算失败,返回前N个因子")
                return all_factors[:limit]
            
            # [FIX] 按指标排序，使用因子名作为二级排序键确保稳定性
            metric_col = self._get_metric_column(metrics_df)
            metrics_df = metrics_df.sort_values(
                [metric_col, 'factor_name'],
                ascending=[False, True]
            )
            
            # 取TopK
            top_factors = metrics_df.head(limit)['factor_name'].tolist()
            
            self.logger.info(
                f"[降级] 按{self.selection_metric}排序,返回Top{len(top_factors)}因子"
            )
            
            return top_factors
            
        except LabelIsolationError:
            raise
        except Exception as e:
            self.logger.error(f"[降级] 简单筛选失败: {e}")
            # 最终保底: 返回前N个因子
            return list(X.columns)[:limit]
    
    def _get_pre_filter_config(self) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """解析预筛选配置
        
        参考 BarraGroupedPreSelector._get_group_strategy_config 的设计，
        解析 pre_filter 配置块并转换为 PreSelectionStrategy 标准格式。
        
        Returns:
            (enabled, strategy_name, strategy_config)
            - enabled: 是否启用预筛选
            - strategy_name: 预筛选策略名称（如'stability_selection'）
            - strategy_config: 策略配置字典（符合PreSelectionStrategy接口）
        """
        pre_filter_cfg = self.config.get('pre_filter', {})
        
        # 配置格式验证
        if not isinstance(pre_filter_cfg, dict):
            return False, None, {}
        
        # 检查是否启用
        enabled = pre_filter_cfg.get('enabled', False)
        if not enabled:
            return False, None, {}
        
        # 获取策略名称（默认使用 stability_selection）
        strategy_name = pre_filter_cfg.get('strategy', 'stability_selection')
        
        # 构建策略配置（转换为 PreSelectionStrategy 标准格式）
        strategy_config = {
            'enabled': True,
            'metrics': {},
            'fallback_top_n': pre_filter_cfg.get('fallback_top_n', self.fallback_top_n)
        }
        
        # 根据策略类型设置对应的配置参数
        if strategy_name == 'stability_selection':
            # Stability Selection 特有配置
            for key in ['base_metric', 'n_bootstrap', 'subsample_ratio', 
                       'selection_threshold', 'selection_mode', 'per_sample_top_k',
                       'per_sample_top_pct', 'per_sample_metric_threshold', 'random_seed',
                       'sampling_mode']:
                if key in pre_filter_cfg:
                    strategy_config[key] = pre_filter_cfg[key]
            
            # 相关性过滤配置
            if 'max_correlation' in pre_filter_cfg:
                strategy_config['metrics']['max_correlation'] = pre_filter_cfg['max_correlation']
            
            # 并行配置
            if 'parallel' in pre_filter_cfg:
                strategy_config['parallel'] = pre_filter_cfg['parallel']
                
        elif strategy_name == 'dcor_cv':
            # dCor CV 模式配置
            for key in ['threshold_search_space', 'cv_folds', 'cv_metric',
                       'cv_val_ratio', 'cv_initial_train_ratio', 'cv_holdout_ratio',
                       'min_abs_metric']:
                if key in pre_filter_cfg:
                    strategy_config[key] = pre_filter_cfg[key]
            
            if 'max_correlation' in pre_filter_cfg:
                strategy_config['metrics']['max_correlation'] = pre_filter_cfg['max_correlation']
                
        elif strategy_name == 'dcor':
            # dCor 固定阈值模式
            if 'min_abs_metric' in pre_filter_cfg:
                strategy_config['metrics']['min_abs_dcor'] = pre_filter_cfg['min_abs_metric']
            if 'max_correlation' in pre_filter_cfg:
                strategy_config['metrics']['max_correlation'] = pre_filter_cfg['max_correlation']
                
        elif strategy_name == 'ic_ir':
            # IC/IR 阈值模式
            if 'min_abs_ic' in pre_filter_cfg:
                strategy_config['metrics']['min_abs_ic'] = pre_filter_cfg['min_abs_ic']
            if 'min_abs_ir' in pre_filter_cfg:
                strategy_config['metrics']['min_abs_ir'] = pre_filter_cfg['min_abs_ir']
            if 'min_abs_metric' in pre_filter_cfg:
                # 兼容：min_abs_metric 作为 min_abs_ic 的别名
                strategy_config['metrics']['min_abs_ic'] = pre_filter_cfg['min_abs_metric']
            if 'max_correlation' in pre_filter_cfg:
                strategy_config['metrics']['max_correlation'] = pre_filter_cfg['max_correlation']
        else:
            # 未知策略，传递所有非标准字段
            for key, value in pre_filter_cfg.items():
                if key not in ['enabled', 'strategy', 'fallback_top_n']:
                    strategy_config[key] = value
        
        return enabled, strategy_name, strategy_config
    
    def _execute_pre_filter(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        strategy_name: str, 
        strategy_config: Dict[str, Any]
    ) -> List[str]:
        """执行可配置的预筛选策略
        
        参考 BarraGroupedPreSelector._select_within_group 的实现，
        使用工厂函数动态创建策略实例并执行筛选。
        
        Args:
            X: 因子数据
            y: 前瞻收益率
            strategy_name: 策略名称（如 'stability_selection'）
            strategy_config: 策略配置字典
            
        Returns:
            预筛选后的因子列表
            
        Raises:
            strict 模式传播原始异常；普通模式捕获异常并返回空列表。
        """
        import time
        start_time = time.time()
        
        try:
            # 使用工厂函数创建策略实例
            strategy = get_preselection_strategy(
                strategy_name, 
                strategy_config, 
                self.logger
            )
            
            self.logger.debug(
                f"[预筛选-{strategy_name}] 策略实例创建成功，开始执行筛选..."
            )
            
            # 执行筛选
            qualified_factors = strategy.select(X, y)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info(
                f"[预筛选-{strategy_name}] 完成 - "
                f"耗时={elapsed_time:.2f}s, 筛选结果={len(qualified_factors)}/{len(X.columns)}"
            )
            
            # 获取性能报告（用于调试）
            perf_report = strategy.get_performance_report()
            if not perf_report.empty and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[预筛选-{strategy_name}] 性能报告前5因子:\n"
                    f"{perf_report.head().to_string()}"
                )
            
            return qualified_factors
            
        except ValueError as e:
            if self.strict_no_fallback:
                raise
            # 策略名称无效或配置错误
            self.logger.warning(
                f"[预筛选-{strategy_name}] 策略配置错误: {e}，"
                f"回退到默认阈值过滤"
            )
            return []
            
        except LabelIsolationError:
            raise
        except Exception as e:
            if self.strict_no_fallback:
                raise
            # 策略执行失败
            self.logger.error(
                f"[预筛选-{strategy_name}] 执行失败: {e}，"
                f"回退到默认阈值过滤"
            )
            import traceback
            self.logger.debug(traceback.format_exc())
            return []
