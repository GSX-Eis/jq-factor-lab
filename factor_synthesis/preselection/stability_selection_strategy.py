# -*- coding: utf-8 -*-
"""
Stability Selection 预筛选策略

基于 Meinshausen & Bühlmann (2010) 的经典论文实现。
通过多次子采样统计每个因子被选中的频率，保留选择概率高于阈值的因子。

核心优势:
- 不依赖单次 CV 的噪声估计
- 选出的因子是稳定的，而非偶然高分的
- 有理论保证：即使原方法不一致，Stability Selection 也能一致

支持的评估指标 (base_metric):
- "dcor": Distance Correlation，捕捉非线性关系
- "ic": Spearman IC (Information Coefficient)，衡量排序相关性
- "ir": Information Ratio = IC均值 / IC标准差，衡量IC稳定性

Reference:
    Meinshausen, N., & Bühlmann, P. (2010). Stability selection.
    Journal of the Royal Statistical Society: Series B, 72(4), 417-473.
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from typing import List, Dict, Any, Optional
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
    y_values: np.ndarray,
    max_samples: int = 5000,
    random_seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    计算单个因子的 dCor (供并行调用)
    
    Performance Optimization (2026-02-04):
    - When n_samples > max_samples, randomly sample to reduce O(n²) complexity
    - This provides 3-5x speedup with minimal accuracy loss
    
    Args:
        col: 因子名称
        factor_values: 因子值数组
        y_values: 收益率数组
        max_samples: 最大样本数，超过时进行采样 (0 = 不采样)
        random_seed: 随机种子（用于采样的可重复性）
        
    Returns:
        包含 factor_name, metric_value 的字典；样本不足返回 None，计算错误向上传播
    """
    try:
        # 处理 NaN
        valid_mask = ~(np.isnan(factor_values) | np.isnan(y_values))
        factor_clean = factor_values[valid_mask]
        y_clean = y_values[valid_mask]
        
        n_valid = len(factor_clean)
        if n_valid < 10:
            return None
        
        # === Sampling optimization for large datasets ===
        if max_samples > 0 and n_valid > max_samples:
            rng = np.random.default_rng(random_seed)
            sample_idx = rng.choice(n_valid, size=max_samples, replace=False)
            factor_clean = factor_clean[sample_idx]
            y_clean = y_clean[sample_idx]
        
        dcor_value = dcor.distance_correlation(
            factor_clean, 
            y_clean, 
            method='AVL'
        )
        
        return {
            'factor_name': col,
            'metric_value': dcor_value,
            'metric_type': 'dcor'
        }
    except LabelIsolationError:
        raise
    except Exception:
        raise


def _compute_single_ic(
    col: str,
    factor_df: pd.DataFrame,
    y_series: pd.Series,
    date_level: int = 0
) -> Optional[Dict[str, Any]]:
    """
    计算单个因子的 IC (Spearman 相关系数均值)
    
    对每个日期计算截面 IC，然后取均值的绝对值。
    
    Args:
        col: 因子名称
        factor_df: 单个因子的 DataFrame (MultiIndex: date, stock)
        y_series: 收益率 Series (MultiIndex: date, stock)
        date_level: 日期所在的索引层级
        
    Returns:
        包含 factor_name, metric_value 的字典；样本不足返回 None，计算错误向上传播
    """
    try:
        # 对齐索引
        common_idx = factor_df.index.intersection(y_series.index)
        if len(common_idx) < 10:
            return None
        
        factor_aligned = factor_df.loc[common_idx]
        y_aligned = y_series.loc[common_idx]
        
        # 获取所有日期
        dates = factor_aligned.index.get_level_values(date_level).unique()
        
        if len(dates) < 3:
            return None
        
        daily_ics = []
        for date in dates:
            try:
                f_day = factor_aligned.xs(date, level=date_level)
                y_day = y_aligned.xs(date, level=date_level)
                
                # 处理 NaN
                valid_mask = ~(f_day.isna() | y_day.isna())
                if valid_mask.sum() < 10:
                    continue
                if f_day[valid_mask].nunique() <= 1 or y_day[valid_mask].nunique() <= 1:
                    continue
                
                ic, _ = spearmanr(f_day[valid_mask], y_day[valid_mask])
                
                if not np.isnan(ic):
                    daily_ics.append(ic)
            except LabelIsolationError:
                raise
            except Exception:
                raise
        
        if len(daily_ics) < 3:
            return None
        
        # 返回 IC 均值的绝对值
        mean_ic = np.abs(np.mean(daily_ics))
        
        return {
            'factor_name': col,
            'metric_value': mean_ic,
            'metric_type': 'ic',
            'n_days': len(daily_ics),
            'daily_ics': daily_ics  # 用于后续计算 IR
        }
    except LabelIsolationError:
        raise
    except Exception:
        raise


def _compute_single_ir(
    col: str,
    factor_df: pd.DataFrame,
    y_series: pd.Series,
    date_level: int = 0
) -> Optional[Dict[str, Any]]:
    """
    计算单个因子的 IR (Information Ratio = IC均值 / IC标准差)
    
    Args:
        col: 因子名称
        factor_df: 单个因子的 DataFrame (MultiIndex: date, stock)
        y_series: 收益率 Series (MultiIndex: date, stock)
        date_level: 日期所在的索引层级
        
    Returns:
        包含 factor_name, metric_value 的字典；样本不足返回 None，计算错误向上传播
    """
    try:
        # 先计算 IC
        ic_result = _compute_single_ic(col, factor_df, y_series, date_level)
        
        if ic_result is None or 'daily_ics' not in ic_result:
            return None
        
        daily_ics = ic_result['daily_ics']
        
        if len(daily_ics) < 5:  # IR 需要足够的样本
            return None
        
        ic_mean = np.mean(daily_ics)
        ic_std = np.std(daily_ics, ddof=1)
        
        if ic_std < 1e-8:
            # 标准差太小，返回 |IC均值| 作为 IR
            ir = np.abs(ic_mean)
        else:
            ir = np.abs(ic_mean) / ic_std
        
        return {
            'factor_name': col,
            'metric_value': ir,
            'metric_type': 'ir',
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'n_days': len(daily_ics)
        }
    except LabelIsolationError:
        raise
    except Exception:
        raise


def _compute_ic_vectorized(
    X: pd.DataFrame,
    y: pd.Series,
    date_level: str = 'date',
    min_stocks_per_day: int = 10,
    min_days: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    向量化计算所有因子的 IC (Performance Optimization)
    
    使用 pandas groupby + rank 替代逐因子、逐日期的 Python 循环。
    Speedup: 2-5x compared to _compute_single_ic in a loop.
    
    Algorithm:
    1. Cross-sectionally rank factors and returns per date
    2. Compute Pearson correlation between ranks (equivalent to Spearman)
    3. Aggregate daily ICs into mean/std
    
    Args:
        X: 因子数据 DataFrame (MultiIndex: date, stock_code)
        y: 收益率 Series (MultiIndex: date, stock_code)
        date_level: 日期索引名称
        min_stocks_per_day: 每日最少股票数
        min_days: 最少有效天数
        
    Returns:
        {factor_name: {metric_value, metric_type, ic_mean, ic_std, n_days, daily_ics}, ...}
    """
    # Align indices
    common_index = X.index.intersection(y.index)
    if len(common_index) == 0:
        return {}
    
    X_aligned = X.loc[common_index]
    y_aligned = y.loc[common_index]
    
    # A sparse factor must not remove dates from other factors.
    daily_counts = y_aligned.groupby(level=date_level).count()
    valid_dates = daily_counts[daily_counts >= min_stocks_per_day].index
    
    if len(valid_dates) < min_days:
        return {}
    
    # Filter to valid dates
    X_valid = X_aligned.loc[X_aligned.index.get_level_values(date_level).isin(valid_dates)]
    y_valid = y_aligned.loc[y_aligned.index.get_level_values(date_level).isin(valid_dates)]
    
    # Rank only after choosing each factor's complete pairs.
    data = X_valid.copy()
    data['__y_rank__'] = y_valid
    
    # === Vectorized daily IC computation ===
    def _daily_pearson_with_y(group):
        """Rank complete pairs; keep the complete-panel vectorized arithmetic."""
        y_col = group['__y_rank__']
        factor_cols = [c for c in group.columns if c != '__y_rank__']
        if not group.isna().any().any():
            values = group[factor_cols].rank(pct=True).corrwith(y_col.rank(pct=True))
            invalid = group[factor_cols].nunique() <= 1
            if y_col.nunique() <= 1:
                values[:] = np.nan
            else:
                values.loc[invalid] = np.nan
            return values
        values = {}
        for col in factor_cols:
            pairs = group[[col, '__y_rank__']].dropna()
            if (len(pairs) < min_stocks_per_day or pairs[col].nunique() <= 1
                    or pairs['__y_rank__'].nunique() <= 1):
                values[col] = np.nan
            else:
                values[col] = pairs[col].rank(pct=True).corr(pairs['__y_rank__'].rank(pct=True))
        return pd.Series(values, dtype=float)
    
    # Compute daily ICs for all factors at once
    daily_ics_df = data.groupby(level=date_level).apply(_daily_pearson_with_y)
    
    # Build results
    results = {}
    for col in X.columns:
        if col not in daily_ics_df.columns:
            continue
        
        ics = daily_ics_df[col].dropna()
        
        if len(ics) < min_days:
            continue
        
        ic_mean = float(ics.mean())
        ic_std = float(ics.std())
        abs_ic_mean = abs(ic_mean)
        
        results[col] = {
            'factor_name': col,
            'metric_value': abs_ic_mean,  # |IC| mean
            'metric_type': 'ic',
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'n_days': len(ics),
            'daily_ics': ics.tolist(),
        }
    
    return results


def _compute_ir_from_ic_results(ic_results: Dict[str, Dict[str, Any]], min_days: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    从 IC 结果计算 IR
    
    Args:
        ic_results: _compute_ic_vectorized 的返回结果
        min_days: 计算 IR 所需的最少天数
        
    Returns:
        {factor_name: {metric_value, metric_type, ic_mean, ic_std, n_days}, ...}
    """
    results = {}
    
    for col, ic_data in ic_results.items():
        n_days = ic_data.get('n_days', 0)
        if n_days < min_days:
            continue
        
        ic_mean = ic_data['ic_mean']
        ic_std = ic_data['ic_std']
        
        if ic_std < 1e-8:
            ir = abs(ic_mean)
        else:
            ir = abs(ic_mean) / ic_std
        
        results[col] = {
            'factor_name': col,
            'metric_value': ir,
            'metric_type': 'ir',
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'n_days': n_days,
        }
    
    return results


@register_strategy
class StabilitySelectionPreStrategy(PreSelectionStrategy):
    """基于 Stability Selection 的因子预筛选策略
    
    通过多次子采样，统计每个因子被选中的频率，
    保留选择概率 >= selection_threshold 的因子。
    
    **工作流程**:
    1. 对训练数据进行 N 次子采样（默认 30 次，每次采样 50% 数据）
    2. 每次子采样后，计算指定指标（dCor/IC/IR）并选择 top_k 因子
    3. 统计每个因子被选中的次数 / N = 选择概率
    4. 保留选择概率 >= selection_threshold 的因子
    5. 可选：对选中因子进行相关性筛选
    
    **无 Look-Ahead Bias 保证**:
    整个 Stability Selection 过程都在训练窗口内进行，
    子采样是对训练窗口数据的采样，不涉及任何外部数据。
    
    配置示例:
    ```json
    {
        "pre_selection": {
            "enabled": true,
            "strategy": "stability_selection",
            "base_metric": "dcor",
            "n_bootstrap": 30,
            "subsample_ratio": 0.5,
            "selection_threshold": 0.6,
            "selection_mode": "top_pct",
            "per_sample_top_pct": 0.3,
            "per_sample_top_k": 5,
            "per_sample_metric_threshold": 0.05,
            "max_correlation": 0.5,
            "fallback_top_n": 3,
            "parallel": {
                "enabled": true,
                "n_jobs": -1,
                "backend": "loky"
            }
        }
    }
    ```
    
    支持的 base_metric:
    - "dcor": Distance Correlation，捕捉非线性关系 (默认)
    - "ic": Spearman IC 绝对值均值，衡量排序相关性
    - "ir": Information Ratio = |IC均值| / IC标准差，衡量IC稳定性
    
    支持的 selection_mode:
    - "top_k": 每次子采样选择指标最大的 top_k 个因子
    - "top_pct": 每次子采样选择指标最大的 top X% 因子 (默认，推荐)
    - "threshold": 每次子采样选择指标 >= per_sample_metric_threshold 的因子
    """
    
    name = "stability_selection"
    
    # 支持的评估指标
    SUPPORTED_METRICS = ('dcor', 'ic', 'ir')
    
    # 支持的选择模式
    SUPPORTED_SELECTION_MODES = ('top_k', 'top_pct', 'threshold')
    
    # 默认参数
    DEFAULT_N_BOOTSTRAP = 30  # 1年数据建议30次，平衡效率和稳定性
    DEFAULT_SUBSAMPLE_RATIO = 0.5
    DEFAULT_SELECTION_THRESHOLD = 0.6
    DEFAULT_PER_SAMPLE_TOP_K = 5
    DEFAULT_PER_SAMPLE_TOP_PCT = 0.3  # 每次采样选择 top 30% 因子
    DEFAULT_BASE_METRIC = 'dcor'
    DEFAULT_PER_SAMPLE_METRIC_THRESHOLD = 0.05
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        执行 Stability Selection 因子筛选
        
        Args:
            X: 因子数据 (MultiIndex: date, stock_code)
            y: 前瞻收益率 (MultiIndex: date, stock_code)
            
        Returns:
            保留的因子名称列表
        """
        self._performance_report = None
        self._selection_details = []
        self._pre_filter_metrics = {}
        self._pre_filtered_factors = None
        self._base_metric = None
        self._validate_inputs(X, y)
        
        # 检查是否启用
        if not self.config.get('enabled', True):
            self.logger.info("[Stability Selection] 未启用，返回所有因子")
            return list(X.columns)
        
        self.logger.debug("执行因子预筛选 (Stability Selection)...")
        
        # 获取配置
        base_metric = self.config.get('base_metric', self.DEFAULT_BASE_METRIC)
        n_bootstrap = self.config.get('n_bootstrap', self.DEFAULT_N_BOOTSTRAP)
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
        subsample_ratio = self.config.get('subsample_ratio', self.DEFAULT_SUBSAMPLE_RATIO)
        selection_threshold = self.config.get('selection_threshold', self.DEFAULT_SELECTION_THRESHOLD)
        selection_mode = self.config.get('selection_mode', 'top_pct')  # 默认使用百分比模式
        per_sample_top_k = self.config.get('per_sample_top_k', self.DEFAULT_PER_SAMPLE_TOP_K)
        per_sample_top_pct = self.config.get('per_sample_top_pct', self.DEFAULT_PER_SAMPLE_TOP_PCT)
        
        # 兼容旧配置: per_sample_dcor_threshold -> per_sample_metric_threshold
        per_sample_metric_threshold = self.config.get(
            'per_sample_metric_threshold',
            self.config.get('per_sample_dcor_threshold', self.DEFAULT_PER_SAMPLE_METRIC_THRESHOLD)
        )
        fallback_top_n = self.config.get('fallback_top_n', 3)
        
        # 验证 base_metric
        if base_metric not in self.SUPPORTED_METRICS:
            self.logger.warning(
                f"不支持的 base_metric '{base_metric}'，使用默认 'dcor'"
            )
            base_metric = 'dcor'
        
        # 验证 selection_mode
        if selection_mode not in self.SUPPORTED_SELECTION_MODES:
            self.logger.warning(
                f"不支持的 selection_mode '{selection_mode}'，使用默认 'top_pct'"
            )
            selection_mode = 'top_pct'
        
        # 索引对齐
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        if X_aligned.empty:
            self.logger.warning("数据为空，返回所有因子")
            return list(X.columns)
        
        initial_count = len(X.columns)
        all_factors = list(X.columns)
        
        # Step 1: 【新增】先进行相关性预过滤，减少冗余因子
        metrics = self.config.get('metrics', {})
        max_corr = metrics.get('max_correlation', self.config.get('max_correlation'))
        
        if max_corr is not None and len(all_factors) > 1:
            pre_filtered_factors = self._pre_filter_by_correlation(
                X_aligned, y_aligned, all_factors, max_corr, base_metric
            )
            self.logger.debug(
                f"[Stability Selection] 相关性预过滤: {len(all_factors)} -> {len(pre_filtered_factors)}"
            )
            # 只保留预过滤后的因子
            X_filtered = X_aligned[pre_filtered_factors]
        else:
            pre_filtered_factors = all_factors
            X_filtered = X_aligned
        
        # Step 2: 运行 Stability Selection（在预过滤后的因子集上）
        factor_counts, selection_details = self._run_stability_selection(
            X_filtered, y_aligned,
            base_metric=base_metric,
            n_bootstrap=n_bootstrap,
            subsample_ratio=subsample_ratio,
            selection_mode=selection_mode,
            per_sample_top_k=per_sample_top_k,
            per_sample_top_pct=per_sample_top_pct,
            per_sample_metric_threshold=per_sample_metric_threshold
        )
        
        # Step 3: 计算选择概率
        selection_probs = {
            f: count / n_bootstrap 
            for f, count in factor_counts.items()
        }
        
        # Step 4: 筛选选择概率 >= threshold 的因子
        selected_factors = [
            f for f, prob in selection_probs.items()
            if prob >= selection_threshold
        ]
        
        # 按选择概率降序排序
        selected_factors = sorted(
            selected_factors, 
            key=lambda f: selection_probs[f], 
            reverse=True
        )
        
        if selection_probs:
            self.logger.debug(
                f"[Stability Selection] 选择概率分布: "
                f"max={max(selection_probs.values()):.2f}, "
                f"min={min(selection_probs.values()):.2f}, "
                f"mean={np.mean(list(selection_probs.values())):.2f}"
            )
        
        final_count = len(selected_factors)
        
        # 构建性能报告（包含预过滤信息）
        self._build_performance_report(
            all_factors, selection_probs, selection_details, base_metric,
            pre_filtered_factors=pre_filtered_factors
        )
        
        # 单行汇总日志
        self.logger.info(
            f"[Stability Selection] {initial_count} -> {len(pre_filtered_factors)} (corr) -> {final_count} "
            f"(metric={base_metric}, n_bootstrap={n_bootstrap}, threshold={selection_threshold:.2f})"
        )
        
        # 安全保底
        if final_count == 0:
            self.logger.warning(
                f"Stability Selection 后无剩余因子！"
                f"将保留选择概率最高的 Top {fallback_top_n} 因子"
            )
            top_factors = sorted(
                selection_probs.keys(),
                key=lambda f: selection_probs[f],
                reverse=True
            )[:fallback_top_n]
            return top_factors
        
        return selected_factors
    
    def _run_stability_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_metric: str,
        n_bootstrap: int,
        subsample_ratio: float,
        selection_mode: str,
        per_sample_top_k: int,
        per_sample_top_pct: float,
        per_sample_metric_threshold: float
    ) -> tuple[Dict[str, int], List[Dict[str, Any]]]:
        """
        执行 Stability Selection 核心算法
        
        Args:
            X: 因子数据
            y: 前瞻收益率
            base_metric: 评估指标 ("dcor", "ic", "ir")
            n_bootstrap: 子采样次数
            subsample_ratio: 每次采样的数据比例
            selection_mode: "top_k", "top_pct" 或 "threshold"
            per_sample_top_k: top_k 模式下每次选择的因子数
            per_sample_top_pct: top_pct 模式下每次选择的因子比例
            per_sample_metric_threshold: threshold 模式下的阈值
            
        Returns:
            (factor_counts, selection_details)
            - factor_counts: {因子名: 被选中次数}
            - selection_details: 每次采样的详细信息列表
        """
        # 初始化计数器
        factor_counts: Dict[str, int] = {col: 0 for col in X.columns}
        selection_details: List[Dict[str, Any]] = []
        
        n_factors = len(X.columns)
        
        # 获取唯一索引 (用于采样)
        # 对于 MultiIndex (date, stock_code)，我们按行采样
        n_samples = len(X)
        sample_size = int(n_samples * subsample_ratio)
        
        if sample_size < 100:
            self.logger.warning(
                f"子采样大小 ({sample_size}) 较小，可能影响稳定性"
            )
        
        # 获取并行配置
        parallel_cfg = self.config.get('parallel', {})
        parallel_enabled = parallel_cfg.get('enabled', True)
        n_jobs = parallel_cfg.get('n_jobs', -1)
        backend = parallel_cfg.get('backend', 'loky')  # 默认使用 loky 进程池
        
        sampling_mode = self.config.get('sampling_mode', 'random_rows')
        if sampling_mode == 'contiguous_time_blocks':
            date_values = pd.to_datetime(X.index.get_level_values(0)).normalize()
            unique_dates = np.sort(date_values.unique())
            block_size = max(1, int(len(unique_dates) * subsample_ratio))
            starts = np.linspace(
                0,
                max(0, len(unique_dates) - block_size),
                num=n_bootstrap,
                dtype=int,
            )
            all_sample_indices = [
                np.flatnonzero(date_values.isin(unique_dates[start:start + block_size]))
                for start in starts
            ]
        else:
            seed = self.config.get('random_seed', 42)
            rng = np.random.default_rng(seed)
            all_sample_indices = [
                rng.choice(n_samples, size=sample_size, replace=False)
                for _ in range(n_bootstrap)
            ]
        
        # 并行执行每次子采样
        if parallel_enabled and n_bootstrap > 1:
            # 使用配置的后端，对于 CPU 密集型计算推荐 loky
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(self._run_single_bootstrap)(
                    X, y, sample_idx,
                    base_metric, selection_mode, 
                    per_sample_top_k, per_sample_top_pct, per_sample_metric_threshold,
                    n_factors
                )
                for sample_idx in all_sample_indices
            )
        else:
            results = [
                self._run_single_bootstrap(
                    X, y, sample_idx,
                    base_metric, selection_mode, 
                    per_sample_top_k, per_sample_top_pct, per_sample_metric_threshold,
                    n_factors
                )
                for sample_idx in all_sample_indices
            ]
        
        # 汇总结果
        for selected_factors, detail in results:
            for f in selected_factors:
                factor_counts[f] += 1
            selection_details.append(detail)
        
        return factor_counts, selection_details
    
    def _run_single_bootstrap(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_indices: np.ndarray,
        base_metric: str,
        selection_mode: str,
        per_sample_top_k: int,
        per_sample_top_pct: float,
        per_sample_metric_threshold: float,
        n_factors: int
    ) -> tuple[List[str], Dict[str, Any]]:
        """
        在单次子采样上执行因子选择
        
        Args:
            X: 完整因子数据
            y: 完整收益率
            sample_indices: 采样索引
            base_metric: 评估指标 ("dcor", "ic", "ir")
            selection_mode: "top_k", "top_pct" 或 "threshold"
            per_sample_top_k: top_k 模式下选择的因子数
            per_sample_top_pct: top_pct 模式下选择的因子比例
            per_sample_metric_threshold: threshold 模式下的阈值
            n_factors: 总因子数 (用于计算百分比)
            
        Returns:
            (selected_factors, detail_info)
        """
        try:
            # 提取子采样数据
            X_sub = X.iloc[sample_indices]
            y_sub = y.iloc[sample_indices]
            
            # 根据 base_metric 计算所有因子的指标
            metric_results = self._compute_metric_for_all_factors(X_sub, y_sub, base_metric)
            
            if not metric_results:
                return [], {'n_samples': len(sample_indices), 'n_selected': 0, 'metric': base_metric}
            
            # 排序结果（所有模式都需要）
            sorted_results = sorted(
                metric_results, 
                key=lambda x: x['metric_value'], 
                reverse=True
            )
            
            # 根据模式选择因子
            if selection_mode == 'top_k':
                # 选择指标最大的 top_k 个因子
                n_select = min(per_sample_top_k, len(sorted_results))
                selected = [r['factor_name'] for r in sorted_results[:n_select]]
                
            elif selection_mode == 'top_pct':
                # 选择指标最大的 top X% 因子
                # 使用实际有效因子数计算，而非总因子数
                n_valid = len(sorted_results)
                n_select = max(1, int(n_valid * per_sample_top_pct))  # 至少选1个
                selected = [r['factor_name'] for r in sorted_results[:n_select]]
                
            else:  # threshold
                # 选择指标 >= threshold 的因子
                selected = [
                    r['factor_name'] 
                    for r in metric_results 
                    if r['metric_value'] >= per_sample_metric_threshold
                ]
                # 如果没有因子通过阈值，至少选 top 3
                if not selected:
                    n_select = min(3, len(sorted_results))
                    selected = [r['factor_name'] for r in sorted_results[:n_select]]
            
            # 构建详细信息
            metric_values = [r['metric_value'] for r in metric_results]
            detail = {
                'n_samples': len(sample_indices),
                'n_selected': len(selected),
                'n_valid_factors': len(metric_results),
                'selection_mode': selection_mode,
                'metric': base_metric,
                'metric_stats': {
                    'max': max(metric_values),
                    'min': min(metric_values),
                    'mean': np.mean(metric_values)
                }
            }
            
            return selected, detail
            
        except LabelIsolationError:
            raise
        except Exception:
            # 计算失败不能作为未选中观测计入选择概率。
            raise
    
    def _compute_metric_for_all_factors(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_metric: str
    ) -> List[Dict[str, Any]]:
        """
        计算所有因子的指定指标
        
        Args:
            X: 因子数据
            y: 收益率
            base_metric: 评估指标 ("dcor", "ic", "ir")
            
        Returns:
            [{factor_name, metric_value, metric_type}, ...]
        """
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
        
        if X_aligned.empty:
            return []
        
        results = []
        
        # Check if vectorized IC/IR is enabled (default: True)
        use_vectorized_ic = self.config.get('use_vectorized_ic', True)
        
        if base_metric == 'dcor':
            # dCor: 使用 numpy 数组直接计算
            # Performance optimization: sample when n_samples > dcor_max_samples
            dcor_max_samples = self.config.get('dcor_max_samples', 5000)
            random_seed = self.config.get('random_seed', 42)
            
            y_values = y_aligned.values
            for col in X_aligned.columns:
                result = _compute_single_dcor(
                    col, X_aligned[col].values, y_values,
                    max_samples=dcor_max_samples,
                    random_seed=random_seed,
                )
                if result is not None:
                    results.append(result)
                    
        elif base_metric == 'ic':
            if use_vectorized_ic:
                # === Vectorized IC computation (2-5x faster) ===
                ic_results = _compute_ic_vectorized(X_aligned, y_aligned)
                results = list(ic_results.values())
            else:
                # Legacy: 逐因子循环
                for col in X_aligned.columns:
                    result = _compute_single_ic(col, X_aligned[col], y_aligned)
                    if result is not None:
                        results.append(result)
                    
        elif base_metric == 'ir':
            if use_vectorized_ic:
                # === Vectorized IR computation (2-5x faster) ===
                ic_results = _compute_ic_vectorized(X_aligned, y_aligned)
                ir_results = _compute_ir_from_ic_results(ic_results)
                results = list(ir_results.values())
            else:
                # Legacy: 逐因子循环
                for col in X_aligned.columns:
                    result = _compute_single_ir(col, X_aligned[col], y_aligned)
                    if result is not None:
                        results.append(result)
        
        return results
    
    def _pre_filter_by_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        all_factors: List[str],
        threshold: float,
        base_metric: str
    ) -> List[str]:
        """
        相关性预过滤：在 Stability Selection 前移除高度相关的冗余因子
        
        算法流程:
        1. 计算所有因子的指标值 (dCor/IC/IR)
        2. 按指标值降序排序
        3. 贪婪筛选：保留与已选因子相关性 <= threshold 的因子
        
        Args:
            X: 因子数据
            y: 前瞻收益率
            all_factors: 所有因子名称列表
            threshold: 相关性阈值
            base_metric: 评估指标 ("dcor", "ic", "ir")
            
        Returns:
            预过滤后的因子列表
        """
        if len(all_factors) <= 1:
            return all_factors
        
        try:
            # Step 1: 计算所有因子的指标值
            metric_results = self._compute_metric_for_all_factors(X, y, base_metric)
            
            if not metric_results:
                self.logger.warning("无法计算指标值，跳过相关性预过滤")
                return all_factors
            
            # Step 2: 按指标值降序排序
            sorted_results = sorted(
                metric_results,
                key=lambda x: x['metric_value'],
                reverse=True
            )
            sorted_factors = [r['factor_name'] for r in sorted_results]
            
            # 记录指标值用于调试
            self._pre_filter_metrics = {
                r['factor_name']: r['metric_value'] for r in sorted_results
            }
            
            # Step 3: 贪婪相关性筛选
            valid_cols = [c for c in sorted_factors if c in X.columns]
            if len(valid_cols) < 2:
                return valid_cols
            
            # 计算相关性矩阵 (with caching)
            cache = get_global_cache()
            corr_matrix, cached_cols = cache.get_correlation_matrix(X, valid_cols)
            col_to_idx = {col: i for i, col in enumerate(cached_cols)}
            
            # 贪婪筛选
            kept_factors = []
            
            for factor in valid_cols:
                if not kept_factors:
                    kept_factors.append(factor)
                    continue
                
                # 检查当前因子与所有已保留因子的相关性
                current_idx = col_to_idx[factor]
                kept_indices = [col_to_idx[f] for f in kept_factors]
                max_corr_with_kept = corr_matrix[current_idx, kept_indices].max()
                
                if max_corr_with_kept <= threshold:
                    kept_factors.append(factor)
            
            return kept_factors
            
        except LabelIsolationError:
            raise
        except Exception as e:
            self.logger.error(f"相关性预过滤失败: {e}")
            raise
    
    def _build_performance_report(
        self,
        all_factors: List[str],
        selection_probs: Dict[str, float],
        selection_details: List[Dict[str, Any]],
        base_metric: str = 'dcor',
        pre_filtered_factors: Optional[List[str]] = None
    ) -> None:
        """
        构建性能报告 DataFrame
        
        Args:
            all_factors: 所有因子名称
            selection_probs: {因子名: 选择概率}
            selection_details: 每次采样的详细信息
            base_metric: 使用的评估指标
            pre_filtered_factors: 相关性预过滤后保留的因子列表
        """
        # 获取预过滤时的指标值
        pre_filter_metrics = getattr(self, '_pre_filter_metrics', {})
        pre_filtered_set = set(pre_filtered_factors) if pre_filtered_factors else set(all_factors)
        
        records = []
        for factor in all_factors:
            records.append({
                'factor_name': factor,
                'selection_probability': selection_probs.get(factor, 0.0),
                'selection_count': int(round(selection_probs.get(factor, 0.0) * len(selection_details))) if selection_details else 0,
                'base_metric': base_metric,
                'pre_filter_metric': pre_filter_metrics.get(factor, np.nan),
                'passed_corr_filter': factor in pre_filtered_set
            })
        
        self._performance_report = pd.DataFrame(records).sort_values(
            'selection_probability', ascending=False
        ).reset_index(drop=True)
        
        self._performance_report.attrs['correlation_filter_version'] = 'input_priority_v2'
        self._performance_report.attrs['report_lifecycle_version'] = 'per_call_v2'
        self._performance_report.attrs['selection_count_version'] = 'rounded_count_v2'
        if base_metric in ('ic', 'ir'):
            self._performance_report.attrs['metric_sample_version'] = 'pairwise10_ddof1_v2'

        # 存储额外统计
        self._selection_details = selection_details
        self._base_metric = base_metric
        self._pre_filtered_factors = pre_filtered_factors
    
    def get_selection_probabilities(self) -> Dict[str, float]:
        """
        获取所有因子的选择概率
        
        Returns:
            {因子名: 选择概率}
        """
        if self._performance_report is None:
            return {}
        
        return dict(zip(
            self._performance_report['factor_name'],
            self._performance_report['selection_probability']
        ))
    
    def get_selection_details(self) -> List[Dict[str, Any]]:
        """
        获取每次 bootstrap 的详细信息
        
        Returns:
            每次采样的详细信息列表
        """
        return getattr(self, '_selection_details', [])
