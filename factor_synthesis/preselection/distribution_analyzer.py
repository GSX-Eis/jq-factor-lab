# -*- coding: utf-8 -*-
"""
因子分布分析器

基于偏度（Skewness）和峰度（Kurtosis）对因子进行分类，
并自动分配默认的预处理方法。

分类阈值（可配置，以下为默认值）：
- Normal:       |Skew| < 2 且 Kurt < 6  → rank_zscore
- Event-Driven: |Skew| > 3 或 Kurt > 10 → binning
- Borderline:   介于两者之间            → winsorize_zscore

Performance Optimizations (2026-02-04):
- Vectorized computation: compute all factors' skew/kurt in one groupby operation
- Date sampling: only use subset of dates for estimation when n_dates > threshold

Author: QuantationAI
Date: 2026-02-04
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import time

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class DistributionConfig:
    """分布筛选配置
    
    Attributes:
        skew_normal_threshold: |Skew| < 此值视为正常分布
        kurtosis_normal_threshold: Kurt < 此值视为正常分布
        skew_event_threshold: |Skew| > 此值视为事件驱动型
        kurtosis_event_threshold: Kurt > 此值视为事件驱动型
        binning_n_bins: 分箱数（用于 binning 预处理）
        winsorize_percentile: 缩尾百分位（用于 winsorize_zscore）
        min_samples_per_date: 每日最少样本数，低于此值不计算统计量
        max_sample_dates: 最大采样日期数，超过此值时随机采样以加速计算（0 表示不采样）
        use_vectorized: 是否使用向量化计算（推荐开启，5-10x 加速）
    """
    skew_normal_threshold: float = 2.0
    kurtosis_normal_threshold: float = 6.0
    skew_event_threshold: float = 3.0
    kurtosis_event_threshold: float = 10.0
    binning_n_bins: int = 5
    winsorize_percentile: float = 0.025
    min_samples_per_date: int = 10
    max_sample_dates: int = 100  # 0 = no sampling, >0 = sample this many dates
    use_vectorized: bool = True  # Use vectorized computation (recommended)


@dataclass
class FactorDistributionResult:
    """单因子分布分析结果
    
    Attributes:
        factor_name: 因子名称
        skewness_mean: 每日偏度的均值
        skewness_std: 每日偏度的标准差
        kurtosis_mean: 每日峰度的均值（超额峰度，正态分布为0）
        kurtosis_std: 每日峰度的标准差
        n_valid_days: 有效计算天数
        category: 分类结果 ('normal', 'event_driven', 'borderline')
        default_preprocess: 默认预处理方法 ('rank_zscore', 'binning', 'winsorize_zscore')
    """
    factor_name: str
    skewness_mean: float
    skewness_std: float
    kurtosis_mean: float
    kurtosis_std: float
    n_valid_days: int
    category: str
    default_preprocess: str


@dataclass
class DistributionAnalysisReport:
    """分布分析汇总报告
    
    Attributes:
        results: 每个因子的分析结果
        category_counts: 各类别因子数量
        preprocess_config: 生成的预处理配置
    """
    results: Dict[str, FactorDistributionResult] = field(default_factory=dict)
    category_counts: Dict[str, int] = field(default_factory=dict)
    preprocess_config: Dict[str, str] = field(default_factory=dict)
    sampling_version: Optional[str] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """将结果转换为 DataFrame 便于查看"""
        records = []
        for r in self.results.values():
            records.append({
                'factor_name': r.factor_name,
                'skewness_mean': r.skewness_mean,
                'skewness_std': r.skewness_std,
                'kurtosis_mean': r.kurtosis_mean,
                'kurtosis_std': r.kurtosis_std,
                'n_valid_days': r.n_valid_days,
                'category': r.category,
                'default_preprocess': r.default_preprocess,
            })
        frame = pd.DataFrame(records)
        if self.sampling_version is not None:
            frame.attrs['sampling_version'] = self.sampling_version
        return frame


class DistributionAnalyzer:
    """因子分布分析器
    
    分析因子的偏度和峰度分布特征，自动分类并分配默认预处理方法。
    
    Usage:
        >>> config = DistributionConfig(skew_normal_threshold=2.0, ...)
        >>> analyzer = DistributionAnalyzer(config, logger)
        >>> report = analyzer.analyze(X)  # X: MultiIndex DataFrame
        >>> preprocess_cfg = analyzer.build_preprocess_config(
        ...     report.results, config_overrides={'DemoSignal1': 'zscore'}
        ... )
    """
    
    def __init__(
        self, 
        config: Optional[DistributionConfig] = None,
        custom_logger: Optional[logging.Logger] = None
    ):
        """
        初始化分布分析器
        
        Args:
            config: 分布筛选配置，若为 None 则使用默认值
            custom_logger: 自定义日志器
        """
        self.config = config or DistributionConfig()
        self.logger = custom_logger or logger
    
    def analyze(self, X: pd.DataFrame) -> DistributionAnalysisReport:
        """
        分析所有因子的分布特征
        
        Args:
            X: 因子数据 DataFrame
               - Index: MultiIndex (date, stock_code)
               - Columns: 因子名称
               
        Returns:
            DistributionAnalysisReport: 包含所有因子分析结果的报告
        """
        if X.empty:
            self.logger.warning("[分布分析] 输入数据为空")
            return DistributionAnalysisReport()
        
        # Use vectorized implementation if enabled (default)
        if self.config.use_vectorized:
            return self._analyze_vectorized(X)
        else:
            return self._analyze_legacy(X)
    
    def _analyze_vectorized(self, X: pd.DataFrame) -> DistributionAnalysisReport:
        """
        向量化分析所有因子的分布特征 (5-10x faster than legacy)
        
        Key optimizations:
        1. Compute all factors' skew/kurt in one groupby operation
        2. Optional date sampling to further reduce computation
        """
        start_time = time.time()
        n_factors = len(X.columns)
        
        # === Step 1: Date sampling (optional) ===
        dates = X.index.get_level_values('date').unique()
        n_dates = len(dates)
        sampling_version = None
        
        if self.config.max_sample_dates > 0 and n_dates > self.config.max_sample_dates:
            dates = dates.sort_values()
            sampling_version = 'sorted_dates_v2'
            # Random sample dates for estimation
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            sampled_dates = rng.choice(dates, size=self.config.max_sample_dates, replace=False)
            X_sampled = X.loc[X.index.get_level_values('date').isin(sampled_dates)]
            self.logger.info(
                f"[分布分析] 采样 {self.config.max_sample_dates}/{n_dates} 个日期进行估计 ({sampling_version})"
            )
        else:
            X_sampled = X
        
        self.logger.info(f"[分布分析] 开始向量化分析 {n_factors} 个因子的分布特征")
        
        # === Step 2: Compute daily counts to filter low-sample dates ===
        min_samples = self.config.min_samples_per_date
        daily_counts = X_sampled.groupby(level='date').count()
        valid_mask = daily_counts >= min_samples  # (n_dates, n_factors) bool DataFrame
        
        # === Step 3: Vectorized skewness computation ===
        # pandas .skew() is C-implemented, much faster than scipy.stats.skew in a loop
        daily_skew = X_sampled.groupby(level='date').skew()  # (n_dates, n_factors)
        
        # === Step 4: Vectorized kurtosis computation ===
        # Use pandas kurtosis (Fisher=True by default, excess kurtosis)
        daily_kurt = X_sampled.groupby(level='date').apply(
            lambda g: g.kurtosis(), include_groups=False
        )  # (n_dates, n_factors)
        
        # === Step 5: Apply validity mask ===
        daily_skew = daily_skew.where(valid_mask)
        daily_kurt = daily_kurt.where(valid_mask)
        
        # === Step 6: Aggregate statistics (vectorized) ===
        skew_mean = daily_skew.mean()
        skew_std = daily_skew.std()
        kurt_mean = daily_kurt.mean()
        kurt_std = daily_kurt.std()
        n_valid_days = daily_skew.notna().sum()
        
        # === Step 7: Classify factors and build results ===
        results: Dict[str, FactorDistributionResult] = {}
        category_counts: Dict[str, int] = {'normal': 0, 'event_driven': 0, 'borderline': 0}
        
        for col in X.columns:
            try:
                stats = {
                    'skew_mean': float(skew_mean[col]) if pd.notna(skew_mean[col]) else 0.0,
                    'skew_std': float(skew_std[col]) if pd.notna(skew_std[col]) else 0.0,
                    'kurt_mean': float(kurt_mean[col]) if pd.notna(kurt_mean[col]) else 0.0,
                    'kurt_std': float(kurt_std[col]) if pd.notna(kurt_std[col]) else 0.0,
                    'n_valid_days': int(n_valid_days[col]),
                }
                category, preprocess = self._classify(stats)
                
                results[col] = FactorDistributionResult(
                    factor_name=col,
                    skewness_mean=stats['skew_mean'],
                    skewness_std=stats['skew_std'],
                    kurtosis_mean=stats['kurt_mean'],
                    kurtosis_std=stats['kurt_std'],
                    n_valid_days=stats['n_valid_days'],
                    category=category,
                    default_preprocess=preprocess,
                )
                category_counts[category] += 1
                
            except LabelIsolationError:
                raise
            except Exception as e:
                self.logger.warning(f"[分布分析] 因子 {col} 分类失败: {e}")
                raise
        
        # Build preprocess config
        preprocess_config = {r.factor_name: r.default_preprocess for r in results.values()}
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"[分布分析] 完成 ({elapsed:.2f}s): Normal={category_counts['normal']}, "
            f"Event-Driven={category_counts['event_driven']}, "
            f"Borderline={category_counts['borderline']}"
        )
        
        self._log_classification_details(results)
        
        return DistributionAnalysisReport(
            results=results,
            category_counts=category_counts,
            preprocess_config=preprocess_config,
            sampling_version=sampling_version,
        )
    
    def _analyze_legacy(self, X: pd.DataFrame) -> DistributionAnalysisReport:
        """
        遗留版本：逐因子循环分析（较慢，仅用于兼容性）
        """
        results: Dict[str, FactorDistributionResult] = {}
        category_counts: Dict[str, int] = {'normal': 0, 'event_driven': 0, 'borderline': 0}
        
        self.logger.info(f"[分布分析] 开始分析 {len(X.columns)} 个因子的分布特征 (legacy mode)")
        
        for col in X.columns:
            try:
                stats = self._calc_cross_sectional_stats(X[col])
                category, preprocess = self._classify(stats)
                
                result = FactorDistributionResult(
                    factor_name=col,
                    skewness_mean=stats['skew_mean'],
                    skewness_std=stats['skew_std'],
                    kurtosis_mean=stats['kurt_mean'],
                    kurtosis_std=stats['kurt_std'],
                    n_valid_days=stats['n_valid_days'],
                    category=category,
                    default_preprocess=preprocess,
                )
                results[col] = result
                category_counts[category] += 1
                
            except LabelIsolationError:
                raise
            except Exception as e:
                self.logger.warning(f"[分布分析] 因子 {col} 分析失败: {e}")
                raise
        
        preprocess_config = {r.factor_name: r.default_preprocess for r in results.values()}
        
        self.logger.info(
            f"[分布分析] 完成: Normal={category_counts['normal']}, "
            f"Event-Driven={category_counts['event_driven']}, "
            f"Borderline={category_counts['borderline']}"
        )
        
        self._log_classification_details(results)
        
        return DistributionAnalysisReport(
            results=results,
            category_counts=category_counts,
            preprocess_config=preprocess_config,
        )
    
    def _calc_cross_sectional_stats(self, series: pd.Series) -> Dict:
        """
        计算单个因子的跨截面分布统计量 (legacy method, used by _analyze_legacy)
        
        对每个日期计算截面上的偏度和峰度，然后聚合为均值和标准差。
        
        Args:
            series: 因子值 Series，MultiIndex (date, stock_code)
            
        Returns:
            包含 skew_mean, skew_std, kurt_mean, kurt_std, n_valid_days 的字典
        """
        min_samples = self.config.min_samples_per_date
        
        def _safe_skew(x: pd.Series) -> float:
            """安全计算偏度 (using pandas)"""
            clean = x.dropna()
            if len(clean) < min_samples:
                return np.nan
            return float(clean.skew())  # pandas skew()
        
        def _safe_kurtosis(x: pd.Series) -> float:
            """安全计算超额峰度（Fisher定义，正态分布为0）"""
            clean = x.dropna()
            if len(clean) < min_samples:
                return np.nan
            return float(clean.kurtosis())  # pandas kurtosis() (Fisher by default)
        
        # 按日期分组计算
        daily_skew = series.groupby(level='date').apply(_safe_skew)
        daily_kurt = series.groupby(level='date').apply(_safe_kurtosis)
        
        # 移除无效值
        daily_skew_clean = daily_skew.dropna()
        daily_kurt_clean = daily_kurt.dropna()
        
        n_valid_days = len(daily_skew_clean)
        
        return {
            'skew_mean': float(daily_skew_clean.mean()) if not daily_skew_clean.empty else 0.0,
            'skew_std': float(daily_skew_clean.std()) if not daily_skew_clean.empty else 0.0,
            'kurt_mean': float(daily_kurt_clean.mean()) if not daily_kurt_clean.empty else 0.0,
            'kurt_std': float(daily_kurt_clean.std()) if not daily_kurt_clean.empty else 0.0,
            'n_valid_days': n_valid_days,
        }
    
    def _classify(self, stats: Dict) -> Tuple[str, str]:
        """
        根据统计值分类因子并分配默认预处理方法
        
        分类逻辑（阈值可配置，以下为默认值）：
        1. Event-Driven: |Skew| > 3 或 Kurt > 10
           - 稀疏信号型因子，大部分值集中在中间，少数极端值具有重要信息
           - 使用 binning 保留尾部信息，避免 rank 破坏信号
           
        2. Normal: |Skew| < 2 且 Kurt < 6
           - 分布较均匀，rank 几乎不丢失信息
           - 使用 rank_zscore 获得稳健性
           
        3. Borderline: 介于两者之间
           - 先 winsorize 压制极值，再 zscore 标准化
        
        Args:
            stats: 包含 skew_mean, kurt_mean 等统计量的字典
            
        Returns:
            (category, default_preprocess) 元组
        """
        skew_abs = abs(stats['skew_mean'])
        kurt = stats['kurt_mean']
        
        # 处理 NaN
        if stats['n_valid_days'] == 0 or np.isnan(skew_abs) or np.isnan(kurt):
            return ('borderline', 'winsorize_zscore')
        
        # Event-Driven: 稀疏信号，高偏度或高峰度
        if skew_abs > self.config.skew_event_threshold or kurt > self.config.kurtosis_event_threshold:
            return ('event_driven', 'binning')
        
        # Normal: 正常分布，适合 rank
        if skew_abs < self.config.skew_normal_threshold and kurt < self.config.kurtosis_normal_threshold:
            return ('normal', 'rank_zscore')
        
        # Borderline: 边界情况
        return ('borderline', 'winsorize_zscore')
    
    def _log_classification_details(self, results: Dict[str, FactorDistributionResult]) -> None:
        """记录分类详情（部分因子）"""
        # 按类别分组
        by_category: Dict[str, List[FactorDistributionResult]] = {
            'normal': [],
            'event_driven': [],
            'borderline': [],
        }
        
        for r in results.values():
            by_category.get(r.category, []).append(r)
        
        # 输出前 10 个 Event-Driven 因子及总数
        if by_category['event_driven']:
            event_names = [r.factor_name for r in by_category['event_driven']]
            self.logger.info(f"[分布分析] Event-Driven 因子 ({len(event_names)}): {event_names[:10]}{'...' if len(event_names) > 10 else ''}")
        
        # 输出前 5 个 Borderline 因子
        if by_category['borderline']:
            borderline_sample = by_category['borderline'][:5]
            self.logger.debug(
                f"[分布分析] Borderline 因子示例: "
                f"{[(r.factor_name, f'skew={r.skewness_mean:.2f}', f'kurt={r.kurtosis_mean:.2f}') for r in borderline_sample]}"
            )
    
    def build_preprocess_config(
        self,
        results: Dict[str, FactorDistributionResult],
        config_overrides: Optional[Dict[str, str]] = None,
        model_type: str = "lasso",
    ) -> Dict[str, str]:
        """
        构建最终预处理配置

        对于线性模型 (lasso/elasticnet/ridge/quantile_elasticnet)，event_driven 因子使用 winsorize_zscore
        以保留连续信号；树模型 (lightgbm) 继续使用 binning。

        优先级:
        1. config_overrides 中显式指定的因子 → 使用指定方法
        2. 未指定的因子 → 使用分布分析的默认方法（按 model_type 调整 event_driven）

        Args:
            results: 分布分析结果
            config_overrides: Config 中显式指定的预处理方法覆盖
                             （不包含 "ALL" 键，仅包含因子名 -> 方法）
            model_type: "lasso" | "elasticnet" | "ridge" | "lightgbm" 等

        Returns:
            因子名 -> 预处理方法 的映射字典
        """
        config_overrides = config_overrides or {}
        preprocess_cfg: Dict[str, str] = {}
        use_linear_safe = model_type.lower() in ("lasso", "elasticnet", "ridge", "quantile_elasticnet")
        if model_type.lower() == "quantile_elasticnet":
            self.logger.info(
                "[分布分析] preprocess_default_version=quantile_linear_v2 model_type=%s",
                model_type,
            )

        override_count = 0
        default_count = 0

        for factor_name, result in results.items():
            if factor_name in config_overrides:
                # Config 显式覆盖
                preprocess_cfg[factor_name] = config_overrides[factor_name]
                override_count += 1
            elif result.category == "event_driven":
                # 线性模型用 winsorize_zscore 保留连续信号；树模型用 binning
                preprocess_cfg[factor_name] = "winsorize_zscore" if use_linear_safe else "binning"
                default_count += 1
            else:
                # 使用分布分析的默认值
                preprocess_cfg[factor_name] = result.default_preprocess
                default_count += 1
        
        if override_count > 0:
            self.logger.info(
                f"[分布分析] 预处理配置: {override_count} 个因子使用 Config 覆盖, "
                f"{default_count} 个使用分布分析默认值"
            )
        
        return preprocess_cfg
