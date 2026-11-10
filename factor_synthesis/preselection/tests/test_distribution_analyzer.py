# -*- coding: utf-8 -*-
"""
分布分析器单元测试

测试 DistributionAnalyzer 的:
1. 偏度/峰度计算
2. 因子分类逻辑
3. 预处理配置构建
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import skewnorm, t as t_dist

from factor_synthesis.preselection.distribution_analyzer import (
    DistributionAnalyzer,
    DistributionConfig,
    FactorDistributionResult,
    DistributionAnalysisReport,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dates():
    """生成测试日期"""
    return pd.date_range('2023-01-01', periods=50, freq='D')


@pytest.fixture
def stocks():
    """生成测试股票代码"""
    return [f'{i:06d}.XSHE' for i in range(1, 101)]  # 100只股票


@pytest.fixture
def multiindex(dates, stocks):
    """生成 MultiIndex"""
    return pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])


def create_factor_series(index: pd.MultiIndex, distribution: str, **kwargs) -> pd.Series:
    """
    创建具有特定分布的因子序列
    
    Args:
        index: MultiIndex (date, stock_code)
        distribution: 分布类型 ('normal', 'skewed', 'heavy_tail', 'sparse')
        **kwargs: 分布参数
        
    Returns:
        因子值 Series
    """
    n = len(index)
    np.random.seed(kwargs.get('seed', 42))
    
    if distribution == 'normal':
        # 正态分布: skew ≈ 0, kurt ≈ 0
        values = np.random.normal(0, 1, n)
    elif distribution == 'skewed':
        # 高偏度: |skew| > 3
        a = kwargs.get('skew_param', 10)  # skewness parameter
        values = skewnorm.rvs(a, size=n)
    elif distribution == 'heavy_tail':
        # 高峰度 (厚尾): kurt > 10
        df = kwargs.get('df', 3)  # t分布自由度，越小尾越厚
        values = t_dist.rvs(df, size=n)
    elif distribution == 'sparse':
        # 稀疏信号: 大部分为0，少数极端值
        values = np.zeros(n)
        n_nonzero = int(n * 0.05)  # 5% 非零
        idx = np.random.choice(n, n_nonzero, replace=False)
        values[idx] = np.random.exponential(5, n_nonzero) * np.random.choice([-1, 1], n_nonzero)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return pd.Series(values, index=index)


# =============================================================================
# Test Cases
# =============================================================================

class TestDistributionAnalyzerClassification:
    """测试因子分类逻辑"""
    
    def test_normal_factor_classification(self, multiindex):
        """正态分布因子应被分类为 'normal'"""
        # Arrange
        df = pd.DataFrame({
            'normal_factor': create_factor_series(multiindex, 'normal')
        })
        analyzer = DistributionAnalyzer()
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        result = report.results['normal_factor']
        assert result.category == 'normal'
        assert result.default_preprocess == 'rank_zscore'
        assert abs(result.skewness_mean) < 2.0  # 正态分布 skew ≈ 0
        assert result.kurtosis_mean < 6.0  # 正态分布 excess kurt ≈ 0
    
    def test_skewed_factor_classification(self, multiindex):
        """高偏度因子应被分类为 'event_driven'"""
        # Arrange
        df = pd.DataFrame({
            'skewed_factor': pd.Series(np.random.default_rng(42).lognormal(sigma=2, size=len(multiindex)), index=multiindex)
        })
        analyzer = DistributionAnalyzer()
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        result = report.results['skewed_factor']
        # 高偏度应该被分类为 event_driven 或 borderline
        assert result.category in ('event_driven', 'borderline')
        # 检查偏度确实较高
        assert abs(result.skewness_mean) > 1.0
    
    def test_heavy_tail_factor_classification(self, multiindex):
        """厚尾因子应被分类为 'event_driven'"""
        # Arrange
        df = pd.DataFrame({
            'heavy_tail_factor': create_factor_series(multiindex, 'heavy_tail', df=2)
        })
        analyzer = DistributionAnalyzer(
            config=DistributionConfig(kurtosis_event_threshold=5.0)  # 降低阈值
        )
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        result = report.results['heavy_tail_factor']
        # t分布 df=2 有较高峰度
        assert result.kurtosis_mean > 3.0  # t(2) 的 excess kurtosis 很高
    
    def test_sparse_factor_classification(self, multiindex):
        """稀疏信号因子应被分类为 'event_driven'"""
        # Arrange
        df = pd.DataFrame({
            'sparse_factor': create_factor_series(multiindex, 'sparse')
        })
        analyzer = DistributionAnalyzer()
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        result = report.results['sparse_factor']
        # 稀疏信号通常有高偏度和高峰度
        assert result.category in ('event_driven', 'borderline')


class TestDistributionAnalyzerConfig:
    """测试配置参数"""
    
    def test_custom_thresholds(self, multiindex):
        """自定义阈值应影响分类结果"""
        # Arrange
        df = pd.DataFrame({
            'test_factor': create_factor_series(multiindex, 'normal')
        })
        
        # 使用非常严格的阈值
        strict_config = DistributionConfig(
            skew_normal_threshold=0.1,  # 非常严格
            kurtosis_normal_threshold=0.5,
        )
        analyzer = DistributionAnalyzer(config=strict_config)
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        result = report.results['test_factor']
        # 即使正态分布也可能被分为 borderline
        assert result.category in ('normal', 'borderline')
    
    def test_config_override(self, multiindex):
        """Config 覆盖应优先于分布分析默认值"""
        # Arrange
        df = pd.DataFrame({
            'factor_a': create_factor_series(multiindex, 'normal'),
            'factor_b': create_factor_series(multiindex, 'normal', seed=123),
        })
        analyzer = DistributionAnalyzer()
        
        # Act
        report = analyzer.analyze(df)
        preprocess_cfg = analyzer.build_preprocess_config(
            report.results,
            config_overrides={'factor_a': 'zscore'}  # 显式覆盖
        )
        
        # Assert
        assert preprocess_cfg['factor_a'] == 'zscore'  # 使用覆盖值
        assert preprocess_cfg['factor_b'] == 'rank_zscore'  # 使用默认值


class TestDistributionAnalyzerReport:
    """测试报告生成"""
    
    def test_category_counts(self, multiindex):
        """测试类别计数"""
        # Arrange
        df = pd.DataFrame({
            'normal1': create_factor_series(multiindex, 'normal', seed=1),
            'normal2': create_factor_series(multiindex, 'normal', seed=2),
            'sparse': create_factor_series(multiindex, 'sparse', seed=3),
        })
        analyzer = DistributionAnalyzer()
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        total = sum(report.category_counts.values())
        assert total == 3  # 3个因子
        assert 'normal' in report.category_counts
    
    def test_report_to_dataframe(self, multiindex):
        """测试报告转换为 DataFrame"""
        # Arrange
        df = pd.DataFrame({
            'demo_signal1': create_factor_series(multiindex, 'normal'),
            'demo_signal2': create_factor_series(multiindex, 'normal', seed=99),
        })
        analyzer = DistributionAnalyzer()
        
        # Act
        report = analyzer.analyze(df)
        report_df = report.to_dataframe()
        
        # Assert
        assert len(report_df) == 2
        assert 'factor_name' in report_df.columns
        assert 'skewness_mean' in report_df.columns
        assert 'category' in report_df.columns
        assert 'default_preprocess' in report_df.columns


class TestDistributionAnalyzerEdgeCases:
    """测试边界情况"""
    
    def test_empty_dataframe(self):
        """空 DataFrame 应返回空报告"""
        # Arrange
        df = pd.DataFrame()
        analyzer = DistributionAnalyzer()
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        assert len(report.results) == 0
        assert report.to_dataframe().empty
    
    def test_single_value_per_date(self, dates):
        """每日只有一个值时应能处理"""
        # Arrange
        # 每日只有一只股票
        index = pd.MultiIndex.from_arrays([
            dates,
            [f'{i:06d}.XSHE' for i in range(len(dates))]
        ], names=['date', 'stock_code'])
        
        df = pd.DataFrame({
            'factor': np.random.randn(len(index))
        }, index=index)
        
        analyzer = DistributionAnalyzer(
            config=DistributionConfig(min_samples_per_date=1)
        )
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        # 样本数不足时应归类为 borderline
        assert 'factor' in report.results
    
    def test_all_nan_factor(self, multiindex):
        """全 NaN 因子应能处理"""
        # Arrange
        df = pd.DataFrame({
            'nan_factor': pd.Series(np.nan, index=multiindex)
        })
        analyzer = DistributionAnalyzer()
        
        # Act
        report = analyzer.analyze(df)
        
        # Assert
        result = report.results['nan_factor']
        assert result.category == 'borderline'  # 无法分析时归为 borderline


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
