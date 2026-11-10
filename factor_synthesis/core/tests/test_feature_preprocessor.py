# -*- coding: utf-8 -*-
"""
Feature Preprocessor 单元测试

测试新增的预处理方法:
1. cross_sectional_binning
2. cross_sectional_winsorize_zscore
3. cross_sectional_rank_zscore
"""

import pytest
import numpy as np
import pandas as pd

from factor_synthesis.core.feature_preprocessor import (
    cross_sectional_standardize,
    cross_sectional_binning,
    cross_sectional_winsorize_zscore,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dates():
    """生成测试日期"""
    return pd.date_range('2023-01-01', periods=10, freq='D')


@pytest.fixture
def stocks():
    """生成测试股票代码"""
    return [f'{i:06d}.XSHE' for i in range(1, 51)]  # 50只股票


@pytest.fixture
def multiindex(dates, stocks):
    """生成 MultiIndex"""
    return pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])


@pytest.fixture
def sample_data(multiindex):
    """生成测试数据"""
    np.random.seed(42)
    return pd.DataFrame({
        'demo_signal1': np.random.randn(len(multiindex)),
        'demo_signal2': np.random.randn(len(multiindex)) * 3 + 10,
    }, index=multiindex)


@pytest.fixture
def sample_series(multiindex):
    """生成测试 Series"""
    np.random.seed(42)
    return pd.Series(np.random.randn(len(multiindex)), index=multiindex)


# =============================================================================
# Test cross_sectional_binning
# =============================================================================

class TestCrossSectionalBinning:
    """测试横截面分箱"""
    
    def test_binning_output_range(self, sample_series):
        """分箱后值应在 [0, 1] 范围内"""
        # Act
        result = cross_sectional_binning(sample_series, n_bins=5)
        
        # Assert
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_binning_discrete_values(self, sample_series):
        """分箱后应只有离散值"""
        # Act
        result = cross_sectional_binning(sample_series, n_bins=5)
        
        # Assert
        unique_values = result.dropna().unique()
        # 5个分箱应产生 0, 0.25, 0.5, 0.75, 1.0 这样的值
        assert len(unique_values) <= 5
    
    def test_binning_dataframe(self, sample_data):
        """测试 DataFrame 输入"""
        # Act
        result = cross_sectional_binning(sample_data, n_bins=5)
        
        # Assert
        assert result.shape == sample_data.shape
        assert (result.min() >= 0).all()
        assert (result.max() <= 1).all()
    
    def test_binning_preserves_index(self, sample_series):
        """分箱应保持索引不变"""
        # Act
        result = cross_sectional_binning(sample_series, n_bins=5)
        
        # Assert
        assert result.index.equals(sample_series.index)
    
    def test_binning_cross_sectional(self, dates, stocks, multiindex):
        """分箱应按日期独立计算"""
        # Arrange
        # 每个日期内排序一致，但日期间不同
        values = []
        for date in dates:
            daily_vals = np.arange(len(stocks)).astype(float)
            values.extend(daily_vals)
        series = pd.Series(values, index=multiindex)
        
        # Act
        result = cross_sectional_binning(series, n_bins=5)
        
        # Assert
        # 每个日期的分箱结果应该相同（因为原始排序相同）
        for date in dates:
            date_result = result.xs(date, level='date')
            assert date_result.min() == 0.0
            assert date_result.max() == 1.0
    
    def test_binning_empty_series(self):
        """空 Series 应返回空结果"""
        # Arrange
        index = pd.MultiIndex.from_arrays([[], []], names=['date', 'stock_code'])
        series = pd.Series(dtype=float, index=index)
        
        # Act
        result = cross_sectional_binning(series)
        
        # Assert
        assert result.empty


# =============================================================================
# Test cross_sectional_winsorize_zscore
# =============================================================================

class TestCrossSectionalWinsorizeZscore:
    """测试横截面缩尾标准化"""
    
    def test_winsorize_zscore_mean_zero(self, sample_series):
        """缩尾标准化后每日均值应接近 0"""
        # Act
        result = cross_sectional_winsorize_zscore(sample_series, percentile=0.025)
        
        # Assert
        daily_means = result.groupby(level='date').mean()
        assert (daily_means.abs() < 1e-10).all()
    
    def test_winsorize_zscore_std_one(self, sample_series):
        """缩尾标准化后每日标准差应接近 1"""
        # Act
        result = cross_sectional_winsorize_zscore(sample_series, percentile=0.025)
        
        # Assert
        daily_stds = result.groupby(level='date').std()
        assert (daily_stds - 1.0).abs().max() < 0.1  # 允许小误差
    
    def test_winsorize_zscore_clips_extremes(self, multiindex):
        """缩尾应裁剪极端值"""
        # Arrange
        # 创建有极端值的数据
        np.random.seed(42)
        values = np.random.randn(len(multiindex))
        # 添加一些极端值
        extreme_idx = np.random.choice(len(values), 10, replace=False)
        values[extreme_idx] = 100  # 极大值
        series = pd.Series(values, index=multiindex)
        
        # Act
        result = cross_sectional_winsorize_zscore(series, percentile=0.05)
        
        # Assert
        # 结果应该没有极端的 z-score
        assert result.abs().max() < 10  # 即使原来有100的值
    
    def test_winsorize_zscore_dataframe(self, sample_data):
        """测试 DataFrame 输入"""
        # Act
        result = cross_sectional_winsorize_zscore(sample_data, percentile=0.025)
        
        # Assert
        assert result.shape == sample_data.shape
        for col in result.columns:
            daily_means = result[col].groupby(level='date').mean()
            assert (daily_means.abs() < 1e-10).all()
    
    def test_winsorize_zscore_preserves_index(self, sample_series):
        """应保持索引不变"""
        # Act
        result = cross_sectional_winsorize_zscore(sample_series)
        
        # Assert
        assert result.index.equals(sample_series.index)


# =============================================================================
# Test cross_sectional_rank_zscore
# =============================================================================



# =============================================================================
# Integration Tests
# =============================================================================

class TestPreprocessingIntegration:
    """集成测试"""
    
    def test_all_methods_same_index(self, sample_series):
        """所有方法应返回相同的索引"""
        # Act
        binning_result = cross_sectional_binning(sample_series)
        winz_result = cross_sectional_winsorize_zscore(sample_series)
        
        # Assert
        assert binning_result.index.equals(sample_series.index)
        assert winz_result.index.equals(sample_series.index)
    
    def test_methods_handle_nan(self, multiindex):
        """所有方法应能处理 NaN"""
        # Arrange
        np.random.seed(42)
        values = np.random.randn(len(multiindex))
        nan_idx = np.random.choice(len(values), 50, replace=False)
        values[nan_idx] = np.nan
        series = pd.Series(values, index=multiindex)
        
        # Act & Assert (不应抛出异常)
        cross_sectional_binning(series)
        cross_sectional_winsorize_zscore(series)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
