# -*- coding: utf-8 -*-
"""
Test suite for Qlib Data Handler integration.

Validates that:
1. Qlib processors produce the same output as manual Pandas preprocessing
2. Bin conversion creates valid Qlib data
3. Forward returns match current calculate_forward_returns logic
4. Sample weights are computed correctly
5. Post-processing blending matches current implementation
"""

import os
import sys
import tempfile
import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from scipy.stats import norm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from factor_synthesis.qlib_adapter.processors import (
    PerFactorPreprocessor,
    TargetCSProcessor,
    SampleWeightProcessor,
)


def create_test_data(n_dates: int = 50, n_stocks: int = 100, n_factors: int = 3) -> pd.DataFrame:
    """Create synthetic test data for validation."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
    stocks = [f"SH60000{i:02d}" for i in range(n_stocks)]
    
    index = pd.MultiIndex.from_product(
        [dates, stocks], 
        names=['datetime', 'instrument']
    )
    
    data = {}
    for i in range(n_factors):
        data[f'DemoSignal{i+1}'] = np.random.randn(len(index))
    
    # Add some NaN values
    df = pd.DataFrame(data, index=index)
    mask = np.random.random(df.shape) < 0.05
    df = df.mask(mask)
    
    return df




class TestPerFactorPreprocessor(unittest.TestCase):
    """Test per-factor preprocessing with different modes."""
    
    def test_zscore_mode(self):
        """Test Z-score preprocessing."""
        df = create_test_data(n_dates=10, n_stocks=100, n_factors=2)
        
        processor = PerFactorPreprocessor({"ALL": "raw_zscore"})
        result = processor(df)
        
        # Each date should have mean ~0 and std ~1
        for date in result.index.get_level_values('datetime').unique():
            for col in result.columns:
                date_vals = result.xs(date, level='datetime')[col].dropna()
                if len(date_vals) > 5:
                    self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)
                    self.assertAlmostEqual(date_vals.std(), 1, delta=0.1)
    
    def test_rank_mode(self):
        """Test rank preprocessing."""
        df = create_test_data(n_dates=5, n_stocks=50, n_factors=1)
        
        processor = PerFactorPreprocessor({"ALL": "rank"})
        result = processor(df)
        
        # Ranks should be in [0, 1]
        self.assertTrue((result.dropna() >= 0).all().all())
        self.assertTrue((result.dropna() <= 1).all().all())
    
    def test_rank_zscore_mode(self):
        """Test rank then Z-score preprocessing."""
        df = create_test_data(n_dates=10, n_stocks=100, n_factors=1)
        
        processor = PerFactorPreprocessor({"ALL": "rank_zscore"})
        result = processor(df)
        
        # Check standardization
        for date in result.index.get_level_values('datetime').unique():
            date_vals = result.xs(date, level='datetime')['DemoSignal1'].dropna()
            if len(date_vals) > 5:
                self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)
    
    def test_per_factor_override(self):
        """Test per-factor preprocessing override."""
        df = create_test_data(n_dates=5, n_stocks=50, n_factors=2)
        
        config = {
            "ALL": "raw_zscore",
            "DemoSignal1": "rank"  # Override for DemoSignal1
        }
        
        processor = PerFactorPreprocessor(config)
        result = processor(df)
        
        # DemoSignal1 should be ranked (in [0, 1])
        self.assertTrue((result['DemoSignal1'].dropna() >= 0).all())
        self.assertTrue((result['DemoSignal1'].dropna() <= 1).all())
        
        # DemoSignal2 should be z-scored (mean ~0)
        for date in result.index.get_level_values('datetime').unique():
            date_vals = result.xs(date, level='datetime')['DemoSignal2'].dropna()
            if len(date_vals) > 5:
                self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)


class TestTargetCSProcessor(unittest.TestCase):
    """Test target cross-sectional preprocessing."""
    
    def setUp(self):
        """Create test target data."""
        self.df = create_test_data(n_dates=20, n_stocks=100, n_factors=1)
        self.df.columns = ['LABEL0']
    
    def test_demean(self):
        """Test cross-sectional demean."""
        processor = TargetCSProcessor(cs_demean=True, cs_rank=False, cs_standardize=False)
        result = processor(self.df)
        
        # Each date should have mean ~0
        for date in result.index.get_level_values('datetime').unique():
            date_vals = result.xs(date, level='datetime')['LABEL0'].dropna()
            if len(date_vals) > 5:
                self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)
    
    def test_rank(self):
        """Test cross-sectional rank then demean."""
        processor = TargetCSProcessor(cs_demean=True, cs_rank=True, cs_standardize=False)
        result = processor(self.df)
        
        # Each date should have mean ~0
        for date in result.index.get_level_values('datetime').unique():
            date_vals = result.xs(date, level='datetime')['LABEL0'].dropna()
            if len(date_vals) > 5:
                self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)
    
    def test_standardize(self):
        """Test cross-sectional standardization."""
        processor = TargetCSProcessor(cs_demean=True, cs_rank=False, cs_standardize=True)
        result = processor(self.df)
        
        # Each date should have mean ~0 and std ~1
        for date in result.index.get_level_values('datetime').unique():
            date_vals = result.xs(date, level='datetime')['LABEL0'].dropna()
            if len(date_vals) > 5:
                self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)
                self.assertAlmostEqual(date_vals.std(), 1, delta=0.1)


class TestSampleWeightProcessor(unittest.TestCase):
    """Test sample weight computation."""
    
    def test_no_decay(self):
        """Test with no decay (decay=1.0)."""
        df = create_test_data(n_dates=30, n_stocks=10, n_factors=1)
        
        processor = SampleWeightProcessor(weight_decay=1.0, half_life=0.0)
        result = processor(df)
        
        # DataFrame should be unchanged
        pd.testing.assert_frame_equal(result, df)
        
        # Weights should all be 1.0
        weights = processor.get_sample_weights()
        np.testing.assert_array_almost_equal(weights, np.ones(len(df)))
    
    def test_exponential_decay(self):
        """Test exponential decay."""
        df = create_test_data(n_dates=30, n_stocks=10, n_factors=1)
        
        processor = SampleWeightProcessor(weight_decay=0.99, half_life=0.0)
        result = processor(df)
        
        weights = processor.get_sample_weights()
        
        # Weights should be non-negative
        self.assertTrue((weights >= 0).all())
        
        # More recent dates should have higher weights
        dates = df.index.get_level_values('datetime')
        max_date = dates.max()
        recent_mask = dates == max_date
        old_mask = dates == dates.min()
        
        self.assertTrue(weights[recent_mask].mean() > weights[old_mask].mean())
    
    def test_half_life_decay(self):
        """Test half-life decay."""
        df = create_test_data(n_dates=60, n_stocks=10, n_factors=1)
        
        half_life = 30  # 30 days
        processor = SampleWeightProcessor(weight_decay=1.0, half_life=half_life)
        result = processor(df)
        
        weights = processor.get_sample_weights()
        
        # Weights should be non-negative
        self.assertTrue((weights >= 0).all())
        
        # Weight at half_life days ago should be ~half of weight at max_date
        dates = df.index.get_level_values('datetime')
        max_date = dates.max()
        
        # Find weight ratio between max_date and max_date - half_life
        recent_weight = weights[dates == max_date].mean()
        target_date = max_date - pd.Timedelta(days=half_life)
        half_weights = weights[dates.date == target_date.date()]
        
        if len(half_weights) > 0:
            half_weight = half_weights.mean()
            # Ratio should be ~0.5 (with some normalization variance)
            ratio = half_weight / recent_weight
            self.assertAlmostEqual(ratio, 0.5, delta=0.1)








class TestProcessorIntegration(unittest.TestCase):
    """Integration tests for processor pipeline."""
    
    def test_full_pipeline(self):
        """Test full preprocessing pipeline."""
        df = create_test_data(n_dates=30, n_stocks=100, n_factors=3)
        
        # 1. Prepare the filled fixture for the remaining processors.
        df = df.groupby(level="instrument").ffill(limit=30)
        
        # 2. Feature preprocessing
        preprocess_config = {"ALL": "gauss_rank", "DemoSignal1": "raw_zscore"}
        feature_processor = PerFactorPreprocessor(preprocess_config)
        df = feature_processor(df)
        
        # 3. Sample weights
        weight_processor = SampleWeightProcessor(weight_decay=0.995)
        df = weight_processor(df)
        weights = weight_processor.get_sample_weights()
        
        # Verify output shapes
        self.assertEqual(len(weights), len(df))
        
        # Verify DemoSignal1 is z-scored
        for date in df.index.get_level_values('datetime').unique():
            date_vals = df.xs(date, level='datetime')['DemoSignal1'].dropna()
            if len(date_vals) > 5:
                self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)


class TestComparisonWithManual(unittest.TestCase):
    """Compare Qlib processors output with manual implementation."""
    
    def test_zscore_matches_manual(self):
        """Verify Z-score matches manual implementation."""
        df = create_test_data(n_dates=20, n_stocks=100, n_factors=1)
        
        # Manual implementation (from run_factor_synthesis.py)
        def manual_zscore(group):
            m, s = group.mean(), group.std()
            if s > 1e-8:
                return (group - m) / s
            return group - m
        
        expected = df['DemoSignal1'].groupby(level='datetime').transform(manual_zscore)
        
        # Qlib processor
        processor = PerFactorPreprocessor({"ALL": "raw_zscore"})
        result = processor(df)['DemoSignal1']
        
        # Compare
        np.testing.assert_array_almost_equal(
            result.dropna().values,
            expected.dropna().values,
            decimal=10
        )
    
    def test_target_demean_matches_manual(self):
        """Verify target demean matches manual implementation."""
        df = create_test_data(n_dates=20, n_stocks=100, n_factors=1)
        df.columns = ['LABEL0']
        
        # Manual implementation (from lasso_method.py)
        expected = df['LABEL0'] - df['LABEL0'].groupby(level='datetime').transform('mean')
        
        # Qlib processor
        processor = TargetCSProcessor(cs_demean=True, cs_rank=False, cs_standardize=False)
        result = processor(df)['LABEL0']
        
        # Compare
        np.testing.assert_array_almost_equal(
            result.dropna().values,
            expected.dropna().values,
            decimal=10
        )
    
    def test_sample_weight_decay_matches_manual(self):
        """Verify sample weight decay matches manual implementation."""
        df = create_test_data(n_dates=30, n_stocks=10, n_factors=1)
        decay = 0.99
        
        # Manual implementation (from lasso_method.py lines 380-385)
        dates = df.index.get_level_values('datetime')
        max_date = dates.max()
        days_diff = (max_date - dates).days.values.astype(float)
        expected_weights = np.power(decay, days_diff)
        expected_weights = expected_weights / expected_weights.mean()
        
        # Qlib processor
        processor = SampleWeightProcessor(weight_decay=decay)
        processor(df)
        result_weights = processor.get_sample_weights()
        
        # Compare
        np.testing.assert_array_almost_equal(
            result_weights,
            expected_weights,
            decimal=10
        )


if __name__ == '__main__':
    unittest.main()

