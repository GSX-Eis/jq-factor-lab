# -*- coding: utf-8 -*-
"""
Test preprocessing utilities that use Qlib processors.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from factor_synthesis.pipeline.preprocessing import (
    apply_target_cs_preprocessing,
    compute_sample_weights,
)


def create_test_data(n_dates: int = 30, n_stocks: int = 50) -> pd.DataFrame:
    """Create synthetic test data."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
    stocks = [f"600{i:03d}.SH" for i in range(n_stocks)]
    
    index = pd.MultiIndex.from_product(
        [dates, stocks], 
        names=['date', 'stock_code']
    )
    
    return pd.DataFrame({
        'DemoSignal1': np.random.randn(len(index)),
        'DemoSignal2': np.random.randn(len(index)),
    }, index=index)


class TestApplyTargetCSPreprocessing(unittest.TestCase):
    """Test target preprocessing utility."""
    
    def test_demean(self):
        """Test cross-sectional demean."""
        df = create_test_data()
        y = df['DemoSignal1']
        
        result = apply_target_cs_preprocessing(y, cs_demean=True, cs_rank=False, cs_standardize=False)
        
        # Each date should have mean ~0
        for date in result.index.get_level_values('date').unique():
            date_vals = result.xs(date, level='date')
            self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)
    
    def test_rank(self):
        """Test rank then demean."""
        df = create_test_data()
        y = df['DemoSignal1']
        
        result = apply_target_cs_preprocessing(y, cs_demean=True, cs_rank=True, cs_standardize=False)
        
        # Each date should have mean ~0
        for date in result.index.get_level_values('date').unique():
            date_vals = result.xs(date, level='date')
            self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)
    
    def test_standardize(self):
        """Test standardization."""
        df = create_test_data()
        y = df['DemoSignal1']
        
        result = apply_target_cs_preprocessing(y, cs_demean=True, cs_rank=False, cs_standardize=True)
        
        # Each date should have mean ~0 and std ~1
        for date in result.index.get_level_values('date').unique():
            date_vals = result.xs(date, level='date')
            self.assertAlmostEqual(date_vals.mean(), 0, delta=1e-10)
            self.assertAlmostEqual(date_vals.std(), 1, delta=0.1)


class TestComputeSampleWeights(unittest.TestCase):
    """Test sample weight computation."""
    
    def test_no_decay(self):
        """Test with no decay."""
        df = create_test_data()
        
        weights = compute_sample_weights(df, weight_decay=1.0, half_life=0.0)
        
        # No weights returned when decay=1
        self.assertIsNone(weights)
    
    def test_decay(self):
        """Test exponential decay."""
        df = create_test_data()
        
        weights = compute_sample_weights(df, weight_decay=0.99, half_life=0.0)
        
        # Weights should be non-negative
        self.assertTrue((weights >= 0).all())
        
        # More recent dates should have higher weights
        dates = df.index.get_level_values('date')
        max_date = dates.max()
        min_date = dates.min()
        
        recent_weight = weights[dates == max_date].mean()
        old_weight = weights[dates == min_date].mean()
        
        self.assertGreater(recent_weight, old_weight)
    
    def test_half_life(self):
        """Test half-life decay."""
        df = create_test_data(n_dates=60)
        
        weights = compute_sample_weights(df, weight_decay=1.0, half_life=30)
        
        # Weights should be non-negative
        self.assertTrue((weights >= 0).all())






if __name__ == '__main__':
    unittest.main()

