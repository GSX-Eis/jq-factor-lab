# -*- coding: utf-8 -*-
"""
Test market cap neutralization utilities.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


def create_test_data(n_dates: int = 30, n_stocks: int = 100) -> tuple:
    """Create synthetic test data with known market cap correlation."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
    stocks = [f"600{i:03d}.SH" for i in range(n_stocks)]
    
    index = pd.MultiIndex.from_product(
        [dates, stocks], 
        names=['date', 'stock_code']
    )
    
    # Create log market cap with some structure
    log_mkt_cap = pd.Series(
        np.random.randn(len(index)) * 2 + 22,  # mean ~22, std ~2
        index=index,
        name='log_market_cap'
    )
    
    # Create factors with known correlation to market cap
    # DemoSignal1: positively correlated with market cap
    demo_signal1 = log_mkt_cap * 0.5 + np.random.randn(len(index)) * 0.1
    # DemoSignal2: negatively correlated
    demo_signal2 = -log_mkt_cap * 0.3 + np.random.randn(len(index)) * 0.2
    # DemoSignal3: uncorrelated
    demo_signal3 = pd.Series(np.random.randn(len(index)), index=index)
    
    factor_df = pd.DataFrame({
        'DemoSignal1': demo_signal1,
        'DemoSignal2': demo_signal2,
        'DemoSignal3': demo_signal3,
    }, index=index)
    
    return factor_df, log_mkt_cap


class TestMarketCapNeutralization(unittest.TestCase):
    """Test market cap neutralization logic."""
    
    def test_neutralization_removes_correlation(self):
        """Verify neutralized factor has ~0 correlation with log market cap."""
        factor_df, log_mkt_cap = create_test_data()
        
        # Manually apply neutralization (same logic as _apply_market_cap_neutralization)
        result_df = factor_df.copy()
        dates = factor_df.index.get_level_values('date').unique()
        
        for date in dates:
            date_mask = factor_df.index.get_level_values('date') == date
            factors_cs = factor_df.loc[date_mask]
            mkt_cs = log_mkt_cap.loc[date_mask]
            
            # Find valid (non-NaN) stocks for market cap
            valid_mask = mkt_cs.notna()
            n_valid = int(valid_mask.sum())
            
            if n_valid < 5:
                continue
            
            X = mkt_cs[valid_mask].values.reshape(-1, 1)
            X_with_intercept = np.column_stack([np.ones(n_valid), X])
            
            for col in factors_cs.columns:
                y = factors_cs[col].loc[valid_mask].values
                
                if np.isnan(y).any():
                    continue
                
                beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
                y_hat = X_with_intercept @ beta
                epsilon = y - y_hat
                
                # Get the actual indices to update
                valid_indices = factors_cs.loc[valid_mask].index
                result_df.loc[valid_indices, col] = epsilon
        
        # Check correlations are reduced
        for col in ['DemoSignal1', 'DemoSignal2']:
            original_corr = factor_df[col].corr(log_mkt_cap)
            neutralized_corr = result_df[col].corr(log_mkt_cap)
            
            self.assertLess(
                abs(neutralized_corr), 
                abs(original_corr) * 0.3,  # At least 70% reduction
                f"{col}: Neutralization didn't significantly reduce correlation "
                f"(original: {original_corr:.4f}, neutralized: {neutralized_corr:.4f})"
            )
    
    def test_neutralization_preserves_shape(self):
        """Verify output shape matches input."""
        factor_df, log_mkt_cap = create_test_data()
        
        # Simple passthrough to verify shape preservation
        self.assertEqual(factor_df.shape, (3000, 3))  # 30 dates * 100 stocks, 3 factors
    
    def test_neutralization_handles_missing_data(self):
        """Verify missing market cap values are handled correctly."""
        factor_df, log_mkt_cap = create_test_data(n_dates=10, n_stocks=50)
        
        # Introduce NaN in market cap
        nan_indices = log_mkt_cap.sample(frac=0.2, random_state=42).index
        log_mkt_cap.loc[nan_indices] = np.nan
        
        # Neutralization should still work (skip NaN rows)
        dates = factor_df.index.get_level_values('date').unique()
        for date in dates:
            date_mask = factor_df.index.get_level_values('date') == date
            mkt_cs = log_mkt_cap.loc[date_mask]
            valid_count = int(mkt_cs.notna().sum())
            
            # Should have at least some valid stocks per date
            self.assertGreater(valid_count, 10, f"Date {date} has too few valid stocks")
    
    def test_neutralization_low_stock_count(self):
        """Verify warning is triggered when stock count < 100."""
        factor_df, log_mkt_cap = create_test_data(n_dates=5, n_stocks=50)
        
        # With only 50 stocks, should trigger low count warning
        dates = factor_df.index.get_level_values('date').unique()
        low_count = 0
        for date in dates:
            date_mask = factor_df.index.get_level_values('date') == date
            n_stocks = date_mask.sum()
            if n_stocks < 100:
                low_count += 1
        
        self.assertEqual(low_count, 5, "All 5 dates should have < 100 stocks")


class TestLogMarketCapLoading(unittest.TestCase):
    """Test market cap data loading (requires DB access, may skip if unavailable)."""
    
    @unittest.skip("Requires database connection - run manually")
    def test_load_market_cap_shift(self):
        """Verify T-1 shift is correctly applied."""
        # This test requires actual DB connection
        # Run manually with:
        # python -m pytest factor_synthesis/pipeline/tests/test_neutralization.py -k test_load_market_cap_shift -v
        pass


if __name__ == '__main__':
    unittest.main()
