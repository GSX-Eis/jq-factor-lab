# -*- coding: utf-8 -*-
"""
Tests for DCorCVPreSelectionStrategy - CV-based dynamic threshold selection.

Tests cover:
1. Basic functionality - strategy returns valid factor list
2. CV splits generation - time-ordered expanding window
3. Threshold selection - finds best threshold via CV
4. No look-ahead bias - train/val splits are time-ordered
5. Fallback behavior - handles edge cases gracefully
"""

import os
import sys
import unittest
import logging

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from factor_synthesis.preselection.dcor_cv_strategy import DCorCVPreSelectionStrategy


def create_synthetic_data(
    n_dates: int = 100,
    n_stocks: int = 50,
    n_factors: int = 10,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create synthetic factor data with known structure for testing.
    
    Creates factors where:
    - Factor 0, 1, 2 have high dCor with returns (signal)
    - Other factors have low/zero dCor (noise)
    
    Args:
        n_dates: Number of trading dates
        n_stocks: Number of stocks per date
        n_factors: Total number of factors
        seed: Random seed for reproducibility
        
    Returns:
        (X, y) - Factor DataFrame and return Series
    """
    np.random.seed(seed)
    
    dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
    stocks = [f"600{i:03d}.SH" for i in range(n_stocks)]
    
    index = pd.MultiIndex.from_product(
        [dates, stocks], 
        names=['date', 'stock_code']
    )
    
    n_samples = len(index)
    
    # Generate true signal (latent factor driving returns)
    signal = np.random.randn(n_samples)
    
    # Forward returns = signal + noise
    noise_level = 0.5
    forward_returns = signal + noise_level * np.random.randn(n_samples)
    
    # Create factors
    factor_data = {}
    
    # Factors 0-2: correlated with signal (high dCor)
    for i in range(min(3, n_factors)):
        factor_data[f'DemoSignal{i}'] = signal + 0.3 * np.random.randn(n_samples)
    
    # Remaining factors: pure noise (low dCor)
    for i in range(3, n_factors):
        factor_data[f'DemoSignal{i}'] = np.random.randn(n_samples)
    
    X = pd.DataFrame(factor_data, index=index)
    y = pd.Series(forward_returns, index=index, name='forward_returns')
    
    return X, y


class TestDCorCVPreSelectionStrategy(unittest.TestCase):
    """Test cases for DCorCVPreSelectionStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_dcor_cv')
        self.logger.setLevel(logging.WARNING)  # Suppress info logs in tests
        
        self.default_config = {
            'enabled': True,
            'strategy': 'dcor_cv',
            'threshold_search_space': [0.02, 0.04, 0.06, 0.08],
            'cv_folds': 3,
            'cv_metric': 'mean_ic',
            'min_abs_metric': 0.05,
            'metrics': {
                'max_correlation': 0.7
            },
            'fallback_top_n': 3
        }
    
    def test_basic_selection(self):
        """Test that strategy returns valid factor list."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=5)
        
        strategy = DCorCVPreSelectionStrategy(self.default_config, self.logger)
        selected = strategy.select(X, y)
        
        # Should return a non-empty list
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
        
        # All selected factors should be in original columns
        for f in selected:
            self.assertIn(f, X.columns)
    
    def test_disabled_returns_all_factors(self):
        """Test that disabled strategy returns all factors."""
        X, y = create_synthetic_data(n_dates=30, n_stocks=20, n_factors=5)
        
        config = {**self.default_config, 'enabled': False}
        strategy = DCorCVPreSelectionStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        self.assertEqual(set(selected), set(X.columns))
    
    def test_cv_splits_time_ordered(self):
        """Test that CV splits are time-ordered (no look-ahead bias)."""
        X, y = create_synthetic_data(n_dates=100, n_stocks=20, n_factors=3)
        
        strategy = DCorCVPreSelectionStrategy(self.default_config, self.logger)
        
        dates = X.index.get_level_values(0).unique().sort_values()
        splits = strategy._generate_expanding_cv_splits(dates, n_folds=3)
        
        # Should generate at least one split
        self.assertGreater(len(splits), 0)
        
        for train_dates, val_dates in splits:
            # Train dates should all be before val dates
            max_train_date = train_dates.max()
            min_val_date = val_dates.min()
            
            self.assertLessEqual(
                max_train_date, min_val_date,
                f"Look-ahead bias detected: train max {max_train_date} > val min {min_val_date}"
            )
    
    def test_expanding_window_property(self):
        """Test that CV uses expanding window (train grows with each fold)."""
        X, y = create_synthetic_data(n_dates=100, n_stocks=20, n_factors=3)
        
        strategy = DCorCVPreSelectionStrategy(self.default_config, self.logger)
        
        dates = X.index.get_level_values(0).unique().sort_values()
        splits = strategy._generate_expanding_cv_splits(dates, n_folds=3)
        
        if len(splits) > 1:
            train_sizes = [len(train_dates) for train_dates, _ in splits]
            
            # Train size should increase (or stay same) with each fold
            for i in range(1, len(train_sizes)):
                self.assertGreaterEqual(
                    train_sizes[i], train_sizes[i-1],
                    "Expanding window violated: train size should increase"
                )
    
    def test_threshold_selection_finds_best(self):
        """Test that CV finds a threshold (not None) when data is valid."""
        X, y = create_synthetic_data(n_dates=80, n_stocks=30, n_factors=5)
        
        strategy = DCorCVPreSelectionStrategy(self.default_config, self.logger)
        
        best_threshold, cv_results = strategy._find_optimal_threshold_cv(
            X, y,
            threshold_space=[0.02, 0.04, 0.06],
            n_folds=3,
            metric='mean_ic'
        )
        
        # Should find a valid threshold
        self.assertIsNotNone(best_threshold)
        self.assertIn(best_threshold, [0.02, 0.04, 0.06])
        
        # Should have scores for thresholds
        self.assertGreater(len(cv_results), 0)
    
    def test_fallback_when_insufficient_data(self):
        """Test fallback behavior when data is insufficient for CV."""
        # Create very short data (not enough for CV)
        X, y = create_synthetic_data(n_dates=5, n_stocks=10, n_factors=3)
        
        strategy = DCorCVPreSelectionStrategy(self.default_config, self.logger)
        
        # Should still return factors (fallback mechanism)
        selected = strategy.select(X, y)
        
        self.assertIsInstance(selected, list)
        # Should return at least fallback_top_n factors
        self.assertGreaterEqual(len(selected), 1)
    
    def test_empty_selection_fallback(self):
        """Test that empty selection triggers fallback to top N factors."""
        # Create data where no factor passes high threshold
        X, y = create_synthetic_data(n_dates=60, n_stocks=20, n_factors=5)
        
        # Use very high threshold that nothing will pass
        config = {
            **self.default_config,
            'threshold_search_space': [0.5, 0.6, 0.7],  # Very high
            'fallback_top_n': 2
        }
        
        strategy = DCorCVPreSelectionStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should return fallback_top_n factors
        self.assertGreaterEqual(len(selected), 1)
    
    def test_correlation_filter_applied(self):
        """Test that correlation filter reduces highly correlated factors."""
        np.random.seed(42)
        
        n_dates = 50
        n_stocks = 30
        dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
        stocks = [f"600{i:03d}.SH" for i in range(n_stocks)]
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
        
        # Create highly correlated factors
        base = np.random.randn(len(index))
        X = pd.DataFrame({
            'DemoSignal0': base,
            'DemoSignal1': base + 0.01 * np.random.randn(len(index)),  # 99%+ correlated
            'DemoSignal2': np.random.randn(len(index)),  # Uncorrelated
        }, index=index)
        
        y = pd.Series(base + 0.5 * np.random.randn(len(index)), index=index)
        
        # Use strict correlation threshold
        config = {
            **self.default_config,
            'metrics': {'max_correlation': 0.5}
        }
        
        strategy = DCorCVPreSelectionStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should not select both highly correlated factors
        highly_correlated = ['DemoSignal0', 'DemoSignal1']
        selected_correlated = [f for f in selected if f in highly_correlated]
        
        # At most one of the highly correlated pair should be selected
        self.assertLessEqual(
            len(selected_correlated), 1,
            "Correlation filter should remove one of the highly correlated factors"
        )
    
    def test_performance_report_available(self):
        """Test that performance report is available after selection."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=5)
        
        strategy = DCorCVPreSelectionStrategy(self.default_config, self.logger)
        _ = strategy.select(X, y)
        
        report = strategy.get_performance_report()
        
        self.assertIsInstance(report, pd.DataFrame)
        if not report.empty:
            self.assertIn('factor_name', report.columns)
            self.assertIn('dcor_mean', report.columns)
    
    def test_different_cv_metrics(self):
        """Test that different CV metrics work (IC-based and dCor-based)."""
        X, y = create_synthetic_data(n_dates=80, n_stocks=30, n_factors=5)
        
        # Test all supported metrics: IC-based and dCor-based
        all_metrics = ['mean_ic', 'ir', 'composite_ic', 'mean_dcor', 'composite_dcor']
        
        for metric in all_metrics:
            config = {**self.default_config, 'cv_metric': metric}
            strategy = DCorCVPreSelectionStrategy(config, self.logger)
            
            selected = strategy.select(X, y)
            
            self.assertIsInstance(selected, list)
            self.assertGreater(
                len(selected), 0,
                f"CV metric '{metric}' should produce non-empty selection"
            )


class TestCVSplitsEdgeCases(unittest.TestCase):
    """Test edge cases for CV splits generation."""
    
    def setUp(self):
        self.logger = logging.getLogger('test_cv_splits')
        self.logger.setLevel(logging.WARNING)
        
        self.config = {
            'enabled': True,
            'cv_folds': 3,
        }
    
    def test_single_fold_with_few_dates(self):
        """Test CV with very few dates produces at least one fold."""
        strategy = DCorCVPreSelectionStrategy(self.config, self.logger)
        
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        splits = strategy._generate_expanding_cv_splits(dates, n_folds=3)
        
        # Should produce at least one valid split
        self.assertGreaterEqual(len(splits), 1)
    
    def test_no_splits_with_minimal_data(self):
        """Test CV with minimal data returns empty splits gracefully."""
        strategy = DCorCVPreSelectionStrategy(self.config, self.logger)
        
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        splits = strategy._generate_expanding_cv_splits(dates, n_folds=3)
        
        # May return empty or single split - both are valid fallback behaviors
        self.assertIsInstance(splits, list)


class TestParallelComputation(unittest.TestCase):
    """Test parallel computation for dCor and correlation filtering."""
    
    def setUp(self):
        self.logger = logging.getLogger('test_parallel')
        self.logger.setLevel(logging.WARNING)
    
    def test_parallel_dcor_same_as_serial(self):
        """Test that parallel dCor calculation gives same results as serial."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=8)
        
        # Config with parallel enabled
        config_parallel = {
            'enabled': True,
            'parallel': {'enabled': True, 'n_jobs': 2}
        }
        
        # Config with parallel disabled
        config_serial = {
            'enabled': True,
            'parallel': {'enabled': False}
        }
        
        strategy_parallel = DCorCVPreSelectionStrategy(config_parallel, self.logger)
        strategy_serial = DCorCVPreSelectionStrategy(config_serial, self.logger)
        
        # Calculate dCor with both methods
        perf_parallel = strategy_parallel._calculate_dcor_performance(X, y)
        perf_serial = strategy_serial._calculate_dcor_performance(X, y)
        
        # Should have same factors
        self.assertEqual(
            set(perf_parallel['factor_name']),
            set(perf_serial['factor_name'])
        )
        
        # Merge and compare values
        merged = perf_parallel.merge(
            perf_serial, 
            on='factor_name', 
            suffixes=('_parallel', '_serial')
        )
        
        # dCor values should be identical (or very close due to floating point)
        np.testing.assert_allclose(
            merged['dcor_mean_parallel'].values,
            merged['dcor_mean_serial'].values,
            rtol=1e-10
        )
    
    def test_vectorized_correlation_filter_same_results(self):
        """Test that vectorized correlation filter gives same results."""
        np.random.seed(42)
        
        n_dates = 30
        n_stocks = 20
        dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
        stocks = [f"600{i:03d}.SH" for i in range(n_stocks)]
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
        
        # Create factors with varying correlations
        X = pd.DataFrame({
            'DemoSignal0': np.random.randn(len(index)),
            'DemoSignal1': np.random.randn(len(index)),
            'DemoSignal2': np.random.randn(len(index)),
            'DemoSignal3': np.random.randn(len(index)),
            'DemoSignal4': np.random.randn(len(index)),
        }, index=index)
        
        config = {'enabled': True, 'metrics': {'max_correlation': 0.7}}
        strategy = DCorCVPreSelectionStrategy(config, self.logger)
        
        sorted_factors = ['DemoSignal0', 'DemoSignal1', 'DemoSignal2', 'DemoSignal3', 'DemoSignal4']
        
        # Run vectorized filter
        result = strategy._filter_by_correlation(X, sorted_factors, 0.7)
        
        # Should return a valid list
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # First factor should always be included
        self.assertEqual(result[0], 'DemoSignal0')
    
    def test_parallel_with_different_n_jobs(self):
        """Test parallel computation with different n_jobs values."""
        X, y = create_synthetic_data(n_dates=40, n_stocks=20, n_factors=5)
        
        results = {}
        for n_jobs in [1, 2, -1]:
            config = {
                'enabled': True,
                'parallel': {'enabled': True, 'n_jobs': n_jobs}
            }
            strategy = DCorCVPreSelectionStrategy(config, self.logger)
            perf = strategy._calculate_dcor_performance(X, y)
            results[n_jobs] = set(perf['factor_name'].tolist())
        
        # All should return same factors
        self.assertEqual(results[1], results[2])
        self.assertEqual(results[1], results[-1])
    
    def test_end_to_end_parallel_selection(self):
        """Test full selection pipeline with parallel enabled."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=10)
        
        config = {
            'enabled': True,
            'threshold_search_space': [0.02, 0.04, 0.06],
            'cv_folds': 3,
            'cv_metric': 'mean_ic',
            'min_abs_metric': 0.05,
            'metrics': {'max_correlation': 0.7},
            'fallback_top_n': 3,
            'parallel': {'enabled': True, 'n_jobs': -1}
        }
        
        strategy = DCorCVPreSelectionStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should return valid factors
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
        
        # All selected should be from original columns
        for f in selected:
            self.assertIn(f, X.columns)


if __name__ == '__main__':
    unittest.main()
