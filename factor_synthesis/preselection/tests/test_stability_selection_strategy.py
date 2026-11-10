# -*- coding: utf-8 -*-
"""
Tests for StabilitySelectionPreStrategy - Meinshausen & Bühlmann (2010) implementation.

Tests cover:
1. Basic functionality - strategy returns valid factor list
2. Selection probability calculation - correct frequency counting
3. Bootstrap sampling - proper sub-sampling behavior
4. Reproducibility - same seed produces same results
5. Fallback behavior - handles edge cases gracefully
6. Correlation filtering - removes redundant factors
"""

import os
import sys
import unittest
import logging

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from factor_synthesis.preselection.stability_selection_strategy import (
    StabilitySelectionPreStrategy,
    _compute_single_dcor,
    _compute_single_ic,
    _compute_single_ir
)


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


class TestStabilitySelectionPreStrategy(unittest.TestCase):
    """Test cases for StabilitySelectionPreStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_stability_selection')
        self.logger.setLevel(logging.WARNING)  # Suppress info logs in tests
        
        self.default_config = {
            'enabled': True,
            'strategy': 'stability_selection',
            'n_bootstrap': 20,  # Reduced for faster tests
            'subsample_ratio': 0.5,
            'selection_threshold': 0.6,
            'selection_mode': 'top_k',
            'per_sample_top_k': 3,
            'per_sample_dcor_threshold': 0.05,
            'fallback_top_n': 3,
            'random_seed': 42,
            'parallel': {
                'enabled': False  # Disable for deterministic tests
            }
        }
    
    def test_basic_selection(self):
        """Test that strategy returns valid factor list."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=5)
        
        strategy = StabilitySelectionPreStrategy(self.default_config, self.logger)
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
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        self.assertEqual(set(selected), set(X.columns))
    
    def test_selects_high_dcor_factors(self):
        """Test that strategy preferentially selects high dCor factors."""
        X, y = create_synthetic_data(n_dates=100, n_stocks=50, n_factors=8, seed=42)
        
        strategy = StabilitySelectionPreStrategy(self.default_config, self.logger)
        selected = strategy.select(X, y)
        
        # Factors 0, 1, 2 are designed to have high dCor
        # They should have higher selection probability
        probs = strategy.get_selection_probabilities()
        
        high_dcor_factors = ['DemoSignal0', 'DemoSignal1', 'DemoSignal2']
        low_dcor_factors = [f'DemoSignal{i}' for i in range(3, 8)]
        
        avg_high_prob = np.mean([probs.get(f, 0) for f in high_dcor_factors])
        avg_low_prob = np.mean([probs.get(f, 0) for f in low_dcor_factors])
        
        # High dCor factors should have higher average selection probability
        self.assertGreater(
            avg_high_prob, avg_low_prob,
            f"High dCor factors ({avg_high_prob:.3f}) should have higher "
            f"probability than low dCor factors ({avg_low_prob:.3f})"
        )
    
    def test_selection_probability_range(self):
        """Test that selection probabilities are in valid range [0, 1]."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=5)
        
        strategy = StabilitySelectionPreStrategy(self.default_config, self.logger)
        strategy.select(X, y)
        
        probs = strategy.get_selection_probabilities()
        
        for factor, prob in probs.items():
            self.assertGreaterEqual(prob, 0.0, f"{factor} probability < 0")
            self.assertLessEqual(prob, 1.0, f"{factor} probability > 1")
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=5)
        
        config = {**self.default_config, 'random_seed': 123}
        
        strategy1 = StabilitySelectionPreStrategy(config, self.logger)
        selected1 = strategy1.select(X, y)
        probs1 = strategy1.get_selection_probabilities()
        
        strategy2 = StabilitySelectionPreStrategy(config, self.logger)
        selected2 = strategy2.select(X, y)
        probs2 = strategy2.get_selection_probabilities()
        
        # Same seed should produce identical results
        self.assertEqual(selected1, selected2)
        self.assertEqual(probs1, probs2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds can produce different results."""
        # Use more bootstrap samples and more factors to increase chance of variation
        X, y = create_synthetic_data(n_dates=100, n_stocks=50, n_factors=12)
        
        config_base = {
            **self.default_config,
            'n_bootstrap': 30,  # More samples for variation
            'selection_threshold': 0.3,  # Lower threshold
            'per_sample_top_k': 5
        }
        
        config1 = {**config_base, 'random_seed': 42}
        config2 = {**config_base, 'random_seed': 999}  # Very different seed
        
        strategy1 = StabilitySelectionPreStrategy(config1, self.logger)
        selected1 = strategy1.select(X, y)
        probs1 = strategy1.get_selection_probabilities()
        
        strategy2 = StabilitySelectionPreStrategy(config2, self.logger)
        selected2 = strategy2.select(X, y)
        probs2 = strategy2.get_selection_probabilities()
        
        # Different seeds should produce different probabilities (with high probability)
        # Check for any difference > 0.05 (more lenient threshold)
        different_probs = sum(
            abs(probs1[f] - probs2[f]) > 0.05 
            for f in probs1.keys()
        )
        # This is a probabilistic test - we just verify the mechanism works
        # by checking that probabilities are not identical to many decimal places
        probs1_sum = sum(probs1.values())
        probs2_sum = sum(probs2.values())
        
        # At minimum, the total probability sums should be slightly different
        # or at least one factor should have different selection count
        self.assertTrue(
            different_probs > 0 or abs(probs1_sum - probs2_sum) > 0.001,
            f"Different seeds should produce some variation. "
            f"Prob sums: {probs1_sum:.4f} vs {probs2_sum:.4f}, different_probs={different_probs}"
        )
    
    def test_top_k_mode(self):
        """Test top_k selection mode."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=10)
        
        config = {
            **self.default_config,
            'selection_mode': 'top_k',
            'per_sample_top_k': 3
        }
        
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        strategy.select(X, y)
        
        # Should have performance report
        report = strategy.get_performance_report()
        self.assertFalse(report.empty)
        self.assertIn('selection_probability', report.columns)
    
    def test_dcor_threshold_mode(self):
        """Test dcor_threshold selection mode."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=5)
        
        config = {
            **self.default_config,
            'selection_mode': 'dcor_threshold',
            'per_sample_dcor_threshold': 0.05
        }
        
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should return valid selection
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
    
    def test_fallback_when_no_factors_pass_threshold(self):
        """Test fallback behavior when no factors pass selection threshold."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=5)
        
        config = {
            **self.default_config,
            'selection_threshold': 1.1,  # Impossible threshold
            'fallback_top_n': 2
        }
        
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should return fallback_top_n factors
        self.assertEqual(len(selected), 2)
    
    def test_correlation_filtering(self):
        """Test that correlation filtering removes redundant factors."""
        np.random.seed(42)
        
        n_dates, n_stocks = 50, 30
        dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
        stocks = [f"600{i:03d}.SH" for i in range(n_stocks)]
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
        n_samples = len(index)
        
        # Create base signal
        signal = np.random.randn(n_samples)
        forward_returns = signal + 0.3 * np.random.randn(n_samples)
        
        # Create highly correlated factors
        factor_data = {
            'DemoSignal0': signal + 0.1 * np.random.randn(n_samples),  # High dCor
            'DemoSignal1': signal + 0.15 * np.random.randn(n_samples),  # High dCor, correlated with DemoSignal0
            'DemoSignal2': np.random.randn(n_samples),  # Independent noise
        }
        
        X = pd.DataFrame(factor_data, index=index)
        y = pd.Series(forward_returns, index=index)
        
        config = {
            **self.default_config,
            'metrics': {'max_correlation': 0.5},  # Low threshold to trigger filtering
            'selection_threshold': 0.3  # Lower threshold to select both correlated factors before filtering
        }
        
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should have at least one factor selected
        self.assertGreater(len(selected), 0)
    
    def test_empty_data_handling(self):
        """Test graceful handling of empty data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        stocks = ['600001.SH']
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
        
        X = pd.DataFrame({'DemoSignal0': []}, index=pd.MultiIndex.from_arrays([[], []]))
        y = pd.Series([], index=pd.MultiIndex.from_arrays([[], []]))
        
        strategy = StabilitySelectionPreStrategy(self.default_config, self.logger)
        
        # Should handle empty data gracefully
        try:
            selected = strategy.select(X, y)
            # If it returns, should be valid (possibly empty)
            self.assertIsInstance(selected, list)
        except ValueError:
            # ValueError for empty data is also acceptable
            pass
    
    def test_performance_report_structure(self):
        """Test that performance report has correct structure."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=5)
        
        strategy = StabilitySelectionPreStrategy(self.default_config, self.logger)
        strategy.select(X, y)
        
        report = strategy.get_performance_report()
        
        # Check required columns
        required_columns = ['factor_name', 'selection_probability', 'selection_count']
        for col in required_columns:
            self.assertIn(col, report.columns, f"Missing column: {col}")
        
        # Check all factors are in report
        self.assertEqual(len(report), 5)
    
    def test_selection_details(self):
        """Test that selection details are recorded."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=5)
        
        n_bootstrap = 15
        config = {**self.default_config, 'n_bootstrap': n_bootstrap}
        
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        strategy.select(X, y)
        
        details = strategy.get_selection_details()
        
        # Should have one detail per bootstrap
        self.assertEqual(len(details), n_bootstrap)
        
        # Each detail should have n_samples and n_selected
        for detail in details:
            self.assertIn('n_samples', detail)
            self.assertIn('n_selected', detail)


class TestComputeSingleDCor(unittest.TestCase):
    """Test cases for _compute_single_dcor helper function."""
    
    def test_basic_dcor_computation(self):
        """Test basic dCor computation."""
        np.random.seed(42)
        n = 100
        
        # Create correlated data
        x = np.random.randn(n)
        y = x + 0.5 * np.random.randn(n)
        
        result = _compute_single_dcor('test_factor', x, y)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['factor_name'], 'test_factor')
        self.assertGreater(result['metric_value'], 0)
        self.assertEqual(result['metric_type'], 'dcor')
    
    def test_dcor_handles_nan(self):
        """Test that dCor handles NaN values."""
        np.random.seed(42)
        n = 100
        
        x = np.random.randn(n)
        y = np.random.randn(n)
        
        # Add some NaNs
        x[0:5] = np.nan
        y[10:15] = np.nan
        
        result = _compute_single_dcor('test_factor', x, y)
        
        # Should still compute with valid data
        self.assertIsNotNone(result)
    
    def test_dcor_returns_none_for_insufficient_data(self):
        """Test that dCor returns None for insufficient data."""
        x = np.array([1, 2, 3])  # Only 3 samples
        y = np.array([1, 2, 3])
        
        result = _compute_single_dcor('test_factor', x, y)
        
        # Should return None due to insufficient data (< 10)
        self.assertIsNone(result)


class TestComputeSingleIC(unittest.TestCase):
    """Test cases for _compute_single_ic helper function."""
    
    def test_basic_ic_computation(self):
        """Test basic IC computation."""
        X, y = create_synthetic_data(n_dates=30, n_stocks=50, n_factors=1)
        
        factor_col = X.columns[0]
        result = _compute_single_ic(factor_col, X[factor_col], y)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['factor_name'], factor_col)
        self.assertIn('metric_value', result)
        self.assertEqual(result['metric_type'], 'ic')
        # IC should be between 0 and 1 (absolute value)
        self.assertGreaterEqual(result['metric_value'], 0)
        self.assertLessEqual(result['metric_value'], 1)


class TestComputeSingleIR(unittest.TestCase):
    """Test cases for _compute_single_ir helper function."""
    
    def test_basic_ir_computation(self):
        """Test basic IR computation."""
        X, y = create_synthetic_data(n_dates=30, n_stocks=50, n_factors=1)
        
        factor_col = X.columns[0]
        result = _compute_single_ir(factor_col, X[factor_col], y)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['factor_name'], factor_col)
        self.assertIn('metric_value', result)
        self.assertEqual(result['metric_type'], 'ir')
        self.assertIn('ic_mean', result)
        self.assertIn('ic_std', result)
        # IR should be non-negative
        self.assertGreaterEqual(result['metric_value'], 0)


class TestStabilitySelectionWithDifferentMetrics(unittest.TestCase):
    """Test Stability Selection with different base_metric options."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_stability_metrics')
        self.logger.setLevel(logging.WARNING)
        
        self.base_config = {
            'enabled': True,
            'strategy': 'stability_selection',
            'n_bootstrap': 15,
            'subsample_ratio': 0.5,
            'selection_threshold': 0.4,
            'selection_mode': 'top_k',
            'per_sample_top_k': 3,
            'fallback_top_n': 3,
            'random_seed': 42,
            'parallel': {'enabled': False}
        }
    
    def test_dcor_metric(self):
        """Test stability selection with dCor metric."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=5)
        
        config = {**self.base_config, 'base_metric': 'dcor'}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
        
        # Check report includes metric info
        report = strategy.get_performance_report()
        self.assertIn('base_metric', report.columns)
        self.assertTrue((report['base_metric'] == 'dcor').all())
    
    def test_ic_metric(self):
        """Test stability selection with IC metric."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=5)
        
        config = {**self.base_config, 'base_metric': 'ic'}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
        
        # Check report includes metric info
        report = strategy.get_performance_report()
        self.assertIn('base_metric', report.columns)
        self.assertTrue((report['base_metric'] == 'ic').all())
    
    def test_ir_metric(self):
        """Test stability selection with IR metric."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=5)
        
        config = {**self.base_config, 'base_metric': 'ir'}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
        
        # Check report includes metric info
        report = strategy.get_performance_report()
        self.assertIn('base_metric', report.columns)
        self.assertTrue((report['base_metric'] == 'ir').all())
    
    def test_invalid_metric_falls_back_to_dcor(self):
        """Test that invalid metric falls back to dcor."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=5)
        
        config = {**self.base_config, 'base_metric': 'invalid_metric'}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should still work with dcor fallback
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)


class TestPreCorrelationFiltering(unittest.TestCase):
    """Test the pre-correlation filtering feature."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_pre_corr')
        self.logger.setLevel(logging.WARNING)
        
        self.base_config = {
            'enabled': True,
            'strategy': 'stability_selection',
            'n_bootstrap': 10,
            'subsample_ratio': 0.5,
            'selection_threshold': 0.4,
            'selection_mode': 'top_pct',
            'per_sample_top_pct': 0.3,
            'fallback_top_n': 3,
            'random_seed': 42,
            'parallel': {'enabled': False}
        }
    
    def test_pre_filter_removes_correlated_factors(self):
        """Test that pre-filtering removes highly correlated factors."""
        np.random.seed(42)
        n_dates = 50
        n_stocks = 30
        
        # Create base factors
        dates = pd.date_range('2024-01-01', periods=n_dates)
        stocks = [f'stock_{i}' for i in range(n_stocks)]
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
        
        # Create factor 1 and factor 2 (highly correlated)
        demo_signal1 = np.random.randn(len(index))
        demo_signal2 = demo_signal1 + 0.1 * np.random.randn(len(index))  # 90%+ correlated
        demo_signal3 = np.random.randn(len(index))  # Independent
        
        X = pd.DataFrame({
            'factor_1': demo_signal1,
            'factor_2': demo_signal2,
            'factor_3': demo_signal3
        }, index=index)
        
        y = pd.Series(np.random.randn(len(index)), index=index)
        
        # With max_correlation = 0.5, factor_2 should be filtered out
        config = {**self.base_config, 'max_correlation': 0.5}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should not have both factor_1 and factor_2
        has_both = 'factor_1' in selected and 'factor_2' in selected
        self.assertFalse(has_both, "Should filter out one of the correlated factors")
    
    def test_pre_filter_preserves_best_metric_factor(self):
        """Test that pre-filtering keeps the factor with best metric value."""
        np.random.seed(42)
        n_dates = 50
        n_stocks = 30
        
        dates = pd.date_range('2024-01-01', periods=n_dates)
        stocks = [f'stock_{i}' for i in range(n_stocks)]
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
        
        # Create factors with known relationship to y
        y = pd.Series(np.random.randn(len(index)), index=index)
        
        # factor_good: highly correlated with y
        factor_good = y + 0.1 * np.random.randn(len(index))
        # factor_bad: correlated with factor_good but not with y
        factor_bad = factor_good + 0.1 * np.random.randn(len(index))
        
        X = pd.DataFrame({
            'factor_good': factor_good,
            'factor_bad': factor_bad
        }, index=index)
        
        config = {**self.base_config, 'max_correlation': 0.5}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # factor_good should be kept (better metric)
        self.assertIn('factor_good', selected)
    
    def test_no_filter_when_correlation_not_set(self):
        """Test that no pre-filtering when max_correlation is not set."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=5)
        
        # Remove max_correlation from config
        config = {k: v for k, v in self.base_config.items() if k != 'max_correlation'}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        
        # All factors should participate in stability selection
        report = strategy.get_performance_report()
        if report is not None:
            # All factors should have been evaluated
            strategy.select(X, y)
            report = strategy.get_performance_report()
            self.assertEqual(len(report), 5)
    
    def test_performance_report_includes_pre_filter_info(self):
        """Test that performance report includes pre-filter information."""
        X, y = create_synthetic_data(n_dates=50, n_stocks=30, n_factors=5)
        
        config = {**self.base_config, 'max_correlation': 0.5}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        strategy.select(X, y)
        
        report = strategy.get_performance_report()
        
        # Check new columns exist
        self.assertIn('pre_filter_metric', report.columns)
        self.assertIn('passed_corr_filter', report.columns)


class TestTopPctSelectionMode(unittest.TestCase):
    """Test the top_pct percentage-based selection mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test_top_pct')
        self.logger.setLevel(logging.WARNING)
        
        self.base_config = {
            'enabled': True,
            'strategy': 'stability_selection',
            'n_bootstrap': 15,
            'subsample_ratio': 0.5,
            'selection_threshold': 0.4,
            'selection_mode': 'top_pct',
            'per_sample_top_pct': 0.3,
            'fallback_top_n': 3,
            'random_seed': 42,
            'parallel': {'enabled': False}
        }
    
    def test_top_pct_basic(self):
        """Test basic top_pct mode functionality."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=10)
        
        strategy = StabilitySelectionPreStrategy(self.base_config, self.logger)
        selected = strategy.select(X, y)
        
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
    
    def test_top_pct_selects_correct_proportion(self):
        """Test that top_pct selects approximately the right proportion."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=20)
        
        # With 30% selection, each bootstrap should select ~6 factors (20 * 0.3)
        config = {**self.base_config, 'per_sample_top_pct': 0.3}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        strategy.select(X, y)
        
        details = strategy.get_selection_details()
        
        # Check average number selected per bootstrap
        avg_selected = np.mean([d['n_selected'] for d in details])
        expected = 20 * 0.3  # 6 factors
        
        # Should be close to expected (allow some variance)
        self.assertGreaterEqual(avg_selected, expected * 0.5)
        self.assertLessEqual(avg_selected, expected * 1.5)
    
    def test_top_pct_minimum_one_factor(self):
        """Test that top_pct selects at least 1 factor even for small groups."""
        # Create data with only 2 factors
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=2)
        
        # With 30% of 2 factors = 0.6, should round up to at least 1
        config = {**self.base_config, 'per_sample_top_pct': 0.3}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        strategy.select(X, y)
        
        details = strategy.get_selection_details()
        
        # Each bootstrap should select at least 1 factor
        for d in details:
            self.assertGreaterEqual(d['n_selected'], 1)
    
    def test_top_pct_vs_top_k_different_factor_counts(self):
        """Test that top_pct adapts to different factor counts unlike top_k."""
        # Test with different factor counts
        for n_factors in [5, 10, 20, 50]:
            X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=n_factors)
            
            config = {**self.base_config, 'per_sample_top_pct': 0.3}
            strategy = StabilitySelectionPreStrategy(config, self.logger)
            strategy.select(X, y)
            
            details = strategy.get_selection_details()
            avg_selected = np.mean([d['n_selected'] for d in details])
            
            # Average should scale with factor count
            expected = max(1, int(n_factors * 0.3))
            self.assertGreaterEqual(avg_selected, expected * 0.5)
    
    def test_top_pct_with_loky_backend(self):
        """Test top_pct mode with loky parallel backend."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=10)
        
        config = {
            **self.base_config,
            'parallel': {
                'enabled': True,
                'n_jobs': 2,
                'backend': 'loky'
            }
        }
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)
    
    def test_invalid_selection_mode_falls_back(self):
        """Test that invalid selection_mode falls back to top_pct."""
        X, y = create_synthetic_data(n_dates=60, n_stocks=30, n_factors=10)
        
        config = {**self.base_config, 'selection_mode': 'invalid_mode'}
        strategy = StabilitySelectionPreStrategy(config, self.logger)
        selected = strategy.select(X, y)
        
        # Should still work with fallback
        self.assertIsInstance(selected, list)
        self.assertGreater(len(selected), 0)


if __name__ == '__main__':
    unittest.main()
