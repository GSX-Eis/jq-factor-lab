# -*- coding: utf-8 -*-
"""
Tests for Dynamic Blend Module

Tests the dynamic blend ratio computation based on IC/ICIR.
"""

import numpy as np
import pandas as pd
import pytest

from factor_synthesis.pipeline.dynamic_blend import (
    DynamicBlendConfig,
    DynamicBlendResult,
    compute_rolling_ic_stats,
    compute_prior_composite,
    compute_dynamic_blend_ratio,
    blend_predictions,
)


class TestDynamicBlendConfig:
    """Tests for DynamicBlendConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DynamicBlendConfig()
        assert config.enabled is True
        assert config.base_ratio == 0.5
        assert config.max_delta == 0.3
        assert config.lookback_window == 60
        assert config.use_icir is False
        assert config.smoothing_scale == 0.1
        assert config.min_samples == 20
    
    def test_valid_custom_config(self):
        """Test custom valid configuration."""
        config = DynamicBlendConfig(
            base_ratio=0.6,
            max_delta=0.2,
            lookback_window=30,
            use_icir=True,
        )
        assert config.base_ratio == 0.6
        assert config.max_delta == 0.2
        assert config.lookback_window == 30
        assert config.use_icir is True
    
    def test_invalid_base_ratio_outside_unit_interval(self):
        """Test that base_ratio outside the unit interval raises error."""
        with pytest.raises(ValueError, match="base_ratio must be in"):
            DynamicBlendConfig(base_ratio=-0.1)

        with pytest.raises(ValueError, match="base_ratio must be in"):
            DynamicBlendConfig(base_ratio=1.1)
    
    def test_invalid_base_ratio_above_one(self):
        """Test that base_ratio > 1 raises error."""
        with pytest.raises(ValueError, match="base_ratio must be in"):
            DynamicBlendConfig(base_ratio=1.5)
    
    def test_invalid_max_delta(self):
        """Test that negative max_delta raises error."""
        with pytest.raises(ValueError, match="max_delta must be non-negative"):
            DynamicBlendConfig(max_delta=-0.1)
    
    def test_ratio_range_rejects_invalid_bounds(self):
        with pytest.raises(ValueError, match="must stay in"):
            DynamicBlendConfig(base_ratio=0.12, max_delta=0.15)

    def test_zero_and_one_base_ratios_are_allowed_with_zero_delta(self):
        for boundary in (0.0, 1.0):
            assert DynamicBlendConfig(base_ratio=boundary, max_delta=0).base_ratio == boundary
    
    def test_invalid_lookback_window(self):
        """Test that lookback_window < 1 raises error."""
        with pytest.raises(ValueError, match="lookback_window must be >= 1"):
            DynamicBlendConfig(lookback_window=0)
    
    def test_invalid_smoothing_scale(self):
        """Test that non-positive smoothing_scale raises error."""
        with pytest.raises(ValueError, match="smoothing_scale must be positive"):
            DynamicBlendConfig(smoothing_scale=0)


class TestComputeRollingICStats:
    """Tests for compute_rolling_ic_stats function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample predictions and returns data."""
        np.random.seed(42)
        n_stocks = 50
        n_days = 100
        
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        stocks = [f"stock_{i:03d}" for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product(
            [dates, stocks], names=["date", "stock_code"]
        )
        
        # Create predictions with some correlation to returns
        returns = pd.Series(np.random.randn(len(index)), index=index)
        predictions = returns * 0.3 + np.random.randn(len(index)) * 0.7
        predictions = pd.Series(predictions, index=index)
        
        return predictions, returns
    
    def test_ic_computation(self, sample_data):
        """Test that IC is computed correctly."""
        predictions, returns = sample_data
        
        ic_mean, ic_std, n_days = compute_rolling_ic_stats(
            predictions, returns, window=30, min_periods=10
        )
        
        # IC should be positive given the positive correlation in sample data
        assert ic_mean > 0
        assert ic_std > 0
        assert n_days > 0
    
    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        empty = pd.Series(dtype=float)
        returns = pd.Series(dtype=float)
        
        ic_mean, ic_std, n_days = compute_rolling_ic_stats(
            empty, returns, window=30, min_periods=10
        )
        
        assert ic_mean == 0.0
        assert ic_std == 1.0
        assert n_days == 0
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        stocks = ["A", "B", "C"]
        
        index = pd.MultiIndex.from_product(
            [dates, stocks], names=["date", "stock_code"]
        )
        
        predictions = pd.Series(np.random.randn(len(index)), index=index)
        returns = pd.Series(np.random.randn(len(index)), index=index)
        
        # min_periods=10 but only 5 days
        ic_mean, ic_std, n_days = compute_rolling_ic_stats(
            predictions, returns, window=30, min_periods=10
        )
        
        assert ic_mean == 0.0
        assert n_days < 10


class TestComputePriorComposite:
    """Tests for compute_prior_composite function."""
    
    @pytest.fixture
    def sample_factor_data(self):
        """Create sample factor data."""
        np.random.seed(42)
        n_stocks = 30
        n_days = 50
        
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        stocks = [f"stock_{i:03d}" for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product(
            [dates, stocks], names=["date", "stock_code"]
        )
        
        factor_df = pd.DataFrame({
            "DemoSignal1": np.random.randn(len(index)),
            "DemoSignal2": np.random.randn(len(index)),
        }, index=index)
        
        return factor_df, index
    
    def test_single_factor_composite(self, sample_factor_data):
        """Test composite with single factor."""
        factor_df, index = sample_factor_data
        
        composite = compute_prior_composite(
            factor_df=factor_df,
            blend_weights={"DemoSignal1": 1.0},
            index=index,
        )
        
        assert len(composite) == len(index)
        assert not composite.isna().all()
        
        # Check that composite is z-scored (approximately zero mean per date)
        daily_means = composite.groupby(level="date").mean()
        assert np.abs(daily_means).mean() < 0.1  # Should be close to 0
    
    def test_multi_factor_composite(self, sample_factor_data):
        """Test composite with multiple factors."""
        factor_df, index = sample_factor_data
        
        composite = compute_prior_composite(
            factor_df=factor_df,
            blend_weights={"DemoSignal1": 0.6, "DemoSignal2": 0.4},
            index=index,
        )
        
        assert len(composite) == len(index)
        assert not composite.isna().all()
    
    def test_missing_factor(self, sample_factor_data):
        """Test handling of missing factor."""
        factor_df, index = sample_factor_data
        
        # DemoSignal3 doesn't exist
        composite = compute_prior_composite(
            factor_df=factor_df,
            blend_weights={"DemoSignal1": 0.5, "DemoSignal3": 0.5},
            index=index,
        )
        
        # Should still work with just DemoSignal1
        assert len(composite) == len(index)


class TestComputeDynamicBlendRatio:
    """Tests for compute_dynamic_blend_ratio function."""
    
    @pytest.fixture
    def sample_blend_data(self):
        """Create sample data for blend ratio computation."""
        np.random.seed(42)
        n_stocks = 50
        n_days = 100
        
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        stocks = [f"stock_{i:03d}" for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product(
            [dates, stocks], names=["date", "stock_code"]
        )
        
        # Create forward returns
        forward_returns = pd.Series(np.random.randn(len(index)), index=index)
        
        # ML predictions with higher correlation to returns
        ml_predictions = forward_returns * 0.4 + np.random.randn(len(index)) * 0.6
        ml_predictions = pd.Series(ml_predictions, index=index)
        
        # Prior composite with lower correlation to returns
        prior_composite = forward_returns * 0.2 + np.random.randn(len(index)) * 0.8
        prior_composite = pd.Series(prior_composite, index=index)
        
        return ml_predictions, prior_composite, forward_returns
    
    def test_dynamic_blend_ml_better(self, sample_blend_data):
        """Test when ML predictions have higher IC."""
        ml_predictions, prior_composite, forward_returns = sample_blend_data
        
        config = DynamicBlendConfig(
            base_ratio=0.5,
            max_delta=0.3,
            lookback_window=60,
            min_samples=20,
        )
        
        result = compute_dynamic_blend_ratio(
            ml_predictions=ml_predictions,
            prior_composite=prior_composite,
            forward_returns=forward_returns,
            config=config,
        )
        
        assert isinstance(result, DynamicBlendResult)
        assert result.computation_successful
        
        # ML has higher IC, so blend_ratio should be < base_ratio
        # (more weight to ML means less weight to prior)
        assert result.blend_ratio < config.base_ratio
        assert result.adjustment < 0  # Negative adjustment when ML is better
    
    def test_dynamic_blend_prior_better(self):
        """Test when prior composite has higher IC."""
        np.random.seed(42)
        n_stocks = 50
        n_days = 100
        
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        stocks = [f"stock_{i:03d}" for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product(
            [dates, stocks], names=["date", "stock_code"]
        )
        
        forward_returns = pd.Series(np.random.randn(len(index)), index=index)
        
        # ML predictions with lower correlation
        ml_predictions = forward_returns * 0.1 + np.random.randn(len(index)) * 0.9
        ml_predictions = pd.Series(ml_predictions, index=index)
        
        # Prior composite with higher correlation
        prior_composite = forward_returns * 0.5 + np.random.randn(len(index)) * 0.5
        prior_composite = pd.Series(prior_composite, index=index)
        
        config = DynamicBlendConfig(
            base_ratio=0.5,
            max_delta=0.3,
            lookback_window=60,
            min_samples=20,
        )
        
        result = compute_dynamic_blend_ratio(
            ml_predictions=ml_predictions,
            prior_composite=prior_composite,
            forward_returns=forward_returns,
            config=config,
        )
        
        assert result.computation_successful
        # Prior has higher IC, so blend_ratio should be > base_ratio
        assert result.blend_ratio > config.base_ratio
        assert result.adjustment > 0  # Positive adjustment when prior is better
    
    def test_disabled_config(self, sample_blend_data):
        """Test that disabled config returns base_ratio."""
        ml_predictions, prior_composite, forward_returns = sample_blend_data
        
        config = DynamicBlendConfig(enabled=False, base_ratio=0.6)
        
        result = compute_dynamic_blend_ratio(
            ml_predictions=ml_predictions,
            prior_composite=prior_composite,
            forward_returns=forward_returns,
            config=config,
        )
        
        assert not result.computation_successful
        assert result.blend_ratio == 0.6
        assert "disabled" in result.fallback_reason.lower()
    
    def test_insufficient_samples(self):
        """Test fallback when insufficient samples."""
        np.random.seed(42)
        n_stocks = 5
        n_days = 10  # Very few days
        
        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        stocks = [f"stock_{i:03d}" for i in range(n_stocks)]
        
        index = pd.MultiIndex.from_product(
            [dates, stocks], names=["date", "stock_code"]
        )
        
        ml_predictions = pd.Series(np.random.randn(len(index)), index=index)
        prior_composite = pd.Series(np.random.randn(len(index)), index=index)
        forward_returns = pd.Series(np.random.randn(len(index)), index=index)
        
        config = DynamicBlendConfig(
            base_ratio=0.5,
            min_samples=30,  # More than available
        )
        
        result = compute_dynamic_blend_ratio(
            ml_predictions=ml_predictions,
            prior_composite=prior_composite,
            forward_returns=forward_returns,
            config=config,
        )
        
        assert not result.computation_successful
        assert result.blend_ratio == 0.5  # Falls back to base_ratio
        assert "insufficient" in result.fallback_reason.lower()
    
    def test_blend_ratio_bounds(self, sample_blend_data):
        """Test that blend ratio stays within [0, 1]."""
        ml_predictions, prior_composite, forward_returns = sample_blend_data
        
        config = DynamicBlendConfig(
            base_ratio=0.5,
            max_delta=0.3,
            lookback_window=60,
            min_samples=20,
        )
        
        result = compute_dynamic_blend_ratio(
            ml_predictions=ml_predictions,
            prior_composite=prior_composite,
            forward_returns=forward_returns,
            config=config,
        )
        
        assert 0.0 <= result.blend_ratio <= 1.0
        # Should be within [0.2, 0.8] given base=0.5, delta=0.3
        assert 0.2 <= result.blend_ratio <= 0.8

    def test_blend_ratio_stays_in_configured_range(self, sample_blend_data):
        ml_predictions, prior_composite, forward_returns = sample_blend_data

        config = DynamicBlendConfig(
            base_ratio=0.12,
            max_delta=0.1,
            lookback_window=60,
            min_samples=20,
        )

        result = compute_dynamic_blend_ratio(
            ml_predictions=ml_predictions,
            prior_composite=prior_composite,
            forward_returns=forward_returns,
            config=config,
        )

        assert result.computation_successful
        assert 0.02 - 1e-12 <= result.blend_ratio <= 0.22 + 1e-12


class TestBlendPredictions:
    """Tests for blend_predictions function."""
    
    def test_blend_50_50(self):
        """Test equal 50-50 blend."""
        index = pd.MultiIndex.from_arrays([
            pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]),
            ["A", "B", "A", "B"]
        ], names=["date", "stock_code"])
        
        ml = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
        prior = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)
        
        result = blend_predictions(ml, prior, blend_ratio=0.5)
        
        expected = pd.Series([0.5, 1.0, 1.5, 2.0], index=index)
        pd.testing.assert_series_equal(result, expected)
    
    def test_blend_ratio_0(self):
        """Test blend with ratio=0 (100% ML)."""
        index = pd.MultiIndex.from_arrays([
            pd.to_datetime(["2023-01-01", "2023-01-01"]),
            ["A", "B"]
        ], names=["date", "stock_code"])
        
        ml = pd.Series([1.0, 2.0], index=index)
        prior = pd.Series([10.0, 20.0], index=index)
        
        result = blend_predictions(ml, prior, blend_ratio=0.0)
        
        pd.testing.assert_series_equal(result, ml)
    
    def test_blend_ratio_1(self):
        """Test blend with ratio=1 (100% prior)."""
        index = pd.MultiIndex.from_arrays([
            pd.to_datetime(["2023-01-01", "2023-01-01"]),
            ["A", "B"]
        ], names=["date", "stock_code"])
        
        ml = pd.Series([1.0, 2.0], index=index)
        prior = pd.Series([10.0, 20.0], index=index)
        
        result = blend_predictions(ml, prior, blend_ratio=1.0)
        
        pd.testing.assert_series_equal(result, prior)
    
    def test_misaligned_indices(self):
        """Test blend with partially overlapping indices."""
        index1 = pd.MultiIndex.from_arrays([
            pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
            ["A", "B", "A"]
        ], names=["date", "stock_code"])
        
        index2 = pd.MultiIndex.from_arrays([
            pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-02"]),
            ["A", "A", "B"]
        ], names=["date", "stock_code"])
        
        ml = pd.Series([1.0, 2.0, 3.0], index=index1)
        prior = pd.Series([10.0, 30.0, 40.0], index=index2)
        
        result = blend_predictions(ml, prior, blend_ratio=0.5)
        
        # Only common indices should be in result
        # Common: (2023-01-01, A), (2023-01-02, A)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
