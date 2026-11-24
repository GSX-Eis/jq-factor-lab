# -*- coding: utf-8 -*-
"""
Test preprocessing utilities that use Qlib processors.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from factor_synthesis.pipeline.preprocessing import (
    apply_target_cs_preprocessing,
    bind_synthesis_panel,
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






class TestBindSynthesisPanel(unittest.TestCase):
    """Test bind_synthesis_panel handling of stub synthesizers and optional config."""

    def test_missing_or_none_synthesizer_config_and_none_config(self):
        panel = create_test_data(n_dates=3, n_stocks=3)
        for synth in (
            SimpleNamespace(),
            SimpleNamespace(config=None),
            SimpleNamespace(config="invalid"),
        ):
            bound_panel, bound_cfg = bind_synthesis_panel(synth, panel, None, {})
            self.assertIs(bound_panel, panel)
            self.assertEqual(bound_cfg, {})

    def test_panel_none_loads_with_safe_empty_synth_config(self):
        panel = create_test_data(n_dates=2, n_stocks=2)
        with patch(
            "factor_synthesis.core.factor_data_loader.load_factor_data",
            return_value=panel,
        ) as mock_load:
            bound_panel, bound_cfg = bind_synthesis_panel(
                SimpleNamespace(),
                None,
                {"factor_names": ["DemoSignal1"], "start_date": "2023-01-01", "end_date": "2023-01-02"},
                {},
            )
        self.assertIs(bound_panel, panel)
        self.assertEqual(bound_cfg["factor_names"], ["DemoSignal1"])
        self.assertEqual(mock_load.call_args.kwargs["config"], {})

    def test_stub_synthesizer_callers_do_not_raise_attribute_error(self):
        import logging
        from factor_synthesis.pipeline.size_conditional_method import run_size_conditional_synthesis
        from factor_synthesis.pipeline.weighted_sum_method import run_weighted_sum_synthesis

        panel = create_test_data(n_dates=2, n_stocks=2)
        returns = panel["DemoSignal1"]
        stub = SimpleNamespace(logger=logging.getLogger(__name__))
        with self.assertRaisesRegex(RuntimeError, "empty factor group"):
            run_weighted_sum_synthesis(stub, "ws_test", {}, panel)
        with self.assertRaises(ValueError):
            run_size_conditional_synthesis(
                stub, panel, returns, returns, "2023-01-01", "2023-01-02", {}
            )


if __name__ == '__main__':
    unittest.main()

