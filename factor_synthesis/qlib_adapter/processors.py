# -*- coding: utf-8 -*-
"""
Custom Qlib Processors for Factor Synthesis

This module provides custom Qlib Processor implementations for:
- Feature preprocessing (zscore, rank, rank_zscore, gauss_rank)
- Target preprocessing (cs_demean, cs_rank, cs_standardize)
- Sample weighting (time-decay, half-life)
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy.stats import norm

try:
    from qlib.data.dataset.processor import Processor
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    # Fallback base class for testing without Qlib
    class Processor:
        def __init__(self, **kwargs):
            pass
        def __call__(self, df):
            return df




class PerFactorPreprocessor(Processor):
    """
    Apply different preprocessing modes to different factors.
    
    Supports per-factor overrides from config like:
    {
        "ALL": "gauss_rank",
        "DemoSignal1": "zscore"
    }
    
    Modes:
        - "raw_zscore": Cross-sectional Z-score normalization (no winsorization)
        - "rank": Cross-sectional percentile rank [0, 1]
        - "rank_zscore": Rank then Z-score
        - "gauss_rank": Gauss Rank (inverse normal transform)
        - "none" or missing: No transformation
    
    Note: "zscore" (with winsorization) is now handled by the Native path in factor_ops.py.
          Use "raw_zscore" for the original unwinsorized Z-score behavior.
    
    Reference: run_factor_synthesis.py lines 841-917 (_apply_feature_preprocessing)
    """
    
    GAUSS_RANK_VERSION = "valid-count-v2"

    def __init__(self, preprocess_config: Dict[str, str], fields_group: str = "feature"):
        self.preprocess_config = preprocess_config
        self.fields_group = fields_group
        self.default_mode = preprocess_config.get("ALL", "none")
    
    def _zscore(self, series: pd.Series) -> pd.Series:
        """Cross-sectional Z-score."""
        m = series.mean()
        s = series.std()
        if s > 1e-8:
            return (series - m) / s
        return series - m
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """Cross-sectional percentile rank."""
        return series.rank(method='average', pct=True)
    
    def _gauss_rank(self, series: pd.Series) -> pd.Series:
        """Gauss Rank v2: use non-missing ranks and preserve missing values."""
        n = int(series.count())
        if n < 2:
            result = pd.Series(0.0, index=series.index)
            return result.where(series.notna())
        ranks = series.rank(method='average', pct=True) * (n / (n + 1))
        return pd.Series(norm.ppf(ranks), index=series.index)
    
    def _apply_mode(self, series: pd.Series, mode: str) -> pd.Series:
        """Apply preprocessing mode to a series."""
        if mode == "raw_zscore":
            return self._zscore(series)
        elif mode == "rank":
            return self._rank(series)
        elif mode == "rank_zscore":
            ranked = self._rank(series)
            return self._zscore(ranked)
        elif mode == "gauss_rank":
            return self._gauss_rank(series)
        else:
            return series
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not self.preprocess_config:
            return df
        
        result = df.copy()
        
        for col in df.columns:
            # Get mode for this factor (specific override or default)
            mode = self.preprocess_config.get(col, self.default_mode)
            
            if mode and mode != "none":
                result[col] = df[col].groupby(level='datetime').transform(
                    lambda x: self._apply_mode(x, mode)
                )
        
        return result


class TargetCSProcessor(Processor):
    """
    Cross-sectional target preprocessing.
    
    Implements target_cs_demean, target_cs_rank, target_cs_standardize from config.
    Applied to labels during training (learn_processor).
    
    Logic:
        - If cs_rank=True: Apply rank transform then demean
        - Else:
            - If cs_demean=True: Subtract cross-sectional mean
            - If cs_standardize=True: Divide by cross-sectional std
    
    Reference: lasso_method.py lines 331-352, lightgbm_method.py lines 536-550
    """
    
    def __init__(
        self,
        cs_demean: bool = True,
        cs_rank: bool = False,
        cs_standardize: bool = False,
        fields_group: str = "label",
        eps: float = 1e-8,
    ):
        self.cs_demean = cs_demean
        self.cs_rank = cs_rank
        self.cs_standardize = cs_standardize
        self.fields_group = fields_group
        self.eps = eps
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        result = df.copy()
        
        for col in df.columns:
            if self.cs_rank:
                # Rank then demean (matches: y_rank - y_rank.groupby().transform("mean"))
                ranked = df[col].groupby(level='datetime').rank(pct=True)
                result[col] = ranked - ranked.groupby(level='datetime').transform('mean')
            else:
                if self.cs_demean:
                    result[col] = df[col] - df[col].groupby(level='datetime').transform('mean')
                if self.cs_standardize:
                    std = result[col].groupby(level='datetime').transform('std')
                    result[col] = result[col] / (std.replace(0, np.nan).fillna(self.eps))
        
        return result


class SampleWeightProcessor(Processor):
    """
    Compute time-decay sample weights.
    
    This processor computes sample weights based on time decay but does NOT
    modify the input DataFrame. Weights are stored internally and accessed
    via get_sample_weights() after processing.
    
    Modes:
        1. Exponential decay: weight = decay ** days_from_max
        2. Half-life decay: weight = exp(-ln(2) * days / half_life)
    
    Reference: lasso_method.py lines 380-385, lightgbm_method.py lines 563-568
    """
    
    def __init__(
        self,
        weight_decay: float = 1.0,
        half_life: float = 0.0,
        fields_group: str = "feature"
    ):
        self.weight_decay = weight_decay
        self.half_life = half_life
        self.fields_group = fields_group
        self._sample_weights: Optional[np.ndarray] = None
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute weights from DataFrame index.
        
        Does NOT modify the DataFrame - only stores computed weights.
        """
        if df.empty:
            self._sample_weights = np.array([])
            return df
        
        # Get datetime from index
        dates = df.index.get_level_values('datetime')
        dates = pd.to_datetime(dates)
        max_date = dates.max()
        
        # Compute days difference
        days_diff = (max_date - dates).days.values.astype(float)
        
        if self.half_life > 0:
            # Half-life decay: weight = exp(-ln(2) * days / half_life)
            self._sample_weights = np.exp(-np.log(2) * days_diff / self.half_life)
        elif self.weight_decay < 1.0:
            # Exponential decay: weight = decay ** days
            self._sample_weights = np.power(self.weight_decay, days_diff)
        else:
            # No decay
            self._sample_weights = np.ones(len(df))
        
        # Normalize weights to mean 1
        if len(self._sample_weights) > 0:
            self._sample_weights = self._sample_weights / self._sample_weights.mean()
        
        return df  # Return unchanged
    
    def get_sample_weights(self) -> Optional[np.ndarray]:
        """Get computed sample weights (call after __call__)."""
        return self._sample_weights
