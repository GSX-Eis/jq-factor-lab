# -*- coding: utf-8 -*-
"""
Dynamic Blend Ratio Module

Implements dynamic blending of ML predictions with prior composite factors
based on recent IC/ICIR performance (AlphaForge 2024 Algorithm 2).

Core idea: Factor momentum effect - factors that performed well recently
tend to continue performing well. The blend ratio is adjusted dynamically
based on the relative IC of ML predictions vs prior composite.

References:
- Bates & Granger (1969): Inverse variance weighting
- AlphaForge (2024): IC-based dynamic factor combination
- EHSANI & LINNAINMAA (2022): Factor Momentum theory
- Magnus et al. (2016): Forecast combination puzzle, shrinkage toward equal weight
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DynamicBlendConfig:
    """
    Configuration for dynamic blend ratio computation.
    
    The dynamic ratio is computed as:
        dynamic_ratio = base_ratio + max_delta * adjustment + bias
    
    Where adjustment is in [-1, 1], computed via tanh smoothing.
    This gives a final ratio in [base_ratio - max_delta + bias, base_ratio + max_delta + bias].
    
    Example with defaults:
        base_ratio=0.5, max_delta=0.3, bias=0.0 → ratio range [0.2, 0.8]
        base_ratio=0.5, max_delta=0.2, bias=0.1 → ratio range [0.4, 0.8]
    
    Attributes:
        enabled: Whether to use dynamic blending (False falls back to static ratio)
        base_ratio: Prior mean blend ratio (center point)
        max_delta: Maximum adjustment amplitude (determines range)
        bias: Fixed offset added to the final ratio (positive = more prior weight)
        lookback_window: Number of trading days for rolling IC calculation
        use_icir: If True, use ICIR (IC/IC_std) instead of raw IC for stability
        smoothing_scale: Scale parameter for tanh smoothing (controls sensitivity)
        min_samples: Minimum samples required to compute IC
        correlation_aware: If True, use unified ICIR+correlation formula instead of tanh
        shrinkage: Shrinkage toward base_ratio (0=trust optimal, 1=ignore optimal)
        topk_ic_pct: Fraction of top stocks (by ML prediction) used for IC computation.
            0.0 = disabled (full cross-section IC, default).
            e.g. 0.3 = compute IC only on stocks in the top 30% of ML predictions per date.
            Useful for quantile regression models where full-sample IC underestimates
            the model's actual ranking quality in the upper tail.
    """
    enabled: bool = True
    base_ratio: float = 0.5
    max_delta: float = 0.3
    bias: float = 0.0
    lookback_window: int = 60
    use_icir: bool = False
    smoothing_scale: float = 0.1
    min_samples: int = 20
    correlation_aware: bool = False
    shrinkage: float = 0.5
    topk_ic_pct: float = 0.0

    def __post_init__(self):
        """Validate configuration parameters."""
        for name in ('max_delta', 'bias', 'smoothing_scale'):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)}")
        if not 0.0 <= self.base_ratio <= 1.0:
            raise ValueError(f"base_ratio must be in [0, 1], got {self.base_ratio}")
        if self.max_delta < 0:
            raise ValueError(f"max_delta must be non-negative, got {self.max_delta}")
        # Validate that the full range [base - delta + bias, base + delta + bias] stays in [0, 1]
        min_ratio = self.base_ratio - self.max_delta + self.bias
        max_ratio = self.base_ratio + self.max_delta + self.bias
        if min_ratio < 0 or max_ratio > 1:
            raise ValueError(
                f"base_ratio ± max_delta + bias must stay in [0, 1], "
                f"got range [{min_ratio}, {max_ratio}]"
            )
        if self.lookback_window < 1:
            raise ValueError(f"lookback_window must be >= 1, got {self.lookback_window}")
        if self.smoothing_scale <= 0:
            raise ValueError(f"smoothing_scale must be positive, got {self.smoothing_scale}")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {self.shrinkage}")
        if not 0.0 <= self.topk_ic_pct < 1.0:
            raise ValueError(f"topk_ic_pct must be in [0, 1), got {self.topk_ic_pct}")


@dataclass
class DynamicBlendResult:
    """
    Result of dynamic blend ratio computation.

    Stores the computed ratio along with diagnostic information.
    """
    blend_ratio: float
    ml_ic: float
    ml_ic_std: float
    prior_ic: float
    prior_ic_std: float
    adjustment: float
    n_samples: int
    computation_successful: bool = True
    fallback_reason: Optional[str] = None
    signal_correlation: float = 0.0
    raw_optimal_ratio: float = 0.0
    sample_contract_version: str = "finite_pairs_10_v2"
    diagnostic_version: str = "finite_statistics_v2"


def compute_rolling_ic_stats(
    predictions: pd.Series,
    forward_returns: pd.Series,
    window: int = 60,
    min_periods: int = 20,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, int]:
    """
    Compute rolling window IC statistics (mean and std) for predictions.
    
    Uses Spearman rank correlation (RankIC) which is more robust to outliers.
    
    Args:
        predictions: Model predictions or factor values, MultiIndex (date, stock_code)
        forward_returns: Forward returns, same index structure
        window: Number of trading days for rolling window
        min_periods: Minimum periods required for valid IC calculation
        logger: Optional logger for diagnostics
        
    Returns:
        Tuple of (ic_mean, ic_std, n_days) where:
            - ic_mean: Signed mean IC over the window. A negative IC means
              that this signal ranks returns in the wrong direction.
            - ic_std: Standard deviation of daily IC
            - n_days: Number of valid trading days used
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Ensure index alignment
    common_index = predictions.index.intersection(forward_returns.index)
    if len(common_index) == 0:
        logger.warning("[DynamicBlend] No common index between predictions and returns")
        return 0.0, 1.0, 0
    
    preds = predictions.loc[common_index]
    rets = forward_returns.loc[common_index]
    
    # Get unique dates and take the most recent `window` days
    dates = preds.index.get_level_values('date').unique().sort_values()
    if len(dates) > window:
        recent_dates = dates[-window:]
    else:
        recent_dates = dates
    
    if len(recent_dates) < min_periods:
        logger.debug(
            f"[DynamicBlend] Insufficient dates for IC: {len(recent_dates)} < {min_periods}"
        )
        return 0.0, 1.0, 0
    
    # Compute daily cross-sectional Spearman IC
    daily_ics = []
    for date in recent_dates:
        try:
            pred_day = preds.xs(date, level='date')
            ret_day = rets.xs(date, level='date')
            
            # Align stocks
            common_stocks = pred_day.index.intersection(ret_day.index)
            if len(common_stocks) < 10:  # Need minimum stocks for meaningful correlation
                continue
            
            p = pred_day.loc[common_stocks]
            r = ret_day.loc[common_stocks]
            finite_pairs = np.isfinite(p) & np.isfinite(r)
            p, r = p.loc[finite_pairs], r.loc[finite_pairs]
            if len(p) < 10:
                continue
            
            # Skip if constant values (would produce NaN correlation)
            if p.std() < 1e-10 or r.std() < 1e-10:
                continue
            
            # Spearman correlation (rank-based)
            ic = p.corr(r, method='spearman')
            if not np.isnan(ic):
                daily_ics.append(ic)
                
        except Exception as e:
            logger.exception("[DynamicBlend] IC calculation failed for %s: %s", date, e)
            raise
    
    if len(daily_ics) < min_periods:
        logger.debug(
            f"[DynamicBlend] Insufficient valid IC days: {len(daily_ics)} < {min_periods}"
        )
        return 0.0, 1.0, len(daily_ics)
    
    ic_series = pd.Series(daily_ics)
    # Preserve direction.  Taking abs here made RankIC=-1 look as useful as
    # RankIC=+1 even though blending the raw signals makes them cancel.
    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())
    
    return ic_mean, ic_std, len(daily_ics)


def compute_prior_composite(
    factor_df: pd.DataFrame,
    blend_weights: Dict[str, float],
    index: pd.Index,
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """
    Compute the prior composite signal from blend_weights factors.
    
    Args:
        factor_df: Factor data DataFrame with columns as factor names
        blend_weights: Dict mapping factor names to weights (e.g., {"DemoSignal1": 1.0})
        index: Target index for the composite (typically predictions.index)
        logger: Optional logger
        
    Returns:
        Prior composite signal as pd.Series
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    composite = pd.Series(0.0, index=index)
    
    for factor_name, weight in blend_weights.items():
        if factor_name not in factor_df.columns:
            logger.warning(f"[DynamicBlend] Blend factor {factor_name} not found in factor_df")
            continue
        if not np.isfinite(weight):
            raise ValueError(f"Blend weight for {factor_name} must be finite, got {weight}")
        
        factor_vals = factor_df[factor_name].reindex(index)
        
        # Cross-sectional z-score for alignment with ML predictions
        factor_zscore = factor_vals.groupby(level='date').transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # Fill NaN with 0 (mean value for z-scored data)
        factor_zscore = factor_zscore.fillna(0.0)
        
        composite += factor_zscore * weight
    
    return composite


def compute_signal_correlation(
    ml_predictions: pd.Series,
    prior_composite: pd.Series,
    window: int = 60,
    min_periods: int = 20,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Compute cross-sectional Spearman correlation between ML predictions and prior composite.

    Uses per-day correlation, then averages across the lookback window.
    Used by the unified correlation-aware blend formula.

    Args:
        ml_predictions: ML model predictions (MultiIndex: date, stock_code)
        prior_composite: Prior composite signal (same index)
        window: Number of trading days for rolling window
        min_periods: Minimum periods required for valid correlation
        logger: Optional logger for diagnostics

    Returns:
        Average cross-sectional correlation (rho) between the two signals
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    common_index = ml_predictions.index.intersection(prior_composite.index)
    if len(common_index) == 0:
        logger.warning("[DynamicBlend] No common index for signal correlation")
        return 0.0

    ml = ml_predictions.loc[common_index]
    prior = prior_composite.loc[common_index]

    dates = ml.index.get_level_values("date").unique().sort_values()
    if len(dates) > window:
        recent_dates = dates[-window:]
    else:
        recent_dates = dates

    if len(recent_dates) < min_periods:
        logger.debug(
            f"[DynamicBlend] Insufficient dates for signal correlation: "
            f"{len(recent_dates)} < {min_periods}"
        )
        return 0.0

    daily_corrs: list[float] = []
    for date in recent_dates:
        try:
            ml_day = ml.xs(date, level="date")
            prior_day = prior.xs(date, level="date")
            common_stocks = ml_day.index.intersection(prior_day.index)
            if len(common_stocks) < 10:
                continue
            p = ml_day.loc[common_stocks]
            q = prior_day.loc[common_stocks]
            finite_pairs = np.isfinite(p) & np.isfinite(q)
            p, q = p.loc[finite_pairs], q.loc[finite_pairs]
            if len(p) < 10:
                continue
            if p.std() < 1e-10 or q.std() < 1e-10:
                continue
            corr = p.corr(q, method="spearman")
            if not np.isnan(corr):
                daily_corrs.append(float(corr))
        except Exception as e:
            logger.exception("[DynamicBlend] Signal correlation failed for %s: %s", date, e)
            raise

    if len(daily_corrs) < min_periods:
        return 0.0

    return float(np.mean(daily_corrs))


def _filter_topk_index(
    ml_predictions: pd.Series,
    topk_pct: float,
) -> pd.Index:
    """
    Return the MultiIndex entries corresponding to the top ``topk_pct``
    fraction of ML predictions on each date.

    For example, ``topk_pct=0.3`` keeps the top 30 % of stocks per date
    (highest predicted values).  The returned index can be used to slice
    ``ml_predictions``, ``prior_composite``, and ``forward_returns``
    before feeding them into IC computation so that IC measures ranking
    quality *within the upper tail* only.

    Args:
        ml_predictions: ML predictions with MultiIndex (date, stock_code).
        topk_pct: Fraction in (0, 1). Stocks whose prediction is at or
            above the ``(1 - topk_pct)`` quantile on a given date are kept.

    Returns:
        pd.Index – a subset of ``ml_predictions.index``.
    """
    if ml_predictions.empty:
        logging.getLogger(__name__).debug("[DynamicBlend] topk_filter_version=empty_index_v2")
        return ml_predictions.index[:0]
    masks: list[pd.Series] = []
    for date, group in ml_predictions.groupby(level="date"):
        threshold = group.quantile(1.0 - topk_pct)
        masks.append(group >= threshold)
    mask = pd.concat(masks)
    return mask[mask].index


def compute_dynamic_blend_ratio(
    ml_predictions: pd.Series,
    prior_composite: pd.Series,
    forward_returns: pd.Series,
    config: DynamicBlendConfig,
    logger: Optional[logging.Logger] = None,
) -> DynamicBlendResult:
    """
    Compute dynamic blend ratio based on recent IC performance.
    
    The algorithm:
    1. Calculate rolling IC for ML predictions and prior composite
    2. Compute adjustment based on IC difference (or ICIR if enabled)
    3. Apply tanh smoothing to bound adjustment to [-1, 1]
    4. Final ratio = base_ratio + max_delta * adjustment
    
    When prior IC > ML IC: adjustment > 0 → increase prior weight
    When ML IC > prior IC: adjustment < 0 → decrease prior weight (more ML)
    
    Args:
        ml_predictions: ML model predictions (MultiIndex: date, stock_code)
        prior_composite: Prior composite signal (same index)
        forward_returns: Forward returns for IC calculation
        config: DynamicBlendConfig instance
        logger: Optional logger
        
    Returns:
        DynamicBlendResult with computed ratio and diagnostics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not config.enabled:
        return DynamicBlendResult(
            blend_ratio=config.base_ratio,
            ml_ic=0.0,
            ml_ic_std=0.0,
            prior_ic=0.0,
            prior_ic_std=0.0,
            adjustment=0.0,
            n_samples=0,
            computation_successful=False,
            fallback_reason="Dynamic blend disabled"
        )
    
    # ---- Top-K IC filtering (for quantile regression models) ----
    # When topk_ic_pct > 0, restrict IC computation to stocks in the
    # top quantile of ML predictions.  This aligns the IC metric with
    # the actual objective of upper-tail ranking models (e.g. tau=0.80).
    ml_for_ic = ml_predictions
    prior_for_ic = prior_composite
    fwd_for_ic = forward_returns
    topk_label = ""

    if config.topk_ic_pct > 0:
        topk_idx = _filter_topk_index(ml_predictions, config.topk_ic_pct)
        ml_for_ic = ml_predictions.loc[ml_predictions.index.isin(topk_idx)]
        prior_for_ic = prior_composite.loc[prior_composite.index.isin(topk_idx)]
        fwd_for_ic = forward_returns.loc[forward_returns.index.isin(topk_idx)]
        topk_label = f" (top-{config.topk_ic_pct:.0%} IC)"
        logger.info(
            f"[DynamicBlend] topk_ic_pct={config.topk_ic_pct:.2f}: "
            f"IC computed on {len(topk_idx)}/{len(ml_predictions)} samples{topk_label}"
        )

    # Compute IC statistics for ML predictions
    ml_ic, ml_ic_std, ml_n = compute_rolling_ic_stats(
        ml_for_ic,
        fwd_for_ic,
        window=config.lookback_window,
        min_periods=config.min_samples,
        logger=logger,
    )
    
    # Compute IC statistics for prior composite
    prior_ic, prior_ic_std, prior_n = compute_rolling_ic_stats(
        prior_for_ic,
        fwd_for_ic,
        window=config.lookback_window,
        min_periods=config.min_samples,
        logger=logger,
    )
    
    n_samples = min(ml_n, prior_n)
    
    # Check if we have enough data
    if n_samples < config.min_samples:
        logger.warning(
            f"[DynamicBlend] Insufficient samples ({n_samples} < {config.min_samples}), "
            f"falling back to base_ratio={config.base_ratio}"
        )
        return DynamicBlendResult(
            blend_ratio=config.base_ratio,
            ml_ic=ml_ic,
            ml_ic_std=ml_ic_std,
            prior_ic=prior_ic,
            prior_ic_std=prior_ic_std,
            adjustment=0.0,
            n_samples=n_samples,
            computation_successful=False,
            fallback_reason=f"Insufficient samples: {n_samples} < {config.min_samples}",
        )

    if not np.isfinite([ml_ic, ml_ic_std, prior_ic, prior_ic_std]).all():
        logger.warning("[DynamicBlend] Non-finite IC statistics, falling back to base_ratio=%s", config.base_ratio)
        return DynamicBlendResult(
            blend_ratio=config.base_ratio,
            ml_ic=ml_ic,
            ml_ic_std=ml_ic_std,
            prior_ic=prior_ic,
            prior_ic_std=prior_ic_std,
            adjustment=0.0,
            n_samples=n_samples,
            computation_successful=False,
            fallback_reason="Non-finite IC statistics",
        )

    if config.correlation_aware:
        # Unified formula: jointly considers ICIR and inter-signal correlation
        rho = compute_signal_correlation(
            ml_predictions,
            prior_composite,
            window=config.lookback_window,
            min_periods=config.min_samples,
            logger=logger,
        )
        icir_ml = ml_ic / (ml_ic_std + 1e-8)
        icir_prior = prior_ic / (prior_ic_std + 1e-8)

        # Bates & Granger (1969) inverse-variance weighting cross-term:
        # Use geometric mean sqrt(|ICIR_ml|·|ICIR_prior|) instead of min()
        # to match the standard σ_ml·σ_prior form (always positive, ρ carries sign)
        geo_mean_icir = float(np.sqrt(abs(icir_ml) * abs(icir_prior)))
        denom = icir_ml + icir_prior - 2 * rho * geo_mean_icir
        if abs(denom) < 1e-8:
            w_raw = 0.5
        else:
            w_raw = (icir_prior - rho * geo_mean_icir) / denom
        w_raw = float(np.clip(w_raw, 0.0, 1.0))

        dynamic_ratio = config.shrinkage * config.base_ratio + (1 - config.shrinkage) * w_raw
        dynamic_ratio = float(np.clip(dynamic_ratio, 0.0, 1.0))

        logger.info(
            f"[DynamicBlend] correlation_aware{topk_label}: ICIR ML={icir_ml:.4f}, Prior={icir_prior:.4f}, "
            f"rho={rho:.4f} | w_raw={w_raw:.3f}, shrinkage={config.shrinkage} → ratio={dynamic_ratio:.3f}"
        )

        return DynamicBlendResult(
            blend_ratio=dynamic_ratio,
            ml_ic=ml_ic,
            ml_ic_std=ml_ic_std,
            prior_ic=prior_ic,
            prior_ic_std=prior_ic_std,
            adjustment=0.0,
            n_samples=n_samples,
            computation_successful=True,
            signal_correlation=rho,
            raw_optimal_ratio=w_raw,
        )

    # Original tanh-based logic (correlation_aware=False)
    if config.use_icir:
        ml_signal = ml_ic / (ml_ic_std + 1e-8)
        prior_signal = prior_ic / (prior_ic_std + 1e-8)
        scale = 0.5
    else:
        ml_signal = ml_ic
        prior_signal = prior_ic
        scale = config.smoothing_scale

    diff = prior_signal - ml_signal
    adjustment = float(np.tanh(diff / scale))
    dynamic_ratio = config.base_ratio + config.max_delta * adjustment + config.bias
    dynamic_ratio = float(np.clip(dynamic_ratio, 0.0, 1.0))

    logger.info(
        f"[DynamicBlend]{topk_label} IC: ML={ml_ic:.4f} (std={ml_ic_std:.4f}), "
        f"Prior={prior_ic:.4f} (std={prior_ic_std:.4f}) | "
        f"Adjustment={adjustment:.3f}, bias={config.bias:.3f} → ratio={dynamic_ratio:.3f} "
        f"(base={config.base_ratio}, delta={config.max_delta})"
    )

    return DynamicBlendResult(
        blend_ratio=dynamic_ratio,
        ml_ic=ml_ic,
        ml_ic_std=ml_ic_std,
        prior_ic=prior_ic,
        prior_ic_std=prior_ic_std,
        adjustment=adjustment,
        n_samples=n_samples,
        computation_successful=True,
    )


def blend_predictions(
    ml_predictions: pd.Series,
    prior_composite: pd.Series,
    blend_ratio: float,
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """
    Blend ML predictions with prior composite using the given ratio.
    
    Formula: result = ml * (1 - ratio) + prior * ratio
    
    When ratio = 0: 100% ML predictions
    When ratio = 1: 100% prior composite
    When ratio = 0.5: Equal blend
    
    Args:
        ml_predictions: ML model predictions
        prior_composite: Prior composite signal
        blend_ratio: Blend ratio in [0, 1] (higher = more prior)
        logger: Optional logger
        
    Returns:
        Blended predictions
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Align indices
    common_index = ml_predictions.index.intersection(prior_composite.index)
    if len(common_index) == 0:
        logger.warning("[DynamicBlend] No common index for blending, returning ML predictions")
        return ml_predictions
    
    ml = ml_predictions.loc[common_index]
    prior = prior_composite.loc[common_index]
    
    result = ml * (1 - blend_ratio) + prior * blend_ratio
    
    return result
