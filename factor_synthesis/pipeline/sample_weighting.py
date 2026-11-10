# -*- coding: utf-8 -*-
"""
Return-based sample weighting for long-only factor synthesis.

Implements asymmetric sample weighting strategies that give higher weight
to top-return stocks, aligned with A-share's long-only constraint where
only positive-return stocks are actionable.

Strategies:
    - return_magnitude: w_i = clip(max(0, y_i), 0, cap) + epsilon
    - quantile: Cross-sectional quantile bucketing (recommended for A-share)
    - softmax: w_i = softmax(y_i / tau) per cross-section
    - asymmetric_linear: w_i = alpha * max(0, y_i) + beta * max(0, -y_i) + epsilon

Reference:
    Guo et al. (2025) "A Novel Loss Function for Deep Learning Based
    Daily Stock Trading System" (arXiv:2502.17493)
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def compute_return_based_weights(
    y_values: np.ndarray,
    y_index: pd.MultiIndex,
    config: dict,
    log: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """
    Compute return-based sample weights from forward returns (or processed labels).

    The weights are computed cross-sectionally (per date) by default so that
    the relative ranking within each date determines the weight, not the
    absolute magnitude across dates.

    Args:
        y_values: 1-D array of label values (forward returns or processed labels).
        y_index: MultiIndex aligned with y_values (level 0 = datetime).
        config: Synthesis config dict containing ``sample_weight_method``
                and ``sample_weight_config``.
        log: Optional logger.

    Returns:
        1-D weight array normalised to mean = 1, or ``None`` if method is
        ``"none"`` or not specified.
    """
    log = log or logger

    method = config.get("sample_weight_method", "none")
    if method == "none" or method is None:
        return None

    sw_cfg = config.get("sample_weight_config", {})
    cross_sectional = sw_cfg.get("cross_sectional", True)

    dispatch = {
        "return_magnitude": _return_magnitude_weights,
        "quantile": _quantile_weights,
        "softmax": _softmax_weights,
        "asymmetric_linear": _asymmetric_linear_weights,
    }

    fn = dispatch.get(method)
    if fn is None:
        log.warning(
            "[ReturnWeight] Unknown sample_weight_method='%s', skipping.",
            method,
        )
        return None

    # Validate the active strategy before singleton date groups can skip it.
    if method == "return_magnitude":
        cap = sw_cfg.get("cap", 0.10)
        if np.isnan(cap) or cap < 0:
            raise ValueError("cap must be nonnegative and not NaN")
        epsilon = sw_cfg.get("epsilon", 0.01)
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be nonnegative and finite")
    elif method == "quantile":
        for key in ("top_pct", "bottom_pct"):
            value = sw_cfg.get(key, 0.20)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and between 0 and 1")
        for key, default in (("top_weight", 5.0), ("mid_weight", 1.0), ("bottom_weight", 0.5)):
            value = sw_cfg.get(key, default)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
    elif method == "softmax":
        if np.isnan(sw_cfg.get("temperature", 1.0)):
            raise ValueError("temperature must not be NaN")
    elif method == "asymmetric_linear":
        for key, default in (("positive_scale", 3.0), ("negative_scale", 1.0), ("epsilon", 0.01)):
            value = sw_cfg.get(key, default)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be nonnegative and finite")

    if cross_sectional:
        weights = _apply_cross_sectional(y_values, y_index, fn, sw_cfg, log)
    else:
        weights = fn(y_values, sw_cfg)

    # Normalise to mean = 1
    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean

    # Log diagnostics
    n_eff = float(weights.sum() ** 2 / (weights ** 2).sum()) if len(weights) > 0 else 0.0
    log.info(
        "[ReturnWeight] method=%s, n=%d, n_eff=%.1f (%.1f%%), "
        "w_min=%.4f, w_max=%.4f, w_std=%.4f",
        method,
        len(weights),
        n_eff,
        100.0 * n_eff / max(len(weights), 1),
        float(weights.min()),
        float(weights.max()),
        float(weights.std()),
    )

    return weights


# ---------------------------------------------------------------------------
# Internal strategy implementations
# ---------------------------------------------------------------------------

def _apply_cross_sectional(
    y_values: np.ndarray,
    y_index: pd.MultiIndex,
    fn,
    sw_cfg: dict,
    log: logging.Logger,
) -> np.ndarray:
    """Apply a weighting function cross-sectionally (per date)."""
    dates = y_index.get_level_values(0)
    unique_dates = dates.unique()
    weights = np.ones(len(y_values), dtype=float)

    for dt in unique_dates:
        mask = dates == dt
        y_cs = y_values[mask]
        if len(y_cs) < 2:
            # Too few stocks on this date – uniform weight
            continue
        w_cs = fn(y_cs, sw_cfg)
        # Per-date normalise to mean = 1 so dates are equally represented
        w_mean = w_cs.mean()
        if w_mean > 0:
            w_cs = w_cs / w_mean
        weights[mask] = w_cs

    return weights


def _return_magnitude_weights(y: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Asymmetric return-magnitude weighting (adapted from arXiv:2502.17493).

    ``w_i = clip(max(0, y_i), 0, cap) + epsilon``

    Only positive-valued labels contribute meaningfully.
    """
    epsilon = cfg.get("epsilon", 0.01)
    cap = cfg.get("cap", 0.10)

    w = np.clip(np.maximum(0.0, y), 0.0, cap) + epsilon
    return w


def _quantile_weights(y: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Quantile-bucketed weighting (recommended for A-share long-only).

    Top ``top_pct`` stocks get ``top_weight``, bottom ``bottom_pct``
    get ``bottom_weight``, middle stocks get ``mid_weight``.
    """
    top_pct = cfg.get("top_pct", 0.20)
    bottom_pct = cfg.get("bottom_pct", 0.20)
    top_weight = cfg.get("top_weight", 5.0)
    mid_weight = cfg.get("mid_weight", 1.0)
    bottom_weight = cfg.get("bottom_weight", 0.5)

    n = len(y)
    if n == 0:
        return np.array([], dtype=float)

    # Rank in ascending order (higher y = higher rank)
    ranks = _rank_array(y)  # 0-based ranks
    pct = ranks / max(n - 1, 1)  # percentile in [0, 1]

    weights = np.full(n, mid_weight, dtype=float)
    weights[pct >= (1.0 - top_pct)] = top_weight
    weights[pct < bottom_pct] = bottom_weight

    return weights


def _softmax_weights(y: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Softmax weighting with temperature control.

    ``w_i = softmax(y_i / tau)`` — low tau concentrates weight on the best stocks.
    """
    tau = cfg.get("temperature", 1.0)
    if tau <= 0:
        tau = 1.0

    z = y / tau
    # Numerical stability: subtract max before exp
    z = z - z.max()
    exp_z = np.exp(z)
    w = exp_z / exp_z.sum()

    # Scale so that mean = 1 (softmax sums to 1, multiply by n)
    w = w * len(y)
    return w


def _asymmetric_linear_weights(y: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Asymmetric linear weighting with separate scales for positive/negative.

    ``w_i = alpha * max(0, y_i) + beta * max(0, -y_i) + epsilon``
    """
    alpha = cfg.get("positive_scale", 3.0)
    beta = cfg.get("negative_scale", 1.0)
    epsilon = cfg.get("epsilon", 0.01)

    w = alpha * np.maximum(0.0, y) + beta * np.maximum(0.0, -y) + epsilon
    return w


# ---------------------------------------------------------------------------
# Combination helpers
# ---------------------------------------------------------------------------

def combine_weights(
    time_weights: Optional[np.ndarray],
    return_weights: Optional[np.ndarray],
    combine: bool = True,
) -> Optional[np.ndarray]:
    """
    Combine time-decay and return-based weights multiplicatively.

    If either is ``None`` the other is returned directly. The result
    is normalised to mean = 1.
    """
    if time_weights is None and return_weights is None:
        return None
    if time_weights is None:
        return return_weights
    if return_weights is None:
        return time_weights
    if not combine:
        # Prefer return-based weights when combine=False
        return return_weights

    combined = time_weights * return_weights
    c_mean = combined.mean()
    if c_mean > 0:
        combined = combined / c_mean
    return combined


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _rank_array(a: np.ndarray) -> np.ndarray:
    """Return 0-based ranks (ascending) with average tie-breaking."""
    temp = a.argsort()
    ranks = np.empty_like(temp, dtype=float)
    sorted_values = a[temp]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(a)]
    ranks[temp] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return ranks
