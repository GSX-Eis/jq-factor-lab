# -*- coding: utf-8 -*-
"""
Custom Objective Functions for Qlib Adapter

Provides custom loss functions (gradient/hessian) and evaluation metrics
for LightGBM models, supporting both global and cross-sectional IC optimization.
"""

from typing import Tuple, Optional, Callable, Union
import numpy as np
import logging

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logger = logging.getLogger(__name__)


class QlibCustomObjectives:
    """
    Custom objective functions for Qlib models
    
    Supports:
    - Pearson IC Loss (Global)
    - Cross-Sectional Pearson IC Loss (per date group)
    - Sharpe Surrogate Loss (softmax-weighted top-K optimization)
    - IC/Sharpe Evaluation Metrics
    """
    
    # Registry of available objectives
    OBJECTIVES = {
        "pearson_ic": "pearson_ic_loss",
        "cross_sectional_ic": "cross_sectional_ic_loss",
        "global_ic": "pearson_ic_loss",
        "sharpe_surrogate": "sharpe_surrogate_loss",
        "topk_sharpe": "sharpe_surrogate_loss",
    }
    
    EVAL_METRICS = {
        "ic": "ic_eval_metric",
        "pearson_ic": "ic_eval_metric",
        "sharpe": "sharpe_eval_metric",
        "topk_sharpe": "sharpe_eval_metric",
    }
    
    # Configuration for Sharpe surrogate (can be overridden)
    SHARPE_CONFIG = {
        "temperature": 1.0,  # Softmax temperature (lower = sharper selection)
        "top_pct": 0.2,      # Approximate top percentage for selection
        "min_group_size": 5, # Minimum stocks per group to compute Sharpe
    }
    
    @classmethod
    def get_lgb_objective(cls, name: str) -> Callable:
        """
        Get LightGBM custom objective function by name
        
        Args:
            name: Objective name ("pearson_ic", "cross_sectional_ic")
            
        Returns:
            Callable that returns (gradient, hessian)
        """
        if name not in cls.OBJECTIVES:
            raise ValueError(
                f"Unknown objective: {name}. "
                f"Available: {list(cls.OBJECTIVES.keys())}"
            )
        
        method_name = cls.OBJECTIVES[name]
        return getattr(cls, method_name)
    
    @classmethod
    def get_lgb_eval_metric(cls, name: str) -> Callable:
        """
        Get LightGBM custom evaluation metric by name
        
        Args:
            name: Metric name ("ic", "pearson_ic")
            
        Returns:
            Callable that returns (metric_name, value, is_higher_better)
        """
        if name not in cls.EVAL_METRICS:
            raise ValueError(
                f"Unknown eval metric: {name}. "
                f"Available: {list(cls.EVAL_METRICS.keys())}"
            )
        
        method_name = cls.EVAL_METRICS[name]
        return getattr(cls, method_name)
    
    @staticmethod
    def _unpack_objective_args(arg0, arg1):
        """Normalise LightGBM custom-objective args to (preds, labels, weights, groups).

        LightGBM's native API (>=4.0) calls ``objective(preds, train_data)`` — the
        ``lgb.Dataset`` is the *second* argument. The sklearn API calls
        ``objective(y_true, y_pred)`` with two ndarrays. Detect the Dataset by
        position-agnostic ``get_label`` probing so the objective is robust to both
        orders. The previous code assumed Dataset-first and crashed on lightgbm>=4
        with ``object of type 'Dataset' has no len()`` (the Dataset landed in the
        preds slot and ``len(preds)`` failed).
        """
        if hasattr(arg0, 'get_label'):
            train_data, preds = arg0, arg1
        elif hasattr(arg1, 'get_label'):
            train_data, preds = arg1, arg0
        else:
            # sklearn API: (y_true, y_pred) -> labels=arg0, preds=arg1
            train_data, preds = None, arg1
        labels = arg0 if train_data is None else train_data.get_label()
        preds, labels = np.asarray(preds), np.asarray(labels)
        if preds.ndim != 1 or labels.ndim != 1 or preds.shape != labels.shape:
            raise ValueError("predictions and labels must be one-dimensional arrays of equal length")
        return (
            preds,
            labels,
            None if train_data is None else train_data.get_weight(),
            None if train_data is None else train_data.get_group(),
        )

    @staticmethod
    def _validate_metadata(n, weights=None, groups=None):
        """Validate callback metadata without changing arrays or empty-group fallback."""
        if weights is not None:
            w = np.asarray(weights)
            if (w.ndim != 1 or w.shape != (n,) or w.dtype.kind not in 'biuf'
                    or not np.isfinite(w).all() or (w < 0).any()):
                raise ValueError(
                    "weights must be a finite nonnegative vector matching sample count"
                )
        if groups is not None:
            g = np.asarray(groups)
            if (g.ndim != 1 or g.dtype.kind not in 'iuf'
                    or not np.isfinite(g).all() or (g <= 0).any()
                    or (g != np.floor(g)).any()
                    or (g.size and sum(int(size) for size in g) != n)):
                raise ValueError("groups must be positive integer sizes summing to sample count")

    @staticmethod
    def pearson_ic_loss(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pearson IC Loss function for LightGBM.
        Objective: Minimize negative Pearson Correlation (-IC).

        Supports both sklearn API and native lgb.train() API (arg order resolved
        by :meth:`_unpack_objective_args`).
        """
        preds, labels, weights, groups = QlibCustomObjectives._unpack_objective_args(y_true, y_pred)
        if groups is not None and len(groups) > 0:
            return QlibCustomObjectives._cross_sectional_pearson_loss(preds, labels, groups, weights)
        return QlibCustomObjectives._global_pearson_loss(preds, labels, weights)

    @staticmethod
    def cross_sectional_ic_loss(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cross-Sectional IC Loss function for LightGBM.
        Calculates IC per group (date) and averages gradients; falls back to
        global IC when no group information is available (sklearn API).
        """
        preds, labels, weights, groups = QlibCustomObjectives._unpack_objective_args(y_true, y_pred)
        if groups is not None and len(groups) > 0:
            return QlibCustomObjectives._cross_sectional_pearson_loss(preds, labels, groups, weights)
        return QlibCustomObjectives._global_pearson_loss(preds, labels, weights)
    
    @staticmethod
    def _global_pearson_loss(
        preds: np.ndarray,
        labels: np.ndarray,
        weights: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Global Pearson IC Loss.
        Loss = -Correlation(preds, labels)
        
        Gradient derivation:
            Loss = -rho = -Cov(p,y) / (std_p * std_y)
            dL/dp_i = -1/(N*std_p*std_y) * (y_i - mean_y - rho * (std_y/std_p) * (p_i - mean_p))
        
        Hessian approximation:
            d²L/dp_i² ≈ 1/var_p (scale-adaptive approximation)
        """
        n = len(preds)
        if n < 2:
            return np.zeros(n), np.ones(n)
        
        p = preds
        y = labels
        
        if weights is not None:
            if (weights.ndim != 1 or weights.shape != preds.shape
                    or weights.dtype.kind not in 'biuf'
                    or not np.isfinite(weights).all() or (weights < 0).any()
                    or not (weights > 0).any()):
                raise ValueError(
                    "weights must be a finite nonnegative vector with positive total weight"
                )
            # Weighted Pearson Correlation
            w_sum = np.sum(weights)
            p_mean = np.average(p, weights=weights)
            y_mean = np.average(y, weights=weights)
            
            p_centered = p - p_mean
            y_centered = y - y_mean
            
            cov = np.sum(weights * p_centered * y_centered)
            var_p = np.sum(weights * p_centered**2)
            var_y = np.sum(weights * y_centered**2)
            
            std_p = np.sqrt(var_p)
            std_y = np.sqrt(var_y)
            
            # Avoid division by zero
            if std_p < 1e-8 or std_y < 1e-8:
                return np.zeros(n), np.ones(n)
            
            rho = cov / (std_p * std_y)
            
            # Gradient
            grad = -(weights * y_centered / (std_p * std_y)) + rho * (weights * p_centered / (std_p**2))
            
            # Hessian approximation
            hess = weights / (std_p**2 + 1e-8)
            
        else:
            p_mean = np.mean(p)
            y_mean = np.mean(y)
            
            p_centered = p - p_mean
            y_centered = y - y_mean
            
            var_p = np.sum(p_centered**2)
            var_y = np.sum(y_centered**2)
            
            std_p = np.sqrt(var_p)
            std_y = np.sqrt(var_y)
            
            if std_p < 1e-8 or std_y < 1e-8:
                return np.zeros(n), np.ones(n)
            
            cov = np.sum(p_centered * y_centered)
            rho = cov / (std_p * std_y)
            
            # Gradient of (-rho)
            grad = -y_centered / (std_p * std_y) + rho * p_centered / var_p
            
            # Hessian
            hess = np.ones(n) / (var_p + 1e-8)
        
        return grad, hess
    
    @staticmethod
    def _cross_sectional_pearson_loss(
        preds: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        weights: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cross-Sectional Pearson IC Loss.
        Calculate IC per group (date) and average gradients.
        
        Args:
            preds: Predictions array
            labels: Labels array
            groups: Group sizes array (e.g., number of stocks per date)
            weights: Optional sample weights
        """
        groups = np.asarray(groups)
        if (groups.ndim != 1 or groups.dtype.kind not in 'iuf'
                or not np.isfinite(groups).all() or (groups <= 0).any()
                or (groups != np.floor(groups)).any()
                or sum(int(size) for size in groups) != len(preds)):
            raise ValueError("groups must be positive integer sizes summing to sample count")
        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        
        current_idx = 0
        for g_size in groups:
            g_size = int(g_size)
            slc = slice(current_idx, current_idx + g_size)
            
            g_preds = preds[slc]
            g_labels = labels[slc]
            g_weights = weights[slc] if weights is not None else None
            
            g_grad, g_hess = QlibCustomObjectives._global_pearson_loss(
                g_preds, g_labels, g_weights
            )
            
            grad[slc] = g_grad
            hess[slc] = g_hess
            
            current_idx += g_size
        
        return grad, hess
    
    @staticmethod
    def ic_eval_metric(preds: np.ndarray, train_data) -> Tuple[str, float, bool]:
        """
        Custom Evaluation Metric: Pearson IC.
        
        Supports both:
        - Native lgb.train() API: train_data is lgb.Dataset
        - sklearn API: train_data is numpy array (labels)
        
        Args:
            preds: Model predictions
            train_data: lgb.Dataset object or numpy array (labels)
            
        Returns:
            Tuple of (metric_name, value, is_higher_better)
        """
        # Handle sklearn API (train_data is numpy array of labels)
        if isinstance(train_data, np.ndarray):
            labels = train_data
            weights = None
            groups = None
        elif hasattr(train_data, 'get_label'):
            # Native lgb.train() API
            labels = train_data.get_label()
            weights = train_data.get_weight()
            groups = train_data.get_group()
        else:
            # Unknown format, try to use as labels
            labels = np.asarray(train_data)
            weights = None
            groups = None
        
        preds, labels = np.asarray(preds), np.asarray(labels)
        if preds.ndim != 1 or labels.ndim != 1 or preds.shape != labels.shape:
            raise ValueError("predictions and labels must be one-dimensional arrays of equal length")
        QlibCustomObjectives._validate_metadata(len(preds), weights, groups)

        if groups is not None and len(groups) > 0:
            # Cross-sectional IC
            ics = []
            current_idx = 0
            for g_size in groups:
                g_size = int(g_size)
                slc = slice(current_idx, current_idx + g_size)
                
                p = preds[slc]
                y = labels[slc]
                w = weights[slc] if weights is not None else None

                if w is not None and not np.any(w):
                    current_idx += g_size
                    continue
                
                if len(p) > 1:
                    if w is None:
                        ic = np.corrcoef(p, y)[0, 1]
                    else:
                        # Weighted correlation
                        mean_p = np.average(p, weights=w)
                        mean_y = np.average(y, weights=w)
                        cov = np.sum(w * (p - mean_p) * (y - mean_y)) / np.sum(w)
                        std_p = np.sqrt(np.sum(w * (p - mean_p)**2) / np.sum(w))
                        std_y = np.sqrt(np.sum(w * (y - mean_y)**2) / np.sum(w))
                        if std_p > 1e-8 and std_y > 1e-8:
                            ic = cov / (std_p * std_y)
                        else:
                            ic = np.nan
                    
                    if not np.isnan(ic):
                        ics.append(ic)
                
                current_idx += g_size
            
            final_ic = np.mean(ics) if ics else 0.0
            return 'ic', float(final_ic), True
        
        else:
            # Global IC
            if weights is not None:
                mean_p = np.average(preds, weights=weights)
                mean_y = np.average(labels, weights=weights)
                cov = np.sum(weights * (preds - mean_p) * (labels - mean_y))
                std_p = np.sqrt(np.sum(weights * (preds - mean_p)**2))
                std_y = np.sqrt(np.sum(weights * (labels - mean_y)**2))
                ic = cov / (std_p * std_y) if std_p > 1e-8 and std_y > 1e-8 else 0.0
            else:
                if len(preds) < 2 or len(labels) < 2:
                    ic = 0.0
                else:
                    ic = np.corrcoef(preds, labels)[0, 1]
            
            return 'ic', float(ic) if not np.isnan(ic) else 0.0, True

    @staticmethod
    def sharpe_surrogate_loss(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sharpe Surrogate Loss for LightGBM.
        
        Optimizes a differentiable approximation of Top-K Sharpe ratio using
        softmax-weighted portfolio returns.
        
        Key insight: Use softmax(predictions / temperature) as portfolio weights,
        then maximize the Sharpe ratio of the weighted returns.
        
        Loss = -mean(weighted_returns) / (std(weighted_returns) + eps)
        
        Supports both sklearn API and native lgb.train() API (arg order resolved
        by :meth:`_unpack_objective_args`).

        Returns:
            Tuple of (gradient, hessian) arrays
        """
        preds, labels, weights, groups = QlibCustomObjectives._unpack_objective_args(y_true, y_pred)
        if groups is not None and len(groups) > 0:
            return QlibCustomObjectives._cross_sectional_sharpe_loss(preds, labels, groups, weights)
        return QlibCustomObjectives._global_sharpe_loss(preds, labels, weights)
    
    @staticmethod
    def _global_sharpe_loss(
        preds: np.ndarray,
        labels: np.ndarray,
        weights: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Global Sharpe Surrogate Loss using softmax-weighted portfolio.
        
        Uses softmax to create differentiable portfolio weights from predictions,
        then computes gradients to maximize the weighted return / volatility ratio.
        
        Args:
            preds: Model predictions
            labels: True returns
            weights: Optional sample weights (not used for portfolio, used for gradient scaling)
            
        Returns:
            Tuple of (gradient, hessian) arrays
        """
        n = len(preds)
        if n < 2:
            return np.zeros(n), np.ones(n)
        
        config = QlibCustomObjectives.SHARPE_CONFIG
        temperature = config.get("temperature", 1.0)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        QlibCustomObjectives._validate_metadata(n, weights)
        eps = 1e-8
        
        # Compute softmax weights from predictions
        # Higher prediction -> higher weight in portfolio
        p_scaled = preds / temperature
        p_max = np.max(p_scaled)
        exp_p = np.exp(p_scaled - p_max)  # Numerical stability
        softmax_weights = exp_p / (np.sum(exp_p) + eps)
        
        # Weighted portfolio return
        weighted_return = np.sum(softmax_weights * labels)
        
        # Weighted portfolio variance (using softmax weights)
        mean_label = np.sum(softmax_weights * labels)
        label_centered = labels - mean_label
        weighted_var = np.sum(softmax_weights * label_centered ** 2)
        weighted_std = np.sqrt(weighted_var + eps)
        
        # Sharpe-like ratio (we want to maximize this, so loss = -sharpe)
        sharpe = weighted_return / weighted_std
        
        # Differentiate the actual V=sum(w*(labels-R)^2), R=sum(w*labels).
        # The epsilon in the softmax denominator means sum(w) is not exactly 1.
        d_var_dw = label_centered ** 2 - 2 * labels * np.sum(softmax_weights * label_centered)
        d_sharpe_dw = labels / weighted_std - weighted_return * d_var_dw / (2 * weighted_std ** 3)
        weighted_d_sharpe = np.sum(d_sharpe_dw * softmax_weights)
        grad = -softmax_weights * (d_sharpe_dw - weighted_d_sharpe) / temperature

        # With epsilon in Z, differentiating the max subtraction adds this term.
        # At ties use the symmetric subgradient of max; away from ties this is exact.
        max_mask = p_scaled == p_max
        missing_mass = eps / (np.sum(exp_p) + eps)
        grad += max_mask * (missing_mass * weighted_d_sharpe / (temperature * np.count_nonzero(max_mask)))
        
        # Apply sample weights if provided
        if weights is not None:
            grad = grad * weights
        
        # Hessian approximation (diagonal, for numerical stability)
        # Use a simple approximation based on softmax variance
        hess = np.abs(softmax_weights * (1 - softmax_weights)) / (temperature ** 2) + eps
        
        return grad, hess
    
    @staticmethod
    def _cross_sectional_sharpe_loss(
        preds: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        weights: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cross-Sectional Sharpe Surrogate Loss.
        
        Computes Sharpe surrogate loss per group (date) and aggregates gradients.
        This is more appropriate for factor models where we select top-K stocks
        per day.
        
        Args:
            preds: Predictions array
            labels: Returns array
            groups: Group sizes array (e.g., number of stocks per date)
            weights: Optional sample weights
            
        Returns:
            Tuple of (gradient, hessian) arrays
        """
        QlibCustomObjectives._validate_metadata(len(preds), weights, groups)
        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        
        config = QlibCustomObjectives.SHARPE_CONFIG
        min_group_size = config.get("min_group_size", 5)
        
        current_idx = 0
        for g_size in groups:
            g_size = int(g_size)
            slc = slice(current_idx, current_idx + g_size)
            
            g_preds = preds[slc]
            g_labels = labels[slc]
            g_weights = weights[slc] if weights is not None else None
            
            if g_size >= min_group_size:
                g_grad, g_hess = QlibCustomObjectives._global_sharpe_loss(
                    g_preds, g_labels, g_weights
                )
            else:
                # For small groups, use IC loss as fallback
                g_grad, g_hess = QlibCustomObjectives._global_pearson_loss(
                    g_preds, g_labels, g_weights
                )
            
            grad[slc] = g_grad
            hess[slc] = g_hess
            
            current_idx += g_size
        
        return grad, hess
    
    @staticmethod
    def sharpe_eval_metric(preds: np.ndarray, train_data) -> Tuple[str, float, bool]:
        """
        Custom Evaluation Metric: Top-K Sharpe Ratio.
        
        Computes the Sharpe ratio of a top-K long-only portfolio based on
        model predictions.
        
        Supports only the native lgb.train() API with a Dataset argument.
        The sklearn evaluation callback uses a different argument order and
        is not supported.
        
        Args:
            preds: Model predictions
            train_data: lgb.Dataset object
            
        Returns:
            Tuple of (metric_name, value, is_higher_better)
        """
        if not (hasattr(train_data, 'get_label') and hasattr(train_data, 'get_group')):
            raise TypeError(
                "sharpe_eval_metric requires a native LightGBM Dataset; "
                "sklearn evaluation is not supported"
            )
        labels = train_data.get_label()
        groups = train_data.get_group()
        
        preds, labels = np.asarray(preds), np.asarray(labels)
        if preds.ndim != 1 or labels.ndim != 1 or preds.shape != labels.shape:
            raise ValueError("predictions and labels must be one-dimensional arrays of equal length")
        QlibCustomObjectives._validate_metadata(len(preds), groups=groups)

        config = QlibCustomObjectives.SHARPE_CONFIG
        top_pct = config.get("top_pct", 0.2)
        eps = 1e-8
        
        if groups is not None and len(groups) > 0:
            # Cross-sectional Sharpe: compute top-K return per group, then Sharpe
            daily_returns = []
            current_idx = 0
            
            for g_size in groups:
                g_size = int(g_size)
                slc = slice(current_idx, current_idx + g_size)
                
                g_preds = preds[slc]
                g_labels = labels[slc]
                
                if g_size >= 3:
                    # Select top-K stocks
                    k = max(1, int(g_size * top_pct))
                    top_k_idx = np.argsort(g_preds)[-k:]
                    top_k_return = np.mean(g_labels[top_k_idx])
                    daily_returns.append(top_k_return)
                
                current_idx += g_size
            
            if len(daily_returns) < 2:
                return 'sharpe', 0.0, True
            
            daily_returns = np.array(daily_returns)
            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns)
            
            # Annualize (assuming daily data, ~252 trading days)
            sharpe = (mean_ret / (std_ret + eps)) * np.sqrt(252)
            
            return 'sharpe', float(sharpe) if not np.isnan(sharpe) else 0.0, True
        
        else:
            # Global top-K Sharpe (single cross-section)
            n = len(preds)
            k = max(1, int(n * top_pct))
            top_k_idx = np.argsort(preds)[-k:]
            
            top_k_labels = labels[top_k_idx]
            mean_ret = np.mean(top_k_labels)
            std_ret = np.std(top_k_labels)
            
            sharpe = mean_ret / (std_ret + eps)
            
            return 'sharpe', float(sharpe) if not np.isnan(sharpe) else 0.0, True
