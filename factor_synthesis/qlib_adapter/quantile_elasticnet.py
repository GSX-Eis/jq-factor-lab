# -*- coding: utf-8 -*-
"""
Quantile ElasticNet via IRLS (Iteratively Reweighted Least Squares)

Solves:
    min_{beta}  (1/n) * sum pinball_loss(y_i - X_i @ beta, tau)
               + alpha * l1_ratio * ||beta||_1
               + 0.5 * alpha * (1 - l1_ratio) * ||beta||_2^2

where pinball_loss(u, tau) = tau * max(u, 0) + (1 - tau) * max(-u, 0).

The IRLS trick recasts the pinball loss as a weighted least squares problem
at each iteration, reusing sklearn's battle-tested ElasticNet coordinate
descent solver.  This preserves exact L1 sparsity (hard zeros) and is
compatible with external sample_weight.

Reference:
    - Chen, Dolado, Gonzalo (2021) "Quantile Factor Models", Econometrica.
    - Hunter & Lange (2000) "Quantile regression via IRLS", J. Comp. Graph. Stat.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import numpy as np
from sklearn.linear_model import ElasticNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pinball (quantile) loss utilities
# ---------------------------------------------------------------------------

def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.8,
) -> float:
    """
    Mean pinball (check / quantile) loss.

    L_tau(u) = tau * max(u, 0) + (1-tau) * max(-u, 0)

    Args:
        y_true: Ground-truth values, shape (n,).
        y_pred: Predicted values, shape (n,).
        quantile: Quantile level in (0, 1).

    Returns:
        Scalar mean pinball loss (lower is better).
    """
    if np.ndim(y_true) != 1 or np.ndim(y_pred) != 1:
        raise ValueError("labels and predictions must be one-dimensional")
    if len(y_true) != len(y_pred):
        raise ValueError("labels and predictions must have equal length")
    residual = y_true - y_pred
    loss = np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)
    return float(np.mean(loss))


def pinball_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.8,
) -> float:
    """
    Negated mean pinball loss (higher is better, sklearn convention).
    """
    return -pinball_loss(y_true, y_pred, quantile)


# ---------------------------------------------------------------------------
# QuantileElasticNet
# ---------------------------------------------------------------------------

class QuantileElasticNet:
    """
    Quantile regression with Elastic Net penalty solved via IRLS.

    At each IRLS iteration the pinball loss is locally approximated by a
    weighted least squares objective whose weights depend on the current
    residuals.  The resulting sub-problem is a standard weighted ElasticNet
    which sklearn solves with coordinate descent.

    The API mirrors ``sklearn.linear_model.ElasticNet`` so it can be used
    as a drop-in replacement inside ``QlibLinearWrapper``.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularisation strength (same semantics as sklearn ElasticNet).
    l1_ratio : float, default=0.5
        Mix between L1 and L2 penalty.  1.0 = pure Lasso, 0.0 = pure Ridge.
    quantile : float, default=0.8
        Target quantile in (0, 1).  0.8 focuses on the 80-th percentile.
    max_iter : int, default=10000
        Maximum iterations for the inner ElasticNet coordinate descent.
    max_irls_iter : int, default=30
        Maximum outer IRLS iterations.
    tol : float, default=1e-5
        Convergence tolerance on the largest absolute coefficient or intercept
        change between IRLS steps.
    irls_eps : float, default=1e-5
        Floor for |residual| in the IRLS weight computation (avoids ÷ 0).
    positive : bool, default=False
        If True, force non-negative coefficients.
    random_state : int or None, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        quantile: float = 0.8,
        max_iter: int = 10_000,
        max_irls_iter: int = 30,
        tol: float = 1e-5,
        irls_eps: float = 1e-5,
        positive: bool = False,
        random_state: int | None = 42,
    ):
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.quantile = quantile
        self.max_iter = max_iter
        self.max_irls_iter = max_irls_iter
        self.tol = tol
        self.irls_eps = irls_eps
        self.positive = positive
        self.random_state = random_state

        # Fitted attributes (set during fit)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_irls_: int = 0

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "QuantileElasticNet":
        """
        Fit the model via IRLS.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Target vector, shape (n_samples,).
            sample_weight: Optional external sample weights, shape (n_samples,).
                           These are multiplied with the IRLS weights at each
                           iteration so both effects compose naturally.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        # --- Step 0: warm-start with standard ElasticNet (MSE loss) ---
        init_model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            positive=self.positive,
            random_state=self.random_state,
        )
        init_model.fit(X, y, sample_weight=sample_weight)

        self.coef_ = init_model.coef_.copy()
        self.intercept_ = float(init_model.intercept_)

        # --- Step 1: IRLS iterations ---
        tau = self.quantile
        eps = self.irls_eps

        for iteration in range(1, self.max_irls_iter + 1):
            y_pred = X @ self.coef_ + self.intercept_
            residual = y - y_pred

            # IRLS weights: derived from sub-gradient of the pinball loss
            #   check loss:  rho_tau(u) = u * (tau - I(u < 0))
            #   IRLS weight:  w_i = tau / max(|r_i|, eps)   if r_i >= 0
            #                        (1-tau) / max(|r_i|, eps)  if r_i <  0
            abs_res = np.maximum(np.abs(residual), eps)
            irls_w = np.where(residual >= 0, tau / abs_res, (1.0 - tau) / abs_res)

            # Normalise IRLS weights to mean=1 so they don't compete with alpha
            irls_mean = irls_w.mean()
            if irls_mean > 0:
                irls_w = irls_w / irls_mean

            # Combine with external sample_weight
            if sample_weight is not None:
                combined_w = irls_w * np.asarray(sample_weight, dtype=np.float64)
            else:
                combined_w = irls_w

            # Solve weighted ElasticNet sub-problem
            sub_model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                positive=self.positive,
                random_state=self.random_state,
                warm_start=False,
            )
            sub_model.fit(X, y, sample_weight=combined_w)

            # Convergence check
            coef_change = np.max(np.abs(sub_model.coef_ - self.coef_))
            intercept_change = abs(sub_model.intercept_ - self.intercept_)
            self.coef_ = sub_model.coef_.copy()
            self.intercept_ = float(sub_model.intercept_)

            if max(coef_change, intercept_change) < self.tol:
                self.n_iter_irls_ = iteration
                break
        else:
            self.n_iter_irls_ = self.max_irls_iter

        return self

    # ------------------------------------------------------------------
    # predict / score
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the negated mean pinball loss (higher is better).

        This differs from ElasticNet.score() which returns R².
        """
        y_pred = self.predict(X)
        return pinball_score(y, y_pred, self.quantile)


# ---------------------------------------------------------------------------
# QuantileElasticNetCV
# ---------------------------------------------------------------------------

class QuantileElasticNetCV:
    """
    Quantile ElasticNet with built-in cross-validated grid search.

    Searches over ``alphas`` x ``l1_ratio`` using mean pinball loss on
    validation folds.  The best (alpha, l1_ratio) is used to refit on the
    full training set.

    The API mirrors ``sklearn.linear_model.ElasticNetCV`` so it can be
    used as a drop-in replacement inside ``QlibLinearWrapper``.

    Parameters
    ----------
    l1_ratio : float or list[float], default=0.5
        L1/L2 mix ratios to search over.
    alphas : array-like or None
        Regularisation strengths to search.  If None a default log-space
        grid is generated.
    quantile : float, default=0.8
        Target quantile in (0, 1).
    cv : int or list of (train_idx, test_idx) tuples, default=5
        Cross-validation strategy.  Accepts pre-computed splits (e.g.
        from PanelPurgedKFold).
    max_iter : int, default=10000
        Max inner ElasticNet iterations.
    max_irls_iter : int, default=30
        Max outer IRLS iterations per model fit.
    tol : float, default=1e-5
        IRLS convergence tolerance.
    positive : bool, default=False
        Force non-negative coefficients.
    random_state : int or None, default=42
        Random seed.
    n_jobs : int, default=1
        Not used (kept for API compatibility).
    """

    def __init__(
        self,
        l1_ratio: float | List[float] = 0.5,
        alphas: np.ndarray | None = None,
        quantile: float = 0.8,
        cv: int | list = 5,
        max_iter: int = 10_000,
        max_irls_iter: int = 30,
        tol: float = 1e-5,
        positive: bool = False,
        random_state: int | None = 42,
        n_jobs: int = 1,
    ):
        self.l1_ratio = l1_ratio if isinstance(l1_ratio, list) else [l1_ratio]
        self.alphas = alphas if alphas is not None else np.logspace(-5, 0, 50)
        self.quantile = quantile
        self.cv = cv
        self.max_iter = max_iter
        self.max_irls_iter = max_irls_iter
        self.tol = tol
        self.positive = positive
        self.random_state = random_state

        # Fitted attributes
        self.alpha_: float = 0.0
        self.l1_ratio_: float = 0.5
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "QuantileElasticNetCV":
        """
        Run cross-validated grid search then refit on full data.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Build CV splits
        cv_splits = self._resolve_cv_splits(X, y)
        if not cv_splits:
            raise ValueError("No CV folds available for QuantileElasticNetCV")

        # Grid search
        best_score = -np.inf
        best_alpha = float(self.alphas[len(self.alphas) // 2])
        best_l1 = self.l1_ratio[0]
        evaluated_candidates = 0

        for alpha in self.alphas:
            for l1 in self.l1_ratio:
                fold_scores: list[float] = []

                for train_idx, val_idx in cv_splits:
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    sw_tr = (
                        sample_weight[train_idx]
                        if sample_weight is not None
                        else None
                    )

                    model = QuantileElasticNet(
                        alpha=float(alpha),
                        l1_ratio=float(l1),
                        quantile=self.quantile,
                        max_iter=self.max_iter,
                        max_irls_iter=self.max_irls_iter,
                        tol=self.tol,
                        positive=self.positive,
                        random_state=self.random_state,
                    )

                    try:
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    except Exception:
                        continue

                    # Evaluate on validation fold using pinball score
                    score = model.score(X_val, y_val)
                    if not np.isnan(score):
                        fold_scores.append(score)

                if fold_scores:
                    evaluated_candidates += 1
                    avg = float(np.mean(fold_scores))
                    if avg > best_score:
                        best_score = avg
                        best_alpha = float(alpha)
                        best_l1 = float(l1)

        if not np.isfinite(best_score):
            raise ValueError("No finite winning CV score for QuantileElasticNetCV")

        self.alpha_ = best_alpha
        self.l1_ratio_ = best_l1

        logger.info(
            "[QuantileElasticNetCV] Best params: alpha=%.6f, l1_ratio=%.2f, "
            "pinball_score=%.6f (tau=%.2f)",
            best_alpha, best_l1, best_score, self.quantile,
        )

        # Refit on full data with best params
        final_model = QuantileElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1,
            quantile=self.quantile,
            max_iter=self.max_iter,
            max_irls_iter=self.max_irls_iter,
            tol=self.tol,
            positive=self.positive,
            random_state=self.random_state,
        )
        final_model.fit(X, y, sample_weight=sample_weight)

        self.coef_ = final_model.coef_
        self.intercept_ = final_model.intercept_
        self.best_score_ = best_score
        self.n_iter_irls_ = final_model.n_iter_irls_
        self.n_splits_ = len(cv_splits)
        self.n_evaluated_candidates_ = evaluated_candidates

        return self

    # ------------------------------------------------------------------
    # predict / score
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return pinball_score(y, y_pred, self.quantile)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _resolve_cv_splits(
        self, X: np.ndarray, y: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Convert cv parameter to list of (train_idx, val_idx) tuples."""
        if isinstance(self.cv, list):
            # Pre-computed splits (e.g. from PanelPurgedKFold)
            return self.cv

        # Integer -> KFold-like splitting (sequential, no shuffle for TS data)
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=int(self.cv), shuffle=False)
        return list(kf.split(X, y))
