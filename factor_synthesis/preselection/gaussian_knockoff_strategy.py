# -*- coding: utf-8 -*-
"""Second-order Gaussian Knockoff+ preselection."""

from typing import List

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

from .base import PreSelectionStrategy
from .registry import register_strategy


def _knockoff_plus_threshold(statistics: np.ndarray, fdr: float) -> float:
    candidates = np.sort(np.unique(np.abs(statistics[statistics != 0])))
    for threshold in candidates:
        false_discoveries = 1 + np.count_nonzero(statistics <= -threshold)
        discoveries = max(1, np.count_nonzero(statistics >= threshold))
        if false_discoveries / discoveries <= fdr:
            return float(threshold)
    return float("inf")


@register_strategy
class GaussianKnockoffPreSelectionStrategy(PreSelectionStrategy):
    """Select factors with a Gaussian second-order Knockoff+ filter."""

    name = "gaussian_knockoff"

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        self._performance_report = pd.DataFrame()
        self._performance_report.attrs['report_lifecycle_version'] = 'per_call_v2'
        self._performance_report.attrs['covariance_validation_version'] = 'psd_guard_v2'
        self._validate_inputs(X, y)
        fdr = float(self.config.get("fdr", 0.2))
        if not 0.0 <= fdr <= 1.0:
            raise ValueError("fdr must be finite and between 0 and 1")
        common = X.index.intersection(y.index)
        frame = X.loc[common].replace([np.inf, -np.inf], np.nan)
        target = y.loc[common].replace([np.inf, -np.inf], np.nan)
        valid_target = target.notna()
        frame, target = frame.loc[valid_target], target.loc[valid_target]
        if frame.empty:
            return []

        values = frame.to_numpy(dtype=float)
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        values = np.where(np.isnan(values), medians, values)
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        varying = scales > 1e-12
        if not varying.any():
            return []

        columns = np.asarray(frame.columns)[varying]
        values = (values[:, varying] - means[varying]) / scales[varying]
        covariance = LedoitWolf().fit(values).covariance_
        covariance = (covariance + covariance.T) / 2
        min_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
        s_value = min(1.0, max(1e-8, 1.98 * min_eigenvalue))
        if 2 * min_eigenvalue < s_value:
            raise ValueError(
                "Degenerate covariance: 2C-S must be positive semidefinite "
                "for Gaussian knockoffs (psd_guard_v2)"
            )
        s_matrix = np.eye(len(columns)) * s_value
        covariance_inverse = np.linalg.pinv(covariance)
        conditional_covariance = (
            2 * s_matrix - s_matrix @ covariance_inverse @ s_matrix
        )
        eigenvalues, eigenvectors = np.linalg.eigh(
            (conditional_covariance + conditional_covariance.T) / 2
        )
        covariance_root = (
            eigenvectors
            @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
            @ eigenvectors.T
        )
        rng = np.random.default_rng(int(self.config.get("random_seed", 42)))
        knockoffs = (
            values @ (np.eye(len(columns)) - covariance_inverse @ s_matrix)
            + rng.standard_normal(values.shape) @ covariance_root
        )
        augmented = np.column_stack([values, knockoffs])

        dates = pd.to_datetime(frame.index.get_level_values(0)).normalize()
        unique_dates = np.sort(dates.unique())
        n_splits = min(int(self.config.get("cv_folds", 3)), len(unique_dates) - 1)
        if n_splits < 2:
            return []
        purge_dates = max(0, int(self.config.get("purge_dates", 5)))
        splits = []
        for train_dates, validation_dates in TimeSeriesSplit(n_splits=n_splits).split(
            unique_dates
        ):
            if purge_dates:
                train_dates = train_dates[:-purge_dates]
            if not len(train_dates):
                continue
            splits.append(
                (
                    np.flatnonzero(np.isin(dates, unique_dates[train_dates])),
                    np.flatnonzero(np.isin(dates, unique_dates[validation_dates])),
                )
            )
        if len(splits) < 2:
            return []

        model = LassoCV(
            cv=splits,
            n_alphas=int(self.config.get("n_alphas", 30)),
            max_iter=int(self.config.get("max_iter", 5000)),
            random_state=int(self.config.get("random_seed", 42)),
        ).fit(augmented, target.to_numpy(dtype=float))
        original, knockoff = np.split(np.abs(model.coef_), 2)
        statistics = original - knockoff
        threshold = _knockoff_plus_threshold(
            statistics, float(self.config.get("fdr", 0.2))
        )
        selected = columns[statistics >= threshold].tolist()
        positive_count = int(np.count_nonzero(statistics > 0))
        negative_count = int(np.count_nonzero(statistics < 0))
        self._performance_report = pd.DataFrame(
            {
                "factor_name": columns,
                "knockoff_statistic": statistics,
                "selected": statistics >= threshold,
            }
        )
        self._performance_report.attrs.update(
            {
                "strategy": self.name,
                "report_lifecycle_version": "per_call_v2",
                "covariance_validation_version": "psd_guard_v2",
                "fdr": float(self.config.get("fdr", 0.2)),
                "threshold": float(threshold),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "zero_count": int(len(statistics) - positive_count - negative_count),
                "selected_count": len(selected),
                "random_seed": int(self.config.get("random_seed", 42)),
                "lasso_alpha": float(model.alpha_),
                "row_count": len(frame),
                "date_count": len(unique_dates),
                "feature_count": len(columns),
                "failure_reason": "" if selected else "knockoff_threshold_empty",
            }
        )
        self.logger.info(
            "[Knockoff+] FDR=%.3f, selected=%d/%d, positive=%d, negative=%d, threshold=%s",
            float(self.config.get("fdr", 0.2)),
            len(selected),
            len(columns),
            positive_count,
            negative_count,
            threshold,
        )
        return selected
