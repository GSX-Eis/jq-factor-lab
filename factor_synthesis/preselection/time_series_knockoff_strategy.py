# -*- coding: utf-8 -*-
"""Time-series Knockoff inference using date subsampling and e-values."""

from typing import List

import numpy as np
import pandas as pd

from .base import PreSelectionStrategy
from .gaussian_knockoff_strategy import GaussianKnockoffPreSelectionStrategy
from .registry import register_strategy


@register_strategy
class TimeSeriesKnockoffPreSelectionStrategy(PreSelectionStrategy):
    """Apply Knockoff+ to interlaced date subsamples and aggregate e-values."""

    name = "time_series_knockoff"

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        self._performance_report = pd.DataFrame()
        self._performance_report.attrs['report_lifecycle_version'] = 'per_call_v2'
        self._validate_inputs(X, y)
        target_fdr = float(self.config.get("fdr", 0.2))
        subsample_fdr = float(self.config.get("subsample_fdr", target_fdr / 2))
        if not 0.0 <= target_fdr <= 1.0:
            raise ValueError("fdr must be finite and between 0 and 1")
        if not 0.0 <= subsample_fdr <= 1.0:
            raise ValueError("subsample_fdr must be finite and between 0 and 1")
        common = X.index.intersection(y.index)
        frame = X.loc[common]
        target = y.loc[common]
        columns = list(frame.columns)
        if frame.empty or not columns:
            return []

        dates = pd.to_datetime(frame.index.get_level_values(0)).normalize()
        unique_dates = np.sort(dates.unique())
        subsampling_q = max(0, int(self.config.get("subsampling_q", 1)))
        group_count = subsampling_q + 1
        base_seed = int(self.config.get("random_seed", 42))
        e_values = pd.Series(0.0, index=columns)
        selected_counts = pd.Series(0, index=columns, dtype=int)
        group_diagnostics = []

        for group in range(group_count):
            group_dates = unique_dates[group::group_count]
            mask = np.isin(dates, group_dates)
            child_config = dict(self.config)
            child_config.update(
                fdr=subsample_fdr,
                random_seed=base_seed + group,
            )
            selector = GaussianKnockoffPreSelectionStrategy(
                child_config, self.logger
            )
            selected = selector.select(frame.loc[mask], target.loc[mask])
            report = selector.get_performance_report()
            attrs = dict(report.attrs)
            threshold = float(attrs.get("threshold", float("inf")))
            threshold_negative_count = (
                int((report["knockoff_statistic"] <= -threshold).sum())
                if "knockoff_statistic" in report else 0
            )
            if selected:
                e_values.loc[selected] += len(columns) / (1 + threshold_negative_count)
                selected_counts.loc[selected] += 1
            group_diagnostics.append(
                {
                    "group": group,
                    "date_count": len(group_dates),
                    "row_count": int(mask.sum()),
                    "selected": selected,
                    "factor_statistics": report.to_dict(orient="records"),
                    **attrs,
                    "threshold_negative_count": threshold_negative_count,
                }
            )

        e_values /= group_count
        ordered = np.sort(e_values.to_numpy())[::-1]
        ranks = np.arange(1, len(columns) + 1)
        qualifying = ordered >= len(columns) / (target_fdr * ranks)
        selected = []
        cutoff = float("inf")
        if qualifying.any():
            selected_rank = int(np.flatnonzero(qualifying)[-1] + 1)
            cutoff = len(columns) / (target_fdr * selected_rank)
            selected = e_values[e_values >= cutoff].index.tolist()

        self._performance_report = pd.DataFrame(
            {
                "factor_name": columns,
                "e_value": e_values.reindex(columns).to_numpy(),
                "subsample_selected_count": selected_counts.reindex(columns).to_numpy(),
                "selected": [name in selected for name in columns],
            }
        )
        self._performance_report.attrs.update(
            {
                "strategy": self.name,
                "report_lifecycle_version": "per_call_v2",
                "e_value_version": "threshold_negative_v2",
                "subsampling_q": subsampling_q,
                "group_count": group_count,
                "subsample_fdr": subsample_fdr,
                "fdr": target_fdr,
                "e_value_cutoff": cutoff,
                "selected_count": len(selected),
                "failure_reason": "" if selected else "e_bh_empty",
                "groups": group_diagnostics,
            }
        )
        self.logger.info(
            "[TSKI] q=%d, subsample_fdr=%.3f, target_fdr=%.3f, selected=%d/%d",
            subsampling_q,
            subsample_fdr,
            target_fdr,
            len(selected),
            len(columns),
        )
        return selected
