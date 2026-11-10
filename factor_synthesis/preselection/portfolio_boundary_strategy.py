"""Fold-local TopK replacement preselection against a fixed baseline."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from factor_synthesis.pipeline.size_conditional_method import (
    prepare_top10_context,
    top10_net_utility,
)

from .base import PreSelectionStrategy
from .registry import register_strategy


@register_strategy
class PortfolioBoundaryReplacementStrategy(PreSelectionStrategy):
    """Keep compact sleeves with positive net TopK replacement contribution."""

    name = "portfolio_boundary_replacement"

    def __init__(self, config: dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.baseline_factor = str(config.get("baseline_factor", "")).strip()
        self.top_k = int(config.get("top_k", 10))
        self.sleeve_count = int(config.get("sleeve_count", 8))
        self.max_factors_per_sleeve = int(config.get("max_factors_per_sleeve", 4))
        self.max_similarity = float(config.get("max_intra_sleeve_similarity", 0.8))
        self.turnover_cost_rate = float(config.get("turnover_cost_rate", 0.003))
        raw_gamma = config.get("residual_gamma")
        self.residual_gamma = float(raw_gamma) if raw_gamma is not None else None
        self.holding_period = int(config.get("holding_period", 1))
        self.min_observations = int(config.get("min_fold_observations", 60))
        self.selection_fraction = float(config.get("selection_fraction", 0.25))
        self.mandatory_factors = tuple(
            str(factor) for factor in config.get("mandatory_factors", [])
        )
        self.fallback = bool(config.get("fallback_on_missing_baseline", False))
        self.empty_on_missing_baseline = bool(
            config.get("empty_on_missing_baseline", False)
        )
        if config.get("sleeve_similarity_metric", "spearman_rank") != "spearman_rank":
            raise ValueError(
                "portfolio boundary supports only sleeve_similarity_metric='spearman_rank'"
            )
        if config.get("replacement_score", "net_topk_replacement_vs_baseline") != (
            "net_topk_replacement_vs_baseline"
        ):
            raise ValueError(
                "portfolio boundary supports only replacement_score='net_topk_replacement_vs_baseline'"
            )
        if not self.baseline_factor:
            raise ValueError("baseline_factor is required")
        if min(self.top_k, self.sleeve_count, self.max_factors_per_sleeve) < 1:
            raise ValueError("top_k and sleeve limits must be positive")
        if not 0.0 <= self.max_similarity <= 1.0:
            raise ValueError("max_intra_sleeve_similarity must be between 0 and 1")
        if not math.isfinite(self.turnover_cost_rate):
            raise ValueError("turnover_cost_rate must be finite")
        if self.turnover_cost_rate < 0 or self.holding_period < 1:
            raise ValueError(
                "turnover cost must be non-negative and holding_period positive"
            )
        if self.residual_gamma is not None and (
            not math.isfinite(self.residual_gamma) or self.residual_gamma < 0
        ):
            raise ValueError("residual_gamma must be finite and non-negative")
        if self.min_observations < 2:
            raise ValueError("min_fold_observations must be at least 2")
        if not 0.0 < self.selection_fraction <= 1.0:
            raise ValueError("selection_fraction must be in (0, 1]")

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        baseline_scores: Optional[pd.Series] = None,
    ) -> list[str]:
        self._performance_report = None
        self._validate_inputs(X, y)
        if not self.config.get("enabled", True):
            return list(X.columns)
        if baseline_scores is None:
            return self._fallback_or_raise(X, "baseline scores unavailable")

        common = X.index.intersection(y.index).intersection(baseline_scores.index)
        X_aligned = X.loc[common]
        returns = y.loc[common]
        baseline = baseline_scores.loc[common]
        if X_aligned.empty:
            return self._fallback_or_raise(X, "no aligned factor/return/baseline rows")

        ranked = X_aligned.groupby(level="date", sort=False).rank(
            method="average", pct=True
        )
        factor_rows: list[dict[str, Any]] = []
        for factor in sorted(ranked.columns):
            orientations = [
                (direction, score)
                for direction in (1, -1)
                if (
                    score := self._score_signal(
                        direction * ranked[factor], returns, baseline
                    )
                )
                is not None
            ]
            if orientations:
                direction, score = max(
                    orientations,
                    key=lambda item: (item[1]["net_contribution"], item[0]),
                )
                factor_rows.append(
                    {
                        "record_type": "factor",
                        "name": factor,
                        "direction": direction,
                        **score,
                    }
                )
        if not factor_rows:
            return self._fallback_or_raise(
                X, "insufficient complete input dates or rebalance observations"
            )
        factor_rows.sort(key=lambda row: (-row["net_contribution"], row["name"]))
        ordered_all_factors = [row["name"] for row in factor_rows]
        ordered_factors = [
            row["name"] for row in factor_rows if row["net_contribution"] > 0
        ]
        directions = {row["name"]: row["direction"] for row in factor_rows}
        aligned_ranked = ranked[ordered_factors].mul(
            pd.Series(directions), axis="columns"
        )
        similarity = ranked[ordered_factors].corr(method="spearman").abs()
        # ponytail: greedy sleeves are the fixed portfolio boundary ceiling; add an optimizer only
        # if this selector proves portfolio alpha and sleeve packing becomes limiting.
        sleeves: list[list[str]] = []
        for factor in ordered_factors:
            destination = next(
                (
                    sleeve
                    for sleeve in sleeves
                    if len(sleeve) < self.max_factors_per_sleeve
                    and all(
                        self._similarity(similarity, factor, member)
                        <= self.max_similarity
                        for member in sleeve
                    )
                ),
                None,
            )
            if destination is not None:
                destination.append(factor)
            elif len(sleeves) < self.sleeve_count:
                sleeves.append([factor])
            if len(sleeves) == self.sleeve_count and all(
                len(sleeve) == self.max_factors_per_sleeve for sleeve in sleeves
            ):
                break

        sleeve_rows: list[dict[str, Any]] = []
        for number, members in enumerate(sleeves, start=1):
            score = self._score_signal(
                aligned_ranked[members].mean(axis=1), returns, baseline
            )
            if score is None:
                continue
            sleeve_rows.append(
                {
                    "record_type": "sleeve",
                    "name": f"sleeve_{number:02d}",
                    "members": list(members),
                    **score,
                }
            )
        sleeve_rows.sort(key=lambda row: (-row["net_contribution"], row["name"]))
        positive_sleeves = [row for row in sleeve_rows if row["net_contribution"] > 0]
        keep = max(1, math.ceil(len(positive_sleeves) * self.selection_fraction))
        selected_sleeves = positive_sleeves[:keep]
        selected_set = {
            factor for sleeve in selected_sleeves for factor in sleeve["members"]
        }
        selected_set.update(
            factor for factor in self.mandatory_factors if factor in X.columns
        )
        selected = [factor for factor in ordered_all_factors if factor in selected_set]
        for factor in self.mandatory_factors:
            if factor in X.columns and factor not in selected:
                selected.append(factor)
                factor_rows.append({
                    "record_type": "mandatory",
                    "name": factor,
                    "selection_reason": "no_valid_score",
                })
        selected_sleeve_names = {row["name"] for row in selected_sleeves}
        for row in factor_rows:
            row["selected"] = row["name"] in selected_set
            row["members"] = [row["name"]]
        for row in sleeve_rows:
            row["selected"] = row["name"] in selected_sleeve_names

        self._performance_report = pd.DataFrame(factor_rows + sleeve_rows)
        dates = pd.DatetimeIndex(common.get_level_values("date"))
        self._performance_report.attrs.update(
            {
                "baseline_factor": self.baseline_factor,
                "report_lifecycle_version": "per_call_v2",
                "mandatory_selection_version": "present_mandatory_v2",
                "fallback_reason": None,
                "input_start": str(dates.min().date()),
                "input_end": str(dates.max().date()),
                "input_row_count": len(common),
                "input_date_count": int(dates.nunique()),
                "input_date_minimum": self.min_observations,
                "rebalance_observation_minimum": 2,
                "holding_period": self.holding_period,
                "top_k": self.top_k,
                "mandatory_factors": list(self.mandatory_factors),
                "sleeves": [
                    {
                        "name": row["name"],
                        "members": row["members"],
                        "net_contribution": row["net_contribution"],
                        "selected": row["selected"],
                    }
                    for row in sleeve_rows
                ],
            }
        )
        return selected

    def _score_signal(
        self,
        signal: pd.Series,
        returns: pd.Series,
        baseline: pd.Series,
    ) -> Optional[dict[str, float | int]]:
        complete = signal.notna() & returns.notna() & baseline.notna()
        if not complete.any():
            return None
        signal = signal.loc[complete]
        returns = returns.loc[complete]
        baseline = baseline.loc[complete]
        candidate = signal
        if self.residual_gamma is not None:
            residual = signal.groupby(level="date").transform(
                lambda values: (values - values.mean()) / (values.std() + 1e-8)
            )
            candidate = baseline + self.residual_gamma * residual
        input_dates = int(signal.index.get_level_values("date").nunique())
        if input_dates < self.min_observations:
            return None
        context = prepare_top10_context(
            returns.to_numpy(), returns.index, top_k=self.top_k
        )
        _, candidate_gross, candidate_turnover = top10_net_utility(
            candidate.to_numpy(),
            returns.to_numpy(),
            returns.index,
            top_k=self.top_k,
            cost_rate=0.0,
            target_period=self.holding_period,
            prepared_context=context,
        )
        _, baseline_gross, baseline_turnover = top10_net_utility(
            baseline.to_numpy(),
            returns.to_numpy(),
            returns.index,
            top_k=self.top_k,
            cost_rate=0.0,
            target_period=self.holding_period,
            prepared_context=context,
        )
        observed = candidate_gross.index.intersection(baseline_gross.index)
        if len(observed) < 2:
            return None
        gross_delta = float(
            (candidate_gross.loc[observed] - baseline_gross.loc[observed]).mean()
        )
        incremental_turnover = float(candidate_turnover - baseline_turnover)
        return {
            "gross_replacement_delta": gross_delta,
            "candidate_turnover": float(candidate_turnover),
            "baseline_turnover": float(baseline_turnover),
            "incremental_turnover": incremental_turnover,
            "net_contribution": gross_delta
            - self.turnover_cost_rate * incremental_turnover,
            "rebalance_observations": len(observed),
            "complete_input_dates": input_dates,
        }

    @staticmethod
    def _similarity(matrix: pd.DataFrame, left: str, right: str) -> float:
        value = matrix.at[left, right]
        return 1.0 if not np.isfinite(value) else float(value)

    def _fallback_or_raise(self, X: pd.DataFrame, reason: str) -> list[str]:
        if self.empty_on_missing_baseline:
            self._performance_report = pd.DataFrame()
            self._performance_report.attrs.update(
                {
                    "baseline_factor": self.baseline_factor,
                    "report_lifecycle_version": "per_call_v2",
                    "mandatory_selection_version": "present_mandatory_v2",
                    "fallback_reason": reason,
                    "input_row_count": len(X),
                    "input_date_count": int(
                        X.index.get_level_values("date").nunique()
                        if isinstance(X.index, pd.MultiIndex)
                        else 0
                    ),
                }
            )
            return []
        if not self.fallback:
            raise ValueError(reason)
        keep = max(1, math.ceil(len(X.columns) * self.selection_fraction))
        selected = list(dict.fromkeys(
            factor for factor in self.mandatory_factors if factor in X.columns
        ))
        target_count = max(keep, len(selected))
        for factor in X.columns:
            if len(selected) >= target_count:
                break
            if factor not in selected:
                selected.append(factor)
        self._performance_report = pd.DataFrame(
            {
                "record_type": "fallback",
                "name": selected,
                "members": [[factor] for factor in selected],
                "selected": True,
            }
        )
        self._performance_report.attrs.update(
            {
                "baseline_factor": self.baseline_factor,
                "report_lifecycle_version": "per_call_v2",
                "mandatory_selection_version": "present_mandatory_v2",
                "fallback_reason": reason,
                "input_row_count": len(X),
                "input_date_count": int(
                    X.index.get_level_values("date").nunique()
                    if isinstance(X.index, pd.MultiIndex)
                    else 0
                ),
            }
        )
        return selected
