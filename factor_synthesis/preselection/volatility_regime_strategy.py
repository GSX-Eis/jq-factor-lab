# -*- coding: utf-8 -*-
"""
Volatility-regime consensus preselection.

The strategy keeps factors whose cross-sectional rank IC survives multiple
label-volatility regimes. It is intentionally conservative: when regimes cannot
be formed reliably it falls back to a deterministic top-N score instead of
failing the synthesis run.
"""

from __future__ import annotations

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import PreSelectionStrategy
from .registry import register_strategy


@register_strategy
class VolatilityBucketConsensusPreSelectionStrategy(PreSelectionStrategy):
    """Select factors that remain useful across volatility buckets."""

    name = "volatility_bucket_consensus"

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.params = self._parse_config()
        self._diagnostics: Dict[str, Any] = {}

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        self._performance_report = None
        self._diagnostics = {}
        self._validate_inputs(X, y)
        if X.index.has_duplicates or y.index.has_duplicates:
            raise ValueError("Volatility preselection requires unique full indices for X and y")

        initial_count = len(X.columns)
        if not self.config.get("enabled", True):
            selected = list(X.columns)
            self._emit_diagnostics(
                selected=selected,
                initial_count=initial_count,
                perf_df=pd.DataFrame(),
                fallback_reason="disabled",
            )
            return selected

        self.logger.info("[波动分桶一致性] 开始预筛选")
        perf_df = pd.DataFrame()

        try:
            X_aligned, y_aligned = self._align_inputs(X, y)
            if X_aligned.empty or y_aligned.empty:
                return self._fallback_selection(perf_df, X, "empty_aligned_inputs")

            date_level = self._resolve_date_level(X_aligned.index)
            if date_level is None:
                return self._fallback_selection(perf_df, X_aligned, "missing_date_level")

            date_proxy = self._compute_date_volatility_proxy(y_aligned, date_level)
            if date_proxy.empty:
                return self._fallback_selection(perf_df, X_aligned, "empty_volatility_proxy")

            date_regimes = self._assign_regimes(date_proxy)
            if date_regimes.empty or date_regimes.nunique() < 2:
                return self._fallback_selection(perf_df, X_aligned, "insufficient_regime_buckets")

            perf_df = self._score_factors_by_regime(X_aligned, y_aligned, date_regimes, date_level)
            self._performance_report = perf_df
            perf_df.attrs['report_lifecycle_version'] = 'per_call_v2'
            perf_df.attrs['input_index_version'] = 'unique_index_v2'
            if perf_df.empty:
                return self._fallback_selection(perf_df, X_aligned, "empty_factor_scores")

            candidates = perf_df[perf_df["selected_before_corr"]].copy()
            if candidates.empty:
                return self._fallback_selection(perf_df, X_aligned, "no_factor_passed_regime_gates")

            sorted_factors = candidates.sort_values(
                ["score", "factor_name"],
                ascending=[False, True],
            )["factor_name"].tolist()

            max_corr = self.params.get("max_correlation")
            if max_corr is not None and len(sorted_factors) > 1:
                selected = self._filter_by_correlation(X_aligned, sorted_factors, float(max_corr))
            else:
                selected = sorted_factors

            if not selected:
                return self._fallback_selection(perf_df, X_aligned, "all_factors_pruned_by_correlation")

            self._emit_diagnostics(
                selected=selected,
                initial_count=initial_count,
                perf_df=perf_df,
                fallback_reason=None,
            )
            self.logger.info(
                "[波动分桶一致性] 筛选完成: %d -> %d",
                initial_count,
                len(selected),
            )
            return selected

        except LabelIsolationError:
            raise
        except Exception as exc:
            self.logger.exception("[波动分桶一致性] 预筛选异常: %s", exc)
            raise

    def _parse_config(self) -> Dict[str, Any]:
        metrics = self.config.get("metrics", {}) or {}
        return {
            "min_abs_ic": self._to_float(metrics.get("min_abs_ic", 0.02), 0.02),
            "min_regimes": max(1, self._to_int(metrics.get("min_regimes", 2), 2)),
            "consensus_sign": str(metrics.get("consensus_sign", "same") or "same").lower(),
            "max_correlation": self._optional_float(metrics.get("max_correlation")),
            "volatility_proxy": str(self.config.get("volatility_proxy", "label_dispersion") or "label_dispersion"),
            "regime_quantiles": self._parse_quantiles(self.config.get("regime_quantiles", [0.33, 0.66])),
            "fallback_top_n": max(1, self._to_int(self.config.get("fallback_top_n", 5), 5)),
            "min_bucket_samples": max(1, self._to_int(self.config.get("min_bucket_samples", 30), 30)),
            "min_bucket_dates": max(1, self._to_int(self.config.get("min_bucket_dates", 3), 3)),
        }

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(numeric):
            return default
        return numeric

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        numeric = self._to_float(value, np.nan)
        return None if not np.isfinite(numeric) else numeric

    def _parse_quantiles(self, value: Any) -> List[float]:
        if not isinstance(value, (list, tuple)):
            return [0.33, 0.66]
        quantiles = sorted(
            {
                self._to_float(item, np.nan)
                for item in value
                if 0.0 < self._to_float(item, np.nan) < 1.0
            }
        )
        return quantiles or [0.33, 0.66]

    def _resolve_date_level(self, index: pd.Index) -> Optional[Any]:
        if not isinstance(index, pd.MultiIndex):
            return None
        names = list(index.names)
        if "date" in names:
            return "date"
        return 0

    def _align_inputs(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        common_index = X.index.intersection(y.index)
        if common_index.empty:
            return X.iloc[0:0].copy(), y.iloc[0:0].copy()

        X_aligned = X.loc[common_index].replace([np.inf, -np.inf], np.nan)
        y_aligned = y.loc[common_index].replace([np.inf, -np.inf], np.nan)
        valid_y = y_aligned.notna()
        return X_aligned.loc[valid_y], y_aligned.loc[valid_y]

    def _compute_date_volatility_proxy(self, y: pd.Series, date_level: Any) -> pd.Series:
        proxy_name = self.params["volatility_proxy"]
        grouped = y.groupby(level=date_level)
        if proxy_name == "absolute_label":
            proxy = grouped.apply(lambda s: s.abs().mean())
        elif proxy_name == "squared_label":
            proxy = grouped.apply(lambda s: np.square(s).mean())
        else:
            proxy = grouped.std()
        return proxy.replace([np.inf, -np.inf], np.nan).dropna().sort_index()

    def _assign_regimes(self, date_proxy: pd.Series) -> pd.Series:
        if date_proxy.nunique(dropna=True) < 2:
            return pd.Series(dtype=object)

        cut_points = sorted({float(date_proxy.quantile(q)) for q in self.params["regime_quantiles"]})
        cut_points = [point for point in cut_points if np.isfinite(point)]
        cut_points = sorted(set(cut_points))
        if not cut_points:
            return pd.Series(dtype=object)

        bins = [-np.inf] + cut_points + [np.inf]
        bins = [bins[0]] + [b for i, b in enumerate(bins[1:], start=1) if b > bins[i - 1]]
        if len(bins) < 3:
            return pd.Series(dtype=object)

        labels = self._regime_labels(len(bins) - 1)
        regimes = pd.cut(date_proxy, bins=bins, labels=labels, include_lowest=True)
        regimes = regimes.dropna().astype(str)
        if regimes.nunique() < 2:
            return pd.Series(dtype=object)
        return regimes

    @staticmethod
    def _regime_labels(bucket_count: int) -> List[str]:
        if bucket_count == 2:
            suffixes = ["low", "high"]
        elif bucket_count == 3:
            suffixes = ["low", "mid", "high"]
        else:
            suffixes = [f"q{i}" for i in range(bucket_count)]
        return [f"regime_{idx}_{suffix}" for idx, suffix in enumerate(suffixes)]

    def _daily_rank_ic(self, x: pd.Series, y: pd.Series, date_level: Any) -> pd.Series:
        aligned = pd.concat({"x": x, "y": y}, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)

        def _corr(group: pd.DataFrame) -> float:
            if len(group) < 3:
                return np.nan
            x_rank = group["x"].rank(method="average")
            y_rank = group["y"].rank(method="average")
            value = x_rank.corr(y_rank)
            return float(value) if pd.notna(value) else np.nan

        return aligned.groupby(level=date_level, sort=True).apply(_corr).dropna()

    def _score_factors_by_regime(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        date_regimes: pd.Series,
        date_level: Any,
    ) -> pd.DataFrame:
        regime_labels = list(date_regimes.dropna().unique())
        rows: List[Dict[str, Any]] = []

        for factor_name in X.columns:
            daily_ic = self._daily_rank_ic(X[factor_name], y, date_level)
            finite_pairs = np.isfinite(X[factor_name]) & np.isfinite(y.reindex(X.index))
            date_sample_counts = finite_pairs.groupby(level=date_level).sum()
            row: Dict[str, Any] = {"factor_name": factor_name}
            valid_values: List[float] = []
            passing_values: List[float] = []

            for regime_label in regime_labels:
                regime_dates = date_regimes[date_regimes == regime_label].index
                regime_ic = daily_ic[daily_ic.index.isin(regime_dates)].dropna()
                n_dates = int(len(regime_ic))
                n_obs = int(date_sample_counts.reindex(regime_ic.index).fillna(0).sum())
                ic_mean = float(regime_ic.mean()) if n_dates else np.nan
                enough_data = (
                    n_dates >= self.params["min_bucket_dates"]
                    and n_obs >= self.params["min_bucket_samples"]
                    and np.isfinite(ic_mean)
                )
                passed = bool(enough_data and abs(ic_mean) >= self.params["min_abs_ic"])

                row[f"{regime_label}_ic_mean"] = ic_mean if np.isfinite(ic_mean) else None
                row[f"{regime_label}_ic_abs"] = abs(ic_mean) if np.isfinite(ic_mean) else None
                row[f"{regime_label}_n_dates"] = n_dates
                row[f"{regime_label}_n_obs"] = n_obs
                row[f"{regime_label}_passed"] = passed

                if enough_data:
                    valid_values.append(ic_mean)
                if passed:
                    passing_values.append(ic_mean)

            consensus_passed = self._passes_consensus(passing_values)
            passed_regime_count = len(passing_values)
            score = float(np.mean(np.abs(valid_values))) if valid_values else 0.0
            row["score"] = score
            row["passed_regime_count"] = passed_regime_count
            row["consensus_sign_passed"] = consensus_passed
            row["selected_before_corr"] = (
                passed_regime_count >= self.params["min_regimes"] and consensus_passed
            )
            rows.append(row)

        report = pd.DataFrame(rows).sort_values(["score", "factor_name"], ascending=[False, True])
        report.attrs['bucket_sample_version'] = 'valid_ic_finite_pairs_v2'
        return report

    def _passes_consensus(self, regime_ic_values: List[float]) -> bool:
        if not regime_ic_values:
            return False
        mode = self.params["consensus_sign"]
        signs = [np.sign(value) for value in regime_ic_values if np.isfinite(value) and value != 0]
        if not signs:
            return False
        if mode == "any":
            return True
        if mode == "positive":
            return all(sign > 0 for sign in signs)
        if mode == "negative":
            return all(sign < 0 for sign in signs)
        return len(set(signs)) == 1

    def _filter_by_correlation(self, X: pd.DataFrame, sorted_factors: List[str], threshold: float) -> List[str]:
        valid_cols = [col for col in sorted_factors if col in X.columns]
        if len(valid_cols) < 2:
            return valid_cols
        try:
            corr_matrix = X[valid_cols].corr(method="spearman").abs().fillna(0.0).values
            kept_indices: List[int] = []
            for idx in range(len(valid_cols)):
                if not kept_indices:
                    kept_indices.append(idx)
                    continue
                max_corr = corr_matrix[idx, np.array(kept_indices)].max()
                if max_corr <= threshold:
                    kept_indices.append(idx)
            return [valid_cols[idx] for idx in kept_indices]
        except LabelIsolationError:
            raise
        except Exception as exc:
            self.logger.exception("[波动分桶一致性] 相关性筛选失败: %s", exc)
            raise

    def _fallback_selection(self, perf_df: pd.DataFrame, X: pd.DataFrame, reason: str) -> List[str]:
        fallback_n = int(self.params["fallback_top_n"])
        if not perf_df.empty and "score" in perf_df.columns:
            selected = (
                perf_df.sort_values(["score", "factor_name"], ascending=[False, True])
                ["factor_name"]
                .head(fallback_n)
                .tolist()
            )
        else:
            selected = list(X.columns)

        candidate_count = len(selected)
        max_corr = self.params.get("max_correlation")
        if max_corr is not None and candidate_count > 1:
            selected = self._filter_by_correlation(X, selected, float(max_corr))
        shortfall = candidate_count - len(selected)
        if shortfall:
            self.logger.warning(
                "[波动分桶一致性] 保底相关上限优先: %d -> %d, shortfall=%d",
                candidate_count, len(selected), shortfall,
            )
        perf_df.attrs['fallback_correlation_version'] = 'cap_first_v2'
        self._performance_report = perf_df
        self._emit_diagnostics(
            selected=selected,
            initial_count=len(X.columns),
            perf_df=perf_df,
            fallback_reason=reason,
        )
        self.logger.warning("[波动分桶一致性] 使用保底选择: %s", reason)
        return selected

    def _emit_diagnostics(
        self,
        *,
        selected: List[str],
        initial_count: int,
        perf_df: pd.DataFrame,
        fallback_reason: Optional[str],
    ) -> None:
        top_factors: List[Dict[str, Any]] = []
        if not perf_df.empty:
            columns = [
                "factor_name",
                "score",
                "passed_regime_count",
                "consensus_sign_passed",
                "selected_before_corr",
            ]
            top_factors = [
                self._json_safe_dict(record)
                for record in perf_df[columns].head(8).to_dict(orient="records")
            ]

        regime_date_counts = self._regime_date_counts(perf_df)
        payload = {
            "report_lifecycle_version": "per_call_v2",
            "input_index_version": "unique_index_v2",
            "bucket_sample_version": "valid_ic_finite_pairs_v2",
            "diagnostic_type": "volatility_bucket_consensus_preselection",
            "strategy": self.name,
            "initial_factor_count": int(initial_count),
            "selected_factor_count": int(len(selected)),
            "selected_preview": selected[:8],
            "candidate_count_before_corr": int(
                perf_df["selected_before_corr"].sum()
            ) if not perf_df.empty and "selected_before_corr" in perf_df.columns else None,
            "correlation_pruned_count": int(max(0, (
                perf_df["selected_before_corr"].sum() - len(selected)
            ))) if not perf_df.empty and "selected_before_corr" in perf_df.columns else None,
            "regime_labels": regime_date_counts and list(regime_date_counts.keys()) or [],
            "regime_date_counts": regime_date_counts,
            "volatility_proxy": self.params["volatility_proxy"],
            "fallback_used": fallback_reason not in (None, "disabled"),
            "fallback_reason": fallback_reason,
            "top_factors": top_factors,
            "config": {
                "min_abs_ic": self.params["min_abs_ic"],
                "min_regimes": self.params["min_regimes"],
                "consensus_sign": self.params["consensus_sign"],
                "max_correlation": self.params["max_correlation"],
                "regime_quantiles": self.params["regime_quantiles"],
                "fallback_top_n": self.params["fallback_top_n"],
                "min_bucket_samples": self.params["min_bucket_samples"],
                "min_bucket_dates": self.params["min_bucket_dates"],
            },
        }
        if 'fallback_correlation_version' in perf_df.attrs:
            payload['fallback_correlation_version'] = perf_df.attrs['fallback_correlation_version']
        self._diagnostics = payload
        self.logger.info(
            "ALGORITHM_SYNTHESIS_DIAGNOSTICS: %s",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _regime_date_counts(perf_df: pd.DataFrame) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if perf_df.empty:
            return counts
        for col in perf_df.columns:
            if col.startswith("regime_") and col.endswith("_n_dates"):
                label = col[: -len("_n_dates")]
                values = perf_df[col].dropna()
                counts[label] = int(values.max()) if not values.empty else 0
        return counts

    @staticmethod
    def _json_safe_dict(record: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (np.bool_, bool)):
                safe[key] = bool(value)
            elif isinstance(value, (np.integer, int)):
                safe[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                safe[key] = None if not np.isfinite(value) else round(float(value), 6)
            else:
                safe[key] = value
        return safe
