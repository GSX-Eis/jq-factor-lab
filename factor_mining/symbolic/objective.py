"""Objective functions for symbolic factor mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .expression import ExpressionNode, expression_complexity


def anchor_matrix_from_panel(panel: pd.DataFrame) -> np.ndarray | None:
    """Extract the ``anchor__*`` columns as a row-aligned float matrix.

    Returns ``None`` when no anchor columns exist (marginal-aware search off),
    so callers keep their non-residualized behavior unchanged.
    """
    anchor_columns = sorted(column for column in panel.columns if str(column).startswith("anchor__"))
    if not anchor_columns:
        return None
    return np.asarray(panel[anchor_columns], dtype=float)


def _residualize_subset(pred_subset: np.ndarray, anchor_subset: np.ndarray) -> np.ndarray:
    """Least-squares residual of ``pred_subset`` against anchors (+ intercept)."""
    design = np.column_stack([anchor_subset, np.ones(anchor_subset.shape[0], dtype=float)])
    beta, _, _, _ = np.linalg.lstsq(design, pred_subset, rcond=None)
    return pred_subset - design @ beta


def residualize_by_date(
    pred: pd.Series | np.ndarray, dates: pd.Series, anchors: np.ndarray
) -> pd.Series:
    """Per-date residual of ``pred`` against ``anchors`` (NaN where infeasible).

    Each cross-section residualizes against the anchors present on that date,
    matching the marginal-contribution gate; dates with too few overlapping
    rows for a well-posed regression yield NaN residuals (dropped downstream).
    """
    pred_values = np.asarray(pred, dtype=float)
    anchor_values = np.asarray(anchors, dtype=float)
    residual = np.full(pred_values.shape, np.nan, dtype=float)
    n_anchors = anchor_values.shape[1]
    date_series = pd.Series(dates).reset_index(drop=True)
    for _, index in pd.Series(np.arange(pred_values.size)).groupby(date_series, sort=False):
        positions = np.asarray(index, dtype=np.int64)
        group_pred = pred_values[positions]
        group_anchors = anchor_values[positions]
        finite = np.isfinite(group_pred) & np.all(np.isfinite(group_anchors), axis=1)
        if int(finite.sum()) < max(3, n_anchors + 2):
            continue
        residual[positions[finite]] = _residualize_subset(group_pred[finite], group_anchors[finite])
    index = pred.index if isinstance(pred, pd.Series) else None
    return pd.Series(residual, index=index, dtype=float)


@dataclass(frozen=True)
class _ResidualizationGroup:
    positions: np.ndarray
    anchor_finite: np.ndarray
    min_rows: int
    design_pinv: np.ndarray | None


class ResidualizationContext:
    """Precomputed per-date anchor residualization for repeated predictions."""

    def __init__(self, dates: pd.Series, anchors: np.ndarray) -> None:
        self.n_rows = int(len(dates))
        self.dates = pd.Series(dates).reset_index(drop=True)
        self.anchor_values = np.asarray(anchors, dtype=float)
        self.n_anchors = int(self.anchor_values.shape[1])
        self.groups = self._build_groups()

    def _build_groups(self) -> tuple[_ResidualizationGroup, ...]:
        groups = []
        min_rows = max(3, self.n_anchors + 2)
        grouped_positions = pd.Series(np.arange(self.n_rows)).groupby(
            self.dates, sort=False
        )
        for _, positions_series in grouped_positions:
            positions = np.asarray(positions_series, dtype=np.int64)
            group_anchors = self.anchor_values[positions]
            anchor_finite = np.all(np.isfinite(group_anchors), axis=1)
            design_pinv = None
            if int(anchor_finite.sum()) >= min_rows:
                design = np.column_stack(
                    [
                        group_anchors[anchor_finite],
                        np.ones(int(anchor_finite.sum()), dtype=float),
                    ]
                )
                design_pinv = np.linalg.pinv(
                    design, rcond=np.finfo(float).eps * max(design.shape)
                )
            groups.append(
                _ResidualizationGroup(
                    positions=positions,
                    anchor_finite=anchor_finite,
                    min_rows=min_rows,
                    design_pinv=design_pinv,
                )
            )
        return tuple(groups)

    def residualize(
        self, pred: pd.Series | np.ndarray, *, index: pd.Index | None = None
    ) -> pd.Series:
        pred_values = np.asarray(pred, dtype=float)
        residual = np.full(pred_values.shape, np.nan, dtype=float)
        for group in self.groups:
            positions = group.positions
            group_pred = pred_values[positions]
            group_anchors = self.anchor_values[positions]
            finite = np.isfinite(group_pred) & group.anchor_finite
            if int(finite.sum()) < group.min_rows:
                continue
            pred_subset = group_pred[finite]
            anchor_subset = group_anchors[finite]
            if (
                group.design_pinv is not None
                and bool(np.array_equal(finite, group.anchor_finite))
            ):
                design = np.column_stack(
                    [
                        anchor_subset,
                        np.ones(anchor_subset.shape[0], dtype=float),
                    ]
                )
                beta = group.design_pinv @ pred_subset
                residual[positions[finite]] = pred_subset - design @ beta
            else:
                residual[positions[finite]] = _residualize_subset(
                    pred_subset, anchor_subset
                )
        return pd.Series(residual, index=index, dtype=float)


def rank_ic_by_date(pred: pd.Series, label: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"pred": pred, "label": label, "date": dates})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    values = {}
    for date, group in frame.groupby("date"):
        if (
            len(group) < 3
            or group["pred"].nunique(dropna=True) < 2
            or group["label"].nunique(dropna=True) < 2
        ):
            continue
        corr, _ = spearmanr(
            group["pred"].to_numpy(dtype=float),
            group["label"].to_numpy(dtype=float),
        )
        if np.isfinite(corr):
            values[date] = float(corr)
    return pd.Series(values, dtype=float)


def rank_ic_stability_score(rank_ic_series: pd.Series) -> Dict[str, float]:
    if rank_ic_series.empty:
        return {
            "rank_ic_mean": 0.0,
            "rank_ic_std": 0.0,
            "rank_ic_ir": 0.0,
            "positive_ratio": 0.0,
            "n_dates": 0.0,
            "stability_score": -1.0,
        }
    mean = float(rank_ic_series.mean())
    std = float(rank_ic_series.std(ddof=1)) if len(rank_ic_series) > 1 else 0.0
    ir = mean / std if std > 1e-12 else 0.0
    positive_ratio = float((rank_ic_series > 0).mean())
    stability = abs(mean) * (0.5 + positive_ratio) + 0.05 * abs(ir)
    return {
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "rank_ic_ir": float(ir),
        "positive_ratio": positive_ratio,
        "n_dates": float(len(rank_ic_series)),
        "stability_score": float(stability),
    }


def long_short_return_by_date(
    pred: pd.Series,
    label: pd.Series,
    dates: pd.Series,
    *,
    quantiles: int = 5,
    min_group_size: int = 1,
) -> pd.Series:
    frame = pd.DataFrame({"pred": pred, "label": label, "date": dates})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    values = {}
    quantiles = max(2, int(quantiles))
    min_group_size = max(1, int(min_group_size))
    for date, group in frame.groupby("date"):
        if (
            len(group) < quantiles * min_group_size
            or group["pred"].nunique(dropna=True) < 2
        ):
            continue
        ranked = group.copy()
        try:
            ranked["bucket"] = pd.qcut(
                ranked["pred"].rank(method="first"),
                q=quantiles,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        ranked = ranked.dropna(subset=["bucket"])
        if ranked.empty:
            continue
        top_bucket = ranked["bucket"].max()
        bottom_bucket = ranked["bucket"].min()
        top = ranked.loc[ranked["bucket"] == top_bucket, "label"]
        bottom = ranked.loc[ranked["bucket"] == bottom_bucket, "label"]
        if len(top) < min_group_size or len(bottom) < min_group_size:
            continue
        value = float(top.mean() - bottom.mean())
        if np.isfinite(value):
            values[date] = value
    return pd.Series(values, dtype=float)


def top_excess_return_by_date(
    pred: pd.Series,
    label: pd.Series,
    dates: pd.Series,
    *,
    quantiles: int = 5,
    min_group_size: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Per-date long-only top-quantile excess return and bucket monotonicity.

    Excess is ``top_bucket.mean() - cross_section.mean()`` — the same quantity
    the post-rerank dual gate scores — so the search reward points at the gate
    instead of the (untradeable) long-short spread. Monotonicity is the
    Spearman correlation between bucket index and bucket mean return.
    """
    frame = pd.DataFrame({"pred": pred, "label": label, "date": dates})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    excess_values: dict[object, float] = {}
    mono_values: dict[object, float] = {}
    quantiles = max(2, int(quantiles))
    min_group_size = max(1, int(min_group_size))
    for date, group in frame.groupby("date"):
        stats = _top_excess_from_arrays(
            np.asarray(group["pred"], dtype=float),
            np.asarray(group["label"], dtype=float),
            quantiles=quantiles,
            min_group_size=min_group_size,
        )
        if stats is None:
            continue
        excess_values[date] = stats[0]
        mono_values[date] = stats[1]
    return (
        pd.Series(excess_values, dtype=float),
        pd.Series(mono_values, dtype=float),
    )


def return_proxy_mode(return_proxy_config: dict) -> str:
    """Resolve the return-proxy series semantics.

    ``top_excess`` (default) rewards the long-only top-quantile excess return
    that the post-rerank dual gate scores; ``long_short`` keeps the legacy
    top-minus-bottom spread for reproducing historical behavior.
    """
    mode = str(return_proxy_config.get("mode", "top_excess") or "top_excess")
    return mode if mode in {"top_excess", "long_short"} else "top_excess"


def return_proxy_score(
    return_series: pd.Series,
    return_proxy_config: dict,
    mono_series: pd.Series | None = None,
) -> Dict[str, float]:
    if return_series.empty:
        return {
            "return_proxy_mean": 0.0,
            "return_proxy_std": 0.0,
            "return_proxy_ir": 0.0,
            "return_proxy_win_rate": 0.0,
            "return_proxy_monotonicity": 0.0,
            "return_proxy_n_dates": 0.0,
            "return_proxy_score": 0.0,
        }
    mean = float(return_series.mean())
    std = float(return_series.std(ddof=1)) if len(return_series) > 1 else 0.0
    ir = mean / std if std > 1e-12 else 0.0
    win_rate = float((return_series > 0).mean())
    monotonicity = (
        float(mono_series.mean())
        if mono_series is not None and len(mono_series)
        else 0.0
    )
    score = (
        float(return_proxy_config.get("mean_weight", 1.0)) * mean
        + float(return_proxy_config.get("ir_weight", 0.05)) * ir
        + float(return_proxy_config.get("win_rate_weight", 0.02)) * (win_rate - 0.5)
        + float(return_proxy_config.get("monotonicity_weight", 0.0)) * monotonicity
    )
    return {
        "return_proxy_mean": mean,
        "return_proxy_std": std,
        "return_proxy_ir": float(ir),
        "return_proxy_win_rate": win_rate,
        "return_proxy_monotonicity": monotonicity,
        "return_proxy_n_dates": float(len(return_series)),
        "return_proxy_score": float(score),
    }


@dataclass(frozen=True)
class _LabelGroup:
    positions: np.ndarray
    label_values: np.ndarray
    label_finite: np.ndarray
    label_full_ranks: np.ndarray | None
    anchor_projection: "_AnchorProjection | None" = None


@dataclass(frozen=True)
class _AnchorProjection:
    mask: np.ndarray
    min_rows: int
    design_pinv: np.ndarray
    label_ranks: np.ndarray | None


@dataclass(frozen=True)
class _LabelContext:
    name: str
    values: np.ndarray
    finite: np.ndarray
    groups: tuple[_LabelGroup, ...]


@dataclass(frozen=True)
class _PreparedLabelScore:
    name: str
    coverage: float
    rank_ic: np.ndarray
    returns: np.ndarray
    neg_returns: np.ndarray | None
    mono: np.ndarray | None = None
    neg_mono: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedScore:
    labels: tuple[_PreparedLabelScore, ...]


class ScoringContext:
    """Precomputed date/label structures for repeated candidate scoring."""

    def __init__(
        self,
        dates: pd.Series,
        labels: pd.Series | pd.DataFrame,
        anchors: np.ndarray | None = None,
    ) -> None:
        self.n_rows = int(len(dates))
        self.dates = pd.Series(dates).reset_index(drop=True)
        if anchors is None:
            self.anchor_matrix: np.ndarray | None = None
        else:
            anchor_array = np.asarray(anchors, dtype=float)
            self.anchor_matrix = anchor_array if anchor_array.shape[1] > 0 else None
        if isinstance(labels, pd.DataFrame):
            label_frame = labels.reset_index(drop=True)
        else:
            label_frame = pd.DataFrame(
                {"label": pd.Series(labels).reset_index(drop=True)}
            )
        self.labels = tuple(
            _LabelContext(
                name=str(column),
                values=np.asarray(label_frame[column], dtype=float),
                finite=np.isfinite(np.asarray(label_frame[column], dtype=float)),
                groups=self._build_label_groups(
                    np.asarray(label_frame[column], dtype=float),
                    self.anchor_matrix,
                ),
            )
            for column in label_frame.columns
        )

    @classmethod
    def from_panel(cls, panel: pd.DataFrame) -> "ScoringContext":
        label_columns = [
            column for column in panel.columns if str(column).startswith("label__")
        ]
        labels = panel[label_columns] if label_columns else panel["label"]
        return cls(panel["date"], labels, anchors=anchor_matrix_from_panel(panel))

    def _build_label_groups(
        self, label_values: np.ndarray, anchor_matrix: np.ndarray | None
    ) -> tuple[_LabelGroup, ...]:
        groups = []
        grouped_positions = pd.Series(np.arange(self.n_rows)).groupby(
            self.dates, sort=False
        )
        for _, positions_series in grouped_positions:
            positions = np.asarray(positions_series, dtype=np.int64)
            group_values = label_values[positions]
            label_finite = np.isfinite(group_values)
            full_ranks = None
            anchor_projection = None
            if (
                label_finite.sum() >= 3
                and np.unique(group_values[label_finite]).size >= 2
            ):
                full_ranks = rankdata(
                    group_values[label_finite], method="average"
                ).astype(float)
            if anchor_matrix is not None:
                group_anchors = anchor_matrix[positions]
                anchor_finite = np.all(np.isfinite(group_anchors), axis=1)
                projection_mask = label_finite & anchor_finite
                min_rows = max(3, group_anchors.shape[1] + 2)
                if int(projection_mask.sum()) >= min_rows:
                    projection_labels = group_values[projection_mask]
                    projection_label_ranks = None
                    if np.unique(projection_labels).size >= 2:
                        projection_label_ranks = rankdata(
                            projection_labels, method="average"
                        ).astype(float)
                    design = np.column_stack(
                        [
                            group_anchors[projection_mask],
                            np.ones(int(projection_mask.sum()), dtype=float),
                        ]
                    )
                    design_pinv = np.linalg.pinv(
                        design, rcond=np.finfo(float).eps * max(design.shape)
                    )
                    anchor_projection = _AnchorProjection(
                        mask=projection_mask,
                        min_rows=min_rows,
                        design_pinv=design_pinv,
                        label_ranks=projection_label_ranks,
                    )
            groups.append(
                _LabelGroup(
                    positions=positions,
                    label_values=group_values,
                    label_finite=label_finite,
                    label_full_ranks=full_ranks,
                    anchor_projection=anchor_projection,
                )
            )
        return tuple(groups)


_BUCKET_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _as_float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _pearson_from_ranks(left: np.ndarray, right: np.ndarray) -> float | None:
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denom = float(
        np.sqrt(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
    )
    if denom <= 1e-12:
        return None
    value = float(np.dot(left_centered, right_centered) / denom)
    return value if np.isfinite(value) else None


def _bucket_labels(n_items: int, quantiles: int) -> np.ndarray:
    key = (int(n_items), int(quantiles))
    cached = _BUCKET_CACHE.get(key)
    if cached is not None:
        return cached
    labels = pd.qcut(
        np.arange(1, int(n_items) + 1, dtype=float),
        q=int(quantiles),
        labels=False,
        duplicates="drop",
    )
    result = np.asarray(labels, dtype=np.int64)
    _BUCKET_CACHE[key] = result
    return result


def _long_short_from_arrays(
    pred_values: np.ndarray,
    label_values: np.ndarray,
    *,
    quantiles: int,
    min_group_size: int,
    descending: bool = False,
) -> float | None:
    if pred_values.size < quantiles * min_group_size or np.unique(pred_values).size < 2:
        return None
    order = np.argsort(-pred_values if descending else pred_values, kind="mergesort")
    buckets = _bucket_labels(pred_values.size, quantiles)
    bottom_bucket = int(buckets.min())
    top_bucket = int(buckets.max())
    ordered_labels = label_values[order]
    top = ordered_labels[buckets == top_bucket]
    bottom = ordered_labels[buckets == bottom_bucket]
    if top.size < min_group_size or bottom.size < min_group_size:
        return None
    value = float(top.mean() - bottom.mean())
    return value if np.isfinite(value) else None


def _top_excess_from_arrays(
    pred_values: np.ndarray,
    label_values: np.ndarray,
    *,
    quantiles: int,
    min_group_size: int,
    descending: bool = False,
) -> tuple[float, float] | None:
    """Long-only top-bucket excess and bucket monotonicity for one date group.

    Returns ``(top_excess, monotonicity)`` where ``top_excess`` is the top
    bucket's mean label minus the full cross-section mean (the gate metric)
    and ``monotonicity`` is the Spearman correlation between bucket index and
    bucket mean label. ``descending=True`` scores the negated orientation
    (the original bottom bucket becomes the long leg).
    """
    if pred_values.size < quantiles * min_group_size or np.unique(pred_values).size < 2:
        return None
    order = np.argsort(-pred_values if descending else pred_values, kind="mergesort")
    buckets = _bucket_labels(pred_values.size, quantiles)
    ordered_labels = label_values[order]
    n_buckets = int(buckets.max()) + 1
    counts = np.bincount(buckets, minlength=n_buckets)
    if counts[-1] < min_group_size:
        return None
    sums = np.bincount(buckets, weights=ordered_labels, minlength=n_buckets)
    bucket_means = sums / np.maximum(counts, 1)
    all_mean = float(ordered_labels.mean())
    top_excess = float(bucket_means[-1] - all_mean)
    if not np.isfinite(top_excess):
        return None
    monotonicity = 0.0
    if n_buckets >= 2 and np.all(np.isfinite(bucket_means)):
        mono = _pearson_from_ranks(
            rankdata(bucket_means, method="average").astype(float),
            np.arange(1, n_buckets + 1, dtype=float),
        )
        if mono is not None:
            monotonicity = float(mono)
    return top_excess, monotonicity


def _residualize_with_projection(
    pred_subset: np.ndarray,
    anchor_subset: np.ndarray,
    projection: _AnchorProjection,
) -> np.ndarray:
    design = np.column_stack(
        [anchor_subset, np.ones(anchor_subset.shape[0], dtype=float)]
    )
    beta = projection.design_pinv @ pred_subset
    return pred_subset - design @ beta


def prepare_score(
    pred: pd.Series | np.ndarray,
    context: ScoringContext,
    config: dict,
    *,
    include_negated: bool = False,
) -> PreparedScore:
    pred_values = _as_float_array(pred)
    pred_finite = np.isfinite(pred_values)
    objective_config = config.get("objective", {})
    use_anchor_projection_cache = not bool(
        config.get("_disable_anchor_projection_cache", False)
    )
    return_proxy_config = objective_config.get("return_proxy", {})
    return_proxy_enabled = bool(return_proxy_config.get("enabled", False))
    proxy_mode = return_proxy_mode(return_proxy_config)
    quantiles = max(2, int(return_proxy_config.get("quantiles", 5)))
    min_group_size = max(1, int(return_proxy_config.get("min_group_size", 1)))
    prepared_labels = []

    for label_context in context.labels:
        coverage = float(
            (pred_finite & label_context.finite).sum() / max(context.n_rows, 1)
        )
        rank_ic_values = []
        return_values = []
        neg_return_values = []
        mono_values = []
        neg_mono_values = []

        anchor_matrix = context.anchor_matrix
        for group in label_context.groups:
            group_pred = pred_values[group.positions]
            mask = np.isfinite(group_pred) & group.label_finite
            group_anchors = None
            min_rows = 3
            if anchor_matrix is not None:
                group_anchors = anchor_matrix[group.positions]
                mask = mask & np.all(np.isfinite(group_anchors), axis=1)
                min_rows = max(3, group_anchors.shape[1] + 2)
            if int(mask.sum()) < min_rows:
                continue

            pred_subset = group_pred[mask]
            label_subset = group.label_values[mask]
            if group_anchors is not None:
                projection = group.anchor_projection
                if (
                    use_anchor_projection_cache
                    and projection is not None
                    and bool(np.array_equal(mask, projection.mask))
                ):
                    pred_subset = _residualize_with_projection(
                        pred_subset, group_anchors[mask], projection
                    )
                else:
                    pred_subset = _residualize_subset(
                        pred_subset, group_anchors[mask]
                    )
            if np.unique(pred_subset).size < 2 or np.unique(label_subset).size < 2:
                continue

            pred_ranks = rankdata(pred_subset, method="average").astype(float)
            if (
                anchor_matrix is None
                and bool(np.array_equal(mask, group.label_finite))
                and group.label_full_ranks is not None
            ):
                label_ranks = group.label_full_ranks
            elif (
                use_anchor_projection_cache
                and group.anchor_projection is not None
                and bool(np.array_equal(mask, group.anchor_projection.mask))
                and group.anchor_projection.label_ranks is not None
            ):
                label_ranks = group.anchor_projection.label_ranks
            else:
                label_ranks = rankdata(label_subset, method="average").astype(float)
            corr = _pearson_from_ranks(pred_ranks, label_ranks)
            if corr is not None:
                rank_ic_values.append(corr)

            if return_proxy_enabled:
                if proxy_mode == "long_short":
                    value = _long_short_from_arrays(
                        pred_subset,
                        label_subset,
                        quantiles=quantiles,
                        min_group_size=min_group_size,
                        descending=False,
                    )
                    if value is not None:
                        return_values.append(value)
                    if include_negated:
                        neg_value = _long_short_from_arrays(
                            pred_subset,
                            label_subset,
                            quantiles=quantiles,
                            min_group_size=min_group_size,
                            descending=True,
                        )
                        if neg_value is not None:
                            neg_return_values.append(neg_value)
                else:
                    stats = _top_excess_from_arrays(
                        pred_subset,
                        label_subset,
                        quantiles=quantiles,
                        min_group_size=min_group_size,
                        descending=False,
                    )
                    if stats is not None:
                        return_values.append(stats[0])
                        mono_values.append(stats[1])
                    if include_negated:
                        neg_stats = _top_excess_from_arrays(
                            pred_subset,
                            label_subset,
                            quantiles=quantiles,
                            min_group_size=min_group_size,
                            descending=True,
                        )
                        if neg_stats is not None:
                            neg_return_values.append(neg_stats[0])
                            neg_mono_values.append(neg_stats[1])

        prepared_labels.append(
            _PreparedLabelScore(
                name=label_context.name,
                coverage=coverage,
                rank_ic=np.asarray(rank_ic_values, dtype=float),
                returns=np.asarray(return_values, dtype=float),
                neg_returns=np.asarray(neg_return_values, dtype=float)
                if include_negated
                else None,
                mono=np.asarray(mono_values, dtype=float),
                neg_mono=np.asarray(neg_mono_values, dtype=float)
                if include_negated
                else None,
            )
        )
    return PreparedScore(labels=tuple(prepared_labels))


def _rank_ic_stability_score_array(values: np.ndarray) -> Dict[str, float]:
    return rank_ic_stability_score(pd.Series(values, dtype=float))


def _apply_return_proxy_values(
    metrics: Dict[str, float],
    return_values: np.ndarray,
    objective_config: dict,
    mono_values: np.ndarray | None = None,
) -> Dict[str, float]:
    return_proxy_config = objective_config.get("return_proxy", {})
    adjusted = dict(metrics)
    if not return_proxy_config.get("enabled", False):
        return adjusted
    proxy_metrics = return_proxy_score(
        pd.Series(return_values, dtype=float),
        return_proxy_config,
        mono_series=pd.Series(mono_values, dtype=float)
        if mono_values is not None
        else None,
    )
    contribution = float(return_proxy_config.get("weight", 1.0)) * float(
        proxy_metrics["return_proxy_score"]
    )
    adjusted.update(proxy_metrics)
    adjusted["return_proxy_contribution"] = float(contribution)
    adjusted["stability_score"] = float(
        adjusted.get("stability_score", 0.0) + contribution
    )
    return adjusted


def summarize_prepared_score(
    prepared: PreparedScore,
    expr: ExpressionNode,
    config: dict,
    *,
    sign: int = 1,
) -> Dict[str, float]:
    objective_config = config.get("objective", {})
    stability_scores = []
    metrics: Dict[str, float] = {}
    multi_label = len(prepared.labels) > 1 or (
        prepared.labels and prepared.labels[0].name.startswith("label__")
    )

    for label_score in prepared.labels:
        rank_ic_values = label_score.rank_ic if sign >= 0 else -label_score.rank_ic
        horizon_metrics = _apply_objective_quality(
            _rank_ic_stability_score_array(rank_ic_values),
            label_score.coverage,
            objective_config,
        )
        if sign >= 0:
            return_values = label_score.returns
            mono_values = label_score.mono
        else:
            return_values = label_score.neg_returns
            mono_values = label_score.neg_mono
            if return_values is None:
                # Sign-flip fallback is only exact for the symmetric long-short
                # spread; the top-excess of the negated factor is the original
                # bottom-bucket excess, which needs the dedicated descending
                # pass (include_negated=True). Without it, contribute nothing
                # rather than a wrong value.
                if return_proxy_mode(objective_config.get("return_proxy", {})) == "long_short":
                    return_values = -label_score.returns
                else:
                    return_values = np.asarray([], dtype=float)
                mono_values = None
        horizon_metrics = _apply_return_proxy_values(
            horizon_metrics, return_values, objective_config, mono_values
        )
        stability_scores.append(float(horizon_metrics["stability_score"]))

        if multi_label:
            safe_name = str(label_score.name).replace("label__", "", 1)
            for key, value in horizon_metrics.items():
                metrics[f"{safe_name}_{key}"] = float(value)
        else:
            metrics.update(
                {key: float(value) for key, value in horizon_metrics.items()}
            )

    if multi_label:
        if stability_scores:
            values = np.asarray(stability_scores, dtype=float)
            std_penalty = float(
                config.get("label", {}).get("horizon_std_penalty", 0.25)
            )
            metrics["horizon_count"] = float(len(values))
            metrics["horizon_score_mean"] = float(values.mean())
            metrics["horizon_score_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            metrics["stability_score"] = float(
                metrics["horizon_score_mean"]
                - std_penalty * metrics["horizon_score_std"]
            )
        else:
            metrics = rank_ic_stability_score(pd.Series(dtype=float))

    penalty = complexity_penalty(
        expr, float(objective_config.get("complexity_penalty", 0.0))
    )
    metrics["complexity_penalty"] = penalty
    metrics["score"] = float(metrics["stability_score"] - penalty)
    return metrics


def score_expression_fast(
    pred: pd.Series | np.ndarray,
    context: ScoringContext,
    expr: ExpressionNode,
    config: dict,
) -> Dict[str, float]:
    return summarize_prepared_score(prepare_score(pred, context, config), expr, config)


def _prediction_coverage(pred: pd.Series, label: pd.Series) -> float:
    frame = pd.DataFrame({"pred": pred, "label": label}).replace(
        [np.inf, -np.inf], np.nan
    )
    if frame.empty:
        return 0.0
    return float(frame.dropna().shape[0] / frame.shape[0])


def _objective_quality_penalty(
    metrics: Dict[str, float], coverage: float, objective_config: dict
) -> float:
    penalty = 0.0

    min_coverage = objective_config.get("min_prediction_coverage")
    if min_coverage is not None:
        penalty += max(0.0, float(min_coverage) - float(coverage)) * float(
            objective_config.get("coverage_penalty_weight", 1.0)
        )

    min_n_dates = objective_config.get("min_n_dates")
    if min_n_dates is not None:
        target = max(1.0, float(min_n_dates))
        observed = float(metrics.get("n_dates", 0.0))
        penalty += max(0.0, (target - observed) / target) * float(
            objective_config.get("n_dates_penalty_weight", 1.0)
        )

    min_positive_ratio = objective_config.get("min_positive_ratio")
    if min_positive_ratio is not None:
        penalty += max(
            0.0, float(min_positive_ratio) - float(metrics.get("positive_ratio", 0.0))
        ) * float(objective_config.get("positive_ratio_penalty_weight", 1.0))

    min_sign_consistency = objective_config.get("min_sign_consistency")
    if min_sign_consistency is not None:
        positive_ratio = float(metrics.get("positive_ratio", 0.0))
        sign_consistency = max(positive_ratio, 1.0 - positive_ratio)
        penalty += max(0.0, float(min_sign_consistency) - sign_consistency) * float(
            objective_config.get("sign_consistency_penalty_weight", 1.0)
        )

    return float(penalty)


def _apply_objective_quality(
    metrics: Dict[str, float],
    coverage: float,
    objective_config: dict,
) -> Dict[str, float]:
    raw_stability = float(metrics.get("stability_score", 0.0))
    quality_penalty = _objective_quality_penalty(metrics, coverage, objective_config)
    adjusted = dict(metrics)
    adjusted["prediction_coverage"] = float(coverage)
    adjusted["raw_stability_score"] = raw_stability
    adjusted["quality_penalty"] = float(quality_penalty)
    adjusted["stability_score"] = float(raw_stability - quality_penalty)
    return adjusted


def _apply_return_proxy(
    metrics: Dict[str, float],
    pred: pd.Series,
    label: pd.Series,
    dates: pd.Series,
    objective_config: dict,
) -> Dict[str, float]:
    return_proxy_config = objective_config.get("return_proxy", {})
    adjusted = dict(metrics)
    if not return_proxy_config.get("enabled", False):
        return adjusted

    mono: pd.Series | None = None
    if return_proxy_mode(return_proxy_config) == "long_short":
        returns = long_short_return_by_date(
            pred,
            label,
            dates,
            quantiles=int(return_proxy_config.get("quantiles", 5)),
            min_group_size=int(return_proxy_config.get("min_group_size", 1)),
        )
    else:
        returns, mono = top_excess_return_by_date(
            pred,
            label,
            dates,
            quantiles=int(return_proxy_config.get("quantiles", 5)),
            min_group_size=int(return_proxy_config.get("min_group_size", 1)),
        )
    proxy_metrics = return_proxy_score(returns, return_proxy_config, mono_series=mono)
    contribution = float(return_proxy_config.get("weight", 1.0)) * float(
        proxy_metrics["return_proxy_score"]
    )
    adjusted.update(proxy_metrics)
    adjusted["return_proxy_contribution"] = float(contribution)
    adjusted["stability_score"] = float(
        adjusted.get("stability_score", 0.0) + contribution
    )
    return adjusted


def complexity_penalty(expr: ExpressionNode, weight: float) -> float:
    return float(expression_complexity(expr) * weight)


def score_expression(
    pred: pd.Series,
    label: pd.Series | pd.DataFrame,
    dates: pd.Series,
    expr: ExpressionNode,
    config: dict,
    anchors: np.ndarray | None = None,
) -> Dict[str, float]:
    objective_config = config.get("objective", {})
    if anchors is not None:
        pred = residualize_by_date(pred, dates, anchors)
    if isinstance(label, pd.DataFrame):
        stability_scores = []
        metrics: Dict[str, float] = {}
        for column in label.columns:
            rank_ic = rank_ic_by_date(pred, label[column], dates)
            horizon_metrics = _apply_objective_quality(
                rank_ic_stability_score(rank_ic),
                _prediction_coverage(pred, label[column]),
                objective_config,
            )
            horizon_metrics = _apply_return_proxy(
                horizon_metrics, pred, label[column], dates, objective_config
            )
            stability_scores.append(float(horizon_metrics["stability_score"]))
            safe_name = str(column).replace("label__", "", 1)
            for key, value in horizon_metrics.items():
                metrics[f"{safe_name}_{key}"] = float(value)

        if stability_scores:
            values = np.asarray(stability_scores, dtype=float)
            std_penalty = float(
                config.get("label", {}).get("horizon_std_penalty", 0.25)
            )
            metrics["horizon_count"] = float(len(values))
            metrics["horizon_score_mean"] = float(values.mean())
            metrics["horizon_score_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            metrics["stability_score"] = float(
                metrics["horizon_score_mean"]
                - std_penalty * metrics["horizon_score_std"]
            )
        else:
            metrics = rank_ic_stability_score(pd.Series(dtype=float))
    else:
        rank_ic = rank_ic_by_date(pred, label, dates)
        metrics = _apply_objective_quality(
            rank_ic_stability_score(rank_ic),
            _prediction_coverage(pred, label),
            objective_config,
        )
        metrics = _apply_return_proxy(metrics, pred, label, dates, objective_config)
    penalty = complexity_penalty(
        expr, float(objective_config.get("complexity_penalty", 0.0))
    )
    metrics["complexity_penalty"] = penalty
    metrics["score"] = float(metrics["stability_score"] - penalty)
    return metrics
