"""Random v0 symbolic search backend."""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from typing import Iterator, List, Set

import numpy as np
import pandas as pd

from .expression import (
    ExpressionNode,
    evaluate_expression_cached,
    expression_complexity,
    expression_hash,
    iter_constants,
    replace_constants,
)
from .feature_space import FeatureSpace, validate_exportable_expression
from .objective import (
    ScoringContext,
    anchor_matrix_from_panel,
    prepare_score,
    rank_ic_by_date,
    return_proxy_mode,
    score_expression,
    summarize_prepared_score,
)
from .search_base import SearchCandidate, SearchResult


def _configured_constants(search_config: dict) -> list[float]:
    constants = search_config.get("constants", {})
    if not constants.get("enabled", False):
        return []
    return [float(value) for value in constants.get("values", [])]


def sample_expression(
    rng: random.Random,
    feature_space: FeatureSpace,
    max_depth: int,
    search_config: dict | None = None,
    expected_type: str = "expr",
) -> ExpressionNode:
    search_config = search_config or {}
    constants = _configured_constants(search_config)

    if expected_type == "window":
        if not feature_space.windows:
            raise ValueError("No symbolic windows configured")
        return ExpressionNode(kind="window", value=rng.choice(feature_space.windows))

    if expected_type == "scalar":
        if not constants:
            raise ValueError("No symbolic constants configured")
        return ExpressionNode(kind="constant", value=rng.choice(constants))

    if max_depth <= 0 or rng.random() < 0.35:
        terminals = [
            ExpressionNode(kind="feature", value=name)
            for name in sorted(feature_space.features)
        ]
        terminals.extend(
            ExpressionNode(kind="constant", value=value) for value in constants
        )
        return rng.choice(terminals)

    operators = list(feature_space.operators.values())
    op = rng.choice(operators)
    children = tuple(
        sample_expression(
            rng, feature_space, max_depth - 1, search_config, expected_type=arg_type
        )
        for arg_type in op.arg_types
    )
    return ExpressionNode(kind="operator", value=op.name, children=children)


def dedupe_or_skip(expr: ExpressionNode, seen_hashes: Set[str]) -> bool:
    digest = expression_hash(expr)
    if digest in seen_hashes:
        return True
    seen_hashes.add(digest)
    return False


def _label_data(panel: pd.DataFrame) -> pd.Series | pd.DataFrame:
    columns = [column for column in panel.columns if column.startswith("label__")]
    if columns:
        return panel[columns]
    return panel["label"]


def _duration(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 6)


def _progress_context(config: dict) -> dict:
    context = config.get("_progress_context")
    return dict(context) if isinstance(context, dict) else {}


def _emit_progress(config: dict, event: str, **payload) -> None:
    progress = config.get("_progress_writer")
    if progress is None:
        return
    progress.emit(event, **_progress_context(config), **payload)


def _expression_eval_cache(config: dict) -> dict | None:
    if bool(config.get("_disable_expression_cache", False)):
        return None
    return config.setdefault("_expression_eval_cache", {})


def _evaluate_expression_for_search(
    expr: ExpressionNode, panel: pd.DataFrame, config: dict
) -> pd.Series:
    return evaluate_expression_cached(expr, panel, _expression_eval_cache(config))


def _fast_scoring_enabled(config: dict) -> bool:
    return not bool(config.get("_disable_fast_scoring", False))


def _scoring_context(
    config: dict, name: str, panel: pd.DataFrame
) -> ScoringContext | None:
    if not _fast_scoring_enabled(config):
        return None
    contexts = config.setdefault("_scoring_contexts", {})
    key = f"{name}:{id(panel)}"
    context = contexts.get(key)
    if context is None:
        context = ScoringContext.from_panel(panel)
        contexts[key] = context
    return context


def _mean_metric(metrics: dict, name: str) -> float | None:
    """Mean of ``name`` across horizons (flat key or ``<horizon>_<name>`` keys)."""
    if name in metrics:
        return float(metrics[name])
    suffix = f"_{name}"
    values = [
        float(value)
        for key, value in metrics.items()
        if str(key).endswith(suffix) and value is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _signed_rank_ic_mean(metrics: dict) -> float | None:
    return _mean_metric(metrics, "rank_ic_mean")


def _constant_perturbation_sharpness(
    expr: ExpressionNode,
    valid_panel: pd.DataFrame,
    robustness: dict,
    config: dict | None = None,
) -> float | None:
    """Mean |valid rank-IC shift| under small Gaussian jitter of the constants.

    A flat (robust) minimum barely moves when its parameters are nudged; a sharp
    (overfit) one swings. SAM-style flat-minimum preference lifts GP
    generalization on noisy/limited data (arXiv 2405.06869). Returns ``None``
    when the expression is constant-free or the panel lacks labels.
    """
    constants = list(iter_constants(expr))
    if (
        not constants
        or "label" not in valid_panel.columns
        or "date" not in valid_panel.columns
    ):
        return None
    sigma = float(robustness.get("perturb_sigma", 0.1))
    n_perturb = max(1, int(robustness.get("n_perturb", 3)))
    rng = np.random.default_rng(int(robustness.get("perturb_seed", 12345)))
    label = valid_panel["label"]
    dates = valid_panel["date"]
    eval_config = config if config is not None else {"_disable_expression_cache": True}
    try:
        base_pred = pd.Series(
            _evaluate_expression_for_search(expr, valid_panel, eval_config),
            index=valid_panel.index,
            dtype=float,
        )
    except Exception:
        return None
    base_ic = rank_ic_by_date(base_pred, label, dates)
    base_mean = float(base_ic.mean()) if len(base_ic) else 0.0
    deviations: list[float] = []
    for _ in range(n_perturb):
        jittered = [
            value + float(rng.normal(0.0, sigma * (abs(value) + 1e-3)))
            for value in constants
        ]
        try:
            pred = pd.Series(
                _evaluate_expression_for_search(
                    replace_constants(expr, jittered), valid_panel, eval_config
                ),
                index=valid_panel.index,
                dtype=float,
            )
        except Exception:
            continue
        ic = rank_ic_by_date(pred, label, dates)
        deviations.append(abs(base_mean - (float(ic.mean()) if len(ic) else 0.0)))
    if not deviations:
        return None
    return float(np.mean(deviations))


def _robust_score(
    base_valid_score: float,
    train_score: float | None,
    expr: ExpressionNode,
    valid_panel: pd.DataFrame | None,
    config: dict,
) -> float:
    """Penalize the selection score for overfitting (train->valid gap + sharpness).

    Default-off passthrough. When ``objective.robustness.enabled`` is set, the
    score becomes ``valid_score - gap_weight * max(0, train - valid)
    - sharpness_weight * sharpness`` so the search prefers candidates that
    generalize and sit in flat minima rather than ones that merely top the
    validation window.
    """
    robustness = config.get("objective", {}).get("robustness", {})
    if not isinstance(robustness, dict) or not robustness.get("enabled", False):
        return base_valid_score
    penalty = 0.0
    gap_weight = float(robustness.get("gap_weight", 0.0))
    if gap_weight > 0.0 and train_score is not None:
        penalty += gap_weight * max(0.0, float(train_score) - base_valid_score)
    sharpness_weight = float(robustness.get("sharpness_weight", 0.0))
    if (
        sharpness_weight > 0.0
        and valid_panel is not None
        and not bool(config.get("_defer_robustness_sharpness", False))
    ):
        sharpness = _constant_perturbation_sharpness(
            expr, valid_panel, robustness, config
        )
        if sharpness is not None:
            penalty += sharpness_weight * sharpness
    return float(base_valid_score - penalty)


@contextmanager
def _temporary_deferred_sharpness(config: dict, enabled: bool) -> Iterator[None]:
    sentinel = object()
    previous = config.get("_defer_robustness_sharpness", sentinel)
    if enabled:
        config["_defer_robustness_sharpness"] = True
    try:
        yield
    finally:
        if previous is sentinel:
            config.pop("_defer_robustness_sharpness", None)
        else:
            config["_defer_robustness_sharpness"] = previous


def _robustness_sharpness_active(config: dict) -> bool:
    robustness = config.get("objective", {}).get("robustness", {})
    if not isinstance(robustness, dict) or not robustness.get("enabled", False):
        return False
    return float(robustness.get("sharpness_weight", 0.0)) > 0.0


def _candidate_can_enter_hall(
    candidate: SearchCandidate,
    hall: List[SearchCandidate],
    top_k: int,
) -> bool:
    limit = max(1, int(top_k))
    if len(hall) < limit:
        return True
    top_scores = sorted((float(item.score) for item in hall), reverse=True)[:limit]
    worst = top_scores[-1]
    return float(candidate.score) > worst


def _raw_score_candidate(
    expr: ExpressionNode,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    config: dict,
    *,
    include_negated: bool = False,
) -> SearchCandidate:
    candidate, _, _ = _raw_score_candidate_with_prepared(
        expr,
        train_panel,
        valid_panel,
        config,
        include_negated=include_negated,
    )
    return candidate


def _raw_score_candidate_with_prepared(
    expr: ExpressionNode,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    config: dict,
    *,
    include_negated: bool = False,
) -> tuple[SearchCandidate, object | None, object | None]:
    expression = expr.to_string()
    train_context = _scoring_context(config, "train", train_panel)
    valid_context = _scoring_context(config, "valid", valid_panel)
    started_at = time.perf_counter()
    train_pred = _evaluate_expression_for_search(expr, train_panel, config)
    _emit_progress(
        config,
        "candidate_step_done",
        step="evaluate_train",
        duration_sec=_duration(started_at),
        expression=expression,
        rows=int(len(train_panel)),
    )
    started_at = time.perf_counter()
    valid_pred = _evaluate_expression_for_search(expr, valid_panel, config)
    _emit_progress(
        config,
        "candidate_step_done",
        step="evaluate_valid",
        duration_sec=_duration(started_at),
        expression=expression,
        rows=int(len(valid_panel)),
    )
    started_at = time.perf_counter()
    train_prepared = None
    if train_context is not None:
        train_prepared = prepare_score(
            train_pred, train_context, config, include_negated=include_negated
        )
        train_metrics = summarize_prepared_score(train_prepared, expr, config)
    else:
        train_metrics = score_expression(
            train_pred,
            _label_data(train_panel),
            train_panel["date"],
            expr,
            config,
            anchors=anchor_matrix_from_panel(train_panel),
        )
    _emit_progress(
        config,
        "candidate_step_done",
        step="score_train",
        duration_sec=_duration(started_at),
        expression=expression,
        score=train_metrics.get("score"),
    )
    started_at = time.perf_counter()
    valid_prepared = None
    if valid_context is not None:
        valid_prepared = prepare_score(
            valid_pred, valid_context, config, include_negated=include_negated
        )
        valid_metrics = summarize_prepared_score(valid_prepared, expr, config)
    else:
        valid_metrics = score_expression(
            valid_pred,
            _label_data(valid_panel),
            valid_panel["date"],
            expr,
            config,
            anchors=anchor_matrix_from_panel(valid_panel),
        )
    _emit_progress(
        config,
        "candidate_step_done",
        step="score_valid",
        duration_sec=_duration(started_at),
        expression=expression,
        score=valid_metrics.get("score"),
    )
    score = _robust_score(
        float(valid_metrics["score"]),
        train_metrics.get("score"),
        expr,
        valid_panel,
        config,
    )
    candidate = SearchCandidate(
        expression=expr,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        score=score,
        complexity=expression_complexity(expr),
    )
    return candidate, train_prepared, valid_prepared


def _candidate_from_prepared(
    expr: ExpressionNode,
    train_prepared: object,
    valid_prepared: object,
    config: dict,
    *,
    sign: int,
) -> SearchCandidate:
    train_metrics = summarize_prepared_score(train_prepared, expr, config, sign=sign)
    valid_metrics = summarize_prepared_score(valid_prepared, expr, config, sign=sign)
    # No panel here (negated variant reuses prepared scores), so sharpness is
    # skipped; the train->valid gap penalty still applies.
    score = _robust_score(
        float(valid_metrics["score"]), train_metrics.get("score"), expr, None, config
    )
    return SearchCandidate(
        expression=expr,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        score=score,
        complexity=expression_complexity(expr),
    )


def score_candidate(
    expr: ExpressionNode,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    config: dict,
    *,
    defer_robustness_sharpness: bool = False,
) -> SearchCandidate | None:
    with _temporary_deferred_sharpness(config, defer_robustness_sharpness):
        return _score_candidate_impl(expr, train_panel, valid_panel, config)


def _score_candidate_impl(
    expr: ExpressionNode,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    config: dict,
) -> SearchCandidate | None:
    try:
        total_started_at = time.perf_counter()
        objective_config = config.get("objective", {})
        auto_negate_enabled = bool(
            objective_config.get("auto_negate_negative_ic", False)
        )
        candidate, train_prepared, valid_prepared = _raw_score_candidate_with_prepared(
            expr,
            train_panel,
            valid_panel,
            config,
            include_negated=auto_negate_enabled,
        )
        if not objective_config.get("auto_negate_negative_ic", False):
            _emit_progress(
                config,
                "candidate_scored",
                duration_sec=_duration(total_started_at),
                score=candidate.score,
                train_score=candidate.train_metrics.get("score"),
                valid_score=candidate.valid_metrics.get("score"),
                auto_negated=False,
            )
            return candidate

        signed_mean = _signed_rank_ic_mean(candidate.train_metrics)
        threshold = float(objective_config.get("auto_negate_threshold", 0.0))
        return_proxy_config = objective_config.get("return_proxy", {})
        proxy_enabled = bool(return_proxy_config.get("enabled", False))
        use_excess_orientation = (
            proxy_enabled and return_proxy_mode(return_proxy_config) == "top_excess"
        )

        neg_expr = ExpressionNode(kind="operator", value="neg", children=(expr,))
        neg_candidate: SearchCandidate | None = None
        orient_orig: float | None = None
        orient_neg: float | None = None
        if use_excess_orientation:
            # Orientation by the gate metric: a strong loser-detector (high
            # |IC|, negative top-bucket excess) is only worth negating when the
            # flipped factor's TOP bucket (= original bottom bucket) actually
            # outperforms. With prepared scores this derivation is summary-only
            # work, so both orientations are always compared.
            if train_prepared is not None and valid_prepared is not None:
                neg_candidate = _candidate_from_prepared(
                    neg_expr, train_prepared, valid_prepared, config, sign=-1
                )
            else:
                neg_candidate = _raw_score_candidate(
                    neg_expr, train_panel, valid_panel, config
                )
            orient_orig = _mean_metric(candidate.train_metrics, "return_proxy_mean")
            orient_neg = _mean_metric(neg_candidate.train_metrics, "return_proxy_mean")
            should_negate = (
                orient_orig is not None
                and orient_neg is not None
                and orient_neg > orient_orig + threshold
            )
        else:
            should_negate = signed_mean is not None and signed_mean < threshold

        if not should_negate:
            candidate.train_metrics["auto_negated"] = 0.0
            candidate.valid_metrics["auto_negated"] = 0.0
            _emit_progress(
                config,
                "candidate_scored",
                duration_sec=_duration(total_started_at),
                score=candidate.score,
                train_score=candidate.train_metrics.get("score"),
                valid_score=candidate.valid_metrics.get("score"),
                auto_negated=False,
                signed_rank_ic_mean=signed_mean,
                top_excess_orig=orient_orig,
                top_excess_neg=orient_neg,
            )
            return candidate

        _emit_progress(
            config,
            "candidate_auto_negate_start",
            signed_rank_ic_mean=signed_mean,
            threshold=threshold,
            orientation="top_excess" if use_excess_orientation else "rank_ic",
            top_excess_orig=orient_orig,
            top_excess_neg=orient_neg,
        )
        neg_started_at = time.perf_counter()
        if neg_candidate is None:
            if train_prepared is not None and valid_prepared is not None:
                neg_candidate = _candidate_from_prepared(
                    neg_expr, train_prepared, valid_prepared, config, sign=-1
                )
            else:
                neg_candidate = _raw_score_candidate(
                    neg_expr, train_panel, valid_panel, config
                )
        _emit_progress(
            config,
            "candidate_step_done",
            step="derive_negated_metrics",
            duration_sec=_duration(neg_started_at),
            expression=neg_expr.to_string(),
            score=neg_candidate.score,
        )
        neg_candidate.train_metrics["auto_negated"] = 1.0
        neg_candidate.valid_metrics["auto_negated"] = 1.0
        if signed_mean is not None:
            neg_candidate.train_metrics["auto_negate_source_rank_ic_mean"] = float(
                signed_mean
            )
            neg_candidate.valid_metrics["auto_negate_source_rank_ic_mean"] = float(
                signed_mean
            )
        if orient_orig is not None and orient_neg is not None:
            neg_candidate.train_metrics["auto_negate_source_top_excess"] = float(
                orient_orig
            )
            neg_candidate.train_metrics["auto_negate_negated_top_excess"] = float(
                orient_neg
            )
        _emit_progress(
            config,
            "candidate_scored",
            duration_sec=_duration(total_started_at),
            score=neg_candidate.score,
            train_score=neg_candidate.train_metrics.get("score"),
            valid_score=neg_candidate.valid_metrics.get("score"),
            auto_negated=True,
            signed_rank_ic_mean=signed_mean,
            top_excess_orig=orient_orig,
            top_excess_neg=orient_neg,
        )
        return neg_candidate
    except Exception as exc:
        _emit_progress(
            config,
            "candidate_score_failed",
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return None


def score_candidate_for_hall(
    expr: ExpressionNode,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    config: dict,
    *,
    hall: List[SearchCandidate],
    top_k: int,
) -> tuple[SearchCandidate | None, bool]:
    """Score cheaply first; fully apply sharpness only for hall contenders."""
    if not _robustness_sharpness_active(config):
        return score_candidate(expr, train_panel, valid_panel, config), False

    candidate = score_candidate(
        expr,
        train_panel,
        valid_panel,
        config,
        defer_robustness_sharpness=True,
    )
    if candidate is None or not _candidate_can_enter_hall(candidate, hall, top_k):
        return candidate, False

    rescored = score_candidate(
        expr,
        train_panel,
        valid_panel,
        config,
        defer_robustness_sharpness=False,
    )
    return (rescored if rescored is not None else candidate), rescored is not None


def maintain_hall_of_fame(
    candidates: List[SearchCandidate], top_k: int
) -> List[SearchCandidate]:
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


class RandomSearchBackend:
    def search(
        self,
        train_panel: pd.DataFrame,
        valid_panel: pd.DataFrame,
        feature_space: FeatureSpace,
        config: dict,
    ) -> SearchResult:
        search_config = config.get("search", {})
        rng = random.Random(int(search_config.get("seed", 42)))
        n_candidates = int(search_config.get("n_candidates", 5000))
        max_depth = int(search_config.get("max_depth", 4))
        hall_top_k = int(search_config.get("hall_top_k", 50))

        seen_hashes: Set[str] = set()
        hall: List[SearchCandidate] = []
        attempted = 0
        scored = 0

        for _ in range(n_candidates):
            attempted += 1
            expr = sample_expression(rng, feature_space, max_depth, search_config)
            if dedupe_or_skip(expr, seen_hashes):
                continue
            try:
                validate_exportable_expression(expr, feature_space)
            except ValueError:
                continue
            candidate, _ = score_candidate_for_hall(
                expr,
                train_panel,
                valid_panel,
                config,
                hall=hall,
                top_k=hall_top_k,
            )
            if candidate is None:
                continue
            scored += 1
            hall.append(candidate)
            hall = maintain_hall_of_fame(hall, hall_top_k)

        if not hall:
            raise RuntimeError("random_v0 produced no valid candidates")

        return SearchResult(
            best=hall[0],
            hall_of_fame=hall,
            metadata={
                "backend": "random_v0",
                "attempted": attempted,
                "scored": scored,
                "unique_expressions": len(seen_hashes),
                "max_depth": max_depth,
                "n_candidates": n_candidates,
            },
        )
