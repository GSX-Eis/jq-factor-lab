"""IC-weighted multi-factor ensemble synthesis for the post-rerank stage.

A single mined expression on a weekly (10-day) horizon empirically caps near
rank IC ~0.10; combining several *decorrelated* expressions (e.g. reversal +
liquidity + size + volume) raises the composite toward the
``IC * sqrt(N / (1 + (N-1) * rho_bar))`` diversification bound and is far more
robust out-of-sample. Rather than selecting a single best candidate, this module
synthesizes a composite factor:

    composite = sum_i  w_i * cs_zscore(oriented_i)

where ``oriented_i`` is the member expression flipped by ``sign(IC_i)`` so every
term contributes positively, ``cs_zscore`` puts the members on a common
cross-sectional scale, and ``w_i = |IC_i| / sum_j |IC_j|`` is the IC weight.

Holdout hygiene: ``IC_i`` MUST be measured on the selection window (the search
validation split, e.g. 2024) -- never on the holdout the composite is gated on.
Fitting member admission, orientation, or weights to holdout ICs makes the
"OOS" gate in-sample for the composite and systematically inflates its holdout
IC (observed signature: holdout IC many times the in-sample IC).

The composite is returned as an ``ExpressionNode`` tree so it materializes,
evaluates, and gates exactly like any other mined candidate -- the reranker then
measures its real OOS IC and runs it through the same hard gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .expression import ExpressionNode

# Operators the composite tree relies on; if the run's feature space lacks any of
# them the ensemble cannot be materialized and synthesis is skipped.
REQUIRED_OPERATORS: tuple[str, ...] = ("add", "mul", "neg", "cs_zscore")


@dataclass(frozen=True)
class EnsembleMember:
    """A candidate eligible for ensembling.

    ``ic`` is the member's SELECTION-window (validation, e.g. 2024) rank IC at
    the gate period (sign carries the factor direction); it must never be a
    holdout quantity, or the composite's weights are fitted on the gate window.
    ``family`` is the dominant feature family used for one-per-family
    diversification. ``prediction`` is the member's factor value series on the
    OOS panel, used only for label-free pairwise correlation pruning.
    """

    rank: int
    expression: ExpressionNode
    ic: float
    family: str
    prediction: Optional[pd.Series] = None


def _rank_corr(a: pd.Series, b: pd.Series) -> float | None:
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 10:
        return None
    try:
        value = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman")
    except Exception:
        return None
    return float(value) if value is not None and np.isfinite(value) else None


def _too_correlated(
    candidate: EnsembleMember,
    selected: Sequence[EnsembleMember],
    max_abs_corr: float,
) -> bool:
    if candidate.prediction is None:
        return False
    for other in selected:
        if other.prediction is None:
            continue
        corr = _rank_corr(candidate.prediction, other.prediction)
        if corr is not None and abs(corr) > max_abs_corr:
            return True
    return False


def select_ensemble_members(
    members: Sequence[EnsembleMember],
    *,
    max_factors: int,
    max_abs_corr: float = 0.7,
    min_member_ic: float = 0.01,
    one_per_family: bool = True,
) -> list[EnsembleMember]:
    """Greedily pick decorrelated, high-|IC| members for the composite.

    Members are ranked by ``|IC|`` descending and admitted while:
      * ``|IC| >= min_member_ic`` (skip noise-level signals),
      * the family has not already been used (when ``one_per_family``),
      * the member is not strongly correlated (``|spearman| > max_abs_corr``)
        with any already-selected member that has a prediction series.

    Members may be individually below the promotion gate -- the point of the
    ensemble is to combine sub-threshold but decorrelated factors into a
    supra-threshold composite.
    """
    ranked = sorted(members, key=lambda member: abs(member.ic), reverse=True)
    selected: list[EnsembleMember] = []
    used_families: set[str] = set()
    cap = max(0, int(max_factors))
    for candidate in ranked:
        if len(selected) >= cap:
            break
        if not np.isfinite(candidate.ic) or abs(candidate.ic) < float(min_member_ic):
            continue
        if one_per_family and candidate.family in used_families:
            continue
        if _too_correlated(candidate, selected, float(max_abs_corr)):
            continue
        selected.append(candidate)
        used_families.add(candidate.family)
    return selected


def _op(name: str, *children: ExpressionNode) -> ExpressionNode:
    return ExpressionNode(kind="operator", value=str(name), children=tuple(children))


def _const(value: float) -> ExpressionNode:
    return ExpressionNode(kind="constant", value=float(value))


def build_ensemble_expression(
    members: Sequence[EnsembleMember],
    *,
    available_operators: set[str] | None = None,
    weight_decimals: int = 4,
    min_members: int = 2,
) -> ExpressionNode | None:
    """Build the IC-weighted composite ``ExpressionNode`` from members.

    Returns ``None`` (caller falls back to single-best selection) when fewer
    than ``min_members`` usable members remain or the feature space lacks a
    required operator.
    """
    usable = [m for m in members if np.isfinite(m.ic) and abs(m.ic) > 0.0]
    if len(usable) < max(2, int(min_members)):
        return None
    if available_operators is not None and not set(REQUIRED_OPERATORS).issubset(
        set(available_operators)
    ):
        return None
    total = float(sum(abs(m.ic) for m in usable))
    if total <= 0.0:
        return None

    terms: list[ExpressionNode] = []
    for member in usable:
        oriented = member.expression if member.ic >= 0 else _op("neg", member.expression)
        normalized = _op("cs_zscore", oriented)
        weight = round(abs(member.ic) / total, int(weight_decimals))
        if weight <= 0.0:
            continue
        terms.append(_op("mul", _const(weight), normalized))

    if len(terms) < max(2, int(min_members)):
        return None
    composite = terms[0]
    for term in terms[1:]:
        composite = _op("add", composite, term)
    return composite
