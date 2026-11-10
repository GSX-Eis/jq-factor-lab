"""Small registries for holdings selection and portfolio exposure rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd


def rank_top_k_equal_weight(
    ranked: pd.Series, effective_top_k: int
) -> dict[Any, float]:
    """Select the highest-ranked names and assign equal stock weights."""
    if effective_top_k <= 0:
        raise ValueError("effective_top_k must be positive")
    names = list(ranked.index[:effective_top_k])
    if not names:
        return {}
    weight = 1.0 / len(names)
    return {name: weight for name in names}


def _position_inputs(
    score: float, config: Mapping[str, Any]
) -> tuple[float, float, float]:
    minimum = float(config.get("min_exposure", 0.5))
    maximum = float(config.get("max_exposure", 1.0))
    values = np.asarray([score, minimum, maximum], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("position plugin inputs must be finite")
    if not 0 <= minimum <= maximum <= 1:
        raise ValueError("invalid position plugin exposure bounds")
    return float(score), minimum, maximum


def one_sided_linear(score: float, config: Mapping[str, Any]) -> float:
    """Reduce exposure linearly for negative scores."""
    score, minimum, maximum = _position_inputs(score, config)
    slope = float(config.get("slope", 0.5))
    if not np.isfinite(slope) or slope < 0:
        raise ValueError("invalid one_sided_linear slope")
    return float(np.clip(maximum + slope * min(score, 0.0), minimum, maximum))


def sign_gate(score: float, config: Mapping[str, Any]) -> float:
    """Use minimum exposure whenever the score is negative."""
    score, minimum, maximum = _position_inputs(score, config)
    return minimum if score < 0 else maximum


def tail_gate(score: float, config: Mapping[str, Any]) -> float:
    """Reduce only below the frozen minus-one-score tail."""
    score, minimum, maximum = _position_inputs(score, config)
    return minimum if score < -1.0 else maximum


def downside_quadratic(score: float, config: Mapping[str, Any]) -> float:
    """Increase de-risking quadratically as the negative score deepens."""
    score, minimum, maximum = _position_inputs(score, config)
    penalty = min(max(-score, 0.0) ** 2, 1.0)
    return maximum - (maximum - minimum) * penalty


def downside_exponential(score: float, config: Mapping[str, Any]) -> float:
    """Apply a smooth convex exposure curve below zero."""
    score, minimum, maximum = _position_inputs(score, config)
    if score >= 0:
        return maximum
    return minimum + (maximum - minimum) * float(np.exp(score))


STRATEGY_PLUGINS: dict[str, Callable[[pd.Series, int], dict[Any, float]]] = {
    "rank_top_k_equal_weight_v1": rank_top_k_equal_weight,
}
POSITION_PLUGINS: dict[str, Callable[[float, Mapping[str, Any]], float]] = {
    "one_sided_linear_v1": one_sided_linear,
    "sign_gate_v1": sign_gate,
    "tail_gate_minus_one_v1": tail_gate,
    "downside_quadratic_v1": downside_quadratic,
    "downside_exponential_v1": downside_exponential,
}


def get_strategy_plugin(plugin_id: str) -> Callable[[pd.Series, int], dict[Any, float]]:
    try:
        return STRATEGY_PLUGINS[plugin_id]
    except KeyError as exc:
        raise ValueError(f"unknown strategy plugin: {plugin_id}") from exc


def get_position_plugin(
    plugin_id: str,
) -> Callable[[float, Mapping[str, Any]], float]:
    try:
        return POSITION_PLUGINS[plugin_id]
    except KeyError as exc:
        raise ValueError(f"unknown position plugin: {plugin_id}") from exc
