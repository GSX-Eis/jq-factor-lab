"""Structural priors for the PyTorch DSO symbolic search backend."""

from __future__ import annotations

from dataclasses import dataclass

from .expression import (
    ExpressionNode,
    expression_complexity,
    expression_depth,
    referenced_features,
)
from .feature_space import FeatureSpace


@dataclass(frozen=True)
class DSOPriorConfig:
    max_depth: int | None
    max_complexity: int | None
    forbid_repeated_unary: bool = True
    min_feature_refs: int = 1
    requires_binary_op: bool = False
    max_rolling_depth: int | None = None
    max_binary_rolling_ops: int | None = None
    max_window: int | None = None


def build_dso_prior_config(search_config: dict) -> DSOPriorConfig:
    priors = search_config.get("priors", {})
    max_length = int(search_config.get("max_length", 20))
    max_depth = priors.get("max_depth", search_config.get("max_depth", max_length))
    max_complexity = priors.get(
        "max_complexity",
        search_config.get("max_complexity", max_length),
    )
    requires_binary_op = priors.get(
        "requires_binary_op",
        search_config.get("requires_binary_op", False),
    )
    return DSOPriorConfig(
        max_depth=int(max_depth) if max_depth is not None else None,
        max_complexity=int(max_complexity) if max_complexity is not None else None,
        forbid_repeated_unary=bool(priors.get("forbid_repeated_unary", True)),
        min_feature_refs=int(priors.get("min_feature_refs", 1)),
        requires_binary_op=bool(requires_binary_op),
        max_rolling_depth=(
            int(priors["max_rolling_depth"])
            if priors.get("max_rolling_depth") is not None
            else None
        ),
        max_binary_rolling_ops=(
            int(priors["max_binary_rolling_ops"])
            if priors.get("max_binary_rolling_ops") is not None
            else None
        ),
        max_window=(
            int(priors["max_window"])
            if priors.get("max_window") is not None
            else None
        ),
    )


def _is_unary_operator(expr: ExpressionNode, feature_space: FeatureSpace) -> bool:
    if expr.kind != "operator":
        return False
    operator = feature_space.operators.get(str(expr.value))
    return bool(operator and operator.arity == 1)


def _has_repeated_unary(expr: ExpressionNode, feature_space: FeatureSpace) -> bool:
    if _is_unary_operator(expr, feature_space) and expr.children:
        child = expr.children[0]
        if _is_unary_operator(child, feature_space):
            return True
    return any(_has_repeated_unary(child, feature_space) for child in expr.children)


def _has_binary_expression_operator(
    expr: ExpressionNode,
    feature_space: FeatureSpace,
) -> bool:
    if expr.kind == "operator":
        operator = feature_space.operators.get(str(expr.value))
        if operator and operator.arg_types.count("expr") >= 2:
            return True
    return any(
        _has_binary_expression_operator(child, feature_space)
        for child in expr.children
    )


def _rolling_stats(
    expr: ExpressionNode,
    feature_space: FeatureSpace,
) -> tuple[int, int, int]:
    rolling_depth = 0
    binary_rolling_ops = 0
    max_window = 0

    if expr.kind == "operator":
        operator = feature_space.operators.get(str(expr.value))
        child_stats = [_rolling_stats(child, feature_space) for child in expr.children]
        child_rolling_depth = max((item[0] for item in child_stats), default=0)
        binary_rolling_ops = sum(item[1] for item in child_stats)
        max_window = max((item[2] for item in child_stats), default=0)
        if operator and operator.category == "time_series":
            rolling_depth = child_rolling_depth + 1
            if operator.arg_types.count("expr") >= 2:
                binary_rolling_ops += 1
        else:
            rolling_depth = child_rolling_depth

        if operator:
            for expected_type, child in zip(operator.arg_types, expr.children):
                if expected_type == "window" and child.kind in {"window", "constant"}:
                    max_window = max(max_window, int(float(child.value)))
    elif expr.kind == "window":
        max_window = int(float(expr.value))

    return rolling_depth, binary_rolling_ops, max_window


def expression_prior_error(
    expr: ExpressionNode,
    feature_space: FeatureSpace,
    prior_config: DSOPriorConfig,
) -> str | None:
    if (
        prior_config.min_feature_refs
        and len(referenced_features(expr)) < prior_config.min_feature_refs
    ):
        return "min_feature_refs"
    if prior_config.requires_binary_op and not _has_binary_expression_operator(
        expr,
        feature_space,
    ):
        return "requires_binary_op"
    if (
        prior_config.max_depth is not None
        and expression_depth(expr) > prior_config.max_depth
    ):
        return "max_depth"
    if (
        prior_config.max_complexity is not None
        and expression_complexity(expr) > prior_config.max_complexity
    ):
        return "max_complexity"
    if prior_config.forbid_repeated_unary and _has_repeated_unary(expr, feature_space):
        return "repeated_unary"
    rolling_depth, binary_rolling_ops, max_window = _rolling_stats(expr, feature_space)
    if (
        prior_config.max_rolling_depth is not None
        and rolling_depth > prior_config.max_rolling_depth
    ):
        return "max_rolling_depth"
    if (
        prior_config.max_binary_rolling_ops is not None
        and binary_rolling_ops > prior_config.max_binary_rolling_ops
    ):
        return "max_binary_rolling_ops"
    if prior_config.max_window is not None and max_window > prior_config.max_window:
        return "max_window"
    return None
