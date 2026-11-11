"""QuantaAlpha structural gates over the existing symbolic AST."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .expression import (
    ExpressionNode,
    canonical_expression_hash,
    referenced_features,
)
from .feature_space import FeatureSpace, validate_exportable_expression


_COMMUTATIVE = {
    "add",
    "mul",
    "eq",
    "ne",
    "logical_and",
    "logical_or",
    "element_max",
    "element_min",
}
_COMPARISONS = {"gt", "ge", "lt", "le", "eq", "ne"}
_LOGICAL = {"logical_and", "logical_or"}


def node_count(expr: ExpressionNode) -> int:
    return 1 + sum(node_count(child) for child in expr.children)


def free_argument_count(expr: ExpressionNode) -> int:
    return int(expr.kind in {"constant", "window"}) + sum(
        free_argument_count(child) for child in expr.children
    )


def _subtree_signature(expr: ExpressionNode) -> tuple:
    if expr.kind == "constant":
        return ("constant", float(expr.value))
    if expr.kind == "window":
        return ("window", int(float(expr.value)))
    if expr.kind == "feature":
        return ("feature", str(expr.value))
    children = tuple(_subtree_signature(child) for child in expr.children)
    if str(expr.value) in _COMMUTATIVE:
        children = tuple(sorted(children, key=repr))
    return ("operator", str(expr.value), children)


def _subtrees(expr: ExpressionNode) -> dict[tuple, int]:
    result = {_subtree_signature(expr): node_count(expr)}
    for child in expr.children:
        result.update(_subtrees(child))
    return result


def largest_common_subtree_size(left: ExpressionNode, right: ExpressionNode) -> int:
    left_subtrees = _subtrees(left)
    right_subtrees = _subtrees(right)
    common = set(left_subtrees).intersection(right_subtrees)
    return max((left_subtrees[key] for key in common), default=0)


def _infer_type(node: ExpressionNode) -> str:
    if node.kind == "feature":
        return "number"
    if node.kind == "constant":
        return "scalar"
    if node.kind == "window":
        return "window"
    if node.kind != "operator":
        raise ValueError(f"Unsupported AST node kind: {node.kind}")

    name = str(node.value)
    child_types = tuple(_infer_type(child) for child in node.children)
    if name in _COMPARISONS:
        if any(item == "bool" for item in child_types):
            raise ValueError(f"Operator {name} expects numeric arguments")
        return "bool"
    if name in _LOGICAL:
        if any(item != "bool" for item in child_types):
            raise ValueError(f"Operator {name} expects boolean arguments")
        return "bool"
    if name == "where":
        if child_types[0] != "bool" or any(item == "bool" for item in child_types[1:]):
            raise ValueError("Operator where expects bool, numeric, numeric")
    elif name == "filter":
        if child_types[1] != "bool":
            raise ValueError("Operator filter expects a boolean condition")
    elif name == "count":
        if child_types[0] != "bool":
            raise ValueError("Operator count expects a boolean condition")
    elif name in {"sum_if", "sumif"}:
        if child_types[2] != "bool":
            raise ValueError("Operator sum_if expects a boolean condition")
    elif any(item == "bool" for item in child_types):
        raise ValueError(f"Operator {name} does not accept boolean arguments")
    return "number"


def structural_gate(
    expr: ExpressionNode,
    feature_space: FeatureSpace,
    *,
    factor_zoo: Iterable[ExpressionNode] = (),
    canonical_hashes: set[str] | None = None,
    max_symbol_length: int = 250,
    max_base_features: int = 6,
    max_free_argument_ratio: float = 0.5,
    max_common_subtree: int = 8,
) -> dict[str, Any]:
    """Return a complete hard-gate report without raising on bad proposals."""

    reasons: list[str] = []
    expression_text = expr.to_string()
    total_nodes = node_count(expr)
    free_args = free_argument_count(expr)
    free_ratio = float(free_args / total_nodes) if total_nodes else 1.0
    features = sorted(referenced_features(expr))
    canonical_hash = canonical_expression_hash(expr)

    try:
        validate_exportable_expression(expr, feature_space)
        if _infer_type(expr) != "number":
            raise ValueError("Factor expression must produce numeric output")
    except ValueError as exc:
        reasons.append(f"invalid_ast:{exc}")
    if len(expression_text) > int(max_symbol_length):
        reasons.append("symbol_length_exceeded")
    if len(features) > int(max_base_features):
        reasons.append("base_feature_limit_exceeded")
    if free_ratio >= float(max_free_argument_ratio):
        reasons.append("free_argument_ratio_exceeded")
    if canonical_hash in set(canonical_hashes or ()):
        reasons.append("canonical_duplicate")

    largest = 0
    matched_hash = ""
    for prior in factor_zoo:
        size = largest_common_subtree_size(expr, prior)
        if size > largest:
            largest = size
            matched_hash = canonical_expression_hash(prior)
    if largest > int(max_common_subtree):
        reasons.append("largest_common_subtree_exceeded")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "metrics": {
            "symbol_length": len(expression_text),
            "base_feature_count": len(features),
            "base_features": features,
            "free_argument_count": free_args,
            "total_node_count": total_nodes,
            "free_argument_ratio": free_ratio,
            "canonical_hash": canonical_hash,
            "largest_common_subtree_size": largest,
            "matched_expression_hash": matched_hash,
        },
    }
