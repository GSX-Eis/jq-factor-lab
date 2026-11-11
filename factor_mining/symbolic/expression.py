"""Expression AST and code generation for symbolic mining."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, MutableMapping, Sequence, Set

import numpy as np
import pandas as pd

from .operator_runtime import runtime_namespace


@dataclass(frozen=True)
class ExpressionNode:
    kind: str
    value: str | float
    children: Sequence["ExpressionNode"] = field(default_factory=tuple)

    def to_string(self) -> str:
        if self.kind == "feature":
            return str(self.value)
        if self.kind == "constant":
            return repr(float(self.value))
        if self.kind == "window":
            return str(int(self.value))
        args = ", ".join(child.to_string() for child in self.children)
        return f"{self.value}({args})"


def expression_hash(expr: ExpressionNode) -> str:
    return hashlib.sha1(expr.to_string().encode("utf-8")).hexdigest()


def canonical_expression_string(expr: ExpressionNode) -> str:
    """Return a conservative canonical string for dedupe.

    Only pure arithmetic subtrees are simplified by sympy. Financial/time-series
    operators keep their tree shape so safe_div, cross-sectional, and domain
    semantics are not rewritten accidentally.
    """

    try:
        import sympy as sp
    except Exception:
        return expr.to_string()

    pure_ops = {"add", "sub", "mul", "neg"}

    def to_sympy(node: ExpressionNode):
        if node.kind == "feature":
            return sp.Symbol(f"F__{str(node.value)}")
        if node.kind == "constant":
            return sp.Float(float(node.value))
        if node.kind == "window":
            return None
        if node.value not in pure_ops:
            return None
        children = [to_sympy(child) for child in node.children]
        if any(child is None for child in children):
            return None
        if node.value == "add":
            return children[0] + children[1]
        if node.value == "sub":
            return children[0] - children[1]
        if node.value == "mul":
            return children[0] * children[1]
        if node.value == "neg":
            return -children[0]
        return None

    def canonical(node: ExpressionNode) -> str:
        symbolic = to_sympy(node)
        if symbolic is not None:
            return sp.sstr(sp.simplify(symbolic))
        if node.kind == "feature":
            return f"feature:{node.value}"
        if node.kind == "constant":
            return f"const:{float(node.value):.12g}"
        if node.kind == "window":
            return f"window:{int(node.value)}"
        args = ",".join(canonical(child) for child in node.children)
        return f"{node.value}({args})"

    return canonical(expr)


def canonical_expression_hash(expr: ExpressionNode) -> str:
    return hashlib.sha1(canonical_expression_string(expr).encode("utf-8")).hexdigest()


def expression_complexity(expr: ExpressionNode) -> int:
    if expr.kind in {"feature", "constant"}:
        return 1
    if expr.kind == "window":
        return 0
    return 1 + sum(expression_complexity(child) for child in expr.children)


def expression_depth(expr: ExpressionNode) -> int:
    if not expr.children:
        return 1
    return 1 + max(expression_depth(child) for child in expr.children)


def referenced_features(expr: ExpressionNode) -> Set[str]:
    if expr.kind == "feature":
        return {str(expr.value)}
    features: Set[str] = set()
    for child in expr.children:
        features.update(referenced_features(child))
    return features


def iter_constants(expr: ExpressionNode) -> Iterator[float]:
    if expr.kind == "constant":
        yield float(expr.value)
    for child in expr.children:
        yield from iter_constants(child)


def replace_constants(expr: ExpressionNode, values: Sequence[float]) -> ExpressionNode:
    iterator = iter(values)

    def replace(node: ExpressionNode) -> ExpressionNode:
        if node.kind == "constant":
            return ExpressionNode(kind="constant", value=float(next(iterator)))
        if not node.children:
            return node
        return ExpressionNode(
            kind=node.kind,
            value=node.value,
            children=tuple(replace(child) for child in node.children),
        )

    return replace(expr)


def to_eval_code(expr: ExpressionNode) -> str:
    if expr.kind == "feature":
        return f"features[{str(expr.value)!r}]"
    if expr.kind == "constant":
        return repr(float(expr.value))
    if expr.kind == "window":
        return repr(int(expr.value))
    args = ", ".join(to_eval_code(child) for child in expr.children)
    return f"{expr.value}({args})"


def to_factor_code(expr: ExpressionNode) -> str:
    return to_eval_code(expr)


def _frame_feature_dict(frame: pd.DataFrame) -> dict:
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"date", "code", "label"}
        and not str(column).startswith("label__")
    ]
    return {name: frame[name] for name in feature_columns}


def _frame_codes(frame: pd.DataFrame) -> pd.Series:
    if "code" in frame.columns:
        return frame["code"]
    return pd.Series(np.arange(len(frame)), index=frame.index)


def _finalize_eval_result(result: Any, frame: pd.DataFrame) -> pd.Series:
    if np.isscalar(result):
        result = pd.Series(float(result), index=frame.index)
    else:
        result = pd.Series(result, index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan)


def _check_referenced_features(expr: ExpressionNode, features: dict) -> None:
    refs = referenced_features(expr)
    missing = refs - set(features)
    if missing:
        raise KeyError(f"Expression references missing features: {sorted(missing)}")


def evaluate_expression(expr: ExpressionNode, frame: pd.DataFrame) -> pd.Series:
    features = _frame_feature_dict(frame)
    _check_referenced_features(expr, features)
    result = eval(
        to_eval_code(expr),
        runtime_namespace(features, frame["date"], _frame_codes(frame)),
        {},
    )
    return _finalize_eval_result(result, frame)


def evaluate_expression_cached(
    expr: ExpressionNode,
    frame: pd.DataFrame,
    cache: MutableMapping[tuple[int, str], Any] | None = None,
    *,
    max_entries: int = 10000,
) -> pd.Series:
    """Evaluate an expression with a run-local subtree cache.

    The cache is intentionally caller-owned. Search/rerank passes share it
    within a run, while the public ``evaluate_expression`` remains stateless for
    tests, ad-hoc inspection, and generated factor parity.
    """
    if cache is None:
        return evaluate_expression(expr, frame)

    features = _frame_feature_dict(frame)
    _check_referenced_features(expr, features)
    namespace = runtime_namespace(features, frame["date"], _frame_codes(frame))
    frame_id = id(frame)

    def eval_node(node: ExpressionNode) -> Any:
        key = (frame_id, expression_hash(node))
        if key in cache:
            return cache[key]

        if node.kind == "feature":
            result = features[str(node.value)]
        elif node.kind == "constant":
            result = float(node.value)
        elif node.kind == "window":
            result = int(node.value)
        else:
            fn = namespace[str(node.value)]
            result = fn(*(eval_node(child) for child in node.children))

        if len(cache) < max_entries:
            cache[key] = result
        return result

    return _finalize_eval_result(eval_node(expr), frame)


def serialize_expression(expr: ExpressionNode) -> dict:
    return {
        "kind": expr.kind,
        "value": expr.value,
        "children": [serialize_expression(child) for child in expr.children],
    }


def deserialize_expression(payload: dict) -> ExpressionNode:
    children = payload.get("children", [])
    return ExpressionNode(
        kind=str(payload.get("kind") or ""),
        value=payload.get("value"),
        children=tuple(deserialize_expression(child) for child in children if isinstance(child, dict)),
    )


def collect_feature_list(exprs: Iterable[ExpressionNode]) -> List[str]:
    features: Set[str] = set()
    for expr in exprs:
        features.update(referenced_features(expr))
    return sorted(features)
