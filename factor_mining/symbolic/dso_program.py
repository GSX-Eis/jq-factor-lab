"""Parse DSO token sequences into symbolic ExpressionNode trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .dso_tokens import DSOTokenVocab
from .expression import ExpressionNode, referenced_features
from .feature_space import FeatureSpace, validate_exportable_expression


@dataclass(frozen=True)
class ParsedProgram:
    expression: ExpressionNode | None
    ok: bool
    error: str = ""


def parse_dso_program(
    token_ids: Sequence[int],
    vocab: DSOTokenVocab,
    feature_space: FeatureSpace,
    max_length: int,
) -> ParsedProgram:
    if len(token_ids) > max_length:
        return ParsedProgram(None, False, "sequence_exceeds_max_length")

    expression_tokens = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab.size:
            return ParsedProgram(None, False, "unknown_token")
        token = vocab.tokens[token_id]
        if token.kind == "end":
            break
        if token.kind == "start":
            continue
        expression_tokens.append(token)

    if not expression_tokens:
        return ParsedProgram(None, False, "empty_expression")

    def parse_at(index: int) -> tuple[ExpressionNode, int]:
        token = expression_tokens[index]
        if token.kind == "feature":
            return ExpressionNode(kind="feature", value=str(token.value)), index + 1
        if token.kind == "constant":
            return ExpressionNode(kind="constant", value=float(token.value)), index + 1
        if token.kind == "window":
            return ExpressionNode(kind="window", value=int(token.value)), index + 1
        if token.kind != "operator":
            raise ValueError("unexpected_token")
        children = []
        cursor = index + 1
        for _ in range(token.arity):
            if cursor >= len(expression_tokens):
                raise ValueError("unterminated_expression")
            child, cursor = parse_at(cursor)
            children.append(child)
        return ExpressionNode(kind="operator", value=str(token.value), children=tuple(children)), cursor

    try:
        expression, consumed = parse_at(0)
    except (IndexError, ValueError) as exc:
        return ParsedProgram(None, False, str(exc) or "parse_failed")

    if consumed != len(expression_tokens):
        return ParsedProgram(None, False, "trailing_tokens")
    if not referenced_features(expression):
        return ParsedProgram(None, False, "no_feature_reference")

    try:
        validate_exportable_expression(expression, feature_space)
    except ValueError as exc:
        return ParsedProgram(None, False, str(exc))

    return ParsedProgram(expression, True)
