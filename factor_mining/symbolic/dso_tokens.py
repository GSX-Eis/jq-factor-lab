"""Token vocabulary for the PyTorch DSO symbolic search backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .feature_space import FeatureSpace, OperatorSpec


TOKEN_START = "<START>"
TOKEN_END = "<END>"


@dataclass(frozen=True)
class DSOToken:
    name: str
    kind: str
    arity: int
    value: str | float | None = None
    complexity: int = 1
    arg_types: Sequence[str] = ()
    category: str = ""

    @property
    def expr_arity(self) -> int:
        return sum(1 for arg in self.arg_types if arg == "expr")


@dataclass(frozen=True)
class DSOTokenVocab:
    tokens: List[DSOToken]
    start_id: int
    end_id: int
    name_to_id: Dict[str, int]

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def terminal_ids(self) -> List[int]:
        return [idx for idx, token in enumerate(self.tokens) if token.kind in {"feature", "constant"}]

    @property
    def scalar_ids(self) -> List[int]:
        return [idx for idx, token in enumerate(self.tokens) if token.kind == "constant"]

    @property
    def window_ids(self) -> List[int]:
        return [idx for idx, token in enumerate(self.tokens) if token.kind == "window"]

    @property
    def expression_ids(self) -> List[int]:
        return [idx for idx, token in enumerate(self.tokens) if token.kind in {"feature", "constant", "operator"}]


def _constant_token_name(value: float) -> str:
    return f"const:{float(value):.12g}"


def _window_token_name(value: int) -> str:
    return f"window:{int(value)}"


def build_dso_vocab(feature_space: FeatureSpace, search_config: dict) -> DSOTokenVocab:
    tokens: List[DSOToken] = [DSOToken(TOKEN_START, "start", 0), DSOToken(TOKEN_END, "end", 0)]

    for feature_name in sorted(feature_space.features):
        tokens.append(DSOToken(name=f"feature:{feature_name}", kind="feature", arity=0, value=feature_name))

    constants = search_config.get("constants", {})
    if constants.get("enabled", False):
        for raw_value in constants.get("values", []):
            value = float(raw_value)
            tokens.append(DSOToken(name=_constant_token_name(value), kind="constant", arity=0, value=value))

    for window in feature_space.windows:
        tokens.append(DSOToken(name=_window_token_name(window), kind="window", arity=0, value=int(window), complexity=0))

    exportable_ops: List[OperatorSpec] = [
        op for op in feature_space.operators.values() if op.exportable and op.arity > 0
    ]
    for op in sorted(exportable_ops, key=lambda item: item.name):
        tokens.append(
            DSOToken(
                name=f"op:{op.name}",
                kind="operator",
                arity=op.arity,
                value=op.name,
                complexity=op.complexity,
                arg_types=op.arg_types,
                category=str(getattr(op, "category", "") or ""),
            )
        )

    name_to_id = {token.name: idx for idx, token in enumerate(tokens)}
    return DSOTokenVocab(
        tokens=tokens,
        start_id=name_to_id[TOKEN_START],
        end_id=name_to_id[TOKEN_END],
        name_to_id=name_to_id,
    )
