"""Search backend contracts for symbolic factor mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import pandas as pd

from .expression import ExpressionNode
from .feature_space import FeatureSpace


@dataclass
class SearchCandidate:
    expression: ExpressionNode
    train_metrics: Dict[str, float]
    valid_metrics: Dict[str, float]
    score: float
    complexity: int


@dataclass
class SearchResult:
    best: SearchCandidate
    hall_of_fame: List[SearchCandidate]
    metadata: Dict[str, Any]


class SymbolicSearchBackend(Protocol):
    def search(
        self,
        train_panel: pd.DataFrame,
        valid_panel: pd.DataFrame,
        feature_space: FeatureSpace,
        config: dict,
    ) -> SearchResult:
        ...

