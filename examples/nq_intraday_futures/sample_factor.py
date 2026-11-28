"""Deterministic, strictly causal intraday factor for the NQ 1min example."""
from __future__ import annotations

import pandas as pd

from utils.factor_base import FactorBase


class NqIntradayRangeFactor(FactorBase):
    """Per-bar close-minus-open move (in index points) on the active session view."""

    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        values = data["close"] - data["open"]
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=values)
