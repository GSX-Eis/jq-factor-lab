"""Reject mixed physical market inputs before cross-sectional operations."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

_SOURCE_FIELDS = ("market_databases", "market_db", "market_factor_inputs", "factors",
                  "slot_data_fingerprint", "data_identity", "market_db_identity")


def market_db_dimensions(*sources: object) -> dict[str, tuple[str, ...]]:
    def collect(source: object, field: str) -> Iterable[str]:
        if isinstance(source, (pd.DataFrame, pd.Series)):
            yield from collect(source.attrs, field)
            if isinstance(source, pd.DataFrame) and field in source:
                yield from source[field].dropna().unique()
            if field in source.index.names:
                yield from source.index.get_level_values(field).dropna().unique()
        elif isinstance(source, Mapping):
            for key in (field, f"{field}s"):
                values = source.get(key)
                if isinstance(values, str):
                    yield values
                elif isinstance(values, (list, tuple, set, frozenset)):
                    yield from values
            for key in _SOURCE_FIELDS:
                yield from collect(source.get(key), field)
        elif isinstance(source, (list, tuple)):
            for item in source:
                yield from collect(item, field)

    return {
        field: tuple(sorted({value for source in sources for value in collect(source, field)}))
        for field in ("db_id", "market_domain")
    }


def require_single_market_db(*sources: object) -> None:
    for field, values in market_db_dimensions(*sources).items():
        if len(values) > 1:
            raise ValueError(f"Cross-sectional inputs span multiple market databases or domains ({field})")
