"""Scheme B storage layout, receipts and shared validation."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

MARKET_FACTOR_IDENTITY_SCHEMA_VERSION = "market_factor_identity.v1"
NAN_CANONICAL_BITS = np.uint64(0x7FF8000000000000)
DEFAULT_ROW_GROUP_SIZE = 65536

VALID_VIEWS = ("ETH", "RTH")
VALID_ANALYSIS_MODES = ("time_series", "cross_sectional")
VALID_BUCKET_STRATEGIES = ("segment_anchored", "session_clock", "session")
VALID_ADJUSTMENTS = ("none", "ratio_back_adjust", "diff_back_adjust")

TABLE_COLUMNS: tuple[str, ...] = (
    "factor_id",
    "factor_identity_hash",
    "frequency",
    "session_view",
    "instrument_id",
    "timestamp_utc",
    "session_id",
    "factor_value",
    "source_build_id",
    "calendar_hash",
    "code_version",
    "created_at",
    "batch_id",
)

VIEW_COLUMNS: tuple[str, ...] = TABLE_COLUMNS + ("year_month",)
LEAF_COLUMNS: tuple[str, ...] = tuple(
    col for col in TABLE_COLUMNS if col not in ("factor_id", "frequency")
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "factor_identity_hash",
    "factor_id",
    "factor_class",
    "frequency",
    "session_view",
    "bucket_strategy",
    "analysis_mode",
    "output_frequency",
    "code_version",
    "code_version_source",
    "source_schema_version",
    "identity_payload_json",
    "created_at",
)

LEAF_ARROW_SCHEMA = pa.schema([
    ("factor_identity_hash", pa.string()),
    ("session_view", pa.string()),
    ("instrument_id", pa.string()),
    ("timestamp_utc", pa.timestamp("ns", tz="UTC")),
    ("session_id", pa.string()),
    ("factor_value", pa.float64()),
    ("source_build_id", pa.string()),
    ("calendar_hash", pa.string()),
    ("code_version", pa.string()),
    ("created_at", pa.timestamp("ns", tz="UTC")),
    ("batch_id", pa.string()),
])

_HIVE_TYPES = {"factor_id": "VARCHAR", "frequency": "VARCHAR", "year_month": "VARCHAR"}
_FACTOR_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SESSION_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class FactorValueConflictError(ValueError):
    """Raised when incoming values conflict with already-committed values for the same key."""

    def __init__(
        self,
        factor_id: str,
        factor_identity_hash: str,
        conflict_count: int,
        examples: Sequence[Mapping[str, Any]],
    ) -> None:
        self.factor_id = factor_id
        self.factor_identity_hash = factor_identity_hash
        self.conflict_count = int(conflict_count)
        self.examples = [dict(item) for item in list(examples)[:5]]
        super().__init__(
            f"Conflict for factor_id={factor_id!r} identity={factor_identity_hash[:12]}...: "
            f"{self.conflict_count} conflicting rows; examples={self.examples}"
        )


@dataclass(frozen=True)
class FactorWriteReceipt:
    """Structured receipt returned by append_factor_values / save_bar_factor_values."""

    batch_id: str
    factor_id: str
    factor_identity_hash: str
    frequency: str
    session_view: str
    inserted: int
    skipped: int
    shards_written: tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shards_written"] = list(self.shards_written)
        return payload


@dataclass(frozen=True)
class FactorReadPlan:
    """Resolved partition filters and exact committed shards for a read query."""

    resolved_factor_ids: tuple[str, ...]
    resolved_frequencies: tuple[str, ...]
    resolved_year_months: tuple[str, ...] | None
    shard_paths: tuple[Path, ...]
    sql: str
    params: tuple[Any, ...]


def sql_canonical_float(expr: str) -> str:
    """Normalize DuckDB SQL IEEE NaN and NULL to a single SQL NULL sentinel."""
    return f"NULLIF(CAST({expr} AS DOUBLE), CAST('nan' AS DOUBLE))"


def canonical_float_bits(arr: Any) -> np.ndarray:
    """Return IEEE-754 uint64 bits with all NaNs unified and -0.0 mapped to +0.0."""
    raw = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
    if raw.dtype == np.bool_:
        raise TypeError("factor_value must be real numbers, not booleans")
    if raw.dtype == object:
        # Reject bools/strings embedded in object arrays before float64 coercion.
        vals = np.empty(len(raw), dtype=np.float64)
        for idx, item in enumerate(raw):
            if item is None or item is pd.NA:
                vals[idx] = np.nan
            elif isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float, np.number)):
                raise TypeError("factor_value must contain real numbers or NaN/None")
            else:
                vals[idx] = float(item)
    else:
        vals = np.asarray(raw, dtype=np.float64).copy()
    nan_mask = np.isnan(vals)
    vals[vals == 0.0] = 0.0
    bits = vals.view(np.uint64)
    bits[nan_mask] = NAN_CANONICAL_BITS
    return bits


def validate_factor_id(factor_id: str) -> str:
    """Validate factor_id as a safe, non-empty single Hive partition path segment."""
    if (
        not isinstance(factor_id, str)
        or not factor_id
        or factor_id in (".", "..")
        or _FACTOR_ID_RE.fullmatch(factor_id) is None
    ):
        raise ValueError(
            f"factor_id must be a non-empty single safe path identifier [A-Za-z0-9_.-], got {factor_id!r}"
        )
    return factor_id


def validate_sha256_hex(value: str, name: str) -> str:
    """Validate a 64-character lowercase SHA256 hex digest."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character lowercase hex SHA256 digest, got {value!r}")
    return value


def _validate_session_date_str(value: str, name: str = "session_id") -> str:
    if not isinstance(value, str) or _SESSION_DATE_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}")
    try:
        dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid calendar date: {value!r}") from exc
    return value


def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()

    class _HashWriter:
        def write(self, data: bytes) -> int:
            digest.update(data)
            return len(data)

    pickle.dump(frame, _HashWriter(), protocol=4)
    return digest.hexdigest()
