"""Canonical evaluation column values and comparison tokens."""
from __future__ import annotations

import datetime as dt
import json
import math
import struct
from collections.abc import Mapping
from sys import float_info
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    _DATE_ONLY_RE,
    BarEvaluationIntegrityError,
    BarEvaluationPreconditionError,
)


def _require(condition: bool, field: str, expected: str) -> None:
    if not condition:
        raise BarEvaluationIntegrityError(f"summary.json {field}: expected {expected}")


def _object(value: Any, field: str, *required: str) -> dict[str, Any]:
    _require(isinstance(value, dict), field, "an object")
    missing = set(required) - value.keys()
    _require(not missing, field, f"required fields {sorted(missing)}")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), field, "an array")
    return value


def _count(value: Any, field: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and minimum <= value < 2**63, field, f"an integer >= {minimum}")
    return value


def _metric(value: Any, field: str, *, probability: bool = False) -> float | None:
    if value is None:
        return None
    _require(type(value) in (float, int) and abs(value) <= float_info.max,
             field, "a finite number or null")
    if probability:
        _require(0 <= value <= 1, field, "a probability in [0, 1]")
    return 0.0 if value == 0 else float(value)


def _nullable_text(value: Any, field: str) -> None:
    _require(value is None or isinstance(value, str), field, "a string or null")


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(float(value)))
    try:
        res = pd.isna(value)
        if isinstance(res, (bool, np.bool_)):
            return bool(res)
    except (TypeError, ValueError):
        pass
    return False


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        num = float(value)
        return None if not math.isfinite(num) else (0.0 if num == 0.0 else num)
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return _format_utc_iso(value)
    return value


def _format_utc_iso(value: Any) -> str | None:
    if _is_missing(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: Any, name: str) -> pd.Timestamp:
    if _is_missing(value):
        raise BarEvaluationPreconditionError(f"{name} must not be null")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise BarEvaluationPreconditionError(f"{name} must be timezone-aware UTC: {value!r}")
    return ts.tz_convert("UTC")


def _parse_filter_timestamp(value: str, *, is_end: bool = False) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise BarEvaluationPreconditionError("Timestamp filter must be a nonempty string")
    text = value.strip()
    if _DATE_ONLY_RE.match(text):
        day = pd.Timestamp(text, tz="UTC")
        if is_end:
            return day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return day
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        raise BarEvaluationPreconditionError(
            f"Timestamp filter must include UTC timezone (or YYYY-MM-DD): {value!r}"
        )
    return ts.tz_convert("UTC")


def _coerce_optional_int(value: Any, field: str) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise BarEvaluationIntegrityError(f"{field} must be an integer, got bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        num = float(value)
        if not math.isfinite(num) or not num.is_integer():
            raise BarEvaluationIntegrityError(f"{field} must be an integer, got {value!r}")
        return int(num)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError as exc:
            raise BarEvaluationIntegrityError(f"{field} must be an integer, got {value!r}") from exc
    raise BarEvaluationIntegrityError(f"{field} must be an integer, got {type(value).__name__}")


def _coerce_optional_float(value: Any, field: str) -> float | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise BarEvaluationIntegrityError(f"{field} must be a real number, got bool")
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise BarEvaluationIntegrityError(f"{field} must be numeric, got {value!r}") from exc
    if math.isnan(num):
        return None
    if not math.isfinite(num):
        raise BarEvaluationIntegrityError(f"{field} must be finite, got {value!r}")
    return 0.0 if num == 0.0 else num


def _coerce_optional_bool(value: Any, field: str) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise BarEvaluationIntegrityError(f"{field} must be a boolean, got {value!r}")


def _coerce_optional_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    return str(value)


def _coerce_optional_json_str(value: Any, field: str) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, (Mapping, list)):
        return _canonical_json(_json_safe(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BarEvaluationIntegrityError(f"{field} contains invalid JSON: {exc}") from exc
        return _canonical_json(_json_safe(parsed))
    raise BarEvaluationIntegrityError(f"{field} must be JSON text, an object, or an array")


def _canonical_compare_token(value: Any, *, kind: str) -> Any:
    """Normalize NaN/NULL, -0.0/+0.0, JSON objects, and timestamps for conflict checks."""
    if _is_missing(value):
        return ("MISSING",)
    if kind == "float":
        num = float(value)
        if math.isnan(num):
            return ("MISSING",)
        if num == 0.0:
            num = 0.0
        return ("FLOAT", struct.pack("!d", num))
    if kind == "int":
        return ("INT", int(value))
    if kind == "bool":
        return ("BOOL", bool(value))
    if kind == "json":
        parsed = json.loads(value) if isinstance(value, str) else value
        return ("JSON", _canonical_json(_json_safe(parsed)))
    if kind == "timestamp":
        return ("TS", _format_utc_iso(value))
    return ("STR", str(value))
