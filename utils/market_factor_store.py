"""Append-only, versioned minute/bar factor value store (Scheme B).

Storage layout under an explicit store root:
  - parquet_data/factor_id=<fid>/frequency=<freq>/year_month=<YYYY-MM>/*.parquet (ZSTD)
  - identities.jsonl: append-only identity registry (factor_value_identities)
  - batches.jsonl: append-only committed shard manifest
  - .ledger.lock: POSIX fcntl.flock advisory lock protecting writers only

Readers never acquire the writer lock and never glob uncommitted Parquet files;
all reads and DuckDB views resolve strictly through committed batch records in
batches.jsonl. Note: fcntl.flock requires a POSIX host (macOS/Linux).
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import inspect
import json
import os
import pickle
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from backtest_and_analysis.time_series.contracts import DuplicateKeyError, _utc
from utils.bar_catalog import BarCatalog, table_exists
from utils.code_version import resolve_code_version
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule, Frequency

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


def _normalize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(params, Mapping) or any(not isinstance(k, str) for k in params):
        raise TypeError("params must be a mapping with string keys")
    normalized: dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        if isinstance(value, pd.DataFrame):
            normalized[key] = ["dataframe", _frame_digest(value)]
        else:
            encoded = _canonical_json(value)
            decoded = json.loads(encoded)
            if decoded != value:
                raise TypeError(f"Parameter {key!r} must use strict JSON types without NaN or coercion")
            normalized[key] = ["json", decoded]
    return normalized


def _normalize_calendar_mapping(
    calendar_hash: str | Mapping[str, str],
    name: str = "calendar_hash",
) -> str | dict[str, str]:
    if isinstance(calendar_hash, str):
        return validate_sha256_hex(calendar_hash, name)
    if isinstance(calendar_hash, Mapping) and calendar_hash:
        return {
            _require_nonempty_str(k, f"{name} key"): validate_sha256_hex(v, f"{name}[{k!r}]")
            for k, v in sorted(calendar_hash.items())
        }
    raise ValueError(f"{name} must be a SHA256 hex string or non-empty instrument->SHA256 mapping")


def _normalize_rth_reference(
    rth_reference_calendar: str | Mapping[str, str] | None,
) -> str | dict[str, str] | None:
    if rth_reference_calendar is None:
        return None
    if isinstance(rth_reference_calendar, str):
        return _require_nonempty_str(rth_reference_calendar, "rth_reference_calendar")
    if isinstance(rth_reference_calendar, Mapping) and rth_reference_calendar:
        return {
            _require_nonempty_str(k, "rth_reference_calendar key"): _require_nonempty_str(
                v, f"rth_reference_calendar[{k!r}]"
            )
            for k, v in sorted(rth_reference_calendar.items())
        }
    raise ValueError("rth_reference_calendar must be a non-empty string, mapping, or None")


def _normalize_source_build_id(value: Any) -> Any:
    if isinstance(value, str):
        return _require_nonempty_str(value, "source_build_id")
    if isinstance(value, Mapping) and value:
        out: dict[str, Any] = {}
        for k, v in sorted(value.items()):
            ks = _require_nonempty_str(k, "source_build_id key")
            if isinstance(v, str):
                out[ks] = _require_nonempty_str(v, f"source_build_id[{k!r}]")
            elif isinstance(v, Mapping) and v:
                out[ks] = {
                    _require_nonempty_str(sk, "session_id"): _require_nonempty_str(
                        sv, f"source_build_id[{k!r}][{sk!r}]"
                    )
                    for sk, sv in sorted(v.items())
                }
            else:
                raise ValueError("source_build_id nested mapping must be non-empty")
        return out
    raise ValueError("source_build_id must be a non-empty string or non-empty mapping")


def compute_factor_identity_hash(
    *,
    factor_id: str,
    factor_class: str,
    source_sha256: str | Sequence[str],
    params: Mapping[str, Any],
    frequency: str,
    output_frequency: str | None = None,
    session_view: str,
    analysis_mode: str,
    bucket_strategy: str,
    calendar_hash: str | Mapping[str, str],
    source_identity: Mapping[str, Any] | None = None,
    source_build_id: str | Mapping[str, Any] | None = None,
    data_version: str | None = None,
    rth_reference_calendar: str | Mapping[str, str] | None = None,
    continuous_adjust: str | None = None,
    roll_build_ids: Mapping[str, Any] | None = None,
    code_version: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compute canonical SHA256 factor_identity_hash and return (hash, canonical_payload)."""
    fid = validate_factor_id(factor_id)
    fclass = _require_nonempty_str(factor_class, "factor_class")
    if isinstance(source_sha256, str):
        src_hashes: list[str] = [validate_sha256_hex(source_sha256, "source_sha256")]
    elif isinstance(source_sha256, Sequence) and not isinstance(source_sha256, (bytes, bytearray)) and source_sha256:
        src_hashes = [validate_sha256_hex(item, "source_sha256[]") for item in source_sha256]
    else:
        raise ValueError("source_sha256 must be a SHA256 hex string or non-empty sequence of SHA256 strings")

    parsed_freq = Frequency.parse(frequency)
    if not parsed_freq.is_valid:
        raise ValueError(f"Invalid frequency: {frequency!r}")
    out_raw = frequency if output_frequency is None else output_frequency
    parsed_out = Frequency.parse(out_raw)
    if not parsed_out.is_valid:
        raise ValueError(f"Invalid output_frequency: {output_frequency!r}")

    if session_view not in VALID_VIEWS:
        raise ValueError(f"session_view must be one of {VALID_VIEWS}, got {session_view!r}")
    if analysis_mode not in VALID_ANALYSIS_MODES:
        raise ValueError(f"analysis_mode must be one of {VALID_ANALYSIS_MODES}, got {analysis_mode!r}")
    if bucket_strategy not in VALID_BUCKET_STRATEGIES:
        raise ValueError(f"bucket_strategy must be one of {VALID_BUCKET_STRATEGIES}, got {bucket_strategy!r}")

    norm_cal = _normalize_calendar_mapping(calendar_hash)
    norm_rth = _normalize_rth_reference(rth_reference_calendar)
    norm_params = _normalize_params(params)

    src_id: dict[str, Any] = {}
    if source_identity is not None:
        if not isinstance(source_identity, Mapping):
            raise TypeError("source_identity must be a mapping")
        src_id = json.loads(_canonical_json(dict(source_identity)))
    if data_version is not None:
        src_id["data_version"] = _require_nonempty_str(data_version, "data_version")
    if source_build_id is not None:
        src_id["source_build_id"] = _normalize_source_build_id(source_build_id)
    if continuous_adjust is not None:
        if continuous_adjust not in VALID_ADJUSTMENTS:
            raise ValueError(f"continuous_adjust must be one of {VALID_ADJUSTMENTS}, got {continuous_adjust!r}")
        cont = dict(src_id.get("continuous") or {})
        cont["adjust"] = continuous_adjust
        src_id["continuous"] = cont
    if roll_build_ids is not None:
        if not isinstance(roll_build_ids, Mapping) or not roll_build_ids:
            raise ValueError("roll_build_ids must be a non-empty mapping when provided")
        cont = dict(src_id.get("continuous") or {})
        cont["roll_build_ids"] = {
            _require_nonempty_str(str(k), "roll session"): _require_nonempty_str(v, "roll build_id")
            for k, v in sorted(roll_build_ids.items())
        }
        src_id["continuous"] = cont
    if isinstance(src_id.get("continuous"), Mapping):
        adj = src_id["continuous"].get("adjust")
        if adj is not None and adj not in VALID_ADJUSTMENTS:
            raise ValueError(f"continuous adjust must be one of {VALID_ADJUSTMENTS}, got {adj!r}")

    has_source_build = bool(
        src_id.get("source_build_id")
        or src_id.get("source_build_ids")
        or src_id.get("source_dataset_version")
    )
    if not has_source_build:
        raise ValueError(
            "source_build_id (or source_identity with source_build_id/source_build_ids/source_dataset_version) is required"
        )

    resolved_cv = _require_nonempty_str(code_version, "code_version") if code_version is not None else resolve_code_version()[0]

    payload: dict[str, Any] = {
        "schema_version": MARKET_FACTOR_IDENTITY_SCHEMA_VERSION,
        "factor_id": fid,
        "factor_class": {"name": fclass, "source_sha256": src_hashes},
        "params": norm_params,
        "frequency": parsed_freq.canonical,
        "output_frequency": parsed_out.canonical,
        "session_view": session_view,
        "analysis_mode": analysis_mode,
        "bucket_strategy": bucket_strategy,
        "calendar_hash": norm_cal,
        "source_identity": src_id,
        "code_version": resolved_cv,
    }
    if norm_rth is not None:
        payload["rth_reference_calendar"] = norm_rth

    canonical_str = _canonical_json(payload)
    digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    return digest, payload


def build_identity_record(
    *,
    factor_identity_hash: str,
    payload: Mapping[str, Any],
    code_version_source: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a factor_value_identities registration dict from computed hash & payload."""
    validate_sha256_hex(factor_identity_hash, "factor_identity_hash")
    now_iso = created_at or dt.datetime.now(dt.timezone.utc).isoformat()
    factor_class_val = payload["factor_class"]
    if isinstance(factor_class_val, Mapping):
        factor_class_name = str(factor_class_val["name"])
    else:
        factor_class_name = str(factor_class_val)
    return {
        "factor_identity_hash": factor_identity_hash,
        "factor_id": str(payload["factor_id"]),
        "factor_class": factor_class_name,
        "frequency": str(payload["frequency"]),
        "session_view": str(payload["session_view"]),
        "bucket_strategy": str(payload["bucket_strategy"]),
        "analysis_mode": str(payload["analysis_mode"]),
        "output_frequency": str(payload["output_frequency"]),
        "code_version": str(payload["code_version"]),
        "code_version_source": _require_nonempty_str(code_version_source, "code_version_source"),
        "source_schema_version": str(payload.get("schema_version", MARKET_FACTOR_IDENTITY_SCHEMA_VERSION)),
        "identity_payload_json": _canonical_json(payload),
        "created_at": now_iso,
    }


def _validate_utc_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must be timezone-aware UTC datetime64, got {series.dtype}")
    tz_str = str(series.dt.tz)
    if tz_str not in ("UTC", "Etc/UTC"):
        raise ValueError(f"{name} must have UTC timezone, got {tz_str!r}")
    if series.isna().any():
        raise ValueError(f"{name} must not contain NaT")
    return pd.to_datetime(series, utc=True)


def _validate_text_series(series: pd.Series, name: str) -> None:
    if series.isna().any():
        raise ValueError(f"{name} must not contain null/NA values")
    for val in series.unique():
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{name} must contain non-empty strings, got {val!r}")


def _validate_calendars_for_batch(
    df: pd.DataFrame,
    session_view: str,
    calendar: SessionCalendar | Mapping[str, SessionCalendar],
) -> None:
    """Verify every (instrument_id, timestamp_utc, session_id) aligns with SessionCalendar.session_at."""
    unique_triples = df[["instrument_id", "timestamp_utc", "session_id"]].drop_duplicates()
    if isinstance(calendar, SessionCalendar):
        for ts, sid in unique_triples[["timestamp_utc", "session_id"]].drop_duplicates().itertuples(index=False):
            owner = calendar.session_at(ts, session_view)
            if owner is None or owner.session_date != sid:
                raise ValueError(
                    f"session_id mismatch at {ts} ({session_view}): "
                    f"calendar says {None if owner is None else owner.session_date!r}, got {sid!r}"
                )
        return
    if not isinstance(calendar, Mapping):
        raise TypeError("calendar must be a SessionCalendar or instrument_id -> SessionCalendar mapping")
    for inst, group in unique_triples.groupby("instrument_id", sort=False):
        cal = calendar.get(inst)
        if not isinstance(cal, SessionCalendar):
            raise ValueError(f"Missing SessionCalendar for instrument_id={inst!r}")
        for ts, sid in group[["timestamp_utc", "session_id"]].itertuples(index=False):
            owner = cal.session_at(ts, session_view)
            if owner is None or owner.session_date != sid:
                raise ValueError(
                    f"session_id mismatch for {inst!r} at {ts} ({session_view}): "
                    f"calendar says {None if owner is None else owner.session_date!r}, got {sid!r}"
                )


_REQUIRED_IDENTITY_PAYLOAD_KEYS = (
    "schema_version",
    "factor_id",
    "factor_class",
    "params",
    "frequency",
    "output_frequency",
    "session_view",
    "analysis_mode",
    "bucket_strategy",
    "calendar_hash",
    "source_identity",
    "code_version",
)


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {token!r} is not permitted in canonical JSON")


def _validate_identity_record(
    identity_record: Mapping[str, Any],
    *,
    expected_hash: str | None = None,
    expected_factor_id: str | None = None,
    expected_frequency: str | None = None,
    expected_session_view: str | None = None,
) -> dict[str, Any]:
    """Validate an identity_record and verify canonical JSON SHA256 against factor_identity_hash."""
    if not isinstance(identity_record, Mapping):
        raise TypeError("identity_record must be a mapping")
    missing_cols = [col for col in IDENTITY_COLUMNS if col not in identity_record]
    if missing_cols:
        raise ValueError(f"identity_record is missing required fields: {missing_cols}")

    rec_hash = validate_sha256_hex(str(identity_record["factor_identity_hash"]), "identity_record.factor_identity_hash")
    if expected_hash is not None and rec_hash != expected_hash:
        raise ValueError("identity_record.factor_identity_hash does not match batch")

    raw_payload_json = identity_record.get("identity_payload_json")
    if not isinstance(raw_payload_json, str) or not raw_payload_json.strip():
        raise ValueError("identity_record.identity_payload_json must be a non-empty JSON string")
    try:
        payload_obj = json.loads(raw_payload_json, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"identity_record.identity_payload_json is invalid JSON: {exc}") from exc
    if not isinstance(payload_obj, dict):
        raise ValueError("identity_record.identity_payload_json must decode to a JSON object")

    missing_payload_keys = [k for k in _REQUIRED_IDENTITY_PAYLOAD_KEYS if k not in payload_obj]
    if missing_payload_keys:
        raise ValueError(
            f"identity_record.identity_payload_json missing required keys: {missing_payload_keys}"
        )

    canonical_payload = _canonical_json(payload_obj)
    recomputed_hash = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    if recomputed_hash != rec_hash:
        raise ValueError(
            f"identity_record SHA256 verification failed: canonical payload hashes to {recomputed_hash}, "
            f"but factor_identity_hash is {rec_hash}"
        )

    schema_ver = _require_nonempty_str(payload_obj.get("schema_version"), "payload.schema_version")
    if schema_ver != MARKET_FACTOR_IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported identity payload schema_version: {schema_ver!r}")
    if str(identity_record["source_schema_version"]) != schema_ver:
        raise ValueError("identity_record.source_schema_version does not match payload schema_version")

    fid = validate_factor_id(str(payload_obj["factor_id"]))
    if str(identity_record["factor_id"]) != fid:
        raise ValueError("identity_record.factor_id does not match payload factor_id")
    if expected_factor_id is not None and fid != expected_factor_id:
        raise ValueError("identity_record.factor_id does not match batch")

    p_freq = Frequency.parse(str(payload_obj["frequency"]))
    r_freq = Frequency.parse(str(identity_record["frequency"]))
    if not p_freq.is_valid or not r_freq.is_valid or p_freq.canonical != r_freq.canonical:
        raise ValueError("identity_record.frequency does not match payload frequency")

    p_out_freq = Frequency.parse(str(payload_obj["output_frequency"]))
    r_out_freq = Frequency.parse(str(identity_record["output_frequency"]))
    if not p_out_freq.is_valid or not r_out_freq.is_valid or p_out_freq.canonical != r_out_freq.canonical:
        raise ValueError("identity_record.output_frequency does not match payload output_frequency")
    if expected_frequency is not None and r_out_freq.canonical != expected_frequency:
        raise ValueError("identity_record.output_frequency does not match batch frequency")

    view = str(payload_obj["session_view"])
    if view not in VALID_VIEWS or str(identity_record["session_view"]) != view:
        raise ValueError("identity_record.session_view does not match payload session_view")
    if expected_session_view is not None and view != expected_session_view:
        raise ValueError("identity_record.session_view does not match batch")

    for field in ("bucket_strategy", "analysis_mode", "code_version"):
        p_val = _require_nonempty_str(payload_obj.get(field), f"payload.{field}")
        if str(identity_record[field]) != p_val:
            raise ValueError(f"identity_record.{field} does not match payload {field}")

    _require_nonempty_str(identity_record["factor_class"], "identity_record.factor_class")
    _require_nonempty_str(identity_record["code_version_source"], "identity_record.code_version_source")
    _require_nonempty_str(identity_record["created_at"], "identity_record.created_at")

    validated = dict(identity_record)
    validated["factor_identity_hash"] = rec_hash
    validated["factor_id"] = fid
    validated["frequency"] = r_freq.canonical
    validated["output_frequency"] = r_out_freq.canonical
    validated["identity_payload_json"] = canonical_payload
    return validated


def _validate_and_prepare_batch(
    df: pd.DataFrame,
    *,
    calendar: SessionCalendar | Mapping[str, SessionCalendar] | None = None,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("batch input must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("batch columns must be unique")
    missing = set(TABLE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = df.loc[:, list(TABLE_COLUMNS)].copy()
    clean["timestamp_utc"] = _validate_utc_series(clean["timestamp_utc"], "timestamp_utc").dt.as_unit("ns")
    clean["created_at"] = _validate_utc_series(clean["created_at"], "created_at").dt.as_unit("ns")

    val_series = clean["factor_value"]
    if is_bool_dtype(val_series.dtype) or is_complex_dtype(val_series.dtype):
        raise ValueError("factor_value must be real numbers (NaN allowed), not bool or complex")
    if not is_numeric_dtype(val_series.dtype):
        # Allow object columns containing floats/ints/None/NaN, reject bools/strings.
        bits_check = canonical_float_bits(val_series.to_numpy(copy=False))
        del bits_check
        clean["factor_value"] = pd.to_numeric(val_series, errors="raise").astype(np.float64)
    else:
        clean["factor_value"] = val_series.to_numpy(dtype=np.float64, copy=True)

    vals = clean["factor_value"].to_numpy(dtype=np.float64, copy=False)
    if np.isinf(vals).any():
        raise ValueError("factor_value must not contain +-inf (NaN is allowed)")
    if (vals == 0.0).any():
        vals = vals.copy()
        vals[vals == 0.0] = 0.0
        clean["factor_value"] = vals

    if clean.empty:
        return clean

    if calendar is None:
        raise ValueError("calendar (SessionCalendar or instrument->SessionCalendar mapping) is required on every write")

    for col in ("factor_id", "factor_identity_hash", "frequency", "session_view"):
        _validate_text_series(clean[col], col)
        if clean[col].nunique(dropna=False) != 1:
            raise ValueError(f"Single batch must share exactly one {col}, got {clean[col].unique().tolist()}")

    fid = validate_factor_id(str(clean["factor_id"].iloc[0]))
    fhash = validate_sha256_hex(str(clean["factor_identity_hash"].iloc[0]), "factor_identity_hash")
    freq_raw = str(clean["frequency"].iloc[0])
    parsed_freq = Frequency.parse(freq_raw)
    if not parsed_freq.is_valid:
        raise ValueError(f"Invalid frequency: {freq_raw!r}")
    clean["frequency"] = parsed_freq.canonical
    clean["factor_id"] = fid
    clean["factor_identity_hash"] = fhash

    view = str(clean["session_view"].iloc[0])
    if view not in VALID_VIEWS:
        raise ValueError(f"Invalid session_view: {view!r}; expected one of {VALID_VIEWS}")

    for col in ("instrument_id", "session_id", "source_build_id", "calendar_hash", "code_version", "batch_id"):
        _validate_text_series(clean[col], col)

    for cal_hex in clean["calendar_hash"].unique():
        validate_sha256_hex(cal_hex, "calendar_hash")
    for sid in clean["session_id"].unique():
        _validate_session_date_str(sid, "session_id")

    dup_mask = clean.duplicated(subset=["factor_identity_hash", "instrument_id", "timestamp_utc"], keep=False)
    if dup_mask.any():
        examples = clean.loc[dup_mask, ["instrument_id", "timestamp_utc"]].drop_duplicates().head(5).to_dict("records")
        raise DuplicateKeyError(f"Duplicate primary keys in batch: {examples}")

    _validate_calendars_for_batch(clean, view, calendar)

    return clean


def _trim_unterminated_trailing_line(path: Path) -> None:
    """Under writer lock, trim an uncommitted unterminated trailing line before appending."""
    if not path.is_file():
        return
    fd = os.open(str(path), os.O_RDWR)
    try:
        size = os.lseek(fd, 0, os.SEEK_END)
        if size <= 0:
            return
        os.lseek(fd, size - 1, os.SEEK_SET)
        if os.read(fd, 1) == b"\n":
            return
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, size)
        last_nl = raw.rfind(b"\n")
        trim_to = 0 if last_nl == -1 else last_nl + 1
        os.ftruncate(fd, trim_to)
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_jsonl_line(path: Path, record: Mapping[str, Any]) -> None:
    _trim_unterminated_trailing_line(path)
    line = (_canonical_json(dict(record)) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = 0
        while written < len(line):
            n = os.write(fd, line[written:])
            if n <= 0:
                raise OSError(f"Short write appending to {path}")
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read complete JSONL records without writer lock; separate mid-file corrupt vs trailing partial lines."""
    if not path.is_file():
        return [], [], []
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    corrupt: list[dict[str, Any]] = []
    trailing_incomplete: list[dict[str, Any]] = []
    pos = 0
    line_no = 0
    while pos < len(raw):
        nl = raw.find(b"\n", pos)
        line_no += 1
        if nl == -1:
            trailing_incomplete.append({"file": str(path), "line": line_no, "reason": "unterminated_line"})
            break
        raw_line = raw[pos : nl + 1]
        pos = nl + 1
        text = raw_line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
            if not isinstance(obj, dict):
                corrupt.append({"file": str(path), "line": line_no, "reason": "non_object_json"})
                continue
            records.append(obj)
        except json.JSONDecodeError as exc:
            corrupt.append({"file": str(path), "line": line_no, "reason": f"json_error: {exc}"})
    return records, corrupt, trailing_incomplete


class MarketFactorValueStore:
    """Append-only Hive-partitioned Parquet + flock-protected JSONL ledger + DuckDB view."""

    def __init__(self, root: str | Path) -> None:
        if root is None or (isinstance(root, str) and not root.strip()):
            raise ValueError("MarketFactorValueStore requires an explicit root directory")
        self.root = Path(root).expanduser().resolve()
        self.parquet_root = self.root / "parquet_data"
        self.identities_path = self.root / "identities.jsonl"
        self.batches_path = self.root / "batches.jsonl"
        self.lock_path = self.root / ".ledger.lock"
        self._last_read_plan: FactorReadPlan | None = None

    @property
    def last_read_plan(self) -> FactorReadPlan | None:
        return self._last_read_plan

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _load_committed_state(self) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        identities_list, id_corrupt, _ = _read_jsonl(self.identities_path)
        batches_list, batch_corrupt, _ = _read_jsonl(self.batches_path)
        if id_corrupt or batch_corrupt:
            raise ValueError(f"Corrupt ledger entries detected: {id_corrupt + batch_corrupt}")
        identities = {str(rec["factor_identity_hash"]): rec for rec in identities_list}
        committed_batches = [rec for rec in batches_list if rec.get("status") == "committed"]
        return identities, committed_batches

    def append_factor_values(
        self,
        df: pd.DataFrame,
        *,
        identity_record: Mapping[str, Any] | None = None,
        calendar: SessionCalendar | Mapping[str, SessionCalendar] | None = None,
        strict_source_build_id: bool = False,
    ) -> FactorWriteReceipt:
        """Validate and idempotently append a single-identity batch; never updates or deletes."""
        clean = _validate_and_prepare_batch(df, calendar=calendar)
        if clean.empty:
            if identity_record is None:
                raise ValueError("Empty batch requires identity_record or at least one row")
            validated_rec = _validate_identity_record(identity_record)
            now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
            return FactorWriteReceipt(
                batch_id=uuid.uuid4().hex,
                factor_id=validated_rec["factor_id"],
                factor_identity_hash=validated_rec["factor_identity_hash"],
                frequency=validated_rec["output_frequency"],
                session_view=str(validated_rec["session_view"]),
                inserted=0,
                skipped=0,
                shards_written=(),
                created_at=now_iso,
            )

        fid = str(clean["factor_id"].iloc[0])
        fhash = str(clean["factor_identity_hash"].iloc[0])
        freq = str(clean["frequency"].iloc[0])
        view = str(clean["session_view"].iloc[0])
        batch_id = str(clean["batch_id"].iloc[0])
        clean["year_month"] = clean["session_id"].str.slice(0, 7)
        incoming_sessions = set(clean["session_id"].unique().tolist())

        validated_id_rec: dict[str, Any] | None = None
        if identity_record is not None:
            validated_id_rec = _validate_identity_record(
                identity_record,
                expected_hash=fhash,
                expected_factor_id=fid,
                expected_frequency=freq,
                expected_session_view=view,
            )

        inc_ts_ns_series = pd.to_datetime(clean["timestamp_utc"], utc=True).dt.as_unit("ns").astype("int64")
        inc_min_ns = int(inc_ts_ns_series.min())
        inc_max_ns = int(inc_ts_ns_series.max())

        with self._write_lock():
            identities, committed_batches = self._load_committed_state()
            existing_id_rec = identities.get(fhash)
            if existing_id_rec is not None:
                existing_out_freq = existing_id_rec.get("output_frequency") or existing_id_rec["frequency"]
                if (
                    existing_id_rec["factor_id"] != fid
                    or existing_out_freq != freq
                    or existing_id_rec["session_view"] != view
                ):
                    raise ValueError(
                        f"Identity hash {fhash} already registered with different metadata: {existing_id_rec}"
                    )
            elif validated_id_rec is None:
                raise ValueError(
                    f"factor_identity_hash {fhash} is not registered in {self.identities_path.name}; "
                    "a verifiable identity_record is required"
                )

            candidate_shards: list[Path] = []
            for batch in committed_batches:
                if batch.get("factor_identity_hash") != fhash:
                    continue
                for shard in batch.get("shards", []):
                    has_ts_bounds = "min_timestamp_ns" in shard and "max_timestamp_ns" in shard
                    ts_overlaps = (
                        not has_ts_bounds
                        or (
                            int(shard["min_timestamp_ns"]) <= inc_max_ns
                            and int(shard["max_timestamp_ns"]) >= inc_min_ns
                        )
                    )
                    sess_overlaps = bool(set(shard.get("sessions", {}).keys()) & incoming_sessions)
                    if ts_overlaps or (strict_source_build_id and sess_overlaps):
                        candidate_shards.append(self.parquet_root / shard["part_path"])

            to_write = clean
            skipped = 0
            if candidate_shards:
                read_cols = ["instrument_id", "timestamp_utc", "factor_value", "session_id", "source_build_id"]
                pq_filters: list[tuple[str, str, Any]] = [("factor_identity_hash", "=", fhash)]
                if not strict_source_build_id:
                    pq_filters.extend(
                        [
                            ("timestamp_utc", ">=", pa.scalar(inc_min_ns, type=pa.timestamp("ns", tz="UTC"))),
                            ("timestamp_utc", "<=", pa.scalar(inc_max_ns, type=pa.timestamp("ns", tz="UTC"))),
                        ]
                    )
                tables = [
                    pq.read_table(
                        str(shard_path),
                        columns=read_cols,
                        filters=pq_filters,
                    )
                    for shard_path in candidate_shards
                ]
                existing_df = pa.concat_tables(tables).to_pandas() if tables else pd.DataFrame(columns=read_cols)
                if not existing_df.empty:
                    existing_df["timestamp_utc"] = pd.to_datetime(existing_df["timestamp_utc"], utc=True).dt.as_unit("ns")
                    if strict_source_build_id:
                        sess_merge = (
                            clean[["instrument_id", "session_id", "source_build_id"]]
                            .drop_duplicates()
                            .merge(
                                existing_df[["instrument_id", "session_id", "source_build_id"]].drop_duplicates(),
                                on=["instrument_id", "session_id"],
                                how="inner",
                                suffixes=("_inc", "_ext"),
                            )
                        )
                        build_diff = sess_merge["source_build_id_inc"] != sess_merge["source_build_id_ext"]
                        if build_diff.any():
                            bad_sess = sess_merge.loc[build_diff].iloc[0]
                            raise ValueError(
                                f"Upstream source_build_id changed for existing session "
                                f"({bad_sess.instrument_id!r}, {bad_sess.session_id!r}) under "
                                f"factor_identity_hash={fhash}: {bad_sess.source_build_id_ext!r} -> "
                                f"{bad_sess.source_build_id_inc!r}; mint a new factor_identity_hash"
                            )

                    merged = clean[["instrument_id", "timestamp_utc", "factor_value"]].merge(
                        existing_df[["instrument_id", "timestamp_utc", "factor_value"]],
                        on=["instrument_id", "timestamp_utc"],
                        how="inner",
                        suffixes=("_inc", "_ext"),
                    )
                    if not merged.empty:
                        inc_bits = canonical_float_bits(merged["factor_value_inc"].to_numpy())
                        ext_bits = canonical_float_bits(merged["factor_value_ext"].to_numpy())
                        diff_mask = inc_bits != ext_bits
                        if diff_mask.any():
                            bad = merged.loc[diff_mask]
                            examples = [
                                {
                                    "instrument_id": str(row.instrument_id),
                                    "timestamp_utc": pd.Timestamp(row.timestamp_utc).isoformat(),
                                    "existing": None if pd.isna(row.factor_value_ext) else float(row.factor_value_ext),
                                    "incoming": None if pd.isna(row.factor_value_inc) else float(row.factor_value_inc),
                                }
                                for row in bad.head(5).itertuples(index=False)
                            ]
                            raise FactorValueConflictError(fid, fhash, int(diff_mask.sum()), examples)

                        dup_keys = merged[["instrument_id", "timestamp_utc"]].assign(_dup=True)
                        tagged = clean.merge(dup_keys, on=["instrument_id", "timestamp_utc"], how="left")
                        keep_mask = tagged["_dup"].isna().to_numpy()
                        skipped = int((~keep_mask).sum())
                        to_write = clean.loc[keep_mask].copy()

            now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
            if to_write.empty:
                return FactorWriteReceipt(
                    batch_id=batch_id,
                    factor_id=fid,
                    factor_identity_hash=fhash,
                    frequency=freq,
                    session_view=view,
                    inserted=0,
                    skipped=skipped,
                    shards_written=(),
                    created_at=now_iso,
                )

            to_write = to_write.sort_values(
                ["factor_identity_hash", "instrument_id", "timestamp_utc"], kind="mergesort"
            ).reset_index(drop=True)

            staging_dir = self.root / f".staging-{batch_id}-{uuid.uuid4().hex[:8]}"
            staging_dir.mkdir(parents=True, exist_ok=True)
            shards_meta: list[dict[str, Any]] = []
            shards_written: list[str] = []
            replaced_dest_files: list[Path] = []
            committed = False
            try:
                for ym, part_df in to_write.groupby("year_month", sort=True):
                    part_dir_rel = Path(f"factor_id={fid}") / f"frequency={freq}" / f"year_month={ym}"
                    part_name = f"part-{fhash[:12]}-{batch_id}-{uuid.uuid4().hex[:6]}.parquet"
                    staged_file = staging_dir / f"{ym}_{part_name}"
                    dest_dir = self.parquet_root / part_dir_rel
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / part_name
                    if dest_file.exists():
                        raise FileExistsError(f"Refusing to overwrite existing Parquet shard: {dest_file}")

                    leaf_df = part_df.loc[:, list(LEAF_COLUMNS)].copy()
                    leaf_df["timestamp_utc"] = pd.to_datetime(leaf_df["timestamp_utc"], utc=True).dt.as_unit("ns")
                    leaf_df["created_at"] = pd.to_datetime(leaf_df["created_at"], utc=True).dt.as_unit("ns")
                    table = pa.Table.from_pandas(leaf_df, schema=LEAF_ARROW_SCHEMA, preserve_index=False)
                    pq.write_table(
                        table,
                        str(staged_file),
                        compression="zstd",
                        row_group_size=DEFAULT_ROW_GROUP_SIZE,
                    )
                    os.replace(staged_file, dest_file)
                    replaced_dest_files.append(dest_file)
                    rel_path = str(part_dir_rel / part_name)
                    sess_counts = {str(k): int(v) for k, v in part_df.groupby("session_id").size().items()}
                    part_ts_ns = leaf_df["timestamp_utc"].astype("int64")
                    shards_meta.append(
                        {
                            "year_month": str(ym),
                            "part_path": rel_path,
                            "row_count": int(len(part_df)),
                            "sessions": sess_counts,
                            "min_timestamp_ns": int(part_ts_ns.min()),
                            "max_timestamp_ns": int(part_ts_ns.max()),
                        }
                    )
                    shards_written.append(rel_path)

                if fhash not in identities:
                    assert validated_id_rec is not None
                    _append_jsonl_line(self.identities_path, validated_id_rec)

                batch_rec = {
                    "record_type": "batch",
                    "status": "committed",
                    "batch_id": batch_id,
                    "factor_id": fid,
                    "factor_identity_hash": fhash,
                    "frequency": freq,
                    "session_view": view,
                    "row_count": int(len(to_write)),
                    "created_at": now_iso,
                    "shards": shards_meta,
                }
                _append_jsonl_line(self.batches_path, batch_rec)
                committed = True
            finally:
                if not committed:
                    for dest_file in replaced_dest_files:
                        try:
                            if dest_file.exists():
                                dest_file.unlink()
                        except OSError:
                            pass
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)

            return FactorWriteReceipt(
                batch_id=batch_id,
                factor_id=fid,
                factor_identity_hash=fhash,
                frequency=freq,
                session_view=view,
                inserted=int(len(to_write)),
                skipped=skipped,
                shards_written=tuple(shards_written),
                created_at=now_iso,
            )

    def list_identities(
        self,
        factor_id: str | None = None,
        frequency: str | None = None,
    ) -> pd.DataFrame:
        """List registered factor_value_identities from the append-only ledger (lock-free)."""
        if factor_id is not None:
            validate_factor_id(factor_id)
        norm_freq: str | None = None
        if frequency is not None:
            parsed = Frequency.parse(frequency)
            if not parsed.is_valid:
                raise ValueError(f"Invalid frequency: {frequency!r}")
            norm_freq = parsed.canonical

        records, corrupt, _ = _read_jsonl(self.identities_path)
        if corrupt:
            raise ValueError(f"Corrupt identity ledger entries: {corrupt}")
        if not records:
            empty = pd.DataFrame(columns=list(IDENTITY_COLUMNS))
            empty["created_at"] = pd.to_datetime(empty["created_at"], utc=True)
            return empty
        df = pd.DataFrame(records)
        for col in IDENTITY_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df.loc[:, list(IDENTITY_COLUMNS)]
        if factor_id is not None:
            df = df.loc[df["factor_id"] == factor_id]
        if norm_freq is not None:
            df = df.loc[df["frequency"] == norm_freq]
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        return df.sort_values(["created_at", "factor_id", "factor_identity_hash"]).reset_index(drop=True)

    def _committed_shards(self) -> list[dict[str, Any]]:
        _, committed_batches = self._load_committed_state()
        items: list[dict[str, Any]] = []
        for batch in committed_batches:
            for shard in batch.get("shards", []):
                items.append(
                    {
                        "factor_id": str(batch["factor_id"]),
                        "factor_identity_hash": str(batch["factor_identity_hash"]),
                        "frequency": str(batch["frequency"]),
                        "session_view": str(batch["session_view"]),
                        "year_month": str(shard["year_month"]),
                        "part_path": str(shard["part_path"]),
                        "sessions": set(shard.get("sessions", {}).keys()),
                        "row_count": int(shard["row_count"]),
                    }
                )
        return items

    def plan_read(
        self,
        *,
        factor_id: str | Sequence[str] | None = None,
        factor_identity_hash: str | Sequence[str] | None = None,
        frequency: str | Sequence[str] | None = None,
        session_view: str | None = None,
        instruments: Sequence[str] | None = None,
        start_utc: dt.datetime | pd.Timestamp | str | None = None,
        end_utc: dt.datetime | pd.Timestamp | str | None = None,
        session_start: str | None = None,
        session_end: str | None = None,
    ) -> FactorReadPlan:
        """Resolve target partitions and committed Parquet shard paths from the ledger."""
        req_fids: list[str] | None = None
        if factor_id is not None:
            raw_fids = [factor_id] if isinstance(factor_id, str) else list(factor_id)
            req_fids = [validate_factor_id(item) for item in raw_fids]

        req_hashes: list[str] | None = None
        if factor_identity_hash is not None:
            raw_hashes = [factor_identity_hash] if isinstance(factor_identity_hash, str) else list(factor_identity_hash)
            req_hashes = [validate_sha256_hex(item, "factor_identity_hash") for item in raw_hashes]

        req_freqs: list[str] | None = None
        if frequency is not None:
            raw_freqs = [frequency] if isinstance(frequency, str) else list(frequency)
            req_freqs = []
            for item in raw_freqs:
                parsed = Frequency.parse(item)
                if not parsed.is_valid or parsed.canonical is None:
                    raise ValueError(f"Invalid frequency: {item!r}")
                req_freqs.append(parsed.canonical)

        if session_view is not None and session_view not in VALID_VIEWS:
            raise ValueError(f"Invalid session_view: {session_view!r}")

        req_instruments: list[str] | None = None
        if instruments is not None:
            if isinstance(instruments, str):
                req_instruments = [_require_nonempty_str(instruments, "instrument")]
            else:
                req_instruments = [_require_nonempty_str(item, "instrument") for item in instruments]

        parsed_start: pd.Timestamp | None = None
        if start_utc is not None:
            ts = pd.Timestamp(start_utc)
            _utc(ts.to_pydatetime(), "start_utc")
            parsed_start = ts
        parsed_end: pd.Timestamp | None = None
        if end_utc is not None:
            ts = pd.Timestamp(end_utc)
            _utc(ts.to_pydatetime(), "end_utc")
            parsed_end = ts
        if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
            raise ValueError("start_utc must not follow end_utc")

        if session_start is not None:
            _validate_session_date_str(session_start, "session_start")
        if session_end is not None:
            _validate_session_date_str(session_end, "session_end")
        if session_start is not None and session_end is not None and session_start > session_end:
            raise ValueError("session_start must not follow session_end")

        identities, _ = self._load_committed_state()
        shards = self._committed_shards()

        # Resolve partition values from ledger when factor_identity_hash is specified.
        resolved_fids: list[str] = list(req_fids) if req_fids is not None else []
        resolved_freqs: list[str] = list(req_freqs) if req_freqs is not None else []
        if req_hashes is not None:
            inferred_fids: list[str] = []
            inferred_freqs: list[str] = []
            for h in req_hashes:
                rec = identities.get(h)
                if rec is not None:
                    inferred_fids.append(str(rec["factor_id"]))
                    inferred_freqs.append(str(rec.get("output_frequency") or rec["frequency"]))
            if req_fids is None:
                resolved_fids = sorted(set(inferred_fids))
            if req_freqs is None:
                resolved_freqs = sorted(set(inferred_freqs))

        selected: list[dict[str, Any]] = []
        for item in shards:
            if resolved_fids and item["factor_id"] not in resolved_fids:
                continue
            if req_fids is not None and item["factor_id"] not in req_fids:
                continue
            if resolved_freqs and item["frequency"] not in resolved_freqs:
                continue
            if req_freqs is not None and item["frequency"] not in req_freqs:
                continue
            if req_hashes is not None and item["factor_identity_hash"] not in req_hashes:
                continue
            if session_view is not None and item["session_view"] != session_view:
                continue
            if session_start is not None and not any(s >= session_start for s in item["sessions"]):
                continue
            if session_end is not None and not any(s <= session_end for s in item["sessions"]):
                continue
            selected.append(item)

        if req_hashes is not None and not selected:
            # Unknown identity or disjoint filter -> zero shards scanned.
            selected = []

        shard_paths = tuple(self.parquet_root / item["part_path"] for item in selected)
        resolved_ym = tuple(sorted({item["year_month"] for item in selected})) if selected else None
        final_fids = tuple(sorted(set(resolved_fids or [item["factor_id"] for item in selected])))
        final_freqs = tuple(sorted(set(resolved_freqs or [item["frequency"] for item in selected])))

        where_clauses: list[str] = []
        params: list[Any] = []
        if final_fids:
            placeholders = ", ".join("?" for _ in final_fids)
            where_clauses.append(f"factor_id IN ({placeholders})")
            params.extend(final_fids)
        if final_freqs:
            placeholders = ", ".join("?" for _ in final_freqs)
            where_clauses.append(f"frequency IN ({placeholders})")
            params.extend(final_freqs)
        if resolved_ym and (session_start is not None or session_end is not None):
            placeholders = ", ".join("?" for _ in resolved_ym)
            where_clauses.append(f"year_month IN ({placeholders})")
            params.extend(resolved_ym)
        if req_hashes is not None:
            placeholders = ", ".join("?" for _ in req_hashes)
            where_clauses.append(f"factor_identity_hash IN ({placeholders})")
            params.extend(req_hashes)
        if session_view is not None:
            where_clauses.append("session_view = ?")
            params.append(session_view)
        if req_instruments is not None:
            if not req_instruments:
                where_clauses.append("1 = 0")
            else:
                placeholders = ", ".join("?" for _ in req_instruments)
                where_clauses.append(f"instrument_id IN ({placeholders})")
                params.extend(req_instruments)
        if parsed_start is not None:
            where_clauses.append("timestamp_utc >= ?")
            params.append(parsed_start.to_pydatetime())
        if parsed_end is not None:
            where_clauses.append("timestamp_utc <= ?")
            params.append(parsed_end.to_pydatetime())
        if session_start is not None:
            where_clauses.append("session_id >= ?")
            params.append(session_start)
        if session_end is not None:
            where_clauses.append("session_id <= ?")
            params.append(session_end)

        cols_sql = ", ".join(TABLE_COLUMNS)
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = (
            f"SELECT {cols_sql} FROM read_parquet(?, hive_partitioning=true, "
            f"hive_types={{'factor_id': 'VARCHAR', 'frequency': 'VARCHAR', 'year_month': 'VARCHAR'}})"
            f"{where_sql} ORDER BY factor_id, factor_identity_hash, instrument_id, timestamp_utc"
        )
        return FactorReadPlan(
            resolved_factor_ids=final_fids,
            resolved_frequencies=final_freqs,
            resolved_year_months=resolved_ym,
            shard_paths=shard_paths,
            sql=sql,
            params=tuple(params),
        )

    def view_sql(self, view_name: str = "market_factor_values") -> str:
        """Build CREATE OR REPLACE VIEW SQL bound strictly to committed shards."""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", view_name):
            raise ValueError(f"Invalid SQL view name: {view_name!r}")
        shards = self._committed_shards()
        cols_sql = ", ".join(VIEW_COLUMNS)
        if not shards:
            return (
                f'CREATE OR REPLACE VIEW "{view_name}" AS '
                "SELECT CAST(NULL AS VARCHAR) AS factor_id, "
                "CAST(NULL AS VARCHAR) AS factor_identity_hash, "
                "CAST(NULL AS VARCHAR) AS frequency, "
                "CAST(NULL AS VARCHAR) AS session_view, "
                "CAST(NULL AS VARCHAR) AS instrument_id, "
                "CAST(NULL AS TIMESTAMPTZ) AS timestamp_utc, "
                "CAST(NULL AS VARCHAR) AS session_id, "
                "CAST(NULL AS DOUBLE) AS factor_value, "
                "CAST(NULL AS VARCHAR) AS source_build_id, "
                "CAST(NULL AS VARCHAR) AS calendar_hash, "
                "CAST(NULL AS VARCHAR) AS code_version, "
                "CAST(NULL AS TIMESTAMPTZ) AS created_at, "
                "CAST(NULL AS VARCHAR) AS batch_id, "
                "CAST(NULL AS VARCHAR) AS year_month "
                "WHERE 1 = 0"
            )
        quoted_paths = ", ".join(
            "'" + str(self.parquet_root / item["part_path"]).replace("'", "''") + "'"
            for item in shards
        )
        return (
            f'CREATE OR REPLACE VIEW "{view_name}" AS '
            f"SELECT {cols_sql} FROM read_parquet([{quoted_paths}], hive_partitioning=true, "
            "hive_types={'factor_id': 'VARCHAR', 'frequency': 'VARCHAR', 'year_month': 'VARCHAR'})"
        )

    def create_view(self, connection: duckdb.DuckDBPyConnection, view_name: str = "market_factor_values") -> None:
        """Create or refresh a DuckDB view over committed shards only."""
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(self.view_sql(view_name))

    def explain_read(self, **kwargs: Any) -> str:
        """Return DuckDB EXPLAIN output for a planned read query."""
        plan = self.plan_read(**kwargs)
        if not plan.shard_paths:
            return "EMPTY_PLAN (0 committed shards matched)"
        with duckdb.connect(":memory:") as con:
            con.execute("SET TimeZone = 'UTC'")
            rows = con.execute(f"EXPLAIN {plan.sql}", [[str(p) for p in plan.shard_paths], *plan.params]).fetchall()
        return "\n".join(str(row[-1]) for row in rows)

    def read_factor_values(
        self,
        *,
        factor_id: str | Sequence[str] | None = None,
        factor_identity_hash: str | Sequence[str] | None = None,
        frequency: str | Sequence[str] | None = None,
        session_view: str | None = None,
        instruments: Sequence[str] | None = None,
        start_utc: dt.datetime | pd.Timestamp | str | None = None,
        end_utc: dt.datetime | pd.Timestamp | str | None = None,
        session_start: str | None = None,
        session_end: str | None = None,
        connection: duckdb.DuckDBPyConnection | None = None,
    ) -> pd.DataFrame:
        """Read committed factor values via DuckDB read_parquet (lock-free)."""
        plan = self.plan_read(
            factor_id=factor_id,
            factor_identity_hash=factor_identity_hash,
            frequency=frequency,
            session_view=session_view,
            instruments=instruments,
            start_utc=start_utc,
            end_utc=end_utc,
            session_start=session_start,
            session_end=session_end,
        )
        self._last_read_plan = plan
        if not plan.shard_paths:
            empty = pd.DataFrame(columns=list(TABLE_COLUMNS))
            empty["timestamp_utc"] = pd.to_datetime(empty["timestamp_utc"], utc=True)
            empty["created_at"] = pd.to_datetime(empty["created_at"], utc=True)
            empty["factor_value"] = empty["factor_value"].astype(np.float64)
            return empty

        file_list = [str(p) for p in plan.shard_paths]
        if connection is not None:
            connection.execute("SET TimeZone = 'UTC'")
            frame = connection.execute(plan.sql, [file_list, *plan.params]).fetchdf()
        else:
            with duckdb.connect(":memory:") as own_con:
                own_con.execute("SET TimeZone = 'UTC'")
                frame = own_con.execute(plan.sql, [file_list, *plan.params]).fetchdf()

        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.as_unit("ns")
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True).dt.as_unit("ns")
        frame["factor_value"] = frame["factor_value"].astype(np.float64)
        return frame.loc[:, list(TABLE_COLUMNS)].reset_index(drop=True)

    def verify_store(self) -> dict[str, Any]:
        """Read-only consistency check reconciling disk shards and staging with JSONL ledger."""
        if not self.root.exists():
            return {
                "ok": False,
                "root": str(self.root),
                "missing_root": True,
                "identity_count": 0,
                "batch_count": 0,
                "committed_shard_count": 0,
                "total_committed_rows": 0,
                "missing_shards": [],
                "uncommitted_shards": [],
                "staging_leftovers": [],
                "corrupt_ledger_lines": [],
                "row_count_mismatches": [],
                "unregistered_identities": [],
            }

        identities_list, id_corrupt, id_trailing = _read_jsonl(self.identities_path)
        batches_list, batch_corrupt, batch_trailing = _read_jsonl(self.batches_path)
        corrupt_lines = id_corrupt + id_trailing + batch_corrupt + batch_trailing

        identities = {str(rec["factor_identity_hash"]): rec for rec in identities_list if "factor_identity_hash" in rec}
        committed_batches = [rec for rec in batches_list if rec.get("status") == "committed"]

        committed_paths: set[str] = set()
        missing_shards: list[str] = []
        row_count_mismatches: list[dict[str, Any]] = []
        unregistered_identities: list[str] = []
        total_committed_rows = 0

        for batch in committed_batches:
            fhash = str(batch.get("factor_identity_hash", ""))
            if fhash not in identities and fhash not in unregistered_identities:
                unregistered_identities.append(fhash)
            for shard in batch.get("shards", []):
                rel = str(shard["part_path"])
                committed_paths.add(rel)
                expected_rows = int(shard["row_count"])
                total_committed_rows += expected_rows
                full_path = self.parquet_root / rel
                if not full_path.is_file():
                    missing_shards.append(rel)
                else:
                    try:
                        actual_rows = pq.read_metadata(str(full_path)).num_rows
                        if actual_rows != expected_rows:
                            row_count_mismatches.append(
                                {"part_path": rel, "expected_rows": expected_rows, "actual_rows": actual_rows}
                            )
                    except Exception as exc:
                        row_count_mismatches.append(
                            {"part_path": rel, "expected_rows": expected_rows, "error": str(exc)}
                        )

        uncommitted_shards: list[str] = []
        if self.parquet_root.is_dir():
            for path in sorted(self.parquet_root.rglob("*.parquet")):
                rel = str(path.relative_to(self.parquet_root))
                if rel not in committed_paths:
                    uncommitted_shards.append(rel)

        staging_leftovers = sorted(
            str(p.relative_to(self.root))
            for p in self.root.iterdir()
            if p.name.startswith(".staging")
        )

        ok = not (
            corrupt_lines
            or missing_shards
            or uncommitted_shards
            or staging_leftovers
            or row_count_mismatches
            or unregistered_identities
        )
        return {
            "ok": ok,
            "root": str(self.root),
            "missing_root": False,
            "identity_count": len(identities),
            "batch_count": len(committed_batches),
            "committed_shard_count": len(committed_paths),
            "total_committed_rows": total_committed_rows,
            "missing_shards": missing_shards,
            "uncommitted_shards": uncommitted_shards,
            "staging_leftovers": staging_leftovers,
            "corrupt_ledger_lines": corrupt_lines,
            "row_count_mismatches": row_count_mismatches,
            "unregistered_identities": unregistered_identities,
        }


def _factor_source_hashes(factor: Any) -> list[str]:
    from backtest_and_analysis.time_series.factor_bridge import session_grid

    implementations = [cls for cls in type(factor).__mro__ if cls is not object]
    implementations.append(session_grid)
    hashes: list[str] = []
    for impl in implementations:
        source_file = inspect.getsourcefile(impl)
        if not source_file:
            raise ValueError(f"Cannot identify source file for {impl.__qualname__}")
        hashes.append(hashlib.sha256(Path(source_file).read_bytes()).hexdigest())
    try:
        cls_source = inspect.getsource(type(factor))
        hashes.append(hashlib.sha256(cls_source.encode("utf-8")).hexdigest())
    except (OSError, TypeError):
        pass
    return hashes


def _open_factor_db_readonly(factor: Any) -> tuple[duckdb.DuckDBPyConnection | None, bool]:
    conn = getattr(factor, "db_connection", None)
    if isinstance(conn, duckdb.DuckDBPyConnection):
        return conn, False
    duckdb_path = getattr(factor, "duckdb_path", None)
    if duckdb_path is not None and Path(str(duckdb_path)).is_file():
        return duckdb.connect(str(duckdb_path), read_only=True), True
    return None, False


def _resolve_bucket_strategy(
    factor: Any,
    frequency: str,
    view: str,
    calendars: Mapping[str, SessionCalendar],
    explicit_strategy: str | None,
    conn: duckdb.DuckDBPyConnection | None,
) -> str:
    if explicit_strategy is not None:
        strategies = {
            BarSchedule(cal, frequency, view=view, bucket_strategy=explicit_strategy).bucket_strategy
            for cal in calendars.values()
        }
        if len(strategies) != 1:
            raise ValueError(f"Multiple bucket strategies resolved: {sorted(strategies)}")
        return next(iter(strategies))

    cont = factor.market_bars.attrs.get("continuous") if hasattr(factor, "market_bars") else None
    if isinstance(cont, Mapping) and cont.get("bucket_strategy"):
        return str(cont["bucket_strategy"])

    if conn is not None:
        config_obj = factor._get_local_data_config().as_dict() if hasattr(factor, "_get_local_data_config") else None
        catalog = BarCatalog(conn, config=config_obj)
        found: set[str] = set()
        for inst in calendars:
            try:
                entry = catalog.resolve(frequency, inst)
            except LookupError:
                continue
            if entry.bucket_strategy:
                found.add(entry.bucket_strategy)
        if len(found) == 1:
            strat = next(iter(found))
            for cal in calendars.values():
                BarSchedule(cal, frequency, view=view, bucket_strategy=strat)
            return strat

    strategies = {
        BarSchedule(cal, frequency, view=view, bucket_strategy=None).bucket_strategy
        for cal in calendars.values()
    }
    if len(strategies) != 1:
        raise ValueError(f"Multiple bucket strategies resolved: {sorted(strategies)}")
    return next(iter(strategies))


def _resolve_build_ids(
    factor: Any,
    frequency: str,
    required_pairs: set[tuple[str, str]],
    explicit_build_ids: str | Mapping[str, Any] | None,
    conn: duckdb.DuckDBPyConnection | None,
) -> dict[tuple[str, str], str]:
    resolved: dict[tuple[str, str], str] = {}

    def _ingest_mapping(mapping: Mapping[str, Any]) -> None:
        for k, v in mapping.items():
            if isinstance(k, tuple) and len(k) == 2 and isinstance(v, str) and v.strip():
                resolved[(str(k[0]), str(k[1]))] = v
            elif isinstance(k, str) and isinstance(v, Mapping):
                for sid, bid in v.items():
                    if isinstance(bid, str) and bid.strip():
                        resolved[(k, str(sid))] = bid
            elif isinstance(k, str) and isinstance(v, str) and v.strip():
                for inst, sid in required_pairs:
                    if inst == k:
                        resolved.setdefault((inst, sid), v)

    # 1. Explicit caller override
    if isinstance(explicit_build_ids, str):
        bid = _require_nonempty_str(explicit_build_ids, "source_build_ids")
        for pair in required_pairs:
            resolved[pair] = bid
    elif isinstance(explicit_build_ids, Mapping):
        _ingest_mapping(explicit_build_ids)

    # 2. Check market_bars.attrs['build_ids'] if present
    missing = required_pairs - set(resolved)
    if missing and hasattr(factor, "market_bars"):
        attrs_bids = factor.market_bars.attrs.get("build_ids")
        if isinstance(attrs_bids, Mapping):
            _ingest_mapping(attrs_bids)

    # 3. Actively query bar_build_log / bar_catalog from factor's DB (never assume attrs has it)
    missing = required_pairs - set(resolved)
    if missing and conn is not None:
        instruments = sorted({inst for inst, _ in missing})
        if table_exists(conn, "bar_build_log"):
            placeholders = ", ".join("?" for _ in instruments)
            rows = conn.execute(
                f"SELECT instrument_id, session_id, build_id FROM bar_build_log "
                f"WHERE frequency = ? AND instrument_id IN ({placeholders})",
                [frequency, *instruments],
            ).fetchall()
            for inst, sid, bid in rows:
                if isinstance(bid, str) and bid.strip():
                    resolved.setdefault((str(inst), str(sid)), bid)

        missing = required_pairs - set(resolved)
        if missing and table_exists(conn, "bar_catalog"):
            placeholders = ", ".join("?" for _ in instruments)
            rows = conn.execute(
                f"SELECT instrument_id, build_id FROM bar_catalog "
                f"WHERE frequency = ? AND (instrument_id IN ({placeholders}) OR instrument_id = '*') "
                f"AND build_id IS NOT NULL",
                [frequency, *instruments],
            ).fetchall()
            by_inst = {str(inst): str(bid) for inst, bid in rows if isinstance(bid, str) and bid.strip()}
            for inst, sid in list(missing):
                bid = by_inst.get(inst) or by_inst.get("*")
                if bid:
                    resolved[(inst, sid)] = bid

    missing = sorted(required_pairs - set(resolved))
    if missing:
        raise ValueError(
            "Cannot resolve non-empty source_build_id for (instrument_id, session_id) pairs "
            f"{missing[:5]} (total missing={len(missing)}): neither attrs['build_ids'], "
            "bar_build_log/bar_catalog, nor explicit source_build_ids provided a build_id"
        )
    return resolved


def save_bar_factor_values(
    factor: Any,
    store: MarketFactorValueStore,
    *,
    factor_id: str | None = None,
    bucket_strategy: str | None = None,
    source_build_ids: str | Mapping[str, Any] | None = None,
    identity_scope: str = "exact",
    batch_id: str | None = None,
) -> FactorWriteReceipt:
    """Persist a minute FactorBase output to MarketFactorValueStore without mutating FactorBase."""
    if identity_scope not in ("exact", "incremental"):
        raise ValueError("identity_scope must be 'exact' or 'incremental'")
    if not isinstance(store, MarketFactorValueStore):
        raise TypeError("store must be a MarketFactorValueStore instance")

    values = factor.get_bar_factor_frame()
    if not hasattr(factor, "market_bars") or not hasattr(factor, "market_calendars") or not hasattr(factor, "market_bar_grid"):
        raise ValueError("factor must populate market_bars, market_bar_grid, and market_calendars")

    if "_get_cache_inputs" not in vars(type(factor)):
        raise ValueError(f"{type(factor).__name__} must declare _get_cache_inputs() for factor_identity_hash")
    params = factor._get_cache_inputs()
    if params is None:
        raise ValueError(f"{type(factor).__name__}._get_cache_inputs() returned None; cannot build factor_identity_hash")

    fid = validate_factor_id(factor_id or getattr(factor, "factor_name", ""))
    raw_out_freq = getattr(factor, "output_frequency", None) or factor.data_frequency
    freq = Frequency.parse(raw_out_freq).canonical
    in_freq = Frequency.parse(factor.data_frequency).canonical
    if not freq or not in_freq:
        raise ValueError(f"Invalid factor frequency: {factor.data_frequency!r}")
    view = str(factor.session_view)
    analysis_mode = str(factor.analysis_mode)
    calendars: dict[str, SessionCalendar] = dict(factor.market_calendars)

    grid_keys = factor.market_bar_grid[["timestamp_utc", "instrument_id", "session_id"]].drop_duplicates()
    merged = values.merge(grid_keys, on=["timestamp_utc", "instrument_id"], how="left", validate="one_to_one")
    if merged["session_id"].isna().any():
        raise ValueError("Unable to resolve session_id for all rows in factor output")

    required_pairs = {(str(r.instrument_id), str(r.session_id)) for r in merged[["instrument_id", "session_id"]].drop_duplicates().itertuples(index=False)}

    conn, should_close = _open_factor_db_readonly(factor)
    try:
        resolved_strategy = _resolve_bucket_strategy(factor, in_freq, view, calendars, bucket_strategy, conn)
        build_map = _resolve_build_ids(factor, in_freq, required_pairs, source_build_ids, conn)

        catalog_meta: dict[str, Any] = {}
        if conn is not None:
            config_obj = factor._get_local_data_config().as_dict() if hasattr(factor, "_get_local_data_config") else None
            catalog = BarCatalog(conn, config=config_obj)
            for inst in sorted(calendars):
                try:
                    entry = catalog.resolve(in_freq, inst)
                except LookupError:
                    continue
                catalog_meta[inst] = {
                    "dataset": entry.dataset,
                    "table_name": entry.table_name,
                    "kind": entry.kind,
                    "source_frequency": entry.source_frequency,
                    "source_version": entry.source_version,
                }
    finally:
        if should_close and conn is not None:
            conn.close()

    cal_hashes = {inst: cal.calendar_hash for inst, cal in sorted(calendars.items())}
    rth_refs = {inst: cal.rth_reference_calendar for inst, cal in sorted(calendars.items())}

    cont_attrs = factor.market_bars.attrs.get("continuous")
    roll_bids = factor.market_bars.attrs.get("roll_build_ids")
    continuous_adjust = str(cont_attrs["adjust"]) if isinstance(cont_attrs, Mapping) and cont_attrs.get("adjust") else None

    source_identity: dict[str, Any] = {}
    if catalog_meta:
        source_identity["catalog"] = catalog_meta
    if isinstance(cont_attrs, Mapping):
        source_identity["continuous"] = {
            "product_root": cont_attrs.get("product_root"),
            "adjust": continuous_adjust,
            "frequency": cont_attrs.get("frequency"),
        }
        if roll_bids:
            source_identity["continuous"]["roll_build_ids"] = dict(sorted(roll_bids.items()))

    explicit_data_version = getattr(factor, "data_version", None)
    row_data_version = explicit_data_version
    if row_data_version is None and "data_version" in factor.market_bars.columns and not factor.market_bars.empty:
        versions = sorted(set(factor.market_bars["data_version"].dropna().astype(str)))
        if versions:
            row_data_version = ",".join(versions)

    if identity_scope == "exact":
        nested_bids: dict[str, dict[str, str]] = {}
        for (inst, sid), bid in sorted(build_map.items()):
            nested_bids.setdefault(inst, {})[sid] = bid
        id_source_build: str | Mapping[str, Any] | None = nested_bids
        id_data_version = row_data_version
    else:
        id_source_build = None
        id_data_version = explicit_data_version
        stable_catalog = {
            inst: {
                "dataset": meta["dataset"],
                "table_name": meta["table_name"],
                "kind": meta["kind"],
                "source_frequency": meta["source_frequency"],
            }
            for inst, meta in catalog_meta.items()
        }
        source_identity.pop("catalog", None)
        source_identity["source_dataset_version"] = {
            "data_version": explicit_data_version,
            "catalog": stable_catalog,
        }

    code_ver, code_ver_src = resolve_code_version()
    fhash, payload = compute_factor_identity_hash(
        factor_id=fid,
        factor_class=f"{type(factor).__module__}.{type(factor).__qualname__}",
        source_sha256=_factor_source_hashes(factor),
        params=params,
        frequency=in_freq,
        output_frequency=freq,
        session_view=view,
        analysis_mode=analysis_mode,
        bucket_strategy=resolved_strategy,
        calendar_hash=cal_hashes,
        source_identity=source_identity,
        source_build_id=id_source_build,
        data_version=id_data_version,
        rth_reference_calendar=rth_refs,
        continuous_adjust=continuous_adjust,
        roll_build_ids=roll_bids if isinstance(roll_bids, Mapping) else None,
        code_version=code_ver,
    )

    now_dt = dt.datetime.now(dt.timezone.utc)
    now_iso = now_dt.isoformat()
    bid_str = batch_id or uuid.uuid4().hex
    identity_rec = build_identity_record(
        factor_identity_hash=fhash,
        payload=payload,
        code_version_source=code_ver_src,
        created_at=now_iso,
    )

    merged["factor_id"] = fid
    merged["factor_identity_hash"] = fhash
    merged["frequency"] = freq
    merged["session_view"] = view
    merged["source_build_id"] = [
        build_map[(str(r.instrument_id), str(r.session_id))]
        for r in merged[["instrument_id", "session_id"]].itertuples(index=False)
    ]
    merged["calendar_hash"] = [cal_hashes[str(inst)] for inst in merged["instrument_id"]]
    merged["code_version"] = code_ver
    merged["created_at"] = pd.Timestamp(now_dt)
    merged["batch_id"] = bid_str

    return store.append_factor_values(
        merged.loc[:, list(TABLE_COLUMNS)],
        identity_record=identity_rec,
        calendar=calendars,
        strict_source_build_id=True,
    )
