"""Factor identity hashing and incoming batch validation."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from backtest_and_analysis.time_series.contracts import DuplicateKeyError
from utils.code_version import resolve_code_version
from utils.session_calendar import SessionCalendar
from utils.time_axis import Frequency

from .schema import (
    IDENTITY_COLUMNS,
    MARKET_FACTOR_IDENTITY_SCHEMA_VERSION,
    TABLE_COLUMNS,
    VALID_ADJUSTMENTS,
    VALID_ANALYSIS_MODES,
    VALID_BUCKET_STRATEGIES,
    VALID_VIEWS,
    _canonical_json,
    _frame_digest,
    _require_nonempty_str,
    _validate_session_date_str,
    canonical_float_bits,
    validate_factor_id,
    validate_sha256_hex,
)


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
