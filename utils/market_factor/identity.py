"""Factor identity hashing and incoming batch validation."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from utils.code_version import resolve_code_version
from utils.data_revision import FactorFrameDigest
from utils.market_db.identity import MARKET_DB_FIELDS, market_db_identity
from utils.time_axis import Frequency

from .schema import (
    IDENTITY_COLUMNS,
    MARKET_FACTOR_IDENTITY_SCHEMA_VERSION,
    SUPPORTED_IDENTITY_SCHEMA_VERSIONS,
    _V2_IDENTITY_COLUMNS,
    VALID_ADJUSTMENTS,
    VALID_ANALYSIS_MODES,
    VALID_BUCKET_STRATEGIES,
    VALID_VIEWS,
    _canonical_json,
    _frame_digest,
    _require_nonempty_str,
    validate_factor_id,
    validate_sha256_hex,
)

_LEGACY_SLOT_ID = "legacy_bar"
_LEGACY_SLOT_CONFIG_HASH = hashlib.sha256(b"market_factor_legacy_bar.v1").hexdigest()
_ADJUSTMENT_ALIASES = {"ratio_back": "ratio_back_adjust", "panama_canal": "diff_back_adjust"}


def _normalize_slot_fields(
    slot_id: str | None,
    slot_config_hash: str | None,
    adjustment_policy: str | None,
    slot_data_fingerprint: str | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if any(value is None for value in (slot_id, slot_config_hash, adjustment_policy, slot_data_fingerprint)):
        raise ValueError("v2 identity requires slot_id, slot_config_hash, adjustment_policy and slot_data_fingerprint")
    slot_id = _require_nonempty_str(slot_id, "slot_id")
    if slot_id != slot_id.strip():
        raise ValueError("slot_id must have no surrounding whitespace")
    slot_config_hash = validate_sha256_hex(slot_config_hash, "slot_config_hash")
    policy = _require_nonempty_str(adjustment_policy, "adjustment_policy")
    policy = _ADJUSTMENT_ALIASES.get(policy, policy)
    if policy not in (*VALID_ADJUSTMENTS, "dynamic_pit"):
        choices = (*VALID_ADJUSTMENTS, "dynamic_pit", *_ADJUSTMENT_ALIASES)
        raise ValueError(f"adjustment_policy must be one of {choices}, got {adjustment_policy!r}")
    if isinstance(slot_data_fingerprint, str):
        fingerprint = _require_nonempty_str(slot_data_fingerprint, "slot_data_fingerprint")
    elif isinstance(slot_data_fingerprint, Mapping) and slot_data_fingerprint:
        fingerprint = json.loads(_canonical_json(dict(slot_data_fingerprint)))
        if fingerprint != slot_data_fingerprint:
            raise ValueError("slot_data_fingerprint mapping must use strict JSON types and string keys")
    else:
        raise ValueError("slot_data_fingerprint must be a non-empty string or JSON mapping")
    return dict(zip(_V2_IDENTITY_COLUMNS, (slot_id, slot_config_hash, policy, fingerprint))) | market_db_identity(fingerprint)


def _normalize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(params, Mapping) or any(not isinstance(k, str) for k in params):
        raise TypeError("params must be a mapping with string keys")
    normalized: dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        if isinstance(value, pd.DataFrame):
            normalized[key] = ["dataframe", _frame_digest(value)]
        elif isinstance(value, FactorFrameDigest):
            normalized[key] = ["dataframe", validate_sha256_hex(value.sha256, f"params[{key!r}]")]
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
    slot_id: str | None = None,
    slot_config_hash: str | None = None,
    adjustment_policy: str | None = None,
    slot_data_fingerprint: str | Mapping[str, Any] | None = None,
    schema_version: str = MARKET_FACTOR_IDENTITY_SCHEMA_VERSION,
) -> tuple[str, dict[str, Any]]:
    """Hash a canonical identity. Source-only callers use a reserved v2 legacy namespace."""
    if schema_version not in SUPPORTED_IDENTITY_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported identity payload schema_version: {schema_version!r}")
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
    slot_fields = (slot_id, slot_config_hash, adjustment_policy, slot_data_fingerprint)
    explicit_slot = any(value is not None for value in slot_fields)
    is_v2 = schema_version == MARKET_FACTOR_IDENTITY_SCHEMA_VERSION
    if not is_v2 and explicit_slot:
        raise ValueError("v1 historical identities do not carry slot metadata")
    if not has_source_build and (not is_v2 or not explicit_slot):
        raise ValueError(
            "Supply v2 slot metadata or a legacy source_build_id "
            "(or source_identity with source_build_id/source_build_ids/source_dataset_version)"
        )
    if is_v2 and not explicit_slot:
        legacy_continuous = src_id.get("continuous")
        legacy_adjustment = legacy_continuous.get("adjust") if isinstance(legacy_continuous, Mapping) else None
        slot_fields = (_LEGACY_SLOT_ID, _LEGACY_SLOT_CONFIG_HASH, legacy_adjustment or "none", src_id)
    if explicit_slot and slot_id == _LEGACY_SLOT_ID:
        raise ValueError(f"slot_id {_LEGACY_SLOT_ID!r} is reserved for legacy source-only callers")

    resolved_cv = _require_nonempty_str(code_version, "code_version") if code_version is not None else resolve_code_version()[0]

    payload: dict[str, Any] = {
        "schema_version": schema_version,
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
    if is_v2:
        payload.update(_normalize_slot_fields(*slot_fields))

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
    record = {
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
        "source_schema_version": payload["schema_version"],
        "identity_payload_json": _canonical_json(payload),
        "created_at": now_iso,
    }
    record.update({field: payload[field] for field in (*_V2_IDENTITY_COLUMNS, *MARKET_DB_FIELDS)
                   if field in payload})
    return _validate_identity_record(record)


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
    for_write: bool = False,
) -> dict[str, Any]:
    """Validate an identity_record and verify canonical JSON SHA256 against factor_identity_hash."""
    if not isinstance(identity_record, Mapping):
        raise TypeError("identity_record must be a mapping")
    schema_ver = identity_record.get("source_schema_version")
    optional = MARKET_DB_FIELDS if schema_ver == MARKET_FACTOR_IDENTITY_SCHEMA_VERSION else (
        *MARKET_DB_FIELDS, *_V2_IDENTITY_COLUMNS,
    )
    columns = (col for col in IDENTITY_COLUMNS if col not in optional)
    missing_cols = [col for col in columns if col not in identity_record]
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
    if schema_ver not in SUPPORTED_IDENTITY_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported identity payload schema_version: {schema_ver!r}")
    if for_write and schema_ver != MARKET_FACTOR_IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"{schema_ver} identities are read-only; supply a v2 identity")
    if str(identity_record["source_schema_version"]) != schema_ver:
        raise ValueError("identity_record.source_schema_version does not match payload schema_version")
    if schema_ver == MARKET_FACTOR_IDENTITY_SCHEMA_VERSION:
        slot_fields = _normalize_slot_fields(*(payload_obj.get(field) for field in _V2_IDENTITY_COLUMNS))
        for field, value in slot_fields.items():
            if payload_obj.get(field) != value or identity_record.get(field) != value:
                raise ValueError(f"identity_record.{field} must match canonical payload {field}")
        if market_db_identity(payload_obj) != market_db_identity(identity_record) or any(
            field not in slot_fields and (field in payload_obj or field in identity_record)
            for field in MARKET_DB_FIELDS
        ):
            raise ValueError("Market database identity must match slot_data_fingerprint")
    elif any(field in identity_record or field in payload_obj for field in (*_V2_IDENTITY_COLUMNS, *MARKET_DB_FIELDS)):
        raise ValueError("v1 historical identities do not carry slot metadata")
    _normalize_calendar_mapping(payload_obj["calendar_hash"])

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
