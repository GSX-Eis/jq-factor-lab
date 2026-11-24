"""Shared schema, compatibility, and normalization helpers for targets/signal_export."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from factor_synthesis.core.calendar_skip import parse_skip_rules
from factor_synthesis.strategy_bundle import StrategyBundle, sha256_file
from targets.contracts import (
    TARGET_ARTIFACT_INTEGRITY_FAILED,
    TARGET_BUNDLE_INVALID,
    TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
    TARGET_SIGNAL_SOURCE_MISSING,
    CompatibilityIssue,
    CompatibilityReport,
    DeployTargetError,
    StrategyBundleSource,
    resolve_contained_file,
    resolve_verified_bundle,
)
from utils.slots.contracts import ExecutionProfile
from utils.time_axis import Frequency

SIGNAL_EXPORT_TARGET_ID = "signal_export"
MAX_INT64 = (1 << 63) - 1
VALID_EXPORT_FORMATS = ("parquet", "jsonl")
SIGNAL_POSITION_ARTIFACT_TYPES = frozenset(
    {"target_positions", "target_positions_parquet", "target_positions_jsonl"}
)
VALID_REBALANCE_FREQS = frozenset({"bar", "daily", "weekly", "monthly"})
VALID_REBALANCE_SCHEDULES = frozenset(
    {"daily", "weekly", "monthly", "quarterly", "report_based"}
)
VALID_SELECTION_MODES = frozenset({"top_k", "top_pct"})

POSITION_REQUIRED_COLUMNS = (
    "timestamp_utc",
    "session_id",
    "instrument_id",
    "target_weight",
)
POSITION_OPTIONAL_COLUMNS = (
    "signal_score",
    "reference_price",
    "lot_size",
    "contract_multiplier",
)
POSITION_BASE_OUTPUT_COLUMNS = (
    "timestamp_utc",
    "session_id",
    "instrument_id",
    "target_weight",
    "side",
    "slot_id",
    "strategy_id",
    "frequency",
    "bundle_hash",
)
ORDER_OUTPUT_COLUMNS = (
    "timestamp_utc",
    "session_id",
    "instrument_id",
    "prev_weight",
    "target_weight",
    "delta_weight",
    "action",
    "slot_id",
    "strategy_id",
    "bundle_hash",
)
VALID_SIDES = frozenset({"LONG", "SHORT", "FLAT"})
VALID_ACTIONS = frozenset({"BUY", "SELL", "SHORT", "COVER", "HOLD"})


def _reject_json_constant(token: str) -> Any:
    raise DeployTargetError(
        f"Non-standard JSON constant {token!r} is not permitted in exported JSONL",
        code=TARGET_ARTIFACT_INTEGRITY_FAILED,
    )


def read_jsonl_records(
    path: Path,
    *,
    label: str,
    error_code: str = TARGET_ARTIFACT_INTEGRITY_FAILED,
) -> list[dict[str, Any]]:
    """Read and parse a strict JSONL file into a list of JSON object dicts."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DeployTargetError(
            f"Failed to read {label}: {exc}",
            code=error_code,
        ) from exc
    records: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            raise DeployTargetError(
                f"{label} line {idx + 1} is unexpectedly blank",
                code=error_code,
            )
        try:
            obj = json.loads(line, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise DeployTargetError(
                f"{label} line {idx + 1} invalid JSON: {exc}",
                code=error_code,
            ) from exc
        if not isinstance(obj, dict):
            raise DeployTargetError(
                f"{label} line {idx + 1} must be a JSON object",
                code=error_code,
            )
        records.append(obj)
    return records


def read_parquet_records(
    path: Path,
    *,
    label: str,
    error_code: str = TARGET_ARTIFACT_INTEGRITY_FAILED,
) -> list[dict[str, Any]]:
    """Read a Parquet file fail-closed and convert rows to a list of dicts."""
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise DeployTargetError(
            f"Failed to read/decode {label}: {exc}",
            code=error_code,
        ) from exc
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise DeployTargetError(
            f"{label} must decode to a DataFrame with unique column names",
            code=error_code,
        )
    return frame.to_dict(orient="records")


def validate_unique_session_per_timestamp(
    pairs: Sequence[tuple[pd.Timestamp, str]],
    *,
    source_name: str,
    code: str,
) -> None:
    """Verify every timestamp_utc maps consistently to a single session_id."""
    seen: dict[pd.Timestamp, str] = {}
    for ts, session_id in pairs:
        prev = seen.get(ts)
        if prev is None:
            seen[ts] = session_id
        elif prev != session_id:
            raise DeployTargetError(
                f"{source_name} has inconsistent session_id values at "
                f"timestamp_utc={format_utc_timestamp(ts)}: {prev!r} vs {session_id!r}",
                code=code,
            )


def load_bundle_signal_positions(
    bundle: StrategyBundle,
    bundle_dir: Path | None,
) -> tuple[list[dict[str, Any]], str] | None:
    """Resolve and load a directory-bound target_positions artifact from StrategyBundle.model_artifacts."""
    signal_refs = [
        ref for ref in bundle.model_artifacts if ref.artifact_type in SIGNAL_POSITION_ARTIFACT_TYPES
    ]
    if not signal_refs:
        return None
    if len(signal_refs) > 1:
        raise DeployTargetError(
            f"StrategyBundle binds multiple ambiguous target_positions artifacts: "
            f"{[ref.path for ref in signal_refs]}",
            code=TARGET_BUNDLE_INVALID,
        )
    if bundle_dir is None:
        raise DeployTargetError(
            "Verified bundle_dir is required to load directory-bound target_positions artifact",
            code=TARGET_BUNDLE_INVALID,
        )

    signal_ref = signal_refs[0]
    resolved = resolve_contained_file(
        bundle_dir,
        signal_ref.path,
        f"model_artifacts[{signal_ref.path!r}]",
    )
    actual_sha = sha256_file(resolved)
    if actual_sha != signal_ref.sha256:
        raise DeployTargetError(
            f"Bundle signal artifact {signal_ref.path!r} sha256 mismatch: "
            f"expected {signal_ref.sha256}, got {actual_sha}",
            code=TARGET_BUNDLE_INVALID,
        )
    if signal_ref.size_bytes is not None and resolved.stat().st_size != signal_ref.size_bytes:
        raise DeployTargetError(
            f"Bundle signal artifact {signal_ref.path!r} size mismatch",
            code=TARGET_BUNDLE_INVALID,
        )

    suffix = resolved.suffix.lower()
    if signal_ref.artifact_type == "target_positions_parquet":
        if suffix != ".parquet":
            raise DeployTargetError(
                f"target_positions_parquet artifact must have .parquet suffix, got {signal_ref.path!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        records = read_parquet_records(resolved, label=signal_ref.path)
    elif signal_ref.artifact_type == "target_positions_jsonl":
        if suffix != ".jsonl":
            raise DeployTargetError(
                f"target_positions_jsonl artifact must have .jsonl suffix, got {signal_ref.path!r}",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )
        records = read_jsonl_records(resolved, label=signal_ref.path)
    else:
        if suffix == ".parquet":
            records = read_parquet_records(resolved, label=signal_ref.path)
        elif suffix == ".jsonl":
            records = read_jsonl_records(resolved, label=signal_ref.path)
        else:
            raise DeployTargetError(
                f"Unsupported signal artifact extension for {signal_ref.path!r}; expected .parquet or .jsonl",
                code=TARGET_ARTIFACT_INTEGRITY_FAILED,
            )

    if not records:
        raise DeployTargetError(
            f"Bundle signal artifact {signal_ref.path!r} contains no position rows",
            code=TARGET_SIGNAL_SOURCE_MISSING,
        )

    expected_meta = (
        ("slot_id", bundle.slot_id),
        ("strategy_id", bundle.strategy_id),
        ("frequency", bundle.frequency),
        ("bundle_hash", bundle.bundle_hash),
    )
    for idx, rec in enumerate(records):
        for meta_key, expected_val in expected_meta:
            if meta_key in rec and rec[meta_key] is not None and rec[meta_key] != expected_val:
                raise DeployTargetError(
                    f"Bundle signal artifact {signal_ref.path!r}[{idx}].{meta_key}={rec[meta_key]!r} "
                    f"does not match bundle {expected_val!r}",
                    code=TARGET_BUNDLE_INVALID,
                )

    return records, signal_ref.path


def require_clean_str(
    value: Any,
    field_name: str,
    *,
    code: str = TARGET_BUNDLE_INVALID,
) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DeployTargetError(
            f"{field_name} must be a nonempty string without surrounding whitespace",
            code=code,
        )
    return value


def require_finite_float(
    value: Any,
    field_name: str,
    *,
    code: str = TARGET_BUNDLE_INVALID,
) -> float:
    if type(value) is bool:
        raise DeployTargetError(f"{field_name} must be a finite number, not bool", code=code)
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        value = value.item()
    if type(value) is bool or not isinstance(value, (int, float)):
        raise DeployTargetError(
            f"{field_name} must be a finite numeric value, got {type(value).__name__}",
            code=code,
        )
    try:
        num = float(value)
    except OverflowError as exc:
        raise DeployTargetError(
            f"{field_name} exceeds finite float range",
            code=code,
        ) from exc
    if not math.isfinite(num):
        raise DeployTargetError(f"{field_name} must be finite, got {value!r}", code=code)
    return 0.0 if num == 0.0 else num


def parse_utc_timestamp(
    value: Any,
    *,
    field_name: str = "timestamp_utc",
    code: str = TARGET_BUNDLE_INVALID,
) -> pd.Timestamp:
    if value is None or isinstance(value, (bool, int, float)):
        raise DeployTargetError(
            f"{field_name} must be a timezone-aware UTC timestamp",
            code=code,
        )
    if isinstance(value, str):
        if not value.strip():
            raise DeployTargetError(f"{field_name} must not be blank", code=code)
    elif not isinstance(value, (pd.Timestamp, datetime)):
        raise DeployTargetError(
            f"{field_name} has unsupported type {type(value).__name__}",
            code=code,
        )
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise DeployTargetError(
            f"{field_name} is not a valid timestamp: {value!r}",
            code=code,
        ) from exc
    if pd.isna(ts) or ts.tzinfo is None:
        raise DeployTargetError(
            f"{field_name} must be timezone-aware UTC: {value!r}",
            code=code,
        )
    return ts.tz_convert("UTC")


def format_utc_timestamp(ts: pd.Timestamp) -> str:
    utc_ts = ts.tz_convert("UTC") if ts.tzinfo is not None else parse_utc_timestamp(ts)
    return utc_ts.isoformat().replace("+00:00", "Z")


def derive_side(weight: float) -> str:
    if weight > 0.0:
        return "LONG"
    if weight < 0.0:
        return "SHORT"
    return "FLAT"


def derive_order_action(prev_weight: float, target_weight: float) -> str:
    delta = target_weight - prev_weight
    if delta == 0.0:
        return "HOLD"
    if delta > 0.0:
        return "COVER" if target_weight <= 0.0 else "BUY"
    return "SELL" if target_weight >= 0.0 else "SHORT"


def normalize_formats(formats: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(formats, (str, bytes)) or not isinstance(formats, Sequence):
        raise DeployTargetError(
            "formats must be a nonempty sequence of 'parquet' / 'jsonl'",
            code=TARGET_BUNDLE_INVALID,
        )
    out: list[str] = []
    for fmt in formats:
        clean = require_clean_str(fmt, "formats[]", code=TARGET_BUNDLE_INVALID).lower()
        if clean not in VALID_EXPORT_FORMATS:
            raise DeployTargetError(
                f"Unsupported export format {fmt!r}; expected subset of {VALID_EXPORT_FORMATS}",
                code=TARGET_BUNDLE_INVALID,
            )
        if clean in out:
            raise DeployTargetError(
                f"Duplicate export format {fmt!r} in formats",
                code=TARGET_BUNDLE_INVALID,
            )
        out.append(clean)
    if not out:
        raise DeployTargetError(
            "formats must contain at least one supported format",
            code=TARGET_BUNDLE_INVALID,
        )
    return tuple(out)


def _records_to_dataframe(
    data: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    name: str,
    missing_error_code: str,
) -> pd.DataFrame:
    if data is None:
        raise DeployTargetError(f"{name} is required", code=missing_error_code)
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise DeployTargetError(f"{name} must not be empty", code=missing_error_code)
        if not data.columns.is_unique:
            raise DeployTargetError(
                f"{name} DataFrame must have unique column names",
                code=TARGET_BUNDLE_INVALID,
            )
        return data.copy()
    if isinstance(data, (str, bytes, Mapping)) or not isinstance(data, Sequence):
        raise DeployTargetError(
            f"{name} must be a pandas DataFrame or sequence of record mappings",
            code=missing_error_code,
        )
    if len(data) == 0:
        raise DeployTargetError(f"{name} must not be empty", code=missing_error_code)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise DeployTargetError(
                f"{name}[{idx}] must be a mapping",
                code=TARGET_BUNDLE_INVALID,
            )
        rows.append(dict(item))
    return pd.DataFrame(rows, dtype=object)


def normalize_target_positions(
    target_positions: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    *,
    allow_short: bool,
    missing_error_code: str = TARGET_SIGNAL_SOURCE_MISSING,
) -> pd.DataFrame:
    """Validate, normalize to UTC, sort deterministically, and reject NaN/Inf/duplicates."""
    frame = _records_to_dataframe(
        target_positions,
        name="target_positions",
        missing_error_code=missing_error_code,
    )
    missing_cols = [col for col in POSITION_REQUIRED_COLUMNS if col not in frame.columns]
    if missing_cols:
        raise DeployTargetError(
            f"target_positions missing required columns: {missing_cols}",
            code=TARGET_BUNDLE_INVALID,
        )

    present_optional = [col for col in POSITION_OPTIONAL_COLUMNS if col in frame.columns]

    ts_col = frame["timestamp_utc"]
    if isinstance(ts_col.dtype, pd.DatetimeTZDtype):
        if ts_col.isna().any():
            raise DeployTargetError(
                "target_positions.timestamp_utc must not contain null values",
                code=TARGET_BUNDLE_INVALID,
            )
        normalized_ts = ts_col.dt.tz_convert("UTC")
    elif pd.api.types.is_datetime64_any_dtype(ts_col.dtype):
        raise DeployTargetError(
            "target_positions.timestamp_utc must be timezone-aware UTC",
            code=TARGET_BUNDLE_INVALID,
        )
    else:
        parsed_list = [
            parse_utc_timestamp(val, field_name=f"target_positions[{idx}].timestamp_utc")
            for idx, val in enumerate(ts_col.tolist())
        ]
        normalized_ts = pd.Series(
            pd.to_datetime(parsed_list, utc=True),
            index=frame.index,
            name="timestamp_utc",
        )

    sessions = [
        require_clean_str(val, f"target_positions[{idx}].session_id")
        for idx, val in enumerate(frame["session_id"].tolist())
    ]
    instruments = [
        require_clean_str(val, f"target_positions[{idx}].instrument_id")
        for idx, val in enumerate(frame["instrument_id"].tolist())
    ]
    weights = [
        require_finite_float(val, f"target_positions[{idx}].target_weight")
        for idx, val in enumerate(frame["target_weight"].tolist())
    ]

    if not allow_short:
        for idx, weight in enumerate(weights):
            if weight < 0.0:
                raise DeployTargetError(
                    f"Negative target_weight {weight} at row {idx} ({instruments[idx]}) is forbidden when allow_short=False",
                    code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
                )

    out_data: dict[str, Any] = {
        "timestamp_utc": normalized_ts.tolist(),
        "session_id": sessions,
        "instrument_id": instruments,
        "target_weight": weights,
    }

    for opt_col in present_optional:
        raw_vals = frame[opt_col].tolist()
        if opt_col == "lot_size":
            clean_lots: list[int] = []
            for idx, val in enumerate(raw_vals):
                if type(val) is bool or (
                    hasattr(val, "item") and not isinstance(val, (bytes, bytearray))
                    and type(val.item()) is bool
                ):
                    raise DeployTargetError(
                        f"target_positions[{idx}].lot_size must be an integer >= 1",
                        code=TARGET_BUNDLE_INVALID,
                    )
                if hasattr(val, "item") and not isinstance(val, (bytes, bytearray)):
                    val = val.item()
                if isinstance(val, float):
                    if not math.isfinite(val) or not val.is_integer():
                        raise DeployTargetError(
                            f"target_positions[{idx}].lot_size must be a finite integer >= 1",
                            code=TARGET_BUNDLE_INVALID,
                        )
                    val = int(val)
                if type(val) is not int or val < 1 or val > MAX_INT64:
                    raise DeployTargetError(
                        f"target_positions[{idx}].lot_size must be an integer in [1, {MAX_INT64}]",
                        code=TARGET_BUNDLE_INVALID,
                    )
                clean_lots.append(val)
            out_data[opt_col] = clean_lots
        else:
            clean_floats: list[float] = []
            for idx, val in enumerate(raw_vals):
                num = require_finite_float(val, f"target_positions[{idx}].{opt_col}")
                if opt_col in ("reference_price", "contract_multiplier") and num <= 0.0:
                    raise DeployTargetError(
                        f"target_positions[{idx}].{opt_col} must be positive, got {num}",
                        code=TARGET_BUNDLE_INVALID,
                    )
                clean_floats.append(num)
            out_data[opt_col] = clean_floats

    normalized = pd.DataFrame(out_data)
    normalized["timestamp_utc"] = pd.to_datetime(normalized["timestamp_utc"], utc=True)

    if normalized.duplicated(subset=["timestamp_utc", "instrument_id"]).any():
        dup_mask = normalized.duplicated(subset=["timestamp_utc", "instrument_id"], keep=False)
        sample = normalized.loc[dup_mask, ["timestamp_utc", "instrument_id"]].iloc[0]
        raise DeployTargetError(
            f"Duplicate primary key (timestamp_utc, instrument_id) in target_positions: "
            f"({format_utc_timestamp(sample['timestamp_utc'])}, {sample['instrument_id']!r})",
            code=TARGET_BUNDLE_INVALID,
        )

    # Verify each timestamp_utc maps consistently to one session_id.
    validate_unique_session_per_timestamp(
        list(zip(normalized["timestamp_utc"].tolist(), normalized["session_id"].tolist())),
        source_name="target_positions",
        code=TARGET_BUNDLE_INVALID,
    )

    normalized = normalized.sort_values(
        by=["timestamp_utc", "instrument_id"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return normalized


def normalize_current_positions(
    current_positions: Mapping[str, Any] | pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    *,
    allow_short: bool,
) -> dict[str, float]:
    """Validate optional initial positions mapping/table into sorted {instrument_id: weight}."""
    if current_positions is None:
        return {}
    out: dict[str, float] = {}
    if isinstance(current_positions, Mapping):
        for raw_inst, raw_w in current_positions.items():
            inst = require_clean_str(raw_inst, "current_positions key")
            weight = require_finite_float(raw_w, f"current_positions[{inst!r}]")
            if not allow_short and weight < 0.0:
                raise DeployTargetError(
                    f"Negative current_position {weight} for {inst!r} is forbidden when allow_short=False",
                    code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
                )
            out[inst] = weight
        return {k: out[k] for k in sorted(out)}

    if isinstance(current_positions, pd.DataFrame):
        if current_positions.empty:
            return {}
        records = current_positions.to_dict(orient="records")
    elif isinstance(current_positions, (str, bytes)) or not isinstance(
        current_positions, Sequence
    ):
        raise DeployTargetError(
            "current_positions must be a mapping, DataFrame, sequence of mappings, or None",
            code=TARGET_BUNDLE_INVALID,
        )
    else:
        records = list(current_positions)

    for idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise DeployTargetError(
                f"current_positions[{idx}] must be a mapping",
                code=TARGET_BUNDLE_INVALID,
            )
        inst = require_clean_str(
            rec.get("instrument_id"),
            f"current_positions[{idx}].instrument_id",
        )
        if inst in out:
            raise DeployTargetError(
                f"Duplicate instrument_id in current_positions: {inst!r}",
                code=TARGET_BUNDLE_INVALID,
            )
        weight_key = next(
            (key for key in ("weight", "current_weight", "target_weight") if key in rec),
            None,
        )
        if weight_key is None:
            raise DeployTargetError(
                f"current_positions[{idx}] missing weight/current_weight/target_weight",
                code=TARGET_BUNDLE_INVALID,
            )
        weight = require_finite_float(rec[weight_key], f"current_positions[{idx}].{weight_key}")
        if not allow_short and weight < 0.0:
            raise DeployTargetError(
                f"Negative current_position {weight} for {inst!r} is forbidden when allow_short=False",
                code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
            )
        out[inst] = weight
    return {k: out[k] for k in sorted(out)}


def validate_rebalance_config(rebalance_config: Any) -> list[CompatibilityIssue]:
    """Validate StrategyBundle.rebalance_config against S5 enums, bounds, and selection consistency."""
    issues: list[CompatibilityIssue] = []
    if not isinstance(rebalance_config, Mapping) or not rebalance_config:
        issues.append(
            CompatibilityIssue(
                code=TARGET_BUNDLE_INVALID,
                message="rebalance_config must be a non-empty mapping",
                field="rebalance_config",
            )
        )
        return issues

    def _add_issue(field: str, message: str) -> None:
        issues.append(
            CompatibilityIssue(
                code=TARGET_BUNDLE_INVALID,
                message=message,
                field=field,
            )
        )

    rebal_freq = rebalance_config.get("rebalance_freq")
    if not isinstance(rebal_freq, str) or not rebal_freq.strip() or rebal_freq != rebal_freq.strip():
        _add_issue(
            "rebalance_config.rebalance_freq",
            "rebalance_config.rebalance_freq must be a nonempty stripped string",
        )
    elif rebal_freq not in VALID_REBALANCE_FREQS and not Frequency.parse(rebal_freq).is_valid:
        _add_issue(
            "rebalance_config.rebalance_freq",
            f"rebalance_config.rebalance_freq {rebal_freq!r} is not a recognized rebalance cadence or Frequency",
        )

    if "rebalance_schedule" in rebalance_config and rebalance_config["rebalance_schedule"] is not None:
        sched = rebalance_config["rebalance_schedule"]
        if not isinstance(sched, str) or not sched.strip() or sched != sched.strip():
            _add_issue(
                "rebalance_config.rebalance_schedule",
                "rebalance_config.rebalance_schedule must be a nonempty stripped string",
            )
        elif sched not in VALID_REBALANCE_SCHEDULES and not Frequency.parse(sched).is_valid:
            _add_issue(
                "rebalance_config.rebalance_schedule",
                f"rebalance_config.rebalance_schedule {sched!r} is not supported",
            )

    for str_field in ("synthesis_name", "trial_id"):
        if str_field in rebalance_config and rebalance_config[str_field] is not None:
            val = rebalance_config[str_field]
            if not isinstance(val, str) or not val.strip() or val != val.strip():
                _add_issue(
                    f"rebalance_config.{str_field}",
                    f"rebalance_config.{str_field} must be a nonempty stripped string",
                )

    for int_field, minimum in (
        ("holding_period", 1),
        ("nth_trading_day", 1),
    ):
        if int_field in rebalance_config and rebalance_config[int_field] is not None:
            val = rebalance_config[int_field]
            if type(val) is bool or not isinstance(val, int) or val < minimum:
                _add_issue(
                    f"rebalance_config.{int_field}",
                    f"rebalance_config.{int_field} must be an integer >= {minimum}",
                )

    top_k = rebalance_config.get("top_k")
    top_k_valid = True
    if top_k is not None:
        if type(top_k) is bool or not isinstance(top_k, int) or top_k < 1:
            top_k_valid = False
            _add_issue(
                "rebalance_config.top_k",
                "rebalance_config.top_k must be an integer >= 1",
            )

    top_pct = rebalance_config.get("top_pct")
    top_pct_valid = True
    if top_pct is not None:
        if type(top_pct) is bool or not isinstance(top_pct, (int, float)):
            top_pct_valid = False
        elif isinstance(top_pct, int):
            top_pct_valid = 0 < top_pct <= 1
        else:
            top_pct_valid = math.isfinite(top_pct) and 0.0 < top_pct <= 1.0
        if not top_pct_valid:
            _add_issue(
                "rebalance_config.top_pct",
                "rebalance_config.top_pct must be finite and within (0, 1]",
            )

    selection_mode = rebalance_config.get("selection_mode")
    if selection_mode is not None:
        if not isinstance(selection_mode, str) or selection_mode not in VALID_SELECTION_MODES:
            _add_issue(
                "rebalance_config.selection_mode",
                f"rebalance_config.selection_mode must be one of {sorted(VALID_SELECTION_MODES)}, got {selection_mode!r}",
            )
        elif selection_mode == "top_k":
            if top_k is None:
                _add_issue(
                    "rebalance_config.top_k",
                    "rebalance_config.selection_mode='top_k' requires a non-null top_k >= 1",
                )
            if top_pct is not None:
                _add_issue(
                    "rebalance_config.top_pct",
                    "rebalance_config.selection_mode='top_k' forbids a non-null top_pct",
                )
        elif selection_mode == "top_pct":
            if top_pct is None:
                _add_issue(
                    "rebalance_config.top_pct",
                    "rebalance_config.selection_mode='top_pct' requires a non-null top_pct in (0, 1]",
                )
            if top_k is not None:
                _add_issue(
                    "rebalance_config.top_k",
                    "rebalance_config.selection_mode='top_pct' forbids a non-null top_k",
                )
    else:
        if top_k is None and top_pct is None:
            _add_issue(
                "rebalance_config",
                "rebalance_config must specify selection_mode or one of top_k / top_pct",
            )
        elif top_k is not None and top_pct is not None and top_k_valid and top_pct_valid:
            _add_issue(
                "rebalance_config",
                "rebalance_config cannot set both top_k and top_pct without explicit selection_mode",
            )

    if "skip_months" in rebalance_config and rebalance_config["skip_months"] is not None:
        skip_raw = rebalance_config["skip_months"]
        if isinstance(skip_raw, (str, bytes, Mapping)) or not isinstance(skip_raw, Sequence):
            _add_issue(
                "rebalance_config.skip_months",
                "rebalance_config.skip_months must be a sequence of month ints (1-12) or MM-DD:MM-DD windows",
            )
        else:
            try:
                parse_skip_rules(skip_raw)
            except (TypeError, ValueError) as exc:
                _add_issue(
                    "rebalance_config.skip_months",
                    f"rebalance_config.skip_months is invalid: {exc}",
                )

    return issues


def check_bundle_compatibility(
    bundle: StrategyBundle | StrategyBundleSource | str | Path,
    *,
    bundle_dir: str | Path | None = None,
    target_id: str = SIGNAL_EXPORT_TARGET_ID,
) -> CompatibilityReport:
    """Validate StrategyBundle completeness, execution_profile, and rebalance_config for signal_export."""
    issues: list[CompatibilityIssue] = []
    bundle_hash = ""
    slot_id = ""
    if not isinstance(bundle, (StrategyBundle, StrategyBundleSource, str, Path)):
        return CompatibilityReport(
            target_id=target_id,
            compatible=False,
            bundle_hash="",
            slot_id="",
            issues=(
                CompatibilityIssue(
                    code=TARGET_BUNDLE_INVALID,
                    message="bundle must be a StrategyBundle, StrategyBundleSource, or bundle_dir path",
                    field="bundle",
                ),
            ),
        )

    bundle_hint = bundle.bundle if isinstance(bundle, StrategyBundleSource) else bundle
    if isinstance(bundle_hint, StrategyBundle):
        raw_hash = getattr(bundle_hint, "bundle_hash", "")
        if isinstance(raw_hash, str) and len(raw_hash) == 64:
            bundle_hash = raw_hash
        raw_slot = getattr(bundle_hint, "slot_id", "")
        if isinstance(raw_slot, str) and raw_slot.strip() and raw_slot == raw_slot.strip():
            slot_id = raw_slot

    try:
        verified_bundle = resolve_verified_bundle(
            bundle,
            bundle_dir=bundle_dir,
            require_verified_source=False,
        )
    except DeployTargetError as exc:
        issues.append(
            CompatibilityIssue(
                code=exc.code,
                message=str(exc),
                field="bundle",
            )
        )
        return CompatibilityReport(
            target_id=target_id,
            compatible=False,
            bundle_hash=bundle_hash,
            slot_id=slot_id,
            issues=tuple(issues),
        )

    bundle_hash = verified_bundle.bundle_hash
    slot_id = verified_bundle.slot_id

    issues.extend(validate_rebalance_config(verified_bundle.rebalance_config))

    exec_profile = verified_bundle.execution_profile
    if not isinstance(exec_profile, Mapping):
        issues.append(
            CompatibilityIssue(
                code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
                message="execution_profile must be a mapping",
                field="execution_profile",
            )
        )
    else:
        try:
            profile_obj = ExecutionProfile(**dict(exec_profile))
        except (TypeError, ValueError) as exc:
            issues.append(
                CompatibilityIssue(
                    code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
                    message=f"Invalid execution_profile: {exc}",
                    field="execution_profile",
                )
            )
        else:
            if verified_bundle.allow_short != profile_obj.allow_short:
                issues.append(
                    CompatibilityIssue(
                        code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
                        message=(
                            f"allow_short ({verified_bundle.allow_short!r}) disagrees with "
                            f"execution_profile.allow_short ({profile_obj.allow_short!r})"
                        ),
                        field="execution_profile.allow_short",
                    )
                )
            if verified_bundle.benchmark_instrument != profile_obj.benchmark_instrument:
                issues.append(
                    CompatibilityIssue(
                        code=TARGET_EXECUTION_CONTRACT_UNSUPPORTED,
                        message=(
                            f"benchmark_instrument ({verified_bundle.benchmark_instrument!r}) disagrees with "
                            f"execution_profile.benchmark_instrument ({profile_obj.benchmark_instrument!r})"
                        ),
                        field="execution_profile.benchmark_instrument",
                    )
                )

    return CompatibilityReport(
        target_id=target_id,
        compatible=not any(item.severity == "error" for item in issues),
        bundle_hash=bundle_hash,
        slot_id=slot_id,
        issues=tuple(issues),
    )


__all__ = [
    "ORDER_OUTPUT_COLUMNS",
    "MAX_INT64",
    "POSITION_BASE_OUTPUT_COLUMNS",
    "POSITION_OPTIONAL_COLUMNS",
    "POSITION_REQUIRED_COLUMNS",
    "SIGNAL_EXPORT_TARGET_ID",
    "SIGNAL_POSITION_ARTIFACT_TYPES",
    "VALID_ACTIONS",
    "VALID_EXPORT_FORMATS",
    "VALID_REBALANCE_FREQS",
    "VALID_REBALANCE_SCHEDULES",
    "VALID_SELECTION_MODES",
    "VALID_SIDES",
    "check_bundle_compatibility",
    "derive_order_action",
    "derive_side",
    "format_utc_timestamp",
    "load_bundle_signal_positions",
    "normalize_current_positions",
    "normalize_formats",
    "normalize_target_positions",
    "parse_utc_timestamp",
    "read_jsonl_records",
    "read_parquet_records",
    "require_clean_str",
    "require_finite_float",
    "validate_rebalance_config",
    "validate_unique_session_per_timestamp",
]
