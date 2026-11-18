"""DuckDB persistence and read-only query store for Bar evaluation results (Stage 6.6).

Stores published `time-series-analysis` evaluation artifacts in an explicitly
specified DuckDB database across four tables:
  - bar_evaluation_runs
  - bar_evaluation_ic
  - bar_evaluation_quantiles
  - bar_evaluation_cross_sections

`per_bar_labels.csv` is NOT stored in DuckDB; only its path and SHA256 digest
are recorded in `bar_evaluation_runs`.

Write locking reuses `utils.duckdb_results_writer.acquire_exclusive_write_conn`
so this module never creates `meta.*` tables or modifies any existing table
such as `factor_analysis_results` or `price_data_*`.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import pandas as pd

from backtest_and_analysis.time_series.contracts import ResultIdentity, sha256_file
from backtest_and_analysis.time_series.evaluator import (
    CS_COLUMNS,
    CS_IC_COLUMNS,
    CS_QUANTILE_COLUMNS,
    RESULT_SCHEMA_VERSION,
    TS_IC_COLUMNS,
    TS_QUANTILE_COLUMNS,
)
from utils.data_store import _reject_sqlite_file
from utils.duckdb_results_writer import acquire_exclusive_write_conn
from utils.time_axis import Frequency


BAR_EVALUATION_TABLES: tuple[str, ...] = (
    "bar_evaluation_runs",
    "bar_evaluation_ic",
    "bar_evaluation_quantiles",
    "bar_evaluation_cross_sections",
)
SUPPORTED_RESULT_SCHEMA_VERSIONS: tuple[str, ...] = (RESULT_SCHEMA_VERSION,)
VALID_ANALYSIS_MODES: tuple[str, ...] = ("time_series", "cross_sectional")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_CSV_STRING_DTYPES: dict[str, str] = {
    "scope": "string",
    "instrument_id": "string",
    "frequency": "string",
    "ic_status": "string",
    "non_overlapping_status": "string",
    "quantile_status": "string",
    "quantile_scope": "string",
    "exclusion_counts": "string",
    "nonzero_exclusion_counts": "string",
}

_IC_TABLE_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "row_index",
    "analysis_mode",
    "scope",
    "instrument_id",
    "frequency",
    "horizon_bars",
    "total_count",
    "valid_count",
    "excluded_count",
    "exclusion_counts",
    "nonzero_exclusion_counts",
    "spearman_ic",
    "ic_status",
    "non_overlapping_ic",
    "non_overlapping_count",
    "non_overlapping_status",
    "requested_quantiles",
    "actual_quantiles",
    "quantile_status",
    "quantile_scope",
    "descriptive_only",
    "mean_spearman_ic",
    "median_spearman_ic",
    "mean_non_overlapping_ic",
    "median_non_overlapping_ic",
    "instrument_count",
    "valid_instrument_count",
    "valid_non_overlapping_instrument_count",
    "mean_ic",
    "median_ic",
    "std_ic",
    "icir",
    "positive_ic_ratio",
    "total_cross_sections",
    "valid_cross_sections",
    "excluded_cross_sections",
    "insufficient_cross_section_count",
    "constant_factor_cross_section_count",
    "constant_return_cross_section_count",
    "mean_valid_instruments",
    "median_valid_instruments",
)

_QUANTILE_TABLE_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "row_index",
    "analysis_mode",
    "instrument_id",
    "frequency",
    "horizon_bars",
    "quantile",
    "requested_quantiles",
    "actual_quantiles",
    "min_actual_quantiles",
    "max_actual_quantiles",
    "quantile_status",
    "quantile_scope",
    "descriptive_only",
    "sample_count",
    "cross_section_count",
    "total_sample_count",
    "mean_return_pct",
    "median_return_pct",
    "positive_ratio",
)

_CROSS_SECTION_TABLE_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "row_index",
    "frequency",
    "horizon_bars",
    "signal_timestamp",
    "total_count",
    "valid_count",
    "excluded_count",
    "exclusion_counts",
    "nonzero_exclusion_counts",
    "valid_instruments",
    "spearman_ic",
    "cs_status",
    "requested_quantiles",
    "actual_quantiles",
    "quantile_status",
    "quantile_scope",
    "descriptive_only",
)

_RUN_SEMANTIC_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "analysis_mode",
    "frequency",
    "view",
    "boundary_policy",
    "bucket_strategy",
    "calendar_id",
    "calendar_hash",
    "config_hash",
    "input_file_hashes",
    "data_identity",
    "code_version",
    "code_version_source",
    "result_schema_version",
    "exclusion_summary",
)

_RUN_ARTIFACT_HASH_COLUMNS: tuple[str, ...] = (
    "summary_sha256",
    "per_bar_labels_sha256",
    "ic_by_horizon_sha256",
    "quantile_stats_sha256",
)

_RUN_ALL_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "analysis_mode",
    "frequency",
    "view",
    "boundary_policy",
    "bucket_strategy",
    "calendar_id",
    "calendar_hash",
    "config_hash",
    "input_file_hashes",
    "data_identity",
    "code_version",
    "code_version_source",
    "result_schema_version",
    "exclusion_summary",
    "summary_path",
    "summary_sha256",
    "per_bar_labels_path",
    "per_bar_labels_sha256",
    "ic_by_horizon_path",
    "ic_by_horizon_sha256",
    "quantile_stats_path",
    "quantile_stats_sha256",
    "run_id",
    "ingested_at_utc",
)

_IC_INT_COLUMNS = frozenset({
    "row_index", "horizon_bars", "total_count", "valid_count", "excluded_count",
    "non_overlapping_count", "requested_quantiles", "actual_quantiles",
    "instrument_count", "valid_instrument_count", "valid_non_overlapping_instrument_count",
    "total_cross_sections", "valid_cross_sections", "excluded_cross_sections",
    "insufficient_cross_section_count", "constant_factor_cross_section_count",
    "constant_return_cross_section_count",
})
_IC_FLOAT_COLUMNS = frozenset({
    "spearman_ic", "non_overlapping_ic", "mean_spearman_ic", "median_spearman_ic",
    "mean_non_overlapping_ic", "median_non_overlapping_ic", "mean_ic", "median_ic",
    "std_ic", "icir", "positive_ic_ratio", "mean_valid_instruments", "median_valid_instruments",
})
_IC_BOOL_COLUMNS = frozenset({"descriptive_only"})
_IC_JSON_COLUMNS = frozenset({"exclusion_counts", "nonzero_exclusion_counts"})

_QUANTILE_INT_COLUMNS = frozenset({
    "row_index", "horizon_bars", "quantile", "requested_quantiles", "actual_quantiles",
    "min_actual_quantiles", "max_actual_quantiles", "sample_count",
    "cross_section_count", "total_sample_count",
})
_QUANTILE_FLOAT_COLUMNS = frozenset({"mean_return_pct", "median_return_pct", "positive_ratio"})
_QUANTILE_BOOL_COLUMNS = frozenset({"descriptive_only"})

_CS_INT_COLUMNS = frozenset({
    "row_index", "horizon_bars", "total_count", "valid_count", "excluded_count",
    "valid_instruments", "requested_quantiles", "actual_quantiles",
})
_CS_FLOAT_COLUMNS = frozenset({"spearman_ic"})
_CS_BOOL_COLUMNS = frozenset({"descriptive_only"})
_CS_JSON_COLUMNS = frozenset({"exclusion_counts", "nonzero_exclusion_counts"})


class BarEvaluationPreconditionError(ValueError):
    """Raised when results DB path, schema prerequisites, or inputs are invalid."""


class BarEvaluationIntegrityError(ValueError):
    """Raised when published artifact files fail completeness or SHA256 verification."""


class BarEvaluationConflictError(ValueError):
    """Raised when re-ingesting an existing identity_hash with different content."""

    def __init__(self, identity_hash: str, message: str) -> None:
        self.identity_hash = identity_hash
        super().__init__(message)


@dataclass(frozen=True)
class BarEvaluationIngestReceipt:
    db_path: Path
    identity_hash: str
    status: str  # "inserted" or "noop"
    analysis_mode: str
    ic_rows: int
    quantile_rows: int
    cross_section_rows: int
    ingested_at_utc: str


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
    if isinstance(value, Mapping):
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
    raise BarEvaluationIntegrityError(f"{field} must be JSON text or mapping")


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


def _resolve_artifact_paths(
    evaluation_dir: str | Path | None,
    artifacts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    expected_hashes: Mapping[str, str] | None,
) -> tuple[dict[str, Path], dict[str, str]]:
    expected: dict[str, str] = {}
    if expected_hashes is not None:
        for key, digest in expected_hashes.items():
            normalized_key = "summary" if key == "evaluation_summary" else key
            expected[normalized_key] = str(digest)

    paths: dict[str, Path] = {}
    required = ("summary", "per_bar_labels", "ic_by_horizon", "quantile_stats")

    if artifacts is not None:
        if isinstance(artifacts, Mapping):
            for key, item in artifacts.items():
                normalized_key = "summary" if key == "evaluation_summary" else key
                if normalized_key not in required:
                    continue
                if isinstance(item, Mapping):
                    raw_path = item.get("path")
                    if not isinstance(raw_path, str) or not raw_path.strip():
                        raise BarEvaluationIntegrityError(f"Artifact {key!r} is missing path")
                    paths[normalized_key] = Path(raw_path).expanduser().resolve()
                    if "sha256" in item and item["sha256"] is not None:
                        expected[normalized_key] = str(item["sha256"])
                else:
                    paths[normalized_key] = Path(item).expanduser().resolve()
        else:
            for item in artifacts:
                if not isinstance(item, Mapping):
                    continue
                atype = item.get("artifact_type")
                if atype not in ("evaluation_summary", "per_bar_labels", "ic_by_horizon", "quantile_stats"):
                    continue
                key = "summary" if atype == "evaluation_summary" else str(atype)
                raw_path = item.get("path")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise BarEvaluationIntegrityError(f"Artifact {atype!r} is missing path")
                paths[key] = Path(raw_path).expanduser().resolve()
                if "sha256" in item and item["sha256"] is not None:
                    expected[key] = str(item["sha256"])

    if evaluation_dir is not None:
        base = Path(evaluation_dir).expanduser().resolve()
        default_names = {
            "summary": "summary.json",
            "per_bar_labels": "per_bar_labels.csv",
            "ic_by_horizon": "ic_by_horizon.csv",
            "quantile_stats": "quantile_stats.csv",
        }
        for key, filename in default_names.items():
            paths.setdefault(key, base / filename)

    missing_keys = [key for key in required if key not in paths]
    if missing_keys:
        raise BarEvaluationIntegrityError(
            f"Missing published evaluation artifact descriptors: {missing_keys}"
        )

    actual_hashes: dict[str, str] = {}
    for key in required:
        file_path = paths[key]
        if not file_path.is_file():
            raise BarEvaluationIntegrityError(
                f"Published evaluation artifact missing ({key}): {file_path}"
            )
        if key not in expected or not _SHA256_RE.fullmatch(expected[key]):
            raise BarEvaluationIntegrityError(f"Missing or invalid Harness SHA256 for artifact {key}")
        computed = sha256_file(file_path)
        if computed != expected[key]:
            raise BarEvaluationIntegrityError(
                f"SHA256 mismatch for published artifact {key} ({file_path}): "
                f"expected {expected[key]}, computed {computed}"
            )
        actual_hashes[key] = computed

    return paths, actual_hashes


def _cross_check_csv_against_summary(
    csv_records: list[dict[str, Any]],
    summary_records: Any,
    *,
    table_label: str,
    expected_columns: list[str],
    int_cols: frozenset[str],
    float_cols: frozenset[str],
    bool_cols: frozenset[str],
    json_cols: frozenset[str],
) -> None:
    if not isinstance(summary_records, list):
        raise BarEvaluationIntegrityError(f"summary.json {table_label} must be a list")
    if len(csv_records) != len(summary_records):
        raise BarEvaluationIntegrityError(
            f"{table_label} row count mismatch between CSV ({len(csv_records)}) "
            f"and summary.json ({len(summary_records)})"
        )
    for idx, (csv_row, json_row) in enumerate(zip(csv_records, summary_records)):
        if not isinstance(json_row, Mapping):
            raise BarEvaluationIntegrityError(f"summary.json {table_label}[{idx}] must be an object")
        for col in expected_columns:
            c_val = csv_row.get(col)
            j_val = json_row.get(col)
            if _is_missing(c_val) and _is_missing(j_val):
                continue
            if _is_missing(c_val) != _is_missing(j_val):
                raise BarEvaluationIntegrityError(
                    f"{table_label}[{idx}].{col} mismatch between CSV ({c_val!r}) "
                    f"and summary.json ({j_val!r})"
                )
            if col in float_cols:
                cf = _coerce_optional_float(c_val, col)
                jf = _coerce_optional_float(j_val, col)
                if cf is None or jf is None or not math.isclose(cf, jf, rel_tol=1e-12, abs_tol=1e-12):
                    raise BarEvaluationIntegrityError(
                        f"{table_label}[{idx}].{col} numeric mismatch between CSV ({cf!r}) "
                        f"and summary.json ({jf!r})"
                    )
            elif col in int_cols:
                if _coerce_optional_int(c_val, col) != _coerce_optional_int(j_val, col):
                    raise BarEvaluationIntegrityError(
                        f"{table_label}[{idx}].{col} mismatch between CSV and summary.json"
                    )
            elif col in bool_cols:
                if _coerce_optional_bool(c_val, col) != _coerce_optional_bool(j_val, col):
                    raise BarEvaluationIntegrityError(
                        f"{table_label}[{idx}].{col} mismatch between CSV and summary.json"
                    )
            elif col in json_cols:
                cj = _coerce_optional_json_str(c_val, col)
                jj = _coerce_optional_json_str(j_val, col)
                if cj != jj:
                    raise BarEvaluationIntegrityError(
                        f"{table_label}[{idx}].{col} JSON mismatch between CSV and summary.json"
                    )
            else:
                if str(c_val) != str(j_val):
                    raise BarEvaluationIntegrityError(
                        f"{table_label}[{idx}].{col} mismatch between CSV ({c_val!r}) "
                        f"and summary.json ({j_val!r})"
                    )


def _extract_data_identity_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw = summary.get("data_identity")
    if not isinstance(raw, Mapping):
        raise BarEvaluationIntegrityError("summary.json data_identity must be a mapping")
    payload = dict(_json_safe(raw))
    # Preserve build_id / roll_build_ids if present anywhere in summary/config without fabricating.
    for extra_key in ("build_id", "build_ids", "roll_build_ids"):
        if extra_key in summary and extra_key not in payload and summary[extra_key] is not None:
            payload[extra_key] = _json_safe(summary[extra_key])
    config = summary.get("config")
    if isinstance(config, Mapping):
        continuous = config.get("continuous_futures")
        if isinstance(continuous, Mapping):
            payload.setdefault("continuous_futures", _json_safe(continuous))
    return payload


def _resolve_frequency_field(data_identity: Mapping[str, Any], ic_rows: list[dict[str, Any]]) -> str:
    freqs = data_identity.get("frequencies")
    if isinstance(freqs, list) and freqs:
        clean = [str(f) for f in freqs]
    else:
        clean = sorted({str(row["frequency"]) for row in ic_rows if not _is_missing(row.get("frequency"))})
    if not clean:
        return "[]"
    if len(clean) == 1:
        return clean[0]
    return _canonical_json(clean)


def _validate_and_prepare_payload(
    paths: dict[str, Path],
    hashes: dict[str, str],
    run_id: str | None,
) -> tuple[dict[str, Any], list[tuple[Any, ...]], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    try:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BarEvaluationIntegrityError(f"Failed to parse summary.json ({paths['summary']}): {exc}") from exc
    if not isinstance(summary, dict):
        raise BarEvaluationIntegrityError("summary.json must contain a JSON object")

    schema_version = summary.get("result_schema_version")
    if schema_version not in SUPPORTED_RESULT_SCHEMA_VERSIONS:
        raise BarEvaluationIntegrityError(
            f"Unsupported result_schema_version: {schema_version!r}; "
            f"supported={SUPPORTED_RESULT_SCHEMA_VERSIONS}"
        )

    analysis_mode = summary.get("analysis_mode")
    if analysis_mode not in VALID_ANALYSIS_MODES:
        raise BarEvaluationIntegrityError(f"Invalid analysis_mode in summary.json: {analysis_mode!r}")

    rid = summary.get("result_identity")
    if not isinstance(rid, Mapping):
        raise BarEvaluationIntegrityError("summary.json missing result_identity object")
    for field in ("config_hash", "input_file_hashes", "calendar_hash", "code_version", "result_schema_version", "identity_hash"):
        if field not in rid:
            raise BarEvaluationIntegrityError(f"summary.json result_identity missing {field}")
    try:
        reconstructed = ResultIdentity(
            config_hash=rid["config_hash"],
            input_file_hashes=rid["input_file_hashes"],
            calendar_hash=rid["calendar_hash"],
            code_version=rid["code_version"],
            result_schema_version=rid["result_schema_version"],
        )
    except Exception as exc:
        raise BarEvaluationIntegrityError(f"Invalid result_identity in summary.json: {exc}") from exc

    identity_hash = str(rid["identity_hash"])
    if not _SHA256_RE.match(identity_hash) or reconstructed.identity_hash() != identity_hash:
        raise BarEvaluationIntegrityError(
            f"result_identity.identity_hash verification failed: expected {reconstructed.identity_hash()}, "
            f"got {identity_hash!r}"
        )
    if reconstructed.result_schema_version != schema_version:
        raise BarEvaluationIntegrityError("result_identity.result_schema_version disagrees with top-level version")

    cal_identity = summary.get("calendar_identity")
    if not isinstance(cal_identity, Mapping):
        raise BarEvaluationIntegrityError("summary.json missing calendar_identity object")
    if cal_identity.get("calendar_hash") is not None and cal_identity.get("calendar_hash") != reconstructed.calendar_hash:
        raise BarEvaluationIntegrityError("calendar_identity.calendar_hash disagrees with result_identity")

    exclusion_summary = summary.get("exclusion_summary")
    if not isinstance(exclusion_summary, Mapping):
        raise BarEvaluationIntegrityError("summary.json missing exclusion_summary object")

    config = summary.get("config")
    if config is not None and not isinstance(config, Mapping):
        raise BarEvaluationIntegrityError("summary.json config must be an object")
    config = dict(config) if config is not None else {}

    data_identity_payload = _extract_data_identity_payload(summary)

    # Read and validate CSV tables.
    expected_ic_cols = TS_IC_COLUMNS if analysis_mode == "time_series" else CS_IC_COLUMNS
    expected_q_cols = TS_QUANTILE_COLUMNS if analysis_mode == "time_series" else CS_QUANTILE_COLUMNS

    try:
        ic_df = pd.read_csv(paths["ic_by_horizon"], dtype=_CSV_STRING_DTYPES)
        q_df = pd.read_csv(paths["quantile_stats"], dtype=_CSV_STRING_DTYPES)
    except Exception as exc:
        raise BarEvaluationIntegrityError(f"Failed to read evaluation CSV artifacts: {exc}") from exc

    if list(ic_df.columns) != list(expected_ic_cols):
        raise BarEvaluationIntegrityError(
            f"ic_by_horizon.csv columns {list(ic_df.columns)} do not match "
            f"{analysis_mode} schema {list(expected_ic_cols)}"
        )
    if list(q_df.columns) != list(expected_q_cols):
        raise BarEvaluationIntegrityError(
            f"quantile_stats.csv columns {list(q_df.columns)} do not match "
            f"{analysis_mode} schema {list(expected_q_cols)}"
        )

    ic_records = ic_df.to_dict("records")
    q_records = q_df.to_dict("records")

    _cross_check_csv_against_summary(
        ic_records,
        summary.get("ic_by_horizon"),
        table_label="ic_by_horizon",
        expected_columns=expected_ic_cols,
        int_cols=_IC_INT_COLUMNS,
        float_cols=_IC_FLOAT_COLUMNS,
        bool_cols=_IC_BOOL_COLUMNS,
        json_cols=_IC_JSON_COLUMNS,
    )
    _cross_check_csv_against_summary(
        q_records,
        summary.get("quantile_stats"),
        table_label="quantile_stats",
        expected_columns=expected_q_cols,
        int_cols=_QUANTILE_INT_COLUMNS,
        float_cols=_QUANTILE_FLOAT_COLUMNS,
        bool_cols=_QUANTILE_BOOL_COLUMNS,
        json_cols=frozenset(),
    )

    cross_sections_raw = summary.get("cross_sections")
    if not isinstance(cross_sections_raw, list):
        raise BarEvaluationIntegrityError("summary.json cross_sections must be a list")
    if analysis_mode == "time_series" and cross_sections_raw:
        raise BarEvaluationIntegrityError("time_series evaluation must not contain cross_sections rows")

    # Normalize child table rows.
    ic_tuples: list[tuple[Any, ...]] = []
    for row_index, rec in enumerate(ic_records):
        row_map: dict[str, Any] = {
            "identity_hash": identity_hash,
            "row_index": row_index,
            "analysis_mode": analysis_mode,
        }
        for col in _IC_TABLE_COLUMNS[3:]:
            if col not in expected_ic_cols:
                row_map[col] = None
            elif col in _IC_INT_COLUMNS:
                row_map[col] = _coerce_optional_int(rec.get(col), col)
            elif col in _IC_FLOAT_COLUMNS:
                row_map[col] = _coerce_optional_float(rec.get(col), col)
            elif col in _IC_BOOL_COLUMNS:
                row_map[col] = _coerce_optional_bool(rec.get(col), col)
            elif col in _IC_JSON_COLUMNS:
                row_map[col] = _coerce_optional_json_str(rec.get(col), col)
            else:
                row_map[col] = _coerce_optional_str(rec.get(col))
        ic_tuples.append(tuple(row_map[col] for col in _IC_TABLE_COLUMNS))

    q_tuples: list[tuple[Any, ...]] = []
    for row_index, rec in enumerate(q_records):
        row_map = {
            "identity_hash": identity_hash,
            "row_index": row_index,
            "analysis_mode": analysis_mode,
        }
        for col in _QUANTILE_TABLE_COLUMNS[3:]:
            if col not in expected_q_cols:
                row_map[col] = None
            elif col in _QUANTILE_INT_COLUMNS:
                row_map[col] = _coerce_optional_int(rec.get(col), col)
            elif col in _QUANTILE_FLOAT_COLUMNS:
                row_map[col] = _coerce_optional_float(rec.get(col), col)
            elif col in _QUANTILE_BOOL_COLUMNS:
                row_map[col] = _coerce_optional_bool(rec.get(col), col)
            else:
                row_map[col] = _coerce_optional_str(rec.get(col))
        q_tuples.append(tuple(row_map[col] for col in _QUANTILE_TABLE_COLUMNS))

    cs_tuples: list[tuple[Any, ...]] = []
    for row_index, rec in enumerate(cross_sections_raw):
        if not isinstance(rec, Mapping):
            raise BarEvaluationIntegrityError(f"cross_sections[{row_index}] must be an object")
        missing_cs = [col for col in CS_COLUMNS if col not in rec]
        if missing_cs:
            raise BarEvaluationIntegrityError(f"cross_sections[{row_index}] missing columns: {missing_cs}")
        row_map = {
            "identity_hash": identity_hash,
            "row_index": row_index,
        }
        for col in _CROSS_SECTION_TABLE_COLUMNS[2:]:
            if col == "signal_timestamp":
                row_map[col] = _parse_utc_timestamp(rec.get(col), f"cross_sections[{row_index}].signal_timestamp").to_pydatetime()
            elif col in _CS_INT_COLUMNS:
                row_map[col] = _coerce_optional_int(rec.get(col), col)
            elif col in _CS_FLOAT_COLUMNS:
                row_map[col] = _coerce_optional_float(rec.get(col), col)
            elif col in _CS_BOOL_COLUMNS:
                row_map[col] = _coerce_optional_bool(rec.get(col), col)
            elif col in _CS_JSON_COLUMNS:
                row_map[col] = _coerce_optional_json_str(rec.get(col), col)
            else:
                row_map[col] = _coerce_optional_str(rec.get(col))
        cs_tuples.append(tuple(row_map[col] for col in _CROSS_SECTION_TABLE_COLUMNS))

    bucket_strategy = config.get("bucket_strategy")
    if bucket_strategy is None and isinstance(config.get("continuous_futures"), Mapping):
        bucket_strategy = config["continuous_futures"].get("bucket_strategy")
    frequency_str = _resolve_frequency_field(data_identity_payload, ic_records)
    ingested_at = dt.datetime.now(dt.timezone.utc)

    run_record: dict[str, Any] = {
        "identity_hash": identity_hash,
        "analysis_mode": analysis_mode,
        "frequency": frequency_str,
        "view": _coerce_optional_str(cal_identity.get("view")),
        "boundary_policy": _coerce_optional_str(cal_identity.get("boundary_policy")),
        "bucket_strategy": _coerce_optional_str(bucket_strategy),
        "calendar_id": _coerce_optional_str(cal_identity.get("calendar_id")),
        "calendar_hash": reconstructed.calendar_hash,
        "config_hash": reconstructed.config_hash,
        "input_file_hashes": _canonical_json(dict(reconstructed.input_file_hashes)),
        "data_identity": _canonical_json(data_identity_payload),
        "code_version": reconstructed.code_version,
        "code_version_source": _coerce_optional_str(rid.get("code_version_source")),
        "result_schema_version": reconstructed.result_schema_version,
        "exclusion_summary": _canonical_json(_json_safe(exclusion_summary)),
        "summary_path": str(paths["summary"]),
        "summary_sha256": hashes["summary"],
        "per_bar_labels_path": str(paths["per_bar_labels"]),
        "per_bar_labels_sha256": hashes["per_bar_labels"],
        "ic_by_horizon_path": str(paths["ic_by_horizon"]),
        "ic_by_horizon_sha256": hashes["ic_by_horizon"],
        "quantile_stats_path": str(paths["quantile_stats"]),
        "quantile_stats_sha256": hashes["quantile_stats"],
        "run_id": _coerce_optional_str(run_id),
        "ingested_at_utc": ingested_at,
    }
    return run_record, ic_tuples, q_tuples, cs_tuples


def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bar_evaluation_runs (
            identity_hash VARCHAR PRIMARY KEY,
            analysis_mode VARCHAR NOT NULL,
            frequency VARCHAR NOT NULL,
            view VARCHAR,
            boundary_policy VARCHAR,
            bucket_strategy VARCHAR,
            calendar_id VARCHAR,
            calendar_hash VARCHAR NOT NULL,
            config_hash VARCHAR NOT NULL,
            input_file_hashes VARCHAR NOT NULL,
            data_identity VARCHAR NOT NULL,
            code_version VARCHAR NOT NULL,
            code_version_source VARCHAR,
            result_schema_version VARCHAR NOT NULL,
            exclusion_summary VARCHAR NOT NULL,
            summary_path VARCHAR NOT NULL,
            summary_sha256 VARCHAR NOT NULL,
            per_bar_labels_path VARCHAR NOT NULL,
            per_bar_labels_sha256 VARCHAR NOT NULL,
            ic_by_horizon_path VARCHAR NOT NULL,
            ic_by_horizon_sha256 VARCHAR NOT NULL,
            quantile_stats_path VARCHAR NOT NULL,
            quantile_stats_sha256 VARCHAR NOT NULL,
            run_id VARCHAR,
            ingested_at_utc TIMESTAMPTZ NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bar_evaluation_ic (
            identity_hash VARCHAR NOT NULL,
            row_index BIGINT NOT NULL,
            analysis_mode VARCHAR NOT NULL,
            scope VARCHAR,
            instrument_id VARCHAR,
            frequency VARCHAR NOT NULL,
            horizon_bars BIGINT NOT NULL,
            total_count BIGINT,
            valid_count BIGINT,
            excluded_count BIGINT,
            exclusion_counts VARCHAR,
            nonzero_exclusion_counts VARCHAR,
            spearman_ic DOUBLE,
            ic_status VARCHAR,
            non_overlapping_ic DOUBLE,
            non_overlapping_count BIGINT,
            non_overlapping_status VARCHAR,
            requested_quantiles BIGINT,
            actual_quantiles BIGINT,
            quantile_status VARCHAR,
            quantile_scope VARCHAR,
            descriptive_only BOOLEAN,
            mean_spearman_ic DOUBLE,
            median_spearman_ic DOUBLE,
            mean_non_overlapping_ic DOUBLE,
            median_non_overlapping_ic DOUBLE,
            instrument_count BIGINT,
            valid_instrument_count BIGINT,
            valid_non_overlapping_instrument_count BIGINT,
            mean_ic DOUBLE,
            median_ic DOUBLE,
            std_ic DOUBLE,
            icir DOUBLE,
            positive_ic_ratio DOUBLE,
            total_cross_sections BIGINT,
            valid_cross_sections BIGINT,
            excluded_cross_sections BIGINT,
            insufficient_cross_section_count BIGINT,
            constant_factor_cross_section_count BIGINT,
            constant_return_cross_section_count BIGINT,
            mean_valid_instruments DOUBLE,
            median_valid_instruments DOUBLE,
            PRIMARY KEY (identity_hash, row_index)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bar_evaluation_quantiles (
            identity_hash VARCHAR NOT NULL,
            row_index BIGINT NOT NULL,
            analysis_mode VARCHAR NOT NULL,
            instrument_id VARCHAR,
            frequency VARCHAR NOT NULL,
            horizon_bars BIGINT NOT NULL,
            quantile BIGINT NOT NULL,
            requested_quantiles BIGINT,
            actual_quantiles BIGINT,
            min_actual_quantiles BIGINT,
            max_actual_quantiles BIGINT,
            quantile_status VARCHAR,
            quantile_scope VARCHAR,
            descriptive_only BOOLEAN,
            sample_count BIGINT,
            cross_section_count BIGINT,
            total_sample_count BIGINT,
            mean_return_pct DOUBLE,
            median_return_pct DOUBLE,
            positive_ratio DOUBLE,
            PRIMARY KEY (identity_hash, row_index)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bar_evaluation_cross_sections (
            identity_hash VARCHAR NOT NULL,
            row_index BIGINT NOT NULL,
            frequency VARCHAR NOT NULL,
            horizon_bars BIGINT NOT NULL,
            signal_timestamp TIMESTAMPTZ NOT NULL,
            total_count BIGINT,
            valid_count BIGINT,
            excluded_count BIGINT,
            exclusion_counts VARCHAR,
            nonzero_exclusion_counts VARCHAR,
            valid_instruments BIGINT,
            spearman_ic DOUBLE,
            cs_status VARCHAR,
            requested_quantiles BIGINT,
            actual_quantiles BIGINT,
            quantile_status VARCHAR,
            quantile_scope VARCHAR,
            descriptive_only BOOLEAN,
            PRIMARY KEY (identity_hash, row_index)
        )"""
    )


def _column_kind(col: str, int_cols: frozenset[str], float_cols: frozenset[str],
                 bool_cols: frozenset[str], json_cols: frozenset[str]) -> str:
    if col in int_cols:
        return "int"
    if col in float_cols:
        return "float"
    if col in bool_cols:
        return "bool"
    if col in json_cols:
        return "json"
    if col == "signal_timestamp":
        return "timestamp"
    return "str"


def _compare_table_rows(
    identity_hash: str,
    table_name: str,
    columns: tuple[str, ...],
    existing_rows: list[tuple[Any, ...]],
    incoming_rows: list[tuple[Any, ...]],
    *,
    int_cols: frozenset[str],
    float_cols: frozenset[str],
    bool_cols: frozenset[str],
    json_cols: frozenset[str],
) -> None:
    if len(existing_rows) != len(incoming_rows):
        raise BarEvaluationConflictError(
            identity_hash,
            f"Conflict for identity_hash={identity_hash} in {table_name}: "
            f"existing {len(existing_rows)} rows != incoming {len(incoming_rows)} rows",
        )
    kinds = [_column_kind(col, int_cols, float_cols, bool_cols, json_cols) for col in columns]
    for row_idx, (old_row, new_row) in enumerate(zip(existing_rows, incoming_rows)):
        for col_idx, (col, kind) in enumerate(zip(columns, kinds)):
            if _canonical_compare_token(old_row[col_idx], kind=kind) != _canonical_compare_token(new_row[col_idx], kind=kind):
                raise BarEvaluationConflictError(
                    identity_hash,
                    f"Conflict for identity_hash={identity_hash} in {table_name} "
                    f"row={row_idx} column={col}: existing={old_row[col_idx]!r} != incoming={new_row[col_idx]!r}",
                )


def _verify_existing_run_matches(
    conn: duckdb.DuckDBPyConnection,
    run_record: dict[str, Any],
    ic_tuples: list[tuple[Any, ...]],
    q_tuples: list[tuple[Any, ...]],
    cs_tuples: list[tuple[Any, ...]],
) -> None:
    identity_hash = run_record["identity_hash"]
    run_cols_sql = ", ".join(_RUN_SEMANTIC_COLUMNS)
    existing_run = conn.execute(
        f"SELECT {run_cols_sql} FROM bar_evaluation_runs WHERE identity_hash=?",
        [identity_hash],
    ).fetchone()
    assert existing_run is not None

    json_run_cols = {"input_file_hashes", "data_identity", "exclusion_summary"}
    for idx, col in enumerate(_RUN_SEMANTIC_COLUMNS):
        kind = "json" if col in json_run_cols else "str"
        if _canonical_compare_token(existing_run[idx], kind=kind) != _canonical_compare_token(run_record[col], kind=kind):
            raise BarEvaluationConflictError(
                identity_hash,
                f"Conflict for identity_hash={identity_hash} in bar_evaluation_runs "
                f"column={col}: existing={existing_run[idx]!r} != incoming={run_record[col]!r}",
            )

    existing_ic = conn.execute(
        f"SELECT {', '.join(_IC_TABLE_COLUMNS)} FROM bar_evaluation_ic "
        "WHERE identity_hash=? ORDER BY row_index",
        [identity_hash],
    ).fetchall()
    _compare_table_rows(
        identity_hash,
        "bar_evaluation_ic",
        _IC_TABLE_COLUMNS,
        existing_ic,
        ic_tuples,
        int_cols=_IC_INT_COLUMNS,
        float_cols=_IC_FLOAT_COLUMNS,
        bool_cols=_IC_BOOL_COLUMNS,
        json_cols=_IC_JSON_COLUMNS,
    )

    existing_q = conn.execute(
        f"SELECT {', '.join(_QUANTILE_TABLE_COLUMNS)} FROM bar_evaluation_quantiles "
        "WHERE identity_hash=? ORDER BY row_index",
        [identity_hash],
    ).fetchall()
    _compare_table_rows(
        identity_hash,
        "bar_evaluation_quantiles",
        _QUANTILE_TABLE_COLUMNS,
        existing_q,
        q_tuples,
        int_cols=_QUANTILE_INT_COLUMNS,
        float_cols=_QUANTILE_FLOAT_COLUMNS,
        bool_cols=_QUANTILE_BOOL_COLUMNS,
        json_cols=frozenset(),
    )

    existing_cs = conn.execute(
        f"SELECT {', '.join(_CROSS_SECTION_TABLE_COLUMNS)} FROM bar_evaluation_cross_sections "
        "WHERE identity_hash=? ORDER BY row_index",
        [identity_hash],
    ).fetchall()
    _compare_table_rows(
        identity_hash,
        "bar_evaluation_cross_sections",
        _CROSS_SECTION_TABLE_COLUMNS,
        existing_cs,
        cs_tuples,
        int_cols=_CS_INT_COLUMNS,
        float_cols=_CS_FLOAT_COLUMNS,
        bool_cols=_CS_BOOL_COLUMNS,
        json_cols=_CS_JSON_COLUMNS,
    )

    hash_cols_sql = ", ".join(_RUN_ARTIFACT_HASH_COLUMNS)
    existing_hashes = conn.execute(
        f"SELECT {hash_cols_sql} FROM bar_evaluation_runs WHERE identity_hash=?",
        [identity_hash],
    ).fetchone()
    assert existing_hashes is not None
    for idx, col in enumerate(_RUN_ARTIFACT_HASH_COLUMNS):
        if _canonical_compare_token(existing_hashes[idx], kind="str") != _canonical_compare_token(run_record[col], kind="str"):
            raise BarEvaluationConflictError(
                identity_hash,
                f"Conflict for identity_hash={identity_hash} in bar_evaluation_runs "
                f"column={col}: existing={existing_hashes[idx]!r} != incoming={run_record[col]!r}",
            )


def _insert_run_row(conn: duckdb.DuckDBPyConnection, run_record: dict[str, Any]) -> None:
    placeholders = ", ".join("?" for _ in _RUN_ALL_COLUMNS)
    conn.execute(
        f"INSERT INTO bar_evaluation_runs ({', '.join(_RUN_ALL_COLUMNS)}) VALUES ({placeholders})",
        [run_record[col] for col in _RUN_ALL_COLUMNS],
    )


def _insert_ic_rows(conn: duckdb.DuckDBPyConnection, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in _IC_TABLE_COLUMNS)
    conn.executemany(
        f"INSERT INTO bar_evaluation_ic ({', '.join(_IC_TABLE_COLUMNS)}) VALUES ({placeholders})",
        rows,
    )


def _insert_quantile_rows(conn: duckdb.DuckDBPyConnection, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in _QUANTILE_TABLE_COLUMNS)
    conn.executemany(
        f"INSERT INTO bar_evaluation_quantiles ({', '.join(_QUANTILE_TABLE_COLUMNS)}) VALUES ({placeholders})",
        rows,
    )


def _insert_cross_section_rows(conn: duckdb.DuckDBPyConnection, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in _CROSS_SECTION_TABLE_COLUMNS)
    conn.executemany(
        f"INSERT INTO bar_evaluation_cross_sections ({', '.join(_CROSS_SECTION_TABLE_COLUMNS)}) VALUES ({placeholders})",
        rows,
    )


def _validate_db_target_path(db_path: str | Path) -> Path:
    if not isinstance(db_path, (str, Path)) or not str(db_path).strip():
        raise BarEvaluationPreconditionError("results_db path must be a nonempty path")
    resolved = Path(db_path).expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise BarEvaluationPreconditionError(f"results_db path is a directory: {resolved}")
    if resolved.exists() and not resolved.is_file():
        raise BarEvaluationPreconditionError(f"results_db is not a regular file: {resolved}")
    try:
        _reject_sqlite_file(resolved)
    except ValueError as exc:
        raise BarEvaluationPreconditionError(str(exc)) from exc
    if resolved.exists():
        # Reject auto-detected CSV/JSON before DuckDB can open them as in-memory catalogs.
        with resolved.open("rb") as handle:
            if handle.read(12)[8:12] != b"DUCK":
                raise BarEvaluationPreconditionError(f"results_db is not a native DuckDB file: {resolved}")
    return resolved


def ingest_evaluation(
    db_path: str | Path,
    evaluation_dir: str | Path | None = None,
    *,
    artifacts: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    run_id: str | None = None,
    retry_interval_sec: float = 0.2,
) -> BarEvaluationIngestReceipt:
    """Verify published evaluation artifacts and atomically ingest into DuckDB.

    Idempotent on identical `identity_hash` content (`status='noop'`).
    All four Harness SHA256 digests are required via artifacts or expected_hashes.
    Raises `BarEvaluationConflictError` (a `ValueError` subclass) on conflicting
    content for an existing `identity_hash` without modifying any table.
    """
    resolved_db = _validate_db_target_path(db_path)
    paths, hashes = _resolve_artifact_paths(evaluation_dir, artifacts, expected_hashes)
    run_record, ic_tuples, q_tuples, cs_tuples = _validate_and_prepare_payload(paths, hashes, run_id)

    identity_hash = run_record["identity_hash"]
    with acquire_exclusive_write_conn(resolved_db, retry_interval_sec=retry_interval_sec) as session:
        conn = session.conn
        active = conn.execute("SELECT current_database()").fetchone()[0]
        databases = conn.execute("PRAGMA database_list").fetchall()
        if not any(name == active and filename and Path(filename).resolve() == resolved_db
                   for _, name, filename in databases):
            raise BarEvaluationPreconditionError(
                f"results_db connection must use the requested persistent DuckDB file: {resolved_db}")
        _validate_db_target_path(resolved_db)
        # Reject a concurrent retry replacing files while we parsed them or waited for the lock.
        for key, path in paths.items():
            if not path.is_file() or sha256_file(path) != hashes[key]:
                raise BarEvaluationIntegrityError(f"Published artifact changed during ingestion: {path}")
        conn.execute("BEGIN TRANSACTION")
        try:
            _create_tables(conn)
            exists = conn.execute(
                "SELECT 1 FROM bar_evaluation_runs WHERE identity_hash=?",
                [identity_hash],
            ).fetchone()
            if exists is not None:
                _verify_existing_run_matches(conn, run_record, ic_tuples, q_tuples, cs_tuples)
                ingested_at = conn.execute(
                    "SELECT ingested_at_utc FROM bar_evaluation_runs WHERE identity_hash=?",
                    [identity_hash],
                ).fetchone()[0]
                conn.execute("COMMIT")
                return BarEvaluationIngestReceipt(
                    db_path=resolved_db,
                    identity_hash=identity_hash,
                    status="noop",
                    analysis_mode=run_record["analysis_mode"],
                    ic_rows=len(ic_tuples),
                    quantile_rows=len(q_tuples),
                    cross_section_rows=len(cs_tuples),
                    ingested_at_utc=_format_utc_iso(ingested_at) or "",
                )

            _insert_run_row(conn, run_record)
            _insert_ic_rows(conn, ic_tuples)
            _insert_quantile_rows(conn, q_tuples)
            _insert_cross_section_rows(conn, cs_tuples)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    return BarEvaluationIngestReceipt(
        db_path=resolved_db,
        identity_hash=identity_hash,
        status="inserted",
        analysis_mode=run_record["analysis_mode"],
        ic_rows=len(ic_tuples),
        quantile_rows=len(q_tuples),
        cross_section_rows=len(cs_tuples),
        ingested_at_utc=_format_utc_iso(run_record["ingested_at_utc"]) or "",
    )


@contextmanager
def _open_read_only(db_path: str | Path) -> Iterator[duckdb.DuckDBPyConnection]:
    if not isinstance(db_path, (str, Path)) or not str(db_path).strip():
        raise BarEvaluationPreconditionError("results_db path must be a nonempty path")
    resolved = Path(db_path).expanduser().resolve()
    if not resolved.exists():
        raise BarEvaluationPreconditionError(f"results_db does not exist: {resolved}")
    if not resolved.is_file():
        raise BarEvaluationPreconditionError(f"results_db is not a regular file: {resolved}")
    _validate_db_target_path(resolved)
    conn = duckdb.connect(str(resolved), read_only=True)
    try:
        yield conn
    finally:
        conn.close()


def _require_bar_evaluation_tables(conn: duckdb.DuckDBPyConnection) -> None:
    existing = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    missing = [name for name in BAR_EVALUATION_TABLES if name not in existing]
    if missing:
        raise BarEvaluationPreconditionError(
            f"Missing bar evaluation table(s): {missing}; run time-series-analysis --results-db first"
        )


def _decode_run_row(columns: Sequence[str], row: tuple[Any, ...]) -> dict[str, Any]:
    record = dict(zip(columns, row))
    for json_col in ("input_file_hashes", "data_identity", "exclusion_summary"):
        raw = record.get(json_col)
        if isinstance(raw, str):
            record[json_col] = json.loads(raw)
    freq_raw = record.get("frequency")
    if isinstance(freq_raw, str) and freq_raw.startswith("["):
        record["frequencies"] = json.loads(freq_raw)
    elif isinstance(freq_raw, str):
        record["frequencies"] = [freq_raw]
    else:
        record["frequencies"] = []
    record["ingested_at_utc"] = _format_utc_iso(record.get("ingested_at_utc"))
    return _json_safe(record)


def _decode_table_rows(
    columns: Sequence[str],
    rows: list[tuple[Any, ...]],
    *,
    json_cols: frozenset[str] = frozenset(),
    ts_cols: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(zip(columns, row))
        for col in json_cols:
            raw = item.get(col)
            if isinstance(raw, str):
                item[col] = json.loads(raw)
        for col in ts_cols:
            item[col] = _format_utc_iso(item.get(col))
        result.append(_json_safe(item))
    return result


def _frequency_matches(run_frequencies: list[str], target: str) -> bool:
    if target in run_frequencies:
        return True
    parsed_target = Frequency.parse(target)
    if not parsed_target.is_valid:
        return False
    for freq in run_frequencies:
        parsed = Frequency.parse(freq)
        if parsed.is_valid and parsed.canonical == parsed_target.canonical:
            return True
    return False


def _query_runs_on_conn(
    conn: duckdb.DuckDBPyConnection,
    *,
    identity_hash: str | None = None,
    analysis_mode: str | None = None,
    frequency: str | None = None,
    calendar_id: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> list[dict[str, Any]]:
    _require_bar_evaluation_tables(conn)
    if analysis_mode is not None and analysis_mode not in VALID_ANALYSIS_MODES:
        raise BarEvaluationPreconditionError(
            f"analysis_mode filter must be one of {VALID_ANALYSIS_MODES}, got {analysis_mode!r}"
        )
    clauses: list[str] = []
    params: list[Any] = []
    if identity_hash is not None:
        clauses.append("identity_hash = ?")
        params.append(identity_hash)
    if analysis_mode is not None:
        clauses.append("analysis_mode = ?")
        params.append(analysis_mode)
    if calendar_id is not None:
        clauses.append("calendar_id = ?")
        params.append(calendar_id)
    start_ts = _parse_filter_timestamp(start_utc, is_end=False) if start_utc is not None else None
    end_ts = _parse_filter_timestamp(end_utc, is_end=True) if end_utc is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise BarEvaluationPreconditionError("start_utc must not follow end_utc")
    if start_ts is not None:
        clauses.append("ingested_at_utc >= ?")
        params.append(start_ts.to_pydatetime())
    if end_ts is not None:
        clauses.append("ingested_at_utc <= ?")
        params.append(end_ts.to_pydatetime())

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        f"SELECT {', '.join(_RUN_ALL_COLUMNS)} FROM bar_evaluation_runs "
        f"{where_sql} ORDER BY ingested_at_utc DESC, identity_hash ASC"
    )
    rows = conn.execute(sql, params).fetchall()
    decoded = [_decode_run_row(_RUN_ALL_COLUMNS, row) for row in rows]
    if frequency is not None:
        decoded = [item for item in decoded if _frequency_matches(item.get("frequencies", []), frequency)]
    return decoded


def query_bar_evaluation_runs(
    db_path: str | Path,
    *,
    identity_hash: str | None = None,
    analysis_mode: str | None = None,
    frequency: str | None = None,
    calendar_id: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
) -> list[dict[str, Any]]:
    """Read-only query of `bar_evaluation_runs` matching the supplied filters."""
    with _open_read_only(db_path) as conn:
        return _query_runs_on_conn(
            conn,
            identity_hash=identity_hash,
            analysis_mode=analysis_mode,
            frequency=frequency,
            calendar_id=calendar_id,
            start_utc=start_utc,
            end_utc=end_utc,
        )


def query_bar_evaluation_detail(
    db_path: str | Path,
    identity_hash: str,
) -> dict[str, Any]:
    """Read-only fetch of one evaluation run and its IC/quantile/cross-section tables."""
    if not isinstance(identity_hash, str) or not identity_hash.strip():
        raise BarEvaluationPreconditionError("identity_hash must be a nonempty string")
    with _open_read_only(db_path) as conn:
        runs = _query_runs_on_conn(conn, identity_hash=identity_hash)
        if not runs:
            raise LookupError(f"No bar evaluation found for identity_hash={identity_hash!r}")
        run = runs[0]
        ic_rows = conn.execute(
            f"SELECT {', '.join(_IC_TABLE_COLUMNS)} FROM bar_evaluation_ic "
            "WHERE identity_hash=? ORDER BY row_index",
            [identity_hash],
        ).fetchall()
        q_rows = conn.execute(
            f"SELECT {', '.join(_QUANTILE_TABLE_COLUMNS)} FROM bar_evaluation_quantiles "
            "WHERE identity_hash=? ORDER BY row_index",
            [identity_hash],
        ).fetchall()
        cs_rows = conn.execute(
            f"SELECT {', '.join(_CROSS_SECTION_TABLE_COLUMNS)} FROM bar_evaluation_cross_sections "
            "WHERE identity_hash=? ORDER BY row_index",
            [identity_hash],
        ).fetchall()

    return {
        "run": run,
        "ic_by_horizon": _decode_table_rows(_IC_TABLE_COLUMNS, ic_rows, json_cols=_IC_JSON_COLUMNS),
        "quantile_stats": _decode_table_rows(_QUANTILE_TABLE_COLUMNS, q_rows),
        "cross_sections": _decode_table_rows(
            _CROSS_SECTION_TABLE_COLUMNS,
            cs_rows,
            json_cols=_CS_JSON_COLUMNS,
            ts_cols=frozenset({"signal_timestamp"}),
        ),
    }


def query_bar_evaluations(
    db_path: str | Path,
    *,
    identity_hash: str | None = None,
    analysis_mode: str | None = None,
    frequency: str | None = None,
    calendar_id: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    include_cross_sections: bool = True,
) -> dict[str, Any]:
    """Read-only query returning matching runs and associated child records."""
    with _open_read_only(db_path) as conn:
        runs = _query_runs_on_conn(
            conn,
            identity_hash=identity_hash,
            analysis_mode=analysis_mode,
            frequency=frequency,
            calendar_id=calendar_id,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        hashes = [run["identity_hash"] for run in runs]
        if not hashes:
            return {
                "run_count": 0,
                "runs": [],
                "ic_by_horizon": [],
                "quantile_stats": [],
                "cross_sections": [],
            }
        placeholders = ", ".join("?" for _ in hashes)
        ic_rows = conn.execute(
            f"SELECT {', '.join(_IC_TABLE_COLUMNS)} FROM bar_evaluation_ic "
            f"WHERE identity_hash IN ({placeholders}) ORDER BY identity_hash, row_index",
            hashes,
        ).fetchall()
        q_rows = conn.execute(
            f"SELECT {', '.join(_QUANTILE_TABLE_COLUMNS)} FROM bar_evaluation_quantiles "
            f"WHERE identity_hash IN ({placeholders}) ORDER BY identity_hash, row_index",
            hashes,
        ).fetchall()
        cs_rows = (
            conn.execute(
                f"SELECT {', '.join(_CROSS_SECTION_TABLE_COLUMNS)} FROM bar_evaluation_cross_sections "
                f"WHERE identity_hash IN ({placeholders}) ORDER BY identity_hash, row_index",
                hashes,
            ).fetchall()
            if include_cross_sections
            else []
        )
    return {
        "run_count": len(runs),
        "runs": runs,
        "ic_by_horizon": _decode_table_rows(_IC_TABLE_COLUMNS, ic_rows, json_cols=_IC_JSON_COLUMNS),
        "quantile_stats": _decode_table_rows(_QUANTILE_TABLE_COLUMNS, q_rows),
        "cross_sections": _decode_table_rows(
            _CROSS_SECTION_TABLE_COLUMNS,
            cs_rows,
            json_cols=_CS_JSON_COLUMNS,
            ts_cols=frozenset({"signal_timestamp"}),
        ),
    }


def get_bar_evaluation_filter_options(db_path: str | Path) -> dict[str, list[str]]:
    """Return sorted distinct filter values for the read-only UI."""
    runs = query_bar_evaluation_runs(db_path)
    modes = sorted({r["analysis_mode"] for r in runs if r.get("analysis_mode")})
    frequencies = sorted({f for r in runs for f in r.get("frequencies", []) if f})
    calendars = sorted({r["calendar_id"] for r in runs if r.get("calendar_id")})
    return {
        "analysis_modes": modes,
        "frequencies": frequencies,
        "calendar_ids": calendars,
    }


def summarize_bar_evaluation_connection(
    conn: duckdb.DuckDBPyConnection,
    *,
    table_names: set[str] | None = None,
) -> dict[str, Any]:
    """Read-only summary of `bar_evaluation_*` tables on an existing connection."""
    existing = table_names if table_names is not None else {
        row[0] for row in conn.execute("SHOW TABLES").fetchall()
    }
    tables_info: dict[str, dict[str, Any]] = {}
    for name in BAR_EVALUATION_TABLES:
        if name in existing:
            count = int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            tables_info[name] = {"exists": True, "row_count": count}
        else:
            tables_info[name] = {"exists": False, "row_count": None}

    latest_ingested_at_utc: str | None = None
    if tables_info["bar_evaluation_runs"]["exists"]:
        row = conn.execute("SELECT MAX(ingested_at_utc) FROM bar_evaluation_runs").fetchone()
        if row and row[0] is not None:
            latest_ingested_at_utc = _format_utc_iso(row[0])

    return {
        "all_tables_exist": all(tables_info[name]["exists"] for name in BAR_EVALUATION_TABLES),
        "any_table_exists": any(tables_info[name]["exists"] for name in BAR_EVALUATION_TABLES),
        "tables": tables_info,
        "run_count": int(tables_info["bar_evaluation_runs"]["row_count"] or 0),
        "latest_ingested_at_utc": latest_ingested_at_utc,
    }


def summarize_bar_evaluation_db(db_path: str | Path) -> dict[str, Any]:
    """Read-only summary of `bar_evaluation_*` tables for a DuckDB path."""
    with _open_read_only(db_path) as conn:
        return summarize_bar_evaluation_connection(conn)
