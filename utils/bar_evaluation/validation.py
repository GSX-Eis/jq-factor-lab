"""Validate published artifacts and prepare DuckDB rows."""
from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_and_analysis.time_series.contracts import ResultIdentity, sha256_file
from backtest_and_analysis.time_series.evaluator import (
    CS_COLUMNS,
    CS_IC_COLUMNS,
    CS_QUANTILE_COLUMNS,
    TS_IC_COLUMNS,
    TS_QUANTILE_COLUMNS,
)
from utils._storage_compat import _legacy_binding
from utils.data_store import _reject_sqlite_file

from .inference_validation import prepare_inference
from .labels_validation import check_labels
from .normalization import (
    _canonical_json,
    _coerce_optional_bool,
    _coerce_optional_float,
    _coerce_optional_int,
    _coerce_optional_json_str,
    _coerce_optional_str,
    _is_missing,
    _json_safe,
    _parse_utc_timestamp,
)
from .schema import (
    _CROSS_SECTION_TABLE_COLUMNS,
    _CS_BOOL_COLUMNS,
    _CS_FLOAT_COLUMNS,
    _CS_INT_COLUMNS,
    _CS_JSON_COLUMNS,
    _CSV_STRING_DTYPES,
    _IC_BOOL_COLUMNS,
    _IC_FLOAT_COLUMNS,
    _IC_INT_COLUMNS,
    _IC_JSON_COLUMNS,
    _IC_TABLE_COLUMNS,
    _QUANTILE_BOOL_COLUMNS,
    _QUANTILE_FLOAT_COLUMNS,
    _QUANTILE_INT_COLUMNS,
    _QUANTILE_TABLE_COLUMNS,
    _SHA256_RE,
    SUPPORTED_RESULT_SCHEMA_VERSIONS,
    VALID_ANALYSIS_MODES,
    BarEvaluationIntegrityError,
    BarEvaluationPreconditionError,
)


def _published_artifact_sha256(path: Path) -> str:
    return _legacy_binding("utils.bar_evaluation_store", "sha256_file", sha256_file)(path)


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
        computed = _published_artifact_sha256(file_path)
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
            if col not in json_row:
                raise BarEvaluationIntegrityError(f"summary.json {table_label}[{idx}] missing {col}")
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


def _summary_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate object key")
    return result


def _summary_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number {value}")
    return number


def _validate_and_prepare_payload(
    paths: dict[str, Path],
    hashes: dict[str, str],
    run_id: str | None,
) -> tuple[dict[str, Any], list[tuple[Any, ...]], list[tuple[Any, ...]], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    try:
        summary = json.loads(
            paths["summary"].read_text(encoding="utf-8"),
            object_pairs_hook=_summary_object, parse_float=_summary_float, parse_constant=_summary_float,
        )
    except (OSError, ValueError) as exc:
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

    expected_ic_cols = TS_IC_COLUMNS if analysis_mode == "time_series" else CS_IC_COLUMNS
    expected_q_cols = TS_QUANTILE_COLUMNS if analysis_mode == "time_series" else CS_QUANTILE_COLUMNS

    try:
        csv_options = dict(dtype=_CSV_STRING_DTYPES, keep_default_na=False, na_values=[""])
        ic_df = pd.read_csv(paths["ic_by_horizon"], **csv_options)
        q_df = pd.read_csv(paths["quantile_stats"], **csv_options)
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
    label_streams, minimum, instruments, frequencies = check_labels(paths["per_bar_labels"], summary)
    inference, split_tuples = prepare_inference(
        summary, identity_hash, label_streams, minimum, instruments=instruments, frequencies=frequencies,
    )

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

    frequency_str = _resolve_frequency_field(data_identity_payload, ic_records)
    ingested_at = dt.datetime.now(dt.timezone.utc)

    run_record: dict[str, Any] = {
        "identity_hash": identity_hash,
        "analysis_mode": analysis_mode,
        "frequency": frequency_str,
        "view": _coerce_optional_str(cal_identity.get("view")),
        "boundary_policy": _coerce_optional_str(cal_identity.get("boundary_policy")),
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
        **inference,
    }
    return run_record, ic_tuples, q_tuples, cs_tuples, split_tuples


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
