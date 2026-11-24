"""Resolve and verify evaluation artifact files and DuckDB paths."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from backtest_and_analysis.time_series.contracts import sha256_file
from utils._storage_compat import _legacy_binding
from utils.data_store import _reject_sqlite_file

from .normalization import (
    _coerce_optional_bool,
    _coerce_optional_float,
    _coerce_optional_int,
    _coerce_optional_json_str,
    _is_missing,
)
from .schema import _SHA256_RE, BarEvaluationIntegrityError, BarEvaluationPreconditionError


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
