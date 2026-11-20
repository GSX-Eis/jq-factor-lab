"""Read-only evaluation queries and monitor summaries."""
from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import duckdb

from utils.time_axis import Frequency

from .normalization import (
    _format_utc_iso,
    _json_safe,
    _parse_filter_timestamp,
)
from .schema import (
    _BASE_TABLES,
    _CALIBRATION_STATUSES,
    _CROSS_SECTION_TABLE_COLUMNS,
    _CS_JSON_COLUMNS,
    _IC_JSON_COLUMNS,
    _IC_TABLE_COLUMNS,
    _QUANTILE_TABLE_COLUMNS,
    _RUN_ALL_COLUMNS,
    _RUN_JSON_COLUMNS,
    _RUN_V2_COLUMN_TYPES,
    _SPLIT_JSON_COLUMNS,
    _SPLIT_NAMES,
    _SPLIT_TABLE,
    _SPLIT_TABLE_COLUMNS,
    BAR_EVALUATION_TABLES,
    VALID_ANALYSIS_MODES,
    BarEvaluationPreconditionError,
    _catalog_columns,
)
from .validation import (
    _validate_db_target_path,
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
    missing = [name for name in _BASE_TABLES if name not in existing]
    if missing:
        raise BarEvaluationPreconditionError(
            f"Missing bar evaluation table(s): {missing}; run time-series-analysis --results-db first"
        )


def _decode_run_row(columns: Sequence[str], row: tuple[Any, ...]) -> dict[str, Any]:
    record = dict(zip(columns, row))
    for json_col in _RUN_JSON_COLUMNS:
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
    safe = _json_safe(record)
    safe["split_config_json"] = record.get("split_config_json")
    return safe


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
        safe = _json_safe(item)
        if "fit_identity_json" in item:
            safe["fit_identity_json"] = item["fit_identity_json"]
        result.append(safe)
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
    run_columns = _catalog_columns(conn).get("bar_evaluation_runs", {})
    projections = [
        col if col not in _RUN_V2_COLUMN_TYPES or col in run_columns else f"NULL AS {col}"
        for col in _RUN_ALL_COLUMNS
    ]
    sql = (
        f"SELECT {', '.join(projections)} FROM bar_evaluation_runs "
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
    """Read-only fetch of one evaluation run and its child tables."""
    if not isinstance(identity_hash, str) or not identity_hash.strip():
        raise BarEvaluationPreconditionError("identity_hash must be a nonempty string")
    detail = query_bar_evaluations(db_path, identity_hash=identity_hash)
    runs = detail.pop("runs")
    if not runs:
        raise LookupError(f"No bar evaluation found for identity_hash={identity_hash!r}")
    detail.pop("run_count")
    return {"run": runs[0], **detail}


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
    split: str | None = None,
    calibration_status: str | None = None,
    descriptive_stats_usable: bool | None = None,
    significance_interpretation_usable: bool | None = None,
    has_trial_ledger: bool | None = None,
) -> dict[str, Any]:
    """Filter runs and split rows by their inference fields. Legacy tables remain run-scoped."""
    if split is not None and split not in _SPLIT_NAMES:
        raise BarEvaluationPreconditionError(f"split must be one of {_SPLIT_NAMES}, got {split!r}")
    if calibration_status is not None and calibration_status not in _CALIBRATION_STATUSES:
        raise BarEvaluationPreconditionError(f"calibration_status must be one of {_CALIBRATION_STATUSES}")
    filters = {
        "split": split,
        "calibration_status": calibration_status,
        "descriptive_stats_usable": descriptive_stats_usable,
        "significance_interpretation_usable": significance_interpretation_usable,
    }
    filters = {key: value for key, value in filters.items() if value is not None}

    def matches(row: dict[str, Any]) -> bool:
        return all(row.get(key) == value for key, value in filters.items()) and (
            frequency is None or _frequency_matches([row["frequency"]], frequency)
        )

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
        if has_trial_ledger is not None:
            runs = [run for run in runs if (run["trial_ledger_json"] is not None) == has_trial_ledger]
        hashes = [run["identity_hash"] for run in runs]
        split_rows: list[dict[str, Any]] = []
        if hashes and _SPLIT_TABLE in _catalog_columns(conn):
            rows = conn.execute(
                f"SELECT {', '.join(_SPLIT_TABLE_COLUMNS)} FROM {_SPLIT_TABLE} "
                "WHERE identity_hash IN (SELECT unnest(?)) ORDER BY identity_hash, row_index",
                [hashes],
            ).fetchall()
            split_rows = _decode_table_rows(_SPLIT_TABLE_COLUMNS, rows, json_cols=_SPLIT_JSON_COLUMNS)
        if filters:
            with_splits = {row["identity_hash"] for row in split_rows}
            split_rows = [row for row in split_rows if matches(row)]
            matching = {row["identity_hash"] for row in split_rows}
            runs = [
                run for run in runs if run["identity_hash"] in matching
                or (
                    run["identity_hash"] not in with_splits and split is None
                    and any(matches(row) for row in run["calibration_summary_json"] or [])
                )
            ]
            hashes = [run["identity_hash"] for run in runs]
        if not hashes:
            return {
                "run_count": 0,
                "runs": [],
                "ic_by_horizon": [],
                "quantile_stats": [],
                "cross_sections": [],
                "split_metrics": [],
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
        "split_metrics": split_rows,
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
        "all_tables_exist": all(tables_info[name]["exists"] for name in _BASE_TABLES),
        "any_table_exists": any(tables_info[name]["exists"] for name in BAR_EVALUATION_TABLES),
        "tables": tables_info,
        "run_count": int(tables_info["bar_evaluation_runs"]["row_count"] or 0),
        "latest_ingested_at_utc": latest_ingested_at_utc,
    }


def summarize_bar_evaluation_db(db_path: str | Path) -> dict[str, Any]:
    """Read-only summary of `bar_evaluation_*` tables for a DuckDB path."""
    with _open_read_only(db_path) as conn:
        return summarize_bar_evaluation_connection(conn)
