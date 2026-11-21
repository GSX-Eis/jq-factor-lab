"""Read-only evaluation queries and monitor summaries."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import duckdb

from utils.time_axis import Frequency

from .normalization import (
    _factor_id_from_config,
    _factor_id_from_ledger,
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
    _RUN_OPTIONAL_COLUMN_TYPES,
    _SPLIT_JSON_COLUMNS,
    _SPLIT_NAMES,
    _SPLIT_TABLE,
    _SPLIT_TABLE_COLUMNS,
    BAR_EVALUATION_TABLES,
    VALID_ANALYSIS_MODES,
    BarEvaluationPreconditionError,
    _catalog_columns,
)
from .validation import _validate_db_target_path


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


def _stored_frequencies(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return json.loads(raw) if raw.startswith("[") else [raw]


def _decode_run_row(columns: Sequence[str], row: tuple[Any, ...]) -> dict[str, Any]:
    record = dict(zip(columns, row))
    for json_col in _RUN_JSON_COLUMNS:
        raw = record.get(json_col)
        if isinstance(raw, str):
            record[json_col] = json.loads(raw)
    record["frequencies"] = _stored_frequencies(record.get("frequency"))
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


def _legacy_factor_id(ledger_json: str | None, summary_path: str, summary_sha256: str) -> str | None:
    factor_id = _factor_id_from_ledger(json.loads(ledger_json)) if ledger_json is not None else None
    if factor_id is not None:
        return factor_id
    path = Path(summary_path)
    try:
        if not path.is_file():
            return None
        evidence = path.read_bytes()
        if sha256(evidence).hexdigest() != summary_sha256:
            return None
        config = json.loads(evidence)["config"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return _factor_id_from_config(config) if isinstance(config, dict) else None


def _validate_page_filters(factor_name: str | None, limit: int | None, offset: int) -> None:
    if limit is not None and (type(limit) is not int or limit < 1):
        raise BarEvaluationPreconditionError("limit must be a positive integer")
    if type(offset) is not int or offset < 0:
        raise BarEvaluationPreconditionError("offset must be a non-negative integer")
    if factor_name is not None and (not isinstance(factor_name, str) or not factor_name.strip()):
        raise BarEvaluationPreconditionError("factor_name must be nonempty")


def _split_filter_sql(filters: Mapping[str, Any], frequency_function: str | None) -> str:
    clauses = [f"{key} = ?" for key in filters]
    if clauses and frequency_function is not None:
        clauses.append(f"{frequency_function}(frequency)")
    return " AND ".join(clauses)


@contextmanager
def _query_run_page(
    conn: duckdb.DuckDBPyConnection,
    *,
    identity_hash: str | None = None,
    analysis_mode: str | None = None,
    frequency: str | None = None,
    calendar_id: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    has_trial_ledger: bool | None = None,
    factor_name: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    validate_only: bool = False,
    split_filters: Mapping[str, Any] | None = None,
) -> Iterator[tuple[list[dict[str, Any]], str | None]]:
    _require_bar_evaluation_tables(conn)
    if analysis_mode is not None and analysis_mode not in VALID_ANALYSIS_MODES:
        raise BarEvaluationPreconditionError(
            f"analysis_mode filter must be one of {VALID_ANALYSIS_MODES}, got {analysis_mode!r}"
        )
    clauses: list[str] = []
    params: list[Any] = []
    if identity_hash is not None:
        clauses.append("r.identity_hash = ?")
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

    if validate_only:
        yield [], None
        return

    catalog = _catalog_columns(conn)
    run_columns = catalog.get("bar_evaluation_runs", {})
    ledger = "r.trial_ledger_json" if "trial_ledger_json" in run_columns else "NULL"
    if has_trial_ledger is not None:
        clauses.append(f"(NULLIF(trim({ledger}), 'null') IS NOT NULL) = ?")
        params.append(has_trial_ledger)
    # DuckDB shares registered functions across connections to the same file.
    with ExitStack() as functions:
        frequency_function = None
        if frequency is not None:
            def frequency_matches(raw: str) -> bool:
                return _frequency_matches(_stored_frequencies(raw), frequency)

            frequency_function = f"bar_frequency_matches_{uuid4().hex}"
            conn.create_function(frequency_function, frequency_matches, [str], bool)
            functions.callback(conn.remove_function, frequency_function)
            clauses.append(f"{frequency_function}(r.frequency)")
        if factor_name is not None:
            factor_function = f"bar_legacy_factor_id_{uuid4().hex}"
            conn.create_function(factor_function, _legacy_factor_id, [str, str, str], str,
                                 null_handling="special")
            functions.callback(conn.remove_function, factor_function)
            legacy_factor = f"{factor_function}({ledger}, r.summary_path, r.summary_sha256)"
            if "factor_id" in run_columns:
                # The legacy reader may touch the filesystem and must run only for NULL factor IDs.
                clauses.append(f"r.factor_id = ? OR CASE WHEN r.factor_id IS NULL THEN {legacy_factor} = ? END")
                params.append(factor_name)
            else:
                clauses.append(f"{legacy_factor} = ?")
            params.append(factor_name)
        if split_filters:
            split_match: list[str] = []
            if _SPLIT_TABLE in catalog:
                same_run = f"SELECT 1 FROM {_SPLIT_TABLE} AS s WHERE s.identity_hash = r.identity_hash"
                split_match.append(f"EXISTS ({same_run} AND {_split_filter_sql(split_filters, frequency_function)})")
                params.extend(split_filters.values())
            if "split" not in split_filters and "calibration_summary_json" in run_columns:
                legacy_filters = [f"json_extract(entry.value, '$.{key}') = to_json(?)" for key in split_filters]
                if frequency_function is not None:
                    legacy_filters.append(
                        f"{frequency_function}(json_extract_string(entry.value, '$.frequency'))"
                    )
                fallback_match = f"""
                    EXISTS (
                        SELECT 1 FROM json_each(r.calibration_summary_json) AS entry
                        WHERE {' AND '.join(legacy_filters)}
                    )
                """
                if _SPLIT_TABLE in catalog:
                    fallback_match = f"NOT EXISTS ({same_run}) AND {fallback_match}"
                split_match.append(fallback_match)
                params.extend(split_filters.values())
            clauses.append(" OR ".join(split_match) or "FALSE")

        where_sql = "WHERE " + " AND ".join(f"({clause})" for clause in clauses) if clauses else ""
        projections = [
            f"r.{col}" if col not in _RUN_OPTIONAL_COLUMN_TYPES or col in run_columns else f"NULL AS {col}"
            for col in _RUN_ALL_COLUMNS
        ]
        sql = (
            f"SELECT {', '.join(projections)} FROM bar_evaluation_runs AS r "
            f"{where_sql} ORDER BY r.ingested_at_utc DESC, r.identity_hash ASC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(min(limit, 2**63 - 1))
        if offset:
            sql += " OFFSET ?"
            params.append(min(offset, 2**63 - 1))
        rows = conn.execute(sql, params).fetchall()
        yield [_decode_run_row(_RUN_ALL_COLUMNS, row) for row in rows], frequency_function


def _query_runs_on_conn(conn: duckdb.DuckDBPyConnection, **filters: Any) -> list[dict[str, Any]]:
    with _query_run_page(conn, **filters) as (runs, _):
        return runs


def query_bar_evaluation_runs(
    db_path: str | Path,
    *,
    identity_hash: str | None = None,
    analysis_mode: str | None = None,
    frequency: str | None = None,
    calendar_id: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    factor_name: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    validate_only: bool = False,
) -> list[dict[str, Any]]:
    """Read a page of matching runs, ordered by ingestion time and identity."""
    _validate_page_filters(factor_name, limit, offset)
    with _open_read_only(db_path) as conn:
        return _query_runs_on_conn(
            conn,
            identity_hash=identity_hash,
            analysis_mode=analysis_mode,
            frequency=frequency,
            calendar_id=calendar_id,
            start_utc=start_utc,
            end_utc=end_utc,
            factor_name=factor_name,
            limit=limit,
            offset=offset,
            validate_only=validate_only,
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
    factor_name: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Read a page of matching runs and their child rows."""
    _validate_page_filters(factor_name, limit, offset)
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

    with (
        _open_read_only(db_path) as conn,
        _query_run_page(
            conn,
            identity_hash=identity_hash,
            analysis_mode=analysis_mode,
            frequency=frequency,
            calendar_id=calendar_id,
            start_utc=start_utc,
            end_utc=end_utc,
            has_trial_ledger=has_trial_ledger,
            factor_name=factor_name,
            limit=limit,
            offset=offset,
            validate_only=validate_only,
            split_filters=filters,
        ) as (runs, frequency_function),
    ):
        if validate_only:
            return {}
        hashes = [run["identity_hash"] for run in runs]
        split_rows: list[dict[str, Any]] = []
        if hashes and _SPLIT_TABLE in _catalog_columns(conn):
            split_where = f" AND {_split_filter_sql(filters, frequency_function)}" if filters else ""
            rows = conn.execute(
                f"SELECT {', '.join(_SPLIT_TABLE_COLUMNS)} FROM {_SPLIT_TABLE} "
                f"WHERE identity_hash IN (SELECT unnest(?)){split_where} ORDER BY identity_hash, row_index",
                [hashes, *filters.values()],
            ).fetchall()
            split_rows = _decode_table_rows(_SPLIT_TABLE_COLUMNS, rows, json_cols=_SPLIT_JSON_COLUMNS)
        if not hashes:
            return {
                "run_count": 0,
                "runs": [],
                "ic_by_horizon": [],
                "quantile_stats": [],
                "cross_sections": [],
                "split_metrics": [],
            }
        ic_rows = conn.execute(
            f"SELECT {', '.join(_IC_TABLE_COLUMNS)} FROM bar_evaluation_ic "
            "WHERE identity_hash IN (SELECT unnest(?)) ORDER BY identity_hash, row_index",
            [hashes],
        ).fetchall()
        q_rows = conn.execute(
            f"SELECT {', '.join(_QUANTILE_TABLE_COLUMNS)} FROM bar_evaluation_quantiles "
            "WHERE identity_hash IN (SELECT unnest(?)) ORDER BY identity_hash, row_index",
            [hashes],
        ).fetchall()
        cs_rows = (
            conn.execute(
                f"SELECT {', '.join(_CROSS_SECTION_TABLE_COLUMNS)} FROM bar_evaluation_cross_sections "
                "WHERE identity_hash IN (SELECT unnest(?)) ORDER BY identity_hash, row_index",
                [hashes],
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
