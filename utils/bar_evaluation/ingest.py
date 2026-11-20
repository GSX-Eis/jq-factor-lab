"""Exclusive, atomic and idempotent DuckDB evaluation ingest."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import duckdb

from utils._storage_compat import _legacy_binding
from utils.duckdb_results_writer import acquire_exclusive_write_conn

from .normalization import _canonical_compare_token, _format_utc_iso
from .schema import (
    _CROSS_SECTION_TABLE_COLUMNS,
    _CS_BOOL_COLUMNS,
    _CS_FLOAT_COLUMNS,
    _CS_INT_COLUMNS,
    _CS_JSON_COLUMNS,
    _IC_BOOL_COLUMNS,
    _IC_FLOAT_COLUMNS,
    _IC_INT_COLUMNS,
    _IC_JSON_COLUMNS,
    _IC_TABLE_COLUMNS,
    _QUANTILE_BOOL_COLUMNS,
    _QUANTILE_FLOAT_COLUMNS,
    _QUANTILE_INT_COLUMNS,
    _QUANTILE_TABLE_COLUMNS,
    _RUN_ALL_COLUMNS,
    _RUN_ARTIFACT_HASH_COLUMNS,
    _RUN_BOOL_COLUMNS,
    _RUN_JSON_COLUMNS,
    _RUN_SEMANTIC_COLUMNS,
    _RUN_V2_COLUMN_TYPES,
    _SPLIT_BOOL_COLUMNS,
    _SPLIT_FLOAT_COLUMNS,
    _SPLIT_INT_COLUMNS,
    _SPLIT_JSON_COLUMNS,
    _SPLIT_TABLE,
    _SPLIT_TABLE_COLUMNS,
    BarEvaluationConflictError,
    BarEvaluationIngestReceipt,
    BarEvaluationIntegrityError,
    BarEvaluationPreconditionError,
    _catalog_columns,
    _column_kind,
    _create_split_table,
    _create_tables,
    _require_write_schema,
)
from .validation import (
    _published_artifact_sha256,
    _resolve_artifact_paths,
    _validate_and_prepare_payload,
    _validate_db_target_path,
)


_LEGACY_MODULE = "utils.bar_evaluation_store"


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
    split_tuples: list[tuple[Any, ...]],
) -> None:
    identity_hash = run_record["identity_hash"]
    run_cols_sql = ", ".join(_RUN_SEMANTIC_COLUMNS)
    existing_run = conn.execute(
        f"SELECT {run_cols_sql} FROM bar_evaluation_runs WHERE identity_hash=?",
        [identity_hash],
    ).fetchone()
    assert existing_run is not None

    for idx, col in enumerate(_RUN_SEMANTIC_COLUMNS):
        kind = _column_kind(col, frozenset(), frozenset(), _RUN_BOOL_COLUMNS, _RUN_JSON_COLUMNS)
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
    existing_splits = conn.execute(
        f"SELECT {', '.join(_SPLIT_TABLE_COLUMNS)} FROM {_SPLIT_TABLE} "
        "WHERE identity_hash=? ORDER BY row_index",
        [identity_hash],
    ).fetchall()
    _compare_table_rows(
        identity_hash, _SPLIT_TABLE, _SPLIT_TABLE_COLUMNS, existing_splits, split_tuples,
        int_cols=_SPLIT_INT_COLUMNS, float_cols=_SPLIT_FLOAT_COLUMNS,
        bool_cols=_SPLIT_BOOL_COLUMNS, json_cols=_SPLIT_JSON_COLUMNS,
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


def _insert_split_rows(conn: duckdb.DuckDBPyConnection, rows: list[tuple[Any, ...]]) -> None:
    if rows:
        placeholders = ", ".join("?" for _ in _SPLIT_TABLE_COLUMNS)
        conn.executemany(
            f"INSERT INTO {_SPLIT_TABLE} ({', '.join(_SPLIT_TABLE_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )


@contextmanager
def _write_transaction(
    db_path: Path, retry_interval_sec: float = 0.2,
) -> Iterator[duckdb.DuckDBPyConnection]:
    acquire = _legacy_binding(_LEGACY_MODULE, "acquire_exclusive_write_conn", acquire_exclusive_write_conn)
    with acquire(db_path, retry_interval_sec=retry_interval_sec) as session:
        conn = session.conn
        active = conn.execute("SELECT current_database()").fetchone()[0]
        if not any(name == active and filename and Path(filename).resolve() == db_path
                   for _, name, filename in conn.execute("PRAGMA database_list").fetchall()):
            raise BarEvaluationPreconditionError(
                f"results_db connection must use the requested persistent DuckDB file: {db_path}")
        _legacy_binding(_LEGACY_MODULE, "_validate_db_target_path", _validate_db_target_path)(db_path)
        conn.execute("BEGIN TRANSACTION")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise


def migrate_bar_evaluation_db(db_path: str | Path) -> None:
    """Explicitly upgrade an existing v1 store while preserving its historical rows."""
    resolved_db = _validate_db_target_path(db_path)
    if not resolved_db.is_file():
        raise BarEvaluationPreconditionError(f"results_db does not exist: {resolved_db}")
    with _write_transaction(resolved_db) as conn:
        _require_write_schema(conn, _catalog_columns(conn), legacy_only=True)
        for column, kind in _RUN_V2_COLUMN_TYPES.items():
            conn.execute(f"ALTER TABLE bar_evaluation_runs ADD COLUMN IF NOT EXISTS {column} {kind}")
        _create_split_table(conn)
        _require_write_schema(conn, _catalog_columns(conn))


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
    resolved_db = _legacy_binding(_LEGACY_MODULE, "_validate_db_target_path", _validate_db_target_path)(db_path)
    paths, hashes = _resolve_artifact_paths(evaluation_dir, artifacts, expected_hashes)
    prepare_payload = _legacy_binding(_LEGACY_MODULE, "_validate_and_prepare_payload", _validate_and_prepare_payload)
    run_record, ic_tuples, q_tuples, cs_tuples, split_tuples = prepare_payload(paths, hashes, run_id)

    identity_hash = run_record["identity_hash"]
    with _write_transaction(resolved_db, retry_interval_sec) as conn:
        # Reject a concurrent retry replacing files while we parsed them or waited for the lock.
        for key, path in paths.items():
            if not path.is_file() or _published_artifact_sha256(path) != hashes[key]:
                raise BarEvaluationIntegrityError(f"Published artifact changed during ingestion: {path}")
        catalog = _catalog_columns(conn)
        if not catalog:
            _create_tables(conn)
            catalog = _catalog_columns(conn)
        _require_write_schema(conn, catalog)
        existing = conn.execute(
            "SELECT ingested_at_utc FROM bar_evaluation_runs WHERE identity_hash=?",
            [identity_hash],
        ).fetchone()
        if existing is not None:
            _verify_existing_run_matches(conn, run_record, ic_tuples, q_tuples, cs_tuples, split_tuples)
            status, ingested_at = "noop", existing[0]
        else:
            _legacy_binding(_LEGACY_MODULE, "_insert_run_row", _insert_run_row)(conn, run_record)
            _legacy_binding(_LEGACY_MODULE, "_insert_ic_rows", _insert_ic_rows)(conn, ic_tuples)
            _legacy_binding(_LEGACY_MODULE, "_insert_quantile_rows", _insert_quantile_rows)(conn, q_tuples)
            _legacy_binding(_LEGACY_MODULE, "_insert_cross_section_rows", _insert_cross_section_rows)(conn, cs_tuples)
            _legacy_binding(_LEGACY_MODULE, "_insert_split_rows", _insert_split_rows)(conn, split_tuples)
            status, ingested_at = "inserted", run_record["ingested_at_utc"]

    return BarEvaluationIngestReceipt(
        db_path=resolved_db,
        identity_hash=identity_hash,
        status=status,
        analysis_mode=run_record["analysis_mode"],
        ic_rows=len(ic_tuples),
        quantile_rows=len(q_tuples),
        cross_section_rows=len(cs_tuples),
        ingested_at_utc=_format_utc_iso(ingested_at) or "",
        split_rows=len(split_tuples),
    )
