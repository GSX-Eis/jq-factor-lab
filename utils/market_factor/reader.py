"""DuckDB query plans, views and lock-free reads of committed shards."""
from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from backtest_and_analysis.time_series.contracts import _utc
from utils.time_axis import Frequency

from .ledger import (
    _read_jsonl,
)
from .schema import (
    IDENTITY_COLUMNS,
    TABLE_COLUMNS,
    VALID_VIEWS,
    VIEW_COLUMNS,
    FactorReadPlan,
    _EPOCH_NS_COLUMNS,
    _require_nonempty_str,
    _validate_session_date_str,
    validate_factor_id,
    validate_sha256_hex,
)


def list_identities(
    identities_path: Path,
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

    records, corrupt, _ = _read_jsonl(identities_path)
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


def _committed_shards(load_committed_state: Callable[[], tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    _, committed_batches = load_committed_state()
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
                    "has_epoch_ns": shard.get("has_epoch_ns", False),
                }
            )
    return items


def _utc_bound_ns(bound: pd.Timestamp) -> int:
    # Keep datetime bounds outside the storage range comparable to stored keys.
    if bound < pd.Timestamp.min.tz_localize("UTC"):
        return pd.Timestamp.min.value
    if bound > pd.Timestamp.max.tz_localize("UTC"):
        return pd.Timestamp.max.value
    return bound.value


def _quote_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _legacy_timestamp_rows(path: Path) -> Iterator[str]:
    """Extract only rows whose original Parquet timestamps require correction."""
    sql_path = _quote_path(path)
    start = 0
    with pq.ParquetFile(path) as file:
        for batch in file.iter_batches(columns=list(_EPOCH_NS_COLUMNS)):
            timestamps, created = (
                batch[name].cast(pa.timestamp("ns", tz="UTC")).cast(pa.int64()).to_numpy()
                for name in _EPOCH_NS_COLUMNS
            )
            for i in np.flatnonzero((timestamps % 1000 != 0) | (created % 1000 != 0)):
                yield f"({sql_path}, {start + i}, {timestamps[i]}, {created[i]})"
            start += batch.num_rows


def _scan_sql(parquet_root: Path, shards: Sequence[dict[str, Any]], paths_sql: str) -> str:
    """Expose integer epoch columns uniformly across new and historical shards."""
    legacy = [shard for shard in shards if not shard["has_epoch_ns"]]
    has_epoch = len(legacy) < len(shards)
    corrections = ", ".join(
        row for shard in legacy for row in _legacy_timestamp_rows(parquet_root / shard["part_path"])
    )
    options = (
        "hive_partitioning=true, "
        "hive_types={'factor_id': 'VARCHAR', 'frequency': 'VARCHAR', 'year_month': 'VARCHAR'}"
    )
    if legacy and has_epoch:
        options += ", union_by_name=true"
    join = ""
    if corrections:
        options += ", file_row_number=true"
        join = (
            f" LEFT JOIN (VALUES {corrections}) AS exact(path, row_no, _timestamp_utc_ns, _created_at_ns)"
            " ON raw.filename = exact.path AND raw.file_row_number = exact.row_no"
        )

    columns = [f"raw.{col}" for col in VIEW_COLUMNS if col not in _EPOCH_NS_COLUMNS]
    for name, epoch_col in _EPOCH_NS_COLUMNS.items():
        expr = f"raw.{epoch_col}" if has_epoch else f"epoch_ns(raw.{name})"
        if legacy and has_epoch:
            expr = f"coalesce({expr}, epoch_ns(raw.{name}))"
        if corrections:
            expr = f"coalesce(exact.{epoch_col}, {expr})"
        columns.append(f"{expr} AS {epoch_col}")
    return f"SELECT {', '.join(columns)} FROM read_parquet({paths_sql}, {options}) AS raw{join}"


def _projection(columns: Sequence[str]) -> str:
    return ", ".join(
        f"make_timestamp_ns({_EPOCH_NS_COLUMNS[col]}) AS {col}" if col in _EPOCH_NS_COLUMNS else col
        for col in columns
    )


def plan_read(
    parquet_root: Path,
    load_committed_state: Callable[[], tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]],
    committed_shards: Callable[[], list[dict[str, Any]]],
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
        parsed_start = pd.Timestamp(start_utc)
        _utc(parsed_start, "start_utc")
    parsed_end: pd.Timestamp | None = None
    if end_utc is not None:
        parsed_end = pd.Timestamp(end_utc)
        _utc(parsed_end, "end_utc")
    if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
        raise ValueError("start_utc must not follow end_utc")

    if session_start is not None:
        _validate_session_date_str(session_start, "session_start")
    if session_end is not None:
        _validate_session_date_str(session_end, "session_end")
    if session_start is not None and session_end is not None and session_start > session_end:
        raise ValueError("session_start must not follow session_end")

    identities, _ = load_committed_state()
    shards = committed_shards()

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

    shard_paths = tuple(parquet_root / item["part_path"] for item in selected)
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
        where_clauses.append("_timestamp_utc_ns >= ?")
        params.append(_utc_bound_ns(parsed_start))
    if parsed_end is not None:
        where_clauses.append("_timestamp_utc_ns <= ?")
        params.append(_utc_bound_ns(parsed_end))
    if session_start is not None:
        where_clauses.append("session_id >= ?")
        params.append(session_start)
    if session_end is not None:
        where_clauses.append("session_id <= ?")
        params.append(session_end)

    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    scan = _scan_sql(parquet_root, selected, "?")
    sql = (
        f"SELECT {_projection(TABLE_COLUMNS)} FROM ({scan})"
        f"{where_sql} ORDER BY factor_id, factor_identity_hash, instrument_id, _timestamp_utc_ns"
    )
    return FactorReadPlan(
        resolved_factor_ids=final_fids,
        resolved_frequencies=final_freqs,
        resolved_year_months=resolved_ym,
        shard_paths=shard_paths,
        sql=sql,
        params=tuple(params),
    )


def view_sql(
    parquet_root: Path,
    committed_shards: Callable[[], list[dict[str, Any]]],
    view_name: str,
) -> str:
    """Build CREATE OR REPLACE VIEW SQL bound strictly to committed shards."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", view_name):
        raise ValueError(f"Invalid SQL view name: {view_name!r}")
    shards = committed_shards()
    if not shards:
        return (
            f'CREATE OR REPLACE VIEW "{view_name}" AS '
            "SELECT CAST(NULL AS VARCHAR) AS factor_id, "
            "CAST(NULL AS VARCHAR) AS factor_identity_hash, "
            "CAST(NULL AS VARCHAR) AS frequency, "
            "CAST(NULL AS VARCHAR) AS session_view, "
            "CAST(NULL AS VARCHAR) AS instrument_id, "
            "CAST(NULL AS TIMESTAMP_NS) AS timestamp_utc, "
            "CAST(NULL AS VARCHAR) AS session_id, "
            "CAST(NULL AS DOUBLE) AS factor_value, "
            "CAST(NULL AS VARCHAR) AS source_build_id, "
            "CAST(NULL AS VARCHAR) AS calendar_hash, "
            "CAST(NULL AS VARCHAR) AS code_version, "
            "CAST(NULL AS TIMESTAMP_NS) AS created_at, "
            "CAST(NULL AS VARCHAR) AS batch_id, "
            "CAST(NULL AS VARCHAR) AS year_month "
            "WHERE 1 = 0"
        )
    paths_sql = ", ".join(_quote_path(parquet_root / item["part_path"]) for item in shards)
    scan = _scan_sql(parquet_root, shards, f"[{paths_sql}]")
    return f'CREATE OR REPLACE VIEW "{view_name}" AS SELECT {_projection(VIEW_COLUMNS)} FROM ({scan})'


def create_view(
    connection: duckdb.DuckDBPyConnection,
    view_name: str,
    view_sql: Callable[[str], str],
) -> None:
    """Create or refresh a DuckDB view over committed shards only."""
    connection.execute("SET TimeZone = 'UTC'")
    connection.execute(view_sql(view_name))


def explain_read(plan: FactorReadPlan) -> str:
    """Return DuckDB EXPLAIN output for a planned read query."""
    if not plan.shard_paths:
        return "EMPTY_PLAN (0 committed shards matched)"
    with duckdb.connect(":memory:") as con:
        con.execute("SET TimeZone = 'UTC'")
        rows = con.execute(f"EXPLAIN {plan.sql}", [[str(p) for p in plan.shard_paths], *plan.params]).fetchall()
    return "\n".join(str(row[-1]) for row in rows)


def read_factor_values(
    plan: FactorReadPlan, connection: duckdb.DuckDBPyConnection | None,
) -> pd.DataFrame:
    """Read committed factor values via DuckDB read_parquet (lock-free)."""
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
