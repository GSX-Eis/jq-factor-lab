"""Parquet shard publication, conflict detection and rollback."""
from __future__ import annotations

import datetime as dt
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils._storage_compat import _legacy_binding
from utils.session_calendar import SessionCalendar

from .identity import (
    _validate_and_prepare_batch,
    _validate_identity_record,
)
from .ledger import (
    _append_jsonl_line,
)
from .schema import (
    DEFAULT_ROW_GROUP_SIZE,
    LEAF_ARROW_SCHEMA,
    LEAF_COLUMNS,
    FactorValueConflictError,
    FactorWriteReceipt,
    canonical_float_bits,
)


_LEGACY_MODULE = "utils.market_factor_store"


def append_factor_values(
    df: pd.DataFrame,
    *,
    root: Path,
    parquet_root: Path,
    identities_path: Path,
    batches_path: Path,
    write_lock: Callable[[], AbstractContextManager[None]],
    load_committed_state: Callable[[], tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]],
    identity_record: Mapping[str, Any] | None = None,
    calendar: SessionCalendar | Mapping[str, SessionCalendar] | None = None,
    strict_source_build_id: bool = False,
) -> FactorWriteReceipt:
    """Validate and idempotently append a single-identity batch; never updates or deletes."""
    clean = _validate_and_prepare_batch(df, calendar=calendar)
    if clean.empty:
        if identity_record is None:
            raise ValueError("Empty batch requires identity_record or at least one row")
        validated_rec = _validate_identity_record(identity_record)
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        return FactorWriteReceipt(
            batch_id=uuid.uuid4().hex,
            factor_id=validated_rec["factor_id"],
            factor_identity_hash=validated_rec["factor_identity_hash"],
            frequency=validated_rec["output_frequency"],
            session_view=str(validated_rec["session_view"]),
            inserted=0,
            skipped=0,
            shards_written=(),
            created_at=now_iso,
        )

    fid = str(clean["factor_id"].iloc[0])
    fhash = str(clean["factor_identity_hash"].iloc[0])
    freq = str(clean["frequency"].iloc[0])
    view = str(clean["session_view"].iloc[0])
    batch_id = str(clean["batch_id"].iloc[0])
    clean["year_month"] = clean["session_id"].str.slice(0, 7)
    incoming_sessions = set(clean["session_id"].unique().tolist())

    validated_id_rec: dict[str, Any] | None = None
    if identity_record is not None:
        validated_id_rec = _validate_identity_record(
            identity_record,
            expected_hash=fhash,
            expected_factor_id=fid,
            expected_frequency=freq,
            expected_session_view=view,
        )

    inc_ts_ns_series = pd.to_datetime(clean["timestamp_utc"], utc=True).dt.as_unit("ns").astype("int64")
    inc_min_ns = int(inc_ts_ns_series.min())
    inc_max_ns = int(inc_ts_ns_series.max())

    with write_lock():
        identities, committed_batches = load_committed_state()
        existing_id_rec = identities.get(fhash)
        if existing_id_rec is not None:
            existing_out_freq = existing_id_rec.get("output_frequency") or existing_id_rec["frequency"]
            if (
                existing_id_rec["factor_id"] != fid
                or existing_out_freq != freq
                or existing_id_rec["session_view"] != view
            ):
                raise ValueError(
                    f"Identity hash {fhash} already registered with different metadata: {existing_id_rec}"
                )
        elif validated_id_rec is None:
            raise ValueError(
                f"factor_identity_hash {fhash} is not registered in {identities_path.name}; "
                "a verifiable identity_record is required"
            )

        candidate_shards: list[Path] = []
        for batch in committed_batches:
            if batch.get("factor_identity_hash") != fhash:
                continue
            for shard in batch.get("shards", []):
                has_ts_bounds = "min_timestamp_ns" in shard and "max_timestamp_ns" in shard
                ts_overlaps = (
                    not has_ts_bounds
                    or (
                        int(shard["min_timestamp_ns"]) <= inc_max_ns
                        and int(shard["max_timestamp_ns"]) >= inc_min_ns
                    )
                )
                sess_overlaps = bool(set(shard.get("sessions", {}).keys()) & incoming_sessions)
                if ts_overlaps or (strict_source_build_id and sess_overlaps):
                    candidate_shards.append(parquet_root / shard["part_path"])

        to_write = clean
        skipped = 0
        if candidate_shards:
            read_cols = ["instrument_id", "timestamp_utc", "factor_value", "session_id", "source_build_id"]
            pq_filters: list[tuple[str, str, Any]] = [("factor_identity_hash", "=", fhash)]
            if not strict_source_build_id:
                pq_filters.extend(
                    [
                        ("timestamp_utc", ">=", pa.scalar(inc_min_ns, type=pa.timestamp("ns", tz="UTC"))),
                        ("timestamp_utc", "<=", pa.scalar(inc_max_ns, type=pa.timestamp("ns", tz="UTC"))),
                    ]
                )
            tables = [
                pq.read_table(
                    str(shard_path),
                    columns=read_cols,
                    filters=pq_filters,
                )
                for shard_path in candidate_shards
            ]
            existing_df = pa.concat_tables(tables).to_pandas() if tables else pd.DataFrame(columns=read_cols)
            if not existing_df.empty:
                existing_df["timestamp_utc"] = pd.to_datetime(existing_df["timestamp_utc"], utc=True).dt.as_unit("ns")
                if strict_source_build_id:
                    sess_merge = (
                        clean[["instrument_id", "session_id", "source_build_id"]]
                        .drop_duplicates()
                        .merge(
                            existing_df[["instrument_id", "session_id", "source_build_id"]].drop_duplicates(),
                            on=["instrument_id", "session_id"],
                            how="inner",
                            suffixes=("_inc", "_ext"),
                        )
                    )
                    build_diff = sess_merge["source_build_id_inc"] != sess_merge["source_build_id_ext"]
                    if build_diff.any():
                        bad_sess = sess_merge.loc[build_diff].iloc[0]
                        raise ValueError(
                            f"Upstream source_build_id changed for existing session "
                            f"({bad_sess.instrument_id!r}, {bad_sess.session_id!r}) under "
                            f"factor_identity_hash={fhash}: {bad_sess.source_build_id_ext!r} -> "
                            f"{bad_sess.source_build_id_inc!r}; mint a new factor_identity_hash"
                        )

                merged = clean[["instrument_id", "timestamp_utc", "factor_value"]].merge(
                    existing_df[["instrument_id", "timestamp_utc", "factor_value"]],
                    on=["instrument_id", "timestamp_utc"],
                    how="inner",
                    suffixes=("_inc", "_ext"),
                )
                if not merged.empty:
                    inc_bits = canonical_float_bits(merged["factor_value_inc"].to_numpy())
                    ext_bits = canonical_float_bits(merged["factor_value_ext"].to_numpy())
                    diff_mask = inc_bits != ext_bits
                    if diff_mask.any():
                        bad = merged.loc[diff_mask]
                        examples = [
                            {
                                "instrument_id": str(row.instrument_id),
                                "timestamp_utc": pd.Timestamp(row.timestamp_utc).isoformat(),
                                "existing": None if pd.isna(row.factor_value_ext) else float(row.factor_value_ext),
                                "incoming": None if pd.isna(row.factor_value_inc) else float(row.factor_value_inc),
                            }
                            for row in bad.head(5).itertuples(index=False)
                        ]
                        raise FactorValueConflictError(fid, fhash, int(diff_mask.sum()), examples)

                    dup_keys = merged[["instrument_id", "timestamp_utc"]].assign(_dup=True)
                    tagged = clean.merge(dup_keys, on=["instrument_id", "timestamp_utc"], how="left")
                    keep_mask = tagged["_dup"].isna().to_numpy()
                    skipped = int((~keep_mask).sum())
                    to_write = clean.loc[keep_mask].copy()

        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        if to_write.empty:
            return FactorWriteReceipt(
                batch_id=batch_id,
                factor_id=fid,
                factor_identity_hash=fhash,
                frequency=freq,
                session_view=view,
                inserted=0,
                skipped=skipped,
                shards_written=(),
                created_at=now_iso,
            )

        to_write = to_write.sort_values(
            ["factor_identity_hash", "instrument_id", "timestamp_utc"], kind="mergesort"
        ).reset_index(drop=True)

        staging_dir = root / f".staging-{batch_id}-{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        shards_meta: list[dict[str, Any]] = []
        shards_written: list[str] = []
        replaced_dest_files: list[Path] = []
        committed = False
        try:
            for ym, part_df in to_write.groupby("year_month", sort=True):
                part_dir_rel = Path(f"factor_id={fid}") / f"frequency={freq}" / f"year_month={ym}"
                part_name = f"part-{fhash[:12]}-{batch_id}-{uuid.uuid4().hex[:6]}.parquet"
                staged_file = staging_dir / f"{ym}_{part_name}"
                dest_dir = parquet_root / part_dir_rel
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / part_name
                if dest_file.exists():
                    raise FileExistsError(f"Refusing to overwrite existing Parquet shard: {dest_file}")

                leaf_df = part_df.loc[:, list(LEAF_COLUMNS)].copy()
                leaf_df["timestamp_utc"] = pd.to_datetime(leaf_df["timestamp_utc"], utc=True).dt.as_unit("ns")
                leaf_df["created_at"] = pd.to_datetime(leaf_df["created_at"], utc=True).dt.as_unit("ns")
                table = pa.Table.from_pandas(leaf_df, schema=LEAF_ARROW_SCHEMA, preserve_index=False)
                pq.write_table(
                    table,
                    str(staged_file),
                    compression="zstd",
                    row_group_size=_legacy_binding(_LEGACY_MODULE, "DEFAULT_ROW_GROUP_SIZE", DEFAULT_ROW_GROUP_SIZE),
                )
                os.replace(staged_file, dest_file)
                replaced_dest_files.append(dest_file)
                rel_path = str(part_dir_rel / part_name)
                sess_counts = {str(k): int(v) for k, v in part_df.groupby("session_id").size().items()}
                part_ts_ns = leaf_df["timestamp_utc"].astype("int64")
                shards_meta.append(
                    {
                        "year_month": str(ym),
                        "part_path": rel_path,
                        "row_count": int(len(part_df)),
                        "sessions": sess_counts,
                        "min_timestamp_ns": int(part_ts_ns.min()),
                        "max_timestamp_ns": int(part_ts_ns.max()),
                    }
                )
                shards_written.append(rel_path)

            if fhash not in identities:
                assert validated_id_rec is not None
                _legacy_binding(_LEGACY_MODULE, "_append_jsonl_line", _append_jsonl_line)(
                    identities_path, validated_id_rec
                )

            batch_rec = {
                "record_type": "batch",
                "status": "committed",
                "batch_id": batch_id,
                "factor_id": fid,
                "factor_identity_hash": fhash,
                "frequency": freq,
                "session_view": view,
                "row_count": int(len(to_write)),
                "created_at": now_iso,
                "shards": shards_meta,
            }
            _legacy_binding(_LEGACY_MODULE, "_append_jsonl_line", _append_jsonl_line)(batches_path, batch_rec)
            committed = True
        finally:
            if not committed:
                for dest_file in replaced_dest_files:
                    try:
                        if dest_file.exists():
                            dest_file.unlink()
                    except OSError:
                        pass
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        return FactorWriteReceipt(
            batch_id=batch_id,
            factor_id=fid,
            factor_identity_hash=fhash,
            frequency=freq,
            session_view=view,
            inserted=int(len(to_write)),
            skipped=skipped,
            shards_written=tuple(shards_written),
            created_at=now_iso,
        )
