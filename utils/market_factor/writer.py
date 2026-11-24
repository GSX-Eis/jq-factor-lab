"""Parquet shard publication, conflict detection and rollback."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from backtest_and_analysis.time_series.contracts import DuplicateKeyError
from utils._storage_compat import _legacy_binding
from utils.slots import CalendarSlot
from utils.time_axis import Frequency

from .identity import (
    _validate_identity_record,
)
from .ledger import (
    _append_jsonl_line,
)
from .schema import (
    DEFAULT_ROW_GROUP_SIZE,
    LEAF_ARROW_SCHEMA,
    LEAF_COLUMNS,
    TABLE_COLUMNS,
    VALID_VIEWS,
    _EPOCH_NS_COLUMNS,
    _shard_ownership_fields,
    _shard_ownership_mismatches,
    _validate_session_date_str,
    FactorValueConflictError,
    FactorWriteReceipt,
    canonical_float_bits,
    validate_factor_id,
    validate_sha256_hex,
)


_LEGACY_MODULE = "utils.market_factor_store"


def _validate_utc_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must be timezone-aware UTC datetime64, got {series.dtype}")
    tz_str = str(series.dt.tz)
    if tz_str not in ("UTC", "Etc/UTC"):
        raise ValueError(f"{name} must have UTC timezone, got {tz_str!r}")
    if series.isna().any():
        raise ValueError(f"{name} must not contain NaT")
    return pd.to_datetime(series, utc=True)


def _validate_text_series(series: pd.Series, name: str) -> None:
    if series.isna().any():
        raise ValueError(f"{name} must not contain null/NA values")
    for val in series.unique():
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{name} must contain non-empty strings, got {val!r}")


def _validate_calendars_for_batch(
    df: pd.DataFrame,
    session_view: str,
    calendar: CalendarSlot | Mapping[str, CalendarSlot],
    frequency: Frequency | None = None,
) -> None:
    """Verify intraday ownership and exact daily/weekly anchors."""
    keys = ["timestamp_utc", "session_id", "calendar_hash"]
    if isinstance(calendar, CalendarSlot):
        groups = (("all instruments", df),)
    elif isinstance(calendar, Mapping):
        groups = df.groupby("instrument_id", sort=False)
    else:
        raise TypeError("calendar must be a CalendarSlot or instrument_id -> CalendarSlot mapping")
    for inst, group in groups:
        cal = calendar.get(inst) if isinstance(calendar, Mapping) else calendar
        if not isinstance(cal, CalendarSlot):
            raise ValueError(f"Missing CalendarSlot for instrument_id={inst!r}")
        for ts, sid, calendar_hash in group[keys].drop_duplicates().itertuples(index=False):
            owner = cal.session_at(ts, session_view, frequency=frequency)
            if owner is None or owner.session_date != sid:
                raise ValueError(
                    f"session_id mismatch for {inst!r} at {ts} ({session_view}): "
                    f"calendar says {None if owner is None else owner.session_date!r}, got {sid!r}"
                )
            if calendar_hash != cal.calendar_hash:
                raise ValueError(f"calendar_hash does not match the supplied calendar for {inst!r}")


def _validate_and_prepare_batch(
    df: pd.DataFrame,
    *,
    calendar: CalendarSlot | Mapping[str, CalendarSlot] | None = None,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("batch input must be a pandas DataFrame")
    if not df.columns.is_unique:
        raise ValueError("batch columns must be unique")
    missing = set(TABLE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = df.loc[:, list(TABLE_COLUMNS)].copy()
    clean["timestamp_utc"] = _validate_utc_series(clean["timestamp_utc"], "timestamp_utc").dt.as_unit("ns")
    clean["created_at"] = _validate_utc_series(clean["created_at"], "created_at").dt.as_unit("ns")

    val_series = clean["factor_value"]
    if is_bool_dtype(val_series.dtype) or is_complex_dtype(val_series.dtype):
        raise ValueError("factor_value must be real numbers (NaN allowed), not bool or complex")
    if not is_numeric_dtype(val_series.dtype):
        # Allow object columns containing floats/ints/None/NaN, reject bools/strings.
        bits_check = canonical_float_bits(val_series.to_numpy(copy=False))
        del bits_check
        clean["factor_value"] = pd.to_numeric(val_series, errors="raise").astype(np.float64)
    else:
        clean["factor_value"] = val_series.to_numpy(dtype=np.float64, copy=True)

    vals = clean["factor_value"].to_numpy(dtype=np.float64, copy=False)
    if np.isinf(vals).any():
        raise ValueError("factor_value must not contain +-inf (NaN is allowed)")
    clean["factor_value"] = canonical_float_bits(vals).view(np.float64)

    if clean.empty:
        return clean

    if calendar is None:
        raise ValueError("calendar (CalendarSlot or instrument->CalendarSlot mapping) is required on every write")

    for col in ("factor_id", "factor_identity_hash", "frequency", "session_view"):
        _validate_text_series(clean[col], col)
        if clean[col].nunique(dropna=False) != 1:
            raise ValueError(f"Single batch must share exactly one {col}, got {clean[col].unique().tolist()}")

    fid = validate_factor_id(str(clean["factor_id"].iloc[0]))
    fhash = validate_sha256_hex(str(clean["factor_identity_hash"].iloc[0]), "factor_identity_hash")
    freq_raw = str(clean["frequency"].iloc[0])
    parsed_freq = Frequency.parse(freq_raw)
    if not parsed_freq.is_valid:
        raise ValueError(f"Invalid frequency: {freq_raw!r}")
    clean["frequency"] = parsed_freq.canonical
    clean["factor_id"] = fid
    clean["factor_identity_hash"] = fhash

    view = str(clean["session_view"].iloc[0])
    if view not in VALID_VIEWS:
        raise ValueError(f"Invalid session_view: {view!r}; expected one of {VALID_VIEWS}")

    for col in ("instrument_id", "session_id", "source_build_id", "calendar_hash", "code_version", "batch_id"):
        _validate_text_series(clean[col], col)

    for cal_hex in clean["calendar_hash"].unique():
        validate_sha256_hex(cal_hex, "calendar_hash")
    for sid in clean["session_id"].unique():
        _validate_session_date_str(sid, "session_id")

    dup_mask = clean.duplicated(subset=["factor_identity_hash", "instrument_id", "timestamp_utc"], keep=False)
    if dup_mask.any():
        examples = clean.loc[dup_mask, ["instrument_id", "timestamp_utc"]].drop_duplicates().head(5).to_dict("records")
        raise DuplicateKeyError(f"Duplicate primary keys in batch: {examples}")

    _validate_calendars_for_batch(clean, view, calendar, parsed_freq)

    return clean


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
    calendar: CalendarSlot | Mapping[str, CalendarSlot] | None = None,
    strict_source_build_id: bool = False,
) -> FactorWriteReceipt:
    """Validate and idempotently append a single-identity batch; never updates or deletes."""
    clean = _validate_and_prepare_batch(df, calendar=calendar)
    if clean.empty:
        if identity_record is None:
            raise ValueError("Empty batch requires identity_record or at least one row")
        validated_rec = _validate_identity_record(identity_record, for_write=True)
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
            for_write=True,
        )

    inc_ts_ns_series = pd.to_datetime(clean["timestamp_utc"], utc=True).dt.as_unit("ns").astype("int64")
    inc_min_ns = int(inc_ts_ns_series.min())
    inc_max_ns = int(inc_ts_ns_series.max())

    with write_lock():
        identities, committed_batches = load_committed_state()
        existing_id_rec = identities.get(fhash)
        if existing_id_rec is None and validated_id_rec is None:
            raise ValueError(
                f"factor_identity_hash {fhash} is not registered in {identities_path.name}; "
                "a verifiable identity_record is required"
            )
        registered = validated_id_rec
        if existing_id_rec is not None:
            registered = _validate_identity_record(
                existing_id_rec, expected_hash=fhash, expected_factor_id=fid,
                expected_frequency=freq, expected_session_view=view, for_write=True,
            )
        assert registered is not None
        payload = json.loads(registered["identity_payload_json"])
        expected_calendars = payload["calendar_hash"]
        for inst, cal_hash in clean[["instrument_id", "calendar_hash"]].drop_duplicates().itertuples(index=False):
            expected = expected_calendars.get(inst) if isinstance(expected_calendars, dict) else expected_calendars
            if cal_hash != expected:
                raise ValueError(f"calendar_hash for {inst!r} does not match the factor identity")
        if not clean["code_version"].eq(registered["code_version"]).all():
            raise ValueError("code_version does not match the factor identity")

        candidate_shards: list[tuple[Path, str | None]] = []
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
                    candidate_shards.append((parquet_root / shard["part_path"], batch.get("batch_id")))

        to_write = clean
        skipped = 0
        if candidate_shards:
            read_cols = ["instrument_id", "timestamp_utc", "factor_value", "session_id", "source_build_id"]
            pq_filters: list[tuple[str, str, Any]] = []
            if not strict_source_build_id:
                pq_filters.extend(
                    [
                        ("timestamp_utc", ">=", pa.scalar(inc_min_ns, type=pa.timestamp("ns", tz="UTC"))),
                        ("timestamp_utc", "<=", pa.scalar(inc_max_ns, type=pa.timestamp("ns", tz="UTC"))),
                    ]
                )
            tables = []
            for shard_path, committed_batch_id in candidate_shards:
                expected = _shard_ownership_fields(fhash, committed_batch_id, registered["source_schema_version"])
                with shard_path.open("rb") as source, pq.ParquetFile(source) as file:
                    mismatches = _shard_ownership_mismatches(file, expected)
                    if mismatches:
                        raise ValueError(f"Committed shard {shard_path} ownership mismatch: {', '.join(mismatches)}")
                    tables.append(pq.read_table(source, columns=read_cols, filters=pq_filters or None))
            existing_df = pa.concat_tables(tables).to_pandas()
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
        to_write["batch_id"] = batch_id

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
                # DuckDB 1.x reads UTC-adjusted Parquet timestamps at microsecond precision.
                for name, epoch_col in _EPOCH_NS_COLUMNS.items():
                    table = table.append_column(epoch_col, table[name].cast(pa.int64()))
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
                        "has_epoch_ns": True,
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
                "slot_id": registered["slot_id"],
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
