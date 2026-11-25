"""Local table and file drivers shared by declarative slot components."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from glob import glob
import io
from pathlib import Path
import re
import threading
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest_and_analysis.time_series.contracts import _utc
from utils.market_db import MarketDbProfile, open_market_db
from utils.market_db.storage import _require_same_hash
from utils.slots import ProviderConfig

_TABLE_RE = re.compile(r'^[A-Za-z_][A-Za-z_0-9]*(\.[A-Za-z_][A-Za-z_0-9]*)?$')
_TZ_OFFSET_RE = re.compile(r'(?:Z|[+-]\d{2}:?\d{2})$')
_CSV_CHUNK_LINES = 512
_CSV_CACHE_MAX = 256
_CSV_CACHE_LOCK = threading.Lock()
_CSV_INDEX_CACHE: OrderedDict[tuple[Any, ...], _CsvFileIndex] = OrderedDict()
_DUCKDB_NULLABLE_PANDAS_DTYPES: dict[str, str] = {
    'TINYINT': 'Int8',
    'SMALLINT': 'Int16',
    'INTEGER': 'Int32',
    'BIGINT': 'Int64',
    'UTINYINT': 'UInt8',
    'USMALLINT': 'UInt16',
    'UINTEGER': 'UInt32',
    'UBIGINT': 'UInt64',
    'BOOLEAN': 'boolean',
}


def _path(base_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Data source paths must be nonempty strings')
    return base_dir / Path(value).expanduser()


def _validate_column_name(name: object, kind: str = 'column') -> str:
    if not isinstance(name, str) or not name or '\x00' in name:
        raise ValueError(f'Invalid {kind} name: {name!r}')
    return name


def _quote_ident(name: str) -> str:
    validated = _validate_column_name(name)
    escaped = validated.replace('"', '""')
    return f'"{escaped}"'


def _validate_source_timezone(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError('source_timezone must be an IANA timezone name')
    try:
        ZoneInfo(value)
    except (KeyError, ValueError) as exc:
        raise ValueError(f'Invalid source_timezone: {value!r}') from exc
    return value


def _utc_bound_datetime(bound: pd.Timestamp, *, upper: bool, unit: str = 'us') -> datetime:
    if unit == 's':
        shifted = bound.ceil('s') if upper else bound.floor('s')
    elif unit == 'ms':
        shifted = bound.ceil('ms') if upper else bound.floor('ms')
    else:
        shifted = bound.ceil('us') if (upper and bound.nanosecond > 0) else bound.floor('us')
    return shifted.to_pydatetime()


def _local_naive_bound(
    bound: pd.Timestamp, source_timezone: str | None, *, upper: bool, unit: str = 'us',
) -> datetime:
    # Naive source timestamps need DST / unknown-offset safety padding before exact post-filtering.
    if source_timezone is None:
        padded = bound + timedelta(days=1) if upper else bound - timedelta(days=1)
        return _utc_bound_datetime(padded, upper=upper, unit=unit).replace(tzinfo=None)
    padded = bound + timedelta(hours=3) if upper else bound - timedelta(hours=3)
    local = _utc_bound_datetime(padded, upper=upper, unit=unit).astimezone(ZoneInfo(source_timezone))
    return local.replace(tzinfo=None)


def _series_to_utc_ns(series: pd.Series, source_timezone: str | None) -> pd.DatetimeIndex:
    if series.empty:
        return pd.DatetimeIndex([], tz='UTC')
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return pd.DatetimeIndex(series.dt.tz_convert('UTC'))
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        if source_timezone is None:
            return pd.DatetimeIndex(pd.to_datetime(series, utc=True, errors='coerce'))
        if source_timezone in ('UTC', 'Etc/UTC'):
            return pd.DatetimeIndex([pd.NaT] * len(series), tz='UTC')
        localized = series.dt.tz_localize(source_timezone, ambiguous='NaT', nonexistent='NaT')
        return pd.DatetimeIndex(localized.dt.tz_convert('UTC'))

    if source_timezone is None and series.dtype == 'string':
        try:
            parsed = pd.to_datetime(series, format='ISO8601', utc=True)
            return pd.DatetimeIndex(parsed)
        except Exception:
            pass
    if series.dtype in ('string', object) and source_timezone is not None:
        text = series.astype('string')
        non_null = text.dropna()
        if not non_null.empty and non_null.str.contains(_TZ_OFFSET_RE).all():
            try:
                parsed = pd.to_datetime(series, format='ISO8601', utc=True)
                return pd.DatetimeIndex(parsed)
            except Exception:
                pass
        if (source_timezone not in ('UTC', 'Etc/UTC') and not non_null.empty
                and not non_null.str.contains(_TZ_OFFSET_RE).any()):
            try:
                parsed = pd.to_datetime(series, format='ISO8601')
                if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
                    localized = parsed.dt.tz_localize(source_timezone, ambiguous='NaT', nonexistent='NaT')
                    return pd.DatetimeIndex(localized.dt.tz_convert('UTC'))
            except Exception:
                pass

    converted: list[pd.Timestamp] = []
    for value in series:
        try:
            if not isinstance(value, (str, datetime)) or pd.isna(value):
                converted.append(pd.NaT)
                continue
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                converted.append(pd.NaT)
                continue
            if ts.tzinfo is None:
                if source_timezone is None:
                    ts = ts.tz_localize('UTC')
                elif source_timezone in ('UTC', 'Etc/UTC'):
                    converted.append(pd.NaT)
                    continue
                else:
                    ts = ts.tz_localize(source_timezone, ambiguous='raise', nonexistent='raise').tz_convert('UTC')
            else:
                ts = ts.tz_convert('UTC')
            converted.append(ts)
        except Exception:
            converted.append(pd.NaT)
    return pd.DatetimeIndex(converted, tz='UTC')


def _filter_frame(
    frame: pd.DataFrame,
    *,
    inst_col: str | None,
    instruments: tuple[str, ...] | None,
    ts_col: str | None,
    lower_utc: pd.Timestamp | None,
    upper_utc: pd.Timestamp | None,
    source_timezone: str | None,
    apply_inst_filter: bool,
    apply_time_filter: bool,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if apply_inst_filter and instruments is not None and inst_col is not None:
        mask &= frame[inst_col].isin(instruments)
    if apply_time_filter and ts_col is not None and (lower_utc is not None or upper_utc is not None):
        utc_ts = _series_to_utc_ns(frame[ts_col], source_timezone)
        invalid = pd.Series(utc_ts.isna(), index=frame.index)
        time_mask = pd.Series(True, index=frame.index, dtype=bool)
        if lower_utc is not None:
            time_mask &= pd.Series(utc_ts >= lower_utc, index=frame.index)
        if upper_utc is not None:
            time_mask &= pd.Series(utc_ts <= upper_utc, index=frame.index)
        # Preserve unparseable timestamps so downstream contracts raise their validation errors.
        mask &= invalid | time_mask
    if mask.all():
        return frame
    return frame.loc[mask].reset_index(drop=True)


@dataclass(frozen=True)
class _CsvChunkMeta:
    byte_start: int
    byte_end: int
    row_count: int
    min_utc: pd.Timestamp | None
    max_utc: pd.Timestamp | None
    has_unparseable_ts: bool
    instruments: frozenset[str] | None


@dataclass(frozen=True)
class _CsvFileIndex:
    header: tuple[str, ...]
    multiline_quotes: bool
    chunks: tuple[_CsvChunkMeta, ...]
    dtypes: Mapping[str, Any]
    bool_object_cols: frozenset[str] = frozenset()


def _scan_csv_byte_chunks(path: Path) -> tuple[tuple[str, ...], bool, list[tuple[int, int, int]]]:
    with path.open('rb') as raw:
        first_line = raw.readline()
        if not first_line:
            return (), False, []
        header_text = first_line.decode('utf-8-sig')
        header = tuple(next(csv.reader([header_text]), []))
        if len(set(header)) != len(header):
            raise ValueError(f'Duplicate CSV columns in {path}')
        chunks: list[tuple[int, int, int]] = []
        multiline = (first_line.count(b'"') % 2) == 1
        chunk_start = raw.tell()
        line_count = 0
        row_count = 0
        while True:
            line = raw.readline()
            if not line:
                break
            if (line.count(b'"') % 2) == 1:
                multiline = True
            line_count += 1
            if line.strip(b' \t\r\n'):
                row_count += 1
            if line_count >= _CSV_CHUNK_LINES:
                chunks.append((chunk_start, raw.tell(), row_count))
                chunk_start = raw.tell()
                line_count = 0
                row_count = 0
        if line_count > 0:
            chunks.append((chunk_start, raw.tell(), row_count))
        return header, multiline, chunks


def _csv_cache_get(key: tuple[Any, ...]) -> _CsvFileIndex | None:
    with _CSV_CACHE_LOCK:
        entry = _CSV_INDEX_CACHE.get(key)
        if entry is not None:
            _CSV_INDEX_CACHE.move_to_end(key)
        return entry


def _csv_cache_put(key: tuple[Any, ...], value: _CsvFileIndex) -> None:
    with _CSV_CACHE_LOCK:
        _CSV_INDEX_CACHE[key] = value
        _CSV_INDEX_CACHE.move_to_end(key)
        while len(_CSV_INDEX_CACHE) > _CSV_CACHE_MAX:
            _CSV_INDEX_CACHE.popitem(last=False)


def _chunk_metadata(
    frame: pd.DataFrame,
    byte_start: int,
    byte_end: int,
    row_count: int,
    inst_col: str | None,
    ts_col: str | None,
    source_timezone: str | None,
) -> _CsvChunkMeta:
    instruments: frozenset[str] | None = None
    if inst_col is not None and inst_col in frame:
        values = frame[inst_col].dropna()
        instruments = frozenset(str(v) for v in values.unique())
    min_utc: pd.Timestamp | None = None
    max_utc: pd.Timestamp | None = None
    has_unparseable = False
    if ts_col is not None and ts_col in frame and not frame.empty:
        utc_ts = _series_to_utc_ns(frame[ts_col], source_timezone)
        if utc_ts.isna().any():
            has_unparseable = True
        valid = utc_ts.dropna()
        if len(valid):
            min_utc = valid.min()
            max_utc = valid.max()
    return _CsvChunkMeta(
        byte_start=byte_start,
        byte_end=byte_end,
        row_count=row_count,
        min_utc=min_utc,
        max_utc=max_utc,
        has_unparseable_ts=has_unparseable,
        instruments=instruments,
    )


def _chunk_matches(
    meta: _CsvChunkMeta,
    *,
    instruments: tuple[str, ...] | None,
    lower_utc: pd.Timestamp | None,
    upper_utc: pd.Timestamp | None,
) -> bool:
    if instruments is not None and meta.instruments is not None:
        if meta.instruments.isdisjoint(instruments):
            return False
    if not meta.has_unparseable_ts:
        if lower_utc is not None and meta.max_utc is not None and meta.max_utc < lower_utc:
            return False
        if upper_utc is not None and meta.min_utc is not None and meta.min_utc > upper_utc:
            return False
    return True


def _detect_bool_object_cols(
    frame: pd.DataFrame, cols: Sequence[str], text_columns: Mapping[str, str],
) -> frozenset[str]:
    detected: set[str] = set()
    for col in cols:
        if col not in frame or col in text_columns or frame[col].dtype != object:
            continue
        non_null = frame[col].dropna()
        if not non_null.empty and bool(non_null.map(type).eq(bool).all()):
            detected.add(col)
    return frozenset(detected)


def _csv_read_dtypes(
    read_cols: Sequence[str],
    text_columns: Mapping[str, str],
    file_dtypes: Mapping[str, Any] | None = None,
    bool_object_cols: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    dtypes: dict[str, Any] = {}
    for col in read_cols:
        if col in text_columns:
            dtypes[col] = text_columns[col]
        elif file_dtypes is not None and col in file_dtypes and col not in bool_object_cols:
            dtypes[col] = file_dtypes[col]
    return dtypes


def _csv_query_columns(
    usecols: Sequence[str],
    inst_col: str | None,
    ts_col: str | None,
    instruments: tuple[str, ...] | None,
    lower_utc: pd.Timestamp | None,
    upper_utc: pd.Timestamp | None,
) -> tuple[list[str], bool, bool]:
    need_temp_ts = ts_col is not None and (lower_utc is not None or upper_utc is not None) and ts_col not in usecols
    need_temp_inst = inst_col is not None and instruments is not None and inst_col not in usecols
    read_cols = list(usecols)
    if need_temp_ts and ts_col is not None and ts_col not in read_cols:
        read_cols.append(ts_col)
    if need_temp_inst and inst_col is not None and inst_col not in read_cols:
        read_cols.append(inst_col)
    return read_cols, need_temp_ts, need_temp_inst


def _read_and_filter_csv_payload(
    payload: bytes,
    *,
    header: Sequence[str],
    usecols: Sequence[str],
    inst_col: str | None,
    ts_col: str | None,
    request: ReadRequest,
    lower_utc: pd.Timestamp | None,
    upper_utc: pd.Timestamp | None,
    source_timezone: str | None,
    text_columns: Mapping[str, str],
    file_dtypes: Mapping[str, Any],
    bool_object_cols: frozenset[str],
) -> pd.DataFrame:
    read_cols, need_temp_ts, need_temp_inst = _csv_query_columns(
        usecols, inst_col, ts_col, request.instruments, lower_utc, upper_utc,
    )
    read_dtype = _csv_read_dtypes(read_cols, text_columns, file_dtypes, bool_object_cols)
    chunk_df = pd.read_csv(
        io.BytesIO(payload),
        header=None,
        names=list(header),
        usecols=read_cols,
        dtype=read_dtype,
        keep_default_na=False,
        na_values=[''],
    )
    for col in usecols:
        if col in bool_object_cols and col in chunk_df and chunk_df[col].dtype != object:
            chunk_df[col] = chunk_df[col].astype(object)
    chunk_df = _filter_frame(
        chunk_df,
        inst_col=inst_col,
        instruments=request.instruments,
        ts_col=ts_col,
        lower_utc=lower_utc,
        upper_utc=upper_utc,
        source_timezone=source_timezone,
        apply_inst_filter=request.instruments is not None,
        apply_time_filter=lower_utc is not None or upper_utc is not None,
    )
    drop_cols = [c for c, needed in ((ts_col, need_temp_ts), (inst_col, need_temp_inst)) if needed and c in chunk_df]
    if drop_cols:
        chunk_df = chunk_df.drop(columns=drop_cols)
    return chunk_df


def _empty_frame_from_dtypes(
    columns: Sequence[str], dtypes: Mapping[str, Any], text_columns: Mapping[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame({
        col: pd.Series(dtype=dtypes.get(col, text_columns.get(col, object)))
        for col in columns
    })


def _read_csv_header(path: Path, expected_header_set: set[str] | None) -> tuple[list[str], set[str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle), [])
    if len(set(header)) != len(header):
        raise ValueError(f'Duplicate CSV columns in {path}')
    header_set = set(header)
    if expected_header_set is not None and header_set != expected_header_set:
        raise ValueError('All data source files must have the same columns')
    return header, (expected_header_set if expected_header_set is not None else header_set)


def _empty_csv_schema_frame(
    path: Path,
    *,
    usecols: Sequence[str],
    prune_columns: bool,
    text_columns: Mapping[str, str],
) -> pd.DataFrame:
    read_usecols = list(usecols) if prune_columns else None
    dtype = _csv_read_dtypes(usecols, text_columns) if prune_columns else dict(text_columns)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return pd.read_csv(
            handle, dtype=dtype, keep_default_na=False, na_values=[''],
            nrows=0, usecols=read_usecols,
        )


def _ensure_csv_cached_dtypes(
    path: Path,
    cache_key: tuple[Any, ...],
    cached_index: _CsvFileIndex,
    usecols: Sequence[str],
    text_columns: Mapping[str, str],
) -> _CsvFileIndex:
    missing_dtype_cols = [col for col in usecols if col not in cached_index.dtypes]
    if not missing_dtype_cols:
        return cached_index
    with path.open(encoding='utf-8-sig', newline='') as handle:
        extra_df = pd.read_csv(
            handle,
            dtype=_csv_read_dtypes(missing_dtype_cols, text_columns),
            keep_default_na=False,
            na_values=[''],
            usecols=missing_dtype_cols,
        )
    merged_dtypes = dict(cached_index.dtypes)
    merged_dtypes.update({col: extra_df[col].dtype for col in missing_dtype_cols if col in extra_df})
    merged_bools = cached_index.bool_object_cols | _detect_bool_object_cols(
        extra_df, missing_dtype_cols, text_columns,
    )
    updated = replace(cached_index, dtypes=merged_dtypes, bool_object_cols=merged_bools)
    _csv_cache_put(cache_key, updated)
    return updated


def _parquet_cols_with_nulls(path: str, cols: Sequence[str]) -> set[str]:
    import pyarrow.compute as pc
    import pyarrow.dataset as pds

    if not cols:
        return set()
    dataset = pds.dataset(path, format='parquet')
    has_null: set[str] = set()
    unknown: set[str] = set()
    pending = set(cols)
    for fragment in dataset.get_fragments():
        if not pending:
            break
        metadata = fragment.metadata
        col_indices = (
            {
                metadata.row_group(0).column(idx).path_in_schema: idx
                for idx in range(metadata.row_group(0).num_columns)
            }
            if metadata.num_row_groups > 0 else {}
        )
        for rg_idx in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg_idx)
            for col in list(pending):
                col_idx = col_indices.get(col)
                if col_idx is None:
                    unknown.add(col)
                    pending.discard(col)
                    continue
                stats = row_group.column(col_idx).statistics
                if stats is None or not stats.has_null_count:
                    unknown.add(col)
                    pending.discard(col)
                elif stats.null_count > 0:
                    has_null.add(col)
                    pending.discard(col)
    if unknown:
        tbl = dataset.to_table(columns=sorted(unknown))
        for col in unknown:
            if col in tbl.schema.names and pc.any(pc.is_null(tbl[col])).as_py():
                has_null.add(col)
    return has_null


def _align_parquet_null_dtypes(
    frame: pd.DataFrame, path: str, schema: Any, cols: Sequence[str],
) -> pd.DataFrame:
    import pyarrow as pa

    candidates: list[tuple[str, Any]] = []
    for col in cols:
        if col not in frame:
            continue
        dtype = frame[col].dtype
        if not (isinstance(dtype, np.dtype) and dtype.kind in ('i', 'u', 'b')):
            continue
        field_type = schema.field(col).type
        if pa.types.is_integer(field_type):
            candidates.append((col, 'float64'))
        elif pa.types.is_boolean(field_type):
            candidates.append((col, object))
    if not candidates:
        return frame
    null_cols = _parquet_cols_with_nulls(path, [col for col, _ in candidates])
    for col, target_dtype in candidates:
        if col in null_cols:
            frame[col] = frame[col].astype(target_dtype)
    return frame


def _align_duckdb_nullable_dtypes(
    connection: Any,
    table_sql: str,
    frame: pd.DataFrame,
    cols: Sequence[str],
    col_types: Mapping[str, str],
    col_nullable: Mapping[str, bool],
) -> pd.DataFrame:
    candidates: list[tuple[str, str]] = []
    for col in cols:
        if col not in frame or not col_nullable.get(col, True):
            continue
        target_dtype = _DUCKDB_NULLABLE_PANDAS_DTYPES.get(col_types.get(col, ''))
        if target_dtype is not None and str(frame[col].dtype) != target_dtype:
            candidates.append((col, target_dtype))
    if not candidates:
        return frame
    checks = ', '.join(
        f'EXISTS (SELECT 1 FROM {table_sql} WHERE {_quote_ident(col)} IS NULL)'
        for col, _ in candidates
    )
    has_nulls = connection.execute(f'SELECT {checks}').fetchone()
    if has_nulls is not None:
        for (col, target_dtype), has_null in zip(candidates, has_nulls):
            if has_null:
                frame[col] = frame[col].astype(target_dtype)
    return frame


@dataclass(frozen=True)
class ReadRequest:
    """Structured query pushdown and column pruning contract for FrameSource."""

    instruments: Sequence[str] | None = None
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    as_of_utc: datetime | None = None
    columns: Sequence[str] | None = None
    optional_columns: Sequence[str] = ()
    schema_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.schema_only, bool):
            raise ValueError('schema_only must be a boolean')
        if self.instruments is not None:
            if isinstance(self.instruments, (str, bytes)) or not isinstance(self.instruments, Sequence):
                raise ValueError('instruments must be a sequence of nonempty strings')
            normalized_instruments = []
            for item in self.instruments:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError('instruments must contain nonempty strings')
                normalized_instruments.append(item)
            object.__setattr__(self, 'instruments', tuple(dict.fromkeys(normalized_instruments)))
        for name in ('start_utc', 'end_utc', 'as_of_utc'):
            value = getattr(self, name)
            if value is not None:
                _utc(value, name)
        if self.start_utc is not None and self.end_utc is not None and self.start_utc > self.end_utc:
            raise ValueError('start_utc must not follow end_utc')
        if self.columns is not None:
            if isinstance(self.columns, (str, bytes)) or not isinstance(self.columns, Sequence):
                raise ValueError('columns must be a sequence of column names')
            if not self.columns:
                raise ValueError('columns must not be empty when specified')
            normalized_cols = tuple(dict.fromkeys(_validate_column_name(col, 'column') for col in self.columns))
            object.__setattr__(self, 'columns', normalized_cols)
        if isinstance(self.optional_columns, (str, bytes)) or not isinstance(self.optional_columns, Sequence):
            raise ValueError('optional_columns must be a sequence of column names')
        normalized_opt = tuple(dict.fromkeys(_validate_column_name(col, 'column') for col in self.optional_columns))
        object.__setattr__(self, 'optional_columns', normalized_opt)


@dataclass(frozen=True)
class FrameSource:
    provider: ProviderConfig
    base_dir: Path
    text_fields: Sequence[str]
    market_db_profile: MarketDbProfile | None = None

    def __post_init__(self) -> None:
        if self.provider.provider not in ('duckdb_table', 'parquet_glob', 'csv_files'):
            raise ValueError(f'Unknown declarative data driver: {self.provider.provider!r}')

    @property
    def database_path(self) -> Path:
        return _path(self.base_dir, self.provider.options.get('path')).resolve()

    def _validated_mapping(self, *, strict_unique_targets: bool) -> tuple[dict[str, str], dict[str, str]]:
        mapping = self.provider.options.get('column_mapping', {})
        if (not isinstance(mapping, Mapping)
                or any(not isinstance(key, str) or not key or '\x00' in key
                       or not isinstance(value, str) or not value or '\x00' in value
                       for key, value in mapping.items())):
            raise ValueError('column_mapping must map source names to column names')
        validated = {
            _validate_column_name(key, 'source column'): _validate_column_name(value, 'logical column')
            for key, value in mapping.items()
        }
        reverse: dict[str, str] = {}
        for source, target in validated.items():
            if target not in reverse:
                reverse[target] = source
        if strict_unique_targets and len(reverse) != len(validated):
            raise ValueError('column_mapping produces duplicate names')
        return validated, reverse

    def _resolve_plan(
        self,
        schema_columns: Sequence[str],
        mapping: Mapping[str, str],
        reverse: Mapping[str, str],
        request: ReadRequest,
    ) -> tuple[list[str], str | None, str | None, bool]:
        schema_list = list(schema_columns)
        schema_set = set(schema_list)
        if len(schema_set) != len(schema_list) or set(mapping) - schema_set:
            raise ValueError('column_mapping references missing or duplicate source columns')

        def resolve_logical(logical: str) -> str | None:
            if logical in reverse:
                return reverse[logical]
            if logical in mapping:
                return None
            if logical in schema_set:
                return logical
            return None

        if request.columns is not None:
            selected_set: set[str] = set()
            missing: list[str] = []
            for col in request.columns:
                src = resolve_logical(col)
                if src is None:
                    missing.append(col)
                else:
                    selected_set.add(src)
            if missing:
                raise ValueError(f'Unknown requested columns: {sorted(missing)}')
            for col in request.optional_columns:
                src = resolve_logical(col)
                if src is not None:
                    selected_set.add(src)
            selected_cols = [col for col in schema_list if col in selected_set]
        else:
            selected_cols = schema_list

        empty_shortcut = False
        inst_col: str | None = None
        if request.instruments is not None:
            inst_col = resolve_logical('instrument_id')
            if inst_col is None:
                static_instrument = self.provider.options.get('instrument_id')
                if static_instrument is not None:
                    if static_instrument not in request.instruments:
                        empty_shortcut = True
                else:
                    raise ValueError('Cannot filter by instruments without an instrument_id column')
            elif not request.instruments:
                empty_shortcut = True

        ts_col: str | None = None
        if request.start_utc is not None or request.end_utc is not None or request.as_of_utc is not None:
            if 'timestamp' in self.text_fields and resolve_logical('timestamp') is not None:
                ts_col = resolve_logical('timestamp')
            elif 'available_at_utc' in self.text_fields and resolve_logical('available_at_utc') is not None:
                ts_col = resolve_logical('available_at_utc')
            else:
                ts_col = resolve_logical('timestamp') or resolve_logical('available_at_utc')
            if ts_col is None:
                raise ValueError('Time filter requires a timestamp or available_at_utc column')

        return selected_cols, inst_col, ts_col, empty_shortcut

    def read(self, request: ReadRequest | None = None, *, schema_only: bool = False) -> pd.DataFrame:
        """Read source data with optional pushdown/pruning, or return an empty schema frame."""
        if request is None:
            request = ReadRequest(schema_only=schema_only)
        elif not isinstance(request, ReadRequest):
            raise TypeError('request must be a ReadRequest')
        elif schema_only and not request.schema_only:
            request = replace(request, schema_only=True)

        options = self.provider.options
        driver = self.provider.provider
        has_query = (
            request.instruments is not None
            or request.start_utc is not None
            or request.end_utc is not None
            or request.as_of_utc is not None
            or request.columns is not None
            or bool(request.optional_columns)
        )
        mapping, reverse = self._validated_mapping(strict_unique_targets=has_query)
        source_timezone = _validate_source_timezone(options.get('source_timezone'))

        lower_utc = pd.Timestamp(request.start_utc) if request.start_utc is not None else None
        upper_candidates = [pd.Timestamp(t) for t in (request.end_utc, request.as_of_utc) if t is not None]
        upper_utc = min(upper_candidates) if upper_candidates else None
        impossible_time = lower_utc is not None and upper_utc is not None and lower_utc > upper_utc

        if driver == 'duckdb_table':
            frame = self._read_duckdb(
                options, mapping, reverse, request, lower_utc, upper_utc, impossible_time, source_timezone,
            )
        elif driver == 'parquet_glob':
            frame = self._read_parquet(
                options, mapping, reverse, request, lower_utc, upper_utc, impossible_time, source_timezone,
            )
        else:
            frame = self._read_csv(
                options, mapping, reverse, request, lower_utc, upper_utc, impossible_time, source_timezone,
            )

        if not frame.columns.is_unique:
            raise ValueError('column_mapping references missing or duplicate source columns')
        return frame

    def _read_duckdb(
        self,
        options: Mapping[str, Any],
        mapping: Mapping[str, str],
        reverse: Mapping[str, str],
        request: ReadRequest,
        lower_utc: pd.Timestamp | None,
        upper_utc: pd.Timestamp | None,
        impossible_time: bool,
        source_timezone: str | None,
    ) -> pd.DataFrame:
        table = options.get('table')
        if not isinstance(table, str) or not _TABLE_RE.fullmatch(table):
            raise ValueError('duckdb_table requires a table name or schema.table')
        table_sql = '.'.join(_quote_ident(part) for part in table.split('.'))
        with open_market_db(self.database_path) as (connection, profile, _):
            if self.market_db_profile is not None:
                _require_same_hash(profile, self.market_db_profile, 'Declarative database changed')
            connection.execute("SET TimeZone='UTC'")
            described = connection.execute(f'DESCRIBE {table_sql}').fetchall()
            schema_cols = [row[0] for row in described]
            col_types = {row[0]: str(row[1]).upper() for row in described}
            col_nullable = {row[0]: str(row[2]).upper() != 'NO' for row in described}
            selected_cols, inst_col, ts_col, empty_shortcut = self._resolve_plan(
                schema_cols, mapping, reverse, request,
            )
            select_cols = selected_cols
            projection = (
                '*' if request.columns is None
                else ', '.join(_quote_ident(col) for col in select_cols)
            )
            if request.schema_only or empty_shortcut or impossible_time:
                frame = connection.execute(f'SELECT {projection} FROM {table_sql} LIMIT 0').fetchdf()
                if not request.schema_only:
                    frame = _align_duckdb_nullable_dtypes(
                        connection, table_sql, frame, select_cols, col_types, col_nullable,
                    )
                return frame

            where_clauses: list[str] = []
            params: list[Any] = []
            if request.instruments is not None and inst_col is not None:
                placeholders = ', '.join('?' for _ in request.instruments)
                where_clauses.append(f'{_quote_ident(inst_col)} IN ({placeholders})')
                params.extend(request.instruments)

            exact_time_pushed = True
            need_temp_ts = False
            if ts_col is not None and (lower_utc is not None or upper_utc is not None):
                ts_type = col_types[ts_col]
                quoted_ts = _quote_ident(ts_col)
                if ts_type in ('TIMESTAMP WITH TIME ZONE', 'TIMESTAMPTZ'):
                    if lower_utc is not None:
                        where_clauses.append(f'{quoted_ts} >= ?')
                        params.append(_utc_bound_datetime(lower_utc, upper=False))
                    if upper_utc is not None:
                        where_clauses.append(f'{quoted_ts} <= ?')
                        params.append(_utc_bound_datetime(upper_utc, upper=True))
                    if ((lower_utc is not None and lower_utc.nanosecond > 0)
                            or (upper_utc is not None and upper_utc.nanosecond > 0)):
                        exact_time_pushed = False
                        need_temp_ts = ts_col not in select_cols
                elif 'TIMESTAMP' in ts_type or ts_type == 'DATE':
                    exact_time_pushed = False
                    need_temp_ts = ts_col not in select_cols
                    if lower_utc is not None:
                        local_low = _local_naive_bound(lower_utc, source_timezone, upper=False)
                        where_clauses.append(f'({quoted_ts} IS NULL OR {quoted_ts} >= ?)')
                        params.append(local_low.date() if ts_type == 'DATE' else local_low)
                    if upper_utc is not None:
                        local_high = _local_naive_bound(upper_utc, source_timezone, upper=True)
                        where_clauses.append(f'({quoted_ts} IS NULL OR {quoted_ts} <= ?)')
                        params.append(local_high.date() if ts_type == 'DATE' else local_high)
                elif any(token in ts_type for token in ('VARCHAR', 'TEXT', 'STRING', 'CHAR')):
                    exact_time_pushed = False
                    need_temp_ts = ts_col not in select_cols
                    if lower_utc is not None:
                        where_clauses.append(
                            f'(TRY_CAST({quoted_ts} AS TIMESTAMPTZ) IS NULL OR '
                            f'TRY_CAST({quoted_ts} AS TIMESTAMPTZ) >= ?)'
                        )
                        params.append(_utc_bound_datetime(lower_utc - timedelta(days=1), upper=False))
                    if upper_utc is not None:
                        where_clauses.append(
                            f'(TRY_CAST({quoted_ts} AS TIMESTAMPTZ) IS NULL OR '
                            f'TRY_CAST({quoted_ts} AS TIMESTAMPTZ) <= ?)'
                        )
                        params.append(_utc_bound_datetime(upper_utc + timedelta(days=1), upper=True))
                else:
                    exact_time_pushed = False
                    need_temp_ts = ts_col not in select_cols

            query_cols = list(select_cols)
            if need_temp_ts and ts_col is not None:
                query_cols.append(ts_col)
            query_projection = (
                '*' if (request.columns is None and not need_temp_ts)
                else ', '.join(_quote_ident(col) for col in query_cols)
            )
            where_sql = f' WHERE {" AND ".join(where_clauses)}' if where_clauses else ''
            frame = connection.execute(f'SELECT {query_projection} FROM {table_sql}{where_sql}', params).fetchdf()
            frame = _align_duckdb_nullable_dtypes(
                connection, table_sql, frame, select_cols, col_types, col_nullable,
            )

        if not exact_time_pushed:
            frame = _filter_frame(
                frame,
                inst_col=inst_col,
                instruments=request.instruments,
                ts_col=ts_col,
                lower_utc=lower_utc,
                upper_utc=upper_utc,
                source_timezone=source_timezone,
                apply_inst_filter=False,
                apply_time_filter=True,
            )
            if need_temp_ts and ts_col in frame:
                frame = frame.drop(columns=[ts_col])

        timestamp = reverse.get('timestamp')
        if (source_timezone and timestamp in frame
                and isinstance(frame[timestamp].dtype, pd.DatetimeTZDtype)):
            # DuckDB projects TIMESTAMPTZ in the connection timezone, regardless of the input offset.
            frame[timestamp] = frame[timestamp].dt.tz_convert(source_timezone)
        return frame

    def _read_parquet(
        self,
        options: Mapping[str, Any],
        mapping: Mapping[str, str],
        reverse: Mapping[str, str],
        request: ReadRequest,
        lower_utc: pd.Timestamp | None,
        upper_utc: pd.Timestamp | None,
        impossible_time: bool,
        source_timezone: str | None,
    ) -> pd.DataFrame:
        import pyarrow as pa
        import pyarrow.compute as pc
        from pyarrow.parquet import ParquetDataset

        paths = sorted(glob(str(_path(self.base_dir, options.get('path'))), recursive=True))
        if not paths:
            raise ValueError('parquet_glob did not match any files')

        schemas = [ParquetDataset(path).schema for path in paths]
        for schema in schemas[1:]:
            if set(schema.names) != set(schemas[0].names):
                raise ValueError('All data source files must have the same columns')

        selected_cols, inst_col, ts_col, empty_shortcut = self._resolve_plan(
            schemas[0].names, mapping, reverse, request,
        )

        if request.schema_only or empty_shortcut or impossible_time:
            empty_frames = []
            for path, schema in zip(paths, schemas):
                empty_df = schema.empty_table().to_pandas()
                if request.columns is not None:
                    empty_df = empty_df.loc[:, [col for col in selected_cols if col in empty_df.columns]]
                if not request.schema_only:
                    empty_df = _align_parquet_null_dtypes(empty_df, path, schema, empty_df.columns)
                empty_frames.append(empty_df)
            return self._combine(empty_frames)

        frames: list[pd.DataFrame] = []
        for path, schema in zip(paths, schemas):
            filter_expr = None
            inst_pushed = True
            if request.instruments is not None and inst_col is not None:
                filter_expr = pc.field(inst_col).isin(list(request.instruments))

            exact_time_pushed = True
            need_temp_ts = False
            if ts_col is not None and (lower_utc is not None or upper_utc is not None):
                ts_type = schema.field(ts_col).type
                if pa.types.is_timestamp(ts_type):
                    time_expr = None
                    if ts_type.tz is not None:
                        if lower_utc is not None:
                            scalar_low = (
                                pa.scalar(lower_utc.value, type=ts_type) if ts_type.unit == 'ns'
                                else pa.scalar(_utc_bound_datetime(lower_utc, upper=False, unit=ts_type.unit), type=ts_type)
                            )
                            time_expr = pc.field(ts_col) >= scalar_low
                        if upper_utc is not None:
                            scalar_high = (
                                pa.scalar(upper_utc.value, type=ts_type) if ts_type.unit == 'ns'
                                else pa.scalar(_utc_bound_datetime(upper_utc, upper=True, unit=ts_type.unit), type=ts_type)
                            )
                            cond = pc.field(ts_col) <= scalar_high
                            time_expr = cond if time_expr is None else (time_expr & cond)
                        if ts_type.unit != 'ns' and (
                            (lower_utc is not None and lower_utc.nanosecond > 0)
                            or (upper_utc is not None and upper_utc.nanosecond > 0)
                        ):
                            exact_time_pushed = False
                            need_temp_ts = ts_col not in selected_cols
                    else:
                        exact_time_pushed = False
                        need_temp_ts = ts_col not in selected_cols
                        if lower_utc is not None:
                            local_low = _local_naive_bound(lower_utc, source_timezone, upper=False, unit=ts_type.unit)
                            time_expr = pc.field(ts_col) >= pa.scalar(local_low, type=ts_type)
                        if upper_utc is not None:
                            local_high = _local_naive_bound(upper_utc, source_timezone, upper=True, unit=ts_type.unit)
                            cond = pc.field(ts_col) <= pa.scalar(local_high, type=ts_type)
                            time_expr = cond if time_expr is None else (time_expr & cond)
                        if schema.field(ts_col).nullable and time_expr is not None:
                            time_expr = pc.field(ts_col).is_null() | time_expr
                    if time_expr is not None:
                        filter_expr = time_expr if filter_expr is None else (filter_expr & time_expr)
                else:
                    exact_time_pushed = False
                    need_temp_ts = ts_col not in selected_cols

            read_cols = list(selected_cols) if request.columns is not None else None
            if read_cols is not None and need_temp_ts and ts_col is not None and ts_col not in read_cols:
                read_cols.append(ts_col)

            dataset = ParquetDataset(path, filters=filter_expr) if filter_expr is not None else ParquetDataset(path)
            df = dataset.read(columns=read_cols).to_pandas()
            if not exact_time_pushed or not inst_pushed:
                df = _filter_frame(
                    df,
                    inst_col=inst_col,
                    instruments=request.instruments,
                    ts_col=ts_col,
                    lower_utc=lower_utc,
                    upper_utc=upper_utc,
                    source_timezone=source_timezone,
                    apply_inst_filter=not inst_pushed,
                    apply_time_filter=not exact_time_pushed,
                )
            if need_temp_ts and ts_col in df:
                df = df.drop(columns=[ts_col])
            if request.columns is not None:
                df = df.loc[:, [col for col in selected_cols if col in df.columns]]
            df = _align_parquet_null_dtypes(df, path, schema, df.columns)
            frames.append(df)
        return self._combine(frames)

    def _read_csv(
        self,
        options: Mapping[str, Any],
        mapping: Mapping[str, str],
        reverse: Mapping[str, str],
        request: ReadRequest,
        lower_utc: pd.Timestamp | None,
        upper_utc: pd.Timestamp | None,
        impossible_time: bool,
        source_timezone: str | None,
    ) -> pd.DataFrame:
        paths = options.get('paths')
        if not isinstance(paths, tuple) or not paths:
            raise ValueError('csv_files requires a nonempty paths list')
        text_columns = {reverse.get(field, field): 'string' for field in self.text_fields}
        has_row_filter = (
            request.instruments is not None
            or lower_utc is not None
            or upper_utc is not None
        )

        frames: list[pd.DataFrame] = []
        expected_header_set: set[str] | None = None
        text_sig = tuple(sorted(text_columns.items()))
        for value in paths:
            path = _path(self.base_dir, value).resolve()
            header, expected_header_set = _read_csv_header(path, expected_header_set)
            selected_cols, inst_col, ts_col, empty_shortcut = self._resolve_plan(
                header, mapping, reverse, request,
            )
            usecols = selected_cols if request.columns is not None else list(header)

            if request.schema_only:
                frames.append(_empty_csv_schema_frame(
                    path,
                    usecols=usecols,
                    prune_columns=request.columns is not None,
                    text_columns=text_columns,
                ))
                continue

            if not has_row_filter:
                read_usecols = usecols if request.columns is not None else None
                read_dtype = _csv_read_dtypes(usecols, text_columns)
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    frames.append(pd.read_csv(
                        handle, dtype=read_dtype, keep_default_na=False,
                        na_values=[''], usecols=read_usecols,
                    ))
                continue

            stat = path.stat()
            cache_key = (str(path), stat.st_mtime_ns, stat.st_size, inst_col, ts_col, source_timezone, text_sig)
            cached_index = _csv_cache_get(cache_key)

            if cached_index is not None:
                if empty_shortcut or impossible_time or (not cached_index.chunks and not cached_index.multiline_quotes):
                    cached_index = _ensure_csv_cached_dtypes(path, cache_key, cached_index, usecols, text_columns)
                    frames.append(_empty_frame_from_dtypes(usecols, cached_index.dtypes, text_columns))
                    continue
                if not cached_index.multiline_quotes:
                    cached_index = _ensure_csv_cached_dtypes(path, cache_key, cached_index, usecols, text_columns)

                    matching = [
                        chunk for chunk in cached_index.chunks
                        if _chunk_matches(chunk, instruments=request.instruments,
                                          lower_utc=lower_utc, upper_utc=upper_utc)
                    ]
                    if not matching:
                        frames.append(_empty_frame_from_dtypes(usecols, cached_index.dtypes, text_columns))
                        continue
                    chunk_frames: list[pd.DataFrame] = []
                    with path.open('rb') as raw:
                        for chunk in matching:
                            raw.seek(chunk.byte_start)
                            payload = raw.read(chunk.byte_end - chunk.byte_start)
                            chunk_df = _read_and_filter_csv_payload(
                                payload,
                                header=cached_index.header,
                                usecols=usecols,
                                inst_col=inst_col,
                                ts_col=ts_col,
                                request=request,
                                lower_utc=lower_utc,
                                upper_utc=upper_utc,
                                source_timezone=source_timezone,
                                text_columns=text_columns,
                                file_dtypes=cached_index.dtypes,
                                bool_object_cols=cached_index.bool_object_cols,
                            )
                            if not chunk_df.empty:
                                chunk_frames.append(chunk_df)
                    if not chunk_frames:
                        frames.append(_empty_frame_from_dtypes(usecols, cached_index.dtypes, text_columns))
                    else:
                        frames.append(pd.concat(chunk_frames, ignore_index=True))
                    continue

            # Cache miss or multiline CSV fallback: read usecols once for file-level dtypes and index chunks.
            if cached_index is not None and cached_index.multiline_quotes:
                scanned_header, multiline, byte_ranges = cached_index.header, True, [(0, 0, 0)]
            else:
                scanned_header, multiline, byte_ranges = _scan_csv_byte_chunks(path)
            if not byte_ranges:
                empty_df = _empty_csv_schema_frame(
                    path,
                    usecols=usecols,
                    prune_columns=request.columns is not None,
                    text_columns=text_columns,
                )
                if not multiline:
                    _csv_cache_put(cache_key, _CsvFileIndex(
                        header=scanned_header, multiline_quotes=False, chunks=(),
                        dtypes={col: empty_df[col].dtype for col in empty_df},
                    ))
                frames.append(empty_df)
                continue

            read_cols, need_temp_ts, need_temp_inst = _csv_query_columns(
                usecols, inst_col, ts_col, request.instruments, lower_utc, upper_utc,
            )
            read_dtype = _csv_read_dtypes(read_cols, text_columns)
            use_all_cols = request.columns is None and not need_temp_ts and not need_temp_inst
            with path.open(encoding='utf-8-sig', newline='') as handle:
                full_df = pd.read_csv(
                    handle,
                    dtype=read_dtype,
                    keep_default_na=False,
                    na_values=[''],
                    usecols=None if use_all_cols else read_cols,
                )
            dtypes_map = {col: full_df[col].dtype for col in usecols if col in full_df}
            bool_obj_cols = _detect_bool_object_cols(full_df, usecols, text_columns)

            if multiline:
                _csv_cache_put(cache_key, _CsvFileIndex(
                    header=scanned_header, multiline_quotes=True, chunks=(),
                    dtypes=dtypes_map, bool_object_cols=bool_obj_cols,
                ))
            elif len(byte_ranges) == 1:
                b_start, b_end, _ = byte_ranges[0]
                meta = _chunk_metadata(
                    full_df, b_start, b_end, len(full_df), inst_col, ts_col, source_timezone,
                )
                _csv_cache_put(cache_key, _CsvFileIndex(
                    header=scanned_header, multiline_quotes=False, chunks=(meta,),
                    dtypes=dtypes_map, bool_object_cols=bool_obj_cols,
                ))
            else:
                total_rows = sum(row_count for _, _, row_count in byte_ranges)
                metas: list[_CsvChunkMeta] = []
                if len(full_df) == total_rows:
                    row_offset = 0
                    for b_start, b_end, row_count in byte_ranges:
                        chunk_slice = full_df.iloc[row_offset:row_offset + row_count]
                        metas.append(_chunk_metadata(
                            chunk_slice, b_start, b_end, row_count, inst_col, ts_col, source_timezone,
                        ))
                        row_offset += row_count
                else:
                    filter_cols = [c for c in (inst_col, ts_col) if c is not None]
                    filter_dtype = _csv_read_dtypes(filter_cols, text_columns)
                    with path.open('rb') as raw:
                        for b_start, b_end, _ in byte_ranges:
                            raw.seek(b_start)
                            payload = raw.read(b_end - b_start)
                            idx_df = pd.read_csv(
                                io.BytesIO(payload),
                                header=None,
                                names=list(scanned_header),
                                usecols=filter_cols,
                                dtype=filter_dtype,
                                keep_default_na=False,
                                na_values=[''],
                            )
                            metas.append(_chunk_metadata(
                                idx_df, b_start, b_end, len(idx_df), inst_col, ts_col, source_timezone,
                            ))
                _csv_cache_put(cache_key, _CsvFileIndex(
                    header=scanned_header, multiline_quotes=False, chunks=tuple(metas),
                    dtypes=dtypes_map, bool_object_cols=bool_obj_cols,
                ))

            if empty_shortcut or impossible_time:
                frames.append(_empty_frame_from_dtypes(usecols, dtypes_map, text_columns))
                continue

            filtered_df = _filter_frame(
                full_df,
                inst_col=inst_col,
                instruments=request.instruments,
                ts_col=ts_col,
                lower_utc=lower_utc,
                upper_utc=upper_utc,
                source_timezone=source_timezone,
                apply_inst_filter=request.instruments is not None,
                apply_time_filter=lower_utc is not None or upper_utc is not None,
            )
            drop_cols = [
                c for c, needed in ((ts_col, need_temp_ts), (inst_col, need_temp_inst))
                if needed and c in filtered_df
            ]
            if drop_cols:
                filtered_df = filtered_df.drop(columns=drop_cols)
            frames.append(filtered_df)
        return self._combine(frames)

    @staticmethod
    def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if any(set(frame.columns) != set(frames[0].columns) for frame in frames):
            raise ValueError('All data source files must have the same columns')
        if len(frames) == 1:
            return frames[0].copy()
        return pd.concat(frames, ignore_index=True)

    def read_mapped(self, request: ReadRequest | None = None, *, schema_only: bool = False) -> pd.DataFrame:
        frame = self.read(request, schema_only=schema_only).rename(
            columns=dict(self.provider.options.get('column_mapping', {}))
        )
        if not frame.columns.is_unique:
            raise ValueError('column_mapping produces duplicate names')
        return frame
