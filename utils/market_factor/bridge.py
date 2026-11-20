"""Bridge factor output and Bar catalog identity into the append-only store."""
from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from utils.bars.catalog import BarCatalog, table_exists
from utils.code_version import resolve_code_version
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule, Frequency

from .identity import (
    build_identity_record,
    compute_factor_identity_hash,
)
from .schema import (
    TABLE_COLUMNS,
    FactorWriteReceipt,
    _require_nonempty_str,
    validate_factor_id,
)
from .store import (
    MarketFactorValueStore,
)


def _factor_source_hashes(factor: Any) -> list[str]:
    from backtest_and_analysis.time_series.factor_bridge import session_grid
    from utils.factor_base import FactorBase
    from utils.factor_core.cache import factor_core_source_hashes

    implementations = [cls for cls in type(factor).__mro__ if cls is not object]
    implementations.append(session_grid)
    hashes: list[str] = []
    for impl in implementations:
        source_file = inspect.getsourcefile(impl)
        if not source_file:
            raise ValueError(f"Cannot identify source file for {impl.__qualname__}")
        path = Path(source_file)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if impl is FactorBase:
            hashes.extend(factor_core_source_hashes(path))
    try:
        cls_source = inspect.getsource(type(factor))
        hashes.append(hashlib.sha256(cls_source.encode("utf-8")).hexdigest())
    except (OSError, TypeError):
        pass
    return hashes


def _open_factor_db_readonly(factor: Any) -> tuple[duckdb.DuckDBPyConnection | None, bool]:
    conn = getattr(factor, "db_connection", None)
    if isinstance(conn, duckdb.DuckDBPyConnection):
        return conn, False
    duckdb_path = getattr(factor, "duckdb_path", None)
    if duckdb_path is not None and Path(str(duckdb_path)).is_file():
        return duckdb.connect(str(duckdb_path), read_only=True), True
    return None, False


def _resolve_bucket_strategy(
    factor: Any,
    frequency: str,
    view: str,
    calendars: Mapping[str, SessionCalendar],
    explicit_strategy: str | None,
    conn: duckdb.DuckDBPyConnection | None,
) -> str:
    if explicit_strategy is not None:
        strategies = {
            BarSchedule(cal, frequency, view=view, bucket_strategy=explicit_strategy).bucket_strategy
            for cal in calendars.values()
        }
        if len(strategies) != 1:
            raise ValueError(f"Multiple bucket strategies resolved: {sorted(strategies)}")
        return next(iter(strategies))

    cont = factor.market_bars.attrs.get("continuous") if hasattr(factor, "market_bars") else None
    if isinstance(cont, Mapping) and cont.get("bucket_strategy"):
        return str(cont["bucket_strategy"])

    if conn is not None:
        config_obj = factor._get_local_data_config().as_dict() if hasattr(factor, "_get_local_data_config") else None
        catalog = BarCatalog(conn, config=config_obj)
        found: set[str] = set()
        for inst in calendars:
            try:
                entry = catalog.resolve(frequency, inst)
            except LookupError:
                continue
            if entry.bucket_strategy:
                found.add(entry.bucket_strategy)
        if len(found) == 1:
            strat = next(iter(found))
            for cal in calendars.values():
                BarSchedule(cal, frequency, view=view, bucket_strategy=strat)
            return strat

    strategies = {
        BarSchedule(cal, frequency, view=view, bucket_strategy=None).bucket_strategy
        for cal in calendars.values()
    }
    if len(strategies) != 1:
        raise ValueError(f"Multiple bucket strategies resolved: {sorted(strategies)}")
    return next(iter(strategies))


def _resolve_build_ids(
    factor: Any,
    frequency: str,
    required_pairs: set[tuple[str, str]],
    explicit_build_ids: str | Mapping[str, Any] | None,
    conn: duckdb.DuckDBPyConnection | None,
) -> dict[tuple[str, str], str]:
    resolved: dict[tuple[str, str], str] = {}

    def _ingest_mapping(mapping: Mapping[str, Any]) -> None:
        for k, v in mapping.items():
            if isinstance(k, tuple) and len(k) == 2 and isinstance(v, str) and v.strip():
                resolved[(str(k[0]), str(k[1]))] = v
            elif isinstance(k, str) and isinstance(v, Mapping):
                for sid, bid in v.items():
                    if isinstance(bid, str) and bid.strip():
                        resolved[(k, str(sid))] = bid
            elif isinstance(k, str) and isinstance(v, str) and v.strip():
                for inst, sid in required_pairs:
                    if inst == k:
                        resolved.setdefault((inst, sid), v)

    # 1. Explicit caller override
    if isinstance(explicit_build_ids, str):
        bid = _require_nonempty_str(explicit_build_ids, "source_build_ids")
        for pair in required_pairs:
            resolved[pair] = bid
    elif isinstance(explicit_build_ids, Mapping):
        _ingest_mapping(explicit_build_ids)

    # 2. Check market_bars.attrs['build_ids'] if present
    missing = required_pairs - set(resolved)
    if missing and hasattr(factor, "market_bars"):
        attrs_bids = factor.market_bars.attrs.get("build_ids")
        if isinstance(attrs_bids, Mapping):
            _ingest_mapping(attrs_bids)

    # 3. Actively query bar_build_log / bar_catalog from factor's DB (never assume attrs has it)
    missing = required_pairs - set(resolved)
    if missing and conn is not None:
        instruments = sorted({inst for inst, _ in missing})
        if table_exists(conn, "bar_build_log"):
            placeholders = ", ".join("?" for _ in instruments)
            rows = conn.execute(
                f"SELECT instrument_id, session_id, build_id FROM bar_build_log "
                f"WHERE frequency = ? AND instrument_id IN ({placeholders})",
                [frequency, *instruments],
            ).fetchall()
            for inst, sid, bid in rows:
                if isinstance(bid, str) and bid.strip():
                    resolved.setdefault((str(inst), str(sid)), bid)

        missing = required_pairs - set(resolved)
        if missing and table_exists(conn, "bar_catalog"):
            placeholders = ", ".join("?" for _ in instruments)
            rows = conn.execute(
                f"SELECT instrument_id, build_id FROM bar_catalog "
                f"WHERE frequency = ? AND (instrument_id IN ({placeholders}) OR instrument_id = '*') "
                f"AND build_id IS NOT NULL",
                [frequency, *instruments],
            ).fetchall()
            by_inst = {str(inst): str(bid) for inst, bid in rows if isinstance(bid, str) and bid.strip()}
            for inst, sid in list(missing):
                bid = by_inst.get(inst) or by_inst.get("*")
                if bid:
                    resolved[(inst, sid)] = bid

    missing = sorted(required_pairs - set(resolved))
    if missing:
        raise ValueError(
            "Cannot resolve non-empty source_build_id for (instrument_id, session_id) pairs "
            f"{missing[:5]} (total missing={len(missing)}): neither attrs['build_ids'], "
            "bar_build_log/bar_catalog, nor explicit source_build_ids provided a build_id"
        )
    return resolved


def save_bar_factor_values(
    factor: Any,
    store: MarketFactorValueStore,
    *,
    factor_id: str | None = None,
    bucket_strategy: str | None = None,
    source_build_ids: str | Mapping[str, Any] | None = None,
    identity_scope: str = "exact",
    batch_id: str | None = None,
) -> FactorWriteReceipt:
    """Persist a minute FactorBase output to MarketFactorValueStore without mutating FactorBase."""
    if identity_scope not in ("exact", "incremental"):
        raise ValueError("identity_scope must be 'exact' or 'incremental'")
    if not isinstance(store, MarketFactorValueStore):
        raise TypeError("store must be a MarketFactorValueStore instance")

    values = factor.get_bar_factor_frame()
    if not hasattr(factor, "market_bars") or not hasattr(factor, "market_calendars") or not hasattr(factor, "market_bar_grid"):
        raise ValueError("factor must populate market_bars, market_bar_grid, and market_calendars")

    if "_get_cache_inputs" not in vars(type(factor)):
        raise ValueError(f"{type(factor).__name__} must declare _get_cache_inputs() for factor_identity_hash")
    params = factor._get_cache_inputs()
    if params is None:
        raise ValueError(f"{type(factor).__name__}._get_cache_inputs() returned None; cannot build factor_identity_hash")

    fid = validate_factor_id(factor_id or getattr(factor, "factor_name", ""))
    raw_out_freq = getattr(factor, "output_frequency", None) or factor.data_frequency
    freq = Frequency.parse(raw_out_freq).canonical
    in_freq = Frequency.parse(factor.data_frequency).canonical
    if not freq or not in_freq:
        raise ValueError(f"Invalid factor frequency: {factor.data_frequency!r}")
    view = str(factor.session_view)
    analysis_mode = str(factor.analysis_mode)
    calendars: dict[str, SessionCalendar] = dict(factor.market_calendars)

    grid_keys = factor.market_bar_grid[["timestamp_utc", "instrument_id", "session_id"]].drop_duplicates()
    merged = values.merge(grid_keys, on=["timestamp_utc", "instrument_id"], how="left", validate="one_to_one")
    if merged["session_id"].isna().any():
        raise ValueError("Unable to resolve session_id for all rows in factor output")

    required_pairs = {(str(r.instrument_id), str(r.session_id)) for r in merged[["instrument_id", "session_id"]].drop_duplicates().itertuples(index=False)}

    conn, should_close = _open_factor_db_readonly(factor)
    try:
        resolved_strategy = _resolve_bucket_strategy(factor, in_freq, view, calendars, bucket_strategy, conn)
        build_map = _resolve_build_ids(factor, in_freq, required_pairs, source_build_ids, conn)

        catalog_meta: dict[str, Any] = {}
        if conn is not None:
            config_obj = factor._get_local_data_config().as_dict() if hasattr(factor, "_get_local_data_config") else None
            catalog = BarCatalog(conn, config=config_obj)
            for inst in sorted(calendars):
                try:
                    entry = catalog.resolve(in_freq, inst)
                except LookupError:
                    continue
                catalog_meta[inst] = {
                    "dataset": entry.dataset,
                    "table_name": entry.table_name,
                    "kind": entry.kind,
                    "source_frequency": entry.source_frequency,
                    "source_version": entry.source_version,
                }
    finally:
        if should_close and conn is not None:
            conn.close()

    cal_hashes = {inst: cal.calendar_hash for inst, cal in sorted(calendars.items())}
    rth_refs = {inst: cal.rth_reference_calendar for inst, cal in sorted(calendars.items())}

    cont_attrs = factor.market_bars.attrs.get("continuous")
    roll_bids = factor.market_bars.attrs.get("roll_build_ids")
    continuous_adjust = str(cont_attrs["adjust"]) if isinstance(cont_attrs, Mapping) and cont_attrs.get("adjust") else None

    source_identity: dict[str, Any] = {}
    if catalog_meta:
        source_identity["catalog"] = catalog_meta
    if isinstance(cont_attrs, Mapping):
        source_identity["continuous"] = {
            "product_root": cont_attrs.get("product_root"),
            "adjust": continuous_adjust,
            "frequency": cont_attrs.get("frequency"),
        }
        if roll_bids:
            source_identity["continuous"]["roll_build_ids"] = dict(sorted(roll_bids.items()))

    explicit_data_version = getattr(factor, "data_version", None)
    row_data_version = explicit_data_version
    if row_data_version is None and "data_version" in factor.market_bars.columns and not factor.market_bars.empty:
        versions = sorted(set(factor.market_bars["data_version"].dropna().astype(str)))
        if versions:
            row_data_version = ",".join(versions)

    if identity_scope == "exact":
        nested_bids: dict[str, dict[str, str]] = {}
        for (inst, sid), bid in sorted(build_map.items()):
            nested_bids.setdefault(inst, {})[sid] = bid
        id_source_build: str | Mapping[str, Any] | None = nested_bids
        id_data_version = row_data_version
    else:
        id_source_build = None
        id_data_version = explicit_data_version
        stable_catalog = {
            inst: {
                "dataset": meta["dataset"],
                "table_name": meta["table_name"],
                "kind": meta["kind"],
                "source_frequency": meta["source_frequency"],
            }
            for inst, meta in catalog_meta.items()
        }
        source_identity.pop("catalog", None)
        source_identity["source_dataset_version"] = {
            "data_version": explicit_data_version,
            "catalog": stable_catalog,
        }

    code_ver, code_ver_src = resolve_code_version()
    fhash, payload = compute_factor_identity_hash(
        factor_id=fid,
        factor_class=f"{type(factor).__module__}.{type(factor).__qualname__}",
        source_sha256=_factor_source_hashes(factor),
        params=params,
        frequency=in_freq,
        output_frequency=freq,
        session_view=view,
        analysis_mode=analysis_mode,
        bucket_strategy=resolved_strategy,
        calendar_hash=cal_hashes,
        source_identity=source_identity,
        source_build_id=id_source_build,
        data_version=id_data_version,
        rth_reference_calendar=rth_refs,
        continuous_adjust=continuous_adjust,
        roll_build_ids=roll_bids if isinstance(roll_bids, Mapping) else None,
        code_version=code_ver,
    )

    now_dt = dt.datetime.now(dt.timezone.utc)
    now_iso = now_dt.isoformat()
    bid_str = batch_id or uuid.uuid4().hex
    identity_rec = build_identity_record(
        factor_identity_hash=fhash,
        payload=payload,
        code_version_source=code_ver_src,
        created_at=now_iso,
    )

    merged["factor_id"] = fid
    merged["factor_identity_hash"] = fhash
    merged["frequency"] = freq
    merged["session_view"] = view
    merged["source_build_id"] = [
        build_map[(str(r.instrument_id), str(r.session_id))]
        for r in merged[["instrument_id", "session_id"]].itertuples(index=False)
    ]
    merged["calendar_hash"] = [cal_hashes[str(inst)] for inst in merged["instrument_id"]]
    merged["code_version"] = code_ver
    merged["created_at"] = pd.Timestamp(now_dt)
    merged["batch_id"] = bid_str

    return store.append_factor_values(
        merged.loc[:, list(TABLE_COLUMNS)],
        identity_record=identity_rec,
        calendar=calendars,
        strict_source_build_id=True,
    )
