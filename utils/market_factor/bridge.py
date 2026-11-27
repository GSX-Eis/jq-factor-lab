"""Persist FactorBase output against its MarketSlot and trading clock."""
from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import re
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from utils.market_db.identity import bind_market_db_fingerprint, slot_market_db_identity
from utils.code_version import resolve_code_version
from utils.data_revision import FactorFrameDigest
from utils.slots import CalendarSlot
from utils.time_axis import BarSchedule, Frequency

from .identity import build_identity_record, compute_factor_identity_hash
from .schema import (
    TABLE_COLUMNS, FactorWriteReceipt, _canonical_json, _frame_digest, canonical_float_bits, validate_factor_id,
)
from .store import MarketFactorValueStore

if TYPE_CHECKING:
    from utils.factor_base import FactorBase


def _factor_source_hashes(factor: FactorBase) -> list[str]:
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


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def factor_rows_on_grid(on_grid: pd.Series, populated: pd.Series) -> pd.Series:
    """Keep scheduled rows and reject any observation without a grid key."""
    if (~on_grid & populated).any():
        raise ValueError("Factor output contains values outside the Slot output grid")
    return on_grid


def _wide_values(
    factor: FactorBase, panel: pd.DataFrame, frequency: Frequency, strategy: str | None,
    bar_grid: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, CalendarSlot]]:
    """Map date labels or UTC bar openings onto authoritative Slot anchors."""
    if not panel.columns.is_unique or not all(isinstance(key, str) and key for key in panel.columns):
        raise ValueError("Factor values require unique, nonempty instrument columns")
    dates = pd.DatetimeIndex(panel.index)
    if dates.hasnans or not dates.is_unique:
        raise ValueError("Factor values require unique, non-null dates")

    if dates.tz is not None:
        dates = dates.tz_convert("UTC")
    values = panel.set_axis(dates.as_unit("ns"), axis=0).rename_axis(index="date", columns=None).reset_index().melt(
        id_vars="date", var_name="instrument_id", value_name="factor_value",
    )
    calendars = {
        inst: getattr(factor, "market_calendars", {}).get(inst, factor.slot.calendar) for inst in panel.columns
    }
    if bar_grid is not None and dates.tz is not None:
        grid = bar_grid[["timestamp_utc", "instrument_id", "session_id"]].assign(
            date=lambda frame: frame.timestamp_utc.dt.as_unit("ns"),
        )
    else:
        grids = []
        for inst, calendar in calendars.items():
            if dates.tz is not None or frequency.kind == "intraday":
                grid = calendar.bar_grid(frequency, factor.start_date, factor.end_date,
                                         view=factor.session_view, bucket_strategy=strategy)
                labels = grid.timestamp_utc
                if dates.tz is None:
                    clock = getattr(calendar, "default", calendar)
                    schedule = BarSchedule(clock, frequency, view=factor.session_view, bucket_strategy=strategy)
                    # Historical minute panels label the end of each intraday bar.
                    ends = pd.DatetimeIndex([schedule.end_at(ts) for ts in labels], tz="UTC")
                    labels = ends.tz_convert(clock.timezone).tz_localize(None)
            else:
                grid = calendar.bar_grid(frequency, dates.min().date(), dates.max().date(),
                                         view=factor.session_view, bucket_strategy=strategy)
                labels = pd.to_datetime(grid.session_id)
            grids.append(grid.assign(date=pd.DatetimeIndex(labels).as_unit("ns"), instrument_id=inst))
        grid = pd.concat(grids, ignore_index=True)
    keyed = values.merge(grid, on=["date", "instrument_id"], how="left", validate="one_to_one")
    on_grid = factor_rows_on_grid(keyed.session_id.notna(), keyed.factor_value.notna())
    return keyed.loc[on_grid].drop(columns="date"), calendars


def _build_ids(
    required: set[tuple[str, str]],
    catalog_entries: dict,
    catalog_builds: dict,
    supplied: str | Mapping | None,
) -> dict[tuple[str, str], str]:
    builds = {}
    for inst, session in sorted(required):
        source = supplied if isinstance(supplied, str) else None
        if isinstance(supplied, Mapping):
            source = supplied.get((inst, session), supplied.get(inst))
            if isinstance(source, Mapping):
                source = source.get(session)
        if source is None:
            source = catalog_builds.get((inst, session)) or getattr(catalog_entries.get(inst), "build_id", None)
        if source is not None:
            if not isinstance(source, str) or not source.strip():
                raise ValueError("source_build_ids must contain nonempty strings")
            builds[inst, session] = source
    return builds


def save_factor_values(
    factor: FactorBase,
    store: MarketFactorValueStore,
    *,
    factor_values: pd.DataFrame | None = None,
    factor_id: str | None = None,
    factor_params: Mapping[str, Any] | None = None,
    bucket_strategy: str | None = None,
    source_build_ids: str | Mapping | None = None,
    identity_scope: str = "exact",
    batch_id: str | None = None,
) -> FactorWriteReceipt:
    """Append calculated or precomputed values, with per-Slot identity and provenance."""
    from backtest_and_analysis.time_series.factor_bridge import incomplete_bar_keys, mask_incomplete_bars
    from utils.factor_core.bar_bridge import _bucket_strategy, _session_rollup

    if not isinstance(store, MarketFactorValueStore):
        raise TypeError("store must be a MarketFactorValueStore")
    if identity_scope not in ("exact", "incremental"):
        raise ValueError("identity_scope must be 'exact' or 'incremental'")

    slot = factor.slot
    feed = slot.bar_feed
    ensure_profile = getattr(feed, "ensure_profile", None)
    if callable(ensure_profile):
        ensure_profile(factor)
        slot = factor.slot
        feed = slot.bar_feed
    view = factor.session_view
    input_frequency = Frequency.parse(factor.data_frequency)
    output_frequency = Frequency.parse(factor.output_frequency)
    if not input_frequency.is_valid or not output_frequency.is_valid:
        raise ValueError("Factor persistence requires valid input and output frequencies")
    bar_output = factor._frequency_route.bar_grid and not _session_rollup(factor)
    panel = factor.get_factor_values() if factor_values is None else factor_values
    if panel.empty:
        raise ValueError("No factor values to persist")
    instruments = panel.columns.tolist()
    # Older 5m factors can emit one value per session through the wide API.
    if (not bar_output and output_frequency.kind == "intraday"
            and isinstance(panel.index, pd.DatetimeIndex) and panel.index.tz is None
            and panel.index.equals(panel.index.normalize())):
        output_frequency = Frequency.parse("1d")

    bar_grid = None
    if bar_output:
        bar_grid = getattr(factor, "market_bar_grid", None)
        if bar_grid is None:
            bar_grid = factor._fetch_market_bars()

    slot = factor.slot
    feed = slot.bar_feed
    provenance = (getattr(feed, "factor_write_provenance", None)
                  if factor._frequency_route.bar_grid or identity_scope == "incremental" else None)
    entries, builds = provenance(input_frequency, instruments) if provenance is not None else ({}, {})
    if callable(ensure_profile := getattr(feed, "ensure_profile", None)):
        ensure_profile(factor)
    slot = factor.slot
    feed = slot.bar_feed
    strategies = {
        bucket_strategy or getattr(entries.get(inst), "bucket_strategy", None)
        or _bucket_strategy(feed, input_frequency, inst) or "segment_anchored"
        for inst in instruments
    }
    if len(strategies) > 1:
        raise ValueError(f"Factor output spans multiple bucket strategies: {sorted(strategies)}")
    strategy = next(iter(strategies), "segment_anchored")
    if input_frequency.kind == "session":
        strategy = "session"
    values, calendars = _wide_values(
        factor, panel, output_frequency, "session" if output_frequency.kind == "session" else strategy,
        bar_grid=bar_grid,
    )
    if bar_grid is not None:
        incomplete = panel.attrs.get("incomplete_bar_keys")
        values = mask_incomplete_bars(values, incomplete if incomplete is not None else incomplete_bar_keys(bar_grid))

    slots_hash = slot.slot_config_hash or factor.slot_config_hash
    if slots_hash is None:
        components = (slot.calendar, feed, slot.universe, slot.aux_features, slot.microstructure)
        slots_hash = _digest({
            "components": [f"{type(part).__module__}.{type(part).__qualname__}" for part in components],
            "execution_profile": asdict(slot.microstructure.execution_profile),
        })
    elif not re.fullmatch(r"[0-9a-f]{64}", slots_hash):
        slots_hash = _digest(slots_hash)
    adjustment = slot.adjustment_policy
    if adjustment is None:
        adjustment = "dynamic_pit" if factor._ashare_panel and factor.pit_enabled and not bar_output else "none"
    inputs = dict(panel.attrs.get("factor_input_fingerprint", {}))
    captured_params = inputs.pop("declared_params", None)
    declared_params = factor_params
    if declared_params is None and captured_params is not None:
        declared_params = {
            key: FactorFrameDigest(value) if kind == "dataframe" else value
            for key, (kind, value) in captured_params.items()
        }
    params = {
        "dynamic_pool_mode": factor.dynamic_pool_mode if factor.use_dynamic_pool else None,
        "universe_id": factor.universe_id,
        "pit_factor_column": factor.pit_factor_column if factor.pit_enabled else None,
        "pit_window": factor.pit_window if factor.pit_enabled else None,
    }
    # The identity API already supports top-level DataFrame-valued dependencies.
    if declared_params is not None:
        if not isinstance(declared_params, Mapping) or any(not isinstance(key, str) for key in declared_params):
            raise TypeError("factor_params must be a mapping with string keys")
        params.update({f"factor.{key}": value for key, value in declared_params.items()})

    values = values.sort_values(["instrument_id", "timestamp_utc"], ignore_index=True)
    required = set(values[["instrument_id", "session_id"]].itertuples(index=False, name=None))
    build_map = _build_ids(required, entries, builds, source_build_ids)
    market_db = slot_market_db_identity(slot)
    registered = slot.slot_data_fingerprint
    if registered == market_db:
        registered = None  # Profile-only bindings still need an input or output revision.
    revision = registered
    if revision is None:
        revision = factor.source_revision or factor.data_version
    source_identity: dict[str, Any] = {"identity_scope": identity_scope}
    if entries:
        source_identity["catalog"] = {
            instrument: {name: getattr(entry, name) for name in ("dataset", "table_name", "kind", "source_frequency")}
            for instrument, entry in entries.items()
        }
    inferred_fingerprint: dict[str, Any] = {}
    if revision is not None:
        inferred_fingerprint["revision"] = revision
    if identity_scope == "incremental":
        if declared_params is None:
            raise ValueError("Incremental writes require factor_params or _get_cache_inputs()")
        if revision is None and required - build_map.keys():
            raise ValueError("Incremental writes require a slot_data_fingerprint or source_build_ids")
    else:
        if inputs:
            if registered is None:
                inferred_fingerprint.update(inputs)
            else:
                source_identity["factor_input_fingerprint"] = inputs
        if registered is None or declared_params is None:
            keys = _frame_digest(values[["instrument_id", "timestamp_utc", "session_id"]])
            bits = canonical_float_bits(values.factor_value).astype("<u8", copy=False).tobytes()
            digest = hashlib.sha256(keys.encode() + bits).hexdigest()
            if registered is None:
                inferred_fingerprint["realized_panel_sha256"] = digest
            elif declared_params is None:
                source_identity["realized_panel_sha256"] = digest

    fingerprint = registered
    if fingerprint is None:
        fingerprint = inferred_fingerprint or {"kind": "per_session_build_ids"}
    fingerprint = bind_market_db_fingerprint(fingerprint, market_db)
    fallback_build = _digest(fingerprint)
    nested_builds: dict[str, dict[str, str]] = {}
    for inst, session in sorted(required):
        nested_builds.setdefault(inst, {})[session] = build_map.get((inst, session), fallback_build)
    cal_hashes = {inst: calendar.calendar_hash for inst, calendar in calendars.items()}
    rth_refs = {
        inst: ref for inst, calendar in calendars.items()
        if (ref := getattr(getattr(calendar, "default", calendar), "rth_reference_calendar", None))
    }
    fid = validate_factor_id(factor_id or factor.factor_name)
    code_version, code_source = resolve_code_version()
    fhash, payload = compute_factor_identity_hash(
        factor_id=fid,
        factor_class=f"{type(factor).__module__}.{type(factor).__qualname__}",
        source_sha256=_factor_source_hashes(factor),
        params=params,
        frequency=input_frequency.canonical,
        output_frequency=output_frequency.canonical,
        session_view=view,
        analysis_mode=factor.analysis_mode or "cross_sectional",
        bucket_strategy=strategy,
        calendar_hash=cal_hashes or slot.calendar.calendar_hash,
        source_identity=source_identity,
        source_build_id=nested_builds if identity_scope == "exact" and nested_builds else None,
        data_version=factor.data_version,
        rth_reference_calendar=rth_refs or None,
        code_version=code_version,
        slot_id=slot.slot_id,
        slot_config_hash=slots_hash,
        adjustment_policy=adjustment,
        slot_data_fingerprint=fingerprint,
    )

    now = dt.datetime.now(dt.timezone.utc)
    values = values.assign(
        factor_id=fid, factor_identity_hash=fhash, frequency=output_frequency.canonical, session_view=view,
        source_build_id=[nested_builds[inst][session] for inst, session in
                         values[["instrument_id", "session_id"]].itertuples(index=False, name=None)],
        calendar_hash=values.instrument_id.map(cal_hashes), code_version=code_version,
        created_at=pd.Timestamp(now), batch_id=batch_id or uuid.uuid4().hex,
    )
    return store.append_factor_values(
        values.loc[:, list(TABLE_COLUMNS)],
        identity_record=build_identity_record(factor_identity_hash=fhash, payload=payload,
                                              code_version_source=code_source, created_at=now.isoformat()),
        calendar=calendars,
        strict_source_build_id=True,
    )


# Existing callers use this name for the same Slot-aware persistence API.
save_bar_factor_values = save_factor_values
