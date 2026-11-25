"""Single-read vectorized and sequential incremental Slot fast paths."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.data_revision import capture_factor_inputs, frame_digest
from utils.market_db.identity import slot_market_db_identity


_BAR_KEYS = ["timestamp_utc", "instrument_id"]
_AVAILABLE = "signal_available_at"


def _normalize_step_output(step_output: Any, bar_slice: pd.DataFrame) -> pd.DataFrame:
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame

    if isinstance(step_output, pd.DataFrame):
        return bar_factor_frame(step_output)
    if isinstance(step_output, pd.Series):
        aligned = step_output.reindex(bar_slice.index).to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "timestamp_utc": bar_slice["timestamp_utc"].to_numpy(),
                "instrument_id": bar_slice["instrument_id"].to_numpy(),
                "factor_value": aligned,
            }
        )
    if isinstance(step_output, Mapping):
        if len(bar_slice["timestamp_utc"].unique()) != 1:
            raise ValueError("Mapping step output requires a single timestamp per step")
        ts = bar_slice["timestamp_utc"].iloc[0]
        return pd.DataFrame(
            {
                "timestamp_utc": [ts] * len(bar_slice),
                "instrument_id": bar_slice["instrument_id"].to_numpy(),
                "factor_value": [float(step_output.get(inst, np.nan)) for inst in bar_slice["instrument_id"]],
            }
        )
    raise TypeError(f"Unsupported incremental step output type: {type(step_output).__name__}")


def run_vectorized_bar_slot(
    self,
    clock: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    read_slot_bars: Callable,
    remember_market_bars: Callable,
    pool_mask_fn: Callable,
    remember: bool,
) -> pd.DataFrame:
    """Single-read causal vectorized fast path for immutable non-A-share Bar slots."""
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame

    if targets.empty:
        if remember:
            remember_market_bars(self, *read_slot_bars(self, expected=clock))
        empty = targets.iloc[:0][_BAR_KEYS].assign(factor_value=pd.Series(dtype=float))
        empty.attrs["factor_input_fingerprint"] = {"declared_params": capture_factor_inputs(self)}
        return empty

    max_cutoff = targets[_AVAILABLE].max()
    effective_clock = clock.loc[clock[_AVAILABLE].le(max_cutoff)].sort_values(["instrument_id", "timestamp_utc"])
    full_bars, full_grid, calendars = read_slot_bars(self, expected=effective_clock, as_of=max_cutoff)
    input_sha256 = (
        frame_digest(full_grid)
        if getattr(self.slot, "slot_data_fingerprint", None) in (None, slot_market_db_identity(self.slot))
        else None
    )
    remember_market_bars(self, full_bars, full_grid, calendars)

    pool_mask = pool_mask_fn(self, full_grid) if (not self._ashare_panel and self.use_dynamic_pool) else None
    calculation_grid = (
        full_grid if pool_mask is None or self.dynamic_pool_mode == "factor" else full_grid.loc[pool_mask]
    )
    self.last_raw_data = calculation_grid
    declared = capture_factor_inputs(self)
    expected = full_grid[_BAR_KEYS]
    raw_values = (
        expected.iloc[:0].assign(factor_value=pd.Series(dtype=float))
        if calculation_grid.empty
        else bar_factor_frame(self.calculate_factor(calculation_grid))
    )
    checked = raw_values.merge(expected, on=_BAR_KEYS, how="left", indicator=True)
    if (checked._merge.eq("left_only") & checked.factor_value.notna()).any():
        raise ValueError("factor output contains keys outside the Session grid")
    values = expected.merge(raw_values, on=_BAR_KEYS, how="left", validate="one_to_one")
    values.loc[~full_grid.observed.to_numpy(), "factor_value"] = np.nan
    if pool_mask is not None and self.dynamic_pool_mode in ("factor", "both"):
        values.loc[~pool_mask, "factor_value"] = np.nan

    if remember and len(effective_clock) < len(clock):
        remember_market_bars(self, *read_slot_bars(self, expected=clock))

    result = targets[_BAR_KEYS].merge(values, on=_BAR_KEYS, how="left", validate="one_to_one")
    provenance: dict[str, Any] = {"declared_params": declared}
    if input_sha256 is not None:
        provenance["factor_input_sha256"] = input_sha256
    result.attrs["factor_input_fingerprint"] = provenance
    return result


def run_incremental_bar_slot(
    self,
    clock: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    read_slot_bars: Callable,
    remember_market_bars: Callable,
    pool_mask_fn: Callable,
    get_incremental_spec_fn: Callable,
    remember: bool,
) -> pd.DataFrame:
    """Single-read sequential incremental fast path for stateful causal Bar factors."""
    if targets.empty:
        if remember:
            remember_market_bars(self, *read_slot_bars(self, expected=clock))
        empty = targets.iloc[:0][_BAR_KEYS].assign(factor_value=pd.Series(dtype=float))
        empty.attrs["factor_input_fingerprint"] = {"declared_params": capture_factor_inputs(self)}
        return empty

    max_cutoff = targets[_AVAILABLE].max()
    effective_clock = clock.loc[clock[_AVAILABLE].le(max_cutoff)].sort_values(["instrument_id", "timestamp_utc"])
    full_bars, full_grid, calendars = read_slot_bars(self, expected=effective_clock, as_of=max_cutoff)
    input_sha256 = (
        frame_digest(full_grid)
        if getattr(self.slot, "slot_data_fingerprint", None) in (None, slot_market_db_identity(self.slot))
        else None
    )
    remember_market_bars(self, full_bars, full_grid, calendars)

    pool_mask = pool_mask_fn(self, full_grid) if (not self._ashare_panel and self.use_dynamic_pool) else None
    calculation_grid = (
        full_grid if pool_mask is None or self.dynamic_pool_mode == "factor" else full_grid.loc[pool_mask]
    )
    self.last_raw_data = calculation_grid
    declared = capture_factor_inputs(self)
    expected = full_grid[_BAR_KEYS]

    spec = get_incremental_spec_fn(self)
    use_builtin = spec is not None
    if use_builtin:
        engine = spec.create_engine() if hasattr(spec, "create_engine") else spec
        state = None
        step_fn = lambda _s, sl: (None, engine.step(sl))
    else:
        state = self.init_incremental_state()
        step_fn = self.step_incremental

    if calculation_grid.empty:
        values = expected.assign(factor_value=np.nan)
    else:
        with_avail = calculation_grid.merge(
            effective_clock[[*_BAR_KEYS, _AVAILABLE]],
            on=_BAR_KEYS,
            how="left",
            validate="one_to_one",
        )
        grid_cols = list(full_grid.columns)
        step_frames: list[pd.DataFrame] = []
        for _, group in with_avail.groupby(_AVAILABLE, sort=True):
            bar_slice = group[grid_cols]
            if use_builtin:
                _, step_df = step_fn(state, bar_slice)
                step_frames.append(step_df)
            else:
                state, raw_out = step_fn(state, bar_slice)
                step_df = _normalize_step_output(raw_out, bar_slice)
                slice_keys = pd.MultiIndex.from_frame(bar_slice[_BAR_KEYS])
                out_keys = pd.MultiIndex.from_frame(step_df[_BAR_KEYS])
                if not out_keys.isin(slice_keys).all():
                    raise ValueError("incremental step emitted keys outside current bar_slice")
                step_frames.append(step_df)

        combined = (
            pd.concat(step_frames, ignore_index=True)
            if step_frames
            else expected.iloc[:0].assign(factor_value=pd.Series(dtype=float))
        )
        values = expected.merge(combined, on=_BAR_KEYS, how="left", validate="one_to_one")

    values.loc[~full_grid.observed.to_numpy(), "factor_value"] = np.nan
    if pool_mask is not None and self.dynamic_pool_mode in ("factor", "both"):
        values.loc[~pool_mask, "factor_value"] = np.nan

    if remember and len(effective_clock) < len(clock):
        remember_market_bars(self, *read_slot_bars(self, expected=clock))

    result = targets[_BAR_KEYS].merge(values, on=_BAR_KEYS, how="left", validate="one_to_one")
    provenance: dict[str, Any] = {"declared_params": declared}
    if input_sha256 is not None:
        provenance["factor_input_sha256"] = input_sha256
    result.attrs["factor_input_fingerprint"] = provenance
    return result


def run_vectorized_panel_slot(
    self,
    clock: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    panel_dates_fn: Callable,
) -> pd.DataFrame:
    """Single-read causal vectorized fast path for non-bar-grid panel Slot factors."""
    result = targets[_BAR_KEYS].copy()
    empty = result.iloc[:0].assign(factor_value=pd.Series(dtype=float))
    if targets.empty:
        empty.attrs["factor_input_fingerprint"] = {"declared_params": capture_factor_inputs(self)}
        return empty

    max_cutoff = targets[_AVAILABLE].max()
    effective_clock = clock.loc[clock[_AVAILABLE].le(max_cutoff)].sort_values(["instrument_id", "timestamp_utc"])
    self._slot_input_window = (effective_clock, max_cutoff)
    try:
        raw = self._get_raw_data_for_factor()
    finally:
        del self._slot_input_window

    input_sha256 = (
        frame_digest(raw)
        if getattr(self.slot, "slot_data_fingerprint", None) in (None, slot_market_db_identity(self.slot))
        else None
    )
    self.last_raw_data = raw
    declared = capture_factor_inputs(self)
    if raw.empty:
        empty.attrs["factor_input_fingerprint"] = {
            "declared_params": declared,
            **({"factor_input_sha256": input_sha256} if input_sha256 is not None else {}),
        }
        return empty

    if self.pit_enabled:
        raw = self._apply_pit_adjustment(raw)
    values = self._normalize_factor_dataframe(self.calculate_factor(raw), raw)
    if values.empty:
        empty.attrs["factor_input_fingerprint"] = {
            "declared_params": declared,
            **({"factor_input_sha256": input_sha256} if input_sha256 is not None else {}),
        }
        return empty

    target_dates = panel_dates_fn(self, targets)
    panel = values.reindex(index=pd.DatetimeIndex(target_dates.unique()))
    result["factor_value"] = [
        panel.at[day, instrument] if instrument in panel.columns else np.nan
        for day, instrument in zip(target_dates, targets.instrument_id)
    ]
    provenance: dict[str, Any] = {"declared_params": declared}
    if input_sha256 is not None:
        provenance["factor_input_sha256"] = input_sha256
    result.attrs["factor_input_fingerprint"] = provenance
    return result


def dispatch_causal_slot_values(
    self,
    clock: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    bindings: tuple,
    read_slot_bars: Callable,
    remember_market_bars: Callable,
    pool_mask_fn: Callable,
    panel_dates_fn: Callable,
    pit_replay_fn: Callable,
    force_recalculate: bool = False,
    remember: bool = False,
    calculate: bool = True,
    execution_mode: str | None = None,
) -> pd.DataFrame:
    """Resolve mode, check pre-read cache, and run vectorized/incremental/PIT-replay path."""
    factor_base_cls, stock_filter_cls, apply_pit_adjustment, cache_mod, mode_mod, _ = bindings
    bar = self._frequency_route.bar_grid

    if not calculate:
        has_revision, _ = mode_mod.slot_has_historical_revision(self)
        if bar and not has_revision:
            as_of = clock[_AVAILABLE].max() if not clock.empty else None
            bars, grid, calendars = read_slot_bars(self, expected=clock, as_of=as_of)
            remember_market_bars(self, bars, grid.reset_index(drop=True), calendars)
            empty = targets.iloc[:0][_BAR_KEYS].assign(factor_value=pd.Series(dtype=float))
            empty.attrs["factor_input_fingerprint"] = {}
            return empty
        return pit_replay_fn(
            self, clock, targets, force_recalculate=force_recalculate, remember=remember, calculate=False
        )

    mode, reason = mode_mod.resolve_slot_execution_mode(
        self, factor_base_cls, requested_mode=execution_mode, bar_grid=bar
    )
    self.last_slot_execution_mode = mode
    metrics = dict(getattr(self, "last_performance_metrics", {}) or {})
    metrics.update({"slot_execution_mode": mode, "slot_execution_reason": reason, "slot_cache_hit": False})

    inc_spec = mode_mod.get_incremental_spec(self)
    pre_key = cache_mod.build_slot_pre_read_cache_key(
        self,
        factor_base_cls,
        stock_filter_cls,
        apply_pit_adjustment,
        clock,
        targets,
        mode,
        remember=remember,
        incremental_spec=inc_spec,
    )
    if not force_recalculate and pre_key is not None:
        cached = cache_mod.load_slot_pre_read_cache(self, pre_key, targets, remember=remember)
        if cached is not None:
            metrics["slot_cache_hit"] = True
            self.last_performance_metrics = metrics
            return cached

    if mode == "vectorized":
        if bar:
            result = run_vectorized_bar_slot(
                self,
                clock,
                targets,
                read_slot_bars=read_slot_bars,
                remember_market_bars=remember_market_bars,
                pool_mask_fn=pool_mask_fn,
                remember=remember,
            )
        else:
            result = run_vectorized_panel_slot(
                self,
                clock,
                targets,
                panel_dates_fn=panel_dates_fn,
            )
    elif mode == "incremental":
        result = run_incremental_bar_slot(
            self,
            clock,
            targets,
            read_slot_bars=read_slot_bars,
            remember_market_bars=remember_market_bars,
            pool_mask_fn=pool_mask_fn,
            get_incremental_spec_fn=mode_mod.get_incremental_spec,
            remember=remember,
        )
    else:
        result = pit_replay_fn(
            self,
            clock,
            targets,
            force_recalculate=force_recalculate,
            remember=remember,
            calculate=True,
        )

    self.last_performance_metrics = metrics
    if pre_key is not None:
        cache_mod.save_slot_pre_read_cache(self, pre_key, result, remember=remember)
    return result
