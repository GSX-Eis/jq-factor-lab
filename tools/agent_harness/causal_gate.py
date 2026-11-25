"""Automatic causal verification gate for Slot and FactorBase fast paths."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils.factor_base import FactorBase
from utils.factor_core import bar_bridge, slot_execution_mode, slot_fast_path

CAUSAL_GATE_SCHEMA_VERSION = "causal_fast_path_gate.v1"
_BAR_KEYS = ["timestamp_utc", "instrument_id"]
_AVAILABLE = "signal_available_at"
_KEY_OR_META_COLS = {
    "timestamp_utc",
    "instrument_id",
    "session_id",
    "observed",
    "frequency",
    "signal_available_at",
    "date",
    "code",
    "stock_code",
}
_SNAPSHOT_ATTRS = (
    "last_raw_data",
    "last_slot_execution_mode",
    "last_performance_metrics",
    "market_bars",
    "market_bar_grid",
    "instrument_metadata",
    "market_calendars",
    "calendar_hash",
    "rth_reference_calendar",
)


class CausalFastPathGateError(RuntimeError):
    """Raised when a factor fails the automatic causal verification gate."""

    code = "causal_fast_path_gate_failed"
    exit_code = 4

    def __init__(self, message: str, *, gate_result: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.gate_result = dict(gate_result)
        self.details = {"causal_gate": self.gate_result}


@dataclass(frozen=True)
class CausalGateResult:
    """Structured outcome and diagnostic evidence from the causal verification gate."""

    schema_version: str
    ok: bool
    causal_passed: bool | None
    status: str
    factor_name: str
    factor_class: str
    fast_path_requested: bool
    fast_path_allowed: bool
    requested_mode: str
    candidate_mode: str
    resolved_mode: str
    resolved_reason: str
    historical_revision: bool
    historical_revision_reason: str
    checks_performed: tuple[str, ...]
    seed: int
    truncation_cutoffs: tuple[str, ...]
    pit_sample_cutoffs: tuple[str, ...]
    max_abs_diff: float | None
    nan_mismatch_count: int
    failure_reason: str | None
    mismatched_samples: tuple[dict[str, Any], ...]
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks_performed"] = list(self.checks_performed)
        payload["truncation_cutoffs"] = list(self.truncation_cutoffs)
        payload["pit_sample_cutoffs"] = list(self.pit_sample_cutoffs)
        payload["mismatched_samples"] = list(self.mismatched_samples)
        return payload


def _snapshot_factor_state(factor: Any) -> dict[str, Any]:
    sentinel = object()
    return {name: getattr(factor, name, sentinel) for name in _SNAPSHOT_ATTRS}, sentinel


def _restore_factor_state(factor: Any, saved: tuple[dict[str, Any], object]) -> None:
    attrs, sentinel = saved
    for name, value in attrs.items():
        if value is sentinel:
            if hasattr(factor, name):
                delattr(factor, name)
        else:
            setattr(factor, name, value)


def _format_ts(ts: Any) -> str:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        return stamp.tz_convert("UTC").isoformat()
    return stamp.isoformat()


def _compare_long_values(
    actual_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    *,
    keys: list[str],
    cutoff_label: str,
    check_type: str,
    rtol: float,
    atol: float,
    max_samples: int = 5,
) -> tuple[bool, float | None, int, list[dict[str, Any]]]:
    actual = actual_df[[*keys, "factor_value"]].rename(columns={"factor_value": "_actual"})
    expected = expected_df[[*keys, "factor_value"]].rename(columns={"factor_value": "_expected"})
    merged = expected.merge(actual, on=keys, how="outer", validate="one_to_one")
    if merged.empty:
        return True, 0.0, 0, []

    exp_vals = merged["_expected"].to_numpy(dtype=float)
    act_vals = merged["_actual"].to_numpy(dtype=float)
    exp_na = np.isnan(exp_vals)
    act_na = np.isnan(act_vals)
    nan_mismatch_mask = exp_na != act_na
    nan_mismatch_count = int(nan_mismatch_mask.sum())

    both_valid = (~exp_na) & (~act_na)
    inf_mismatch_mask = np.zeros(len(merged), dtype=bool)
    value_mismatch_mask = np.zeros(len(merged), dtype=bool)
    max_abs_diff: float | None = 0.0 if both_valid.any() else None

    if both_valid.any():
        exp_v = exp_vals[both_valid]
        act_v = act_vals[both_valid]
        exp_fin = np.isfinite(exp_v)
        act_fin = np.isfinite(act_v)
        fin_mismatch = exp_fin != act_fin
        both_fin = exp_fin & act_fin
        if fin_mismatch.any():
            valid_idx = np.flatnonzero(both_valid)
            inf_mismatch_mask[valid_idx[fin_mismatch]] = True
        if both_fin.any():
            diffs = np.abs(act_v[both_fin] - exp_v[both_fin])
            max_abs_diff = float(np.max(diffs))
            tol = atol + rtol * np.abs(exp_v[both_fin])
            bad = diffs > tol
            if bad.any():
                fin_idx = np.flatnonzero(both_valid)[both_fin]
                value_mismatch_mask[fin_idx[bad]] = True
        else:
            max_abs_diff = None

        # Non-finite values with matching inf sign are allowed; opposite signs fail.
        both_inf = (~exp_fin) & (~act_fin)
        if both_inf.any():
            sign_bad = exp_v[both_inf] != act_v[both_inf]
            if sign_bad.any():
                inf_idx = np.flatnonzero(both_valid)[both_inf]
                inf_mismatch_mask[inf_idx[sign_bad]] = True

    any_mismatch = nan_mismatch_mask | inf_mismatch_mask | value_mismatch_mask
    passed = not bool(any_mismatch.any())
    samples: list[dict[str, Any]] = []
    if not passed:
        for idx in np.flatnonzero(any_mismatch)[:max_samples]:
            row = merged.iloc[idx]
            act_num = float(row["_actual"]) if np.isfinite(row["_actual"]) else None
            exp_num = float(row["_expected"]) if np.isfinite(row["_expected"]) else None
            sample: dict[str, Any] = {
                "check_type": check_type,
                "cutoff": cutoff_label,
                "instrument_id": str(row[keys[1]]),
                "fast_value": act_num if not np.isnan(row["_actual"]) else None,
                "reference_value": exp_num if not np.isnan(row["_expected"]) else None,
            }
            sample[keys[0]] = _format_ts(row[keys[0]])
            samples.append(sample)
    return passed, max_abs_diff, nan_mismatch_count, samples


def _wide_panel_to_long(panel: pd.DataFrame, *, key_name: str = "date") -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=[key_name, "instrument_id", "factor_value"])
    melted = (
        panel.rename_axis(index=key_name, columns="instrument_id")
        .reset_index()
        .melt(id_vars=key_name, var_name="instrument_id", value_name="factor_value")
    )
    melted[key_name] = pd.to_datetime(melted[key_name])
    melted["factor_value"] = pd.to_numeric(melted["factor_value"], errors="coerce")
    return melted


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in _KEY_OR_META_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]


def _find_boundary_cutoffs(
    grid: pd.DataFrame,
    clock: pd.DataFrame,
    interior: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    if grid is None or grid.empty or not interior:
        return []
    interior_set = set(interior)
    num_cols = _numeric_feature_columns(grid)
    keyed = grid.merge(clock[[*_BAR_KEYS, _AVAILABLE]], on=_BAR_KEYS, how="inner")
    if keyed.empty:
        return []
    keyed = keyed.sort_values(["instrument_id", "timestamp_utc"])
    is_gap = ~keyed["observed"].astype(bool) if "observed" in keyed else pd.Series(False, index=keyed.index)
    if num_cols:
        is_nan = keyed[num_cols].isna().any(axis=1)
        is_missing = is_gap | is_nan
    else:
        is_missing = is_gap
    found: list[pd.Timestamp] = []
    for _, inst_df in keyed.groupby("instrument_id", sort=False):
        miss_flags = is_missing.loc[inst_df.index].to_numpy(dtype=bool)
        avail_vals = inst_df[_AVAILABLE].tolist()
        for pos, flag in enumerate(miss_flags[:-1]):
            if flag and not miss_flags[pos + 1]:
                ts = pd.Timestamp(avail_vals[pos])
                if ts in interior_set and ts not in found:
                    found.append(ts)
    return found


def _select_cutoffs(
    candidates: list[pd.Timestamp],
    count: int,
    rng: np.random.Generator,
    *,
    priority: list[pd.Timestamp] | None = None,
) -> list[pd.Timestamp]:
    if not candidates or count <= 0:
        return []
    chosen: list[pd.Timestamp] = []
    candidate_set = set(candidates)
    for ts in priority or []:
        if ts in candidate_set and ts not in chosen:
            chosen.append(ts)
            if len(chosen) >= count:
                return sorted(chosen)
    remaining = [ts for ts in candidates if ts not in chosen]
    need = min(count - len(chosen), len(remaining))
    if need > 0:
        indices = rng.choice(len(remaining), size=need, replace=False)
        for idx in indices:
            chosen.append(remaining[int(idx)])
    return sorted(chosen)


def _merge_max_diff(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def _run_bar_bfill_nan_probe(
    factor: Any,
    full_grid: pd.DataFrame,
    clock: pd.DataFrame,
    probe_tau: pd.Timestamp,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, bool, float | None, int, list[dict[str, Any]], str | None]:
    """Mask an interior bar row with NaNs to catch bfill() even on dense inputs."""
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame

    num_cols = _numeric_feature_columns(full_grid)
    if not num_cols or full_grid.empty:
        return True, False, None, 0, [], None
    with_avail = full_grid.merge(clock[[*_BAR_KEYS, _AVAILABLE]], on=_BAR_KEYS, how="inner")
    target_mask = with_avail[_AVAILABLE].eq(probe_tau)
    if not target_mask.any():
        return True, False, None, 0, [], None

    probe_full = with_avail.copy()
    for col in num_cols:
        probe_full[col] = probe_full[col].astype(float)
        probe_full.loc[target_mask, col] = np.nan
    probe_trunc = probe_full.loc[probe_full[_AVAILABLE].le(probe_tau)].copy()
    grid_cols = list(full_grid.columns)

    try:
        full_raw = bar_factor_frame(factor.calculate_factor(probe_full[grid_cols]))
        trunc_raw = bar_factor_frame(factor.calculate_factor(probe_trunc[grid_cols]))
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        return True, False, None, 0, [], f"bfill_nan_probe_skipped:{type(exc).__name__}"

    trunc_keys = probe_trunc[_BAR_KEYS]
    full_slice = trunc_keys.merge(full_raw, on=_BAR_KEYS, how="left", validate="one_to_one")
    trunc_slice = trunc_keys.merge(trunc_raw, on=_BAR_KEYS, how="left", validate="one_to_one")
    passed, diff, nan_cnt, samples = _compare_long_values(
        full_slice,
        trunc_slice,
        keys=_BAR_KEYS,
        cutoff_label=_format_ts(probe_tau),
        check_type="bfill_nan_probe",
        rtol=rtol,
        atol=atol,
    )
    return passed, True, diff, nan_cnt, samples, None


def _run_panel_bfill_nan_probe(
    factor: Any,
    raw_data: pd.DataFrame,
    date_col: str,
    probe_tau: pd.Timestamp,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, bool, float | None, int, list[dict[str, Any]], str | None]:
    num_cols = _numeric_feature_columns(raw_data)
    if not num_cols or raw_data.empty:
        return True, False, None, 0, [], None
    dates = pd.to_datetime(raw_data[date_col])
    target_mask = dates.eq(probe_tau)
    if not target_mask.any():
        return True, False, None, 0, [], None

    probe_full = raw_data.copy()
    for col in num_cols:
        probe_full[col] = probe_full[col].astype(float)
        probe_full.loc[target_mask, col] = np.nan
    probe_trunc = probe_full.loc[dates.le(probe_tau)].copy()

    try:
        full_panel = factor._normalize_factor_dataframe(factor.calculate_factor(probe_full), probe_full)
        trunc_panel = factor._normalize_factor_dataframe(factor.calculate_factor(probe_trunc), probe_trunc)
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        return True, False, None, 0, [], f"bfill_nan_probe_skipped:{type(exc).__name__}"

    full_long = _wide_panel_to_long(full_panel, key_name="date")
    trunc_long = _wide_panel_to_long(trunc_panel, key_name="date")
    full_long = full_long.loc[full_long["date"].le(probe_tau)]
    passed, diff, nan_cnt, samples = _compare_long_values(
        full_long,
        trunc_long,
        keys=["date", "instrument_id"],
        cutoff_label=_format_ts(probe_tau),
        check_type="bfill_nan_probe",
        rtol=rtol,
        atol=atol,
    )
    return passed, True, diff, nan_cnt, samples, None


def _verify_non_ashare_slot_factor(
    factor: Any,
    *,
    candidate_mode: str,
    bar_grid: bool,
    n_truncations: int,
    n_pit_samples: int,
    seed: int,
    rtol: float,
    atol: float,
) -> tuple[bool, list[str], list[str], list[str], float | None, int, str | None, list[dict[str, Any]]]:
    clock = factor._slot_clock()
    if clock.empty:
        return True, ["empty_clock"], [], [], 0.0, 0, None, []

    cutoffs = list(pd.DatetimeIndex(clock[_AVAILABLE].drop_duplicates().sort_values()))
    interior = cutoffs[:-1] if len(cutoffs) >= 2 else cutoffs
    rng = np.random.default_rng(seed)

    if bar_grid:
        runner_fn = (
            (lambda c, t: slot_fast_path.run_vectorized_bar_slot(
                factor,
                c,
                t,
                read_slot_bars=bar_bridge._read_slot_bars,
                remember_market_bars=bar_bridge._remember_market_bars,
                pool_mask_fn=bar_bridge._pool_mask,
                remember=False,
            ))
            if candidate_mode == "vectorized"
            else (lambda c, t: slot_fast_path.run_incremental_bar_slot(
                factor,
                c,
                t,
                read_slot_bars=bar_bridge._read_slot_bars,
                remember_market_bars=bar_bridge._remember_market_bars,
                pool_mask_fn=bar_bridge._pool_mask,
                get_incremental_spec_fn=slot_execution_mode.get_incremental_spec,
                remember=False,
            ))
        )
    else:
        runner_fn = lambda c, t: slot_fast_path.run_vectorized_panel_slot(
            factor,
            c,
            t,
            panel_dates_fn=bar_bridge._panel_dates,
        )

    full_df = runner_fn(clock, clock)
    full_grid = getattr(factor, "last_raw_data", None)

    boundary_cutoffs = _find_boundary_cutoffs(full_grid, clock, interior) if bar_grid else []
    trunc_cutoffs = _select_cutoffs(interior, n_truncations, rng, priority=boundary_cutoffs)
    pit_cutoffs = _select_cutoffs(cutoffs, n_pit_samples, rng, priority=boundary_cutoffs)

    checks: list[str] = ["random_time_truncation"]
    max_abs_diff: float | None = None
    nan_mismatch_total = 0
    failure_reason: str | None = None
    mismatched_samples: list[dict[str, Any]] = []

    # 1. Random interior time truncation consistency
    for tau in trunc_cutoffs:
        sub_clock = clock.loc[clock[_AVAILABLE].le(tau)].copy()
        trunc_df = runner_fn(sub_clock, sub_clock)
        full_slice = sub_clock[_BAR_KEYS].merge(full_df, on=_BAR_KEYS, how="left", validate="one_to_one")
        passed, diff, nan_cnt, samples = _compare_long_values(
            full_slice,
            trunc_df,
            keys=_BAR_KEYS,
            cutoff_label=_format_ts(tau),
            check_type="random_time_truncation",
            rtol=rtol,
            atol=atol,
        )
        max_abs_diff = _merge_max_diff(max_abs_diff, diff)
        nan_mismatch_total += nan_cnt
        if not passed:
            if failure_reason is None:
                failure_reason = "lookahead_truncation_mismatch"
            mismatched_samples.extend(samples[: max(0, 5 - len(mismatched_samples))])

    # 2. Sampled PIT replay reference comparison
    if pit_cutoffs:
        sampled_targets = clock.loc[clock[_AVAILABLE].isin(pit_cutoffs)].copy()
        try:
            pit_df = bar_bridge._pit_replay_slot_values(
                factor,
                clock,
                sampled_targets,
                force_recalculate=True,
                remember=False,
                calculate=True,
            )
            checks.append("pit_replay_sample_comparison")
            full_slice = sampled_targets[_BAR_KEYS].merge(
                full_df, on=_BAR_KEYS, how="left", validate="one_to_one"
            )
            passed, diff, nan_cnt, samples = _compare_long_values(
                full_slice,
                pit_df,
                keys=_BAR_KEYS,
                cutoff_label=",".join(_format_ts(ts) for ts in pit_cutoffs[:3]),
                check_type="pit_replay_sample_comparison",
                rtol=rtol,
                atol=atol,
            )
            max_abs_diff = _merge_max_diff(max_abs_diff, diff)
            nan_mismatch_total += nan_cnt
            if not passed:
                if failure_reason is None:
                    failure_reason = "pit_replay_mismatch"
                mismatched_samples.extend(samples[: max(0, 5 - len(mismatched_samples))])
        except NotImplementedError:
            if candidate_mode != "incremental":
                raise
            checks.append("pit_replay_skipped_incremental_only")

    # 3. Synthetic NaN probe for bfill() on dense datasets
    if (
        failure_reason is None
        and candidate_mode == "vectorized"
        and len(cutoffs) >= 2
        and isinstance(full_grid, pd.DataFrame)
        and not full_grid.empty
    ):
        probe_tau = boundary_cutoffs[0] if boundary_cutoffs else interior[len(interior) // 2]
        if bar_grid:
            passed, ran, diff, nan_cnt, samples, skip_reason = _run_bar_bfill_nan_probe(
                factor, full_grid, clock, probe_tau, rtol=rtol, atol=atol
            )
        else:
            raw_date_col = "date" if "date" in full_grid.columns else "timestamp_utc"
            probe_date = (
                pd.Timestamp(bar_bridge._panel_dates(factor, clock.loc[clock[_AVAILABLE].eq(probe_tau)]).iloc[0])
                if raw_date_col == "date"
                else probe_tau
            )
            passed, ran, diff, nan_cnt, samples, skip_reason = _run_panel_bfill_nan_probe(
                factor, full_grid, raw_date_col, probe_date, rtol=rtol, atol=atol
            )
        if ran:
            checks.append("bfill_nan_probe")
            max_abs_diff = _merge_max_diff(max_abs_diff, diff)
            nan_mismatch_total += nan_cnt
            if not passed:
                failure_reason = "lookahead_bfill_nan_probe_mismatch"
                mismatched_samples.extend(samples[: max(0, 5 - len(mismatched_samples))])
        elif skip_reason:
            checks.append(skip_reason)

    return (
        failure_reason is None,
        checks,
        [_format_ts(ts) for ts in trunc_cutoffs],
        [_format_ts(ts) for ts in pit_cutoffs],
        max_abs_diff,
        nan_mismatch_total,
        failure_reason,
        mismatched_samples,
    )


def _verify_ashare_factor(
    factor: Any,
    *,
    bar_grid: bool,
    n_truncations: int,
    seed: int,
    rtol: float,
    atol: float,
) -> tuple[bool, list[str], list[str], list[str], float | None, int, str | None, list[dict[str, Any]]]:
    """Random time-truncation consistency check for A-share factors declaring CAUSAL_VECTORIZED."""
    from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame

    rng = np.random.default_rng(seed)
    if bar_grid:
        raw = factor._fetch_market_bars()
        if raw.empty:
            return True, ["empty_raw"], [], [], 0.0, 0, None, []
        time_col = "timestamp_utc"
        full_df = bar_factor_frame(factor.calculate_factor(raw))
        cutoffs = list(pd.DatetimeIndex(raw[time_col].drop_duplicates().sort_values()))
        interior = cutoffs[:-1] if len(cutoffs) >= 2 else cutoffs
        trunc_cutoffs = _select_cutoffs(interior, n_truncations, rng)
        checks = ["random_time_truncation"]
        max_abs_diff: float | None = None
        nan_mismatch_total = 0
        failure_reason: str | None = None
        mismatched_samples: list[dict[str, Any]] = []
        for tau in trunc_cutoffs:
            sub_raw = raw.loc[pd.to_datetime(raw[time_col]).le(tau)].copy()
            trunc_df = bar_factor_frame(factor.calculate_factor(sub_raw))
            full_slice = sub_raw[_BAR_KEYS].merge(full_df, on=_BAR_KEYS, how="left", validate="one_to_one")
            passed, diff, nan_cnt, samples = _compare_long_values(
                full_slice,
                trunc_df,
                keys=_BAR_KEYS,
                cutoff_label=_format_ts(tau),
                check_type="random_time_truncation",
                rtol=rtol,
                atol=atol,
            )
            max_abs_diff = _merge_max_diff(max_abs_diff, diff)
            nan_mismatch_total += nan_cnt
            if not passed:
                if failure_reason is None:
                    failure_reason = "lookahead_truncation_mismatch"
                mismatched_samples.extend(samples[: max(0, 5 - len(mismatched_samples))])
        return (
            failure_reason is None,
            checks,
            [_format_ts(ts) for ts in trunc_cutoffs],
            [],
            max_abs_diff,
            nan_mismatch_total,
            failure_reason,
            mismatched_samples,
        )

    raw = factor._get_raw_data_for_factor()
    if raw.empty:
        return True, ["empty_raw"], [], [], 0.0, 0, None, []
    if getattr(factor, "pit_enabled", False):
        raw = factor._apply_pit_adjustment(raw)
    raw_dates = pd.to_datetime(raw["date"])
    cutoffs = list(pd.DatetimeIndex(raw_dates.drop_duplicates().sort_values()))
    interior = cutoffs[:-1] if len(cutoffs) >= 2 else cutoffs
    trunc_cutoffs = _select_cutoffs(interior, n_truncations, rng)
    full_panel = factor._normalize_factor_dataframe(factor.calculate_factor(raw), raw)
    full_long = _wide_panel_to_long(full_panel, key_name="date")

    checks = ["random_time_truncation"]
    max_abs_diff = None
    nan_mismatch_total = 0
    failure_reason = None
    mismatched_samples = []
    for tau in trunc_cutoffs:
        sub_raw = raw.loc[raw_dates.le(tau)].copy()
        trunc_panel = factor._normalize_factor_dataframe(factor.calculate_factor(sub_raw), sub_raw)
        trunc_long = _wide_panel_to_long(trunc_panel, key_name="date")
        full_slice = full_long.loc[full_long["date"].le(tau)]
        passed, diff, nan_cnt, samples = _compare_long_values(
            full_slice,
            trunc_long,
            keys=["date", "instrument_id"],
            cutoff_label=_format_ts(tau),
            check_type="random_time_truncation",
            rtol=rtol,
            atol=atol,
        )
        max_abs_diff = _merge_max_diff(max_abs_diff, diff)
        nan_mismatch_total += nan_cnt
        if not passed:
            if failure_reason is None:
                failure_reason = "lookahead_truncation_mismatch"
            mismatched_samples.extend(samples[: max(0, 5 - len(mismatched_samples))])

    if failure_reason is None and len(cutoffs) >= 2:
        probe_tau = interior[len(interior) // 2]
        passed, ran, diff, nan_cnt, samples, skip_reason = _run_panel_bfill_nan_probe(
            factor, raw, "date", probe_tau, rtol=rtol, atol=atol
        )
        if ran:
            checks.append("bfill_nan_probe")
            max_abs_diff = _merge_max_diff(max_abs_diff, diff)
            nan_mismatch_total += nan_cnt
            if not passed:
                failure_reason = "lookahead_bfill_nan_probe_mismatch"
                mismatched_samples.extend(samples[: max(0, 5 - len(mismatched_samples))])
        elif skip_reason:
            checks.append(skip_reason)

    return (
        failure_reason is None,
        checks,
        [_format_ts(ts) for ts in trunc_cutoffs],
        [],
        max_abs_diff,
        nan_mismatch_total,
        failure_reason,
        mismatched_samples,
    )


def verify_factor_causality(
    factor: Any,
    *,
    requested_mode: str | None = None,
    n_truncations: int = 3,
    n_pit_samples: int = 5,
    seed: int = 42,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    on_failure: str = "reject",
    apply_to_factor: bool = True,
) -> CausalGateResult:
    """Verify causality via random time truncation and sampled PIT replay before enabling fast paths."""
    if on_failure not in ("reject", "fallback"):
        raise ValueError(f"on_failure must be 'reject' or 'fallback', got {on_failure!r}")
    if n_truncations < 1 or n_pit_samples < 1:
        raise ValueError("n_truncations and n_pit_samples must be >= 1")
    if not math.isfinite(rtol) or rtol < 0 or not math.isfinite(atol) or atol < 0:
        raise ValueError("rtol and atol must be finite and >= 0")

    started = time.perf_counter()
    factor_name = str(getattr(factor, "factor_name", type(factor).__name__))
    factor_class = f"{type(factor).__module__}.{type(factor).__qualname__}"
    requested = slot_execution_mode.validate_slot_execution_mode(
        requested_mode if requested_mode is not None else getattr(factor, "slot_execution_mode", "auto")
    )
    ashare_panel = bool(getattr(factor, "_ashare_panel", True))
    bar_grid = bool(getattr(getattr(factor, "_frequency_route", None), "bar_grid", False))
    fp_requested, candidate_mode, req_reason = slot_execution_mode.fast_path_request_status(
        factor, FactorBase, requested_mode=requested, bar_grid=bar_grid
    )

    has_rev = False
    rev_reason = "not_applicable"
    if not ashare_panel:
        has_rev, rev_reason = slot_execution_mode.slot_has_historical_revision(factor)

    def _finish(result: CausalGateResult, *, gate_passed: bool | None) -> CausalGateResult:
        if apply_to_factor:
            if gate_passed is not None:
                factor._causal_gate_passed = gate_passed
            factor._causal_gate_report = result.to_dict()
            if gate_passed is False and on_failure == "fallback" and not ashare_panel:
                factor.slot_execution_mode = "pit_replay"
        return result

    if requested == "pit_replay":
        return _finish(
            CausalGateResult(
                schema_version=CAUSAL_GATE_SCHEMA_VERSION,
                ok=True,
                causal_passed=None,
                status="skipped_pit_replay",
                factor_name=factor_name,
                factor_class=factor_class,
                fast_path_requested=False,
                fast_path_allowed=False,
                requested_mode=requested,
                candidate_mode="pit_replay",
                resolved_mode="pit_replay",
                resolved_reason="explicit_pit_replay",
                historical_revision=has_rev,
                historical_revision_reason=rev_reason,
                checks_performed=(),
                seed=seed,
                truncation_cutoffs=(),
                pit_sample_cutoffs=(),
                max_abs_diff=None,
                nan_mismatch_count=0,
                failure_reason=None,
                mismatched_samples=(),
                elapsed_sec=time.perf_counter() - started,
            ),
            gate_passed=None,
        )

    if has_rev:
        return _finish(
            CausalGateResult(
                schema_version=CAUSAL_GATE_SCHEMA_VERSION,
                ok=True,
                causal_passed=None,
                status="blocked_historical_revision",
                factor_name=factor_name,
                factor_class=factor_class,
                fast_path_requested=fp_requested,
                fast_path_allowed=False,
                requested_mode=requested,
                candidate_mode=candidate_mode,
                resolved_mode="pit_replay",
                resolved_reason=f"fallback_pit_replay:{rev_reason}",
                historical_revision=True,
                historical_revision_reason=rev_reason,
                checks_performed=("slot_historical_revision_check",),
                seed=seed,
                truncation_cutoffs=(),
                pit_sample_cutoffs=(),
                max_abs_diff=None,
                nan_mismatch_count=0,
                failure_reason=None,
                mismatched_samples=(),
                elapsed_sec=time.perf_counter() - started,
            ),
            gate_passed=None,
        )

    if not fp_requested:
        return _finish(
            CausalGateResult(
                schema_version=CAUSAL_GATE_SCHEMA_VERSION,
                ok=True,
                causal_passed=None,
                status="skipped_not_requested",
                factor_name=factor_name,
                factor_class=factor_class,
                fast_path_requested=False,
                fast_path_allowed=False,
                requested_mode=requested,
                candidate_mode="pit_replay",
                resolved_mode="pit_replay",
                resolved_reason=req_reason,
                historical_revision=False,
                historical_revision_reason=rev_reason,
                checks_performed=(),
                seed=seed,
                truncation_cutoffs=(),
                pit_sample_cutoffs=(),
                max_abs_diff=None,
                nan_mismatch_count=0,
                failure_reason=None,
                mismatched_samples=(),
                elapsed_sec=time.perf_counter() - started,
            ),
            gate_passed=None,
        )

    saved = _snapshot_factor_state(factor)
    try:
        if not ashare_panel:
            (
                causal_passed,
                checks,
                trunc_strs,
                pit_strs,
                max_diff,
                nan_cnt,
                failure_reason,
                samples,
            ) = _verify_non_ashare_slot_factor(
                factor,
                candidate_mode=candidate_mode,
                bar_grid=bar_grid,
                n_truncations=n_truncations,
                n_pit_samples=n_pit_samples,
                seed=seed,
                rtol=rtol,
                atol=atol,
            )
        else:
            (
                causal_passed,
                checks,
                trunc_strs,
                pit_strs,
                max_diff,
                nan_cnt,
                failure_reason,
                samples,
            ) = _verify_ashare_factor(
                factor,
                bar_grid=bar_grid,
                n_truncations=n_truncations,
                seed=seed,
                rtol=rtol,
                atol=atol,
            )
    finally:
        _restore_factor_state(factor, saved)

    if causal_passed:
        if not ashare_panel:
            resolved_mode, resolved_reason = slot_execution_mode.resolve_slot_execution_mode(
                factor,
                FactorBase,
                requested_mode=requested,
                bar_grid=bar_grid,
                causal_verified=True,
                require_causal_verification=True,
            )
        else:
            resolved_mode = candidate_mode
            resolved_reason = "causal_gate_passed"
        status = "passed"
        ok = True
        fast_allowed = True
    elif ashare_panel:
        # A-share panel/bar routes have no pit_replay execution path, so fallback cannot prevent leakage.
        resolved_mode = "rejected"
        resolved_reason = f"rejected_no_pit_replay:{failure_reason or 'causal_gate_failed'}"
        status = "failed_rejected"
        ok = False
        fast_allowed = False
    else:
        resolved_mode = "pit_replay"
        resolved_reason = f"fallback_pit_replay:{failure_reason or 'causal_gate_failed'}"
        status = "failed_fallback" if on_failure == "fallback" else "failed_rejected"
        ok = on_failure == "fallback"
        fast_allowed = False

    return _finish(
        CausalGateResult(
            schema_version=CAUSAL_GATE_SCHEMA_VERSION,
            ok=ok,
            causal_passed=causal_passed,
            status=status,
            factor_name=factor_name,
            factor_class=factor_class,
            fast_path_requested=True,
            fast_path_allowed=fast_allowed,
            requested_mode=requested,
            candidate_mode=candidate_mode,
            resolved_mode=resolved_mode,
            resolved_reason=resolved_reason,
            historical_revision=False,
            historical_revision_reason=rev_reason,
            checks_performed=tuple(checks),
            seed=seed,
            truncation_cutoffs=tuple(trunc_strs),
            pit_sample_cutoffs=tuple(pit_strs),
            max_abs_diff=max_diff,
            nan_mismatch_count=nan_cnt,
            failure_reason=failure_reason,
            mismatched_samples=tuple(samples),
            elapsed_sec=time.perf_counter() - started,
        ),
        gate_passed=causal_passed,
    )


def validate_causal_gate_evidence(evidence: Mapping[str, Any]) -> None:
    """Validate structured causal gate evidence embedded in Harness/run_result metadata."""
    if not isinstance(evidence, Mapping):
        raise ValueError("causal_gate evidence must be a mapping")
    required = {
        "schema_version",
        "ok",
        "causal_passed",
        "status",
        "fast_path_requested",
        "fast_path_allowed",
        "resolved_mode",
        "truncation_cutoffs",
        "pit_sample_cutoffs",
        "max_abs_diff",
        "nan_mismatch_count",
        "failure_reason",
    }
    missing = sorted(required - set(evidence.keys()))
    if missing:
        raise ValueError(f"causal_gate evidence missing required fields: {missing}")
    if evidence.get("schema_version") != CAUSAL_GATE_SCHEMA_VERSION:
        raise ValueError("causal_gate schema_version mismatch")
    if not isinstance(evidence.get("ok"), bool):
        raise ValueError("causal_gate ok must be a boolean")
    causal_passed_val = evidence.get("causal_passed")
    if causal_passed_val is not None and not isinstance(causal_passed_val, bool):
        raise ValueError("causal_gate causal_passed must be a boolean or None")
    if evidence.get("status") in ("passed", "failed_rejected", "failed_fallback") and not isinstance(
        causal_passed_val, bool
    ):
        raise ValueError("causal_gate causal_passed must be a boolean when verification ran")
    for list_key in ("truncation_cutoffs", "pit_sample_cutoffs"):
        if not isinstance(evidence.get(list_key), list):
            raise ValueError(f"causal_gate {list_key} must be a list")
    if type(evidence.get("nan_mismatch_count")) is not int or evidence["nan_mismatch_count"] < 0:
        raise ValueError("causal_gate nan_mismatch_count must be a non-negative integer")
