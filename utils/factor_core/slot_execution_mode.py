"""Tiered non-A-share Slot execution mode resolution and built-in incremental engines."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


SLOT_EXECUTION_MODES = ("auto", "vectorized", "incremental", "pit_replay")
RESOLVED_EXECUTION_MODES = ("vectorized", "incremental", "pit_replay")

_BAR_KEYS = ["timestamp_utc", "instrument_id"]
_SUPPORTED_ROLLING_AGGS = ("mean", "sum", "std", "min", "max", "median")


@runtime_checkable
class IncrementalFactorEngine(Protocol):
    """Stateful step-wise incremental causal engine for ordered Bar slices."""

    def step(self, bar_slice: pd.DataFrame) -> pd.DataFrame: ...


def _validate_non_bool_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class IncrementalRollingSpec:
    """Declarative causal rolling specification matching session_rolling semantics."""

    column: str
    window: int
    min_periods: int | None = None
    within_session: bool = True
    warmup_sessions: int = 0
    warmup_bars: int = 0
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("column must be a nonempty string")
        object.__setattr__(self, "window", _validate_non_bool_int("window", self.window, 1))
        if self.min_periods is not None:
            mp = _validate_non_bool_int("min_periods", self.min_periods, 0)
            if mp > self.window:
                raise ValueError("min_periods must not exceed window")
            object.__setattr__(self, "min_periods", mp)
        object.__setattr__(self, "within_session", bool(self.within_session))
        object.__setattr__(self, "warmup_sessions", _validate_non_bool_int("warmup_sessions", self.warmup_sessions, 0))
        object.__setattr__(self, "warmup_bars", _validate_non_bool_int("warmup_bars", self.warmup_bars, 0))
        if self.aggregation not in _SUPPORTED_ROLLING_AGGS:
            raise ValueError(f"aggregation must be one of {_SUPPORTED_ROLLING_AGGS}")

    def create_engine(self) -> "IncrementalRollingEngine":
        return IncrementalRollingEngine(self)


class IncrementalRollingEngine:
    """Incremental O(T) rolling engine numerically aligned with session_rolling."""

    def __init__(self, spec: IncrementalRollingSpec) -> None:
        self.spec = spec
        self._min_periods = spec.window if spec.min_periods is None else spec.min_periods
        self._buffers: dict[tuple[str, ...], deque[float]] = {}
        self._slot_counts: dict[str, int] = {}
        self._session_orders: dict[str, dict[str, int]] = {}

    def _aggregate(self, finite_vals: list[float]) -> float:
        count = len(finite_vals)
        if count == 0:
            if self._min_periods == 0 and self.spec.aggregation == "sum":
                return 0.0
            return float("nan")
        if count < self._min_periods:
            return float("nan")
        agg = self.spec.aggregation
        if agg == "sum":
            return float(math.fsum(finite_vals))
        if agg == "mean":
            return float(math.fsum(finite_vals) / count)
        if agg == "std":
            if count < 2:
                return float("nan")
            mean = math.fsum(finite_vals) / count
            var = math.fsum((x - mean) ** 2 for x in finite_vals) / (count - 1)
            return float(math.sqrt(var))
        if agg == "min":
            return float(min(finite_vals))
        if agg == "max":
            return float(max(finite_vals))
        return float(statistics_median(finite_vals))

    def step(self, bar_slice: pd.DataFrame) -> pd.DataFrame:
        if bar_slice.empty:
            return bar_slice.iloc[:0][_BAR_KEYS].assign(factor_value=pd.Series(dtype=float))
        ordered = bar_slice.sort_values(["instrument_id", "timestamp_utc"])
        col = self.spec.column
        if col not in ordered.columns:
            raise ValueError(f"bar_slice missing required column {col!r}")
        instruments = ordered["instrument_id"].tolist()
        sessions = ordered["session_id"].tolist()
        observed_flags = ordered["observed"].to_numpy(dtype=bool)
        raw_values = ordered[col].to_numpy(dtype=float, na_value=np.nan)

        out_values = np.full(len(ordered), np.nan, dtype=float)
        for idx, (inst, sess, obs, raw_val) in enumerate(
            zip(instruments, sessions, observed_flags, raw_values)
        ):
            slot_idx = self._slot_counts.get(inst, 0)
            self._slot_counts[inst] = slot_idx + 1
            sess_map = self._session_orders.setdefault(inst, {})
            sess_key = str(sess)
            sess_idx = sess_map.get(sess_key)
            if sess_idx is None:
                sess_idx = len(sess_map)
                sess_map[sess_key] = sess_idx

            buf_key = (inst, sess_key) if self.spec.within_session else (inst,)
            buf = self._buffers.get(buf_key)
            if buf is None:
                buf = deque(maxlen=self.spec.window)
                self._buffers[buf_key] = buf
            buf.append(float(raw_val) if not np.isnan(raw_val) else float("nan"))

            if not obs or slot_idx < self.spec.warmup_bars or sess_idx < self.spec.warmup_sessions:
                continue
            finite_vals = [v for v in buf if not math.isnan(v)]
            out_values[idx] = self._aggregate(finite_vals)

        return pd.DataFrame(
            {
                "timestamp_utc": ordered["timestamp_utc"].to_numpy(),
                "instrument_id": ordered["instrument_id"].to_numpy(),
                "factor_value": out_values,
            }
        )


def statistics_median(values: list[float]) -> float:
    import statistics

    return statistics.median(values)


@dataclass(frozen=True)
class IncrementalEMASpec:
    """Declarative causal EMA specification matching pandas Series.ewm(...).mean()."""

    column: str
    span: float | None = None
    alpha: float | None = None
    min_periods: int = 1
    adjust: bool = True
    within_session: bool = False
    warmup_sessions: int = 0
    warmup_bars: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("column must be a nonempty string")
        if (self.span is None) == (self.alpha is None):
            raise ValueError("Specify exactly one of span or alpha")
        if self.span is not None:
            if isinstance(self.span, bool) or not isinstance(self.span, Real) or not math.isfinite(self.span) or self.span < 1:
                raise ValueError("span must be finite and >= 1")
            object.__setattr__(self, "span", float(self.span))
        if self.alpha is not None:
            if (
                isinstance(self.alpha, bool)
                or not isinstance(self.alpha, Real)
                or not math.isfinite(self.alpha)
                or not (0.0 < float(self.alpha) <= 1.0)
            ):
                raise ValueError("alpha must be finite and in (0, 1]")
            object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "min_periods", _validate_non_bool_int("min_periods", self.min_periods, 0))
        object.__setattr__(self, "adjust", bool(self.adjust))
        object.__setattr__(self, "within_session", bool(self.within_session))
        object.__setattr__(self, "warmup_sessions", _validate_non_bool_int("warmup_sessions", self.warmup_sessions, 0))
        object.__setattr__(self, "warmup_bars", _validate_non_bool_int("warmup_bars", self.warmup_bars, 0))

    @property
    def effective_alpha(self) -> float:
        if self.alpha is not None:
            return self.alpha
        assert self.span is not None
        return 2.0 / (self.span + 1.0)

    def create_engine(self) -> "IncrementalEMAEngine":
        return IncrementalEMAEngine(self)


class IncrementalEMAEngine:
    """O(T) incremental EMA engine matching pandas ewm(ignore_na=False).mean()."""

    def __init__(self, spec: IncrementalEMASpec) -> None:
        self.spec = spec
        self._alpha = spec.effective_alpha
        self._decay = 1.0 - self._alpha
        # state per group:
        # - adjust=True:  [weighted_sum, weight_sum, valid_count]
        # - adjust=False: [ema_value, pending_decay_power, valid_count]
        self._state: dict[tuple[str, ...], list[float]] = {}
        self._slot_counts: dict[str, int] = {}
        self._session_orders: dict[str, dict[str, int]] = {}

    def step(self, bar_slice: pd.DataFrame) -> pd.DataFrame:
        if bar_slice.empty:
            return bar_slice.iloc[:0][_BAR_KEYS].assign(factor_value=pd.Series(dtype=float))
        ordered = bar_slice.sort_values(["instrument_id", "timestamp_utc"])
        col = self.spec.column
        if col not in ordered.columns:
            raise ValueError(f"bar_slice missing required column {col!r}")
        instruments = ordered["instrument_id"].tolist()
        sessions = ordered["session_id"].tolist()
        observed_flags = ordered["observed"].to_numpy(dtype=bool)
        raw_values = ordered[col].to_numpy(dtype=float, na_value=np.nan)

        out_values = np.full(len(ordered), np.nan, dtype=float)
        for idx, (inst, sess, obs, raw_val) in enumerate(
            zip(instruments, sessions, observed_flags, raw_values)
        ):
            slot_idx = self._slot_counts.get(inst, 0)
            self._slot_counts[inst] = slot_idx + 1
            sess_map = self._session_orders.setdefault(inst, {})
            sess_key = str(sess)
            sess_idx = sess_map.get(sess_key)
            if sess_idx is None:
                sess_idx = len(sess_map)
                sess_map[sess_key] = sess_idx

            key = (inst, sess_key) if self.spec.within_session else (inst,)
            st = self._state.get(key)
            if st is None:
                st = [0.0, 0.0 if self.spec.adjust else 1.0, 0]
                self._state[key] = st

            is_valid = not math.isnan(raw_val)
            if self.spec.adjust:
                st[0] = st[0] * self._decay + (float(raw_val) if is_valid else 0.0)
                st[1] = st[1] * self._decay + (1.0 if is_valid else 0.0)
                if is_valid:
                    st[2] += 1
                ema = st[0] / st[1] if st[1] > 0.0 else float("nan")
            else:
                if is_valid:
                    if st[2] == 0:
                        st[0] = float(raw_val)
                    else:
                        w_old = st[1] * self._decay
                        st[0] = (self._alpha * float(raw_val) + w_old * st[0]) / (self._alpha + w_old)
                    st[1] = 1.0
                    st[2] += 1
                elif st[2] > 0:
                    st[1] *= self._decay
                ema = st[0] if st[2] > 0 else float("nan")

            if (
                obs
                and slot_idx >= self.spec.warmup_bars
                and sess_idx >= self.spec.warmup_sessions
                and st[2] >= self.spec.min_periods
            ):
                out_values[idx] = ema

        return pd.DataFrame(
            {
                "timestamp_utc": ordered["timestamp_utc"].to_numpy(),
                "instrument_id": ordered["instrument_id"].to_numpy(),
                "factor_value": out_values,
            }
        )


def validate_slot_execution_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in SLOT_EXECUTION_MODES:
        raise ValueError(f"slot_execution_mode must be one of {SLOT_EXECUTION_MODES}, got {mode!r}")
    return mode


def slot_has_historical_revision(factor) -> tuple[bool, str]:
    """Conservatively decide whether a Slot's BarFeed/Universe can revise history."""
    slot = getattr(factor, "slot", None)
    if slot is None:
        return True, "missing_slot"

    policy = getattr(slot, "adjustment_policy", None)
    if policy not in (None, "none"):
        return True, f"adjustment_policy={policy}"

    feed = slot.bar_feed
    universe = slot.universe
    if getattr(slot, "has_historical_revision", None) is True:
        return True, "slot_has_historical_revision"
    if getattr(feed, "has_historical_revision", None) is True:
        return True, "feed_has_historical_revision"
    if getattr(universe, "has_historical_revision", None) is True:
        return True, "universe_has_historical_revision"

    from slots.declarative.slot import DeclarativeBarFeed, DeclarativeUniverse
    from slots.futures.slot import _FuturesData

    feed_static = (
        getattr(slot, "has_historical_revision", None) is False
        or getattr(feed, "has_historical_revision", None) is False
        or isinstance(feed, DeclarativeBarFeed)
        or (isinstance(feed, _FuturesData) and getattr(feed, "adjustment", None) == "none")
    )
    if not feed_static:
        return True, "unverified_bar_feed"

    universe_static = (
        getattr(universe, "has_historical_revision", None) is False
        or isinstance(universe, (DeclarativeUniverse, _FuturesData))
        or hasattr(universe, "instrument_metadata")
    )
    if not universe_static:
        return True, "unverified_universe"

    return False, "immutable_no_revision"


def get_incremental_spec(factor) -> Any:
    getter = getattr(factor, "get_incremental_engine", None)
    if callable(getter):
        spec = getter()
        if spec is not None:
            return spec
    return getattr(factor, "INCREMENTAL_ENGINE", None)


def factor_supports_incremental(factor, factor_base_cls: type) -> bool:
    if get_incremental_spec(factor) is not None:
        return True
    cls = type(factor)
    has_init = getattr(cls, "init_incremental_state", None) is not getattr(factor_base_cls, "init_incremental_state", None)
    has_step = getattr(cls, "step_incremental", None) is not getattr(factor_base_cls, "step_incremental", None)
    return has_init and has_step


def resolve_slot_execution_mode(
    factor,
    factor_base_cls: type,
    *,
    requested_mode: str | None = None,
    bar_grid: bool = True,
) -> tuple[str, str]:
    """Resolve safe non-A-share Slot execution mode and human-readable rationale."""
    requested = validate_slot_execution_mode(
        requested_mode if requested_mode is not None else getattr(factor, "slot_execution_mode", "auto")
    )
    if requested == "pit_replay":
        return "pit_replay", "explicit_pit_replay"

    has_revision, revision_reason = slot_has_historical_revision(factor)
    if has_revision:
        if requested in ("vectorized", "incremental"):
            factor.logger.warning(
                "Slot execution mode %s rejected due to %s; falling back to pit_replay",
                requested,
                revision_reason,
            )
        return "pit_replay", f"fallback_pit_replay:{revision_reason}"

    supports_inc = bar_grid and factor_supports_incremental(factor, factor_base_cls)
    causal_flag = getattr(factor, "CAUSAL_VECTORIZED", None)

    if requested == "incremental":
        if not supports_inc:
            factor.logger.warning(
                "Slot execution mode incremental unavailable for %s (bar_grid=%s); falling back to pit_replay",
                type(factor).__name__,
                bar_grid,
            )
            return "pit_replay", "fallback_pit_replay:incremental_not_implemented"
        return "incremental", "explicit_incremental"

    if requested == "vectorized":
        if causal_flag is not True:
            reason = "non_causal_factor" if causal_flag is False else "unverified_factor_causality"
            factor.logger.warning(
                "Slot execution mode vectorized blocked because CAUSAL_VECTORIZED is %r (requires True); "
                "falling back to pit_replay",
                causal_flag,
            )
            return "pit_replay", f"fallback_pit_replay:{reason}"
        return "vectorized", "explicit_vectorized"

    # requested == "auto"
    if supports_inc:
        return "incremental", "auto_incremental"
    if causal_flag is True:
        return "vectorized", "auto_causal_vectorized"
    return "pit_replay", "fallback_pit_replay:unverified_factor_causality"
