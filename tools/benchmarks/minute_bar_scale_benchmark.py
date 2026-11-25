"""Parameterized minute-bar scale benchmark for non-A-share MarketSlot fast paths.

Standalone one-off / on-demand scale benchmark CLI; full-year minute runs are
intentionally decoupled from pytest / tools.test_runner routine dev suites.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

from backtest_and_analysis.time_series.factor_bridge import bar_factor_frame
from slots import build_slot_from_config
from utils.factor_base import FactorBase
from utils.factor_core import bar_bridge
from utils.factor_core.slot_execution_mode import IncrementalEMASpec, IncrementalRollingSpec
from utils.local_outputs import json_compatible, write_json
from utils.session_calendar import SessionCalendar
from utils.slots import ExecutionProfile, MarketSlot, ProviderConfig, SlotConfig
from utils.time_axis import BarSchedule, Frequency

SCHEMA_VERSION = "minute_bar_scale_benchmark.v1"
_BAR_KEYS = ["timestamp_utc", "instrument_id"]
_AVAILABLE = "signal_available_at"
_CALENDAR_MODES = ("trading_session", "crypto_24x7")
_SOURCE_FORMATS = ("parquet", "csv")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated parameterization for the minute-bar scale benchmark."""

    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    frequency: str = "1m"
    calendar_mode: str = "trading_session"
    source_format: str = "parquet"
    num_instruments: int = 5
    num_factors: int = 100
    incremental_count: int = 10
    pit_sample_cutoffs: int = 12
    pit_sample_factors: int = 4
    with_gaps_and_nans: bool = True
    enable_cache: bool = True
    warm_pass: bool = True
    enable_tracemalloc: bool = True
    seed: int = 42
    rtol: float = 1e-10
    atol: float = 1e-12
    work_dir: Path | None = None
    cleanup_work_dir: bool = True
    json_output: Path | None = None
    md_output: Path | None = None

    def __post_init__(self) -> None:
        start_ts = pd.Timestamp(self.start_date)
        end_ts = pd.Timestamp(self.end_date)
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
            raise ValueError(f"Invalid date range: start_date={self.start_date!r}, end_date={self.end_date!r}")
        freq = Frequency.parse(self.frequency)
        if not freq.is_valid or freq.kind != "intraday":
            raise ValueError(f"frequency must be a valid intraday minute/hour frequency, got {self.frequency!r}")
        if self.calendar_mode not in _CALENDAR_MODES:
            raise ValueError(f"calendar_mode must be one of {_CALENDAR_MODES}, got {self.calendar_mode!r}")
        if self.source_format not in _SOURCE_FORMATS:
            raise ValueError(f"source_format must be one of {_SOURCE_FORMATS}, got {self.source_format!r}")
        if self.num_instruments < 1:
            raise ValueError("num_instruments must be >= 1")
        if self.num_factors < 1:
            raise ValueError("num_factors must be >= 1")
        if self.incremental_count < 0 or self.incremental_count > self.num_factors:
            raise ValueError("incremental_count must satisfy 0 <= incremental_count <= num_factors")
        if self.pit_sample_cutoffs < 0:
            raise ValueError("pit_sample_cutoffs must be >= 0")
        if self.pit_sample_factors < 0:
            raise ValueError("pit_sample_factors must be >= 0")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("rtol and atol must be nonnegative")


class _BenchRollingVecFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def __init__(
        self,
        *args: Any,
        column: str = "close",
        window: int = 10,
        min_periods: int | None = 2,
        within_session: bool = True,
        warmup_bars: int = 1,
        aggregation: str = "mean",
        **kwargs: Any,
    ) -> None:
        self.column = column
        self.window = window
        self.min_periods = min_periods
        self.within_session = within_session
        self.warmup_bars = warmup_bars
        self.aggregation = aggregation
        super().__init__(*args, **kwargs)

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(
            data,
            self.column,
            self.window,
            min_periods=self.min_periods,
            within_session=self.within_session,
            warmup_bars=self.warmup_bars,
            aggregation=self.aggregation,
        )
        return data[_BAR_KEYS].assign(factor_value=vals)

    def _get_cache_inputs(self) -> dict[str, Any]:
        return {
            "family": "rolling_vec",
            "column": self.column,
            "window": self.window,
            "min_periods": self.min_periods,
            "within_session": self.within_session,
            "warmup_bars": self.warmup_bars,
            "aggregation": self.aggregation,
        }


class _BenchRollingIncFactor(_BenchRollingVecFactor):
    CAUSAL_VECTORIZED = False

    def get_incremental_engine(self) -> IncrementalRollingSpec:
        return IncrementalRollingSpec(
            column=self.column,
            window=self.window,
            min_periods=self.min_periods,
            within_session=self.within_session,
            warmup_bars=self.warmup_bars,
            aggregation=self.aggregation,
        )

    def _get_cache_inputs(self) -> dict[str, Any]:
        return {**super()._get_cache_inputs(), "family": "rolling_inc"}


class _BenchEMAVecFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def __init__(
        self,
        *args: Any,
        column: str = "close",
        span: float = 12.0,
        min_periods: int = 2,
        adjust: bool = True,
        within_session: bool = False,
        warmup_bars: int = 1,
        **kwargs: Any,
    ) -> None:
        self.column = column
        self.span = span
        self.min_periods = min_periods
        self.adjust = adjust
        self.within_session = within_session
        self.warmup_bars = warmup_bars
        super().__init__(*args, **kwargs)

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        groups = ["instrument_id", "session_id"] if self.within_session else ["instrument_id"]
        ema = ordered.groupby(groups, sort=False)[self.column].transform(
            lambda s: s.ewm(
                span=self.span,
                adjust=self.adjust,
                min_periods=self.min_periods,
                ignore_na=False,
            ).mean()
        )
        slots = ordered.groupby("instrument_id").cumcount()
        ema.loc[(slots < self.warmup_bars) | ~ordered["observed"]] = np.nan
        return ordered[_BAR_KEYS].assign(factor_value=ema)

    def _get_cache_inputs(self) -> dict[str, Any]:
        return {
            "family": "ema_vec",
            "column": self.column,
            "span": self.span,
            "min_periods": self.min_periods,
            "adjust": self.adjust,
            "within_session": self.within_session,
            "warmup_bars": self.warmup_bars,
        }


class _BenchEMAIncFactor(_BenchEMAVecFactor):
    CAUSAL_VECTORIZED = False

    def get_incremental_engine(self) -> IncrementalEMASpec:
        return IncrementalEMASpec(
            column=self.column,
            span=self.span,
            min_periods=self.min_periods,
            adjust=self.adjust,
            within_session=self.within_session,
            warmup_bars=self.warmup_bars,
        )

    def _get_cache_inputs(self) -> dict[str, Any]:
        return {**super()._get_cache_inputs(), "family": "ema_inc"}


class _BenchMomentumVecFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def __init__(
        self,
        *args: Any,
        lag: int = 5,
        within_session: bool = False,
        log_return: bool = False,
        warmup_bars: int = 1,
        **kwargs: Any,
    ) -> None:
        self.lag = lag
        self.within_session = within_session
        self.log_return = log_return
        self.warmup_bars = warmup_bars
        super().__init__(*args, **kwargs)

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        groups = ["instrument_id", "session_id"] if self.within_session else ["instrument_id"]
        prev = ordered.groupby(groups, sort=False)["close"].shift(self.lag)
        ratio = ordered["close"] / prev.where(prev > 0.0)
        vals = np.log(ratio.where(ratio > 0.0)) if self.log_return else (ratio - 1.0)
        slots = ordered.groupby("instrument_id").cumcount()
        vals.loc[(slots < self.warmup_bars) | ~ordered["observed"]] = np.nan
        return ordered[_BAR_KEYS].assign(factor_value=vals)

    def _get_cache_inputs(self) -> dict[str, Any]:
        return {
            "family": "momentum_vec",
            "lag": self.lag,
            "within_session": self.within_session,
            "log_return": self.log_return,
            "warmup_bars": self.warmup_bars,
        }


class _BenchRangeLiquidityVecFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def __init__(
        self,
        *args: Any,
        mode: str = "range_ratio",
        window: int = 15,
        min_periods: int = 3,
        within_session: bool = True,
        warmup_bars: int = 1,
        **kwargs: Any,
    ) -> None:
        self.mode = mode
        self.window = window
        self.min_periods = min_periods
        self.within_session = within_session
        self.warmup_bars = warmup_bars
        super().__init__(*args, **kwargs)

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        if self.mode == "range_ratio":
            raw_sig = (ordered["high"] - ordered["low"]) / ordered["close"].where(ordered["close"] != 0.0)
        else:
            raw_sig = ordered["close"] * ordered["volume"]
        enriched = ordered.assign(_bench_sig=raw_sig)
        vals = self.session_rolling(
            enriched,
            "_bench_sig",
            self.window,
            min_periods=self.min_periods,
            within_session=self.within_session,
            warmup_bars=self.warmup_bars,
            aggregation="mean",
        )
        return enriched[_BAR_KEYS].assign(factor_value=vals)

    def _get_cache_inputs(self) -> dict[str, Any]:
        return {
            "family": "range_liquidity_vec",
            "mode": self.mode,
            "window": self.window,
            "min_periods": self.min_periods,
            "within_session": self.within_session,
            "warmup_bars": self.warmup_bars,
        }


@dataclass(frozen=True)
class _FactorSpec:
    name: str
    family: str
    cls: type[FactorBase]
    params: dict[str, Any]


def build_factor_specs(num_factors: int, incremental_count: int) -> list[_FactorSpec]:
    """Build deterministic causal vectorized and incremental factor specs."""
    vec_count = num_factors - incremental_count
    specs: list[_FactorSpec] = []

    rolling_pool: list[tuple[str, int, str, bool]] = []
    for col in ("close", "volume", "high", "low"):
        for agg in ("mean", "sum", "std", "min", "max", "median"):
            for win in (5, 10, 20, 40, 60, 120):
                for within in (True, False):
                    rolling_pool.append((col, win, agg, within))

    ema_pool: list[tuple[str, float, bool, bool]] = []
    for col in ("close", "volume", "high", "low"):
        for span in (4.0, 8.0, 12.0, 24.0, 48.0, 96.0):
            for adjust in (True, False):
                for within in (False, True):
                    ema_pool.append((col, span, adjust, within))

    mom_pool: list[tuple[int, bool, bool]] = []
    for lag in (1, 2, 3, 5, 10, 15, 30, 60, 120):
        for within in (False, True):
            for log_ret in (False, True):
                mom_pool.append((lag, within, log_ret))

    range_pool: list[tuple[str, int, bool]] = []
    for mode in ("range_ratio", "dollar_volume"):
        for win in (5, 10, 15, 30, 60, 120):
            for within in (True, False):
                range_pool.append((mode, win, within))

    # Build vectorized factors cycling across all 4 vectorized families.
    family_cursors = [0, 0, 0, 0]
    for idx in range(vec_count):
        bucket = idx % 4
        if bucket == 0:
            col, win, agg, within = rolling_pool[family_cursors[0] % len(rolling_pool)]
            family_cursors[0] += 1
            specs.append(
                _FactorSpec(
                    name=f"f{idx:03d}_roll_vec_{col}_{agg}_w{win}_{'s' if within else 'g'}",
                    family="rolling_vec",
                    cls=_BenchRollingVecFactor,
                    params={
                        "column": col,
                        "window": win,
                        "min_periods": min(2, win),
                        "within_session": within,
                        "warmup_bars": 1,
                        "aggregation": agg,
                    },
                )
            )
        elif bucket == 1:
            col, span, adjust, within = ema_pool[family_cursors[1] % len(ema_pool)]
            family_cursors[1] += 1
            specs.append(
                _FactorSpec(
                    name=f"f{idx:03d}_ema_vec_{col}_s{int(span)}_{'adj' if adjust else 'raw'}_{'s' if within else 'g'}",
                    family="ema_vec",
                    cls=_BenchEMAVecFactor,
                    params={
                        "column": col,
                        "span": span,
                        "min_periods": 2,
                        "adjust": adjust,
                        "within_session": within,
                        "warmup_bars": 1,
                    },
                )
            )
        elif bucket == 2:
            lag, within, log_ret = mom_pool[family_cursors[2] % len(mom_pool)]
            family_cursors[2] += 1
            specs.append(
                _FactorSpec(
                    name=f"f{idx:03d}_mom_vec_l{lag}_{'log' if log_ret else 'pct'}_{'s' if within else 'g'}",
                    family="momentum_vec",
                    cls=_BenchMomentumVecFactor,
                    params={
                        "lag": lag,
                        "within_session": within,
                        "log_return": log_ret,
                        "warmup_bars": 1,
                    },
                )
            )
        else:
            mode, win, within = range_pool[family_cursors[3] % len(range_pool)]
            family_cursors[3] += 1
            specs.append(
                _FactorSpec(
                    name=f"f{idx:03d}_rng_vec_{mode}_w{win}_{'s' if within else 'g'}",
                    family="range_liquidity_vec",
                    cls=_BenchRangeLiquidityVecFactor,
                    params={
                        "mode": mode,
                        "window": win,
                        "min_periods": 2,
                        "within_session": within,
                        "warmup_bars": 1,
                    },
                )
            )

    inc_roll_pool = [
        ("close", 5, "mean", True),
        ("close", 15, "sum", False),
        ("volume", 10, "std", True),
        ("high", 20, "max", False),
        ("low", 20, "min", True),
        ("close", 9, "median", False),
    ]
    inc_ema_pool = [
        ("close", 6.0, True, False),
        ("close", 12.0, False, True),
        ("volume", 10.0, True, False),
        ("high", 18.0, False, False),
    ]
    for j in range(incremental_count):
        idx = vec_count + j
        if j % 2 == 0:
            col, win, agg, within = inc_roll_pool[(j // 2) % len(inc_roll_pool)]
            win_adj = win + (j // len(inc_roll_pool))
            specs.append(
                _FactorSpec(
                    name=f"f{idx:03d}_roll_inc_{col}_{agg}_w{win_adj}_{'s' if within else 'g'}",
                    family="rolling_inc",
                    cls=_BenchRollingIncFactor,
                    params={
                        "column": col,
                        "window": win_adj,
                        "min_periods": 2,
                        "within_session": within,
                        "warmup_bars": 1,
                        "aggregation": agg,
                    },
                )
            )
        else:
            col, span, adjust, within = inc_ema_pool[(j // 2) % len(inc_ema_pool)]
            span_adj = span + float(j // len(inc_ema_pool))
            specs.append(
                _FactorSpec(
                    name=f"f{idx:03d}_ema_inc_{col}_s{int(span_adj)}_{'adj' if adjust else 'raw'}_{'s' if within else 'g'}",
                    family="ema_inc",
                    cls=_BenchEMAIncFactor,
                    params={
                        "column": col,
                        "span": span_adj,
                        "min_periods": 2,
                        "adjust": adjust,
                        "within_session": within,
                        "warmup_bars": 1,
                    },
                )
            )
    return specs


def _custom_sessions_for_mode(calendar_mode: str) -> dict[str, Any]:
    if calendar_mode == "crypto_24x7":
        return {
            "timezone": "UTC",
            "weekdays": [0, 1, 2, 3, 4, 5, 6],
            "segments": [{"open": "00:00", "close": "00:00", "open_day_offset": 0, "close_day_offset": 1}],
        }
    # trading_session: Monday-Friday 09:30-16:00 UTC (390 1m bars/day)
    return {
        "timezone": "UTC",
        "weekdays": [0, 1, 2, 3, 4],
        "segments": [{"open": "09:30", "close": "16:00", "open_day_offset": 0, "close_day_offset": 0}],
    }


def _build_synthetic_slot(
    cfg: BenchmarkConfig,
    data_dir: Path,
) -> tuple[MarketSlot, dict[str, Any], list[str]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    custom_sessions = _custom_sessions_for_mode(cfg.calendar_mode)
    calendar_id = "bench_crypto_utc" if cfg.calendar_mode == "crypto_24x7" else "bench_session_utc"
    cal = SessionCalendar(
        calendar_id,
        cfg.start_date,
        cfg.end_date,
        custom_sessions=custom_sessions,
    )
    freq = Frequency.parse(cfg.frequency)
    base_grid = cal.bar_grid(freq, cfg.start_date, cfg.end_date, view="ETH", bucket_strategy="segment_anchored")
    if base_grid.empty:
        raise ValueError("Calendar produced an empty bar grid for the requested date range and calendar_mode")

    n_bars = len(base_grid)
    instruments = [f"SYM-{i + 1:02d}" for i in range(cfg.num_instruments)]
    rng = np.random.default_rng(cfg.seed)

    parts: list[pd.DataFrame] = []
    gap_rows = 0
    nan_rows = 0
    times_ns = base_grid["timestamp_utc"].to_numpy()

    for inst_idx, inst in enumerate(instruments):
        ret = rng.normal(0.0, 0.0008, size=n_bars)
        close = (100.0 + 15.0 * inst_idx) * np.exp(np.cumsum(ret))
        open_px = np.empty_like(close)
        open_px[0] = 100.0 + 15.0 * inst_idx
        open_px[1:] = close[:-1]
        spread = np.abs(rng.normal(0.001, 0.0005, size=n_bars)) * close
        high = np.maximum(open_px, close) + spread
        low = np.maximum(0.01, np.minimum(open_px, close) - spread)
        volume = 100.0 + rng.integers(1, 500, size=n_bars).astype(float)

        keep_mask = np.ones(n_bars, dtype=bool)
        if cfg.with_gaps_and_nans and n_bars >= 10:
            # Deterministic sparse gaps on instrument 0 and sparse NaNs on close.
            if inst_idx == 0:
                gap_step = max(7, min(503, n_bars // 4))
                keep_mask[gap_step - 1 :: gap_step] = False
                gap_rows += int((~keep_mask).sum())
            nan_step = max(11, min(617, n_bars // 3))
            nan_mask = np.zeros(n_bars, dtype=bool)
            nan_mask[nan_step - 1 :: nan_step] = True
            nan_mask &= keep_mask
            close = close.copy()
            close[nan_mask] = np.nan
            nan_rows += int(nan_mask.sum())

        df_inst = pd.DataFrame(
            {
                "time": pd.DatetimeIndex(times_ns[keep_mask], tz="UTC"),
                "instrument_id": inst,
                "open": open_px[keep_mask],
                "high": high[keep_mask],
                "low": low[keep_mask],
                "close": close[keep_mask],
                "volume": volume[keep_mask],
            }
        )
        parts.append(df_inst)

    bars_df = pd.concat(parts, ignore_index=True)

    if cfg.source_format == "parquet":
        data_file = data_dir / "bars.parquet"
        bars_df.to_parquet(data_file, index=False)
        bar_feed_cfg = ProviderConfig(
            "parquet_glob",
            {
                "path": "bars.parquet",
                "frequency": cfg.frequency,
                "timestamp_anchor": "bar_start",
                "source_timezone": "UTC",
                "volume_mode": "interval",
                "data_version": "bench-v1",
                "column_mapping": {
                    "time": "timestamp",
                    "instrument_id": "instrument_id",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
            },
        )
    else:
        data_file = data_dir / "bars.csv"
        csv_df = bars_df.copy()
        csv_df["time"] = csv_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        csv_df.to_csv(data_file, index=False)
        bar_feed_cfg = ProviderConfig(
            "csv_files",
            {
                "paths": ["bars.csv"],
                "frequency": cfg.frequency,
                "timestamp_anchor": "bar_start",
                "source_timezone": "UTC",
                "volume_mode": "interval",
                "data_version": "bench-v1",
                "column_mapping": {
                    "time": "timestamp",
                    "instrument_id": "instrument_id",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
            },
        )

    dataset_bytes = data_file.stat().st_size
    file_digest = hashlib.sha256(data_file.read_bytes()).hexdigest()[:16]

    slot_cfg = SlotConfig(
        schema_version=1,
        slot_id=f"bench_{cfg.calendar_mode}_{freq.canonical}",
        calendar=ProviderConfig(
            "custom_sessions",
            {
                "calendar_id": calendar_id,
                "start": cfg.start_date,
                "end": cfg.end_date,
                "custom_sessions": custom_sessions,
            },
        ),
        bar_feed=bar_feed_cfg,
        universe=ProviderConfig(
            "static",
            {
                "instruments": {
                    inst: {
                        "asset_class": "crypto" if cfg.calendar_mode == "crypto_24x7" else "equity",
                        "currency": "USD",
                        "lot_size": 1,
                        "contract_multiplier": 1.0,
                    }
                    for inst in instruments
                },
                "universes": {"bench": instruments},
            },
        ),
        aux_features=ProviderConfig("none", {}),
        microstructure=ProviderConfig("open", {}),
        execution_profile=ExecutionProfile(
            "open", "close", 1, 0, "decimal", "endpoints_only", "cross_session"
        ),
    )
    slot = build_slot_from_config(slot_cfg, base_dir=data_dir)
    slot = replace(
        slot,
        slot_config_hash=f"bench-cfg-{file_digest}",
        slot_data_fingerprint=f"bench-fp-{file_digest}",
    )

    dataset_stats = {
        "total_sessions": int(len(cal.sessions())),
        "cutoffs_per_instrument": int(n_bars),
        "total_scheduled_grid_rows": int(n_bars * len(instruments)),
        "observed_bar_rows": int(len(bars_df)),
        "injected_gap_rows": int(gap_rows),
        "injected_nan_rows": int(nan_rows),
        "dataset_file": data_file.name,
        "dataset_bytes": int(dataset_bytes),
        "dataset_mb": round(dataset_bytes / (1024 ** 2), 4),
    }
    return slot, dataset_stats, instruments


class _IOCounter:
    """Thread-local style counter wrapper around bar_feed.read_signal_bars and source.read."""

    def __init__(self, slot: MarketSlot) -> None:
        self._slot = slot
        self._orig_read_signal_bars = slot.bar_feed.read_signal_bars
        self._orig_source_read = slot.bar_feed.source.read
        self.reset()

    def reset(self) -> None:
        self.read_signal_bars_calls = 0
        self.read_signal_bars_rows = 0
        self.read_signal_bars_bytes = 0
        self.source_read_calls = 0
        self.source_read_rows = 0

    def install(self) -> None:
        def counted_signal_bars(*args: Any, **kwargs: Any) -> pd.DataFrame:
            df = self._orig_read_signal_bars(*args, **kwargs)
            self.read_signal_bars_calls += 1
            self.read_signal_bars_rows += int(len(df))
            self.read_signal_bars_bytes += int(df.memory_usage(index=True, deep=False).sum())
            return df

        def counted_source_read(*args: Any, **kwargs: Any) -> pd.DataFrame:
            df = self._orig_source_read(*args, **kwargs)
            self.source_read_calls += 1
            self.source_read_rows += int(len(df))
            return df

        self._slot.bar_feed.read_signal_bars = counted_signal_bars  # type: ignore[method-assign]
        object.__setattr__(self._slot.bar_feed.source, "read", counted_source_read)

    def uninstall(self) -> None:
        self._slot.bar_feed.read_signal_bars = self._orig_read_signal_bars  # type: ignore[method-assign]
        if "read" in self._slot.bar_feed.source.__dict__:
            object.__delattr__(self._slot.bar_feed.source, "read")

    def snapshot(self) -> dict[str, int]:
        return {
            "read_signal_bars_calls": self.read_signal_bars_calls,
            "read_signal_bars_rows": self.read_signal_bars_rows,
            "read_signal_bars_bytes": self.read_signal_bars_bytes,
            "source_read_calls": self.source_read_calls,
            "source_read_rows": self.source_read_rows,
        }


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if platform.system() == "Darwin" else int(value * 1024)


def _current_rss_bytes() -> int:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024
    return _peak_rss_bytes()


def _machine_info() -> dict[str, Any]:
    cpu_model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    ram_total_bytes: int | None = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    ram_total_bytes = int(parts[1]) * 1024
                break
    elif hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names and "SC_PAGE_SIZE" in os.sysconf_names:
        try:
            ram_total_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            ram_total_bytes = None

    git_commit = None
    git_branch = None
    repo_root = Path(__file__).resolve().parents[2]
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        pass

    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "ram_total_bytes": ram_total_bytes,
        "ram_total_gib": round(ram_total_bytes / (1024 ** 3), 2) if ram_total_bytes else None,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "git_commit": git_commit,
        "git_branch": git_branch,
    }


def _cache_storage_stats(cache_dir: Path | None) -> dict[str, Any]:
    if cache_dir is None or not cache_dir.is_dir():
        return {
            "cache_file_count": 0,
            "cache_pkl_count": 0,
            "cache_slotmeta_count": 0,
            "cache_pkl_bytes": 0,
            "cache_slotmeta_bytes": 0,
            "cache_total_bytes": 0,
            "cache_total_mb": 0.0,
        }
    pkl_bytes = 0
    meta_bytes = 0
    pkl_count = 0
    meta_count = 0
    other_bytes = 0
    other_count = 0
    for entry in cache_dir.iterdir():
        if not entry.is_file():
            continue
        size = entry.stat().st_size
        if entry.name.endswith(".pkl"):
            pkl_count += 1
            pkl_bytes += size
        elif entry.name.endswith(".slotmeta"):
            meta_count += 1
            meta_bytes += size
        else:
            other_count += 1
            other_bytes += size
    total = pkl_bytes + meta_bytes + other_bytes
    return {
        "cache_file_count": pkl_count + meta_count + other_count,
        "cache_pkl_count": pkl_count,
        "cache_slotmeta_count": meta_count,
        "cache_pkl_bytes": pkl_bytes,
        "cache_slotmeta_bytes": meta_bytes,
        "cache_total_bytes": total,
        "cache_total_mb": round(total / (1024 ** 2), 4),
    }


def _compare_fast_vs_pit(
    fast_panel: pd.DataFrame,
    pit_df: pd.DataFrame,
    sampled_targets: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Verify exact numerical and NaN-mask alignment between fast path and pit_replay."""
    fast_long = bar_factor_frame(fast_panel)
    expected = sampled_targets[_BAR_KEYS].merge(
        pit_df[_BAR_KEYS + ["factor_value"]], on=_BAR_KEYS, how="left", validate="one_to_one"
    )
    actual = sampled_targets[_BAR_KEYS].merge(
        fast_long[_BAR_KEYS + ["factor_value"]], on=_BAR_KEYS, how="left", validate="one_to_one"
    )

    exp_vals = expected["factor_value"].to_numpy(dtype=float)
    act_vals = actual["factor_value"].to_numpy(dtype=float)
    exp_nan = np.isnan(exp_vals)
    act_nan = np.isnan(act_vals)
    nan_mismatch_count = int(np.sum(exp_nan != act_nan))

    both_finite = (~exp_nan) & (~act_nan) & np.isfinite(exp_vals) & np.isfinite(act_vals)
    if np.any(both_finite):
        abs_diffs = np.abs(act_vals[both_finite] - exp_vals[both_finite])
        rel_diffs = abs_diffs / np.maximum(np.abs(exp_vals[both_finite]), 1e-12)
        max_abs_diff = float(np.max(abs_diffs))
        max_rel_diff = float(np.max(rel_diffs))
    else:
        max_abs_diff = 0.0
        max_rel_diff = 0.0

    values_close = bool(np.allclose(act_vals, exp_vals, rtol=rtol, atol=atol, equal_nan=True))
    passed = values_close and nan_mismatch_count == 0
    return {
        "compared_points": int(len(sampled_targets)),
        "finite_compared_points": int(np.sum(both_finite)),
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "nan_mismatch_count": nan_mismatch_count,
        "passed": passed,
    }


def _select_sample_cutoffs(clock: pd.DataFrame, count: int) -> list[pd.Timestamp]:
    if count <= 0 or clock.empty:
        return []
    cutoffs = list(pd.DatetimeIndex(clock[_AVAILABLE].drop_duplicates().sort_values()))
    if len(cutoffs) <= count:
        return cutoffs
    indices = sorted({int(round(i)) for i in np.linspace(0, len(cutoffs) - 1, count)})
    return [cutoffs[idx] for idx in indices]


def _select_sample_specs(specs: list[_FactorSpec], count: int) -> list[_FactorSpec]:
    if count <= 0 or not specs:
        return []
    # Interleave vectorized and incremental families so sampled checks always cover both modes.
    by_family: dict[str, list[_FactorSpec]] = {}
    for spec in specs:
        by_family.setdefault(spec.family, []).append(spec)
    vec_fams = [fam for fam in by_family if not fam.endswith("_inc")]
    inc_fams = [fam for fam in by_family if fam.endswith("_inc")]
    family_names: list[str] = []
    for i in range(max(len(vec_fams), len(inc_fams))):
        if i < len(vec_fams):
            family_names.append(vec_fams[i])
        if i < len(inc_fams):
            family_names.append(inc_fams[i])

    selected: list[_FactorSpec] = []
    depth = 0
    while len(selected) < min(count, len(specs)):
        added = False
        for fam in family_names:
            bucket = by_family[fam]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) >= min(count, len(specs)):
                    break
        if not added:
            break
        depth += 1
    return selected


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "total": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return {
        "count": len(ordered),
        "total": round(sum(ordered), 6),
        "mean": round(statistics.mean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "p95": round(ordered[p95_idx], 6),
        "max": round(ordered[-1], 6),
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a structured Markdown summary from a benchmark result dictionary."""
    cfg = report["config"]
    ds = report["dataset_stats"]
    st = report["stage_timings_seconds"]
    fs = report["factor_summary"]
    mem = report["memory"]
    io = report["io"]
    stor = report["storage"]
    pit = report["pit_consistency"]
    tp = report["throughput"]
    mach = report["machine"]

    lines = [
        "# Minute-Bar Scale Benchmark Report",
        "",
        f"- **Generated (UTC)**: `{report['timestamp_utc']}`",
        f"- **Schema**: `{report['schema_version']}`",
        f"- **Git**: `{mach.get('git_branch')}` @ `{mach.get('git_commit')}`",
        f"- **Host**: `{mach.get('hostname')}` | `{mach.get('platform')}` | CPU `{mach.get('cpu_model')}` ({mach.get('cpu_count')} cores) | RAM `{mach.get('ram_total_gib')} GiB`",
        f"- **Runtime**: Python `{mach.get('python_version')}`, pandas `{mach.get('pandas_version')}`, numpy `{mach.get('numpy_version')}`",
        "",
        "## Workload & Dataset Scale",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Date Range | `{cfg['start_date']}` .. `{cfg['end_date']}` ({ds['total_sessions']} sessions) |",
        f"| Frequency / Calendar | `{cfg['frequency']}` / `{cfg['calendar_mode']}` (`{cfg['source_format']}`) |",
        f"| Instruments | {cfg['num_instruments']} |",
        f"| Cutoffs / Instrument | {ds['cutoffs_per_instrument']:,} |",
        f"| Scheduled Grid Rows | {ds['total_scheduled_grid_rows']:,} |",
        f"| Observed Bar Rows | {ds['observed_bar_rows']:,} (gaps: {ds['injected_gap_rows']:,}, NaNs: {ds['injected_nan_rows']:,}) |",
        f"| Total Factors | {cfg['num_factors']} (vectorized: {fs['vectorized_count']}, incremental: {fs['incremental_count']}, pit_replay fallback: {fs['pit_replay_fallback_count']}) |",
        f"| Dataset File Size | {ds['dataset_mb']:.4f} MiB ({ds['dataset_bytes']:,} bytes) |",
        "",
        "## Stage Timings & Throughput",
        "",
        "| Stage | Seconds | Notes |",
        "| --- | ---: | --- |",
        f"| Data & Slot Generation | {st['data_generation_seconds']:.4f} | Synthetic dataset + MarketSlot build |",
        f"| Cold Batch Compute ({cfg['num_factors']} factors) | {st['cold_batch_compute_seconds']:.4f} | Vectorized: {fs['vectorized_latency']['total']:.4f}s, Incremental: {fs['incremental_latency']['total']:.4f}s |",
        f"| Warm Cache-Hit Pass ({cfg['num_factors']} factors) | {st['warm_cache_hit_seconds']:.4f} | Hits: {fs['warm_cache_hits']}/{cfg['num_factors']} |",
        f"| Sampled `pit_replay` Check | {st['pit_replay_sample_seconds']:.4f} | {pit['factors_sampled']} factors x {pit['cutoffs_sampled']} cutoffs |",
        f"| Total Benchmark Wall Time | {st['total_benchmark_seconds']:.4f} | End-to-end |",
        "",
        f"- **Cold Throughput**: {tp['cold_factor_bar_rows_per_second']:,.1f} factor-bar-rows/s ({tp['cold_factors_per_second']:.2f} factors/s)",
        f"- **Warm Cache Speedup**: {tp['warm_vs_cold_speedup']:.2f}x",
        f"- **Extrapolated Full-Year `pit_replay` ({cfg['num_factors']} factors)**: ~{tp['extrapolated_full_pit_replay_seconds']:,.1f}s (approx. {tp['cold_vs_extrapolated_pit_speedup']:,.1f}x slower than cold fast path)",
        "",
        "## Memory, IO & Cache Storage",
        "",
        "| Resource Metric | Value |",
        "| --- | --- |",
        f"| Peak Process RSS | {mem['peak_rss_mb']:.2f} MiB ({mem['peak_rss_bytes']:,} bytes) |",
        f"| `tracemalloc` Peak Traced | {mem['tracemalloc_peak_mb']:.2f} MiB ({mem['tracemalloc_peak_bytes']:,} bytes) |",
        f"| Cold `read_signal_bars` Calls | {io['cold']['read_signal_bars_calls']:,} ({io['cold']['read_signal_bars_rows']:,} rows) |",
        f"| Cold `source.read` Calls | {io['cold']['source_read_calls']:,} ({io['cold']['source_read_rows']:,} rows) |",
        f"| Warm `read_signal_bars` Calls | {io['warm']['read_signal_bars_calls']:,} |",
        f"| Sampled `pit_replay` `read_signal_bars` Calls | {io['pit_sample']['read_signal_bars_calls']:,} ({io['pit_sample']['read_signal_bars_rows']:,} rows) |",
        f"| Cache Files / Total Size | {stor['cache_file_count']:,} files / {stor['cache_total_mb']:.2f} MiB (.pkl: {stor['cache_pkl_bytes'] / (1024 ** 2):.2f} MiB, .slotmeta: {stor['cache_slotmeta_bytes'] / (1024 ** 2):.2f} MiB) |",
        "",
        "## Sampled `pit_replay` Consistency Verification",
        "",
        f"- **Overall Status**: `{'PASSED' if pit['passed'] else 'FAILED'}` (rtol={pit['rtol']}, atol={pit['atol']})",
        f"- **Overall Max Abs Diff**: `{pit['overall_max_abs_diff']:.3e}` | **Max Rel Diff**: `{pit['overall_max_rel_diff']:.3e}` | **NaN Mismatches**: `{pit['total_nan_mismatch_count']}`",
        "",
        "| Factor | Mode | Sampled Cutoffs | Points | Max Abs Diff | Max Rel Diff | NaN Mismatches | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in pit["factors_checked"]:
        lines.append(
            f"| `{row['factor_name']}` | `{row['fast_mode']}` | {row['sample_cutoffs']} | "
            f"{row['compared_points']} | {row['max_abs_diff']:.3e} | {row['max_rel_diff']:.3e} | "
            f"{row['nan_mismatch_count']} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    """Execute the minute-bar scale benchmark and return structured metrics."""
    bench_t0 = time.perf_counter()
    work_root = (
        Path(cfg.work_dir).resolve()
        if cfg.work_dir is not None
        else (Path(__file__).resolve().parents[2] / "work" / "minute_bar_scale_benchmark").resolve()
    )
    run_dir = work_root / f"run_{os.getpid()}_{time.time_ns()}"
    data_dir = run_dir / "data"
    cache_dir = run_dir / "cache" if cfg.enable_cache else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    rss_before = _current_rss_bytes()
    specs = build_factor_specs(cfg.num_factors, cfg.incremental_count)
    sample_specs = _select_sample_specs(specs, cfg.pit_sample_factors)
    sample_names = {s.name for s in sample_specs}

    try:
        t0 = time.perf_counter()
        slot, dataset_stats, instruments = _build_synthetic_slot(cfg, data_dir)
        data_gen_sec = time.perf_counter() - t0
        rss_after_gen = _current_rss_bytes()

        io_counter = _IOCounter(slot)
        io_counter.install()

        try:
            if cfg.enable_tracemalloc:
                tracemalloc.start()

            # Phase 2: Cold batch execution across N factors
            io_counter.reset()
            cold_t0 = time.perf_counter()
            vec_times: list[float] = []
            inc_times: list[float] = []
            all_times: list[float] = []
            mode_counts = {"vectorized": 0, "incremental": 0, "pit_replay": 0}
            cold_cache_hits = 0
            sampled_panels: dict[str, tuple[pd.DataFrame, FactorBase, str]] = {}
            clock_ref: pd.DataFrame | None = None

            for spec in specs:
                factor = spec.cls(
                    None,
                    spec.name,
                    cfg.start_date,
                    cfg.end_date,
                    stock_pool=instruments,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    slot=slot,
                    data_frequency=cfg.frequency,
                    analysis_mode="time_series",
                    slot_execution_mode="auto",
                    **spec.params,
                )
                if clock_ref is None:
                    clock_ref = factor._slot_clock()
                ft0 = time.perf_counter()
                panel = factor.get_factor_values()
                dt = time.perf_counter() - ft0
                all_times.append(dt)
                resolved_mode = factor.last_slot_execution_mode or "unknown"
                if resolved_mode in mode_counts:
                    mode_counts[resolved_mode] += 1
                if resolved_mode == "vectorized":
                    vec_times.append(dt)
                elif resolved_mode == "incremental":
                    inc_times.append(dt)
                if bool((factor.last_performance_metrics or {}).get("slot_cache_hit")):
                    cold_cache_hits += 1

                if spec.name in sample_names:
                    sampled_panels[spec.name] = (panel, factor, resolved_mode)
                else:
                    del panel

            cold_sec = time.perf_counter() - cold_t0
            cold_io = io_counter.snapshot()
            rss_after_cold = _current_rss_bytes()
            storage_stats = _cache_storage_stats(cache_dir)

            # Phase 3: Warm cache-hit pass
            warm_sec = 0.0
            warm_hits = 0
            warm_io = {
                "read_signal_bars_calls": 0,
                "read_signal_bars_rows": 0,
                "read_signal_bars_bytes": 0,
                "source_read_calls": 0,
                "source_read_rows": 0,
            }
            if cfg.enable_cache and cfg.warm_pass:
                io_counter.reset()
                warm_t0 = time.perf_counter()
                for spec in specs:
                    warm_factor = spec.cls(
                        None,
                        spec.name,
                        cfg.start_date,
                        cfg.end_date,
                        stock_pool=instruments,
                        cache_dir=str(cache_dir),
                        slot=slot,
                        data_frequency=cfg.frequency,
                        analysis_mode="time_series",
                        slot_execution_mode="auto",
                        **spec.params,
                    )
                    warm_panel = warm_factor.get_factor_values()
                    if bool((warm_factor.last_performance_metrics or {}).get("slot_cache_hit")):
                        warm_hits += 1
                    del warm_panel
                warm_sec = time.perf_counter() - warm_t0
                warm_io = io_counter.snapshot()
            rss_after_warm = _current_rss_bytes()

            # Phase 4: Sampled pit_replay verification
            assert clock_ref is not None
            sample_cutoffs = _select_sample_cutoffs(clock_ref, cfg.pit_sample_cutoffs)
            sampled_targets = (
                clock_ref.loc[clock_ref[_AVAILABLE].isin(sample_cutoffs)].copy()
                if sample_cutoffs
                else clock_ref.iloc[:0].copy()
            )

            io_counter.reset()
            pit_t0 = time.perf_counter()
            checked_rows: list[dict[str, Any]] = []
            overall_max_abs = 0.0
            overall_max_rel = 0.0
            total_nan_mismatches = 0
            total_compared_points = 0
            all_passed = True

            if sample_cutoffs and sample_specs:
                for spec in sample_specs:
                    fast_panel, factor_obj, fast_mode = sampled_panels[spec.name]
                    pit_df = bar_bridge._pit_replay_slot_values(
                        factor_obj,
                        clock_ref,
                        sampled_targets,
                        force_recalculate=True,
                        remember=False,
                        calculate=True,
                    )
                    cmp_res = _compare_fast_vs_pit(
                        fast_panel,
                        pit_df,
                        sampled_targets,
                        rtol=cfg.rtol,
                        atol=cfg.atol,
                    )
                    overall_max_abs = max(overall_max_abs, cmp_res["max_abs_diff"])
                    overall_max_rel = max(overall_max_rel, cmp_res["max_rel_diff"])
                    total_nan_mismatches += cmp_res["nan_mismatch_count"]
                    total_compared_points += cmp_res["compared_points"]
                    all_passed = all_passed and cmp_res["passed"]
                    checked_rows.append(
                        {
                            "factor_name": spec.name,
                            "family": spec.family,
                            "fast_mode": fast_mode,
                            "sample_cutoffs": len(sample_cutoffs),
                            **cmp_res,
                        }
                    )
            pit_sample_sec = time.perf_counter() - pit_t0
            pit_io = io_counter.snapshot()
            rss_after_pit = _current_rss_bytes()

            if cfg.enable_tracemalloc:
                tm_current, tm_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            else:
                tm_current, tm_peak = 0, 0
        finally:
            if cfg.enable_tracemalloc and tracemalloc.is_tracing():
                tracemalloc.stop()
            io_counter.uninstall()

        total_sec = time.perf_counter() - bench_t0
        observed_rows = max(dataset_stats["observed_bar_rows"], 1)
        total_factor_bar_rows = observed_rows * cfg.num_factors

        # Estimate full-year pit_replay wall time from sampled cutoffs.
        n_sampled_evals = max(len(checked_rows) * len(sample_cutoffs), 1)
        sec_per_pit_cutoff = pit_sample_sec / n_sampled_evals if checked_rows and sample_cutoffs else 0.0
        extrapolated_full_pit = sec_per_pit_cutoff * dataset_stats["cutoffs_per_instrument"] * cfg.num_factors

        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "machine": _machine_info(),
            "config": {
                **asdict(cfg),
                "work_dir": str(run_dir),
                "json_output": str(cfg.json_output) if cfg.json_output else None,
                "md_output": str(cfg.md_output) if cfg.md_output else None,
            },
            "dataset_stats": dataset_stats,
            "stage_timings_seconds": {
                "data_generation_seconds": round(data_gen_sec, 6),
                "cold_batch_compute_seconds": round(cold_sec, 6),
                "warm_cache_hit_seconds": round(warm_sec, 6),
                "pit_replay_sample_seconds": round(pit_sample_sec, 6),
                "total_benchmark_seconds": round(total_sec, 6),
            },
            "factor_summary": {
                "total_factors": cfg.num_factors,
                "vectorized_count": mode_counts["vectorized"],
                "incremental_count": mode_counts["incremental"],
                "pit_replay_fallback_count": mode_counts["pit_replay"],
                "cold_cache_hits": cold_cache_hits,
                "warm_cache_hits": warm_hits,
                "all_latency": _latency_summary(all_times),
                "vectorized_latency": _latency_summary(vec_times),
                "incremental_latency": _latency_summary(inc_times),
            },
            "throughput": {
                "cold_factor_bar_rows_per_second": round(total_factor_bar_rows / max(cold_sec, 1e-9), 2),
                "cold_factors_per_second": round(cfg.num_factors / max(cold_sec, 1e-9), 4),
                "warm_vs_cold_speedup": round(cold_sec / max(warm_sec, 1e-9), 2) if warm_sec > 0 else 0.0,
                "extrapolated_full_pit_replay_seconds": round(extrapolated_full_pit, 2),
                "cold_vs_extrapolated_pit_speedup": (
                    round(extrapolated_full_pit / max(cold_sec, 1e-9), 2) if extrapolated_full_pit > 0 else 0.0
                ),
            },
            "memory": {
                "peak_rss_bytes": _peak_rss_bytes(),
                "peak_rss_mb": round(_peak_rss_bytes() / (1024 ** 2), 2),
                "tracemalloc_current_bytes": int(tm_current),
                "tracemalloc_peak_bytes": int(tm_peak),
                "tracemalloc_peak_mb": round(tm_peak / (1024 ** 2), 2),
                "rss_snapshots_mb": {
                    "before": round(rss_before / (1024 ** 2), 2),
                    "after_data_gen": round(rss_after_gen / (1024 ** 2), 2),
                    "after_cold_batch": round(rss_after_cold / (1024 ** 2), 2),
                    "after_warm_batch": round(rss_after_warm / (1024 ** 2), 2),
                    "after_pit_sample": round(rss_after_pit / (1024 ** 2), 2),
                },
            },
            "io": {
                "cold_read_signal_bars_calls": cold_io["read_signal_bars_calls"],
                "warm_read_signal_bars_calls": warm_io["read_signal_bars_calls"],
                "pit_sample_read_signal_bars_calls": pit_io["read_signal_bars_calls"],
                "cold": cold_io,
                "warm": warm_io,
                "pit_sample": pit_io,
            },
            "storage": storage_stats,
            "pit_consistency": {
                "passed": all_passed,
                "rtol": cfg.rtol,
                "atol": cfg.atol,
                "factors_sampled": len(checked_rows),
                "cutoffs_sampled": len(sample_cutoffs),
                "sample_cutoffs_utc": [pd.Timestamp(ts).isoformat() for ts in sample_cutoffs],
                "total_compared_points": total_compared_points,
                "overall_max_abs_diff": overall_max_abs,
                "overall_max_rel_diff": overall_max_rel,
                "total_nan_mismatch_count": total_nan_mismatches,
                "factors_checked": checked_rows,
            },
        }
        return json_compatible(report)
    finally:
        gc.collect()
        if cfg.cleanup_work_dir and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)


def _parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--frequency", default="1m")
    parser.add_argument("--calendar-mode", choices=_CALENDAR_MODES, default="trading_session")
    parser.add_argument("--source-format", choices=_SOURCE_FORMATS, default="parquet")
    parser.add_argument("--num-instruments", type=int, default=5)
    parser.add_argument("--num-factors", type=int, default=100)
    parser.add_argument("--incremental-count", type=int, default=10)
    parser.add_argument("--pit-sample-cutoffs", type=int, default=12)
    parser.add_argument("--pit-sample-factors", type=int, default=4)
    parser.add_argument("--no-gaps-and-nans", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-warm-pass", action="store_true")
    parser.add_argument("--no-tracemalloc", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
    args = parser.parse_args(argv)
    return BenchmarkConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        frequency=args.frequency,
        calendar_mode=args.calendar_mode,
        source_format=args.source_format,
        num_instruments=args.num_instruments,
        num_factors=args.num_factors,
        incremental_count=args.incremental_count,
        pit_sample_cutoffs=args.pit_sample_cutoffs,
        pit_sample_factors=args.pit_sample_factors,
        with_gaps_and_nans=not args.no_gaps_and_nans,
        enable_cache=not args.no_cache,
        warm_pass=not args.no_warm_pass,
        enable_tracemalloc=not args.no_tracemalloc,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
        work_dir=args.work_dir,
        cleanup_work_dir=not args.keep_work_dir,
        json_output=args.json_output,
        md_output=args.md_output,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    report = run_benchmark(cfg)
    if cfg.json_output is not None:
        cfg.json_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(cfg.json_output, report)
    if cfg.md_output is not None:
        cfg.md_output.parent.mkdir(parents=True, exist_ok=True)
        cfg.md_output.write_text(render_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pit_consistency"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
