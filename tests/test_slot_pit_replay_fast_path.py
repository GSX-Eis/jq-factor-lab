"""Tests for non-A-share Slot PIT Replay fast paths, pre-read caching, and causal safety."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import pytest

from futures_roll_samples import warm_market_calendars  # noqa: F401
from market_db_samples import crypto_profile
from slots import build_slot_from_config
from slots.cn_futures import build_cn_futures_slot
from test_slot_futures import _populate
from utils.bars.futures_roll import build_roll_schedule
from utils.factor_base import FactorBase
from utils.factor_core.slot_execution_mode import IncrementalEMASpec, IncrementalRollingSpec
from utils.market_db.identity import bind_market_db_fingerprint, slot_market_db_identity
from utils.market_db.storage import _create_profile_table
from utils.session_calendar import SessionCalendar
from utils.slots import MarketSlot, ProviderConfig, load_slot_config
from utils.time_axis import BarSchedule


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "declarative_crypto"


def _build_minute_declarative_slot(
    tmp_path: Path,
    *,
    days: tuple[str, str] = ("2026-01-05", "2026-01-06"),
    frequency: str = "5min",
    instruments: tuple[str, ...] = ("BTC-USD", "ETH-USD"),
    mutate_future_after: pd.Timestamp | None = None,
    with_gaps_and_nans: bool = True,
    slot_config_hash: str | None = "decl-fast-v1",
    slot_data_fingerprint: str | dict | None = "fp-v1",
) -> MarketSlot:
    config = load_slot_config(EXAMPLE / "slot.yaml")
    cal = SessionCalendar(
        "crypto_utc",
        days[0],
        days[1],
        custom_sessions=config.calendar.options["custom_sessions"],
    )
    sched = BarSchedule(cal, frequency, view="ETH", bucket_strategy="segment_anchored")
    rows = []
    for inst_idx, inst in enumerate(instruments):
        step = 0
        for day in cal.sessions():
            slots = sched.for_session(day.strftime("%Y-%m-%d"))
            for ts in slots["start_utc"]:
                step += 1
                # Inject deterministic gap on instrument 0 at step 7 and NaN at step 11.
                if with_gaps_and_nans and inst_idx == 0 and step == 7:
                    continue
                base = 100.0 + 10.0 * inst_idx + 0.5 * step + np.sin(step * 0.3)
                vol = 50.0 + (step % 13) * 3.0
                if mutate_future_after is not None and ts > mutate_future_after:
                    base += 5000.0
                    vol += 9000.0
                close_val = np.nan if (with_gaps_and_nans and inst_idx == 0 and step == 11) else base + 0.2
                rows.append(
                    {
                        "time": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "instrument_id": inst,
                        "open": base,
                        "high": base + 1.0,
                        "low": base - 1.0,
                        "close": close_val,
                        "volume": vol,
                    }
                )
    bars_df = pd.DataFrame(rows)
    data_dir = tmp_path / ("mutated" if mutate_future_after is not None else "base")
    data_dir.mkdir(parents=True, exist_ok=True)
    bars_df.to_csv(data_dir / "bars.csv", index=False)
    pd.read_csv(EXAMPLE / "funding.csv").to_csv(data_dir / "funding.csv", index=False)

    universe_opts = dict(config.universe.options)
    universe_opts["instruments"] = {
        inst: {
            "asset_class": "crypto",
            "currency": "USD",
            "lot_size": 1,
            "contract_multiplier": 1.0,
        }
        for inst in instruments
    }
    universe_opts["universes"] = {"spot": list(instruments)}
    bar_opts = {
        **config.bar_feed.options,
        "frequency": frequency,
        "timestamp_anchor": "bar_start",
        "column_mapping": {
            "time": "timestamp",
            "instrument_id": "instrument_id",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        "paths": ["bars.csv"],
    }
    bar_opts.pop("instrument_id", None)
    new_cfg = replace(
        config,
        calendar=replace(
            config.calendar,
            options={**config.calendar.options, "start": days[0], "end": days[1]},
        ),
        bar_feed=ProviderConfig("csv_files", bar_opts),
        universe=ProviderConfig("static", universe_opts),
    )
    slot = build_slot_from_config(new_cfg, base_dir=data_dir)
    if slot_config_hash is not None or slot_data_fingerprint is not None:
        slot = replace(
            slot,
            **({"slot_config_hash": slot_config_hash} if slot_config_hash is not None else {}),
            **({"slot_data_fingerprint": slot_data_fingerprint} if slot_data_fingerprint is not None else {}),
        )
    return slot


class _RollingMeanVecFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def __init__(self, *args, window: int = 4, min_periods: int | None = 2,
                 within_session: bool = True, warmup_sessions: int = 0,
                 warmup_bars: int = 0, aggregation: str = "mean", **kwargs):
        self.window = window
        self.min_periods = min_periods
        self.within_session = within_session
        self.warmup_sessions = warmup_sessions
        self.warmup_bars = warmup_bars
        self.aggregation = aggregation
        super().__init__(*args, **kwargs)

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(
            data,
            "close",
            self.window,
            min_periods=self.min_periods,
            within_session=self.within_session,
            warmup_sessions=self.warmup_sessions,
            warmup_bars=self.warmup_bars,
            aggregation=self.aggregation,
        )
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)

    def _get_cache_inputs(self) -> dict:
        return {
            "window": self.window,
            "min_periods": self.min_periods,
            "within_session": self.within_session,
            "warmup_sessions": self.warmup_sessions,
            "warmup_bars": self.warmup_bars,
            "aggregation": self.aggregation,
        }


class _RollingMeanBuiltinIncFactor(_RollingMeanVecFactor):
    CAUSAL_VECTORIZED = False

    def get_incremental_engine(self) -> IncrementalRollingSpec:
        return IncrementalRollingSpec(
            column="close",
            window=self.window,
            min_periods=self.min_periods,
            within_session=self.within_session,
            warmup_sessions=self.warmup_sessions,
            warmup_bars=self.warmup_bars,
            aggregation=self.aggregation,
        )

    def _get_cache_inputs(self) -> dict:
        return super()._get_cache_inputs()


class _EMAReferenceFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def __init__(self, *args, span: float = 5.0, min_periods: int = 2,
                 adjust: bool = True, within_session: bool = False,
                 warmup_bars: int = 1, **kwargs):
        self.span = span
        self.min_periods = min_periods
        self.adjust = adjust
        self.within_session = within_session
        self.warmup_bars = warmup_bars
        super().__init__(*args, **kwargs)

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        groups = ["instrument_id", "session_id"] if self.within_session else ["instrument_id"]
        ema = ordered.groupby(groups, sort=False)["close"].transform(
            lambda s: s.ewm(span=self.span, adjust=self.adjust, min_periods=self.min_periods, ignore_na=False).mean()
        )
        slots = ordered.groupby("instrument_id").cumcount()
        ema.loc[(slots < self.warmup_bars) | ~ordered.observed] = np.nan
        return ordered[["timestamp_utc", "instrument_id"]].assign(factor_value=ema.reindex(data.index))

    def _get_cache_inputs(self) -> dict:
        return {
            "span": self.span,
            "min_periods": self.min_periods,
            "adjust": self.adjust,
            "within_session": self.within_session,
            "warmup_bars": self.warmup_bars,
        }


class _EMABuiltinIncFactor(_EMAReferenceFactor):
    CAUSAL_VECTORIZED = False

    def get_incremental_engine(self) -> IncrementalEMASpec:
        return IncrementalEMASpec(
            column="close",
            span=self.span,
            min_periods=self.min_periods,
            adjust=self.adjust,
            within_session=self.within_session,
            warmup_bars=self.warmup_bars,
        )

    def _get_cache_inputs(self) -> dict:
        return super()._get_cache_inputs()


class _CustomIncrementalCumSumFactor(FactorBase):
    """Custom incremental protocol factor matching expanding close sum."""

    CAUSAL_VECTORIZED = False

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        vals = ordered.groupby("instrument_id", sort=False)["close"].transform(
            lambda s: s.expanding(min_periods=1).sum()
        )
        vals.loc[~ordered.observed] = np.nan
        return ordered[["timestamp_utc", "instrument_id"]].assign(factor_value=vals.reindex(data.index))

    def init_incremental_state(self) -> dict[str, float]:
        return {}

    def step_incremental(self, state: dict[str, float], bar_slice: pd.DataFrame):
        new_state = dict(state)
        out = []
        for row in bar_slice.sort_values(["instrument_id", "timestamp_utc"]).itertuples(index=False):
            if row.observed and not np.isnan(row.close):
                new_state[row.instrument_id] = new_state.get(row.instrument_id, 0.0) + float(row.close)
            val = new_state.get(row.instrument_id, np.nan) if row.observed else np.nan
            out.append((row.timestamp_utc, row.instrument_id, val))
        return new_state, pd.DataFrame(out, columns=["timestamp_utc", "instrument_id", "factor_value"])

    def _get_cache_inputs(self) -> dict:
        return {"kind": "cumsum"}


class _LookaheadCrossSectionFactor(FactorBase):
    """Non-causal factor using full-sample mean; must NOT enter single-pass vectorized."""

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        wide = data.pivot(index="timestamp_utc", columns="instrument_id", values="close")
        return wide - wide.mean(axis=0)

    def _get_cache_inputs(self) -> dict:
        return {"kind": "lookahead_mean"}


class _FuturesVecFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(data, "close", 2, min_periods=1, within_session=False)
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)

    def _get_cache_inputs(self) -> dict:
        return {"window": 2}


def _assert_panels_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual.isna(), expected.isna())


@pytest.mark.parametrize("aggregation", ["mean", "sum", "std", "min", "max", "median"])
@pytest.mark.parametrize("within_session", [True, False])
def test_vectorized_and_incremental_rolling_match_pit_replay_with_nans_and_gaps(
    tmp_path: Path, aggregation: str, within_session: bool
) -> None:
    slot = _build_minute_declarative_slot(
        tmp_path, days=("2026-01-05", "2026-01-06"), frequency="60min", with_gaps_and_nans=True
    )
    common = dict(
        start_date="2026-01-05",
        end_date="2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"],
        slot=slot,
        data_frequency="60min",
        analysis_mode="time_series",
        window=4,
        min_periods=2,
        within_session=within_session,
        warmup_sessions=1 if not within_session else 0,
        warmup_bars=2,
        aggregation=aggregation,
    )
    pit_factor = _RollingMeanVecFactor(None, "pit", **common, slot_execution_mode="pit_replay")
    vec_factor = _RollingMeanVecFactor(None, "vec", **common, slot_execution_mode="auto")
    inc_factor = _RollingMeanBuiltinIncFactor(None, "inc", **common, slot_execution_mode="auto")

    pit_panel = pit_factor.get_factor_values()
    vec_panel = vec_factor.get_factor_values()
    inc_panel = inc_factor.get_factor_values()

    assert pit_factor.last_slot_execution_mode == "pit_replay"
    assert vec_factor.last_slot_execution_mode == "vectorized"
    assert inc_factor.last_slot_execution_mode == "incremental"

    _assert_panels_equal(vec_panel, pit_panel)
    _assert_panels_equal(inc_panel, pit_panel)
    _assert_panels_equal(vec_factor.get_bar_factor_frame(), pit_factor.get_bar_factor_frame())
    _assert_panels_equal(inc_factor.get_bar_factor_frame(), pit_factor.get_bar_factor_frame())
    assert vec_panel.attrs["incomplete_bar_keys"] == pit_panel.attrs["incomplete_bar_keys"]
    assert inc_panel.attrs["incomplete_bar_keys"] == pit_panel.attrs["incomplete_bar_keys"]


@pytest.mark.parametrize("adjust", [True, False])
def test_incremental_ema_and_custom_protocol_match_pit_replay(tmp_path: Path, adjust: bool) -> None:
    slot = _build_minute_declarative_slot(
        tmp_path, days=("2026-01-05", "2026-01-06"), frequency="60min", with_gaps_and_nans=True
    )
    common = dict(
        start_date="2026-01-05",
        end_date="2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"],
        slot=slot,
        data_frequency="60min",
        analysis_mode="time_series",
        span=4.0,
        min_periods=2,
        adjust=adjust,
        warmup_bars=1,
    )
    pit_ema = _EMAReferenceFactor(None, "pit_ema", **common, slot_execution_mode="pit_replay")
    vec_ema = _EMAReferenceFactor(None, "vec_ema", **common, slot_execution_mode="auto")
    inc_ema = _EMABuiltinIncFactor(None, "inc_ema", **common, slot_execution_mode="auto")

    pit_panel = pit_ema.get_factor_values()
    vec_panel = vec_ema.get_factor_values()
    inc_panel = inc_ema.get_factor_values()
    assert vec_ema.last_slot_execution_mode == "vectorized"
    assert inc_ema.last_slot_execution_mode == "incremental"
    _assert_panels_equal(vec_panel, pit_panel)
    _assert_panels_equal(inc_panel, pit_panel)

    pit_cus = _CustomIncrementalCumSumFactor(
        None, "pit_cus", "2026-01-05", "2026-01-06", stock_pool=["BTC-USD", "ETH-USD"],
        slot=slot, data_frequency="60min", analysis_mode="time_series", slot_execution_mode="pit_replay",
    )
    inc_cus = _CustomIncrementalCumSumFactor(
        None, "inc_cus", "2026-01-05", "2026-01-06", stock_pool=["BTC-USD", "ETH-USD"],
        slot=slot, data_frequency="60min", analysis_mode="time_series", slot_execution_mode="auto",
    )
    assert inc_cus.get_factor_values() is not None
    assert inc_cus.last_slot_execution_mode == "incremental"
    _assert_panels_equal(inc_cus.get_factor_values(), pit_cus.get_factor_values())


@pytest.mark.parametrize("pool_mode", ["raw", "factor", "both"])
def test_dynamic_pool_modes_match_pit_replay_on_vectorized_and_incremental(
    tmp_path: Path, pool_mode: str
) -> None:
    slot = _build_minute_declarative_slot(
        tmp_path, days=("2026-01-05", "2026-01-06"), frequency="60min", with_gaps_and_nans=True
    )
    pools = {
        "2026-01-05": ["BTC-USD", "ETH-USD"],
        "2026-01-06": ["BTC-USD"],
    }
    common = dict(
        start_date="2026-01-05",
        end_date="2026-01-06",
        use_dynamic_pool=True,
        stock_pools_by_period=pools,
        dynamic_pool_mode=pool_mode,
        slot=slot,
        data_frequency="60min",
        analysis_mode="time_series",
        window=3,
        min_periods=1,
        within_session=False,
    )
    pit_f = _RollingMeanVecFactor(None, "pit_pool", **common, slot_execution_mode="pit_replay")
    vec_f = _RollingMeanVecFactor(None, "vec_pool", **common, slot_execution_mode="auto")
    inc_f = _RollingMeanBuiltinIncFactor(None, "inc_pool", **common, slot_execution_mode="auto")

    pit_df = pit_f.get_factor_values()
    vec_df = vec_f.get_factor_values()
    inc_df = inc_f.get_factor_values()
    assert vec_f.last_slot_execution_mode == "vectorized"
    assert inc_f.last_slot_execution_mode == "incremental"
    _assert_panels_equal(vec_df, pit_df)
    _assert_panels_equal(inc_df, pit_df)

    # Verify rebalance pool replay works identically under fast paths too.
    pit_rep = pit_f._get_factor_values_by_rebalance_pool(2, rebalance_only=False)
    vec_rep = vec_f._get_factor_values_by_rebalance_pool(2, rebalance_only=False)
    inc_rep = inc_f._get_factor_values_by_rebalance_pool(2, rebalance_only=False)
    _assert_panels_equal(vec_rep, pit_rep)
    _assert_panels_equal(inc_rep, pit_rep)


def test_zero_lookahead_future_bar_mutation_and_unsafe_route_fallback(tmp_path: Path, caplog) -> None:
    cutoff = pd.Timestamp("2026-01-05T12:00:00Z")
    base_slot = _build_minute_declarative_slot(
        tmp_path / "base", days=("2026-01-05", "2026-01-06"), frequency="60min"
    )
    mutated_slot = _build_minute_declarative_slot(
        tmp_path / "mut", days=("2026-01-05", "2026-01-06"), frequency="60min",
        mutate_future_after=cutoff,
    )

    for cls, expected_mode in [
        (_RollingMeanVecFactor, "vectorized"),
        (_RollingMeanBuiltinIncFactor, "incremental"),
        (_EMABuiltinIncFactor, "incremental"),
        (_CustomIncrementalCumSumFactor, "incremental"),
    ]:
        kwargs = dict(
            start_date="2026-01-05",
            end_date="2026-01-06",
            stock_pool=["BTC-USD", "ETH-USD"],
            data_frequency="60min",
            analysis_mode="time_series",
        )
        f_base = cls(None, "base", slot=base_slot, **kwargs)
        f_mut = cls(None, "mut", slot=mutated_slot, **kwargs)
        p_base = f_base.get_factor_values()
        p_mut = f_mut.get_factor_values()
        assert f_base.last_slot_execution_mode == expected_mode
        assert f_mut.last_slot_execution_mode == expected_mode
        # Historical prefix up to and including cutoff must be untouched by future price/volume changes.
        hist_idx = p_base.index <= cutoff
        _assert_panels_equal(p_mut.loc[hist_idx], p_base.loc[hist_idx])
        assert not np.allclose(
            p_mut.loc[~hist_idx].fillna(0.0).to_numpy(),
            p_base.loc[~hist_idx].fillna(0.0).to_numpy(),
        )

    # Non-causal un-annotated factor MUST fall back to pit_replay and remain causal.
    lk_base = _LookaheadCrossSectionFactor(
        None, "lk_base", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=base_slot,
        data_frequency="60min", analysis_mode="time_series",
    )
    lk_mut = _LookaheadCrossSectionFactor(
        None, "lk_mut", "2026-01-05", "2026-01-06",
        stock_pool=["BTC-USD", "ETH-USD"], slot=mutated_slot,
        data_frequency="60min", analysis_mode="time_series",
    )
    lk_p_base = lk_base.get_factor_values()
    lk_p_mut = lk_mut.get_factor_values()
    assert lk_base.last_slot_execution_mode == "pit_replay"
    assert lk_mut.last_slot_execution_mode == "pit_replay"
    hist_idx = lk_p_base.index <= cutoff
    _assert_panels_equal(lk_p_mut.loc[hist_idx], lk_p_base.loc[hist_idx])

    # Explicit CAUSAL_VECTORIZED = False or unverified None MUST block forced slot_execution_mode='vectorized'.
    class _ExplicitUnsafe(_LookaheadCrossSectionFactor):
        CAUSAL_VECTORIZED = False

    for cls, expected_reason in [
        (_LookaheadCrossSectionFactor, "fallback_pit_replay:unverified_factor_causality"),
        (_ExplicitUnsafe, "fallback_pit_replay:non_causal_factor"),
    ]:
        with caplog.at_level("WARNING"):
            forced = cls(
                None, "unsafe", "2026-01-05", "2026-01-06",
                stock_pool=["BTC-USD", "ETH-USD"], slot=base_slot,
                data_frequency="60min", analysis_mode="time_series",
                slot_execution_mode="vectorized",
            )
            _assert_panels_equal(forced.get_factor_values(), lk_p_base)
        assert forced.last_slot_execution_mode == "pit_replay"
        assert forced.last_performance_metrics["slot_execution_reason"] == expected_reason
        assert "Slot execution mode vectorized blocked" in caplog.text
        caplog.clear()

        forced_mut = cls(
            None, "unsafe_mut", "2026-01-05", "2026-01-06",
            stock_pool=["BTC-USD", "ETH-USD"], slot=mutated_slot,
            data_frequency="60min", analysis_mode="time_series",
            slot_execution_mode="vectorized",
        )
        _assert_panels_equal(forced_mut.get_factor_values().loc[hist_idx], lk_p_base.loc[hist_idx])
        assert forced_mut.last_slot_execution_mode == "pit_replay"


@pytest.mark.usefixtures("warm_market_calendars")
def test_futures_dynamic_adjustment_falls_back_to_pit_replay_while_none_uses_fast_path() -> None:
    start, end = "2026-01-05", "2026-01-09"
    with duckdb.connect(config={"threads": 1}) as conn:
        _populate(conn, root="CU", exchange="SHFE", start=start, end=end, expiries=["2026-02", "2026-03"])
        build_roll_schedule(conn, product_roots=["CU"], start="2026-01-06", end=end)

        # Dynamic roll adjustment -> MUST fall back to pit_replay even when factor declares vectorized/forced.
        for adj in ("ratio_back", "panama_canal"):
            slot_adj = build_cn_futures_slot(conn, product_root="CU", start=start, end=end, adjustment=adj)
            f_auto = _FuturesVecFactor(
                None, f"fut_{adj}", "2026-01-06", end, stock_pool=["CU"],
                slot=slot_adj, data_frequency="1d", analysis_mode="time_series",
                slot_execution_mode="vectorized",
            )
            f_pit = _FuturesVecFactor(
                None, f"fut_{adj}_pit", "2026-01-06", end, stock_pool=["CU"],
                slot=slot_adj, data_frequency="1d", analysis_mode="time_series",
                slot_execution_mode="pit_replay",
            )
            _assert_panels_equal(f_auto.get_factor_values(), f_pit.get_factor_values())
            assert f_auto.last_slot_execution_mode == "pit_replay"

        # Unadjusted futures slot -> safe for vectorized/incremental fast path & matches pit_replay.
        slot_none = build_cn_futures_slot(conn, product_root="CU", start=start, end=end, adjustment="none")
        f_none_vec = _FuturesVecFactor(
            None, "fut_none_vec", "2026-01-06", end, stock_pool=["CU"],
            slot=slot_none, data_frequency="1d", analysis_mode="time_series",
        )
        f_none_pit = _FuturesVecFactor(
            None, "fut_none_pit", "2026-01-06", end, stock_pool=["CU"],
            slot=slot_none, data_frequency="1d", analysis_mode="time_series",
            slot_execution_mode="pit_replay",
        )
        _assert_panels_equal(f_none_vec.get_factor_values(), f_none_pit.get_factor_values())
        assert f_none_vec.last_slot_execution_mode == "vectorized"


def test_pre_read_cache_and_io_counts_eliminate_redundant_reads(tmp_path: Path, monkeypatch) -> None:
    slot = _build_minute_declarative_slot(
        tmp_path, days=("2026-01-05", "2026-01-05"), frequency="30min",
        instruments=("BTC-USD",), with_gaps_and_nans=False,
    )
    read_calls = 0
    orig_read = slot.bar_feed.read_signal_bars

    def counted_read(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return orig_read(*args, **kwargs)

    monkeypatch.setattr(slot.bar_feed, "read_signal_bars", counted_read)
    cache_dir = str(tmp_path / "cache")

    # 1. PIT Replay does O(T) reads (48 bars for 1 day of 30min crypto = 48 cutoffs).
    pit_f = _RollingMeanVecFactor(
        None, "io_probe", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
        slot=slot, data_frequency="30min", analysis_mode="time_series",
        window=4, slot_execution_mode="pit_replay",
    )
    pit_panel = pit_f.get_factor_values()
    assert read_calls == 48

    # 2. Vectorized fast path drops IO from O(T) to O(1), and 2nd run hits pre-read cache with 0 IO.
    read_calls = 0
    vec_f = _RollingMeanVecFactor(
        None, "io_probe", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
        slot=slot, data_frequency="30min", analysis_mode="time_series",
        window=4, cache_dir=cache_dir, slot_execution_mode="auto",
    )
    vec_panel = vec_f.get_factor_values()
    assert read_calls == 1
    assert vec_f.last_slot_execution_mode == "vectorized"
    assert vec_f.last_performance_metrics["slot_cache_hit"] is False
    _assert_panels_equal(vec_panel, pit_panel)

    # Fresh instance hitting pre-read cache: 0 bar reads and 0 calculate_factor calls!
    read_calls = 0
    vec_cached = _RollingMeanVecFactor(
        None, "io_probe", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
        slot=slot, data_frequency="30min", analysis_mode="time_series",
        window=4, cache_dir=cache_dir, slot_execution_mode="auto",
    )
    monkeypatch.setattr(vec_cached, "calculate_factor", lambda _: pytest.fail("cache should skip compute"))
    cached_bar_frame = vec_cached.get_bar_factor_frame()
    assert read_calls == 0
    assert vec_cached.last_performance_metrics["slot_cache_hit"] is True
    _assert_panels_equal(cached_bar_frame, vec_f.get_bar_factor_frame())

    # 3. Incremental fast path also does O(1) read on miss and 0 reads on cache hit.
    read_calls = 0
    inc_f = _RollingMeanBuiltinIncFactor(
        None, "io_probe_inc", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
        slot=slot, data_frequency="30min", analysis_mode="time_series",
        window=4, cache_dir=cache_dir, slot_execution_mode="auto",
    )
    inc_panel = inc_f.get_factor_values()
    assert read_calls == 1
    assert inc_f.last_slot_execution_mode == "incremental"
    _assert_panels_equal(inc_panel, pit_panel)

    read_calls = 0
    inc_cached = _RollingMeanBuiltinIncFactor(
        None, "io_probe_inc", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
        slot=slot, data_frequency="30min", analysis_mode="time_series",
        window=4, cache_dir=cache_dir, slot_execution_mode="auto",
    )
    _assert_panels_equal(inc_cached.get_factor_values(), pit_panel)
    assert read_calls == 0
    assert inc_cached.last_performance_metrics["slot_cache_hit"] is True

    # 4. Invalidation on fingerprint change & force_recalculate=True.
    read_calls = 0
    _assert_panels_equal(inc_cached.get_factor_values(force_recalculate=True), pit_panel)
    assert read_calls == 1
    assert inc_cached.last_performance_metrics["slot_cache_hit"] is False

    read_calls = 0
    slot_v2 = replace(slot, slot_data_fingerprint="fp-v2")
    inc_v2 = _RollingMeanBuiltinIncFactor(
        None, "io_probe_inc", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
        slot=slot_v2, data_frequency="30min", analysis_mode="time_series",
        window=4, cache_dir=cache_dir, slot_execution_mode="auto",
    )
    _assert_panels_equal(inc_v2.get_factor_values(), pit_panel)
    assert read_calls == 1
    assert inc_v2.last_performance_metrics["slot_cache_hit"] is False

    # 5. Unfingerprinted mutable slot must never serve stale pre-read cache across disk edits.
    unfp_slot = _build_minute_declarative_slot(
        tmp_path / "unfp", days=("2026-01-05", "2026-01-05"), frequency="60min",
        instruments=("BTC-USD",), with_gaps_and_nans=False, slot_data_fingerprint=None,
    )
    unfp_cache = str(tmp_path / "unfp_cache")
    f_unfp1 = _RollingMeanVecFactor(
        None, "unfp", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
        slot=unfp_slot, data_frequency="60min", analysis_mode="time_series",
        window=4, cache_dir=unfp_cache, slot_execution_mode="auto",
    )
    p_unfp1 = f_unfp1.get_factor_values()
    csv_path = tmp_path / "unfp" / "base" / "bars.csv"
    edited = pd.read_csv(csv_path)
    edited["close"] = edited["close"] + 100.0
    edited.to_csv(csv_path, index=False)
    f_unfp2 = _RollingMeanVecFactor(
        None, "unfp", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
        slot=unfp_slot, data_frequency="60min", analysis_mode="time_series",
        window=4, cache_dir=unfp_cache, slot_execution_mode="auto",
    )
    p_unfp2 = f_unfp2.get_factor_values()
    assert f_unfp2.last_performance_metrics["slot_cache_hit"] is False
    assert np.allclose((p_unfp2 - p_unfp1).dropna().to_numpy(), 100.0)

    with pytest.raises(ValueError, match="slot_execution_mode"):
        _RollingMeanVecFactor(
            None, "bad_mode", "2026-01-05", "2026-01-05", stock_pool=["BTC-USD"],
            slot=slot, data_frequency="60min", analysis_mode="time_series",
            slot_execution_mode="unsafe",
        )


def test_duckdb_table_market_db_slot_invalidates_after_bar_update_for_all_modes(tmp_path: Path, monkeypatch) -> None:
    """Market DB identity alone is not a content fingerprint; SQL bar edits must never return stale caches."""
    config = load_slot_config(EXAMPLE / "slot.yaml")
    db_path = tmp_path / "market.duckdb"
    raw = pd.DataFrame(
        [
            {"symbol": "BTC-USD", "time": "2026-01-03T01:00:00Z", "open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 12.0},
            {"symbol": "BTC-USD", "time": "2026-01-03T02:00:00Z", "open": 101.0, "high": 105.0, "low": 100.0, "close": 101.0, "volume": 13.0},
            {"symbol": "BTC-USD", "time": "2026-01-03T03:00:00Z", "open": 102.0, "high": 106.0, "low": 101.0, "close": 102.0, "volume": 14.0},
        ]
    )
    with duckdb.connect(str(db_path)) as conn:
        _create_profile_table(conn, crypto_profile())
        conn.from_df(raw.assign(time=pd.to_datetime(raw["time"], utc=True))).create("bars")

    bar_opts = {k: v for k, v in config.bar_feed.options.items() if k != "paths"}
    bar_opts.update(path=db_path.name, table="main.bars")
    decl_cfg = replace(
        config,
        bar_feed=ProviderConfig("duckdb_table", bar_opts),
        aux_features=ProviderConfig("none", {}),
    )
    slot = build_slot_from_config(decl_cfg, base_dir=tmp_path)
    assert slot.slot_config_hash is not None
    assert slot.slot_data_fingerprint == slot_market_db_identity(slot)

    cache_dir = str(tmp_path / "cache")
    common = dict(
        start_date="2026-01-03",
        end_date="2026-01-03",
        stock_pool=["BTC-USD"],
        slot=slot,
        data_frequency="1h",
        analysis_mode="time_series",
        window=2,
        min_periods=1,
        within_session=True,
    )

    pit_before = _RollingMeanVecFactor(None, "db_pit", **common, cache_dir=cache_dir, slot_execution_mode="pit_replay").get_factor_values()
    vec_before = _RollingMeanVecFactor(None, "db_vec", **common, cache_dir=cache_dir, slot_execution_mode="vectorized").get_factor_values()
    _assert_panels_equal(vec_before, pit_before)

    # Historical bar revision directly in DuckDB.
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("UPDATE main.bars SET close = close + 1000.0")

    pit_uncached = _RollingMeanVecFactor(None, "db_pit", **common, cache_dir=None, slot_execution_mode="pit_replay").get_factor_values()
    pit_after = _RollingMeanVecFactor(None, "db_pit", **common, cache_dir=cache_dir, slot_execution_mode="pit_replay")
    vec_after = _RollingMeanVecFactor(None, "db_vec", **common, cache_dir=cache_dir, slot_execution_mode="vectorized")
    auto_after = _RollingMeanVecFactor(None, "db_vec", **common, cache_dir=cache_dir, slot_execution_mode="auto")

    pit_after_df = pit_after.get_factor_values()
    vec_after_df = vec_after.get_factor_values()
    auto_after_df = auto_after.get_factor_values()

    assert pit_after.last_performance_metrics["slot_cache_hit"] is False
    assert vec_after.last_performance_metrics["slot_cache_hit"] is False
    assert auto_after.last_performance_metrics["slot_cache_hit"] is False
    _assert_panels_equal(pit_after_df, pit_uncached)
    _assert_panels_equal(vec_after_df, pit_uncached)
    _assert_panels_equal(auto_after_df, pit_uncached)
    assert np.allclose((pit_after_df - pit_before).dropna().to_numpy(), 1000.0)

    # Explicit content-level fingerprint alongside market_db_identity enables pre-read cache with 0 IO.
    pinned_slot = replace(
        slot,
        slot_data_fingerprint=bind_market_db_fingerprint({"snapshot": "s1"}, slot_market_db_identity(slot)),
    )
    pinned_cold = _RollingMeanVecFactor(
        None, "db_pinned", **{**common, "slot": pinned_slot}, cache_dir=cache_dir, slot_execution_mode="vectorized"
    )
    pinned_df = pinned_cold.get_factor_values()
    assert pinned_cold.last_performance_metrics["slot_cache_hit"] is False

    read_calls = 0
    orig_read = pinned_slot.bar_feed.read_signal_bars

    def counted(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return orig_read(*args, **kwargs)

    monkeypatch.setattr(pinned_slot.bar_feed, "read_signal_bars", counted)
    pinned_warm = _RollingMeanVecFactor(
        None, "db_pinned", **{**common, "slot": pinned_slot}, cache_dir=cache_dir, slot_execution_mode="vectorized"
    )
    _assert_panels_equal(pinned_warm.get_factor_values(), pinned_df)
    assert read_calls == 0
    assert pinned_warm.last_performance_metrics["slot_cache_hit"] is True


def test_non_bar_grid_panel_vectorized_fast_path(tmp_path: Path, monkeypatch) -> None:
    slot = _build_minute_declarative_slot(
        tmp_path, days=("2026-01-05", "2026-01-06"), frequency="60min",
        instruments=("BTC-USD",), with_gaps_and_nans=False,
    )

    class _PanelTurnover(FactorBase):
        CAUSAL_VECTORIZED = True

        def _get_raw_data_for_factor(self):
            return self._fetch_raw_data(["close", "volume"], table_name="price_data")

        def calculate_factor(self, data):
            return data.assign(val=data.close * data.volume).pivot(
                index="date", columns="code", values="val"
            )

        def _get_cache_inputs(self):
            return {"kind": "panel_turnover"}

    reads = 0
    orig_read = slot.bar_feed.read_signal_bars

    def counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return orig_read(*args, **kwargs)

    monkeypatch.setattr(slot.bar_feed, "read_signal_bars", counted)
    pit_f = _PanelTurnover(
        None, "pt", "2026-01-05", "2026-01-06", stock_pool=["BTC-USD"],
        slot=slot, data_frequency="60min", analysis_mode=None, slot_execution_mode="pit_replay",
    )
    pit_df = pit_f.get_factor_values()
    assert reads == 48

    reads = 0
    vec_f = _PanelTurnover(
        None, "pt", "2026-01-05", "2026-01-06", stock_pool=["BTC-USD"],
        slot=slot, data_frequency="60min", analysis_mode=None,
        cache_dir=str(tmp_path / "cache"), slot_execution_mode="auto",
    )
    vec_df = vec_f.get_factor_values()
    assert reads == 1
    assert vec_f.last_slot_execution_mode == "vectorized"
    _assert_panels_equal(vec_df, pit_df)

    reads = 0
    _assert_panels_equal(vec_f.get_factor_values(), pit_df)
    assert reads == 0


@pytest.mark.slow
def test_minute_bar_performance_and_io_benchmark(tmp_path: Path, monkeypatch) -> None:
    # 15min bars over 3 full 24x7 days x 2 instruments = 288 cutoffs, 576 bar-rows.
    slot = _build_minute_declarative_slot(
        tmp_path, days=("2026-01-05", "2026-01-07"), frequency="15min",
        instruments=("BTC-USD", "ETH-USD"), with_gaps_and_nans=False,
    )
    io_count = 0
    orig_read = slot.bar_feed.read_signal_bars

    def counted_read(*args, **kwargs):
        nonlocal io_count
        io_count += 1
        return orig_read(*args, **kwargs)

    monkeypatch.setattr(slot.bar_feed, "read_signal_bars", counted_read)
    cache_dir = str(tmp_path / "bench_cache")
    common = dict(
        start_date="2026-01-05",
        end_date="2026-01-07",
        stock_pool=["BTC-USD", "ETH-USD"],
        slot=slot,
        data_frequency="15min",
        analysis_mode="time_series",
        window=8,
        min_periods=2,
        within_session=False,
    )

    # Benchmark pit_replay
    io_count = 0
    t0 = time.perf_counter()
    pit_panel = _RollingMeanVecFactor(
        None, "bench", **common, slot_execution_mode="pit_replay"
    ).get_factor_values()
    pit_sec = time.perf_counter() - t0
    pit_io = io_count

    # Benchmark vectorized
    io_count = 0
    t0 = time.perf_counter()
    vec_panel = _RollingMeanVecFactor(
        None, "bench", **common, cache_dir=cache_dir, slot_execution_mode="auto"
    ).get_factor_values()
    vec_sec = time.perf_counter() - t0
    vec_io = io_count

    # Benchmark incremental
    io_count = 0
    t0 = time.perf_counter()
    inc_panel = _RollingMeanBuiltinIncFactor(
        None, "bench_inc", **common, cache_dir=cache_dir, slot_execution_mode="auto"
    ).get_factor_values()
    inc_sec = time.perf_counter() - t0
    inc_io = io_count

    # Benchmark pre-read cache hit
    io_count = 0
    t0 = time.perf_counter()
    cached_panel = _RollingMeanBuiltinIncFactor(
        None, "bench_inc", **common, cache_dir=cache_dir, slot_execution_mode="auto"
    ).get_factor_values()
    cache_sec = time.perf_counter() - t0
    cache_io = io_count

    _assert_panels_equal(vec_panel, pit_panel)
    _assert_panels_equal(inc_panel, pit_panel)
    _assert_panels_equal(cached_panel, pit_panel)

    assert pit_io == 288 * 2
    assert vec_io == 2
    assert inc_io == 2
    assert cache_io == 0
    assert vec_sec < pit_sec
    assert inc_sec < pit_sec

    summary = {
        "cutoffs": 288,
        "instruments": 2,
        "pit_replay": {"seconds": round(pit_sec, 4), "io_calls": pit_io},
        "vectorized": {"seconds": round(vec_sec, 4), "io_calls": vec_io, "speedup": round(pit_sec / max(vec_sec, 1e-6), 2)},
        "incremental": {"seconds": round(inc_sec, 4), "io_calls": inc_io, "speedup": round(pit_sec / max(inc_sec, 1e-6), 2)},
        "cache_hit": {"seconds": round(cache_sec, 4), "io_calls": cache_io},
    }
    print("SLOT_FAST_PATH_BENCHMARK=" + json.dumps(summary, sort_keys=True))
