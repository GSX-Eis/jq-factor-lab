"""Alignment and fail-fast tests for ExecutionScheduler across evaluation, synthesis, and backtest."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.evaluation import UnifiedReturnLabeler
from backtest_and_analysis.time_series.contracts import ContractValidationError
from factor_synthesis import load_strategy_bundle
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from run_factor_synthesis import FactorSynthesizer
from slots.declarative import build_declarative_slot
from tools.agent_harness import cli as harness_cli
from utils.execution_schedule import (
    ExecutionScheduler,
    validate_effective_execution_profile,
)
from utils.market_db.identity import bind_market_db_fingerprint
from utils.market_factor import (
    MarketFactorValueStore,
    build_identity_record,
    compute_factor_identity_hash,
)
from utils.slots import ExecutionProfile, SlotConfig


def _make_daily_declarative_slot(
    tmp_path: Path,
    *,
    slot_id: str,
    min_entry_offset_bars: int = 1,
    exit_adjustment: int = 0,
    entry_price_field: str = "open",
    exit_price_field: str = "close",
    boundary_policy: str = "cross_session",
) -> tuple[Path, object, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2026-01-05", "2026-01-09", freq="B")
    symbols = ["A", "B"]
    # Prices chosen so bar-by-bar returns clearly distinguish T-1 vs T-2 selection.
    # A grows +10% on day 1, +2% on day 2, +1% on day 3, +1% on day 4.
    # B stays flat at 100 on all days.
    a_closes = [100.0, 110.0, 112.2, 113.322, 114.45522]
    rows = []
    for i, day in enumerate(dates):
        rows.append(
            {
                "symbol": "A",
                "time": f"{day.strftime('%Y-%m-%d')}T16:00:00Z",
                "open": 100.0 if i == 0 else a_closes[i - 1],
                "high": a_closes[i] + 1.0,
                "low": 99.0,
                "close": a_closes[i],
                "volume": 1000.0,
            }
        )
        rows.append(
            {
                "symbol": "B",
                "time": f"{day.strftime('%Y-%m-%d')}T16:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        )
    bars_csv = tmp_path / f"{slot_id}_bars.csv"
    pd.DataFrame(rows).to_csv(bars_csv, index=False)

    slot_cfg = {
        "schema_version": 1,
        "slot_id": slot_id,
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": f"{slot_id}_cal",
                "start": "2026-01-05",
                "end": "2026-01-09",
                "custom_sessions": {
                    "timezone": "UTC",
                    "weekdays": [0, 1, 2, 3, 4],
                    "segments": [{"open": "00:00", "close": "16:00"}],
                },
            },
        },
        "bar_feed": {
            "provider": "csv_files",
            "options": {
                "paths": [bars_csv.name],
                "frequency": "1d",
                "timestamp_anchor": "bar_end",
                "source_timezone": "UTC",
                "volume_mode": "interval",
                "data_version": "v1",
                "column_mapping": {
                    "symbol": "instrument_id",
                    "time": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
            },
        },
        "universe": {
            "provider": "static",
            "options": {
                "instruments": {s: {"asset_class": "equity", "currency": "USD"} for s in symbols},
                "universes": {"all": symbols},
            },
        },
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {
            "entry_price_field": entry_price_field,
            "exit_price_field": exit_price_field,
            "min_entry_offset_bars": min_entry_offset_bars,
            "exit_adjustment": exit_adjustment,
            "return_unit": "decimal",
            "gap_rule": "endpoints_only",
            "boundary_policy": boundary_policy,
            "allow_short": False,
            "commission_bps": 0.0,
            "sell_tax_bps": 0.0,
            "slippage_bps": 0.0,
            "benchmark_instrument": None,
        },
    }
    slot_path = tmp_path / f"{slot_id}_slot.json"
    slot_path.write_text(json.dumps(slot_cfg), encoding="utf-8")
    slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
    grid = slot.calendar.bar_grid("1d")
    from utils.time_axis import Frequency

    bars = slot.bar_feed.read_execution_bars(
        symbols,
        Frequency.parse("1d"),
        start_utc=grid["timestamp_utc"].min(),
        end_utc=grid["timestamp_utc"].max(),
    )
    close_df = bars.pivot(index="timestamp_utc", columns="instrument_id", values="close").sort_index()
    open_df = bars.pivot(index="timestamp_utc", columns="instrument_id", values="open").sort_index()
    return slot_path, slot, close_df, open_df, pd.DatetimeIndex(grid["timestamp_utc"])


def _save_scheme_b_factor(
    store_root: Path,
    *,
    slot: object,
    factor_id: str,
    frame: pd.DataFrame,
) -> str:
    digest, payload = compute_factor_identity_hash(
        factor_id=factor_id,
        factor_class="tests.SchedulerTestFactor",
        source_sha256="a" * 64,
        params={"seed": 1},
        frequency="1d",
        session_view="ETH",
        analysis_mode="cross_sectional",
        bucket_strategy="session",
        calendar_hash=slot.calendar.calendar_hash,
        slot_id=slot.slot_id,
        slot_config_hash=slot.slot_config_hash,
        adjustment_policy="none",
        slot_data_fingerprint=bind_market_db_fingerprint({"fixture": "scheduler"}, {}),
        code_version="scheduler-test-v1",
    )
    record = build_identity_record(
        factor_identity_hash=digest,
        payload=payload,
        code_version_source="test",
    )
    MarketFactorValueStore(store_root).append_factor_values(
        frame.assign(
            factor_id=factor_id,
            factor_identity_hash=digest,
            frequency="1d",
            session_view="ETH",
            source_build_id="build-1",
            calendar_hash=slot.calendar.calendar_hash,
            code_version="scheduler-test-v1",
            created_at=pd.Timestamp("2026-02-01T00:00:00Z"),
            batch_id=f"batch-{factor_id}",
        ),
        identity_record=record,
        calendar=slot.calendar,
    )
    return digest


def test_execution_scheduler_contract_and_fail_fast_validation() -> None:
    profile = ExecutionProfile("open", "close", 2, 0, "decimal", "endpoints_only", "cross_session")
    sched = ExecutionScheduler(profile)
    assert sched.resolved_entry_offset_bars == 2
    assert sched.entry_bar_index(3) == 5
    assert sched.exit_bar_index(5, 3) == 8
    assert sched.trade_bar_indices(3, 3) == (5, 8)
    assert sched.lookahead_bars(3) == 5
    assert sched.signal_bar_index(1) is None
    assert sched.signal_bar_index(2) == 0
    assert sched.signal_bar_index(4) == 2

    explicit = ExecutionScheduler(profile, entry_offset_bars=3)
    assert explicit.resolved_entry_offset_bars == 3
    assert explicit.trade_bar_indices(1, 2) == (4, 6)
    assert explicit.signal_timestamp_map(["d0", "d1", "d2", "d3", "d4"]) == {"d3": "d0", "d4": "d1"}

    for bad_offset in (1, 0, -1, True, 1.5):
        with pytest.raises(ValueError, match="entry_offset_bars"):
            ExecutionScheduler(profile, entry_offset_bars=bad_offset)  # type: ignore[arg-type]

    # Backtest validation rejects unsupported combinations explicitly.
    with pytest.raises(ValueError, match="same_session"):
        ExecutionScheduler.for_backtest(replace(profile, boundary_policy="same_session"))
    with pytest.raises(ValueError, match="exit_adjustment"):
        ExecutionScheduler.for_backtest(replace(profile, exit_adjustment=1))
    with pytest.raises(ValueError, match="exit_adjustment=-1"):
        ExecutionScheduler.for_backtest(
            ExecutionProfile("close", "close", 1, -1, "decimal", "endpoints_only", "cross_session")
        )
    with pytest.raises(ValueError, match="entry_offset_bars=0"):
        ExecutionScheduler.for_backtest(
            ExecutionProfile("close", "close", 0, 0, "decimal", "endpoints_only", "cross_session"),
            entry_offset_bars=0,
        )
    with pytest.raises(ValueError, match="entry_price_field"):
        ExecutionScheduler.for_backtest(profile, effective_entry_price_field="close")

    # OPEN_TO_CLOSE (exit_adjustment=-1 with open->close) is explicitly supported.
    otc_sched = ExecutionScheduler.for_backtest(
        ExecutionProfile("open", "close", 1, -1, "percent", "require_complete_bars", "cross_session")
    )
    assert otc_sched.resolved_entry_offset_bars == 1


@pytest.mark.parametrize("offset", [1, 2])
def test_labeler_runner_and_strategy_bundle_align_for_entry_offsets(tmp_path: Path, offset: int) -> None:
    slot_path, slot, close_df, open_df, grid_ts = _make_daily_declarative_slot(
        tmp_path,
        slot_id=f"align_offset_{offset}",
        min_entry_offset_bars=offset,
        exit_adjustment=0,
        entry_price_field="open",
        exit_price_field="close",
        boundary_policy="cross_session",
    )

    # Signal flips after day 0: day 0 prefers A (+10 > -10), days 1..4 prefer B (+10 > -10).
    # For offset=1: bar 0 is skipped; bar 1 enters A (using day 0 signal); bar 2 enters B (using day 1 signal).
    # For offset=2: bars 0 & 1 are skipped (NO fallback to T-1!); bar 2 enters A (using day 0 signal); bar 3 enters B.
    scores = pd.DataFrame(
        {
            "A": [10.0, -10.0, -10.0, -10.0, -10.0],
            "B": [-10.0, 10.0, 10.0, 10.0, 10.0],
        },
        index=close_df.index,
    )

    # 1. Evaluation labeler side
    labeler = UnifiedReturnLabeler.from_slot(slot)
    labels = labeler.label(["A", "B"], "1d", start_date="2026-01-05", end_date="2026-01-09", horizons=1)
    a_day0 = labels.loc[(labels.instrument_id == "A") & (labels.signal_timestamp == grid_ts[0])].iloc[0]
    assert int(a_day0.entry_offset_bars) == offset
    assert a_day0.entry_timestamp == grid_ts[offset].tz_convert("UTC") - pd.Timedelta(hours=16)
    assert a_day0.exit_bar_timestamp == grid_ts[offset + 1]

    # 2. VectorBTBacktestRunner (input_factor mode)
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "enable_tday_filter": False,
            }
        },
        slot=slot,
        frequency="1d",
    )
    res = runner.run_backtest(
        scores,
        close_df,
        open_df=open_df,
        rebalance_dates=[d.to_pydatetime() for d in close_df.index],
    )
    runner.close()
    assert "error" not in res, res
    assert runner.effective_execution_profile is not None
    assert runner.effective_execution_profile.min_entry_offset_bars == offset
    assert runner.effective_execution_profile.entry_price_field == "open"
    assert res["execution_diagnostics"]["effective_min_entry_offset_bars"] == offset

    orders = res["portfolio"].orders.records_readable
    first_buy = orders.loc[orders["Side"] == "Buy"].iloc[0]
    first_col = first_buy["Column"][-1] if isinstance(first_buy["Column"], tuple) else first_buy["Column"]
    assert first_col == "A"
    assert pd.Timestamp(first_buy["Timestamp"]).tz_convert("UTC") == close_df.index[offset]

    tw = runner._last_target_weights
    for skip_i in range(offset):
        assert (tw.iloc[skip_i] == 0.0).all()
    assert tw.iloc[offset]["A"] == pytest.approx(100_000.0)
    assert tw.iloc[offset + 1]["B"] > 0.0

    # 3. End-to-end Harness factor-synthesis -> StrategyBundle alignment (bundle score mode)
    store_root = tmp_path / f"store_{offset}"
    grid = slot.calendar.bar_grid("1d")
    f_rows = grid.merge(pd.DataFrame({"instrument_id": ["A", "B"]}), how="cross")
    f_rows["factor_value"] = [
        float(scores.loc[row.timestamp_utc, row.instrument_id]) for row in f_rows.itertuples(index=False)
    ]
    _save_scheme_b_factor(store_root, slot=slot, factor_id="sig", frame=f_rows)

    synth_cfg = {
        "analysis_period": {"start_date": "2026-01-05", "end_date": "2026-01-09"},
        "factor_groups": {"g": {"factors": ["sig"]}},
        "synthesis_methods": {
            "ws": {
                "method": "weighted_sum",
                "factor_group": "g",
                "weights": {"sig": 1.0},
                "preprocess": {"sig": "none"},
            }
        },
        "performance_evaluation": {
            "top_k_strategy": {
                "enabled": True,
                "top_k": 1,
                "holding_period": 1,
                "rebalance_freq": "daily",
                "cost_rate": 0.0,
                "enable_tday_filter": False,
            }
        },
        "output_settings": {"composite_factor_prefix": "P"},
        "cache_settings": {"use_parquet_cache": False},
    }
    synth_path = tmp_path / f"synth_{offset}.json"
    synth_path.write_text(json.dumps(synth_cfg), encoding="utf-8")
    h_root = tmp_path / f"harness_{offset}"
    rc = harness_cli.main(
        [
            "factor-synthesis",
            "--config", str(synth_path),
            "--slot-config", str(slot_path),
            "--slot-id", slot.slot_id,
            "--factor-values-store", str(store_root),
            "--frequency", "1d",
            "--output-root", str(h_root),
            "--run-id", f"run-{offset}",
        ]
    )
    assert rc == 0
    summary = json.loads((h_root / "factor-synthesis" / f"run-{offset}" / "summary.json").read_text(encoding="utf-8"))
    bundle = load_strategy_bundle(summary["metrics"]["strategy_bundle"]["bundle_dir"])
    assert bundle.execution_profile["min_entry_offset_bars"] == offset
    assert bundle.execution_profile["entry_price_field"] == "open"
    assert bundle.execution_profile["exit_adjustment"] == 0
    assert bundle.execution_profile["boundary_policy"] == "cross_session"

    bt_ret = pd.read_csv(Path(summary["metrics"]["strategy_bundle"]["bundle_dir"]) / bundle.backtest_returns.path)["net_return"].to_numpy()
    # Day 0..offset have 0.0 return (no prior signal; bar `offset` enters at open and closes at day `offset` close).
    np.testing.assert_allclose(bt_ret[:offset], 0.0, atol=1e-12)
    # On bar `offset`, portfolio holds A from open[offset] -> close[offset].
    expected_entry_day_ret = float(close_df["A"].iloc[offset] / open_df["A"].iloc[offset] - 1.0)
    assert bt_ret[offset] == pytest.approx(expected_entry_day_ret, abs=1e-12)


def test_boundary_policy_same_session_enforced_in_labeler_and_rejected_in_backtest(tmp_path: Path) -> None:
    # 1. UnifiedReturnLabeler enforces same_session boundary on intraday bars.
    bars_csv = tmp_path / "intraday.csv"
    t_day1 = pd.date_range("2026-01-05 09:00:00Z", periods=3, freq="1h")
    t_day2 = pd.date_range("2026-01-06 09:00:00Z", periods=3, freq="1h")
    rows = []
    for ts in list(t_day1) + list(t_day2):
        rows.append(
            {
                "symbol": "A",
                "time": (ts + pd.Timedelta(hours=1)).isoformat(),
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 100.0,
            }
        )
    pd.DataFrame(rows).to_csv(bars_csv, index=False)
    slot_cfg = {
        "schema_version": 1,
        "slot_id": "intraday_same_session",
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "intraday_same_sess",
                "start": "2026-01-05",
                "end": "2026-01-06",
                "custom_sessions": {
                    "timezone": "UTC",
                    "weekdays": [0, 1, 2, 3, 4],
                    "segments": [{"open": "09:00", "close": "12:00"}],
                },
            },
        },
        "bar_feed": {
            "provider": "csv_files",
            "options": {
                "paths": [bars_csv.name],
                "frequency": "1h",
                "timestamp_anchor": "bar_end",
                "source_timezone": "UTC",
                "volume_mode": "interval",
                "data_version": "v1",
                "column_mapping": {
                    "symbol": "instrument_id",
                    "time": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
            },
        },
        "universe": {
            "provider": "static",
            "options": {
                "instruments": {"A": {"asset_class": "equity", "currency": "USD"}},
                "universes": {"all": ["A"]},
            },
        },
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {
            "entry_price_field": "open",
            "exit_price_field": "close",
            "min_entry_offset_bars": 1,
            "exit_adjustment": -1,
            "return_unit": "decimal",
            "gap_rule": "endpoints_only",
            "boundary_policy": "same_session",
            "allow_short": False,
            "commission_bps": 0.0,
            "sell_tax_bps": 0.0,
            "slippage_bps": 0.0,
            "benchmark_instrument": None,
        },
    }
    intraday_slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
    labels = UnifiedReturnLabeler.from_slot(intraday_slot).label(
        ["A"], "1h", start_date="2026-01-05", end_date="2026-01-06", horizons=1
    )
    # Bar 2 (11:00Z on 2026-01-05) would enter on 2026-01-06 -> must be excluded as session_boundary.
    assert labels.loc[labels.signal_timestamp == pd.Timestamp("2026-01-05 09:00:00Z"), "status"].item() == "valid"
    assert labels.loc[labels.signal_timestamp == pd.Timestamp("2026-01-05 11:00:00Z"), "exclusion_reason"].item() == "session_boundary"

    # 2. VectorBTBacktestRunner must fail fast on same_session (never silently fall back to cross_session).
    from utils.time_axis import Frequency

    bars = intraday_slot.bar_feed.read_execution_bars(
        ["A"], Frequency.parse("1h"), start_utc=t_day1[0], end_utc=t_day2[-1]
    )
    close_df = bars.pivot(index="timestamp_utc", columns="instrument_id", values="close").sort_index()
    open_df = bars.pivot(index="timestamp_utc", columns="instrument_id", values="open").sort_index()
    scores = pd.DataFrame(1.0, index=close_df.index, columns=["A"])
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor"}},
        slot=intraday_slot,
        frequency="1h",
    )
    with pytest.raises(ValueError, match="same_session"):
        runner.run_backtest(scores, close_df, open_df=open_df)
    runner.close()


def test_backtest_runner_and_bundle_fail_fast_on_unsupported_configs(tmp_path: Path) -> None:
    # 1. Unsupported exit_adjustment=1 fails fast in VectorBTBacktestRunner.
    _, bad_adj_slot, close_df, open_df, _ = _make_daily_declarative_slot(
        tmp_path,
        slot_id="bad_adj",
        min_entry_offset_bars=1,
        exit_adjustment=1,
        entry_price_field="open",
    )
    scores = pd.DataFrame({"A": 2.0, "B": 1.0}, index=close_df.index)
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor"}},
        slot=bad_adj_slot,
        frequency="1d",
    )
    with pytest.raises(ValueError, match="exit_adjustment"):
        runner.run_backtest(scores, close_df, open_df=open_df)
    runner.close()

    # 2. Explicit entry_offset_bars < profile.min_entry_offset_bars fails fast (no T-1 fallback).
    _, offset2_slot, close_df2, open_df2, _ = _make_daily_declarative_slot(
        tmp_path,
        slot_id="min_offset_2",
        min_entry_offset_bars=2,
        exit_adjustment=0,
        entry_price_field="open",
    )
    bad_runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "holding_period": 1,
                "entry_offset_bars": 1,
                "score_source": "input_factor",
            }
        },
        slot=offset2_slot,
        frequency="1d",
    )
    with pytest.raises(ValueError, match="entry_offset_bars"):
        bad_runner.run_backtest(scores, close_df2, open_df=open_df2)
    bad_runner.close()

    # 3. _resolve_effective_execution_profile fails fast if runner's effective profile diverges from slot's declared profile.
    synth = object.__new__(FactorSynthesizer)
    mismatched_runner = SimpleNamespace(
        effective_entry_price_field="open",
        effective_execution_profile=replace(offset2_slot.microstructure.execution_profile, min_entry_offset_bars=1),
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        slippage=0.0,
        allow_short=False,
        benchmark_instrument=None,
        holding_period=1,
    )
    with pytest.raises(ValueError, match="min_entry_offset_bars"):
        synth._resolve_effective_execution_profile(runner=mismatched_runner, slot=offset2_slot)

    # 4. ContractValidationError preserved for UnifiedReturnLabeler when entry_offset_bars < min_entry_offset_bars.
    with pytest.raises(ContractValidationError, match="entry_offset_bars"):
        UnifiedReturnLabeler.from_slot(offset2_slot).label(
            ["A"], "1d", start_date="2026-01-05", end_date="2026-01-09", horizons=1, entry_offset_bars=1
        )

    # 5. validate_effective_execution_profile detects mismatched exit_adjustment directly.
    declared = ExecutionProfile("open", "close", 1, 0, "decimal", "endpoints_only", "cross_session")
    with pytest.raises(ValueError, match="exit_adjustment"):
        validate_effective_execution_profile(declared, replace(declared, exit_adjustment=-1))


def test_bundle_mode_skips_initial_offset_bars_consistently_with_earlier_calendar(tmp_path: Path) -> None:
    _, slot, close_df, open_df, _ = _make_daily_declarative_slot(
        tmp_path,
        slot_id="earlier_cal_offset2",
        min_entry_offset_bars=2,
        exit_adjustment=0,
        entry_price_field="open",
        exit_price_field="close",
        boundary_policy="cross_session",
    )
    # Rebuild slot with a calendar that starts earlier than price_df (2026-01-01 vs 2026-01-05).
    slot_cfg = json.loads((tmp_path / "earlier_cal_offset2_slot.json").read_text(encoding="utf-8"))
    slot_cfg["calendar"]["options"]["start"] = "2026-01-01"
    early_slot_path = tmp_path / "earlier_cal_offset2_early_slot.json"
    early_slot_path.write_text(json.dumps(slot_cfg), encoding="utf-8")
    early_slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)

    store_root = tmp_path / "early_store"
    grid = early_slot.calendar.bar_grid("1d")
    f_rows = grid.merge(pd.DataFrame({"instrument_id": ["A", "B"]}), how="cross")
    f_rows["factor_value"] = np.where(f_rows["instrument_id"] == "A", 10.0, -10.0)
    digest = _save_scheme_b_factor(store_root, slot=early_slot, factor_id="sig", frame=f_rows)

    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "index": {
                    "models": [
                        {
                            "train_end": "2025-12-31",
                            "date_range": ["2026-01-01", "2026-01-09"],
                            "feature_names": ["sig"],
                            "weights": {"sig": 1.0},
                            "standardization_method": {"sig": "none"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "holding_period": 1,
                "score_source": "bundle",
                "cost_rate": 0.0,
                "enable_tday_filter": False,
            }
        },
        bundle_path=str(bundle_path),
        slot=early_slot,
        frequency="1d",
        factor_values_store=store_root,
        factor_identities={"sig": digest},
    )
    scores = pd.DataFrame({"A": 10.0, "B": -10.0}, index=close_df.index)
    res = runner.run_backtest(
        scores,
        close_df,
        open_df=open_df,
        rebalance_dates=[d.to_pydatetime() for d in close_df.index],
    )
    runner.close()
    assert "error" not in res, res
    tw = runner._last_target_weights
    # Both bar 0 and bar 1 must be skipped (0.0) when min_entry_offset_bars=2.
    assert (tw.iloc[0] == 0.0).all()
    assert (tw.iloc[1] == 0.0).all()
    assert tw.iloc[2]["A"] == pytest.approx(100_000.0)
