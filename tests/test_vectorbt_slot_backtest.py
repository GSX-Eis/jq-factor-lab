"""Stage 5-2 tests: MarketSlot VectorBT backtest, Scheme B FactorLoader, parity, and anti-regression."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import duckdb

from market_db_samples import declare_futures_bars, declare_market_db
import numpy as np
import pandas as pd
import pytest

from data_pipeline.loaders.market_bar_importer import import_market_bars
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from factor_synthesis.vectorbt_adapter.factor_loader import FactorLoader
from factor_synthesis.vectorbt_adapter.quality_filter import (
    QualityFilterError,
    apply_pit_quality_filter,
)
from factor_synthesis.vectorbt_adapter.universe_filter import (
    UniverseFilterError,
    apply_universe_filter,
)
from slots.cn_futures import build_cn_futures_slot
from slots.declarative import build_declarative_slot
from slots.jq_ashare_local import build_jq_ashare_local, daily_slot_config
from utils.market_db import read_market_db_profile
from utils.market_db.identity import bind_market_db_fingerprint, market_db_identity, slot_market_db_identity
from utils.market_factor.identity import build_identity_record, compute_factor_identity_hash
from utils.market_factor_store import MarketFactorValueStore
from utils.session_calendar import SessionCalendar
from utils.slots import SlotConfig


ROOT = Path(__file__).resolve().parents[1]
BASELINE_MASTER = "680e50b5f79c37fd65f52ac40777488a76dbcce4"


def _save_scheme_b_factor(
    store_root: Path,
    *,
    slot_id: str,
    slot_config_hash: str | None,
    calendar: SessionCalendar,
    factor_id: str,
    frequency: str,
    frame: pd.DataFrame,
    source_seed: int = 1,
    market_db: Mapping[str, str] | None = None,
) -> str:
    digest, payload = compute_factor_identity_hash(
        factor_id=factor_id,
        factor_class="SyntheticFactor",
        source_sha256="b" * 64,
        params={"seed": source_seed},
        frequency=frequency,
        session_view="ETH",
        analysis_mode="cross_sectional",
        bucket_strategy="session" if frequency == "1d" else "segment_anchored",
        calendar_hash=calendar.calendar_hash,
        slot_id=slot_id,
        slot_config_hash=slot_config_hash,
        adjustment_policy="none",
        slot_data_fingerprint=bind_market_db_fingerprint({"seed": source_seed}, market_db or {}),
        code_version="test-v1",
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
            frequency=frequency,
            session_view="ETH",
            source_build_id="test-build",
            calendar_hash=calendar.calendar_hash,
            code_version="test-v1",
            created_at=pd.Timestamp("2026-02-01T00:00:00Z"),
            batch_id=f"batch-{factor_id}-{source_seed}",
        ),
        identity_record=record,
        calendar=calendar,
    )
    return digest


def _create_ashare_parity_db(db_path: Path) -> tuple[list[str], pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, dict[str, list[str]], str]:
    start, end = "2023-01-03", "2023-03-31"
    calendar = SessionCalendar.from_pmc(start, end)
    sessions = calendar.sessions()
    stocks = [f"00000{i}.XSHE" for i in range(1, 7)]
    rng = np.random.default_rng(20261122)

    close_mat = 20.0 * np.exp(rng.normal(0.0005, 0.015, (len(sessions), len(stocks))).cumsum(axis=0))
    open_mat = close_mat * (1.0 + rng.normal(0.0, 0.003, close_mat.shape))
    close_df = pd.DataFrame(close_mat, index=sessions, columns=stocks)
    open_df = pd.DataFrame(open_mat, index=sessions, columns=stocks)

    conn = duckdb.connect(str(db_path))
    declare_market_db(conn, 'cn_equity')
    try:
        cal_cfg = json.loads((ROOT / "config/trading_calendar.json").read_text(encoding="utf-8"))
        all_days = pd.date_range(start, end)
        conn.execute("CREATE TABLE trading_calendar (calendar_id VARCHAR, date DATE, is_open BOOLEAN)")
        conn.executemany(
            "INSERT INTO trading_calendar VALUES (?, ?, ?)",
            [(cal_cfg["calendar_id"], d.date(), d in sessions) for d in all_days],
        )

        conn.execute(
            "CREATE TABLE price_data_daily ("
            "stock_code VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, "
            "volume DOUBLE, money DOUBLE, factor DOUBLE, high_limit DOUBLE, low_limit DOUBLE, paused INTEGER)"
        )
        price_rows = []
        for i, day in enumerate(sessions):
            for j, stock in enumerate(stocks):
                o_px = float(open_mat[i, j])
                c_px = float(close_mat[i, j])
                # Deterministic pause & limit events to verify identical directional handling.
                paused = 1 if (i == 12 and j == 0) else 0
                hi_lim = o_px * 0.99 if (i == 18 and j == 1) else max(o_px, c_px) * 1.15
                lo_lim = min(o_px, c_px) * 0.85
                price_rows.append(
                    (stock, day.date(), o_px, max(o_px, c_px) * 1.01, min(o_px, c_px) * 0.99, c_px,
                     1_000_000.0, o_px * 1_000_000.0, 1.0, hi_lim, lo_lim, paused)
                )
        conn.executemany(
            "INSERT INTO price_data_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            price_rows,
        )

        conn.execute("CREATE TABLE factor_data (stock_code VARCHAR, date DATE, is_st VARCHAR)")
        st_rows = [
            (stock, day.date(), "true" if (j == 5 and i >= 20) else "false")
            for i, day in enumerate(sessions)
            for j, stock in enumerate(stocks)
        ]
        conn.executemany("INSERT INTO factor_data VALUES (?, ?, ?)", st_rows)

        conn.execute("CREATE TABLE factor_values (date DATE, stock_code VARCHAR, factor_name VARCHAR, factor_value DOUBLE)")
        f_rows = []
        for day in sessions:
            for stock in stocks:
                f_rows.append((day.date(), stock, "fac_a", float(rng.normal())))
                f_rows.append((day.date(), stock, "fac_b", float(rng.normal())))
        conn.executemany("INSERT INTO factor_values VALUES (?, ?, ?, ?)", f_rows)

        # Multi-year PIT dividend quality history for quality_filter parity.
        conn.execute(
            "CREATE TABLE dividend_quality_financial_events ("
            "statement_type VARCHAR, stock_code VARCHAR, pub_date DATE, start_date DATE, end_date DATE, "
            "net_profit DOUBLE, net_operate_cash_flow DOUBLE, total_liability DOUBLE, total_assets DOUBLE)"
        )
        fin_rows = []
        for j, stock in enumerate(stocks):
            profit_scale = 100.0 + 15.0 * j
            for period in pd.period_range("2018Q1", "2022Q4", freq="Q"):
                pend = period.end_time.date()
                pub = (period.end_time + pd.Timedelta(days=30)).date()
                fin_rows.append(("income", stock, pub, period.start_time.date(), pend, profit_scale * period.quarter, 40.0 * period.quarter, None, None))
                fin_rows.append(("balance", stock, pub, period.start_time.date(), pend, None, None, 500.0, 1000.0 + 50.0 * j))
        conn.executemany(
            "INSERT INTO dividend_quality_financial_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            fin_rows,
        )

        conn.execute(
            "CREATE TABLE dividend_events ("
            "stock_code VARCHAR, report_date DATE, implementation_pub_date DATE, "
            "bonus_type VARCHAR, bonus_amount_rmb DOUBLE, bonus_ratio_rmb DOUBLE)"
        )
        div_rows = []
        for j, stock in enumerate(stocks):
            # Stock index 4 has zero dividends so quality_filter rejects it.
            if j == 4:
                continue
            for year in (2019, 2020, 2021):
                div_rows.append((stock, f"{year}-12-31", f"{year + 1}-06-30", "年度分红", 0.02 + 0.005 * j, 1.0 + 0.1 * j))
        conn.executemany("INSERT INTO dividend_events VALUES (?, ?, ?, ?, ?, ?)", div_rows)

        conn.execute("CREATE TABLE valuation_daily (stock_code VARCHAR, date DATE, market_cap DOUBLE)")
        val_rows = [(stock, day.date(), 1000.0 + 100.0 * j) for day in sessions for j, stock in enumerate(stocks)]
        conn.executemany("INSERT INTO valuation_daily VALUES (?, ?, ?)", val_rows)

        conn.execute("CREATE TABLE industry_membership (stock_code VARCHAR, date DATE, sw_l1_name VARCHAR, sw_l1_code VARCHAR)")
        ind_rows = [(stock, day.date(), "公用事业I", "801160") for day in sessions for stock in stocks]
        conn.executemany("INSERT INTO industry_membership VALUES (?, ?, ?, ?)", ind_rows)
    finally:
        conn.close()

    bundle_path = db_path.parent / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "index": {
                    "models": [
                        {
                            "train_end": "2023-01-03",
                            "weights": {"fac_a": 0.6, "fac_b": 0.4},
                            "feature_names": ["fac_a", "fac_b"],
                            "standardization_method": {"ALL": "zscore"},
                            "blend_only": False,
                            "blend_with_original_ratio": 0.0,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    dynamic_pools = {
        sessions[d].strftime("%Y-%m-%d"): list(stocks)
        for d in range(2, len(sessions), 5)
    }
    return stocks, sessions, close_df, open_df, dynamic_pools, str(bundle_path)




def test_scheme_b_market_factor_store_feeds_factor_loader_and_runner(tmp_path: Path) -> None:
    db_path = tmp_path / "ashare_scheme_b.duckdb"
    stocks, sessions, close_df, open_df, dynamic_pools, bundle_path = _create_ashare_parity_db(db_path)

    # Drop legacy factor_values table so only Scheme B MarketFactorValueStore exists.
    conn_w = duckdb.connect(str(db_path))
    market_db = market_db_identity(read_market_db_profile(conn_w)[0])
    raw_df = conn_w.execute("SELECT date, stock_code, factor_name, factor_value FROM factor_values").fetchdf()
    conn_w.execute("DROP TABLE factor_values")
    conn_w.close()

    store_root = tmp_path / "factor_store"
    calendar = SessionCalendar.from_pmc("2023-01-03", "2023-03-31")
    grid = calendar.bar_grid("1d")
    hashes: dict[str, str] = {}
    for fname in ("fac_a", "fac_b"):
        sub = raw_df.loc[raw_df["factor_name"].eq(fname)].copy()
        sub["session_id"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")
        merged = sub.rename(columns={"stock_code": "instrument_id"}).merge(
            grid, on="session_id", how="inner"
        )[["instrument_id", "timestamp_utc", "session_id", "factor_value"]]
        hashes[fname] = _save_scheme_b_factor(
            store_root,
            slot_id="jq_ashare_local",
            slot_config_hash=daily_slot_config(start="2023-01-03", end="2023-03-31").slot_config_hash,
            calendar=calendar,
            factor_id=fname,
            frequency="1d",
            frame=merged,
            market_db=market_db,
        )

    loader = FactorLoader(
        factor_values_store=store_root,
        slot_id="jq_ashare_local",
        frequency="1d",
        factor_identities=hashes,
    )
    sample_day = sessions[2].strftime("%Y-%m-%d")
    loaded = loader.load_factors(sample_day, stocks, ["fac_a", "fac_b"])
    assert loaded.shape == (len(stocks), 2)
    assert loaded.notna().all().all()
    by_hash = loader.load_factors(sample_day, stocks, [hashes["fac_a"], hashes["fac_b"]])
    np.testing.assert_allclose(loaded.to_numpy(), by_hash.to_numpy(), rtol=0, atol=1e-12)

    panel = loader.load_factor_panel(sessions[0].strftime("%Y-%m-%d"), sessions[-1].strftime("%Y-%m-%d"), stocks, ["fac_a"])
    assert panel["fac_a"].shape == (len(sessions), len(stocks))

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_jq_ashare_local(conn, start="2023-01-03", end="2023-03-31")
        runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 2, "cost_rate": 0.001, "holding_period": 1}},
            slot=slot,
            bundle_path=bundle_path,
            factor_values_store=store_root,
            factor_identities=hashes,
        )
        dummy_factor = pd.DataFrame(1.0, index=sessions, columns=stocks)
        res = runner.run_backtest(dummy_factor, close_df, open_df=open_df, dynamic_pools=dynamic_pools)
        runner.close()
    finally:
        conn.close()
    assert "error" not in res, res
    assert np.isfinite(res["annual_return"])
    assert res["trade_count"] > 0


def test_cn_futures_slot_end_to_end_and_optional_filter_semantics(tmp_path: Path) -> None:
    from utils.bars.futures_roll import build_roll_schedule
    from utils.time_axis import BarSchedule, Frequency
    from utils.time_axis.futures import futures_calendar

    db_path = tmp_path / "futures.duckdb"
    start, end = "2026-01-05", "2026-01-16"
    fcal = futures_calendar("RB")
    cal = SessionCalendar(
        fcal["calendar_id"],
        start,
        end,
        rth_reference_calendar=fcal["rth_reference_calendar"],
    )
    sessions = cal.sessions()
    schedule = BarSchedule(cal, "1d", bucket_strategy="segment_anchored")
    slot_times = pd.concat([schedule.for_session(day) for day in sessions], ignore_index=True)["start_utc"]
    contracts = ["RB2605.SHFE", "RB2610.SHFE"]
    rng = np.random.default_rng(7)

    conn = duckdb.connect(str(db_path))
    meta = pd.DataFrame(
        [
            dict(
                instrument_id=c,
                product_root="RB",
                asset_class="futures",
                exchange="SHFE",
                timezone=cal.timezone,
                contract_expiry="2026-05-15" if c == "RB2605.SHFE" else "2026-10-15",
                contract_month="2026-05" if c == "RB2605.SHFE" else "2026-10",
                price_adjustment="none",
                point_value=10.0,
                rth_reference_calendar=fcal["rth_reference_calendar"],
            )
            for c in contracts
        ]
    )
    bars_list = []
    for j, c in enumerate(contracts):
        close_vals = 3500.0 + j * 100.0 + np.cumsum(rng.normal(0.0, 20.0, len(slot_times)))
        open_vals = close_vals - 5.0
        bars_list.append(
            pd.DataFrame(
                dict(
                    timestamp_utc=slot_times,
                    instrument_id=c,
                    frequency="1d",
                    open=open_vals,
                    high=close_vals + 15.0,
                    low=open_vals - 15.0,
                    close=close_vals,
                    open_interest=300.0 if j == 0 else 100.0,
                    volume=500.0,
                    source="SYNTH",
                    data_version="v1",
                )
            )
        )
    declare_futures_bars(conn, "SHFE")
    import_market_bars(
        conn,
        pd.concat(bars_list, ignore_index=True),
        meta,
        bucket_strategy="segment_anchored",
        legacy_compatibility=False,
    )
    roll_start = sessions[1].strftime("%Y-%m-%d")
    build_roll_schedule(conn, product_roots=["RB"], start=roll_start, end=end)

    slot = build_cn_futures_slot(
        conn, product_root="RB", start=roll_start, end=end, commission_bps=2.0, slippage_bps=1.0
    )
    assert slot.microstructure.execution_profile.allow_short is True
    assert slot.microstructure.execution_profile.benchmark_instrument is None

    pools = {d.strftime("%Y-%m-%d"): list(contracts) for d in slot.calendar.sessions()[::2]}
    # Non-A-share slot without UniverseFilterPlugin passes through by default.
    passed, diag = apply_universe_filter(pools, config={"enabled": True}, slot=slot)
    assert passed == pools and diag["skipped_no_plugin"] is True
    # Explicit required=True fails closed with clear code.
    with pytest.raises(UniverseFilterError, match="UNIVERSE_FILTER_PLUGIN_MISSING"):
        apply_universe_filter(pools, config={"enabled": True, "required": True}, slot=slot)
    with pytest.raises(QualityFilterError, match="UNIVERSE_FILTER_PLUGIN_MISSING"):
        apply_pit_quality_filter(pools, slot=slot, config={"enabled": True, "required": True})

    grid = slot.calendar.bar_grid("1d")
    exec_bars = slot.bar_feed.read_execution_bars(
        contracts,
        Frequency.parse("1d"),
        start_utc=grid["timestamp_utc"].min(),
        end_utc=grid["timestamp_utc"].max(),
    )
    close_wide = exec_bars.pivot(index="timestamp_utc", columns="instrument_id", values="close").sort_index()
    open_wide = exec_bars.pivot(index="timestamp_utc", columns="instrument_id", values="open").sort_index()
    score_wide = pd.DataFrame(
        [[1.0, -1.0] if i % 2 == 0 else [-1.0, 1.0] for i in range(len(close_wide))],
        index=close_wide.index,
        columns=contracts,
    )

    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "score_source": "input_factor",
                "holding_period": 1,
                "quality_filter": {"enabled": True},
                "risk_overlay": {"regime_filter": {"enabled": True}},
            }
        },
        slot=slot,
        frequency="1d",
    )
    assert runner.allow_short is True
    assert runner.cost_rate == pytest.approx(0.0002)
    assert runner.slippage == pytest.approx(0.0001)
    res = runner.run_backtest(score_wide, close_wide, open_df=open_wide, dynamic_pools=pools)
    runner.close()
    conn.close()

    assert "error" not in res, res
    assets = res["portfolio"].asset_value(group_by=False)
    assert (assets > 0).any().any() and (assets < 0).any().any()
    assert np.isfinite(res["annual_return"])


def test_declarative_24x7_long_short_end_to_end_with_scheme_b(tmp_path: Path) -> None:
    bars_csv = tmp_path / "crypto_bars.csv"
    hours = pd.date_range("2026-01-03 00:00:00Z", "2026-01-05 23:00:00Z", freq="1h")
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"]
    rng = np.random.default_rng(19)
    bar_rows = []
    for j, sym in enumerate(symbols):
        px = 100.0 * (j + 1) * np.exp(rng.normal(0.0, 0.004, len(hours)).cumsum())
        for ts, p in zip(hours, px):
            bar_rows.append(
                {
                    "symbol": sym,
                    "time": (ts + pd.Timedelta(hours=1)).isoformat(),
                    "open": float(p),
                    "high": float(p * 1.005),
                    "low": float(p * 0.995),
                    "close": float(p * (1.0 + 0.001 * (1 if j % 2 == 0 else -1))),
                    "volume": 1000.0,
                }
            )
    pd.DataFrame(bar_rows).to_csv(bars_csv, index=False)

    cfg_dict = {
        "schema_version": 1,
        "slot_id": "crypto_24x7",
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "crypto_utc",
                "start": "2026-01-03",
                "end": "2026-01-05",
                "custom_sessions": {
                    "timezone": "UTC",
                    "weekdays": [0, 1, 2, 3, 4, 5, 6],
                    "segments": [
                        {"open": "00:00", "close": "00:00", "open_day_offset": 0, "close_day_offset": 1}
                    ],
                },
            },
        },
        "bar_feed": {
            "provider": "csv_files",
            "options": {
                "paths": ["crypto_bars.csv"],
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
                "instruments": {sym: {"asset_class": "crypto", "currency": "USD"} for sym in symbols},
                "universes": {"spot": symbols},
            },
        },
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {
            "entry_price_field": "open",
            "exit_price_field": "close",
            "min_entry_offset_bars": 1,
            "exit_adjustment": 0,
            "return_unit": "decimal",
            "gap_rule": "endpoints_only",
            "boundary_policy": "cross_session",
            "allow_short": True,
            "commission_bps": 5.0,
            "sell_tax_bps": 0.0,
            "slippage_bps": 2.0,
            "benchmark_instrument": None,
        },
    }
    slot = build_declarative_slot(SlotConfig.from_mapping(cfg_dict), base_dir=tmp_path)
    from utils.time_axis import Frequency

    exec_bars = slot.bar_feed.read_execution_bars(
        symbols, Frequency.parse("1h"), start_utc=hours[0], end_utc=hours[-1]
    )
    close_df = exec_bars.pivot(index="timestamp_utc", columns="instrument_id", values="close").sort_index()
    open_df = exec_bars.pivot(index="timestamp_utc", columns="instrument_id", values="open").sort_index()

    # Store 1h momentum factor in Scheme B MarketFactorValueStore.
    store_root = tmp_path / "crypto_factors"
    grid = slot.calendar.bar_grid("1h")
    factor_rows = grid.merge(pd.DataFrame({"instrument_id": symbols}), how="cross")
    score_vals = np.tile([3.0, 1.0, -1.0, -3.0], len(grid))
    fhash = _save_scheme_b_factor(
        store_root,
        slot_id="crypto_24x7",
        slot_config_hash=slot.slot_config_hash,
        calendar=slot.calendar,
        factor_id="crypto_mom",
        frequency="1h",
        frame=factor_rows.assign(factor_value=score_vals),
        market_db=slot_market_db_identity(slot),
    )

    bundle_path = tmp_path / "crypto_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "index": {
                    "models": [
                        {
                            "train_end": "2026-01-03",
                            "weights": {"crypto_mom": 1.0},
                            "feature_names": ["crypto_mom"],
                            "standardization_method": {"ALL": "raw_zscore"},
                            "blend_only": False,
                            "blend_with_original_ratio": 0.0,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 6}},
        slot=slot,
        frequency="1h",
        bundle_path=str(bundle_path),
        factor_values_store=store_root,
        factor_identities={"crypto_mom": fhash},
        return_series_output_dir=str(tmp_path / "returns"),
        return_series_trial_id="crypto_24x7",
    )
    assert runner.allow_short is True
    assert runner.benchmark_instrument is None
    assert runner.cost_rate == pytest.approx(0.0005)
    assert runner.slippage == pytest.approx(0.0002)

    dummy = pd.DataFrame(1.0, index=close_df.index, columns=symbols)
    res = runner.run_backtest(
        dummy,
        close_df,
        open_df=open_df,
        rebalance_freq="bar",
    )
    runner.close()

    assert "error" not in res, res
    assets = res["portfolio"].asset_value(group_by=False)
    btc_vals = np.asarray(
        assets.xs("BTC-USD", axis=1, level=-1)
        if isinstance(assets.columns, pd.MultiIndex)
        else assets["BTC-USD"]
    )
    ada_vals = np.asarray(
        assets.xs("ADA-USD", axis=1, level=-1)
        if isinstance(assets.columns, pd.MultiIndex)
        else assets["ADA-USD"]
    )
    assert bool((btc_vals > 0).any()) and bool((ada_vals < 0).any())
    assert res["return_series_artifact"] is not None
    assert np.isfinite(res["annual_return"])


def test_static_anti_regression_no_ashare_hardcodes_in_vectorbt_core() -> None:
    forbidden = (
        "price_data_daily",
        "all_a_paused_daily",
        "historical_st_daily",
        "dividend_events",
        "dividend_quality_financial_events",
        "valuation_daily",
        "index_price_daily",
        "style_factor_exposure",
        "industry_membership",
        "000300.XSHG",
        "399101.XSHE",
        "load_trading_calendar",
    )
    adapter_dir = ROOT / "factor_synthesis" / "vectorbt_adapter"
    for filename in ("backtest_runner.py", "quality_filter.py", "universe_filter.py"):
        text = (adapter_dir / filename).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"Forbidden hardcoded token {token!r} found in {filename}"
    assert len((adapter_dir / "backtest_runner.py").read_text(encoding="utf-8").splitlines()) < 4577


def _make_24x7_hourly_slot(
    tmp_path: Path,
    *,
    start: str = "2026-01-03",
    end: str = "2026-01-09",
    entry_price_field: str = "open",
    commission_bps: float = 0.0,
    sell_tax_bps: float = 0.0,
    allow_short: bool = True,
    symbols: tuple[str, ...] = ("A", "B"),
    open_price: float = 100.0,
    close_price: float = 100.0,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    bars_csv = tmp_path / f"bars_{start}_{end}.csv"
    hours = pd.date_range(f"{start} 00:00:00Z", f"{end} 23:00:00Z", freq="1h")
    rows = []
    for sym in symbols:
        for ts in hours:
            rows.append(
                {
                    "symbol": sym,
                    "time": (ts + pd.Timedelta(hours=1)).isoformat(),
                    "open": float(open_price),
                    "high": float(max(open_price, close_price) * 1.01),
                    "low": float(min(open_price, close_price) * 0.99),
                    "close": float(close_price),
                    "volume": 1000.0,
                }
            )
    pd.DataFrame(rows).to_csv(bars_csv, index=False)
    cfg = {
        "schema_version": 1,
        "slot_id": "crypto_24x7_probe",
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "crypto_utc",
                "start": start,
                "end": end,
                "custom_sessions": {
                    "timezone": "UTC",
                    "weekdays": [0, 1, 2, 3, 4, 5, 6],
                    "segments": [
                        {"open": "00:00", "close": "00:00", "open_day_offset": 0, "close_day_offset": 1}
                    ],
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
                "instruments": {sym: {"asset_class": "crypto", "currency": "USD"} for sym in symbols},
                "universes": {"spot": list(symbols)},
            },
        },
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {
            "entry_price_field": entry_price_field,
            "exit_price_field": "close",
            "min_entry_offset_bars": 1,
            "exit_adjustment": 0,
            "return_unit": "decimal",
            "gap_rule": "endpoints_only",
            "boundary_policy": "cross_session",
            "allow_short": allow_short,
            "commission_bps": commission_bps,
            "sell_tax_bps": sell_tax_bps,
            "slippage_bps": 0.0,
            "benchmark_instrument": None,
        },
    }
    slot = build_declarative_slot(SlotConfig.from_mapping(cfg), base_dir=tmp_path)
    from utils.time_axis import Frequency

    bars = slot.bar_feed.read_execution_bars(
        list(symbols), Frequency.parse("1h"), start_utc=hours[0], end_utc=hours[-1]
    )
    close_df = bars.pivot(index="timestamp_utc", columns="instrument_id", values="close").sort_index()
    open_df = bars.pivot(index="timestamp_utc", columns="instrument_id", values="open").sort_index()
    return slot, close_df, open_df


def test_intraday_dynamic_pool_respects_bar_timestamps_and_session_dates(tmp_path: Path) -> None:
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-05", end="2026-01-06", allow_short=False
    )
    scores = pd.DataFrame(1.0, index=close_df.index, columns=["A", "B"])
    dynamic_pools = {
        "2026-01-05 09:00:00Z": ["A"],
        "2026-01-05 15:00:00Z": ["B"],
    }
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    res = runner.run_backtest(scores, close_df, open_df=open_df, dynamic_pools=dynamic_pools)
    runner.close()
    assert "error" not in res, res
    tw = runner._last_target_weights
    t_09 = pd.Timestamp("2026-01-05 09:00:00Z")
    t_15 = pd.Timestamp("2026-01-05 15:00:00Z")
    assert tw.loc[t_09, "A"] == pytest.approx(100_000.0)
    assert pd.isna(tw.loc[t_09, "B"])
    assert tw.loc[t_15, "A"] == pytest.approx(0.0)
    assert tw.loc[t_15, "B"] == pytest.approx(100_000.0)

    # Date-only pool key resolves to the Slot session's opening bar even when
    # the session opens on the prior UTC calendar day (e.g. Asia/Shanghai night/morning).
    from dataclasses import replace
    sh_cal = SessionCalendar("XSHG", "2026-01-05", "2026-01-06")
    sh_grid = sh_cal.bar_grid("5min")
    sh_bars_csv = tmp_path / "sh_bars.csv"
    sh_rows = []
    for sym in ("A", "B"):
        for ts in sh_grid["timestamp_utc"]:
            sh_rows.append(
                {
                    "symbol": sym,
                    "time": (ts + pd.Timedelta(minutes=5)).isoformat(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
            )
    pd.DataFrame(sh_rows).to_csv(sh_bars_csv, index=False)
    sh_cfg = {
        "schema_version": 1,
        "slot_id": "xshg_5m_probe",
        "calendar": {"provider": "pmc", "options": {"calendar_id": "XSHG", "start": "2026-01-05", "end": "2026-01-06"}},
        "bar_feed": {
            "provider": "csv_files",
            "options": {
                "paths": [sh_bars_csv.name],
                "frequency": "5min",
                "timestamp_anchor": "bar_end",
                "source_timezone": "UTC",
                "volume_mode": "interval",
                "data_version": "v1",
                "column_mapping": {"symbol": "instrument_id", "time": "timestamp", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"},
            },
        },
        "universe": {"provider": "static", "options": {"instruments": {s: {"asset_class": "equity", "currency": "CNY"} for s in ("A", "B")}, "universes": {"all": ["A", "B"]}}},
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {"entry_price_field": "open", "exit_price_field": "close", "min_entry_offset_bars": 1, "exit_adjustment": 0, "return_unit": "decimal", "gap_rule": "endpoints_only", "boundary_policy": "cross_session", "allow_short": False, "commission_bps": 0.0, "sell_tax_bps": 0.0, "slippage_bps": 0.0, "benchmark_instrument": None},
    }
    sh_slot = build_declarative_slot(SlotConfig.from_mapping(sh_cfg), base_dir=tmp_path)
    from utils.time_axis import Frequency
    sh_exec = sh_slot.bar_feed.read_execution_bars(["A", "B"], Frequency.parse("5min"), start_utc=sh_grid["timestamp_utc"].min(), end_utc=sh_grid["timestamp_utc"].max())
    sh_close = sh_exec.pivot(index="timestamp_utc", columns="instrument_id", values="close").sort_index()
    sh_scores = pd.DataFrame(1.0, index=sh_close.index, columns=["A", "B"])
    sh_runner = VectorBTBacktestRunner({"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}}, slot=sh_slot, frequency="5min")
    sh_res = sh_runner.run_backtest(sh_scores, sh_close, dynamic_pools={"2026-01-05": ["A"], "2026-01-06": ["B"]})
    sh_runner.close()
    assert "error" not in sh_res, sh_res
    jan6_open = sh_grid.loc[sh_grid["session_id"].eq("2026-01-06"), "timestamp_utc"].iloc[0]
    assert jan6_open.strftime("%H:%M") == "01:30"
    assert sh_runner._last_target_weights.loc[jan6_open, "B"] == pytest.approx(100_000.0)


def test_misaligned_factor_index_never_overwritten_onto_earlier_bars(tmp_path: Path) -> None:
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-05", end="2026-01-05", allow_short=False
    )
    jan5_close = close_df.loc["2026-01-05 00:00:00Z":"2026-01-05 23:00:00Z"]
    jan5_open = open_df.loc[jan5_close.index]

    naive_jan6 = pd.DataFrame(
        1.0,
        index=pd.date_range("2026-01-06 00:00:00", periods=len(jan5_close), freq="1h"),
        columns=["A", "B"],
    )
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        runner.run_backtest(naive_jan6, jan5_close, open_df=jan5_open)

    utc_jan6 = naive_jan6.set_axis(naive_jan6.index.tz_localize("UTC"), axis=0)
    res = runner.run_backtest(
        utc_jan6,
        jan5_close,
        open_df=jan5_open,
        rebalance_dates=[pd.Timestamp("2026-01-05 01:00:00Z").to_pydatetime()],
    )
    runner.close()
    assert res.get("error") == "All weights NaN" or res.get("trade_count", 0) == 0


def test_short_orders_and_exits_enforce_directional_tradability_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-05", end="2026-01-05", allow_short=True
    )
    sub_close = close_df.iloc[:4]
    sub_open = open_df.iloc[:4]
    scores = pd.DataFrame({"A": [2.0, 2.0, 2.0, 2.0], "B": [1.0, 1.0, 1.0, 1.0]}, index=sub_close.index)
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    t1 = sub_close.index[1]
    t2 = sub_close.index[2]
    t3 = sub_close.index[3]
    seen_calls: list[tuple[str, str, bool]] = []

    def mask(date_str: str, stocks: list[str], side: str = "both", require_non_st: bool = True) -> list[str]:
        ts = pd.Timestamp(date_str).tz_convert("UTC") if pd.Timestamp(date_str).tzinfo else pd.Timestamp(date_str, tz="UTC")
        for s in stocks:
            seen_calls.append((s, side, require_non_st))
        if ts == t1 and side == "sell":
            # B allows buy but forbids sell -> cannot short B at t1
            return [s for s in stocks if s != "B"]
        if ts == t2 and side == "buy":
            # Later short cover on B at t2 is blocked when buy is forbidden
            return [s for s in stocks if s != "B"]
        return list(stocks)

    monkeypatch.setattr(runner, "_filter_tradeable_stocks", mask)
    weights = runner._compute_target_weights(
        scores,
        sub_close,
        {t1.isoformat(): ["A", "B"]},
        [t1.to_pydatetime()],
        open_df=sub_open,
    )
    assert weights.loc[t1, "A"] == pytest.approx(50_000.0)
    assert pd.isna(weights.loc[t1, "B"])

    # Now verify closing an existing short uses side='buy' (not 'sell') on liquidation & retry
    seen_calls.clear()
    def mask_short_exit(date_str: str, stocks: list[str], side: str = "both", require_non_st: bool = True) -> list[str]:
        ts = pd.Timestamp(date_str).tz_convert("UTC") if pd.Timestamp(date_str).tzinfo else pd.Timestamp(date_str, tz="UTC")
        for s in stocks:
            seen_calls.append((s, side, require_non_st))
        if ts == t2 and side == "buy":
            return [s for s in stocks if s != "B"]
        return list(stocks)

    monkeypatch.setattr(runner, "_filter_tradeable_stocks", mask_short_exit)
    weights2 = runner._compute_target_weights(
        scores,
        sub_close,
        {t1.isoformat(): ["A", "B"], t2.isoformat(): []},
        [t1.to_pydatetime(), t2.to_pydatetime()],
        open_df=sub_open,
    )
    assert weights2.loc[t1, "B"] == pytest.approx(-50_000.0)
    assert weights2.loc[t2, "A"] == pytest.approx(0.0)
    assert pd.isna(weights2.loc[t2, "B"])  # short cover blocked at t2 (buy forbidden)
    assert weights2.loc[t3, "B"] == pytest.approx(0.0)  # retried and covered at t3
    assert ("B", "buy", False) in seen_calls


def test_ashare_default_cost_rate_and_asymmetric_sell_tax_parity(tmp_path: Path) -> None:
    db_path = tmp_path / "ashare_fees.duckdb"
    _, sessions, close_df, open_df, dynamic_pools, bundle_path = _create_ashare_parity_db(db_path)
    dummy_factor = pd.DataFrame(1.0, index=sessions, columns=close_df.columns)
    cfg = {
        "top_k_strategy": {
            "top_k": 2,
            "holding_period": 1,
            "enable_tday_filter": True,
        },
        "calendar_filters": {"skip_months": []},
    }
    legacy = VectorBTBacktestRunner(
        cfg, bundle_path=bundle_path, factor_db_path=str(db_path), price_db_path=str(db_path)
    )
    legacy_res = legacy.run_backtest(dummy_factor, close_df, open_df=open_df, dynamic_pools=dynamic_pools)
    legacy.close()
    assert legacy.cost_rate == pytest.approx(0.003)

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        default_slot = build_jq_ashare_local(conn, start="2023-01-03", end="2023-03-31")
        slot_runner = VectorBTBacktestRunner(
            cfg, slot=default_slot, bundle_path=bundle_path, factor_db_path=str(db_path)
        )
        slot_res = slot_runner.run_backtest(dummy_factor, close_df, open_df=open_df, dynamic_pools=dynamic_pools)
        slot_runner.close()
        assert slot_runner.cost_rate == pytest.approx(0.003)
        np.testing.assert_allclose(
            slot_res["portfolio"].value().to_numpy(dtype=float),
            legacy_res["portfolio"].value().to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )

        zero_slot = build_jq_ashare_local(
            conn, start="2023-01-03", end="2023-03-31", commission_bps=0.0, sell_tax_bps=0.0
        )
        zero_runner = VectorBTBacktestRunner(
            cfg, slot=zero_slot, bundle_path=bundle_path, factor_db_path=str(db_path)
        )
        zero_res = zero_runner.run_backtest(dummy_factor, close_df, open_df=open_df, dynamic_pools=dynamic_pools)
        zero_runner.close()
        assert zero_runner.cost_rate == pytest.approx(0.0)
        assert zero_res["portfolio"].value().iloc[2] > slot_res["portfolio"].value().iloc[2]

        asym_slot = build_jq_ashare_local(
            conn, start="2023-01-03", end="2023-03-31", commission_bps=5.0, sell_tax_bps=10.0
        )
        asym_runner = VectorBTBacktestRunner(
            cfg, slot=asym_slot, bundle_path=bundle_path, factor_db_path=str(db_path)
        )
        asym_res = asym_runner.run_backtest(dummy_factor, close_df, open_df=open_df, dynamic_pools=dynamic_pools)
        asym_runner.close()
        orders = asym_res["portfolio"].orders.records_readable
        buys = orders[orders["Side"] == "Buy"]
        sells = orders[orders["Side"] == "Sell"]
        assert not buys.empty and not sells.empty
        np.testing.assert_allclose(
            buys["Fees"].to_numpy(dtype=float),
            (buys["Size"] * buys["Price"]).to_numpy(dtype=float) * 0.0005,
            rtol=0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            sells["Fees"].to_numpy(dtype=float),
            (sells["Size"] * sells["Price"]).to_numpy(dtype=float) * 0.0015,
            rtol=0,
            atol=1e-9,
        )
    finally:
        conn.close()


def test_ashare_held_st_stock_cannot_be_bought_but_can_exit(tmp_path: Path) -> None:
    db_path = tmp_path / "ashare_st.duckdb"
    dates = pd.DatetimeIndex(["2023-01-04", "2023-01-05", "2023-01-06"])
    stocks = ["000001.XSHE", "000002.XSHE"]
    conn = duckdb.connect(str(db_path))
    declare_market_db(conn, 'cn_equity')
    try:
        conn.execute(
            "CREATE TABLE price_data_daily ("
            "stock_code VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, "
            "volume DOUBLE, money DOUBLE, factor DOUBLE, high_limit DOUBLE, low_limit DOUBLE, paused INTEGER)"
        )
        for d in dates:
            for s in stocks:
                conn.execute(
                    "INSERT INTO price_data_daily VALUES (?, ?, 10.0, 11.0, 9.0, 10.0, 1000.0, 10000.0, 1.0, 11.0, 9.0, 0)",
                    [s, d.date()],
                )
        conn.execute("CREATE TABLE historical_st_daily (stock_code VARCHAR, date DATE, is_st BOOLEAN)")
        conn.execute(
            "INSERT INTO historical_st_daily VALUES ('000001.XSHE', DATE '2023-01-05', true), ('000001.XSHE', DATE '2023-01-06', true)"
        )
    finally:
        conn.close()

    conn_ro = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_jq_ashare_local(conn_ro, start="2023-01-04", end="2023-01-06")
        # S2/S4 default tradability_mask still excludes ST
        grid = slot.calendar.bar_grid("1d", "2023-01-05", "2023-01-05")
        assert not bool(slot.microstructure.tradability_mask(grid.assign(side="sell"), instrument_id="000001.XSHE").iloc[0])

        scores = pd.DataFrame(
            [[2.0, 1.0], [3.0, 1.0], [3.0, 1.0]],
            index=dates,
            columns=stocks,
        )
        prices = pd.DataFrame(10.0, index=dates, columns=stocks)
        runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 1, "score_source": "input_factor", "cost_rate": 0.0}},
            slot=slot,
        )
        assert runner._filter_tradeable_stocks("2023-01-05", ["000001.XSHE"], side="buy", require_non_st=True) == []
        assert runner._filter_tradeable_stocks("2023-01-05", ["000001.XSHE"], side="sell", require_non_st=False) == ["000001.XSHE"]

        weights = runner._compute_target_weights(
            scores, prices, {}, [dates[1], dates[2]], open_df=prices
        )
        assert pd.isna(weights.loc[dates[1], "000001.XSHE"])
        assert weights.loc[dates[1], "000002.XSHE"] > 0
        runner.close()
    finally:
        conn_ro.close()

    # Transition from non-ST on 2023-01-05 to ST on 2023-01-06
    conn_w = duckdb.connect(str(db_path))
    conn_w.execute("DELETE FROM historical_st_daily WHERE date = DATE '2023-01-05'")
    conn_w.close()
    conn_ro2 = duckdb.connect(str(db_path), read_only=True)
    try:
        slot2 = build_jq_ashare_local(conn_ro2, start="2023-01-04", end="2023-01-06")
        runner2 = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 1, "score_source": "input_factor", "cost_rate": 0.0}},
            slot=slot2,
        )
        weights2 = runner2._compute_target_weights(
            scores, prices, {}, [dates[1], dates[2]], open_df=prices
        )
        assert weights2.loc[dates[1], "000001.XSHE"] > 0
        assert weights2.loc[dates[2], "000001.XSHE"] == pytest.approx(0.0)
        assert weights2.loc[dates[2], "000002.XSHE"] > 0
        runner2.close()
    finally:
        conn_ro2.close()


def test_slot_entry_price_field_open_executes_without_explicit_open_df(tmp_path: Path) -> None:
    slot, close_df, _ = _make_24x7_hourly_slot(
        tmp_path,
        start="2026-01-05",
        end="2026-01-05",
        entry_price_field="open",
        allow_short=False,
        open_price=110.0,
        close_price=100.0,
    )
    sub_close = close_df.iloc[:3]
    scores = pd.DataFrame({"A": [2.0, 2.0, 2.0], "B": [1.0, 1.0, 1.0]}, index=sub_close.index)
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    res = runner.run_backtest(
        scores,
        sub_close,
        rebalance_dates=[sub_close.index[1].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    orders = res["portfolio"].orders.records_readable
    assert len(orders) == 1
    assert float(orders["Price"].iloc[0]) == pytest.approx(110.0)


def test_legacy_ashare_scheme_b_factor_loader_isolates_by_jq_ashare_local_slot(tmp_path: Path) -> None:
    db_path = tmp_path / "ashare_isolation.duckdb"
    _, sessions, close_df, open_df, dynamic_pools, _ = _create_ashare_parity_db(db_path)
    store_root = tmp_path / "factor_store"
    ext_cal = SessionCalendar("XSHG", "2023-01-03", "2023-03-31")
    ext_grid = ext_cal.bar_grid("1d")
    ext_rows = ext_grid.assign(instrument_id="000001.XSHE", factor_value=42.0)[
        ["instrument_id", "timestamp_utc", "session_id", "factor_value"]
    ]
    ext_hash = _save_scheme_b_factor(
        store_root,
        slot_id="external_other_slot",
        slot_config_hash="c" * 64,
        calendar=ext_cal,
        factor_id="momentum",
        frequency="1d",
        frame=ext_rows,
    )

    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1}},
        price_db_path=str(db_path),
        factor_values_store=store_root,
    )
    loader = runner._slot_provider.create_factor_loader(
        factor_db_path=None,
        factor_db_conn=None,
        shared_cache=None,
        price_db_path=str(db_path),
        price_db_conn=None,
    )
    assert loader is not None
    assert loader.slot_id == "jq_ashare_local"
    day_str = sessions[2].strftime("%Y-%m-%d")
    by_name = loader.load_factors(day_str, ["000001.XSHE"], ["momentum"])
    by_hash = loader.load_factors(day_str, ["000001.XSHE"], [ext_hash])
    assert pd.isna(by_name.loc["000001.XSHE", "momentum"])
    assert pd.isna(by_hash.loc["000001.XSHE", ext_hash])

    pinned_loader = FactorLoader(
        factor_values_store=store_root,
        slot_id="jq_ashare_local",
        frequency="1d",
        factor_identities={"momentum": ext_hash},
    )
    with pytest.raises(ValueError, match="slot_id='jq_ashare_local'"):
        pinned_loader.load_factors(day_str, ["000001.XSHE"], ["momentum"])
    runner.close()


def test_annualization_invariant_across_weekday_and_weekend_subwindows_of_24x7_slot(tmp_path: Path) -> None:
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-03", end="2026-01-09", allow_short=True
    )
    # Mon-Wed (weekdays only) vs Sat-Sun (weekend only) from the exact same 24/7 slot
    weekday_close = close_df.loc["2026-01-05 00:00:00Z":"2026-01-06 23:00:00Z"]
    weekend_close = close_df.loc["2026-01-03 00:00:00Z":"2026-01-04 23:00:00Z"]
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    wf, wy, wbars = runner._slot_provider.annualization_spec()
    assert pd.Timedelta(wf) == pd.Timedelta("1h")
    assert wbars == pytest.approx(8760.0)
    assert runner._vbt_freq_and_year_freq(pd.DatetimeIndex(weekday_close.index)) == (
        wf,
        pd.Timedelta(days=365),
    )
    assert runner._vbt_freq_and_year_freq(pd.DatetimeIndex(weekend_close.index)) == (
        wf,
        pd.Timedelta(days=365),
    )

    # Also verify A-share daily preserves 252 convention
    ashare_runner = VectorBTBacktestRunner({"top_k_strategy": {"top_k": 1}})
    from utils.session_calendar import TradingCalendar
    b_dates = pd.bdate_range("2026-01-05", periods=5)
    b_cal = TradingCalendar(b_dates, b_dates[0], b_dates[-1])
    assert ashare_runner._slot_provider.annualization_spec(calendar=b_cal) == ("1D", "252D", 252.0)

    scores_wd = pd.DataFrame({"A": 2.0, "B": 1.0}, index=weekday_close.index)
    res_wd = runner.run_backtest(
        scores_wd,
        weekday_close,
        open_df=open_df.loc[weekday_close.index],
        rebalance_dates=[weekday_close.index[1].to_pydatetime()],
    )
    orders_val = float((res_wd["portfolio"].orders.records_readable["Size"] * res_wd["portfolio"].orders.records_readable["Price"]).abs().sum())
    avg_val = float(res_wd["portfolio"].value().mean())
    expected_turnover = orders_val / avg_val / (len(weekday_close) / 8760.0) / 2.0
    assert res_wd["annual_turnover"] == pytest.approx(expected_turnover, rel=0, abs=1e-9)
    runner.close()


def test_top_k_benchmark_instrument_propagates_to_regime_and_crisis_spread(tmp_path: Path) -> None:
    slot, _, _ = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-05", end="2026-01-06", allow_short=False
    )
    cal_grid = slot.calendar.bar_grid("1d")
    n_days = len(cal_grid)
    aux_calls: list[list[str]] = []

    class SpreadAux:
        def read_aux_features(self, grid: pd.DataFrame, *, features: list[str]) -> pd.DataFrame:
            aux_calls.append(sorted(set(grid["instrument_id"])))
            out = grid[["instrument_id", "timestamp_utc", "session_id"]].copy()
            out["available_at_utc"] = out["timestamp_utc"]
            out["index_price_daily.close"] = np.where(
                out["instrument_id"].eq("MICRO"), 100.0 + np.arange(len(out)) * 0.5, 100.0
            )
            return out

    from dataclasses import replace

    # Build a longer 60-day calendar so rolling spread has enough observations
    long_cal = SessionCalendar("XSHG", "2026-01-05", "2026-04-30")
    slot_with_aux = replace(slot, calendar=long_cal, aux_features=SpreadAux())
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "benchmark_instrument": "REF",
            }
        },
        slot=slot_with_aux,
    )
    assert runner.benchmark_instrument == "REF"
    rf = {"micro_index": "MICRO", "spread_window_days": 2, "zscore_window_days": 20}
    spread = runner._regime_spread_zscore(rf)
    assert spread is not None and not spread.empty
    assert aux_calls[-1] == ["MICRO", "REF"]

    # Explicit sub-strategy benchmark_index retains priority over top_k_strategy.benchmark_instrument
    rf_sub = {
        "micro_index": "MICRO",
        "benchmark_index": "SUB_BENCH",
        "spread_window_days": 2,
        "zscore_window_days": 20,
    }
    spread_sub = runner._regime_spread_zscore(rf_sub)
    assert spread_sub is not None and not spread_sub.empty
    assert aux_calls[-1] == ["MICRO", "SUB_BENCH"]

    # Crisis guard spread path also receives REF
    cg = {
        "signals": {
            "limit_down": {"enabled": False},
            "turnover_collapse": {"enabled": False},
            "breadth": {"enabled": False},
            "spread": {
                "enabled": True,
                "micro_index": "MICRO",
                "benchmark_index": None,
                "spread_window_days": 2,
                "zscore_window_days": 20,
                "z_threshold": -1.5,
            },
        },
        "min_triggers": 1,
        "derisk_exposure": 0.5,
        "hard_exit_triggers": 2,
        "cooldown_days": 1,
    }
    votes = runner._crisis_guard_vote_series(cg, ["A"], pd.DataFrame())
    assert len(votes) == 1
    assert aux_calls[-1] == ["MICRO", "REF"]
    runner.close()


def test_truncated_window_date_pool_does_not_override_same_day_update(tmp_path: Path) -> None:
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-05", end="2026-01-05", allow_short=False
    )
    trunc_close = close_df.loc["2026-01-05 13:00:00Z":]
    trunc_open = open_df.loc[trunc_close.index]
    scores = pd.DataFrame({"A": 1.0, "B": 2.0}, index=trunc_close.index)
    dynamic_pools = {
        "2026-01-05": ["A"],
        "2026-01-05 12:00:00Z": ["B"],
    }
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    t_14 = pd.Timestamp("2026-01-05 14:00:00Z")
    res = runner.run_backtest(
        scores,
        trunc_close,
        open_df=trunc_open,
        dynamic_pools=dynamic_pools,
        rebalance_dates=[t_14.to_pydatetime()],
    )
    assert "error" not in res, res
    tw = runner._last_target_weights
    assert tw.loc[t_14, "B"] == pytest.approx(100_000.0)
    assert pd.isna(tw.loc[t_14, "A"])

    # Also verify automatic pool rebalance date mapping stays in-sample while preserving 00:00Z < 12:00Z order
    reb = runner._slot_provider.resolve_pool_rebalance_dates(trunc_close.index, dynamic_pools.keys())
    assert reb == [pd.Timestamp("2026-01-05 13:00:00Z").to_pydatetime()]
    res_auto = runner.run_backtest(scores, trunc_close, open_df=trunc_open, dynamic_pools=dynamic_pools)
    runner.close()
    assert res_auto.get("error") == "All weights NaN" or not (
        res_auto["portfolio"].orders.records_readable["Column"].astype(str).eq("A").any()
    )


def test_overlay_reallocation_rechecks_directional_masks_for_restricted_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    symbols = ("A", "B", "C", "D")
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-05", end="2026-01-05", allow_short=True, symbols=symbols
    )
    sub_close = close_df.iloc[:4]
    sub_open = open_df.iloc[:4]
    t1 = sub_close.index[1]
    t2 = sub_close.index[2]
    scores = pd.DataFrame(
        {
            "A": [4.0, 4.0, 4.0, 4.0],
            "D": [3.0, 3.0, 3.0, 3.0],
            "B": [-3.0, -3.0, -3.0, -3.0],
            "C": [-4.0, -4.0, -4.0, -4.0],
        },
        index=sub_close.index,
    )
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "banding": {"enabled": True},
            }
        },
        slot=slot,
        frequency="1h",
    )

    def mask(date_str: str, stocks: list[str], side: str = "both", require_non_st: bool = True) -> list[str]:
        ts = pd.Timestamp(date_str).tz_convert("UTC") if pd.Timestamp(date_str).tzinfo else pd.Timestamp(date_str, tz="UTC")
        if ts == t2:
            if side == "buy":
                return [s for s in stocks if s != "B"]
            if side == "sell":
                return [s for s in stocks if s != "C"]
        return list(stocks)

    monkeypatch.setattr(runner, "_filter_tradeable_stocks", mask)
    weights = runner._compute_target_weights(
        scores,
        sub_close,
        {},
        [t1.to_pydatetime(), t2.to_pydatetime()],
        open_df=sub_open,
        exposure_by_date={t1: 1.0, t2: 0.8},
    )
    # t1 opens equal-weight long/short: A=+25k, D=+25k, B=-25k, C=-25k
    assert weights.loc[t1, "B"] == pytest.approx(-25_000.0)
    assert weights.loc[t1, "C"] == pytest.approx(-25_000.0)
    # t2: B cannot buy (stays NaN); C cannot sell, so C must NEVER receive a target < -25,000
    assert pd.isna(weights.loc[t2, "B"])
    c_t2 = weights.loc[t2, "C"]
    assert pd.isna(c_t2) or c_t2 >= -25_000.0 - 1e-9

    pf = runner._simulate_portfolio(weights, sub_close, sub_open, True)
    orders = pf.orders.records_readable
    c_sells_at_t2 = orders[
        orders["Column"].astype(str).eq("C")
        & orders["Side"].eq("Sell")
        & (pd.to_datetime(orders["Timestamp"], utc=True) == t2)
    ]
    assert c_sells_at_t2.empty
    runner.close()


def test_utc_bar_skip_month_liquidates_long_and_short_at_first_bar_of_skip_interval(
    tmp_path: Path,
) -> None:
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-31", end="2026-02-01", allow_short=True, symbols=("A", "B")
    )
    sub_close = close_df.loc["2026-01-31 21:00:00Z":"2026-02-01 03:00:00Z"]
    sub_open = open_df.loc[sub_close.index]
    scores = pd.DataFrame({"A": 2.0, "B": -2.0}, index=sub_close.index)
    t_open = pd.Timestamp("2026-01-31 23:00:00Z")
    t_feb_first = pd.Timestamp("2026-02-01 00:00:00Z")

    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
            },
            "calendar_filters": {"skip_months": [2]},
        },
        slot=slot,
        frequency="1h",
    )
    res = runner.run_backtest(
        scores,
        sub_close,
        open_df=sub_open,
        rebalance_dates=[t_open.to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    tw = runner._last_target_weights
    assert tw.loc[t_open, "A"] == pytest.approx(50_000.0)
    assert tw.loc[t_open, "B"] == pytest.approx(-50_000.0)
    assert tw.loc[t_feb_first, "A"] == pytest.approx(0.0)
    assert tw.loc[t_feb_first, "B"] == pytest.approx(0.0)

    orders = res["portfolio"].orders.records_readable
    feb_orders = orders[pd.to_datetime(orders["Timestamp"], utc=True) == t_feb_first]
    assert len(feb_orders) == 2
    assert set(feb_orders["Side"]) == {"Buy", "Sell"}
    assets = res["portfolio"].asset_value(group_by=False)
    assert np.allclose(assets.loc[t_feb_first:].to_numpy(dtype=float), 0.0)


def test_explicit_open_entry_field_rejects_missing_open_and_marks_nan_untradeable(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    slot, close_df, _ = _make_24x7_hourly_slot(
        tmp_path,
        start="2026-01-05",
        end="2026-01-05",
        entry_price_field="open",
        allow_short=False,
        open_price=110.0,
        close_price=100.0,
    )
    sub_close = close_df.iloc[:3]
    scores = pd.DataFrame({"A": [2.0, 2.0, 2.0], "B": [1.0, 1.0, 1.0]}, index=sub_close.index)
    orig_feed = slot.bar_feed

    class _WrappedFeed:
        def __init__(self, transform):
            self._transform = transform

        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
            return orig_feed.read_signal_bars(
                instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
            )

        def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc):
            df = orig_feed.read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)
            return self._transform(df)

    no_open_slot = replace(slot, bar_feed=_WrappedFeed(lambda df: df.drop(columns=["open"])))
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=no_open_slot,
        frequency="1h",
    )
    with pytest.raises(ValueError, match="entry_price_field='open'"):
        runner.run_backtest(scores, sub_close, rebalance_dates=[sub_close.index[1].to_pydatetime()])
    runner.close()

    def _raise_feed(_df):
        raise RuntimeError("feed unavailable")

    fail_slot = replace(slot, bar_feed=_WrappedFeed(_raise_feed))
    runner_fail = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=fail_slot,
        frequency="1h",
    )
    with pytest.raises(ValueError, match="entry_price_field='open'"):
        runner_fail.run_backtest(scores, sub_close, rebalance_dates=[sub_close.index[1].to_pydatetime()])
    runner_fail.close()

    # Partial NaN in open column marks only the missing bar untradeable (no close fallback)
    t1 = sub_close.index[1]

    def _mask_t1_open(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mask = out["instrument_id"].eq("A") & (pd.to_datetime(out["timestamp_utc"], utc=True) == t1)
        out.loc[mask, "open"] = np.nan
        return out

    partial_slot = replace(slot, bar_feed=_WrappedFeed(_mask_t1_open))
    runner_partial = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=partial_slot,
        frequency="1h",
    )
    res_partial = runner_partial.run_backtest(scores, sub_close, rebalance_dates=[t1.to_pydatetime()])
    runner_partial.close()
    assert res_partial.get("error") == "All weights NaN" or res_partial.get("trade_count", 0) == 0


def test_hourly_vol_targeting_uses_bar_timestamp_key_and_strictly_prior_returns(
    tmp_path: Path,
) -> None:
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path, start="2026-01-05", end="2026-01-05", allow_short=False, symbols=("A", "B")
    )
    sub_close = close_df.iloc[:10].copy()
    # Non-constant hourly prices for A so 02:00..06:00 returns have positive std
    a_px = np.array([100.0, 100.0, 102.0, 99.0, 103.0, 98.0, 104.0, 104.0, 104.0, 104.0])
    sub_close["A"] = a_px
    sub_open = sub_close.copy()
    scores = pd.DataFrame({"A": 2.0, "B": 1.0}, index=sub_close.index)
    t_01 = sub_close.index[1]
    t_07 = sub_close.index[7]
    assert t_07 == pd.Timestamp("2026-01-05 07:00:00Z")

    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "risk_overlay": {
                    "vol_targeting": {
                        "enabled": True,
                        "target_annual_vol": 0.10,
                        "lookback_days": 5,
                        "min_exposure": 0.01,
                        "max_exposure": 1.0,
                    }
                },
            }
        },
        slot=slot,
        frequency="1h",
    )
    res = runner.run_backtest(
        scores,
        sub_close,
        open_df=sub_open,
        rebalance_dates=[t_01.to_pydatetime(), t_07.to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    prior_5_rets = sub_close["A"].pct_change().iloc[2:7]
    expected_exp = float(np.clip(0.10 / (float(prior_5_rets.std()) * np.sqrt(8760.0)), 0.01, 1.0))
    assert expected_exp < 1.0
    assert runner._last_exposure_by_date[t_07] == pytest.approx(expected_exp, rel=0, abs=1e-12)
    assert pd.Timestamp("2026-01-05 00:00:00Z") not in runner._last_exposure_by_date
    tw = runner._last_target_weights
    nav_at_07 = float(res["portfolio"].value().iloc[6])
    # Pass-2 orders at 07:00 UTC reflect the scaled exposure rather than falling back to 1.0
    assert tw.loc[t_07, "A"] == pytest.approx(nav_at_07 * expected_exp, rel=1e-6)


def test_explicit_open_entry_field_resolves_slot_open_when_supplied_open_df_unaligned(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    slot, close_df, _ = _make_24x7_hourly_slot(
        tmp_path,
        start="2026-01-05",
        end="2026-01-05",
        entry_price_field="open",
        allow_short=False,
        open_price=110.0,
        close_price=100.0,
    )
    sub_close = close_df.iloc[:3]
    scores = pd.DataFrame({"A": [2.0, 2.0, 2.0], "B": [1.0, 1.0, 1.0]}, index=sub_close.index)

    # Disjoint columns in caller-supplied open_df must re-resolve from Slot BarFeed (110.0), never fall back to close (100.0)
    disjoint_open = pd.DataFrame(
        {"X": 99.0, "Y": 99.0},
        index=sub_close.index,
    )
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    res = runner.run_backtest(
        scores,
        sub_close,
        open_df=disjoint_open,
        rebalance_dates=[sub_close.index[1].to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    orders = res["portfolio"].orders.records_readable
    assert len(orders) == 1
    assert float(orders["Price"].iloc[0]) == pytest.approx(110.0)

    # Disjoint dates in caller-supplied open_df also re-resolve from Slot BarFeed (110.0)
    wrong_date_open = pd.DataFrame(
        {"A": 99.0, "B": 99.0},
        index=close_df.index[-3:],
    )
    runner_dates = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    res_dates = runner_dates.run_backtest(
        scores,
        sub_close,
        open_df=wrong_date_open,
        rebalance_dates=[sub_close.index[1].to_pydatetime()],
    )
    runner_dates.close()
    assert "error" not in res_dates, res_dates
    orders_dates = res_dates["portfolio"].orders.records_readable
    assert len(orders_dates) == 1
    assert float(orders_dates["Price"].iloc[0]) == pytest.approx(110.0)

    # If BarFeed also lacks usable open prices when open_df is unaligned, fail closed with ValueError
    orig_feed = slot.bar_feed

    class _NoOpenFeed:
        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
            return orig_feed.read_signal_bars(
                instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
            )

        def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc):
            return orig_feed.read_execution_bars(
                instruments, frequency, start_utc=start_utc, end_utc=end_utc
            ).drop(columns=["open"])

    no_open_slot = replace(slot, bar_feed=_NoOpenFeed())
    runner_fail = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=no_open_slot,
        frequency="1h",
    )
    with pytest.raises(ValueError, match="entry_price_field='open'"):
        runner_fail.run_backtest(
            scores,
            sub_close,
            open_df=disjoint_open,
            rebalance_dates=[sub_close.index[1].to_pydatetime()],
        )
    runner_fail.close()


def test_overlay_liquidity_cap_preserves_capped_weights_and_enforces_caps_on_reallocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    symbols = ("A", "B", "C")
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path,
        start="2026-01-05",
        end="2026-01-05",
        allow_short=True,
        symbols=symbols,
        open_price=100.0,
        close_price=100.0,
    )
    sub_close = close_df.iloc[:4, :2]
    sub_open = open_df.iloc[:4, :2]
    t1 = sub_close.index[1]
    scores_ls = pd.DataFrame({"A": 2.0, "B": -2.0}, index=sub_close.index)

    # 1. No blocked orders: +0.25 / -0.50 capped weights must NOT be scaled up to 100k gross
    runner_ls = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "init_cash": 100_000.0,
                "liquidity_cap": {
                    "enabled": True,
                    "max_participation": 0.25,
                    "adv_lookback_days": 5,
                },
            }
        },
        slot=slot,
        frequency="1h",
    )
    monkeypatch.setattr(
        runner_ls,
        "_average_daily_value",
        lambda stocks, as_of, lookback: {"A": 100_000.0, "B": 200_000.0},
    )
    res_ls = runner_ls.run_backtest(
        scores_ls,
        sub_close,
        open_df=sub_open,
        rebalance_dates=[t1.to_pydatetime()],
    )
    runner_ls.close()
    assert "error" not in res_ls, res_ls
    tw_ls = runner_ls._last_target_weights
    assert tw_ls.loc[t1, "A"] == pytest.approx(25_000.0)
    assert tw_ls.loc[t1, "B"] == pytest.approx(-50_000.0)
    orders_ls = res_ls["portfolio"].orders.records_readable
    a_ord = orders_ls[orders_ls["Column"].map(lambda c: (c[-1] if isinstance(c, tuple) else c) == "A")]
    b_ord = orders_ls[orders_ls["Column"].map(lambda c: (c[-1] if isinstance(c, tuple) else c) == "B")]
    assert float((a_ord["Size"] * a_ord["Price"]).abs().sum()) == pytest.approx(25_000.0)
    assert float((b_ord["Size"] * b_ord["Price"]).abs().sum()) == pytest.approx(50_000.0)

    # 2. Signed caps also enforce ADV participation on short legs
    runner_ls_sym = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 1,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "init_cash": 100_000.0,
                "liquidity_cap": {
                    "enabled": True,
                    "max_participation": 0.25,
                    "adv_lookback_days": 5,
                },
            }
        },
        slot=slot,
        frequency="1h",
    )
    monkeypatch.setattr(
        runner_ls_sym,
        "_average_daily_value",
        lambda stocks, as_of, lookback: {"A": 100_000.0, "B": 100_000.0},
    )
    tw_sym = runner_ls_sym._compute_target_weights(
        scores_ls, sub_close, {}, [t1.to_pydatetime()], open_df=sub_open
    )
    runner_ls_sym.close()
    assert tw_sym.loc[t1, "A"] == pytest.approx(25_000.0)
    assert tw_sym.loc[t1, "B"] == pytest.approx(-25_000.0)

    # 3. Blocked order reallocation must re-enforce BOTH ADV caps and directional masks
    symbols_4 = ("A", "B", "C", "D")
    long_dir = tmp_path / "long_sub"
    long_dir.mkdir(parents=True, exist_ok=True)
    slot_long, close_long, open_long = _make_24x7_hourly_slot(
        long_dir,
        start="2026-01-05",
        end="2026-01-05",
        allow_short=False,
        symbols=symbols_4,
        open_price=100.0,
        close_price=100.0,
    )
    sub_c3 = close_long.iloc[:4]
    sub_o3 = open_long.iloc[:4]
    t1_3 = sub_c3.index[1]
    t2_3 = sub_c3.index[2]
    scores_3 = pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0, 2.0],
            "C": [3.0, 3.0, 3.0, 3.0],
            "D": [1.5, 1.5, 1.5, 1.5],
        },
        index=sub_c3.index,
    )
    runner_3 = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 3,
                "holding_period": 1,
                "score_source": "input_factor",
                "cost_rate": 0.0,
                "init_cash": 100_000.0,
                "enable_tday_filter": False,
                "banding": {"enabled": True},
                "liquidity_cap": {
                    "enabled": True,
                    "max_participation": 0.25,
                    "adv_lookback_days": 5,
                },
            }
        },
        slot=slot_long,
        frequency="1h",
    )
    monkeypatch.setattr(
        runner_3,
        "_average_daily_value",
        lambda stocks, as_of, lookback: {
            "A": 100_000.0,   # cap 25,000
            "B": 240_000.0,   # cap 60,000
            "C": 100_000.0,   # cap 25,000
            "D": 100_000.0,   # cap 25,000
        },
    )

    def _mask_3(date_str: str, stocks: list[str], side: str = "both", require_non_st: bool = True) -> list[str]:
        ts = pd.Timestamp(date_str).tz_convert("UTC") if pd.Timestamp(date_str).tzinfo else pd.Timestamp(date_str, tz="UTC")
        if ts == t2_3 and side == "buy":
            # At t2, C cannot buy (0 -> 25k blocked initially) and B cannot buy (50k -> 60k triggered by reallocation)
            return [s for s in stocks if s not in {"B", "C"}]
        return list(stocks)

    monkeypatch.setattr(runner_3, "_filter_tradeable_stocks", _mask_3)
    tw_3 = runner_3._compute_target_weights(
        scores_3,
        sub_c3,
        {t1_3.isoformat(): ["A", "B", "D"], t2_3.isoformat(): ["A", "B", "C"]},
        [t1_3.to_pydatetime(), t2_3.to_pydatetime()],
        open_df=sub_o3,
    )
    runner_3.close()
    # At t1 (A, B, D): A=25k (capped), D=25k (capped), B=50k (absorbed excess <= 60k)
    assert tw_3.loc[t1_3, "A"] == pytest.approx(25_000.0)
    assert tw_3.loc[t1_3, "B"] == pytest.approx(50_000.0)
    assert tw_3.loc[t1_3, "D"] == pytest.approx(25_000.0)
    # At t2 (A, B, C): D exits (0.0), C blocked (0 -> 25k forbidden), B reallocation buy (50k -> 60k) forbidden:
    # A must stay <= 25k ADV cap, B must remain NaN (no forbidden buy above held 50k), C remains NaN
    assert tw_3.loc[t2_3, "A"] == pytest.approx(25_000.0)
    assert pd.isna(tw_3.loc[t2_3, "B"])
    assert pd.isna(tw_3.loc[t2_3, "C"])
    assert tw_3.loc[t2_3, "D"] == pytest.approx(0.0)

    # 4. Unpatched SlotDataProvider.average_daily_value accepts UTC ISO timestamps without tz mismatch
    adv_real = runner_ls._slot_provider.average_daily_value(
        ["A", "B"], "2026-01-05T02:00:00+00:00", 2, None, None
    )
    assert adv_real == {"A": pytest.approx(100_000.0), "B": pytest.approx(100_000.0)}


def test_explicit_open_entry_field_handles_nan_intersection_and_all_nan_barfeed(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    slot, close_df, _ = _make_24x7_hourly_slot(
        tmp_path,
        start="2026-01-05",
        end="2026-01-05",
        entry_price_field="open",
        allow_short=False,
        open_price=110.0,
        close_price=100.0,
    )
    sub_close = close_df.iloc[:4]
    t1 = sub_close.index[1]
    t2 = sub_close.index[2]
    scores = pd.DataFrame(
        {"A": [2.0, 1.0, 1.0, 1.0], "B": [1.0, 2.0, 2.0, 2.0]},
        index=sub_close.index,
    )

    # 1. open_df intersects both columns (A) and dates (sub_close.index),
    #    but the intersection cells are all NaN (valid X in-sample, valid A only out-of-sample).
    #    Must NOT shrink common_stocks to ["A"] or skip BarFeed resolution.
    nan_intersection_open = pd.DataFrame(
        {
            "A": [np.nan] * len(sub_close) + [99.0, 99.0],
            "X": [99.0] * len(sub_close) + [np.nan, np.nan],
        },
        index=close_df.index[: len(sub_close) + 2],
    )
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=slot,
        frequency="1h",
    )
    res = runner.run_backtest(
        scores,
        sub_close,
        open_df=nan_intersection_open,
        rebalance_dates=[t1.to_pydatetime(), t2.to_pydatetime()],
    )
    runner.close()
    assert "error" not in res, res
    orders = res["portfolio"].orders.records_readable
    assert len(orders) == 3  # t1 buy A, t2 sell A + buy B (B preserved!)
    assert all(float(p) == pytest.approx(110.0) for p in orders["Price"])
    traded_cols = {c[-1] if isinstance(c, tuple) else c for c in orders["Column"]}
    assert traded_cols == {"A", "B"}

    # 2. BarFeed has an 'open' column whose values are all NaN -> fail closed with ValueError
    orig_feed = slot.bar_feed

    class _AllNanOpenFeed:
        def read_signal_bars(self, instruments, frequency, *, start_utc, end_utc, as_of_utc):
            return orig_feed.read_signal_bars(
                instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
            )

        def read_execution_bars(self, instruments, frequency, *, start_utc, end_utc):
            df = orig_feed.read_execution_bars(
                instruments, frequency, start_utc=start_utc, end_utc=end_utc
            ).copy()
            df["open"] = np.nan
            return df

    all_nan_slot = replace(slot, bar_feed=_AllNanOpenFeed())
    for supplied_open in (None, nan_intersection_open):
        runner_fail = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
            slot=all_nan_slot,
            frequency="1h",
        )
        with pytest.raises(ValueError, match="entry_price_field='open'"):
            runner_fail.run_backtest(
                scores,
                sub_close,
                open_df=supplied_open,
                rebalance_dates=[t1.to_pydatetime()],
            )
        runner_fail.close()

    # 3. Partial missing open prices preserve skip-order semantics without raising
    def _mask_t1_a_open(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mask = out["instrument_id"].eq("A") & (pd.to_datetime(out["timestamp_utc"], utc=True) == t1)
        out.loc[mask, "open"] = np.nan
        return out

    partial_slot = replace(slot, bar_feed=type("_PartialOpenFeed", (), {
        "read_signal_bars": lambda _s, instruments, frequency, *, start_utc, end_utc, as_of_utc: orig_feed.read_signal_bars(
            instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
        ),
        "read_execution_bars": lambda _s, instruments, frequency, *, start_utc, end_utc: _mask_t1_a_open(
            orig_feed.read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)
        ),
    })())
    runner_partial = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=partial_slot,
        frequency="1h",
    )
    res_partial = runner_partial.run_backtest(
        scores,
        sub_close,
        open_df=nan_intersection_open,
        rebalance_dates=[t1.to_pydatetime(), t2.to_pydatetime()],
    )
    runner_partial.close()
    assert "error" not in res_partial, res_partial
    partial_orders = res_partial["portfolio"].orders.records_readable
    assert len(partial_orders) == 1
    assert pd.Timestamp(partial_orders["Timestamp"].iloc[0]).tz_convert("UTC") == t2
    assert float(partial_orders["Price"].iloc[0]) == pytest.approx(110.0)


def test_overlay_liquidity_cap_reallocation_caps_total_gross_including_blocked_holdings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    symbols = ("A", "C", "D")
    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path,
        start="2026-01-05",
        end="2026-01-05",
        allow_short=False,
        symbols=symbols,
        open_price=100.0,
        close_price=100.0,
    )
    sub_close = close_df.iloc[:4]
    sub_open = open_df.iloc[:4]
    t1 = sub_close.index[1]
    t2 = sub_close.index[2]
    scores = pd.DataFrame({"A": 2.0, "C": 2.0, "D": 1.0}, index=sub_close.index)

    def _make_runner() -> VectorBTBacktestRunner:
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": 2,
                    "holding_period": 1,
                    "score_source": "input_factor",
                    "cost_rate": 0.0,
                    "init_cash": 100_000.0,
                    "enable_tday_filter": False,
                    "banding": {"enabled": True},
                    "liquidity_cap": {
                        "enabled": True,
                        "max_participation": 0.30,
                        "adv_lookback_days": 5,
                    },
                }
            },
            slot=slot,
            frequency="1h",
        )
        # Round 1 (t1, pool A/D): A cap=30k, D cap=20k -> holds A=30k, D=20k
        # Round 2 (t2, pool A/C): A cap=30k, C cap=30k -> initial_gross=60k
        def _adv(stocks, as_of, lookback):
            ts = pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of, tz="UTC")
            if ts < t1:
                return {"A": 100_000.0, "C": 100_000.0, "D": 20_000.0 / 0.30}
            return {"A": 100_000.0, "C": 100_000.0, "D": 100_000.0}

        monkeypatch.setattr(runner, "_average_daily_value", _adv)
        return runner

    # 1. D sell blocked at t2: flexible budget = min(100k, 60k) - 20k = 40k -> A=20k, C=20k, D=20k (total 60k)
    runner_1 = _make_runner()

    def _mask_d_sell(date_str: str, stocks: list[str], side: str = "both", require_non_st: bool = True) -> list[str]:
        ts = pd.Timestamp(date_str).tz_convert("UTC") if pd.Timestamp(date_str).tzinfo else pd.Timestamp(date_str, tz="UTC")
        if ts == t2 and side == "sell":
            return [s for s in stocks if s != "D"]
        return list(stocks)

    monkeypatch.setattr(runner_1, "_filter_tradeable_stocks", _mask_d_sell)
    res_1 = runner_1.run_backtest(
        scores,
        sub_close,
        open_df=sub_open,
        dynamic_pools={t1.isoformat(): ["A", "D"], t2.isoformat(): ["A", "C"]},
        rebalance_dates=[t1.to_pydatetime(), t2.to_pydatetime()],
    )
    runner_1.close()
    assert "error" not in res_1, res_1
    tw_1 = runner_1._last_target_weights
    assert tw_1.loc[t1, "A"] == pytest.approx(30_000.0)
    assert tw_1.loc[t1, "D"] == pytest.approx(20_000.0)
    assert tw_1.loc[t2, "A"] == pytest.approx(20_000.0)
    assert tw_1.loc[t2, "C"] == pytest.approx(20_000.0)
    assert pd.isna(tw_1.loc[t2, "D"])
    av_1 = res_1["portfolio"].asset_value(group_by=False)
    assert np.asarray(av_1.loc[t2, "A"]).item() == pytest.approx(20_000.0)
    assert np.asarray(av_1.loc[t2, "C"]).item() == pytest.approx(20_000.0)
    assert np.asarray(av_1.loc[t2, "D"]).item() == pytest.approx(20_000.0)
    assert float(av_1.loc[t2].abs().sum()) == pytest.approx(60_000.0)

    # 2. If A's scale-down sell (30k -> 20k) is also sell-blocked at t2, iterative re-check locks A=30k too,
    #    leaving 60k - 20k - 30k = 10k for C (total 60k, respecting ADV caps & directional masks).
    runner_2 = _make_runner()

    def _mask_ad_sell(date_str: str, stocks: list[str], side: str = "both", require_non_st: bool = True) -> list[str]:
        ts = pd.Timestamp(date_str).tz_convert("UTC") if pd.Timestamp(date_str).tzinfo else pd.Timestamp(date_str, tz="UTC")
        if ts == t2 and side == "sell":
            return [s for s in stocks if s not in {"A", "D"}]
        return list(stocks)

    monkeypatch.setattr(runner_2, "_filter_tradeable_stocks", _mask_ad_sell)
    res_2 = runner_2.run_backtest(
        scores,
        sub_close,
        open_df=sub_open,
        dynamic_pools={t1.isoformat(): ["A", "D"], t2.isoformat(): ["A", "C"]},
        rebalance_dates=[t1.to_pydatetime(), t2.to_pydatetime()],
    )
    runner_2.close()
    assert "error" not in res_2, res_2
    tw_2 = runner_2._last_target_weights
    assert pd.isna(tw_2.loc[t2, "A"])
    assert pd.isna(tw_2.loc[t2, "D"])
    assert tw_2.loc[t2, "C"] == pytest.approx(10_000.0)
    av_2 = res_2["portfolio"].asset_value(group_by=False)
    assert np.asarray(av_2.loc[t2, "A"]).item() == pytest.approx(30_000.0)
    assert np.asarray(av_2.loc[t2, "C"]).item() == pytest.approx(10_000.0)
    assert np.asarray(av_2.loc[t2, "D"]).item() == pytest.approx(20_000.0)
    assert float(av_2.loc[t2].abs().sum()) == pytest.approx(60_000.0)


def test_default_ashare_all_nan_open_df_preserves_panel_and_skips_orders(tmp_path: Path) -> None:
    db_path = tmp_path / "ashare_all_nan_open.duckdb"
    _, _, close_df, _, dynamic_pools, bundle_path = _create_ashare_parity_db(db_path)
    dummy_factor = pd.DataFrame(1.0, index=close_df.index, columns=close_df.columns)
    cfg = {
        "top_k_strategy": {"top_k": 2, "holding_period": 1, "enable_tday_filter": True},
        "calendar_filters": {"skip_months": []},
    }
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        slot = build_jq_ashare_local(conn, start="2023-01-03", end="2023-03-31")
        assert slot.microstructure.execution_profile.entry_price_field == "close"
        for bad_val in (np.nan, np.inf, 0.0):
            bad_open = pd.DataFrame(bad_val, index=close_df.index, columns=close_df.columns)
            runner = VectorBTBacktestRunner(
                cfg, slot=slot, bundle_path=bundle_path, factor_db_path=str(db_path)
            )
            res = runner.run_backtest(
                dummy_factor, close_df, open_df=bad_open, dynamic_pools=dynamic_pools
            )
            runner.close()
            assert res == {"error": "All weights NaN"}
    finally:
        conn.close()


def test_partial_valid_open_df_treats_non_finite_and_non_positive_as_missing(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    slot, close_df, open_df = _make_24x7_hourly_slot(
        tmp_path,
        start="2026-01-05",
        end="2026-01-05",
        entry_price_field="open",
        allow_short=False,
        open_price=110.0,
        close_price=100.0,
    )
    sub_close = close_df.iloc[:4]
    sub_open = open_df.iloc[:4]
    t1 = sub_close.index[1]
    t2 = sub_close.index[2]
    scores = pd.DataFrame(
        {"A": [2.0, 1.0, 1.0, 1.0], "B": [1.0, 2.0, 2.0, 2.0]},
        index=sub_close.index,
    )
    cfg = {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}}

    for bad_val in (np.inf, -np.inf, 0.0, -5.0):
        bad_open = sub_open.copy()
        bad_open.loc[t1, "A"] = bad_val
        runner = VectorBTBacktestRunner(cfg, slot=slot, frequency="1h")
        res = runner.run_backtest(
            scores,
            sub_close,
            open_df=bad_open,
            rebalance_dates=[t1.to_pydatetime(), t2.to_pydatetime()],
        )
        runner.close()
        assert "error" not in res, res
        orders = res["portfolio"].orders.records_readable
        assert len(orders) == 1
        col = orders["Column"].iloc[0]
        assert (col[-1] if isinstance(col, tuple) else col) == "B"
        assert pd.Timestamp(orders["Timestamp"].iloc[0]).tz_convert("UTC") == t2
        assert float(orders["Price"].iloc[0]) == pytest.approx(110.0)

    # BarFeed itself returning +inf on (t1, "A") must also skip (t1, "A") and buy B at t2 at 110.0.
    orig_feed = slot.bar_feed

    def _inf_t1_a_open(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mask = out["instrument_id"].eq("A") & (pd.to_datetime(out["timestamp_utc"], utc=True) == t1)
        out.loc[mask, "open"] = np.inf
        return out

    inf_feed_slot = replace(
        slot,
        bar_feed=type(
            "_InfOpenFeed",
            (),
            {
                "read_signal_bars": lambda _s, instruments, frequency, *, start_utc, end_utc, as_of_utc: orig_feed.read_signal_bars(
                    instruments, frequency, start_utc=start_utc, end_utc=end_utc, as_of_utc=as_of_utc
                ),
                "read_execution_bars": lambda _s, instruments, frequency, *, start_utc, end_utc: _inf_t1_a_open(
                    orig_feed.read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)
                ),
            },
        )(),
    )
    runner_feed = VectorBTBacktestRunner(cfg, slot=inf_feed_slot, frequency="1h")
    res_feed = runner_feed.run_backtest(
        scores,
        sub_close,
        rebalance_dates=[t1.to_pydatetime(), t2.to_pydatetime()],
    )
    runner_feed.close()
    assert "error" not in res_feed, res_feed
    feed_orders = res_feed["portfolio"].orders.records_readable
    assert len(feed_orders) == 1
    feed_col = feed_orders["Column"].iloc[0]
    assert (feed_col[-1] if isinstance(feed_col, tuple) else feed_col) == "B"
    assert pd.Timestamp(feed_orders["Timestamp"].iloc[0]).tz_convert("UTC") == t2
    assert float(feed_orders["Price"].iloc[0]) == pytest.approx(110.0)

    # All-+inf open_df on an explicit entry_price_field='open' slot has no usable open prices -> re-resolves from BarFeed (110.0).
    all_inf_open = pd.DataFrame(np.inf, index=sub_close.index, columns=sub_close.columns)
    runner_fallback = VectorBTBacktestRunner(cfg, slot=slot, frequency="1h")
    res_fallback = runner_fallback.run_backtest(
        scores,
        sub_close,
        open_df=all_inf_open,
        rebalance_dates=[t1.to_pydatetime(), t2.to_pydatetime()],
    )
    runner_fallback.close()
    assert "error" not in res_fallback, res_fallback
    fb_orders = res_fallback["portfolio"].orders.records_readable
    assert len(fb_orders) == 3
    assert all(float(p) == pytest.approx(110.0) for p in fb_orders["Price"])
