"""Stage 5-3 tests: StrategyBundle contract, Harness CLI all-market runs, and master parity gate."""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace

import duckdb

from market_db_samples import declare_futures_bars, declare_market_db
import numpy as np
import pandas as pd
import pytest

from data_pipeline.loaders.market_bar_importer import import_market_bars
from factor_synthesis import (
    STRATEGY_BUNDLE_SCHEMA_VERSION,
    BundleArtifactRef,
    StrategyBundle,
    StrategyBundleError,
    StrategyFactorSpec,
    load_strategy_bundle,
    verify_strategy_bundle,
    write_strategy_bundle,
)
from factor_synthesis.core.factor_data_loader import load_factor_data
from factor_synthesis.core.factor_ops import (
    apply_feature_preprocessing,
    calculate_forward_returns,
)
from factor_synthesis.pipeline.lasso_method import run_lasso_synthesis
from factor_synthesis.pipeline.lightgbm_method import run_lightgbm_synthesis
from factor_synthesis.pipeline.weighted_sum_method import run_weighted_sum_synthesis
from factor_synthesis.strategy_bundle import sha256_file
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from factor_synthesis.vectorbt_adapter.quality_filter import apply_pit_quality_filter
from slots import build_slot_from_config
from slots.declarative import build_declarative_slot
from slots.jq_ashare_local import build_jq_ashare_local
from tools.agent_harness import cli as harness_cli
from tools.agent_harness.run_result import config_sha256, read_execution_result
from utils.market_db.identity import bind_market_db_fingerprint, market_db_identity, slot_market_db_identity
from utils.market_factor import (
    MarketFactorValueStore,
    build_identity_record,
    compute_factor_identity_hash,
)
from utils.session_calendar import SessionCalendar
from utils.slots import ExecutionProfile, ProviderConfig, SlotConfig
from utils.time_axis import Frequency

ROOT = Path(__file__).resolve().parents[1]
BASELINE_MASTER = "680e50b5f79c37fd65f52ac40777488a76dbcce4"
FACTORS = ["alpha", "beta"]
START, END = "2023-01-03", "2023-03-31"
TRAIN_START, TRAIN_END = "2023-01-04", "2023-02-24"
STOCKS = [f"00000{i}.XSHE" for i in range(1, 9)]


def _save_scheme_b_factor(
    store_root: Path,
    *,
    slot_id: str,
    slot_config_hash: str,
    calendar: SessionCalendar,
    factor_id: str,
    frequency: str,
    frame: pd.DataFrame,
    seed: int = 1,
    market_db: Mapping[str, str] | None = None,
) -> str:
    digest, payload = compute_factor_identity_hash(
        factor_id=factor_id,
        factor_class="tests.SyntheticFactor",
        source_sha256="b" * 64,
        params={"seed": seed},
        frequency=frequency,
        session_view="ETH",
        analysis_mode="cross_sectional",
        bucket_strategy="session" if frequency == "1d" else "segment_anchored",
        calendar_hash=calendar.calendar_hash,
        slot_id=slot_id,
        slot_config_hash=slot_config_hash,
        adjustment_policy="none",
        slot_data_fingerprint=bind_market_db_fingerprint({"fixture": "stage5-3", "seed": seed}, market_db or {}),
        code_version="synthetic-v1",
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
            source_build_id=f"build-{seed}",
            calendar_hash=calendar.calendar_hash,
            code_version="synthetic-v1",
            created_at=pd.Timestamp("2026-02-01T00:00:00Z"),
            batch_id=f"batch-{factor_id}-{seed}",
        ),
        identity_record=record,
        calendar=calendar,
    )
    return digest


def _make_sample_strategy_bundle(tmp_path: Path) -> tuple[StrategyBundle, dict[str, Path]]:
    model_file = tmp_path / "src_weights.json"
    model_file.write_text(
        json.dumps(
            {
                "index": {
                    "models": [
                        {
                            "train_end": "1900-01-01",
                            "weights": {"alpha": 0.7, "beta": -0.3},
                            "feature_names": ["alpha", "beta"],
                        }
                    ]
                },
                "metadata": {"method": "weighted_sum"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    returns_file = tmp_path / "src_returns.csv"
    returns_file.write_text("date,net_return\n2023-01-04,0.01\n", encoding="utf-8")
    nav_file = tmp_path / "src_nav.csv"
    nav_file.write_text("date,nav\n2023-01-04,1.01\n", encoding="utf-8")
    exec_file = tmp_path / "src_exec.json"
    exec_file.write_text(
        json.dumps(
            {
                "schema_version": "execution_result.v1",
                "command": "factor-synthesis",
                "run_id": "run-1",
                "execution_id": "exec-1",
                "config_sha256": "c" * 64,
                "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
                "metrics": {},
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    bundle = StrategyBundle(
        strategy_id="demo_strategy",
        slot_id="jq_ashare_local",
        slot_config_hash="1" * 64,
        calendar_hash="2" * 64,
        slot_data_fingerprint={"snapshot": "v1"},
        frequency="1d",
        factors=(
            StrategyFactorSpec(
                factor_id="alpha",
                factor_identity_hash="3" * 64,
                frequency="1d",
                preprocessing="zscore",
                aux_features_required=("factor_data.natural_log_of_market_cap",),
                source_module="factors.alpha",
            ),
        ),
        synthesis_method="weighted_sum",
        model_artifacts=(
            BundleArtifactRef(
                path="models/weights.json",
                sha256=sha256_file(model_file),
                artifact_type="model_bundle",
                size_bytes=model_file.stat().st_size,
            ),
        ),
        preprocessing_params={"preprocess": {"alpha": "zscore"}},
        random_seed=42,
        rebalance_config={"rebalance_freq": "weekly", "top_k": 5},
        execution_profile=asdict(
            ExecutionProfile(
                "close", "close", 0, 0, "decimal", "endpoints_only", "cross_session",
                allow_short=False, commission_bps=3.0, sell_tax_bps=10.0, slippage_bps=0.0,
                benchmark_instrument="000300.XSHG",
            )
        ),
        universe_filter={"use_dynamic_pools": False},
        allow_short=False,
        benchmark_instrument="000300.XSHG",
        performance_summary={
            "annual_return": 0.15,
            "sharpe_ratio": 1.4,
            "max_drawdown": -0.08,
            "annual_turnover": 2.1,
        },
        backtest_returns=BundleArtifactRef(
            path="backtest/returns.csv",
            sha256=sha256_file(returns_file),
            artifact_type="backtest_returns",
            size_bytes=returns_file.stat().st_size,
        ),
        backtest_nav=BundleArtifactRef(
            path="backtest/nav.csv",
            sha256=sha256_file(nav_file),
            artifact_type="backtest_nav",
            size_bytes=nav_file.stat().st_size,
        ),
        execution_result_ref="provenance/execution_result.json",
        execution_result_sha256=sha256_file(exec_file),
    )
    sources = {
        "models/weights.json": model_file,
        "backtest/returns.csv": returns_file,
        "backtest/nav.csv": nav_file,
        "provenance/execution_result.json": exec_file,
    }
    return bundle, sources


def test_strategy_bundle_contract_immutability_and_fail_closed_verification(tmp_path: Path) -> None:
    bundle, sources = _make_sample_strategy_bundle(tmp_path)
    assert bundle.schema_version == STRATEGY_BUNDLE_SCHEMA_VERSION
    assert len(bundle.bundle_hash) == 64

    with pytest.raises((FrozenInstanceError, TypeError)):
        bundle.slot_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        bundle.execution_profile["allow_short"] = True  # type: ignore[index]

    bundle_dir = tmp_path / "bundle_out"
    written = write_strategy_bundle(bundle, bundle_dir, source_files=sources)
    assert written == bundle_dir.resolve()
    verified = verify_strategy_bundle(bundle_dir)
    loaded = load_strategy_bundle(bundle_dir)
    assert loaded.bundle_hash == verified.bundle_hash == bundle.bundle_hash

    # Overwriting existing immutable bundle is rejected.
    with pytest.raises(StrategyBundleError, match="already exists"):
        write_strategy_bundle(bundle, bundle_dir, source_files=sources)

    # Tampering with a model file fails SHA-256 verification.
    (bundle_dir / "models/weights.json").write_text('{"alpha": 0.99}\n', encoding="utf-8")
    with pytest.raises(StrategyBundleError, match="sha256 mismatch"):
        verify_strategy_bundle(bundle_dir)

    # Restore model file and tamper with manifest field -> bundle_hash mismatch.
    (bundle_dir / "models/weights.json").write_bytes(sources["models/weights.json"].read_bytes())
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["random_seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StrategyBundleError, match="bundle_hash mismatch"):
        verify_strategy_bundle(bundle_dir)

    # Path traversal in relative path is rejected at construction.
    with pytest.raises(StrategyBundleError, match="escape"):
        BundleArtifactRef(path="../escape.json", sha256="a" * 64, artifact_type="model_bundle")

    # Symlink escaping bundle directory is rejected.
    escape_dir = tmp_path / "symlink_bundle"
    src2 = tmp_path / "src2"
    src2.mkdir()
    escape_bundle, escape_sources = _make_sample_strategy_bundle(src2)
    write_strategy_bundle(escape_bundle, escape_dir, source_files=escape_sources)
    outside = tmp_path / "outside.json"
    outside.write_text('{"alpha": 0.7, "beta": -0.3}\n', encoding="utf-8")
    target = escape_dir / "models/weights.json"
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(StrategyBundleError, match="escapes"):
        verify_strategy_bundle(escape_dir)


def _create_ashare_parity_fixture(tmp_path: Path) -> dict:
    db_path = tmp_path / "ashare_parity.duckdb"
    store_root = tmp_path / "ashare_store"
    calendar = SessionCalendar.from_pmc(START, END)
    sessions = calendar.sessions()
    rng = np.random.default_rng(20261123)

    close_mat = 25.0 * np.exp(rng.normal(0.0004, 0.015, (len(sessions), len(STOCKS))).cumsum(axis=0))
    open_mat = close_mat * (1.0 + rng.normal(0.0, 0.0025, close_mat.shape))
    close_df = pd.DataFrame(close_mat, index=sessions, columns=STOCKS)
    open_df = pd.DataFrame(open_mat, index=sessions, columns=STOCKS)

    raw_index = pd.MultiIndex.from_product([sessions, STOCKS], names=["date", "stock_code"])
    raw_factors = pd.DataFrame(rng.normal(size=(len(raw_index), 2)), index=raw_index, columns=FACTORS)
    # Induce mild cross-sectional signal correlated with next-day returns.
    fwd = close_df.pct_change().shift(-1).stack().reindex(raw_index).fillna(0.0)
    raw_factors["alpha"] = raw_factors["alpha"] + 15.0 * fwd.to_numpy()
    raw_factors["beta"] = raw_factors["beta"] - 10.0 * fwd.to_numpy()

    log_caps = pd.Series(
        [20.5 + 0.25 * (j % len(STOCKS)) + 0.01 * i for i in range(len(sessions)) for j in range(len(STOCKS))],
        index=raw_index,
        name="natural_log_of_market_cap",
    )
    industries = pd.Series(
        ["801160" if j % 2 == 0 else "801170" for _ in sessions for j in range(len(STOCKS))],
        index=raw_index,
        name="sw_l1_code",
    )

    precalculated = raw_factors.copy()
    for _, group in raw_factors.groupby(level="date"):
        design = np.column_stack([
            np.ones(len(group)),
            log_caps.loc[group.index].to_numpy(),
            industries.loc[group.index].eq("801170").to_numpy(dtype=float),
        ])
        for col in FACTORS:
            y = group[col].to_numpy()
            precalculated.loc[group.index, col] = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]

    with duckdb.connect(str(db_path)) as conn:
        market_db = market_db_identity(declare_market_db(conn, 'cn_equity'))
        cal_cfg = json.loads((ROOT / "config/trading_calendar.json").read_text(encoding="utf-8"))
        all_days = pd.date_range(START, END)
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
            for j, stock in enumerate(STOCKS):
                o_px = float(open_mat[i, j])
                c_px = float(close_mat[i, j])
                paused = 1 if (i == 20 and j == 0) else 0
                hi_lim = o_px * 0.99 if (i == 25 and j == 1) else max(o_px, c_px) * 1.15
                lo_lim = min(o_px, c_px) * 0.85
                price_rows.append(
                    (stock, day.date(), o_px, max(o_px, c_px) * 1.01, min(o_px, c_px) * 0.99, c_px,
                     1_000_000.0, o_px * 1_000_000.0, 1.0, hi_lim, lo_lim, paused)
                )
        conn.executemany(
            "INSERT INTO price_data_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            price_rows,
        )
        conn.execute(
            "CREATE TABLE factor_data (stock_code VARCHAR, date DATE, is_st VARCHAR, natural_log_of_market_cap DOUBLE)"
        )
        fdata_rows = [
            (stock, day.date(), "true" if (j == 7 and i >= 22) else "false", float(log_caps.loc[(day, stock)]))
            for i, day in enumerate(sessions)
            for j, stock in enumerate(STOCKS)
        ]
        conn.executemany("INSERT INTO factor_data VALUES (?, ?, ?, ?)", fdata_rows)

        conn.execute("CREATE TABLE industry_membership (stock_code VARCHAR, date DATE, sw_l1_name VARCHAR, sw_l1_code VARCHAR)")
        ind_rows = [
            (stock, day.date(), f"Ind_{industries.loc[(day, stock)]}", str(industries.loc[(day, stock)]))
            for day in sessions
            for stock in STOCKS
        ]
        conn.executemany("INSERT INTO industry_membership VALUES (?, ?, ?, ?)", ind_rows)

        legacy = raw_factors.rename_axis(columns="factor_name").stack().rename("factor_value").reset_index()
        conn.execute("CREATE TABLE factor_values AS SELECT * FROM legacy")
        neu = precalculated.rename_axis(columns="raw_factor_name").stack().rename("factor_value").reset_index()
        conn.execute(
            "CREATE TABLE neutralized_factor_values AS SELECT *, raw_factor_name || '__neu' AS factor_name FROM neu"
        )

        # Quality filter PIT dividend + financial events; exclude stock index 5.
        conn.execute(
            "CREATE TABLE dividend_quality_financial_events ("
            "statement_type VARCHAR, stock_code VARCHAR, pub_date DATE, start_date DATE, end_date DATE, "
            "net_profit DOUBLE, net_operate_cash_flow DOUBLE, total_liability DOUBLE, total_assets DOUBLE)"
        )
        fin_rows = []
        for j, stock in enumerate(STOCKS):
            profit_scale = 100.0 + 15.0 * j
            for period in pd.period_range("2018Q1", "2022Q4", freq="Q"):
                pend = period.end_time.date()
                pub = (period.end_time + pd.Timedelta(days=30)).date()
                fin_rows.append(("income", stock, pub, period.start_time.date(), pend, profit_scale * period.quarter, 40.0 * period.quarter, None, None))
                fin_rows.append(("balance", stock, pub, period.start_time.date(), pend, None, None, 500.0, 1000.0 + 50.0 * j))
        conn.executemany("INSERT INTO dividend_quality_financial_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", fin_rows)

        conn.execute(
            "CREATE TABLE dividend_events ("
            "stock_code VARCHAR, report_date DATE, implementation_pub_date DATE, "
            "bonus_type VARCHAR, bonus_amount_rmb DOUBLE, bonus_ratio_rmb DOUBLE)"
        )
        div_rows = []
        for j, stock in enumerate(STOCKS):
            if j == 5:
                continue
            for year in (2019, 2020, 2021):
                div_rows.append((stock, f"{year}-12-31", f"{year + 1}-06-30", "年度分红", 0.02 + 0.005 * j, 1.0 + 0.1 * j))
        conn.executemany("INSERT INTO dividend_events VALUES (?, ?, ?, ?, ?, ?)", div_rows)

        conn.execute("CREATE TABLE valuation_daily (stock_code VARCHAR, date DATE, market_cap DOUBLE)")
        conn.executemany(
            "INSERT INTO valuation_daily VALUES (?, ?, ?)",
            [(stock, day.date(), 1000.0 + 100.0 * j) for day in sessions for j, stock in enumerate(STOCKS)],
        )

    # Save identical raw factors to Scheme B store under jq_ashare_local.
    grid = calendar.bar_grid("1d", START, END)
    rows = grid.merge(pd.DataFrame({"instrument_id": STOCKS}), how="cross")
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(rows.session_id), rows.instrument_id], names=["date", "stock_code"])
    identities = {}
    for fid in FACTORS:
        identities[fid] = _save_scheme_b_factor(
            store_root,
            slot_id="jq_ashare_local",
            slot_config_hash="0" * 64,
            calendar=calendar,
            factor_id=fid,
            frequency="1d",
            frame=rows.assign(factor_value=raw_factors[fid].reindex(keys).to_numpy()),
            market_db=market_db,
        )

    dynamic_pools = {
        sessions[d].strftime("%Y-%m-%d"): list(STOCKS)
        for d in range(2, len(sessions), 5)
    }
    return {
        "db_path": db_path,
        "store_root": store_root,
        "market_db": market_db,
        "sessions": sessions,
        "close_df": close_df,
        "open_df": open_df,
        "identities": identities,
        "dynamic_pools": dynamic_pools,
    }




def test_harness_factor_synthesis_cn_futures_end_to_end(tmp_path: Path) -> None:
    from utils.bars.futures_roll import build_roll_schedule
    from utils.time_axis import BarSchedule
    from utils.time_axis.futures import futures_calendar

    db_path = tmp_path / "futures.duckdb"
    store_root = tmp_path / "futures_store"
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
    slot_times = pd.concat(
        [schedule.for_session(day) for day in sessions], ignore_index=True
    )["start_utc"]
    contracts = ["RB2605.SHFE", "RB2610.SHFE"]
    rng = np.random.default_rng(77)

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

    roll_start = sessions[1].strftime("%Y-%m-%d")
    with duckdb.connect(str(db_path)) as conn:
        declare_futures_bars(conn, "SHFE")
        import_market_bars(
            conn,
            pd.concat(bars_list, ignore_index=True),
            meta,
            bucket_strategy="segment_anchored",
            legacy_compatibility=False,
        )
        build_roll_schedule(conn, product_roots=["RB"], start=roll_start, end=end)

    slot_cfg = {
        "schema_version": 1,
        "slot_id": "cn_futures",
        "calendar": {"provider": "cn_futures", "options": {"start": roll_start, "end": end}},
        "bar_feed": {"provider": "cn_futures", "options": {"product_root": "RB"}},
        "universe": {"provider": "cn_futures"},
        "aux_features": {"provider": "cn_futures"},
        "microstructure": {"provider": "cn_futures"},
        "execution_profile": {
            "entry_price_field": "open",
            "exit_price_field": "close",
            "min_entry_offset_bars": 1,
            "exit_adjustment": -1,
            "return_unit": "percent",
            "gap_rule": "require_complete_bars",
            "boundary_policy": "cross_session",
            "allow_short": True,
            "commission_bps": 2.0,
            "sell_tax_bps": 0.0,
            "slippage_bps": 1.0,
            "benchmark_instrument": None,
        },
    }
    slot_cfg_path = tmp_path / "cn_futures_slot.json"
    slot_cfg_path.write_text(json.dumps(slot_cfg), encoding="utf-8")

    with duckdb.connect(str(db_path), read_only=True) as conn:
        slot = build_slot_from_config(slot_cfg_path, connection=conn)
    grid = slot.calendar.bar_grid("1d", roll_start, end)
    factor_rows = grid.merge(pd.DataFrame({"instrument_id": contracts}), how="cross")
    _save_scheme_b_factor(
        store_root,
        slot_id="cn_futures",
        slot_config_hash=slot.slot_config_hash,
        calendar=slot.calendar,
        factor_id="fut_mom",
        frequency="1d",
        frame=factor_rows.assign(
            factor_value=np.tile([1.5, -1.5], len(grid)) + 0.1 * np.sin(np.arange(len(factor_rows)))
        ),
        market_db=slot_market_db_identity(slot),
    )
    _save_scheme_b_factor(
        store_root,
        slot_id="cn_futures",
        slot_config_hash=slot.slot_config_hash,
        calendar=slot.calendar,
        factor_id="fut_carry",
        frequency="1d",
        frame=factor_rows.assign(
            factor_value=np.tile([0.8, -0.8], len(grid)) + 0.1 * np.cos(np.arange(len(factor_rows)))
        ),
        market_db=slot_market_db_identity(slot),
    )

    synth_cfg_path = tmp_path / "futures_synthesis.json"
    synth_cfg_path.write_text(
        json.dumps(
            {
                "slot_db": str(db_path),
                "analysis_period": {"start_date": roll_start, "end_date": end},
                "factor_groups": {"fut": {"factors": ["fut_mom", "fut_carry"]}},
                "synthesis_methods": {
                    "ws_fut": {
                        "method": "weighted_sum",
                        "factor_group": "fut",
                        "weights": {"fut_mom": 0.6, "fut_carry": 0.4},
                        "preprocess": {"ALL": "raw_zscore"},
                    }
                },
                "performance_evaluation": {
                    "top_k_strategy": {"enabled": True, "top_k": 1, "holding_period": 1, "rebalance_freq": "daily"}
                },
                "output_settings": {"composite_factor_prefix": "FUT"},
                "cache_settings": {"use_parquet_cache": False},
            }
        ),
        encoding="utf-8",
    )

    harness_root = tmp_path / "harness"
    rc = harness_cli.main([
        "factor-synthesis",
        "--config", str(synth_cfg_path),
        "--slot-config", str(slot_cfg_path),
        "--slot-id", "cn_futures",
        "--factor-values-store", str(store_root),
        "--frequency", "1d",
        "--output-root", str(harness_root),
        "--run-id", "fut-e2e",
    ])
    assert rc == 0
    summary = json.loads((harness_root / "factor-synthesis" / "fut-e2e" / "summary.json").read_text(encoding="utf-8"))
    bundle = load_strategy_bundle(summary["metrics"]["strategy_bundle"]["bundle_dir"])
    assert bundle.slot_id == "cn_futures"
    assert bundle.frequency == "1d"
    assert bundle.allow_short is True
    assert bundle.execution_profile["entry_price_field"] == "open"
    assert [f.factor_id for f in bundle.factors] == ["fut_mom", "fut_carry"]
    assert np.isfinite(bundle.performance_summary["annual_return"])
    if os.environ.get("STAGE5_PRINT_PARITY_EVIDENCE") == "1":
        print(
            "STAGE5_EVIDENCE_FUTURES=" + json.dumps(
                {
                    "slot_id": bundle.slot_id,
                    "frequency": bundle.frequency,
                    "allow_short": bundle.allow_short,
                    "entry_price_field": bundle.execution_profile["entry_price_field"],
                    "return_unit": bundle.execution_profile["return_unit"],
                    "factors": [f.factor_id for f in bundle.factors],
                    "bundle_hash": bundle.bundle_hash,
                    "trade_count": int(summary["metrics"]["backtest_summary"]["trade_count"]),
                    "performance_summary": dict(bundle.performance_summary),
                },
                sort_keys=True,
                default=dict,
            ),
            flush=True,
        )


def test_harness_factor_synthesis_declarative_24x7_long_short_end_to_end(tmp_path: Path) -> None:
    bars_csv = tmp_path / "crypto_bars.csv"
    store_root = tmp_path / "crypto_store"
    hours = pd.date_range("2026-01-03 00:00:00Z", "2026-01-06 23:00:00Z", freq="1h")
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

    slot_cfg = {
        "schema_version": 1,
        "slot_id": "crypto_24x7",
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "crypto_utc",
                "start": "2026-01-03",
                "end": "2026-01-06",
                "custom_sessions": {
                    "timezone": "UTC",
                    "weekdays": [0, 1, 2, 3, 4, 5, 6],
                    "segments": [{"open": "00:00", "close": "00:00", "open_day_offset": 0, "close_day_offset": 1}],
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
            "exit_price_field": "open",
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
    slot_cfg_path = tmp_path / "crypto_slot.json"
    slot_cfg_path.write_text(json.dumps(slot_cfg), encoding="utf-8")
    slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)

    grid = slot.calendar.bar_grid("1h")
    factor_rows = grid.merge(pd.DataFrame({"instrument_id": symbols}), how="cross")
    _save_scheme_b_factor(
        store_root,
        slot_id="crypto_24x7",
        slot_config_hash=slot.slot_config_hash,
        calendar=slot.calendar,
        factor_id="crypto_mom",
        frequency="1h",
        frame=factor_rows.assign(factor_value=np.tile([3.0, 1.0, -1.0, -3.0], len(grid))),
    )
    _save_scheme_b_factor(
        store_root,
        slot_id="crypto_24x7",
        slot_config_hash=slot.slot_config_hash,
        calendar=slot.calendar,
        factor_id="crypto_rev",
        frequency="1h",
        frame=factor_rows.assign(factor_value=np.tile([1.5, 0.5, -0.5, -1.5], len(grid))),
    )

    synth_cfg_path = tmp_path / "crypto_synthesis.json"
    synth_cfg_path.write_text(
        json.dumps(
            {
                "analysis_period": {"start_date": "2026-01-03", "end_date": "2026-01-06"},
                "factor_groups": {"crypto": {"factors": ["crypto_mom", "crypto_rev"]}},
                "synthesis_methods": {
                    "ws_crypto": {
                        "method": "weighted_sum",
                        "factor_group": "crypto",
                        "weights": {"crypto_mom": 0.7, "crypto_rev": 0.3},
                        "preprocess": {"ALL": "raw_zscore"},
                    }
                },
                "performance_evaluation": {
                    "top_k_strategy": {"enabled": True, "top_k": 1, "holding_period": 6, "rebalance_freq": "bar"}
                },
                "output_settings": {"composite_factor_prefix": "CRYPTO"},
                "cache_settings": {"use_parquet_cache": False},
            }
        ),
        encoding="utf-8",
    )

    harness_root = tmp_path / "harness"
    rc = harness_cli.main([
        "factor-synthesis",
        "--config", str(synth_cfg_path),
        "--slot-config", str(slot_cfg_path),
        "--slot-id", "crypto_24x7",
        "--factor-values-store", str(store_root),
        "--frequency", "1h",
        "--output-root", str(harness_root),
        "--run-id", "crypto-e2e",
    ])
    assert rc == 0
    summary = json.loads((harness_root / "factor-synthesis" / "crypto-e2e" / "summary.json").read_text(encoding="utf-8"))
    bundle = load_strategy_bundle(summary["metrics"]["strategy_bundle"]["bundle_dir"])
    assert bundle.slot_id == "crypto_24x7"
    assert bundle.frequency == "60min"
    assert bundle.allow_short is True
    assert bundle.benchmark_instrument is None
    assert bundle.execution_profile["allow_short"] is True
    assert bundle.execution_profile["commission_bps"] == 5.0
    assert bundle.execution_profile["slippage_bps"] == 2.0
    assert np.isfinite(bundle.performance_summary["annual_return"])
    assert summary["metrics"]["backtest_summary"]["trade_count"] > 0
    if os.environ.get("STAGE5_PRINT_PARITY_EVIDENCE") == "1":
        print(
            "STAGE5_EVIDENCE_CRYPTO=" + json.dumps(
                {
                    "slot_id": bundle.slot_id,
                    "frequency": bundle.frequency,
                    "allow_short": bundle.allow_short,
                    "commission_bps": bundle.execution_profile["commission_bps"],
                    "slippage_bps": bundle.execution_profile["slippage_bps"],
                    "factors": [f.factor_id for f in bundle.factors],
                    "bundle_hash": bundle.bundle_hash,
                    "trade_count": int(summary["metrics"]["backtest_summary"]["trade_count"]),
                    "performance_summary": dict(bundle.performance_summary),
                },
                sort_keys=True,
                default=dict,
            ),
            flush=True,
        )




def test_multi_method_multi_top_pct_and_reuse_strategy_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point 1: bind backtests/bundles per (synthesis_name, trial_id) across multi-method, multi-top_pct, and synthesis_reuse."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    fixture = _create_ashare_parity_fixture(tmp_path)
    db_path: Path = fixture["db_path"]
    store_root: Path = fixture["store_root"]
    local_out_db = tmp_path / "multi_results.duckdb"
    reuse_weights_dir = tmp_path / "shared_weights"

    cfg_path = tmp_path / "multi_method_cfg.json"
    cfg = {
        "database": {
            "factor_db": {"duckdb_path": str(db_path), "factor_values_table": "factor_values"},
            "price_db": {"duckdb_path": str(db_path), "price_table": "price_data_daily", "factor_data_table": "factor_data"},
            "local_db": {"duckdb_path": str(local_out_db), "results_table": "factor_analysis_results"},
            "price_table": "price_data_daily",
        },
        "factor_values_store": str(store_root),
        "analysis_period": {"start_date": "2023-02-06", "end_date": "2023-03-24"},
        "rolling_window": {"train_window_days": 20, "predict_window_days": 10, "step_size_days": 10, "min_train_samples": 10},
        "factor_groups": {"all": {"factors": FACTORS}},
        "synthesis_methods": {
            "lgb_first": {
                "method": "lightgbm",
                "factor_group": "all",
                "target_period": 1,
                "target_cs_demean": False,
                "preprocess": {"alpha": "zscore", "beta": "zscore"},
                "lgb_n_estimators": 8,
                "lgb_min_child_samples": 4,
                "lgb_n_jobs": 1,
                "lgb_random_state": 42,
            },
            "lasso_second": {
                "method": "lasso",
                "factor_group": "all",
                "target_period": 1,
                "target_cs_demean": False,
                "preprocess": {"alpha": "zscore", "beta": "zscore"},
                "lasso_cv_folds": 2,
                "random_state": 42,
                "blend_with_original_ratio": 0.0,
            },
        },
        "performance_evaluation": {
            "skip_alphalens_analysis": True,
            "top_k_strategy": {
                "enabled": True,
                "top_pct": [0.25, 0.50],
                "holding_period": 1,
                "enable_tday_filter": True,
            },
        },
        "output_settings": {
            "composite_factor_prefix": "MULTI",
            "weights_output_dir": str(reuse_weights_dir),
        },
        "cache_settings": {"use_parquet_cache": False},
        "logging": {"level": "INFO"},
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    harness_root = tmp_path / "harness"
    rc = harness_cli.main([
        "factor-synthesis",
        "--config", str(cfg_path),
        "--output-root", str(harness_root),
        "--run-id", "multi-run",
    ])
    assert rc == 0
    summary = json.loads((harness_root / "factor-synthesis" / "multi-run" / "summary.json").read_text(encoding="utf-8"))
    bundles_meta = summary["metrics"]["strategy_bundles"]
    assert len(bundles_meta) == 4
    assert len({b["bundle_dir"] for b in bundles_meta}) == 4
    assert len({b["trial_id"] for b in bundles_meta}) == 4

    lgb_bundles = [b for b in bundles_meta if b["synthesis_name"] == "lgb_first"]
    lasso_bundles = [b for b in bundles_meta if b["synthesis_name"] == "lasso_second"]
    assert len(lgb_bundles) == 2 and len(lasso_bundles) == 2

    for meta in bundles_meta:
        loaded = verify_strategy_bundle(meta["bundle_dir"])
        if meta["synthesis_name"] == "lgb_first":
            assert loaded.synthesis_method == "lightgbm"
            assert any("lgb_model_bundle_lgb_first.json" in ref.path for ref in loaded.model_artifacts)
            assert not any("lasso" in ref.path for ref in loaded.model_artifacts)
        else:
            assert loaded.synthesis_method == "lasso"
            assert any("lasso_model_bundle_lasso_second.json" in ref.path for ref in loaded.model_artifacts)
            assert not any("lgb" in ref.path for ref in loaded.model_artifacts)
        assert loaded.rebalance_config["top_pct"] == meta["top_pct"]

    # Two top_pct candidates under the same method must preserve distinct returns/configs.
    l0 = verify_strategy_bundle(lasso_bundles[0]["bundle_dir"])
    l1 = verify_strategy_bundle(lasso_bundles[1]["bundle_dir"])
    assert l0.rebalance_config["top_pct"] == 0.25
    assert l1.rebalance_config["top_pct"] == 0.50
    assert l0.bundle_hash != l1.bundle_hash
    r0 = pd.read_csv(Path(lasso_bundles[0]["bundle_dir"]) / l0.backtest_returns.path)["net_return"].to_numpy()
    r1 = pd.read_csv(Path(lasso_bundles[1]["bundle_dir"]) / l1.backtest_returns.path)["net_return"].to_numpy()
    assert not np.allclose(r0, r1)

    # All 4 bundles are registered & verified in execution_result.v1.
    exec_art = next(a for a in summary["artifacts"] if a["artifact_type"] == "execution_result")
    bound_payload = read_execution_result(
        exec_art["path"],
        command="factor-synthesis",
        run_id="multi-run",
        execution_id=Path(exec_art["path"]).parent.name,
        source_sha256=config_sha256(cfg_path),
    )
    sb_artifacts = [a for a in bound_payload["artifacts"] if a["artifact_type"] == "strategy_bundle"]
    assert len(sb_artifacts) == 4

    # synthesis_reuse replay on single LASSO method must still emit a verified StrategyBundle.
    attempt_dir = Path(exec_art["path"]).parent
    reuse_source = attempt_dir / "outputs"
    if not (reuse_source / "lasso_model_bundle_lasso_second.json").is_file():
        reuse_source = reuse_weights_dir
    reuse_cfg_path = tmp_path / "reuse_cfg.json"
    reuse_cfg = {
        **cfg,
        "synthesis_methods": {"lasso_second": cfg["synthesis_methods"]["lasso_second"]},
        "synthesis_reuse": {"enabled": True, "source_dir": str(reuse_source)},
        "performance_evaluation": {
            "skip_alphalens_analysis": True,
            "top_k_strategy": {
                "enabled": True,
                "top_pct": 0.25,
                "holding_period": 1,
                "enable_tday_filter": True,
            },
        },
    }
    reuse_cfg_path.write_text(json.dumps(reuse_cfg), encoding="utf-8")
    rc_reuse = harness_cli.main([
        "factor-synthesis",
        "--config", str(reuse_cfg_path),
        "--output-root", str(harness_root),
        "--run-id", "lasso-reuse",
    ])
    assert rc_reuse == 0
    reuse_summary = json.loads((harness_root / "factor-synthesis" / "lasso-reuse" / "summary.json").read_text(encoding="utf-8"))
    reuse_bundle = verify_strategy_bundle(reuse_summary["metrics"]["strategy_bundle"]["bundle_dir"])
    assert reuse_bundle.strategy_id == "lasso_second"
    assert reuse_bundle.synthesis_method == "lasso"
    assert any("lasso_model_bundle_lasso_second.json" in ref.path for ref in reuse_bundle.model_artifacts)
    reuse_ret = pd.read_csv(Path(reuse_summary["metrics"]["strategy_bundle"]["bundle_dir"]) / reuse_bundle.backtest_returns.path)["net_return"].to_numpy()
    np.testing.assert_allclose(reuse_ret, r0, rtol=0, atol=1e-12)


def test_explicit_slot_entry_price_field_and_effective_filter_manifest(tmp_path: Path) -> None:
    """Point 2: explicit slot honors entry_price_field='close' vs 'open', and manifest records effective runner params."""
    bars_csv = tmp_path / "bars_100_110.csv"
    store_root = tmp_path / "store_100_110"
    dates = pd.date_range("2026-01-05", "2026-01-09", freq="D")
    symbols = ["A", "B", "C", "D"]
    rows = []
    # Open is constant 100.0; close moves from 110.0 (+10%/day compounding for A, flat for B/C/D).
    for i, day in enumerate(dates):
        for j, sym in enumerate(symbols):
            close_px = 110.0 * ((1.02) ** i if j == 0 else 1.0)
            rows.append({
                "symbol": sym,
                "time": f"{day.strftime('%Y-%m-%d')}T16:00:00Z",
                "open": 100.0,
                "high": max(100.0, close_px) + 1.0,
                "low": min(100.0, close_px) - 1.0,
                "close": close_px,
                "volume": 1000.0,
            })
    pd.DataFrame(rows).to_csv(bars_csv, index=False)

    def _run_with_entry_field(entry_field: str, quality_filter_cfg: dict | None, run_id: str):
        slot_cfg = {
            "schema_version": 1,
            "slot_id": f"slot_{entry_field}",
            "calendar": {
                "provider": "custom_sessions",
                "options": {
                    "calendar_id": "daily_utc",
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
                    "paths": ["bars_100_110.csv"],
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
                "entry_price_field": entry_field,
                "exit_price_field": entry_field,
                "min_entry_offset_bars": 0,
                "exit_adjustment": 0,
                "return_unit": "decimal",
                "gap_rule": "endpoints_only",
                "boundary_policy": "cross_session",
                "allow_short": False,
                "commission_bps": 0.0,
                "sell_tax_bps": 0.0,
                "slippage_bps": 0.0,
                "benchmark_instrument": None,
            },
        }
        slot_path = tmp_path / f"slot_{entry_field}.json"
        slot_path.write_text(json.dumps(slot_cfg), encoding="utf-8")
        slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
        grid = slot.calendar.bar_grid("1d")
        f_rows = grid.merge(pd.DataFrame({"instrument_id": symbols}), how="cross")
        _save_scheme_b_factor(
            store_root,
            slot_id=slot.slot_id,
            slot_config_hash=slot.slot_config_hash,
            calendar=slot.calendar,
            factor_id="sig",
            frequency="1d",
            frame=f_rows.assign(factor_value=np.tile([4.0, 3.0, 2.0, 1.0], len(grid))),
        )
        top_k_strat: dict = {
            "enabled": True,
            "top_k": 1,
            "holding_period": 1,
            "rebalance_freq": "daily",
            "enable_tday_filter": False,
        }
        if quality_filter_cfg is not None:
            top_k_strat["quality_filter"] = quality_filter_cfg
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
            "performance_evaluation": {"top_k_strategy": top_k_strat},
            "output_settings": {"composite_factor_prefix": "P"},
            "cache_settings": {"use_parquet_cache": False},
        }
        synth_path = tmp_path / f"synth_{entry_field}.json"
        synth_path.write_text(json.dumps(synth_cfg), encoding="utf-8")
        h_root = tmp_path / "h_out"
        rc = harness_cli.main([
            "factor-synthesis",
            "--config", str(synth_path),
            "--slot-config", str(slot_path),
            "--slot-id", slot.slot_id,
            "--factor-values-store", str(store_root),
            "--frequency", "1d",
            "--output-root", str(h_root),
            "--run-id", run_id,
        ])
        assert rc == 0
        summary = json.loads((h_root / "factor-synthesis" / run_id / "summary.json").read_text(encoding="utf-8"))
        return slot, load_strategy_bundle(summary["metrics"]["strategy_bundle"]["bundle_dir"]), Path(summary["metrics"]["strategy_bundle"]["bundle_dir"])

    close_slot, close_bundle, close_dir = _run_with_entry_field("close", None, "close-run")
    open_slot, open_bundle, open_dir = _run_with_entry_field("open", {"enabled": True, "min_qualified_count": 2}, "open-run")

    # Because factor signal uses T-1 alignment (shift(1)), day 0 has NaN signal and the first trade enters on day 1 (close=110*1.02=112.2, open=100.0).
    # On close slot: day 1 return = 0.0; on open slot: day 1 return = 112.2/100 - 1 = +0.122.
    close_ret = pd.read_csv(close_dir / close_bundle.backtest_returns.path)["net_return"].to_numpy()
    open_ret = pd.read_csv(open_dir / open_bundle.backtest_returns.path)["net_return"].to_numpy()
    assert close_ret[:2] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert open_ret[0] == pytest.approx(0.0, abs=1e-12)
    assert open_ret[1] == pytest.approx((110.0 * 1.02) / 100.0 - 1.0, abs=1e-12)
    assert close_bundle.execution_profile["entry_price_field"] == "close"
    assert open_bundle.execution_profile["entry_price_field"] == "open"
    assert close_bundle.universe_filter["quality_filter_enabled"] is False
    assert close_bundle.universe_filter["quality_filter"] == {}
    assert close_bundle.universe_filter["enable_tday_filter"] is False
    assert open_bundle.universe_filter["quality_filter_enabled"] is True
    assert open_bundle.universe_filter["quality_filter"]["min_qualified_count"] == 2

    # Direct VectorBTBacktestRunner must also honor explicit slot entry_price_field='close' even if open_df is supplied.
    close_wide = pd.DataFrame(110.0, index=dates, columns=symbols)
    open_wide = pd.DataFrame(100.0, index=dates, columns=symbols)
    score_wide = pd.DataFrame(np.tile([4.0, 3.0, 2.0, 1.0], (len(dates), 1)), index=dates, columns=symbols)
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "score_source": "input_factor", "cost_rate": 0.0}},
        slot=close_slot,
        frequency="1d",
    )
    res = runner.run_backtest(score_wide, close_wide, open_df=open_wide, rebalance_dates=[d.to_pydatetime() for d in dates])
    runner.close()
    assert "error" not in res
    assert runner.effective_entry_price_field == "close"
    assert all(float(p) == pytest.approx(110.0) for p in res["portfolio"].orders.records_readable["Price"])




def test_bundle_records_blend_fallback_factors_and_data_version_digests(tmp_path: Path) -> None:
    """Point 4: bundle records blend/fallback dependency factors and detects bar/shard changes."""
    fixture = _create_ashare_parity_fixture(tmp_path)
    db_path: Path = fixture["db_path"]
    store_root: Path = fixture["store_root"]

    # Add a third factor ONLY used in blend_weights (100% blend, outside training factor_group).
    cal = SessionCalendar.from_pmc(START, END)
    grid = cal.bar_grid("1d", START, END)
    rows = grid.merge(pd.DataFrame({"instrument_id": STOCKS}), how="cross")
    blend_id = _save_scheme_b_factor(
        store_root,
        slot_id="jq_ashare_local",
        slot_config_hash="0" * 64,
        calendar=cal,
        factor_id="Factor1_blend_parent",
        frequency="1d",
        frame=rows.assign(factor_value=np.linspace(-1.0, 1.0, len(rows))),
        market_db=fixture["market_db"],
    )

    local_out_db = tmp_path / "blend_local.duckdb"
    cfg_path = tmp_path / "blend_cfg.json"
    cfg = {
        "database": {
            "factor_db": {"duckdb_path": str(db_path), "factor_values_table": "factor_values"},
            "price_db": {"duckdb_path": str(db_path), "price_table": "price_data_daily", "factor_data_table": "factor_data"},
            "local_db": {"duckdb_path": str(local_out_db), "results_table": "factor_analysis_results"},
            "price_table": "price_data_daily",
        },
        "factor_values_store": str(store_root),
        "analysis_period": {"start_date": "2023-02-06", "end_date": "2023-03-24"},
        "rolling_window": {"train_window_days": 20, "predict_window_days": 10, "step_size_days": 10, "min_train_samples": 10},
        "factor_groups": {"train_only": {"factors": FACTORS}},
        "synthesis_methods": {
            "d9_blend": {
                "method": "lasso",
                "factor_group": "train_only",
                "target_period": 1,
                "target_cs_demean": False,
                "preprocess": {"alpha": "zscore", "beta": "zscore", "Factor1_blend_parent": "zscore"},
                "lasso_cv_folds": 2,
                "random_state": 42,
                "blend_weights": {"Factor1_blend_parent": 1.0},
                "blend_with_original_ratio": 1.0,
            }
        },
        "performance_evaluation": {
            "skip_alphalens_analysis": True,
            "top_k_strategy": {"enabled": True, "top_k": 2, "holding_period": 1, "enable_tday_filter": True},
        },
        "output_settings": {"composite_factor_prefix": "D9"},
        "cache_settings": {"use_parquet_cache": False},
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    h_root = tmp_path / "blend_harness"
    rc = harness_cli.main([
        "factor-synthesis",
        "--config", str(cfg_path),
        "--output-root", str(h_root),
        "--run-id", "blend-deps",
    ])
    assert rc == 0
    summary = json.loads((h_root / "factor-synthesis" / "blend-deps" / "summary.json").read_text(encoding="utf-8"))
    bundle = load_strategy_bundle(summary["metrics"]["strategy_bundle"]["bundle_dir"])
    factor_map = {spec.factor_id: spec for spec in bundle.factors}
    assert set(factor_map) == {"alpha", "beta", "Factor1_blend_parent"}
    assert factor_map["Factor1_blend_parent"].factor_identity_hash == blend_id
    assert factor_map["Factor1_blend_parent"].source_module == "tests.SyntheticFactor"
    assert bundle.slot_data_fingerprint["factor_shards"]

    # 24/7 OHLCV content change must change slot_data_fingerprint & bundle_hash even when slot_id/dates/identities stay constant.
    bars_csv = tmp_path / "crypto_fp.csv"
    crypto_store = tmp_path / "crypto_fp_store"
    hours = pd.date_range("2026-01-03 00:00:00Z", "2026-01-04 23:00:00Z", freq="1h")
    syms = ["BTC-USD", "ETH-USD"]
    bar_rows = []
    for j, sym in enumerate(syms):
        for k, ts in enumerate(hours):
            px = 100.0 * (j + 1) + 0.5 * k
            bar_rows.append({"symbol": sym, "time": (ts + pd.Timedelta(hours=1)).isoformat(), "open": px, "high": px + 1.0, "low": px - 1.0, "close": px + 0.2, "volume": 100.0})
    pd.DataFrame(bar_rows).to_csv(bars_csv, index=False)

    slot_cfg = {
        "schema_version": 1,
        "slot_id": "crypto_fp",
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "crypto_utc",
                "start": "2026-01-03",
                "end": "2026-01-04",
                "custom_sessions": {"timezone": "UTC", "weekdays": [0, 1, 2, 3, 4, 5, 6], "segments": [{"open": "00:00", "close": "00:00", "open_day_offset": 0, "close_day_offset": 1}]},
            },
        },
        "bar_feed": {
            "provider": "csv_files",
            "options": {
                "paths": ["crypto_fp.csv"], "frequency": "1h", "timestamp_anchor": "bar_end", "source_timezone": "UTC", "volume_mode": "interval", "data_version": "v1",
                "column_mapping": {"symbol": "instrument_id", "time": "timestamp", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"},
            },
        },
        "universe": {"provider": "static", "options": {"instruments": {s: {"asset_class": "crypto", "currency": "USD"} for s in syms}, "universes": {"spot": syms}}},
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {"entry_price_field": "open", "exit_price_field": "open", "min_entry_offset_bars": 1, "exit_adjustment": 0, "return_unit": "decimal", "gap_rule": "endpoints_only", "boundary_policy": "cross_session", "allow_short": True, "commission_bps": 0.0, "sell_tax_bps": 0.0, "slippage_bps": 0.0, "benchmark_instrument": None},
    }
    slot_path = tmp_path / "crypto_fp_slot.json"
    slot_path.write_text(json.dumps(slot_cfg), encoding="utf-8")
    c_slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
    c_grid = c_slot.calendar.bar_grid("1h")
    c_rows = c_grid.merge(pd.DataFrame({"instrument_id": syms}), how="cross")
    _save_scheme_b_factor(crypto_store, slot_id="crypto_fp", slot_config_hash=c_slot.slot_config_hash, calendar=c_slot.calendar, factor_id="cmom", frequency="1h", frame=c_rows.assign(factor_value=np.tile([1.0, -1.0], len(c_grid))))

    c_synth_path = tmp_path / "crypto_fp_synth.json"
    c_synth_path.write_text(json.dumps({
        "analysis_period": {"start_date": "2026-01-03", "end_date": "2026-01-04"},
        "factor_groups": {"c": {"factors": ["cmom"]}},
        "synthesis_methods": {"ws_c": {"method": "weighted_sum", "factor_group": "c", "weights": {"cmom": 1.0}, "preprocess": {"cmom": "none"}}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True, "top_k": 1, "holding_period": 6, "rebalance_freq": "bar"}},
        "output_settings": {"composite_factor_prefix": "C"},
        "cache_settings": {"use_parquet_cache": False},
    }), encoding="utf-8")

    assert harness_cli.main(["factor-synthesis", "--config", str(c_synth_path), "--slot-config", str(slot_path), "--slot-id", "crypto_fp", "--factor-values-store", str(crypto_store), "--frequency", "1h", "--output-root", str(h_root), "--run-id", "fp-1"]) == 0
    b1 = load_strategy_bundle(json.loads((h_root / "factor-synthesis" / "fp-1" / "summary.json").read_text())["metrics"]["strategy_bundle"]["bundle_dir"])

    # Mutate one OHLCV bar price in crypto_fp.csv without changing slot_id, dates, or factor identities.
    bar_rows[10]["close"] = bar_rows[10]["close"] + 15.0
    pd.DataFrame(bar_rows).to_csv(bars_csv, index=False)
    assert harness_cli.main(["factor-synthesis", "--config", str(c_synth_path), "--slot-config", str(slot_path), "--slot-id", "crypto_fp", "--factor-values-store", str(crypto_store), "--frequency", "1h", "--output-root", str(h_root), "--run-id", "fp-2"]) == 0
    b2 = load_strategy_bundle(json.loads((h_root / "factor-synthesis" / "fp-2" / "summary.json").read_text())["metrics"]["strategy_bundle"]["bundle_dir"])

    assert b1.slot_data_fingerprint["execution_bars_sha256"] != b2.slot_data_fingerprint["execution_bars_sha256"]
    assert b1.slot_data_fingerprint != b2.slot_data_fingerprint
    assert b1.bundle_hash != b2.bundle_hash


def test_size_conditional_model_bundle_and_missing_model_fail_closed(tmp_path: Path) -> None:
    """Point 5: size_conditional_elasticnet persists replayable model bundle; missing model or diagnostics-only fails."""
    from run_factor_synthesis import FactorSynthesizer

    synth = object.__new__(FactorSynthesizer)
    synth.config_path = str(tmp_path / "cfg.json")
    Path(synth.config_path).write_text("{}", encoding="utf-8")
    synth.config = {
        "analysis_period": {"start_date": "2023-02-01", "end_date": "2023-02-28"},
        "factor_groups": {"g": {"factors": ["f1", "f2"]}},
        "output_settings": {"weights_output_dir": str(tmp_path / "weights")},
    }
    synth.execution_dir = tmp_path / "exec"
    synth.execution_dir.mkdir()
    synth.run_id = "sc-test"
    synth.frequency = "1d"
    synth.factor_values_store = None
    synth.factor_duckdb_path = None
    synth.factor_identities = {}
    synth._slot_id = "jq_ashare_local"
    cal = SessionCalendar.from_pmc("2023-02-01", "2023-02-28")
    synth._get_slot = lambda: SimpleNamespace(
        slot_id="jq_ashare_local",
        slot_config_hash="0" * 64,
        slot_data_fingerprint=None,
        calendar=cal,
        aux_features=SimpleNamespace(universe_filter_plugin=None),
    )

    model_infos = [{
        "train_start": "2023-01-01",
        "train_end": "2023-01-31",
        "embargo_train_end": "2023-01-31",
        "validation_start": "2023-02-01",
        "validation_end": "2023-02-28",
        "selected_policy": {"gate": [0.0, 0.5], "alpha": 0.01, "l1_ratio": 0.5, "interaction_shrinkage": 2.0},
        "stable_support": ["f1"],
        "stability_fallback": None,
        "coefficients": {
            "global_coefficients": {"f1": 0.4, "f2": -0.2},
            "bucket_deviations": {"f1": {"b0": 0.1, "b1": -0.1}, "f2": {"b0": 0.0, "b1": 0.0}},
            "effective_bucket_coefficients": {"f1": {"b0": 0.5, "b1": 0.3}, "f2": {"b0": -0.2, "b1": -0.2}},
        },
        "intercept": 0.01,
        "protocol": {"bucket_edges": [0.0, 0.5, 1.0]},
    }]
    sc_bundle_path = synth._write_size_conditional_model_bundle(
        output_dir=tmp_path / "weights",
        synthesis_name="sc_demo",
        factor_group="g",
        model_infos=model_infos,
        synthesis_config={"preprocess": {"f1": "zscore", "f2": "zscore"}},
    )
    assert sc_bundle_path is not None and sc_bundle_path.is_file()
    sc_payload = json.loads(sc_bundle_path.read_text(encoding="utf-8"))
    assert sc_payload["model_type"] == "size_conditional_elasticnet"
    assert sc_payload["index"]["models"][0]["coefficients"]["effective_bucket_coefficients"]["f1"]["b0"] == 0.5
    assert sc_payload["index"]["models"][0]["intercept"] == 0.01

    # Diagnostics-only or missing model files must fail _emit_strategy_bundle (no synthesis_spec fallback).
    diag_only = tmp_path / "diag_only.json"
    diag_only.write_text('{"diagnostics": true}\n', encoding="utf-8")
    synth._bundle_syntheses = {
        "sc_demo": {
            "synthesis_name": "sc_demo",
            "method": "size_conditional_elasticnet",
            "training_factor_names": ["f1", "f2"],
            "blend_factor_names": [],
            "baseline_factor_names": [],
            "blend_with_original_ratio": 0.0,
            "dynamic_blend_enabled": False,
            "market_factor_inputs": {},
            "factor_panel_sha256": "f" * 64,
            "preprocess_cfg": {"f1": "zscore", "f2": "zscore"},
            "preprocessing_params": {"preprocess": {"f1": "zscore", "f2": "zscore"}},
            "aux_features_required": [],
            "random_seed": 42,
            "model_artifact_files": [("diagnostics", str(diag_only))],
            "model_infos": model_infos,
        }
    }
    synth._bundle_backtests = [{
        "synthesis_name": "sc_demo",
        "trial_id": "sc_trial_1",
        "summary": {"annual_return": 0.1, "sharpe_ratio": 1.0, "max_drawdown": -0.05, "annual_turnover": 1.0, "trade_count": 2},
        "returns_series": pd.Series([0.01], index=pd.DatetimeIndex(["2023-02-01"])),
        "return_series_artifact": None,
        "rebalance_config": {"rebalance_freq": "weekly"},
        "allow_short": False,
        "benchmark_instrument": None,
        "execution_profile": asdict(ExecutionProfile("close", "close", 0, 0, "decimal", "endpoints_only", "cross_session")),
        "universe_filter": {"quality_filter_enabled": False},
    }]
    with pytest.raises(StrategyBundleError, match="missing required replayable model"):
        synth._emit_strategy_bundle(execution={"expected": 1, "completed": 1, "failed": 0, "errors": []}, metrics={})


def test_strategy_bundle_and_run_result_reject_malformed_probes(tmp_path: Path) -> None:
    """Point 6: validator & run_result reject minimal provenance, bad perf types, empty NAV, mismatched allow_short, missing bundle_dir, and mismatched registrations."""
    bundle, sources = _make_sample_strategy_bundle(tmp_path)

    # 1. Contradictory allow_short between top-level and execution_profile.
    with pytest.raises(StrategyBundleError, match="allow_short"):
        replace(bundle, allow_short=True, bundle_hash="")

    # 2. Non-numeric / string performance_summary metric.
    with pytest.raises(StrategyBundleError, match="finite number"):
        replace(
            bundle,
            performance_summary={**dict(bundle.performance_summary), "annual_return": "0.15"},
            bundle_hash="",
        )

    # 3. Missing backtest_nav.
    with pytest.raises(StrategyBundleError, match="backtest_nav"):
        replace(bundle, backtest_nav=None, bundle_hash="")

    # 4. Forbidden synthesis_spec / diagnostic-only model_artifacts.
    with pytest.raises(StrategyBundleError, match="non-replayable"):
        replace(
            bundle,
            model_artifacts=(
                BundleArtifactRef(
                    path="models/synthesis_spec.json",
                    sha256=sources["models/weights.json"].stat().st_size and sha256_file(sources["models/weights.json"]),
                    artifact_type="synthesis_spec",
                ),
            ),
            bundle_hash="",
        )

    # 5. Empty NAV CSV (header only) fails verify_strategy_bundle.
    empty_nav_src = tmp_path / "empty_nav.csv"
    empty_nav_src.write_text("date,nav\n", encoding="utf-8")
    empty_nav_bundle = replace(
        bundle,
        backtest_nav=BundleArtifactRef(
            path="backtest/nav.csv",
            sha256=sha256_file(empty_nav_src),
            artifact_type="backtest_nav",
            size_bytes=empty_nav_src.stat().st_size,
        ),
        bundle_hash="",
    )
    empty_nav_dir = tmp_path / "empty_nav_bundle"
    with pytest.raises(StrategyBundleError, match="at least one row"):
        write_strategy_bundle(empty_nav_bundle, empty_nav_dir, source_files={**sources, "backtest/nav.csv": empty_nav_src})

    # 6. Minimal execution_result_ref containing only schema_version fails verify_strategy_bundle.
    min_exec = tmp_path / "min_exec.json"
    min_exec.write_text(json.dumps({"schema_version": "execution_result.v1"}), encoding="utf-8")
    min_exec_bundle = replace(bundle, execution_result_sha256=sha256_file(min_exec), bundle_hash="")
    with pytest.raises(StrategyBundleError, match="Invalid execution_result_ref"):
        write_strategy_bundle(min_exec_bundle, tmp_path / "min_exec_bundle", source_files={**sources, "provenance/execution_result.json": min_exec})

    # 7. run_result.py must NOT skip verification when bundle_dir is omitted or outer manifest diverges.
    exec_dir = tmp_path / "exec_probe"
    exec_dir.mkdir()
    cfg_path = exec_dir / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    cfg_hash = config_sha256(cfg_path)
    valid_exec_json = {
        "schema_version": "execution_result.v1",
        "command": "factor-synthesis",
        "run_id": "probe-run",
        "execution_id": "exec_probe",
        "config_sha256": cfg_hash,
        "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
        "metrics": {},
        "artifacts": [],
    }
    probe_exec = tmp_path / "probe_exec.json"
    probe_exec.write_text(json.dumps(valid_exec_json), encoding="utf-8")
    probe_bundle = replace(bundle, execution_result_sha256=sha256_file(probe_exec), bundle_hash="")
    probe_bundle_dir = exec_dir / "strategy_bundle"
    write_strategy_bundle(probe_bundle, probe_bundle_dir, source_files={**sources, "provenance/execution_result.json": probe_exec})

    outer_manifest = exec_dir / "strategy_bundle_manifest.json"
    outer_manifest.write_bytes((probe_bundle_dir / "manifest.json").read_bytes())
    result_path = exec_dir / "result.json"
    result_payload = {
        **valid_exec_json,
        "metrics": {
            "strategy_bundle": {
                "schema_version": probe_bundle.schema_version,
                "bundle_dir": str(probe_bundle_dir),
                "manifest_path": str(outer_manifest),
                "bundle_hash": probe_bundle.bundle_hash,
            }
        },
        "artifacts": [
            {
                "artifact_type": "strategy_bundle",
                "path": str(outer_manifest),
                "sha256": config_sha256(outer_manifest),
                "bundle_dir": str(probe_bundle_dir),
                "bundle_hash": probe_bundle.bundle_hash,
                "schema_version": probe_bundle.schema_version,
            }
        ],
    }
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    assert read_execution_result(result_path, command="factor-synthesis", run_id="probe-run", execution_id="exec_probe", source_sha256=cfg_hash)["execution"]["status"] == "succeeded"

    # Removing bundle_dir from artifact entry (and deleting weight file) must fail!
    no_bdir_payload = json.loads(json.dumps(result_payload))
    no_bdir_payload["artifacts"][0].pop("bundle_dir")
    (probe_bundle_dir / "models/weights.json").unlink()
    result_path.write_text(json.dumps(no_bdir_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle_dir"):
        read_execution_result(result_path, command="factor-synthesis", run_id="probe-run", execution_id="exec_probe", source_sha256=cfg_hash)

    # Restore weight file + bundle_dir, but point outer manifest to another valid manifest -> mismatch must fail!
    (probe_bundle_dir / "models/weights.json").write_bytes(sources["models/weights.json"].read_bytes())
    other_bundle_dir = exec_dir / "other_bundle"
    other_model = tmp_path / "other_weights.json"
    other_model.write_text(
        json.dumps(
            {
                "index": {
                    "models": [
                        {
                            "train_end": "1900-01-01",
                            "weights": {"alpha": 0.5},
                            "feature_names": ["alpha"],
                        }
                    ]
                },
                "metadata": {"method": "weighted_sum"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    other_bundle = replace(
        probe_bundle,
        model_artifacts=(
            BundleArtifactRef(
                path="models/weights.json",
                sha256=sha256_file(other_model),
                artifact_type="model_bundle",
                size_bytes=other_model.stat().st_size,
            ),
        ),
        bundle_hash="",
    )
    write_strategy_bundle(other_bundle, other_bundle_dir, source_files={**sources, "models/weights.json": other_model, "provenance/execution_result.json": probe_exec})
    outer_manifest.write_bytes((other_bundle_dir / "manifest.json").read_bytes())
    mismatched_payload = json.loads(json.dumps(result_payload))
    mismatched_payload["artifacts"][0]["sha256"] = config_sha256(outer_manifest)
    result_path.write_text(json.dumps(mismatched_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="registered manifest differs"):
        read_execution_result(result_path, command="factor-synthesis", run_id="probe-run", execution_id="exec_probe", source_sha256=cfg_hash)


def test_partial_candidate_failure_reports_trial_id_and_fails_execution(tmp_path: Path) -> None:
    """Round 2 Point 1: partial candidate failure or missing candidate under one method must fail execution with trial_id."""
    from run_factor_synthesis import FactorSynthesizer
    from tools.agent_harness import runner as harness_runner

    cfg_path = tmp_path / "partial_cfg.json"
    cfg_payload = {
        "synthesis_methods": {"ws": {"method": "weighted_sum"}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True, "top_pct": [0.1, 0.2]}},
        "output_settings": {"composite_factor_prefix": "P"},
    }
    cfg_path.write_text(json.dumps(cfg_payload), encoding="utf-8")

    synth = object.__new__(FactorSynthesizer)
    synth.config_path = str(cfg_path)
    synth.config = cfg_payload
    synth.logger = __import__("logging").getLogger(__name__)
    synth._run_local_reranker_profile = lambda: None
    synth._emit_strategy_bundle = lambda **kw: (_ for _ in ()).throw(AssertionError("should not emit on failure"))

    expected = synth._expected_backtest_trials("ws", cfg_payload["synthesis_methods"]["ws"])
    assert len(expected) == 2
    t_ok, t_fail = expected[0]["trial_id"], expected[1]["trial_id"]

    # Case A: one candidate succeeded, second failed with error.
    def _partial_fail(sname: str, scfg: dict) -> None:
        synth._vectorbt_result_summaries.extend([
            {
                "synthesis_name": sname,
                "factor_name": "P_ws",
                "trial_id": t_ok,
                "annual_return": 0.12,
                "sharpe_ratio": 1.1,
                "max_drawdown": -0.05,
            },
            {
                "synthesis_name": sname,
                "factor_name": "P_ws",
                "trial_id": t_fail,
                "error": "simulated_backtest_failure",
            },
        ])
        synth._bundle_backtests.append({"synthesis_name": sname, "trial_id": t_ok})

    synth.rolling_window_synthesis = _partial_fail
    execution_a: dict = {}
    with pytest.raises(RuntimeError, match=t_fail):
        synth.run_synthesis(execution=execution_a, metrics={})
    assert execution_a["expected"] == 1
    assert execution_a["completed"] == 0
    assert execution_a["failed"] == 1
    assert any(t_fail in err["error"] for err in execution_a["errors"])

    # Case B: one candidate succeeded, second missing entirely.
    def _partial_missing(sname: str, scfg: dict) -> None:
        synth._vectorbt_result_summaries.append({
            "synthesis_name": sname,
            "factor_name": "P_ws",
            "trial_id": t_ok,
            "annual_return": 0.12,
            "sharpe_ratio": 1.1,
            "max_drawdown": -0.05,
        })
        synth._bundle_backtests.append({"synthesis_name": sname, "trial_id": t_ok})

    synth.rolling_window_synthesis = _partial_missing
    execution_b: dict = {}
    with pytest.raises(RuntimeError, match=t_fail):
        synth.run_synthesis(execution=execution_b, metrics={})
    assert execution_b["completed"] == 0 and execution_b["failed"] == 1

    # Case C: _bound_result_metrics rejects result.json claiming succeeded with failed candidate.
    res_path = tmp_path / "attempt" / "result.json"
    res_path.parent.mkdir(parents=True)
    res_path.write_text(
        json.dumps({
            "schema_version": "execution_result.v1",
            "command": "factor-synthesis",
            "run_id": "r1",
            "execution_id": "attempt",
            "config_sha256": config_sha256(cfg_path),
            "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
            "metrics": {
                "backtest_runs": [
                    {"trial_id": t_ok, "annual_return": 0.1, "sharpe_ratio": 1.0, "max_drawdown": -0.05},
                    {"trial_id": t_fail, "error": "boom"},
                ]
            },
            "artifacts": [],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=t_fail):
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="r1",
            execution_id="attempt",
            source_sha256=config_sha256(cfg_path),
            config_payload=cfg_payload,
        )


def test_reuse_lasso_leaf_bundle_packages_parent_baseline_and_replays_standalone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 2 Point 2: synthesis_reuse on LASSO leaf packages parent_baseline and replays solely from bundle_dir."""
    import shutil
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    fixture = _create_ashare_parity_fixture(tmp_path)
    db_path: Path = fixture["db_path"]
    store_root: Path = fixture["store_root"]
    sessions = fixture["sessions"]
    local_out_db = tmp_path / "leaf_local.duckdb"
    weights_dir = tmp_path / "leaf_weights"

    # Build a frozen parent_baseline composite parquet covering all sessions & stocks.
    parent_idx = pd.MultiIndex.from_product([sessions, STOCKS], names=["date", "stock_code"])
    rng = np.random.default_rng(77)
    parent_series = pd.Series(rng.normal(size=len(parent_idx)), index=parent_idx, name="composite_value")
    parent_path = tmp_path / "frozen_parent.parquet"
    parent_series.to_frame().to_parquet(parent_path)
    parent_sha = sha256_file(parent_path)

    cfg = {
        "database": {
            "factor_db": {"duckdb_path": str(db_path), "factor_values_table": "factor_values"},
            "price_db": {"duckdb_path": str(db_path), "price_table": "price_data_daily", "factor_data_table": "factor_data"},
            "local_db": {"duckdb_path": str(local_out_db), "results_table": "factor_analysis_results"},
            "price_table": "price_data_daily",
        },
        "factor_values_store": str(store_root),
        "analysis_period": {"start_date": "2023-02-06", "end_date": "2023-03-24"},
        "rolling_window": {"train_window_days": 20, "predict_window_days": 10, "step_size_days": 10, "min_train_samples": 10},
        "factor_groups": {"all": {"factors": FACTORS}},
        "synthesis_methods": {
            "lasso_leaf": {
                "method": "lasso",
                "factor_group": "all",
                "target_period": 1,
                "target_cs_demean": False,
                "preprocess": {"alpha": "zscore", "beta": "zscore"},
                "lasso_cv_folds": 2,
                "random_state": 42,
                "blend_with_original_ratio": 0.5,
                "parent_baseline": {
                    "path": str(parent_path),
                    "sha256": parent_sha,
                    "parent_winner_id": "parent_demo",
                    "parent_run_id": "parent_run_0",
                },
                "target_residualization": {"enabled": True, "parent_baseline": True},
            }
        },
        "performance_evaluation": {
            "skip_alphalens_analysis": True,
            "top_k_strategy": {"enabled": True, "top_k": 2, "holding_period": 1, "enable_tday_filter": True},
        },
        "output_settings": {
            "composite_factor_prefix": "LEAF",
            "weights_output_dir": str(weights_dir),
        },
        "cache_settings": {"use_parquet_cache": False},
        "logging": {"level": "INFO"},
    }
    train_cfg_path = tmp_path / "leaf_train.json"
    train_cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    harness_root = tmp_path / "leaf_harness"
    assert harness_cli.main([
        "factor-synthesis",
        "--config", str(train_cfg_path),
        "--output-root", str(harness_root),
        "--run-id", "leaf-train",
    ]) == 0
    train_summary = json.loads((harness_root / "factor-synthesis" / "leaf-train" / "summary.json").read_text(encoding="utf-8"))
    exec_art = next(a for a in train_summary["artifacts"] if a["artifact_type"] == "execution_result")
    train_exec_dir = Path(exec_art["path"]).parent
    reuse_source = train_exec_dir / "outputs"
    if not (reuse_source / "lasso_model_bundle_lasso_leaf.json").is_file():
        reuse_source = weights_dir

    # Run synthesis_reuse from the trained leaf outputs.
    reuse_cfg_path = tmp_path / "leaf_reuse.json"
    reuse_cfg = {
        **cfg,
        "synthesis_reuse": {"enabled": True, "source_dir": str(reuse_source)},
    }
    reuse_cfg_path.write_text(json.dumps(reuse_cfg), encoding="utf-8")
    assert harness_cli.main([
        "factor-synthesis",
        "--config", str(reuse_cfg_path),
        "--output-root", str(harness_root),
        "--run-id", "leaf-reuse",
    ]) == 0
    reuse_summary = json.loads((harness_root / "factor-synthesis" / "leaf-reuse" / "summary.json").read_text(encoding="utf-8"))
    reuse_bundle_dir = Path(reuse_summary["metrics"]["strategy_bundle"]["bundle_dir"])
    reuse_bundle = verify_strategy_bundle(reuse_bundle_dir)

    pb_refs = [ref for ref in reuse_bundle.model_artifacts if ref.artifact_type == "parent_baseline"]
    assert len(pb_refs) == 1
    assert pb_refs[0].sha256 == parent_sha
    assert (reuse_bundle_dir / pb_refs[0].path).is_file()

    # Copy ONLY the packaged StrategyBundle directory to an isolated location, then delete original execution/source/parent files!
    standalone_dir = tmp_path / "standalone_bundle"
    shutil.copytree(reuse_bundle_dir, standalone_dir)
    expected_returns = pd.read_csv(reuse_bundle_dir / reuse_bundle.backtest_returns.path)["net_return"].to_numpy()
    shutil.rmtree(harness_root)
    shutil.rmtree(weights_dir, ignore_errors=True)
    parent_path.unlink()

    standalone_bundle = verify_strategy_bundle(standalone_dir)
    model_bundle_rel = next(
        ref.path for ref in standalone_bundle.model_artifacts if ref.artifact_type == "model_bundle"
    )
    bundle_ret_df = pd.read_csv(standalone_dir / standalone_bundle.backtest_returns.path)
    expected_returns = bundle_ret_df["net_return"].to_numpy()
    expected_dates = pd.to_datetime(bundle_ret_df["date"])
    with duckdb.connect(str(db_path), read_only=True, config={"threads": 1}) as conn:
        slot = build_jq_ashare_local(conn, start=START, end=END)
        price_wide = fixture["close_df"].reindex(expected_dates)
        open_wide = fixture["open_df"].reindex(expected_dates)
        dummy_factor_df = pd.DataFrame(0.0, index=price_wide.index, columns=price_wide.columns)
        rebal_cfg = dict(standalone_bundle.rebalance_config)
        runner = VectorBTBacktestRunner(
            {
                "top_k_strategy": {
                    "top_k": rebal_cfg.get("top_k", 2),
                    "holding_period": rebal_cfg.get("holding_period", 1),
                    "enable_tday_filter": standalone_bundle.universe_filter["enable_tday_filter"],
                },
                "calendar_filters": {"skip_months": list(rebal_cfg.get("skip_months", []))},
            },
            slot=slot,
            frequency=standalone_bundle.frequency,
            bundle_path=str(standalone_dir / model_bundle_rel),
            factor_values_store=store_root,
            factor_identities=fixture["identities"],
        )
        replay_res = runner.run_backtest(
            dummy_factor_df,
            price_wide,
            open_df=open_wide,
            dynamic_pools=None,
            rebalance_freq=str(rebal_cfg.get("rebalance_freq", "weekly")),
        )
        runner.close()
        assert "error" not in replay_res, replay_res
        replay_returns = replay_res["portfolio"].returns().to_numpy()
        np.testing.assert_allclose(replay_returns, expected_returns, rtol=0, atol=1e-12, equal_nan=True)


def test_execution_bars_fingerprint_tracks_volume_only_mutation_for_turnover_collapse(
    tmp_path: Path,
) -> None:
    """Round 3 Point 2: mutating only lookback bar volume under active risk_overlay.crisis_guard updates votes and execution_bars_sha256."""
    from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner

    bars_csv = tmp_path / "lookback_bars.csv"
    store_root = tmp_path / "lookback_store"
    all_days = pd.bdate_range("2023-11-01", "2024-06-10")
    symbols = ["A", "B"]
    mutated_day = pd.Timestamp("2024-01-15")
    assert mutated_day in all_days

    def _write_bars(mutated_volume: float) -> None:
        rows = []
        for idx, day in enumerate(all_days):
            vol = mutated_volume if day == mutated_day else 100.0
            for j, sym in enumerate(symbols):
                close_px = 10.0 + 0.05 * idx + 0.1 * j
                rows.append({
                    "symbol": sym,
                    "time": f"{day.strftime('%Y-%m-%d')}T16:00:00Z",
                    "open": close_px,
                    "high": close_px + 0.5,
                    "low": close_px - 0.5,
                    "close": close_px,
                    "volume": vol,
                })
        pd.DataFrame(rows).to_csv(bars_csv, index=False)

    slot_cfg = {
        "schema_version": 1,
        "slot_id": "lookback_bar_slot",
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "daily_utc",
                "start": "2023-11-01",
                "end": "2024-06-10",
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
                "paths": ["lookback_bars.csv"],
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
            "entry_price_field": "close",
            "exit_price_field": "close",
            "min_entry_offset_bars": 0,
            "exit_adjustment": 0,
            "return_unit": "decimal",
            "gap_rule": "endpoints_only",
            "boundary_policy": "cross_session",
            "allow_short": False,
            "commission_bps": 0.0,
            "sell_tax_bps": 0.0,
            "slippage_bps": 0.0,
            "benchmark_instrument": None,
        },
    }
    slot_path = tmp_path / "lookback_slot.json"
    slot_path.write_text(json.dumps(slot_cfg), encoding="utf-8")
    slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
    grid = slot.calendar.bar_grid("1d")
    f_rows = grid.merge(pd.DataFrame({"instrument_id": symbols}), how="cross")
    _save_scheme_b_factor(
        store_root,
        slot_id=slot.slot_id,
        slot_config_hash=slot.slot_config_hash,
        calendar=slot.calendar,
        factor_id="sig",
        frequency="1d",
        frame=f_rows.assign(factor_value=np.tile([2.0, 1.0], len(grid))),
    )
    top_k_cfg = {
        "enabled": True,
        "top_k": 1,
        "holding_period": 1,
        "rebalance_freq": "daily",
        "enable_tday_filter": False,
        "liquidity_cap": {"enabled": True, "adv_lookback_days": 80},
        "risk_overlay": {
            "crisis_guard": {
                "enabled": True,
                "min_triggers": 1,
                "derisk_exposure": 0.2,
                "cooldown_days": 0,
                "signals": {
                    "limit_down": {"enabled": False},
                    "breadth": {"enabled": False},
                    "spread": {"enabled": False},
                    "turnover_collapse": {
                        "enabled": True,
                        "short_window": 5,
                        "long_window": 150,
                        "ratio_threshold": 0.5,
                    },
                },
            }
        },
    }
    synth_cfg = {
        "analysis_period": {"start_date": "2024-06-01", "end_date": "2024-06-10"},
        "factor_groups": {"g": {"factors": ["sig"]}},
        "synthesis_methods": {
            "ws": {
                "method": "weighted_sum",
                "factor_group": "g",
                "weights": {"sig": 1.0},
                "preprocess": {"sig": "none"},
            }
        },
        "performance_evaluation": {"top_k_strategy": top_k_cfg},
        "output_settings": {"composite_factor_prefix": "VOL"},
        "cache_settings": {"use_parquet_cache": False},
    }
    synth_path = tmp_path / "lookback_synth.json"
    synth_path.write_text(json.dumps(synth_cfg), encoding="utf-8")
    h_root = tmp_path / "vol_harness"

    def _active_crisis_votes() -> pd.Series:
        s_obj = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
        bt_runner = VectorBTBacktestRunner({"top_k_strategy": top_k_cfg}, slot=s_obj, frequency="1d")
        try:
            eval_idx = pd.date_range("2024-06-03", "2024-06-10", freq="B", tz="UTC")
            price_stub = pd.DataFrame(10.0, index=eval_idx, columns=symbols)
            votes = bt_runner._crisis_guard_vote_series(
                bt_runner.overlay["risk_overlay"]["crisis_guard"], symbols, price_stub
            )
            assert len(votes) == 1
            return votes[0]
        finally:
            bt_runner.close()

    def _vote_on_day(votes: pd.Series, day_str: str) -> bool:
        idx = pd.to_datetime(votes.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        matched = votes.loc[idx.normalize() == pd.Timestamp(day_str)]
        assert not matched.empty, f"missing crisis_guard vote for {day_str} in {list(votes.index[-5:])}"
        return bool(matched.iloc[0])

    _write_bars(100.0)
    votes_base = _active_crisis_votes()
    assert not _vote_on_day(votes_base, "2024-06-03")
    assert harness_cli.main([
        "factor-synthesis",
        "--config", str(synth_path),
        "--slot-config", str(slot_path),
        "--slot-id", slot.slot_id,
        "--factor-values-store", str(store_root),
        "--frequency", "1d",
        "--output-root", str(h_root),
        "--run-id", "vol-100",
    ]) == 0
    b1 = load_strategy_bundle(
        json.loads((h_root / "factor-synthesis" / "vol-100" / "summary.json").read_text())["metrics"]["strategy_bundle"]["bundle_dir"]
    )

    _write_bars(50000.0)
    votes_mutated = _active_crisis_votes()
    assert _vote_on_day(votes_mutated, "2024-06-03")
    assert not votes_base.equals(votes_mutated)
    assert harness_cli.main([
        "factor-synthesis",
        "--config", str(synth_path),
        "--slot-config", str(slot_path),
        "--slot-id", slot.slot_id,
        "--factor-values-store", str(store_root),
        "--frequency", "1d",
        "--output-root", str(h_root),
        "--run-id", "vol-30000",
    ]) == 0
    b2 = load_strategy_bundle(
        json.loads((h_root / "factor-synthesis" / "vol-30000" / "summary.json").read_text())["metrics"]["strategy_bundle"]["bundle_dir"]
    )

    assert b1.slot_data_fingerprint["execution_bars_sha256"] != b2.slot_data_fingerprint["execution_bars_sha256"]
    assert b1.bundle_hash != b2.bundle_hash


def test_strategy_bundle_rejects_metadata_only_model_bundle_and_missing_parent_ref(
    tmp_path: Path,
) -> None:
    """Round 2 Points 2 & 4: reject metadata-only model_bundle and missing/unregistered parent_baseline refs."""
    bundle, sources = _make_sample_strategy_bundle(tmp_path)

    # 1. Metadata-only JSON with valid hash & size for model_bundle must be rejected!
    meta_only = tmp_path / "meta_only.json"
    meta_only.write_text(json.dumps({"method": "weighted_sum", "random_seed": 42}) + "\n", encoding="utf-8")
    meta_bundle = replace(
        bundle,
        model_artifacts=(
            BundleArtifactRef(
                path="models/weights.json",
                sha256=sha256_file(meta_only),
                artifact_type="model_bundle",
                size_bytes=meta_only.stat().st_size,
            ),
        ),
        bundle_hash="",
    )
    with pytest.raises(StrategyBundleError, match="index"):
        write_strategy_bundle(
            meta_bundle,
            tmp_path / "meta_only_bundle",
            source_files={**sources, "models/weights.json": meta_only},
        )

    # 2. index.models entry with no replayable weights must be rejected!
    empty_model_entry = tmp_path / "empty_model_entry.json"
    empty_model_entry.write_text(json.dumps({"index": {"models": [{"train_end": "1900-01-01"}]}}), encoding="utf-8")
    empty_entry_bundle = replace(
        bundle,
        model_artifacts=(
            BundleArtifactRef(
                path="models/weights.json",
                sha256=sha256_file(empty_model_entry),
                artifact_type="model_bundle",
                size_bytes=empty_model_entry.stat().st_size,
            ),
        ),
        bundle_hash="",
    )
    with pytest.raises(StrategyBundleError, match="missing valid weights"):
        write_strategy_bundle(
            empty_entry_bundle,
            tmp_path / "empty_entry_bundle",
            source_files={**sources, "models/weights.json": empty_model_entry},
        )

    # 3. LASSO model_bundle referencing parent_baseline that is absent from model_artifacts must be rejected!
    pb_file = tmp_path / "parent_baseline_abc.parquet"
    pb_file.write_bytes(b"PAR1fake")
    lasso_pb_bundle_file = tmp_path / "lasso_pb_bundle.json"
    lasso_pb_bundle_file.write_text(
        json.dumps({
            "version": "1.0",
            "model_type": "lasso",
            "index": {
                "requires_parent_baseline": True,
                "models": [{
                    "train_end": "2023-02-01",
                    "weights": {"alpha": 0.5},
                    "feature_names": ["alpha"],
                    "parent_baseline": {"path": "parent_baseline_abc.parquet", "sha256": sha256_file(pb_file)},
                }],
            },
        }),
        encoding="utf-8",
    )
    unregistered_pb_bundle = replace(
        bundle,
        synthesis_method="lasso",
        model_artifacts=(
            BundleArtifactRef(
                path="models/lasso_model_bundle_demo.json",
                sha256=sha256_file(lasso_pb_bundle_file),
                artifact_type="model_bundle",
                size_bytes=lasso_pb_bundle_file.stat().st_size,
            ),
        ),
        bundle_hash="",
    )
    with pytest.raises(StrategyBundleError, match="parent_baseline"):
        write_strategy_bundle(
            unregistered_pb_bundle,
            tmp_path / "unregistered_pb_bundle",
            source_files={**sources, "models/lasso_model_bundle_demo.json": lasso_pb_bundle_file},
        )


def test_run_result_rejects_mismatched_primary_and_candidate_manifest_paths(tmp_path: Path) -> None:
    """Round 2 Point 5: validate both primary strategy_bundle and strategy_bundles list including manifest_path."""
    bundle, sources = _make_sample_strategy_bundle(tmp_path)
    exec_dir = tmp_path / "exec_multi_probe"
    exec_dir.mkdir()
    cfg_path = exec_dir / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    cfg_hash = config_sha256(cfg_path)
    valid_exec_json = {
        "schema_version": "execution_result.v1",
        "command": "factor-synthesis",
        "run_id": "multi-probe",
        "execution_id": "exec_multi_probe",
        "config_sha256": cfg_hash,
        "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
        "metrics": {},
        "artifacts": [],
    }
    probe_exec = tmp_path / "multi_probe_exec.json"
    probe_exec.write_text(json.dumps(valid_exec_json), encoding="utf-8")
    probe_bundle = replace(bundle, execution_result_sha256=sha256_file(probe_exec), bundle_hash="")
    b_dir = exec_dir / "strategy_bundles" / "cand1"
    write_strategy_bundle(probe_bundle, b_dir, source_files={**sources, "provenance/execution_result.json": probe_exec})
    m_path = b_dir / "manifest.json"
    cand_desc = {
        "schema_version": probe_bundle.schema_version,
        "bundle_dir": str(b_dir),
        "manifest_path": str(m_path),
        "bundle_hash": probe_bundle.bundle_hash,
    }
    base_result = {
        **valid_exec_json,
        "metrics": {
            "strategy_bundles": [cand_desc],
            "strategy_bundle": dict(cand_desc),
        },
        "artifacts": [
            {
                "artifact_type": "strategy_bundle",
                "path": str(m_path),
                "sha256": config_sha256(m_path),
                "bundle_dir": str(b_dir),
                "bundle_hash": probe_bundle.bundle_hash,
                "schema_version": probe_bundle.schema_version,
            }
        ],
    }
    result_path = exec_dir / "result.json"
    result_path.write_text(json.dumps(base_result), encoding="utf-8")
    assert read_execution_result(
        result_path, command="factor-synthesis", run_id="multi-probe", execution_id="exec_multi_probe", source_sha256=cfg_hash
    )["execution"]["status"] == "succeeded"

    # 1. Primary strategy_bundle pointing to nonexistent dir must fail even when strategy_bundles is valid.
    bad_primary = json.loads(json.dumps(base_result))
    bad_primary["metrics"]["strategy_bundle"]["bundle_dir"] = str(exec_dir / "missing_bundle")
    result_path.write_text(json.dumps(bad_primary), encoding="utf-8")
    with pytest.raises(ValueError, match="strategy_bundle"):
        read_execution_result(
            result_path, command="factor-synthesis", run_id="multi-probe", execution_id="exec_multi_probe", source_sha256=cfg_hash
        )

    # 2. Candidate manifest_path pointing to nonexistent file must fail even when bundle_dir & bundle_hash match.
    bad_manifest = json.loads(json.dumps(base_result))
    bad_manifest["metrics"]["strategy_bundles"][0]["manifest_path"] = str(exec_dir / "missing_manifest.json")
    result_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_path"):
        read_execution_result(
            result_path, command="factor-synthesis", run_id="multi-probe", execution_id="exec_multi_probe", source_sha256=cfg_hash
        )


def test_harness_rejects_missing_duplicate_or_cross_trial_candidates_and_bundles(
    tmp_path: Path,
) -> None:
    """Round 3 Point 1: Harness enforces exact expected trial_ids across backtest_runs, strategy_bundles, and manifests."""
    from tools.agent_harness import runner as harness_runner
    from tools.agent_harness.run_result import expected_backtest_trials

    bundle, sources = _make_sample_strategy_bundle(tmp_path)
    cfg_payload = {
        "synthesis_methods": {"ws": {"method": "weighted_sum"}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True, "top_pct": [0.1, 0.2]}},
        "output_settings": {"composite_factor_prefix": "P"},
    }
    exec_dir = tmp_path / "exec_trials"
    exec_dir.mkdir()
    cfg_path = exec_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg_payload), encoding="utf-8")
    cfg_hash = config_sha256(cfg_path)

    expected = expected_backtest_trials(cfg_payload, config_path=cfg_path)
    assert len(expected) == 2
    t1, t2 = expected[0]["trial_id"], expected[1]["trial_id"]

    exec_base = {
        "schema_version": "execution_result.v1",
        "command": "factor-synthesis",
        "run_id": "trial-probe",
        "execution_id": "exec_trials",
        "config_sha256": cfg_hash,
        "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
        "metrics": {},
        "artifacts": [],
    }
    probe_exec = tmp_path / "trials_exec_snap.json"
    probe_exec.write_text(json.dumps(exec_base), encoding="utf-8")
    exec_sha = sha256_file(probe_exec)

    def _write_trial_bundle(tid: str, pct: float) -> tuple[dict, dict]:
        b_dir = exec_dir / "strategy_bundles" / tid
        tb = replace(
            bundle,
            rebalance_config={
                "rebalance_freq": "weekly",
                "selection_mode": "top_pct",
                "top_pct": pct,
                "top_k": None,
                "synthesis_name": "ws",
                "trial_id": tid,
            },
            execution_result_sha256=exec_sha,
            bundle_hash="",
        )
        write_strategy_bundle(tb, b_dir, source_files={**sources, "provenance/execution_result.json": probe_exec})
        m_path = b_dir / "manifest.json"
        desc = {
            "schema_version": tb.schema_version,
            "synthesis_name": "ws",
            "trial_id": tid,
            "selection_mode": "top_pct",
            "top_pct": pct,
            "top_k": None,
            "bundle_dir": str(b_dir),
            "manifest_path": str(m_path),
            "bundle_hash": tb.bundle_hash,
        }
        art = {
            "artifact_type": "strategy_bundle",
            "path": str(m_path),
            "sha256": config_sha256(m_path),
            "bundle_dir": str(b_dir),
            "bundle_hash": tb.bundle_hash,
            "schema_version": tb.schema_version,
            "synthesis_name": "ws",
            "trial_id": tid,
            "selection_mode": "top_pct",
            "top_pct": pct,
            "top_k": None,
        }
        return desc, art

    desc1, art1 = _write_trial_bundle(t1, 0.1)
    desc2, art2 = _write_trial_bundle(t2, 0.2)
    run1 = {
        "synthesis_name": "ws",
        "factor_name": "P_ws",
        "trial_id": t1,
        "selection_mode": "top_pct",
        "top_pct": 0.1,
        "top_k": None,
        "annual_return": 0.1,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.05,
    }
    run2 = {
        "synthesis_name": "ws",
        "factor_name": "P_ws",
        "trial_id": t2,
        "selection_mode": "top_pct",
        "top_pct": 0.2,
        "top_k": None,
        "annual_return": 0.12,
        "sharpe_ratio": 1.1,
        "max_drawdown": -0.04,
    }

    res_path = exec_dir / "result.json"
    good_payload = {
        **exec_base,
        "metrics": {
            "backtest_runs": [run1, run2],
            "strategy_bundles": [desc1, desc2],
            "strategy_bundle": dict(desc2),
        },
        "artifacts": [art1, art2],
    }
    res_path.write_text(json.dumps(good_payload), encoding="utf-8")
    ok_metrics = harness_runner._bound_result_metrics(
        res_path, action="factor-synthesis", run_id="trial-probe", execution_id="exec_trials",
        source_sha256=cfg_hash, config_payload=cfg_payload, config_path=cfg_path,
    )
    assert ok_metrics["execution"]["status"] == "succeeded"

    # 1. Duplicate 0.1 records, missing 0.2, zero artifacts -> must fail!
    dup_no_art = {
        **exec_base,
        "metrics": {"backtest_runs": [dict(run1), dict(run1)]},
        "artifacts": [],
    }
    res_path.write_text(json.dumps(dup_no_art), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        harness_runner._bound_result_metrics(
            res_path, action="factor-synthesis", run_id="trial-probe", execution_id="exec_trials",
            source_sha256=cfg_hash, config_payload=cfg_payload,
        )

    # 2. Both backtest_runs valid, zero artifacts/bundles -> must fail!
    no_bundles = {
        **exec_base,
        "metrics": {"backtest_runs": [run1, run2]},
        "artifacts": [],
    }
    res_path.write_text(json.dumps(no_bundles), encoding="utf-8")
    with pytest.raises(ValueError, match="strategy_bundle_missing_for_trials"):
        harness_runner._bound_result_metrics(
            res_path, action="factor-synthesis", run_id="trial-probe", execution_id="exec_trials",
            source_sha256=cfg_hash, config_payload=cfg_payload,
        )

    # 3. Two valid backtest_runs, only 0.1 artifact registered, but 0.1 bundle cross-bound as 0.2 in strategy_bundles & strategy_bundle -> must fail!
    cross_bound_desc = {**desc1, "trial_id": t2}
    cross_bound = {
        **exec_base,
        "metrics": {
            "backtest_runs": [run1, run2],
            "strategy_bundles": [desc1, cross_bound_desc],
            "strategy_bundle": dict(cross_bound_desc),
        },
        "artifacts": [art1],
    }
    res_path.write_text(json.dumps(cross_bound), encoding="utf-8")
    with pytest.raises(ValueError, match="trial_id"):
        harness_runner._bound_result_metrics(
            res_path, action="factor-synthesis", run_id="trial-probe", execution_id="exec_trials",
            source_sha256=cfg_hash, config_payload=cfg_payload,
        )


def test_harness_rejects_id_preserving_selection_param_mismatches(tmp_path: Path) -> None:
    """Round 4 Point 1: reject ID-preserving parameter mismatches across backtest_runs, descriptors, and manifests."""
    from tools.agent_harness import runner as harness_runner
    from tools.agent_harness.run_result import expected_backtest_trials

    bundle, sources = _make_sample_strategy_bundle(tmp_path)
    cfg_payload = {
        "synthesis_methods": {"ws": {"method": "weighted_sum"}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True, "top_pct": [0.1, 0.2]}},
        "output_settings": {"composite_factor_prefix": "P"},
    }
    exec_dir = tmp_path / "exec_sel_probe"
    exec_dir.mkdir()
    cfg_path = exec_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg_payload), encoding="utf-8")
    cfg_hash = config_sha256(cfg_path)

    expected = expected_backtest_trials(cfg_payload, config_path=cfg_path)
    assert len(expected) == 2
    t1, t2 = expected[0]["trial_id"], expected[1]["trial_id"]

    exec_base = {
        "schema_version": "execution_result.v1",
        "command": "factor-synthesis",
        "run_id": "sel-probe",
        "execution_id": "exec_sel_probe",
        "config_sha256": cfg_hash,
        "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
        "metrics": {},
        "artifacts": [],
    }
    probe_exec = tmp_path / "sel_probe_exec.json"
    probe_exec.write_text(json.dumps(exec_base), encoding="utf-8")
    exec_sha = sha256_file(probe_exec)

    def _build_bundle(
        sub_dir: str,
        *,
        tid: str,
        synthesis_name: str = "ws",
        selection_mode: str = "top_pct",
        top_pct: float | None = 0.1,
        top_k: int | None = None,
    ) -> tuple[dict, dict]:
        b_dir = exec_dir / sub_dir
        tb = replace(
            bundle,
            rebalance_config={
                "rebalance_freq": "weekly",
                "selection_mode": selection_mode,
                "top_pct": top_pct,
                "top_k": top_k,
                "synthesis_name": synthesis_name,
                "trial_id": tid,
            },
            execution_result_sha256=exec_sha,
            bundle_hash="",
        )
        write_strategy_bundle(tb, b_dir, source_files={**sources, "provenance/execution_result.json": probe_exec})
        m_path = b_dir / "manifest.json"
        desc = {
            "schema_version": tb.schema_version,
            "synthesis_name": synthesis_name,
            "trial_id": tid,
            "selection_mode": selection_mode,
            "top_pct": top_pct,
            "top_k": top_k,
            "bundle_dir": str(b_dir),
            "manifest_path": str(m_path),
            "bundle_hash": tb.bundle_hash,
        }
        art = {
            "artifact_type": "strategy_bundle",
            "path": str(m_path),
            "sha256": config_sha256(m_path),
            "bundle_dir": str(b_dir),
            "bundle_hash": tb.bundle_hash,
            "schema_version": tb.schema_version,
            "synthesis_name": synthesis_name,
            "trial_id": tid,
            "selection_mode": selection_mode,
            "top_pct": top_pct,
            "top_k": top_k,
        }
        return desc, art

    desc1, art1 = _build_bundle("strategy_bundles/t1", tid=t1, top_pct=0.1)
    desc2, art2 = _build_bundle("strategy_bundles/t2", tid=t2, top_pct=0.2)
    run1 = {
        "synthesis_name": "ws",
        "factor_name": "P_ws",
        "trial_id": t1,
        "selection_mode": "top_pct",
        "top_pct": 0.1,
        "top_k": None,
        "annual_return": 0.10,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.05,
        "return_series_artifact": {"trial_id": t1, "selection_mode": "top_pct", "top_pct": 0.1, "top_k": None},
    }
    run2 = {
        "synthesis_name": "ws",
        "factor_name": "P_ws",
        "trial_id": t2,
        "selection_mode": "top_pct",
        "top_pct": 0.2,
        "top_k": None,
        "annual_return": 0.12,
        "sharpe_ratio": 1.1,
        "max_drawdown": -0.04,
        "return_series_artifact": {"trial_id": t2, "selection_mode": "top_pct", "top_pct": 0.2, "top_k": None},
    }
    good_payload = {
        **exec_base,
        "metrics": {
            "backtest_runs": [run1, run2],
            "strategy_bundles": [desc1, desc2],
            "strategy_bundle": dict(desc2),
        },
        "artifacts": [art1, art2],
    }
    res_path = exec_dir / "result.json"
    res_path.write_text(json.dumps(good_payload), encoding="utf-8")
    assert (
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="sel-probe",
            execution_id="exec_sel_probe",
            source_sha256=cfg_hash,
            config_payload=cfg_payload,
            config_path=cfg_path,
        )["execution"]["status"]
        == "succeeded"
    )

    # 1. Probe 1: copy 0.1 full backtest record into 0.2 slot keeping only trial_id=t2 -> must fail!
    copied_run2 = {**run1, "trial_id": t2}
    bad_copied_run = json.loads(json.dumps(good_payload))
    bad_copied_run["metrics"]["backtest_runs"] = [run1, copied_run2]
    res_path.write_text(json.dumps(bad_copied_run), encoding="utf-8")
    with pytest.raises(ValueError, match="top_pct mismatch"):
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="sel-probe",
            execution_id="exec_sel_probe",
            source_sha256=cfg_hash,
            config_payload=cfg_payload,
            config_path=cfg_path,
        )

    # 1b. Backtest record with matching trial_id but wrong selection_mode / synthesis_name / missing top_pct -> must fail!
    for mutated_fields, pattern in (
        ({"selection_mode": "top_k", "top_pct": None, "top_k": 5}, "selection_mode mismatch"),
        ({"synthesis_name": "other_method"}, "synthesis_name mismatch"),
        ({"top_k": 5}, "top_k mismatch"),
    ):
        bad_run_fields = json.loads(json.dumps(good_payload))
        bad_run_fields["metrics"]["backtest_runs"][1].update(mutated_fields)
        res_path.write_text(json.dumps(bad_run_fields), encoding="utf-8")
        with pytest.raises(ValueError, match=pattern):
            harness_runner._bound_result_metrics(
                res_path,
                action="factor-synthesis",
                run_id="sel-probe",
                execution_id="exec_sel_probe",
                source_sha256=cfg_hash,
                config_payload=cfg_payload,
                config_path=cfg_path,
            )

    # 2. Probe 2: valid backtest_runs, mutate top_pct in strategy_bundles[1] & strategy_bundle to 0.1 while keeping trial_id=t2 -> must fail in both read_execution_result & _bound_result_metrics!
    bad_desc_both = json.loads(json.dumps(good_payload))
    bad_desc_both["metrics"]["strategy_bundles"][1]["top_pct"] = 0.1
    bad_desc_both["metrics"]["strategy_bundle"]["top_pct"] = 0.1
    res_path.write_text(json.dumps(bad_desc_both), encoding="utf-8")
    with pytest.raises(ValueError, match="top_pct"):
        read_execution_result(
            res_path,
            command="factor-synthesis",
            run_id="sel-probe",
            execution_id="exec_sel_probe",
            source_sha256=cfg_hash,
        )
    with pytest.raises(ValueError, match="top_pct"):
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="sel-probe",
            execution_id="exec_sel_probe",
            source_sha256=cfg_hash,
            config_payload=cfg_payload,
            config_path=cfg_path,
        )

    # 2b. Mutate ONLY primary strategy_bundle top_pct or omit selection fields on candidate descriptor -> must fail!
    bad_primary_only = json.loads(json.dumps(good_payload))
    bad_primary_only["metrics"]["strategy_bundle"]["top_pct"] = 0.1
    res_path.write_text(json.dumps(bad_primary_only), encoding="utf-8")
    with pytest.raises(ValueError, match="top_pct"):
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="sel-probe",
            execution_id="exec_sel_probe",
            source_sha256=cfg_hash,
            config_payload=cfg_payload,
            config_path=cfg_path,
        )

    stripped_desc = json.loads(json.dumps(good_payload))
    stripped_desc["metrics"]["strategy_bundles"][1].pop("top_pct")
    stripped_desc["metrics"]["strategy_bundle"].pop("top_pct")
    res_path.write_text(json.dumps(stripped_desc), encoding="utf-8")
    with pytest.raises(ValueError, match="top_pct"):
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="sel-probe",
            execution_id="exec_sel_probe",
            source_sha256=cfg_hash,
            config_payload=cfg_payload,
            config_path=cfg_path,
        )

    # 3. Manifest itself has trial_id=t2 but wrong selection param (top_pct=0.1) with valid hashes -> must fail against config!
    mismatched_desc2, mismatched_art2 = _build_bundle("strategy_bundles/t2_wrong_pct", tid=t2, top_pct=0.1)
    bad_manifest_cfg = {
        **exec_base,
        "metrics": {
            "backtest_runs": [run1, run2],
            "strategy_bundles": [desc1, mismatched_desc2],
            "strategy_bundle": dict(mismatched_desc2),
        },
        "artifacts": [art1, mismatched_art2],
    }
    res_path.write_text(json.dumps(bad_manifest_cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="top_pct mismatch"):
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="sel-probe",
            execution_id="exec_sel_probe",
            source_sha256=cfg_hash,
            config_payload=cfg_payload,
            config_path=cfg_path,
        )

    # 4. top_k mode config: reject top_k mismatch in backtest_runs or descriptors while keeping trial_id!
    topk_cfg = {
        "synthesis_methods": {"ws": {"method": "weighted_sum"}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True, "top_k": 5}},
        "output_settings": {"composite_factor_prefix": "P"},
    }
    topk_dir = tmp_path / "exec_topk_probe"
    topk_dir.mkdir()
    topk_cfg_path = topk_dir / "config.json"
    topk_cfg_path.write_text(json.dumps(topk_cfg), encoding="utf-8")
    topk_hash = config_sha256(topk_cfg_path)
    topk_expected = expected_backtest_trials(topk_cfg, config_path=topk_cfg_path)
    assert len(topk_expected) == 1
    tk_tid = topk_expected[0]["trial_id"]
    topk_exec_base = {
        "schema_version": "execution_result.v1",
        "command": "factor-synthesis",
        "run_id": "topk-probe",
        "execution_id": "exec_topk_probe",
        "config_sha256": topk_hash,
        "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
        "metrics": {},
        "artifacts": [],
    }
    topk_exec_snap = tmp_path / "topk_exec_snap.json"
    topk_exec_snap.write_text(json.dumps(topk_exec_base), encoding="utf-8")
    tk_bdir = topk_dir / "strategy_bundle"
    tk_bundle = replace(
        bundle,
        rebalance_config={
            "rebalance_freq": "weekly",
            "selection_mode": "top_k",
            "top_pct": None,
            "top_k": 5,
            "synthesis_name": "ws",
            "trial_id": tk_tid,
        },
        execution_result_sha256=sha256_file(topk_exec_snap),
        bundle_hash="",
    )
    write_strategy_bundle(tk_bundle, tk_bdir, source_files={**sources, "provenance/execution_result.json": topk_exec_snap})
    tk_mpath = tk_bdir / "manifest.json"
    tk_desc = {
        "schema_version": tk_bundle.schema_version,
        "synthesis_name": "ws",
        "trial_id": tk_tid,
        "selection_mode": "top_k",
        "top_pct": None,
        "top_k": 5,
        "bundle_dir": str(tk_bdir),
        "manifest_path": str(tk_mpath),
        "bundle_hash": tk_bundle.bundle_hash,
    }
    tk_art = {
        "artifact_type": "strategy_bundle",
        "path": str(tk_mpath),
        "sha256": config_sha256(tk_mpath),
        "bundle_dir": str(tk_bdir),
        "bundle_hash": tk_bundle.bundle_hash,
        "schema_version": tk_bundle.schema_version,
        "synthesis_name": "ws",
        "trial_id": tk_tid,
        "selection_mode": "top_k",
        "top_pct": None,
        "top_k": 5,
    }
    tk_run = {
        "synthesis_name": "ws",
        "factor_name": "P_ws",
        "trial_id": tk_tid,
        "selection_mode": "top_k",
        "top_pct": None,
        "top_k": 5,
        "annual_return": 0.11,
        "sharpe_ratio": 1.05,
        "max_drawdown": -0.05,
    }
    tk_good = {
        **topk_exec_base,
        "metrics": {
            "backtest_runs": [tk_run],
            "strategy_bundles": [tk_desc],
            "strategy_bundle": dict(tk_desc),
        },
        "artifacts": [tk_art],
    }
    tk_res = topk_dir / "result.json"
    tk_res.write_text(json.dumps(tk_good), encoding="utf-8")
    assert (
        harness_runner._bound_result_metrics(
            tk_res,
            action="factor-synthesis",
            run_id="topk-probe",
            execution_id="exec_topk_probe",
            source_sha256=topk_hash,
            config_payload=topk_cfg,
            config_path=topk_cfg_path,
        )["execution"]["status"]
        == "succeeded"
    )
    tk_bad = json.loads(json.dumps(tk_good))
    tk_bad["metrics"]["strategy_bundles"][0]["top_k"] = 3
    tk_bad["metrics"]["strategy_bundle"]["top_k"] = 3
    tk_res.write_text(json.dumps(tk_bad), encoding="utf-8")
    with pytest.raises(ValueError, match="top_k"):
        harness_runner._bound_result_metrics(
            tk_res,
            action="factor-synthesis",
            run_id="topk-probe",
            execution_id="exec_topk_probe",
            source_sha256=topk_hash,
            config_payload=topk_cfg,
            config_path=topk_cfg_path,
        )


def test_strategy_bundle_rejects_unloadable_lightgbm_content(tmp_path: Path) -> None:
    """Round 3 Point 3: reject LightGBM bundle whose model content has valid sha256 but cannot load as lgb.Booster."""
    bundle, sources = _make_sample_strategy_bundle(tmp_path)
    bad_lgb_json = tmp_path / "bad_lgb_bundle.json"
    bad_lgb_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "model_type": "lightgbm",
                "index": {
                    "models": [
                        {
                            "train_end": "2023-02-01",
                            "model_file": "lgb_model_2023-02-01.txt",
                            "feature_names": ["alpha", "beta"],
                        }
                    ]
                },
                "models": {"lgb_model_2023-02-01.txt": "metadata only"},
            }
        ),
        encoding="utf-8",
    )
    bad_lgb_bundle = replace(
        bundle,
        synthesis_method="lightgbm",
        model_artifacts=(
            BundleArtifactRef(
                path="models/lgb_bundle.json",
                sha256=sha256_file(bad_lgb_json),
                artifact_type="model_bundle",
                size_bytes=bad_lgb_json.stat().st_size,
            ),
        ),
        bundle_hash="",
    )
    with pytest.raises(StrategyBundleError, match="LightGBM Booster"):
        write_strategy_bundle(
            bad_lgb_bundle,
            tmp_path / "bad_lgb_out",
            source_files={**sources, "models/lgb_bundle.json": bad_lgb_json},
        )

    # Also verify verify_strategy_bundle fails on an existing bundle directory with valid file hashes but unloadable content.
    import lightgbm as lgb

    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    y = np.array([0.1, 0.2, 0.3, 0.4])
    booster = lgb.train(
        {"objective": "regression", "verbosity": -1, "num_leaves": 2, "min_data_in_leaf": 1},
        lgb.Dataset(X, label=y),
        num_boost_round=1,
    )
    good_lgb_json = tmp_path / "good_lgb_bundle.json"
    good_lgb_json.write_text(
        json.dumps(
            {
                "version": "1.0",
                "model_type": "lightgbm",
                "index": {
                    "models": [
                        {
                            "train_end": "2023-02-01",
                            "model_file": "lgb_model_2023-02-01.txt",
                            "feature_names": ["alpha", "beta"],
                        }
                    ]
                },
                "models": {"lgb_model_2023-02-01.txt": booster.model_to_string()},
            }
        ),
        encoding="utf-8",
    )
    good_lgb_bundle = replace(
        bundle,
        synthesis_method="lightgbm",
        model_artifacts=(
            BundleArtifactRef(
                path="models/lgb_bundle.json",
                sha256=sha256_file(good_lgb_json),
                artifact_type="model_bundle",
                size_bytes=good_lgb_json.stat().st_size,
            ),
        ),
        bundle_hash="",
    )
    valid_dir = write_strategy_bundle(
        good_lgb_bundle,
        tmp_path / "valid_lgb_out",
        source_files={**sources, "models/lgb_bundle.json": good_lgb_json},
    )
    assert verify_strategy_bundle(valid_dir).synthesis_method == "lightgbm"

    # Mutate content + update file sha256/size + recompute bundle_hash -> verify_strategy_bundle must still reject!
    from factor_synthesis.strategy_bundle import compute_bundle_hash

    (valid_dir / "models/lgb_bundle.json").write_bytes(bad_lgb_json.read_bytes())
    manifest = json.loads((valid_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["model_artifacts"][0]["sha256"] = sha256_file(valid_dir / "models/lgb_bundle.json")
    manifest["model_artifacts"][0]["size_bytes"] = (valid_dir / "models/lgb_bundle.json").stat().st_size
    manifest["bundle_hash"] = compute_bundle_hash(manifest)
    (valid_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(StrategyBundleError, match="LightGBM Booster"):
        verify_strategy_bundle(valid_dir)


def test_strategy_bundle_rejects_incomplete_lasso_parent_expected_model_count(tmp_path: Path) -> None:
    """Round 3 Point 4: reject LASSO bundle with complete parent parquet refs when expected_model_count disagrees with models."""
    from factor_synthesis.strategy_bundle import compute_bundle_hash

    bundle, sources = _make_sample_strategy_bundle(tmp_path)
    pb_file = tmp_path / "parent_baseline.parquet"
    pb_idx = pd.MultiIndex.from_product([[pd.Timestamp("2023-01-03")], ["A", "B"]], names=["date", "stock_code"])
    pd.DataFrame({"composite_value": [1.0, 2.0]}, index=pb_idx).to_parquet(pb_file)
    pb_sha = sha256_file(pb_file)

    def _make_lasso_json(expected_count: int) -> Path:
        p = tmp_path / f"lasso_exp_{expected_count}.json"
        p.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "model_type": "lasso",
                    "index": {
                        "requires_parent_baseline": True,
                        "expected_model_count": expected_count,
                        "models": [
                            {
                                "train_end": "2023-02-01",
                                "weights": {"alpha": 0.5},
                                "feature_names": ["alpha"],
                                "parent_baseline": {"path": "parent_baseline.parquet", "sha256": pb_sha},
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        return p

    incomplete_json = _make_lasso_json(2)
    incomplete_bundle = replace(
        bundle,
        synthesis_method="lasso",
        model_artifacts=(
            BundleArtifactRef(
                path="models/lasso_model_bundle.json",
                sha256=sha256_file(incomplete_json),
                artifact_type="model_bundle",
                size_bytes=incomplete_json.stat().st_size,
            ),
            BundleArtifactRef(
                path="models/parent_baseline.parquet",
                sha256=pb_sha,
                artifact_type="parent_baseline",
                size_bytes=pb_file.stat().st_size,
            ),
        ),
        bundle_hash="",
    )
    with pytest.raises(StrategyBundleError, match="incomplete parent bundle"):
        write_strategy_bundle(
            incomplete_bundle,
            tmp_path / "incomplete_lasso_out",
            source_files={
                **sources,
                "models/lasso_model_bundle.json": incomplete_json,
                "models/parent_baseline.parquet": pb_file,
            },
        )

    complete_json = _make_lasso_json(1)
    complete_bundle = replace(
        incomplete_bundle,
        model_artifacts=(
            BundleArtifactRef(
                path="models/lasso_model_bundle.json",
                sha256=sha256_file(complete_json),
                artifact_type="model_bundle",
                size_bytes=complete_json.stat().st_size,
            ),
            incomplete_bundle.model_artifacts[1],
        ),
        bundle_hash="",
    )
    out_dir = write_strategy_bundle(
        complete_bundle,
        tmp_path / "complete_lasso_out",
        source_files={
            **sources,
            "models/lasso_model_bundle.json": complete_json,
            "models/parent_baseline.parquet": pb_file,
        },
    )
    assert verify_strategy_bundle(out_dir).synthesis_method == "lasso"

    # Mutate expected_model_count=2 + recompute hashes -> verify_strategy_bundle must reject!
    (out_dir / "models/lasso_model_bundle.json").write_bytes(incomplete_json.read_bytes())
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["model_artifacts"][0]["sha256"] = sha256_file(out_dir / "models/lasso_model_bundle.json")
    manifest["model_artifacts"][0]["size_bytes"] = (out_dir / "models/lasso_model_bundle.json").stat().st_size
    manifest["bundle_hash"] = compute_bundle_hash(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(StrategyBundleError, match="incomplete parent bundle"):
        verify_strategy_bundle(out_dir)


def test_harness_rejects_near_equal_top_pct_crossing_selection_count_boundary(tmp_path: Path) -> None:
    """Round 5 Point 1: reject near-equal top_pct (e.g. 0.19999999999 vs 0.2) that crosses selection_count boundaries."""
    from tools.agent_harness import runner as harness_runner
    from tools.agent_harness.run_result import expected_backtest_trials
    from utils.strategy_kernel import selection_count

    assert selection_count(10, top_pct=0.2) == 2
    assert selection_count(10, top_pct=0.19999999999) == 1
    assert selection_count(10, top_pct=0.19999999999999998) == 1

    bundle, sources = _make_sample_strategy_bundle(tmp_path)
    cfg_payload = {
        "synthesis_methods": {"ws": {"method": "weighted_sum"}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True, "top_pct": 0.2}},
        "output_settings": {"composite_factor_prefix": "P"},
    }
    exec_dir = tmp_path / "exec_near_pct"
    exec_dir.mkdir()
    cfg_path = exec_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg_payload), encoding="utf-8")
    cfg_hash = config_sha256(cfg_path)

    expected = expected_backtest_trials(cfg_payload, config_path=cfg_path)
    assert len(expected) == 1
    tid = expected[0]["trial_id"]

    exec_base = {
        "schema_version": "execution_result.v1",
        "command": "factor-synthesis",
        "run_id": "near-pct",
        "execution_id": "exec_near_pct",
        "config_sha256": cfg_hash,
        "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
        "metrics": {},
        "artifacts": [],
    }
    exec_snap = tmp_path / "near_pct_exec_snap.json"
    exec_snap.write_text(json.dumps(exec_base), encoding="utf-8")
    exec_sha = sha256_file(exec_snap)

    def _build_bundle(sub_dir: str, *, pct_value: float) -> tuple[dict, dict]:
        b_dir = exec_dir / sub_dir
        tb = replace(
            bundle,
            rebalance_config={
                "rebalance_freq": "weekly",
                "selection_mode": "top_pct",
                "top_pct": pct_value,
                "top_k": None,
                "synthesis_name": "ws",
                "trial_id": tid,
            },
            execution_result_sha256=exec_sha,
            bundle_hash="",
        )
        write_strategy_bundle(tb, b_dir, source_files={**sources, "provenance/execution_result.json": exec_snap})
        m_path = b_dir / "manifest.json"
        desc = {
            "schema_version": tb.schema_version,
            "synthesis_name": "ws",
            "trial_id": tid,
            "selection_mode": "top_pct",
            "top_pct": pct_value,
            "top_k": None,
            "bundle_dir": str(b_dir),
            "manifest_path": str(m_path),
            "bundle_hash": tb.bundle_hash,
        }
        art = {
            "artifact_type": "strategy_bundle",
            "path": str(m_path),
            "sha256": config_sha256(m_path),
            "bundle_dir": str(b_dir),
            "bundle_hash": tb.bundle_hash,
            "schema_version": tb.schema_version,
            "synthesis_name": "ws",
            "trial_id": tid,
            "selection_mode": "top_pct",
            "top_pct": pct_value,
            "top_k": None,
        }
        return desc, art

    desc_ok, art_ok = _build_bundle("strategy_bundle_ok", pct_value=0.2)
    run_ok = {
        "synthesis_name": "ws",
        "factor_name": "P_ws",
        "trial_id": tid,
        "selection_mode": "top_pct",
        "top_pct": 0.2,
        "top_k": None,
        "annual_return": 0.12,
        "sharpe_ratio": 1.1,
        "max_drawdown": -0.04,
        "return_series_artifact": {"trial_id": tid, "selection_mode": "top_pct", "top_pct": 0.2, "top_k": None},
    }
    good_payload = {
        **exec_base,
        "metrics": {
            "backtest_runs": [run_ok],
            "strategy_bundles": [desc_ok],
            "strategy_bundle": dict(desc_ok),
        },
        "artifacts": [art_ok],
    }
    res_path = exec_dir / "result.json"
    res_path.write_text(json.dumps(good_payload), encoding="utf-8")
    assert (
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="near-pct",
            execution_id="exec_near_pct",
            source_sha256=cfg_hash,
            config_payload=cfg_payload,
            config_path=cfg_path,
        )["execution"]["status"]
        == "succeeded"
    )

    for near_pct in (0.19999999999, 0.19999999999999998, 0.20000000001):
        # 1. Backtest record changed to near_pct while keeping trial_id -> must fail!
        bad_run = json.loads(json.dumps(good_payload))
        bad_run["metrics"]["backtest_runs"][0]["top_pct"] = near_pct
        res_path.write_text(json.dumps(bad_run), encoding="utf-8")
        with pytest.raises(ValueError, match="top_pct mismatch"):
            harness_runner._bound_result_metrics(
                res_path,
                action="factor-synthesis",
                run_id="near-pct",
                execution_id="exec_near_pct",
                source_sha256=cfg_hash,
                config_payload=cfg_payload,
                config_path=cfg_path,
            )

        # 2. Embedded return_series_artifact changed to near_pct while keeping trial_id -> must fail!
        bad_ret_art = json.loads(json.dumps(good_payload))
        bad_ret_art["metrics"]["backtest_runs"][0]["return_series_artifact"]["top_pct"] = near_pct
        res_path.write_text(json.dumps(bad_ret_art), encoding="utf-8")
        with pytest.raises(ValueError, match="top_pct mismatch"):
            harness_runner._bound_result_metrics(
                res_path,
                action="factor-synthesis",
                run_id="near-pct",
                execution_id="exec_near_pct",
                source_sha256=cfg_hash,
                config_payload=cfg_payload,
                config_path=cfg_path,
            )

        # 3. Both strategy_bundles[0] and primary strategy_bundle changed to near_pct -> must fail!
        bad_desc = json.loads(json.dumps(good_payload))
        bad_desc["metrics"]["strategy_bundles"][0]["top_pct"] = near_pct
        bad_desc["metrics"]["strategy_bundle"]["top_pct"] = near_pct
        res_path.write_text(json.dumps(bad_desc), encoding="utf-8")
        with pytest.raises(ValueError, match="top_pct"):
            read_execution_result(
                res_path,
                command="factor-synthesis",
                run_id="near-pct",
                execution_id="exec_near_pct",
                source_sha256=cfg_hash,
            )
        with pytest.raises(ValueError, match="top_pct"):
            harness_runner._bound_result_metrics(
                res_path,
                action="factor-synthesis",
                run_id="near-pct",
                execution_id="exec_near_pct",
                source_sha256=cfg_hash,
                config_payload=cfg_payload,
                config_path=cfg_path,
            )

    # 4. Manifest rebalance_config + descriptors all use 0.19999999999 with valid hashes & original trial_id -> must fail against config!
    desc_near, art_near = _build_bundle("strategy_bundle_near", pct_value=0.19999999999)
    bad_manifest_near = {
        **exec_base,
        "metrics": {
            "backtest_runs": [run_ok],
            "strategy_bundles": [desc_near],
            "strategy_bundle": dict(desc_near),
        },
        "artifacts": [art_near],
    }
    res_path.write_text(json.dumps(bad_manifest_near), encoding="utf-8")
    with pytest.raises(ValueError, match="top_pct mismatch"):
        harness_runner._bound_result_metrics(
            res_path,
            action="factor-synthesis",
            run_id="near-pct",
            execution_id="exec_near_pct",
            source_sha256=cfg_hash,
            config_payload=cfg_payload,
            config_path=cfg_path,
        )


def test_explicit_null_top_k_uses_default_5_across_candidates_backtest_return_artifact_and_harness(
    tmp_path: Path,
) -> None:
    """Round 5 Point 2: explicit top_k=None without top_pct resolves to default 5 across candidate gen, backtest, return artifact, and Harness."""
    from tools.agent_harness.run_result import DEFAULT_TOP_K, expected_backtest_trials

    assert DEFAULT_TOP_K == 5
    cfg_null = {
        "synthesis_methods": {"ws_c": {"method": "weighted_sum"}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True, "top_k": None}},
        "output_settings": {"composite_factor_prefix": "C"},
    }
    cfg_omitted = {
        "synthesis_methods": {"ws_c": {"method": "weighted_sum"}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True}},
        "output_settings": {"composite_factor_prefix": "C"},
    }
    exp_null = expected_backtest_trials(cfg_null)
    exp_omitted = expected_backtest_trials(cfg_omitted)
    assert exp_null == exp_omitted
    assert len(exp_null) == 1
    assert exp_null[0]["selection_mode"] == "top_k"
    assert exp_null[0]["top_pct"] is None
    assert exp_null[0]["top_k"] == 5

    # Runner direct construction with explicit top_k=None or conflicting (top_k, top_pct) stays aligned.
    runner_null = VectorBTBacktestRunner({"top_k_strategy": {"top_k": None, "top_pct": None}})
    assert runner_null.top_k == 5
    assert runner_null.top_pct is None
    runner_both = VectorBTBacktestRunner({"top_k_strategy": {"top_k": 3, "top_pct": 0.2}})
    assert runner_both.top_k == 3
    assert runner_both.top_pct is None

    # End-to-end Harness CLI run with explicit top_k=null (no top_pct) + return_series_artifact export.
    bars_csv = tmp_path / "null_topk_bars.csv"
    store_root = tmp_path / "null_topk_store"
    hours = pd.date_range("2026-01-03 00:00:00Z", "2026-01-04 23:00:00Z", freq="1h")
    syms = [f"COIN{i}-USD" for i in range(1, 7)]
    bar_rows = []
    for j, sym in enumerate(syms):
        for k, ts in enumerate(hours):
            px = 100.0 * (j + 1) + 0.5 * k
            bar_rows.append(
                {
                    "symbol": sym,
                    "time": (ts + pd.Timedelta(hours=1)).isoformat(),
                    "open": px,
                    "high": px + 1.0,
                    "low": px - 1.0,
                    "close": px + 0.2,
                    "volume": 100.0,
                }
            )
    pd.DataFrame(bar_rows).to_csv(bars_csv, index=False)

    slot_cfg = {
        "schema_version": 1,
        "slot_id": "crypto_null_topk",
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "crypto_utc_null_topk",
                "start": "2026-01-03",
                "end": "2026-01-04",
                "custom_sessions": {
                    "timezone": "UTC",
                    "weekdays": [0, 1, 2, 3, 4, 5, 6],
                    "segments": [{"open": "00:00", "close": "00:00", "open_day_offset": 0, "close_day_offset": 1}],
                },
            },
        },
        "bar_feed": {
            "provider": "csv_files",
            "options": {
                "paths": ["null_topk_bars.csv"],
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
                "instruments": {s: {"asset_class": "crypto", "currency": "USD"} for s in syms},
                "universes": {"spot": syms},
            },
        },
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {
            "entry_price_field": "open",
            "exit_price_field": "open",
            "min_entry_offset_bars": 1,
            "exit_adjustment": 0,
            "return_unit": "decimal",
            "gap_rule": "endpoints_only",
            "boundary_policy": "cross_session",
            "allow_short": True,
            "commission_bps": 0.0,
            "sell_tax_bps": 0.0,
            "slippage_bps": 0.0,
            "benchmark_instrument": None,
        },
    }
    slot_path = tmp_path / "null_topk_slot.json"
    slot_path.write_text(json.dumps(slot_cfg), encoding="utf-8")
    slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
    grid = slot.calendar.bar_grid("1h")
    f_rows = grid.merge(pd.DataFrame({"instrument_id": syms}), how="cross")
    f_vals = np.tile(np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0]), len(grid))
    _save_scheme_b_factor(
        store_root,
        slot_id="crypto_null_topk",
        slot_config_hash=slot.slot_config_hash,
        calendar=slot.calendar,
        factor_id="cmom",
        frequency="1h",
        frame=f_rows.assign(factor_value=f_vals),
    )

    synth_cfg = {
        "analysis_period": {"start_date": "2026-01-03", "end_date": "2026-01-04"},
        "factor_groups": {"c": {"factors": ["cmom"]}},
        "synthesis_methods": {
            "ws_c": {
                "method": "weighted_sum",
                "factor_group": "c",
                "weights": {"cmom": 1.0},
                "preprocess": {"cmom": "none"},
            }
        },
        "performance_evaluation": {
            "top_k_strategy": {"enabled": True, "top_k": None, "holding_period": 6, "rebalance_freq": "bar"}
        },
        "output_settings": {"composite_factor_prefix": "C"},
        "cache_settings": {"use_parquet_cache": False},
    }
    synth_path = tmp_path / "null_topk_synth.json"
    synth_path.write_text(json.dumps(synth_cfg), encoding="utf-8")
    expected_e2e = expected_backtest_trials(synth_cfg, config_path=synth_path)
    assert len(expected_e2e) == 1
    expected_tid = expected_e2e[0]["trial_id"]
    assert expected_e2e[0]["top_k"] == 5
    assert expected_e2e[0]["top_pct"] is None

    h_root = tmp_path / "null_topk_harness"
    rc = harness_cli.main([
        "factor-synthesis",
        "--config", str(synth_path),
        "--slot-config", str(slot_path),
        "--slot-id", "crypto_null_topk",
        "--factor-values-store", str(store_root),
        "--frequency", "1h",
        "--output-root", str(h_root),
        "--run-id", "null-topk-e2e",
    ])
    assert rc == 0
    summary = json.loads((h_root / "factor-synthesis" / "null-topk-e2e" / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["metrics"]["execution"]["status"] == "succeeded"
    runs = summary["metrics"]["backtest_runs"]
    assert len(runs) == 1
    assert runs[0]["trial_id"] == expected_tid
    assert runs[0]["selection_mode"] == "top_k"
    assert runs[0]["top_k"] == 5
    assert runs[0]["top_pct"] is None

    ret_art = runs[0]["return_series_artifact"]
    assert isinstance(ret_art, dict)
    assert ret_art["trial_id"] == expected_tid
    assert ret_art["selection_mode"] == "top_k"
    assert ret_art["top_k"] == 5
    assert ret_art["top_pct"] is None
    assert Path(ret_art["path"]).is_file()
    ret_meta = json.loads(Path(ret_art["metadata_path"]).read_text(encoding="utf-8"))
    assert ret_meta["trial_id"] == expected_tid
    assert ret_meta["selection_mode"] == "top_k"
    assert ret_meta["top_k"] == 5
    assert ret_meta["top_pct"] is None

    sb_desc = summary["metrics"]["strategy_bundle"]
    assert sb_desc["trial_id"] == expected_tid
    assert sb_desc["selection_mode"] == "top_k"
    assert sb_desc["top_k"] == 5
    assert sb_desc["top_pct"] is None
    loaded = verify_strategy_bundle(sb_desc["bundle_dir"])
    assert loaded.rebalance_config["trial_id"] == expected_tid
    assert loaded.rebalance_config["selection_mode"] == "top_k"
    assert loaded.rebalance_config["top_k"] == 5
    assert loaded.rebalance_config["top_pct"] is None


def test_top_k_takes_priority_over_scalar_and_list_top_pct_across_entries_and_e2e(
    tmp_path: Path,
) -> None:
    """Round 6: top_k > top_pct > DEFAULT_TOP_K holds before candidate expansion for both scalar and list top_pct."""
    from tools.agent_harness import runner as harness_runner
    from tools.agent_harness.run_result import (
        build_backtest_trial_id,
        expected_backtest_trials,
        resolve_backtest_candidates,
        resolve_candidate_selection,
    )
    from utils.strategy_kernel import selection_count

    assert selection_count(10, top_k=3, top_pct=0.2) == 3
    assert selection_count(10, top_k=None, top_pct=0.2) == 2

    assert resolve_candidate_selection(top_pct=0.2, top_k=3) == ("top_k", None, 3)
    assert resolve_backtest_candidates({"enabled": True, "top_k": 3, "top_pct": 0.2}) == [
        ("top_k", None, 3)
    ]
    assert resolve_backtest_candidates({"enabled": True, "top_k": 3, "top_pct": [0.1, 0.2]}) == [
        ("top_k", None, 3)
    ]

    def _base_cfg(top_k_strategy: dict) -> dict:
        return {
            "analysis_period": {"start_date": "2026-01-03", "end_date": "2026-01-04"},
            "factor_groups": {"c": {"factors": ["cmom"]}},
            "synthesis_methods": {
                "ws_p": {
                    "method": "weighted_sum",
                    "factor_group": "c",
                    "weights": {"cmom": 1.0},
                    "preprocess": {"cmom": "none"},
                }
            },
            "performance_evaluation": {"top_k_strategy": top_k_strategy},
            "output_settings": {"composite_factor_prefix": "P"},
            "cache_settings": {"use_parquet_cache": False},
        }

    exp_only = expected_backtest_trials(
        _base_cfg({"enabled": True, "top_k": 3})
    )
    exp_scalar = expected_backtest_trials(
        _base_cfg({"enabled": True, "top_k": 3, "top_pct": 0.2})
    )
    exp_list = expected_backtest_trials(
        _base_cfg({"enabled": True, "top_k": 3, "top_pct": [0.1, 0.2]})
    )
    assert exp_scalar == exp_only
    assert exp_list == exp_only
    assert len(exp_only) == 1
    assert exp_only[0]["selection_mode"] == "top_k"
    assert exp_only[0]["top_k"] == 3
    assert exp_only[0]["top_pct"] is None

    for runner_cfg in (
        {"top_k_strategy": {"top_k": 3, "top_pct": 0.2}},
        {"top_k_strategy": {"top_k": 3, "top_pct": [0.1, 0.2]}},
    ):
        direct = VectorBTBacktestRunner(runner_cfg)
        assert direct.top_k == 3
        assert direct.top_pct is None
        assert direct._get_effective_top_k(10) == 3

    # Build a 10-instrument declarative 24/7 slot where top_k=3 selects 3 instruments while top_pct=0.2 selects 2.
    bars_csv = tmp_path / "prio_bars.csv"
    store_root = tmp_path / "prio_store"
    hours = pd.date_range("2026-01-03 00:00:00Z", "2026-01-04 23:00:00Z", freq="1h")
    syms = [f"COIN{i:02d}-USD" for i in range(1, 11)]
    bar_rows = []
    for j, sym in enumerate(syms):
        slope = 0.2 * (j + 1) * ((-1) ** j)
        for k, ts in enumerate(hours):
            px = 100.0 + 10.0 * j + slope * k + 0.05 * ((k % 3) - 1) * (j + 1)
            bar_rows.append(
                {
                    "symbol": sym,
                    "time": (ts + pd.Timedelta(hours=1)).isoformat(),
                    "open": px,
                    "high": px + 1.0,
                    "low": px - 1.0,
                    "close": px + 0.1 * (j + 1),
                    "volume": 100.0,
                }
            )
    pd.DataFrame(bar_rows).to_csv(bars_csv, index=False)

    slot_cfg = {
        "schema_version": 1,
        "slot_id": "crypto_prio",
        "calendar": {
            "provider": "custom_sessions",
            "options": {
                "calendar_id": "crypto_utc_prio",
                "start": "2026-01-03",
                "end": "2026-01-04",
                "custom_sessions": {
                    "timezone": "UTC",
                    "weekdays": [0, 1, 2, 3, 4, 5, 6],
                    "segments": [{"open": "00:00", "close": "00:00", "open_day_offset": 0, "close_day_offset": 1}],
                },
            },
        },
        "bar_feed": {
            "provider": "csv_files",
            "options": {
                "paths": ["prio_bars.csv"],
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
                "instruments": {s: {"asset_class": "crypto", "currency": "USD"} for s in syms},
                "universes": {"spot": syms},
            },
        },
        "aux_features": {"provider": "none"},
        "microstructure": {"provider": "open"},
        "execution_profile": {
            "entry_price_field": "open",
            "exit_price_field": "open",
            "min_entry_offset_bars": 1,
            "exit_adjustment": 0,
            "return_unit": "decimal",
            "gap_rule": "endpoints_only",
            "boundary_policy": "cross_session",
            "allow_short": True,
            "commission_bps": 0.0,
            "sell_tax_bps": 0.0,
            "slippage_bps": 0.0,
            "benchmark_instrument": None,
        },
    }
    slot_path = tmp_path / "prio_slot.json"
    slot_path.write_text(json.dumps(slot_cfg), encoding="utf-8")
    slot = build_declarative_slot(SlotConfig.from_mapping(slot_cfg), base_dir=tmp_path)
    grid = slot.calendar.bar_grid("1h")
    f_rows = grid.merge(pd.DataFrame({"instrument_id": syms}), how="cross")
    f_vals = np.tile(np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), len(grid))
    _save_scheme_b_factor(
        store_root,
        slot_id="crypto_prio",
        slot_config_hash=slot.slot_config_hash,
        calendar=slot.calendar,
        factor_id="cmom",
        frequency="1h",
        frame=f_rows.assign(factor_value=f_vals),
    )

    def _run_harness_case(label: str, top_k_strategy: dict) -> tuple[dict, np.ndarray]:
        cfg = _base_cfg(top_k_strategy)
        cfg_path = tmp_path / f"{label}_synth.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        h_root = tmp_path / f"{label}_harness"
        rc = harness_cli.main([
            "factor-synthesis",
            "--config", str(cfg_path),
            "--slot-config", str(slot_path),
            "--slot-id", "crypto_prio",
            "--factor-values-store", str(store_root),
            "--frequency", "1h",
            "--output-root", str(h_root),
            "--run-id", label,
        ])
        assert rc == 0
        summary = json.loads((h_root / "factor-synthesis" / label / "summary.json").read_text(encoding="utf-8"))
        assert summary["ok"] is True
        assert summary["metrics"]["execution"]["status"] == "succeeded"
        ret_csv = Path(summary["metrics"]["backtest_runs"][0]["return_series_artifact"]["path"])
        rets = pd.read_csv(ret_csv)["net_return"].to_numpy(dtype=float)
        return summary, rets

    sum_topk_only, rets_topk_only = _run_harness_case(
        "topk3_only", {"enabled": True, "top_k": 3, "holding_period": 6, "rebalance_freq": "bar"}
    )
    sum_scalar, rets_scalar = _run_harness_case(
        "topk3_scalar_pct",
        {"enabled": True, "top_k": 3, "top_pct": 0.2, "holding_period": 6, "rebalance_freq": "bar"},
    )
    sum_list, rets_list = _run_harness_case(
        "topk3_list_pct",
        {"enabled": True, "top_k": 3, "top_pct": [0.1, 0.2], "holding_period": 6, "rebalance_freq": "bar"},
    )
    _, rets_pct_only = _run_harness_case(
        "pct_02_only", {"enabled": True, "top_k": None, "top_pct": 0.2, "holding_period": 6, "rebalance_freq": "bar"}
    )

    assert not np.allclose(rets_topk_only, rets_pct_only, atol=1e-12)
    assert np.allclose(rets_scalar, rets_topk_only, atol=1e-12)
    assert np.allclose(rets_list, rets_topk_only, atol=1e-12)

    expected_tid = exp_only[0]["trial_id"]
    for label, summary in (("scalar", sum_scalar), ("list", sum_list)):
        runs = summary["metrics"]["backtest_runs"]
        assert len(runs) == 1, label
        assert runs[0]["trial_id"] == expected_tid
        assert runs[0]["selection_mode"] == "top_k"
        assert runs[0]["top_k"] == 3
        assert runs[0]["top_pct"] is None
        assert summary["metrics"]["strategy_bundles"] and len(summary["metrics"]["strategy_bundles"]) == 1

        ret_art = runs[0]["return_series_artifact"]
        assert ret_art["trial_id"] == expected_tid
        assert ret_art["selection_mode"] == "top_k"
        assert ret_art["top_k"] == 3
        assert ret_art["top_pct"] is None
        ret_meta = json.loads(Path(ret_art["metadata_path"]).read_text(encoding="utf-8"))
        assert ret_meta["trial_id"] == expected_tid
        assert ret_meta["selection_mode"] == "top_k"
        assert ret_meta["top_k"] == 3
        assert ret_meta["top_pct"] is None

        sb_desc = summary["metrics"]["strategy_bundle"]
        assert sb_desc["trial_id"] == expected_tid
        assert sb_desc["selection_mode"] == "top_k"
        assert sb_desc["top_k"] == 3
        assert sb_desc["top_pct"] is None
        loaded = verify_strategy_bundle(sb_desc["bundle_dir"])
        assert loaded.rebalance_config["trial_id"] == expected_tid
        assert loaded.rebalance_config["selection_mode"] == "top_k"
        assert loaded.rebalance_config["top_k"] == 3
        assert loaded.rebalance_config["top_pct"] is None

    # Harness must reject a result claiming top_pct=0.2 when config has both top_k=3 and top_pct.
    pct_trial_id = build_backtest_trial_id(
        factor_name="P_ws_p",
        selection_mode="top_pct",
        top_pct=0.2,
        top_k=None,
        synthesis_name="ws_p",
    )
    for bad_cfg in (
        _base_cfg({"enabled": True, "top_k": 3, "top_pct": 0.2}),
        _base_cfg({"enabled": True, "top_k": 3, "top_pct": [0.1, 0.2]}),
    ):
        bad_exec_dir = tmp_path / ("bad_scalar" if bad_cfg["performance_evaluation"]["top_k_strategy"]["top_pct"] == 0.2 else "bad_list")
        bad_exec_dir.mkdir(exist_ok=True)
        bad_cfg_path = bad_exec_dir / "cfg.json"
        bad_cfg_path.write_text(json.dumps(bad_cfg), encoding="utf-8")
        bad_cfg_hash = config_sha256(bad_cfg_path)
        bad_res_path = bad_exec_dir / "result.json"
        bad_res_path.write_text(
            json.dumps({
                "schema_version": "execution_result.v1",
                "command": "factor-synthesis",
                "run_id": "bad-prio",
                "execution_id": "exec_bad_prio",
                "config_sha256": bad_cfg_hash,
                "execution": {"expected": 1, "completed": 1, "failed": 0, "errors": [], "status": "succeeded"},
                "metrics": {
                    "backtest_runs": [
                        {
                            "synthesis_name": "ws_p",
                            "factor_name": "P_ws_p",
                            "trial_id": pct_trial_id,
                            "selection_mode": "top_pct",
                            "top_pct": 0.2,
                            "top_k": None,
                            "annual_return": 0.1,
                            "sharpe_ratio": 1.0,
                            "max_drawdown": -0.05,
                        }
                    ]
                },
                "artifacts": [],
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="backtest_candidate_unexpected"):
            harness_runner._bound_result_metrics(
                bad_res_path,
                action="factor-synthesis",
                run_id="bad-prio",
                execution_id="exec_bad_prio",
                source_sha256=bad_cfg_hash,
                config_payload=bad_cfg,
                config_path=bad_cfg_path,
            )
