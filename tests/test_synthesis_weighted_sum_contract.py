"""Regression tests for NERE-1401 weighted_sum contract ([二-3], [二-4], [二-9])."""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from factor_synthesis.core.factor_data_loader import (
    PANEL_DATE_ALIGNMENT_ATTR,
    PANEL_DATE_ALIGNMENT_RAW,
    PANEL_DATE_ALIGNMENT_TRADE_DATE,
    load_factor_data,
)
from factor_synthesis.core.factor_ops import apply_feature_preprocessing, calculate_forward_returns
from factor_synthesis.pipeline.preprocessing import bind_synthesis_panel
from factor_synthesis.pipeline.weighted_sum_method import run_weighted_sum_synthesis
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from market_db_samples import declare_market_db
from calendar_test_support import seed_calendar
from run_factor_synthesis import FactorSynthesizer
from slots.jq_ashare_local import build_jq_ashare_local
from utils import strategy_kernel as kernel
from utils.market_db.identity import bind_market_db_fingerprint, slot_market_db_identity
from utils.market_factor import (
    MarketFactorValueStore,
    build_identity_record,
    compute_factor_identity_hash,
)

LOG = logging.getLogger(__name__)
NO_CACHE = {"data_bindings": {}, "cache_settings": {"use_parquet_cache": False}}
STOCKS = ["000001.XSHE", "000002.XSHE", "000003.XSHE", "000004.XSHE"]
FACTORS = ["f_a", "f_b"]


def _seed_ashare_fixture(
    tmp_path: Path,
    *,
    start: str = "2026-01-05",
    end: str = "2026-01-23",
    raw_values: pd.DataFrame | None = None,
    stocks: list[str] | None = None,
    close_panel: pd.DataFrame | None = None,
):
    stock_list = list(stocks) if stocks is not None else STOCKS
    db_path = tmp_path / "ashare_contract.duckdb"
    store_root = tmp_path / "store"
    with duckdb.connect(str(db_path), config={"threads": 1}) as conn:
        declare_market_db(conn, "cn_equity")
        seed_calendar(conn, start, end)
        slot = build_jq_ashare_local(conn, start=start, end=end)
        sessions = slot.calendar.sessions(start, end)
        index = pd.MultiIndex.from_product([sessions, stock_list], names=["date", "stock_code"])
        if raw_values is None:
            rng = np.random.default_rng(1401)
            raw = pd.DataFrame(
                {
                    "f_a": np.arange(len(index), dtype=float) * 1.7 + rng.normal(0.0, 0.2, len(index)),
                    "f_b": np.sin(np.arange(len(index), dtype=float)) * 10.0 + 50.0,
                },
                index=index,
            )
        else:
            raw = raw_values.reindex(index)

        px_rows = raw.reset_index()
        if close_panel is not None:
            aligned_close = (
                close_panel.reindex(index=sessions, columns=stock_list)
                .stack(future_stack=True)
                .reindex(index)
                .to_numpy(dtype=float)
            )
            px_rows["close"] = aligned_close
        else:
            px_rows["day_idx"] = px_rows.groupby("stock_code").cumcount()
            px_rows["stock_idx"] = px_rows["stock_code"].map({s: k for k, s in enumerate(stock_list)})
            px_rows["close"] = 10.0 + 0.1 * px_rows["day_idx"] + 0.05 * px_rows["stock_idx"]
        px_rows["open"] = px_rows["close"] - 0.02
        px_rows["high"] = px_rows["close"] + 0.05
        px_rows["low"] = px_rows["close"] - 0.05
        px_rows["volume"] = 1000.0
        px_rows["factor"] = 1.0
        conn.execute(
            "CREATE TABLE price_data_daily AS "
            "SELECT date::DATE AS date, stock_code, open, high, low, close, volume, factor FROM px_rows"
        )
        legacy = raw.rename_axis(columns="factor_name").stack().rename("factor_value").reset_index()
        conn.execute("CREATE TABLE factor_values AS SELECT date::DATE AS date, stock_code, factor_name, factor_value FROM legacy")
        conn.execute(
            "CREATE TABLE factor_data AS "
            "SELECT date::DATE AS date, stock_code, 20.0 AS natural_log_of_market_cap FROM px_rows"
        )

    store = MarketFactorValueStore(store_root)
    grid = slot.calendar.bar_grid("1d", start, end)
    rows = grid.merge(pd.DataFrame({"instrument_id": stock_list}), how="cross")
    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(rows.session_id), rows.instrument_id], names=["date", "stock_code"]
    )
    identities = {}
    for factor_id in FACTORS:
        identity, payload = compute_factor_identity_hash(
            factor_id=factor_id,
            factor_class="tests.ContractFactor",
            source_sha256="a" * 64,
            params={},
            frequency="1d",
            session_view="ETH",
            analysis_mode="cross_sectional",
            bucket_strategy="session",
            calendar_hash=slot.calendar.calendar_hash,
            slot_id=slot.slot_id,
            slot_config_hash="b" * 64,
            adjustment_policy="none",
            slot_data_fingerprint=bind_market_db_fingerprint(
                {"fixture": "v1"}, slot_market_db_identity(slot)
            ),
            code_version="synthetic",
        )
        record = build_identity_record(
            factor_identity_hash=identity, payload=payload, code_version_source="test"
        )
        batch = rows.assign(
            factor_id=factor_id,
            factor_identity_hash=identity,
            frequency="1d",
            session_view="ETH",
            factor_value=raw[factor_id].reindex(keys).to_numpy(),
            source_build_id="fixture",
            calendar_hash=slot.calendar.calendar_hash,
            code_version="synthetic",
            created_at=pd.Timestamp("2026-02-01", tz="UTC"),
            batch_id="fixture",
        )
        store.append_factor_values(batch, identity_record=record, calendar=slot.calendar)
        identities[factor_id] = identity

    return db_path, store_root, slot, sessions, raw, identities


def _make_stub_synthesizer(tmp_path: Path, *, analysis_period: dict | None = None):
    captured: dict[str, pd.Series] = {}
    cfg = {
        **NO_CACHE,
        "factor_groups": {"all": {"factors": FACTORS}},
        "output_settings": {
            "composite_factor_prefix": "composite",
            "weights_output_dir": str(tmp_path / "weights"),
        },
    }
    if analysis_period is not None:
        cfg["analysis_period"] = analysis_period
    synth = SimpleNamespace(
        logger=LOG,
        config=cfg,
        _apply_feature_preprocessing=apply_feature_preprocessing,
        direct_analyze_composite_factor=lambda s, n, **kw: captured.update({n: s}),
    )
    return synth, captured


def test_weighted_sum_three_path_date_alignment_no_double_shift(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """[二-3] All 3 paths map trading day T strictly to T-1 raw signal (never T-2 or T)."""
    db_path, store_root, _, sessions, raw, identities = _seed_ashare_fixture(
        tmp_path, start="2026-01-05", end="2026-01-12"
    )
    # Assign distinct, non-collinear day levels per stock so T, T-1, T-2 are unmistakable.
    for day_idx, day in enumerate(sessions):
        for s_idx, stock in enumerate(STOCKS):
            raw.loc[(day, stock), "f_a"] = 100.0 * (day_idx + 1) + float(s_idx)
            raw.loc[(day, stock), "f_b"] = 10.0 * (day_idx + 1) - float(s_idx)

    # Re-seed with deterministic distinct daily values.
    tmp_sub = tmp_path / "distinct"
    tmp_sub.mkdir()
    db_path, store_root, _, sessions, raw, identities = _seed_ashare_fixture(
        tmp_sub, start="2026-01-05", end="2026-01-12", raw_values=raw
    )
    weights_cfg = {
        "factor_group": "all",
        "weights": {"f_a": 1.0, "f_b": 0.5},
        "preprocess": {"ALL": "none"},
    }

    # Path (a): Direct raw unaligned panel passed to run_weighted_sum_synthesis.
    raw_dir = tmp_sub / "path_a"
    synth_a, cap_a = _make_stub_synthesizer(raw_dir)
    raw_panel = raw.copy()
    raw_panel.attrs[PANEL_DATE_ALIGNMENT_ATTR] = PANEL_DATE_ALIGNMENT_RAW
    with caplog.at_level(logging.WARNING):
        run_weighted_sum_synthesis(synth_a, "ws", weights_cfg, raw_panel, slot_id="jq_ashare_local", frequency="1d")
    assert not any(PANEL_DATE_ALIGNMENT_ATTR in rec.message for rec in caplog.records)
    scores_a = cap_a["composite_ws"]

    # Path (a2): Direct unaligned panel WITHOUT attrs logs a warning and shifts once (T-1 -> T).
    raw_dir2 = tmp_sub / "path_a2"
    synth_a2, cap_a2 = _make_stub_synthesizer(raw_dir2)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run_weighted_sum_synthesis(synth_a2, "ws", weights_cfg, raw.copy(), slot_id="jq_ashare_local", frequency="1d")
    assert any(PANEL_DATE_ALIGNMENT_ATTR in rec.message for rec in caplog.records)
    pd.testing.assert_series_equal(cap_a2["composite_ws"], scores_a)

    # Path (b): load_factor_data (fresh + Parquet processed cache hit) -> run_weighted_sum_synthesis.
    cache_dir = tmp_sub / "factor_cache"
    cache_cfg = {"data_bindings": {}, "cache_settings": {"use_parquet_cache": True, "cache_dir": str(cache_dir)}}
    with duckdb.connect(str(db_path), read_only=True, config={"threads": 1}) as conn:
        slot = build_jq_ashare_local(conn, start="2026-01-05", end="2026-01-12")
        loaded_fresh = load_factor_data(
            None, "factor_values", FACTORS, "2026-01-05", "2026-01-12",
            config=cache_cfg, slot=slot, factor_values_store=store_root,
        )
        loaded_cached = load_factor_data(
            None, "factor_values", FACTORS, "2026-01-05", "2026-01-12",
            config=cache_cfg, slot=slot, factor_values_store=store_root,
        )
        assert loaded_fresh.attrs.get(PANEL_DATE_ALIGNMENT_ATTR) == PANEL_DATE_ALIGNMENT_TRADE_DATE
        assert loaded_cached.attrs.get(PANEL_DATE_ALIGNMENT_ATTR) == PANEL_DATE_ALIGNMENT_TRADE_DATE

        synth_b, cap_b = _make_stub_synthesizer(tmp_sub / "path_b")
        run_weighted_sum_synthesis(synth_b, "ws", weights_cfg, loaded_cached, slot=slot)
        scores_b = cap_b["composite_ws"]

        # Both path (a) and path (b) must have trading day T == 1.0*raw[T-1].f_a + 0.5*raw[T-1].f_b.
        expected_dates = sessions[1:]
        assert list(scores_b.index.get_level_values("date").unique()) == list(expected_dates)
        pd.testing.assert_series_equal(scores_b, scores_a)
        for idx_pos in range(1, len(sessions)):
            t_day = sessions[idx_pos]
            t_prev = sessions[idx_pos - 1]
            expected_t = raw.loc[t_prev, "f_a"] + 0.5 * raw.loc[t_prev, "f_b"]
            np.testing.assert_allclose(scores_b.loc[t_day].reindex(STOCKS), expected_t.reindex(STOCKS), atol=1e-12)

        # Path (c): Bundle replay queries T-1 raw factor for trading day T.
        bundle_path = tmp_sub / "path_b" / "weights" / "weighted_sum_bundle_ws.json"
        runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 2, "holding_period": 1, "rebalance_freq": "daily", "enable_tday_filter": False}},
            slot=slot,
            frequency="1d",
            bundle_path=str(bundle_path),
            factor_values_store=store_root,
            factor_identities=identities,
        )
        try:
            prices = conn.execute("SELECT date, stock_code, open, close FROM price_data_daily ORDER BY date, stock_code").fetchdf()
            prices["date"] = pd.to_datetime(prices["date"])
            close_df = prices.pivot(index="date", columns="stock_code", values="close").reindex(index=sessions, columns=STOCKS)
            open_df = prices.pivot(index="date", columns="stock_code", values="open").reindex(index=sessions, columns=STOCKS)
            score_wide = scores_b.unstack("stock_code").reindex(index=sessions, columns=STOCKS)
            res = runner.run_backtest(score_wide, close_df, open_df=open_df, rebalance_freq="daily")
            assert "error" not in res, res
            for idx_pos in range(1, len(sessions)):
                t_key = sessions[idx_pos].strftime("%Y-%m-%d")
                prev_key = runner._backtest_previous_dates[t_key]
                assert prev_key == sessions[idx_pos - 1].strftime("%Y-%m-%d")
                replayed, _, raw_fetched = runner._get_synthesis_score(prev_key, STOCKS)
                np.testing.assert_allclose(
                    raw_fetched["f_a"].reindex(STOCKS), raw.loc[sessions[idx_pos - 1], "f_a"].reindex(STOCKS), atol=1e-12
                )
                np.testing.assert_allclose(
                    replayed.reindex(STOCKS), scores_b.loc[sessions[idx_pos]].reindex(STOCKS), atol=1e-10
                )
        finally:
            runner.close()

    # Invalid alignment attr must fail closed in bind_synthesis_panel.
    bad_panel = raw.copy()
    bad_panel.attrs[PANEL_DATE_ALIGNMENT_ATTR] = "invalid_alignment"
    with pytest.raises(ValueError, match="panel_date_alignment"):
        bind_synthesis_panel(SimpleNamespace(), bad_panel, {}, {})


@pytest.mark.parametrize(
    ("preprocess_cfg", "extra_cfg"),
    [
        ({"ALL": "none"}, {}),
        ({"f_a": "none", "f_b": "zscore"}, {}),
        ({"ALL": "rank"}, {}),
        ({"ALL": "zscore"}, {}),
        ({"ALL": "rank_zscore"}, {}),
        ({"ALL": "zscore", "f_b": "none"}, {}),
        ({"ALL": "zscore"}, {"postprocess_cs_standardize": True}),
        ({"ALL": "rank"}, {"standardize": False}),
    ],
)
def test_weighted_sum_local_and_bundle_replay_exact_parity(
    tmp_path: Path, preprocess_cfg: dict, extra_cfg: dict
) -> None:
    """[二-4] Local weighted_sum and strategy_kernel / VectorBT bundle replay agree on scores, rank, and Top-K."""
    db_path, store_root, _, sessions, _, identities = _seed_ashare_fixture(
        tmp_path, start="2026-01-05", end="2026-01-16"
    )
    with duckdb.connect(str(db_path), read_only=True, config={"threads": 1}) as conn:
        slot = build_jq_ashare_local(conn, start="2026-01-05", end="2026-01-16")
        loaded = load_factor_data(
            None, "factor_values", FACTORS, "2026-01-05", "2026-01-16",
            config=NO_CACHE, slot=slot, factor_values_store=store_root,
        )
        synth, captured = _make_stub_synthesizer(tmp_path)
        weights_cfg = {
            "factor_group": "all",
            "weights": {"f_a": 0.7, "f_b": -0.3},
            "preprocess": preprocess_cfg,
            **extra_cfg,
        }
        run_weighted_sum_synthesis(synth, "ws", weights_cfg, loaded, slot=slot)
        local_scores = captured["composite_ws"]

        bundle_path = tmp_path / "weights" / "weighted_sum_bundle_ws.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        model = bundle["index"]["models"][0]
        assert model["standardize"] is bool(extra_cfg.get("standardize", True))
        assert model["postprocess_cs_standardize"] is bool(extra_cfg.get("postprocess_cs_standardize", False))
        assert "ALL" in model["standardization_method"]
        for fname in FACTORS:
            assert fname in model["standardization_method"]

        runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 2, "holding_period": 1, "rebalance_freq": "daily", "enable_tday_filter": False}},
            slot=slot,
            frequency="1d",
            bundle_path=str(bundle_path),
            factor_values_store=store_root,
            factor_identities=identities,
        )
        try:
            prices = conn.execute("SELECT date, stock_code, open, close FROM price_data_daily ORDER BY date, stock_code").fetchdf()
            prices["date"] = pd.to_datetime(prices["date"])
            close_df = prices.pivot(index="date", columns="stock_code", values="close").reindex(index=sessions, columns=STOCKS)
            open_df = prices.pivot(index="date", columns="stock_code", values="open").reindex(index=sessions, columns=STOCKS)
            score_wide = local_scores.unstack("stock_code").reindex(index=sessions, columns=STOCKS)
            res = runner.run_backtest(score_wide, close_df, open_df=open_df, rebalance_freq="daily")
            assert "error" not in res, res

            for trade_day in local_scores.index.get_level_values("date").unique():
                t_key = trade_day.strftime("%Y-%m-%d")
                prev_key = runner._backtest_previous_dates[t_key]
                replayed_runner, _, raw_cross = runner._get_synthesis_score(prev_key, STOCKS)
                replayed_kernel, _ = kernel.linear_signal(raw_cross, model)

                local_day = local_scores.loc[trade_day].reindex(STOCKS)
                np.testing.assert_allclose(replayed_kernel.reindex(STOCKS), local_day, atol=1e-10)
                np.testing.assert_allclose(replayed_runner.reindex(STOCKS), local_day, atol=1e-10)

                local_rank = kernel.rank_scores(local_day)
                replay_rank = kernel.rank_scores(replayed_kernel)
                assert replay_rank == local_rank
                k = kernel.selection_count(len(local_rank), top_k=2)
                assert replay_rank[:k] == local_rank[:k]
        finally:
            runner.close()


def test_strategy_kernel_py36_and_legacy_bundle_defaults() -> None:
    """Verify strategy_kernel.py parses under Python 3.6 and preserves legacy default semantics."""
    source = Path(kernel.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 6))

    raw = pd.DataFrame({"a": [1.0, 2.0, 100.0, 4.0]}, index=["S1", "S2", "S3", "S4"])
    # Explicit "none" keeps raw values and explicit postprocess_cs_standardize=False keeps scale.
    score_none, proc_none = kernel.linear_signal(
        raw,
        {
            "weights": {"a": 2.0},
            "standardization_method": {"ALL": "none"},
            "postprocess_cs_standardize": False,
        },
    )
    np.testing.assert_allclose(proc_none["a"], raw["a"], atol=1e-12)
    np.testing.assert_allclose(score_none, raw["a"] * 2.0, atol=1e-12)

    # Legacy bundle omitting standardization_method and postprocess_cs_standardize keeps zscore defaults.
    score_legacy, proc_legacy = kernel.linear_signal(raw, {"weights": {"a": 1.0}})
    expected_proc = kernel.preprocess_series(raw["a"], "zscore")
    np.testing.assert_allclose(proc_legacy["a"], expected_proc, atol=1e-12)
    np.testing.assert_allclose(
        score_legacy,
        (expected_proc - expected_proc.mean()) / (expected_proc.std() + 1e-8),
        atol=1e-12,
    )


@pytest.mark.parametrize("use_time_series", [False, True])
def test_weighted_sum_trims_warmup_before_analysis_period_start_date(
    tmp_path: Path, use_time_series: bool
) -> None:
    """[二-9] Warmup bars before analysis_period.start_date never leak into composite outputs or Alphalens IC."""
    cal_start, cal_end = "2026-01-05", "2026-02-13"
    analysis_start, analysis_end = "2026-01-19", "2026-02-06"
    db_path, store_root, _, sessions, _, _ = _seed_ashare_fixture(
        tmp_path, start=cal_start, end=cal_end
    )

    preprocess_cfg = (
        {"ALL": "time_series_zscore", "_ts_window": 4, "_ts_min_periods": 3}
        if use_time_series
        else {"ALL": "zscore"}
    )
    cfg = {
        "database": {
            "factor_db": {"duckdb_path": str(db_path), "factor_values_table": "factor_values"},
            "price_db": {"duckdb_path": str(db_path), "price_table": "price_data_daily", "factor_data_table": "factor_data"},
            "local_db": {"duckdb_path": str(tmp_path / "local.duckdb"), "results_table": "factor_analysis_results"},
            "price_table": "price_data_daily",
        },
        "factor_values_store": str(store_root),
        "analysis_period": {"start_date": analysis_start, "end_date": analysis_end},
        # Rolling window should NOT pull pre-start_date warmup rows into weighted_sum output.
        "rolling_window": {"train_window_days": 15, "predict_window_days": 5, "step_size_days": 5},
        "factor_groups": {"all": {"factors": FACTORS}},
        "synthesis_methods": {
            "ws": {
                "method": "weighted_sum",
                "factor_group": "all",
                "weights": {"f_a": 0.7, "f_b": 0.3},
                "preprocess": preprocess_cfg,
            }
        },
        "performance_evaluation": {
            "quantiles": 2,
            "ic_periods": [1],
            "skip_alphalens_analysis": False,
            "top_k_strategy": {"enabled": False},
        },
        "output_settings": {
            "composite_factor_prefix": "WARMUP",
            "weights_output_dir": str(tmp_path / "weights"),
        },
        "cache_settings": {"use_parquet_cache": False},
        "logging": {"level": "INFO"},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    analyzed: dict[str, pd.Series] = {}
    alphalens_dates: list[pd.DatetimeIndex] = []
    alphalens_ic_dates: list[pd.DatetimeIndex] = []

    synth = FactorSynthesizer(str(cfg_path))
    orig_direct = synth.direct_analyze_composite_factor
    orig_alphalens = synth._run_alphalens_analysis

    def _capture_direct(composite_series, factor_name, run_alphalens=True, **kwargs):
        analyzed[factor_name] = composite_series.copy()
        return orig_direct(composite_series, factor_name, run_alphalens=run_alphalens, **kwargs)

    def _capture_alphalens(factor_df, pricing, factor_name, max_loss=0.35, **kwargs):
        assert kwargs.get("execution_lag") == 0
        alphalens_dates.append(pd.DatetimeIndex(factor_df.index))
        res = orig_alphalens(factor_df, pricing, factor_name, max_loss=max_loss, **kwargs)
        if isinstance(res, dict) and res.get("ic") is not None:
            alphalens_ic_dates.append(pd.DatetimeIndex(res["ic"].index))
        return res

    synth.direct_analyze_composite_factor = _capture_direct
    synth._run_alphalens_analysis = _capture_alphalens
    try:
        synth.rolling_window_synthesis("ws", cfg["synthesis_methods"]["ws"])
    finally:
        synth._close_conn_attr("price_conn")

    composite = analyzed["WARMUP_ws"]
    out_dates = pd.DatetimeIndex(composite.index.get_level_values("date").unique()).sort_values()
    assert not out_dates.empty
    # Buffer history loaded prior to analysis_start preserves analysis_start itself with its T-1 signal.
    assert out_dates.min() == pd.Timestamp(analysis_start)
    assert out_dates.max() <= pd.Timestamp(analysis_end)

    assert len(alphalens_dates) == 1
    assert alphalens_dates[0].min() == pd.Timestamp(analysis_start)
    assert alphalens_dates[0].max() <= pd.Timestamp(analysis_end)
    assert len(alphalens_ic_dates) == 1
    assert alphalens_ic_dates[0].min() == pd.Timestamp(analysis_start)
    assert alphalens_ic_dates[0].max() <= pd.Timestamp(analysis_end)

    with duckdb.connect(str(tmp_path / "local.duckdb"), read_only=True) as res_conn:
        persisted_dates = [
            pd.Timestamp(row[0])
            for row in res_conn.execute(
                "SELECT DISTINCT date FROM factor_analysis_results WHERE factor_name = 'WARMUP_ws' ORDER BY date"
            ).fetchall()
        ]
        assert persisted_dates
        assert min(persisted_dates) == pd.Timestamp(analysis_start)
        assert max(persisted_dates) <= pd.Timestamp(analysis_end)

    # Standalone call with pre-loaded panel containing earlier warmup dates also trims strictly.
    with duckdb.connect(str(db_path), read_only=True, config={"threads": 1}) as conn:
        slot = build_jq_ashare_local(conn, start=cal_start, end=cal_end)
        full_panel = load_factor_data(
            None, "factor_values", FACTORS, cal_start, cal_end,
            config=NO_CACHE, slot=slot, factor_values_store=store_root,
        )
    stub, cap = _make_stub_synthesizer(
        tmp_path / "standalone",
        analysis_period={"start_date": analysis_start, "end_date": analysis_end},
    )
    run_weighted_sum_synthesis(stub, "ws", {"factor_group": "all", "weights": {"f_a": 1.0}, "preprocess": preprocess_cfg}, full_panel, slot=slot)
    stub_dates = pd.DatetimeIndex(cap["composite_ws"].index.get_level_values("date").unique())
    assert stub_dates.min() == pd.Timestamp(analysis_start)
    assert stub_dates.max() <= pd.Timestamp(analysis_end)


@pytest.mark.parametrize("use_precomputed_returns", [False, True])
def test_rolling_window_weighted_sum_alternating_returns_eval_ic_aligns_with_training_signal(
    tmp_path: Path, use_precomputed_returns: bool
) -> None:
    """[二-2 + 二-3] End-to-end rolling_window_synthesis(weighted_sum) yields +1.0 1D IC under alternating returns."""
    stocks = ["000001.XSHE", "000002.XSHE", "000003.XSHE", "000004.XSHE", "000005.XSHE"]
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    delta = 0.01
    n_signal_dates = 15
    extra_future_sessions = 2
    total_sessions = 1 + n_signal_dates + extra_future_sessions

    sessions = pd.bdate_range("2026-01-05", periods=total_sessions)
    cal_start, cal_end = sessions[0].strftime("%Y-%m-%d"), sessions[-1].strftime("%Y-%m-%d")
    analysis_start = sessions[1].strftime("%Y-%m-%d")
    analysis_end = sessions[n_signal_dates].strftime("%Y-%m-%d")

    closes = np.zeros((total_sessions, len(stocks)), dtype=float)
    closes[0, :] = 100.0
    for t in range(total_sessions - 1):
        sign = 1.0 if t % 2 == 0 else -1.0
        closes[t + 1, :] = closes[t, :] * (1.0 + sign * scores * delta)
    close_panel = pd.DataFrame(closes, index=sessions, columns=stocks)

    raw_rows = {}
    for u in range(n_signal_dates):
        sign_next = 1.0 if (u + 1) % 2 == 0 else -1.0
        for i, stock in enumerate(stocks):
            raw_rows[(sessions[u], stock)] = {
                "f_a": float(sign_next * scores[i]),
                "f_b": float(sign_next * (2.0 * scores[i] + 5.0)),
            }
    raw_df = pd.DataFrame.from_dict(raw_rows, orient="index")
    raw_df.index = pd.MultiIndex.from_tuples(raw_df.index, names=["date", "stock_code"])

    db_path, store_root, _, _, _, _ = _seed_ashare_fixture(
        tmp_path,
        start=cal_start,
        end=cal_end,
        raw_values=raw_df,
        stocks=stocks,
        close_panel=close_panel,
    )
    fr_cache_dir = tmp_path / "fr_cache"
    cfg = {
        "database": {
            "factor_db": {"duckdb_path": str(db_path), "factor_values_table": "factor_values"},
            "price_db": {"duckdb_path": str(db_path), "price_table": "price_data_daily", "factor_data_table": "factor_data"},
            "local_db": {"duckdb_path": str(tmp_path / "local.duckdb"), "results_table": "factor_analysis_results"},
            "price_table": "price_data_daily",
        },
        "factor_values_store": str(store_root),
        "analysis_period": {"start_date": analysis_start, "end_date": analysis_end},
        "use_precomputed_returns": use_precomputed_returns,
        "factor_groups": {"all": {"factors": FACTORS}},
        "synthesis_methods": {
            "ws": {
                "method": "weighted_sum",
                "factor_group": "all",
                "weights": {"f_a": 0.7, "f_b": 0.3},
                "preprocess": {"ALL": "zscore"},
            }
        },
        "performance_evaluation": {
            "quantiles": 5,
            "ic_periods": [1],
            "allow_high_data_loss": True,
            "skip_alphalens_analysis": False,
            "forward_returns_cache_dir": str(fr_cache_dir),
            "top_k_strategy": {"enabled": False},
        },
        "output_settings": {
            "composite_factor_prefix": "ALIGN",
            "weights_output_dir": str(tmp_path / "weights"),
        },
        "cache_settings": {"use_parquet_cache": False},
        "logging": {"level": "INFO"},
    }
    cfg_path = tmp_path / "cfg_align.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    analyzed: dict[str, pd.Series] = {}
    captured_lags: list[int | None] = []
    eval_ics: list[pd.Series] = []
    synth = FactorSynthesizer(str(cfg_path))
    orig_direct = synth.direct_analyze_composite_factor
    orig_alphalens = synth._run_alphalens_analysis

    def _capture_direct(composite_series, factor_name, run_alphalens=True, **kwargs):
        analyzed[factor_name] = composite_series.copy()
        return orig_direct(composite_series, factor_name, run_alphalens=run_alphalens, **kwargs)

    def _capture_alphalens(factor_df, pricing, factor_name, max_loss=0.35, **kwargs):
        captured_lags.append(kwargs.get("execution_lag"))
        res = orig_alphalens(factor_df, pricing, factor_name, max_loss=max_loss, **kwargs)
        if isinstance(res, dict) and res.get("ic") is not None:
            eval_ics.append(res["ic"]["1D"].dropna())
        return res

    synth.direct_analyze_composite_factor = _capture_direct
    synth._run_alphalens_analysis = _capture_alphalens
    try:
        synth.rolling_window_synthesis("ws", cfg["synthesis_methods"]["ws"])
        composite = analyzed["ALIGN_ws"]
        conn = synth._get_price_conn()
        train_labels = calculate_forward_returns(
            composite.to_frame("ALIGN_ws"),
            conn,
            "price_data_daily",
            "factor_data",
            period=1,
            pit_enabled=True,
        )
    finally:
        synth._close_conn_attr("price_conn")

    assert captured_lags == [0]
    assert len(eval_ics) == 1
    eval_ic = eval_ics[0]
    train_ic = (
        composite.unstack("stock_code")
        .corrwith(train_labels.unstack("stock_code"), axis=1, method="spearman")
        .dropna()
    )
    assert len(train_ic) == n_signal_dates
    assert len(eval_ic) == n_signal_dates
    np.testing.assert_allclose(train_ic, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(eval_ic, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(eval_ic.mean(), train_ic.mean(), rtol=0, atol=1e-12)

    if use_precomputed_returns:
        cache_files = list(fr_cache_dir.glob("forward_returns_*.pkl"))
        assert len(cache_files) == 1
        assert cache_files[0].name.endswith("_lag0.pkl")

    with duckdb.connect(str(tmp_path / "local.duckdb"), read_only=True) as res_conn:
        persisted = res_conn.execute(
            "SELECT date, ic FROM factor_analysis_results "
            "WHERE factor_name = 'ALIGN_ws' AND period = 1 AND quantile IS NULL ORDER BY date"
        ).fetchall()
    assert len(persisted) == n_signal_dates
    np.testing.assert_allclose([row[1] for row in persisted], 1.0, rtol=0, atol=1e-12)
