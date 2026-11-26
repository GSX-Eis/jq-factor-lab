"""Regression and Synthesis-Replay Parity tests for NERE-1429 (P1-02, P2-03)."""
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

from calendar_test_support import seed_calendar
from factor_synthesis.core.factor_data_loader import load_factor_data
from factor_synthesis.core.factor_ops import apply_feature_preprocessing
from factor_synthesis.core.feature_preprocessor import (
    TIME_SERIES_METHODS as PREPROC_TS_METHODS,
    time_series_standardize as preproc_ts_standardize,
)
from factor_synthesis.pipeline.weighted_sum_method import run_weighted_sum_synthesis
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from market_db_samples import declare_market_db
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


def _seed_store_and_db(
    tmp_path: Path,
    *,
    start: str,
    end: str,
    stocks: list[str],
    factors: list[str],
    raw_values: pd.DataFrame,
):
    db_path = tmp_path / "parity.duckdb"
    store_root = tmp_path / "store"
    with duckdb.connect(str(db_path), config={"threads": 1}) as conn:
        declare_market_db(conn, "cn_equity")
        seed_calendar(conn, start, end)
        slot = build_jq_ashare_local(conn, start=start, end=end)
        sessions = slot.calendar.sessions(start, end)
        index = pd.MultiIndex.from_product([sessions, stocks], names=["date", "stock_code"])
        raw = raw_values.reindex(index)

        px_rows = pd.DataFrame(index=index).reset_index()
        px_rows["day_idx"] = px_rows.groupby("stock_code").cumcount()
        px_rows["stock_idx"] = px_rows["stock_code"].map({s: k for k, s in enumerate(stocks)})
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
        legacy = (
            raw.rename_axis(columns="factor_name")
            .stack(future_stack=True)
            .rename("factor_value")
            .reset_index()
        )
        conn.execute(
            "CREATE TABLE factor_values AS "
            "SELECT date::DATE AS date, stock_code, factor_name, factor_value FROM legacy"
        )
        conn.execute(
            "CREATE TABLE factor_data AS "
            "SELECT date::DATE AS date, stock_code, 20.0 AS natural_log_of_market_cap FROM px_rows"
        )

    store = MarketFactorValueStore(store_root)
    grid = slot.calendar.bar_grid("1d", start, end)
    rows = grid.merge(pd.DataFrame({"instrument_id": stocks}), how="cross")
    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(rows.session_id), rows.instrument_id], names=["date", "stock_code"]
    )
    identities: dict[str, str] = {}
    for factor_id in factors:
        identity, payload = compute_factor_identity_hash(
            factor_id=factor_id,
            factor_class="tests.ParityFactor",
            source_sha256="c" * 64,
            params={},
            frequency="1d",
            session_view="ETH",
            analysis_mode="cross_sectional",
            bucket_strategy="session",
            calendar_hash=slot.calendar.calendar_hash,
            slot_id=slot.slot_id,
            slot_config_hash="d" * 64,
            adjustment_policy="none",
            slot_data_fingerprint=bind_market_db_fingerprint(
                {"fixture": "nere1429"}, slot_market_db_identity(slot)
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


def _make_stub_synth(tmp_path: Path, factors: list[str], *, analysis_period: dict | None = None):
    captured: dict[str, pd.Series] = {}
    cfg: dict = {
        **NO_CACHE,
        "factor_groups": {"all": {"factors": factors}},
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


def test_strategy_kernel_py36_and_ssot_exports() -> None:
    source = Path(kernel.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 6))
    assert kernel.TIME_SERIES_METHODS == PREPROC_TS_METHODS
    assert kernel.time_series_standardize is preproc_ts_standardize


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("postprocess_cs_standardize", [False, True])
def test_p1_02_all_missing_stock_excluded_from_score_and_topk(
    standardize: bool, postprocess_cs_standardize: bool
) -> None:
    """[P1-02] A=-4, B=-3, C=-2, D=NaN: D must never receive 0.0 or win Top-1."""
    raw = pd.DataFrame({"f": [-4.0, -3.0, -2.0, np.nan]}, index=["A", "B", "C", "D"])
    model = {
        "weights": {"f": 1.0},
        "standardize": standardize,
        "standardization_method": {"ALL": "zscore"} if standardize else {"ALL": "none"},
        "postprocess_cs_standardize": postprocess_cs_standardize,
    }
    score, processed = kernel.linear_signal(raw, model)

    assert "D" not in score.index
    assert list(score.index) == ["A", "B", "C"]
    assert pd.isna(processed.loc["D", "f"])
    ranked = kernel.rank_scores(score)
    assert ranked == ["C", "B", "A"]
    assert ranked[0] == "C"

    if not standardize and not postprocess_cs_standardize:
        np.testing.assert_allclose(score.reindex(["A", "B", "C"]), [-4.0, -3.0, -2.0], atol=1e-12)
    elif postprocess_cs_standardize:
        # Mean and std must be computed strictly on {A, B, C}, excluding D.
        assert score.mean() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "method",
    ["zscore", "winsorize_zscore", "raw_zscore", "rank_zscore", "rank", "gauss_rank", "binning"],
)
def test_p1_02_degenerate_cross_section_preserves_nan_and_excludes_missing_rows(
    method: str,
) -> None:
    """[P1-02] Degenerate std==0 or all-NaN cross-sections never turn NaN into 0.0."""
    partial = pd.Series([1.0, 1.0, np.nan], index=["A", "B", "D"])
    out_partial = kernel.preprocess_cross_section_series(partial, method)
    assert pd.isna(out_partial.loc["D"])
    assert pd.notna(out_partial.loc["A"]) and pd.notna(out_partial.loc["B"])

    all_nan = pd.Series([np.nan, np.nan, np.nan], index=["A", "B", "D"])
    out_all_nan = kernel.preprocess_cross_section_series(all_nan, method)
    assert out_all_nan.isna().all()

    raw_df = pd.DataFrame({"f": partial})
    score, processed = kernel.linear_signal(
        raw_df,
        {
            "weights": {"f": 1.0},
            "standardize": True,
            "standardization_method": {"ALL": method},
            "postprocess_cs_standardize": True,
        },
    )
    assert list(score.index) == ["A", "B"]
    assert pd.isna(processed.loc["D", "f"])

    # Single valid stock + NaNs must not become NaN under postprocess_cs_standardize.
    single_valid = pd.DataFrame({"f": [-4.0, np.nan]}, index=["A", "D"])
    score_single, _ = kernel.linear_signal(
        single_valid,
        {
            "weights": {"f": 1.0},
            "standardize": False,
            "standardization_method": {"ALL": "none"},
            "postprocess_cs_standardize": True,
        },
    )
    assert list(score_single.index) == ["A"]
    assert score_single.loc["A"] == pytest.approx(-4.0)


def test_p1_02_multifactor_partial_missing_vs_all_missing_and_blend_only() -> None:
    """Valid rows fill single missing factors with 0.0; all-missing rows stay excluded."""
    raw = pd.DataFrame(
        {
            "f1": [2.0, 4.0, np.nan, np.nan],
            "f2": [6.0, np.nan, 8.0, np.nan],
        },
        index=["A", "B", "C", "D"],
    )
    model = {
        "weights": {"f1": 0.5, "f2": 0.5},
        "standardize": False,
        "standardization_method": {"ALL": "none"},
        "postprocess_cs_standardize": False,
    }
    score, processed = kernel.linear_signal(raw, model)
    assert list(score.index) == ["A", "B", "C"]
    np.testing.assert_allclose(score.loc[["A", "B", "C"]], [4.0, 2.0, 4.0], atol=1e-12)
    assert processed.loc["B", "f2"] == pytest.approx(0.0)
    assert processed.loc["C", "f1"] == pytest.approx(0.0)
    assert processed.loc["D"].isna().all()

    blend_only_model = {
        "weights": {},
        "blend_only": True,
        "blend_weights": {"f1": 0.5, "f2": 0.5},
        "blend_with_original_ratio": 0.8,
        "standardize": False,
        "standardization_method": {"ALL": "none"},
        "postprocess_cs_standardize": False,
    }
    blend_score, _ = kernel.linear_signal(raw, blend_only_model)
    assert list(blend_score.index) == ["A", "B", "C"]
    np.testing.assert_allclose(blend_score.loc[["A", "B", "C"]], [3.2, 1.6, 3.2], atol=1e-12)


@pytest.mark.parametrize("ts_method", ["time_series_zscore", "time_series_rank"])
def test_p2_03_linear_signal_requires_history_and_is_strictly_causal(ts_method: str) -> None:
    """[P2-03] time_series_* fails closed without history and ignores future > asof."""
    dates = pd.bdate_range("2026-01-05", periods=8)
    stocks = ["S1", "S2", "S3"]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])
    panel = pd.DataFrame(
        {
            "f_ts": np.arange(len(idx), dtype=float) + np.sin(np.arange(len(idx))),
            "f_cs": np.cos(np.arange(len(idx), dtype=float)) * 10.0,
        },
        index=idx,
    )
    model = {
        "weights": {"f_ts": 0.6, "f_cs": 0.4},
        "standardize": True,
        "standardization_method": {
            "ALL": "none",
            "f_ts": ts_method,
            "f_cs": "zscore",
            "_ts_window": 4,
            "_ts_min_periods": 3,
        },
        "postprocess_cs_standardize": False,
    }
    asof = dates[5]
    raw_asof = panel.loc[asof]

    with pytest.raises(ValueError, match="time_series_history_required"):
        kernel.linear_signal(raw_asof, model)

    # MultiIndex causal history vs mutated future (> asof) must agree identically.
    mutated = panel.copy()
    mutated.loc[mutated.index.get_level_values("date") > asof, "f_ts"] = 999999.0
    score_clean, proc_clean = kernel.linear_signal(
        raw_asof,
        model,
        history=panel.loc[panel.index.get_level_values("date") <= asof],
        asof=asof,
    )
    score_mutated, proc_mutated = kernel.linear_signal(
        raw_asof, model, history=mutated, asof=asof
    )
    pd.testing.assert_series_equal(score_clean, score_mutated)
    pd.testing.assert_frame_equal(proc_clean, proc_mutated)

    # Wide {factor: DataFrame} history format must produce identical scores.
    wide_history = {
        "f_ts": panel["f_ts"].unstack("stock_code"),
        "f_cs": panel["f_cs"].unstack("stock_code"),
    }
    score_wide, _ = kernel.linear_signal(raw_asof, model, history=wide_history, asof=asof)
    pd.testing.assert_series_equal(score_clean, score_wide)


@pytest.mark.parametrize(
    ("preprocess_cfg", "extra_cfg"),
    [
        ({"ALL": "none"}, {}),
        ({"ALL": "zscore"}, {}),
        ({"ALL": "time_series_zscore"}, {}),
        ({"ALL": "time_series_rank", "_ts_window": 4, "_ts_min_periods": 3}, {}),
        (
            {
                "ALL": "none",
                "f_a": "time_series_zscore",
                "f_b": "zscore",
                "_ts_window": 4,
                "_ts_min_periods": 3,
            },
            {"postprocess_cs_standardize": True},
        ),
    ],
)
def test_p2_03_and_synthesis_replay_daily_parity_with_warmup_and_missing(
    tmp_path: Path, preprocess_cfg: dict, extra_cfg: dict
) -> None:
    """[Synthesis-Replay Parity] Daily parity of valid stocks, scores, and Top-K with warmup & NaNs."""
    cal_start, cal_end = "2026-01-05", "2026-02-20"
    analysis_start, analysis_end = "2026-02-04", "2026-02-18"
    stocks = ["000001.XSHE", "000002.XSHE", "000003.XSHE", "000004.XSHE"]
    factors = ["f_a", "f_b"]

    full_sessions = pd.bdate_range(cal_start, cal_end)
    rng = np.random.default_rng(1429)
    idx = pd.MultiIndex.from_product([full_sessions, stocks], names=["date", "stock_code"])
    raw_df = pd.DataFrame(
        {
            "f_a": np.linspace(-3.0, 5.0, len(idx)) + rng.normal(0.0, 0.4, len(idx)),
            "f_b": np.sin(np.arange(len(idx), dtype=float)) * 4.0 + rng.normal(0.0, 0.2, len(idx)),
        },
        index=idx,
    )
    # Inject partial missing and all-missing observations in both warmup and evaluation window.
    raw_df.loc[(full_sessions[5], stocks[3]), "f_a"] = np.nan
    raw_df.loc[(full_sessions[10], stocks[3]), :] = np.nan
    raw_df.loc[(full_sessions[23], stocks[3]), :] = np.nan
    raw_df.loc[(full_sessions[25], stocks[2]), "f_b"] = np.nan

    db_path, store_root, slot, sessions, _, identities = _seed_store_and_db(
        tmp_path,
        start=cal_start,
        end=cal_end,
        stocks=stocks,
        factors=factors,
        raw_values=raw_df,
    )

    cfg = {
        "database": {
            "factor_db": {"duckdb_path": str(db_path), "factor_values_table": "factor_values"},
            "price_db": {
                "duckdb_path": str(db_path),
                "price_table": "price_data_daily",
                "factor_data_table": "factor_data",
            },
            "local_db": {
                "duckdb_path": str(tmp_path / "local.duckdb"),
                "results_table": "factor_analysis_results",
            },
            "price_table": "price_data_daily",
        },
        "factor_values_store": str(store_root),
        "analysis_period": {"start_date": analysis_start, "end_date": analysis_end},
        "factor_groups": {"all": {"factors": factors}},
        "synthesis_methods": {
            "ws": {
                "method": "weighted_sum",
                "factor_group": "all",
                "weights": {"f_a": 0.7, "f_b": -0.3},
                "preprocess": preprocess_cfg,
                **extra_cfg,
            }
        },
        "performance_evaluation": {
            "quantiles": 2,
            "ic_periods": [1],
            "skip_alphalens_analysis": True,
            "top_k_strategy": {"enabled": False},
        },
        "output_settings": {
            "composite_factor_prefix": "PARITY",
            "weights_output_dir": str(tmp_path / "weights"),
        },
        "cache_settings": {"use_parquet_cache": False},
        "logging": {"level": "INFO"},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    captured: dict[str, pd.Series] = {}
    synth = FactorSynthesizer(str(cfg_path))
    synth.direct_analyze_composite_factor = lambda s, n, **kw: captured.update({n: s.copy()})
    try:
        synth.rolling_window_synthesis("ws", cfg["synthesis_methods"]["ws"])
    finally:
        synth._close_conn_attr("price_conn")

    local_scores = captured["PARITY_ws"]
    eval_dates = pd.DatetimeIndex(local_scores.index.get_level_values("date").unique()).sort_values()
    assert eval_dates.min() == pd.Timestamp(analysis_start)
    assert eval_dates.max() <= pd.Timestamp(analysis_end)

    bundle_path = tmp_path / "weights" / "weighted_sum_bundle_ws.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    model_std = bundle["index"]["models"][0]["standardization_method"]
    if any(
        v in kernel.TIME_SERIES_METHODS
        for k, v in model_std.items()
        if k in ("ALL", *factors)
    ):
        expected_win = preprocess_cfg.get("_ts_window", 20)
        expected_min = preprocess_cfg.get("_ts_min_periods", expected_win)
        assert model_std["_ts_window"] == expected_win
        assert model_std["_ts_min_periods"] == expected_min

    # Replay through VectorBTBacktestRunner over evaluation window only (warmup resides in store).
    with duckdb.connect(str(db_path), read_only=True, config={"threads": 1}) as conn:
        eval_slot = build_jq_ashare_local(conn, start=cal_start, end=cal_end)
        prices = conn.execute(
            "SELECT date, stock_code, open, close FROM price_data_daily ORDER BY date, stock_code"
        ).fetchdf()
        prices["date"] = pd.to_datetime(prices["date"])
        eval_sessions = sessions[(sessions >= pd.Timestamp(analysis_start)) & (sessions <= pd.Timestamp(analysis_end))]
        close_df = prices.pivot(index="date", columns="stock_code", values="close").reindex(
            index=eval_sessions, columns=stocks
        )
        open_df = prices.pivot(index="date", columns="stock_code", values="open").reindex(
            index=eval_sessions, columns=stocks
        )

        runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 2, "holding_period": 1, "rebalance_freq": "daily", "enable_tday_filter": False}},
            slot=eval_slot,
            frequency="1d",
            bundle_path=str(bundle_path),
            factor_values_store=store_root,
            factor_identities=identities,
        )
        try:
            score_wide = local_scores.unstack("stock_code").reindex(index=eval_sessions, columns=stocks)
            res = runner.run_backtest(score_wide, close_df, open_df=open_df, rebalance_freq="daily")
            assert "error" not in res, res

            # Also verify day 1 of evaluation window (uses pre-start_date warmup strictly <= T-1).
            for trade_day in eval_dates:
                t_key = trade_day.strftime("%Y-%m-%d")
                prev_key = runner._backtest_previous_dates.get(t_key) or runner._get_previous_trading_day(t_key)
                assert prev_key is not None
                replayed, _, _ = runner._get_synthesis_score(prev_key, stocks)
                local_day = local_scores.loc[trade_day].dropna()

                assert set(replayed.dropna().index) == set(local_day.index)
                np.testing.assert_allclose(
                    replayed.reindex(local_day.index), local_day, rtol=0, atol=1e-10
                )
                local_rank = kernel.rank_scores(local_day)
                replay_rank = kernel.rank_scores(replayed)
                assert replay_rank == local_rank
                k = kernel.selection_count(len(local_rank), top_k=2)
                assert replay_rank[:k] == local_rank[:k]
        finally:
            runner.close()


def test_p2_03_backtest_runner_ts_replay_blocks_future_leakage(tmp_path: Path) -> None:
    """Mutating factor values after date_str must not change VectorBT TS replay score at date_str."""
    start, end = "2026-01-05", "2026-01-23"
    stocks = ["000001.XSHE", "000002.XSHE", "000003.XSHE", "000004.XSHE"]
    factors = ["f_a"]
    sessions = pd.bdate_range(start, end)
    idx = pd.MultiIndex.from_product([sessions, stocks], names=["date", "stock_code"])
    raw_df = pd.DataFrame({"f_a": np.arange(len(idx), dtype=float)}, index=idx)

    anchor_date = sessions[8]
    mutated_df = raw_df.copy()
    mutated_df.loc[mutated_df.index.get_level_values("date") > anchor_date, "f_a"] = 1e9

    outputs = []
    for sub_name, frame in (("clean", raw_df), ("future_mutated", mutated_df)):
        sub_dir = tmp_path / sub_name
        sub_dir.mkdir()
        db_path, store_root, slot, _, _, identities = _seed_store_and_db(
            sub_dir,
            start=start,
            end=end,
            stocks=stocks,
            factors=factors,
            raw_values=frame,
        )
        bundle_path = sub_dir / "ws_ts_bundle.json"
        bundle_path.write_text(
            json.dumps(
                {
                    "index": {
                        "models": [
                            {
                                "train_end": "1900-01-01",
                                "weights": {"f_a": 1.0},
                                "feature_names": ["f_a"],
                                "blend_only": False,
                                "blend_with_original_ratio": 0.0,
                                "blend_weights": {},
                                "standardize": True,
                                "standardization_method": {
                                    "ALL": "time_series_zscore",
                                    "f_a": "time_series_zscore",
                                    "_ts_window": 4,
                                    "_ts_min_periods": 3,
                                },
                                "postprocess_cs_standardize": False,
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 2}},
            slot=slot,
            frequency="1d",
            bundle_path=str(bundle_path),
            factor_values_store=store_root,
            factor_identities=identities,
        )
        try:
            score, _, _ = runner._compute_synthesis_score(anchor_date.strftime("%Y-%m-%d"), stocks)
            outputs.append(score)
        finally:
            runner.close()

    pd.testing.assert_series_equal(outputs[0], outputs[1])


def test_lgb_prediction_preprocessing_respects_none_all_and_ts_methods() -> None:
    """LGB prediction honors 'none', 'ALL' in blend_weights, and time_series_* methods."""
    dates = pd.bdate_range("2026-01-05", periods=6)
    stocks = ["S1", "S2", "S3"]
    panel_ts = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [4.0, 8.0, 12.0], [5.0, 10.0, 15.0], [9.0, 1.0, 2.0]],
        index=dates,
        columns=stocks,
    )
    panel_none = pd.DataFrame(
        np.tile([10.0, 20.0, 30.0], (len(dates), 1)),
        index=dates,
        columns=stocks,
    )

    class _PanelLoader:
        def load_factors(self, date_str, req_stocks, factors):
            ts = pd.Timestamp(date_str)
            data = {}
            for f in factors:
                src = panel_ts if f == "f_ts" else panel_none
                data[f] = src.loc[ts, req_stocks]
            return pd.DataFrame(data, index=req_stocks)

        def load_factor_panel(self, _start, end, req_stocks, factors):
            end_ts = pd.Timestamp(end)
            out = {}
            for f in factors:
                src = panel_ts if f == "f_ts" else panel_none
                out[f] = src.loc[src.index <= end_ts, req_stocks]
            return out

    class _IdentityBooster:
        def predict(self, values):
            return values[:, 0] + values[:, 1]

    runner = VectorBTBacktestRunner({"top_k_strategy": {}, "calendar_filters": {}})
    runner._factor_loader = _PanelLoader()
    runner._load_lgb_booster_for_date = lambda _d: _IdentityBooster()
    runner._get_model_for_date = lambda _d: {
        "feature_names": ["f_none", "f_ts"],
        "standardization_method": {
            "ALL": "none",
            "f_ts": "time_series_zscore",
            "_ts_window": 4,
            "_ts_min_periods": 3,
        },
        "blend_with_original_ratio": 0.5,
        "blend_weights": {"f_none": 1.0},
        "postprocess_cs_standardize": False,
    }

    asof_str = dates[-1].strftime("%Y-%m-%d")
    score, processed, _ = runner._compute_lgb_prediction(asof_str, stocks, return_raw_factors=True)
    # f_none must remain raw [10, 20, 30] (not z-scored), both in features and blend.
    np.testing.assert_allclose(processed["f_none"].reindex(stocks), [10.0, 20.0, 30.0], atol=1e-12)
    expected_ts = kernel.time_series_standardize_at_asof(
        panel_ts.loc[dates[-1], stocks],
        "f_ts",
        "time_series_zscore",
        {"_ts_window": 4, "_ts_min_periods": 3},
        history={"f_ts": panel_ts},
        asof=dates[-1],
    )
    np.testing.assert_allclose(processed["f_ts"].reindex(stocks), expected_ts, atol=1e-12)
    expected_score = (processed["f_none"] + processed["f_ts"]) * 0.5 + pd.Series([10.0, 20.0, 30.0], index=stocks) * 0.5
    np.testing.assert_allclose(score.reindex(stocks), expected_score, atol=1e-12)


def _assert_rolling_long_gap_parity(
    work_dir: Path,
    *,
    db_path: Path,
    store_root: Path,
    price_table: str,
    expect_slot: bool,
    factors: list[str],
    preprocess_cfg: dict,
    analysis_period: dict,
    local_direct: pd.Series,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    method_cfg = {
        "method": "weighted_sum",
        "factor_group": "all",
        "weights": {"f": 1.0},
        "preprocess": preprocess_cfg,
        "postprocess_cs_standardize": False,
    }
    synth_cfg = {
        "database": {
            "factor_db": {"duckdb_path": str(db_path), "factor_values_table": "factor_values"},
            "price_db": {
                "duckdb_path": str(db_path),
                "price_table": price_table,
                "factor_data_table": "factor_data",
            },
            "local_db": {
                "duckdb_path": str(work_dir / "local.duckdb"),
                "results_table": "factor_analysis_results",
            },
            "price_table": price_table,
        },
        "factor_values_store": str(store_root),
        "analysis_period": analysis_period,
        "factor_groups": {"all": {"factors": factors}},
        "synthesis_methods": {"gap_ws": method_cfg},
        "performance_evaluation": {
            "quantiles": 2,
            "ic_periods": [1],
            "skip_alphalens_analysis": True,
            "top_k_strategy": {"enabled": False},
        },
        "output_settings": {
            "composite_factor_prefix": "GAP",
            "weights_output_dir": str(work_dir / "synth_weights"),
        },
        "cache_settings": {"use_parquet_cache": False},
        "logging": {"level": "INFO"},
    }
    cfg_path = work_dir / "gap_cfg.json"
    cfg_path.write_text(json.dumps(synth_cfg), encoding="utf-8")
    cap_rolling: dict[str, pd.Series] = {}
    synth = FactorSynthesizer(str(cfg_path))
    assert synth._use_slot_feeds() is expect_slot
    synth.direct_analyze_composite_factor = lambda s, n, **kw: cap_rolling.update({n: s.copy()})
    try:
        synth.rolling_window_synthesis("gap_ws", method_cfg)
    finally:
        synth._close_conn_attr("price_conn")
    local_rolling = cap_rolling["GAP_gap_ws"]
    for day in local_direct.index.get_level_values("date").unique():
        expected = local_direct.loc[day].dropna()
        actual = local_rolling.loc[day].dropna()
        assert set(actual.index) == set(expected.index), (price_table, day)
        np.testing.assert_allclose(actual.reindex(expected.index), expected, rtol=0, atol=1e-12)
        assert kernel.rank_scores(actual) == kernel.rank_scores(expected), (price_table, day)
    pd.testing.assert_series_equal(local_rolling, local_direct, check_names=False, atol=1e-12)


@pytest.mark.parametrize("ts_method", ["time_series_zscore", "time_series_rank"])
def test_p2_03_long_gap_exceeding_lookback_matches_local_replay_and_both_history_shapes(
    tmp_path: Path, ts_method: str
) -> None:
    """Long missing-data gaps exceeding fixed lookback preserve valid set, scores, and Top-K across all paths."""
    cal_start, cal_end = "2026-01-05", "2026-04-10"
    analysis_start, analysis_end = "2026-04-08", "2026-04-09"
    stocks = ["A", "B", "C"]
    factors = ["f"]
    # Build values strictly on the actual seeded calendar sessions (matching store & DB).
    probe_db = tmp_path / "probe_cal.duckdb"
    with duckdb.connect(str(probe_db), config={"threads": 1}) as conn:
        declare_market_db(conn, "cn_equity")
        seed_calendar(conn, cal_start, cal_end)
        cal_sessions = build_jq_ashare_local(conn, start=cal_start, end=cal_end).calendar.sessions(
            cal_start, cal_end
        )

    idx = pd.MultiIndex.from_product([cal_sessions, stocks], names=["date", "stock_code"])
    raw_df = pd.DataFrame(np.nan, index=idx, columns=factors, dtype=float)

    # Stock A: 2026-01-06 -> 1, 2026-01-07 -> 3, missing > 50 trading days, 2026-04-07 -> 5, 2026-04-08 -> 7.
    # For window=2, min_periods=2, on trade day 2026-04-08 (asof 2026-04-07):
    #   zscore = (5 - mean([1, 3])) / std([1, 3], ddof=0) = (5 - 2) / 1 = 3.0
    #   rank   = 2 / 2 = 1.0
    raw_df.loc[(pd.Timestamp("2026-01-06"), "A"), "f"] = 1.0
    raw_df.loc[(pd.Timestamp("2026-01-07"), "A"), "f"] = 3.0
    raw_df.loc[(pd.Timestamp("2026-04-07"), "A"), "f"] = 5.0
    raw_df.loc[(pd.Timestamp("2026-04-08"), "A"), "f"] = 7.0

    # Stocks B and C: dense daily series designed so their zscore/rank on 2026-04-07..08 stay below A.
    for day_i, day in enumerate(cal_sessions):
        raw_df.loc[(day, "B"), "f"] = 10.0 + float(day_i % 3)
        raw_df.loc[(day, "C"), "f"] = 20.0 - 0.1 * float(day_i)
    apr07_pos = int(cal_sessions.get_loc(pd.Timestamp("2026-04-07")))
    raw_df.loc[(cal_sessions[apr07_pos - 2], "B"), "f"] = 10.0
    raw_df.loc[(cal_sessions[apr07_pos - 1], "B"), "f"] = 12.0
    raw_df.loc[(cal_sessions[apr07_pos], "B"), "f"] = 11.0
    raw_df.loc[(cal_sessions[apr07_pos + 1], "B"), "f"] = 11.5
    raw_df.loc[(cal_sessions[apr07_pos - 2], "C"), "f"] = 20.0
    raw_df.loc[(cal_sessions[apr07_pos - 1], "C"), "f"] = 22.0
    raw_df.loc[(cal_sessions[apr07_pos], "C"), "f"] = 19.0
    raw_df.loc[(cal_sessions[apr07_pos + 1], "C"), "f"] = 18.0

    gap_days = apr07_pos - int(cal_sessions.get_loc(pd.Timestamp("2026-01-07"))) - 1
    assert gap_days >= 50

    db_path, store_root, slot, _, seeded_raw, identities = _seed_store_and_db(
        tmp_path,
        start=cal_start,
        end=cal_end,
        stocks=stocks,
        factors=factors,
        raw_values=raw_df,
    )

    preprocess_cfg = {"ALL": ts_method, "_ts_window": 2, "_ts_min_periods": 2}
    # Path 1: Direct run_weighted_sum_synthesis on full panel (exports bundle).
    direct_dir = tmp_path / "direct"
    synth_direct, cap_direct = _make_stub_synth(
        direct_dir, factors, analysis_period={"start_date": analysis_start, "end_date": analysis_end}
    )
    direct_panel = seeded_raw.copy()
    direct_panel.attrs["panel_date_alignment"] = "raw_signal"
    run_weighted_sum_synthesis(
        synth_direct,
        "gap_ws",
        {
            "factor_group": "all",
            "weights": {"f": 1.0},
            "preprocess": preprocess_cfg,
            "postprocess_cs_standardize": False,
        },
        direct_panel,
        slot=slot,
    )
    local_direct = cap_direct["composite_gap_ws"]
    bundle_path = direct_dir / "weights" / "weighted_sum_bundle_gap_ws.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    model = bundle["index"]["models"][0]

    expected_a_apr08 = 3.0 if ts_method == "time_series_zscore" else 1.0
    apr08_local = local_direct.loc[pd.Timestamp("2026-04-08")].dropna()
    assert "A" in apr08_local.index
    assert apr08_local.loc["A"] == pytest.approx(expected_a_apr08, abs=1e-12)
    assert kernel.rank_scores(apr08_local)[0] == "A"

    # Path 2: FactorSynthesizer.rolling_window_synthesis with analysis_period starting 2026-04-08
    # (default 30-day buffer only reaches ~2026-03-09, so per-instrument backfill must activate).
    # A custom price view disables slot feeds and exercises the verified-calendar backfill branch.
    with duckdb.connect(str(db_path), config={"threads": 1}) as conn:
        conn.execute("CREATE VIEW review_price_data AS SELECT * FROM price_data_daily")
    for price_table, expect_slot in (("price_data_daily", True), ("review_price_data", False)):
        _assert_rolling_long_gap_parity(
            tmp_path / f"rolling_{price_table}",
            db_path=db_path,
            store_root=store_root,
            price_table=price_table,
            expect_slot=expect_slot,
            factors=factors,
            preprocess_cfg=preprocess_cfg,
            analysis_period={"start_date": analysis_start, "end_date": analysis_end},
            local_direct=local_direct,
        )

    # Path 3: VectorBTBacktestRunner bundle replay (initial lookback starts around 2026-02-25,
    # well after A's Jan 6-7 observations; per-instrument backfill must recover A strictly <= asof).
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "holding_period": 1, "rebalance_freq": "daily", "enable_tday_filter": False}},
        slot=slot,
        frequency="1d",
        bundle_path=str(bundle_path),
        factor_values_store=store_root,
        factor_identities=identities,
    )
    sparse_history = seeded_raw.dropna(how="all")
    wide_history = {"f": sparse_history["f"].unstack("stock_code")}
    try:
        assert runner._resolve_ts_lookback_start("2026-04-07", 2) > "2026-01-07"
        for trade_day_str, prev_day_str in (("2026-04-08", "2026-04-07"), ("2026-04-09", "2026-04-08")):
            local_day = local_direct.loc[pd.Timestamp(trade_day_str)].dropna()
            replayed, _, _ = runner._get_synthesis_score(prev_day_str, stocks)
            raw_asof = seeded_raw.loc[pd.Timestamp(prev_day_str)].reindex(stocks)

            # Path 4a & 4b: linear_signal with sparse MultiIndex vs wide dict history.
            score_mi, _ = kernel.linear_signal(raw_asof, model, history=sparse_history, asof=prev_day_str)
            score_wide, _ = kernel.linear_signal(raw_asof, model, history=wide_history, asof=prev_day_str)

            for candidate in (replayed, score_mi, score_wide):
                assert set(candidate.dropna().index) == set(local_day.index)
                np.testing.assert_allclose(
                    candidate.reindex(local_day.index), local_day, rtol=0, atol=1e-10
                )
                assert kernel.rank_scores(candidate) == kernel.rank_scores(local_day)
                assert kernel.rank_scores(candidate)[:1] == kernel.rank_scores(local_day)[:1]
    finally:
        runner.close()

    # Path 5: Legacy DuckDB FactorLoader backfill path also recovers A strictly <= asof.
    with duckdb.connect(str(db_path), read_only=True, config={"threads": 1}) as legacy_conn:
        legacy_runner = VectorBTBacktestRunner(
            {"top_k_strategy": {"top_k": 1}},
            slot=slot,
            frequency="1d",
            bundle_path=str(bundle_path),
            factor_db_conn=legacy_conn,
        )
        try:
            legacy_score, _, _ = legacy_runner._compute_synthesis_score("2026-04-07", stocks)
            assert set(legacy_score.dropna().index) == set(apr08_local.index)
            np.testing.assert_allclose(
                legacy_score.reindex(apr08_local.index), apr08_local, rtol=0, atol=1e-10
            )
            assert kernel.rank_scores(legacy_score)[:1] == ["A"]
        finally:
            legacy_runner.close()

    # Causality check on the backfill path: mutating observations strictly after 2026-04-07
    # must not change the backfilled replay score at 2026-04-07.
    mutated_dir = tmp_path / "mutated_future"
    mutated_dir.mkdir()
    mutated_raw = seeded_raw.copy()
    mutated_raw.loc[mutated_raw.index.get_level_values("date") > pd.Timestamp("2026-04-07"), "f"] = 1e9
    _, mut_store_root, mut_slot, _, _, mut_identities = _seed_store_and_db(
        mutated_dir,
        start=cal_start,
        end=cal_end,
        stocks=stocks,
        factors=factors,
        raw_values=mutated_raw,
    )
    mut_runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1}},
        slot=mut_slot,
        frequency="1d",
        bundle_path=str(bundle_path),
        factor_values_store=mut_store_root,
        factor_identities=mut_identities,
    )
    try:
        mut_score, _, _ = mut_runner._compute_synthesis_score("2026-04-07", stocks)
        np.testing.assert_allclose(
            mut_score.reindex(apr08_local.index), apr08_local, rtol=0, atol=1e-12
        )
    finally:
        mut_runner.close()


def test_p2_03_multifactor_sparse_vs_wide_history_parity_with_partial_and_full_gaps() -> None:
    """MultiIndex and wide-dict history agree on both full-row gaps and single-factor NaN rows."""
    dates = pd.bdate_range("2026-01-05", periods=10)
    stocks = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_code"])
    raw_df = pd.DataFrame(
        {
            "f_ts": np.arange(len(idx), dtype=float) + 1.0,
            "f_cs": np.linspace(1.0, 10.0, len(idx)),
        },
        index=idx,
    )
    # Full gap (both factors NaN) for A on dates[2..5]: dropped from sparse history.
    for d in dates[2:6]:
        raw_df.loc[(d, "A"), :] = np.nan
    # Partial gap (only f_ts NaN, f_cs finite) for B on dates[6]: row stays alive in both shapes.
    raw_df.loc[(dates[6], "B"), "f_ts"] = np.nan

    sparse_panel = raw_df.dropna(how="all")
    wide_history = {col: sparse_panel[col].unstack("stock_code") for col in ("f_ts", "f_cs")}
    model = {
        "weights": {"f_ts": 0.7, "f_cs": 0.3},
        "standardize": True,
        "standardization_method": {
            "ALL": "none",
            "f_ts": "time_series_zscore",
            "f_cs": "zscore",
            "_ts_window": 2,
            "_ts_min_periods": 2,
        },
        "postprocess_cs_standardize": False,
    }
    for asof in dates[6:]:
        raw_asof = raw_df.loc[asof].reindex(stocks)
        score_mi, proc_mi = kernel.linear_signal(raw_asof, model, history=sparse_panel, asof=asof)
        score_wide, proc_wide = kernel.linear_signal(raw_asof, model, history=wide_history, asof=asof)
        pd.testing.assert_series_equal(score_mi, score_wide, atol=1e-12)
        pd.testing.assert_frame_equal(proc_mi, proc_wide, atol=1e-12)
        assert kernel.rank_scores(score_mi) == kernel.rank_scores(score_wide)


@pytest.mark.parametrize(
    ("preprocess_cfg", "expected_g"),
    [
        ({"ALL": None}, [100.0, 200.0, 300.0]),
        ({"ALL": ""}, [100.0, 200.0, 300.0]),
        ({"ALL": "rank", "f": None}, [1.0 / 3.0, 2.0 / 3.0, 1.0]),
        ({"ALL": "rank", "f": ""}, [1.0 / 3.0, 2.0 / 3.0, 1.0]),
    ],
)
def test_lgb_and_kernel_preserve_explicit_null_and_empty_preprocess_pass_through(
    preprocess_cfg: dict, expected_g: list[float]
) -> None:
    """Explicit None/'' disables preprocessing for features, blend factors, and linear_signal."""
    stocks = ["S1", "S2", "S3"]
    asof = pd.Timestamp("2026-01-06")
    raw_frame = pd.DataFrame(
        {"f": [10.0, 20.0, 30.0], "g": [100.0, 200.0, 300.0]},
        index=stocks,
    )
    local_panel = raw_frame.copy()
    local_panel.index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([asof] * len(stocks)), stocks], names=["date", "stock_code"]
    )
    local_proc = apply_feature_preprocessing(local_panel, preprocess_cfg).xs(asof, level="date")
    np.testing.assert_allclose(local_proc["f"].reindex(stocks), [10.0, 20.0, 30.0], atol=1e-12)
    np.testing.assert_allclose(local_proc["g"].reindex(stocks), expected_g, atol=1e-12)

    class _StaticLoader:
        def load_factors(self, _date_str, req_stocks, factors):
            return raw_frame.reindex(index=req_stocks, columns=factors)

    class _IdentityBooster:
        def predict(self, values):
            return values[:, 0] + values[:, 1]

    runner = VectorBTBacktestRunner({"top_k_strategy": {}, "calendar_filters": {}})
    runner._factor_loader = _StaticLoader()
    runner._load_lgb_booster_for_date = lambda _d: _IdentityBooster()
    runner._get_model_for_date = lambda _d: {
        "feature_names": ["f", "g"],
        "standardization_method": preprocess_cfg,
        "blend_with_original_ratio": 0.5,
        "blend_weights": {"f": 1.0, "g": 2.0},
        "postprocess_cs_standardize": False,
    }

    score, processed, _ = runner._compute_lgb_prediction("2026-01-06", stocks, return_raw_factors=True)
    np.testing.assert_allclose(processed["f"].reindex(stocks), [10.0, 20.0, 30.0], atol=1e-12)
    np.testing.assert_allclose(processed["g"].reindex(stocks), expected_g, atol=1e-12)
    expected_pred = np.array([10.0, 20.0, 30.0]) + np.array(expected_g)
    expected_blend = np.array([10.0, 20.0, 30.0]) * 1.0 + np.array(expected_g) * 2.0
    np.testing.assert_allclose(
        score.reindex(stocks),
        expected_pred * 0.5 + expected_blend * 0.5,
        atol=1e-12,
    )

    # linear_signal must also preserve explicit None / "" pass-through.
    lin_score, lin_proc = kernel.linear_signal(
        raw_frame,
        {
            "weights": {"f": 1.0, "g": 1.0},
            "standardize": True,
            "standardization_method": preprocess_cfg,
            "postprocess_cs_standardize": False,
        },
    )
    np.testing.assert_allclose(lin_proc["f"].reindex(stocks), [10.0, 20.0, 30.0], atol=1e-12)
    np.testing.assert_allclose(lin_proc["g"].reindex(stocks), expected_g, atol=1e-12)
    np.testing.assert_allclose(lin_score.reindex(stocks), expected_pred, atol=1e-12)


def test_lgb_absent_preprocess_keys_retain_defaults() -> None:
    """Only omitted keys (absent from standardization_method and ALL) use default fallbacks."""
    stocks = ["S1", "S2", "S3"]
    raw_frame = pd.DataFrame({"f": [10.0, 20.0, 30.0]}, index=stocks)

    class _StaticLoader:
        def load_factors(self, _date_str, req_stocks, factors):
            return raw_frame.reindex(index=req_stocks, columns=factors)

    class _IdentityBooster:
        def predict(self, values):
            return values[:, 0]

    runner = VectorBTBacktestRunner({"top_k_strategy": {}, "calendar_filters": {}})
    runner._factor_loader = _StaticLoader()
    runner._load_lgb_booster_for_date = lambda _d: _IdentityBooster()
    runner._get_model_for_date = lambda _d: {
        "feature_names": ["f"],
        "standardization_method": {},
        "blend_with_original_ratio": 0.5,
        "blend_weights": {"f": 1.0},
        "postprocess_cs_standardize": False,
    }

    score, processed, _ = runner._compute_lgb_prediction("2026-01-06", stocks, return_raw_factors=True)
    expected_feat = kernel.preprocess_series(raw_frame["f"], "rank_zscore")
    expected_blend = kernel.preprocess_series(raw_frame["f"], "zscore")
    np.testing.assert_allclose(processed["f"].reindex(stocks), expected_feat, atol=1e-12)
    np.testing.assert_allclose(
        score.reindex(stocks),
        expected_feat * 0.5 + expected_blend * 0.5,
        atol=1e-12,
    )
