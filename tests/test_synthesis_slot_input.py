"""Stage 5 input parity and market-independent synthesis on synthetic data."""

from dataclasses import replace
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb

from market_db_samples import declare_market_db
import numpy as np
import pandas as pd
import pytest

from factor_synthesis.core.factor_data_loader import (
    get_complete_price_data, get_factor_ratio_data, get_open_price_data, get_pricing_data,
    load_factor_data, load_market_cap_data,
)
from factor_synthesis.core.factor_ops import apply_feature_preprocessing, calculate_forward_returns
from factor_synthesis.factor_reader import FactorReader
from factor_synthesis.pipeline.lasso_method import run_lasso_synthesis
from factor_synthesis.pipeline.lightgbm_method import run_lightgbm_synthesis
from factor_synthesis.pipeline.weighted_sum_method import run_weighted_sum_synthesis
from slots import build_slot_from_config
from slots.jq_ashare_local import build_jq_ashare_local
from utils.market_db.identity import bind_market_db_fingerprint, slot_market_db_identity
from utils.market_factor import MarketFactorValueStore, build_identity_record, compute_factor_identity_hash
from utils.slots import ProviderConfig, load_slot_config
from utils.time_axis import Frequency


START, END = "2026-01-05", "2026-01-30"
FACTORS = ["alpha", "beta"]
STOCKS = [f"{i:06d}.XSHE" for i in range(1, 13)]
LOG = logging.getLogger(__name__)
NO_CACHE = {"data_bindings": {}, "cache_settings": {"use_parquet_cache": False}}
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "declarative_crypto"


def _save_panel(store, slot, frequency, panel):
    freq = Frequency.parse(frequency)
    grid = slot.calendar.bar_grid(freq, START, END)
    instruments = panel.index.get_level_values("stock_code").unique()
    rows = grid.merge(pd.DataFrame({"instrument_id": instruments}), how="cross")
    dates = pd.to_datetime(rows.session_id) if frequency == "1d" else rows.timestamp_utc
    keys = pd.MultiIndex.from_arrays([dates, rows.instrument_id], names=panel.index.names)
    identities = {}
    for factor in panel:
        identity, payload = compute_factor_identity_hash(
            factor_id=factor, factor_class="tests.Alpha", source_sha256="a" * 64, params={},
            frequency=frequency, session_view="ETH", analysis_mode="cross_sectional",
            bucket_strategy="session" if frequency == "1d" else "segment_anchored",
            calendar_hash=slot.calendar.calendar_hash, slot_id=slot.slot_id,
            slot_config_hash="b" * 64, adjustment_policy="none",
            slot_data_fingerprint=bind_market_db_fingerprint({"fixture": "v1"}, slot_market_db_identity(slot)), code_version="synthetic",
        )
        record = build_identity_record(factor_identity_hash=identity, payload=payload, code_version_source="test")
        batch = rows.assign(
            factor_id=factor, factor_identity_hash=identity, frequency=freq.canonical, session_view="ETH",
            factor_value=panel[factor].reindex(keys).to_numpy(), source_build_id="fixture",
            calendar_hash=slot.calendar.calendar_hash, code_version="synthetic",
            created_at=pd.Timestamp("2026-02-01", tz="UTC"), batch_id="fixture",
        )
        store.append_factor_values(batch, identity_record=record, calendar=slot.calendar)
        identities[factor] = identity
    return identities


def _inputs(slot, frequency, instruments):
    grid = slot.calendar.bar_grid(frequency, START, END)
    dates = pd.to_datetime(grid.session_id) if frequency == "1d" else grid.timestamp_utc
    index = pd.MultiIndex.from_product([dates, instruments], names=["date", "stock_code"])
    rng = np.random.default_rng(2026)
    return pd.DataFrame(rng.normal(size=(len(index), 2)), index=index, columns=FACTORS)


@pytest.fixture
def ashare():
    with duckdb.connect(config={"threads": 1}) as conn:
        declare_market_db(conn, 'cn_equity')
        slot = build_jq_ashare_local(conn, start=START, end=END)
        panel = _inputs(slot, "1d", STOCKS)
        i = np.arange(len(panel), dtype=float)
        sources = panel.reset_index().assign(
            close=80 + i / 200 + (i % len(STOCKS)) + panel.alpha.to_numpy() / 10,
            factor=np.where(i < 80, 0.5, 1.0),
            natural_log_of_market_cap=21 + (i % len(STOCKS)) / 4 + panel.beta.to_numpy() / 10,
            sw_l1_code=np.where(i % 3 == 0, "I1", "I2"),
        )
        conn.execute("""CREATE TABLE price_data_daily AS SELECT
            date::DATE AS date, stock_code, close, close - 1 AS open, close AS high,
            close AS low, 1.0 AS volume, factor FROM sources""")
        conn.execute("""CREATE TABLE factor_data AS SELECT
            date::DATE AS date, stock_code, natural_log_of_market_cap FROM sources""")
        conn.execute("""CREATE TABLE industry_membership AS SELECT
            date::DATE AS date, stock_code, sw_l1_code FROM sources""")
        yield conn, slot, panel, sources


def _crypto_slot(tmp_path, instruments, frequency="1h"):
    config = load_slot_config(EXAMPLE / "slot.yaml")
    options = {**config.bar_feed.options, "frequency": frequency}
    universe = ProviderConfig("static", {
        "instruments": {stock: {"asset_class": "crypto", "currency": "USD"} for stock in instruments},
        "universes": {"spot": instruments},
    })
    calendar = ProviderConfig(config.calendar.provider, dict(config.calendar.options, start=START, end=END))
    config = replace(config, bar_feed=ProviderConfig("csv_files", options), calendar=calendar,
                     universe=universe, aux_features=ProviderConfig("none"))
    instants = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta(days=1), freq="1h", tz="UTC")[1:]
    prices = pd.MultiIndex.from_product([instants, instruments], names=["time", "symbol"]).to_frame(index=False)
    close = 40 + np.arange(len(prices)) / 50 + np.sin(np.arange(len(prices)) / 3)
    prices.assign(open=close, high=close, low=close, close=close, volume=10).to_csv(tmp_path / "bars.csv", index=False)
    return build_slot_from_config(config, base_dir=tmp_path)


@pytest.mark.parametrize("frequency", ["1d", "5min", "15min", "1h", "4h", "1w"])
def test_scheme_b_synthesis_grid_and_slot_isolation(tmp_path, ashare, frequency):
    conn, slot, _, _ = ashare
    store = MarketFactorValueStore(tmp_path / "store")
    panel = _inputs(slot, frequency, STOCKS[:2])
    ids = _save_panel(store, slot, frequency, panel)
    other = replace(slot, slot_id="simulated_futures")
    _save_panel(store, other, frequency, panel + 10)
    conn.execute("CREATE TABLE factor_values AS SELECT 'alpha' AS factor_name, 999 AS factor_value")
    reader = FactorReader(str(tmp_path / "absent.duckdb"), data_bindings={}, slot=slot,
                          factor_values_store=store, factor_identities=ids, frequency=frequency)
    actual = reader.get_multiple_factors(FACTORS, START, END)
    for name in FACTORS:
        np.testing.assert_allclose(actual[name], panel[name].unstack(), rtol=0, atol=1e-12)
        assert actual[name].attrs["market_factor_inputs"]["factors"] == [{
            "factor_id": name, "factor_identity_hash": ids[name], "slot_id": slot.slot_id,
            "frequency": Frequency.parse(frequency).canonical,
            **slot_market_db_identity(slot),
        }]
    other_reader = FactorReader(str(tmp_path / "absent.duckdb"), data_bindings={}, slot_id=other.slot_id,
                                factor_values_store=store, frequency=frequency)
    np.testing.assert_allclose(other_reader.get_factor_values("alpha", START, END),
                               panel.alpha.unstack() + 10, rtol=0, atol=1e-12)
    pinned_other = other_reader.with_market_input(slot=None, slot_id=None, frequency=None, factor_values_store=None,
                                                 factor_identities=ids)
    with pytest.raises(ValueError, match="factor_identities"):
        pinned_other.get_factor_values("alpha", START, END)
    loaded = load_factor_data(conn, "factor_values", FACTORS, START, END, config=NO_CACHE,
                              slot=slot, frequency=frequency, factor_values_store=store)
    expected = panel.unstack().shift(1).stack(future_stack=True).dropna(how="all")
    pd.testing.assert_index_equal(loaded.index, expected.index)
    np.testing.assert_allclose(loaded, expected, rtol=0, atol=1e-12, equal_nan=True)


@pytest.fixture
def local_model_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("QUANT_EXECUTION_DIR", str(tmp_path / "model-tracking"))
    yield
    from factor_synthesis.qlib_adapter.qlib_lgb_model import QLIB_LGB_AVAILABLE, _QLIB_INITIALIZED

    if QLIB_LGB_AVAILABLE and _QLIB_INITIALIZED:
        from qlib.workflow import R

        R.end_exp()


def _synthetic_orchestrator(tmp_path):
    captured = {}
    return SimpleNamespace(
        logger=LOG,
        config={
            **NO_CACHE,
            "rolling_window": {"predict_window_days": 5, "min_train_samples": 10},
            "factor_groups": {"all": {"factors": FACTORS}},
            "output_settings": {"composite_factor_prefix": "composite", "weights_output_dir": str(tmp_path)},
        },
        _apply_feature_preprocessing=apply_feature_preprocessing,
        direct_analyze_composite_factor=lambda series, name, **kwargs: captured.update({name: series}),
    ), captured


def _method_options():
    return {
        "target_period": 1, "target_cs_demean": False, "preprocess": {"alpha": "zscore", "beta": "zscore"},
        "lasso_cv_folds": 2, "random_state": 42, "lgb_n_estimators": 12, "lgb_min_child_samples": 4,
        "lgb_n_jobs": 1, "lgb_random_state": 42, "blend_with_original_ratio": 0.0,
    }




def _assert_exact_panel(actual, expected):
    pd.testing.assert_index_equal(actual.index, expected.index)
    pd.testing.assert_index_equal(actual.columns, expected.columns)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)




@pytest.mark.parametrize("instruments", [["BTC-USD"], ["BTC-USD", "ETH-USD", "SOL-USD"]], ids=["one", "many"])
def test_crypto_slot_without_equity_features_runs_models_and_causal_preprocessing(tmp_path, instruments, local_model_tracking):
    slot = _crypto_slot(tmp_path, instruments)
    store = MarketFactorValueStore(tmp_path / "store")
    _save_panel(store, slot, "1h", _inputs(slot, "1h", instruments))
    config = {"enabled": True}
    panel = load_factor_data(
        None, "factor_values", FACTORS, START, END, config=NO_CACHE, slot=slot, frequency="1h",
        factor_values_store=store, neutralization_config=config,
    )
    assert panel.index.get_level_values("date").tz is not None
    assert not apply_feature_preprocessing(panel, {"ALL": "zscore"}).empty
    modes = {"alpha": "time_series_zscore", "beta": "time_series_rank", "_ts_window": 4}
    prefix = panel.iloc[:100]
    changed = pd.concat([prefix, panel.iloc[100:] * 200])
    expected = apply_feature_preprocessing(prefix, modes)
    np.testing.assert_allclose(apply_feature_preprocessing(changed, modes).loc[prefix.index],
                               expected, rtol=0, atol=1e-12, equal_nan=True)
    one = panel.loc[panel.index.get_level_values("stock_code") == instruments[0]]
    # The first normalized value uses exactly the four preceding observations.
    historical = one.alpha.iloc[:4]
    assert apply_feature_preprocessing(one, modes).alpha.iloc[4] == pytest.approx(
        (one.alpha.iloc[4] - historical.mean()) / historical.std(ddof=0))
    with pytest.raises(ValueError, match="neutralization feature"):
        load_factor_data(None, "factor_values", FACTORS, START, END, config=NO_CACHE,
                         slot=slot, frequency="1h", factor_values_store=store,
                         neutralization_config={**config, "on_missing_aux": "error"})

    market_dates = panel.index.get_level_values("date").unique()
    sample = panel.loc[panel.index.get_level_values("date") < market_dates[-2]]
    labels = calculate_forward_returns(sample, None, "", "", period=1, slot=slot, frequency="1h")
    assert labels.attrs["label_exit_dates"][market_dates[0].isoformat()] == (market_dates[0] + pd.Timedelta(hours=2)).isoformat()
    prices = get_pricing_data(None, "", instruments, market_dates[:12], slot=slot, frequency="1h", future_periods=1)
    assert prices.index.get_level_values("date").nunique() == 13
    synth, captured = _synthetic_orchestrator(tmp_path)
    run_weighted_sum_synthesis(
        synth, "weighted", {
            "factor_group": "all", "weights": {"alpha": 1}, "preprocess": modes,
            "factor_names": FACTORS, "start_date": START, "end_date": END, "instruments": instruments,
        }, None, slot=slot, frequency="1h", factor_values_store=store,
    )
    weighted = captured["composite_weighted"]
    assert not weighted.empty
    assert [item["factor_id"] for item in weighted.attrs["market_factor_inputs"]] == ["alpha"]
    config = {**_method_options(), "preprocess": modes}
    config.pop("target_cs_demean")
    for run in [run_lasso_synthesis, run_lightgbm_synthesis]:
        scores, info = run(
            synth, panel, labels, "2026-01-05", "2026-01-15T12:00:00Z",
            config, slot=slot, frequency="1h",
        )
        assert not scores.empty and info["market_factor_inputs"]
        assert scores.index.get_level_values("date").min() > pd.Timestamp("2026-01-15T12:00:00Z")


def test_composite_factor_base_consumes_explicit_scheme_b_slot(tmp_path):
    from factor_synthesis.composite_factor_base import CompositeFactorBase
    from factor_synthesis.factor_composer import SynthesisConfig

    symbols = ["BTC-USD", "ETH-USD"]
    slot = _crypto_slot(tmp_path, symbols)
    store = MarketFactorValueStore(tmp_path / "store")
    identities = _save_panel(store, slot, "1h", _inputs(slot, "1h", symbols))
    factor = CompositeFactorBase(
        None, "crypto_composite", FACTORS,
        SynthesisConfig(method="pca", training_window=8, prediction_window=1, update_frequency=20,
                        min_observations=4, pca_n_components=1),
        START, END, stock_pool=symbols, auto_synthesize=False, results_database_path=str(tmp_path / "absent.duckdb"),
        data_bindings={}, slot=slot, slot_id=slot.slot_id, frequency="1h",
        factor_values_store=store, factor_identities=identities,
    )
    try:
        values = factor.get_factor_values()
        assert not values.empty
        assert values.index.tz is not None
        assert {item["factor_id"] for item in factor.get_synthesis_diagnostics()["market_factor_inputs"]} == set(FACTORS)
        assert store.read_factor_values(factor_id="crypto_composite", slot_id=slot.slot_id, frequency="1h").shape[0] > 0
    finally:
        factor.close()


def test_domestic_futures_skips_equity_exposures_and_reads_real_slot_prices():
    from factor_synthesis.core.factor_ops import neutralize_market_features
    from slots.cn_futures import build_cn_futures_slot
    from test_slot_futures import _populate

    start, end = "2026-01-05", "2026-01-09"
    instruments = ["CU2602.SHFE", "CU2603.SHFE"]
    with duckdb.connect(config={"threads": 1}) as conn:
        source, _ = _populate(conn, root="CU", exchange="SHFE", start=start, end=end, expiries=["2026-02", "2026-03"])
        slot = build_cn_futures_slot(conn, product_root="CU", start=start, end=end)
        actual = get_complete_price_data(None, "", instruments, start, end, slot=slot)
        reference = source.assign(date=pd.to_datetime(source.session_id)).pivot(
            index="date", columns="instrument_id", values="close",
        )
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
        panel = actual.stack().rename("alpha").to_frame()
        result = neutralize_market_features(panel, {"enabled": True}, slot=slot, start_date=start, end_date=end)
        _assert_exact_panel(result, panel)
        values = apply_feature_preprocessing(result, {"alpha": "zscore"})["alpha"]
        assert values.groupby(level="date").nunique().eq(2).all()


def test_loader_has_no_direct_ashare_price_table():
    from factor_synthesis.core import factor_data_loader

    assert "price_data_daily" not in Path(factor_data_loader.__file__).read_text()


def test_label_exit_conflict_and_exact_boundary_isolation(tmp_path, local_model_tracking):
    from factor_synthesis.pipeline.lasso_method import _strict_label_exit_index
    from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError

    instruments = ["BTC-USD", "ETH-USD", "SOL-USD"]
    slot = _crypto_slot(tmp_path, instruments)
    store = MarketFactorValueStore(tmp_path / "store")
    panel = _inputs(slot, "1h", instruments)
    _save_panel(store, slot, "1h", panel)
    loaded = load_factor_data(
        None, "factor_values", FACTORS, START, END, config=NO_CACHE,
        slot=slot, frequency="1h", factor_values_store=store,
    )
    dates = loaded.index.get_level_values("date")
    market_dates = dates.unique()
    sample = loaded.loc[dates < market_dates[-2]]
    labels = calculate_forward_returns(sample, None, "", "", period=1, slot=slot, frequency="1h")
    entry_09 = pd.Timestamp("2026-01-15T09:00:00Z")
    entry_10 = pd.Timestamp("2026-01-15T10:00:00Z")
    boundary = pd.Timestamp("2026-01-15T12:00:00Z")
    assert labels.attrs["label_exit_dates"][entry_10.isoformat()] == boundary.isoformat()

    eligible = _strict_label_exit_index(
        labels.index, {"_label_exit_dates": labels.attrs["label_exit_dates"]}, boundary,
    )
    eligible_dates = eligible.get_level_values("date").unique()
    assert entry_09 in eligible_dates
    assert entry_10 not in eligible_dates

    synth, _ = _synthetic_orchestrator(tmp_path)
    modes = {"alpha": "time_series_zscore", "beta": "time_series_rank", "_ts_window": 4}
    base_cfg = {**_method_options(), "preprocess": modes}
    base_cfg.pop("target_cs_demean")

    forged = {
        **labels.attrs["label_exit_dates"],
        entry_10.isoformat(): (entry_10 + pd.Timedelta(hours=1)).isoformat(),
    }
    for run in (run_lasso_synthesis, run_lightgbm_synthesis):
        with pytest.raises(ValueError, match="label_exit_dates"):
            run(synth, loaded, labels, "2026-01-05", "2026-01-15T11:00:00Z",
                {**base_cfg, "_label_exit_dates": forged}, slot=slot, frequency="1h")

    boundary_only = loaded.loc[loaded.index.get_level_values("date") >= entry_10]
    boundary_labels = labels.loc[labels.index.get_level_values("date") >= entry_10]
    boundary_labels.attrs["label_exit_dates"] = {
        k: v for k, v in labels.attrs["label_exit_dates"].items() if pd.Timestamp(k) >= entry_10
    }
    for run in (run_lasso_synthesis, run_lightgbm_synthesis):
        with pytest.raises(LabelIsolationError):
            run(synth, boundary_only, boundary_labels, "2026-01-15T10:00:00Z", "2026-01-15T11:00:00Z",
                base_cfg, slot=slot, frequency="1h")






def test_ashare_5min_market_cap_availability_and_neutralization(tmp_path):
    from factor_synthesis.core.pool_manager import MARKET_CAP_FEATURE, aligned_aux_feature

    start, end = "2026-01-05", "2026-01-07"
    stocks = STOCKS[:8]
    with duckdb.connect(config={"threads": 1}) as conn:
        declare_market_db(conn, 'cn_equity')
        slot = build_jq_ashare_local(conn, start=start, end=end)
        days = slot.calendar.sessions(start, end)
        cap_rows = []
        ind_rows = []
        for day_idx, day in enumerate(days):
            for s_idx, stock in enumerate(stocks):
                cap_rows.append((day.date(), stock, 20.0 + day_idx * 10.0 + s_idx / 4.0))
                ind_rows.append((day.date(), stock, "I1" if s_idx % 2 == 0 else "I2"))
        conn.execute("CREATE TABLE factor_data (date DATE, stock_code VARCHAR, natural_log_of_market_cap DOUBLE)")
        conn.executemany("INSERT INTO factor_data VALUES (?, ?, ?)", cap_rows)
        conn.execute("CREATE TABLE industry_membership (date DATE, stock_code VARCHAR, sw_l1_code VARCHAR)")
        conn.executemany("INSERT INTO industry_membership VALUES (?, ?, ?)", ind_rows)

        # Unshifted exposure on the full range: 2026-01-05 unpublished -> NaN; 2026-01-06 uses 2026-01-05 cap.
        raw_cap = aligned_aux_feature(slot, MARKET_CAP_FEATURE, stocks, start, end, frequency="5min", shift=0)
        jan5_bars = slot.calendar.bar_grid("5min", "2026-01-05", "2026-01-05").timestamp_utc
        jan6_bars = slot.calendar.bar_grid("5min", "2026-01-06", "2026-01-06").timestamp_utc
        jan7_bars = slot.calendar.bar_grid("5min", "2026-01-07", "2026-01-07").timestamp_utc
        assert not raw_cap.index.get_level_values("date").isin(jan5_bars).any()
        jan6_vals = raw_cap.loc[raw_cap.index.get_level_values("date").isin(jan6_bars)]
        assert not jan6_vals.empty
        np.testing.assert_allclose(jan6_vals.loc[(jan6_bars[0], stocks[0])], 20.0, rtol=0, atol=1e-12)
        jan7_vals = raw_cap.loc[raw_cap.index.get_level_values("date").isin(jan7_bars)]
        np.testing.assert_allclose(jan7_vals.loc[(jan7_bars[0], stocks[0])], 30.0, rtol=0, atol=1e-12)

        # Requesting 2026-01-06..2026-01-07 directly still resolves the prior trading day's (2026-01-05) cap.
        window_cap = aligned_aux_feature(slot, MARKET_CAP_FEATURE, stocks, "2026-01-06", "2026-01-07", frequency="5min", shift=0)
        np.testing.assert_allclose(window_cap.loc[(jan6_bars[0], stocks[0])], 20.0, rtol=0, atol=1e-12)

        # T-1 market cap loader preserves shift(1) while seeing prior-session publications.
        lagged_cap = load_market_cap_data(None, "factor_data", start, end, set(stocks), slot=slot, frequency="5min")
        assert (jan6_bars[0], stocks[0]) not in lagged_cap.index
        np.testing.assert_allclose(lagged_cap.loc[(jan6_bars[1], stocks[0])], 20.0, rtol=0, atol=1e-12)

        # Save 5min factor panel whose alpha is purely linear in that day's available market cap.
        store = MarketFactorValueStore(tmp_path / "store_5m")
        grid = slot.calendar.bar_grid("5min", start, end)
        panel_rows = grid.merge(pd.DataFrame({"instrument_id": stocks}), how="cross")
        day_to_base = {"2026-01-05": 20.0, "2026-01-06": 20.0, "2026-01-07": 30.0}
        stock_offset = {s: idx / 4.0 for idx, s in enumerate(stocks)}
        alpha_vals = [
            day_to_base[row.session_id] + stock_offset[row.instrument_id]
            for row in panel_rows.itertuples(index=False)
        ]
        idx = pd.MultiIndex.from_arrays([panel_rows.timestamp_utc, panel_rows.instrument_id], names=["date", "stock_code"])
        panel = pd.DataFrame({"alpha": alpha_vals, "beta": np.arange(len(idx), dtype=float)}, index=idx)
        identities = {}
        for factor in FACTORS:
            identity, payload = compute_factor_identity_hash(
                factor_id=factor, factor_class="tests.Alpha", source_sha256="a" * 64, params={},
                frequency="5min", session_view="ETH", analysis_mode="cross_sectional",
                bucket_strategy="segment_anchored", calendar_hash=slot.calendar.calendar_hash,
                slot_id=slot.slot_id, slot_config_hash="b" * 64, adjustment_policy="none",
                slot_data_fingerprint=bind_market_db_fingerprint({"fixture": "v1"}, slot_market_db_identity(slot)), code_version="synthetic",
            )
            record = build_identity_record(factor_identity_hash=identity, payload=payload, code_version_source="test")
            batch = panel_rows.assign(
                factor_id=factor, factor_identity_hash=identity, frequency="5min", session_view="ETH",
                factor_value=panel[factor].to_numpy(), source_build_id="fixture",
                calendar_hash=slot.calendar.calendar_hash, code_version="synthetic",
                created_at=pd.Timestamp("2026-02-01", tz="UTC"), batch_id="fixture",
            )
            store.append_factor_values(batch, identity_record=record, calendar=slot.calendar)
            identities[factor] = identity

        # Single unpublished day fails strict aux policy.
        with pytest.raises(ValueError, match="neutralization feature"):
            load_factor_data(
                None, "factor_values", FACTORS, "2026-01-05", "2026-01-05", config=NO_CACHE,
                slot=slot, frequency="5min", factor_values_store=store,
                neutralization_config={"enabled": True, "on_missing_aux": "error"},
            )

        neutral = load_factor_data(
            None, "factor_values", FACTORS, "2026-01-06", "2026-01-07", config=NO_CACHE,
            slot=slot, frequency="5min", factor_values_store=store,
            neutralization_config={"enabled": True, "on_missing_aux": "error"},
        )
        # Raw alpha on 2026-01-06 equals 2026-01-05's market cap -> residualized to zero on 2026-01-06 (shifted to bar >= 1).
        np.testing.assert_allclose(neutral.loc[jan6_bars[1], "alpha"], 0.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("frequency", ["1d", "5min", "1h"])
def test_interrupted_identity_registration_does_not_block_committed_reads(tmp_path, ashare, frequency):
    _, slot, _, _ = ashare
    store = MarketFactorValueStore(tmp_path / f"store_{frequency}")
    panel = _inputs(slot, frequency, STOCKS[:2])
    committed_ids = _save_panel(store, slot, frequency, panel)

    uncommitted_hash, payload = compute_factor_identity_hash(
        factor_id="alpha", factor_class="tests.Alpha", source_sha256="c" * 64, params={"aborted": True},
        frequency=frequency, session_view="ETH", analysis_mode="cross_sectional",
        bucket_strategy="session" if frequency == "1d" else "segment_anchored",
        calendar_hash=slot.calendar.calendar_hash, slot_id=slot.slot_id,
        slot_config_hash="b" * 64, adjustment_policy="none",
        slot_data_fingerprint=bind_market_db_fingerprint({"fixture": "v2"}, slot_market_db_identity(slot)), code_version="synthetic",
    )
    uncommitted_record = build_identity_record(
        factor_identity_hash=uncommitted_hash, payload=payload, code_version_source="test",
    )
    with store.identities_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(uncommitted_record, sort_keys=True) + "\n")

    assert len(store.list_identities(factor_id="alpha", slot_id=slot.slot_id, frequency=frequency)) == 2
    reader = FactorReader(
        str(tmp_path / "absent.duckdb"), data_bindings={}, slot=slot,
        factor_values_store=store, frequency=frequency,
    )
    assert reader.get_available_factors() == ["alpha", "beta"]
    values = reader.get_factor_values("alpha", START, END)
    np.testing.assert_allclose(values, panel.alpha.unstack(), rtol=0, atol=1e-12)
    assert values.attrs["market_factor_inputs"]["identities"]["alpha"] == committed_ids["alpha"]

    loaded = load_factor_data(
        None, "factor_values", ["alpha"], START, END, config=NO_CACHE,
        slot=slot, frequency=frequency, factor_values_store=store,
    )
    expected = panel[["alpha"]].unstack().shift(1).stack(future_stack=True).dropna(how="all")
    np.testing.assert_allclose(loaded, expected, rtol=0, atol=1e-12, equal_nan=True)

    pinned_committed = reader.with_market_input(
        slot=None, slot_id=None, frequency=None, factor_values_store=None,
        factor_identities={"alpha": committed_ids["alpha"]},
    )
    np.testing.assert_allclose(pinned_committed.get_factor_values("alpha", START, END),
                               panel.alpha.unstack(), rtol=0, atol=1e-12)

    pinned_uncommitted = reader.with_market_input(
        slot=None, slot_id=None, frequency=None, factor_values_store=None,
        factor_identities={"alpha": uncommitted_hash},
    )
    with pytest.raises(ValueError, match="factor_identities"):
        pinned_uncommitted.get_factor_values("alpha", START, END)
    with pytest.raises(ValueError, match="factor_identities"):
        load_factor_data(
            None, "factor_values", ["alpha"], START, END, config=NO_CACHE,
            slot=slot, frequency=frequency, factor_values_store=store,
            factor_identities={"alpha": uncommitted_hash},
        )


def test_size_conditional_prepared_panel_drops_full_day_and_series_label_exits(tmp_path):
    from factor_synthesis.pipeline.preprocessing import bind_synthesis_panel
    from factor_synthesis.pipeline.size_conditional_method import (
        FIXED_BUCKET_EDGES, FIXED_GATE_CANDIDATES, load_or_build_size_conditional_snapshot,
        prepare_size_conditional_panel, run_size_conditional_synthesis,
        size_conditional_snapshot_identity,
    )

    sc_start, sc_end = "2026-01-05", "2026-02-27"
    sc_stocks = [f"{i:06d}.XSHE" for i in range(1, 26)]
    with duckdb.connect(config={"threads": 1}) as conn:
        declare_market_db(conn, 'cn_equity')
        slot = build_jq_ashare_local(conn, start=sc_start, end=sc_end)
        days = slot.calendar.sessions(sc_start, sc_end)
        index = pd.MultiIndex.from_product([days, sc_stocks], names=["date", "stock_code"])
        rng = np.random.default_rng(2026)
        raw = pd.DataFrame(rng.normal(size=(len(index), 2)), index=index, columns=FACTORS)
        i = np.arange(len(raw), dtype=float)
        sources = raw.reset_index().assign(
            close=80 + i / 200 + (i % len(sc_stocks)),
            factor=1.0,
            natural_log_of_market_cap=21 + (i % len(sc_stocks)) / 4 + raw.beta.to_numpy() / 10,
        )
        conn.execute("""CREATE TABLE price_data_daily AS SELECT
            date::DATE AS date, stock_code, close, close - 1 AS open, close AS high,
            close AS low, 1.0 AS volume, factor FROM sources""")
        conn.execute("CREATE TABLE factor_data AS SELECT date::DATE AS date, stock_code, natural_log_of_market_cap FROM sources")

        store = MarketFactorValueStore(tmp_path / "store_sc")
        for factor in FACTORS:
            identity, payload = compute_factor_identity_hash(
                factor_id=factor, factor_class="tests.Alpha", source_sha256="a" * 64, params={},
                frequency="1d", session_view="ETH", analysis_mode="cross_sectional",
                bucket_strategy="session", calendar_hash=slot.calendar.calendar_hash,
                slot_id=slot.slot_id, slot_config_hash="b" * 64, adjustment_policy="none",
                slot_data_fingerprint=bind_market_db_fingerprint({"fixture": "v1"}, slot_market_db_identity(slot)), code_version="synthetic",
            )
            record = build_identity_record(factor_identity_hash=identity, payload=payload, code_version_source="test")
            grid = slot.calendar.bar_grid("1d", sc_start, sc_end)
            rows = grid.merge(pd.DataFrame({"instrument_id": sc_stocks}), how="cross")
            keys = pd.MultiIndex.from_arrays([pd.to_datetime(rows.session_id), rows.instrument_id], names=raw.index.names)
            batch = rows.assign(
                factor_id=factor, factor_identity_hash=identity, frequency="1d", session_view="ETH",
                factor_value=raw[factor].reindex(keys).to_numpy(), source_build_id="fixture",
                calendar_hash=slot.calendar.calendar_hash, code_version="synthetic",
                created_at=pd.Timestamp("2026-03-01", tz="UTC"), batch_id="fixture",
            )
            store.append_factor_values(batch, identity_record=record, calendar=slot.calendar)

        loaded = load_factor_data(
            None, "factor_values", FACTORS, sc_start, sc_end, config=NO_CACHE,
            slot=slot, factor_values_store=store,
        )
        model_panel = loaded.loc[loaded.index.get_level_values("date") < pd.Timestamp(sc_end)]
        base_labels = calculate_forward_returns(model_panel, None, "", "", period=1, slot=slot)
        labels = (0.04 * loaded["alpha"].reindex(base_labels.index) + 0.001).rename("forward_return")
        labels.attrs = dict(base_labels.attrs)

        cap = load_market_cap_data(None, "factor_data", sc_start, sc_end, set(sc_stocks), slot=slot)
        dropped_day = loaded.index.get_level_values("date").min()
        cap.loc[cap.index.get_level_values("date") == dropped_day] = np.nan

        synth, _ = _synthetic_orchestrator(tmp_path)
        synth.config["performance_evaluation"] = {
            "top_k_strategy": {
                "enabled": True, "top_k": 10, "holding_period": 1,
                "cost_rate": 0.001, "enable_tday_filter": True,
            }
        }
        sc_cfg = {
            "target_period": 1,
            "target_cs_demean": False,
            "preprocess": {"ALL": "zscore"},
            "lasso_alpha_range": [0.001, 0.01],
            "n_alphas": 2,
            "l1_ratio_range": [0.9],
            "size_conditioning": {
                "bucket_edges": list(FIXED_BUCKET_EDGES),
                "gate_candidates": [list(g) for g in FIXED_GATE_CANDIDATES],
                "interaction_shrinkage_grid": [1.0],
                "inner_folds": 2,
                "embargo_days": 1,
                "stability_threshold": 0.6,
                "bootstrap_samples": 10,
            },
        }

        prepared = prepare_size_conditional_panel(synth, loaded, labels, cap, sc_cfg)
        dropped_key = dropped_day.strftime("%Y-%m-%d")
        assert dropped_key not in prepared["label_context"]["_label_exit_dates"]
        assert dropped_key in prepared["raw_returns"].attrs["label_exit_dates"]

        scores, info = run_size_conditional_synthesis(
            synth, prepared["predict_factors"], prepared["raw_returns"], cap,
            "2026-01-05", "2026-02-20", {**sc_cfg, **prepared["label_context"]},
            prepared_panel=prepared, slot=slot,
        )
        assert not scores.empty and info["selected_policy"]

        # pd.Series mappings remain supported, keep producer exit times authoritative, and reject missing/conflicting active keys.
        series_exits = pd.Series(prepared["label_context"]["_label_exit_dates"])
        identity = size_conditional_snapshot_identity(
            {}, {**sc_cfg, "_label_exit_dates": series_exits}, FACTORS, sc_start, sc_end, [],
        )
        assert identity["label_producer"]["exit_dates"] == prepared["label_context"]["_label_exit_dates"]

        series_labels = prepared["raw_returns"].copy()
        series_labels.attrs = dict(prepared["raw_returns"].attrs, label_exit_dates=pd.Series(labels.attrs["label_exit_dates"]))
        _, bound_cfg = bind_synthesis_panel(
            synth, prepared["predict_factors"], {**sc_cfg, "_label_exit_dates": series_exits},
            {"slot": slot}, series_labels,
        )
        assert isinstance(bound_cfg["_label_exit_dates"], pd.Series)
        assert bound_cfg["_label_exit_dates"].loc[dropped_key] == labels.attrs["label_exit_dates"][dropped_key]

        with pytest.raises(ValueError, match="label_exit_dates"):
            bind_synthesis_panel(
                synth, prepared["predict_factors"],
                {**sc_cfg, "_label_exit_dates": series_exits.iloc[1:]},
                {"slot": slot}, series_labels,
            )
        conflicting = series_exits.copy()
        conflicting.iloc[0] = "2026-01-01"
        with pytest.raises(ValueError, match="label_exit_dates"):
            bind_synthesis_panel(
                synth, prepared["predict_factors"],
                {**sc_cfg, "_label_exit_dates": conflicting},
                {"slot": slot}, series_labels,
            )

        series_panel = dict(prepared, raw_returns=series_labels)
        first_snap, first_info = load_or_build_size_conditional_snapshot(
            tmp_path / "sc_snapshot", identity, lambda: series_panel, enabled=True,
        )
        second_snap, second_info = load_or_build_size_conditional_snapshot(
            tmp_path / "sc_snapshot", identity, lambda: series_panel, enabled=True,
        )
        assert first_info["status"] == "miss"
        assert second_info["status"] == "hit"
        assert second_snap["raw_returns"].attrs["label_exit_dates"] == labels.attrs["label_exit_dates"]
        assert first_snap["label_context"] == second_snap["label_context"]


def test_ashare_5min_missing_market_cap_day_consistent_across_windows(tmp_path):
    from factor_synthesis.core.pool_manager import MARKET_CAP_FEATURE, aligned_aux_feature

    start, end = "2026-01-05", "2026-01-07"
    stocks = STOCKS[:8]
    with duckdb.connect(config={"threads": 1}) as conn:
        declare_market_db(conn, 'cn_equity')
        slot = build_jq_ashare_local(conn, start=start, end=end)
        days = slot.calendar.sessions(start, end)
        # 2026-01-05 and 2026-01-07 have market cap, while 2026-01-06 is missing.
        cap_rows = []
        ind_rows = []
        for day_idx, day in enumerate(days):
            for s_idx, stock in enumerate(stocks):
                if day != pd.Timestamp("2026-01-06"):
                    cap_rows.append((day.date(), stock, 20.0 + day_idx * 10.0 + s_idx / 4.0))
                ind_rows.append((day.date(), stock, "I1" if s_idx % 2 == 0 else "I2"))
        conn.execute("CREATE TABLE factor_data (date DATE, stock_code VARCHAR, natural_log_of_market_cap DOUBLE)")
        conn.executemany("INSERT INTO factor_data VALUES (?, ?, ?)", cap_rows)
        conn.execute("CREATE TABLE industry_membership (date DATE, stock_code VARCHAR, sw_l1_code VARCHAR)")
        conn.executemany("INSERT INTO industry_membership VALUES (?, ?, ?)", ind_rows)

        jan6_bars = slot.calendar.bar_grid("5min", "2026-01-06", "2026-01-06").timestamp_utc
        jan7_bars = slot.calendar.bar_grid("5min", "2026-01-07", "2026-01-07").timestamp_utc

        cap_05_07 = aligned_aux_feature(slot, MARKET_CAP_FEATURE, stocks, "2026-01-05", "2026-01-07", frequency="5min", shift=0)
        cap_06_07 = aligned_aux_feature(slot, MARKET_CAP_FEATURE, stocks, "2026-01-06", "2026-01-07", frequency="5min", shift=0)
        cap_07_07 = aligned_aux_feature(slot, MARKET_CAP_FEATURE, stocks, "2026-01-07", "2026-01-07", frequency="5min", shift=0)

        # 2026-01-06 receives 2026-01-05's market cap; 2026-01-07 has no valid prior-session cap across all windows.
        np.testing.assert_allclose(cap_05_07.loc[(jan6_bars[0], stocks[0])], 20.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cap_06_07.loc[(jan6_bars[0], stocks[0])], 20.0, rtol=0, atol=1e-12)
        assert not cap_05_07.index.get_level_values("date").isin(jan7_bars).any()
        assert not cap_06_07.index.get_level_values("date").isin(jan7_bars).any()
        assert cap_07_07.empty

        store = MarketFactorValueStore(tmp_path / "store_5m_gap")
        grid = slot.calendar.bar_grid("5min", start, end)
        panel_rows = grid.merge(pd.DataFrame({"instrument_id": stocks}), how="cross")
        stock_offset = {s: idx / 4.0 for idx, s in enumerate(stocks)}
        alpha_vals = [20.0 + stock_offset[row.instrument_id] for row in panel_rows.itertuples(index=False)]
        idx = pd.MultiIndex.from_arrays([panel_rows.timestamp_utc, panel_rows.instrument_id], names=["date", "stock_code"])
        panel = pd.DataFrame({"alpha": alpha_vals, "beta": np.arange(len(idx), dtype=float)}, index=idx)
        for factor in FACTORS:
            identity, payload = compute_factor_identity_hash(
                factor_id=factor, factor_class="tests.Alpha", source_sha256="a" * 64, params={},
                frequency="5min", session_view="ETH", analysis_mode="cross_sectional",
                bucket_strategy="segment_anchored", calendar_hash=slot.calendar.calendar_hash,
                slot_id=slot.slot_id, slot_config_hash="b" * 64, adjustment_policy="none",
                slot_data_fingerprint=bind_market_db_fingerprint({"fixture": "v1"}, slot_market_db_identity(slot)), code_version="synthetic",
            )
            record = build_identity_record(factor_identity_hash=identity, payload=payload, code_version_source="test")
            batch = panel_rows.assign(
                factor_id=factor, factor_identity_hash=identity, frequency="5min", session_view="ETH",
                factor_value=panel[factor].to_numpy(), source_build_id="fixture",
                calendar_hash=slot.calendar.calendar_hash, code_version="synthetic",
                created_at=pd.Timestamp("2026-02-01", tz="UTC"), batch_id="fixture",
            )
            store.append_factor_values(batch, identity_record=record, calendar=slot.calendar)

        for neu_cfg in (
            {"enabled": True, "industry": False, "on_missing_aux": "skip"},
            {"enabled": True, "industry": True, "on_missing_aux": "skip"},
        ):
            wide = load_factor_data(
                None, "factor_values", FACTORS, "2026-01-05", "2026-01-07", config=NO_CACHE,
                slot=slot, frequency="5min", factor_values_store=store, neutralization_config=neu_cfg,
            )
            mid = load_factor_data(
                None, "factor_values", FACTORS, "2026-01-06", "2026-01-07", config=NO_CACHE,
                slot=slot, frequency="5min", factor_values_store=store, neutralization_config=neu_cfg,
            )
            narrow = load_factor_data(
                None, "factor_values", FACTORS, "2026-01-07", "2026-01-07", config=NO_CACHE,
                slot=slot, frequency="5min", factor_values_store=store, neutralization_config=neu_cfg,
            )
            jan6_overlap = wide.index.intersection(mid.index)
            jan6_overlap = jan6_overlap[jan6_overlap.get_level_values("date").isin(jan6_bars[1:])]
            _assert_exact_panel(wide.loc[jan6_overlap], mid.loc[jan6_overlap])

            jan7_overlap = wide.index.intersection(narrow.index)
            assert len(jan7_overlap) == len(jan7_bars[1:]) * len(stocks)
            _assert_exact_panel(wide.loc[jan7_overlap], narrow.loc[jan7_overlap])
            _assert_exact_panel(mid.loc[jan7_overlap], narrow.loc[jan7_overlap])
            if not neu_cfg["industry"]:
                np.testing.assert_allclose(
                    narrow.loc[(jan7_bars[1], stocks[0]), "alpha"], 20.0, rtol=0, atol=1e-12,
                )


def test_lasso_and_lightgbm_align_non_null_labels_and_enforce_active_exits(tmp_path, ashare, local_model_tracking):
    from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError

    _, slot, raw, _ = ashare
    stocks = STOCKS[:2]
    panel_raw = raw.loc[raw.index.get_level_values("stock_code").isin(stocks)]
    store = MarketFactorValueStore(tmp_path / "store_nan_labels")
    _save_panel(store, slot, "1d", panel_raw)
    loaded = load_factor_data(
        None, "factor_values", FACTORS, START, END, config=NO_CACHE,
        slot=slot, factor_values_store=store,
    )
    model_panel = loaded.loc[loaded.index.get_level_values("date") < pd.Timestamp(END)]
    base_labels = calculate_forward_returns(model_panel, None, "", "", period=1, slot=slot)

    # Retain 2026-01-06 in the factor panel while making all labels on that date NaN
    # and keeping producer exits as a pd.Series covering only valid label dates.
    nan_day = pd.Timestamp("2026-01-06")
    nan_key = nan_day.strftime("%Y-%m-%d")
    labels_with_gap = base_labels.reindex(model_panel.index)
    labels_with_gap.loc[labels_with_gap.index.get_level_values("date") == nan_day] = np.nan
    active_exits = pd.Series({
        day: exit_day
        for day, exit_day in base_labels.attrs["label_exit_dates"].items()
        if day != nan_key
    })
    labels_with_gap.attrs = dict(base_labels.attrs, label_exit_dates=active_exits)

    synth, _ = _synthetic_orchestrator(tmp_path)
    synth.config["rolling_window"]["min_train_samples"] = 4
    cfg = _method_options()

    for run in (run_lasso_synthesis, run_lightgbm_synthesis):
        scores, info = run(synth, loaded, labels_with_gap, "2026-01-06", "2026-01-22", cfg, slot=slot)
        assert not scores.empty and info

    # Missing exit on an active (non-null) label date still fails isolation.
    active_day = "2026-01-07"
    incomplete_labels = labels_with_gap.copy()
    incomplete_labels.attrs = dict(
        base_labels.attrs,
        label_exit_dates=active_exits.drop(labels=[active_day]),
    )
    for run in (run_lasso_synthesis, run_lightgbm_synthesis):
        with pytest.raises(LabelIsolationError, match="missing label exit"):
            run(synth, loaded, incomplete_labels, "2026-01-06", "2026-01-22", cfg, slot=slot)

    # Conflicting configured exit on an active label date still fails validation.
    conflicting_cfg = {
        **cfg,
        "_label_exit_dates": active_exits.copy(),
    }
    conflicting_cfg["_label_exit_dates"].loc[active_day] = "2026-01-01"
    for run in (run_lasso_synthesis, run_lightgbm_synthesis):
        with pytest.raises(ValueError, match="label_exit_dates"):
            run(synth, loaded, labels_with_gap, "2026-01-06", "2026-01-22", conflicting_cfg, slot=slot)
