"""Regression tests for synthesis training label and final evaluation alignment."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import run_factor_synthesis
from backtest_and_analysis.alphalens_runner import AlphalensRunner
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from calendar_test_support import seed_calendar
from factor_synthesis.core.factor_data_loader import process_raw_factor_df
from factor_synthesis.core.factor_ops import calculate_forward_returns
from utils.session_calendar import CalendarCoverageError


def _build_alternating_panel(
    *,
    n_signal_dates: int = 15,
    extra_future_sessions: int = 2,
) -> tuple[pd.DatetimeIndex, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct an alternating-return market and loader-shifted factor panel.

    For trading session index t:
    - 1D close-to-close return close[t+1] / close[t] - 1 = (-1)^t * s_i * delta.
    - Next-day 1D return close[t+2] / close[t+1] - 1 has strictly opposite sign.
    - Raw factor on raw date u is (-1)^(u+1) * s_i so that after loader shift(1),
      trading row t holds (-1)^t * s_i, giving +1.0 IC at lag=0 and -1.0 IC at lag=1.
    """
    stocks = ["S0", "S1", "S2", "S3", "S4"]
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    delta = 0.01

    total_sessions = 1 + n_signal_dates + extra_future_sessions
    all_dates = pd.bdate_range("2025-01-02", periods=total_sessions)

    closes = np.zeros((total_sessions, len(stocks)), dtype=float)
    closes[0, :] = 100.0
    for t in range(total_sessions - 1):
        sign = 1.0 if t % 2 == 0 else -1.0
        step_ret = sign * scores * delta
        closes[t + 1, :] = closes[t, :] * (1.0 + step_ret)

    price_rows = []
    for t, day in enumerate(all_dates):
        for i, stock in enumerate(stocks):
            price_rows.append(
                {
                    "date": day,
                    "stock_code": stock,
                    "close": float(closes[t, i]),
                    "factor": 1.0,
                    "paused": False,
                }
            )
    price_df = pd.DataFrame(price_rows)

    raw_rows = []
    for u in range(n_signal_dates):
        sign_next = 1.0 if (u + 1) % 2 == 0 else -1.0
        for i, stock in enumerate(stocks):
            raw_rows.append(
                {
                    "date": all_dates[u],
                    "stock_code": stock,
                    "factor_name": "comp_signal",
                    "factor_value": float(sign_next * scores[i]),
                }
            )
    raw_df = pd.DataFrame(raw_rows)
    close_wide = price_df.pivot(index="date", columns="stock_code", values="close").sort_index()
    return all_dates, stocks, price_df, raw_df, close_wide


def _seed_price_db(db_path: Path, price_df: pd.DataFrame, all_dates: pd.DatetimeIndex) -> None:
    start = all_dates.min().strftime("%Y-%m-%d")
    end = all_dates.max().strftime("%Y-%m-%d")
    with duckdb.connect(str(db_path)) as conn:
        conn.register("_test_prices", price_df)
        conn.execute("CREATE TABLE price_data_daily AS SELECT * FROM _test_prices")
        seed_calendar(conn, start=start, end=end)


def _make_synthesizer(
    price_db_path: Path,
    *,
    use_precomputed_returns: bool = False,
    cache_dir: Path | None = None,
    quantiles: int = 5,
    ic_periods: tuple[int, ...] = (1,),
    execution_lag: int | None = None,
) -> run_factor_synthesis.FactorSynthesizer:
    synth = object.__new__(run_factor_synthesis.FactorSynthesizer)
    synth.logger = logging.getLogger("test_run_factor_synthesis_eval_alignment")
    synth.price_duckdb_path = str(price_db_path)
    synth.price_conn = None
    synth.price_table = "price_data_daily"
    synth.fundamental_table = "factor_data"
    synth.frequency = "1d"
    synth.pit_config = {"enabled": True}
    synth.skip_months = None
    synth.skip_months_evaluation = False
    synth.result_writer = object()

    pe_cfg: dict[str, object] = {
        "quantiles": quantiles,
        "ic_periods": list(ic_periods),
        "allow_high_data_loss": True,
        "top_k_strategy": {"enabled": False},
    }
    if execution_lag is not None:
        pe_cfg["execution_lag"] = execution_lag
    if cache_dir is not None:
        pe_cfg["forward_returns_cache_dir"] = str(cache_dir)

    synth.config = {
        "use_precomputed_returns": use_precomputed_returns,
        "performance_evaluation": pe_cfg,
    }
    return synth


def _load_shifted_composite(
    raw_df: pd.DataFrame,
    price_conn: duckdb.DuckDBPyConnection,
    all_dates: pd.DatetimeIndex,
    n_signal_dates: int,
) -> pd.DataFrame:
    return process_raw_factor_df(
        raw_df,
        ["comp_signal"],
        all_dates[0].strftime("%Y-%m-%d"),
        all_dates[n_signal_dates].strftime("%Y-%m-%d"),
        config={"missing_data_strategy": "drop"},
        price_conn=price_conn,
    )


def _daily_spearman_ic(factor_wide: pd.DataFrame, returns_wide: pd.DataFrame) -> pd.Series:
    return factor_wide.corrwith(returns_wide, axis=1, method="spearman").dropna()


@pytest.mark.parametrize("use_precomputed_returns", [False, True])
def test_alternating_returns_training_and_eval_1d_ic_align(
    tmp_path: Path, use_precomputed_returns: bool
) -> None:
    """Training label IC and final 1D evaluation IC must both be +1.0 under alternating returns."""
    n_signal_dates = 15
    all_dates, stocks, price_df, raw_df, close_wide = _build_alternating_panel(
        n_signal_dates=n_signal_dates,
        extra_future_sessions=2,
    )
    db_path = tmp_path / "prices.duckdb"
    _seed_price_db(db_path, price_df, all_dates)

    cache_dir = tmp_path / "fr_cache"
    synth = _make_synthesizer(
        db_path,
        use_precomputed_returns=use_precomputed_returns,
        cache_dir=cache_dir,
    )
    try:
        conn = synth._get_price_conn()
        shifted_factor = _load_shifted_composite(raw_df, conn, all_dates, n_signal_dates)
        assert len(shifted_factor) == n_signal_dates * len(stocks)

        # Verify strict alternating 1D return signs in the synthetic market.
        ret_t = (close_wide.shift(-1) / close_wide - 1.0).loc[shifted_factor.index.get_level_values("date").unique()]
        ret_t_plus_1 = (close_wide.shift(-2) / close_wide.shift(-1) - 1.0).loc[
            shifted_factor.index.get_level_values("date").unique()
        ]
        cross_corr = ret_t.corrwith(ret_t_plus_1, axis=1, method="spearman").dropna()
        np.testing.assert_allclose(cross_corr, -1.0, rtol=0, atol=1e-12)

        # 1. Training label IC (factor_ops.calculate_forward_returns uses CLOSE_TO_CLOSE).
        train_labels = calculate_forward_returns(
            shifted_factor,
            conn,
            "price_data_daily",
            "factor_data",
            period=1,
            pit_enabled=True,
        )
        factor_wide = shifted_factor["comp_signal"].unstack("stock_code")
        train_ic = _daily_spearman_ic(factor_wide, train_labels.unstack("stock_code"))
        assert len(train_ic) == n_signal_dates
        np.testing.assert_allclose(train_ic, 1.0, rtol=0, atol=1e-12)

        # 2. Final evaluation via _get_pricing_data + _run_alphalens_analysis.
        dates = factor_wide.index
        pricing = synth._get_pricing_data(stocks, dates)
        results = synth._run_alphalens_analysis(factor_wide, pricing, "SYN_COMP", max_loss=0.99)
        assert results is not None and "ic" in results
        eval_ic = results["ic"]["1D"].dropna()
        assert len(eval_ic) == n_signal_dates
        np.testing.assert_allclose(eval_ic, 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(eval_ic.mean(), train_ic.mean(), rtol=0, atol=1e-12)

        if use_precomputed_returns:
            cache_files = list(cache_dir.glob("forward_returns_*.pkl"))
            assert len(cache_files) == 1
            assert cache_files[0].name.endswith("_lag0.pkl")

            # Seed/create a lag=1 cache entry to prove lag0 does not cross-talk with lag1.
            lag1_calc = ForwardReturnsCalculator(
                cache_dir=str(cache_dir),
                trading_calendar=pricing.attrs.get("trading_calendar"),
            )
            lag1_returns = lag1_calc.calculate_or_load_forward_returns(
                pricing,
                stocks,
                dates.min().strftime("%Y-%m-%d"),
                dates.max().strftime("%Y-%m-%d"),
                (1,),
                execution_lag=1,
            )
            lag1_ic = _daily_spearman_ic(factor_wide, lag1_returns["1D"].unstack("asset"))
            np.testing.assert_allclose(lag1_ic, -1.0, rtol=0, atol=1e-12)
            assert any(p.name.endswith("_lag1.pkl") for p in cache_dir.glob("forward_returns_*.pkl"))

            # Re-running default lag0 hits lag0 cache and stays +1.0.
            reloaded = synth._run_alphalens_analysis(factor_wide, pricing, "SYN_COMP", max_loss=0.99)
            np.testing.assert_allclose(reloaded["ic"]["1D"].dropna(), 1.0, rtol=0, atol=1e-12)
    finally:
        synth._close_conn_attr("price_conn")


@pytest.mark.parametrize("use_precomputed_returns", [False, True])
def test_direct_analyze_composite_factor_boundary_and_persistence(
    tmp_path: Path, use_precomputed_returns: bool
) -> None:
    """direct_analyze_composite_factor works with exact max(ic_periods) future sessions and yields +1.0 IC."""
    n_signal_dates = 15
    # Only 1 future session after dates.max(): sufficient for lag=0, period=1; lag=1 would fail coverage.
    all_dates, _, price_df, raw_df, _ = _build_alternating_panel(
        n_signal_dates=n_signal_dates,
        extra_future_sessions=1,
    )
    db_path = tmp_path / "prices_exact_boundary.duckdb"
    _seed_price_db(db_path, price_df, all_dates)

    cache_dir = tmp_path / "fr_cache"
    synth = _make_synthesizer(
        db_path,
        use_precomputed_returns=use_precomputed_returns,
        cache_dir=cache_dir,
        ic_periods=(1,),
    )
    saved: dict[str, dict] = {}
    synth._save_analysis_results = lambda name, res: saved.update({name: res}) or True

    try:
        conn = synth._get_price_conn()
        shifted_factor = _load_shifted_composite(raw_df, conn, all_dates, n_signal_dates)
        composite_series = shifted_factor["comp_signal"]

        synth.direct_analyze_composite_factor(composite_series, "SYN_BOUNDARY")
        assert "SYN_BOUNDARY" in saved
        eval_ic = saved["SYN_BOUNDARY"]["ic"]["1D"].dropna()
        assert len(eval_ic) == n_signal_dates
        np.testing.assert_allclose(eval_ic, 1.0, rtol=0, atol=1e-12)

        # Explicitly requesting execution_lag=1 on this 1-future-session DB fails boundary coverage.
        with pytest.raises(CalendarCoverageError, match="fewer than 2 sessions"):
            synth.direct_analyze_composite_factor(
                composite_series, "SYN_BOUNDARY_LAG1", execution_lag=1
            )
    finally:
        synth._close_conn_attr("price_conn")


@pytest.mark.parametrize("via_config", [False, True])
@pytest.mark.parametrize("use_precomputed_returns", [False, True])
def test_explicit_execution_lag_override_and_validation(
    tmp_path: Path, via_config: bool, use_precomputed_returns: bool
) -> None:
    """Explicit execution_lag override (via arg or config) propagates across pricing, cache, and runner."""
    n_signal_dates = 15
    all_dates, stocks, price_df, raw_df, _ = _build_alternating_panel(
        n_signal_dates=n_signal_dates,
        extra_future_sessions=2,
    )
    db_path = tmp_path / "prices_override.duckdb"
    _seed_price_db(db_path, price_df, all_dates)

    cache_dir = tmp_path / "fr_cache"
    synth = _make_synthesizer(
        db_path,
        use_precomputed_returns=use_precomputed_returns,
        cache_dir=cache_dir,
        execution_lag=1 if via_config else None,
    )
    saved: dict[str, dict] = {}
    synth._save_analysis_results = lambda name, res: saved.update({name: res}) or True

    try:
        conn = synth._get_price_conn()
        shifted_factor = _load_shifted_composite(raw_df, conn, all_dates, n_signal_dates)
        composite_series = shifted_factor["comp_signal"]

        if via_config:
            synth.direct_analyze_composite_factor(composite_series, "SYN_LAG1")
        else:
            synth.direct_analyze_composite_factor(
                composite_series, "SYN_LAG1", execution_lag=1
            )

        eval_ic = saved["SYN_LAG1"]["ic"]["1D"].dropna()
        assert len(eval_ic) == n_signal_dates
        np.testing.assert_allclose(eval_ic, -1.0, rtol=0, atol=1e-12)

        if use_precomputed_returns:
            cache_files = list(cache_dir.glob("forward_returns_*.pkl"))
            assert len(cache_files) == 1
            assert cache_files[0].name.endswith("_lag1.pkl")

        for bad_lag in (-1, True, "0"):
            with pytest.raises(ValueError, match="execution_lag must be a non-negative integer"):
                synth._run_alphalens_analysis(
                    composite_series.unstack("stock_code"),
                    pd.Series(dtype=float),
                    "SYN_BAD",
                    execution_lag=bad_lag,  # type: ignore[arg-type]
                )
            with pytest.raises(ValueError, match="execution_lag must be a non-negative integer"):
                synth.direct_analyze_composite_factor(
                    composite_series,
                    "SYN_BAD",
                    execution_lag=bad_lag,  # type: ignore[arg-type]
                )
    finally:
        synth._close_conn_attr("price_conn")


def test_standalone_single_factor_entrypoints_keep_default_execution_lag_1(tmp_path: Path) -> None:
    """Standalone single-factor defaults must remain execution_lag=1."""
    assert inspect.signature(AlphalensRunner.__init__).parameters["execution_lag"].default == 1
    assert AlphalensRunner().execution_lag == 1
    assert (
        inspect.signature(ForwardReturnsCalculator.calculate_forward_returns)
        .parameters["execution_lag"]
        .default
        == 1
    )
    assert (
        inspect.signature(ForwardReturnsCalculator.calculate_or_load_forward_returns)
        .parameters["execution_lag"]
        .default
        == 1
    )

    n_signal_dates = 15
    all_dates, stocks, price_df, raw_df, _ = _build_alternating_panel(
        n_signal_dates=n_signal_dates,
        extra_future_sessions=2,
    )
    # Raw single-factor signal on date u predicts close[u+2]/close[u+1] - 1 (execution_lag=1).
    raw_wide = (
        raw_df.pivot(index="date", columns="stock_code", values="factor_value")
        .sort_index()
    )
    pricing = (
        price_df.set_index(["stock_code", "date"])["close"]
        .rename_axis(["asset", "date"])
        .sort_index()
    )
    from utils.trading_calendar import TradingCalendar

    calendar = TradingCalendar(all_dates, all_dates.min(), all_dates.max())
    pricing.attrs["trading_calendar"] = calendar

    # Default AlphalensRunner() uses execution_lag=1 -> +1.0 IC on unshifted raw factor.
    res_direct = AlphalensRunner(n_quantiles=5, periods=(1,)).run(raw_wide, pricing, max_loss=0.99)
    np.testing.assert_allclose(res_direct["ic"]["1D"].dropna(), 1.0, rtol=0, atol=1e-12)

    # Default ForwardReturnsCalculator.calculate_or_load_forward_returns uses execution_lag=1 -> _lag1 cache and +1.0 IC.
    calc = ForwardReturnsCalculator(cache_dir=str(tmp_path), trading_calendar=calendar)
    precomputed = calc.calculate_or_load_forward_returns(
        pricing,
        stocks,
        all_dates[0].strftime("%Y-%m-%d"),
        all_dates[n_signal_dates - 1].strftime("%Y-%m-%d"),
        (1,),
    )
    cache_files = list(tmp_path.glob("forward_returns_*.pkl"))
    assert len(cache_files) == 1 and cache_files[0].name.endswith("_lag1.pkl")
    res_precomputed = AlphalensRunner(n_quantiles=5, periods=(1,)).run(
        raw_wide, pricing, forward_returns=precomputed, max_loss=0.99
    )
    np.testing.assert_allclose(res_precomputed["ic"]["1D"].dropna(), 1.0, rtol=0, atol=1e-12)
