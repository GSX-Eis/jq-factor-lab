"""Regression tests for split/dividend factor-adjusted return labels across all three entry points."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from backtest_and_analysis.price_fetcher import PriceFetcher
from calendar_test_support import seed_calendar
from factor_synthesis.composite_factor_base import CompositeFactorBase
from factor_synthesis.factor_composer import SynthesisConfig
from market_db_samples import declare_market_db
from slots.jq_ashare_local import build_jq_ashare_local, daily_slot_config
from slots.jq_ashare_local.feed import read_analysis_price_rows
from utils.factor_base import FactorBase
from utils.market_factor import (
    MarketFactorValueStore,
    build_identity_record,
    compute_factor_identity_hash,
)
from utils.market_db.identity import bind_market_db_fingerprint, slot_market_db_identity
from utils.pit import (
    PRICE_ADJUSTED_ATTR,
    apply_pit_adjustment,
)
from utils.price_adjustment import adjust_prices_by_factor


SPLIT_START = "2024-01-02"
SPLIT_DAY = "2024-01-03"
SPLIT_EXTENDED = "2024-01-04"


def _create_split_db(path: Path, *, include_third_day: bool = False) -> None:
    """Create a synthetic A-share DuckDB with a 1-for-2 split on stock C (and 1-for-2 split on single stock S)."""
    rows = [
        # Single-stock 1-for-2 split: 100 -> 50, factor 1 -> 2 => 0% return
        ("S", SPLIT_START, 100.0, 100.0, 100.0, 100.0, 1000.0, 100000.0, 1.0),
        ("S", SPLIT_DAY, 50.0, 50.0, 50.0, 50.0, 2000.0, 100000.0, 2.0),
        # 3-stock Rank IC fixture: A=+1%, B=+2%, C=+3% (100 -> 51.5 with factor 1 -> 2)
        ("A", SPLIT_START, 100.0, 100.0, 100.0, 100.0, 1000.0, 100000.0, 1.0),
        ("A", SPLIT_DAY, 101.0, 101.0, 101.0, 101.0, 1000.0, 101000.0, 1.0),
        ("B", SPLIT_START, 100.0, 100.0, 100.0, 100.0, 1000.0, 100000.0, 1.0),
        ("B", SPLIT_DAY, 102.0, 102.0, 102.0, 102.0, 1000.0, 102000.0, 1.0),
        ("C", SPLIT_START, 100.0, 100.0, 100.0, 100.0, 1000.0, 100000.0, 1.0),
        ("C", SPLIT_DAY, 51.5, 51.5, 51.5, 51.5, 2000.0, 103000.0, 2.0),
    ]
    if include_third_day:
        rows.extend([
            ("S", SPLIT_EXTENDED, 26.0, 26.0, 26.0, 26.0, 4000.0, 104000.0, 4.0),
            ("A", SPLIT_EXTENDED, 102.01, 102.01, 102.01, 102.01, 1000.0, 102010.0, 1.0),
            ("B", SPLIT_EXTENDED, 104.04, 104.04, 104.04, 104.04, 1000.0, 104040.0, 1.0),
            ("C", SPLIT_EXTENDED, 26.5225, 26.5225, 26.5225, 26.5225, 4000.0, 106090.0, 4.0),
        ])
    frame = pd.DataFrame(
        rows,
        columns=["stock_code", "date", "open", "high", "low", "close", "volume", "money", "factor"],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    # Place 5m bars on consecutive grid slots (14:55 and 15:00 on SPLIT_START, plus 09:35 on SPLIT_DAY)
    minute_rows = []
    for _, row in frame.iterrows():
        if row["date"] == pd.Timestamp(SPLIT_START):
            minute_rows.append({**row.to_dict(), "date": row["date"] + pd.Timedelta(hours=14, minutes=55)})
            minute_rows.append({**row.to_dict(), "date": row["date"] + pd.Timedelta(hours=15, minutes=0)})
        else:
            minute_rows.append({**row.to_dict(), "date": row["date"] + pd.Timedelta(hours=9, minutes=35)})
    minute = pd.DataFrame(minute_rows)
    with duckdb.connect(str(path)) as conn:
        declare_market_db(conn, "cn_equity")
        seed_calendar(conn, "2024-01-01", "2024-03-01")
        conn.from_df(frame).create("price_data_daily")
        conn.from_df(minute).create("price_data_5m")


def _rank_ic_factors() -> pd.DataFrame:
    dates = pd.to_datetime([SPLIT_START, SPLIT_DAY])
    return pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        index=dates,
        columns=["A", "B", "C"],
    )


class _CloseFactor(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(["close"], table_name="price_data")

    def calculate_factor(self, data):
        return data.pivot(index="date", columns="code", values="close")


def test_one_to_two_split_returns_zero_across_all_three_paths(tmp_path: Path) -> None:
    db = tmp_path / "split.duckdb"
    _create_split_db(db, include_third_day=True)

    # Path 1: PriceFetcher + ForwardReturnsCalculator (no future-anchor lookahead)
    fetcher = PriceFetcher(db, "price_data_daily")
    try:
        p_short = fetcher.fetch(["S"], SPLIT_START, SPLIT_DAY, future_periods=0)
        p_long = fetcher.fetch(["S"], SPLIT_START, SPLIT_EXTENDED, future_periods=0)
        assert p_short.attrs[PRICE_ADJUSTED_ATTR] is True
        np.testing.assert_allclose(p_short.to_numpy(), [100.0, 100.0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            p_long.loc[("S", pd.Timestamp(SPLIT_START))], 100.0, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            p_long.loc[("S", pd.Timestamp(SPLIT_DAY))], 100.0, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            p_long.loc[("S", pd.Timestamp(SPLIT_EXTENDED))], 104.0, rtol=0, atol=1e-12
        )

        fwd_calc = ForwardReturnsCalculator().calculate_forward_returns(
            p_short, periods=(1,), execution_lag=0
        )
        assert fwd_calc.loc[(pd.Timestamp(SPLIT_START), "S"), "1D"] == pytest.approx(0.0, abs=1e-12)

        # Also check 5m alias through PriceFetcher
        fetcher_5m = PriceFetcher(db, "price_data_5m")
        try:
            p_5m = fetcher_5m.fetch(["S"], SPLIT_START, SPLIT_DAY, future_periods=0)
            np.testing.assert_allclose(p_5m.to_numpy(), [100.0, 100.0], rtol=0, atol=1e-12)
        finally:
            fetcher_5m.close()
    finally:
        fetcher.close()

    # Path 2: FactorBase.get_price_data + calculate_forward_returns (pit_enabled True & False, legacy & bar_grid)
    with duckdb.connect(str(db), read_only=True) as conn:
        for pit in (True, False):
            factor = _CloseFactor(
                conn, "split_probe", SPLIT_START, SPLIT_DAY, stock_pool=["S"], pit_enabled=pit,
            )
            prices = factor.get_price_data("close")
            assert prices.attrs[PRICE_ADJUSTED_ATTR] is True
            assert prices.loc[pd.Timestamp(SPLIT_DAY), "S"] / prices.loc[pd.Timestamp(SPLIT_START), "S"] - 1 == pytest.approx(0.0, abs=1e-12)
            returns = factor.calculate_forward_returns(1)
            assert returns.loc[pd.Timestamp(SPLIT_START), "S"] == pytest.approx(0.0, abs=1e-12)

            # Volume adjusts inversely across split (2000 / 2 == 1000 / 1)
            volumes = factor.get_price_data("volume")
            assert volumes.loc[pd.Timestamp(SPLIT_DAY), "S"] == pytest.approx(
                volumes.loc[pd.Timestamp(SPLIT_START), "S"], abs=1e-12
            )

        # Bar-grid A-share route (analysis_mode='cross_sectional')
        bar_factor = _CloseFactor(
            conn, "split_bar_probe", SPLIT_START, SPLIT_DAY, stock_pool=["S"],
            data_frequency="1d", analysis_mode="cross_sectional", pit_enabled=False,
        )
        bar_returns = bar_factor.calculate_forward_returns(1)
        assert bar_returns.iloc[0, 0] == pytest.approx(0.0, abs=1e-12)

        # Path 3: CompositeFactorBase._get_target_returns (1d & 5min)
        for pit in (True, False):
            comp = CompositeFactorBase(
                conn, "comp_split", ["alpha"], SynthesisConfig(prediction_window=1),
                SPLIT_START, SPLIT_DAY, stock_pool=["S"], auto_synthesize=False,
                results_database_path=str(tmp_path / "results.duckdb"), data_bindings={},
            )
            comp.pit_enabled = pit
            target_ret = comp._get_target_returns()
            assert target_ret.loc[pd.Timestamp(SPLIT_START), "S"] == pytest.approx(0.0, abs=1e-12)

        slot_5m = build_jq_ashare_local(conn, start=SPLIT_START, end=SPLIT_DAY)
        comp_5m = CompositeFactorBase(
            conn, "comp_split_5m", ["alpha"], SynthesisConfig(prediction_window=1),
            SPLIT_START, SPLIT_DAY, stock_pool=["S"], auto_synthesize=False,
            results_database_path=str(tmp_path / "results.duckdb"), data_bindings={},
            slot=slot_5m, frequency="5min",
        )
        target_5m = comp_5m._get_target_returns()
        assert not target_5m.empty
        non_null = target_5m["S"].dropna()
        assert len(non_null) == 2
        np.testing.assert_allclose(non_null.to_numpy(), 0.0, rtol=0, atol=1e-12)


def test_three_stock_split_rank_ic_is_plus_one_across_all_three_paths(tmp_path: Path) -> None:
    db = tmp_path / "rank_ic.duckdb"
    _create_split_db(db)
    factors = _rank_ic_factors()
    stocks = ["A", "B", "C"]

    # Path 1: PriceFetcher + ForwardReturnsCalculator + daily_evaluation_slot + AlphalensRunner
    fetcher = PriceFetcher(db, "price_data_daily")
    try:
        pricing = fetcher.fetch(stocks, SPLIT_START, SPLIT_DAY, future_periods=0)
        fwd_calc = (
            ForwardReturnsCalculator()
            .calculate_forward_returns(pricing, periods=(1,), execution_lag=0)["1D"]
            .xs(pd.Timestamp(SPLIT_START), level="date")
            .reindex(stocks)
        )
        assert fwd_calc["A"] == pytest.approx(0.01, abs=1e-12)
        assert fwd_calc["B"] == pytest.approx(0.02, abs=1e-12)
        assert fwd_calc["C"] == pytest.approx(0.03, abs=1e-12)
        assert factors.loc[pd.Timestamp(SPLIT_START), stocks].corr(fwd_calc, method="spearman") == pytest.approx(
            1.0, abs=1e-12
        )

        # Also check daily_evaluation_slot (anti-double-adjustment end-to-end)
        slot_cfg = daily_slot_config(start=SPLIT_START, end=SPLIT_DAY)
        eval_slot = fetcher.daily_evaluation_slot(pricing, slot_cfg)
        from backtest_and_analysis.evaluation import UnifiedReturnLabeler

        labels = UnifiedReturnLabeler.from_slot(eval_slot).label(
            stocks, "1d", start_date=SPLIT_START, end_date=SPLIT_DAY, horizons=(1,), entry_offset_bars=0,
        )
        valid = labels.loc[labels.status.eq("valid")].set_index("instrument_id")["forward_return"]
        assert valid["A"] == pytest.approx(0.01, abs=1e-12)
        assert valid["B"] == pytest.approx(0.02, abs=1e-12)
        assert valid["C"] == pytest.approx(0.03, abs=1e-12)
    finally:
        fetcher.close()

    # AlphalensRunner requires >= 50 factor observations; verify full AlphalensRunner IC == +1.0 across a split
    runner_db = tmp_path / "alphalens_rank_ic.duckdb"
    sessions = pd.bdate_range("2024-01-02", periods=22)
    al_rows = []
    for idx, day in enumerate(sessions):
        # A: +1%/day, B: +2%/day, C: +3%/day with a 1-for-2 split on day 10
        f_c = 2.0 if idx >= 10 else 1.0
        for code, daily_ret, f_val in (("A", 0.01, 1.0), ("B", 0.02, 1.0), ("C", 0.03, f_c)):
            px = 100.0 * ((1.0 + daily_ret) ** idx) / f_val
            al_rows.append((code, day, px, px, px, px, 1000.0 * f_val, px * 1000.0 * f_val, f_val))
    al_frame = pd.DataFrame(
        al_rows,
        columns=["stock_code", "date", "open", "high", "low", "close", "volume", "money", "factor"],
    )
    with duckdb.connect(str(runner_db)) as conn:
        declare_market_db(conn, "cn_equity")
        seed_calendar(conn, "2024-01-01", "2024-03-01")
        conn.from_df(al_frame).create("price_data_daily")
    al_factors = pd.DataFrame(
        np.tile([1.0, 2.0, 3.0], (len(sessions) - 1, 1)),
        index=sessions[:-1],
        columns=stocks,
    )
    al_fetcher = PriceFetcher(runner_db, "price_data_daily")
    try:
        al_pricing = al_fetcher.fetch(
            stocks, sessions[0].strftime("%Y-%m-%d"), sessions[-2].strftime("%Y-%m-%d"), future_periods=1
        )
        runner = AlphalensRunner(n_quantiles=3, periods=(1,), execution_lag=0)
        al_res = runner.run(al_factors, al_pricing)
        np.testing.assert_allclose(al_res["ic"]["1D"].to_numpy(), 1.0, rtol=0, atol=1e-12)
    finally:
        al_fetcher.close()

    # Path 2: FactorBase.calculate_forward_returns (pit_enabled True & False)
    with duckdb.connect(str(db), read_only=True) as conn:
        for pit in (True, False):
            factor = _CloseFactor(
                conn, "ic_probe", SPLIT_START, SPLIT_DAY, stock_pool=stocks, pit_enabled=pit,
            )
            fwd = factor.calculate_forward_returns(1).loc[pd.Timestamp(SPLIT_START), stocks]
            assert fwd["A"] == pytest.approx(0.01, abs=1e-12)
            assert fwd["B"] == pytest.approx(0.02, abs=1e-12)
            assert fwd["C"] == pytest.approx(0.03, abs=1e-12)
            rank_ic = factors.loc[pd.Timestamp(SPLIT_START), stocks].corr(fwd, method="spearman")
            assert rank_ic == pytest.approx(1.0, abs=1e-12)

        # Path 3: CompositeFactorBase._get_target_returns + Rank IC
        for pit in (True, False):
            comp = CompositeFactorBase(
                conn, "comp_ic", ["alpha"], SynthesisConfig(prediction_window=1),
                SPLIT_START, SPLIT_DAY, stock_pool=stocks, auto_synthesize=False,
                results_database_path=str(tmp_path / "results.duckdb"), data_bindings={},
            )
            comp.pit_enabled = pit
            target_ret = comp._get_target_returns().loc[pd.Timestamp(SPLIT_START), stocks]
            assert target_ret["A"] == pytest.approx(0.01, abs=1e-12)
            assert target_ret["B"] == pytest.approx(0.02, abs=1e-12)
            assert target_ret["C"] == pytest.approx(0.03, abs=1e-12)
            comp_ic = factors.loc[pd.Timestamp(SPLIT_START), stocks].corr(target_ret, method="spearman")
            assert comp_ic == pytest.approx(1.0, abs=1e-12)


def test_anti_double_adjustment_contract_across_pit_and_factor_paths(tmp_path: Path) -> None:
    raw = pd.DataFrame({
        "date": pd.to_datetime([SPLIT_START, SPLIT_DAY]),
        "code": ["C", "C"],
        "close": [100.0, 51.5],
        "volume": [1000.0, 2000.0],
        "factor": [1.0, 2.0],
    })

    # 1. Calling apply_pit_adjustment twice or mixing apply_pit_adjustment + adjust_prices_by_factor is idempotent
    pit_once = apply_pit_adjustment(raw)
    assert pit_once.attrs[PRICE_ADJUSTED_ATTR] is True
    pit_twice = apply_pit_adjustment(pit_once)
    assert pit_twice is pit_once
    np.testing.assert_allclose(pit_twice["close"].to_numpy(), [50.0, 51.5], rtol=0, atol=1e-12)

    mixed_after_pit = adjust_prices_by_factor(pit_once)
    assert mixed_after_pit is pit_once

    scaled_once = adjust_prices_by_factor(raw)
    assert scaled_once.attrs[PRICE_ADJUSTED_ATTR] is True
    np.testing.assert_allclose(scaled_once["close"].to_numpy(), [100.0, 103.0], rtol=0, atol=1e-12)
    assert adjust_prices_by_factor(scaled_once) is scaled_once
    assert apply_pit_adjustment(scaled_once) is scaled_once

    # 2. FactorBase.get_price_data / calculate_forward_returns never double-adjust pre-adjusted _fetch_raw_data
    db = tmp_path / "double_guard.duckdb"
    _create_split_db(db)
    with duckdb.connect(str(db), read_only=True) as conn:
        for pit in (True, False):
            factor = _CloseFactor(
                conn, "pre_adjusted", SPLIT_START, SPLIT_DAY, stock_pool=["C"], pit_enabled=pit,
            )
            factor._fetch_raw_data = lambda *args, **kwargs: pit_once.copy()
            prices = factor.get_price_data("close")
            np.testing.assert_allclose(prices["C"].to_numpy(), [50.0, 51.5], rtol=0, atol=1e-12)
            assert factor.calculate_forward_returns(1).loc[pd.Timestamp(SPLIT_START), "C"] == pytest.approx(
                0.03, abs=1e-12
            )

        # 3. Verify explicit adjusted=False returns raw unadjusted close
        factor_raw = _CloseFactor(
            conn, "raw_opt_out", SPLIT_START, SPLIT_DAY, stock_pool=["C"], pit_enabled=False,
        )
        raw_prices = factor_raw.get_price_data("close", adjusted=False)
        np.testing.assert_allclose(raw_prices["C"].to_numpy(), [100.0, 51.5], rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad_factor", [None, 0.0, -1.0, np.inf])
def test_invalid_or_missing_factor_raises_explicitly(tmp_path: Path, bad_factor) -> None:
    db = tmp_path / "bad_factor.duckdb"
    _create_split_db(db)
    with duckdb.connect(str(db)) as conn:
        conn.execute(
            "UPDATE price_data_daily SET factor = ? WHERE stock_code = 'C' AND date = ?",
            [bad_factor, pd.Timestamp(SPLIT_DAY)],
        )

    # Path 1: read_analysis_price_rows / PriceFetcher fail closed on invalid observed factor
    fetcher = PriceFetcher(db, "price_data_daily")
    try:
        with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
            fetcher.fetch(["C"], SPLIT_START, SPLIT_DAY, future_periods=0)
    finally:
        fetcher.close()

    # Path 2: FactorBase.get_price_data fails closed even when pit_enabled=False if factor column exists and is bad
    with duckdb.connect(str(db), read_only=True) as conn:
        for pit in (True, False):
            factor = _CloseFactor(
                conn, "bad_probe", SPLIT_START, SPLIT_DAY, stock_pool=["C"], pit_enabled=pit,
            )
            with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
                factor.get_price_data("close")


def test_read_analysis_price_rows_explicit_require_factor_and_unobserved_rows(tmp_path: Path) -> None:
    db = tmp_path / "no_factor.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE minimal_prices AS SELECT DATE '2024-01-02' AS date, 'A' AS stock_code, "
            "10. AS open, 10. AS high, 10. AS low, 10. AS close, 1. AS volume, 10. AS money"
        )
        # Minimal factor-free table allowed by default, rejected when require_factor=True
        df = read_analysis_price_rows(
            conn, ["A"], "2024-01-02", "2024-01-02",
            table="minimal_prices", date_column="date", instrument_column="stock_code",
        )
        assert df["close"].tolist() == [10.0]
        assert df.attrs[PRICE_ADJUSTED_ATTR] is False
        with pytest.raises(ValueError, match="pit_factor_column_missing"):
            read_analysis_price_rows(
                conn, ["A"], "2024-01-02", "2024-01-02",
                table="minimal_prices", date_column="date", instrument_column="stock_code",
                require_factor=True,
            )

        # Unobserved NaN price row with NaN/0 factor is tolerated, observed row still adjusts
        conn.execute(
            "CREATE TABLE sparse_prices AS SELECT * FROM (VALUES "
            "(DATE '2024-01-02', 'A', 10., 10., 10., 10., 100., 1000., 1.0), "
            "(DATE '2024-01-03', 'A', NULL, NULL, NULL, NULL, NULL, 0., 0.0), "
            "(DATE '2024-01-04', 'A', 5., 5., 5., 5., 200., 1000., 2.0)"
            ") t(date, stock_code, open, high, low, close, volume, money, factor)"
        )
        sparse = read_analysis_price_rows(
            conn, ["A"], "2024-01-02", "2024-01-04",
            table="sparse_prices", date_column="date", instrument_column="stock_code",
        )
        assert sparse.attrs[PRICE_ADJUSTED_ATTR] is True
        assert sparse["close"].iloc[0] == pytest.approx(10.0)
        assert pd.isna(sparse["close"].iloc[1])
        assert pd.isna(sparse["money"].iloc[1])
        assert sparse["close"].iloc[2] == pytest.approx(10.0)


def _create_full_session_5m_split_db(path: Path, *, include_third_day: bool = False) -> None:
    """Create daily + complete 5m A-share bars with a 2-for-1 split on SPLIT_DAY."""
    from utils.session_calendar import SessionCalendar

    _create_split_db(path, include_third_day=include_third_day)
    end_day = SPLIT_EXTENDED if include_third_day else SPLIT_DAY
    calendar = SessionCalendar("XSHG", SPLIT_START, end_day)
    grid = calendar.bar_grid("5min")
    with duckdb.connect(str(path)) as conn:
        daily = conn.execute("SELECT * FROM price_data_daily").fetchdf()
        minute_rows = []
        for _, row in daily.iterrows():
            session_id = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            session_bars = grid.loc[grid["session_id"] == session_id, "timestamp_utc"]
            bar_ends = (session_bars + pd.Timedelta(minutes=5)).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
            for ts in bar_ends:
                minute_rows.append({**row.to_dict(), "date": ts})
        conn.execute("DROP TABLE price_data_5m")
        conn.from_df(pd.DataFrame(minute_rows)).create("price_data_5m")


@pytest.mark.parametrize("pit", [False, True])
def test_p103_session_daily_pricing_split_zero_return_and_idempotency(tmp_path: Path, pit: bool) -> None:
    from backtest_and_analysis.time_series.factor_bridge import session_daily_pricing
    from utils.factor_core.bar_bridge import _read_slot_bars
    from utils.session_calendar import SessionCalendar

    db = tmp_path / "p103_5m_split.duckdb"
    _create_full_session_5m_split_db(db)

    with duckdb.connect(str(db), read_only=True) as conn:
        factor = _CloseFactor(
            conn, "p103_5m", SPLIT_START, SPLIT_DAY, stock_pool=["S"],
            data_frequency="5m", analysis_mode="cross_sectional", output_frequency="1d",
            pit_enabled=pit,
        )
        pricing = factor.get_session_daily_pricing()
        assert pricing.attrs[PRICE_ADJUSTED_ATTR] is True
        series = pricing.unstack("asset")["S"]
        expected_prices = [50.0, 50.0] if pit else [100.0, 100.0]
        np.testing.assert_allclose(series.to_numpy(), expected_prices, rtol=0, atol=1e-12)
        assert series.pct_change().dropna().iloc[0] == pytest.approx(0.0, abs=1e-12)

        raw = factor.get_session_daily_pricing(adjusted=False)
        assert raw.attrs[PRICE_ADJUSTED_ATTR] is False
        np.testing.assert_allclose(raw.unstack("asset")["S"].to_numpy(), [100.0, 50.0], rtol=0, atol=1e-12)

        # Direct session_daily_pricing call on raw execution grid adjusts and preserves idempotency
        _, raw_grid, calendars = _read_slot_bars(factor, execution=True)
        direct = session_daily_pricing(raw_grid, calendars, pit_enabled=pit)
        assert direct.attrs[PRICE_ADJUSTED_ATTR] is True
        direct_series = direct.unstack("asset")["S"]
        np.testing.assert_allclose(direct_series.to_numpy(), expected_prices, rtol=0, atol=1e-12)
        assert direct_series.pct_change().dropna().iloc[0] == pytest.approx(0.0, abs=1e-12)

        # Pre-adjusted grid with attrs['price_adjusted']=True and factor column still present must not double-adjust
        pre_adjusted_grid = raw_grid.copy()
        pre_adjusted_grid.attrs = {**raw_grid.attrs, PRICE_ADJUSTED_ATTR: True}
        direct_pre = session_daily_pricing(pre_adjusted_grid, calendars, pit_enabled=pit)
        assert direct_pre.attrs[PRICE_ADJUSTED_ATTR] is True
        np.testing.assert_allclose(direct_pre.unstack("asset")["S"].to_numpy(), [100.0, 50.0], rtol=0, atol=1e-12)

        # FactorBase.get_session_daily_pricing also respects pre-adjusted bar_feed output
        original_read = factor.slot.bar_feed.read_execution_bars

        def read_pre_adjusted(*args, **kwargs):
            bars = original_read(*args, **kwargs).copy()
            bars.attrs[PRICE_ADJUSTED_ATTR] = True
            return bars

        factor.slot.bar_feed.read_execution_bars = read_pre_adjusted
        try:
            factor_pre = factor.get_session_daily_pricing()
            assert factor_pre.attrs[PRICE_ADJUSTED_ATTR] is True
            np.testing.assert_allclose(factor_pre.unstack("asset")["S"].to_numpy(), [100.0, 50.0], rtol=0, atol=1e-12)
        finally:
            factor.slot.bar_feed.read_execution_bars = original_read


def test_session_daily_pricing_matches_execution_panel_completeness_and_dynamic_pool(tmp_path: Path) -> None:
    db = tmp_path / "p103_parity_masks.duckdb"
    _create_full_session_5m_split_db(db, include_third_day=True)
    stocks = ["A", "C"]
    pools = {SPLIT_START: ["A", "C"], SPLIT_DAY: ["A"], SPLIT_EXTENDED: ["A", "C"]}

    with duckdb.connect(str(db), read_only=True) as conn:
        for pit in (False, True):
            factor = _CloseFactor(
                conn, "p103_pool", SPLIT_START, SPLIT_EXTENDED,
                use_dynamic_pool=True, stock_pools_by_period=pools,
                data_frequency="5m", analysis_mode="cross_sectional", output_frequency="1d",
                pit_enabled=pit,
            )
            pricing_wide = factor.get_session_daily_pricing().unstack("asset")
            exec_wide = factor.get_price_data("close")
            pd.testing.assert_frame_equal(pricing_wide, exec_wide, check_names=False, rtol=0, atol=1e-12)
            assert np.isnan(pricing_wide.loc[pd.Timestamp(SPLIT_DAY), "C"])

            # Incomplete final bar of SPLIT_DAY must be masked to NaN identically across both paths
            original_read = factor.slot.bar_feed.read_execution_bars
            last_slot_split_day = pd.Timestamp(f"{SPLIT_DAY} 07:00Z") - pd.Timedelta(minutes=5)

            def read_with_incomplete(instruments, frequency, *, start_utc, end_utc):
                bars = original_read(instruments, frequency, start_utc=start_utc, end_utc=end_utc).copy()
                bars["is_complete"] = ~(
                    bars["instrument_id"].eq("A") & bars["timestamp_utc"].eq(last_slot_split_day)
                )
                return bars

            factor.slot.bar_feed.read_execution_bars = read_with_incomplete
            try:
                masked_pricing = factor.get_session_daily_pricing().unstack("asset")
                masked_exec = factor.get_price_data("close")
                pd.testing.assert_frame_equal(masked_pricing, masked_exec, check_names=False, rtol=0, atol=1e-12)
                assert np.isnan(masked_pricing.loc[pd.Timestamp(SPLIT_DAY), "A"])
            finally:
                factor.slot.bar_feed.read_execution_bars = original_read


@pytest.mark.parametrize("pit", [False, True])
def test_evaluation_parity_across_daily_execution_panel_and_session_pricing(tmp_path: Path, pit: bool) -> None:
    db = tmp_path / "eval_parity.duckdb"
    _create_full_session_5m_split_db(db, include_third_day=True)
    stocks = ["A", "B", "C"]
    signal_scores = pd.Series([1.0, 2.0, 3.0], index=stocks)

    fetcher = PriceFetcher(db, "price_data_daily")
    try:
        pf_pricing = fetcher.fetch(stocks, SPLIT_START, SPLIT_EXTENDED, future_periods=0)
        pf_returns = (
            ForwardReturnsCalculator()
            .calculate_forward_returns(pf_pricing, periods=(1,), execution_lag=0)["1D"]
            .unstack("asset")[stocks]
        )
    finally:
        fetcher.close()

    with duckdb.connect(str(db), read_only=True) as conn:
        daily_factor = _CloseFactor(
            conn, "parity_daily", SPLIT_START, SPLIT_EXTENDED, stock_pool=stocks,
            data_frequency="daily", pit_enabled=pit,
        )
        daily_returns = daily_factor.calculate_forward_returns(1)[stocks].iloc[:-1]

        bar_5m_factor = _CloseFactor(
            conn, "parity_5m", SPLIT_START, SPLIT_EXTENDED, stock_pool=stocks,
            data_frequency="5m", analysis_mode="cross_sectional", output_frequency="1d",
            pit_enabled=pit,
        )
        exec_panel_prices = bar_5m_factor.get_price_data("close")[stocks]
        exec_panel_returns = bar_5m_factor.calculate_forward_returns(1)[stocks].iloc[:-1]
        session_pricing = bar_5m_factor.get_session_daily_pricing()
        session_prices = session_pricing.unstack("asset")[stocks]
        session_returns = (
            ForwardReturnsCalculator()
            .calculate_forward_returns(session_pricing, periods=(1,), execution_lag=0)["1D"]
            .unstack("asset")[stocks]
        )

        # Adjusted close panels and forward returns match across all evaluation paths
        pd.testing.assert_frame_equal(session_prices, exec_panel_prices, check_names=False, rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(
            session_prices, daily_factor.get_price_data("close")[stocks], check_names=False, rtol=0, atol=1e-12
        )
        pd.testing.assert_frame_equal(session_returns, exec_panel_returns, check_names=False, rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(session_returns, daily_returns, check_names=False, rtol=0, atol=1e-12)
        pd.testing.assert_frame_equal(session_returns, pf_returns, check_names=False, rtol=0, atol=1e-12)

        # Exact expected returns across split (A=+1%, B=+2%, C=+3%) and Rank IC == +1.0
        for day in (pd.Timestamp(SPLIT_START), pd.Timestamp(SPLIT_DAY)):
            row = session_returns.loc[day]
            assert row["A"] == pytest.approx(0.01, abs=1e-12)
            assert row["B"] == pytest.approx(0.02, abs=1e-12)
            assert row["C"] == pytest.approx(0.03, abs=1e-12)
            assert signal_scores.corr(row, method="spearman") == pytest.approx(1.0, abs=1e-12)


def test_session_daily_pricing_market_bars_and_alphalens_split_ic_parity(tmp_path: Path) -> None:
    from data_pipeline.loaders.market_bar_importer import import_market_bars
    from market_db_samples import declare_equity_bars
    from utils.session_calendar import SessionCalendar

    db = tmp_path / "market_bars_split.duckdb"
    stocks = ["600000.XSHG", "600001.XSHG", "600002.XSHG"]
    calendar = SessionCalendar("XSHG", "2024-01-02", "2024-02-05")
    sessions = calendar.sessions()[:19]
    grid = calendar.bar_grid("5min", sessions[0].strftime("%Y-%m-%d"), sessions[-1].strftime("%Y-%m-%d"))

    bar_frames = []
    meta_rows = []
    for code, daily_ret, has_split in zip(stocks, (0.01, 0.02, 0.03), (False, False, True)):
        meta_rows.append(
            dict(
                instrument_id=code, asset_class="equity", exchange="XSHG", timezone="Asia/Shanghai",
                contract_expiry=None, price_adjustment="split_dividend_factor",
                point_value=1.0, calendar_id="XSHG", rth_reference_calendar="XSHG",
            )
        )
        for idx, day in enumerate(sessions):
            session_id = day.strftime("%Y-%m-%d")
            times = grid.loc[grid["session_id"] == session_id, "timestamp_utc"]
            f_val = 2.0 if (has_split and idx >= 9) else 1.0
            px = 100.0 * ((1.0 + daily_ret) ** idx) / f_val
            bar_frames.append(
                pd.DataFrame(
                    dict(
                        timestamp_utc=times, instrument_id=code, frequency="5min",
                        open=px, high=px, low=px, close=px, volume=1000.0 * f_val,
                        source="SYNTHETIC", data_version="v1", factor=f_val,
                    )
                )
            )

    with duckdb.connect(str(db)) as conn:
        declare_equity_bars(conn)
        import_market_bars(conn, pd.concat(bar_frames, ignore_index=True), pd.DataFrame(meta_rows))

    al_factors = pd.DataFrame(
        np.tile([1.0, 2.0, 3.0], (len(sessions) - 1, 1)),
        index=sessions[:-1],
        columns=stocks,
    )
    with duckdb.connect(str(db), read_only=True) as conn:
        for pit in (False, True):
            factor_5m = _CloseFactor(
                conn, "mb_5m", sessions[0].strftime("%Y-%m-%d"), sessions[-1].strftime("%Y-%m-%d"),
                stock_pool=stocks, data_frequency="5m", analysis_mode="cross_sectional",
                output_frequency="1d", data_version="v1", session_calendar=calendar, pit_enabled=pit,
            )
            pricing = factor_5m.get_session_daily_pricing()
            assert pricing.attrs[PRICE_ADJUSTED_ATTR] is True
            result = AlphalensRunner(n_quantiles=3, periods=(1,), execution_lag=0).run(al_factors, pricing)
            np.testing.assert_allclose(result["ic"]["1D"].to_numpy(), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("pit", [False, True])
def test_ashare_all_null_factor_raises_while_price_adjustment_none_skips(tmp_path: Path, pit: bool) -> None:
    from backtest_and_analysis.time_series.factor_bridge import session_daily_pricing
    from data_pipeline.loaders.market_bar_importer import import_market_bars
    from market_db_samples import declare_equity_bars
    from utils.factor_core.bar_bridge import _read_slot_bars
    from utils.session_calendar import SessionCalendar

    # 1. Legacy price_data_5m with all-NULL factor must fail closed on execution_panel & get_session_daily_pricing
    legacy_db = tmp_path / "all_null_legacy_5m.duckdb"
    _create_full_session_5m_split_db(legacy_db)
    with duckdb.connect(str(legacy_db)) as conn:
        conn.execute("UPDATE price_data_5m SET factor = NULL")

    with duckdb.connect(str(legacy_db), read_only=True) as conn:
        factor_legacy = _CloseFactor(
            conn, "null_legacy_5m", SPLIT_START, SPLIT_DAY, stock_pool=["S"],
            data_frequency="5m", analysis_mode="cross_sectional", output_frequency="1d",
            pit_enabled=pit,
        )
        with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
            factor_legacy.get_price_data("close")
        with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
            factor_legacy.calculate_forward_returns(1)
        with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
            factor_legacy.get_session_daily_pricing()
        _, raw_grid, calendars = _read_slot_bars(factor_legacy, execution=True)
        with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
            session_daily_pricing(raw_grid, calendars, pit_enabled=pit)

    # 2. market_bars with price_adjustment='split_dividend_factor' and all-NULL factor must fail closed
    calendar = SessionCalendar("XSHG", SPLIT_START, SPLIT_DAY)
    sessions = calendar.sessions()
    grid = calendar.bar_grid("5min", SPLIT_START, SPLIT_DAY)
    adj_db = tmp_path / "all_null_market_bars_adj.duckdb"
    bars_s0 = pd.concat([
        pd.DataFrame(dict(
            timestamp_utc=grid.loc[grid["session_id"] == day.strftime("%Y-%m-%d"), "timestamp_utc"],
            instrument_id="600000.XSHG", frequency="5min",
            open=100.0 if idx == 0 else 50.0,
            high=100.0 if idx == 0 else 50.0,
            low=100.0 if idx == 0 else 50.0,
            close=100.0 if idx == 0 else 50.0,
            volume=1000.0, source="SYNTHETIC", data_version="v1",
        ))
        for idx, day in enumerate(sessions)
    ], ignore_index=True)
    meta_s0 = pd.DataFrame([dict(
        instrument_id="600000.XSHG", asset_class="equity", exchange="XSHG",
        timezone="Asia/Shanghai", contract_expiry=None, price_adjustment="split_dividend_factor",
        point_value=1.0, calendar_id="XSHG", rth_reference_calendar="XSHG",
    )])
    with duckdb.connect(str(adj_db)) as conn:
        declare_equity_bars(conn)
        import_market_bars(conn, bars_s0, meta_s0)

    with duckdb.connect(str(adj_db), read_only=True) as conn:
        factor_mb_adj = _CloseFactor(
            conn, "null_mb_adj", SPLIT_START, SPLIT_DAY, stock_pool=["600000.XSHG"],
            data_frequency="5m", analysis_mode="cross_sectional", output_frequency="1d",
            data_version="v1", session_calendar=calendar, pit_enabled=pit,
        )
        with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
            factor_mb_adj.get_price_data("close")
        with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
            factor_mb_adj.get_session_daily_pricing()

    # 3. market_bars with price_adjustment='none' and all-NULL factor skips adjustment without error;
    #    mixed grid adjusts only instruments not declaring price_adjustment='none'
    mixed_db = tmp_path / "mixed_market_bars.duckdb"
    bars_split = pd.concat([
        pd.DataFrame(dict(
            timestamp_utc=grid.loc[grid["session_id"] == day.strftime("%Y-%m-%d"), "timestamp_utc"],
            instrument_id="600000.XSHG", frequency="5min",
            open=100.0 if idx == 0 else 50.0,
            high=100.0 if idx == 0 else 50.0,
            low=100.0 if idx == 0 else 50.0,
            close=100.0 if idx == 0 else 50.0,
            volume=1000.0 if idx == 0 else 2000.0,
            source="SYNTHETIC", data_version="v1",
            factor=1.0 if idx == 0 else 2.0,
        ))
        for idx, day in enumerate(sessions)
    ], ignore_index=True)
    bars_none = bars_s0.assign(instrument_id="600001.XSHG", close=20.0, open=20.0, high=20.0, low=20.0)
    meta_mixed = pd.DataFrame([
        dict(
            instrument_id="600000.XSHG", asset_class="equity", exchange="XSHG",
            timezone="Asia/Shanghai", contract_expiry=None, price_adjustment="split_dividend_factor",
            point_value=1.0, calendar_id="XSHG", rth_reference_calendar="XSHG",
        ),
        dict(
            instrument_id="600001.XSHG", asset_class="equity", exchange="XSHG",
            timezone="Asia/Shanghai", contract_expiry=None, price_adjustment="none",
            point_value=1.0, calendar_id="XSHG", rth_reference_calendar="XSHG",
        ),
    ])
    with duckdb.connect(str(mixed_db)) as conn:
        declare_equity_bars(conn)
        import_market_bars(conn, pd.concat([bars_split, bars_none], ignore_index=True), meta_mixed)

    with duckdb.connect(str(mixed_db), read_only=True) as conn:
        # Pure price_adjustment='none' stock passes with NULL factor
        factor_none = _CloseFactor(
            conn, "none_only", SPLIT_START, SPLIT_DAY, stock_pool=["600001.XSHG"],
            data_frequency="5m", analysis_mode="cross_sectional", output_frequency="1d",
            data_version="v1", session_calendar=calendar, pit_enabled=pit,
        )
        np.testing.assert_allclose(
            factor_none.get_session_daily_pricing().to_numpy(), [20.0, 20.0], rtol=0, atol=1e-12,
        )
        # Mixed grid adjusts 600000.XSHG while leaving 600001.XSHG unadjusted
        factor_mixed = _CloseFactor(
            conn, "mixed_adj", SPLIT_START, SPLIT_DAY, stock_pool=["600000.XSHG", "600001.XSHG"],
            data_frequency="5m", analysis_mode="cross_sectional", output_frequency="1d",
            data_version="v1", session_calendar=calendar, pit_enabled=pit,
        )
        mixed_pricing = factor_mixed.get_session_daily_pricing().unstack("asset")
        mixed_exec = factor_mixed.get_price_data("close")
        pd.testing.assert_frame_equal(mixed_pricing, mixed_exec, check_names=False, rtol=0, atol=1e-12)
        expected_split = [50.0, 50.0] if pit else [100.0, 100.0]
        np.testing.assert_allclose(mixed_pricing["600000.XSHG"].to_numpy(), expected_split, rtol=0, atol=1e-12)
        np.testing.assert_allclose(mixed_pricing["600001.XSHG"].to_numpy(), [20.0, 20.0], rtol=0, atol=1e-12)
