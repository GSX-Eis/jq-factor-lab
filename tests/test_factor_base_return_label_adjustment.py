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
