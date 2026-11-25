"""S4 execution conventions and parity with the pre-refactor daily calculator."""

import json
from collections.abc import Callable, Iterable
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from fixtures.eval_engine.baseline.regenerate import BASELINE_DIR, PERIODS, _freeze_frame, synthetic_inputs
from eval_engine_label_test_support import _bars
from utils.session_calendar import SessionCalendar
from utils.trading_calendar import TradingCalendar


DAILY_BASELINE = BASELINE_DIR / "s4_forward_returns.json"


def daily_input(case: str) -> tuple[pd.Series, tuple[int, ...]]:
    if case == "s1_a":
        _, _, pricing, _ = synthetic_inputs()
        return pricing, PERIODS

    dates = pd.bdate_range("2024-01-02", periods=10, name="date")
    calendar = TradingCalendar(dates, dates[0], dates[-1])
    # Market-wide missing session at t+2, plus a suspension for asset B at t+4.
    close = pd.DataFrame(
        {"A": [100., 103., np.nan, 108., 110., 115., 113., 120., 122., 125.],
         "B": [57., 56., np.nan, 54., np.nan, 56., 57., 52., 50., 53.]},
        index=dates,
    ).rename_axis(columns="asset")
    if case == "single_asset":
        close = close[["A"]]
    pricing = close.stack().swaplevel().sort_index(ascending=False)
    pricing.attrs["trading_calendar"] = calendar
    return pricing, (5, 1, 3, 5, 12)


@pytest.mark.parametrize("case", ["s1_a", "single_asset", "multi_asset"])
@pytest.mark.parametrize("lag", [0, 1, 2])
def test_daily_calculator_matches_frozen_legacy_values(case: str, lag: int) -> None:
    pricing, periods = daily_input(case)
    expected = json.loads(DAILY_BASELINE.read_text(encoding="utf-8"))[case][str(lag)]
    result = ForwardReturnsCalculator().calculate_forward_returns(pricing, periods, lag)
    actual = _freeze_frame(result)

    for axis in ("index", "index_names", "columns", "column_names"):
        assert actual[axis] == expected[axis]
    values = np.asarray(actual["values"], dtype=float)
    frozen = np.asarray(expected["values"], dtype=float)
    np.testing.assert_array_equal(np.isnan(values), np.isnan(frozen))
    np.testing.assert_allclose(values, frozen, rtol=0, atol=1e-15)


@pytest.mark.parametrize("make_horizons", [
    pytest.param(lambda: 1, id="int"),
    pytest.param(lambda: np.int64(1), id="numpy-int"),
    pytest.param(lambda: (1,), id="tuple"),
    pytest.param(lambda: iter([1]), id="iterator"),
])
def test_explicit_execution_conventions_use_their_own_prices_times_and_units(
    make_horizons: Callable[[], Iterable[int] | int],
) -> None:
    from backtest_and_analysis.evaluation.labels import ExecutionConvention, label_forward_returns

    days = pd.bdate_range("2026-03-09", periods=3, name="date")
    times = pd.date_range("2026-03-09 13:30Z", periods=3, freq="1min")
    bars = _bars(times).assign(close=[95., 105., 103.95])
    prices = bars.assign(date=days, asset=bars.instrument_id).set_index(["date", "asset"]).close
    daily_convention = ExecutionConvention.CLOSE_TO_CLOSE
    bar_convention = ExecutionConvention.OPEN_TO_CLOSE
    daily = label_forward_returns(
        prices, convention=daily_convention, calendar=TradingCalendar(days, days[0], days[-1]),
        entry_offset=1, horizons=make_horizons(),
    )
    signal = (days[0], "SYNTHETIC")
    assert daily.returns.index.tolist() == [signal]
    assert daily.returns.at[signal, "1D"] == pytest.approx(-0.01, rel=0, abs=1e-15)
    entry_exit = daily.endpoints(1).loc[signal]
    assert (entry_exit.entry_date, entry_exit.exit_date) == (days[1], days[2])
    assert (entry_exit.entry_price, entry_exit.exit_price) == (105., 103.95)
    assert daily_convention.value.unit == "decimal"
    assert daily_convention.value.gap_rule == "endpoints_only"

    intraday = label_forward_returns(
        bars, convention=bar_convention, calendar=SessionCalendar("NQ", days[0], days[0]),
        entry_offset=1, horizons=make_horizons(), view="RTH", signal_timestamps=times[:1],
    )
    bar = intraday.iloc[0]
    assert len(intraday) == 1
    assert bar.status == "valid"
    assert bar.signal_timestamp == times[0]
    assert bar.signal_available_at == bar.entry_timestamp == bar.exit_bar_timestamp == times[1]
    assert bar.exit_timestamp == times[2]
    assert (bar.entry_price, bar.exit_price) == (100., 105.)
    assert bar.return_pct == pytest.approx(5.0, rel=0, abs=1e-14)
    assert bar_convention.value.unit == "percent"
    assert bar_convention.value.gap_rule == "require_complete_bars"


def test_daily_endpoints_retain_the_calendar_grid_across_missing_sessions() -> None:
    from backtest_and_analysis.evaluation.labels import ExecutionConvention, label_forward_returns

    prices, _ = daily_input("multi_asset")
    dates = prices.attrs["trading_calendar"].sessions()
    daily = label_forward_returns(
        prices, convention=ExecutionConvention.CLOSE_TO_CLOSE, entry_offset=1, horizons=(1, 3, 12),
    )
    signal = (dates[0], "A")
    assert pd.isna(daily.returns.at[signal, "1D"])  # Scheduled exit is missing.
    assert signal not in daily.endpoints(1).index
    endpoints = daily.endpoints(3)
    assert (dates[0], "B") not in endpoints.index  # Suspended at the scheduled exit.
    row = endpoints.loc[signal]
    assert (row.entry_date, row.exit_date) == (dates[1], dates[4])
    assert (row.entry_price, row.exit_price) == (103., 110.)
    assert daily.returns.at[signal, "3D"] == pytest.approx(110. / 103. - 1)
    assert daily.endpoints(12).empty


@pytest.mark.parametrize(("lag", "signal", "entry", "exit", "prices_expected"), [
    (0, 0, 0, 3, (100., 108.)),
    (2, 1, 3, 6, (108., 113.)),
])
def test_daily_endpoints_respect_execution_lag(
    lag: int, signal: int, entry: int, exit: int, prices_expected: tuple[float, float],
) -> None:
    from backtest_and_analysis.evaluation.labels import ExecutionConvention, label_forward_returns

    prices, _ = daily_input("single_asset")
    dates = prices.attrs["trading_calendar"].sessions()
    daily = label_forward_returns(
        prices, convention=ExecutionConvention.CLOSE_TO_CLOSE, entry_offset=lag, horizons=(3,),
    )
    row = daily.endpoints(3).loc[(dates[signal], "A")]
    assert (row.entry_date, row.exit_date) == (dates[entry], dates[exit])
    assert (row.entry_price, row.exit_price) == prices_expected


def test_bar_convention_rejects_daily_zero_entry_offset() -> None:
    from backtest_and_analysis.evaluation.labels import ExecutionConvention, label_forward_returns
    from backtest_and_analysis.time_series.contracts import ContractValidationError

    times = pd.date_range("2026-03-09 13:30Z", periods=3, freq="1min")
    with pytest.raises(ContractValidationError, match="entry_offset_bars must be an integer >= 1"):
        label_forward_returns(
            _bars(times), convention=ExecutionConvention.OPEN_TO_CLOSE,
            calendar=SessionCalendar("NQ", "2026-03-09", "2026-03-09"), entry_offset=0, horizons=1,
        )


def test_unsupported_convention_is_not_inferred_from_price_shape() -> None:
    from backtest_and_analysis.evaluation.labels import label_forward_returns

    prices, _ = daily_input("single_asset")
    with pytest.raises(ValueError, match="convention"):
        label_forward_returns(prices, convention="close_to_close", entry_offset=1, horizons=(1,))


def test_legacy_labelers_keep_their_horizon_validation() -> None:
    from backtest_and_analysis.time_series.contracts import ContractValidationError
    from backtest_and_analysis.time_series.labels import BarReturnLabeler

    pricing, _ = daily_input("single_asset")
    with pytest.raises(ValueError, match="periods must be a non-empty iterable"):
        ForwardReturnsCalculator().calculate_forward_returns(pricing, periods=1)
    with pytest.raises(ContractValidationError, match="horizons must be a positive integer or nonempty sequence"):
        BarReturnLabeler(SessionCalendar("NQ", "2026-03-09", "2026-03-09"), horizons=iter([1]))


@pytest.fixture
def long_daily_prices() -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=260, name="date")
    pricing = pd.Series(
        np.arange(100., 360.),
        index=pd.MultiIndex.from_product([dates, ["A"]], names=["date", "asset"]),
    )
    pricing.attrs["trading_calendar"] = TradingCalendar(dates, dates[0], dates[-1])
    return pricing


@pytest.mark.parametrize("lag", [np.uint8(1), np.uint64(1)], ids=["uint8", "uint64"])
def test_unsigned_daily_lag_matches_trade_endpoints(long_daily_prices: pd.Series, lag: np.unsignedinteger) -> None:
    from backtest_and_analysis.evaluation.labels import ExecutionConvention, label_forward_returns

    daily = label_forward_returns(
        long_daily_prices, convention=ExecutionConvention.CLOSE_TO_CLOSE,
        entry_offset=lag, horizons=(1,),
    )
    endpoints = daily.endpoints(1)
    expected_keys = long_daily_prices.index[:-2]
    pd.testing.assert_index_equal(daily.returns.index, expected_keys)
    pd.testing.assert_index_equal(endpoints.index, expected_keys)

    dates = long_daily_prices.index.get_level_values("date")
    np.testing.assert_array_equal(endpoints.entry_date.to_numpy(), dates[1:-1].to_numpy())
    np.testing.assert_array_equal(endpoints.exit_date.to_numpy(), dates[2:].to_numpy())
    np.testing.assert_array_equal(endpoints.entry_price.to_numpy(), long_daily_prices.to_numpy()[1:-1])
    np.testing.assert_array_equal(endpoints.exit_price.to_numpy(), long_daily_prices.to_numpy()[2:])
    recomputed = endpoints.exit_price / endpoints.entry_price - 1
    np.testing.assert_array_equal(daily.returns["1D"].to_numpy(), recomputed.to_numpy())

    legacy = ForwardReturnsCalculator()
    normal = legacy.calculate_forward_returns(long_daily_prices, (1,), 1)
    pd.testing.assert_frame_equal(daily.returns, normal, check_exact=True)
    pd.testing.assert_frame_equal(
        legacy.calculate_forward_returns(long_daily_prices, (1,), lag), normal, check_exact=True,
    )


def test_daily_cache_recomputes_pre_normalization_unsigned_lag_labels(
    long_daily_prices: pd.Series, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def v3_key(
        code_list: list[str], start_date: str, end_date: str,
        periods: tuple[int, ...], source_signature: str = "",
    ) -> str:
        key_data = {
            "schema_version": "forward_returns.v3",
            "code_list": sorted(code_list),
            "start_date": start_date,
            "end_date": end_date,
            "periods": list(periods),
            "source_signature": source_signature,
        }
        return md5(str(key_data).encode()).hexdigest()

    calculator = ForwardReturnsCalculator(cache_dir=str(tmp_path))
    days = long_daily_prices.index.get_level_values("date")
    args = (long_daily_prices, ["A"], str(days[0].date()), str(days[-1].date()), (1,))
    corrupted = pd.DataFrame(
        {"1D": np.arange(357., 360.) / np.arange(100., 103.) - 1},
        index=long_daily_prices.index[255:258],
    )
    with monkeypatch.context() as patch:
        patch.setattr(calculator, "get_cache_key", v3_key)
        patch.setattr(calculator, "calculate_forward_returns", lambda *_: corrupted)
        calculator.calculate_or_load_forward_returns(*args, execution_lag=np.uint8(1))
    stale, = tmp_path.glob("*.pkl")
    old_bytes = stale.read_bytes()

    expected = calculator.calculate_forward_returns(long_daily_prices, (1,), 1)
    pd.testing.assert_frame_equal(
        calculator.calculate_or_load_forward_returns(*args, execution_lag=1), expected, check_exact=True,
    )
    monkeypatch.setattr(calculator, "calculate_forward_returns", lambda *_: pytest.fail("expected a cache hit"))
    pd.testing.assert_frame_equal(
        calculator.calculate_or_load_forward_returns(*args, execution_lag=np.uint8(1)), expected, check_exact=True,
    )
    assert stale.read_bytes() == old_bytes
