"""Synthetic regressions for evaluation identity, ties, and temporal isolation."""
from utils.trading_calendar import TradingCalendar
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner, InsufficientCrossSectionAssets


def panel():
    rng = np.random.default_rng(77)
    dates = pd.bdate_range('2024-01-02', periods=80).delete(20)
    prices = pd.DataFrame(
        100 * np.exp(rng.normal(0, .02, (len(dates), 10)).cumsum(axis=0)),
        index=dates, columns=[f'S{i:02d}' for i in range(10)],
    )
    signal = prices.shift(-2) / prices.shift(-1) - 1
    canonical = prices.stack().reorder_levels([1, 0]).sort_index()
    canonical.index.names = ['asset', 'date']
    calendar = TradingCalendar(dates, dates.min(), dates.max())
    prices.attrs['trading_calendar'] = calendar
    canonical.attrs['trading_calendar'] = calendar
    return prices, signal, canonical


def test_constant_factor_has_no_valid_ic_or_quantile_returns():
    prices, _, pricing = panel()
    result = AlphalensRunner(periods=(1,)).run(prices * 0 + 1, pricing, max_loss=1.)
    assert result['ic'].dropna(how='all').empty
    assert result['quantile_returns'].empty


def test_daily_runner_rejects_single_asset_before_quantiling() -> None:
    _, signal, pricing = panel()
    one_asset = pricing.loc[['S00']]
    one_asset.attrs['trading_calendar'] = pricing.attrs['trading_calendar']
    with pytest.raises(InsufficientCrossSectionAssets, match='at least 2 assets.*time-series-analysis'):
        AlphalensRunner(periods=(1,)).run(signal[['S00']], one_asset)


def test_tied_values_retain_raw_spearman_ic():
    prices, _, pricing = panel()
    factor = pd.DataFrame(np.tile([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.], (len(prices), 1)),
                          index=prices.index, columns=prices.columns)
    runner = AlphalensRunner(periods=(1,))
    result = runner.run(factor, pricing)['ic']['1D']
    returns = prices.shift(-2) / prices.shift(-1) - 1
    expected = factor.corrwith(returns, axis=1, method='spearman').dropna()
    pd.testing.assert_series_equal(result.dropna(), expected, check_names=False)


def test_daily_ic_keeps_cross_sections_with_two_nonconstant_pairs() -> None:
    days = pd.bdate_range('2024-01-02', periods=32, name='date')
    factor = pd.DataFrame({'S1': 1., 'S2': 2.}, index=days[:30])
    steps = np.arange(len(days))
    prices = pd.DataFrame({'S1': 100 * 1.01**steps, 'S2': 100 * 1.02**steps}, index=days)
    pricing = prices.rename_axis(columns='asset').stack()
    pricing.attrs['trading_calendar'] = TradingCalendar(days, days.min(), days.max())
    result = AlphalensRunner(n_quantiles=2, periods=(1,)).run(factor, pricing)

    ic = result['ic']['1D']
    assert ic.index.tolist() == factor.index.tolist()
    np.testing.assert_allclose(ic, 1.0, rtol=0, atol=1e-14)


def test_price_order_and_index_layout_do_not_change_results():
    prices, signal, canonical = panel()
    shuffled = prices.stack().sample(frac=1, random_state=123)
    shuffled.index.names = ['date', 'asset']
    shuffled.attrs['trading_calendar'] = canonical.attrs['trading_calendar']
    runner = AlphalensRunner(periods=(1,))
    first = runner.run(signal.copy(), canonical)
    assert first['ic']['1D'].mean() == pytest.approx(1)
    for alternative in (shuffled, shuffled.reorder_levels([1, 0])):
        second = runner.run(signal.copy(), alternative)
        for key in first:
            pd.testing.assert_frame_equal(first[key], second[key])


@pytest.mark.parametrize('lag', [0, 1, 2])
def test_run_keeps_periods_lag_and_real_sessions(lag):
    prices, _, canonical = panel()
    signal = prices.shift(-lag - 1) / prices.shift(-lag) - 1
    result = AlphalensRunner(periods=(1, 3), n_quantiles=3, execution_lag=lag).run(
        signal, canonical
    )
    assert list(result['ic'].columns) == ['1D', '3D']
    for period in (1, 3):
        assert result['ic'][f'{period}D'].dropna().index.max() == prices.index[-lag - period - 1]
    np.testing.assert_allclose(result['ic']['1D'].dropna(), 1.0, rtol=0, atol=1e-14)


def test_future_ties_do_not_change_historical_quantiles():
    dates = pd.bdate_range('2024-01-02', periods=60)
    frame = pd.DataFrame(np.arange(1200, dtype=float).reshape(60, 20), index=dates,
                         columns=[f'S{i:02d}' for i in range(20)])
    frame.loc[dates[52]] = 1500.
    series = frame.stack()
    series.index.names = ['date', 'asset']
    returns = pd.DataFrame({'1D': np.arange(len(series)) / 10000}, index=series.index)
    runner = AlphalensRunner(periods=(1,))
    full = runner._prepare_factor_data_with_precomputed_returns(series, returns)
    prefix = series.loc[series.index.get_level_values('date') <= dates[49]]
    short = runner._prepare_factor_data_with_precomputed_returns(prefix, returns.loc[prefix.index])
    pd.testing.assert_series_equal(full.loc[short.index, 'factor_quantile'], short.factor_quantile)
    assert dates[52] not in full.index.get_level_values('date')
    pd.testing.assert_series_equal(full.factor, series.loc[full.index], check_names=False)


@pytest.mark.parametrize("mixed_source", ("factor", "pricing", "returns", "across_inputs"))
def test_daily_run_rejects_inputs_from_multiple_market_databases(mixed_source):
    prices, factor, pricing = panel()
    forward_returns = (prices.shift(-2) / prices.shift(-1) - 1).stack().to_frame("1D")
    if mixed_source == "across_inputs":
        factor.attrs["db_id"] = "db_a"
        pricing.attrs["db_id"] = "db_b"
    else:
        {"factor": factor, "pricing": pricing, "returns": forward_returns}[mixed_source].attrs["db_ids"] = [
            "db_a", "db_b",
        ]
    with pytest.raises(ValueError, match="multiple market databases"):
        AlphalensRunner(periods=(1,)).run(factor, pricing, forward_returns=forward_returns)


@pytest.mark.parametrize("force_raw", [False, True], ids=["excess", "raw"])
def test_fixed_signal_cross_section_quantiles_independent_of_requested_periods_and_missing_returns(
    force_raw: bool,
) -> None:
    """Signal-time quantiles must use all valid factor values, invariant to period set and future missingness."""
    dates = pd.bdate_range("2024-01-02", periods=16, name="date")
    assets_6 = list("ABCDEF")
    prices_6 = pd.DataFrame(
        {
            asset: 100.0 * (1.0 + 0.002 * (idx + 1)) ** np.arange(len(dates))
            for idx, asset in enumerate(assets_6)
        },
        index=dates,
    )
    # On signal date t0 = dates[0], with execution_lag=1:
    # 1D exits at dates[2], 5D exits at dates[6]. Missing A at dates[2] makes A's 1D NaN while 5D remains valid.
    prices_6.loc[dates[2], "A"] = np.nan

    factor_6 = pd.DataFrame(
        np.tile(np.arange(1.0, 7.0), (len(dates), 1)),
        index=dates,
        columns=assets_6,
    )
    # Give each date distinct 1D returns ordering so quantile means are non-trivial.
    prices_6.loc[dates[2], ["B", "C", "D", "E", "F"]] *= np.array([1.01, 0.99, 1.03, 0.98, 1.04])

    pricing_6 = prices_6.stack().reorder_levels([1, 0]).sort_index()
    pricing_6.index.names = ["asset", "date"]
    pricing_6.attrs["trading_calendar"] = TradingCalendar(dates, dates.min(), dates.max())

    r_1d = AlphalensRunner(n_quantiles=3, periods=(1,), execution_lag=1, _force_raw_quantiles=force_raw)
    r_15d = AlphalensRunner(n_quantiles=3, periods=(1, 5), execution_lag=1, _force_raw_quantiles=force_raw)

    res_1d = r_1d.run(factor_6, pricing_6, max_loss=1.0)
    res_15d = r_15d.run(factor_6, pricing_6, max_loss=1.0)

    t0 = dates[0]
    np.testing.assert_allclose(
        res_1d["ic"].loc[t0, "1D"],
        res_15d["ic"].loc[t0, "1D"],
        rtol=0,
        atol=1e-12,
    )

    if force_raw:
        q_1d = res_1d["quantile_returns"].loc[t0, "1D"].sort_index()
        q_15d = res_15d["quantile_returns"].loc[t0, "1D"].sort_index()
    else:
        q_1d = res_1d["quantile_returns"]["1D"].xs(t0, level="date").sort_index()
        q_15d = res_15d["quantile_returns"]["1D"].xs(t0, level="date").sort_index()
        se_1d = res_1d["std_err"]["1D"].xs(t0, level="date").sort_index()
        se_15d = res_15d["std_err"]["1D"].xs(t0, level="date").sort_index()
        pd.testing.assert_series_equal(se_1d, se_15d, rtol=0, atol=1e-12, check_names=False)

    pd.testing.assert_series_equal(q_1d, q_15d, rtol=0, atol=1e-12, check_names=False)
    # Top - Bottom spread on t0 must match identically across periods=(1,) and (1, 5).
    np.testing.assert_allclose(q_1d.loc[3] - q_1d.loc[1], q_15d.loc[3] - q_15d.loc[1], rtol=0, atol=1e-12)

    # Verify direct quantile assignment on the 6-stock cross-section (1..6 -> [1, 1, 2, 2, 3, 3]).
    f_t0 = factor_6.iloc[[0]].stack()
    f_t0.index.names = ["date", "asset"]
    ret_both = pd.DataFrame(
        {
            "1D": [np.nan, 0.01, -0.01, 0.03, -0.02, 0.04],
            "5D": [0.05, 0.02, 0.03, 0.04, 0.05, 0.06],
        },
        index=f_t0.index,
    )
    ret_1d_only = ret_both[["1D"]].dropna()
    clean_1d = r_1d._prepare_factor_data_with_precomputed_returns(f_t0, ret_1d_only)
    clean_15d = r_15d._prepare_factor_data_with_precomputed_returns(f_t0, ret_both)
    pd.testing.assert_series_equal(
        clean_1d["factor_quantile"],
        clean_15d.loc[clean_1d.index, "factor_quantile"],
        check_exact=True,
    )
    assert clean_1d["factor_quantile"].tolist() == [1, 2, 2, 3, 3]

    # Append stock G (factor=0.0) with both 1D and 5D returns completely missing.
    # A-F quantiles must be determined by the full 7-stock cross-section [0, 1, 2, 3, 4, 5, 6]
    # -> G:1, A:1, B:2, C:2, D:3, E:3, F:3 (strictly shifting B and D vs the 6-stock cross-section).
    factor_7 = factor_6.assign(G=0.0)
    prices_7 = prices_6.assign(G=np.nan)
    prices_7.loc[dates[0], "G"] = 100.0
    pricing_7 = prices_7.stack().reorder_levels([1, 0]).sort_index()
    pricing_7.index.names = ["asset", "date"]
    pricing_7.attrs["trading_calendar"] = pricing_6.attrs["trading_calendar"]

    f_t0_7 = factor_7.iloc[[0]].stack()
    f_t0_7.index.names = ["date", "asset"]
    clean_7_1d = r_1d._prepare_factor_data_with_precomputed_returns(f_t0_7, ret_1d_only)
    clean_7_15d = r_15d._prepare_factor_data_with_precomputed_returns(f_t0_7, ret_both)
    pd.testing.assert_series_equal(
        clean_7_1d["factor_quantile"],
        clean_7_15d.loc[clean_7_1d.index, "factor_quantile"],
        check_exact=True,
    )
    assert clean_7_1d["factor_quantile"].tolist() == [2, 2, 3, 3, 3]

    res_7_1d = r_1d.run(factor_7, pricing_7, max_loss=1.0)
    res_7_15d = r_15d.run(factor_7, pricing_7, max_loss=1.0)
    np.testing.assert_allclose(res_7_1d["ic"].loc[t0, "1D"], res_7_15d["ic"].loc[t0, "1D"], rtol=0, atol=1e-12)
    if force_raw:
        q7_1d = res_7_1d["quantile_returns"].loc[t0, "1D"].dropna().sort_index()
        q7_15d = res_7_15d["quantile_returns"].loc[t0, "1D"].dropna().sort_index()
    else:
        q7_1d = res_7_1d["quantile_returns"]["1D"].xs(t0, level="date").dropna().sort_index()
        q7_15d = res_7_15d["quantile_returns"]["1D"].xs(t0, level="date").dropna().sort_index()
    pd.testing.assert_series_equal(q7_1d, q7_15d, rtol=0, atol=1e-12, check_names=False)
