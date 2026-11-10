"""Synthetic regressions for evaluation identity, ties, and temporal isolation."""
from utils.trading_calendar import TradingCalendar
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner


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


def test_tied_values_retain_raw_spearman_ic():
    prices, _, pricing = panel()
    factor = pd.DataFrame(np.tile([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.], (len(prices), 1)),
                          index=prices.index, columns=prices.columns)
    runner = AlphalensRunner(periods=(1,))
    result = runner.run(factor, pricing)['ic']['1D']
    returns = prices.shift(-2) / prices.shift(-1) - 1
    expected = factor.corrwith(returns, axis=1, method='spearman').dropna()
    pd.testing.assert_series_equal(result.dropna(), expected, check_names=False)


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
