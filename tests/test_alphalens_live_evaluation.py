"""Live evaluation preserves raw ties and real-session, pairwise horizons."""
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from utils.trading_calendar import TradingCalendar


@pytest.mark.parametrize('precomputed', [False, True])
@pytest.mark.parametrize('lag', [0, 1])
def test_live_evaluation_keeps_ties_and_pairwise_horizons(precomputed, lag):
    dates = pd.bdate_range('2024-01-02', periods=15).delete(4)
    assets = list('ABCDEFGH')
    prices = pd.DataFrame(
        100 * np.exp(.002 * np.outer(np.arange(len(dates)), np.arange(1, 9))),
        index=dates, columns=assets,
    )
    factor = pd.DataFrame(np.tile([0., 0., 1., 1., 2., 3., 4., 4.], (len(dates), 1)),
                          index=dates, columns=assets)
    pricing = prices.stack()
    pricing.index.names = ['date', 'asset']
    calendar = TradingCalendar(dates, dates.min(), dates.max())
    pricing.attrs['trading_calendar'] = calendar
    runner = AlphalensRunner(periods=(1, 3), n_quantiles=2, execution_lag=lag)
    raw = factor.stack()
    raw.index.names = ['date', 'asset']
    prepared_input = runner._break_factor_ties(raw).unstack()
    forward = (ForwardReturnsCalculator(trading_calendar=calendar)
               .calculate_forward_returns(pricing, (1, 3), lag)) if precomputed else None

    result = runner.run(prepared_input, pricing, forward)

    for period in (1, 3):
        returns = prices.shift(-lag - period) / prices.shift(-lag) - 1
        expected = factor.corrwith(returns, axis=1, method='spearman').dropna()
        pd.testing.assert_series_equal(result['ic'][f'{period}D'].dropna(), expected,
                                       check_names=False, check_exact=True)
        assert len(expected) == len(dates) - lag - period
    assert len(result['ic']['1D'].dropna()) > len(result['ic']['3D'].dropna())
    assert not result['quantile_returns'].empty
