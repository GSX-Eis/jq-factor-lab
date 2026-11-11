"""Synthetic coverage for the owner-approved loss denominator and any-horizon rule."""
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner
from utils.trading_calendar import TradingCalendar


def loss_panel():
    dates = pd.bdate_range('2024-01-02', periods=10)
    stocks = list('ABCDEFGHIJ')
    factors = pd.DataFrame(np.tile(np.arange(10.), (10, 1)), index=dates, columns=stocks)
    prices = pd.DataFrame(100. + np.arange(100).reshape(10, 10), index=dates, columns=stocks)
    pricing = prices.stack()
    pricing.index.names = ['date', 'asset']
    pricing.attrs['trading_calendar'] = TradingCalendar(dates, dates.min(), dates.max())
    returns = pd.DataFrame({'1D': np.tile(np.arange(10.) / 100, 10)}, index=pricing.index)
    return factors, pricing, returns


@pytest.mark.parametrize('limit,allowed', [(0., False), (.79, False), (.8, True), (1., True)])
def test_loss_threshold_counts_original_pairs_and_allows_equality(limit, allowed):
    factors, pricing, returns = loss_panel()
    returns = returns.loc[returns.index.get_level_values('asset').isin(['A', 'B'])]
    runner = AlphalensRunner(periods=(1,))
    if not allowed:
        with pytest.raises(ValueError, match=r'max_loss exceeded.*retained=20/100'):
            runner.run(factors, pricing, returns, max_loss=limit)
    else:
        result = runner.run(factors, pricing, returns, max_loss=limit)
        assert len(result['ic']['1D'].dropna()) == 10


@pytest.mark.parametrize('kind', ['alignment', 'constant', 'tail', 'no_overlap', 'too_small', 'low_coverage'])
def test_loss_includes_rows_lost_before_or_during_preparation(kind):
    factors, pricing, returns = loss_panel()
    limit = .19
    if kind == 'alignment':
        pricing = pricing.loc[pricing.index.get_level_values('date').isin(factors.index[2:])]
    elif kind == 'constant':
        factors.iloc[:2] = 1.
    elif kind == 'tail':
        returns.loc[returns.index.get_level_values('date').isin(factors.index[-2:]), '1D'] = np.nan
    elif kind == 'no_overlap':
        returns.index = returns.index.set_levels(returns.index.levels[0] + pd.Timedelta(days=100), level=0)
    elif kind == 'too_small':
        factors = factors.iloc[:4]
    else:
        # Coverage gate also produces zero final retained observations.
        factors = factors.reindex(columns=[*factors.columns, *[f'M{i}' for i in range(300)]])
    with pytest.raises(ValueError, match='max_loss exceeded'):
        AlphalensRunner(periods=(1,)).run(factors, pricing, returns, max_loss=limit)


def test_any_requested_horizon_is_enough():
    factors, pricing, returns = loss_panel()
    returns['3D'] = np.nan
    result = AlphalensRunner(periods=(1, 3)).run(factors, pricing, returns, max_loss=0.)
    assert len(result['ic']['1D'].dropna()) == 10
    assert result['ic']['3D'].isna().all()


@pytest.mark.parametrize('kind', ['unrequested', 'infinite'])
def test_unrequested_or_nonfinite_returns_do_not_count_as_retained(kind):
    factors, pricing, returns = loss_panel()
    if kind == 'unrequested':
        returns = returns.rename(columns={'1D': '3D'})
    else:
        returns['1D'] = np.inf
    with pytest.raises(ValueError, match=r'max_loss exceeded.*retained=0/100'):
        AlphalensRunner(periods=(1,)).run(factors, pricing, returns, max_loss=.99)


def test_missing_input_cells_are_not_part_of_denominator():
    factors, pricing, returns = loss_panel()
    factors.iloc[:, 8:] = np.nan
    returns = returns.loc[returns.index.get_level_values('asset').isin(list('ABCDEFGH'))]
    result = AlphalensRunner(periods=(1,)).run(factors, pricing, returns, max_loss=0.)
    assert len(result['ic']['1D'].dropna()) == 10


@pytest.mark.parametrize('precomputed', [False, True])
def test_calculated_and_precomputed_returns_obey_same_tail_loss(precomputed):
    from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
    factors, pricing, _ = loss_panel()
    returns = (ForwardReturnsCalculator(trading_calendar=pricing.attrs['trading_calendar'])
               .calculate_forward_returns(pricing, (1,), 1)) if precomputed else None
    runner = AlphalensRunner(periods=(1,))
    with pytest.raises(ValueError, match=r'max_loss exceeded.*retained=80/100'):
        runner.run(factors, pricing, returns, max_loss=.19)
    assert len(runner.run(factors, pricing, returns, max_loss=.2)['ic'].dropna()) == 8


@pytest.mark.parametrize('kind', ['empty', 'missing', 'constant'])
def test_zero_denominator_or_full_loss_at_limit_one_keeps_minimal_results(kind):
    factors, pricing, returns = loss_panel()
    if kind == 'empty':
        factors = factors.iloc[:0]
    elif kind == 'missing':
        factors[:] = np.nan
    else:
        factors[:] = 1.
    result = AlphalensRunner(periods=(1,)).run(factors, pricing, returns, max_loss=1. if kind == 'constant' else 0.)
    assert result['ic'].isna().all().all()
    assert result['quantile_returns'].empty


@pytest.mark.parametrize('invalid', [-.01, 1.01, np.nan, np.inf, -np.inf, None, '0.35', True])
def test_invalid_loss_limit_is_rejected(invalid):
    factors, pricing, returns = loss_panel()
    with pytest.raises(ValueError, match='max_loss must be a finite number between 0 and 1'):
        AlphalensRunner(periods=(1,)).run(factors, pricing, returns, max_loss=invalid)
