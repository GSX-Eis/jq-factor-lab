"""Hand-calculated S0-B cases and characterizations of the legacy daily runner."""


import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, rankdata, spearmanr

from backtest_and_analysis.alphalens_runner import AlphalensRunner


def test_spearman_ranks_only_pairwise_valid_observations() -> None:
    factor = np.array([1, 2, 3, 4])
    returns = np.array([2, np.nan, 1, 3])
    valid = np.isfinite(factor) & np.isfinite(returns)

    correct = spearmanr(factor[valid], returns[valid]).statistic
    ranked_before_filtering = pearsonr(rankdata(factor)[valid], rankdata(returns[valid]))[0]

    assert correct == pytest.approx(0.5, abs=1e-10, rel=0)
    assert ranked_before_filtering == pytest.approx(0.3273268354, abs=1e-10, rel=0)


def test_missing_five_day_labels_do_not_change_one_day_statistics() -> None:
    days = pd.bdate_range('2024-01-02', periods=4, name='date')
    index = pd.MultiIndex.from_product([days, ['A', 'B', 'C', 'D']], names=['date', 'asset'])
    factor = pd.Series(
        [
            1, 1, 3, 4,  # Tied factor values
            2, 2, 2, 2,  # Constant factor
            np.nan, np.nan, np.nan, np.nan,  # Empty cross-section
            1, 2, 3, 4,
        ],
        index=index,
        name='factor',
    )
    labels = pd.DataFrame(
        {
            '1D': [
                .02, .01, .03, .04,
                .01, .02, .03, .04,
                np.nan, np.nan, np.nan, np.nan,
                .01, np.nan, .03, .04,
            ],
            '5D': [.05, .10, .15, .20] * 2 + [np.nan] * 4 + [.05, .10, .15, .20],
        },
        index=index,
    )
    missing_long_labels = labels.copy()
    missing_long_labels.loc[days[0], '5D'] = np.nan
    missing_long_labels.loc[(days[3], 'B'), '5D'] = np.nan

    # Frozen counts include the constant factor row's valid pairs. Legacy
    # cleaning removes that row before calculating IC and quantile returns.
    expected_counts = [4, 4, 0, 3]
    expected_ic = [0.9486832980505138, np.nan, np.nan, 1.0]
    expected_status = [None, 'constant_factor', 'insufficient_cross_section', None]
    expected_raw = [[.015, .035], [np.nan, np.nan], [np.nan, np.nan], [.01, .035]]
    expected_excess = [[-.01, .01], [np.nan, np.nan], [np.nan, np.nan],
                       [-.0166666666666667, .0083333333333333]]

    runner = AlphalensRunner(n_quantiles=2, periods=(1, 5))
    for forward_returns in (labels, missing_long_labels):
        counts, correlations, statuses, raw_returns = [], [], [], []
        for day in days:
            signal = factor.loc[day]
            returns = forward_returns.loc[day, '1D']
            valid = np.isfinite(signal) & np.isfinite(returns)
            count = int(valid.sum())
            counts.append(count)

            if count < 2 or signal[valid].nunique() == 1:
                statuses.append('insufficient_cross_section' if count < 2 else 'constant_factor')
                correlations.append(np.nan)
                raw_returns.append([np.nan, np.nan])
                continue

            statuses.append(None)
            correlations.append(spearmanr(signal[valid], returns[valid]).statistic)
            buckets = pd.qcut(signal, 2, labels=False, duplicates='drop') + 1
            raw_returns.append(returns[valid].groupby(buckets[valid]).mean().reindex([1, 2]))

        assert counts == expected_counts
        assert statuses == expected_status
        np.testing.assert_allclose(correlations, expected_ic, rtol=0, atol=1e-12)
        np.testing.assert_allclose(raw_returns, expected_raw, rtol=0, atol=1e-12)

        clean = runner._prepare_factor_data_with_precomputed_returns(factor, forward_returns)
        result = runner._process_alphalens_results(clean)
        retained = clean['1D'].notna().groupby(level='date').sum().reindex(days, fill_value=0)
        assert retained.tolist() == [4, 0, 0, 3]
        assert result['ic'].index.tolist() == [days[0], days[3]]
        np.testing.assert_allclose(result['ic']['1D'].reindex(days), expected_ic, rtol=0, atol=1e-12)
        daily_buckets = result['quantile_returns']['1D'].unstack('factor_quantile')
        np.testing.assert_allclose(daily_buckets.reindex(days), expected_excess, rtol=0, atol=1e-12)
        if forward_returns is labels:
            assert result['ic'].at[days[0], '5D'] == pytest.approx(0.9486832980505138)
        else:
            assert pd.isna(result['ic'].at[days[0], '5D'])


@pytest.mark.parametrize('force_fallback', [False, True], ids=['cross_section_excess', 'fallback_raw'])
def test_legacy_quantile_return_response_to_common_shift(
    force_fallback: bool
) -> None:
    day = pd.Timestamp('2024-01-02')
    index = pd.MultiIndex.from_product([[day], ['A', 'B', 'C', 'D']], names=['date', 'asset'])
    data = pd.DataFrame(
        {'factor': [1, 2, 3, 4], 'factor_quantile': [1, 1, 2, 2],
         '1D': [.01, .03, .05, .07]},
        index=index,
    )
    runner = AlphalensRunner(n_quantiles=2, periods=(1,), _force_raw_quantiles=force_fallback)
    original = runner._process_alphalens_results(data)
    shift = 0.125
    shifted = runner._process_alphalens_results(data.assign(**{'1D': data['1D'] + shift}))

    if force_fallback:
        assert original['std_err'].empty and shifted['std_err'].empty
        by_bucket = original['quantile_returns'].loc[day, '1D']
        shifted_by_bucket = shifted['quantile_returns'].loc[day, '1D']
        assert by_bucket.index.tolist() == shifted_by_bucket.index.tolist() == [1, 2]
        assert by_bucket.tolist() == pytest.approx([.02, .06])
        assert shifted_by_bucket.tolist() == pytest.approx([.145, .185])
        assert (shifted_by_bucket - by_bucket).tolist() == pytest.approx([shift, shift])
    else:
        assert not original['std_err'].empty and not shifted['std_err'].empty
        by_bucket = original['quantile_returns']['1D'].xs(day, level='date')
        shifted_by_bucket = shifted['quantile_returns']['1D'].xs(day, level='date')
        assert by_bucket.index.tolist() == shifted_by_bucket.index.tolist() == [1, 2]
        assert by_bucket.tolist() == pytest.approx([-.02, .02])
        pd.testing.assert_series_equal(shifted_by_bucket, by_bucket, rtol=0, atol=1e-12)


def test_quantile_time_weights_and_paired_spread_standard_error() -> None:
    observations = pd.DataFrame(
        {
            'time': [1] * 3 + [2] * 11 + [3],
            'bucket': [1, 1, 2] + [1] * 10 + [2, 2],
            'return': [0, 0, .02] + [.06] * 10 + [.12, .50],
        }
    )
    daily = observations.groupby(['time', 'bucket'])['return'].mean().unstack('bucket')
    low_bucket = observations.loc[observations['bucket'] == 1]
    assert low_bucket.groupby('time').size().tolist() == [2, 10]
    assert daily[1].dropna().tolist() == pytest.approx([0.00, 0.06])
    assert daily[1].mean() == pytest.approx(0.03)
    assert low_bucket['return'].mean() == pytest.approx(0.05)

    paired = daily[[1, 2]].dropna()
    spread = paired[2] - paired[1]
    assert spread.index.tolist() == [1, 2]
    assert spread.tolist() == pytest.approx([.02, .06])
    assert spread.mean() == pytest.approx(.04)
    assert spread.sem(ddof=1) == pytest.approx(.02)
    assert daily[2].mean() - daily[1].mean() == pytest.approx(.1833333333333333)


# S0 intentional difference in §8. None is the JSON representation of missing ICIR.
SINGLE_VALID_IC_EXPECTED = (None, 'insufficient_valid_cross_sections')


def test_single_valid_daily_ic_legacy_ir_and_frozen_icir() -> None:
    days = pd.bdate_range('2024-01-02', periods=2, name='date')
    index = pd.MultiIndex.from_product([days, ['A', 'B']], names=['date', 'asset'])
    factor = pd.Series([1, 2, 1, 2], index=index, name='factor')
    labels = pd.DataFrame({'1D': [.01, .02, .01, np.nan]}, index=index)
    runner = AlphalensRunner(n_quantiles=2, periods=(1,))

    clean = runner._prepare_factor_data_with_precomputed_returns(factor, labels)
    result = runner._process_alphalens_results(clean)
    np.testing.assert_allclose(result['ic']['1D'].reindex(days), [1, np.nan], rtol=0, atol=1e-12)
    valid_ic = result['ic']['1D'].dropna()
    assert valid_ic.index.tolist() == [days[0]]
    assert result['ir'].at['1D', 'ir'] == pytest.approx(1.0)

    # Independent check of the frozen ICIR expectation using sample standard deviation.
    ic_std = valid_ic.std(ddof=1)
    assert pd.isna(ic_std)
    reason = 'insufficient_valid_cross_sections' if len(valid_ic) < 2 else None
    assert (None if pd.isna(ic_std) else valid_ic.mean() / ic_std, reason) == SINGLE_VALID_IC_EXPECTED


@pytest.mark.parametrize(
    'factor, returns, direction, hits, count, ups, accuracy, baseline, lift',
    [
        ([1, 1, 1, 1, 1], [.01, .02, .03, .04, -.01], 1, 4, 5, 4, .80, .80, 0),
        ([1, -1, 0, -1, 1, np.nan, 1], [-.02, .03, .02, 0, -.01, .03, np.nan],
         -1, 3, 4, 1, .75, .25, .50),
    ],
    ids=['always_long', 'zero_signal_zero_return_and_missing_pairs'],
)
def test_directional_accuracy_uses_same_denominator_for_hits_and_up_baseline(
    factor: list[float], returns: list[float], direction: int, hits: int, count: int,
    ups: int, accuracy: float, baseline: float, lift: float
) -> None:
    eligible = [(direction * x, r) for x, r in zip(factor, returns)
                if np.isfinite(x) and np.isfinite(r) and x != 0]
    actual_count = len(eligible)
    actual_hits = sum(signal * r > 0 for signal, r in eligible)
    actual_ups = sum(r > 0 for _, r in eligible)
    actual_accuracy = actual_hits / actual_count
    actual_baseline = actual_ups / actual_count

    assert (actual_hits, actual_count, actual_ups) == (hits, count, ups)
    assert actual_accuracy == pytest.approx(accuracy)
    assert actual_baseline == pytest.approx(baseline)
    assert actual_accuracy - actual_baseline == pytest.approx(lift)


def test_bartlett_lag_four_understates_five_step_long_run_variance() -> None:
    # Independent unit-variance innovations contribute once per shared term.
    current = set(range(5))
    autocovariances = [len(current.intersection(range(lag, lag + 5))) for lag in range(5)]
    assert autocovariances == [5, 4, 3, 2, 1]

    long_run = autocovariances[0] + 2 * sum(autocovariances[1:])
    # At L=4, weights are (5-lag)/5. Multiply before dividing for exact inputs.
    bartlett = autocovariances[0] + 2 * sum(
        (5 - lag) * covariance for lag, covariance in enumerate(autocovariances[1:], start=1)
    ) / 5
    assert long_run == 25
    assert bartlett == 17


def test_hac_preserves_missing_slots_in_lag_one_products() -> None:
    ic = pd.Series([.5, np.nan, -.5, .5, -.5])
    observed = ic.notna().to_numpy()
    assert ic.mean() == 0
    assert ic.count() == 4
    assert np.count_nonzero(observed[1:] & observed[:-1]) == 2

    centered = ic - ic.mean()
    grid = centered.fillna(0).to_numpy()
    compacted = centered.dropna().to_numpy()
    assert compacted.size - 1 == 3
    assert np.dot(grid[1:], grid[:-1]) == -.50
    assert np.dot(compacted[1:], compacted[:-1]) == -.75

    # At L=1, twice the Bartlett weight is 1. Both denominators are n^2.
    grid_variance = (np.dot(grid, grid) + np.dot(grid[1:], grid[:-1])) / ic.count()**2
    compacted_variance = (
        np.dot(compacted, compacted) + np.dot(compacted[1:], compacted[:-1])
    ) / ic.count()**2
    assert grid_variance == .03125
    assert compacted_variance == .015625
