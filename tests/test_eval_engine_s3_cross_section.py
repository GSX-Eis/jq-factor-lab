"""Frozen counterexamples and Alphalens parity for the cross-sectional layer."""

from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.stats import pearsonr

from backtest_and_analysis.evaluation.cross_section import bartlett_hac, evaluate_cross_section
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.types import ReturnMode
from reference.alphalens_ref import performance as ref
from reference.alphalens_ref.utils import quantize_factor

TOL = {"abs": 1e-12, "rel": 0}
ARRAY_TOL = {"atol": 1e-12, "rtol": 0}


def _panel(factor: npt.ArrayLike, returns: npt.ArrayLike, **horizons: npt.ArrayLike) -> EvaluationPanel:
    factor = np.asarray(factor)
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-02", periods=factor.shape[0]), range(factor.shape[1])],
        names=["date", "asset"],
    )
    return EvaluationPanel.from_long(pd.DataFrame(
        {"factor": factor.ravel(), "1D": np.ravel(returns),
         **{key: np.ravel(value) for key, value in horizons.items()}},
        index=index,
    ))


def test_pairwise_ranks_and_horizon_independence() -> None:
    factor = [[1, 1, 3, 4], [2, 2, 2, 2], [np.nan] * 4, [1, 2, 3, 4], [1, 2, 3, 4]]
    returns = [[.02, .01, .03, .04], [.01, .02, .03, .04],
               [np.nan] * 4, [.01, np.nan, .03, .04], [2, np.nan, 1, 3]]
    long = [[.05, .10, .15, .20]] * 5
    short_ic = [.9486832980505138, np.nan, np.nan, 1, .5]
    short_returns = [[.015, .035], [np.nan, np.nan], [np.nan, np.nan], [.01, .035], [2, 2]]
    complete = evaluate_cross_section(_panel(factor, returns, **{"5D": long}), hac_lag=1, n_quantiles=2)
    missing = evaluate_cross_section(
        _panel(factor, returns, **{"5D": [[np.nan] * 4] + long[1:]}), hac_lag=1, n_quantiles=2,
    )
    np.testing.assert_array_equal(complete.quantiles.labels, missing.quantiles.labels)
    assert complete.quantile_reasons.tolist() == [
        None, "unable_to_form_quantiles", "unable_to_form_quantiles", None, None,
    ]
    for result in (complete, missing):
        short = result.horizons["1D"]
        np.testing.assert_allclose(short.ic.values, short_ic, **ARRAY_TOL)
        assert short.ic.counts.tolist() == [4, 4, 0, 3, 3]
        assert short.ic.reasons.tolist() == [None, "constant_factor", "insufficient_cross_section", None, None]
        np.testing.assert_allclose(short.bucket_returns, short_returns, **ARRAY_TOL)
        assert short.bucket_counts.tolist() == [[2, 2], [0, 0], [0, 0], [1, 2], [1, 2]]
        valid_ic = pd.Series(short_ic).dropna()
        assert short.ic_summary.mean == pytest.approx(valid_ic.mean(), **TOL)
        assert short.ic_summary.median == pytest.approx(valid_ic.median(), **TOL)
        assert short.ic_summary.sample_std == pytest.approx(valid_ic.std(), **TOL)
        assert short.ic_summary.icir == pytest.approx(valid_ic.mean() / valid_ic.std(), **TOL)
    assert np.isfinite(complete.horizons["5D"].ic.values[0])
    assert np.isnan(missing.horizons["5D"].ic.values[0])
    assert missing.horizons["5D"].bucket_counts[0].tolist() == [0, 0]


def test_ic_thresholds_single_valid_and_zero_variance() -> None:
    panel = _panel([[1, 2], [1, 2]], [[.01, .02], [.01, np.nan]])
    daily = evaluate_cross_section(panel, hac_lag=3).horizons["1D"]
    assert daily.ic_summary.n_valid == 1
    assert daily.ic_summary.mean == daily.ic_summary.median == pytest.approx(1, **TOL)
    assert daily.ic_summary.positive_fraction == 1
    assert np.isnan(daily.ic_summary.sample_std) and np.isnan(daily.ic_summary.icir)
    assert daily.ic_summary.reason == daily.hac.reason == "insufficient_valid_cross_sections"
    assert daily.ic.reasons.tolist() == [None, "insufficient_cross_section"]

    bar = evaluate_cross_section(panel, hac_lag=0, frequency="bar").horizons["1D"]
    assert bar.ic_summary.n_valid == 0
    assert np.isnan(bar.ic_summary.positive_fraction)
    assert bar.ic_summary.reason == "no_valid_cross_sections"
    assert bar.ic.reasons.tolist() == ["insufficient_cross_section"] * 2
    override = evaluate_cross_section(panel, hac_lag=0, frequency="bar", min_samples=2)
    np.testing.assert_array_equal(override.horizons["1D"].ic.values, daily.ic.values)

    # Perfect correlations from differently sized rows must not create a spurious ICIR.
    flat = _panel([[1, 2, 3, 4, 5, 6]] * 2, [[1, 2, np.nan, np.nan, np.nan, np.nan], [1, 2, 3, 4, 5, 6]])
    repeated = evaluate_cross_section(flat, hac_lag=0).horizons["1D"]
    assert repeated.ic_summary.n_valid == 2
    assert repeated.ic_summary.sample_std == 0
    assert np.isnan(repeated.ic_summary.icir)
    assert repeated.ic_summary.reason == "zero_ic_std"
    assert repeated.hac.reason == "degenerate_hac_variance"


@pytest.mark.parametrize("sign", [-1, 1])
def test_two_pair_pearson_returns_exact_signed_units(sign: int) -> None:
    panel = _panel([[.1, .2], [.1, .2]], sign * np.array([[.2, .5], [.1, .2]]))
    result = evaluate_cross_section(panel, hac_lag=0, ic_method="pearson").horizons["1D"]
    np.testing.assert_array_equal(result.ic.values, [sign, sign])
    assert result.ic_summary.reason == "zero_ic_std"
    assert result.hac.reason == "degenerate_hac_variance"


@pytest.mark.parametrize("sign", [-1, 1])
def test_pearson_linear_rescalings_produce_zero_ic_variance(sign: int) -> None:
    factor = np.array([0., 1., 4.])
    panel = _panel([factor, factor], sign * factor * np.array([[1], [3]]) / 128)
    result = evaluate_cross_section(panel, hac_lag=0, ic_method="pearson").horizons["1D"]
    np.testing.assert_array_equal(result.ic.values, [sign, sign])
    assert result.ic_summary.sample_std == 0
    assert np.isnan(result.ic_summary.icir)
    assert result.ic_summary.reason == "zero_ic_std"
    assert result.hac.reason == "degenerate_hac_variance"


@pytest.mark.parametrize("sign", [-1, 1])
def test_pearson_retains_a_representable_gap_from_unit_correlation(sign: int) -> None:
    factor = np.array([-1., 0., 1.])
    perturbed = factor + 2.0**-26 * np.array([1, -2, 1])
    panel = _panel([factor, factor], sign * np.array([factor, perturbed]))
    result = evaluate_cross_section(panel, hac_lag=0, ic_method="pearson").horizons["1D"]
    # Orthogonal perturbation gives |r| = (1 + 3 * 2**-52)**(-1/2).
    np.testing.assert_array_equal(result.ic.values, [sign, sign * (1 - 3 * 2.0**-53)])
    assert result.ic_summary.sample_std > 0


@pytest.mark.parametrize("reverse", [False, True], ids=["original", "reversed"])
def test_pearson_zero_and_tiny_positive_ic_survive_asset_reordering(reverse: bool) -> None:
    factor = np.array([0, 0, 0, 0, 1])
    returns = np.array([0., 4., 4., 4., 3.]) / 128
    tiny_positive = returns.copy()
    tiny_positive[-1] = np.nextafter(returns[-1], np.inf)
    asset_order = slice(None, None, -1 if reverse else 1)
    result = evaluate_cross_section(
        _panel([factor[asset_order]], [returns[asset_order]]), hac_lag=0, ic_method="pearson",
    ).horizons["1D"]
    assert result.ic.values[0] == 0
    assert result.ic_summary.positive_fraction == 0
    result = evaluate_cross_section(
        _panel([factor[asset_order]], [tiny_positive[asset_order]]), hac_lag=0, ic_method="pearson",
    ).horizons["1D"]
    # The exact covariance numerator is 4 * delta, and the norm product is
    # very close to the unperturbed value sqrt(4 * 60) / 128.
    delta = tiny_positive[-1] - returns[-1]
    expected = 4 * delta * 128 / np.sqrt(4 * 60)
    assert result.ic.values[0] == pytest.approx(expected, rel=1e-14, abs=0)
    assert result.ic_summary.positive_fraction == 1


def test_pearson_zero_with_cancelling_binary_float_products() -> None:
    rng = np.random.default_rng(821)
    factor_groups = rng.normal(size=(8, 8))
    shocks = rng.normal(size=(8, 8))
    # Each factor value sees opposite returns. Both the return sum and
    # the sum of cross products are exactly zero for every asset order.
    factor = np.column_stack((factor_groups, factor_groups))
    returns = np.column_stack((shocks, -shocks))
    shuffled = np.argsort(rng.random(factor.shape), axis=1)
    panel = _panel(
        np.take_along_axis(factor, shuffled, axis=1),
        np.take_along_axis(returns, shuffled, axis=1),
    )
    result = evaluate_cross_section(panel, hac_lag=0, ic_method="pearson").horizons["1D"]
    np.testing.assert_array_equal(result.ic.values, np.zeros(len(factor)))
    assert result.ic_summary.positive_fraction == 0


def test_time_weighted_buckets_and_simultaneously_paired_spread() -> None:
    # Both buckets are assigned on every row. Exit-label availability varies afterward.
    panel = _panel(
        [list(range(1, 21))] * 3,
        [[0, 0] + [np.nan] * 8 + [.02] + [np.nan] * 9,
         [.06] * 10 + [.12] + [np.nan] * 9,
         [np.nan] * 10 + [.50] + [np.nan] * 9],
    )
    result = evaluate_cross_section(panel, hac_lag=0, n_quantiles=2).horizons["1D"]
    np.testing.assert_allclose(result.bucket_returns, [[0, .02], [.06, .12], [np.nan, .50]], **ARRAY_TOL)
    assert result.bucket_counts.tolist() == [[2, 1], [10, 1], [0, 1]]
    low = result.bucket_summary[1]
    assert (low.mean, low.standard_error, low.n_times) == pytest.approx((.03, .03, 2), **TOL)
    assert np.average(result.bucket_returns[:2, 0], weights=result.bucket_counts[:2, 0]) == pytest.approx(.05)
    np.testing.assert_allclose(result.spread, [.02, .06, np.nan], **ARRAY_TOL)
    spread = result.spread_summary
    assert (spread.mean, spread.standard_error, spread.n_times) == pytest.approx((.04, .02, 2), **TOL)


@pytest.mark.parametrize("return_mode", ["raw", "cross_section_excess"])
def test_bucket_mode_and_common_shift(return_mode: ReturnMode) -> None:
    factor = [[1, 2, 3, 4]]
    returns = np.array([[.01, .03, .05, .07]])
    kwargs = {"hac_lag": 0, "n_quantiles": 2, "return_mode": return_mode}
    original = evaluate_cross_section(_panel(factor, returns), **kwargs)
    shifted = evaluate_cross_section(_panel(factor, returns + .125), **kwargs)
    assert original.return_mode == shifted.return_mode == return_mode
    expected = np.array([[.02, .06] if return_mode == "raw" else [-.02, .02]])
    np.testing.assert_allclose(original.horizons["1D"].bucket_returns, expected, **ARRAY_TOL)
    delta = .125 if return_mode == "raw" else 0
    np.testing.assert_allclose(shifted.horizons["1D"].bucket_returns, expected + delta, **ARRAY_TOL)


def test_bucket_and_spread_missing_states_and_fixed_extreme_buckets() -> None:
    panel = _panel([[1, 1, 3, 4], [1, 1, 3, 4]], [[.01, .02, np.nan, np.nan], [np.nan, np.nan, .04, .05]])
    result = evaluate_cross_section(panel, hac_lag=0, n_quantiles=4)
    horizon = result.horizons["1D"]
    assert result.quantiles.n_bins.tolist() == [3, 3]
    assert horizon.bucket_summary[4].reason == "empty_bucket"
    assert horizon.bucket_summary[4].n_times == 0
    assert np.isnan(horizon.bucket_summary[4].mean)
    assert horizon.bucket_summary[1].reason == "insufficient_time_points"
    assert horizon.bucket_summary[1].n_times == 1
    assert np.isnan(horizon.bucket_summary[1].standard_error)
    assert horizon.spread_summary.reason == "no_paired_bucket_returns"
    assert horizon.spread_summary.n_times == 0

    # Selecting existing buckets still requires both to be present at the same time.
    custom = evaluate_cross_section(panel, hac_lag=0, n_quantiles=4, high_quantile=3)
    assert custom.horizons["1D"].spread_summary.reason == "no_paired_bucket_returns"
    paired = evaluate_cross_section(
        _panel([[1, 2, 3, 4]], [[.01, .03, .05, .07]]), hac_lag=0, n_quantiles=2,
    ).horizons["1D"].spread_summary
    assert paired.n_times == 1 and paired.mean == pytest.approx(.04, **TOL)
    assert paired.reason == "insufficient_time_points" and np.isnan(paired.standard_error)

    one_bin = evaluate_cross_section(_panel([[1, 1, 2]], [[.01, .02, .03]]), hac_lag=0, n_quantiles=2)
    assert one_bin.quantiles.n_bins.tolist() == [1]
    assert one_bin.quantile_reasons.tolist() == ["unable_to_form_quantiles"]
    assert one_bin.horizons["1D"].bucket_counts.tolist() == [[0, 0]]


def test_bartlett_hac_preserves_missing_grid_slots() -> None:
    ic = np.array([.5, np.nan, -.5, .5, -.5])
    grid = bartlett_hac(ic, lag=1)
    compacted = bartlett_hac(ic[np.isfinite(ic)], lag=1)
    assert grid.n_valid == compacted.n_valid == 4
    assert grid.grid_size == 5 and grid.pair_counts.tolist() == [2]
    assert compacted.grid_size == 4 and compacted.pair_counts.tolist() == [3]
    assert grid.variance == .03125
    assert compacted.variance == .015625
    half_width = NormalDist().inv_cdf(.975) * np.sqrt(.03125)
    assert (grid.lower, grid.upper) == pytest.approx((-half_width, half_width), **TOL)

    # An entirely absent observation time is still part of the explicit panel grid.
    factor = [[1, 2, 3]] * 5
    returns = [[2, 1, 3], [np.nan] * 3, [2, 3, 1], [2, 1, 3], [2, 3, 1]]
    unfiltered = _panel(factor, returns)
    observed = unfiltered.to_long().drop(index=unfiltered.datetimes[1], level="date")
    panel = EvaluationPanel.from_long(observed, datetimes=unfiltered.datetimes)
    result = evaluate_cross_section(panel, hac_lag=1).horizons["1D"]
    np.testing.assert_allclose(result.ic.values, ic, **ARRAY_TOL)
    assert result.hac.pair_counts.tolist() == [2]
    assert result.hac.variance == pytest.approx(grid.variance, **TOL)


def test_five_step_overlap_bartlett_attenuation() -> None:
    # Opposite impulses nine steps apart yield two isolated signed five-step runs.
    # The centered lag products sum to twice the population autocovariances.
    ic = np.convolve([1] + [0] * 8 + [-1], np.ones(5))
    products = [np.dot(ic, ic)] + [np.dot(ic[k:], ic[:-k]) for k in range(1, 5)]
    assert products == [10, 8, 6, 4, 2]
    assert (products[0] + 2 * sum(products[1:])) / 2 == 25
    interval = bartlett_hac(ic, lag=4)
    assert interval.pair_counts.tolist() == [13, 12, 11, 10]
    assert interval.variance * ic.size**2 / 2 == pytest.approx(17, **TOL)


@pytest.mark.parametrize(
    "ic, lag, reason, counts",
    [
        ([np.nan, .3], 2, "insufficient_valid_cross_sections", [0, 0]),
        ([1, -1], 3, "insufficient_hac_grid", [1, 0, 0]),
        ([.3, .3, .3], 1, "degenerate_hac_variance", [2]),
    ],
)
def test_hac_missing_reason_precedence(ic: list[float], lag: int, reason: str, counts: list[int]) -> None:
    interval = bartlett_hac(ic, lag=lag)
    assert interval.reason == reason
    assert interval.pair_counts.tolist() == counts
    assert np.isnan([interval.variance, interval.lower, interval.upper]).all()


def test_directional_accuracy_always_long() -> None:
    result = evaluate_cross_section(
        _panel([[1] * 5], [[.01, .02, .03, .04, -.01]]), hac_lag=0,
    ).horizons["1D"]
    accuracy = result.direction
    assert (accuracy.hits, accuracy.denominator, accuracy.up_count) == (4, 5, 4)
    assert (accuracy.accuracy, accuracy.up_baseline, accuracy.lift) == pytest.approx((.80, .80, 0), **TOL)
    assert result.direction_by_bucket[1].reason == "no_directional_samples"


def test_direction_is_time_weighted_with_matching_denominators_by_bucket() -> None:
    panel = _panel(
        [[-2, -1, 0, 1, 2, np.nan], [-2, -1, 0, 1, 2, np.nan], [0] * 6],
        [[.01, .01, .01, 0, -.01, .01], [0, np.nan, .01, np.nan, 0, .01], [.01] * 6],
    )
    result = evaluate_cross_section(panel, hac_lag=0, n_quantiles=2, direction=-1).horizons["1D"]
    accuracy = result.direction
    assert (accuracy.hits, accuracy.denominator, accuracy.up_count) == (3, 6, 2)
    assert (accuracy.accuracy, accuracy.up_baseline, accuracy.lift) == pytest.approx((.375, .25, .125), **TOL)
    low, high = result.direction_by_bucket.values()
    assert (low.hits, low.denominator, low.up_count) == (2, 3, 2)
    assert (low.accuracy, low.up_baseline, low.lift) == pytest.approx((.5, .5, 0), **TOL)
    assert (high.hits, high.denominator, high.up_count) == (1, 3, 0)
    assert (high.accuracy, high.up_baseline, high.lift) == pytest.approx((.25, 0, .25), **TOL)


def test_signal_validity_and_return_mask_remain_independent() -> None:
    panel = _panel([[1, 2, 3, 4, 5]] * 2, [[.1, .2, .3, .4, .5]] * 2)
    panel.valid["1D"][0, 4] = False
    signal_valid = np.ones(panel.factor.shape, dtype=bool)
    signal_valid[:, 0] = False
    result = evaluate_cross_section(
        panel, hac_lag=0, n_quantiles=2, return_mode="cross_section_excess", signal_valid=signal_valid,
    )
    horizon = result.horizons["1D"]
    assert result.quantiles.labels.tolist() == [[0, 1, 1, 2, 2]] * 2
    assert horizon.ic.counts.tolist() == [3, 4]
    assert horizon.bucket_counts.tolist() == [[2, 1], [2, 2]]
    np.testing.assert_allclose(horizon.bucket_returns, [[-.05, .10], [-.10, .10]], **ARRAY_TOL)
    assert result.turnover[1].tolist() == [0, 0]


@pytest.mark.parametrize("period", [1, 2])
def test_synthetic_panel_matches_alphalens_reference(period: int) -> None:
    dates = pd.bdate_range("2024-01-02", periods=8, name="date")
    assets = pd.Index(list("ABCDEFGH"), name="asset")
    index = pd.MultiIndex.from_product([dates, assets])
    rng = np.random.default_rng(791)
    data = pd.DataFrame({"factor": rng.integers(0, 30, size=len(index)),
                         "1D": rng.normal(0, .02, size=len(index))}, index=index)
    # Changing universes exercise rank-then-match and turnover's current-set denominator.
    data = data.drop([(dates[1], "A"), (dates[2], "H"), (dates[4], "B")])
    # Keep an empty signal date to exercise period shifts along the frozen grid.
    data = data.drop(index=dates[3], level="date")
    data["factor_quantile"] = quantize_factor(data, quantiles=3)
    panel = EvaluationPanel.from_long(data[["factor", "1D"]], datetimes=dates)
    for mode in ("raw", "cross_section_excess"):
        result = evaluate_cross_section(
            panel, n_quantiles=3, hac_lag=2, return_mode=mode, diagnostic_period=period,
        )
        horizon = result.horizons["1D"]
        expected_ic = ref.factor_information_coefficient(data)["1D"].reindex(dates)
        np.testing.assert_allclose(horizon.ic.values, expected_ic, **ARRAY_TOL)
        assert horizon.ic_summary.positive_fraction == (expected_ic.dropna() > 0).mean()

        demeaned = mode == "cross_section_excess"
        means, _ = ref.mean_return_by_quantile(data, by_date=True, demeaned=demeaned)
        means = means["1D"].unstack("factor_quantile").reindex(dates)
        np.testing.assert_allclose(horizon.bucket_returns, means, **ARRAY_TOL)
        summary, errors = ref.mean_return_by_quantile(data, by_date=False, demeaned=demeaned)
        for q, bucket in horizon.bucket_summary.items():
            assert bucket.mean == pytest.approx(summary.loc[q, "1D"], **TOL)
            assert bucket.standard_error == pytest.approx(errors.loc[q, "1D"], **TOL)

        autocorrelation = ref.factor_rank_autocorrelation(data, period).reindex(dates)
        np.testing.assert_allclose(result.rank_autocorrelation, autocorrelation, **ARRAY_TOL)
        for q in range(1, 4):
            turnover = ref.quantile_turnover(data["factor_quantile"], q, period).reindex(dates)
            np.testing.assert_allclose(result.turnover[:, q - 1], turnover, **ARRAY_TOL)

    pearson = evaluate_cross_section(panel, hac_lag=0, ic_method="pearson")
    expected = data.groupby(level="date").apply(lambda row: pearsonr(row["factor"], row["1D"]).statistic)
    np.testing.assert_allclose(pearson.horizons["1D"].ic.values, expected.reindex(dates), **ARRAY_TOL)


def test_empty_panel_and_input_validation() -> None:
    panel = _panel([[1, 2]], [[.1, .2]])
    empty = EvaluationPanel.from_long(panel.to_long().iloc[:0])
    result = evaluate_cross_section(empty, hac_lag=0, n_quantiles=2)
    assert result.rank_autocorrelation.size == result.turnover.size == 0
    assert result.horizons["1D"].ic_summary.reason == "no_valid_cross_sections"
    assert result.horizons["1D"].direction.reason == "no_directional_samples"
    with pytest.raises(ValueError, match="signal grid"):
        bartlett_hac([[1, -1]], lag=0)
    with pytest.raises(ValueError, match="lag"):
        bartlett_hac([1, -1], lag=-1)
    with pytest.raises(ValueError, match="alpha"):
        bartlett_hac([1, -1], lag=0, alpha=1)
    with pytest.raises(ValueError, match="quantile"):
        evaluate_cross_section(panel, hac_lag=0, low_quantile=2, high_quantile=1)
    with pytest.raises(ValueError, match="chronological"):
        unordered = _panel([[1, 2]] * 2, [[.1, .2]] * 2)
        evaluate_cross_section(
            EvaluationPanel.from_long(unordered.to_long(), datetimes=unordered.datetimes[::-1]), hac_lag=0,
        )
