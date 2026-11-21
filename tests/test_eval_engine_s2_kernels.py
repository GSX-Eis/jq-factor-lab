"""S2 kernel parity, frozen S0 counterexamples, and vectorization checks."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, rankdata

from backtest_and_analysis.evaluation import kernels
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from reference import eval_slow_ref

TOL = {"atol": 1e-12, "rtol": 0, "equal_nan": True}


@pytest.fixture
def observations() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = np.array([
        [1, 2, 2, 4, 5, 6],
        [1, 1, 1, 1, np.nan, np.inf],
        [2, 3, 7, 8, 9, 0],
        [np.nan, 2, 3, 4, np.nan, -np.inf],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [4, 4, 4, 4, 4, 4],
        [2, 0, 0, 0, 1, 2],
    ], dtype=float)
    returns = np.array([
        [3, np.nan, 4, 4, 5, 7],
        [1, 2, 3, 4, 5, 6],
        [2, 2, 2, 2, 2, 2],
        [1, 2, np.nan, np.inf, 3, 4],
        [1, 2, 3, 4, 5, 6],
        [2, 2, 2, 2, 2, 2],
        [3, 7, 1, 10, 4, 6],
    ], dtype=float)
    valid = np.ones(factor.shape, dtype=bool)
    valid[0, -1] = False
    valid[6, 1] = False
    return factor, returns, valid


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
@pytest.mark.parametrize("minimum", [2, 3])
def test_ranks_and_correlations_match_slow_reference(
    observations: tuple[np.ndarray, np.ndarray, np.ndarray], axis: int, minimum: int,
) -> None:
    x, y, valid = observations
    for actual, expected in zip(
        kernels.average_rank(x, y, axis=axis, valid=valid),
        eval_slow_ref.average_rank(x, y, valid, axis),
    ):
        np.testing.assert_array_equal(actual, expected)

    for method in (kernels.pearson, kernels.spearman):
        actual = method(x, y, axis=axis, valid=valid, min_samples=minimum)
        expected = eval_slow_ref.correlation(x, y, valid, axis, minimum, method.__name__)
        np.testing.assert_allclose(actual.values, expected[0], **TOL)
        np.testing.assert_array_equal(actual.counts, expected[1])
        np.testing.assert_array_equal(actual.reasons, expected[2])


def test_correlation_states_prefer_sample_size_then_constant_factor() -> None:
    x = [[1, 2, 3], [5, 5, 5], [2, 3, 4], [7, 7, 7], [2, 3, 4], [np.nan] * 3]
    y = [[3, 2, 1], [5, 6, 7], [8, 8, 8], [9, 9, 9], [np.nan, 1, np.nan], [1, 2, 3]]
    expected = [None, "constant_factor", "constant_return", "constant_factor",
                "insufficient_cross_section", "insufficient_cross_section"]
    for method in (kernels.pearson, kernels.spearman):
        stats = method(x, y)
        np.testing.assert_allclose(stats.values, [-1, *([np.nan] * 5)], **TOL)
        np.testing.assert_array_equal(stats.counts, [3, 3, 3, 3, 1, 0])
        assert stats.reasons.tolist() == expected


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integers_keep_exact_order_and_ties_before_ranking(dtype: type) -> None:
    x = np.array([[2**53 + 1, 2**53 + 2, 2**53, 2**53 + 1]], dtype=dtype)
    y = np.array([[2, 3, 1, 2]])
    valid = np.array([[True, False, True, True]])
    expected_ranks = eval_slow_ref.average_rank(x, y, valid, 1)
    np.testing.assert_array_equal(kernels.average_rank(x, y, valid=valid), expected_ranks)
    for method in (kernels.spearman, kernels.pearson):
        expected = eval_slow_ref.correlation(x, y, valid, 1, 2, method.__name__)
        actual = method(x, y, valid=valid)
        np.testing.assert_allclose(actual.values, expected[0], **TOL)
        np.testing.assert_array_equal(actual.counts, expected[1])
        np.testing.assert_array_equal(actual.reasons, expected[2])


def test_pearson_preserves_variation_at_extreme_float_scales() -> None:
    large = np.finfo(float).max
    tiny = np.nextafter(0.0, 1.0)
    x = [[-large, 0, large], [tiny, 2 * tiny, 3 * tiny], [np.nextafter(large, 0), large, large]]
    y = [[1, 2, 3], [3, 2, 1], [1, 2, 2]]
    result = kernels.pearson(x, y, min_samples=3)
    np.testing.assert_allclose(result.values, [1, -1, 1], **TOL)
    assert result.reasons.tolist() == [None, None, None]


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("n_quantiles", [1, 2, 4, 7, 10])
def test_qcut_edges_labels_and_bucket_reductions_match_slow_reference(
    observations: tuple[np.ndarray, np.ndarray, np.ndarray], axis: int, n_quantiles: int,
) -> None:
    x, y, valid = observations
    actual = kernels.quantile_buckets(x, n_quantiles, axis=axis, valid=valid)
    labels, edges, n_bins = eval_slow_ref.quantile_buckets(x, valid, axis, n_quantiles)
    np.testing.assert_array_equal(actual.labels, labels)
    np.testing.assert_allclose(actual.edges, edges, **TOL)
    np.testing.assert_array_equal(actual.n_bins, n_bins)

    paired = valid & np.isfinite(x) & np.isfinite(y)
    buckets = kernels.bucket_sums(y, labels, n_quantiles, axis=axis, valid=paired)
    expected_sums, expected_counts = eval_slow_ref.bucket_sums(y, labels, paired, axis, n_quantiles)
    np.testing.assert_allclose(buckets.sums, expected_sums, **TOL)
    np.testing.assert_array_equal(buckets.counts, expected_counts)


def test_qcut_right_closed_boundaries_and_unoccupied_bin_numbers() -> None:
    result = kernels.quantile_buckets([[0, 1, 2], [1, 1, 2], [3, 3, 3]], 4)
    np.testing.assert_array_equal(result.labels, [[1, 2, 4], [1, 1, 2], [0, 0, 0]])
    np.testing.assert_array_equal(result.edges, [
        [0, .5, 1, 1.5, 2], [1, 1.5, 2, np.nan, np.nan], [3, np.nan, np.nan, np.nan, np.nan],
    ])
    np.testing.assert_array_equal(result.n_bins, [4, 2, 0])


def test_qcut_thirds_preserve_pandas_percentile_rounding() -> None:
    x = np.array([[-.93, -.09, .56, .65]])
    actual = kernels.quantile_buckets(x, 3)
    expected = eval_slow_ref.quantile_buckets(x, np.ones(x.shape, dtype=bool), 1, 3)
    np.testing.assert_array_equal(actual.labels, expected[0])
    np.testing.assert_array_equal(actual.edges, expected[1])


@pytest.mark.parametrize("axis", [0, 1])
def test_qcut_preserves_float32_interpolation(axis: int) -> None:
    rows = np.array([[-.1, .2, np.nan], [.25, .75, 1]], dtype=np.float32)
    keep = np.array([[1, 1, 1], [1, 0, 1]], dtype=bool)
    x, valid = (rows, keep) if axis == 1 else (rows.T, keep.T)
    result = kernels.quantile_buckets(x, 2, axis=axis, valid=valid)
    labels, edges, counts = eval_slow_ref.quantile_buckets(x, valid, axis, 2)
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.edges, edges, **TOL)
    np.testing.assert_array_equal(result.n_bins, counts)


@pytest.mark.parametrize("n_quantiles", [1, 2, 4, 9])
def test_qcut_matches_pandas_when_endpoint_difference_overflows(n_quantiles: int) -> None:
    large = np.finfo(float).max
    x = large * np.array([[-1, 1, np.nan, np.nan, np.nan, np.nan],
                          [-1, .92, -.9, 1, -.85, -.3],
                          [1, -.85, -.9, .92, -1, -.3]])
    valid = np.ones(x.shape, dtype=bool)
    valid[1:, -1] = False
    with np.errstate(over="ignore", invalid="ignore"):
        labels, edges, counts = eval_slow_ref.quantile_buckets(x, valid, 1, n_quantiles)
    result = kernels.quantile_buckets(x, n_quantiles, valid=valid)
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_array_equal(result.edges, edges)
    np.testing.assert_array_equal(result.n_bins, counts)


@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 3), (2, 3)])
@pytest.mark.parametrize("axis", [0, 1])
def test_empty_and_all_missing_panels_keep_requested_shapes(shape: tuple[int, int], axis: int) -> None:
    x = np.full(shape, np.nan)
    valid = np.ones(shape, dtype=bool)
    x_rank, y_rank = kernels.average_rank(x, x, axis=axis)
    np.testing.assert_array_equal(x_rank, x)
    np.testing.assert_array_equal(y_rank, x)

    size = shape[1 - axis]
    for method in (kernels.pearson, kernels.spearman):
        result = method(x, x, axis=axis)
        assert result.values.shape == (size,)
        assert np.isnan(result.values).all()
        np.testing.assert_array_equal(result.counts, np.zeros(size))
        np.testing.assert_array_equal(
            result.reasons, eval_slow_ref.correlation(x, x, valid, axis, 2, method.__name__)[2],
        )

    expected_shape = list(shape)
    expected_shape[axis] = 3
    quantiles = kernels.quantile_buckets(x, 2, axis=axis)
    assert quantiles.edges.shape == tuple(expected_shape)
    assert np.isnan(quantiles.edges).all()
    np.testing.assert_array_equal(quantiles.labels, np.zeros(shape, dtype=int))
    np.testing.assert_array_equal(quantiles.n_bins, np.zeros(size, dtype=int))
    buckets = kernels.bucket_sums(x, quantiles.labels, 3, axis=axis)
    np.testing.assert_array_equal(buckets.sums, np.zeros(expected_shape))
    np.testing.assert_array_equal(buckets.counts, np.zeros(expected_shape))
    assert buckets.sums.dtype == np.float64


def test_panel_round_trip_preserves_sparse_observations_and_supplied_grid() -> None:
    dates = pd.date_range("2024-01-02", periods=4, tz="UTC", name="datetime")
    assets = pd.Index(["a", "unobserved", "b"], name="asset")
    index = pd.MultiIndex.from_tuples(
        [(dates[3], "b"), (dates[0], "a"), (dates[3], "a"), (dates[1], "b")],
        names=["datetime", "asset"],
    )
    frame = pd.DataFrame({
        5: pd.array([pd.NA, .25, .5, pd.NA], dtype="Float64"),
        "signal": [1, np.inf, 2, np.nan],
        1: [.02, .04, np.nan, np.nan],
    }, index=index)
    panel = EvaluationPanel.from_long(frame, factor_column="signal", datetimes=dates, assets=assets)
    pd.testing.assert_index_equal(panel.datetimes, dates)
    pd.testing.assert_index_equal(panel.assets, assets)
    np.testing.assert_array_equal(panel.factor, [
        [np.inf, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
        [2, np.nan, 1],
    ])
    np.testing.assert_array_equal(panel.valid[1], [
        [False, False, False], [False, False, False], [False, False, False], [False, False, True],
    ])
    np.testing.assert_array_equal(panel.valid[5], [
        [False, False, False], [False, False, False], [False, False, False], [True, False, False],
    ])
    pd.testing.assert_frame_equal(panel.to_long(), frame.astype(float))

    inferred = EvaluationPanel.from_long(frame, factor_column="signal")
    pd.testing.assert_index_equal(inferred.datetimes, dates[[0, 1, 3]])
    assert inferred.assets.tolist() == ["b", "a"]
    pd.testing.assert_frame_equal(inferred.to_long(), frame.astype(float))
    empty = EvaluationPanel.from_long(frame.iloc[:0], factor_column="signal")
    assert empty.factor.shape == (0, 0)
    pd.testing.assert_frame_equal(empty.to_long(), frame.iloc[:0].astype(float))


@pytest.mark.parametrize("column", ["factor", "1D"])
@pytest.mark.parametrize("dtype, value", [
    ("int64", 2**53 + 1),
    ("Int64", 2**53 + 1),
    (pd.SparseDtype("int64"), 2**53 + 1),
    ("uint64", 2**64 - 1),
    ("UInt64", 2**64 - 1),
])
def test_panel_rejects_integer_values_that_would_lose_precision(
    column: str, dtype: str | pd.SparseDtype, value: int,
) -> None:
    dates = pd.date_range("2024-01-02", periods=2)
    index = pd.MultiIndex.from_product([dates, ["a"]], names=["datetime", "asset"])
    frame = pd.DataFrame({"factor": [1, 2], "1D": [3, 4]}, index=index)
    frame[column] = pd.array([2**53 + 2, value], dtype=dtype)
    with pytest.raises(ValueError, match=rf"{column}.*exactly.*float64"):
        EvaluationPanel.from_long(frame)

    frame[column] = pd.array([2**53 + 2, 2**53 + 4], dtype=dtype)
    panel = EvaluationPanel.from_long(frame)
    pd.testing.assert_frame_equal(panel.to_long(), frame.astype(float))


def test_panel_rejects_ambiguous_keys() -> None:
    dates = pd.date_range("2024-01-02", periods=2)
    index = pd.MultiIndex.from_product([dates, ["a"]], names=["datetime", "asset"])
    frame = pd.DataFrame({"factor": [1, 2], "1D": [3, 4]}, index=index)
    with pytest.raises(ValueError, match="unique two-level"):
        EvaluationPanel.from_long(pd.concat([frame, frame]))
    with pytest.raises(ValueError, match="contain every observed key"):
        EvaluationPanel.from_long(frame, datetimes=dates[1:])
    with pytest.raises(ValueError, match="without missing values"):
        EvaluationPanel.from_long(frame.rename(index={"a": None}, level="asset"))
    with pytest.raises(ValueError, match="unique columns"):
        EvaluationPanel.from_long(frame, factor_column="absent")


def test_kernels_reject_misaligned_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        kernels.pearson([1, 2], [1, 2])
    with pytest.raises(ValueError, match="same panel shape"):
        kernels.spearman([[1, 2]], [[1], [2]])
    with pytest.raises(ValueError, match="boolean mask"):
        kernels.average_rank([[1, 2]], [[1, 2]], valid=[[1, 0]])
    with pytest.raises(ValueError, match="axis"):
        kernels.quantile_buckets([[1, 2]], 2, axis=2)
    with pytest.raises(ValueError, match="n_quantiles"):
        kernels.quantile_buckets([[1, 2]], 0)
    with pytest.raises(ValueError, match="min_samples"):
        kernels.pearson([[1, 2]], [[1, 2]], min_samples=1)
    with pytest.raises(ValueError, match="labels must be in"):
        kernels.bucket_sums([[1, 2]], [[2, 3]], 2)


def test_s0_spearman_filters_pairs_before_ranking() -> None:
    x = np.array([[1, 2, 3, 4]])
    y = np.array([[2, np.nan, 1, 3]])
    result = kernels.spearman(x, y)
    assert result.values[0] == pytest.approx(0.5, abs=1e-12, rel=0)
    assert result.counts.tolist() == [3]
    assert result.reasons.tolist() == [None]
    mask = np.isfinite(y[0])
    incorrectly_ranked = pearsonr(rankdata(x[0])[mask], rankdata(y[0, mask]))[0]
    assert incorrectly_ranked == pytest.approx(0.3273268354, abs=1e-10, rel=0)


def test_s0_horizon_masks_leave_short_ic_and_bucket_returns_unchanged() -> None:
    days = pd.bdate_range("2024-01-02", periods=4)
    index = pd.MultiIndex.from_product([days, list("ABCD")], names=["datetime", "asset"])
    frame = pd.DataFrame({
        "factor": [1, 1, 3, 4, 2, 2, 2, 2, *([np.nan] * 4), 1, 2, 3, 4],
        "1D": [.02, .01, .03, .04, .01, .02, .03, .04, *([np.nan] * 4), .01, np.nan, .03, .04],
        "5D": [.05, .10, .15, .20] * 2 + [np.nan] * 4 + [.05, .10, .15, .20],
    }, index=index)
    missing_long = frame.copy()
    missing_long.loc[days[0], "5D"] = np.nan
    missing_long.loc[(days[3], "B"), "5D"] = np.nan

    for long_data, expected_long_counts in ((frame, [4, 4, 0, 4]), (missing_long, [0, 4, 0, 3])):
        panel = EvaluationPanel.from_long(long_data)
        short = kernels.spearman(panel.factor, panel.returns["1D"], valid=panel.valid["1D"])
        long = kernels.spearman(panel.factor, panel.returns["5D"], valid=panel.valid["5D"])
        np.testing.assert_allclose(short.values, [0.9486832980505138, np.nan, np.nan, 1.0], **TOL)
        assert short.counts.tolist() == [4, 4, 0, 3]
        assert short.reasons.tolist() == [None, "constant_factor", "insufficient_cross_section", None]
        assert long.counts.tolist() == expected_long_counts

        quantiles = kernels.quantile_buckets(panel.factor, 2)
        assert quantiles.labels[3].tolist() == [1, 1, 2, 2]
        buckets = kernels.bucket_sums(panel.returns["1D"], quantiles.labels, 2, valid=panel.valid["1D"])
        np.testing.assert_array_equal(buckets.counts, [[2, 2], [0, 0], [0, 0], [1, 2]])
        raw = np.divide(buckets.sums, buckets.counts, out=np.full((4, 2), np.nan), where=buckets.counts > 0)
        np.testing.assert_allclose(raw, [[.015, .035], [np.nan] * 2, [np.nan] * 2, [.01, .035]], **TOL)
        excess = raw - np.array([.025, .025, np.nan, .08 / 3])[:, None]
        np.testing.assert_allclose(excess, [
            [-.01, .01], [np.nan] * 2, [np.nan] * 2, [-.0166666666666667, .0083333333333333],
        ], **TOL)


def test_s0_bucket_counts_preserve_time_weighting_and_return_shift() -> None:
    labels = np.array([[1, 1, 2, *([0] * 8)], [1] * 10 + [2], [2, *([0] * 10)]])
    values = np.array([[0, 0, .02, *([np.nan] * 8)], [.06] * 10 + [.12], [.50, *([np.nan] * 10)]])
    stats = kernels.bucket_sums(values, labels, 2)
    shifted = kernels.bucket_sums(values + .125, labels, 2)
    np.testing.assert_array_equal(stats.counts, [[2, 1], [10, 1], [0, 1]])
    available = stats.counts != 0
    raw = np.divide(stats.sums, stats.counts, out=np.full(stats.sums.shape, np.nan), where=available)
    assert raw[:2, 0].mean() == pytest.approx(.03)
    assert stats.sums[:, 0].sum() / stats.counts[:, 0].sum() == pytest.approx(.05)
    np.testing.assert_allclose(raw[:2, 1] - raw[:2, 0], [.02, .06], **TOL)
    np.testing.assert_allclose((shifted.sums - stats.sums)[available] / stats.counts[available], .125, **TOL)


def test_kernels_do_not_dispatch_python_work_per_slice() -> None:
    tree = ast.parse(Path(kernels.__file__).read_bytes())
    forbidden_calls = {"apply", "apply_along_axis", "applymap", "groupby", "iterrows",
                       "itertuples", "map", "vectorize", "frompyfunc"}
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.While, ast.comprehension))
        if isinstance(node, ast.For):
            assert ast.unparse(node.iter) in {
                "range(capacity.bit_length())", "range((width - 1).bit_length())",
                "edges.T",  # A bounded loop over bucket boundaries across all slices.
            }
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            assert name not in forbidden_calls


def test_long_single_instrument_quantiles_match_sort_reference() -> None:
    rng = np.random.default_rng(827)
    values = rng.integers(-9, 10, size=(257, 1)).astype(float)
    values[::7] = np.nan
    valid = rng.random(values.shape) > .2
    actual = kernels.quantile_buckets(values, 5, axis=0, valid=valid)
    labels, edges, counts = eval_slow_ref.quantile_buckets(values, valid, 0, 5)
    np.testing.assert_array_equal(actual.labels, labels)
    np.testing.assert_array_equal(actual.edges, edges)
    np.testing.assert_array_equal(actual.n_bins, counts)


def test_long_single_series_spearman_matches_axis_kernel_with_ties_and_gaps():
    rng = np.random.default_rng(827)
    factor = rng.integers(0, 32, size=(65_539, 1), dtype=np.int64) + 2**53
    returns = rng.integers(0, 48, size=factor.shape).astype(float)
    returns[::57] = np.nan
    valid = (factor % 5 != 0)
    expected = kernels.spearman(
        np.repeat(factor, 2, axis=1), np.repeat(returns, 2, axis=1),
        axis=0, valid=np.repeat(valid, 2, axis=1),
    )
    actual = kernels.spearman(factor, returns, axis=0, valid=valid)
    np.testing.assert_allclose(actual.values, expected.values[:1], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(actual.counts, expected.counts[:1])
    np.testing.assert_array_equal(actual.reasons, expected.reasons[:1])
