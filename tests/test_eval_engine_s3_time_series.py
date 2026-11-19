"""Synthetic checks for S3-B's instrument-level time-series statistics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.time_series import evaluate_time_series


def _panel(factor, returns, *, times=None, assets=("A",)):
    if times is None:
        times = pd.date_range("2023-01-03 09:30", periods=len(factor), freq="min", tz="UTC")
    index = pd.MultiIndex.from_product([times, assets], names=["datetime", "asset"])
    frame = pd.DataFrame(
        {"factor": np.asarray(factor).ravel(), **{h: np.asarray(y).ravel() for h, y in returns.items()}},
        index=index,
    )
    return EvaluationPanel.from_long(frame, datetimes=times)


def _intervals(panel):
    entry = panel.datetimes.to_numpy(dtype="datetime64[ns]")[:, None] + np.timedelta64(1, "m")
    entry = np.broadcast_to(entry, panel.factor.shape)
    return {h: (entry.copy(), entry + np.timedelta64(h, "m")) for h in panel.returns}


def _evaluate(panel, intervals=None, **overrides):
    options = dict(block_length=max(panel.returns), seed=791, n_boot=9, min_valid_replicates=2)
    options.update(overrides)
    return evaluate_time_series(panel, _intervals(panel) if intervals is None else intervals, **options)


def test_bootstrap_reranks_moving_blocks_and_matches_independent_reference():
    x = np.array([1, 2, 3, 5, 8, 13, 13, 34, 55], dtype=float)
    y = np.array([8, 3, -9, 0, np.nan, 2, 7, 7, -2])
    signal_valid = np.ones((len(x), 1), dtype=bool)
    signal_valid[2] = False
    valid = np.isfinite(y) & signal_valid[:, 0]
    options = dict(block_length=4, seed=17, n_boot=64, alpha=0.2, signal_valid=signal_valid)

    def evaluate(x, y, **kwargs):
        panel = _panel(x, {2: y})
        return _evaluate(panel, **options, **kwargs).horizons[2].instruments["A"]

    result = evaluate(x, y)
    interval = result.bootstrap
    rng = np.random.default_rng(options["seed"])
    replicates, frozen_rank_replicates = [], []
    original_x_ranks = np.full(len(x), np.nan)
    original_y_ranks = original_x_ranks.copy()
    original_x_ranks[valid], original_y_ranks[valid] = rankdata(x[valid]), rankdata(y[valid])
    for _ in range(options["n_boot"]):
        starts = rng.integers(0, 6, size=3)
        block = np.concatenate([np.arange(start, start + 4) for start in starts])[:9]
        selected = [i for i in block if valid[i]]
        left, right = x[selected], y[selected]
        if len(selected) >= 3 and len(set(left)) > 1 and len(set(right)) > 1:
            replicates.append(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
            frozen_rank_replicates.append(
                np.corrcoef(original_x_ranks[selected], original_y_ranks[selected])[0, 1],
            )

    expected = np.quantile(replicates, [0.1, 0.9])
    assert np.max(np.abs(np.array(replicates) - frozen_rank_replicates)) > 0.1
    assert expected != pytest.approx(np.quantile(frozen_rank_replicates, [0.1, 0.9]))
    assert (interval.lower, interval.upper) == pytest.approx(expected, abs=1e-12)
    assert interval.valid_replicates == len(replicates)
    assert (interval.grid_size, interval.block_length, interval.seed, interval.n_boot,
            interval.min_valid_replicates, interval.alpha) == (9, 4, 17, 64, 2, 0.2)
    assert interval.confidence_level == 0.8
    assert interval.reason is None
    assert result.ic.counts.tolist() == [7]
    assert result.ic.values[0] == pytest.approx(np.corrcoef(rankdata(x[valid]), rankdata(y[valid]))[0, 1])
    assert evaluate(x, y).bootstrap == interval
    assert evaluate(x**2, y**3).bootstrap == interval

    pearson = evaluate(x, y, ic_method="pearson")
    assert pearson.ic.values[0] == pytest.approx(np.corrcoef(x[valid], y[valid])[0, 1])
    assert pearson.spearman_ic.values[0] == result.ic.values[0]
    assert pearson.bootstrap == interval


@pytest.mark.parametrize(
    "x,y,block_length,point_reason,interval_reason,valid_replicates",
    [
        ([1, 2, 3], [3, 1, 2], 3, None, "insufficient_bootstrap_grid", 0),
        ([1, 2, 3], [3, 1, 2], 4, None, "insufficient_bootstrap_grid", 0),
        ([1, 2, np.nan], [3, 1, 2], 3, "insufficient_samples", "insufficient_samples", 0),
        ([1, 1, 1], [2, 2, 2], 3, "constant_factor", "constant_factor", 0),
        ([1, 2, 3], [2, 2, 2], 1, "constant_return", "constant_return", 0),
        ([1, 2, 3, np.nan, np.nan], [3, 1, 2, np.nan, np.nan], 2, None,
         "insufficient_bootstrap_replicates", 1),
    ],
)
def test_bootstrap_missing_reasons(x, y, block_length, point_reason, interval_reason, valid_replicates):
    result = _evaluate(
        _panel(x, {1: y}), block_length=block_length, seed=0, n_boot=2,
    ).horizons[1].instruments["A"]
    assert result.ic.reasons[0] == point_reason
    assert result.bootstrap.reason == interval_reason
    assert result.bootstrap.valid_replicates == valid_replicates
    assert np.isnan(result.bootstrap.lower) and np.isnan(result.bootstrap.upper)


def test_return_distribution_and_direction_by_frozen_bucket():
    x = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    y = [-4, -2, 0, 2, 4, -2, 6, 8, np.nan, 999]
    panel = _panel(x, {1: y})
    labels = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 3, 4]]).T
    mask = np.ones_like(labels, dtype=bool)
    mask[-1] = False
    result = _evaluate(panel, bucket_labels=labels, n_quantiles=4, signal_valid=mask)
    row = result.horizons[1].instruments["A"]
    assert result.quantiles.n_bins.tolist() == [4]
    assert result.quantiles.labels[8:, 0].tolist() == [3, 0]
    assert labels[-1, 0] == 4

    dist = row.returns
    assert dist.n_samples == 8 and dist.reason is None
    assert (dist.mean, dist.standard_error) == pytest.approx((1.5, 1.5))
    assert (dist.p5, dist.p25, dist.p50, dist.p75, dist.p95) == pytest.approx((-3.3, -2, 1, 4.5, 7.3))
    assert (dist.mean_le_p5, dist.mean_le_p25, dist.mean_ge_p75, dist.mean_ge_p95) == pytest.approx(
        (-4, -8 / 3, 7, 8),
    )
    low = row.returns_by_bucket[1]
    assert low.n_samples == 5 and low.reason is None
    assert (low.mean, low.standard_error) == pytest.approx((0, np.sqrt(2)))
    assert (low.p5, low.p25, low.p50, low.p75, low.p95) == pytest.approx((-3.6, -2, 0, 2, 3.6))
    assert (low.mean_le_p5, low.mean_le_p25, low.mean_ge_p75, low.mean_ge_p95) == (-4, -3, 3, 4)
    assert row.returns_by_bucket[2].n_samples == 3
    for bucket in (3, 4):
        assert row.returns_by_bucket[bucket].n_samples == 0
        assert row.returns_by_bucket[bucket].reason == "empty_bucket"
        assert np.isnan(row.returns_by_bucket[bucket].p50)
        assert row.direction_by_bucket[bucket].denominator == 0
        assert row.direction_by_bucket[bucket].reason == "no_directional_samples"
    assert (row.direction.hits, row.direction.denominator, row.direction.up_count) == (6, 7, 4)
    assert (row.direction.accuracy, row.direction.up_baseline, row.direction.lift) == pytest.approx(
        (6 / 7, 4 / 7, 2 / 7),
    )
    assert (row.direction_by_bucket[1].accuracy, row.direction_by_bucket[1].lift) == (1, 0.5)
    assert (row.direction_by_bucket[2].accuracy, row.direction_by_bucket[2].lift) == (2 / 3, 0)


def test_direction_counts_zero_returns_and_applies_frozen_sign():
    panel = _panel([1, 2, 3, 4, 5, 0], {1: [0, 1, 2, 3, 4, 10]})
    positive = _evaluate(panel).horizons[1].instruments["A"].direction
    negative = _evaluate(panel, direction=-1).horizons[1].instruments["A"].direction
    assert (positive.hits, positive.denominator, positive.up_count) == (4, 5, 4)
    assert (positive.accuracy, positive.up_baseline, positive.lift) == (0.8, 0.8, 0)
    assert (negative.hits, negative.denominator, negative.up_count) == (0, 5, 4)
    assert (negative.accuracy, negative.up_baseline, negative.lift) == (0, 0.8, -0.8)
    abstained = _evaluate(_panel([0, 0, 0], {1: [1, 2, 3]})).horizons[1].instruments["A"]
    assert abstained.returns.n_samples == 3
    assert (abstained.direction.hits, abstained.direction.denominator, abstained.direction.up_count) == (0, 0, 0)
    assert np.isnan(abstained.direction.accuracy) and abstained.direction.reason == "no_directional_samples"


def test_local_buckets_ignore_missing_labels_and_describe_singletons():
    panel = _panel([1, 2, 3, 4], {1: [np.nan, 20, 30, 40], 2: [10, 20, np.nan, 40]})
    result = _evaluate(panel, n_quantiles=2)
    assert result.quantiles.labels[:, 0].tolist() == [1, 1, 2, 2]
    assert result.quantile_reasons.tolist() == [None]
    for horizon, bucket, value in ((1, 1, 20), (2, 2, 40)):
        dist = result.horizons[horizon].instruments["A"].returns_by_bucket[bucket]
        assert (dist.n_samples, dist.mean, dist.p5, dist.p95) == (1, value, value, value)
        assert dist.reason == "insufficient_time_points" and np.isnan(dist.standard_error)


def _two_sessions():
    times = pd.date_range("2023-01-03 09:30", periods=6, freq="min", tz="UTC")
    times = times.append(times + pd.Timedelta(days=1))
    y = [10, 1, np.nan, 2, 9, np.nan, 8, 3, np.nan, 4, 7, np.nan]
    panel = _panel(np.arange(12), {2: y}, times=times)
    intervals = _intervals(panel)
    for timestamps in intervals[2]:
        timestamps[~panel.valid[2]] = np.datetime64("NaT")
    return panel, intervals


def test_all_offsets_keep_gap_slots_and_accept_touching_half_open_intervals():
    panel, intervals = _two_sessions()
    entry, exit = intervals[2]
    assert exit[1, 0] == entry[3, 0]
    phases = _evaluate(panel, intervals).horizons[2].instruments["A"].non_overlapping_ic
    assert list(phases) == [0, 1]
    assert [ic.counts[0] for ic in phases.values()] == [4, 4]
    assert [ic.values[0] for ic in phases.values()] == [-1, 1]
    assert [ic.reasons[0] for ic in phases.values()] == [None, None]


@pytest.mark.parametrize("earlier,later,offset", [(1, 3, 1), (4, 6, 0)])
def test_overlap_fails_within_and_across_sessions(earlier, later, offset):
    panel, intervals = _two_sessions()
    entry, exit = intervals[2]
    exit[earlier, 0] = entry[later, 0] + np.timedelta64(1, "ns")
    with pytest.raises(AssertionError, match=f"'A', horizon 2, offset {offset}"):
        _evaluate(panel, intervals)


def test_all_offsets_report_short_and_empty_phases():
    horizon = _evaluate(_panel([1, 2, 3], {5: [3, 2, 1]})).horizons[5]
    result = horizon.instruments["A"]
    assert list(result.non_overlapping_ic) == list(range(5))
    assert [ic.counts[0] for ic in result.non_overlapping_ic.values()] == [1, 1, 1, 0, 0]
    assert all(ic.reasons[0] == "insufficient_samples" for ic in result.non_overlapping_ic.values())
    assert all(summary.reason == "no_valid_instruments" for summary in horizon.non_overlapping_ic_summary.values())


def test_instrument_statistics_use_own_grids_and_equal_instrument_weight():
    panel = _panel(
        [[0, 50, 1], [1, 7, 1], [2, 8, 1], [3, 9, 1], [4, 10, 1], [5, 11, 1]],
        {2: [[0, 999, 2], [1, 15, 3], [2, 14, 4], [3, 13, 5], [4, 12, 6], [5, 11, 7]]},
        assets=("A", "B", "C"),
    )
    grid = np.ones_like(panel.factor, dtype=bool)
    grid[0, 1] = False
    result = _evaluate(panel, expected_grid=grid, n_quantiles=2)
    horizon = result.horizons[2]
    a, b, c = horizon.instruments.values()
    assert [row.ic.counts[0] for row in (a, b, c)] == [6, 5, 6]
    assert (a.ic.values[0], b.ic.values[0], c.ic.reasons[0]) == (1, -1, "constant_factor")
    summary = horizon.ic_summary
    assert (summary.n_instruments, summary.n_valid, summary.mean, summary.median) == (3, 2, 0, 0)
    assert b.bootstrap.grid_size == 5 and b.returns.p95 < 16
    assert [ic.counts[0] for ic in b.non_overlapping_ic.values()] == [3, 2]
    assert horizon.non_overlapping_ic_summary[0].mean == 0
    assert horizon.non_overlapping_ic_summary[1].mean == 1
    assert horizon.non_overlapping_ic_summary[1].n_valid == 1
    assert result.quantiles.labels[0, 1] == 0
    assert result.quantile_reasons.tolist() == [None, None, "unable_to_form_quantiles"]
    assert c.returns_by_bucket == {} and c.returns.n_samples == 6


def test_missing_interval_on_valid_pair_is_an_input_error():
    panel, intervals = _two_sessions()
    intervals[2][0][0, 0] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="entry_timestamp < exit_timestamp"):
        _evaluate(panel, intervals)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"block_length": 1}, "block_length must be >= horizon"),
        ({"seed": -1}, "seed"),
        ({"min_valid_replicates": 1}, "min_valid_replicates"),
        ({"alpha": 1.0}, "alpha"),
        ({"expected_grid": [True]}, "expected_grid"),
        ({"bucket_labels": [[6], [1], [2]]}, "bucket_labels"),
    ],
)
def test_invalid_evaluation_parameters(options, match):
    with pytest.raises(ValueError, match=match):
        _evaluate(_panel([1, 2, 3], {2: [3, 2, 1]}), **options)
