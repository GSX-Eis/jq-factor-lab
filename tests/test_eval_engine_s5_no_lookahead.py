"""Synthetic future-data tampering at the three point-in-time boundaries."""

import json
from dataclasses import asdict
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.evaluation.labels import ExecutionConvention, label_forward_returns
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.quantiles import fit_quantiles
from backtest_and_analysis.evaluation.split import SplitConfig, align_released_features, evaluate_splits
from eval_engine_label_test_support import _bars
from utils.session_calendar import SessionCalendar
from utils.trading_calendar import TradingCalendar


TS_OPTIONS = dict(block_length=2, seed=13, n_boot=15, min_valid_replicates=2)


@pytest.mark.parametrize("publish_delay", [-1, 0, 2])
def test_unclosed_higher_timeframe_mutation_leaves_earlier_signals_unchanged(publish_delay):
    start = pd.Timestamp("2024-01-02 14:30Z")
    ends = pd.date_range(start, periods=3, freq="15min")
    signals = pd.date_range(start + pd.Timedelta(minutes=1), ends[2], freq="1min")
    published = ends + pd.to_timedelta([0, publish_delay, 0], unit="min")
    high_bars = pd.DataFrame({
        "high_tf_bar_end": ends, "signal_available_at": published,
        "high": [104., 106., 108.], "close": [103., 105., 107.],
    })
    changed = high_bars.copy()
    changed.loc[1, ["high", "close"]] = [8.e9, 2.e9]

    def inputs(history):
        aligned = align_released_features(history, signals)
        index = pd.MultiIndex.from_product([signals, ["A"]], names=["datetime", "asset"])
        values = pd.DataFrame({"factor": (aligned.high - aligned.close).to_numpy(), 1: 0.01}, index=index)
        return aligned, EvaluationPanel.from_long(values)

    aligned, original = inputs(high_bars)
    tampered, mutated = inputs(changed)
    release = max(ends[1], published[1])
    prior = signals < release
    pd.testing.assert_frame_equal(aligned.loc[prior], tampered.loc[prior], check_exact=True)
    np.testing.assert_array_equal(original.factor[prior], mutated.factor[prior])
    np.testing.assert_array_equal(original.returns[1][prior], mutated.returns[1][prior])
    np.testing.assert_array_equal(original.valid[1][prior], mutated.valid[1][prior])
    assert tampered.loc[release, "high_tf_bar_end"] == ends[1]
    assert mutated.factor[signals.get_loc(release), 0] == 6.e9
    if publish_delay > 0:
        assert tampered.loc[ends[1], "high_tf_bar_end"] == ends[0]


def test_late_older_bar_does_not_replace_a_newer_released_bar():
    ends = pd.date_range("2024-01-02 14:30Z", periods=2, freq="15min")
    frame = pd.DataFrame({
        "high_tf_bar_end": ends, "signal_available_at": [ends[1] + pd.Timedelta("1min"), ends[1]],
        "close": [100., 200.],
    })
    signals = pd.date_range(ends[1] - pd.Timedelta("1min"), periods=3, freq="min")
    np.testing.assert_array_equal(align_released_features(frame, signals).close, [np.nan, 200., 200.])


def _assert_identical_training(actual, expected):
    for left, right in ((actual.statistics.datetimes, expected.statistics.datetimes),
                        (actual.statistics.assets, expected.statistics.assets)):
        pd.testing.assert_index_equal(left, right, exact=True)
    left, right = asdict(actual), asdict(expected)
    for side in (left, right):
        del side["statistics"]["datetimes"], side["statistics"]["assets"]
    np.testing.assert_equal(left, right)


def _utc_matrix(timestamps: pd.Series) -> np.ndarray:
    return timestamps.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()[:, None]


def _bar_evaluation(bars, times):
    labels = label_forward_returns(
        bars, convention=ExecutionConvention.OPEN_TO_CLOSE, calendar=SessionCalendar("NQ", "2024-01-02", "2024-01-02"),
        entry_offset=1, horizons=(1, 2), view="RTH", include_scheduled_exit=True,
    )
    keys = ["signal_timestamp", "instrument_id"]
    table = labels.set_index([*keys, "horizon_bars"])
    signals = bars.set_index(["timestamp_utc", "instrument_id"]).factor_value.rename("factor").rename_axis(keys)
    panel = EvaluationPanel.from_long(signals.to_frame().join(table.return_pct.unstack() / 100))
    available = pd.DatetimeIndex(labels.drop_duplicates("signal_timestamp").signal_available_at)
    entries = {h: _utc_matrix(table.xs(h, level="horizon_bars").entry_timestamp) for h in (1, 2)}
    exits = {h: _utc_matrix(table.xs(h, level="horizon_bars").scheduled_exit_timestamp) for h in (1, 2)}
    result = evaluate_splits(
        panel, {"train_end": available[7], "oos_start": available[8]}, mode="time_series",
        signal_available_at=available, entry_timestamps=entries, exit_timestamps=exits,
        direction=-1, n_quantiles=4, standardization="zscore", **TS_OPTIONS,
    )
    cutoff = times[9].tz_localize(None).to_datetime64()
    assert exits[1][7, 0] == exits[2][6, 0] == cutoff
    assert exits[2][7, 0] == times[10].tz_localize(None).to_datetime64()
    return result


@pytest.mark.parametrize("mutation", ["extreme", "missing"])
def test_mutating_oos_bar_prices_and_factors_cannot_change_any_training_statistics(mutation):
    times = pd.date_range("2024-01-02 14:30Z", periods=18, freq="1min")
    bars = _bars(times).assign(
        factor_value=np.arange(1., 19.) - 4,
        close=100 + np.array([0., 2., 3., 1., 2., 4., 6., 3., 5., 4., 7., 9., 8., 9., 8., 6., 8., 4.]),
    )
    original = _bar_evaluation(bars, times)
    changed = bars.copy()
    # Bar 8 closes at the first OOS signal, so its price is already known at the cutoff.
    changed.loc[8:, "factor_value"] = np.nan if mutation == "missing" else -1.e6 * changed.loc[8:, "factor_value"]
    price_fields = ["open", "high", "low", "close"]
    changed.loc[9:, price_fields] = np.nan if mutation == "missing" else 1.e6 * changed.loc[9:, price_fields]
    tampered = _bar_evaluation(changed, times)

    train = original["train"]
    _assert_identical_training(train, tampered["train"])
    assert train.split_purged_count == {1: 0, 2: 1}
    for h, keys in ((1, times[:8]), (2, times[:7])):
        pd.testing.assert_index_equal(train.statistics.datetimes[train.valid[h][:, 0]], keys, check_names=False)
    assert train.statistics.direction == -1
    assert train.fit_identity["instruments"][0]["training_edges"] == [-3., -1.25, 0.5, 2.25, 4.]
    assert train.fit_identity["instruments"][0]["center"] == 0.5
    assert train.fit_identity["instruments"][0]["scale"] == np.std(np.arange(-3., 5.))
    assert tampered["oos"].fit_identity == train.fit_identity
    assert not np.array_equal(tampered["oos"].valid[1], original["oos"].valid[1]) or not np.array_equal(
        tampered["oos"].statistics.quantiles.labels, original["oos"].statistics.quantiles.labels,
    )


def test_daily_uses_close_time_and_keeps_purge_count_after_exit_price_is_deleted():
    dates = pd.bdate_range("2024-11-25", periods=6)
    # 2024 Thanksgiving is absent from this synthetic exchange's schedule.
    dates = dates.delete(3)
    closes = pd.Series(
        pd.to_datetime(["2024-11-25 21:00Z", "2024-11-26 21:00Z", "2024-11-27 21:00Z",
                        "2024-11-29 18:00Z", "2024-12-02 21:00Z"]),
        index=dates,
    )
    calendar = TradingCalendar(dates, dates[0], dates[-1])
    keys = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "asset"])
    prices = pd.Series(np.arange(100., 115.), index=keys)
    factor = pd.Series(np.tile([1., 2., 3.], len(dates)), index=keys, name="factor")

    def run(pricing, signals, start):
        daily = label_forward_returns(
            pricing, convention=ExecutionConvention.CLOSE_TO_CLOSE, calendar=calendar,
            entry_offset=0, horizons=(1, 2),
        )
        assert daily.endpoints(1).at[(dates[2], "A"), "exit_date"] == dates[3]
        assert daily.scheduled_exit_dates(2).loc[dates[2]] == dates[4]
        panel = EvaluationPanel.from_long(signals.to_frame().join(daily.returns), datetimes=dates)
        exit_timestamps = {
            f"{h}D": np.repeat(_utc_matrix(daily.scheduled_exit_dates(h).map(closes)), 3, axis=1)
            for h in (1, 2)
        }
        return evaluate_splits(
            panel, SplitConfig(closes.iloc[2], start), mode="cross_section", hac_lag=1, n_quantiles=3,
            signal_available_at=pd.DatetimeIndex(closes), exit_timestamps=exit_timestamps,
        )["train"]

    # The configured bound lies between closes. The next scheduled signal is the early close.
    start = pd.Timestamp("2024-11-29 14:00Z")
    train = run(prices, factor, start)
    changed = run(prices.mask(keys.get_level_values(0) > dates[3]),
                  factor.mask(keys.get_level_values(0) >= dates[3]), start)
    _assert_identical_training(train, changed)
    assert train.split_purged_count == {"1D": 0, "2D": 3}
    np.testing.assert_array_equal(train.valid["1D"][-1], [True, True, True])
    np.testing.assert_array_equal(train.valid["2D"][-1], [False, False, False])


@pytest.mark.parametrize("replacement", [np.nan, 1.e6])
def test_cross_section_assignments_survive_later_exit_price_mutation(replacement):
    dates = pd.bdate_range("2024-01-02", periods=5)
    times = dates.tz_localize("UTC") + pd.Timedelta(hours=7)
    calendar = TradingCalendar(dates, dates[0], dates[-1])
    index = pd.MultiIndex.from_product([dates, list("ABCD")], names=["date", "asset"])
    prices = pd.Series(np.arange(100., 120.), index=index)
    factor = pd.Series(np.tile([1., 2., 3., 4.], len(dates)), index=index, name="factor")

    def evaluate(pricing):
        daily = label_forward_returns(
            pricing, convention=ExecutionConvention.CLOSE_TO_CLOSE, calendar=calendar,
            entry_offset=0, horizons=(1, 2),
        )
        panel = EvaluationPanel.from_long(factor.to_frame().join(daily.returns))
        closes = pd.Series(times, index=dates)
        exits = {
            f"{h}D": np.repeat(_utc_matrix(daily.scheduled_exit_dates(h).map(closes)), 4, axis=1)
            for h in (1, 2)
        }
        return evaluate_splits(
            panel, SplitConfig(times[1], times[2]), mode="cross_section", n_quantiles=2,
            hac_lag=1, signal_available_at=times, exit_timestamps=exits,
        )["oos"].statistics

    original = evaluate(prices)
    changed = prices.copy()
    changed.loc[(dates[4], "A")] = replacement
    tampered = evaluate(changed)
    np.testing.assert_array_equal(tampered.quantiles.labels, original.quantiles.labels)
    np.testing.assert_array_equal(tampered.quantiles.labels[0], [1, 1, 2, 2])
    # A forward exit from this signal appears only in the two-day label.
    np.testing.assert_array_equal(tampered.horizons["1D"].bucket_counts[0], [2, 2])
    np.testing.assert_array_equal(tampered.horizons["2D"].bucket_counts[0], [1 if np.isnan(replacement) else 2, 2])
    if not np.isnan(replacement):
        assert tampered.horizons["2D"].bucket_returns[0, 0] > 1000


def test_fitted_quantiles_merge_ties_extend_endpoints_and_preserve_empty_bucket():
    times = pd.date_range("2024-01-02 14:31Z", periods=15, freq="min")
    keys = pd.MultiIndex.from_product([times, ["A"]], names=["time", "asset"])
    frame = pd.DataFrame({
        "factor": [-4.] * 5 + [0.] * 2 + [4.] * 3 + [2.e9, -1.e6, -2., 1.e307, 1.e6],
        1: [0.01] * 11 + [0.02, 0.03, -0.01, -0.02],
    }, index=keys)
    # Training quantiles must include signal-only rows and exclude unavailable signals.
    frame.iloc[5, 1] = np.nan
    mask = np.ones((len(times), 1), dtype=bool)
    mask[[10, 14], 0] = False
    expected = times.tz_localize(None).to_numpy()[:, None]
    result = evaluate_splits(
        EvaluationPanel.from_long(frame), SplitConfig(times[10], times[11]), mode="time_series",
        signal_available_at=times, entry_timestamps={1: expected},
        exit_timestamps={1: expected + np.timedelta64(1, "m")}, signal_valid=mask,
        n_quantiles=4, direction=-1, standardization="median_mad", **TS_OPTIONS,
    )
    original, oos = result["train"], result["oos"]
    identity = original.fit_identity
    assert oos.fit_identity == identity
    assert identity["direction"] == -1
    assert identity["instruments"] == [
        {"asset": "A", "training_edges": [-4., -2., 3., 4.], "center": -2., "scale": 2.},
    ]
    payload = {key: value for key, value in identity.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert identity["sha256"] == sha256(encoded).hexdigest()
    stats = oos.statistics
    np.testing.assert_array_equal(stats.quantiles.labels[:, 0], [1, 1, 3, 0])
    np.testing.assert_array_equal(stats.quantiles.edges[:, 0], [-np.inf, -2., 3., np.inf, np.nan])
    np.testing.assert_array_equal(stats.quantiles.n_bins, [3])
    row = stats.horizons[1].instruments["A"]
    assert {bucket: value.n_samples for bucket, value in row.returns_by_bucket.items()} == {1: 2, 2: 0, 3: 1}
    assert row.returns_by_bucket[2].reason == "empty_bucket"
    # -2 is the frozen center and abstains. The two remaining directions both hit.
    assert (row.direction.hits, row.direction.denominator) == (2, 2)


def test_daily_date_bounds_preserve_the_gap_and_never_fit_on_oos():
    dates = pd.bdate_range("2024-01-02", periods=6)
    times = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    keys = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "asset"])
    frame = pd.DataFrame({"factor": [2., 2., 2., 999., -99., 999.], 1: 0.01}, index=keys)
    # Naive datetime64 trade instants encode UTC, even with a local signal clock.
    entries = times.tz_convert("UTC").tz_localize(None).to_numpy()[:, None]
    exits = entries + np.timedelta64(1, "h")
    result = evaluate_splits(
        EvaluationPanel.from_long(frame), {"train_end": "2024-01-04", "oos_start": "2024-01-08"},
        mode="time_series", signal_available_at=times, entry_timestamps={1: entries},
        exit_timestamps={1: exits}, n_quantiles=3, **TS_OPTIONS,
    )
    train, oos = result["train"], result["oos"]
    pd.testing.assert_index_equal(train.statistics.datetimes, dates[:3], check_names=False)
    pd.testing.assert_index_equal(oos.statistics.datetimes, dates[4:], check_names=False)
    assert oos.fit_identity["instruments"][0]["training_edges"] == [2.]
    np.testing.assert_array_equal(oos.statistics.quantiles.labels, [[0], [0]])
    assert oos.statistics.quantile_reasons.tolist() == ["unable_to_form_quantiles"]
    assert oos.statistics.horizons[1].instruments["A"].returns_by_bucket == {}


@pytest.mark.parametrize("bad_input", ["overlap", "missing_exit"])
def test_split_rejects_ambiguous_or_missing_hold_boundaries(bad_input):
    times = pd.date_range("2024-01-02 14:31Z", periods=3, freq="1min")
    keys = pd.MultiIndex.from_product([times, ["A"]], names=["time", "asset"])
    panel = EvaluationPanel.from_long(pd.DataFrame({"factor": [1., 2., 3.], 1: 0.01}, index=keys))
    exits = times.tz_localize(None).to_numpy()[:, None].copy() + np.timedelta64(1, "m")
    train_end, message = times[0], "Missing exit timestamp"
    if bad_input == "overlap":
        train_end, message = times[1], "train_end < oos_start"
    else:
        exits[0, 0] = np.datetime64("NaT")
    with pytest.raises(ValueError, match=message):
        evaluate_splits(
            panel, SplitConfig(train_end, times[1]), mode="cross_section", signal_available_at=times,
            exit_timestamps={1: exits}, hac_lag=1,
        )


@pytest.mark.parametrize("method", ["zscore", "median_mad"])
def test_training_fit_keeps_large_finite_factors_estimable(method):
    values = np.array([[-1.6e308], [1.6e308]])
    assets = pd.Index(["A"])
    fit = fit_quantiles(values, assets, np.isfinite(values), n_quantiles=4, direction=1, standardization=method)
    instrument, = fit.identity()["instruments"]
    np.testing.assert_allclose(
        instrument["training_edges"], [-1.6e308, -8.e307, 0., 8.e307, 1.6e308], rtol=2e-16, atol=0,
    )
    assert (instrument["center"], instrument["scale"]) == (0., 1.6e308)


@pytest.mark.parametrize("include_off_grid", [False, True])
@pytest.mark.parametrize("missing_first_factor", [False, True])
def test_purge_uses_first_scheduled_oos_row(include_off_grid, missing_first_factor):
    times = pd.date_range("2024-01-02 14:30Z", periods=4, freq="1min")
    oos_start = times[1]
    values = [[1., 2.], [np.nan, np.nan], [np.nan, np.nan if missing_first_factor else 3.], [4., 5.]]
    grid = np.array([[True, True], [False, False], [False, True], [True, True]])
    exits = np.repeat((times.tz_localize(None) + pd.Timedelta("2min")).to_numpy()[:, None], 2, axis=1)
    keys = pd.MultiIndex.from_product([times, ["A", "B"]], names=["time", "asset"])
    frame = pd.DataFrame({"factor": np.ravel(values), 1: 0.01, 2: 0.01}, index=keys)
    if not include_off_grid:
        frame = frame.drop(times[1], level="time")
        times = times.delete(1)
        grid = np.delete(grid, 1, axis=0)
        exits = np.delete(exits, 1, axis=0)

    result = evaluate_splits(
        EvaluationPanel.from_long(frame), SplitConfig(times[0], oos_start), mode="cross_section",
        signal_available_at=times, expected_grid=grid,
        exit_timestamps={1: exits, 2: exits + np.timedelta64(1, "m")}, hac_lag=1,
    )
    assert result["train"].split_purged_count == {1: 0, 2: 2}
    np.testing.assert_array_equal(result["train"].valid[1], [[True, True]])
    np.testing.assert_array_equal(result["train"].valid[2], [[False, False]])


@pytest.mark.parametrize(("method", "training", "oos", "center", "scale", "denominator"), [
    ("zscore", [0.1, 0.1, 0.1], [0.1, 0.1], 0.1, 1., 0),
    ("median_mad", [1.e-100, 2.e-100, 3.e-100, 4.e-100, 1.e308], [3.e-100, 2.e-100], 3.e-100, 1.e-100, 1),
])
def test_frozen_standardization_preserves_oos_abstentions(method, training, oos, center, scale, denominator):
    times = pd.date_range("2024-01-02 14:30Z", periods=len(training) + len(oos), freq="1min")
    keys = pd.MultiIndex.from_product([times, ["A"]], names=["time", "asset"])
    frame = pd.DataFrame({"factor": training + oos, 1: -0.01}, index=keys)
    entries = times.tz_localize(None).to_numpy()[:, None]
    result = evaluate_splits(
        EvaluationPanel.from_long(frame), SplitConfig(times[len(training) - 1], times[len(training)]),
        mode="time_series", signal_available_at=times, entry_timestamps={1: entries},
        exit_timestamps={1: entries + np.timedelta64(1, "m")}, standardization=method, **TS_OPTIONS,
    )
    oos_result = result["oos"]
    instrument, = oos_result.fit_identity["instruments"]
    assert instrument["center"] == center
    assert instrument["scale"] == pytest.approx(scale, rel=1e-15, abs=0)
    accuracy = oos_result.statistics.horizons[1].instruments["A"].direction
    assert (accuracy.hits, accuracy.denominator) == (denominator, denominator)


def test_release_gate_preserves_subsecond_signal_times():
    close = pd.DatetimeIndex(["2024-01-02 14:45:00Z"]).as_unit("s")
    signals = (close[0] + pd.to_timedelta([-1, 0, 1], unit="ms")).as_unit("ns")
    high_bars = pd.DataFrame({
        "high_tf_bar_end": close, "signal_available_at": close, "close": [105.],
    })
    aligned = align_released_features(high_bars, signals)
    np.testing.assert_array_equal(aligned.close, [np.nan, 105., 105.])
    pd.testing.assert_index_equal(aligned.index, signals.rename("signal_available_at"))
