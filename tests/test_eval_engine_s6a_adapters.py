"""Exercise the unified facade and replay the legacy adapters on synthetic panels."""

import json


import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner
from backtest_and_analysis.evaluation import EvaluationEngine
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.split import SplitConfig
from backtest_and_analysis.time_series import BarCrossSectionalEvaluator, TimeSeriesEvaluator
from fixtures.eval_engine.baseline.regenerate import (
    BASELINE_DIR, MODES, _freeze_frame, run_snapshot, synthetic_inputs,
)
from test_time_series_evaluator import labels


@pytest.mark.parametrize("mode", MODES)
def test_replay_frozen_s1a_synthetic(mode: str) -> None:
    frozen = json.loads((BASELINE_DIR / "synthetic.json").read_text())
    identity, *inputs = synthetic_inputs()
    assert identity == frozen["input"]
    expected = frozen["paths"][mode]
    actual = run_snapshot(*inputs, mode=mode)

    assert actual["max_loss"] == expected["max_loss"]
    assert actual["results"].keys() == expected["results"].keys()
    for metric, frame in actual["results"].items():
        oracle = expected["results"][metric]
        for field in ("index", "index_names", "columns", "column_names"):
            assert frame[field] == oracle[field], (metric, field)
        np.testing.assert_allclose(
            np.asarray(frame["values"], dtype=float), np.asarray(oracle["values"], dtype=float),
            atol=1e-12, rtol=0, equal_nan=True, err_msg=metric,
        )


def _daily_panel() -> EvaluationPanel:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=3), ["A", "B", "C", "D"]], names=["date", "asset"],
    )
    return EvaluationPanel.from_long(pd.DataFrame({
        "factor": [1., 2., 3., 4.] * 3,
        "1D": [.01, .03, .05, .07, .02, .01, .04, .03, .03, .02, .01, 0.],
        "5D": [.02, .03, .04, .05] + [np.nan] * 8,
    }, index=index))


def test_engine_daily_full_sample_metrics_and_single_icir() -> None:
    panel = _daily_panel()
    engine = EvaluationEngine("cross_sectional", "daily")
    run = engine.evaluate(panel, n_quantiles=2, quantile_scheme="per_cross_section",
                          return_mode="cross_section_excess", hac_lag=1)
    assert (run.analysis_mode, run.frequency, run.return_mode) == (
        "cross_sectional", "daily", "cross_section_excess",
    )
    full = run.splits["full"]
    assert full.fit_identity is None and full.quantile_scheme == "per_cross_section"
    stats = full.statistics
    daily, weekly = stats.horizons["1D"], stats.horizons["5D"]
    np.testing.assert_allclose(daily.ic.values, [1., .6, -1.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(daily.bucket_returns[0], [-.02, .02], atol=1e-12)
    assert daily.spread[0] == pytest.approx(.04)
    assert daily.hac.lower < daily.ic_summary.mean < daily.hac.upper
    assert daily.direction.denominator == 12
    assert stats.rank_autocorrelation[1] == pytest.approx(1.)
    np.testing.assert_array_equal(stats.turnover[1], [0., 0.])
    np.testing.assert_array_equal(weekly.ic.counts, [4, 0, 0])
    assert np.isnan(weekly.ic_summary.icir)
    assert weekly.ic_summary.reason == "insufficient_valid_cross_sections"
    assert weekly.hac.reason == "insufficient_valid_cross_sections"


def test_runner_keeps_single_ic_as_legacy_ir(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_corr(*args: object, **kwargs: object) -> None:
        pytest.fail("legacy runner must use the panel correlation kernel")

    monkeypatch.setattr(pd.Series, "corr", unexpected_corr)
    _, factor, pricing, returns = synthetic_inputs()
    result = AlphalensRunner(n_quantiles=3, periods=(1, 5)).run(factor, pricing, returns, max_loss=1.)
    single_ic = result["ic"]["5D"].dropna()
    assert len(single_ic) == 1
    assert result["ir"].loc["5D", "ir"] == pytest.approx(single_ic.iloc[0], rel=0, abs=1e-12)


def test_minute_cross_section_uses_signal_buckets_and_legacy_uses_valid_label_buckets() -> None:
    bars = pd.concat([labels([value, value], [value, 5 - value], str(value))
                      for value in range(1, 5)], ignore_index=True)
    bars.loc[0, ["status", "exclusion_reason"]] = ["excluded", "missing_bar"]
    long = bars.set_index(["signal_timestamp", "instrument_id"])
    panel = EvaluationPanel.from_long(pd.DataFrame({
        "factor": long.factor_value, 1: long.return_pct.where(long.status.eq("valid")) / 100,
    }))
    statistics = EvaluationEngine("cross_sectional", "bar").evaluate(
        panel, n_quantiles=2,
    ).splits["full"].statistics
    legacy = BarCrossSectionalEvaluator(analysis_mode="cross_sectional", n_quantiles=2).evaluate(bars)
    result = statistics.horizons[1]
    np.testing.assert_allclose(result.ic.values, legacy.cross_sections.spearman_ic, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result.bucket_counts[0], [1, 2])
    assert statistics.quantiles.labels[0, 0] == 1
    assert legacy.quantile_stats.total_sample_count.tolist() == [4, 3]
    assert legacy.quantile_stats.mean_return_pct.tolist() == pytest.approx([3., 2.75])


@pytest.mark.parametrize("multi_instrument", [False, True])
def test_bar_time_series_full_sample_and_legacy_phase_zero(multi_instrument: bool) -> None:
    a = labels(np.arange(12.), horizon=2)
    a.loc[0, ["status", "exclusion_reason"]] = ["excluded", "missing_bar"]
    bars = a
    if multi_instrument:
        b = labels(np.arange(11.), -np.arange(11.), "B", horizon=2)
        t = pd.Timedelta(minutes=1)
        for column in ["signal_timestamp", "signal_available_at", "entry_timestamp", "exit_timestamp"]:
            b[column] += t
        bars = pd.concat([a, b], ignore_index=True)
    long = bars.set_index(["signal_timestamp", "instrument_id"])
    panel = EvaluationPanel.from_long(pd.DataFrame({
        "factor": long.factor_value, 2: long.return_pct.where(long.status.eq("valid")) / 100,
    }))
    endpoints = {
        field: {2: long[field].unstack().reindex(index=panel.datetimes, columns=panel.assets)
                .to_numpy(dtype="datetime64[ns]")}
        for field in ("entry_timestamp", "exit_timestamp")
    }
    observed = long.status.notna().unstack(fill_value=False).reindex(
        index=panel.datetimes, columns=panel.assets, fill_value=False,
    ).to_numpy()
    statistics = EvaluationEngine("time_series", "bar").evaluate(
        panel, entry_timestamps=endpoints["entry_timestamp"], exit_timestamps=endpoints["exit_timestamp"],
        expected_grid=observed, n_quantiles=2, n_boot=20, min_valid_replicates=16,
    ).splits["full"].statistics
    adapter = TimeSeriesEvaluator(analysis_mode="time_series", n_quantiles=2).evaluate(bars)
    assert statistics.intervals_verified
    rows = adapter.ic_by_horizon.set_index("instrument_id")
    for asset, summary in statistics.horizons[2].instruments.items():
        assert rows.loc[asset, "spearman_ic"] == pytest.approx(summary.ic.values[0], abs=1e-12)
        assert rows.loc[asset, "non_overlapping_count"] == summary.non_overlapping_ic[0].counts[0]
        assert rows.loc[asset, "non_overlapping_ic"] == pytest.approx(summary.non_overlapping_ic[0].values[0])
        assert set(summary.non_overlapping_ic) == {0, 1}
        assert summary.bootstrap.lower == pytest.approx(summary.ic.values[0])
        assert summary.returns.p5 < summary.returns.p95
        assert summary.direction.denominator > 0
    assert rows.loc["A", "non_overlapping_count"] == 5
    assert rows.loc["__ALL__", "mean_spearman_ic"] == pytest.approx(0. if multi_instrument else 1.)
    assert adapter.quantile_stats.descriptive_only.all()


def test_engine_split_freezes_bins_and_purges_at_the_scheduled_exit() -> None:
    times = pd.date_range("2024-01-02", periods=8, freq="1min", tz="UTC")
    index = pd.MultiIndex.from_product([times, ["A"]])
    panel = EvaluationPanel.from_long(pd.DataFrame({
        "factor": [0., 1., 2., 3., 4., 5., -100., 100.],
        2: np.arange(8.) / 100,
    }, index=index))
    entries = (times + pd.Timedelta(minutes=1)).to_numpy(dtype="datetime64[ns]")[:, None]
    exits = (times + pd.Timedelta(minutes=3)).to_numpy(dtype="datetime64[ns]")[:, None]
    run = EvaluationEngine("time_series", "bar").evaluate(
        panel, split=SplitConfig(times[3], times[4]), signal_available_at=times,
        entry_timestamps={2: entries}, exit_timestamps={2: exits}, quantile_scheme="fitted_on_train",
        n_quantiles=2, standardization="zscore", n_boot=8, min_valid_replicates=5,
    )
    train, oos = run.splits["train"], run.splits["oos"]
    assert train.fit_identity == oos.fit_identity and train.fit_identity["sha256"]
    assert train.fit_identity["instruments"][0]["training_edges"] == [0., 1.5, 3.]
    assert train.split_purged_count == {2: 2}
    np.testing.assert_array_equal(train.valid[2][:, 0], [True, True, False, False])
    np.testing.assert_array_equal(oos.statistics.quantiles.labels[:, 0], [2, 2, 1, 2])
    assert not oos.split_purged_count[2]


@pytest.mark.parametrize("mode", MODES)
def test_runner_accepts_object_numeric_returns(mode: str) -> None:
    _, factor, prices, returns = synthetic_inputs()
    expected = run_snapshot(factor, prices, returns, mode=mode)
    runner = AlphalensRunner(n_quantiles=3, periods=(1, 5), _force_raw_quantiles=mode == "fallback")
    result = runner.run(factor, prices, returns.astype(object),
                        max_loss=expected["max_loss"]["fraction"])
    assert {name: _freeze_frame(frame) for name, frame in result.items()} == expected["results"]


def test_runner_preserves_legacy_ir_for_roundoff_only_ic_variation() -> None:
    _, factor, prices, _ = synthetic_inputs()
    factor.iloc[:, :] = np.arange(factor.shape[1])
    sizes = 2 + np.arange(len(factor)) % 2
    values = np.where(np.arange(factor.shape[1]) < sizes[:, None], factor, np.nan)
    index = factor.stack().index
    returns = pd.DataFrame({"1D": values.ravel()}, index=index)
    result = AlphalensRunner(n_quantiles=2, periods=(1,)).run(factor, prices, returns, max_loss=1.)
    expected_ic = pd.Series(
        [pd.Series(np.arange(n)).corr(pd.Series(np.arange(n)), method="spearman") for n in sizes],
    )
    np.testing.assert_array_equal(result["ic"]["1D"], expected_ic)
    assert result["ir"].loc["1D", "ir"] == expected_ic.mean() / expected_ic.std()

    panel = EvaluationPanel.from_long(returns.assign(factor=factor.stack()))
    summary = EvaluationEngine("cross_sectional").evaluate(panel).splits["full"].statistics.horizons["1D"]
    assert np.isnan(summary.ic_summary.icir)
    assert summary.ic_summary.reason == "zero_ic_std"


def test_runner_keeps_constant_legacy_icir_missing_with_tied_ranks() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10, name="date")
    assets = pd.Index(list("ABCDEF"), name="asset")
    factor = pd.DataFrame(
        np.tile([[0, 0, 0, 0, 2, 2], [0, 0, 0, 1, 1, 1]], (5, 1)),
        index=dates, columns=assets,
    )
    returns = pd.DataFrame(
        np.tile([[2, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]], (5, 1)),
        index=dates, columns=assets,
    ).stack().rename("1D").to_frame()
    pricing = pd.DataFrame(100., index=dates, columns=assets).stack()

    result = AlphalensRunner(n_quantiles=2, periods=(1,)).run(factor, pricing, returns, max_loss=0.)
    expected_ic = pd.Series(0.447213595499958, index=dates, name="1D")
    pd.testing.assert_series_equal(result["ic"]["1D"], expected_ic, check_exact=True, check_freq=False)
    assert np.isnan(result["ir"].loc["1D", "ir"])


@pytest.mark.parametrize("cross_sectional", [False, True], ids=["time_series", "cross_sectional"])
def test_bar_adapters_preserve_integer_rank_and_ties(cross_sectional: bool) -> None:
    values = 2**53 + np.array([0, 1, 0, 1], dtype=np.int64)
    if cross_sectional:
        bars = pd.concat([labels([0], [i], str(i)) for i in range(4)], ignore_index=True)
        bars["factor_value"] = values
        result = BarCrossSectionalEvaluator(analysis_mode="cross_sectional", n_quantiles=2).evaluate(bars)
        row = result.cross_sections.iloc[0]
        status = row.cs_status
        assert result.ic_by_horizon.iloc[0].valid_cross_sections == 1
    else:
        bars = labels([0, 1, 2, 3])
        bars["factor_value"] = values
        result = TimeSeriesEvaluator(analysis_mode="time_series", n_quantiles=2).evaluate(bars)
        row = result.ic_by_horizon.iloc[0]
        status = row.ic_status
        assert result.ic_by_horizon.iloc[1].valid_instrument_count == 1
    assert status == "ok"
    assert row.spearman_ic == pytest.approx(1 / np.sqrt(5))
    assert row.quantile_status == "unable_to_form_quantiles"
    assert result.quantile_stats.empty


@pytest.mark.parametrize("frequency,short,long", [("daily", "1D", "5D"), ("bar", 1, 5)])
def test_engine_hac_respects_each_horizon_overlap_floor(
    frequency: str, short: str | int, long: str | int,
) -> None:
    dates = pd.date_range("2024-01-02", periods=8)
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]])
    outcome = [1., 2., 3.] * 4 + [3., 2., 1.] * 4
    panel = EvaluationPanel.from_long(pd.DataFrame({"factor": [1., 2., 3.] * 8,
                                                   short: outcome, long: outcome}, index=index))
    engine = EvaluationEngine("cross_sectional", frequency=frequency)
    for options, short_lag, long_lag in (({}, 0, 4), ({"hac_lag": 1}, 1, 4), ({"hac_lag": 5}, 5, 5)):
        stats = engine.evaluate(panel, **options).splits["full"].statistics
        assert stats.horizons[short].hac.lag == short_lag
        assert stats.horizons[long].hac.lag == long_lag
    horizons = engine.evaluate(panel).splits["full"].statistics.horizons
    assert horizons[short].hac.variance == pytest.approx(.125)
    assert horizons[long].hac.variance == pytest.approx(.25)

    available = dates.tz_localize("UTC")
    exits = {
        key: np.broadcast_to(
            (dates + pd.Timedelta(days=steps)).to_numpy()[:, None], panel.factor.shape,
        )
        for key, steps in ((short, 1), (long, 5))
    }
    splits = engine.evaluate(
        panel, split=SplitConfig(available[3], available[4]), signal_available_at=available,
        exit_timestamps=exits, hac_lag=1,
    ).splits
    assert all(part.statistics.horizons[long].hac.lag == 4 for part in splits.values())
    assert splits["oos"].statistics.horizons[long].hac.reason == "insufficient_hac_grid"


@pytest.mark.parametrize("single_precision", [("1D", "5D"), ("1D",)])
def test_daily_main_preserves_float32_quantile_rounding(single_precision):
    from reference.alphalens_ref.performance import mean_return_by_quantile

    _, factor, pricing, returns = synthetic_inputs()
    returns = returns.astype(dict.fromkeys(single_precision, np.float32))
    runner = AlphalensRunner(n_quantiles=3, periods=(1, 5))
    actual = runner.run(factor, pricing, returns, max_loss=1)
    clean = runner._prepare_factor_data_with_precomputed_returns(
        factor.stack().rename_axis(["date", "asset"]), returns,
    )
    means, errors = mean_return_by_quantile(clean, by_date=True)
    pd.testing.assert_frame_equal(actual["quantile_returns"], means, rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(actual["std_err"], errors, rtol=0, atol=1e-12)


@pytest.mark.parametrize("nullable", [("1D", "5D"), ("1D",)])
def test_daily_nullable_float32_preserves_default_raw_fallback(nullable):
    _, factor, pricing, returns = synthetic_inputs()
    returns = returns.astype(dict.fromkeys(nullable, "Float32"))
    runner = AlphalensRunner(n_quantiles=3, periods=(1, 5))
    actual = runner.run(factor, pricing, returns, max_loss=1)
    clean = runner._prepare_factor_data_with_precomputed_returns(
        factor.stack().rename_axis(["date", "asset"]), returns,
    )
    # The legacy fallback aggregates the original Float32 values as float64.
    raw = clean[["1D", "5D"]].astype(float)
    expected = raw.groupby([raw.index.get_level_values(0), clean.factor_quantile]).mean().unstack()
    expected.columns.names = ["period", "quantile"]
    assert actual["quantile_returns"].shape == (9, 6)
    pd.testing.assert_frame_equal(actual["quantile_returns"], expected.astype(float), rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(actual["std_err"], pd.DataFrame())


def test_daily_float32_demeaning_preserves_ragged_widths_and_missing_values():
    from backtest_and_analysis.alphalens_runner import _demean_float32_by_date

    rng = np.random.default_rng(827)
    widths = [7, 8, 9, 8, 129, 128]
    dates = pd.date_range("2023-01-01", periods=len(widths))
    index = pd.MultiIndex.from_arrays([
        dates.repeat(widths), np.concatenate([np.arange(width) for width in widths]),
    ])
    values = rng.normal(-0.03, 0.08, len(index)).astype(np.float32)
    values[::13] = np.nan
    source = pd.Series(values, index=index).sample(frac=1, random_state=827)
    expected = source.groupby(level=0).transform(lambda day: day - day.mean())
    pd.testing.assert_series_equal(_demean_float32_by_date(source), expected, check_exact=True)
