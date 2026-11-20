"""Hand-calculated daily and minute evaluation acceptance on one price path."""

from dataclasses import replace
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from backtest_and_analysis.evaluation import EvaluationEngine
from backtest_and_analysis.evaluation.labels import ExecutionConvention, label_forward_returns
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.split import SplitConfig
from backtest_and_analysis.time_series import BarReturnLabeler
from data_pipeline.loaders import market_bar_importer
from test_time_series_cli_end_to_end import read_outputs, run_cli
from utils.bar_quality import BarDataQualityWarning
from utils.continuous_bars import ContinuousBarSource
from utils.session_calendar import SessionCalendar
from utils.trading_calendar import TradingCalendar


ASSETS = ("NQH24", "NQM24", "NQU24", "NQZ24")
HORIZONS = (1, 2)
LAG = 2
BOOTSTRAP = dict(seed=916, n_boot=9, min_valid_replicates=2)
AVAILABLE_ROWS = 14
TRAIN_ROWS = range(8)
OOS_ROWS = range(8, AVAILABLE_ROWS)
# Net changes for both conventions are C[i + lag + h] - C[i + lag].
MOVES = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 0, 2, -2, 4, -4, 6, 0], dtype=float)


@pytest.fixture
def prices_and_scores():
    rng = np.random.default_rng(916)
    reference = 16000 + rng.integers(0, 400, len(ASSETS)) / 4
    scores = MOVES[:, None] + np.arange(len(ASSETS))
    return reference, reference + scores, scores[:AVAILABLE_ROWS]


def _pairs(prices, rows, horizon, cutoff, *, unit=1):
    pairs = {}
    for i in rows:
        entry, exit = i + LAG, i + LAG + horizon
        if exit < len(prices) and exit <= cutoff:
            pairs[i] = (prices[exit] / prices[entry] - 1) * unit
    return pairs


def _direction(scores, returns):
    active = [(score, value) for score, value in zip(scores, returns, strict=True) if score != 0]
    return (
        sum((score > 0 and value > 0) or (score < 0 and value < 0) for score, value in active),
        len(active),
        sum(value > 0 for _, value in active),
    )


def _assert_direction(result, expected):
    values = result if isinstance(result, dict) else vars(result)
    assert tuple(values[k] for k in ("hits", "denominator", "up_count")) == expected


def _assert_fit(row, asset_index):
    assert row["training_edges"] == [asset_index - 4, asset_index + 3, asset_index + 10]
    assert row["center"] == asset_index + 3
    assert row["scale"] == pytest.approx(np.sqrt(21), abs=1e-12)


def _daily_inputs(prices, scores, horizons=HORIZONS):
    dates = pd.bdate_range("2024-03-07", periods=len(prices))
    available = dates.tz_localize("America/New_York") + pd.DateOffset(hours=16)
    calendar = TradingCalendar(dates, dates[0], dates[-1])
    keys = pd.MultiIndex.from_product([dates, ASSETS], names=["date", "asset"])
    daily = label_forward_returns(
        pd.Series(prices.ravel(), index=keys), convention=ExecutionConvention.CLOSE_TO_CLOSE,
        calendar=calendar, entry_offset=LAG, horizons=horizons,
    )
    signals = pd.Series(scores.ravel(), index=keys[:scores.size], name="factor")
    panel = EvaluationPanel.from_long(signals.to_frame().join(daily.returns))
    closes = pd.Series(available.tz_convert("UTC").tz_localize(None), index=dates)
    shape = panel.factor.shape
    entries, exits = {}, {}
    for h in horizons:
        key = f"{h}D"
        expected = _pairs(prices, range(AVAILABLE_ROWS), h, len(dates) - 1)
        endpoint = daily.endpoints(h)
        assert len(endpoint) == len(expected) * len(ASSETS)
        for i, returns in expected.items():
            for column, asset in enumerate(ASSETS):
                row = endpoint.loc[(dates[i], asset)]
                assert row.entry_date == dates[i + LAG]
                assert row.exit_date == dates[i + LAG + h]
                assert row.entry_price == prices[i + LAG, column]
                assert row.exit_price == prices[i + LAG + h, column]
                assert daily.returns.at[(dates[i], asset), key] == pytest.approx(returns[column], abs=1e-12)
        np.testing.assert_array_equal(panel.valid[key].all(axis=1), np.isin(range(AVAILABLE_ROWS), list(expected)))
        entry = endpoint.entry_date.map(closes).unstack().reindex(index=panel.datetimes, columns=panel.assets)
        entries[key] = entry.to_numpy(dtype="datetime64[ns]")
        scheduled = daily.scheduled_exit_dates(h).map(closes).reindex(panel.datetimes)
        exits[key] = np.broadcast_to(scheduled.to_numpy()[:, None], shape)
    # Friday before and Monday after US DST retain the same local close.
    assert available[1].tz_convert("UTC") == pd.Timestamp("2024-03-08 21:00Z")
    assert available[2].tz_convert("UTC") == pd.Timestamp("2024-03-11 20:00Z")
    return panel, available[:AVAILABLE_ROWS], entries, exits


@pytest.mark.parametrize("mode", ["time_series", "cross_sectional"])
def test_daily_closes_to_engine_result(prices_and_scores, mode):
    _, prices, scores = prices_and_scores
    panel, available, entries, exits = _daily_inputs(prices, scores)
    time_series = mode == "time_series"
    result = EvaluationEngine(mode, "daily").evaluate(
        panel, split=SplitConfig(available[7], available[8]), signal_available_at=available,
        entry_timestamps=entries, exit_timestamps=exits, n_quantiles=2,
        **(BOOTSTRAP | {"standardization": "zscore"} if time_series else {}),
    )
    assert (result.analysis_mode, result.frequency, result.return_mode) == (mode, "daily", "raw")
    train, oos = result.splits["train"], result.splits["oos"]
    assert set(train.statistics.horizons) == ({1, 2} if time_series else {"1D", "2D"})
    if time_series:
        assert train.statistics.intervals_verified and oos.statistics.intervals_verified
        assert train.quantile_scheme == oos.quantile_scheme == "fitted_on_train"
        assert train.fit_identity == oos.fit_identity
        for column, fit in enumerate(train.fit_identity["instruments"]):
            assert fit["asset"] == ASSETS[column]
            _assert_fit(fit, column)
        np.testing.assert_array_equal(oos.statistics.quantiles.labels[:, 0], [1, 1, 1, 2, 1, 2])
    else:
        assert train.fit_identity is oos.fit_identity is None
        assert train.quantile_scheme == oos.quantile_scheme == "per_cross_section"
    for split, rows, cutoff in ((train, TRAIN_ROWS, 8), (oos, OOS_ROWS, 14)):
        for h in HORIZONS:
            key = h if time_series else f"{h}D"
            pairs = _pairs(prices, rows, h, cutoff)
            valid = np.repeat(np.isin(rows, list(pairs))[:, None], len(ASSETS), axis=1)
            np.testing.assert_array_equal(split.valid[key], valid)
            assert split.split_purged_count[key] == ((h + 1) * len(ASSETS) if split is train else 0)
            stats = split.statistics.horizons[key]
            returns = np.array(list(pairs.values()))
            signals = scores[list(pairs)]
            if time_series:
                for col, asset in enumerate(ASSETS):
                    row = stats.instruments[asset]
                    assert row.bootstrap.block_length_method == "ppw2009"
                    assert row.bootstrap.min_block_length == h
                    if h == 2:
                        assert row.bootstrap.block_length == 0
                    assert row.ic.counts[0] == row.returns.n_samples == len(pairs)
                    assert row.ic.values[0] == pytest.approx(spearmanr(signals[:, col], returns[:, col]).statistic)
                    assert row.returns.mean == pytest.approx(returns[:, col].mean(), abs=1e-12)
                    centered = signals[:, col] - (col + 3)
                    _assert_direction(row.direction, _direction(centered, returns[:, col]))
                    assert set(row.non_overlapping_ic) == set(range(h))
                    for offset, phase in row.non_overlapping_ic.items():
                        assert phase.counts[0] == sum((i - rows.start) % h == offset for i in pairs)
                    for bucket, members in ((1, centered <= 0), (2, centered > 0)):
                        assert row.returns_by_bucket[bucket].n_samples == members.sum()
                        _assert_direction(row.direction_by_bucket[bucket],
                                          _direction(centered[members], returns[members, col]))
            else:
                np.testing.assert_array_equal(split.statistics.quantiles.labels, [[1, 1, 2, 2]] * len(rows))
                np.testing.assert_array_equal(stats.ic.counts, valid.sum(axis=1))
                np.testing.assert_allclose(stats.ic.values[valid[:, 0]], [
                    spearmanr(x, y).statistic for x, y in zip(signals, returns, strict=True)
                ], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(stats.bucket_counts, valid[:, [0, 2]] * 2)
                np.testing.assert_allclose(stats.bucket_returns[valid[:, 0]],
                                           returns.reshape(-1, 2, 2).mean(axis=2), atol=1e-12, rtol=0)
                _assert_direction(stats.direction, _direction(signals.ravel(), returns.ravel()))
                for bucket, cols in ((1, slice(0, 2)), (2, slice(2, 4))):
                    _assert_direction(stats.direction_by_bucket[bucket],
                                      _direction(signals[:, cols].ravel(), returns[:, cols].ravel()))


def _minute_bars(times, prices, scores):
    frame = pd.DataFrame(index=pd.MultiIndex.from_product(
        [times, ASSETS], names=["timestamp_utc", "instrument_id"],
    )).reset_index()
    opening, closing = prices[:-1].ravel(), prices[1:].ravel()
    return frame.assign(
        frequency="1min", open=opening, close=closing,
        high=np.maximum(opening, closing) + 0.25, low=np.minimum(opening, closing) - 0.25,
        volume=1, source="SYNTHETIC", data_version="SYNTHETIC_NERE916_SEED916",
        factor_value=scores.ravel(),
    )


@pytest.mark.parametrize("mode", ["time_series", "cross_sectional"])
def test_minute_csv_to_published_evaluation(tmp_path, prices_and_scores, mode):
    reference, prices, scores = prices_and_scores
    start = pd.Timestamp("2024-03-11 13:30Z")
    minute = pd.Timedelta(minutes=1)
    times = pd.date_range(start, periods=AVAILABLE_ROWS, freq="min")
    bars = _minute_bars(times, prices, scores)
    path = tmp_path / "SYNTHETIC_NQ_1min.csv"
    bars.to_csv(path, index=False, lineterminator="\n")
    input_hash = sha256(path.read_bytes()).hexdigest()
    assert input_hash == "208257ec987567c7b4910735c942f21ea119fa7e9924f10a370eae729fae7368"
    ohlc = bars[["open", "high", "low", "close"]].to_numpy()
    assert np.isfinite(ohlc).all() and (ohlc > 0).all()
    np.testing.assert_array_equal(ohlc * 4, np.rint(ohlc * 4))
    assert (bars.high >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars.low <= bars[["open", "close"]].min(axis=1)).all()
    np.testing.assert_array_equal(bars.factor_value, bars.open - np.tile(reference, AVAILABLE_ROWS))

    day = "2024-03-11"
    calendar = SessionCalendar("NQ", day, day)
    config = dict(
        bars_csv=str(path), analysis_mode=mode, calendar="NQ", view="RTH", start_date=day, end_date=day,
        horizons=list(HORIZONS), entry_offset_bars=LAG, n_quantiles=2,
        split=dict(train_end=(times[7] + minute).isoformat(), oos_start=(times[8] + minute).isoformat()),
    )
    time_series = mode == "time_series"
    if time_series:
        config.update(bootstrap=BOOTSTRAP, standardization="zscore")
    payload = run_cli(tmp_path, config)
    labels, _, _, summary = read_outputs(payload)
    report = summary
    assert (report["analysis_mode"], report["frequency"], report["return_unit"]) == (mode, "bar", "percent")
    assert report["result_identity"]["input_file_hashes"] == {str(path): input_hash}
    assert report["calendar_identity"]["calendar_hash"] == calendar.calendar_hash
    assert report["calendar_identity"]["view"] == "RTH"
    assert report["data_identity"]["sources"] == ["SYNTHETIC"]
    assert report["data_identity"]["data_versions"] == ["SYNTHETIC_NERE916_SEED916"]
    assert calendar.expected_bar_opens(day, "1min", "RTH")[:len(times)].equals(times)
    assert len(labels) == AVAILABLE_ROWS * len(HORIZONS) * len(ASSETS)
    assert not labels.duplicated(["instrument_id", "signal_timestamp", "horizon_bars"]).any()
    for row in labels.itertuples():
        signal = pd.Timestamp(row.signal_timestamp)
        i, col = times.get_loc(signal), ASSETS.index(row.instrument_id)
        assert signal.tzinfo is not None
        assert pd.Timestamp(row.signal_available_at) == signal + minute
        assert row.factor_value == prices[i, col] - reference[col]
        assert row.entry_offset_bars == LAG
        entry, exit = i + LAG, i + LAG + row.horizon_bars
        if exit >= len(prices):
            assert row.status == "excluded" and row.exclusion_reason == "insufficient_history"
            assert pd.isna(row.entry_timestamp) and pd.isna(row.exit_timestamp) and pd.isna(row.return_pct)
        else:
            assert row.status == "valid" and pd.isna(row.exclusion_reason)
            assert pd.Timestamp(row.entry_timestamp) == times[entry]
            assert pd.Timestamp(row.exit_bar_timestamp) == times[exit - 1]
            assert pd.Timestamp(row.exit_timestamp) == times[exit - 1] + minute
            assert (row.entry_price, row.exit_price) == (prices[entry, col], prices[exit, col])
            expected_return = (prices[exit, col] / prices[entry, col] - 1) * 100
            assert row.return_pct == pytest.approx(expected_return, abs=1e-12)

    train, oos = (report["splits"][name] for name in ("train", "oos"))
    assert train["fit_identity"] == oos["fit_identity"]
    if time_series:
        assert train["quantile_scheme"] == oos["quantile_scheme"] == "fitted_on_train"
        for col, asset in enumerate(ASSETS):
            fit, = train["fit_identity"]["1min"][asset]["instruments"]
            assert fit["asset"] == asset
            _assert_fit(fit, col)
    else:
        assert train["fit_identity"] is None
        assert train["quantile_scheme"] == oos["quantile_scheme"] == "per_cross_section"

    for split, rows, cutoff in ((train, TRAIN_ROWS, 9), (oos, OOS_ROWS, 14)):
        for h in HORIZONS:
            pairs = _pairs(prices, rows, h, cutoff, unit=100)
            returns = np.array(list(pairs.values()))
            signals = scores[list(pairs)]
            stats = [row for row in split["horizons"] if row["horizon_bars"] == h]
            assert split["split_purged_count"][str(h)] == (h * len(ASSETS) if split is train else 0)
            if time_series:
                assert {row["instrument_id"] for row in stats} == set(ASSETS)
                for row in stats:
                    col = ASSETS.index(row["instrument_id"])
                    assert row["intervals_verified"]
                    assert row["bootstrap"]["block_length_method"] == "ppw2009"
                    assert row["bootstrap"]["min_block_length"] == h
                    assert row["ic"]["sample_count"] == row["return_distribution"]["n_samples"] == len(pairs)
                    assert row["ic"]["spearman_ic"] == pytest.approx(spearmanr(signals[:, col], returns[:, col]).statistic)
                    assert row["return_distribution"]["mean"] == pytest.approx(returns[:, col].mean(), abs=1e-12)
                    centered = signals[:, col] - (col + 3)
                    _assert_direction(row["direction"], _direction(centered, returns[:, col]))
                    assert row["non_overlapping_ic"].keys() == {str(i) for i in range(h)}
                    for offset, phase in row["non_overlapping_ic"].items():
                        assert phase["sample_count"] == sum((i - rows.start) % h == int(offset) for i in pairs)
                    for bucket, members in ((1, centered <= 0), (2, centered > 0)):
                        data = row["buckets"][str(bucket)]
                        assert data["return_distribution"]["n_samples"] == members.sum()
                        _assert_direction(data["direction"], _direction(centered[members], returns[members, col]))
            else:
                row, = stats
                assert row["ic"]["n_valid"] == len(pairs)
                assert row["ic"]["mean"] == pytest.approx(np.mean([
                    spearmanr(x, y).statistic for x, y in zip(signals, returns, strict=True)
                ]))
                hac = row["hac"]
                assert hac["bandwidth_method"] == "nw94_prewhitened"
                assert hac["min_lag"] == h - 1
                if hac["reason"] is None:
                    assert h - 1 <= hac["lag"] <= (hac["grid_size"] - 1) // 4
                _assert_direction(row["direction"], _direction(signals.ravel(), returns.ravel()))
                for bucket, cols in ((1, slice(0, 2)), (2, slice(2, 4))):
                    data = row["buckets"][str(bucket)]
                    assert data["n_times"] == len(pairs)
                    assert data["mean"] == pytest.approx(returns[:, cols].mean(), abs=1e-12)
                    _assert_direction(row["direction_by_bucket"][str(bucket)],
                                      _direction(signals[:, cols].ravel(), returns[:, cols].ravel()))


@pytest.mark.parametrize("columns", [("1D", "2d"), (1, 2), (np.int64(1), "2D")])
def test_daily_time_series_full_sample_aligns_horizon_aliases(prices_and_scores, columns):
    _, prices, scores = prices_and_scores
    panel, _, entries, exits = _daily_inputs(prices, scores)
    panel = EvaluationPanel.from_long(panel.to_long().rename(columns=dict(zip(("1D", "2D"), columns))))
    first_mask = panel.valid[columns[0]].copy()
    first_mask[1, 0] = False
    panel = replace(panel, valid={1: first_mask, columns[1]: panel.valid[columns[1]]})
    # The public daily path matches equivalent keys across labels and endpoints.
    full = EvaluationEngine("time_series").evaluate(
        panel, entry_timestamps={1: entries["1D"], 2: entries["2D"]},
        exit_timestamps=exits, **BOOTSTRAP,
    ).splits["full"]
    assert full.statistics.intervals_verified
    assert full.valid.keys() == full.split_purged_count.keys() == full.statistics.horizons.keys() == {1, 2}
    np.testing.assert_array_equal(full.valid[1], first_mask)
    np.testing.assert_array_equal(full.valid[2], panel.valid[columns[1]])
    for h, count in ((1, 11), (2, 11)):
        row = full.statistics.horizons[h].instruments[ASSETS[0]]
        assert row.bootstrap.min_block_length == h
        assert h <= row.bootstrap.block_length <= count // 4
        assert row.ic.counts[0] == count
    assert panel.to_long().columns.tolist() == ["factor", *columns]


@pytest.mark.parametrize("invalid", ["0D", "-1D", "1.5D", "5min", "5", 0, True])
def test_daily_time_series_rejects_invalid_horizon_keys(prices_and_scores, invalid):
    _, prices, scores = prices_and_scores
    panel, _, _, _ = _daily_inputs(prices, scores)
    panel = EvaluationPanel.from_long(panel.to_long().rename(columns={"2D": invalid}))
    with pytest.raises(ValueError, match=r"horizon .* must be an integer >= 1"):
        EvaluationEngine("time_series").evaluate(panel)


@pytest.mark.parametrize("where", ["returns", "valid", "entries", "exits"])
def test_daily_time_series_rejects_duplicate_horizon_aliases(prices_and_scores, where):
    _, prices, scores = prices_and_scores
    panel, _, entries, exits = _daily_inputs(prices, scores)
    values = {"returns": panel.returns, "valid": panel.valid, "entries": entries, "exits": exits}
    values[where] = values[where] | {1: values[where]["1D"]}
    panel = replace(panel, returns=values["returns"], valid=values["valid"])
    with pytest.raises(ValueError, match="Duplicate daily horizon"):
        EvaluationEngine("time_series").evaluate(
            panel, entry_timestamps=values["entries"], exit_timestamps=values["exits"],
        )


def test_daily_time_series_rejects_mismatched_endpoint_horizons(prices_and_scores):
    _, prices, scores = prices_and_scores
    panel, _, entries, exits = _daily_inputs(prices, scores)
    with pytest.raises(ValueError, match="same horizons"):
        EvaluationEngine("time_series").evaluate(
            panel, entry_timestamps=entries, exit_timestamps={"1D": exits["1D"], "5D": exits["2D"]},
        )

    with pytest.raises(ValueError, match="Valid masks must contain the same horizons"):
        EvaluationEngine("time_series").evaluate(replace(panel, valid={"1D": panel.valid["1D"]}))


def test_daily_time_series_uses_each_native_horizon_for_adaptive_block(prices_and_scores):
    _, prices, scores = prices_and_scores
    panel, _, entries, exits = _daily_inputs(prices, scores, horizons=(1, 5))
    full = EvaluationEngine("time_series").evaluate(
        panel, entry_timestamps=entries, exit_timestamps=exits, **BOOTSTRAP,
    ).splits["full"]
    assert full.statistics.horizons.keys() == {1, 5}
    for h, pair_count in ((1, 12), (5, 8)):
        row = full.statistics.horizons[h].instruments[ASSETS[0]]
        assert row.bootstrap.min_block_length == h
        if h == 1:
            assert h <= row.bootstrap.block_length <= pair_count // 4
        else:
            assert row.bootstrap.block_length == 0
            assert row.bootstrap.reason == "insufficient_bootstrap_grid"
        assert row.ic.counts[0] == pair_count
        assert [x.counts[0] for x in row.non_overlapping_ic.values()] == [
            len(range(offset, pair_count, h)) for offset in range(h)
        ]


def _dst_boundary():
    minute = pd.Timedelta(minutes=1)
    friday = pd.date_range("2024-03-08 21:57Z", periods=3, freq="min")
    sunday = pd.date_range("2024-03-10 22:00Z", periods=6, freq="min")
    return friday.append(sunday), minute


def test_synthetic_nq_csv_distinguishes_closed_session_and_missing_minute(tmp_path, prices_and_scores):
    _, prices, scores = prices_and_scores
    times, minute = _dst_boundary()
    friday_end, sunday_open, gap = times[2] + minute, times[3], times[5]
    calendar = SessionCalendar("NQ", "2024-03-08", "2024-03-11")
    assert calendar.session_on("2024-03-08").segments[0][0] == pd.Timestamp("2024-03-07 23:00Z")
    assert calendar.session_on("2024-03-11").segments[0][0] == sunday_open
    assert calendar.session_at(times[2]).session_date == "2024-03-08"
    assert calendar.session_at(friday_end) is None
    assert calendar.session_at(sunday_open - minute) is None
    assert calendar.session_at(sunday_open).session_date == "2024-03-11"
    assert calendar.session_at(gap).session_date == "2024-03-11"
    assert calendar.expected_bar_opens_in_range(friday_end, sunday_open - minute, "1min").empty

    bars = _minute_bars(times, prices[:len(times) + 1], scores[:len(times)])
    path = tmp_path / "SYNTHETIC_NQ_DST_MISSING.csv"
    bars.loc[bars.timestamp_utc.ne(gap) & bars.instrument_id.eq(ASSETS[0])].to_csv(path, index=False)
    payload = run_cli(tmp_path, dict(
        bars_csv=str(path), analysis_mode="time_series", calendar="NQ", view="ETH",
        start_date="2024-03-08", end_date="2024-03-11", horizons=[1, 2], n_quantiles=2,
        bootstrap=BOOTSTRAP,
    ))
    labels, _, _, summary = read_outputs(payload)
    assert summary["calendar_identity"]["calendar_hash"] == calendar.calendar_hash
    assert summary["data_identity"]["sources"] == ["SYNTHETIC"]
    timestamps = pd.DatetimeIndex(pd.to_datetime(labels.signal_timestamp, utc=True))
    assert timestamps.unique().equals(times.delete(5))
    assert pd.DatetimeIndex(pd.to_datetime(labels.signal_available_at, utc=True)).equals(timestamps + minute)
    rows = labels.set_index([timestamps, "horizon_bars"])
    for i, h, reason in [(1, 2, "session_boundary"), (2, 1, "session_boundary"),
                         (3, 2, "missing_bar"), (4, 1, "missing_bar"), (8, 1, "insufficient_history")]:
        label = rows.loc[(times[i], h)]
        assert label.status == "excluded" and label.exclusion_reason == reason
        assert pd.isna(label.entry_timestamp) and pd.isna(label.exit_timestamp) and pd.isna(label.return_pct)
    label = rows.loc[(sunday_open, 1)]
    assert label.status == "valid"
    assert pd.Timestamp(label.entry_timestamp) == times[4]
    assert pd.Timestamp(label.exit_timestamp) == gap
    assert (label.entry_price, label.exit_price) == (prices[4, 0], prices[5, 0])
    assert label.return_pct == pytest.approx((prices[5, 0] / prices[4, 0] - 1) * 100, abs=1e-12)


def test_synthetic_nq_continuous_roll_keeps_origin_contract(tmp_path, monkeypatch, prices_and_scores):
    # Register a fixture-only CME product and predeclare the switch at the next Session.
    calendar_id = "CME Globex Equity"
    calendar_map = tmp_path / "synthetic_product_calendars.json"
    calendar_map.write_text(json.dumps({"NQ": {"calendar_id": calendar_id, "rth_reference_calendar": "NYSE"}}))
    monkeypatch.setattr(market_bar_importer, "futures_calendar",
                        partial(market_bar_importer.futures_calendar, path=calendar_map))
    times, minute = _dst_boundary()
    friday_close_bar, sunday_open = times[2], times[3]
    _, prices, scores = prices_and_scores
    bars = _minute_bars(times, prices[:len(times) + 1], scores[:len(times)])
    bars = bars.loc[bars.instrument_id.isin(ASSETS[:2])].copy()
    # A distinct new-contract price level makes any stitching error visible.
    bars.loc[bars.instrument_id.eq(ASSETS[1]), ["open", "high", "low", "close"]] += 200
    meta = pd.DataFrame({
        "instrument_id": ASSETS[:2], "asset_class": "futures", "exchange": "CME",
        "timezone": "America/Chicago", "calendar_id": calendar_id, "rth_reference_calendar": "NYSE",
        "product_root": "NQ", "contract_expiry": ["2024-03-15", "2024-06-21"],
        "contract_month": ["2024-03", "2024-06"], "price_adjustment": "none", "point_value": 20.,
    })
    with duckdb.connect(":memory:", config={"threads": 1}) as connection:
        market_bar_importer.import_market_bars(connection, bars, meta, legacy_compatibility=False)
        connection.execute("""
            CREATE TABLE futures_roll_schedule AS SELECT * FROM (VALUES
                ('NQ', '2024-03-08', '2024-03-07', NULL, 'NQH24', 'synthetic-v1'),
                ('NQ', '2024-03-11', '2024-03-08', 'NQH24', 'NQM24', 'synthetic-v1')
            ) AS plan(product_root, effective_session_id, decision_session_id, from_contract, to_contract, build_id)
        """)
        source = ContinuousBarSource(connection)
        with pytest.warns(BarDataQualityWarning, match="incomplete_buckets="):
            continuous = source.read("NQ", "1min", "2024-03-08", "2024-03-11", adjust="none")
        assert pd.DatetimeIndex(continuous.timestamp_utc).equals(times)
        assert continuous.contract_id.tolist() == [ASSETS[0]] * 3 + [ASSETS[1]] * 6
        assert continuous.is_roll_session.tolist() == [False] * 3 + [True] * 6
        continuous["factor_value"] = 1.
        calendars = source.get_calendars(continuous)
        with pytest.raises(ValueError, match="requires continuous_source"):
            BarReturnLabeler(calendars, boundary_policy="cross_session", horizons=[2])
        for policy in ("same_session", "cross_session"):
            labeler = BarReturnLabeler(calendars, boundary_policy=policy, horizons=[2], continuous_source=source)
            with pytest.warns(BarDataQualityWarning, match="incomplete_buckets="):
                result = labeler.label(continuous, signal_timestamps=[friday_close_bar, sunday_open])
            old, new = list(result.itertuples())
            assert old.signal_available_at == friday_close_bar + minute
            assert new.signal_available_at == sunday_open + minute
            assert new.status == "valid"
            assert (new.entry_price, new.exit_price) == (prices[4, 1] + 200, prices[6, 1] + 200)
            if policy == "same_session":
                assert old.status == "excluded" and old.exclusion_reason == "session_boundary"
                assert pd.isna(old.return_pct)
            else:
                assert old.status == "valid"
                assert (old.entry_timestamp, old.exit_timestamp) == (sunday_open, sunday_open + 2 * minute)
                assert (old.entry_price, old.exit_price) == (prices[3, 0], prices[5, 0])
                assert old.return_pct == pytest.approx((prices[5, 0] / prices[3, 0] - 1) * 100, abs=1e-12)
        connection.execute("DELETE FROM market_bars_1min WHERE instrument_id=? AND timestamp_utc=?",
                           [ASSETS[0], sunday_open + minute])
        with pytest.warns(BarDataQualityWarning, match="incomplete_buckets="):
            missing, = labeler.label(continuous, signal_timestamps=[friday_close_bar]).itertuples()
        assert missing.status == "excluded" and missing.exclusion_reason == "missing_bar"
        assert pd.isna(missing.return_pct)
