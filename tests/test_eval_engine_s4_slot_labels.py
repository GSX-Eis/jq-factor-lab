"""Slot execution labels and compatibility with the historical evaluation inputs."""

from dataclasses import replace
from shutil import copyfile

import duckdb

from market_db_samples import declare_market_db
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.evaluation import UnifiedReturnLabeler
from backtest_and_analysis.evaluation.labels import _LabelAxis, label_daily_returns
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from backtest_and_analysis.time_series import BarReturnLabeler
from futures_roll_samples import warm_market_calendars  # noqa: F401
from slots import build_slot_from_config
from slots.cn_futures import build_cn_futures_slot
from slots.jq_ashare_local import build_jq_ashare_local
from test_slot_declarative import CONFIG, EXAMPLE
from test_slot_futures import _populate
from test_slot_jq_ashare_local import DATES, _prices
from utils.bars.futures_roll import build_roll_schedule
from utils.slots import CalendarSlot, ExecutionProfile, ProviderConfig
from utils.time_axis import BarSchedule, Frequency
from utils.trading_calendar import TradingCalendar


@pytest.mark.parametrize("missing_last_close", [False, True])
def test_jq_slot_returns_match_daily_pit_prices_and_calendar_endpoints(tmp_path, missing_last_close):
    with duckdb.connect() as conn:
        declare_market_db(conn, 'cn_equity')
        _prices(conn, "price_data_daily", DATES)
        if missing_last_close:
            conn.execute("UPDATE price_data_daily SET close=NULL WHERE stock_code='A' AND date=?", [DATES[-1]])
        slot = build_jq_ashare_local(conn, start=DATES[0], end=DATES[-1])
        starts, ends = DATES[0], DATES[-1]
        days = pd.DatetimeIndex(DATES, name="date")
        calendar = TradingCalendar(days, days[0], days[-1])
        close_keys = slot.calendar.bar_grid("1d").timestamp_utc
        adjusted = slot.bar_feed.read_signal_bars(
            ["A", "B"], Frequency.parse("1d"), start_utc=close_keys.iloc[0],
            end_utc=close_keys.iloc[-1], as_of_utc=close_keys.iloc[-1],
        )
        pricing = adjusted.assign(date=pd.to_datetime(adjusted.session_id), asset=adjusted.instrument_id)
        pricing = pricing.set_index(["date", "asset"]).close
        reference = label_daily_returns(pricing, calendar, 0, (2, 1, 3))
        labeler = UnifiedReturnLabeler.from_slot(slot)
        labels = labeler.label(["A", "B"], "1d", start_date=starts, end_date=ends, horizons=(2, 1, 3))
        assert labels.return_unit.unique().tolist() == ["decimal"]
        for horizon in (1, 2, 3):
            rows = labels.loc[(labels.horizon_bars == horizon) & labels.instrument_id.eq("A")]
            expected_dates = reference.scheduled_exit_dates(horizon)
            expected = expected_dates.map(dict(zip(days, close_keys)))
            pd.testing.assert_series_equal(rows.scheduled_exit_timestamp.reset_index(drop=True),
                                           expected.reset_index(drop=True), check_names=False)
            actual = rows.loc[rows.status.eq("valid")]
            valid = reference.endpoints(horizon)
            assert actual.entry_timestamp.tolist() == valid.entry_date.map(dict(zip(days, close_keys))).tolist()
            assert actual.exit_timestamp.tolist() == valid.exit_date.map(dict(zip(days, close_keys))).tolist()
            np.testing.assert_allclose(
                actual[["entry_return_price", "exit_return_price"]].to_numpy(),
                valid[["entry_price", "exit_price"]].to_numpy(), rtol=0, atol=1e-12,
            )

        calculator = ForwardReturnsCalculator(cache_dir=str(tmp_path), slot=slot)
        for periods in ((2, 1, 3), (1, 1, 2)):
            expected = label_daily_returns(pricing, calendar, 0, periods).returns
            from_keys = calculator.calculate_forward_returns(pricing, periods, 0)
            from_bounds = calculator.calculate_or_load_forward_returns(None, ["A", "B"], starts, ends, periods, 0)
            for actual in (from_keys, from_bounds):
                pd.testing.assert_frame_equal(actual, expected, rtol=0, atol=1e-12)
        assert list(tmp_path.iterdir()) == []

        row = labels.loc[labels.instrument_id.eq("A") & labels.signal_session_id.eq(starts)
                         & labels.horizon_bars.eq(1)].iloc[0]
        assert (row.entry_price, row.exit_price) == (102.0, 201.0)
        assert row.forward_return == pytest.approx(100.5 / 25.5 - 1, rel=0, abs=1e-12)


def test_jq_slot_mask_is_opt_in_and_identifies_the_untradable_endpoint():
    with duckdb.connect() as conn:
        declare_market_db(conn, 'cn_equity')
        _prices(conn, "price_data_daily", DATES)
        slot = build_jq_ashare_local(conn, start=DATES[0], end=DATES[-1])
        args = dict(start_date=DATES[0], end_date=DATES[-1], horizons=(1, 2))

        def labels(mask):
            return UnifiedReturnLabeler.from_slot(slot, apply_tradability_mask=mask).label(["A"], "1d", **args)

        unfiltered = labels(False)
        assert unfiltered.status.tolist() == ["valid", "valid", "valid", "excluded", "excluded", "excluded"]
        masked = labels(True)
        assert masked.exclusion_reason.tolist() == [
            "exit_not_tradable", None, "entry_not_tradable",
            "insufficient_history", "insufficient_history", "insufficient_history",
        ]
        assert pd.isna(masked.iloc[0].forward_return)
        assert masked.iloc[0].scheduled_exit_timestamp == unfiltered.iloc[0].exit_timestamp

        conn.execute("UPDATE price_data_daily SET high_limit=close WHERE date=?", [DATES[0]])
        assert labels(True).iloc[1].exclusion_reason == "entry_not_tradable"
        conn.execute("UPDATE price_data_daily SET high_limit=close+1 WHERE date=?", [DATES[0]])
        conn.execute("CREATE TABLE historical_st_daily AS SELECT 'A' AS stock_code, ? AS date, true AS is_st",
                     [DATES[-1]])
        assert labels(True).iloc[1].exclusion_reason == "exit_not_tradable"
        conn.execute("DROP TABLE historical_st_daily")
        conn.execute("UPDATE price_data_daily SET low_limit=close WHERE date=?", [DATES[-1]])
        assert labels(True).iloc[1].exclusion_reason == "exit_not_tradable"


@pytest.mark.parametrize("halt_at_close", [False, True], ids=["regular-close", "halt-at-close"])
def test_slot_mask_checks_the_actual_close_at_a_segment_boundary(tmp_path, halt_at_close):
    copyfile(EXAMPLE / "bars.csv", tmp_path / "bars.csv")
    microstructure = CONFIG.microstructure
    if halt_at_close:
        (tmp_path / "halts.csv").write_text(
            "instrument_id,available_at_utc,tradable\nBTC-USD,2026-01-03T02:00:00Z,false\n",
            encoding="utf-8",
        )
        microstructure = ProviderConfig("csv_files", {"paths": ["halts.csv"]})
    options = CONFIG.calendar.options
    sessions = {**options["custom_sessions"], "segments": [
        {"open": "00:00", "close": "02:00"},
        {"open": "03:00", "close": "05:00"},
    ]}
    calendar = replace(CONFIG.calendar, options={**options, "custom_sessions": sessions})
    profile = replace(CONFIG.execution_profile, exit_adjustment=-1)
    slot = build_slot_from_config(
        replace(CONFIG, calendar=calendar, execution_profile=profile, microstructure=microstructure),
        base_dir=tmp_path,
    )
    args = dict(start_date="2026-01-03", end_date="2026-01-03", horizons=1)
    unmasked = UnifiedReturnLabeler.from_slot(slot).label(["BTC-USD"], "1h", **args).iloc[0]
    assert unmasked.status == "valid"
    assert unmasked.entry_timestamp == pd.Timestamp("2026-01-03T01:00:00Z")
    assert unmasked.exit_timestamp == pd.Timestamp("2026-01-03T02:00:00Z")

    masked = UnifiedReturnLabeler.from_slot(slot, apply_tradability_mask=True)
    row = masked.label(["BTC-USD"], "1h", **args).iloc[0]
    assert row.scheduled_exit_timestamp == unmasked.exit_timestamp
    if halt_at_close:
        assert row.exclusion_reason == "exit_not_tradable"
        assert pd.isna(row.forward_return)
    else:
        assert row.status == "valid"
        assert (row.entry_price, row.exit_price) == (103.0, 106.0)
        assert row.exit_timestamp == unmasked.exit_timestamp
        assert row.forward_return == 106.0 / 103.0 - 1
        opening_at_close = pd.DataFrame({
            "timestamp_utc": [row.exit_timestamp],
            "session_id": [row.signal_session_id],
            "execution_price_field": ["open"],
        })
        assert not slot.microstructure.tradability_mask(opening_at_close, instrument_id="BTC-USD").item()


def test_slot_retains_exit_schedule_beyond_the_price_query(monkeypatch):
    with duckdb.connect() as conn:
        declare_market_db(conn, 'cn_equity')
        _prices(conn, "price_data_daily", DATES)
        slot = build_jq_ashare_local(conn, start="2026-01-05", end="2026-01-09")
        read_execution_bars = slot.bar_feed.read_execution_bars
        bounds = []

        def record_bounds(instruments, frequency, *, start_utc, end_utc):
            bounds.append((start_utc, end_utc))
            return read_execution_bars(instruments, frequency, start_utc=start_utc, end_utc=end_utc)

        monkeypatch.setattr(slot.bar_feed, "read_execution_bars", record_bounds)
        labels = UnifiedReturnLabeler.from_slot(slot).label(
            ["A"], "1d", start_date=DATES[0], end_date=DATES[-1], horizons=(1, 2),
        )
        assert bounds == [(pd.Timestamp(f"{DATES[0]} 07:00Z"), pd.Timestamp(f"{DATES[-1]} 07:00Z"))]
        assert labels.signal_session_id.unique().tolist() == list(DATES)
        scheduled = labels.set_index(["signal_session_id", "horizon_bars"])
        for key in ((DATES[-1], 1), (DATES[1], 2)):
            row = scheduled.loc[key]
            assert row.scheduled_exit_timestamp == pd.Timestamp("2026-01-09 07:00Z")
            assert row.exclusion_reason == "insufficient_history"
            assert pd.isna(row.exit_timestamp)
        assert pd.isna(scheduled.loc[(DATES[-1], 2), "scheduled_exit_timestamp"])


@pytest.mark.parametrize("invalid_factor", [0, None])
def test_jq_slot_rejects_a_corrupt_pit_factor_at_an_observed_bar(invalid_factor):
    with duckdb.connect() as conn:
        declare_market_db(conn, 'cn_equity')
        _prices(conn, "price_data_daily", DATES)
        conn.execute("UPDATE price_data_daily SET factor=? WHERE stock_code='A' AND date=?",
                     [invalid_factor, DATES[1]])
        slot = build_jq_ashare_local(conn, start=DATES[0], end=DATES[-1])
        with pytest.raises(ValueError, match="pit_factor_missing_or_invalid"):
            UnifiedReturnLabeler.from_slot(slot).label(
                ["A"], "1d", start_date=DATES[0], end_date=DATES[-1], horizons=2,
            )


def test_slot_uses_only_the_public_calendar_protocol():
    with duckdb.connect() as conn:
        declare_market_db(conn, 'cn_equity')
        _prices(conn, "price_data_daily", DATES)
        slot = build_jq_ashare_local(conn, start=DATES[0], end=DATES[-1])
        calendar = slot.calendar

        class PublicCalendar:
            calendar_hash = calendar.calendar_hash
            sessions = staticmethod(calendar.sessions)
            session_at = staticmethod(calendar.session_at)
            bar_grid = staticmethod(calendar.bar_grid)

        public = PublicCalendar()
        assert isinstance(public, CalendarSlot)
        args = dict(start_date=DATES[0], end_date=DATES[-1], horizons=1)
        expected = UnifiedReturnLabeler.from_slot(slot).label(["A"], "1d", **args)
        actual = UnifiedReturnLabeler.from_slot(replace(slot, calendar=public)).label(["A"], "1d", **args)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

        frequency = Frequency.parse("45min")
        for strategy in ("segment_anchored", "session_clock"):
            grid = public.bar_grid(frequency, DATES[0], DATES[0], bucket_strategy=strategy)
            axis = _LabelAxis.from_slot_grid(public, grid, frequency, "ETH")
            schedule = BarSchedule(calendar, frequency, bucket_strategy=strategy).for_session(DATES[0])
            pd.testing.assert_index_equal(axis.opens, pd.DatetimeIndex(schedule.start_utc), check_names=False)
            pd.testing.assert_index_equal(axis.ends, pd.DatetimeIndex(schedule.end_utc), check_names=False)


@pytest.mark.usefixtures("warm_market_calendars")
def test_cn_futures_slot_excludes_rolls_and_uses_concrete_execution_prices():
    start, end = "2026-01-05", "2026-01-09"
    with duckdb.connect(config={"threads": 1}) as conn:
        _populate(conn, root="CU", exchange="SHFE", start=start, end=end,
                  expiries=["2026-01", "2026-03"])
        build_roll_schedule(conn, product_roots=["CU"], start="2026-01-06", end=end)
        slot = build_cn_futures_slot(conn, product_root="CU", start=start, end=end, adjustment="ratio_back",
                                     exclude_delivery_month=True)
        profile = replace(slot.microstructure.execution_profile, entry_price_field="close",
                          min_entry_offset_bars=0, exit_adjustment=0)
        labels = UnifiedReturnLabeler.from_slot(slot, profile=profile, apply_tradability_mask=True).label(
            ["CU"], "1d", start_date="2026-01-06", end_date=end, horizons=(1, 2),
        )
        rows = labels.set_index(["signal_session_id", "horizon_bars"])
        across_roll = rows.loc[("2026-01-06", 1)]
        assert across_roll.exclusion_reason == "contract_roll"
        assert pd.isna(across_roll.forward_return)
        assert pd.notna(across_roll.scheduled_exit_timestamp)
        for key, entry, exit in [(("2026-01-07", 1), 125., 127.), (("2026-01-07", 2), 125., 129.)]:
            row = rows.loc[key]
            assert row.status == "valid"
            assert (row.entry_price, row.exit_price) == (entry, exit)
            assert row.forward_return == (exit / entry - 1) * 100
        default = UnifiedReturnLabeler.from_slot(slot).label(
            ["CU"], "1d", start_date="2026-01-06", end_date=end, horizons=1,
        ).iloc[0]
        assert (default.entry_price, default.exit_price) == (124., 125.)
        assert default.forward_return == (125. / 124. - 1) * 100
        delivery = UnifiedReturnLabeler.from_slot(slot, apply_tradability_mask=True).label(
            ["CU2601.SHFE"], "1d", start_date="2026-01-06", end_date=end, horizons=1,
        ).iloc[0]
        assert delivery.exclusion_reason == "entry_not_tradable"


def test_declarative_slot_reuses_bar_window_rules_and_accepts_profile_overrides(tmp_path):
    raw = pd.DataFrame({
        "symbol": "BTC-USD",
        "time": pd.to_datetime(["2026-01-03 01:00Z", "2026-01-03 02:00Z", "2026-01-03 04:00Z"]),
        "open": [100., 103., 107.], "high": [104., 108., 111.], "low": [99., 102., 106.],
        "close": [103., 106., 110.], "volume": [12., 14., 15.],
    })
    raw.to_csv(tmp_path / "bars.csv", index=False)
    slot = build_slot_from_config(CONFIG, base_dir=tmp_path)
    day = "2026-01-03"
    args = dict(start_date=day, end_date=day, horizons=(1, 3))
    instruments = ["BTC-USD"]
    labeler = UnifiedReturnLabeler.from_slot(slot)
    default = labeler.label(instruments, "1h", **args)
    assert default.iloc[0].exclusion_reason == "missing_bar"
    factors = pd.DataFrame({
        "instrument_id": "BTC-USD",
        "timestamp_utc": pd.to_datetime([f"{day} 00:00Z", f"{day} 02:00Z"]),
        "factor_value": [np.nan, 2.],
    })
    explicit = labeler.label(instruments, "1h", start_date=day, end_date=day, horizons=1, factors=factors)
    assert explicit.signal_timestamp.tolist() == factors.timestamp_utc.tolist()
    assert explicit.exclusion_reason.tolist() == ["factor_missing", "insufficient_history"]
    override = ExecutionProfile("close", "open", 1, -1, "decimal", "endpoints_only", "cross_session")
    result = UnifiedReturnLabeler.from_slot(slot, profile=override).label(instruments, "1h", **args)
    assert result.iloc[0].exclusion_reason == "invalid_execution_order"
    across_gap = result.iloc[1]
    assert across_gap.status == "valid"
    assert (across_gap.entry_price, across_gap.exit_price) == (106., 107.)
    assert across_gap.forward_return == 107. / 106. - 1
    assert across_gap.exit_timestamp == pd.Timestamp(f"{day} 03:00Z")
    assert across_gap.scheduled_exit_timestamp == across_gap.exit_timestamp

    legacy_profile = ExecutionProfile("open", "close", 1, -1, "percent", "require_complete_bars", "same_session")
    candidate = UnifiedReturnLabeler.from_slot(slot, profile=legacy_profile, apply_tradability_mask=True)
    compared = candidate.label(instruments, "1h", **args)
    bars = slot.bar_feed.read_execution_bars(instruments, Frequency.parse("1h"),
                                             start_utc=pd.Timestamp(f"{day} 00:00Z"),
                                             end_utc=pd.Timestamp(f"{day} 23:00Z")).assign(factor_value=1.)
    baseline = BarReturnLabeler(slot.calendar, horizons=(1, 3), include_scheduled_exit=True).label(bars)
    compared = compared.drop(columns=["signal_session_id", "return_unit", "entry_return_price", "exit_return_price"]).rename(
        columns={"forward_return": "return_pct"},
    )
    pd.testing.assert_frame_equal(compared, baseline, check_exact=True)
