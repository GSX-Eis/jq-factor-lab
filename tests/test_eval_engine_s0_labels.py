"""NERE-792 hand-computed label fixtures using synthetic prices only."""

import duckdb
import pandas as pd
import pytest

from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from backtest_and_analysis.time_series.labels import BarReturnLabeler
from calendar_test_support import synthetic_calendar
from eval_engine_label_test_support import _bars
from futures_roll_samples import futures_sample, warm_market_calendars  # noqa: F401
from utils.continuous_bars import ContinuousBarSource
from utils.futures_roll import build_roll_schedule
from utils.session_calendar import SessionCalendar


def test_daily_close_to_close_and_bar_open_to_close_use_different_labels() -> None:
    days = pd.DatetimeIndex(['2026-03-09', '2026-03-10', '2026-03-11'], name='date')
    times = pd.date_range('2026-03-09 13:30Z', periods=3, freq='1min')
    prices = _bars(times).assign(close=[95., 105., 103.95])
    closes = prices.assign(date=days, asset=prices.instrument_id).set_index(['date', 'asset']).close
    daily = ForwardReturnsCalculator(trading_calendar=synthetic_calendar(days[0], days[-1]))
    signal_key = (pd.Timestamp('2026-03-09'), 'SYNTHETIC')
    entry_key = (pd.Timestamp('2026-03-10'), 'SYNTHETIC')
    exit_key = (pd.Timestamp('2026-03-11'), 'SYNTHETIC')
    assert (closes.loc[entry_key], closes.loc[exit_key]) == (105., 103.95)
    labels = daily.calculate_forward_returns(closes, periods=(1,), execution_lag=1)
    assert labels.index.tolist() == [signal_key]
    assert labels.loc[signal_key, '1D'] == pytest.approx(-0.01)
    same_day_entry = daily.calculate_forward_returns(closes, periods=(1,), execution_lag=0)
    assert same_day_entry.loc[signal_key, '1D'] == pytest.approx(0.10526315789473684)

    # The daily result has no endpoint columns. Perturb each dated close in isolation.
    changed_entry = closes.copy()
    changed_entry.loc[entry_key] = 99.
    assert daily.calculate_forward_returns(changed_entry, (1,), 1).loc[
        signal_key, '1D'] == pytest.approx(0.05)
    changed_exit = closes.copy()
    changed_exit.loc[exit_key] = 126.
    assert daily.calculate_forward_returns(changed_exit, (1,), 1).loc[
        signal_key, '1D'] == pytest.approx(0.2)

    calendar = SessionCalendar('NQ', '2026-03-09', '2026-03-09')
    label, = BarReturnLabeler(calendar, horizons=1, entry_offset_bars=1, view='RTH').label_records(
        prices, signal_timestamps=[times[0]])
    assert label.status == 'valid'
    assert label.signal_timestamp == pd.Timestamp('2026-03-09 13:30Z')
    assert label.signal_available_at == label.entry_timestamp == pd.Timestamp('2026-03-09 13:31Z')
    assert label.exit_bar_timestamp == pd.Timestamp('2026-03-09 13:31Z')
    assert label.exit_timestamp == pd.Timestamp('2026-03-09 13:32Z')
    assert (label.entry_price, label.exit_price) == (100., 105.)
    assert label.return_pct == pytest.approx(5.0)


def test_missing_bar_excludes_only_the_horizon_that_spans_the_gap() -> None:
    times = pd.date_range('2026-03-09 13:30Z', periods=5, freq='1min')
    calendar = SessionCalendar('NQ', '2026-03-09', '2026-03-09')
    labeler = BarReturnLabeler(calendar, horizons=(1, 3), view='RTH')
    complete = _bars(times)
    _, intact = labeler.label_records(complete, signal_timestamps=[times[0]])
    assert intact.status == 'valid'
    assert (intact.entry_timestamp, intact.exit_bar_timestamp, intact.exit_timestamp) == (
        times[1], times[3], times[4])

    # Missing interior slot. There is an observed Bar after the scheduled exit.
    short, long = labeler.label_records(complete.drop(index=2), signal_timestamps=[times[0]])
    assert short.status == 'valid'
    assert (short.horizon_bars, short.entry_timestamp, short.exit_timestamp) == (1, times[1], times[2])
    assert short.return_pct == pytest.approx(5.0)
    assert (long.horizon_bars, long.status, long.exclusion_reason) == (3, 'excluded', 'missing_bar')
    assert long.entry_timestamp is long.exit_bar_timestamp is long.exit_timestamp is None
    assert long.return_pct is None


def test_missing_and_incomplete_signal_bars_have_distinct_candidate_counts() -> None:
    times = pd.date_range('2026-03-09 13:30Z', periods=5, freq='1min')
    calendar = SessionCalendar('NQ', '2026-03-09', '2026-03-09')
    labeler = BarReturnLabeler(calendar, horizons=1, entry_offset_bars=1, view='RTH')
    bars = _bars(times)
    signals = times[1:3]

    # The absent 13:32 signal produces no candidate, but 13:31 needs it for entry.
    preceding, = labeler.label_records(bars.drop(index=2), signal_timestamps=signals)
    assert preceding.signal_timestamp == times[1]
    assert (preceding.status, preceding.exclusion_reason) == ('excluded', 'missing_bar')

    incomplete = bars.assign(is_complete=[True, True, False, True, True])
    preceding, signal = labeler.label_records(incomplete, signal_timestamps=signals)
    assert (preceding.status, preceding.exclusion_reason) == ('excluded', 'missing_bar')
    assert signal.signal_timestamp == times[2]
    assert (signal.status, signal.exclusion_reason) == ('excluded', 'missing_bar')

    incomplete.loc[2, 'factor_value'] = float('nan')
    signal, = labeler.label_records(incomplete, signal_timestamps=[times[2]])
    assert (signal.status, signal.exclusion_reason) == ('excluded', 'factor_missing')


def test_early_close_accepts_last_bar_and_excludes_longer_horizon() -> None:
    calendar = SessionCalendar('ES', '2026-11-27', '2026-11-30')
    before = pd.date_range('2026-11-27 17:45Z', periods=3, freq='5min')
    after = pd.date_range('2026-11-30 14:30Z', periods=3, freq='5min')
    labeler = BarReturnLabeler(calendar, horizons=(2, 3), view='RTH')
    short, long = labeler.label_records(_bars(before.append(after), '5min'), signal_timestamps=[before[0]])
    assert short.status == 'valid'
    assert (short.horizon_bars, short.entry_timestamp, short.exit_bar_timestamp, short.exit_timestamp) == (
        2, pd.Timestamp('2026-11-27 17:50Z'), pd.Timestamp('2026-11-27 17:55Z'),
        pd.Timestamp('2026-11-27 18:00Z'))
    assert short.return_pct == pytest.approx(5.0)
    assert (long.horizon_bars, long.status, long.exclusion_reason) == (3, 'excluded', 'session_boundary')
    assert long.entry_timestamp is long.exit_bar_timestamp is long.exit_timestamp is None
    assert long.return_pct is None


def test_dst_switch_moves_utc_endpoints_without_changing_bar_count() -> None:
    calendar = SessionCalendar('NQ', '2026-03-06', '2026-03-09')
    friday = pd.date_range('2026-03-06 14:30Z', periods=3, freq='1min')
    monday = pd.date_range('2026-03-09 13:30Z', periods=3, freq='1min')
    before, after = BarReturnLabeler(calendar, horizons=2, view='RTH').label_records(
        _bars(friday.append(monday)), signal_timestamps=[friday[0], monday[0]])
    assert (before.entry_timestamp, before.exit_bar_timestamp, before.exit_timestamp) == (
        pd.Timestamp('2026-03-06 14:31Z'), pd.Timestamp('2026-03-06 14:32Z'),
        pd.Timestamp('2026-03-06 14:33Z'))
    assert (after.entry_timestamp, after.exit_bar_timestamp, after.exit_timestamp) == (
        pd.Timestamp('2026-03-09 13:31Z'), pd.Timestamp('2026-03-09 13:32Z'),
        pd.Timestamp('2026-03-09 13:33Z'))
    for label in (before, after):
        assert label.status == 'valid'
        assert label.horizon_bars == 2
        assert label.entry_timestamp.tz_convert('America/New_York').strftime('%H:%M') == '09:31'
        assert label.return_pct == pytest.approx(5.0)


@pytest.mark.slow
@pytest.mark.usefixtures('warm_market_calendars')
def test_futures_roll_uses_raw_prices_from_the_signal_contract() -> None:
    with duckdb.connect(':memory:') as conn:
        futures_sample(conn, end='2026-01-09', oi=[
            (300, 200, 100), (300, 200, 100), (100, 400, 200), (100, 400, 200), (100, 400, 200),
        ])
        build_roll_schedule(conn, product_roots=['CU'], start='2026-01-06', end='2026-01-09')
        source = ContinuousBarSource(conn)
        prices = source.read('CU', '1d', '2026-01-06', '2026-01-09', adjust='none').assign(factor_value=1.)
        # Hold from Jan 7 across the Jan 8 switch in the displayed chain.
        assert prices.contract_id.tolist() == ['CU2602.SHFE', 'CU2602.SHFE', 'CU2603.SHFE', 'CU2603.SHFE']
        assert prices.is_roll_session.tolist() == [False, False, True, False]
        assert (prices.open.iloc[1], prices.close.iloc[3]) == (104., 129.)
        signal = prices.timestamp_utc.iloc[0]
        labeler = BarReturnLabeler(source.get_calendars(prices), horizons=3, boundary_policy='cross_session',
                                  bucket_strategy='session', continuous_source=source)
        label, = labeler.label_records(prices, signal_timestamps=[signal])

    assert label.status == 'valid'
    assert label.horizon_bars == 3
    assert label.entry_timestamp == prices.timestamp_utc.iloc[1]
    assert label.exit_bar_timestamp == prices.timestamp_utc.iloc[3]
    assert label.exit_timestamp == pd.Timestamp('2026-01-09 07:00Z')
    assert (label.entry_price, label.exit_price) == (104., 109.)
    assert label.return_pct == pytest.approx(4.8076923076923075)
