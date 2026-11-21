"""Independent clock boundaries and the frozen daily calendar regression."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from utils.session_calendar import DEFAULT_OVERRIDES, SessionCalendar
from utils.trading_calendar import CalendarCoverageError, TradingCalendar


def ts(value):
    return pd.Timestamp(value, tz='UTC')


@pytest.fixture(scope='module')
def nq():
    return SessionCalendar('NQ', '2026-01-01', '2026-12-31')


@pytest.fixture(scope='module')
def ashare():
    return SessionCalendar()


@pytest.mark.parametrize('day,opening,closing,overnight,maintenance,next_open', [
    ('2026-12-01', '14:30', '21:00', '2026-11-30 23:00', '22:00', '23:00'),
    ('2026-07-07', '13:30', '20:00', '2026-07-06 22:00', '21:00', '22:00'),
    ('2026-03-09', '13:30', '20:00', '2026-03-08 22:00', '21:00', '22:00'),
    ('2026-11-02', '14:30', '21:00', '2026-11-01 23:00', '22:00', '23:00'),
])
def test_dst_rth_eth_boundaries_and_grid(nq, day, opening, closing, overnight, maintenance, next_open):
    a, b = ts(f'{day} {opening}'), ts(f'{day} {closing}')
    assert nq.session_at(a - pd.Timedelta(minutes=1), 'RTH') is None
    assert nq.session_at(a, 'RTH').session_date == day
    assert nq.session_at(b - pd.Timedelta(minutes=1), 'RTH').session_date == day
    assert nq.session_at(b, 'RTH') is None
    assert nq.session_at(ts(overnight)).session_date == day
    close, restart = ts(f'{day} {maintenance}'), ts(f'{day} {next_open}')
    assert nq.session_at(close - pd.Timedelta(minutes=1)).session_date == day
    assert nq.session_at(close) is None
    assert nq.session_at(restart - pd.Timedelta(minutes=1)) is None
    assert nq.session_at(restart).session_date == str(pd.Timestamp(day).date() + pd.Timedelta(days=1))
    for freq, count in [('1min', 390), ('5min', 78)]:
        grid = nq.expected_bar_opens(day, freq, 'RTH')
        assert len(grid) == count
        assert grid.equals(pd.date_range(a, b, freq=freq, inclusive='left'))
    eth = nq.expected_bar_opens(day, '5min')
    assert len(eth) == 276
    assert eth[0] == ts(overnight)
    assert eth[-1] == close - pd.Timedelta(minutes=5)
    assert nq.expected_bar_opens_in_range(close, restart, '1min').empty


@pytest.mark.parametrize('day,opening,closing,count', [
    ('2026-11-27', '14:30', '18:00', 42),
    ('2026-12-24', '14:30', '18:00', 42),
    ('2026-07-02', '13:30', '20:00', 78),
    ('2025-07-03', '13:30', '17:00', 42),
])
def test_reference_early_closes(day, opening, closing, count):
    cal = SessionCalendar('ES', '2025-01-01', '2026-12-31')
    grid = cal.expected_bar_opens(day, '5min', 'RTH')
    assert len(grid) == count
    assert grid[0] == ts(f'{day} {opening}')
    assert grid[-1] == ts(f'{day} {closing}') - pd.Timedelta(minutes=5)


def test_nyse_holiday_intersection(nq):
    assert nq.session_at(ts('2026-11-26 15:00')) is not None
    assert nq.session_at(ts('2026-11-26 15:00'), 'RTH') is None
    assert nq.expected_bar_opens('2026-11-26', '1min', 'RTH').empty
    assert nq.expected_bar_opens('2026-12-25', '1min').empty


def test_xshg_lunch_and_half_open_boundaries(ashare):
    for clock, expected in [('01:29', False), ('01:30', True), ('03:29', True),
                            ('03:30', False), ('04:59', False), ('05:00', True),
                            ('06:59', True), ('07:00', False)]:
        for view in ('ETH', 'RTH'):
            assert (ashare.session_at(ts(f'2026-09-30 {clock}'), view) is not None) == expected
    for freq, count in [('1min', 240), ('5min', 48)]:
        eth = ashare.expected_bar_opens('2026-09-30', freq)
        assert len(eth) == count
        assert eth.equals(ashare.expected_bar_opens('2026-09-30', freq, 'RTH'))
        assert eth.is_monotonic_increasing and eth.is_unique and str(eth.tz) == 'UTC'
    assert ashare.expected_bar_opens_in_range(ts('2026-09-30 03:30'), ts('2026-09-30 05:00'), '1min').empty
    assert ashare.session_at(ts('2026-09-27 02:00')) is None


def test_range_uses_session_anchor_not_query_start(nq):
    grid = nq.expected_bar_opens_in_range(ts('2026-12-01 14:32'), ts('2026-12-01 14:41'), '5min', 'RTH')
    assert grid.equals(pd.DatetimeIndex([ts('2026-12-01 14:35'), ts('2026-12-01 14:40')]))
    overnight = nq.expected_bar_opens_in_range(ts('2026-12-01 21:55'), ts('2026-12-01 23:05'), '5min')
    assert overnight.equals(pd.DatetimeIndex([ts('2026-12-01 21:55'), ts('2026-12-01 23:00')]))


@pytest.mark.parametrize('value', [pd.Timestamp('2026-09-30'), pd.NaT,
                                  pd.Timestamp('2026-09-30', tz='Asia/Shanghai'),
                                  pd.Timestamp('2026-09-30', tz='Europe/London')])
def test_reject_non_utc(ashare, value):
    with pytest.raises(ValueError, match='UTC'):
        ashare.session_at(value)
    with pytest.raises(ValueError, match='UTC'):
        ashare.expected_bar_opens_in_range(value, ts('2026-09-30 07:00'), '1min')


def test_etc_utc_accepted(ashare):
    assert ashare.session_at(pd.Timestamp('2026-09-30 02:00', tz='Etc/UTC')) is not None


@pytest.mark.parametrize('frequency', ['0min', '-1min', 'bad', '', '7min', None, 5])
def test_frequency_validation(ashare, frequency):
    with pytest.raises(ValueError):
        ashare.expected_bar_opens('2026-09-30', frequency)


def test_coverage_including_overnight_edges(ashare):
    for day in ['2016-01-03', '2026-10-01']:
        with pytest.raises(CalendarCoverageError):
            ashare.session_at(ts(f'{day} 02:00'))
        with pytest.raises(CalendarCoverageError):
            ashare.expected_bar_opens(day, '1min')
    cal = SessionCalendar('NQ', '2026-12-01', '2026-12-01')
    assert cal.session_at(ts('2026-11-30 23:00')).session_date == '2026-12-01'
    with pytest.raises(CalendarCoverageError):
        cal.session_at(ts('2026-12-01 23:00'))
    with pytest.raises(CalendarCoverageError):
        ashare.expected_bar_opens_in_range(ts('2026-09-30 07:00'), ts('2026-10-01 00:01'), '1min')


def test_invalid_config_and_view(ashare):
    with pytest.raises(ValueError):
        SessionCalendar('NQ')
    with pytest.raises(ValueError):
        SessionCalendar('XSHG', '2026-09-30', '2016-01-04')
    with pytest.raises(ValueError):
        ashare.session_at(ts('2026-09-30 02:00'), 'bad')
    with pytest.raises(ValueError):
        ashare.expected_bar_opens_in_range(ts('2026-09-30 06:00'), ts('2026-09-30 02:00'), '1min')


def test_override_dates_segments_and_hash(tmp_path):
    base = SessionCalendar('XSHG', '2026-09-25', '2026-09-30')
    spec = json.loads(DEFAULT_OVERRIDES.read_text())
    corrections = spec['calendars']['XSHG']
    corrections['remove_sessions'] = [{'session_date': '2026-09-28', 'source': 'synthetic test closure'}]
    corrections['add_sessions'] = [{'session_date': '2026-09-27', 'source': 'synthetic test opening',
                                    'segments': [['2026-09-27T01:30Z', '2026-09-27T02:30Z']]}]
    corrections['special_sessions'] = [{'session_date': '2026-09-30', 'source': 'synthetic early close',
                                        'segments': [['2026-09-30T01:30Z', '2026-09-30T02:30Z']]}]
    path = tmp_path / 'overrides.json'
    path.write_text(json.dumps(spec))
    changed = SessionCalendar('XSHG', '2026-09-25', '2026-09-30', overrides_path=path)
    assert changed.calendar_hash != base.calendar_hash
    assert changed.session_at(ts('2026-09-28 02:00')) is None
    assert changed.session_at(ts('2026-09-27 02:00')).session_date == '2026-09-27'
    assert len(changed.expected_bar_opens('2026-09-30', '5min')) == 12
    assert changed.expected_bar_opens('2026-09-30', '5min').equals(changed.expected_bar_opens('2026-09-30', '5min', 'RTH'))
    assert len(changed.calendar_hash) == 64
    int(changed.calendar_hash, 16)
    assert changed.calendar_hash == SessionCalendar('XSHG', '2026-09-25', '2026-09-30', overrides_path=path).calendar_hash
    corrections['special_sessions'][0]['source'] = 'updated verification record'
    path.write_text(json.dumps(spec))
    assert changed.calendar_hash != SessionCalendar('XSHG', '2026-09-25', '2026-09-30', overrides_path=path).calendar_hash


def test_daily_dates_and_fingerprint_regression(ashare):
    days = ashare.sessions()
    assert len(days) == 2611
    old = TradingCalendar(days, '2016-01-04', '2026-09-30', 'LOCAL_A_SHARE_XSHG_CROSSCHECKED')
    new = TradingCalendar.from_pmc()
    assert old.fingerprint == new.fingerprint == \
        'efd8f921da9c8ae20aec39c7c3c73502ad470c8051851133e7af17a9ac0703ea'
    assert old.sessions().equals(new.sessions())


def test_reference_override_uses_intersection():
    cal = SessionCalendar('NQ', '2026-12-01', '2026-12-02', rth_reference_calendar='XSHG')
    assert cal.expected_bar_opens('2026-12-01', '5min', 'RTH').equals(
        pd.date_range(ts('2026-12-01 01:30'), periods=24, freq='5min').append(
            pd.date_range(ts('2026-12-01 05:00'), periods=24, freq='5min')))


def test_cme_historical_intraday_pause():
    cal = SessionCalendar('CME_Equity', '2019-12-02', '2019-12-03')
    assert cal.session_at(ts('2019-12-02 21:14')) is not None
    assert cal.session_at(ts('2019-12-02 21:15')) is None
    assert cal.session_at(ts('2019-12-02 21:29')) is None
    assert cal.session_at(ts('2019-12-02 21:30')) is not None
    grid = cal.expected_bar_opens('2019-12-02', '5min')
    assert ts('2019-12-02 21:15') not in grid
    assert ts('2019-12-02 21:30') in grid


@pytest.mark.parametrize('entry', [
    {'session_date': '2026-09-30', 'source': '',
     'segments': [['2026-09-30T01:30Z', '2026-09-30T02:30Z']]},
    {'session_date': '2026-09-30', 'source': 'synthetic',
     'segments': [['2026-09-30T01:30', '2026-09-30T02:30']]},
    {'session_date': '2026-09-30', 'source': 'synthetic',
     'segments': [['2026-09-30T02:30Z', '2026-09-30T01:30Z']]},
    {'session_date': '2026-09-30', 'source': 'synthetic',
     'segments': [['2026-09-30T01:30Z', '2026-09-30T03:30Z'],
                  ['2026-09-30T02:30Z', '2026-09-30T04:30Z']]},
])
def test_reject_invalid_corrections(tmp_path, entry):
    path = tmp_path / 'overrides.json'
    path.write_text(json.dumps({'schema_version': 1, 'calendars': {'XSHG': {'special_sessions': [entry]}}}))
    with pytest.raises(ValueError):
        SessionCalendar('XSHG', '2026-09-30', '2026-09-30', overrides_path=path)
