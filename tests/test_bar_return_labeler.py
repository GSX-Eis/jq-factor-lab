"""SYNTHETIC Bar labels: independent panel series and scheduled clock alignment."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series import (
    BarReturnLabeler, ContractValidationError, DuplicateKeyError, LabelRecord,
    filter_bars_by_view,
)
from utils.session_calendar import SessionCalendar


T = pd.Timestamp('2026-12-01T14:30Z')


def bars(times=None, instrument='NQZ26', frequency='1min'):
    if times is None:
        times = pd.date_range(T, periods=80, freq=frequency)
    return pd.DataFrame(dict(
        instrument_id=instrument, timestamp_utc=times, frequency=frequency,
        open=101., high=107., low=100., close=106., volume=1.,
        source='SYNTHETIC', data_version='SYNTHETIC_NERE668_v1', factor_value=1.,
    ))


@pytest.fixture(scope='module')
def nq():
    return SessionCalendar('NQ', '2026-12-01', '2026-12-02')


@pytest.fixture(scope='module')
def ashare():
    return SessionCalendar('XSHG', '2026-09-29', '2026-09-30')


@pytest.mark.parametrize('offset', [1, 2, 3])
@pytest.mark.parametrize('frequency', ['1min', '5min'])
def test_offsets_preserve_each_horizon(nq, offset, frequency):
    frame = bars(frequency=frequency)
    # 80 five-minute bars run past RTH close but remain within ETH.
    labeler = BarReturnLabeler(nq, entry_offset_bars=offset)
    actual = labeler.label(frame, signal_timestamps=[T])
    assert len(actual) == 4
    assert actual.horizon_bars.tolist() == [1, 5, 15, 60]
    assert actual.columns.tolist() == list(LabelRecord.__dataclass_fields__)
    for row in actual.itertuples():
        assert row.status == 'valid' and row.exclusion_reason is None
        assert row.signal_available_at == T + pd.Timedelta(frequency)
        assert row.entry_timestamp == T + offset * pd.Timedelta(frequency)
        assert row.exit_bar_timestamp == T + (offset + row.horizon_bars - 1) * pd.Timedelta(frequency)
        assert row.exit_timestamp - row.entry_timestamp == row.horizon_bars * pd.Timedelta(frequency)
        assert row.price_change_points == 5
        assert row.return_pct == pytest.approx(500 / 101)


def test_panel_same_signal_has_six_independent_states(nq):
    extended = SessionCalendar('NQ', '2026-12-01', '2026-12-02',
                               rth_reference_calendar='CME Globex Equity')
    signal = pd.Timestamp('2026-12-01T20:57Z')
    times = pd.date_range(signal, periods=8, freq='1min')
    names = ['valid', 'missing_bar', 'factor_missing', 'invalid_price',
             'insufficient_history', 'session_boundary']
    frames = {name: bars(times, name) for name in names}
    frames['missing_bar'] = frames['missing_bar'].drop(index=3)
    frames['factor_missing'].loc[0, 'factor_value'] = np.nan
    frames['invalid_price'].loc[3, 'close'] = 0
    frames['insufficient_history'] = frames['insufficient_history'].iloc[:3]
    calendars = dict.fromkeys(names, extended) | {'session_boundary': nq}
    frame = pd.concat(frames.values(), ignore_index=True).sample(frac=1, random_state=42)
    before = frame.copy(deep=True)
    labeler = BarReturnLabeler(calendars, horizons=5, view='RTH')
    actual = labeler.label(frame, signal_timestamps=[signal]).set_index('instrument_id')
    assert len(actual) == 6 and actual.signal_timestamp.eq(signal).all()
    assert actual.loc['valid', 'status'] == 'valid'
    for name in names[1:]:
        assert actual.loc[name, 'status'] == 'excluded'
        assert actual.loc[name, 'exclusion_reason'] == name
        assert actual.loc[name, ['entry_timestamp', 'exit_bar_timestamp', 'exit_timestamp',
                                 'entry_price', 'exit_price', 'price_change_points', 'return_pct']].isna().all()
    for name in names:
        alone = labeler.label(frames[name], signal_timestamps=[signal]).set_index('instrument_id')
        pd.testing.assert_frame_equal(actual.loc[[name]], alone)
    pd.testing.assert_frame_equal(frame, before)
    assert frame.attrs == before.attrs
    records = labeler.label_records(frame, signal_timestamps=[signal])
    assert len(records) == 6
    assert all(r.entry_price is None and r.entry_timestamp is None
               for r in records if r.status == 'excluded')


def test_cross_market_panel_aligns_lunch_independently(ashare):
    nq = SessionCalendar('NQ', '2026-09-29', '2026-09-30')
    signal = pd.Timestamp('2026-09-30T03:29Z')
    china_times = pd.DatetimeIndex([signal]).append(pd.date_range('2026-09-30T05:00Z', periods=5, freq='1min'))
    us_times = pd.date_range(signal, periods=6, freq='1min')
    frame = pd.concat([bars(china_times, '000001'), bars(us_times)], ignore_index=True)
    actual = BarReturnLabeler({'000001': ashare, 'NQZ26': nq}, horizons=5).label(
        frame, signal_timestamps=[signal]).set_index('instrument_id')
    assert actual.status.eq('valid').all()
    assert actual.loc['000001', 'entry_timestamp'] == pd.Timestamp('2026-09-30T05:00Z')
    assert actual.loc['000001', 'exit_timestamp'] == pd.Timestamp('2026-09-30T05:05Z')
    assert actual.loc['NQZ26', 'entry_timestamp'] == signal + pd.Timedelta('1min')
    assert actual.loc['NQZ26', 'exit_timestamp'] == signal + pd.Timedelta('6min')


@pytest.mark.parametrize('view', ['ETH', 'RTH'])
def test_ashare_close_is_session_boundary_with_next_day_coverage(ashare, view):
    signal = pd.Timestamp('2026-09-29T06:57Z')
    times = pd.date_range(signal, periods=3, freq='1min').append(
        pd.date_range('2026-09-30T01:30Z', periods=6, freq='1min'))
    actual = BarReturnLabeler(ashare, horizons=5, view=view).label_records(
        bars(times, '000001'), signal_timestamps=[signal])
    assert len(actual) == 1 and actual[0].exclusion_reason == 'session_boundary'


def test_frequency_coverage_and_factors_join_are_independent(nq):
    slow = bars(pd.DatetimeIndex([T]), frequency='5min')
    fast = bars().drop(columns='factor_value')
    frame = pd.concat([slow.drop(columns='factor_value'), fast], ignore_index=True)
    factors = fast[['timestamp_utc', 'instrument_id']].assign(factor_value=1.)
    factors = factors.sample(frac=1, random_state=9)
    actual = BarReturnLabeler(nq, horizons=1).label(frame, factors, signal_timestamps=[T]).set_index('frequency')
    assert actual.loc['1min', 'status'] == 'valid'
    assert actual.loc['5min', 'exclusion_reason'] == 'insufficient_history'
    absent = factors.iloc[:0]
    actual = BarReturnLabeler(nq, horizons=1).label(frame, absent, signal_timestamps=[T])
    assert actual.exclusion_reason.eq('factor_missing').all()


@pytest.mark.parametrize('damage,reason', [('gap', 'missing_bar'), ('price', None)])
def test_entry_delay_checks_missing_bars_but_only_held_prices(nq, damage, reason):
    frame = bars()
    if damage == 'gap':
        frame = frame.drop(index=1)
    else:
        frame.loc[1, 'close'] = 0
    actual = BarReturnLabeler(nq, horizons=5, entry_offset_bars=3).label_records(
        frame, signal_timestamps=[T])
    assert actual[0].exclusion_reason == reason
    if reason is None:
        assert actual[0].entry_timestamp == T + pd.Timedelta('3min')


@pytest.mark.parametrize('column,value', [('open', 0), ('high', 100), ('low', 108),
                                          ('close', np.inf), ('close', np.nan)])
def test_any_held_bar_bad_ohlc_is_invalid(nq, column, value):
    frame = bars()
    frame.loc[3, column] = value
    actual = BarReturnLabeler(nq, horizons=5).label_records(frame, signal_timestamps=[T])
    assert actual[0].exclusion_reason == 'invalid_price'


def test_all_candidates_retained_and_no_view_candidates_excluded(nq):
    times = pd.date_range('2026-12-01T14:27Z', periods=9, freq='1min')
    frame = bars(times)
    labeler = BarReturnLabeler(nq, horizons=[1, 5], view='RTH')
    actual = labeler.label(frame)
    assert len(actual) == 12
    assert actual.signal_timestamp.min() == T
    assert set(actual.status) == {'valid', 'excluded'}
    assert actual.signal_timestamp.value_counts().eq(2).all()
    empty = labeler.label(frame, signal_timestamps=times[:3])
    assert empty.empty and list(empty.columns) == list(LabelRecord.__dataclass_fields__)
    assert labeler.label_records(frame, signal_timestamps=[]) == []
    assert labeler.label(frame.iloc[:0]).empty


def test_filtered_bars_preserve_raw_tail_and_reject_different_view_or_calendar(nq):
    times = pd.date_range('2026-12-01T20:57Z', periods=8, freq='1min')
    frame = bars(times)
    selected = filter_bars_by_view(frame, nq, 'RTH')
    labeler = BarReturnLabeler(nq, horizons=5, view='RTH')
    pd.testing.assert_frame_equal(labeler.label(selected), labeler.label(frame))
    assert labeler.label(selected).exclusion_reason.eq('session_boundary').all()
    with pytest.raises(ContractValidationError, match='view'):
        BarReturnLabeler(nq).label(selected)
    changed = SessionCalendar('NQ', '2026-12-01', '2026-12-01')
    with pytest.raises(ContractValidationError, match='calendar'):
        BarReturnLabeler(changed, view='RTH').label(selected)
    del selected.attrs['raw_coverage_by_key']
    with pytest.raises(ContractValidationError, match='raw_coverage'):
        labeler.label(selected)


@pytest.mark.parametrize('kwargs', [
    {'entry_offset_bars': value} for value in [True, np.bool_(True), 0, -1, 1.5, '1', None]
] + [
    {'horizons': value} for value in [[], (), [1, 1], [True], True, False, 0, [-1], [1.5], '5', None]
] + [{'view': 'bad'}, {'boundary_policy': 'cross_session'}, {'calendar': None},
     {'calendar': {'NQZ26': 'NQ'}}])
def test_reject_invalid_constructor(nq, kwargs):
    with pytest.raises(ContractValidationError):
        BarReturnLabeler(**({'calendar': nq} | kwargs))


def test_input_validation_and_missing_mapping(nq):
    with pytest.raises(ContractValidationError, match='NQZ26'):
        BarReturnLabeler({}).label(bars())
    with pytest.raises(ContractValidationError, match='factors'):
        BarReturnLabeler(nq).label(bars().drop(columns='factor_value'))
    with pytest.raises(DuplicateKeyError):
        BarReturnLabeler(nq).label(pd.concat([bars(), bars()]))
    with pytest.raises(ContractValidationError, match='grid'):
        BarReturnLabeler(nq).label(bars(pd.date_range(T + pd.Timedelta('1min'), periods=3, freq='5min'), frequency='5min'))
    for signals in [[T.tz_localize(None)], [pd.NaT], [T.tz_convert('Asia/Shanghai')]]:
        with pytest.raises(ContractValidationError, match='UTC'):
            BarReturnLabeler(nq).label(bars(), signal_timestamps=signals)


def test_calendar_can_be_imported_before_time_series_package():
    result = subprocess.run([sys.executable, '-c',
        'from utils.session_calendar import SessionCalendar; '
        'from backtest_and_analysis.time_series import BarReturnLabeler'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
