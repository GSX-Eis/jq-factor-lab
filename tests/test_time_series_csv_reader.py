"""Synthetic CSV boundaries, view selection and scheduled-versus-missing bars."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series.csv_reader import (
    BarAvailability, filter_bars_by_view, read_bar_csv, read_factor_csv,
)
from backtest_and_analysis.time_series import ContractValidationError, DuplicateKeyError
from utils.session_calendar import CalendarCoverageError, SessionCalendar


T = pd.Timestamp('2026-12-01T14:30Z')
FIXTURES = Path(__file__).parent / 'fixtures/time_series'
READERS = {'bars': read_bar_csv, 'factors': read_factor_csv}


def row(kind='bars', **changes):
    values = dict(instrument_id='NQZ26', timestamp_utc=T.isoformat())
    if kind == 'bars':
        values.update(frequency='1min', open=101., high=107., low=100., close=106.,
                      volume=1., source='SYNTHETIC', data_version='v1')
    else:
        values.update(factor_value=1.)
    return values | changes


def csv_path(tmp_path, rows, name='input.csv'):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope='module')
def nq():
    return SessionCalendar('NQ', '2026-11-01', '2026-12-31')


@pytest.fixture(scope='module')
def ashare():
    return SessionCalendar('XSHG', '2026-09-25', '2026-09-30')


@pytest.mark.parametrize('kind', READERS)
@pytest.mark.parametrize('instruments', [['NQZ26'], ['NQZ26', 'ESZ26', '000001']])
def test_read_single_and_multi_instrument_csv(tmp_path, kind, instruments):
    rows = [row(kind, instrument_id=instrument, note='extra') for instrument in instruments]
    actual = READERS[kind](csv_path(tmp_path, rows))
    assert actual.instrument_id.tolist() == sorted(instruments)
    assert actual.note.tolist() == ['extra'] * len(rows)
    assert str(actual.timestamp_utc.dtype) == 'datetime64[ns, UTC]'
    assert actual.timestamp_utc.eq(T).all()
    assert actual.index.equals(pd.RangeIndex(len(rows)))


@pytest.mark.parametrize('kind', READERS)
def test_empty_csv_has_contract_dtypes(tmp_path, kind):
    path = tmp_path / 'empty.csv'
    pd.DataFrame(columns=row(kind)).to_csv(path, index=False)
    actual = READERS[kind](path)
    assert actual.empty and isinstance(actual.index, pd.RangeIndex)
    assert str(actual.timestamp_utc.dtype) == 'datetime64[ns, UTC]'
    numeric = ['open', 'high', 'low', 'close', 'volume'] if kind == 'bars' else ['factor_value']
    for name in numeric:
        assert actual[name].dtype == np.dtype('float64')
    assert pd.api.types.is_string_dtype(actual.instrument_id.dtype)
    frozen = read_factor_csv(FIXTURES / 'SYNTHETIC_absent_factors.csv')
    assert frozen.empty and frozen.factor_value.dtype == np.dtype('float64')


@pytest.mark.parametrize('kind', READERS)
def test_mixed_explicit_utc_formats_and_nanoseconds(tmp_path, kind):
    times = ['2026-12-01T14:30:00Z', '2026-12-01T14:31:00+00:00',
             '2026-12-01 14:32:00+0000', '2026-12-01T14:33:00.000000123Z']
    actual = READERS[kind](csv_path(tmp_path, [row(kind, timestamp_utc=t) for t in times]))
    assert actual.timestamp_utc.tolist() == [pd.Timestamp(t) for t in times]


@pytest.mark.parametrize('kind', READERS)
@pytest.mark.parametrize('timestamp', [
    '2026-12-01T14:30:00', '2026-12-01', '2026-12-01T14:30:00+08:00',
    '2026-12-01T14:30:00-05:00', '', None, 'NaT', 'garbageZ',
    '2026-13-01T14:30:00Z', '2026-12-01T14:30:00-00:00',
])
def test_reject_non_explicit_utc(tmp_path, kind, timestamp):
    with pytest.raises(ContractValidationError, match='timestamp_utc'):
        READERS[kind](csv_path(tmp_path, [row(kind, timestamp_utc=timestamp)]))


@pytest.mark.parametrize('kind', READERS)
def test_missing_columns_and_duplicate_keys(tmp_path, kind):
    missing = row(kind)
    del missing['instrument_id']
    with pytest.raises(ContractValidationError, match='instrument_id'):
        READERS[kind](csv_path(tmp_path, [missing]))
    with pytest.raises(DuplicateKeyError, match='NQZ26') as exc:
        READERS[kind](csv_path(tmp_path, [row(kind), row(kind)]))
    assert '2026-12-01' in str(exc.value)


@pytest.mark.parametrize('kind', READERS)
@pytest.mark.parametrize('field', ['timestamp_utc', 'instrument_id'])
def test_missing_key_columns(tmp_path, kind, field):
    values = row(kind)
    del values[field]
    with pytest.raises(ContractValidationError, match=field):
        READERS[kind](csv_path(tmp_path, [values]))


@pytest.mark.parametrize('kind', READERS)
def test_duplicate_csv_headers_are_rejected(tmp_path, kind):
    path = csv_path(tmp_path, [row(kind)])
    lines = path.read_text().splitlines()
    lines[0] += ',instrument_id'
    lines[1] += ',ESZ26'
    path.write_text('\n  \n' + '\n'.join(lines) + '\n')
    with pytest.raises(ContractValidationError, match='column names must be unique'):
        READERS[kind](path)


@pytest.mark.parametrize('changes', [
    {'frequency': 'bad'}, {'frequency': '0min'}, {'frequency': '-1min'},
    {'volume': -1}, {'volume': np.nan}, {'volume': np.inf},
    {'open': 'bad'}, {'close': True}, {'source': ''}, {'data_version': ''},
])
def test_invalid_bar_fields(tmp_path, changes):
    with pytest.raises(ContractValidationError):
        read_bar_csv(csv_path(tmp_path, [row(**changes)]))


@pytest.mark.parametrize('value', [np.inf, -np.inf, 'bad', True])
def test_invalid_factor_values(tmp_path, value):
    with pytest.raises(ContractValidationError):
        read_factor_csv(csv_path(tmp_path, [row('factors', factor_value=value)]))


@pytest.mark.parametrize('kind', READERS)
def test_unsorted_csv_equals_sorted_with_range_index(tmp_path, kind):
    rows = [row(kind, instrument_id=instrument, timestamp_utc=(T + pd.Timedelta(minutes=i)).isoformat())
            for instrument in ['ESZ26', 'NQZ26'] for i in range(3)]
    a = READERS[kind](csv_path(tmp_path, rows, 'sorted.csv'))
    b = READERS[kind](csv_path(tmp_path, rows[::-1], 'unsorted.csv'))
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs == b.attrs
    assert a.index.equals(pd.RangeIndex(6))


def test_invalid_prices_and_missing_factors_are_retained(tmp_path):
    bars = read_bar_csv(csv_path(tmp_path, [row(close=0), row(close=np.nan, timestamp_utc='2026-12-01T14:31Z')]))
    assert len(bars) == 2 and bars.close.iloc[0] == 0 and pd.isna(bars.close.iloc[1])
    factors = read_factor_csv(csv_path(tmp_path, [row('factors', factor_value=np.nan)]))
    assert len(factors) == 1 and pd.isna(factors.factor_value.iloc[0])


def test_view_keeps_raw_coverage_and_attrs_before_filtering(tmp_path, nq):
    times = ['2026-12-01T14:29Z', '2026-12-01T14:30Z', '2026-12-01T21:05Z']
    raw = read_bar_csv(csv_path(tmp_path, [row(timestamp_utc=t) for t in times]))
    raw.attrs['caller_note'] = 'keep'
    original = raw.copy(deep=True)
    selected = filter_bars_by_view(raw, nq, 'RTH')
    assert selected.timestamp_utc.tolist() == [T]
    assert selected.index.equals(pd.RangeIndex(1))
    assert selected.attrs['caller_note'] == 'keep'
    assert selected.attrs['raw_coverage_start_utc'] == pd.Timestamp(times[0])
    assert selected.attrs['raw_coverage_end_utc'] == pd.Timestamp(times[-1])
    pd.testing.assert_frame_equal(raw, original)
    assert raw.attrs == original.attrs
    repeated = filter_bars_by_view(selected, nq, 'RTH')
    assert repeated.attrs == selected.attrs
    audit = BarAvailability(selected, nq, 'RTH')
    assert audit.status('NQZ26', T, '1min') == 'observed'
    assert audit.status('NQZ26', T + pd.Timedelta('1min'), '1min') == 'missing_bar'
    assert audit.status('NQZ26', pd.Timestamp('2026-12-02T14:30Z'), '1min') == 'insufficient_history'
    assert audit.status('NQZ26', pd.Timestamp(times[-1]), '1min') == 'market_closed'


def test_coverage_is_per_instrument_and_frequency(tmp_path, nq):
    rows = [row(), row(timestamp_utc='2026-12-01T14:35Z'),
            row(instrument_id='ESZ26'), row(frequency='5min')]
    raw = read_bar_csv(csv_path(tmp_path, rows))
    audit = BarAvailability(filter_bars_by_view(raw, nq, 'RTH'), nq, 'RTH')
    assert raw.attrs['raw_coverage_by_instrument']['ESZ26'] == (T, T)
    assert audit.status('NQZ26', T + pd.Timedelta('1min'), '1min') == 'missing_bar'
    assert audit.status('ESZ26', T + pd.Timedelta('1min'), '1min') == 'insufficient_history'
    assert audit.status('NQZ26', T + pd.Timedelta('5min'), '5min') == 'insufficient_history'
    assert audit.status('UNKNOWN', T, '1min') == 'insufficient_history'


@pytest.mark.parametrize('calendar_name,view,closed', [
    ('nq', 'ETH', '2026-12-01T22:00Z'),
    ('nq', 'RTH', '2026-12-01T21:00Z'),
    ('nq', 'RTH', '2026-11-26T15:00Z'),
    ('nq', 'RTH', '2026-11-27T18:00Z'),
    ('ashare', 'ETH', '2026-09-30T03:30Z'),
    ('ashare', 'RTH', '2026-09-30T04:59Z'),
    ('ashare', 'RTH', '2026-09-27T02:00Z'),
])
def test_market_closed_comes_from_calendar(tmp_path, request, calendar_name, view, closed):
    calendar = request.getfixturevalue(calendar_name)
    opening = '2026-09-30T01:30Z' if calendar_name == 'ashare' else T.isoformat()
    raw = read_bar_csv(csv_path(tmp_path, [row(timestamp_utc=opening)]))
    audit = BarAvailability(raw, calendar, view)
    assert audit.status('NQZ26', pd.Timestamp(closed), '1min') == 'market_closed'
    mixed = read_bar_csv(csv_path(tmp_path, [row(timestamp_utc=opening), row(timestamp_utc=closed)]))
    selected = filter_bars_by_view(mixed, calendar, view)
    assert selected.timestamp_utc.tolist() == [pd.Timestamp(opening)]
    assert selected.index.equals(pd.RangeIndex(1))


def test_coverage_without_csv_attrs_and_availability_snapshot(tmp_path, nq):
    raw = read_bar_csv(csv_path(tmp_path, [row(), row(timestamp_utc='2026-12-01T14:32Z')]))
    raw.attrs.clear()
    selected = filter_bars_by_view(raw, nq, 'RTH')
    assert not raw.attrs
    assert selected.attrs['raw_coverage_by_instrument']['NQZ26'] == (T, T + pd.Timedelta('2min'))
    audit = BarAvailability(selected, nq, 'RTH')
    selected.attrs['raw_coverage_by_key'][('NQZ26', '1min')] = (T, T)
    selected.loc[:, 'instrument_id'] = 'CHANGED'
    assert audit.status('NQZ26', T, '1min') == 'observed'
    assert audit.status('NQZ26', T + pd.Timedelta('1min'), '1min') == 'missing_bar'


def test_no_selected_bars_preserves_coverage_and_empty_input(tmp_path, nq):
    raw = read_bar_csv(csv_path(tmp_path, [row(timestamp_utc='2026-12-01T21:05Z')]))
    selected = filter_bars_by_view(raw, nq, 'RTH')
    assert selected.empty and selected.attrs['raw_coverage_end_utc'] == pd.Timestamp('2026-12-01T21:05Z')
    assert BarAvailability(selected, nq, 'RTH').status('NQZ26', T, '1min') == 'missing_bar'
    empty = raw.iloc[:0].copy()
    empty.attrs.clear()
    assert filter_bars_by_view(empty, nq, 'RTH').empty
    assert BarAvailability(empty, nq, 'RTH').status('NQZ26', T, '1min') == 'insufficient_history'


def test_query_and_view_validation(tmp_path, nq):
    raw = read_bar_csv(csv_path(tmp_path, [row()]))
    audit = BarAvailability(raw, nq, 'RTH')
    for timestamp in [T.tz_localize(None), T.tz_convert('Asia/Shanghai'), pd.NaT]:
        with pytest.raises(ContractValidationError, match='UTC'):
            audit.status('NQZ26', timestamp, '1min')
    with pytest.raises(ContractValidationError, match='grid'):
        audit.status('NQZ26', T + pd.Timedelta('1min'), '5min')
    with pytest.raises(ContractValidationError, match='frequency'):
        audit.status('NQZ26', T, '0min')
    with pytest.raises(CalendarCoverageError):
        audit.status('NQZ26', pd.Timestamp('2027-01-04T14:30Z'), '1min')
    for operation in [filter_bars_by_view, BarAvailability]:
        with pytest.raises(ContractValidationError, match='view'):
            operation(raw, nq, 'bad')
        selected = filter_bars_by_view(raw, nq, 'RTH')
        with pytest.raises(ContractValidationError, match='view'):
            operation(selected, nq, 'ETH')
        changed_calendar = SessionCalendar('NQ', '2026-12-01', '2026-12-02')
        with pytest.raises(ContractValidationError, match='calendar'):
            operation(selected, changed_calendar, 'RTH')
