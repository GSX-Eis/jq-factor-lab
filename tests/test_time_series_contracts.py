"""Synthetic contract checks, independent of market data and research execution."""
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series import (
    EXCLUSION_PRIORITY, Bar, ContractValidationError, DuplicateKeyError,
    ExclusionReason, FactorValue, LabelRecord, ResultIdentity, invalid_price_mask,
    primary_exclusion_reason, primary_exclusion_reasons, sha256_file,
    validate_bar_frame, validate_factor_frame,
)


T = pd.Timestamp('2026-12-01T14:30Z')


def bar(**changes) -> Bar:
    return Bar(**(dict(instrument_id='NQZ26', timestamp_utc=T, frequency='1min',
                      open=101., high=107., low=100., close=106., volume=0.,
                      source='SYNTHETIC', data_version='v1') | changes))


def label(**changes) -> LabelRecord:
    return LabelRecord(**(dict(
        instrument_id='NQZ26', frequency='1min', entry_offset_bars=1, horizon_bars=5,
        signal_timestamp=T, signal_available_at=T + pd.Timedelta('1min'),
        entry_timestamp=T + pd.Timedelta('1min'),
        exit_bar_timestamp=T + pd.Timedelta('5min'), exit_timestamp=T + pd.Timedelta('6min'),
        entry_price=101., exit_price=106., price_change_points=5.,
        return_pct=(106 / 101 - 1) * 100, status='valid', exclusion_reason=None,
    ) | changes))


def bar_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(bar()), asdict(bar(instrument_id='ESZ26')),
                         asdict(bar(timestamp_utc=T + pd.Timedelta('1min')))], index=[9, 2, 5])


def factor_frame() -> pd.DataFrame:
    return pd.DataFrame(dict(timestamp_utc=[T + pd.Timedelta('1min'), T, T],
                             instrument_id=['NQZ26', 'NQZ26', 'ESZ26'],
                             factor_value=[np.nan, 2., -3.]), index=[9, 2, 5])


def identity(**changes) -> ResultIdentity:
    return ResultIdentity(**(dict(config_hash='a' * 64,
                                 input_file_hashes={'b.csv': 'b' * 64, 'a.csv': 'c' * 64},
                                 calendar_hash='d' * 64, code_version='b989fefa',
                                 result_schema_version='time_series.v1') | changes))


def test_bar_end_and_frozen_records():
    assert bar().bar_end_utc == T + pd.Timedelta('1min')
    assert bar(frequency='5min').bar_end_utc == T + pd.Timedelta('5min')
    assert bar().has_valid_prices
    assert bar(open=101, high=101, low=101, close=101).has_valid_prices
    for record in (bar(), FactorValue(T, 'NQZ26', 1.), label(), identity()):
        with pytest.raises(FrozenInstanceError):
            record.instrument_id = 'changed'


def test_native_datetime_preserves_submicrosecond_frequency():
    start = T.to_pydatetime()
    assert bar(timestamp_utc=start, frequency='1ns').bar_end_utc == T + pd.Timedelta('1ns')
    assert label(frequency='1ns', horizon_bars=1, signal_timestamp=start,
                 signal_available_at=T + pd.Timedelta('1ns'),
                 entry_timestamp=T + pd.Timedelta('1ns'),
                 exit_bar_timestamp=T + pd.Timedelta('1ns'),
                 exit_timestamp=T + pd.Timedelta('2ns'))


def test_bar_frame_rejects_nonzero_timezone_named_utc():
    frame = bar_frame()
    frame['timestamp_utc'] = frame.timestamp_utc.dt.tz_convert(timezone(timedelta(hours=1), 'UTC'))
    with pytest.raises(ContractValidationError, match='UTC'):
        validate_bar_frame(frame)


@pytest.mark.parametrize('timestamp', [T.to_pydatetime(), datetime(2026, 12, 1, tzinfo=timezone.utc),
                                      datetime(2026, 12, 1, tzinfo=ZoneInfo('UTC'))])
def test_accept_utc_datetimes(timestamp):
    assert bar(timestamp_utc=timestamp).timestamp_utc == timestamp
    assert FactorValue(timestamp, 'A', 0.).timestamp_utc == timestamp


@pytest.mark.parametrize('timestamp', [T.tz_localize(None), T.tz_convert('America/New_York'),
                                      T.tz_convert('Europe/London'), pd.NaT, None, str(T), 0,
                                      datetime(2026, 12, 1, tzinfo=timezone(timedelta(hours=1), 'UTC'))])
def test_reject_naive_non_utc_missing_and_wrong_timestamp_types(timestamp):
    with pytest.raises(ContractValidationError, match='UTC'):
        bar(timestamp_utc=timestamp)
    with pytest.raises(ContractValidationError, match='UTC'):
        FactorValue(timestamp, 'A', 1.)


@pytest.mark.parametrize('field', ['instrument_id', 'source', 'data_version'])
@pytest.mark.parametrize('value', ['', '  ', None, 123, b'A'])
def test_bar_text_fields(field, value):
    with pytest.raises(ContractValidationError, match=field):
        bar(**{field: value})


@pytest.mark.parametrize('frequency', ['', 'not-a-frequency', '0min', '-1min', 'NaT', '1M', None, 1])
def test_invalid_frequency_scalar_and_frame(frequency):
    with pytest.raises(ContractValidationError, match='frequency'):
        bar(frequency=frequency)
    frame = bar_frame()
    frame['frequency'] = frequency
    with pytest.raises(ContractValidationError, match='frequency'):
        validate_bar_frame(frame)


@pytest.mark.parametrize('value', [-1., np.nan, np.inf, -np.inf, '1', None, True, 1j])
def test_invalid_volume(value):
    with pytest.raises(ContractValidationError, match='volume'):
        bar(volume=value)
    frame = bar_frame()
    frame['volume'] = value
    with pytest.raises(ContractValidationError, match='volume'):
        validate_bar_frame(frame)


@pytest.mark.parametrize('field', ['open', 'high', 'low', 'close'])
@pytest.mark.parametrize('value', [True, '101', None, 1j])
def test_price_type_is_structural(field, value):
    with pytest.raises(ContractValidationError, match=field):
        bar(**{field: value})
    frame = bar_frame()
    frame[field] = value
    with pytest.raises(ContractValidationError, match=field):
        validate_bar_frame(frame)


@pytest.mark.parametrize('changes', [
    {'open': np.nan}, {'high': np.nan}, {'low': np.nan}, {'close': np.nan},
    {'open': 0}, {'high': 0}, {'low': 0}, {'close': -1},
    {'open': np.inf}, {'close': -np.inf}, {'high': 102}, {'low': 102},
    {'low': 110, 'high': 109},
])
def test_price_anomalies_survive_validation_and_are_flagged(changes):
    bad = bar(**changes)
    assert not bad.has_valid_prices
    frame = pd.DataFrame([asdict(bar()), asdict(bad)], index=[7, 3])
    pd.testing.assert_series_equal(invalid_price_mask(frame),
                                   pd.Series([False, True], index=[7, 3], name='invalid_price'))
    validated = validate_bar_frame(frame.assign(instrument_id=['A', 'B']))
    assert len(validated) == 2
    assert invalid_price_mask(validated).tolist() == [False, True]


@pytest.mark.parametrize('value', [0., -1., np.nan, np.float64(1)])
def test_factor_allows_nan_and_signed_finite_values(value):
    result = FactorValue(T, 'A', value)
    assert result.factor_value == value or np.isnan(result.factor_value)


@pytest.mark.parametrize('value', [np.inf, -np.inf, '1', None, True, 1j])
def test_factor_rejects_infinite_and_wrong_types(value):
    with pytest.raises(ContractValidationError, match='factor_value'):
        FactorValue(T, 'A', value)
    frame = factor_frame()
    frame['factor_value'] = value
    with pytest.raises(ContractValidationError, match='factor_value'):
        validate_factor_frame(frame)


@pytest.mark.parametrize('value', ['', None, 1])
def test_factor_rejects_bad_instrument(value):
    with pytest.raises(ContractValidationError, match='instrument_id'):
        FactorValue(T, value, 1.)


@pytest.mark.parametrize('make,validate,key', [
    (bar_frame, validate_bar_frame, ['instrument_id', 'timestamp_utc', 'frequency']),
    (factor_frame, validate_factor_frame, ['timestamp_utc', 'instrument_id']),
])
def test_frame_sort_preserves_values_rows_columns_index_and_input(make, validate, key):
    original = make().assign(extra=['first', 'second', 'third'])
    before = original.copy(deep=True)
    expected = original.sort_values(key, kind='stable')
    pd.testing.assert_frame_equal(validate(original), expected)
    pd.testing.assert_frame_equal(validate(original.sample(frac=1, random_state=17)), expected)
    pd.testing.assert_frame_equal(original, before)
    assert validate(original) is not original


@pytest.mark.parametrize('make,validate', [(bar_frame, validate_bar_frame),
                                         (factor_frame, validate_factor_frame)])
def test_duplicate_keys_report_actual_key(make, validate):
    frame = make()
    frame = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(DuplicateKeyError, match='NQZ26') as error:
        validate(frame)
    assert 'timestamp_utc' in str(error.value)
    assert '2026-12-01' in str(error.value)
    assert isinstance(error.value, ContractValidationError)


def test_bar_frequency_is_part_of_key():
    frame = pd.DataFrame([asdict(bar()), asdict(bar(frequency='5min'))])
    assert len(validate_bar_frame(frame)) == 2


@pytest.mark.parametrize('make,validate', [(bar_frame, validate_bar_frame),
                                         (factor_frame, validate_factor_frame)])
@pytest.mark.parametrize('change', ['missing', 'duplicate_column', 'naive', 'non_utc', 'nat',
                                  'strings', 'empty_id', 'numeric_id', 'bytes_id'])
def test_frame_structure(make, validate, change):
    frame = make()
    if change == 'missing':
        frame = frame.drop(columns='instrument_id')
    elif change == 'duplicate_column':
        frame = pd.concat([frame, frame[['instrument_id']]], axis=1)
    elif change == 'naive':
        frame['timestamp_utc'] = frame.timestamp_utc.dt.tz_localize(None)
    elif change == 'non_utc':
        frame['timestamp_utc'] = frame.timestamp_utc.dt.tz_convert('America/New_York')
    elif change == 'nat':
        frame.loc[frame.index[0], 'timestamp_utc'] = pd.NaT
    elif change == 'strings':
        frame['timestamp_utc'] = frame.timestamp_utc.astype(str)
    elif change == 'empty_id':
        frame.loc[frame.index[0], 'instrument_id'] = ' '
    elif change == 'numeric_id':
        frame.loc[frame.index[0], 'instrument_id'] = 123
    elif change == 'bytes_id':
        frame.loc[frame.index[0], 'instrument_id'] = b'A'
    with pytest.raises(ContractValidationError):
        validate(frame)


@pytest.mark.parametrize('field', ['source', 'data_version'])
@pytest.mark.parametrize('value', ['', None, 1, b'SYNTHETIC'])
def test_bar_frame_metadata(field, value):
    frame = bar_frame()
    frame[field] = value
    with pytest.raises(ContractValidationError, match=field):
        validate_bar_frame(frame)


def test_empty_typed_frames_and_nullable_values():
    assert validate_bar_frame(bar_frame().iloc[:0]).empty
    assert validate_factor_frame(factor_frame().iloc[:0]).empty
    assert invalid_price_mask(bar_frame().iloc[:0]).empty
    frame = bar_frame()
    frame['close'] = pd.array([pd.NA, 106., 106.], dtype='Float64')
    assert invalid_price_mask(frame).tolist() == [True, False, False]
    assert len(validate_bar_frame(frame)) == 3
    factors = factor_frame()
    factors['factor_value'] = factors.factor_value.astype('Float64')
    assert len(validate_factor_frame(factors)) == 3


@pytest.mark.parametrize('validate', [validate_bar_frame, validate_factor_frame, invalid_price_mask,
                                     primary_exclusion_reasons])
def test_dataframe_required(validate):
    with pytest.raises(ContractValidationError, match='DataFrame'):
        validate([])


def test_exclusion_priority_all_32_combinations_scalar_and_vector():
    expected_order = ['factor_missing', 'insufficient_history', 'session_boundary',
                      'missing_bar', 'invalid_price']
    assert [r.value for r in EXCLUSION_PRIORITY] == expected_order
    combinations = list(product([False, True], repeat=5))
    expected = []
    for combination in combinations:
        flags = dict(zip(EXCLUSION_PRIORITY, combination))
        winner = next((r for r in EXCLUSION_PRIORITY if flags[r]), None)
        assert primary_exclusion_reason(flags) is winner
        expected.append(winner)
    frame = pd.DataFrame(combinations, columns=expected_order, index=range(31, -1, -1))
    pd.testing.assert_series_equal(primary_exclusion_reasons(frame),
                                   pd.Series(expected, index=frame.index, name='exclusion_reason'))
    assert primary_exclusion_reason({}) is None
    assert primary_exclusion_reasons(frame.iloc[:0]).empty
    assert primary_exclusion_reasons(frame.rename(columns=dict(zip(expected_order, EXCLUSION_PRIORITY)))).tolist() == expected


@pytest.mark.parametrize('flags', [{'unknown': True}, {ExclusionReason.MISSING_BAR: 1},
                                  {ExclusionReason.MISSING_BAR: None}, []])
def test_bad_scalar_exclusion_flags(flags):
    with pytest.raises(ContractValidationError):
        primary_exclusion_reason(flags)


@pytest.mark.parametrize('change', ['missing', 'null', 'integer'])
def test_bad_vector_exclusion_flags(change):
    frame = pd.DataFrame([[False] * 5], columns=[r.value for r in EXCLUSION_PRIORITY])
    if change == 'missing':
        frame = frame.drop(columns='missing_bar')
    elif change == 'null':
        frame['missing_bar'] = pd.Series([pd.NA], dtype='boolean')
    else:
        frame['missing_bar'] = 1
    with pytest.raises(ContractValidationError):
        primary_exclusion_reasons(frame)


def test_label_percent_units_and_absolute_tolerance():
    record = label()
    assert record.return_pct == pytest.approx(4.950495049504955)
    assert record.price_change_points == 5
    assert label(return_pct=record.return_pct + 0.5e-9)
    assert label(price_change_points=5 + 0.5e-9)
    with pytest.raises(ContractValidationError, match='return_pct'):
        label(return_pct=5 / 101)
    with pytest.raises(ContractValidationError, match='return_pct'):
        label(return_pct=record.return_pct + 2e-9)
    with pytest.raises(ContractValidationError, match='price_change_points'):
        label(price_change_points=5 + 2e-9)
    assert label(entry_price=106., exit_price=101., price_change_points=-5.,
                 return_pct=(101 / 106 - 1) * 100)


@pytest.mark.parametrize('field', ['entry_offset_bars', 'horizon_bars'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True, None, '1'])
def test_label_positive_integer_bar_counts(field, value):
    with pytest.raises(ContractValidationError, match=field):
        label(**{field: value})


@pytest.mark.parametrize('changes', [
    {'status': 'bad'}, {'exclusion_reason': ExclusionReason.MISSING_BAR},
    {'status': 'excluded'}, {'status': 'excluded', 'exclusion_reason': 'missing_bar'},
    {'instrument_id': ''}, {'frequency': '0min'},
    {'signal_available_at': T}, {'entry_timestamp': T},
    {'entry_timestamp': T + pd.Timedelta('6min')},
    {'exit_bar_timestamp': T}, {'exit_timestamp': T + pd.Timedelta('5min')},
    {'exit_timestamp': T},
])
def test_label_status_and_time_invariants(changes):
    with pytest.raises(ContractValidationError):
        label(**changes)


@pytest.mark.parametrize('field', ['signal_timestamp', 'signal_available_at', 'entry_timestamp',
                                  'exit_bar_timestamp', 'exit_timestamp'])
@pytest.mark.parametrize('value', [None, pd.NaT, T.tz_localize(None), T.tz_convert('Asia/Shanghai')])
def test_label_requires_utc_and_valid_record_times(field, value):
    with pytest.raises(ContractValidationError):
        label(**{field: value})


@pytest.mark.parametrize('field', ['entry_price', 'exit_price', 'price_change_points', 'return_pct'])
@pytest.mark.parametrize('value', [None, np.nan, np.inf, -np.inf, '1', True])
def test_valid_label_requires_finite_numbers(field, value):
    with pytest.raises(ContractValidationError, match=field):
        label(**{field: value})


@pytest.mark.parametrize('field', ['entry_price', 'exit_price'])
@pytest.mark.parametrize('value', [0., -1.])
def test_valid_label_requires_positive_prices(field, value):
    with pytest.raises(ContractValidationError, match=field):
        label(**{field: value})


def test_excluded_label_allows_missing_execution_fields_but_checks_supplied_times():
    empty = dict(entry_timestamp=None, exit_bar_timestamp=None, exit_timestamp=None,
                 entry_price=None, exit_price=None, price_change_points=None, return_pct=None,
                 status='excluded', exclusion_reason=ExclusionReason.INSUFFICIENT_HISTORY)
    assert label(**empty).status == 'excluded'
    assert label(status='excluded', exclusion_reason=ExclusionReason.INVALID_PRICE,
                 entry_price=np.nan, exit_price=0., return_pct=None)
    for changes in ({'signal_timestamp': None}, {'signal_available_at': None},
                    {'entry_timestamp': T}, {'exit_bar_timestamp': T},
                    {'exit_timestamp': T.tz_localize(None)}, {'entry_price': 'bad'},
                    {'entry_timestamp': T + pd.Timedelta('3min'),
                     'exit_timestamp': T + pd.Timedelta('2min')}):
        with pytest.raises(ContractValidationError):
            label(**(empty | changes))


def test_sha256_file_bytes_and_missing_file(tmp_path: Path):
    path = tmp_path / 'synthetic.bin'
    content = b'abc\x00\n' * 300000
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()
    assert sha256_file(str(path)) == sha256_file(path)
    path.write_bytes(b'')
    assert sha256_file(path) == hashlib.sha256(b'').hexdigest()
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / 'missing')


def test_identity_hash_is_canonical_and_mapping_immutable():
    record = identity()
    reverse = identity(input_file_hashes={'a.csv': 'c' * 64, 'b.csv': 'b' * 64})
    assert record.identity_hash() == reverse.identity_hash()
    payload = dict(config_hash=record.config_hash, input_file_hashes=dict(record.input_file_hashes),
                   calendar_hash=record.calendar_hash, code_version=record.code_version,
                   result_schema_version=record.result_schema_version)
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert record.identity_hash() == expected
    source = {'a.csv': 'a' * 64}
    copied = identity(input_file_hashes=source)
    before = copied.identity_hash()
    source['a.csv'] = 'f' * 64
    assert copied.identity_hash() == before
    with pytest.raises(TypeError):
        copied.input_file_hashes['a.csv'] = 'f' * 64


@pytest.mark.parametrize('changes', [
    {'config_hash': 'e' * 64}, {'calendar_hash': 'e' * 64}, {'code_version': 'new-code'},
    {'result_schema_version': 'v2'}, {'input_file_hashes': {'a.csv': 'e' * 64}},
    {'input_file_hashes': {'renamed.csv': 'c' * 64, 'b.csv': 'b' * 64}},
])
def test_identity_hash_changes_with_every_field(changes):
    assert identity().identity_hash() != identity(**changes).identity_hash()


@pytest.mark.parametrize('changes', [
    {'config_hash': ''}, {'calendar_hash': 'x' * 64}, {'config_hash': 1},
    {'calendar_hash': 'A' * 64}, {'code_version': ''}, {'result_schema_version': None},
    {'input_file_hashes': []}, {'input_file_hashes': {'': 'a' * 64}},
    {'input_file_hashes': {'a.csv': 'invalid'}}, {'input_file_hashes': {1: 'a' * 64}},
])
def test_identity_field_validation(changes):
    with pytest.raises(ContractValidationError):
        identity(**changes)
