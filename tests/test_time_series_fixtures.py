"""Check the synthetic oracle now, independently of all pending adapters."""
from collections import Counter
from datetime import date
import json
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtest_and_analysis.time_series import (
    DuplicateKeyError, ExclusionReason, invalid_price_mask,
    primary_exclusion_reason, validate_bar_frame, validate_factor_frame,
)
import time_series_acceptance_support as support


CASES = support.load_expected('labels')['cases']
RECORDS = [(case, view, index, raw) for case in CASES
           for view, records in case['records_by_view'].items()
           for index, raw in enumerate(records)]


def fixture_frame(name: str) -> pd.DataFrame:
    frame = pd.read_csv(support.FIXTURES / name)
    frame['timestamp_utc'] = pd.to_datetime(frame['timestamp_utc'], utc=True)
    if 'factor_value' in frame:
        # A header-only CSV cannot infer its numeric dtype.
        frame['factor_value'] = pd.to_numeric(frame['factor_value'])
    return frame


@pytest.mark.parametrize('path', sorted(support.FIXTURES.glob('*.csv')), ids=lambda p: p.name)
def test_csv_synthetic_identity_and_structure(path):
    assert path.name.startswith('SYNTHETIC_')
    frame = fixture_frame(path.name)
    assert {'source', 'data_version', 'timestamp_utc', 'instrument_id'} <= set(frame.columns)
    assert frame['source'].eq('SYNTHETIC').all()
    assert frame['data_version'].eq(support.DATA_VERSION).all()
    assert frame['instrument_id'].eq('NQZ26').all()
    assert frame['timestamp_utc'].dt.date.eq(date(2026, 12, 1)).all()
    validator = validate_bar_frame if 'frequency' in frame else validate_factor_frame
    if 'frequency' in frame:
        assert frame['frequency'].eq('1min').all()
    if 'duplicate' in path.name:
        with pytest.raises(DuplicateKeyError, match='NQZ26'):
            validator(frame)
    else:
        assert len(validator(frame)) == len(frame)


@pytest.mark.parametrize('name', ['labels', 'calendar', 'evaluation'])
def test_expected_json_synthetic_identity(name):
    raw = (support.FIXTURES / f'expected_{name}.json').read_text()
    # Reject non-standard JSON NaN/Infinity; missing values use JSON null.
    payload = json.loads(raw, parse_constant=lambda value: pytest.fail(f'non-JSON {value}'))
    assert payload['source'] == 'SYNTHETIC'
    assert payload['data_version'] == support.DATA_VERSION
    assert 'SYNTHETIC' in (support.FIXTURES / 'README.md').read_text()


@pytest.mark.parametrize('case, view, index, raw', RECORDS,
                         ids=[f"{c['id']}-{v}-{i}" for c, v, i, _ in RECORDS])
def test_expected_label_records_and_priority(case, view, index, raw):
    label = support.decode_record(raw)
    assert label.entry_offset_bars == case['entry_offset_bars'] == 1
    assert label.horizon_bars == case['horizon_bars'] == 5
    assert label.signal_timestamp.isoformat().replace('+00:00', 'Z') in case['signal_timestamps']
    assert label.signal_timestamp.isoformat().endswith('+00:00')
    reasons = case['reasons_by_view'][view][index]
    assert len(reasons) == len(set(reasons))
    primary = primary_exclusion_reason({ExclusionReason(reason): True for reason in reasons})
    assert label.exclusion_reason == primary
    assert (label.status == 'valid') == (not reasons)


def test_valid_five_bar_arithmetic_and_observation_times():
    case = next(case for case in CASES if case['id'] == 'valid_five')
    row = support.decode_record(case['records_by_view']['RTH'][0])
    bars = fixture_frame(case['bars']).set_index('timestamp_utc')
    assert row.signal_timestamp == pd.Timestamp('2026-12-01T14:30Z')
    assert row.signal_available_at == row.entry_timestamp == pd.Timestamp('2026-12-01T14:31Z')
    assert row.exit_bar_timestamp == pd.Timestamp('2026-12-01T14:35Z')
    assert row.exit_timestamp == pd.Timestamp('2026-12-01T14:36Z')
    assert bars.loc[row.entry_timestamp, 'open'] == row.entry_price == 101
    assert bars.loc[row.exit_bar_timestamp, 'close'] == row.exit_price == 106
    assert row.price_change_points == 106 - 101 == 5
    assert row.return_pct == pytest.approx((106 - 101) / 101 * 100, abs=1e-6, rel=0)


@pytest.mark.parametrize('name', ['SYNTHETIC_zero_bars.csv', 'SYNTHETIC_nan_bars.csv'])
def test_price_anomaly_is_inside_holding_window(name):
    frame = fixture_frame(name)
    assert frame.loc[invalid_price_mask(frame), 'timestamp_utc'].tolist() == [pd.Timestamp('2026-12-01T14:33Z')]
    if 'zero' in name:
        assert frame.loc[3, 'close'] == 0
    else:
        assert pd.isna(frame.loc[3, 'close'])


def test_gap_tail_and_factor_evidence():
    gap = fixture_frame('SYNTHETIC_missing_bars.csv')
    assert gap['timestamp_utc'].tolist() == [pd.Timestamp(f'2026-12-01T14:{m}:00Z') for m in ['30', '31', '32', '34', '35']]
    tail = fixture_frame('SYNTHETIC_tail_bars.csv')
    assert tail['timestamp_utc'].max() == pd.Timestamp('2026-12-01T14:33Z')
    assert len(tail) == 4
    assert fixture_frame('SYNTHETIC_nan_factors.csv')['factor_value'].isna().all()
    assert fixture_frame('SYNTHETIC_absent_factors.csv').empty
    assert len([reasons for case in CASES for rows in case['reasons_by_view'].values()
                for reasons in rows if len(reasons) >= 2]) >= 2


@pytest.mark.parametrize('kind, sorted_name', [('bars', 'valid_bars'), ('factors', 'sorted_factors')])
def test_shuffled_inputs_preserve_rows(kind, sorted_name):
    validator = validate_bar_frame if kind == 'bars' else validate_factor_frame
    shuffled = fixture_frame(f'SYNTHETIC_unsorted_{kind}.csv')
    original = fixture_frame(f'SYNTHETIC_{sorted_name}.csv')
    assert not shuffled['timestamp_utc'].is_monotonic_increasing
    pd.testing.assert_frame_equal(validator(shuffled).reset_index(drop=True), validator(original).reset_index(drop=True))


def test_view_oracles_and_end_to_end_counts_agree():
    summary = support.load_expected('evaluation')
    case = next(case for case in CASES if case['id'] == summary['case_id'])
    calendar = support.load_expected('calendar')
    for view in ['ETH', 'RTH']:
        rows = case['records_by_view'][view]
        assert [row['signal_timestamp'] for row in rows] == calendar['views'][view]
        status = Counter(row['status'] for row in rows)
        reasons = Counter(row['exclusion_reason'] for row in rows if row['exclusion_reason'])
        assert dict(status, exclusion_counts=dict(reasons)) == summary['views'][view]
    assert case['records_by_view']['ETH'][0]['signal_timestamp'] == '2026-12-01T14:27:00Z'
    assert case['records_by_view']['ETH'][0]['status'] == 'valid'
    assert '2026-12-01T14:27:00Z' not in calendar['views']['RTH']


def test_calendar_oracle_uses_winter_utc_and_half_open_intervals():
    opening = pd.Timestamp('2026-12-01T14:30Z')
    assert opening.day_name() == 'Tuesday'
    local = opening.tz_convert(ZoneInfo('America/New_York'))
    assert (local.hour, local.minute, local.utcoffset().total_seconds()) == (9, 30, -18000)
    for grid in support.load_expected('calendar')['grids']:
        times = [pd.Timestamp(value) for value in grid['timestamps']]
        assert times == sorted(set(times))
        assert all(pd.Timestamp(grid['start']) <= t < pd.Timestamp(grid['end']) for t in times)
