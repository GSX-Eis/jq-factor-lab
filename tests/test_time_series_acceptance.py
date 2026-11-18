"""Frozen acceptance assertions; only support adapters are wired in later stages."""
from dataclasses import asdict

import pandas as pd
import pytest

from backtest_and_analysis.time_series import DuplicateKeyError
import time_series_acceptance_support as support


CASES = support.load_expected('labels')['cases']
CALENDAR = support.load_expected('calendar')
LABEL_VIEWS = [(case, view) for case in CASES for view in case['records_by_view']]


@pytest.mark.parametrize('sample', CALENDAR['sessions'],
                         ids=lambda row: f"{row['view']}-{row['timestamp'][11:16]}")
def test_session_membership(sample):
    assert support.session_date(pd.Timestamp(sample['timestamp']), sample['view']) == sample['session_date']


@pytest.mark.parametrize('sample', CALENDAR['grids'],
                         ids=lambda row: f"{row['view']}-{row['start'][11:16]}")
def test_expected_bar_grid(sample):
    actual = support.expected_bar_times(pd.Timestamp(sample['start']), pd.Timestamp(sample['end']), sample['view'])
    assert actual == [pd.Timestamp(value) for value in sample['timestamps']]


@pytest.mark.parametrize('view', ['ETH', 'RTH'])
def test_eth_rth_view(view):
    actual = support.select_view(support.FIXTURES / 'SYNTHETIC_view_bars.csv', view)
    assert actual['timestamp_utc'].tolist() == [pd.Timestamp(t) for t in CALENDAR['views'][view]]


@pytest.mark.parametrize('kind', ['bars', 'factors'])
def test_csv_duplicate_keys(kind):
    with pytest.raises(DuplicateKeyError, match='NQZ26'):
        support.read_csv(support.FIXTURES / f'SYNTHETIC_duplicate_{kind}.csv', kind)


@pytest.mark.parametrize('kind, sorted_name', [('bars', 'valid_bars'), ('factors', 'sorted_factors')])
def test_csv_unsorted_matches_sorted(kind, sorted_name):
    sorted_frame = support.read_csv(support.FIXTURES / f'SYNTHETIC_{sorted_name}.csv', kind)
    unsorted_frame = support.read_csv(support.FIXTURES / f'SYNTHETIC_unsorted_{kind}.csv', kind)
    minutes = ['30', '31', '32', '33', '34', '35'] if kind == 'bars' else ['30', '31', '32']
    assert sorted_frame['timestamp_utc'].tolist() == [pd.Timestamp(f'2026-12-01T14:{m}Z') for m in minutes]
    assert sorted_frame['instrument_id'].eq('NQZ26').all()
    assert sorted_frame['source'].eq('SYNTHETIC').all()
    assert sorted_frame['data_version'].eq(support.DATA_VERSION).all()
    if kind == 'bars':
        assert sorted_frame['open'].tolist() == [101] * 6
        assert sorted_frame['close'].tolist() == [106] * 6
    else:
        assert sorted_frame['factor_value'].tolist() == [1] * 3
    pd.testing.assert_frame_equal(unsorted_frame, sorted_frame)


@pytest.mark.parametrize('case, view', LABEL_VIEWS,
                         ids=[f"{case['id']}-{view}" for case, view in LABEL_VIEWS])
def test_label_scenarios(case, view):
    actual = support.label_case(case, view)
    expected = [support.decode_record(row) for row in case['records_by_view'][view]]
    assert len(actual) == len(expected)
    for observed, oracle in zip(actual, expected):
        observed_values, oracle_values = asdict(observed), asdict(oracle)
        if oracle.return_pct is not None:
            assert observed_values.pop('return_pct') == pytest.approx(oracle_values.pop('return_pct'), abs=1e-6, rel=0)
        assert observed_values == oracle_values


def test_unsorted_labels_match_sorted():
    case = next(case for case in CASES if case['id'] == 'valid_five')
    shuffled = dict(case, bars='SYNTHETIC_unsorted_bars.csv')
    assert list(support.label_case(shuffled, 'RTH')) == list(support.label_case(case, 'RTH'))


@pytest.mark.parametrize('view', ['ETH', 'RTH'])
def test_end_to_end_counts(view):
    oracle = support.load_expected('evaluation')
    case = next(case for case in CASES if case['id'] == oracle['case_id'])
    assert support.evaluate_case(case, view) == oracle['views'][view]
