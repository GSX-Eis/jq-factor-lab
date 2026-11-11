"""Calendar warm-up must preserve constructor dates and ordinary factor plans."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.factor_runtime import compute_factor_batch


def _runner(inlined):
    if not inlined:
        return compute_factor_batch
    source = (Path(__file__).resolve().parents[1] / 'utils/factor_runtime.py').read_text()
    ast.parse(source, feature_version=(3, 6))
    namespace = {}
    exec(compile(source, 'inlined_factor_runtime', 'exec'), namespace)
    return namespace['compute_factor_batch']


@pytest.mark.parametrize('inlined', [False, True])
def test_calendar_history_uses_inclusive_calendar_floor_and_exact_output_date(inlined):
    asof = '2022-03-08'
    floor = pd.Timestamp('2021-12-08')
    dates = pd.bdate_range('2021-09-01', asof)
    dates = dates[~dates.isin(pd.date_range('2022-01-31', '2022-02-04'))]
    source = pd.DataFrame({'date': np.tile(dates, 2), 'code': np.repeat(['A', 'B'], len(dates)),
                           'close': np.arange(len(dates) * 2, dtype=float) + 1})
    source = source.loc[~(source.code.eq('B') & source.date.eq(floor))]
    frames, output_dates, requests = {}, {}, []

    class Adapter:
        def __init__(self, dependencies, **kwargs):
            self.window = kwargs['window']

        def fetch_price(self, stocks, day, fields):
            requests.append(self.window)
            return source.groupby('code', sort=False).tail(self.window).copy()

        def run_factor(self, cls, frame, **kwargs):
            frames[cls] = frame.copy()
            output_dates[cls] = kwargs['output_date']
            return pd.DataFrame({stock: [len(frame.loc[frame.code.eq(stock)])] for stock in ['A', 'B']},
                                index=pd.DatetimeIndex([asof]))

    requirements = {'Warm': {'requirements': {'daily': {'min_days': 11, 'fields': ['close']}},
        'runtime': {'window': 91, 'history_calendar_days': 90, 'output_date': 'exact'}},
        'Ordinary': {'requirements': {'daily': {'min_days': 2, 'fields': ['close']}}}}
    runner = _runner(inlined)
    result = runner({name: name for name in requirements}, requirements, {}, ['A', 'B'], asof, Adapter,
                    windows={'Warm': {'daily': 13}})
    assert requests == [91]
    expected = source.loc[source.date.ge(floor)].sort_values(['code', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(frames['Warm'].reset_index(drop=True), expected)
    assert output_dates == {'Ordinary': None, 'Warm': asof}
    assert frames['Warm'].date.min() == floor and frames['Warm'].date.max() == pd.Timestamp(asof)
    assert result['Ordinary'].tolist() == [2, 2]
    assert result['Warm'].tolist() == [len(expected.loc[expected.code.eq(stock)]) for stock in ['A', 'B']]
    for name in requirements:
        single = runner({name: name}, requirements, {}, ['A', 'B'], asof, Adapter)
        pd.testing.assert_series_equal(single[name], result[name])


@pytest.mark.parametrize('days', [0, -1, True, 1.5, '90'])
def test_invalid_calendar_history_fails_before_fetch(days):
    requirements = {'F': {'requirements': {'daily': {'min_days': 91, 'fields': ['close']}},
                         'runtime': {'history_calendar_days': days}}}
    with pytest.raises(ValueError, match='invalid_factor_calendar_history'):
        compute_factor_batch({'F': object}, requirements, {}, ['A'], '2022-03-08',
                             lambda *a, **kw: pytest.fail('invalid plan reached the producer'))


@pytest.mark.parametrize('override', [None, 13])
def test_calendar_history_rejects_insufficient_fetch_and_window_override(override):
    requirements = {'F': {'requirements': {'daily': {'min_days': 13 if override is None else 91, 'fields': ['close']}},
                         'runtime': {'history_calendar_days': 90}}}
    with pytest.raises(ValueError, match='factor_calendar_history_fetch_too_short'):
        compute_factor_batch({'F': object}, requirements, {}, ['A'], '2022-03-08',
                             lambda *a, **kw: pytest.fail('short fetch plan reached the producer'),
                             window_override=override)


def test_calendar_history_rejects_intraday_contract():
    requirements = {'F': {'requirements': {'5m': {'min_bars': 91, 'fields': ['close']}},
                         'runtime': {'history_calendar_days': 90}}}
    with pytest.raises(ValueError, match='invalid_factor_calendar_history'):
        compute_factor_batch({'F': object}, requirements, {}, ['A'], '2022-03-08',
                             lambda *a, **kw: pytest.fail('intraday calendar plan reached the producer'))
