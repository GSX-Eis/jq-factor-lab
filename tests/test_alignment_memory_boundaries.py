import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize('pit', [False, True])
def test_large_price_fetch_preserves_api_batches_fields_and_pit(monkeypatch, pit):
    import utils.jq_factor_adapter as module

    monkeypatch.setattr(module, 'EXTERNAL_PIT_MODE', None)
    days = pd.bdate_range('2021-01-04', periods=120)
    offsets = np.r_[np.arange(575, 695, 5), np.arange(785, 905, 5)]
    times = np.repeat(days.to_numpy(), 48) + np.tile(offsets, len(days)).astype('timedelta64[m]')
    stocks = [f'S{index:02}' for index in reversed(range(36))]
    raw = pd.DataFrame(dict(date=np.repeat(times, len(stocks)),
        code=np.tile(stocks, len(times)), open=10., close=11.))
    raw.loc[raw.index[::53], 'close'] = np.nan
    calls = []

    def make_adapter():
        def api(**kwargs):
            calls.append(tuple(kwargs['security']))
            return (raw.loc[raw.code.isin(kwargs['security']), ['date', 'code'] + kwargs['fields']]
                    .sort_values(['date', 'code']).reset_index(drop=True))

        adapter = module.JQFactorAdapter({}, freq='5m', window=5760,
            get_price_func=api, strict=True, pit_mode='research')
        adapter._probed = True
        adapter.pit_enabled = pit

        def factors(securities, start, end):
            dates = np.repeat(days.to_numpy(), len(securities))
            codes = np.tile(securities, len(days))
            return pd.DataFrame(dict(date=dates, code=codes,
                factor=(np.asarray([int(code[1:]) + 1 for code in codes])
                        * np.where(pd.DatetimeIndex(dates).day > 15, 2., 1.))))

        adapter._fetch_factor_separately = factors
        return adapter

    fields = ['open', 'close', 'factor'] if pit else ['open', 'close']
    original = make_adapter()._fetch_price_unbatched(stocks, '2022-01-04', fields)
    original_calls = list(calls)
    calls.clear()
    bounded = make_adapter().fetch_price(stocks, '2022-01-04', fields)
    pd.testing.assert_frame_equal(original, bounded, check_exact=True)
    assert calls == original_calls
    assert len(bounded) == len(raw) and not bounded.duplicated(['date', 'code']).any()


@pytest.mark.parametrize('frequency,expanding,fields', [
    ('daily', False, ['close']), ('5m', True, ['close']), ('5m', False, ['custom_field'])])
def test_unrelated_price_fetch_paths_keep_the_original_call(frequency, expanding, fields):
    from utils.jq_factor_adapter import JQFactorAdapter, bounded_price_fetch

    adapter = JQFactorAdapter({}, freq=frequency, window=10000)
    adapter.use_expanding_window = expanding
    stocks = ['A', 'B'] * 20
    sentinel = object()
    calls = []

    def fetch_one(securities, end, requested):
        calls.append((securities, end, requested))
        return sentinel

    assert bounded_price_fetch(adapter, stocks, '2022-01-04', fields, fetch_one) is sentinel
    assert calls == [(stocks, '2022-01-04', fields)]


@pytest.mark.parametrize('frame', [
    pd.DataFrame(),
    pd.DataFrame(dict(value=[np.nan, np.inf, -np.inf, 1e-300],
                      text=['中文', 'a,b', 'a\nb', '"quoted"'])),
    pd.DataFrame(dict(date=pd.date_range('2020-01-01', periods=3), value=[True, False, None]))])
def test_streaming_input_fingerprint_preserves_the_default_csv_bytes(frame):
    source = Path(__file__).resolve().parents[1] / 'jq_strategies/cloud_factor_dumper.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'streaming_csv_fingerprint')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    assert namespace['streaming_csv_fingerprint'](frame) == hashlib.sha256(
        frame.to_csv(index=False).encode('utf-8')).hexdigest()


@pytest.mark.parametrize('count', [1, 2])
def test_single_and_shared_factor_groups_preserve_results(count):
    from utils.factor_runtime import compute_factor_batch

    class First:
        multiplier = 1

    class Second:
        multiplier = 2

    fetches = []

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_price(self, stocks, asof, fields):
            fetches.append((list(stocks), asof, list(fields)))
            return pd.DataFrame(dict(date=np.repeat(pd.bdate_range('2023-01-02', periods=5), 2),
                code=['A', 'B'] * 5, close=np.arange(1., 11.)))

        def run_factor(self, factor, frame, **kwargs):
            return frame.pivot(index='date', columns='code', values='close') * factor.multiplier

    classes = dict(First=First, Second=Second) if count == 2 else dict(First=First)
    requirements = {'factors': {name: {'requirements': {'daily': {'min_bars': 3, 'fields': ['close']}}}
                                for name in classes}}
    result = compute_factor_batch(classes, requirements, {}, ['A', 'B'], '2023-01-06', Adapter)
    assert len(fetches) == 1
    assert result.First.tolist() == [9., 10.]
    if count == 2:
        assert result.Second.tolist() == [18., 20.]
