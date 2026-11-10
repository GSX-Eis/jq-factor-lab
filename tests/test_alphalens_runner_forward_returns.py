from utils.trading_calendar import TradingCalendar
import numpy as np
import pandas as pd
import threading
import time
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator


@pytest.mark.parametrize('precomputed', [False, True])
def test_runner_uses_real_sessions_and_requires_full_lagged_horizon(precomputed) -> None:
    dates = pd.bdate_range("2025-01-02", periods=60).delete(10)
    assets = [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(dates), len(assets))), axis=0)),
        index=dates,
        columns=assets,
    )
    factors = pd.DataFrame(
        rng.normal(size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    pricing = prices.stack().reorder_levels([1, 0]).sort_index()
    pricing.index.names = ["asset", "date"]
    # Explicit fixture marks the omitted weekday as closed, independent of observations.
    pricing.attrs["trading_calendar"] = TradingCalendar(dates, dates.min(), dates.max())

    forward_returns = ForwardReturnsCalculator().calculate_forward_returns(pricing, (20,)) if precomputed else None
    results = AlphalensRunner(periods=(20,)).run(factors, pricing, forward_returns, max_loss=21 / len(dates))

    assert results["ic"].index.max() == dates[-22]
    assert len(results["ic"].dropna()) == len(dates) - 21
    assert results['ic'].attrs['trading_calendar'].fingerprint == pricing.attrs['trading_calendar'].fingerprint
    from utils.duckdb_results_writer import DuckDBResultsWriter
    monthly = object.__new__(DuckDBResultsWriter)._calculate_monthly_metrics(results['ic'])
    assert monthly.ic_t_stat.notna().all()


def test_alignment_keeps_future_prices_without_inventing_weekdays() -> None:
    price_dates = pd.bdate_range("2025-01-02", periods=30).delete(10)
    factor_dates = price_dates[::5]
    runner = AlphalensRunner(periods=(20,))

    factors, prices = runner._fix_business_day_alignment(
        pd.DataFrame({"S": range(len(factor_dates))}, index=factor_dates),
        pd.DataFrame({"S": range(len(price_dates))}, index=price_dates),
    )

    assert factors.index.equals(factor_dates)
    assert prices.index.equals(price_dates)
    assert prices.index.max() > factors.index.max()


def test_forward_return_cache_is_single_writer_and_source_versioned(tmp_path, monkeypatch) -> None:
    calculator = ForwardReturnsCalculator(cache_dir=str(tmp_path))
    dates = pd.bdate_range("2025-01-02", periods=8)
    pricing = pd.DataFrame({"A": range(10, 18)}, index=dates).stack()
    pricing.index.names = ["date", "asset"]
    pricing.attrs["trading_calendar"] = TradingCalendar(dates, dates.min(), dates.max())
    calls = 0

    original = calculator.calculate_forward_returns

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.03)
        return original(*args, **kwargs)

    monkeypatch.setattr(calculator, "calculate_forward_returns", counted)
    outputs = []

    def load(signature):
        outputs.append(
            calculator.calculate_or_load_forward_returns(
                pricing,
                ["A"],
                "2025-01-02",
                "2025-01-13",
                (1,),
                source_signature=signature,
            )
        )

    threads = [threading.Thread(target=load, args=("source-v1",)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    pd.testing.assert_frame_equal(outputs[0], outputs[1])

    load("source-v2")
    assert calls == 2


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('bad', [0., -1., np.inf, -np.inf, 'bad', True, 1+2j])
def test_invalid_prices_raise_with_keys(tmp_path, cached, bad):
    dates = pd.bdate_range('2024-01-02', periods=4)
    pricing = pd.Series([100., bad, 110., 121.], dtype=object,
        index=pd.MultiIndex.from_product([['A'], dates], names=['asset', 'date']))
    calculator = ForwardReturnsCalculator(str(tmp_path), trading_calendar=TradingCalendar(dates, dates.min(), dates.max()))
    with pytest.raises(ValueError, match='Invalid price.*2024-01-03.*A'):
        if cached:
            calculator.calculate_or_load_forward_returns(pricing, ['A'], str(dates.min()), str(dates.max()), (1,), 0)
        else:
            calculator.calculate_forward_returns(pricing, (1,), 0)
    assert not list(tmp_path.glob('forward_returns_*.pkl'))


@pytest.mark.parametrize('kind', ['empty', 'missing', 'too_short', 'disjoint'])
def test_no_valid_labels_raises(kind):
    dates = pd.bdate_range('2024-01-02', periods=4)
    values = {'empty': [], 'missing': [np.nan]*4, 'too_short': [100.], 'disjoint': [100., np.nan, 110., np.nan]}[kind]
    pricing = pd.Series(values, dtype=float, index=pd.MultiIndex.from_product([['A'], dates[:len(values)]], names=['asset','date']))
    calc = ForwardReturnsCalculator(trading_calendar=TradingCalendar(dates, dates.min(), dates.max()))
    with pytest.raises(ValueError, match='未能计算出任何前瞻收益率'):
        calc.calculate_forward_returns(pricing, (1,), 0)


def test_missing_prices_and_unavailable_long_horizon_remain_missing():
    dates = pd.bdate_range('2024-01-02', periods=4)
    pricing = pd.Series([100., pd.NA, 110., 121.], dtype='Float64',
        index=pd.MultiIndex.from_product([['A'], dates], names=['asset', 'date']))
    calc = ForwardReturnsCalculator(trading_calendar=TradingCalendar(dates, dates.min(), dates.max()))
    result = calc.calculate_forward_returns(pricing, (1, 20), 0)
    assert result.index.tolist() == [(dates[2], 'A')]
    assert result.iloc[0]['1D'] == pytest.approx(.1)
    assert result['20D'].isna().all()


def test_old_price_contract_cache_cannot_bypass_validation(tmp_path):
    import hashlib
    import pickle
    dates = pd.bdate_range('2024-01-02', periods=4)
    pricing = pd.Series([100., 0., 110., 121.], index=pd.MultiIndex.from_product([['A'], dates], names=['asset','date']))
    calendar = TradingCalendar(dates, dates.min(), dates.max())
    signature = hashlib.sha256(calendar.fingerprint.encode() + pd.util.hash_pandas_object(pricing, index=True).values.tobytes()).hexdigest()
    old_data = dict(code_list=['A'], start_date='2024-01-02', end_date='2024-01-05', periods=[1], source_signature=signature)
    key = hashlib.md5(str(old_data).encode()).hexdigest() + '_lag0'
    stale = tmp_path / f'forward_returns_{key}.pkl'
    stale.write_bytes(pickle.dumps(pd.DataFrame({'1D': [-1.]})))
    original = stale.read_bytes()
    calc = ForwardReturnsCalculator(str(tmp_path), trading_calendar=calendar)
    with pytest.raises(ValueError, match='Invalid price'):
        calc.calculate_or_load_forward_returns(pricing, ['A'], '2024-01-02', '2024-01-05', (1,), 0)
    assert stale.read_bytes() == original


def test_alphalens_does_not_hide_invalid_price_failure():
    dates = pd.bdate_range('2024-01-02', periods=30)
    assets = list('ABCDE')
    wide = pd.DataFrame(100.+np.arange(150).reshape(30,5), index=dates, columns=assets)
    wide.iloc[10,0] = 0.
    pricing = wide.stack().swaplevel()
    pricing.index.names = ['asset','date']
    pricing.attrs['trading_calendar'] = TradingCalendar(dates, dates.min(), dates.max())
    factors = pd.DataFrame(np.tile(np.arange(5), (30,1)), index=dates, columns=assets)
    with pytest.raises(ValueError, match='Invalid price'):
        AlphalensRunner(periods=(1,)).run(factors, pricing)


@pytest.fixture
def cached_returns_case(tmp_path):
    dates = pd.bdate_range('2024-01-02', periods=8)
    pricing = pd.Series(np.arange(8)+100., index=pd.MultiIndex.from_product(
        [['A'], dates], names=['asset', 'date']))
    calc = ForwardReturnsCalculator(str(tmp_path), trading_calendar=TradingCalendar(dates, dates.min(), dates.max()))
    def load(data=pricing, periods=(1, 3)):
        return calc.calculate_or_load_forward_returns(data, ['A'], '2024-01-02', '2024-01-11', periods, 0)
    return calc, pricing, load, tmp_path


def test_cache_does_not_hide_changed_index_semantics(cached_returns_case):
    calc, pricing, load, _ = cached_returns_case
    load()
    renamed = pricing.rename_axis(['date', 'asset'])
    with pytest.raises(ValueError, match='无法解析的日期'):
        load(renamed)
    # A real level swap, including its values, remains valid.
    reordered = pricing.swaplevel().sort_index()
    pd.testing.assert_frame_equal(load(reordered), calc.calculate_forward_returns(reordered, (1, 3), 0))


def test_cached_period_order_matches_direct_calculation(cached_returns_case, monkeypatch):
    calc, pricing, load, _ = cached_returns_case
    load()
    expected = calc.calculate_forward_returns(pricing, (3, 1), 0)
    pd.testing.assert_frame_equal(load(periods=(3, 1)), expected)
    pd.testing.assert_frame_equal(load(periods=(3, 1, 3)), expected)
    monkeypatch.setattr(calc, 'calculate_forward_returns', lambda *a: pytest.fail('valid cache should hit'))
    pd.testing.assert_frame_equal(load(periods=(3, 1, 3)), expected)


def test_object_price_types_cannot_bypass_validation(cached_returns_case):
    _, pricing, load, _ = cached_returns_case
    numeric = pricing.astype(object)
    load(numeric)
    text = numeric.astype(str).astype(object)
    # pandas' object hashing alone treats these different inputs identically.
    assert pd.util.hash_pandas_object(numeric).equals(pd.util.hash_pandas_object(text))
    with pytest.raises(ValueError, match='Invalid price'):
        load(text)


@pytest.mark.parametrize('kind', [
    'object', 'empty', 'flat_index', 'index_names', 'duplicate_index',
    'null_asset', 'string_dates', 'wrong_columns', 'column_order',
    'infinite', 'text', 'boolean', 'complex', 'empty_row', 'truncated_pickle',
])
def test_malformed_return_cache_is_recomputed(cached_returns_case, monkeypatch, kind):
    import pickle
    calc, pricing, load, folder = cached_returns_case
    expected = load()
    bad = expected.copy()
    if kind == 'object': bad = 'not a DataFrame'
    elif kind == 'empty': bad = bad.iloc[:0]
    elif kind == 'flat_index': bad = bad.reset_index(drop=True)
    elif kind == 'index_names': bad = bad.rename_axis(['asset', 'date'])
    elif kind == 'duplicate_index': bad = pd.concat([bad, bad.iloc[:1]])
    elif kind == 'null_asset':
        bad.index = pd.MultiIndex.from_arrays([bad.index.get_level_values(0), [None]*len(bad)], names=['date','asset'])
    elif kind == 'string_dates':
        bad.index = pd.MultiIndex.from_arrays([bad.index.get_level_values(0).astype(str), bad.index.get_level_values(1)], names=['date','asset'])
    elif kind == 'wrong_columns': bad = bad.rename(columns={'1D':'2D'})
    elif kind == 'column_order': bad = bad[['3D','1D']]
    elif kind == 'infinite': bad.iloc[0,0] = np.inf
    elif kind == 'text': bad = bad.astype(str)
    elif kind == 'boolean': bad = bad.notna()
    elif kind == 'complex': bad = bad.astype(complex)
    elif kind == 'empty_row': bad.iloc[0,:] = np.nan
    path = next(folder.glob('forward_returns_*.pkl'))
    path.write_bytes(b'broken pickle' if kind == 'truncated_pickle' else pickle.dumps(bad))
    calls = []
    original = calc.calculate_forward_returns
    def calculate(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(calc, 'calculate_forward_returns', calculate)
    pd.testing.assert_frame_equal(load(), expected)
    assert calls == [1]
    pd.testing.assert_frame_equal(load(), expected)
    assert calls == [1]


def test_direct_cache_reader_rejects_wrong_object(cached_returns_case):
    calc, _, _, _ = cached_returns_case
    calc.save_to_cache('wrong', 'not a DataFrame')
    assert calc.load_from_cache('wrong') is None
