"""Strict PIT contracts at helpers and real DuckDB business entry points."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from calendar_test_support import seed_calendar
from factor_synthesis.core.factor_data_loader import (
    PITValidationError, validate_pit_price_rows,
    apply_pit_adjustment_factor_end, apply_pit_adjustment_factor_end_df,
    get_pricing_data, get_open_price_data, get_complete_price_data,
)
from factor_synthesis.core.factor_ops import calculate_forward_returns


def panel(kind, values):
    dates = pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04'])
    if kind == 'series':
        return pd.Series(values, index=pd.MultiIndex.from_product([['A'], dates], names=['asset', 'date']), name='close')
    return pd.DataFrame({'A': values}, index=dates.rename('date'))


def adjust(kind, prices, factors):
    fn = apply_pit_adjustment_factor_end if kind == 'series' else apply_pit_adjustment_factor_end_df
    return fn(prices, factors)


@pytest.mark.parametrize('kind', ['series', 'wide'])
@pytest.mark.parametrize('bad', [None, 0., -1., np.inf, -np.inf, 'bad', '2', True, 1+2j])
def test_observed_prices_require_finite_positive_numeric_factors(kind, bad):
    prices = panel(kind, [100., 50., 25.])
    factors = panel(kind, [1., bad, 4.])
    with pytest.raises(ValueError, match='PIT.*factor.*A.*2023-01-03'):
        adjust(kind, prices, factors)


@pytest.mark.parametrize('kind', ['series', 'wide'])
def test_unsorted_input_uses_latest_observed_anchor_and_preserves_order(kind):
    prices = panel(kind, [100., 50., np.nan]).iloc[[2, 0, 1]]
    factors = panel(kind, [1., 2., 999.]).iloc[[1, 2, 0]]
    original = prices.copy(deep=True)
    expected = panel(kind, [50., 50., np.nan]).iloc[[2, 0, 1]]
    result = adjust(kind, prices, factors)
    if kind == 'series':
        pd.testing.assert_series_equal(result, expected)
        pd.testing.assert_series_equal(prices, original)
    else:
        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(prices, original)


@pytest.mark.parametrize('kind', ['series', 'wide'])
def test_empty_or_missing_factors_fail_only_where_prices_are_observed(kind):
    prices = panel(kind, [np.nan, 50., np.nan])
    factors = panel(kind, [np.nan, 2., 0.])
    result = adjust(kind, prices, factors)
    np.testing.assert_allclose(np.asarray(result), np.asarray(prices), equal_nan=True)
    with pytest.raises(ValueError, match='PIT.*factor'):
        adjust(kind, prices, factors.iloc[:0])
    blank = prices * np.nan
    assert adjust(kind, blank, factors.iloc[:0]).isna().all(axis=None)
    assert adjust(kind, prices.iloc[:0], factors.iloc[:0]).empty


@pytest.mark.parametrize('kind', ['series', 'wide'])
@pytest.mark.parametrize('which', ['prices', 'factors'])
def test_duplicate_keys_are_rejected(kind, which):
    prices = panel(kind, [100., 50., 25.])
    factors = panel(kind, [1., 2., 4.])
    if which == 'prices': prices = pd.concat([prices, prices.iloc[[0]]])
    else: factors = pd.concat([factors, factors.iloc[[0]]])
    with pytest.raises(ValueError, match='PIT.*duplicate'):
        adjust(kind, prices, factors)


def test_sparse_wide_stocks_have_independent_last_observed_anchors():
    prices = panel('wide', [100., 50., np.nan]).assign(B=[np.nan, 60., 30.], EMPTY=np.nan)
    factors = panel('wide', [1., 2., 999.]).assign(B=[np.nan, 3., 6.], EMPTY=np.nan)
    expected = panel('wide', [50., 50., np.nan]).assign(B=[np.nan, 30., 30.], EMPTY=np.nan)
    pd.testing.assert_frame_equal(adjust('wide', prices, factors), expected)


def read_entry(kind, conn, enabled=True, table='prices', stocks=None):
    stocks = stocks or ['A']
    if kind == 'pricing':
        return get_pricing_data(conn, table, stocks, pd.DatetimeIndex(['2023-01-02']),
                                pit_enabled=enabled, future_periods=1)
    if kind == 'training':
        signals = pd.DataFrame({'x': [1.]}, index=pd.MultiIndex.from_product(
            [pd.to_datetime(['2023-01-02']), ['A']], names=['date', 'stock_code']))
        return calculate_forward_returns(signals, conn, table, 'fallback', period=1, pit_enabled=enabled)
    fn = get_open_price_data if kind == 'open' else get_complete_price_data
    return fn(conn, table, stocks, '2023-01-02', '2023-01-03', pit_enabled=enabled)


@pytest.fixture
def connection():
    with duckdb.connect() as conn:
        seed_calendar(conn, '2023-01-02', '2023-01-04')
        conn.execute("CREATE TABLE prices (date TIMESTAMP, stock_code VARCHAR, close DOUBLE, open DOUBLE, factor DOUBLE)")
        conn.execute("INSERT INTO prices VALUES ('2023-01-02','A',100,100,1),('2023-01-03','A',50,50,2)")
        conn.execute('CREATE TABLE factor_data AS SELECT date, stock_code, close AS close_price FROM prices')
        conn.execute('CREATE TABLE fallback AS SELECT date, stock_code, close FROM prices')
        yield conn


@pytest.mark.parametrize('kind', ['pricing', 'open', 'complete', 'training'])
@pytest.mark.parametrize('bad', [None, 0., -1., np.inf])
def test_business_entrypoints_propagate_bad_factor(connection, kind, bad):
    connection.execute("UPDATE prices SET factor=? WHERE date='2023-01-03'", [bad])
    with pytest.raises(ValueError, match='PIT.*factor'):
        read_entry(kind, connection)
    # Deliberately unadjusted reads do not require a valid factor.
    assert read_entry(kind, connection, enabled=False) is not None


@pytest.mark.parametrize('kind', ['pricing', 'open', 'complete', 'training'])
def test_missing_factor_column_never_disables_enabled_pit(connection, kind):
    connection.execute('ALTER TABLE prices DROP COLUMN factor')
    with pytest.raises(Exception, match='PIT'):
        read_entry(kind, connection)
    assert read_entry(kind, connection, enabled=False) is not None


@pytest.mark.parametrize('kind', ['pricing', 'training'])
def test_raw_fallback_requires_explicitly_disabled_pit(connection, kind):
    connection.execute('DROP TABLE prices')
    with pytest.raises(PITValidationError, match='query failed'):
        read_entry(kind, connection)
    result = read_entry(kind, connection, enabled=False)
    expected = [-0.5] if kind == 'training' else [100., 50.]
    np.testing.assert_allclose(np.asarray(result).ravel(), expected)


@pytest.mark.parametrize('kind', ['pricing', 'complete', 'training'])
def test_intraday_aggregation_cannot_hide_missing_factor(connection, kind):
    connection.execute("UPDATE prices SET date=date + INTERVAL 15 HOUR WHERE date='2023-01-02'")
    connection.execute("INSERT INTO prices VALUES ('2023-01-02 09:30:00','A',90,90,NULL)")
    with pytest.raises(ValueError, match='PIT.*factor'):
        read_entry(kind, connection)


@pytest.mark.parametrize('kind', ['pricing', 'open', 'complete', 'training'])
def test_duplicate_timestamp_is_not_silently_aggregated(connection, kind):
    connection.execute("INSERT INTO prices SELECT * FROM prices WHERE date='2023-01-02'")
    with pytest.raises(ValueError, match='PIT.*duplicate'):
        read_entry(kind, connection)


def test_failed_stock_batch_does_not_return_partial_prices(connection):
    class BrokenSecondBatch:
        def __init__(self): self.calls = 0
        def execute(self, sql):
            self.calls += 1
            if self.calls == 2: raise RuntimeError('second batch failed')
            return connection.execute(sql)
    with pytest.raises(Exception, match='PIT'):
        read_entry('complete', BrokenSecondBatch(), stocks=['A'] + [f'S{i}' for i in range(500)])


@pytest.mark.parametrize('kind', ['pricing', 'open', 'complete', 'training'])
def test_valid_business_results_and_explicit_raw_mode(connection, kind):
    adjusted = read_entry(kind, connection)
    raw = read_entry(kind, connection, enabled=False)
    if kind == 'training':
        assert adjusted.iloc[0] == pytest.approx(0.)
        assert raw.iloc[0] == pytest.approx(-0.5)
    else:
        np.testing.assert_allclose(np.asarray(adjusted).ravel(), [50., 50.])
        np.testing.assert_allclose(np.asarray(raw).ravel(), [100., 50.])


@pytest.mark.parametrize('kind', ['pricing', 'complete', 'training'])
def test_empty_intraday_price_cannot_supply_daily_factor(connection, kind):
    connection.execute("INSERT INTO prices VALUES ('2023-01-02 15:00:00','A',NULL,NULL,999)")
    result = read_entry(kind, connection)
    expected = [0.] if kind == 'training' else [50., 50.]
    np.testing.assert_allclose(np.asarray(result).ravel(), expected)


@pytest.mark.parametrize('kind', ['series', 'wide'])
@pytest.mark.parametrize('values', [[1e308, 1., 1e-308], [1e-308, 1., 1e308]])
def test_overflow_or_underflow_cannot_produce_invalid_ratios(kind, values):
    with pytest.raises(PITValidationError, match='factor ratio'):
        adjust(kind, panel(kind, [100., 50., 25.]), panel(kind, values))


@pytest.mark.parametrize('kind', ['series', 'wide'])
def test_sorted_valid_panels_keep_previous_formula(kind):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2023-01-02', periods=50, name='date')
    prices = pd.DataFrame(rng.uniform(10, 100, size=(50, 8)), index=dates, columns=list('ABCDEFGH'))
    factors = pd.DataFrame(rng.uniform(.1, 3, size=prices.shape), index=dates, columns=prices.columns)
    expected = prices * factors.div(factors.iloc[-1], axis=1)
    if kind == 'series':
        def series(frame):
            result = frame.rename_axis(columns='asset').stack().swaplevel().sort_index()
            result.name = 'close'
            return result
        pd.testing.assert_series_equal(adjust(kind, series(prices), series(factors)), series(expected))
    else:
        pd.testing.assert_frame_equal(adjust(kind, prices, factors), expected)


def test_series_aligns_named_levels_without_changing_output_index():
    prices = panel('series', [100., 50., 25.]).swaplevel()
    factors = panel('series', [1., 2., 4.]).rename_axis(['stock_code', 'date'])
    expected = panel('series', [25., 25., 25.]).swaplevel()
    pd.testing.assert_series_equal(adjust('series', prices, factors), expected)


@pytest.mark.parametrize('bad_date', ['not-a-date', None])
def test_invalid_row_dates_raise_pit_error(bad_date):
    rows = pd.DataFrame({'stock_code': ['A'], 'date': [bad_date], 'close': [100.], 'factor': [1.]})
    with pytest.raises(PITValidationError, match='PIT.*dates'):
        validate_pit_price_rows(rows, 'close')


@pytest.mark.parametrize('kind', ['series', 'wide'])
def test_bad_value_diagnostics_report_total_and_five_original_values(kind):
    from factor_synthesis.core.factor_data_loader import _pit_bad_values

    dates = pd.date_range('2023-01-01', periods=3)
    frame = pd.DataFrame([['bad', '2'], [None, True], [-1, 'sixth']], index=dates, columns=['A', 'B'])
    if kind == 'series':
        frame = pd.Series(frame.to_numpy().ravel(), index=pd.MultiIndex.from_product([dates, ['A', 'B']]))
    original = frame.copy(deep=True)
    with pytest.raises(PITValidationError) as error:
        _pit_bad_values(frame, np.ones(frame.shape, dtype=bool), 'invalid factor')
    message = str(error.value)
    assert '6 invalid observations' in message
    assert message.count('value=') == 5
    assert 'sixth' not in message
    for value in ['bad', '2', None, True, -1]:
        assert f'value={value!r}' in message
    assert 'A' in message and 'B' in message and '2023-01-01' in message
    if kind == 'series':
        pd.testing.assert_series_equal(frame, original)
    else:
        pd.testing.assert_frame_equal(frame, original)


def test_bad_value_diagnostics_scan_sparse_noncontiguous_mask_in_row_order():
    from factor_synthesis.core.factor_data_loader import _pit_bad_values

    frame = pd.DataFrame(np.arange(12000).reshape(6000, 2), columns=['A', 'B'])
    mask = np.zeros((2, 6000), dtype=bool).T
    mask[4500, 1] = mask[5500, 0] = True
    assert not mask.flags.c_contiguous
    with pytest.raises(PITValidationError) as error:
        _pit_bad_values(frame, mask, 'invalid factor')
    message = str(error.value)
    assert '2 invalid observations' in message
    assert f"'B' / 4500: value={frame.iloc[4500, 1]!r}" in message
    assert f"'A' / 5500: value={frame.iloc[5500, 0]!r}" in message
    assert message.index("'B' / 4500") < message.index("'A' / 5500")
    assert _pit_bad_values(frame, np.zeros(frame.shape, dtype=bool), 'invalid factor') is None
