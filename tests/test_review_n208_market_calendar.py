"""Market-cap lag uses the explicit research calendar, never filtered row union."""
import duckdb
import pandas as pd
import pytest

from factor_synthesis.core.factor_data_loader import load_market_cap_data
from utils.trading_calendar import CalendarCoverageError, TradingCalendar


@pytest.fixture
def calendar():
    return TradingCalendar(['2025-01-02', '2025-01-03', '2025-01-06'],
                           '2025-01-01', '2025-01-07', 'bound-market')


@pytest.fixture
def connection():
    with duckdb.connect(':memory:') as conn:
        conn.execute('CREATE TABLE cap(date DATE, stock_code VARCHAR, natural_log_of_market_cap DOUBLE)')
        conn.executemany('INSERT INTO cap VALUES (?, ?, ?)', [
            ('2025-01-02', 'A', 10.), ('2025-01-03', 'A', None),
            ('2025-01-06', 'A', 30.), ('2025-01-03', 'B', 90.),
        ])
        yield conn


@pytest.mark.parametrize('stocks', [{'A'}, {'A', 'B'}, None])
@pytest.mark.parametrize('missing_row', [False, True])
def test_gap_lag_is_independent_of_stock_union(connection, calendar, stocks, missing_row):
    if missing_row:
        connection.execute("DELETE FROM cap WHERE stock_code='A' AND date='2025-01-03'")
    result = load_market_cap_data(connection, 'cap', '2025-01-02', '2025-01-06',
                                  stocks, trading_calendar=calendar)
    assert result.xs('A', level='stock_code').to_dict() == {pd.Timestamp('2025-01-03'): 10.}
    if stocks != {'A'}:
        assert result.xs('B', level='stock_code').to_dict() == {pd.Timestamp('2025-01-06'): 90.}
    assert result.name == 'log_market_cap'


def test_last_observation_moves_to_next_market_session(connection, calendar):
    connection.execute("DELETE FROM cap WHERE date>'2025-01-02'")
    result = load_market_cap_data(connection, 'cap', '2025-01-02', '2025-01-06',
                                  {'A'}, trading_calendar=calendar)
    assert result.to_dict() == {(pd.Timestamp('2025-01-03'), 'A'): 10.}


def test_window_does_not_load_warmup_or_fill_gap(connection, calendar):
    result = load_market_cap_data(connection, 'cap', '2025-01-03', '2025-01-06',
                                  {'A'}, trading_calendar=calendar)
    assert result.empty
    assert result.name == 'log_market_cap'


@pytest.mark.parametrize('start,end', [('2024-12-31', '2025-01-06'), ('2025-01-02', '2025-01-08')])
def test_coverage_failure_precedes_any_query(calendar, start, end):
    class NoQuery:
        def execute(self, *args, **kwargs):
            raise AssertionError('coverage must be checked before SQL')
    with pytest.raises(CalendarCoverageError):
        load_market_cap_data(NoQuery(), 'cap', start, end, trading_calendar=calendar)


def test_calendar_keyword_has_no_default(connection):
    with pytest.raises(TypeError, match='trading_calendar'):
        load_market_cap_data(connection, 'cap', '2025-01-02', '2025-01-06')
