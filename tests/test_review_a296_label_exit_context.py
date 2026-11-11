"""Legacy label producer exports its actual calendar exits, without changing labels."""
import duckdb
import pandas as pd
import pytest

from calendar_test_support import seed_calendar
from factor_synthesis.core.factor_ops import calculate_forward_returns


@pytest.mark.parametrize('route', ['main', 'fallback', 'pit'])
@pytest.mark.parametrize('period', [1, 2])
def test_legacy_exits_use_full_calendar_and_final_rows(route, period):
    with duckdb.connect(':memory:') as conn:
        seed_calendar(conn, '2025-01-01', '2025-01-15')
        conn.execute('CREATE TABLE prices(date DATE, stock_code VARCHAR, close DOUBLE, factor DOUBLE)')
        conn.executemany('INSERT INTO prices VALUES (?,?,?,?)', [
            (d, s, v, 1.) for d, v in [('2025-01-03', 10.), ('2025-01-06', 20.),
                                     ('2025-01-07', 40.), ('2025-01-08', 80.)]
            for s in ['A', 'B'] if not (s == 'B' and d == '2025-01-06')
        ])
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp(d), s) for d, s in
            [('2025-01-06', 'A'), ('2025-01-03', 'B'), ('2025-01-03', 'A'), ('2025-01-04', 'A')]],
            names=['date', 'stock_code'])
        result = calculate_forward_returns(pd.DataFrame({'x': 1.}, index=idx), conn,
            'missing' if route == 'fallback' else 'prices', 'prices', period=period,
            pit_enabled=route == 'pit')
        assert result.attrs['label_exit_dates'] == {
            '2025-01-06': '2025-01-07' if period == 1 else '2025-01-08',
            '2025-01-03': '2025-01-06' if period == 1 else '2025-01-07'}
        assert result.tolist() == ([1., 1.] if period == 1 else [3., 3., 3.])
        assert result.name == 'forward_return'
        assert result.index.tolist() == [x for x in idx if x[0].day != 4 and not
                                        (period == 1 and x[1] == 'B')]


def test_no_surviving_labels_has_empty_exit_map():
    with duckdb.connect(':memory:') as conn:
        seed_calendar(conn, '2025-01-01', '2025-01-15')
        conn.execute("CREATE TABLE prices AS SELECT DATE '2025-01-03' date, 'A' stock_code, 10.0 AS close")
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp('2025-01-03'), 'A')], names=['date', 'stock_code'])
        result = calculate_forward_returns(pd.DataFrame({'x': [1.]}, index=idx), conn,
                                           'prices', 'prices', period=1, pit_enabled=False)
        assert result.empty
        assert result.attrs['label_exit_dates'] == {}
