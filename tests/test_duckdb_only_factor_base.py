"""Local factor reads must use one DuckDB source, including in-memory callers."""
import duckdb
import pandas as pd
import pytest

from utils.factor_base import FactorBase


class CloseSignal(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(['close'], table_name='price_data')

    def calculate_factor(self, data):
        return data.pivot(index='date', columns='code', values='close')


def make_factor(connection=None, **kwargs):
    return CloseSignal(connection, 'CloseSignal', '2024-01-02', '2024-01-03',
                       stock_pool=['A'], pit_enabled=False, **kwargs)


@pytest.mark.parametrize('frequency,times', [
    ('daily', ['2024-01-02 00:00:00', '2024-01-03 00:00:00']),
    ('5m', ['2024-01-02 09:35:00', '2024-01-03 14:55:00']),
])
def test_memory_duckdb_factor_and_price_reads_include_end_day(frequency, times):
    with duckdb.connect(':memory:') as conn:
        frame = pd.DataFrame({'date': pd.to_datetime(times + ['2024-01-04'], format='mixed'),
                              'stock_code': ['A'] * 3, 'close': [10., 12., 99.]})
        conn.register('source', frame)
        conn.execute(f'CREATE TABLE price_data_{frequency} AS SELECT * FROM source')
        factor = make_factor(conn, data_frequency=frequency)
        actual = factor.get_factor_values(force_recalculate=True)
        pd.testing.assert_frame_equal(actual, factor.get_price_data())
        assert actual['A'].tolist() == [10., 12.]
        assert conn.execute('SELECT 1').fetchone()[0] == 1


def test_path_backed_factor_prices_and_forward_returns(tmp_path):
    path = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-01-02' AS date, 'A' AS stock_code, 10. AS close UNION ALL SELECT DATE '2024-01-03', 'A', 12.")
    factor = make_factor(duckdb_path=str(path))
    assert factor.get_price_data()['A'].tolist() == [10., 12.]
    assert factor.calculate_forward_returns(1)['A'].dropna().iloc[0] == pytest.approx(.2)
    factor._duckdb_adapter.close()


def test_empty_adapter_result_does_not_read_borrowed_connection(tmp_path):
    path = tmp_path / 'empty.duckdb'
    with duckdb.connect(str(path)) as conn:
        conn.execute('CREATE TABLE price_data_daily(date DATE, stock_code VARCHAR, close DOUBLE)')
    with duckdb.connect(':memory:') as borrowed:
        borrowed.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-01-02' AS date, 'A' AS stock_code, 99. AS close")
        factor = make_factor(borrowed, duckdb_path=str(path))
        assert factor.get_price_data().empty
        factor._duckdb_adapter.close()


def test_unsupported_backend_and_connection_fail_explicitly():
    with pytest.raises(ValueError, match='Only the duckdb'):
        make_factor(db_backend='unsupported')
    with pytest.raises(TypeError, match='DuckDB connection'):
        make_factor(object())
