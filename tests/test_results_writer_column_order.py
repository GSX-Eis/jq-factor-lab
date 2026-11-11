"""An existing compatible table need not have the writer's physical column order."""
import duckdb
import pandas as pd
import pytest

from utils.duckdb_results_writer import DuckDBResultsWriter


@pytest.mark.parametrize('borrowed', [False, True])
@pytest.mark.parametrize('ranged', [False, True])
def test_factor_values_existing_reordered_table(tmp_path, borrowed, ranged):
    path = tmp_path / 'results.duckdb'
    with duckdb.connect(str(path)) as conn:
        conn.execute('''CREATE TABLE factor_values (
            date DATE, factor_name VARCHAR, stock_code VARCHAR,
            factor_value DOUBLE, created_at TIMESTAMP,
            PRIMARY KEY(date, stock_code, factor_name))''')
        conn.execute("""INSERT INTO factor_values VALUES
            ('2024-01-02','alpha','A',7,'2024-01-01'),
            ('2024-01-03','alpha','A',8,'2024-01-01'),
            ('2024-01-02','other','A',9,'2024-01-01')""")
    writer = DuckDBResultsWriter(str(path), export_parquet=False)
    frame = pd.DataFrame({'A': [1.], 'B': [2.]}, index=pd.to_datetime(['2024-01-02']))
    options = {'date_range': ('2024-01-02', '2024-01-02'), 'skip_parquet': True} if ranged else {}
    if borrowed:
        with writer.acquire_write_session() as session:
            session.conn.execute('BEGIN')
            assert writer.save_factor_values(frame, 'alpha', duckdb_conn=session.conn, **options) == 2
            session.conn.execute('COMMIT')
    else:
        assert writer.save_factor_values(frame, 'alpha', **options) == 2
    with duckdb.connect(str(path), read_only=True) as conn:
        rows = conn.execute('''SELECT CAST(date AS VARCHAR), stock_code, factor_name, factor_value
            FROM factor_values ORDER BY date, stock_code, factor_name''').fetchall()
        assert conn.execute("SELECT count(*) FROM factor_values WHERE factor_name='alpha' AND created_at IS NOT NULL").fetchone()[0] == (3 if ranged else 2)
    expected = [('2024-01-02','A','alpha',1.), ('2024-01-02','A','other',9.),
                ('2024-01-02','B','alpha',2.)]
    if ranged:
        expected.append(('2024-01-03','A','alpha',8.))
    assert rows == expected
