import csv
import json

import duckdb
import pytest
from pandas.testing import assert_frame_equal

from data_pipeline.loaders import import_raw_financial_statements_duckdb as loader


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_interruption_rolls_back_rows_and_revision(tmp_path, monkeypatch, error_type):
    columns = ['id', 'code', 'pub_date', 'end_date', 'profit']
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'income': {'columns': columns}}))
    source = tmp_path / 'income'
    source.mkdir()
    csv_file = source / 'raw_income_statement_2024.csv'

    def write(profit):
        with csv_file.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerow([1, 'A', '2024-04-30', '2023-12-31', profit])

    with duckdb.connect() as conn:
        write(10)
        loader.import_statement(conn, 'income', tmp_path, contract)
        queries = ['SELECT * FROM raw_income_statement_events', 'SELECT * FROM meta.cache_revision']
        before = [conn.execute(query).fetchdf() for query in queries]
        advance = loader.advance_revision

        def interrupt(connection):
            advance(connection)
            raise error_type('cancelled')

        monkeypatch.setattr(loader, 'advance_revision', interrupt)
        write(99)
        with pytest.raises(error_type, match='cancelled'):
            loader.import_statement(conn, 'income', tmp_path, contract)
        for query, expected in zip(queries, before):
            assert_frame_equal(expected, conn.execute(query).fetchdf(), check_exact=True)
        conn.execute('BEGIN')
        conn.execute('ROLLBACK')
