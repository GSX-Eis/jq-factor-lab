"""Financial retries must preserve schema and write by column name."""
import json

import duckdb
import pytest

from data_pipeline.loaders.import_raw_financial_statements_duckdb import import_statement


COLUMNS = ['id', 'code', 'pub_date', 'end_date', 'revenue']
TABLE = 'raw_income_statement_events'


def inputs(tmp_path):
    source = tmp_path / 'input'
    (source / 'income').mkdir(parents=True)
    (source / 'income/raw_income_statement_2024.csv').write_text(
        'id,code,pub_date,end_date,revenue\n1,DEMO000,2024-04-01,2023-12-31,100.5\n'
    )
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'income': {'columns': COLUMNS}}))
    return source, contract


def test_failed_contract_does_not_leave_schema_and_correct_retry_succeeds(tmp_path):
    source, contract = inputs(tmp_path)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'income': {'columns': COLUMNS[:-1] + ['missing_revenue']}}))
    with duckdb.connect(str(tmp_path / 'data.duckdb')) as conn:
        with pytest.raises(duckdb.Error):
            import_statement(conn, 'income', source, bad)
        assert conn.execute('show tables').fetchall() == []
        result = import_statement(conn, 'income', source, contract)
        assert result['imported_rows'] == 1
        assert conn.execute(f'select revenue from {TABLE}').fetchall() == [(100.5,)]


@pytest.mark.parametrize('schema', [
    'id BIGINT PRIMARY KEY, code VARCHAR, pub_date DATE, end_date DATE, missing_revenue DOUBLE',
    'id BIGINT PRIMARY KEY, code VARCHAR, pub_date DATE, end_date DATE, revenue VARCHAR',
])
def test_incompatible_existing_schema_is_rejected_without_data_changes(tmp_path, schema):
    source, contract = inputs(tmp_path)
    with duckdb.connect(':memory:') as conn:
        conn.execute(f'create table {TABLE} ({schema})')
        conn.execute(f"insert into {TABLE} values (9, 'EXISTING', '2024-01-01', '2023-12-31', 7)")
        before = conn.execute(f'select * from {TABLE}').fetchall()
        with pytest.raises(ValueError, match='Schema mismatch'):
            import_statement(conn, 'income', source, contract)
        assert conn.execute(f'select * from {TABLE}').fetchall() == before


def test_reordered_existing_columns_are_written_by_name_and_updates_work(tmp_path):
    source, contract = inputs(tmp_path)
    with duckdb.connect(':memory:') as conn:
        conn.execute(f'create table {TABLE} (revenue DOUBLE, end_date DATE, id BIGINT PRIMARY KEY, pub_date DATE, code VARCHAR)')
        import_statement(conn, 'income', source, contract)
        assert conn.execute(f'select id, revenue, code from {TABLE}').fetchall() == [(1, 100.5, 'DEMO000')]
        csv = source / 'income/raw_income_statement_2024.csv'
        csv.write_text(csv.read_text().replace('100.5', '200.25'))
        import_statement(conn, 'income', source, contract)
        assert conn.execute(f'select id, revenue from {TABLE}').fetchall() == [(1, 200.25)]
