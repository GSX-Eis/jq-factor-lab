"""Invalid constituent keys cannot contaminate or replace an existing pool."""
import pandas as pd
import pytest

from data_pipeline.loaders.import_index_constituents_duckdb import DuckDBIndexConstituentsImporter


def write_rows(path, **changes):
    row = dict(date='2024-01-02', stock_code='000001', index_code='INDEX_A')
    row.update(changes)
    pd.DataFrame([row]).to_csv(path, index=False)


@pytest.mark.parametrize('bad', [
    {'date':'not-a-date'}, {'date':None}, {'stock_code':'   '},
    {'stock_code':None}, {'index_code':'   '},
])
def test_invalid_keys_roll_back_replace(tmp_path, bad):
    path = tmp_path/'constituents.csv'
    write_rows(path)
    importer = DuckDBIndexConstituentsImporter(str(tmp_path/'index.duckdb'))
    try:
        importer.import_csv(str(path))
        write_rows(path, **bad)
        with pytest.raises(ValueError, match='invalid rows'):
            importer.import_csv(str(path), replace=True)
        assert importer.conn.execute('SELECT * FROM index_constituents').fetchall() == [
            ('2024-01-02', '000001', 'INDEX_A')]
    finally:
        importer.close()


def test_valid_keys_trim_and_keep_leading_zeroes_and_quoted_path(tmp_path):
    path = tmp_path/"user's.csv"
    write_rows(path, stock_code=' 000001 ', index_code=' INDEX_A ')
    importer = DuckDBIndexConstituentsImporter(str(tmp_path/'index.duckdb'))
    try:
        assert importer.import_csv(str(path))['total_records'] == 1
        assert importer.conn.execute('SELECT * FROM index_constituents').fetchall() == [
            ('2024-01-02', '000001', 'INDEX_A')]
    finally:
        importer.close()


def test_header_only_file_cannot_clear_existing_pool(tmp_path):
    path = tmp_path/'constituents.csv'
    write_rows(path)
    importer = DuckDBIndexConstituentsImporter(str(tmp_path/'index.duckdb'))
    try:
        importer.import_csv(str(path))
        path.write_text('date,stock_code,index_code\n')
        with pytest.raises(ValueError, match='no index constituent rows'):
            importer.import_csv(str(path), replace=True)
        assert importer.conn.execute('SELECT count(*) FROM index_constituents').fetchone()[0] == 1
    finally:
        importer.close()
