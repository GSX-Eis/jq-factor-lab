"""Owned CLI connections and approved analyst staging identity contract."""
import hashlib
import json
import sys

import duckdb
import pandas as pd
import pytest

from data_pipeline.loaders import import_index_constituents_duckdb as index
from data_pipeline.loaders import import_regime_history_duckdb as history
from utils.data_store import DataRef, migrate_table, read_reference
from utils.input_store import _keys, publish_input_batch

TABLE = 'regime_all_a_analyst_consensus_daily'
COLUMNS = ['stock_code', 'asof_date', 'forecast_year_level', 'source_id', 'value']
SCHEMA = 'stock_code VARCHAR, asof_date DATE, forecast_year_level INTEGER, source_id BIGINT PRIMARY KEY, value DOUBLE'


@pytest.mark.parametrize('where', ['import', 'export'])
@pytest.mark.parametrize('error', [ValueError, KeyboardInterrupt])
def test_index_cli_closes_on_failure(tmp_path, monkeypatch, where, error):
    path = tmp_path / 'index.csv'
    path.write_text('date,stock_code,index_code\n2024-01-02,000001,I\n')
    owner = index.DuckDBIndexConstituentsImporter(str(tmp_path / 'index.duckdb'))
    connection = owner.connect()
    monkeypatch.setattr(index, 'DuckDBIndexConstituentsImporter', lambda **kw: owner)
    def fail(*args, **kwargs):
        raise error('injected')
    monkeypatch.setattr(owner, 'import_csv' if where == 'import' else 'export_to_parquet', fail)
    monkeypatch.setattr(sys, 'argv', ['import', '--csv', str(path), '--export-parquet'])
    try:
        if error is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):
                index.main()
        else:
            assert index.main() == 1
        assert owner.conn is None
        with pytest.raises(duckdb.ConnectionException):
            connection.execute('SELECT 1')
    finally:
        owner.close()


def setup_parent(root, monkeypatch):
    source, parent, stage = [root / name for name in ('source.duckdb', 'parent.duckdb', 'stage.duckdb')]
    downloads = root / 'downloads'
    downloads.mkdir()
    with duckdb.connect(str(source)) as conn:
        conn.execute(f'CREATE TABLE {TABLE}({SCHEMA})')
        conn.execute(f"INSERT INTO {TABLE} VALUES ('a','2016-01-04',1,1,10),('b','2018-01-02',1,2,20)")
    ref = migrate_table(source, parent, TABLE, role='input', execute=True)['ref']
    bindings = {'input_db': str(parent), 'inputs': {TABLE: ref}}
    monkeypatch.setenv('QUANT_INPUT_PARENT_BINDINGS', json.dumps(bindings))
    monkeypatch.setenv('QUANT_INPUT_BATCH_STAGING', '1')
    monkeypatch.setenv('QUANT_INPUT_BATCH_REPLACE', '0')
    return parent, stage, downloads, ref


def download(root, rows):
    for year in [2016, 2017]:
        path = root / f'analyst_{year}.csv.gz'
        pd.DataFrame(rows if year == 2016 else [], columns=COLUMNS).to_csv(path, index=False)
        path.with_suffix('.manifest.json').write_text(json.dumps({
            'validated': True, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }))


@pytest.mark.parametrize('case,rows', [
    ('duplicate', [['c', '2016-01-05', 1, 3, 30], ['d', '2016-01-06', 1, 3, 40]]),
    ('null', [['c', '2016-01-05', 1, None, 30]]),
    ('parent_reuse', [['c', '2016-01-05', 1, 2, 30]]),
    ('discarded_parent_reuse', [['c', '2016-01-05', 1, 2, 30], ['c', '2016-01-05', 1, 4, 40]]),
])
def test_analyst_rejects_invalid_identity_before_seed(tmp_path, monkeypatch, case, rows):
    parent, stage, downloads, ref = setup_parent(tmp_path, monkeypatch)
    before = read_reference(parent, DataRef(**ref))
    download(downloads, rows)
    with pytest.raises(ValueError, match='source_id'):
        history.load_history('analyst', stage, downloads)
    with duckdb.connect(str(stage)) as conn:
        assert not conn.execute('SELECT 1 FROM information_schema.tables WHERE table_name=?', [TABLE]).fetchall()
    pd.testing.assert_frame_equal(before, read_reference(parent, DataRef(**ref)), check_exact=True)


@pytest.mark.parametrize('preexisting_stage', [False, True, 'primary_key'])
def test_analyst_retains_primary_key_and_retries(tmp_path, monkeypatch, preexisting_stage):
    parent, stage, downloads, ref = setup_parent(tmp_path, monkeypatch)
    before = read_reference(parent, DataRef(**ref))
    if preexisting_stage:
        with duckdb.connect(str(stage)) as conn:
            schema = SCHEMA if preexisting_stage == 'primary_key' else SCHEMA.replace(' PRIMARY KEY', '')
            conn.execute(f'CREATE TABLE {TABLE}({schema})')
    download(downloads, [['a', '2016-01-04', 1, 1, 10], ['c', '2016-01-05', 1, 3, 30], ['c', '2016-01-05', 1, 4, 40]])
    history.load_history('analyst', stage, downloads)
    history.load_history('analyst', stage, downloads)
    with duckdb.connect(str(stage)) as conn:
        assert _keys(conn, TABLE, conn.execute('SELECT current_database()').fetchone()[0]) == ['source_id']
        assert conn.execute("SELECT constraint_column_names FROM duckdb_constraints() WHERE database_name=current_database() AND constraint_type='PRIMARY KEY' AND table_name=?", [TABLE]).fetchall() == [(['source_id'],)]
        assert conn.execute(f'SELECT source_id,value FROM {TABLE} ORDER BY 1').fetchall() == [(1, 10), (4, 40)]
    args = dict(tables=[TABLE], metadata={'source': 'a222', 'source_hash': 'one', 'loader_hash': 'history'}, parents={TABLE: ref}, replace=False)
    published = publish_input_batch(parent, stage, **args)
    assert publish_input_batch(parent, stage, **args) == published
    after = read_reference(parent, DataRef(**published['results'][0]['ref']))
    assert sorted(after.source_id.tolist()) == [1, 2, 4]
    pd.testing.assert_frame_equal(before, read_reference(parent, DataRef(**ref)), check_exact=True)
    # A published parent is a view: identity must come from registered metadata.
    monkeypatch.setenv('QUANT_INPUT_PARENT_BINDINGS', json.dumps({
        'input_db': str(parent), 'inputs': {TABLE: published['results'][0]['ref']},
    }))
    view_stage = tmp_path / 'view_stage.duckdb'
    history.load_history('analyst', view_stage, downloads)
    with duckdb.connect(str(view_stage)) as conn:
        assert _keys(conn, TABLE, conn.execute('SELECT current_database()').fetchone()[0]) == ['source_id']
    download(downloads, [['a', '2016-01-04', 1, 1, 99]])
    with pytest.raises(AssertionError, match='conflicting existing natural keys'):
        history.load_history('analyst', tmp_path / 'conflict.duckdb', downloads)


@pytest.mark.parametrize('ids', [(9, 9), (9, None)])
def test_analyst_invalid_old_staging_rolls_back(tmp_path, monkeypatch, ids):
    parent, stage, downloads, ref = setup_parent(tmp_path, monkeypatch)
    with duckdb.connect(str(stage)) as conn:
        conn.execute(f'CREATE TABLE {TABLE}({SCHEMA.replace(" PRIMARY KEY", "")})')
        conn.executemany(f'INSERT INTO {TABLE} VALUES (?, ?, 1, ?, 30)', [
            ('c', '2016-01-05', ids[0]), ('d', '2016-01-06', ids[1]),
        ])
        before = conn.execute(f'SELECT * FROM {TABLE} ORDER BY stock_code').fetchdf()
    download(downloads, [['a', '2016-01-04', 1, 1, 10]])
    with pytest.raises(duckdb.ConstraintException):
        history.load_history('analyst', stage, downloads)
    with duckdb.connect(str(stage)) as conn:
        pd.testing.assert_frame_equal(before, conn.execute(f'SELECT * FROM {TABLE} ORDER BY stock_code').fetchdf(), check_exact=True)


def test_analyst_does_not_rebuild_dependent_old_staging(tmp_path, monkeypatch):
    parent, stage, downloads, ref = setup_parent(tmp_path, monkeypatch)
    with duckdb.connect(str(stage)) as conn:
        conn.execute(f'CREATE TABLE {TABLE}({SCHEMA.replace(" PRIMARY KEY", "")})')
        conn.execute(f'CREATE UNIQUE INDEX source_key ON {TABLE}(source_id)')
    download(downloads, [['a', '2016-01-04', 1, 1, 10]])
    # DuckDB cannot add a PK while an external index depends on the table.
    # Do not silently discard/rebuild caller-owned indexes to make it succeed.
    with pytest.raises(duckdb.DependencyException):
        history.load_history('analyst', stage, downloads)
    with duckdb.connect(str(stage)) as conn:
        assert conn.execute(f'SELECT count(*) FROM {TABLE}').fetchone()[0] == 0
        assert conn.execute("SELECT index_name FROM duckdb_indexes() WHERE index_name='source_key'").fetchall() == [('source_key',)]
