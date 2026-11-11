import hashlib
import json

import duckdb
import pandas as pd
import pytest

from data_pipeline.loaders.import_regime_history_duckdb import load_history
from utils.data_store import DataRef, migrate_table, read_reference
from utils.input_store import publish_input_batch


def download(root, kind, columns, rows):
    for year in ([2016, 2017] if kind == 'analyst' else range(2016, 2026)):
        frame = pd.DataFrame(rows if year == (2016 if kind == 'analyst' else 2019) else [], columns=columns)
        path = root / f'{kind}_{year}.csv.gz'
        frame.to_csv(path, index=False)
        path.with_suffix('.manifest.json').write_text(json.dumps({
            'validated': True, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }))


@pytest.mark.parametrize('kind', ['analyst', 'billboard'])
def test_history_preserves_parent_checks_and_retries(tmp_path, monkeypatch, kind):
    source, live, stage = (tmp_path / name for name in ('old.duckdb', 'input.duckdb', 'stage.duckdb'))
    downloads = tmp_path / 'downloads'
    downloads.mkdir()
    table = 'regime_all_a_analyst_consensus_daily' if kind == 'analyst' else 'billboard_events'
    with duckdb.connect(str(source)) as conn:
        if kind == 'analyst':
            conn.execute(f'CREATE TABLE {table}(stock_code VARCHAR, asof_date DATE, forecast_year_level INTEGER, source_id BIGINT, value DOUBLE, PRIMARY KEY(stock_code,asof_date,forecast_year_level))')
            conn.execute(f"INSERT INTO {table} VALUES ('a','2016-01-04',1,1,10),('b','2018-01-02',1,2,20)")
        else:
            conn.execute(f'CREATE TABLE {table}(stock_code VARCHAR,date DATE,direction VARCHAR,rank INTEGER,abnormal_code VARCHAR,sales_depart_name VARCHAR,buy_value DOUBLE,sell_value DOUBLE,amount DOUBLE,buy_rate DOUBLE,sell_rate DOUBLE)')
            conn.execute(f"INSERT INTO {table} VALUES ('a','2019-01-02','buy',1,'x','d',1,1,10,10,10),('a','2019-01-02','buy',2,'x','old',2,1,10,20,10),('a','2015-01-02','buy',1,'x','d',1,1,10,10,10),('a','2026-01-02','buy',1,'x','d',1,1,10,10,10),('unknown',NULL,'buy',1,'x','d',1,1,10,10,10)")
        columns = [row[0] for row in conn.execute(f'DESCRIBE {table}').fetchall()]
    original = migrate_table(source, live, table, role='input', execute=True)['ref']
    source.unlink()
    bindings = {'input_db': str(live), 'inputs': {table: original}}
    monkeypatch.setenv('QUANT_INPUT_PARENT_BINDINGS', json.dumps(bindings))
    monkeypatch.setenv('QUANT_INPUT_BATCH_STAGING', '1')
    monkeypatch.setenv('QUANT_INPUT_BATCH_REPLACE', '1' if kind == 'billboard' else '0')
    rows = ([['a', '2016-01-04', 1, 1, 10], ['c', '2016-01-05', 1, 3, 30], ['c', '2016-01-05', 1, 4, 40]]
            if kind == 'analyst' else [['a', '2019-01-02', 'buy', 1, 'x', 'd', 3, 2, 10, 999, 999]])
    download(downloads, kind, columns, rows)
    report = load_history(kind, stage, downloads)
    assert report['parent_ref'] == original and report['staging_only']
    assert report['snapshot_path'] is None
    with duckdb.connect(str(stage)) as conn:
        if kind == 'analyst':
            assert conn.execute(f'SELECT stock_code,value FROM {table} ORDER BY 1').fetchall() == [('a', 10), ('c', 40)]
        else:
            assert conn.execute(f'SELECT buy_rate,sell_rate FROM {table} WHERE date=\'2019-01-02\'').fetchall() == [(30, 20)]
    args = dict(tables=[table], metadata={'source': str(downloads), 'source_hash': 'one', 'loader_hash': 'history'}, parents=bindings['inputs'], replace=kind == 'billboard')
    result = publish_input_batch(live, stage, **args)
    assert publish_input_batch(live, stage, **args) == result
    before = read_reference(live, DataRef(**original))
    after = read_reference(live, DataRef(**result['results'][0]['ref']))
    if kind == 'analyst':
        assert before.sort_values('stock_code').value.tolist() == [10, 20]
        assert after.sort_values('stock_code').value.tolist() == [10, 20, 40]
        rows[0][-1] = 99
        download(downloads, kind, columns, rows)
        with pytest.raises(AssertionError, match='conflicting existing natural keys'):
            load_history(kind, tmp_path / 'conflict.duckdb', downloads)
        monkeypatch.setenv('QUANT_INPUT_BATCH_REPLACE', '1')
        with pytest.raises(ValueError, match='upsert publication'):
            load_history(kind, tmp_path / 'wrong-mode.duckdb', downloads)
    else:
        assert len(before) == 5 and len(after) == 4  # obsolete rank removed, outside/null dates retained
        pd.testing.assert_frame_equal(before[before.date.isna()].reset_index(drop=True), after[after.date.isna()].reset_index(drop=True))
        rows[0][0] = 'missing-old-day'
        download(downloads, kind, columns, rows)
        with pytest.raises(AssertionError, match='old event-days absent'):
            load_history(kind, tmp_path / 'conflict.duckdb', downloads)
        monkeypatch.setenv('QUANT_INPUT_BATCH_REPLACE', '0')
        with pytest.raises(ValueError, match='snapshot publication'):
            load_history(kind, tmp_path / 'wrong-mode.duckdb', downloads)
    pd.testing.assert_frame_equal(after, read_reference(live, DataRef(**result['results'][0]['ref'])))
    assert not source.exists()
