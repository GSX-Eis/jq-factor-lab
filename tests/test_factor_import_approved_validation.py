import duckdb
import pytest
import data_pipeline.loaders.factor_data_to_duckdb as module

@pytest.mark.parametrize('bad',['garbage','2024-02-30','01/02/2024','2024-1-2','2024-01-02 extra'])
def test_nonempty_invalid_dates_rollback(tmp_path,bad):
    src=tmp_path/'csv';src.mkdir();(src/'A.csv').write_text(f'date,alpha\n2024-01-02,1\n{bad},2\n')
    imp=module.DuckDBFactorImporter(str(tmp_path/'test.duckdb'));imp.connect()
    try:
        imp.init_factor_table({'date':'VARCHAR','alpha':'DOUBLE'})
        imp.conn.execute("INSERT INTO factor_data VALUES ('A','2024-01-02',9)")
        with pytest.raises(ValueError,match='date'):imp.import_csv_files(str(src))
        assert imp.conn.execute('SELECT * FROM factor_data').fetchall()==[('A','2024-01-02',9.)]
    finally:imp.close()

def test_missing_dates_are_counted_without_return_change(tmp_path,capsys):
    src=tmp_path/'csv';src.mkdir();(src/'A.csv').write_text('date,alpha\n2024-01-02,1\n,2\n   ,3\n')
    imp=module.DuckDBFactorImporter(str(tmp_path/'test.duckdb'))
    try:
        assert imp.import_csv_files(str(src))==(1,1,1)
        assert 'Dropped 2 rows with missing or empty date' in capsys.readouterr().out
    finally:imp.close()

@pytest.mark.parametrize('values',['1\n2024-01-02,2','\n2024-01-02,2','nan\n2024-01-02,2'])
def test_conflicting_duplicate_rolls_back_prior_batch(tmp_path,monkeypatch,values):
    src=tmp_path/'csv';src.mkdir()
    (src/'A.csv').write_text('date,alpha\n2024-01-02,2\n')
    (src/'B.csv').write_text('date,alpha\n2024-01-02,'+values+'\n')
    monkeypatch.setattr(module,'BULK_BATCH_SIZE',1)
    imp=module.DuckDBFactorImporter(str(tmp_path/'test.duckdb'));imp.connect()
    try:
        imp.init_factor_table({'date':'VARCHAR','alpha':'DOUBLE'})
        imp.conn.execute("INSERT INTO factor_data VALUES ('A','2024-01-02',9)")
        with pytest.raises(ValueError,match='Conflicting'):imp.import_csv_files(str(src))
        assert imp.conn.execute('SELECT * FROM factor_data').fetchall()==[('A','2024-01-02',9.)]
    finally:imp.close()

@pytest.mark.parametrize('value',['1','','nan'])
def test_identical_payload_still_deduplicates(tmp_path,value):
    src=tmp_path/'csv';src.mkdir();(src/'A.csv').write_text(f'date,alpha\n2024-01-02,{value}\n2024-01-02,{value}\n')
    imp=module.DuckDBFactorImporter(str(tmp_path/'test.duckdb'))
    try:
        assert imp.import_csv_files(str(src))==(1,1,1)
    finally:imp.close()
