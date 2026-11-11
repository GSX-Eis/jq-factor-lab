"""run_import releases its owned connection without changing commit boundaries."""
import duckdb
import pytest

from data_pipeline.loaders.price_data_to_duckdb import DuckDBPriceImporter


@pytest.mark.parametrize('failure', ['connect', 'init', 'daily', '5m', 'interrupt', 'export'])
def test_run_import_closes_on_failure(tmp_path, monkeypatch, failure):
    daily, minute = tmp_path/'daily', tmp_path/'5m'
    daily.mkdir(); minute.mkdir()
    header = 'date,open,close,high,low,volume,money\n'
    (daily/'A.csv').write_text(header+'2024-01-02,10,11,12,9,100,1000\n')
    (minute/'A.csv').write_text(header+'2024-01-02 09:35:00,10,11,12,9,100,1000\n')
    importer = DuckDBPriceImporter(str(tmp_path/'input.duckdb'),str(tmp_path/'parquet'),failure=='export')
    connect = importer.connect
    handles = []
    def capture_connection():
        conn = connect()
        handles.append(conn)
        if failure == 'connect':
            raise RuntimeError('injected connection configuration failure')
        return conn
    monkeypatch.setattr(importer,'connect',capture_connection)
    expected = RuntimeError
    if failure == 'init':
        def fail_init(): raise RuntimeError('injected initialization failure')
        monkeypatch.setattr(importer,'init_tables',fail_init)
    elif failure in ['daily','5m']:
        folder = daily if failure=='daily' else minute
        (folder/'A.csv').write_text('bad\n1\n')
        expected = ValueError
    elif failure == 'interrupt':
        bulk = importer._bulk_import_batch
        def interrupted(files,table,freq):
            if freq=='5m': raise KeyboardInterrupt('injected 5m interruption')
            return bulk(files,table,freq)
        monkeypatch.setattr(importer,'_bulk_import_batch',interrupted)
        expected = KeyboardInterrupt
    elif failure == 'export':
        (importer.parquet_dir/'price_data_daily').write_text('blocks export directory')
        expected = FileExistsError
    try:
        with pytest.raises(expected):
            importer.run_import(str(minute),str(daily))
        assert importer.conn is None
        with pytest.raises(duckdb.ConnectionException):
            handles[0].execute('SELECT 1')
        with duckdb.connect(str(importer.db_path)) as conn:
            if failure in ['connect','init']:
                assert not conn.execute("SELECT 1 FROM information_schema.tables WHERE table_name LIKE 'price_data_%'").fetchall()
            else:
                expected_daily = 0 if failure=='daily' else 1
                expected_minute = 1 if failure=='export' else 0
                assert conn.execute('SELECT count(*) FROM price_data_daily').fetchone()[0] == expected_daily
                assert conn.execute('SELECT count(*) FROM price_data_5m').fetchone()[0] == expected_minute
                assert conn.execute('SELECT count(*) FROM import_progress').fetchone()[0] == expected_daily+expected_minute
    finally:
        importer.close()
