"""Only unconditional A230 approvals; production source preconditions are separate."""
import csv
from pathlib import Path

import duckdb
import pytest
from pandas.testing import assert_frame_equal

from data_pipeline.loaders import factor_data_to_duckdb as factor
from data_pipeline.loaders import price_data_to_duckdb as price
from data_pipeline.loaders import import_index_constituents_duckdb as index
from data_pipeline.loaders import portfolio_data_to_duckdb as portfolio

PRICE_HEADER = 'date,open,close,high,low,volume,money'
PRICE_ROW = '2024-01-02,10,11,12,9,100,1000'
FIELDS = {
    'security_master': ['stock_code','display_name','name','start_date','end_date','security_type'],
    'historical_st': ['stock_code','date','is_st'],
    'shareholder_top10': ['stock_code','pub_date','end_date','report_type','holder_scope','holder_rank','shareholder_name','shareholder_class_id','share_number','share_ratio'],
    'valuation': ['stock_code','date',*portfolio.VALUATION_FIELDS],
}
ROWS = {
    'security_master': ['A','Display','NAME','2000-01-01','2200-01-01','stock'],
    'historical_st': ['A','2024-01-02','false'],
    'shareholder_top10': ['A','2024-04-30','2024-03-31','0','all','1','Holder','2','100','10'],
    'valuation': ['A','2024-01-02',*range(1,10)],
}

def write_portfolio(root, key):
    root.mkdir(parents=True, exist_ok=True)
    p = root / portfolio.DATASETS[key].csv_glob.replace('*','2024')
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(FIELDS[key]);w.writerow(ROWS[key])
    return root

@pytest.mark.parametrize('batch_size', [1,32])
@pytest.mark.parametrize('key_only', [False, True])
def test_factor_absent_columns_preserve_per_file(tmp_path, monkeypatch, batch_size, key_only):
    monkeypatch.setattr(factor,'BULK_BATCH_SIZE',batch_size)
    src=tmp_path/'csv';src.mkdir()
    imp=factor.DuckDBFactorImporter(str(tmp_path/'db'))
    try:
        imp.init_factor_table({'alpha':'DOUBLE','beta':'DOUBLE','old_only':'DOUBLE'})
        imp.conn.execute("INSERT INTO factor_data VALUES ('A','2024-01-02',1,2,3),('B','2024-01-02',4,5,6)")
        (src/'A.csv').write_text('date\n2024-01-02\n2024-01-03\n' if key_only else 'date,alpha\n2024-01-02,10\n2024-01-03,11\n')
        (src/'B.csv').write_text('date,beta\n2024-01-02,\n2024-01-03,20\n')
        assert imp.import_csv_files(str(src))==(2,2,4)
        assert imp.conn.execute('SELECT * FROM factor_data ORDER BY stock_code,date').fetchall()==[
            ('A','2024-01-02',1 if key_only else 10,2,3),('A','2024-01-03',None if key_only else 11,None,None),
            ('B','2024-01-02',4,None,6),('B','2024-01-03',None,20,None)]
    finally:imp.close()

@pytest.mark.parametrize('kind',['factor','index','daily','5m','daily_unpartitioned'])
def test_quoted_export_path(tmp_path,kind):
    out=tmp_path/"owner's parquet"
    if kind=='factor':
        imp=factor.DuckDBFactorImporter(str(tmp_path/'db'),str(out),True);imp.init_factor_table({'alpha':'DOUBLE'})
        imp.conn.execute("INSERT INTO factor_data VALUES ('A','2024-01-02',42)");table='factor_data';export=imp.export_to_parquet
    elif kind=='index':
        imp=index.DuckDBIndexConstituentsImporter(str(tmp_path/'db'),str(out),True);imp.init_table()
        imp.conn.execute("INSERT INTO index_constituents VALUES ('2024-01-02','A','I')");table='index_constituents';export=imp.export_to_parquet
    else:
        imp=price.DuckDBPriceImporter(str(tmp_path/'db'),str(out),True);imp.init_tables()
        table='price_data_5m' if kind=='5m' else 'price_data_daily'
        imp.conn.execute(f"INSERT INTO {table}(stock_code,date,open) VALUES ('A','2024-01-02',42)")
        export=lambda:imp.export_to_parquet(table,None if kind=='daily_unpartitioned' else ['year','month'])
    try:
        export();paths=list((out/table).rglob('*.parquet'));assert paths
        assert imp.conn.execute('SELECT stock_code FROM read_parquet(?)',[[str(p) for p in paths]]).fetchall()==[('A',)]
    finally:imp.close()

@pytest.mark.parametrize('key', FIELDS)
def test_portfolio_quoted_source(tmp_path,key):
    with duckdb.connect() as c:
        spec=portfolio.DATASETS[key]
        portfolio.import_dataset(c,spec,write_portfolio(tmp_path/"owner's source",key),False)
        assert c.execute('SELECT count(*) FROM '+spec.table).fetchone()==(1,)

@pytest.mark.parametrize('namespace',['schema','catalog'])
def test_factor_migration_uses_resolved_table(tmp_path,namespace):
    imp=price.DuckDBPriceImporter(str(tmp_path/'db'));imp.connect()
    try:
        if namespace=='schema':imp.conn.execute('CREATE SCHEMA other');other='other'
        else:imp.conn.execute("ATTACH ':memory:' AS other");other='other.main'
        imp.conn.execute(f'CREATE TABLE {other}.price_data_daily(stock_code VARCHAR,factor DOUBLE)')
        imp.conn.execute('CREATE TABLE price_data_daily(stock_code VARCHAR)');imp.conn.execute("INSERT INTO price_data_daily VALUES ('A')")
        imp._migrate_add_factor_column('price_data_daily')
        assert imp.conn.execute('SELECT stock_code,factor FROM price_data_daily').fetchall()==[('A',None)]
    finally:imp.close()

@pytest.mark.parametrize('case',['missing','view','Factor'])
def test_migration_true_errors_and_case_identity(tmp_path,case):
    imp=price.DuckDBPriceImporter(str(tmp_path/'db'));imp.connect()
    try:
        if case=='view':imp.conn.execute('CREATE VIEW price_data_daily AS SELECT 1 AS x')
        if case=='Factor':
            imp.conn.execute('CREATE TABLE price_data_daily(Factor DOUBLE)');imp.conn.execute('INSERT INTO price_data_daily VALUES (.5)')
            imp._migrate_add_factor_column('price_data_daily')
            assert imp.conn.execute('SELECT * FROM price_data_daily').fetchall()==[(.5,)]
        else:
            with pytest.raises(duckdb.Error):imp._migrate_add_factor_column('price_data_daily')
    finally:imp.close()

@pytest.mark.parametrize('freq',['daily','5m'])
@pytest.mark.parametrize('suffix',['CSV','Csv'])
def test_price_suffix_preserves_stock_code(tmp_path,freq,suffix):
    src=tmp_path/'src';src.mkdir();(src/('A.'+suffix)).write_text(PRICE_HEADER+'\n'+PRICE_ROW+'\n')
    imp=price.DuckDBPriceImporter(str(tmp_path/'db'));imp.init_tables()
    try:
        assert imp.import_directory(freq,str(src),'price_data_'+freq)==(1,1,1)
        assert imp.conn.execute('SELECT stock_code FROM price_data_'+freq).fetchall()==[('A',)]
    finally:imp.close()

@pytest.mark.parametrize('freq',['daily','5m'])
def test_case_variant_duplicate_filename_rejected_before_write(tmp_path,monkeypatch,freq):
    src=tmp_path/'src';src.mkdir();(src/'A.csv').write_text(PRICE_HEADER+'\n'+PRICE_ROW+'\n')
    # Works on case-insensitive filesystems too; the directory enumeration is the contract.
    monkeypatch.setattr(price.os,'listdir',lambda p:['A.csv','A.CSV'])
    imp=price.DuckDBPriceImporter(str(tmp_path/'db'));imp.init_tables()
    touched=[];monkeypatch.setattr(imp,'_bulk_import_batch',lambda *a:touched.append(a))
    try:
        with pytest.raises(ValueError,match='[Dd]uplicate|[Aa]mbiguous'):
            imp.import_directory(freq,str(src),'price_data_'+freq)
        assert not touched
        assert imp.conn.execute('SELECT count(*) FROM import_progress').fetchone()==(0,)
    finally:imp.close()
