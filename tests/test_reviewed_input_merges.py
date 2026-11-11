import json

import duckdb
import pytest

from utils.data_store import DataRef, migrate_table, read_reference
from utils.input_consolidation import merge_input_tables
from utils.input_store import publish_input_batch


def heads(path):
    with duckdb.connect(str(path),read_only=True) as c:
        return {t.removeprefix('main.'):{'role':'input','table':t,'version':v}
                for t,v in c.execute('SELECT target_table,version FROM meta.input_heads').fetchall()}


def migrate(tmp_path, ddl, spec):
    old,live=tmp_path/'old.duckdb',tmp_path/'input.duckdb'
    with duckdb.connect(str(old)) as c:c.execute(ddl)
    refs={t:migrate_table(old,live,t,role='input',execute=True)['ref'] for t in ['items','regime_items']}
    spec={'target':'main.items','source':'main.regime_items',**spec}
    before={t:read_reference(live,DataRef(**r)) for t,r in refs.items()}
    assert merge_input_tables(live,spec)['ok']
    result=merge_input_tables(live,spec,execute=True)
    for t,r in refs.items():
        assert read_reference(live,DataRef(**r)).equals(before[t])
    assert merge_input_tables(live,spec,execute=True)['already_merged']
    return live,result


def publish(live,stage,table,version):
    args=dict(tables=[table],metadata={'source':'test','source_hash':version,'loader_hash':'test'},parents=heads(live))
    result=publish_input_batch(live,stage,**args)
    assert result==publish_input_batch(live,stage,**args)
    return result


@pytest.mark.parametrize('profile',['forecast','fund','price','restructuring'])
def test_reviewed_profiles_preserve_old_refs_and_import_fields(tmp_path,profile):
    if profile=='forecast':
        ddl="""CREATE TABLE items(stock_code VARCHAR,pub_date DATE,report_date DATE,forecast_type INTEGER,forecast_abstract VARCHAR,forecast_content VARCHAR,profit_min DOUBLE);
        INSERT INTO items VALUES ('000712.XSHE','2019-07-15','2019-06-30',1,'盈利','预计盈利',NULL);
        CREATE TABLE regime_items(stock_code VARCHAR,pub_date DATE,report_date DATE,forecast_type_id VARCHAR,forecast_type VARCHAR,forecast_content VARCHAR,profit_min DOUBLE);
        INSERT INTO regime_items VALUES ('000712.XSHE','2019-07-15','2019-06-30','1','盈利','预计盈利'||chr(13)||'100万元',1000000);"""
        spec={'profile':profile,'keys':['stock_code','pub_date','report_date','forecast_type'],'mapping':{'forecast_type_id':'forecast_type','forecast_type':'forecast_abstract'}}
    elif profile=='fund':
        ddl="""CREATE TABLE items(source_record_id BIGINT,stock_code VARCHAR,proportion DOUBLE,extra DOUBLE);
        INSERT INTO items VALUES (1,'600033',0,10);
        CREATE TABLE regime_items(source_id BIGINT,stock_code VARCHAR,proportion DOUBLE);
        INSERT INTO regime_items VALUES (1,'600033.XSHG',NULL),(2,'000001.XSHE',0.5);"""
        spec={'profile':profile,'keys':['source_record_id'],'mapping':{'source_id':'source_record_id'}}
    elif profile=='price':
        ddl="""CREATE TABLE items(stock_code VARCHAR,date DATE,close DOUBLE,money DOUBLE,paused INTEGER,open DOUBLE);
        INSERT INTO items VALUES ('600033.XSHG','2019-01-02',10,0,1,10);
        CREATE TABLE regime_items(stock_code VARCHAR,date DATE,close DOUBLE,money DOUBLE,paused DOUBLE);
        INSERT INTO regime_items VALUES ('600033.XSHG','2019-01-02',10,NULL,NULL),('000001.XSHE','2019-01-02',20,100,0);"""
        spec={'profile':profile,'keys':['stock_code','date']}
    else:
        detail=json.dumps({'ID':1,'EndDate':'2018-12-31 00:00:00','AgreementDate':None,'ChangeType':None,'FinalAmount':None,'InitialAmount':None})
        ddl=f"""CREATE TABLE items(source_table VARCHAR,source_id VARCHAR,stock_code VARCHAR,pub_date DATE,event_date DATE,event_type VARCHAR,amount VARCHAR,details_json VARCHAR);
        INSERT INTO items VALUES ('jy.LC_Merger','1','600033.XSHG','2019-03-01',NULL,'merger',NULL,'{detail}');
        CREATE TABLE regime_items(source_key VARCHAR,source_table VARCHAR,stock_code VARCHAR,event_date DATE,availability_date DATE,details_json VARCHAR);
        INSERT INTO regime_items VALUES ('jy.LC_Merger:1','jy.LC_Merger','600033.XSHG','2019-03-01','2019-03-01','{detail}');"""
        spec={'profile':profile,'keys':['source_table','source_id'],'alias_mapping':{'event_date':'availability_date','availability_date':'pub_date'}}
    live,result=migrate(tmp_path,ddl,spec)
    stage=tmp_path/'stage.duckdb'
    with duckdb.connect(str(stage)) as c:
        c.execute(ddl)
        if profile=='fund':c.execute('UPDATE regime_items SET proportion=NULL')
        if profile=='price':c.execute('UPDATE regime_items SET money=NULL,paused=NULL')
    publish(live,stage,'regime_items','new')
    if profile=='forecast':
        publish(live,stage,'items','older_csv')
    with duckdb.connect(str(live),read_only=True) as c:
        if profile=='forecast':
            assert c.execute('SELECT profit_min,forecast_abstract FROM items').fetchone()==(1000000,'盈利')
            assert c.execute('SELECT forecast_content FROM items').fetchone()[0].endswith('100万元')
        if profile=='fund':
            assert c.execute('SELECT stock_code,proportion,extra FROM items WHERE source_record_id=1').fetchone()==('600033.XSHG',0,10)
        if profile=='price':
            assert c.execute("SELECT money,paused,open FROM items WHERE stock_code='600033.XSHG'").fetchone()==(0,1,10)
        if profile=='restructuring':
            assert str(c.execute('SELECT event_date FROM items').fetchone()[0])=='2018-12-31'
            assert str(c.execute('SELECT event_date FROM regime_items').fetchone()[0])=='2019-03-01'


def test_shareholder_batches_keep_multiplicity_scope_names_and_provider_ids(tmp_path):
    ddl="""CREATE TABLE items(stock_code VARCHAR,pub_date DATE,end_date DATE,holder_scope VARCHAR,holder_rank INTEGER,shareholder_name VARCHAR,share_ratio DOUBLE,shareholder_class_id INTEGER);
    INSERT INTO items VALUES ('600033.XSHG','2019-03-01','2018-12-31','total',1,'甲',1,7),('600033.XSHG','2019-03-01','2018-12-31','total',2,'乙',1,7),('600033.XSHG','2019-03-01','2018-12-31','floating',1,'丙',2,7);
    CREATE TABLE regime_items(source_id VARCHAR,stock_code VARCHAR,pub_date DATE,report_date DATE,share_ratio DOUBLE,shareholder_class_id VARCHAR);
    INSERT INTO regime_items VALUES ('101','600033.XSHG','2019-03-01','2018-12-31',1,'7'),('102','600033.XSHG','2019-03-01','2018-12-31',1,'7'),('103','000001.XSHE','2019-03-01','2018-12-31',3,'8');"""
    spec={'profile':'shareholder','keys':['_input_row_id'],'alias_mapping':{'report_date':'end_date'}}
    live,result=migrate(tmp_path,ddl,spec)
    assert result['rows_after']==4
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT source_id FROM regime_items ORDER BY source_id').fetchall()==[('101',),('102',),('103',)]
        assert c.execute("SELECT count(*) FROM items WHERE holder_scope='floating'").fetchone()[0]==1
        assert c.execute("SELECT count(*) FROM items WHERE shareholder_name IS NOT NULL").fetchone()[0]==3
    stage=tmp_path/'stage.duckdb'
    with duckdb.connect(str(stage)) as c:
        c.execute(ddl)
        c.execute("DELETE FROM items; INSERT INTO items VALUES ('000001.XSHE','2019-03-01','2018-12-31','total',1,'补齐姓名',3,8)")
    publish(live,stage,'items','rich')
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT count(*) FROM items').fetchone()[0]==4
        assert c.execute("SELECT shareholder_name FROM items WHERE stock_code='000001.XSHE'").fetchone()[0]=='补齐姓名'
        assert c.execute('SELECT source_id FROM regime_items ORDER BY source_id').fetchall()==[('101',),('102',),('103',)]
    with duckdb.connect(str(stage)) as c:
        c.execute("INSERT INTO regime_items VALUES ('104','000002.XSHE','2019-04-01','2018-12-31',4,'8')")
    publish(live,stage,'regime_items','thin')
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT count(*) FROM items').fetchone()[0]==5
        assert c.execute('SELECT source_id FROM regime_items ORDER BY source_id').fetchall()==[('101',),('102',),('103',),('104',)]
    with duckdb.connect(str(stage)) as c:c.execute("UPDATE regime_items SET share_ratio=99 WHERE source_id='101'")
    with pytest.raises(ValueError,match='disclosure batch differs'):publish(live,stage,'regime_items','conflict')
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT count(*) FROM items').fetchone()[0]==5
        assert c.execute('SELECT max(share_ratio) FROM regime_items').fetchone()[0]==4


def test_reviewed_profile_still_refuses_conflicting_nonnull_prices(tmp_path):
    old,live=tmp_path/'old.duckdb',tmp_path/'input.duckdb'
    with duckdb.connect(str(old)) as c:
        c.execute('CREATE TABLE items(stock_code VARCHAR,date DATE,close DOUBLE); CREATE TABLE regime_items AS SELECT * FROM items')
        c.execute("INSERT INTO items VALUES ('600033.XSHG','2019-01-02',10); INSERT INTO regime_items VALUES ('600033.XSHG','2019-01-02',11)")
    for table in ['items','regime_items']:migrate_table(old,live,table,role='input',execute=True)
    result=merge_input_tables(live,{'target':'main.items','source':'main.regime_items','profile':'price','keys':['stock_code','date']},execute=True)
    assert not result['ok'] and result['differences']==1
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT close FROM items').fetchone()[0]==10
        assert c.execute('SELECT close FROM regime_items').fetchone()[0]==11


def test_security_web_resolution_preserves_snapshots_and_prevents_name_regression(tmp_path):
    ddl="""CREATE TABLE items(stock_code VARCHAR,display_name VARCHAR,name VARCHAR,start_date DATE,end_date DATE,security_type VARCHAR);
    INSERT INTO items VALUES ('600439.XSHG','瑞贝卡','RBK','2003-07-10','2200-01-01','stock');
    CREATE TABLE regime_items AS SELECT * FROM items;
    UPDATE regime_items SET display_name='ST瑞贝卡',name='STRBQ';
    INSERT INTO regime_items VALUES ('000001.XSHE','平安银行','PAYH','1991-04-03','2200-01-01','stock');"""
    resolution={'stock_code':'600439.XSHG','display_name':'ST瑞贝卡','name':'STRBK','accepted_names':[['瑞贝卡','RBK'],['ST瑞贝卡','STRBQ']],
                'checked_at':'2026-09-15T05:51:00+00:00','evidence_urls':['https://example.test/notice']}
    spec={'profile':'security','keys':['stock_code'],'name_resolutions':[resolution]}
    live,result=migrate(tmp_path,ddl,spec)
    assert result['rows_after']==2
    stage=tmp_path/'stage.duckdb'
    with duckdb.connect(str(stage)) as c:c.execute(ddl)
    publish(live,stage,'items','old-file')
    publish(live,stage,'regime_items','old-typo')
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute("SELECT display_name,name FROM items WHERE stock_code='600439.XSHG'").fetchone()==('ST瑞贝卡','STRBK')
        assert c.execute('SELECT * FROM items EXCEPT ALL SELECT * FROM regime_items').fetchall()==[]
    with duckdb.connect(str(stage)) as c:c.execute("UPDATE items SET display_name='未知更名'")
    with pytest.raises(ValueError,match='fresh evidence'):publish(live,stage,'items','future-unverified')
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute("SELECT name FROM items WHERE stock_code='600439.XSHG'").fetchone()[0]=='STRBK'


@pytest.mark.parametrize('dtype', ['VARCHAR', 'DECIMAL(3,1)'])
@pytest.mark.parametrize('source', [False, True])
def test_shareholder_normalize_rejects_fractional_class(dtype, source):
    from utils.input_merge_rules import normalize

    with duckdb.connect(':memory:') as conn:
        conn.execute('CREATE TABLE sample(source_id VARCHAR,report_date DATE,shareholder_class_id '+dtype+')')
        conn.execute("INSERT INTO sample VALUES ('1','2020-01-01','7.5')")
        with pytest.raises(ValueError, match='Lossy integer conversion: shareholder_class_id'):
            normalize(conn, 'sample', 'shareholder', source=source)
        assert conn.execute('SELECT CAST(shareholder_class_id AS VARCHAR) FROM sample').fetchone()[0] == '7.5'


@pytest.mark.parametrize('dtype', ['VARCHAR', 'DECIMAL(3,1)'])
def test_shareholder_merge_rejects_fractional_class_without_changes(tmp_path, dtype):
    old, live = tmp_path/'source.duckdb', tmp_path/'live.duckdb'
    with duckdb.connect(str(old)) as conn:
        conn.execute('CREATE TABLE items(stock_code VARCHAR,pub_date DATE,end_date DATE,holder_scope VARCHAR,share_ratio DOUBLE,shareholder_class_id INTEGER)')
        conn.execute("INSERT INTO items VALUES ('A','2020-03-01','2019-12-31','total',1,7)")
        conn.execute('CREATE TABLE regime_items(source_id VARCHAR,stock_code VARCHAR,pub_date DATE,report_date DATE,share_ratio DOUBLE,shareholder_class_id '+dtype+')')
        conn.execute("INSERT INTO regime_items VALUES ('101','B','2020-03-01','2019-12-31',2,'7.5')")
    refs = {t:migrate_table(old,live,t,role='input',execute=True)['ref'] for t in ['items','regime_items']}
    with duckdb.connect(str(live),read_only=True) as conn:
        names = conn.execute("SELECT table_schema,table_name FROM information_schema.tables ORDER BY ALL").fetchall()
        before = {(s,t):conn.execute(f'SELECT * FROM "{s}"."{t}" ORDER BY ALL').fetchall() for s,t in names}
    spec = {'target':'main.items','source':'main.regime_items','profile':'shareholder','keys':['_input_row_id'],'alias_mapping':{'report_date':'end_date'}}
    for execute in [False, True]:
        with pytest.raises(ValueError, match='Lossy integer conversion: shareholder_class_id'):
            merge_input_tables(live, spec, execute=execute)
        with duckdb.connect(str(live),read_only=True) as conn:
            names = conn.execute("SELECT table_schema,table_name FROM information_schema.tables ORDER BY ALL").fetchall()
            after = {(s,t):conn.execute(f'SELECT * FROM "{s}"."{t}" ORDER BY ALL').fetchall() for s,t in names}
        assert before == after
    assert str(read_reference(live,DataRef(**refs['regime_items'])).shareholder_class_id.iloc[0]) == '7.5'
