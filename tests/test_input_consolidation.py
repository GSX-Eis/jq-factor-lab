import json

import duckdb
import pytest

from utils.data_store import DataRef, migrate_table, read_reference, connect_input
from utils.input_consolidation import merge_input_tables
from utils.input_store import publish_input_batch


def setup_pair(tmp_path, conflict=False):
    old, live = tmp_path/'old.duckdb', tmp_path/'input.duckdb'
    with duckdb.connect(str(old)) as c:
        c.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, value DOUBLE, extra DOUBLE)')
        c.execute('INSERT INTO items VALUES (1,10,100),(2,20,200)')
        c.execute('CREATE TABLE regime_items (id BIGINT PRIMARY KEY, value DOUBLE, run_id VARCHAR)')
        c.execute('INSERT INTO regime_items VALUES (1,?,\'r\'),(3,30,\'r\')',[99 if conflict else 10])
    refs={t:migrate_table(old,live,t,role='input',execute=True)['ref'] for t in ['items','regime_items']}
    return live,refs,{'target':'main.items','source':'main.regime_items','keys':['id']}


def current_refs(live):
    with duckdb.connect(str(live),read_only=True) as c:
        return {t.removeprefix('main.'):{'role':'input','table':t,'version':v}
                for t,v in c.execute('SELECT target_table,version FROM meta.input_heads').fetchall()}


def test_merge_then_old_and_new_loaders_publish_to_one_head(tmp_path):
    live,original,spec=setup_pair(tmp_path)
    with duckdb.connect(str(live)) as c:
        c.execute('CREATE INDEX item_values ON items(value)')
        c.execute('CREATE UNIQUE INDEX item_identity ON items(id)')
    assert merge_input_tables(live,spec)['differences']==0
    merged=merge_input_tables(live,spec,execute=True)
    assert merged['rows_after']==3
    with duckdb.connect(str(live),read_only=True) as c:
        indexes=c.execute("SELECT index_name,is_unique,table_name FROM duckdb_indexes() WHERE index_name LIKE '__input_index_%'").fetchall()
        assert len(indexes)==2 and all(t.startswith('__input_base_') for _,_,t in indexes)
        assert sorted(u for _,u,_ in indexes)==[False,True]
    assert merge_input_tables(live,spec,execute=True)['already_merged']
    assert read_reference(live,DataRef(**original['items'])).value.tolist()==[10,20]
    assert read_reference(live,DataRef(**original['regime_items'])).value.tolist()==[10,30]
    frozen=current_refs(live)
    with connect_input(live,bindings={'input_db':str(live),'inputs':frozen}) as c:
        assert c.execute('SELECT count(*) FROM items').fetchone()[0]==3
        assert c.execute('SELECT count(*) FROM regime_items').fetchone()[0]==3
        assert [r[0] for r in c.execute('DESCRIBE regime_items').fetchall()]==['id','value','run_id']
    stage=tmp_path/'stage.duckdb'
    with duckdb.connect(str(stage)) as c:
        c.execute("CREATE TABLE regime_items (id BIGINT PRIMARY KEY,value DOUBLE,run_id VARCHAR)")
        c.execute("INSERT INTO regime_items VALUES (1,11,'new'),(4,40,'new')")
    metadata={'source':'source','source_hash':'one','loader_hash':'loader'}
    args=dict(tables=['regime_items'],metadata=metadata,parents=frozen)
    result=publish_input_batch(live,stage,**args)
    assert result['results'][0]['ref']['table']=='main.items'
    assert publish_input_batch(live,stage,**args)==result
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT id,value,extra FROM items ORDER BY id').fetchall()==[(1,11,100),(2,20,200),(3,30,None),(4,40,None)]
        assert c.execute('SELECT value FROM regime_items WHERE id=1').fetchone()[0]==11
        assert c.execute("SELECT table_type FROM information_schema.tables WHERE table_schema='main' AND table_name='regime_items'").fetchone()[0]=='VIEW'
    # The frozen alias from before import still resolves the previous snapshot.
    assert sorted(read_reference(live,DataRef(**frozen['regime_items'])).value)==[10,20,30]
    with pytest.raises(ValueError,match='erase other coverage'):
        publish_input_batch(live,stage,**{**args,'replace':True})
    with duckdb.connect(str(stage)) as c:
        c.execute("CREATE TABLE items (id INTEGER PRIMARY KEY,value DOUBLE,extra DOUBLE)")
        c.execute('INSERT INTO items VALUES (1,12,101)')
    new=publish_input_batch(live,stage,tables=['items'],metadata={**metadata,'source_hash':'two'},parents=current_refs(live))
    # Retrying the older alias batch must not roll either head back.
    assert publish_input_batch(live,stage,**args)==result
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT value FROM regime_items WHERE id=1').fetchone()[0]==12
        assert c.execute('SELECT run_id FROM items WHERE id=1').fetchone()[0]=='new'
    with duckdb.connect(str(stage)) as c:c.execute('UPDATE regime_items SET value=999 WHERE id=1')
    with pytest.raises(ValueError,match='changed after publication'):
        publish_input_batch(live,stage,**args)
    assert read_reference(live,DataRef(**original['items'])).value.tolist()==[10,20]


def test_merge_refuses_conflicts_and_preserves_data(tmp_path):
    live,refs,spec=setup_pair(tmp_path,conflict=True)
    assert merge_input_tables(live,spec,execute=True)['differences']==1
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT count(*) FROM meta.input_merges').fetchone()[0]==0
        assert c.execute('SELECT value FROM regime_items WHERE id=1').fetchone()[0]==99


def test_nullable_event_keys_and_explicit_field_mapping(tmp_path):
    old,live=tmp_path/'old.duckdb',tmp_path/'input.duckdb'
    with duckdb.connect(str(old)) as c:
        c.execute('CREATE TABLE events (id VARCHAR, end_date DATE, value VARCHAR)')
        c.execute("INSERT INTO events VALUES ('1',NULL,'10'),('2','2019-01-01','20')")
        c.execute('CREATE TABLE regime_events (source_id BIGINT, end_date DATE, value DOUBLE)')
        c.execute("INSERT INTO regime_events VALUES (1,NULL,10),(3,NULL,30)")
    for t in ['events','regime_events']:migrate_table(old,live,t,role='input',execute=True)
    spec={'target':'main.events','source':'main.regime_events','mapping':{'source_id':'id'},
          'keys':['id','end_date'],'types':{'id':'BIGINT','value':'DOUBLE'}}
    assert merge_input_tables(live,spec,execute=True)['rows_after']==3
    stage=tmp_path/'stage.duckdb'
    with duckdb.connect(str(stage)) as c:
        c.execute('CREATE TABLE regime_events (source_id BIGINT,end_date DATE,value DOUBLE)')
        c.execute('INSERT INTO regime_events VALUES (1,NULL,11)')
    publish_input_batch(live,stage,tables=['regime_events'],metadata={'source':'s','source_hash':'h','loader_hash':'l'},parents=current_refs(live))
    with duckdb.connect(str(live),read_only=True) as c:
        assert c.execute('SELECT id,value FROM events ORDER BY id').fetchall()==[(1,11),(2,20),(3,30)]
