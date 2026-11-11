"""Verified history import: analyst append; user-authorized full-A billboard replacement with snapshot."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import duckdb
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.data_revision import advance_revision

from utils.input_store import _keys, attach_input_parent, seed_input_rows

ROOT = Path('./data/import/regime_history_missing_2016_2018')

def load_history(kind, db, source=ROOT):
    source = Path(source)
    if kind not in {'analyst', 'billboard'}:
        raise ValueError('Unknown history dataset')
    bindings = json.loads(os.environ.get('QUANT_INPUT_PARENT_BINDINGS', '{}'))
    versioned = os.environ.get('QUANT_INPUT_BATCH_STAGING') == '1'
    if versioned != bool(bindings):
        raise ValueError('History staging requires a frozen parent binding')
    if versioned and kind == 'billboard' and os.environ.get('QUANT_INPUT_BATCH_REPLACE') != '1':
        raise ValueError('Bounded billboard replacement requires snapshot publication')
    if versioned and kind == 'analyst' and os.environ.get('QUANT_INPUT_BATCH_REPLACE') != '0':
        raise ValueError('Analyst history requires upsert publication to preserve unrelated rows')
    recovery = Path(db).parent / 'history_receipts'
    recovery.mkdir(parents=True, exist_ok=True)
    table, day, cutoff = ('regime_all_a_analyst_consensus_daily', 'asof_date', '2018-01-01') if kind == 'analyst' else ('billboard_events', 'date', '2026-01-01')
    years = [2016, 2017] if kind == 'analyst' else list(range(2016,2026))
    files = [source / f'{kind}_{y}.csv.gz' for y in years]
    for path in files:
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        assert manifest['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest(), path
        assert manifest['validated'] is True, path
    with duckdb.connect(str(db)) as con:
        con.execute('SET threads=2')
        parent = attach_input_parent(con, table, bindings=bindings) if versioned else table
        cols = con.execute(f'DESCRIBE {parent}').fetchall()
        expressions=[]
        for name, dtype, *_ in cols:
            if kind=='billboard' and name in ['buy_rate','sell_rate']:
                value=name.replace('_rate','_value')
                expressions.append(f'round(100.0*CAST({value} AS DOUBLE)/nullif(CAST(amount AS DOUBLE),0),4) AS {name}')
            else:
                expressions.append(f'CAST("{name}" AS {dtype}) AS "{name}"')
        casts=', '.join(expressions)
        con.execute(f'CREATE TEMP TABLE incoming AS SELECT {casts} FROM read_csv(?, header=true, all_varchar=true)', [[str(p) for p in files]])
        assert con.execute(f"SELECT count(*) FROM incoming WHERE {day} IS NULL OR stock_code IS NULL OR {day} < DATE '2016-01-01' OR {day} >= DATE '{cutoff}'").fetchone()[0] == 0
        if kind == 'analyst':
            con.execute('CREATE TEMP TABLE dedup AS SELECT * FROM incoming QUALIFY row_number() OVER (PARTITION BY stock_code,asof_date,forecast_year_level ORDER BY source_id DESC)=1')
        else:
            con.execute('CREATE TEMP TABLE dedup AS SELECT DISTINCT * FROM incoming')
            assert con.execute('SELECT count(*) FROM (SELECT stock_code,date,direction,rank,abnormal_code,sales_depart_name FROM dedup GROUP BY ALL HAVING count(*)>1)').fetchone()[0] == 0
        key = ['stock_code', day, 'forecast_year_level'] if kind == 'analyst' else ['stock_code','date','direction','rank','abnormal_code','sales_depart_name']
        source_id_key = False
        if versioned:
            if kind == 'analyst':
                # Preserve the parent's registered identity, including parents
                # published as views whose constraints are held in metadata.
                with duckdb.connect(bindings['input_db'], read_only=True) as frozen:
                    has_keys = frozen.execute("SELECT 1 FROM information_schema.tables WHERE table_schema='meta' AND table_name='input_keys'").fetchone()
                    stored = frozen.execute('SELECT columns_json FROM meta.input_keys WHERE target_table=?', ['main.' + table]).fetchone() if has_keys else None
                    registered = json.loads(stored[0]) if stored else _keys(frozen, table, frozen.execute('SELECT current_database()').fetchone()[0])
                source_id_key = registered == ['source_id']
                if source_id_key:
                    if con.execute('SELECT 1 FROM incoming WHERE source_id IS NULL LIMIT 1').fetchone():
                        raise ValueError('Null analyst source_id')
                    if con.execute('SELECT source_id FROM (SELECT DISTINCT source_id,stock_code,asof_date,forecast_year_level FROM incoming) GROUP BY source_id HAVING count(*)>1 LIMIT 1').fetchone():
                        raise ValueError('Conflicting analyst source_id natural keys')
                    same_key = ' AND '.join(f'd."{k}" IS NOT DISTINCT FROM p."{k}"' for k in key)
                    if con.execute(f'SELECT 1 FROM incoming d JOIN {parent} p ON d.source_id=p.source_id WHERE NOT ({same_key}) LIMIT 1').fetchone():
                        raise ValueError('Analyst source_id reused for a different frozen-parent natural key')
                    # Include staging DDL and seeding in the existing transaction.
                    con.execute('BEGIN TRANSACTION')
                    con.execute(f'CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM {parent} WHERE false')
                    staging_keys = _keys(con, table, con.execute('SELECT current_database()').fetchone()[0])
                    if staging_keys and staging_keys != ['source_id']:
                        raise ValueError('Analyst staging key differs from registered source_id')
                    primary = con.execute("SELECT 1 FROM duckdb_constraints() WHERE database_name=current_database() AND schema_name='main' AND table_name=? AND constraint_type='PRIMARY KEY'", [table]).fetchone()
                    if not primary:
                        con.execute(f'ALTER TABLE {table} ADD PRIMARY KEY (source_id)')
                seed_input_rows(con, table, 'dedup', key, bindings=bindings)
            else:
                # The frozen parent preserves replaced history; only retain rows outside the replacement window in staging.
                con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM {parent} WHERE {day} IS NULL OR {day}>=DATE '{cutoff}' OR {day}<DATE '2016-01-01'")
        join = ' AND '.join(f'd."{k}" IS NOT DISTINCT FROM t."{k}"' for k in key)
        same = ' AND '.join(f'd."{c[0]}" IS NOT DISTINCT FROM t."{c[0]}"' for c in cols)
        if not source_id_key:
            con.execute('BEGIN TRANSACTION')
        if kind=='analyst':
            assert con.execute(f'SELECT count(*) FROM dedup d JOIN {table} t ON {join} WHERE NOT ({same})').fetchone()[0] == 0, 'conflicting existing natural keys'
        before = con.execute(f'SELECT count(*),sum(hash(t)) FROM {table if kind == "analyst" else parent} t').fetchone()
        old = con.execute(f"SELECT count(*),sum(hash(t)) FROM {parent} t WHERE {day} IS NULL OR {day}>=DATE '{cutoff}' OR {day}<DATE '2016-01-01'").fetchone() if kind == 'billboard' or not versioned else None
        snapshot=None
        if kind=='billboard':
            missing=con.execute(f"SELECT DISTINCT stock_code,date FROM {parent} WHERE date BETWEEN '2016-01-01' AND '2025-12-31' EXCEPT SELECT stock_code,date FROM dedup").fetchall()
            (recovery/'billboard_old_days_missing_in_download.json').write_text(json.dumps(missing,default=str,indent=2))
            assert not missing, 'old event-days absent from current full-A download; inspect before replacing'
            snapshot=None if versioned else source/'billboard_before_full_a_20260911.parquet'
            if snapshot is not None and not snapshot.exists():
                con.execute('COPY billboard_events TO ? (FORMAT PARQUET)',[str(snapshot)])
                assert before==con.execute('SELECT count(*),sum(hash(t)) FROM read_parquet(?) t',[str(snapshot)]).fetchone()
            con.execute("DELETE FROM billboard_events WHERE date BETWEEN '2016-01-01' AND '2025-12-31'")
        con.execute(f'INSERT INTO {table} SELECT * FROM dedup EXCEPT SELECT * FROM {table}')
        if old is not None:
            assert old == con.execute(f"SELECT count(*),sum(hash(t)) FROM {table} t WHERE {day} IS NULL OR {day}>=DATE '{cutoff}' OR {day}<DATE '2016-01-01'").fetchone(), 'existing history changed'
        after = con.execute(f'SELECT count(*),sum(hash(t)) FROM {table} t').fetchone()
        assert con.execute(f'SELECT count(*) FROM (SELECT * FROM dedup EXCEPT SELECT * FROM {table})').fetchone()[0] == 0
        coverage = con.execute(f'SELECT year({day}),count(*),count(DISTINCT stock_code),min({day}),max({day}) FROM {table} WHERE {day}<DATE \'{cutoff}\' GROUP BY 1 ORDER BY 1').fetchall()
        advance_revision(con)
        con.execute('COMMIT')
    result = dict(table=table, before=before, after=after, inserted=None if versioned and kind=='analyst' else after[0]-before[0], existing_history_preserved=(kind=='analyst'), outside_replacement_window_preserved=True, coverage=coverage, mode='append' if kind=='analyst' else 'replace_2016_2025_full_a', snapshot_path=str(snapshot) if snapshot else None, parent_ref=bindings.get('inputs',{}).get(table), staging_only=versioned)
    (recovery / f'import_{kind}.json').write_text(json.dumps(result, indent=2, default=str))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['analyst', 'billboard'])
    parser.add_argument('--db', required=True)
    parser.add_argument('--source', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(load_history(args.kind, args.db, args.source), default=str))

if __name__ == '__main__':
    main()
