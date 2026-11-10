"""Consolidate coverage-only inputs while keeping exact historical references."""
from __future__ import annotations

import json
from pathlib import Path

from utils.data_store import (
    _exists, _set_role, audit_relation, canonical, describe, digest,
    input_catalog_name, quote, table_name, write_store,
)
from utils.input_store import _catalog, _parent, _preserve_base

from utils.input_merge_rules import PROFILES, GROUP, normalize, holder_union, text_extension

METADATA_COLUMNS = {'run_id', 'partition_year'}


def aliases(conn):
    if not _exists(conn, 'meta.input_aliases'):
        return {}
    return {a: json.loads(body) for a, body in conn.execute(
        'SELECT alias_table, rule_json FROM meta.input_aliases'
    ).fetchall()}


def _catalog_aliases(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS meta.input_aliases '
                 '(alias_table VARCHAR PRIMARY KEY, target_table VARCHAR NOT NULL, rule_json VARCHAR NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS meta.input_merges '
                 '(merge_id VARCHAR PRIMARY KEY, result_json VARCHAR NOT NULL)')


def _cast(column, dtype):
    return f'CAST({column} AS {dtype})'


def _project(conn, relation, source_schema, schema, mapping):
    """Make every conversion explicit and reject integer rounding."""
    translated = {mapping.get(col, col): col for col in source_schema}
    if len(translated) != len(source_schema) or set(translated) - set(schema):
        raise ValueError('Input column mapping is ambiguous or incomplete')
    parts = []
    for col, dtype in schema.items():
        source = translated.get(col)
        value = quote(source) if source else 'NULL'
        if source and dtype in {'INTEGER', 'BIGINT'} and source_schema[source] != dtype:
            if conn.execute(f'SELECT 1 FROM {relation} WHERE {value} IS NOT NULL AND '
                            f'CAST({value} AS DOUBLE) IS DISTINCT FROM CAST(CAST({value} AS {dtype}) AS DOUBLE) LIMIT 1').fetchone():
                raise ValueError(f'Lossy integer conversion: {source}')
        parts.append(f'{_cast(value, dtype)} AS {quote(col)}')
    return 'SELECT ' + ','.join(parts) + ' FROM ' + relation


def _register(conn, target, physical, version, audit, source):
    identity = digest({'source': source, 'target': target, 'version': version})
    conn.execute('INSERT INTO meta.datasets '
                 '(dataset_id,target_table,physical_table,source,version,audit_json,follow_head) '
                 'VALUES (?,?,?,?,?,?,false) ON CONFLICT DO NOTHING',
                 [identity, target, physical, source, version, canonical(audit)])
    conn.execute('INSERT INTO meta.input_heads VALUES (?,?) '
                 'ON CONFLICT DO UPDATE SET version=excluded.version', [target, version])


def refresh_aliases(conn, target):
    """Publish typed aliases to the new head; existing frozen refs stay unchanged."""
    parent = _parent(conn, target)
    catalog = quote(input_catalog_name(conn))
    for alias, rule in aliases(conn).items():
        if alias == target or rule['target'] != target:
            continue
        version = digest({'canonical_version': parent[0], 'alias': alias, 'rule': rule})
        physical = 'input_versions.alias_' + version
        projection = ','.join(
            f'CAST({quote(rule["mapping"].get(col, col))} AS {dtype}) AS {quote(col)}'
            for col in rule['source_columns']
            for dtype in [rule['source_schema'][col]]
        )
        query = f'SELECT {projection} FROM {catalog}.{table_name(parent[1])}'
        if rule.get('profile') == 'shareholder':
            # Keep provider IDs in the thin projection without assigning them to named holders.
            base = rule['identity_base']
            group = 'c.stock_code IS NOT DISTINCT FROM o.stock_code AND c.pub_date IS NOT DISTINCT FROM o.pub_date AND c.end_date IS NOT DISTINCT FROM o.report_date'
            query = (f'SELECT o.* FROM {catalog}.{table_name(base)} o WHERE EXISTS '
                     f'(SELECT 1 FROM {catalog}.{table_name(parent[1])} c WHERE c.holder_scope=\'total\' AND {group}) UNION ALL '
                     f'SELECT {projection} FROM {catalog}.{table_name(parent[1])} c WHERE holder_scope=\'total\' '
                     f'AND NOT EXISTS (SELECT 1 FROM {catalog}.{table_name(base)} o WHERE {group})')
        conn.execute(f'CREATE VIEW IF NOT EXISTS {table_name(physical)} AS {query}')
        conn.execute(f'CREATE OR REPLACE VIEW {table_name(alias)} AS SELECT * FROM {catalog}.{table_name(physical)}')
        audit = {'kind': 'input_alias_v1', 'schema': [[col,rule['source_schema'][col]] for col in rule['source_columns']],
                 'rows': conn.execute(f'SELECT count(*) FROM {table_name(physical)}').fetchone()[0] if rule.get('profile') == 'shareholder' else json.loads(parent[2])['rows'], 'parent_version': parent[0]}
        _register(conn, alias, physical, version, audit, 'input-alias://' + alias)


def merge_input_tables(path, spec, *, execute=False):
    """Validate and commit one table pair atomically. Repeating a plan is a no-op."""
    import duckdb
    target, source = spec['target'], spec['source']
    if any(not t.startswith('main.') or t.split('.')[1].startswith('__') for t in (target, source)) or target == source:
        raise ValueError('Expected distinct public main-schema tables')
    profile = spec.get('profile')
    if profile is not None and profile not in PROFILES:
        raise ValueError('Unknown reviewed input merge profile')
    merge_id = digest(spec)
    # Even preview opens a read-only connection, so no catalog or journal is created.
    context = write_store(Path(path)) if execute else duckdb.connect(str(path), read_only=True)
    with context as conn:
        if _exists(conn, 'meta.input_merges'):
            old = conn.execute('SELECT result_json FROM meta.input_merges WHERE merge_id=?', [merge_id]).fetchone()
            if old:
                return {**json.loads(old[0]), 'already_merged': True}
        if execute:
            _set_role(conn, 'input'); _catalog(conn); _catalog_aliases(conn)
        # _parent supports stores which have not yet published an incremental head.
        def parent(t):
            if _exists(conn, 'meta.input_heads'):
                return _parent(conn, t)
            found = conn.execute('SELECT DISTINCT version,physical_table,audit_json FROM meta.datasets '
                                 'WHERE target_table=? AND physical_table=?', [t, t]).fetchall()
            if len(found) != 1:
                raise ValueError('Input needs an unambiguous current version: ' + t)
            return found[0]
        pa, pb = parent(target), parent(source)
        if not pa or not pb:
            raise ValueError('Both input tables require registered parents')
        original_a, original_b = dict(describe(conn, table_name(pa[1]))), dict(describe(conn, table_name(pb[1])))
        def prepared(p, source):
            relation = table_name(p[1])
            if profile:
                sql, cols = normalize(conn, relation, profile, source=source, resolutions=spec.get('name_resolutions'))
                return '('+sql+')', cols
            return relation, dict(describe(conn,relation))
        ar, sa = prepared(pa, False)
        br, sb = prepared(pb, True)
        mapping = spec.get('mapping', {})
        schema = {**sa}
        for col, dtype in sb.items():
            schema.setdefault(mapping.get(col, col), dtype)
        for col, dtype in spec.get('types', {}).items():
            if col not in schema or dtype not in {'VARCHAR','DATE','DOUBLE','BIGINT','INTEGER','BOOLEAN'}:
                raise ValueError('Unsupported canonical type override')
            schema[col] = dtype
        keys = spec['keys']
        if not keys or len(keys) != len(set(keys)) or set(keys) - (set(sa) & {mapping.get(x,x) for x in sb}):
            raise ValueError('Merge needs shared, explicit keys')
        av = _project(conn, ar, sa, schema, {})
        bv = _project(conn, br, sb, schema, mapping)
        stats = []
        for sql in (av, bv):
            row = conn.execute('SELECT count(*),count(DISTINCT (' + ','.join(map(quote,keys)) + ')) FROM ('+sql+')').fetchone()
            if row[0] != row[1]:
                raise ValueError('Ambiguous input event key; merge refused')
            stats.append(row[0])
        join = ' AND '.join(f'a.{quote(k)} IS NOT DISTINCT FROM b.{quote(k)}' for k in keys)
        shared = (set(sa) & {mapping.get(x,x) for x in sb}) - set(keys) - METADATA_COLUMNS
        def mismatch(k):
            a, b = f'a.{quote(k)}', f'b.{quote(k)}'
            test = f'{a} IS DISTINCT FROM {b}'
            if profile:
                test = f'({a} IS NOT NULL AND {b} IS NOT NULL AND {test})'
            if profile == 'forecast' and k == 'forecast_content':
                test = f'({test} AND NOT {text_extension(a,b)})'
            return test
        conflict = ' OR '.join(mismatch(k) for k in sorted(shared)) or 'false'
        overlap, differences = conn.execute(f'WITH a AS ({av}),b AS ({bv}) SELECT count(*),'
            f'count(*) FILTER(WHERE {conflict}) FROM a JOIN b ON {join}').fetchone()
        if profile == 'shareholder':
            merged_sql = holder_union(conn,av,bv,schema)
            after = conn.execute('SELECT count(*) FROM ('+merged_sql+')').fetchone()[0]
            overlap, differences = sum(stats)-after, 0
        result = {'ok': differences == 0, 'target': target, 'source': source, 'rows_before': stats,
                  'overlap': overlap, 'differences': differences, 'rows_after': sum(stats)-overlap,
                  'keys': keys, 'schema': schema, 'parents': [pa[0],pb[0]], 'executed': False}
        if differences or not execute:
            return result
        # Keep originals for frozen refs using the established input-store base preservation.
        audits = [audit_relation(conn, table_name(p[1])) for p in (pa,pb)]
        for p, actual in zip((pa,pb),audits):
            frozen = json.loads(p[2])
            if 'sum_hash' in frozen and any(actual[k] != frozen[k] for k in ('schema','rows','sum_hash','xor_hash')):
                raise ValueError('Current data no longer matches its registered frozen version')
        pa, pb = _preserve_base(conn, target, pa), _preserve_base(conn, source, pb)
        ar, _ = prepared(pa, False)
        br, _ = prepared(pb, True)
        av = _project(conn, ar, sa, schema, {})
        bv = _project(conn, br, sb, schema, mapping)
        version = digest({'merge': spec, 'parents': result['parents'], 'audits': audits})
        physical = 'input_versions.merged_' + version
        projection = ','.join((f'coalesce(b.{quote(k)},a.{quote(k)})' if profile == 'forecast' and k == 'forecast_content' else f'coalesce(a.{quote(k)},b.{quote(k)})') + f' AS {quote(k)}' for k in schema)
        merged_sql = holder_union(conn,av,bv,schema) if profile == 'shareholder' else f'WITH a AS ({av}),b AS ({bv}) SELECT {projection} FROM a FULL OUTER JOIN b ON {join}'
        conn.execute(f'CREATE TABLE {table_name(physical)} AS {merged_sql}')
        actual = audit_relation(conn, table_name(physical))
        if actual['rows'] != result['rows_after']:
            raise ValueError('Merged row count mismatch')
        # Verify all original business values, including NULLs, not just hashes or row counts.
        if profile:
            # Exhaustive hash buckets bound join/sort memory; no rows are sampled out.
            buckets = 8 if result['rows_after'] > 1_000_000 else 1
            partition_keys = GROUP if profile == 'shareholder' else keys
            partition = 'hash(' + ','.join(map(quote,partition_keys)) + ')'
            for bucket in range(buckets):
                def bounded(sql):
                    return f'SELECT * FROM ({sql}) WHERE {partition} % {buckets} = {bucket}'
                aa, bb = bounded(av), bounded(bv)
                expected = holder_union(conn,aa,bb,schema) if profile == 'shareholder' else f'WITH a AS ({aa}),b AS ({bb}) SELECT {projection} FROM a FULL OUTER JOIN b ON {join}'
                observed = bounded(f'SELECT * FROM {table_name(physical)}')
                for left,right in ((expected,observed),(observed,expected)):
                    if conn.execute(f'SELECT 1 FROM (({left}) EXCEPT ALL ({right})) LIMIT 1').fetchone():
                        raise ValueError('Reviewed merge content verification failed')
        for sql, columns in ([] if profile else ((av,set(sa)),(bv,{mapping.get(x,x) for x in sb}))):
            equal = ' AND '.join(f'p.{quote(k)} IS NOT DISTINCT FROM m.{quote(k)}'
                                  for k in sorted(columns - METADATA_COLUMNS))
            match = ' AND '.join(f'p.{quote(k)} IS NOT DISTINCT FROM m.{quote(k)}' for k in keys)
            if conn.execute(f'WITH p AS ({sql}) SELECT 1 FROM p WHERE NOT EXISTS '
                            f'(SELECT 1 FROM {table_name(physical)} m WHERE {match} AND {equal}) LIMIT 1').fetchone():
                raise ValueError('Merge lost an original business value')
        for alias, original, rename in ((target,original_a,{}),(source,original_b,spec.get('alias_mapping',mapping))):
            rule = {'target':target,'source_schema':original,'source_columns':list(original),'mapping':rename,'keys':keys}
            if profile:
                rule.update(profile=profile, source=alias==source, normalize_mapping=mapping if alias==source else {})
                if profile == 'security':
                    rule['name_resolutions'] = spec['name_resolutions']
                if profile == 'shareholder' and alias == source:
                    rule['identity_base'] = pb[1]
            conn.execute('INSERT INTO meta.input_aliases VALUES (?,?,?)', [alias,target,canonical(rule)])
        catalog = quote(input_catalog_name(conn))
        conn.execute(f'CREATE OR REPLACE VIEW {table_name(target)} AS SELECT * FROM {catalog}.{table_name(physical)}')
        _register(conn,target,physical,version,actual,'input-merge://'+merge_id)
        conn.execute('INSERT INTO meta.input_keys VALUES (?,?) ON CONFLICT DO UPDATE SET columns_json=excluded.columns_json',
                     [target,canonical(keys)])
        refresh_aliases(conn,target)
        result.update(executed=True, version=version, physical_table=physical, audit=actual,
                      frozen_originals=[pa[1],pb[1]])
        conn.execute('INSERT INTO meta.input_merges VALUES (?,?)',[merge_id,canonical(result)])
        return result


def publish_merged_batch(conn, table, *, metadata, raw_audit, parents, replace):
    """Route old loader names to the canonical table and preserve unsupplied columns."""
    from utils.input_store import _publish
    rules = aliases(conn)
    alias = table if '.' in table else 'main.'+table
    rule = rules.get(alias)
    if not rule:
        return None
    target = rule['target']
    if replace:
        raise ValueError('Replacing one source of a merged table would erase other coverage; use incremental import')
    snapshot = rule.get('profile') == 'shareholder'
    batch = digest({'metadata':metadata,'table':target,'replace':snapshot})
    previous = conn.execute('SELECT source_audit_json,result_json FROM meta.input_batches WHERE batch_id=? AND target_table=?',
                            [batch,target]).fetchone()
    if previous:
        if json.loads(previous[0]).get('raw_source_audit') != raw_audit:
            raise ValueError('Input batch changed after publication')
        return json.loads(previous[1])
    parent = _parent(conn,target)
    expected = parents.get(target.removeprefix('main.'), {})
    if expected.get('role') != 'input' or expected.get('table') != target or expected.get('version') != parent[0]:
        raise ValueError('Canonical input head changed while loader was running')
    if alias != target:
        ref = parents.get(table, {})
        head = _parent(conn,alias)
        if ref.get('table') != alias or ref.get('version') != head[0]:
            raise ValueError('Alias input head changed while loader was running')
    schema = dict(describe(conn,table_name(parent[1])))
    src = dict(raw_audit['schema'])
    source_relation = 'input_batch.'+table_name(table)
    mapping = rule.get('normalize_mapping',rule['mapping'])
    if rule.get('profile'):
        normalized, src = normalize(conn,source_relation,rule['profile'],source=rule['source'],resolutions=rule.get('name_resolutions'))
        source_relation = '('+normalized+')'
    supplied = {mapping.get(x,x) for x in src}
    if set(rule['keys']) - supplied:
        raise ValueError('Merged input batch lacks keys')
    sql = _project(conn,source_relation,src,schema,mapping)
    join = ' AND '.join(f's.{quote(k)} IS NOT DISTINCT FROM p.{quote(k)}' for k in rule['keys'])
    def value(k):
        s, p = f's.{quote(k)}', f'p.{quote(k)}'
        if rule.get('profile') == 'forecast' and k == 'forecast_content' and k in supplied:
            return f'CASE WHEN {text_extension(s,p)} THEN {p} ELSE coalesce({s},{p}) END'
        return f'coalesce({s},{p})' if rule.get('profile') and k in supplied else (s if k in supplied else p)
    cols = ','.join(value(k) + f' AS {quote(k)}' for k in schema)
    view = 'normalized_' + batch
    query = f'WITH s AS ({sql}) SELECT {cols} FROM s LEFT JOIN {table_name(parent[1])} p ON {join}'
    if rule.get('profile') == 'shareholder':
        # Keep the existing rich representation; imports cannot silently revise a disclosure batch.
        current = f'SELECT * FROM {table_name(parent[1])}'
        query = holder_union(conn,current,sql,schema) if rule['source'] else holder_union(conn,sql,current,schema)
        # Keep the provider-ID carrier separate from the named-holder representation.
        for name, alias_rule in rules.items():
            if name == target or alias_rule['target'] != target:
                continue
            identity = _parent(conn,name)[1]
            if rule['source']:
                delta = 'input_versions.holder_ids_' + batch
                columns = ','.join(map(quote,alias_rule['source_columns']))
                conn.execute(f'CREATE TABLE {table_name(delta)} AS SELECT {columns} FROM input_batch.{table_name(table)}')
                identity_view = 'input_versions.holder_identity_' + batch
                group = 'o.stock_code IS NOT DISTINCT FROM n.stock_code AND o.pub_date IS NOT DISTINCT FROM n.pub_date AND o.report_date IS NOT DISTINCT FROM n.report_date'
                conn.execute(f'CREATE VIEW {table_name(identity_view)} AS SELECT o.* FROM {table_name(identity)} o WHERE NOT EXISTS '
                             f'(SELECT 1 FROM {table_name(delta)} n WHERE {group}) UNION ALL SELECT * FROM {table_name(delta)}')
                identity = identity_view
            alias_rule['identity_base'] = identity
            conn.execute('UPDATE meta.input_aliases SET rule_json=? WHERE alias_table=?',[canonical(alias_rule),name])
    conn.execute(f'CREATE TEMP VIEW {quote(view)} AS {query}')
    # ponytail: shareholder full snapshots preserve whole batches; use group deltas if frequent imports make copying costly.
    audit = audit_relation(conn,quote(view))
    audit['raw_source_audit'] = raw_audit
    result = _publish(conn,target,metadata=metadata,source_audit=audit,source_keys=rule['keys'],
                      parent_version=parent[0],replace=snapshot,source_relation=quote(view),allow_nullable_keys=True)
    refresh_aliases(conn,target)
    return result
