"""Publish downloaded input batches without changing frozen table versions."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import duckdb

from utils.data_store import (
    DataRef,
    _exists,
    _set_role,
    audit_relation,
    canonical,
    describe,
    digest,
    input_catalog_name,
    quote,
    table_name,
    write_store,
)


def _catalog(conn):
    conn.execute("ALTER TABLE meta.store ADD COLUMN IF NOT EXISTS catalog_name VARCHAR")
    current = conn.execute("SELECT current_database()").fetchone()[0]
    if input_catalog_name(conn) != current:
        raise ValueError(
            "Input database was renamed; restore its catalog name before publishing"
        )
    conn.execute(
        "UPDATE meta.store SET catalog_name=? WHERE catalog_name IS NULL", [current]
    )
    conn.execute("CREATE SCHEMA IF NOT EXISTS input_versions")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta.input_heads (target_table VARCHAR PRIMARY KEY, version VARCHAR NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta.input_keys (target_table VARCHAR PRIMARY KEY, columns_json VARCHAR NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta.input_batches (batch_id VARCHAR, target_table VARCHAR, metadata_json VARCHAR, "
        "source_audit_json VARCHAR, result_json VARCHAR, PRIMARY KEY(batch_id,target_table))"
    )


def attach_input_parent(conn, table, *, bindings):
    """Return the qualified relation for a loader's exact, read-only parent version."""
    path = Path(bindings["input_db"]).resolve(strict=True)
    current = conn.execute("SELECT current_database()").fetchone()[0]
    databases = dict(
        (name, file)
        for _, name, file in conn.execute("PRAGMA database_list").fetchall()
    )
    if databases.get(current) and Path(databases[current]).resolve() == path:
        raise ValueError("Input loader must write staging, not its frozen parent")
    with duckdb.connect(str(path), read_only=True) as source:
        catalog = input_catalog_name(source)
    if catalog in databases:
        if Path(databases[catalog]).resolve() != path:
            raise ValueError("Input parent attachment changed")
    else:
        conn.execute(
            "ATTACH '"
            + str(path).replace("'", "''")
            + "' AS "
            + quote(catalog)
            + " (READ_ONLY)"
        )
    ref = DataRef(**bindings["inputs"][table])
    if ref.role != "input" or ref.table != (table if "." in table else "main." + table):
        raise ValueError("Loader parent binding belongs to a different table")
    rows = conn.execute(
        f"SELECT DISTINCT physical_table FROM {quote(catalog)}.meta.datasets WHERE target_table=? AND version=?",
        [ref.table, ref.version],
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("Loader parent version is missing or ambiguous")
    return quote(catalog) + "." + table_name(rows[0][0])


def seed_input_rows(conn, table, key_table, keys, *, bindings):
    """Seed only a loader's affected keys from its frozen parent, never the whole table."""
    source = attach_input_parent(conn, table, bindings=bindings)
    target = table_name(table)
    if not keys or any(key not in dict(describe(conn, source)) for key in keys):
        raise ValueError("Loader requires valid affected-key columns")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {target} AS SELECT * FROM {source} WHERE false"
    )
    if describe(conn, source) != describe(conn, target):
        raise ValueError("Loader staging schema differs from the frozen parent")
    match = " AND ".join(
        f"p.{quote(key)} IS NOT DISTINCT FROM k.{quote(key)}" for key in keys
    )
    conn.execute(
        f"INSERT INTO {target} SELECT p.* FROM {source} p "
        f"WHERE EXISTS (SELECT 1 FROM {table_name(key_table)} k WHERE {match}) "
        f"AND NOT EXISTS (SELECT 1 FROM {target} k WHERE {match})"
    )


def _keys(conn, table, database=""):
    schema, name = (table if "." in table else "main." + table).split(".")
    row = conn.execute(
        "SELECT constraint_column_names FROM duckdb_constraints() WHERE schema_name=? AND table_name=? "
        "AND constraint_type='PRIMARY KEY'"
        + (" AND database_name=?" if database else ""),
        [schema, name, *([database] if database else [])],
    ).fetchone()
    if row:
        return list(row[0])
    indexes = conn.execute(
        "SELECT expressions FROM duckdb_indexes() WHERE schema_name=? AND table_name=? AND is_unique"
        + (" AND database_name=?" if database else ""),
        [schema, name, *([database] if database else [])],
    ).fetchall()
    if not indexes:
        return []
    available = {
        column
        for column, _ in describe(
            conn, (quote(database) + "." if database else "") + table_name(table)
        )
    }
    choices = []
    for (expression,) in indexes:
        columns = [
            name.strip().strip('"').replace('""', '"')
            for name in expression.strip("[]").split(",")
        ]
        if not columns or any(column not in available for column in columns):
            raise ValueError(
                f"Expression-based unique index needs an explicit input key contract: {table}"
            )
        choices.append(columns)
    return min(choices, key=lambda columns: (len(columns), columns))


def _parent(conn, target):
    head = conn.execute(
        "SELECT version FROM meta.input_heads WHERE target_table=?", [target]
    ).fetchone()
    rows = conn.execute(
        "SELECT DISTINCT version,physical_table,audit_json FROM meta.datasets WHERE target_table=?"
        + (" AND version=?" if head else " AND physical_table=?"),
        [target, head[0] if head else target],
    ).fetchall()
    if len(rows) > 1:
        raise ValueError(f"Input table needs an explicit head: {target}")
    if rows:
        return rows[0]
    if _exists(conn, target):
        raise ValueError(f"Refusing to update an unversioned input table: {target}")
    return None


def _publish(
    conn, table, *, metadata, source_audit, source_keys, parent_version, replace,
    source_relation=None, allow_nullable_keys=False
):
    target = table if "." in table else "main." + table
    schema, name = target.split(".")
    if schema != "main" or name.startswith("__"):
        raise ValueError("Live input tables must be ordinary main-schema tables")
    batch = digest({"metadata": metadata, "table": target, "replace": replace})
    previous = conn.execute(
        "SELECT source_audit_json,result_json FROM meta.input_batches WHERE batch_id=? AND target_table=?",
        [batch, target],
    ).fetchone()
    if previous:
        if json.loads(previous[0]) != source_audit:
            raise ValueError("An import batch changed after publication")
        return json.loads(previous[1])
    parent = _parent(conn, target)
    actual_version = parent[0] if parent else None
    if actual_version != parent_version:
        raise ValueError(f"Input head changed while loader was running: {target}")
    source = source_relation or "input_batch." + table_name(table)
    if parent:
        expected_schema = describe(conn, table_name(parent[1]))
        if dict(expected_schema) != dict(source_audit["schema"]):
            raise ValueError(
                f"Input schema changed; an explicit schema migration is required: {target}"
            )
        source = (
            "(SELECT "
            + ",".join(quote(column) for column, _ in expected_schema)
            + f" FROM {source})"
        )
    stored_keys = conn.execute(
        "SELECT columns_json FROM meta.input_keys WHERE target_table=?", [target]
    ).fetchone()
    keys = (
        json.loads(stored_keys[0])
        if stored_keys
        else (
            _keys(
                conn, parent[1], conn.execute("SELECT current_database()").fetchone()[0]
            )
            if parent
            else []
        )
    )
    if keys and source_keys and keys != source_keys:
        raise ValueError(f"Input primary key changed: {target}")
    keys = keys or source_keys
    columns = [column for column, _ in source_audit["schema"]]
    key_columns = keys or columns
    grouped = ",".join(quote(column) for column in key_columns)
    if (
        keys and not allow_nullable_keys
        and conn.execute(
            f"SELECT 1 FROM {source} WHERE "
            + " OR ".join(f"{quote(k)} IS NULL" for k in keys)
            + " LIMIT 1"
        ).fetchone()
    ):
        raise ValueError(f"Null input primary key: {target}")
    if conn.execute(
        f"SELECT 1 FROM {source} GROUP BY {grouped} HAVING count(*)>1 LIMIT 1"
    ).fetchone():
        raise ValueError(f"Duplicate keys in input batch: {target}")
    match = " AND ".join(
        f"p.{quote(k)} IS NOT DISTINCT FROM s.{quote(k)}" for k in key_columns
    )
    equal = " AND ".join(
        f"p.{quote(k)} IS NOT DISTINCT FROM s.{quote(k)}" for k in columns
    )
    # Old rows stay in their original data pages; only new/changed rows are copied.
    delta = "input_versions.delta_" + batch
    condition = (
        ""
        if replace or not parent
        else f" WHERE NOT EXISTS (SELECT 1 FROM {table_name(parent[1])} p WHERE {match} AND {equal})"
    )
    conn.execute(
        f"CREATE TABLE {table_name(delta)} AS SELECT s.* FROM {source} s{condition}"
    )
    audit = audit_relation(conn, table_name(delta))
    replaced = 0
    if parent and not replace:
        replaced = conn.execute(
            f"SELECT count(*) FROM {table_name(parent[1])} p WHERE EXISTS (SELECT 1 FROM {table_name(delta)} s WHERE {match})"
        ).fetchone()[0]
    version = digest(
        {
            "parent": actual_version,
            "delta": audit,
            "replace": replace,
            "keys": key_columns,
        }
    )
    if parent and not replace and audit["rows"] == 0:
        conn.execute(f"DROP TABLE {table_name(delta)}")
        version, physical, complete_audit = parent[0], parent[1], json.loads(parent[2])
    else:
        parent = _preserve_base(conn, target, parent)
        physical = "input_versions.snapshot_" + digest(
            {"table": target, "version": version}
        )
        catalog = quote(input_catalog_name(conn))
        query = f"SELECT * FROM {catalog}.{table_name(delta)}"
        if parent and not replace:
            query = (
                f"SELECT p.* FROM {catalog}.{table_name(parent[1])} p WHERE NOT EXISTS (SELECT 1 FROM {catalog}.{table_name(delta)} s WHERE {match}) UNION ALL "
                + query
            )
        conn.execute(f"CREATE VIEW {table_name(physical)} AS {query}")
        conn.execute(
            f"CREATE OR REPLACE VIEW {table_name(target)} AS SELECT * FROM {catalog}.{table_name(physical)}"
        )
        rows = audit["rows"] + (
            json.loads(parent[2])["rows"] - replaced if parent and not replace else 0
        )
        # The content identity covers the immutable parent and complete delta, without rescanning the parent.
        complete_audit = {
            "kind": "input_delta_v1",
            "schema": audit["schema"],
            "rows": rows,
            "parent_version": actual_version,
            "delta_audit": audit,
            "content_version": version,
        }
        conn.execute(
            "INSERT INTO meta.datasets (dataset_id,target_table,physical_table,source,version,audit_json) VALUES (?,?,?,?,?,?)",
            [
                batch,
                target,
                physical,
                metadata["source"],
                version,
                canonical(complete_audit),
            ],
        )
    conn.execute(
        "INSERT INTO meta.input_heads VALUES (?,?) ON CONFLICT DO UPDATE SET version=excluded.version",
        [target, version],
    )
    conn.execute(
        "INSERT INTO meta.input_keys VALUES (?,?) ON CONFLICT DO NOTHING",
        [target, canonical(keys)],
    )
    result = {
        "ok": True,
        "ref": asdict(DataRef("input", target, version)),
        "batch_id": batch,
        "physical_table": physical,
        "input_rows": source_audit["rows"],
        "changed_rows": audit["rows"],
        "replaced_rows": replaced,
        "audit": complete_audit,
    }
    conn.execute(
        "INSERT INTO meta.input_batches VALUES (?,?,?,?,?)",
        [
            batch,
            target,
            canonical(metadata),
            canonical(source_audit),
            canonical(result),
        ],
    )
    return result


def _preserve_base(conn, target, parent):
    if parent and parent[1] == target:
        base_name = '__input_base_' + digest({'table': target, 'version': parent[0]})
        base = target.split('.')[0] + '.' + base_name
        # Explicit indexes prevent ALTER TABLE RENAME in DuckDB. Rebuild them on
        # the preserved base inside this same transaction, retaining uniqueness.
        import re
        schema, name = target.split('.')
        indexes = conn.execute(
            'SELECT index_name,is_unique,sql FROM duckdb_indexes() WHERE database_name=current_database() '
            'AND schema_name=? AND table_name=?', [schema, name]
        ).fetchall()
        rewrites = []
        identifiers = [name, quote(name), schema+'.'+name, table_name(target),
                       quote(schema)+'.'+name, schema+'.'+quote(name)]
        for index, unique, sql in indexes:
            pattern = r' ON (?:' + '|'.join(re.escape(x) for x in identifiers) + r')(?=\s*\()'
            match = re.search(pattern, sql)
            if not match:
                raise ValueError('Cannot safely preserve index definition: '+index)
            # DuckDB cannot reuse a dropped UNIQUE index name in the same transaction.
            preserved_index = '__input_index_' + digest({'base':base,'index':index})
            rewritten = ('CREATE ' + ('UNIQUE ' if unique else '') + 'INDEX ' + quote(preserved_index)
                         + ' ON ' + table_name(base) + sql[match.end():])
            rewrites.append((index,rewritten))
        for index, _ in rewrites:
            conn.execute(f'DROP INDEX {quote(schema)}.{quote(index)}')
        conn.execute(f'ALTER TABLE {table_name(target)} RENAME TO {quote(base_name)}')
        for _, sql in rewrites:
            conn.execute(sql)
        conn.execute('UPDATE meta.datasets SET physical_table=? WHERE physical_table=?', [base, target])
        parent = (parent[0], base, parent[2])
    return parent


def publish_price_factors(path, staging, *, tables, metadata, parents, replace=False):
    """Publish daily factor overrides without copying any intraday price rows."""
    target = 'main.price_data_5m'
    if tables != ['price_data_5m'] or replace:
        raise ValueError('Daily factor refresh only updates price_data_5m.factor')
    if any(not metadata.get(key) for key in ('source', 'source_hash', 'loader_hash')):
        raise ValueError('Factor refresh requires source and loader identities')
    ref = parents.get('price_data_5m', {})
    if ref.get('role') != 'input' or ref.get('table') != target:
        raise ValueError('Factor refresh requires an exact intraday parent version')
    path, staging = Path(path).resolve(), Path(staging).resolve(strict=True)
    if path == staging:
        raise ValueError('Factor refresh must use a separate staging database')
    with write_store(path) as conn:
        _set_role(conn, 'input')
        _catalog(conn)
        conn.execute("ATTACH '" + str(staging).replace("'", "''") + "' AS input_batch (READ_ONLY)")
        source = 'input_batch.main.price_data_5m'
        contract = conn.execute('SELECT start_date,end_date,reference_date,parent_version,custody_json FROM input_batch.meta.factor_refresh').fetchall()
        if len(contract) != 1 or contract[0][3] != ref['version']:
            raise ValueError('Factor refresh custody belongs to another parent version')
        start, end, reference, _, custody = contract[0]
        if start is None or end is None or start > end or reference != end or not json.loads(custody).get('ok'):
            raise ValueError('Invalid factor refresh range or bar custody')
        source_audit = audit_relation(conn, source)
        if dict(source_audit['schema']) != {'stock_code':'VARCHAR','trade_date':'DATE','factor':'DOUBLE'} or not source_audit['rows']:
            raise ValueError('Factor refresh must contain daily stock/date/factor rows')
        source_audit['refresh_contract'] = [str(start),str(end),str(reference),ref['version'],json.loads(custody)]
        batch = digest({'metadata':metadata,'table':target,'operation':'daily_price_factor_patch_v1'})
        previous = conn.execute('SELECT source_audit_json,result_json FROM meta.input_batches WHERE batch_id=? AND target_table=?',[batch,target]).fetchone()
        if previous:
            if json.loads(previous[0]) != source_audit:
                raise ValueError('Factor refresh changed after publication')
            return {'ok':True,'results':[json.loads(previous[1])]}
        parent = _parent(conn, target)
        if not parent or parent[0] != ref['version']:
            raise ValueError('Input head changed while factor refresh was running')
        if conn.execute(f'SELECT 1 FROM {source} WHERE stock_code IS NULL OR trade_date IS NULL OR trade_date<? OR trade_date>? OR (factor IS NOT NULL AND (NOT isfinite(factor) OR factor<=0)) LIMIT 1',[start,end]).fetchone():
            raise ValueError('Invalid daily factor key, range or value')
        if conn.execute(f'SELECT 1 FROM {source} GROUP BY stock_code,trade_date HAVING count(*)>1 LIMIT 1').fetchone():
            raise ValueError('Duplicate daily factor keys')
        parent_schema = describe(conn,table_name(parent[1]))
        factor_type = dict(parent_schema).get('factor')
        if factor_type != 'DOUBLE':
            raise ValueError('Intraday factor schema requires an explicit migration')
        stored_keys = conn.execute('SELECT columns_json FROM meta.input_keys WHERE target_table=?',[target]).fetchone()
        keys = json.loads(stored_keys[0]) if stored_keys else _keys(conn,parent[1],conn.execute('SELECT current_database()').fetchone()[0])
        if set(keys) != {'stock_code','date'}:
            raise ValueError('Factor refresh requires verified intraday natural keys')
        changed, missing = conn.execute(f'''SELECT
            count(*) FILTER (WHERE f.factor>0 AND p.factor IS DISTINCT FROM f.factor),
            count(*) FILTER (WHERE p.open>0 AND p.close>0 AND f.factor IS NULL)
            FROM {table_name(parent[1])} p LEFT JOIN {source} f
            ON p.stock_code=f.stock_code AND CAST(p.date AS DATE)=f.trade_date
            WHERE p.date>=? AND p.date<CAST(? AS DATE)+INTERVAL 1 DAY''',[start,end]).fetchone()
        version, physical, complete_audit = parent[0], parent[1], json.loads(parent[2])
        patch_rows = 0
        if changed:
            patch = 'input_versions.factor_patch_' + batch
            conn.execute(f'CREATE TABLE {table_name(patch)} AS SELECT * FROM {source} WHERE factor>0')
            patch_audit = audit_relation(conn,table_name(patch))
            patch_rows = patch_audit['rows']
            version = digest({'kind':'daily_price_factor_patch_v1','parent':parent[0],'patch':patch_audit})
            parent = _preserve_base(conn,target,parent)
            physical = 'input_versions.snapshot_' + digest({'table':target,'version':version})
            catalog = quote(input_catalog_name(conn))
            # ponytail: one daily join per refresh; compact patch chains if repeated refreshes slow reads.
            conn.execute(f'''CREATE VIEW {table_name(physical)} AS SELECT p.* REPLACE
                (CASE WHEN f.factor>0 THEN f.factor ELSE p.factor END AS factor)
                FROM {catalog}.{table_name(parent[1])} p LEFT JOIN {catalog}.{table_name(patch)} f
                ON p.stock_code=f.stock_code AND CAST(p.date AS DATE)=f.trade_date''')
            conn.execute(f'CREATE OR REPLACE VIEW {table_name(target)} AS SELECT * FROM {catalog}.{table_name(physical)}')
            complete_audit = {'kind':'daily_price_factor_patch_v1','schema':parent_schema,
                              'rows':json.loads(parent[2])['rows'],'parent_version':parent[0],
                              'patch_audit':patch_audit,'content_version':version}
            conn.execute('INSERT INTO meta.datasets (dataset_id,target_table,physical_table,source,version,audit_json) VALUES (?,?,?,?,?,?)',
                         [batch,target,physical,metadata['source'],version,canonical(complete_audit)])
        conn.execute('INSERT INTO meta.input_heads VALUES (?,?) ON CONFLICT DO UPDATE SET version=excluded.version',[target,version])
        conn.execute('INSERT INTO meta.input_keys VALUES (?,?) ON CONFLICT DO NOTHING',[target,canonical(keys)])
        result = {'ok':True,'ref':asdict(DataRef('input',target,version)),'batch_id':batch,'physical_table':physical,
                  'input_rows':source_audit['rows'],'changed_rows':changed,'patch_rows':patch_rows,'audit':complete_audit,
                  'status':'complete_with_source_gaps' if missing else 'complete','research_ready':not bool(missing),
                  'valid_price_bars_without_fresh_factor':missing,'reference_date':str(reference),'backup_ref':ref}
        conn.execute('INSERT INTO meta.input_batches VALUES (?,?,?,?,?)',[batch,target,canonical(metadata),canonical(source_audit),canonical(result)])
    return {'ok':True,'results':[result]}


def publish_input_batch(path, staging, *, tables, metadata, parents, replace=False):
    """Atomically publish every table produced by one isolated loader invocation."""
    if not isinstance(replace, bool):
        raise ValueError("replace must be a bool")
    if (
        not metadata.get("source")
        or not metadata.get("source_hash")
        or not metadata.get("loader_hash")
    ):
        raise ValueError(
            "Input publication requires source, source_hash and loader_hash"
        )
    if not tables or len(tables) != len(set(tables)):
        raise ValueError("An input batch needs distinct declared tables")
    path, staging = Path(path).resolve(), Path(staging).resolve(strict=True)
    if path == staging:
        raise ValueError("Loader staging cannot be the live input database")
    inspected = {}
    with duckdb.connect(str(staging), read_only=True) as reader:
        reader.execute("SET threads=1")
        reader.execute("SET memory_limit='3GB'")
        for table in tables:
            inspected[table] = (
                audit_relation(reader, table_name(table)),
                _keys(reader, table),
            )
            if inspected[table][0]["rows"] == 0:
                raise ValueError(f"Loader produced an empty input table: {table}")
    with write_store(path) as conn:
        _set_role(conn, "input")
        _catalog(conn)
        conn.execute(
            "ATTACH '"
            + str(staging).replace("'", "''")
            + "' AS input_batch (READ_ONLY)"
        )
        results = []
        for table, (audit, keys) in inspected.items():
            if audit_relation(conn, "input_batch." + table_name(table)) != audit:
                raise ValueError("Staging data changed before publication")
            from utils.input_consolidation import publish_merged_batch
            merged = publish_merged_batch(conn, table, metadata=metadata, raw_audit=audit,
                                          parents=parents, replace=replace)
            if merged is not None:
                results.append(merged)
                continue
            ref = parents.get(table)
            if ref and (
                ref["role"] != "input"
                or ref["table"] != (table if "." in table else "main." + table)
            ):
                raise ValueError("Parent binding belongs to a different input table")
            results.append(
                _publish(
                    conn,
                    table,
                    metadata=metadata,
                    source_audit=audit,
                    source_keys=keys,
                    parent_version=ref["version"] if ref else None,
                    replace=replace,
                )
            )
    return {"ok": True, "results": results}
