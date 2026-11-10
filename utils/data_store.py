"""Versioned DuckDB data references and atomic publication.

This module belongs to the execution repository. The harness invokes its JSON
CLI through the wrapper boundary; neither repository imports the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

HASH_ENGINE_VERSION = "1.5.3"
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _reject_sqlite_file(path: str | Path) -> None:
    """Fail before DuckDB can inspect a legacy SQLite file by extension."""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return
    with candidate.open("rb") as handle:
        if handle.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC:
            raise ValueError(
                "SQLite database files are unsupported; export this source to DuckDB or Parquet first"
            )


def _check_hash_engine():
    if duckdb.__version__ != HASH_ENGINE_VERSION:
        raise RuntimeError(
            f"Versioned data requires DuckDB {HASH_ENGINE_VERSION}; installed {duckdb.__version__}. "
            "Install the pinned requirements before reading audits or publishing."
        )


def quote(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def table_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"Invalid table name: {name!r}")
    parts = name.split(".")
    if len(parts) == 1:
        parts.insert(0, "main")
    if len(parts) != 2:
        raise ValueError("Table must be schema.table")
    return ".".join(quote(part) for part in parts)


def canonical(value: Any) -> str:
    checked = set()

    def check_keys(item):
        if not isinstance(item, (dict, list, tuple)) or id(item) in checked:
            return
        # Leave circular-reference rejection to json.dumps, as before.
        checked.add(id(item))
        if isinstance(item, dict):
            for key in item:
                if not isinstance(key, str):
                    raise TypeError("Canonical JSON requires string dictionary keys")
            children = item.values()
        else:
            children = item
        for child in children:
            check_keys(child)

    check_keys(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class DataRef:
    role: str
    table: str
    version: str

    def __post_init__(self):
        if (self.role not in {"input", "output"}
                or not isinstance(self.version, str) or not self.version.strip()):
            raise ValueError("A data reference needs a role and an exact version")
        table_name(self.table)


def _catalog(conn) -> None:
    conn.execute("CREATE SCHEMA IF NOT EXISTS meta")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta.store (singleton BOOLEAN PRIMARY KEY CHECK(singleton), role VARCHAR NOT NULL)"
    )
    conn.execute("""CREATE TABLE IF NOT EXISTS meta.artifacts (
        artifact_id VARCHAR PRIMARY KEY, target_table VARCHAR NOT NULL,
        schema_json VARCHAR NOT NULL, row_count BIGINT NOT NULL,
        content_hash VARCHAR NOT NULL, metadata_json VARCHAR NOT NULL,
        original_file VARCHAR, access_class VARCHAR NOT NULL,
        published_at TIMESTAMPTZ DEFAULT current_timestamp
    )""")
    # DuckDB 1.5.3 can rewrite values on repeated ADD COLUMN IF NOT EXISTS with DEFAULT.
    if "storage_table" not in {row[0] for row in conn.execute("DESCRIBE meta.artifacts").fetchall()}:
        conn.execute("ALTER TABLE meta.artifacts ADD COLUMN storage_table VARCHAR")
    conn.execute("""CREATE TABLE IF NOT EXISTS meta.documents (
        artifact_id VARCHAR PRIMARY KEY, document JSON NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS meta.datasets (
        dataset_id VARCHAR PRIMARY KEY, target_table VARCHAR NOT NULL,
        physical_table VARCHAR NOT NULL, source VARCHAR NOT NULL,
        version VARCHAR NOT NULL, audit_json VARCHAR NOT NULL,
        imported_at TIMESTAMPTZ DEFAULT current_timestamp,
        UNIQUE(target_table, source, version)
    )""")
    if "follow_head" not in {row[0] for row in conn.execute("DESCRIBE meta.datasets").fetchall()}:
        conn.execute("ALTER TABLE meta.datasets ADD COLUMN follow_head BOOLEAN DEFAULT true")
    if "source_table" not in {row[0] for row in conn.execute("DESCRIBE meta.datasets").fetchall()}:
        conn.execute("ALTER TABLE meta.datasets ADD COLUMN source_table VARCHAR")


def _set_role(conn, role: str) -> None:
    row = conn.execute("SELECT role FROM meta.store").fetchone()
    if row and row[0] != role:
        raise ValueError(f"Cannot publish {role} data in a {row[0]} store")
    conn.execute("INSERT INTO meta.store (singleton,role) VALUES (true, ?) ON CONFLICT DO NOTHING", [role])


def _configure_resources(conn):
    """Apply the same resource budget to readers and writers."""
    conn.execute("SET memory_limit=?", [os.environ.get("QUANT_DATA_STORE_MEMORY_LIMIT", "3GB")])
    conn.execute("SET threads=?", [int(os.environ.get("QUANT_DATA_STORE_THREADS", "2"))])


@contextmanager
def write_store(path: str | Path, *, session=None) -> Iterator[Any]:
    _check_hash_engine()
    # Reuse the existing cross-process writer lock. All shared writes use it.
    from utils.duckdb_results_writer import acquire_exclusive_write_conn

    path = Path(path).expanduser().resolve()
    _reject_sqlite_file(path)
    with acquire_exclusive_write_conn(path) if session is None else nullcontext(session) as session:
        if session.lock_path != Path(f"{path}.write.lock"):
            raise ValueError("Write session belongs to another database")
        conn = session.conn
        _configure_resources(conn)
        conn.execute("BEGIN TRANSACTION")
        try:
            _catalog(conn)
            yield conn
            from utils.data_revision import advance_revision
            advance_revision(conn)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def describe(conn, relation: str) -> list[list[str]]:
    return [[str(row[0]), str(row[1])] for row in conn.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]


def audit_relation(conn, relation: str) -> dict:
    """Order-independent, duplicate-sensitive content check, including every value.

    DuckDB hash is engine-version dependent, so the engine version is part of
    the identity. Source and destination are checked by the same engine.
    TIMESTAMPTZ bounds use UTC, preserving the original UTC string format
    without changing the caller's session timezone or existing catalog hashes.
    Republishing a legacy non-UTC audit can produce a different identity;
    existing exact references remain catalog keys and are not rewritten.
    """
    _check_hash_engine()
    schema = describe(conn, relation)
    columns = ", ".join(quote(name) for name, _ in schema)
    null_sql = ", ".join(f"count(*) - count({quote(name)})" for name, _ in schema)
    row = conn.execute(
        f"SELECT count(*), coalesce(sum(hash({columns})::HUGEINT), 0), "
        f"coalesce(bit_xor(hash({columns})), 0), {null_sql} FROM {relation}"
    ).fetchone()
    dates = {}
    for name, dtype in schema:
        if dtype.startswith(("DATE", "TIMESTAMP")) or name in {"date", "asof", "effective_date", "observed_through"}:
            # DuckDB 1.5.3 can fail while optimizing MIN/MAX on an empty filtered version.
            # The aggregate above already established emptiness; its date bounds are NULL.
            bounds = [f"{operation}({quote(name)})" for operation in ("min", "max")]
            if dtype == "TIMESTAMP WITH TIME ZONE":
                # Append the original UTC offset only to finite timestamps;
                # NULL and +/-infinity retain their existing representation.
                bounds = [
                    f"CASE WHEN isfinite({bound}) THEN "
                    f"CAST({bound} AT TIME ZONE 'UTC' AS VARCHAR) || '+00' "
                    f"ELSE CAST({bound} AS VARCHAR) END"
                    for bound in bounds
                ]
            else:
                bounds = [f"CAST({bound} AS VARCHAR)" for bound in bounds]
            low, high = conn.execute(
                f"SELECT {', '.join(bounds)} FROM {relation}"
            ).fetchone() if row[0] else (None, None)
            dates[name] = [str(low) if low is not None else None, str(high) if high is not None else None]
    return {
        "schema": schema,
        "rows": row[0],
        "sum_hash": str(row[1]),
        "xor_hash": str(row[2]),
        "nulls": dict(zip((x[0] for x in schema), row[3:], strict=True)),
        "dates": dates,
        "duckdb_version": duckdb.__version__,
    }


def _exists(conn, table: str) -> bool:
    schema, name = (table if "." in table else "main." + table).split(".")
    return bool(
        conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_catalog=current_database() "
            "AND table_schema=? AND table_name=?",
            [schema, name],
        ).fetchone()[0]
    )


def publish_relation(
    conn, relation: str, target: str, *, metadata: dict, original_file: str = "", access_class: str = "feature"
) -> DataRef:
    """Publish on the caller's transaction. Never mutate a published version."""
    _set_role(conn, "output")
    target = target if "." in target else "main." + target
    target_sql = table_name(target)
    if access_class not in {"feature", "label", "evaluation", "document", "attachment"}:
        raise ValueError("Unknown artifact access class")
    if not metadata or not any(metadata.get(key) for key in ("run_id", "legacy_source", "build_id")):
        raise ValueError("Publication requires run_id, build_id or explicit legacy_source provenance")
    if target.startswith("regime.") and target.split(".")[1] in {"targets", "observations", "predictions"}:
        access_class = "label"
    elif target.startswith(("backtest.", "alignment.")) or target in {
        "graph.evaluation_rows", "graph.daily_ic", "graph.research_daily"
    }:
        access_class = "evaluation"
    audit = audit_relation(conn, relation)
    keys = {
        "main.factor_values": ["date", "stock_code", "factor_name"],
        "main.neutralized_factor_values": ["date", "stock_code", "factor_name", "pool_id"],
        "main.factor_analysis_results": ["factor_name", "date", "code"],
        "main.monthly_pool_results": ["factor_name", "pool_type", "year_month", "period_days"],
    }.get(target)
    if keys and not metadata.get("legacy_source"):
        grouped = ", ".join(quote(key) for key in keys)
        if conn.execute(f"SELECT 1 FROM {relation} GROUP BY {grouped} HAVING count(*)>1 LIMIT 1").fetchone():
            raise ValueError(f"Duplicate logical keys within artifact for {target}")
    if any(name == "_artifact_id" for name, _ in audit["schema"]):
        raise ValueError("_artifact_id is reserved")
    content_hash = digest(audit)
    identity = {"table": target, "content_hash": content_hash, "metadata": metadata, "access_class": access_class}
    artifact_id = digest(identity)
    existing = conn.execute("SELECT target_table FROM meta.artifacts WHERE artifact_id=?", [artifact_id]).fetchone()
    if existing:
        return DataRef("output", target, artifact_id)
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quote(target.split('.')[0])}")
    storage = target
    expected_schema = [["_artifact_id", "VARCHAR"], *audit["schema"]]
    if _exists(conn, target) and describe(conn, target_sql) != expected_schema:
        if not metadata.get("schema_version"):
            raise ValueError(f"Schema mismatch for {target}; an explicit schema_version is required")
        # Different declared schemas have separate physical tables; never coerce a union.
        storage = target + "__schema_" + digest(audit["schema"])[:16]
    storage_sql = table_name(storage)
    if _exists(conn, storage):
        if describe(conn, storage_sql) != expected_schema:
            raise ValueError("Schema identity collision")
    else:
        conn.execute(f"CREATE TABLE {storage_sql} AS SELECT ''::VARCHAR AS _artifact_id, * FROM {relation} LIMIT 0")
    conn.execute(f"INSERT INTO {storage_sql} SELECT ?, * FROM {relation}", [artifact_id])
    conn.execute(
        "INSERT INTO meta.artifacts (artifact_id,target_table,schema_json,row_count,content_hash,metadata_json,original_file,access_class,storage_table) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [
            artifact_id,
            target,
            canonical(audit["schema"]),
            audit["rows"],
            content_hash,
            canonical(metadata),
            original_file,
            access_class,
            storage,
        ],
    )
    return DataRef("output", target, artifact_id)


def publish_artifact(
    path: str | Path,
    target: str,
    *,
    frame=None,
    file: str | Path | None = None,
    metadata: dict,
    access_class: str = "feature",
    _session=None,
) -> DataRef:
    if (frame is None) == (file is None):
        raise ValueError("Supply exactly one frame or frozen file")
    if file is not None:
        file = freeze_artifact_file(path, file)
    with write_store(path, session=_session) as conn:
        if file is not None:
            if file.suffix == ".parquet":
                conn.register("_publication", conn.from_parquet(str(file)))
            elif file.suffix == ".csv":
                conn.register("_publication", conn.read_csv(str(file), header=True))
            else:
                raise ValueError("Tabular publication supports Parquet and CSV")
        else:
            conn.register("_publication", frame)
        try:
            return publish_relation(
                conn,
                "_publication",
                target,
                metadata=metadata,
                original_file=str(file or ""),
                access_class=access_class,
            )
        finally:
            conn.unregister("_publication")


def publish_attachment(path: str | Path, file: str | Path, *, metadata: dict, _session=None) -> DataRef:
    """Register opaque immutable files without manufacturing numerical table rows."""
    source = freeze_artifact_file(path, file)
    content = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content.update(chunk)
    content_hash = content.hexdigest()
    artifact_id = digest({"file_hash": content_hash, "metadata": metadata, "kind": "attachment"})
    with write_store(path, session=_session) as conn:
        _set_role(conn, "output")
        conn.execute(
            "INSERT INTO meta.artifacts (artifact_id,target_table,schema_json,row_count,content_hash,metadata_json,original_file,access_class) "
            "VALUES (?, 'meta.artifacts', '[]', 0, ?, ?, ?, 'attachment') ON CONFLICT DO NOTHING",
            [artifact_id, content_hash, canonical(metadata), str(source)],
        )
    return DataRef("output", "meta.artifacts", artifact_id)


def publish_document(path: str | Path, document: Any, *, metadata: dict, original_file: str = "", _session=None) -> DataRef:
    if original_file:
        original_file = str(freeze_artifact_file(path, original_file))
        if json.dumps(json.loads(Path(original_file).read_text()), sort_keys=True) != json.dumps(document, sort_keys=True):
            raise ValueError("Document differs from its frozen file")
    try:
        body = canonical(document)
    except ValueError:
        if not metadata.get("legacy_source") or not original_file:
            raise
        # Preserve non-standard historical JSON verbatim; never replace NaN/Inf with guessed values.
        raw = Path(original_file).read_text(encoding="utf-8")
        body = canonical({"legacy_json_text": raw})
        metadata = {**metadata, "document_encoding": "legacy-json-text.v1"}
    content_hash = hashlib.sha256(body.encode()).hexdigest()
    artifact_id = digest({"table": "meta.documents", "content_hash": content_hash, "metadata": metadata})
    with write_store(path, session=_session) as conn:
        _set_role(conn, "output")
        conn.execute(
            "INSERT INTO meta.artifacts (artifact_id,target_table,schema_json,row_count,content_hash,metadata_json,original_file,access_class,storage_table) "
            "VALUES (?, 'meta.documents', ?, 1, ?, ?, ?, 'document', 'meta.documents') ON CONFLICT DO NOTHING",
            [artifact_id, canonical([["document", "JSON"]]), content_hash, canonical(metadata), original_file],
        )
        conn.execute(
            "INSERT INTO meta.documents VALUES (?, CAST(? AS JSON)) ON CONFLICT DO NOTHING", [artifact_id, body]
        )
    return DataRef("output", "meta.documents", artifact_id)


def read_document(path: str | Path, ref: DataRef) -> Any:
    _reject_sqlite_file(path)
    with duckdb.connect(str(path), read_only=True) as conn:
        artifact = resolve_artifact(conn, ref)
        if artifact["access_class"] != "document":
            raise TypeError("Expected a document artifact")
        row = conn.execute("SELECT document FROM meta.documents WHERE artifact_id=?", [ref.version]).fetchone()
        if row is None:
            raise ValueError("Document body is missing for the published artifact")
        value = json.loads(row[0])
        metadata = json.loads(artifact["metadata_json"])
        return (
            json.loads(value["legacy_json_text"])
            if metadata.get("document_encoding") == "legacy-json-text.v1"
            else value
        )


def resolve_artifact(conn, ref: DataRef, *, allow_labels: bool = False) -> dict:
    if ref.role != "output":
        raise ValueError("Expected an output reference")
    row = conn.execute(
        "SELECT target_table,schema_json,row_count,content_hash,metadata_json,original_file,access_class,coalesce(storage_table,target_table) "
        "FROM meta.artifacts WHERE artifact_id=?",
        [ref.version],
    ).fetchone()
    if row is None or row[0] != (ref.table if "." in ref.table else "main." + ref.table):
        raise ValueError("Published artifact reference does not exist")
    if row[6] in {"label", "evaluation"} and not allow_labels:
        raise PermissionError("Labels and evaluations require the protocol-authorized evaluation context")
    return dict(
        zip(
            (
                "table",
                "schema_json",
                "row_count",
                "content_hash",
                "metadata_json",
                "original_file",
                "access_class",
                "storage_table",
            ),
            row,
            strict=True,
        )
    )


def read_reference(
    path: str | Path,
    ref: DataRef,
    *,
    fields: list[str] | None = None,
    stocks: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    date_column: str = "date",
    stock_column: str = "stock_code",
    as_of: str | None = None,
    known_at_column: str = "known_at",
    allow_labels: bool = False,
):
    _reject_sqlite_file(path)
    with duckdb.connect(str(path), read_only=True) as conn:
        params = []
        predicates = []
        if ref.role == "output":
            artifact = resolve_artifact(conn, ref, allow_labels=allow_labels)
            if artifact["access_class"] in {"attachment", "document"}:
                raise TypeError("This artifact is not a numerical table")
            relation = table_name(artifact["storage_table"])
            predicates.append("_artifact_id = ?")
            params.append(ref.version)
            projection = ", ".join(quote(x) for x in fields) if fields is not None else "* EXCLUDE (_artifact_id)"
        else:
            rows = conn.execute(
                "SELECT DISTINCT physical_table FROM meta.datasets WHERE target_table=? AND version=?",
                [ref.table if "." in ref.table else "main." + ref.table, ref.version],
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("Input version missing or ambiguous")
            relation = table_name(rows[0][0])
            projection = ", ".join(quote(x) for x in fields) if fields is not None else "*"
        for bound, operator in ((start, ">="), (end, "<=")):
            if bound is not None:
                predicates.append(f"{quote(date_column)} {operator} ?")
                params.append(bound)
        if stocks is not None:
            predicates.append(f"{quote(stock_column)} IN (SELECT unnest(?))")
            params.append(stocks)
        if as_of is not None:
            predicates.append(f"{quote(known_at_column)} <= ?")
            params.append(as_of)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        return conn.execute(f"SELECT {projection} FROM {relation}{where}", params).fetchdf()


def _existing_input(conn, target, version, follow_head, source):
    policies = {row[0] for row in conn.execute(
        "SELECT DISTINCT follow_head FROM meta.datasets WHERE target_table=? AND source=?", [target, str(source)]
    ).fetchall()}
    if policies and policies != {follow_head}:
        raise ValueError("Input source follow_head policy is immutable")
    if not follow_head and conn.execute(
        "SELECT 1 FROM meta.datasets WHERE target_table=? AND source=? AND version<>?", [target, str(source), version]
    ).fetchone():
        raise ValueError("Frozen input source changed its content version")
    rows = conn.execute("SELECT DISTINCT physical_table FROM meta.datasets WHERE target_table=? AND version=?",
                        [target, version]).fetchall()
    if len(rows) > 1:
        raise ValueError("Input content version has ambiguous physical tables")
    return rows[0][0] if rows else None


def _register_input(conn, source_id, target, physical, source, version, audit, follow_head, source_table):
    existing = conn.execute("SELECT target_table FROM meta.datasets WHERE dataset_id=?", [source_id]).fetchone()
    if existing and existing[0] != target:
        raise ValueError("Input dataset is already registered with a different target")
    conn.execute(
        "INSERT INTO meta.datasets (dataset_id,target_table,physical_table,source,version,audit_json,follow_head,source_table) "
        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
        [source_id, target, physical, str(source), version, canonical(audit), follow_head,
         source_table if "." in source_table else "main." + source_table],
    )


def _migrate_indexed_table(source: Path, destination: Path, table: str, target: str, *, follow_head=True) -> dict:
    """Copy stable source row-id ranges in atomic batches, retaining source constraints.

    Range reads avoid a global primary-key sort. Source file identity and copy
    policy bind the checkpoints; completed ranges survive interruption. Only a
    fully checked table enters the input catalog.
    """
    if Path(str(source) + ".wal").exists():
        raise ValueError("Chunked migration requires a checkpointed source without WAL")
    stat = source.stat()
    source_state = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    with duckdb.connect(str(source), read_only=True) as reader:
        reader.execute("SET threads=1")
        reader.execute("SET memory_limit='3GB'")
        relation = table_name(table)
        audit = audit_relation(reader, relation)
        if any(name.lower() == "rowid" for name, _ in audit["schema"]):
            raise ValueError("Chunked migration requires an unshadowed DuckDB rowid")
        version = digest(audit)
        source_id = digest({"source": str(source), "table": table, "version": version})
        schema, name = (table if "." in table else "main." + table).split(".")
        create_sql = reader.execute(
            "SELECT sql FROM duckdb_tables() WHERE schema_name=? AND table_name=?", [schema, name]
        ).fetchone()[0]
        with write_store(destination) as conn:
            _set_role(conn, "input")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta.migration_chunks (migration_id VARCHAR, chunk_id VARCHAR, "
                "PRIMARY KEY(migration_id,chunk_id))"
            )
            existing = _existing_input(conn, target, version, follow_head, source)
            if existing:
                if audit_relation(conn, table_name(existing)) != audit:
                    raise ValueError("Previously migrated input changed")
                _register_input(conn, source_id, target, existing, source, version, audit, follow_head, table)
                return {"ok": True, "ref": asdict(DataRef("input", target, version)),
                        "physical_table": existing, "audit": audit, "follow_head": follow_head}
            physical = target if not _exists(conn, target) else "version_" + version[:16] + "." + name
            if _exists(conn, physical):
                raise ValueError("Unregistered destination appeared; refusing to overwrite")
            target_schema, target_name = physical.split(".")
            staging = target_schema + ".__migrate_rows_" + source_id[:24]
            checkpoint = "source:" + digest({"state": source_state, "follow_head": follow_head, "physical": physical})
            completed = {row[0] for row in conn.execute(
                "SELECT chunk_id FROM meta.migration_chunks WHERE migration_id=?", [source_id]
            ).fetchall()}
            if completed and checkpoint not in completed:
                raise ValueError("Migration source snapshot or copy policy changed; cannot reuse row-id checkpoints")
            if completed and not _exists(conn, staging):
                raise ValueError("Migration staging table is missing for existing checkpoints")
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quote(target_schema)}")
            if not _exists(conn, staging):
                conn.execute(f"CREATE TABLE {table_name(staging)} " + create_sql[create_sql.index("(") :])
            conn.execute("INSERT INTO meta.migration_chunks VALUES (?,?) ON CONFLICT DO NOTHING", [source_id, checkpoint])
        maximum = reader.execute(f"SELECT max(rowid) FROM {relation}").fetchone()[0]
        for lower in range(0, int(maximum) + 1 if maximum is not None else 0, 500_000):
            upper = lower + 500_000
            chunk = f"rows:{lower}:{upper}"
            if chunk in completed:
                continue
            batch = reader.execute(f"SELECT * FROM {relation} WHERE rowid >= ? AND rowid < ?",
                                   [lower, upper]).fetch_arrow_table()
            with write_store(destination) as conn:
                if conn.execute("SELECT 1 FROM meta.migration_chunks WHERE migration_id=? AND chunk_id=?",
                                [source_id, chunk]).fetchone():
                    continue
                conn.register("_chunk_source", batch)
                conn.execute(f"INSERT INTO {table_name(staging)} SELECT * FROM _chunk_source")
                conn.execute("INSERT INTO meta.migration_chunks VALUES (?,?)", [source_id, chunk])
        stat = source.stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != source_state or Path(str(source) + ".wal").exists():
            raise ValueError("Migration source changed during range copying")
        with write_store(destination) as conn:
            copied = audit_relation(conn, table_name(staging))
            if copied != audit:
                raise ValueError(f"Chunked migration verification failed for {table}")
            if _exists(conn, physical):
                raise ValueError("Destination appeared during staged migration; refusing to overwrite")
            conn.execute(f"ALTER TABLE {table_name(staging)} RENAME TO {quote(target_name)}")
            _register_input(conn, source_id, target, physical, source, version, audit, follow_head, table)
        return {"ok": True, "ref": asdict(DataRef("input", target, version)), "physical_table": physical,
                "audit": audit, "copy_mode": "rowid_chunks", "follow_head": follow_head}



def parquet_file_snapshot(source: str | Path) -> dict[str, str]:
    """Content identity of an explicitly migrated Parquet file or partition directory."""
    source = Path(source).resolve(strict=True)
    files = sorted(path for path in source.rglob("*.parquet") if not path.name.startswith("._")) if source.is_dir() else [source]
    if not files or any(path.suffix != ".parquet" for path in files):
        raise ValueError("Parquet migration requires a nonempty Parquet source")
    snapshot = {}
    for file in files:
        with file.open("rb") as handle:
            snapshot[str(file)] = hashlib.file_digest(handle, "sha256").hexdigest()
    return snapshot

def migrate_table(
    source: str | Path,
    destination: str | Path,
    table: str,
    *,
    role: str,
    target: str | None = None,
    execute: bool = False,
    chunked: bool = False,
    verify_existing: bool = False,
    follow_head: bool = True,
    source_format: str = "duckdb",
    metadata: dict | None = None,
) -> dict:
    source = Path(source).resolve(strict=True)
    _reject_sqlite_file(source)
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("Migration source and destination must differ")
    target = target or table
    target = target if "." in target else "main." + target
    source_table = table_name(table)
    table_name(target)
    if role not in {"input", "output"}:
        raise ValueError("Unknown database role")
    if not isinstance(follow_head, bool) or (role != "input" and not follow_head):
        raise ValueError("follow_head must be boolean and only applies to input sources")
    if verify_existing and (role != "input" or chunked):
        raise ValueError("Existing-table verification requires an unchunked input migration")
    if source_format == "sqlite":
        raise ValueError(
            "SQLite snapshots are unsupported; export the source to DuckDB or Parquet before publishing"
        )
    if source_format not in {"duckdb", "parquet"}:
        raise ValueError("Unknown migration source format")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict) or (metadata and role != "output"):
        raise ValueError("Migration metadata is only for output artifacts")
    if {"legacy_source", "legacy_table", "legacy_format", "legacy_files"} & metadata.keys():
        raise ValueError("Migration source provenance cannot be overridden")
    aliases = metadata.get("legacy_source_aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in aliases):
        raise ValueError("Legacy source aliases must be absolute paths")
    if source_format == "parquet":
        if role != "output" or chunked or verify_existing:
            raise ValueError("Parquet migration requires an output artifact")
        factor_name = metadata.get("legacy_factor_name")
        if factor_name is not None and (target != "main.factor_values" or not isinstance(factor_name, str) or not factor_name):
            raise ValueError("Parquet factor identity requires a factor_values table and a nonempty name")
        snapshot = parquet_file_snapshot(source)
        from pyarrow.parquet import read_schema

        schema = read_schema(next(iter(snapshot)))
        if any(not read_schema(file).equals(schema, check_metadata=False) for file in snapshot):
            raise ValueError("Parquet partitions have different schemas; declare separate artifact versions")
        if not execute:
            with duckdb.connect() as reader:
                reader.read_parquet(list(snapshot), hive_partitioning=False).create_view("_legacy_parquet")
                return {"source": str(source), "destination": str(destination), "table": table,
                        "target": target, "role": role, "schema": describe(reader, "_legacy_parquet"),
                        "files": len(snapshot), "execute": False, "source_format": source_format}
        frozen = []
        for file, checksum in snapshot.items():
            owned = freeze_artifact_file(destination, file)
            if owned.parent.name != checksum:
                raise ValueError("Parquet source changed while freezing")
            frozen.append({"source": file, "sha256": checksum, "file": str(owned)})
        if parquet_file_snapshot(source) != snapshot:
            raise ValueError("Parquet partition set changed while freezing")
        with write_store(destination) as conn:
            conn.register("_legacy_parquet", conn.read_parquet([item["file"] for item in frozen], hive_partitioning=False))
            try:
                audit = audit_relation(conn, "_legacy_parquet")
                if metadata.get("legacy_factor_name") and conn.execute(
                    "SELECT count(*) FROM _legacy_parquet WHERE factor_name IS DISTINCT FROM ?", [metadata["legacy_factor_name"]]
                ).fetchone()[0]:
                    raise ValueError("Parquet factor name differs from its declared identity")
                ref = publish_relation(conn, "_legacy_parquet", target,
                    metadata={"schema_version": "legacy-parquet-v1", **metadata, "legacy_source": str(source),
                              "legacy_table": table, "legacy_format": "parquet", "legacy_files": frozen},
                    access_class="evaluation" if target.endswith(("analysis_results", "monthly_pool_results")) else "feature")
            finally:
                conn.unregister("_legacy_parquet")
        return {"ok": True, "ref": asdict(ref), "audit": audit, "files": len(frozen), "source_format": source_format}
    if chunked and role != "input":
        raise ValueError("Chunked primary-key migration is for input tables")
    if not execute:
        with duckdb.connect(str(source), read_only=True) as conn:
            return {
                "source": str(source),
                "destination": str(destination),
                "table": table,
                "target": target,
                "role": role,
                "schema": describe(conn, source_table),
                "execute": False,
                "follow_head": follow_head,
            }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if chunked:
        return _migrate_indexed_table(source, destination, table, target, follow_head=follow_head)
    audit = None
    if verify_existing:
        # Keep large source/destination indexes out of the same DuckDB instance.
        # The physical clone already exists; verification requires no ATTACH/INSERT.
        with duckdb.connect(str(source), read_only=True) as reader:
            reader.execute("SET memory_limit='3GB'")
            reader.execute("SET threads=1")
            audit = audit_relation(reader, source_table)
    with write_store(destination) as conn:
        _set_role(conn, role)
        if audit is None:
            conn.execute("ATTACH '" + str(source).replace("'", "''") + "' AS migration_source (READ_ONLY)")
            relation = "migration_source." + source_table
            audit = audit_relation(conn, relation)
        if role == "output":
            ref = publish_relation(
                conn,
                relation,
                target,
                metadata={**metadata, "legacy_source": str(source), "legacy_table": table},
                original_file=str(source),
                access_class="evaluation"
                if target.endswith(("analysis_results", "monthly_pool_results"))
                else "feature",
            )
            return {"ok": True, "ref": asdict(ref), "audit": audit}
        version = digest(audit)
        source_id = digest({"source": str(source), "table": table, "version": version})
        old = _existing_input(conn, target, version, follow_head, source)
        physical = old or target
        if verify_existing and not _exists(conn, physical):
            raise ValueError(f"Expected cloned table is missing: {physical}")
        copied = None
        if not old and _exists(conn, physical):
            current = audit_relation(conn, table_name(physical))
            if current == audit:
                copied = current
            if current != audit:
                if verify_existing:
                    raise ValueError(f"Cloned table differs from the frozen source: {physical}")
                physical = "source_" + digest(str(source))[:16] + "." + target.split(".")[1]
                if _exists(conn, physical) and audit_relation(conn, table_name(physical)) != audit:
                    physical = "version_" + version[:16] + "." + target.split(".")[1]
        if not _exists(conn, physical):
            schema, name = physical.split(".")
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quote(schema)}")
            source_schema, source_name = (table if "." in table else "main." + table).split(".")
            create_sql = conn.execute(
                "SELECT sql FROM duckdb_tables() WHERE database_name='migration_source' AND schema_name=? AND table_name=?",
                [source_schema, source_name],
            ).fetchone()[0]
            conn.execute(f"CREATE TABLE {table_name(physical)} " + create_sql[create_sql.index("(") :])
            conn.execute(f"INSERT INTO {table_name(physical)} SELECT * FROM {relation}")
        copied = copied or audit_relation(conn, table_name(physical))
        if copied != audit:
            raise ValueError(f"Migration verification failed for {table}")
        _register_input(conn, source_id, target, physical, source, version, audit, follow_head, table)
        return {
            "ok": True,
            "ref": asdict(DataRef("input", target, version)),
            "physical_table": physical,
            "audit": audit,
            "follow_head": follow_head,
        }


def environment_json(name: str, default="{}"):
    import os

    raw = os.environ.get(name, default)
    if raw.startswith("@sha256:"):
        _, expected, path = raw.split(":", 2)
        content = Path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Environment transport checksum mismatch: {name}")
        return json.loads(content)
    return json.loads(raw)


def binding_contract(explicit: dict | None = None) -> dict:
    return validate_binding_contract(explicit if explicit is not None else environment_json("QUANT_DATA_BINDINGS"))


def environment_value(payload: dict, directory: Path) -> str:
    from utils.local_outputs import write_json

    raw = json.dumps(payload, sort_keys=True)
    if len(raw.encode()) <= 32768:
        return raw
    path = directory.resolve() / f"data-transport-{hashlib.sha256(raw.encode()).hexdigest()}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
    content = path.read_bytes()
    if json.loads(content) != payload:
        raise ValueError("Existing data transport differs from its content identity")
    return f"@sha256:{hashlib.sha256(content).hexdigest()}:{path}"


def validate_binding_contract(contract):
    """Reject retired path identities before reading or reusing frozen work."""
    if not isinstance(contract, dict):
        raise ValueError("Data bindings must be an object")
    if contract.get("sources"):
        raise ValueError(
            "Retired data binding sources are not supported; migrate the configuration "
            "to input_db/output_db and explicit table/version references"
        )
    return contract


def resolve_connection_binding(path, contract, *, allow_labels=False):
    """Resolve only the physical stores declared by an explicit version binding."""
    validate_binding_contract(contract)
    if not contract:
        return path, contract
    requested = str(Path(path).expanduser().resolve())
    role = next((r for r in ("input", "output")
                 if contract.get(r + "_db") and str(Path(contract[r + "_db"]).expanduser().resolve()) == requested), None)
    if role is None:
        raise ValueError(
            f"Database path has no frozen binding: {requested}; "
            "migrate the configuration to input_db/output_db and explicit table/version references"
        )
    selected = contract
    if role == "output" and allow_labels:
        selected = {**selected, "outputs": {**selected.get("outputs", {}), **selected.get("evaluations", {})}}
    return contract[role + "_db"], selected


def database_exists(path, *, bindings=None) -> bool:
    """Check the declared physical store without accepting retired path aliases."""
    destination, _ = resolve_connection_binding(path, binding_contract(bindings))
    return Path(destination).expanduser().is_file()


def bound_source_path(*candidates):
    """Find a declared physical store among config-relative path candidates."""
    contract = binding_contract()
    known = {str(Path(p).expanduser().resolve()) for p in (
        contract.get("input_db"), contract.get("output_db")
    ) if p}
    return next((p for p in candidates if str(p.resolve()) in known), None)


def _view_name(conn, alias: str) -> str:
    name = table_name(alias)
    schema = alias.split(".")[0] if "." in alias else "main"
    if schema in {"bound_input", "bound_output", "bound_store"}:
        raise ValueError("Binding alias conflicts with a reserved database name")
    if schema != "main":
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quote(schema)}")
    return name


def _output_selections(conn, raw, *, allow_labels: bool):
    """One alias may combine disjoint, explicitly named factors from frozen versions."""
    items = raw if isinstance(raw, list) else [raw]
    if not items:
        raise ValueError("An output binding cannot be empty")
    selected, claimed, shape = [], set(), None
    for item in items:
        ref = DataRef(**item.get("ref", item))
        artifact = resolve_artifact(conn, ref, allow_labels=allow_labels)
        if artifact["access_class"] in {"document", "attachment"}:
            raise ValueError("Documents and attachments are not numerical table bindings")
        names = item.get("factor_names") if "ref" in item else None
        if names is not None:
            if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x for x in names):
                raise ValueError("factor_names must be a nonempty list of names")
            if len(set(names)) != len(names) or claimed.intersection(names):
                raise ValueError("A factor cannot select multiple artifact versions in one binding")
            if "factor_name" not in {column[0] for column in json.loads(artifact["schema_json"])}:
                raise ValueError("Artifact has no factor_name column")
            claimed.update(names)
        elif len(items) > 1:
            raise ValueError("Multiple artifact versions require explicit disjoint factor_names")
        current = (artifact["table"], artifact["schema_json"])
        if shape is not None and shape != current:
            raise ValueError("A binding cannot implicitly combine different tables or schemas")
        shape = current
        selected.append((ref, artifact["storage_table"], names))
    return selected


def resolve_database_config(config: dict, *, bindings: dict | None = None) -> dict:
    """Resolve legacy path aliases once at the execution entry point."""
    from copy import deepcopy

    contract = binding_contract(bindings)
    if not contract:
        return config
    if not contract.get("input_db") or not contract.get("output_db"):
        raise ValueError("A data binding contract requires input_db and output_db")
    resolved = deepcopy(config)
    database = resolved.setdefault("database", {})
    if any(name in database for name in ("price_db", "factor_db", "local_db", "all_a_db")):
        for name, role in (
            ("price_db", "input"),
            ("all_a_db", "input"),
            ("factor_db", "output"),
            ("local_db", "output"),
        ):
            if name in database and not (name == "local_db" and database[name].get("isolated")):
                database[name]["duckdb_path"] = contract[role + "_db"]
    else:
        database["duckdb_path"] = contract["input_db"]
        database["db_backend"] = "duckdb"
    if "results_database" in resolved:
        resolved["results_database"].update(duckdb_path=contract["output_db"], db_backend="duckdb")
    resolved["data_bindings"] = contract
    return resolved


def input_catalog_name(source):
    """Persistent input views bind their catalog explicitly, including when attached."""
    columns = source.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema='meta' AND table_name='store'"
    ).fetchall()
    if ("catalog_name",) in columns:
        row = source.execute("SELECT catalog_name FROM meta.store").fetchone()
        if row and row[0]:
            return row[0]
    return source.execute("SELECT current_database()").fetchone()[0]


def read_frozen_attachment(path, content_hash, *, bindings=None):
    """Resolve the exact recorded file content after a working source has changed."""
    path = Path(path)
    if path.is_file():
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() == content_hash:
            return body
    contract = binding_contract(bindings)
    if contract.get("output_db"):
        _reject_sqlite_file(contract["output_db"])
        with duckdb.connect(contract["output_db"], read_only=True) as conn:
            rows = conn.execute(
                "SELECT DISTINCT original_file FROM meta.artifacts WHERE access_class='attachment' AND content_hash=?",
                [content_hash],
            ).fetchall()
        for (frozen,) in rows:
            if frozen and Path(frozen).is_file():
                body = Path(frozen).read_bytes()
                if hashlib.sha256(body).hexdigest() == content_hash:
                    return body
    raise ValueError(f"Frozen attachment with the recorded hash is unavailable: {path}")


def connect_input(path: str | Path, *, bindings: dict | None = None):
    """Legacy explicit paths replay unchanged; bound inputs expose exact versions."""
    contract = binding_contract(bindings)
    path, contract = resolve_connection_binding(path, contract)
    _reject_sqlite_file(path)
    source = duckdb.connect(str(path), read_only=True)
    try:
        _configure_resources(source)
        has_role = source.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='store'"
        ).fetchone()[0]
        if has_role and source.execute("SELECT role FROM meta.store").fetchone() == ("output",):
            raise ValueError("Output tables require exact artifact bindings through connect_output")
        if not contract.get("input_db") or Path(path).resolve() != Path(contract["input_db"]).resolve():
            return source
        refs = contract.get("inputs", {})
        if not refs:
            raise ValueError("Bound input reads require exact table versions")
        catalog = input_catalog_name(source)
        physical = {}
        for alias, raw in refs.items():
            ref = DataRef(**raw)
            if ref.role != "input":
                raise ValueError("Input connection cannot bind output artifacts")
            rows = source.execute(
                "SELECT DISTINCT physical_table FROM meta.datasets WHERE target_table=? AND version=?",
                [ref.table if "." in ref.table else "main." + ref.table, ref.version],
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(f"Input version missing or ambiguous: {ref}")
            physical[alias] = rows[0][0]
    except BaseException:
        source.close()
        raise
    source.close()
    conn = duckdb.connect()
    try:
        _configure_resources(conn)
        literal = "'" + str(Path(path).resolve()).replace("'", "''") + "'"
        conn.execute(f"ATTACH {literal} AS {quote(catalog)} (READ_ONLY)")
        for alias, table in physical.items():
            conn.execute(f"CREATE VIEW {_view_name(conn, alias)} AS SELECT * FROM {quote(catalog)}.{table_name(table)}")
        return conn
    except BaseException:
        conn.close()
        raise


def connect_output(path: str | Path, *, bindings: dict | None = None, allow_labels: bool = False):
    """Expose only explicitly frozen artifacts, with their original column shape."""
    contract = binding_contract(bindings)
    path, contract = resolve_connection_binding(path, contract, allow_labels=allow_labels)
    _reject_sqlite_file(path)
    source = duckdb.connect(str(path), read_only=True)
    try:
        _configure_resources(source)
        has_catalog = bool(
            source.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='artifacts'"
            ).fetchone()[0]
        )
        if not has_catalog:
            return source
        if not contract.get("output_db") or Path(path).resolve() != Path(contract["output_db"]).resolve():
            raise ValueError("Versioned output reads require an explicit output_db binding")
        refs = contract.get("outputs", {})
        if not refs:
            raise ValueError("Versioned output reads require exact artifact references")
        selections = {
            alias: _output_selections(source, raw, allow_labels=allow_labels)
            for alias, raw in refs.items()
        }
    except BaseException:
        source.close()
        raise
    source.close()
    conn = duckdb.connect()
    try:
        _configure_resources(conn)
        literal_path = "'" + str(Path(path).resolve()).replace("'", "''") + "'"
        conn.execute(f"ATTACH {literal_path} AS bound_output (READ_ONLY)")
        for alias, parts in selections.items():
            queries = []
            for ref, physical, names in parts:
                literal = "'" + ref.version.replace("'", "''") + "'"
                sql = f"SELECT * EXCLUDE (_artifact_id) FROM bound_output.{table_name(physical)} WHERE _artifact_id={literal}"
                if names is not None:
                    values = ",".join("'" + name.replace("'", "''") + "'" for name in names)
                    sql += f" AND factor_name IN ({values})"
                queries.append(sql)
            conn.execute(f"CREATE VIEW {_view_name(conn, alias)} AS " + " UNION ALL ".join(queries))
        return conn
    except BaseException:
        conn.close()
        raise


def bind_input_source(connection, source, tables, namespace, *, bindings=None):
    """Add source-specific aliases to an existing bound input connection."""
    path, selected = resolve_connection_binding(source, binding_contract(bindings))
    if not selected or not selected.get("input_db") or Path(path).resolve() != Path(selected["input_db"]).resolve():
        raise ValueError("Source aliases require a frozen input binding")
    catalogs = [name for _, name, file in connection.execute("PRAGMA database_list").fetchall()
                if file and Path(file).resolve() == Path(path).resolve()]
    if len(catalogs) != 1:
        raise ValueError("Frozen input database is not attached exactly once")
    catalog = quote(catalogs[0])
    selections = []
    for table in tables:
        raw = selected.get("inputs", {}).get(table)
        if raw is None:
            raise ValueError(f"Source table is not bound: {table}")
        ref = DataRef(**raw)
        if ref.role != "input":
            raise ValueError("Source aliases cannot expose output artifacts")
        rows = connection.execute(
            f"SELECT DISTINCT physical_table FROM {catalog}.meta.datasets WHERE target_table=? AND version=?",
            [ref.table if "." in ref.table else "main." + ref.table, ref.version],
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(f"Input version missing or ambiguous: {ref}")
        selections.append((table, rows[0][0]))
    for table, physical in selections:
        alias = _view_name(connection, namespace + "." + table)
        connection.execute(f"CREATE OR REPLACE VIEW {alias} AS SELECT * FROM {catalog}.{table_name(physical)}")


def connect_read_only(path: str | Path, *, bindings: dict | None = None, allow_labels: bool = False):
    contract = binding_contract(bindings)
    path, contract = resolve_connection_binding(path, contract, allow_labels=allow_labels)
    if contract.get("output_db") and Path(path).resolve() == Path(contract["output_db"]).resolve():
        return connect_output(path, bindings=contract, allow_labels=allow_labels)
    return connect_input(path, bindings=contract)


def output_context(explicit: dict | None = None) -> dict:

    context = explicit if explicit is not None else environment_json("QUANT_OUTPUT_CONTEXT")
    if context and not context.get("legacy_source"):
        required = {"pool_id", "source_hash", "config_hash", "input_versions", "created_at"}
        if (not (context.get("run_id") or context.get("build_id"))
                or not required.issubset(context) or any(context[key] is None for key in required)):
            raise ValueError("Output context must freeze run/build, pool, source, config, inputs and created_at")
    return context


def freeze_artifact_file(path: str | Path, file: str | Path) -> Path:
    """Keep a verified immutable copy beside the output store, reusable after interruption."""
    import shutil

    source = Path(file).resolve(strict=True)
    content = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content.update(chunk)
    frozen = Path(path).resolve().parent / "artifacts" / content.hexdigest() / source.name
    if frozen.parent.parent.is_symlink() or frozen.parent.is_symlink():
        raise ValueError("Frozen artifact directory cannot be a symlink")
    if source.parent.parent == frozen.parent.parent:
        if source.parent.name != content.hexdigest():
            raise ValueError("Frozen artifact content changed")
        return source
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.is_symlink():
        raise ValueError("Frozen artifact cannot be a symlink")
    import os
    import tempfile
    # Publish a fully verified file without replacing an existing immutable object.
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=frozen.parent) as staging:
        temporary = Path(staging) / source.name
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != content.hexdigest():
                raise ValueError("Source changed during artifact publication")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        from utils.local_outputs import output_lock
        # ponytail: one publication lock per artifact root; shard by hash if throughput requires it.
        # Rename also supports external filesystems without hard-link support.
        with output_lock(frozen.parent.parent, metadata=True):
            if frozen.is_symlink():
                raise ValueError("Frozen artifact cannot be a symlink")
            if frozen.exists():
                with frozen.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != content.hexdigest():
                        raise ValueError("Frozen artifact content collision")
            else:
                os.replace(temporary, frozen)
    return frozen


def publish_outputs(files: dict[str, str | Path], *, metadata: dict | None = None, bindings: dict | None = None) -> dict:
    """Publish a completed producer's files, retaining immutable replay copies."""
    contract = binding_contract(bindings)
    if not contract.get("output_db") or contract.get("publish_outputs") is False:
        return {}
    context = output_context(metadata)
    if not context:
        raise ValueError("Configured output publication requires frozen provenance")
    refs = {}
    for target, raw in files.items():
        source = Path(raw).resolve(strict=True)
        frozen = freeze_artifact_file(contract["output_db"], source)
        provenance = {**context, "original_source": str(source)}
        if target == "meta.artifacts" or target.startswith("meta.artifacts."):
            ref = publish_attachment(contract["output_db"], frozen, metadata=provenance)
        elif source.suffix == ".json":
            ref = publish_document(
                contract["output_db"], json.loads(frozen.read_text()), metadata=provenance, original_file=str(frozen)
            )
        elif source.suffix not in {".parquet", ".csv"}:
            ref = publish_attachment(contract["output_db"], frozen, metadata=provenance)
        else:
            ref = publish_artifact(contract["output_db"], target, file=frozen, metadata=provenance)
        refs[target] = asdict(ref)
    return refs


def migrate_tables(tables: list[dict], *, execute: bool = False, checkpoint: str | None = None) -> dict:
    """One wrapper process, independent per-table transactions and durable receipts."""
    results = []

    def save():
        result = {"ok": len(results) == len(tables), "execute": execute, "results": results,
                  "manifest_hash": digest(tables)}
        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(canonical(result) + "\n")
            temporary.replace(path)
        return result

    save()  # A failed invocation must not leave an earlier success receipt in place.
    for entry in tables:
        request = {**entry, "execute": execute}
        action = request.pop("action", "migrate")
        if action == "clone":
            from utils.database_copy import clone_database

            result = clone_database(**request)
        elif action == "migrate":
            result = migrate_table(**request)
        elif action == "adopt_files":
            result = adopt_artifact_files(**request)
        else:
            raise ValueError(f"Unsupported migration action: {action}")
        if result.get("ok") is False:
            raise ValueError(f"Migration failed for {entry['source']}")
        results.append({"source": entry["source"], "table": entry.get("table", "<database>"), **result})
        save()
    return save()


def adopt_artifact_files(source: str | Path, artifact_ids: list[str], *, execute: bool = False) -> dict:
    """Move file custody without changing published identities or deleting source files."""
    if not artifact_ids or len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("Supply distinct published artifact IDs")
    _reject_sqlite_file(source)
    columns = "artifact_id,original_file,access_class,content_hash,row_count,metadata_json"
    with duckdb.connect(str(source), read_only=True) as conn:
        catalog = {row[0]: row for row in conn.execute(f"SELECT {columns} FROM meta.artifacts").fetchall()}
    if set(artifact_ids) - catalog.keys():
        raise ValueError("Unknown published artifact ID")
    results = []
    with duckdb.connect() as conn:
        for artifact in artifact_ids:
            _, original, kind, expected, count, metadata = catalog[artifact]
            file = Path(original).resolve(strict=True)
            if not original or not file.is_file():
                raise ValueError("Artifact original_file must identify a regular file")
            if kind not in {"document", "attachment"} and file.suffix not in {".parquet", ".csv"}:
                raise ValueError("Only frozen files can be adopted; database sources use table migration")
            if not execute:
                results.append({"artifact_id": artifact, "original_file": str(file), "bytes": file.stat().st_size})
                continue
            frozen = freeze_artifact_file(source, file)
            if kind == "document":
                raw = frozen.read_text(encoding="utf-8")
                value = ({"legacy_json_text": raw} if json.loads(metadata).get("document_encoding") == "legacy-json-text.v1"
                         else json.loads(raw))
                actual, rows = digest(value), 1
            elif kind == "attachment":
                with frozen.open("rb") as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                rows = 0
            else:
                relation = conn.from_parquet(str(frozen)) if frozen.suffix == ".parquet" else conn.read_csv(str(frozen), header=True)
                relation.create_view("_custody")
                audit = audit_relation(conn, "_custody")
                actual, rows = digest(audit), audit["rows"]
            if actual != expected or rows != count:
                raise ValueError(f"Frozen file differs from published artifact: {artifact}")
            results.append({"artifact_id": artifact, "original_file": original, "frozen_file": str(frozen), "content_hash": actual})
    if execute:
        with write_store(source) as conn:
            _set_role(conn, "output")
            current = {row[0]: row for row in conn.execute(f"SELECT {columns} FROM meta.artifacts").fetchall()}
            if any(current.get(artifact) != catalog[artifact] for artifact in artifact_ids):
                raise ValueError("Artifact catalog changed during file adoption; retry against the current catalog")
            conn.executemany("UPDATE meta.artifacts SET original_file=? WHERE artifact_id=?",
                             [(row["frozen_file"], row["artifact_id"]) for row in results])
    return {"ok": True, "execute": execute, "artifacts": len(results), "files": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="JSON request file from the harness")
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)
    Path(args.response).unlink(missing_ok=True)
    _check_hash_engine()
    request = json.loads(Path(args.request).read_text())
    action = request.pop("action")
    if action == "clone":
        from utils.database_copy import clone_database

        result = clone_database(**request)
    elif action == "migrate":
        result = migrate_table(**request)
    elif action == "migrate_batch":
        result = migrate_tables(**request)
    elif action == "merge_input":
        from utils.input_consolidation import merge_input_tables
        result = merge_input_tables(**request)
    elif action == "publish_input":
        from utils.input_store import publish_input_batch

        result = publish_input_batch(**request)
    elif action == "publish_price_factors":
        from utils.input_store import publish_price_factors

        result = publish_price_factors(**request)
    elif action == "publish":
        result = asdict(publish_artifact(**request))
    elif action == "publish_files":
        result = {"ok": True, "refs": publish_outputs(**request)}
    elif action == "publish_batch":
        from utils.duckdb_results_writer import acquire_exclusive_write_conn

        results = []
        with acquire_exclusive_write_conn(request["path"]) as session:
            for item in request["items"]:
                source = Path(item["file"])
                if not source.is_file():
                    results.append({"file": str(source), "ok": False, "error": "frozen artifact missing"})
                    if request.get("checkpoint"):
                        Path(request["checkpoint"]).write_text(canonical(results) + "\n")
                    continue
                try:
                    if "sha256" in item:
                        source = freeze_artifact_file(request["path"], source)
                        if source.parent.name != item["sha256"]:
                            raise ValueError("Frozen import file does not match the declared checksum")
                    if item.get("table") == "meta.artifacts":
                        ref = publish_attachment(request["path"], source, metadata=item["metadata"], _session=session)
                    elif source.suffix == ".json":
                        ref = publish_document(
                            request["path"],
                            json.loads(source.read_text()),
                            metadata=item["metadata"],
                            original_file=str(source),
                            _session=session,
                        )
                    else:
                        ref = publish_artifact(
                            request["path"],
                            item["table"],
                            file=source,
                            metadata=item["metadata"],
                            access_class=item.get("access_class", "feature"),
                            _session=session,
                        )
                    results.append({"file": str(item["file"]), "ok": True, "ref": asdict(ref)})
                except (ValueError, OSError, duckdb.Error) as exc:
                    if "corrupt database file" in str(exc).lower() or "no space left" in str(exc).lower():
                        raise  # Storage failures stop the batch; do not keep writing a damaged store.
                    results.append({"file": str(item["file"]), "ok": False, "error": str(exc)})
                if request.get("checkpoint"):
                    Path(request["checkpoint"]).write_text(canonical(results) + "\n")
        result = {"ok": all(item["ok"] for item in results), "results": results}
    elif action == "document":
        result = asdict(publish_document(**request))
    elif action == "export":
        output = Path(request.pop("destination"))
        request["ref"] = DataRef(**request["ref"])
        frame = read_reference(**request)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        result = {"ok": True, "rows": len(frame), "path": str(output)}
    else:
        parser.error(f"Unknown data store action: {action}")
    Path(args.response).write_text(canonical(result) + "\n")
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
