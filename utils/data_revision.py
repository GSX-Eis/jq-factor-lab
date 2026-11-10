"""Import generations for disposable caches, independent of research versions.

Writers advance the generation in the data transaction. Readers never create
metadata, commit a borrowed connection, or move a frozen input binding.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import json
import os
import uuid


def advance_revision(conn):
    """Call inside the writer's transaction, immediately before COMMIT."""
    conn.execute('CREATE SCHEMA IF NOT EXISTS meta')
    conn.execute('CREATE TABLE IF NOT EXISTS meta.cache_revision '
                 '(singleton BOOLEAN PRIMARY KEY CHECK(singleton), generation VARCHAR NOT NULL)')
    generation = uuid.uuid4().hex
    conn.execute('INSERT OR REPLACE INTO meta.cache_revision VALUES (TRUE, ?)', [generation])
    return generation


def connection_revision(conn):
    """Read generations from all attached catalogs, including bound inputs."""
    from utils.data_store import quote
    catalogs = conn.execute('PRAGMA database_list').fetchall()
    registered = {r[0] for r in conn.execute(
        "SELECT database_name FROM duckdb_tables() "
        "WHERE schema_name='meta' AND table_name='cache_revision'").fetchall()}
    result = []
    for _, name, path in catalogs:
        generation = None
        if name in registered:
            rows = conn.execute(f'SELECT generation FROM {quote(name)}.meta.cache_revision').fetchall()
            if len(rows) != 1 or not isinstance(rows[0][0], str) or not rows[0][0]:
                raise RuntimeError(f'Invalid cache revision in {name}')
            generation = rows[0][0]
        result.append((name, path, generation))
    return tuple(result)


def database_revision(path):
    """Read a file's committed generation without writing or applying bindings."""
    import duckdb
    with duckdb.connect(str(path), read_only=True) as conn:
        return [list(row) for row in connection_revision(conn)]


_PARQUET_MARKER = '.cache-revision.json'


def parquet_revision(directory):
    """Unmanaged Parquet is readable but cannot reuse application caches."""
    marker = Path(directory) / _PARQUET_MARKER
    if not marker.exists():
        return None
    payload = json.loads(marker.read_text())
    if (payload.get('state') != 'ready' or not payload.get('generation')
            or ('version' in payload and
                (payload['version'] != 2 or payload.get('pending_datasets') != []))):
        raise RuntimeError(f'Parquet export is incomplete: {directory}; rerun the export')
    return payload['generation']


@contextmanager
def parquet_publication(directory, dataset):
    """Block cached/partial reads while an exporter changes a directory.

    Dataset is its relative output subdirectory. Failed datasets stay pending
    until their own successful export. Legacy pending requires manual recovery.
    """
    import fcntl
    if (not isinstance(dataset, str) or not dataset or Path(dataset).is_absolute()
            or dataset != Path(dataset).as_posix() or '..' in Path(dataset).parts
            or dataset == '.'):
        raise ValueError('Parquet dataset must be a relative output subdirectory')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.cache-revision.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / _PARQUET_MARKER
        pending = set()
        if marker.exists():
            payload = json.loads(marker.read_text())
            if 'version' not in payload:
                if payload.get('state') != 'ready' or not payload.get('generation'):
                    raise RuntimeError(f'Parquet export is incomplete: {directory}; legacy pending requires manual recovery')
            else:
                entries = payload.get('pending_datasets')
                if (payload['version'] != 2 or not isinstance(entries, list)
                        or any(not isinstance(name, str) or not name or name == '.'
                               or Path(name).is_absolute() or '..' in Path(name).parts
                               or name != Path(name).as_posix() for name in entries)
                        or payload.get('state') != ('pending' if entries else 'ready')
                        or not payload.get('generation')):
                    raise RuntimeError(f'Parquet export is incomplete: {directory}; invalid publication marker')
                pending.update(entries)
        def mark(state):
            generation = uuid.uuid4().hex
            temporary = root / f'.cache-revision-{generation}.tmp'
            temporary.write_text(json.dumps({'version': 2, 'state': state,
                                            'generation': generation,
                                            'pending_datasets': sorted(pending)}))
            os.replace(temporary, marker)
        pending.add(dataset)
        mark('pending')
        yield
        pending.remove(dataset)
        mark('pending' if pending else 'ready')


def published_parquet(function=None, *, dataset=None):
    """Use a fixed dataset, or the price exporter's table_name argument."""
    if function is None:
        from functools import partial
        return partial(published_parquet, dataset=dataset)
    @wraps(function)
    def wrapped(owner, *args, **kwargs):
        if not owner.export_parquet:
            return function(owner, *args, **kwargs)
        selected = dataset if dataset is not None else (args[0] if args else kwargs['table_name'])
        with parquet_publication(owner.parquet_dir, selected):
            return function(owner, *args, **kwargs)
    return wrapped


def reset_parquet_dataset(output_path):
    """Remove prior derived partitions while the root publication is pending.

    These exporters copy the complete database table. Leaving a partition that
    disappeared from the table would expose old rows under the new generation.
    The DuckDB source remains authoritative and a failed export can be rerun.
    """
    for path in Path(output_path).rglob('*.parquet'):
        path.unlink()


def revision_cached(function):
    """Validate before a hit and reject a source change during a calculation.

    Owners supply cache_revision() and clear_cache(). None means unversioned
    Parquet: clear both before and after this call, without changing its result.
    """
    @wraps(function)
    def wrapped(owner, *args, **kwargs):
        try:
            before = owner.cache_revision()
        except Exception:
            owner.clear_cache()
            raise
        if before is None or (hasattr(owner, '_cache_revision') and before != owner._cache_revision):
            owner.clear_cache()
        owner._cache_revision = before
        try:
            result = function(owner, *args, **kwargs)
            if owner.cache_revision() != before:
                raise RuntimeError('Input data changed during calculation; retry with fixed inputs')
            return result
        except BaseException:
            owner.clear_cache()
            raise
        finally:
            if before is None:
                owner.clear_cache()
    return wrapped


def main(argv=None):
    """Explicit invalidation after maintenance outside the supported importers."""
    import argparse
    import duckdb
    from data_pipeline.loaders._transactions import import_transaction
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.database.is_file():
        parser.error('database must be an existing file')
    with duckdb.connect(str(args.database)) as conn:
        with import_transaction(conn):
            pass
        print(json.dumps({'database': str(args.database.resolve()),
                          'cache_revision': connection_revision(conn)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
