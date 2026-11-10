"""Keep each local import operation atomic, including explicit table replacement."""

from functools import wraps
from contextlib import contextmanager

import duckdb
from utils.data_revision import advance_revision


def atomic_import(function):
    """Run an importer entry point in one DuckDB transaction."""

    @wraps(function)
    def wrapped(owner, *args, **kwargs):
        connection = owner if isinstance(owner, duckdb.DuckDBPyConnection) else owner.conn
        if connection is None:
            connection = owner.connect()
        with import_transaction(connection):
            if hasattr(owner, "import_failed"):
                owner.import_failed = False
            result = function(owner, *args, **kwargs)
            if getattr(owner, "import_failed", False):
                raise RuntimeError("Import failed; database changes were rolled back")
            return result

    return wrapped


@contextmanager
def import_transaction(connection):
    """Commit data and its cache generation together, or roll back both."""
    connection.execute("BEGIN TRANSACTION")
    try:
        yield connection
        advance_revision(connection)
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
