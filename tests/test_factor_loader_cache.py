"""Tests for the cross-trial factor-read cache in FactorLoader.

The cache lets a multi-trial backtest sweep (e.g. a top_pct grid) reuse the
identical per-date raw factor reads instead of re-hitting DuckDB every trial.

Contract:
1. A cache hit returns data equal to a fresh (uncached) query.
2. Cache hits do not issue new DB queries (verified via a query counter).
3. Mutating a returned frame never corrupts the cached entry.
4. With no shared cache (default), behaviour is unchanged (no caching).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

duckdb = pytest.importorskip("duckdb")

FactorLoader = importlib.import_module(
    "factor_synthesis.vectorbt_adapter.factor_loader"
).FactorLoader


def _make_conn():
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE factor_values ("
        "date VARCHAR, stock_code VARCHAR, factor_name VARCHAR, factor_value DOUBLE)"
    )
    rows = []
    rng = np.random.default_rng(0)
    for date in ("2024-01-02", "2024-01-03"):
        for stock in ("S1", "S2", "S3"):
            for fname in ("facA", "facB"):
                rows.append((date, stock, fname, float(rng.normal())))
    conn.executemany(
        "INSERT INTO factor_values VALUES (?, ?, ?, ?)", rows
    )
    return conn


class _CountingConn:
    """Wrap a duckdb connection to count execute() calls."""

    def __init__(self, conn):
        self._conn = conn
        self.execute_count = 0

    def execute(self, *args, **kwargs):
        if str(args[0]).lstrip().upper().startswith("SELECT DATE, STOCK_CODE, FACTOR_NAME, FACTOR_VALUE"):
            self.execute_count += 1
        return self._conn.execute(*args, **kwargs)


def test_cache_hit_matches_fresh_query():
    conn = _make_conn()
    shared: dict = {}
    loader = FactorLoader(existing_conn=conn, shared_cache=shared)

    first = loader.load_factors("2024-01-02", ["S1", "S2", "S3"], ["facA", "facB"])
    second = loader.load_factors("2024-01-02", ["S1", "S2", "S3"], ["facA", "facB"])

    pd.testing.assert_frame_equal(first, second)
    assert len(shared) == 1


def test_cache_avoids_second_db_query():
    counting = _CountingConn(_make_conn())
    shared: dict = {}
    loader = FactorLoader(existing_conn=counting, shared_cache=shared)

    loader.load_factors("2024-01-02", ["S1", "S2"], ["facA"])
    count_after_first = counting.execute_count
    loader.load_factors("2024-01-02", ["S1", "S2"], ["facA"])

    assert counting.execute_count == count_after_first  # no new query on hit


def test_mutation_does_not_corrupt_cache():
    conn = _make_conn()
    shared: dict = {}
    loader = FactorLoader(existing_conn=conn, shared_cache=shared)

    first = loader.load_factors("2024-01-02", ["S1", "S2"], ["facA"])
    first.iloc[0, 0] = 999.0  # mutate the returned frame

    second = loader.load_factors("2024-01-02", ["S1", "S2"], ["facA"])
    assert second.iloc[0, 0] != 999.0


def test_disabled_cache_is_default():
    conn = _make_conn()
    loader = FactorLoader(existing_conn=conn)  # no shared_cache

    out = loader.load_factors("2024-01-02", ["S1", "S2"], ["facA"])
    assert not out.empty
    assert loader._load_cache is None


def test_cross_loader_sharing():
    """Two loaders sharing a cache reuse each other's reads (sweep scenario)."""
    counting = _CountingConn(_make_conn())
    shared: dict = {}
    loader_a = FactorLoader(existing_conn=counting, shared_cache=shared)
    loader_b = FactorLoader(existing_conn=counting, shared_cache=shared)

    loader_a.load_factors("2024-01-03", ["S1", "S2", "S3"], ["facA", "facB"])
    count_after_a = counting.execute_count
    # Second loader, identical query -> served from shared cache, no new query.
    loader_b.load_factors("2024-01-03", ["S1", "S2", "S3"], ["facA", "facB"])

    assert counting.execute_count == count_after_a


def test_prefetch_preserves_exact_pool_missing_data_and_fallback():
    counting = _CountingConn(_make_conn())
    fresh = FactorLoader(existing_conn=counting)
    batched = FactorLoader(existing_conn=counting)
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    stocks = ["S3", "S1", "missing"]
    factors = ["facB", "facA", "missing"]
    expected = [fresh.load_factors(date, stocks, factors) for date in dates]
    batched.prefetch(dates, stocks, factors)
    after_batch = counting.execute_count
    for date, baseline in zip(dates, expected):
        actual = batched.load_factors(date, stocks, factors)
        pd.testing.assert_frame_equal(actual, baseline)
        actual.iloc[0, 0] = 999
        pd.testing.assert_frame_equal(batched.load_factors(date, stocks, factors), baseline)
    pd.testing.assert_frame_equal(
        batched.load_factors(dates[0], ["S1"], ["facA"]),
        expected[0].loc[["S1"], ["facA"]],
    )
    assert counting.execute_count == after_batch
    # A pool or date outside the batch still uses the ordinary query path.
    pd.testing.assert_frame_equal(
        batched.load_factors(dates[0], ["S2"], ["facA"]),
        fresh.load_factors(dates[0], ["S2"], ["facA"]),
    )
    assert counting.execute_count == after_batch + 2
    batched.prefetch(["2023-12-31"], stocks, factors)
    pd.testing.assert_frame_equal(
        batched.load_factors("2023-12-31", stocks, factors),
        fresh.load_factors("2023-12-31", stocks, factors),
    )

def test_shared_factor_cache_and_prefetch_expire_on_import():
    from data_pipeline.loaders._transactions import import_transaction
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE factor_values AS SELECT '2024-01-31' AS date, 'A' stock_code, 'f' factor_name, 10. AS factor_value")
        shared = {}
        first = FactorLoader(existing_conn=conn, shared_cache=shared)
        second = FactorLoader(existing_conn=conn, shared_cache=shared)
        first.prefetch(['2024-01-31'], ['A'], ['f'])
        assert first.load_factors('2024-01-31', ['A'], ['f']).iloc[0, 0] == 10
        assert second.load_factors('2024-01-31', ['A'], ['f']).iloc[0, 0] == 10
        with import_transaction(conn):
            conn.execute('UPDATE factor_values SET factor_value=20')
        assert second.load_factors('2024-01-31', ['A'], ['f']).iloc[0, 0] == 20
        assert first.load_factors('2024-01-31', ['A'], ['f']).iloc[0, 0] == 20


def test_shared_cache_isolates_memory_tables_with_same_attached_file(tmp_path):
    path = tmp_path / 'shared.duckdb'
    with duckdb.connect(str(path)) as disk:
        disk.execute('CREATE TABLE unrelated(i INTEGER)')
    first, second = _make_conn(), _make_conn()
    try:
        for conn, value in [(first, 10), (second, 99)]:
            conn.execute('UPDATE factor_values SET factor_value=?', [value])
            conn.execute("ATTACH '" + str(path).replace("'", "''") + "' AS shared (READ_ONLY)")
        shared = {}
        loaders = [FactorLoader(existing_conn=c, shared_cache=shared) for c in (first, second)]
        assert loaders[0].load_factors('2024-01-02', ['S1'], ['facA']).iloc[0, 0] == 10
        assert loaders[1].load_factors('2024-01-02', ['S1'], ['facA']).iloc[0, 0] == 99
    finally:
        first.close()
        second.close()


def test_shared_cache_isolates_different_connect_input_versions(tmp_path):
    from utils.data_store import connect_input, migrate_table
    source, destination = tmp_path / 'source.duckdb', tmp_path / 'input.duckdb'
    with duckdb.connect(str(source)) as conn:
        conn.execute("CREATE TABLE factor_values AS SELECT DATE '2024-01-02' AS date, 'S1' stock_code, 'facA' factor_name, 10.0 AS factor_value")
    old = migrate_table(source, destination, 'factor_values', role='input', execute=True)['ref']
    with duckdb.connect(str(source)) as conn:
        conn.execute('UPDATE factor_values SET factor_value=99')
    new = migrate_table(source, destination, 'factor_values', role='input', execute=True)['ref']
    assert old['version'] != new['version']
    with connect_input(destination, bindings={'input_db':str(destination), 'inputs':{'factor_values':old}}) as first:
        with connect_input(destination, bindings={'input_db':str(destination), 'inputs':{'factor_values':new}}) as second:
            shared = {}
            loaders = [FactorLoader(existing_conn=c, shared_cache=shared) for c in (first, second)]
            assert loaders[0].load_factors('2024-01-02', ['S1'], ['facA']).iloc[0, 0] == 10
            assert loaders[1].load_factors('2024-01-02', ['S1'], ['facA']).iloc[0, 0] == 99


def test_factor_loader_rejects_missing_relation_and_recovers():
    import pytest
    with duckdb.connect() as conn:
        shared = {}
        loader = FactorLoader(existing_conn=conn, shared_cache=shared)
        with pytest.raises(duckdb.CatalogException):
            loader.load_factors('2024-01-02', ['A'], ['f'])
        assert shared == {}
        conn.execute("CREATE TABLE factor_values AS SELECT DATE '2024-01-02' AS date, 'A' stock_code, 'f' factor_name, 10.0 AS factor_value")
        assert loader.load_factors('2024-01-02', ['A'], ['f']).iloc[0, 0] == 10


def test_factor_loader_rejects_duplicate_keys_and_clears_prefetch():
    import pytest
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE factor_values AS SELECT DATE '2024-01-02' AS date, 'A' stock_code, 'f' factor_name, 10.0 AS factor_value")
        conn.execute("INSERT INTO factor_values VALUES ('2024-01-02','A','f',11), ('2024-01-02','B','f',20)")
        shared = {}
        loader = FactorLoader(existing_conn=conn, shared_cache=shared)
        loader.prefetch(['2024-01-02'], ['A', 'B'], ['f'])
        with pytest.raises(ValueError, match='duplicate'):
            loader.load_factors('2024-01-02', ['A', 'B'], ['f'])
        assert shared == {} and loader._prefetched == {} and loader._prefetch_dates == set()
        conn.execute('DELETE FROM factor_values WHERE factor_value=11')
        actual = loader.load_factors('2024-01-02', ['A', 'B'], ['f'])
        assert actual['f'].tolist() == [10, 20]
