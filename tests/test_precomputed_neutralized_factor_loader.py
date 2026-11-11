from __future__ import annotations

import hashlib
import importlib
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

duckdb = pytest.importorskip("duckdb")

load_factor_data = importlib.import_module(
    "factor_synthesis.core.factor_data_loader"
).load_factor_data
factor_data_loader = importlib.import_module("factor_synthesis.core.factor_data_loader")


def _make_conn():
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE factor_values ("
        "date VARCHAR, stock_code VARCHAR, factor_name VARCHAR, factor_value DOUBLE)"
    )
    conn.execute(
        "CREATE TABLE neutralized_factor_values ("
        "date VARCHAR, stock_code VARCHAR, factor_name VARCHAR, raw_factor_name VARCHAR, "
        "pool_id VARCHAR, factor_value DOUBLE, run_id VARCHAR, created_at TIMESTAMP)"
    )
    raw_rows = [
        ("2024-01-01", "S1", "FactorA", 1.0),
        ("2024-01-01", "S2", "FactorA", 2.0),
        ("2024-01-02", "S1", "FactorA", 3.0),
        ("2024-01-02", "S2", "FactorA", 4.0),
        ("2024-01-01", "S1", "FactorB", 10.0),
        ("2024-01-01", "S2", "FactorB", 20.0),
        ("2024-01-02", "S1", "FactorB", 30.0),
        ("2024-01-02", "S2", "FactorB", 40.0),
    ]
    neutralized_rows = [
        (
            "2024-01-01",
            "S1",
            "FactorA__neu",
            "FactorA",
            "wind_micro",
            101.0,
            "r1",
            None,
        ),
        (
            "2024-01-01",
            "S2",
            "FactorA__neu",
            "FactorA",
            "wind_micro",
            102.0,
            "r1",
            None,
        ),
        (
            "2024-01-02",
            "S1",
            "FactorA__neu",
            "FactorA",
            "wind_micro",
            103.0,
            "r1",
            None,
        ),
        (
            "2024-01-02",
            "S2",
            "FactorA__neu",
            "FactorA",
            "wind_micro",
            104.0,
            "r1",
            None,
        ),
    ]
    conn.executemany("INSERT INTO factor_values VALUES (?, ?, ?, ?)", raw_rows)
    conn.executemany(
        "INSERT INTO neutralized_factor_values VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        neutralized_rows,
    )
    return conn


def _load(conn, factor_names, *, neutralization_config=None, config=None, logger=None):
    from calendar_test_support import seed_calendar
    with duckdb.connect(":memory:") as price_conn:
        seed_calendar(price_conn)
        return load_factor_data(
            factor_conn=conn,
            price_conn=price_conn,
            factor_table="factor_values",
            factor_names=factor_names,
            start_date="2024-01-01",
            end_date="2024-01-02",
            config=config or {"cache_settings": {"use_parquet_cache": False}},
            neutralization_config=neutralization_config,
            custom_logger=logger or logging.getLogger("test_neutralized_loader"),
        )


def test_stock_filter_sql_chunks_are_canonical(monkeypatch):
    queries = []

    class RecordingConnection:
        def execute(self, sql):
            queries.append(" ".join(sql.split()))
            return self

        def fetchdf(self):
            return pd.DataFrame()

    factor_data_loader._read_factor_rows(
        RecordingConnection(),
        "SELECT date, stock_code, factor_name, factor_value",
        "factor_values",
        "factor_name = 'FactorA'",
        {"S3", "S1", "S2"},
        logging.getLogger("test_factor_loader_order"),
    )

    assert "stock_code IN ('S1', 'S2', 'S3')" in queries[0]


def test_precomputed_neutralized_input_preserves_raw_column_name():
    out = _load(
        _make_conn(),
        ["FactorA"],
        neutralization_config={"enabled": True},
    )

    assert list(out.columns) == ["FactorA"]
    assert out.loc[(pd.Timestamp("2024-01-02"), "S1"), "FactorA"] == 101.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "S2"), "FactorA"] == 102.0


def test_missing_neutralized_input_warns_and_falls_back_to_raw(caplog):
    test_logger = logging.getLogger("test_missing_neutralized_loader")
    caplog.set_level(logging.WARNING, logger=test_logger.name)

    out = _load(
        _make_conn(),
        ["FactorB"],
        neutralization_config={"enabled": True},
        logger=test_logger,
    )

    assert out.loc[(pd.Timestamp("2024-01-02"), "S1"), "FactorB"] == 10.0
    assert (
        "[中性化输入缺失] raw=FactorB expected=FactorB__neu fallback=raw" in caplog.text
    )


def test_skip_factors_always_load_raw_even_when_neutralized_exists():
    out = _load(
        _make_conn(),
        ["FactorA"],
        neutralization_config={"enabled": True, "skip_factors": ["FactorA"]},
    )

    assert out.loc[(pd.Timestamp("2024-01-02"), "S1"), "FactorA"] == 1.0
    assert out.loc[(pd.Timestamp("2024-01-02"), "S2"), "FactorA"] == 2.0


def test_precomputed_neutralized_mode_does_not_reuse_raw_parquet_cache(
    tmp_path: Path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    factor_hash = hashlib.md5("FactorA".encode()).hexdigest()[:8]
    stock_hash = hashlib.md5("ALL".encode()).hexdigest()[:8]
    stale_cache = (
        cache_dir / f"factors_{factor_hash}_2024-01-01_2024-01-02_{stock_hash}.parquet"
    )
    stale_cache.write_text("stale raw cache", encoding="utf-8")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("neutralized mode must not read the raw parquet cache")

    monkeypatch.setattr(pd, "read_parquet", _fail_if_called)

    out = _load(
        _make_conn(),
        ["FactorA"],
        config={
            "cache_settings": {"use_parquet_cache": True, "cache_dir": str(cache_dir)}
        },
        neutralization_config={"enabled": True},
    )

    assert out.loc[(pd.Timestamp("2024-01-02"), "S1"), "FactorA"] == 101.0


def test_raw_cache_is_invalidated_when_factor_values_change(tmp_path: Path):
    db_path = tmp_path / "factors.duckdb"
    cache_dir = tmp_path / "cache"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE factor_values ("
        "date VARCHAR, stock_code VARCHAR, factor_name VARCHAR, factor_value DOUBLE)"
    )
    conn.execute(
        "INSERT INTO factor_values VALUES "
        "('2024-01-01', 'S1', 'FactorA', 1.0), "
        "('2024-01-02', 'S1', 'FactorA', 2.0)"
    )
    config = {"cache_settings": {"use_parquet_cache": True, "cache_dir": str(cache_dir)}}

    first = _load(conn, ["FactorA"], config=config)
    assert first.loc[(pd.Timestamp("2024-01-02"), "S1"), "FactorA"] == 1.0
    conn.close()

    conn = duckdb.connect(str(db_path))
    conn.execute(
        "UPDATE factor_values SET factor_value = 9.0 "
        "WHERE date = '2024-01-01' AND factor_name = 'FactorA'"
    )
    conn.close()

    conn = duckdb.connect(str(db_path), read_only=True)
    second = _load(conn, ["FactorA"], config=config)
    conn.close()

    assert second.loc[(pd.Timestamp("2024-01-02"), "S1"), "FactorA"] == 9.0
    raw_caches = [
        path
        for path in cache_dir.glob("factors_*.parquet")
        if "_processed_calendar_v3_" not in path.name
    ]
    assert len(raw_caches) == 2


def test_raw_cache_survives_unrelated_result_write(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "factors.duckdb"
    cache_dir = tmp_path / "cache"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE factor_values ("
        "date VARCHAR, stock_code VARCHAR, factor_name VARCHAR, factor_value DOUBLE)"
    )
    conn.execute(
        "CREATE TABLE factor_analysis_results (run_id VARCHAR, metric DOUBLE)"
    )
    conn.execute(
        "INSERT INTO factor_values VALUES "
        "('2024-01-01', 'S1', 'FactorA', 1.0), "
        "('2024-01-02', 'S1', 'FactorA', 2.0)"
    )
    config = {"cache_settings": {"use_parquet_cache": True, "cache_dir": str(cache_dir)}}

    first = _load(conn, ["FactorA"], config=config)
    conn.execute("INSERT INTO factor_analysis_results VALUES ('r1', 1.0)")
    conn.close()

    monkeypatch.setattr(
        factor_data_loader,
        "process_raw_factor_df",
        lambda *_args, **_kwargs: pytest.fail("processed cache was not used"),
    )
    conn = duckdb.connect(str(db_path), read_only=True)
    second = _load(conn, ["FactorA"], config=config)
    conn.close()

    pd.testing.assert_frame_equal(first, second, check_exact=True)
    raw_caches = [
        path
        for path in cache_dir.glob("factors_*.parquet")
        if "_processed_calendar_v3_" not in path.name
    ]
    assert len(raw_caches) == 1


def test_processed_factor_cache_preserves_exact_panel(tmp_path: Path, monkeypatch):
    import os
    cache_dir = tmp_path / "cache"
    config = {
        "cache_settings": {"use_parquet_cache": True, "cache_dir": str(cache_dir)}
    }
    conn = _make_conn()

    first = _load(conn, ["FactorA", "FactorB"], config=config)
    processed = next(cache_dir.glob("*_processed_calendar_v3_*.parquet"))
    os.utime(processed, (1, 1))

    monkeypatch.setattr(
        factor_data_loader,
        "process_raw_factor_df",
        lambda *_args, **_kwargs: pytest.fail("processed cache was not used"),
    )
    second = _load(conn, ["FactorA", "FactorB"], config=config)

    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert processed.stat().st_atime > 1
    assert processed.stat().st_mtime == 1  # do not invalidate downstream source fingerprints
    monkeypatch.setattr(os, "utime", lambda *a, **k: (_ for _ in ()).throw(PermissionError("read only")))
    pd.testing.assert_frame_equal(first, _load(conn, ["FactorA", "FactorB"], config=config), check_exact=True)


def test_lasso_and_lightgbm_do_not_call_legacy_market_cap_neutralization():
    import inspect

    lasso = importlib.import_module("factor_synthesis.pipeline.lasso_method")
    lightgbm = importlib.import_module("factor_synthesis.pipeline.lightgbm_method")

    for fn in (
        lasso._run_lasso_with_qlib_adapter,
        lightgbm._run_lightgbm_with_qlib_adapter,
    ):
        source = inspect.getsource(fn)
        assert "_apply_market_cap_neutralization" not in source
        assert "_load_market_cap_data" not in source


def test_bound_neutralization_requires_complete_pool_scoped_versions():
    import pytest

    bindings = {'output_db':'output.duckdb','inputs':{},'outputs':{'neutralized_factor_values':{'role':'output','table':'main.neutralized_factor_values','version':'frozen'}}}
    config = {'data_bindings':bindings,'pool_id':'wind_micro','cache_settings':{'use_parquet_cache':False}}
    with _make_conn() as conn:
        out=_load(conn,['FactorA'],neutralization_config={'enabled':True},config=config)
        assert out.loc[(pd.Timestamp('2024-01-02'),'S1'),'FactorA']==101.0
        with pytest.raises(ValueError, match='unavailable'):
            _load(conn,['FactorB'],neutralization_config={'enabled':True},config=config)
        with pytest.raises(ValueError, match='unavailable'):
            _load(conn,['FactorA'],neutralization_config={'enabled':True},config={**config,'pool_id':'other_pool'})
        conn.execute("INSERT INTO neutralized_factor_values SELECT date,stock_code,factor_name,raw_factor_name,'other_pool',factor_value,run_id,created_at FROM neutralized_factor_values")
        with pytest.raises(ValueError, match='mixes pools'):
            _load(conn,['FactorA'],neutralization_config={'enabled':True},config={key:value for key,value in config.items() if key!='pool_id'})
        conn.execute('DROP TABLE neutralized_factor_values')
        with pytest.raises(ValueError, match='binding is missing'):
            _load(conn,['FactorA'],neutralization_config={'enabled':True},config=config)
