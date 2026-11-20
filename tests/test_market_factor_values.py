"""Tests for Stage 6.5b MarketFactorValueStore (Scheme B), FactorBase integration, and CLI."""
from __future__ import annotations

import datetime as dt
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.loaders.market_bar_importer import import_market_bars
from tools.agent_harness.cli import main as harness_main
from utils.bar_builder import build_bars
from utils.factor_base import FactorBase
from utils.market_factor_store import (
    LEAF_ARROW_SCHEMA,
    LEAF_COLUMNS,
    TABLE_COLUMNS,
    FactorValueConflictError,
    MarketFactorValueStore,
    build_identity_record,
    canonical_float_bits,
    compute_factor_identity_hash,
    save_bar_factor_values,
)
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule


def _make_sample_batch(
    *,
    factor_id: str = "alpha_test",
    frequency: str = "5min",
    session_view: str = "ETH",
    calendar_id: str = "XSHG",
    start_date: str = "2026-01-05",
    end_date: str = "2026-01-06",
    instruments: tuple[str, ...] = ("000001.XSHE", "600000.XSHG"),
    source_build_id: str = "build-001",
    code_version: str = "cv-test-1",
    params: dict | None = None,
    source_sha256: str | None = None,
    continuous_adjust: str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict, SessionCalendar]:
    cal = SessionCalendar(calendar_id, start_date, end_date, rth_reference_calendar=calendar_id)
    schedule = BarSchedule(cal, frequency, view=session_view, bucket_strategy="segment_anchored")
    rng = np.random.default_rng(seed)
    rows = []
    now_ts = pd.Timestamp(dt.datetime.now(dt.timezone.utc))
    batch_id = uuid.uuid4().hex
    fhash, payload = compute_factor_identity_hash(
        factor_id=factor_id,
        factor_class="tests.SyntheticFactor",
        source_sha256=source_sha256 or ("a" * 64),
        params=params or {"window": 5},
        frequency=frequency,
        session_view=session_view,
        analysis_mode="time_series",
        bucket_strategy="segment_anchored",
        calendar_hash=cal.calendar_hash,
        source_build_id=source_build_id,
        continuous_adjust=continuous_adjust,
        code_version=code_version,
    )
    id_rec = build_identity_record(
        factor_identity_hash=fhash,
        payload=payload,
        code_version_source="explicit",
        created_at=now_ts.isoformat(),
    )
    for day in cal.sessions():
        sid = day.strftime("%Y-%m-%d")
        slots = schedule.for_session(sid)
        for inst in instruments:
            vals = rng.normal(0.0, 1.0, size=len(slots))
            for ts, val in zip(slots["start_utc"], vals):
                rows.append(
                    {
                        "factor_id": factor_id,
                        "factor_identity_hash": fhash,
                        "frequency": frequency,
                        "session_view": session_view,
                        "instrument_id": inst,
                        "timestamp_utc": ts,
                        "session_id": sid,
                        "factor_value": float(val),
                        "source_build_id": source_build_id,
                        "calendar_hash": cal.calendar_hash,
                        "code_version": code_version,
                        "created_at": now_ts,
                        "batch_id": batch_id,
                    }
                )
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df, id_rec, cal


def _mp_lock_holder_worker(
    store_root: str,
    df_json_path: str,
    id_rec_json_path: str,
    lock_acquired_evt,
    release_lock_evt,
    second_df_json_path: str,
) -> None:
    store = MarketFactorValueStore(Path(store_root))
    df1 = pd.read_pickle(df_json_path)
    id_rec = json.loads(Path(id_rec_json_path).read_text(encoding="utf-8"))
    cal = SessionCalendar("XSHG", "2026-01-05", "2026-01-06", rth_reference_calendar="XSHG")
    store.append_factor_values(df1, identity_record=id_rec, calendar=cal)
    with store._write_lock():
        lock_acquired_evt.set()
        release_lock_evt.wait(timeout=10.0)
    df2 = pd.read_pickle(second_df_json_path)
    store.append_factor_values(df2, identity_record=id_rec, calendar=cal)


def test_store_requires_explicit_root_and_has_no_update_or_delete() -> None:
    with pytest.raises(ValueError, match="explicit root"):
        MarketFactorValueStore("")
    for forbidden in (
        "update_factor_values",
        "delete_factor_values",
        "overwrite_factor_values",
        "save_factor_values",
        "drop_factor_values",
        "replace_factor_values",
    ):
        assert not hasattr(MarketFactorValueStore, forbidden)
    source = (Path(__file__).resolve().parents[1] / "utils" / "market_factor_store.py").read_text(encoding="utf-8")
    assert "sqlite3" not in source


def test_identity_hash_sensitivity_and_canonical_stability() -> None:
    cal1 = SessionCalendar("XSHG", "2026-01-05", "2026-01-06", rth_reference_calendar="XSHG")
    cal2 = SessionCalendar("XSHG", "2026-01-05", "2026-01-07", rth_reference_calendar="XSHG")
    base = dict(
        factor_id="alpha_sens",
        factor_class="pkg.AlphaOne",
        source_sha256="1" * 64,
        params={"b": 2, "a": 1},
        frequency="5min",
        session_view="ETH",
        analysis_mode="time_series",
        bucket_strategy="segment_anchored",
        calendar_hash=cal1.calendar_hash,
        source_build_id="build-1",
        continuous_adjust="none",
        code_version="cv-1",
    )
    h0, _ = compute_factor_identity_hash(**base)
    # Key insertion order in params must not alter hash
    h_reordered, _ = compute_factor_identity_hash(**{**base, "params": {"a": 1, "b": 2}})
    assert h_reordered == h0

    mutations = [
        {"factor_class": "pkg.AlphaTwo"},
        {"source_sha256": "2" * 64},
        {"params": {"a": 1, "b": 3}},
        {"code_version": "cv-2"},
        {"source_build_id": "build-2"},
        {"session_view": "RTH"},
        {"bucket_strategy": "session_clock"},
        {"calendar_hash": cal2.calendar_hash},
        {"continuous_adjust": "ratio_back_adjust"},
        {"continuous_adjust": "diff_back_adjust"},
        {"frequency": "15min"},
        {"analysis_mode": "cross_sectional"},
    ]
    for patch in mutations:
        h_mut, _ = compute_factor_identity_hash(**{**base, **patch})
        assert h_mut != h0, f"Identity failed to change for {patch}"

    with pytest.raises(ValueError):
        compute_factor_identity_hash(**{**base, "params": {"bad": float("nan")}})


def test_validation_rejects_inf_naive_non_utc_invalid_freq_view_and_c_rules(tmp_path: Path) -> None:
    store = MarketFactorValueStore(tmp_path / "store")
    df, id_rec, cal = _make_sample_batch()

    # +-inf rejected
    for bad_val in (np.inf, -np.inf):
        bad = df.copy()
        bad.loc[0, "factor_value"] = bad_val
        with pytest.raises(ValueError, match="inf"):
            store.append_factor_values(bad, identity_record=id_rec, calendar=cal)

    # Naive timestamp rejected
    bad_naive = df.copy()
    bad_naive["timestamp_utc"] = bad_naive["timestamp_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        store.append_factor_values(bad_naive, identity_record=id_rec, calendar=cal)

    # Non-UTC timestamp rejected
    bad_sh = df.copy()
    bad_sh["timestamp_utc"] = bad_sh["timestamp_utc"].dt.tz_convert("Asia/Shanghai")
    with pytest.raises(ValueError, match="UTC"):
        store.append_factor_values(bad_sh, identity_record=id_rec, calendar=cal)

    # Invalid frequency & view rejected
    bad_freq = df.copy()
    bad_freq["frequency"] = "5M"
    with pytest.raises(ValueError, match="frequency"):
        store.append_factor_values(bad_freq, calendar=cal)

    bad_view = df.copy()
    bad_view["session_view"] = "ALL"
    with pytest.raises(ValueError, match="session_view"):
        store.append_factor_values(bad_view, calendar=cal)

    # Reviewer (c): single batch must have only one identity
    bad_multi_id = df.copy()
    bad_multi_id.loc[0, "factor_identity_hash"] = "f" * 64
    with pytest.raises(ValueError, match="Single batch must share exactly one factor_identity_hash"):
        store.append_factor_values(bad_multi_id, calendar=cal)

    # Reviewer (c): duplicate primary keys within batch rejected
    bad_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate primary keys"):
        store.append_factor_values(bad_dup, identity_record=id_rec, calendar=cal)

    # Reviewer (c): session_id must match SessionCalendar.session_at(timestamp_utc)
    bad_sess = df.copy()
    bad_sess.loc[0, "session_id"] = "2026-01-06" if bad_sess.loc[0, "session_id"] == "2026-01-05" else "2026-01-05"
    with pytest.raises(ValueError, match="session_id mismatch"):
        store.append_factor_values(bad_sess, identity_record=id_rec, calendar=cal)


def test_bitwise_idempotency_nan_signed_zero_and_conflict_rollback(tmp_path: Path) -> None:
    store = MarketFactorValueStore(tmp_path / "store")
    df, id_rec, cal = _make_sample_batch()
    df.loc[0, "factor_value"] = np.nan
    df.loc[1, "factor_value"] = -0.0
    df.loc[2, "factor_value"] = 1.25

    r1 = store.append_factor_values(df, identity_record=id_rec, calendar=cal)
    assert r1.inserted == len(df)
    assert r1.skipped == 0

    # Same values + NaN vs NaN + -0.0 vs +0.0 -> pure no-op
    replay = df.copy()
    replay.loc[1, "factor_value"] = +0.0
    r2 = store.append_factor_values(replay, identity_record=id_rec, calendar=cal)
    assert r2.inserted == 0
    assert r2.skipped == len(df)
    assert len(store.read_factor_values(factor_id="alpha_test")) == len(df)

    # 1-ULP difference rejected
    ulp_bad = df.copy()
    ulp_bad.loc[2, "factor_value"] = float(np.nextafter(1.25, np.inf))
    with pytest.raises(FactorValueConflictError) as exc_info:
        store.append_factor_values(ulp_bad, identity_record=id_rec, calendar=cal)
    assert exc_info.value.factor_id == "alpha_test"
    assert exc_info.value.conflict_count == 1
    assert len(exc_info.value.examples) == 1

    # NaN -> finite rejected
    nan_to_val = df.copy()
    nan_to_val.loc[0, "factor_value"] = 0.5
    with pytest.raises(FactorValueConflictError) as exc_nan:
        store.append_factor_values(nan_to_val, identity_record=id_rec, calendar=cal)
    assert exc_nan.value.conflict_count == 1

    # Finite -> NaN rejected
    val_to_nan = df.copy()
    val_to_nan.loc[2, "factor_value"] = np.nan
    with pytest.raises(FactorValueConflictError) as exc_fin:
        store.append_factor_values(val_to_nan, identity_record=id_rec, calendar=cal)
    assert exc_fin.value.conflict_count == 1

    # Partial overlap batch with 1 conflict and brand-new rows -> zero partial writes
    df_day3, _, cal3 = _make_sample_batch(
        start_date="2026-01-05", end_date="2026-01-07", source_build_id="build-1"
    )
    # Build overlapping partial batch sharing identity hash of df, changing row 2 and adding 2026-01-07
    new_day = df_day3.loc[df_day3["session_id"] == "2026-01-07"].copy()
    new_day["factor_identity_hash"] = df["factor_identity_hash"].iloc[0]
    new_day["calendar_hash"] = df["calendar_hash"].iloc[0]
    partial_bad = pd.concat([ulp_bad, new_day], ignore_index=True)
    with pytest.raises(FactorValueConflictError):
        store.append_factor_values(partial_bad, identity_record=id_rec, calendar=cal3)

    after = store.read_factor_values(factor_id="alpha_test")
    assert len(after) == len(df)
    assert np.array_equal(canonical_float_bits(after["factor_value"]), canonical_float_bits(
        df.sort_values(["factor_id", "factor_identity_hash", "instrument_id", "timestamp_utc"])["factor_value"]
    ))
    assert store.verify_store()["ok"] is True


def test_multiple_identities_coexist_under_same_factor_id(tmp_path: Path) -> None:
    store = MarketFactorValueStore(tmp_path / "store")
    df1, id1, cal1 = _make_sample_batch(params={"window": 5}, seed=1)
    df2, id2, cal2 = _make_sample_batch(params={"window": 10}, seed=2)
    assert id1["factor_identity_hash"] != id2["factor_identity_hash"]

    store.append_factor_values(df1, identity_record=id1, calendar=cal1)
    store.append_factor_values(df2, identity_record=id2, calendar=cal2)

    identities = store.list_identities(factor_id="alpha_test")
    assert len(identities) == 2
    all_rows = store.read_factor_values(factor_id="alpha_test")
    assert len(all_rows) == len(df1) + len(df2)
    sub1 = store.read_factor_values(factor_identity_hash=id1["factor_identity_hash"])
    assert len(sub1) == len(df1)


def test_reviewer_b_uncommitted_shards_and_staging_invisible_but_verified(tmp_path: Path) -> None:
    store = MarketFactorValueStore(tmp_path / "store")
    df, id_rec, cal = _make_sample_batch()
    store.append_factor_values(df, identity_record=id_rec, calendar=cal)

    # Plant an uncommitted Parquet shard directly inside the Hive partition tree
    uncommitted_dir = store.parquet_root / "factor_id=alpha_test" / "frequency=5min" / "year_month=2026-01"
    uncommitted_file = uncommitted_dir / "part-uncommitted-rogue.parquet"
    rogue_df = df.copy()
    rogue_df["factor_value"] = 9999.0
    rogue_df["instrument_id"] = "999999.XSHG"
    leaf = rogue_df.loc[:, list(LEAF_COLUMNS)].copy()
    leaf["timestamp_utc"] = pd.to_datetime(leaf["timestamp_utc"], utc=True).dt.as_unit("ns")
    leaf["created_at"] = pd.to_datetime(leaf["created_at"], utc=True).dt.as_unit("ns")
    pq.write_table(pa.Table.from_pandas(leaf, schema=LEAF_ARROW_SCHEMA, preserve_index=False), str(uncommitted_file))

    # Plant a .staging leftover
    staging_leftover = store.root / ".staging-crashed-abc123"
    staging_leftover.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(leaf, schema=LEAF_ARROW_SCHEMA, preserve_index=False),
        str(staging_leftover / "staged.parquet"),
    )

    # read_factor_values must NOT see rogue rows
    read_back = store.read_factor_values(factor_id="alpha_test")
    assert len(read_back) == len(df)
    assert "999999.XSHG" not in set(read_back["instrument_id"])

    # DuckDB view must NOT see rogue rows
    with duckdb.connect(":memory:") as con:
        store.create_view(con)
        view_df = con.execute("SELECT * FROM market_factor_values").fetchdf()
        assert len(view_df) == len(df)
        assert "999999.XSHG" not in set(view_df["instrument_id"])

    # verify_store must report both anomalies
    report = store.verify_store()
    assert report["ok"] is False
    assert any("part-uncommitted-rogue.parquet" in p for p in report["uncommitted_shards"])
    assert ".staging-crashed-abc123" in report["staging_leftovers"]


def test_reviewer_d_identity_hash_lookup_prunes_partitions_via_ledger(tmp_path: Path) -> None:
    store = MarketFactorValueStore(tmp_path / "store")
    df_a, id_a, cal_a = _make_sample_batch(factor_id="factor_a", frequency="5min", seed=10)
    df_b, id_b, cal_b = _make_sample_batch(factor_id="factor_b", frequency="15min", seed=20)
    store.append_factor_values(df_a, identity_record=id_a, calendar=cal_a)
    store.append_factor_values(df_b, identity_record=id_b, calendar=cal_b)

    target_hash = id_a["factor_identity_hash"]
    out = store.read_factor_values(factor_identity_hash=target_hash)
    assert len(out) == len(df_a)
    assert set(out["factor_id"]) == {"factor_a"}

    plan = store.last_read_plan
    assert plan is not None
    assert plan.resolved_factor_ids == ("factor_a",)
    assert plan.resolved_frequencies == ("5min",)
    assert len(plan.shard_paths) > 0
    assert all("factor_id=factor_a/frequency=5min/" in str(p) for p in plan.shard_paths)
    assert all("factor_b" not in str(p) and "15min" not in str(p) for p in plan.shard_paths)

    explain_text = store.explain_read(factor_identity_hash=target_hash)
    assert "READ_PARQUET" in explain_text
    assert "factor_id" in explain_text and "frequency" in explain_text


def _build_derived_factor_db(db_path: Path, *, with_build: bool) -> tuple[SessionCalendar, str]:
    cal = SessionCalendar("XSHG", "2026-01-05", "2026-01-07", rth_reference_calendar="XSHG")
    schedule = BarSchedule(cal, "5min", view="ETH", bucket_strategy="segment_anchored")
    frames = []
    for day in cal.sessions():
        sid = day.strftime("%Y-%m-%d")
        slots = schedule.for_session(sid)
        x = np.arange(len(slots), dtype=float) + 100.0
        frames.append(
            pd.DataFrame(
                {
                    "instrument_id": "000001.XSHE",
                    "timestamp_utc": slots["start_utc"],
                    "frequency": "5min",
                    "open": x,
                    "high": x + 1.0,
                    "low": x - 1.0,
                    "close": x + 0.5,
                    "volume": 100.0,
                    "money": 10000.0,
                    "factor": 1.0,
                    "source": "synthetic",
                    "data_version": "v1",
                }
            )
        )
    bars = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(
        [
            {
                "instrument_id": "000001.XSHE",
                "asset_class": "equity",
                "exchange": "XSHE",
                "timezone": "Asia/Shanghai",
                "contract_expiry": None,
                "price_adjustment": "none",
                "point_value": 1.0,
                "calendar_id": "XSHG",
                "rth_reference_calendar": "XSHG",
            }
        ]
    )
    with duckdb.connect(str(db_path)) as conn:
        import_market_bars(conn, bars, meta, bucket_strategy="segment_anchored", legacy_compatibility=False)
        if with_build:
            build_bars(
                conn,
                source="5min",
                targets=["15min"],
                instruments=["000001.XSHE"],
                start="2026-01-05",
                end="2026-01-07",
                bucket_strategy="segment_anchored",
            )
    return cal, "000001.XSHE"


class _MinuteMeanFactor(FactorBase):
    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(data, "close", 3)
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)

    def _get_cache_inputs(self) -> dict:
        return {"window": 3}


def test_reviewer_a_factor_base_integration_and_granularity_rules(tmp_path: Path) -> None:
    db_derived = tmp_path / "derived.duckdb"
    cal, inst = _build_derived_factor_db(db_derived, with_build=True)
    store = MarketFactorValueStore(tmp_path / "store")

    # Derived 15min bars: FactorBase._fetch_market_bars drops attrs['build_ids'],
    # so save_bar_factor_values must actively query bar_build_log.
    f_derived = _MinuteMeanFactor(
        None,
        "minute_mean",
        "2026-01-05",
        "2026-01-06",
        stock_pool=[inst],
        duckdb_path=str(db_derived),
        data_frequency="15min",
        analysis_mode="time_series",
        session_view="ETH",
        session_calendar=cal,
    )
    f_derived.get_bar_factor_frame()
    assert "build_ids" not in f_derived.market_bars.attrs

    receipt = save_bar_factor_values(f_derived, store, identity_scope="incremental")
    assert receipt.inserted > 0

    # Incremental append of next session with fixed calendar keeps SAME factor_identity_hash
    f_next = _MinuteMeanFactor(
        None,
        "minute_mean",
        "2026-01-07",
        "2026-01-07",
        stock_pool=[inst],
        duckdb_path=str(db_derived),
        data_frequency="15min",
        analysis_mode="time_series",
        session_view="ETH",
        session_calendar=cal,
    )
    r_next = save_bar_factor_values(f_next, store, identity_scope="incremental")
    assert r_next.factor_identity_hash == receipt.factor_identity_hash
    assert r_next.inserted > 0

    # Expanding calendar range changes calendar_hash -> new factor_identity_hash
    cal_expanded = SessionCalendar("XSHG", "2026-01-05", "2026-01-08", rth_reference_calendar="XSHG")
    f_exp = _MinuteMeanFactor(
        None,
        "minute_mean",
        "2026-01-05",
        "2026-01-06",
        stock_pool=[inst],
        duckdb_path=str(db_derived),
        data_frequency="15min",
        analysis_mode="time_series",
        session_view="ETH",
        session_calendar=cal_expanded,
    )
    r_exp = save_bar_factor_values(f_exp, store, identity_scope="incremental")
    assert r_exp.factor_identity_hash != receipt.factor_identity_hash

    # Upstream build-bars rebuild changes source_build_id:
    # - incremental scope rejects rewriting existing session under old identity
    # - exact scope generates a new version identity
    with duckdb.connect(str(db_derived)) as conn:
        conn.execute("UPDATE market_bars_5min SET close = close + 0.01 WHERE timestamp_utc = (SELECT min(timestamp_utc) FROM market_bars_5min)")
        build_bars(
            conn,
            source="5min",
            targets=["15min"],
            instruments=[inst],
            start="2026-01-05",
            end="2026-01-05",
            bucket_strategy="segment_anchored",
        )
    f_rebuilt = _MinuteMeanFactor(
        None,
        "minute_mean",
        "2026-01-05",
        "2026-01-05",
        stock_pool=[inst],
        duckdb_path=str(db_derived),
        data_frequency="15min",
        analysis_mode="time_series",
        session_view="ETH",
        session_calendar=cal,
    )
    with pytest.raises(ValueError, match="source_build_id changed"):
        save_bar_factor_values(f_rebuilt, store, identity_scope="incremental")
    r_exact = save_bar_factor_values(f_rebuilt, store, identity_scope="exact")
    assert r_exact.factor_identity_hash != receipt.factor_identity_hash

    # Native 5min DB without bar_build_log and without explicit source_build_ids must fail clearly
    db_native = tmp_path / "native.duckdb"
    _build_derived_factor_db(db_native, with_build=False)
    f_native = _MinuteMeanFactor(
        None,
        "minute_mean_native",
        "2026-01-05",
        "2026-01-05",
        stock_pool=[inst],
        duckdb_path=str(db_native),
        data_frequency="5min",
        analysis_mode="time_series",
        session_view="ETH",
        data_version="v1",
        session_calendar=cal,
    )
    with pytest.raises(ValueError, match="source_build_id"):
        save_bar_factor_values(f_native, store)

    # Explicit source_build_ids works
    r_nat = save_bar_factor_values(f_native, store, source_build_ids="native-import-v1")
    assert r_nat.inserted > 0


def test_persisted_bar_identity_tracks_extracted_rolling_source(tmp_path: Path) -> None:
    database = tmp_path / "native.duckdb"
    _build_derived_factor_db(database, with_build=False)
    root = Path(__file__).resolve().parents[1]
    export = tmp_path / "export"
    utils = export / "utils"
    shutil.copytree(root / "utils", utils, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "config", export / "config")

    command = """
import json
from pathlib import Path
import sys

from test_market_factor_values import _MinuteMeanFactor
from utils.factor_base import FactorBase
from utils.market_factor_store import MarketFactorValueStore, save_bar_factor_values
from utils.session_calendar import SessionCalendar

assert Path(sys.modules[FactorBase.__module__].__file__).parent == Path.cwd() / "utils"
factor = _MinuteMeanFactor(
    None, "minute_mean", "2026-01-05", "2026-01-05", stock_pool=["000001.XSHE"],
    duckdb_path=sys.argv[1], data_frequency="5min", analysis_mode="time_series",
    session_calendar=SessionCalendar("XSHG", "2026-01-05", "2026-01-07", rth_reference_calendar="XSHG"),
)
store = MarketFactorValueStore(Path(sys.argv[2]))
receipt = save_bar_factor_values(
    factor, store, source_build_ids="native-import-v1", identity_scope="incremental",
)
written = store.read_factor_values()
assert written["code_version"].unique().tolist() == ["review-export"]
print(json.dumps([receipt.factor_identity_hash, written["factor_value"].dropna().iloc[0]]))
"""
    env = {
        **os.environ,
        "QUANTATION_CODE_VERSION": "review-export",
        "PYTHONPATH": os.pathsep.join(map(str, (export, root, root / "tests"))),
    }

    def save_in_export(name: str) -> tuple[str, float]:
        result = subprocess.run(
            [sys.executable, "-B", "-c", command, str(database), str(tmp_path / name)],
            cwd=export, env=env, check=True, capture_output=True, text=True, timeout=90,
        )
        identity, value = json.loads(result.stdout)
        return identity, value

    original_hash, original_value = save_in_export("original")
    bridge = utils / "factor_core/bar_bridge.py"
    original_source = bridge.read_text()
    call = "return session_rolling(data, column, window, **kwargs)"
    assert original_source.count(call) == 1
    bridge.write_text(original_source.replace(call, f"{call} * 2"))
    edited_hash, edited_value = save_in_export("edited")

    assert original_value == pytest.approx(101.5)
    assert edited_value == pytest.approx(original_value * 2)
    assert original_hash != edited_hash


def test_multiprocess_concurrent_write_lock_and_immediate_read(tmp_path: Path) -> None:
    store_root = tmp_path / "mp_store"
    df1, id1, _ = _make_sample_batch(start_date="2026-01-05", end_date="2026-01-05", seed=101)
    df2, _, _ = _make_sample_batch(start_date="2026-01-06", end_date="2026-01-06", seed=102)
    df2["factor_identity_hash"] = df1["factor_identity_hash"].iloc[0]

    p1 = tmp_path / "df1.pkl"
    p2 = tmp_path / "df2.pkl"
    id_p = tmp_path / "id1.json"
    df1.to_pickle(p1)
    df2.to_pickle(p2)
    id_p.write_text(json.dumps(id1), encoding="utf-8")

    ctx = multiprocessing.get_context("spawn")
    lock_acquired = ctx.Event()
    release_lock = ctx.Event()
    proc = ctx.Process(
        target=_mp_lock_holder_worker,
        args=(str(store_root), str(p1), str(id_p), lock_acquired, release_lock, str(p2)),
    )
    proc.start()
    try:
        assert lock_acquired.wait(timeout=15.0), "Writer did not acquire lock in time"
        store = MarketFactorValueStore(store_root)
        t0 = time.monotonic()
        immediate = store.read_factor_values(factor_id="alpha_test")
        elapsed = time.monotonic() - t0
        # Must succeed immediately without waiting for writer lock release
        assert len(immediate) == len(df1)
        assert elapsed < 5.0
    finally:
        release_lock.set()
        proc.join(timeout=15.0)
    assert proc.exitcode == 0

    final = MarketFactorValueStore(store_root).read_factor_values(factor_id="alpha_test")
    assert len(final) == len(df1) + len(df2)


def test_cli_end_to_end_write_read_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store_dir = tmp_path / "cli_store"
    out_root = tmp_path / "harness_out"
    csv_path = tmp_path / "factors.csv"

    cal = SessionCalendar("XSHG", "2026-01-05", "2026-01-05", rth_reference_calendar="XSHG")
    slots = BarSchedule(cal, "15min", view="ETH", bucket_strategy="segment_anchored").for_session("2026-01-05")
    factor_df = pd.DataFrame(
        {
            "timestamp_utc": slots["start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "instrument_id": "000001.XSHE",
            "factor_value": np.linspace(0.1, 1.6, len(slots)),
        }
    )
    factor_df.to_csv(csv_path, index=False)

    # dry-run does not create store
    dry_json = tmp_path / "dry.json"
    rc = harness_main(
        [
            "write-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--input",
            str(csv_path),
            "--dry-run",
            "--output-root",
            str(out_root),
            "--json-out",
            str(dry_json),
        ]
    )
    capsys.readouterr()
    assert rc == 0
    assert not store_dir.exists()
    assert json.loads(dry_json.read_text(encoding="utf-8"))["summary"]["reason"] == "dry_run"

    # Real write
    write_json = tmp_path / "write.json"
    rc = harness_main(
        [
            "write-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--input",
            str(csv_path),
            "--factor-id",
            "cli_alpha",
            "--frequency",
            "15min",
            "--session-view",
            "ETH",
            "--analysis-mode",
            "time_series",
            "--bucket-strategy",
            "segment_anchored",
            "--calendar",
            "XSHG",
            "--calendar-start",
            "2026-01-05",
            "--calendar-end",
            "2026-01-05",
            "--source-build-id",
            "cli-build-1",
            "--output-root",
            str(out_root),
            "--json-out",
            str(write_json),
        ]
    )
    capsys.readouterr()
    assert rc == 0
    w_payload = json.loads(write_json.read_text(encoding="utf-8"))
    assert w_payload["metrics"]["inserted"] == len(slots)
    fhash = w_payload["metrics"]["factor_identity_hash"]

    # Idempotent rerun
    rc = harness_main(
        [
            "write-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--input",
            str(csv_path),
            "--factor-id",
            "cli_alpha",
            "--frequency",
            "15min",
            "--session-view",
            "ETH",
            "--analysis-mode",
            "time_series",
            "--bucket-strategy",
            "segment_anchored",
            "--calendar",
            "XSHG",
            "--calendar-start",
            "2026-01-05",
            "--calendar-end",
            "2026-01-05",
            "--source-build-id",
            "cli-build-1",
            "--output-root",
            str(out_root),
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["metrics"]["inserted"] == 0
    assert out["metrics"]["skipped"] == len(slots)

    # Read & export via CLI
    export_csv = tmp_path / "exported.csv"
    rc = harness_main(
        [
            "read-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--factor-identity-hash",
            fhash,
            "--list-identities",
            "--out",
            str(export_csv),
            "--output-root",
            str(out_root),
        ]
    )
    read_out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert read_out["metrics"]["row_count"] == len(slots)
    assert read_out["metrics"]["identity_count"] == 1
    assert export_csv.is_file()

    # Verify clean store -> exit 0
    rc = harness_main(
        [
            "verify-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--output-root",
            str(out_root),
        ]
    )
    capsys.readouterr()
    assert rc == 0

    # Bad input (+inf) -> exit 2 (EXIT_PRECONDITION)
    bad_csv = tmp_path / "bad_inf.csv"
    bad_df = factor_df.copy()
    bad_df.loc[0, "factor_value"] = np.inf
    bad_df.to_csv(bad_csv, index=False)
    rc = harness_main(
        [
            "write-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--input",
            str(bad_csv),
            "--factor-id",
            "cli_alpha",
            "--frequency",
            "15min",
            "--analysis-mode",
            "time_series",
            "--calendar",
            "XSHG",
            "--calendar-start",
            "2026-01-05",
            "--calendar-end",
            "2026-01-05",
            "--source-build-id",
            "cli-build-1",
            "--output-root",
            str(out_root),
        ]
    )
    capsys.readouterr()
    assert rc == 2

    # Conflict on full schema CSV -> exit 4 (EXIT_QUALITY_GATE)
    stored_df = pd.read_csv(export_csv)
    stored_df.loc[0, "factor_value"] = 999.999
    conflict_csv = tmp_path / "conflict_full.csv"
    stored_df.to_csv(conflict_csv, index=False)

    # Full schema CSV without --calendar must fail with exit 2 (EXIT_PRECONDITION)
    rc_no_cal = harness_main(
        [
            "write-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--input",
            str(conflict_csv),
            "--output-root",
            str(out_root),
        ]
    )
    capsys.readouterr()
    assert rc_no_cal == 2

    rc = harness_main(
        [
            "write-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--input",
            str(conflict_csv),
            "--calendar",
            "XSHG",
            "--calendar-start",
            "2026-01-05",
            "--calendar-end",
            "2026-01-05",
            "--output-root",
            str(out_root),
        ]
    )
    conf_out = json.loads(capsys.readouterr().out)
    assert rc == 4
    assert conf_out["error"]["code"] == "factor_value_conflict"
    assert conf_out["metrics"]["conflict_count"] == 1

    # Corrupt input Parquet must map PyArrow execution error to exit 3 (EXIT_EXECUTION)
    bad_pq = tmp_path / "corrupt_input.parquet"
    bad_pq.write_bytes(b"NOT_A_PARQUET_FILE")
    rc_pq = harness_main(
        [
            "write-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--input",
            str(bad_pq),
            "--calendar",
            "XSHG",
            "--calendar-start",
            "2026-01-05",
            "--calendar-end",
            "2026-01-05",
            "--output-root",
            str(out_root),
        ]
    )
    pq_out = json.loads(capsys.readouterr().out)
    assert rc_pq == 3
    assert pq_out["error"]["code"] == "factor_store_execution_failed"

    # Verify CLI returns exit 4 when uncommitted shard is planted
    uncommitted = store_dir / "parquet_data" / "factor_id=cli_alpha" / "frequency=15min" / "year_month=2026-01" / "rogue.parquet"
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "factor_identity_hash": [fhash],
                    "session_view": ["ETH"],
                    "instrument_id": ["ROGUE.XSHG"],
                    "timestamp_utc": pd.to_datetime(["2026-01-05T01:30:00Z"]).as_unit("ns"),
                    "session_id": ["2026-01-05"],
                    "factor_value": [123.0],
                    "source_build_id": ["rogue"],
                    "calendar_hash": [cal.calendar_hash],
                    "code_version": ["cv"],
                    "created_at": pd.to_datetime(["2026-01-05T02:00:00Z"]).as_unit("ns"),
                    "batch_id": ["rogue-batch"],
                }
            ),
            schema=LEAF_ARROW_SCHEMA,
            preserve_index=False,
        ),
        str(uncommitted),
    )
    rc = harness_main(
        [
            "verify-factor-values",
            "--factor-values-store",
            str(store_dir),
            "--output-root",
            str(out_root),
        ]
    )
    verify_out = json.loads(capsys.readouterr().out)
    assert rc == 4
    assert verify_out["metrics"]["ok"] is False
    assert verify_out["metrics"]["uncommitted_shards"]


def test_read_filters_empty_store_and_bad_factor_id_rejection(tmp_path: Path) -> None:
    empty_store = MarketFactorValueStore(tmp_path / "empty_store")
    empty_df = empty_store.read_factor_values(factor_id="nonexistent")
    assert empty_df.empty
    assert list(empty_df.columns) == list(TABLE_COLUMNS)
    assert str(empty_df["timestamp_utc"].dtype) == "datetime64[ns, UTC]"

    df, id_rec, cal = _make_sample_batch(start_date="2026-01-05", end_date="2026-01-06")
    store = MarketFactorValueStore(tmp_path / "filter_store")
    store.append_factor_values(df, identity_record=id_rec, calendar=cal)

    # Filter by instrument + session range
    sub = store.read_factor_values(
        factor_id="alpha_test",
        instruments=["000001.XSHE"],
        session_start="2026-01-06",
        session_end="2026-01-06",
    )
    assert not sub.empty
    assert set(sub["instrument_id"]) == {"000001.XSHE"}
    assert set(sub["session_id"]) == {"2026-01-06"}

    # Filter by UTC time bounds
    t0 = sub["timestamp_utc"].min()
    t1 = sub["timestamp_utc"].iloc[2]
    sub_utc = store.read_factor_values(
        factor_id="alpha_test",
        instruments=["000001.XSHE"],
        start_utc=t0,
        end_utc=t1,
    )
    assert len(sub_utc) == 3

    # Reject unsafe factor_id
    bad_id = df.copy()
    bad_id["factor_id"] = "../escape"
    with pytest.raises(ValueError, match="factor_id"):
        store.append_factor_values(bad_id, calendar=cal)
    bad_eq = df.copy()
    bad_eq["factor_id"] = "a=b"
    with pytest.raises(ValueError, match="factor_id"):
        store.append_factor_values(bad_eq, calendar=cal)


def test_missing_calendar_or_session_id_mismatch_rejected_and_pk_conflict_across_sessions(
    tmp_path: Path,
) -> None:
    store = MarketFactorValueStore(tmp_path / "pk_store")
    df, id_rec, cal = _make_sample_batch(start_date="2026-01-05", end_date="2026-01-05")
    first = df.iloc[[0]].copy()
    first["factor_value"] = 1.25
    store.append_factor_values(first, identity_record=id_rec, calendar=cal)
    assert len(store.read_factor_values(factor_id="alpha_test")) == 1

    # Omitting calendar must be rejected and leave committed row count unchanged
    replay_diff = first.copy()
    replay_diff["session_id"] = "2026-02-02"
    replay_diff["factor_value"] = 9.5
    with pytest.raises(ValueError, match="calendar"):
        store.append_factor_values(replay_diff)
    assert len(store.read_factor_values(factor_id="alpha_test")) == 1

    # Passing calendar with forged session_id must be rejected and leave committed row count unchanged
    cal_feb = SessionCalendar("XSHG", "2026-01-05", "2026-02-05", rth_reference_calendar="XSHG")
    with pytest.raises(ValueError, match="session_id mismatch"):
        store.append_factor_values(replay_diff, calendar=cal_feb)
    assert len(store.read_factor_values(factor_id="alpha_test")) == 1

    # Even if a shard's recorded session set does not intersect incoming session_id,
    # primary key conflict detection still catches (factor_identity_hash, instrument_id, timestamp_utc)
    # via timestamp range overlap without session_id filtering.
    lines = store.batches_path.read_text(encoding="utf-8").splitlines()
    batch_obj = json.loads(lines[0])
    batch_obj["shards"][0]["sessions"] = {"2026-02-02": 1}
    store.batches_path.write_text(json.dumps(batch_obj) + "\n", encoding="utf-8")

    conflict_same_ts = first.copy()
    conflict_same_ts["factor_value"] = 9.5
    with pytest.raises(FactorValueConflictError) as exc_info:
        store.append_factor_values(conflict_same_ts, calendar=cal)
    assert exc_info.value.conflict_count == 1
    assert len(store.read_factor_values(factor_id="alpha_test")) == 1


def test_unregistered_identity_and_tampered_identity_record_rejected(tmp_path: Path) -> None:
    store = MarketFactorValueStore(tmp_path / "id_store")
    df, id_rec, cal = _make_sample_batch(start_date="2026-01-05", end_date="2026-01-05")

    # Unregistered hash without identity_record must be rejected with zero placeholder written
    with pytest.raises(ValueError, match="identity_record is required"):
        store.append_factor_values(df, calendar=cal)
    assert not store.identities_path.exists()
    assert store.read_factor_values(factor_id="alpha_test").empty

    # Empty "{}" payload in identity_record must be rejected
    bad_empty = dict(id_rec)
    bad_empty["identity_payload_json"] = "{}"
    with pytest.raises(ValueError, match="missing required keys"):
        store.append_factor_values(df, identity_record=bad_empty, calendar=cal)

    # Tampered payload that does not hash to factor_identity_hash must be rejected
    tampered = dict(id_rec)
    payload_obj = json.loads(id_rec["identity_payload_json"])
    payload_obj["params"] = {"window": 999}
    tampered["identity_payload_json"] = json.dumps(payload_obj, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="SHA256 verification failed"):
        store.append_factor_values(df, identity_record=tampered, calendar=cal)
    assert not store.identities_path.exists()

    # Valid registration succeeds and stores full canonical JSON payload
    r1 = store.append_factor_values(df.iloc[[0]], identity_record=id_rec, calendar=cal)
    assert r1.inserted == 1
    registered = store.list_identities(factor_id="alpha_test")
    assert len(registered) == 1
    assert registered["identity_payload_json"].iloc[0] == id_rec["identity_payload_json"]
    assert registered["analysis_mode"].iloc[0] == "time_series"

    # Subsequent append for already-registered identity can omit identity_record
    r2 = store.append_factor_values(df.iloc[[1]], calendar=cal)
    assert r2.inserted == 1
    assert len(store.read_factor_values(factor_id="alpha_test")) == 2


def test_trailing_unterminated_ledger_line_allows_reads_and_subsequent_appends(tmp_path: Path) -> None:
    store = MarketFactorValueStore(tmp_path / "tail_store")
    df, id_rec, cal = _make_sample_batch(start_date="2026-01-05", end_date="2026-01-06")
    df_day1 = df.loc[df["session_id"] == "2026-01-05"].copy()
    df_day2 = df.loc[df["session_id"] == "2026-01-06"].copy()

    store.append_factor_values(df_day1, identity_record=id_rec, calendar=cal)
    assert len(store.read_factor_values(factor_id="alpha_test")) == len(df_day1)

    # Append an unterminated half-line (no trailing '\n') to batches.jsonl
    with store.batches_path.open("ab") as fh:
        fh.write(b'{"record_type":"batch","status":"comm')

    # Reader still returns all previously committed rows; verify_store reports the tail
    read_with_tail = store.read_factor_values(factor_id="alpha_test")
    assert len(read_with_tail) == len(df_day1)
    report = store.verify_store()
    assert report["ok"] is False
    assert len(report["corrupt_ledger_lines"]) == 1
    assert report["corrupt_ledger_lines"][0]["reason"] == "unterminated_line"

    # Subsequent append can proceed cleanly and heals the uncommitted tail
    r2 = store.append_factor_values(df_day2, identity_record=id_rec, calendar=cal)
    assert r2.inserted == len(df_day2)
    read_after = store.read_factor_values(factor_id="alpha_test")
    assert len(read_after) == len(df_day1) + len(df_day2)
    assert store.verify_store()["ok"] is True

    # Mid-file corrupt line (terminated by '\n') still raises ValueError
    with store.batches_path.open("ab") as fh:
        fh.write(b'{"broken":\n')
    with pytest.raises(ValueError, match="Corrupt ledger entries detected"):
        store.read_factor_values(factor_id="alpha_test")


def test_multimonth_partial_failure_cleans_up_replaced_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MarketFactorValueStore(tmp_path / "cleanup_store")
    cal = SessionCalendar("XSHG", "2026-01-30", "2026-02-02", rth_reference_calendar="XSHG")
    df_jan, id_rec, _ = _make_sample_batch(start_date="2026-01-30", end_date="2026-01-30", seed=7)
    df_feb, _, _ = _make_sample_batch(start_date="2026-02-02", end_date="2026-02-02", seed=8)
    fhash, payload = compute_factor_identity_hash(
        factor_id="alpha_test",
        factor_class="tests.SyntheticFactor",
        source_sha256="a" * 64,
        params={"window": 5},
        frequency="5min",
        session_view="ETH",
        analysis_mode="time_series",
        bucket_strategy="segment_anchored",
        calendar_hash=cal.calendar_hash,
        source_build_id="build-001",
        code_version="cv-test-1",
    )
    id_rec = build_identity_record(factor_identity_hash=fhash, payload=payload, code_version_source="explicit")
    multi_df = pd.concat([df_jan, df_feb], ignore_index=True)
    multi_df["factor_identity_hash"] = fhash
    multi_df["calendar_hash"] = cal.calendar_hash

    orig_replace = __import__("os").replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated disk error on second month shard")
        return orig_replace(src, dst)

    monkeypatch.setattr("utils.market_factor_store.os.replace", flaky_replace)
    with pytest.raises(OSError, match="simulated disk error"):
        store.append_factor_values(multi_df, identity_record=id_rec, calendar=cal)

    assert list(store.parquet_root.rglob("*.parquet")) == []
    assert store.verify_store()["ok"] is True
