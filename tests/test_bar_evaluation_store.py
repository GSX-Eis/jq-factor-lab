"""Stage 6.6 tests: Bar evaluation DuckDB persistence, CLI, monitor-refresh, and Streamlit viewer."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from utils import bar_evaluation_store as store
from utils.bar_evaluation_store import (
    BAR_EVALUATION_TABLES,
    BarEvaluationConflictError,
    BarEvaluationIntegrityError,
    BarEvaluationPreconditionError,
    get_bar_evaluation_filter_options,
    ingest_evaluation,
    query_bar_evaluation_detail,
    query_bar_evaluation_runs,
    query_bar_evaluations,
)

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(tmp_path: Path, *args: str, command: str = "time-series-analysis") -> tuple[int, dict]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.agent_harness.cli",
            command,
            "--output-root",
            str(tmp_path / "harness"),
            *args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)
    assert proc.returncode == payload["exit_code"], proc.stderr
    return proc.returncode, payload


def _make_ts_bars(tmp_path: Path) -> Path:
    times = pd.date_range("2026-12-01T20:56Z", periods=12, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp_utc": times,
            "instrument_id": "000001",
            "frequency": "1min",
            "open": 100.0 + np.arange(12),
            "high": 102.0 + np.arange(12),
            "low": 99.0 + np.arange(12),
            "close": 101.0 + np.arange(12),
            "volume": 1.0,
            "source": "SYNTHETIC",
            "data_version": "NERE749_v1",
            "factor_value": np.arange(12) // 2,
        }
    )
    path = tmp_path / "ts_bars.csv"
    frame.to_csv(path, index=False)
    return path


def _make_cs_bars_and_factors(tmp_path: Path) -> tuple[Path, Path]:
    records = []
    for i in range(6):
        for j, timestamp in enumerate(pd.date_range("2026-12-01T14:30Z", periods=8, freq="5min")):
            if i == 1 and j == 5:
                continue
            close = 100.0 + (6 - i if j == 2 else i + 1)
            if i == 0 and j == 5:
                close = 0.0
            factor = 1.0 if j == 2 else float(i)
            if j == 3 and i >= 2:
                factor = np.nan
            records.append(
                {
                    "timestamp_utc": timestamp,
                    "instrument_id": f"A{i}",
                    "frequency": "5min",
                    "open": 100.0,
                    "close": close,
                    "high": max(100.0, close) + 1.0,
                    "low": min(100.0, close) - 1.0,
                    "volume": 1.0,
                    "source": "SYNTHETIC",
                    "data_version": "NERE749_v1",
                    "factor_value": factor,
                }
            )
    frame = pd.DataFrame(records)
    bars_path = tmp_path / "cs_bars.csv"
    factors_path = tmp_path / "cs_factors.csv"
    frame[["timestamp_utc", "instrument_id", "factor_value"]].to_csv(factors_path, index=False)
    frame.drop(columns="factor_value").to_csv(bars_path, index=False)
    return bars_path, factors_path


def _assert_value_matches(db_val, csv_val) -> None:
    db_missing = db_val is None or (isinstance(db_val, float) and math.isnan(db_val))
    csv_missing = csv_val is None or pd.isna(csv_val)
    if db_missing or csv_missing:
        assert db_missing and csv_missing, f"Missingness mismatch: db={db_val!r}, csv={csv_val!r}"
        return
    if isinstance(db_val, bool) or isinstance(csv_val, (bool, np.bool_)):
        assert bool(db_val) is bool(csv_val)
        return
    if isinstance(db_val, (int, np.integer)) and not isinstance(db_val, bool):
        assert int(db_val) == int(csv_val)
        return
    if isinstance(db_val, (float, np.floating)):
        assert float(db_val) == pytest.approx(float(csv_val), rel=1e-12, abs=1e-12)
        return
    if isinstance(db_val, (dict, list)) or (
        isinstance(csv_val, str) and csv_val.startswith("{")
    ):
        db_obj = json.loads(db_val) if isinstance(db_val, str) else db_val
        csv_obj = json.loads(csv_val) if isinstance(csv_val, str) else csv_val
        assert db_obj == csv_obj
        return
    assert str(db_val) == str(csv_val)


def _assert_df_matches_records(df: pd.DataFrame, db_rows: list[dict]) -> None:
    assert len(df) == len(db_rows)
    for row_idx, (_, csv_row) in enumerate(df.iterrows()):
        db_row = db_rows[row_idx]
        for col in df.columns:
            _assert_value_matches(db_row[col], csv_row[col])


def _republished_artifacts(payload: dict) -> list[dict]:
    # Simulate a worker publishing modified content under the same evaluation identity.
    return [
        {**a, "sha256": store.sha256_file(a["path"])}
        for a in payload["artifacts"] if "sha256" in a
    ]


def test_end_to_end_both_modes_and_csv_parity(tmp_path: Path) -> None:
    results_db = tmp_path / "results" / "bar_eval.duckdb"

    # 1) time_series mode
    ts_bars = _make_ts_bars(tmp_path)
    ts_cfg = {
        "bars_csv": str(ts_bars),
        "analysis_mode": "time_series",
        "calendar": "NQ",
        "start_date": "2026-12-01",
        "end_date": "2026-12-01",
        "horizons": [1, 2, 5],
        "n_quantiles": 3,
    }
    ts_cfg_path = tmp_path / "ts_config.json"
    ts_cfg_path.write_text(json.dumps(ts_cfg), encoding="utf-8")

    rc, ts_payload = _run_cli(
        tmp_path,
        "--config",
        str(ts_cfg_path),
        "--results-db",
        str(results_db),
        "--run-id",
        "ts_run_1",
    )
    assert rc == 0 and ts_payload["ok"] is True
    ts_id_hash = ts_payload["metrics"]["result_identity"]["identity_hash"]
    assert ts_payload["metrics"]["results_db"] == {
        "path": str(results_db.resolve()),
        "identity_hash": ts_id_hash,
        "status": "inserted",
    }

    ts_artifacts = {a["artifact_type"]: a for a in ts_payload["artifacts"]}
    ts_ic_csv = pd.read_csv(
        ts_artifacts["ic_by_horizon"]["path"],
        dtype={"instrument_id": "string", "frequency": "string", "scope": "string"},
    )
    ts_q_csv = pd.read_csv(
        ts_artifacts["quantile_stats"]["path"],
        dtype={"instrument_id": "string", "frequency": "string"},
    )
    ts_summary = json.loads(Path(ts_artifacts["evaluation_summary"]["path"]).read_text(encoding="utf-8"))

    # 2) cross_sectional mode via single-factor-analysis router + config results_db overridden by CLI
    cs_bars, cs_factors = _make_cs_bars_and_factors(tmp_path)
    unused_db = tmp_path / "should_not_be_created.duckdb"
    cs_cfg = {
        "bars_csv": str(cs_bars),
        "factors_csv": str(cs_factors),
        "analysis_mode": "cross_sectional",
        "calendar": "NQ",
        "start_date": "2026-12-01",
        "end_date": "2026-12-01",
        "horizons": [1],
        "n_quantiles": 2,
        "results_db": str(unused_db),
    }
    cs_cfg_path = tmp_path / "cs_config.json"
    cs_cfg_path.write_text(json.dumps(cs_cfg), encoding="utf-8")

    rc2, cs_payload = _run_cli(
        tmp_path,
        "--config",
        str(cs_cfg_path),
        "--results-db",
        str(results_db),
        "--run-id",
        "cs_run_1",
        command="single-factor-analysis",
    )
    assert rc2 == 0 and cs_payload["ok"] is True
    assert not unused_db.exists()
    cs_id_hash = cs_payload["metrics"]["result_identity"]["identity_hash"]
    assert cs_payload["metrics"]["results_db"]["status"] == "inserted"

    cs_artifacts = {a["artifact_type"]: a for a in cs_payload["artifacts"]}
    cs_ic_csv = pd.read_csv(cs_artifacts["ic_by_horizon"]["path"], dtype={"frequency": "string"})
    cs_q_csv = pd.read_csv(cs_artifacts["quantile_stats"]["path"], dtype={"frequency": "string"})
    cs_summary = json.loads(Path(cs_artifacts["evaluation_summary"]["path"]).read_text(encoding="utf-8"))

    # Verify DB contents directly
    with duckdb.connect(str(results_db), read_only=True) as conn:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        assert set(BAR_EVALUATION_TABLES) == tables
        assert "per_bar_labels" not in " ".join(tables)
        assert conn.execute("SELECT COUNT(*) FROM bar_evaluation_runs").fetchone()[0] == 2

    ts_detail = query_bar_evaluation_detail(results_db, ts_id_hash)
    assert ts_detail["run"]["analysis_mode"] == "time_series"
    assert ts_detail["run"]["frequency"] == "1min"
    assert ts_detail["run"]["calendar_id"] == "CME Globex Equity"
    assert ts_detail["run"]["per_bar_labels_path"] == ts_artifacts["per_bar_labels"]["path"]
    assert ts_detail["run"]["per_bar_labels_sha256"] == ts_artifacts["per_bar_labels"]["sha256"]
    assert ts_detail["run"]["summary_sha256"] == ts_artifacts["evaluation_summary"]["sha256"]
    assert ts_detail["run"]["run_id"] == "ts_run_1"
    assert ts_detail["run"]["exclusion_summary"] == ts_summary["exclusion_summary"]
    assert ts_detail["cross_sections"] == []
    _assert_df_matches_records(ts_ic_csv, ts_detail["ic_by_horizon"])
    _assert_df_matches_records(ts_q_csv, ts_detail["quantile_stats"])
    # Preserve leading zero on instrument_id
    assert ts_detail["ic_by_horizon"][0]["instrument_id"] == "000001"

    cs_detail = query_bar_evaluation_detail(results_db, cs_id_hash)
    assert cs_detail["run"]["analysis_mode"] == "cross_sectional"
    assert cs_detail["run"]["frequency"] == "5min"
    _assert_df_matches_records(cs_ic_csv, cs_detail["ic_by_horizon"])
    _assert_df_matches_records(cs_q_csv, cs_detail["quantile_stats"])
    assert len(cs_detail["cross_sections"]) == len(cs_summary["cross_sections"])
    for db_cs, sum_cs in zip(cs_detail["cross_sections"], cs_summary["cross_sections"]):
        assert db_cs["signal_timestamp"] == sum_cs["signal_timestamp"]
        assert db_cs["valid_instruments"] == sum_cs["valid_instruments"]
        assert db_cs["cs_status"] == sum_cs["cs_status"]
        _assert_value_matches(db_cs["spearman_ic"], sum_cs["spearman_ic"])


def test_idempotency_and_conflict_rejection(tmp_path: Path) -> None:
    results_db = tmp_path / "results.duckdb"
    ts_bars = _make_ts_bars(tmp_path)
    eval_dir = tmp_path / "eval_out"
    cfg = {
        "bars_csv": str(ts_bars),
        "analysis_mode": "time_series",
        "calendar": "NQ",
        "start_date": "2026-12-01",
        "end_date": "2026-12-01",
        "horizons": [1, 2, 5],
        "n_quantiles": 3,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # First run writes (inserted)
    rc1, p1 = _run_cli(
        tmp_path, "--config", str(cfg_path), "--output-dir", str(eval_dir),
        "--results-db", str(results_db), "--run-id", "run_a",
    )
    assert rc1 == 0
    assert p1["metrics"]["results_db"]["status"] == "inserted"
    id_hash = p1["metrics"]["result_identity"]["identity_hash"]

    # Second run with different run_id & attempt dir is a pure noop (NaN/NULL normalized)
    rc2, p2 = _run_cli(
        tmp_path, "--config", str(cfg_path), "--results-db", str(results_db), "--run-id", "run_b",
    )
    assert rc2 == 0
    assert p2["metrics"]["results_db"]["status"] == "noop"
    assert p2["metrics"]["results_db"]["identity_hash"] == id_hash

    with duckdb.connect(str(results_db), read_only=True) as conn:
        counts_before = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in BAR_EVALUATION_TABLES
        }

    # Tamper one metric inside ic_by_horizon.csv AND summary.json keeping the same identity_hash
    ic_path = eval_dir / "ic_by_horizon.csv"
    ic_df = pd.read_csv(ic_path, dtype=store._CSV_STRING_DTYPES)
    ic_df.loc[0, "spearman_ic"] = 0.999123456
    ic_df.to_csv(ic_path, index=False)

    summary_path = eval_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ic_by_horizon"][0]["spearman_ic"] = 0.999123456
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with pytest.raises(BarEvaluationConflictError, match="Conflict") as exc_info:
        ingest_evaluation(results_db, eval_dir, artifacts=_republished_artifacts(p1))
    assert isinstance(exc_info.value, ValueError)

    with duckdb.connect(str(results_db), read_only=True) as conn:
        counts_after = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in BAR_EVALUATION_TABLES
        }
    assert counts_after == counts_before


def test_transaction_atomicity_rollback_on_mid_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_db = tmp_path / "atomic.duckdb"
    ts_bars = _make_ts_bars(tmp_path)
    eval_dir = tmp_path / "eval"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "bars_csv": str(ts_bars),
                "analysis_mode": "time_series",
                "calendar": "NQ",
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
                "horizons": [1, 2, 5],
                "n_quantiles": 3,
            }
        ),
        encoding="utf-8",
    )
    rc, payload = _run_cli(tmp_path, "--config", str(cfg_path), "--output-dir", str(eval_dir))
    assert rc == 0

    def boom_before_second_table(conn, rows):
        raise RuntimeError("injected failure before bar_evaluation_ic insert")

    monkeypatch.setattr(store, "_insert_ic_rows", boom_before_second_table)
    with pytest.raises(RuntimeError, match="injected failure"):
        ingest_evaluation(results_db, eval_dir, artifacts=payload["artifacts"])

    with duckdb.connect(str(results_db), read_only=True) as conn:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        for t in tables:
            assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0


def test_rejects_tampered_sha256_and_missing_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_db = tmp_path / "results.duckdb"
    ts_bars = _make_ts_bars(tmp_path)
    eval_dir = tmp_path / "eval"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "bars_csv": str(ts_bars),
                "analysis_mode": "time_series",
                "calendar": "NQ",
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
                "horizons": [1, 2, 5],
                "n_quantiles": 3,
            }
        ),
        encoding="utf-8",
    )
    rc, payload = _run_cli(tmp_path, "--config", str(cfg_path), "--output-dir", str(eval_dir))
    assert rc == 0

    # A retry may replace a published file while ingestion waits for the DB lock.
    labels_path = eval_dir / "per_bar_labels.csv"
    original_labels = labels_path.read_bytes()
    original_prepare = store._validate_and_prepare_payload

    def replace_after_parse(*args, **kwargs):
        prepared = original_prepare(*args, **kwargs)
        labels_path.write_bytes(original_labels + b"\n")
        return prepared

    with monkeypatch.context() as patch:
        patch.setattr(store, "_validate_and_prepare_payload", replace_after_parse)
        with pytest.raises(BarEvaluationIntegrityError, match="changed during ingestion"):
            ingest_evaluation(results_db, eval_dir, artifacts=payload["artifacts"])
    labels_path.write_bytes(original_labels)
    with duckdb.connect(str(results_db), read_only=True) as conn:
        assert conn.execute("SHOW TABLES").fetchall() == []
    results_db.unlink()

    # Tamper per_bar_labels.csv -> sha256 mismatch against Harness artifacts descriptor
    labels_path.write_text(labels_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BarEvaluationIntegrityError, match="SHA256 mismatch"):
        ingest_evaluation(results_db, eval_dir, artifacts=payload["artifacts"])
    assert not results_db.exists()

    # Delete summary.json -> rejected
    (eval_dir / "summary.json").unlink()
    with pytest.raises(BarEvaluationIntegrityError, match="missing"):
        ingest_evaluation(results_db, eval_dir, artifacts=payload["artifacts"])
    assert not results_db.exists()

    # Also test CLI write-failure path removes summary.json and sets distinct reason
    from tools.agent_harness import cli

    parsed = cli.build_parser().parse_args(
        [
            "time-series-analysis",
            "--config",
            str(cfg_path),
            "--output-root",
            str(tmp_path / "harness2"),
            "--output-dir",
            str(tmp_path / "eval_fail"),
            "--results-db",
            str(results_db),
        ]
    )
    for error, exit_code, reason in (
        (BarEvaluationConflictError("deadbeef" * 8, "simulated conflict"), 4, "conflict"),
        (BarEvaluationIntegrityError("simulated hash mismatch"), 2, "precondition_failed"),
        (RuntimeError("simulated DuckDB failure"), 3, "write_failed"),
    ):
        def fake_ingest(*args, **kwargs):
            raise error

        monkeypatch.setattr(store, "ingest_evaluation", fake_ingest)
        res = cli._handle_time_series_analysis(parsed)
        assert res.exit_code == exit_code
        assert res.error["code"] == f"bar_evaluation_store_{reason}"
        assert not (tmp_path / "eval_fail" / "summary.json").exists()


def test_requires_all_harness_hashes_before_opening_db(tmp_path: Path) -> None:
    for name in ("summary.json", "per_bar_labels.csv", "ic_by_horizon.csv", "quantile_stats.csv"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    db_path = tmp_path / "unverified.duckdb"
    with pytest.raises(BarEvaluationIntegrityError, match="Harness SHA256"):
        ingest_evaluation(db_path, tmp_path)
    with pytest.raises(BarEvaluationIntegrityError, match="Harness SHA256"):
        ingest_evaluation(db_path, tmp_path, expected_hashes={
            "summary": store.sha256_file(tmp_path / "summary.json"),
        })
    assert not db_path.exists()


@pytest.mark.slow
def test_without_results_db_creates_no_db_and_matches_baseline_artifacts(tmp_path: Path) -> None:
    ts_bars = _make_ts_bars(tmp_path)
    cfg = {
        "bars_csv": str(ts_bars),
        "analysis_mode": "time_series",
        "calendar": "NQ",
        "start_date": "2026-12-01",
        "end_date": "2026-12-01",
        "horizons": [1, 2, 5],
        "n_quantiles": 3,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    eval_no_db = tmp_path / "eval_no_db"
    rc_no, p_no = _run_cli(
        tmp_path, "--config", str(cfg_path), "--output-dir", str(eval_no_db), "--run-id", "no_db"
    )
    assert rc_no == 0
    assert "results_db" not in p_no["metrics"]
    assert {p.name for p in eval_no_db.iterdir()} == {
        "per_bar_labels.csv",
        "ic_by_horizon.csv",
        "quantile_stats.csv",
        "summary.json",
    }
    assert list(tmp_path.rglob("*.duckdb")) == []

    # Config-only results_db must also preserve evaluation output and identity.
    results_db = tmp_path / "with_db.duckdb"
    eval_with_db = tmp_path / "eval_with_db"
    cfg_path.write_text(json.dumps({**cfg, "results_db": str(results_db)}), encoding="utf-8")
    rc_with, p_with = _run_cli(
        tmp_path,
        "--config",
        str(cfg_path),
        "--output-dir",
        str(eval_with_db),
        "--run-id",
        "with_db",
    )
    assert rc_with == 0
    assert json.loads((eval_no_db / "summary.json").read_text(encoding="utf-8")) == json.loads(
        (eval_with_db / "summary.json").read_text(encoding="utf-8")
    )
    assert p_no["metrics"]["result_identity"] == p_with["metrics"]["result_identity"]


def test_preserves_existing_factor_analysis_results_table(tmp_path: Path) -> None:
    results_db = tmp_path / "shared_results.duckdb"
    with duckdb.connect(str(results_db)) as conn:
        conn.execute(
            """CREATE TABLE factor_analysis_results (
                factor_name VARCHAR NOT NULL,
                date DATE NOT NULL,
                code VARCHAR NOT NULL,
                ic DOUBLE
            )"""
        )
        conn.execute(
            "INSERT INTO factor_analysis_results VALUES ('alpha_daily', DATE '2026-01-05', '000001.XSHE', 0.1234)"
        )
        schema_before = conn.execute("DESCRIBE factor_analysis_results").fetchall()
        rows_before = conn.execute("SELECT * FROM factor_analysis_results ORDER BY factor_name, date").fetchall()

    ts_bars = _make_ts_bars(tmp_path)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "bars_csv": str(ts_bars),
                "analysis_mode": "time_series",
                "calendar": "NQ",
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
                "horizons": [1, 2, 5],
                "n_quantiles": 3,
            }
        ),
        encoding="utf-8",
    )
    rc, payload = _run_cli(tmp_path, "--config", str(cfg_path), "--results-db", str(results_db))
    assert rc == 0 and payload["ok"] is True
    run = query_bar_evaluation_runs(results_db)[0]
    assert run["run_id"] == payload["command"]["run_id"]

    with duckdb.connect(str(results_db), read_only=True) as conn:
        schema_after = conn.execute("DESCRIBE factor_analysis_results").fetchall()
        rows_after = conn.execute("SELECT * FROM factor_analysis_results ORDER BY factor_name, date").fetchall()
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}

    assert schema_after == schema_before
    assert rows_after == rows_before
    assert tables == {"factor_analysis_results", *BAR_EVALUATION_TABLES}


@pytest.mark.slow
def test_read_bar_evaluations_filters_readonly_and_errors(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.duckdb"
    rc_missing, p_missing = _run_cli(
        tmp_path, "--results-db", str(missing_db), command="read-bar-evaluations"
    )
    assert rc_missing == 2
    assert p_missing["error"]["code"] == "bar_evaluation_read_precondition_failed"

    empty_db = tmp_path / "empty.duckdb"
    duckdb.connect(str(empty_db)).close()
    rc_empty, p_empty = _run_cli(
        tmp_path, "--results-db", str(empty_db), command="read-bar-evaluations"
    )
    assert rc_empty == 2
    assert "Missing bar evaluation table" in p_empty["error"]["message"]

    # Populate with two modes/frequencies
    results_db = tmp_path / "populated.duckdb"
    ts_bars = _make_ts_bars(tmp_path)
    cs_bars, cs_factors = _make_cs_bars_and_factors(tmp_path)

    ts_cfg = tmp_path / "ts.json"
    ts_cfg.write_text(
        json.dumps(
            {
                "bars_csv": str(ts_bars),
                "analysis_mode": "time_series",
                "calendar": "NQ",
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
                "horizons": [1],
                "n_quantiles": 3,
            }
        ),
        encoding="utf-8",
    )
    cs_cfg = tmp_path / "cs.json"
    cs_cfg.write_text(
        json.dumps(
            {
                "bars_csv": str(cs_bars),
                "factors_csv": str(cs_factors),
                "analysis_mode": "cross_sectional",
                "calendar": "XSHG",
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
                "horizons": [1],
                "n_quantiles": 2,
            }
        ),
        encoding="utf-8",
    )

    _, ts_res = _run_cli(tmp_path, "--config", str(ts_cfg), "--results-db", str(results_db))
    _, cs_res = _run_cli(tmp_path, "--config", str(cs_cfg), "--results-db", str(results_db))
    ts_hash = ts_res["metrics"]["result_identity"]["identity_hash"]
    cs_hash = cs_res["metrics"]["result_identity"]["identity_hash"]

    # Filter by identity_hash
    json_out = tmp_path / "query_out.json"
    rc_q1, q1 = _run_cli(
        tmp_path,
        "--results-db",
        str(results_db),
        "--identity-hash",
        ts_hash,
        "--json-out",
        str(json_out),
        command="read-bar-evaluations",
    )
    assert rc_q1 == 0
    assert q1["metrics"]["run_count"] == 1
    assert q1["metrics"]["runs"][0]["identity_hash"] == ts_hash
    assert json.loads(json_out.read_text(encoding="utf-8"))["metrics"]["run_count"] == 1

    # Filter by analysis_mode, frequency, calendar_id, time range
    rc_q2, q2 = _run_cli(
        tmp_path,
        "--results-db",
        str(results_db),
        "--analysis-mode",
        "cross_sectional",
        "--frequency",
        "5min",
        "--calendar-id",
        "XSHG",
        "--start-utc",
        "2020-01-01T00:00:00Z",
        "--end-utc",
        "2099-12-31T23:59:59Z",
        command="read-bar-evaluations",
    )
    assert rc_q2 == 0
    assert q2["metrics"]["run_count"] == 1
    assert q2["metrics"]["runs"][0]["identity_hash"] == cs_hash

    # Out-of-range time filter returns 0 runs
    rc_q3, q3 = _run_cli(
        tmp_path,
        "--results-db",
        str(results_db),
        "--start-utc",
        "2099-01-01T00:00:00Z",
        command="read-bar-evaluations",
    )
    assert rc_q3 == 0 and q3["metrics"]["run_count"] == 0

    # Read-only connection rejects any write attempt
    with store._open_read_only(results_db) as ro_conn:
        with pytest.raises(duckdb.InvalidInputException):
            ro_conn.execute("DELETE FROM bar_evaluation_runs")


def test_monitor_refresh_three_database_states(tmp_path: Path) -> None:
    # 1) New empty DuckDB (no bar tables, no factor_analysis_results)
    empty_db = tmp_path / "new.duckdb"
    duckdb.connect(str(empty_db)).close()
    rc1, m1 = _run_cli(tmp_path, "--database-path", str(empty_db), command="monitor-refresh")
    assert rc1 == 0 and m1["ok"] is True
    assert m1["metrics"]["backend"] == "duckdb"
    assert m1["metrics"]["available"] is True
    assert m1["metrics"]["table_count"] == 0
    assert m1["metrics"]["factor_count"] == 0
    assert m1["metrics"]["database_path"] == str(empty_db)
    assert m1["metrics"]["bar_evaluation"]["all_tables_exist"] is False
    assert m1["metrics"]["bar_evaluation"]["any_table_exists"] is False
    assert m1["metrics"]["bar_evaluation"]["latest_ingested_at_utc"] is None
    for tbl in BAR_EVALUATION_TABLES:
        assert m1["metrics"]["bar_evaluation"]["tables"][tbl] == {"exists": False, "row_count": None}

    # 2) Legacy DuckDB (has factor_analysis_results only, no bar tables)
    legacy_db = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(legacy_db)) as conn:
        conn.execute("CREATE TABLE factor_analysis_results (factor_name VARCHAR, ic DOUBLE)")
        conn.execute("INSERT INTO factor_analysis_results VALUES ('f1', 0.1), ('f2', 0.2)")
    rc2, m2 = _run_cli(tmp_path, "--database-path", str(legacy_db), command="monitor-refresh")
    assert rc2 == 0 and m2["ok"] is True
    assert m2["metrics"]["table_count"] == 1
    assert m2["metrics"]["factor_count"] == 2
    assert m2["metrics"]["bar_evaluation"]["all_tables_exist"] is False
    assert m2["metrics"]["bar_evaluation"]["run_count"] == 0

    # 3) DuckDB with both factor_analysis_results and bar_evaluation_* tables
    with duckdb.connect(str(legacy_db)) as conn:
        pass
    ts_bars = _make_ts_bars(tmp_path)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "bars_csv": str(ts_bars),
                "analysis_mode": "time_series",
                "calendar": "NQ",
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
                "horizons": [1],
                "n_quantiles": 3,
            }
        ),
        encoding="utf-8",
    )
    rc_eval, _ = _run_cli(tmp_path, "--config", str(cfg_path), "--results-db", str(legacy_db))
    assert rc_eval == 0

    rc3, m3 = _run_cli(tmp_path, "--database-path", str(legacy_db), command="monitor-refresh")
    assert rc3 == 0 and m3["ok"] is True
    assert m3["metrics"]["table_count"] == 5
    assert m3["metrics"]["factor_count"] == 2
    bar_eval = m3["metrics"]["bar_evaluation"]
    assert bar_eval["all_tables_exist"] is True
    assert bar_eval["any_table_exists"] is True
    assert bar_eval["run_count"] == 1
    assert bar_eval["tables"]["bar_evaluation_runs"] == {"exists": True, "row_count": 1}
    assert bar_eval["tables"]["bar_evaluation_ic"]["exists"] is True
    assert bar_eval["tables"]["bar_evaluation_ic"]["row_count"] == 2
    assert isinstance(bar_eval["latest_ingested_at_utc"], str)
    assert bar_eval["latest_ingested_at_utc"].endswith("Z")




def test_single_factor_daily_rejects_results_db_and_sqlite_rejected(tmp_path: Path) -> None:
    daily_cfg = tmp_path / "daily.json"
    daily_cfg.write_text(json.dumps({"analysis_mode": "cross_sectional"}), encoding="utf-8")
    rc, payload = _run_cli(
        tmp_path,
        "--config",
        str(daily_cfg),
        "--results-db",
        str(tmp_path / "out.duckdb"),
        command="single-factor-analysis",
    )
    assert rc == 2
    assert payload["error"]["code"] == "results_db_unsupported_for_daily"

    sqlite_file = tmp_path / "legacy.db"
    sqlite_file.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)
    with pytest.raises(ValueError, match="SQLite"):
        ingest_evaluation(sqlite_file, tmp_path)
    rc, payload = _run_cli(tmp_path, "--results-db", str(sqlite_file))
    assert rc == 2
    assert payload["error"]["code"] == "bar_evaluation_store_precondition_failed"


def test_nan_null_normalization_and_empty_quantiles(tmp_path: Path) -> None:
    times = pd.date_range("2026-12-01T20:56Z", periods=10, freq="1min")
    frame = pd.DataFrame(
        {
            "timestamp_utc": times,
            "instrument_id": "NQZ26",
            "frequency": "1min",
            "open": 100.0 + np.arange(10),
            "high": 102.0 + np.arange(10),
            "low": 99.0 + np.arange(10),
            "close": 101.0 + np.arange(10),
            "volume": 1.0,
            "source": "SYNTHETIC",
            "data_version": "NERE749_nan",
            "factor_value": 1.0,  # constant factor -> NaN IC & 0 quantile rows
        }
    )
    bars_path = tmp_path / "const_bars.csv"
    frame.to_csv(bars_path, index=False)
    cfg_path = tmp_path / "const_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "bars_csv": str(bars_path),
                "analysis_mode": "time_series",
                "calendar": "NQ",
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
                "horizons": [1],
                "n_quantiles": 3,
            }
        ),
        encoding="utf-8",
    )
    eval_dir = tmp_path / "const_eval"
    results_db = tmp_path / "const.duckdb"
    rc1, p1 = _run_cli(
        tmp_path, "--config", str(cfg_path), "--output-dir", str(eval_dir), "--results-db", str(results_db)
    )
    assert rc1 == 0 and p1["metrics"]["results_db"]["status"] == "inserted"
    id_hash = p1["metrics"]["result_identity"]["identity_hash"]
    detail = query_bar_evaluation_detail(results_db, id_hash)
    assert detail["quantile_stats"] == []
    assert detail["ic_by_horizon"][0]["ic_status"] == "constant_factor"
    assert detail["ic_by_horizon"][0]["spearman_ic"] is None

    # Re-ingesting NaN vs DuckDB NULL is a noop
    receipt = ingest_evaluation(results_db, eval_dir, artifacts=p1["artifacts"])
    assert receipt.status == "noop"

    # Changing NaN/NULL to finite 0.0 triggers conflict
    ic_path = eval_dir / "ic_by_horizon.csv"
    ic_df = pd.read_csv(ic_path, dtype=store._CSV_STRING_DTYPES)
    ic_df.loc[0, "spearman_ic"] = 0.0
    ic_df.to_csv(ic_path, index=False)
    summary_path = eval_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ic_by_horizon"][0]["spearman_ic"] = 0.0
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with pytest.raises(BarEvaluationConflictError, match="spearman_ic"):
        ingest_evaluation(results_db, eval_dir, artifacts=_republished_artifacts(p1))


def test_precondition_errors_and_dry_run_flags(tmp_path: Path) -> None:
    dir_target = tmp_path / "a_dir"
    dir_target.mkdir()
    with pytest.raises(BarEvaluationPreconditionError, match="directory"):
        ingest_evaluation(dir_target, tmp_path)
    stale_output = tmp_path / "stale_output"
    stale_output.mkdir()
    marker = stale_output / "summary.json"
    marker.write_text("{}", encoding="utf-8")
    config = tmp_path / "invalid_db_config.json"
    config.write_text(json.dumps({"output_dir": str(stale_output)}), encoding="utf-8")
    rc, _ = _run_cli(tmp_path, "--config", str(config), "--results-db", str(dir_target))
    assert rc == 2
    assert not marker.exists()
    with pytest.raises(BarEvaluationPreconditionError, match="does not exist"):
        query_bar_evaluation_runs(tmp_path / "absent.duckdb")

    empty_db = tmp_path / "empty.duckdb"
    duckdb.connect(str(empty_db)).close()
    with pytest.raises(BarEvaluationPreconditionError, match="Missing bar evaluation table"):
        query_bar_evaluation_detail(empty_db, "a" * 64)

    rc_dry_read, p_dry_read = _run_cli(
        tmp_path, "--results-db", str(empty_db), "--dry-run", command="read-bar-evaluations"
    )
    assert rc_dry_read == 0 and p_dry_read["summary"]["reason"] == "dry_run"

    rc_dry_ts, p_dry_ts = _run_cli(
        tmp_path, "--results-db", str(empty_db), "--dry-run", command="time-series-analysis"
    )
    assert rc_dry_ts == 0 and p_dry_ts["metrics"]["planned_results_db"] == str(empty_db.resolve())


def test_write_lock_waits_for_active_writer_process(tmp_path: Path) -> None:
    import multiprocessing
    import time
    from utils.duckdb_results_writer import acquire_exclusive_write_conn

    results_db = tmp_path / "locked.duckdb"
    ts_bars = _make_ts_bars(tmp_path)
    eval_dir = tmp_path / "eval"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "bars_csv": str(ts_bars),
                "analysis_mode": "time_series",
                "calendar": "NQ",
                "start_date": "2026-12-01",
                "end_date": "2026-12-01",
                "horizons": [1],
                "n_quantiles": 3,
            }
        ),
        encoding="utf-8",
    )
    rc, payload = _run_cli(tmp_path, "--config", str(cfg_path), "--output-dir", str(eval_dir))
    assert rc == 0

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_lock_worker, args=(str(results_db), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=30)
        start = time.monotonic()

        def _release_soon():
            time.sleep(0.25)
            release.set()

        import threading

        t = threading.Thread(target=_release_soon, daemon=True)
        t.start()
        receipt = ingest_evaluation(results_db, eval_dir, artifacts=payload["artifacts"], retry_interval_sec=0.05)
        elapsed = time.monotonic() - start
        assert receipt.status == "inserted"
        assert elapsed >= 0.2
    finally:
        release.set()
        proc.join(timeout=15)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
    # Keep unused import check satisfied and ensure lock helper is same repo lock.
    assert acquire_exclusive_write_conn is not None


def _hold_lock_worker(db_path: str, ready, release) -> None:
    from utils.duckdb_results_writer import acquire_exclusive_write_conn

    with acquire_exclusive_write_conn(db_path, retry_interval_sec=0.05):
        ready.set()
        release.wait(timeout=30)
