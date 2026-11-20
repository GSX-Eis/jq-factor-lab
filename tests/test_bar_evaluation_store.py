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


def _make_legacy_schema(conn: duckdb.DuckDBPyConnection) -> str:
    conn.execute("DROP TABLE bar_evaluation_split_metrics")
    for col in ("has_split", "split_config_json", "trial_ledger_json", "calibration_summary_json"):
        conn.execute(f"ALTER TABLE bar_evaluation_runs DROP COLUMN {col}")
    current, config, inputs, calendar, version = conn.execute("""
        SELECT identity_hash, config_hash, input_file_hashes, calendar_hash, code_version
        FROM bar_evaluation_runs
    """).fetchone()
    previous = store.ResultIdentity(
        config_hash=config, input_file_hashes=json.loads(inputs), calendar_hash=calendar,
        code_version=version, result_schema_version="bar_evaluation.v1",
    ).identity_hash()
    for table in BAR_EVALUATION_TABLES:
        if table != "bar_evaluation_split_metrics":
            conn.execute(f"UPDATE {table} SET identity_hash=? WHERE identity_hash=?", [previous, current])
    conn.execute("UPDATE bar_evaluation_runs SET result_schema_version='bar_evaluation.v1'")
    return previous


def test_train_oos_round_trip_and_inference_filters(tmp_path: Path) -> None:
    config = tmp_path / "train_oos.json"
    config.write_text(json.dumps({
        "bars_csv": str(_make_ts_bars(tmp_path)), "analysis_mode": "time_series",
        "calendar": "NQ", "start_date": "2026-12-01", "end_date": "2026-12-01",
        "horizons": [1, 2], "n_quantiles": 3,
        "split": {"train_end": "2026-12-01T21:02:00Z", "oos_start": "2026-12-01T21:03:00Z"},
        "bootstrap": {"n_boot": 9, "min_valid_replicates": 2},
    }), encoding="utf-8")
    db, output = tmp_path / "split.duckdb", tmp_path / "split"
    code, payload = _run_cli(tmp_path, "--config", str(config), "--results-db", str(db),
                             "--output-dir", str(output))
    assert code == 0, payload
    identity = payload["metrics"]["results_db"]["identity_hash"]
    before = db.read_bytes()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    detail = query_bar_evaluation_detail(db, identity)
    run = detail["run"]
    assert run["has_split"] is True
    assert run["trial_ledger_json"] == summary["trial_ledger"]
    assert run["calibration_summary_json"] == summary["interval_usability"]
    assert run["trial_ledger_json"]["split"] == "train"
    assert run["trial_ledger_json"]["selection_complete"] is False
    assert len(detail["split_metrics"]) == 4
    for split, report in summary["splits"].items():
        assert run["split_config_json"][split] == {k: v for k, v in report.items() if k != "horizons"}
        rows = [row for row in detail["split_metrics"] if row["split"] == split]
        assert [r["metrics_json"] for r in rows] == report["horizons"]
        for row in rows:
            metrics = row["metrics_json"]
            assert row["split_purged_count"] == metrics["split_purged_count"]
            assert row["fit_identity_json"] == report["fit_identity"]["1min"]["000001"]
            assert row["interval_lower"] == metrics["bootstrap"]["lower"]
            assert row["n_observations"] == metrics["ic"]["sample_count"]
            assert metrics["bootstrap"]["method_version"] == "ts_boot_adaptive_v1"
            assert row["calibration_status"] == "uncalibrated"
            assert row["significance_interpretation_usable"] is False
            assert row["p_value"] is None
            assert row["n_dependence_adjusted"] is None
            assert row["p_value"] is None
            assert row["significance_interpretation_usable"] is False
    assert any(row["split_purged_count"] > 0 for row in detail["split_metrics"])
    code, read = _run_cli(tmp_path, "--results-db", str(db), "--split", "oos",
                          "--calibration-status", "uncalibrated", "--has-trial-ledger",
                          "--no-significance-interpretation-usable", command="read-bar-evaluations")
    assert code == 0 and {row["split"] for row in read["metrics"]["split_metrics"]} == {"oos"}
    assert query_bar_evaluations(db, split="full")["runs"] == []
    assert query_bar_evaluations(db, has_trial_ledger=False)["runs"] == []
    code, monitor = _run_cli(tmp_path, "--database-path", str(db), command="monitor-refresh")
    assert code == 0
    assert monitor["metrics"]["bar_evaluation"]["tables"]["bar_evaluation_split_metrics"]["row_count"] == 4
    assert db.read_bytes() == before

    receipt = ingest_evaluation(db, output, artifacts=payload["artifacts"])
    assert receipt.status == "noop" and receipt.split_rows == 4

    # Stored JSON compares by structure, and new typed float fields normalize missing values.
    with duckdb.connect(str(db)) as conn:
        conn.execute("UPDATE bar_evaluation_runs SET trial_ledger_json=?", [json.dumps(run["trial_ledger_json"], indent=2)])
        conn.execute("UPDATE bar_evaluation_split_metrics SET p_value = 'NaN'::DOUBLE")
    assert ingest_evaluation(db, output, artifacts=payload["artifacts"]).status == "noop"

    for mutation, conflict in (
        ("UPDATE bar_evaluation_runs SET has_split = false", "has_split"),
        ("UPDATE bar_evaluation_split_metrics SET split_purged_count = 42 WHERE row_index = 0", "split_purged_count"),
    ):
        with duckdb.connect(str(db)) as conn:
            conn.execute(mutation)
        changed = db.read_bytes()
        failed_output = tmp_path / f"conflict-{conflict}"
        code, result = _run_cli(tmp_path, "--config", str(config), "--results-db", str(db),
                                "--output-dir", str(failed_output))
        assert code == 4 and result["error"]["code"] == "bar_evaluation_store_conflict", result
        assert conflict in result["error"]["message"]
        assert not (failed_output / "summary.json").exists()
        assert db.read_bytes() == changed
        with duckdb.connect(str(db)) as conn:
            conn.execute("UPDATE bar_evaluation_runs SET has_split = true")


def test_legacy_read_only_and_explicit_migration_preserve_history(tmp_path: Path) -> None:
    db = tmp_path / "legacy.duckdb"
    bars, factors = _make_cs_bars_and_factors(tmp_path)
    config = tmp_path / "cross.json"
    options = {
        "bars_csv": str(bars), "factors_csv": str(factors), "analysis_mode": "cross_sectional",
        "calendar": "NQ", "start_date": "2026-12-01", "end_date": "2026-12-01",
        "horizons": [1], "n_quantiles": 2,
    }
    config.write_text(json.dumps(options), encoding="utf-8")
    code, old = _run_cli(tmp_path, "--config", str(config), "--results-db", str(db))
    assert code == 0, old
    with duckdb.connect(str(db)) as conn:
        identity = _make_legacy_schema(conn)
        conn.execute("CREATE TABLE unrelated (value INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO unrelated VALUES (91)")
        legacy_tables = tuple(row[0] for row in conn.execute("SHOW TABLES").fetchall())
        historical = {table: conn.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
                      for table in legacy_tables}
    before = db.read_bytes()

    code, read = _run_cli(tmp_path, "--results-db", str(db), command="read-bar-evaluations")
    assert code == 0, read
    assert read["metrics"]["split_metrics"] == []
    assert read["metrics"]["cross_sections"]
    run, = read["metrics"]["runs"]
    assert run["result_schema_version"] == "bar_evaluation.v1"
    new_columns = ("has_split", "split_config_json", "trial_ledger_json", "calibration_summary_json")
    assert all(run[c] is None for c in new_columns)
    assert query_bar_evaluation_detail(db, identity)["split_metrics"] == []
    code, report = _run_cli(tmp_path, "--database-path", str(db), command="monitor-refresh")
    assert code == 0, report
    monitor = report["metrics"]["bar_evaluation"]
    assert monitor["all_tables_exist"] is True
    assert monitor["tables"]["bar_evaluation_split_metrics"] == {"exists": False, "row_count": None}

    options["trial_selection"] = {"method": "bh"}
    config.write_text(json.dumps(options), encoding="utf-8")
    output = tmp_path / "evaluation"
    args = ("--config", str(config), "--results-db", str(db), "--output-dir", str(output))
    code, rejected = _run_cli(tmp_path, *args)
    assert code == 2 and rejected["error"]["code"] == "bar_evaluation_store_precondition_failed", rejected
    message = rejected["error"]["message"]
    assert all(col in message for col in new_columns)
    assert "bar_evaluation_split_metrics" in message and "migrate_bar_evaluation_db" in message
    assert not (output / "summary.json").exists()
    assert db.read_bytes() == before

    store.migrate_bar_evaluation_db(db)
    store.migrate_bar_evaluation_db(db)
    with duckdb.connect(str(db), read_only=True) as conn:
        for table, rows in historical.items():
            migrated = conn.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
            expected = [row + (None,) * len(new_columns) for row in rows] if table == "bar_evaluation_runs" else rows
            assert migrated == expected
        assert conn.execute("SELECT count(*) FROM bar_evaluation_split_metrics").fetchone() == (0,)
        assert conn.execute("SELECT schema_name FROM duckdb_schemas() WHERE schema_name='meta'").fetchall() == []

    code, fresh = _run_cli(tmp_path, *args)
    assert code == 0, fresh
    detail = query_bar_evaluation_detail(db, fresh["metrics"]["results_db"]["identity_hash"])
    trial, = detail["run"]["trial_ledger_json"]["candidates"]
    row, = detail["split_metrics"]
    assert row["split"] == "full" and row["interval_method"] == "hac"
    assert row["p_value"] == trial["p_value"] and row["p_value"] is not None
    assert row["n_observations"] == trial["n_observations"]
    assert row["n_dependence_adjusted"] is None
    assert row["significant"] is False
    assert query_bar_evaluation_detail(db, identity)["run"] == run


def _make_calibrated_hac_config(tmp_path: Path) -> Path:
    rng = np.random.default_rng(511)
    start = pd.Timestamp("2026-12-01T11:31Z")
    times = pd.date_range(start, periods=525, freq="1min")
    frames = []
    for i in range(32):
        price = rng.uniform(99, 120, len(times))
        noise = rng.uniform(-1, 1, len(times))
        factor = rng.normal(size=len(times))
        frames.append(pd.DataFrame(dict(
            timestamp_utc=times, instrument_id=f"A{i:02}", frequency="1min", open=price,
            high=price + 2, low=price - 2, close=price + noise + np.tanh(np.r_[0, factor[:-1]]),
            volume=1.0, source="SYNTHETIC", data_version="NERE931", factor_value=factor,
        )))
    bars = tmp_path / "dense.csv"
    pd.concat(frames).to_csv(bars, index=False)
    config = tmp_path / "calibrated.json"
    split = {
        "train_end": (start + pd.Timedelta(minutes=512)).isoformat(),
        "oos_start": (start + pd.Timedelta(minutes=516)).isoformat(),
    }
    config.write_text(json.dumps({
        "bars_csv": str(bars), "analysis_mode": "cross_sectional",
        "calendar": "NQ", "start_date": "2026-12-01", "end_date": "2026-12-01", "horizons": [1],
        "split": split,
        "calibration_scenario": "iid",
    }), encoding="utf-8")
    return config


@pytest.mark.parametrize("hac_lag", [None, "auto", 0], ids=["default", "auto", "fixed"])
def test_calibrated_hac_significance_round_trip(tmp_path: Path, hac_lag: str | int | None) -> None:
    config = _make_calibrated_hac_config(tmp_path)
    if hac_lag is not None:
        settings = json.loads(config.read_text(encoding="utf-8"))
        settings["hac_lag"] = hac_lag
        if hac_lag == 0:
            settings["calibration_scenario"] = "independent_null"
        config.write_text(json.dumps(settings), encoding="utf-8")
    db = tmp_path / "calibrated.duckdb"
    code, payload = _run_cli(tmp_path, "--config", str(config), "--results-db", str(db))
    assert code == 0, payload
    data = query_bar_evaluations(db, calibration_status="calibrated_pass", significance_interpretation_usable=True)
    row, = data["split_metrics"]
    ledger = data["runs"][0]["trial_ledger_json"]
    assert ledger["best"] == ["factor_value", 1]
    assert row["split"] == "train" and row["significant"] is True
    assert row["n_observations"] == 512 and row["n_dependence_adjusted"] is None
    method = "two_sided_normal_hac" if hac_lag == 0 else "two_sided_student_t_hac"
    assert row["p_value_method"] == ledger["candidates"][0]["p_value_method"] == method
    assert row["p_value"] == row["adjusted_p_value"] == row["inference_adjusted_p_value"] < 0.05
    assert row["metrics_json"]["interval_usability"]["calibration_evidence"]["passes"] is True
    assert row["interval_lower"] == row["metrics_json"]["hac"]["lower"]
    assert query_bar_evaluations(db, split="oos")["split_metrics"][0]["significant"] is None

    summary_path = Path(data["runs"][0]["summary_path"])
    original = summary_path.read_text(encoding="utf-8")
    cases = [("variance", "HAC bounds"), ("lower", "HAC bounds"), ("p_value", "HAC p value")]
    if hac_lag != 0:
        cases.extend([("pattern", "HAC bandwidth and missing pattern"),
                      ("evidence", "interval usability matching")])
    for changed, diagnostic in cases:
        summary = json.loads(original)
        train = summary["splits"]["train"]["horizons"][0]
        oos = summary["splits"]["oos"]["horizons"][0]
        candidate, = summary["trial_ledger"]["candidates"]
        # Keep the trained ledger intact while testing forged adaptive evidence on OOS.
        row_under_test = train if hac_lag == 0 else oos
        if changed == "variance":
            row_under_test["hac"]["variance"] = 1
        elif changed == "lower":
            row_under_test["hac"]["lower"] -= 0.001
            if hac_lag == 0:
                candidate["interval"][0] = train["hac"]["lower"]
        elif changed == "p_value":
            row_under_test["p_value"] = 1e-15
            if hac_lag == 0:
                for field in ("p_value", "adjusted_p_value", "inference_adjusted_p_value"):
                    candidate[field] = train["p_value"]
        elif changed == "pattern":
            oos["hac"]["missing_pattern"] = "periodic_17_3"
        else:
            assert oos["interval_usability"]["calibration_status"] == "uncalibrated"
            oos["interval_usability"] = train["interval_usability"]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        rejected = tmp_path / f"inconsistent_hac_{changed}.duckdb"
        duckdb.connect(str(rejected)).close()
        with pytest.raises(BarEvaluationIntegrityError, match=diagnostic):
            ingest_evaluation(rejected, artifacts=_republished_artifacts(payload))
        with duckdb.connect(str(rejected), read_only=True) as conn:
            assert conn.execute("SHOW TABLES").fetchall() == []


@pytest.mark.parametrize("per_instrument", [False, True], ids=["single_calendar", "per_instrument"])
def test_python_export_ingests_adaptive_hac_with_periodic_gaps(tmp_path: Path, per_instrument: bool) -> None:
    from backtest_and_analysis.time_series import (
        RESULT_SCHEMA_VERSION, BarReturnLabeler, evaluate_bars, read_bar_csv, write_evaluation_artifacts,
    )
    from utils.session_calendar import SessionCalendar

    config = json.loads(_make_calibrated_hac_config(tmp_path).read_text(encoding="utf-8"))
    bars = read_bar_csv(config["bars_csv"])
    minute = (bars["timestamp_utc"] - bars["timestamp_utc"].min()) // pd.Timedelta(minutes=1)
    calendar = SessionCalendar("NQ", config["start_date"], config["end_date"])
    calendars = dict.fromkeys(bars["instrument_id"].unique(), calendar) if per_instrument else calendar
    result = evaluate_bars(
        bars.loc[minute % 17 < 15], labeler=BarReturnLabeler(calendars, horizons=[1]),
        analysis_mode="cross_sectional", split=config["split"], calibration_scenario="gaps",
    )
    train, = result.splits["train"]["horizons"]
    assert train["hac"]["grid_size"] == 512 and train["hac"]["n_valid"] == 422
    assert train["hac"]["missing_pattern"] == "periodic_17_3"
    assert train["interval_usability"]["calibration_status"] == "calibrated_pass"

    identity = store.ResultIdentity("a" * 64, {"bars": "b" * 64}, calendar.calendar_hash,
                                    "test-build", RESULT_SCHEMA_VERSION)
    output = tmp_path / "python_export"
    paths = write_evaluation_artifacts(result, output, result_identity=identity)
    db = tmp_path / "python_export.duckdb"
    receipt = ingest_evaluation(db, output, expected_hashes={k: store.sha256_file(v) for k, v in paths.items()})
    assert receipt.status == "inserted"
    row, = query_bar_evaluations(db, split="train")["split_metrics"]
    assert row["calibration_status"] == "calibrated_pass"
    assert row["significance_interpretation_usable"] is True

    original = paths["summary"].read_text(encoding="utf-8")
    if per_instrument:
        for instrument in ("A00", "A31"):
            for field, value in (("end_date", "2026-12-02"), ("rth_reference_calendar", "CME Globex Equity")):
                summary = json.loads(original)
                summary["calendar_identity"]["calendars"][instrument]["schedule"][field] = value
                paths["summary"].write_text(json.dumps(summary), encoding="utf-8")
                with pytest.raises(BarEvaluationIntegrityError, match="interval usability matching"):
                    ingest_evaluation(tmp_path / f"forged_{instrument}_{field}.duckdb", output,
                                      expected_hashes={k: store.sha256_file(v) for k, v in paths.items()})
        return

    for change, error in (("pattern", "HAC bandwidth and missing pattern"),
                          ("coverage", "interval usability matching")):
        summary = json.loads(original)
        assert summary["config"] == {}
        if change == "pattern":
            summary["splits"]["train"]["horizons"][0]["hac"]["missing_pattern"] = "periodic_17_5"
        else:
            summary["calendar_identity"]["schedule"]["end_date"] = "2026-12-02"
        paths["summary"].write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(BarEvaluationIntegrityError, match=error):
            ingest_evaluation(tmp_path / f"forged_{change}.duckdb", output,
                              expected_hashes={k: store.sha256_file(v) for k, v in paths.items()})


@pytest.mark.parametrize("config_source", [
    "calendar", "matching", "legacy", "legacy_continuous", "conflict",
    "default_match", "default_match_descriptive", "default_conflict", "default_unconfigured",
])
def test_store_uses_verified_bucket_strategy(tmp_path: Path, config_source: str) -> None:
    from backtest_and_analysis.time_series import (
        RESULT_SCHEMA_VERSION, BarReturnLabeler, evaluate_bars, read_bar_csv, write_evaluation_artifacts,
    )
    from utils.session_calendar import SessionCalendar

    calendar = SessionCalendar("NQ", "2026-12-01", "2026-12-01")
    published_strategy = None if config_source.startswith("default_") else "session_clock"
    options = {} if config_source == "default_match_descriptive" else {
        "bootstrap": {"n_boot": 9, "min_valid_replicates": 2},
    }
    result = evaluate_bars(
        read_bar_csv(_make_ts_bars(tmp_path)),
        labeler=BarReturnLabeler(calendar, horizons=[1], bucket_strategy=published_strategy),
        analysis_mode="time_series", **options,
    )
    identity = store.ResultIdentity("a" * 64, {"bars": "b" * 64}, calendar.calendar_hash,
                                    "test-build", RESULT_SCHEMA_VERSION)
    config = None if config_source in ("calendar", "default_unconfigured") else {
        "bucket_strategy": ("segment_anchored"
                            if config_source == "conflict" or config_source.startswith("default_match")
                            else "session_clock"),
    }
    if config_source in ("legacy", "legacy_continuous"):
        del result.calendar_identity["bucket_strategy"]
        del result.calendar_identity["schedule"]
        config = ({"continuous_futures": config} if config_source == "legacy_continuous"
                  else config | {"start_date": "2026-12-01", "end_date": "2026-12-01"})
    output = tmp_path / "bucket_export"
    paths = write_evaluation_artifacts(result, output, result_identity=identity, config=config)
    hashes = {key: store.sha256_file(path) for key, path in paths.items()}
    db = tmp_path / "bucket.duckdb"
    if config_source in ("conflict", "default_conflict"):
        with pytest.raises(BarEvaluationIntegrityError, match="bucket_strategy.*disagree"):
            ingest_evaluation(db, output, expected_hashes=hashes)
    else:
        assert ingest_evaluation(db, output, expected_hashes=hashes).status == "inserted"
        row, = query_bar_evaluation_runs(db)
        expected = config["bucket_strategy"] if config_source.startswith("default_match") else published_strategy
        assert row["bucket_strategy"] == expected


@pytest.mark.parametrize("strategy", ["segment_anchored", "session_clock", None])
def test_default_bucket_strategy_with_empty_rth_labels(tmp_path: Path, strategy: str | None) -> None:
    from backtest_and_analysis.time_series import (
        RESULT_SCHEMA_VERSION, BarReturnLabeler, evaluate_bars, read_bar_csv, write_evaluation_artifacts,
    )
    from utils.session_calendar import SessionCalendar

    bars = read_bar_csv(_make_ts_bars(tmp_path)).assign(
        instrument_id="NQ", timestamp_utc=pd.date_range("2026-12-01T10:00Z", periods=12, freq="min"),
    )
    bars.attrs.clear()
    calendar = SessionCalendar("NQ", "2026-12-01", "2026-12-01")
    result = evaluate_bars(bars, labeler=BarReturnLabeler(calendar, horizons=[1], view="RTH"),
                           analysis_mode="time_series")
    assert result.per_bar_labels.empty
    assert result.data_identity["instruments"] == ["NQ"] and result.data_identity["frequencies"] == ["1min"]
    assert result.calendar_identity["bucket_strategy"] is None

    identity = store.ResultIdentity("a" * 64, {"bars": "b" * 64}, calendar.calendar_hash,
                                    "test-build", RESULT_SCHEMA_VERSION)
    output = tmp_path / "rth_export"
    config = {"bucket_strategy": strategy} if strategy is not None else {}
    paths = write_evaluation_artifacts(result, output, result_identity=identity, config=config)
    hashes = {key: store.sha256_file(path) for key, path in paths.items()}
    db = tmp_path / "rth.duckdb"
    if strategy == "session_clock":
        with pytest.raises(BarEvaluationIntegrityError, match="bucket_strategy.*disagree"):
            ingest_evaluation(db, output, expected_hashes=hashes)
    else:
        assert ingest_evaluation(db, output, expected_hashes=hashes).status == "inserted"
        row, = query_bar_evaluation_runs(db)
        assert row["bucket_strategy"] == strategy

    if strategy == "segment_anchored":
        summary = json.loads(paths["summary"].read_text())
        summary["data_identity"]["frequencies"] = []
        paths["summary"].write_text(json.dumps(summary))
        hashes["summary"] = store.sha256_file(paths["summary"])
        with pytest.raises(BarEvaluationIntegrityError, match=r"data_identity\.frequencies"):
            ingest_evaluation(tmp_path / "missing_frequency.duckdb", output, expected_hashes=hashes)


def test_adaptive_bootstrap_observation_round_trips_without_calibration(tmp_path: Path) -> None:
    config = _make_calibrated_hac_config(tmp_path)
    settings = json.loads(config.read_text(encoding="utf-8"))
    bars = pd.read_csv(settings["bars_csv"])
    bars[bars["instrument_id"] == "A00"].to_csv(settings["bars_csv"], index=False)
    settings.update(
        analysis_mode="time_series",
        bootstrap={"block_length": "auto", "n_boot": 9, "min_valid_replicates": 2},
    )
    config.write_text(json.dumps(settings), encoding="utf-8")

    db = tmp_path / "adaptive_bootstrap.duckdb"
    code, payload = _run_cli(tmp_path, "--config", str(config), "--results-db", str(db))
    assert code == 0, payload
    train, = query_bar_evaluations(db, split="train")["split_metrics"]
    use = train["metrics_json"]["interval_usability"]
    assert train["interval_lower"] is not None
    assert train["p_value"] is None and train["calibration_status"] == "uncalibrated"
    assert train["significance_interpretation_usable"] is False
    assert use["adaptive_observation"]["method_version"] == "ts_boot_adaptive_v1"
    assert use["adaptive_observation"]["n_valid"] == 512
    assert use["calibration_evidence"] is None


def test_split_grid_keeps_missing_slots_and_truncated_bars_across_closure(tmp_path: Path) -> None:
    from utils.session_calendar import SessionCalendar
    from utils.time_axis import BarSchedule

    calendar = SessionCalendar("NQ", start_date="2026-12-04", end_date="2026-12-07")
    schedule = BarSchedule(calendar, "60min", view="RTH")
    friday = schedule.for_session("2026-12-04")
    monday = schedule.for_session("2026-12-07")
    assert friday.iloc[-1].is_partial and monday.iloc[-1].is_partial
    slots = pd.concat([friday, monday.drop(index=1)], ignore_index=True)
    assert len(slots) == len(friday) + len(monday) - 1
    bars_path = tmp_path / "sparse_bars.csv"
    price = 100.0 + np.arange(len(slots))
    pd.DataFrame(dict(
        timestamp_utc=slots.start_utc, instrument_id="NQ", frequency="60min", open=price,
        high=price + 2, low=price - 2, close=price + 1, factor_value=np.arange(len(price)),
        volume=1, source="SYNTHETIC", data_version="NERE931",
    )).to_csv(bars_path, index=False)
    config = tmp_path / "sparse.json"
    config.write_text(json.dumps({
        "bars_csv": str(bars_path), "analysis_mode": "time_series", "calendar": "NQ", "view": "RTH",
        "start_date": "2026-12-04", "end_date": "2026-12-07", "horizons": [1],
        "split": {"train_end": monday.iloc[1].end_utc.isoformat(), "oos_start": monday.iloc[2].end_utc.isoformat()},
        "bootstrap": {"n_boot": 9, "min_valid_replicates": 2},
    }), encoding="utf-8")
    db = tmp_path / "sparse.duckdb"
    code, payload = _run_cli(tmp_path, "--config", str(config), "--results-db", str(db))
    assert code == 0, payload
    rows = query_bar_evaluations(db)["split_metrics"]
    train, oos = rows
    assert train["split"] == "train"
    train_metrics = train["metrics_json"]
    assert pd.Timestamp(train_metrics["grid_end_utc"]) == monday.iloc[1].start_utc
    assert train_metrics["bootstrap"]["grid_size"] == len(friday) + 2
    assert oos["split"] == "oos"
    assert oos["metrics_json"]["bootstrap"]["grid_size"] == len(monday) - 2


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

    # Keep the published artifacts consistent while changing the stored value.
    with duckdb.connect(str(results_db)) as conn:
        conn.execute("UPDATE bar_evaluation_ic SET spearman_ic = 0.999123456 WHERE row_index = 0")

    with pytest.raises(BarEvaluationConflictError, match="Conflict") as exc_info:
        ingest_evaluation(results_db, eval_dir, artifacts=p1["artifacts"])
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
                "trial_selection": {"method": "holm"},
            }
        ),
        encoding="utf-8",
    )
    rc, payload = _run_cli(tmp_path, "--config", str(cfg_path), "--output-dir", str(eval_dir))
    assert rc == 0

    insert_splits = store._insert_split_rows

    def boom_after_split_insert(conn, rows):
        assert rows
        insert_splits(conn, rows)
        raise RuntimeError("injected failure after split insert")

    monkeypatch.setattr(store, "_insert_split_rows", boom_after_split_insert)
    with pytest.raises(RuntimeError, match="injected failure"):
        ingest_evaluation(results_db, eval_dir, artifacts=payload["artifacts"])

    with duckdb.connect(str(results_db), read_only=True) as conn:
        assert conn.execute("SHOW TABLES").fetchall() == []


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
    assert m3["metrics"]["table_count"] == 1 + len(BAR_EVALUATION_TABLES)
    assert m3["metrics"]["factor_count"] == 2
    bar_eval = m3["metrics"]["bar_evaluation"]
    assert bar_eval["all_tables_exist"] is True
    assert bar_eval["any_table_exists"] is True
    assert bar_eval["run_count"] == 1
    assert bar_eval["tables"]["bar_evaluation_runs"] == {"exists": True, "row_count": 1}
    assert bar_eval["tables"]["bar_evaluation_ic"]["exists"] is True
    assert bar_eval["tables"]["bar_evaluation_ic"]["row_count"] == 2
    assert bar_eval["tables"]["bar_evaluation_split_metrics"] == {"exists": True, "row_count": 0}
    assert isinstance(bar_eval["latest_ingested_at_utc"], str)
    assert bar_eval["latest_ingested_at_utc"].endswith("Z")




@pytest.fixture
def dashboard_capture() -> tuple[SimpleNamespace, list[tuple[str | None, pd.DataFrame]], list[dict]]:
    recorded_frames: list[tuple[str | None, pd.DataFrame]] = []
    recorded_json: list[dict] = []
    pending_caption: str | None = None

    def record_caption(caption: str | None) -> None:
        nonlocal pending_caption
        pending_caption = caption

    def record_frame(frame: pd.DataFrame, **kwargs: object) -> None:
        recorded_frames.append((pending_caption, frame))
        record_caption(None)

    class FakeSidebar:
        def header(self, *args, **kwargs):
            pass

        def selectbox(self, label, options, **kwargs):
            return options[0]

    class FakeColumn:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    fake_st = SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        caption=record_caption,
        subheader=lambda *args, **kwargs: record_caption(None),
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(f"Unexpected st.error: {args}")),
        json=lambda data: recorded_json.append(data),
        dataframe=record_frame,
        columns=lambda n: [FakeColumn() for _ in range(n)],
        selectbox=lambda label, options, **kwargs: options[0],
        sidebar=FakeSidebar(),
    )
    return fake_st, recorded_frames, recorded_json


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

    # Changing the stored NaN/NULL to finite 0.0 triggers a conflict on replay.
    with duckdb.connect(str(results_db)) as conn:
        conn.execute("UPDATE bar_evaluation_ic SET spearman_ic = 0.0 WHERE row_index = 0")
    with pytest.raises(BarEvaluationConflictError, match="spearman_ic"):
        ingest_evaluation(results_db, eval_dir, artifacts=p1["artifacts"])


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
