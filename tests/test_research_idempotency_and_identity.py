"""Release Gate contract 7: Research Idempotency and factor_id persistence."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from slots import build_slot_from_config
from tools.agent_harness import causal_gate as causal_gate_mod
from tools.agent_harness.causal_gate import validate_causal_gate_evidence
from tools.agent_harness.time_series_analysis import evaluate
from utils.bar_evaluation_store import ingest_evaluation, query_bar_evaluations
from utils.session_calendar import SessionCalendar
from utils.slots import ExecutionProfile, ProviderConfig, load_slot_config
from utils.time_axis import BarSchedule


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "declarative_crypto"


def _build_slot_fixture(tmp_path: Path) -> Path:
    config = load_slot_config(EXAMPLE / "slot.yaml")
    days = ("2026-01-05", "2026-01-06")
    frequency = "60min"
    instruments = ("BTC-USD", "ETH-USD")
    cal = SessionCalendar(
        "crypto_utc",
        days[0],
        days[1],
        custom_sessions=config.calendar.options["custom_sessions"],
    )
    sched = BarSchedule(cal, frequency, view="ETH", bucket_strategy="segment_anchored")
    rows = []
    for inst_idx, inst in enumerate(instruments):
        step = 0
        for day in cal.sessions():
            slots = sched.for_session(day.strftime("%Y-%m-%d"))
            for ts in slots["start_utc"]:
                step += 1
                base = 100.0 + 10.0 * inst_idx + 0.5 * step + np.sin(step * 0.25)
                rows.append(
                    {
                        "time": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "instrument_id": inst,
                        "open": base,
                        "high": base + 1.0,
                        "low": base - 1.0,
                        "close": base + 0.2,
                        "volume": 50.0 + (step % 7) * 5.0,
                    }
                )
    data_dir = tmp_path / "slot_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / "bars.csv", index=False)

    universe_opts = dict(config.universe.options)
    universe_opts["instruments"] = {
        inst: {
            "asset_class": "crypto",
            "currency": "USD",
            "lot_size": 1,
            "contract_multiplier": 1.0,
        }
        for inst in instruments
    }
    universe_opts["universes"] = {"spot": list(instruments)}
    bar_opts = {
        **config.bar_feed.options,
        "frequency": frequency,
        "timestamp_anchor": "bar_start",
        "column_mapping": {
            "time": "timestamp",
            "instrument_id": "instrument_id",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        },
        "paths": ["bars.csv"],
    }
    bar_opts.pop("instrument_id", None)
    new_cfg = replace(
        config,
        calendar=replace(
            config.calendar,
            options={**config.calendar.options, "start": days[0], "end": days[1]},
        ),
        bar_feed=ProviderConfig("csv_files", bar_opts),
        universe=ProviderConfig("static", universe_opts),
        aux_features=ProviderConfig("none", {}),
        execution_profile=ExecutionProfile(
            "open", "close", 1, -1, "decimal", "endpoints_only", "cross_session"
        ),
    )
    slot_json = data_dir / "slot.json"
    slot_json.write_text(json.dumps(new_cfg.as_dict(), indent=2), encoding="utf-8")
    build_slot_from_config(new_cfg, base_dir=data_dir)
    return slot_json


def _write_probe_factor(path: Path, *, window: int = 3, lookahead: bool = False) -> Path:
    body = (
        "        ordered = data.sort_values(['instrument_id', 'timestamp_utc'])\n"
        "        vals = ordered.groupby('instrument_id', sort=False)['close'].shift(-1)\n"
        "        return ordered[['timestamp_utc', 'instrument_id']].assign(factor_value=vals.reindex(data.index))\n"
        if lookahead
        else f"        vals = self.session_rolling(data, 'close', {window}, min_periods=1, within_session=False)\n"
        "        return data[['timestamp_utc', 'instrument_id']].assign(factor_value=vals)\n"
    )
    path.write_text(
        "import pandas as pd\n"
        "from utils.factor_base import FactorBase\n\n"
        "class DerivedProbeFactor(FactorBase):\n"
        "    CAUSAL_VECTORIZED = True\n\n"
        "    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:\n"
        + body,
        encoding="utf-8",
    )
    return path


def _base_slot_config(slot_json: Path, factor_py: Path, **overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "slot": str(slot_json),
        "factor_file": str(factor_py),
        "analysis_mode": "time_series",
        "frequency": "60min",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "horizons": [1, 2],
        "n_quantiles": 2,
        "min_ic_samples": 2,
        "min_cross_section_size": 2,
    }
    cfg.update(overrides)
    return cfg


def test_p201_repeated_slot_evaluation_ignores_elapsed_sec_and_ingests_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot_json = _build_slot_fixture(tmp_path)
    factor_py = _write_probe_factor(tmp_path / "derived_probe_factor.py")
    real_verify = causal_gate_mod.verify_factor_causality
    timings = iter([0.123456, 9.876543])

    def _verify_with_injected_elapsed(*args: Any, **kwargs: Any):
        result = real_verify(*args, **kwargs)
        return replace(result, elapsed_sec=next(timings))

    monkeypatch.setattr(causal_gate_mod, "verify_factor_causality", _verify_with_injected_elapsed)

    dest1 = tmp_path / "eval_run1"
    dest2 = tmp_path / "eval_run2"
    pub1 = evaluate(_base_slot_config(slot_json, factor_py), dest1)
    pub2 = evaluate(_base_slot_config(slot_json, factor_py), dest2)

    assert pub1["metrics"]["causal_gate"]["elapsed_sec"] == pytest.approx(0.123456)
    assert pub2["metrics"]["causal_gate"]["elapsed_sec"] == pytest.approx(9.876543)
    validate_causal_gate_evidence(pub1["metrics"]["causal_gate"])
    validate_causal_gate_evidence(pub2["metrics"]["causal_gate"])

    summary1 = json.loads((dest1 / "summary.json").read_text(encoding="utf-8"))
    summary2 = json.loads((dest2 / "summary.json").read_text(encoding="utf-8"))
    assert "elapsed_sec" not in summary1["config"].get("causal_gate", {})
    assert "elapsed_sec" not in json.dumps(summary1)
    assert summary1["config"] == summary2["config"]
    assert summary1["result_identity"]["config_hash"] == summary2["result_identity"]["config_hash"]
    assert summary1["result_identity"]["identity_hash"] == summary2["result_identity"]["identity_hash"]

    results_db = tmp_path / "results.duckdb"
    r1 = ingest_evaluation(results_db, evaluation_dir=dest1, artifacts=pub1["artifacts"], run_id="r1")
    r2 = ingest_evaluation(results_db, evaluation_dir=dest2, artifacts=pub2["artifacts"], run_id="r2")
    assert r1.status == "inserted"
    assert r2.status == "noop"
    assert r1.identity_hash == r2.identity_hash

    with duckdb.connect(str(results_db), read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM bar_evaluation_runs").fetchone()[0] == 1


def test_p201_real_cli_repeated_execution_produces_noop(tmp_path: Path) -> None:
    slot_json = _build_slot_fixture(tmp_path)
    factor_py = _write_probe_factor(tmp_path / "derived_probe_factor.py")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(_base_slot_config(slot_json, factor_py)), encoding="utf-8")
    results_db = tmp_path / "cli_results.duckdb"

    payloads = []
    for idx in range(2):
        proc = subprocess.run(
            [
                sys.executable, "-m", "tools.agent_harness.cli",
                "time-series-analysis",
                "--config", str(cfg_path),
                "--results-db", str(results_db),
                "--output-root", str(tmp_path / f"out_{idx}"),
                "--run-id", f"run_{idx}",
            ],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120, check=False,
        )
        assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
        payload = json.loads(proc.stdout)
        assert payload["ok"] is True
        validate_causal_gate_evidence(payload["metrics"]["causal_gate"])
        payloads.append(payload)

    assert payloads[0]["metrics"]["results_db"]["status"] == "inserted"
    assert payloads[1]["metrics"]["results_db"]["status"] == "noop"
    assert payloads[0]["metrics"]["result_identity"] == payloads[1]["metrics"]["result_identity"]

    found = query_bar_evaluations(results_db, factor_name="DerivedProbeFactor")
    assert found["run_count"] == 1
    assert found["runs"][0]["factor_id"] == "DerivedProbeFactor"
    assert query_bar_evaluations(results_db, factor_name="factor_value")["run_count"] == 0


@pytest.mark.parametrize(
    ("analysis_mode", "extra_cfg", "expect_ledger"),
    [
        ("time_series", {"trial_selection": {"method": "holm"}}, True),
        ("time_series", {}, False),
        ("cross_sectional", {}, True),
    ],
    ids=["ts_with_trial_selection", "ts_plain", "cross_sectional"],
)
def test_p204_omitted_factor_id_persists_derived_class_name_end_to_end(
    tmp_path: Path, analysis_mode: str, extra_cfg: dict[str, Any], expect_ledger: bool
) -> None:
    slot_json = _build_slot_fixture(tmp_path)
    factor_py = _write_probe_factor(tmp_path / "derived_probe_factor.py")
    dest = tmp_path / "eval_out"
    pub = evaluate(
        _base_slot_config(slot_json, factor_py, analysis_mode=analysis_mode, **extra_cfg),
        dest,
    )
    summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["factor_id"] == "DerivedProbeFactor"
    if expect_ledger:
        assert summary["trial_ledger"] is not None
        assert summary["trial_ledger"]["candidates"]
        assert all(row["factor_id"] == "DerivedProbeFactor" for row in summary["trial_ledger"]["candidates"])

    results_db = tmp_path / "p204.duckdb"
    ingest_evaluation(results_db, evaluation_dir=dest, artifacts=pub["artifacts"], run_id="p204")
    derived_runs = query_bar_evaluations(results_db, factor_name="DerivedProbeFactor")
    assert derived_runs["run_count"] == 1
    assert derived_runs["runs"][0]["factor_id"] == "DerivedProbeFactor"
    assert query_bar_evaluations(results_db, factor_name="factor_value")["run_count"] == 0


def test_p204_explicit_factor_id_override_and_blank_rejection(tmp_path: Path) -> None:
    slot_json = _build_slot_fixture(tmp_path)
    factor_py = _write_probe_factor(tmp_path / "derived_probe_factor.py")
    dest = tmp_path / "explicit_eval"
    pub = evaluate(
        _base_slot_config(
            slot_json,
            factor_py,
            factor_id="CustomProbeAlias",
            trial_selection={"method": "holm"},
        ),
        dest,
    )
    summary = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["factor_id"] == "CustomProbeAlias"
    assert summary["trial_ledger"] is not None
    assert summary["trial_ledger"]["candidates"]
    assert all(row["factor_id"] == "CustomProbeAlias" for row in summary["trial_ledger"]["candidates"])

    results_db = tmp_path / "explicit.duckdb"
    ingest_evaluation(results_db, evaluation_dir=dest, artifacts=pub["artifacts"], run_id="explicit")
    assert query_bar_evaluations(results_db, factor_name="CustomProbeAlias")["run_count"] == 1
    assert query_bar_evaluations(results_db, factor_name="DerivedProbeFactor")["run_count"] == 0

    # Explicit non-empty factor_id with surrounding whitespace must be preserved verbatim.
    spaced_dest = tmp_path / "spaced_eval"
    spaced_pub = evaluate(
        _base_slot_config(
            slot_json,
            factor_py,
            factor_id=" Probe ",
            trial_selection={"method": "holm"},
        ),
        spaced_dest,
    )
    spaced_summary = json.loads((spaced_dest / "summary.json").read_text(encoding="utf-8"))
    assert spaced_summary["config"]["factor_id"] == " Probe "
    assert spaced_summary["trial_ledger"] is not None
    assert spaced_summary["trial_ledger"]["candidates"]
    assert all(row["factor_id"] == " Probe " for row in spaced_summary["trial_ledger"]["candidates"])

    spaced_db = tmp_path / "spaced.duckdb"
    ingest_evaluation(spaced_db, evaluation_dir=spaced_dest, artifacts=spaced_pub["artifacts"], run_id="spaced")
    assert query_bar_evaluations(spaced_db, factor_name=" Probe ")["run_count"] == 1
    assert query_bar_evaluations(spaced_db, factor_name="Probe")["run_count"] == 0

    trimmed_pub = evaluate(
        _base_slot_config(
            slot_json,
            factor_py,
            factor_id="Probe",
            trial_selection={"method": "holm"},
        ),
        tmp_path / "trimmed_eval",
    )
    assert (
        spaced_pub["metrics"]["result_identity"]["config_hash"]
        != trimmed_pub["metrics"]["result_identity"]["config_hash"]
    )
    assert (
        spaced_pub["metrics"]["result_identity"]["identity_hash"]
        != trimmed_pub["metrics"]["result_identity"]["identity_hash"]
    )

    for bad_factor_id in ("", "   ", 123):
        with pytest.raises(ValueError, match="factor_id must be a nonempty string"):
            evaluate(_base_slot_config(slot_json, factor_py, factor_id=bad_factor_id), tmp_path / "bad_out")


def test_p201_nested_config_key_order_is_idempotent(tmp_path: Path) -> None:
    slot_json = _build_slot_fixture(tmp_path)
    factor_py = _write_probe_factor(tmp_path / "derived_probe_factor.py")
    dest1 = tmp_path / "nested_order_1"
    dest2 = tmp_path / "nested_order_2"

    pub1 = evaluate(
        _base_slot_config(
            slot_json,
            factor_py,
            execution_profile={
                "entry_price_field": "open",
                "exit_price_field": "close",
                "min_entry_offset_bars": 1,
            },
            trial_selection={"method": "holm", "alpha": 0.05},
        ),
        dest1,
    )
    pub2 = evaluate(
        _base_slot_config(
            slot_json,
            factor_py,
            execution_profile={
                "min_entry_offset_bars": 1,
                "exit_price_field": "close",
                "entry_price_field": "open",
            },
            trial_selection={"alpha": 0.05, "method": "holm"},
        ),
        dest2,
    )

    assert (dest1 / "summary.json").read_bytes() == (dest2 / "summary.json").read_bytes()
    assert pub1["metrics"]["result_identity"] == pub2["metrics"]["result_identity"]

    results_db = tmp_path / "nested_order.duckdb"
    r1 = ingest_evaluation(results_db, evaluation_dir=dest1, artifacts=pub1["artifacts"], run_id="no1")
    r2 = ingest_evaluation(results_db, evaluation_dir=dest2, artifacts=pub2["artifacts"], run_id="no2")
    assert r1.status == "inserted"
    assert r2.status == "noop"
    assert r1.identity_hash == r2.identity_hash


def test_p201_semantic_changes_avoid_identity_collisions(tmp_path: Path) -> None:
    slot_json = _build_slot_fixture(tmp_path)
    factor_py = _write_probe_factor(tmp_path / "factor_same_path.py", window=3)

    pub_default = evaluate(_base_slot_config(slot_json, factor_py), tmp_path / "default")
    pub_reject = evaluate(
        _base_slot_config(slot_json, factor_py, causal_gate_policy="reject"),
        tmp_path / "reject",
    )
    pub_fallback = evaluate(
        _base_slot_config(slot_json, factor_py, causal_gate_policy="fallback"),
        tmp_path / "fallback",
    )
    pub_renamed = evaluate(
        _base_slot_config(slot_json, factor_py, factor_id="RenamedProbe"),
        tmp_path / "renamed",
    )
    # Overwrite the exact same factor_file path so the test proves source-content changes alter both hashes.
    _write_probe_factor(factor_py, window=4)
    pub_changed_src = evaluate(
        _base_slot_config(slot_json, factor_py),
        tmp_path / "changed_src",
    )

    id_default = pub_default["metrics"]["result_identity"]
    id_reject = pub_reject["metrics"]["result_identity"]
    id_fallback = pub_fallback["metrics"]["result_identity"]
    id_renamed = pub_renamed["metrics"]["result_identity"]
    id_changed_src = pub_changed_src["metrics"]["result_identity"]

    assert id_default["config_hash"] == id_reject["config_hash"]
    assert id_default["identity_hash"] == id_reject["identity_hash"]

    distinct_configs = {
        id_reject["config_hash"],
        id_fallback["config_hash"],
        id_renamed["config_hash"],
        id_changed_src["config_hash"],
    }
    distinct_identities = {
        id_reject["identity_hash"],
        id_fallback["identity_hash"],
        id_renamed["identity_hash"],
        id_changed_src["identity_hash"],
    }
    assert len(distinct_configs) == 4
    assert len(distinct_identities) == 4


def test_p201_fallback_gate_and_stale_causal_gate_config_remain_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot_json = _build_slot_fixture(tmp_path)
    lookahead_py = _write_probe_factor(tmp_path / "lookahead_probe.py", lookahead=True)
    real_verify = causal_gate_mod.verify_factor_causality
    timings = iter([0.25, 4.75])

    def _verify_with_elapsed(*args: Any, **kwargs: Any):
        result = real_verify(*args, **kwargs)
        return replace(result, elapsed_sec=next(timings))

    monkeypatch.setattr(causal_gate_mod, "verify_factor_causality", _verify_with_elapsed)

    dest1 = tmp_path / "fb_1"
    dest2 = tmp_path / "fb_2"
    pub1 = evaluate(
        _base_slot_config(slot_json, lookahead_py, causal_gate_policy="fallback"),
        dest1,
    )
    # Re-evaluate passing a stale causal_gate dict containing elapsed_sec; it must be overwritten cleanly.
    pub2 = evaluate(
        _base_slot_config(
            slot_json,
            lookahead_py,
            causal_gate_policy="fallback",
            causal_gate={"elapsed_sec": 999.0, "status": "stale"},
        ),
        dest2,
    )

    assert pub1["metrics"]["causal_gate"]["status"] == "failed_fallback"
    assert pub1["metrics"]["causal_gate"]["resolved_mode"] == "pit_replay"
    assert pub1["metrics"]["result_identity"] == pub2["metrics"]["result_identity"]

    results_db = tmp_path / "fallback.duckdb"
    r1 = ingest_evaluation(results_db, evaluation_dir=dest1, artifacts=pub1["artifacts"], run_id="fb1")
    r2 = ingest_evaluation(results_db, evaluation_dir=dest2, artifacts=pub2["artifacts"], run_id="fb2")
    assert r1.status == "inserted"
    assert r2.status == "noop"
