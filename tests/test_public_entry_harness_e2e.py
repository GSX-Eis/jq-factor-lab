"""Release Gate Contract 8: Public Entry E2E tests for Harness external-cwd behavior."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tools.agent_harness import cli
from utils.slots import ExecutionProfile, ProviderConfig, SlotConfig, load_slot_config
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SLOT = REPO_ROOT / "examples" / "declarative_crypto" / "slot.yaml"
CALLER_INSTRUMENTS = ("BTC-USD", "ETH-USD")
SUBDIR_INSTRUMENTS = ("SOL-USD", "XRP-USD")


def _slot_instruments(summary: dict[str, Any]) -> list[str]:
    return summary["config"]["slot_config"]["universe"]["options"]["universes"]["spot"]


def _write_declarative_slot(
    slot_dir: Path,
    *,
    days: tuple[str, str] = ("2026-01-05", "2026-01-06"),
    frequency: str = "60min",
    instruments: tuple[str, ...] = CALLER_INSTRUMENTS,
) -> Path:
    slot_dir.mkdir(parents=True, exist_ok=True)
    config = load_slot_config(EXAMPLE_SLOT)
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
                base = 100.0 + 15.0 * inst_idx + 0.4 * step + np.sin(step * 0.25)
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
    pd.DataFrame(rows).to_csv(slot_dir / "bars.csv", index=False)

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
    slot_json = slot_dir / "slot.json"
    slot_json.write_text(json.dumps(new_cfg.as_dict(), indent=2), encoding="utf-8")
    return slot_json


def _write_factor(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import pandas as pd\nfrom utils.factor_base import FactorBase\n\n" + body.lstrip("\n"),
        encoding="utf-8",
    )
    return path


def _prepare_external_workspace(tmp_path: Path, name: str = "external_caller") -> Path:
    caller_dir = (tmp_path / name).resolve()
    caller_dir.mkdir(parents=True, exist_ok=True)
    assert not caller_dir.is_relative_to(REPO_ROOT.resolve()), (
        f"External cwd fixture unexpectedly inside REPO_ROOT: {caller_dir}"
    )
    _write_declarative_slot(caller_dir)
    _write_factor(
        caller_dir / "my_probe_factor.py",
        """
class ProbeRollingFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(data, "close", 3, min_periods=1, within_session=False)
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)
""",
    )
    _write_factor(
        caller_dir / "factors" / "config_probe_factor.py",
        """
class ConfigProbeFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        vals = self.session_rolling(data, "close", 4, min_periods=1, within_session=False)
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=vals)
""",
    )
    _write_factor(
        caller_dir / "lookahead_factor.py",
        """
class LookaheadShiftFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        ordered = data.sort_values(["instrument_id", "timestamp_utc"])
        vals = ordered.groupby("instrument_id", sort=False)["close"].shift(-1)
        return ordered[["timestamp_utc", "instrument_id"]].assign(factor_value=vals.reindex(data.index))
""",
    )
    base_cfg = {
        "analysis_mode": "time_series",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "frequency": "60min",
        "horizons": [1, 2],
        "n_quantiles": 2,
        "min_ic_samples": 2,
        "min_cross_section_size": 2,
    }
    (caller_dir / "config.json").write_text(json.dumps(base_cfg, indent=2), encoding="utf-8")
    return caller_dir


def _run_public_cli(caller_dir: Path, argv: list[str]) -> tuple[int, dict[str, Any], str]:
    env = dict(os.environ)
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_py}" if existing_py else str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tools.agent_harness import cli; sys.exit(cli.main(sys.argv[1:], public_bar_cli=True))",
            *argv,
        ],
        cwd=str(caller_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    payload = json.loads(proc.stdout)
    assert proc.returncode == payload["exit_code"], f"stderr={proc.stderr}\nstdout={proc.stdout}"
    return proc.returncode, payload, proc.stderr


def test_public_entry_cli_external_cwd_relative_factor_and_read_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_dir = _prepare_external_workspace(tmp_path)
    assert not (REPO_ROOT / "my_probe_factor.py").exists()

    rc, payload, stderr = _run_public_cli(
        caller_dir,
        [
            "time-series-analysis",
            "--config", "config.json",
            "--slot", "slot.json",
            "--factor", "my_probe_factor.py",
            "--results-db", "results/eval.duckdb",
            "--output-dir", "out/eval1",
        ],
    )
    assert rc == 0, f"stderr={stderr}\npayload={payload}"
    assert payload["ok"] is True
    assert payload["metrics"]["causal_gate"]["status"] == "passed"
    assert payload["metrics"]["results_db"]["status"] == "inserted"
    identity_hash = payload["metrics"]["results_db"]["identity_hash"]

    eval_summary_path = caller_dir / "out" / "eval1" / "summary.json"
    assert eval_summary_path.is_file()
    summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    assert summary["config"]["factor_file"] == str((caller_dir / "my_probe_factor.py").resolve())
    assert summary["config"]["factor_identity_hash"] == hashlib.sha256(
        (caller_dir / "my_probe_factor.py").read_bytes()
    ).hexdigest()
    assert (caller_dir / "results" / "eval.duckdb").is_file()

    # Read back through public CLI subprocess from external cwd
    read_rc, read_payload, read_stderr = _run_public_cli(
        caller_dir,
        ["read-bar-evaluations", "--results-db", "results/eval.duckdb", "--identity-hash", identity_hash],
    )
    assert read_rc == 0, f"stderr={read_stderr}\npayload={read_payload}"
    assert len(read_payload["metrics"]["runs"]) == 1
    assert read_payload["metrics"]["runs"][0]["identity_hash"] == identity_hash

    # Verify in-process handler with public_output=ExitStack() and read handler from external cwd
    monkeypatch.chdir(caller_dir)
    args = cli.build_parser(public_bar_cli=True).parse_args(
        [
            "time-series-analysis",
            "--config", "config.json",
            "--slot", "slot.json",
            "--factor", "my_probe_factor.py",
            "--results-db", "results/eval_handler.duckdb",
            "--output-dir", "out/eval_handler",
        ]
    )
    with ExitStack() as output:
        direct_result = cli._handle_time_series_analysis(args, public_output=output)
    assert direct_result.ok is True, direct_result.error
    assert direct_result.metrics["results_db"]["status"] == "inserted"
    assert (caller_dir / "out" / "eval_handler" / "summary.json").is_file()

    read_args = cli.build_parser(public_bar_cli=True).parse_args(
        [
            "read-bar-evaluations",
            "--results-db", "results/eval_handler.duckdb",
            "--identity-hash", direct_result.metrics["results_db"]["identity_hash"],
        ]
    )
    direct_read = cli._handle_read_bar_evaluations(read_args)
    assert direct_read.ok is True
    assert len(direct_read.metrics["runs"]) == 1


def test_public_entry_config_relative_factor_file_and_cli_override_precedence(tmp_path: Path) -> None:
    caller_dir = _prepare_external_workspace(tmp_path)
    base_cfg = json.loads((caller_dir / "config.json").read_text(encoding="utf-8"))

    # Case 1: relative factor_file and all relative paths inside config.json (no --factor CLI arg)
    cfg_only = {
        **base_cfg,
        "slot": "slot.json",
        "factor_file": "factors/config_probe_factor.py",
        "results_db": "results/eval2.duckdb",
        "output_dir": "out/eval2",
    }
    (caller_dir / "config_with_factor.json").write_text(json.dumps(cfg_only, indent=2), encoding="utf-8")

    rc, payload, stderr = _run_public_cli(
        caller_dir,
        ["time-series-analysis", "--config", "config_with_factor.json"],
    )
    assert rc == 0, f"stderr={stderr}\npayload={payload}"
    assert payload["ok"] is True
    assert payload["metrics"]["results_db"]["status"] == "inserted"
    summary2 = json.loads((caller_dir / "out" / "eval2" / "summary.json").read_text(encoding="utf-8"))
    expected_cfg_factor = (caller_dir / "factors" / "config_probe_factor.py").resolve()
    assert summary2["config"]["factor_file"] == str(expected_cfg_factor)
    assert summary2["config"]["factor_identity_hash"] == hashlib.sha256(expected_cfg_factor.read_bytes()).hexdigest()

    read_rc, read_payload, _ = _run_public_cli(
        caller_dir,
        [
            "read-bar-evaluations",
            "--results-db", "results/eval2.duckdb",
            "--identity-hash", payload["metrics"]["results_db"]["identity_hash"],
        ],
    )
    assert read_rc == 0
    assert len(read_payload["metrics"]["runs"]) == 1

    # Case 2: a file Slot resolves from the caller cwd even with slot_base_dir, and CLI --factor overrides
    # config["factor_file"]. The two Slots carry different instruments so a silent source switch is visible.
    sub_slot = _write_declarative_slot(caller_dir / "slot_subdir", instruments=SUBDIR_INSTRUMENTS)
    cfg_override = {
        **base_cfg,
        "slot_base_dir": "slot_subdir",
        "slot": "slot.json",
        "factor_file": "factors/config_probe_factor.py",
        "results_db": "results/eval3.duckdb",
        "output_dir": "out/eval3",
    }
    (caller_dir / "config_override.json").write_text(json.dumps(cfg_override, indent=2), encoding="utf-8")

    rc3, payload3, stderr3 = _run_public_cli(
        caller_dir,
        [
            "time-series-analysis",
            "--config", "config_override.json",
            "--factor", "my_probe_factor.py",
        ],
    )
    assert rc3 == 0, f"stderr={stderr3}\npayload={payload3}"
    assert payload3["ok"] is True
    summary3 = json.loads((caller_dir / "out" / "eval3" / "summary.json").read_text(encoding="utf-8"))
    expected_cli_factor = (caller_dir / "my_probe_factor.py").resolve()
    assert summary3["config"]["factor_file"] == str(expected_cli_factor)
    assert summary3["config"]["factor_identity_hash"] == hashlib.sha256(expected_cli_factor.read_bytes()).hexdigest()
    assert summary3["config"]["slot_base_dir"] == str(caller_dir)
    assert _slot_instruments(summary3) == list(CALLER_INSTRUMENTS)

    # The same config with CLI --slot selects the same caller-cwd Slot file.
    rc4, payload4, stderr4 = _run_public_cli(
        caller_dir,
        [
            "time-series-analysis",
            "--config", "config_override.json",
            "--slot", "slot.json",
            "--output-dir", "out/eval4",
            "--results-db", "results/eval4.duckdb",
        ],
    )
    assert rc4 == 0, f"stderr={stderr4}\npayload={payload4}"
    summary4 = json.loads((caller_dir / "out" / "eval4" / "summary.json").read_text(encoding="utf-8"))
    assert summary4["config"]["slot_base_dir"] == str(caller_dir)
    assert _slot_instruments(summary4) == list(CALLER_INSTRUMENTS)

    # slot_base_dir anchors the relative data paths of an inline Slot declaration.
    cfg_inline = {
        **base_cfg,
        "slot_base_dir": "slot_subdir",
        "slot": json.loads(sub_slot.read_text(encoding="utf-8")),
        "factor_file": "factors/config_probe_factor.py",
    }
    (caller_dir / "config_inline.json").write_text(json.dumps(cfg_inline, indent=2), encoding="utf-8")
    rc5, payload5, stderr5 = _run_public_cli(
        caller_dir,
        ["time-series-analysis", "--config", "config_inline.json", "--output-dir", "out/eval5"],
    )
    assert rc5 == 0, f"stderr={stderr5}\npayload={payload5}"
    summary5 = json.loads((caller_dir / "out" / "eval5" / "summary.json").read_text(encoding="utf-8"))
    assert summary5["config"]["slot_base_dir"] == str((caller_dir / "slot_subdir").resolve())
    assert _slot_instruments(summary5) == list(SUBDIR_INSTRUMENTS)


def test_private_mode_retains_repo_relative_path_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_repo = _prepare_external_workspace(tmp_path, name="fake_repo")
    caller_dir = (tmp_path / "outside_caller").resolve()
    caller_dir.mkdir(parents=True, exist_ok=True)
    _write_factor(
        caller_dir / "caller_only_factor.py",
        """
class CallerOnlyFactor(FactorBase):
    CAUSAL_VECTORIZED = True

    def calculate_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[["timestamp_utc", "instrument_id"]].assign(factor_value=data["close"])
""",
    )

    priv_cfg = json.loads((fake_repo / "config.json").read_text(encoding="utf-8"))
    priv_cfg.update(
        slot=str((fake_repo / "slot.json").resolve()),
        output_dir="out/priv_eval",
        results_db="results/priv.duckdb",
    )
    (fake_repo / "priv_cfg.json").write_text(json.dumps(priv_cfg, indent=2), encoding="utf-8")

    monkeypatch.setattr(cli, "REPO_ROOT", fake_repo)
    existing_py = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", f"{REPO_ROOT}{os.pathsep}{existing_py}" if existing_py else str(REPO_ROOT))
    monkeypatch.chdir(caller_dir)

    args = cli.build_parser().parse_args(
        [
            "time-series-analysis",
            "--config", "priv_cfg.json",
            "--factor", "my_probe_factor.py",
            "--output-root", str(tmp_path / "priv_harness"),
        ]
    )
    result = cli._handle_time_series_analysis(args)
    assert result.ok is True, result.error
    assert (fake_repo / "out" / "priv_eval" / "summary.json").is_file()
    assert (fake_repo / "results" / "priv.duckdb").is_file()
    assert not (caller_dir / "out").exists()
    assert not (caller_dir / "results").exists()

    # Private mode config["factor_file"] also resolves against REPO_ROOT, not caller cwd
    priv_cfg_with_factor = {
        **priv_cfg,
        "factor_file": "factors/config_probe_factor.py",
        "output_dir": "out/priv_eval_cfg",
        "results_db": "results/priv_cfg.duckdb",
    }
    (fake_repo / "priv_cfg_factor.json").write_text(json.dumps(priv_cfg_with_factor, indent=2), encoding="utf-8")
    cfg_args = cli.build_parser().parse_args(
        [
            "time-series-analysis",
            "--config", "priv_cfg_factor.json",
            "--output-root", str(tmp_path / "priv_harness_cfg"),
        ]
    )
    cfg_result = cli._handle_time_series_analysis(cfg_args)
    assert cfg_result.ok is True, cfg_result.error
    assert (fake_repo / "out" / "priv_eval_cfg" / "summary.json").is_file()
    assert (fake_repo / "results" / "priv_cfg.duckdb").is_file()

    # Private mode must NOT fall back to caller cwd for --factor
    missing_args = cli.build_parser().parse_args(
        [
            "time-series-analysis",
            "--config", "priv_cfg.json",
            "--factor", "caller_only_factor.py",
            "--output-root", str(tmp_path / "priv_harness_missing"),
        ]
    )
    missing_res = cli._handle_time_series_analysis(missing_args)
    assert missing_res.ok is False
    assert missing_res.exit_code == cli.EXIT_PRECONDITION
    assert missing_res.error["code"] == "time_series_precondition_failed"
    assert str(fake_repo / "caller_only_factor.py") in missing_res.error["message"]


def test_public_entry_invalid_config_path_returns_precondition_error(tmp_path: Path) -> None:
    caller_dir = tmp_path.resolve()
    assert not caller_dir.is_relative_to(REPO_ROOT.resolve())
    config = {"slot": "sub/slot.json", "slot_db": "market\0.duckdb"}
    (caller_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    rc, payload, stderr = _run_public_cli(
        caller_dir,
        ["time-series-analysis", "--config", "config.json", "--output-dir", "out"],
    )

    assert rc == cli.EXIT_PRECONDITION
    assert payload["ok"] is False
    assert payload["error"]["code"] == "time_series_precondition_failed"
    assert payload["summary"]["reason"] == "time_series_precondition_failed"
    assert "Traceback" not in stderr
    assert not (caller_dir / "out").exists()


def test_public_entry_lookahead_rejected_without_bypass_hints(tmp_path: Path) -> None:
    caller_dir = _prepare_external_workspace(tmp_path)
    rc, payload, stderr = _run_public_cli(
        caller_dir,
        [
            "time-series-analysis",
            "--config", "config.json",
            "--slot", "slot.json",
            "--factor", "lookahead_factor.py",
        ],
    )
    assert rc == cli.EXIT_QUALITY_GATE
    assert payload["ok"] is False
    assert payload["error"]["code"] == "causal_fast_path_gate_failed"
    assert payload["summary"]["reason"] == "causal_fast_path_gate_failed"

    combined = (json.dumps(payload, ensure_ascii=False) + "\n" + stderr).lower()
    assert not ("disable" in combined and "causal_vectorized" in combined)
    assert "bypass" not in combined
    assert "关闭校验" not in combined


def _write_market_db_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    import duckdb

    from utils.market_db import init_market_db, load_market_db_profile_yaml
    from utils.market_db.research import open_research_slot
    from utils.market_factor_store import (
        MarketFactorValueStore, build_identity_record, compute_factor_identity_hash,
    )

    monkeypatch.delenv("QUANT_OUTPUT_CONTEXT", raising=False)
    caller_dir = (tmp_path / "market_caller").resolve()
    sub = caller_dir / "sub"
    sub.mkdir(parents=True)
    assert not caller_dir.is_relative_to(REPO_ROOT.resolve())
    profile = load_market_db_profile_yaml(REPO_ROOT / "config/market_databases/us_equity.yaml")
    db = sub / "market.duckdb"
    init_market_db(db, profile)
    with duckdb.connect(str(db)) as connection:
        connection.execute("""
            CREATE TABLE bars AS
            SELECT DATE '2026-01-05' + i::INT + INTERVAL '16 hours' AS timestamp,
                   'AAPL.XNYS' AS instrument_id, 10.0+i AS open, 11.0+i AS high,
                   9.0+i AS low, 10.5+i AS close, 100 AS volume
            FROM range(5) data(i)
        """)
    monkeypatch.setenv("QUANT_DATA_BINDINGS", json.dumps({"market_dbs": {profile.db_id: str(db)}}))
    slot_config = SlotConfig(
        1, "ny_fixture",
        ProviderConfig("pmc", {"calendar_id": "XNYS", "start": "2026-01-05", "end": "2026-01-09"}),
        ProviderConfig("duckdb_table", {
            "table": "bars", "frequency": "1d", "timestamp_anchor": "bar_end",
            "source_timezone": "America/New_York", "volume_mode": "interval", "data_version": "fixture",
            "column_mapping": {name: name for name in (
                "instrument_id", "timestamp", "open", "high", "low", "close", "volume",
            )},
        }),
        ProviderConfig("static", {"instruments": {"AAPL.XNYS": {"asset_class": "equity", "currency": "USD"}}}),
        ProviderConfig("none"), ProviderConfig("open"),
        ExecutionProfile("close", "close", 0, 0, "decimal", "endpoints_only", "cross_session"),
    )
    (sub / "slot.json").write_text(json.dumps(slot_config.as_dict()), encoding="utf-8")

    store = MarketFactorValueStore(caller_dir / "factors")
    with open_research_slot(profile.db_id, slot_config, sub) as (slot, _):
        fingerprint = slot.slot_data_fingerprint
        identity_hash, payload = compute_factor_identity_hash(
            factor_id="alpha", factor_class="tests.ProfileFactor", source_sha256="a" * 64, params={},
            frequency="1d", session_view="ETH", analysis_mode="cross_sectional", bucket_strategy="session",
            calendar_hash=slot.calendar.calendar_hash, code_version="test", slot_id=slot.slot_id,
            slot_config_hash=slot.slot_config_hash, adjustment_policy="none", slot_data_fingerprint=fingerprint,
        )
        record = build_identity_record(factor_identity_hash=identity_hash, payload=payload, code_version_source="test")
        grid = slot.calendar.bar_grid("1d")
        rows = grid[["timestamp_utc", "session_id"]].assign(
            factor_id="alpha", factor_identity_hash=identity_hash, frequency="1d", session_view="ETH",
            instrument_id="AAPL.XNYS", factor_value=np.arange(len(grid), dtype=float), source_build_id="test-build",
            calendar_hash=slot.calendar.calendar_hash, code_version="test",
            created_at=pd.Timestamp("2026-02-01T00:00:00Z"), batch_id=identity_hash[:24],
        )
        store.append_factor_values(rows, identity_record=record, calendar=slot.calendar)
    return caller_dir, profile.db_id


def test_public_entry_market_db_preflight_and_worker_share_slot_db_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_dir, db_id = _write_market_db_workspace(tmp_path, monkeypatch)
    monkeypatch.chdir(caller_dir)

    def run(slot_db: str, name: str):
        cfg = {
            "slot": "sub/slot.json", "slot_db": slot_db, "factor_store_root": "factors", "factor_id": "alpha",
            "analysis_mode": "time_series", "frequency": "1d", "horizons": [1],
            "start_date": "2026-01-05", "end_date": "2026-01-09",
        }
        (caller_dir / f"{name}.json").write_text(json.dumps(cfg), encoding="utf-8")
        args = cli.build_parser(public_bar_cli=True).parse_args([
            "time-series-analysis", "--config", f"{name}.json", "--market-db", db_id,
            "--output-dir", f"out/{name}", "--results-db", f"results/{name}.duckdb",
            "--output-root", str(tmp_path / "harness" / name),
        ])
        with ExitStack() as output:
            return cli._handle_time_series_analysis(args, public_output=output)

    # slot_db is relative to the caller cwd for both preflight and worker.
    ok = run("sub/market.duckdb", "bound")
    assert ok.ok is True, ok.error
    assert ok.metrics["results_db"]["status"] == "inserted"
    summary = json.loads((caller_dir / "out" / "bound" / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["slot_db"] == str(caller_dir / "sub" / "market.duckdb")

    # A slot_db relative to the Slot file directory names ./market.duckdb, which the preflight now rejects
    # instead of letting the worker fail after the preflight accepted it.
    rejected = run("market.duckdb", "unbound")
    assert rejected.ok is False
    assert rejected.exit_code == cli.EXIT_PRECONDITION
    assert rejected.error["code"] == "market_db_precondition_failed"
    assert "Declared database conflicts" in rejected.error["message"]
    assert not (caller_dir / "out" / "unbound" / "summary.json").exists()


@pytest.mark.parametrize(
    "variant",
    ["cli_file_slot", "config_file_slot", "inline_slot", "inline_slot_no_output_dir"],
)
def test_public_entry_valid_slot_dry_run_and_live_execution_parity(
    tmp_path: Path, variant: str,
) -> None:
    caller_dir = _prepare_external_workspace(tmp_path)
    base_cfg = json.loads((caller_dir / "config.json").read_text(encoding="utf-8"))
    sub_slot = _write_declarative_slot(caller_dir / "slot_subdir", instruments=SUBDIR_INSTRUMENTS)
    eval_dir = caller_dir / "out" / "eval"
    results_db = caller_dir / "results" / "eval.duckdb"
    json_out = caller_dir / "dry" / "plan.json"

    if variant == "cli_file_slot":
        cfg_name = "config.json"
        cmd = [
            "time-series-analysis",
            "--config", cfg_name,
            "--slot", "slot.json",
            "--factor", "my_probe_factor.py",
            "--output-dir", "out/eval",
            "--json-out", "dry/plan.json",
            "--results-db", "results/eval.duckdb",
        ]
        expect_output_dir = True
        expect_results_db = True
        expect_json_out = True
    elif variant == "config_file_slot":
        cfg_name = "config_file_slot.json"
        (caller_dir / cfg_name).write_text(
            json.dumps(
                {
                    **base_cfg,
                    "slot": "slot.json",
                    "factor_file": "factors/config_probe_factor.py",
                    "output_dir": "out/eval",
                    "results_db": "results/eval.duckdb",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cmd = [
            "time-series-analysis",
            "--config", cfg_name,
            "--json-out", "dry/plan.json",
        ]
        expect_output_dir = True
        expect_results_db = True
        expect_json_out = True
    elif variant == "inline_slot":
        cfg_name = "config_inline_slot.json"
        (caller_dir / cfg_name).write_text(
            json.dumps(
                {
                    **base_cfg,
                    "slot_base_dir": "slot_subdir",
                    "slot": json.loads(sub_slot.read_text(encoding="utf-8")),
                    "factor_file": "factors/config_probe_factor.py",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cmd = [
            "time-series-analysis",
            "--config", cfg_name,
            "--output-dir", "out/eval",
            "--json-out", "dry/plan.json",
            "--results-db", "results/eval.duckdb",
        ]
        expect_output_dir = True
        expect_results_db = True
        expect_json_out = True
    else:
        cfg_name = "config_inline_minimal.json"
        (caller_dir / cfg_name).write_text(
            json.dumps(
                {
                    **base_cfg,
                    "slot_base_dir": "slot_subdir",
                    "slot": json.loads(sub_slot.read_text(encoding="utf-8")),
                    "factor_file": "factors/config_probe_factor.py",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cmd = ["time-series-analysis", "--config", cfg_name]
        expect_output_dir = False
        expect_results_db = False
        expect_json_out = False

    dry_rc, dry_payload, dry_stderr = _run_public_cli(caller_dir, [*cmd, "--dry-run"])
    assert dry_rc == 0, f"stderr={dry_stderr}\npayload={dry_payload}"
    assert dry_payload["ok"] is True
    assert dry_payload["command"]["mode"] == "dry_run"
    assert dry_payload["summary"]["reason"] == "dry_run"
    assert dry_payload["metrics"]["planned_config"] == cfg_name
    assert "Traceback" not in dry_stderr
    assert not eval_dir.exists()
    assert not results_db.exists()
    if expect_json_out:
        assert json_out.is_file()
        assert json.loads(json_out.read_text(encoding="utf-8")) == dry_payload
    if variant in {"cli_file_slot", "inline_slot"}:
        assert dry_payload["metrics"]["planned_results_db"] == str(results_db.resolve())

    live_rc, live_payload, live_stderr = _run_public_cli(caller_dir, cmd)
    assert live_rc == 0, f"stderr={live_stderr}\npayload={live_payload}"
    assert live_payload["ok"] is True
    assert live_payload["command"]["mode"] == "subprocess"
    if expect_output_dir:
        assert (eval_dir / "summary.json").is_file()
        assert (eval_dir / "per_bar_labels.csv").is_file()
    else:
        eval_summary = Path(
            next(
                item["path"]
                for item in live_payload["artifacts"]
                if item["artifact_type"] == "evaluation_summary"
            )
        )
        assert eval_summary.is_file()
        assert json.loads(eval_summary.read_text(encoding="utf-8"))["config"]["slot_base_dir"] == str(
            (caller_dir / "slot_subdir").resolve()
        )
    if expect_results_db:
        assert live_payload["metrics"]["results_db"]["status"] == "inserted"
        assert results_db.is_file()


def test_public_entry_market_db_slot_dry_run_and_live_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from utils.market_db import load_market_db_profile_yaml
    from utils.market_db.identity import market_db_identity

    caller_dir, db_id = _write_market_db_workspace(tmp_path, monkeypatch)
    profile = load_market_db_profile_yaml(REPO_ROOT / "config/market_databases/us_equity.yaml")
    cfg = {
        "slot": "sub/slot.json",
        "slot_db": "sub/market.duckdb",
        "factor_store_root": "factors",
        "factor_id": "alpha",
        "analysis_mode": "time_series",
        "frequency": "1d",
        "horizons": [1],
        "start_date": "2026-01-05",
        "end_date": "2026-01-09",
    }
    (caller_dir / "market_cfg.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    eval_dir = caller_dir / "out" / "market_eval"
    results_db = caller_dir / "results" / "market_eval.duckdb"
    json_out = caller_dir / "dry" / "market_plan.json"
    cmd = [
        "time-series-analysis",
        "--config", "market_cfg.json",
        "--market-db", db_id,
        "--output-dir", "out/market_eval",
        "--json-out", "dry/market_plan.json",
        "--results-db", "results/market_eval.duckdb",
    ]

    dry_rc, dry_payload, dry_stderr = _run_public_cli(caller_dir, [*cmd, "--dry-run"])
    assert dry_rc == 0, f"stderr={dry_stderr}\npayload={dry_payload}"
    assert dry_payload["ok"] is True
    assert dry_payload["command"]["mode"] == "dry_run"
    assert dry_payload["summary"]["reason"] == "dry_run"
    assert dry_payload["metrics"]["market_db"] == market_db_identity(profile)
    assert dry_payload["metrics"]["planned_results_db"] == str(results_db.resolve())
    assert json.loads(json_out.read_text(encoding="utf-8")) == dry_payload
    assert not eval_dir.exists()
    assert not results_db.exists()

    live_rc, live_payload, live_stderr = _run_public_cli(caller_dir, cmd)
    assert live_rc == 0, f"stderr={live_stderr}\npayload={live_payload}"
    assert live_payload["ok"] is True
    assert live_payload["command"]["mode"] == "subprocess"
    assert live_payload["metrics"]["results_db"]["status"] == "inserted"
    assert (eval_dir / "summary.json").is_file()
    assert results_db.is_file()
