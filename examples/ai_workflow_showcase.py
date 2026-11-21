"""Run a preregistered offline factor screen through the public CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_and_analysis.evaluation.calibration import Trial
from backtest_and_analysis.evaluation.types import (
    AdaptiveCalibrationObservation, CalibrationEvidence, IntervalUsability,
)
from backtest_and_analysis.time_series import BarTrialFamily


CANDIDATES = ("sensor_a", "sensor_b", "constant_return_control", "constant_signal_control", "short_sample")
MIN_TRAIN_IC = 0.20
MIN_TRAIN_PAIRS = 512
ROUND_TRIP_COST_BPS = 2


def invoke(root: Path, name: str, *args: str, expected_exit: int = 0) -> dict[str, Any]:
    command = [sys.executable, "-m", "tools.demo_cli", *args, "--compact"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (root / f"{name}.json").write_text(result.stdout, encoding="utf-8")
    (root / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    payload = json.loads(result.stdout)
    if (result.returncode, payload["exit_code"], payload["ok"]) != (expected_exit, expected_exit, expected_exit == 0):
        raise RuntimeError(f"{name}: unexpected exit or response. See {root / f'{name}.json'}")
    return payload


def write_config(root: Path, name: str, bars: pd.DataFrame, split: dict[str, str]) -> Path:
    csv_path = root / f"{name}.csv"
    bars.to_csv(csv_path, index=False)
    config = root / f"{name}-config.json"
    config.write_text(json.dumps({
        "bars_csv": str(csv_path), "calendar": "NQ", "start_date": "2026-12-01", "end_date": "2026-12-01",
        "analysis_mode": "time_series", "factor_id": name, "horizons": [1], "entry_offset_bars": 1,
        "min_ic_samples": 4, "split": split, "calibration_scenario": "positive_effect",
        "bootstrap": {"block_length": "auto", "seed": 314}, "trial_selection": {"method": "holm"},
    }, indent=2), encoding="utf-8")
    return config


def evaluation_command(root: Path, name: str, config: Path) -> tuple[str, ...]:
    return (
        "time-series-analysis", "--config", str(config), "--diagnostics-report",
        "--output-dir", str(root / name), "--output-root", str(root / "harness"),
    )


def evaluate(root: Path, name: str, config: Path, *options: str) -> dict[str, Any]:
    command = (*evaluation_command(root, name, config), *options)
    plan = invoke(root, f"{name}-plan", *command, "--dry-run")
    if not plan["dry_run"] or (root / name).exists():
        raise RuntimeError(f"{name}: expected a plan without artifacts")
    result = invoke(root, f"{name}-evaluation", *command)
    for artifact in result["artifacts"]:
        if "sha256" in artifact:
            actual = hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
            if actual != artifact["sha256"]:
                raise RuntimeError(f"{name}: artifact hash mismatch for {artifact['path']}")
    return result["metrics"]


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="jq-ai-workflow-", dir=Path.cwd()))
    print(f"Evidence directory: {root}", file=sys.stderr, flush=True)
    rng = np.random.default_rng(511)
    times = pd.date_range("2026-12-01T00:01Z", periods=1048, freq="1min")
    latent = rng.normal(size=len(times))
    signals = pd.DataFrame({
        "sensor_a": latent + rng.normal(scale=0.5, size=len(times)),
        "sensor_b": latent + rng.normal(scale=0.5, size=len(times)),
    })
    # Generate the next Bar return from the previous Bar's latent state.
    # Each sensor/return pair is Gaussian with population correlation ≈ 0.35.
    returns = 0.00085 * np.r_[0, latent[:-1]] + rng.normal(scale=0.002, size=len(times))
    close = 100 * (1 + returns)
    bars = pd.DataFrame({
        "timestamp_utc": times, "instrument_id": "DEMO", "frequency": "1min",
        "open": 100.0, "high": np.maximum(100, close) + 1, "low": np.minimum(100, close) - 1, "close": close,
        "volume": 1, "source": "SYNTHETIC", "data_version": "showcase-v1",
    })
    split = {"train_end": times[512].isoformat(), "oos_start": times[516].isoformat()}
    registrations = [(name, 1) for name in CANDIDATES]
    family = BarTrialFamily(registrations, method="holm")
    preregistration = {
        "hypothesis": "Two independent noisy measurements of a synthetic state predict the next Bar return.",
        "candidates": [{"factor_id": name, "horizon": horizon} for name, horizon in registrations],
        "split": split, "short_sample": {"observed_factor_values": 8},
        "min_train_ic": MIN_TRAIN_IC, "min_train_pairs": MIN_TRAIN_PAIRS,
        "required_calibration": "calibrated_pass", "weights": "equal across eligible sensors",
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS, "significance_method": family.method,
    }
    (root / "preregistration.json").write_text(json.dumps(preregistration, indent=2), encoding="utf-8")
    # Exercise a genuine CLI precondition failure and retain the error envelope.
    invalid_config = write_config(root, "invalid", bars.assign(factor_value=signals.sensor_a), {
        "train_end": split["oos_start"], "oos_start": split["train_end"],
    })
    invalid = invoke(root, "invalid-plan", *evaluation_command(root, "invalid", invalid_config),
                     "--dry-run", expected_exit=2)

    screened = {}
    eligible = []
    for name, horizon in registrations:
        frame = bars.assign(factor_value=signals[name] if name in signals else signals.sensor_a)
        if name == "constant_return_control":
            frame = frame.assign(close=100.0)
        elif name == "constant_signal_control":
            frame = frame.assign(factor_value=1.0)
        elif name == "short_sample":
            frame = frame.assign(factor_value=signals.sensor_a.where(times < times[8]))
        config = write_config(root, name, frame, split)
        metrics = evaluate(root, name, config)
        training = metrics["splits"]["train"]
        ledger = metrics["trial_ledger"]
        trial, = ledger["candidates"]
        if (ledger["split"] != "train" or ledger["method"] != family.method or ledger["alpha"] != family.alpha
                or any(pd.Timestamp(training[key]) != pd.Timestamp(value) for key, value in split.items())
                or (trial["factor_id"], trial["horizon"]) != (name, horizon)):
            raise RuntimeError(f"{name}: trial does not match the preregistered family")

        # The CLI emits one ledger per evaluation. Import its recorded trial after verifying the split.
        use = trial["interval_usability"]
        if use is not None:
            evidence = use["calibration_evidence"]
            observation = use["adaptive_observation"]
            use = IntervalUsability(
                **{**use, "calibration_evidence": CalibrationEvidence(**evidence) if evidence else None,
                   "adaptive_observation": AdaptiveCalibrationObservation(**observation) if observation else None}
            )
        bounds = trial["interval"]
        family.record((name, horizon), Trial(
            trial["statistic"], trial["p_value"], trial["n_observations"],
            tuple(bounds) if bounds is not None else None, use, trial["p_value_method"],
        ), reason=trial["reason"])

        train, = training["horizons"]
        ic = train["ic"]
        status = train["interval_usability"]["calibration_status"]
        accepted = (status == "calibrated_pass" and ic["sample_count"] >= MIN_TRAIN_PAIRS
                    and ic["spearman_ic"] is not None and ic["spearman_ic"] >= MIN_TRAIN_IC)
        if accepted:
            eligible.append(name)
        screened[name] = {
            "eligible_for_toy_backtest": accepted, "sample_count": ic["sample_count"],
            "spearman_ic": ic["spearman_ic"], "reason": ic["reason"], "calibration_status": status,
            "p_value": trial["p_value"], "trial_reason": trial["reason"],
        }

    if eligible != list(CANDIDATES[:2]):
        raise RuntimeError(f"Unexpected synthetic screening results: {screened}")
    combined = bars.assign(factor_value=signals[eligible].mean(axis=1))
    config = write_config(root, "equal_weight", combined, split)
    db = root / "evaluations.duckdb"
    metrics = evaluate(root, "equal_weight", config, "--results-db", str(db))

    # One position per nonoverlapping 1-Bar label, opened and closed on that Bar.
    labels = pd.read_csv(root / "equal_weight" / "per_bar_labels.csv",
                         parse_dates=["signal_available_at", "signal_timestamp"])
    oos = labels.loc[labels["status"].eq("valid") & labels.signal_available_at.ge(pd.Timestamp(split["oos_start"]))]
    score = combined.set_index("timestamp_utc")["factor_value"].reindex(oos.signal_timestamp)
    # Label returns are in percentage points, so two basis points cost 0.02.
    net = np.sign(score.to_numpy()) * oos.return_pct.to_numpy() - ROUND_TRIP_COST_BPS / 100
    if not len(net) or not np.isfinite(net).all():
        raise RuntimeError("No finite held-out toy backtest returns")
    digest = hashlib.sha256(db.read_bytes()).digest()
    read = invoke(root, "read-back", "read-bar-evaluations", "--results-db", str(db),
                  "--factor-name", "equal_weight")
    dashboard = invoke(root, "dashboard-plan", "dashboard", "--results-db", str(db), "--dry-run")
    if (read["metrics"]["run_count"] != 1 or not dashboard["plan"]["read_only"]
            or digest != hashlib.sha256(db.read_bytes()).digest()):
        raise RuntimeError("Unexpected read-back or dashboard plan")

    report = {
        "precondition": {"ok": invalid["ok"], "exit_code": invalid["exit_code"], "error": invalid["error"]},
        "candidates": screened, "trial_family": {**family.report(), "selection_window": split},
        "synthesis": {"factors": eligible, "results_db_status": metrics["results_db"]["status"]},
        "toy_backtest": {"oos_trades": len(net), "mean_net_return_pct": round(float(net.mean()), 8)},
        "read_back": {"run_count": read["metrics"]["run_count"], "dashboard_read_only": dashboard["plan"]["read_only"]},
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
