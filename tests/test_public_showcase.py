"""Regression for the installed-package research walkthrough's trial family."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_showcase_uses_one_preregistered_trial_family(tmp_path):
    pytest.importorskip("streamlit")
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "examples.ai_workflow_showcase"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    root, = tmp_path.glob("jq-ai-workflow-*")
    preregistration = json.loads((root / "preregistration.json").read_text())
    registered = [(row["factor_id"], row["horizon"]) for row in preregistration["candidates"]]
    assert len(registered) == 5
    recorded = []
    for name, horizon in registered:
        config = json.loads((root / f"{name}-config.json").read_text())
        assert config["split"] == preregistration["split"]
        payload = json.loads((root / f"{name}-evaluation.json").read_text())
        train = payload["metrics"]["splits"]["train"]
        assert all(datetime.fromisoformat(train[key]) == datetime.fromisoformat(bound)
                   for key, bound in config["split"].items())
        entry, = payload["metrics"]["trial_ledger"]["candidates"]
        assert (entry["factor_id"], entry["horizon"]) == (name, horizon)
        assert entry["p_value"] is None
        recorded.append(entry)

    short = report["candidates"]["short_sample"]
    assert short["sample_count"] == 8 and short["calibration_status"] == "uncalibrated"
    assert short["eligible_for_toy_backtest"] is False
    family = report["trial_family"]
    assert family["method"] == "holm" and family["attempt_count"] == len(registered)
    assert family["selection_window"] == preregistration["split"]
    assert family["pending_candidates"] == []
    assert family["missing_p_value_candidates"] == [list(key) for key in registered]
    assert family["selection_complete"] is False and family["best"] is None
    assert family["candidates"] == recorded
