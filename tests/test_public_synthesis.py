"""Public entrypoints use only synthetic inputs and isolated outputs."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from examples.synthesis_demo import METHODS, run_demo, synthetic_panel
from tools.demo_cli import main


@pytest.mark.parametrize("method", METHODS)
def test_synthesis_pipeline_produces_predictions(method, tmp_path):
    output = tmp_path / method
    if method != "weighted_sum":
        from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
        with pytest.raises(LabelIsolationError, match="same-row artificial target"):
            run_demo(method, output)
        assert not output.exists()
        return
    summary = run_demo(method, output)
    scores = pd.read_csv(output / "scores.csv")
    assert summary["rows"] == len(scores) > 0
    assert np.isfinite(scores.score).all()
    expected_codes = set(synthetic_panel()[0].index.get_level_values("stock_code"))
    assert set(scores.stock_code) == expected_codes
    assert not scores.duplicated(["date", "stock_code"]).any()
    if method != "weighted_sum":
        assert pd.to_datetime(scores.date).min() > pd.Timestamp(summary["training_end"])
    else:
        X, _ = synthetic_panel()
        shifted = X.unstack("stock_code").shift().stack("stock_code", future_stack=True).dropna()
        ranked = shifted.groupby(level="date").rank(pct=True)
        expected = ranked.groupby(level="date").transform(lambda s: (s - s.mean()) / s.std()).mean(axis=1)
        actual = scores.assign(date=pd.to_datetime(scores.date)).set_index(["date", "stock_code"]).score
        pd.testing.assert_series_equal(actual, expected, check_names=False)
    before = (output / "scores.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_demo(method, output)
    assert (output / "scores.csv").read_bytes() == before


def test_public_synthesis_cli(tmp_path):
    target = tmp_path / "cli"
    assert main(["factor-synthesis-demo", "--method", "weighted_sum", "--output-dir", str(target)]) == 0
    assert json.loads((target / "summary.json").read_text())["data"] == "synthetic"


def test_unknown_method_does_not_create_output(tmp_path):
    with pytest.raises(ValueError):
        run_demo("unknown", tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()


def test_public_algorithm_packages_import():
    for module in ("core", "pipeline", "preselection", "qlib_adapter", "diagnostics", "weighting"):
        importlib.import_module("factor_synthesis." + module)


@pytest.mark.parametrize("method", [m for m in METHODS if m != "weighted_sum"])
def test_public_cli_rejects_unverified_artificial_label(method, tmp_path, capsys):
    target = tmp_path / method
    assert main(["factor-synthesis-demo", "--method", method, "--output-dir", str(target)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["fix"]
    error = captured.err
    assert "verified label exit context" in error
    assert "--method weighted_sum" in error
    assert "examples.local_factor_workflow" in error
    assert "not supervised training" in error
    assert "Traceback" not in error
    assert not target.exists()


def test_public_synthesis_cli_defaults_to_runnable_method(tmp_path):
    target = tmp_path / "default"
    assert main(["factor-synthesis-demo", "--output-dir", str(target)]) == 0
    assert json.loads((target / "summary.json").read_text())["method"] == "weighted_sum"


def test_public_synthesis_cli_explains_missing_optional_dependency(tmp_path, monkeypatch, capsys):
    from examples import synthesis_demo

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("scikit-learn is absent", name="sklearn")

    monkeypatch.setattr(synthesis_demo, "run_demo", missing)
    assert main(["factor-synthesis-demo", "--output-dir", str(tmp_path / "out")]) == 1
    captured = capsys.readouterr()
    error = json.loads(captured.out)["error"]
    assert error["code"] == "missing_optional_dependency" and error["retryable"] is False
    assert "jq-factor-lab[synthesis]" in error["fix"] and "jq-factor-lab[synthesis]" in captured.err


def test_synthesis_dry_run_checks_dependencies_and_does_not_write(tmp_path, monkeypatch, capsys):
    from tools import demo_cli

    monkeypatch.chdir(tmp_path)
    args = ["factor-synthesis-demo", "--dry-run", "--compact", "--output-dir", "out"]
    assert main(args) == 0
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    result = json.loads(output)
    assert result["dry_run"] and result["ok"] and result["exit_code"] == 0
    assert result["plan"]["method"] == "weighted_sum"
    assert result["plan"]["output_dir"] == str(tmp_path / "out")
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(demo_cli, "find_spec", lambda name: None)
    assert main(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["code"] == "missing_optional_dependency"
    assert list(tmp_path.iterdir()) == []
