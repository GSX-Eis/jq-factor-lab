import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.generate_data import make_prices
from examples.momentum import DemoMomentum
from tools.demo_cli import analyze, main, read_prices
from utils.pit import apply_pit_adjustment
from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator


def test_full_public_demo_and_no_output_overwrite(tmp_path):
    frame = make_prices(sessions=80)
    frame.to_csv(tmp_path / "input.csv", index=False)
    config = {
        "input_csv": "input.csv", "output_dir": "result",
        "calendar_file": str(Path(__file__).resolve().parents[1] / "config/demo_calendar.json"),
        "factor_file": str(Path(__file__).resolve().parents[1] / "examples/momentum.py"),
        "factor_class": "DemoMomentum", "periods": [1, 5], "execution_lag": 1,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = analyze(path)
    assert result["assets"] == 20
    assert result["ic_observations"]["5D"] == 69
    assert np.isfinite(list(result["ic_mean"].values())).all()
    assert (tmp_path / "result/summary.json").is_file()
    with pytest.raises(FileExistsError):
        analyze(path)


@pytest.mark.parametrize("issue", ["duplicate", "code", "price", "date", "factor"])
def test_csv_rejects_invalid_input(tmp_path, issue):
    frame = make_prices(sessions=2, assets=2)
    if issue == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif issue == "code":
        frame.loc[0, "stock_code"] = "X';DROP TABLE prices;--"
    elif issue == "price":
        frame.loc[0, "close"] = np.nan
    elif issue == "factor":
        frame.loc[0, "factor"] = 0
    else:
        frame.loc[0, "date"] = pd.NaT
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_prices(path)


def test_lagged_returns_have_exact_entry_and_exit():
    dates = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10., 20., 30., 40., 50.]}, index=dates).stack()
    prices.index.names = ["date", "asset"]
    from utils.trading_calendar import TradingCalendar
    result = ForwardReturnsCalculator(trading_calendar=TradingCalendar(dates, dates.min(), dates.max())).calculate_forward_returns(prices, (2,), execution_lag=1)
    assert result.loc[(dates[0], "A"), "2D"] == pytest.approx(1.0)
    assert result.index.get_level_values("date").max() == dates[1]


def test_pit_adjustment_uses_window_anchor_and_rejects_invalid_factor():
    frame = pd.DataFrame({"code": ["A", "A"], "date": pd.date_range("2020-01-01", periods=2),
                          "close": [100., 50.], "volume": [10., 20.], "factor": [1., 2.]})
    result = apply_pit_adjustment(frame)
    assert result.close.tolist() == [50., 50.]
    assert result.volume.tolist() == [20., 20.]
    frame.loc[0, "factor"] = 0
    with pytest.raises(ValueError):
        apply_pit_adjustment(frame)


def test_demo_signal_does_not_read_future_rows():
    frame = make_prices(sessions=20, assets=2).rename(columns={"stock_code": "code"})
    factor = object.__new__(DemoMomentum)
    full = factor.calculate_factor(frame)
    prefix = factor.calculate_factor(frame[frame.date <= full.index[10]])
    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)


def test_cli_rejects_constant_factor_without_publishing_results(tmp_path, capsys):
    make_prices(sessions=80).to_csv(tmp_path / 'input.csv', index=False)
    factor_file = tmp_path / 'constant.py'
    factor_file.write_text(
        'from examples.momentum import DemoMomentum\n'
        'class ConstantSignal(DemoMomentum):\n'
        '    def calculate_factor(self, data):\n'
        '        return data.pivot(index="date", columns="code", values="close") * 0 + 1\n'
    )
    config = tmp_path / 'constant.json'
    config.write_text(json.dumps({
        'input_csv': 'input.csv', 'output_dir': 'result',
        'calendar_file': str(Path(__file__).resolve().parents[1] / 'config/demo_calendar.json'),
        'factor_file': str(factor_file), 'factor_class': 'ConstantSignal',
        'periods': [1, 5], 'execution_lag': 1,
    }))
    assert main(['single-factor-analysis', '--config', str(config)]) == 1
    error = assert_cli_error(capsys, 1, "analysis_failed")
    assert "max_loss exceeded: loss=1, max_loss=0.35, retained=0/1600" in error["message"]
    assert not (tmp_path / 'result').exists()


@pytest.mark.parametrize('payload, expected', [
    ([], 'JSON object'),
    ({'input_csv': 'a', 'factor_file': 'b', 'factor_class': 'C', 'output_dir': 'out', 'periods': 1}, 'periods'),
    ({'input_csv': 123, 'factor_file': 'b', 'factor_class': 'C', 'output_dir': 'out'}, 'input_csv'),
    ({'input_csv': 'a', 'factor_file': 'b', 'output_dir': 'out'}, 'factor_class'),
])
def test_cli_config_errors_are_actionable_without_tracebacks(tmp_path, capsys, payload, expected):
    config = tmp_path/'bad.json'
    config.write_text(json.dumps(payload))
    assert main(['single-factor-analysis', '--config', str(config)]) == 2
    assert expected in assert_cli_error(capsys, 2, "invalid_input")["message"]


def test_cli_stdout_is_one_json_document(tmp_path, monkeypatch):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str(root))
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    make_prices(sessions=80).to_csv(tmp_path / "input.csv", index=False)
    (tmp_path / "noisy.py").write_text(
        "import ctypes\nfrom examples.momentum import DemoMomentum\n"
        "print('user module log')\nctypes.CDLL(None).puts(b'buffered native log')\n"
        "class NoisyMomentum(DemoMomentum):\n    pass\n"
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "input_csv": "input.csv", "factor_file": "noisy.py", "factor_class": "NoisyMomentum",
        "calendar_file": str(root / "config/demo_calendar.json"), "output_dir": "out", "periods": [1, 5],
    }))
    process = subprocess.run(
        [sys.executable, "-m", "tools.demo_cli", "single-factor-analysis", "--config", str(config)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["ok"] is True and payload["exit_code"] == 0
    assert payload["factor"] == "NoisyMomentum" and payload["next_actions"]
    assert "user module log" in process.stderr
    assert "buffered native log" in process.stderr


def test_generate_existing_file_has_short_error(tmp_path):
    import subprocess
    import sys
    path = tmp_path/'existing.csv'
    path.write_text('keep me')
    result = subprocess.run([sys.executable, '-m', 'examples.generate_data', '--output', str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'choose a new writable file path' in result.stderr and 'Traceback' not in result.stderr
    assert path.read_text() == 'keep me'


def test_cli_rejects_constant_returns_without_publishing_results(tmp_path, capsys):
    prices = make_prices(sessions=80)
    prices['close'] = prices['stock_code'].map(
        {code: float(i + 1) for i, code in enumerate(sorted(prices.stock_code.unique()))}
    )
    prices['factor'] = 1.0
    prices.to_csv(tmp_path / 'input.csv', index=False)
    factor_file = tmp_path / 'levels.py'
    factor_file.write_text(
        'from examples.momentum import DemoMomentum\n'
        'class LevelSignal(DemoMomentum):\n'
        '    def calculate_factor(self, data):\n'
        '        return data.pivot(index="date", columns="code", values="close")\n'
    )
    config = tmp_path / 'levels.json'
    config.write_text(json.dumps({
        'input_csv': 'input.csv', 'output_dir': 'result',
        'calendar_file': str(Path(__file__).resolve().parents[1] / 'config/demo_calendar.json'),
        'factor_file': str(factor_file), 'factor_class': 'LevelSignal',
        'periods': [1, 5], 'execution_lag': 1,
    }))
    assert main(['single-factor-analysis', '--config', str(config)]) == 1
    assert "no usable IC observations" in assert_cli_error(capsys, 1, "analysis_failed")["message"]
    assert not (tmp_path / 'result').exists()


def test_daily_demo_allocates_distinct_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    outputs = []
    for _ in range(2):
        assert main(["daily-demo"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["assets"] == 20
        output = Path(payload["output_dir"])
        assert (output / "summary.json").is_file()
        outputs.append(output)
    assert outputs[0] != outputs[1]
    assert (outputs[0] / "summary.json").is_file()


def test_daily_demo_explicit_overwrite_preserves_unrelated_files(tmp_path, capsys):
    output = tmp_path / "daily"
    args = ["daily-demo", "--output-dir", str(output)]
    assert main(args) == 0
    summary = output / "summary.json"
    before = summary.read_bytes()
    keep = output / "keep.txt"
    keep.write_text("keep")
    capsys.readouterr()
    assert main(args) == 2
    assert "--overwrite" in assert_cli_error(capsys, 2, "invalid_input")["message"]
    assert summary.read_bytes() == before
    summary.write_text("old")
    assert main([*args, "--overwrite"]) == 0
    assert summary.read_bytes() == before
    assert keep.read_text() == "keep"


def test_single_asset_has_structured_cross_section_error(tmp_path, capsys):
    make_prices(assets=1).to_csv(tmp_path / "one.csv", index=False)
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "input_csv": "one.csv", "factor_class": "DemoMomentum",
        "factor_file": str(root / "examples/momentum.py"),
        "calendar_file": str(root / "config/demo_calendar.json"),
    }))
    assert main(["single-factor-analysis", "--config", str(config)]) == 1
    error = assert_cli_error(capsys, 1, "insufficient_cross_section_assets")
    assert "time-series-analysis" in error["message"]
    assert "at least 2" in error["message"]
    assert not (tmp_path / "outputs").exists()


def test_bar_demo_reruns_and_reads_synthetic_database(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    output_dirs = []
    for _ in range(2):
        assert main(["bar-demo"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["metrics"]["analysis_mode"] == "time_series"
        assert payload["metrics"]["diagnostics_report"]
        output_dirs.append(Path(next(item["path"] for item in payload["artifacts"]
                                     if item["artifact_type"] == "evaluation_summary")).parent)
    assert output_dirs[0] != output_dirs[1]
    assert all((path / "summary.json").is_file() for path in output_dirs)
    database = Path(payload["metrics"]["results_db"]["path"]).relative_to(tmp_path)
    original = database.read_bytes()
    assert main(["read-bar-evaluations", "--results-db", str(database), "--factor-name", "factor_value", "--limit", "1"]) == 0
    read = json.loads(capsys.readouterr().out)
    assert read["ok"] and read["metrics"]["run_count"] == 1
    identity = read["metrics"]["runs"][0]["identity_hash"]
    assert {row["identity_hash"] for row in read["metrics"]["ic_by_horizon"]} == {identity}
    assert not read["artifacts"]
    assert main(["read-bar-evaluations", "--results-db", str(database), "--factor-name", "other"]) == 0
    assert json.loads(capsys.readouterr().out)["metrics"]["run_count"] == 0
    before = sorted(tmp_path.rglob("*"))
    assert main(["read-bar-evaluations", "--results-db", str(database), "--limit", "1", "--dry-run",
                 "--output-root", "reader", "--json-out", "read-plan.json"]) == 0
    read_plan = json.loads(capsys.readouterr().out)["plan"]
    assert read_plan["results_db"] == str(database.resolve())
    assert sorted(tmp_path.rglob("*")) == before
    assert database.read_bytes() == original


def test_bar_cli_requires_overwrite_for_explicit_existing_output(tmp_path, monkeypatch, capsys):
    from examples.synthetic_bars import write_demo_bars
    monkeypatch.chdir(tmp_path)
    config = write_demo_bars(tmp_path / "bars.csv")
    config.update(bars_csv="bars.csv", output_dir="evaluation")
    Path("bar.json").write_text(json.dumps(config))
    args = ["time-series-analysis", "--config", "bar.json"]
    assert main(args) == 0
    capsys.readouterr()
    summary = Path("evaluation/summary.json")
    summary.write_text("old")
    assert main(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "time_series_precondition_failed"
    assert "--overwrite" in payload["error"]["message"]
    assert summary.read_text() == "old"
    assert main([*args, "--overwrite"]) == 0
    assert json.loads(summary.read_text())["analysis_mode"] == "time_series"


@pytest.mark.parametrize("options", [[], ["--dry-run"]], ids=["execute", "dry-run"])
def test_bar_cli_rejects_an_explicit_run_directory_before_replacing_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    options: list[str],
) -> None:
    from examples.synthetic_bars import write_demo_bars
    monkeypatch.chdir(tmp_path)
    Path("config.json").write_text(json.dumps(write_demo_bars(Path("bars.csv"))))
    summary = Path("h/time-series-analysis/r/summary.json")
    summary.parent.mkdir(parents=True)
    marker = b'{"marker": "historical"}\r\n'
    summary.write_bytes(marker)

    args = ["time-series-analysis", "--config", "config.json", "--output-root", "h",
            "--run-id", "r", "--output-dir", str(summary.parent)]
    assert main([*args, *options]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "--overwrite" in payload["error"]["message"]
    assert summary.read_bytes() == marker
    assert not (summary.parent / "attempts").exists()
    if not options:
        assert main([*args, "--overwrite"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result.pop("next_actions")
        assert json.loads(summary.read_text()) == result


def test_bar_demo_keeps_scratch_and_cache_under_caller_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from utils import local_outputs

    monkeypatch.chdir(tmp_path)
    package = tmp_path / "package"
    monkeypatch.setattr(local_outputs, "__file__", str(package / "utils/local_outputs.py"))
    monkeypatch.delenv("TASK_MANAGER_ROOT", raising=False)
    assert main(["bar-demo"]) == 0
    result = json.loads(capsys.readouterr().out)
    summary = Path(next(item["path"] for item in result["artifacts"] if item["artifact_type"] == "summary"))
    bindings = json.loads(summary.with_name("output_bindings.json").read_text())
    harness_root = summary.parent.parents[3]
    assert Path(bindings["scratch"]).parent == harness_root / "tmp"
    assert Path(bindings["cache"]).parent == harness_root / "cache"
    assert not package.exists()


def test_bar_cli_config_database_resolves_from_caller_for_guard_write_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.synthetic_bars import write_demo_bars
    from tools.agent_harness import cli
    monkeypatch.chdir(tmp_path)
    config_path = Path("configs/research.json")
    config_path.parent.mkdir()
    config = write_demo_bars(Path("bars.csv"))
    config.update(output_dir="evaluation", results_db="evaluation/summary.json")
    config_path.write_text(json.dumps(config))
    args = ["time-series-analysis", "--config", str(config_path)]
    assert main(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "bar_evaluation_store_precondition_failed"
    assert "overlaps an output path" in payload["error"]["message"]
    assert not Path("evaluation").exists()

    config["results_db"] = "review-eval.duckdb"
    config_path.write_text(json.dumps(config))
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    database = tmp_path / config["results_db"]
    assert Path(result["metrics"]["results_db"]["path"]) == database
    original = database.read_bytes()
    assert main(["read-bar-evaluations", "--results-db", database.name]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert database.read_bytes() == original
    assert not (cli.REPO_ROOT / database.name).exists()

    config.update(factor_id="second_signal", output_dir="evaluation-2")
    config_path.write_text(json.dumps(config))
    assert main(args) == 0
    capsys.readouterr()
    read_args = ["read-bar-evaluations", "--results-db", database.name, "--limit", "1"]
    assert main(read_args) == 0
    latest = json.loads(capsys.readouterr().out)["metrics"]
    assert latest["run_count"] == 1
    assert Path(latest["runs"][0]["summary_path"]).parent == tmp_path / "evaluation-2"
    assert main([*read_args, "--factor-name", "factor_value"]) == 0
    matching = json.loads(capsys.readouterr().out)["metrics"]
    assert matching["run_count"] == 1
    assert Path(matching["runs"][0]["summary_path"]).parent == tmp_path / "evaluation"

    summary = Path(latest["runs"][0]["summary_path"])
    summary.write_text(summary.read_text() + "\n")
    assert main([*read_args, "--factor-name", "second_signal"]) == 0
    assert json.loads(capsys.readouterr().out)["metrics"]["run_count"] == 0


def assert_cli_error(capsys, code, error_code):
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["ok"] is False and payload["exit_code"] == code
    error = payload["error"]
    assert error["code"] == error_code
    assert error["message"] and error["fix"] and error["retryable"] is False
    assert len(output.err.splitlines()) == 1 and "Traceback" not in output.err
    return error


@pytest.mark.parametrize("args", [[], ["unknown"], ["single-factor-analysis"], ["read-bar-evaluations"]])
def test_argument_errors_are_json(args, capsys):
    assert main(["--compact", *args]) == 2
    assert "--help" in assert_cli_error(capsys, 2, "invalid_arguments")["fix"]


def test_help_lists_only_public_workflows(capsys):
    assert main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert all(command in help_text for command in ("daily-demo", "time-series-analysis", "dashboard"))
    assert "SML20" not in help_text and "Factor1" not in help_text
    assert main(["time-series-analysis", "--help"]) == 0
    assert "--config" in capsys.readouterr().out


def test_compact_output_replaces_nonfinite_metrics(monkeypatch, capsys):
    from tools import demo_cli

    monkeypatch.setattr(demo_cli, "_daily_demo", lambda *_args, **_kwargs: {
        "ic_mean": {"1D": float("nan"), "5D": np.float32("inf"), "10D": -float("inf")},
    })
    assert main(["daily-demo", "--compact"]) == 0
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert json.loads(output)["ic_mean"] == {"1D": None, "5D": None, "10D": None}


@pytest.mark.parametrize("command", ["daily-demo", "bar-demo"])
def test_demo_dry_run_checks_output_without_writing(command, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main([command, "--dry-run", "--output-dir", "new/nested"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["dry_run"] and result["exit_code"] == 0
    assert result["plan"]["output_dir"] == str(tmp_path / "new/nested")
    assert list(tmp_path.iterdir()) == []

    Path("existing").mkdir()
    assert main([command, "--dry-run", "--output-dir", "existing"]) == 2
    assert "--overwrite" in assert_cli_error(capsys, 2, "invalid_input")["message"]
    assert main([command, "--dry-run", "--output-dir", "existing", "--overwrite"]) == 0
    assert list(Path("existing").iterdir()) == []


def test_single_dry_run_validates_calendar_without_importing_factor(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    make_prices(sessions=8).to_csv("prices.csv", index=False)
    Path("trusted.py").write_text("raise AssertionError('dry-run imported factor')\n")
    config = {"factor_file": "trusted.py", "factor_class": "DemoMomentum", "input_csv": "prices.csv",
              "calendar_file": str(Path(__file__).resolve().parents[1] / "config/demo_calendar.json")}
    Path("config.json").write_text(json.dumps(config))
    before = sorted(tmp_path.iterdir())
    argv = ["single-factor-analysis", "--dry-run", "--config", "config.json", "--output-dir", "out"]
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["calendar_sha256"] and plan["assets"] == 20
    assert sorted(tmp_path.iterdir()) == before

    prices = pd.read_csv("prices.csv").assign(date=lambda frame: frame.date.str.replace("2020", "2040"))
    prices.to_csv("prices.csv", index=False)
    assert main(argv) == 2
    assert "calendar" in assert_cli_error(capsys, 2, "invalid_input")["message"].lower()
    assert sorted(tmp_path.iterdir()) == before


def test_bar_analysis_dry_run_validates_inputs_without_writing(tmp_path, monkeypatch, capsys):
    from examples.synthetic_bars import write_demo_bars

    monkeypatch.chdir(tmp_path)
    config = write_demo_bars(Path("bars.csv"))
    config.update(direction=-1, bootstrap={"n_boot": 1, "block_length": 5})
    Path("config.json").write_text(json.dumps(config))
    argv = ["time-series-analysis", "--config", "config.json", "--results-db", "db.duckdb",
            "--output-root", "scratch", "--json-out", "plan.json", "--dry-run"]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["plan"]["horizons"] == [1, 2, 5]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bars.csv", "config.json"]
    Path("bars.csv").unlink()
    assert main(argv) == 2
    assert_cli_error(capsys, 2, "time_series_precondition_failed")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["config.json"]


def test_bar_analysis_dry_run_rejects_invalid_evaluator_options(tmp_path, monkeypatch, capsys):
    from examples.synthetic_bars import write_demo_bars

    monkeypatch.chdir(tmp_path)
    config = write_demo_bars(Path("bars.csv"))
    config["n_quantiles"] = 0
    Path("config.json").write_text(json.dumps(config))
    argv = ["time-series-analysis", "--config", "config.json", "--output-dir", "evaluation",
            "--output-root", "scratch", "--json-out", "plan.json", "--dry-run"]
    assert main(argv) == 2
    error = assert_cli_error(capsys, 2, "time_series_precondition_failed")
    assert "n_quantiles must be an integer >= 2" in error["message"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bars.csv", "config.json"]


@pytest.mark.parametrize(("options", "message"), [
    ({"direction": 0}, "direction must be -1 or 1"),
    ({"bootstrap": {"n_boot": 0}}, "n_boot must be an integer >= 1"),
    ({"bootstrap": {"block_length": 2}}, "block_length must be >= horizon (5)"),
    ({"bootstrap": {"alpha": 1}}, "alpha must be in (0, 1)"),
    ({"bootstrap": {"seed": -1}}, "seed must be an integer >= 0"),
    ({"bootstrap": {"min_valid_replicates": 1}}, "min_valid_replicates must be an integer >= 2"),
    ({"analysis_mode": "cross_sectional", "hac_lag": -1}, "hac_lag must be an integer >= 0"),
    ({"split": {"train_end": "2026-12-01T14:40Z", "oos_start": "2026-12-01T14:50Z"},
      "standardization": "invalid"}, "standardization must be None, 'zscore', or 'median_mad'"),
    ({"split": {"train_end": "2026-12-01T14:50Z", "oos_start": "2026-12-01T14:40Z"}},
     "Expected train_end < oos_start"),
], ids=["direction", "n-boot", "block-length", "bootstrap-alpha", "bootstrap-seed",
        "min-valid-replicates", "hac-lag", "standardization", "split-order"])
def test_bar_analysis_dry_run_rejects_invalid_inference_options(
    options, message, tmp_path, monkeypatch, capsys,
):
    from examples.synthetic_bars import write_demo_bars

    monkeypatch.chdir(tmp_path)
    config = write_demo_bars(Path("bars.csv"))
    config.update(options)
    Path("config.json").write_text(json.dumps(config))
    argv = ["time-series-analysis", "--config", "config.json", "--output-dir", "evaluation",
            "--output-root", "scratch", "--json-out", "plan.json"]
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    dry_exit = main([*argv, "--dry-run"])
    dry_payload = json.loads(capsys.readouterr().out)
    assert set(tmp_path.rglob("*")) == before.keys()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert main(argv) == 2
    execution_error = assert_cli_error(capsys, 2, "time_series_precondition_failed")
    assert execution_error["message"] == message
    assert dry_exit == dry_payload["exit_code"] == 2
    assert dry_payload["ok"] is False
    assert dry_payload["error"]["code"] == execution_error["code"]
    assert dry_payload["error"]["message"] == execution_error["message"]


@pytest.mark.parametrize(("output", "message"), [
    ("bars.csv", "must be a directory"),
    ("blocked/child", "not writable"),
], ids=["ordinary-file", "unwritable-parent"])
def test_bar_analysis_dry_run_rejects_unusable_output(output, message, tmp_path, monkeypatch, capsys):
    from examples.synthetic_bars import write_demo_bars

    monkeypatch.chdir(tmp_path)
    Path("config.json").write_text(json.dumps(write_demo_bars(Path("bars.csv"))))
    original_bars = Path("bars.csv").read_bytes()
    blocked = Path("blocked")
    blocked.mkdir(mode=0o500)
    argv = ["time-series-analysis", "--config", "config.json", "--output-dir", output, "--overwrite",
            "--output-root", "scratch", "--json-out", "plan.json", "--dry-run"]
    try:
        assert main(argv) == 2
        error = assert_cli_error(capsys, 2, "time_series_precondition_failed")
        assert message in error["message"]
        assert list(blocked.iterdir()) == []
        assert Path("bars.csv").read_bytes() == original_bars
        assert sorted(path.name for path in tmp_path.iterdir()) == ["bars.csv", "blocked", "config.json"]
    finally:
        blocked.chmod(0o700)


@pytest.mark.parametrize(("database_exists", "filters", "message"), [
    (False, [], "results_db does not exist"),
    (True, ["--limit", "0"], "limit must be a positive integer"),
    (True, ["--factor-name", " "], "factor_name must be nonempty"),
    (True, ["--start-utc", "2026-02-02", "--end-utc", "2026-02-01"], "start_utc must not follow end_utc"),
], ids=["missing-db", "invalid-limit", "blank-factor", "reversed-dates"])
def test_bar_read_dry_run_rejects_invalid_inputs(
    database_exists, filters, message, tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(tmp_path)
    database = Path("missing.duckdb")
    if database_exists:
        assert main(["bar-demo"]) == 0
        database = Path(json.loads(capsys.readouterr().out)["metrics"]["results_db"]["path"])
    before = sorted(tmp_path.rglob("*"))
    original = database.read_bytes() if database_exists else None
    argv = ["read-bar-evaluations", "--results-db", str(database), *filters,
            "--output-root", "reader", "--json-out", "plan.json", "--dry-run"]
    assert main(argv) == 2
    error = assert_cli_error(capsys, 2, "bar_evaluation_read_precondition_failed")
    assert message in error["message"]
    assert sorted(tmp_path.rglob("*")) == before
    if database_exists:
        assert database.read_bytes() == original
    else:
        assert not database.exists()


def test_dashboard_dry_run_uses_env_and_checks_port_and_dependency(tmp_path, monkeypatch, capsys):
    from tools import demo_cli

    monkeypatch.chdir(tmp_path)
    db = tmp_path / "results.duckdb"
    db.write_bytes(b"fixture")
    monkeypatch.setenv("QUANT_BAR_RESULTS_DB", str(db))
    monkeypatch.setattr(demo_cli.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("dashboard launched"))
    monkeypatch.setattr(demo_cli, "find_spec", lambda name: None)
    assert main(["dashboard", "--dry-run"]) == 1
    assert assert_cli_error(capsys, 1, "missing_optional_dependency")["fix"] == (
        "python3 -m pip install 'jq-factor-lab[viz]'"
    )

    monkeypatch.setattr(demo_cli, "find_spec", lambda name: object())
    assert main(["dashboard", "--dry-run", "--port", "8877"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["port"] == 8877 and plan["read_only"] is True and plan["results_db"] == str(db)
    assert plan["command"][-2:] == ["--results-db", str(db)]
    assert main(["dashboard", "--dry-run", "--results-db", "missing"]) == 2
    assert "missing" in assert_cli_error(capsys, 2, "invalid_input")["message"]
    assert main(["dashboard", "--dry-run", "--port", "65536"]) == 2
    assert "--port" in assert_cli_error(capsys, 2, "invalid_input")["message"]
    assert db.read_bytes() == b"fixture"
    assert list(tmp_path.iterdir()) == [db]
