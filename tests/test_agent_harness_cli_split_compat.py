import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.agent_harness import cli


def test_cli_legacy_imports() -> None:
    callables = (
        "main build_parser HarnessResult "
        "_utc_now _tail _write_json _safe_read_json _resolve_repo_relative_path "
        "_bound_result_metrics _output_dir _success_result _failure_result _complete_result "
        "_fake_result _kill_process_group _build_extra_metrics _classify_subprocess_failure "
        "_run_subprocess _run_subprocess_body _resolve_args_output_root _common_artifacts "
        "_strip_log_prefix _extract_neutralized_input_warnings _pct_to_ratio _coerce_signature_float "
        "_build_backtest_signature _parse_factor_synthesis_ic_summary "
        "_parse_factor_synthesis_backtest_runs _parse_factor_synthesis_diagnostics "
        "_parse_vectorbt_result_summaries _parse_benchmark_reports _compact_benchmark_report "
        "_assemble_factor_synthesis_metrics _extract_factor_synthesis_metrics "
        "_extract_symbolic_factor_mining_metrics "
        "_bar_artifact_paths _validate_bar_output_paths _parse_strict_utc_csv_column _duckdb_count_tables "
        "_handle_time_series_analysis _handle_read_bar_evaluations _handle_single_factor_analysis "
        "_handle_symbolic_factor_mining _handle_quanta_alpha_realize _handle_factor_synthesis "
        "_handle_parent_factor_screen _handle_monitor_refresh _handle_alignment_run _handle_bar_storage _handle_factor_values_store "
        "config_sha256 read_execution_result D9CapacityAuditError build_capacity_audit "
        "load_capacity_audit_context preflight_capacity_audit write_capacity_audit "
        "dataclass datetime timezone Path Dict List Mapping Sequence"
    ).split()
    for name in callables:
        assert callable(getattr(cli, name)), name
    other_bindings = (
        "Any argparse json math os re signal sys traceback uuid annotations "
        "_LOG_PREFIX_RE _IC_PERIOD_RE _IC_OVERALL_RE _IR_PERIOD_RE _IR_OVERALL_RE "
        "_SYNTHESIS_DIAGNOSTICS_RE _VECTORBT_RESULT_SUMMARIES_RE _BENCHMARK_REPORT_RE "
        "_BENCHMARK_DETAIL_KEYS _BACKTEST_HEADER_RE _BACKTEST_METRIC_RE _ANNUAL_RETURN_RE "
        "_NEUTRALIZED_INPUT_WARNING_TOKEN"
    ).split()
    for name in other_bindings:
        assert name in vars(cli), name
    assert cli.subprocess is subprocess
    assert (cli.EXIT_SUCCESS, cli.EXIT_PRECONDITION, cli.EXIT_EXECUTION, cli.EXIT_QUALITY_GATE) == (0, 2, 3, 4)
    assert cli.REPO_ROOT == Path(cli.__file__).resolve().parents[2]
    assert cli.DEFAULT_OUTPUT_ROOT == cli.REPO_ROOT / "outputs" / "agent_harness"
    assert cli.DEFAULT_MONITOR_DB == cli.REPO_ROOT / "data" / "duckdb" / "analysis_results.duckdb"


def test_cli_subcommand_inventory() -> None:
    parser = cli.build_parser()
    commands = next(action.choices for action in parser._actions if action.dest == "command")
    assert list(commands) == [
        "outputs-gc", "market-db", "build-bars", "audit-bars", "import-bars", "build-roll-schedule",
        "write-factor-values", "read-factor-values", "verify-factor-values",
        "export-strategy", "verify-target-alignment", "alignment-run",
        "single-factor-analysis", "time-series-analysis", "read-bar-evaluations",
        "symbolic-factor-mining", "quanta-alpha-realize", "factor-synthesis", "parent-factor-screen",
        "monitor-refresh",
    ]


@pytest.mark.parametrize("command", ["time-series-analysis", "single-factor-analysis"])
def test_cli_rejects_retired_bar_sidecar_flag(command, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args([command, "--evaluation-v2"])
    assert exit_info.value.code == 2
    assert "unrecognized arguments: --evaluation-v2" in capsys.readouterr().err


def test_cli_monitor_defaults_and_result_hooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "monitor.duckdb"
    db.touch()
    output_root = tmp_path / "reports"
    ts = "2026-01-01T00:00:00+00:00"
    count = Mock(return_value={"available": True, "table_count": 17})
    complete = Mock(side_effect=lambda result, **_: result)
    monkeypatch.delenv("QUANTATION_OUTPUTS_DIR", raising=False)
    monkeypatch.delenv("QUANT_DATA_BINDINGS", raising=False)
    monkeypatch.setattr(cli, "DEFAULT_MONITOR_DB", db)
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_utc_now", lambda: ts)
    monkeypatch.setattr(cli, "_duckdb_count_tables", count)
    monkeypatch.setattr(cli, "_complete_result", complete)

    assert cli.build_parser().parse_args(["outputs-gc"]).task_root == str(tmp_path)
    assert cli.main(["monitor-refresh", "--run-id", "split"]) == cli.EXIT_SUCCESS
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["database_path"] == str(db)
    assert payload["metrics"]["table_count"] == 17
    assert payload["summary"]["started_at"] == payload["summary"]["finished_at"] == ts
    assert Path(payload["summary"]["output_dir"]).is_relative_to(output_root / "monitor-refresh" / "split")
    count.assert_called_once_with(db, bindings={})
    complete.assert_called_once()


def test_cli_single_factor_uses_patched_subprocess_and_bound_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    output = tmp_path / "attempt"
    bound_metrics = Mock(return_value={"bound": True})

    def run_subprocess(**kwargs) -> dict:
        assert kwargs["command"][1] == str(tmp_path / "run_analysis_main.py")
        assert kwargs["output_dir"] == output
        return kwargs["success_metrics_builder"]("stdout", "stderr")

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_output_dir", lambda *args: output)
    monkeypatch.setattr(cli, "_run_subprocess", run_subprocess)
    monkeypatch.setattr(cli, "_bound_result_metrics", bound_metrics)
    args = cli.build_parser().parse_args(["single-factor-analysis", "--config", cfg.name])

    assert cli._handle_single_factor_analysis(args) == {"bound": True}
    bound_metrics.assert_called_once()
    assert bound_metrics.call_args.kwargs["source_sha256"] == cli.config_sha256(cfg)


@pytest.mark.parametrize("replace_binding", [False, True], ids=["patch-run", "replace-module"])
def test_cli_bar_worker_uses_patched_subprocess_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_binding: bool,
) -> None:
    attempt = tmp_path / "forced-attempt"
    calls = []

    def run_worker(command, *, cwd, **kwargs) -> subprocess.CompletedProcess:
        assert command[2:4] == ["tools.agent_harness.time_series_analysis", str(attempt / "config" / "resolved.json")]
        assert cwd == tmp_path
        calls.append(command)
        payload = {"ok": True, "metrics": {"worker": "patched"}, "artifacts": []}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_output_dir", lambda *args: attempt)
    monkeypatch.setenv("FACTORPILOT_ROOT", str(tmp_path))

    original_run = subprocess.run

    def run_with_worker_probe(command, *args, **kwargs) -> subprocess.CompletedProcess:
        if not isinstance(command, (list, tuple)) or tuple(command[1:3]) != (
            "-m", "tools.agent_harness.time_series_analysis",
        ):
            return original_run(command, *args, **kwargs)
        if replace_binding:
            raise AssertionError("Bar worker bypassed the CLI subprocess binding")
        return run_worker(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_with_worker_probe)
    if replace_binding:
        monkeypatch.setattr(cli, "subprocess", SimpleNamespace(run=run_worker))
    args = cli.build_parser().parse_args(["time-series-analysis"])

    result = cli._handle_time_series_analysis(args)
    assert len(calls) == 1
    assert result.output_dir == attempt
    assert result.metrics == {"worker": "patched"}
    assert json.loads((attempt / "summary.json").read_text()) == result.to_dict()


def test_cli_runner_uses_replaced_subprocess_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class ProbeError(Exception):
        pass

    pipe = object()
    popen = Mock(side_effect=ProbeError("façade Popen called"))
    original_popen = Mock(side_effect=AssertionError("unexpected call to stdlib subprocess.Popen"))
    monkeypatch.setattr(subprocess, "Popen", original_popen)
    monkeypatch.setattr(cli, "subprocess", SimpleNamespace(Popen=popen, PIPE=pipe))
    command = [sys.executable, "-c", "print('child-ran')"]

    with pytest.raises(ProbeError, match="Popen called"):
        cli._run_subprocess_body(
            action="split-probe",
            output_dir=tmp_path / "attempt",
            run_id="split",
            command=command,
            timeout_sec=10,
            started_at=cli._utc_now(),
            artifacts=[],
            json_out="",
        )
    popen.assert_called_once_with(
        command, cwd=str(cli.REPO_ROOT), env=None, pass_fds=(),
        stdout=pipe, stderr=pipe, text=True, start_new_session=True,
    )
    original_popen.assert_not_called()


def test_cli_runner_uses_patched_cwd_and_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    complete = Mock(side_effect=lambda result, **_: result)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_complete_result", complete)

    result = cli._run_subprocess_body(
        action="split-probe",
        output_dir=tmp_path / "attempt",
        run_id="split",
        command=[sys.executable, "-c", "from pathlib import Path; print(Path.cwd())"],
        timeout_sec=10,
        started_at=cli._utc_now(),
        artifacts=[],
        json_out="",
    )
    assert result.exit_code == cli.EXIT_SUCCESS
    assert result.stdout_tail.strip() == str(tmp_path)
    complete.assert_called_once_with(result, json_out="")


def test_cli_fake_result_and_handler_dispatch_are_late_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_result(**kwargs) -> cli.HarnessResult:
        return cli._success_result(
            **{key: kwargs[key] for key in ("action", "output_dir", "run_id", "started_at", "command")},
            mode="fake", metrics={"patched": kwargs["fake_result"]}, artifacts=[],
        )

    command = ["single-factor-analysis", "--output-root", str(tmp_path)]
    fake = Mock(side_effect=fake_result)
    monkeypatch.setattr(cli, "_fake_result", fake)
    assert cli.main([*command, "--fake-result", "success"]) == cli.EXIT_SUCCESS
    assert json.loads(capsys.readouterr().out)["metrics"] == {"patched": "success"}
    fake.assert_called_once()

    handler = Mock(return_value=fake_result(**fake.call_args.kwargs))
    monkeypatch.setattr(cli, "_handle_single_factor_analysis", handler)
    assert cli.main(command) == cli.EXIT_SUCCESS
    handler.assert_called_once_with(cli.build_parser().parse_args(command))


def test_cli_split_modules_can_run_without_loading_facade(tmp_path: Path) -> None:
    script = """
import sys
from tools.agent_harness import metrics_parsers, parser, runner
from tools.agent_harness.handlers import analysis

args = parser.build_parser().parse_args(["single-factor-analysis", "--fake-result", "success",
                                        "--output-root", sys.argv[1]])
assert analysis._handle_single_factor_analysis(args).ok
assert runner._resolve_repo_relative_path("relative") == runner.REPO_ROOT / "relative"
assert isinstance(metrics_parsers._extract_factor_synthesis_metrics("", "", "/absent"), dict)
assert "tools.agent_harness.cli" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=cli.REPO_ROOT, check=True)
