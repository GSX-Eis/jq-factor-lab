from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_harness.cli", *args],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=30,
    )


def _payload(proc: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(proc.stdout)


def test_dry_run_shape_for_all_commands(tmp_path: Path) -> None:
    commands = [
        ["single-factor-analysis", "--dry-run", "--factor", "factors/exp/example.py"],
        ["factor-synthesis", "--dry-run", "--config", "missing.json"],
        ["monitor-refresh", "--dry-run"],
    ]
    required_top_level = {
        "ok",
        "command",
        "summary",
        "artifacts",
        "metrics",
        "warnings",
        "exit_code",
    }

    for idx, command in enumerate(commands):
        json_out = tmp_path / f"dry_run_{idx}.json"
        proc = _run(*command, "--json-out", str(json_out))
        assert proc.returncode == 0, proc.stderr
        payload = _payload(proc)
        assert required_top_level.issubset(payload.keys())
        assert payload["ok"] is True
        assert payload["summary"]["reason"] == "dry_run"
        assert json_out.exists()
        mirrored = json.loads(json_out.read_text(encoding="utf-8"))
        assert mirrored["command"]["name"] == payload["command"]["name"]


def test_symbolic_quality_gate_failure_preserves_explicit_reason(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from tools.agent_harness import cli as harness_cli

    reason = (
        "post_rerank_quality_gate_failed: "
        "no final candidate passed lightweight/QD gates; "
        "best_miss=rank=1 period=10 rank_ic=0.077250"
    )
    payload = {
        "ok": False,
        "exit_code": harness_cli.EXIT_QUALITY_GATE,
        "run_id": "sym_fail",
        "failure_reason": reason,
        "post_rerank": {
            "enabled": True,
            "required": True,
            "dual_gate_passed": False,
            "selected_reason": "fallback_search_best_no_dual_gate_pass",
            "quality_gate_failure_reason": reason,
            "top_k": 9,
            "qd_bucket_count": 0,
        },
        "validation": {"ok": True},
    }
    command = [
        sys.executable,
        "-c",
        "import sys; print(sys.argv[1]); sys.exit(4)",
        json.dumps(payload),
    ]

    result = harness_cli._run_subprocess(
        action="symbolic-factor-mining",
        output_dir=tmp_path / "out",
        run_id="sym_fail",
        command=command,
        timeout_sec=30,
        started_at=harness_cli._utc_now(),
        artifacts=[],
        json_out="",
        success_metrics_builder=harness_cli._extract_symbolic_factor_mining_metrics,
    )

    assert result.ok is False
    assert result.exit_code == harness_cli.EXIT_QUALITY_GATE
    assert result.reason == "post_rerank_quality_gate_failed"
    assert result.metrics["failure_reason"] == reason
    assert result.metrics["post_rerank"]["dual_gate_passed"] is False
    assert reason in result.stderr_tail




def test_fake_result_is_machine_readable() -> None:
    proc = _run("single-factor-analysis", "--fake-result", "success")
    payload = _payload(proc)
    assert proc.returncode == 0
    assert payload["ok"] is True
    assert payload["command"]["name"] == "single-factor-analysis"


def test_single_factor_materialize_only_is_forwarded() -> None:
    proc = _run(
        "single-factor-analysis",
        "--dry-run",
        "--config",
        "config/dynamic.json",
        "--factor",
        "factors/exp/example.py",
        "--materialize-only",
    )

    assert proc.returncode == 0, proc.stderr
    payload = _payload(proc)
    assert payload["command"]["argv"][-1] == "--materialize-only"
    assert payload["metrics"]["planned_materialize_only"] is True


def test_single_factor_materialize_only_requires_explicit_factor() -> None:
    proc = _run(
        "single-factor-analysis",
        "--dry-run",
        "--config",
        "config/dynamic.json",
        "--materialize-only",
    )

    assert proc.returncode == 2
    assert "--materialize-only requires --factor" in proc.stderr


def test_single_factor_mining_only_is_forwarded() -> None:
    proc = _run(
        "single-factor-analysis",
        "--dry-run",
        "--factor",
        "factors/exp/example.py",
        "--mining-only",
    )

    assert proc.returncode == 0, proc.stderr
    assert "--mining-only" in _payload(proc)["command"]["argv"]


def test_factor_synthesis_resolves_relative_config_from_repo_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from tools.agent_harness import cli as harness_cli

    repo_root = tmp_path / "execution_repo"
    config_path = repo_root / "config" / "test_factor_synthesis.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(harness_cli, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        harness_cli, "DEFAULT_OUTPUT_ROOT", repo_root / "outputs" / "agent_harness"
    )
    monkeypatch.setattr(
        harness_cli,
        "DEFAULT_MONITOR_DB",
        repo_root / "data" / "duckdb" / "analysis_results.duckdb",
    )

    captured: dict[str, object] = {}

    def _fake_run_subprocess(**kwargs):
        captured.update(kwargs)
        return harness_cli.HarnessResult(
            action="factor-synthesis",
            ok=True,
            exit_code=harness_cli.EXIT_SUCCESS,
            status="succeeded",
            reason="ok",
            mode="subprocess",
            output_dir=Path(kwargs["output_dir"]),
            command=list(kwargs["command"]),
            metrics={},
            artifacts=[],
            warnings=[],
            run_id="",
        )

    monkeypatch.setattr(harness_cli, "_run_subprocess", _fake_run_subprocess)

    args = argparse.Namespace(
        config="config/test_factor_synthesis.json",
        output_root="",
        run_id="",
        fake_result="",
        dry_run=False,
        json_out="",
        timeout_sec=30,
        return_series_output_dir="",
    )

    result = harness_cli._handle_factor_synthesis(args)

    assert result.ok is True
    assert captured["command"][:-4] == [
        sys.executable,
        str(repo_root / "run_factor_synthesis.py"),
        "--config",
        str(config_path),
        "--return-series-output-dir",
        str(
            repo_root
            / "outputs"
            / "agent_harness"
            / "factor-synthesis"
            / captured["output_dir"].name
            / "attempts"
            / captured["command"][-1]
            / "returns"
        ),
        "--run-id",
        captured["output_dir"].name,
    ]
    from tools.agent_harness.run_result import execution_result

    command = captured["command"]
    with execution_result(command[-3], command="factor-synthesis", run_id=captured["output_dir"].name,
                          execution_id=command[-1], config_path=config_path) as payload:
        payload["execution"].update(expected=1, completed=1)
    metrics = captured["success_metrics_builder"]("", "")
    assert metrics["execution"]["completed"] == 1


def test_factor_synthesis_extracts_structured_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from tools.agent_harness import cli as harness_cli

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"synthesis_methods": {"ar_demo_p1": {"method": "lightgbm"}}}),
        encoding="utf-8",
    )
    stdout = "\n".join(
        [
            "📊 TM_AR_DEMO IC数据分析:",
            "  1D: 均值=0.0320, 标准差=0.1100, 范围=[-0.2100, 0.2600], 正值占比=63.00%, 样本数=100",
            "🎯 TM_AR_DEMO 整体IC: 0.0320",
            "📈 TM_AR_DEMO IR数据分析:",
            "  1D: IR=0.4500",
            "🎯 TM_AR_DEMO 整体IR: 0.4500",
            "📋 TM_AR_DEMO 分位数收益分析:",
            "📊 VectorBT 回测结果 (top_pct=0.06):",
            "  年化收益: 82.4%",
            "  夏普比率: 2.32",
            "  Calmar比率: 3.35",
            "  Sortino比率: 3.66",
            "  最大回撤: -24.6%",
            "  总收益: 909.4%",
            "  交易次数: 88",
            "📅 年度收益分解:",
            "  2024: 149.8%",
            (
                "ALGORITHM_SYNTHESIS_DIAGNOSTICS: "
                '{"synthesis_name":"ar_demo_p1","method_family":"lightgbm",'
                '"model_count":4,"nonzero_feature_count_mean":2.0,'
                '"feature_importance_sum_mean":18.5,"final_composite_std":0.91,'
                '"blend_ratio_mean":0.42,"blend_ratio_max":0.5,'
                '"backtest_signature":{"sharpe_ratio":2.32}}'
            ),
        ]
    )

    metrics = harness_cli._extract_factor_synthesis_metrics(
        stdout, "", str(config_path)
    )

    assert metrics["synthesis_method_names"] == ["ar_demo_p1"]
    assert metrics["overall_ic"] == 0.032
    assert metrics["overall_ir"] == 0.45
    assert metrics["quality_metrics"]["overall_ic"] == 0.032
    assert metrics["quality_metrics"]["overall_ir"] == 0.45
    assert metrics["synthesis_diagnostic"]["synthesis_name"] == "ar_demo_p1"
    assert metrics["synthesis_diagnostic"]["feature_importance_sum_mean"] == 18.5
    assert metrics["backtest_summary"]["backtest_signature"]["annual_return"] == 0.824
    assert metrics["backtest_summary"]["backtest_signature"]["yearly_returns"] == {
        "2024": 1.498
    }


def test_factor_synthesis_extracts_return_series_artifacts_from_structured_summary(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from tools.agent_harness import cli as harness_cli

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"synthesis_methods": {"demo": {}}}), encoding="utf-8"
    )
    artifact = {
        "artifact_type": "algorithm_return_series",
        "format": "csv",
        "path": "outputs/agent_harness/run/returns/trial_a.csv",
        "metadata_path": "outputs/agent_harness/run/returns/trial_a.metadata.json",
        "trial_id": "trial_a",
        "selection_mode": "top_pct",
        "top_pct": 0.06,
        "top_k": None,
        "frequency": "daily",
        "return_column": "net_return",
    }
    stdout = "VECTORBT_RESULT_SUMMARIES: " + json.dumps(
        [
            {
                "trial_id": "trial_a",
                "selection_mode": "top_pct",
                "top_pct": 0.06,
                "annual_return": 0.22,
                "sharpe_ratio": 1.4,
                "return_series_artifact": artifact,
            }
        ]
    )

    metrics = harness_cli._extract_factor_synthesis_metrics(
        stdout, "", str(config_path)
    )

    assert metrics["backtest_runs"][0]["return_series_artifact"] == artifact
    assert metrics["backtest_summary"]["trial_id"] == "trial_a"


def test_factor_synthesis_does_not_load_unbound_benchmark_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from tools.agent_harness import cli as harness_cli

    output_dir = tmp_path / "execution"
    output_dir.mkdir()
    report = {
        "benchmarks": {
            "factor1_top": {
                "net": {
                    "annual_turnover": 2.5,
                    "holding_count_violation_details": [{"orders": ["large"]}],
                }
            },
            "candidate": {
                "active_vs_parent": {
                    "benchmark": "factor1_top",
                    "annual_active_return": 0.02,
                    "information_ratio": 0.3,
                }
            },
        }
    }
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "synthesis_methods": {"demo": {}},
                "output_settings": {"diagnostics_output_dir": str(output_dir)},
            }
        ),
        encoding="utf-8",
    )

    metrics = harness_cli._extract_factor_synthesis_metrics(
        "\n".join(
            [
                'VECTORBT_RESULT_SUMMARIES: [{"annual_turnover": 1.0}]',
                "ALGORITHM_BENCHMARK_REPORT_PATH: benchmark_report.json",
            ]
        ),
        "",
        str(config_path),
    )

    assert "benchmark_report" not in metrics
    assert "benchmark_report" not in metrics["backtest_summary"]
    assert metrics["quality_metrics"]["has_benchmark_report"] is False


def test_factor_synthesis_neutralized_input_warning_is_structured(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from tools.agent_harness import cli as harness_cli

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"synthesis_methods": {"demo": {}}}), encoding="utf-8"
    )
    warning = "[中性化输入缺失] raw=FactorA expected=FactorA__neu fallback=raw"

    metrics = harness_cli._extract_factor_synthesis_metrics(
        "",
        f"2026-07-10 12:00:00,000 - FactorSynthesizer - WARNING - {warning}",
        str(config_path),
    )

    assert metrics["neutralized_input_warning_count"] == 1
    assert metrics["neutralized_input_warnings"] == [warning]
    result = harness_cli._success_result(
        action="factor-synthesis",
        mode="subprocess",
        output_dir=tmp_path,
        run_id="run",
        command=["factor-synthesis"],
        started_at=harness_cli._utc_now(),
        metrics={k: v for k, v in metrics.items() if k != "_warnings"},
        artifacts=[],
        warnings=metrics["_warnings"],
    )
    assert result.to_dict()["warnings"] == [warning]


