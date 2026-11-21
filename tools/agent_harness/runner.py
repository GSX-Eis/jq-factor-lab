"""Build Harness results and manage bound subprocess execution."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess as subprocess  # Default for direct imports via _cli_bindings.
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from tools.agent_harness import metrics_parsers, run_result
from tools.agent_harness._cli_compat import _cli_bindings
from tools.agent_harness._json_io import _safe_read_json as _safe_read_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "agent_harness"
DEFAULT_MONITOR_DB = REPO_ROOT / "data" / "duckdb" / "analysis_results.duckdb"

EXIT_SUCCESS = 0
EXIT_PRECONDITION = 2
EXIT_EXECUTION = 3
EXIT_QUALITY_GATE = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    from utils.local_outputs import write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _resolve_repo_relative_path(raw_path: str) -> Path:
    cli = _cli_bindings(__name__)
    candidate = Path(str(raw_path or "")).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (cli.REPO_ROOT / candidate).resolve()


def _bound_result_metrics(path, *, action, run_id, execution_id, source_sha256,
                          config_payload=None, stdout="", stderr=""):
    cli = _cli_bindings(__name__, metrics_parsers, run_result)
    payload = cli.read_execution_result(
        path, command=action, run_id=run_id, execution_id=execution_id,
        source_sha256=source_sha256,
    )
    raw = payload["metrics"]
    if action == "factor-synthesis":
        for key in ("ic_summaries", "backtest_runs", "synthesis_diagnostics", "benchmark_reports"):
            if not isinstance(raw.get(key, []), list) or any(not isinstance(item, dict) for item in raw.get(key, [])):
                raise ValueError(f"invalid synthesis result field: {key}")
        top_k = (config_payload or {}).get("performance_evaluation", {}).get("top_k_strategy", {})
        if payload["execution"]["status"] == "succeeded" and top_k.get("enabled") and not any(
            "error" not in item and all(isinstance(item.get(key), (int, float))
                                       and math.isfinite(item[key])
                                       for key in ("annual_return", "sharpe_ratio", "max_drawdown"))
            for item in raw.get("backtest_runs", [])
        ):
            raise ValueError("backtest_metrics_missing")
        metrics = cli._assemble_factor_synthesis_metrics(
            config_payload or {}, raw.get("ic_summaries", []), raw.get("backtest_runs", []),
            raw.get("synthesis_diagnostics", []), raw.get("benchmark_reports", []),
            cli._extract_neutralized_input_warnings(stdout, stderr),
        )
    else:
        metrics = dict(raw)
    metrics["execution"] = payload["execution"]
    metrics["_execution_failed"] = payload["execution"]["status"] != "succeeded"
    metrics["_artifacts"] = payload["artifacts"]
    return metrics


@dataclass
class HarnessResult:
    action: str
    ok: bool
    exit_code: int
    status: str
    reason: str
    mode: str
    output_dir: Path
    command: List[str]
    metrics: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    warnings: List[str]
    stdout_tail: str = ""
    stderr_tail: str = ""
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0
    run_id: str = ""
    error: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        summary_path = self.output_dir / "summary.json"
        return {
            "ok": self.ok,
            "command": {
                "name": self.action,
                "argv": list(self.command),
                "mode": self.mode,
                "run_id": self.run_id,
            },
            "summary": {
                "action": self.action,
                "status": self.status,
                "reason": self.reason,
                "output_dir": str(self.output_dir),
                "summary_path": str(summary_path),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_sec": self.elapsed_sec,
            },
            "artifacts": list(self.artifacts),
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
        }


def _output_dir(action: str, output_root: Path, run_id: str) -> Path:
    if run_id:
        suffix = run_id
    else:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_root / action / suffix
    # These two producers already create their own bound execution result attempt.
    if action in {"single-factor-analysis", "factor-synthesis"}:
        return root
    return root / "attempts" / uuid.uuid4().hex


def _success_result(
    *,
    action: str,
    mode: str,
    output_dir: Path,
    run_id: str,
    command: Sequence[str],
    started_at: str,
    metrics: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    warnings: List[str] | None = None,
    reason: str = "ok",
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> HarnessResult:
    cli = _cli_bindings(__name__)
    finished_at = cli._utc_now()
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    return cli.HarnessResult(
        action=action,
        ok=True,
        exit_code=cli.EXIT_SUCCESS,
        status="succeeded" if mode == "subprocess" else mode,
        reason=reason,
        mode=mode,
        output_dir=output_dir,
        command=list(command),
        metrics=metrics,
        artifacts=artifacts,
        warnings=list(warnings or []),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_sec=max((finished_dt - started_dt).total_seconds(), 0.0),
        run_id=run_id,
    )


def _failure_result(
    *,
    action: str,
    mode: str,
    output_dir: Path,
    run_id: str,
    command: Sequence[str],
    started_at: str,
    reason: str,
    exit_code: int,
    metrics: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    warnings: List[str] | None = None,
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> HarnessResult:
    cli = _cli_bindings(__name__)
    finished_at = cli._utc_now()
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    return cli.HarnessResult(
        action=action,
        ok=False,
        exit_code=exit_code,
        status="failed",
        reason=reason,
        mode=mode,
        output_dir=output_dir,
        command=list(command),
        metrics=metrics,
        artifacts=artifacts,
        warnings=list(warnings or []),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_sec=max((finished_dt - started_dt).total_seconds(), 0.0),
        run_id=run_id,
        error={"code": reason, "message": reason, "details": dict(metrics)},
    )


def _complete_result(result: HarnessResult, json_out: str = "") -> HarnessResult:
    cli = _cli_bindings(__name__)
    payload = result.to_dict()
    cli._write_json(result.output_dir / "summary.json", payload)
    if result.output_dir.parent.name == "attempts":
        cli._write_json(result.output_dir.parent.parent / "summary.json", payload)
    if json_out:
        cli._write_json(Path(json_out).expanduser().resolve(), payload)
    return result


def _fake_result(
    *,
    action: str,
    fake_result: str,
    output_dir: Path,
    run_id: str,
    started_at: str,
    command: Sequence[str],
    json_out: str = "",
) -> HarnessResult:
    cli = _cli_bindings(__name__)
    artifacts = [{"artifact_type": "summary", "path": str(output_dir / "summary.json")}]
    if fake_result == "success":
        return cli._complete_result(
            cli._success_result(
                action=action,
                mode="fake",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                metrics={"fake_result": fake_result},
                artifacts=artifacts,
            ),
            json_out=json_out,
        )
    if fake_result == "alignment_failed":
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="fake",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                reason="alignment_threshold_failed",
                exit_code=cli.EXIT_QUALITY_GATE,
                metrics={"fake_result": fake_result, "alignment_status": "failed"},
                artifacts=artifacts,
            ),
            json_out=json_out,
        )
    if fake_result == "degraded":
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="fake",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                reason="research_metrics_degraded",
                exit_code=cli.EXIT_QUALITY_GATE,
                metrics={"fake_result": fake_result, "oos_non_degraded": False},
                artifacts=artifacts,
            ),
            json_out=json_out,
        )
    return cli._complete_result(
        cli._failure_result(
            action=action,
            mode="fake",
            output_dir=output_dir,
            run_id=run_id,
            command=command,
            started_at=started_at,
            reason="command_failed",
            exit_code=cli.EXIT_EXECUTION,
            metrics={"fake_result": fake_result},
            artifacts=artifacts,
        ),
        json_out=json_out,
    )


def _kill_process_group(pgid: int) -> None:
    """Kill the entire process group by PGID."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    import time

    time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _build_extra_metrics(
    success_metrics_builder, stdout: str, stderr: str
) -> Dict[str, Any]:
    if not success_metrics_builder:
        return {}
    return success_metrics_builder(stdout or "", stderr or "") or {}


def _classify_subprocess_failure(
    *,
    action: str,
    subprocess_exit_code: int,
    extra_metrics: Mapping[str, Any],
) -> tuple[str, int, str]:
    cli = _cli_bindings(__name__)
    if (
        action == "symbolic-factor-mining"
        and str(extra_metrics.get("failure_reason") or "") == "data_preflight_failed"
    ):
        # Data backend unreachable (e.g. unmounted volume): structured code so
        # operators and the task manager can distinguish it from real
        # execution failures and avoid burning further runs.
        return "data_preflight_failed", cli.EXIT_EXECUTION, "data_preflight_failed"
    if action == "symbolic-factor-mining" and subprocess_exit_code == cli.EXIT_QUALITY_GATE:
        post_rerank = extra_metrics.get("post_rerank")
        if (
            isinstance(post_rerank, Mapping)
            and post_rerank.get("dual_gate_passed") is False
        ):
            detail = str(
                post_rerank.get("quality_gate_failure_reason")
                or extra_metrics.get("failure_reason")
                or post_rerank.get("selected_reason")
                or "post_rerank_quality_gate_failed"
            )
            return "post_rerank_quality_gate_failed", cli.EXIT_QUALITY_GATE, detail
        detail = str(
            extra_metrics.get("failure_reason") or "symbolic_quality_gate_failed"
        )
        return "symbolic_quality_gate_failed", cli.EXIT_QUALITY_GATE, detail
    return "command_failed", cli.EXIT_EXECUTION, ""


def _run_subprocess(**kwargs) -> HarnessResult:
    from utils.output_paths import execution_environment
    cli = _cli_bindings(__name__)
    command = kwargs["command"]
    if "--result-out" in command:
        attempt = Path(command[command.index("--result-out") + 1]).parent
    else:
        attempt = kwargs["output_dir"]
        if attempt.parent.name != "attempts":
            attempt = attempt / "attempts" / uuid.uuid4().hex
    try:
        with execution_environment(attempt) as (environment, leases):
            if "--config" in command:
                source = Path(command[command.index("--config") + 1])
                if source.is_file():
                    config_dir = attempt / "config"
                    config_dir.mkdir(exist_ok=True)
                    (config_dir / "source.json").write_bytes(source.read_bytes())
                    settings = json.loads(source.read_text()).get("output_settings", {})
                    if settings.get("diagnostic_logs") is True:
                        environment["QUANT_DEBUG_OUTPUTS"] = "1"
            result = cli._run_subprocess_body(**kwargs, environment=environment, leases=leases)
            if not result.ok:
                raise cli.subprocess.CalledProcessError(result.exit_code, command)
    except cli.subprocess.CalledProcessError:
        if "result" not in locals():
            raise
    return result


def _run_subprocess_body(
    *,
    action: str,
    output_dir: Path,
    run_id: str,
    command: Sequence[str],
    timeout_sec: int,
    started_at: str,
    artifacts: List[Dict[str, Any]],
    json_out: str,
    success_metrics_builder=None,
    environment=None, leases=(),
) -> HarnessResult:
    cli = _cli_bindings(__name__)
    popen = cli.subprocess.Popen(
        list(command),
        cwd=str(cli.REPO_ROOT),
        env=environment,
        pass_fds=leases,
        stdout=cli.subprocess.PIPE,
        stderr=cli.subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = popen.communicate(timeout=timeout_sec)
        proc = cli.subprocess.CompletedProcess(
            list(command), popen.returncode, stdout, stderr
        )
    except cli.subprocess.TimeoutExpired:
        cli._kill_process_group(popen.pid)
        stdout, stderr = "", ""
        try:
            stdout, stderr = popen.communicate(timeout=5)
        except Exception:
            pass
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="subprocess",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                reason="timeout",
                exit_code=cli.EXIT_EXECUTION,
                metrics={"timeout_sec": timeout_sec},
                artifacts=artifacts,
                stdout_tail=cli._tail(stdout or ""),
                stderr_tail=cli._tail((stderr or "") + "\ncommand_timeout"),
            ),
            json_out=json_out,
        )

    except BaseException:
        cli._kill_process_group(popen.pid)
        try:
            popen.communicate(timeout=5)
        except BaseException:
            pass
        raise

    try:
        extra_metrics = cli._build_extra_metrics(
            success_metrics_builder, proc.stdout or "", proc.stderr or ""
        )
    except Exception as exc:
        return cli._complete_result(
            cli._failure_result(
                action=action, mode="subprocess", output_dir=output_dir, run_id=run_id,
                command=command, started_at=started_at, reason="invalid_execution_result",
                exit_code=cli.EXIT_EXECUTION, artifacts=artifacts,
                metrics={"subprocess_exit_code": proc.returncode, "result_error": str(exc)},
                stdout_tail=cli._tail(proc.stdout), stderr_tail=cli._tail(proc.stderr),
            ), json_out=json_out,
        )
    structured_warnings = list(extra_metrics.pop("_warnings", []))
    artifacts.extend(extra_metrics.pop("_artifacts", []))
    execution_failed = extra_metrics.pop("_execution_failed", False)

    if proc.returncode != 0 or execution_failed:
        reason, exit_code, failure_detail = cli._classify_subprocess_failure(
            action=action,
            subprocess_exit_code=int(proc.returncode),
            extra_metrics=extra_metrics,
        )
        metrics = {"subprocess_exit_code": proc.returncode, **extra_metrics}
        if failure_detail:
            metrics["failure_reason"] = failure_detail
        stderr_tail = proc.stderr
        if failure_detail and failure_detail not in (stderr_tail or ""):
            stderr_tail = (stderr_tail or "") + "\n" + failure_detail
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="subprocess",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                reason=reason,
                exit_code=exit_code,
                metrics=metrics,
                artifacts=artifacts,
                warnings=structured_warnings,
                stdout_tail=cli._tail(proc.stdout),
                stderr_tail=cli._tail(stderr_tail),
            ),
            json_out=json_out,
        )

    return cli._complete_result(
        cli._success_result(
            action=action,
            mode="subprocess",
            output_dir=output_dir,
            run_id=run_id,
            command=command,
            started_at=started_at,
            metrics={"subprocess_exit_code": proc.returncode, **extra_metrics},
            artifacts=artifacts,
            warnings=structured_warnings,
            reason="ok",
            stdout_tail=cli._tail(proc.stdout),
            stderr_tail=cli._tail(proc.stderr),
        ),
        json_out=json_out,
    )


def _resolve_args_output_root(raw: str) -> Path:
    cli = _cli_bindings(__name__)
    if raw:
        return Path(raw).expanduser().resolve()
    configured = os.environ.get("QUANTATION_OUTPUTS_DIR")
    return Path(configured).expanduser().resolve() / "agent_harness" if configured else cli.DEFAULT_OUTPUT_ROOT


def _common_artifacts(output_dir: Path) -> List[Dict[str, Any]]:
    return [{"artifact_type": "summary", "path": str(output_dir / "summary.json")}]
