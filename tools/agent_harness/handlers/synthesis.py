"""Handle factor synthesis and parent-factor screening."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from tools.agent_harness import d9_capacity_audit, run_result, runner
from tools.agent_harness._cli_compat import _cli_bindings
from tools.agent_harness.runner import HarnessResult


def _handle_factor_synthesis(args: argparse.Namespace) -> HarnessResult:
    cli = _cli_bindings(__name__, d9_capacity_audit, run_result, runner)
    action = "factor-synthesis"
    output_dir = cli._output_dir(
        action, cli._resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = cli._utc_now()
    resolved_config_path = cli._resolve_repo_relative_path(args.config)
    execution_id = uuid.uuid4().hex
    return_series_output_dir = (
        Path(args.return_series_output_dir).expanduser().resolve()
        if args.return_series_output_dir
        else output_dir / "attempts" / execution_id / "returns"
    )
    effective_run_id = args.run_id or output_dir.name
    result_path = output_dir / "attempts" / execution_id / "result.json"
    command = [
        sys.executable,
        str(cli.REPO_ROOT / "run_factor_synthesis.py"),
        "--config",
        str(resolved_config_path),
        "--return-series-output-dir",
        str(return_series_output_dir),
        "--run-id",
        effective_run_id,
        "--result-out", str(result_path),
        "--execution-id", execution_id,
    ]
    artifacts = cli._common_artifacts(output_dir) + [{"artifact_type": "execution_result", "path": str(result_path)}]
    artifacts.append(
        {
            "artifact_type": "synthesis_output_dir",
            "path": str(result_path.parent),
        }
    )
    artifacts.append(
        {
            "artifact_type": "algorithm_return_series_dir",
            "path": str(return_series_output_dir),
        }
    )
    if args.fake_result:
        return cli._fake_result(
            action=action,
            fake_result=args.fake_result,
            output_dir=output_dir,
            run_id=args.run_id,
            started_at=started_at,
            command=command,
            json_out=args.json_out,
        )
    if args.dry_run:
        return cli._complete_result(
            cli._success_result(
                action=action,
                mode="dry_run",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                metrics={"planned_config": str(resolved_config_path)},
                artifacts=artifacts,
                reason="dry_run",
            ),
            json_out=args.json_out,
        )
    if not resolved_config_path.exists():
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="preflight",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="missing_config",
                exit_code=cli.EXIT_PRECONDITION,
                metrics={"missing_path": str(resolved_config_path)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    try:
        audit_context = cli.load_capacity_audit_context(resolved_config_path)
        audit_preflight = (
            cli.preflight_capacity_audit(audit_context)
            if audit_context is not None
            else None
        )
    except cli.D9CapacityAuditError as exc:
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="preflight",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason=exc.code,
                exit_code=cli.EXIT_PRECONDITION,
                metrics={
                    "d9_capacity_audit_error": {
                        "message": str(exc),
                        "details": exc.details,
                    }
                },
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    source_sha256 = cli.config_sha256(resolved_config_path)
    config_payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    result = cli._run_subprocess(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        timeout_sec=args.timeout_sec,
        started_at=started_at,
        artifacts=artifacts,
        json_out=args.json_out,
        success_metrics_builder=lambda stdout, stderr: cli._bound_result_metrics(
            result_path, action=action, run_id=effective_run_id, execution_id=execution_id,
            source_sha256=source_sha256, config_payload=config_payload,
            stdout=stdout, stderr=stderr,
        ),
    )
    if audit_context is None or not result.ok:
        return result
    try:
        audit = cli.build_capacity_audit(
            audit_context,
            audit_preflight or {},
            result.metrics,
        )
        audit_path = cli.write_capacity_audit(output_dir / "d9_capacity_audit.json", audit)
    except cli.D9CapacityAuditError as exc:
        failed_metrics = dict(result.metrics)
        failed_metrics["d9_capacity_audit_error"] = {
            "message": str(exc),
            "details": exc.details,
        }
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="subprocess",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason=exc.code,
                exit_code=cli.EXIT_QUALITY_GATE,
                metrics=failed_metrics,
                artifacts=list(result.artifacts),
                warnings=result.warnings,
                stdout_tail=result.stdout_tail,
                stderr_tail=result.stderr_tail,
            ),
            json_out=args.json_out,
        )
    result.metrics["d9_capacity_audit"] = {
        "schema_version": audit["schema_version"],
        "path": str(audit_path),
        "preflight": audit["preflight"],
        "aggregate": audit["aggregate"],
        "return_series_artifact": audit["return_series_artifact"],
    }
    result.artifacts.append(
        {
            "artifact_type": "d9_capacity_audit",
            "path": str(audit_path),
            "schema_version": audit["schema_version"],
        }
    )
    return cli._complete_result(result, json_out=args.json_out)


def _handle_parent_factor_screen(args: argparse.Namespace) -> HarnessResult:
    cli = _cli_bindings(__name__, runner)
    action = "parent-factor-screen"
    output_dir = cli._output_dir(
        action, cli._resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = cli._utc_now()
    job_path = Path(args.job).expanduser().resolve()
    evidence_path = Path(args.output).expanduser().resolve()
    command = [
        sys.executable,
        "-m",
        "tools.agent_harness.cli",
        action,
        "--job",
        str(job_path),
        "--output",
        str(evidence_path),
    ]
    artifacts = cli._common_artifacts(output_dir) + [
        {"artifact_type": "algorithm_parent_factor_screen", "path": str(evidence_path)}
    ]
    if args.dry_run:
        return cli._complete_result(
            cli._success_result(
                action=action,
                mode="dry_run",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                metrics={"job": str(job_path), "output": str(evidence_path)},
                artifacts=artifacts,
                reason="dry_run",
            ),
            json_out=args.json_out,
        )
    try:
        from factor_synthesis.parent_factor_screen import (
            ParentFactorScreenError,
            run_parent_factor_screen,
        )

        evidence = run_parent_factor_screen(job_path, evidence_path)
    except (FileNotFoundError, json.JSONDecodeError, ParentFactorScreenError, ValueError) as exc:
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="preflight",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="parent_factor_screen_invalid",
                exit_code=cli.EXIT_PRECONDITION,
                metrics={"error": str(exc)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    except Exception as exc:
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="execution",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="parent_factor_screen_failed",
                exit_code=cli.EXIT_EXECUTION,
                metrics={"error": str(exc)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    return cli._complete_result(
        cli._success_result(
            action=action,
            mode="execution",
            output_dir=output_dir,
            run_id=args.run_id,
            command=command,
            started_at=started_at,
            metrics={
                "tested_factor_count": evidence["tested_factor_count"],
                "computed_factor_count": evidence["computed_factor_count"],
                "classification_counts": evidence["classification_counts"],
                "evidence_path": str(evidence_path),
            },
            artifacts=artifacts,
        ),
        json_out=args.json_out,
    )
