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
    slot_config_arg = str(getattr(args, "slot_config", "") or "").strip()
    slot_id_arg = str(getattr(args, "slot_id", "") or "").strip()
    factor_store_arg = str(getattr(args, "factor_values_store", "") or "").strip()
    frequency_arg = str(getattr(args, "frequency", "") or "").strip()
    command = [
        sys.executable,
        str(cli.REPO_ROOT / "run_factor_synthesis.py"),
        "--config",
        str(resolved_config_path),
        "--return-series-output-dir",
        str(return_series_output_dir),
        "--run-id",
        effective_run_id,
    ]
    resolved_slot_cfg = None
    if slot_config_arg:
        resolved_slot_cfg = (
            slot_config_arg
            if slot_config_arg == "jq_ashare_local"
            else str(cli._resolve_repo_relative_path(slot_config_arg))
        )
        command.extend(["--slot-config", resolved_slot_cfg])
    market_db_arg = getattr(args, "market_db", None)
    if market_db_arg is not None:
        command.extend(["--market-db", market_db_arg])
    if slot_id_arg:
        command.extend(["--slot-id", slot_id_arg])
    if factor_store_arg:
        command.extend(
            ["--factor-values-store", str(cli._resolve_repo_relative_path(factor_store_arg))]
        )
    if frequency_arg:
        command.extend(["--frequency", frequency_arg])
    command.extend(["--result-out", str(result_path), "--execution-id", execution_id])
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
    selection = None
    try:
        raw_config = cli._safe_read_json(resolved_config_path) if resolved_config_path.is_file() else {}
        market_db = market_db_arg if market_db_arg is not None else raw_config.get("market_db")
        if market_db is not None:
            from utils.market_db.identity import slot_market_db_identity
            from utils.market_db.research import (
                check_synthesis_market_outputs, open_research_slot, resolve_research_market_db,
                resolve_research_slot, synthesis_market_db_paths,
            )

            source = resolve_research_market_db(market_db)
            if market_db_arg is None:
                command.extend(["--market-db", market_db])
            check_synthesis_market_outputs(
                source, raw_config, result_path.parent, return_series_output_dir,
                files=(*runner._result_output_paths(output_dir, args.json_out), output_dir / "d9_capacity_audit.json"),
                directories=(result_path.parent,), cwd=cli.REPO_ROOT,
            )
            if not args.fake_result:
                if not resolved_config_path.is_file():
                    raise FileNotFoundError(resolved_config_path)
                declared_db, _ = synthesis_market_db_paths(raw_config)
                if not factor_store_arg and not raw_config.get("factor_values_store") and not (
                    raw_config.get("database", {}).get("factor_db", {}).get("duckdb_path")
                ):
                    raise ValueError("--market-db requires a Scheme B factor store")
                declaration = resolved_slot_cfg or raw_config.get("slot_config") or raw_config.get("slot")
                base_dir = Path(raw_config.get("slot_base_dir") or resolved_config_path.parent).expanduser().resolve()
                if isinstance(declaration, str) and declaration != "jq_ashare_local":
                    from_repo = cli._resolve_repo_relative_path(declaration)
                    if from_repo.is_file():
                        declaration = str(from_repo)
                frozen, base_dir = resolve_research_slot(declaration, slot_id_arg or raw_config.get("slot_id"), base_dir)
                with open_research_slot(market_db, frozen, base_dir, declared_db=declared_db) as (slot, _):
                    selection = slot_market_db_identity(slot)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        result = cli._failure_result(
            action=action, mode="preflight", output_dir=output_dir, run_id=args.run_id,
            command=command, started_at=started_at, reason="market_db_precondition_failed",
            exit_code=cli.EXIT_PRECONDITION, metrics={"error": str(exc)}, artifacts=[],
        )
        result.error["message"] = str(exc)
        return result
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
        dry_metrics = {"planned_config": str(resolved_config_path)}
        if selection is not None:
            dry_metrics["market_db"] = selection
        if slot_config_arg:
            dry_metrics["planned_slot_config"] = slot_config_arg
        if slot_id_arg:
            dry_metrics["planned_slot_id"] = slot_id_arg
        if factor_store_arg:
            dry_metrics["planned_factor_values_store"] = factor_store_arg
        if frequency_arg:
            dry_metrics["planned_frequency"] = frequency_arg
        return cli._complete_result(
            cli._success_result(
                action=action,
                mode="dry_run",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                metrics=dry_metrics,
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
            config_path=resolved_config_path, stdout=stdout, stderr=stderr,
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
