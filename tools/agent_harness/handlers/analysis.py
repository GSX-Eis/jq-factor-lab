"""Handle single-factor analysis, symbolic mining, and alpha realization."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from tools.agent_harness import _json_io, metrics_parsers, run_result, runner
from tools.agent_harness._cli_compat import _cli_bindings
from tools.agent_harness.handlers import time_series
from tools.agent_harness.runner import HarnessResult


def _handle_single_factor_analysis(args: argparse.Namespace) -> HarnessResult:
    cli = _cli_bindings(__name__, _json_io, time_series, run_result, runner)
    if (getattr(args, "market_db", None) is not None or args.bars_csv is not None
            or args.analysis_mode == "time_series"
            or getattr(args, "slot", None) is not None or getattr(args, "slot_id", None) is not None):
        return cli._handle_time_series_analysis(args)
    config = {}
    if args.config:
        config = cli._safe_read_json(cli._resolve_repo_relative_path(args.config))
        if (config.keys() & {"bars_csv", "time_series_analysis", "continuous_futures", "slot", "slot_id",
                             "slot_config", "factor_store_root", "market_db"}
                or config.get("analysis_mode") == "time_series"):
            return cli._handle_time_series_analysis(args)
    action = "single-factor-analysis"
    output_dir = cli._output_dir(
        action, cli._resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = cli._utc_now()
    execution_id = uuid.uuid4().hex
    effective_run_id = args.run_id or output_dir.name
    result_path = output_dir / "attempts" / execution_id / "result.json"
    command = [sys.executable, str(cli.REPO_ROOT / "run_analysis_main.py"),
               "--result-out", str(result_path), "--run-id", effective_run_id,
               "--execution-id", execution_id]
    evaluation_db = None
    if getattr(args, "results_db", None) is not None:
        from utils.bar_evaluation_store import _validate_db_target_path

        try:
            requested_db = Path(args.results_db).expanduser().resolve()
            cli._validate_bar_output_paths(requested_db, output_dir, args.json_out)
            cli._validate_bar_output_paths(requested_db, result_path.parent, args.json_out,
                                           result_path.parent / "bar-evaluations")
            if requested_db.is_relative_to(result_path.parent):
                raise ValueError("results_db must differ from bound analysis output paths")
            if args.materialize_only or args.mining_only or config.get("materialize_only") or config.get("mining_only"):
                raise ValueError("--results-db requires a full daily analysis")
            evaluation_db = _validate_db_target_path(args.results_db)
            database = config.get("database", config)
            source_path = database.get("duckdb_path") if isinstance(database, dict) else None
            if isinstance(source_path, str) and source_path:
                source = cli._resolve_repo_relative_path(source_path)
                if source == evaluation_db or (source.is_file() and evaluation_db.is_file()
                                                 and source.samefile(evaluation_db)):
                    raise ValueError("results_db must differ from the input database")
            command.extend(["--bar-results-db", str(evaluation_db)])
        except (OSError, ValueError, TypeError) as exc:
            result = cli._failure_result(
                action=action, mode="preflight", output_dir=output_dir, run_id=args.run_id,
                command=command, started_at=started_at, reason="bar_evaluation_store_precondition_failed",
                exit_code=cli.EXIT_PRECONDITION, metrics={"error": str(exc)}, artifacts=[],
            )
            result.error["message"] = str(exc)
            return result
    if args.config:
        command.extend(["--config", args.config])
    if args.factor:
        command.extend(["--factor", args.factor])
    if getattr(args, "causal_gate_policy", None) is not None:
        command.extend(["--causal-gate-policy", args.causal_gate_policy])
    if args.use_dynamic:
        command.append("--use-dynamic")
    if args.materialize_only:
        command.append("--materialize-only")
    if args.mining_only:
        command.append("--mining-only")
    if args.pool:
        command.extend(["--pools", *args.pool])

    artifacts = cli._common_artifacts(output_dir) + [{"artifact_type": "execution_result", "path": str(result_path)}]
    artifacts.append({"artifact_type": "analysis_output_dir", "path": str(output_dir)})
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
        metrics = {
            "planned_factor": args.factor or "",
            "planned_config": args.config or "",
            "planned_materialize_only": bool(args.materialize_only),
        }
        return cli._complete_result(
            cli._success_result(
                action=action,
                mode="dry_run",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                metrics=metrics,
                artifacts=artifacts,
                reason="dry_run",
            ),
            json_out=args.json_out,
        )
    try:
        source_sha256 = cli.config_sha256(
            cli._resolve_repo_relative_path(args.config) if args.config
            else cli.REPO_ROOT / "backtest_and_analysis" / "config.py"
        )
    except OSError as exc:
        return cli._complete_result(cli._failure_result(
            action=action, mode="preflight", output_dir=output_dir, run_id=args.run_id,
            command=command, started_at=started_at, reason="missing_config",
            exit_code=cli.EXIT_PRECONDITION, metrics={"error": str(exc)}, artifacts=artifacts,
        ), json_out=args.json_out)

    def bound_metrics(_stdout: str, _stderr: str) -> dict:
        metrics = cli._bound_result_metrics(
            result_path, action=action, run_id=effective_run_id, execution_id=execution_id,
            source_sha256=source_sha256,
        )
        if evaluation_db is not None and not metrics["_execution_failed"]:
            artifact, = [item for item in metrics["_artifacts"] if item["artifact_type"] == "bar_evaluation_manifest"]
            manifest = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
            metrics["results_db"] = manifest["results_db"]
            metrics["_artifacts"].extend(manifest["artifacts"])
        gate_report = metrics.get("causal_gate")
        if isinstance(gate_report, dict) and gate_report.get("status") == "failed_fallback":
            metrics.setdefault("_warnings", []).append(
                f"Causal fast-path gate fell back to pit_replay for {gate_report.get('factor_name')}: "
                f"{gate_report.get('failure_reason')}"
            )
        return metrics

    return cli._run_subprocess(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        timeout_sec=args.timeout_sec,
        started_at=started_at,
        artifacts=artifacts,
        json_out=args.json_out,
        success_metrics_builder=bound_metrics,
    )


def _handle_symbolic_factor_mining(args: argparse.Namespace) -> HarnessResult:
    cli = _cli_bindings(__name__, metrics_parsers, runner)
    action = "symbolic-factor-mining"
    output_dir = cli._output_dir(
        action, cli._resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = cli._utc_now()
    command = [
        sys.executable,
        str(cli.REPO_ROOT / "run_symbolic_factor_mining.py"),
        "--config",
        args.config,
    ]
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    if getattr(args, "novelty_context", ""):
        command.extend(["--novelty-context", args.novelty_context])
    if getattr(args, "marginal_context", ""):
        command.extend(["--marginal-context", args.marginal_context])
    if args.skip_validation:
        command.append("--skip-validation")
    if getattr(args, "materialize_candidate_rank", 0):
        command.extend(
            ["--materialize-candidate-rank", str(args.materialize_candidate_rank)]
        )
    if getattr(args, "factor_path", ""):
        command.extend(["--factor-path", args.factor_path])

    artifacts = cli._common_artifacts(output_dir)
    artifacts.append(
        {"artifact_type": "symbolic_harness_output_dir", "path": str(output_dir)}
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
                metrics={
                    "planned_config": args.config,
                    "skip_validation": args.skip_validation,
                },
                artifacts=artifacts,
                reason="dry_run",
            ),
            json_out=args.json_out,
        )

    return cli._run_subprocess(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        timeout_sec=args.timeout_sec,
        started_at=started_at,
        artifacts=artifacts,
        json_out=args.json_out,
        success_metrics_builder=cli._extract_symbolic_factor_mining_metrics,
    )


def _handle_quanta_alpha_realize(args: argparse.Namespace) -> HarnessResult:
    cli = _cli_bindings(__name__, runner)
    action = "quanta-alpha-realize"
    output_dir = cli._output_dir(
        action, cli._resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = cli._utc_now()
    realization_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else output_dir / "realization.json"
    )
    command = [action, "--input", args.input, "--output", str(realization_path)]
    artifacts = cli._common_artifacts(output_dir) + [
        {
            "artifact_type": "quanta_alpha_realization",
            "path": str(realization_path),
            "schema_version": "quanta_alpha_realization_v1",
        }
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
                metrics={"input": args.input, "output": str(realization_path)},
                artifacts=artifacts,
                reason="dry_run",
            ),
            json_out=args.json_out,
        )
    try:
        from factor_mining.symbolic.quanta_alpha import (
            QuantaAlphaEvidenceError,
            realize_proposals,
        )

        realization = realize_proposals(args.input, realization_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="preflight",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="quanta_alpha_input_invalid",
                exit_code=cli.EXIT_PRECONDITION,
                metrics={"error": str(exc)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    except QuantaAlphaEvidenceError as exc:
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="execution",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="quanta_alpha_evidence_blocked",
                exit_code=cli.EXIT_EXECUTION,
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
                reason="quanta_alpha_realization_failed",
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
                "profile": realization["profile"],
                "top_k": realization["top_k"],
                "input_hash": realization["input_hash"],
                "output_hash": realization["output_hash"],
                "selected_proposal_ids": realization["selected_proposal_ids"],
                "proposal_count": len(realization["proposals"]),
            },
            artifacts=artifacts,
        ),
        json_out=args.json_out,
    )
