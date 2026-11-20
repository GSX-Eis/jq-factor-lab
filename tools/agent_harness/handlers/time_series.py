"""Handle Bar evaluation and read-only evaluation queries."""
from __future__ import annotations

import argparse
import json
import subprocess as subprocess  # Default for direct imports via _cli_bindings.
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict

from tools.agent_harness import _json_io, runner
from tools.agent_harness._cli_compat import _cli_bindings
from tools.agent_harness.runner import HarnessResult


def _bar_artifact_paths(evaluation_dir: Path, diagnostics_report: bool, *,
                        evaluation_v2: bool = False) -> set[Path]:
    names = ("summary.json", "per_bar_labels.csv", "ic_by_horizon.csv", "quantile_stats.csv")
    if diagnostics_report:
        from backtest_and_analysis.time_series.artifacts import (
            DIAGNOSTICS_CSV,
            DIAGNOSTICS_MARKDOWN,
        )
        names += (DIAGNOSTICS_CSV, DIAGNOSTICS_MARKDOWN)
    if evaluation_v2:
        from backtest_and_analysis.time_series.evaluation_v2 import EVALUATION_V2_FILENAME
        names += (EVALUATION_V2_FILENAME,)
    return {evaluation_dir / name for name in names}


def _validate_bar_output_paths(db_path: Path, output_dir: Path, json_out: str,
                               evaluation_dir: Path | None = None, *, diagnostics_report: bool = False,
                               evaluation_v2: bool = False) -> None:
    from utils.bar_evaluation_store import BarEvaluationPreconditionError

    cli = _cli_bindings(__name__)
    targets = [output_dir / "summary.json", output_dir / "config/resolved.json",
               output_dir / "output_bindings.json"]
    if output_dir.parent.name == "attempts":
        targets.append(output_dir.parent.parent / "summary.json")
    if json_out:
        targets.append(Path(json_out).expanduser())
    if evaluation_dir is not None:
        targets.extend(cli._bar_artifact_paths(evaluation_dir, diagnostics_report, evaluation_v2=evaluation_v2))
    db_path = db_path.resolve()
    for target in targets:
        target = target.resolve()
        if (db_path == target or db_path in target.parents or target in db_path.parents
                or (db_path.exists() and target.exists() and db_path.samefile(target))):
            raise BarEvaluationPreconditionError(
                f"results_db path overlaps an output path: {db_path} / {target}")


def _handle_time_series_analysis(
    args: argparse.Namespace, *, public_output: ExitStack | None = None,
) -> HarnessResult:
    from tools.agent_harness.time_series_analysis import evaluation_v2_requested, load_config
    from utils.bar_evaluation_store import (
        BarEvaluationConflictError,
        BarEvaluationIntegrityError,
        BarEvaluationPreconditionError,
        _validate_db_target_path,
        ingest_evaluation,
    )
    from utils.output_paths import execution_environment

    cli = _cli_bindings(__name__, _json_io, runner)
    action = args.command
    output_root = cli._resolve_args_output_root(args.output_root)
    output_dir = cli._output_dir("time-series-analysis", output_root, args.run_id)
    harness_summary_dirs = {output_dir.resolve()}
    if output_dir.parent.name == "attempts":
        harness_summary_dirs.add(output_dir.parent.parent.resolve())
    destination = output_dir / "evaluation"
    command = [sys.executable, "-m", "tools.agent_harness.time_series_analysis"]
    started_at = cli._utc_now()
    common = dict(action=action, output_dir=output_dir, run_id=args.run_id,
                  started_at=started_at, command=command)
    phase = "preflight"
    config_error = None
    config_path = (Path(args.config).expanduser().resolve() if public_output is not None
                   else cli._resolve_repo_relative_path(args.config))
    try:
        config = load_config(config_path) if args.config else {}
    except Exception as exc:
        config, config_error = {}, exc
    guard_config = config
    if config_error is not None:
        try:
            guard_config = cli._safe_read_json(config_path)
        except (OSError, ValueError, TypeError):
            pass
    diagnostics_requested = bool(getattr(args, 'diagnostics_report', False) or guard_config.get('diagnostics_report') is True)
    v2_requested = bool(getattr(args, 'evaluation_v2', False) or evaluation_v2_requested(guard_config))
    guard_reason = "time_series_precondition_failed"
    try:
        requested_destination = None
        if args.output_dir:
            requested_destination = Path(args.output_dir).expanduser()
        elif guard_config.get("output_dir"):
            requested_destination = (Path(guard_config["output_dir"]).expanduser() if public_output is not None
                                     else cli._resolve_repo_relative_path(guard_config["output_dir"]))
        if requested_destination is not None:
            if public_output is not None:
                if requested_destination.is_symlink():
                    raise ValueError(f"Output directory must not be a symlink: {requested_destination}")
                if not args.overwrite and requested_destination.exists():
                    raise FileExistsError(
                        f"Output already exists: {requested_destination}. Pass --overwrite or choose a new directory."
                    )
            requested_destination = requested_destination.resolve()
        guarded_destination = requested_destination or destination
        if guarded_destination in harness_summary_dirs:
            guarded_destination = destination

        # A retry can target an artifact published by an earlier attempt.
        if args.json_out and Path(args.json_out).expanduser().resolve() in {
            path.resolve() for path in cli._bar_artifact_paths(
                guarded_destination, diagnostics_report=True, evaluation_v2=True
            )
        }:
            raise ValueError("--json-out must differ from evaluation artifact paths")

        guard_reason = "bar_evaluation_store_precondition_failed"
        cli_results_db = getattr(args, "results_db", None)
        raw_db = cli_results_db if cli_results_db is not None else guard_config.get("results_db")
        guarded_db = None
        if isinstance(raw_db, str) and raw_db.strip():
            guarded_db = (Path(raw_db).expanduser().resolve()
                          if cli_results_db is not None or public_output is not None
                          else cli._resolve_repo_relative_path(raw_db))
        if guarded_db is not None:
            cli._validate_bar_output_paths(guarded_db, output_dir, args.json_out, guarded_destination,
                                       diagnostics_report=diagnostics_requested, evaluation_v2=v2_requested)
        if (public_output is not None and requested_destination is not None and config_error is None
                and not (args.fake_result or args.dry_run)):
            from utils.local_outputs import reserve_output_directory

            guard_reason = "time_series_precondition_failed"
            public_output.enter_context(reserve_output_directory(requested_destination, overwrite=args.overwrite))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        # Report path failures on stdout before writing any summary or clearing a marker.
        result = cli._failure_result(**common, mode="preflight",
            reason=guard_reason, exit_code=cli.EXIT_PRECONDITION,
            metrics={"error": str(exc)}, artifacts=[])
        result.error["message"] = str(exc)
        return result
    if args.fake_result:
        return cli._fake_result(**common, fake_result=args.fake_result, json_out=args.json_out)
    if args.dry_run:
        dry_metrics: Dict[str, Any] = {"planned_config": args.config}
        if getattr(args, "results_db", None) is not None:
            dry_metrics["planned_results_db"] = str(Path(args.results_db).expanduser().resolve())
        return cli._complete_result(cli._success_result(**common, mode="dry_run", reason="dry_run",
            metrics=dry_metrics, artifacts=cli._common_artifacts(output_dir)), args.json_out)

    try:
        if public_output is not None and config_error is not None:
            raise config_error
        # Clear an explicit CLI retry's marker even if config loading itself fails.
        if args.output_dir or (config_error is None and config.get("output_dir")):
            destination = guarded_destination
            (destination / "summary.json").unlink(missing_ok=True)
        if config_error is not None:
            raise config_error
        config.pop("results_db", None)
        if raw_db is not None and guarded_db is None:
            option = "--results-db" if cli_results_db is not None else "config results_db"
            raise BarEvaluationPreconditionError(f"{option} must be a nonempty path")
        results_db_path = _validate_db_target_path(guarded_db) if guarded_db is not None else None

        for name in ("bars_csv", "factors_csv", "analysis_mode", "view", "calendar",
                     "entry_offset_bars", "n_quantiles"):
            value = getattr(args, name)
            if value is not None:
                config[name] = value
        if public_output is not None:
            for name in ("bars_csv", "factors_csv"):
                value = config.get(name)
                if isinstance(value, str) and value.strip():
                    config[name] = str(Path(value).expanduser().resolve())
            continuous = config.get("continuous_futures")
            if isinstance(continuous, dict):
                db = continuous.get("db")
                if isinstance(db, str) and db.strip():
                    continuous["db"] = str(Path(db).expanduser().resolve())
        if args.horizons is not None:
            config["horizons"] = [int(h.strip()) for h in args.horizons.split(",")]
        if args.calendar is not None:
            config.pop("calendar_by_instrument", None)
        if getattr(args, 'diagnostics_report', False):
            config['diagnostics_report'] = True
        if getattr(args, 'evaluation_v2', False):
            config['evaluation_v2'] = True
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config = output_dir / "config" / "resolved.json"
        cli._write_json(resolved_config, config)
        command.extend([str(resolved_config), str(destination)])
        phase = "subprocess"
        with execution_environment(output_dir, task_root=output_root if public_output is not None else None) as (environment, leases):
            proc = cli.subprocess.run(command, cwd=cli.REPO_ROOT, env=environment, pass_fds=leases,
                capture_output=True, text=True, timeout=args.timeout_sec, check=False)
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Bar evaluation worker returned invalid JSON (exit {proc.returncode}): "
                               f"{cli._tail(proc.stderr or proc.stdout)}") from exc
        if not payload["ok"]:
            result = cli._failure_result(**common, mode="subprocess", reason=payload["error"]["code"],
                exit_code=payload["exit_code"], metrics={}, artifacts=cli._common_artifacts(output_dir))
            result.error = payload["error"]
        elif proc.returncode != 0:
            raise RuntimeError(f"Bar evaluation worker exited {proc.returncode}: {proc.stderr}")
        else:
            metrics_out = dict(payload["metrics"])
            if results_db_path is not None:
                phase = "results_db"
                receipt = ingest_evaluation(
                    results_db_path,
                    evaluation_dir=destination,
                    artifacts=payload["artifacts"],
                    run_id=args.run_id,
                )
                metrics_out["results_db"] = {
                    "path": str(receipt.db_path),
                    "identity_hash": receipt.identity_hash,
                    "status": receipt.status,
                }
            result = cli._success_result(**common, mode="subprocess", metrics=metrics_out,
                artifacts=cli._common_artifacts(output_dir) + payload["artifacts"])
    except Exception as exc:
        if phase == "results_db":
            if isinstance(exc, BarEvaluationConflictError):
                reason = "bar_evaluation_store_conflict"
                exit_code = cli.EXIT_QUALITY_GATE
            elif isinstance(exc, (BarEvaluationPreconditionError, BarEvaluationIntegrityError, ValueError, FileNotFoundError)):
                reason = "bar_evaluation_store_precondition_failed"
                exit_code = cli.EXIT_PRECONDITION
            else:
                reason = "bar_evaluation_store_write_failed"
                exit_code = cli.EXIT_EXECUTION
            result = cli._failure_result(
                **common,
                mode="results_db",
                reason=reason,
                exit_code=exit_code,
                metrics={"exception_type": type(exc).__name__, "error": str(exc)},
                artifacts=cli._common_artifacts(output_dir),
            )
            result.error["message"] = str(exc)
        else:
            precondition = phase == "preflight" and isinstance(exc, (OSError, ValueError, TypeError))
            reason = (
                "bar_evaluation_store_precondition_failed"
                if isinstance(exc, BarEvaluationPreconditionError)
                else ("time_series_precondition_failed" if precondition else "time_series_execution_failed")
            )
            result = cli._failure_result(**common, mode="preflight" if precondition else "subprocess",
                reason=reason,
                exit_code=cli.EXIT_PRECONDITION if precondition else cli.EXIT_EXECUTION,
                metrics={"exception_type": type(exc).__name__, "error": str(exc)},
                artifacts=cli._common_artifacts(output_dir))
            result.error["message"] = str(exc)
    if not result.ok:
        try:
            (destination / "summary.json").unlink(missing_ok=True)
        except OSError as exc:
            result.error["details"]["completion_cleanup_error"] = str(exc)
    return cli._complete_result(result, args.json_out)


def _handle_read_bar_evaluations(args: argparse.Namespace) -> HarnessResult:
    from utils.bar_evaluation_store import (
        BarEvaluationPreconditionError,
        query_bar_evaluations,
    )

    cli = _cli_bindings(__name__, runner)
    action = "read-bar-evaluations"
    output_dir = cli._output_dir(
        action, cli._resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = cli._utc_now()
    db_path = Path(args.results_db).expanduser().resolve()
    command = [action, "--results-db", str(db_path)]
    artifacts = cli._common_artifacts(output_dir)
    common = dict(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        started_at=started_at,
        artifacts=artifacts,
    )
    try:
        cli._validate_bar_output_paths(db_path, output_dir, args.json_out)
    except (OSError, ValueError, TypeError) as exc:
        result = cli._failure_result(**{**common, "artifacts": []}, mode="preflight",
            reason="bar_evaluation_read_precondition_failed", exit_code=cli.EXIT_PRECONDITION,
            metrics={"results_db": str(db_path), "error": str(exc)})
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
        return cli._complete_result(
            cli._success_result(
                **common,
                mode="dry_run",
                reason="dry_run",
                metrics={"results_db": str(db_path)},
            ),
            json_out=args.json_out,
        )

    try:
        payload = query_bar_evaluations(
            db_path,
            identity_hash=args.identity_hash,
            analysis_mode=args.analysis_mode,
            frequency=args.frequency,
            calendar_id=args.calendar_id,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
        )
        return cli._complete_result(
            cli._success_result(
                **common,
                mode="local",
                metrics={"results_db": str(db_path), **payload},
            ),
            json_out=args.json_out,
        )
    except (BarEvaluationPreconditionError, FileNotFoundError, ValueError) as exc:
        result = cli._failure_result(
            **common,
            mode="preflight",
            reason="bar_evaluation_read_precondition_failed",
            exit_code=cli.EXIT_PRECONDITION,
            metrics={"results_db": str(db_path), "error": str(exc)},
        )
        result.error["message"] = str(exc)
        return cli._complete_result(result, json_out=args.json_out)
    except Exception as exc:
        result = cli._failure_result(
            **common,
            mode="local",
            reason="bar_evaluation_read_failed",
            exit_code=cli.EXIT_EXECUTION,
            metrics={"results_db": str(db_path), "error": str(exc)},
        )
        result.error["message"] = str(exc)
        return cli._complete_result(result, json_out=args.json_out)
