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


def _bar_artifact_paths(evaluation_dir: Path, diagnostics_report: bool) -> set[Path]:
    from backtest_and_analysis.time_series.artifacts import (
        DIAGNOSTICS_CSV, DIAGNOSTICS_MARKDOWN, LEGACY_EVALUATION_SIDECAR,
    )

    # Reserve the retired name so auxiliary outputs cannot recreate a sidecar.
    names = ("summary.json", "per_bar_labels.csv", "ic_by_horizon.csv", "quantile_stats.csv",
             LEGACY_EVALUATION_SIDECAR)
    if diagnostics_report:
        names += (DIAGNOSTICS_CSV, DIAGNOSTICS_MARKDOWN)
    return {evaluation_dir / name for name in names}


def _validate_bar_output_paths(db_path: Path, output_dir: Path, json_out: str,
                               evaluation_dir: Path | None = None, *, diagnostics_report: bool = False,
                               database_label: str = "results_db") -> None:
    from utils.bar_evaluation_store import BarEvaluationPreconditionError

    cli = _cli_bindings(__name__)
    targets = [output_dir / "summary.json", output_dir / "config/resolved.json",
               output_dir / "output_bindings.json"]
    if output_dir.parent.name == "attempts":
        targets.append(output_dir.parent.parent / "summary.json")
    if json_out:
        targets.append(Path(json_out).expanduser())
    if evaluation_dir is not None:
        targets.extend(cli._bar_artifact_paths(evaluation_dir, diagnostics_report))
    db_path = db_path.resolve()
    for target in targets:
        target = target.resolve()
        if (db_path == target or db_path in target.parents or target in db_path.parents
                or (db_path.exists() and target.exists() and db_path.samefile(target))):
            raise BarEvaluationPreconditionError(
                f"{database_label} path overlaps an output path: {db_path} / {target}")


def _resolve_public_config_paths(config: Dict[str, Any]) -> None:
    """Resolve public entry file paths from the caller cwd before preflight and the worker read them."""
    for name in ("bars_csv", "factors_csv", "factor_file", "slot_db", "duckdb_path", "factor_store_root",
                 "slot_base_dir", "slot", "slot_config"):
        value = config.get(name)
        if isinstance(value, str) and value.strip() and (name != "slot" or value != "jq_ashare_local"):
            config[name] = str(Path(value).expanduser().resolve())
    raw_slot_base = config.get("slot_base_dir")
    if (any(isinstance(config.get(name), dict) for name in ("slot", "slot_config"))
            and not (isinstance(raw_slot_base, str) and raw_slot_base.strip())):
        config["slot_base_dir"] = str(Path.cwd().resolve())
    continuous = config.get("continuous_futures")
    if isinstance(continuous, dict):
        db = continuous.get("db")
        if isinstance(db, str) and db.strip():
            continuous["db"] = str(Path(db).expanduser().resolve())


def _handle_time_series_analysis(
    args: argparse.Namespace, *, public_output: ExitStack | None = None,
    write_dry_run: bool = True,
) -> HarnessResult:
    from backtest_and_analysis.time_series.artifacts import (
        DIAGNOSTICS_CSV, DIAGNOSTICS_MARKDOWN, validate_evaluation_output_dir,
    )
    from backtest_and_analysis.time_series.contracts import sha256_file
    from tools.agent_harness.result_content import validate_bar_content
    from tools.agent_harness.time_series_analysis import (
        BarEvaluationStoreFailure, load_config, resolve_slot_request, revoke_evaluation_completion,
    )
    from utils.bar_evaluation_store import (
        BarEvaluationPreconditionError,
        _validate_db_target_path,
        ingest_evaluation,
    )
    from utils.local_outputs import check_output_directory
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
    for name in ("bars_csv", "factors_csv", "analysis_mode", "view", "calendar",
                 "entry_offset_bars", "n_quantiles", "slot", "slot_id", "market_db"):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
    if getattr(args, "factor", None):
        config["factor_file"] = (
            str(Path(args.factor).expanduser().resolve())
            if public_output is not None
            else str(cli._resolve_repo_relative_path(args.factor))
        )
    if getattr(args, "factor_class", None):
        config["factor_class"] = args.factor_class
    if getattr(args, "causal_gate_policy", None) is not None:
        config["causal_gate_policy"] = args.causal_gate_policy
    cli_slot = getattr(args, "slot", None)
    if cli_slot is not None:
        config.pop("slot_config", None)
        if cli_slot and cli_slot != "jq_ashare_local":
            config["slot"] = str(Path(cli_slot).expanduser().resolve())
    selection = None
    guard_config = config
    if config_error is not None:
        try:
            guard_config = cli._safe_read_json(config_path)
        except (OSError, ValueError, TypeError):
            pass
    diagnostics_override = getattr(args, 'diagnostics_report', None)
    diagnostics_requested = (guard_config.get('diagnostics_report') is True
                             if diagnostics_override is None else diagnostics_override)
    guard_reason = "time_series_precondition_failed"
    try:
        if public_output is not None:
            _resolve_public_config_paths(config)
        requested_destination = None
        if args.output_dir:
            requested_destination = Path(args.output_dir).expanduser()
        elif guard_config.get("output_dir"):
            requested_destination = (Path(guard_config["output_dir"]).expanduser() if public_output is not None
                                     else cli._resolve_repo_relative_path(guard_config["output_dir"]))
        if requested_destination is not None:
            if public_output is not None:
                requested_destination = check_output_directory(
                    requested_destination, overwrite=args.overwrite)
            requested_destination = requested_destination.resolve()
        guarded_destination = requested_destination or destination
        if guarded_destination in harness_summary_dirs:
            guarded_destination = destination
        if public_output is not None:
            check_output_directory(output_dir, overwrite=True)
        validate_evaluation_output_dir(guarded_destination)

        # A retry can target an artifact published by an earlier attempt.
        if args.json_out:
            json_out = Path(args.json_out).expanduser().resolve()
            protected = (path.resolve() for path in cli._bar_artifact_paths(
                guarded_destination, diagnostics_report=True))
            if any(path == json_out or path in json_out.parents for path in protected):
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
                                       diagnostics_report=diagnostics_requested)
        if config.get("market_db") is not None and not args.fake_result:
            from utils.market_db import market_db_sidecar_path
            from utils.market_db.identity import slot_market_db_identity
            from utils.market_db.research import open_research_slot, same_database

            guard_reason = "market_db_precondition_failed"
            if config_error is not None:
                raise config_error
            if any(getattr(args, name, None) for name in ("materialize_only", "mining_only", "use_dynamic", "pool", "factor")):
                raise ValueError("--market-db selects a Slot evaluation over committed Scheme B factors")
            frozen, base_dir = resolve_slot_request(config)
            with open_research_slot(
                config["market_db"], frozen, base_dir,
                declared_db=config.get("slot_db", config.get("duckdb_path")),
            ) as (slot, source):
                selection = slot_market_db_identity(slot)
            for protected in (source, market_db_sidecar_path(source)):
                cli._validate_bar_output_paths(protected, output_dir, args.json_out, guarded_destination,
                                               diagnostics_report=diagnostics_requested, database_label="market_db")
                if guarded_db is not None and same_database(protected, guarded_db):
                    raise ValueError("results_db must differ from the market database and its profile")
            config["market_db_identity"] = selection
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
    if args.fake_result and not (public_output is not None and args.dry_run):
        return cli._fake_result(**common, fake_result=args.fake_result, json_out=args.json_out)
    if args.dry_run and public_output is None:
        dry_metrics: Dict[str, Any] = {"planned_config": args.config}
        if selection is not None:
            dry_metrics["market_db"] = selection
        if getattr(args, "results_db", None) is not None:
            dry_metrics["planned_results_db"] = str(Path(args.results_db).expanduser().resolve())
        return cli._complete_result(cli._success_result(**common, mode="dry_run", reason="dry_run",
            metrics=dry_metrics, artifacts=cli._common_artifacts(output_dir)), args.json_out)

    try:
        if public_output is not None and config_error is not None:
            raise config_error
        # Clear an explicit CLI retry's marker even if config loading itself fails.
        if (args.output_dir or (config_error is None and config.get("output_dir"))) and not args.dry_run:
            destination = guarded_destination
            (destination / "summary.json").unlink(missing_ok=True)
        if config_error is not None:
            raise config_error
        config.pop("results_db", None)
        if raw_db is not None and guarded_db is None:
            option = "--results-db" if cli_results_db is not None else "config results_db"
            raise BarEvaluationPreconditionError(f"{option} must be a nonempty path")
        results_db_path = _validate_db_target_path(guarded_db) if guarded_db is not None else None

        source_db = config.get("slot_db", config.get("duckdb_path"))
        if (results_db_path is not None and isinstance(source_db, str)
                and any(config.get(key) is not None for key in ("slot", "slot_id", "slot_config", "factor_store_root"))):
            source = cli._resolve_repo_relative_path(source_db)
            if source == results_db_path or (source.is_file() and results_db_path.is_file()
                                             and source.samefile(results_db_path)):
                raise BarEvaluationPreconditionError("results_db must differ from the Slot source database")
        if args.horizons is not None:
            config["horizons"] = [int(h.strip()) for h in args.horizons.split(",")]
        if args.calendar is not None:
            config.pop("calendar_by_instrument", None)
        if diagnostics_override is not None:
            config['diagnostics_report'] = diagnostics_override
        if args.dry_run:
            from tools.agent_harness.time_series_analysis import evaluate

            plan = evaluate(config, None)
            plan.update(config=str(config_path) if args.config else None,
                        planned_config=args.config,
                        output_dir=str(guarded_destination), output_root=str(output_root),
                        results_db=str(results_db_path) if results_db_path else None,
                        overwrite=args.overwrite, diagnostics_report=diagnostics_requested)
            if selection is not None:
                plan['market_db'] = selection
            if results_db_path is not None:
                plan['planned_results_db'] = str(results_db_path)
            dry_result = cli._success_result(**common, mode='dry_run', reason='dry_run',
                                             metrics=plan, artifacts=[])
            return dry_result if not write_dry_run else cli._complete_result(dry_result, args.json_out)
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
            err_details = payload["error"].get("details") if isinstance(payload["error"], dict) else None
            fail_metrics = (
                {"causal_gate": err_details["causal_gate"]}
                if isinstance(err_details, dict) and "causal_gate" in err_details
                else {}
            )
            result = cli._failure_result(**common, mode="subprocess", reason=payload["error"]["code"],
                exit_code=payload["exit_code"], metrics=fail_metrics, artifacts=cli._common_artifacts(output_dir))
            result.error = payload["error"]
        elif proc.returncode != 0:
            raise RuntimeError(f"Bar evaluation worker exited {proc.returncode}: {proc.stderr}")
        else:
            metrics_out = payload["metrics"]
            validate_bar_content(metrics_out)
            if metrics_out["analysis_mode"] != config.get("analysis_mode"):
                raise ValueError("Bar evaluation mode differs from the request")
            horizons = config.get("horizons")
            if horizons is not None and {row["horizon_bars"] for row in metrics_out["ic_by_horizon"]} != set(horizons):
                raise ValueError("Bar evaluation horizons differ from the request")
            expected_files = {
                "evaluation_summary": "summary.json", "per_bar_labels": "per_bar_labels.csv",
                "ic_by_horizon": "ic_by_horizon.csv", "quantile_stats": "quantile_stats.csv",
            }
            if diagnostics_requested:
                expected_files.update(diagnostics_report_csv=DIAGNOSTICS_CSV,
                                      diagnostics_report_markdown=DIAGNOSTICS_MARKDOWN)
            directory = destination.resolve()
            for artifact in payload["artifacts"]:
                kind = artifact["artifact_type"]
                if kind not in expected_files:
                    raise ValueError(f"unexpected or duplicate Bar evaluation artifact: {kind}")
                path = directory / expected_files.pop(kind)
                if (Path(artifact["path"]).resolve() != path or not path.is_file()
                        or sha256_file(path) != artifact.get("sha256")):
                    raise ValueError(f"missing or hash-mismatched Bar evaluation artifact: {kind}")
            if expected_files:
                raise ValueError(f"missing Bar evaluation artifacts: {sorted(expected_files)}")
            published = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            published.pop("config")
            if published != {key: value for key, value in metrics_out.items() if key != "causal_gate"}:
                raise ValueError("Bar worker metrics differ from the published evaluation summary")
            if "causal_gate" in metrics_out:
                from tools.agent_harness.causal_gate import validate_causal_gate_evidence

                validate_causal_gate_evidence(metrics_out["causal_gate"])
            warnings_out: list[str] = []
            gate_report = metrics_out.get("causal_gate")
            if isinstance(gate_report, dict) and gate_report.get("status") == "failed_fallback":
                warnings_out.append(
                    f"Causal fast-path gate fell back to pit_replay for {gate_report.get('factor_name')}: "
                    f"{gate_report.get('failure_reason')}"
                )
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
                artifacts=cli._common_artifacts(output_dir) + payload["artifacts"], warnings=warnings_out)
    except Exception as exc:
        if phase == "results_db":
            failure = BarEvaluationStoreFailure(exc, [destination])
            result = cli._failure_result(
                **common,
                mode="results_db",
                reason=failure.code,
                exit_code=failure.exit_code,
                metrics=failure.details,
                artifacts=cli._common_artifacts(output_dir),
            )
            result.error["message"] = str(failure)
            result.error["details"].update(failure.cleanup)
        else:
            precondition = phase == "preflight" and isinstance(exc, (OSError, ValueError, TypeError, KeyError))
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
    if args.dry_run and not write_dry_run:
        return result
    if not result.ok and phase != "results_db" and not args.dry_run:
        result.error["details"].update(revoke_evaluation_completion([destination]))
    return cli._complete_result(result, args.json_out)


def _handle_read_bar_evaluations(args: argparse.Namespace, *, public_read: bool = False) -> HarnessResult:
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
    slot_id = getattr(args, "slot_id", None)
    if slot_id is not None:
        command.extend(["--slot-id", slot_id])
    artifacts = [] if public_read and (args.dry_run or not args.json_out) else cli._common_artifacts(output_dir)
    common = dict(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        started_at=started_at,
        artifacts=artifacts,
    )
    def finish(result: HarnessResult, json_out: str) -> HarnessResult:
        if public_read and (args.dry_run or not json_out):
            return result
        return cli._complete_result(result, json_out)

    try:
        cli._validate_bar_output_paths(db_path, output_dir, args.json_out)
    except (OSError, ValueError, TypeError) as exc:
        result = cli._failure_result(**{**common, "artifacts": []}, mode="preflight",
            reason="bar_evaluation_read_precondition_failed", exit_code=cli.EXIT_PRECONDITION,
            metrics={"results_db": str(db_path), "error": str(exc)})
        result.error["message"] = str(exc)
        return result
    if args.fake_result and not (public_read and args.dry_run):
        return cli._fake_result(
            action=action,
            fake_result=args.fake_result,
            output_dir=output_dir,
            run_id=args.run_id,
            started_at=started_at,
            command=command,
            json_out=args.json_out,
        )
    if args.dry_run and not public_read:
        return finish(
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
            slot_id=slot_id,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            split=args.split,
            calibration_status=args.calibration_status,
            descriptive_stats_usable=args.descriptive_stats_usable,
            significance_interpretation_usable=args.significance_interpretation_usable,
            has_trial_ledger=args.has_trial_ledger,
            factor_name=args.factor_name,
            limit=args.limit,
            offset=args.offset,
            validate_only=public_read and args.dry_run,
        )
        if args.dry_run:
            return cli._success_result(**common, mode="dry_run", reason="dry_run",
                                       metrics={"results_db": str(db_path)})
        return finish(
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
        return finish(result, json_out=args.json_out)
    except Exception as exc:
        result = cli._failure_result(
            **common,
            mode="local",
            reason="bar_evaluation_read_failed",
            exit_code=cli.EXIT_EXECUTION,
            metrics={"results_db": str(db_path), "error": str(exc)},
        )
        result.error["message"] = str(exc)
        return finish(result, json_out=args.json_out)
