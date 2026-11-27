"""Handle monitor inspection and alignment evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from tools.agent_harness import runner
from tools.agent_harness._cli_compat import _cli_bindings
from tools.agent_harness.result_content import validate_monitor_content
from tools.agent_harness.runner import HarnessResult


def _duckdb_count_tables(db_path: Path, *, bindings=None) -> Dict[str, Any]:
    from utils.bar_evaluation_store import summarize_bar_evaluation_connection

    try:
        from utils.data_store import connect_read_only
    except Exception:
        return {
            "backend": "duckdb",
            "available": False,
            "table_count": 0,
            "factor_count": 0,
        }
    conn = connect_read_only(db_path, bindings=bindings, allow_labels=True)
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {row[0] for row in tables}
        factor_count = 0
        if "factor_analysis_results" in table_names:
            factor_count = int(
                conn.execute("SELECT COUNT(*) FROM factor_analysis_results").fetchone()[
                    0
                ]
            )
        return {
            "backend": "duckdb",
            "available": True,
            "table_count": len(tables),
            "factor_count": factor_count,
            # Includes optional split metrics even when a legacy Bar schema is still present.
            "bar_evaluation": summarize_bar_evaluation_connection(conn, table_names=table_names),
        }
    finally:
        conn.close()


def _handle_monitor_refresh(args: argparse.Namespace) -> HarnessResult:
    cli = _cli_bindings(__name__, runner)
    action = "monitor-refresh"
    output_dir = cli._output_dir(
        action, cli._resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = cli._utc_now()
    command = ["monitor_refresh", "--database-path", args.database_path]
    artifacts = cli._common_artifacts(output_dir)
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
                metrics={"database_path": args.database_path, "mode": "summary"},
                artifacts=artifacts,
                reason="dry_run",
            ),
            json_out=args.json_out,
        )

    from utils.data_store import binding_contract, resolve_connection_binding

    bindings = binding_contract()
    db_path = Path(args.database_path)
    if bindings:
        destination, bindings = resolve_connection_binding(db_path, bindings, allow_labels=True)
        if Path(destination).resolve() != Path(bindings.get("output_db", "")).resolve():
            raise ValueError("monitor-refresh requires an output database binding")
        db_path = Path(destination)
    if not db_path.exists():
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="inspection",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="monitor_database_missing",
                exit_code=cli.EXIT_PRECONDITION,
                metrics={"missing_path": args.database_path},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )

    if not bindings and db_path.suffix != ".duckdb":
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="inspection",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="monitor_requires_duckdb",
                exit_code=cli.EXIT_PRECONDITION,
                metrics={"database_path": str(db_path)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    metrics = cli._duckdb_count_tables(db_path, bindings=bindings)
    metrics["database_path"] = str(db_path)
    if not metrics.get("available"):
        return cli._complete_result(
            cli._failure_result(
                action=action,
                mode="inspection",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="monitor_backend_unavailable",
                exit_code=cli.EXIT_PRECONDITION,
                metrics=metrics,
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    try:
        validate_monitor_content(metrics)
    except ValueError as exc:
        return cli._complete_result(cli._failure_result(
            action=action, mode="inspection", output_dir=output_dir, run_id=args.run_id,
            command=command, started_at=started_at, reason="invalid_monitor_result",
            exit_code=cli.EXIT_EXECUTION, metrics={"error": str(exc)}, artifacts=artifacts,
        ), json_out=args.json_out)
    return cli._complete_result(
        cli._success_result(
            action=action,
            mode="inspection",
            output_dir=output_dir,
            run_id=args.run_id,
            command=command,
            started_at=started_at,
            metrics=metrics,
            artifacts=artifacts,
        ),
        json_out=args.json_out,
    )


def _handle_alignment_run(args):
    from tools.agent_harness.alignment import run

    cli = _cli_bindings(__name__, runner)
    started = cli._utc_now()
    try:
        payload = run(args, cli.REPO_ROOT)
        ok = payload.get("status") == "PASS"
        failed = payload.get("failed_stage", "")
        code = (
            cli.EXIT_SUCCESS
            if ok
            else (
                cli.EXIT_QUALITY_GATE
                if failed
                in (
                    "compare",
                    "factor",
                    "preprocess",
                    "score",
                    "targets",
                    "orders",
                    "cash",
                    "nav",
                    "inputs",
                )
                else cli.EXIT_EXECUTION
                if "execution" in failed
                or failed
                in (
                    "upload",
                    "download",
                    "wait_kernel",
                    "notebook",
                    "cloud_start",
                    "cloud_freshness",
                    "local_analysis",
                )
                else cli.EXIT_PRECONDITION
            )
        )
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "error_code": str(exc),
            "failed_stage": "preflight",
            "layers": {},
        }
        ok, code = False, cli.EXIT_PRECONDITION
    result = cli.HarnessResult(
        action="alignment-run",
        ok=ok,
        exit_code=code,
        status="succeeded" if ok else "failed",
        reason=payload.get("error_code") or payload.get("failed_stage") or "ok",
        mode="subprocess" if args.real_cloud else "preflight",
        output_dir=Path(args.output_dir).resolve(),
        command=["alignment-run"],
        metrics=payload,
        artifacts=[
            {"artifact_type": "alignment_evidence", "path": p, "sha256": h}
            for p, h in payload.get("artifact_hashes", {}).items()
        ],
        warnings=[],
        started_at=started,
        finished_at=cli._utc_now(),
        run_id=args.run_id,
    )
    return cli._complete_result(result, json_out=args.json_out)
