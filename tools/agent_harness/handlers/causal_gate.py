"""Handle automatic causal verification gate checks for Slot and FactorBase factors."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from tools.agent_harness import runner
from tools.agent_harness._cli_compat import _cli_bindings
from tools.agent_harness.causal_gate import verify_factor_causality
from tools.agent_harness.runner import HarnessResult


def _resolve_slot_and_defaults(
    *,
    slot_arg: str | None,
    slot_id: str | None,
    slot_db: str | None,
    market_db: str | None,
    base_dir: Path,
    stack: ExitStack,
):
    from slots import build_slot_from_config
    from utils.duckdb_results_writer import connect_read_only_with_retry
    from utils.market_db.research import open_research_slot, resolve_research_slot

    frozen, resolved_base = resolve_research_slot(slot_arg, slot_id, base_dir)
    if market_db is not None:
        slot, _ = stack.enter_context(
            open_research_slot(market_db, frozen, resolved_base, declared_db=slot_db)
        )
    else:
        connection = None
        if frozen.bar_feed.provider in ("jq_ashare_local", "cn_futures", "global_futures"):
            if not isinstance(slot_db, str) or not slot_db.strip():
                raise ValueError("Native Slot causal verification requires --slot-db")
            connection = connect_read_only_with_retry(Path(slot_db).expanduser().resolve())
            stack.callback(connection.close)
            connection.execute("BEGIN TRANSACTION")
        slot = build_slot_from_config(frozen, connection=connection, base_dir=resolved_base)
    return slot, frozen


def _default_instruments_for_slot(slot: Any, explicit: str | None) -> list[str]:
    if explicit:
        items = [item.strip() for item in explicit.split(",") if item.strip()]
        if not items or len(set(items)) != len(items):
            raise ValueError("--instruments must contain distinct nonempty instrument IDs")
        return items
    static = getattr(slot.universe, "static", None)
    if isinstance(static, dict) and static:
        first_pool = next(iter(static.values()))
        if first_pool:
            return list(first_pool)
    specs = getattr(slot.universe, "specs", None)
    if isinstance(specs, dict) and specs:
        return sorted(specs.keys())
    root = getattr(slot.bar_feed, "root", None)
    if isinstance(root, str) and root:
        return [root]
    raise ValueError("--instruments is required when the Slot universe has no static instruments")


def _handle_verify_factor_causality(args: argparse.Namespace) -> HarnessResult:
    from backtest_and_analysis.factor_loader import FactorLoader

    cli = _cli_bindings(__name__, runner)
    action = "verify-factor-causality"
    output_dir = cli._output_dir(
        action, cli._resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = cli._utc_now()
    command = [action]
    if args.factor:
        command.extend(["--factor", args.factor])
    if getattr(args, "slot", None):
        command.extend(["--slot", args.slot])
    artifacts = cli._common_artifacts(output_dir)
    common = dict(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        started_at=started_at,
        artifacts=artifacts,
    )

    if args.fake_result:
        return cli._fake_result(**common, fake_result=args.fake_result, json_out=args.json_out)

    if args.dry_run:
        return cli._complete_result(
            cli._success_result(
                **common,
                mode="dry_run",
                reason="dry_run",
                metrics={
                    "planned_factor": args.factor or "",
                    "planned_slot": getattr(args, "slot", None) or "",
                    "on_failure": args.on_failure,
                    "n_truncations": args.n_truncations,
                    "n_pit_samples": args.n_pit_samples,
                    "seed": args.seed,
                },
            ),
            json_out=args.json_out,
        )

    try:
        if not args.factor:
            raise ValueError("--factor is required")
        factor_path = cli._resolve_repo_relative_path(args.factor)
        if not factor_path.is_file():
            raise FileNotFoundError(f"Factor file not found: {factor_path}")
        classes = FactorLoader.load_factor_class_from_file(str(factor_path), args.factor_class)
        if not classes:
            raise ValueError(f"No FactorBase subclasses found in {factor_path}")

        with ExitStack() as stack:
            slot_arg = getattr(args, "slot", None)
            market_db = getattr(args, "market_db", None)
            slot_db = getattr(args, "slot_db", None)
            slot = None
            frozen = None
            if slot_arg is not None or market_db is not None:
                slot, frozen = _resolve_slot_and_defaults(
                    slot_arg=slot_arg,
                    slot_id=getattr(args, "slot_id", None),
                    slot_db=slot_db,
                    market_db=market_db,
                    base_dir=factor_path.parent,
                    stack=stack,
                )
            else:
                raise ValueError("--slot or --market-db is required for verify-factor-causality")

            sessions = slot.calendar.sessions()
            if sessions.empty:
                raise ValueError("Slot calendar has no sessions")
            start_date = args.start_date or sessions[0].strftime("%Y-%m-%d")
            end_date = args.end_date or sessions[-1].strftime("%Y-%m-%d")
            frequency = args.frequency
            if frequency is None:
                feed_freq = getattr(slot.bar_feed, "frequency", None)
                frequency = feed_freq.canonical if feed_freq is not None else "1d"
            instruments = _default_instruments_for_slot(slot, args.instruments)

            reports: list[dict[str, Any]] = []
            warnings: list[str] = []
            for cls in classes:
                factor = cls(
                    None,
                    cls.__name__,
                    start_date,
                    end_date,
                    stock_pool=instruments,
                    slot=slot,
                    data_frequency=frequency,
                    analysis_mode=args.analysis_mode,
                    session_view=args.view,
                    slot_execution_mode=args.slot_execution_mode,
                )
                gate = verify_factor_causality(
                    factor,
                    n_truncations=args.n_truncations,
                    n_pit_samples=args.n_pit_samples,
                    seed=args.seed,
                    on_failure=args.on_failure,
                    apply_to_factor=True,
                )
                report = gate.to_dict()
                reports.append(report)
                if gate.status == "failed_fallback":
                    warnings.append(
                        f"{cls.__name__} failed causal gate ({gate.failure_reason}); "
                        f"falling back to pit_replay"
                    )

        primary = reports[0]
        all_ok = all(item["ok"] for item in reports)
        metrics: dict[str, Any] = {
            "causal_gate": primary,
            "causal_gates": reports,
            "slot_id": slot.slot_id,
            "frequency": frequency,
            "start_date": start_date,
            "end_date": end_date,
            "instruments": instruments,
        }
        if not all_ok:
            failed = next(item for item in reports if not item["ok"])
            metrics["failure_reason"] = "causal_fast_path_gate_failed"
            result = cli._failure_result(
                **common,
                mode="local",
                reason="causal_fast_path_gate_failed",
                exit_code=cli.EXIT_QUALITY_GATE,
                metrics=metrics,
                warnings=warnings,
            )
            result.error = {
                "code": "causal_fast_path_gate_failed",
                "message": (
                    f"Causal fast-path gate rejected {failed['factor_name']}: "
                    f"{failed['failure_reason']}"
                ),
                "details": {"causal_gate": failed},
            }
            return cli._complete_result(result, json_out=args.json_out)

        return cli._complete_result(
            cli._success_result(
                **common,
                mode="local",
                metrics=metrics,
                warnings=warnings,
            ),
            json_out=args.json_out,
        )
    except (OSError, ValueError, TypeError, LookupError, ImportError) as exc:
        result = cli._failure_result(
            **common,
            mode="preflight",
            reason="causal_gate_precondition_failed",
            exit_code=cli.EXIT_PRECONDITION,
            metrics={"exception_type": type(exc).__name__, "error": str(exc)},
        )
        result.error["message"] = str(exc)
        return cli._complete_result(result, json_out=args.json_out)
    except Exception as exc:
        result = cli._failure_result(
            **common,
            mode="local",
            reason="causal_gate_execution_failed",
            exit_code=cli.EXIT_EXECUTION,
            metrics={"exception_type": type(exc).__name__, "error": str(exc)},
        )
        result.error["message"] = str(exc)
        return cli._complete_result(result, json_out=args.json_out)
