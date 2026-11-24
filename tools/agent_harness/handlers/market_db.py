"""Harness entry point for physical market database profiles."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

from tools.agent_harness import runner
from utils.market_db import (
    MarketDbProfileError,
    adopt_market_db_sidecar,
    init_market_db,
    load_market_db_profile_yaml,
    market_db_sidecar_path,
    migrate_market_db_profile,
    open_market_db,
    resolve_market_db_path,
)
from utils.market_db.errors import MARKET_DB_BINDING_MISSING, MARKET_DB_OUTPUT_WRITE_FAILED, MARKET_DB_PROFILE_INVALID
from utils.market_db.profile import is_market_db_id


def _validate_outputs(output: Path, json_out: str, protected: list[Path]) -> tuple[Path, ...]:
    destinations = []
    if json_out:
        destinations.append(Path(json_out).expanduser())
    destinations.extend([output / "summary.json", output.parent.parent / "summary.json"])
    targets = tuple(dict.fromkeys(path.resolve() for path in destinations))
    sources = [source.resolve() for source in protected]
    for target in targets:
        for source in sources:
            if (source == target or source in target.parents or target in source.parents
                    or (source.exists() and target.exists() and source.samefile(target))):
                raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Result output overlaps a market database input: {target}")
        if target.exists() and not target.is_file():
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Result output must be a regular file: {target}")
        if any(other in target.parents for other in targets):
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Result outputs overlap: {target}")
        parent = next(path for path in target.parents if path.exists())
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, f"Result directory is unavailable: {parent}")
    return targets


def handle_market_db(args: argparse.Namespace) -> runner.HarnessResult:
    action, started = args.market_db_action, runner._utc_now()
    by_id = args.db is None
    command = ["market-db", action, "--db", args.db] if not by_id else [
        "market-db", action, "--market-db", args.market_db,
    ]
    declaration = getattr(args, "profile", None)
    if declaration:
        command.extend(["--profile", declaration])
    if action == "adopt":
        command.append("--sidecar")
    if args.dry_run:
        command.append("--dry-run")
    common = dict(action="market-db", mode="dry-run" if args.dry_run else "local",
                  run_id=args.run_id, command=command, started_at=started, artifacts=[])
    metrics = {"operation": action, "dry_run": args.dry_run, "operation_committed": False}
    if by_id:
        metrics["db_id"] = args.market_db
    destinations = ()
    try:
        output = runner._output_dir("market-db", runner._resolve_args_output_root(args.output_root), args.run_id)
        common["output_dir"] = output
        selection = args.market_db if by_id else Path(args.db).expanduser()
        if by_id and not is_market_db_id(selection):
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, "--market-db requires a valid db_id")
        try:
            path = resolve_market_db_path(selection)
        except MarketDbProfileError as exc:
            if exc.code == MARKET_DB_BINDING_MISSING:
                # An unmapped ID has no database path to protect.
                destinations = _validate_outputs(output, args.json_out, [])
            raise
        protected = [path, market_db_sidecar_path(path), Path(f"{path}.wal")]
        if declaration is not None:
            protected.append(Path(declaration).expanduser())
        destinations = _validate_outputs(output, args.json_out, protected)
        if not by_id:
            metrics["database_path"] = str(path)

        if action == "show":
            binding = {"market_dbs": {selection: str(path)}} if by_id else None
            with open_market_db(selection, data_bindings=binding) as (_, profile, source):
                pass
        else:
            profile = load_market_db_profile_yaml(declaration)
            source = "table"
            if action == "init":
                init_market_db(selection, profile, dry_run=args.dry_run)
                metrics["operation_committed"] = not args.dry_run
            elif action == "adopt":
                sidecar = adopt_market_db_sidecar(path, profile, dry_run=args.dry_run)
                metrics["operation_committed"] = not args.dry_run
                source = "sidecar"
                metrics["sidecar_path"] = str(sidecar)
            else:
                previous = migrate_market_db_profile(path, profile, dry_run=args.dry_run)
                metrics["previous_profile_hash"] = previous.profile_hash
                metrics["profile_changed"] = previous.profile_hash != profile.profile_hash
                metrics["operation_committed"] = not args.dry_run and metrics["profile_changed"]
                if not metrics["profile_changed"]:
                    profile = previous
        metrics.update(db_id=profile.db_id, profile=profile.as_dict(), profile_hash=profile.profile_hash, source_kind=source,
                       validation_status="passed")
        result = runner._success_result(**common, metrics=metrics)
    except (ValueError, OSError, RuntimeError, duckdb.Error) as exc:
        common.setdefault("output_dir", runner.DEFAULT_OUTPUT_ROOT / "market-db")
        message = "Market database lookup failed. See the error code for details." if by_id else str(exc)
        metrics.update(message=message, validation_status="failed")
        result = runner._failure_result(
            **common, reason=getattr(exc, "code", MARKET_DB_PROFILE_INVALID),
            exit_code=runner.EXIT_PRECONDITION, metrics=metrics,
        )
    try:
        for destination in destinations:
            runner._write_json(destination, result.to_dict())
    except (OSError, ValueError, RuntimeError) as exc:
        message = "Could not write market database result reports." if by_id else str(exc)
        result = runner._failure_result(
            **common, reason=MARKET_DB_OUTPUT_WRITE_FAILED, exit_code=runner.EXIT_EXECUTION,
            metrics={**metrics, "message": message},
        )
        for target in destinations:
            if target == destination:
                continue
            try:
                runner._write_json(target, result.to_dict())
            except (OSError, ValueError, RuntimeError):
                # Stdout still reports the persistence error and operation status.
                pass
    return result
