"""Harness entry points for offline strategy candidates and replay evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.agent_harness import runner
from tools.agent_harness.run_result import config_sha256, execution_result

if TYPE_CHECKING:
    from factor_synthesis.strategy_bundle import StrategyBundle
    from targets.contracts import AlignmentReport, DeployTarget


def _read_json(path: str) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _read_positions(path: str) -> list[dict[str, Any]]:
    from targets.signal_export.schema import read_jsonl_records, read_parquet_records

    source = Path(path).expanduser()
    if source.suffix.lower() == ".parquet":
        return read_parquet_records(source, label=str(source))
    if source.suffix.lower() == ".jsonl":
        return read_jsonl_records(source, label=str(source))
    raise ValueError("Position file must have a .parquet or .jsonl suffix")


def _export(
    args: argparse.Namespace, target: DeployTarget, bundle: StrategyBundle, payload: dict[str, Any],
) -> None:
    from targets.contracts import (
        DEPLOY_MANIFEST_FILENAME,
        TARGET_ARTIFACT_INTEGRITY_FAILED,
        DeployTargetError,
    )

    options = {}
    if args.target == "jq_cloud":
        if args.target_positions or args.current_positions:
            raise ValueError("--target-positions and --current-positions require signal_export")
        options.update(factor_files=args.factor_file, requirements_path=args.requirements)
    elif args.factor_file or args.requirements:
        raise ValueError("--factor-file and --requirements require jq_cloud")

    compatibility = target.check_compatibility(bundle, bundle_dir=args.bundle_dir, **options)
    payload["metrics"]["compatibility"] = compatibility.to_dict()
    if not compatibility.compatible:
        issue = compatibility.issues[0]
        raise DeployTargetError(issue.message, code=issue.code)

    if args.target_positions:
        options["target_positions"] = _read_positions(args.target_positions)
    if args.current_positions:
        positions = _read_json(args.current_positions)
        if not isinstance(positions, dict):
            raise ValueError("--current-positions requires an instrument-to-weight JSON object")
        options["current_positions"] = positions

    root = Path(args.output_dir).expanduser().resolve()
    protected = (runner.REPO_ROOT / "state",
                 runner.REPO_ROOT / "scripts/build/cloud_runtime", runner.REPO_ROOT / "factors",
                 runner.REPO_ROOT / "reports/NERE-936", runner.REPO_ROOT / "reports/NERE-962", Path("/Volumes"))
    if any(root.is_relative_to(path.resolve()) for path in protected):
        raise DeployTargetError(f"Output directory overlaps a protected input: {root}",
                                code=TARGET_ARTIFACT_INTEGRITY_FAILED)
    manifest = target.export(bundle, root, bundle_dir=args.bundle_dir, **options)
    payload["metrics"]["deploy_manifest"] = manifest.to_dict()
    manifest_path = root / DEPLOY_MANIFEST_FILENAME
    payload["artifacts"].append(dict(
        artifact_type="deploy_manifest", path=str(manifest_path), sha256=config_sha256(manifest_path),
    ))
    payload["artifacts"].extend(
        dict(entry.to_dict(), path=str(root / entry.path)) for entry in manifest.artifacts
    )


def _verify(args: argparse.Namespace, target: DeployTarget, bundle: StrategyBundle) -> AlignmentReport:
    if args.target == "signal_export":
        if args.replay_inputs or args.reference_factors or args.trusted_manifest_hash:
            raise ValueError("--replay-inputs, --reference-factors and --trusted-manifest-hash require jq_cloud")
        options = {
            "reference_positions": _read_positions(args.reference_positions) if args.reference_positions else None,
        }
    else:
        import pandas as pd

        virtual = None
        if args.replay_inputs:
            data = _read_json(args.replay_inputs)
            if (not isinstance(data, dict)
                    or data.keys() - {"stocks", "asof", "bars_by_frequency", "pool_history"}
                    or not {"stocks", "asof", "bars_by_frequency"} <= data.keys()
                    or not isinstance(data["bars_by_frequency"], dict)):
                raise ValueError("Replay JSON requires stocks, asof and bars_by_frequency, plus optional pool_history")
            virtual = dict(data, bars_by_frequency={
                frequency: pd.DataFrame(rows) for frequency, rows in data["bars_by_frequency"].items()
            })
        weights = _read_json(args.reference_positions) if args.reference_positions else None
        if args.reference_positions and not isinstance(weights, dict):
            raise ValueError("JoinQuant reference positions require an instrument-to-weight JSON object")
        options = {
            "trusted_manifest_hash": args.trusted_manifest_hash,
            "virtual_cloud_inputs": virtual,
            "local_factor_values": pd.read_parquet(Path(args.reference_factors).expanduser())
            if args.reference_factors else None,
            "local_positions": weights,
        }
    return target.verify_alignment(bundle, args.artifact_dir, bundle_dir=args.bundle_dir, **options)


def handle_deploy(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from factor_synthesis.strategy_bundle import verify_strategy_bundle
    from targets import DeployTargetError, TARGET_BUNDLE_INVALID, build_default_target_registry

    action = args.command
    output = runner._output_dir(action, runner._resolve_args_output_root(""), "")
    request = output / "request.json"
    result = output / "execution_result.json"
    runner._write_json(request, vars(args))
    exit_code = runner.EXIT_SUCCESS
    try:
        with execution_result(
            result, command=action, run_id=output.parent.parent.name, execution_id=output.name,
            config_path=request,
        ) as payload:
            payload["execution"]["expected"] = 1
            payload["metrics"]["target_id"] = args.target
            try:
                try:
                    bundle = verify_strategy_bundle(args.bundle_dir)
                except (ValueError, OSError, RuntimeError) as exc:
                    raise DeployTargetError(str(exc), code=TARGET_BUNDLE_INVALID) from exc
                payload["metrics"]["bundle_hash"] = bundle.bundle_hash
                target = build_default_target_registry().get(args.target)
                if action == "export-strategy":
                    _export(args, target, bundle, payload)
                else:
                    report = _verify(args, target, bundle)
                    payload["metrics"]["alignment_report"] = report.to_dict()
                    if not report.passed:
                        exit_code = runner.EXIT_QUALITY_GATE
                        failure = next(check for check in report.checks if not check.passed)
                        raise DeployTargetError(failure.reason, code=report.error_code)
            except Exception as exc:
                payload["execution"]["failed"] = 1
                payload["metrics"]["error_code"] = getattr(exc, "code", "DEPLOY_TARGET_REQUEST_FAILED")
                if exit_code == runner.EXIT_SUCCESS:
                    exit_code = (runner.EXIT_PRECONDITION if isinstance(exc, (ValueError, OSError))
                                 else runner.EXIT_EXECUTION)
                raise
            payload["execution"]["completed"] = 1
    except Exception:
        if exit_code == runner.EXIT_SUCCESS or not result.is_file():
            raise
    return json.loads(result.read_text(encoding="utf-8")), exit_code
