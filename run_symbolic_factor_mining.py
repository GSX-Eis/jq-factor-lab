#!/usr/bin/env python3
"""Standalone entry point for symbolic factor mining."""

from __future__ import annotations

import argparse
import json

from factor_mining.symbolic.pipeline import (
    EXIT_EXECUTION,
    materialize_rerank_candidate,
    run_symbolic_mining_pipeline,
)
from factor_mining.symbolic.preflight import (
    DATA_PREFLIGHT_ERROR_CODE,
    DataPreflightError,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run symbolic factor mining and generate a FactorBase factor file"
    )
    parser.add_argument(
        "--config", required=True, help="Path to symbolic mining JSON config"
    )
    parser.add_argument("--run-id", default="", help="Optional stable run id")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip generated factor smoke validation",
    )
    parser.add_argument(
        "--novelty-context",
        default="",
        help="Run-local symbolic novelty context JSON",
    )
    parser.add_argument(
        "--marginal-context",
        default="",
        help="Run-local symbolic marginal-aware search context JSON",
    )
    parser.add_argument(
        "--materialize-candidate-rank",
        type=int,
        default=0,
        help="Materialize and validate a rerank candidate rank from an existing run",
    )
    parser.add_argument(
        "--factor-path",
        default="",
        help="Override final factor path for candidate materialization",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.materialize_candidate_rank:
            if not args.run_id:
                raise ValueError(
                    "--run-id is required with --materialize-candidate-rank"
                )
            summary = materialize_rerank_candidate(
                args.config,
                run_id=args.run_id,
                rank=int(args.materialize_candidate_rank),
                factor_path=args.factor_path or None,
            )
        else:
            if not args.novelty_context:
                raise ValueError("--novelty-context is required")
            summary = run_symbolic_mining_pipeline(
                args.config,
                run_id=args.run_id or None,
                skip_validation=args.skip_validation,
                novelty_context_path=args.novelty_context,
                marginal_context_path=args.marginal_context or None,
            )
    except DataPreflightError as exc:
        summary = {
            "ok": False,
            "exit_code": EXIT_EXECUTION,
            "failure_reason": DATA_PREFLIGHT_ERROR_CODE,
            "error": {"code": DATA_PREFLIGHT_ERROR_CODE, "message": str(exc)},
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return EXIT_EXECUTION
    except Exception as exc:
        summary = {
            "ok": False,
            "exit_code": EXIT_EXECUTION,
            "error": {"code": "symbolic_mining_failed", "message": str(exc)},
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return EXIT_EXECUTION

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(summary.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
