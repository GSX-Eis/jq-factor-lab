#!/usr/bin/env python3
"""Agent Harness entry point and compatibility exports."""
from __future__ import annotations

import argparse as argparse
import json
import math as math
import os as os
import re as re
import signal as signal
import subprocess
import sys as sys
import traceback as traceback
import uuid as uuid
from contextlib import ExitStack
from dataclasses import dataclass as dataclass
from datetime import datetime as datetime, timezone as timezone
from pathlib import Path
from typing import (
    Any as Any,
    Dict as Dict,
    List as List,
    Mapping as Mapping,
    Sequence,
)

from tools.agent_harness.d9_capacity_audit import (
    D9CapacityAuditError as D9CapacityAuditError,
    build_capacity_audit as build_capacity_audit,
    load_capacity_audit_context as load_capacity_audit_context,
    preflight_capacity_audit as preflight_capacity_audit,
    write_capacity_audit as write_capacity_audit,
)
from tools.agent_harness.handlers.analysis import (
    _handle_quanta_alpha_realize as _handle_quanta_alpha_realize,
    _handle_single_factor_analysis as _handle_single_factor_analysis,
    _handle_symbolic_factor_mining as _handle_symbolic_factor_mining,
)
from tools.agent_harness.handlers.monitor import (
    _duckdb_count_tables as _duckdb_count_tables,
    _handle_alignment_run as _handle_alignment_run,
    _handle_monitor_refresh as _handle_monitor_refresh,
)
from tools.agent_harness.handlers.storage import (
    _handle_bar_storage as _handle_bar_storage,
    _handle_factor_values_store as _handle_factor_values_store,
    _parse_strict_utc_csv_column as _parse_strict_utc_csv_column,
)
from tools.agent_harness.handlers.synthesis import (
    _handle_factor_synthesis as _handle_factor_synthesis,
    _handle_parent_factor_screen as _handle_parent_factor_screen,
)
from tools.agent_harness.handlers.time_series import (
    _bar_artifact_paths as _bar_artifact_paths,
    _handle_read_bar_evaluations as _handle_read_bar_evaluations,
    _handle_time_series_analysis as _handle_time_series_analysis,
    _validate_bar_output_paths as _validate_bar_output_paths,
)
from tools.agent_harness.metrics_parsers import (
    _ANNUAL_RETURN_RE as _ANNUAL_RETURN_RE,
    _BACKTEST_HEADER_RE as _BACKTEST_HEADER_RE,
    _BACKTEST_METRIC_RE as _BACKTEST_METRIC_RE,
    _BENCHMARK_DETAIL_KEYS as _BENCHMARK_DETAIL_KEYS,
    _BENCHMARK_REPORT_RE as _BENCHMARK_REPORT_RE,
    _IC_OVERALL_RE as _IC_OVERALL_RE,
    _IC_PERIOD_RE as _IC_PERIOD_RE,
    _IR_OVERALL_RE as _IR_OVERALL_RE,
    _IR_PERIOD_RE as _IR_PERIOD_RE,
    _LOG_PREFIX_RE as _LOG_PREFIX_RE,
    _NEUTRALIZED_INPUT_WARNING_TOKEN as _NEUTRALIZED_INPUT_WARNING_TOKEN,
    _SYNTHESIS_DIAGNOSTICS_RE as _SYNTHESIS_DIAGNOSTICS_RE,
    _VECTORBT_RESULT_SUMMARIES_RE as _VECTORBT_RESULT_SUMMARIES_RE,
    _assemble_factor_synthesis_metrics as _assemble_factor_synthesis_metrics,
    _build_backtest_signature as _build_backtest_signature,
    _coerce_signature_float as _coerce_signature_float,
    _compact_benchmark_report as _compact_benchmark_report,
    _extract_factor_synthesis_metrics as _extract_factor_synthesis_metrics,
    _extract_neutralized_input_warnings as _extract_neutralized_input_warnings,
    _extract_symbolic_factor_mining_metrics as _extract_symbolic_factor_mining_metrics,
    _parse_benchmark_reports as _parse_benchmark_reports,
    _parse_factor_synthesis_backtest_runs as _parse_factor_synthesis_backtest_runs,
    _parse_factor_synthesis_diagnostics as _parse_factor_synthesis_diagnostics,
    _parse_factor_synthesis_ic_summary as _parse_factor_synthesis_ic_summary,
    _parse_vectorbt_result_summaries as _parse_vectorbt_result_summaries,
    _pct_to_ratio as _pct_to_ratio,
    _strip_log_prefix as _strip_log_prefix,
)
from tools.agent_harness.parser import build_parser as build_parser
from tools.agent_harness.run_result import (
    config_sha256 as config_sha256,
    read_execution_result as read_execution_result,
)
from tools.agent_harness.runner import (
    DEFAULT_MONITOR_DB as DEFAULT_MONITOR_DB,
    DEFAULT_OUTPUT_ROOT as DEFAULT_OUTPUT_ROOT,
    EXIT_EXECUTION as EXIT_EXECUTION,
    EXIT_PRECONDITION as EXIT_PRECONDITION,
    EXIT_QUALITY_GATE as EXIT_QUALITY_GATE,
    EXIT_SUCCESS as EXIT_SUCCESS,
    REPO_ROOT as REPO_ROOT,
    HarnessResult as HarnessResult,
    _bound_result_metrics as _bound_result_metrics,
    _build_extra_metrics as _build_extra_metrics,
    _classify_subprocess_failure as _classify_subprocess_failure,
    _common_artifacts as _common_artifacts,
    _complete_result as _complete_result,
    _failure_result as _failure_result,
    _fake_result as _fake_result,
    _kill_process_group as _kill_process_group,
    _output_dir as _output_dir,
    _resolve_args_output_root as _resolve_args_output_root,
    _resolve_repo_relative_path as _resolve_repo_relative_path,
    _run_subprocess as _run_subprocess,
    _run_subprocess_body as _run_subprocess_body,
    _safe_read_json as _safe_read_json,
    _success_result as _success_result,
    _tail as _tail,
    _utc_now as _utc_now,
    _write_json as _write_json,
)


def main(argv: Sequence[str] | None = None, *, public_bar_cli: bool = False) -> int:
    parser = build_parser(public_bar_cli=True) if public_bar_cli else build_parser()
    args = parser.parse_args(argv)

    if args.command == "outputs-gc":
        command = [args.task_manager, "--task-root", str(Path(args.task_root).resolve()), "--json",
                   "state", "gc", "--scope", args.scope, "--completed-hours", str(args.completed_hours),
                   "--failed-days", str(args.failed_days), "--cache-days", str(args.cache_days)]
        if args.execute:
            command.append("--execute")
        try:
            return subprocess.run(command, check=False).returncode
        except OSError as exc:
            parser.error(str(exc))
    if args.command in ('build-bars', 'audit-bars', 'import-bars', 'build-roll-schedule'):
        result = _handle_bar_storage(args)
    elif args.command in ('write-factor-values', 'read-factor-values', 'verify-factor-values'):
        result = _handle_factor_values_store(args)
    elif args.command == "alignment-run":
        result = _handle_alignment_run(args)
    elif args.command == "time-series-analysis":
        if public_bar_cli:
            with ExitStack() as output:
                result = _handle_time_series_analysis(args, public_output=output)
        else:
            result = _handle_time_series_analysis(args)
    elif args.command == "read-bar-evaluations":
        result = _handle_read_bar_evaluations(args)
    elif args.command == "single-factor-analysis":
        if args.materialize_only and not args.factor:
            parser.error("--materialize-only requires --factor")
        result = _handle_single_factor_analysis(args)
    elif args.command == "symbolic-factor-mining":
        result = _handle_symbolic_factor_mining(args)
    elif args.command == "quanta-alpha-realize":
        result = _handle_quanta_alpha_realize(args)
    elif args.command == "factor-synthesis":
        result = _handle_factor_synthesis(args)
    elif args.command == "parent-factor-screen":
        result = _handle_parent_factor_screen(args)
    elif args.command == "monitor-refresh":
        result = _handle_monitor_refresh(args)
    else:
        raise ValueError(f"unsupported command: {args.command}")

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
