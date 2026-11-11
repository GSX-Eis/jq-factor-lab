#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from tools.agent_harness.run_result import config_sha256, read_execution_result

from tools.agent_harness.d9_capacity_audit import (
    D9CapacityAuditError,
    build_capacity_audit,
    load_capacity_audit_context,
    preflight_capacity_audit,
    write_capacity_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "agent_harness"
DEFAULT_MONITOR_DB = REPO_ROOT / "data" / "duckdb" / "analysis_results.duckdb"

EXIT_SUCCESS = 0
EXIT_PRECONDITION = 2
EXIT_EXECUTION = 3
EXIT_QUALITY_GATE = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    from utils.local_outputs import write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _resolve_repo_relative_path(raw_path: str) -> Path:
    candidate = Path(str(raw_path or "")).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


_LOG_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - .*? - [A-Z]+ - "
)
_IC_PERIOD_RE = re.compile(
    r"^\s*(?P<period>\d+D): 均值=(?P<mean>-?\d+(?:\.\d+)?), 标准差=(?P<std>-?\d+(?:\.\d+)?), "
    r"范围=\[(?P<min>-?\d+(?:\.\d+)?), (?P<max>-?\d+(?:\.\d+)?)\], 正值占比=(?P<pos>[\d.]+)%, 样本数=(?P<n>\d+)"
)
_IC_OVERALL_RE = re.compile(
    r"^\s*🎯 (?P<factor>.+?) 整体IC: (?P<value>-?\d+(?:\.\d+)?)$"
)
_IR_PERIOD_RE = re.compile(r"^\s*(?P<period>\d+D): IR=(?P<value>-?\d+(?:\.\d+)?)$")
_IR_OVERALL_RE = re.compile(
    r"^\s*🎯 (?P<factor>.+?) 整体IR: (?P<value>-?\d+(?:\.\d+)?)$"
)
_SYNTHESIS_DIAGNOSTICS_RE = re.compile(
    r"ALGORITHM_SYNTHESIS_DIAGNOSTICS:\s*(?P<payload>\{.*\})\s*$"
)
_VECTORBT_RESULT_SUMMARIES_RE = re.compile(
    r"VECTORBT_RESULT_SUMMARIES:\s*(?P<payload>\[.*\])\s*$"
)
_BENCHMARK_REPORT_RE = re.compile(
    r"ALGORITHM_BENCHMARK_REPORT:\s*(?P<payload>\{.*\})\s*$"
)
_BENCHMARK_DETAIL_KEYS = {
    "excess_names",
    "excess_order_history",
    "holding_count_violation_dates",
    "holding_count_violation_details",
}
_BACKTEST_HEADER_RE = re.compile(
    r"^📊 VectorBT 回测结果(?: \(top_pct=(?P<top_pct>[^)]+)\)| \(top_k=(?P<top_k>[^)]+)\))?:$"
)
_BACKTEST_METRIC_RE = {
    "annual_return": re.compile(r"^\s*年化收益: (?P<value>-?\d+(?:\.\d+)?)%$"),
    "sharpe_ratio": re.compile(r"^\s*夏普比率: (?P<value>-?\d+(?:\.\d+)?)$"),
    "calmar_ratio": re.compile(r"^\s*Calmar比率: (?P<value>-?\d+(?:\.\d+)?)$"),
    "sortino_ratio": re.compile(r"^\s*Sortino比率: (?P<value>-?\d+(?:\.\d+)?)$"),
    "max_drawdown": re.compile(r"^\s*最大回撤: (?P<value>-?\d+(?:\.\d+)?)%$"),
    "total_return": re.compile(r"^\s*总收益: (?P<value>-?\d+(?:\.\d+)?)%$"),
    "trade_count": re.compile(r"^\s*交易次数: (?P<value>\d+)$"),
}
_ANNUAL_RETURN_RE = re.compile(r"^\s*(?P<year>\d{4}): (?P<value>-?\d+(?:\.\d+)?)%$")
_NEUTRALIZED_INPUT_WARNING_TOKEN = "[中性化输入缺失]"


def _strip_log_prefix(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", line.rstrip("\n"))


def _extract_neutralized_input_warnings(stdout: str, stderr: str) -> List[str]:
    warnings: List[str] = []
    seen: set[str] = set()
    for text in (stdout, stderr):
        for raw_line in text.splitlines():
            line = _strip_log_prefix(raw_line).strip()
            if _NEUTRALIZED_INPUT_WARNING_TOKEN not in line or line in seen:
                continue
            warnings.append(line)
            seen.add(line)
    return warnings


def _pct_to_ratio(value: str) -> float:
    return float(value) / 100.0


def _coerce_signature_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _build_backtest_signature(run: Mapping[str, Any]) -> Dict[str, Any]:
    annual_returns = run.get("annual_returns")
    yearly: Dict[str, float] = {}
    if isinstance(annual_returns, Mapping):
        for year, value in annual_returns.items():
            numeric = _coerce_signature_float(value)
            if numeric is not None:
                yearly[str(year)] = numeric
    signature = {
        "annual_return": _coerce_signature_float(run.get("annual_return")),
        "total_return": _coerce_signature_float(run.get("total_return")),
        "max_drawdown": _coerce_signature_float(run.get("max_drawdown")),
        "sharpe_ratio": _coerce_signature_float(run.get("sharpe_ratio")),
        "sortino_ratio": _coerce_signature_float(run.get("sortino_ratio")),
        "top_pct": _coerce_signature_float(run.get("top_pct")),
        "yearly_returns": yearly,
    }
    return signature


def _parse_factor_synthesis_ic_summary(text: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = _strip_log_prefix(raw_line)
        if not line.strip():
            continue

        if "IC数据分析:" in line:
            if current:
                sections.append(current)
            factor_name = line.split("IC数据分析:", 1)[0].replace("📊", "").strip()
            current = {
                "factor_name": factor_name,
                "periods": {},
                "ir_periods": {},
            }
            continue

        if current is None:
            continue

        if "分位数收益分析:" in line or "分析完成" in line:
            sections.append(current)
            current = None
            continue

        match = _IC_PERIOD_RE.match(line)
        if match:
            current["periods"][match.group("period")] = {
                "mean_ic": float(match.group("mean")),
                "std_ic": float(match.group("std")),
                "min_ic": float(match.group("min")),
                "max_ic": float(match.group("max")),
                "positive_ratio": _pct_to_ratio(match.group("pos")),
                "sample_count": int(match.group("n")),
            }
            continue

        match = _IC_OVERALL_RE.match(line)
        if match:
            current["overall_ic"] = float(match.group("value"))
            current["factor_name"] = match.group("factor").strip()
            continue

        match = _IR_PERIOD_RE.match(line)
        if match:
            current["ir_periods"][match.group("period")] = float(match.group("value"))
            continue

        match = _IR_OVERALL_RE.match(line)
        if match:
            current["overall_ir"] = float(match.group("value"))
            current["factor_name"] = match.group("factor").strip()
            continue

    if current:
        sections.append(current)

    return sections


def _parse_factor_synthesis_backtest_runs(text: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    in_annual_returns = False

    for raw_line in text.splitlines():
        line = _strip_log_prefix(raw_line)
        if not line.strip():
            continue

        header = _BACKTEST_HEADER_RE.match(line)
        if header:
            if current:
                runs.append(current)
            current = {
                "selection_mode": "top_pct"
                if header.group("top_pct") is not None
                else "top_k",
                "top_pct": float(header.group("top_pct"))
                if header.group("top_pct") is not None
                else None,
                "top_k": int(header.group("top_k"))
                if header.group("top_k") is not None
                else None,
                "annual_returns": {},
            }
            in_annual_returns = False
            continue

        if current is None:
            continue

        metric_matched = False
        for field_name, pattern in _BACKTEST_METRIC_RE.items():
            match = pattern.match(line)
            if match:
                value = match.group("value")
                if field_name == "trade_count":
                    current[field_name] = int(value)
                elif field_name == "max_drawdown":
                    current[field_name] = _pct_to_ratio(value)
                elif field_name in {"annual_return", "total_return"}:
                    current[field_name] = _pct_to_ratio(value)
                else:
                    current[field_name] = float(value)
                metric_matched = True
                break
        if metric_matched:
            continue

        if line.strip() == "📅 年度收益分解:":
            in_annual_returns = True
            continue

        if in_annual_returns:
            match = _ANNUAL_RETURN_RE.match(line)
            if match:
                current["annual_returns"][int(match.group("year"))] = _pct_to_ratio(
                    match.group("value")
                )
                continue
            in_annual_returns = False

        # Any non-metric line ends the current block.
        if current:
            runs.append(current)
            current = None

    if current:
        runs.append(current)

    return runs


def _parse_factor_synthesis_diagnostics(text: str) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = _strip_log_prefix(raw_line)
        match = _SYNTHESIS_DIAGNOSTICS_RE.search(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            diagnostics.append(payload)
    return diagnostics


def _parse_vectorbt_result_summaries(text: str) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = _strip_log_prefix(raw_line)
        match = _VECTORBT_RESULT_SUMMARIES_RE.search(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    summaries.append(item)
    return summaries


def _parse_benchmark_reports(text: str) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = _strip_log_prefix(raw_line)
        match = _BENCHMARK_REPORT_RE.search(line)
        if not match:
            continue
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            reports.append(payload)
    return reports


def _compact_benchmark_report(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _compact_benchmark_report(item)
            for key, item in value.items()
            if key not in _BENCHMARK_DETAIL_KEYS
        }
    if isinstance(value, list):
        return [_compact_benchmark_report(item) for item in value]
    return value


def _assemble_factor_synthesis_metrics(
    config_payload, ic_sections, backtest_runs, diagnostics, benchmark_reports,
    neutralized_input_warnings,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if neutralized_input_warnings:
        metrics["neutralized_input_warnings"] = neutralized_input_warnings
        metrics["neutralized_input_warning_count"] = len(neutralized_input_warnings)
        metrics["_warnings"] = neutralized_input_warnings

    if ic_sections:
        ic_summary = ic_sections[-1]
        metrics["ic_summary"] = ic_summary
        if isinstance(ic_summary, Mapping):
            overall_ic = ic_summary.get("overall_ic")
            overall_ir = ic_summary.get("overall_ir")
            if overall_ic is not None:
                metrics["overall_ic"] = overall_ic
            if overall_ir is not None:
                metrics["overall_ir"] = overall_ir
        if len(ic_sections) > 1:
            metrics["ic_summaries"] = ic_sections

    if backtest_runs:
        for run in backtest_runs:
            run["backtest_signature"] = _build_backtest_signature(run)
        metrics["backtest_summary"] = backtest_runs[-1]
        metrics["backtest_runs"] = backtest_runs

    if diagnostics:
        metrics["synthesis_diagnostic"] = diagnostics[-1]
        metrics["synthesis_diagnostics"] = diagnostics

    if benchmark_reports:
        compact_reports = [
            _compact_benchmark_report(item) for item in benchmark_reports
        ]
        metrics["benchmark_report"] = compact_reports[-1]
        metrics["benchmark_reports"] = compact_reports
        for run in metrics.get("backtest_runs", []):
            run["benchmark_report"] = compact_reports[-1]
        if metrics.get("backtest_runs"):
            metrics["backtest_summary"] = metrics["backtest_runs"][-1]

    synthesis_names = list((config_payload.get("synthesis_methods") or {}).keys())
    if synthesis_names:
        metrics["synthesis_method_names"] = synthesis_names

    quality_metrics: Dict[str, Any] = {
        "has_ic_summary": "ic_summary" in metrics,
        "has_backtest_summary": "backtest_summary" in metrics,
        "has_synthesis_diagnostic": "synthesis_diagnostic" in metrics,
        "has_benchmark_report": "benchmark_report" in metrics,
    }
    if "ic_summary" in metrics:
        quality_metrics["ic_summary"] = metrics["ic_summary"]
        ic_summary = metrics["ic_summary"]
        if isinstance(ic_summary, Mapping):
            quality_metrics["overall_ic"] = ic_summary.get("overall_ic")
            quality_metrics["overall_ir"] = ic_summary.get("overall_ir")
            quality_metrics["periods"] = ic_summary.get("periods", {})
            quality_metrics["ir_periods"] = ic_summary.get("ir_periods", {})
    if "backtest_summary" in metrics:
        quality_metrics["backtest_summary"] = metrics["backtest_summary"]
        backtest_summary = metrics["backtest_summary"]
        if isinstance(backtest_summary, Mapping):
            for key in (
                "annual_return",
                "total_return",
                "max_drawdown",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "trade_count",
                "top_pct",
            ):
                quality_metrics[key] = backtest_summary.get(key)
    if "synthesis_diagnostic" in metrics:
        quality_metrics["synthesis_diagnostic"] = metrics["synthesis_diagnostic"]

    if any(
        quality_metrics.get(key)
        for key in (
            "has_ic_summary",
            "has_backtest_summary",
            "has_synthesis_diagnostic",
            "has_benchmark_report",
        )
    ):
        metrics["quality_metrics"] = quality_metrics

    return metrics


def _extract_factor_synthesis_metrics(stdout: str, stderr: str, config_path: str) -> Dict[str, Any]:
    """Compatibility reader for historical logs; live runs use execution_result.v1."""
    return _assemble_factor_synthesis_metrics(
        _safe_read_json(Path(config_path)),
        _parse_factor_synthesis_ic_summary(stderr) or _parse_factor_synthesis_ic_summary(stdout),
        _parse_vectorbt_result_summaries(stdout) + _parse_vectorbt_result_summaries(stderr)
        or _parse_factor_synthesis_backtest_runs(stdout) or _parse_factor_synthesis_backtest_runs(stderr),
        _parse_factor_synthesis_diagnostics(stdout) + _parse_factor_synthesis_diagnostics(stderr),
        _parse_benchmark_reports(stdout) + _parse_benchmark_reports(stderr),
        _extract_neutralized_input_warnings(stdout, stderr),
    )


def _bound_result_metrics(path, *, action, run_id, execution_id, source_sha256,
                          config_payload=None, stdout="", stderr=""):
    payload = read_execution_result(
        path, command=action, run_id=run_id, execution_id=execution_id,
        source_sha256=source_sha256,
    )
    raw = payload["metrics"]
    if action == "factor-synthesis":
        for key in ("ic_summaries", "backtest_runs", "synthesis_diagnostics", "benchmark_reports"):
            if not isinstance(raw.get(key, []), list) or any(not isinstance(item, dict) for item in raw.get(key, [])):
                raise ValueError(f"invalid synthesis result field: {key}")
        top_k = (config_payload or {}).get("performance_evaluation", {}).get("top_k_strategy", {})
        if payload["execution"]["status"] == "succeeded" and top_k.get("enabled") and not any(
            "error" not in item and all(isinstance(item.get(key), (int, float))
                                       and math.isfinite(item[key])
                                       for key in ("annual_return", "sharpe_ratio", "max_drawdown"))
            for item in raw.get("backtest_runs", [])
        ):
            raise ValueError("backtest_metrics_missing")
        metrics = _assemble_factor_synthesis_metrics(
            config_payload or {}, raw.get("ic_summaries", []), raw.get("backtest_runs", []),
            raw.get("synthesis_diagnostics", []), raw.get("benchmark_reports", []),
            _extract_neutralized_input_warnings(stdout, stderr),
        )
    else:
        metrics = dict(raw)
    metrics["execution"] = payload["execution"]
    metrics["_execution_failed"] = payload["execution"]["status"] != "succeeded"
    metrics["_artifacts"] = payload["artifacts"]
    return metrics


@dataclass
class HarnessResult:
    action: str
    ok: bool
    exit_code: int
    status: str
    reason: str
    mode: str
    output_dir: Path
    command: List[str]
    metrics: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    warnings: List[str]
    stdout_tail: str = ""
    stderr_tail: str = ""
    started_at: str = ""
    finished_at: str = ""
    elapsed_sec: float = 0.0
    run_id: str = ""
    error: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        summary_path = self.output_dir / "summary.json"
        return {
            "ok": self.ok,
            "command": {
                "name": self.action,
                "argv": list(self.command),
                "mode": self.mode,
                "run_id": self.run_id,
            },
            "summary": {
                "action": self.action,
                "status": self.status,
                "reason": self.reason,
                "output_dir": str(self.output_dir),
                "summary_path": str(summary_path),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_sec": self.elapsed_sec,
            },
            "artifacts": list(self.artifacts),
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "error": self.error,
        }


def _output_dir(action: str, output_root: Path, run_id: str) -> Path:
    if run_id:
        suffix = run_id
    else:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_root / action / suffix
    # These two producers already create their own bound execution result attempt.
    if action in {"single-factor-analysis", "factor-synthesis"}:
        return root
    return root / "attempts" / uuid.uuid4().hex


def _success_result(
    *,
    action: str,
    mode: str,
    output_dir: Path,
    run_id: str,
    command: Sequence[str],
    started_at: str,
    metrics: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    warnings: List[str] | None = None,
    reason: str = "ok",
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> HarnessResult:
    finished_at = _utc_now()
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    return HarnessResult(
        action=action,
        ok=True,
        exit_code=EXIT_SUCCESS,
        status="succeeded" if mode == "subprocess" else mode,
        reason=reason,
        mode=mode,
        output_dir=output_dir,
        command=list(command),
        metrics=metrics,
        artifacts=artifacts,
        warnings=list(warnings or []),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_sec=max((finished_dt - started_dt).total_seconds(), 0.0),
        run_id=run_id,
    )


def _failure_result(
    *,
    action: str,
    mode: str,
    output_dir: Path,
    run_id: str,
    command: Sequence[str],
    started_at: str,
    reason: str,
    exit_code: int,
    metrics: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    warnings: List[str] | None = None,
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> HarnessResult:
    finished_at = _utc_now()
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    return HarnessResult(
        action=action,
        ok=False,
        exit_code=exit_code,
        status="failed",
        reason=reason,
        mode=mode,
        output_dir=output_dir,
        command=list(command),
        metrics=metrics,
        artifacts=artifacts,
        warnings=list(warnings or []),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_sec=max((finished_dt - started_dt).total_seconds(), 0.0),
        run_id=run_id,
        error={"code": reason, "message": reason, "details": dict(metrics)},
    )


def _complete_result(result: HarnessResult, json_out: str = "") -> HarnessResult:
    payload = result.to_dict()
    _write_json(result.output_dir / "summary.json", payload)
    if result.output_dir.parent.name == "attempts":
        _write_json(result.output_dir.parent.parent / "summary.json", payload)
    if json_out:
        _write_json(Path(json_out).expanduser().resolve(), payload)
    return result


def _fake_result(
    *,
    action: str,
    fake_result: str,
    output_dir: Path,
    run_id: str,
    started_at: str,
    command: Sequence[str],
    json_out: str = "",
) -> HarnessResult:
    artifacts = [{"artifact_type": "summary", "path": str(output_dir / "summary.json")}]
    if fake_result == "success":
        return _complete_result(
            _success_result(
                action=action,
                mode="fake",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                metrics={"fake_result": fake_result},
                artifacts=artifacts,
            ),
            json_out=json_out,
        )
    if fake_result == "alignment_failed":
        return _complete_result(
            _failure_result(
                action=action,
                mode="fake",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                reason="alignment_threshold_failed",
                exit_code=EXIT_QUALITY_GATE,
                metrics={"fake_result": fake_result, "alignment_status": "failed"},
                artifacts=artifacts,
            ),
            json_out=json_out,
        )
    if fake_result == "degraded":
        return _complete_result(
            _failure_result(
                action=action,
                mode="fake",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                reason="research_metrics_degraded",
                exit_code=EXIT_QUALITY_GATE,
                metrics={"fake_result": fake_result, "oos_non_degraded": False},
                artifacts=artifacts,
            ),
            json_out=json_out,
        )
    return _complete_result(
        _failure_result(
            action=action,
            mode="fake",
            output_dir=output_dir,
            run_id=run_id,
            command=command,
            started_at=started_at,
            reason="command_failed",
            exit_code=EXIT_EXECUTION,
            metrics={"fake_result": fake_result},
            artifacts=artifacts,
        ),
        json_out=json_out,
    )


def _kill_process_group(pgid: int) -> None:
    """Kill the entire process group by PGID."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    import time

    time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _build_extra_metrics(
    success_metrics_builder, stdout: str, stderr: str
) -> Dict[str, Any]:
    if not success_metrics_builder:
        return {}
    return success_metrics_builder(stdout or "", stderr or "") or {}


def _classify_subprocess_failure(
    *,
    action: str,
    subprocess_exit_code: int,
    extra_metrics: Mapping[str, Any],
) -> tuple[str, int, str]:
    if (
        action == "symbolic-factor-mining"
        and str(extra_metrics.get("failure_reason") or "") == "data_preflight_failed"
    ):
        # Data backend unreachable (e.g. unmounted volume): structured code so
        # operators and the task manager can distinguish it from real
        # execution failures and avoid burning further runs.
        return "data_preflight_failed", EXIT_EXECUTION, "data_preflight_failed"
    if action == "symbolic-factor-mining" and subprocess_exit_code == EXIT_QUALITY_GATE:
        post_rerank = extra_metrics.get("post_rerank")
        if (
            isinstance(post_rerank, Mapping)
            and post_rerank.get("dual_gate_passed") is False
        ):
            detail = str(
                post_rerank.get("quality_gate_failure_reason")
                or extra_metrics.get("failure_reason")
                or post_rerank.get("selected_reason")
                or "post_rerank_quality_gate_failed"
            )
            return "post_rerank_quality_gate_failed", EXIT_QUALITY_GATE, detail
        detail = str(
            extra_metrics.get("failure_reason") or "symbolic_quality_gate_failed"
        )
        return "symbolic_quality_gate_failed", EXIT_QUALITY_GATE, detail
    return "command_failed", EXIT_EXECUTION, ""


def _run_subprocess(**kwargs) -> HarnessResult:
    from utils.output_paths import execution_environment
    command = kwargs["command"]
    if "--result-out" in command:
        attempt = Path(command[command.index("--result-out") + 1]).parent
    else:
        attempt = kwargs["output_dir"]
        if attempt.parent.name != "attempts":
            attempt = attempt / "attempts" / uuid.uuid4().hex
    try:
        with execution_environment(attempt) as (environment, leases):
            if "--config" in command:
                source = Path(command[command.index("--config") + 1])
                if source.is_file():
                    config_dir = attempt / "config"
                    config_dir.mkdir(exist_ok=True)
                    (config_dir / "source.json").write_bytes(source.read_bytes())
                    settings = json.loads(source.read_text()).get("output_settings", {})
                    if settings.get("diagnostic_logs") is True:
                        environment["QUANT_DEBUG_OUTPUTS"] = "1"
            result = _run_subprocess_body(**kwargs, environment=environment, leases=leases)
            if not result.ok:
                raise subprocess.CalledProcessError(result.exit_code, command)
    except subprocess.CalledProcessError:
        if "result" not in locals():
            raise
    return result


def _run_subprocess_body(
    *,
    action: str,
    output_dir: Path,
    run_id: str,
    command: Sequence[str],
    timeout_sec: int,
    started_at: str,
    artifacts: List[Dict[str, Any]],
    json_out: str,
    success_metrics_builder=None,
    environment=None, leases=(),
) -> HarnessResult:
    popen = subprocess.Popen(
        list(command),
        cwd=str(REPO_ROOT),
        env=environment,
        pass_fds=leases,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = popen.communicate(timeout=timeout_sec)
        proc = subprocess.CompletedProcess(
            list(command), popen.returncode, stdout, stderr
        )
    except subprocess.TimeoutExpired:
        _kill_process_group(popen.pid)
        stdout, stderr = "", ""
        try:
            stdout, stderr = popen.communicate(timeout=5)
        except Exception:
            pass
        return _complete_result(
            _failure_result(
                action=action,
                mode="subprocess",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                reason="timeout",
                exit_code=EXIT_EXECUTION,
                metrics={"timeout_sec": timeout_sec},
                artifacts=artifacts,
                stdout_tail=_tail(stdout or ""),
                stderr_tail=_tail((stderr or "") + "\ncommand_timeout"),
            ),
            json_out=json_out,
        )

    except BaseException:
        _kill_process_group(popen.pid)
        try:
            popen.communicate(timeout=5)
        except BaseException:
            pass
        raise

    try:
        extra_metrics = _build_extra_metrics(
            success_metrics_builder, proc.stdout or "", proc.stderr or ""
        )
    except Exception as exc:
        return _complete_result(
            _failure_result(
                action=action, mode="subprocess", output_dir=output_dir, run_id=run_id,
                command=command, started_at=started_at, reason="invalid_execution_result",
                exit_code=EXIT_EXECUTION, artifacts=artifacts,
                metrics={"subprocess_exit_code": proc.returncode, "result_error": str(exc)},
                stdout_tail=_tail(proc.stdout), stderr_tail=_tail(proc.stderr),
            ), json_out=json_out,
        )
    structured_warnings = list(extra_metrics.pop("_warnings", []))
    artifacts.extend(extra_metrics.pop("_artifacts", []))
    execution_failed = extra_metrics.pop("_execution_failed", False)

    if proc.returncode != 0 or execution_failed:
        reason, exit_code, failure_detail = _classify_subprocess_failure(
            action=action,
            subprocess_exit_code=int(proc.returncode),
            extra_metrics=extra_metrics,
        )
        metrics = {"subprocess_exit_code": proc.returncode, **extra_metrics}
        if failure_detail:
            metrics["failure_reason"] = failure_detail
        stderr_tail = proc.stderr
        if failure_detail and failure_detail not in (stderr_tail or ""):
            stderr_tail = (stderr_tail or "") + "\n" + failure_detail
        return _complete_result(
            _failure_result(
                action=action,
                mode="subprocess",
                output_dir=output_dir,
                run_id=run_id,
                command=command,
                started_at=started_at,
                reason=reason,
                exit_code=exit_code,
                metrics=metrics,
                artifacts=artifacts,
                warnings=structured_warnings,
                stdout_tail=_tail(proc.stdout),
                stderr_tail=_tail(stderr_tail),
            ),
            json_out=json_out,
        )

    return _complete_result(
        _success_result(
            action=action,
            mode="subprocess",
            output_dir=output_dir,
            run_id=run_id,
            command=command,
            started_at=started_at,
            metrics={"subprocess_exit_code": proc.returncode, **extra_metrics},
            artifacts=artifacts,
            warnings=structured_warnings,
            reason="ok",
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
        ),
        json_out=json_out,
    )


def _resolve_args_output_root(raw: str) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    configured = os.environ.get("QUANTATION_OUTPUTS_DIR")
    return Path(configured).expanduser().resolve() / "agent_harness" if configured else DEFAULT_OUTPUT_ROOT


def _common_artifacts(output_dir: Path) -> List[Dict[str, Any]]:
    return [{"artifact_type": "summary", "path": str(output_dir / "summary.json")}]


def _handle_single_factor_analysis(args: argparse.Namespace) -> HarnessResult:
    action = "single-factor-analysis"
    output_dir = _output_dir(
        action, _resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = _utc_now()
    execution_id = uuid.uuid4().hex
    effective_run_id = args.run_id or output_dir.name
    result_path = output_dir / "attempts" / execution_id / "result.json"
    command = [sys.executable, str(REPO_ROOT / "run_analysis_main.py"),
               "--result-out", str(result_path), "--run-id", effective_run_id,
               "--execution-id", execution_id]
    if args.config:
        command.extend(["--config", args.config])
    if args.factor:
        command.extend(["--factor", args.factor])
    if args.use_dynamic:
        command.append("--use-dynamic")
    if args.materialize_only:
        command.append("--materialize-only")
    if args.mining_only:
        command.append("--mining-only")
    if args.pool:
        command.extend(["--pools", *args.pool])

    artifacts = _common_artifacts(output_dir) + [{"artifact_type": "execution_result", "path": str(result_path)}]
    artifacts.append({"artifact_type": "analysis_output_dir", "path": str(output_dir)})
    if args.fake_result:
        return _fake_result(
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
        return _complete_result(
            _success_result(
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
        source_sha256 = config_sha256(
            _resolve_repo_relative_path(args.config) if args.config
            else REPO_ROOT / "backtest_and_analysis" / "config.py"
        )
    except OSError as exc:
        return _complete_result(_failure_result(
            action=action, mode="preflight", output_dir=output_dir, run_id=args.run_id,
            command=command, started_at=started_at, reason="missing_config",
            exit_code=EXIT_PRECONDITION, metrics={"error": str(exc)}, artifacts=artifacts,
        ), json_out=args.json_out)
    return _run_subprocess(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        timeout_sec=args.timeout_sec,
        started_at=started_at,
        artifacts=artifacts,
        json_out=args.json_out,
        success_metrics_builder=lambda stdout, stderr: _bound_result_metrics(
            result_path, action=action, run_id=effective_run_id, execution_id=execution_id,
            source_sha256=source_sha256,
        ),
    )


def _extract_symbolic_factor_mining_metrics(stdout: str, stderr: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for text in (stdout, stderr):
        text = (text or "").strip()
        if not text:
            continue
        try:
            parsed = json.loads(text[text.find("{") :])
        except Exception:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    metrics: Dict[str, Any] = {}
    if not payload:
        return metrics

    metrics["pipeline_ok"] = payload.get("ok")
    metrics["pipeline_exit_code"] = payload.get("exit_code")
    if payload.get("failure_reason"):
        metrics["failure_reason"] = payload.get("failure_reason")
    error = payload.get("error")
    if isinstance(error, Mapping):
        if error.get("code"):
            metrics["error_code"] = error.get("code")
        if error.get("message"):
            metrics["error_message"] = error.get("message")
    metrics["symbolic_run_id"] = payload.get("run_id")
    metrics["symbolic_factor_file"] = payload.get("factor_file")
    best = payload.get("best_expression")
    if isinstance(best, Mapping):
        metrics["best_expression"] = best.get("expression")
        metrics["best_score"] = best.get("score")
        metrics["best_complexity"] = best.get("complexity")
        metrics["valid_metrics"] = best.get("valid_metrics", {})
    selected = payload.get("selected_best")
    if isinstance(selected, Mapping):
        metrics["selected_expression"] = selected.get("expression")
        metrics["selected_score"] = selected.get("score")
        metrics["selected_complexity"] = selected.get("complexity")
    post_rerank = payload.get("post_rerank")
    if isinstance(post_rerank, Mapping):
        metrics["post_rerank"] = {
            "enabled": post_rerank.get("enabled"),
            "required": post_rerank.get("required"),
            "dual_gate_passed": post_rerank.get("dual_gate_passed"),
            "selected_rank": post_rerank.get("selected_rank"),
            "selected_reason": post_rerank.get("selected_reason"),
            "quality_gate_failure_reason": post_rerank.get(
                "quality_gate_failure_reason"
            ),
            "top_k": post_rerank.get("top_k"),
            "seed_count": post_rerank.get("seed_count"),
            "seed_variant_lightweight_pass_count": post_rerank.get(
                "seed_variant_lightweight_pass_count"
            ),
            "qd_bucket_count": post_rerank.get("qd_bucket_count"),
        }
    panel = payload.get("panel")
    if isinstance(panel, Mapping):
        metrics["panel"] = dict(panel)
    validation = payload.get("validation")
    if isinstance(validation, Mapping):
        metrics["validation_ok"] = validation.get("ok")
    return metrics


def _handle_symbolic_factor_mining(args: argparse.Namespace) -> HarnessResult:
    action = "symbolic-factor-mining"
    output_dir = _output_dir(
        action, _resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = _utc_now()
    command = [
        sys.executable,
        str(REPO_ROOT / "run_symbolic_factor_mining.py"),
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

    artifacts = _common_artifacts(output_dir)
    artifacts.append(
        {"artifact_type": "symbolic_harness_output_dir", "path": str(output_dir)}
    )

    if args.fake_result:
        return _fake_result(
            action=action,
            fake_result=args.fake_result,
            output_dir=output_dir,
            run_id=args.run_id,
            started_at=started_at,
            command=command,
            json_out=args.json_out,
        )
    if args.dry_run:
        return _complete_result(
            _success_result(
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

    return _run_subprocess(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        timeout_sec=args.timeout_sec,
        started_at=started_at,
        artifacts=artifacts,
        json_out=args.json_out,
        success_metrics_builder=_extract_symbolic_factor_mining_metrics,
    )


def _handle_quanta_alpha_realize(args: argparse.Namespace) -> HarnessResult:
    action = "quanta-alpha-realize"
    output_dir = _output_dir(
        action, _resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = _utc_now()
    realization_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else output_dir / "realization.json"
    )
    command = [action, "--input", args.input, "--output", str(realization_path)]
    artifacts = _common_artifacts(output_dir) + [
        {
            "artifact_type": "quanta_alpha_realization",
            "path": str(realization_path),
            "schema_version": "quanta_alpha_realization_v1",
        }
    ]
    if args.dry_run:
        return _complete_result(
            _success_result(
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
        return _complete_result(
            _failure_result(
                action=action,
                mode="preflight",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="quanta_alpha_input_invalid",
                exit_code=EXIT_PRECONDITION,
                metrics={"error": str(exc)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    except QuantaAlphaEvidenceError as exc:
        return _complete_result(
            _failure_result(
                action=action,
                mode="execution",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="quanta_alpha_evidence_blocked",
                exit_code=EXIT_EXECUTION,
                metrics={"error": str(exc)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    except Exception as exc:
        return _complete_result(
            _failure_result(
                action=action,
                mode="execution",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="quanta_alpha_realization_failed",
                exit_code=EXIT_EXECUTION,
                metrics={"error": str(exc)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    return _complete_result(
        _success_result(
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


def _handle_factor_synthesis(args: argparse.Namespace) -> HarnessResult:
    action = "factor-synthesis"
    output_dir = _output_dir(
        action, _resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = _utc_now()
    resolved_config_path = _resolve_repo_relative_path(args.config)
    execution_id = uuid.uuid4().hex
    return_series_output_dir = (
        Path(args.return_series_output_dir).expanduser().resolve()
        if args.return_series_output_dir
        else output_dir / "attempts" / execution_id / "returns"
    )
    effective_run_id = args.run_id or output_dir.name
    result_path = output_dir / "attempts" / execution_id / "result.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "run_factor_synthesis.py"),
        "--config",
        str(resolved_config_path),
        "--return-series-output-dir",
        str(return_series_output_dir),
        "--run-id",
        effective_run_id,
        "--result-out", str(result_path),
        "--execution-id", execution_id,
    ]
    artifacts = _common_artifacts(output_dir) + [{"artifact_type": "execution_result", "path": str(result_path)}]
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
    if args.fake_result:
        return _fake_result(
            action=action,
            fake_result=args.fake_result,
            output_dir=output_dir,
            run_id=args.run_id,
            started_at=started_at,
            command=command,
            json_out=args.json_out,
        )
    if args.dry_run:
        return _complete_result(
            _success_result(
                action=action,
                mode="dry_run",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                metrics={"planned_config": str(resolved_config_path)},
                artifacts=artifacts,
                reason="dry_run",
            ),
            json_out=args.json_out,
        )
    if not resolved_config_path.exists():
        return _complete_result(
            _failure_result(
                action=action,
                mode="preflight",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="missing_config",
                exit_code=EXIT_PRECONDITION,
                metrics={"missing_path": str(resolved_config_path)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    try:
        audit_context = load_capacity_audit_context(resolved_config_path)
        audit_preflight = (
            preflight_capacity_audit(audit_context)
            if audit_context is not None
            else None
        )
    except D9CapacityAuditError as exc:
        return _complete_result(
            _failure_result(
                action=action,
                mode="preflight",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason=exc.code,
                exit_code=EXIT_PRECONDITION,
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
    source_sha256 = config_sha256(resolved_config_path)
    config_payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    result = _run_subprocess(
        action=action,
        output_dir=output_dir,
        run_id=args.run_id,
        command=command,
        timeout_sec=args.timeout_sec,
        started_at=started_at,
        artifacts=artifacts,
        json_out=args.json_out,
        success_metrics_builder=lambda stdout, stderr: _bound_result_metrics(
            result_path, action=action, run_id=effective_run_id, execution_id=execution_id,
            source_sha256=source_sha256, config_payload=config_payload,
            stdout=stdout, stderr=stderr,
        ),
    )
    if audit_context is None or not result.ok:
        return result
    try:
        audit = build_capacity_audit(
            audit_context,
            audit_preflight or {},
            result.metrics,
        )
        audit_path = write_capacity_audit(output_dir / "d9_capacity_audit.json", audit)
    except D9CapacityAuditError as exc:
        failed_metrics = dict(result.metrics)
        failed_metrics["d9_capacity_audit_error"] = {
            "message": str(exc),
            "details": exc.details,
        }
        return _complete_result(
            _failure_result(
                action=action,
                mode="subprocess",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason=exc.code,
                exit_code=EXIT_QUALITY_GATE,
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
    return _complete_result(result, json_out=args.json_out)




def _handle_parent_factor_screen(args: argparse.Namespace) -> HarnessResult:
    action = "parent-factor-screen"
    output_dir = _output_dir(
        action, _resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = _utc_now()
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
    artifacts = _common_artifacts(output_dir) + [
        {"artifact_type": "algorithm_parent_factor_screen", "path": str(evidence_path)}
    ]
    if args.dry_run:
        return _complete_result(
            _success_result(
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
        return _complete_result(
            _failure_result(
                action=action,
                mode="preflight",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="parent_factor_screen_invalid",
                exit_code=EXIT_PRECONDITION,
                metrics={"error": str(exc)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    except Exception as exc:
        return _complete_result(
            _failure_result(
                action=action,
                mode="execution",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="parent_factor_screen_failed",
                exit_code=EXIT_EXECUTION,
                metrics={"error": str(exc)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    return _complete_result(
        _success_result(
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




















def _duckdb_count_tables(db_path: Path, *, bindings=None) -> Dict[str, Any]:
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
        factor_count = 0
        if any(row[0] == "factor_analysis_results" for row in tables):
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
        }
    finally:
        conn.close()


def _handle_monitor_refresh(args: argparse.Namespace) -> HarnessResult:
    action = "monitor-refresh"
    output_dir = _output_dir(
        action, _resolve_args_output_root(args.output_root), args.run_id
    )
    started_at = _utc_now()
    command = ["monitor_refresh", "--database-path", args.database_path]
    artifacts = _common_artifacts(output_dir)
    if args.fake_result:
        return _fake_result(
            action=action,
            fake_result=args.fake_result,
            output_dir=output_dir,
            run_id=args.run_id,
            started_at=started_at,
            command=command,
            json_out=args.json_out,
        )
    if args.dry_run:
        return _complete_result(
            _success_result(
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
        return _complete_result(
            _failure_result(
                action=action,
                mode="inspection",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="monitor_database_missing",
                exit_code=EXIT_PRECONDITION,
                metrics={"missing_path": args.database_path},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )

    if not bindings and db_path.suffix != ".duckdb":
        return _complete_result(
            _failure_result(
                action=action,
                mode="inspection",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="monitor_requires_duckdb",
                exit_code=EXIT_PRECONDITION,
                metrics={"database_path": str(db_path)},
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    metrics = _duckdb_count_tables(db_path, bindings=bindings)
    metrics["database_path"] = str(db_path)
    if not metrics.get("available"):
        return _complete_result(
            _failure_result(
                action=action,
                mode="inspection",
                output_dir=output_dir,
                run_id=args.run_id,
                command=command,
                started_at=started_at,
                reason="monitor_backend_unavailable",
                exit_code=EXIT_PRECONDITION,
                metrics=metrics,
                artifacts=artifacts,
            ),
            json_out=args.json_out,
        )
    return _complete_result(
        _success_result(
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

    started = _utc_now()
    try:
        payload = run(args, REPO_ROOT)
        ok = payload.get("status") == "PASS"
        failed = payload.get("failed_stage", "")
        code = (
            EXIT_SUCCESS
            if ok
            else (
                EXIT_QUALITY_GATE
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
                else EXIT_EXECUTION
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
                else EXIT_PRECONDITION
            )
        )
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "error_code": str(exc),
            "failed_stage": "preflight",
            "layers": {},
        }
        ok, code = False, EXIT_PRECONDITION
    result = HarnessResult(
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
        finished_at=_utc_now(),
        run_id=args.run_id,
    )
    return _complete_result(result, json_out=args.json_out)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent-native harness wrappers for quantation_factor_database"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    gc = subparsers.add_parser("outputs-gc", help="Delegate owned tmp/cache cleanup to the public Task Manager GC")
    gc.add_argument("--task-manager", default="task-manager", help="Installed Task Manager CLI executable")
    gc.add_argument("--task-root", default=str(REPO_ROOT), help="Owner root to collect")
    gc.add_argument("--scope", choices=("temporary", "cache", "local"), default="local")
    mode = gc.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    gc.add_argument("--completed-hours", type=float, default=24)
    gc.add_argument("--failed-days", type=float, default=7)
    gc.add_argument("--cache-days", type=float, default=30)


    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--run-id", default="", help="Stable run identifier for output colocation"
        )
        subparser.add_argument(
            "--output-root", default="", help="Override output root for summaries"
        )
        subparser.add_argument(
            "--json-out",
            default="",
            help="Optional path to write the final JSON payload",
        )
        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help="Plan the command and emit structured JSON only",
        )
        subparser.add_argument(
            "--fake-result",
            choices=("success", "failure", "degraded", "alignment_failed"),
            default="",
            help="Bypass heavy execution and emit a deterministic fake result",
        )
        subparser.add_argument(
            "--timeout-sec",
            type=int,
            default=1800,
            help="Subprocess timeout in seconds",
        )

    alignment = subparsers.add_parser(
        "alignment-run",
        help="Build, replay and compare factor or frozen-strategy alignment",
    )
    add_common(alignment)
    target = alignment.add_mutually_exclusive_group(required=True)
    target.add_argument("--factor")
    target.add_argument("--strategy-config")
    alignment.add_argument("--factor-class")
    alignment.add_argument("--output-dir", required=True)
    alignment.add_argument("--real-cloud", action="store_true")
    alignment.add_argument("--resume", action="store_true")
    alignment.add_argument("--iteration", type=int, default=1)
    alignment.add_argument("--max-iterations", type=int, default=1)
    alignment.add_argument("--result-file")
    alignment.add_argument("--work-log")

    analyze = subparsers.add_parser(
        "single-factor-analysis", help="Wrap run_analysis_main.py"
    )
    add_common(analyze)
    analyze.add_argument(
        "--config",
        default="",
        help="Optional JSON config path passed to run_analysis_main.py",
    )
    analyze.add_argument("--factor", default="", help="Optional factor module path")
    analyze.add_argument(
        "--use-dynamic", action="store_true", help="Forward --use-dynamic"
    )
    analyze.add_argument(
        "--materialize-only",
        action="store_true",
        help="Forward --materialize-only",
    )
    analyze.add_argument(
        "--mining-only",
        action="store_true",
        help="Forward --mining-only",
    )
    analyze.add_argument(
        "--pool", action="append", default=[], help="Append a dynamic pool name"
    )

    symbolic = subparsers.add_parser(
        "symbolic-factor-mining", help="Wrap run_symbolic_factor_mining.py"
    )
    add_common(symbolic)
    symbolic.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "symbolic_mining" / "default.json"),
    )

    quanta_alpha = subparsers.add_parser(
        "quanta-alpha-realize",
        help="Compile and fast-screen one three-expression QuantaAlpha proposal",
    )
    add_common(quanta_alpha)
    quanta_alpha.add_argument("--input", required=True, help="Frozen proposal JSON")
    quanta_alpha.add_argument(
        "--output", default="", help="Frozen realization JSON output path"
    )
    symbolic.add_argument(
        "--novelty-context", default="", help="Run-local symbolic novelty context JSON"
    )
    symbolic.add_argument(
        "--marginal-context",
        default="",
        help="Run-local symbolic marginal-aware search context JSON",
    )
    symbolic.add_argument(
        "--skip-validation", action="store_true", help="Forward --skip-validation"
    )
    symbolic.add_argument(
        "--materialize-candidate-rank",
        type=int,
        default=0,
        help="Materialize and validate a rerank candidate rank from an existing symbolic run",
    )
    symbolic.add_argument(
        "--factor-path",
        default="",
        help="Override final factor path for candidate materialization",
    )

    synthesis = subparsers.add_parser(
        "factor-synthesis", help="Wrap run_factor_synthesis.py"
    )
    add_common(synthesis)
    synthesis.add_argument(
        "--config", default=str(REPO_ROOT / "config" / "factor_synthesis_config.json")
    )
    synthesis.add_argument(
        "--return-series-output-dir",
        default="",
        help="Override output directory for per-candidate return CSV artifacts",
    )

    parent_screen = subparsers.add_parser(
        "parent-factor-screen",
        help="Rank eligible factors by residual RankIC against one exact parent",
    )
    add_common(parent_screen)
    parent_screen.add_argument("--job", required=True, help="Frozen parent screen job JSON")
    parent_screen.add_argument("--output", required=True, help="Evidence JSON output path")

    monitor = subparsers.add_parser(
        "monitor-refresh", help="Build a structured monitor readiness summary"
    )
    add_common(monitor)
    monitor.add_argument("--database-path", default=str(DEFAULT_MONITOR_DB))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
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
    if args.command == "alignment-run":
        result = _handle_alignment_run(args)
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
