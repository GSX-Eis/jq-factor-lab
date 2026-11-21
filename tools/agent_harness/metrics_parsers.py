"""Parse historical synthesis and symbolic mining logs into Harness metrics."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from tools.agent_harness import _json_io
from tools.agent_harness._cli_compat import _cli_bindings

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
    cli = _cli_bindings(__name__)
    return cli._LOG_PREFIX_RE.sub("", line.rstrip("\n"))


def _extract_neutralized_input_warnings(stdout: str, stderr: str) -> List[str]:
    cli = _cli_bindings(__name__)
    warnings: List[str] = []
    seen: set[str] = set()
    for text in (stdout, stderr):
        for raw_line in text.splitlines():
            line = cli._strip_log_prefix(raw_line).strip()
            if cli._NEUTRALIZED_INPUT_WARNING_TOKEN not in line or line in seen:
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
    cli = _cli_bindings(__name__)
    annual_returns = run.get("annual_returns")
    yearly: Dict[str, float] = {}
    if isinstance(annual_returns, Mapping):
        for year, value in annual_returns.items():
            numeric = cli._coerce_signature_float(value)
            if numeric is not None:
                yearly[str(year)] = numeric
    signature = {
        "annual_return": cli._coerce_signature_float(run.get("annual_return")),
        "total_return": cli._coerce_signature_float(run.get("total_return")),
        "max_drawdown": cli._coerce_signature_float(run.get("max_drawdown")),
        "sharpe_ratio": cli._coerce_signature_float(run.get("sharpe_ratio")),
        "sortino_ratio": cli._coerce_signature_float(run.get("sortino_ratio")),
        "top_pct": cli._coerce_signature_float(run.get("top_pct")),
        "yearly_returns": yearly,
    }
    return signature


def _parse_factor_synthesis_ic_summary(text: str) -> List[Dict[str, Any]]:
    cli = _cli_bindings(__name__)
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = cli._strip_log_prefix(raw_line)
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

        match = cli._IC_PERIOD_RE.match(line)
        if match:
            current["periods"][match.group("period")] = {
                "mean_ic": float(match.group("mean")),
                "std_ic": float(match.group("std")),
                "min_ic": float(match.group("min")),
                "max_ic": float(match.group("max")),
                "positive_ratio": cli._pct_to_ratio(match.group("pos")),
                "sample_count": int(match.group("n")),
            }
            continue

        match = cli._IC_OVERALL_RE.match(line)
        if match:
            current["overall_ic"] = float(match.group("value"))
            current["factor_name"] = match.group("factor").strip()
            continue

        match = cli._IR_PERIOD_RE.match(line)
        if match:
            current["ir_periods"][match.group("period")] = float(match.group("value"))
            continue

        match = cli._IR_OVERALL_RE.match(line)
        if match:
            current["overall_ir"] = float(match.group("value"))
            current["factor_name"] = match.group("factor").strip()
            continue

    if current:
        sections.append(current)

    return sections


def _parse_factor_synthesis_backtest_runs(text: str) -> List[Dict[str, Any]]:
    cli = _cli_bindings(__name__)
    runs: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    in_annual_returns = False

    for raw_line in text.splitlines():
        line = cli._strip_log_prefix(raw_line)
        if not line.strip():
            continue

        header = cli._BACKTEST_HEADER_RE.match(line)
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
        for field_name, pattern in cli._BACKTEST_METRIC_RE.items():
            match = pattern.match(line)
            if match:
                value = match.group("value")
                if field_name == "trade_count":
                    current[field_name] = int(value)
                elif field_name == "max_drawdown":
                    current[field_name] = cli._pct_to_ratio(value)
                elif field_name in {"annual_return", "total_return"}:
                    current[field_name] = cli._pct_to_ratio(value)
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
            match = cli._ANNUAL_RETURN_RE.match(line)
            if match:
                current["annual_returns"][int(match.group("year"))] = cli._pct_to_ratio(
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
    cli = _cli_bindings(__name__)
    diagnostics: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = cli._strip_log_prefix(raw_line)
        match = cli._SYNTHESIS_DIAGNOSTICS_RE.search(line)
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
    cli = _cli_bindings(__name__)
    summaries: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = cli._strip_log_prefix(raw_line)
        match = cli._VECTORBT_RESULT_SUMMARIES_RE.search(line)
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
    cli = _cli_bindings(__name__)
    reports: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = cli._strip_log_prefix(raw_line)
        match = cli._BENCHMARK_REPORT_RE.search(line)
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
    cli = _cli_bindings(__name__)
    if isinstance(value, dict):
        return {
            key: cli._compact_benchmark_report(item)
            for key, item in value.items()
            if key not in cli._BENCHMARK_DETAIL_KEYS
        }
    if isinstance(value, list):
        return [cli._compact_benchmark_report(item) for item in value]
    return value


def _assemble_factor_synthesis_metrics(
    config_payload, ic_sections, backtest_runs, diagnostics, benchmark_reports,
    neutralized_input_warnings,
) -> Dict[str, Any]:
    cli = _cli_bindings(__name__)
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
            run["backtest_signature"] = cli._build_backtest_signature(run)
        metrics["backtest_summary"] = backtest_runs[-1]
        metrics["backtest_runs"] = backtest_runs

    if diagnostics:
        metrics["synthesis_diagnostic"] = diagnostics[-1]
        metrics["synthesis_diagnostics"] = diagnostics

    if benchmark_reports:
        compact_reports = [
            cli._compact_benchmark_report(item) for item in benchmark_reports
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
    cli = _cli_bindings(__name__, _json_io)
    return cli._assemble_factor_synthesis_metrics(
        cli._safe_read_json(Path(config_path)),
        cli._parse_factor_synthesis_ic_summary(stderr) or cli._parse_factor_synthesis_ic_summary(stdout),
        cli._parse_vectorbt_result_summaries(stdout) + cli._parse_vectorbt_result_summaries(stderr)
        or cli._parse_factor_synthesis_backtest_runs(stdout) or cli._parse_factor_synthesis_backtest_runs(stderr),
        cli._parse_factor_synthesis_diagnostics(stdout) + cli._parse_factor_synthesis_diagnostics(stderr),
        cli._parse_benchmark_reports(stdout) + cli._parse_benchmark_reports(stderr),
        cli._extract_neutralized_input_warnings(stdout, stderr),
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
