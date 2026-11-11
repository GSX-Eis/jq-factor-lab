#!/usr/bin/env python3
"""Run the deterministic pre-cloud portion of one alignment repair round.

Codex calls this once per internal round.  It owns static review, target-only
dumper generation, strategy build, and virtual-cloud validation; Codex only
needs to inspect the compact JSON result and make a source-level repair.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from evidence import (
    archive_previous,
    file_hashes,
    intact,
    intact_inputs,
    input_manifest,
    layer_results,
)

# The skill directory can be symlinked from the shared skills repository.
# Alignment always invokes this script with the isolated runtime workspace as
# cwd; that is the only reliable execution-repository root.
REPO_ROOT = Path.cwd().resolve()
WORKSPACE_SCRIPTS = REPO_ROOT / ".claude/skills/alignment-pipeline/scripts"
BROWSER_ERROR_CODES = (
    "browser_session_stale_unrecoverable",
    "browser_session_missing",
    "browser_required_tab_missing",
    "kernel_status_unreadable",
)
CLOUD_WAIT_SECONDS = str(int(os.environ.get("ALIGNMENT_CLOUD_WAIT_SECONDS", "5400")))
if int(CLOUD_WAIT_SECONDS) <= 0:
    raise ValueError("ALIGNMENT_CLOUD_WAIT_SECONDS must be positive")
LOCAL_MEMORY_LIMIT_BYTES = 6_000_000_000
LOCAL_THREAD_ENV = {
    key: "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
        "QUANT_DATA_STORE_THREADS",
    )
}


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _final_issues(result: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for stage in result.get("stages", []):
        if not isinstance(stage, dict):
            continue
        for issue in stage.get("issues", []):
            if not isinstance(issue, dict) or issue.get("status") not in {
                "ISSUE",
                "WARN",
            }:
                continue
            text = ": ".join(
                part
                for part in (
                    str(issue.get("check_id") or ""),
                    str(issue.get("message") or ""),
                )
                if part
            )
            if text and text not in issues:
                issues.append(text)
        if not stage.get("ok") and stage.get("reason"):
            text = f"{stage.get('name')}: {stage['reason']}"
            if text not in issues:
                issues.append(text)
    return issues


def _should_finalize(
    result: dict[str, Any], *, real_cloud: bool, iteration: int, max_iterations: int
) -> bool:
    if not real_cloud:
        return False
    return (
        result.get("status") == "PASS"
        or iteration >= max_iterations
        or result.get("error_code") == "browser_session_missing"
    )


def _write_final_artifacts(
    *,
    result: dict[str, Any],
    factor_path: Path,
    iteration: int,
    result_file: Path,
    work_log: Path,
) -> None:
    metrics = dict(result.get("final_metrics") or {})
    if metrics.get("spearman_mean") is not None:
        metrics.setdefault("spearman_daily", metrics["spearman_mean"])
    issues = _final_issues(result)
    status = str(result.get("status") or "FAIL")
    error_code = str(result.get("error_code") or "")
    summary = (
        "Script-driven cloud/local comparison passed."
        if status == "PASS"
        else f"Script-driven alignment stopped at {result.get('failed_stage') or 'unknown'}: {error_code}."
    )
    modified_files = [_display_path(result_file), _display_path(work_log)]
    payload = {
        "schema_version": 2,
        "fingerprint": result.get("fingerprint"),
        "artifact_hashes": result.get("artifact_hashes", {}),
        "layers": result.get("layers", {}),
        "status": status,
        "factor_path": str(factor_path),
        "iterations": iteration,
        "final_metrics": metrics,
        "issues_found": issues,
        "modified_files": modified_files,
        "fix_summary": summary,
        "error": None if status == "PASS" else error_code or "alignment_failed",
    }
    stage_lines = [
        f"- [{'PASS' if stage.get('ok') else 'FAIL'}] {stage.get('name')}"
        + (f"; sampled peak process-tree RSS: {stage['resources']['sampled_peak_tree_rss_bytes']} bytes; limit: {stage['resources']['memory_limit_bytes']} bytes"
           if stage.get('resources') else "")
        for stage in result.get("stages", [])
        if isinstance(stage, dict)
    ]
    issue_lines = [f"- {issue}" for issue in issues] or ["- None"]
    work_log_text = "\n".join(
        [
            "# Alignment Repair Result",
            "",
            f"- Status: {status}",
            f"- Factor: {factor_path}",
            f"- Iteration: {iteration}",
            f"- Error: {payload['error'] or 'None'}",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Stages",
            "",
            *stage_lines,
            "",
            "## Issues",
            "",
            *issue_lines,
            "",
        ]
    )
    _atomic_write_text(work_log, work_log_text)
    # The result file is the completion marker; write it only after the work log.
    _atomic_write_json(result_file, payload)
    _emit(
        "final_artifacts_written",
        result_file=str(result_file),
        work_log=str(work_log),
        status=status,
    )


def _structured_error(log_text: str) -> str:
    return next((code for code in BROWSER_ERROR_CODES if code in log_text), "")


def _run(
    name: str,
    command: list[str],
    output_dir: Path,
    finalize: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    log_path = output_dir / f"{name}.log"
    started = time.monotonic()
    monitor_memory = name == "local_analysis"
    peak_rss = 0
    memory_exceeded = False
    if monitor_memory:
        import psutil
    heartbeat_seconds = max(
        0.1, float(os.environ.get("ALIGNMENT_HEARTBEAT_SECONDS", "15"))
    )
    _emit("stage_started", stage=name, log=str(log_path))
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\nOUTPUT\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **LOCAL_THREAD_ENV} if monitor_memory else None,
            )
        except OSError as exc:
            log.write(f"{exc}\n")
            returncode = 127
        else:
            next_heartbeat = started + heartbeat_seconds
            while True:
                if monitor_memory:
                    try:
                        parent = psutil.Process(process.pid)
                        family = [parent, *parent.children(recursive=True)]
                        rss = 0
                        for member in family:
                            try:
                                rss += member.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                        peak_rss = max(peak_rss, rss)
                        if rss > LOCAL_MEMORY_LIMIT_BYTES:
                            memory_exceeded = True
                            log.write(f"local_memory_limit_exceeded: rss={rss}, limit={LOCAL_MEMORY_LIMIT_BYTES}\n")
                            log.flush()
                            for member in reversed(family):
                                try:
                                    member.terminate()
                                except psutil.NoSuchProcess:
                                    pass
                            _, alive = psutil.wait_procs(family, timeout=5)
                            for member in alive:
                                try:
                                    member.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            returncode = process.wait()
                            break
                    except psutil.NoSuchProcess:
                        pass
                try:
                    returncode = process.wait(timeout=min(0.25, heartbeat_seconds) if monitor_memory else heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= next_heartbeat:
                        _emit(
                            "heartbeat",
                            stage=name,
                            elapsed_seconds=round(time.monotonic() - started, 1),
                            **({"sampled_peak_rss_bytes": peak_rss} if monitor_memory else {}),
                        )
                        next_heartbeat = time.monotonic() + heartbeat_seconds
    stage = {
        "name": name,
        "ok": returncode == 0,
        "returncode": returncode,
        "log": str(log_path),
    }
    if finalize is not None:
        finalize(stage)
    if monitor_memory:
        stage["resources"] = {
            "sampled_peak_tree_rss_bytes": peak_rss,
            "sample_interval_seconds": min(0.25, heartbeat_seconds),
            "memory_limit_bytes": LOCAL_MEMORY_LIMIT_BYTES,
            "thread_env": LOCAL_THREAD_ENV,
        }
        if memory_exceeded:
            stage.update(ok=False, reason="local_memory_limit_exceeded")
    error_code = _structured_error(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if error_code:
        stage.update(ok=False, reason=error_code)
    elif (
        name == "wait_kernel"
        and returncode
        and "[wait_kernel] TIMEOUT" in log_path.read_text()
    ):
        stage.update(ok=False, reason="cloud_execution_pending")
    _emit(
        "stage_ok" if stage["ok"] else "stage_failed",
        stage=name,
        elapsed_seconds=round(time.monotonic() - started, 1),
        error_code=str(stage.get("reason") or ""),
        log=str(log_path),
        **({"resources": stage["resources"]} if monitor_memory else {}),
    )
    return stage


def _classify_static_review(stage: dict[str, Any], report_path: Path) -> None:
    """CHECK-11 is global REQ parity noise, not a target-factor repair gate."""
    if not report_path.exists():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        issues = report.get("issues") or []
    except (OSError, json.JSONDecodeError):
        return
    fatal = [
        item
        for item in issues
        if item.get("status") == "ISSUE" and item.get("check_id") != "CHECK-11"
    ]
    stage["issues"] = issues
    stage["ignored_global_issues"] = [
        item for item in issues if item.get("check_id") == "CHECK-11"
    ]
    stage["ok"] = not fatal
    if fatal:
        stage["returncode"] = 1


def _classify_local_analysis(stage: dict[str, Any]) -> None:
    log_text = Path(stage["log"]).read_text(encoding="utf-8", errors="replace")
    coverage_matches = re.findall(r"数据覆盖率:\s*([\d.]+)%", log_text)
    if coverage_matches:
        coverage = float(coverage_matches[-1]) / 100
        stage["metrics"] = {"local_factor_coverage": coverage}
        if coverage < 0.10:
            stage.update(
                ok=False,
                returncode=1,
                reason=f"local_factor_coverage_below_threshold:{coverage:.4f}",
            )
            return
    if re.search(
        r"0\s*/\s*\d+\s*个组合成功分析|值形状\s*\(已过滤\):\s*\(0,\s*0\)", log_text
    ):
        stage.update(ok=False, returncode=1, reason="local_factor_output_empty")


def _attach_virtual_warnings(stage: dict[str, Any], report_path: Path) -> None:
    try:
        warnings = (
            json.loads(report_path.read_text(encoding="utf-8")).get("warnings") or []
        )
    except (AttributeError, OSError, json.JSONDecodeError):
        return
    stage["issues"] = [
        {"check_id": "virtual_cloud", "status": "WARN", "message": warning}
        for warning in warnings
    ]


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"real-cloud outputs must be under the runtime workspace: {path}"
        ) from exc


def _results_db_path() -> Path:
    config_path = REPO_ROOT / "config" / "small_mkt_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw = str(
        (payload.get("results_database") or {}).get("duckdb_path")
        or "analysis_results.duckdb"
    )
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _conda_env_python(env_name: str) -> Path | None:
    """Resolve the environment interpreter without inheriting an active venv PATH."""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        prefixes = json.loads(result.stdout).get("envs", [])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    return next(
        (
            Path(prefix) / "bin" / "python"
            for prefix in prefixes
            if Path(prefix).name == env_name
        ),
        None,
    )


def _apply_comparison_verdict(stage: dict[str, Any], report_dir: Path) -> None:
    report_path = report_dir / "alignment_results.json"
    if not report_path.exists():
        stage.update(ok=False, returncode=1, reason="comparison_report_missing")
        return
    try:
        results = json.loads(report_path.read_text(encoding="utf-8"))
        result = results[0] if isinstance(results, list) and results else {}
        summary = result.get("summary") if isinstance(result, dict) else {}
        metrics = {
            "spearman_mean": (summary or {}).get("avg_spearman"),
            "spearman_min": (summary or {}).get("min_spearman"),
            "stock_pool_overlap": result.get("stock_pool_overlap")
            if isinstance(result, dict)
            else None,
            "max_abs_diff": (summary or {}).get("max_max_abs_diff"),
            "numerical_match": result.get("numerical_match"),
            "missingness_match": result.get("missingness_match"),
            "cloud_rows": result.get("cloud_rows")
            if isinstance(result, dict)
            else None,
            "local_rows": result.get("local_rows")
            if isinstance(result, dict)
            else None,
            "merged_rows": result.get("merged_rows")
            if isinstance(result, dict)
            else None,
            "cloud_rows_in_local_range": result.get("cloud_rows_in_local_range")
            if isinstance(result, dict)
            else None,
            "local_date_min": result.get("local_date_min")
            if isinstance(result, dict)
            else None,
            "local_date_max": result.get("local_date_max")
            if isinstance(result, dict)
            else None,
        }
        status = str(result.get("status") or "")
        stage["metrics"] = metrics
        stage["report_status"] = status
        if result.get("first_difference"):
            stage["first_difference"] = result["first_difference"]
        if status in {"NO_DATA", "NO_OVERLAP", "INSUFFICIENT_DATA"}:
            reason = f"comparison_{status.lower()}"
            if status == "NO_DATA":
                empty_sides = [
                    side
                    for side in ("cloud", "local")
                    if not metrics.get(f"{side}_rows")
                ]
                if empty_sides:
                    reason += ":" + ",".join(empty_sides)
            stage.update(ok=False, returncode=1, reason=reason)
            return
        mean = float(metrics["spearman_mean"])
        minimum = float(metrics["spearman_min"])
        overlap = float(metrics["stock_pool_overlap"])
    except (IndexError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        stage.update(ok=False, returncode=1, reason=f"comparison_report_invalid:{exc}")
        return
    diagnosis = result.get("worst_date_diagnosis")
    if isinstance(diagnosis, dict):
        diagnosis_path = report_dir / "worst_date_diagnosis.json"
        diagnosis_path.write_text(
            json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stage["worst_date_diagnosis"] = diagnosis
        stage["diagnosis_path"] = str(diagnosis_path)
    passed = (
        mean > 0.99
        and minimum > 0.95
        and overlap >= 0.95
        and metrics["numerical_match"] is True
        and metrics["missingness_match"] is True
    )
    stage["ok"] = passed
    stage["returncode"] = 0 if passed else 1
    if not passed:
        stage["reason"] = result.get("first_difference", {}).get(
            "reason", "alignment_metrics_below_threshold"
        )


def _cloud_artifact_error(
    cloud_csv: Path, cloud_log: Path, factor_class: str, fingerprint: str | None = None
) -> str:
    if not cloud_log.exists():
        return "cloud_log_missing"
    log_text = cloud_log.read_text(encoding="utf-8", errors="replace")
    traceback_match = list(
        re.finditer(r"Traceback\s*\(most recent call last\)", log_text)
    )
    traceback_at = traceback_match[-1].start() if traceback_match else -1
    if traceback_at >= 0:
        traceback_lines = [
            line.strip()
            for line in log_text[traceback_at:].splitlines()
            if line.strip()
        ]
        detail = next(
            (
                line
                for line in reversed(traceback_lines)
                if "Error:" in line or "Exception:" in line
            ),
            traceback_lines[-1],
        )
        return f"cloud_execution_failed:{detail[:300]}"
    if "No data generated." in log_text:
        return "cloud_no_data"
    total_rows = re.findall(r"Total Rows:\s*(\d+)", log_text)
    expected_rows = int(total_rows[-1]) if total_rows else None
    if expected_rows == 0:
        return "cloud_no_data"
    if not cloud_csv.exists():
        return "cloud_output_missing"
    try:
        with cloud_csv.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "factor" not in reader.fieldnames:
                return "cloud_output_invalid:factor_column_missing"
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        return f"cloud_output_invalid:{exc}"
    available = sorted({str(row.get("factor") or "") for row in rows} - {""})
    if available != [factor_class]:
        return f"cloud_output_factor_mismatch:expected={factor_class};available={','.join(available)}"
    if fingerprint:
        import hashlib

        expected = (
            "ALIGNMENT_ARTIFACT|"
            + fingerprint
            + "|"
            + hashlib.sha256(cloud_csv.read_bytes()).hexdigest()
        )
        if expected not in log_text:
            return "cloud_artifact_version_mismatch"
    elif expected_rows is None:
        # Without a hash-bound completion marker, a truncated notebook log
        # cannot prove that the CSV belongs to the completed execution.
        return "cloud_execution_incomplete"
    if not rows:
        return "cloud_no_data"
    if expected_rows is not None and len(rows) != expected_rows:
        return f"cloud_output_row_count_mismatch:log={expected_rows};csv={len(rows)}"
    return ""


def run_cycle(
    *,
    factor: str,
    factor_class: str,
    output_dir: Path,
    real_cloud: bool = False,
    resume: bool = False,
    iteration: int = 1,
    max_iterations: int = 3,
    result_file: Path | None = None,
    work_log: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_path = Path(factor)
    if not factor_path.is_absolute():
        factor_path = (REPO_ROOT / factor_path).resolve()
    prepared_dumper = output_dir / "cloud_factor_dumper.py"
    bundle = output_dir / "jq_strategy.py"
    static_json = output_dir / "static_review.json"
    compat_json = output_dir / "virtual_cloud.json"
    # [ALIGNMENT-FIX] Finish also runs during preflight, before real-cloud setup.
    report_dir = output_dir / "comparison"
    checkpoint_path = output_dir / "alignment_cycle_state.json"
    manifest = input_manifest(REPO_ROOT, factor_path, factor_class)
    fingerprint = manifest["fingerprint"]
    archive_previous(output_dir, fingerprint)
    _atomic_write_json(output_dir / "alignment_manifest.json", manifest)
    saved: dict[str, Any] = {}
    if resume and checkpoint_path.exists():
        try:
            candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (
                candidate.get("schema_version") == 2
                and candidate.get("fingerprint") == fingerprint
            ):
                saved = candidate
        except (OSError, json.JSONDecodeError):
            pass
    cached_stages = {
        stage["name"]: stage
        for stage in saved.get("stages", [])
        if isinstance(stage, dict) and stage.get("name")
    }
    stages: list[dict[str, Any]] = []
    reusing = bool(saved)
    remote_started = float(saved.get("remote_started") or 0.0)

    def save_checkpoint() -> None:
        _atomic_write_json(
            checkpoint_path,
            {
                "schema_version": 2,
                "fingerprint": fingerprint,
                "factor": str(factor_path),
                "factor_class": factor_class,
                "remote_started": remote_started,
                "stages": stages,
            },
        )

    def run_stage(
        name: str,
        command: list[str],
        *,
        required_outputs: tuple[Path, ...] = (),
        finalize: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        nonlocal reusing
        cached = cached_stages.get(name)
        if (
            reusing
            and cached
            and cached.get("ok")
            and intact(cached.get("output_hashes", {}))
            and all(
                str(path.resolve()) in cached.get("output_hashes", {})
                for path in required_outputs
            )
        ):
            stage = dict(cached)
            stages.append(stage)
            _emit("stage_resumed", stage=name, log=str(stage.get("log") or ""))
            save_checkpoint()
            return stage
        reusing = False
        for path in required_outputs:
            path.unlink(missing_ok=True)
        started_at = time.time()
        stage = _run(name, command, output_dir, finalize)
        stage["started_at_unix"] = started_at
        stage["finished_at_unix"] = time.time()
        missing = [str(p) for p in required_outputs if not p.is_file()]
        if stage["ok"] and missing:
            stage.update(
                ok=False, reason="artifact_missing:" + ",".join(missing), returncode=1
            )
        inputs = {}
        if name == "local_analysis" and stage["ok"]:
            try:
                record = json.loads((output_dir / "local_analysis.json").read_text())
                inputs = record["input_hashes"]
                valid_paths = all(
                    Path(p).resolve().is_relative_to(output_dir.resolve())
                    for p in inputs
                )
                if (
                    record.get("schema_version") != 2
                    or record.get("fingerprint") != fingerprint
                    or not valid_paths
                    or not intact(inputs)
                ):
                    raise ValueError("local_input_snapshot_missing_or_changed")
            except (OSError, ValueError, KeyError, TypeError):
                stage.update(
                    ok=False,
                    reason="local_input_snapshot_missing_or_changed",
                    returncode=1,
                )
        stage["output_hashes"] = file_hashes([*required_outputs, Path(stage["log"])])
        if stage["ok"]:
            stage["output_hashes"].update(inputs)
        stages.append(stage)
        save_checkpoint()
        return stage

    def finish() -> dict[str, Any]:
        if not intact_inputs(manifest):
            stages.append(
                {
                    "name": "input_version",
                    "ok": False,
                    "returncode": 1,
                    "log": "",
                    "reason": "alignment_inputs_changed_during_run",
                }
            )
        failed = next((stage for stage in stages if not stage["ok"]), None)
        error_code = ""
        if failed and failed.get("reason"):
            error_code = str(failed["reason"])
        elif failed and failed["name"] == "virtual_cloud" and compat_json.exists():
            try:
                error_code = str(
                    json.loads(compat_json.read_text(encoding="utf-8")).get(
                        "error_code"
                    )
                    or "virtual_cloud_failed"
                )
            except (OSError, json.JSONDecodeError):
                error_code = "virtual_cloud_failed"
        elif failed:
            error_code = f"{failed['name']}_failed"
        if error_code == "cloud_execution_pending":
            next_action = "wait_then_resume"
        elif error_code in BROWSER_ERROR_CODES:
            next_action = "repair_browser_then_resume"
        elif error_code.startswith("cloud_"):
            next_action = "repair_cloud_execution_then_rerun_cycle"
        elif error_code.startswith("local_factor_"):
            next_action = "repair_local_data_then_rerun_cycle"
        elif error_code == "alignment_metrics_below_threshold":
            next_action = "repair_source_then_next_round"
        elif failed:
            next_action = "repair_source_then_rerun_cycle"
        else:
            next_action = "complete" if real_cloud else "real_cloud_alignment"
        result = {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "manifest": str(output_dir / "alignment_manifest.json"),
            "layers": layer_results(stages),
            "status": "PASS" if failed is None else "FAIL",
            "failed_stage": "" if failed is None else failed["name"],
            "error_code": error_code,
            "repairable": failed is not None,
            "resumable": error_code in BROWSER_ERROR_CODES
            or error_code == "cloud_execution_pending",
            "checkpoint": str(checkpoint_path),
            "stages": stages,
            "evidence_paths": [stage["log"] for stage in stages if stage.get("log")]
            + [
                stage["diagnosis_path"]
                for stage in stages
                if stage.get("diagnosis_path")
            ]
            + [str(path) for path in (static_json, compat_json) if path.exists()],
            "prepared_dumper": str(prepared_dumper) if prepared_dumper.exists() else "",
            "bundle": str(bundle) if bundle.exists() else "",
            "final_metrics": next(
                (
                    stage.get("metrics")
                    for stage in reversed(stages)
                    if isinstance(stage.get("metrics"), dict)
                ),
                {},
            ),
            "worst_date_diagnosis": next(
                (
                    stage.get("worst_date_diagnosis")
                    for stage in reversed(stages)
                    if isinstance(stage.get("worst_date_diagnosis"), dict)
                ),
                {},
            ),
            "next_action": next_action,
        }
        result["artifact_hashes"] = file_hashes(
            [
                prepared_dumper,
                bundle,
                static_json,
                compat_json,
                output_dir / "alignment_manifest.json",
                output_dir / "cloud/cloud_factor_dump.csv",
                output_dir / "cloud/cloud_data_dumper_log.txt",
                output_dir / "comparison/alignment_results.json",
                *report_dir.glob("*_local_snapshot.csv"),
                output_dir / "local_analysis.json",
                output_dir / "local_factor_snapshot.csv",
                *(output_dir / "inputs").rglob("*.parquet"),
            ]
        )
        _atomic_write_json(output_dir / "alignment_cycle.json", result)
        if result_file is not None and _should_finalize(
            result,
            real_cloud=real_cloud,
            iteration=iteration,
            max_iterations=max_iterations,
        ):
            _write_final_artifacts(
                result=result,
                factor_path=factor_path,
                iteration=iteration,
                result_file=result_file,
                work_log=work_log or output_dir / "work_log.md",
            )
        _emit("cycle_finished", **result)
        return result

    preflight = (
        (
            "static_review",
            [
                sys.executable,
                str(SCRIPT_DIR / "static_code_review.py"),
                "--factor",
                str(factor_path),
                "--json",
                "--output",
                str(static_json),
            ],
            (static_json,),
            lambda stage: _classify_static_review(stage, static_json),
        ),
        (
            "prepare_dumper",
            [
                sys.executable,
                str(SCRIPT_DIR / "prepare_cloud_dumper.py"),
                "--factor",
                factor_class,
                "--output",
                str(prepared_dumper),
                "--fingerprint",
                fingerprint,
            ],
            (prepared_dumper,),
            None,
        ),
        (
            "build_strategy",
            [
                sys.executable,
                "scripts/build_jq_strategy.py",
                "--single-file",
                "--factor-files",
                str(factor_path),
                "--factors",
                factor_class,
                "--out",
                str(bundle),
            ],
            (bundle,),
            None,
        ),
        (
            "virtual_cloud",
            [
                sys.executable,
                str(SCRIPT_DIR / "virtual_cloud_check.py"),
                "--factor",
                str(factor_path),
                "--factor-class",
                factor_class,
                "--bundle",
                str(bundle),
                "--requirements",
                str(prepared_dumper),
                "--output",
                str(compat_json),
            ],
            (compat_json,),
            lambda stage: _attach_virtual_warnings(stage, compat_json),
        ),
    )
    for name, command, required_outputs, finalize_stage in preflight:
        if not run_stage(
            name,
            command,
            required_outputs=required_outputs,
            finalize=finalize_stage,
        )["ok"]:
            return finish()

    if not real_cloud:
        return finish()

    cloud_dir = output_dir / "cloud"
    cloud_csv = cloud_dir / "cloud_factor_dump.csv"
    cloud_log = cloud_dir / "cloud_data_dumper_log.txt"
    request_csv = cloud_dir / "requests.csv"
    report_dir = output_dir / "comparison"
    qlib_python = _conda_env_python("qlib_m3")
    local_analysis_command = (
        [str(qlib_python), "-m", "scripts.alignment.strategy_cycle"]
        if qlib_python is not None
        else [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "qlib_m3",
            "python",
            "-m",
            "scripts.alignment.strategy_cycle",
        ]
    )
    real_stages = (
        (
            "upload",
            [
                "bash",
                str(WORKSPACE_SCRIPTS / "upload_files_to_research.sh"),
                "--deploy",
                _relative_to_repo(bundle),
                "--dumper",
                _relative_to_repo(prepared_dumper),
            ],
            (),
        ),
        (
            "notebook",
            [sys.executable, str(WORKSPACE_SCRIPTS / "run_notebook.py")],
            (),
        ),
        (
            "download_requests",
            [sys.executable, str(SCRIPT_DIR / "request_manifest.py"),
             "--fingerprint", fingerprint, "--factor", factor_class,
             "--output", str(request_csv)],
            (request_csv,),
        ),
        (
            "local_analysis",
            local_analysis_command + [
                "--factor", str(factor_path), "--factor-class", factor_class,
                "--bundle", str(bundle), "--cloud-csv", str(request_csv),
                "--config", "config/small_mkt_config.json",
                "--output", str(output_dir), "--fingerprint", fingerprint,
            ],
            (output_dir / "local_analysis.json", output_dir / "local_factor_snapshot.csv"),
        ),
        (
            "wait_kernel",
            ["bash", str(WORKSPACE_SCRIPTS / "wait_kernel_idle.sh"), CLOUD_WAIT_SECONDS],
            (),
        ),
        (
            "download",
            [
                sys.executable,
                str(SCRIPT_DIR / "download_cloud_evidence.py"),
                "--fingerprint", fingerprint,
                "--output-dir",
                str(cloud_dir),
                "--log-filename",
                cloud_log.name,
            ],
            (cloud_csv, cloud_log),
        ),
    )
    for name, command, required_outputs in real_stages:
        if name == "download" and (
            not cloud_log.exists()
            or not cloud_log.read_text(encoding="utf-8", errors="replace").strip()
        ):
            reusing = False
        if name == "upload" and not (reusing and cached_stages.get(name, {}).get("ok")):
            remote_started = time.time()
            save_checkpoint()
        finalize_stage = _classify_local_analysis if name == "local_analysis" else None
        stage = run_stage(
            name,
            command,
            required_outputs=required_outputs,
            finalize=finalize_stage,
        )
        if not stage["ok"]:
            if name == "local_analysis":
                run_stage(
                    "wait_kernel",
                    ["bash", str(WORKSPACE_SCRIPTS / "wait_kernel_idle.sh"), CLOUD_WAIT_SECONDS],
                )
            return finish()

    cached_freshness = cached_stages.get("cloud_freshness")
    if (
        reusing
        and cached_freshness
        and cached_freshness.get("ok")
        and intact(cached_freshness.get("output_hashes", {}))
    ):
        freshness_stage = dict(cached_freshness)
        freshness_stage["output_hashes"] = file_hashes([cloud_csv, cloud_log])
        stages.append(freshness_stage)
        _emit("stage_resumed", stage="cloud_freshness", log="")
    else:
        reusing = False
        _emit("stage_started", stage="cloud_freshness", log="")
        artifact_error = _cloud_artifact_error(
            cloud_csv, cloud_log, factor_class, fingerprint
        )
        freshness_ok = (
            not artifact_error
            and cloud_csv.exists()
            and cloud_csv.stat().st_mtime >= remote_started
        )
        freshness_stage = {
            "name": "cloud_freshness",
            "ok": freshness_ok,
            "returncode": 0 if freshness_ok else 1,
            "log": "",
            "reason": ""
            if freshness_ok
            else artifact_error or "cloud_output_missing_or_stale",
        }
        freshness_stage["output_hashes"] = file_hashes([cloud_csv, cloud_log])
        stages.append(freshness_stage)
        _emit(
            "stage_ok" if freshness_ok else "stage_failed",
            stage="cloud_freshness",
            elapsed_seconds=0.0,
            error_code=str(freshness_stage.get("reason") or ""),
            log="",
        )
    save_checkpoint()
    if not freshness_stage["ok"]:
        return finish()

    request_stage = run_stage(
        "request_schedule_match",
        [sys.executable, str(SCRIPT_DIR / "request_manifest.py"),
         "--fingerprint", fingerprint, "--factor", factor_class,
         "--output", str(request_csv), "--cloud-csv", str(cloud_csv)],
    )
    if not request_stage["ok"]:
        return finish()

    comparison_stage = run_stage(
        "compare",
        [
            sys.executable,
            str(SCRIPT_DIR / "compare_factors.py"),
            "--factor",
            factor_class,
            "--cloud-csv",
            str(cloud_csv),
            "--local-csv",
            str(output_dir / "local_factor_snapshot.csv"),
            "--output",
            str(report_dir),
        ],
        required_outputs=(
            report_dir / "alignment_results.json",
            report_dir / (factor_class + "_local_snapshot.csv"),
        ),
        finalize=lambda stage: _apply_comparison_verdict(stage, report_dir),
    )
    if not comparison_stage["ok"]:
        return finish()
    return finish()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one script-driven pre-cloud alignment cycle"
    )
    parser.add_argument("--factor", required=True)
    parser.add_argument("--factor-class", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--real-cloud",
        action="store_true",
        help="Run local analysis and the existing real JoinQuant scripts after preflight",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful stages from this output directory when the factor is unchanged",
    )
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--work-log", type=Path)
    args = parser.parse_args()
    if (
        args.iteration < 1
        or args.max_iterations < 1
        or args.iteration > args.max_iterations
    ):
        parser.error("iteration must be between 1 and max-iterations")
    if args.work_log is not None and args.result_file is None:
        parser.error("--work-log requires --result-file")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".alignment_cycle.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _emit(
                "cycle_failed",
                status="FAIL",
                error_code="alignment_cycle_already_running",
                next_action="attach_to_existing_process",
            )
            return 3
        lock.write(str(os.getpid()))
        lock.flush()
        result = run_cycle(
            factor=args.factor,
            factor_class=args.factor_class,
            output_dir=output_dir,
            real_cloud=args.real_cloud,
            resume=args.resume,
            iteration=args.iteration,
            max_iterations=args.max_iterations,
            result_file=args.result_file,
            work_log=args.work_log,
        )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
