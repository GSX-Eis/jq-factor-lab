"""Validation helpers for generated symbolic factors."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import duckdb

from backtest_and_analysis.factor_loader import FactorLoader
from utils.data_store import environment_json, environment_value

from .config import load_analysis_context
from .data_loader import resolve_universe


def import_smoke_check(factor_file: str | Path) -> Dict[str, Any]:
    try:
        classes = FactorLoader.load_factor_class_from_file(str(factor_file))
        return {
            "ok": bool(classes),
            "classes": [cls.__name__ for cls in classes],
            "error": None,
        }
    except Exception as exc:
        return {"ok": False, "classes": [], "error": str(exc)}


def calculation_smoke_check(factor_file: str | Path, analysis_config: str) -> Dict[str, Any]:
    try:
        context = load_analysis_context(analysis_config)
        runtime = context.runtime_config
        code_list = runtime.get("code_list")
        if not code_list:
            try:
                code_list = resolve_universe(context)[:20]
            except Exception as exc:
                return {"ok": False, "rows": 0, "columns": 0, "error": f"failed to resolve dynamic universe for calculation smoke: {exc}"}
        if not code_list:
            return {"ok": False, "rows": 0, "columns": 0, "error": "analysis config resolved an empty universe for calculation smoke"}

        engine = duckdb.connect(str(runtime["duckdb_path"]), read_only=True)
        try:
            classes = FactorLoader.load_factor_class_from_file(str(factor_file))
            factors = [
                cls(
                    engine,
                    cls.__name__,
                    runtime["start_date"],
                    runtime["end_date"],
                    code_list,
                    cache_dir=runtime.get("cache_dir", "./outputs/cache"),
                    db_backend="duckdb",
                    duckdb_path=runtime.get("duckdb_path"),
                    parquet_dir=runtime.get("parquet_dir"),
                    use_duckdb_cache=runtime.get("use_duckdb_cache", True),
                    duckdb_cache_size_mb=runtime.get("duckdb_cache_size_mb", 2000),
                    price_table=runtime.get("price_table", "price_data_daily"),
                    fundamental_table=runtime.get("fundamental_table", "factor_data"),
                )
                for cls in classes
            ]
            if not factors:
                return {"ok": False, "rows": 0, "columns": 0, "error": "no factor instances loaded"}
            factor_df = factors[0].get_factor_values(force_recalculate=True)
            return {
                "ok": not factor_df.empty,
                "rows": int(factor_df.shape[0]),
                "columns": int(factor_df.shape[1]),
                "error": None if not factor_df.empty else "factor values are empty",
            }
        finally:
            engine.close()
    except Exception as exc:
        return {"ok": False, "rows": 0, "columns": 0, "error": str(exc)}


def run_analysis_smoke(factor_file: str | Path, analysis_config: str, timeout_sec: int = 600) -> Dict[str, Any]:
    command = [
        sys.executable,
        "run_analysis_main.py",
        "--config",
        str(analysis_config),
        "--factor",
        str(factor_file),
    ]
    env = dict(os.environ)
    context = environment_json("QUANT_OUTPUT_CONTEXT")
    if context:
        context.update(
            generated_factor_source_hash=hashlib.sha256(Path(factor_file).read_bytes()).hexdigest(),
            effective_analysis_config_hash=hashlib.sha256(Path(analysis_config).read_bytes()).hexdigest(),
        )
        env["QUANT_OUTPUT_CONTEXT"] = environment_value(context, Path(analysis_config).resolve().parent)
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec, env=env)
        return {
            "ok": proc.returncode == 0,
            "output_context": context,
            "exit_code": proc.returncode,
            "command": command,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": 3,
            "command": command,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": "analysis smoke timed out",
        }


def build_validation_summary(
    factor_file: str | Path,
    analysis_config: str,
    *,
    run_analysis: bool = True,
) -> Dict[str, Any]:
    import_result = import_smoke_check(factor_file)
    calculation_result = calculation_smoke_check(factor_file, analysis_config) if import_result["ok"] else {
        "ok": False,
        "rows": 0,
        "columns": 0,
        "error": "import smoke failed",
    }
    analysis_result = None
    if run_analysis and import_result["ok"]:
        analysis_result = run_analysis_smoke(factor_file, analysis_config)
    ok = bool(import_result["ok"] and calculation_result["ok"] and (analysis_result is None or analysis_result["ok"]))
    return {
        "ok": ok,
        "factor_file": str(factor_file),
        "analysis_config": str(analysis_config),
        "import": import_result,
        "calculation": calculation_result,
        "analysis": analysis_result,
    }
