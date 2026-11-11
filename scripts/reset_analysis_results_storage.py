#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.analysis_results_guard import (
    assert_no_dynamic_pool_config,
    assert_no_legacy_pooled_results_duckdb,
)
from utils.duckdb_results_writer import DuckDBResultsWriter


def discover_result_paths(config_root: Path) -> dict[str, Path]:
    config_path = config_root.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert_no_dynamic_pool_config(payload)
    results_cfg = payload.get("results_database", {}) if isinstance(payload, dict) else {}
    if not isinstance(results_cfg, dict):
        results_cfg = {}
    backend = str(results_cfg.get("db_backend", "auto") or "auto").lower()
    if backend == "sqlite":
        raise ValueError("SQLite results storage is no longer supported; configure DuckDB")
    if backend not in {"auto", "duckdb"}:
        raise ValueError("results_database.db_backend must be 'duckdb' or 'auto'")
    duckdb_path = str(results_cfg.get("duckdb_path") or "").strip()
    if not duckdb_path:
        raise ValueError("results_database.duckdb_path is required")
    paths: dict[str, Path] = {"duckdb": Path(duckdb_path).expanduser()}
    parquet_dir = str(results_cfg.get("parquet_dir") or "").strip()
    if parquet_dir:
        paths["parquet"] = Path(parquet_dir).expanduser()
    return paths


def backup_existing_results(paths: dict[str, Path], backup_root: Path) -> dict[str, Path]:
    backup_root.mkdir(parents=True, exist_ok=True)
    backups: dict[str, Path] = {}
    for name, path in paths.items():
        if not path.exists():
            continue
        target = backup_root / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)
        backups[name] = target
    return backups


def delete_existing_results(paths: dict[str, Path]) -> None:
    for path in paths.values():
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def initialize_clean_results_storage(paths: dict[str, Path]) -> None:
    duckdb_path = paths.get("duckdb")
    parquet_dir = paths.get("parquet")
    if duckdb_path is None:
        raise RuntimeError("DuckDB results path is required")
    DuckDBResultsWriter(
        duckdb_path=str(duckdb_path),
        parquet_dir=str(parquet_dir) if parquet_dir is not None else None,
        output_table="factor_analysis_results",
        factor_values_table="factor_values",
        monthly_output_table="factor_monthly_results",
        export_parquet=True,
    )


def validate_clean_storage(paths: dict[str, Path]) -> None:
    duckdb_path = paths.get("duckdb")
    parquet_dir = paths.get("parquet")
    if duckdb_path is None:
        raise RuntimeError("DuckDB results path is required")
    import duckdb

    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        assert_no_legacy_pooled_results_duckdb(conn)
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        required = {"factor_analysis_results", "factor_values", "factor_monthly_results"}
        missing = required - tables
        if missing:
            raise RuntimeError(f"missing clean result tables: {sorted(missing)}")
    finally:
        conn.close()
    if parquet_dir is not None:
        factor_values_root = parquet_dir / "factor_values"
        if factor_values_root.exists():
            for child in factor_values_root.rglob("*"):
                if "pool_type=" in child.as_posix():
                    raise RuntimeError("parquet still contains pool_type partitions after reset")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset pooled analysis results and initialize clean base-name-only storage")
    parser.add_argument(
        "--config",
        default=str((Path(__file__).resolve().parents[1] / "config" / "small_mkt_config.json")),
        help="Path to the active analysis JSON config",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    paths = discover_result_paths(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = config_path.parent / "backup" / timestamp
    backups = backup_existing_results(paths, backup_root)
    delete_existing_results(paths)
    initialize_clean_results_storage(paths)
    validate_clean_storage(paths)
    print(
        json.dumps(
            {
                "ok": True,
                "config": str(config_path),
                "paths": {key: str(value) for key, value in paths.items()},
                "backups": {key: str(value) for key, value in backups.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
