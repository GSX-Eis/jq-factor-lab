"""Configuration loading for symbolic factor mining."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from backtest_and_analysis.config import CONFIG as ANALYSIS_DEFAULT_CONFIG
from utils.config_loader import load_and_parse_config

from .artifacts import write_json


@dataclass
class AnalysisContext:
    config_path: str
    raw_config: Dict[str, Any]
    database_config: Dict[str, Any]
    analysis_config: Dict[str, Any]
    pool_configs: Dict[str, Any]
    runtime_config: Dict[str, Any]


def _normalize_duckdb_backend(
    runtime: Dict[str, Any], *, backend_key: str, path_key: str, component: str
) -> None:
    """Normalize a legacy automatic setting to the only supported backend."""
    configured = str(runtime.get(backend_key, "auto") or "auto").strip().lower()
    if configured == "sqlite":
        raise ValueError(
            f"{component} SQLite backend is no longer supported; migrate the config to DuckDB"
        )
    if configured not in {"auto", "duckdb"}:
        raise ValueError(
            f"{component} backend must be 'duckdb' or 'auto', got {configured!r}"
        )
    path = str(runtime.get(path_key) or "").strip()
    if not path:
        raise ValueError(f"{component} requires a DuckDB path in {path_key}")
    runtime[backend_key] = "duckdb"
    runtime[path_key] = str(Path(path).expanduser())


def load_mining_config(path: str) -> Dict[str, Any]:
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Mining config not found: {path}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mining config must be a JSON object")
    payload["_config_path"] = str(target.resolve())
    return payload


def _load_raw_json(path: str) -> Dict[str, Any]:
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Analysis config not found: {path}")
    return json.loads(target.read_text(encoding="utf-8"))


def load_analysis_context(analysis_config_path: str) -> AnalysisContext:
    raw = _load_raw_json(analysis_config_path)
    runtime = ANALYSIS_DEFAULT_CONFIG.copy()

    if "database" in raw or "analysis_settings" in raw or "dynamic_pools" in raw:
        db_config, analysis_config, pool_configs = load_and_parse_config(analysis_config_path)
        runtime["factor_table"] = db_config.get("factor_table", db_config.get("factor_data_table", runtime.get("factor_table", "factor_data")))
        runtime["price_table"] = db_config.get("price_table", runtime.get("price_table", "price_data_daily"))
        runtime["fundamental_table"] = db_config.get("fundamental_table", runtime.get("fundamental_table", "factor_data"))
        runtime["db_backend"] = db_config.get("db_backend", runtime.get("db_backend", "auto"))
        runtime["duckdb_path"] = db_config.get("duckdb_path", runtime.get("duckdb_path"))
        runtime["parquet_dir"] = db_config.get("parquet_dir", runtime.get("parquet_dir"))
        runtime["use_duckdb_cache"] = db_config.get("use_duckdb_cache", runtime.get("use_duckdb_cache", True))
        runtime["duckdb_cache_size_mb"] = db_config.get("duckdb_cache_size_mb", runtime.get("duckdb_cache_size_mb", 2000))
        runtime.update(analysis_config)
        runtime["dynamic_pools"] = pool_configs
        runtime["use_dynamic_pools"] = bool(pool_configs)
    else:
        db_config = {}
        pool_configs = {}
        analysis_config = raw
        runtime.update(raw)

    _normalize_duckdb_backend(
        runtime,
        backend_key="db_backend",
        path_key="duckdb_path",
        component="analysis database",
    )
    if runtime.get("results_duckdb_path") or runtime.get("results_db_backend"):
        _normalize_duckdb_backend(
            runtime,
            backend_key="results_db_backend",
            path_key="results_duckdb_path",
            component="analysis results database",
        )

    return AnalysisContext(
        config_path=str(Path(analysis_config_path).expanduser().resolve()),
        raw_config=raw,
        database_config=db_config,
        analysis_config=analysis_config,
        pool_configs=pool_configs,
        runtime_config=runtime,
    )


def _require_positive_int(search_config: Dict[str, Any], field: str) -> None:
    try:
        value = int(search_config.get(field))
    except Exception as exc:
        raise ValueError(f"search.{field} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"search.{field} must be a positive integer")


def _require_positive_float(search_config: Dict[str, Any], field: str) -> None:
    try:
        value = float(search_config.get(field))
    except Exception as exc:
        raise ValueError(f"search.{field} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"search.{field} must be a positive number")


def _require_rate(config: Dict[str, Any], field_path: str, value: Any) -> None:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{field_path} must be between 0 and 1") from exc
    if parsed < 0 or parsed > 1:
        raise ValueError(f"{field_path} must be between 0 and 1")


def _validate_dso_search_config(search_config: Dict[str, Any]) -> None:
    defaults = {
        "batch_size": 256,
        "epochs": 20,
        "max_length": 20,
        "epsilon": 0.2,
        "learning_rate": 0.001,
        "hall_top_k": 50,
    }
    merged = {**defaults, **search_config}
    for field in ("batch_size", "epochs", "max_length", "hall_top_k"):
        _require_positive_int(merged, field)
    for field in ("learning_rate",):
        _require_positive_float(merged, field)

    epsilon = float(merged["epsilon"])
    if epsilon <= 0 or epsilon > 1:
        raise ValueError("search.epsilon must be > 0 and <= 1")

    optimizer = str(search_config.get("optimizer", "risk_pg"))
    if optimizer not in {"risk_pg", "ppo"}:
        raise ValueError("search.optimizer must be one of: risk_pg, ppo")
    if optimizer == "ppo":
        _require_positive_float({**search_config, "ppo_clip": search_config.get("ppo_clip", 0.2)}, "ppo_clip")
        _require_positive_int({**search_config, "ppo_epochs": search_config.get("ppo_epochs", 2)}, "ppo_epochs")
        _require_positive_int(
            {**search_config, "ppo_minibatch_size": search_config.get("ppo_minibatch_size", merged["batch_size"])},
            "ppo_minibatch_size",
        )

    device = str(search_config.get("device", "auto"))
    if device not in {"auto", "cpu", "mps", "cuda"} and not device.startswith("cuda:"):
        raise ValueError("search.device must be auto, cpu, mps, cuda, or cuda:<index>")

    constants = search_config.get("constants", {})
    if constants.get("enabled", False):
        values = constants.get("values", [])
        if not isinstance(values, list) or not values:
            raise ValueError("search.constants.values must be a non-empty list when constants are enabled")
        parsed_values = []
        for idx, raw_value in enumerate(values):
            try:
                parsed = float(raw_value)
            except Exception as exc:
                raise ValueError(f"search.constants.values[{idx}] must be numeric") from exc
            if not parsed == parsed or parsed in {float("inf"), float("-inf")}:
                raise ValueError(f"search.constants.values[{idx}] must be finite")
            parsed_values.append(parsed)
        if len(set(parsed_values)) != len(parsed_values):
            raise ValueError("search.constants.values must not contain duplicates")

    windows = search_config.get("windows", [3, 5, 10, 20, 60])
    if not isinstance(windows, list) or not windows:
        raise ValueError("search.windows must be a non-empty list of positive integers")
    parsed_windows = []
    for idx, raw_window in enumerate(windows):
        try:
            parsed = int(raw_window)
        except Exception as exc:
            raise ValueError(f"search.windows[{idx}] must be a positive integer") from exc
        if parsed <= 0 or float(raw_window) != float(parsed):
            raise ValueError(f"search.windows[{idx}] must be a positive integer")
        parsed_windows.append(parsed)
    if len(set(parsed_windows)) != len(parsed_windows):
        raise ValueError("search.windows must not contain duplicates")

    optimize = constants.get("optimize", {})
    if optimize.get("enabled", False):
        _require_positive_int({**optimize, "maxiter": optimize.get("maxiter", 20)}, "maxiter")
        _require_positive_int({**optimize, "max_constants": optimize.get("max_constants", 4)}, "max_constants")

    priors = search_config.get("priors", {})
    for field in (
        "max_depth",
        "max_complexity",
        "min_feature_refs",
        "max_rolling_depth",
        "max_binary_rolling_ops",
        "max_window",
    ):
        if field in priors and priors[field] is not None:
            try:
                parsed = int(priors[field])
            except Exception as exc:
                raise ValueError(f"search.priors.{field} must be a positive integer") from exc
            if parsed <= 0:
                raise ValueError(f"search.priors.{field} must be a positive integer")

    gp_meld = search_config.get("gp_meld", {})
    if gp_meld.get("enabled", False):
        for field in ("every_epochs", "population_size", "generations", "inject_top_k"):
            _require_positive_int({**gp_meld, field: gp_meld.get(field, 1)}, field)
        _require_rate(gp_meld, "search.gp_meld.crossover_rate", gp_meld.get("crossover_rate", 0.5))
        _require_rate(gp_meld, "search.gp_meld.mutation_rate", gp_meld.get("mutation_rate", 0.3))


def _validate_post_rerank_config(config: Dict[str, Any]) -> None:
    rerank = config.get("post_rerank", {})
    if not rerank or not rerank.get("enabled", False):
        return

    _require_positive_int({**rerank, "top_k": rerank.get("top_k", 10)}, "top_k")
    if "analysis_timeout_sec" in rerank:
        _require_positive_int(rerank, "analysis_timeout_sec")

    periods = rerank.get("periods", [1, 5, 10])
    if not isinstance(periods, list) or not periods:
        raise ValueError("post_rerank.periods must be a non-empty list of positive integers")
    seen_periods = set()
    for idx, raw_period in enumerate(periods):
        try:
            parsed = int(raw_period)
        except Exception as exc:
            raise ValueError(f"post_rerank.periods[{idx}] must be a positive integer") from exc
        if parsed <= 0 or float(raw_period) != float(parsed):
            raise ValueError(f"post_rerank.periods[{idx}] must be a positive integer")
        seen_periods.add(parsed)
    if len(seen_periods) != len(periods):
        raise ValueError("post_rerank.periods must not contain duplicates")

    for field in (
        "validation_start",
        "validation_end",
        "holdout_confirm_start",
        "holdout_confirm_end",
    ):
        if field in rerank:
            try:
                datetime.fromisoformat(str(rerank[field]))
            except Exception as exc:
                raise ValueError(f"post_rerank.{field} must be an ISO date string") from exc

    for field in (
        "min_rank_ic_mean",
        "min_top_quantile_excess_return",
        "min_top_quantile_excess_t",
        "min_valid_rank_ic_t",
    ):
        if field in rerank:
            try:
                parsed = float(rerank[field])
            except Exception as exc:
                raise ValueError(f"post_rerank.{field} must be numeric") from exc
            if not parsed == parsed or parsed in {float("inf"), float("-inf")}:
                raise ValueError(f"post_rerank.{field} must be finite")

    if "holdout_confirm_looks" in rerank:
        _require_positive_int(rerank, "holdout_confirm_looks")

    constant_optimization = str(rerank.get("constant_optimization", "none"))
    if constant_optimization not in {"none", "disabled", "top_k_constants_only"}:
        raise ValueError("post_rerank.constant_optimization must be none, disabled, or top_k_constants_only")

    portfolio_gate = rerank.get("portfolio_gate", {})
    if portfolio_gate and portfolio_gate.get("enabled", False):
        mode = str(portfolio_gate.get("mode", "hard"))
        if mode != "hard":
            raise ValueError("post_rerank.portfolio_gate.mode must be hard")
        reference_factors = portfolio_gate.get("reference_factors", [])
        if not isinstance(reference_factors, list) or not reference_factors:
            raise ValueError("post_rerank.portfolio_gate.reference_factors must be a non-empty list")
        if any(not str(item).strip() for item in reference_factors):
            raise ValueError("post_rerank.portfolio_gate.reference_factors must contain non-empty names")
        for field in ("max_abs_factor_corr", "min_marginal_sharpe_delta"):
            if field in portfolio_gate:
                try:
                    parsed = float(portfolio_gate[field])
                except Exception as exc:
                    raise ValueError(f"post_rerank.portfolio_gate.{field} must be numeric") from exc
                if not parsed == parsed or parsed in {float("inf"), float("-inf")}:
                    raise ValueError(f"post_rerank.portfolio_gate.{field} must be finite")


def _validate_objective_config(config: Dict[str, Any]) -> None:
    objective = config.get("objective", {})
    return_proxy = objective.get("return_proxy", {})
    if return_proxy and return_proxy.get("enabled", False):
        _require_positive_int({**return_proxy, "quantiles": return_proxy.get("quantiles", 5)}, "quantiles")
        _require_positive_int({**return_proxy, "min_group_size": return_proxy.get("min_group_size", 1)}, "min_group_size")
        for field in ("weight", "mean_weight", "ir_weight", "win_rate_weight"):
            if field in return_proxy:
                try:
                    parsed = float(return_proxy[field])
                except Exception as exc:
                    raise ValueError(f"objective.return_proxy.{field} must be numeric") from exc
                if not parsed == parsed or parsed in {float("inf"), float("-inf")}:
                    raise ValueError(f"objective.return_proxy.{field} must be finite")


def validate_mining_config(config: Dict[str, Any], analysis_context: AnalysisContext) -> None:
    required_sections = ("analysis_config", "search", "features", "objective", "output")
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Mining config missing required sections: {missing}")

    backend = config.get("search", {}).get("backend")
    if backend not in {"random_v0", "dso_pytorch"}:
        raise ValueError(f"Unsupported symbolic search backend: {backend}")
    if backend == "dso_pytorch":
        _validate_dso_search_config(config.get("search", {}))
    _validate_objective_config(config)
    _validate_post_rerank_config(config)

    if not analysis_context.runtime_config.get("start_date") or not analysis_context.runtime_config.get("end_date"):
        raise ValueError("Analysis context must define start_date and end_date")
    if config.get("post_rerank", {}).get("enabled", False):
        runtime = analysis_context.runtime_config
        if not runtime.get("results_duckdb_path"):
            raise ValueError("post_rerank requires a DuckDB analysis results database")

    feature_config = config.get("features", {})
    if not any(feature_config.get(key) for key in ("daily_price_fields", "intraday_5m_fields", "factor_data_fields")):
        raise ValueError("At least one feature field must be configured")


def make_run_id(config: Dict[str, Any]) -> str:
    seed_material = json.dumps(
        {
            "config_path": config.get("_config_path", ""),
            "search": config.get("search", {}),
            "features": config.get("features", {}),
            "time": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:8]
    return f"sym_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{digest}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def snapshot_config(config: Dict[str, Any], analysis_context: AnalysisContext, output_dir: str | Path) -> None:
    write_json(
        Path(output_dir) / "config_snapshot.json",
        {
            "mining_config": _json_safe(config),
            "analysis_config_path": analysis_context.config_path,
            "analysis_runtime_config": _json_safe(analysis_context.runtime_config),
            "dynamic_pool_names": list(analysis_context.pool_configs),
        },
    )
