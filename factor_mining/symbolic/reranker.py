"""Post-search validation reranking for symbolic mining."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

try:
    import duckdb
    from utils.data_store import connect_read_only as _connect_data_read_only
except ImportError:  # pragma: no cover - environment dependent
    duckdb = None

from .artifacts import write_json
from .compiler import compile_factor_file
from .config import AnalysisContext
from .data_loader import _configure_duckdb_spill
from .ensemble import (
    EnsembleMember,
    build_ensemble_expression,
    select_ensemble_members,
)
from .expression import (
    deserialize_expression,
    evaluate_expression,
    evaluate_expression_cached,
    expression_complexity,
    expression_hash,
    iter_constants,
    referenced_features,
    replace_constants,
    serialize_expression,
)
from .feature_space import FeatureSpace
from .objective import (
    ResidualizationContext,
    ScoringContext,
    _bucket_labels,
    _long_short_from_arrays,
    anchor_matrix_from_panel,
    prepare_score,
    rank_ic_by_date,
    residualize_by_date,
    score_expression,
)
from .search_base import SearchCandidate, SearchResult
from utils.data_store import binding_contract, resolve_connection_binding, table_name
from .search_random import _label_data, score_candidate
from .validator import import_smoke_check, run_analysis_smoke


@dataclass(frozen=True)
class PostRerankResult:
    selected_candidate: SearchCandidate
    selected_factor_path: Path | None
    dual_gate_passed: bool
    summary: Dict[str, Any]


def _stable_hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _task_manager_ast_hash(expr) -> str:
    return _stable_hash_payload({"ast": serialize_expression(expr)})


def _metric_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any) -> str:
    numeric = _metric_float(value)
    return "none" if numeric is None else f"{numeric:.6f}"


def _expression_eval_cache(config: dict) -> dict | None:
    if bool(config.get("_disable_expression_cache", False)):
        return None
    return config.setdefault("_expression_eval_cache", {})


def _evaluate_expression_for_rerank(
    expr: Any, panel: pd.DataFrame, config: dict
) -> pd.Series:
    cache = _expression_eval_cache(config)
    if cache is None:
        return evaluate_expression(expr, panel)
    return evaluate_expression_cached(expr, panel, cache)


_EULER_MASCHERONI = 0.5772156649015329


def _ic_tstat(
    ic_values: Any,
    *,
    method: str = "newey_west",
    lag: int = 0,
) -> float | None:
    """t-statistic of a daily rank-IC series.

    ``simple`` uses the iid standard error ``std/sqrt(n)`` (= IR * sqrt(n)).
    ``newey_west`` uses a Bartlett-kernel HAC long-run variance up to ``lag``
    lags, which corrects the autocorrelation induced by overlapping multi-day
    forward-return labels (a 5/10/20-day label makes consecutive daily ICs
    positively autocorrelated, so the naive t over-states significance).

    Returns ``None`` when the series is too short (< 3 finite points) or has no
    dispersion.
    """
    arr = np.asarray(ic_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return None
    mean = float(arr.mean())
    # Dispersion floor on the IC scale (correlations in [-1, 1]); anything below
    # is numerically constant (e.g. float noise from a degenerate series).
    if float(arr.std(ddof=1)) <= 1e-12:
        return None
    if str(method) == "simple":
        se = float(arr.std(ddof=1)) / math.sqrt(n)
    else:
        demeaned = arr - mean
        gamma0 = float(np.dot(demeaned, demeaned) / n)
        max_lag = min(max(0, int(lag)), n - 1)
        lrv = gamma0
        for k in range(1, max_lag + 1):
            weight = 1.0 - k / (max_lag + 1)
            cov = float(np.dot(demeaned[k:], demeaned[:-k]) / n)
            lrv += 2.0 * weight * cov
        if lrv <= 1e-24:
            return None
        se = math.sqrt(lrv / n)
    if se <= 1e-15:
        return None
    value = mean / se
    return float(value) if np.isfinite(value) else None


def _deflated_t_hurdle(n_trials: int) -> float | None:
    """Selection-aware t hurdle = expected maximum of ``n_trials`` iid N(0,1).

    This is the multiple-testing threshold from the Deflated Sharpe Ratio
    (Bailey & Lopez de Prado): with ``N`` independent search trials the most
    extreme t-stat is expected to reach this level by luck alone, so a genuine
    factor's t must exceed it. ``None`` when ``N < 2`` (no correction possible).
    """
    n = int(n_trials)
    if n < 2:
        return None
    normal = NormalDist()
    z1 = normal.inv_cdf(1.0 - 1.0 / n)
    z2 = normal.inv_cdf(1.0 - 1.0 / (n * math.e))
    hurdle = (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    return float(hurdle) if np.isfinite(hurdle) else None


def _selected_period_tstat(oos_metrics: Any, best_period: Any) -> float | None:
    """IC t-stat of the selected candidate at its gate period.

    Prefers the dual-gate ``best_period`` (the period the selection used) and
    falls back to the period with the strongest OOS rank-IC mean -- matching how
    the decay gate picks a like-for-like period downstream.
    """
    if not isinstance(oos_metrics, dict) or not oos_metrics:
        return None
    if best_period is not None:
        entry = oos_metrics.get(str(best_period))
        if isinstance(entry, dict):
            value = _metric_float(entry.get("rank_ic_t"))
            if value is not None:
                return value
    best_key: str | None = None
    best_val: float | None = None
    for key, entry in oos_metrics.items():
        if not isinstance(entry, dict):
            continue
        value = _metric_float(entry.get("rank_ic_mean"))
        if value is not None and (best_val is None or value > best_val):
            best_val = value
            best_key = str(key)
    if best_key is None:
        return None
    entry = oos_metrics.get(best_key)
    return _metric_float(entry.get("rank_ic_t")) if isinstance(entry, dict) else None


def _selected_period_rank_ic(oos_metrics: Any, best_period: Any) -> float | None:
    """Rank-IC mean of the selected candidate at its gate period.

    Mirrors :func:`_selected_period_tstat` but returns the mean rank-IC, so a
    selection-window metric (e.g. the 2024 validation IC fed to the config
    search) is read at the same period as its t-stat.
    """
    if not isinstance(oos_metrics, dict) or not oos_metrics:
        return None
    if best_period is not None:
        entry = oos_metrics.get(str(best_period))
        if isinstance(entry, dict):
            value = _metric_float(entry.get("rank_ic_mean"))
            if value is not None:
                return value
    best_key: str | None = None
    best_val: float | None = None
    for key, entry in oos_metrics.items():
        if not isinstance(entry, dict):
            continue
        value = _metric_float(entry.get("rank_ic_mean"))
        if value is not None and (best_val is None or value > best_val):
            best_val = value
            best_key = str(key)
    if best_key is None:
        return None
    entry = oos_metrics.get(best_key)
    return _metric_float(entry.get("rank_ic_mean")) if isinstance(entry, dict) else None


def _resolve_effective_trials(rerank_config: dict) -> int:
    """Number of independent holdout looks ``N`` for the deflated t hurdle.

    Selection (dual gate / top-K / QD / ensemble / PBO) happens entirely on the
    selection window, so per-run expression breadth no longer consumes holdout
    multiplicity. The only thing the deflated-t hurdle must price is how many
    times the campaign has *confirmed* a survivor on the holdout: the
    orchestrator injects ``holdout_confirm_looks`` (= historical confirm count
    + 1 for this run's own look). Standalone runs without injection default to
    1 (a single look needs no max-of-N correction).
    """
    return max(1, int(rerank_config.get("holdout_confirm_looks", 1) or 1))


def _best_lightweight_miss(candidate_rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    best_key = (-1.0e18, -1.0e18, -1.0e18, 0.0)
    for row in candidate_rows:
        if row.get("duplicate_seed"):
            continue
        rank = int(row.get("rank") or 0)
        periods = row.get("lightweight_period_metrics")
        if not isinstance(periods, dict):
            continue
        for period, metrics in periods.items():
            if not isinstance(metrics, dict):
                continue
            rank_ic = _metric_float(metrics.get("rank_ic_mean"))
            rank_ic_t = _metric_float(metrics.get("rank_ic_t"))
            excess = _metric_float(metrics.get("top_quantile_excess_return_mean"))
            excess_t = _metric_float(metrics.get("top_quantile_excess_return_t"))
            key = (
                rank_ic if rank_ic is not None else -1.0e18,
                excess if excess is not None else -1.0e18,
                excess_t if excess_t is not None else -1.0e18,
                -float(rank),
            )
            if key > best_key:
                candidate = (
                    row.get("candidate")
                    if isinstance(row.get("candidate"), dict)
                    else {}
                )
                best_key = key
                best = {
                    "rank": rank,
                    "period": str(period),
                    "expression": str(candidate.get("expression") or ""),
                    "rank_ic_mean": rank_ic,
                    "rank_ic_t": rank_ic_t,
                    "top_quantile_excess_return_mean": excess,
                    "top_quantile_excess_return_t": excess_t,
                }
    return best


def _quality_gate_failure_reason(
    *,
    candidate_rows: list[Dict[str, Any]],
    qd_bucket_count: int,
    seed_variant_lightweight_pass_count: int,
    rerank_config: dict,
) -> str:
    thresholds = {
        "min_rank_ic_mean": float(rerank_config.get("min_rank_ic_mean", 0.08)),
        "min_valid_rank_ic_t": float(rerank_config.get("min_valid_rank_ic_t", 0.0)),
        "min_top_quantile_excess_return": float(
            rerank_config.get("min_top_quantile_excess_return", 0.0)
        ),
        "min_top_quantile_excess_t": float(
            rerank_config.get("min_top_quantile_excess_t", 0.0)
        ),
    }
    nonduplicate_count = sum(
        1 for row in candidate_rows if not bool(row.get("duplicate_seed"))
    )
    dual_gate_count = sum(
        1
        for row in candidate_rows
        if not bool(row.get("duplicate_seed")) and bool(row.get("dual_gate_passed"))
    )
    correlation_blocked = [
        row
        for row in candidate_rows
        if not bool(row.get("duplicate_seed"))
        and bool(row.get("lightweight_primary_gate_passed"))
        and not bool(row.get("portfolio_gate", {}).get("passed", True))
    ]
    best = _best_lightweight_miss(candidate_rows)
    message = (
        "post_rerank_quality_gate_failed: "
        "no final candidate passed lightweight/QD gates; "
        f"topk={len(candidate_rows)} nonduplicate={nonduplicate_count} "
        f"dual_gate_passed={dual_gate_count} qd_buckets={qd_bucket_count} "
        f"seed_variants_lightweight_passed={seed_variant_lightweight_pass_count}; "
        "thresholds="
        f"rank_ic>={thresholds['min_rank_ic_mean']:.6f},"
        f"rank_ic_t>={thresholds['min_valid_rank_ic_t']:.6f},"
        f"top_quantile_excess>{thresholds['min_top_quantile_excess_return']:.6f},"
        f"excess_t>={thresholds['min_top_quantile_excess_t']:.6f}"
    )
    if correlation_blocked:
        gate_cfg = rerank_config.get("portfolio_gate", {}) or {}
        threshold = float(gate_cfg.get("max_abs_factor_corr", 0.7))
        references = gate_cfg.get("reference_factors", [])
        observed = [
            float(row["portfolio_gate"]["observed_mean_abs_corr"])
            for row in correlation_blocked
            if row.get("portfolio_gate", {}).get("observed_mean_abs_corr") is not None
        ]
        message += (
            "; correlation_gate_blocked="
            f"{len(correlation_blocked)} (mean_abs_corr>{threshold:.2f} "
            f"vs reference_factors={list(references)}"
        )
        if observed:
            message += f"; min_blocked_corr={min(observed):.4f}"
        message += ")"
    if best:
        message += (
            "; best_miss="
            f"rank={best['rank']} period={best['period']} "
            f"rank_ic={_format_metric(best.get('rank_ic_mean'))} "
            f"rank_ic_t={_format_metric(best.get('rank_ic_t'))} "
            f"top_quantile_excess={_format_metric(best.get('top_quantile_excess_return_mean'))} "
            f"excess_t={_format_metric(best.get('top_quantile_excess_return_t'))} "
            f"expression={best.get('expression')}"
        )
    return message


def _load_novelty_context(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        raise ValueError("symbolic post-rerank requires --novelty-context")
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"novelty context not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"novelty context is invalid JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"novelty context must be a JSON object: {target}")
    hashes = payload.get("existing_expression_hashes")
    if not isinstance(hashes, list):
        raise ValueError("novelty context requires existing_expression_hashes[]")
    return {
        **payload,
        "loaded": True,
        "path": str(target.resolve()),
        "existing_expression_hashes": {
            str(item).strip() for item in hashes if str(item).strip()
        },
    }


def post_rerank_enabled(config: dict) -> bool:
    return bool(config.get("post_rerank", {}).get("enabled", False))


def _validation_window(config: dict) -> tuple[str, str]:
    rerank_config = config.get("post_rerank", {})
    start = str(rerank_config.get("validation_start", "2024-01-01"))
    end = str(rerank_config.get("validation_end", "2024-12-31"))
    return start, end


def _validation_month_window(start: str, end: str) -> tuple[str, str]:
    return pd.Timestamp(start).strftime("%Y-%m"), pd.Timestamp(end).strftime("%Y-%m")


def _quote_identifier(name: str) -> str:
    return table_name(name)


def _period_list_sql(periods: Iterable[int]) -> str:
    values = sorted({int(period) for period in periods})
    if not values:
        raise ValueError("post_rerank.periods must not be empty")
    return ", ".join(str(value) for value in values)


def _candidate_factor_path(factor_dir: Path, run_id: str, rank: int) -> Path:
    return factor_dir / f"symbolic_mined_factor_{run_id}_rerank_{rank:02d}.py"


def _write_smoke_analysis_config(
    analysis_config_path: str,
    run_root: Path,
    rerank_config: dict,
) -> tuple[str, dict]:
    smoke_config = rerank_config.get("smoke_analysis", {})
    if smoke_config.get("enabled") is False:
        return analysis_config_path, {
            "enabled": False,
            "config_path": str(analysis_config_path),
        }

    source_path = Path(analysis_config_path)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception:
        return analysis_config_path, {
            "enabled": False,
            "config_path": str(analysis_config_path),
            "reason": "analysis_config_not_json",
        }

    validation_start = pd.Timestamp(
        str(rerank_config.get("validation_start", "2024-01-01"))
    ).normalize()
    validation_end = pd.Timestamp(
        str(rerank_config.get("validation_end", validation_start))
    ).normalize()
    smoke_start = pd.Timestamp(
        str(smoke_config.get("start_date", smoke_config.get("start", validation_start)))
    ).normalize()
    if smoke_config.get("end_date") or smoke_config.get("end"):
        smoke_end = pd.Timestamp(
            str(smoke_config.get("end_date", smoke_config.get("end")))
        ).normalize()
    else:
        smoke_days = max(10, int(smoke_config.get("days", 90)))
        smoke_end = min(validation_end, smoke_start + pd.Timedelta(days=smoke_days - 1))
    if smoke_end < smoke_start:
        smoke_end = smoke_start

    raw.setdefault("analysis_period", {})
    raw["analysis_period"]["start_date"] = smoke_start.strftime("%Y-%m-%d")
    raw["analysis_period"]["end_date"] = smoke_end.strftime("%Y-%m-%d")

    raw.setdefault("analysis_settings", {})
    raw["analysis_settings"]["cache_dir"] = str(
        (run_root / "analysis_smoke_cache").resolve()
    )
    raw["analysis_settings"]["max_workers"] = int(smoke_config.get("max_workers", 1))

    duckdb_path = run_root / "analysis_smoke_results.duckdb"
    raw.setdefault("results_database", {})
    raw["results_database"]["db_backend"] = "duckdb"
    raw["results_database"]["duckdb_path"] = str(duckdb_path.resolve())
    raw["results_database"]["parquet_dir"] = str(
        (run_root / "analysis_smoke_parquet").resolve()
    )
    raw["results_database"]["export_parquet"] = bool(
        smoke_config.get("export_parquet", False)
    )
    raw["results_database"]["output_table"] = str(
        smoke_config.get("output_table", "factor_analysis_results")
    )
    raw["results_database"]["monthly_output_table"] = str(
        smoke_config.get("monthly_output_table", "monthly_pool_results")
    )

    target = run_root / "post_rerank_smoke_analysis_config.json"
    write_json(target, raw)
    return str(target), {
        "enabled": True,
        "config_path": str(target.resolve()),
        "source_config_path": str(source_path.resolve()),
        "start_date": smoke_start.strftime("%Y-%m-%d"),
        "end_date": smoke_end.strftime("%Y-%m-%d"),
        "results_backend": "duckdb",
        "results_db": str(duckdb_path.resolve()),
        "export_parquet": bool(raw["results_database"]["export_parquet"]),
    }


def _optimize_constants_for_train(
    candidate: SearchCandidate,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    config: dict,
) -> tuple[SearchCandidate, Dict[str, Any]]:
    values = list(iter_constants(candidate.expression))
    info: Dict[str, Any] = {
        "attempted": False,
        "optimized": False,
        "reason": "",
        "constant_count": int(len(values)),
        "train_rows": int(len(train_panel)),
        "train_start": pd.Timestamp(train_panel["date"].min()).strftime("%Y-%m-%d")
        if not train_panel.empty
        else "",
        "train_end": pd.Timestamp(train_panel["date"].max()).strftime("%Y-%m-%d")
        if not train_panel.empty
        else "",
    }
    mode = str(config.get("post_rerank", {}).get("constant_optimization", "none"))
    if mode != "top_k_constants_only":
        info["reason"] = "disabled"
        return candidate, info
    if not values:
        info["reason"] = "no_constants"
        return candidate, info

    optimize_config = config.get("search", {}).get("constants", {}).get("optimize", {})
    max_constants = int(optimize_config.get("max_constants", 4))
    if len(values) > max_constants:
        info["reason"] = "too_many_constants"
        return candidate, info

    try:
        from scipy.optimize import minimize
    except Exception:
        info["reason"] = "scipy_unavailable"
        return candidate, info

    info["attempted"] = True
    original_values = np.asarray(values, dtype=float)
    best_values = original_values.copy()
    best_score = -math.inf
    train_anchors = anchor_matrix_from_panel(train_panel)

    def train_score(raw_values: np.ndarray) -> float:
        nonlocal best_score, best_values
        try:
            expr = replace_constants(
                candidate.expression, [float(value) for value in raw_values]
            )
            pred = _evaluate_expression_for_rerank(expr, train_panel, config)
            metrics = score_expression(
                pred,
                _label_data(train_panel),
                train_panel["date"],
                expr,
                config,
                anchors=train_anchors,
            )
            score = float(metrics.get("score", -math.inf))
        except Exception:
            return -math.inf
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_values = np.asarray(raw_values, dtype=float)
        return score if np.isfinite(score) else -math.inf

    train_score(original_values)
    bounds = None
    if "min_value" in optimize_config and "max_value" in optimize_config:
        bounds = [
            (float(optimize_config["min_value"]), float(optimize_config["max_value"]))
            for _ in range(len(original_values))
        ]

    try:
        minimize(
            lambda raw_values: -train_score(np.asarray(raw_values, dtype=float)),
            original_values,
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(optimize_config.get("maxiter", 20)), "disp": False},
        )
    except Exception as exc:
        info["reason"] = f"minimize_failed: {exc}"
        return candidate, info

    if not np.isfinite(best_score):
        info["reason"] = "no_finite_train_score"
        return candidate, info

    optimized_expr = replace_constants(
        candidate.expression, [float(value) for value in best_values]
    )
    optimized_candidate = score_candidate(
        optimized_expr, train_panel, valid_panel, config
    )
    if optimized_candidate is None:
        info["reason"] = "optimized_score_failed"
        return candidate, info

    info.update(
        {
            "optimized": optimized_expr.to_string() != candidate.expression.to_string(),
            "reason": "ok",
            "original_values": [float(value) for value in original_values],
            "optimized_values": [float(value) for value in best_values],
            "train_score": float(best_score),
        }
    )
    return optimized_candidate, info


def _fetch_duckdb_df(path: str, sql: str, params: list[Any], *, table: str, allow_labels=False) -> pd.DataFrame:
    if duckdb is None:
        raise RuntimeError("duckdb is not available")
    contract = binding_contract()
    if contract:
        destination, selected = resolve_connection_binding(path, contract, allow_labels=allow_labels)
        if not contract.get("output_db") or Path(destination).resolve() != Path(contract["output_db"]).resolve():
            raise ValueError("Reranker requires an output database binding")
        if table.removeprefix("main.") not in {name.removeprefix("main.") for name in selected.get("outputs", {})}:
            raise ValueError(f"Reranker output table has no frozen version: {table}")
    conn = _connect_data_read_only(path, allow_labels=allow_labels)
    try:
        _configure_duckdb_spill(conn)
        return conn.execute(sql, params).df()
    finally:
        conn.close()


def _query_validation_metrics(
    analysis_context: AnalysisContext,
    factor_name: str,
    rerank_config: dict,
) -> tuple[Dict[str, Dict[str, float | int | None]], list[str]]:
    runtime = analysis_context.runtime_config
    periods = [int(period) for period in rerank_config.get("periods", [1, 5, 10])]
    period_sql = _period_list_sql(periods)
    validation_start = str(rerank_config.get("validation_start", "2024-01-01"))
    validation_end = str(rerank_config.get("validation_end", "2024-12-31"))
    month_start, month_end = _validation_month_window(validation_start, validation_end)
    factor_table = _quote_identifier(
        runtime.get("results_output_table", "factor_analysis_results")
    )
    monthly_table = _quote_identifier(
        runtime.get("results_monthly_output_table", "monthly_pool_results")
    )
    warnings: list[str] = []

    duckdb_path = _results_duckdb_path(runtime)
    factor_sql = f"""
            SELECT CAST(period AS INTEGER) AS period,
                   AVG(CAST(ic AS DOUBLE)) AS rank_ic_mean,
                   COUNT(ic) AS rank_ic_count
            FROM {factor_table}
            WHERE factor_name = ?
              AND CAST(date AS DATE) >= CAST(? AS DATE)
              AND CAST(date AS DATE) <= CAST(? AS DATE)
              AND CAST(period AS INTEGER) IN ({period_sql})
            GROUP BY CAST(period AS INTEGER)
    """
    monthly_sql = f"""
            SELECT CAST(period_days AS INTEGER) AS period,
                   AVG(CAST(long_short_return AS DOUBLE)) AS long_short_return_mean,
                   AVG(
                       CASE
                           WHEN long_short_return IS NULL THEN NULL
                           WHEN CAST(long_short_return AS DOUBLE) > 0 THEN 1.0
                           ELSE 0.0
                       END
                   ) AS long_short_return_win_rate,
                   COUNT(long_short_return) AS long_short_return_count
            FROM {monthly_table}
            WHERE factor_name = ?
              AND SUBSTR(CAST(year_month AS VARCHAR), 1, 7) >= ?
              AND SUBSTR(CAST(year_month AS VARCHAR), 1, 7) <= ?
              AND CAST(period_days AS INTEGER) IN ({period_sql})
            GROUP BY CAST(period_days AS INTEGER)
    """
    factor_df = _fetch_duckdb_df(
        duckdb_path,
        factor_sql,
        [factor_name, validation_start, validation_end],
        table=runtime.get("results_output_table", "factor_analysis_results"),
        allow_labels=True,
    )
    monthly_df = _fetch_duckdb_df(
        duckdb_path,
        monthly_sql,
        [factor_name, month_start, month_end],
        table=runtime.get("results_monthly_output_table", "monthly_pool_results"),
        allow_labels=True,
    )

    metrics: Dict[str, Dict[str, float | int | None]] = {
        str(period): {
            "rank_ic_mean": None,
            "rank_ic_count": 0,
            "long_short_return_mean": None,
            "long_short_return_win_rate": None,
            "long_short_return_count": 0,
        }
        for period in periods
    }
    for _, row in factor_df.iterrows():
        key = str(int(row["period"]))
        if key not in metrics:
            continue
        metrics[key]["rank_ic_mean"] = (
            float(row["rank_ic_mean"]) if pd.notna(row["rank_ic_mean"]) else None
        )
        metrics[key]["rank_ic_count"] = (
            int(row["rank_ic_count"]) if pd.notna(row["rank_ic_count"]) else 0
        )
    for _, row in monthly_df.iterrows():
        key = str(int(row["period"]))
        if key not in metrics:
            continue
        metrics[key]["long_short_return_mean"] = (
            float(row["long_short_return_mean"])
            if pd.notna(row["long_short_return_mean"])
            else None
        )
        metrics[key]["long_short_return_win_rate"] = (
            float(row["long_short_return_win_rate"])
            if pd.notna(row["long_short_return_win_rate"])
            else None
        )
        metrics[key]["long_short_return_count"] = (
            int(row["long_short_return_count"])
            if pd.notna(row["long_short_return_count"])
            else 0
        )
    if factor_df.empty:
        warnings.append(f"no factor_analysis_results rows found for {factor_name}")
    if monthly_df.empty:
        warnings.append(f"no monthly_pool_results rows found for {factor_name}")
    return metrics, warnings


def _results_duckdb_path(runtime: dict) -> str:
    backend = str(runtime.get("results_db_backend", "auto") or "auto").lower()
    if backend == "sqlite":
        raise ValueError("SQLite results backend is no longer supported; configure results_duckdb_path")
    if backend not in {"auto", "duckdb"}:
        raise ValueError("results_db_backend must be 'duckdb' or 'auto'")
    path = runtime.get("results_duckdb_path") or binding_contract().get("output_db")
    if not path:
        raise ValueError("results_duckdb_path is required for post-rerank")
    return str(path)


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


def _query_factor_values(
    analysis_context: AnalysisContext,
    factor_names: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    if not factor_names:
        return pd.DataFrame(columns=["date", "code", "factor_name", "factor_value"])
    runtime = analysis_context.runtime_config
    values_table = _quote_identifier(
        runtime.get("factor_values_table", "factor_values")
    )
    params: list[Any] = list(factor_names) + [start, end]
    sql = f"""
        SELECT CAST(date AS DATE) AS date,
               CAST(stock_code AS VARCHAR) AS code,
               CAST(factor_name AS VARCHAR) AS factor_name,
               CAST(factor_value AS DOUBLE) AS factor_value
        FROM {values_table}
        WHERE factor_name IN ({_placeholders(len(factor_names))})
          AND CAST(date AS DATE) >= CAST(? AS DATE)
          AND CAST(date AS DATE) <= CAST(? AS DATE)
    """
    df = _fetch_duckdb_df(
        _results_duckdb_path(runtime),
        sql,
        params,
        table=runtime.get("factor_values_table", "factor_values"),
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["date", "code", "factor_name", "factor_value"])


def _query_monthly_long_short(
    analysis_context: AnalysisContext,
    factor_names: list[str],
    period: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    if not factor_names:
        return pd.DataFrame(columns=["factor_name", "year_month", "long_short_return"])
    runtime = analysis_context.runtime_config
    monthly_table = _quote_identifier(
        runtime.get("results_monthly_output_table", "monthly_pool_results")
    )
    month_start, month_end = _validation_month_window(start, end)
    params: list[Any] = list(factor_names) + [month_start, month_end, int(period)]
    sql = f"""
        SELECT CAST(factor_name AS VARCHAR) AS factor_name,
               SUBSTR(CAST(year_month AS VARCHAR), 1, 7) AS year_month,
               CAST(long_short_return AS DOUBLE) AS long_short_return
        FROM {monthly_table}
        WHERE factor_name IN ({_placeholders(len(factor_names))})
          AND SUBSTR(CAST(year_month AS VARCHAR), 1, 7) >= ?
          AND SUBSTR(CAST(year_month AS VARCHAR), 1, 7) <= ?
          AND CAST(period_days AS INTEGER) = ?
    """
    df = _fetch_duckdb_df(
        _results_duckdb_path(runtime),
        sql,
        params,
        table=runtime.get("results_monthly_output_table", "monthly_pool_results"),
        allow_labels=True,
    )
    return df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["factor_name", "year_month", "long_short_return"]
    )


def _compute_factor_correlation_metrics(
    values: pd.DataFrame, candidate_name: str, reference_factors: list[str]
) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if values.empty:
        return {
            "available": False,
            "passed": False,
            "reason": "missing_factor_values",
        }, ["factor_values query returned no rows"]
    present = set(values["factor_name"].astype(str))
    missing = sorted(set(reference_factors) - present)
    if candidate_name not in present:
        warnings.append(f"candidate factor_values missing for {candidate_name}")
    if missing:
        warnings.append(f"reference factor_values missing for: {missing}")
    if candidate_name not in present or missing:
        return {
            "available": False,
            "missing_reference_factors": missing,
            "passed": False,
            "reason": "missing_factor_values",
        }, warnings

    pivot = values.pivot_table(
        index=["date", "code"],
        columns="factor_name",
        values="factor_value",
        aggfunc="last",
    )
    rows = []
    for ref in reference_factors:
        pair = pivot[[candidate_name, ref]].replace([np.inf, -np.inf], np.nan).dropna()
        if pair.empty:
            warnings.append(
                f"no overlapping factor_values for {candidate_name} and {ref}"
            )
            continue
        by_date = []
        for date, group in pair.groupby(level=0):
            if (
                len(group) < 3
                or group[candidate_name].nunique(dropna=True) < 2
                or group[ref].nunique(dropna=True) < 2
            ):
                continue
            corr = group[candidate_name].corr(group[ref], method="spearman")
            if np.isfinite(corr):
                by_date.append(float(corr))
        if by_date:
            arr = np.asarray(by_date, dtype=float)
            rows.append(
                {
                    "reference_factor": ref,
                    "mean_corr": float(arr.mean()),
                    "mean_abs_corr": float(np.abs(arr).mean()),
                    "max_abs_corr": float(np.abs(arr).max()),
                    "n_dates": int(len(arr)),
                }
            )
        else:
            warnings.append(f"no valid daily cross-sectional correlations for {ref}")

    if not rows:
        return {
            "available": False,
            "passed": False,
            "reason": "no_valid_correlations",
        }, warnings
    max_abs = max(row["max_abs_corr"] for row in rows)
    mean_abs = float(np.mean([row["mean_abs_corr"] for row in rows]))
    return {
        "available": True,
        "reference_count": int(len(reference_factors)),
        "max_abs_factor_corr": float(max_abs),
        "mean_abs_factor_corr": mean_abs,
        "by_reference": rows,
    }, warnings


def _sharpe(series: pd.Series) -> float:
    clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(clean.mean() / std)


def _compute_marginal_sharpe_metrics(
    returns: pd.DataFrame,
    candidate_name: str,
    reference_factors: list[str],
) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if returns.empty:
        return {"available": False, "reason": "missing_monthly_returns"}, [
            "monthly long_short_return query returned no rows"
        ]
    pivot = returns.pivot_table(
        index="year_month",
        columns="factor_name",
        values="long_short_return",
        aggfunc="mean",
    )
    missing = sorted(set(reference_factors) - set(pivot.columns))
    if candidate_name not in pivot.columns:
        warnings.append(
            f"candidate monthly long_short_return missing for {candidate_name}"
        )
    if missing:
        warnings.append(f"reference monthly long_short_return missing for: {missing}")
    if candidate_name not in pivot.columns or missing:
        return {
            "available": False,
            "missing_reference_factors": missing,
            "reason": "missing_monthly_returns",
        }, warnings

    reference_frame = pivot[reference_factors].dropna(how="all")
    candidate = pivot[candidate_name]
    baseline = reference_frame.mean(axis=1, skipna=True)
    aligned = pd.DataFrame({"baseline": baseline, "candidate": candidate}).dropna()
    if aligned.empty:
        return {"available": False, "reason": "no_overlapping_monthly_returns"}, [
            "no overlapping monthly returns for marginal Sharpe"
        ]

    reference_count = len(reference_factors)
    combined = (aligned["baseline"] * reference_count + aligned["candidate"]) / float(
        reference_count + 1
    )
    baseline_sharpe = _sharpe(aligned["baseline"])
    combined_sharpe = _sharpe(combined)
    return {
        "available": True,
        "baseline_sharpe": float(baseline_sharpe),
        "combined_sharpe": float(combined_sharpe),
        "marginal_sharpe_delta": float(combined_sharpe - baseline_sharpe),
        "n_months": int(len(aligned)),
    }, warnings


def _compute_turnover_metrics(
    values: pd.DataFrame, candidate_name: str, quantiles: int = 5
) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    candidate = (
        values[values["factor_name"] == candidate_name].copy()
        if not values.empty
        else pd.DataFrame()
    )
    if candidate.empty:
        return {
            "available": False,
            "reason": "missing_candidate_factor_values",
            "top_turnover_mean": None,
            "bottom_turnover_mean": None,
            "long_short_turnover_mean": None,
            "n_transitions": 0,
        }, [f"candidate factor_values missing for turnover: {candidate_name}"]

    quantiles = max(2, int(quantiles))
    top_sets = []
    bottom_sets = []
    for date, group in candidate.groupby("date", sort=True):
        work = group.dropna(subset=["factor_value"])
        if len(work) < quantiles or work["factor_value"].nunique(dropna=True) < 2:
            continue
        ranks = work["factor_value"].rank(method="first", pct=True)
        top_sets.append(
            (date, set(work.loc[ranks > 1.0 - 1.0 / quantiles, "code"].astype(str)))
        )
        bottom_sets.append(
            (date, set(work.loc[ranks <= 1.0 / quantiles, "code"].astype(str)))
        )

    def average_turnover(
        sets: list[tuple[pd.Timestamp, set[str]]],
    ) -> tuple[float | None, int]:
        values_out = []
        for (_, previous), (_, current) in zip(sets, sets[1:]):
            if not previous:
                continue
            values_out.append(1.0 - len(previous & current) / float(len(previous)))
        if not values_out:
            return None, 0
        return float(np.mean(values_out)), len(values_out)

    top_turnover, top_n = average_turnover(top_sets)
    bottom_turnover, bottom_n = average_turnover(bottom_sets)
    values_for_mean = [
        value for value in (top_turnover, bottom_turnover) if value is not None
    ]
    return {
        "available": bool(values_for_mean),
        "quantiles": int(quantiles),
        "top_turnover_mean": top_turnover,
        "bottom_turnover_mean": bottom_turnover,
        "long_short_turnover_mean": float(np.mean(values_for_mean))
        if values_for_mean
        else None,
        "n_transitions": int(max(top_n, bottom_n)),
    }, warnings


def _validation_panel_for_lightweight(
    valid_panel: pd.DataFrame, rerank_config: dict
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    cache_key = (
        id(valid_panel),
        int(len(valid_panel)),
        str(rerank_config.get("validation_start", "")),
        str(rerank_config.get("validation_end", "")),
    )
    panel_cache = None
    if not bool(rerank_config.get("_disable_panel_cache", False)):
        panel_cache = rerank_config.setdefault("_validation_panel_cache", {})
        cached = panel_cache.get(cache_key)
        if cached is not None:
            panel, cached_warnings = cached
            return panel, list(cached_warnings)

    if valid_panel.empty or "date" not in valid_panel.columns:
        result = valid_panel.copy()
        if panel_cache is not None:
            panel_cache[cache_key] = (result, tuple(warnings))
        return result, warnings

    work = valid_panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date"])
    validation_start = pd.Timestamp(
        str(rerank_config.get("validation_start", work["date"].min()))
    ).normalize()
    validation_end = pd.Timestamp(
        str(rerank_config.get("validation_end", work["date"].max()))
    ).normalize()
    filtered = work[
        (work["date"] >= validation_start) & (work["date"] <= validation_end)
    ]
    if filtered.empty and not work.empty:
        warnings.append(
            "validation window has no rows in valid_panel; using provided valid_panel"
        )
        filtered = work
    result = filtered.sort_values(
        ["date", "code"] if "code" in filtered.columns else ["date"]
    ).reset_index(drop=True)
    if panel_cache is not None:
        panel_cache[cache_key] = (result, tuple(warnings))
    return result, warnings


def _period_label_from_panel(panel: pd.DataFrame, period: int) -> tuple[pd.Series, str]:
    exact_names = [
        f"label__ret_1_{int(period)}",
        f"label__ret_{int(period)}",
        f"label__{int(period)}D",
    ]
    for name in exact_names:
        if name in panel.columns:
            return pd.Series(panel[name], index=panel.index, dtype=float), name

    if {"date", "code", "daily_close"}.issubset(panel.columns):
        work = panel[["date", "code", "daily_close"]].copy()
        work["_original_index"] = np.arange(len(work), dtype=np.int64)
        work = work.sort_values(["code", "date"])
        grouped_close = work.groupby("code", sort=False)["daily_close"]
        future = grouped_close.shift(-int(period))
        work["_label"] = future / work["daily_close"] - 1.0
        label = work.sort_values("_original_index")["_label"].reset_index(drop=True)
        if int(np.isfinite(np.asarray(label, dtype=float)).sum()) >= 3:
            return pd.Series(
                label, index=panel.index, dtype=float
            ), f"computed_daily_close_ret_{int(period)}"

    if "label" in panel.columns:
        return pd.Series(panel["label"], index=panel.index, dtype=float), "label"
    return pd.Series(np.nan, index=panel.index, dtype=float), "missing"


def _period_label_frame(
    panel: pd.DataFrame, periods: list[int]
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    labels: dict[str, pd.Series] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []
    for period in periods:
        label, source = _period_label_from_panel(panel, period)
        key = str(int(period))
        labels[key] = label.reset_index(drop=True)
        sources[key] = source
        if source == "missing":
            warnings.append(f"no label data available for period {period}")
        elif source == "label":
            warnings.append(f"using generic label column for period {period}")
    return pd.DataFrame(labels), sources, warnings


def _lightweight_scoring_context(
    panel: pd.DataFrame, periods: list[int], rerank_config: dict
) -> tuple[pd.DataFrame, dict[str, str], list[str], ScoringContext]:
    cache_key = (id(panel), int(len(panel)), tuple(int(period) for period in periods))
    context_cache = None
    if not bool(rerank_config.get("_disable_scoring_context_cache", False)):
        context_cache = rerank_config.setdefault(
            "_lightweight_scoring_context_cache", {}
        )
        cached = context_cache.get(cache_key)
        if cached is not None:
            label_frame, label_sources, label_warnings, scoring_context = cached
            return (
                label_frame,
                dict(label_sources),
                list(label_warnings),
                scoring_context,
            )

    label_frame, label_sources, label_warnings = _period_label_frame(panel, periods)
    scoring_context = ScoringContext(
        panel["date"], label_frame, anchors=anchor_matrix_from_panel(panel)
    )
    if context_cache is not None:
        context_cache[cache_key] = (
            label_frame,
            dict(label_sources),
            tuple(label_warnings),
            scoring_context,
        )
    return label_frame, label_sources, label_warnings, scoring_context


def _residualization_context_for_panel(
    panel: pd.DataFrame, anchors: np.ndarray, rerank_config: dict
) -> ResidualizationContext | None:
    if bool(rerank_config.get("_disable_residualization_context_cache", False)):
        return None
    cache_key = (id(panel), int(len(panel)), tuple(int(item) for item in anchors.shape))
    context_cache = rerank_config.setdefault("_residualization_context_cache", {})
    context = context_cache.get(cache_key)
    if context is None:
        context = ResidualizationContext(panel["date"], anchors)
        context_cache[cache_key] = context
    return context


def _long_short_by_date(
    pred: pd.Series,
    label: pd.Series,
    dates: pd.Series,
    *,
    quantiles: int,
    min_group_size: int,
) -> pd.Series:
    pred_values = np.asarray(pred, dtype=float)
    label_values = np.asarray(label, dtype=float)
    date_values = pd.to_datetime(dates, errors="coerce").to_numpy()
    values: dict[pd.Timestamp, float] = {}
    grouped_positions = pd.Series(np.arange(len(pred_values))).groupby(
        date_values, sort=False
    )
    for date, positions_series in grouped_positions:
        if pd.isna(date):
            continue
        positions = np.asarray(positions_series, dtype=np.int64)
        mask = np.isfinite(pred_values[positions]) & np.isfinite(
            label_values[positions]
        )
        if int(mask.sum()) < 1:
            continue
        value = _long_short_from_arrays(
            pred_values[positions][mask],
            label_values[positions][mask],
            quantiles=quantiles,
            min_group_size=min_group_size,
        )
        if value is not None:
            values[pd.Timestamp(date)] = value
    return pd.Series(values, dtype=float)


def _top_quantile_excess_from_arrays(
    pred_values: np.ndarray,
    label_values: np.ndarray,
    *,
    quantiles: int,
    min_group_size: int,
) -> float | None:
    """Long-only excess return of the top prediction quantile for one date.

    ``mean(label | top quantile) - mean(label | all stocks)``: the realizable
    economics of a long-only weekly rotation in a no-short small-cap pool,
    where the long-short spread's bottom leg cannot actually be traded.
    """
    if pred_values.size < quantiles * min_group_size or np.unique(pred_values).size < 2:
        return None
    order = np.argsort(pred_values, kind="mergesort")
    buckets = _bucket_labels(pred_values.size, quantiles)
    top = label_values[order][buckets == int(buckets.max())]
    if top.size < min_group_size:
        return None
    value = float(top.mean() - label_values.mean())
    return value if np.isfinite(value) else None


def _top_quantile_excess_by_date(
    pred: pd.Series,
    label: pd.Series,
    dates: pd.Series,
    *,
    quantiles: int,
    min_group_size: int,
) -> pd.Series:
    pred_values = np.asarray(pred, dtype=float)
    label_values = np.asarray(label, dtype=float)
    date_values = pd.to_datetime(dates, errors="coerce").to_numpy()
    values: dict[pd.Timestamp, float] = {}
    grouped_positions = pd.Series(np.arange(len(pred_values))).groupby(
        date_values, sort=False
    )
    for date, positions_series in grouped_positions:
        if pd.isna(date):
            continue
        positions = np.asarray(positions_series, dtype=np.int64)
        mask = np.isfinite(pred_values[positions]) & np.isfinite(
            label_values[positions]
        )
        if int(mask.sum()) < 1:
            continue
        value = _top_quantile_excess_from_arrays(
            pred_values[positions][mask],
            label_values[positions][mask],
            quantiles=quantiles,
            min_group_size=min_group_size,
        )
        if value is not None:
            values[pd.Timestamp(date)] = value
    return pd.Series(values, dtype=float)


def _monthly_long_short_metrics(
    daily_returns: pd.Series,
) -> tuple[float | None, float | None, int]:
    clean = (
        pd.Series(daily_returns, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if clean.empty:
        return None, None, 0
    monthly = clean.groupby(clean.index.to_period("M")).mean()
    monthly = monthly.replace([np.inf, -np.inf], np.nan).dropna()
    if monthly.empty:
        return None, None, 0
    return float(monthly.mean()), float((monthly > 0).mean()), int(len(monthly))


def _factor_values_from_prediction(
    panel: pd.DataFrame, pred: pd.Series, factor_name: str
) -> pd.DataFrame:
    if not {"date", "code"}.issubset(panel.columns):
        return pd.DataFrame(columns=["date", "code", "factor_name", "factor_value"])
    return (
        pd.DataFrame(
            {
                "date": pd.to_datetime(panel["date"], errors="coerce").dt.normalize(),
                "code": panel["code"].astype(str).to_numpy(),
                "factor_name": factor_name,
                "factor_value": np.asarray(pred, dtype=float),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["date", "code", "factor_value"])
    )


def _evaluate_candidate_lightweight(
    candidate: SearchCandidate,
    valid_panel: pd.DataFrame,
    rerank_config: dict,
) -> Dict[str, Any]:
    started = time.perf_counter()
    warnings: list[str] = []
    try:
        panel, panel_warnings = _validation_panel_for_lightweight(
            valid_panel, rerank_config
        )
        warnings.extend(panel_warnings)
        if panel.empty:
            raise ValueError("valid_panel is empty")

        eval_started = time.perf_counter()
        pred = _evaluate_expression_for_rerank(
            candidate.expression, panel, rerank_config
        )
        pred = pd.Series(pred, index=panel.index, dtype=float).replace(
            [np.inf, -np.inf], np.nan
        )
        evaluate_duration = time.perf_counter() - eval_started

        periods = [int(period) for period in rerank_config.get("periods", [1, 5, 10])]
        label_frame, label_sources, label_warnings, scoring_context = (
            _lightweight_scoring_context(panel, periods, rerank_config)
        )
        warnings.extend(label_warnings)
        prepared = prepare_score(
            pred, scoring_context, {"objective": {"return_proxy": {"enabled": False}}}
        )
        rank_ic_by_period = {
            label_score.name: label_score.rank_ic for label_score in prepared.labels
        }

        quantiles = max(
            2,
            int(
                rerank_config.get(
                    "quantiles", rerank_config.get("turnover_quantiles", 5)
                )
            ),
        )
        min_group_size = max(1, int(rerank_config.get("min_group_size", 1)))
        tstat_method = str(rerank_config.get("tstat_method", "newey_west"))
        period_metrics: Dict[str, Dict[str, float | int | None]] = {}
        for period in periods:
            key = str(int(period))
            rank_ic_values = rank_ic_by_period.get(key, np.asarray([], dtype=float))
            rank_ic_mean = float(rank_ic_values.mean()) if rank_ic_values.size else None
            rank_ic_std = (
                float(rank_ic_values.std(ddof=1)) if rank_ic_values.size > 1 else None
            )
            # Overlapping multi-day forward-return labels autocorrelate the daily
            # IC, so the Newey-West HAC lag tracks the holding period.
            rank_ic_t = _ic_tstat(
                rank_ic_values,
                method=tstat_method,
                lag=max(1, int(period) - 1),
            )
            daily_returns = _long_short_by_date(
                pred,
                label_frame[key],
                panel["date"],
                quantiles=quantiles,
                min_group_size=min_group_size,
            )
            return_mean, win_rate, return_count = _monthly_long_short_metrics(
                daily_returns
            )
            # Long-only realizable economics: top-quantile excess over the
            # equal-weight pool. The t-stat reuses the Newey-West machinery on
            # the daily excess series (overlapping labels autocorrelate it just
            # like the daily IC).
            daily_excess = _top_quantile_excess_by_date(
                pred,
                label_frame[key],
                panel["date"],
                quantiles=quantiles,
                min_group_size=min_group_size,
            )
            excess_values = np.asarray(daily_excess, dtype=float)
            excess_finite = excess_values[np.isfinite(excess_values)]
            excess_mean = float(excess_finite.mean()) if excess_finite.size else None
            excess_t = _ic_tstat(
                excess_values,
                method=tstat_method,
                lag=max(1, int(period) - 1),
            )
            period_metrics[key] = {
                "rank_ic_mean": rank_ic_mean,
                "rank_ic_std": rank_ic_std,
                "rank_ic_t": rank_ic_t,
                "rank_ic_count": int(rank_ic_values.size),
                "top_quantile_excess_return_mean": excess_mean,
                "top_quantile_excess_return_t": excess_t,
                "top_quantile_excess_return_count": int(excess_finite.size),
                "long_short_return_mean": return_mean,
                "long_short_return_win_rate": win_rate,
                "long_short_return_count": int(return_count),
                "label_source": label_sources.get(key, "unknown"),
            }

        primary_passed, best_period, sort_values = _evaluate_gate(
            period_metrics, rerank_config
        )
        factor_values = _factor_values_from_prediction(
            panel, pred, "__lightweight_candidate__"
        )
        turnover_metrics, turnover_warnings = _compute_turnover_metrics(
            factor_values,
            "__lightweight_candidate__",
            int(rerank_config.get("turnover_quantiles", quantiles)),
        )
        warnings.extend(turnover_warnings)
        portfolio_config = rerank_config.get("portfolio_gate", {})
        portfolio_gate = (
            {"enabled": False, "passed": True, "reason": "disabled"}
            if not portfolio_config or not portfolio_config.get("enabled", False)
            else {
                "enabled": True,
                "passed": True,
                "reason": "not_evaluated_by_lightweight_rerank",
            }
        )
        return {
            "ok": True,
            "duration_sec": float(time.perf_counter() - started),
            "evaluate_duration_sec": float(evaluate_duration),
            "period_metrics": period_metrics,
            "primary_passed": bool(primary_passed),
            "best_period": best_period,
            "sort_values": [float(value) for value in sort_values],
            "turnover_metrics": turnover_metrics,
            "portfolio_gate": portfolio_gate,
            "warnings": warnings,
            "fail_reason": "",
        }
    except Exception as exc:
        # Surface the failure cause instead of silently returning empty metrics:
        # the 2024 validation re-scoring depends on this path, and a swallowed
        # exception here is exactly what made ``selected_valid_rank_ic`` vanish
        # (which fails the downstream valid_floor gate open).
        return {
            "ok": False,
            "duration_sec": float(time.perf_counter() - started),
            "evaluate_duration_sec": 0.0,
            "period_metrics": {},
            "primary_passed": False,
            "best_period": None,
            "sort_values": [-1.0e18, -1.0e18, -1.0e18],
            "turnover_metrics": {
                "available": False,
                "reason": "lightweight_evaluation_failed",
                "top_turnover_mean": None,
                "bottom_turnover_mean": None,
                "long_short_turnover_mean": None,
                "n_transitions": 0,
            },
            "portfolio_gate": {"enabled": False, "passed": True, "reason": "disabled"},
            "warnings": [*warnings, f"lightweight evaluation failed: {exc}"],
            "fail_reason": f"{type(exc).__name__}: {exc}",
        }


def _evaluate_portfolio_gate(
    analysis_context: AnalysisContext,
    candidate_name: str,
    best_period: str | None,
    rerank_config: dict,
) -> tuple[dict, dict, list[str]]:
    portfolio_config = rerank_config.get("portfolio_gate", {})
    validation_start = str(rerank_config.get("validation_start", "2024-01-01"))
    validation_end = str(rerank_config.get("validation_end", "2024-12-31"))
    turnover_quantiles = int(rerank_config.get("turnover_quantiles", 5))
    warnings: list[str] = []

    factor_values = pd.DataFrame()
    try:
        names = [candidate_name] + list(portfolio_config.get("reference_factors", []))
        factor_values = _query_factor_values(
            analysis_context, names, validation_start, validation_end
        )
    except Exception as exc:
        warnings.append(f"failed to query factor_values: {exc}")

    turnover_metrics, turnover_warnings = _compute_turnover_metrics(
        factor_values, candidate_name, turnover_quantiles
    )
    warnings.extend(turnover_warnings)

    if not portfolio_config or not portfolio_config.get("enabled", False):
        return (
            {"enabled": False, "passed": True, "reason": "disabled"},
            turnover_metrics,
            warnings,
        )

    reference_factors = [
        str(item) for item in portfolio_config.get("reference_factors", [])
    ]
    corr_metrics, corr_warnings = _compute_factor_correlation_metrics(
        factor_values, candidate_name, reference_factors
    )
    warnings.extend(corr_warnings)

    period = (
        int(best_period)
        if best_period is not None
        else int(rerank_config.get("periods", [1])[0])
    )
    try:
        returns = _query_monthly_long_short(
            analysis_context,
            [candidate_name] + reference_factors,
            period,
            validation_start,
            validation_end,
        )
        sharpe_metrics, sharpe_warnings = _compute_marginal_sharpe_metrics(
            returns, candidate_name, reference_factors
        )
    except Exception as exc:
        sharpe_metrics = {"available": False, "reason": "monthly_query_failed"}
        sharpe_warnings = [f"failed to query monthly long_short_return: {exc}"]
    warnings.extend(sharpe_warnings)

    max_corr_threshold = float(portfolio_config.get("max_abs_factor_corr", 0.7))
    min_delta = float(portfolio_config.get("min_marginal_sharpe_delta", 0.0))
    corr_passed = bool(
        corr_metrics.get("available")
        and float(corr_metrics.get("max_abs_factor_corr", 1.0e18)) <= max_corr_threshold
    )
    sharpe_passed = bool(
        sharpe_metrics.get("available")
        and float(sharpe_metrics.get("marginal_sharpe_delta", -1.0e18)) >= min_delta
    )
    passed = bool(corr_passed and sharpe_passed)
    reason = (
        "passed"
        if passed
        else (
            "correlation_gate_failed"
            if not corr_passed
            else "marginal_sharpe_gate_failed"
        )
    )
    return (
        {
            "enabled": True,
            "mode": str(portfolio_config.get("mode", "hard")),
            "passed": passed,
            "reason": reason,
            "period": int(period),
            "thresholds": {
                "max_abs_factor_corr": max_corr_threshold,
                "min_marginal_sharpe_delta": min_delta,
            },
            "correlation": corr_metrics,
            "marginal_sharpe": sharpe_metrics,
        },
        turnover_metrics,
        warnings,
    )


def _evaluate_reference_correlation_gate(
    candidate: SearchCandidate,
    oos_panel: pd.DataFrame,
    analysis_context: AnalysisContext,
    rerank_config: dict,
) -> dict:
    """Correlation gate for a freshly mined candidate against reference factors.

    Unlike :func:`_evaluate_portfolio_gate` -- which assumes the candidate
    already exists in the analysis ``factor_values`` table -- this variant scores
    the candidate from its in-memory out-of-sample prediction (a newly mined
    factor is not persisted yet) and pulls only the *reference* factors (e.g.
    size / market-cap proxies) from the DB over the same OOS window. The gate
    blocks a candidate whose mean daily cross-sectional |Spearman corr| with any
    reference factor exceeds ``max_abs_factor_corr``: such a candidate is
    effectively a clone of an already-validated factor and adds no independent
    breadth.

    Failure to evaluate (no DB overlap, query error) is treated conservatively:
    in ``hard`` mode the candidate is blocked with a distinct reason so an
    infrastructure problem surfaces loudly instead of silently disabling the
    screen.
    """
    portfolio_config = rerank_config.get("portfolio_gate", {}) or {}
    mode = str(portfolio_config.get("mode", "hard"))
    reference_factors = [
        str(item).strip()
        for item in portfolio_config.get("reference_factors", [])
        if str(item).strip()
    ]
    max_corr_threshold = float(portfolio_config.get("max_abs_factor_corr", 0.7))
    base: dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "metric_source": "oos_prediction",
        "reference_factors": reference_factors,
        "thresholds": {"max_abs_factor_corr": max_corr_threshold},
    }
    if not reference_factors:
        return {**base, "passed": True, "reason": "no_reference_factors"}

    candidate_name = "__candidate__"
    warnings: list[str] = []
    try:
        panel, panel_warnings = _validation_panel_for_lightweight(
            oos_panel, rerank_config
        )
        warnings.extend(panel_warnings)
        if panel.empty:
            raise ValueError("oos panel is empty for correlation gate")
        pred = _evaluate_expression_for_rerank(
            candidate.expression, panel, rerank_config
        )
        pred = pd.Series(pred, index=panel.index, dtype=float).replace(
            [np.inf, -np.inf], np.nan
        )
        candidate_values = _factor_values_from_prediction(panel, pred, candidate_name)
        start = str(rerank_config.get("validation_start", panel["date"].min()))
        end = str(rerank_config.get("validation_end", panel["date"].max()))
        reference_values = _query_factor_values(
            analysis_context, reference_factors, start, end
        )
        frames = [
            frame for frame in (candidate_values, reference_values) if not frame.empty
        ]
        combined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else candidate_values.iloc[0:0]
        )
        corr_metrics, corr_warnings = _compute_factor_correlation_metrics(
            combined, candidate_name, reference_factors
        )
        warnings.extend(corr_warnings)
    except Exception as exc:  # noqa: BLE001 - a gate error must not crash rerank
        # An error computing correlation is not evidence of a clone. Mirror the
        # "missing reference does not block" contract: pass but flag loudly.
        return {
            **base,
            "passed": True,
            "reason": "correlation_gate_error",
            "error": str(exc),
            "warnings": [*warnings, f"size correlation gate errored: {exc}"],
        }

    if not corr_metrics.get("available"):
        # No DB overlap between the candidate's OOS prediction and the reference
        # factors -> cannot measure correlation. Do not block (a DB gap must not
        # silently kill every run); surface the reason for operator follow-up.
        return {
            **base,
            "passed": True,
            "reason": "correlation_gate_unevaluable",
            "detail": str(corr_metrics.get("reason", "no_valid_correlations")),
            "correlation": corr_metrics,
            "warnings": [
                *warnings,
                "size correlation gate unevaluable (no DB overlap)",
            ],
        }

    observed = float(corr_metrics.get("mean_abs_factor_corr", 1.0e18))
    corr_passed = observed <= max_corr_threshold
    passed = corr_passed if mode == "hard" else True
    return {
        **base,
        "passed": bool(passed),
        "reason": "passed" if corr_passed else "correlation_gate_failed",
        "observed_mean_abs_corr": observed,
        "correlation": corr_metrics,
        "warnings": warnings,
    }


def _evaluate_gate(
    period_metrics: Dict[str, Dict[str, float | int | None]],
    rerank_config: dict,
) -> tuple[bool, str | None, list[float]]:
    """Selection-window dual gate.

    Components (all at the locked ``gate_period`` when configured):
      * ``rank_ic_mean >= min_rank_ic_mean`` -- cross-sectional ranking skill;
      * ``rank_ic_t >= min_valid_rank_ic_t`` -- the former evaluator-side
        valid-floor t requirement, folded into selection (0 disables);
      * ``top_quantile_excess_return_mean > min_top_quantile_excess_return`` --
        long-only realizable economics (replaces the long-short mean, whose
        short leg is untradeable in the small-cap pool);
      * ``top_quantile_excess_return_t >= min_top_quantile_excess_t`` -- the
        excess must be statistically distinguishable from zero (replaces the
        12-monthly-sample win-rate cut, which had no statistical power).
    Long-short mean / win rate remain recorded as observe-only fields.
    """
    min_ic = float(rerank_config.get("min_rank_ic_mean", 0.08))
    min_ic_t = float(rerank_config.get("min_valid_rank_ic_t", 0.0))
    min_excess = float(rerank_config.get("min_top_quantile_excess_return", 0.0))
    min_excess_t = float(rerank_config.get("min_top_quantile_excess_t", 0.0))
    # A fixed ``gate_period`` locks the pass/fail decision and the selected
    # period to the strategy's true holding horizon (e.g. 10 days). Other
    # periods are still scored for selection->confirm decay diagnostics, but a
    # candidate can never clear the gate on a shorter, off-strategy horizon.
    gate_period = rerank_config.get("gate_period")
    gate_key = str(int(gate_period)) if gate_period is not None else None

    passing: list[tuple[str, float, float, float]] = []
    for period, metrics in period_metrics.items():
        ic = metrics.get("rank_ic_mean")
        ic_t = metrics.get("rank_ic_t")
        excess = metrics.get("top_quantile_excess_return_mean")
        excess_t = metrics.get("top_quantile_excess_return_t")
        passed = (
            ic is not None
            and excess is not None
            and float(ic) >= min_ic
            and (min_ic_t <= 0.0 or (ic_t is not None and float(ic_t) >= min_ic_t))
            and float(excess) > min_excess
            and (
                min_excess_t <= 0.0
                or (excess_t is not None and float(excess_t) >= min_excess_t)
            )
        )
        metrics["gate_passed"] = bool(passed)
        if passed:
            passing.append(
                (
                    period,
                    float(ic),
                    float(excess),
                    float(excess_t) if excess_t is not None else 0.0,
                )
            )

    if gate_key is not None:
        locked = period_metrics.get(gate_key)
        if locked is None or not locked.get("gate_passed", False):
            return False, None, [-1.0e18, -1.0e18, -1.0e18]
        locked_excess_t = locked.get("top_quantile_excess_return_t")
        return (
            True,
            gate_key,
            [
                float(locked["rank_ic_mean"]),
                float(locked["top_quantile_excess_return_mean"]),
                float(locked_excess_t) if locked_excess_t is not None else 0.0,
            ],
        )

    if not passing:
        return False, None, [-1.0e18, -1.0e18, -1.0e18]
    best_period, ic, excess, excess_t = max(
        passing, key=lambda item: (item[1], item[2], item[3], -int(item[0]))
    )
    return True, best_period, [ic, excess, excess_t]


def _candidate_payload(candidate: SearchCandidate) -> Dict[str, Any]:
    return {
        "expression": candidate.expression.to_string(),
        "expression_ast": serialize_expression(candidate.expression),
        "expression_hash": expression_hash(candidate.expression),
        "score": float(candidate.score),
        "complexity": int(candidate.complexity),
        "train_metrics": candidate.train_metrics,
        "valid_metrics": candidate.valid_metrics,
    }


def _feature_family(ref: str) -> str:
    value = str(ref or "").strip().lower()
    if not value:
        return "unknown"
    if any(marker in value for marker in ("imbalance", "concentration", "gap")):
        return "intraday_microstructure"
    if (
        "market_cap" in value
        or value in {"size", "cube_of_size"}
        or value.endswith("_size")
    ):
        return "size.market_cap"
    if "volume" in value or "turnover" in value:
        return "liquidity.volume"
    if "money" in value or "amount" in value:
        return "liquidity.money"
    if value in {"open", "high", "low", "close", "vwap"} or (
        value.startswith("daily_")
        and any(
            value.endswith(f"_{name}")
            for name in ("open", "high", "low", "close", "vwap")
        )
    ):
        return "price"
    if value.startswith("intraday_"):
        return "intraday"
    if value.startswith("fd_"):
        return "factor_data"
    return "other"


def _window_values(expr) -> list[int]:
    values: list[int] = []
    if getattr(expr, "kind", "") == "window":
        try:
            values.append(int(float(expr.value)))
        except Exception:
            return values
    for child in getattr(expr, "children", []) or []:
        values.extend(_window_values(child))
    return values


def _complexity_band(complexity: int) -> str:
    if complexity <= 4:
        return "c01_04"
    if complexity <= 8:
        return "c05_08"
    if complexity <= 16:
        return "c09_16"
    return "c17_plus"


def _window_band(expr) -> str:
    windows = _window_values(expr)
    if not windows:
        return "w_none"
    max_window = max(windows)
    if max_window <= 5:
        return "w_short"
    if max_window <= 20:
        return "w_medium"
    return "w_long"


def _transform_class(expr, feature_space: FeatureSpace) -> str:
    if getattr(expr, "kind", "") != "operator":
        return str(getattr(expr, "kind", "unknown"))
    op_name = str(expr.value)
    operator = feature_space.operators.get(op_name)
    category = operator.category if operator is not None else "operator"
    return f"{category}:{op_name}"


def _qd_bucket_key(expr, feature_space: FeatureSpace) -> str:
    refs = sorted(referenced_features(expr))
    families = sorted({_feature_family(ref) for ref in refs}) or ["unknown"]
    return "|".join(
        [
            "+".join(families[:3]),
            _transform_class(expr, feature_space),
            f"refs{len(refs)}",
            _complexity_band(expression_complexity(expr)),
            _window_band(expr),
        ]
    )


def _replace_feature(expr, old_ref: str, new_ref: str):
    if expr.kind == "feature" and str(expr.value) == old_ref:
        return type(expr)(kind="feature", value=new_ref)
    if not expr.children:
        return expr
    return type(expr)(
        kind=expr.kind,
        value=expr.value,
        children=tuple(
            _replace_feature(child, old_ref, new_ref) for child in expr.children
        ),
    )


def _window_node(window: int):
    from .expression import ExpressionNode

    return ExpressionNode(kind="window", value=int(window))


def _feature_node(name: str):
    from .expression import ExpressionNode

    return ExpressionNode(kind="feature", value=str(name))


def _op_node(name: str, *children):
    from .expression import ExpressionNode

    return ExpressionNode(kind="operator", value=str(name), children=children)


def _ensemble_ic_key(rerank_config: dict) -> str:
    gate_period = rerank_config.get("gate_period")
    if gate_period is not None:
        return str(int(gate_period))
    periods = [int(p) for p in rerank_config.get("periods", [1, 5, 10])]
    return str(periods[-1]) if periods else "10"


def _prediction_has_cross_sectional_dispersion(
    prediction: "pd.Series | None",
    dates: pd.Series,
    *,
    min_good_date_frac: float = 0.5,
) -> bool:
    """Return ``True`` when a member can be cross-sectionally z-scored.

    ``build_ensemble_expression`` wraps every member in ``cs_zscore`` and folds
    the terms together with ``add``. A member that is constant within each date
    -- e.g. ``sign(...)`` of a same-sign base expression -- has zero
    cross-sectional variance, so ``cs_zscore`` divides by a zero std and yields
    NaN on every date. A single such all-NaN term then NaNs the entire ``add``
    composite (``rank_ic_count == 0``), which is exactly how the IC-weighted
    blend silently collapses. Reject these members before they enter the blend.
    """
    if prediction is None:
        return False
    values = (
        pd.Series(prediction, dtype=float)
        .reset_index(drop=True)
        .replace([np.inf, -np.inf], np.nan)
    )
    date_values = pd.Series(np.asarray(dates)).reset_index(drop=True)
    total_dates = int(date_values.nunique())
    if total_dates <= 0:
        return False
    frame = pd.DataFrame({"date": date_values, "value": values}).dropna(
        subset=["value"]
    )
    if frame.empty:
        return False
    good_dates = int((frame.groupby("date")["value"].nunique() >= 2).sum())
    return (good_dates / total_dates) >= float(min_good_date_frac)


def _ensure_member_valid_metrics(
    row: Dict[str, Any],
    member_expr: Any,
    valid_panel: pd.DataFrame | None,
    valid_rerank_config: dict,
) -> Dict[str, Any]:
    """Selection-window (valid, e.g. 2024) period metrics for a candidate row.

    Member selection for the ensemble / elite-diversification passes must read
    SELECTION-window metrics, never the holdout: sourcing member admission,
    orientation, or weights from the OOS panel fits the composite on the very
    window it is gated on and inflates its holdout IC (the leak that admitted
    composites with near-zero in-sample signal). Rows that already carry
    ``valid_period_metrics`` (primary-gate passers score it in ``build_row``)
    are returned as-is; other rows are scored once on the valid panel with the
    same lightweight machinery and the result is backfilled onto the row so
    downstream consumers (e.g. the harness alternate-candidate path) see it.
    """
    existing = row.get("valid_period_metrics")
    if isinstance(existing, dict) and existing:
        return existing
    if valid_panel is None or valid_panel.empty:
        return {}
    candidate = SearchCandidate(
        expression=member_expr,
        train_metrics={},
        valid_metrics={},
        score=0.0,
        complexity=expression_complexity(member_expr),
    )
    lightweight = _evaluate_candidate_lightweight(
        candidate, valid_panel, valid_rerank_config
    )
    metrics = lightweight["period_metrics"] if lightweight.get("ok") else {}
    if metrics:
        row["valid_period_metrics"] = metrics
    return metrics if isinstance(metrics, dict) else {}


def _maybe_add_ensemble_candidate(
    *,
    candidate_rows: list[Dict[str, Any]],
    qd_archive: dict,
    oos_panel: pd.DataFrame,
    valid_panel: pd.DataFrame | None,
    valid_rerank_config: dict,
    rerank_config: dict,
    feature_space: FeatureSpace,
    build_row: Any,
    ensemble_rank: int,
) -> None:
    """Synthesize one IC-weighted composite candidate and let it compete.

    Builds a composite from the top decorrelated single-factor candidate rows,
    evaluates it through the same lightweight gate via ``build_row``, appends it
    to ``candidate_rows`` and (when it passes the primary gate) to the QD
    archive so the standard selection picks it up. No-op when ensembling is
    disabled or fewer than two usable members exist (single-best behavior).

    Holdout hygiene: every SUPERVISED member decision -- admission, the
    ``min_member_ic`` floor, sign orientation, IC weights, and the
    economic-alignment check -- reads the SELECTION window
    (``valid_period_metrics``, e.g. 2024), never the OOS holdout the composite
    is subsequently gated on. The holdout panel is only used for label-free
    structure: the cross-sectional dispersion guard and pairwise member
    decorrelation, which keep the composite evaluable/diversified on the gate
    window without observing its returns.
    """
    ensemble_cfg = rerank_config.get("ensemble", {}) or {}
    if not ensemble_cfg.get("enabled", False):
        return

    ic_key = _ensemble_ic_key(rerank_config)
    max_factors = int(ensemble_cfg.get("max_factors", 4))
    # Bound member scoring: only the strongest candidates can plausibly enter a
    # max_factors-member blend. ``consider_cap_factor`` widens the pool when
    # elite diversification enriches candidate_rows with variants; it defaults
    # to 4 so behaviour is unchanged unless explicitly raised.
    consider_cap = max(max_factors * int(ensemble_cfg.get("consider_cap_factor", 4)), 8)

    # Pre-cap by the search's own robust train/valid score (a holdout-clean
    # quantity present on every candidate payload) so the valid-window scoring
    # work below stays bounded without ever ranking members by holdout metrics.
    pool: list[tuple[float, Dict[str, Any]]] = []
    for row in candidate_rows:
        if bool(row.get("duplicate_seed")) or row.get("source") == "ensemble":
            continue
        payload = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        score = _metric_float(payload.get("score"))
        pool.append((score if score is not None else -1.0e18, row))
    pool.sort(key=lambda item: item[0], reverse=True)

    members: list[EnsembleMember] = []
    for _, row in pool[:consider_cap]:
        ast = (row.get("candidate") or {}).get("expression_ast")
        if not isinstance(ast, dict):
            continue
        try:
            member_expr = deserialize_expression(ast)
        except Exception:
            continue
        valid_metrics = _ensure_member_valid_metrics(
            row, member_expr, valid_panel, valid_rerank_config
        )
        entry = valid_metrics.get(ic_key) if isinstance(valid_metrics, dict) else None
        ic = (
            _metric_float(entry.get("rank_ic_mean"))
            if isinstance(entry, dict)
            else None
        )
        if ic is None:
            continue
        # Economic alignment (Lever 2): build_ensemble_expression orients each
        # member by sign(IC). A member whose long-short return disagrees with its
        # IC sign would have its long-short flipped *negative* once oriented,
        # cancelling the spread the long-short strategy actually trades. Keep only
        # members whose IC and long-short agree (ic * ls > 0) -- both measured on
        # the selection window -- so the blend compounds a positive long-short.
        ls = (
            _metric_float(entry.get("long_short_return_mean"))
            if isinstance(entry, dict)
            else None
        )
        if ls is not None and float(ic) * float(ls) <= 0.0:
            continue
        refs = sorted(referenced_features(member_expr))
        family = _feature_family(refs[0]) if refs else "unknown"
        try:
            prediction = _evaluate_expression_for_rerank(
                member_expr, oos_panel, rerank_config
            )
        except Exception:
            prediction = None
        # Cross-sectional dispersion guard: a member that is constant within each
        # date (e.g. a ``sign(...)`` variant of a same-sign base) z-scores to NaN
        # and, via the composite's ``add``, would NaN the whole blend. Drop it.
        # (Label-free use of the OOS panel: no holdout returns are observed.)
        if not _prediction_has_cross_sectional_dispersion(
            prediction, oos_panel["date"]
        ):
            continue
        members.append(
            EnsembleMember(
                rank=int(row.get("rank") or 0),
                expression=member_expr,
                ic=float(ic),
                family=family,
                prediction=prediction,
            )
        )

    selected_members = select_ensemble_members(
        members,
        max_factors=max_factors,
        max_abs_corr=float(ensemble_cfg.get("max_abs_corr", 0.7)),
        min_member_ic=float(ensemble_cfg.get("min_member_ic", 0.01)),
        one_per_family=bool(ensemble_cfg.get("one_per_family", True)),
    )
    composite_expr = build_ensemble_expression(
        selected_members,
        available_operators=set(feature_space.operators),
        min_members=int(ensemble_cfg.get("min_members", 2)),
    )
    if composite_expr is None:
        return

    ensemble_candidate = SearchCandidate(
        expression=composite_expr,
        train_metrics={},
        valid_metrics={},
        score=0.0,
        complexity=expression_complexity(composite_expr),
    )
    ensemble_row, ensemble_passed, ensemble_sort_key = build_row(
        int(ensemble_rank),
        ensemble_candidate,
        {"attempted": False, "optimized": False, "reason": "ensemble_synthesis"},
        source="ensemble",
        duplicate_seed=False,
    )
    ensemble_row["ensemble_members"] = [
        {
            "rank": member.rank,
            "ic": member.ic,
            "family": member.family,
            "expression": member.expression.to_string(),
        }
        for member in selected_members
    ]
    candidate_rows.append(ensemble_row)
    if ensemble_passed:
        _add_qd_candidate(
            qd_archive,
            str(ensemble_row["qd_bucket_key"]),
            (ensemble_sort_key, ensemble_row, ensemble_candidate, int(ensemble_rank)),
        )


def _maybe_diversify_elites(
    *,
    candidate_rows: list[Dict[str, Any]],
    qd_archive: dict,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    valid_rerank_config: dict,
    feature_space: FeatureSpace,
    mining_config: dict,
    rerank_config: dict,
    build_row: Any,
    existing_expression_hashes: set[str],
    start_rank: int,
) -> dict[str, int]:
    """Spin decorrelated variants around the run's own top elites (Lever 3).

    The original seeded-QD only generates variants when the search *exactly*
    rediscovers a known/promoted expression (``duplicate_seed`` -- an AST-hash
    match against the novelty context). With a mutating GP and the size family
    excluded that match essentially never fires (``seed_count`` stays 0), so the
    IC-weighted ensemble starves on the three or four decorrelated members a
    single search surfaces and its composite IC tops out below the gate even
    though the decorrelation ceiling is well above it.

    This pass decouples variant generation from exact rediscovery: it takes the
    strongest *sign-aligned* in-run elites -- ranked on the selection-window
    (valid) metrics, the same holdout-clean pool the ensemble draws from -- and
    spins ``_generate_seed_variants`` around each, scoring the variants
    through the identical lightweight gate and appending them to
    ``candidate_rows`` (and the QD archive when they pass). The ensemble builder
    then has materially more orthogonal members to decorrelate, lifting the
    composite toward the achievable ceiling without touching any gate threshold.

    No-op unless ``post_rerank.ensemble.diversify_elites.enabled`` is set, so the
    default behaviour is byte-for-byte unchanged.
    """
    summary = {"generated": 0, "scored": 0, "passed": 0, "next_rank": int(start_rank)}
    ensemble_cfg = rerank_config.get("ensemble", {}) or {}
    div_cfg = ensemble_cfg.get("diversify_elites", {}) or {}
    if not div_cfg.get("enabled", False):
        return summary

    elite_count = max(1, int(div_cfg.get("elite_count", 3)))
    max_variants_per_elite = max(0, int(div_cfg.get("max_variants_per_elite", 16)))
    if max_variants_per_elite == 0:
        return summary
    min_member_ic = float(ensemble_cfg.get("min_member_ic", 0.01))
    ic_key = _ensemble_ic_key(rerank_config)

    # Rank the in-run search elites the way the ensemble does: strongest |IC|
    # first, keep only sign-aligned (ic * long_short > 0) non-duplicate singles.
    # Elite choice is a SUPERVISED selection decision, so it reads the
    # selection-window (valid) metrics -- never the OOS holdout -- to keep the
    # variant pool (and thus the downstream ensemble) holdout-clean.
    eligible: list[tuple[float, Dict[str, Any], Any]] = []
    for row in candidate_rows:
        if row.get("source") != "search_hall" or bool(row.get("duplicate_seed")):
            continue
        ast = (row.get("candidate") or {}).get("expression_ast")
        if not isinstance(ast, dict):
            continue
        try:
            seed_expr = deserialize_expression(ast)
        except Exception:
            continue
        valid_metrics = _ensure_member_valid_metrics(
            row, seed_expr, valid_panel, valid_rerank_config
        )
        entry = valid_metrics.get(ic_key) if isinstance(valid_metrics, dict) else None
        if not isinstance(entry, dict):
            continue
        ic = _metric_float(entry.get("rank_ic_mean"))
        if ic is None or abs(ic) < min_member_ic:
            continue
        ls = _metric_float(entry.get("long_short_return_mean"))
        if ls is not None and float(ic) * float(ls) <= 0.0:
            continue
        eligible.append((abs(float(ic)), row, seed_expr))
    eligible.sort(key=lambda item: item[0], reverse=True)

    # Dedup variants against the novelty context AND everything already scored.
    seen_hashes: set[str] = set(existing_expression_hashes)
    for row in candidate_rows:
        row_hash = row.get("expression_hash")
        if row_hash:
            seen_hashes.add(str(row_hash))

    rank_cursor = int(start_rank)
    for _, row, seed_expr in eligible[:elite_count]:
        variants = _generate_seed_variants(
            seed_expr,
            feature_space,
            mining_config,
            max_variants=max_variants_per_elite,
        )
        summary["generated"] += len(variants)
        seed_hash = str(row.get("expression_hash") or "")
        seed_string = seed_expr.to_string()
        for variant_expr, variant_operator in variants:
            variant_hash = _task_manager_ast_hash(variant_expr)
            if variant_hash in seen_hashes:
                continue
            seen_hashes.add(variant_hash)
            scored = score_candidate(
                variant_expr, train_panel, valid_panel, mining_config
            )
            if scored is None:
                continue
            summary["scored"] += 1
            variant_row, variant_passed, variant_sort_key = build_row(
                rank_cursor,
                scored,
                {"attempted": False, "optimized": False, "reason": "elite_variant"},
                source="elite_variant",
                duplicate_seed=False,
                seed_rank=int(row.get("rank") or 0),
                seed_expression=seed_string,
                seed_expression_hash=seed_hash,
                variant_operator=variant_operator,
            )
            candidate_rows.append(variant_row)
            rank_cursor += 1
            if variant_passed:
                summary["passed"] += 1
                _add_qd_candidate(
                    qd_archive,
                    str(variant_row["qd_bucket_key"]),
                    (variant_sort_key, variant_row, scored, int(variant_row["rank"])),
                )
    summary["next_rank"] = rank_cursor
    return summary


def _candidate_windows(feature_space: FeatureSpace, mining_config: dict) -> list[int]:
    raw = feature_space.windows or tuple(
        mining_config.get("search", {}).get("windows", [3, 5, 10, 20])
    )
    values: list[int] = []
    for value in raw:
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0 and parsed not in values:
            values.append(parsed)
    return sorted(values)[:5]


def _seed_substitution_features(ref: str, feature_space: FeatureSpace) -> list[str]:
    features = sorted(feature_space.features)
    same_family = [
        name
        for name in features
        if name != ref and _feature_family(name) == _feature_family(ref)
    ]
    other_family = [
        name for name in features if name != ref and name not in same_family
    ]
    return (same_family + other_family)[:10]


def _append_variant(
    variants: list[tuple[Any, str]],
    seen_strings: set[str],
    expr,
    operator: str,
    *,
    max_variants: int,
) -> None:
    if len(variants) >= max_variants:
        return
    key = expr.to_string()
    if key in seen_strings:
        return
    seen_strings.add(key)
    variants.append((expr, operator))


def _generate_seed_variants(
    seed_expr,
    feature_space: FeatureSpace,
    mining_config: dict,
    *,
    max_variants: int,
) -> list[tuple[Any, str]]:
    variants: list[tuple[Any, str]] = []
    seen_strings = {seed_expr.to_string()}
    windows = _candidate_windows(feature_space, mining_config)

    for op_name in (
        "neg",
        "abs",
        "log",
        "sqrt",
        "cs_rank",
        "cs_zscore",
        "cs_demean",
        "cs_scale",
        "sign",
        "neutralize_size",
        "neutralize_styles",
    ):
        if op_name not in feature_space.operators:
            continue
        if seed_expr.kind == "operator" and str(seed_expr.value) == op_name:
            continue
        _append_variant(
            variants,
            seen_strings,
            _op_node(op_name, seed_expr),
            f"unary:{op_name}",
            max_variants=max_variants,
        )

    refs = sorted(referenced_features(seed_expr))
    for ref in refs:
        for replacement in _seed_substitution_features(ref, feature_space):
            _append_variant(
                variants,
                seen_strings,
                _replace_feature(seed_expr, ref, replacement),
                f"feature_substitution:{ref}->{replacement}",
                max_variants=max_variants,
            )

    feature_refs = sorted(feature_space.features)[:10]
    for ref in feature_refs:
        feature = _feature_node(ref)
        for op_name in ("add", "sub", "mul", "div"):
            if op_name not in feature_space.operators:
                continue
            _append_variant(
                variants,
                seen_strings,
                _op_node(op_name, seed_expr, feature),
                f"binary:{op_name}:seed_feature",
                max_variants=max_variants,
            )
            if op_name in {"sub", "div"}:
                _append_variant(
                    variants,
                    seen_strings,
                    _op_node(op_name, feature, seed_expr),
                    f"binary:{op_name}:feature_seed",
                    max_variants=max_variants,
                )

    for op_name in (
        "delay",
        "delta",
        "pct_change",
        "ts_mean",
        "ts_std",
        "ts_rank",
        "ts_zscore",
        "decay_linear",
        "ema",
    ):
        if op_name not in feature_space.operators:
            continue
        for window in windows:
            _append_variant(
                variants,
                seen_strings,
                _op_node(op_name, seed_expr, _window_node(window)),
                f"window:{op_name}:{window}",
                max_variants=max_variants,
            )

    if "ts_residual" in feature_space.operators:
        for ref in feature_refs[:5]:
            for window in windows[:3]:
                _append_variant(
                    variants,
                    seen_strings,
                    _op_node(
                        "ts_residual",
                        seed_expr,
                        _feature_node(ref),
                        _window_node(window),
                    ),
                    f"residual:ts_residual:{ref}:{window}",
                    max_variants=max_variants,
                )

    return variants


def _row_sort_key(
    candidate: SearchCandidate, lightweight: Dict[str, Any]
) -> tuple[float, float, float, float]:
    sort_values = lightweight["sort_values"]
    return (
        float(sort_values[0]),
        float(sort_values[1]),
        float(sort_values[2]),
        float(-candidate.complexity),
    )


QDArchiveItem = tuple[
    tuple[float, float, float, float],
    Dict[str, Any],
    SearchCandidate,
    int,
]


def _add_qd_candidate(
    archive: dict[str, QDArchiveItem],
    key: str,
    item: QDArchiveItem,
) -> None:
    existing = archive.get(key)
    if existing is None or item[0] > existing[0]:
        archive[key] = item


def _pool_residualized_ic(
    candidate: SearchCandidate,
    oos_panel: pd.DataFrame,
    rerank_config: dict,
    period: int,
) -> pd.Series | None:
    """Date-indexed residualized daily rank IC for one candidate over the OOS window.

    Residualizes the candidate's OOS prediction against the panel anchors (size +
    mainline) per date -- the same incremental signal the marginal-aware search
    optimizes -- then computes the daily cross-sectional Spearman IC against the
    ``period`` forward-return label. Returning a date-indexed series lets the
    whole candidate pool share a common time axis for CSCV/PBO. ``None`` when the
    panel is empty, lacks the label, or yields no valid daily IC.
    """
    if oos_panel is None or oos_panel.empty or "date" not in oos_panel.columns:
        return None
    try:
        panel, _ = _validation_panel_for_lightweight(oos_panel, rerank_config)
        if panel.empty:
            return None
        pred = _evaluate_expression_for_rerank(
            candidate.expression, panel, rerank_config
        )
        pred = pd.Series(pred, index=panel.index, dtype=float).replace(
            [np.inf, -np.inf], np.nan
        )
        label, source = _period_label_from_panel(panel, int(period))
        if source == "missing":
            return None
        anchors = anchor_matrix_from_panel(panel)
        if anchors is not None:
            residual_context = _residualization_context_for_panel(
                panel, anchors, rerank_config
            )
            if residual_context is not None:
                pred = residual_context.residualize(pred, index=pred.index)
            else:
                pred = residualize_by_date(pred, panel["date"], anchors)
        ic = rank_ic_by_date(pred, label, panel["date"])
    except Exception:
        return None
    if ic is None or ic.empty:
        return None
    return ic.sort_index()


def _estimate_pbo(
    ic_by_candidate: dict[int, pd.Series],
    *,
    n_blocks: int,
    embargo: int,
    min_candidates: int,
    sign_by_candidate: dict[int, float] | None = None,
) -> dict:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV.

    Builds a (dates x candidates) residualized-IC matrix, splits the dates into
    ``n_blocks`` contiguous time blocks, and for every balanced split of blocks
    into in-sample / out-of-sample halves picks the IS-best candidate and reads
    off its OOS relative rank ``w``. ``PBO`` is the fraction of splits whose
    logit ``ln(w/(1-w)) <= 0`` -- i.e. the IS winner lands below the OOS median,
    the signature of an overfit selection (Bailey/Borwein/Lopez de Prado/Zhu).

    A neighbor-aware embargo trims ``embargo`` rows from any OOS block edge
    adjacent to an IS block, preventing leakage through the overlapping
    multi-day return labels. Candidate series are sign-aligned (a consistently
    negative-IC factor is deployed negated) so the ranking compares factor
    quality, not sign. The deployment orientation must be fixed BEFORE the
    window the matrix spans: pass ``sign_by_candidate`` (sign of each
    candidate's selection-window IC, keyed like ``ic_by_candidate``). Columns
    without a usable provided sign fall back to the legacy full-window mean
    sign -- a mild peek that slightly understates PBO -- and are counted in
    ``sign_fallback_count``. Returns ``available=False`` with a ``reason`` when
    there are too few candidates or dates to form the splits.
    """
    result: dict[str, Any] = {
        "available": False,
        "value": None,
        "n_blocks": int(n_blocks),
        "embargo": int(embargo),
        "n_candidates": int(len(ic_by_candidate)),
        "n_combos": 0,
        "reason": "",
    }
    usable = {
        rank: series
        for rank, series in ic_by_candidate.items()
        if isinstance(series, pd.Series) and not series.empty
    }
    if len(usable) < int(min_candidates):
        result["reason"] = "insufficient_candidates"
        result["n_candidates"] = int(len(usable))
        return result

    matrix = pd.DataFrame(usable).sort_index().dropna(how="any")
    n_dates = int(matrix.shape[0])
    n_candidates = int(matrix.shape[1])
    result["n_candidates"] = n_candidates
    if n_candidates < int(min_candidates):
        result["reason"] = "insufficient_overlapping_candidates"
        return result

    # CSCV requires an even number of blocks to split into equal halves.
    s_blocks = int(n_blocks)
    if s_blocks % 2 == 1:
        s_blocks -= 1
    if s_blocks < 2:
        result["reason"] = "too_few_blocks"
        return result
    if n_dates < s_blocks * 2:
        result["reason"] = "insufficient_dates"
        return result

    values = matrix.to_numpy(dtype=float)
    # Sign-align each candidate so IS/OOS ranking reflects quality, not sign.
    # Preferred source: the caller-provided selection-window orientation (fixed
    # before this matrix's window). Fallback: full-window mean (legacy, peeks).
    fallback_signs = np.where(np.nanmean(values, axis=0) < 0.0, -1.0, 1.0)
    sign_fallback_count = 0
    if sign_by_candidate:
        resolved_signs = np.empty(n_candidates, dtype=float)
        for idx, column_rank in enumerate(matrix.columns):
            provided = sign_by_candidate.get(column_rank)
            if (
                provided is not None
                and np.isfinite(provided)
                and float(provided) != 0.0
            ):
                resolved_signs[idx] = 1.0 if float(provided) > 0.0 else -1.0
            else:
                resolved_signs[idx] = float(fallback_signs[idx])
                sign_fallback_count += 1
        signs = resolved_signs
    else:
        signs = fallback_signs
        sign_fallback_count = n_candidates
    result["sign_source"] = (
        "pre_window" if sign_by_candidate else "full_window_fallback"
    )
    result["sign_fallback_count"] = int(sign_fallback_count)
    values = values * signs

    block_positions = [
        np.asarray(part, dtype=np.int64)
        for part in np.array_split(np.arange(n_dates), s_blocks)
    ]
    embargo_rows = max(0, int(embargo))
    half = s_blocks // 2
    logits: list[float] = []
    for is_blocks in combinations(range(s_blocks), half):
        is_set = set(is_blocks)
        is_rows = np.concatenate([block_positions[i] for i in is_set])
        oos_rows_parts: list[np.ndarray] = []
        for j in range(s_blocks):
            if j in is_set:
                continue
            rows = block_positions[j]
            if embargo_rows and rows.size:
                if (j - 1) in is_set:
                    rows = rows[embargo_rows:]
                if (j + 1) in is_set and rows.size:
                    rows = rows[: rows.size - embargo_rows]
            if rows.size:
                oos_rows_parts.append(rows)
        if not oos_rows_parts:
            continue
        oos_rows = np.concatenate(oos_rows_parts)
        is_perf = np.nanmean(values[is_rows, :], axis=0)
        oos_perf = np.nanmean(values[oos_rows, :], axis=0)
        if not np.any(np.isfinite(is_perf)) or not np.any(np.isfinite(oos_perf)):
            continue
        best = int(np.nanargmax(is_perf))
        ranks = pd.Series(oos_perf).rank(method="average").to_numpy()
        relative_rank = float(ranks[best] / (n_candidates + 1))
        relative_rank = min(max(relative_rank, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))

    if not logits:
        result["reason"] = "no_valid_splits"
        return result
    logit_array = np.asarray(logits, dtype=float)
    result.update(
        {
            "available": True,
            "value": float(np.mean(logit_array <= 0.0)),
            "n_combos": int(logit_array.size),
            "blocks_used": int(s_blocks),
            "n_dates": n_dates,
            "reason": "ok",
        }
    )
    return result


def _compute_pbo_summary(
    *,
    pbo_pool_candidates: list[tuple[int, SearchCandidate]],
    train_panel: pd.DataFrame,
    selection_panel: pd.DataFrame,
    rerank_config: dict,
    is_rerank_config: dict,
    selected_best_period: Any,
    periods_list: list[int],
) -> dict[str, Any]:
    pbo_config = rerank_config.get("pbo", {}) or {}
    pbo_summary: dict[str, Any] = {"enabled": bool(pbo_config.get("enabled", False))}
    if not pbo_config.get("enabled", False):
        pbo_summary.update({"available": False, "value": None, "reason": "disabled"})
        return pbo_summary

    pbo_period = (
        int(selected_best_period)
        if selected_best_period is not None
        else (periods_list[0] if periods_list else 1)
    )
    ic_by_candidate: dict[int, pd.Series] = {}
    sign_by_candidate: dict[int, float] = {}
    for pool_rank, pool_candidate in pbo_pool_candidates:
        series = _pool_residualized_ic(
            pool_candidate, selection_panel, rerank_config, pbo_period
        )
        if series is None:
            continue
        ic_by_candidate[pool_rank] = series
        train_series = _pool_residualized_ic(
            pool_candidate, train_panel, is_rerank_config, pbo_period
        )
        if train_series is not None and not train_series.empty:
            train_mean = float(train_series.mean())
            if np.isfinite(train_mean) and train_mean != 0.0:
                sign_by_candidate[pool_rank] = 1.0 if train_mean > 0.0 else -1.0
    pbo_result = _estimate_pbo(
        ic_by_candidate,
        n_blocks=int(pbo_config.get("blocks", 10)),
        embargo=int(pbo_config.get("embargo_days", 20)),
        min_candidates=int(pbo_config.get("min_candidates", 4)),
        sign_by_candidate=sign_by_candidate,
    )
    pbo_threshold = float(pbo_config.get("threshold", 0.5))
    pbo_value = pbo_result.get("value")
    pbo_summary.update(
        {
            "mode": str(pbo_config.get("mode", "observe")),
            "period": int(pbo_period),
            "threshold": pbo_threshold,
            "pool_size": int(len(pbo_pool_candidates)),
            "pool_source": "lightweight_qd_survivors",
            "would_block": bool(
                pbo_value is not None and float(pbo_value) > pbo_threshold
            ),
            **pbo_result,
        }
    )
    return pbo_summary


def _evaluate_holdout_confirm(
    candidate: SearchCandidate | None,
    holdout_panel: pd.DataFrame | None,
    rerank_config: dict,
    *,
    dual_gate_passed: bool,
    gate_period: int | None,
) -> dict[str, Any]:
    """Single-look confirmation of the selected candidate on the holdout panel.

    This is the ONLY code path allowed to consume the holdout window. It runs
    once per selection, only for a dual-gate-passing survivor, and its output
    feeds the ic-significance (deflated-t) and selection->confirm decay gates
    downstream. Runs that fail the selection gate never look at the holdout,
    so they do not burn campaign multiplicity.
    """
    base: dict[str, Any] = {
        "evaluated": False,
        "reason": "",
        "period": int(gate_period) if gate_period is not None else None,
        "rank_ic_mean": None,
        "rank_ic_t": None,
        "top_quantile_excess_return_mean": None,
        "top_quantile_excess_return_t": None,
        "period_metrics": {},
        "window": {},
        "confirms_this_run": 0,
    }
    if not dual_gate_passed or candidate is None:
        base["reason"] = "dual_gate_failed_no_holdout_look"
        return base
    if holdout_panel is None or holdout_panel.empty:
        base["reason"] = "holdout_panel_empty"
        return base
    # Bound the confirmation look to the configured confirm window (e.g. 2025)
    # so data beyond it (e.g. 2026+) stays pristine for independent audits.
    confirm_start = str(rerank_config.get("holdout_confirm_start") or "").strip()
    confirm_end = str(rerank_config.get("holdout_confirm_end") or "").strip()
    if (confirm_start or confirm_end) and "date" in holdout_panel.columns:
        panel_dates = pd.to_datetime(holdout_panel["date"], errors="coerce")
        mask = panel_dates.notna()
        if confirm_start:
            mask &= panel_dates >= pd.Timestamp(confirm_start)
        if confirm_end:
            mask &= panel_dates <= pd.Timestamp(confirm_end)
        holdout_panel = holdout_panel.loc[mask]
    if holdout_panel.empty:
        base["reason"] = "holdout_confirm_window_empty"
        return base
    confirm_config = {
        key: value
        for key, value in rerank_config.items()
        if key not in {"validation_start", "validation_end"}
    }
    lightweight = _evaluate_candidate_lightweight(
        candidate, holdout_panel, confirm_config
    )
    if not lightweight.get("ok"):
        base["reason"] = str(
            lightweight.get("fail_reason") or "holdout_lightweight_failed"
        )
        return base
    period_metrics = lightweight["period_metrics"]
    period_key = str(int(gate_period)) if gate_period is not None else None
    if period_key is None and period_metrics:
        period_key = next(iter(period_metrics))
    entry = period_metrics.get(period_key) if period_key is not None else None
    entry = entry if isinstance(entry, dict) else {}
    dates = pd.to_datetime(holdout_panel["date"], errors="coerce").dropna()
    return {
        **base,
        "evaluated": True,
        "reason": "",
        "period": int(period_key) if period_key is not None else None,
        "rank_ic_mean": _metric_float(entry.get("rank_ic_mean")),
        "rank_ic_t": _metric_float(entry.get("rank_ic_t")),
        "top_quantile_excess_return_mean": _metric_float(
            entry.get("top_quantile_excess_return_mean")
        ),
        "top_quantile_excess_return_t": _metric_float(
            entry.get("top_quantile_excess_return_t")
        ),
        "period_metrics": period_metrics,
        "window": {
            "start": str(dates.min().date()) if not dates.empty else "",
            "end": str(dates.max().date()) if not dates.empty else "",
        },
        "confirms_this_run": 1,
    }


def run_post_rerank(
    search_result: SearchResult,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    feature_space: FeatureSpace,
    mining_config: dict,
    analysis_context: AnalysisContext,
    analysis_config_path: str,
    run_root: Path,
    factor_dir: Path,
    run_id: str,
    novelty_context_path: str | Path,
    holdout_panel: pd.DataFrame | None = None,
) -> PostRerankResult:
    rerank_config = mining_config.get("post_rerank", {})
    # Selection/confirmation split: ALL supervised selection activity (dual
    # gate, top-K ordering, QD elites, seed/elite variants, ensemble, PBO) runs
    # on the SELECTION window -- the search valid panel filtered by
    # ``validation_start/end`` (e.g. 2024). The holdout panel (e.g. 2025) is
    # consumed exactly once, AFTER selection, to confirm the single selected
    # candidate (the ``holdout_confirm`` block); no other code path may touch
    # it. This reverses the legacy design where every candidate's gate ran on
    # the holdout and each run burnt campaign-level multiplicity.
    selection_panel = valid_panel
    # In-sample (train-window) metrics for decay diagnostics. Dropping the
    # validation window makes the lightweight evaluator score the whole train
    # panel instead of the selection window.
    is_rerank_config = {
        key: value
        for key, value in rerank_config.items()
        if key not in {"validation_start", "validation_end"}
    }
    # Window-unfiltered config: used for the train-window pass and the
    # holdout-confirm pass (each scores the entire panel it is given).
    valid_rerank_config = dict(is_rerank_config)
    novelty_context = _load_novelty_context(novelty_context_path)
    existing_expression_hashes = set(novelty_context["existing_expression_hashes"])
    top_k = max(1, int(rerank_config.get("top_k", 10)))
    seeded_qd_config = rerank_config.get("seeded_qd", {})
    if not isinstance(seeded_qd_config, dict):
        seeded_qd_config = {}
    max_seed_variants = max(
        0,
        int(
            novelty_context.get(
                "max_seed_variants",
                seeded_qd_config.get("max_seed_variants", 50),
            )
        ),
    )
    timeout_sec = int(
        rerank_config.get(
            "analysis_timeout_sec",
            mining_config.get("validation", {}).get("analysis_timeout_sec", 600),
        )
    )
    max_full_analysis_attempts = max(
        1, int(rerank_config.get("max_full_analysis_attempts", 3))
    )
    smoke_analysis_config_path, smoke_analysis = _write_smoke_analysis_config(
        analysis_config_path,
        run_root,
        rerank_config,
    )
    candidates = list(search_result.hall_of_fame[:top_k])
    validation_start, validation_end = _validation_window(mining_config)
    candidate_rows: list[Dict[str, Any]] = []
    selected_row: Dict[str, Any] | None = None
    selected_candidate: SearchCandidate | None = None
    selected_factor_path: Path | None = None
    qd_archive: dict[str, QDArchiveItem] = {}
    seed_count = 0
    seed_variant_generated_count = 0
    seed_variant_scored_count = 0
    seed_variant_lightweight_pass_count = 0
    seed_variant_duplicate_skipped_count = 0
    best_seed_variant_score: float | None = None
    best_seed_variant_rank: int | None = None
    next_variant_rank = top_k + 1

    def build_row(
        rank: int,
        candidate: SearchCandidate,
        optimization_info: Dict[str, Any],
        *,
        source: str,
        duplicate_seed: bool,
        seed_rank: int | None = None,
        seed_expression: str = "",
        seed_expression_hash: str = "",
        variant_operator: str = "",
    ) -> tuple[Dict[str, Any], bool, tuple[float, float, float, float]]:
        factor_path = _candidate_factor_path(factor_dir, run_id, rank)
        lightweight = _evaluate_candidate_lightweight(
            candidate, selection_panel, rerank_config
        )
        portfolio_gate = dict(lightweight["portfolio_gate"])
        # Only spend extra passes on candidates that could be selected (those
        # that clear the selection primary gate): the decay diagnostics only
        # consume the selected candidate's IS metrics, and the
        # reference-correlation gate (a real DB lookup) is wasted on candidates
        # the primary gate already rejected.
        is_period_metrics: Dict[str, Any] = {}
        # Per-candidate selection-window metrics, stored under the legacy
        # ``valid_period_metrics`` key so the harness alternate-candidate
        # materialization path (which has no panels) can read the *selected*
        # candidate's selection-window IC. With selection running on the valid
        # panel these are the primary lightweight metrics themselves.
        valid_period_metrics: Dict[str, Any] = {}
        if lightweight.get("ok") and lightweight.get("primary_passed"):
            is_lightweight = _evaluate_candidate_lightweight(
                candidate, train_panel, is_rerank_config
            )
            if is_lightweight.get("ok"):
                is_period_metrics = is_lightweight["period_metrics"]
            valid_period_metrics = lightweight["period_metrics"]
            portfolio_config = rerank_config.get("portfolio_gate", {}) or {}
            if portfolio_config.get("enabled") and portfolio_config.get(
                "reference_factors"
            ):
                portfolio_gate = _evaluate_reference_correlation_gate(
                    candidate, selection_panel, analysis_context, rerank_config
                )
        sort_key = _row_sort_key(candidate, lightweight)
        passed = bool(
            lightweight["primary_passed"] and portfolio_gate.get("passed", True)
        )
        task_manager_expression_hash = _task_manager_ast_hash(candidate.expression)
        qd_bucket_key = _qd_bucket_key(candidate.expression, feature_space)
        row = {
            "rank": int(rank),
            "source": source,
            "factor_name": factor_path.stem,
            "factor_file": "",
            "planned_factor_file": str(factor_path.resolve()),
            "candidate": {
                **_candidate_payload(candidate),
                "task_manager_expression_hash": task_manager_expression_hash,
            },
            "expression_hash": task_manager_expression_hash,
            "task_manager_expression_hash": task_manager_expression_hash,
            "duplicate_seed": bool(duplicate_seed),
            "seed_rank": int(seed_rank) if seed_rank is not None else None,
            "seed_expression": seed_expression,
            "seed_expression_hash": seed_expression_hash,
            "variant_operator": variant_operator,
            "qd_bucket_key": qd_bucket_key,
            "qd_gate_passed": False,
            "constant_optimization": optimization_info,
            "analysis": {
                "ok": None,
                "skipped": True,
                "reason": "deferred_until_lightweight_selected",
            },
            "validation_start": validation_start,
            "validation_end": validation_end,
            "metric_source": "lightweight",
            "lightweight": {
                "ok": bool(lightweight["ok"]),
                "duration_sec": float(lightweight["duration_sec"]),
                "evaluate_duration_sec": float(lightweight["evaluate_duration_sec"]),
            },
            "lightweight_period_metrics": lightweight["period_metrics"],
            "period_metrics": lightweight["period_metrics"],
            "is_period_metrics": is_period_metrics,
            "valid_period_metrics": valid_period_metrics,
            "primary_gate_passed": bool(lightweight["primary_passed"]),
            "lightweight_primary_gate_passed": bool(lightweight["primary_passed"]),
            "portfolio_gate": portfolio_gate,
            "portfolio_gate_passed": bool(portfolio_gate.get("passed", True)),
            "turnover_metrics": lightweight["turnover_metrics"],
            "dual_gate_passed": bool(passed),
            "best_gate_period": lightweight["best_period"],
            "warnings": list(lightweight["warnings"]),
            "selection_sort_key": [
                float(sort_key[0]),
                float(sort_key[1]),
                float(sort_key[2]),
                float(sort_key[3]),
            ],
        }
        if duplicate_seed:
            row["excluded_from_full_analysis_reason"] = "exact_duplicate_seed"
        return row, passed, sort_key

    for rank, original_candidate in enumerate(candidates, start=1):
        candidate, optimization_info = _optimize_constants_for_train(
            original_candidate,
            train_panel,
            valid_panel,
            mining_config,
        )
        task_manager_expression_hash = _task_manager_ast_hash(candidate.expression)
        duplicate_seed = task_manager_expression_hash in existing_expression_hashes
        row, passed, sort_key = build_row(
            rank,
            candidate,
            optimization_info,
            source="search_hall",
            duplicate_seed=duplicate_seed,
        )
        candidate_rows.append(row)

        if duplicate_seed:
            seed_count += 1
            variants = _generate_seed_variants(
                candidate.expression,
                feature_space,
                mining_config,
                max_variants=max_seed_variants,
            )
            seed_variant_generated_count += len(variants)
            for variant_expr, variant_operator in variants:
                variant_hash = _task_manager_ast_hash(variant_expr)
                if (
                    variant_hash in existing_expression_hashes
                    or variant_hash == task_manager_expression_hash
                ):
                    seed_variant_duplicate_skipped_count += 1
                    continue
                scored = score_candidate(
                    variant_expr, train_panel, valid_panel, mining_config
                )
                if scored is None:
                    continue
                seed_variant_scored_count += 1
                variant_rank = next_variant_rank
                next_variant_rank += 1
                variant_row, variant_passed, variant_sort_key = build_row(
                    variant_rank,
                    scored,
                    {"attempted": False, "optimized": False, "reason": "seed_variant"},
                    source="seed_variant",
                    duplicate_seed=False,
                    seed_rank=rank,
                    seed_expression=candidate.expression.to_string(),
                    seed_expression_hash=task_manager_expression_hash,
                    variant_operator=variant_operator,
                )
                candidate_rows.append(variant_row)
                if variant_passed:
                    seed_variant_lightweight_pass_count += 1
                    if (
                        best_seed_variant_score is None
                        or scored.score > best_seed_variant_score
                    ):
                        best_seed_variant_score = float(scored.score)
                        best_seed_variant_rank = int(variant_rank)
                    _add_qd_candidate(
                        qd_archive,
                        str(variant_row["qd_bucket_key"]),
                        (variant_sort_key, variant_row, scored, int(variant_rank)),
                    )
            continue

        if passed:
            _add_qd_candidate(
                qd_archive,
                str(row["qd_bucket_key"]),
                (sort_key, row, candidate, int(rank)),
            )

    # Lever 3: revive seeded-QD diversity for the size-excluded regime by spinning
    # variants around the run's own top elites (not just exact rediscoveries),
    # enriching the pool the ensemble builder draws members from. No-op unless
    # post_rerank.ensemble.diversify_elites.enabled is set.
    elite_diversify_summary = _maybe_diversify_elites(
        candidate_rows=candidate_rows,
        qd_archive=qd_archive,
        train_panel=train_panel,
        valid_panel=valid_panel,
        valid_rerank_config=valid_rerank_config,
        feature_space=feature_space,
        mining_config=mining_config,
        rerank_config=rerank_config,
        build_row=build_row,
        existing_expression_hashes=existing_expression_hashes,
        start_rank=next_variant_rank,
    )
    next_variant_rank = int(elite_diversify_summary["next_rank"])

    _maybe_add_ensemble_candidate(
        candidate_rows=candidate_rows,
        qd_archive=qd_archive,
        oos_panel=selection_panel,
        valid_panel=valid_panel,
        valid_rerank_config=valid_rerank_config,
        rerank_config=rerank_config,
        feature_space=feature_space,
        build_row=build_row,
        ensemble_rank=next_variant_rank,
    )

    qd_ranks = {int(item[3]) for item in qd_archive.values()}
    for row in candidate_rows:
        rank = int(row.get("rank") or 0)
        qd_gate_passed = rank in qd_ranks and not bool(row.get("duplicate_seed", False))
        row["qd_gate_passed"] = bool(qd_gate_passed)
        if (
            row.get("dual_gate_passed")
            and not qd_gate_passed
            and not row.get("duplicate_seed")
        ):
            row["excluded_from_full_analysis_reason"] = "qd_bucket_not_elite"

    lightweight_passes = sorted(
        qd_archive.values(), key=lambda item: item[0], reverse=True
    )
    periods_list = [int(period) for period in rerank_config.get("periods", [1, 5, 10])]
    pbo_pool_candidates = [
        (int(rank), candidate)
        for _, row, candidate, rank in lightweight_passes
        if bool(row.get("lightweight_primary_gate_passed"))
        and bool((row.get("portfolio_gate") or {}).get("passed", True))
        and not bool(row.get("duplicate_seed", False))
    ]
    pbo_best_period = (
        lightweight_passes[0][1].get("best_gate_period") if lightweight_passes else None
    )
    pbo_summary = _compute_pbo_summary(
        pbo_pool_candidates=pbo_pool_candidates,
        train_panel=train_panel,
        selection_panel=selection_panel,
        rerank_config=rerank_config,
        is_rerank_config=is_rerank_config,
        selected_best_period=pbo_best_period,
        periods_list=periods_list,
    )
    full_analysis_attempts = 0
    for _, row, candidate, rank in lightweight_passes[:max_full_analysis_attempts]:
        full_analysis_attempts += 1
        factor_path = _candidate_factor_path(factor_dir, run_id, rank)
        row["analysis"] = {"ok": False, "skipped": True, "reason": "not_started"}
        row["full_analysis_attempted"] = True
        compile_factor_file(candidate, mining_config, feature_space, factor_path)
        import_result = import_smoke_check(factor_path)
        factor_name = (
            import_result["classes"][0]
            if import_result.get("ok") and import_result.get("classes")
            else factor_path.stem
        )
        row["factor_name"] = factor_name
        row["factor_file"] = str(factor_path.resolve())
        row["import"] = import_result
        warnings = list(row.get("warnings", []))

        if import_result.get("ok"):
            analysis_result = run_analysis_smoke(
                factor_path, smoke_analysis_config_path, timeout_sec=timeout_sec
            )
        else:
            analysis_result = {
                "ok": False,
                "exit_code": None,
                "command": [],
                "stdout_tail": "",
                "stderr_tail": "import smoke failed",
            }
            warnings.append("candidate import failed")
        row["analysis"] = analysis_result
        row["analysis_smoke"] = smoke_analysis
        row["framework_smoke_passed"] = bool(analysis_result.get("ok"))
        if not analysis_result.get("ok"):
            warnings.append("candidate analysis failed")

        row["warnings"] = warnings
        passed = bool(
            analysis_result.get("ok")
            and row.get("lightweight_primary_gate_passed", False)
            and row.get("portfolio_gate", {}).get("passed", True)
        )
        row["dual_gate_passed"] = bool(passed)
        if passed:
            selected_row = row
            selected_candidate = candidate
            selected_factor_path = factor_path
            break

    dual_gate_passed = selected_candidate is not None
    if selected_candidate is None:
        selected_candidate = search_result.best

    def output_order(
        row: Dict[str, Any],
    ) -> tuple[int, float, float, float, float, int]:
        rank = int(row.get("rank") or 0)
        sort_key = (
            row.get("selection_sort_key")
            if isinstance(row.get("selection_sort_key"), list)
            else []
        )
        values = (
            [float(value) for value in sort_key[:4]]
            if len(sort_key) >= 4
            else [-1.0e18] * 4
        )
        if bool(row.get("qd_gate_passed")):
            group = 0
        elif bool(row.get("duplicate_seed")):
            group = 1
        else:
            group = 2
        return (group, -values[0], -values[1], -values[2], -values[3], rank)

    candidate_rows = sorted(candidate_rows, key=output_order)
    quality_gate_failure_reason = (
        ""
        if dual_gate_passed
        else _quality_gate_failure_reason(
            candidate_rows=candidate_rows,
            qd_bucket_count=len(qd_archive),
            seed_variant_lightweight_pass_count=seed_variant_lightweight_pass_count,
            rerank_config=rerank_config,
        )
    )

    # --- Selection-window metrics of the selected candidate ---
    selected_selection_metrics = (selected_row or {}).get(
        "lightweight_period_metrics"
    ) or {}
    selected_best_period = (selected_row or {}).get("best_gate_period")

    # --- Selection-window metrics for the config-search objective ---
    # The outer Optuna loop must optimise a window the holdout never gates, so
    # the *selected* candidate's selection-window metrics are surfaced under
    # the legacy ``selected_valid_*`` keys. With selection running on the valid
    # panel these come straight from the selected row; the fallback evaluation
    # covers the dual-gate-fail path (``selected_row`` is None and
    # ``selected_candidate`` is ``search_result.best``).
    # ``selected_valid_rank_ic_reason`` is always set so a None value is
    # explainable rather than silent.
    selected_valid_rank_ic: float | None = None
    selected_valid_rank_ic_t: float | None = None
    selected_valid_rank_ic_reason = ""
    valid_best_period = selected_best_period
    if valid_best_period is None:
        gate_period = rerank_config.get("gate_period")
        valid_best_period = (
            int(gate_period)
            if gate_period is not None
            else (periods_list[0] if periods_list else None)
        )
    selected_valid_metrics: dict[str, Any] = dict(selected_selection_metrics)
    if not selected_valid_metrics:
        if selected_candidate is None:
            selected_valid_rank_ic_reason = "no_selected_candidate"
        elif selection_panel is None or selection_panel.empty:
            selected_valid_rank_ic_reason = "valid_panel_empty"
        else:
            valid_lightweight = _evaluate_candidate_lightweight(
                selected_candidate, selection_panel, rerank_config
            )
            if valid_lightweight.get("ok"):
                selected_valid_metrics = valid_lightweight["period_metrics"]
                selected_selection_metrics = selected_valid_metrics
            else:
                selected_valid_rank_ic_reason = str(
                    valid_lightweight.get("fail_reason") or "valid_lightweight_failed"
                )
    if selected_valid_metrics:
        selected_valid_rank_ic = _selected_period_rank_ic(
            selected_valid_metrics, valid_best_period
        )
        selected_valid_rank_ic_t = _selected_period_tstat(
            selected_valid_metrics, valid_best_period
        )
        if selected_valid_rank_ic is None:
            selected_valid_rank_ic_reason = "valid_rank_ic_unavailable_for_period"

    # --- Holdout single-look confirmation + deflated-t multiplicity ---
    # The holdout is consumed exactly once, here, for the dual-gate survivor.
    # The deflated-t hurdle prices the campaign's cumulative number of such
    # confirmation looks (injected by the orchestrator), NOT per-run expression
    # breadth -- selection breadth is absorbed by the selection window.
    holdout_confirm = _evaluate_holdout_confirm(
        selected_candidate,
        holdout_panel,
        rerank_config,
        dual_gate_passed=dual_gate_passed,
        gate_period=int(valid_best_period) if valid_best_period is not None else None,
    )
    selected_oos_rank_ic_t = _selected_period_tstat(
        holdout_confirm.get("period_metrics") or {}, holdout_confirm.get("period")
    )
    multiplicity_correction = str(rerank_config.get("multiplicity_correction", "off"))
    min_rank_ic_t = float(rerank_config.get("min_rank_ic_t", 0.0))
    n_trials = _resolve_effective_trials(rerank_config)
    holdout_confirm["looks"] = int(n_trials)
    if multiplicity_correction == "deflated":
        hurdle_t = _deflated_t_hurdle(n_trials)
    elif multiplicity_correction == "fixed_t":
        hurdle_t = min_rank_ic_t
    else:
        hurdle_t = None
    ic_significance_would_block = bool(
        hurdle_t is not None
        and selected_oos_rank_ic_t is not None
        and selected_oos_rank_ic_t < hurdle_t
    )
    multiplicity_summary = {
        "correction": multiplicity_correction,
        "mode": str(rerank_config.get("multiplicity_mode", "observe")),
        "tstat_method": str(rerank_config.get("tstat_method", "newey_west")),
        "min_rank_ic_t": min_rank_ic_t,
        "effective_trials": int(n_trials),
        "trials_source": "holdout_confirm_looks",
        "selected_rank_ic_t": selected_oos_rank_ic_t,
        "hurdle_t": hurdle_t,
        "would_block": ic_significance_would_block,
        "available": bool(hurdle_t is not None and selected_oos_rank_ic_t is not None),
    }

    summary = {
        "enabled": True,
        "novelty_context": {
            "loaded": True,
            "path": str(novelty_context.get("path") or ""),
            "existing_expression_hash_count": int(len(existing_expression_hashes)),
            "hash_type": str(
                novelty_context.get("hash_type") or "task_manager_ast_hash"
            ),
        },
        "seeded_qd_enabled": True,
        "max_seed_variants": int(max_seed_variants),
        "seed_count": int(seed_count),
        "seed_variant_generated_count": int(seed_variant_generated_count),
        "seed_variant_scored_count": int(seed_variant_scored_count),
        "seed_variant_lightweight_pass_count": int(seed_variant_lightweight_pass_count),
        "seed_variant_duplicate_skipped_count": int(
            seed_variant_duplicate_skipped_count
        ),
        "diversify_elites_enabled": bool(
            (rerank_config.get("ensemble", {}) or {})
            .get("diversify_elites", {})
            .get("enabled", False)
        ),
        "elite_variant_generated_count": int(elite_diversify_summary["generated"]),
        "elite_variant_scored_count": int(elite_diversify_summary["scored"]),
        "elite_variant_passed_count": int(elite_diversify_summary["passed"]),
        "qd_bucket_count": int(len(qd_archive)),
        "best_seed_variant_score": best_seed_variant_score,
        "best_seed_variant_rank": best_seed_variant_rank,
        "top_k": int(top_k),
        "required": bool(rerank_config.get("required", False)),
        "validation_start": validation_start,
        "validation_end": validation_end,
        "periods": [int(period) for period in rerank_config.get("periods", [1, 5, 10])],
        "thresholds": {
            "min_rank_ic_mean": float(rerank_config.get("min_rank_ic_mean", 0.08)),
            "min_valid_rank_ic_t": float(rerank_config.get("min_valid_rank_ic_t", 0.0)),
            "min_top_quantile_excess_return": float(
                rerank_config.get("min_top_quantile_excess_return", 0.0)
            ),
            "min_top_quantile_excess_t": float(
                rerank_config.get("min_top_quantile_excess_t", 0.0)
            ),
        },
        "lightweight_enabled": True,
        "analysis_smoke": smoke_analysis,
        "full_analysis_attempts": int(full_analysis_attempts),
        "max_full_analysis_attempts": int(max_full_analysis_attempts),
        "dual_gate_passed": bool(dual_gate_passed),
        "selected_reason": "dual_gate_best_validation_candidate"
        if dual_gate_passed
        else "fallback_search_best_no_dual_gate_pass",
        "quality_gate_failure_reason": quality_gate_failure_reason,
        "selected_rank": selected_row["rank"] if selected_row else None,
        "selected_factor_file": str(selected_factor_path.resolve())
        if selected_factor_path
        else "",
        "selected_from_seed_variant": bool(
            selected_row and selected_row.get("source") == "seed_variant"
        ),
        "selected_candidate": _candidate_payload(selected_candidate),
        "search_best": _candidate_payload(search_result.best),
        "candidates": candidate_rows,
        "selected_is_period_metrics": (selected_row or {}).get("is_period_metrics")
        or {},
        # Selection-window (e.g. 2024) metrics of the selected candidate: the
        # denominator side of the selection->confirm decay gate.
        "selected_selection_period_metrics": selected_selection_metrics,
        # Holdout-confirm (e.g. 2025) metrics: single-look confirmation of the
        # selected candidate; the numerator side of the decay gate and the
        # source of the deflated-t significance check.
        "selected_oos_period_metrics": holdout_confirm.get("period_metrics") or {},
        "selected_best_period": (selected_row or {}).get("best_gate_period"),
        "selected_portfolio_gate": (selected_row or {}).get("portfolio_gate") or {},
        "selected_oos_rank_ic_t": selected_oos_rank_ic_t,
        "selected_valid_rank_ic": selected_valid_rank_ic,
        "selected_valid_rank_ic_t": selected_valid_rank_ic_t,
        "selected_valid_rank_ic_reason": selected_valid_rank_ic_reason,
        "multiplicity": multiplicity_summary,
        "pbo": pbo_summary,
        "holdout_confirm": holdout_confirm,
        "holdout_evaluated": bool(holdout_confirm.get("evaluated", False)),
    }
    write_json(run_root / "rerank_summary.json", summary)
    return PostRerankResult(
        selected_candidate=selected_candidate,
        selected_factor_path=selected_factor_path,
        dual_gate_passed=dual_gate_passed,
        summary=summary,
    )


def materialize_selected_factor(
    selected_factor_path: Path | None,
    selected_candidate: SearchCandidate,
    mining_config: dict,
    feature_space: FeatureSpace,
    factor_path: Path,
    generated_copy_path: Path,
) -> None:
    if selected_factor_path is not None and selected_factor_path.exists():
        shutil.copyfile(selected_factor_path, factor_path)
    else:
        compile_factor_file(
            selected_candidate, mining_config, feature_space, factor_path
        )
    shutil.copyfile(factor_path, generated_copy_path)
