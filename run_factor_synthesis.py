"""
因子合成主程序
使用LASSO和PCA方法将多个因子合成为新因子，并进行性能分析

功能：
1. 基于配置文件的因子合成
2. 1年滚动窗口训练
3. 合成因子存储到analysis_results.db
4. IC/IR等性能指标计算
5. 与现有分析框架集成
"""

from factor_synthesis.qlib_adapter.panel_purged_kfold import LabelIsolationError
import os
import sys
import json
import hashlib
import logging
import random
import shutil
import subprocess
import warnings
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# [FIX] 固定 PYTHONHASHSEED 以确保 set/dict 迭代顺序可重复
# 必须在 import numpy/pandas 之前设置
if os.environ.get("PYTHONHASHSEED") is None:
    os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd

# [FIX] 固定全局随机种子以确保可重复性
GLOBAL_RANDOM_SEED = 42
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)


# DuckDB support (required)
import duckdb

from utils.dynamic_stock_pool import (
    DynamicStockPoolManager,
    StockFilter,
    apply_date_level_pool_filter_to_long_df,
)
from utils.config_loader import load_pit_config
from factor_synthesis.pipeline import (
    run_lasso_synthesis,
    run_lightgbm_synthesis,
    run_pca_synthesis,
    run_size_conditional_synthesis,
    run_weighted_sum_synthesis,
    write_size_conditional_diagnostics,
)
from factor_synthesis.pipeline.lasso_method import InsufficientSurvivingFactorsError
from factor_synthesis.pipeline.size_conditional_method import (
    load_or_build_size_conditional_snapshot,
    prepare_size_conditional_panel,
    size_conditional_snapshot_identity,
)
from factor_synthesis.research_split import apply_research_split
from factor_synthesis.weighting import get_weight_strategy
from factor_synthesis.vectorbt_adapter.benchmark_suite import (
    fixed_benchmark_cache_path,
    is_equal_rank_average_config,
    run_benchmark_suite,
    winsorized_zscore_panel,
)

# Core utilities (extracted for modularity)
from factor_synthesis.core.factor_data_loader import PITValidationError
from factor_synthesis.core import (
    cross_sectional_standardize,
    check_table_exists,
    initialize_local_tables,
    # Phase 2: Data loader functions
    load_factor_data as core_load_factor_data,
    process_raw_factor_df as core_process_raw_factor_df,
    get_pricing_data as core_get_pricing_data,
    get_open_price_data as core_get_open_price_data,
    get_complete_price_data as core_get_complete_price_data,
    get_factor_ratio_data as core_get_factor_ratio_data,
    load_market_cap_data as core_load_market_cap_data,
    # Phase 3: Pool manager functions
    apply_dynamic_stock_pool_filter as core_apply_dynamic_stock_pool_filter,
    get_all_pool_stocks as core_get_all_pool_stocks,
    compute_predict_period_dynamic_pools as core_compute_predict_period_dynamic_pools,
    compute_dynamic_pools_for_topk as core_compute_dynamic_pools_for_topk,
    apply_skip_months as core_apply_skip_months,
    parse_skip_rules as core_parse_skip_rules,
    skip_mask as core_skip_mask,
    describe_skip_rules as core_describe_skip_rules,
    # Phase 3: Factor operations functions
    LABEL_MODE_NEXT_REBALANCE_CLOSE,
    LABEL_MODE_NEXT_REBALANCE_OPEN,
    calculate_forward_returns as core_calculate_forward_returns,
    calculate_scheduled_price_returns as core_calculate_scheduled_price_returns,
    resolve_label_definition as core_resolve_label_definition,
    apply_feature_preprocessing as core_apply_feature_preprocessing,
)
from factor_synthesis.vectorbt_adapter.utils import compute_rebalance_dates
from utils.duckdb_results_writer import connect_read_only_with_retry


def validate_regime_placeholders(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed for enabled regime consumers that are not implemented yet."""
    methods = config.get("synthesis_methods")
    enabled_methods = [
        str(name)
        for name, method in (methods.items() if isinstance(methods, Mapping) else [])
        if isinstance(method, Mapping)
        and isinstance(method.get("regime_conditioning"), Mapping)
        and method["regime_conditioning"].get("enabled") is True
    ]
    top_k = (config.get("performance_evaluation") or {}).get("top_k_strategy") or {}
    risk = top_k.get("risk_overlay") or {}
    risk_conditioning = risk.get("regime_conditioning") or {}
    risk_enabled = (
        isinstance(risk_conditioning, Mapping)
        and risk_conditioning.get("enabled") is True
    )
    context = config.get("regime_context") or {}
    context_enabled = isinstance(context, Mapping) and context.get("enabled") is True

    if enabled_methods:
        raise ValueError(
            "synthesis regime_conditioning is not implemented: "
            + ", ".join(enabled_methods)
        )
    if risk_enabled:
        raise ValueError("risk overlay regime_conditioning is not implemented")
    if context_enabled:
        raise ValueError(
            "regime_context.enabled requires an enabled regime_conditioning consumer"
        )
    return config


def _bounded_prediction_window(train_end, requested_days: int, analysis_end):
    predict_start = pd.to_datetime(train_end) + timedelta(days=1)
    predict_end = min(
        predict_start + timedelta(days=requested_days), pd.to_datetime(analysis_end)
    )
    return predict_start, predict_end, max((predict_end - predict_start).days, 0)


def _build_lasso_selection_record(model_info: dict, synthesis_config: dict) -> dict:
    """Return the auditable CV selection for one rolling Lasso/EN window."""
    alpha = model_info.get("alpha")
    l1_ratio = model_info.get("l1_ratio")
    alpha_min = model_info.get("alpha_grid_min")
    alpha_max = model_info.get("alpha_grid_max")
    if alpha_min is None or alpha_max is None:
        raw_alpha_range = synthesis_config.get("lasso_alpha_range", [])
        if isinstance(raw_alpha_range, (list, tuple)) and raw_alpha_range:
            alpha_min = float(min(raw_alpha_range))
            alpha_max = float(max(raw_alpha_range))
    l1_grid = model_info.get("l1_ratio_grid") or synthesis_config.get(
        "l1_ratio_range", []
    )
    l1_grid = [float(value) for value in l1_grid]
    alpha_value = float(alpha) if alpha is not None else None
    l1_value = float(l1_ratio) if l1_ratio is not None else None
    tolerance = 1e-10
    alpha_in_grid = (
        alpha_value is not None
        and alpha_min is not None
        and alpha_max is not None
        and float(alpha_min) - tolerance <= alpha_value <= float(alpha_max) + tolerance
    )
    return {
        "train_start": model_info.get("train_start"),
        "train_end": model_info.get("train_end"),
        "validation_start": model_info.get("validation_start"),
        "validation_end": model_info.get("validation_end"),
        "alpha": alpha_value,
        "l1_ratio": l1_value,
        "cv_score": model_info.get("cv_score"),
        "cv_metric": model_info.get(
            "cv_metric", synthesis_config.get("param_search_metric")
        ),
        "cv_folds": int(
            model_info.get("cv_folds", synthesis_config.get("lasso_cv_folds", 0))
        ),
        "cv_candidate_count": int(
            model_info.get("cv_candidate_count", 0)
            or len(model_info.get("cv_results", []))
        ),
        "alpha_grid_min": alpha_min,
        "alpha_grid_max": alpha_max,
        "alpha_in_grid": alpha_in_grid,
        "alpha_invalid": alpha_value is not None and not alpha_in_grid,
        "l1_ratio_grid": l1_grid,
        "alpha_at_lower_bound": (
            alpha_value is not None
            and alpha_min is not None
            and abs(alpha_value - float(alpha_min)) <= tolerance
        ),
        "alpha_at_upper_bound": (
            alpha_value is not None
            and alpha_max is not None
            and abs(alpha_value - float(alpha_max)) <= tolerance
        ),
        "l1_at_lower_bound": (
            l1_value is not None
            and bool(l1_grid)
            and abs(l1_value - min(l1_grid)) <= tolerance
        ),
        "l1_at_upper_bound": (
            l1_value is not None
            and bool(l1_grid)
            and abs(l1_value - max(l1_grid)) <= tolerance
        ),
        "n_selected": model_info.get("n_selected"),
        "n_features": model_info.get("n_features"),
        "residual_sleeve_n_selected": model_info.get("residual_sleeve_n_selected"),
        "residual_sleeve_selected_features": model_info.get(
            "residual_sleeve_selected_features", []
        ),
        "residual_sleeve_coefficients": model_info.get(
            "residual_sleeve_coefficients", {}
        ),
        "cv_factor_count": model_info.get("cv_factor_count"),
        "factor_count_grid": model_info.get("factor_count_grid", []),
        "selected_residual_gamma": model_info.get("selected_residual_gamma"),
        "selected_blend_ratio": model_info.get("selected_blend_ratio"),
        "pooled_oof_periods": model_info.get("pooled_oof_periods"),
        "cv_dsr_min_threshold": model_info.get("cv_dsr_min_threshold"),
        "cv_fallback_reason": model_info.get("cv_fallback_reason"),
        "target_residualization_applied": model_info.get(
            "target_residualization_applied", False
        ),
        "selection_baseline_attached": model_info.get(
            "selection_baseline_attached", False
        ),
        "preselection_diagnostics": model_info.get("preselection_diagnostics", {}),
        "model_type": synthesis_config.get("model_type", model_info.get("method")),
        "sample_weight_decay": synthesis_config.get("sample_weight_decay"),
        "neutralization": synthesis_config.get("neutralization"),
        "target_period": synthesis_config.get("target_period"),
    }


def _lasso_bundle_blend_ratio(
    model_info: Optional[dict], configured_ratio: float
) -> Tuple[float, dict]:
    if not model_info:
        return float(configured_ratio), {}
    raw_blend_info = model_info.get("blend_info")
    blend_info = dict(raw_blend_info) if isinstance(raw_blend_info, dict) else {}
    ratio = blend_info.get("blend_ratio_used")
    if ratio is None:
        ratio = model_info.get("selected_blend_ratio")
    return float(configured_ratio if ratio is None else ratio), blend_info


def _materialize_parent_baseline_contract(
    synthesis_config: Mapping[str, Any], output_dir: str | Path
) -> Dict[str, str]:
    """Copy the frozen parent signal beside the deployable model bundle.

    A leaf bundle must be self-contained: keeping only the absolute research
    path would make replay depend on task-manager state and allowed the parent
    signal to disappear silently.  The copied artifact remains content-addressed
    by the hash frozen in the candidate configuration.
    """
    raw_contract = synthesis_config.get("parent_baseline")
    if not isinstance(raw_contract, Mapping):
        return {}
    source_value = str(raw_contract.get("path", "")).strip()
    expected_sha256 = str(raw_contract.get("sha256", "")).strip()
    if not source_value or not expected_sha256:
        raise ValueError("parent_baseline requires path and sha256")
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"parent baseline not found: {source}")
    actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"parent baseline hash mismatch: {source}")

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"parent_baseline_{expected_sha256[:16]}.parquet"
    if source != destination.resolve():
        if destination.exists():
            copied_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
            if copied_sha256 != expected_sha256:
                raise ValueError(
                    f"existing parent baseline hash mismatch: {destination}"
                )
        else:
            shutil.copy2(source, destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"materialized parent baseline hash mismatch: {destination}")

    contract = {
        key: str(raw_contract.get(key, ""))
        for key in ("parent_winner_id", "parent_run_id")
        if str(raw_contract.get(key, "")).strip()
    }
    contract.update({"path": destination.name, "sha256": expected_sha256})
    return contract


def _merge_replay_scores_into_parent_cache(
    cache_path: str | Path, snapshot_path: str | Path
) -> Path:
    """Merge exact VectorBT replay scores into the daily parent cache."""
    cache_path = Path(cache_path)
    snapshot_path = Path(snapshot_path)
    cache_frame = pd.read_parquet(cache_path)
    if "composite_value" not in cache_frame or not isinstance(
        cache_frame.index, pd.MultiIndex
    ):
        raise ValueError(f"invalid parent composite cache: {cache_path}")

    if snapshot_path.suffix == ".parquet":
        snapshot = pd.read_parquet(snapshot_path)
    else:
        snapshot = pd.read_csv(snapshot_path)
    required = {"factor_date", "stock_code", "composite_score"}
    if not required.issubset(snapshot.columns):
        raise ValueError(f"invalid replay score snapshot: {snapshot_path}")

    replay = snapshot.loc[:, sorted(required)].rename(
        columns={"factor_date": "date", "composite_score": "composite_value"}
    )
    replay["date"] = pd.to_datetime(replay["date"])
    replay["composite_value"] = pd.to_numeric(
        replay["composite_value"], errors="coerce"
    )
    replay = replay.set_index(["date", "stock_code"])["composite_value"]
    if replay.index.duplicated().any() or replay.isna().any():
        raise ValueError(f"invalid replay score values: {snapshot_path}")

    baseline = pd.to_numeric(cache_frame["composite_value"], errors="coerce")
    if baseline.index.duplicated().any() or baseline.isna().any():
        raise ValueError(f"invalid parent composite values: {cache_path}")
    merged = pd.concat([baseline, replay])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    temporary = cache_path.with_name(f".{cache_path.name}.tmp")
    try:
        merged.rename("composite_value").to_frame().to_parquet(temporary)
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return cache_path


def _build_lasso_bundle_model(
    *,
    train_end: Any,
    weights: Mapping[str, float],
    model_info: Optional[Mapping[str, Any]],
    synthesis_config: Mapping[str, Any],
    parent_baseline: Mapping[str, str],
) -> Optional[Dict[str, Any]]:
    """Build one replay model, retaining parent-only fallback windows."""
    configured_ratio = float(
        synthesis_config.get("blend_with_original_ratio", 0.0)
    )
    blend_weights = synthesis_config.get("blend_weights", {})
    blend_weights = dict(blend_weights) if isinstance(blend_weights, Mapping) else {}
    info = dict(model_info) if isinstance(model_info, Mapping) else None
    blend_ratio, blend_info = _lasso_bundle_blend_ratio(info, configured_ratio)
    clean_weights = {str(key): float(value) for key, value in weights.items()}
    parent_contract = dict(parent_baseline)
    if not clean_weights and not (
        blend_ratio > 0 and (blend_weights or parent_contract)
    ):
        return None

    actual_preprocess = (
        info.get("preprocess_config", synthesis_config.get("preprocess", {}))
        if info
        else synthesis_config.get("preprocess", {})
    )
    selection = (
        _build_lasso_selection_record(info, dict(synthesis_config)) if info else None
    )
    return {
        "train_end": str(train_end),
        "model_id": f"lasso_{str(train_end).replace('-', '_')}",
        "feature_names": list(clean_weights),
        "weights": clean_weights,
        "blend_only": not clean_weights,
        "standardize": synthesis_config.get("standardize", True),
        "standardization_method": actual_preprocess,
        "postprocess_cs_standardize": synthesis_config.get(
            "postprocess_cs_standardize", True
        ),
        "blend_with_original_ratio": blend_ratio,
        "blend_weights": blend_weights,
        "parent_baseline": parent_contract,
        "blend_info": blend_info,
        "selection": selection,
    }


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _number_summary(values: List[Any]) -> Dict[str, Optional[float]]:
    numbers = [number for value in values if (number := _finite_float(value)) is not None]
    if not numbers:
        return {"median": None, "min": None, "max": None}
    return {
        "median": float(np.median(numbers)),
        "min": float(np.min(numbers)),
        "max": float(np.max(numbers)),
    }


def _model_attributions(method: str, model_info: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
    raw: Any
    if method == "lightgbm":
        raw = model_info.get("feature_importance", {})
        kind = "feature_importance"
    else:
        raw = model_info.get("coefficients", {})
        if isinstance(raw, dict) and isinstance(raw.get("global_coefficients"), dict):
            raw = raw["global_coefficients"]
        kind = "coefficient"
    if not isinstance(raw, dict):
        return kind, {}
    return kind, {
        str(factor): number
        for factor, value in raw.items()
        if (number := _finite_float(value)) is not None
    }


def _build_algorithm_model_evidence(
    *,
    method: str,
    synthesis_name: str,
    model_infos: List[dict],
    diagnostics: Dict[str, Any],
    synthesis_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the compact, model-family-aware evidence used by research agents."""
    outer_folds = diagnostics.get("outer_folds", []) if isinstance(diagnostics, dict) else []
    outer_by_start = {
        str(fold.get("start")): dict(fold)
        for fold in outer_folds
        if isinstance(fold, dict) and fold.get("start")
    }
    factor_values: Dict[str, List[float]] = {}
    factor_selected: Dict[str, int] = {}
    factor_survived: Dict[str, int] = {}
    factor_missing: Dict[str, List[float]] = {}
    factor_removed: Dict[str, int] = {}
    windows: List[Dict[str, Any]] = []

    for index, raw_info in enumerate(model_infos, start=1):
        info = raw_info if isinstance(raw_info, dict) else {}
        audit = info.get("training_window_audit")
        audit = audit if isinstance(audit, dict) else {}
        missing = audit.get("missing_rate_by_factor")
        missing = missing if isinstance(missing, dict) else {}
        removed = {str(value) for value in audit.get("removed_factors", [])}
        attribution_kind, attributions = _model_attributions(method, info)
        feature_names = {str(value) for value in info.get("feature_names", [])}
        ranked = sorted(attributions.items(), key=lambda item: (-abs(item[1]), item[0]))
        ranks = {factor: rank for rank, (factor, _value) in enumerate(ranked, start=1)}
        factors = []
        factor_universe = sorted(set(attributions) | set(missing) | removed | feature_names)
        for factor in factor_universe:
            value = attributions.get(factor)
            rank = ranks.get(factor)
            selected = value is not None and abs(value) > 1e-12
            factors.append(
                {
                    "factor_id": factor,
                    "attribution": value,
                    "abs_rank": rank,
                    "selected": selected,
                    "sign": (
                        1
                        if attribution_kind == "coefficient" and value is not None and value > 0
                        else -1
                        if attribution_kind == "coefficient" and value is not None and value < 0
                        else 0
                        if attribution_kind == "coefficient" and value is not None
                        else None
                    ),
                    "missing_rate": _finite_float(missing.get(factor)),
                    "removed": factor in removed,
                }
            )
            if value is not None:
                factor_values.setdefault(factor, []).append(value)
            factor_selected[factor] = factor_selected.get(factor, 0) + int(selected)
            factor_survived[factor] = factor_survived.get(factor, 0) + int(factor not in removed)
            factor_removed[factor] = factor_removed.get(factor, 0) + int(factor in removed)
            if (missing_value := _finite_float(missing.get(factor))) is not None:
                factor_missing.setdefault(factor, []).append(missing_value)

        prediction_start = str(info.get("predict_start") or info.get("validation_start") or "")
        prediction_end = str(info.get("predict_end") or info.get("validation_end") or "")
        window_outer = outer_by_start.get(prediction_start)
        if window_outer is None and index <= len(outer_folds) and isinstance(outer_folds[index - 1], dict):
            window_outer = dict(outer_folds[index - 1])
        outer_evidence = dict(window_outer or {})
        outer_evidence.setdefault("annual_return", None)
        outer_evidence.setdefault("sharpe_ratio", None)
        outer_evidence.setdefault("max_drawdown", None)
        outer_evidence.setdefault("observations", None)
        outer_evidence.setdefault("turnover", outer_evidence.get("mean_turnover"))
        outer_evidence.setdefault("cost_rate", diagnostics.get("cost_rate"))
        outer_evidence.setdefault("slippage", diagnostics.get("slippage"))
        outer_evidence.setdefault("realized_cost", None)
        selection = (
            _build_lasso_selection_record(info, synthesis_config or {})
            if method != "lightgbm"
            else {
                "cv_score": _finite_float(info.get("cv_score")),
                "cv_metric": info.get("cv_metric"),
                "cv_candidate_count": info.get("cv_candidate_count"),
                "cv_folds": info.get("cv_folds"),
                "cv_fallback_reason": info.get("cv_fallback_reason"),
                "n_selected": info.get("n_selected"),
                "n_features": info.get("n_features"),
            }
        )
        windows.append(
            {
                "window": index,
                "dates": {
                    "train_start": info.get("train_start"),
                    "train_end": info.get("train_end"),
                    "validation_start": info.get("validation_start"),
                    "validation_end": info.get("validation_end"),
                    "prediction_start": prediction_start or None,
                    "prediction_end": prediction_end or None,
                },
                "model_family": method,
                "attribution_kind": attribution_kind,
                "factors": factors,
                "fit": {
                    "train_r2": _finite_float(info.get("train_r2", info.get("r2_score"))),
                    "validation_r2": _finite_float(info.get("validation_r2")),
                    "best_iteration": info.get("best_iteration"),
                    "objective": info.get("objective"),
                },
                "selection": selection,
                "training_audit": {
                    "survival_ratio": _finite_float(audit.get("survival_ratio")),
                    "surviving_factor_count": audit.get("surviving_factor_count"),
                    "removed_factors": sorted(removed),
                },
                "outer_fold": outer_evidence,
            }
        )

    factor_summary: Dict[str, Any] = {}
    window_count = len(windows)
    for factor in sorted(set(factor_values) | set(factor_survived)):
        values = factor_values.get(factor, [])
        nonzero = [value for value in values if abs(value) > 1e-12]
        signs = [1 if value > 0 else -1 for value in nonzero]
        factor_summary[factor] = {
            "selection_frequency": factor_selected.get(factor, 0) / window_count if window_count else 0.0,
            "window_survival_rate": factor_survived.get(factor, 0) / window_count if window_count else 0.0,
            "removed_window_count": factor_removed.get(factor, 0),
            "missing_rate": _number_summary(factor_missing.get(factor, [])),
            "attribution_median": float(np.median(values)) if values else None,
            "attribution_iqr": (
                float(np.percentile(values, 75) - np.percentile(values, 25)) if values else None
            ),
            "sign_stability": (
                max(signs.count(1), signs.count(-1)) / len(signs)
                if attribution_kind == "coefficient" and signs
                else None
            ),
            "mean_abs_drift": (
                float(np.mean(np.abs(np.diff(values)))) if len(values) > 1 else 0.0
            ),
        }

    return {
        "version": "1.0",
        "synthesis_name": synthesis_name,
        "method": method,
        "windows": windows,
        "summary": {
            "window_count": window_count,
            "factors": factor_summary,
            "train_r2": _number_summary([window["fit"]["train_r2"] for window in windows]),
            "validation_r2": _number_summary(
                [window["fit"]["validation_r2"] for window in windows]
            ),
            "cv_score": _number_summary([window["selection"]["cv_score"] for window in windows]),
            "outer": {
                "worst_outer_fold": diagnostics.get("worst_outer_fold"),
                "annual_turnover": diagnostics.get("annual_turnover"),
                "net_annual_return": diagnostics.get("net_annual_return"),
                "net_max_drawdown": diagnostics.get("net_max_drawdown"),
            },
        },
    }


def _write_algorithm_model_evidence(
    output_dir: Optional[Path], evidence: Dict[str, Any], *, syntheses=None
) -> Optional[Path]:
    syntheses = syntheses if syntheses is not None else {}
    syntheses[str(evidence["synthesis_name"])] = evidence
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "algorithm_model_evidence.json"
    path.write_text(
        json.dumps({"version": "1.0", "syntheses": syntheses}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def _lasso_outer_fold_diagnostics(
    model_infos: List[dict],
    backtests: List[dict],
) -> dict:
    """Summarize untouched rolling OOS returns by prediction window."""
    summary = next(
        (
            item
            for item in reversed(backtests)
            if isinstance(item.get("return_series_artifact"), dict)
        ),
        None,
    )
    if not summary:
        return {}
    path = Path(summary["return_series_artifact"].get("path", ""))
    if not path.exists():
        return {}
    returns = pd.read_csv(path, parse_dates=["date"]).set_index("date")["net_return"]
    windows = sorted(
        (
            pd.Timestamp(info["validation_start"]),
            pd.Timestamp(info.get("validation_end")) if info.get("validation_end") else None,
        )
        for info in model_infos
        if info.get("validation_start")
    )
    folds = []
    for index, (start, configured_end) in enumerate(windows):
        end = configured_end or (
            windows[index + 1][0] - pd.Timedelta(days=1)
            if index + 1 < len(windows)
            else returns.index.max()
        )
        fold = returns.loc[(returns.index >= start) & (returns.index <= end)].dropna()
        if fold.empty:
            continue
        curve = (1.0 + fold).cumprod()
        years = len(fold) / 252.0
        folds.append(
            {
                "fold": index + 1,
                "start": str(start.date()),
                "end": str(end.date()),
                "observations": len(fold),
                "annual_return": float(curve.iloc[-1] ** (1.0 / years) - 1.0),
                "sharpe_ratio": (
                    float(np.sqrt(252.0) * fold.mean() / fold.std())
                    if fold.std() > 0
                    else 0.0
                ),
                "max_drawdown": float((curve / curve.cummax() - 1.0).min()),
            }
        )
    return {
        "method": "lasso",
        "outer_fold_count": len(folds),
        "outer_folds": folds,
        "worst_outer_fold": (
            min(folds, key=lambda item: item["annual_return"]) if folds else None
        ),
        "net_annual_return": summary.get("annual_return"),
        "net_max_drawdown": summary.get("max_drawdown"),
        "annual_turnover": summary.get("annual_turnover"),
        "cost_rate": summary.get("cost_rate"),
        "slippage": summary.get("slippage"),
    }


# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_and_analysis.result_writer import ResultWriter
from backtest_and_analysis.alphalens_runner import AlphalensRunner

# Import monthly pool result writer
try:
    from backtest_and_analysis.monthly_pool_result_writer import MonthlyPoolResultWriter

    MONTHLY_WRITER_AVAILABLE = True
except ImportError:
    MONTHLY_WRITER_AVAILABLE = False
    print("Warning: Monthly pool result writer not available")

# Import VectorBT adapter for Top-K backtesting (fail-fast if unavailable)
from factor_synthesis.vectorbt_adapter import VectorBTBacktestRunner


def _pure_blend_fallback_factor(bundle_path: str | None) -> str | None:
    """Return the sole 100% blend factor when every bundle model is blend-only."""
    if not bundle_path:
        return None
    try:
        with open(bundle_path, encoding="utf-8") as handle:
            models = json.load(handle).get("index", {}).get("models", [])
    except (OSError, ValueError, AttributeError):
        return None
    factor_name = None
    for model in models:
        weights = model.get("blend_weights", {})
        if (
            not model.get("blend_only")
            or model.get("weights")
            or not np.isclose(float(model.get("blend_with_original_ratio", 0.0)), 1.0)
            or len(weights) != 1
        ):
            return None
        current_name, current_weight = next(iter(weights.items()))
        if not np.isclose(float(current_weight), 1.0):
            return None
        if factor_name not in (None, current_name):
            return None
        factor_name = current_name
    return factor_name if models else None


warnings.filterwarnings("ignore")


def _collect_lightgbm_index_models(model_infos: List[dict]) -> List[Dict[str, Any]]:
    """Normalize LightGBM model metadata for bundle/index generation."""
    index_models: List[Dict[str, Any]] = []
    for mi in model_infos:
        if not isinstance(mi, dict):
            continue

        model_path = mi.get("model_path")
        if not model_path:
            continue

        model_file = os.path.basename(str(mi.get("model_file") or model_path))
        mi["model_file"] = model_file

        index_models.append(
            {
                "train_start": mi.get("train_start"),
                "train_end": mi.get("train_end"),
                "predict_start": mi.get("predict_start"),
                "predict_end": mi.get("predict_end"),
                "model_file": model_file,
                "model_path": model_path,
                "n_train_samples": mi.get("n_train_samples"),
                "n_features": mi.get("n_features"),
                "train_score": mi.get("train_score"),
                "best_iteration": mi.get("best_iteration"),
                "objective": mi.get("objective"),
                "feature_names": mi.get("feature_names"),
                "blend_with_original_ratio": mi.get("blend_with_original_ratio", 0.0),
                "blend_weights": mi.get("blend_weights", {}),
                "postprocess_cs_standardize": mi.get(
                    "postprocess_cs_standardize", True
                ),
                "standardize": mi.get("standardize", True),
                "standardization_method": mi.get("standardization_method"),
            }
        )

    def _safe_date(value: Any) -> str:
        return str(value) if value is not None else ""

    return sorted(index_models, key=lambda model: _safe_date(model.get("train_end")))


def _build_lightgbm_bundle_payload(
    model_infos: List[dict],
    synthesis_name: str,
    factor_group: str,
    experiment: str,
    *,
    created_at: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Build normalized LightGBM index/bundle payloads from saved model files."""
    index_models = _collect_lightgbm_index_models(model_infos)
    if not index_models:
        return [], {}, {}

    created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    index_data = {
        "synthesis_name": synthesis_name,
        "factor_group": factor_group,
        "created_at": created_at,
        "models": index_models,
    }
    bundle_data = {
        "version": "1.0",
        "model_type": "lightgbm",
        "experiment": experiment,
        "index": index_data,
        "models": {},
    }

    missing_model_paths: List[str] = []
    for model in index_models:
        model_path = model.get("model_path")
        model_file = model.get("model_file")
        if not model_path or not os.path.exists(model_path):
            missing_model_paths.append(str(model_path))
            continue

        with open(model_path, "r", encoding="utf-8") as model_file_handle:
            bundle_data["models"][model_file] = model_file_handle.read()

    if missing_model_paths:
        raise FileNotFoundError(
            "LightGBM bundle 缺少模型文件: " + ", ".join(missing_model_paths)
        )

    return index_models, index_data, bundle_data


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _mean_or_none(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _max_or_none(values: List[float]) -> Optional[float]:
    return float(np.max(values)) if values else None


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_backtest_signature(summary: Dict[str, Any]) -> Dict[str, Any]:
    annual_returns = summary.get("annual_returns", {})
    yearly: Dict[str, float] = {}
    if isinstance(annual_returns, dict):
        for year, value in annual_returns.items():
            numeric = _safe_float(value)
            if numeric is not None:
                yearly[str(year)] = round(numeric, 6)
    fields = {
        "annual_return": summary.get("annual_return"),
        "total_return": summary.get("total_return"),
        "max_drawdown": summary.get("max_drawdown"),
        "sharpe_ratio": summary.get("sharpe_ratio"),
        "sortino_ratio": summary.get("sortino_ratio"),
        "top_pct": summary.get("top_pct"),
    }
    signature = {
        key: (
            round(numeric, 6) if (numeric := _safe_float(value)) is not None else None
        )
        for key, value in fields.items()
    }
    signature["yearly_returns"] = yearly
    return signature


def _lightgbm_synthesis_diagnostics(
    *,
    config_path: str,
    synthesis_name: str,
    composite_name: str,
    synthesis_config: Dict[str, Any],
    model_infos: List[dict],
    final_composite: pd.Series,
    vectorbt_summaries: List[dict],
) -> Dict[str, Any]:
    importance_sums: List[float] = []
    nonzero_counts: List[float] = []
    blend_ratios: List[float] = []
    dynamic_blend_count = 0
    for info in model_infos:
        if not isinstance(info, dict):
            continue
        importances = info.get("feature_importance", {})
        if isinstance(importances, dict):
            values = [_safe_float(value) or 0.0 for value in importances.values()]
            importance_sums.append(float(sum(abs(value) for value in values)))
            nonzero_counts.append(float(sum(1 for value in values if abs(value) > 0.0)))
        blend_info = info.get("blend_info", {})
        if isinstance(blend_info, dict):
            if blend_info.get("dynamic_blend_enabled"):
                dynamic_blend_count += 1
            blend_ratio = _safe_float(blend_info.get("blend_ratio_used"))
        else:
            blend_ratio = None
        if blend_ratio is None:
            blend_ratio = _safe_float(info.get("blend_with_original_ratio"))
        if blend_ratio is not None:
            blend_ratios.append(blend_ratio)

    matching_backtests = [
        item
        for item in vectorbt_summaries
        if isinstance(item, dict)
        and item.get("factor_name") == composite_name
        and "error" not in item
    ]
    selected_backtest = matching_backtests[-1] if matching_backtests else {}
    composite_values = final_composite.dropna()
    composite_std = (
        _safe_float(composite_values.std()) if not composite_values.empty else None
    )
    return {
        "synthesis_name": synthesis_name,
        "composite_factor_name": composite_name,
        "method_family": "lightgbm",
        "factor_group": synthesis_config.get("factor_group"),
        "target_period": synthesis_config.get("target_period"),
        "config_sha256": _sha256_file(config_path)
        if config_path and os.path.exists(config_path)
        else "",
        "model_count": len([info for info in model_infos if isinstance(info, dict)]),
        "nonzero_feature_count_mean": _mean_or_none(nonzero_counts),
        "feature_importance_sum_mean": _mean_or_none(importance_sums),
        "final_composite_std": composite_std,
        "dynamic_blend_enabled_count": dynamic_blend_count,
        "blend_ratio_mean": _mean_or_none(blend_ratios),
        "blend_ratio_max": _max_or_none(blend_ratios),
        "backtest_signature": _build_backtest_signature(selected_backtest)
        if selected_backtest
        else {},
    }


class FactorSynthesizer:
    """因子合成器主类"""

    def __init__(
        self,
        config_path: str,
        *,
        return_series_output_dir: str = "",
        run_id: str = "",
    ):
        """初始化因子合成器"""
        self.config_path = config_path
        self.return_series_output_dir = (
            str(Path(return_series_output_dir).expanduser().resolve())
            if return_series_output_dir
            else ""
        )
        self.run_id = run_id
        from utils.output_paths import bind_output_defaults, execution_directory
        self.execution_dir = execution_directory()
        self.config = bind_output_defaults(self._load_config(), directory=self.execution_dir)
        from utils.local_outputs import write_json
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.execution_dir / "resolved_output_paths.json", {
            "schema_version": "execution_outputs.v1",
            "source_config": str(Path(config_path).resolve()),
            "source_config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "output_settings": self.config["output_settings"],
            "cache_settings": self.config["cache_settings"],
        })
        self.logger = self._setup_logger()

        # 读取后端配置 (仅支持 DuckDB)
        db_config = self.config.get("database", {})
        self.db_backend = "duckdb"

        # 设置 DuckDB 连接
        self.logger.info("使用 DuckDB 后端")
        factor_duckdb_path = db_config.get("factor_db", {}).get("duckdb_path")
        price_duckdb_path = db_config.get("price_db", {}).get("duckdb_path")
        local_duckdb_path = db_config.get("local_db", {}).get("duckdb_path")

        # Fail-fast if DuckDB paths are incomplete
        missing_paths = []
        if not factor_duckdb_path:
            missing_paths.append("factor_db.duckdb_path")
        if not price_duckdb_path:
            missing_paths.append("price_db.duckdb_path")
        if not local_duckdb_path:
            missing_paths.append("local_db.duckdb_path")
        if missing_paths:
            raise ValueError(f"DuckDB 路径配置不完整，缺少: {', '.join(missing_paths)}")

        self.factor_duckdb_path = str(Path(factor_duckdb_path).expanduser().resolve())
        self.price_duckdb_path = str(Path(price_duckdb_path).expanduser().resolve())
        self.local_duckdb_path = str(Path(local_duckdb_path).expanduser().resolve())

        # 初始化结果写入器（支持 DuckDB）
        self.result_writer = ResultWriter(
            db_conn_str=self.local_duckdb_path,
            output_table=self.config["database"]["local_db"]["results_table"],
            db_backend=self.db_backend,
            duckdb_path=self.local_duckdb_path,
            parquet_dir=self.config.get("cache_settings", {}).get("parquet_dir"),
        )
        self.logger.info(
            "[REPAIR_DEBUG] ResultWriter initialized for local DuckDB path: %s",
            self.local_duckdb_path,
        )

        # 使用结果写入器的排他写会话初始化本地表，避免在同一进程中持有两个写连接
        with self.result_writer.duckdb_write_session() as write_session:
            initialize_local_tables(write_session.conn, self.db_backend, self.logger)

        # 同库(reader=read_only)在写入前必须全部释放，否则 DuckDB 会拒绝打开写连接。
        # factor/local 改为短生命周期 reader；price 保持 lazy reader，如与 local 同库则写前统一关闭。
        self._conn_paths = {
            "local_conn": self.local_duckdb_path,
            "factor_conn": self.factor_duckdb_path,
            "price_conn": self.price_duckdb_path,
        }
        self.local_conn = None
        self.factor_conn = None
        self.price_conn = None
        self.logger.info("DuckDB reader lifecycle:")
        self.logger.info(
            f"  local_db: {self.local_duckdb_path} (write-only via ResultWriter)"
        )
        self.logger.info(f"  factor_db: {self.factor_duckdb_path} (short-lived reader)")
        self.logger.info(f"  price_db: {self.price_duckdb_path} (lazy reader)")
        self._vectorbt_result_summaries: List[dict] = []

        # 获取表名配置
        db_config = self.config.get("database", {})
        self.factor_table = db_config.get("factor_table", "factor_values")
        self.price_table = db_config.get("price_table", "price_data")
        self.fundamental_table = db_config.get(
            "fundamental_table", "factor_data"
        )  # Used as fallback for price

        # 自动解析正确的价格表
        self.price_table = self._resolve_price_table()

        self.logger.info(
            f"使用表名配置: Factor={self.factor_table}, Price={self.price_table}, Fundamental={self.fundamental_table}"
        )

        # 解析日历过滤配置（跳过指定月份）
        cal_cfg = self.config.get("calendar_filters", {})
        if not isinstance(cal_cfg, dict):
            cal_cfg = {}

        # 规则支持整月 (1-12) 与 "MM-DD:MM-DD" 日期窗口混用，见 core/calendar_skip.py
        self.skip_months = core_parse_skip_rules(cal_cfg.get("skip_months"))
        self.skip_months_training = bool(cal_cfg.get("apply_to_training", True))
        self.skip_months_evaluation = bool(cal_cfg.get("apply_to_evaluation", True))
        self.skip_months_backtest = bool(cal_cfg.get("apply_to_backtest", True))

        if self.skip_months:
            self.logger.info(
                f"配置skip_months: {core_describe_skip_rules(self.skip_months)} "
                f"(training={self.skip_months_training}, "
                f"evaluation={self.skip_months_evaluation}, "
                f"backtest={self.skip_months_backtest})"
            )
        else:
            self.logger.info("未配置skip_months，使用全月份数据进行训练与评估")

        # ==========================================================================
        # PIT (Point-In-Time) 动态复权配置
        # 使用 factor_end 方式自动复权，无需额外 adjuster 实例
        # ==========================================================================
        self.pit_config = load_pit_config(self.config)

        if self.pit_config.get("enabled", True):
            self.logger.info(f"[PIT] 动态复权已启用 (factor_end 方式)")
        else:
            self.logger.info("[PIT] 动态复权未启用（使用原始价格）")

        self.logger.info("因子合成器初始化完成 - 多数据库连接建立")

    def _build_backtest_trial_id(
        self,
        *,
        factor_name: str,
        selection_mode: str,
        top_pct: Any,
        top_k: Any,
    ) -> str:
        payload = {
            "config_path": str(Path(self.config_path).expanduser().resolve()),
            "factor_name": factor_name,
            "selection_mode": selection_mode,
            "top_pct": top_pct,
            "top_k": top_k,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        selector = top_pct if selection_mode == "top_pct" else top_k
        raw = f"{factor_name}_{selection_mode}_{selector}_{digest}"
        safe = "".join(
            ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw
        )
        return safe.strip("._") or f"trial_{digest}"

    def _open_read_conn(self, duckdb_path: str, label: str):
        """Open a read-only DuckDB connection."""
        conn = connect_read_only_with_retry(duckdb_path)
        self.logger.debug(f"打开 {label}: {duckdb_path} (read_only)")
        return conn

    @contextmanager
    def _temporary_read_conn(self, duckdb_path: str, label: str):
        """Open a short-lived read-only DuckDB connection and close it immediately after use."""
        conn = self._open_read_conn(duckdb_path, label)
        try:
            yield conn
        finally:
            with suppress(Exception):
                conn.close()
            self.logger.debug(f"关闭 {label}: {duckdb_path}")

    def _get_price_conn(self):
        """Get or lazily create the price reader connection."""
        if self.price_conn is None:
            self.price_conn = self._open_read_conn(self.price_duckdb_path, "price_conn")
        return self.price_conn

    def _close_conn_attr(self, attr_name: str) -> bool:
        """Close a managed connection attribute if it is currently open."""
        conn = getattr(self, attr_name, None)
        if conn is None:
            return False
        try:
            conn.close()
        except Exception as exc:
            self.logger.debug(f"关闭 {attr_name} 失败: {exc}")
        setattr(self, attr_name, None)
        return True

    def _release_result_db_readers(self):
        """Release any open readers that point at the result DuckDB before acquiring a write session."""
        released = []
        for attr_name, duckdb_path in self._conn_paths.items():
            if duckdb_path != self.local_duckdb_path:
                continue
            if self._close_conn_attr(attr_name):
                released.append(attr_name)
        if released:
            self.logger.info(f"写入前释放同库 reader: {', '.join(released)}")

    def _resolve_price_table(self) -> str:
        """解析正确的价格表名（检查是否存在）"""
        configured_table = self.price_table
        price_conn = self._get_price_conn()

        # 检查配置的表是否存在
        if check_table_exists(price_conn, configured_table, self.db_backend):
            return configured_table

        # 检查备选表
        alternatives = ["price_data_daily", "price_data_5m", "price_data"]
        for alt in alternatives:
            if alt == configured_table:
                continue
            if check_table_exists(price_conn, alt, self.db_backend):
                self.logger.warning(
                    f"配置的价格表 '{configured_table}' 不存在，自动切换到 '{alt}'"
                )
                return alt

        self.logger.warning(
            f"未找到有效的价格表 (尝试了: {configured_table}, {alternatives})，将保持使用配置值 '{configured_table}'"
        )
        return configured_table

    def _check_table_exists(self, conn, table_name):
        """检查表是否存在 - 代理到核心模块"""
        return check_table_exists(conn, table_name, self.db_backend)

    def _load_config(self) -> dict:
        """加载并验证配置文件"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        from utils.data_store import resolve_database_config
        config = resolve_database_config(config)
        validate_regime_placeholders(config)

        # 基本配置验证
        required_keys = [
            "database",
            "factor_groups",
            "synthesis_methods",
            "analysis_period",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件缺少必需的键: {key}")

        # 验证因子组中的因子是否为非空列表
        for group_name, group_config in config["factor_groups"].items():
            if (
                not isinstance(group_config.get("factors", []), list)
                or not group_config["factors"]
            ):
                raise ValueError(f"因子组 {group_name} 的因子列表为空或格式错误")

        # 验证合成方法配置
        for method_name, method_config in config["synthesis_methods"].items():
            if "factor_group" not in method_config:
                raise ValueError(f"合成方法 {method_name} 缺少factor_group配置")
            if method_config["factor_group"] not in config["factor_groups"]:
                raise ValueError(
                    f"合成方法 {method_name} 引用了不存在的因子组: {method_config['factor_group']}"
                )

        return apply_research_split(config)

    def _apply_dynamic_stock_pool_filter(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """按配置可选地应用动态股票池过滤 - 代理到核心模块"""
        return core_apply_dynamic_stock_pool_filter(
            df=df,
            price_conn=self._get_price_conn(),
            config=self.config,
            fundamental_table=self.fundamental_table,
            price_table=self.price_table,
            db_backend=self.db_backend,
            start_date=start_date,
            end_date=end_date,
            custom_logger=self.logger,
        )

    def _get_all_pool_stocks(self, start_date: str, end_date: str) -> Optional[set]:
        """预计算动态股票池中所有股票的并集 - 代理到核心模块"""
        return core_get_all_pool_stocks(
            price_conn=self._get_price_conn(),
            config=self.config,
            fundamental_table=self.fundamental_table,
            price_table=self.price_table,
            db_backend=self.db_backend,
            start_date=start_date,
            end_date=end_date,
            custom_logger=self.logger,
        )

    def _apply_skip_months(self, obj):
        """按配置跳过指定月份的数据 - 代理到核心模块"""
        return core_apply_skip_months(
            obj=obj,
            skip_months=getattr(self, "skip_months", None),
            custom_logger=self.logger,
        )

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("FactorSynthesizer")
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))

        if not logger.handlers:
            handler = logging.StreamHandler()
            log_format = self.config["logging"].get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            formatter = logging.Formatter(log_format)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_local_tables(self):
        """初始化本地数据库表结构 - 代理到核心模块"""
        with self.result_writer.duckdb_write_session() as write_session:
            initialize_local_tables(write_session.conn, self.db_backend, self.logger)

    def load_factor_data(
        self,
        factor_names: List[str],
        start_date: str,
        end_date: str,
        stock_filter: Optional[set] = None,
        neutralization_config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """从数据库加载因子数据 - 代理到核心模块"""
        with self._temporary_read_conn(
            self.factor_duckdb_path, "factor_conn"
        ) as factor_conn:
            return core_load_factor_data(
                factor_conn=factor_conn,
                factor_table=self.factor_table,
                factor_names=factor_names,
                start_date=start_date,
                end_date=end_date,
                stock_filter=stock_filter,
                config=self.config,
                price_conn=self._get_price_conn(),
                price_table=self.price_table,
                neutralization_config=neutralization_config,
                custom_logger=self.logger,
            )

    def _process_raw_factor_df(
        self, df: pd.DataFrame, factor_names: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """处理原始因子DataFrame - 代理到核心模块"""
        return core_process_raw_factor_df(
            df=df,
            factor_names=factor_names,
            start_date=start_date,
            end_date=end_date,
            config=self.config,
            price_conn=self._get_price_conn(),
            price_table=self.price_table,
            custom_logger=self.logger,
        )


    def _label_rebalance_dates(self, factor_df: pd.DataFrame) -> List[pd.Timestamp]:
        """Resolve the same rebalance schedule used by the production runner."""
        dates = pd.DatetimeIndex(factor_df.index.get_level_values("date")).normalize()
        start_date = dates.min().strftime("%Y-%m-%d")
        end_date = dates.max().strftime("%Y-%m-%d")
        dynamic_pools = self._compute_dynamic_pools_for_topk(start_date, end_date)
        top_k_cfg = self.config.get("performance_evaluation", {}).get(
            "top_k_strategy", {}
        )
        holding_period = max(1, int(top_k_cfg.get("holding_period", 1)))
        if dynamic_pools:
            schedule = sorted(pd.to_datetime(list(dynamic_pools.keys())))
            return [
                pd.Timestamp(date)
                for index, date in enumerate(schedule)
                if index % holding_period == 0
            ]

        stock_pool_cfg = self.config.get("stock_pool", {})
        schedule = compute_rebalance_dates(
            pd.DatetimeIndex(dates.unique()).sort_values(),
            start_date=dates.min().to_pydatetime(),
            end_date=dates.max().to_pydatetime(),
            rebalance_freq=stock_pool_cfg.get("rebalance_schedule", "weekly"),
            nth_trading_day=int(top_k_cfg.get("nth_trading_day", 3)),
        )
        return [pd.Timestamp(date) for date in schedule]

    def _label_entry_exit_dates(
        self,
        rebalance_dates: List[pd.Timestamp],
        price_dates: pd.Index,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Resolve production holding events, including skip-window liquidation."""
        schedule = pd.DatetimeIndex(pd.to_datetime(rebalance_dates)).normalize()
        schedule = schedule.drop_duplicates().sort_values()
        if len(schedule) < 2:
            return []

        price_calendar = pd.DatetimeIndex(pd.to_datetime(price_dates)).normalize()
        price_calendar = price_calendar.drop_duplicates().sort_values()
        apply_skip = bool(getattr(self, "skip_months_backtest", True))
        skip_rules = getattr(self, "skip_months", None) if apply_skip else None
        in_skip = core_skip_mask(price_calendar, skip_rules)
        previous_in_skip = np.concatenate(([False], in_skip[:-1]))
        forced_exits = price_calendar[in_skip & ~previous_in_skip]

        pairs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for entry, next_rebalance in zip(schedule[:-1], schedule[1:]):
            if bool(core_skip_mask([entry], skip_rules)[0]):
                continue
            later_forced = forced_exits[forced_exits > entry]
            exit_date = pd.Timestamp(next_rebalance)
            if len(later_forced) and later_forced[0] < exit_date:
                exit_date = pd.Timestamp(later_forced[0])
            pairs.append((pd.Timestamp(entry), exit_date))
        return pairs

    def calculate_forward_returns(
        self, factor_df: pd.DataFrame, period: int = 5
    ) -> pd.Series:
        """Calculate the configured supervised label; legacy mode is the default."""
        label_contract = core_resolve_label_definition(
            self.config.get("label_definition")
        )
        label_mode = label_contract["mode"]
        if label_mode in {
            LABEL_MODE_NEXT_REBALANCE_OPEN,
            LABEL_MODE_NEXT_REBALANCE_CLOSE,
        }:
            rebalance_dates = self._label_rebalance_dates(factor_df)
            if len(rebalance_dates) < 2:
                raise ValueError(
                    f"{label_mode} label could not resolve two rebalance dates"
                )
            stocks = list(factor_df.index.get_level_values("stock_code").unique())
            if label_mode == LABEL_MODE_NEXT_REBALANCE_OPEN:
                prices = self._get_open_price_data(
                    stocks, rebalance_dates[0], rebalance_dates[-1], for_label=True
                )
            else:
                prices, _ = self._get_complete_price_data_for_vectorbt(
                    stocks,
                    rebalance_dates[0].strftime("%Y-%m-%d"),
                    rebalance_dates[-1].strftime("%Y-%m-%d"),
                )
            if prices is None or prices.empty:
                raise ValueError(f"{label_mode} label could not load prices")
            entry_exit_dates = self._label_entry_exit_dates(
                rebalance_dates, prices.index
            )
            if not entry_exit_dates:
                raise ValueError(
                    f"{label_mode} label could not resolve entry/exit events"
                )
            result = core_calculate_scheduled_price_returns(
                factor_df=factor_df,
                price_df=prices,
                entry_exit_dates=entry_exit_dates,
                custom_logger=self.logger,
            )
            result.attrs["label_definition"] = label_contract
            return result
        return core_calculate_forward_returns(
            factor_df=factor_df,
            price_conn=self._get_price_conn(),
            price_table=self.price_table,
            fundamental_table=self.fundamental_table,
            period=period,
            custom_logger=self.logger,
            pit_enabled=self.pit_config.get("enabled", True),
        )


    def _apply_cross_sectional_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """对索引为(date, stock_code)的特征做逐日横截面标准化 - 代理到核心模块"""
        return cross_sectional_standardize(df)

    def _load_market_cap_data(
        self, start_date: str, end_date: str, stock_filter: Optional[set] = None,
        *, trading_calendar,
    ) -> pd.Series:
        """加载市值数据用于中性化 - 代理到核心模块"""
        return core_load_market_cap_data(
            price_conn=self._get_price_conn(),
            fundamental_table=self.fundamental_table,
            start_date=start_date,
            end_date=end_date,
            stock_filter=stock_filter,
            custom_logger=self.logger,
            trading_calendar=trading_calendar,
        )


    def _apply_feature_preprocessing(
        self, df: pd.DataFrame, preprocess_cfg: Dict[str, str]
    ) -> pd.DataFrame:
        """根据配置对特征逐因子进行按日截面预处理 - 代理到核心模块"""
        return core_apply_feature_preprocessing(
            df=df,
            preprocess_cfg=preprocess_cfg,
            custom_logger=self.logger,
        )

    def direct_analyze_composite_factor(
        self, composite_series: pd.Series, factor_name: str, run_alphalens: bool = True
    ):
        """直接分析合成因子而不保存到数据库 - 内存模式

        Args:
            run_alphalens: 是否运行 alphalens/IC 分析。在 synthesis_reuse 复用回放模式下，
                IC 结果与基线一致且与 overlay 无关，可置 False 以跳过冗余计算与落库。
        """
        if composite_series.empty:
            raise RuntimeError(f"empty composite factor: {factor_name}")

        # 可选：评估阶段按月份跳过部分数据（例如策略在1、4月不交易）
        if getattr(self, "skip_months_evaluation", False) and getattr(
            self, "skip_months", None
        ):
            composite_series = self._apply_skip_months(composite_series)
            if composite_series.empty:
                raise RuntimeError(f"empty composite after calendar filter: {factor_name}")

        self.logger.info(
            f"开始直接分析合成因子: {factor_name}，数据点: {len(composite_series)}"
        )

        try:
            # 获取因子数据的股票池和时间范围
            dates = composite_series.index.get_level_values("date").unique()
            stocks = composite_series.index.get_level_values("stock_code").unique()

            self.logger.info(
                f"因子时间范围: {dates.min()} 到 {dates.max()} ({len(dates)}天)"
            )
            self.logger.info(f"因子股票数量: {len(stocks)}")

            # 因子数值质量检查
            factor_stats = composite_series.describe()
            self.logger.info(f"因子统计信息:\n{factor_stats}")

            # 检查因子值的变化程度
            factor_std = composite_series.std()
            factor_var = composite_series.var()
            unique_values = composite_series.nunique()

            self.logger.info(
                f"因子变异性检查: std={factor_std:.6f}, var={factor_var:.6f}, 唯一值数量={unique_values}"
            )

            if factor_std < 1e-10:
                self.logger.warning(
                    f"因子 {factor_name} 标准差极小 ({factor_std:.2e})，可能是常数因子，IC计算会失败"
                )

            if unique_values < 5:
                self.logger.warning(
                    f"因子 {factor_name} 唯一值过少 ({unique_values})，可能影响IC计算精度"
                )

            # 转换为DataFrame格式 (alphalens需要的格式)
            factor_df = composite_series.unstack(level="stock_code")

            # 数据质量检查（动态股票池感知）
            pe_cfg = (
                self.config.get("performance_evaluation", {})
                if isinstance(self.config.get("performance_evaluation", {}), dict)
                else {}
            )
            use_dynamic_nan = bool(pe_cfg.get("use_dynamic_pool_nan_metrics", True))
            ffill_within_membership = bool(pe_cfg.get("ffill_within_membership", False))

            if use_dynamic_nan:
                # 构造按日/按股的池成员掩码（由非NaN合成值决定）
                try:
                    membership_series = pd.Series(
                        1, index=composite_series.dropna().index
                    )
                    membership_df = membership_series.unstack("stock_code")
                    membership_df = (
                        membership_df.reindex_like(factor_df).fillna(0).astype(bool)
                    )

                    active_cells = int(membership_df.sum().sum())
                    nonmissing_cells = int(
                        (membership_df & factor_df.notna()).sum().sum()
                    )
                    nan_count_dynamic = active_cells - nonmissing_cells
                    nan_ratio_dynamic = (
                        (nan_count_dynamic / active_cells) if active_cells > 0 else 0.0
                    )

                    self.logger.info(
                        f"动态池内数据质量: 活跃单元={active_cells}, NaN={nan_count_dynamic} ({nan_ratio_dynamic:.2%})"
                    )

                    # 仅当动态池内 NaN 比例很高时告警
                    if nan_ratio_dynamic > 0.5:
                        self.logger.warning(
                            f"因子 {factor_name} 动态池内NaN比例偏高: {nan_ratio_dynamic:.2%}"
                        )

                    # 仅在池内进行前向填充（可选）
                    if ffill_within_membership:
                        tmp_ffill = factor_df.ffill()
                        factor_df_filled = tmp_ffill.where(membership_df)
                    else:
                        # 不进行跨池填充，池外保持NaN
                        factor_df_filled = factor_df.where(membership_df)

                    nan_after_fill = int(factor_df_filled.isna().sum().sum())
                    self.logger.info(f"应用动态池掩码后NaN数量: {nan_after_fill}")

                    # 日度覆盖率日志
                    try:
                        daily_active = membership_df.sum(axis=1)
                        daily_available = (membership_df & factor_df.notna()).sum(
                            axis=1
                        )
                        valid_days = daily_active > 0
                        if valid_days.any():
                            coverage_ratio = (
                                daily_available[valid_days] / daily_active[valid_days]
                            ).astype(float)
                            cov_mean = float(coverage_ratio.mean())
                            cov_p10 = float(coverage_ratio.quantile(0.10))
                            cov_p90 = float(coverage_ratio.quantile(0.90))
                            self.logger.info(
                                f"日度覆盖率（池内）: 平均={cov_mean:.2%}, P10={cov_p10:.2%}, P90={cov_p90:.2%}"
                            )
                    except Exception as e:
                        self.logger.debug(f"日度覆盖率统计失败: {e}")
                except Exception as e:
                    self.logger.warning(f"动态池NaN统计失败，回退到矩形统计: {e}")
                    # 回退：矩形统计与全局前向填充
                    total_cells = factor_df.shape[0] * factor_df.shape[1]
                    nan_count = factor_df.isna().sum().sum()
                    nan_ratio = nan_count / total_cells if total_cells > 0 else 0
                    self.logger.info(
                        f"因子数据质量: {factor_df.shape[0]}日 × {factor_df.shape[1]}股 = {total_cells}个数据点"
                    )
                    self.logger.info(f"NaN数量: {nan_count} ({nan_ratio:.2%})")
                    if nan_ratio > 0.5:
                        self.logger.warning(
                            f"因子 {factor_name} NaN比例过高: {nan_ratio:.2%}"
                        )
                    factor_df_filled = factor_df.ffill()
                    nan_after_fill = factor_df_filled.isna().sum().sum()
                    self.logger.info(f"前向填充后NaN数量: {nan_after_fill}")
            else:
                # 旧行为：对完整矩形表统计并前向填充
                total_cells = factor_df.shape[0] * factor_df.shape[1]
                nan_count = factor_df.isna().sum().sum()
                nan_ratio = nan_count / total_cells if total_cells > 0 else 0
                self.logger.info(
                    f"因子数据质量: {factor_df.shape[0]}日 × {factor_df.shape[1]}股 = {total_cells}个数据点"
                )
                self.logger.info(f"NaN数量: {nan_count} ({nan_ratio:.2%})")
                if nan_ratio > 0.5:
                    self.logger.warning(
                        f"因子 {factor_name} NaN比例过高: {nan_ratio:.2%}"
                    )
                factor_df_filled = factor_df.ffill()
                nan_after_fill = factor_df_filled.isna().sum().sum()
                self.logger.info(f"前向填充后NaN数量: {nan_after_fill}")

            # 获取价格数据
            pricing = self._get_pricing_data(stocks, dates)

            if pricing.empty:
                raise RuntimeError(f"missing pricing: {factor_name}")

            self.logger.info(f"价格数据记录数: {len(pricing)}")

            # 检查因子数据和价格数据的对齐情况
            factor_dates = set(factor_df_filled.index)
            price_dates = set(pricing.index.get_level_values("date"))
            price_assets = set(pricing.index.get_level_values("asset"))
            common_dates = factor_dates & price_dates
            common_assets = set(stocks) & price_assets

            self.logger.info(f"数据对齐检查:")
            self.logger.info(
                f"  因子日期数: {len(factor_dates)}, 价格日期数: {len(price_dates)}, 共同日期数: {len(common_dates)}"
            )
            self.logger.info(
                f"  因子股票数: {len(stocks)}, 价格股票数: {len(price_assets)}, 共同股票数: {len(common_assets)}"
            )

            date_alignment_ratio = (
                len(common_dates) / len(factor_dates) if len(factor_dates) > 0 else 0
            )
            asset_alignment_ratio = (
                len(common_assets) / len(stocks) if len(stocks) > 0 else 0
            )

            self.logger.info(
                f"  日期对齐率: {date_alignment_ratio:.2%}, 股票对齐率: {asset_alignment_ratio:.2%}"
            )

            if date_alignment_ratio < 0.5:
                self.logger.warning(
                    f"因子和价格数据日期对齐率低: {date_alignment_ratio:.2%}"
                )

            if asset_alignment_ratio < 0.8:
                self.logger.warning(
                    f"因子和价格数据股票对齐率低: {asset_alignment_ratio:.2%}"
                )

            # ==========================================
            # Top-K 策略模拟 (Qlib 回测已移除)
            # ==========================================
            pe_config = self.config.get("performance_evaluation", {})
            top_k_cfg = pe_config.get("top_k_strategy", {})
            if top_k_cfg.get("enabled", False):
                self._run_vectorbt_backtest(factor_df_filled, factor_name, pricing)
            # ==========================================

            if not run_alphalens:
                self.logger.info(
                    f"[REUSE] 跳过 alphalens/IC 分析 (复用回放模式): {factor_name}"
                )
                return

            # 运行alphalens分析
            if self.config.get("performance_evaluation", {}).get(
                "allow_high_data_loss", True
            ):
                # 动态池分析通常会产生大量的数据丢失（非池内股票），因此需要提高max_loss
                results = self._run_alphalens_analysis(
                    factor_df_filled, pricing, factor_name, max_loss=0.99
                )
            else:
                results = self._run_alphalens_analysis(
                    factor_df_filled, pricing, factor_name
                )

            # 保存分析结果到数据库
            if results is not None and isinstance(results, dict) and results:
                if self._save_analysis_results(factor_name, results):
                    self.logger.info(
                        f"合成因子 {factor_name} 分析完成，已保存IC/IR结果"
                    )
                else:
                    raise RuntimeError(f"analysis persistence failed: {factor_name}")
            else:
                raise RuntimeError(f"analysis returned no results: {factor_name}")

        except Exception as e:
            self.logger.error(f"直接分析合成因子 {factor_name} 失败: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _run_vectorbt_backtest(
        self, factor_df: pd.DataFrame, factor_name: str, pricing: pd.Series
    ) -> None:
        """
        使用VectorBT运行Top-K策略回测

        Args:
            factor_df: 因子值DataFrame (Index=date, Columns=stocks)
            factor_name: 因子名称
            pricing: 价格数据 MultiIndex Series (asset, date)
        """

        try:
            # self.logger.info(f"[VectorBT] 开始Top-K策略回测: {factor_name}")

            # 配置
            pe_config = self.config.get("performance_evaluation", {})
            top_k_cfg = pe_config.get("top_k_strategy", {})
            cal_cfg = self.config.get("calendar_filters", {})
            cal_cfg_for_backtest = dict(cal_cfg) if isinstance(cal_cfg, dict) else {}
            if not getattr(self, "skip_months_backtest", True):
                cal_cfg_for_backtest["skip_months"] = []

            # [CLOUD-ALIGNED] 使用 bundle_path 加载模型 (与云端对齐)
            bundle_path = None
            factor_db_path = None

            synthesis_methods = self.config.get("synthesis_methods", {})
            if synthesis_methods:
                # 使用第一个合成方法的配置
                method_name = list(synthesis_methods.keys())[0]
                method_cfg = synthesis_methods[method_name]
                synthesis_method = method_cfg.get("method", "lasso")

                # [CLOUD-ALIGNED] 对于 Lasso/LightGBM/weighted_sum 方法，使用 bundle 文件
                if synthesis_method == "lasso":
                    output_dir = self.config.get("output_settings", {}).get(
                        "weights_output_dir", "./results"
                    )
                    bundle_path = os.path.join(
                        output_dir, f"lasso_model_bundle_{method_name}.json"
                    )

                    if os.path.exists(bundle_path):
                        self.logger.info(
                            f"[VectorBT] 使用云端对齐管线 (Lasso): {bundle_path}"
                        )
                    else:
                        self.logger.warning(
                            f"[VectorBT] Bundle 文件未找到: {bundle_path}，跳过回测"
                        )
                        return

                    # 获取因子数据库路径
                    db_cfg = self.config.get("database", {}).get("factor_db", {})
                    factor_db_path = db_cfg.get("duckdb_path")
                elif synthesis_method == "lightgbm":
                    output_dir = self.config.get("output_settings", {}).get(
                        "model_output_dir", "./results/lightgbm_models"
                    )
                    bundle_path = os.path.join(
                        output_dir, f"lgb_model_bundle_{method_name}.json"
                    )

                    if os.path.exists(bundle_path):
                        self.logger.info(
                            f"[VectorBT] 使用云端对齐管线 (LightGBM): {bundle_path}"
                        )
                    else:
                        self.logger.warning(
                            f"[VectorBT] Bundle 文件未找到: {bundle_path}，跳过回测"
                        )
                        return

                    # 获取因子数据库路径
                    db_cfg = self.config.get("database", {}).get("factor_db", {})
                    factor_db_path = db_cfg.get("duckdb_path")
                elif synthesis_method == "weighted_sum":
                    # [FIX] weighted_sum 也使用 bundle 文件进行 VectorBT 回测
                    output_dir = self.config.get("output_settings", {}).get(
                        "weights_output_dir", "./results"
                    )
                    bundle_path = os.path.join(
                        output_dir, f"weighted_sum_bundle_{method_name}.json"
                    )

                    if os.path.exists(bundle_path):
                        self.logger.info(
                            f"[VectorBT] 使用云端对齐管线 (weighted_sum): {bundle_path}"
                        )
                    else:
                        self.logger.warning(
                            f"[VectorBT] Bundle 文件未找到: {bundle_path}，跳过回测"
                        )
                        return

                    # 获取因子数据库路径
                    db_cfg = self.config.get("database", {}).get("factor_db", {})
                    factor_db_path = db_cfg.get("duckdb_path")
                elif synthesis_method == "size_conditional_elasticnet":
                    # Scores are already strictly outer-OOS; reuse the existing
                    # VectorBT input-factor source instead of recomputing them.
                    bundle_path = None
                    factor_db_path = None
                else:
                    self.logger.warning(
                        f"[VectorBT] 不支持的合成方法 ({synthesis_method})，跳过回测"
                    )
                    return

            if factor_db_path:
                factor_db_path = str(Path(factor_db_path).expanduser().resolve())
            else:
                factor_db_path = self.factor_duckdb_path

            # [ALIGN] 传递 price_db 连接用于 T-1 交易日查询 (与 JQ Local 对齐)
            price_db_path = self.price_duckdb_path
            price_conn = self._get_price_conn()

            # [ALIGNMENT DATA] Check if alignment data should be saved
            output_cfg = self.config.get("output_settings", {})
            save_alignment_data = output_cfg.get("save_alignment_data", False)
            alignment_output_format = output_cfg.get(
                "alignment_output_format", "parquet"
            )

            # Derive experiment name from synthesis method or config
            experiment_name = output_cfg.get(
                "composite_factor_prefix", "unnamed_experiment"
            )
            if synthesis_methods:
                method_name = list(synthesis_methods.keys())[0]
                experiment_name = method_name

            # [JQ-ALIGNED V5] 获取完整价格数据（包括 skip months），不使用已过滤的 pricing 参数
            # VectorBT 需要 skip month 日期来正确追踪资产价值和执行清仓
            start_date = self.config["analysis_period"]["start_date"]
            end_date = self.config["analysis_period"]["end_date"]

            # 获取所有股票代码（从原始 pricing 参数提取，但后续会重新获取完整价格）
            try:
                all_stocks = pricing.index.get_level_values("asset").unique().tolist()
            except Exception:
                all_stocks = (
                    list(factor_df.columns) if hasattr(factor_df, "columns") else []
                )

            # 直接从数据库获取完整价格数据（不过滤 skip months）
            price_wide, factor_ratio_df = self._get_complete_price_data_for_vectorbt(
                all_stocks, start_date, end_date
            )

            if price_wide is None or price_wide.empty:
                self.logger.warning(
                    "[VectorBT] 无法获取完整价格数据，回退到已过滤的价格数据"
                )
                # 回退逻辑
                try:
                    price_wide = pricing.unstack(level="asset")
                except Exception:
                    price_reset = pricing.reset_index()
                    price_reset.columns = ["asset", "date", "close"]
                    price_wide = price_wide.pivot(
                        index="date", columns="asset", values="close"
                    )
                price_wide.index = pd.to_datetime(price_wide.index)
                price_wide = price_wide.sort_index()
                factor_ratio_df = None  # 回退时无法获取除权因子

            # self.logger.info(f"[VectorBT] 收盘价数据: {price_wide.shape[0]} 日期, {price_wide.shape[1]} 股票")

            # [JoinQuant 对齐] 获取开盘价数据用于订单执行（同样不过滤 skip months）
            open_wide = self._get_open_price_data(
                list(price_wide.columns),
                start_date,  # 使用完整日期范围
                end_date,
            )
            if open_wide is not None:
                # self.logger.info(f"[VectorBT] 开盘价数据: {open_wide.shape[0]} 日期, {open_wide.shape[1]} 股票")
                pass
            else:
                self.logger.warning("[VectorBT] 无法获取开盘价，将使用收盘价执行订单")

            # 确保因子数据格式正确 (Wide: Index=date, Columns=stocks)
            if isinstance(factor_df.index, pd.MultiIndex):
                # 是长表，需要转换
                factor_wide = factor_df.unstack(level="stock_code")
                if isinstance(factor_wide.columns, pd.MultiIndex):
                    factor_wide.columns = factor_wide.columns.get_level_values(-1)
            else:
                factor_wide = factor_df

            factor_wide.index = pd.to_datetime(factor_wide.index)
            factor_wide = factor_wide.sort_index()
            pure_blend_parent_panel = None
            pure_blend_factor = _pure_blend_fallback_factor(bundle_path)
            if pure_blend_factor:
                raw_parent = self.load_factor_data(
                    [pure_blend_factor],
                    start_date,
                    end_date,
                    stock_filter=set(price_wide.columns),
                )
                pure_blend_parent_panel = winsorized_zscore_panel(
                    self._wide_factor_panel(raw_parent, pure_blend_factor)
                )
                if pure_blend_parent_panel.empty:
                    self.logger.warning(
                        "[VectorBT] 纯基准回退缺少 %s 数据，停止回测",
                        pure_blend_factor,
                    )
                    return
                factor_wide = pure_blend_parent_panel

            # self.logger.info(f"[VectorBT] 因子数据: {factor_wide.shape[0]} 日期, {factor_wide.shape[1]} 股票")

            # 计算动态股票池 - 使用配置的日期范围，与 _get_all_pool_stocks 保持一致
            pool_start_date = self.config["analysis_period"]["start_date"]
            pool_end_date = self.config["analysis_period"]["end_date"]
            dynamic_pools = self._compute_dynamic_pools_for_topk(
                pool_start_date, pool_end_date
            )

            # self.logger.info(f"[VectorBT] 动态池: {len(dynamic_pools)} 个调仓日")

            # [MULTI-TOP-PCT] 支持多个 top_pct 值，逐一运行回测
            top_pct_raw = top_k_cfg.get("top_pct", None)
            if top_pct_raw is None:
                top_pct_list: list[float | None] = [None]  # will use top_k or default
            elif isinstance(top_pct_raw, list):
                top_pct_list = top_pct_raw
            else:
                top_pct_list = [top_pct_raw]

            benchmark_cfg = pe_config.get("benchmark_suite", {})
            reuse_scores = len(top_pct_list) > 1 or benchmark_cfg.get(
                "enabled", False
            )
            factor_load_cache: dict | None = {} if reuse_scores else None
            synthesis_score_cache: dict | None = {} if reuse_scores else None

            for pct in top_pct_list:
                # 为每个 top_pct 值创建独立的 runner_config
                iter_top_k_cfg = dict(top_k_cfg)
                if (
                    synthesis_method == "size_conditional_elasticnet"
                    or pure_blend_parent_panel is not None
                ):
                    iter_top_k_cfg["score_source"] = "input_factor"
                iter_top_k_cfg["top_pct"] = pct  # override to single float (or None)
                iter_runner_config = {
                    "top_k_strategy": iter_top_k_cfg,
                    "calendar_filters": cal_cfg_for_backtest,
                }

                selection_mode = "top_pct" if pct is not None else "top_k"
                top_k_value = (
                    None if pct is not None else iter_top_k_cfg.get("top_k", 5)
                )
                trial_id = self._build_backtest_trial_id(
                    factor_name=factor_name,
                    selection_mode=selection_mode,
                    top_pct=pct,
                    top_k=top_k_value,
                )
                pct_label = (
                    f"top_pct={pct}" if pct is not None else f"top_k={top_k_value}"
                )
                self.logger.info(f"[VectorBT] 运行回测: {pct_label}")

                runner = None
                try:
                    runner = VectorBTBacktestRunner(
                        iter_runner_config,
                        logger=self.logger,
                        bundle_path=bundle_path,
                        factor_db_path=factor_db_path,
                        factor_db_conn=None,
                        price_db_path=price_db_path,
                        price_db_conn=price_conn,
                        save_alignment_data=save_alignment_data,
                        alignment_output_dir=output_cfg.get(
                            "alignment_output_dir", "outputs/synthesis/alignment"
                        ),
                        experiment_name=experiment_name,
                        alignment_output_format=alignment_output_format,
                        return_series_output_dir=self.return_series_output_dir,
                        return_series_trial_id=trial_id
                        if self.return_series_output_dir
                        else None,
                        return_series_metadata={
                            "run_id": self.run_id,
                            "selected_config_path": str(
                                Path(self.config_path).expanduser().resolve()
                            ),
                        },
                        diagnostics_output_dir=self.config.get(
                            "output_settings", {}
                        ).get("diagnostics_output_dir"),
                        factor_load_cache=factor_load_cache,
                        synthesis_score_cache=synthesis_score_cache,
                    )

                    # 运行回测 - 传入开盘价以对齐JoinQuant执行方式
                    results = runner.run_backtest(
                        factor_df=factor_wide,
                        price_df=price_wide,
                        open_df=open_wide,  # JoinQuant对齐: 开盘价执行
                        factor_ratio_df=factor_ratio_df,  # Evidence only; prices already include PIT adjustment.
                        dynamic_pools=dynamic_pools,
                        rebalance_freq="weekly",
                    )
                finally:
                    if runner is not None:
                        with suppress(Exception):
                            runner.close()

                if (
                    "error" not in results
                    and save_alignment_data
                    and synthesis_method == "lasso"
                ):
                    snapshot_suffix = (
                        "parquet" if alignment_output_format == "parquet" else "csv"
                    )
                    snapshot_path = (
                        Path(
                            output_cfg.get(
                                "alignment_output_dir", "outputs/synthesis/alignment"
                            )
                        )
                        / experiment_name
                        / f"factor_values_snapshot.{snapshot_suffix}"
                    )
                    _merge_replay_scores_into_parent_cache(
                        self._composite_cache_path(method_name), snapshot_path
                    )

                candidate_gross_result = None
                if (
                    "error" not in results
                    and benchmark_cfg.get("enabled", False)
                    and not getattr(self, "_p2_benchmark_done", False)
                ):
                    gross_top_k_cfg = dict(iter_top_k_cfg)
                    gross_top_k_cfg["cost_rate"] = 0.0
                    gross_top_k_cfg["slippage"] = 0.0
                    gross_runner = VectorBTBacktestRunner(
                        {
                            "top_k_strategy": gross_top_k_cfg,
                            "calendar_filters": cal_cfg_for_backtest,
                        },
                        logger=self.logger,
                        bundle_path=bundle_path,
                        factor_db_path=factor_db_path,
                        price_db_path=price_db_path,
                        price_db_conn=price_conn,
                        return_series_metadata={
                            "run_id": self.run_id,
                            "selected_config_path": str(
                                Path(self.config_path).expanduser().resolve()
                            ),
                        },
                        factor_load_cache=factor_load_cache,
                        synthesis_score_cache=synthesis_score_cache,
                    )
                    try:
                        candidate_gross_result = gross_runner.run_backtest(
                            factor_df=factor_wide,
                            price_df=price_wide,
                            open_df=open_wide,
                            factor_ratio_df=factor_ratio_df,
                            dynamic_pools=dynamic_pools,
                            rebalance_freq="weekly",
                        )
                    finally:
                        gross_runner.close()

                # 输出结果
                if "error" not in results:
                    summary = {
                        "factor_name": factor_name,
                        "trial_id": trial_id,
                        "selection_mode": selection_mode,
                        "top_k": top_k_value,
                        "top_pct": pct,
                        "annual_return": results.get("annual_return"),
                        "sharpe_ratio": results.get("sharpe_ratio"),
                        "calmar_ratio": results.get("calmar_ratio"),
                        "sortino_ratio": results.get("sortino_ratio"),
                        "total_return": results.get("total_return"),
                        "max_drawdown": results.get("max_drawdown"),
                        "trade_count": results.get("trade_count"),
                        "annual_returns": results.get("annual_returns", {}),
                        "return_series_artifact": results.get("return_series_artifact"),
                        # [PORTFOLIO OVERLAY] turnover/exposure diagnostics + overlay echo
                        "annual_turnover": results.get("annual_turnover"),
                        "cost_rate": iter_top_k_cfg.get("cost_rate", 0.003),
                        "slippage": iter_top_k_cfg.get("slippage", 0.0),
                        "avg_holding_count": results.get("avg_holding_count"),
                        "max_holding_count": results.get("max_holding_count"),
                        "holding_count_violation_days": results.get(
                            "holding_count_violation_days"
                        ),
                        "max_excess_holding_notional": results.get(
                            "max_excess_holding_notional"
                        ),
                        "holding_count_violation_dates": results.get(
                            "holding_count_violation_dates"
                        ),
                        "avg_exposure": results.get("avg_exposure"),
                        "overlay": results.get("overlay"),
                    }
                    self._vectorbt_result_summaries.append(summary)
                    if benchmark_cfg.get("enabled", False) and not getattr(
                        self, "_p2_benchmark_done", False
                    ):
                        benchmark_report = self._run_p2_benchmark_suite(
                            base_config=iter_runner_config,
                            price_wide=price_wide,
                            open_wide=open_wide,
                            factor_ratio_df=factor_ratio_df,
                            dynamic_pools=dynamic_pools,
                            candidate_result=results,
                            candidate_gross_result=candidate_gross_result,
                            factor1_name=benchmark_cfg.get(
                                "factor1", "Factor1_wind_micro"
                            ),
                            rank_average_factors=benchmark_cfg.get(
                                "rank_average_factors", []
                            ),
                            quality_filter_factors=benchmark_cfg.get(
                                "quality_filter_factors", []
                            ),
                            random_seeds=int(benchmark_cfg.get("random_seeds", 100)),
                            factor1_panel_override=pure_blend_parent_panel,
                        )
                        benchmark_path = self._write_p2_benchmark_report(
                            benchmark_report
                        )
                        summary["benchmark_report_path"] = benchmark_path
                        summary["benchmark_cache_hit"] = benchmark_report.get(
                            "fixed_benchmark_cache", {}
                        ).get("hit")
                        self._p2_benchmark_done = True
                        self.logger.info(
                            "ALGORITHM_BENCHMARK_REPORT_PATH: %s",
                            benchmark_path,
                        )
                    result_str = runner.format_results(results)
                    self.logger.info(result_str)
                else:
                    self._vectorbt_result_summaries.append(
                        {
                            "factor_name": factor_name,
                            "trial_id": trial_id,
                            "selection_mode": selection_mode,
                            "top_k": top_k_value,
                            "top_pct": pct,
                            "error": results.get("error"),
                        }
                    )
                    self.logger.warning(
                        f"[VectorBT] 回测失败 ({pct_label}): {results.get('error')}"
                    )

        except PITValidationError:
            # Invalid PIT prices must fail the synthesis run, not skip evaluation.
            raise
        except Exception as e:
            self.logger.error(f"[VectorBT] Top-K回测异常: {e}")
            import traceback

            traceback.print_exc()

    @staticmethod
    def _wide_factor_panel(frame: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """Convert one loaded long factor column into the VectorBT score matrix."""
        if factor_name not in frame.columns:
            return pd.DataFrame()
        series = frame[factor_name]
        if not isinstance(series.index, pd.MultiIndex):
            return pd.DataFrame(series)
        return series.unstack(level="stock_code").sort_index()

    def _run_p2_benchmark_suite(
        self,
        *,
        base_config: Dict[str, Any],
        price_wide: pd.DataFrame,
        open_wide: Optional[pd.DataFrame],
        factor_ratio_df: Optional[pd.DataFrame],
        dynamic_pools: Dict[str, list[str]],
        candidate_result: Dict[str, Any],
        candidate_gross_result: Optional[Dict[str, Any]],
        factor1_name: str,
        rank_average_factors: List[str],
        quality_filter_factors: List[str],
        random_seeds: int,
        factor1_panel_override: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        start_date = self.config["analysis_period"]["start_date"]
        end_date = self.config["analysis_period"]["end_date"]
        stock_filter = set(price_wide.columns)
        requested = [factor1_name, *rank_average_factors, *quality_filter_factors]
        requested = list(dict.fromkeys(name for name in requested if name))
        raw = self.load_factor_data(
            requested, start_date, end_date, stock_filter=stock_filter
        )
        factor1_panel = factor1_panel_override
        if factor1_panel is None:
            factor1_panel = self._wide_factor_panel(raw, factor1_name)
            if not factor1_panel.empty:
                factor1_panel = winsorized_zscore_panel(factor1_panel)

        rank_average_score = None
        if rank_average_factors:
            panels = [
                self._wide_factor_panel(raw, name) for name in rank_average_factors
            ]
            panels = [panel for panel in panels if not panel.empty]
            if panels:
                ranked = [panel.rank(axis=1, pct=True) for panel in panels]
                rank_average_score = sum(ranked) / len(ranked)

        quality_score = None
        if quality_filter_factors:
            panels = [
                self._wide_factor_panel(raw, name) for name in quality_filter_factors
            ]
            panels = [panel for panel in panels if not panel.empty]
            if panels:
                # Keep only names above the cross-sectional median quality score;
                # all-equal weighting then creates a genuine quality-filtered EW.
                mean_rank = sum(panel.rank(axis=1, pct=True) for panel in panels) / len(
                    panels
                )
                quality_score = mean_rank.where(mean_rank >= 0.5)

        methods = self.config.get("synthesis_methods", {})
        method_config = next(iter(methods.values()), {}) if len(methods) == 1 else {}
        factor_group = method_config.get("factor_group")
        candidate_factors = (
            self.config.get("factor_groups", {})
            .get(factor_group, {})
            .get("factors", [])
            if factor_group
            else []
        )
        candidate_is_rank_average = is_equal_rank_average_config(
            method_config, list(candidate_factors), rank_average_factors
        )

        cache_path = None
        benchmark_config = (
            self.config.get("performance_evaluation", {})
            .get("benchmark_suite", {})
        )
        parent_benchmark = str(
            benchmark_config.get("parent_benchmark", "factor1_top")
        )
        selection_rows = (
            candidate_result.get("selection_diagnostics", {}).get("rows", [])
            if isinstance(candidate_result, dict)
            else []
        )
        candidate_rebalance_dates = [
            pd.Timestamp(row["date"]).normalize()
            for row in selection_rows
            if isinstance(row, dict) and row.get("date")
        ]
        if (
            benchmark_config.get("cache_fixed_benchmarks", True)
        ):
            cache_dir = (
                benchmark_config.get(
                    "fixed_cache_dir", "outputs/synthesis/benchmark_cache"
                )
            )
            cache_path = fixed_benchmark_cache_path(
                cache_dir=cache_dir,
                base_config=base_config,
                price_df=price_wide,
                open_df=open_wide,
                factor_ratio_df=factor_ratio_df,
                dynamic_pools=dynamic_pools,
                factor1_panel=factor1_panel,
                rank_average_score=rank_average_score,
                quality_score=quality_score,
                candidate_is_rank_average=candidate_is_rank_average,
                random_seeds=random_seeds,
                rebalance_dates=candidate_rebalance_dates,
            )

        report = run_benchmark_suite(
            base_config=base_config,
            price_df=price_wide,
            open_df=open_wide,
            factor_ratio_df=factor_ratio_df,
            dynamic_pools=dynamic_pools,
            factor1_panel=factor1_panel,
            rank_average_score=rank_average_score,
            quality_score=quality_score,
            candidate_result=candidate_result,
            candidate_gross_result=candidate_gross_result,
            candidate_is_rank_average=candidate_is_rank_average,
            random_seeds=random_seeds,
            fixed_cache_path=cache_path,
            parent_benchmark=parent_benchmark,
        )
        report["assumptions"]["pit"] = dict(getattr(self, "pit_config", {}) or {})
        report["assumptions"]["tradability"] = {
            "enable_tday_filter": bool(
                base_config.get("top_k_strategy", {}).get("enable_tday_filter", True)
            ),
            "pool_filters": list(self.config.get("stock_pool", {}).get("filters", [])),
        }
        report["assumptions"]["cost_and_slippage"] = {
            "cost_rate": base_config.get("top_k_strategy", {}).get("cost_rate", 0.003),
            "slippage": base_config.get("top_k_strategy", {}).get("slippage", 0.0),
        }
        return report

    def _write_p2_benchmark_report(self, report: Dict[str, Any]) -> Optional[str]:
        if hasattr(self, "_execution_metrics"):
            self._execution_metrics["benchmark_reports"].append(report)
        output_dir = self.config.get("output_settings", {}).get(
            "diagnostics_output_dir"
        )
        if not output_dir:
            return None
        path = Path(output_dir).expanduser() / "benchmark_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return str(path)

    def _compute_predict_period_dynamic_pools(
        self, train_end, predict_window_days: int, method_name: str = "synthesis"
    ) -> Optional[Dict[str, set]]:
        """计算预测期的动态股票池 - 代理到核心模块"""
        return core_compute_predict_period_dynamic_pools(
            price_conn=self._get_price_conn(),
            config=self.config,
            fundamental_table=self.fundamental_table,
            price_table=self.price_table,
            db_backend=self.db_backend,
            train_end=train_end,
            predict_window_days=predict_window_days,
            method_name=method_name,
            custom_logger=self.logger,
        )

    def _compute_dynamic_pools_for_topk(self, start_date: str, end_date: str) -> dict:
        """为Top-K策略评估计算动态股票池 - 代理到核心模块"""
        return core_compute_dynamic_pools_for_topk(
            price_conn=self._get_price_conn(),
            config=self.config,
            fundamental_table=self.fundamental_table,
            price_table=self.price_table,
            db_backend=self.db_backend,
            start_date=start_date,
            end_date=end_date,
            custom_logger=self.logger,
        )

    def _get_pricing_data(self, stocks: list, dates: pd.Index) -> pd.Series:
        """获取价格数据用于alphalens分析 - 代理到核心模块"""
        return core_get_pricing_data(
            price_conn=self._get_price_conn(),
            price_table=self.price_table,
            stocks=stocks,
            dates=dates,
            skip_months=getattr(self, "skip_months", None),
            skip_months_evaluation=getattr(self, "skip_months_evaluation", False),
            custom_logger=self.logger,
            pit_enabled=self.pit_config.get("enabled", True)
            if hasattr(self, "pit_config")
            else True,
            future_periods=max(self.config['performance_evaluation'].get('ic_periods') or (1, 5, 10)) + 1,
        )

    def _get_open_price_data(
        self, stocks: list, start_date, end_date, *, for_label: bool = False
    ) -> Optional[pd.DataFrame]:
        """获取开盘价数据用于VectorBT回测 - 代理到核心模块"""
        return core_get_open_price_data(
            price_conn=self._get_price_conn(),
            price_table=self.price_table,
            stocks=stocks,
            start_date=start_date,
            end_date=end_date,
            custom_logger=self.logger,
            pit_enabled=self.pit_config.get("enabled", True)
            if hasattr(self, "pit_config")
            else True,
            for_label=for_label,
        )

    def _get_complete_price_data_for_vectorbt(
        self, stocks: list, start_date: str, end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """获取完整收盘价数据和除权因子比率用于VectorBT

        Returns:
            Tuple[price_df, factor_ratio_df]:
                - price_df: Wide DataFrame (Index=date, Columns=stocks) with close prices
                - factor_ratio_df: Wide DataFrame with factor ratios for corporate action adjustment
        """
        price_df = core_get_complete_price_data(
            price_conn=self._get_price_conn(),
            price_table=self.price_table,
            stocks=stocks,
            start_date=start_date,
            end_date=end_date,
            custom_logger=self.logger,
            pit_enabled=self.pit_config.get("enabled", True)
            if hasattr(self, "pit_config")
            else True,
        )

        # 获取除权因子比率数据
        factor_ratio_df = core_get_factor_ratio_data(
            price_conn=self._get_price_conn(),
            price_table=self.price_table,
            stocks=stocks,
            start_date=start_date,
            end_date=end_date,
            custom_logger=self.logger,
        )

        return price_df, factor_ratio_df

    def _run_alphalens_analysis(
        self,
        factor_df: pd.DataFrame,
        pricing: pd.Series,
        factor_name: str,
        max_loss: float = 0.35,
    ) -> dict:
        """运行alphalens分析"""
        try:
            runner = AlphalensRunner(
                n_quantiles=self.config["performance_evaluation"].get("quantiles", 10),
                periods=tuple(
                    self.config["performance_evaluation"].get("ic_periods")
                    or (1, 5, 10)
                ),
            )

            # 准备前瞻收益率
            forward_returns = None
            if self.config.get("use_precomputed_returns", False):
                from backtest_and_analysis.forward_returns_calculator import (
                    ForwardReturnsCalculator,
                )

                returns_calculator = ForwardReturnsCalculator(cache_dir="./cache")

                periods = tuple(self.config["performance_evaluation"]["ic_periods"])
                stock_list = list(factor_df.columns)
                start_date = factor_df.index.min().strftime("%Y-%m-%d")
                end_date = factor_df.index.max().strftime("%Y-%m-%d")

                forward_returns = returns_calculator.calculate_or_load_forward_returns(
                    pricing, stock_list, start_date, end_date, periods
                )

            # 运行分析
            results = runner.run(factor_df, pricing, forward_returns, max_loss=max_loss)
            return results

        except Exception as e:
            self.logger.error(f"alphalens分析失败: {e}")
            return None

    def _safe_extract_scalar_value(self, data, metric_name: str) -> float:
        """
        安全地从pandas对象中提取标量值

        Args:
            data: 可能是DataFrame、Series或标量的数据
            metric_name: 指标名称，用于日志

        Returns:
            提取的标量值，如果无法提取则返回NaN
        """
        try:
            # 如果已经是标量
            if np.isscalar(data) or isinstance(data, (int, float)):
                return float(data) if not pd.isna(data) else np.nan

            # 如果是DataFrame
            if isinstance(data, pd.DataFrame):
                # 对于IR DataFrame，取'ir'列的第一个值
                if "ir" in data.columns:
                    ir_series = data["ir"]
                    if not ir_series.empty:
                        return (
                            float(ir_series.iloc[0])
                            if not pd.isna(ir_series.iloc[0])
                            else np.nan
                        )

                # 对于IC DataFrame，计算均值
                if data.shape[1] > 0:
                    mean_val = data.mean().mean()  # 先对每列求均值，再对所有列求均值
                    return float(mean_val) if not pd.isna(mean_val) else np.nan

            # 如果是Series
            elif isinstance(data, pd.Series):
                if not data.empty:
                    # 对于IC Series，计算均值
                    if metric_name.lower() == "ic":
                        mean_val = data.mean()
                        return float(mean_val) if not pd.isna(mean_val) else np.nan
                    # 对于其他Series，取第一个值
                    else:
                        return (
                            float(data.iloc[0]) if not pd.isna(data.iloc[0]) else np.nan
                        )

            self.logger.warning(f"无法从 {type(data)} 类型中提取 {metric_name} 标量值")
            return np.nan

        except Exception as e:
            self.logger.warning(f"提取 {metric_name} 标量值时出错: {e}")
            return np.nan

    def _clear_existing_results(self, factor_name: str):
        """清空指定因子的所有历史结果（确保全量覆盖）"""
        try:
            table_name = self.config["database"]["local_db"]["results_table"]
            self.logger.info(
                f"正在清空因子 {factor_name} 的所有历史分析结果 (表: {table_name})..."
            )

            self.logger.info(
                "[REPAIR_DEBUG] Clearing existing results through ResultWriter write session for %s",
                factor_name,
            )
            with self.result_writer.duckdb_write_session() as session:
                result = session.conn.execute(
                    f"DELETE FROM {table_name} WHERE factor_name = ?",
                    [factor_name],
                )
                # DuckDB 不直接返回 rowcount，使用 fetchone 或估计值
                rows = 0
                try:
                    # 尝试获取变更行数（DuckDB 某些版本支持）
                    rows = result.fetchone()[0] if result else 0
                except (TypeError, IndexError, AttributeError):
                    rows = -1  # 无法获取

            self.logger.info(f"已删除 {rows} 条历史记录 (全量清除)")

        except Exception as e:
            self.logger.error(f"清空历史结果失败: {e}")

    def _save_analysis_results(self, factor_name: str, results: dict) -> bool:
        """保存IC/IR分析结果到数据库"""
        try:
            # 详细的结果结构分析
            self.logger.info(f"=== 因子 {factor_name} 分析结果详情 ===")
            self.logger.info(f"结果字段: {list(results.keys())}")

            for key, value in results.items():
                if hasattr(value, "shape"):
                    self.logger.info(
                        f"  {key}: {type(value).__name__} 形状={value.shape}"
                    )
                else:
                    self.logger.info(f"  {key}: {type(value).__name__}")

            self._release_result_db_readers()

            # 1. 首先全量清除旧数据 (用户要求)
            self._clear_existing_results(factor_name)

            # 2. 保存新结果到数据库
            with self.result_writer.duckdb_write_session() as session:
                self.result_writer.write(factor_name, results, duckdb_conn=session.conn)
            self.logger.info(f"✅ 因子 {factor_name} 结果已成功写入数据库")

            # 写入月度池级别结果
            if MONTHLY_WRITER_AVAILABLE:
                try:
                    monthly_writer = MonthlyPoolResultWriter(
                        self.local_duckdb_path,
                        "monthly_pool_results",
                        duckdb_path=self.local_duckdb_path,
                    )

                    # 从结果数据中推断股票数量
                    stock_count = 0
                    ic_data = results.get("ic")
                    if ic_data is not None and hasattr(ic_data, "shape"):
                        # 从IC数据推断股票数量（每日IC计算的股票数量）
                        stock_count = ic_data.shape[0] if ic_data.shape[0] > 0 else 0

                    # 准备月度写入器需要的数据格式
                    monthly_results = {
                        "ic": results.get("ic"),
                        "quantile_returns": results.get(
                            "quantile_returns", pd.DataFrame()
                        ),
                        "pool_type": "synthesis",  # 标识为合成因子
                        "factor_statistics": {"stock_count": stock_count},
                    }

                    monthly_writer.write(factor_name, monthly_results)
                    self.logger.info(
                        f"✅ 因子 {factor_name} 的月度池级别结果已写入数据库"
                    )

                except Exception as e:
                    self.logger.error(f"❌ 写入月度池级别结果失败: {e}")
                    # 不抛出异常，避免整个流程中断

            # 详细的IC分析
            if "ic" in results:
                ic_data = results["ic"]
                self._analyze_and_log_ic_data(factor_name, ic_data)
            else:
                self.logger.warning(f"⚠️ 因子 {factor_name} 缺少IC数据")

            # 详细的IR分析
            if "ir" in results:
                ir_data = results["ir"]
                self._analyze_and_log_ir_data(factor_name, ir_data)
            else:
                self.logger.warning(f"⚠️ 因子 {factor_name} 缺少IR数据")

            # 分位数收益分析
            if "quantile_returns" in results and not results["quantile_returns"].empty:
                qr_data = results["quantile_returns"]
                self._analyze_and_log_quantile_returns(factor_name, qr_data)
            else:
                self.logger.warning(f"⚠️ 因子 {factor_name} 缺少分位数收益数据")

            overall_ic_debug = (
                self._safe_extract_scalar_value(results.get("ic"), "ic")
                if "ic" in results
                else None
            )
            overall_ir_debug = (
                self._safe_extract_scalar_value(results.get("ir"), "ir")
                if "ir" in results
                else None
            )
            latest_backtest = None
            if getattr(self, "_vectorbt_result_summaries", None):
                for item in reversed(self._vectorbt_result_summaries):
                    if (
                        isinstance(item, dict)
                        and item.get("factor_name") == factor_name
                        and "error" not in item
                    ):
                        latest_backtest = item
                        break
            quality_debug = {
                "factor_name": factor_name,
                "has_ic": "ic" in results and results.get("ic") is not None,
                "has_ir": "ir" in results and results.get("ir") is not None,
                "has_quantile_returns": "quantile_returns" in results
                and results.get("quantile_returns") is not None,
                "backtest_summary_present": latest_backtest is not None,
            }
            if overall_ic_debug is not None and not pd.isna(overall_ic_debug):
                quality_debug["overall_ic"] = float(overall_ic_debug)
            if overall_ir_debug is not None and not pd.isna(overall_ir_debug):
                quality_debug["overall_ir"] = float(overall_ir_debug)
            if latest_backtest is not None:
                quality_debug["backtest_summary"] = {
                    key: latest_backtest.get(key)
                    for key in (
                        "selection_mode",
                        "top_pct",
                        "top_k",
                        "annual_return",
                        "sharpe_ratio",
                        "calmar_ratio",
                        "sortino_ratio",
                        "total_return",
                        "max_drawdown",
                        "trade_count",
                    )
                }
            self.logger.info(
                "[REPAIR_DEBUG] quality_metrics_snapshot=%s",
                json.dumps(quality_debug, ensure_ascii=False, sort_keys=True),
            )

            if hasattr(self, "_execution_metrics"):
                summary = {"factor_name": factor_name, "periods": {}, "ir_periods": {}}
                for key in ("overall_ic", "overall_ir"):
                    if key in quality_debug:
                        summary[key] = quality_debug[key]
                ic = results.get("ic")
                if isinstance(ic, pd.DataFrame):
                    for period in ic.columns:
                        values = ic[period].dropna()
                        if not values.empty:
                            summary["periods"][str(period)] = {
                                "mean_ic": float(values.mean()), "std_ic": float(values.std()),
                                "min_ic": float(values.min()), "max_ic": float(values.max()),
                                "positive_ratio": float((values > 0).mean()), "sample_count": len(values),
                            }
                ir = results.get("ir")
                if isinstance(ir, pd.DataFrame) and "ir" in ir:
                    summary["ir_periods"] = {str(k): float(v) for k, v in ir["ir"].dropna().items()}
                self._execution_metrics["ic_summaries"].append(summary)
            self.logger.info(f"=== 因子 {factor_name} 分析完成 ===")
            return True

        except Exception as e:
            self.logger.error(f"❌ 保存分析结果失败: {e}")
            import traceback

            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 不抛出异常，避免整个流程中断
            return False

    def _analyze_and_log_ic_data(self, factor_name: str, ic_data):
        """详细分析和记录IC数据"""
        try:
            if isinstance(ic_data, pd.DataFrame):
                # 多期间IC分析
                self.logger.info(f"📊 {factor_name} IC数据分析:")
                for period in ic_data.columns:
                    period_ic = ic_data[period].dropna()
                    if not period_ic.empty:
                        ic_mean = period_ic.mean()
                        ic_std = period_ic.std()
                        ic_min = period_ic.min()
                        ic_max = period_ic.max()
                        positive_ratio = (period_ic > 0).mean()

                        self.logger.info(
                            f"  {period}: 均值={ic_mean:.4f}, 标准差={ic_std:.4f}, 范围=[{ic_min:.4f}, {ic_max:.4f}], 正值占比={positive_ratio:.2%}, 样本数={len(period_ic)}"
                        )

                        if abs(ic_mean) > 0.02:
                            self.logger.info(f"  ✅ {period} IC表现良好 (|IC| > 0.02)")
                        elif abs(ic_mean) > 0.01:
                            self.logger.info(f"  🔶 {period} IC表现一般 (|IC| > 0.01)")
                        else:
                            self.logger.warning(
                                f"  ⚠️ {period} IC表现较弱 (|IC| ≤ 0.01)"
                            )
                    else:
                        self.logger.warning(f"  ❌ {period}: 无有效IC数据")

                # 总体IC评估
                overall_ic = self._safe_extract_scalar_value(ic_data, "ic")
                if not pd.isna(overall_ic):
                    self.logger.info(f"🎯 {factor_name} 整体IC: {overall_ic:.4f}")
                else:
                    self.logger.warning(f"⚠️ {factor_name} 整体IC: 无法计算")

            else:
                # 单一IC值
                ic_scalar = self._safe_extract_scalar_value(ic_data, "ic")
                if not pd.isna(ic_scalar):
                    self.logger.info(f"🎯 {factor_name} IC: {ic_scalar:.4f}")
                else:
                    self.logger.warning(f"⚠️ {factor_name} IC: 无法计算")

        except Exception as e:
            self.logger.error(f"IC数据分析失败: {e}")

    def _analyze_and_log_ir_data(self, factor_name: str, ir_data):
        """详细分析和记录IR数据"""
        try:
            if isinstance(ir_data, pd.DataFrame):
                # 多期间IR分析
                self.logger.info(f"📈 {factor_name} IR数据分析:")
                for index, row in ir_data.iterrows():
                    if "ir" in ir_data.columns:
                        ir_value = row["ir"]
                        if not pd.isna(ir_value):
                            self.logger.info(f"  {index}: IR={ir_value:.4f}")

                            if abs(ir_value) > 0.5:
                                self.logger.info(
                                    f"  ✅ {index} IR表现优秀 (|IR| > 0.5)"
                                )
                            elif abs(ir_value) > 0.2:
                                self.logger.info(
                                    f"  🔶 {index} IR表现良好 (|IR| > 0.2)"
                                )
                            else:
                                self.logger.warning(
                                    f"  ⚠️ {index} IR表现一般 (|IR| ≤ 0.2)"
                                )
                        else:
                            self.logger.warning(f"  ❌ {index}: IR为NaN")

                # 总体IR评估
                overall_ir = self._safe_extract_scalar_value(ir_data, "ir")
                if not pd.isna(overall_ir):
                    self.logger.info(f"🎯 {factor_name} 整体IR: {overall_ir:.4f}")
                else:
                    self.logger.warning(f"⚠️ {factor_name} 整体IR: 无法计算")

            else:
                # 单一IR值
                ir_scalar = self._safe_extract_scalar_value(ir_data, "ir")
                if not pd.isna(ir_scalar):
                    self.logger.info(f"🎯 {factor_name} IR: {ir_scalar:.4f}")
                else:
                    self.logger.warning(f"⚠️ {factor_name} IR: 无法计算")

        except Exception as e:
            self.logger.error(f"IR数据分析失败: {e}")

    def _analyze_and_log_quantile_returns(self, factor_name: str, qr_data):
        """详细分析和记录分位数收益数据"""
        try:
            self.logger.info(f"📋 {factor_name} 分位数收益分析:")
            self.logger.info(f"  数据形状: {qr_data.shape}")

            if hasattr(qr_data.columns, "levels"):
                # MultiIndex columns
                periods = qr_data.columns.get_level_values(0).unique()
                for period in periods:
                    period_data = qr_data[period]
                    if not period_data.empty:
                        mean_returns = period_data.mean()
                        self.logger.info(
                            f"  {period} 平均收益: {mean_returns.to_dict()}"
                        )

                        # 检查单调性
                        if len(mean_returns) > 1:
                            is_monotonic = (
                                mean_returns.is_monotonic_increasing
                                or mean_returns.is_monotonic_decreasing
                            )
                            if is_monotonic:
                                self.logger.info(f"  ✅ {period} 收益具有单调性")
                            else:
                                self.logger.warning(f"  ⚠️ {period} 收益缺乏单调性")

        except Exception as e:
            self.logger.error(f"分位数收益分析失败: {e}")

    def _composite_cache_path(
        self, synthesis_name: str, base_dir: str | None = None
    ) -> Path:
        """Path of the cached composite factor parquet (synthesis-reuse replay)."""
        output_dir = base_dir or self.config.get("output_settings", {}).get(
            "weights_output_dir", "./results"
        )
        return Path(output_dir) / f"composite_cache_{synthesis_name}.parquet"

    def _save_composite_cache(self, synthesis_name: str, composite: pd.Series) -> None:
        """Persist the synthesized composite factor so overlay replays can skip the Lasso fit."""
        try:
            path = self._composite_cache_path(synthesis_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = composite.rename("composite_value").to_frame()
            frame.to_parquet(path)
            self.logger.info(f"[REUSE] 已缓存合成因子: {path} (rows={len(frame)})")
        except Exception as e:
            self.logger.warning(f"[REUSE] 合成因子缓存失败: {e}")

    def _maybe_reuse_synthesis(
        self, synthesis_name: str, synthesis_config: dict, method: str
    ) -> bool:
        """If synthesis_reuse is enabled and prior artifacts exist, skip the (expensive)
        rolling Lasso fit and replay the backtest from the cached bundle + composite.

        Returns True when the reuse path handled the synthesis (caller should return early).
        """
        reuse_cfg = self.config.get("synthesis_reuse", {})
        if not isinstance(reuse_cfg, dict) or not reuse_cfg.get("enabled", False):
            return False
        if method != "lasso":
            self.logger.warning(
                f"[REUSE] synthesis_reuse 仅支持 lasso，当前 method={method}，回退到完整合成"
            )
            return False
        output_dir = self.config.get("output_settings", {}).get(
            "weights_output_dir", "./results"
        )
        source_dir = Path(reuse_cfg.get("source_dir") or output_dir)
        bundle_path = source_dir / f"lasso_model_bundle_{synthesis_name}.json"
        cache_path = self._composite_cache_path(
            synthesis_name, base_dir=str(source_dir)
        )
        if not bundle_path.exists() or not cache_path.exists():
            self.logger.warning(
                f"[REUSE] 缺少可复用产物 (bundle={bundle_path.exists()}, cache={cache_path.exists()})，"
                f"回退到完整合成 (source_dir={source_dir})"
            )
            return False
        try:
            # The backtest path loads the bundle from this run's weights_output_dir, so make
            # sure it is present there (no-op when source_dir == weights_output_dir).
            dest_dir = Path(output_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_bundle = dest_dir / bundle_path.name
            if bundle_path.resolve() != dest_bundle.resolve():
                shutil.copy2(bundle_path, dest_bundle)
            frame = pd.read_parquet(cache_path)
            composite = frame.iloc[:, 0]
            composite.name = None
            try:
                pool_start = self.config["analysis_period"]["start_date"]
                pool_end = self.config["analysis_period"]["end_date"]
                dynamic_pools = self._compute_dynamic_pools_for_topk(
                    pool_start, pool_end
                )
                if dynamic_pools:
                    pool_index = pd.MultiIndex.from_tuples(
                        [
                            (pd.to_datetime(date), stock)
                            for date, stocks in dynamic_pools.items()
                            for stock in (stocks or [])
                        ],
                        names=["date", "stock_code"],
                    )
                    expanded_index = composite.index.union(pool_index)
                    added = len(expanded_index) - len(composite.index)
                    if added > 0:
                        composite = composite.reindex(expanded_index)
                        self.logger.info(
                            f"[REUSE] 扩展复用cache universe: +{added} 个动态池索引点，"
                            f"stocks={len(expanded_index.get_level_values('stock_code').unique())}"
                        )
            except Exception as e:
                self.logger.warning(
                    f"[REUSE] 扩展动态池universe失败，继续使用原cache: {e}"
                )
            composite_name = f"{self.config['output_settings']['composite_factor_prefix']}_{synthesis_name}"
            self.logger.info(
                f"[REUSE] 复用已有合成 (跳过 Lasso 训练): bundle={dest_bundle}, "
                f"cache={cache_path}, rows={len(composite)}"
            )
            self.direct_analyze_composite_factor(
                composite, composite_name, run_alphalens=False
            )
            return True
        except Exception as e:
            self.logger.error(f"[REUSE] 复用合成失败，回退到完整合成: {e}")
            import traceback

            traceback.print_exc()
            return False

    def rolling_window_synthesis(self, synthesis_name: str, synthesis_config: dict):
        """
        滚动窗口因子合成

        Args:
            synthesis_name: 合成方法名称
            synthesis_config: 合成配置
        """
        self.logger.info(f"开始滚动窗口合成: {synthesis_name}")

        # 获取因子组
        factor_group = synthesis_config["factor_group"]
        training_factor_names = list(
            self.config["factor_groups"][factor_group]["factors"]
        )
        method = synthesis_config["method"]
        label_definition = self.config.get("label_definition")
        effective_label = core_resolve_label_definition(label_definition)
        label_mode = effective_label["mode"]
        if label_mode in {
            LABEL_MODE_NEXT_REBALANCE_OPEN,
            LABEL_MODE_NEXT_REBALANCE_CLOSE,
        }:
            top_k_cfg = self.config.get("performance_evaluation", {}).get(
                "top_k_strategy", {}
            )
            holding_period = max(1, int(top_k_cfg.get("holding_period", 1)))
            cadence = self.config.get("stock_pool", {}).get(
                "rebalance_schedule", "weekly"
            )
            cadence_days = {"daily": 1, "weekly": 5, "monthly": 23, "quarterly": 66}
            embargo_days = int(
                effective_label.get("embargo_trading_days")
                or cadence_days.get(str(cadence).strip().lower(), 5) * holding_period
            )
            synthesis_config["_label_embargo_days"] = embargo_days
            synthesis_config["_label_inner_embargo_observations"] = 1
            self.logger.info(
                "[LABEL] 有效合同=%s，outer embargo=%d个交易日",
                effective_label,
                embargo_days,
            )

        # [REUSE] synthesis_reuse 复用回放：跳过昂贵的滚动 Lasso 训练，直接用已有
        # bundle + 缓存的合成因子重放回测（overlay/cost 仅影响组合构建，不影响合成）。
        if label_mode not in {
            LABEL_MODE_NEXT_REBALANCE_OPEN,
            LABEL_MODE_NEXT_REBALANCE_CLOSE,
        } and self._maybe_reuse_synthesis(synthesis_name, synthesis_config, method):
            self.logger.info(
                f"[REUSE] 合成复用完成，跳过完整滚动合成: {synthesis_name}"
            )
            return

        # [Fix] Inject global data_quality config if not present in method config
        # This ensures max_factor_missing_rate from root config is respected
        if "data_quality" not in synthesis_config and "data_quality" in self.config:
            synthesis_config["data_quality"] = self.config["data_quality"]
            self.logger.info(
                f"应用全局 data_quality 配置: {self.config['data_quality']}"
            )

        # [Fix] Inject global distribution_prefilter config if not present in method config
        # This enables skewness/kurtosis-based preprocessing classification
        if (
            "distribution_prefilter" not in synthesis_config
            and "distribution_prefilter" in self.config
        ):
            synthesis_config["distribution_prefilter"] = self.config[
                "distribution_prefilter"
            ]
            self.logger.info(
                f"应用全局 distribution_prefilter 配置: enabled={self.config['distribution_prefilter'].get('enabled', False)}"
            )

        # [关键修复] 确保 blend_weights 中的因子也被加载（与 factors 列表取并集）
        # 但 LASSO 训练只使用 training_factor_names，blend 因子仅用于混合步骤
        blend_weights = synthesis_config.get("blend_weights", {})
        all_factor_names = list(training_factor_names)  # 复制一份用于加载

        if blend_weights:
            blend_factor_names = set(blend_weights.keys())
            existing_factors = set(training_factor_names)
            blend_only_factors = blend_factor_names - existing_factors

            if blend_only_factors:
                self.logger.info(
                    "[因子加载] blend_weights 包含 %d 个仅用于混合的因子（不参与LASSO训练）: %s",
                    len(blend_only_factors),
                    list(blend_only_factors),
                )
                all_factor_names.extend(list(blend_only_factors))
            else:
                self.logger.info(
                    "[因子加载] blend_weights 中的 %d 个因子均已在训练因子列表中",
                    len(blend_factor_names),
                )

        # 将训练因子列表传递给 synthesis_config，供 lasso_method 使用
        synthesis_config["_training_factor_names"] = training_factor_names
        self.logger.info(
            "[因子加载] 训练因子: %d 个, 总加载因子: %d 个 (包含 %d 个仅用于混合)",
            len(training_factor_names),
            len(all_factor_names),
            len(all_factor_names) - len(training_factor_names),
        )

        # 获取时间范围
        # start_date / end_date 定义合成结果（预测）的输出范围
        start_date = self.config["analysis_period"]["start_date"]
        end_date = self.config["analysis_period"]["end_date"]

        # [自动扩展] 向前扩展 train_window_days + buffer 以满足第一个训练窗口的数据需求
        # start_date 是合成输出的起始日期，第一个训练窗口为 [start_date - train_window_days, start_date]
        train_window_days = self.config["rolling_window"]["train_window_days"]
        _buffer_days = 30  # buffer 用于 shift(1) 和交易日历对齐
        data_start_date = (
            pd.to_datetime(start_date)
            - timedelta(days=train_window_days + _buffer_days)
        ).strftime("%Y-%m-%d")
        self.logger.info(
            f"[日期范围] 合成结果范围: {start_date} ~ {end_date}, "
            f"数据读取范围自动扩展为: {data_start_date} ~ {end_date} "
            f"(向前扩展 {train_window_days}+{_buffer_days} 天以满足训练窗口需求)"
        )

        all_pool_stocks = None

        def _load_filtered_factor_panel():
            nonlocal all_pool_stocks
            if all_pool_stocks is None and self.config.get("stock_pool", {}).get(
                "use_dynamic_pools", False
            ):
                all_pool_stocks = self._get_all_pool_stocks(data_start_date, end_date)
            frame = self.load_factor_data(
                all_factor_names,
                data_start_date,
                end_date,
                stock_filter=all_pool_stocks,
                neutralization_config=synthesis_config.get("neutralization"),
            )
            frame = self._apply_dynamic_stock_pool_filter(
                frame, start_date=data_start_date, end_date=end_date
            ).sort_index()
            if getattr(self, "skip_months_training", False) and getattr(
                self, "skip_months", None
            ):
                frame = self._apply_skip_months(frame)
            return frame

        prepared_panel = None
        snapshot_info = None
        forward_returns = pd.Series(dtype=float)
        log_market_cap = pd.Series(dtype=float)
        if method == "size_conditional_elasticnet":
            size_cfg = synthesis_config.get("size_conditioning", {})
            prepared_cache_cfg = size_cfg.get("prepared_cache", {})
            if self.config.get("stock_pool", {}).get("use_dynamic_pools", False):
                all_pool_stocks = self._get_all_pool_stocks(data_start_date, end_date)
            factor_hash = hashlib.md5(
                "_".join(sorted(all_factor_names)).encode()
            ).hexdigest()[:8]
            stock_hash = hashlib.md5(
                str(sorted(all_pool_stocks) if all_pool_stocks else "ALL").encode()
            ).hexdigest()[:8]
            raw_cache_root = Path(
                self.config.get("cache_settings", {}).get(
                    "cache_dir", "./cache/factor_data"
                )
            )
            raw_factor_cache = raw_cache_root / (
                f"factors_{factor_hash}_{data_start_date}_{end_date}_{stock_hash}.parquet"
            )
            price_source = (
                self.config.get("database", {}).get("price_db", {}).get("duckdb_path")
            )
            source_paths = [str(raw_factor_cache)]
            if price_source:
                source_paths.append(price_source)
            from utils.trading_calendar import load_trading_calendar
            market_cap_calendar = load_trading_calendar(self._get_price_conn())
            market_cap_sessions = market_cap_calendar.sessions(data_start_date, end_date)
            market_cap_alignment = {
                "contract": "market-cap-calendar-shift1-v1",
                "calendar_id": market_cap_calendar.calendar_id,
                "sessions_sha256": hashlib.sha256(json.dumps(
                    market_cap_sessions.strftime("%Y-%m-%d").tolist(),
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
                "start": pd.Timestamp(data_start_date).normalize().strftime("%Y-%m-%d"),
                "end": pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d"),
            }
            # Runtime metadata from a previous window/run cannot identify this producer.
            snapshot_config = dict(synthesis_config)
            for key in ("_label_exit_dates", "_label_trading_calendar", "_label_effective_period", "_label_calendar_source"):
                snapshot_config.pop(key, None)
            snapshot_config["_effective_label_definition"] = dict(effective_label)
            snapshot_identity = size_conditional_snapshot_identity(
                self.config,
                snapshot_config,
                all_factor_names,
                data_start_date,
                end_date,
                source_paths,
                source_markers={"market_cap_alignment": market_cap_alignment},
            )

            def _build_prepared_panel():
                frame = _load_filtered_factor_panel()
                if frame.empty:
                    raise ValueError(f"无因子数据可用于 {synthesis_name}")
                returns = self.calculate_forward_returns(
                    frame, synthesis_config.get("target_period", 5)
                )
                market_cap = self._load_market_cap_data(
                    data_start_date,
                    end_date,
                    stock_filter=all_pool_stocks,
                    trading_calendar=market_cap_calendar,
                )
                return prepare_size_conditional_panel(
                    self, frame, returns, market_cap, snapshot_config
                )

            cache_root = Path(
                prepared_cache_cfg.get("cache_root")
                or self.config.get("cache_settings", {}).get(
                    "cache_dir", "./cache/factor_data"
                )
            )
            cache_enabled = bool(prepared_cache_cfg.get("enabled", True))
            if not raw_factor_cache.exists():
                self.logger.info(
                    "[SIZE_PREPARED_CACHE] raw factor cache absent; prepared cache starts next run: %s",
                    raw_factor_cache,
                )
                cache_enabled = False
            prepared_panel, snapshot_info = load_or_build_size_conditional_snapshot(
                cache_root,
                snapshot_identity,
                _build_prepared_panel,
                enabled=cache_enabled,
                logger=self.logger,
            )
            factor_df = prepared_panel["predict_factors"]
            forward_returns = prepared_panel["raw_returns"]
            synthesis_config.update(prepared_panel["label_context"])
            synthesis_config["_prepared_snapshot"] = snapshot_info
            self.logger.info(
                "[SIZE_PREPARED_CACHE] status=%s key=%s",
                snapshot_info.get("status"),
                snapshot_info.get("key"),
            )
        else:
            factor_df = _load_filtered_factor_panel()

        # weighted_sum 不需要滚动窗口与训练，直接在全样本上做线性合成
        if method == "weighted_sum":
            self.logger.info(f"使用 weighted_sum 固定权重合成: {synthesis_name}")
            run_weighted_sum_synthesis(
                self, synthesis_name, synthesis_config, factor_df
            )
            return

        # 若为LASSO，则收集每个窗口的系数以便最终导出CSV
        lasso_weight_rows = [] if method == "lasso" else None

        if factor_df.empty:
            raise RuntimeError(f"no factor data for {synthesis_name}")

        # 加载前瞻收益率（监督学习方法需要）
        if method in ["lasso", "lightgbm", "pca"]:
            target_period = synthesis_config.get("target_period", 5)
            forward_returns = self.calculate_forward_returns(factor_df, target_period)
            if forward_returns.attrs.get("label_exit_dates"):
                synthesis_config["_label_exit_dates"] = dict(
                    forward_returns.attrs["label_exit_dates"]
                )
                synthesis_config["_effective_label_definition"] = dict(effective_label)

        # 滚动窗口参数
        train_window_days = self.config["rolling_window"]["train_window_days"]
        step_size_days = self.config["rolling_window"]["step_size_days"]

        # 滚动窗口合成
        all_composite_factors = []
        model_infos = []
        lasso_selection_records: List[Dict[str, Any]] = []
        training_window_audits: List[Dict[str, Any]] = []
        diagnostics_output_dir = (
            Path(
                self.config.get("output_settings", {}).get("diagnostics_output_dir", "")
            ).expanduser()
            if self.config.get("output_settings", {}).get("diagnostics_output_dir")
            else None
        )

        def _write_training_window_audits() -> None:
            if diagnostics_output_dir is None:
                return
            diagnostics_output_dir.mkdir(parents=True, exist_ok=True)
            (diagnostics_output_dir / "training_window_factor_audit.json").write_text(
                json.dumps(
                    {
                        "synthesis_name": synthesis_name,
                        "method": method,
                        "windows": training_window_audits,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        # 为 LightGBM 准备模型输出目录（用于后续在云端加载）
        model_output_dir = None
        if method == "lightgbm":
            output_settings = self.config.get("output_settings", {})
            base_model_dir = output_settings.get(
                "model_output_dir", "./results/lightgbm_models"
            )
            try:
                os.makedirs(base_model_dir, exist_ok=True)
            except Exception as e:
                self.logger.error(f"创建 LightGBM 模型输出目录失败: {e}")
            model_output_dir = base_model_dir

        # [自动扩展] start_date 定义合成输出的起始日期，第一个训练窗口结束于 start_date
        # 因此 current_date 直接从 start_date 开始（而非 start_date + train_window_days）
        current_date = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        self.logger.info(
            f"滚动窗口参数: 训练窗口={train_window_days}天, "
            f"预测窗口={self.config['rolling_window']['predict_window_days']}天, "
            f"步长={step_size_days}天, "
            f"首个训练期: {(current_date - timedelta(days=train_window_days)).strftime('%Y-%m-%d')} ~ {current_date.strftime('%Y-%m-%d')}"
        )

        while current_date <= end_date_dt:
            # 计算训练窗口
            train_start = current_date - timedelta(days=train_window_days)
            train_end = current_date

            train_start_str = train_start.strftime("%Y-%m-%d")
            train_end_str = train_end.strftime("%Y-%m-%d")

            # 计算对应的预测期
            predict_start, predict_end, bounded_predict_window_days = (
                _bounded_prediction_window(
                    train_end,
                    self.config["rolling_window"]["predict_window_days"],
                    end_date_dt,
                )
            )

            self.logger.info(
                f"处理窗口: 训练期 {train_start_str} ~ {train_end_str}, 预测期 {predict_start.strftime('%Y-%m-%d')} ~ {predict_end.strftime('%Y-%m-%d')}"
            )

            # 执行合成
            if method == "lasso":
                # 计算预测期的动态股票池（支持Blending）
                predict_window_days = bounded_predict_window_days
                daily_pools_for_predict = self._compute_predict_period_dynamic_pools(
                    train_end, predict_window_days, method_name="Lasso"
                )

                try:
                    composite_result, model_info = run_lasso_synthesis(
                        self,
                        factor_df,
                        forward_returns,
                        train_start_str,
                        train_end_str,
                        synthesis_config,
                        daily_pools=daily_pools_for_predict,
                    )
                except InsufficientSurvivingFactorsError as exc:
                    training_window_audits.append(
                        {
                            "train_start": train_start_str,
                            "train_end": train_end_str,
                            "status": "failed_survival_gate",
                            **exc.audit,
                        }
                    )
                    _write_training_window_audits()
                    raise

                if isinstance(model_info, dict):
                    model_info.setdefault("train_start", train_start_str)
                    model_info.setdefault("train_end", train_end_str)
                    model_info.setdefault("predict_start", predict_start.strftime("%Y-%m-%d"))
                    model_info.setdefault(
                        "predict_end",
                        model_info.get("validation_end")
                        or predict_end.strftime("%Y-%m-%d"),
                    )
                if not composite_result.empty:
                    all_composite_factors.append(composite_result)
                    model_infos.append(model_info)
                if isinstance(model_info, dict) and (
                    model_info.get("alpha") is not None
                    or model_info.get("validation_start") is not None
                ):
                    lasso_selection_records.append(
                        _build_lasso_selection_record(model_info, synthesis_config)
                    )
                if isinstance(model_info, dict) and model_info.get(
                    "training_window_audit"
                ):
                    training_window_audits.append(
                        {
                            "train_start": train_start_str,
                            "train_end": train_end_str,
                            "status": "passed",
                            **model_info["training_window_audit"],
                        }
                    )

                # 收集本训练窗口的LASSO权重（以验证起始日为行名）
                if (
                    lasso_weight_rows is not None
                    and isinstance(model_info, dict)
                    and model_info.get("coefficients")
                ):
                    try:
                        validation_start = model_info.get("validation_start")
                        coeffs = model_info.get("coefficients", {})

                        weight_series = pd.Series(coeffs, name=validation_start)

                        lasso_weight_rows.append(weight_series)
                    except Exception as e:
                        self.logger.warning(f"记录LASSO权重失败: {e}")

            elif method == "pca":
                composite_results, model_info = run_pca_synthesis(
                    self,
                    factor_df,
                    forward_returns,
                    train_start_str,
                    train_end_str,
                    synthesis_config,
                )

                if composite_results:
                    all_composite_factors.extend(composite_results)
                    model_infos.append(model_info)

            elif method == "lightgbm":
                # 计算预测期的动态股票池（与本地回测脚本对齐）
                predict_window_days = bounded_predict_window_days
                daily_pools_for_predict = self._compute_predict_period_dynamic_pools(
                    train_end, predict_window_days, method_name="LightGBM"
                )

                try:
                    composite_result, model_info = run_lightgbm_synthesis(
                        self,
                        factor_df,
                        forward_returns,
                        train_start_str,
                        train_end_str,
                        synthesis_config,
                        output_dir=model_output_dir,
                        daily_pools=daily_pools_for_predict,
                    )
                except InsufficientSurvivingFactorsError as exc:
                    training_window_audits.append(
                        {
                            "train_start": train_start_str,
                            "train_end": train_end_str,
                            "status": "failed_survival_gate",
                            **exc.audit,
                        }
                    )
                    _write_training_window_audits()
                    raise

                if isinstance(model_info, dict):
                    model_info.setdefault("train_start", train_start_str)
                    model_info.setdefault("train_end", train_end_str)
                    model_info.setdefault("predict_start", predict_start.strftime("%Y-%m-%d"))
                    model_info.setdefault(
                        "predict_end",
                        model_info.get("validation_end")
                        or predict_end.strftime("%Y-%m-%d"),
                    )
                if not composite_result.empty:
                    all_composite_factors.append(composite_result)
                    model_infos.append(model_info)
                if isinstance(model_info, dict) and model_info.get(
                    "training_window_audit"
                ):
                    training_window_audits.append(
                        {
                            "train_start": train_start_str,
                            "train_end": train_end_str,
                            "status": "passed",
                            **model_info["training_window_audit"],
                        }
                    )

            elif method == "size_conditional_elasticnet":
                predict_window_days = bounded_predict_window_days
                daily_pools_for_predict = self._compute_predict_period_dynamic_pools(
                    train_end,
                    predict_window_days,
                    method_name="SizeConditionalElasticNet",
                )
                try:
                    composite_result, model_info = run_size_conditional_synthesis(
                        self,
                        factor_df,
                        forward_returns,
                        log_market_cap,
                        train_start_str,
                        train_end_str,
                        synthesis_config,
                        daily_pools=daily_pools_for_predict,
                        prepared_panel=prepared_panel,
                    )
                except InsufficientSurvivingFactorsError as exc:
                    training_window_audits.append(
                        {
                            "train_start": train_start_str,
                            "train_end": train_end_str,
                            "status": "failed_survival_gate",
                            **exc.audit,
                        }
                    )
                    _write_training_window_audits()
                    raise
                if isinstance(model_info, dict):
                    model_info.setdefault("train_start", train_start_str)
                    model_info.setdefault("train_end", train_end_str)
                    model_info.setdefault("predict_start", predict_start.strftime("%Y-%m-%d"))
                    model_info.setdefault(
                        "predict_end",
                        model_info.get("validation_end")
                        or predict_end.strftime("%Y-%m-%d"),
                    )
                if not composite_result.empty:
                    all_composite_factors.append(composite_result)
                    model_infos.append(model_info)
                if isinstance(model_info, dict) and model_info.get(
                    "training_window_audit"
                ):
                    training_window_audits.append(
                        {
                            "train_start": train_start_str,
                            "train_end": train_end_str,
                            "status": "passed",
                            **model_info["training_window_audit"],
                        }
                    )

            # 移动到下一个窗口
            current_date += timedelta(days=step_size_days)

        _write_training_window_audits()
        if not all_composite_factors:
            if method == "size_conditional_elasticnet":
                raise RuntimeError(
                    "size_conditional_elasticnet produced no outer-OOS composite factors"
                )
            raise RuntimeError(f"no composite factors produced: {synthesis_name}")

        # 合并所有时间窗口的结果并直接分析
        if method in ["lasso", "lightgbm", "size_conditional_elasticnet"]:
            # LASSO和LightGBM只有一个合成因子
            final_composite = pd.concat(all_composite_factors).sort_index()

            # 检查并处理重复索引
            if final_composite.index.duplicated().any():
                duplicates_count = final_composite.index.duplicated().sum()
                self.logger.warning(
                    f"发现 {duplicates_count} 个重复索引，保留最后出现的值"
                )
                final_composite = final_composite[
                    ~final_composite.index.duplicated(keep="last")
                ]

            composite_name = f"{self.config['output_settings']['composite_factor_prefix']}_{synthesis_name}"

            # [REUSE] 缓存合成因子，供后续 overlay/cost 回放跳过 Lasso 训练。
            if method == "lasso":
                self._save_composite_cache(synthesis_name, final_composite)

            # [REORDERED] 先保存LASSO模型，再运行回测（避免VectorBT加载旧模型）
            # 若为LASSO，导出累积权重CSV（行=验证开始日期，列=因子名）
            if method == "lasso" and lasso_weight_rows:
                try:
                    coef_df = pd.DataFrame(lasso_weight_rows)
                    # 确保列顺序与因子组一致，缺失视为0
                    coef_df = coef_df.reindex(columns=training_factor_names).fillna(0.0)
                    # 去重行（若有重复验证起始日，保留最后一次）
                    if coef_df.index.duplicated().any():
                        dup_count = int(coef_df.index.duplicated().sum())
                        self.logger.warning(
                            f"LASSO权重表存在 {dup_count} 个重复验证起始日，保留最后一个"
                        )
                        coef_df = coef_df[~coef_df.index.duplicated(keep="last")]
                    # 输出目录与文件名（系数）
                    output_dir = self.config.get("output_settings", {}).get(
                        "weights_output_dir", "./results"
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir, f"lasso_weights_{synthesis_name}.csv"
                    )
                    coef_df.to_csv(output_path, encoding="utf-8-sig")
                    self.logger.info(
                        f"已保存LASSO权重CSV: {output_path} (形状={coef_df.shape})"
                    )

                    # Persist the parameter selected in every rolling window;
                    # the coefficient table alone cannot reveal grid-boundary
                    # hits or whether CV scores changed with the search space.
                    selection_path = os.path.join(
                        output_dir, f"lasso_model_selection_{synthesis_name}.json"
                    )
                    selection_data = {
                        "version": "1.0",
                        "synthesis_name": synthesis_name,
                        "method": method,
                        "windows": lasso_selection_records,
                        "summary": {
                            "window_count": len(lasso_selection_records),
                            "alpha_lower_boundary_count": sum(
                                bool(row.get("alpha_at_lower_bound"))
                                for row in lasso_selection_records
                            ),
                            "alpha_upper_boundary_count": sum(
                                bool(row.get("alpha_at_upper_bound"))
                                for row in lasso_selection_records
                            ),
                            "alpha_invalid_count": sum(
                                bool(row.get("alpha_invalid"))
                                for row in lasso_selection_records
                            ),
                            "l1_lower_boundary_count": sum(
                                bool(row.get("l1_at_lower_bound"))
                                for row in lasso_selection_records
                            ),
                            "l1_upper_boundary_count": sum(
                                bool(row.get("l1_at_upper_bound"))
                                for row in lasso_selection_records
                            ),
                        },
                    }
                    with open(selection_path, "w", encoding="utf-8") as f_selection:
                        json.dump(
                            selection_data, f_selection, ensure_ascii=False, indent=2
                        )
                    self.logger.info(
                        "已保存LASSO参数选择审计: %s (窗口=%d)",
                        selection_path,
                        len(lasso_selection_records),
                    )

                    # ============================================================
                    # 打包 Lasso 模型到 JSON Bundle，供云端部署
                    # ============================================================
                    try:
                        bundle_filename = f"lasso_model_bundle_{synthesis_name}.json"
                        bundle_path = os.path.join(output_dir, bundle_filename)

                        parent_baseline_contract = (
                            _materialize_parent_baseline_contract(
                                synthesis_config, output_dir
                            )
                        )

                        # [DYNAMIC_BLEND] 创建从 validation_start 到 model_info 的映射
                        # 用于获取每个窗口的动态 blend ratio
                        model_info_by_date = {}
                        for mi in model_infos:
                            if isinstance(mi, dict):
                                val_start = mi.get("validation_start")
                                if val_start:
                                    model_info_by_date[str(val_start)] = mi

                        bundle_models = []
                        dynamic_blend_count = 0
                        for train_end in coef_df.index:
                            weights = {}
                            for factor in coef_df.columns:
                                coef = coef_df.loc[train_end, factor]
                                if pd.notna(coef) and coef != 0:
                                    weights[factor] = float(coef)

                            # [DYNAMIC_BLEND] 获取该窗口的动态 blend ratio（如果启用）
                            mi = model_info_by_date.get(str(train_end))
                            model = _build_lasso_bundle_model(
                                train_end=train_end,
                                weights=weights,
                                model_info=mi,
                                synthesis_config=synthesis_config,
                                parent_baseline=parent_baseline_contract,
                            )
                            if model and model["blend_info"].get(
                                "dynamic_blend_enabled"
                            ):
                                dynamic_blend_count += 1
                            if model:
                                bundle_models.append(model)

                        if bundle_models:
                            bundle_data = {
                                "version": "1.0",
                                "model_type": "lasso",
                                "experiment": self.config.get(
                                    "output_settings", {}
                                ).get("composite_factor_prefix", "LASSO"),
                                "index": {
                                    "synthesis_name": synthesis_name,
                                    "factor_group": factor_group,
                                    "created_at": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    "requires_parent_baseline": bool(
                                        parent_baseline_contract
                                    ),
                                    "expected_model_count": len(coef_df.index),
                                    "models": sorted(
                                        bundle_models, key=lambda x: x["train_end"]
                                    ),
                                },
                            }

                            with open(bundle_path, "w", encoding="utf-8") as f_bundle:
                                json.dump(
                                    bundle_data, f_bundle, ensure_ascii=False, indent=2
                                )

                            # [DYNAMIC_BLEND] 记录动态 blend 统计
                            if dynamic_blend_count > 0:
                                self.logger.info(
                                    f"已生成 Lasso 模型打包文件: {bundle_path} "
                                    f"(包含 {len(bundle_models)} 个模型, "
                                    f"其中 {dynamic_blend_count} 个使用动态 blend ratio)"
                                )
                            else:
                                self.logger.info(
                                    f"已生成 Lasso 模型打包文件: {bundle_path} (包含 {len(bundle_models)} 个模型)"
                                )
                        else:
                            self.logger.warning(
                                "Lasso Bundle 没有可用的模型信息，跳过打包"
                            )

                    except Exception as e:
                        self.logger.error(f"生成 Lasso 模型打包文件失败: {e}")

                except Exception as e:
                    self.logger.error(f"保存LASSO权重CSV失败: {e}")

            # [REPAIR] LightGBM 需要在 direct_analyze_composite_factor 触发 VectorBT 之前先落盘 bundle
            if method == "lightgbm" and model_infos and model_output_dir:
                try:
                    os.makedirs(model_output_dir, exist_ok=True)
                    bundle_filename = f"lgb_model_bundle_{synthesis_name}.json"
                    bundle_path = os.path.join(model_output_dir, bundle_filename)
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    experiment = self.config.get("output_settings", {}).get(
                        "composite_factor_prefix", "LightGBM"
                    )
                    index_models, _, bundle_data = _build_lightgbm_bundle_payload(
                        model_infos,
                        synthesis_name,
                        factor_group,
                        experiment,
                        created_at=created_at,
                    )

                    if index_models:
                        with open(bundle_path, "w", encoding="utf-8") as f_bundle:
                            json.dump(bundle_data, f_bundle, ensure_ascii=False)
                        self.logger.info(
                            f"已在回测前生成 LightGBM 模型打包文件: {bundle_path} (包含 {len(bundle_data['models'])} 个模型)"
                        )
                    else:
                        self.logger.warning(
                            "LightGBM 没有可用于预生成 bundle 的模型信息，VectorBT 回测将被跳过"
                        )
                except Exception as e:
                    self.logger.error(f"回测前生成 LightGBM 模型打包文件失败: {e}")

            # =========================================================================
            # [NEW] 多重共线性诊断（仅对 LASSO 方法执行）
            # =========================================================================
            if method == "lasso" and lasso_weight_rows:
                try:
                    from factor_synthesis.diagnostics import (
                        run_multicollinearity_diagnostics,
                    )

                    diagnostic_result = run_multicollinearity_diagnostics(
                        lasso_weight_rows=lasso_weight_rows,
                        training_factor_names=training_factor_names,
                        factor_df=factor_df,
                        config=self.config,  # 传入完整配置而非synthesis_config
                        logger=self.logger,
                    )

                    # 可选：将诊断结果记录到模型信息中
                    if diagnostic_result:
                        self.logger.info(
                            f"多重共线性诊断完成，诊断通过: {diagnostic_result.get('diagnostics_passed', False)}"
                        )
                        if not diagnostic_result.get("diagnostics_passed", False):
                            self.logger.warning(
                                f"发现 {len(diagnostic_result.get('issues_found', []))} 个潜在问题"
                            )

                except ImportError:
                    self.logger.warning("诊断模块未安装，跳过多重共线性检查")
                except LabelIsolationError:
                    raise
                except Exception as e:
                    self.logger.error(
                        f"多重共线性诊断异常（不影响后续流程）: {e}", exc_info=True
                    )

            # 直接分析而不保存到数据库
            self.direct_analyze_composite_factor(final_composite, composite_name)
            self.logger.info(f"完成{method.upper()}合成因子分析: {composite_name}")
            model_evidence_diagnostics: Dict[str, Any] = {}
            if method == "lasso":
                diagnostics = _lasso_outer_fold_diagnostics(
                    model_infos,
                    getattr(self, "_vectorbt_result_summaries", []),
                )
                model_evidence_diagnostics = diagnostics
                if diagnostics:
                    self.logger.info(
                        "ALGORITHM_SYNTHESIS_DIAGNOSTICS: %s",
                        json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                    )
            if method == "size_conditional_elasticnet":
                if diagnostics_output_dir is None:
                    raise ValueError(
                        "output_settings.diagnostics_output_dir is required for size_conditional_elasticnet"
                    )
                diagnostic_paths = write_size_conditional_diagnostics(
                    model_infos,
                    diagnostics_output_dir,
                    synthesis_name,
                    model_infos[0]["protocol"],
                )
                model_evidence_diagnostics = {
                    "method": method,
                    "outer_fold_count": len(model_infos),
                    "candidate_trial_count": model_infos[0]["protocol"]["candidate_trial_count"],
                    "artifact_paths": diagnostic_paths,
                    "outer_folds": [
                        {
                            "fold": index,
                            "start": info.get("validation_start"),
                            "end": info.get("validation_end"),
                            "sharpe_ratio": info.get("outer_proxy", {}).get("net_sharpe"),
                            "turnover": info.get("outer_proxy", {}).get("mean_turnover"),
                        }
                        for index, info in enumerate(model_infos, start=1)
                    ],
                }
                self.logger.info(
                    "ALGORITHM_SYNTHESIS_DIAGNOSTICS: %s",
                    json.dumps(
                        {
                            "method": "size_conditional_elasticnet",
                            "outer_fold_count": len(model_infos),
                            "candidate_trial_count": model_infos[0]["protocol"][
                                "candidate_trial_count"
                            ],
                            "artifact_paths": diagnostic_paths,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            if method == "lightgbm":
                try:
                    diagnostics = _lightgbm_synthesis_diagnostics(
                        config_path=self.config_path,
                        synthesis_name=synthesis_name,
                        composite_name=composite_name,
                        synthesis_config=synthesis_config,
                        model_infos=model_infos,
                        final_composite=final_composite,
                        vectorbt_summaries=getattr(
                            self, "_vectorbt_result_summaries", []
                        ),
                    )
                    model_evidence_diagnostics = diagnostics
                    self.logger.info(
                        "ALGORITHM_SYNTHESIS_DIAGNOSTICS: %s",
                        json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                    )
                except LabelIsolationError:
                    raise
                except Exception as e:
                    self.logger.warning("LightGBM structured diagnostics failed: %s", e)

            if model_evidence_diagnostics and hasattr(self, "_execution_metrics"):
                self._execution_metrics["synthesis_diagnostics"].append(model_evidence_diagnostics)
            _write_algorithm_model_evidence(
                diagnostics_output_dir,
                _build_algorithm_model_evidence(
                    method=method,
                    synthesis_name=synthesis_name,
                    model_infos=model_infos,
                    diagnostics=model_evidence_diagnostics,
                    synthesis_config=synthesis_config,
                ),
                syntheses=self._model_evidence,
            )

            # 对 LightGBM：根据滚动窗口训练信息生成模型索引文件，供云端脚本加载
            if method == "lightgbm" and model_infos and model_output_dir:
                try:
                    # ============================================================
                    # Feature Importance Analysis
                    # ============================================================
                    self.logger.info("=" * 60)
                    self.logger.info("🔍 LightGBM Feature Importance Analysis")
                    self.logger.info("=" * 60)

                    feature_stats = {}

                    # Aggregate feature importances
                    for info in model_infos:
                        if not isinstance(info, dict):
                            continue
                        importances = info.get("feature_importance", {})
                        for feat, imp in importances.items():
                            if feat not in feature_stats:
                                feature_stats[feat] = []
                            feature_stats[feat].append(imp)

                    if feature_stats:
                        # Calculate mean and std
                        stats_list = []
                        for feat, values in feature_stats.items():
                            if values:
                                mean_imp = np.mean(values)
                                std_imp = np.std(values)
                                stats_list.append(
                                    {
                                        "Feature": feat,
                                        "Mean Importance": mean_imp,
                                        "Std Dev": std_imp,
                                        "N_Periods": len(values),
                                    }
                                )

                        # Sort by Mean Importance descending
                        stats_df = pd.DataFrame(stats_list)
                        stats_df = stats_df.sort_values(
                            "Mean Importance", ascending=False
                        )

                        # Log the table
                        self.logger.info(f"\n{stats_df.to_string(index=False)}")
                        self.logger.info("-" * 60)
                    else:
                        self.logger.warning("No feature importance data found.")

                    index_filename = f"model_index_{synthesis_name}.json"
                    index_path = os.path.join(model_output_dir, index_filename)
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    experiment = self.config.get("output_settings", {}).get(
                        "composite_factor_prefix", "LightGBM"
                    )
                    index_models, index_data, bundle_data = (
                        _build_lightgbm_bundle_payload(
                            model_infos,
                            synthesis_name,
                            factor_group,
                            experiment,
                            created_at=created_at,
                        )
                    )

                    if index_models:
                        with open(index_path, "w", encoding="utf-8") as f:
                            json.dump(index_data, f, ensure_ascii=False, indent=2)

                        self.logger.info(
                            f"已保存 LightGBM 模型索引文件: {index_path} (模型数={len(index_models)})"
                        )

                        # ============================================================
                        # 打包所有模型到一个JSON文件，便于云端部署
                        # ============================================================
                        try:
                            bundle_filename = f"lgb_model_bundle_{synthesis_name}.json"
                            bundle_path = os.path.join(
                                model_output_dir, bundle_filename
                            )

                            with open(bundle_path, "w", encoding="utf-8") as f_bundle:
                                json.dump(bundle_data, f_bundle, ensure_ascii=False)

                            self.logger.info(
                                f"已生成 LightGBM 模型打包文件: {bundle_path} (包含 {len(bundle_data['models'])} 个模型)"
                            )
                        except Exception as e:
                            self.logger.error(f"生成 LightGBM 模型打包文件失败: {e}")

                    else:
                        self.logger.warning(
                            "LightGBM 没有可用于索引的模型信息，跳过索引文件生成"
                        )
                except Exception as e:
                    self.logger.error(f"保存 LightGBM 模型索引失败: {e}")

        elif method == "pca":
            # PCA有多个主成分
            n_components = synthesis_config.get("n_components", 2)

            for i in range(n_components):
                # 合并第i个主成分
                component_series_list = [
                    cf
                    for cf in all_composite_factors
                    if cf.name == f"pca_component_{i + 1}"
                ]
                if component_series_list:
                    final_component = pd.concat(component_series_list).sort_index()

                    # 检查并处理重复索引
                    if final_component.index.duplicated().any():
                        duplicates_count = final_component.index.duplicated().sum()
                        self.logger.warning(
                            f"PCA成分 {i + 1} 发现 {duplicates_count} 个重复索引，保留最后出现的值"
                        )
                        final_component = final_component[
                            ~final_component.index.duplicated(keep="last")
                        ]

                    composite_name = f"{self.config['output_settings']['composite_factor_prefix']}_{synthesis_name}_PC{i + 1}"

                    # 直接分析而不保存到数据库
                    self.direct_analyze_composite_factor(
                        final_component, composite_name
                    )
                    self.logger.info(f"完成PCA合成因子分析: {composite_name}")

        self.logger.info(f"滚动窗口合成和分析完成: {synthesis_name}")

    def run_synthesis(self, *, execution=None, metrics=None):
        """运行所有配置的因子合成"""
        self.logger.info("开始因子合成任务")

        synthesis_methods = self.config["synthesis_methods"]
        execution = execution if execution is not None else {}
        execution.update(expected=len(synthesis_methods), completed=0, failed=0, errors=[])
        self._execution_metrics = metrics if metrics is not None else {}
        self._vectorbt_result_summaries = []
        self._p2_benchmark_done = False
        self._model_evidence = {}
        self._execution_metrics.update(
            ic_summaries=[], synthesis_diagnostics=[], benchmark_reports=[],
            backtest_runs=self._vectorbt_result_summaries, model_evidence=self._model_evidence,
        )
        if not synthesis_methods:
            raise RuntimeError("no synthesis methods configured")
        failures = []

        for synthesis_name, synthesis_config in synthesis_methods.items():
            try:
                self.logger.info(f"执行合成方法: {synthesis_name}")
                summary_start = len(getattr(self, "_vectorbt_result_summaries", []))
                self.rolling_window_synthesis(synthesis_name, synthesis_config)
                top_k_enabled = self.config.get("performance_evaluation", {}).get(
                    "top_k_strategy", {}
                ).get("enabled", False)
                new_summaries = getattr(self, "_vectorbt_result_summaries", [])[summary_start:]
                if top_k_enabled and not any(
                    isinstance(item, dict)
                    and "error" not in item
                    and all(
                        item.get(key) is not None
                        for key in ("annual_return", "sharpe_ratio", "max_drawdown")
                    )
                    for item in new_summaries
                ):
                    raise RuntimeError("backtest_metrics_missing")
                execution["completed"] += 1

            except Exception as e:
                self.logger.error(f"合成方法 {synthesis_name} 执行失败: {e}")
                failures.append(e)
                execution["failed"] += 1
                execution["errors"].append({"item": synthesis_name, "error": str(e)})
                if isinstance(e, LabelIsolationError):
                    raise
                if synthesis_config.get("method") == "size_conditional_elasticnet":
                    raise
                continue

        if failures:
            raise failures[0]

        self._run_local_reranker_profile()
        if getattr(self, "_vectorbt_result_summaries", None):
            self.logger.info(
                "VECTORBT_RESULT_SUMMARIES: %s",
                json.dumps(
                    self._vectorbt_result_summaries, ensure_ascii=False, default=str
                ),
            )
            self.logger.info("所有因子合成任务完成")

    def _run_local_reranker_profile(self) -> None:
        top_k = self.config.get("performance_evaluation", {}).get(
            "top_k_strategy", {}
        )
        profile = str(top_k.get("local_reranker_profile") or "").strip().lower()
        if not profile:
            return
        if profile != "en_top30_binary_lambdarank_v1":
            raise ValueError(f"unsupported local_reranker_profile: {profile}")

        methods = self.config.get("synthesis_methods", {})
        if len(methods) != 1:
            raise ValueError(f"{profile} requires exactly one synthesis method")
        method_name, method = next(iter(methods.items()))
        if (
            method.get("method") != "lasso"
            or method.get("model_type") != "elasticnet"
            or int(top_k.get("top_k") or 0) != 10
        ):
            raise ValueError(
                f"{profile} requires one ElasticNet synthesis method and top_k=10"
            )

        execution_dir = Path(
            self.config.get("output_settings", {}).get("diagnostics_output_dir", "")
        ).expanduser()
        run_dir = execution_dir.parent
        script = Path(__file__).resolve().parent / (
            "scripts/wind_micro/diagnose_en_lambdarank_rerank.py"
        )
        output_dir = execution_dir / "local_reranker" / profile
        command = [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--candidate-k",
            "30",
            "--top-k",
            "10",
            "--min-train-groups",
            "52",
            "--lookback-groups",
            "104",
            "--lambdarank-weight",
            "0.1",
            "--label-type",
            "binary",
            "--rank-feature-universe",
            "full",
            "--ranker-type",
            "lambdarank",
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{profile} failed: {(completed.stderr or completed.stdout)[-2000:]}"
            )

        result = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        harness_summary = result["harness_summary"]
        harness_summary.update(
            factor_name=f"{method_name}__{profile}",
            trial_id=f"{method_name}__{profile}",
            selection_mode="top_k",
            top_k=10,
            top_pct=None,
            local_reranker_profile=profile,
        )
        self._vectorbt_result_summaries.append(harness_summary)
        if hasattr(self, "_execution_metrics"):
            self._execution_metrics["synthesis_diagnostics"].append({
                "method": profile, "artifact_path": str(output_dir / "summary.json"),
                "prediction_start": result["design"]["prediction_start"],
            })
        self.logger.info(
            "ALGORITHM_SYNTHESIS_DIAGNOSTICS: %s",
            json.dumps(
                {
                    "method": profile,
                    "artifact_path": str(output_dir / "summary.json"),
                    "prediction_start": result["design"]["prediction_start"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    def __del__(self):
        """清理资源"""
        for attr_name in ("factor_conn", "price_conn", "local_conn"):
            with suppress(Exception):
                self._close_conn_attr(attr_name)


def main():
    """主函数 - 内存模式因子合成和分析"""
    import argparse

    parser = argparse.ArgumentParser(description="因子合成系统")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/factor_synthesis_config.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--return-series-output-dir",
        type=str,
        default="",
        help="可选：输出每个 VectorBT candidate 的逐期收益 CSV+metadata",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="可选：写入收益序列 metadata 的 harness run id",
    )
    parser.add_argument("--result-out", default="")
    parser.add_argument("--execution-id", default="")
    args = parser.parse_args()
    from utils.output_paths import execution_directory, process_output_environment
    directory = Path(args.result_out).resolve().parent if args.result_out else execution_directory()
    if not args.result_out:
        args.result_out = str(directory / "result.json")
    if not args.execution_id:
        args.execution_id = directory.name
    if not args.run_id:
        args.run_id = args.execution_id
    with process_output_environment(directory):

        config_path = args.config

        if not os.path.exists(config_path):
            parser.error(f"配置文件不存在: {config_path}")

        print("=" * 60)
        print("因子合成系统启动 (内存模式)")
        print("=" * 60)

        # 执行因子合成和直接分析
        print("\n🚀 开始因子合成和alphalens分析...")
        from tools.agent_harness.run_result import execution_result

        with execution_result(args.result_out, command="factor-synthesis", run_id=args.run_id,
                              execution_id=args.execution_id, config_path=config_path) as result:
            synthesizer = FactorSynthesizer(
                config_path, return_series_output_dir=args.return_series_output_dir, run_id=args.run_id,
            )
            synthesizer.run_synthesis(execution=result["execution"], metrics=result["metrics"])

        print("\n" + "=" * 60)
        print("✅ 因子合成和分析任务完成")
        print("💾 IC/IR/分层收益结果已保存到数据库")
        print("🔍 合成因子未保存，仅保留分析结果")
        print("=" * 60)


if __name__ == "__main__":
    main()
