from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]


class D9CapacityAuditError(ValueError):
    def __init__(
        self, code: str, message: str, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = json.loads(json.dumps(dict(details or {}), default=str))


def _load_object(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D9CapacityAuditError(
            code,
            f"cannot read JSON object: {path}",
            {"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise D9CapacityAuditError(
            code, f"expected JSON object: {path}", {"path": str(path)}
        )
    return payload


def _configured_path(raw: Any, config_path: Path, key: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", f"missing {key}", {"key": key}
        )
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_capacity_audit_context(config_path: Path) -> dict[str, Any] | None:
    config_path = config_path.expanduser().resolve()
    config = _load_object(config_path, "d9_audit_invalid_config")
    audit = config.get("d9_capacity_audit")
    if audit is None:
        return None
    if not isinstance(audit, Mapping):
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", "d9_capacity_audit must be an object"
        )
    if not audit.get("enabled", False):
        return None
    required = (
        "frozen_protocol_path",
        "factor_snapshot_path",
        "expected_snapshot_sha256",
        "expected_factor_count",
        "anchor_factor_id",
        "max_allowed_date",
        "persist_window_audit",
    )
    missing = [key for key in required if key not in audit]
    if missing:
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", "missing audit keys", {"missing": missing}
        )
    expected_count = audit["expected_factor_count"]
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", "expected_factor_count must be positive"
        )
    protocol_path = _configured_path(
        audit["frozen_protocol_path"], config_path, "frozen_protocol_path"
    )
    snapshot_path = _configured_path(
        audit["factor_snapshot_path"], config_path, "factor_snapshot_path"
    )
    protocol = _load_object(protocol_path, "d9_audit_invalid_config")
    snapshot = _load_object(snapshot_path, "d9_audit_invalid_config")
    methods = config.get("synthesis_methods")
    if not isinstance(methods, Mapping) or len(methods) != 1:
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", "audit requires exactly one synthesis method"
        )
    synthesis_name, method_config = next(iter(methods.items()))
    if not isinstance(method_config, Mapping):
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", "synthesis method must be an object"
        )
    group_name = method_config.get("factor_group")
    groups = config.get("factor_groups")
    group = groups.get(group_name) if isinstance(groups, Mapping) else None
    if not isinstance(group_name, str) or not isinstance(group, Mapping):
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", "synthesis factor group is missing"
        )
    outputs = config.get("output_settings")
    if not isinstance(outputs, Mapping):
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", "output_settings must be an object"
        )
    weights_dir = _configured_path(
        outputs.get("weights_output_dir"), config_path, "weights_output_dir"
    )
    diagnostics_dir = _configured_path(
        outputs.get("diagnostics_output_dir"), config_path, "diagnostics_output_dir"
    )
    factor_ids = group.get("factors")
    if not isinstance(factor_ids, list) or not all(
        isinstance(factor, str) and factor for factor in factor_ids
    ):
        raise D9CapacityAuditError(
            "d9_audit_invalid_config", "factor group must contain string identities"
        )
    return {
        "config_path": str(config_path),
        "config": config,
        "audit_config": dict(audit),
        "protocol_path": str(protocol_path),
        "protocol": protocol,
        "snapshot_path": str(snapshot_path),
        "snapshot": snapshot,
        "synthesis_name": str(synthesis_name),
        "synthesis_config": dict(method_config),
        "group_name": group_name,
        "factor_ids": list(factor_ids),
        "model_bundle_path": str(
            weights_dir / f"lasso_model_bundle_{synthesis_name}.json"
        ),
        "training_audit_path": str(
            diagnostics_dir / "training_window_factor_audit.json"
        ),
        "expected_factor_count": expected_count,
        "max_allowed_date": str(audit["max_allowed_date"]),
        "is_d9_profile": protocol.get("experiment_id")
        == "wind_micro_wide62_capacity_d9_20260815",
    }


def _check(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any) -> None:
    checks.append(
        {
            "name": name,
            "passed": actual == expected,
            "actual": actual,
            "expected": expected,
        }
    )


def preflight_capacity_audit(context: Mapping[str, Any]) -> dict[str, Any]:
    config = context["config"]
    method = context["synthesis_config"]
    snapshot = context["snapshot"]
    protocol = context["protocol"]
    audit = context["audit_config"]
    expected_count = context["expected_factor_count"]
    factors = context["factor_ids"]
    checks: list[dict[str, Any]] = []
    snapshot_path = Path(context["snapshot_path"])
    _check(
        checks,
        "snapshot_sha256",
        hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        audit["expected_snapshot_sha256"],
    )
    snapshot_factors = snapshot.get("factor_ids")
    snapshot_factors_valid = isinstance(snapshot_factors, list) and all(
        isinstance(factor, str) and factor for factor in snapshot_factors
    )
    _check(checks, "snapshot_factor_ids_valid", snapshot_factors_valid, True)
    _check(
        checks, "snapshot_factor_count", snapshot.get("factor_count"), expected_count
    )
    _check(
        checks,
        "snapshot_factor_list_count",
        len(snapshot_factors) if isinstance(snapshot_factors, list) else None,
        expected_count,
    )
    _check(checks, "config_factor_count", len(factors), expected_count)
    _check(checks, "factor_identity_and_order", factors, snapshot_factors)
    _check(
        checks,
        "factor_id_uniqueness",
        len(set(factors)),
        len(factors),
    )
    _check(checks, "anchor_excluded", audit["anchor_factor_id"] not in factors, True)
    _check(checks, "persist_window_audit", audit["persist_window_audit"], True)
    _check(checks, "method", method.get("method"), "lasso")
    _check(checks, "model_type", method.get("model_type"), "elasticnet")
    _check(checks, "positive", method.get("positive"), True)
    _check(
        checks, "factor_count_grid", method.get("factor_count_grid"), [expected_count]
    )
    _check(
        checks,
        "analysis_start",
        config.get("analysis_period", {}).get("start_date"),
        "2018-01-01",
    )
    _check(
        checks,
        "analysis_end",
        config.get("analysis_period", {}).get("end_date"),
        context["max_allowed_date"],
    )
    evaluation_dates = [
        date
        for fold in protocol.get("evaluation_folds", [])
        if isinstance(fold, Mapping)
        for date in fold.get("evaluation", [])
    ]
    _check(
        checks,
        "evaluation_dates_bounded",
        all(
            isinstance(date, str) and date <= context["max_allowed_date"]
            for date in evaluation_dates
        ),
        True,
    )
    if context["is_d9_profile"]:
        fixed = protocol.get("fixed_context", {})
        d9_expected = {
            "expected_factor_count": (expected_count, 62),
            "n_alphas": (method.get("n_alphas"), fixed.get("n_alphas")),
            "lasso_alpha_range": (
                method.get("lasso_alpha_range"),
                fixed.get("lasso_alpha_range"),
            ),
            "l1_ratio_range": (
                method.get("l1_ratio_range"),
                fixed.get("l1_ratio_range"),
            ),
            "lasso_cv_folds": (
                method.get("lasso_cv_folds"),
                fixed.get("lasso_cv_folds"),
            ),
            "lasso_max_iter": (
                method.get("lasso_max_iter"),
                fixed.get("lasso_max_iter"),
            ),
            "param_search_metric": (
                method.get("param_search_metric"),
                fixed.get("param_search_metric"),
            ),
            "cv_cost_rate": (method.get("cv_cost_rate"), fixed.get("cv_cost_rate")),
            "cv_min_oof_periods": (
                method.get("cv_min_oof_periods"),
                fixed.get("cv_min_oof_periods"),
            ),
            "cv_rebalance_stride": (method.get("cv_rebalance_stride"), 5),
            "cv_n_jobs": (method.get("cv_n_jobs"), 1),
            "cv_dsr_min_threshold": (method.get("cv_dsr_min_threshold"), 0.0),
            "pooled_oof_dsr": (method.get("pooled_oof_dsr"), True),
            "residual_gamma_grid": (
                method.get("residual_gamma_grid"),
                fixed.get("residual_gamma_grid"),
            ),
            "target_period": (method.get("target_period"), 5),
            "rolling_train_window_days": (
                config.get("rolling_window", {}).get("train_window_days"),
                fixed.get("rolling_train_window_days"),
            ),
            "rolling_predict_window_days": (
                config.get("rolling_window", {}).get("predict_window_days"),
                fixed.get("rolling_predict_window_days"),
            ),
            "rolling_step_days": (
                config.get("rolling_window", {}).get("step_size_days"),
                fixed.get("rolling_step_days"),
            ),
            "top_k": (
                config.get("performance_evaluation", {})
                .get("top_k_strategy", {})
                .get("top_k"),
                fixed.get("top_k"),
            ),
            "execution_cost_rate": (
                config.get("performance_evaluation", {})
                .get("top_k_strategy", {})
                .get("cost_rate"),
                fixed.get("execution_cost_rate"),
            ),
            "slippage": (
                config.get("performance_evaluation", {})
                .get("top_k_strategy", {})
                .get("slippage"),
                fixed.get("slippage"),
            ),
        }
        for name, (actual, expected) in d9_expected.items():
            _check(checks, name, actual, expected)
        for block in ("pre_selection", "barra_preselection", "hierarchical_clustering"):
            _check(
                checks, f"{block}_disabled", method.get(block, {}).get("enabled"), False
            )
        _check(
            checks,
            "anchor_blend",
            method.get("blend_weights"),
            {audit["anchor_factor_id"]: 1.0},
        )
        _check(
            checks, "anchor_blend_ratio", method.get("blend_with_original_ratio"), 1.0
        )
        _check(
            checks,
            "target_residualization_enabled",
            method.get("target_residualization", {}).get("enabled"),
            True,
        )
        _check(
            checks,
            "anchor_residualization",
            method.get("target_residualization", {}).get("baseline_factors"),
            [audit["anchor_factor_id"]],
        )
        _check(
            checks,
            "calendar_filters_locked",
            config.get("calendar_filters", {}).get("skip_months"),
            ["01-16:01-31", "04-16:04-30"],
        )
        _check(
            checks,
            "calendar_apply_flags",
            [
                config.get("calendar_filters", {}).get(key)
                for key in (
                    "apply_to_backtest",
                    "apply_to_evaluation",
                    "apply_to_training",
                )
            ],
            [True, True, True],
        )
        filters = config.get("stock_pool", {}).get("filters", [])
        pool_projection = [
            (item.get("field"), item.get("operator"), item.get("value"))
            for item in filters
            if isinstance(item, Mapping)
        ]
        _check(
            checks,
            "stock_pool_filters",
            pool_projection,
            [
                ("paused", "==", 0),
                ("index_code", "==", "8841431.WI"),
                ("is_st", "==", 0),
                ("is_new_stock", "==", 0),
                ("stock_code", "not_starts_with", "68"),
                ("val_market_cap", "percentile_bottom", 50),
            ],
        )
        _check(
            checks,
            "stock_pool_settings",
            [
                config.get("stock_pool", {}).get("use_dynamic_pools"),
                config.get("stock_pool", {}).get("include_basic_filters"),
                config.get("stock_pool", {}).get("rebalance_schedule"),
            ],
            [True, False, "weekly"],
        )
    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise D9CapacityAuditError(
            "d9_audit_invalid_config",
            "capacity audit preflight failed",
            {"failed_checks": failed, "checks": checks},
        )
    return {
        "passed": True,
        "checks": checks,
        "factor_ids": list(factors),
        "factor_count": expected_count,
        "anchor_factor_id": audit["anchor_factor_id"],
        "max_allowed_date": context["max_allowed_date"],
    }


def _artifact_error(message: str, **details: Any) -> D9CapacityAuditError:
    return D9CapacityAuditError("d9_audit_artifact_contract_failed", message, details)


def build_capacity_audit(
    context: Mapping[str, Any],
    preflight: Mapping[str, Any],
    harness_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = _load_object(
        Path(context["model_bundle_path"]), "d9_audit_artifact_contract_failed"
    )
    training = _load_object(
        Path(context["training_audit_path"]), "d9_audit_artifact_contract_failed"
    )
    models = bundle.get("index", {}).get("models")
    audits = training.get("windows")
    if (
        not isinstance(models, list)
        or not isinstance(audits, list)
        or len(models) != 12
        or len(audits) != 12
    ):
        raise _artifact_error(
            "expected exactly 12 model and training windows",
            model_count=len(models) if isinstance(models, list) else None,
            audit_count=len(audits) if isinstance(audits, list) else None,
        )
    if bundle.get("index", {}).get("synthesis_name") not in (
        None,
        context["synthesis_name"],
    ) or bundle.get("index", {}).get("factor_group") not in (
        None,
        context["group_name"],
    ):
        raise _artifact_error("model bundle identity mismatch")
    if training.get("synthesis_name") not in (
        None,
        context["synthesis_name"],
    ) or training.get("method") not in (None, "lasso"):
        raise _artifact_error("training audit identity mismatch")
    backtest = harness_metrics.get("backtest_summary")
    artifact = (
        backtest.get("return_series_artifact")
        if isinstance(backtest, Mapping)
        else None
    )
    if not isinstance(artifact, Mapping):
        raise _artifact_error("missing exact return-series artifact metadata")
    return_path = Path(str(artifact.get("path", "")))
    metadata_path = Path(str(artifact.get("metadata_path", "")))
    if (
        artifact.get("artifact_type") != "algorithm_return_series"
        or artifact.get("format") != "csv"
        or not return_path.is_file()
        or not metadata_path.is_file()
    ):
        raise _artifact_error(
            "return-series artifact contract mismatch", artifact=artifact
        )
    return_metadata = _load_object(metadata_path, "d9_audit_artifact_contract_failed")
    if return_metadata.get("end_date", "9999-99-99") > context["max_allowed_date"]:
        raise _artifact_error(
            "future-dated return metadata", end_date=return_metadata.get("end_date")
        )
    expected_sha = return_metadata.get("sha256")
    actual_sha = hashlib.sha256(return_path.read_bytes()).hexdigest()
    if not expected_sha or expected_sha != actual_sha:
        raise _artifact_error(
            "return-series hash mismatch", expected=expected_sha, actual=actual_sha
        )
    windows: list[dict[str, Any]] = []
    positive_counts: list[int] = []
    selected: Counter[str] = Counter()
    signs: dict[str, Counter[str]] = {}
    fallbacks: Counter[str] = Counter()
    previous_validation_start = ""
    for ordinal, (model, audit) in enumerate(zip(models, audits), 1):
        if not isinstance(model, Mapping) or not isinstance(audit, Mapping):
            raise _artifact_error("window entries must be objects", ordinal=ordinal)
        selection = model.get("selection")
        weights = model.get("weights")
        if not isinstance(selection, Mapping) or not isinstance(weights, Mapping):
            raise _artifact_error("model selection or weights missing", ordinal=ordinal)
        if selection.get("cv_factor_count") not in (
            None,
            context["expected_factor_count"],
        ):
            raise _artifact_error("model factor-count mismatch", ordinal=ordinal)
        dates = {
            key: selection.get(key)
            for key in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
            )
        }
        if any(not isinstance(value, str) for value in dates.values()) or any(
            value > context["max_allowed_date"] for value in dates.values()
        ):
            raise _artifact_error(
                "invalid or future-dated model window", ordinal=ordinal, dates=dates
            )
        if (
            dates["validation_start"] <= previous_validation_start
            or audit.get("train_start") != dates["train_start"]
            or audit.get("train_end") != dates["train_end"]
        ):
            raise _artifact_error(
                "window order or training-window pairing mismatch", ordinal=ordinal
            )
        previous_validation_start = dates["validation_start"]
        gamma = selection.get("selected_residual_gamma")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, (int, float))
            or not math.isfinite(gamma)
        ):
            raise _artifact_error(
                "invalid residual gamma", ordinal=ordinal, gamma=gamma
            )
        fitted_coefficients = selection.get("residual_sleeve_coefficients") or weights
        if not isinstance(fitted_coefficients, Mapping):
            raise _artifact_error("fitted coefficients missing", ordinal=ordinal)
        for factor, coefficient in fitted_coefficients.items():
            if (
                isinstance(coefficient, bool)
                or not isinstance(coefficient, (int, float))
                or not math.isfinite(coefficient)
            ):
                raise _artifact_error(
                    "non-finite coefficient", ordinal=ordinal, factor=factor
                )
        active = dict(weights) if gamma > 0 else {}
        if gamma > 0:
            positive_counts.append(len(active))
            for factor, coefficient in active.items():
                selected[str(factor)] += 1
                signs.setdefault(str(factor), Counter())[
                    "positive"
                    if coefficient > 0
                    else "negative"
                    if coefficient < 0
                    else "zero"
                ] += 1
        fallback = selection.get("cv_fallback_reason")
        fallbacks[str(fallback) if fallback else "none"] += 1
        fitted_count = selection.get("residual_sleeve_n_selected")
        if not isinstance(fitted_count, int):
            fitted_count = sum(value != 0 for value in fitted_coefficients.values())
        windows.append(
            {
                "ordinal": ordinal,
                **dates,
                "configured_input_count": audit.get("configured_factor_count"),
                "surviving_input_count": audit.get("surviving_factor_count"),
                "missing_data_status": audit.get("status"),
                "selected_gamma": gamma,
                "cv_fallback_reason": fallback,
                "fitted_residual_nonzero_count": fitted_count,
                "active_residual_nonzero_count": len(active),
                "active_factor_ids": sorted(active),
                "active_coefficients": {key: active[key] for key in sorted(active)},
                "date_boundary_passed": True,
            }
        )
        if audit.get("configured_factor_count") != context["expected_factor_count"]:
            raise _artifact_error(
                "training audit factor-count mismatch", ordinal=ordinal
            )
    if [window["validation_start"] for window in windows] != sorted(
        window["validation_start"] for window in windows
    ):
        raise _artifact_error("windows are not chronologically ordered")
    surviving = [window["surviving_input_count"] for window in windows]
    if any(not isinstance(value, int) for value in surviving):
        raise _artifact_error("surviving input count missing")
    frequency = (
        {factor: selected[factor] / len(positive_counts) for factor in sorted(selected)}
        if positive_counts
        else {}
    )
    stability = {
        factor: max(counts.values()) / sum(counts.values())
        for factor, counts in sorted(signs.items())
    }
    aggregate = {
        "window_count": len(windows),
        "positive_gamma_window_count": len(positive_counts),
        "median_active_nonzero_count": statistics.median(positive_counts)
        if positive_counts
        else 0.0,
        "mean_active_nonzero_count": statistics.fmean(positive_counts)
        if positive_counts
        else 0.0,
        "factor_selection_frequency": frequency,
        "coefficient_sign_stability": stability,
        "fallback_counts": dict(sorted(fallbacks.items())),
        "pure_anchor_count": sum(window["selected_gamma"] == 0 for window in windows),
        "minimum_surviving_input_count": min(surviving),
        "all_windows_at_least_40": min(surviving) >= 40,
    }
    return {
        "schema_version": "d9_capacity_audit.v1",
        "synthesis_name": context["synthesis_name"],
        "factor_group": context["group_name"],
        "protocol_path": context["protocol_path"],
        "snapshot_path": context["snapshot_path"],
        "preflight": dict(preflight),
        "windows": windows,
        "aggregate": aggregate,
        "backtest_summary": dict(backtest),
        "return_series_artifact": dict(artifact),
        "return_series_metadata": return_metadata,
    }


def write_capacity_audit(output_path: Path, payload: Mapping[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
