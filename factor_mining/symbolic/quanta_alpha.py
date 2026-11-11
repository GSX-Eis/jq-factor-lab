"""Execution-side realization and cheap screening for QuantaAlpha v2."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .compiler import compile_factor_file
from .config import load_analysis_context, load_mining_config
from .data_loader import _label_targets, build_training_panel
from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_long_df
from .expression import (
    ExpressionNode,
    canonical_expression_hash,
    deserialize_expression,
    evaluate_expression_cached,
    expression_complexity,
    referenced_features,
    serialize_expression,
)
from .feature_space import build_feature_space
from .objective import rank_ic_by_date, residualize_by_date
from .search_base import SearchCandidate
from .structural import structural_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_RESEARCH_DATE = pd.Timestamp("2023-12-31")
PROFILES = {"paper_v3_parity", "paper_v3_funnel"}


class QuantaAlphaEvidenceError(RuntimeError):
    """Required research evidence could not be computed without guessing."""


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_factor_value_cache(
    cache_dir: Path,
    *,
    source_hash: str,
    panel: pd.DataFrame,
    prediction: pd.Series,
    panel_hash: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    from utils.data_access_config import load_data_access_config
    cache_dir.mkdir(parents=True, exist_ok=True)
    values = panel[["date", "code"]].copy()
    values["factor"] = pd.Series(prediction, index=panel.index)
    wide = values.pivot(index="date", columns="code", values="factor").sort_index()
    data_path = cache_dir / f"{source_hash}.parquet"
    temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp.parquet")
    wide.to_parquet(temporary)
    temporary.replace(data_path)
    metadata = {
        "schema_version": "quanta_alpha.factor_value_cache.v1",
        "data_access": load_data_access_config().as_dict(),
        "source_sha256": source_hash,
        "data_file": data_path.name,
        "data_sha256": _file_hash(data_path),
        "panel_sha256": panel_hash,
        "start_date": start_date,
        "end_date": end_date,
        "rows": int(wide.shape[0]),
        "columns": int(wide.shape[1]),
    }
    metadata_path = cache_dir / f"{source_hash}.json"
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": _file_hash(metadata_path),
        **metadata,
    }


def _write_sealed_returns(
    output_dir: Path,
    *,
    payload: Mapping[str, Any],
    config: dict,
    input_path: Path,
    panel: pd.DataFrame,
    panel_hash: str,
    is_start: str,
    is_end: str,
) -> dict[str, str]:
    """Freeze already computed research labels; never query or reconstruct them."""
    targets = _label_targets(config)
    if len(targets) != 1 or int(targets[0]["start"]) != 2:
        raise QuantaAlphaEvidenceError("sealed returns require one T+1 label target")
    period = int(targets[0]["end"]) - int(targets[0]["start"]) + 1
    if period <= 0 or pd.Timestamp(is_end) > MAX_RESEARCH_DATE:
        raise QuantaAlphaEvidenceError("invalid sealed returns IS window or period")
    analysis_raw = payload.get("analysis_config")
    production_raw = payload.get("production_analysis_config")
    if not analysis_raw or not production_raw or not payload.get("data_snapshot_hash"):
        raise QuantaAlphaEvidenceError("sealed returns require frozen campaign analysis and data identity")
    analysis_path = _resolve_path(str(analysis_raw), base=input_path.parent)
    production_path = _resolve_path(str(production_raw), base=input_path.parent)
    analysis = load_analysis_context(str(analysis_path))
    production = load_analysis_context(str(production_path))
    pool_names = list(analysis.pool_configs)
    if len(pool_names) != 1 or pool_names != list(production.pool_configs):
        raise QuantaAlphaEvidenceError("sealed returns require one matching frozen pool")
    feature_start = str(payload.get("research_start", "2018-01-01"))
    if feature_start != is_start or str(production.runtime_config.get("factor_history_start_date", "")) != feature_start:
        raise QuantaAlphaEvidenceError("research and production feature history origins differ")
    if str(production.runtime_config.get("start_date")) != is_start or str(production.runtime_config.get("end_date")) != is_end:
        raise QuantaAlphaEvidenceError("production and research IS windows differ")
    runtime = analysis.runtime_config
    pool_raw = runtime.get("pool_snapshot_path")
    pool_sha = str(runtime.get("pool_snapshot_sha256") or "")
    if not pool_raw or not pool_sha or runtime.get("pool_snapshot_read_only") is not True:
        raise QuantaAlphaEvidenceError("sealed returns require a hash-bound read-only pool snapshot")
    pool_path = _resolve_path(str(pool_raw), base=analysis_path.parent)
    if _file_hash(pool_path) != pool_sha:
        raise QuantaAlphaEvidenceError("sealed returns pool snapshot hash mismatch")
    production_runtime = production.runtime_config
    if (production_runtime.get("pool_snapshot_read_only") is not True
        or _resolve_path(str(production_runtime.get("pool_snapshot_path") or ""), base=production_path.parent) != pool_path
        or production_runtime.get("pool_snapshot_sha256") != pool_sha):
        raise QuantaAlphaEvidenceError("production and research pool snapshots differ")
    pools = _load_json(pool_path).get("pools", {}).get(pool_names[0])
    if not pools:
        raise QuantaAlphaEvidenceError("sealed returns pool snapshot has no members")
    values = panel[["date", "code", "label"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce").dt.normalize()
    feature_keys = values.dropna(subset=["date", "code"])
    if feature_keys["date"].lt(pd.Timestamp(feature_start)).any() or feature_keys.duplicated(["date", "code"]).any():
        raise QuantaAlphaEvidenceError("research feature panel violates its frozen origin or row identity")
    feature_members = apply_date_level_pool_filter_to_long_df(
        feature_keys.rename(columns={"code": "stock_code"}).set_index(["date", "stock_code"]), pools
    )
    if len(feature_members) != len(feature_keys):
        raise QuantaAlphaEvidenceError("research feature panel includes out-of-pool rows")
    values = values.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "code", "label"])
    values = values[values["date"].between(pd.Timestamp(is_start), pd.Timestamp(is_end))]
    values = values.sort_values(["date", "code"]).reset_index(drop=True)
    if values.empty or values.duplicated(["date", "code"]).any():
        raise QuantaAlphaEvidenceError("sealed returns require unique, nonempty date/code labels")
    dated_members = apply_date_level_pool_filter_to_long_df(
        values.rename(columns={"code": "stock_code"}).set_index(["date", "stock_code"]), pools
    )
    if len(dated_members) != len(values):
        raise QuantaAlphaEvidenceError("research labels include out-of-pool rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = (output_dir / "sealed_returns.parquet").resolve()
    temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp.parquet")
    values.to_parquet(temporary, index=False)
    temporary.replace(data_path)
    receipt = {
        "schema_version": "quanta_alpha.sealed_returns.v1",
        "period": period, "execution_lag": 1,
        "pool_id": pool_names[0], "is_start": is_start, "is_end": is_end,
        "feature_start_date": feature_start,
        "analysis_config_path": str(analysis_path), "analysis_config_sha256": _file_hash(analysis_path),
        "pool_snapshot_path": str(pool_path), "pool_snapshot_sha256": pool_sha,
        "data_snapshot_hash": str(payload["data_snapshot_hash"]),
        "input_hash": _file_hash(input_path), "panel_hash": panel_hash,
        "data_path": str(data_path), "data_sha256": _file_hash(data_path), "rows": len(values),
    }
    receipt_path = output_dir / "sealed_returns.receipt.json"
    temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    return {"path": str(data_path), "sha256": receipt["data_sha256"],
            "receipt_path": str(receipt_path.resolve()), "receipt_sha256": _file_hash(receipt_path)}


def _resolve_path(raw: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("QuantaAlpha realization input must be a JSON object")
    return payload


def _load_panel(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        panel = pd.read_parquet(path)
    elif suffix in {".pkl", ".pickle"}:
        panel = pd.read_pickle(path)
    elif suffix == ".csv":
        panel = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported panel file: {path}")
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("Panel artifact did not contain a DataFrame")
    return panel


def _proposal_expression(proposal: Mapping[str, Any]) -> ExpressionNode:
    raw = proposal.get("expression") or proposal.get("expression_ast")
    if not isinstance(raw, dict):
        raise ValueError("Each proposal must contain a serialized expression AST")
    return deserialize_expression(raw)


def _bound_text_hash(value: Any) -> str:
    return _stable_hash(str(value or ""))


def _consistency_gate(
    proposal: Mapping[str, Any], expr: ExpressionNode
) -> dict[str, Any]:
    report = proposal.get("consistency_report")
    reasons: list[str] = []
    if not isinstance(report, Mapping):
        return {"passed": False, "reasons": ["missing_consistency_report"]}

    reviewer = str(report.get("reviewer") or "").strip()
    severity = str(report.get("severity", "critical")).strip().lower()
    if not reviewer:
        reasons.append("missing_consistency_reviewer")
    if severity not in {"pass", "minor", "major", "critical"}:
        reasons.append("invalid_consistency_severity")
    if severity in {"major", "critical"}:
        reasons.append("material_consistency_failure")
    bindings = {
        "hypothesis_hash": _bound_text_hash(proposal.get("hypothesis")),
        "semantic_hash": _bound_text_hash(proposal.get("semantic_description")),
        "expression_hash": _stable_hash(serialize_expression(expr)),
    }
    if str(proposal.get("expression_hash") or "") != bindings["expression_hash"]:
        reasons.append("proposal_expression_hash_mismatch")
    for key, expected in bindings.items():
        actual = str(report.get(key) or "")
        if actual != expected:
            reasons.append(f"{key}_mismatch")
    rewrite_count = int(report.get("rewrite_count", 0) or 0)
    if rewrite_count < 0 or rewrite_count > 3:
        reasons.append("consistency_rewrite_limit_exceeded")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "reviewer": reviewer,
        "severity": severity,
        "rewrite_count": rewrite_count,
        "bindings": bindings,
    }


def _window(
    payload: Mapping[str, Any], key: str, default_start: str, default_end: str
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    windows = payload.get("windows")
    raw = payload.get(key)
    if raw is None and isinstance(windows, Mapping):
        raw = windows.get(key)
    config = raw if isinstance(raw, Mapping) else {}
    start = pd.Timestamp(config.get("start", default_start)).normalize()
    end = pd.Timestamp(config.get("end", default_end)).normalize()
    if start > end:
        raise ValueError(f"{key} start must not be after end")
    if end > MAX_RESEARCH_DATE:
        raise ValueError(f"{key} may not access dates after {MAX_RESEARCH_DATE.date()}")
    default_samples = 252 if key == "train" else 126
    samples = int(config.get("sample_dates", default_samples))
    if samples <= 0:
        raise ValueError(f"{key}.sample_dates must be positive")
    return start, end, samples


def _sample_dates(
    panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, count: int
) -> list[pd.Timestamp]:
    dates = sorted(
        pd.to_datetime(panel["date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .loc[lambda values: (values >= start) & (values <= end)]
        .unique()
    )
    normalized = [pd.Timestamp(value).normalize() for value in dates]
    if len(normalized) <= count:
        return normalized
    positions = np.linspace(0, len(normalized) - 1, num=count, dtype=int)
    return [normalized[index] for index in sorted(set(positions.tolist()))]


def _partition(panel: pd.DataFrame, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    selected = set(dates)
    normalized = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    return panel[normalized.isin(selected)].copy()


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _rank_ic_mean(pred: pd.Series, panel: pd.DataFrame) -> float | None:
    values = rank_ic_by_date(pred, panel["label"], panel["date"])
    return _finite_float(values.mean()) if not values.empty else None


def _partition_metrics(pred: pd.Series, panel: pd.DataFrame) -> dict[str, Any]:
    if panel.empty:
        return {"rank_ic": None, "coverage": 0.0, "variance": None, "n_dates": 0}
    aligned = pd.Series(pred, index=panel.index).replace([np.inf, -np.inf], np.nan)
    finite = aligned.notna() & panel["label"].notna()
    return {
        "rank_ic": _rank_ic_mean(aligned, panel),
        "coverage": float(finite.mean()),
        "variance": _finite_float(aligned[finite].var(ddof=1))
        if finite.any()
        else None,
        "n_dates": int(panel["date"].nunique()),
    }


def _screen_metrics(
    pred: pd.Series,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
) -> dict[str, Any]:
    train_pred = pred.loc[train_panel.index]
    valid_pred = pred.loc[valid_panel.index]
    train_metrics = _partition_metrics(train_pred, train_panel)
    valid_metrics = _partition_metrics(valid_pred, valid_panel)
    anchor_columns = sorted(
        column for column in valid_panel if str(column).startswith("anchor__")
    )[:40]
    anchors = (
        np.asarray(valid_panel[anchor_columns], dtype=float) if anchor_columns else None
    )
    marginal_rank_ic = valid_metrics["rank_ic"]
    max_anchor_corr = None
    if anchors is not None:
        residual = residualize_by_date(valid_pred, valid_panel["date"], anchors)
        marginal_rank_ic = _rank_ic_mean(residual, valid_panel)
        correlations = [
            abs(value)
            for column in anchor_columns
            if (
                value := _mean_pair_rank_corr(
                    valid_pred,
                    valid_panel[column],
                    valid_panel["date"],
                )
            )
            is not None
        ]
        max_anchor_corr = max(correlations, default=None)
    return {
        "train_rank_ic": train_metrics["rank_ic"],
        "validation_rank_ic": valid_metrics["rank_ic"],
        "sampled_marginal_rank_ic": marginal_rank_ic,
        "coverage": valid_metrics["coverage"],
        "variance": valid_metrics["variance"],
        "anchor_count": len(anchor_columns),
        "max_abs_anchor_correlation": max_anchor_corr,
        "max_abs_output_correlation": None,
        "train_dates": int(train_panel["date"].nunique()),
        "validation_dates": int(valid_panel["date"].nunique()),
    }


def _mean_pair_rank_corr(
    left: pd.Series, right: pd.Series, dates: pd.Series
) -> float | None:
    values = rank_ic_by_date(left, right, dates)
    return _finite_float(values.mean()) if not values.empty else None


def _prospective_rank_key(row: Mapping[str, Any]) -> tuple:
    metrics = row.get("funnel_metrics") or {}
    marginal = metrics.get(
        "direction_adjusted_sampled_marginal_rank_ic",
        metrics.get("sampled_marginal_rank_ic"),
    )
    validation = metrics.get(
        "direction_adjusted_validation_rank_ic",
        metrics.get("validation_rank_ic"),
    )
    return (
        not bool((row.get("funnel_gate") or {}).get("passed")),
        -float(marginal) if marginal is not None else math.inf,
        -float(validation) if validation is not None else math.inf,
        float(metrics.get("max_abs_output_correlation") or 0.0),
        float(metrics.get("complexity") or 0.0),
        str(row.get("expression_hash") or ""),
    )


def _panel_hash(panel: pd.DataFrame, source_path: Path | None) -> str:
    if source_path is not None:
        return _file_hash(source_path)
    required = {"date", "code", "label"}
    if not required <= set(panel.columns):
        raise ValueError("panel identity requires date, code, and label")
    evidence = sorted(
        column
        for column in panel.columns
        if str(column).startswith(("label__", "anchor__"))
    )
    identity = panel[["date", "code", "label", *evidence]].copy()
    identity["date"] = pd.to_datetime(identity["date"], errors="coerce").astype(str)
    identity = identity.sort_values(["date", "code"]).reset_index(drop=True)
    values = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_config_and_panel(
    payload: Mapping[str, Any], input_path: Path
) -> tuple[dict, pd.DataFrame, Path | None]:
    raw_config = payload.get("symbolic_config")
    config_path_raw = payload.get("config") or payload.get("mining_config")
    if isinstance(raw_config, Mapping):
        config = copy.deepcopy(dict(raw_config))
    elif config_path_raw:
        config_path = _resolve_path(str(config_path_raw), base=input_path.parent)
        config = load_mining_config(str(config_path))
    else:
        config_path = (
            REPO_ROOT / "config" / "symbolic_mining" / "dso_dsl_full_smoke.json"
        )
        config = load_mining_config(str(config_path))

    config.setdefault("search", {})["dsl"] = "full_alpha"
    config["search"].setdefault("windows", [3, 5, 10, 20, 60])
    constants = config["search"].setdefault("constants", {})
    constants["enabled"] = True
    constants.setdefault("values", [-2, -1, -0.5, 0.5, 1, 2])
    factor_data_fields = payload.get("factor_data_fields")
    if factor_data_fields is not None:
        if not isinstance(factor_data_fields, list) or any(
            not isinstance(field, str) or not field.strip()
            for field in factor_data_fields
        ):
            raise ValueError(
                "factor_data_fields must be a list of non-empty field names"
            )
        config.setdefault("features", {})["factor_data_fields"] = list(
            dict.fromkeys(field.strip() for field in factor_data_fields)
        )

    expression_rows = [
        *list(payload.get("proposals") or []),
        *list(payload.get("zoo_expressions") or []),
    ]
    requested_features: set[str] = set()
    for row in expression_rows:
        raw_expression = row.get("expression", row) if isinstance(row, Mapping) else row
        if isinstance(raw_expression, Mapping):
            requested_features.update(
                referenced_features(deserialize_expression(dict(raw_expression)))
            )
    if requested_features:
        available = build_feature_space(config).features
        selected = [
            available[name] for name in sorted(requested_features) if name in available
        ]
        feature_config = config.setdefault("features", {})
        feature_config["daily_price_fields"] = sorted(
            {spec.field for spec in selected if spec.source == "price_daily"}
        )
        feature_config["factor_data_fields"] = sorted(
            {spec.field for spec in selected if spec.source == "factor_data"}
        )
        intraday = [spec for spec in selected if spec.source == "price_5m"]
        feature_config["intraday_5m_fields"] = sorted({spec.field for spec in intraday})
        feature_config["intraday_aggregators"] = sorted(
            {str(spec.aggregator) for spec in intraday if spec.aggregator}
        )

    anchors = [str(item) for item in payload.get("anchors", [])][:40]

    def require_anchor_evidence(panel: pd.DataFrame) -> None:
        missing = [name for name in anchors if f"anchor__{name}" not in panel.columns]
        if missing:
            raise QuantaAlphaEvidenceError(
                "missing requested anchor evidence: " + ", ".join(missing)
            )

    panel_path_raw = payload.get("panel_path")
    if panel_path_raw:
        panel_path = _resolve_path(str(panel_path_raw), base=input_path.parent)
        panel = _load_panel(panel_path)
        require_anchor_evidence(panel)
        return config, panel, panel_path

    explicit_analysis = payload.get("analysis_config")
    analysis_raw = explicit_analysis or config.get("analysis_config")
    if not analysis_raw:
        raise ValueError("Input requires panel_path or analysis_config")
    analysis_path = _resolve_path(
        str(analysis_raw), base=input_path.parent if explicit_analysis else REPO_ROOT
    )
    context = load_analysis_context(str(analysis_path))
    context.runtime_config = dict(context.runtime_config)
    context.runtime_config["start_date"] = str(
        payload.get("research_start", "2018-01-01")
    )
    context.runtime_config["end_date"] = str(payload.get("research_end", "2023-12-31"))
    context.runtime_config["data_snapshot_hash"] = str(
        payload.get("data_snapshot_hash") or ""
    )
    if pd.Timestamp(context.runtime_config["end_date"]) > MAX_RESEARCH_DATE:
        raise ValueError("QuantaAlpha realization may not load dates after 2023-12-31")
    anchor_db_path = payload.get("anchor_db_path")
    panel = build_training_panel(
        config,
        context,
        anchor_factor_names=anchors,
        anchor_db_path=str(_resolve_path(str(anchor_db_path), base=input_path.parent))
        if anchor_db_path
        else None,
        drop_missing_labels=False,
    )
    require_anchor_evidence(panel)
    return config, panel, None


def _safe_module_name(proposal_id: str, digest: str) -> str:
    stem = re.sub(r"[^0-9a-zA-Z_]+", "_", proposal_id).strip("_").lower() or "proposal"
    if stem[0].isdigit():
        stem = f"p_{stem}"
    return f"quanta_alpha_{stem}_{digest[:10]}"


def calibrate_funnel_top_k(
    rows: Iterable[Mapping[str, Any]], *, min_reduction: float = 0.3
) -> dict[str, Any]:
    """Choose the smallest sibling rank K retaining every historical pass."""

    records = [dict(row) for row in rows if bool(row.get("eligible", True))]
    known_passes = [row for row in records if bool(row.get("production_pass"))]
    if not records or not known_passes:
        return {
            "enabled": False,
            "top_k": None,
            "pass_recall": None,
            "reduction": 0.0,
            "eligible_count": len(records),
            "reason": "insufficient_pass_evidence",
        }
    diagnostics = []
    for top_k in (1, 2, 3):
        selected = [row for row in records if int(row.get("rank", 99)) <= top_k]
        retained = sum(int(row.get("rank", 99)) <= top_k for row in known_passes)
        recall = float(retained / len(known_passes))
        reduction = float((len(records) - len(selected)) / len(records))
        diagnostics.append((top_k, recall, reduction))
        if recall == 1.0 and reduction >= float(min_reduction):
            return {
                "enabled": True,
                "top_k": top_k,
                "pass_recall": recall,
                "reduction": reduction,
                "eligible_count": len(records),
                "reason": "calibrated",
            }
    best_k, best_recall, best_reduction = max(
        diagnostics, key=lambda item: (item[1], item[2], -item[0])
    )
    return {
        "enabled": False,
        "top_k": None,
        "best_top_k": best_k,
        "pass_recall": best_recall,
        "reduction": best_reduction,
        "eligible_count": len(records),
        "reason": "reduction_or_recall_target_not_met",
    }


def realize_proposals(
    input_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Compile and fast-screen exactly three hash-bound factor proposals."""

    source_input = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    payload = _load_json(source_input)
    proposals = payload.get("proposals")
    if (
        not isinstance(proposals, list)
        or len(proposals) != 3
        or any(not isinstance(item, dict) for item in proposals)
    ):
        raise ValueError(
            "QuantaAlpha realization requires exactly three proposal objects"
        )

    profile = str(payload.get("profile", "paper_v3_parity"))
    if profile not in PROFILES:
        raise ValueError(f"Unsupported QuantaAlpha profile: {profile}")
    top_k = 3 if profile == "paper_v3_parity" else int(payload.get("top_k", 0))
    if top_k not in {1, 2, 3}:
        raise ValueError("paper_v3_funnel requires top_k in {1,2,3}")

    train_start, train_end, train_count = _window(
        payload, "train", "2018-01-01", "2022-12-31"
    )
    valid_start, valid_end, valid_count = _window(
        payload, "validation", "2023-01-01", "2023-12-31"
    )
    if train_end >= valid_start:
        raise ValueError("Training window must end before validation starts")

    config, panel, panel_source = _load_config_and_panel(payload, source_input)
    required = {"date", "code", "label"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {sorted(missing)}")
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["date", "code"])
    panel = panel[panel["date"] <= MAX_RESEARCH_DATE].sort_values(["date", "code"])
    if panel.empty:
        raise ValueError("No pre-2024 panel rows available")

    # Labels select scoring rows only; AST evaluation keeps the full feature panel.
    scoring_panel = panel.dropna(subset=["label"])
    if profile == "paper_v3_parity":
        train_count = valid_count = 10**9
    train_dates = _sample_dates(scoring_panel, train_start, train_end, train_count)
    valid_dates = _sample_dates(scoring_panel, valid_start, valid_end, valid_count)
    train_panel = _partition(scoring_panel, train_dates)
    valid_panel = _partition(scoring_panel, valid_dates)
    shadow_funnel = payload.get("shadow_funnel")
    shadow_funnel = shadow_funnel if isinstance(shadow_funnel, Mapping) else {}
    funnel_train_count = int(shadow_funnel.get("train_sample_dates", 252))
    funnel_valid_count = int(shadow_funnel.get("validation_sample_dates", 126))
    if funnel_train_count <= 0 or funnel_valid_count <= 0:
        raise ValueError("shadow_funnel sample counts must be positive")
    funnel_train_dates = _sample_dates(
        scoring_panel, train_start, train_end, funnel_train_count
    )
    funnel_valid_dates = _sample_dates(
        scoring_panel, valid_start, valid_end, funnel_valid_count
    )
    funnel_train_panel = _partition(scoring_panel, funnel_train_dates)
    funnel_valid_panel = _partition(scoring_panel, funnel_valid_dates)
    if train_panel.empty or valid_panel.empty:
        raise ValueError("Sampled train or validation partition is empty")
    if funnel_train_panel.empty or funnel_valid_panel.empty:
        raise ValueError("Prospective funnel train or validation partition is empty")

    feature_space = build_feature_space(config)
    zoo_payload = payload.get("zoo_expressions", [])
    zoo_records: list[tuple[str, ExpressionNode]] = []
    for index, item in enumerate(zoo_payload):
        if not isinstance(item, dict):
            continue
        raw_expression = (
            item.get("expression") if isinstance(item.get("expression"), dict) else item
        )
        zoo_records.append(
            (
                str(item.get("proposal_id") or f"zoo_{index + 1}"),
                deserialize_expression(raw_expression),
            )
        )
    zoo = [expr for _, expr in zoo_records]
    seen_hashes = {canonical_expression_hash(item) for item in zoo}
    source_dir = _resolve_path(
        str(payload.get("output_dir", target.parent / "compiled")),
        base=source_input.parent,
    )
    source_dir.mkdir(parents=True, exist_ok=True)
    factor_value_cache_dir = source_dir / "factor_value_cache"
    panel_digest = _panel_hash(panel, panel_source)
    cache: dict[tuple[int, str], Any] = {}
    rows: list[dict[str, Any]] = []
    predictions: dict[str, pd.Series] = {}
    intraday_proposals: set[str] = set()
    prior_predictions: dict[str, pd.Series] = {}
    for prior_id, prior_expr in zoo_records:
        try:
            prior_predictions[prior_id] = evaluate_expression_cached(
                prior_expr, panel, cache
            )
        except Exception as exc:
            raise QuantaAlphaEvidenceError(
                f"Unable to evaluate prior campaign proposal {prior_id}: {exc}"
            ) from exc

    for attempt_index, proposal in enumerate(proposals, start=1):
        expr = _proposal_expression(proposal)
        proposal_id = str(proposal.get("proposal_id") or f"attempt_{attempt_index}")
        expr_hash = _stable_hash(serialize_expression(expr))
        report = structural_gate(
            expr,
            feature_space,
            factor_zoo=zoo,
            canonical_hashes=seen_hashes,
        )
        consistency = _consistency_gate(proposal, expr)
        row: dict[str, Any] = {
            "proposal_id": proposal_id,
            "trajectory_id": str(proposal.get("trajectory_id") or ""),
            "attempt_index": int(proposal.get("attempt_index") or attempt_index),
            "expression_hash": expr_hash,
            "canonical_hash": canonical_expression_hash(expr),
            "structural_gate": report,
            "consistency_gate": consistency,
            "source": None,
            "fast_metrics": None,
            "funnel_metrics": None,
            "output_correlations": {},
            "eligible": False,
            "rank": None,
        }
        if not report["passed"] or not consistency["passed"]:
            rows.append(row)
            continue

        pred = evaluate_expression_cached(expr, panel, cache)
        complexity = expression_complexity(expr)
        fast_metrics = _screen_metrics(pred, train_panel, valid_panel)
        fast_metrics["complexity"] = complexity
        funnel_metrics = _screen_metrics(
            pred,
            funnel_train_panel,
            funnel_valid_panel,
        )
        funnel_metrics["complexity"] = complexity
        row["fast_metrics"] = fast_metrics
        row["funnel_metrics"] = funnel_metrics
        required_metrics = (
            "train_rank_ic",
            "validation_rank_ic",
            "sampled_marginal_rank_ic",
            "coverage",
            "variance",
        )
        if any(
            _finite_float(metrics[name]) is None
            for metrics in (fast_metrics, funnel_metrics)
            for name in required_metrics
        ):
            raise QuantaAlphaEvidenceError(
                f"Incomplete fast evidence for {proposal_id}"
            )
        train_ic = float(fast_metrics["train_rank_ic"])
        direction = 1 if train_ic > 0 else -1 if train_ic < 0 else 0
        for metrics in (fast_metrics, funnel_metrics):
            metrics["frozen_direction"] = direction
            metrics["direction_adjusted_train_rank_ic"] = direction * float(
                metrics["train_rank_ic"]
            )
            metrics["direction_adjusted_validation_rank_ic"] = direction * float(
                metrics["validation_rank_ic"]
            )
            metrics["direction_adjusted_sampled_marginal_rank_ic"] = direction * float(
                metrics["sampled_marginal_rank_ic"]
            )
        fast_reasons = []
        if float(fast_metrics["variance"]) <= 1e-12:
            fast_reasons.append("zero_variance")
        if direction == 0:
            fast_reasons.append("train_direction_zero")
        elif direction * float(fast_metrics["validation_rank_ic"]) < 0:
            fast_reasons.append("train_validation_direction_mismatch")
        row["fast_gate"] = {"passed": not fast_reasons, "reasons": fast_reasons}
        if fast_reasons:
            rows.append(row)
            continue
        funnel_reasons = []
        if float(funnel_metrics["variance"]) <= 1e-12:
            funnel_reasons.append("zero_variance")
        if direction * float(funnel_metrics["validation_rank_ic"]) < 0:
            funnel_reasons.append("train_validation_direction_mismatch")
        funnel_passed = not funnel_reasons
        row["funnel_gate"] = {
            "passed": funnel_passed,
            "reasons": funnel_reasons,
        }
        if profile == "paper_v3_funnel" and not funnel_passed:
            rows.append(row)
            continue

        candidate = SearchCandidate(
            expression=expr,
            train_metrics={
                "rank_ic": float(fast_metrics["direction_adjusted_train_rank_ic"])
            },
            valid_metrics={
                "rank_ic": float(fast_metrics["direction_adjusted_validation_rank_ic"])
            },
            score=float(fast_metrics["direction_adjusted_sampled_marginal_rank_ic"]),
            complexity=complexity,
        )
        module_name = _safe_module_name(proposal_id, expr_hash)
        uses_intraday = any(
            feature_space.features[name].source == "price_5m"
            for name in referenced_features(expr)
        )
        factor_path = compile_factor_file(
            candidate,
            config,
            feature_space,
            source_dir / f"{module_name}.py",
            factor_value_cache_dir=factor_value_cache_dir if uses_intraday else None,
            date_level_pool_filter=True,
        )
        row["source"] = {
            "path": str(factor_path),
            "module_name": module_name,
            "sha256": _file_hash(factor_path),
        }
        row["eligible"] = True
        predictions[proposal_id] = pred
        if uses_intraday:
            intraday_proposals.add(proposal_id)
        seen_hashes.add(row["canonical_hash"])
        zoo.append(expr)
        rows.append(row)

    for row in rows:
        if not row["eligible"]:
            continue
        proposal_id = row["proposal_id"]
        peers = dict(prior_predictions)
        peers.update(
            {
                other_id: other_pred
                for other_id, other_pred in predictions.items()
                if other_id != proposal_id
            }
        )
        active_pair_corrs: list[float] = []
        funnel_pair_corrs: list[float] = []
        for other_id, other_pred in peers.items():
            funnel_corr = _mean_pair_rank_corr(
                predictions[proposal_id].loc[funnel_valid_panel.index],
                other_pred.loc[funnel_valid_panel.index],
                funnel_valid_panel["date"],
            )
            active_corr = _mean_pair_rank_corr(
                predictions[proposal_id].loc[valid_panel.index],
                other_pred.loc[valid_panel.index],
                valid_panel["date"],
            )
            if funnel_corr is None or active_corr is None:
                raise QuantaAlphaEvidenceError(
                    f"Output correlation unavailable for {proposal_id} vs {other_id}"
                )
            row["output_correlations"][other_id] = active_corr
            funnel_pair_corrs.append(funnel_corr)
            active_pair_corrs.append(active_corr)
        row["fast_metrics"]["max_abs_output_correlation"] = max(
            (abs(value) for value in active_pair_corrs), default=None
        )
        row["funnel_metrics"]["max_abs_output_correlation"] = max(
            (abs(value) for value in funnel_pair_corrs), default=None
        )

    ranked = sorted(
        (row for row in rows if row["eligible"]),
        key=_prospective_rank_key,
    )
    ranked.extend(
        sorted(
            (row for row in rows if not row["eligible"]),
            key=lambda row: str(row.get("expression_hash") or ""),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    selected = [row["proposal_id"] for row in ranked[:top_k]]
    for row in ranked[:top_k]:
        proposal_id = str(row["proposal_id"])
        if proposal_id not in intraday_proposals:
            continue
        source = row.get("source") or {}
        row["factor_value_cache"] = _write_factor_value_cache(
            factor_value_cache_dir,
            source_hash=str(source["sha256"]),
            panel=panel,
            prediction=predictions[proposal_id],
            panel_hash=panel_digest,
            start_date=str(train_start.date()),
            end_date=str(valid_end.date()),
        )
    result: dict[str, Any] = {
        "schema_version": "quanta_alpha_realization_v1",
        "profile": profile,
        "top_k": top_k,
        "input_hash": _file_hash(source_input),
        "panel_hash": panel_digest,
        "windows": {
            "train": {
                "start": str(train_start.date()),
                "end": str(train_end.date()),
                "sample_dates": len(train_dates),
            },
            "validation": {
                "start": str(valid_start.date()),
                "end": str(valid_end.date()),
                "sample_dates": len(valid_dates),
            },
            "shadow_funnel": {
                "train_sample_dates": len(funnel_train_dates),
                "validation_sample_dates": len(funnel_valid_dates),
            },
        },
        "proposals": rows,
        "selected_proposal_ids": selected,
        "sealed_returns": _write_sealed_returns(
            target.parent, payload=payload, config=config, input_path=source_input,
            panel=panel, panel_hash=panel_digest,
            is_start=str(train_start.date()), is_end=str(valid_end.date()),
        ),
    }
    result["output_hash"] = _stable_hash(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return result
