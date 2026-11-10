"""Hierarchical size-conditional ElasticNet synthesis.

The outer walk-forward remains owned by :mod:`run_factor_synthesis`.  This
module selects one model-plus-gate policy using only each outer window's
purged training history, then emits scores for the untouched prediction
window.  Portfolio evaluation remains on the existing VectorBT path.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, enet_path

from factor_synthesis.qlib_adapter.panel_purged_kfold import (
    PanelPurgedKFold,
    LabelIsolationError,
    resolve_label_exit_dates,
)

from .lasso_method import (
    InsufficientSurvivingFactorsError,
    _filter_by_missing_rate,
    _outer_prediction_index,
    _strict_label_exit_index,
)
from .preprocessing import apply_target_cs_preprocessing, compute_sample_weights


FIXED_BUCKET_EDGES = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
PREPARED_SNAPSHOT_SCHEMA = "size_conditional_prepared.v4"
TOP10_UTILITY_VERSION = "return_date_v2"
FIXED_GATE_CANDIDATES = (
    (0.0, 40.0),
    (20.0, 60.0),
    (40.0, 80.0),
    (60.0, 100.0),
    (0.0, 60.0),
    (20.0, 80.0),
    (40.0, 100.0),
    (0.0, 80.0),
    (20.0, 100.0),
    (0.0, 100.0),
)


def _finite_positive(values: Sequence[Any], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive finite values")
    return result


def validate_size_conditional_config(
    config: Mapping[str, Any], top_k_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and normalize the fail-closed method contract."""
    size = config.get("size_conditioning")
    if not isinstance(size, Mapping):
        raise ValueError("size_conditioning is required")

    edges = tuple(float(value) for value in size.get("bucket_edges", ()))
    if edges != FIXED_BUCKET_EDGES:
        raise ValueError(f"bucket_edges must equal {list(FIXED_BUCKET_EDGES)}")
    gates = tuple(
        tuple(float(value) for value in gate)
        for gate in size.get("gate_candidates", ())
    )
    if gates != FIXED_GATE_CANDIDATES:
        raise ValueError(
            "gate_candidates must equal the pre-registered contiguous gate grid"
        )

    target_period = int(config.get("target_period", 5))
    embargo_days = int(size.get("embargo_days", 0))
    if target_period < 1 or embargo_days < target_period:
        raise ValueError("embargo_days must be at least target_period")
    inner_embargo_observations = int(
        config.get("_label_inner_embargo_observations", embargo_days)
    )
    if inner_embargo_observations < 1:
        raise ValueError("inner label embargo observations must be positive")
    inner_folds = int(size.get("inner_folds", 0))
    if inner_folds < 2:
        raise ValueError("inner_folds must be >= 2")
    stability_threshold = float(size.get("stability_threshold", -1))
    if not 0 <= stability_threshold <= 1:
        raise ValueError("stability_threshold must be in [0, 1]")

    shrinkages = _finite_positive(
        size.get("interaction_shrinkage_grid", ()), "interaction_shrinkage_grid"
    )
    raw_alpha = _finite_positive(
        config.get("lasso_alpha_range", ()), "lasso_alpha_range"
    )
    n_alphas = int(config.get("n_alphas", len(raw_alpha)))
    if len(raw_alpha) == 2:
        if raw_alpha[0] >= raw_alpha[1] or n_alphas < 1:
            raise ValueError(
                "lasso_alpha_range must increase and n_alphas must be positive"
            )
        alpha_grid = tuple(
            float(value)
            for value in np.logspace(
                np.log10(raw_alpha[0]), np.log10(raw_alpha[1]), n_alphas
            )
        )
    else:
        alpha_grid = raw_alpha
    l1_grid = tuple(float(value) for value in config.get("l1_ratio_range", ()))
    if not l1_grid or any(not 0 < value <= 1 for value in l1_grid):
        raise ValueError("l1_ratio_range must contain values in (0, 1]")

    if not bool(top_k_config.get("enabled", False)):
        raise ValueError("TopK evaluation must be enabled")
    if int(top_k_config.get("top_k", 0)) != 10:
        raise ValueError("top_k must equal 10")
    if int(top_k_config.get("holding_period", 0)) != 1:
        raise ValueError("holding_period must equal 1")
    if not math.isclose(float(top_k_config.get("cost_rate", -1)), 0.001):
        raise ValueError("cost_rate must equal 0.001")
    if not bool(top_k_config.get("enable_tday_filter", False)):
        raise ValueError("enable_tday_filter must be true")

    return {
        "bucket_edges": edges,
        "gate_candidates": gates,
        "interaction_shrinkage_grid": shrinkages,
        "alpha_grid": alpha_grid,
        "l1_ratio_grid": l1_grid,
        "inner_folds": inner_folds,
        "embargo_days": embargo_days,
        "inner_embargo_observations": inner_embargo_observations,
        "stability_threshold": stability_threshold,
        "target_period": target_period,
        "top_k": 10,
        "cost_rate": 0.001,
        "max_iter": int(config.get("lasso_max_iter", 10000)),
        "tol": float(config.get("lasso_tol", 1e-4)),
        "random_state": int(size.get("random_state", config.get("random_state", 42))),
        "bootstrap_block_days": int(size.get("bootstrap_block_days", 5)),
        "bootstrap_samples": int(size.get("bootstrap_samples", 1000)),
    }


def daily_size_percentile(log_market_cap: pd.Series, index: pd.MultiIndex) -> pd.Series:
    """Return deterministic daily market-cap percentile ranks in ``[0, 100]``."""
    aligned = pd.to_numeric(log_market_cap.reindex(index), errors="coerce")
    level = "date" if "date" in index.names else 0
    ranked = (
        aligned.groupby(level=level, sort=False).rank(method="average", pct=True)
        * 100.0
    )
    ranked.name = "size_percentile"
    return ranked


def _database_cache_revision(path):
    from utils.data_revision import database_revision
    # Source lists may also include Parquet and frozen JSON evidence.
    if path.suffix.lower() == '.duckdb':
        return database_revision(path)
    return None


def size_conditional_snapshot_identity(
    root_config: Mapping[str, Any],
    method_config: Mapping[str, Any],
    factor_names: Sequence[str],
    start_date: str,
    end_date: str,
    source_paths: Sequence[str],
    source_markers: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the complete deterministic key payload for the prepared panel."""
    sources = []
    for raw_path in source_paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            stat = path.stat()
            sources.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "revision": _database_cache_revision(path),
                }
            )
        except OSError:
            sources.append({"path": str(path), "missing": True})
    target_keys = (
        "target_period",
        "target_cs_demean",
        "target_cs_rank",
        "target_cs_standardize",
    )
    preprocess = method_config.get("preprocess", method_config.get("preprocessing", {}))
    schema = PREPARED_SNAPSHOT_SCHEMA
    if preprocess and any(preprocess.get(name, preprocess.get("ALL", "none")) == "gauss_rank"
           for name in factor_names):
        from factor_synthesis.qlib_adapter.processors import PerFactorPreprocessor

        # Main's pre-versioned processor uses total-row-count normalization.
        version = getattr(PerFactorPreprocessor, "GAUSS_RANK_VERSION", "total-count-v1")
        schema += "/" + version
    label_producer = {
        "version": "strict_label_exit_v2",
        "effective_definition": method_config.get("_effective_label_definition", root_config.get("label_definition", {})),
        "effective_period": method_config.get("_label_effective_period"),
        "calendar_source": method_config.get("_label_calendar_source"),
    }
    mapping = method_config.get("_label_exit_dates")
    calendar = method_config.get("_label_trading_calendar")
    if mapping is not None or calendar is not None:
        days = list(mapping) if mapping is not None else pd.DatetimeIndex(calendar)
        if mapping is None:
            days = days[(days >= pd.Timestamp(start_date)) & (days <= pd.Timestamp(end_date))]
        keys = pd.MultiIndex.from_product([days, ["snapshot"]], names=["date", "stock_code"])
        context = _snapshot_label_context(keys, method_config)
        label_producer["exit_dates"] = context["_label_exit_dates"]
        label_producer["exit_dates_sha256"] = hashlib.sha256(json.dumps(
            context["_label_exit_dates"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
    return {
        "schema": schema,
        "factor_names": sorted(str(name) for name in factor_names),
        "start_date": start_date,
        "end_date": end_date,
        "stock_pool": root_config.get("stock_pool", {}),
        "calendar_filters": root_config.get("calendar_filters", {}),
        "label_definition": root_config.get("label_definition", {}),
        "pit": root_config.get("pit", root_config.get("pit_config", {})),
        "preprocess": method_config.get(
            "preprocess", method_config.get("preprocessing", {})
        ),
        "target": {key: method_config.get(key) for key in target_keys},
        "label_producer": label_producer,
        "neutralization": method_config.get("neutralization", {}),
        "sources": sources,
        "source_markers": dict(source_markers or {}),
    }


def _snapshot_label_context(index: pd.MultiIndex, context: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize proven exits so a cache hit needs no builder side effects."""
    if not isinstance(context, Mapping):
        raise LabelIsolationError("prepared snapshot is missing label context")
    exits = resolve_label_exit_dates(
        index, context.get("_label_exit_dates"),
        trading_calendar=context.get("_label_trading_calendar"),
        label_period=context.get("_label_effective_period"),
        calendar_source=context.get("_label_calendar_source"),
    )
    return {
        "_label_exit_dates": {
            str(pd.Timestamp(day).normalize().date()): str(exit_day.date())
            for day, exit_day in zip(index.get_level_values(0), exits)
        },
        "_label_trading_calendar": None,
        "_label_effective_period": context.get("_label_effective_period"),
        "_label_calendar_source": context.get("_label_calendar_source"),
        "_effective_label_definition": dict(context.get("_effective_label_definition", {})),
    }


def prepare_size_conditional_panel(
    synthesizer: Any,
    factor_df: pd.DataFrame,
    forward_returns: pd.Series,
    log_market_cap: pd.Series,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare label-eligible training and size-eligible prediction views."""
    preprocess = config.get("preprocess", config.get("preprocessing", {}))
    size = log_market_cap.reindex(factor_df.index)
    predict_raw = factor_df.loc[size.notna()]
    raw_returns = forward_returns.reindex(predict_raw.index)
    train_raw = predict_raw.loc[raw_returns.notna()]
    context = dict(config)
    if "label_exit_dates" in forward_returns.attrs:
        context["_label_exit_dates"] = forward_returns.attrs["label_exit_dates"]
    if "label_definition" in forward_returns.attrs:
        context["_effective_label_definition"] = forward_returns.attrs["label_definition"]
    label_context = _snapshot_label_context(train_raw.index, context)
    train_factors = (
        synthesizer._apply_feature_preprocessing(train_raw, preprocess)
        if preprocess
        else train_raw.copy()
    )
    predict_factors = (
        synthesizer._apply_feature_preprocessing(predict_raw, preprocess)
        if preprocess
        else predict_raw.copy()
    )
    fit_target = apply_target_cs_preprocessing(
        raw_returns.reindex(train_factors.index),
        cs_demean=bool(config.get("target_cs_demean", True)),
        cs_rank=bool(config.get("target_cs_rank", False)),
        cs_standardize=bool(config.get("target_cs_standardize", False)),
        logger=synthesizer.logger,
    )
    return {
        "label_context": label_context,
        "train_factors": train_factors,
        "predict_factors": predict_factors,
        "raw_returns": raw_returns,
        "fit_target": fit_target,
        "train_percentiles": daily_size_percentile(log_market_cap, train_factors.index),
        "predict_percentiles": daily_size_percentile(
            log_market_cap, predict_factors.index
        ),
    }


def _snapshot_frame(panel: Mapping[str, Any]) -> pd.DataFrame:
    train_factors = panel["train_factors"]
    predict_factors = panel["predict_factors"]
    reserved_prefixes = ("__train__", "__predict__", "__value__")
    if any(
        str(column).startswith(reserved_prefixes) for column in predict_factors.columns
    ):
        raise ValueError("factor names collide with prepared snapshot columns")
    frame = predict_factors.add_prefix("__predict__").join(
        train_factors.add_prefix("__train__"), how="outer"
    )
    frame["__value__train_member"] = frame.index.isin(train_factors.index)
    frame["__value__predict_member"] = frame.index.isin(predict_factors.index)
    frame["__value__train_order"] = pd.Series(np.arange(len(train_factors)), index=train_factors.index)
    frame["__value__predict_order"] = pd.Series(np.arange(len(predict_factors)), index=predict_factors.index)
    frame["__value__raw_return"] = panel["raw_returns"].reindex(frame.index)
    frame["__value__fit_target"] = panel["fit_target"].reindex(frame.index)
    frame["__value__train_percentile"] = panel["train_percentiles"].reindex(frame.index)
    frame["__value__predict_percentile"] = panel["predict_percentiles"].reindex(
        frame.index
    )
    return frame


def load_or_build_size_conditional_snapshot(
    cache_root: Path,
    identity: Mapping[str, Any],
    builder: Callable[[], Mapping[str, Any]],
    *,
    enabled: bool,
    logger: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild I/O corruption; label-isolation failures always propagate."""
    if not enabled:
        panel = dict(builder())
        panel["label_context"] = _snapshot_label_context(panel["train_factors"].index, panel.get("label_context"))
        return panel, {"status": "disabled"}
    if identity.get("schema", PREPARED_SNAPSHOT_SCHEMA).split("/", 1)[0] != PREPARED_SNAPSHOT_SCHEMA:
        return dict(builder()), {"status": "miss", "reason": "old_schema"}
    if not identity.get("label_producer") or (
        not identity["label_producer"].get("exit_dates_sha256")
        and (not identity.get("sources") or any(s.get("missing") for s in identity["sources"]))
    ):
        panel = dict(builder())
        panel["label_context"] = _snapshot_label_context(panel["train_factors"].index, panel.get("label_context"))
        return panel, {"status": "miss", "reason": "unverified_label_source"}
    canonical = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    cache_dir = Path(cache_root).expanduser() / PREPARED_SNAPSHOT_SCHEMA
    parquet_path = cache_dir / f"{key}.parquet"
    metadata_path = cache_dir / f"{key}.json"
    if parquet_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema") != PREPARED_SNAPSHOT_SCHEMA:
                return dict(builder()), {"status": "miss", "reason": "old_schema"}
            if metadata.get("key") != key or metadata.get("identity") != identity:
                raise ValueError("prepared snapshot metadata mismatch")
            frame = pd.read_parquet(parquet_path)
            factor_columns = metadata["factor_columns"]
            panel = {
                "train_factors": frame.loc[frame["__value__train_member"],
                    [f"__train__{column}" for column in factor_columns]
                ]
                .rename(columns=lambda column: column.removeprefix("__train__")),
                "predict_factors": frame.loc[frame["__value__predict_member"],
                    [f"__predict__{column}" for column in factor_columns]
                ].rename(columns=lambda column: column.removeprefix("__predict__")),
                "raw_returns": frame["__value__raw_return"],
                "fit_target": frame["__value__fit_target"],
                "train_percentiles": frame["__value__train_percentile"],
                "predict_percentiles": frame["__value__predict_percentile"],
            }
            for role in ("train", "predict"):
                index = frame.loc[frame[f"__value__{role}_member"]].sort_values(
                    f"__value__{role}_order", kind="stable"
                ).index
                panel[f"{role}_factors"] = panel[f"{role}_factors"].reindex(index)
            for key_name, role in (("raw_returns", "predict"), ("fit_target", "train"),
                                   ("train_percentiles", "train"), ("predict_percentiles", "predict")):
                panel[key_name] = panel[key_name].reindex(panel[f"{role}_factors"].index)
                panel[key_name].name = metadata["panel_metadata"][key_name]["name"]
            for key_name, value in panel.items():
                value.attrs = metadata["panel_metadata"][key_name]["attrs"]
            context = metadata.get("label_context")
            context_hash = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if context is None or context_hash != metadata.get("label_context_sha256"):
                raise LabelIsolationError("prepared snapshot label context digest mismatch")
            panel["label_context"] = _snapshot_label_context(panel["train_factors"].index, context)
            expected = identity["label_producer"].get("exit_dates")
            if expected is not None and any(expected.get(day) != exit_day for day, exit_day in panel["label_context"]["_label_exit_dates"].items()):
                raise LabelIsolationError("prepared snapshot exits differ from current producer")
            if logger:
                logger.info(
                    "[SIZE_PREPARED_CACHE] hit=%s rows=%d", parquet_path, len(frame)
                )
            return panel, {
                "status": "hit",
                "key": key,
                "parquet_path": str(parquet_path),
                "metadata_path": str(metadata_path),
            }
        except LabelIsolationError:
            raise
        except Exception as exc:
            if logger:
                logger.warning(
                    "[SIZE_PREPARED_CACHE] invalid snapshot, rebuilding: %s", exc
                )

    panel = dict(builder())
    panel["label_context"] = _snapshot_label_context(panel["train_factors"].index, panel.get("label_context"))
    expected = identity["label_producer"].get("exit_dates")
    if expected is not None and any(expected.get(day) != exit_day for day, exit_day in panel["label_context"]["_label_exit_dates"].items()):
        raise LabelIsolationError("prepared builder exits differ from current producer")
    frame = _snapshot_frame(panel)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": PREPARED_SNAPSHOT_SCHEMA,
        "key": key,
        "identity": identity,
        "factor_columns": list(panel["predict_factors"].columns),
        "row_count": len(frame),
        "label_context": panel["label_context"],
        "label_context_sha256": hashlib.sha256(json.dumps(
            panel["label_context"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "panel_metadata": {name: {"name": getattr(value, "name", None), "attrs": value.attrs}
                           for name, value in panel.items() if name != "label_context"},
    }
    parquet_handle = tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=cache_dir, delete=False
    )
    metadata_handle = tempfile.NamedTemporaryFile(
        suffix=".json", dir=cache_dir, delete=False
    )
    parquet_temp, metadata_temp = parquet_handle.name, metadata_handle.name
    parquet_handle.close()
    metadata_handle.close()
    try:
        frame.to_parquet(parquet_temp)
        Path(metadata_temp).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(parquet_temp, parquet_path)
        os.replace(metadata_temp, metadata_path)
    except Exception as exc:
        for temporary in (parquet_temp, metadata_temp):
            try:
                os.unlink(temporary)
            except OSError:
                pass
        if isinstance(exc, LabelIsolationError):
            raise
        if logger:
            logger.warning("[SIZE_PREPARED_CACHE] write failed, continuing: %s", exc)
        return panel, {"status": "miss-write-failed", "key": key}
    if logger:
        logger.info(
            "[SIZE_PREPARED_CACHE] miss-write=%s rows=%d", parquet_path, len(frame)
        )
    return panel, {
        "status": "miss",
        "key": key,
        "parquet_path": str(parquet_path),
        "metadata_path": str(metadata_path),
    }


def gate_mask(percentiles: pd.Series, gate: tuple[float, float]) -> np.ndarray:
    """Return the disjoint ``[low, high)`` gate mask (100 is inclusive)."""
    low, high = gate
    values = percentiles.to_numpy(dtype=float, copy=False)
    upper = values <= high if high == 100 else values < high
    return np.isfinite(values) & (values >= low) & upper


def _bucket_ids(percentiles: pd.Series, edges: Sequence[float]) -> np.ndarray:
    values = percentiles.to_numpy(dtype=float, copy=False)
    ids = np.searchsorted(np.asarray(edges, dtype=float), values, side="right") - 1
    ids[values == edges[-1]] = len(edges) - 2
    invalid = ~np.isfinite(values) | (values < edges[0]) | (values > edges[-1])
    ids[invalid] = -1
    return ids


def build_effect_coded_features(
    factors: pd.DataFrame,
    percentiles: pd.Series,
    bucket_edges: Sequence[float],
    interaction_shrinkage: float,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build global factors plus four sum-to-zero size-deviation contrasts."""
    if interaction_shrinkage <= 0 or not math.isfinite(interaction_shrinkage):
        raise ValueError("interaction_shrinkage must be positive and finite")
    if len(bucket_edges) < 3:
        raise ValueError("at least two size buckets are required")
    aligned_percentiles = percentiles.reindex(factors.index)
    bucket_ids = _bucket_ids(aligned_percentiles, bucket_edges)
    if (bucket_ids < 0).any():
        raise ValueError("market-cap percentile is missing or outside bucket edges")

    global_values = factors.to_numpy(dtype=np.float32, copy=True)
    global_values[~np.isfinite(global_values)] = 0.0
    n_rows, n_factors = global_values.shape
    n_buckets = len(bucket_edges) - 1
    reference_bucket = n_buckets - 1
    interactions = np.zeros((n_rows, n_factors * (n_buckets - 1)), dtype=np.float32)
    interaction_names: list[str] = []
    for bucket in range(reference_bucket):
        contrast = np.zeros(n_rows, dtype=np.float32)
        contrast[bucket_ids == bucket] = 1.0
        contrast[bucket_ids == reference_bucket] = -1.0
        block = global_values * contrast[:, None] / np.float32(interaction_shrinkage)
        start = bucket * n_factors
        interactions[:, start : start + n_factors] = block
        low, high = bucket_edges[bucket], bucket_edges[bucket + 1]
        interaction_names.extend(
            f"{factor}::size_dev_{low:g}_{high:g}" for factor in factors.columns
        )

    feature_names = [str(column) for column in factors.columns] + interaction_names
    metadata = {
        "global_feature_count": n_factors,
        "interaction_feature_count": interactions.shape[1],
        "reference_bucket": reference_bucket,
        "interaction_shrinkage": float(interaction_shrinkage),
        "coefficient_rescale": 1.0 / float(interaction_shrinkage),
        "bucket_edges": [float(value) for value in bucket_edges],
    }
    return (
        np.concatenate((global_values, interactions), axis=1),
        feature_names,
        metadata,
    )


def prepare_top10_context(
    raw_returns: np.ndarray, index: pd.MultiIndex, *, top_k: int
) -> dict[str, Any]:
    """Prepare immutable weekly row groups for repeated TopK scoring."""
    values = np.asarray(raw_returns, dtype=float)
    if len(values) != len(index):
        raise ValueError("raw_returns and index lengths differ")
    dates = pd.DatetimeIndex(index.get_level_values(0))
    stocks = index.get_level_values(1).astype(str).to_numpy()
    stock_codes = np.unique(stocks)
    stock_ids = np.searchsorted(stock_codes, stocks)
    valid_return = np.isfinite(values)
    weeks = dates.to_period("W-SUN")

    rebalance_dates: list[np.datetime64] = []
    row_positions: list[np.ndarray] = []
    for week in weeks.unique().sort_values():
        week_positions = np.flatnonzero((weeks == week) & valid_return)
        if not len(week_positions):
            continue
        date = dates.to_numpy()[week_positions].min()
        positions = week_positions[dates.to_numpy()[week_positions] == date]
        rebalance_dates.append(np.datetime64(date))
        row_positions.append(positions)
    return {
        "rebalance_dates": np.asarray(rebalance_dates, dtype="datetime64[ns]"),
        "row_positions": tuple(row_positions),
        "stock_ids": stock_ids.astype(np.int32, copy=False),
        "returns": values,
        "row_count": len(index),
        "top_k": int(top_k),
    }


def _topk_exact(scores: np.ndarray, stock_ids: np.ndarray, top_k: int) -> np.ndarray:
    """Argpartition TopK with the exact ``(-score, stock_code)`` cutoff rule."""
    if len(scores) < top_k:
        return np.empty(0, dtype=np.intp)
    threshold_index = len(scores) - top_k
    threshold = scores[np.argpartition(scores, threshold_index)[threshold_index]]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    remaining = top_k - len(above)
    if remaining:
        tied = tied[np.argsort(stock_ids[tied], kind="stable")[:remaining]]
    return np.concatenate((above, tied))


def _score_prepared_top10(
    prediction: np.ndarray, context: Mapping[str, Any], *, cost_rate: float
) -> tuple[float, pd.Series, float]:
    prediction = np.asarray(prediction, dtype=float)
    if len(prediction) != int(context["row_count"]):
        raise ValueError("prediction and prepared context lengths differ")
    top_k = int(context["top_k"])
    previous: Optional[np.ndarray] = None
    net_returns: dict[pd.Timestamp, float] = {}
    turnovers: list[float] = []
    for date, positions in zip(context["rebalance_dates"], context["row_positions"]):
        finite = np.isfinite(prediction[positions])
        eligible = positions[finite]
        if len(eligible) < top_k:
            continue
        selected_local = _topk_exact(
            prediction[eligible], context["stock_ids"][eligible], top_k
        )
        selected = eligible[selected_local]
        selected_stock_ids = context["stock_ids"][selected]
        turnover = (
            0.5
            if previous is None
            else 1.0
            - len(np.intersect1d(previous, selected_stock_ids, assume_unique=True))
            / top_k
        )
        gross = float(context["returns"][selected].mean())
        net_returns[pd.Timestamp(date)] = gross - cost_rate * turnover
        turnovers.append(turnover)
        previous = selected_stock_ids

    series = pd.Series(net_returns, name="inner_weekly_net_return").sort_index()
    standard_deviation = float(series.std(ddof=1)) if len(series) >= 2 else math.nan
    if (
        len(series) < 2
        or not math.isfinite(standard_deviation)
        or standard_deviation == 0
    ):
        return -math.inf, series, float(np.mean(turnovers)) if turnovers else math.nan
    sharpe = float(series.mean() / standard_deviation * math.sqrt(52.0))
    return sharpe, series, float(np.mean(turnovers))


def top10_net_utility(
    prediction: np.ndarray,
    raw_returns: np.ndarray,
    index: pd.MultiIndex,
    *,
    top_k: int,
    cost_rate: float,
    target_period: int,
    prepared_context: Optional[Mapping[str, Any]] = None,
) -> tuple[float, pd.Series, float]:
    """Compatibility wrapper for exact prepared weekly TopK utility."""
    del target_period  # The weekly rebalance cadence determines annualization.
    prediction = np.asarray(prediction, dtype=float)
    raw_returns = np.asarray(raw_returns, dtype=float)
    if prepared_context is None:
        prepared_context = prepare_top10_context(raw_returns, index, top_k=top_k)
    sharpe, weekly, turnover = _score_prepared_top10(prediction, prepared_context, cost_rate=cost_rate)
    weekly.attrs["top10_utility_version"] = TOP10_UTILITY_VERSION
    return sharpe, weekly, turnover


def _fit_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    sample_weight: Optional[np.ndarray],
    max_iter: int,
    tol: float,
) -> ElasticNet:
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iter,
        tol=tol,
        random_state=42,
        selection="cyclic",
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def prepare_elasticnet_path(
    X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]
) -> dict[str, Any]:
    """Weighted-center once and precompute Gram/Xy for all l1-ratio paths."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if sample_weight is None:
        X_offset = X.mean(axis=0)
        y_offset = float(y.mean())
        X_work = X - X_offset
        y_work = y - y_offset
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if (
            len(weights) != len(y)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0)
        ):
            raise ValueError(
                "sample_weight must be finite, nonnegative, and row-aligned"
            )
        if weights.sum() <= 0:
            raise ValueError("sample_weight must have positive total weight")
        weights = weights * (len(weights) / weights.sum())
        X_offset = np.average(X, axis=0, weights=weights)
        y_offset = float(np.average(y, weights=weights))
        scale = np.sqrt(weights)
        X_work = (X - X_offset) * scale[:, None]
        y_work = (y - y_offset) * scale
    X_work = np.asfortranarray(X_work)
    return {
        "X": X_work,
        "y": y_work,
        "X_offset": X_offset,
        "y_offset": y_offset,
        "gram": np.asarray(X_work.T @ X_work, order="C"),
        "Xy": X_work.T @ y_work,
    }


def solve_elasticnet_path(
    prepared: Mapping[str, Any],
    *,
    alphas: Sequence[float],
    l1_ratio: float,
    max_iter: int,
    tol: float,
) -> dict[str, Any]:
    """Solve one descending alpha path from a shared centered Gram problem."""
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    requested = np.asarray(alphas, dtype=np.float64)
    descending = np.sort(requested)[::-1]
    started = time.perf_counter()
    solved_alphas, coefficients, dual_gaps, n_iter = enet_path(
        prepared["X"],
        prepared["y"],
        l1_ratio=l1_ratio,
        alphas=descending,
        precompute=prepared["gram"],
        Xy=prepared["Xy"],
        copy_X=False,
        return_n_iter=True,
        check_input=False,
        max_iter=max_iter,
        tol=tol,
        selection="cyclic",
    )
    elapsed = time.perf_counter() - started
    coefficients = coefficients.T
    intercepts = np.asarray(
        [
            prepared["y_offset"] - prepared["X_offset"] @ coefficient
            for coefficient in coefficients
        ]
    )
    return {
        "alphas": solved_alphas,
        "coefficients": coefficients,
        "intercepts": intercepts,
        "dual_gaps": np.asarray(dual_gaps),
        "n_iter": np.asarray(n_iter),
        "elapsed_sec": elapsed,
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _policy_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    gate = tuple(record["gate"])
    return (
        -float(record["mean_score"]),
        0 if gate == (0.0, 100.0) else 1,
        -float(record["interaction_shrinkage"]),
        -float(record["alpha"]),
        -float(record["l1_ratio"]),
        json.dumps(record["policy"], sort_keys=True),
    )


def select_inner_policy(
    factors: pd.DataFrame,
    fit_target: pd.Series,
    raw_returns: pd.Series,
    percentiles: pd.Series,
    sample_weight: Optional[np.ndarray],
    normalized_config: Mapping[str, Any],
    *,
    label_context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Select exactly one policy without receiving outer-window labels."""
    common = factors.index.intersection(fit_target.dropna().index)
    common = common.intersection(raw_returns.dropna().index)
    common = common.intersection(percentiles.dropna().index)
    if len(common) == 0:
        raise ValueError("no aligned rows for inner selection")
    if sample_weight is not None:
        weights = pd.Series(sample_weight, index=factors.index).reindex(common).to_numpy(dtype=float)
    else:
        weights = None
    factors = factors.reindex(common).fillna(0.0)
    fit_target = fit_target.reindex(common)
    raw_returns = raw_returns.reindex(common)
    percentiles = percentiles.reindex(common)
    context = label_context or {}
    exits = resolve_label_exit_dates(
        common, context.get("_label_exit_dates"),
        trading_calendar=context.get("_label_trading_calendar"),
        label_period=context.get("_label_effective_period"),
        calendar_source=context.get("_label_calendar_source"),
    )
    splitter = PanelPurgedKFold(
        n_splits=int(normalized_config["inner_folds"]),
        embargo_td=int(
            normalized_config.get(
                "inner_embargo_observations", normalized_config["embargo_days"]
            )
        ),
    )
    groups = pd.DatetimeIndex(common.get_level_values(0)).to_numpy()
    splits = list(splitter.split(
        np.empty((len(common), 0)), groups=groups, label_exit_dates=exits.to_numpy()
    ))
    if len(splits) < 2:
        raise ValueError("fewer than two valid purged inner folds")

    trials: list[dict[str, Any]] = []
    performance = {
        "validation_context_prepare_count": 0,
        "gram_prepare_count": 0,
        "path_solve_count": 0,
        "path_elapsed_sec": 0.0,
        "scoring_elapsed_sec": 0.0,
    }
    for gate in normalized_config["gate_candidates"]:
        gate_folds: list[dict[str, Any]] = []
        valid_fold_data = True
        for train_idx, validation_idx in splits:
            train_gate = gate_mask(percentiles.iloc[train_idx], gate)
            validation_gate = gate_mask(percentiles.iloc[validation_idx], gate)
            if (
                train_gate.sum() < 20
                or validation_gate.sum() < normalized_config["top_k"]
            ):
                valid_fold_data = False
                break
            train_rows = train_idx[train_gate]
            validation_rows = validation_idx[validation_gate]
            validation_return = raw_returns.iloc[validation_rows].to_numpy(dtype=float)
            validation_index = factors.index[validation_rows]
            gate_folds.append(
                {
                    "train_rows": train_rows,
                    "validation_rows": validation_rows,
                    "train_target": fit_target.iloc[train_rows].to_numpy(dtype=float),
                    "train_weight": weights[train_rows]
                    if weights is not None
                    else None,
                    "validation_context": prepare_top10_context(
                        validation_return,
                        validation_index,
                        top_k=int(normalized_config["top_k"]),
                    ),
                }
            )
            performance["validation_context_prepare_count"] += 1

        for shrinkage in normalized_config["interaction_shrinkage_grid"]:
            solved_folds: list[dict[float, dict[str, Any]]] = []
            if valid_fold_data:
                for fold in gate_folds:
                    X_train, _, _ = build_effect_coded_features(
                        factors.iloc[fold["train_rows"]],
                        percentiles.iloc[fold["train_rows"]],
                        normalized_config["bucket_edges"],
                        shrinkage,
                    )
                    X_validation, _, _ = build_effect_coded_features(
                        factors.iloc[fold["validation_rows"]],
                        percentiles.iloc[fold["validation_rows"]],
                        normalized_config["bucket_edges"],
                        shrinkage,
                    )
                    path_input = prepare_elasticnet_path(
                        X_train, fold["train_target"], fold["train_weight"]
                    )
                    performance["gram_prepare_count"] += 1
                    paths: dict[float, dict[str, Any]] = {}
                    for l1_ratio in normalized_config["l1_ratio_grid"]:
                        # A warm-start multi-alpha path changes the finite-tolerance
                        # solution at tol=1e-4.  Single-alpha paths retain exact
                        # legacy coefficients while sharing centering, Gram and Xy.
                        alpha_solutions = [
                            solve_elasticnet_path(
                                path_input,
                                alphas=[alpha],
                                l1_ratio=float(l1_ratio),
                                max_iter=int(normalized_config["max_iter"]),
                                tol=float(normalized_config["tol"]),
                            )
                            for alpha in normalized_config["alpha_grid"]
                        ]
                        solved = {
                            "alphas": np.concatenate(
                                [item["alphas"] for item in alpha_solutions]
                            ),
                            "coefficients": np.concatenate(
                                [item["coefficients"] for item in alpha_solutions]
                            ),
                            "intercepts": np.concatenate(
                                [item["intercepts"] for item in alpha_solutions]
                            ),
                            "dual_gaps": np.concatenate(
                                [item["dual_gaps"] for item in alpha_solutions]
                            ),
                            "n_iter": np.concatenate(
                                [item["n_iter"] for item in alpha_solutions]
                            ),
                            "elapsed_sec": sum(
                                item["elapsed_sec"] for item in alpha_solutions
                            ),
                        }
                        solved["predictions"] = (
                            X_validation @ solved["coefficients"].T
                            + solved["intercepts"]
                        ).T
                        paths[float(l1_ratio)] = solved
                        performance["path_solve_count"] += len(alpha_solutions)
                        performance["path_elapsed_sec"] += solved["elapsed_sec"]
                    solved_folds.append(paths)

            for alpha in normalized_config["alpha_grid"]:
                for l1_ratio in normalized_config["l1_ratio_grid"]:
                    fold_scores: list[float] = []
                    fold_turnover: list[float] = []
                    path_diagnostics: list[dict[str, Any]] = []
                    rejection_reason = None
                    if valid_fold_data:
                        for fold, paths in zip(gate_folds, solved_folds):
                            solved = paths[float(l1_ratio)]
                            alpha_position = int(
                                np.flatnonzero(
                                    np.isclose(solved["alphas"], alpha, rtol=0, atol=0)
                                )[0]
                            )
                            if not np.any(
                                np.abs(solved["coefficients"][alpha_position]) > 1e-12
                            ):
                                # A constant intercept has no cross-sectional
                                # ranking content. Its lexical TopK tie-break
                                # must never compete as a factor policy.
                                fold_scores = []
                                rejection_reason = "all_zero_inner_fit"
                                break
                            scoring_started = time.perf_counter()
                            score, _, turnover = top10_net_utility(
                                solved["predictions"][alpha_position],
                                fold["validation_context"]["returns"],
                                factors.index[fold["validation_rows"]],
                                top_k=int(normalized_config["top_k"]),
                                cost_rate=float(normalized_config["cost_rate"]),
                                target_period=int(normalized_config["target_period"]),
                                prepared_context=fold["validation_context"],
                            )
                            performance["scoring_elapsed_sec"] += (
                                time.perf_counter() - scoring_started
                            )
                            if not math.isfinite(score):
                                fold_scores = []
                                rejection_reason = "non_finite_inner_utility"
                                break
                            fold_scores.append(score)
                            fold_turnover.append(turnover)
                            path_diagnostics.append(
                                {
                                    "elapsed_sec": solved["elapsed_sec"],
                                    "n_iter": int(solved["n_iter"][alpha_position]),
                                    "dual_gap": float(
                                        solved["dual_gaps"][alpha_position]
                                    ),
                                }
                            )
                    policy = {
                        "gate": [float(value) for value in gate],
                        "interaction_shrinkage": float(shrinkage),
                        "alpha": float(alpha),
                        "l1_ratio": float(l1_ratio),
                    }
                    trials.append(
                        {
                            "policy": policy,
                            **policy,
                            "fold_scores": fold_scores,
                            "mean_score": float(np.mean(fold_scores))
                            if fold_scores
                            else -math.inf,
                            "mean_turnover": float(np.mean(fold_turnover))
                            if fold_turnover
                            else None,
                            "valid_fold_count": len(fold_scores),
                            "path_diagnostics": path_diagnostics,
                            "rejection_reason": rejection_reason,
                        }
                    )

    valid_trials = [trial for trial in trials if math.isfinite(trial["mean_score"])]
    if not valid_trials:
        rejected_all_zero = sum(
            trial["rejection_reason"] == "all_zero_inner_fit" for trial in trials
        )
        raise ValueError(
            "no valid inner model-plus-gate candidate "
            f"(all_zero_inner_fit={rejected_all_zero}/{len(trials)})"
        )
    selected = min(valid_trials, key=_policy_key)
    selected_gate = tuple(selected["gate"])
    selected_shrinkage = float(selected["interaction_shrinkage"])

    support_count: Counter[str] = Counter()
    sign_count: dict[str, Counter[int]] = defaultdict(Counter)
    feature_names: list[str] = []
    for train_idx, _ in splits:
        train_gate = gate_mask(percentiles.iloc[train_idx], selected_gate)
        train_rows = train_idx[train_gate]
        X_train, feature_names, _ = build_effect_coded_features(
            factors.iloc[train_rows],
            percentiles.iloc[train_rows],
            normalized_config["bucket_edges"],
            selected_shrinkage,
        )
        model = _fit_elasticnet(
            X_train,
            fit_target.iloc[train_rows].to_numpy(dtype=float),
            alpha=float(selected["alpha"]),
            l1_ratio=float(selected["l1_ratio"]),
            sample_weight=weights[train_rows] if weights is not None else None,
            max_iter=int(normalized_config["max_iter"]),
            tol=float(normalized_config["tol"]),
        )
        for name, coefficient in zip(feature_names, model.coef_):
            if abs(float(coefficient)) > 1e-12:
                support_count[name] += 1
                sign_count[name][1 if coefficient > 0 else -1] += 1

    threshold = float(normalized_config["stability_threshold"])
    stable = {
        name for name in feature_names if support_count[name] / len(splits) >= threshold
    }
    for name in list(stable):
        if "::size_dev_" in name:
            stable.add(name.split("::", 1)[0])
    stable_support = [name for name in feature_names if name in stable]

    stability = {
        name: {
            "selection_frequency": support_count[name] / len(splits),
            "direction_agreement": (
                max(sign_count[name].values()) / sum(sign_count[name].values())
                if sign_count[name]
                else 0.0
            ),
        }
        for name in feature_names
    }
    scores = np.asarray(selected["fold_scores"], dtype=float)
    standard_error = (
        float(scores.std(ddof=1) / math.sqrt(len(scores)))
        if len(scores) > 1
        else math.inf
    )
    z_score = (
        float(scores.mean() / standard_error)
        if standard_error > 0 and math.isfinite(standard_error)
        else 0.0
    )
    trial_count = len(trials)
    corrected_z = z_score - math.sqrt(2.0 * math.log(max(trial_count, 1)))
    dsr = {
        "method": "gaussian_dsr_style_inner_approximation",
        "candidate_count": trial_count,
        "selected_mean_score": float(scores.mean()),
        "selected_score_standard_error": standard_error
        if math.isfinite(standard_error)
        else None,
        "uncorrected_z": z_score,
        "corrected_z": corrected_z,
        "probability": _normal_cdf(corrected_z),
    }
    return {
        "selected_policy": selected["policy"],
        "trials": trials,
        "stable_support": stable_support,
        "stability_status": "stable_support_available"
        if stable_support
        else "no_stable_support",
        "stability": stability,
        "dsr": dsr,
        "performance": performance,
        "inner_folds": [
            {
                "train_start": str(pd.Timestamp(groups[train_idx].min()).date()),
                "train_end": str(pd.Timestamp(groups[train_idx].max()).date()),
                "validation_start": str(
                    pd.Timestamp(groups[validation_idx].min()).date()
                ),
                "validation_end": str(
                    pd.Timestamp(groups[validation_idx].max()).date()
                ),
            }
            for train_idx, validation_idx in splits
        ],
    }


def _coefficient_payload(
    feature_names: Sequence[str],
    coefficients: np.ndarray,
    factor_names: Sequence[str],
    bucket_edges: Sequence[float],
    shrinkage: float,
) -> dict[str, Any]:
    raw = dict(zip(feature_names, (float(value) for value in coefficients)))
    global_coefficients = {name: raw.get(name, 0.0) for name in factor_names}
    deviations: dict[str, dict[str, float]] = {}
    effective: dict[str, dict[str, float]] = {}
    reference_label = f"{bucket_edges[-2]:g}_{bucket_edges[-1]:g}"
    for factor in factor_names:
        by_bucket: dict[str, float] = {}
        for bucket in range(len(bucket_edges) - 2):
            label = f"{bucket_edges[bucket]:g}_{bucket_edges[bucket + 1]:g}"
            by_bucket[label] = raw.get(f"{factor}::size_dev_{label}", 0.0) / shrinkage
        by_bucket[reference_label] = -sum(by_bucket.values())
        deviations[factor] = by_bucket
        effective[factor] = {
            label: global_coefficients[factor] + delta
            for label, delta in by_bucket.items()
        }
    return {
        "global_coefficients": global_coefficients,
        "bucket_deviations": deviations,
        "effective_bucket_coefficients": effective,
    }


def run_size_conditional_synthesis(
    synthesizer: Any,
    factor_df: pd.DataFrame,
    forward_returns: pd.Series,
    log_market_cap: pd.Series,
    train_start: str,
    train_end: str,
    config: dict[str, Any],
    daily_pools: Optional[Mapping[str, set[str]]] = None,
    prepared_panel: Optional[Mapping[str, Any]] = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fit one nested-selected policy and score the next outer window."""
    logger = synthesizer.logger
    top_k_config = synthesizer.config.get("performance_evaluation", {}).get(
        "top_k_strategy", {}
    )
    normalized = validate_size_conditional_config(config, top_k_config)
    if prepared_panel is not None:
        config = dict(config)
        config.update(_snapshot_label_context(prepared_panel["train_factors"].index, prepared_panel.get("label_context")))
        factor_df = prepared_panel["predict_factors"]
        forward_returns = prepared_panel["raw_returns"]
    train_start_dt = pd.Timestamp(train_start)
    train_end_dt = pd.Timestamp(train_end)
    predict_index = _outer_prediction_index(
        synthesizer, factor_df, train_end_dt, daily_pools, logger
    )
    predict_size_values = (
        prepared_panel["predict_percentiles"]
        if prepared_panel is not None
        else log_market_cap
    )
    predict_index = predict_index.intersection(predict_size_values.dropna().index)
    if len(predict_index) == 0:
        raise ValueError("no eligible size-conditional outer prediction rows")
    boundary = predict_index.get_level_values("date").min()
    training_names = config.get("_training_factor_names") or list(factor_df.columns)
    available = [name for name in training_names if name in factor_df.columns]
    train_source = (
        prepared_panel["train_factors"] if prepared_panel is not None else factor_df
    )
    train_dates = pd.DatetimeIndex(train_source.index.get_level_values("date"))
    train_mask = (train_dates >= train_start_dt) & (train_dates <= train_end_dt)
    X_train_raw = train_source.loc[train_mask, available]
    common = X_train_raw.index.intersection(forward_returns.dropna().index)
    train_size_values = (
        prepared_panel["train_percentiles"]
        if prepared_panel is not None
        else log_market_cap
    )
    common = common.intersection(train_size_values.dropna().index)
    if len(common):
        common = _strict_label_exit_index(common, config, boundary)
    X_train_raw = X_train_raw.reindex(common)
    raw_train_returns = forward_returns.reindex(common)
    if X_train_raw.empty:
        raise ValueError("no aligned size-conditional training rows")

    quality = config.get("data_quality", {})
    X_train_raw, _, audit = _filter_by_missing_rate(
        X_train_raw,
        float(quality.get("max_factor_missing_rate", 0.05)),
        logger,
        min_surviving_ratio=float(quality.get("min_surviving_factor_ratio", 0.0)),
        configured_factor_count=len(training_names),
        configured_factor_names=training_names,
        return_audit=True,
    )
    if X_train_raw.empty:
        raise InsufficientSurvivingFactorsError(audit)

    columns = list(X_train_raw.columns)
    X_predict_raw = factor_df.loc[predict_index, columns]

    if prepared_panel is not None:
        X_train = X_train_raw.fillna(0.0)
        X_predict = X_predict_raw.fillna(0.0)
        fit_target = prepared_panel["fit_target"].reindex(X_train.index)
        train_percentiles = prepared_panel["train_percentiles"].reindex(X_train.index)
        predict_percentiles = prepared_panel["predict_percentiles"].reindex(
            X_predict.index
        )
    else:
        combined = pd.concat((X_train_raw, X_predict_raw)).sort_index()
        preprocess = config.get("preprocess", config.get("preprocessing", {}))
        processed = (
            synthesizer._apply_feature_preprocessing(combined, preprocess)
            if preprocess
            else combined
        )
        X_train = processed.reindex(X_train_raw.index).fillna(0.0)
        X_predict = processed.reindex(X_predict_raw.index).fillna(0.0)
        fit_target = apply_target_cs_preprocessing(
            raw_train_returns,
            cs_demean=bool(config.get("target_cs_demean", True)),
            cs_rank=bool(config.get("target_cs_rank", False)),
            cs_standardize=bool(config.get("target_cs_standardize", False)),
            logger=logger,
        )
        train_percentiles = daily_size_percentile(log_market_cap, X_train.index)
        predict_percentiles = daily_size_percentile(log_market_cap, X_predict.index)
    weights = compute_sample_weights(
        X_train,
        weight_decay=float(config.get("sample_weight_decay", 1.0)),
        half_life=float(config.get("sample_weight_half_life", 0.0)),
        logger=logger,
    )
    selection = select_inner_policy(
        X_train,
        fit_target,
        raw_train_returns,
        train_percentiles,
        weights,
        normalized,
        label_context=config,
    )
    policy = selection["selected_policy"]
    gate = tuple(policy["gate"])
    train_gate = gate_mask(train_percentiles, gate)
    gated_train = X_train.loc[train_gate]
    gated_target = fit_target.reindex(gated_train.index)
    gated_weights = weights[train_gate] if weights is not None else None
    X_matrix, feature_names, _ = build_effect_coded_features(
        gated_train,
        train_percentiles.loc[gated_train.index],
        normalized["bucket_edges"],
        float(policy["interaction_shrinkage"]),
    )
    stable_positions = [
        feature_names.index(name) for name in selection["stable_support"]
    ]
    stability_fallback = None
    if not stable_positions:
        # The policy was selected solely from purged inner Top10 utility.  An
        # empty 60%-frequency support is an instability diagnostic, not proof
        # that the selected ElasticNet has no signal.  Keep the final selected
        # sparse fit intact and report this window explicitly; only an actually
        # all-zero final fit remains invalid below.
        stable_positions = list(range(len(feature_names)))
        stability_fallback = "full_selected_sparse_model"
    model = _fit_elasticnet(
        X_matrix[:, stable_positions],
        gated_target.to_numpy(dtype=float),
        alpha=float(policy["alpha"]),
        l1_ratio=float(policy["l1_ratio"]),
        sample_weight=gated_weights,
        max_iter=normalized["max_iter"],
        tol=normalized["tol"],
    )
    if not np.any(np.abs(model.coef_) > 1e-12):
        if len(stable_positions) == len(feature_names):
            raise ValueError("selected ElasticNet is all-zero on the full feature set")
        # Re-fitting the stable subset applies the same lasso penalty a second
        # time and can collapse an otherwise valid selected sparse model to a
        # constant. Preserve the original selected-model semantics instead of
        # emitting lexical TopK ties.
        stable_positions = list(range(len(feature_names)))
        stability_fallback = "stable_refit_all_zero_full_selected_sparse_model"
        model = _fit_elasticnet(
            X_matrix,
            gated_target.to_numpy(dtype=float),
            alpha=float(policy["alpha"]),
            l1_ratio=float(policy["l1_ratio"]),
            sample_weight=gated_weights,
            max_iter=normalized["max_iter"],
            tol=normalized["tol"],
        )
        if not np.any(np.abs(model.coef_) > 1e-12):
            raise ValueError("selected ElasticNet is all-zero on the full feature set")

    predict_gate = gate_mask(predict_percentiles, gate)
    gated_predict = X_predict.loc[predict_gate]
    if gated_predict.empty:
        raise ValueError("selected gate has no outer prediction rows")
    X_outer, outer_names, _ = build_effect_coded_features(
        gated_predict,
        predict_percentiles.loc[gated_predict.index],
        normalized["bucket_edges"],
        float(policy["interaction_shrinkage"]),
    )
    if outer_names != feature_names:
        raise ValueError("outer feature identity differs from training")
    composite = pd.Series(
        model.predict(X_outer[:, stable_positions]),
        index=gated_predict.index,
        name="size_conditional_elasticnet_composite",
    )

    full_coefficients = np.zeros(len(feature_names), dtype=float)
    full_coefficients[stable_positions] = model.coef_
    coefficient_payload = _coefficient_payload(
        feature_names,
        full_coefficients,
        columns,
        normalized["bucket_edges"],
        float(policy["interaction_shrinkage"]),
    )
    outer_raw = forward_returns.reindex(composite.index)
    outer_valid = outer_raw.notna()
    if outer_valid.any():
        outer_sharpe, outer_weekly, outer_turnover = top10_net_utility(
            composite.loc[outer_valid].to_numpy(dtype=float),
            outer_raw.loc[outer_valid].to_numpy(dtype=float),
            composite.loc[outer_valid].index,
            top_k=normalized["top_k"],
            cost_rate=normalized["cost_rate"],
            target_period=normalized["target_period"],
        )
    else:
        outer_sharpe, outer_weekly, outer_turnover = (
            -math.inf,
            pd.Series(dtype=float),
            math.nan,
        )
    protocol = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in normalized.items()
        if key not in {"alpha_grid", "gate_candidates"}
    }
    protocol["alpha_grid"] = list(normalized["alpha_grid"])
    protocol["gate_candidates"] = [
        list(gate_value) for gate_value in normalized["gate_candidates"]
    ]
    protocol["candidate_trial_count"] = len(selection["trials"])
    protocol["prepared_snapshot"] = config.get("_prepared_snapshot", {})
    return composite, {
        "method": "size_conditional_elasticnet",
        "label_isolation_version": "strict_label_exit_v2",
        "train_start": train_start,
        "train_end": train_end,
        "embargo_train_end": str(train_end_dt.date()),
        "validation_start": str(gated_predict.index.get_level_values(0).min().date()),
        "validation_end": str(gated_predict.index.get_level_values(0).max().date()),
        "selected_policy": policy,
        "inner_folds": selection["inner_folds"],
        "inner_trials": selection["trials"],
        "stable_support": selection["stable_support"],
        "stability_status": selection["stability_status"],
        "stability_fallback": stability_fallback,
        "stability": selection["stability"],
        "dsr": selection["dsr"],
        "performance": selection["performance"],
        "coefficients": coefficient_payload,
        "intercept": float(model.intercept_),
        "training_window_audit": audit,
        "protocol": protocol,
        "outer_proxy": {
            "top10_utility_version": TOP10_UTILITY_VERSION,
            "net_sharpe": outer_sharpe if math.isfinite(outer_sharpe) else None,
            "mean_turnover": outer_turnover if math.isfinite(outer_turnover) else None,
            "weekly_returns": {
                str(index.date()): float(value) for index, value in outer_weekly.items()
            },
        },
    }


def _json_safe(payload: Any) -> Any:
    """Convert diagnostic-only non-finite numbers to valid JSON nulls."""
    if isinstance(payload, Mapping):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    if isinstance(payload, np.generic):
        payload = payload.item()
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(
                _json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False
            )
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass  # Preserve the original write failure if cleanup is denied.
        raise


def _bootstrap_diagnostics(
    returns: pd.Series, *, samples: int, block_days: int, random_state: int
) -> dict[str, Any]:
    if len(returns) < 4 or samples < 1:
        return {
            "status": "insufficient_outer_returns",
            "observation_count": len(returns),
        }
    values = returns.to_numpy(dtype=float)
    block = max(1, int(round(block_days / 5)))
    rng = np.random.default_rng(random_state)
    means = np.empty(samples, dtype=float)
    for sample in range(samples):
        drawn: list[float] = []
        while len(drawn) < len(values):
            start = int(rng.integers(0, len(values)))
            drawn.extend(
                values.take(np.arange(start, start + block) % len(values)).tolist()
            )
        means[sample] = float(np.mean(drawn[: len(values)]))
    return {
        "status": "complete",
        "observation_count": len(values),
        "block_observations": block,
        "samples": samples,
        "mean_weekly_return": float(values.mean()),
        "mean_5pct": float(np.quantile(means, 0.05)),
        "mean_95pct": float(np.quantile(means, 0.95)),
        "probability_mean_le_zero": float((means <= 0).mean()),
    }


def write_size_conditional_diagnostics(
    model_infos: Sequence[Mapping[str, Any]],
    output_dir: Path,
    synthesis_name: str,
    protocol: Mapping[str, Any],
) -> list[str]:
    """Atomically persist the complete nested-selection evidence bundle."""
    if not model_infos or any(not info.get("selected_policy") for info in model_infos):
        raise ValueError("complete size-conditional model info is required")
    output_dir = Path(output_dir)
    outer_results = [
        {
            "train_start": info["train_start"],
            "train_end": info["train_end"],
            "embargo_train_end": info["embargo_train_end"],
            "outer_start": info["validation_start"],
            "outer_end": info["validation_end"],
            "selected_policy": info["selected_policy"],
            "outer_proxy": info["outer_proxy"],
        }
        for info in model_infos
    ]
    inner_results = [
        {
            "outer_start": info["validation_start"],
            "folds": info["inner_folds"],
            "selected_policy": info["selected_policy"],
            "trials": info["inner_trials"],
            "performance": info.get("performance", {}),
        }
        for info in model_infos
    ]
    feature_names = sorted(
        {name for info in model_infos for name in info.get("stability", {})}
    )
    factor_stability = {
        "outer_window_count": len(model_infos),
        "features": {
            name: {
                "stable_window_frequency": sum(
                    name in info.get("stable_support", ()) for info in model_infos
                )
                / len(model_infos),
                "mean_inner_selection_frequency": float(
                    np.mean(
                        [
                            info.get("stability", {})
                            .get(name, {})
                            .get("selection_frequency", 0.0)
                            for info in model_infos
                        ]
                    )
                ),
                "mean_direction_agreement": float(
                    np.mean(
                        [
                            info.get("stability", {})
                            .get(name, {})
                            .get("direction_agreement", 0.0)
                            for info in model_infos
                        ]
                    )
                ),
            }
            for name in feature_names
        },
    }
    coefficients = [
        {
            "outer_start": info["validation_start"],
            "selected_policy": info["selected_policy"],
            **info["coefficients"],
        }
        for info in model_infos
    ]
    policy_counts = Counter(
        json.dumps(info["selected_policy"], sort_keys=True) for info in model_infos
    )
    policy_stability = {
        "outer_window_count": len(model_infos),
        "selection_counts": [
            {"policy": json.loads(policy), "count": count}
            for policy, count in sorted(policy_counts.items())
        ],
    }
    outer_return_parts = [
        pd.Series(info.get("outer_proxy", {}).get("weekly_returns", {}), dtype=float)
        for info in model_infos
    ]
    stitched = (
        pd.concat(outer_return_parts) if outer_return_parts else pd.Series(dtype=float)
    )
    if not stitched.empty:
        stitched.index = pd.to_datetime(stitched.index)
        stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index()
    multiple_testing = {
        "per_window": [
            {"outer_start": info["validation_start"], **info["dsr"]}
            for info in model_infos
        ],
        "aggregate_block_bootstrap": _bootstrap_diagnostics(
            stitched,
            samples=int(protocol.get("bootstrap_samples", 1000)),
            block_days=int(protocol.get("bootstrap_block_days", 5)),
            random_state=int(protocol.get("random_state", 42)),
        ),
    }
    artifacts = {
        "nested_cv_protocol.json": {"synthesis_name": synthesis_name, **dict(protocol)},
        "outer_fold_results.json": outer_results,
        "inner_selection_results.json": inner_results,
        "factor_stability.json": factor_stability,
        "size_interaction_coefficients.json": coefficients,
        "policy_stability.json": policy_stability,
        "multiple_testing_correction.json": multiple_testing,
    }
    paths: list[str] = []
    for filename, payload in artifacts.items():
        path = output_dir / filename
        _atomic_json(path, payload)
        paths.append(str(path))
    return paths


__all__ = [
    "build_effect_coded_features",
    "daily_size_percentile",
    "gate_mask",
    "run_size_conditional_synthesis",
    "select_inner_policy",
    "top10_net_utility",
    "validate_size_conditional_config",
    "write_size_conditional_diagnostics",
]
