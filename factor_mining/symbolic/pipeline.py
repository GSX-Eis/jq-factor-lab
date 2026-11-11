"""Top-level orchestration for symbolic factor mining."""

from __future__ import annotations

import json
import shutil
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict

from .artifacts import ProgressWriter, write_hall_of_fame, write_json, write_summary
from .config import (
    load_analysis_context,
    load_mining_config,
    make_run_id,
    snapshot_config,
    validate_mining_config,
)
from .data_loader import (
    build_training_panel,
    downsample_panel_dates,
    max_label_horizon,
    split_train_valid_holdout_with_metadata,
)
from .elite_pool import update_elite_pool
from .expression import deserialize_expression, expression_hash, serialize_expression
from .feature_space import build_feature_space
from .preflight import run_data_preflight
from .reranker import (
    _deflated_t_hurdle,
    _evaluate_holdout_confirm,
    _selected_period_rank_ic,
    _selected_period_tstat,
    materialize_selected_factor,
    post_rerank_enabled,
    run_post_rerank,
)
from .search_base import SearchCandidate
from .search_random import RandomSearchBackend
from .validator import build_validation_summary

EXIT_SUCCESS = 0
EXIT_PRECONDITION = 2
EXIT_EXECUTION = 3
EXIT_QUALITY_GATE = 4


def _build_candidate_payload(candidate, search_metadata=None) -> Dict[str, Any]:
    return {
        "expression": candidate.expression.to_string(),
        "expression_ast": serialize_expression(candidate.expression),
        "expression_hash": expression_hash(candidate.expression),
        "score": candidate.score,
        "complexity": candidate.complexity,
        "train_metrics": candidate.train_metrics,
        "valid_metrics": candidate.valid_metrics,
        "search_metadata": search_metadata or {},
    }


def _warm_start_config(mining_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the ``search.warm_start`` block with safe defaults.

    The cross-run elite pool is shared, so ``pool_path`` resolves relative to the
    execution-repo working directory (the harness CWD) rather than the per-run
    output directory.
    """
    raw = mining_config.get("search", {}).get("warm_start", {})
    if not isinstance(raw, dict):
        return {"enabled": False}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "pool_path": str(raw.get("pool_path") or "state/symbolic_elite_pool.json"),
        "max_seeds": max(0, int(raw.get("max_seeds", 24) or 0)),
        "collect_top_k": max(0, int(raw.get("collect_top_k", 12) or 0)),
        "max_pool_size": max(1, int(raw.get("max_pool_size", 200) or 1)),
    }


def _build_search_backend(config: dict):
    backend = config.get("search", {}).get("backend", "random_v0")
    if backend == "random_v0":
        return RandomSearchBackend()
    if backend == "dso_pytorch":
        from .search_dso import DSOPyTorchSearchBackend

        return DSOPyTorchSearchBackend()
    raise ValueError(f"Unsupported symbolic search backend: {backend}")


def _duration(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 6)


def _clear_run_local_caches(mining_config: Dict[str, Any]) -> Dict[str, Any]:
    """Release in-memory symbolic scoring caches before returning run results."""
    summary: Dict[str, Any] = {}
    for key in ("_expression_eval_cache", "_scoring_contexts"):
        value = mining_config.pop(key, None)
        if hasattr(value, "__len__"):
            summary[f"{key}_entries_cleared"] = int(len(value))
        else:
            summary[f"{key}_entries_cleared"] = 0
        if hasattr(value, "clear"):
            with suppress(Exception):
                value.clear()
    return summary


def _cleanup_run_disk_caches(run_root: Path) -> Dict[str, Any]:
    """Delete per-run smoke-analysis caches; keep audit artifacts intact."""
    deleted: list[str] = []
    missing: list[str] = []
    errors: list[Dict[str, str]] = []
    targets = [
        run_root / "analysis_smoke_cache",
        run_root / "analysis_smoke_parquet",
        *(run_root.glob("analysis_smoke_results.duckdb*")),
    ]
    seen: set[Path] = set()
    for raw_target in targets:
        target = raw_target.resolve()
        if target in seen:
            continue
        seen.add(target)
        try:
            target.relative_to(run_root.resolve())
        except ValueError:
            errors.append({"path": str(target), "error": "outside_run_root"})
            continue
        if not target.exists():
            missing.append(str(target))
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted.append(str(target))
        except Exception as exc:
            errors.append({"path": str(target), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "deleted": deleted,
        "missing": missing,
        "errors": errors,
        "deleted_count": len(deleted),
        "error_count": len(errors),
    }


def _load_marginal_context(path: str | None) -> Dict[str, Any]:
    """Resolve the marginal-aware search context into a flat anchor spec.

    Returns ``{"enabled": False, ...}`` when no context is supplied or it is
    disabled.  When enabled, the per-anchor ``anchor_factor_name_sets`` are
    flattened, de-duplicated, sorted and capped at ``max_anchor_factors`` into a
    single union of anchor factor names to residualize candidates against.  A
    supplied-but-missing or malformed context propagates as an error.
    """
    if not path:
        return {"enabled": False, "anchor_factor_names": [], "analysis_db_path": ""}
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"marginal context not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"marginal context must be a JSON object: {target}")
    if not bool(payload.get("enabled", False)):
        return {"enabled": False, "anchor_factor_names": [], "analysis_db_path": ""}
    sets = payload.get("anchor_factor_name_sets")
    if not isinstance(sets, list) or not sets:
        raise ValueError("marginal context requires anchor_factor_name_sets[]")
    analysis_db_path = str(payload.get("analysis_db_path") or "").strip()
    if not analysis_db_path:
        raise ValueError("marginal context requires analysis_db_path")
    max_anchor_factors = max(1, int(payload.get("max_anchor_factors") or 40))
    seen: set[str] = set()
    flat: list[str] = []
    for group in sets:
        if not isinstance(group, list):
            continue
        for raw in group:
            name = str(raw or "").strip()
            if name and name not in seen:
                seen.add(name)
                flat.append(name)
    flat = sorted(flat)[:max_anchor_factors]
    return {
        "enabled": True,
        "anchor_factor_names": flat,
        "analysis_db_path": analysis_db_path,
    }


def run_symbolic_mining_pipeline(
    config_path: str,
    run_id: str | None = None,
    skip_validation: bool = False,
    novelty_context_path: str | None = None,
    marginal_context_path: str | None = None,
) -> Dict[str, Any]:
    if not novelty_context_path:
        raise ValueError("symbolic mining requires --novelty-context")
    marginal_context = _load_marginal_context(marginal_context_path)
    mining_config = load_mining_config(config_path)
    analysis_config_path = mining_config["analysis_config"]
    analysis_context = load_analysis_context(analysis_config_path)
    validate_mining_config(mining_config, analysis_context)

    run_id = run_id or make_run_id(mining_config)
    output_config = mining_config.get("output", {})
    run_root = Path(output_config.get("run_dir", "outputs/symbolic_mining")) / run_id
    factor_dir = Path(output_config.get("factor_dir", "factors/exp/symbolic_mined"))
    run_root.mkdir(parents=True, exist_ok=True)
    factor_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressWriter(run_root / "progress.jsonl", run_id=run_id)
    progress.emit(
        "pipeline_start",
        config_path=str(Path(config_path).resolve()),
        analysis_config=str(Path(analysis_config_path).resolve()),
        skip_validation=bool(skip_validation),
    )

    phase_start = time.perf_counter()
    snapshot_config(mining_config, analysis_context, run_root)
    mining_config["_progress_writer"] = progress
    progress.emit(
        "phase_done", phase="snapshot_config", duration_sec=_duration(phase_start)
    )

    # Fail-fast data preflight: abort before any panel construction when the
    # data backend is unreachable (e.g. unmounted volume) or the weekly pool
    # schema is broken, so the failure is cheap and carries a structured
    # ``data_preflight_failed`` code instead of a deep loader traceback.
    phase_start = time.perf_counter()
    preflight_summary = run_data_preflight(analysis_context)
    progress.emit(
        "phase_done",
        phase="data_preflight",
        duration_sec=_duration(phase_start),
        checks=preflight_summary.get("checks", []),
    )

    phase_start = time.perf_counter()
    anchor_factor_names = list(marginal_context.get("anchor_factor_names") or [])
    panel = build_training_panel(
        mining_config,
        analysis_context,
        anchor_factor_names=anchor_factor_names,
        anchor_db_path=str(marginal_context.get("analysis_db_path") or "") or None,
    )
    anchor_columns = [column for column in panel.columns if str(column).startswith("anchor__")]
    progress.emit(
        "phase_done",
        phase="build_training_panel",
        duration_sec=_duration(phase_start),
        rows=int(len(panel)),
        dates=int(panel["date"].nunique()) if "date" in panel.columns else 0,
        columns=int(len(panel.columns)),
        marginal_aware=bool(marginal_context.get("enabled")),
        anchor_factor_count=len(anchor_columns),
    )
    split_ratio = float(mining_config.get("search", {}).get("train_ratio", 0.7))
    label_horizon = max_label_horizon(mining_config)
    embargo_days = int(
        mining_config.get("search", {}).get("embargo_days", label_horizon)
    )
    # The valid->holdout boundary needs the same label-horizon embargo as
    # train->valid: without it the last valid dates carry forward returns that
    # overlap the first holdout returns, letting selection peek at the gate.
    holdout_embargo_days = int(
        mining_config.get("search", {}).get("holdout_embargo_days", embargo_days)
    )
    train_end_date = mining_config.get("search", {}).get("train_end_date")
    valid_start_date = mining_config.get("search", {}).get("valid_start_date")
    valid_end_date = mining_config.get("search", {}).get("valid_end_date")
    holdout_start_date = mining_config.get("search", {}).get("holdout_start_date")
    phase_start = time.perf_counter()
    train_panel, valid_panel, holdout_panel, split_metadata = (
        split_train_valid_holdout_with_metadata(
            panel,
            split_ratio,
            embargo_days=embargo_days,
            train_end_date=str(train_end_date) if train_end_date else None,
            valid_start_date=str(valid_start_date) if valid_start_date else None,
            valid_end_date=str(valid_end_date) if valid_end_date else None,
            holdout_start_date=str(holdout_start_date) if holdout_start_date else None,
            holdout_embargo_days=holdout_embargo_days,
        )
    )
    # Optional coarser search cadence (e.g. monthly): thin only the search-facing
    # train/valid cross-sections to curb overfitting to daily noise. The holdout
    # panel that feeds the OOS gate is always left at full daily resolution.
    date_sampling = str(
        mining_config.get("search", {}).get("train_date_sampling", "none")
    )
    train_panel, train_sampling_meta = downsample_panel_dates(train_panel, date_sampling)
    valid_panel, valid_sampling_meta = downsample_panel_dates(valid_panel, date_sampling)
    progress.emit(
        "phase_done",
        phase="split_train_valid",
        duration_sec=_duration(phase_start),
        train_rows=int(len(train_panel)),
        valid_rows=int(len(valid_panel)),
        holdout_rows=int(len(holdout_panel)),
        train_dates=int(train_panel["date"].nunique())
        if "date" in train_panel.columns
        else 0,
        valid_dates=int(valid_panel["date"].nunique())
        if "date" in valid_panel.columns
        else 0,
        holdout_dates=int(holdout_panel["date"].nunique())
        if "date" in holdout_panel.columns and not holdout_panel.empty
        else 0,
        date_sampling=date_sampling,
        train_date_sampling=train_sampling_meta,
        valid_date_sampling=valid_sampling_meta,
        split=split_metadata,
    )
    phase_start = time.perf_counter()
    feature_space = build_feature_space(mining_config)
    feature_space.features = {
        name: spec
        for name, spec in feature_space.features.items()
        if name in panel.columns
    }
    progress.emit(
        "phase_done",
        phase="build_feature_space",
        duration_sec=_duration(phase_start),
        features=int(len(feature_space.features)),
    )

    phase_start = time.perf_counter()
    progress.emit(
        "phase_start",
        phase="search",
        backend=str(mining_config.get("search", {}).get("backend", "random_v0")),
        epochs=int(mining_config.get("search", {}).get("epochs", 0)),
        batch_size=int(mining_config.get("search", {}).get("batch_size", 0)),
    )
    search_result = _build_search_backend(mining_config).search(
        train_panel, valid_panel, feature_space, mining_config
    )
    progress.emit(
        "phase_done",
        phase="search",
        duration_sec=_duration(phase_start),
        attempted=int(search_result.metadata.get("attempted", 0)),
        scored=int(search_result.metadata.get("scored", 0)),
        invalid=int(search_result.metadata.get("invalid", 0)),
        duplicates=int(search_result.metadata.get("duplicates", 0)),
    )
    phase_start = time.perf_counter()
    write_hall_of_fame(run_root / "hall_of_fame.csv", search_result.hall_of_fame)
    search_best_payload = _build_candidate_payload(
        search_result.best, search_result.metadata
    )
    write_json(run_root / "search_best_expression.json", search_best_payload)
    warm_start_cfg = _warm_start_config(mining_config)
    elite_pool_update: Dict[str, Any] | None = None
    if warm_start_cfg.get("enabled") and search_result.hall_of_fame:
        try:
            elite_pool_update = update_elite_pool(
                warm_start_cfg["pool_path"],
                search_result.hall_of_fame,
                run_id=run_id,
                collect_top_k=int(warm_start_cfg.get("collect_top_k", 12)),
                max_pool_size=int(warm_start_cfg.get("max_pool_size", 200)),
            )
        except Exception as exc:  # pool persistence must never crash a run
            elite_pool_update = {"error": f"{type(exc).__name__}: {exc}"}
    progress.emit(
        "phase_done",
        phase="write_search_artifacts",
        duration_sec=_duration(phase_start),
        elite_pool_update=elite_pool_update,
    )

    factor_path = factor_dir / f"symbolic_mined_factor_{run_id}.py"
    generated_factor_path = run_root / "generated_factor.py"
    selected_candidate = search_result.best
    selected_factor_path = None
    rerank_summary = None
    if post_rerank_enabled(mining_config):
        phase_start = time.perf_counter()
        progress.emit(
            "phase_start", phase="post_rerank", skip_validation=bool(skip_validation)
        )
        if skip_validation:
            rerank_summary = {
                "enabled": True,
                "skipped": True,
                "reason": "skip_validation",
                "dual_gate_passed": None,
                "selected_candidate": search_best_payload,
                "search_best": search_best_payload,
            }
            write_json(run_root / "rerank_summary.json", rerank_summary)
        else:
            rerank_result = run_post_rerank(
                search_result,
                train_panel,
                valid_panel,
                feature_space,
                mining_config,
                analysis_context,
                analysis_config_path,
                run_root,
                factor_dir,
                run_id,
                novelty_context_path,
                holdout_panel=holdout_panel,
            )
            selected_candidate = rerank_result.selected_candidate
            selected_factor_path = rerank_result.selected_factor_path
            rerank_summary = rerank_result.summary
        progress.emit(
            "phase_done",
            phase="post_rerank",
            duration_sec=_duration(phase_start),
            skipped=bool(rerank_summary.get("skipped", False))
            if isinstance(rerank_summary, dict)
            else False,
            dual_gate_passed=rerank_summary.get("dual_gate_passed")
            if isinstance(rerank_summary, dict)
            else None,
        )

    phase_start = time.perf_counter()
    selected_payload = _build_candidate_payload(
        selected_candidate, search_result.metadata
    )
    write_json(run_root / "best_expression.json", selected_payload)
    materialize_selected_factor(
        selected_factor_path,
        selected_candidate,
        mining_config,
        feature_space,
        factor_path,
        generated_factor_path,
    )
    progress.emit(
        "phase_done",
        phase="materialize_selected_factor",
        duration_sec=_duration(phase_start),
        factor_path=str(factor_path.resolve()),
    )

    rerank_required = bool(mining_config.get("post_rerank", {}).get("required", False))
    rerank_ok = not (
        rerank_required
        and rerank_summary is not None
        and not rerank_summary.get("skipped", False)
        and rerank_summary.get("dual_gate_passed") is False
    )
    cache_cleanup = _clear_run_local_caches(mining_config)
    progress.emit(
        "cache_cleared",
        **cache_cleanup,
    )

    validation_summary = None
    if not skip_validation and not rerank_ok:
        validation_summary = {
            "ok": True,
            "skipped": True,
            "reason": "post_rerank_quality_gate_failed",
        }
        write_json(run_root / "validation_summary.json", validation_summary)
        progress.emit(
            "phase_done",
            phase="validation",
            duration_sec=0.0,
            ok=True,
            skipped=True,
            reason="post_rerank_quality_gate_failed",
        )
    elif not skip_validation:
        phase_start = time.perf_counter()
        progress.emit("phase_start", phase="validation")
        validation_summary = build_validation_summary(
            factor_path,
            analysis_config_path,
            run_analysis=bool(
                mining_config.get("validation", {}).get("run_analysis", True)
            ),
        )
        write_json(run_root / "validation_summary.json", validation_summary)
        progress.emit(
            "phase_done",
            phase="validation",
            duration_sec=_duration(phase_start),
            ok=bool(validation_summary.get("ok", False)),
        )

    validation_ok = validation_summary["ok"] if validation_summary is not None else True
    ok = bool(validation_ok and rerank_ok)
    exit_code = EXIT_SUCCESS if ok else EXIT_QUALITY_GATE
    failure_reason = ""
    if not rerank_ok and isinstance(rerank_summary, dict):
        failure_reason = str(
            rerank_summary.get("quality_gate_failure_reason")
            or rerank_summary.get("selected_reason")
            or "post_rerank_quality_gate_failed"
        )
    elif not validation_ok:
        failure_reason = "validation_failed"
    disk_cache_cleanup = _cleanup_run_disk_caches(run_root)
    summary = {
        "ok": ok,
        "run_id": run_id,
        "exit_code": exit_code,
        "failure_reason": failure_reason,
        "config_path": str(Path(config_path).resolve()),
        "analysis_config": str(Path(analysis_config_path).resolve()),
        "run_dir": str(run_root.resolve()),
        "factor_file": str(factor_path.resolve()),
        "generated_factor_copy": str((run_root / "generated_factor.py").resolve()),
        "panel": {
            "rows": int(len(panel)),
            "train_rows": int(len(train_panel)),
            "valid_rows": int(len(valid_panel)),
            "features": int(len(feature_space.features)),
            "embargo_days": int(embargo_days),
            "holdout_embargo_days": int(holdout_embargo_days),
            "max_label_horizon": int(label_horizon),
            "split": split_metadata,
        },
        "search_best": search_best_payload,
        "selected_best": selected_payload,
        "best_expression": selected_payload,
        "post_rerank": rerank_summary,
        "validation": validation_summary,
        "cache_cleanup": cache_cleanup,
        "disk_cache_cleanup": disk_cache_cleanup,
        "artifacts": {
            "summary": str((run_root / "summary.json").resolve()),
            "config_snapshot": str((run_root / "config_snapshot.json").resolve()),
            "hall_of_fame": str((run_root / "hall_of_fame.csv").resolve()),
            "best_expression": str((run_root / "best_expression.json").resolve()),
            "search_best_expression": str(
                (run_root / "search_best_expression.json").resolve()
            ),
            "rerank_summary": str((run_root / "rerank_summary.json").resolve())
            if rerank_summary is not None
            else "",
            "validation_summary": str((run_root / "validation_summary.json").resolve())
            if validation_summary is not None
            else "",
            "progress": str((run_root / "progress.jsonl").resolve()),
        },
    }
    write_summary(run_root / "summary.json", summary)
    progress.emit(
        "disk_cache_cleared",
        **disk_cache_cleanup,
    )
    progress.emit(
        "pipeline_done",
        ok=bool(ok),
        exit_code=int(exit_code),
        failure_reason=failure_reason,
    )
    mining_config.pop("_progress_writer", None)
    return summary


def _build_holdout_panel(
    mining_config: Dict[str, Any], analysis_context: Any
) -> Any:
    """Rebuild the holdout split exactly as ``run_symbolic_mining_pipeline``.

    Anchors are not needed: the holdout confirm only computes label
    correlations / quantile returns of the candidate's raw prediction.
    """
    panel = build_training_panel(mining_config, analysis_context)
    split_ratio = float(mining_config.get("search", {}).get("train_ratio", 0.7))
    label_horizon = max_label_horizon(mining_config)
    embargo_days = int(
        mining_config.get("search", {}).get("embargo_days", label_horizon)
    )
    holdout_embargo_days = int(
        mining_config.get("search", {}).get("holdout_embargo_days", embargo_days)
    )
    search_config = mining_config.get("search", {})
    train_end_date = search_config.get("train_end_date")
    valid_start_date = search_config.get("valid_start_date")
    valid_end_date = search_config.get("valid_end_date")
    holdout_start_date = search_config.get("holdout_start_date")
    _, _, holdout_panel, _ = split_train_valid_holdout_with_metadata(
        panel,
        split_ratio,
        embargo_days=embargo_days,
        train_end_date=str(train_end_date) if train_end_date else None,
        valid_start_date=str(valid_start_date) if valid_start_date else None,
        valid_end_date=str(valid_end_date) if valid_end_date else None,
        holdout_start_date=str(holdout_start_date) if holdout_start_date else None,
        holdout_embargo_days=holdout_embargo_days,
    )
    return holdout_panel


def _refresh_alternate_holdout_confirm(
    rerank_summary: Dict[str, Any],
    *,
    row: Dict[str, Any],
    candidate: SearchCandidate,
    mining_config: Dict[str, Any],
    analysis_context: Any,
    gate_period: Any,
) -> None:
    """Confirm a materialized alternate candidate on the holdout (one look).

    Each alternate materialization is one additional holdout look on top of
    the run's original confirm: ``confirms_this_run`` increments and the
    deflated-t hurdle is recomputed at the updated look count so the charge is
    paid where it is incurred. On any failure the confirm block fails CLOSED
    (``evaluated=False`` with a reason and no stale metrics survive), letting
    the evaluator's ic-significance gate hold the run for review instead of
    advancing on the original candidate's numbers.
    """
    rerank_config = mining_config.get("post_rerank", {}) or {}
    existing = rerank_summary.get("holdout_confirm")
    existing = existing if isinstance(existing, dict) else {}
    prev_confirms = int(existing.get("confirms_this_run") or 0)
    base_looks = max(1, int(rerank_config.get("holdout_confirm_looks", 1) or 1))
    period = None
    if gate_period is not None:
        with suppress(TypeError, ValueError):
            period = int(gate_period)
    if period is None:
        raw_gate_period = rerank_config.get("gate_period")
        if raw_gate_period is not None:
            period = int(raw_gate_period)
    try:
        holdout_panel = _build_holdout_panel(mining_config, analysis_context)
        confirm = _evaluate_holdout_confirm(
            candidate,
            holdout_panel,
            rerank_config,
            dual_gate_passed=bool(row.get("dual_gate_passed", False)),
            gate_period=period,
        )
    except Exception as exc:
        confirm = {
            "evaluated": False,
            "reason": f"alternate_confirm_failed: {type(exc).__name__}: {exc}",
            "period": period,
            "rank_ic_mean": None,
            "rank_ic_t": None,
            "top_quantile_excess_return_mean": None,
            "top_quantile_excess_return_t": None,
            "period_metrics": {},
            "window": {},
            "confirms_this_run": 0,
        }
    confirms_this_run = prev_confirms + (1 if confirm.get("evaluated") else 0)
    looks = base_looks + max(0, confirms_this_run - 1)
    confirm["confirms_this_run"] = int(confirms_this_run)
    confirm["looks"] = int(looks)
    confirm["confirm_source"] = "alternate_materialization"

    selected_oos_rank_ic_t = _selected_period_tstat(
        confirm.get("period_metrics") or {}, confirm.get("period")
    )
    multiplicity = rerank_summary.get("multiplicity")
    multiplicity = dict(multiplicity) if isinstance(multiplicity, dict) else {}
    correction = str(
        multiplicity.get("correction")
        or rerank_config.get("multiplicity_correction", "off")
    )
    min_rank_ic_t = float(rerank_config.get("min_rank_ic_t", 0.0))
    if correction == "deflated":
        hurdle_t = _deflated_t_hurdle(looks)
    elif correction == "fixed_t":
        hurdle_t = min_rank_ic_t
    else:
        hurdle_t = None
    would_block = bool(
        hurdle_t is not None
        and selected_oos_rank_ic_t is not None
        and selected_oos_rank_ic_t < hurdle_t
    )
    multiplicity.update(
        {
            "effective_trials": int(looks),
            "trials_source": "holdout_confirm_looks",
            "selected_rank_ic_t": selected_oos_rank_ic_t,
            "hurdle_t": hurdle_t,
            "would_block": would_block,
            "available": bool(
                hurdle_t is not None and selected_oos_rank_ic_t is not None
            ),
        }
    )
    rerank_summary["holdout_confirm"] = confirm
    rerank_summary["holdout_evaluated"] = bool(confirm.get("evaluated", False))
    rerank_summary["selected_oos_period_metrics"] = confirm.get("period_metrics") or {}
    rerank_summary["selected_oos_rank_ic_t"] = selected_oos_rank_ic_t
    rerank_summary["multiplicity"] = multiplicity


def materialize_rerank_candidate(
    config_path: str,
    *,
    run_id: str,
    rank: int,
    factor_path: str | None = None,
) -> Dict[str, Any]:
    mining_config = load_mining_config(config_path)
    analysis_config_path = mining_config["analysis_config"]
    analysis_context = load_analysis_context(analysis_config_path)
    validate_mining_config(mining_config, analysis_context)
    # Same fail-fast preflight as the mining entry: materialization rebuilds
    # the holdout panel and runs smoke validation against the live backend.
    run_data_preflight(analysis_context)

    output_config = mining_config.get("output", {})
    run_root = Path(output_config.get("run_dir", "outputs/symbolic_mining")) / run_id
    factor_dir = Path(output_config.get("factor_dir", "factors/exp/symbolic_mined"))
    rerank_path = run_root / "rerank_summary.json"
    if not rerank_path.exists():
        raise FileNotFoundError(f"rerank summary not found: {rerank_path}")

    rerank_summary = json.loads(rerank_path.read_text(encoding="utf-8"))
    candidates = rerank_summary.get("candidates", [])
    row = next(
        (item for item in candidates if int(item.get("rank", -1)) == int(rank)), None
    )
    if not isinstance(row, dict):
        raise ValueError(f"rerank candidate rank not found: {rank}")
    payload = row.get("candidate")
    if not isinstance(payload, dict):
        raise ValueError(f"rerank candidate rank {rank} has no candidate payload")
    ast = payload.get("expression_ast")
    if not isinstance(ast, dict):
        raise ValueError(f"rerank candidate rank {rank} has no expression_ast")

    candidate = SearchCandidate(
        expression=deserialize_expression(ast),
        train_metrics=dict(payload.get("train_metrics") or {}),
        valid_metrics=dict(payload.get("valid_metrics") or {}),
        score=float(payload.get("score") or 0.0),
        complexity=int(payload.get("complexity") or 0),
    )
    feature_space = build_feature_space(mining_config)
    planned_path_raw = str(row.get("planned_factor_file") or "").strip()
    planned_path = Path(planned_path_raw).expanduser()
    if not planned_path_raw:
        planned_path = (
            factor_dir / f"symbolic_mined_factor_{run_id}_rerank_{rank:02d}.py"
        )
    compile_target = (
        planned_path if planned_path.is_absolute() else planned_path.resolve()
    )
    compile_target.parent.mkdir(parents=True, exist_ok=True)
    from .compiler import compile_factor_file

    compile_factor_file(candidate, mining_config, feature_space, compile_target)
    final_factor_path = (
        Path(factor_path).expanduser()
        if factor_path
        else factor_dir / f"symbolic_mined_factor_{run_id}.py"
    )
    generated_factor_path = run_root / "generated_factor.py"
    materialize_selected_factor(
        compile_target,
        candidate,
        mining_config,
        feature_space,
        final_factor_path,
        generated_factor_path,
    )
    validation_summary = build_validation_summary(
        final_factor_path,
        analysis_config_path,
        run_analysis=bool(
            mining_config.get("validation", {}).get("run_analysis", True)
        ),
    )
    write_json(run_root / "validation_summary.json", validation_summary)
    selected_payload = _build_candidate_payload(
        candidate, {"source": "harness_topk_novelty", "selected_rank": int(rank)}
    )
    write_json(run_root / "best_expression.json", selected_payload)

    row["factor_file"] = str(final_factor_path.resolve())
    row["harness_full_validation_attempted"] = True
    row["harness_full_validation"] = validation_summary
    row["harness_full_validation_passed"] = bool(validation_summary.get("ok"))
    rerank_summary["selected_candidate"] = selected_payload
    rerank_summary["selected_rank"] = int(rank)
    rerank_summary["selected_factor_file"] = str(final_factor_path.resolve())
    rerank_summary["selected_reason"] = "harness_topk_novelty_candidate"
    # Refresh the selection-window metrics so downstream gates evaluate THIS
    # alternate, not whatever run_post_rerank originally selected. The
    # selection-window values come from the per-candidate metrics stored on the
    # row at search time (valid_period_metrics == the selection lightweight).
    row_best_period = row.get("best_gate_period")
    row_valid_metrics = row.get("valid_period_metrics")
    if not (isinstance(row_valid_metrics, dict) and row_valid_metrics):
        row_valid_metrics = row.get("lightweight_period_metrics")
    if isinstance(row_valid_metrics, dict) and row_valid_metrics:
        rerank_summary["selected_selection_period_metrics"] = row_valid_metrics
        rerank_summary["selected_valid_rank_ic"] = _selected_period_rank_ic(
            row_valid_metrics, row_best_period
        )
        rerank_summary["selected_valid_rank_ic_t"] = _selected_period_tstat(
            row_valid_metrics, row_best_period
        )
        rerank_summary["selected_valid_rank_ic_reason"] = ""
    else:
        rerank_summary["selected_selection_period_metrics"] = {}
        rerank_summary["selected_valid_rank_ic"] = None
        rerank_summary["selected_valid_rank_ic_t"] = None
        rerank_summary["selected_valid_rank_ic_reason"] = (
            "alternate_candidate_no_valid_period_metrics"
        )
    # Holdout single-look confirmation for the alternate. Selection no longer
    # touches the holdout, so the original confirm covered the ORIGINAL
    # selection only; this alternate needs (and is charged) its own look.
    _refresh_alternate_holdout_confirm(
        rerank_summary,
        row=row,
        candidate=candidate,
        mining_config=mining_config,
        analysis_context=analysis_context,
        gate_period=row_best_period,
    )
    write_json(rerank_path, rerank_summary)

    summary_path = run_root / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    summary.update(
        {
            "ok": bool(validation_summary.get("ok")),
            "exit_code": EXIT_SUCCESS
            if validation_summary.get("ok")
            else EXIT_QUALITY_GATE,
            "factor_file": str(final_factor_path.resolve()),
            "generated_factor_copy": str(generated_factor_path.resolve()),
            "selected_best": selected_payload,
            "best_expression": selected_payload,
            "post_rerank": rerank_summary,
            "validation": validation_summary,
            "cache_cleanup": _clear_run_local_caches(mining_config),
            "disk_cache_cleanup": _cleanup_run_disk_caches(run_root),
        }
    )
    artifacts = dict(summary.get("artifacts") or {})
    artifacts["best_expression"] = str((run_root / "best_expression.json").resolve())
    artifacts["validation_summary"] = str(
        (run_root / "validation_summary.json").resolve()
    )
    artifacts["rerank_summary"] = str(rerank_path.resolve())
    summary["artifacts"] = artifacts
    write_summary(summary_path, summary)
    return summary
