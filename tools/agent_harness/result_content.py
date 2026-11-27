"""Content checks for current Harness successes after envelope validation."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.agent_harness.run_result import _is_sha256_hex, expected_synthesis_items

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSITE_SKIP_REASONS = (
    "skip_alphalens_analysis",
    "non_daily_frequency",
    "synthesis_reuse",
    "no_local_db",
    "single_instrument",
)
_ANALYSIS_STAT_TABLES = ("ic", "ir", "quantile_returns", "alphalens_data")


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _finite(value: Any) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _count(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _records(value: Any, field: str) -> list[dict]:
    if not isinstance(value, list) or not value or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{field} must contain structured records")
    return value


def _match_completed(action: str, actual: int, completed: int) -> None:
    if actual < 1 or completed != actual:
        raise ValueError(f"{action} completed count mismatch: reported {completed}, verified {actual}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_slot_config_obj(
    config: Mapping | None,
    *,
    config_path: str | Path | None = None,
    slot_config: Any = None,
) -> tuple[Any, Mapping | None]:
    cfg = config or {}
    raw_slot = slot_config or cfg.get("slot_config") or cfg.get("slot")
    if isinstance(raw_slot, Mapping):
        return None, raw_slot
    if isinstance(raw_slot, str):
        raw_slot = raw_slot.strip()
        if raw_slot and raw_slot != "jq_ashare_local":
            from utils.slots import load_slot_config

            slot_path = Path(raw_slot).expanduser()
            if not slot_path.is_absolute():
                repo_candidate = _REPO_ROOT / slot_path
                if repo_candidate.exists() or config_path is None:
                    slot_path = repo_candidate
                else:
                    slot_path = Path(config_path).expanduser().resolve().parent / slot_path
            return load_slot_config(slot_path.resolve()), None
    return None, None


def _resolve_synthesis_frequency(
    config: Mapping | None,
    *,
    config_path: str | Path | None = None,
    slot_config: Any = None,
    frequency: str | None = None,
) -> str:
    from utils.time_axis import Frequency

    cfg = config or {}
    slot_obj, slot_mapping = _load_slot_config_obj(
        cfg, config_path=config_path, slot_config=slot_config,
    )
    slot_freq = None
    if slot_obj is not None:
        slot_freq = slot_obj.bar_feed.options.get("frequency")
    elif isinstance(slot_mapping, Mapping):
        bar_feed = slot_mapping.get("bar_feed") or {}
        if isinstance(bar_feed, Mapping):
            options = bar_feed.get("options") or {}
            if isinstance(options, Mapping):
                slot_freq = options.get("frequency")
    raw_freq = str(frequency or cfg.get("frequency") or slot_freq or "1d").strip()
    parsed = Frequency.parse(raw_freq)
    if not parsed.is_valid or parsed.canonical is None:
        raise ValueError(f"Invalid synthesis frequency: {raw_freq!r}")
    return parsed.canonical


def _allowed_composite_only_reasons(
    config: Mapping,
    *,
    method: str,
    effective_frequency: str,
    slot_config: Any = None,
    slot_id: str | None = None,
    market_db: str | None = None,
) -> set[str]:
    from utils.data_store import resolve_database_config

    resolved_cfg = resolve_database_config(dict(config))
    reasons: set[str] = set()
    pe_cfg = resolved_cfg.get("performance_evaluation") or {}
    if isinstance(pe_cfg, Mapping) and pe_cfg.get("skip_alphalens_analysis", False):
        reasons.add("skip_alphalens_analysis")
    if effective_frequency != "1d":
        reasons.add("non_daily_frequency")
    reuse_cfg = resolved_cfg.get("synthesis_reuse") or {}
    if isinstance(reuse_cfg, Mapping) and reuse_cfg.get("enabled", False) and method == "lasso":
        from factor_synthesis.core import (
            LABEL_MODE_NEXT_REBALANCE_CLOSE,
            LABEL_MODE_NEXT_REBALANCE_OPEN,
            resolve_label_definition,
        )

        label_mode = resolve_label_definition(resolved_cfg.get("label_definition")).get("mode")
        if label_mode not in {LABEL_MODE_NEXT_REBALANCE_OPEN, LABEL_MODE_NEXT_REBALANCE_CLOSE}:
            reasons.add("synthesis_reuse")
    explicit_slot = bool(
        slot_config
        or resolved_cfg.get("slot_config")
        or resolved_cfg.get("slot")
        or market_db
        or resolved_cfg.get("market_db")
        or (slot_id not in (None, "", "jq_ashare_local"))
        or (resolved_cfg.get("slot_id") not in (None, "", "jq_ashare_local"))
    )
    db_cfg = resolved_cfg.get("database") or {}
    local_db = db_cfg.get("local_db") if isinstance(db_cfg, Mapping) else None
    has_local_db = isinstance(local_db, Mapping) and bool(local_db.get("duckdb_path"))
    if explicit_slot and not has_local_db:
        reasons.add("no_local_db")
    return reasons


def _resolve_existing_dir(raw_dir: str | Path, *, config_path: str | Path | None = None) -> Path | None:
    candidate = Path(raw_dir).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.is_dir() else None
    roots: list[Path] = []
    if config_path is not None:
        roots.append(Path(config_path).expanduser().resolve().parent)
    roots.extend((Path.cwd(), _REPO_ROOT))
    for base in roots:
        resolved = (base / candidate).resolve()
        if resolved.is_dir():
            return resolved
    return None


def _valid_reuse_model(model: Any, *, source_dir: Path, require_parent: bool) -> bool:
    from factor_synthesis.strategy_bundle import (
        _has_finite_numeric_weights,
        _has_nonempty_str_sequence,
    )

    if not isinstance(model, Mapping) or not _name(model.get("train_end")):
        return False
    parent = model.get("parent_baseline")
    if require_parent and (not isinstance(parent, Mapping) or not parent):
        return False
    if parent is not None and parent != {}:
        if not isinstance(parent, Mapping):
            return False
        rel_raw = str(parent.get("path", "")).strip()
        expected_sha = parent.get("sha256")
        if not rel_raw or not _is_sha256_hex(expected_sha):
            return False
        rel_path = Path(rel_raw)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            return False
        src_parent = (source_dir / rel_path).resolve()
        if not src_parent.is_relative_to(source_dir.resolve()) or not src_parent.is_file():
            return False
        if _sha256_file(src_parent) != expected_sha:
            return False
    weights = model.get("weights")
    if weights is not None and weights != {} and not _has_finite_numeric_weights(weights):
        return False
    blend_weights = model.get("blend_weights")
    if blend_weights is not None and blend_weights != {} and not _has_finite_numeric_weights(blend_weights):
        return False
    has_weights = _has_finite_numeric_weights(weights) and _has_nonempty_str_sequence(model.get("feature_names"))
    is_blend_only = (
        model.get("blend_only") is True
        and (_has_finite_numeric_weights(blend_weights) or (isinstance(parent, Mapping) and bool(parent)))
    )
    return has_weights or is_blend_only


def _synthesis_reuse_verified(
    row: Mapping,
    config: Mapping,
    *,
    synthesis_name: str,
    config_path: str | Path | None = None,
) -> bool:
    reuse_cfg = config.get("synthesis_reuse") or {}
    output_settings = config.get("output_settings") or {}
    raw_source = (
        reuse_cfg.get("source_dir")
        or (output_settings.get("weights_output_dir") if isinstance(output_settings, Mapping) else None)
    )
    if not _name(raw_source):
        return False
    source_dir = _resolve_existing_dir(str(raw_source), config_path=config_path)
    if source_dir is None:
        return False
    bundle_path = source_dir / f"lasso_model_bundle_{synthesis_name}.json"
    cache_path = source_dir / f"composite_cache_{synthesis_name}.parquet"
    if not bundle_path.is_file() or not cache_path.is_file():
        return False
    bundle_sha = row.get("reuse_bundle_sha256")
    cache_sha = row.get("reuse_cache_sha256")
    if (
        not _is_sha256_hex(bundle_sha)
        or not _is_sha256_hex(cache_sha)
        or _sha256_file(bundle_path) != bundle_sha
        or _sha256_file(cache_path) != cache_sha
    ):
        return False
    try:
        bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    index_block = bundle_payload.get("index") if isinstance(bundle_payload, Mapping) else None
    if not isinstance(index_block, Mapping):
        return False
    if index_block.get("synthesis_name") is not None and index_block.get("synthesis_name") != synthesis_name:
        return False
    models = index_block.get("models")
    if not isinstance(models, list) or not models:
        return False
    requires_parent = bool(index_block.get("requires_parent_baseline", False))
    if requires_parent and (
        type(index_block.get("expected_model_count")) is not int
        or index_block.get("expected_model_count") != len(models)
    ):
        return False
    if not all(_valid_reuse_model(m, source_dir=source_dir, require_parent=requires_parent) for m in models):
        return False

    from numbers import Number
    import numpy as np
    import pandas as pd
    from factor_synthesis.core.calendar_skip import parse_skip_rules
    from factor_synthesis.core.pool_manager import apply_skip_months

    try:
        frame = pd.read_parquet(cache_path)
    except Exception:
        return False
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[1] < 1:
        return False
    if not isinstance(frame.index, pd.MultiIndex):
        return False
    if frame.index.names.count("date") != 1 or frame.index.names.count("stock_code") != 1:
        return False
    if not frame.index.is_unique:
        return False
    date_vals = frame.index.get_level_values("date")
    if not isinstance(date_vals, pd.DatetimeIndex) and any(isinstance(val, Number) for val in date_vals):
        return False
    try:
        parsed_dates = pd.to_datetime(date_vals)
    except Exception:
        return False
    if parsed_dates.hasnans:
        return False
    stock_vals = frame.index.get_level_values("stock_code")
    if any(not _name(str(stock)) if pd.notna(stock) else True for stock in stock_vals):
        return False
    composite = frame.iloc[:, 0]
    if not pd.api.types.is_numeric_dtype(composite) or pd.api.types.is_bool_dtype(composite):
        return False
    try:
        composite.unstack(level="stock_code")
    except Exception:
        return False
    cal_cfg = config.get("calendar_filters")
    if isinstance(cal_cfg, Mapping) and bool(cal_cfg.get("apply_to_evaluation", True)):
        try:
            skip_rules = parse_skip_rules(cal_cfg.get("skip_months"))
            if skip_rules:
                composite = apply_skip_months(composite, skip_rules)
        except Exception:
            return False
    if composite is None or composite.empty:
        return False
    expected_finite = int(np.isfinite(composite.to_numpy(dtype=float)).sum())
    if expected_finite < 1 or row.get("finite_score_count") != expected_finite:
        return False
    cache_stocks = int(composite.index.get_level_values("stock_code").nunique())
    if "stock_count" in row and row["stock_count"] is not None:
        stock_pool_cfg = config.get("stock_pool")
        period_cfg = config.get("analysis_period")
        has_dynamic_pool = (
            isinstance(stock_pool_cfg, Mapping)
            and bool(stock_pool_cfg.get("use_dynamic_pools", False))
            and isinstance(period_cfg, Mapping)
            and bool(period_cfg.get("start_date"))
            and bool(period_cfg.get("end_date"))
        )
        if not _count(row["stock_count"], minimum=cache_stocks):
            return False
        if not has_dynamic_pool and row["stock_count"] != cache_stocks:
            return False
    return True


def _requested_single_instrument(
    config: Mapping | None,
    *,
    config_path: str | Path | None = None,
    slot_config: Any = None,
) -> list[str] | str | None:
    if config is None:
        return None
    top_k_cfg = (config.get("performance_evaluation") or {}).get("top_k_strategy") or {}
    if isinstance(top_k_cfg, Mapping) and top_k_cfg.get("enabled"):
        top_k = top_k_cfg.get("top_k")
        if type(top_k) is int and top_k >= 2:
            return "contradicted"
    candidates: list[Any] = []
    for container in (config, config.get("stock_pool")):
        if isinstance(container, Mapping):
            for key in ("code_list", "instruments", "stocks"):
                if container.get(key) is not None:
                    candidates.append(container.get(key))
    slot_obj, slot_mapping = _load_slot_config_obj(
        config, config_path=config_path, slot_config=slot_config,
    )
    universe_opts = None
    if slot_obj is not None:
        universe_opts = slot_obj.universe.options
    elif isinstance(slot_mapping, Mapping):
        universe = slot_mapping.get("universe")
        if isinstance(universe, Mapping) and isinstance(universe.get("options"), Mapping):
            universe_opts = universe.get("options")
    if isinstance(universe_opts, Mapping):
        for key in ("instruments", "symbols", "code_list"):
            if universe_opts.get(key) is not None:
                candidates.append(universe_opts.get(key))
    resolved: list[str] | None = None
    for raw in candidates:
        if not isinstance(raw, (list, tuple)) or not raw:
            continue
        unique = list(dict.fromkeys(str(item).strip() for item in raw if _name(str(item).strip())))
        if len(unique) >= 2:
            return "contradicted"
        if len(unique) == 1:
            if resolved is not None and resolved != unique:
                return "contradicted"
            resolved = unique
    return resolved


def _composite_scores_artifact_verified(
    row: Mapping,
    artifacts: Sequence[Mapping],
) -> bool:
    import numpy as np
    import pandas as pd

    sname, fname = row.get("synthesis_name"), row.get("factor_name")
    instruments = row.get("instruments")
    for art in artifacts:
        if not isinstance(art, Mapping) or art.get("artifact_type") != "composite_scores":
            continue
        if art.get("synthesis_name") != sname or art.get("factor_name") != fname:
            continue
        raw_path = art.get("path")
        digest = art.get("sha256")
        if not _name(raw_path) or not _is_sha256_hex(digest):
            return False
        art_path = Path(raw_path)
        if not art_path.is_file() or _sha256_file(art_path) != digest:
            return False
        try:
            frame = pd.read_csv(art_path)
        except Exception:
            return False
        stock_col = "stock_code" if "stock_code" in frame.columns else "instrument_id"
        if stock_col not in frame.columns or "composite_value" not in frame.columns or frame.empty:
            return False
        scores = pd.to_numeric(frame["composite_value"], errors="coerce").to_numpy(dtype=float)
        finite_count = int(np.isfinite(scores).sum())
        unique_stocks = sorted({str(v).strip() for v in frame[stock_col].dropna() if _name(str(v).strip())})
        return finite_count == row.get("finite_score_count") and unique_stocks == instruments
    return False


def _single_instrument_verified(
    row: Mapping,
    config: Mapping | None,
    *,
    config_path: str | Path | None = None,
    slot_config: Any = None,
    artifacts: Sequence[Mapping] = (),
) -> bool:
    if row.get("skip_reason") != "single_instrument" or row.get("stock_count") != 1:
        return False
    instruments = row.get("instruments")
    if not isinstance(instruments, list) or len(instruments) != 1 or not _name(instruments[0]):
        return False
    requested = _requested_single_instrument(
        config, config_path=config_path, slot_config=slot_config,
    )
    if requested == "contradicted":
        return False
    if requested is not None:
        return requested == instruments
    return _composite_scores_artifact_verified(row, artifacts)


def _validate_composite_only_row(
    row: Mapping,
    *,
    config: Mapping | None,
    expected_methods: Mapping[tuple[str, str], str],
    effective_frequency: str,
    config_path: str | Path | None = None,
    slot_config: Any = None,
    slot_id: str | None = None,
    market_db: str | None = None,
    artifacts: Sequence[Mapping] = (),
) -> bool:
    if not _count(row.get("finite_score_count"), minimum=1):
        return False
    row_freq = row.get("frequency")
    row_freq_canonical = None
    if row_freq is not None:
        from utils.time_axis import Frequency

        parsed_row_freq = Frequency.parse(str(row_freq).strip())
        if not parsed_row_freq.is_valid or parsed_row_freq.canonical != effective_frequency:
            return False
        row_freq_canonical = parsed_row_freq.canonical
    skip_reason = row.get("skip_reason")
    if skip_reason is not None and skip_reason not in _COMPOSITE_SKIP_REASONS:
        return False
    if config is None:
        if skip_reason is None:
            return False
        if skip_reason == "non_daily_frequency":
            return row_freq_canonical is not None and row_freq_canonical != "1d"
        if skip_reason == "single_instrument":
            return _single_instrument_verified(row, None, artifacts=artifacts)
        return True
    key = row.get("synthesis_name"), row.get("factor_name")
    if key not in expected_methods:
        return False
    allowed = _allowed_composite_only_reasons(
        config,
        method=expected_methods[key],
        effective_frequency=effective_frequency,
        slot_config=slot_config,
        slot_id=slot_id,
        market_db=market_db,
    )
    if skip_reason == "single_instrument":
        return _single_instrument_verified(
            row, config, config_path=config_path, slot_config=slot_config, artifacts=artifacts,
        )
    if skip_reason is None:
        if not allowed:
            return False
        if allowed == {"synthesis_reuse"}:
            return _synthesis_reuse_verified(row, config, synthesis_name=key[0], config_path=config_path)
        return True
    if skip_reason not in allowed:
        return False
    if skip_reason == "synthesis_reuse":
        return _synthesis_reuse_verified(row, config, synthesis_name=key[0], config_path=config_path)
    return True


def _synthesis_content(
    metrics: Mapping,
    completed: int,
    config: Mapping | None,
    *,
    config_path: str | Path | None = None,
    slot_config: Any = None,
    slot_id: str | None = None,
    frequency: str | None = None,
    market_db: str | None = None,
    artifacts: Sequence[Mapping] = (),
) -> None:
    expected_methods = expected_synthesis_items(config) if config is not None else {}
    effective_frequency = _resolve_synthesis_frequency(
        config, config_path=config_path, slot_config=slot_config, frequency=frequency,
    )
    seen: set[tuple[str, str]] = set()
    for row in metrics.get("ic_summaries", []):
        periods = row.get("periods") or {}
        ir_periods = row.get("ir_periods") or {}
        valid = (
            any(_finite(row.get(key)) for key in ("overall_ic", "overall_ir"))
            or isinstance(periods, dict) and any(
                isinstance(stats, dict) and _count(stats.get("sample_count"), minimum=1)
                and _finite(stats.get("mean_ic")) for stats in periods.values()
            )
            or isinstance(ir_periods, dict) and any(_finite(value) for value in ir_periods.values())
        )
        _record_synthesis_item(seen, row, valid)
    for row in metrics.get("synthesis_diagnostics", []):
        # Model and reranker diagnostics accompany the scored output. A skipped
        # IC evaluation is allowed only when justified by the request or verified runtime condition.
        if row.get("evaluation_mode") == "composite_only":
            valid = _validate_composite_only_row(
                row,
                config=config,
                expected_methods=expected_methods,
                effective_frequency=effective_frequency,
                config_path=config_path,
                slot_config=slot_config,
                slot_id=slot_id,
                market_db=market_db,
                artifacts=artifacts,
            )
            _record_synthesis_item(seen, row, valid)
    if config is not None:
        expected = set(expected_methods)
        if not expected or seen != expected:
            raise ValueError(
                f"synthesis item coverage mismatch: missing={sorted(expected - seen)!r}, "
                f"unexpected={sorted(seen - expected)!r}"
            )
    _match_completed("factor-synthesis", len(seen), completed)


def _record_synthesis_item(seen: set[tuple[str, str]], row: dict, valid: bool) -> None:
    key = row.get("synthesis_name"), row.get("factor_name")
    if not all(map(_name, key)) or row.get("status") != "succeeded" or not valid:
        raise ValueError(f"invalid synthesis item: {key!r}")
    if key in seen:
        raise ValueError(f"duplicate synthesis item: {key!r}")
    seen.add(key)


def apply_analysis_cli_overrides(
    config: dict[str, Any],
    *,
    factor: str | None = None,
    pools: Sequence[str] = (),
    use_dynamic: bool = False,
    materialize_only: bool = False,
    mining_only: bool = False,
    causal_gate_policy: str | None = None,
    filter_pools: bool = False,
) -> dict[str, Any]:
    if factor:
        config["factor_file"] = factor
    if use_dynamic:
        config["use_dynamic_pools"] = True
    if materialize_only:
        config["materialize_only"] = True
    if mining_only:
        config["mining_only"] = True
    if causal_gate_policy:
        config["causal_gate_policy"] = causal_gate_policy
    if filter_pools and pools and isinstance(config.get("dynamic_pools"), Mapping):
        config["dynamic_pools"] = {
            name: settings for name, settings in config["dynamic_pools"].items()
            if name in pools
        }
    return config


def resolve_effective_analysis_config(
    raw_config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
    factor: str | None = None,
    pools: Sequence[str] = (),
    use_dynamic: bool = False,
    materialize_only: bool = False,
    mining_only: bool = False,
    causal_gate_policy: str | None = None,
    filter_pools: bool = False,
    dynamic_pool_available: bool = True,
    parse_pool_filters: bool = False,
) -> dict[str, Any]:
    from backtest_and_analysis.config import CONFIG

    effective: dict[str, Any] = dict(CONFIG)
    if raw_config is None and config_path is not None:
        raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if raw_config is not None:
        configured_pools = raw_config.get("dynamic_pools")
        has_dynamic = (
            dynamic_pool_available
            and isinstance(configured_pools, Mapping)
            and bool(configured_pools)
        )
        # Match run_analysis_main: nested only when dynamic_pools is non-empty or database is present.
        nested = has_dynamic or ("database" in raw_config)
        if nested:
            from utils.data_store import resolve_database_config

            resolved_doc = resolve_database_config(dict(raw_config))
            if resolved_doc.get("data_bindings") is not None:
                effective["data_bindings"] = resolved_doc["data_bindings"]
            db_cfg = resolved_doc.get("database")
            if isinstance(db_cfg, Mapping):
                effective["factor_table"] = db_cfg.get("factor_table", effective.get("factor_table"))
                effective["price_table"] = db_cfg.get("price_table", effective.get("price_table"))
                effective["fundamental_table"] = db_cfg.get("fundamental_table", effective.get("fundamental_table"))
                effective["db_backend"] = db_cfg.get("db_backend", "auto")
                effective["duckdb_path"] = db_cfg.get("duckdb_path")
                effective["parquet_dir"] = db_cfg.get("parquet_dir")
                effective["use_duckdb_cache"] = db_cfg.get("use_duckdb_cache", True)
                effective["duckdb_cache_size_mb"] = db_cfg.get("duckdb_cache_size_mb", 2000)
                if db_cfg.get("data_config_path") and config_path is not None:
                    effective["data_config_path"] = str(
                        (Path(config_path).resolve().parent / Path(str(db_cfg["data_config_path"])).expanduser()).resolve()
                    )
                else:
                    effective["data_config_path"] = db_cfg.get("data_config_path")
                effective["source_policy"] = db_cfg.get("source_policy")
            for section in ("analysis_settings", "analysis_period", "database_safety"):
                payload = resolved_doc.get(section)
                if isinstance(payload, Mapping):
                    effective.update(payload)
            results_db = resolved_doc.get("results_database")
            if isinstance(results_db, Mapping) and results_db:
                effective["results_output_table"] = results_db.get("output_table", "factor_analysis_results")
                effective["results_monthly_output_table"] = results_db.get(
                    "monthly_output_table", "monthly_pool_results"
                )
                r_backend = str(results_db.get("db_backend", "auto") or "auto").lower()
                if r_backend == "sqlite":
                    raise ValueError("results_database SQLite backend is no longer supported; configure DuckDB")
                if r_backend not in {"auto", "duckdb"}:
                    raise ValueError("results_database.db_backend must be 'duckdb' or 'auto'")
                r_duckdb = results_db.get("duckdb_path")
                if not r_duckdb:
                    raise ValueError("results_database.duckdb_path is required for DuckDB results")
                effective["results_db_backend"] = "duckdb"
                effective["results_duckdb_path"] = r_duckdb
                effective["results_parquet_dir"] = results_db.get("parquet_dir")
                effective["results_export_parquet"] = results_db.get("export_parquet", True)
                if "factor_values_store" in results_db:
                    effective["factor_values_store"] = results_db["factor_values_store"]
            if parse_pool_filters:
                from utils.config_loader import create_pool_configs_from_json

                effective["dynamic_pools"] = create_pool_configs_from_json(dict(resolved_doc))
            else:
                effective["dynamic_pools"] = (
                    {
                        name: dict(options)
                        for name, options in configured_pools.items()
                        if isinstance(options, Mapping) and options.get("enabled", True)
                    }
                    if has_dynamic
                    else {}
                )
            effective["use_dynamic_pools"] = has_dynamic
        else:
            effective.update(raw_config)
            if effective.get("data_config_path") and config_path is not None:
                effective["data_config_path"] = str(
                    (Path(config_path).resolve().parent / Path(str(effective["data_config_path"])).expanduser()).resolve()
                )
    return apply_analysis_cli_overrides(
        effective,
        factor=factor,
        pools=pools,
        use_dynamic=use_dynamic,
        materialize_only=materialize_only,
        mining_only=mining_only,
        causal_gate_policy=causal_gate_policy,
        filter_pools=filter_pools,
    )


def _validate_analysis_stat_summary(row: Mapping, observed_mode: str) -> None:
    summary = row.get("stat_summary")
    if not isinstance(summary, dict) or not summary:
        raise ValueError(f"analysis item {row.get('factor_name')!r} missing stat_summary")
    allowed_tables = {"ic"} if observed_mode == "mining_only" else set(_ANALYSIS_STAT_TABLES)
    total_samples = 0
    for table_name, table_metrics in summary.items():
        if table_name not in allowed_tables or not isinstance(table_metrics, dict) or not table_metrics:
            raise ValueError(
                f"invalid analysis stat_summary table {table_name!r} for {row.get('factor_name')!r}"
            )
        for metric_key, stats in table_metrics.items():
            if (
                not _name(metric_key)
                or not isinstance(stats, dict)
                or not _count(stats.get("sample_count"), minimum=1)
                or not _finite(stats.get("mean"))
            ):
                raise ValueError(
                    f"invalid analysis stat_summary metric {table_name}.{metric_key!r} for {row.get('factor_name')!r}"
                )
            total_samples += stats["sample_count"]
    if total_samples < 1:
        raise ValueError(f"analysis item {row.get('factor_name')!r} has zero finite stats")
    if "finite_stat_count" in row and row["finite_stat_count"] is not None:
        if row["finite_stat_count"] != total_samples:
            raise ValueError(f"analysis item {row.get('factor_name')!r} finite_stat_count mismatch")


def _analysis_content(
    metrics: Mapping, completed: int, config: Mapping | None, *,
    config_path: str | Path | None = None, factor: str | None,
    pools: Sequence[str], use_dynamic: bool, materialize_only: bool, mining_only: bool,
) -> None:
    rows = _records(metrics.get("analysis_items"), "analysis_items")
    settings = resolve_effective_analysis_config(
        config,
        config_path=config_path,
        factor=factor,
        pools=pools,
        use_dynamic=use_dynamic,
        materialize_only=materialize_only,
        mining_only=mining_only,
        filter_pools=True,
    )
    configured_pools = settings.get("dynamic_pools") or {}
    factor = settings.get("factor_file")
    factor_class = settings.get("factor_class")
    allowed_sources = None
    if not factor and settings.get("factor_dir"):
        allowed_sources = {
            str(path.resolve()) for path in Path(settings["factor_dir"]).glob("*.py")
            if path.is_file() and not path.name.startswith("__")
        }
    mode = (
        "materialize_only" if settings.get("materialize_only")
        else "mining_only" if settings.get("mining_only") else "evaluation"
    )
    dynamic = bool(settings.get("use_dynamic_pools"))
    if dynamic:
        enabled = {
            name for name, options in configured_pools.items()
            if isinstance(options, Mapping) and options.get("enabled", True)
        }
        expected_pools = set(pools) if pools else enabled
        if not expected_pools or not expected_pools <= enabled:
            raise ValueError("analysis request has unavailable dynamic pools")
        if mode != "evaluation" and len(expected_pools) != 1:
            raise ValueError(f"analysis {mode} requires exactly one dynamic pool")
    else:
        if pools or mode != "evaluation":
            raise ValueError("analysis request requires dynamic pools")
        expected_pools = {None}

    expected_source = str(Path(factor).resolve()) if factor else None
    seen: dict[tuple[str, str], set[str | None]] = {}
    for row in rows:
        name, source, pool = row.get("factor_name"), row.get("factor_source"), row.get("pool")
        observed_mode = row.get("mode")
        if (
            not _name(name) or not _name(source) or (pool is not None and not _name(pool))
            or row.get("status") != "succeeded"
            or observed_mode not in ("evaluation", "materialize_only", "mining_only")
            or (mode is not None and mode != observed_mode)
            or (observed_mode != "evaluation" and pool is None)
            or (expected_source is not None and source != expected_source)
            or (allowed_sources is not None and source not in allowed_sources)
            or (factor_class is not None and name != factor_class)
        ):
            raise ValueError(f"invalid analysis item: {name!r}, pool={pool!r}")
        if observed_mode == "materialize_only":
            if not _count(row.get("materialized_rows"), minimum=1):
                raise ValueError(f"invalid analysis item: {name!r}, pool={pool!r}")
        else:
            _validate_analysis_stat_summary(row, observed_mode)
        factor_pools = seen.setdefault((source, name), set())
        if pool in factor_pools:
            raise ValueError(f"duplicate analysis item: {name!r}, pool={pool!r}")
        factor_pools.add(pool)
    if expected_pools is not None and any(actual != expected_pools for actual in seen.values()):
        raise ValueError("analysis pool coverage mismatch")
    if mode == "materialize_only" and len(seen) != 1:
        raise ValueError("materialization requires exactly one factor")
    _match_completed("single-factor-analysis", sum(map(len, seen.values())), completed)


def validate_deploy_content(
    action: str, metrics: Mapping, completed: int, *,
    expected_target: str | None = None, expected_bundle_hash: str | None = None,
) -> None:
    target, digest = metrics.get("target_id"), metrics.get("bundle_hash")
    if (
        not _name(target) or not _is_sha256_hex(digest)
        or (expected_target is not None and target != expected_target)
        or (expected_bundle_hash is not None and digest != expected_bundle_hash)
    ):
        raise ValueError("invalid deployment target_id or bundle_hash")

    field = "deploy_manifest" if action == "export-strategy" else "alignment_report"
    report = metrics.get(field)
    if not isinstance(report, dict) or (report.get("target_id"), report.get("bundle_hash")) != (target, digest):
        raise ValueError(f"invalid {field} identity")
    if action == "export-strategy":
        from targets.contracts import CompatibilityReport, DeployArtifactManifest

        compat = metrics.get("compatibility")
        if not isinstance(compat, dict) or compat.get("compatible") is not True:
            raise ValueError("invalid deploy compatibility")
        try:
            compatibility = CompatibilityReport(**compat)
        except TypeError as exc:
            raise ValueError("invalid deploy compatibility fields") from exc
        manifest = DeployArtifactManifest.from_mapping(report)
        if (compatibility.target_id, compatibility.bundle_hash, compatibility.slot_id) != (
            target, digest, manifest.slot_id
        ) or not manifest.artifacts:
            raise ValueError("invalid deploy compatibility or manifest artifacts")
    else:
        from targets.contracts import AlignmentReport

        try:
            alignment = AlignmentReport(**report)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid alignment_report: {exc}") from exc
        names = {check.name for check in alignment.checks}
        if not alignment.passed or not names or len(names) != len(alignment.checks) or any(
            check.code for check in alignment.checks
        ):
            raise ValueError("alignment_report requires distinct passing checks")
        _match_completed(action, len(names), completed)
        return
    _match_completed(action, 1, completed)


def validate_monitor_content(metrics: Mapping) -> None:
    if (
        metrics.get("available") is not True or metrics.get("backend") != "duckdb"
        or not all(_count(metrics.get(key)) for key in ("table_count", "factor_count"))
    ):
        raise ValueError("monitor result requires available DuckDB and valid counts")


def validate_bar_content(metrics: Mapping) -> None:
    if (
        not isinstance(metrics, dict) or metrics.get("result_schema_version") != "bar_evaluation.v2"
        or metrics.get("analysis_mode") not in ("time_series", "cross_sectional")
    ):
        raise ValueError("invalid Bar evaluation schema or mode")
    identity = metrics.get("result_identity")
    if not isinstance(identity, dict) or not _is_sha256_hex(identity.get("identity_hash")):
        raise ValueError("invalid Bar evaluation identity")
    totals = metrics.get("exclusion_summary")
    if not isinstance(totals, dict) or not _count(totals.get("total_count"), minimum=1):
        raise ValueError("Bar evaluation has no label counts")
    rows = _records(metrics.get("ic_by_horizon"), "ic_by_horizon")
    for row in [totals, *rows]:
        if not (
            all(_count(row.get(key)) for key in ("total_count", "valid_count", "excluded_count"))
            and row["total_count"] == row["valid_count"] + row["excluded_count"]
        ):
            raise ValueError("invalid Bar evaluation label counts")
    for row in rows:
        if not _count(row.get("horizon_bars"), minimum=1) or row.get("ic_status") not in (
            "ok", "no_valid_instruments", "no_valid_cross_sections", "constant_factor",
            "constant_return", "insufficient_samples", "insufficient_cross_section",
        ):
            raise ValueError("invalid Bar evaluation horizon or IC status")
        estimates = ("spearman_ic", "mean_spearman_ic", "mean_ic")
        if row["ic_status"] == "ok" and not any(_finite(row.get(key)) for key in estimates):
            raise ValueError("Bar evaluation reports IC success without a finite estimate")

    quantiles = metrics.get("quantile_stats")
    if not isinstance(quantiles, list) or any(
        not isinstance(row, dict) or not _count(row.get("quantile"), minimum=1)
        or not _finite(row.get("mean_return_pct")) for row in quantiles
    ):
        raise ValueError("invalid Bar evaluation quantile statistics")


def _has_number(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_number(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_number(item) for item in value)
    return _finite(value)


def validate_bound_content(
    action: str, metrics: Mapping, completed: int, *, config: Mapping | None = None,
    config_path: str | Path | None = None, factor: str | None = None,
    pools: Sequence[str] = (), use_dynamic: bool = False,
    materialize_only: bool = False, mining_only: bool = False,
    slot_config: Any = None, slot_id: str | None = None,
    frequency: str | None = None, market_db: str | None = None,
    artifacts: Sequence[Mapping] = (),
) -> None:
    if action == "factor-synthesis":
        _synthesis_content(
            metrics, completed, config, config_path=config_path,
            slot_config=slot_config, slot_id=slot_id,
            frequency=frequency, market_db=market_db,
            artifacts=artifacts,
        )
    elif action == "single-factor-analysis":
        _analysis_content(
            metrics, completed, config, config_path=config_path,
            factor=factor, pools=pools, use_dynamic=use_dynamic,
            materialize_only=materialize_only, mining_only=mining_only,
        )
    elif action in ("export-strategy", "verify-alignment", "verify-target-alignment"):
        request = config or {}
        validate_deploy_content(
            action, metrics, completed, expected_target=request.get("target"),
            expected_bundle_hash=request.get("bundle_hash"),
        )
    elif action in ("monitor-refresh", "time-series-analysis"):
        (validate_monitor_content if action == "monitor-refresh" else validate_bar_content)(metrics)
        _match_completed(action, 1, completed)
    else:
        seen = set()
        for row in _records(metrics.get("items"), "items"):
            item_id, values = row.get("item_id"), row.get("metrics")
            if (
                not _name(item_id) or item_id in seen or row.get("status") != "succeeded"
                or not isinstance(values, dict) or not _has_number(values)
            ):
                raise ValueError(f"invalid {action} result item: {item_id!r}")
            seen.add(item_id)
        _match_completed(action, len(seen), completed)
