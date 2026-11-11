"""Data loading and panel construction for symbolic factor mining."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from utils.data_store import binding_contract

try:
    import duckdb
    from utils.duckdb_results_writer import connect_read_only_with_retry as _connect_data_read_only
except ImportError:  # pragma: no cover - environment dependent
    duckdb = None

from utils.dynamic_stock_pool import (
    DynamicStockPoolManager,
    apply_date_level_pool_filter_to_long_df,
)
from utils.factor_base import apply_pit_adjustment
from utils.pit import MONEY_NORMALIZATION_VERSION, normalize_price_money

from .config import AnalysisContext
from .feature_space import FeatureSpec, build_feature_space
from .operator_runtime import aggregate_intraday_feature_series


# New derived caches only; retain historical caches under their original identities.
_PIT_CACHE_SEMANTICS = MONEY_NORMALIZATION_VERSION


# Connections inherit the shared data-store memory/thread budget. Redirect
# temporary spill to the configured volume without overriding that budget.
_DUCKDB_TEMP_DIR = os.environ.get(
    "QUANTATION_DUCKDB_TEMP_DIR", "./data/duckdb_tmp"
)


def _configure_duckdb_spill(conn) -> None:
    """Configure the temporary spill directory for an initialized connection."""
    if _DUCKDB_TEMP_DIR:
        try:
            Path(_DUCKDB_TEMP_DIR).mkdir(parents=True, exist_ok=True)
            conn.execute(f"SET temp_directory='{_DUCKDB_TEMP_DIR}'")
        except Exception:
            pass


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _input_duckdb_path(context: AnalysisContext) -> str:
    path = context.runtime_config.get("duckdb_path") or binding_contract().get("input_db")
    if not path:
        raise ValueError("duckdb_path is required; SQLite is no longer supported")
    if duckdb is None:
        raise RuntimeError("duckdb is required for symbolic mining")
    return str(path)


def _panel_cache_path(
    config: dict,
    context: AnalysisContext,
    stocks: List[str],
    anchor_factor_names: List[str],
    *,
    drop_missing_labels: bool = True,
) -> Path | None:
    cache_dir = context.runtime_config.get("cache_dir")
    if not cache_dir:
        return None
    pool_snapshot_path = Path(
        str(context.runtime_config.get("pool_snapshot_path") or "")
    ).expanduser()
    pool_snapshot_stat = (
        pool_snapshot_path.stat() if pool_snapshot_path.is_file() else None
    )
    payload = {
        "pit_cache_semantics": _PIT_CACHE_SEMANTICS,
        "data_bindings": binding_contract(),
        "drop_missing_labels": drop_missing_labels,
        "analysis_config": context.config_path,
        "analysis_period": {
            "start_date": context.runtime_config.get("start_date"),
            "end_date": context.runtime_config.get("end_date"),
        },
        "features": config.get("features", {}),
        "label": config.get("label", {}),
        "objective": config.get("objective", {}),
        "search_windows": config.get("search", {}).get("windows", []),
        "stocks": list(stocks),
        "anchor_factor_names": list(anchor_factor_names),
        "pool_snapshot": {
            "path": str(pool_snapshot_path),
            "size": pool_snapshot_stat.st_size if pool_snapshot_stat else 0,
            "mtime_ns": pool_snapshot_stat.st_mtime_ns if pool_snapshot_stat else 0,
        },
    }
    digest = hashlib.sha1(
        json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return Path(cache_dir) / f"symbolic_panel_{digest}.pkl"


def _anchor_panel_cache_path(
    cache_dir: str | None,
    anchor_db_path: str,
    anchor_factor_names: List[str],
    stocks: List[str],
    start_date: str,
    end_date: str,
) -> Path | None:
    """Cache key for the anchor wide-panel.

    Deliberately excludes the per-run feature/objective/search config that busts
    ``_panel_cache_path`` -- the anchor panel only depends on the anchor set, the
    window and the universe, so this key hits for every run in a batch after the
    first one builds it.
    """
    if not cache_dir:
        return None
    payload = {
        "anchor_db": str(anchor_db_path),
        "data_bindings": binding_contract(),
        "anchors": sorted(str(name) for name in anchor_factor_names),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "universe": sorted(str(code) for code in stocks),
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return Path(cache_dir) / f"anchor_panel_{digest}.parquet"


def _intraday_feature_cache_path(
    context: AnalysisContext,
    spec: FeatureSpec,
    stocks: List[str],
) -> Path | None:
    cache_dir = context.runtime_config.get(
        "intraday_cache_dir"
    ) or context.runtime_config.get("cache_dir")
    if not cache_dir:
        return None
    database = Path(str(context.runtime_config.get("duckdb_path") or "")).expanduser()
    stat = database.stat() if database.is_file() else None
    source_identity = str(context.runtime_config.get("data_snapshot_hash") or "")
    if not source_identity:
        source_identity = json.dumps(
            {
                "path": str(database),
                "size": stat.st_size if stat else 0,
                "mtime_ns": stat.st_mtime_ns if stat else 0,
                "duckdb_path": context.runtime_config.get("duckdb_path"),
            },
            sort_keys=True,
        )
    payload = {
        "pit_cache_semantics": _PIT_CACHE_SEMANTICS,
        "data_bindings": binding_contract(),
        "source": source_identity,
        "table": "price_data_5m",
        "start_date": context.runtime_config.get("start_date"),
        "end_date": context.runtime_config.get("end_date"),
        "stocks": sorted(str(code) for code in stocks),
        "pit_adjustment": context.raw_config.get("pit_adjustment", {}),
        "spec": {
            "name": spec.name,
            "source": spec.source,
            "field": spec.field,
            "frequency": spec.frequency,
            "aggregator": spec.aggregator,
            "dependencies": list(spec.dependencies),
        },
    }
    digest = hashlib.sha1(
        json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return Path(cache_dir) / "intraday_features" / f"{spec.name}_{digest}.parquet"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _stock_filter(codes: Iterable[str]) -> str:
    values = "', '".join(str(code) for code in codes)
    return f"('{values}')"


def _table_columns(context: AnalysisContext, table: str) -> set[str]:
    conn = _connect_data_read_only(_input_duckdb_path(context))
    try:
        escaped_table = table.replace("'", "''")
        rows = conn.execute(f"PRAGMA table_info('{escaped_table}')").fetchall()
    finally:
        conn.close()
    return {str(row[1]) for row in rows}


def resolve_universe(analysis_context: AnalysisContext) -> List[str]:
    config = analysis_context.runtime_config
    snapshot = _pool_snapshot(analysis_context)
    if snapshot:
        return sorted({code for codes in snapshot.values() for code in codes})
    if config.get("code_list"):
        return list(config["code_list"])

    pool_configs = analysis_context.pool_configs
    if pool_configs:
        manager = DynamicStockPoolManager(
            None,
            fundamental_table=config.get("fundamental_table", "factor_data"),
            price_table=config.get("price_table", "price_data_daily"),
            db_backend="duckdb",
            duckdb_path=_input_duckdb_path(analysis_context),
            parquet_dir=config.get("parquet_dir"),
            use_duckdb_cache=config.get("use_duckdb_cache", True),
            duckdb_cache_size_mb=config.get("duckdb_cache_size_mb", 2000),
        )
        stocks = set()
        try:
            first_pool = next(iter(pool_configs.values()))
            pools = manager.get_stock_pools_for_period(
                config["start_date"],
                config["end_date"],
                first_pool.get("filters", []),
                first_pool.get("max_stocks"),
                first_pool.get("sort_by"),
                first_pool.get("ascending", False),
                rebalance_period_months=first_pool.get("rebalance_period_months", 1),
                rebalance_schedule=first_pool.get("rebalance_schedule", "quarterly"),
            )
            for pool_stocks in pools.values():
                stocks.update(pool_stocks)
        finally:
            connection = getattr(manager, "_duckdb_conn", None)
            if connection is not None:
                connection.close()
        if stocks:
            return sorted(stocks)

    raise ValueError("Unable to resolve mining universe from analysis config")


def _pool_snapshot(context: AnalysisContext) -> dict[str, list[str]]:
    raw_path = str(context.runtime_config.get("pool_snapshot_path") or "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    pools = payload.get("pools") if isinstance(payload, dict) else None
    if not isinstance(pools, dict):
        return {}
    pool_name = next(iter(context.pool_configs), "")
    selected = pools.get(pool_name)
    if not isinstance(selected, dict):
        return {}
    return {
        str(period): [str(code) for code in codes]
        for period, codes in selected.items()
        if isinstance(codes, list)
    }


def _query_table(
    context: AnalysisContext, table: str, fields: List[str], stocks: List[str]
) -> pd.DataFrame:
    config = context.runtime_config
    available_columns = _table_columns(context, table)
    price_frequency = ('5m' if table == 'price_data_5m' else 'daily'
                       if table in {config.get('price_table', 'price_data_daily'), 'price_data_daily'} else None)
    helper_paused = (price_frequency is not None and 'money' in fields
                     and 'paused' not in fields and 'paused' in available_columns)
    if helper_paused:
        fields = [*fields, 'paused']
    selected = ["date", "stock_code"] + [
        field
        for field in fields
        if field not in {"date", "stock_code", "code"} and field in available_columns
    ]
    if "date" not in available_columns or "stock_code" not in available_columns:
        raise ValueError(f"Table {table} must contain date and stock_code columns")
    selected_sql = ", ".join(_quote(field) for field in selected)
    sql = f"""
        SELECT {selected_sql}
        FROM {table}
        WHERE date >= '{config["start_date"]}'
          AND date <= '{config["end_date"]}'
          AND stock_code IN {_stock_filter(stocks)}
        ORDER BY date, stock_code
    """

    conn = _connect_data_read_only(_input_duckdb_path(context))
    try:
        _configure_duckdb_spill(conn)
        df = conn.execute(sql).df()
    finally:
        conn.close()

    if df.empty:
        return df
    df = df.rename(columns={"stock_code": "code"})
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date", "code"])
    if price_frequency is not None:
        df = normalize_price_money(df, source='local', frequency=price_frequency)
    return df.drop(columns='paused') if helper_paused else df


def load_daily_price_panel(
    context: AnalysisContext, fields: List[str], stocks: List[str]
) -> pd.DataFrame:
    table = context.runtime_config.get("price_table", "price_data_daily")
    if table == "price_data":
        table = "price_data_daily"
    pit = context.raw_config.get("pit_adjustment", {})
    factor_column = str(pit.get("factor_column", "factor"))
    query_fields = list(fields)
    if pit.get("enabled", True) and factor_column not in query_fields:
        query_fields.append(factor_column)
    df = _query_table(context, table, query_fields, stocks)
    if df.empty:
        return df
    if pit.get("enabled", True):
        df = apply_pit_adjustment(df, factor_column)
    rename = {field: f"daily_{field}" for field in fields if field in df.columns}
    return df.rename(columns=rename)


def load_factor_data_panel(
    context: AnalysisContext, fields: List[str], stocks: List[str]
) -> pd.DataFrame:
    if not fields:
        return pd.DataFrame()
    table = context.runtime_config.get(
        "factor_table", context.runtime_config.get("fundamental_table", "factor_data")
    )
    df = _query_table(context, table, fields, stocks)
    if df.empty:
        return df
    rename = {field: f"fd_{field}" for field in fields if field in df.columns}
    return df.rename(columns=rename)


def load_intraday_5m_panel(
    context: AnalysisContext, fields: List[str], stocks: List[str]
) -> pd.DataFrame:
    if not fields:
        return pd.DataFrame()
    pit = context.raw_config.get("pit_adjustment", {})
    factor_column = str(pit.get("factor_column", "factor"))
    query_fields = list(fields)
    if pit.get("enabled", True) and factor_column not in query_fields:
        query_fields.append(factor_column)
    panel = _query_table(context, "price_data_5m", query_fields, stocks)
    return (
        apply_pit_adjustment(panel, factor_column)
        if pit.get("enabled", True)
        else panel
    )


def load_anchor_factor_panel(
    anchor_db_path: str,
    anchor_factor_names: List[str],
    stocks: List[str],
    start_date: str,
    end_date: str,
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load mainline-anchor factor values into a wide ``anchor__*`` panel.

    The pivot is pushed into DuckDB (one ``MAX(CASE WHEN factor_name = ...)``
    column per anchor, grouped by ``(date, stock_code)``) so the wide panel is
    materialised directly -- no tens-of-millions-row tall transfer and no
    single-threaded pandas pivot.  Anchors absent over this window yield no
    column (per-date residualization uses whichever anchors are present,
    matching the marginal gate).

    When ``cache_dir`` is set the result is cached as parquet keyed only by the
    anchor set, window and universe (see :func:`_anchor_panel_cache_path`), so
    every run in a batch after the first reads it back in seconds.  DuckDB
    connection / query failures propagate.
    """
    if not anchor_factor_names:
        return pd.DataFrame()
    cache_path = _anchor_panel_cache_path(
        cache_dir, anchor_db_path, anchor_factor_names, stocks, start_date, end_date
    )
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)
    if duckdb is None:
        raise RuntimeError("duckdb is required to load anchor factor values")
    case_columns = ",\n        ".join(
        f"MAX(CASE WHEN factor_name = {_sql_literal(name)} THEN factor_value END) "
        f"AS {_quote('anchor__' + name)}"
        for name in anchor_factor_names
    )
    names_in = ", ".join(_sql_literal(name) for name in anchor_factor_names)
    sql = f"""
        SELECT date, stock_code AS code,
        {case_columns}
        FROM factor_values
        WHERE factor_name IN ({names_in})
          AND date >= {_sql_literal(start_date)}
          AND date <= {_sql_literal(end_date)}
          AND stock_code IN {_stock_filter(stocks)}
          AND factor_value IS NOT NULL
        GROUP BY date, stock_code
    """
    conn = _connect_data_read_only(str(anchor_db_path))
    try:
        _configure_duckdb_spill(conn)
        wide = conn.execute(sql).df()
    finally:
        conn.close()
    if wide.empty:
        return wide
    wide["date"] = pd.to_datetime(wide["date"], format="mixed", errors="coerce")
    wide = wide.dropna(subset=["date", "code"])
    wide = wide.dropna(axis=1, how="all")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(cache_path, index=False)
    return wide


def aggregate_5m_to_daily_features(
    df: pd.DataFrame, specs: Dict[str, FeatureSpec]
) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["trading_date"] = work["date"].dt.normalize()
    grouped = work.groupby(["trading_date", "code"], sort=True)

    selected = [
        spec
        for spec in specs.values()
        if spec.source == "price_5m" and spec.field in work.columns
    ]
    simple = {
        spec.name: pd.NamedAgg(column=spec.field, aggfunc=str(spec.aggregator))
        for spec in selected
        if spec.aggregator
        in {"first", "last", "max", "min", "median", "mean", "std", "sum"}
    }
    pieces = [grouped.agg(**simple)] if simple else []

    range_specs = [spec for spec in selected if spec.aggregator == "range"]
    if range_specs:
        fields = sorted({spec.field for spec in range_specs})
        extrema = grouped[fields].agg(["min", "max"])
        pieces.append(
            pd.DataFrame(
                {
                    spec.name: extrema[(spec.field, "max")]
                    - extrema[(spec.field, "min")]
                    for spec in range_specs
                }
            )
        )

    batched = set(simple) | {spec.name for spec in range_specs}
    for spec in selected:
        if spec.name in batched:
            continue
        series = aggregate_intraday_feature_series(work, grouped, spec)
        if series is None:
            continue
        pieces.append(series.rename(spec.name))

    if not pieces:
        return pd.DataFrame()

    result = pd.concat(pieces, axis=1).reset_index()
    return result.rename(columns={"trading_date": "date"})


def _load_cached_intraday_feature_panel(
    context: AnalysisContext,
    specs: Dict[str, FeatureSpec],
    stocks: List[str],
) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame()

    panels: list[pd.DataFrame] = []
    missing: dict[str, FeatureSpec] = {}
    paths: dict[str, Path | None] = {}
    for name, spec in specs.items():
        path = _intraday_feature_cache_path(context, spec, stocks)
        paths[name] = path
        if path is None or not path.exists():
            missing[name] = spec
            continue
        cached = pd.read_parquet(path)
        if {"date", "code", name} <= set(cached.columns):
            panels.append(cached[["date", "code", name]])
        else:
            missing[name] = spec

    if missing:
        fields = sorted(
            {
                field
                for spec in missing.values()
                for field in (spec.field, *tuple(spec.dependencies))
            }
        )
        chunk_size = int(
            context.runtime_config.get("five_minute_stock_chunk_size", 0) or 0
        )
        # Preserve each stock's complete history/PIT endpoint. Only the daily
        # summaries survive between batches; publish full-universe caches only
        # after every batch has succeeded. Nonpositive sizes keep legacy loading.
        query_stocks = list(dict.fromkeys(stocks))
        batch_size = chunk_size if chunk_size > 0 else max(len(query_stocks), 1)
        daily_parts: list[pd.DataFrame] = []
        for offset in range(0, len(query_stocks), batch_size):
            raw = load_intraday_5m_panel(
                context, fields, query_stocks[offset : offset + batch_size]
            )
            daily = aggregate_5m_to_daily_features(raw, missing)
            del raw
            if not daily.empty:
                daily_parts.append(daily)
            del daily
        realized = (
            pd.concat(daily_parts, ignore_index=True)
            .sort_values(["date", "code"])
            .reset_index(drop=True)
            if daily_parts
            else pd.DataFrame()
        )
        for name in missing:
            if name not in realized:
                continue
            panel = realized[["date", "code", name]]
            panels.append(panel)
            path = paths[name]
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
                panel.to_parquet(temporary, index=False)
                temporary.replace(path)

    if not panels:
        return pd.DataFrame()
    return _merge_panels(panels)


def calculate_forward_return(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if "daily_close" not in panel.columns:
        raise ValueError("daily_close is required to calculate forward returns")
    work = panel.sort_values(["code", "date"]).copy()
    work["future_close"] = work.groupby("code")["daily_close"].shift(-int(horizon))
    work["label"] = work["future_close"] / work["daily_close"] - 1.0
    return work.drop(columns=["future_close"])


def _label_targets(config: dict) -> list[dict]:
    targets = config.get("label", {}).get("targets", [])
    if targets:
        return [
            {
                "name": str(
                    target.get("name")
                    or f"ret_{int(target.get('start', 1))}_{int(target.get('end', 1))}"
                ),
                "start": int(target.get("start", 1)),
                "end": int(target.get("end", target.get("start", 1))),
            }
            for target in targets
        ]
    horizon = int(config.get("label", {}).get("target_period", 5))
    return [{"name": f"ret_1_{horizon}", "start": 1, "end": horizon}]


def max_label_horizon(config: dict) -> int:
    return max(target["end"] for target in _label_targets(config))


def calculate_forward_returns(
    panel: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, list[str]]:
    if "daily_close" not in panel.columns:
        raise ValueError("daily_close is required to calculate forward returns")

    work = panel.sort_values(["code", "date"]).copy()
    label_columns: list[str] = []
    grouped_close = work.groupby("code")["daily_close"]
    for target in _label_targets(config):
        start = max(1, int(target["start"]))
        end = max(start, int(target["end"]))
        base = work["daily_close"] if start <= 1 else grouped_close.shift(-(start - 1))
        future = grouped_close.shift(-end)
        column = f"label__{target['name']}"
        work[column] = future / base - 1.0
        label_columns.append(column)

    if len(label_columns) == 1:
        work["label"] = work[label_columns[0]]
    else:
        work["label"] = work[label_columns].mean(axis=1, skipna=True)
    return work, label_columns


def _merge_panels(panels: List[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [panel for panel in panels if panel is not None and not panel.empty]
    if not non_empty:
        raise ValueError("No feature panels were loaded")
    merged = non_empty[0]
    for panel in non_empty[1:]:
        merged = merged.merge(panel, on=["date", "code"], how="outer")
    return merged


def build_training_panel(
    config: dict,
    context: AnalysisContext,
    *,
    anchor_factor_names: List[str] | None = None,
    anchor_db_path: str | None = None,
    drop_missing_labels: bool = True,
) -> pd.DataFrame:
    feature_config = config.get("features", {})
    feature_space = build_feature_space(config)
    stocks = resolve_universe(context)
    anchors = list(anchor_factor_names or [])
    expected_anchor_columns = {f"anchor__{name}" for name in anchors}
    cache_path = _panel_cache_path(
        config, context, stocks, anchors, drop_missing_labels=drop_missing_labels
    )
    if cache_path is not None and cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if (
            isinstance(cached, pd.DataFrame)
            and not cached.empty
            and expected_anchor_columns <= set(cached.columns)
        ):
            return cached

    daily_fields = sorted(set(feature_config.get("daily_price_fields", [])) | {"close"})
    daily = load_daily_price_panel(context, daily_fields, stocks)
    intraday_specs = {
        name: spec
        for name, spec in feature_space.features.items()
        if spec.source == "price_5m"
    }
    factor_fields = sorted(
        {
            spec.field
            for spec in feature_space.features.values()
            if spec.source == "factor_data"
        }
    )
    factor_data = load_factor_data_panel(context, factor_fields, stocks)
    intraday = _load_cached_intraday_feature_panel(context, intraday_specs, stocks)

    panel = _merge_panels([daily, factor_data, intraday])
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["date", "code"])
    panel, label_columns = calculate_forward_returns(panel, config)
    pool_snapshot = _pool_snapshot(context)
    if pool_snapshot:
        panel = (
            apply_date_level_pool_filter_to_long_df(
                panel.rename(columns={"code": "stock_code"}).set_index(
                    ["date", "stock_code"]
                ),
                pool_snapshot,
            )
            .reset_index()
            .rename(columns={"stock_code": "code"})
        )

    feature_names = list(feature_space.features)
    available = [name for name in feature_names if name in panel.columns]
    if not available:
        raise ValueError(
            "No configured symbolic features were available in the loaded panel"
        )

    anchor_columns: List[str] = []
    if anchors and anchor_db_path:
        anchor_panel = load_anchor_factor_panel(
            anchor_db_path,
            anchors,
            stocks,
            str(context.runtime_config["start_date"]),
            str(context.runtime_config["end_date"]),
            cache_dir=context.runtime_config.get("cache_dir"),
        )
        if not anchor_panel.empty:
            anchor_panel["date"] = pd.to_datetime(
                anchor_panel["date"], errors="coerce"
            ).dt.normalize()
            anchor_panel = anchor_panel.dropna(subset=["date", "code"])
            panel = panel.merge(anchor_panel, on=["date", "code"], how="left")
            anchor_columns = [
                column for column in panel.columns if str(column).startswith("anchor__")
            ]

    missing_anchor_columns = sorted(expected_anchor_columns - set(anchor_columns))
    if missing_anchor_columns:
        raise ValueError(
            "missing requested anchor columns: " + ", ".join(missing_anchor_columns)
        )

    panel = panel[
        ["date", "code", "label"] + label_columns + available + anchor_columns
    ].replace([np.inf, -np.inf], np.nan)
    if drop_missing_labels:
        panel = panel.dropna(subset=["label"])
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_pickle(cache_path)
    return panel


def split_train_valid_with_metadata(
    panel: pd.DataFrame,
    train_ratio: float,
    embargo_days: int = 0,
    train_end_date: str | None = None,
    valid_start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    date_values = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    unique_dates = sorted(date_values.dropna().unique())
    unique_dates = [pd.Timestamp(date).normalize() for date in unique_dates]
    if len(unique_dates) < 2:
        raise ValueError("Need at least two dates for train/validation split")

    explicit_train_end = (
        pd.Timestamp(train_end_date).normalize() if train_end_date else None
    )
    explicit_valid_start = (
        pd.Timestamp(valid_start_date).normalize() if valid_start_date else None
    )

    if explicit_train_end is not None:
        requested_train_end = explicit_train_end
    else:
        split_idx = max(
            1, min(len(unique_dates) - 1, int(len(unique_dates) * train_ratio))
        )
        requested_train_end = unique_dates[split_idx - 1]

    train_end_auto_adjusted = False
    dropped_train_dates = 0
    if explicit_valid_start is not None:
        valid_start = explicit_valid_start
        valid_start_index = sum(date < valid_start for date in unique_dates)
        if valid_start_index >= len(unique_dates):
            raise ValueError(
                "Explicit valid_start_date produced an empty validation partition"
            )
        max_train_index = valid_start_index - max(0, int(embargo_days)) - 1
        if max_train_index < 0:
            raise ValueError("Not enough pre-validation dates to satisfy embargo_days")
        max_allowed_train_end = unique_dates[max_train_index]
        effective_train_end = min(requested_train_end, max_allowed_train_end)
        if effective_train_end < requested_train_end:
            train_end_auto_adjusted = True
            dropped_train_dates = sum(
                max_allowed_train_end < date <= requested_train_end
                for date in unique_dates
            )
    else:
        effective_train_end = requested_train_end
        split_idx = max(1, sum(date <= effective_train_end for date in unique_dates))
        embargo_idx = min(len(unique_dates) - 1, split_idx + max(0, int(embargo_days)))
        valid_start = unique_dates[embargo_idx]

    train = panel[date_values <= effective_train_end].copy()
    if train.empty:
        raise ValueError("Train/validation split produced an empty training partition")
    split_date = max(
        pd.to_datetime(train["date"], errors="coerce").dt.normalize().dropna().unique()
    )
    split_date = pd.Timestamp(split_date).normalize()
    split_idx = max(1, sum(date <= split_date for date in unique_dates))

    valid = panel[date_values >= valid_start].copy()
    if train.empty or valid.empty:
        raise ValueError("Train/validation split produced an empty partition")

    split_index = max(0, sum(date <= split_date for date in unique_dates) - 1)
    valid_start_index = sum(
        date < pd.Timestamp(valid_start).normalize() for date in unique_dates
    )
    metadata = {
        "split_date": pd.Timestamp(split_date).strftime("%Y-%m-%d"),
        "valid_start": pd.Timestamp(valid_start).strftime("%Y-%m-%d"),
        "train_ratio": float(train_ratio),
        "embargo_days": int(embargo_days),
        "required_embargo_days": int(embargo_days),
        "split_index": int(split_index),
        "valid_start_index": int(valid_start_index),
        "n_dates": int(len(unique_dates)),
        "requested_train_end_date": pd.Timestamp(requested_train_end).strftime(
            "%Y-%m-%d"
        ),
        "effective_train_end_date": pd.Timestamp(effective_train_end).strftime(
            "%Y-%m-%d"
        ),
        "train_end_auto_adjusted": bool(train_end_auto_adjusted),
        "dropped_train_dates": int(dropped_train_dates),
        "train_end_date_explicit": pd.Timestamp(explicit_train_end).strftime("%Y-%m-%d")
        if explicit_train_end is not None
        else "",
        "valid_start_date_explicit": pd.Timestamp(explicit_valid_start).strftime(
            "%Y-%m-%d"
        )
        if explicit_valid_start is not None
        else "",
    }
    return train, valid, metadata


def split_train_valid(
    panel: pd.DataFrame,
    train_ratio: float,
    embargo_days: int = 0,
    train_end_date: str | None = None,
    valid_start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, valid, _ = split_train_valid_with_metadata(
        panel,
        train_ratio,
        embargo_days=embargo_days,
        train_end_date=train_end_date,
        valid_start_date=valid_start_date,
    )
    return train, valid


def split_train_valid_holdout_with_metadata(
    panel: pd.DataFrame,
    train_ratio: float,
    embargo_days: int = 0,
    train_end_date: str | None = None,
    valid_start_date: str | None = None,
    valid_end_date: str | None = None,
    holdout_start_date: str | None = None,
    holdout_embargo_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Three-way ``train`` / ``valid`` / ``holdout`` split.

    ``valid`` is bounded to ``[valid_start, valid_end]`` and the ``holdout``
    partition (``date >= holdout_start``) is carved out so it never enters the
    search. When neither ``valid_end_date`` nor ``holdout_start_date`` is
    provided, the holdout partition is empty and ``valid`` matches the legacy
    two-way :func:`split_train_valid_with_metadata` behaviour.

    ``holdout_embargo_days`` mirrors the train->valid ``embargo_days`` semantics
    at the valid->holdout boundary: that many trading dates are kept strictly
    between the last valid date and ``holdout_start`` by trimming the valid
    TAIL (the holdout window stays as configured). Without it, the forward
    return labels of the last valid dates (up to the max label horizon) overlap
    the first holdout returns, so selection on valid would peek at the gate
    window.
    """
    train, valid_full, metadata = split_train_valid_with_metadata(
        panel,
        train_ratio,
        embargo_days=embargo_days,
        train_end_date=train_end_date,
        valid_start_date=valid_start_date,
    )
    valid_dates = pd.to_datetime(valid_full["date"], errors="coerce").dt.normalize()
    explicit_valid_end = (
        pd.Timestamp(valid_end_date).normalize() if valid_end_date else None
    )
    explicit_holdout_start = (
        pd.Timestamp(holdout_start_date).normalize() if holdout_start_date else None
    )

    if explicit_valid_end is not None:
        valid = valid_full[valid_dates <= explicit_valid_end].copy()
    elif explicit_holdout_start is not None:
        valid = valid_full[valid_dates < explicit_holdout_start].copy()
    else:
        valid = valid_full.copy()
    if valid.empty:
        raise ValueError(
            "valid_end_date/holdout_start_date produced an empty validation partition"
        )

    holdout_embargo_dropped_valid_dates = 0
    holdout_embargo_valid_end = ""
    if explicit_holdout_start is not None and int(holdout_embargo_days) > 0:
        all_dates = sorted(
            pd.to_datetime(panel["date"], errors="coerce")
            .dt.normalize()
            .dropna()
            .unique()
        )
        holdout_start_index = sum(date < explicit_holdout_start for date in all_dates)
        max_valid_index = holdout_start_index - int(holdout_embargo_days) - 1
        if max_valid_index < 0:
            raise ValueError(
                "Not enough pre-holdout dates to satisfy holdout_embargo_days"
            )
        max_valid_end = pd.Timestamp(all_dates[max_valid_index]).normalize()
        holdout_embargo_valid_end = max_valid_end.strftime("%Y-%m-%d")
        valid_only_dates = pd.to_datetime(valid["date"], errors="coerce").dt.normalize()
        holdout_embargo_dropped_valid_dates = int(
            valid_only_dates[valid_only_dates > max_valid_end].nunique()
        )
        if holdout_embargo_dropped_valid_dates > 0:
            valid = valid[valid_only_dates <= max_valid_end].copy()
        if valid.empty:
            raise ValueError(
                "holdout_embargo_days trimmed the validation partition to empty"
            )

    if explicit_holdout_start is not None:
        holdout = valid_full[valid_dates >= explicit_holdout_start].copy()
    else:
        holdout = valid_full.iloc[0:0].copy()

    holdout_dates = (
        pd.to_datetime(holdout["date"], errors="coerce").dt.normalize()
        if not holdout.empty
        else None
    )
    valid_only_dates = pd.to_datetime(valid["date"], errors="coerce").dt.normalize()
    metadata = {
        **metadata,
        "valid_end": explicit_valid_end.strftime("%Y-%m-%d")
        if explicit_valid_end is not None
        else "",
        "holdout_start": explicit_holdout_start.strftime("%Y-%m-%d")
        if explicit_holdout_start is not None
        else "",
        "valid_end_date_explicit": explicit_valid_end.strftime("%Y-%m-%d")
        if explicit_valid_end is not None
        else "",
        "holdout_start_date_explicit": explicit_holdout_start.strftime("%Y-%m-%d")
        if explicit_holdout_start is not None
        else "",
        "valid_rows": int(len(valid)),
        "valid_dates": int(valid_only_dates.nunique()),
        "holdout_rows": int(len(holdout)),
        "holdout_dates": int(holdout_dates.nunique())
        if holdout_dates is not None
        else 0,
        "holdout_embargo_days": int(holdout_embargo_days),
        "holdout_embargo_valid_end": holdout_embargo_valid_end,
        "holdout_embargo_dropped_valid_dates": int(holdout_embargo_dropped_valid_dates),
    }
    return train, valid, holdout, metadata


def select_sampled_dates(dates: Iterable, sampling: str) -> list[pd.Timestamp]:
    """Reduce a date axis to a coarser rebalance cadence.

    ``sampling`` accepts ``none`` (keep every date), ``monthly`` / ``weekly``
    (the last available trading day of each calendar month / ISO week), or
    ``every_n_days:N`` (every Nth distinct sorted date). Unknown values fall back
    to ``none``.

    Sampling only thins the training/search cross-sections to curb overfitting
    to daily noise; the holdout/evaluation panel is never sampled. Forward-return
    labels are precomputed per row from the full daily series before any
    sampling, so retaining month-end rows preserves the true forward horizon and
    yields near non-overlapping 10-day returns.
    """
    normalized = sorted(
        pd.to_datetime(pd.Series(list(dates)), errors="coerce")
        .dropna()
        .dt.normalize()
        .unique()
    )
    if not normalized:
        return []
    normalized = [pd.Timestamp(value) for value in normalized]
    mode = str(sampling or "none").strip().lower()
    if mode in {"", "none", "daily", "all"}:
        return normalized
    if mode.startswith("every_n_days:"):
        try:
            step = max(1, int(mode.split(":", 1)[1]))
        except ValueError:
            return normalized
        return normalized[::step]
    if mode == "monthly":
        keys = [(ts.year, ts.month) for ts in normalized]
    elif mode == "weekly":
        iso = [ts.isocalendar() for ts in normalized]
        keys = [(item[0], item[1]) for item in iso]
    else:
        return normalized
    # ``normalized`` is ascending, so the last write per bucket is its last day.
    last_by_key: Dict[tuple, pd.Timestamp] = {}
    for key, ts in zip(keys, normalized):
        last_by_key[key] = ts
    return sorted(last_by_key.values())


def downsample_panel_dates(
    panel: pd.DataFrame, sampling: str
) -> tuple[pd.DataFrame, dict]:
    """Filter ``panel`` rows down to the sampled rebalance dates.

    Returns the (possibly unchanged) panel plus telemetry describing the cadence
    and the date/row reduction. A no-op for ``none``/missing ``date`` column.
    """
    mode = str(sampling or "none").strip().lower()
    has_date = "date" in panel.columns
    base_dates = int(panel["date"].nunique()) if has_date else 0
    if not has_date or panel.empty or mode in {"", "none", "daily", "all"}:
        return panel, {
            "sampling": mode or "none",
            "applied": False,
            "dates_before": base_dates,
            "dates_after": base_dates,
            "rows_before": int(len(panel)),
            "rows_after": int(len(panel)),
        }
    dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    keep = set(select_sampled_dates(dates.dropna().unique(), mode))
    filtered = panel[dates.isin(keep)].copy()
    return filtered, {
        "sampling": mode,
        "applied": True,
        "dates_before": base_dates,
        "dates_after": int(filtered["date"].nunique()) if not filtered.empty else 0,
        "rows_before": int(len(panel)),
        "rows_after": int(len(filtered)),
    }
