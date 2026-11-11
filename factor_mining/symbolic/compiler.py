"""Compile symbolic expressions into standalone FactorBase factor files."""

from __future__ import annotations

import re
from pathlib import Path

from .expression import referenced_features, to_factor_code
from .feature_space import FeatureSpace
from .operator_runtime import (
    INTRADAY_AGGREGATION_SOURCE,
    runtime_namespace_source_for_operators,
    runtime_source_for_operators,
)
from .search_base import SearchCandidate


def _class_name_from_path(path: Path) -> str:
    stem = re.sub(r"[^0-9a-zA-Z_]+", "_", path.stem)
    parts = [part for part in stem.split("_") if part]
    if not parts:
        return "SymbolicMinedFactor"
    name = "".join(part[:1].upper() + part[1:] for part in parts)
    if name[0].isdigit():
        name = "Symbolic" + name
    return name


def _expression_uses_operator(node, names: set[str]) -> bool:
    if node.kind == "operator" and str(node.value) in names:
        return True
    return any(_expression_uses_operator(child, names) for child in node.children)


_LAG_OPERATORS = {"delay", "delta", "pct_change"}
_ROLLING_OPERATORS = {
    "ts_mean",
    "ts_median",
    "ts_sum",
    "ts_std",
    "ts_mad",
    "ts_min",
    "ts_max",
    "ts_argmin",
    "ts_argmax",
    "ts_rank",
    "ts_zscore",
    "ts_iqr",
    "ts_cv",
    "ts_skew",
    "ts_kurt",
    "ts_autocorr",
    "ts_slope",
    "decay_linear",
    "ema",
    "ts_corr",
    "ts_cov",
    "ts_beta",
    "ts_residual",
}


def _used_operators(node) -> set[str]:
    operators = {str(node.value)} if node.kind == "operator" else set()
    for child in node.children:
        operators.update(_used_operators(child))
    return operators


def _required_history_periods(node, feature_space: FeatureSpace | None = None) -> int:
    if node.kind == "feature":
        return 1
    if node.kind in {"constant", "window"}:
        return 0

    child_history = max(
        (_required_history_periods(child, feature_space) for child in node.children),
        default=1,
    )
    windows = [
        max(1, int(float(child.value)))
        for child in node.children
        if child.kind == "window"
    ]
    if not windows:
        return child_history
    window = max(windows)
    operator = str(node.value)
    spec = feature_space.operators.get(operator) if feature_space is not None else None
    lookback = spec.lookback if spec is not None else ""
    if lookback == "lag" or operator in _LAG_OPERATORS:
        return child_history + window
    if lookback == "rolling" or operator in _ROLLING_OPERATORS:
        return child_history + window - 1
    return max(child_history, 1)


def _namespace_expression_code(node) -> str:
    if node.kind == "feature":
        return f"features[{str(node.value)!r}]"
    if node.kind == "constant":
        return repr(float(node.value))
    if node.kind == "window":
        return repr(int(node.value))
    args = ", ".join(_namespace_expression_code(child) for child in node.children)
    return f"namespace[{str(node.value)!r}]({args})"


def _used_specs(candidate: SearchCandidate, feature_space: FeatureSpace):
    refs = referenced_features(candidate.expression)
    if _expression_uses_operator(
        candidate.expression, {"neutralize_size", "neutralize_styles"}
    ):
        for style_name in (
            "fd_val_market_cap",
            "fd_natural_log_of_market_cap",
            "fd_cube_of_size",
        ):
            if style_name in feature_space.features:
                refs.add(style_name)
    return {name: feature_space.features[name] for name in refs}


def compile_factor_file(
    candidate: SearchCandidate,
    config: dict,
    feature_space: FeatureSpace,
    output_path: str | Path,
    *,
    factor_value_cache_dir: str | Path | None = None,
    date_level_pool_filter: bool = False,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_factor_class(
            candidate,
            config,
            feature_space,
            path,
            factor_value_cache_dir=factor_value_cache_dir,
            date_level_pool_filter=date_level_pool_filter,
        ),
        encoding="utf-8",
    )
    return path


def render_factor_class(
    candidate: SearchCandidate,
    config: dict,
    feature_space: FeatureSpace,
    output_path: Path,
    *,
    factor_value_cache_dir: str | Path | None = None,
    date_level_pool_filter: bool = False,
) -> str:
    specs = _used_specs(candidate, feature_space)
    class_name = _class_name_from_path(output_path)
    expression_code = to_factor_code(candidate.expression)
    runtime_expression_code = _namespace_expression_code(candidate.expression)
    expression_text = candidate.expression.to_string()
    used_operators = _used_operators(candidate.expression)
    runtime_source = runtime_source_for_operators(used_operators)
    runtime_namespace_source = runtime_namespace_source_for_operators(used_operators)
    uses_intraday = any(spec.source == "price_5m" for spec in specs.values())
    data_frequency = "5m" if uses_intraday else "daily"
    required_history_window = _required_history_periods(
        candidate.expression, feature_space
    ) * (48 if uses_intraday else 1)
    price_fields = sorted(
        {
            field
            for spec in specs.values()
            if spec.source in {"price_daily", "price_5m"}
            for field in (spec.field, *tuple(spec.dependencies))
        }
    )
    if price_fields and "factor" not in price_fields:
        price_fields.append("factor")
    factor_data_fields = sorted(
        {spec.field for spec in specs.values() if spec.source == "factor_data"}
    )

    daily_features = sorted(
        name for name, spec in specs.items() if spec.source == "price_daily"
    )
    intraday_specs = {
        name: {
            "name": name,
            "field": spec.field,
            "aggregator": spec.aggregator,
            "dependencies": list(spec.dependencies),
        }
        for name, spec in specs.items()
        if spec.source == "price_5m"
    }
    factor_features = sorted(
        name for name, spec in specs.items() if spec.source == "factor_data"
    )
    intraday_prepare_source = (
        """        work["trading_date"] = work["date"].dt.normalize()
        grouped = work.groupby(["trading_date", "code"], sort=True)
        pieces = []

        base_price_fields = sorted(
            {name.replace("daily_", "", 1) for name in DAILY_FEATURES}
            | {
                field
                for spec in INTRADAY_SPECS.values()
                for field in [spec["field"], *spec.get("dependencies", [])]
            }
        )
        for field in base_price_fields:
            if field not in work.columns:
                continue
            agg = _daily_agg_for_field(field)
            pieces.append(grouped[field].agg(agg).rename(f"daily_{field}"))

        for feature_name, spec in INTRADAY_SPECS.items():
            series = _aggregate_intraday_feature_series(work, grouped, spec)
            if series is not None:
                pieces.append(series.rename(feature_name))

        for feature_name in FACTOR_FEATURES:
            if feature_name in work.columns:
                pieces.append(grouped[feature_name].last().rename(feature_name))

        if not pieces:
            return pd.DataFrame()
        frame = pd.concat(pieces, axis=1).reset_index().rename(columns={"trading_date": "date"})"""
        if uses_intraday
        else '        raise ValueError("Daily symbolic factor cannot use intraday preparation")'
    )

    # Keep legacy generated bytes unchanged unless the caller opts in.
    pool_import_source = (
        "from utils.dynamic_stock_pool import apply_date_level_pool_filter_to_long_df\n"
        if date_level_pool_filter else ""
    )
    pool_evaluation_source = (
        """        return self._calculate_from_prepared_features(feature_frame)

    def get_factor_values_unfiltered_chunked(self, chunk_size):
        # Reduce stock chunks first; CS/rolling math sees all prepared rows once.
        return self._get_factor_values_from_prepared_stock_chunks(
            chunk_size,
            lambda data: _prepare_feature_frame(data, uses_intraday="""
        + repr(uses_intraday)
        + """),
            self._calculate_from_prepared_features,
            mode="date_level_pool_feature_panel",
        )

    def _calculate_from_prepared_features(self, feature_frame):
        if feature_frame.empty:
            return pd.DataFrame()
        pools = getattr(self, "monthly_stock_pools", None)
        if not pools:
            raise ValueError("Date-level pool symbolic factor requires monthly_stock_pools")
        # unfiltered execution temporarily disables use_dynamic_pool, but retains
        # the already verified snapshot mapping supplied by run_analysis_main.
        feature_frame = feature_frame.sort_values(["date", "code"], kind="mergesort")
        feature_frame = apply_date_level_pool_filter_to_long_df(
            feature_frame.rename(columns={"code": "stock_code"}).set_index(["date", "stock_code"]),
            pools,
        ).reset_index().rename(columns={"stock_code": "code"}).reset_index(drop=True)
        if feature_frame.empty:
            return pd.DataFrame()

"""
        if date_level_pool_filter else ""
    )

    return f'''"""
Generated symbolic factor.

Expression: {expression_text}
Validation score: {candidate.score:.8f}
Complexity: {candidate.complexity}

This file is self-contained and intentionally does not depend on the symbolic
mining runtime or generated artifacts.
"""

import numpy as np
import pandas as pd

from utils.factor_base import FactorBase
{pool_import_source}

EXPRESSION_CODE = {expression_code!r}
EXPRESSION_TEXT = {expression_text!r}
PRICE_FIELDS = {price_fields!r}
FACTOR_DATA_FIELDS = {factor_data_fields!r}
DAILY_FEATURES = {daily_features!r}
INTRADAY_SPECS = {intraday_specs!r}
FACTOR_FEATURES = {factor_features!r}
DATA_FREQUENCY = {data_frequency!r}
REQUIRED_HISTORY_WINDOW = {required_history_window!r}
FACTOR_VALUE_CACHE_DIR = {str(Path(factor_value_cache_dir).resolve()) if factor_value_cache_dir else None!r}
SYMBOLIC_FACTOR_METADATA = {{
    "price_fields": PRICE_FIELDS,
    "factor_data_fields": FACTOR_DATA_FIELDS,
    "daily_features": DAILY_FEATURES,
    "intraday_specs": INTRADAY_SPECS,
    "factor_features": FACTOR_FEATURES,
    "data_frequency": DATA_FREQUENCY,
    "required_history_window": REQUIRED_HISTORY_WINDOW,
}}


{runtime_source}

{INTRADAY_AGGREGATION_SOURCE if uses_intraday else ""}

{runtime_namespace_source}


def _normalize_date_column(df):
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], format="mixed", errors="coerce")
    result = result.dropna(subset=["date", "code"])
    return result


def _daily_agg_for_field(field):
    if field == "open":
        return "first"
    if field == "high":
        return "max"
    if field == "low":
        return "min"
    if field in {{"close", "factor"}}:
        return "last"
    if field in {{"volume", "money"}}:
        return "sum"
    return "last"


def _prepare_feature_frame(data, uses_intraday):
    if data.empty:
        return pd.DataFrame()

    work = _normalize_date_column(data)
    if work.empty:
        return pd.DataFrame()

    for raw_name in FACTOR_DATA_FIELDS:
        feature_name = f"fd_{{raw_name}}"
        if raw_name in work.columns and feature_name not in work.columns:
            work[feature_name] = work[raw_name]

    if uses_intraday:
{intraday_prepare_source}
    else:
        frame = (
            work.sort_values(["date", "code"], kind="mergesort")
            .drop_duplicates(subset=["date", "code"], keep="last")
            .copy()
        )
        frame["date"] = frame["date"].dt.normalize()
        for feature_name in DAILY_FEATURES:
            raw_name = feature_name.replace("daily_", "", 1)
            if raw_name in frame.columns and feature_name not in frame.columns:
                frame[feature_name] = frame[raw_name]

    return frame.replace([np.inf, -np.inf], np.nan)


class {class_name}(FactorBase):
    """Symbolic mined factor generated for run_analysis_main.py."""

    EXPRESSION_CODE = EXPRESSION_CODE
    EXPRESSION_TEXT = EXPRESSION_TEXT
    PRICE_FIELDS = PRICE_FIELDS
    FACTOR_DATA_FIELDS = FACTOR_DATA_FIELDS
    DAILY_FEATURES = DAILY_FEATURES
    INTRADAY_SPECS = INTRADAY_SPECS
    FACTOR_FEATURES = FACTOR_FEATURES
    DATA_FREQUENCY = DATA_FREQUENCY
    REQUIRED_HISTORY_WINDOW = REQUIRED_HISTORY_WINDOW
    FACTOR_VALUE_CACHE_DIR = FACTOR_VALUE_CACHE_DIR
    SYMBOLIC_FACTOR_METADATA = SYMBOLIC_FACTOR_METADATA

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("data_frequency", DATA_FREQUENCY)
        super().__init__(*args, **kwargs)

    def _get_raw_data_for_factor(self):
        frames = []
        if PRICE_FIELDS:
            price_df = self._fetch_raw_data(fields=PRICE_FIELDS, table_name="price_data")
            if not price_df.empty:
                frames.append(price_df)

        if FACTOR_DATA_FIELDS:
            factor_df = self._fetch_raw_data(fields=FACTOR_DATA_FIELDS, table_name="factor_data")
            if not factor_df.empty:
                factor_df = _normalize_date_column(factor_df)
                factor_df["date_key"] = factor_df["date"].dt.normalize()
                rename_map = {{field: f"fd_{{field}}" for field in FACTOR_DATA_FIELDS if field in factor_df.columns}}
                factor_df = factor_df.rename(columns=rename_map)
                keep_cols = ["date_key", "code"] + list(rename_map.values())
                factor_df = factor_df[keep_cols].drop_duplicates(subset=["date_key", "code"], keep="last")

                if frames:
                    base = frames[0].copy()
                    base = _normalize_date_column(base)
                    base["date_key"] = base["date"].dt.normalize()
                    frames[0] = base.merge(factor_df, on=["date_key", "code"], how="left").drop(columns=["date_key"])
                else:
                    frames.append(factor_df.rename(columns={{"date_key": "date"}}))

        if not frames:
            return pd.DataFrame()
        return frames[0]

    def calculate_factor(self, data):
        feature_frame = _prepare_feature_frame(data, uses_intraday={uses_intraday!r})
        if feature_frame.empty:
            return pd.DataFrame()

{pool_evaluation_source}        features = {{name: feature_frame[name] for name in {sorted(specs)!r} if name in feature_frame.columns}}
        missing = set({sorted(specs)!r}) - set(features)
        if missing:
            self.logger.warning("Symbolic factor missing features: %s", sorted(missing))
            return pd.DataFrame()

        namespace = runtime_namespace(features, feature_frame["date"], feature_frame["code"])
        try:
            values = {runtime_expression_code}
        except Exception as exc:
            self.logger.error("Symbolic expression evaluation failed: %s", exc)
            return pd.DataFrame()

        if np.isscalar(values):
            values = pd.Series(float(values), index=feature_frame.index)
        else:
            values = pd.Series(values, index=feature_frame.index)
        values = values.replace([np.inf, -np.inf], np.nan)

        result = pd.DataFrame({{
            "date": feature_frame["date"],
            "code": feature_frame["code"],
            "factor": values,
        }}).dropna(subset=["date", "code"])
        if result.empty:
            return pd.DataFrame()
        factor_df = result.pivot(index="date", columns="code", values="factor").sort_index()
        factor_df = factor_df[(factor_df.index >= pd.to_datetime(self.start_date)) & (factor_df.index <= pd.to_datetime(self.end_date))]
        return factor_df
'''
