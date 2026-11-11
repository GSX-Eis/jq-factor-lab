"""Feature and operator definitions for symbolic mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .expression import ExpressionNode, referenced_features


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    field: str
    frequency: str
    aggregator: str | None = None
    dependencies: Sequence[str] = ()


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    complexity: int = 1
    exportable: bool = True
    category: str = "basic"
    arg_types: Sequence[str] = ()
    max_nested: int | None = None
    lookback: str = "none"
    nan_behavior: str = "propagate"

    def __post_init__(self):
        if not self.arg_types:
            object.__setattr__(
                self, "arg_types", tuple("expr" for _ in range(self.arity))
            )
        else:
            object.__setattr__(self, "arg_types", tuple(self.arg_types))
        if self.lookback == "rolling":
            object.__setattr__(self, "nan_behavior", "full_window")
        elif self.category == "cross_section":
            object.__setattr__(self, "nan_behavior", "ignore_within_group")
        elif self.category == "logical":
            object.__setattr__(self, "nan_behavior", "numpy_boolean")


@dataclass
class FeatureSpace:
    features: Dict[str, FeatureSpec]
    operators: Dict[str, OperatorSpec]
    windows: tuple[int, ...] = ()


DEFAULT_INTRADAY_AGGREGATORS = ("mean", "std", "last")
SUPPORTED_INTRADAY_AGGREGATORS = (
    "first",
    "last",
    "max",
    "min",
    "median",
    "mean",
    "std",
    "sum",
    "vwap",
    "realized_vol",
    "range",
    "high_low_range",
    "close_vwap_gap",
    "close_open_gap",
    "volume_imbalance",
    "money_imbalance",
    "volume_concentration",
    "money_concentration",
    "morning_mean",
    "afternoon_mean",
    "last_n_mean",
)


def _full_dsl_enabled(config: dict) -> bool:
    search_config = config.get("search", {})
    return bool(search_config.get("enable_full_dsl", False)) or str(
        search_config.get("dsl", "")
    ).lower() in {
        "full",
        "full_alpha",
        "full_dsl",
    }


def build_intraday_aggregator_specs(
    fields: List[str], aggregators: Sequence[str] | None = None
) -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []
    selected = tuple(aggregators or DEFAULT_INTRADAY_AGGREGATORS)
    price_like_fields = {"open", "high", "low", "close", "factor"}
    for field in fields:
        for agg in selected:
            if agg not in SUPPORTED_INTRADAY_AGGREGATORS:
                continue
            if agg == "vwap" and field not in price_like_fields:
                continue
            if (
                agg in {"realized_vol", "high_low_range", "close_vwap_gap"}
                and field != "close"
            ):
                continue
            if agg == "close_open_gap" and field != "close":
                continue
            if (
                agg in {"volume_imbalance", "volume_concentration"}
                and field != "volume"
            ):
                continue
            if agg in {"money_imbalance", "money_concentration"} and field != "money":
                continue
            dependencies: tuple[str, ...] = ()
            if agg in {"vwap", "close_vwap_gap"}:
                dependencies = ("money", "volume")
            elif agg == "high_low_range":
                dependencies = ("high", "low", "close")
            elif agg == "close_open_gap":
                dependencies = ("open", "close")
            specs.append(
                FeatureSpec(
                    name=f"intraday_{agg}_{field}",
                    source="price_5m",
                    field=field,
                    frequency="5m",
                    aggregator=agg,
                    dependencies=dependencies,
                )
            )
    return specs


def build_operator_registry(config: dict) -> Dict[str, OperatorSpec]:
    constants_enabled = bool(
        config.get("search", {}).get("constants", {}).get("enabled", False)
    )
    include_scalar_ops = constants_enabled

    operators = {
        "add": OperatorSpec("add", 2, 1, category="basic"),
        "sub": OperatorSpec("sub", 2, 1, category="basic"),
        "mul": OperatorSpec("mul", 2, 2, category="basic"),
        "div": OperatorSpec("div", 2, 3, category="basic"),
        "neg": OperatorSpec("neg", 1, 1, category="basic"),
        "abs": OperatorSpec("abs", 1, 1, category="mathematical"),
        "log": OperatorSpec("log", 1, 2, category="mathematical"),
        "sqrt": OperatorSpec("sqrt", 1, 2, category="mathematical"),
        "cs_rank": OperatorSpec("cs_rank", 1, 2, category="cross_section"),
        "cs_zscore": OperatorSpec("cs_zscore", 1, 2, category="cross_section"),
    }
    if not _full_dsl_enabled(config):
        return operators

    operators.update(
        {
            "sign": OperatorSpec("sign", 1, 1, category="mathematical"),
            "exp": OperatorSpec("exp", 1, 2, category="mathematical"),
            "floor": OperatorSpec("floor", 1, 1, category="mathematical"),
            "inv": OperatorSpec("inv", 1, 2, category="mathematical"),
            "element_max": OperatorSpec("element_max", 2, 1, category="mathematical"),
            "element_min": OperatorSpec("element_min", 2, 1, category="mathematical"),
            "cs_demean": OperatorSpec("cs_demean", 1, 2, category="cross_section"),
            "cs_scale": OperatorSpec("cs_scale", 1, 2, category="cross_section"),
            "cs_quantile": OperatorSpec("cs_quantile", 1, 2, category="cross_section"),
            "cs_mean": OperatorSpec("cs_mean", 1, 2, category="cross_section"),
            "cs_std": OperatorSpec("cs_std", 1, 2, category="cross_section"),
            "cs_skew": OperatorSpec("cs_skew", 1, 3, category="cross_section"),
            "cs_kurt": OperatorSpec("cs_kurt", 1, 3, category="cross_section"),
            "cs_max": OperatorSpec("cs_max", 1, 2, category="cross_section"),
            "cs_min": OperatorSpec("cs_min", 1, 2, category="cross_section"),
            "cs_median": OperatorSpec("cs_median", 1, 2, category="cross_section"),
            "delay": OperatorSpec(
                "delay",
                2,
                2,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="lag",
            ),
            "delta": OperatorSpec(
                "delta",
                2,
                2,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="lag",
            ),
            "pct_change": OperatorSpec(
                "pct_change",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="lag",
            ),
            "ts_pctchange": OperatorSpec(
                "ts_pctchange",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="lag",
            ),
            "ts_mean": OperatorSpec(
                "ts_mean",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_median": OperatorSpec(
                "ts_median",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_sum": OperatorSpec(
                "ts_sum",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_cumsum": OperatorSpec(
                "ts_cumsum",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_product": OperatorSpec(
                "ts_product",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_std": OperatorSpec(
                "ts_std",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_var": OperatorSpec(
                "ts_var",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_mad": OperatorSpec(
                "ts_mad",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_min": OperatorSpec(
                "ts_min",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_max": OperatorSpec(
                "ts_max",
                2,
                3,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_argmin": OperatorSpec(
                "ts_argmin",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_argmax": OperatorSpec(
                "ts_argmax",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "highday": OperatorSpec(
                "highday",
                2,
                4,
                category="auxiliary",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "lowday": OperatorSpec(
                "lowday",
                2,
                4,
                category="auxiliary",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_rank": OperatorSpec(
                "ts_rank",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_zscore": OperatorSpec(
                "ts_zscore",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_iqr": OperatorSpec(
                "ts_iqr",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_cv": OperatorSpec(
                "ts_cv",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_skew": OperatorSpec(
                "ts_skew",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_kurt": OperatorSpec(
                "ts_kurt",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_autocorr": OperatorSpec(
                "ts_autocorr",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_slope": OperatorSpec(
                "ts_slope",
                2,
                4,
                category="time_series",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "decay_linear": OperatorSpec(
                "decay_linear",
                2,
                4,
                category="technical",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "decaylinear": OperatorSpec(
                "decaylinear",
                2,
                4,
                category="technical",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ema": OperatorSpec(
                "ema",
                2,
                4,
                category="technical",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "wma": OperatorSpec(
                "wma",
                2,
                4,
                category="technical",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "ts_corr": OperatorSpec(
                "ts_corr",
                3,
                5,
                category="time_series",
                arg_types=("expr", "expr", "window"),
                lookback="rolling",
            ),
            "ts_cov": OperatorSpec(
                "ts_cov",
                3,
                5,
                category="time_series",
                arg_types=("expr", "expr", "window"),
                lookback="rolling",
            ),
            "ts_covariance": OperatorSpec(
                "ts_covariance",
                3,
                5,
                category="time_series",
                arg_types=("expr", "expr", "window"),
                lookback="rolling",
            ),
            "ts_beta": OperatorSpec(
                "ts_beta",
                3,
                5,
                category="technical",
                arg_types=("expr", "expr", "window"),
                lookback="rolling",
            ),
            "ts_residual": OperatorSpec(
                "ts_residual",
                3,
                5,
                category="technical",
                arg_types=("expr", "expr", "window"),
                lookback="rolling",
            ),
            "regbeta": OperatorSpec(
                "regbeta",
                3,
                5,
                category="technical",
                arg_types=("expr", "expr", "window"),
                lookback="rolling",
            ),
            "regresi": OperatorSpec(
                "regresi",
                3,
                5,
                category="technical",
                arg_types=("expr", "expr", "window"),
                lookback="rolling",
            ),
            "sequence": OperatorSpec(
                "sequence", 1, 2, category="auxiliary", arg_types=("window",)
            ),
            "rsi": OperatorSpec(
                "rsi",
                2,
                4,
                category="technical",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "bb_middle": OperatorSpec(
                "bb_middle",
                2,
                4,
                category="technical",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "bb_upper": OperatorSpec(
                "bb_upper",
                2,
                5,
                category="technical",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "bb_lower": OperatorSpec(
                "bb_lower",
                2,
                5,
                category="technical",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "macd": OperatorSpec(
                "macd",
                3,
                5,
                category="technical",
                arg_types=("expr", "window", "window"),
                lookback="rolling",
            ),
            "gt": OperatorSpec("gt", 2, 1, category="logical"),
            "ge": OperatorSpec("ge", 2, 1, category="logical"),
            "lt": OperatorSpec("lt", 2, 1, category="logical"),
            "le": OperatorSpec("le", 2, 1, category="logical"),
            "eq": OperatorSpec("eq", 2, 1, category="logical"),
            "ne": OperatorSpec("ne", 2, 1, category="logical"),
            "logical_and": OperatorSpec("logical_and", 2, 1, category="logical"),
            "logical_or": OperatorSpec("logical_or", 2, 1, category="logical"),
            "where": OperatorSpec("where", 3, 2, category="logical"),
            "filter": OperatorSpec(
                "filter",
                2,
                2,
                category="auxiliary",
                nan_behavior="condition_false_to_zero",
            ),
            "count": OperatorSpec(
                "count",
                2,
                3,
                category="auxiliary",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "sum_if": OperatorSpec(
                "sum_if",
                3,
                4,
                category="auxiliary",
                arg_types=("expr", "window", "expr"),
                lookback="rolling",
            ),
            "sumif": OperatorSpec(
                "sumif",
                3,
                4,
                category="auxiliary",
                arg_types=("expr", "window", "expr"),
                lookback="rolling",
            ),
            "prod": OperatorSpec(
                "prod",
                2,
                4,
                category="auxiliary",
                arg_types=("expr", "window"),
                lookback="rolling",
            ),
            "neutralize_size": OperatorSpec(
                "neutralize_size", 1, 4, category="neutralization"
            ),
            "neutralize_styles": OperatorSpec(
                "neutralize_styles", 1, 5, category="neutralization"
            ),
        }
    )
    if include_scalar_ops:
        operators.update(
            {
                "power": OperatorSpec(
                    "power", 2, 3, category="mathematical", arg_types=("expr", "scalar")
                ),
                "pow": OperatorSpec(
                    "pow", 2, 3, category="mathematical", arg_types=("expr", "scalar")
                ),
                "clip": OperatorSpec(
                    "clip", 2, 3, category="basic", arg_types=("expr", "scalar")
                ),
                "winsorize": OperatorSpec(
                    "winsorize", 2, 3, category="basic", arg_types=("expr", "scalar")
                ),
                "cs_scale_to": OperatorSpec(
                    "cs_scale_to",
                    2,
                    3,
                    category="cross_section",
                    arg_types=("expr", "scalar"),
                ),
                "ts_quantile": OperatorSpec(
                    "ts_quantile",
                    3,
                    4,
                    category="time_series",
                    arg_types=("expr", "window", "scalar"),
                    lookback="rolling",
                ),
                "percentile": OperatorSpec(
                    "percentile",
                    3,
                    4,
                    category="time_series",
                    arg_types=("expr", "scalar", "window"),
                    lookback="rolling",
                ),
                "sma": OperatorSpec(
                    "sma",
                    3,
                    4,
                    category="technical",
                    arg_types=("expr", "window", "scalar"),
                    lookback="rolling",
                ),
            }
        )
    return operators


def build_feature_space(config: dict) -> FeatureSpace:
    feature_config = config.get("features", {})
    features: Dict[str, FeatureSpec] = {}

    for field in feature_config.get("daily_price_fields", []):
        features[f"daily_{field}"] = FeatureSpec(
            name=f"daily_{field}",
            source="price_daily",
            field=field,
            frequency="daily",
        )

    for spec in build_intraday_aggregator_specs(
        feature_config.get("intraday_5m_fields", []),
        feature_config.get("intraday_aggregators"),
    ):
        features[spec.name] = spec

    factor_fields = list(feature_config.get("factor_data_fields", []))
    # Size is a deliberately removable signal family. When the live strategy
    # already carries a strong standalone size factor, re-mining size only wastes
    # search budget and collapses the population into a size monoculture
    # (every run rediscovers neg(market_cap)). ``exclude_size`` strips size from
    # the selectable inputs AND prevents style-neutralization from re-appending
    # it -- the auto-append below would otherwise silently defeat an upstream
    # ``no_size`` family exclusion, which is exactly what was observed.
    size_fields = ("val_market_cap", "natural_log_of_market_cap", "cube_of_size")
    exclude_size = bool(feature_config.get("exclude_size", False))
    if exclude_size:
        factor_fields = [field for field in factor_fields if field not in size_fields]
    elif feature_config.get("enable_style_neutralization", False):
        for field in size_fields:
            if field not in factor_fields:
                factor_fields.append(field)

    for field in factor_fields:
        features[f"fd_{field}"] = FeatureSpec(
            name=f"fd_{field}",
            source="factor_data",
            field=field,
            frequency="daily",
        )

    operators = build_operator_registry(config)
    if exclude_size:
        # With size absent from the panel the size/style neutralizers degrade to
        # identity passthroughs (operator_runtime returns the input unchanged);
        # drop them so the search does not spend capacity wrapping expressions in
        # no-ops or minting trivially-duplicate variants.
        operators.pop("neutralize_size", None)
        operators.pop("neutralize_styles", None)
    windows = ()
    if _full_dsl_enabled(config):
        windows = tuple(
            sorted(
                {
                    int(window)
                    for window in config.get("search", {}).get(
                        "windows", [3, 5, 10, 20, 60]
                    )
                    if int(window) > 0
                }
            )
        )
    return FeatureSpace(features=features, operators=operators, windows=windows)


def validate_exportable_expression(
    expr: ExpressionNode, feature_space: FeatureSpace
) -> None:
    missing = referenced_features(expr) - set(feature_space.features)
    if missing:
        raise ValueError(f"Expression uses unknown features: {sorted(missing)}")

    def validate_node(node: ExpressionNode) -> None:
        if node.kind != "operator":
            return
        operator = feature_space.operators.get(str(node.value))
        if operator is None or not operator.exportable:
            raise ValueError(f"Expression uses unknown operator: {node.value}")
        if len(node.children) != operator.arity:
            raise ValueError(
                f"Operator {node.value} expects {operator.arity} args, got {len(node.children)}"
            )
        for expected_type, child in zip(operator.arg_types, node.children):
            if expected_type == "expr" and child.kind == "window":
                raise ValueError(f"Operator {node.value} expects expression argument")
            if expected_type == "scalar" and child.kind != "constant":
                raise ValueError(
                    f"Operator {node.value} expects {expected_type} constant argument"
                )
            if expected_type == "window":
                if child.kind not in {"window", "constant"}:
                    raise ValueError(
                        f"Operator {node.value} expects window constant argument"
                    )
                try:
                    window = int(float(child.value))
                except Exception as exc:
                    raise ValueError(
                        f"Operator {node.value} expects integer window argument"
                    ) from exc
                if float(child.value) != float(window):
                    raise ValueError(
                        f"Operator {node.value} expects integer window argument"
                    )
                if feature_space.windows and window not in feature_space.windows:
                    raise ValueError(
                        f"Operator {node.value} uses unsupported window: {window}"
                    )
        for child in node.children:
            validate_node(child)

    validate_node(expr)
