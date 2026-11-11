"""Comparable reference portfolios using the production VectorBT engine.

This module deliberately does not implement a second backtest engine.  Every
reference portfolio is expressed as an input score panel and evaluated through
``VectorBTBacktestRunner`` with the candidate's calendar, tradability and cost
configuration copied verbatim.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .backtest_runner import VectorBTBacktestRunner
from .factor_loader import cross_sectional_zscore


_CACHE_SCHEMA_VERSION = 4


def _frame_digest(frame: Optional[pd.DataFrame]) -> str:
    if frame is None:
        return "none"
    digest = hashlib.sha256()
    digest.update(repr(tuple(frame.shape)).encode())
    digest.update(repr(tuple(map(str, frame.columns))).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def fixed_benchmark_cache_path(
    *,
    cache_dir: str | Path,
    base_config: Mapping[str, Any],
    price_df: pd.DataFrame,
    open_df: Optional[pd.DataFrame],
    factor_ratio_df: Optional[pd.DataFrame],
    dynamic_pools: Mapping[str, list[str]],
    factor1_panel: Optional[pd.DataFrame],
    rank_average_score: Optional[pd.DataFrame],
    quality_score: Optional[pd.DataFrame],
    candidate_is_rank_average: bool,
    random_seeds: int,
    rebalance_dates: list[pd.Timestamp],
) -> Path:
    """Build a content-addressed path for candidate-independent benchmarks."""
    benchmark_config = deepcopy(dict(base_config))
    benchmark_config.get("top_k_strategy", {}).pop("score_decay", None)
    payload = {
        "schema": _CACHE_SCHEMA_VERSION,
        "base_config": benchmark_config,
        "price": _frame_digest(price_df),
        "open": _frame_digest(open_df),
        "factor_ratio": _frame_digest(factor_ratio_df),
        "pools": {
            str(date): sorted(map(str, stocks or []))
            for date, stocks in dynamic_pools.items()
        },
        "factor1": _frame_digest(factor1_panel),
        "rank_average": _frame_digest(rank_average_score),
        "quality": _frame_digest(quality_score),
        "candidate_is_rank_average": candidate_is_rank_average,
        "random_seeds": int(random_seeds),
        "rebalance_dates": [str(pd.Timestamp(date).date()) for date in rebalance_dates],
    }
    encoded = json.dumps(
        payload, sort_keys=True, default=str, separators=(",", ":")
    ).encode()
    return (
        Path(cache_dir) / f"fixed_benchmark_{hashlib.sha256(encoded).hexdigest()}.pkl"
    )


def _load_fixed_cache(
    path: Optional[Path],
) -> Optional[tuple[Dict[str, Dict[str, Any]], Dict[str, pd.Series], np.ndarray]]:
    if path is None or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("schema") != _CACHE_SCHEMA_VERSION:
            return None
        records = payload.get("records")
        return_series = payload.get("return_series")
        random_values = payload.get("random_values")
        if (
            not isinstance(records, dict)
            or not isinstance(return_series, dict)
            or not all(isinstance(value, pd.Series) for value in return_series.values())
            or not isinstance(random_values, np.ndarray)
        ):
            return None
        return (
            deepcopy(records),
            {name: value.copy() for name, value in return_series.items()},
            random_values.copy(),
        )
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
        return None


def _store_fixed_cache(
    path: Optional[Path],
    records: Dict[str, Dict[str, Any]],
    return_series: Dict[str, pd.Series],
    random_values: np.ndarray,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "schema": _CACHE_SCHEMA_VERSION,
                    "records": records,
                    "return_series": return_series,
                    "random_values": random_values,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def is_equal_rank_average_config(
    method_config: Mapping[str, Any],
    factor_ids: list[str],
    reference_factor_ids: list[str],
) -> bool:
    """Return whether the candidate is exactly the configured rank average."""
    if method_config.get("method") != "weighted_sum":
        return False
    if len(factor_ids) != len(reference_factor_ids) or set(factor_ids) != set(
        reference_factor_ids
    ):
        return False
    weights = method_config.get("weights", {})
    preprocess = method_config.get("preprocess", {})
    try:
        values = np.asarray([float(weights[name]) for name in factor_ids], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        len(values)
        and np.isfinite(values).all()
        and np.allclose(values, values[0], rtol=0.0, atol=1e-12)
        and values[0] != 0.0
        and all(preprocess.get(name) == "rank" for name in factor_ids)
    )


def equal_score_panel(price_df: pd.DataFrame) -> pd.DataFrame:
    """Score every available stock equally; all-equal selection buys the pool."""
    return pd.DataFrame(1.0, index=price_df.index, columns=price_df.columns)


def random_score_panel(price_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Deterministic iid ranks for a random top-k reference portfolio."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.random(price_df.shape), index=price_df.index, columns=price_df.columns
    )


def rank_average_panel(factors: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally rank each factor then average, avoiding scale leakage."""
    if isinstance(factors.columns, pd.MultiIndex):
        raise ValueError("rank_average_panel expects one wide score panel per factor")
    return factors.rank(axis=1, pct=True, method="average").mean(axis=1).to_frame()


def winsorized_zscore_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply the governed serving-side ``zscore`` to each date."""
    return panel.apply(cross_sectional_zscore, axis=1)


def _returns(result: Mapping[str, Any]) -> pd.Series:
    portfolio = result.get("portfolio")
    if portfolio is None:
        return pd.Series(dtype=float)
    values = portfolio.returns()
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0] if values.shape[1] == 1 else values.mean(axis=1)
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna()


def _metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "annual_return",
            "sharpe_ratio",
            "max_drawdown",
            "annual_turnover",
            "avg_holding_count",
            "max_holding_count",
            "holding_count_violation_days",
            "max_excess_holding_notional",
            "holding_count_violation_dates",
            "holding_count_violation_details",
            "execution_diagnostics",
            "trade_count",
        )
    }


def _run_one(
    config: Mapping[str, Any],
    score_panel: pd.DataFrame,
    price_df: pd.DataFrame,
    open_df: Optional[pd.DataFrame],
    factor_ratio_df: Optional[pd.DataFrame],
    dynamic_pools: Mapping[str, list[str]],
    tradeability_cache: Optional[Dict[tuple[str, tuple[str, ...]], list[str]]] = None,
    *,
    all_equal_weight: bool = False,
    score_direction: float = 1.0,
    rebalance_dates: Optional[list[pd.Timestamp]] = None,
) -> tuple[Dict[str, Any], pd.Series]:
    runner_config = deepcopy(dict(config))
    top_cfg = runner_config.setdefault("top_k_strategy", {})
    top_cfg["score_source"] = "input_factor"
    top_cfg["score_direction"] = score_direction
    top_cfg["all_equal_weight"] = all_equal_weight
    top_cfg.pop("score_decay", None)
    if all_equal_weight:
        # EW intentionally assigns a common score to every eligible name; the
        # candidate's anti-tie quality gate is not applicable to that reference.
        top_cfg["selection_quality"] = {"enabled": False}
    runner = VectorBTBacktestRunner(
        runner_config, tradeability_cache=tradeability_cache
    )
    try:
        result = runner.run_backtest(
            factor_df=score_panel,
            price_df=price_df,
            open_df=open_df,
            factor_ratio_df=factor_ratio_df,
            dynamic_pools=dict(dynamic_pools),
            rebalance_freq="weekly",
            rebalance_dates=rebalance_dates,
        )
        return result, _returns(result)
    finally:
        runner.close()


def run_benchmark_suite(
    *,
    base_config: Mapping[str, Any],
    price_df: pd.DataFrame,
    open_df: Optional[pd.DataFrame],
    factor_ratio_df: Optional[pd.DataFrame],
    dynamic_pools: Mapping[str, list[str]],
    factor1_panel: Optional[pd.DataFrame] = None,
    rank_average_score: Optional[pd.DataFrame] = None,
    quality_score: Optional[pd.DataFrame] = None,
    candidate_result: Optional[Mapping[str, Any]] = None,
    candidate_gross_result: Optional[Mapping[str, Any]] = None,
    candidate_is_rank_average: bool = False,
    random_seeds: int = 100,
    fixed_cache_path: Optional[Path] = None,
    parent_benchmark: str = "factor1_top",
) -> Dict[str, Any]:
    """Run P2 references under identical engine/cost/calendar assumptions.

    ``candidate_result`` may be supplied by the caller for the current LASSO or
    LightGBM result.  It is retained as a same-engine net result; gross is
    separately rerun where a score panel is available.
    """
    selection_rows = (
        candidate_result.get("selection_diagnostics", {}).get("rows", [])
        if isinstance(candidate_result, Mapping)
        else []
    )
    rebalance_dates = [
        pd.Timestamp(row["date"]).normalize()
        for row in selection_rows
        if isinstance(row, Mapping) and row.get("date")
    ]
    cached = _load_fixed_cache(fixed_cache_path)
    cache_hit = cached is not None
    if cached is not None:
        records, return_series, random_values = cached
    else:
        records = {}
        return_series = {}
        tradeability_cache: Dict[tuple[str, tuple[str, ...]], list[str]] = {}

        def add(name: str, panel: pd.DataFrame, **kwargs: Any) -> None:
            net, net_returns = _run_one(
                base_config,
                panel,
                price_df,
                open_df,
                factor_ratio_df,
                dynamic_pools,
                tradeability_cache=tradeability_cache,
                rebalance_dates=rebalance_dates or None,
                **kwargs,
            )
            gross_config = deepcopy(dict(base_config))
            gross_config.setdefault("top_k_strategy", {})["cost_rate"] = 0.0
            gross_config["top_k_strategy"]["slippage"] = 0.0
            gross, _ = _run_one(
                gross_config,
                panel,
                price_df,
                open_df,
                factor_ratio_df,
                dynamic_pools,
                tradeability_cache=tradeability_cache,
                rebalance_dates=rebalance_dates or None,
                **kwargs,
            )
            records[name] = {"net": _metrics(net), "gross": _metrics(gross)}
            return_series[name] = net_returns

        add("pool_ew", equal_score_panel(price_df), all_equal_weight=True)
        if factor1_panel is not None and not factor1_panel.empty:
            add("factor1_top", factor1_panel, score_direction=1.0)
            add("factor1_bottom", factor1_panel, score_direction=-1.0)
        if (
            not candidate_is_rank_average
            and rank_average_score is not None
            and not rank_average_score.empty
        ):
            add("rank_average", rank_average_score)
        if quality_score is not None and not quality_score.empty:
            add("quality_ew", quality_score, all_equal_weight=True)

        random_sharpes = []
        random_max_holding_counts: list[int] = []
        random_violation_days: list[int] = []
        for seed in range(int(random_seeds)):
            random_result, _ = _run_one(
                base_config,
                random_score_panel(price_df, seed),
                price_df,
                open_df,
                factor_ratio_df,
                dynamic_pools,
                tradeability_cache=tradeability_cache,
                rebalance_dates=rebalance_dates or None,
            )
            random_sharpes.append(float(random_result.get("sharpe_ratio", np.nan)))
            random_max_holding_counts.append(
                int(random_result.get("max_holding_count") or 0)
            )
            random_violation_days.append(
                int(random_result.get("holding_count_violation_days") or 0)
            )
        random_values = np.asarray(
            [x for x in random_sharpes if np.isfinite(x)], dtype=float
        )
        records["random_top_k"] = {
            "distribution": {
                "seed_count": int(len(random_values)),
                "sharpe_mean": float(np.mean(random_values))
                if len(random_values)
                else None,
                "sharpe_p05": float(np.quantile(random_values, 0.05))
                if len(random_values)
                else None,
                "sharpe_p50": float(np.quantile(random_values, 0.50))
                if len(random_values)
                else None,
                "sharpe_p95": float(np.quantile(random_values, 0.95))
                if len(random_values)
                else None,
                "max_observed_holding_count": max(random_max_holding_counts)
                if random_max_holding_counts
                else 0,
                "holding_limit_violation_seed_count": int(
                    sum(value > 0 for value in random_violation_days)
                ),
                "holding_limit_violation_day_count": int(sum(random_violation_days)),
            }
        }
        _store_fixed_cache(
            fixed_cache_path,
            records,
            return_series,
            random_values,
        )

    if candidate_is_rank_average and candidate_result is not None:
        records["rank_average"] = {"net": _metrics(candidate_result)}
        if candidate_gross_result is not None:
            records["rank_average"]["gross"] = _metrics(candidate_gross_result)
        return_series["rank_average"] = _returns(candidate_result)

    if candidate_result is not None:
        candidate_returns = _returns(candidate_result)
        return_series["candidate"] = candidate_returns
        records["candidate"] = {"net": _metrics(candidate_result)}
        if candidate_gross_result is not None:
            records["candidate"]["gross"] = _metrics(candidate_gross_result)
        pool_active = _active_metrics(
            candidate_returns,
            return_series.get("pool_ew", pd.Series(dtype=float)),
            "pool_ew",
        )
        if pool_active:
            records["candidate"]["active"] = {
                "annual_active_return_vs_pool_ew": pool_active["annual_active_return"],
                "tracking_error": pool_active["tracking_error"],
                "information_ratio": pool_active["information_ratio"],
                "comparison_start": pool_active["comparison_start"],
                "comparison_end": pool_active["comparison_end"],
                "observations": pool_active["observations"],
            }
        parent_active = _active_metrics(
            candidate_returns,
            return_series.get(parent_benchmark, pd.Series(dtype=float)),
            parent_benchmark,
        )
        if parent_active:
            records["candidate"]["active_vs_parent"] = parent_active
        random_dist = records["random_top_k"]["distribution"]
        candidate_sharpe = records["candidate"]["net"].get("sharpe_ratio")
        if candidate_sharpe is not None and len(random_values):
            random_dist["candidate_sharpe_percentile"] = float(
                np.mean(random_values <= float(candidate_sharpe))
            )

    return {
        "schema_version": "1.0",
        "engine": "VectorBTBacktestRunner",
        "parent_benchmark": parent_benchmark,
        "assumptions": {
            "calendar_filters": dict(base_config.get("calendar_filters", {})),
            "top_k_strategy": dict(base_config.get("top_k_strategy", {})),
            "rebalance_frequency": "weekly",
            "execution": "next_open_when_available",
            "rebalance_dates": [
                str(pd.Timestamp(date).date()) for date in rebalance_dates
            ],
        },
        "benchmarks": records,
        "fixed_benchmark_cache": {
            "enabled": fixed_cache_path is not None,
            "hit": cache_hit,
            "path": str(fixed_cache_path) if fixed_cache_path is not None else "",
        },
    }


def _active_metrics(
    candidate_returns: pd.Series,
    benchmark_returns: pd.Series,
    benchmark_name: str,
) -> Dict[str, Any]:
    aligned = pd.concat(
        [candidate_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        return {}
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = (
        float(active.std(ddof=1) * np.sqrt(252)) if len(active) > 1 else None
    )
    active_return = float((1.0 + active).prod() ** (252 / len(active)) - 1.0)
    return {
        "benchmark": benchmark_name,
        "annual_active_return": active_return,
        "tracking_error": tracking_error,
        "information_ratio": active_return / tracking_error
        if tracking_error and tracking_error > 0
        else None,
        "comparison_start": str(pd.Timestamp(aligned.index.min()).date()),
        "comparison_end": str(pd.Timestamp(aligned.index.max()).date()),
        "observations": int(len(aligned)),
    }
