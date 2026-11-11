"""Parent-specific residual RankIC screen for algorithm-research mutations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
from utils.data_store import connect_read_only as _connect_data_read_only
import numpy as np
import pandas as pd

SCHEMA_VERSION = "algorithm_parent_factor_screen.v1"
EVIDENCE_VERSION = "algorithm_parent_factor_screen_evidence.v1"


class ParentFactorScreenError(RuntimeError):
    """Raised when frozen parent-screen inputs cannot be evaluated safely."""


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _newey_west_tstat(values: Sequence[float], *, lag: int) -> float:
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < 2:
        return 0.0
    residual = series - series.mean()
    variance = float(np.dot(residual, residual) / len(series))
    for offset in range(1, min(max(0, lag), len(series) - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        variance += 2.0 * weight * float(
            np.dot(residual[offset:], residual[:-offset]) / len(series)
        )
    standard_error = math.sqrt(max(variance, 0.0) / len(series))
    return float(series.mean() / standard_error) if standard_error else 0.0


def _partial_rank_ic(frame: pd.DataFrame, *, min_stocks: int) -> pd.DataFrame:
    """Daily partial Spearman(candidate, return | exact-parent score)."""
    keys = ["factor_id", "date"]
    required = {
        "factor_id",
        "date",
        "stock_code",
        "factor_value",
        "parent_score",
        "forward_return",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=["factor_id", "date", "stock_count", "raw_rank_ic", "residual_rank_ic"]
        )
    work = frame.dropna(
        subset=["factor_value", "parent_score", "forward_return"]
    ).copy()
    work = work[
        work.groupby(keys)["stock_code"].transform("size") >= max(3, min_stocks)
    ]
    if work.empty:
        return pd.DataFrame(
            columns=["factor_id", "date", "stock_count", "raw_rank_ic", "residual_rank_ic"]
        )

    grouped = work.groupby(keys, sort=True)
    centered: dict[str, pd.Series] = {}
    for column in ("factor_value", "parent_score", "forward_return"):
        ranks = grouped[column].rank(method="average")
        centered[column] = ranks - ranks.groupby([work["factor_id"], work["date"]]).transform("mean")
    x = centered["factor_value"]
    p = centered["parent_score"]
    y = centered["forward_return"]
    products = work[keys].copy()
    products["xx"] = x * x
    products["pp"] = p * p
    products["yy"] = y * y
    products["xp"] = x * p
    products["xy"] = x * y
    products["py"] = p * y
    products["stock_count"] = 1
    sums = products.groupby(keys, sort=True).sum().reset_index()
    sums["raw_rank_ic"] = sums["xy"] / np.sqrt(sums["xx"] * sums["yy"])
    r_xp = sums["xp"] / np.sqrt(sums["xx"] * sums["pp"])
    r_py = sums["py"] / np.sqrt(sums["pp"] * sums["yy"])
    denominator = np.sqrt((1.0 - r_xp**2).clip(lower=0.0) * (1.0 - r_py**2).clip(lower=0.0))
    numerator = sums["raw_rank_ic"] - r_xp * r_py
    sums["residual_rank_ic"] = np.divide(
        numerator,
        denominator,
        out=np.zeros(len(sums), dtype=float),
        where=denominator > 1e-12,
    )
    return sums[["factor_id", "date", "stock_count", "raw_rank_ic", "residual_rank_ic"]].replace(
        [np.inf, -np.inf], np.nan
    )


def summarize_screen_frame(
    frame: pd.DataFrame,
    *,
    candidates: Sequence[Mapping[str, Any]],
    target_period: int,
    min_stocks: int,
) -> list[dict[str, Any]]:
    """Summarize one already-aligned parent/candidate/return panel."""
    daily = _partial_rank_ic(frame, min_stocks=min_stocks)
    by_factor = {str(item["factor_id"]): item for item in candidates}
    results: list[dict[str, Any]] = []
    for factor_id in sorted(by_factor):
        item = by_factor[factor_id]
        sample = daily[daily["factor_id"] == factor_id].dropna(subset=["residual_rank_ic"]).copy()
        directions = {
            str(year): int(direction)
            for year, direction in dict(item.get("direction_by_year") or {}).items()
            if int(direction) in {-1, 1}
        }
        sample["year"] = pd.to_datetime(sample["date"]).dt.year.astype(str)
        sample["direction"] = sample["year"].map(directions)
        sample = sample.dropna(subset=["direction"])
        if sample.empty:
            results.append(
                {
                    "factor_id": factor_id,
                    "status": "unavailable",
                    "reason": "no_direction_adjusted_residual_ic",
                    "source_hash": str(item.get("source_hash", "")),
                }
            )
            continue
        sample["direction_adjusted_residual_rank_ic"] = (
            sample["residual_rank_ic"] * sample["direction"]
        )
        annual: dict[str, Any] = {}
        for year, values in sample.groupby("year", sort=True):
            annual[str(year)] = {
                "direction": directions[str(year)],
                "residual_rank_ic_mean": float(values["residual_rank_ic"].mean()),
                "direction_adjusted_residual_rank_ic_mean": float(
                    values["direction_adjusted_residual_rank_ic"].mean()
                ),
                "valid_dates": int(len(values)),
            }
        signed = sample["direction_adjusted_residual_rank_ic"].to_numpy(dtype=float)
        mean = float(np.mean(signed))
        positive_year_ratio = sum(
            row["direction_adjusted_residual_rank_ic_mean"] > 0 for row in annual.values()
        ) / len(annual)
        classification = (
            "redundant"
            if mean <= 0
            else "incremental_candidate"
            if positive_year_ratio >= 0.6
            else "unstable"
        )
        results.append(
            {
                "factor_id": factor_id,
                "status": "computed",
                "classification": classification,
                "source_hash": str(item.get("source_hash", "")),
                "raw_rank_ic_mean": float(sample["raw_rank_ic"].mean()),
                "residual_rank_ic_mean": float(sample["residual_rank_ic"].mean()),
                "direction_adjusted_residual_rank_ic_mean": mean,
                "direction_adjusted_residual_rank_ic_hac_t": _newey_west_tstat(
                    signed, lag=max(0, int(target_period) - 1)
                ),
                "positive_year_ratio": float(positive_year_ratio),
                "valid_dates": int(len(sample)),
                "valid_years": int(len(annual)),
                "mean_stock_count": float(sample["stock_count"].mean()),
                "annual": annual,
            }
        )
    return sorted(
        results,
        key=lambda row: (
            row.get("status") != "computed",
            -float(row.get("direction_adjusted_residual_rank_ic_mean", float("-inf"))),
            str(row["factor_id"]),
        ),
    )


def _load_parent(job: Mapping[str, Any]) -> pd.DataFrame:
    parent = job.get("parent")
    if not isinstance(parent, Mapping):
        raise ParentFactorScreenError("parent contract is missing")
    path = Path(str(parent.get("path", ""))).expanduser().resolve()
    expected = str(parent.get("sha256", ""))
    if not path.is_file() or not expected or _sha256(path) != expected:
        raise ParentFactorScreenError("parent score hash mismatch")
    frame = pd.read_parquet(path).reset_index()
    required = {"date", "stock_code", "composite_value"}
    if not required.issubset(frame.columns):
        raise ParentFactorScreenError("parent score schema is invalid")
    frame = frame.rename(columns={"composite_value": "parent_score"})[
        ["date", "stock_code", "parent_score"]
    ]
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["parent_score"] = pd.to_numeric(frame["parent_score"], errors="coerce")
    if frame.duplicated(["date", "stock_code"]).any() or frame["parent_score"].isna().any():
        raise ParentFactorScreenError("parent score values are invalid")
    return frame


def _load_forward_returns(
    parent: pd.DataFrame,
    *,
    price_db_path: Path,
    price_table: str,
    target_period: int,
    development_end: str,
) -> pd.DataFrame:
    start = pd.Timestamp(parent["date"].min()).date()
    end = date.fromisoformat(development_end)
    query_end = end + timedelta(days=(target_period + 10) * 3)
    stocks = sorted(parent["stock_code"].unique())
    query = (
        f"WITH px AS (SELECT stock_code, date, close * factor AS adj FROM {price_table} "
        "WHERE stock_code IN (SELECT UNNEST(?::VARCHAR[])) AND date BETWEEN ? AND ?), "
        "fwd AS (SELECT stock_code, date, LEAD(adj, 1) OVER w AS entry_adj, "
        "LEAD(adj, 1 + ?) OVER w AS exit_adj, LEAD(date, 1 + ?) OVER w AS exit_date "
        "FROM px WINDOW w AS (PARTITION BY stock_code ORDER BY date)) "
        "SELECT date, stock_code, exit_adj / entry_adj - 1.0 AS forward_return, exit_date FROM fwd "
        "WHERE entry_adj IS NOT NULL AND exit_adj IS NOT NULL AND entry_adj <> 0 AND exit_date <= ?"
    )
    with _connect_data_read_only(str(price_db_path)) as connection:
        frame = connection.execute(
            query,
            [stocks, start, query_end, target_period, target_period, end],
        ).fetch_df()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame[["date", "stock_code", "forward_return"]]


def run_parent_factor_screen(job_path: Path, output_path: Path) -> dict[str, Any]:
    """Execute one frozen parent screen and write deterministic evidence."""
    raw = job_path.read_bytes()
    job = json.loads(raw)
    if not isinstance(job, dict) or job.get("schema_version") != SCHEMA_VERSION:
        raise ParentFactorScreenError("invalid parent screen job")
    development = job.get("development")
    database = job.get("database")
    candidates = job.get("candidates")
    if not isinstance(development, Mapping) or not isinstance(database, Mapping):
        raise ParentFactorScreenError("parent screen contract is incomplete")
    if not isinstance(candidates, list) or not candidates:
        raise ParentFactorScreenError("parent screen has no candidates")
    start = str(development.get("start_date", ""))
    end = str(development.get("end_date", ""))
    if not start or not end or start > end or end > "2023-12-31":
        raise ParentFactorScreenError("parent screen development window is invalid")
    target_period = int(job.get("target_period", 0))
    min_stocks = int(job.get("min_stocks", 20))
    if target_period <= 0 or min_stocks < 3:
        raise ParentFactorScreenError("parent screen parameters are invalid")
    factor_db_path = Path(str(database.get("factor_db_path", ""))).expanduser().resolve()
    price_db_path = Path(str(database.get("price_db_path", ""))).expanduser().resolve()
    if not factor_db_path.is_file() or not price_db_path.is_file():
        raise ParentFactorScreenError("parent screen database is unavailable")

    parent = _load_parent(job)
    parent = parent[parent["date"].between(start, end)]
    forward = _load_forward_returns(
        parent,
        price_db_path=price_db_path,
        price_table=str(database.get("price_table", "price_data_daily")),
        target_period=target_period,
        development_end=end,
    )
    base = parent.merge(forward, on=["date", "stock_code"], how="inner", validate="one_to_one")
    factor_table = str(database.get("factor_table", "factor_values"))
    factor_ids = [str(item.get("factor_id", "")) for item in candidates if isinstance(item, Mapping)]
    if len(factor_ids) != len(set(factor_ids)) or any(not factor_id for factor_id in factor_ids):
        raise ParentFactorScreenError("parent screen candidate identities are invalid")

    panels: list[pd.DataFrame] = []
    with _connect_data_read_only(str(factor_db_path)) as connection:
        for offset in range(0, len(factor_ids), 32):
            batch = factor_ids[offset : offset + 32]
            values = connection.execute(
                f"SELECT date, stock_code, factor_name AS factor_id, factor_value FROM {factor_table} "
                "WHERE factor_name IN (SELECT UNNEST(?::VARCHAR[])) AND date BETWEEN ? AND ? "
                "AND factor_value IS NOT NULL",
                [batch, date.fromisoformat(start), date.fromisoformat(end)],
            ).fetch_df()
            if values.empty:
                continue
            values["date"] = pd.to_datetime(values["date"]).dt.normalize()
            values["stock_code"] = values["stock_code"].astype(str)
            if values.duplicated(["factor_id", "date", "stock_code"]).any():
                raise ParentFactorScreenError("candidate factor values are duplicated")
            panels.append(values.merge(base, on=["date", "stock_code"], how="inner", validate="many_to_one"))
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    results = summarize_screen_frame(
        panel,
        candidates=candidates,
        target_period=target_period,
        min_stocks=min_stocks,
    )
    payload = {
        "schema_version": EVIDENCE_VERSION,
        "job_sha256": hashlib.sha256(raw).hexdigest(),
        "pool_id": str(job.get("pool_id", "")),
        "parent_winner_id": str(job["parent"].get("parent_winner_id", "")),
        "parent_run_id": str(job["parent"].get("parent_run_id", "")),
        "parent_score_sha256": str(job["parent"].get("sha256", "")),
        "development": {"start_date": start, "end_date": end},
        "target_period": target_period,
        "tested_factor_count": len(factor_ids),
        "computed_factor_count": sum(item.get("status") == "computed" for item in results),
        "classification_counts": {
            name: sum(item.get("classification") == name for item in results)
            for name in ("incremental_candidate", "redundant", "unstable")
        },
        "ranking": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return payload


__all__ = [
    "EVIDENCE_VERSION",
    "ParentFactorScreenError",
    "SCHEMA_VERSION",
    "run_parent_factor_screen",
    "summarize_screen_frame",
]
