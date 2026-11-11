"""Point-in-time dividend-quality eligibility for top-k selection."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import duckdb
from utils.data_store import connect_read_only as _connect_data_read_only
import numpy as np
import pandas as pd


class QualityFilterError(RuntimeError):
    """Raised when an enabled quality filter cannot produce a valid mask."""


def apply_pit_quality_filter(
    dynamic_pools: Mapping[str, list[str]],
    *,
    db_path: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Filter each rebalance pool using only data disclosed by that date.

    ``mode=dividend_cut_veto`` only removes names whose latest disclosed
    comparable annual DPS is below the prior year. The default mode keeps the
    original dividend/financial quality screen.
    """

    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return {str(k): list(v or []) for k, v in dynamic_pools.items()}, {
            "enabled": False,
            "qualified_counts": {},
        }
    if not dynamic_pools:
        raise QualityFilterError("quality filter enabled but dynamic pool is empty")

    dates = sorted(pd.Timestamp(k).normalize() for k in dynamic_pools)
    stocks = sorted({str(s) for values in dynamic_pools.values() for s in (values or [])})
    if not stocks:
        raise QualityFilterError("quality filter enabled but dynamic pool has no stocks")
    placeholders = ",".join("?" for _ in stocks)
    start_date, end_date = dates[0].date(), dates[-1].date()
    conn = _connect_data_read_only(str(db_path))
    try:
        financials = conn.execute(
            f"""
            SELECT statement_type, stock_code, pub_date, start_date, end_date,
                   net_profit, net_operate_cash_flow, total_liability, total_assets
            FROM dividend_quality_financial_events
            WHERE stock_code IN ({placeholders}) AND pub_date <= ?
            """,
            [*stocks, end_date],
        ).df()
        dividend_sql = f"""
            SELECT stock_code, report_date, implementation_pub_date,
                   board_plan_pub_date, shareholders_plan_pub_date,
                   bonus_type, bonus_amount_rmb, bonus_ratio_rmb,
                   dividend_ratio, transfer_ratio
            FROM dividend_events
            WHERE stock_code IN ({placeholders})
              AND COALESCE(board_plan_pub_date, shareholders_plan_pub_date,
                           implementation_pub_date) IS NOT NULL
              AND COALESCE(board_plan_pub_date, shareholders_plan_pub_date,
                           implementation_pub_date) <= ?
              AND bonus_type = '年度分红'
              AND COALESCE(bonus_amount_rmb, 0) > 0
        """
        try:
            dividends = conn.execute(dividend_sql, [*stocks, end_date]).df()
        except duckdb.BinderException:
            # Legacy fixtures/DBs predate the early announcement columns.
            dividends = conn.execute(
                f"""
                SELECT stock_code, report_date, implementation_pub_date,
                       bonus_type, bonus_amount_rmb, bonus_ratio_rmb
                FROM dividend_events
                WHERE stock_code IN ({placeholders})
                  AND implementation_pub_date IS NOT NULL
                  AND implementation_pub_date <= ?
                  AND bonus_type = '年度分红'
                  AND COALESCE(bonus_amount_rmb, 0) > 0
                """,
                [*stocks, end_date],
            ).df()
            for column in ("board_plan_pub_date", "shareholders_plan_pub_date", "dividend_ratio", "transfer_ratio"):
                dividends[column] = pd.NaT if column.endswith("_pub_date") else 0.0
        valuation = conn.execute(
            f"""
            SELECT stock_code, date, market_cap
            FROM valuation_daily
            WHERE stock_code IN ({placeholders}) AND date BETWEEN ? AND ?
            """,
            [*stocks, start_date, end_date],
        ).df()
        industry = conn.execute(
            f"""
            SELECT stock_code, date, sw_l1_name
            FROM industry_membership
            WHERE stock_code IN ({placeholders}) AND date BETWEEN ? AND ?
            """,
            [*stocks, start_date, end_date],
        ).df()
    finally:
        conn.close()

    for frame, cols in (
        (financials, ["pub_date", "start_date", "end_date"]),
        (dividends, [
            "implementation_pub_date", "board_plan_pub_date",
            "shareholders_plan_pub_date", "report_date",
        ]),
        (valuation, ["date"]),
        (industry, ["date"]),
    ):
        for col in cols:
            if col in frame:
                frame[col] = pd.to_datetime(frame[col])

    financials = financials.sort_values(["stock_code", "pub_date", "end_date"])
    dividends["announcement_date"] = dividends[
        ["board_plan_pub_date", "shareholders_plan_pub_date", "implementation_pub_date"]
    ].bfill(axis=1).iloc[:, 0]
    dividends["comparable_dps"] = (
        pd.to_numeric(dividends["bonus_ratio_rmb"], errors="coerce") / 10.0
    ) * (
        1.0
        + pd.to_numeric(dividends["dividend_ratio"], errors="coerce").fillna(0.0) / 10.0
        + pd.to_numeric(dividends["transfer_ratio"], errors="coerce").fillna(0.0) / 10.0
    )
    dividends = dividends.sort_values(["stock_code", "report_date", "announcement_date"])
    valuation = valuation.sort_values(["stock_code", "date"])
    industry = industry.sort_values(["stock_code", "date"])
    fin_by_stock = {k: v for k, v in financials.groupby("stock_code", sort=False)}
    div_by_stock = {k: v for k, v in dividends.groupby("stock_code", sort=False)}
    val_by_stock = {k: v for k, v in valuation.groupby("stock_code", sort=False)}
    ind_by_stock = {k: v for k, v in industry.groupby("stock_code", sort=False)}

    if str(cfg.get("mode", "") or "") == "dividend_cut_veto":
        return _apply_dividend_cut_veto(dynamic_pools, div_by_stock, cfg)

    filtered: dict[str, list[str]] = {}
    diagnostics: dict[str, Any] = {"enabled": True, "qualified_counts": {}, "eligible_counts": {}}
    for raw_date, pool in dynamic_pools.items():
        asof = pd.Timestamp(raw_date).normalize()
        rows = []
        for stock in pool or []:
            row = _features_for_stock(
                stock,
                asof,
                fin_by_stock.get(stock),
                div_by_stock.get(stock),
                val_by_stock.get(stock),
                ind_by_stock.get(stock),
            )
            if row is not None:
                row["stock_code"] = stock
                rows.append(row)
        frame = pd.DataFrame(rows)
        if frame.empty:
            filtered[str(raw_date)] = []
            diagnostics["qualified_counts"][str(raw_date)] = 0
            diagnostics["eligible_counts"][str(raw_date)] = 0
            continue
        hard = (
            (frame["dividend_years"] >= 3)
            & frame["payout_mean"].between(0.10, 1.00, inclusive="both")
            & (frame["roe_count"] >= 12)
        )
        eligible = frame.loc[hard].copy()
        diagnostics["eligible_counts"][str(raw_date)] = int(len(eligible))
        if eligible.empty:
            filtered[str(raw_date)] = []
            diagnostics["qualified_counts"][str(raw_date)] = 0
            continue
        eligible["is_finance"] = eligible["industry_name"].fillna("").str.contains(
            "银行|非银金融|金融", regex=True
        )
        score_parts = ["roe_mean", "roe_change", "roe_stability", "dividend_yield"]
        for group_key, group in eligible.groupby("is_finance", sort=False):
            idx = group.index
            parts = [group[col].rank(pct=True) for col in score_parts]
            if not bool(group_key):
                parts.append(group["cfo_liability"].rank(pct=True))
            eligible.loc[idx, "quality_score"] = pd.concat(parts, axis=1).mean(axis=1)
        threshold = float(cfg.get("min_score_percentile", 0.5))
        selected = eligible.loc[eligible["quality_score"] >= threshold, "stock_code"]
        filtered[str(raw_date)] = selected.astype(str).tolist()
        diagnostics["qualified_counts"][str(raw_date)] = int(len(selected))

    diagnostics["min_qualified_count"] = min(diagnostics["qualified_counts"].values(), default=0)
    diagnostics["min_eligible_count"] = min(diagnostics["eligible_counts"].values(), default=0)
    if diagnostics["min_qualified_count"] < int(cfg.get("min_qualified_count", 1)):
        raise QualityFilterError(
            f"quality filter produced too few names: min={diagnostics['min_qualified_count']}"
        )
    return filtered, diagnostics


def _apply_dividend_cut_veto(
    dynamic_pools: Mapping[str, list[str]],
    div_by_stock: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    min_growth = float(cfg.get("min_comparable_dps_growth", 0.0))
    filtered: dict[str, list[str]] = {}
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "mode": "dividend_cut_veto",
        "min_comparable_dps_growth": min_growth,
        "qualified_counts": {},
        "veto_counts": {},
        "unknown_counts": {},
    }
    for raw_date, pool in dynamic_pools.items():
        asof = pd.Timestamp(raw_date).normalize()
        selected: list[str] = []
        vetoed = unknown = 0
        for stock in pool or []:
            growth = _latest_disclosed_dps_growth(div_by_stock.get(stock), asof)
            if growth is None:
                unknown += 1
                selected.append(stock)
            elif growth < min_growth:
                vetoed += 1
            else:
                selected.append(stock)
        filtered[str(raw_date)] = selected
        diagnostics["qualified_counts"][str(raw_date)] = len(selected)
        diagnostics["veto_counts"][str(raw_date)] = vetoed
        diagnostics["unknown_counts"][str(raw_date)] = unknown
    diagnostics["min_qualified_count"] = min(diagnostics["qualified_counts"].values(), default=0)
    if diagnostics["min_qualified_count"] < int(cfg.get("min_qualified_count", 1)):
        raise QualityFilterError(
            f"dividend cut veto produced too few names: min={diagnostics['min_qualified_count']}"
        )
    return filtered, diagnostics


def _latest_disclosed_dps_growth(dividends: pd.DataFrame | None, asof: pd.Timestamp) -> float | None:
    if dividends is None or dividends.empty:
        return None
    disclosed = dividends.loc[dividends["announcement_date"] <= asof].copy()
    if disclosed.empty:
        return None
    disclosed["year"] = disclosed["report_date"].dt.year
    yearly = disclosed.groupby("year")["comparable_dps"].sum().sort_index()
    if len(yearly) < 2 or int(yearly.index[-1]) != int(yearly.index[-2]) + 1:
        return None
    prior = float(yearly.iloc[-2])
    return float(yearly.iloc[-1]) / prior - 1.0 if prior > 0 else None


def _features_for_stock(
    stock: str,
    asof: pd.Timestamp,
    financials: pd.DataFrame | None,
    dividends: pd.DataFrame | None,
    valuation: pd.DataFrame | None,
    industry: pd.DataFrame | None,
) -> dict[str, Any] | None:
    if financials is None or dividends is None:
        return None
    fin = financials.loc[financials["pub_date"] <= asof]
    annual = fin.loc[(fin["statement_type"] == "income") & fin["end_date"].dt.month.eq(12)]
    annual = annual.sort_values("pub_date").drop_duplicates("end_date", keep="last")
    div = dividends.loc[dividends["announcement_date"] <= asof].copy()
    if div.empty:
        return None
    div["year"] = div["report_date"].dt.year
    yearly = div.groupby("year", as_index=False).agg(
        bonus_amount_rmb=("bonus_amount_rmb", "sum"),
        comparable_dps=("comparable_dps", "sum"),
    )
    annual_years = set(annual["end_date"].dt.year.astype(int))
    available_years = sorted(set(yearly["year"].astype(int)) & annual_years)
    recent_years = _latest_three_consecutive_years(available_years)
    yearly = yearly.loc[yearly["year"].isin(recent_years)]
    if len(recent_years) != 3:
        return None
    payout = []
    for _, item in yearly.iterrows():
        profit = annual.loc[annual["end_date"].dt.year.eq(int(item["year"])), "net_profit"]
        if profit.empty or float(profit.iloc[-1]) <= 0:
            continue
        payout.append(float(item["bonus_amount_rmb"]) * 10000.0 / float(profit.iloc[-1]))
    if len(payout) != 3:
        return None

    quarters = fin.loc[fin["statement_type"] == "income"].sort_values("pub_date")
    quarters = quarters.drop_duplicates("end_date", keep="last")
    balances = fin.loc[fin["statement_type"] == "balance"].sort_values("pub_date")
    balances = balances.drop_duplicates("end_date", keep="last").set_index("end_date")
    roe_values = []
    cfo_values = []
    for end in sorted(quarters["end_date"].unique())[-16:]:
        end = pd.Timestamp(end)
        if end > asof or end.month not in (3, 6, 9, 12):
            continue
        current = quarters.loc[quarters["end_date"].eq(end)].iloc[-1]
        annual_end = pd.Timestamp(year=end.year, month=12, day=31)
        prior_annual = quarters.loc[quarters["end_date"].eq(pd.Timestamp(year=end.year - 1, month=12, day=31))]
        prior_same = quarters.loc[quarters["end_date"].eq(pd.Timestamp(year=end.year - 1, month=end.month, day=end.day))]
        if end.month == 12:
            ttm_profit = current["net_profit"]
            ttm_cfo = current["net_operate_cash_flow"]
        elif prior_annual.empty or prior_same.empty:
            continue
        else:
            ttm_profit = current["net_profit"] + prior_annual.iloc[-1]["net_profit"] - prior_same.iloc[-1]["net_profit"]
            ttm_cfo = current["net_operate_cash_flow"] + prior_annual.iloc[-1]["net_operate_cash_flow"] - prior_same.iloc[-1]["net_operate_cash_flow"]
        if end not in balances.index:
            continue
        equity = float(balances.loc[end, "total_assets"] - balances.loc[end, "total_liability"])
        prior_end = end - pd.offsets.QuarterEnd()
        if prior_end in balances.index:
            prior_equity = float(balances.loc[prior_end, "total_assets"] - balances.loc[prior_end, "total_liability"])
        else:
            prior_equity = equity
        if equity <= 0 or prior_equity <= 0 or not np.isfinite(ttm_profit):
            continue
        roe_values.append(float(ttm_profit) / ((equity + prior_equity) / 2.0))
        cfo_values.append(float(ttm_cfo) / max(abs(float(balances.loc[end, "total_liability"])), 1e-12))
    if len(roe_values) < 12:
        return None
    market_cap = _asof_value(valuation, asof, "market_cap")
    if market_cap is None or market_cap <= 0:
        return None
    dividend_yield = float(yearly["bonus_amount_rmb"].sum()) / float(market_cap) / 10000.0
    roe = np.asarray(roe_values[-12:], dtype=float)
    cfo = np.asarray(cfo_values[-12:], dtype=float)
    industry_name = ""
    if industry is not None and not industry.empty:
        item = industry.loc[industry["date"] <= asof]
        if not item.empty:
            industry_name = str(item.iloc[-1]["sw_l1_name"] or "")
    return {
        "dividend_years": len(yearly),
        "payout_mean": float(np.mean(payout)),
        "roe_count": len(roe),
        "roe_mean": float(np.mean(roe)),
        "roe_change": float(np.mean(roe[-4:]) - np.mean(roe[:4])),
        "roe_stability": float(-np.std(roe)),
        "cfo_liability": float(np.nanmean(cfo)) if np.isfinite(cfo).any() else 0.0,
        "dividend_yield": dividend_yield,
        "industry_name": industry_name,
    }


def _asof_value(frame: pd.DataFrame | None, asof: pd.Timestamp, column: str) -> float | None:
    if frame is None or frame.empty:
        return None
    rows = frame.loc[frame["date"] <= asof]
    if rows.empty:
        return None
    value = rows.iloc[-1][column]
    return float(value) if pd.notna(value) else None


def _latest_three_consecutive_years(years: list[int]) -> list[int]:
    """Return the latest disclosed consecutive three-year block."""
    ordered = sorted(set(int(year) for year in years))
    blocks = [ordered[i : i + 3] for i in range(max(0, len(ordered) - 2))]
    blocks = [block for block in blocks if block[-1] - block[0] == 2]
    return blocks[-1] if blocks else []
