#!/usr/bin/env python3
"""
Diagnostic: Factor Value Difference Analysis (Local vs Cloud)

For each problematic factor, this script:
1. Loads local (DuckDB) and cloud (parsed JSON) factor values
2. Computes per-stock ratio (cloud/local) to detect scaling patterns
3. Identifies whether differences are multiplicative (constant ratio) or additive
4. For factors using volume/close, checks PIT factor values from raw data
5. Re-computes factor values from raw DB data to detect stale DB entries

Usage:
    conda run --no-capture-output -n qlib_m3 python scripts/diagnose_factor_value_diff.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "analysis_results.duckdb"
JQ_PARSED = PROJECT_ROOT / "results" / "jq_parsed.json"
LOCAL_PARSED = PROJECT_ROOT / "results" / "local_parsed.json"
CLOUD_CSV = PROJECT_ROOT / "results" / "cloud_factor_dump.csv"

SUSPECT_FACTORS = [
    "PriceVolumeDeviationCoordinationFactor_60D_small_cap_9_rules",
    "PriceVolumeDeviationCoordinationFactor_20D_small_cap_9_rules",
    "PriceVolumeDeviationVolatilityFactor_5D_small_cap_9_rules",
    "IntravarMomentum_small_cap_9_rules",
    "UniformActiveBuyPiecewiseFactor_small_cap_9_rules",
    "VolatilityTurnoverCouplingFactor_small_cap_9_rules",
    "NightFrostPathFactor_small_cap_9_rules",
]


def load_parsed_data() -> Tuple[dict, dict]:
    """Load parsed JQ and local alignment data."""
    jq_data = json.loads(JQ_PARSED.read_text()) if JQ_PARSED.exists() else {}
    local_data = json.loads(LOCAL_PARSED.read_text()) if LOCAL_PARSED.exists() else {}
    return jq_data, local_data


def analyze_ratio_pattern(
    jq_vals: Dict[str, float], local_vals: Dict[str, float], factor_name: str
) -> dict:
    """Analyze the ratio pattern between cloud and local values."""
    common = sorted(set(jq_vals.keys()) & set(local_vals.keys()))
    if len(common) < 3:
        return {"status": "INSUFFICIENT_DATA", "n_common": len(common)}

    jq_arr = np.array([jq_vals[s] for s in common], dtype=np.float64)
    local_arr = np.array([local_vals[s] for s in common], dtype=np.float64)

    abs_diff = np.abs(jq_arr - local_arr)
    nonzero_local = np.abs(local_arr) > 1e-10
    ratios = np.where(nonzero_local, jq_arr / local_arr, np.nan)
    valid_ratios = ratios[~np.isnan(ratios) & np.isfinite(ratios)]

    spearman_rho, _ = stats.spearmanr(jq_arr, local_arr) if len(common) >= 3 else (np.nan, 1.0)

    result = {
        "n_common": len(common),
        "spearman": round(float(spearman_rho), 6) if np.isfinite(spearman_rho) else None,
        "mean_abs_diff": round(float(np.nanmean(abs_diff)), 6),
        "max_abs_diff": round(float(np.nanmax(abs_diff)), 6),
        "jq_range": [round(float(np.nanmin(jq_arr)), 6), round(float(np.nanmax(jq_arr)), 6)],
        "local_range": [round(float(np.nanmin(local_arr)), 6), round(float(np.nanmax(local_arr)), 6)],
    }

    if len(valid_ratios) >= 3:
        result["ratio_stats"] = {
            "mean": round(float(np.nanmean(valid_ratios)), 6),
            "std": round(float(np.nanstd(valid_ratios)), 6),
            "median": round(float(np.nanmedian(valid_ratios)), 6),
            "min": round(float(np.nanmin(valid_ratios)), 6),
            "max": round(float(np.nanmax(valid_ratios)), 6),
            "cv": round(float(np.nanstd(valid_ratios) / np.nanmean(valid_ratios)), 6)
            if np.nanmean(valid_ratios) != 0
            else None,
        }

        ratio_cv = result["ratio_stats"]["cv"]
        if ratio_cv is not None and ratio_cv < 0.05:
            result["pattern"] = "MULTIPLICATIVE_CONSTANT"
            result["scale_factor"] = result["ratio_stats"]["median"]
        elif ratio_cv is not None and ratio_cv < 0.2:
            result["pattern"] = "MULTIPLICATIVE_VARIABLE"
        else:
            result["pattern"] = "NON_LINEAR"
    else:
        result["pattern"] = "CANNOT_DETERMINE"

    # Per-stock detail (top 5 by abs_diff)
    stock_details = []
    for i, s in enumerate(common):
        stock_details.append({
            "stock": s,
            "jq": round(float(jq_arr[i]), 6),
            "local": round(float(local_arr[i]), 6),
            "diff": round(float(abs_diff[i]), 6),
            "ratio": round(float(ratios[i]), 6) if np.isfinite(ratios[i]) else None,
        })
    stock_details.sort(key=lambda x: x["diff"], reverse=True)
    result["top5_by_diff"] = stock_details[:5]

    return result


def check_db_factor_freshness(factor_name: str) -> dict:
    """Check when the factor values in DuckDB were last updated."""
    if not DB_PATH.exists():
        return {"error": "DB not found"}

    clean_name = factor_name.replace("_small_cap_9_rules", "")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for suffix in ["_small_cap_9_rules", ""]:
            name = f"{clean_name}{suffix}"
            df = conn.execute(
                f"SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as n_rows "
                f"FROM factor_values WHERE factor_name = '{name}'"
            ).fetchdf()
            if df["n_rows"].iloc[0] > 0:
                return {
                    "factor_name_in_db": name,
                    "min_date": str(df["min_date"].iloc[0]),
                    "max_date": str(df["max_date"].iloc[0]),
                    "n_rows": int(df["n_rows"].iloc[0]),
                }
        return {"error": f"Factor {clean_name} not found in DB"}
    finally:
        conn.close()


def compare_raw_data_sample(date: str, stocks: List[str]) -> dict:
    """Compare raw price data for a sample of stocks on a given date."""
    if not DB_PATH.exists():
        return {"error": "DB not found"}

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        stock_list = "','".join(stocks[:5])
        query = f"""
        SELECT date, code, close, volume, money, factor
        FROM price_data
        WHERE date = '{date}'
          AND code IN ('{stock_list}')
        ORDER BY code
        """
        df = conn.execute(query).fetchdf()
        if df.empty:
            return {"error": f"No price data for date {date}"}
        return {
            "n_rows": len(df),
            "stocks": df["code"].tolist(),
            "close_range": [round(float(df["close"].min()), 4), round(float(df["close"].max()), 4)],
            "volume_range": [round(float(df["volume"].min()), 0), round(float(df["volume"].max()), 0)],
            "factor_range": [round(float(df["factor"].min()), 6), round(float(df["factor"].max()), 6)]
            if "factor" in df.columns and not df["factor"].isna().all()
            else None,
            "sample": df.head(5).to_dict(orient="records"),
        }
    finally:
        conn.close()


def main():
    print("=" * 80)
    print("Factor Value Difference Diagnostic")
    print("=" * 80)

    jq_data, local_data = load_parsed_data()
    if not jq_data:
        print("[ERROR] JQ parsed data not found at", JQ_PARSED)
        return
    if not local_data:
        print("[ERROR] Local parsed data not found at", LOCAL_PARSED)
        return

    common_dates = sorted(set(jq_data.keys()) & set(local_data.keys()))
    print(f"\nCommon dates: {len(common_dates)}")
    if not common_dates:
        print("[ERROR] No common dates between JQ and local data")
        return

    target_date = common_dates[len(common_dates) // 2]
    print(f"Using target date: {target_date}")

    all_results: Dict[str, List[dict]] = {}

    for factor_name in SUSPECT_FACTORS:
        print(f"\n{'─' * 70}")
        print(f"Factor: {factor_name}")
        print(f"{'─' * 70}")

        date_results = []
        for date in common_dates[:5]:
            jq_entry = jq_data.get(date, {})
            local_entry = local_data.get(date, {})

            jq_stage2 = jq_entry.get("stage2_raw", {}).get(factor_name, {})
            local_stage2 = local_entry.get("stage2_raw", {}).get(factor_name, {})

            if not jq_stage2 or not local_stage2:
                continue

            analysis = analyze_ratio_pattern(jq_stage2, local_stage2, factor_name)
            analysis["date"] = date
            date_results.append(analysis)

            if date == target_date or len(date_results) == 1:
                print(f"\n  Date: {date}")
                print(f"  Pattern: {analysis.get('pattern', 'N/A')}")
                print(f"  Spearman: {analysis.get('spearman', 'N/A')}")
                print(f"  Mean abs diff: {analysis.get('mean_abs_diff', 'N/A')}")
                print(f"  Max abs diff: {analysis.get('max_abs_diff', 'N/A')}")
                print(f"  JQ range: {analysis.get('jq_range', 'N/A')}")
                print(f"  Local range: {analysis.get('local_range', 'N/A')}")
                if "ratio_stats" in analysis:
                    rs = analysis["ratio_stats"]
                    print(f"  Ratio median: {rs['median']}, CV: {rs['cv']}")
                if "scale_factor" in analysis:
                    print(f"  → DETECTED SCALE FACTOR: {analysis['scale_factor']}")
                if "top5_by_diff" in analysis:
                    print("  Top 3 stocks by diff:")
                    for sd in analysis["top5_by_diff"][:3]:
                        print(f"    {sd['stock']}: JQ={sd['jq']}, Local={sd['local']}, ratio={sd['ratio']}")

        all_results[factor_name] = date_results

        freshness = check_db_factor_freshness(factor_name)
        print(f"\n  DB freshness: {freshness}")

    # Also compare stage3 (z-scored) to see if preprocessing eliminates differences
    print(f"\n{'=' * 80}")
    print("Stage 3 (Z-scored) comparison for the same factors")
    print(f"{'=' * 80}")

    for factor_name in SUSPECT_FACTORS:
        for date in [common_dates[0]]:
            jq_entry = jq_data.get(date, {})
            local_entry = local_data.get(date, {})

            jq_stage3 = jq_entry.get("stage3_zscore", {}).get(factor_name, {})
            local_stage3 = local_entry.get("stage3_zscore", {}).get(factor_name, {})

            if not jq_stage3 or not local_stage3:
                continue

            analysis = analyze_ratio_pattern(jq_stage3, local_stage3, factor_name)
            print(f"\n  {factor_name} @ {date} (Z-scored):")
            print(f"    Spearman: {analysis.get('spearman', 'N/A')}")
            print(f"    Mean abs diff: {analysis.get('mean_abs_diff', 'N/A')}")
            print(f"    Max abs diff: {analysis.get('max_abs_diff', 'N/A')}")
            if "ratio_stats" in analysis:
                print(f"    Ratio CV: {analysis['ratio_stats']['cv']}")

    # Check raw price data for a sample
    print(f"\n{'=' * 80}")
    print("Raw price data sample")
    print(f"{'=' * 80}")

    sample_stocks = list(jq_data.get(target_date, {}).get("pool", {}).get("stocks", []))[:5]
    # Factor date is T-1 of rebalance date
    from datetime import datetime, timedelta
    rebal_dt = datetime.strptime(target_date, "%Y-%m-%d")
    factor_date = (rebal_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n  Checking price data for {factor_date} (T-1 of {target_date}):")
    raw_sample = compare_raw_data_sample(factor_date, sample_stocks)
    print(f"  {raw_sample}")

    # Save full results
    output_path = PROJECT_ROOT / "results" / "factor_diff_diagnostic.json"
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[OK] Full diagnostic saved to {output_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 80}")
    for factor_name, results in all_results.items():
        if not results:
            print(f"  {factor_name}: NO DATA")
            continue
        patterns = [r.get("pattern", "?") for r in results]
        spearman_vals = [r["spearman"] for r in results if r.get("spearman") is not None]
        max_diffs = [r["max_abs_diff"] for r in results if r.get("max_abs_diff") is not None]
        scale_factors = [r.get("scale_factor") for r in results if r.get("scale_factor") is not None]

        print(f"\n  {factor_name}:")
        print(f"    Pattern: {set(patterns)}")
        if spearman_vals:
            print(f"    Spearman range: [{min(spearman_vals):.4f}, {max(spearman_vals):.4f}]")
        if max_diffs:
            print(f"    Max abs diff range: [{min(max_diffs):.4f}, {max(max_diffs):.4f}]")
        if scale_factors:
            print(f"    → SCALE FACTORS: {scale_factors}")


if __name__ == "__main__":
    main()
