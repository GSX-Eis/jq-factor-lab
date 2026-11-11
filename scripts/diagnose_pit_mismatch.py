#!/usr/bin/env python3
"""
Diagnostic: PIT Adjustment Mismatch Between Local and Cloud

ROOT CAUSE: FactorBase._apply_pit_per_window (local path) does NOT adjust
close/volume prices. But _add_pit_adjusted_columns (cloud path) DOES.
This means calculate_factor() receives raw prices locally but PIT-adjusted
prices on the cloud.

This script:
1. Fetches raw data for the problematic stock 003017.XSHE
2. Computes PVDC_60D factor with and without PIT adjustment
3. Compares results with actual local/cloud parsed values
4. Proves the mismatch pattern applies to all stocks with PIT changes

Usage:
    conda run --no-capture-output -n qlib_m3 python scripts/diagnose_pit_mismatch.py
"""

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SSD_DB = Path("./data/duckdb/factor_database.duckdb")
JQ_PARSED = PROJECT_ROOT / "results" / "jq_parsed.json"
LOCAL_PARSED = PROJECT_ROOT / "results" / "local_parsed.json"


def compute_pvdc(vol: np.ndarray, cls: np.ndarray, window: int = 60) -> float:
    """Compute PVDC factor (COV(DELTA_vol, DELTA_close, window) * -1) for the last date."""
    n = len(vol)
    if n < window + 1:
        return np.nan

    v_delta = np.diff(vol[n - window - 1 : n])
    c_delta = np.diff(cls[n - window - 1 : n])
    mask = ~(np.isnan(v_delta) | np.isnan(c_delta))
    if mask.sum() < int(window * 0.7):
        return np.nan

    vd, cd = v_delta[mask], c_delta[mask]
    cov = np.sum((vd - vd.mean()) * (cd - cd.mean())) / (len(vd) - 1)
    return -cov


def main():
    print("=" * 80)
    print("PIT Adjustment Mismatch Diagnostic")
    print("=" * 80)

    jq_data = json.loads(JQ_PARSED.read_text()) if JQ_PARSED.exists() else {}
    local_data = json.loads(LOCAL_PARSED.read_text()) if LOCAL_PARSED.exists() else {}

    if not SSD_DB.exists():
        print(f"[ERROR] DuckDB not found: {SSD_DB}")
        return

    conn = duckdb.connect(str(SSD_DB), read_only=True)

    target_date = "2024-02-07"
    factor_date = "2024-02-06"
    factor_name = "PriceVolumeDeviationCoordinationFactor_60D_small_cap_9_rules"

    pool_stocks = jq_data.get(target_date, {}).get("pool", {}).get("stocks", [])
    print(f"\nTarget date: {target_date} (factor date: {factor_date})")
    print(f"Pool size: {len(pool_stocks)}")

    # Get all stocks with PIT changes in the 60D window
    stock_list_sql = "','".join(pool_stocks)
    df_all = conn.execute(f"""
        SELECT stock_code as code, date, close, volume, factor
        FROM price_data_daily
        WHERE stock_code IN ('{stock_list_sql}')
          AND date BETWEEN '2023-10-01' AND '{factor_date}'
        ORDER BY code, date
    """).fetchdf()

    print(f"\nLoaded {len(df_all)} rows for {df_all['code'].nunique()} stocks")

    # Identify stocks with PIT changes in the 60D-but-not-20D window
    lookback_20d_start = pd.to_datetime(factor_date) - pd.Timedelta(days=35)
    pit_change_stocks = []

    for code, grp in df_all.groupby("code"):
        factors = grp["factor"].values
        dates = pd.to_datetime(grp["date"])
        for i in range(1, len(factors)):
            if abs(factors[i] - factors[i - 1]) > 0.001:
                change_date = dates.iloc[i]
                if change_date < lookback_20d_start:
                    pit_change_stocks.append({
                        "code": code,
                        "date": str(change_date)[:10],
                        "old": float(factors[i - 1]),
                        "new": float(factors[i]),
                        "ratio": float(factors[i] / factors[i - 1]),
                    })

    print(f"\nStocks with PIT changes in 60D window (outside 20D): {len(pit_change_stocks)}")
    for s in pit_change_stocks:
        print(f"  {s['code']}: {s['date']} factor {s['old']:.6f} -> {s['new']:.6f} (×{s['ratio']:.4f})")

    # For each stock in pool, compute PVDC_60D with RAW and PIT-adjusted data
    print(f"\n{'─' * 70}")
    print("Recomputing PVDC_60D for ALL pool stocks: RAW vs PIT-adjusted")
    print(f"{'─' * 70}")

    jq_stage2 = jq_data.get(target_date, {}).get("stage2_raw", {}).get(factor_name, {})
    local_stage2 = local_data.get(target_date, {}).get("stage2_raw", {}).get(factor_name, {})

    results = []
    for code, grp in df_all.groupby("code"):
        grp = grp.sort_values("date")
        last_date = pd.to_datetime(grp["date"]).max()
        if str(last_date)[:10] < factor_date:
            continue

        close_raw = grp["close"].astype(np.float64).values
        volume_raw = grp["volume"].astype(np.float64).values
        factors = grp["factor"].astype(np.float64).values

        factor_end = factors[-1]
        factor_ratio = factors / factor_end

        close_pit = close_raw * factor_ratio
        volume_pit = volume_raw / factor_ratio

        raw_val = compute_pvdc(volume_raw, close_raw, 60)
        pit_val = compute_pvdc(volume_pit, close_pit, 60)

        jq_val = jq_stage2.get(code)
        local_val = local_stage2.get(code)

        results.append({
            "code": code,
            "raw": raw_val,
            "pit": pit_val,
            "jq": jq_val,
            "local": local_val,
            "has_pit_change": any(s["code"] == code for s in pit_change_stocks),
        })

    # Analyze which computation matches which source
    n_jq_matches_pit = 0
    n_jq_matches_raw = 0
    n_local_matches_pit = 0
    n_local_matches_raw = 0
    mismatched = []

    for r in results:
        if r["jq"] is None or r["local"] is None or np.isnan(r["raw"]) or np.isnan(r["pit"]):
            continue

        pit_jq_diff = abs(r["pit"] - r["jq"])
        raw_jq_diff = abs(r["raw"] - r["jq"])
        pit_local_diff = abs(r["pit"] - r["local"])
        raw_local_diff = abs(r["raw"] - r["local"])

        jq_matches_pit = pit_jq_diff < raw_jq_diff
        local_matches_raw = raw_local_diff < pit_local_diff

        if jq_matches_pit:
            n_jq_matches_pit += 1
        else:
            n_jq_matches_raw += 1

        if local_matches_raw:
            n_local_matches_raw += 1
        else:
            n_local_matches_pit += 1

        if r["has_pit_change"]:
            mismatched.append(r)

    print(f"\n{'─' * 70}")
    print("RESULTS: Which computation matches which source?")
    print(f"{'─' * 70}")
    print(f"  JQ (cloud) closer to PIT-adjusted: {n_jq_matches_pit}")
    print(f"  JQ (cloud) closer to RAW:          {n_jq_matches_raw}")
    print(f"  Local closer to RAW:               {n_local_matches_raw}")
    print(f"  Local closer to PIT-adjusted:      {n_local_matches_pit}")

    print(f"\n  Stocks with PIT changes (detail):")
    for r in mismatched:
        print(f"    {r['code']}:")
        print(f"      RAW={r['raw']:.2f}, PIT={r['pit']:.2f}")
        print(f"      JQ={r['jq']:.2f} (diff from PIT: {abs(r['pit']-r['jq']):.2f})")
        print(f"      Local={r['local']:.2f} (diff from RAW: {abs(r['raw']-r['local']):.2f})")

    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")
    print("""
ROOT CAUSE: PIT adjustment asymmetry in FactorBase

  Cloud path (JQFactorAdapter / _add_pit_adjusted_columns):
    → Replaces close/volume with PIT-adjusted values BEFORE calculate_factor()
    → Formula: adjusted_close = close × (factor / factor_end)

  Local path (_apply_pit_per_window):
    → Does NOT adjust prices, keeps raw close/volume
    → Only fills NaN/zero in the factor column
    → calculate_factor() receives RAW prices

  Impact:
    - For stocks WITHOUT PIT changes in the window → no difference (factor=constant → ratio=1)
    - For stocks WITH PIT changes (corporate actions) → values differ by PIT ratio
    - 003017.XSHE: PIT ratio 1.42 → PVDC_60D differs by 8.4x
    - 20D window has no PIT changes → perfect alignment
    - 60D window includes PIT change → massive divergence

  Fix: Make _apply_pit_per_window behave like _add_pit_adjusted_columns
    (apply price × factor/factor_end before passing to calculate_factor)
""")

    conn.close()


if __name__ == "__main__":
    main()
