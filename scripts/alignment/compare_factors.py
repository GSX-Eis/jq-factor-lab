#!/usr/bin/env python3
"""Compare cloud values with a frozen CSV or an exact output artifact binding.

Run from the execution repository, for example:
    python scripts/alignment/compare_factors.py --factor F \
        --cloud-csv cloud.csv --local-csv frozen_local.csv
    python scripts/alignment/compare_factors.py --factor F \
        --cloud-csv cloud.csv --data-bindings bindings.json --local-factor F_pool
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_cloud_data(csv_path: str, factor_name: str) -> pd.DataFrame:
    """Load cloud factor data from CSV dump."""
    df = pd.read_csv(csv_path, on_bad_lines="error")
    factor_df = df[df["factor"] == factor_name].copy()
    if factor_df.empty:
        available = sorted(df["factor"].unique().tolist())
        print(f"[ERROR] Factor '{factor_name}' not found in cloud CSV.")
        print(f"  Available: {available}")
    return factor_df


def load_local_data(db_path: str | None, factor_name: str, *, bindings=None,
                    local_factor_name: str | None = None) -> pd.DataFrame:
    """Read only the frozen output version; never guess a shared result version."""
    from utils.data_store import binding_contract, connect_output

    contract = binding_contract(bindings)
    output = contract.get("output_db")
    if not output or not contract.get("outputs", {}).get("factor_values"):
        raise ValueError("Supply frozen data bindings with output_db and outputs.factor_values, or --local-csv")
    if db_path and Path(db_path).resolve() != Path(output).resolve():
        raise ValueError("--db must match the frozen output_db binding")
    names = [local_factor_name] if local_factor_name else [
        factor_name + suffix for suffix in ("_small_cap_percentile_10", "_small_cap_9_rules", "")
    ]
    with connect_output(output, bindings=contract) as conn:
        found = conn.execute(
            "SELECT DISTINCT factor_name FROM factor_values WHERE factor_name IN ("
            + ",".join("?" for _ in names) + ")", names,
        ).fetchall()
        if len(found) > 1:
            raise ValueError("Multiple pool factors match; specify --local-factor with the exact name")
        name = found[0][0] if found else names[0]
        frame = conn.execute(
            "SELECT CAST(date AS DATE)::VARCHAR AS date, stock_code AS code, factor_value AS value "
            "FROM factor_values WHERE factor_name = ? ORDER BY date, stock_code", [name],
        ).fetchdf()
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("Bound factor contains duplicate date/code rows")
    return frame


def compare_single_factor(
    cloud_df: pd.DataFrame,
    local_df: pd.DataFrame,
    factor_name: str,
) -> Dict:
    """Compare cloud vs local for a single factor."""
    result = {
        "factor": factor_name,
        "cloud_rows": len(cloud_df),
        "local_rows": len(local_df),
        "status": "UNKNOWN",
        "daily_stats": [],
        "summary": {},
    }

    if cloud_df.empty or local_df.empty:
        result["status"] = "NO_DATA"
        return result

    cloud_df = cloud_df.copy()
    local_df = local_df.copy()
    for frame in (cloud_df, local_df):
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        if frame.duplicated(["date", "code"]).any():
            result["status"] = "INVALID_DATA"
            return result
    paired = cloud_df[["date", "code", "value"]].merge(
        local_df[["date", "code", "value"]],
        on=["date", "code"],
        how="left",
        indicator=True,
        suffixes=("_cloud", "_local"),
    )
    coverage = paired["_merge"].eq("both")
    missingness = paired.value_cloud.isna() == paired.value_local.isna()
    numerical = np.isclose(
        paired.value_cloud, paired.value_local, rtol=1e-6, atol=1e-6, equal_nan=True
    )
    result["coverage_match"] = bool(len(paired) and coverage.all())
    result["missingness_match"] = bool(
        len(paired) and coverage.all() and missingness.all()
    )
    result["numerical_match"] = bool(len(paired) and coverage.all() and numerical.all())
    differences = paired.loc[~(coverage & missingness & numerical)].sort_values(
        ["date", "code"], kind="mergesort"
    )
    if not differences.empty:
        first = differences.iloc[0]
        result["first_difference"] = {
            "date": str(first.date),
            "code": str(first.code),
            "stage": "factor",
            "reason": "missing_row"
            if first["_merge"] != "both"
            else "missing_value"
            if pd.isna(first.value_cloud) != pd.isna(first.value_local)
            else "numerical_mismatch",
        }
    result["tolerance"] = {"rtol": 1e-6, "atol": 1e-6}
    valid_cloud = cloud_df.dropna(subset=["value"])
    valid_local = local_df.dropna(subset=["value"])
    result["cloud_valid_rows"] = len(valid_cloud)
    result["local_valid_rows"] = len(valid_local)

    # Merge on date + code
    merged = pd.merge(
        valid_cloud[["date", "code", "value"]],
        valid_local[["date", "code", "value"]],
        on=["date", "code"],
        how="inner",
        suffixes=("_cloud", "_local"),
    )

    result["merged_rows"] = len(merged)

    # Only non-null factor observations belong in the overlap gate.  The cloud
    # dump also contains NaN placeholder rows for every stock in the pool.
    cloud_dates = pd.to_datetime(valid_cloud["date"], errors="coerce")
    local_dates = pd.to_datetime(valid_local["date"], errors="coerce")
    lmin, lmax = local_dates.min(), local_dates.max()
    if pd.notna(lmin) and pd.notna(lmax):
        in_range = (cloud_dates >= lmin) & (cloud_dates <= lmax)
        result["cloud_rows_in_local_range"] = int(in_range.sum())
        result["local_date_min"] = str(lmin.date())
        result["local_date_max"] = str(lmax.date())
    else:
        result["cloud_rows_in_local_range"] = int(len(valid_cloud))
        result["local_date_min"] = None
        result["local_date_max"] = None
    _civ = result["cloud_rows_in_local_range"]
    result["stock_pool_overlap"] = (len(merged) / _civ) if _civ else None

    if merged.empty:
        result["status"] = "NO_OVERLAP"
        return result

    # Per-date statistics
    daily_stats = []
    for date, group in merged.groupby("date"):
        group = group.dropna(subset=["value_cloud", "value_local"])
        if len(group) < 5:
            continue

        cv = group["value_cloud"].values
        lv = group["value_local"].values
        abs_diff = np.abs(cv - lv)

        # [ALIGNMENT-FIX] Rank on the stated 1e-6 alignment precision.  The
        # factor is frequently exactly zero, and sub-1e-6 float noise would
        # otherwise split tied ranks despite every value being an exact match
        # under the validation tolerance.  Keep abs_diff above unrounded as a
        # diagnostic indicator.
        rounded_cv = np.round(cv, 6)
        rounded_lv = np.round(lv, 6)
        if np.all(abs_diff < 1e-6):
            spearman = 1.0
        elif np.array_equal(rounded_cv, rounded_lv) and (
            np.ptp(rounded_cv) == 0 or np.ptp(rounded_lv) == 0
        ):
            # [ALIGNMENT-FIX] Identical constant cross-sections are aligned;
            # scipy returns NaN because rank correlation is formally undefined.
            spearman = 1.0
        else:
            spearman, p_val = stats.spearmanr(rounded_cv, rounded_lv)

        daily_stats.append(
            {
                "date": date,
                "n_stocks": len(group),
                "spearman": round(spearman, 6),
                "mean_abs_diff": round(float(np.mean(abs_diff)), 8),
                "max_abs_diff": round(float(np.max(abs_diff)), 8),
                "exact_match_1e6": round(float((abs_diff < 1e-6).mean()), 4),
                "exact_match_1e4": round(float((abs_diff < 1e-4).mean()), 4),
            }
        )

    result["daily_stats"] = daily_stats

    if not daily_stats:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    valid_daily_stats = [
        item for item in daily_stats if np.isfinite(item.get("spearman", np.nan))
    ]
    if valid_daily_stats:
        worst = min(
            valid_daily_stats,
            key=lambda item: (item["spearman"], -item["max_abs_diff"]),
        )
        worst_date = str(worst["date"])
        worst_rows = merged[merged["date"].astype(str) == worst_date].copy()
        worst_rows["abs_diff"] = np.abs(
            worst_rows["value_cloud"] - worst_rows["value_local"]
        )
        largest = worst_rows.nlargest(10, "abs_diff")
        result["worst_date_diagnosis"] = {
            **worst,
            "date": worst_date,
            "largest_stock_differences": [
                {
                    "code": str(row.code),
                    "cloud": float(row.value_cloud),
                    "local": float(row.value_local),
                    "abs_diff": float(row.abs_diff),
                }
                for row in largest.itertuples()
            ],
        }

    # Summary
    df_stats = pd.DataFrame(daily_stats)
    avg_spearman = df_stats["spearman"].mean()
    min_spearman = df_stats["spearman"].min()
    result["summary"] = {
        "avg_spearman": round(avg_spearman, 6),
        "min_spearman": round(min_spearman, 6),
        "median_spearman": round(float(df_stats["spearman"].median()), 6),
        "avg_mean_abs_diff": round(float(df_stats["mean_abs_diff"].mean()), 8),
        "max_max_abs_diff": round(float(df_stats["max_abs_diff"].max()), 8),
        "n_dates": len(daily_stats),
        "n_dates_spearman_lt_099": int((df_stats["spearman"] < 0.99).sum()),
        "n_dates_spearman_lt_095": int((df_stats["spearman"] < 0.95).sum()),
    }

    # Determine status
    if (
        avg_spearman > 0.99
        and min_spearman > 0.95
        and result["numerical_match"]
        and result["missingness_match"]
    ):
        result["status"] = "PASS"
    elif (
        result["numerical_match"]
        and result["missingness_match"]
        and min_spearman >= 0.90
    ):
        result["status"] = "WARN"
    else:
        result["status"] = "FAIL"

    return result


def generate_report(results: List[Dict], output_dir: str) -> str:
    """Generate markdown comparison report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Factor Alignment Comparison Report",
        "",
        f"> Generated: {now}",
        "",
        "## Summary",
        "",
        "| Factor | Status | Avg Spearman | Min Spearman | Overlap (in-range) | Dates |",
        "|--------|--------|-------------|-------------|--------------------|-------|",
    ]

    for r in results:
        s = r.get("summary", {})
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(r["status"], "❓")
        ov = r.get("stock_pool_overlap")
        ov_str = f"{ov:.4f}" if isinstance(ov, (int, float)) else "N/A"
        lines.append(
            f"| `{r['factor']}` | {icon} {r['status']} | "
            f"{s.get('avg_spearman', 'N/A')} | {s.get('min_spearman', 'N/A')} | "
            f"{ov_str} | {s.get('n_dates', 0)} |"
        )

    # Per-factor details
    for r in results:
        if not r.get("daily_stats"):
            continue

        lines.extend(
            [
                "",
                "---",
                "",
                f"## {r['factor']} ({r['status']})",
                "",
            ]
        )

        # Worst dates
        df_daily = pd.DataFrame(r["daily_stats"])
        worst = df_daily.nsmallest(10, "spearman")

        lines.extend(
            [
                "### Worst Dates (by Spearman)",
                "",
                "| Date | Stocks | Spearman | Mean Diff | Max Diff |",
                "|------|--------|----------|-----------|----------|",
            ]
        )
        for _, row in worst.iterrows():
            lines.append(
                f"| {row['date']} | {row['n_stocks']} | {row['spearman']:.4f} | "
                f"{row['mean_abs_diff']:.2e} | {row['max_abs_diff']:.2e} |"
            )

    # Decision tree
    lines.extend(
        [
            "",
            "---",
            "",
            "## Next Steps",
            "",
            "```",
            "IF status == PASS (numeric, missingness, rank and coverage gates):",
            "    → Retain evidence; independent input, strategy and publication gates still apply",
            "",
            "IF status == WARN (min Spearman 0.90~0.95):",
            "    → Check floating point precision (Numba JIT vs Python)",
            "    → Check NaN handling differences",
            "    → May be acceptable if Top-N overlap is high",
            "",
            "IF status == FAIL (min Spearman < 0.90):",
            "    → Run find_max_diff.py to identify worst date",
            "    → Add debug logs (see references/log_templates.md)",
            "    → Re-run Phase 2-4 with debug logs",
            "    → Compare logs to find root cause",
            "```",
        ]
    )

    report_path = output_path / "alignment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Report: {report_path}")

    # Save JSON
    json_path = output_path / "alignment_results.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[OK] JSON: {json_path}")

    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Compare cloud vs local factor values")
    parser.add_argument(
        "--factor", required=True, help="Factor name(s), comma-separated"
    )
    parser.add_argument("--cloud-csv", required=True, help="Cloud factor dump CSV path")
    parser.add_argument(
        "--db", help="Optional output path; must match frozen bindings"
    )
    parser.add_argument(
        "--output", default="outputs/alignment/report/", help="Output dir"
    )

    parser.add_argument(
        "--local-csv", help="Frozen local computation output with date,code,value columns"
    )
    parser.add_argument("--data-bindings", help="Harness frozen data bindings JSON")
    parser.add_argument("--local-factor", help="Exact local factor name including pool suffix")
    args = parser.parse_args()
    if args.local_csv and (args.db or args.data_bindings or args.local_factor):
        parser.error("--local-csv cannot be combined with database selection options")
    bindings = json.loads(Path(args.data_bindings).read_text()) if args.data_bindings else None
    factor_names = [f.strip() for f in args.factor.split(",")]
    if (args.local_csv or args.local_factor) and len(factor_names) != 1:
        parser.error("--local-csv / --local-factor requires one factor")

    all_results = []
    for factor_name in factor_names:
        print(f"\n{'=' * 60}")
        print(f"Comparing: {factor_name}")
        print(f"{'=' * 60}")

        cloud_df = load_cloud_data(args.cloud_csv, factor_name)
        local_df = (
            pd.read_csv(args.local_csv)
            if args.local_csv
            else load_local_data(args.db, factor_name, bindings=bindings, local_factor_name=args.local_factor)
        )
        Path(args.output).mkdir(parents=True, exist_ok=True)
        local_df.to_csv(
            Path(args.output) / (factor_name + "_local_snapshot.csv"), index=False
        )
        result = compare_single_factor(cloud_df, local_df, factor_name)
        all_results.append(result)

        # Print summary
        s = result.get("summary", {})
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(result["status"], "❓")
        print(f"\n  {icon} {result['status']}")
        if s:
            print(f"  Avg Spearman: {s.get('avg_spearman', 'N/A')}")
            print(f"  Min Spearman: {s.get('min_spearman', 'N/A')}")
            print(f"  Dates < 0.99: {s.get('n_dates_spearman_lt_099', 0)}")

    report_path = generate_report(all_results, args.output)

    print(f"\n{'=' * 60}")
    print("OVERALL RESULT")
    print(f"{'=' * 60}")
    for r in all_results:
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(r["status"], "❓")
        print(f"  {r['factor']}: {icon} {r['status']}")
    print(f"\nReport: {report_path}")
    return 0 if all(r["status"] == "PASS" for r in all_results) else 4


if __name__ == "__main__":
    raise SystemExit(main())
