#!/usr/bin/env python3
"""
Multi-level comparison of JQ cloud vs local VectorBT backtest results.

Compares at 5 levels:
  Level 1: Stock Pool - pool composition overlap per rebalance date
  Level 2: Factor Values - raw factor value correlation (ALIGN_STAGE2 / FULL_RANK)
  Level 3: Preprocessed Values - zscore/rank_zscore correlation (ALIGN_STAGE3)
  Level 4: Composite Scores - final synthesis score correlation (ALIGN_STAGE4 / FINAL_SCORE)
  Level 5: Trading Decisions - target stock overlap, buy/sell agreement

Usage:
    python compare_backtest_alignment.py \
        --jq results/jq_parsed.json \
        --local results/local_parsed.json \
        --output results/backtest_alignment_report.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _overlap_pct(set_a: set, set_b: set) -> float:
    """Overlap percentage: |intersection| / max(|a|, |b|)."""
    if not set_a and not set_b:
        return 1.0
    denom = max(len(set_a), len(set_b))
    if denom == 0:
        return 1.0
    return len(set_a & set_b) / denom


def _spearman(x_dict: dict[str, float], y_dict: dict[str, float]) -> Optional[float]:
    """Compute Spearman rank correlation between two dicts keyed by stock."""
    common = sorted(set(x_dict.keys()) & set(y_dict.keys()))
    if len(common) < 3:
        return None
    x_vals = np.array([x_dict[k] for k in common])
    y_vals = np.array([y_dict[k] for k in common])
    # Rank
    x_ranks = np.argsort(np.argsort(-x_vals)).astype(float) + 1
    y_ranks = np.argsort(np.argsort(-y_vals)).astype(float) + 1
    n = len(common)
    d_sq = np.sum((x_ranks - y_ranks) ** 2)
    rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
    return float(rho)


def _max_abs_diff(
    x_dict: dict[str, float], y_dict: dict[str, float]
) -> Optional[float]:
    """Max absolute difference between common keys."""
    common = set(x_dict.keys()) & set(y_dict.keys())
    if not common:
        return None
    diffs = [abs(x_dict[k] - y_dict[k]) for k in common]
    return max(diffs)


def compare_date(jq_data: dict, local_data: dict, date: str) -> dict[str, Any]:
    """Compare a single rebalance date across all levels."""
    result: dict[str, Any] = {"date": date, "levels": {}}

    # Level 1: Stock Pool
    jq_pool = set(jq_data.get("pool", {}).get("stocks", []))
    local_pool = set(local_data.get("pool", {}).get("stocks", []))
    pool_result = {
        "jq_size": len(jq_pool),
        "local_size": len(local_pool),
        "overlap": len(jq_pool & local_pool),
        "overlap_pct": round(_overlap_pct(jq_pool, local_pool), 4),
        "jaccard": round(_jaccard(jq_pool, local_pool), 4),
        "jq_only": sorted(jq_pool - local_pool)[:10],
        "local_only": sorted(local_pool - jq_pool)[:10],
    }
    pool_result["status"] = (
        "PASS"
        if pool_result["overlap_pct"] >= 0.95
        else ("WARN" if pool_result["overlap_pct"] >= 0.80 else "FAIL")
    )
    result["levels"]["L1_pool"] = pool_result

    # Level 2: Raw Factor Values (stage2_raw)
    jq_raw = jq_data.get("stage2_raw", {})
    local_raw = local_data.get("stage2_raw", {})
    common_factors = sorted(set(jq_raw.keys()) & set(local_raw.keys()))
    factor_correlations = {}
    for fname in common_factors:
        rho = _spearman(jq_raw[fname], local_raw[fname])
        mad = _max_abs_diff(jq_raw[fname], local_raw[fname])
        n_common = len(set(jq_raw[fname].keys()) & set(local_raw[fname].keys()))
        factor_correlations[fname] = {
            "spearman": round(rho, 6) if rho is not None else None,
            "max_abs_diff": round(mad, 8) if mad is not None else None,
            "n_common_stocks": n_common,
        }
    avg_rho = (
        np.mean(
            [
                v["spearman"]
                for v in factor_correlations.values()
                if v["spearman"] is not None
            ]
        )
        if factor_correlations
        else None
    )
    raw_result = {
        "n_common_factors": len(common_factors),
        "jq_factors": sorted(jq_raw.keys()),
        "local_factors": sorted(local_raw.keys()),
        "per_factor": factor_correlations,
        "avg_spearman": round(float(avg_rho), 6) if avg_rho is not None else None,
    }
    if avg_rho is not None:
        raw_result["status"] = (
            "PASS" if avg_rho >= 0.99 else ("WARN" if avg_rho >= 0.95 else "FAIL")
        )
    else:
        raw_result["status"] = "NO_DATA"
    result["levels"]["L2_raw_factors"] = raw_result

    # Level 3: Preprocessed Values (stage3_zscore)
    jq_zs = jq_data.get("stage3_zscore", {})
    local_zs = local_data.get("stage3_zscore", {})
    common_zs_factors = sorted(set(jq_zs.keys()) & set(local_zs.keys()))
    zs_correlations = {}
    for fname in common_zs_factors:
        rho = _spearman(jq_zs[fname], local_zs[fname])
        mad = _max_abs_diff(jq_zs[fname], local_zs[fname])
        n_common = len(set(jq_zs[fname].keys()) & set(local_zs[fname].keys()))
        zs_correlations[fname] = {
            "spearman": round(rho, 6) if rho is not None else None,
            "max_abs_diff": round(mad, 8) if mad is not None else None,
            "n_common_stocks": n_common,
        }
    avg_zs_rho = (
        np.mean(
            [
                v["spearman"]
                for v in zs_correlations.values()
                if v["spearman"] is not None
            ]
        )
        if zs_correlations
        else None
    )
    zs_result = {
        "n_common_factors": len(common_zs_factors),
        "per_factor": zs_correlations,
        "avg_spearman": round(float(avg_zs_rho), 6) if avg_zs_rho is not None else None,
    }
    if avg_zs_rho is not None:
        zs_result["status"] = (
            "PASS" if avg_zs_rho >= 0.99 else ("WARN" if avg_zs_rho >= 0.95 else "FAIL")
        )
    else:
        zs_result["status"] = "NO_DATA"
    result["levels"]["L3_preprocessed"] = zs_result

    # Level 4: Composite Scores (stage4_composite)
    jq_scores = {s: v["score"] for s, v in jq_data.get("stage4_composite", {}).items()}
    local_scores = {
        s: v["score"] for s, v in local_data.get("stage4_composite", {}).items()
    }
    # Also try top10_final if stage4 is sparse
    if not jq_scores and jq_data.get("top10_final"):
        jq_scores = {s: v for s, v in jq_data["top10_final"]}
    if not local_scores and local_data.get("top10_final"):
        local_scores = {s: v for s, v in local_data["top10_final"]}

    composite_rho = _spearman(jq_scores, local_scores)
    composite_mad = _max_abs_diff(jq_scores, local_scores)
    n_common_scored = len(set(jq_scores.keys()) & set(local_scores.keys()))

    # Top-K overlap
    jq_top10 = set(list(jq_scores.keys())[:10]) if jq_scores else set()
    local_top10 = set(list(local_scores.keys())[:10]) if local_scores else set()
    top10_overlap = len(jq_top10 & local_top10)

    # Rank match: compare explicit rank values from stage4_composite
    jq_ranks_raw = {
        s: v["rank"]
        for s, v in jq_data.get("stage4_composite", {}).items()
        if isinstance(v, dict) and "rank" in v
    }
    local_ranks_raw = {
        s: v["rank"]
        for s, v in local_data.get("stage4_composite", {}).items()
        if isinstance(v, dict) and "rank" in v
    }
    common_ranked = set(jq_ranks_raw.keys()) & set(local_ranks_raw.keys())
    rank_match_count = sum(
        1 for s in common_ranked if jq_ranks_raw[s] == local_ranks_raw[s]
    )
    rank_match_pct = (
        round(rank_match_count / len(common_ranked), 4) if common_ranked else None
    )

    # Dynamic top-K overlap based on target_count
    jq_target_count = jq_data.get("rebalance", {}).get("target_count", 0)
    local_target_count = local_data.get("rebalance", {}).get("target_count", 0)
    top_k = max(jq_target_count, local_target_count, 1)
    jq_top_k = {s for s, v in sorted(jq_ranks_raw.items(), key=lambda x: x[1])[:top_k]}
    local_top_k = {
        s for s, v in sorted(local_ranks_raw.items(), key=lambda x: x[1])[:top_k]
    }
    top_k_overlap = len(jq_top_k & local_top_k)
    top_k_overlap_pct = round(top_k_overlap / top_k, 4) if top_k > 0 else None

    composite_result = {
        "jq_n_scored": len(jq_scores),
        "local_n_scored": len(local_scores),
        "n_common": n_common_scored,
        "spearman": round(composite_rho, 6) if composite_rho is not None else None,
        "max_abs_diff": round(composite_mad, 8) if composite_mad is not None else None,
        "top10_overlap": top10_overlap,
        "top10_overlap_pct": round(top10_overlap / 10, 2)
        if jq_top10 or local_top10
        else None,
        "rank_match_pct": rank_match_pct,
        "n_common_ranked": len(common_ranked),
        "top_k": top_k,
        "top_k_overlap": top_k_overlap,
        "top_k_overlap_pct": top_k_overlap_pct,
    }
    if composite_rho is not None:
        composite_result["status"] = (
            "PASS"
            if composite_rho >= 0.99
            else ("WARN" if composite_rho >= 0.95 else "FAIL")
        )
    else:
        composite_result["status"] = "NO_DATA"
    result["levels"]["L4_composite"] = composite_result

    # Level 5: Trading Decisions - target holdings overlap + buy/sell agreement
    jq_target = set(jq_data.get("rebalance", {}).get("top3", []))
    jq_buys = set(jq_data.get("trades", {}).get("buy", []))
    jq_sells = set(jq_data.get("trades", {}).get("sell", []))
    jq_holds = set(jq_data.get("trades", {}).get("hold", []))
    local_target = set(local_data.get("rebalance", {}).get("top3", []))
    local_buys = set(local_data.get("trades", {}).get("buy", []))
    local_sells = set(local_data.get("trades", {}).get("sell", []))
    local_holds = set(local_data.get("trades", {}).get("hold", []))

    # Full target holdings: buys + holds = all stocks held after rebalance
    jq_all_holdings = jq_buys | jq_holds if (jq_buys or jq_holds) else jq_target
    local_all_holdings = (
        local_buys | local_holds if (local_buys or local_holds) else local_target
    )

    # Target overlap
    target_overlap = len(jq_target & local_target)
    target_denom = max(len(jq_target), len(local_target))
    target_overlap_pct = (
        round(target_overlap / target_denom, 4) if target_denom > 0 else None
    )

    # Holdings overlap (buys + holds)
    holdings_overlap = len(jq_all_holdings & local_all_holdings)
    holdings_denom = max(len(jq_all_holdings), len(local_all_holdings))
    holdings_overlap_pct = (
        round(holdings_overlap / holdings_denom, 4) if holdings_denom > 0 else None
    )

    # Buy/sell agreement
    buy_overlap = len(jq_buys & local_buys)
    sell_overlap = len(jq_sells & local_sells)

    trade_result = {
        "jq_target_count": jq_data.get("rebalance", {}).get("target_count", 0),
        "local_target_count": local_data.get("rebalance", {}).get("target_count", 0),
        "target_overlap": target_overlap,
        "target_overlap_pct": target_overlap_pct,
        "target_jq_only": sorted(jq_target - local_target)[:10],
        "target_local_only": sorted(local_target - jq_target)[:10],
        "jq_holdings": len(jq_all_holdings),
        "local_holdings": len(local_all_holdings),
        "holdings_overlap": holdings_overlap,
        "holdings_overlap_pct": holdings_overlap_pct,
        "holdings_jq_only": sorted(jq_all_holdings - local_all_holdings)[:10],
        "holdings_local_only": sorted(local_all_holdings - jq_all_holdings)[:10],
        "jq_buys": len(jq_buys),
        "local_buys": len(local_buys),
        "buy_overlap": buy_overlap,
        "jq_sells": len(jq_sells),
        "local_sells": len(local_sells),
        "sell_overlap": sell_overlap,
    }
    # Status: based on holdings overlap (most actionable metric)
    if holdings_overlap_pct is not None:
        trade_result["status"] = (
            "PASS"
            if holdings_overlap_pct >= 1.0
            else ("WARN" if holdings_overlap_pct >= 0.80 else "FAIL")
        )
    elif target_overlap_pct is not None:
        trade_result["status"] = (
            "PASS"
            if target_overlap_pct >= 1.0
            else ("WARN" if target_overlap_pct >= 0.80 else "FAIL")
        )
    else:
        trade_result["status"] = "NO_DATA"
    result["levels"]["L5_trades"] = trade_result

    # Weights comparison
    jq_weights = jq_data.get("weights", {})
    local_weights = local_data.get("weights", {})
    if jq_weights and local_weights:
        common_w = set(jq_weights.keys()) & set(local_weights.keys())
        weight_diffs = {f: abs(jq_weights[f] - local_weights[f]) for f in common_w}
        max_w_diff = max(weight_diffs.values()) if weight_diffs else 0
        result["weights"] = {
            "n_common": len(common_w),
            "jq_only": sorted(set(jq_weights.keys()) - set(local_weights.keys())),
            "local_only": sorted(set(local_weights.keys()) - set(jq_weights.keys())),
            "max_weight_diff": round(max_w_diff, 8),
            "weights_match": max_w_diff < 1e-6,
        }

    return result


def compare_all(jq_data: dict, local_data: dict) -> dict[str, Any]:
    """Compare all common rebalance dates."""
    jq_dates = set(jq_data.keys())
    local_dates = set(local_data.keys())
    common_dates = sorted(jq_dates & local_dates)

    report = {
        "summary": {
            "jq_dates": len(jq_dates),
            "local_dates": len(local_dates),
            "common_dates": len(common_dates),
            "jq_only_dates": sorted(jq_dates - local_dates)[:10],
            "local_only_dates": sorted(local_dates - jq_dates)[:10],
        },
        "per_date": {},
        "aggregate": {},
    }

    level_statuses: dict[str, list[str]] = {
        "L1_pool": [],
        "L2_raw_factors": [],
        "L3_preprocessed": [],
        "L4_composite": [],
        "L5_trades": [],
    }

    for date in common_dates:
        date_result = compare_date(jq_data[date], local_data[date], date)
        report["per_date"][date] = date_result
        for level_name, level_data in date_result["levels"].items():
            if "status" in level_data:
                level_statuses[level_name].append(level_data["status"])

    # Aggregate
    for level_name, statuses in level_statuses.items():
        if not statuses:
            report["aggregate"][level_name] = "NO_DATA"
            continue
        fail_count = statuses.count("FAIL")
        warn_count = statuses.count("WARN")
        pass_count = statuses.count("PASS")
        total = len(statuses)
        report["aggregate"][level_name] = {
            "PASS": pass_count,
            "WARN": warn_count,
            "FAIL": fail_count,
            "NO_DATA": statuses.count("NO_DATA"),
            "total": total,
            "overall": "PASS"
            if fail_count == 0 and warn_count == 0
            else ("WARN" if fail_count == 0 else "FAIL"),
        }

    return report


def print_report(report: dict):
    """Print formatted comparison report."""
    s = report["summary"]
    print(f"\n{'=' * 70}")
    print(f"  Backtest Alignment Report")
    print(f"{'=' * 70}")
    print(
        f"  JQ dates: {s['jq_dates']}  |  Local dates: {s['local_dates']}  |  Common: {s['common_dates']}"
    )

    if s["jq_only_dates"]:
        print(f"  JQ-only dates: {', '.join(s['jq_only_dates'][:5])}")
    if s["local_only_dates"]:
        print(f"  Local-only dates: {', '.join(s['local_only_dates'][:5])}")

    print(f"\n{'─' * 70}")
    print(f"  Aggregate Results")
    print(f"{'─' * 70}")

    level_labels = {
        "L1_pool": "L1 Stock Pool",
        "L2_raw_factors": "L2 Raw Factors",
        "L3_preprocessed": "L3 Preprocessed",
        "L4_composite": "L4 Composite Score",
        "L5_trades": "L5 Trades",
    }

    for level_key, label in level_labels.items():
        agg = report["aggregate"].get(level_key, "NO_DATA")
        if isinstance(agg, dict):
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(agg["overall"], "❓")
            print(
                f"  {icon} {label}: {agg['overall']} "
                f"(P={agg['PASS']} W={agg['WARN']} F={agg['FAIL']} / {agg['total']})"
            )
        else:
            print(f"  ❓ {label}: {agg}")

    # Per-date details for FAIL/WARN dates
    print(f"\n{'─' * 70}")
    print(f"  Date Details (FAIL/WARN only)")
    print(f"{'─' * 70}")

    shown = 0
    for date, date_result in sorted(report["per_date"].items()):
        has_issue = any(
            level.get("status") in ("FAIL", "WARN")
            for level in date_result["levels"].values()
        )
        if not has_issue:
            continue
        shown += 1
        if shown > 10:
            print(f"  ... (more dates with issues)")
            break

        print(f"\n  📅 {date}")
        for level_key, level_data in date_result["levels"].items():
            status = level_data.get("status", "?")
            if status in ("FAIL", "WARN"):
                icon = "❌" if status == "FAIL" else "⚠️"
                detail = ""
                if "spearman" in level_data and level_data["spearman"] is not None:
                    detail += f" Spearman={level_data['spearman']:.4f}"
                if "overlap_pct" in level_data:
                    detail += f" Overlap={level_data['overlap_pct']:.2%}"
                if (
                    "max_abs_diff" in level_data
                    and level_data["max_abs_diff"] is not None
                ):
                    detail += f" MaxDiff={level_data['max_abs_diff']:.6f}"
                if "top10_overlap" in level_data:
                    detail += f" Top10={level_data['top10_overlap']}/10"
                if (
                    "rank_match_pct" in level_data
                    and level_data["rank_match_pct"] is not None
                ):
                    detail += f" RankMatch={level_data['rank_match_pct']:.2%}"
                if (
                    "top_k_overlap_pct" in level_data
                    and level_data["top_k_overlap_pct"] is not None
                ):
                    detail += (
                        f" TopK={level_data['top_k_overlap']}/{level_data['top_k']}"
                    )
                if (
                    "holdings_overlap_pct" in level_data
                    and level_data["holdings_overlap_pct"] is not None
                ):
                    detail += f" Holdings={level_data['holdings_overlap']}/{level_data['jq_holdings']}"
                if (
                    "target_overlap_pct" in level_data
                    and level_data["target_overlap_pct"] is not None
                ):
                    detail += f" Target={level_data['target_overlap_pct']:.2%}"
                print(
                    f"    {icon} {level_labels.get(level_key, level_key)}: {status}{detail}"
                )

    if shown == 0:
        print("  ✅ All dates PASS across all levels")


def main():
    parser = argparse.ArgumentParser(
        description="Compare JQ vs local backtest alignment"
    )
    parser.add_argument(
        "--jq", required=True, help="JQ parsed JSON (from parse_jq_backtest_log.py)"
    )
    parser.add_argument(
        "--local",
        required=True,
        help="Local parsed JSON (from parse_local_backtest_log.py)",
    )
    parser.add_argument(
        "--output",
        default="results/backtest_alignment_report.json",
        help="Output report path",
    )
    args = parser.parse_args()

    jq_data = json.loads(Path(args.jq).read_text(encoding="utf-8"))
    local_data = json.loads(Path(args.local).read_text(encoding="utf-8"))

    report = compare_all(jq_data, local_data)
    print_report(report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
