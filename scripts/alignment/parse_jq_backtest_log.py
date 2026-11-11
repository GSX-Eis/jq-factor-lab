#!/usr/bin/env python3
"""
Parse JoinQuant backtest log into structured alignment data.

Extracts per-rebalance-date data:
- POOL_DUMP: stock pool composition
- ALIGN_STAGE2/3/4: raw factor values, preprocessed values, composite scores
- WEIGHT: model weights per factor
- TOP10_FINAL: top 10 stocks by composite score
- REBALANCE: target count and top stocks
- 买入/卖出: actual trade actions

Usage:
    python parse_jq_backtest_log.py --log results/backtest_log.txt --output results/jq_parsed.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def parse_jq_log(log_path: str) -> dict[str, Any]:
    """Parse JQ backtest log and return structured data keyed by rebalance date.

    Handles the T/T-1 date split: JQ strategy logs POOL_DUMP and REBALANCE
    with the rebalance date (T), but ALIGN_STAGE2/3/4, WEIGHT, TOP10_FINAL
    use the factor date (T-1 = context.previous_date). This function merges
    T-1 factor entries into their corresponding T rebalance entries.
    """
    content = Path(log_path).read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    dates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "pool": {"size": 0, "stocks": []},
            "weights": {},
            "stage2_raw": {},  # {factor: {stock: value}}
            "stage3_zscore": {},  # {factor: {stock: value}}
            "stage4_composite": {},  # {stock: {score, rank}}
            "top10_final": [],  # [(stock, score)]
            "rebalance": {"target_count": 0, "top3": []},
            "trades": {"buy": [], "sell": [], "hold": []},
            "align_summary": {},
        }
    )

    for line in lines:
        # POOL_DUMP|2025-01-08|127|000001.XSHE,...
        m = re.search(r"POOL_DUMP\|(\d{4}-\d{2}-\d{2})\|(\d+)\|(.+)", line)
        if m:
            date, size, stocks_str = m.group(1), int(m.group(2)), m.group(3)
            dates[date]["pool"]["size"] = size
            dates[date]["pool"]["stocks"] = stocks_str.split(",")
            continue

        # WEIGHT|2025-01-07|FactorName|w=0.123456
        m = re.search(r"WEIGHT\|(\d{4}-\d{2}-\d{2})\|(.+?)\|w=([0-9.eE+-]+)", line)
        if m:
            date, fname, w = m.group(1), m.group(2), float(m.group(3))
            dates[date]["weights"][fname] = w
            continue

        # ALIGN_STAGE2|2025-01-07|FactorName|000001.XSHE|raw=1.234567
        m = re.search(
            r"ALIGN_STAGE2\|(\d{4}-\d{2}-\d{2})\|(.+?)\|(\d+\.\w+)\|raw=(-?[0-9.eE+-]+)",
            line,
        )
        if m:
            date, fname, stock, val = (
                m.group(1),
                m.group(2),
                m.group(3),
                float(m.group(4)),
            )
            dates[date]["stage2_raw"].setdefault(fname, {})[stock] = val
            continue

        # ALIGN_STAGE3|2025-01-07|FactorName|000001.XSHE|zscore=1.234567
        m = re.search(
            r"ALIGN_STAGE3\|(\d{4}-\d{2}-\d{2})\|(.+?)\|(\d+\.\w+)\|zscore=(-?[0-9.eE+-]+)",
            line,
        )
        if m:
            date, fname, stock, val = (
                m.group(1),
                m.group(2),
                m.group(3),
                float(m.group(4)),
            )
            dates[date]["stage3_zscore"].setdefault(fname, {})[stock] = val
            continue

        # ALIGN_STAGE4|2025-01-07|000001.XSHE|composite=1.234567|rank=1
        m = re.search(
            r"ALIGN_STAGE4\|(\d{4}-\d{2}-\d{2})\|(\d+\.\w+)\|composite=(-?[0-9.eE+-]+)\|rank=(\d+)",
            line,
        )
        if m:
            date, stock, score, rank = (
                m.group(1),
                m.group(2),
                float(m.group(3)),
                int(m.group(4)),
            )
            dates[date]["stage4_composite"][stock] = {"score": score, "rank": rank}
            continue

        # TOP10_FINAL|2025-01-07|1:000001.XSHE:1.234567|2:...
        m = re.search(r"TOP10_FINAL\|(\d{4}-\d{2}-\d{2})\|(.+)", line)
        if m:
            date, entries_str = m.group(1), m.group(2)
            entries = []
            for entry in entries_str.split("|"):
                parts = entry.split(":")
                if len(parts) >= 3:
                    entries.append((parts[1], float(parts[2])))
            dates[date]["top10_final"] = entries
            continue

        # REBALANCE|2025-01-08|target_count=6|top3=000001.XSHE,000002.XSHE,...
        m = re.search(
            r"REBALANCE\|(\d{4}-\d{2}-\d{2})\|target_count=(\d+)\|top3=(.+)", line
        )
        if m:
            date, count, top3_str = m.group(1), int(m.group(2)), m.group(3)
            dates[date]["rebalance"]["target_count"] = count
            dates[date]["rebalance"]["top3"] = top3_str.split(",")
            continue

        # ALIGN_SUMMARY|2025-01-07|n_stocks=127|n_factors=5|blend_ratio=0.3|top1=...
        m = re.search(r"ALIGN_SUMMARY\|(\d{4}-\d{2}-\d{2})\|(.+)", line)
        if m:
            date, kv_str = m.group(1), m.group(2)
            for kv in kv_str.split("|"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        dates[date]["align_summary"][k] = float(v)
                    except ValueError:
                        dates[date]["align_summary"][k] = v
            continue

        # ZSCORE_POOL|2025-01-07|n_valid=114|stocks_sample=000001.XSHE,...
        m = re.search(r"ZSCORE_POOL\|(\d{4}-\d{2}-\d{2})\|n_valid=(\d+)", line)
        if m:
            date, n_valid = m.group(1), int(m.group(2))
            dates[date]["zscore_pool_size"] = n_valid
            continue

        # 买入[000001.XSHE]
        m = re.search(r"买入\[(\d+\.\w+)\]", line)
        if m:
            stock = m.group(1)
            # Find the date context from nearby REBALANCE log
            # We'll associate with the latest date seen
            if dates:
                latest_date = max(dates.keys())
                dates[latest_date]["trades"]["buy"].append(stock)
            continue

        # 卖出[000001.XSHE]
        m = re.search(r"卖出\[(\d+\.\w+)\]", line)
        if m:
            stock = m.group(1)
            if dates:
                latest_date = max(dates.keys())
                dates[latest_date]["trades"]["sell"].append(stock)
            continue

        # 已持有[000001.XSHE]
        m = re.search(r"已持有\[(\d+\.\w+)\]", line)
        if m:
            stock = m.group(1)
            if dates:
                latest_date = max(dates.keys())
                dates[latest_date]["trades"]["hold"].append(stock)
            continue

    # --- Merge T-1 factor entries into T rebalance entries ---
    # JQ strategy uses context.current_dt for POOL_DUMP/REBALANCE (T date)
    # but context.previous_date for ALIGN_STAGE/WEIGHT/TOP10_FINAL (T-1 date).
    # Identify "factor-only" dates (have stage data but no pool) and merge them
    # into the next "pool" date.
    merged = _merge_factor_dates_into_pool_dates(dict(dates))
    return merged


def _merge_factor_dates_into_pool_dates(
    dates: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge T-1 factor-only entries into their corresponding T pool entries.

    A "factor-only" date has stage2/3/4 or weights but no pool data.
    A "pool" date has pool data but no stage2/3/4.
    For each factor-only date, find the next pool date and merge factor data into it.
    """
    sorted_dates = sorted(dates.keys())

    # Classify dates
    factor_only_dates: list[str] = []
    pool_dates: list[str] = []
    for d in sorted_dates:
        entry = dates[d]
        has_pool = entry["pool"]["size"] > 0
        has_factors = bool(
            entry.get("stage2_raw")
            or entry.get("stage3_zscore")
            or entry.get("stage4_composite")
            or entry.get("weights")
        )
        if has_pool and has_factors:
            # Already complete, no merge needed
            continue
        elif has_factors and not has_pool:
            factor_only_dates.append(d)
        elif has_pool and not has_factors:
            pool_dates.append(d)

    if not factor_only_dates:
        return dates  # Nothing to merge

    # Build mapping: factor_date -> next pool_date
    for factor_date in factor_only_dates:
        # Find the nearest pool date AFTER this factor date
        target_pool_date: Optional[str] = None
        for pd_ in pool_dates:
            if pd_ > factor_date:
                target_pool_date = pd_
                break

        if target_pool_date is None:
            continue  # No matching pool date found, keep as-is

        # Merge factor data into pool date entry
        src = dates[factor_date]
        dst = dates[target_pool_date]

        for key in ("stage2_raw", "stage3_zscore"):
            if src.get(key):
                dst.setdefault(key, {}).update(src[key])

        if src.get("stage4_composite"):
            dst.setdefault("stage4_composite", {}).update(src["stage4_composite"])

        if src.get("weights"):
            dst.setdefault("weights", {}).update(src["weights"])

        if src.get("top10_final"):
            dst["top10_final"] = src["top10_final"]

        if src.get("align_summary"):
            dst.setdefault("align_summary", {}).update(src["align_summary"])

        if src.get("zscore_pool_size"):
            dst["zscore_pool_size"] = src["zscore_pool_size"]

        # Store the factor_date for reference
        dst["factor_date"] = factor_date

    # Remove factor-only dates that were merged
    for factor_date in factor_only_dates:
        # Only remove if it was actually merged (has a pool date after it)
        has_target = any(pd_ > factor_date for pd_ in pool_dates)
        if has_target:
            del dates[factor_date]

    return dates


def main():
    parser = argparse.ArgumentParser(description="Parse JQ backtest log")
    parser.add_argument("--log", required=True, help="JQ backtest log file path")
    parser.add_argument(
        "--output", default="results/jq_parsed.json", help="Output JSON path"
    )
    args = parser.parse_args()

    data = parse_jq_log(args.log)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    n_dates = len(data)
    print(f"✓ Parsed {n_dates} rebalance dates from {args.log}")
    for date in sorted(data.keys())[:5]:
        d = data[date]
        print(
            f"  {date}: pool={d['pool']['size']}, weights={len(d['weights'])}, "
            f"stage4={len(d['stage4_composite'])}, buys={len(d['trades']['buy'])}"
        )
    if n_dates > 5:
        print(f"  ... ({n_dates - 5} more dates)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
