#!/usr/bin/env python3
"""
Parse local VectorBT backtest alignment data into structured JSON.

Reads structured files saved by AlignmentDataWriter (when save_alignment_data=true):
  - stage2_raw_factors.parquet: raw factor values per stock per date
  - stage3_preprocessed.parquet: preprocessed (zscore) factor values
  - stage4_composite.parquet: final composite scores and ranks
  - lasso_weights.json: model weights per date
  - stock_pool_snapshot.csv: stock pool composition
  - holdings_snapshot.csv: target holdings and trade actions

Falls back to legacy factor_values_snapshot.parquet if stage files are absent.

Output format matches parse_jq_backtest_log.py for use with compare_backtest_alignment.py.

Usage:
    python parse_local_backtest_log.py \
        --alignment-dir outputs/synthesis/alignment/exp_58f_e5 \
        --output results/local_parsed.json

    # Legacy mode: parse text log (best-effort, limited data)
    python parse_local_backtest_log.py \
        --log results/local_backtest.log \
        --output results/local_parsed.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Primary path: read structured alignment files from AlignmentDataWriter
# ---------------------------------------------------------------------------


def parse_from_alignment_dir(
    alignment_dir: Path, bundle_path: Optional[Path] = None, *, data_bindings: Optional[dict] = None
) -> dict[str, Any]:
    """Parse structured alignment output files into per-date dict."""
    from utils.data_store import DataRef, binding_contract, read_document, read_reference
    binding = binding_contract(data_bindings)
    if binding:
        if not binding.get("output_db"):
            raise ValueError("Alignment reads require output_db and frozen references")
        if bundle_path is not None:
            raise ValueError("Bound alignment reads require published weights instead of a bundle file")
        selected = {**binding.get("outputs", {}), **binding.get("evaluations", {})}
        summary_ref = binding.get("documents", {}).get("meta.alignment_summary", selected.get("meta.alignment_summary"))
        if not isinstance(summary_ref, dict) or summary_ref.get("role") != "output" or summary_ref.get("table") != "meta.documents":
            raise ValueError("Alignment requires a frozen summary reference")
        summary = read_document(binding['output_db'], DataRef(**summary_ref))
        frozen = summary.get('data_refs')
        if not isinstance(frozen, dict) or not frozen:
            raise ValueError("Alignment summary does not bind its published stages")
        expected = {'alignment.' + ('lasso_weights' if name == 'lasso_weights_table' else name)
                    for name in summary['output_files'] if name not in {'summary', 'lasso_weights'}}
        if expected != {name for name in frozen if name.startswith('alignment.')}:
            raise ValueError("Alignment summary is missing a declared stage reference")
        for target, ref in selected.items():
            if target.startswith('alignment.') and frozen.get(target) != ref:
                raise ValueError(f"Alignment stage differs from frozen summary: {target}")
        selected = frozen
        if "alignment.stock_pool_snapshot" not in selected:
            raise ValueError("Alignment stock pool snapshot is not bound")
        frames = {}
        for name in ('stock_pool_snapshot', 'holdings_snapshot', 'stage2_raw_factors',
                     'stage3_preprocessed', 'stage4_composite', 'factor_values_snapshot', 'lasso_weights'):
            target = 'alignment.' + name
            ref = selected.get(target)
            if ref is None:
                frames[name] = pd.DataFrame()
                continue
            if not isinstance(ref, dict) or ref.get('role') != 'output' or ref.get('table') != target:
                raise ValueError(f"Expected one exact {target} reference")
            frames[name] = read_reference(binding['output_db'], DataRef(**ref), allow_labels=True)
        pool_df, holdings_df = frames['stock_pool_snapshot'], frames['holdings_snapshot']
        stage2_df, stage3_df, stage4_df = (frames[key] for key in ('stage2_raw_factors', 'stage3_preprocessed', 'stage4_composite'))
        legacy_fv_df = frames['factor_values_snapshot'] if stage2_df.empty else pd.DataFrame()
        weights = frames['lasso_weights']
        if not weights.empty and weights.duplicated(['factor_date', 'factor_name']).any():
            raise ValueError("Alignment weights have duplicate date/factor keys")
        weights_data = {str(date): dict(zip(frame['factor_name'], frame['weight']))
                        for date, frame in weights.groupby('factor_date')} if not weights.empty else {}
        bundle_weights = []
    else:
        pool_df, holdings_df, stage2_df, stage3_df, stage4_df, legacy_fv_df, weights_data, bundle_weights = _load_files(alignment_dir, bundle_path)

    return _parse_frames(pool_df, holdings_df, stage2_df, stage3_df, stage4_df, legacy_fv_df, weights_data, bundle_weights)


def _load_files(alignment_dir, bundle_path):

    # --- Load files ---
    pool_path = alignment_dir / "stock_pool_snapshot.csv"
    holdings_path = alignment_dir / "holdings_snapshot.csv"
    stage2_path = _find_file(alignment_dir, "stage2_raw_factors")
    stage3_path = _find_file(alignment_dir, "stage3_preprocessed")
    stage4_path = _find_file(alignment_dir, "stage4_composite")
    weights_path = alignment_dir / "lasso_weights.json"
    legacy_fv_path = _find_file(alignment_dir, "factor_values_snapshot")

    pool_df = pd.read_csv(pool_path) if pool_path.exists() else pd.DataFrame()
    holdings_df = (
        pd.read_csv(holdings_path) if holdings_path.exists() else pd.DataFrame()
    )
    stage2_df = _read_df(stage2_path)
    stage3_df = _read_df(stage3_path)
    stage4_df = _read_df(stage4_path)
    weights_data = json.loads(weights_path.read_text()) if weights_path.exists() else {}
    legacy_fv_df = (
        _read_df(legacy_fv_path)
        if stage2_df.empty and legacy_fv_path
        else pd.DataFrame()
    )

    # Also try bundle for weights if lasso_weights.json is empty
    bundle_weights = _load_bundle_weights(bundle_path) if bundle_path else []
    return pool_df, holdings_df, stage2_df, stage3_df, stage4_df, legacy_fv_df, weights_data, bundle_weights


def _parse_frames(pool_df, holdings_df, stage2_df, stage3_df, stage4_df, legacy_fv_df, weights_data, bundle_weights):
    dates = {}
    for frame in (pool_df, holdings_df, stage2_df, stage3_df, stage4_df, legacy_fv_df):
        for column in ('rebalance_date', 'factor_date'):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column]).dt.strftime('%Y-%m-%d')
    weights_data = {str(pd.Timestamp(date).date()): weights for date, weights in weights_data.items()}

    # --- Collect all rebalance dates ---
    all_dates: set[str] = set()
    for df in [pool_df, holdings_df, stage2_df, stage3_df, stage4_df]:
        if not df.empty and "rebalance_date" in df.columns:
            all_dates.update(df["rebalance_date"].astype(str).unique())

    for rd in sorted(all_dates):
        entry = _empty_date_entry()

        # Pool
        if not pool_df.empty:
            pool_slice = pool_df[pool_df["rebalance_date"].astype(str) == rd]
            pool_stocks = sorted(
                pool_slice[pool_slice["in_pool"] == True]["stock_code"].tolist()
            )
            entry["pool"]["size"] = len(pool_stocks)
            entry["pool"]["stocks"] = pool_stocks

        # Stage 2: raw factor values
        if not stage2_df.empty:
            s2 = stage2_df[stage2_df["rebalance_date"].astype(str) == rd]
            meta_cols = {"rebalance_date", "factor_date", "stock_code"}
            factor_cols = [c for c in s2.columns if c not in meta_cols]
            for col in factor_cols:
                vals = s2[["stock_code", col]].dropna(subset=[col])
                if not vals.empty:
                    entry["stage2_raw"][col] = dict(
                        zip(vals["stock_code"], vals[col].astype(float))
                    )
        elif not legacy_fv_df.empty:
            # Fallback: extract from legacy factor_values_snapshot
            fv = legacy_fv_df[legacy_fv_df["rebalance_date"].astype(str) == rd]
            raw_cols = [c for c in fv.columns if c.endswith("_raw")]
            for col in raw_cols:
                fname = col[: -len("_raw")]
                vals = fv[["stock_code", col]].dropna(subset=[col])
                if not vals.empty:
                    entry["stage2_raw"][fname] = dict(
                        zip(vals["stock_code"], vals[col].astype(float))
                    )

        # Stage 3: preprocessed (zscore) values
        if not stage3_df.empty:
            s3 = stage3_df[stage3_df["rebalance_date"].astype(str) == rd]
            meta_cols = {"rebalance_date", "factor_date", "stock_code"}
            factor_cols = [c for c in s3.columns if c not in meta_cols]
            for col in factor_cols:
                vals = s3[["stock_code", col]].dropna(subset=[col])
                if not vals.empty:
                    entry["stage3_zscore"][col] = dict(
                        zip(vals["stock_code"], vals[col].astype(float))
                    )
        elif not legacy_fv_df.empty:
            fv = legacy_fv_df[legacy_fv_df["rebalance_date"].astype(str) == rd]
            proc_cols = [c for c in fv.columns if c.endswith("_processed")]
            for col in proc_cols:
                fname = col[: -len("_processed")]
                vals = fv[["stock_code", col]].dropna(subset=[col])
                if not vals.empty:
                    entry["stage3_zscore"][fname] = dict(
                        zip(vals["stock_code"], vals[col].astype(float))
                    )

        # Stage 4: composite scores
        if not stage4_df.empty:
            s4 = stage4_df[stage4_df["rebalance_date"].astype(str) == rd]
            for _, row in s4.iterrows():
                entry["stage4_composite"][row["stock_code"]] = {
                    "score": float(row["composite_score"]),
                    "rank": int(row["rank"]),
                }
            top10 = s4.nsmallest(10, "rank")
            entry["top10_final"] = [
                (r["stock_code"], float(r["composite_score"]))
                for _, r in top10.iterrows()
            ]
        elif not legacy_fv_df.empty:
            fv = legacy_fv_df[legacy_fv_df["rebalance_date"].astype(str) == rd]
            if "composite_score" in fv.columns and "rank" in fv.columns:
                score_slice = fv[["stock_code", "composite_score", "rank"]].dropna(
                    subset=["composite_score"]
                )
                for _, row in score_slice.iterrows():
                    entry["stage4_composite"][row["stock_code"]] = {
                        "score": float(row["composite_score"]),
                        "rank": int(row["rank"]),
                    }
                top10 = score_slice.nsmallest(10, "rank")
                entry["top10_final"] = [
                    (r["stock_code"], float(r["composite_score"]))
                    for _, r in top10.iterrows()
                ]

        # Weights
        factor_date = rd
        for frame in (pool_df, stage2_df):
            if not frame.empty and "factor_date" in frame:
                candidates = frame.loc[frame['rebalance_date'] == rd, 'factor_date'].dropna().unique()
                if len(candidates) > 1:
                    raise ValueError("Rebalance date has multiple factor dates")
                if len(candidates):
                    factor_date = str(candidates[0])
                    break
        selected_weights = weights_data.get(factor_date, weights_data.get(rd))
        if selected_weights is not None:
            entry["weights"] = {k: float(v) for k, v in selected_weights.items()}
        elif bundle_weights:
            entry["weights"] = _get_weights_for_date(bundle_weights, factor_date)

        # Holdings / trades
        if not holdings_df.empty:
            h = holdings_df[holdings_df["rebalance_date"].astype(str) == rd]
            entry["rebalance"]["target_count"] = len(h)
            top3 = h.nsmallest(3, "rank")["stock_code"].tolist() if not h.empty else []
            entry["rebalance"]["top3"] = top3
            if "action" in h.columns:
                entry["trades"]["buy"] = h[h["action"] == "BUY"]["stock_code"].tolist()
                entry["trades"]["sell"] = h[h["action"] == "SELL"][
                    "stock_code"
                ].tolist()
                entry["trades"]["hold"] = h[h["action"] == "HOLD"][
                    "stock_code"
                ].tolist()

        # zscore pool size (count of stocks with any stage3 data)
        if entry["stage3_zscore"]:
            first_factor = next(iter(entry["stage3_zscore"].values()))
            entry["zscore_pool_size"] = len(first_factor)

        dates[rd] = entry

    return dates


# ---------------------------------------------------------------------------
# Fallback path: parse text log (best-effort, kept for backward compat)
# ---------------------------------------------------------------------------


def parse_from_text_log(log_path: str) -> dict[str, Any]:
    """Legacy: parse VectorBT text log. Limited data extraction."""
    content = Path(log_path).read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    dates: dict[str, dict[str, Any]] = defaultdict(lambda: _empty_date_entry())

    for line in lines:
        # POOL_DUMP|2025-01-08|127|000001.XSHE,...
        m = re.search(r"POOL_DUMP\|(\d{4}-\d{2}-\d{2})\|(\d+)\|(.+)", line)
        if m:
            date, size, stocks_str = m.group(1), int(m.group(2)), m.group(3)
            dates[date]["pool"]["size"] = size
            dates[date]["pool"]["stocks"] = stocks_str.split(",")
            continue

        # WEIGHT|2025-01-07|FactorName|w=0.123456
        m = re.search(r"WEIGHT\|(\d{4}-\d{2}-\d{2})\|(.+?)\|w=([0-9.e+-]+)", line)
        if m:
            date, fname, w = m.group(1), m.group(2), float(m.group(3))
            dates[date]["weights"][fname] = w
            continue

        # FULL_RANK|2025-01-07|FactorName|n=114|stock1:raw1:rank1|...
        m = re.search(r"FULL_RANK\|(\d{4}-\d{2}-\d{2})\|(.+?)\|n=(\d+)\|(.+)", line)
        if m:
            date, fname, _, rank_data = (
                m.group(1),
                m.group(2),
                int(m.group(3)),
                m.group(4),
            )
            for entry in rank_data.split("|"):
                parts = entry.split(":")
                if len(parts) >= 2:
                    dates[date]["stage2_raw"].setdefault(fname, {})[parts[0]] = float(
                        parts[1]
                    )
            continue

        # AFTER_ZSCORE|2025-01-07|FactorName|stock|raw=...|zscore=...
        m = re.search(
            r"AFTER_ZSCORE\|(\d{4}-\d{2}-\d{2})\|(.+?)\|(\d+\.\w+)\|raw=([0-9.e+-]+)\|zscore=([0-9.e+-]+)",
            line,
        )
        if m:
            date, fname, stock = m.group(1), m.group(2), m.group(3)
            dates[date]["stage2_raw"].setdefault(fname, {})[stock] = float(m.group(4))
            dates[date]["stage3_zscore"].setdefault(fname, {})[stock] = float(
                m.group(5)
            )
            continue

        # FINAL_SCORE|2025-01-07|TOP10|stock1:score1|...
        m = re.search(r"FINAL_SCORE\|(\d{4}-\d{2}-\d{2})\|TOP10\|(.+)", line)
        if m:
            date, entries_str = m.group(1), m.group(2)
            for rank, entry in enumerate(entries_str.split("|"), 1):
                parts = entry.split(":")
                if len(parts) >= 2:
                    stock, score = parts[0], float(parts[1])
                    dates[date]["stage4_composite"][stock] = {
                        "score": score,
                        "rank": rank,
                    }
                    dates[date]["top10_final"].append((stock, score))
            continue

        # [VectorBT] Rebalance ... target=6 stocks
        m = re.search(
            r"\[VectorBT\].*?Rebalance.*?(\d{4}-\d{2}-\d{2}).*?target=(\d+)", line
        )
        if m:
            date, target = m.group(1), int(m.group(2))
            dates[date]["rebalance"]["target_count"] = target
            continue

    return dict(dates)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_date_entry() -> dict[str, Any]:
    return {
        "pool": {"size": 0, "stocks": []},
        "weights": {},
        "stage2_raw": {},
        "stage3_zscore": {},
        "stage4_composite": {},
        "top10_final": [],
        "rebalance": {"target_count": 0, "top3": []},
        "trades": {"buy": [], "sell": [], "hold": []},
        "align_summary": {},
        "zscore_pool_size": 0,
    }


def _find_file(directory: Path, stem: str) -> Optional[Path]:
    """Find a file by stem, preferring parquet over csv."""
    parquet = directory / f"{stem}.parquet"
    if parquet.exists():
        return parquet
    csv = directory / f"{stem}.csv"
    if csv.exists():
        return csv
    return parquet  # Return parquet path as default (may not exist)


def _read_df(path: Optional[Path]) -> pd.DataFrame:
    """Read a DataFrame from parquet or csv, returning empty if not found."""
    if path is None or not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_bundle_weights(bundle_path: Path) -> list[dict]:
    """Load model weights from bundle file."""
    if not bundle_path.exists():
        return []
    bundle = json.loads(bundle_path.read_text())
    return bundle.get("index", {}).get("models", [])


def _get_weights_for_date(models: list[dict], factor_date: str) -> dict[str, float]:
    """Find the model whose train_end < factor_date and return its weights."""
    best_model: dict | None = None
    for m in models:
        if m.get("train_end", "") < factor_date:
            best_model = m
        else:
            break
    if best_model is None:
        return {}
    return best_model.get("weights", {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse local backtest data into structured JSON for alignment comparison"
    )
    parser.add_argument(
        "--alignment-dir",
        help="Path to alignment output directory (from AlignmentDataWriter). "
        "This is the preferred input source.",
    )
    parser.add_argument(
        "--bundle",
        help="Path to lasso model bundle JSON (optional, for weight lookup fallback)",
    )
    parser.add_argument(
        "--log",
        help="Path to text log file (legacy fallback, limited data extraction)",
    )
    parser.add_argument(
        "--output",
        default="results/local_parsed.json",
        help="Output JSON path",
    )
    parser.add_argument('--data-bindings', type=Path, help='Frozen output table references JSON')
    args = parser.parse_args()
    from utils.data_store import binding_contract
    binding = binding_contract(json.loads(args.data_bindings.read_text()) if args.data_bindings else None)

    if not args.alignment_dir and not args.log and not binding:
        parser.error("Either --data-bindings, --alignment-dir or --log is required")
    if binding and args.log:
        parser.error("Bound alignment reads require a published summary instead of a text log")

    # Prefer structured alignment data over text log
    if args.alignment_dir or binding:
        adir = Path(args.alignment_dir or '.')
        if not binding and not adir.exists():
            print(f"ERROR: Alignment directory not found: {adir}")
            raise SystemExit(1)
        bundle = Path(args.bundle) if args.bundle else None
        data = parse_from_alignment_dir(adir, bundle, data_bindings=binding)
        source = str(adir)
    else:
        data = parse_from_text_log(args.log)
        source = args.log

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # Summary
    n_dates = len(data)
    print(f"✓ Parsed {n_dates} rebalance dates from {source}")
    for date in sorted(data.keys())[:5]:
        d = data[date]
        print(
            f"  {date}: pool={d['pool']['size']}, weights={len(d['weights'])}, "
            f"stage2={len(d['stage2_raw'])}, stage3={len(d['stage3_zscore'])}, "
            f"stage4={len(d['stage4_composite'])}"
        )
    if n_dates > 5:
        print(f"  ... ({n_dates - 5} more dates)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
