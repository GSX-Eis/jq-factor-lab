"""Check three explicit-version read paths and a small signal flow, without writes to stores.

Run: python -m examples.canonical_data_reads --config config.json --receipt receipt.json
The config selects canonical input/output databases, two price DataRefs (current
and frozen), one factor DataRef/name, stocks and a bounded development date range.
No environment binding, source alias, training or result publication is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from factor_synthesis.factor_reader import FactorReader
from utils.data_store import DataRef, read_reference
from utils.duckdb_adapter import get_data_adapter


def _panel(frame, value):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame.pivot(index="date", columns="stock_code", values=value).sort_index().sort_index(axis=1)


def _summary(panel):
    return {
        "rows": int(panel.count().sum()), "dates": len(panel), "stocks": len(panel.columns),
        "sha256": hashlib.sha256(panel.to_csv(float_format="%.17g").encode()).hexdigest(),
    }


def _check_connections(frame, expected):
    paths = {str(Path(p).resolve()) for p in frame["file"] if p}
    if paths != {str(Path(expected).resolve())}:
        raise ValueError(f"Unexpected attached database files: {sorted(paths)}")
    return sorted(paths)


def run_checks(config):
    """Read only the requested sample; fail on missing data or version mismatches."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    required = {"input_db", "output_db", "price_versions", "factor_version", "factor_name",
                "stocks", "start_date", "end_date"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    for key in ("input_db", "output_db", "factor_name", "start_date", "end_date"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if "sources" in config or "data_bindings" in config:
        raise ValueError("Use explicit DataRefs, not sources or inherited data_bindings")
    if not isinstance(config["price_versions"], dict) or set(config["price_versions"]) != {"current", "frozen"}:
        raise ValueError("price_versions requires current and frozen DataRefs")
    for ref in [*config["price_versions"].values(), config["factor_version"]]:
        if not isinstance(ref, dict):
            raise ValueError("Each DataRef must be a JSON object")
        DataRef(**ref)
    stocks = config["stocks"]
    if (not isinstance(stocks, list) or not stocks
            or any(not isinstance(stock, str) or not stock.strip() for stock in stocks)
            or len(stocks) != len(set(stocks))):
        raise ValueError("stocks must be a nonempty list of unique non-empty strings")
    start, end = config["start_date"], config["end_date"]
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start_date must not follow end_date")
    paths = {key: str(Path(config[key]).expanduser().resolve(strict=True)) for key in ("input_db", "output_db")}
    result = {"status": "passed", "price_reads": {}}
    prices = {}
    for name, raw_ref in config["price_versions"].items():
        ref = DataRef(**raw_ref)
        if ref.role != "input":
            raise ValueError("Price DataRefs must select input versions")
        bindings = {"input_db": paths["input_db"], "inputs": {"price_data_daily": raw_ref}}
        adapter = get_data_adapter(duckdb_path=paths["input_db"], data_bindings=bindings, use_cache=False)
        try:
            actual = adapter.fetch_price_data(stocks, start, end, ["close"])
            if actual.empty:
                raise ValueError(f"No {name} price rows in the selected sample")
            actual = _panel(actual.rename(columns={"code": "stock_code"}), "close")
            expected = _panel(read_reference(paths["input_db"], ref, fields=["date", "stock_code", "close"],
                                             stocks=stocks, start=start, end=end), "close")
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
            opened = _check_connections(adapter.execute_sql("PRAGMA database_list"), paths["input_db"])
        finally:
            adapter.close()
        prices[name] = actual
        result["price_reads"][name] = {**_summary(actual), "ref": raw_ref, "opened_databases": opened,
                                        "exact_reference_match": True}

    ref = DataRef(**config["factor_version"])
    if ref.role != "output":
        raise ValueError("Factor DataRef must select an output artifact")
    bindings = {"output_db": paths["output_db"], "outputs": {"factor_values": config["factor_version"]}}
    reader = FactorReader(paths["output_db"], data_bindings=bindings)
    actual = reader.get_factor_values(config["factor_name"], start, end, stock_pool=stocks, pivot_format=False)
    if actual.empty:
        raise ValueError("No factor rows in the selected sample")
    actual = _panel(actual, "factor_value")
    expected = read_reference(paths["output_db"], ref,
                              fields=["date", "stock_code", "factor_name", "factor_value"],
                              stocks=stocks, start=start, end=end)
    expected = _panel(expected.loc[expected.factor_name == config["factor_name"]], "factor_value")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    with reader.get_connection() as conn:
        opened = _check_connections(conn.execute("PRAGMA database_list").fetchdf(), paths["output_db"])
    result["factor_read"] = {**_summary(actual), "ref": config["factor_version"],
                             "opened_databases": opened, "exact_reference_match": True}

    momentum = prices["frozen"].pct_change(fill_method=None).stack().rename("price_change")
    aligned = pd.concat([momentum, actual.stack().rename("stored_factor")], axis=1).dropna()
    if aligned.empty or not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Minimal flow needs finite overlapping prices and factor values")
    score = aligned.groupby(level="date").rank(pct=True).mean(axis=1).unstack("stock_code")
    result["minimal_flow"] = {"aligned_rows": len(aligned), "score": _summary(score),
                              "method": "equal mean of cross-sectional ranks; smoke check only",
                              "trade_or_publication_executed": False}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.receipt.exists():
        parser.error("Receipt already exists; choose a new path")
    try:
        config_bytes = args.config.read_bytes()
        result = run_checks(json.loads(config_bytes))
        result["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
        with args.receipt.open("x") as output:
            output.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(json.dumps({"ok": False, "error": {
            "code": "canonical_data_reads_config_error", "message": str(exc), "details": {},
        }}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
