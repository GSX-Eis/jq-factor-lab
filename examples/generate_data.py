"""Generate deterministic fictional prices; no downloaded market data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_prices(seed=42, sessions=180, assets=20):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=sessions)
    changes = rng.normal(0, 0.015, (sessions, assets))
    prices = 100 * np.exp(changes.cumsum(axis=0))
    frame = pd.DataFrame(prices, index=dates, columns=[f"DEMO{i:03d}" for i in range(assets)])
    frame.index.name, frame.columns.name = "date", "stock_code"
    result = frame.stack().rename("close").reset_index()
    result["factor"] = 1.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            make_prices().to_csv(stream, index=False)
    except OSError as exc:
        parser.exit(1, f"Cannot write --output {args.output}: {exc}; choose a new writable file path\n")
    print(json.dumps({"data": "synthetic", "output": str(args.output)}))


if __name__ == "__main__":
    main()
