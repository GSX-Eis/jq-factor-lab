"""Generate S1-A snapshots from the legacy runner's public entry point."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
from utils.trading_calendar import TradingCalendar

BASELINE_DIR = Path(__file__).resolve().parent
SEED = 791
PERIODS = (1, 5)
N_QUANTILES = 3
EXECUTION_LAG = 1
MODES = ("main", "fallback")


def synthetic_inputs() -> tuple[dict, pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2024-01-02", periods=12, name="date")
    assets = pd.Index([f"SYN{i:02d}" for i in range(12)], name="asset")
    factor = pd.DataFrame(rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets)
    factor.iloc[1, 0] = np.nan
    factor.iloc[2, :3] = -3.0
    factor.iloc[3] = 1.0

    closes = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.02, factor.shape), axis=0),
        index=dates,
        columns=assets,
    )
    closes.iloc[6, -1] = np.nan
    pricing = closes.stack().rename("close")
    calendar = TradingCalendar(dates, dates[0], dates[-1])
    pricing.attrs["trading_calendar"] = calendar
    returns = ForwardReturnsCalculator(trading_calendar=calendar).calculate_forward_returns(
        pricing, PERIODS, EXECUTION_LAG
    )

    # Keep one 5D IC date and a second date with just one valid 5D pair.
    keep_5d = (returns.index.get_level_values("date") == dates[0]) | (
        returns.index == (dates[4], assets[0])
    )
    returns.loc[~keep_5d, "5D"] = np.nan

    return {"seed": SEED}, factor, pricing, returns




def _labels(index: pd.Index) -> list:
    if isinstance(index, pd.MultiIndex):
        levels = [_labels(index.get_level_values(i)) for i in range(index.nlevels)]
        return [list(row) for row in zip(*levels)]
    if isinstance(index, pd.DatetimeIndex):
        return [timestamp.isoformat() for timestamp in index]
    return index.tolist()


def _freeze_frame(frame: pd.DataFrame) -> dict:
    values = frame.to_numpy(dtype=float)
    return {
        "index_names": list(frame.index.names),
        "index": _labels(frame.index),
        "column_names": list(frame.columns.names),
        "columns": _labels(frame.columns),
        "values": np.where(np.isnan(values), None, values).tolist(),
    }


def run_snapshot(
    factor: pd.DataFrame,
    pricing: pd.Series,
    returns: pd.DataFrame | None,
    *,
    mode: str,
) -> dict:
    from backtest_and_analysis.alphalens_runner import AlphalensRunner, MaxLossExceededError

    runner = AlphalensRunner(
        n_quantiles=N_QUANTILES, periods=PERIODS, execution_lag=EXECUTION_LAG,
        _force_raw_quantiles=mode == "fallback",
    )

    # The public loss error exposes the runner's exact counts. Re-run at the
    # measured loss to also freeze its acceptance boundary.
    try:
        runner.run(factor, pricing, returns, max_loss=0)
    except MaxLossExceededError as exc:
        match = re.search(r"retained=(\d+)/(\d+)$", str(exc))
        if match is None:
            raise
        retained, total = map(int, match.groups())
    else:
        raise AssertionError("Expected the baseline input to lose some observations")
    lost = total - retained
    loss = lost / total

    result = runner.run(factor, pricing, returns, max_loss=loss)

    if result["quantile_returns"].empty or result["std_err"].empty != (mode == "fallback"):
        raise AssertionError(f"Legacy runner did not complete the {mode} quantile path")

    return {
        "max_loss": {"input": total, "retained": retained, "lost": lost, "fraction": loss},
        "results": {name: _freeze_frame(frame) for name, frame in result.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", choices=("synthetic",))
    parser.add_argument("--overwrite", action="store_true", help="Replace the selected frozen snapshot")
    args = parser.parse_args()
    output = BASELINE_DIR / f"{args.sample}.json"
    if output.exists() and not args.overwrite:
        parser.error(f"{output.name} already exists; pass --overwrite to replace it")

    identity, *inputs = synthetic_inputs()
    snapshot = {"input": identity, "paths": {mode: run_snapshot(*inputs, mode=mode) for mode in MODES}}
    payload = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    with output.open("w" if args.overwrite else "x", encoding="utf-8") as file:
        file.write(payload)


if __name__ == "__main__":
    main()
