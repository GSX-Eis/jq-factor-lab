"""Evaluate one trusted factor against user-supplied daily CSV prices."""
import argparse
import json
import logging
from pathlib import Path
import sys
from contextlib import redirect_stdout


def read_prices(path):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path, dtype={"stock_code": str})
    required = {"date", "stock_code", "close"}
    if not required <= set(frame.columns):
        raise ValueError("CSV requires date, stock_code, close")
    if frame.empty:
        raise ValueError("CSV is empty")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame["date"].dt.tz is not None:
        raise ValueError("Dates must be non-null and timezone-naive")
    if not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("Daily input requires dates at midnight")
    if not frame["stock_code"].str.fullmatch(r"[A-Za-z0-9_.-]+", na=False).all():
        raise ValueError("Invalid stock_code")
    if frame.duplicated(["date", "stock_code"]).any():
        raise ValueError("Duplicate date/stock_code")
    if "factor" not in frame:
        frame["factor"] = 1.0
    for column in ("close", "factor"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not (np.isfinite(frame[column]) & frame[column].gt(0)).all():
            raise ValueError(f"{column} must contain finite positive values")
    return frame.sort_values(["date", "stock_code"]).reset_index(drop=True)


def analyze(config_path, output_dir=None):
    import duckdb
    import pandas as pd
    from backtest_and_analysis.alphalens_runner import AlphalensRunner
    from backtest_and_analysis.factor_loader import FactorLoader
    from utils.local_outputs import write_json

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object with input_csv, factor_file, factor_class and output_dir")
    for key in ("input_csv", "factor_file", "factor_class") + (() if output_dir else ("output_dir",)):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string; set it in the config")
    relative = lambda key: (config_path.parent / config[key]).resolve()
    periods = config.get("periods", [1, 5, 10])
    lag = config.get("execution_lag", 1)
    quantiles = config.get("n_quantiles", 5)
    if not isinstance(periods, list) or not periods or any(type(p) is not int or p < 1 for p in periods):
        raise ValueError("periods must be a non-empty list of positive integers, e.g. [1, 5]")
    if type(lag) is not int or lag < 0:
        raise ValueError("execution_lag must be a nonnegative integer")
    if type(quantiles) is not int or quantiles < 2:
        raise ValueError("n_quantiles must be an integer >= 2")
    target = Path(output_dir).resolve() if output_dir else relative("output_dir")
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    from utils.trading_calendar import TradingCalendar, load_trading_calendar
    keys = [key for key in ('calendar_file', 'calendar_duckdb_path') if config.get(key)]
    if len(keys) != 1:
        raise ValueError('Set exactly one of calendar_file or calendar_duckdb_path; prices do not define sessions')
    if keys[0] == 'calendar_file':
        spec = json.loads(relative('calendar_file').read_text())
        calendar = TradingCalendar(spec['sessions'], spec['start_date'], spec['end_date'], spec['calendar_id'])
    else:
        calendar = load_trading_calendar(duckdb_path=relative('calendar_duckdb_path'))
    frame = read_prices(relative("input_csv"))
    calendar.validate_coverage(frame.date.min(), frame.date.max())
    classes = FactorLoader._load_classes_from_file(str(relative("factor_file")), config["factor_class"])
    if len(classes) != 1:
        raise ValueError("Select exactly one factor class")
    # A source CSV and isolated in-memory DuckDB keep the demo self-contained.
    with duckdb.connect(":memory:") as connection:
        stored = frame.copy()
        connection.register("input_prices", stored)
        connection.execute("CREATE TABLE price_data_daily AS SELECT * FROM input_prices")
        connection.unregister("input_prices")
        factor = classes[0](
            connection, config["factor_class"],
            frame.date.min().strftime("%Y-%m-%d"), frame.date.max().strftime("%Y-%m-%d"),
            stock_pool=sorted(frame.stock_code.unique()), db_backend="duckdb", pit_enabled=True,
        )
        values = factor.get_factor_values(force_recalculate=True)
    pricing = frame.assign(adjusted_close=frame.close * frame.factor).set_index(
        ["stock_code", "date"]
    )["adjusted_close"]
    pricing.index.names = ["asset", "date"]
    pricing.attrs["trading_calendar"] = calendar
    results = AlphalensRunner(
        n_quantiles=quantiles, periods=tuple(periods), execution_lag=lag,
    ).run(values, pricing)
    ic = results["ic"]
    if ic.empty or not isinstance(ic.index, pd.DatetimeIndex) or ic.dropna(how="all").empty:
        raise RuntimeError("Factor evaluation failed or returned no usable IC observations")
    summary = {
        "factor": config["factor_class"], "rows": len(frame),
        "assets": int(frame.stock_code.nunique()), "periods": periods,
        "execution_lag": lag, "ic_mean": ic.mean().to_dict(),
        "calendar_id": calendar.calendar_id, "calendar_sha256": calendar.fingerprint,
        "ic_observations": ic.count().to_dict(),
        "return_definition": "price[t + lag + period] / price[t + lag] - 1",
    }
    target.mkdir(parents=True, exist_ok=False)
    values.to_csv(target / "factor_values.csv")
    for name, value in results.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value.to_csv(target / f"{name}.csv")
    write_json(target / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single-factor-analysis")
    single.add_argument("--config", type=Path, required=True)
    single.add_argument("--output-dir", type=Path)
    from examples.synthesis_demo import METHODS, run_demo
    synthesis = commands.add_parser("factor-synthesis-demo", help="Run a synthetic factor synthesis example")
    synthesis.add_argument("--method", choices=METHODS, default="weighted_sum",
                           help="weighted_sum runs directly; other methods require verified forward-label exits (see docs/SYNTHESIS.md)")
    synthesis.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "factor-synthesis-demo":
                summary = run_demo(args.method, args.output_dir)
            else:
                summary = analyze(args.config, args.output_dir)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
