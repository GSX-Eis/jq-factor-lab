"""Run offline demos or evaluate trusted factors against local market data."""
import argparse
from collections.abc import Sequence
import json
import logging
from pathlib import Path
import sys
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
from typing import Any


def read_prices(path: Path):
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


def analyze(
    config_path: Path, output_dir: Path | None = None, *, overwrite: bool = False,
) -> dict[str, Any]:
    import duckdb
    import pandas as pd
    from backtest_and_analysis.alphalens_runner import AlphalensRunner, InsufficientCrossSectionAssets
    from backtest_and_analysis.factor_loader import FactorLoader
    from utils.local_outputs import new_run_directory, stage_demo_output, write_json

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object with input_csv, factor_file and factor_class")
    keys = ("input_csv", "factor_file", "factor_class")
    if output_dir is None and "output_dir" in config:
        keys += ("output_dir",)
    for key in keys:
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
    target = Path(output_dir).expanduser().absolute() if output_dir is not None else None
    if target is None and "output_dir" in config:
        target = config_path.parent / config["output_dir"]
    if target is not None:
        target = target.parent.resolve() / target.name
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {target}; pass --overwrite or choose a new directory")
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
    assets = int(frame.stock_code.nunique())
    if assets < 2:
        raise InsufficientCrossSectionAssets()
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
    isolated = target is None
    if target is None:
        target = new_run_directory(Path("outputs/daily"))
    summary = {
        "factor": config["factor_class"], "rows": len(frame),
        "assets": assets, "periods": periods, "output_dir": str(target),
        "execution_lag": lag, "ic_mean": ic.mean().to_dict(),
        "calendar_id": calendar.calendar_id, "calendar_sha256": calendar.fingerprint,
        "ic_observations": ic.count().to_dict(),
        "return_definition": "price[t + lag + period] / price[t + lag] - 1",
    }
    with stage_demo_output(target, overwrite=overwrite or isolated) as staging:
        values.to_csv(staging / "factor_values.csv")
        for name, value in results.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value.to_csv(staging / f"{name}.csv")
        write_json(staging / "summary.json", summary)
    return summary


def _daily_demo(output_dir: Path | None, *, overwrite: bool) -> dict[str, Any]:
    from examples.generate_data import make_prices

    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="jq-factor-lab-daily-") as directory:
        input_dir = Path(directory)
        make_prices().to_csv(input_dir / "prices.csv", index=False)
        config = input_dir / "config.json"
        config.write_text(json.dumps({
            "input_csv": "prices.csv", "factor_file": str(root / "examples/momentum.py"),
            "factor_class": "DemoMomentum", "calendar_file": str(root / "config/demo_calendar.json"),
        }), encoding="utf-8")
        return analyze(config, output_dir, overwrite=overwrite)


def _bar_demo(output_dir: Path | None, *, overwrite: bool) -> int:
    from examples.synthetic_bars import write_demo_bars
    from tools.agent_harness.cli import main as harness_main
    from utils.local_outputs import new_run_directory, write_json

    run = new_run_directory(Path("outputs/bar-demo"))
    config = run / "config.json"
    write_json(config, write_demo_bars(run / "bars.csv"))
    argv = [
        "time-series-analysis", "--config", str(config), "--diagnostics-report",
        "--output-root", str(run / "harness"), "--results-db", str(run / "evaluations.duckdb"),
    ]
    if output_dir is not None:
        argv.extend(["--output-dir", str(output_dir)])
    if overwrite:
        argv.append("--overwrite")
    return harness_main(argv, public_bar_cli=True)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"time-series-analysis", "read-bar-evaluations"}:
        from tools.agent_harness.cli import main as harness_main
        return harness_main(argv, public_bar_cli=True)

    parser = argparse.ArgumentParser(prog="jq-factor-lab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single-factor-analysis", help="Evaluate a daily multi-asset CSV")
    single.add_argument("--config", type=Path, required=True)
    daily = commands.add_parser("daily-demo", help="Evaluate fictional daily prices across multiple assets")
    bar = commands.add_parser("bar-demo", help="Evaluate synthetic minute Bars for one instrument")
    commands.add_parser("time-series-analysis", help="Evaluate local Bar data using the Harness")
    commands.add_parser("read-bar-evaluations", help="Read a Bar evaluation DuckDB")
    from examples.synthesis_demo import METHODS
    synthesis = commands.add_parser("factor-synthesis-demo", help="Run a synthetic factor synthesis example")
    synthesis.add_argument("--method", choices=METHODS, default="weighted_sum",
                           help="weighted_sum runs directly; other methods require verified forward-label exits (see docs/SYNTHESIS.md)")
    for command in (single, daily, bar, synthesis):
        command.add_argument("--output-dir", type=Path, help="Explicit directory (default: a new isolated run or the configured path)")
        command.add_argument("--overwrite", action="store_true", help="Update generated files in an existing output directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    from backtest_and_analysis.alphalens_runner import InsufficientCrossSectionAssets
    if args.command == "bar-demo":
        try:
            return _bar_demo(args.output_dir, overwrite=args.overwrite)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Cannot prepare Bar demo: {exc}\n")
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "factor-synthesis-demo":
                from examples.synthesis_demo import run_demo
                summary = run_demo(args.method, args.output_dir, overwrite=args.overwrite)
            elif args.command == "daily-demo":
                summary = _daily_demo(args.output_dir, overwrite=args.overwrite)
            else:
                summary = analyze(args.config, args.output_dir, overwrite=args.overwrite)
    except InsufficientCrossSectionAssets as exc:
        print(json.dumps({"ok": False, "exit_code": 1, "error": {"code": exc.code, "message": str(exc)}}))
        parser.exit(1, f"Analysis failed: {exc}\n")
    except ModuleNotFoundError as exc:
        if args.command != "factor-synthesis-demo" or exc.name not in {
            "sklearn", "dcor", "joblib", "threadpoolctl", "lightgbm",
        }:
            raise
        parser.exit(1, "Install synthesis dependencies with python3 -m pip install 'jq-factor-lab[synthesis]'\n")
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
