"""Run offline demos or evaluate trusted factors against local market data."""
import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout
import ctypes
from importlib.util import find_spec
from collections.abc import Sequence
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any

from utils.local_outputs import check_output_directory


INSTALL = "python3 -m pip install 'jq-factor-lab{extra}'"
NEXT_ACTIONS = {
    "single-factor-analysis": "Inspect ic_observations and the evaluation CSVs.",
    "daily-demo": "Try single-factor-analysis with a local CSV and a trusted factor.",
    "bar-demo": "Pass metrics.results_db.path to read-bar-evaluations --results-db.",
    "time-series-analysis": "Inspect calibration_status and diagnostics before drawing conclusions.",
    "read-bar-evaluations": "Use dashboard --results-db to inspect the results interactively.",
    "factor-synthesis-demo": "Inspect scores.csv and summary.json in output_dir.",
    "dashboard": "Open the local dashboard URL printed on stderr.",
}


class CommandError(Exception):
    def __init__(self, code: str, message: str, fix: str, exit_code: int = 2):
        super().__init__(message)
        self.code, self.fix, self.exit_code = code, fix, exit_code


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CommandError("invalid_arguments", message, f"Run {self.prog} --help.")


def _readable_file(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist or is not a regular file: {path}")
    with path.open("rb"):
        pass
    return path.resolve()


def _require_extra(extra: str, *modules: str) -> None:
    missing = [module for module in modules if find_spec(module) is None]
    if missing:
        raise CommandError("missing_optional_dependency", f"Missing [{extra}] dependencies: {', '.join(missing)}",
                           INSTALL.format(extra=f"[{extra}]"), exit_code=1)


def _plan(**fields: Any) -> dict[str, Any]:
    return {"dry_run": True, "plan": fields}


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
    config_path: Path, output_dir: Path | None = None, *, overwrite: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    import duckdb
    import pandas as pd
    from backtest_and_analysis.alphalens_runner import AlphalensRunner, InsufficientCrossSectionAssets
    from backtest_and_analysis.factor_loader import FactorLoader
    from utils.local_outputs import new_run_directory, stage_demo_output, write_json

    config_path = _readable_file(Path(config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
    target = Path(output_dir) if output_dir is not None else None
    if target is None and "output_dir" in config:
        target = config_path.parent / config["output_dir"]
    if target is not None:
        target = check_output_directory(target, overwrite=overwrite)
    from utils.trading_calendar import load_trading_calendar

    keys = [key for key in ("calendar_file", "calendar_duckdb_path") if config.get(key)]
    if len(keys) != 1:
        raise ValueError("Set exactly one of calendar_file or calendar_duckdb_path; prices do not define sessions")
    calendar_key = keys[0]
    if not isinstance(config[calendar_key], str) or not config[calendar_key].strip():
        raise ValueError(f"{calendar_key} must be a non-empty path")
    calendar_path = _readable_file(relative(calendar_key))
    calendar = (_load_calendar_file(calendar_path) if calendar_key == "calendar_file"
                else load_trading_calendar(duckdb_path=calendar_path))
    prices = _readable_file(relative("input_csv"))
    frame = read_prices(prices)
    assets = int(frame.stock_code.nunique())
    if assets < 2:
        raise InsufficientCrossSectionAssets()
    calendar.validate_coverage(frame.date.min(), frame.date.max())
    factor_path = _readable_file(relative("factor_file"))
    if dry_run:
        if factor_path.suffix != ".py" or not config["factor_class"].isidentifier():
            raise ValueError("factor_file must be a .py file and factor_class a Python class name")
        compile(factor_path.read_bytes(), str(factor_path), "exec")
        return _plan(config=str(config_path), input_csv=str(prices), factor_file=str(factor_path),
                     factor=config["factor_class"], rows=len(frame), assets=assets,
                     calendar_id=calendar.calendar_id, calendar_sha256=calendar.fingerprint,
                     periods=periods, execution_lag=lag, n_quantiles=quantiles,
                     **_output_plan(target, Path("outputs/daily"), overwrite))
    from tools.agent_harness.causal_gate import CausalFastPathGateError, verify_factor_causality
    from utils.factor_base import FactorBase

    classes = FactorLoader._load_classes_from_file(str(factor_path), config["factor_class"])
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
        if isinstance(factor, FactorBase):
            gate = verify_factor_causality(factor, on_failure="reject", apply_to_factor=True)
            if not gate.ok:
                raise CausalFastPathGateError(
                    f"Causal fast-path gate rejected {gate.factor_name}: {gate.failure_reason}",
                    gate_result=gate.to_dict(),
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


def _load_calendar_file(path: Path):
    from utils.trading_calendar import TradingCalendar

    spec = json.loads(path.read_text(encoding="utf-8"))
    return TradingCalendar(spec["sessions"], spec["start_date"], spec["end_date"], spec["calendar_id"])


def _output_plan(target: Path | None, root: Path, overwrite: bool) -> dict[str, Any]:
    if target is None:
        return {"output_root": str(check_output_directory(root, overwrite=True)), "overwrite": overwrite}
    return {"output_dir": str(target), "overwrite": overwrite}


def _daily_demo(output_dir: Path | None, *, overwrite: bool, dry_run: bool) -> dict[str, Any]:
    from examples.generate_data import make_prices

    root = Path(__file__).resolve().parents[1]
    calendar_file = root / "config/demo_calendar.json"
    factor_file = _readable_file(root / "examples/momentum.py")
    target = check_output_directory(output_dir, overwrite=overwrite) if output_dir is not None else None
    output = _output_plan(target, Path("outputs/daily"), overwrite)
    frame = make_prices()
    if dry_run:
        calendar = _load_calendar_file(calendar_file)
        calendar.validate_coverage(frame.date.min(), frame.date.max())
        return _plan(data="synthetic", factor="DemoMomentum", calendar_id=calendar.calendar_id,
                     rows=len(frame), assets=int(frame.stock_code.nunique()), **output)
    with TemporaryDirectory(prefix="jq-factor-lab-daily-") as directory:
        input_dir = Path(directory)
        frame.to_csv(input_dir / "prices.csv", index=False)
        config = input_dir / "config.json"
        config.write_text(json.dumps({
            "input_csv": "prices.csv", "factor_file": str(factor_file),
            "factor_class": "DemoMomentum", "calendar_file": str(calendar_file),
        }), encoding="utf-8")
        return analyze(config, target, overwrite=overwrite)


def _bar_command(argv: list[str]) -> dict[str, Any]:
    from tools.agent_harness.parser import build_parser
    from tools.agent_harness.handlers.time_series import (
        _handle_read_bar_evaluations, _handle_time_series_analysis,
    )

    args = build_parser(public_bar_cli=True, parser_class=JsonArgumentParser).parse_args(argv)
    if args.command == "time-series-analysis":
        with ExitStack() as output:
            result = _handle_time_series_analysis(args, public_output=output)
    else:
        result = _handle_read_bar_evaluations(args, public_read=True)

    if args.dry_run and result.ok:
        return _plan(**result.metrics)
    return result.to_dict()


def _bar_demo(output_dir: Path | None, *, overwrite: bool, dry_run: bool) -> dict[str, Any]:
    from examples.synthetic_bars import demo_bar_config, write_demo_bars
    from utils.local_outputs import new_run_directory, write_json

    root = Path("outputs/bar-demo")
    target = check_output_directory(output_dir, overwrite=overwrite) if output_dir is not None else None
    run_root = check_output_directory(root, overwrite=True)
    if target is not None and run_root.is_relative_to(target) and not overwrite:
        raise ValueError("Bar demo creates its scratch under this output directory; choose another --output-dir or pass --overwrite")
    if dry_run:
        from utils.session_calendar import SessionCalendar

        config = demo_bar_config()
        calendar = SessionCalendar(config["calendar"], config["start_date"], config["end_date"])
        if calendar.sessions().empty:
            raise ValueError("Demo calendar has no sessions in the demo window")
        return _plan(data="synthetic", calendar_id=calendar.calendar_id,
                     results_db="evaluations.duckdb (inside the isolated run)", **_output_plan(target, run_root, overwrite),
                     run_root=str(run_root))
    run = new_run_directory(run_root)
    config = run / "config.json"
    write_json(config, write_demo_bars(run / "bars.csv"))
    argv = [
        "time-series-analysis", "--config", str(config), "--diagnostics-report",
        "--output-root", str(run / "harness"), "--results-db", str(run / "evaluations.duckdb"),
    ]
    if target is not None:
        argv.extend(["--output-dir", str(target)])
    if overwrite:
        argv.append("--overwrite")
    return _bar_command(argv)


def _dashboard(results_db: str | None, port: int, *, dry_run: bool) -> dict[str, Any]:
    if not 1 <= port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    _require_extra("viz", "streamlit")
    from utils import bar_evaluation_app

    db_args = ["--results-db", results_db] if results_db is not None else []
    path = _readable_file(bar_evaluation_app.resolve_results_db_path(db_args))
    command = [sys.executable, "-m", "streamlit", "run", str(Path(bar_evaluation_app.__file__).resolve()),
               "--server.port", str(port), "--server.address", "127.0.0.1", "--server.headless", "true",
               "--browser.gatherUsageStats", "false", "--", "--results-db", str(path)]
    plan = {"command": command, "results_db": str(path), "port": port, "read_only": True}
    if dry_run:
        return _plan(**plan)
    try:
        subprocess.run(command, check=True, stdout=sys.stderr)
    except subprocess.CalledProcessError as exc:
        raise CommandError("dashboard_failed", f"Dashboard exited with status {exc.returncode}",
                           "Check the Streamlit logs or choose a different --port.", exit_code=1) from exc
    return plan


def _parse_args(argv: list[str]) -> argparse.Namespace:
    from examples.synthesis_demo import METHODS

    parser = JsonArgumentParser(prog="jq-factor-lab", description=__doc__)
    parser.add_argument("--compact", action="store_true", help="Print single-line JSON (also accepted after the command)")
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single-factor-analysis", help="Evaluate a daily multi-asset CSV")
    single.add_argument("--config", type=Path, required=True)
    daily = commands.add_parser("daily-demo", help="Evaluate fictional daily prices across multiple assets")
    bar = commands.add_parser("bar-demo", help="Evaluate synthetic minute Bars for one instrument")
    commands.add_parser("time-series-analysis", help="Evaluate local Bar data using the Harness")
    commands.add_parser("read-bar-evaluations", help="Read a Bar evaluation DuckDB")
    synthesis = commands.add_parser("factor-synthesis-demo", help="Run a synthetic factor synthesis example")
    synthesis.add_argument("--method", choices=METHODS, default="weighted_sum",
                           help="weighted_sum runs directly; other methods require verified forward-label exits (see docs/SYNTHESIS.md)")
    dashboard = commands.add_parser("dashboard", help="Launch the read-only Bar results dashboard")
    dashboard.add_argument("--results-db", help="Results DuckDB (default: QUANT_BAR_RESULTS_DB)")
    dashboard.add_argument("--port", type=int, default=8501)
    for command in (single, daily, bar, synthesis, dashboard):
        command.add_argument("--dry-run", action="store_true", help="Validate inputs and print a plan")
    for command in (single, daily, bar, synthesis):
        command.add_argument("--output-dir", type=Path, help="Explicit directory (default: a new isolated run or the configured path)")
        command.add_argument("--overwrite", action="store_true", help="Update generated files in an existing output directory")
    return parser.parse_args(argv)


def _dispatch(argv: list[str]) -> tuple[str, dict[str, Any]]:
    if argv and argv[0] in {"time-series-analysis", "read-bar-evaluations"}:
        return argv[0], _bar_command(argv)

    args = _parse_args(argv)
    if args.command == "dashboard":
        return args.command, _dashboard(args.results_db, args.port, dry_run=args.dry_run)
    options = {"overwrite": args.overwrite, "dry_run": args.dry_run}
    if args.command == "bar-demo":
        result = _bar_demo(args.output_dir, **options)
    elif args.command == "daily-demo":
        result = _daily_demo(args.output_dir, **options)
    elif args.command == "factor-synthesis-demo":
        from examples.synthesis_demo import run_demo

        try:
            if args.dry_run and args.method == "weighted_sum":
                _require_extra("synthesis", "sklearn", "dcor", "joblib", "threadpoolctl")
            if args.output_dir is not None:
                args.output_dir = check_output_directory(args.output_dir, overwrite=args.overwrite)
            result = run_demo(args.method, args.output_dir, **options)
            if args.dry_run:
                result["plan"].update(_output_plan(args.output_dir, Path("outputs/synthesis"), args.overwrite))
        except ModuleNotFoundError as exc:
            if exc.name not in {"sklearn", "dcor", "joblib", "threadpoolctl", "lightgbm", "matplotlib", "seaborn"}:
                raise
            raise CommandError("missing_optional_dependency", str(exc),
                               INSTALL.format(extra="[synthesis]"), exit_code=1) from exc
    else:
        result = analyze(args.config, args.output_dir, **options)
    return args.command, result


@contextmanager
def _logs_to_stderr():
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            try:
                yield
            finally:
                # Flush C stdio while fd 1 still points at the log stream.
                ctypes.CDLL(None).fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def _error_result(exit_code: int, code: str, message: str, fix: str) -> dict[str, Any]:
    return {"ok": False, "exit_code": exit_code, "error": {
        "code": code, "message": message, "fix": fix, "retryable": False,
    }}


def main(argv: Sequence[str] | None = None) -> int:
    from utils.local_outputs import json_compatible

    argv = list(sys.argv[1:] if argv is None else argv)
    end = argv.index("--") if "--" in argv else len(argv)
    compact = "--compact" in argv[:end]
    # argparse prints help directly. Operational commands reserve stdout for JSON.
    help_requested = any(arg in {"-h", "--help"} for arg in argv[:end])
    argv = [arg for arg in argv[:end] if arg != "--compact"] + argv[end:]
    logging.basicConfig(level=logging.WARNING)
    fix = "Check input paths, configuration and permissions, then run the command again."
    try:
        if help_requested:
            # Both parsers stop at --help before any handler runs.
            name, result = _dispatch(argv)
        else:
            with _logs_to_stderr():
                name, result = _dispatch(argv)
        if result.get("ok") is False:
            error = result["error"]
            error.setdefault("fix", fix)
            error.setdefault("retryable", False)
            result["exit_code"] = 2 if result["exit_code"] == 2 else 1
        else:
            next_action = ("Run again without --dry-run to execute this plan."
                           if result.get("dry_run") else NEXT_ACTIONS[name])
            result.update(ok=True, exit_code=0, next_actions=[next_action])
    except CommandError as exc:
        result = _error_result(exc.exit_code, exc.code, str(exc), exc.fix)
    except KeyboardInterrupt:
        result = _error_result(1, "interrupted", "Command interrupted.", "Run the command again when ready.")
    except SystemExit as exc:
        if help_requested and exc.code == 0:
            return 0
        result = _error_result(1, "analysis_failed", f"Analysis exited early ({exc.code}).", fix)
    except ImportError as exc:
        result = _error_result(1, "missing_dependency", str(exc), INSTALL.format(extra=""))
    except Exception as exc:
        with _logs_to_stderr():
            from backtest_and_analysis.alphalens_runner import InsufficientCrossSectionAssets, MaxLossExceededError
            from tools.agent_harness.causal_gate import CausalFastPathGateError

        if isinstance(exc, InsufficientCrossSectionAssets):
            exit_code, code = 1, exc.code
            fix = "Supply two or more assets or use time-series-analysis."
        elif isinstance(exc, CausalFastPathGateError):
            exit_code, code = 1, exc.code
            fix = "Remove future/lookahead references from calculate_factor() so it uses only data available at the evaluation timestamp."
        elif isinstance(exc, MaxLossExceededError):
            exit_code, code = 1, "analysis_failed"
        elif isinstance(exc, (ValueError, TypeError, KeyError, FileNotFoundError, FileExistsError, PermissionError, SyntaxError)):
            exit_code, code = 2, "invalid_input"
        else:
            exit_code, code = 1, "analysis_failed"
        result = _error_result(exit_code, code, str(exc) or type(exc).__name__, fix)
    if not result["ok"]:
        error = result["error"]
        lines = error["message"].partition("Traceback (most recent call last):")[0].strip().splitlines()
        message = lines[0] if lines else "Command failed."
        print(f"Error: {message} {error['fix']}", file=sys.stderr)
    if compact:
        for key in ("stdout_tail", "stderr_tail"):
            result.pop(key, None)
    print(json.dumps(json_compatible(result), ensure_ascii=False, allow_nan=False,
                     default=str, indent=None if compact else 2, separators=(",", ":") if compact else None))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
