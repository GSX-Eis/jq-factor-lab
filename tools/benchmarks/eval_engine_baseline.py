"""Benchmark the legacy evaluators on deterministic, precomputed labels."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import date
from importlib.metadata import version
import json
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
import platform
import resource
import signal
from statistics import median
import subprocess
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 791
PREPARATION_TIMEOUT_SECONDS = 180
NQ_COVERAGE = ("2020-01-01", "2025-12-31")
REFERENCE = "scipy_spearman_reference"
SCENARIOS = {
    "daily_cross_section": ("AlphalensRunner", (1,)),
    "minute_cross_section": ("BarCrossSectionalEvaluator", (1,)),
    "nq_time_series": ("TimeSeriesEvaluator", (1, 5, 15, 60)),
}


def _daily_inputs(
    rng: np.random.Generator, smoke: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    from backtest_and_analysis.forward_returns_calculator import ForwardReturnsCalculator
    from utils.trading_calendar import TradingCalendar

    days, assets = (12, 12) if smoke else (1_000, 2_000)
    dates = pd.bdate_range("2020-01-02", periods=days + 2, name="date")
    symbols = pd.Index([f"SYN{i:04d}" for i in range(assets)], name="asset")
    values = rng.standard_normal((days, assets))
    changes = rng.normal(0, 0.01, (days + 2, assets))
    changes[2:] += 0.003 * values
    prices = pd.DataFrame(100 * np.exp(np.cumsum(changes, axis=0)), index=dates, columns=symbols)
    pricing = prices.stack(future_stack=True).rename("close")
    calendar = TradingCalendar(dates, dates[0], dates[-1])
    returns = ForwardReturnsCalculator(trading_calendar=calendar).calculate_forward_returns(
        pricing, periods=(1,), execution_lag=1,
    )
    values.reshape(-1)[::113] = np.nan
    factor = pd.DataFrame(values, index=dates[:days], columns=symbols)
    return factor, pricing, returns


def _bar_inputs(
    rng: np.random.Generator, time_series: bool, smoke: bool, horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, int, int, dict]:
    from backtest_and_analysis.time_series.labels import BarReturnLabeler
    from utils.session_calendar import SessionCalendar
    from utils.time_axis import BarSchedule

    start, end = NQ_COVERAGE
    if smoke or not time_series:
        end = "2020-04-01"
    calendar = SessionCalendar("NQ", start, end)
    schedule = BarSchedule(calendar, "1min", view="ETH")
    grids = [pd.DatetimeIndex(schedule.for_session(day).start_utc) for day in calendar.sessions()]
    dates = grids[0].append(grids[1:])
    if smoke:
        dates = dates[:96 if time_series else 24]
    elif not time_series:
        dates = dates[:60_000]
    assets = 1 if time_series else (5 if smoke else 50)
    symbols = ["NQ"] if time_series else [f"SYN{i:02d}" for i in range(assets)]
    shape = (assets, len(dates))
    values = rng.standard_normal(shape)
    changes = rng.normal(0, 0.0003, shape)
    changes[:, 1:] += 0.0001 * values[:, :-1]
    closes = 100 * np.exp(np.cumsum(changes, axis=1))
    opens = np.empty_like(closes)
    opens[:, 0] = 100
    opens[:, 1:] = closes[:, :-1]
    values.reshape(-1)[::113] = np.nan
    # Keep instrument-major order, matching the evaluator's canonical keys.
    bars = pd.DataFrame({
        "timestamp_utc": pd.DatetimeIndex(np.tile(dates.asi8, assets), dtype=dates.dtype),
        "instrument_id": np.repeat(np.array(symbols, dtype=object), len(dates)),
        "frequency": "1min",
        "open": opens.ravel(),
        "high": np.maximum(opens, closes).ravel(),
        "low": np.minimum(opens, closes).ravel(),
        "close": closes.ravel(),
        "volume": 1.0,
        "source": "synthetic",
        "data_version": f"seed-{SEED}",
        "factor_value": values.ravel(),
    })
    labels = BarReturnLabeler(calendar, horizons=horizons, view="ETH").label(bars)
    # Every synthetic Bar is a signal for every horizon, including excluded labels.
    if len(labels) != len(bars) * len(horizons):
        raise RuntimeError("Synthetic bars did not produce all expected label rows")
    # The labeler sorts by instrument, signal time, then horizon.
    labels["factor_value"] = np.repeat(values.ravel(), len(horizons))
    metadata = {
        "calendar": calendar.calendar_id, "calendar_hash": calendar.calendar_hash,
        "calendar_coverage": [start, end], "view": "ETH", "boundary_policy": "same_session",
        "signal_start_utc": dates[0].isoformat(), "signal_end_utc": dates[-1].isoformat(),
    }
    return labels, len(dates), assets, metadata


def _metric(horizon: int, count: int, mean_ic: float, groups: int) -> dict:
    return {
        "holding_period": horizon, "valid_pairs": count, "ic_groups": groups,
        "mean_ic": float(mean_ic) if np.isfinite(mean_ic) else None,
    }


def _reference(pairs: pd.DataFrame, time_series: bool, horizons: tuple[int, ...]) -> list[dict]:
    def correlation(group: pd.DataFrame) -> float:
        return float(spearmanr(group["factor"], group["return"]).statistic)

    metrics = []
    for horizon in horizons:
        valid = pairs.loc[pairs.holding_period.eq(horizon), ["factor", "return"]].dropna()
        if time_series:
            value = correlation(valid)
            groups = int(np.isfinite(value))
        else:
            ic = valid.groupby(level="date")[["factor", "return"]].apply(correlation)
            value, groups = ic.mean(), int(ic.count())
        metrics.append(_metric(horizon, len(valid), value, groups))
    return metrics


def _prepare(
    scenario: str, method: str, smoke: bool,
) -> tuple[dict, Callable[[], list[dict]]]:
    evaluator, horizons = SCENARIOS[scenario]
    rng = np.random.default_rng(SEED)
    time_series = scenario == "nq_time_series"
    if scenario == "daily_cross_section":
        factor, pricing, returns = _daily_inputs(rng, smoke)
        timestamps, assets = factor.shape
        pairs = returns.rename(columns={"1D": "return"}).join(
            factor.stack(future_stack=True).rename("factor"), validate="one_to_one",
        ).assign(holding_period=1)
        valid_pairs = int(np.isfinite(pairs[["factor", "return"]].to_numpy()).all(axis=1).sum())
        finite_factors = int(np.isfinite(factor.to_numpy()).sum())
        if valid_pairs != finite_factors:
            raise RuntimeError(
                f"Daily benchmark requires a finite label for each finite factor: "
                f"{valid_pairs}/{finite_factors} pairs"
            )
        counts = {1: valid_pairs}
        metadata = {"calendar": "synthetic weekdays", "holding_period_unit": "sessions"}
        label_rows = len(pairs)

        def legacy() -> list[dict]:
            from backtest_and_analysis.alphalens_runner import AlphalensRunner

            # The runner verifies retained pairs against all finite factors.
            ic = AlphalensRunner(periods=horizons).run(
                factor, pricing, returns, max_loss=0,
            )["ic"]["1D"]
            return [_metric(1, counts[1], ic.mean(), int(ic.count()))]
    else:
        labels, timestamps, assets, metadata = _bar_inputs(rng, time_series, smoke, horizons)
        metadata["holding_period_unit"] = "1min bars"
        label_rows = len(labels)
        counts = {int(h): int(n) for h, n in labels.loc[
            labels.factor_value.notna() & labels.return_pct.notna(), "horizon_bars",
        ].value_counts().items()}
        if method == REFERENCE:
            pairs = labels[
                ["signal_timestamp", "factor_value", "return_pct", "horizon_bars"]
            ].set_index("signal_timestamp").rename_axis("date").rename(columns={
                "factor_value": "factor", "return_pct": "return", "horizon_bars": "holding_period",
            })

        def legacy() -> list[dict]:
            from backtest_and_analysis.time_series.evaluator import (
                BarCrossSectionalEvaluator, TimeSeriesEvaluator,
            )

            if time_series:
                result = TimeSeriesEvaluator(analysis_mode="time_series").evaluate(labels)
                rows = result.ic_by_horizon.query("scope == 'multi_instrument_summary'")
                ic_column, groups_column = "mean_spearman_ic", "valid_instrument_count"
            else:
                rows = BarCrossSectionalEvaluator(analysis_mode="cross_sectional").evaluate(
                    labels,
                ).ic_by_horizon
                ic_column, groups_column = "mean_ic", "valid_cross_sections"
            return [
                _metric(int(h), int(n), ic, int(groups))
                for h, n, ic, groups in rows[
                    ["horizon_bars", "valid_count", ic_column, groups_column]
                ].itertuples(index=False, name=None)
            ]

    if method == REFERENCE:
        def evaluate() -> list[dict]:
            return _reference(pairs, time_series, horizons)
    elif method == evaluator:
        # Imports belong to setup, including the legacy runner's plotting dependencies.
        if scenario == "daily_cross_section":
            import backtest_and_analysis.alphalens_runner
        evaluate = legacy
    else:
        raise ValueError(f"Unknown method {method} for {scenario}")
    if sorted(counts) != list(horizons):
        raise RuntimeError(f"No valid prepared pairs for a horizon in {scenario}")
    return {
        "input_size": {
            "rows": timestamps * assets, "time_points": timestamps, "assets": assets,
            "horizons": len(horizons), "label_rows": label_rows,
        },
        "input_metadata": metadata,
        "prepared_valid_pairs": counts,
    }, evaluate


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


class _EvaluationTimeout(BaseException):
    pass


_Prepare = Callable[[str, str, bool], tuple[dict, Callable[[], list[dict] | dict]]]


def _worker(
    connection: Connection, scenario: str, method: str, smoke: bool, timeout: float,
    prepare: _Prepare,
) -> None:
    def expire(_signum: int, _frame: Any) -> None:
        raise _EvaluationTimeout

    with connection:
        setup, evaluate = prepare(scenario, method, smoke)
        setup_peak = _peak_rss_bytes()
        connection.send((setup, setup_peak))
        signal.signal(signal.SIGALRM, expire)
        status, result = "complete", {"per_horizon": []}
        start = perf_counter()
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            result = evaluate()
        except _EvaluationTimeout:
            status = "timeout"
        finally:
            elapsed = perf_counter() - start
            signal.setitimer(signal.ITIMER_REAL, 0)
        connection.send({
            "status": status, "wall_seconds": elapsed, "peak_rss_bytes": _peak_rss_bytes(),
            "setup_peak_rss_bytes": setup_peak,
            **(result if isinstance(result, dict) else {"per_horizon": result}),
            "completed_stage": "evaluation" if status == "complete" else "prepared_inputs",
            "timeout_stage": "evaluation" if status == "timeout" else None,
        })


def _attempt(
    scenario: str, method: str, smoke: bool, timeout: float, *, prepare: _Prepare = _prepare,
) -> tuple[dict | None, dict]:
    context = get_context("spawn")
    reader, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_worker, args=(sender, scenario, method, smoke, timeout, prepare))
    worker.start()
    sender.close()
    try:
        with reader:
            if not reader.poll(PREPARATION_TIMEOUT_SECONDS):
                return None, {
                    "status": "timeout", "timeout_stage": "preparation", "completed_stage": None,
                    "wall_seconds": None, "peak_rss_bytes": None, "setup_peak_rss_bytes": None,
                    "per_horizon": [],
                }
            setup, setup_peak = reader.recv()
            start = perf_counter()
            if reader.poll(timeout + 5):
                return setup, reader.recv()
            return setup, {
                "status": "timeout", "wall_seconds": perf_counter() - start,
                "peak_rss_bytes": None, "setup_peak_rss_bytes": setup_peak, "per_horizon": [],
                "completed_stage": "prepared_inputs", "timeout_stage": "evaluation",
            }
    except EOFError as exc:
        raise RuntimeError(f"{scenario}/{method} benchmark worker exited unexpectedly") from exc
    finally:
        worker.join(timeout=1)
        if worker.is_alive():
            worker.kill()
            worker.join()
        worker.close()


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        hardware = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"], check=True, capture_output=True,
        )
        specs = json.loads(hardware.stdout)["SPHardwareDataType"][0]
        return specs.get("chip_type") or specs.get("cpu_type") or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def _report_metadata(smoke: bool, repeats: int, timeout: float) -> dict:
    return {
        "seed": SEED, "smoke": smoke, "repeats_requested": repeats, "timeout_seconds": timeout,
        "preparation_timeout_seconds": PREPARATION_TIMEOUT_SECONDS,
        "machine": {
            "cpu": _cpu_name(), "platform": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": version("scipy"),
            "alphalens_reloaded": version("alphalens-reloaded"),
            "pandas_market_calendars": version("pandas-market-calendars"),
        },
        "wall_measurement": "perf_counter around evaluation on precomputed labels, excluding worker startup and fixture preparation",
        "memory_measurement": "resource.RUSAGE_SELF.ru_maxrss in bytes, fresh worker per attempt; includes imports and fixture preparation",
        "scenarios": {},
    }


def run_benchmark(*, smoke: bool, repeats: int, timeout: float) -> dict:
    report = _report_metadata(smoke, repeats, timeout)
    for name, (evaluator, _) in SCENARIOS.items():
        setup = None
        baseline_metrics = None
        methods = {}
        for method in (evaluator, REFERENCE):
            attempts = []
            for repeat in range(repeats):
                print(f"{name} / {method} / {repeat + 1} of {repeats}", flush=True)
                current, attempt = _attempt(name, method, smoke, timeout)
                if current is not None:
                    if setup is not None and current != setup:
                        raise RuntimeError(f"Synthetic inputs changed between attempts for {name}")
                    setup = current
                attempts.append(attempt)
                if attempt["status"] == "timeout":
                    break
                metrics = attempt["per_horizon"]
                counts = {entry["holding_period"]: entry["valid_pairs"] for entry in metrics}
                if counts != setup["prepared_valid_pairs"] or any(
                    entry["mean_ic"] is None or entry["ic_groups"] <= 0 for entry in metrics
                ):
                    raise RuntimeError(f"{name}/{method} did not evaluate the expected pairs")
                if baseline_metrics is None:
                    baseline_metrics = metrics
                for actual, expected in zip(metrics, baseline_metrics, strict=True):
                    if (actual["ic_groups"] != expected["ic_groups"]
                            or abs(actual["mean_ic"] - expected["mean_ic"]) > 1e-12):
                        raise RuntimeError(f"{name}/{method} IC differs from the first completed run")
            complete = len(attempts) == repeats and attempts[-1]["status"] == "complete"
            methods[method] = {
                "status": "complete" if complete else "timeout",
                "median_wall_seconds": median(a["wall_seconds"] for a in attempts) if complete else None,
                "attempts": attempts,
            }
        if setup is None:
            setup = {"input_size": None, "input_metadata": None, "prepared_valid_pairs": None}
        report["scenarios"][name] = {**setup, "methods": methods}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run all three scenarios on small inputs")
    parser.add_argument("--repeats", type=int, default=3, help="Fresh worker processes per method")
    parser.add_argument("--timeout", type=float, default=180, help="Evaluation deadline in seconds")
    parser.add_argument("--output", type=Path, help="Override the dated JSON report path")
    args = parser.parse_args()
    if args.repeats < 1 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--repeats must be positive and --timeout must be finite and positive")
    report = run_benchmark(smoke=args.smoke, repeats=args.repeats, timeout=args.timeout)
    from utils.local_outputs import write_json

    name = "perf_baseline_smoke" if args.smoke else "perf_baseline"
    output = args.output or Path(f"reports/NERE-791/{name}_{date.today():%Y%m%d}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(output, flush=True)


if __name__ == "__main__":
    main()
