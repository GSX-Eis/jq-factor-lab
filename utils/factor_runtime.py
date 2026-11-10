"""Shared factor execution; this module is also inlined into JoinQuant bundles."""

import math
import pandas as pd


def factor_plan(name, requirements, dependencies, windows=None):
    entry = requirements.get("factors", requirements).get(name)
    if not isinstance(entry, dict) or not entry.get("requirements"):
        raise ValueError("factor_requirement_missing: " + name)
    reqs = entry["requirements"]
    freq = next(
        (f for f in ("1m", "5m", "15m", "30m", "60m", "daily") if f in reqs), None
    )
    if freq is None:
        raise ValueError("factor_frequency_missing: " + name)
    req = reqs[freq]
    window = (windows or {}).get(name, {}).get(freq)
    if window is None:
        window = int(
            math.ceil(
                req.get(
                    "min_bars",
                    req.get("min_days", 0)
                    * {"1m": 240, "5m": 48, "15m": 16, "30m": 8, "60m": 4, "daily": 1}[
                        freq
                    ],
                )
                * (1 + req.get("slack_ratio", 0))
            )
        )
    runtime = entry.get("runtime", {})
    window = int(runtime.get("window", window))
    fields = sorted(set(req.get("fields") or dependencies.get(name, {}).get(freq, [])))
    fields = [f for f in fields if f not in ("date", "code", "stock_code")]
    if window <= 0 or not fields:
        raise ValueError("factor_window_or_fields_missing: " + name)
    return freq, window, fields, runtime


def factor_cross_section(result, asof, output_rule="latest"):
    if result is None or result.empty:
        raise ValueError("factor_output_empty")
    timestamp = pd.Timestamp(asof)
    date_only = len(str(asof)) <= 10
    cutoff = timestamp.normalize()
    last_allowed = (
        cutoff + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        if date_only
        else timestamp
    )
    if isinstance(result, pd.Series) and not isinstance(result.index, pd.MultiIndex):
        if isinstance(result.index, pd.DatetimeIndex) or result.index.has_duplicates:
            raise ValueError("factor_output_schema_invalid")
        return result
    if isinstance(result, pd.Series):
        result = result.rename("value").to_frame()
    if isinstance(result.index, pd.DatetimeIndex):
        if result.index.has_duplicates or result.columns.has_duplicates:
            raise ValueError("factor_output_duplicate_key")
        result = result.sort_index(kind="mergesort")
        days = pd.to_datetime(result.index).normalize()
        eligible = days == cutoff if output_rule == "exact" else days <= cutoff
        eligible = eligible & (pd.to_datetime(result.index) <= last_allowed)
        selected = result.loc[eligible]
        if selected.empty:
            raise ValueError("factor_output_date_missing")
        return selected.iloc[-1]
    frame = result.reset_index()
    stock = next(
        (c for c in ("stock", "code", "stock_code", "security") if c in frame.columns),
        None,
    )
    if stock is None or "date" not in frame or "value" not in frame:
        raise ValueError("factor_output_schema_invalid")
    frame = frame[pd.to_datetime(frame["date"]) <= last_allowed].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = (
        frame[frame.date == cutoff]
        if output_rule == "exact"
        else frame[frame.date <= cutoff]
    )
    if frame.empty:
        raise ValueError("factor_output_date_missing")
    frame = frame[frame.date == frame.date.max()]
    if frame.duplicated(stock).any():
        raise ValueError("factor_output_duplicate_stock")
    return frame.set_index(stock)["value"]


def compute_factor_batch(
    classes,
    requirements,
    dependencies,
    stocks,
    asof,
    adapter_factory,
    windows=None,
    pool_history=None,
    cutoff=None,
    trace=None,
    window_override=None,
    return_history=False,
):
    """Share fetches, then give each factor its own window and computation universe."""
    if not stocks:
        return pd.DataFrame(index=stocks)
    if not classes or any(cls is None for cls in classes.values()):
        raise ValueError("factor_class_missing")
    cutoff = pd.Timestamp(cutoff or asof).normalize()
    known_pools = {
        str(k)[:10]: list(v)
        for k, v in (pool_history or {}).items()
        if pd.Timestamp(k).normalize() <= cutoff
    }
    groups = {}
    for name, cls in sorted(classes.items()):
        freq, window, fields, runtime = factor_plan(
            name, requirements, dependencies, windows
        )
        window = int(window_override or window)
        calendar_days = runtime.get("history_calendar_days")
        if calendar_days is not None:
            if freq != "daily" or isinstance(calendar_days, bool) or not isinstance(calendar_days, int) or calendar_days < 1:
                raise ValueError("invalid_factor_calendar_history: " + name)
            if window < calendar_days + 1:
                raise ValueError("factor_calendar_history_fetch_too_short: " + name)
        mode = runtime.get("computation_pool", "current")
        if mode not in ("current", "historical_union", "analysis_union"):
            raise ValueError("invalid_computation_pool: " + mode)
        if mode != "current" and not known_pools:
            raise ValueError("missing_pool_history_for_cutoff")
        # Future pool membership is never part of a live computation universe.
        universe = sorted(
            set(stocks)
            if mode == "current"
            else {s for values in known_pools.values() for s in values}
        )
        pools = {} if runtime.get("pool_filter") == "none" else known_pools
        key = (
            freq,
            tuple(universe),
            int(runtime.get("min_full_day_tolerance", 0)),
            bool(runtime.get("pit_enabled", True)),
        )
        groups.setdefault(key, []).append((name, cls, window, fields, runtime, pools))
    output = pd.DataFrame(index=stocks)
    histories = {}
    for (freq, universe, tolerance, pit_enabled), plans in groups.items():
        largest = max(plan[2] for plan in plans)
        fields = sorted({f for plan in plans for f in plan[3]})
        adapter = adapter_factory(
            dependencies,
            freq=freq,
            window=largest,
            strict=True,
            allow_field_fallback=False,
            min_full_day_tolerance=tolerance,
        )
        adapter.pit_enabled = pit_enabled
        data = adapter.fetch_price(list(universe), str(asof), fields)
        if data is None or data.empty:
            raise ValueError("factor_input_empty")
        missing = set(fields) - set(data.columns)
        if missing:
            raise ValueError(
                "factor_input_fields_missing: " + ",".join(sorted(missing))
            )
        data = data.sort_values(["code", "date"], kind="mergesort")
        if data.duplicated(["code", "date"]).any():
            raise ValueError("factor_input_duplicate_key")
        timestamp = pd.Timestamp(asof)
        allowed = (
            timestamp.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            if len(str(asof)) <= 10
            else timestamp
        )
        if (pd.to_datetime(data.date) > allowed).any():
            raise ValueError("factor_input_future_data")
        for name, cls, window, required, runtime, pools in plans:
            bars = {"1m": 240, "5m": 48, "15m": 16, "30m": 8, "60m": 4, "daily": 1}[
                freq
            ]
            count = int(math.ceil(float(window) / bars)) * bars
            frame = data.groupby("code", sort=False).tail(count).copy()
            calendar_days = runtime.get("history_calendar_days")
            if calendar_days is not None:
                # The class's calendar warm-up must not become a trading-day seed.
                first = pd.Timestamp(asof).normalize() - pd.Timedelta(days=calendar_days)
                frame = frame.loc[pd.to_datetime(frame["date"]) >= first].copy()
            # A shared price fetch must not leak adjustment placeholders into
            # unrelated fundamental-only factors and their input evidence.
            price_fields = {"open", "high", "low", "close", "volume", "pre_close",
                            "avg", "high_limit", "low_limit"}
            extra_factor = {"factor"} if "factor" in frame and set(required) & price_fields else set()
            frame = frame[
                ["date", "code"]
                + sorted(set(required) | extra_factor)
            ]
            # A single-factor group no longer needs the fetched full-frame copy.
            if len(plans) == 1:
                data = None
            adapter.window = window
            try:
                result = adapter.run_factor(
                    cls,
                    frame,
                    monthly_stock_pools=pools or None,
                    output_date=asof if runtime.get("output_date") == "exact" else None,
                )
                if trace is not None:
                    trace(name, frame, result)
                histories[name] = result
                output[name] = factor_cross_section(
                    result, asof, runtime.get("output_date", "latest")
                ).reindex(stocks)
            except Exception as exc:
                raise ValueError(
                    "factor_execution_failed: %s asof=%s frequency=%s window=%s "
                    "rows=%s range=%s..%s: %s"
                    % (
                        name,
                        asof,
                        freq,
                        window,
                        len(frame),
                        frame.date.min(),
                        frame.date.max(),
                        exc,
                    )
                ) from exc
    if return_history:
        if len(histories) != 1:
            raise ValueError("history_output_requires_one_factor")
        return next(iter(histories.values()))
    return output
