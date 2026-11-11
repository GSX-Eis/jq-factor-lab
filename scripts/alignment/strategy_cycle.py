"""Frozen strategy replay using the existing VectorBT and JoinQuant engines."""

from __future__ import annotations

import ast
import base64
import gzip
import json
import sys
import math
import types
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.alignment.evidence import (
    input_manifest,
    file_hashes,
    intact,
    intact_inputs,
    archive_previous,
)
from scripts.alignment.snapshots import read_records, compare_records, STAGES
from utils import strategy_kernel as kernel
from utils.jq_factor_adapter import rolling_money_flow_sum

ROOT = Path(__file__).resolve().parents[2]


def load_json(path):
    path = Path(path)
    return json.loads(
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _prepare_stage_outputs(expected, replace):
    if replace:
        for path in expected:
            path.unlink(missing_ok=True)


def literals(path, names):
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        ):
            result[node.targets[0].id] = ast.literal_eval(node.value)
    return result


def load_wind_ranker(source, deploy, rank):
    # Load just the frozen tree evaluator and signal functions, without platform imports.
    namespace = {"np": np, "pd": pd, "math": math}
    tree_source = Path(deploy).read_text()
    for node in ast.parse(tree_source).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_tree_value",
            "predict_record",
        ):
            exec(ast.get_source_segment(tree_source, node), namespace)
    namespace["EN_BUNDLE"] = literals(deploy, {"EN_BUNDLE"})["EN_BUNDLE"]
    env = {
        "np": np,
        "pd": pd,
        "shared": kernel,
        "deploy": types.SimpleNamespace(**namespace),
        "g": types.SimpleNamespace(rank_bundle=load_json(rank)),
    }
    source = Path(source).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            try:
                exec(ast.get_source_segment(source, node), env)
            except (NameError, AttributeError):
                pass
        elif isinstance(node, ast.FunctionDef) and node.name in ("_rank", "_en_model"):
            exec(ast.get_source_segment(source, node), env)
    return env


def local_adapter_factory(data, contract):
    from utils.jq_factor_adapter import JQFactorAdapter
    from utils.pit import apply_pit_adjustment

    class LocalAdapter(JQFactorAdapter):
        def fetch_price(self, stocks, asof, fields):
            bars = self._bars_per_day(self.freq)
            days = int(np.ceil(self.window / bars))
            calendar = data.trading_calendar()
            beginning = calendar.previous_sessions(asof, days, include_anchor=True)[0].strftime('%Y-%m-%d')
            by_table = {}
            for field in fields:
                if field not in contract:
                    raise ValueError("field_contract_missing: " + field)
                source = contract[field]["local"]
                by_table.setdefault(
                    source["table"].replace("{frequency}", self.freq), []
                ).append((field, source["column"]))
            rolling_history = max([0] + [
                int(contract[field]['local'].get('rolling_days', 1)) - 1 for field in fields])
            if rolling_history:
                beginning = calendar.previous_sessions(asof, days + rolling_history,
                                                       include_anchor=True)[0].strftime('%Y-%m-%d')
            if (bars == 1 and not any(table.startswith("price_data_") for table in by_table)
                    and any("rolling_days" in contract[field]["local"] for field in fields)):
                # Match the extra-only cloud fetch before rolling: earlier rows
                # can change pandas' compensated-sum state even after warmup.
                beginning = str((pd.Timestamp(asof) - pd.Timedelta(days=days * 2 + 30)).date())
            pieces = []
            for table, mapping in sorted(
                by_table.items(),
                key=lambda item: (not item[0].startswith("price_data_"), item[0]),
            ):
                requested_columns = {col for _, col in mapping}
                amount_column = next((col for field, col in mapping if field == "money"), None)
                normalise_paused_amount = self.freq == "daily" and table.startswith("price_data_") and amount_column is not None
                if normalise_paused_amount:
                    requested_columns.add("paused")
                frame = data.fetch_table_data(
                    table, stocks, beginning, asof, sorted(requested_columns)
                )
                missing = requested_columns - set(frame.columns)
                if frame.empty or missing:
                    raise ValueError(
                        "local_input_missing: "
                        + table
                        + ":"
                        + ",".join(sorted(missing))
                    )
                if normalise_paused_amount:
                    from utils.pit import normalize_price_money
                    amount = frame[[amount_column, 'paused']].rename(columns={amount_column: 'money'})
                    frame[amount_column] = normalize_price_money(
                        amount, source='local', frequency=self.freq)['money']
                derived = any("rolling_days" in contract[field]["local"] for field, _ in mapping)
                if derived:
                    if frame.duplicated(["date", "code"]).any():
                        raise ValueError("local_input_duplicate_key: " + table)
                    frame = frame.sort_values(["code", "date"], kind="mergesort")
                values = {}
                for field, col in mapping:
                    source = contract[field]["local"]
                    if "rolling_days" in source:
                        window = source["rolling_days"]
                        minimum = source.get("min_periods", window)
                        if (not isinstance(window, int) or isinstance(window, bool)
                                or window < 1 or not isinstance(minimum, int)
                                or isinstance(minimum, bool) or not 0 <= minimum <= window):
                            raise ValueError("local_rolling_contract_invalid: " + field)
                        values[field] = frame.groupby("code")[col].transform(
                            lambda series: rolling_money_flow_sum(series, window, minimum)
                        )
                    else:
                        values[field] = frame[col]
                frame = frame[["date", "code"]].assign(**values)
                pieces.append(frame)
            frame = pieces[0]
            for other in pieces[1:]:
                if bars > 1:
                    frame = frame.assign(_day=pd.to_datetime(frame.date).dt.normalize())
                    other = other.rename(columns={"date": "_day"})
                    frame = frame.merge(
                        other, on=["_day", "code"], how="left", validate="many_to_one"
                    ).drop(columns="_day")
                else:
                    frame = frame.merge(
                        other, on=["date", "code"], how="outer", validate="one_to_one"
                    )
            frame = frame.sort_values(["code", "date"], kind="mergesort")
            frame = frame.groupby("code", sort=False).tail(days * bars).copy()
            if bars > 1:
                frame = self._trim_to_full_days(frame, bars)
            if self.pit_enabled and "factor" in frame:
                frame = apply_pit_adjustment(frame)
            return frame

    LocalAdapter.duckdb_path = str(data.duckdb_path) if data.duckdb_path else None
    LocalAdapter.data_bindings = getattr(data, "data_bindings", None)
    return LocalAdapter


def run_local(config, output, fingerprint, bundle):
    import duckdb
    from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter
    from utils.duckdb_adapter import DuckDBAdapter
    from utils.factor_runtime import compute_factor_batch
    from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner

    start, end = config["start"], config["end"]
    db = resolve(config["database"])
    local_config = load_json(resolve(config["local_config"]))
    contract = load_json(ROOT / "config/cloud_api_mapping.json")["field_contracts"]
    build = literals(
        bundle,
        {"FACTOR_DEPENDENCIES", "FACTOR_REQUIRED_WINDOWS", "FACTOR_REQUIREMENTS"},
    )
    from backtest_and_analysis.factor_loader import FactorLoader

    classes = {}
    for name, raw_path in config["factors"].items():
        _, module = FactorLoader._load_module_from_path(resolve(raw_path), serial=True)
        classes[name] = getattr(module, name)
    bindings = local_config.get("data_bindings")
    data = DuckDBAdapter(duckdb_path=str(db), use_cache=False, data_bindings=bindings)

    LocalAdapter = local_adapter_factory(data, contract)

    runner = None
    from utils.data_store import connect_input
    connection = connect_input(str(db), bindings=bindings)
    try:
        from utils.trading_calendar import load_trading_calendar
        market_calendar = load_trading_calendar(connection)
        previous = market_calendar.previous_sessions(start, 1)[0]
        calendar = market_calendar.sessions(previous, end).strftime('%Y-%m-%d').tolist()
        trade_dates = [d for d in calendar if d >= start]
        if not trade_dates:
            raise ValueError("local_trading_calendar_empty")
        if config.get("pool_snapshot"):
            pools = load_json(resolve(config["pool_snapshot"]))
        else:
            pool_config = local_config["stock_pool"]
            manager = DynamicStockPoolManager(
                None,
                price_table="price_data_daily",
                db_backend="duckdb",
                duckdb_path=str(db),
                use_duckdb_cache=False,
                data_bindings=bindings,
            )
            if pool_config.get("kind") == "wind_micro_w11":
                raise ValueError("pool kind wind_micro_w11 requires the private research extension")

                for table in (
                    "valuation_daily",
                    "security_master",
                    "factor_data",
                    "historical_st_daily",
                    "price_data_daily",
                ):
                    connection.execute(
                        "CREATE TEMP VIEW " + table + "_all AS SELECT * FROM " + table
                    )
                from datetime import datetime

                days = manager.get_weekly_run_dates(
                    datetime.fromisoformat(start),
                    datetime.fromisoformat(end),
                    target_day_index=3,
                )
                pools = {
                    str(day)[:10]: stock_pool(
                        connection,
                        str(day)[:10],
                        calendar[calendar.index(str(day)[:10]) - 1],
                    )[0]
                    for day in days
                }
            else:
                pools = manager.get_stock_pools_for_period(
                    start,
                    end,
                    [StockFilter(**f) for f in pool_config["filters"]],
                    max_stocks=pool_config.get("max_stocks"),
                    rebalance_schedule=pool_config.get("rebalance_schedule", "weekly"),
                )
        pools = {day: stocks for day, stocks in pools.items() if day in trade_dates}
        if not pools:
            raise ValueError("local_pool_empty")
        stocks = sorted({s for values in pools.values() for s in values})
        market = connection.execute(
            "SELECT * FROM price_data_daily WHERE stock_code IN (SELECT unnest(?)) AND CAST(date AS DATE) BETWEEN ? AND ?",
            [stocks, calendar[0], end],
        ).fetchdf()
        market.to_parquet(output / "local_market_snapshot.parquet", index=False)
        (output / "local_pool_snapshot.json").write_text(json.dumps(pools))
        market["date"] = pd.to_datetime(market.date).dt.normalize()

        def wide(field):
            return market.pivot(
                index="date", columns="stock_code", values=field
            ).reindex(columns=stocks)

        prices, opens = wide("close"), wide("open")
        scores_frame = pd.DataFrame(index=prices.index, columns=stocks, dtype=float)
        raw_records = {}
        warmup_dates = []
        pool_history = {}
        wind = (
            load_wind_ranker(
                resolve(config["source"]),
                resolve(config["models"]["deploy"]),
                resolve(config["models"]["rank"]),
            )
            if config["strategy"] == "wind_micro_w11"
            else None
        )
        models = load_json(resolve(config["models"]["lasso"])) if wind is None else None
        from factor_synthesis.core.calendar_skip import skip_mask, parse_skip_rules

        snapshots = []
        for day, pool in sorted(pools.items()):
            if bool(
                skip_mask(
                    pd.DatetimeIndex([day]),
                    parse_skip_rules(
                        local_config.get("calendar_filters", {}).get("skip_months", [])
                    ),
                )[0]
            ):
                continue
            factor_date = calendar[calendar.index(day) - 1]
            pool_history[factor_date] = pool
            if wind and wind["_en_model"](factor_date) is None:
                warmup_dates.append(day)
                print(
                    json.dumps(
                        {
                            "event": "model_not_available",
                            "date": day,
                            "asof": factor_date,
                        }
                    ),
                    flush=True,
                )
                continue
            hashes = {}

            def trace(name, frame, result):
                hashes[name] = kernel.input_digest(frame)
                destination = output / "inputs" / day
                destination.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(destination / (name + ".parquet"), index=False)

            active_classes = classes
            if not wind:
                model = kernel.model_for_date(models["index"]["models"], factor_date)
                if model is None:
                    raise ValueError("model_not_available: " + factor_date)
                suffix = config.get("factor_suffix", "")
                needed = set(model.get("weights", {})) | set(
                    model.get("feature_names", [])
                )
                if model.get("blend_with_original_ratio", 0) > 0:
                    needed.update(model.get("blend_weights", {}))
                bare = {
                    name[: -len(suffix)] if suffix and name.endswith(suffix) else name
                    for name in needed
                }
                active_classes = {name: classes[name] for name in bare}
            raw = compute_factor_batch(
                active_classes,
                build["FACTOR_REQUIREMENTS"],
                build["FACTOR_DEPENDENCIES"],
                pool,
                factor_date,
                LocalAdapter,
                windows=build["FACTOR_REQUIRED_WINDOWS"],
                pool_history=pool_history,
                trace=trace,
            )
            if wind:
                ranking, raw, _, _ = wind["_rank"](pool, raw, factor_date, day)
                score = ranking.set_index("stock").final_score
                processed = wind["g"].alignment_signal["processed"]
            else:
                model = kernel.model_for_date(models["index"]["models"], factor_date)
                if model is None:
                    raise ValueError("model_not_available: " + factor_date)
                suffix = config.get("factor_suffix", "")
                raw = raw.rename(columns={name: name + suffix for name in raw})
                score, processed = kernel.linear_signal(raw, model)
            scores_frame.loc[pd.Timestamp(factor_date), score.index] = score
            raw_records[day] = (factor_date, pool, raw, processed, score, hashes)
            print(
                json.dumps({"event": "local_signal", "date": day, "pool": len(pool)}),
                flush=True,
            )
        engine_config = dict(local_config)
        engine_config["top_k_strategy"] = dict(
            local_config["top_k_strategy"], score_source="input_factor"
        )
        runner = VectorBTBacktestRunner(
            engine_config,
            price_db_path=str(db),
            price_db_conn=connection,
            save_alignment_data=True,
            alignment_output_dir=str(output),
            experiment_name="local_stages",
            diagnostics_output_dir=str(output),
            return_series_trial_id="alignment",
        )
        result = runner.run_backtest(
            scores_frame,
            prices,
            open_df=opens,
            factor_ratio_df=wide("factor") / wide("factor").shift(1),
            dynamic_pools=pools,
            rebalance_dates=[pd.Timestamp(d) for d in pools],
        )
        if "portfolio" not in result:
            raise RuntimeError(
                "local_execution_failed: "
                + str(result.get("error", "portfolio_missing"))
            )
        targets = {
            s.rebalance_date: s
            for s in runner._alignment_writer.snapshots
        }
        for day, (
            factor_date,
            pool,
            raw,
            processed,
            score,
            hashes,
        ) in raw_records.items():
            if day not in targets:
                raise ValueError("local_target_snapshot_missing: " + day)
            snapshots.extend(
                kernel.signal_snapshot(
                    fingerprint,
                    day,
                    factor_date,
                    pool,
                    raw,
                    processed,
                    targets[day].composite_scores,
                    targets[day].target_holdings,
                    hashes,
                    target_slots=wind["HOLDINGS"] if wind else None,
                    target_weights=(
                        {k: v / (targets[day].nav * runner.init_cash)
                         for k, v in targets[day].target_values.items()}
                        if config.get("compare_target_values") else None
                    ),
                )
            )
        portfolio = result["portfolio"]
        orders = portfolio.orders.records_readable
        for day in trade_dates:
            day_orders = orders[
                pd.to_datetime(orders.Timestamp).dt.strftime("%Y-%m-%d") == day
            ]
            fills = [
                {
                    "stock": str(
                        r.Column[-1] if isinstance(r.Column, tuple) else r.Column
                    ),
                    "side": str(r.Side).lower(),
                    "amount": float(r.Size),
                    "price": float(r.Price),
                    "fees": float(r.Fees),
                }
                for r in day_orders.itertuples()
            ]
            snapshots.extend(
                kernel.snapshot_records(
                    fingerprint,
                    day,
                    "orders",
                    {
                        "fills": sorted(
                            fills, key=lambda x: (x["stock"], x["side"], x["price"])
                        )
                    },
                )
            )
            for stage, values in [
                ("cash", portfolio.cash()),
                ("nav", portfolio.value()),
            ]:
                snapshots.extend(
                    kernel.snapshot_records(
                        fingerprint,
                        day,
                        stage,
                        {"value": float(values.loc[pd.Timestamp(day)])},
                    )
                )
        (output / "local_snapshots.jsonl").write_text("\n".join(snapshots) + "\n")
        return {
            "signal_dates": sorted(raw_records),
            "trade_dates": trade_dates,
            "warmup_dates": warmup_dates,
        }
    finally:
        connection.close()
        data.close()
        if runner is not None:
            runner.close()


def build_cloud_source(config, output, fingerprint):
    """Inline modules and frozen models into a run-scoped backtest source."""
    source = resolve(config["source"]).read_text()
    modules = {"jq_strategy": output / "jq_strategy.py"}
    if config["strategy"] == "wind_micro_w11":
        modules["wm_lr_20260731_1505_deploy"] = resolve(config["models"]["deploy"])
    prefix = "import sys, types, base64, gzip\n"
    for name, path in modules.items():
        encoded = base64.b64encode(gzip.compress(path.read_bytes())).decode()
        prefix += (
            "_m = types.ModuleType(%r)\nexec(compile(gzip.decompress(base64.b64decode(%r)), %r, 'exec'), _m.__dict__)\nsys.modules[%r] = _m\n"
            % (name, encoded, str(path.name), name)
        )
    models = (
        {"lasso_model_bundle_exp_58f_e6.json": config["models"]["lasso"]}
        if config["strategy"] == "lasso"
        else {"wm_lr_20260731_1505_models.json.gz": config["models"]["rank"]}
    )
    encoded_models = {
        name: base64.b64encode(resolve(path).read_bytes()).decode()
        for name, path in models.items()
    }
    encoded_models.update({name: base64.b64encode(resolve(path).read_bytes()).decode()
                           for name, path in config.get("cloud_files", {}).items()})
    suffix = """
_ALIGNMENT_FINGERPRINT = %r
_ALIGNMENT_MODELS = %r
_alignment_read_file = read_file
def read_file(name):
    return base64.b64decode(_ALIGNMENT_MODELS[name]) if name in _ALIGNMENT_MODELS else _alignment_read_file(name)
_alignment_initialize = initialize
_alignment_after_end = globals().get('after_trading_end')
def initialize(context):
    _alignment_initialize(context)
    g.alignment_input_hashes = {}
    g.alignment_emitted = set()
    jq_runtime.set_external_pit_mode('research')
    jq_runtime.set_external_fundamentals_api(
        get_trade_days, query, valuation, get_fundamentals,
        globals().get('get_fundamentals_continuously'))
    original = jq_runtime.compute_factor_batch
    def traced(*args, **kwargs):
        g.alignment_input_hashes = {}
        def capture(name, frame, result):
            g.alignment_input_hashes[name] = jq_runtime.input_digest(frame)
        kwargs['trace'] = capture
        return original(*args, **kwargs)
    jq_runtime.compute_factor_batch = traced

def after_trading_end(context):
    if _alignment_after_end:
        _alignment_after_end(context)
    day = str(context.current_dt.date())
    state = getattr(g, 'alignment_signal', None)
    if state and state['factor_date'] == str(context.previous_date)[:10] and day not in g.alignment_emitted:
        targets = state.get('targets')
        if targets is None:
            raise ValueError('target_snapshot_missing')
        for line in jq_runtime.signal_snapshot(_ALIGNMENT_FINGERPRINT, day, state['factor_date'], state['pool'],
                state['raw'], state['processed'], state['scores'], targets, g.alignment_input_hashes, state.get('target_slots'), state.get('target_weights')):
            log.info('ALIGNMENT_SNAPSHOT|' + line)
        g.alignment_emitted.add(day)
    orders = list(get_orders().values())
    fills = [{'stock': o.security, 'side': 'buy' if o.is_buy else 'sell', 'amount': float(o.filled),
              'price': float(o.price), 'fees': float(o.commission)} for o in orders if o.filled]
    stages = {'orders': {'fills': sorted(fills, key=lambda x: (x['stock'], x['side'], x['price']))},
              'cash': {'value': float(context.portfolio.available_cash)}, 'nav': {'value': float(context.portfolio.total_value)}}
    for stage, values in stages.items():
        for line in jq_runtime.snapshot_records(_ALIGNMENT_FINGERPRINT, day, stage, values):
            log.info('ALIGNMENT_SNAPSHOT|' + line)
""" % (fingerprint, encoded_models)
    result = (
        prefix
        + "\n"
        + source
        + "\njq_runtime = "
        + ("jq" if config["strategy"] == "lasso" else "shared")
        + "\n"
        + suffix
    )
    ast.parse(result, feature_version=(3, 6))
    path = output / "strategy.py"
    path.write_text(result)
    return path


def run_strategy(config_path, output, *, real_cloud=False, resume=False, timeout=1800):
    import time
    from scripts.alignment.alignment_cycle import _atomic_write_json, _run

    output.mkdir(parents=True, exist_ok=True)
    config = load_json(config_path)
    local_config = load_json(resolve(config["local_config"]))
    initial_cash = float(
        local_config.get("top_k_strategy", {}).get("init_cash", 100000)
    )
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("invalid_initial_cash")
    required = ["source", "local_config", "database"]
    paths = (
        [resolve(config[k]) for k in required]
        + [resolve(p) for p in config["factors"].values()]
        + [resolve(p) for p in config["models"].values()]
        + [resolve(p) for p in config.get("cloud_files", {}).values()]
    )
    if config.get("pool_snapshot"):
        paths.append(resolve(config["pool_snapshot"]))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError("alignment_input_missing: " + ",".join(missing))
    if config.get("frequency", "day") not in ("day", "minute"):
        raise ValueError("backtest_frequency_unsupported")
    if config.get("strategy") not in ("lasso", "wind_micro_w11"):
        raise ValueError("strategy_kind_unsupported")
    if not "2022-01-01" <= config["start"] <= config["end"] <= "2023-12-31":
        raise ValueError("alignment_development_window_required")
    sources = [p for p in paths if p != resolve(config["database"])]
    manifest = input_manifest(
        ROOT, resolve(config["source"]), config["strategy"], [config_path] + sources
    )
    stat = resolve(config["database"]).stat()
    manifest["strategy_data_version"] = {
        "path": str(resolve(config["database"])),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    import hashlib

    manifest.pop("fingerprint")
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    manifest["fingerprint"] = fingerprint
    state_path = output / "alignment_cycle_state.json"
    saved = load_json(state_path) if resume and state_path.is_file() else {}
    if saved.get("fingerprint") != fingerprint or saved.get("schema_version") != 2:
        saved = {}
    archive_previous(output, fingerprint)
    _atomic_write_json(output / "alignment_manifest.json", manifest)
    stages = []
    result = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "status": "FAIL",
        "layers": {
            n: {"status": "NOT_VERIFIED"}
            for n in ("inputs", "factor", "signal", "execution")
        },
        "publication_status": "pending_approval",
        "stages": stages,
    }
    reuse = bool(saved)

    def stage(name, action, expected, *, replace_outputs=True):
        nonlocal reuse
        cached = next((s for s in saved.get("stages", []) if s["name"] == name), {})
        if (
            reuse
            and cached.get("ok")
            and {str(p.resolve()) for p in expected}.issubset(
                cached.get("output_hashes", {})
            )
            and intact(cached.get("output_hashes", {}))
        ):
            stages.append(cached)
            return
        reuse = False
        item = {"name": name, "ok": False}
        stages.append(item)
        try:
            _prepare_stage_outputs(expected, replace_outputs)
            action()
            if not all(p.is_file() for p in expected):
                raise ValueError("artifact_missing: " + name)
            retained = expected + (
                list((output / "inputs").rglob("*.parquet"))
                if name == "local_execution"
                else []
            )
            item.update(ok=True, output_hashes=file_hashes(retained))
        except Exception as exc:
            item["reason"] = str(exc)
            raise
        finally:
            _atomic_write_json(
                state_path,
                {"schema_version": 2, "fingerprint": fingerprint, "stages": stages},
            )

    def command(name, argv):
        item = _run(name, argv, output)
        if not item["ok"]:
            raise RuntimeError(item.get("reason") or name + "_execution_failed")

    def validate_fields():
        import duckdb
        from utils.factor_runtime import factor_plan

        mapping = load_json(ROOT / "config/cloud_api_mapping.json")["field_contracts"]
        declarations = literals(
            output / "jq_strategy.py",
            {"FACTOR_REQUIREMENTS", "FACTOR_DEPENDENCIES", "FACTOR_REQUIRED_WINDOWS"},
        )
        checked = {}
        from utils.data_store import connect_input
        with connect_input(
            str(resolve(config["database"])), bindings=local_config.get("data_bindings")
        ) as connection:
            for name in config["factors"]:
                freq, _, fields, _ = factor_plan(
                    name,
                    declarations["FACTOR_REQUIREMENTS"],
                    declarations["FACTOR_DEPENDENCIES"],
                    declarations["FACTOR_REQUIRED_WINDOWS"],
                )
                for field in fields:
                    source = mapping[field]["local"]
                    table = source["table"].replace("{frequency}", freq)
                    if table not in checked:
                        checked[table] = {
                            r[0].lower()
                            for r in connection.execute(
                                'DESCRIBE "' + table.replace('"', '""') + '"'
                            ).fetchall()
                        }
                    if source["column"].lower() not in checked[table]:
                        raise ValueError(
                            "local_field_contract_missing: " + name + ":" + field
                        )
        _atomic_write_json(
            output / "local_field_check.json",
            {"status": "PASS", "tables": {k: sorted(v) for k, v in checked.items()}},
        )

    try:
        bundle = output / "jq_strategy.py"
        stage(
            "build",
            lambda: command(
                "build",
                [
                    sys.executable,
                    str(ROOT / "scripts/build_jq_strategy.py"),
                    "--factors",
                    *config["factors"],
                    "--factor-files",
                    *map(str, map(resolve, config["factors"].values())),
                    "--single-file",
                    "--out",
                    str(bundle),
                ],
            ),
            [bundle],
        )
        stage("local_field_check", validate_fields, [output / "local_field_check.json"])
        stage(
            "build_strategy",
            lambda: build_cloud_source(config, output, fingerprint),
            [output / "strategy.py"],
        )
        if not real_cloud:
            result.update(status="PASS", next_action="real_cloud_alignment")
        else:

            def local():
                metadata = run_local(config, output, fingerprint, bundle)
                _atomic_write_json(output / "local_execution.json", metadata)

            stage(
                "local_execution",
                local,
                [
                    output / "local_snapshots.jsonl",
                    output / "local_execution.json",
                    output / "local_market_snapshot.parquet",
                    output / "local_pool_snapshot.json",
                ],
            )
            metadata = load_json(output / "local_execution.json")
            stage(
                "cloud_start",
                lambda: command(
                    "cloud_start",
                    [
                        sys.executable,
                        str(ROOT / "scripts/wind_micro/start_joinquant_backtest.py"),
                        "--strategy",
                        str(output / "strategy.py"),
                        "--start",
                        config["start"],
                        "--end",
                        config["end"],
                        "--capital",
                        str(initial_cash),
                        "--frequency",
                        config.get("frequency", "day"),
                        "--output",
                        str(output / "cloud_build.json"),
                    ],
                ),
                [output / "cloud_build.json"],
            )
            build = load_json(output / "cloud_build.json")
            identity = build.get("data", {})
            backtest_id = (
                identity.get("backtestId")
                or identity.get("backtest_id")
                or build.get("backtestId")
            )
            if not backtest_id:
                raise ValueError("cloud_backtest_identity_missing")

            def cloud():
                deadline = time.monotonic() + timeout
                while True:
                    command(
                        "cloud_sync",
                        [
                            sys.executable,
                            str(ROOT / "scripts/wind_micro/sync_joinquant_backtest.py"),
                            "--backtest-id",
                            str(backtest_id),
                            "--output-dir",
                            str(output / "cloud"),
                        ],
                    )
                    path = output / "cloud/cloud_dynamic_log.txt"
                    content = path.read_text()
                    producer = load_json(output / "cloud/cloud_dynamic_sync.json")
                    if producer.get("state") == "3" or "Traceback (most recent call last)" in content:
                        detail = next((line for line in reversed(content.splitlines()) if "Error" in line or "factor_execution_failed" in line), "")
                        raise RuntimeError("cloud_execution_failed: " + detail[-2000:])
                    if "ALIGNMENT_SNAPSHOT|" in content:
                        records = read_records(path, fingerprint)
                        if producer.get("state") == "2" and set(metadata["trade_dates"]).issubset(
                            {k[0] for k in records if k[1] == "nav"}
                        ):
                            return
                    if producer.get("state") == "2":
                        raise RuntimeError("cloud_completed_with_missing_snapshots")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("cloud_execution_incomplete")
                    print(
                        json.dumps(
                            {
                                "event": "heartbeat",
                                "stage": "cloud_execution",
                                "backtest_id": backtest_id,
                            }
                        ),
                        flush=True,
                    )
                    time.sleep(20)

            stage(
                "cloud_execution",
                cloud,
                [
                    output / "cloud/cloud_dynamic_log.txt",
                    output / "cloud/cloud_dynamic_sync.json",
                ],
                replace_outputs=False,
            )

            def compare():
                expected = {
                    s: metadata[
                        "trade_dates"
                        if s in ("orders", "cash", "nav")
                        else "signal_dates"
                    ]
                    for s in STAGES
                }
                report = compare_records(
                    read_records(output / "local_snapshots.jsonl", fingerprint),
                    read_records(output / "cloud/cloud_dynamic_log.txt", fingerprint),
                    expected,
                    config.get("tolerances"),
                )
                _atomic_write_json(output / "alignment_comparison.json", report)

            stage("compare", compare, [output / "alignment_comparison.json"])
            result.update(load_json(output / "alignment_comparison.json"))
    except Exception as exc:
        result.update(
            status="FAIL",
            error_code=str(exc),
            failed_stage=next((s["name"] for s in stages if not s["ok"]), "preflight"),
        )
    if not intact_inputs(manifest):
        result.update(
            status="FAIL",
            error_code="input_version_changed",
            failed_stage="input_version",
        )
        result["layers"]["inputs"] = {
            "status": "FAIL",
            "reason": "input_version_changed",
        }
    result["artifact_hashes"] = file_hashes(
        [output / "alignment_manifest.json"]
        + [Path(p) for s in stages for p in s.get("output_hashes", {})]
    )
    _atomic_write_json(output / "alignment_cycle.json", result)
    return result


def run_factor_replay(
    factor_file, factor_class, bundle, cloud_csv, output, config_path, fingerprint
):
    """Replay the declared cloud requests using independently fetched local data.

    Only request dates and pool membership come from the cloud artifact. Cloud
    factor values never enter the local computation. Independent pool agreement
    remains NOT_VERIFIED and belongs to the strategy input layer.
    """
    from backtest_and_analysis.factor_loader import FactorLoader
    from utils.duckdb_adapter import DuckDBAdapter
    from utils.factor_runtime import compute_factor_batch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    requests = pd.read_csv(
        cloud_csv, usecols=["date", "rebalance_date", "pool_date", "code", "factor"]
    )
    if (
        requests.empty
        or set(requests.factor) != {factor_class}
        or requests.duplicated(["rebalance_date", "code"]).any()
    ):
        raise ValueError("factor_request_snapshot_invalid")
    if not requests.date.between("2022-01-01", "2023-12-31").all():
        raise ValueError("alignment_development_window_required")
    history = {
        str(day): sorted(group.code.tolist())
        for day, group in requests.groupby("rebalance_date")
    }
    declarations = literals(
        bundle,
        {"FACTOR_REQUIREMENTS", "FACTOR_DEPENDENCIES", "FACTOR_REQUIRED_WINDOWS"},
    )
    _, module = FactorLoader._load_module_from_path(resolve(factor_file), serial=True)
    cls = getattr(module, factor_class)
    config = load_json(config_path)
    data = DuckDBAdapter(duckdb_path=config["database"]["duckdb_path"], use_cache=False)
    adapter = local_adapter_factory(
        data, load_json(ROOT / "config/cloud_api_mapping.json")["field_contracts"]
    )
    records = []
    snapshots = []
    try:
        for day, request in requests.groupby("rebalance_date", sort=True):
            if request.date.nunique() != 1 or request.pool_date.nunique() != 1:
                raise ValueError("factor_request_date_ambiguous")
            asof = str(request.date.iloc[0])

            def trace(name, frame, result):
                path = output / "inputs" / str(day) / (name + ".parquet")
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                snapshots.append(
                    {
                        "date": str(day),
                        "asof": asof,
                        "factor": name,
                        "input": str(path.resolve()),
                        "rows": len(frame),
                        "stocks": sorted(frame.code.unique().tolist()),
                    }
                )

            values = compute_factor_batch(
                {factor_class: cls},
                declarations["FACTOR_REQUIREMENTS"],
                declarations["FACTOR_DEPENDENCIES"],
                history[str(day)],
                asof,
                adapter,
                windows=declarations["FACTOR_REQUIRED_WINDOWS"],
                pool_history=history,
                cutoff=str(day),
                trace=trace,
            )[factor_class]
            records.extend(
                {"date": asof, "code": code, "value": value}
                for code, value in values.items()
            )
            print(
                json.dumps(
                    {
                        "event": "local_factor_replay",
                        "date": asof,
                        "stocks": len(values),
                    }
                ),
                flush=True,
            )
        frame = pd.DataFrame(records)
        if frame.empty or not frame.value.notna().any():
            raise ValueError("local_factor_output_empty")
        frame.to_csv(output / "local_factor_snapshot.csv", index=False)
        payload = {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "ok": True,
            "request_source": "cloud_request_schedule",
            "independent_pool_validation": "NOT_VERIFIED",
            "requests": snapshots,
            "input_hashes": file_hashes([Path(s["input"]) for s in snapshots]),
        }
        (output / "local_analysis.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    finally:
        data.close()
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Internal local replay backend for alignment-run"
    )
    for argument in (
        "factor",
        "factor-class",
        "bundle",
        "cloud-csv",
        "output",
        "config",
        "fingerprint",
    ):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    run_factor_replay(
        args.factor,
        args.factor_class,
        args.bundle,
        args.cloud_csv,
        args.output,
        args.config,
        args.fingerprint,
    )
