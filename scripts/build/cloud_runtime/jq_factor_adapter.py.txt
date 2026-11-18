"""JoinQuant data adapter, also inlined verbatim into cloud bundles."""

from utils.pit import apply_pit_adjustment
import pandas as pd


def rolling_money_flow_sum(values, window, min_periods):
    """Sum each finite trailing window independently of pandas' rolling state."""
    import math
    import numpy as np
    import pandas as pd
    if (not isinstance(window, int) or isinstance(window, bool) or window < 1
            or not isinstance(min_periods, int) or isinstance(min_periods, bool)
            or not 0 <= min_periods <= window):
        raise ValueError('money_flow_rolling_contract_invalid')
    array = np.asarray(values, dtype=np.float64)
    output = np.full(len(array), np.nan, dtype=np.float64)
    for end in range(len(array)):
        trailing = array[max(0, end - window + 1):end + 1]
        finite = trailing[np.isfinite(trailing)]
        if len(finite) >= min_periods:
            output[end] = math.fsum(float(value) for value in finite)
    return pd.Series(output, index=values.index, name=values.name)


def bounded_price_fetch(adapter, securities, end_date, fields, fetch_one, max_rows=200000):
    """Bound normalization/PIT temporaries without changing any stock's window."""
    frequency = adapter._normalize_freq(adapter.freq)
    bars = adapter._bars_per_day(frequency)
    if (not isinstance(securities, (list, tuple)) or len(securities) <= 1 or bars <= 1
            or adapter.use_expanding_window
            or set(fields or []) - adapter._WHITELIST.get(frequency, set())):
        return fetch_one(securities, end_date, fields)
    desired = ((int(adapter.window) + bars - 1) // bars) * bars
    if desired <= 0 or desired * len(securities) <= max_rows:
        return fetch_one(securities, end_date, fields)
    # Preserve the existing API request boundaries, including their returned
    # auxiliary index column, while applying normalization/PIT to small groups.
    api_size = len(adapter._security_batches_for_price(securities, desired)[0])
    batch_size = api_size * max(1, int(max_rows // (desired * api_size)))
    if batch_size >= len(securities):
        return fetch_one(securities, end_date, fields)
    parts = []
    for offset in range(0, len(securities), batch_size):
        part = fetch_one(list(securities[offset:offset + batch_size]), end_date, fields)
        if part is not None:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, axis=0, ignore_index=True)
    del parts
    if result.empty:
        return result
    return result.sort_values(['code', 'date']).reset_index(drop=True)


CLOUD_API_MAPPING = {}

# Optional global hook to register an external get_price implementation
EXTERNAL_GET_PRICE = None


def set_external_get_price(fn):
    # Register a process-wide get_price fallback function
    global EXTERNAL_GET_PRICE
    EXTERNAL_GET_PRICE = fn


# Optional global hook to register an external jqfactor.get_factor_values implementation
EXTERNAL_GET_FACTOR_VALUES = None


def set_external_get_factor_values(fn):
    # Register a process-wide get_factor_values fallback function
    global EXTERNAL_GET_FACTOR_VALUES
    EXTERNAL_GET_FACTOR_VALUES = fn


# Optional global hook to register an external get_valuation implementation
EXTERNAL_GET_VALUATION = None
EXTERNAL_FUNDAMENTALS_API = None


def set_external_get_valuation(fn):
    # Register a process-wide get_valuation fallback function
    global EXTERNAL_GET_VALUATION
    EXTERNAL_GET_VALUATION = fn


def set_external_fundamentals_api(
    get_trade_days_fn,
    query_fn,
    valuation_table,
    get_fundamentals_fn,
    get_fundamentals_continuously_fn=None,
):
    global EXTERNAL_FUNDAMENTALS_API
    EXTERNAL_FUNDAMENTALS_API = (
        get_trade_days_fn,
        query_fn,
        valuation_table,
        get_fundamentals_fn,
        get_fundamentals_continuously_fn,
    )


# Optional global hook to register external PIT mode ('backtest' or 'research')
EXTERNAL_PIT_MODE = None


def set_external_pit_mode(mode):
    # Register a process-wide PIT mode setting
    # 'backtest': use JQ dynamic PIT (fq='pre'), default for backtesting
    # 'research': use Dynamic PIT (fq=None + factor/factor_end), matches JQ fq='pre'
    global EXTERNAL_PIT_MODE
    if mode in ("backtest", "research"):
        EXTERNAL_PIT_MODE = mode


class JQFactorAdapter:
    # Stable contract with explicit policy controls
    _JQ_GET_PRICE_FIELDS = {
        "open",
        "close",
        "high",
        "low",
        "volume",
        "money",
        "open_interest",
        "high_limit",
        "low_limit",
        "paused",
        "pre_close",
        "avg",
        "factor",
    }
    _WHITELIST = {
        "daily": {
            "open",
            "close",
            "high",
            "low",
            "volume",
            "money",
            "factor",
            "high_limit",
            "low_limit",
            "avg",
            "pre_close",
            "paused",
        },
        "1m": {
            "open",
            "close",
            "high",
            "low",
            "volume",
            "money",
            "factor",
            "high_limit",
            "low_limit",
            "avg",
            "pre_close",
            "paused",
        },
        "5m": {
            "open",
            "close",
            "high",
            "low",
            "volume",
            "money",
            "factor",
            "high_limit",
            "low_limit",
        },  # [ALIGNMENT-FIX] preserve minute-bar PIT and limit-price fields used by 5m factors
        "15m": {
            "open",
            "close",
            "high",
            "low",
            "volume",
            "money",
            "factor",
        },  # [ALIGNMENT-FIX] keep PIT-capable field contract consistent across minute frequencies
        "30m": {
            "open",
            "close",
            "high",
            "low",
            "volume",
            "money",
            "factor",
        },  # [ALIGNMENT-FIX] keep PIT-capable field contract consistent across minute frequencies
        "60m": {
            "open",
            "close",
            "high",
            "low",
            "volume",
            "money",
            "factor",
        },  # [ALIGNMENT-FIX] keep PIT-capable field contract consistent across minute frequencies
    }

    def __init__(
        self,
        dependencies,
        freq="daily",
        window=120,
        strict=True,
        allow_field_fallback=False,
        end_of_day_on_minute=True,
        min_full_day_tolerance=0,
        use_expanding_window=False,
        expanding_start_date=None,
        get_price_func=None,
        pit_mode="backtest",
    ):
        self.dependencies = dependencies
        self.freq = freq
        self.window = window
        self.strict = bool(strict)
        self.allow_field_fallback = bool(allow_field_fallback)
        self.end_of_day_on_minute = bool(end_of_day_on_minute)
        # Allow incomplete days up to tolerance bars (e.g., 2 => keep days with >= 46 bars for 5m)
        try:
            self.min_full_day_tolerance = int(min_full_day_tolerance)
        except Exception:
            self.min_full_day_tolerance = 0
        # Expanding window mode: use start_date..end_date instead of count-based window
        self.use_expanding_window = bool(use_expanding_window)
        self.expanding_start_date = expanding_start_date  # str or datetime/date
        self._probed = False
        self._returns_panel = False
        self._supported_fields = None
        # Optional external get_price function
        self._get_price_func = get_price_func
        # Optional external get_factor_values function
        self._get_factor_values_func = None
        # PIT mode for cloud environment alignment:
        # 'backtest': use JQ dynamic PIT (fq='pre'), default for backtesting
        # 'research': use Dynamic PIT (fq=None + factor/factor_end), matches JQ fq='pre'
        self.pit_mode = pit_mode if pit_mode in ("backtest", "research") else "backtest"
        self.pit_enabled = True

    def set_get_price(self, fn):
        self._get_price_func = fn

    def _resolve_pit_mode(self):
        # Resolve PIT mode: external hook takes precedence over instance setting
        try:
            if "EXTERNAL_PIT_MODE" in globals() and EXTERNAL_PIT_MODE is not None:
                return EXTERNAL_PIT_MODE
        except Exception:
            pass
        return self.pit_mode

    def _resolve_get_price(self):
        fn = self._get_price_func
        if callable(fn):
            return fn
        try:
            if "EXTERNAL_GET_PRICE" in globals() and callable(EXTERNAL_GET_PRICE):
                return EXTERNAL_GET_PRICE
        except Exception:
            pass
        try:
            return get_price
        except NameError:
            return None

    def _resolve_get_factor_values(self):
        fn = self._get_factor_values_func
        if callable(fn):
            return fn
        try:
            if "EXTERNAL_GET_FACTOR_VALUES" in globals() and callable(
                EXTERNAL_GET_FACTOR_VALUES
            ):
                return EXTERNAL_GET_FACTOR_VALUES
        except Exception:
            pass
        try:
            from jqfactor import get_factor_values as _gf  # type: ignore

            return _gf
        except Exception:
            return None

    def _resolve_get_valuation(self):
        try:
            if "EXTERNAL_GET_VALUATION" in globals() and callable(
                EXTERNAL_GET_VALUATION
            ):
                return EXTERNAL_GET_VALUATION
        except Exception:
            pass
        try:
            from jqdata import get_valuation as _gv

            return _gv
        except Exception:
            return None

    def _call_get_price(self, **kwargs):
        fn = self._resolve_get_price()
        if fn is None:
            raise RuntimeError(
                "No get_price function available. Set via set_external_get_price(fn) or adapter.set_get_price(fn), or run in JoinQuant."
            )
        return fn(**kwargs)

    def _security_batches_for_price(self, securities, desired_rows_per_security):
        if not isinstance(securities, (list, tuple)) or len(securities) <= 1:
            return [securities]
        try:
            per_security = max(1, int(desired_rows_per_security or 1))
        except Exception:
            per_security = 1
        # Keep multi-security responses bounded while avoiding one API call per
        # stock for ordinary intraday windows. The current research API returns
        # complete 33k-row panel=False responses; 30k leaves a safety margin.
        max_rows = 30000
        batch_size = max(1, int(max_rows // per_security))
        if len(securities) <= batch_size:
            return [securities]
        return [
            list(securities[i : i + batch_size])
            for i in range(0, len(securities), batch_size)
        ]

    def _concat_get_price_batches(self, batches, **kwargs):
        frames = []
        for batch in batches:
            call_kwargs = dict(kwargs)
            call_kwargs["security"] = batch
            part = self._call_get_price(**call_kwargs)
            if part is None:
                continue
            try:
                if hasattr(part, "empty") and part.empty:
                    continue
            except Exception:
                if self.strict:
                    raise
                pass
            frames.append(part)
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        try:
            return pd.concat(frames, axis=0)
        except Exception:
            if self.strict:
                raise
            return frames[0]

    def jq_fields_for(self, fields):
        mapping = {"amount": "money", "turnover": "volume"}
        out = []
        for f in fields:
            if f in ("date", "code"):
                continue
            out.append(mapping.get(f, f))
        return out

    def _normalize_freq(self, freq: str) -> str:
        if freq in ("1m", "5m", "15m", "30m", "60m", "daily"):
            return freq
        return "daily"

    def _resolve_fields(self, freq: str, requested):
        req = set(requested or [])
        wl = self._WHITELIST.get(freq, self._WHITELIST["daily"])
        usable = sorted(list(req & wl)) if req else sorted(list(wl))
        missing = sorted(list(req - wl))
        return usable, missing

    def _probe_capabilities(self, securities, end_dt, freq_arg):
        if self._probed:
            return
        try:
            sec = (
                securities[0]
                if isinstance(securities, (list, tuple)) and securities
                else securities
            )
            # skip_paused=False 用于保持与本地一致，防止因为零成交量分钟被过滤导致整天被丢弃
            res = self._call_get_price(
                security=sec,
                end_date=end_dt,
                count=1,
                frequency=freq_arg,
                skip_paused=False,
                fill_paused=False,
                panel=False,
            )
            self._returns_panel = (
                hasattr(res, "items")
                and hasattr(res, "get")
                and callable(getattr(res, "get", None))
            )
            if self._returns_panel:
                try:
                    self._supported_fields = (
                        set(list(res.items)) if hasattr(res, "items") else set()
                    )
                except Exception:
                    self._supported_fields = None
            else:
                self._supported_fields = (
                    set(res.columns) if hasattr(res, "columns") else None
                )
        except Exception:
            self._returns_panel = False
            self._supported_fields = None
        finally:
            self._probed = True

    def _bars_per_day(self, freq: str) -> int:
        if freq == "1m":
            return 240
        if freq == "5m":
            return 48
        if freq == "15m":
            return 16
        if freq == "30m":
            return 8
        if freq == "60m":
            return 4
        return 1

    def _fetch_valuation_data(self, securities, start_date, end_date, target_fields):
        if not target_fields:
            return pd.DataFrame()
        gv = self._resolve_get_valuation()
        if not callable(gv):
            return self._fetch_valuation_continuously(
                securities, start_date, end_date, target_fields
            )

        # Resolve mapping
        val_map = {}
        if "CLOUD_API_MAPPING" in globals() and isinstance(CLOUD_API_MAPPING, dict):
            val_map = CLOUD_API_MAPPING.get("get_valuation", {})

        jq_fields = []
        aliases = []
        for f in target_fields:
            s = str(f)
            if s in val_map:
                mapped = val_map[s]
                jq_fields.append(mapped)
                aliases.append((s, mapped))

        if not jq_fields:
            return pd.DataFrame()

        # 动态分批获取逻辑：get_valuation 单次最多返回 10000 条数据
        # 估算数据量：stocks * days，若超过 10000 则分批
        MAX_ROWS = 10000
        n_stocks = len(securities) if isinstance(securities, (list, tuple)) else 1

        try:
            s_dt = pd.to_datetime(start_date)
            e_dt = pd.to_datetime(end_date)
            n_days = max(1, (e_dt - s_dt).days + 1)
        except Exception:
            if self.strict:
                raise
            n_days = 1

        estimated_rows = n_stocks * n_days

        try:
            if estimated_rows <= MAX_ROWS:
                # 单次获取
                res = gv(
                    securities,
                    start_date=start_date,
                    end_date=end_date,
                    fields=jq_fields,
                )
            else:
                # 分批获取：按日期区间分批
                days_per_batch = max(1, MAX_ROWS // n_stocks)
                all_results = []
                current_start = s_dt

                while current_start <= e_dt:
                    batch_end = min(
                        current_start + pd.Timedelta(days=days_per_batch - 1), e_dt
                    )
                    batch_res = gv(
                        securities,
                        start_date=current_start.strftime("%Y-%m-%d"),
                        end_date=batch_end.strftime("%Y-%m-%d"),
                        fields=jq_fields,
                    )
                    if batch_res is not None and not batch_res.empty:
                        all_results.append(batch_res)
                    current_start = batch_end + pd.Timedelta(days=1)

                res = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

            if res is None or res.empty:
                return self._fetch_valuation_continuously(
                    securities, start_date, end_date, target_fields
                )
            if "day" in res.columns:
                res = res.rename(columns={"day": "date"})
            if "date" in res.columns:
                res["date"] = pd.to_datetime(res["date"])
            for field, raw in aliases:
                if raw in res.columns:
                    res[field] = res[raw]
            if any(field not in res.columns for field in target_fields):
                return self._fetch_valuation_continuously(
                    securities, start_date, end_date, target_fields
                )
            keep = ["date", "code"] + [f for f in target_fields if f in res.columns]
            return res[keep]
        except Exception:
            if self.strict:
                raise
            return pd.DataFrame()

    def _fetch_valuation_continuously(
        self, securities, start_date, end_date, target_fields
    ):
        """Use the strategy-safe historical valuation API when get_valuation is empty."""
        api = globals().get("EXTERNAL_FUNDAMENTALS_API")
        if api is not None:
            (
                get_trade_days,
                query,
                valuation,
                get_fundamentals,
                get_fundamentals_continuously,
            ) = api
        else:
            from jqdata import get_trade_days, query, valuation  # type: ignore

            try:
                from jqdata import get_fundamentals_continuously  # type: ignore
            except ImportError:
                get_fundamentals_continuously = None
                from jqdata import get_fundamentals  # type: ignore

        mapping = CLOUD_API_MAPPING.get("get_valuation", {})
        raw_fields = sorted({mapping[field] for field in target_fields})
        days = list(get_trade_days(start_date=start_date, end_date=end_date))
        stocks = (
            list(securities) if isinstance(securities, (list, tuple)) else [securities]
        )
        if not days or not stocks:
            return pd.DataFrame()
        columns = [valuation.code] + [getattr(valuation, field) for field in raw_fields]
        frames = []
        if get_fundamentals_continuously is None:
            request = query(*columns).filter(valuation.code.in_(stocks))
            for day in days:
                frame = get_fundamentals(request, date=pd.Timestamp(day).date())
                if frame is not None and not frame.empty:
                    frame = frame.copy()
                    frame["date"] = pd.Timestamp(day)
                    frames.append(frame)
        else:
            batch_size = max(1, 10000 // len(days))
            for offset in range(0, len(stocks), batch_size):
                batch = stocks[offset : offset + batch_size]
                request = query(*columns).filter(valuation.code.in_(batch))
                frame = get_fundamentals_continuously(
                    request, end_date=days[-1], count=len(days), panel=False
                )
                if frame is not None and not frame.empty:
                    frames.append(
                        frame.reset_index() if "day" not in frame.columns else frame
                    )
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        if "day" in result.columns:
            result = result.rename(columns={"day": "date"})
        if "date" not in result.columns:
            raise ValueError("continuous_valuation_date_missing")
        result["date"] = pd.to_datetime(result["date"])
        result = result[
            result["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ]
        for field, raw in mapping.items():
            if field in target_fields and raw in result.columns:
                result[field] = result[raw]
        keep = ["date", "code"] + [field for field in target_fields if field in result]
        return result[keep].drop_duplicates(["date", "code"], keep="last")

    def _fetch_money_flow(self, securities, start_date, end_date, fields):
        from jqdata import get_money_flow

        mapping = CLOUD_API_MAPPING.get("get_money_flow", {})
        raw = sorted({mapping[f]["field"] for f in fields})
        frame = get_money_flow(
            securities,
            start_date=start_date,
            end_date=end_date,
            fields=["date", "sec_code"] + raw,
        )
        if frame is None or frame.empty:
            raise ValueError("money_flow_input_empty")
        rolling_fields = [f for f in fields if mapping[f].get("rolling_days")]
        if rolling_fields:
            history_count = max(mapping[f]["rolling_days"] for f in rolling_fields) - 1
            if history_count > 0:
                history = get_money_flow(
                    securities,
                    end_date=(pd.Timestamp(start_date).normalize() - pd.Timedelta(days=1)).date(),
                    count=history_count,
                    fields=["date", "sec_code"] + sorted({mapping[f]["field"] for f in rolling_fields}),
                )
                if history is not None and not history.empty:
                    history = history[pd.to_datetime(history["date"]) < pd.Timestamp(start_date)]
                    frame = pd.concat([history, frame])
        frame = frame.rename(columns={"sec_code": "code"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["code", "date"], kind="mergesort")
        for name in fields:
            spec = mapping[name]
            if spec.get("rolling_days"):
                frame[name] = frame.groupby("code")[spec["field"]].transform(
                    lambda values: rolling_money_flow_sum(
                        values, spec["rolling_days"], spec["min_periods"]
                    )
                )
            else:
                frame[name] = frame[spec["field"]]
        if rolling_fields:
            selected = [securities] if isinstance(securities, str) else list(securities)
            frame = frame[
                frame["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
                & frame["code"].isin(selected)
            ]
        return frame[["date", "code"] + fields]

    def _fetch_extra_factors(self, securities, start_date, end_date, target_fields):
        target_fields = [f for f in (target_fields or []) if f]
        if not target_fields:
            return pd.DataFrame()

        # 1. Split fields into valuation vs factor based on CLOUD_API_MAPPING
        val_map = {}
        factor_fields_set = set()
        factor_sources = {}
        if "CLOUD_API_MAPPING" in globals() and isinstance(CLOUD_API_MAPPING, dict):
            val_map = CLOUD_API_MAPPING.get("get_valuation", {})
            factor_fields_set = set(CLOUD_API_MAPPING.get("get_factor_values", []))
            factor_sources = {
                name: spec["cloud"].get("field", name)
                for name, spec in CLOUD_API_MAPPING.get("field_contracts", {}).items()
                if spec.get("cloud", {}).get("api") == "get_factor_values"
            }
            factor_fields_set.update(factor_sources)

        val_fields = [f for f in target_fields if f in val_map]
        # Only include fields that are explicitly in get_factor_values list
        factor_fields = [f for f in target_fields if f in factor_fields_set]
        industry_fields = [
            f for f in target_fields if f in ("sw_l1_code", "sw_l1_name")
        ]

        flow_fields = [
            f for f in target_fields if f in CLOUD_API_MAPPING.get("get_money_flow", {})
        ]
        out = (
            self._fetch_money_flow(securities, start_date, end_date, flow_fields)
            if flow_fields
            else None
        )

        # 2. Fetch Valuation
        if val_fields:
            val_df = self._fetch_valuation_data(
                securities, start_date, end_date, val_fields
            )
            if not val_df.empty:
                out = (
                    val_df
                    if out is None
                    else out.merge(val_df, on=["date", "code"], how="outer")
                )

        # 3. Fetch Factors
        if factor_fields:
            source_fields = list(dict.fromkeys(factor_sources.get(name, name) for name in factor_fields))
            gf = self._resolve_get_factor_values()
            if callable(gf):
                # JQ rejects oversized stock x date requests; stock batches keep
                # long-history fundamental pulls below the API response limit.
                security_list = (
                    list(securities)
                    if isinstance(securities, (list, tuple))
                    else [securities]
                )
                batch_results = []
                for offset in range(0, len(security_list), 25):
                    try:
                        batch_results.append(
                            gf(
                                securities=security_list[offset : offset + 25],
                                factors=source_fields,
                                start_date=start_date,
                                end_date=end_date,
                            )
                        )
                    except Exception:
                        if self.strict:
                            raise
                        continue
                res_direct = batch_results[0] if len(batch_results) == 1 else None
                if len(batch_results) > 1 and all(
                    isinstance(item, dict) for item in batch_results
                ):
                    res_direct = {
                        factor_name: pd.concat(
                            [
                                item[factor_name]
                                for item in batch_results
                                if item.get(factor_name) is not None
                            ],
                            axis=1,
                        )
                        for factor_name in source_fields
                        if any(
                            item.get(factor_name) is not None for item in batch_results
                        )
                    }

                # Merge logic for factors
                temp_factor = None
                if isinstance(res_direct, dict):
                    for factor_name in factor_fields:
                        try:
                            df = res_direct.get(factor_sources.get(factor_name, factor_name))
                            if df is None:
                                continue
                            longf = df.stack(dropna=False).reset_index()
                            cols = list(longf.columns)
                            if len(cols) >= 3:
                                longf = longf.rename(
                                    columns={
                                        cols[0]: "date",
                                        cols[1]: "code",
                                        cols[2]: factor_name,
                                    }
                                )
                            longf["date"] = pd.to_datetime(longf["date"])
                            if temp_factor is None:
                                temp_factor = longf[
                                    ["date", "code", factor_name]
                                ].copy()
                            else:
                                temp_factor = pd.merge(
                                    temp_factor,
                                    longf[["date", "code", factor_name]],
                                    on=["date", "code"],
                                    how="outer",
                                )
                        except Exception:
                            if self.strict:
                                raise
                            continue
                elif isinstance(res_direct, pd.DataFrame):
                    try:
                        longf = res_direct.stack(dropna=False).reset_index()
                        cols = list(longf.columns)
                        if len(cols) >= 3:
                            longf = longf.rename(
                                columns={cols[0]: "date", cols[1]: "code"}
                            )
                            longf["date"] = pd.to_datetime(longf["date"])
                            temp_factor = longf
                    except Exception:
                        if self.strict:
                            raise
                        pass

                if temp_factor is not None and not temp_factor.empty:
                    if out is None:
                        out = temp_factor
                    else:
                        out = pd.merge(
                            out, temp_factor, on=["date", "code"], how="outer"
                        )

        if industry_fields:
            try:
                from jqdata import get_industry, get_trade_days

                industry_rows = []
                for day in get_trade_days(start_date=start_date, end_date=end_date):
                    snapshot = get_industry(securities, date=day) or {}
                    for code, detail in snapshot.items():
                        level = (detail or {}).get("sw_l1") or {}
                        row = {"date": pd.to_datetime(day), "code": code}
                        if "sw_l1_code" in industry_fields:
                            row["sw_l1_code"] = level.get("industry_code")
                        if "sw_l1_name" in industry_fields:
                            row["sw_l1_name"] = level.get("industry_name")
                        industry_rows.append(row)
                industry_df = pd.DataFrame(industry_rows)
                if not industry_df.empty:
                    out = (
                        industry_df
                        if out is None
                        else pd.merge(
                            out, industry_df, on=["date", "code"], how="outer"
                        )
                    )
            except Exception:
                if self.strict:
                    raise
                pass

        if out is None:
            return pd.DataFrame()
        keep_cols = ["date", "code"] + [f for f in target_fields if f in out.columns]
        return out[keep_cols]

    def _fetch_historical_daily_price_fields(self, securities, dates, fields):
        frames = []
        adapter = JQFactorAdapter(
            self.dependencies,
            freq="daily",
            window=1,
            strict=self.strict,
            allow_field_fallback=self.allow_field_fallback,
            get_price_func=self._get_price_func,
            pit_mode=self.pit_mode,
        )
        for date in sorted(set(pd.DatetimeIndex(pd.to_datetime(dates)).floor("D"))):
            frame = adapter.fetch_price(securities, date, fields)
            if frame is not None and not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _trim_to_full_days(
        self, df: "pd.DataFrame", bars_per_day: int
    ) -> "pd.DataFrame":
        if df is None or df.empty or bars_per_day <= 1:
            return df
        tmp = df.copy()
        # Ensure datetime type
        try:
            tmp["date"] = pd.to_datetime(tmp["date"])
        except Exception:
            return df
        tmp["__day__"] = tmp["date"].dt.floor("D")
        cnt = tmp.groupby(["code", "__day__"])["date"].count()
        # Apply tolerance: allow keeping days with counts >= bars_per_day - tolerance
        threshold = max(1, bars_per_day - (self.min_full_day_tolerance or 0))
        keep_idx = cnt[cnt >= threshold].index
        if len(keep_idx) == 0:
            return pd.DataFrame(columns=df.columns)
        out = (
            tmp.set_index(["code", "__day__"])
            .loc[keep_idx]
            .reset_index()
            .drop(columns=["__day__"])
        )
        return out.sort_values(["code", "date"]).reset_index(drop=True)

    def _apply_dynamic_pit(self, df):
        return apply_pit_adjustment(df)

    def _fetch_factor_separately(self, securities, start_date, end_date):
        """Fetch real factor values using fq='pre' (required for research mode)

        When fq=None, factor is always 1.0. We need a separate call with fq='pre'
        to get the actual cumulative adjustment factor.
        """
        try:
            try:
                n_calendar_days = max(
                    1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
                )
            except Exception:
                if self.strict:
                    raise
                n_calendar_days = 1
            batches = self._security_batches_for_price(securities, n_calendar_days)
            df = self._concat_get_price_batches(
                batches,
                start_date=start_date,
                end_date=end_date,
                frequency="daily",
                fields=["factor"],
                skip_paused=False,
                fill_paused=False,
                panel=False,
                fq="pre",  # Required to get real factor
            )
            if df is None or df.empty:
                return None
            df = df.reset_index()
            if "time" in df.columns:
                df = df.rename(columns={"time": "date"})
            if "index" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"index": "date"})
            if "security" in df.columns:
                df = df.rename(columns={"security": "code"})
            # Inject code if single security
            if "code" not in df.columns:
                if isinstance(securities, (list, tuple)) and len(securities) == 1:
                    df["code"] = securities[0]
                elif isinstance(securities, str):
                    df["code"] = securities
            df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
            return df[["date", "code", "factor"]].drop_duplicates()
        except Exception:
            if self.strict:
                raise
            return None

    def fetch_price(self, securities, end_date, fields):
        requested = sorted(set(fields or []))
        result = self._fetch_price(securities, end_date, requested)
        if result is not None and not result.empty:
            custom = set(requested) - self._WHITELIST.get(
                self._normalize_freq(self.freq), self._WHITELIST["daily"]
            )
            missing_custom = sorted(custom - set(result.columns))
            if missing_custom:
                extra = self._fetch_extra_factors(
                    securities,
                    result["date"].min().date(),
                    result["date"].max().date(),
                    missing_custom,
                )
                if extra is not None and not extra.empty:
                    result = result.assign(
                        _day=pd.to_datetime(result["date"]).dt.floor("D")
                    )
                    extra = extra.rename(columns={"date": "_day"})
                    extra["_day"] = pd.to_datetime(extra["_day"]).dt.floor("D")
                    result = result.merge(
                        extra, on=["_day", "code"], how="left", validate="many_to_one"
                    ).drop(columns="_day")
        if self.strict:
            if result is None or result.empty:
                raise ValueError("factor_input_empty")
            missing = set(requested) - set(result.columns)
            if missing:
                raise ValueError(
                    "factor_input_fields_missing: " + ",".join(sorted(missing))
                )
            if result.duplicated(["date", "code"]).any():
                raise ValueError("factor_input_duplicate_key")
        return result

    def _fetch_price(self, securities, end_date, fields):
        return bounded_price_fetch(self, securities, end_date, fields, self._fetch_price_unbatched)

    def _fetch_price_unbatched(self, securities, end_date, fields):
        import datetime as _dt

        # Resolve frequency and field whitelist
        use_freq = self._normalize_freq(self.freq)
        requested = set(fields or [])
        needs_pit = self.pit_enabled and bool(
            requested
            & {
                "open",
                "close",
                "high",
                "low",
                "volume",
                "factor",
                "pre_close",
                "avg",
                "high_limit",
                "low_limit",
            }
        )
        wl = self._WHITELIST.get(use_freq, self._WHITELIST["daily"])
        # Partition into price fields and custom factor fields
        custom_fields = sorted(list(requested - wl))
        price_requested = sorted(list(requested & wl))
        usable, missing = (
            self._resolve_fields(use_freq, price_requested)
            if price_requested
            else ([], [])
        )
        if self.strict and missing:
            raise ValueError(
                "Unsupported fields for frequency {}: {}".format(use_freq, missing)
            )
        req_fields = set(self.jq_fields_for(set(usable)))
        synthetic_price_fields = sorted(list(req_fields - self._JQ_GET_PRICE_FIELDS))
        req_fields = req_fields & self._JQ_GET_PRICE_FIELDS
        if "pre_close" in synthetic_price_fields:
            req_fields.add("close")
        if "avg" in synthetic_price_fields:
            req_fields.update({"money", "volume"})
        if "paused" in synthetic_price_fields:
            req_fields.add("volume")
        if not req_fields and synthetic_price_fields:
            req_fields.add("close")
        req_fields = list(req_fields)

        # Make sure minute end_date includes the day; if only date, extend to end-of-day
        # Parse end_date robustly (allow str/date/datetime)
        end_dt = end_date
        try:
            if isinstance(end_date, str):
                end_dt = pd.to_datetime(end_date)
        except Exception:
            if self.strict:
                raise
            end_dt = end_date
        try:
            # Check if we need to extend end_date to end-of-day for intraday frequencies
            # This applies if original input is a date object OR a string that parsed to midnight
            should_extend = False
            if isinstance(end_date, _dt.date) and not isinstance(
                end_date, _dt.datetime
            ):
                should_extend = True
            elif isinstance(end_date, str):
                # If parsed end_dt is exactly midnight, assume it was a date string
                if (
                    isinstance(end_dt, _dt.datetime)
                    and end_dt.hour == 0
                    and end_dt.minute == 0
                    and end_dt.second == 0
                ):
                    should_extend = True

            if (
                should_extend
                and use_freq in ("1m", "5m", "15m", "30m", "60m")
                and self.end_of_day_on_minute
            ):
                if isinstance(end_dt, _dt.datetime):
                    end_dt = end_dt.replace(hour=23, minute=59, second=59)
                else:
                    end_dt = _dt.datetime.combine(end_dt, _dt.time(23, 59, 59))
        except Exception:
            if self.strict:
                raise
            pass

        # Probe once to decide path and available fields
        self._probe_capabilities(securities, end_dt, use_freq)

        # Call get_price; if fields rejected and allow_field_fallback then retry without fields
        res = None
        # Align count to whole-day multiples for intraday
        # Align count to whole-day multiples for intraday
        bars_pd = self._bars_per_day(use_freq)
        desired_count = self.window
        if bars_pd > 1:
            q, r = divmod(desired_count, bars_pd)
            if r != 0:
                desired_count = (q + 1) * bars_pd
        historical_daily_fields = []
        if bars_pd > 1 and desired_count > 1:
            historical_daily_fields = sorted(
                set(req_fields) & {"high_limit", "low_limit"}
            )
            req_fields = [
                field for field in req_fields if field not in historical_daily_fields
            ]

        # Always use skip_paused=False to ensure consistency with local data and avoid ragged days
        # skip_paused_flag = not is_multi  <-- logic removed
        skip_paused_flag = False

        # PIT mode: select fq parameter
        # - backtest: use JQ dynamic PIT (fq='pre')
        # - research: use raw prices (fq=None) + separate factor fetch, then apply Dynamic PIT
        fq_param = "pre"  # default for backtest mode (JQ dynamic PIT)
        research_mode = self._resolve_pit_mode() == "research" or not self.pit_enabled
        needs_factor = needs_pit or "factor" in requested
        if research_mode:
            fq_param = None  # raw prices, optionally adjusted below
        if research_mode and needs_factor:
            # Remove 'factor' from req_fields - it will be fetched separately
            # because fq=None always returns factor=1.0
            req_fields = [f for f in req_fields if f != "factor"]

        if not req_fields:
            use_full_interval = False
            try:
                end_for_extra = pd.to_datetime(end_dt).date()
            except Exception:
                if self.strict:
                    raise
                end_for_extra = end_dt
            try:
                start_for_extra = (
                    pd.Timestamp(self.expanding_start_date).date()
                    if self.use_expanding_window and self.expanding_start_date
                    else (
                        pd.to_datetime(end_for_extra)
                        - pd.Timedelta(days=int(desired_count * 2 + 30))
                    ).date()
                )
                use_full_interval = bool(
                    self.use_expanding_window and self.expanding_start_date
                    and pd.notna(start_for_extra)
                )
            except Exception:
                if self.strict:
                    raise
                start_for_extra = end_for_extra
            extra_only = self._fetch_extra_factors(
                securities, start_for_extra, end_for_extra, custom_fields
            )
            if "factor" in requested:
                factors = self._fetch_factor_separately(
                    securities, start_for_extra, end_for_extra
                )
                if factors is None or factors.empty:
                    raise ValueError("pit_factor_missing")
                extra_only = (
                    factors
                    if extra_only.empty
                    else extra_only.merge(
                        factors, on=["date", "code"], how="outer", validate="one_to_one"
                    )
                )
            if extra_only is None or extra_only.empty:
                return pd.DataFrame()
            try:
                extra_only["date"] = pd.to_datetime(extra_only["date"])
            except Exception:
                if self.strict:
                    raise
                pass
            extra_only = extra_only.sort_values(["code", "date"])
            if use_full_interval:
                return extra_only
            return extra_only.groupby("code", group_keys=False).tail(desired_count)

        batches = self._security_batches_for_price(securities, desired_count)
        res = self._concat_get_price_batches(
            batches,
            end_date=end_dt,
            frequency=use_freq,
            **(
                {"start_date": self.expanding_start_date}
                if self.use_expanding_window and self.expanding_start_date
                else {"count": desired_count}
            ),
            fields=req_fields,
            skip_paused=skip_paused_flag,
            fill_paused=False,
            panel=False,
            fq=fq_param,
        )

        if res is None:
            return pd.DataFrame()

        # Unify return to DataFrame with columns ['date','code', *fields]
        try:
            if isinstance(res, pd.DataFrame):
                df = res.copy()
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                    if "time" in df.columns and "date" not in df.columns:
                        df = df.rename(columns={"time": "date"})
                    if "security" in df.columns and "code" not in df.columns:
                        df = df.rename(columns={"security": "code"})
                else:
                    df = df.reset_index()
                    if "time" in df.columns and "date" not in df.columns:
                        df = df.rename(columns={"time": "date"})
                # Handle MultiIndex columns from panel=False multi-security fetch
                if isinstance(df.columns, pd.MultiIndex):
                    # Columns are (field, security) tuples; need to stack to long form
                    cols = list(df.columns)
                    # Determine which level is field vs security
                    try:
                        field_universe = set(self._WHITELIST.get(use_freq, set()))
                        field_universe.update(
                            {
                                "avg",
                                "pre_close",
                                "high_limit",
                                "low_limit",
                                "paused",
                                "factor",
                            }
                        )
                    except Exception:
                        if self.strict:
                            raise
                        field_universe = {
                            "open",
                            "close",
                            "high",
                            "low",
                            "volume",
                            "money",
                        }
                    sample = cols[0]
                    lvl_field = None
                    if isinstance(sample, tuple) and len(sample) >= 2:
                        if sample[0] in field_universe:
                            lvl_field = 0
                        elif sample[1] in field_universe:
                            lvl_field = 1
                    if lvl_field is None:
                        lvl_field = 0
                    lvl_code = 1 if lvl_field == 0 else 0
                    # Stack security level to convert to long form
                    df = df.set_index("date") if "date" in df.columns else df
                    stacked = df.stack(level=lvl_code).reset_index()
                    code_col = "level_{}".format(lvl_code)
                    if code_col in stacked.columns:
                        stacked = stacked.rename(columns={code_col: "code"})
                    if "date" not in stacked.columns and "index" in stacked.columns:
                        stacked = stacked.rename(columns={"index": "date"})
                    if "date" in stacked.columns:
                        stacked["date"] = pd.to_datetime(stacked["date"])
                    df = stacked
                # If single-security and no code column, inject it
                if "code" not in df.columns:
                    codeval = None
                    if isinstance(securities, (list, tuple)) and len(securities) == 1:
                        codeval = securities[0]
                    elif isinstance(securities, str):
                        codeval = securities
                    if codeval is not None:
                        df["code"] = codeval
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                # Attach custom factor fields if requested
                # Keep only full days for intraday
                result = self._trim_to_full_days(df, bars_pd)
                if historical_daily_fields and result is not None and not result.empty:
                    daily = self._fetch_historical_daily_price_fields(
                        securities, result["date"], historical_daily_fields
                    )
                    if daily is not None and not daily.empty:
                        result["_merge_date"] = pd.to_datetime(result["date"]).dt.floor(
                            "D"
                        )
                        daily["_merge_date"] = pd.to_datetime(daily["date"]).dt.floor(
                            "D"
                        )
                        daily = daily.drop(columns=["date"]).drop_duplicates(
                            ["_merge_date", "code"], keep="last"
                        )
                        result = pd.merge(
                            result, daily, on=["_merge_date", "code"], how="left"
                        ).drop(columns=["_merge_date"])
                # For research mode: fetch factor separately and merge
                if research_mode and needs_factor:
                    try:
                        start_dt_factor = result["date"].min()
                        end_dt_factor = result["date"].max()
                        factor_df = self._fetch_factor_separately(
                            securities, start_dt_factor, end_dt_factor
                        )
                        if factor_df is not None and not factor_df.empty:
                            # Remove existing 'factor' column if present (fq=None returns factor=1.0)
                            # to avoid merge conflict (factor_x, factor_y)
                            if "factor" in result.columns:
                                result = result.drop(columns=["factor"])
                            # Merge factor on date (floor to day) and code
                            result["_merge_date"] = pd.to_datetime(
                                result["date"]
                            ).dt.floor("D")
                            result = pd.merge(
                                result,
                                factor_df.rename(columns={"date": "_merge_date"}),
                                on=["_merge_date", "code"],
                                how="left",
                            )
                            result = result.drop(columns=["_merge_date"])
                        else:
                            raise ValueError("pit_factor_missing")
                    except Exception:
                        if self.strict:
                            raise
                        raise ValueError("pit_factor_missing")
                    # Apply Dynamic PIT adjustment
                    if needs_pit:
                        result = self._apply_dynamic_pit(result)
                return result
            else:
                # Assume Panel-like: items=fields, major=time, minor=security
                merged = None
                # Use requested usable fields when available; otherwise infer from panel items
                merge_fields = (
                    req_fields if req_fields else list(getattr(res, "items", []))
                )
                for f in merge_fields:
                    try:
                        dff = res[f]
                    except Exception:
                        if self.strict:
                            raise
                        continue
                    s = dff.stack(dropna=False).reset_index()
                    cols = list(s.columns)
                    if len(cols) >= 3:
                        s = s.rename(
                            columns={cols[0]: "date", cols[1]: "code", cols[2]: f}
                        )
                    else:
                        continue
                    if merged is None:
                        merged = s
                    else:
                        merged = pd.merge(merged, s, on=["date", "code"], how="outer")
                if merged is None:
                    return pd.DataFrame()
                merged["date"] = pd.to_datetime(merged["date"])
                df = merged
                # Attach custom factor fields if requested
                result = self._trim_to_full_days(df, bars_pd)
                # For research mode: fetch factor separately and merge
                if research_mode and needs_factor:
                    try:
                        start_dt_factor = result["date"].min()
                        end_dt_factor = result["date"].max()
                        factor_df = self._fetch_factor_separately(
                            securities, start_dt_factor, end_dt_factor
                        )
                        if factor_df is not None and not factor_df.empty:
                            # Remove existing 'factor' column if present (fq=None returns factor=1.0)
                            # to avoid merge conflict (factor_x, factor_y)
                            if "factor" in result.columns:
                                result = result.drop(columns=["factor"])
                            result["_merge_date"] = pd.to_datetime(
                                result["date"]
                            ).dt.floor("D")
                            result = pd.merge(
                                result,
                                factor_df.rename(columns={"date": "_merge_date"}),
                                on=["_merge_date", "code"],
                                how="left",
                            )
                            result = result.drop(columns=["_merge_date"])
                        else:
                            raise ValueError("pit_factor_missing")
                    except Exception:
                        if self.strict:
                            raise
                        raise ValueError("pit_factor_missing")
                    # Apply Dynamic PIT adjustment
                    if needs_pit:
                        result = self._apply_dynamic_pit(result)
                return result
        except Exception:
            if self.strict:
                raise
            return pd.DataFrame()

    def normalize_for_factor(self, df, factor_cls=None):
        result = df.copy()
        factor_data_fields = []
        try:
            fields = getattr(factor_cls, "FACTOR_DATA_FIELDS", None)
            if fields:
                factor_data_fields.extend(list(fields))
        except Exception:
            pass
        try:
            cls_name = getattr(factor_cls, "__name__", "")
            prefixed_name = "{}_FACTOR_DATA_FIELDS".format(cls_name) if cls_name else ""
            for global_name in (prefixed_name, "FACTOR_DATA_FIELDS"):
                fields = globals().get(global_name)
                if fields:
                    factor_data_fields.extend(list(fields))
        except Exception:
            pass
        for raw_name in sorted(set(factor_data_fields)):
            feature_name = "fd_{}".format(raw_name)
            if raw_name in result.columns and feature_name not in result.columns:
                result[feature_name] = result[raw_name]
        return result

    def run_factor(
        self,
        factor_cls,
        df,
        monthly_stock_pools=None,
        use_dynamic_pool=False,
        output_date=None,
        stock_pools_by_period=None,
    ):
        if monthly_stock_pools is not None and stock_pools_by_period is not None:
            raise ValueError("Use only stock_pools_by_period or monthly_stock_pools, not both")
        if stock_pools_by_period is None:
            stock_pools_by_period = monthly_stock_pools
        stock_pool = sorted(df["code"].unique())
        factor_date = pd.to_datetime(output_date) if output_date is not None else None
        init_kwargs = dict(
            db_connection=None,
            factor_name=factor_cls.__name__,
            start_date=str(
                (factor_date if factor_date is not None else df["date"].min()).date()
            ),
            end_date=str(
                (factor_date if factor_date is not None else df["date"].max()).date()
            ),
            stock_pool=stock_pool,
            data_frequency=self.freq,
        )
        fac = factor_cls(**init_kwargs)
        local_duckdb_path = getattr(self, "duckdb_path", None)
        if local_duckdb_path:
            # Local alignment adapters already own the canonical read-only DB.
            # Expose its path to factors with auxiliary PIT tables without
            # making FactorBase open a redundant adapter for every replay date.
            fac.duckdb_path = str(local_duckdb_path)
            fac.data_bindings = getattr(self, "data_bindings", None)
        if stock_pools_by_period is not None:
            fac.stock_pools_by_period = stock_pools_by_period
            fac.use_dynamic_pool = True
            all_stocks_set = set()
            for stocks in stock_pools_by_period.values():
                all_stocks_set.update(stocks)
            fac.all_stocks = list(all_stocks_set)
        try:
            self.last_factor_instance = fac
        except Exception:
            pass
        df = self.normalize_for_factor(df, factor_cls)
        result = fac.calculate_factor(df)
        if isinstance(result, pd.DataFrame) and hasattr(result, "index"):
            if not isinstance(result.index, pd.MultiIndex):
                if result.index.name is None or result.index.name == "":
                    result.index.name = "date"
        return result
