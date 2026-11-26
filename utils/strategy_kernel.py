"""Pure decision functions shared by local and generated cloud strategies."""

import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_FLOOR


def preprocess_series(
    series, method="zscore", percentile=0.025, min_samples=0, zero_degenerate=True
):
    if not 0 <= percentile < 0.5:
        raise ValueError("invalid_winsorize_percentile")
    if method in (None, "", "none") or series.empty:
        return series.copy()
    if method == "rank":
        return series.rank(pct=True, method="average")
    if method == "rank_zscore":
        values = series.rank(pct=True, method="average")
    elif method in ("zscore", "winsorize_zscore"):
        if series.notna().sum() < min_samples:
            return series - series.mean()
        values = series.clip(
            lower=series.quantile(percentile), upper=series.quantile(1 - percentile)
        )
    elif method == "raw_zscore":
        values = series
    elif method == "gauss_rank":
        from scipy.stats import norm

        n = series.notna().sum()
        if n < 2:
            return series * 0.0
        return pd.Series(
            norm.ppf(series.rank(pct=True) * n / (n + 1)), index=series.index
        )
    else:
        raise ValueError("unsupported_preprocessing: " + str(method))
    mean, std = values.mean(), values.std()
    if pd.isna(std) or std == 0 or (not zero_degenerate and std < 1e-8):
        return values.where(values.isna(), 0.0) if zero_degenerate else values - mean
    return (values - mean) / std


TIME_SERIES_METHODS = frozenset(["time_series_zscore", "time_series_rank"])


def resolve_preprocess_method(methods, name, default="zscore"):
    cfg = methods if hasattr(methods, "get") else {}
    sentinel = object()
    mode = cfg.get(name, sentinel)
    if mode is not sentinel:
        return "none" if mode in (None, "") else mode
    fallback = cfg.get("ALL", sentinel)
    if fallback is not sentinel:
        return "none" if fallback in (None, "") else fallback
    return default


def resolve_time_series_params(methods=None):
    cfg = methods if hasattr(methods, "get") else {}
    window = cfg.get("_ts_window", 20)
    if window is None:
        window = 20
    min_periods = cfg.get("_ts_min_periods", None)
    if min_periods is None:
        min_periods = window
    return window, min_periods


def _validate_time_series_params(method, window, min_periods):
    from numbers import Integral

    minimum = window if min_periods is None else min_periods
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in (window, minimum)):
        raise ValueError("Time-series window and min_periods must be integers")
    if not 2 <= minimum <= window:
        raise ValueError("Time-series preprocessing requires 2 <= min_periods <= window")
    if method not in TIME_SERIES_METHODS:
        raise ValueError("Unknown time-series preprocessing method: " + str(method))
    return int(window), int(minimum)


def time_series_standardize(values, method, window=20, min_periods=None):
    """Compare each observation with only its instrument's preceding observations."""
    window, minimum = _validate_time_series_params(method, window, min_periods)
    if not values.index.is_unique:
        raise ValueError("Time-series preprocessing requires unique date and instrument keys")

    def normalize(group):
        if method == "time_series_zscore":
            history = group.shift(1).rolling(window, min_periods=minimum)
            std = history.std(ddof=0)
            return ((group - history.mean()) / std.where(std > 0)).mask(
                std.eq(0) & group.notna(), 0.0
            )

        def percentile(sample):
            current = sample[-1]
            history = sample[:-1]
            if np.isnan(current):
                return np.nan
            below = np.count_nonzero(history < current)
            ties = np.count_nonzero(history == current)
            return (below + 0.5 * ties) / np.count_nonzero(~np.isnan(history))

        return group.rolling(window + 1, min_periods=minimum + 1).apply(
            percentile, raw=True
        )

    cleaned = values.replace([np.inf, -np.inf], np.nan)
    if isinstance(cleaned, pd.DataFrame):
        ordered = cleaned.sort_index()
        return ordered.apply(normalize).reindex(values.index)
    if isinstance(cleaned.index, pd.MultiIndex):
        ordered = cleaned.sort_index(level="date")
        return ordered.groupby(level="stock_code", sort=False).transform(normalize).reindex(values.index)
    ordered = cleaned.sort_index()
    return normalize(ordered).reindex(values.index)


def _align_timestamp_to_index(ts, index):
    target = pd.Timestamp(ts)
    if hasattr(index, "tz") and index.tz is not None:
        if target.tzinfo is None:
            return target.tz_localize(index.tz)
        return target.tz_convert(index.tz)
    if target.tzinfo is not None:
        return target.tz_convert("UTC").tz_localize(None)
    return target


def _normalize_ts_history_panel(history, factor_names, stocks, asof=None):
    """Canonicalize MultiIndex or wide-dict history to sparse (date, stock_code) <= asof."""
    if isinstance(history, pd.DataFrame) and isinstance(history.index, pd.MultiIndex):
        active_cols = (
            [c for c in factor_names if c in history.columns]
            if factor_names is not None
            else list(history.columns)
        )
        date_level = "date" if "date" in history.index.names else 0
        stock_level = "stock_code" if "stock_code" in history.index.names else 1
        date_vals = pd.DatetimeIndex(history.index.get_level_values(date_level))
        stock_vals = history.index.get_level_values(stock_level)
        if asof is not None:
            asof_ts = _align_timestamp_to_index(asof, date_vals)
        else:
            if history.empty:
                return (
                    pd.DataFrame(
                        columns=active_cols,
                        index=pd.MultiIndex.from_tuples([], names=["date", "stock_code"]),
                        dtype=float,
                    ),
                    None,
                )
            asof_ts = date_vals.max()
        keep_mask = (date_vals <= asof_ts) & stock_vals.isin(stocks)
        sub_idx = pd.MultiIndex.from_arrays(
            [date_vals[keep_mask], stock_vals[keep_mask]],
            names=["date", "stock_code"],
        )
        if not sub_idx.is_unique:
            raise ValueError(
                "Time-series preprocessing requires unique date and instrument keys"
            )
        if not len(sub_idx) or not active_cols:
            return (
                pd.DataFrame(
                    columns=active_cols,
                    index=pd.MultiIndex.from_tuples([], names=["date", "stock_code"]),
                    dtype=float,
                ),
                asof_ts,
            )
        sub = history.loc[keep_mask]
        frame = pd.DataFrame(
            {
                c: np.asarray(pd.to_numeric(sub[c], errors="coerce"), dtype=float)
                for c in active_cols
            },
            index=sub_idx,
        )
        return frame.dropna(how="all").sort_index(), asof_ts

    if hasattr(history, "get"):
        active_cols = (
            [c for c in factor_names if c in history]
            if factor_names is not None
            else sorted(k for k, v in history.items() if isinstance(v, pd.DataFrame))
        )
        parsed = {}
        for col_name in active_cols:
            wide = history[col_name]
            if not isinstance(wide, pd.DataFrame):
                raise ValueError("time_series_history_invalid: " + str(col_name))
            wide_idx = pd.DatetimeIndex(wide.index)
            if not wide_idx.is_unique:
                raise ValueError(
                    "Time-series preprocessing requires unique date and instrument keys"
                )
            parsed[col_name] = wide.set_axis(wide_idx, axis=0).sort_index()
        if asof is not None:
            ref_idx = next(
                (w.index for w in parsed.values() if not w.empty),
                next(
                    (w.index for w in parsed.values()),
                    pd.DatetimeIndex([]),
                ),
            )
            asof_ts = _align_timestamp_to_index(asof, ref_idx)
        else:
            max_dates = [w.index.max() for w in parsed.values() if not w.empty]
            if not max_dates:
                return (
                    pd.DataFrame(
                        columns=active_cols,
                        index=pd.MultiIndex.from_tuples([], names=["date", "stock_code"]),
                        dtype=float,
                    ),
                    None,
                )
            asof_ts = max(max_dates)

        series_map = {}
        for col_name in active_cols:
            wide = parsed[col_name]
            col_asof = _align_timestamp_to_index(asof_ts, wide.index)
            wide = wide.loc[wide.index <= col_asof]
            if wide.empty:
                continue
            wide = wide.reindex(columns=stocks)
            numeric = wide.apply(pd.to_numeric, errors="coerce")
            arr = np.asarray(numeric.values, dtype=float)
            rows_i, cols_j = np.nonzero(~np.isnan(arr))
            if len(rows_i) == 0:
                continue
            idx = pd.MultiIndex.from_arrays(
                [wide.index.take(rows_i), wide.columns.take(cols_j)],
                names=["date", "stock_code"],
            )
            series_map[col_name] = pd.Series(arr[rows_i, cols_j], index=idx)

        if series_map:
            frame = pd.concat(series_map, axis=1).reindex(columns=active_cols).sort_index()
            frame.index = frame.index.set_names(["date", "stock_code"])
        else:
            frame = pd.DataFrame(
                columns=active_cols,
                index=pd.MultiIndex.from_tuples([], names=["date", "stock_code"]),
                dtype=float,
            )
        return frame, asof_ts

    raise ValueError(
        "time_series_history_invalid: expected MultiIndex DataFrame or factor-panel mapping"
    )


def time_series_standardize_at_asof(
    raw_series, name, method, methods=None, history=None, asof=None, active_columns=None
):
    """Evaluate causal time-series standardization at ``asof`` from strict prior history."""
    if history is None:
        raise ValueError(
            "time_series_history_required: " + str(name) + " (" + str(method) + ")"
        )
    window, min_periods = resolve_time_series_params(methods)
    _validate_time_series_params(method, window, min_periods)
    if raw_series.empty:
        return raw_series.copy()
    if not raw_series.index.is_unique:
        raise ValueError("Time-series preprocessing requires unique instrument keys")
    current_vals = np.asarray(pd.to_numeric(raw_series, errors="coerce"), dtype=float)

    if isinstance(history, pd.DataFrame) and isinstance(history.index, pd.MultiIndex):
        if name not in history.columns:
            raise ValueError("model_features_missing: " + str(name))
    elif hasattr(history, "get"):
        if name not in history:
            raise ValueError(
                "time_series_history_required: " + str(name) + " (" + str(method) + ")"
            )
        if not isinstance(history[name], pd.DataFrame):
            raise ValueError("time_series_history_invalid: " + str(name))
    else:
        raise ValueError(
            "time_series_history_invalid: expected MultiIndex DataFrame or factor-panel mapping"
        )

    panel, asof_ts = _normalize_ts_history_panel(
        history, active_columns, raw_series.index, asof=asof
    )
    if asof_ts is None:
        return pd.Series(np.nan, index=raw_series.index, dtype=float)

    if not panel.empty and name in panel.columns:
        col = panel[name]
        col_dates = pd.DatetimeIndex(col.index.get_level_values("date"))
        asof_ts = _align_timestamp_to_index(asof_ts, col_dates)
        prior_col = col.loc[col_dates < asof_ts]
    else:
        prior_col = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_tuples([], names=["date", "stock_code"]),
        )

    current_idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([asof_ts] * len(raw_series)), raw_series.index],
        names=["date", "stock_code"],
    )
    current_col = pd.Series(current_vals, index=current_idx)
    combined = current_col if prior_col.empty else pd.concat([prior_col, current_col])
    standardized = time_series_standardize(
        combined, method, window=window, min_periods=min_periods
    )
    return standardized.xs(asof_ts, level="date").reindex(raw_series.index)


def preprocess_cross_section_series(series, method, methods=None):
    cfg = methods if hasattr(methods, "get") else {}
    if method in (None, "", "none"):
        return series.copy()
    if method == "binning":
        return bin_series(series, cfg.get("_binning_n_bins", 5))
    percentile = (
        cfg.get("_winsorize_percentile", 0.025)
        if method in ("zscore", "winsorize_zscore")
        else 0.025
    )
    return preprocess_series(series, method, percentile)


def standardize_composite_scores(scores):
    valid = scores.dropna()
    if len(valid) > 1:
        standardized = (valid - valid.mean()) / (valid.std() + 1e-8)
        result = scores.copy()
        result.loc[valid.index] = standardized
        return result
    return scores


def valid_factor_mask(values, factor_names):
    cols = [name for name in factor_names if name in values.columns]
    if not cols:
        return pd.Series(False, index=values.index, dtype=bool)
    return values.reindex(columns=cols).notna().any(axis=1)


def model_for_date(models, asof, date_field="train_end", inclusive=False):
    key = str(asof)[:10]
    eligible = [
        m
        for m in models
        if str(m[date_field])[:10] < key
        or (inclusive and str(m[date_field])[:10] == key)
    ]
    return max(eligible, key=lambda m: str(m[date_field])) if eligible else None


def weighted_score(values, weights):
    unknown = set(weights) - set(values.columns)
    if unknown:
        raise ValueError("model_features_missing: " + ",".join(sorted(unknown)))
    w = pd.Series(weights, dtype=float).sort_index()
    if not np.isfinite(w.values).all():
        raise ValueError("model_weights_nonfinite")
    if w.empty or values.empty:
        return pd.Series(dtype=float, index=values.index[:0])
    valid = valid_factor_mask(values, w.index)
    if not valid.any():
        return pd.Series(dtype=float, index=values.index[:0])
    return (
        values.loc[valid]
        .reindex(columns=w.index)
        .fillna(0.0)
        .mul(w, axis=1)
        .sum(axis=1)
    )


def rank_scores(scores):
    frame = scores.dropna().rename("score").to_frame()
    frame.index = frame.index.rename(None)
    frame["stock"] = frame.index.astype(str)
    return frame.sort_values(
        ["score", "stock"], ascending=[False, True], kind="mergesort"
    ).index.tolist()


def selection_count(n_valid, top_k=None, top_pct=None):
    if n_valid <= 0:
        return 0
    count = (
        top_k
        if top_k is not None
        else (
            max(1, int((Decimal(str(n_valid)) * Decimal(str(top_pct)))
                       .to_integral_value(rounding=ROUND_FLOOR)))
            if top_pct is not None else 5
        )
    )
    if count < 0:
        raise ValueError("invalid_target_count")
    return min(int(count), n_valid)


def equal_target_weights(stocks, slots=None):
    unique = list(dict.fromkeys(stocks))
    denominator = len(unique) if slots is None else int(slots)
    if denominator < len(unique):
        raise ValueError("target_count_exceeds_slots")
    return {stock: 1.0 / denominator for stock in unique} if unique else {}


def bin_series(series, n_bins=5):
    if series.notna().sum() < n_bins:
        return series.rank(pct=True)
    bins = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    return bins / bins.max() if bins.max() > 0 else series.rank(pct=True)


def linear_signal(raw, model, parent_scores=None, history=None, asof=None):
    """Frozen-model replay, using unified local/VectorBT synthesis semantics."""
    weights = model.get("weights", {}) if not model.get("blend_only", False) else {}
    blend = model.get("blend_weights", {})
    ratio = float(model.get("blend_with_original_ratio", 0.0))
    if not 0 <= ratio <= 1:
        raise ValueError("invalid_blend_ratio")
    if not weights and parent_scores is not None:
        if ratio <= 0:
            raise ValueError("parent-only bundle model requires a positive blend ratio")
        return parent_scores, None
    needed = set(weights) | (
        set(blend) if ratio > 0 and parent_scores is None else set()
    )
    missing = needed - set(raw.columns)
    if missing:
        raise ValueError("model_features_missing: " + ",".join(sorted(missing)))
    methods = model.get("standardization_method", {})
    pre_fill = pd.DataFrame(index=raw.index)
    standardize = model.get("standardize", True)
    needed_sorted = sorted(needed)
    normalized_history = None
    normalized_asof = asof
    for name in needed_sorted:
        method = resolve_preprocess_method(methods, name, default="zscore")
        if not standardize:
            values = raw[name]
        elif method in TIME_SERIES_METHODS:
            if normalized_history is None and history is not None:
                if (
                    isinstance(history, pd.DataFrame)
                    and isinstance(history.index, pd.MultiIndex)
                    and name in history.columns
                ) or (
                    hasattr(history, "get")
                    and name in history
                    and isinstance(history[name], pd.DataFrame)
                ):
                    normalized_history, normalized_asof = _normalize_ts_history_panel(
                        history, needed_sorted, raw.index, asof=asof
                    )
            ts_hist = (
                normalized_history
                if normalized_history is not None and name in normalized_history.columns
                else history
            )
            ts_asof = normalized_asof if ts_hist is normalized_history else asof
            values = time_series_standardize_at_asof(
                raw[name],
                name,
                method,
                methods,
                history=ts_hist,
                asof=ts_asof,
                active_columns=needed_sorted,
            )
        else:
            values = preprocess_cross_section_series(raw[name], method, methods)
        pre_fill[name] = values
    valid_any = valid_factor_mask(pre_fill, needed)
    processed = pre_fill.copy()
    if valid_any.any():
        processed.loc[valid_any] = processed.loc[valid_any].fillna(0.0)
    if weights:
        score = weighted_score(pre_fill, weights)
        if model.get("postprocess_cs_standardize", True):
            score = standardize_composite_scores(score)
    else:
        score = pd.Series(dtype=float, index=raw.index[:0])
    if ratio > 0 and parent_scores is not None:
        parent = parent_scores.reindex(raw.index).dropna()
        if parent.empty:
            raise ValueError("parent baseline has no residual-score overlap")
        score = score.reindex(parent.index).fillna(0.0) * (1 - ratio) + parent * ratio
    elif ratio > 0 and blend:
        blend_score = weighted_score(pre_fill, blend)
        if weights:
            combined_idx = raw.index[
                valid_factor_mask(pre_fill, weights)
                | valid_factor_mask(pre_fill, blend)
            ]
            score = (
                score.reindex(combined_idx).fillna(0.0) * (1 - ratio)
                + blend_score.reindex(combined_idx).fillna(0.0) * ratio
            )
        else:
            score = blend_score * ratio
    return score, processed


def input_digest(frame):
    import hashlib

    values = frame.sort_values(["date", "code"], kind="mergesort").copy()
    values["date"] = pd.to_datetime(values.date).dt.strftime("%Y-%m-%d %H:%M:%S")
    observed_fields = [c for c in ("open", "high", "low", "close", "volume") if c in values]
    if observed_fields:
        empty = values[observed_fields].isna().all(axis=1)
        if "money" in values:
            empty &= values.money.isna() | values.money.eq(0)
        # Local storage uses zero turnover / placeholder factors for empty
        # paused bars; JQ returns NaN. Preserve raw evidence and canonicalize
        # only those empty bars, never observed prices or positive turnover.
        for column in ("money", "factor"):
            if column in values:
                values.loc[empty, column] = float("nan")
    # Adjustment coefficients have a provider-specific absolute base. Compare
    # their relative path at the request anchor; retained inputs keep raw bytes.
    if "factor" in values:
        factors = values["factor"].astype("float64")
        values["factor"] = factors / factors.groupby(values["code"]).transform("last")
    for column in values:
        if pd.api.types.is_numeric_dtype(values[column]):
            # Eight significant digits stay stricter than the 1e-6 comparison
            # tolerance while ignoring provider/PIT roundoff (~1e-12). This
            # canonicalizes evidence only; calculation frames remain unchanged.
            # Fixed scientific precision also avoids old Python's differing
            # removal of trailing zeros in general-format exponent notation.
            values[column] = values[column].map(
                lambda value: (
                    None
                    if pd.isna(value)
                    else (format(float(value), ".7e") if value else "0")
                )
            )
    return hashlib.sha256(
        values.reindex(sorted(values.columns), axis=1)
        .to_json(orient="split", index=False)
        .encode("utf-8")
    ).hexdigest()


def snapshot_records(fingerprint, day, stage, values):
    import json

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if hasattr(value, "item"):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    items = sorted(values.items())
    for offset in range(0, len(items), 10):
        yield json.dumps(
            {
                "fingerprint": fingerprint,
                "date": str(day)[:10],
                "stage": stage,
                "key": "__batch__:%d" % (offset // 10),
                "value": {
                    str(key): clean(value) for key, value in items[offset : offset + 10]
                },
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )


def signal_snapshot(
    fingerprint,
    day,
    factor_date,
    pool,
    raw,
    processed,
    scores,
    targets,
    input_hashes=None,
    target_slots=None,
    target_weights=None,
):
    stages = {
        "inputs": {"pool": sorted(pool), "asof": str(factor_date)[:10]},
        "factor": raw.to_dict("index"),
        "score": scores.to_dict(),
        "targets": {
            "weights": equal_target_weights(targets, target_slots) if target_weights is None else target_weights,
            "ranking": rank_scores(scores),
        },
    }
    if input_hashes is not None:
        stages["inputs"]["input_signatures"] = input_hashes
        stages["inputs"]["signature_rule"] = "pit_relative_empty_bars_8_significant_scientific"
    if processed is not None:
        stages["preprocess"] = processed.to_dict("index")
    for stage, values in stages.items():
        for line in snapshot_records(fingerprint, day, stage, values):
            yield line
