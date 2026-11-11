"""Shared symbolic DSL runtime used by mining and generated factors."""

from __future__ import annotations

import ast

RUNTIME_SOURCE = r"""
import numpy as np
import pandas as pd


def _as_series(value, index=None):
    if isinstance(value, pd.Series):
        return value
    if np.isscalar(value):
        return pd.Series(float(value), index=index)
    return pd.Series(value, index=index)


def _window(value):
    return max(1, int(round(float(value))))


def safe_div(a, b):
    if isinstance(b, pd.Series):
        return a / b.replace(0, np.nan)
    return a / (b + 1e-12)


def neg(a):
    if isinstance(a, pd.Series):
        return -pd.to_numeric(a, errors="coerce").astype(float)
    if np.isscalar(a):
        return -float(a)
    return -np.asarray(a, dtype=float)


def log(a):
    if isinstance(a, pd.Series):
        return np.log(a.abs().clip(lower=1e-12))
    return np.log(max(abs(float(a)), 1e-12))


def sqrt(a):
    if isinstance(a, pd.Series):
        return np.sqrt(a.clip(lower=0))
    return np.sqrt(max(float(a), 0.0))


def exp(a):
    if isinstance(a, pd.Series):
        return np.exp(a.clip(lower=-50.0, upper=50.0))
    return np.exp(max(-50.0, min(50.0, float(a))))


def floor(a):
    return np.floor(a)


def inv(a):
    return safe_div(1.0, a)


def element_max(a, b):
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        index = a.index if isinstance(a, pd.Series) else b.index
        return pd.Series(a, index=index).combine(
            pd.Series(b, index=index),
            lambda x, y: x if x is pd.NA or y is pd.NA else max(x, y),
        )
    return a if a is pd.NA or b is pd.NA else max(float(a), float(b))


def element_min(a, b):
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        index = a.index if isinstance(a, pd.Series) else b.index
        return pd.Series(a, index=index).combine(
            pd.Series(b, index=index),
            lambda x, y: x if x is pd.NA or y is pd.NA else min(x, y),
        )
    return a if a is pd.NA or b is pd.NA else min(float(a), float(b))


def sign(a):
    return np.sign(a)


def power(a, b):
    if isinstance(a, pd.Series):
        return np.sign(a) * np.power(a.abs(), float(b))
    return np.sign(float(a)) * (abs(float(a)) ** float(b))


def pow(a, b):
    return power(a, b)


def clip(a, limit):
    bound = abs(float(limit))
    if isinstance(a, pd.Series):
        return a.clip(lower=-bound, upper=bound)
    return max(-bound, min(bound, float(a)))


def winsorize(a, limit):
    if not isinstance(a, pd.Series):
        return a
    q = min(max(float(limit), 0.0), 0.49)
    lo = a.quantile(q)
    hi = a.quantile(1.0 - q)
    return a.clip(lower=lo, upper=hi)


def cs_rank(a, dates):
    series = pd.Series(a)
    return series.groupby(dates).rank(pct=True)


def cs_zscore(a, dates):
    series = pd.Series(a)
    grouped = series.groupby(dates)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (series - mean) / std


def cs_demean(a, dates):
    series = pd.Series(a)
    return series - series.groupby(dates).transform("mean")


def cs_scale(a, dates):
    series = pd.Series(a)
    denom = series.abs().groupby(dates).transform("sum").replace(0, np.nan)
    return series / denom


def cs_quantile(a, dates):
    return cs_rank(a, dates)


def _cs_transform(a, dates, method):
    return pd.Series(a).groupby(dates).transform(method)


def cs_mean(a, dates):
    return _cs_transform(a, dates, "mean")


def cs_std(a, dates):
    return _cs_transform(a, dates, "std")


def cs_skew(a, dates):
    return _cs_transform(a, dates, "skew")


def cs_kurt(a, dates):
    return pd.Series(a).groupby(dates).transform(lambda item: item.kurt())


def cs_max(a, dates):
    return _cs_transform(a, dates, "max")


def cs_min(a, dates):
    return _cs_transform(a, dates, "min")


def cs_median(a, dates):
    return _cs_transform(a, dates, "median")


def cs_scale_to(a, target, dates):
    return cs_scale(a, dates) * float(target)


def _rolling_unary(a, codes, window, func):
    w = _window(window)
    series = pd.Series(a)
    return series.groupby(codes, group_keys=False).transform(lambda item: func(item.rolling(w, min_periods=w)))


def _rolling_grouped_stat(a, codes, window, method):
    w = _window(window)
    series = pd.Series(a)
    result = getattr(series.groupby(codes, sort=False).rolling(w, min_periods=w), method)()
    return result.reset_index(level=0, drop=True).sort_index()


def _rolling_full_window_transform(a, codes, window, transform):
    w = _window(window)
    series = pd.Series(a)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if series.empty:
        return out

    for _, group in series.groupby(codes, sort=False):
        values = group.to_numpy(dtype=float, copy=True)
        if values.size < w:
            continue
        windows = np.lib.stride_tricks.as_strided(
            values,
            shape=(values.size - w + 1, w),
            strides=(values.strides[0], values.strides[0]),
        )
        valid = ~np.isnan(windows).any(axis=1)
        rolling_values = np.full(windows.shape[0], np.nan, dtype=float)
        if valid.any():
            rolling_values[valid] = np.asarray(transform(windows[valid]), dtype=float)
        group_values = np.full(values.size, np.nan, dtype=float)
        group_values[w - 1 :] = rolling_values
        out.loc[group.index] = group_values
    return out


def delay(a, window, codes):
    return pd.Series(a).groupby(codes).shift(_window(window))


def delta(a, window, codes):
    series = pd.Series(a)
    return series - delay(series, window, codes)


def pct_change(a, window, codes):
    lag = delay(a, window, codes)
    return safe_div(pd.Series(a) - lag, lag.abs())


def ts_pctchange(a, window, codes):
    return pct_change(a, window, codes)


def ts_mean(a, window, codes):
    return _rolling_grouped_stat(a, codes, window, "mean")


def ts_median(a, window, codes):
    return _rolling_grouped_stat(a, codes, window, "median")


def ts_sum(a, window, codes):
    return _rolling_grouped_stat(a, codes, window, "sum")


def ts_cumsum(a, window, codes):
    return ts_sum(a, window, codes)


def ts_product(a, window, codes):
    return _rolling_full_window_transform(a, codes, window, lambda windows: np.prod(windows, axis=1))


def prod(a, window, codes):
    return ts_product(a, window, codes)


def ts_std(a, window, codes):
    return _rolling_grouped_stat(a, codes, window, "std")


def ts_var(a, window, codes):
    return _rolling_grouped_stat(a, codes, window, "var")


def ts_quantile(a, window, quantile, codes):
    q = min(max(float(quantile), 0.0), 1.0)
    return _rolling_full_window_transform(a, codes, window, lambda windows: np.quantile(windows, q, axis=1))


def percentile(a, quantile, window, codes):
    return ts_quantile(a, window, quantile, codes)


def ts_mad(a, window, codes):
    return _rolling_full_window_transform(
        a,
        codes,
        window,
        lambda windows: np.median(
            np.abs(windows - np.median(windows, axis=1)[:, None]),
            axis=1,
        ),
    )


def ts_min(a, window, codes):
    return _rolling_unary(a, codes, window, lambda rolling: rolling.min())


def ts_max(a, window, codes):
    return _rolling_unary(a, codes, window, lambda rolling: rolling.max())


def ts_argmin(a, window, codes):
    return _rolling_full_window_transform(
        a,
        codes,
        window,
        lambda windows: np.argmin(windows, axis=1) + 1.0,
    )


def ts_argmax(a, window, codes):
    return _rolling_full_window_transform(
        a,
        codes,
        window,
        lambda windows: np.argmax(windows, axis=1) + 1.0,
    )


def highday(a, window, codes):
    return _rolling_full_window_transform(
        a,
        codes,
        window,
        lambda windows: windows.shape[1] - 1.0 - np.argmax(windows, axis=1),
    )


def lowday(a, window, codes):
    return _rolling_full_window_transform(
        a,
        codes,
        window,
        lambda windows: windows.shape[1] - 1.0 - np.argmin(windows, axis=1),
    )


def ts_rank(a, window, codes):
    def rank_last(windows):
        last = windows[:, [-1]]
        less = (windows < last).sum(axis=1)
        equal = (windows == last).sum(axis=1)
        return (less + (equal + 1.0) * 0.5) / windows.shape[1]

    return _rolling_full_window_transform(a, codes, window, rank_last)


def ts_zscore(a, window, codes):
    mean = ts_mean(a, window, codes)
    std = ts_std(a, window, codes).replace(0, np.nan)
    return (pd.Series(a) - mean) / std


def ts_iqr(a, window, codes):
    w = _window(window)
    series = pd.Series(a)
    rolling = series.groupby(codes, sort=False).rolling(w, min_periods=w)
    result = rolling.quantile(0.75) - rolling.quantile(0.25)
    return result.reset_index(level=0, drop=True).sort_index()


def ts_cv(a, window, codes):
    mean = ts_mean(a, window, codes).replace(0, np.nan)
    return ts_std(a, window, codes) / mean.abs()


def ts_skew(a, window, codes):
    return _rolling_grouped_stat(a, codes, window, "skew")


def ts_kurt(a, window, codes):
    return _rolling_unary(a, codes, window, lambda rolling: rolling.kurt())


def ts_autocorr(a, window, codes):
    w = _window(window)
    series, code_series = _series_for_codes(a, codes)
    if w < 2:
        return pd.Series(np.nan, index=series.index)
    frame = pd.DataFrame({"value": series, "code": code_series})
    frame["lag"] = frame.groupby("code", sort=False)["value"].shift(1)

    def apply_group(group):
        return group["value"].rolling(w - 1, min_periods=w - 1).corr(group["lag"])

    return frame.groupby("code", sort=False, group_keys=False)[["value", "lag"]].apply(apply_group).sort_index()


def ts_slope(a, window, codes):
    w = _window(window)
    if w < 2:
        return pd.Series(np.nan, index=pd.Series(a).index, dtype=float)
    x = np.arange(w, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))

    def slope(windows):
        centered = windows - np.nanmean(windows, axis=1)[:, None]
        return centered @ x / denom

    return _rolling_full_window_transform(a, codes, w, slope)


def decay_linear(a, window, codes):
    w = _window(window)
    weights = np.arange(1, w + 1, dtype=float)
    weights = weights / weights.sum()
    return _rolling_full_window_transform(a, codes, w, lambda windows: windows @ weights)


def decaylinear(a, window, codes):
    return decay_linear(a, window, codes)


def wma(a, window, codes):
    w = _window(window)
    weights = 0.9 ** np.arange(w - 1, -1, -1, dtype=float)
    weights = weights / weights.sum()
    return _rolling_full_window_transform(a, codes, w, lambda windows: windows @ weights)


def sma(a, window, modifier, codes):
    w = _window(window)
    alpha = min(max(float(modifier) / float(w), 1e-6), 1.0)
    series = pd.Series(a)
    return series.groupby(codes, group_keys=False).transform(
        lambda item: item.ewm(alpha=alpha, min_periods=w, adjust=False).mean()
    )


def ema(a, window, codes):
    w = _window(window)
    alpha = 2.0 / (w + 1.0)
    series = pd.Series(a)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for _, group in series.groupby(codes, sort=False):
        values = group.to_numpy(dtype=float, copy=False)
        result = np.full(len(group), np.nan, dtype=float)
        state = np.nan
        valid_count = 0
        for position, value in enumerate(values):
            if not np.isfinite(value):
                state = np.nan
                valid_count = 0
                continue
            state = value if valid_count == 0 else alpha * value + (1.0 - alpha) * state
            valid_count += 1
            if valid_count >= w:
                result[position] = state
        out.loc[group.index] = pd.Series(result, index=group.index).ffill()
    return out


def _series_for_codes(value, codes):
    code_series = pd.Series(codes)
    return _as_series(value, index=code_series.index), code_series


def _binary_frame(a, b, codes):
    left, code_series = _series_for_codes(a, codes)
    right = _as_series(b, index=code_series.index)
    return pd.DataFrame({"a": left, "b": right, "code": code_series})


def _rolling_binary_moments(a, b, codes, window):
    w = _window(window)
    frame = _binary_frame(a, b, codes).replace([np.inf, -np.inf], np.nan)
    outputs = [np.full(len(frame), np.nan, dtype=float) for _ in range(3)]

    def rolling_sum(values):
        cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
        return cumulative[w:] - cumulative[:-w]

    for positions in frame.groupby("code", sort=False).indices.values():
        if len(positions) < w:
            continue
        left = frame["a"].iloc[positions].to_numpy(dtype=float)
        right = frame["b"].iloc[positions].to_numpy(dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        if not valid.any():
            continue
        first = int(np.flatnonzero(valid)[0])
        left = np.where(valid, left - left[first], 0.0)
        right = np.where(valid, right - right[first], 0.0)
        valid_count = rolling_sum(valid.astype(float))
        sum_a = rolling_sum(left)
        sum_b = rolling_sum(right)
        cross = rolling_sum(left * right) - sum_a * sum_b / float(w)
        var_a = rolling_sum(left * left) - sum_a * sum_a / float(w)
        var_b = rolling_sum(right * right) - sum_b * sum_b / float(w)
        full = valid_count == float(w)
        target = np.asarray(positions)[w - 1 :]
        outputs[0][target] = np.where(full, cross, np.nan)
        outputs[1][target] = np.where(full, var_a, np.nan)
        outputs[2][target] = np.where(full, var_b, np.nan)

    return tuple(pd.Series(values, index=frame.index) for values in outputs)


def ts_corr(a, b, window, codes):
    cross, var_a, var_b = _rolling_binary_moments(a, b, codes, window)
    denom = np.sqrt(var_a * var_b).replace(0, np.nan)
    return cross / denom


def ts_cov(a, b, window, codes):
    w = _window(window)
    if w < 2:
        return pd.Series(np.nan, index=pd.Series(a).index, dtype=float)
    cross, _, _ = _rolling_binary_moments(a, b, codes, w)
    return cross / float(w - 1)


def ts_covariance(a, b, window, codes):
    return ts_cov(a, b, window, codes)


def ts_beta(a, b, window, codes):
    cross, _, var_b = _rolling_binary_moments(a, b, codes, window)
    return (cross / var_b).mask((~np.isfinite(var_b)) | (var_b.abs() < 1e-12))


def ts_residual(a, b, window, codes):
    beta = ts_beta(a, b, window, codes)
    return pd.Series(a) - beta * pd.Series(b)


def sequence(window, codes):
    code_series = pd.Series(codes)
    return code_series.groupby(code_series, sort=False).cumcount().astype(float) + 1.0


def regbeta(a, b, window, codes):
    return ts_beta(a, b, window, codes)


def regresi(a, b, window, codes):
    return ts_residual(a, b, window, codes)


def rsi(a, window, codes):
    change = delta(a, 1, codes)
    gains = pd.Series(change).clip(lower=0.0)
    losses = -pd.Series(change).clip(upper=0.0)
    avg_gain = ts_mean(gains, window, codes)
    avg_loss = ts_mean(losses, window, codes)
    relative = safe_div(avg_gain, avg_loss)
    return 100.0 - 100.0 / (1.0 + relative)


def macd(a, short_window, long_window, codes):
    return ema(a, short_window, codes) - ema(a, long_window, codes)


def bb_middle(a, window, codes):
    return ts_mean(a, window, codes)


def bb_upper(a, window, codes):
    return ts_mean(a, window, codes) + 2.0 * ts_std(a, window, codes)


def bb_lower(a, window, codes):
    return ts_mean(a, window, codes) - 2.0 * ts_std(a, window, codes)


def gt(a, b):
    return a > b


def ge(a, b):
    return a >= b


def lt(a, b):
    return a < b


def le(a, b):
    return a <= b


def eq(a, b):
    return a == b


def ne(a, b):
    return a != b


def logical_and(a, b):
    return np.logical_and(a, b)


def logical_or(a, b):
    return np.logical_or(a, b)


def where(condition, when_true, when_false):
    index = condition.index if isinstance(condition, pd.Series) else None
    mask = condition.fillna(False).astype(bool) if isinstance(condition, pd.Series) else condition
    return pd.Series(np.where(mask, when_true, when_false), index=index)


def filter(a, condition):
    return pd.Series(a).where(pd.Series(condition).astype(bool), 0.0)


def count(condition, window, codes):
    return ts_sum(pd.Series(condition).astype(float), window, codes)


def sum_if(a, window, condition, codes):
    values = pd.Series(a).where(pd.Series(condition).astype(bool), 0.0)
    return ts_sum(values, window, codes)


def sumif(a, window, condition, codes):
    return sum_if(a, window, condition, codes)


def _neutralize_frame(target, exposures, dates):
    series = pd.Series(target)
    frame = pd.DataFrame({"target": series, "date": dates})
    exposure_names = []
    for idx, exposure in enumerate(exposures):
        name = f"x{idx}"
        frame[name] = pd.Series(exposure)
        exposure_names.append(name)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for _, group in frame.replace([np.inf, -np.inf], np.nan).dropna().groupby("date", sort=False):
        if len(group) <= len(exposure_names) + 1:
            continue
        x = group[exposure_names].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(group)), x])
        y = group["target"].to_numpy(dtype=float)
        try:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ coef
        except Exception:
            continue
        out.loc[group.index] = resid
    return out


def neutralize_size(a, features, dates):
    exposure = features.get("fd_natural_log_of_market_cap")
    if exposure is None:
        exposure = features.get("fd_val_market_cap")
    if exposure is None:
        exposure = features.get("fd_cube_of_size")
    if exposure is None:
        return pd.Series(a)
    return _neutralize_frame(a, [exposure], dates)


def neutralize_styles(a, features, dates):
    exposures = [
        features[name]
        for name in ("fd_natural_log_of_market_cap", "fd_val_market_cap", "fd_cube_of_size")
        if name in features
    ]
    if not exposures:
        return pd.Series(a)
    return _neutralize_frame(a, exposures, dates)
"""

INTRADAY_AGGREGATION_SOURCE = r"""

def _spec_get(spec, key, default=None):
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _safe_group_ratio(numerator, denominator):
    if hasattr(denominator, "replace"):
        denominator = denominator.replace(0, np.nan)
        return numerator / denominator
    if not np.isfinite(denominator) or abs(float(denominator)) < 1e-12:
        return np.nan
    return numerator / denominator


def _grouped_frame_apply(grouped, func):
    try:
        return grouped.apply(func, include_groups=False)
    except TypeError:
        return grouped.apply(func)


def _aggregate_intraday_feature_series(work, grouped, spec):
    feature_name = _spec_get(spec, "name")
    field = _spec_get(spec, "field")
    aggregator = _spec_get(spec, "aggregator")
    if not feature_name or not field or not aggregator:
        return None

    if field not in work.columns and aggregator not in {"vwap", "realized_vol", "high_low_range", "close_vwap_gap"}:
        return None

    if aggregator == "first":
        series = grouped[field].first()
    elif aggregator == "last":
        series = grouped[field].last()
    elif aggregator == "max":
        series = grouped[field].max()
    elif aggregator == "min":
        series = grouped[field].min()
    elif aggregator == "median":
        series = grouped[field].median()
    elif aggregator == "mean":
        series = grouped[field].mean()
    elif aggregator == "std":
        series = grouped[field].std()
    elif aggregator == "sum":
        series = grouped[field].sum()
    elif aggregator == "range":
        series = grouped[field].max() - grouped[field].min()
    elif aggregator == "morning_mean":
        series = grouped[field].apply(lambda item: item.iloc[: max(1, len(item) // 2)].mean())
    elif aggregator == "afternoon_mean":
        series = grouped[field].apply(lambda item: item.iloc[len(item) // 2 :].mean())
    elif aggregator == "last_n_mean":
        series = grouped[field].apply(lambda item: item.tail(6).mean())
    elif aggregator == "vwap":
        if field in work.columns and "volume" in work.columns:
            series = _grouped_frame_apply(grouped, lambda item: _safe_group_ratio((item[field] * item["volume"]).sum(), item["volume"].sum()))
        elif "money" in work.columns and "volume" in work.columns:
            series = _grouped_frame_apply(grouped, lambda item: _safe_group_ratio(item["money"].sum(), item["volume"].sum()))
        else:
            return None
    elif aggregator == "realized_vol":
        if "close" not in work.columns:
            return None
        returns = grouped["close"].pct_change(fill_method=None)
        series = returns.groupby([work["trading_date"], work["code"]]).std()
    elif aggregator == "high_low_range":
        if not {"high", "low", "close"}.issubset(work.columns):
            return None
        high = grouped["high"].max()
        low = grouped["low"].min()
        close = grouped["close"].last().abs().replace(0, np.nan)
        series = (high - low) / close
    elif aggregator == "close_open_gap":
        if not {"open", "close"}.issubset(work.columns):
            return None
        open_ = grouped["open"].first().abs().replace(0, np.nan)
        series = (grouped["close"].last() - grouped["open"].first()) / open_
    elif aggregator == "close_vwap_gap":
        if "close" not in work.columns:
            return None
        close = grouped["close"].last()
        if field in work.columns and "volume" in work.columns:
            vwap = _grouped_frame_apply(grouped, lambda item: _safe_group_ratio((item[field] * item["volume"]).sum(), item["volume"].sum()))
        elif "money" in work.columns and "volume" in work.columns:
            vwap = _grouped_frame_apply(grouped, lambda item: _safe_group_ratio(item["money"].sum(), item["volume"].sum()))
        else:
            return None
        series = _safe_group_ratio(close - vwap, vwap.abs())
    elif aggregator in {"volume_imbalance", "money_imbalance"}:
        series = grouped[field].apply(
            lambda item: _safe_group_ratio(
                item.iloc[len(item) // 2 :].sum() - item.iloc[: len(item) // 2].sum(),
                item.abs().sum(),
            )
        )
    elif aggregator in {"volume_concentration", "money_concentration"}:
        series = grouped[field].apply(lambda item: _safe_group_ratio(item.tail(6).sum(), item.abs().sum()))
    else:
        return None

    return series.rename(feature_name)
"""

RUNTIME_NAMESPACE_SOURCE = r"""

def runtime_namespace(features, dates, codes):
    namespace = globals().copy()
    namespace.update(
        {
            "features": features,
            "dates": dates,
            "codes": codes,
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: safe_div(a, b),
            "neg": lambda a: neg(a),
            "abs": lambda a: a.abs() if isinstance(a, pd.Series) else abs(a),
            "log": lambda a: log(a),
            "sqrt": lambda a: sqrt(a),
            "sign": lambda a: sign(a),
            "exp": lambda a: exp(a),
            "floor": lambda a: floor(a),
            "inv": lambda a: inv(a),
            "element_max": lambda a, b: element_max(a, b),
            "element_min": lambda a, b: element_min(a, b),
            "power": lambda a, b: power(a, b),
            "pow": lambda a, b: pow(a, b),
            "clip": lambda a, limit: clip(a, limit),
            "winsorize": lambda a, limit: winsorize(a, limit),
            "cs_rank": lambda a: cs_rank(a, dates),
            "cs_zscore": lambda a: cs_zscore(a, dates),
            "cs_demean": lambda a: cs_demean(a, dates),
            "cs_scale": lambda a: cs_scale(a, dates),
            "cs_quantile": lambda a: cs_quantile(a, dates),
            "cs_mean": lambda a: cs_mean(a, dates),
            "cs_std": lambda a: cs_std(a, dates),
            "cs_skew": lambda a: cs_skew(a, dates),
            "cs_kurt": lambda a: cs_kurt(a, dates),
            "cs_max": lambda a: cs_max(a, dates),
            "cs_min": lambda a: cs_min(a, dates),
            "cs_median": lambda a: cs_median(a, dates),
            "cs_scale_to": lambda a, target: cs_scale_to(a, target, dates),
            "delay": lambda a, w: delay(a, w, codes),
            "delta": lambda a, w: delta(a, w, codes),
            "pct_change": lambda a, w: pct_change(a, w, codes),
            "ts_pctchange": lambda a, w: ts_pctchange(a, w, codes),
            "ts_mean": lambda a, w: ts_mean(a, w, codes),
            "ts_median": lambda a, w: ts_median(a, w, codes),
            "ts_sum": lambda a, w: ts_sum(a, w, codes),
            "ts_cumsum": lambda a, w: ts_cumsum(a, w, codes),
            "ts_product": lambda a, w: ts_product(a, w, codes),
            "prod": lambda a, w: prod(a, w, codes),
            "ts_std": lambda a, w: ts_std(a, w, codes),
            "ts_var": lambda a, w: ts_var(a, w, codes),
            "ts_quantile": lambda a, w, q: ts_quantile(a, w, q, codes),
            "percentile": lambda a, q, w: percentile(a, q, w, codes),
            "ts_mad": lambda a, w: ts_mad(a, w, codes),
            "ts_min": lambda a, w: ts_min(a, w, codes),
            "ts_max": lambda a, w: ts_max(a, w, codes),
            "ts_argmin": lambda a, w: ts_argmin(a, w, codes),
            "ts_argmax": lambda a, w: ts_argmax(a, w, codes),
            "highday": lambda a, w: highday(a, w, codes),
            "lowday": lambda a, w: lowday(a, w, codes),
            "ts_rank": lambda a, w: ts_rank(a, w, codes),
            "ts_zscore": lambda a, w: ts_zscore(a, w, codes),
            "ts_iqr": lambda a, w: ts_iqr(a, w, codes),
            "ts_cv": lambda a, w: ts_cv(a, w, codes),
            "ts_skew": lambda a, w: ts_skew(a, w, codes),
            "ts_kurt": lambda a, w: ts_kurt(a, w, codes),
            "ts_autocorr": lambda a, w: ts_autocorr(a, w, codes),
            "ts_slope": lambda a, w: ts_slope(a, w, codes),
            "decay_linear": lambda a, w: decay_linear(a, w, codes),
            "decaylinear": lambda a, w: decaylinear(a, w, codes),
            "ema": lambda a, w: ema(a, w, codes),
            "wma": lambda a, w: wma(a, w, codes),
            "sma": lambda a, w, m: sma(a, w, m, codes),
            "ts_corr": lambda a, b, w: ts_corr(a, b, w, codes),
            "ts_cov": lambda a, b, w: ts_cov(a, b, w, codes),
            "ts_covariance": lambda a, b, w: ts_covariance(a, b, w, codes),
            "ts_beta": lambda a, b, w: ts_beta(a, b, w, codes),
            "ts_residual": lambda a, b, w: ts_residual(a, b, w, codes),
            "sequence": lambda w: sequence(w, codes),
            "regbeta": lambda a, b, w: regbeta(a, b, w, codes),
            "regresi": lambda a, b, w: regresi(a, b, w, codes),
            "rsi": lambda a, w: rsi(a, w, codes),
            "macd": lambda a, short, long: macd(a, short, long, codes),
            "bb_middle": lambda a, w: bb_middle(a, w, codes),
            "bb_upper": lambda a, w: bb_upper(a, w, codes),
            "bb_lower": lambda a, w: bb_lower(a, w, codes),
            "gt": lambda a, b: gt(a, b),
            "ge": lambda a, b: ge(a, b),
            "lt": lambda a, b: lt(a, b),
            "le": lambda a, b: le(a, b),
            "eq": lambda a, b: eq(a, b),
            "ne": lambda a, b: ne(a, b),
            "logical_and": lambda a, b: logical_and(a, b),
            "logical_or": lambda a, b: logical_or(a, b),
            "where": lambda c, a, b: where(c, a, b),
            "filter": lambda a, c: filter(a, c),
            "count": lambda c, w: count(c, w, codes),
            "sum_if": lambda a, w, c: sum_if(a, w, c, codes),
            "sumif": lambda a, w, c: sumif(a, w, c, codes),
            "neutralize_size": lambda a: neutralize_size(a, features, dates),
            "neutralize_styles": lambda a: neutralize_styles(a, features, dates),
            "__builtins__": {},
        }
    )
    return namespace
"""

_RUNTIME_OPERATOR_BINDINGS = {
    "add": "lambda a, b: a + b",
    "sub": "lambda a, b: a - b",
    "mul": "lambda a, b: a * b",
    "div": "lambda a, b: safe_div(a, b)",
    "neg": "lambda a: neg(a)",
    "abs": "lambda a: a.abs() if isinstance(a, pd.Series) else abs(a)",
    "log": "lambda a: log(a)",
    "sqrt": "lambda a: sqrt(a)",
    "sign": "lambda a: sign(a)",
    "exp": "lambda a: exp(a)",
    "floor": "lambda a: floor(a)",
    "inv": "lambda a: inv(a)",
    "element_max": "lambda a, b: element_max(a, b)",
    "element_min": "lambda a, b: element_min(a, b)",
    "power": "lambda a, b: power(a, b)",
    "pow": "lambda a, b: pow(a, b)",
    "clip": "lambda a, limit: clip(a, limit)",
    "winsorize": "lambda a, limit: winsorize(a, limit)",
    "cs_rank": "lambda a: cs_rank(a, dates)",
    "cs_zscore": "lambda a: cs_zscore(a, dates)",
    "cs_demean": "lambda a: cs_demean(a, dates)",
    "cs_scale": "lambda a: cs_scale(a, dates)",
    "cs_quantile": "lambda a: cs_quantile(a, dates)",
    "cs_mean": "lambda a: cs_mean(a, dates)",
    "cs_std": "lambda a: cs_std(a, dates)",
    "cs_skew": "lambda a: cs_skew(a, dates)",
    "cs_kurt": "lambda a: cs_kurt(a, dates)",
    "cs_max": "lambda a: cs_max(a, dates)",
    "cs_min": "lambda a: cs_min(a, dates)",
    "cs_median": "lambda a: cs_median(a, dates)",
    "cs_scale_to": "lambda a, target: cs_scale_to(a, target, dates)",
    "delay": "lambda a, w: delay(a, w, codes)",
    "delta": "lambda a, w: delta(a, w, codes)",
    "pct_change": "lambda a, w: pct_change(a, w, codes)",
    "ts_pctchange": "lambda a, w: ts_pctchange(a, w, codes)",
    "ts_mean": "lambda a, w: ts_mean(a, w, codes)",
    "ts_median": "lambda a, w: ts_median(a, w, codes)",
    "ts_sum": "lambda a, w: ts_sum(a, w, codes)",
    "ts_cumsum": "lambda a, w: ts_cumsum(a, w, codes)",
    "ts_product": "lambda a, w: ts_product(a, w, codes)",
    "prod": "lambda a, w: prod(a, w, codes)",
    "ts_std": "lambda a, w: ts_std(a, w, codes)",
    "ts_var": "lambda a, w: ts_var(a, w, codes)",
    "ts_quantile": "lambda a, w, q: ts_quantile(a, w, q, codes)",
    "percentile": "lambda a, q, w: percentile(a, q, w, codes)",
    "ts_mad": "lambda a, w: ts_mad(a, w, codes)",
    "ts_min": "lambda a, w: ts_min(a, w, codes)",
    "ts_max": "lambda a, w: ts_max(a, w, codes)",
    "ts_argmin": "lambda a, w: ts_argmin(a, w, codes)",
    "ts_argmax": "lambda a, w: ts_argmax(a, w, codes)",
    "highday": "lambda a, w: highday(a, w, codes)",
    "lowday": "lambda a, w: lowday(a, w, codes)",
    "ts_rank": "lambda a, w: ts_rank(a, w, codes)",
    "ts_zscore": "lambda a, w: ts_zscore(a, w, codes)",
    "ts_iqr": "lambda a, w: ts_iqr(a, w, codes)",
    "ts_cv": "lambda a, w: ts_cv(a, w, codes)",
    "ts_skew": "lambda a, w: ts_skew(a, w, codes)",
    "ts_kurt": "lambda a, w: ts_kurt(a, w, codes)",
    "ts_autocorr": "lambda a, w: ts_autocorr(a, w, codes)",
    "ts_slope": "lambda a, w: ts_slope(a, w, codes)",
    "decay_linear": "lambda a, w: decay_linear(a, w, codes)",
    "decaylinear": "lambda a, w: decaylinear(a, w, codes)",
    "ema": "lambda a, w: ema(a, w, codes)",
    "wma": "lambda a, w: wma(a, w, codes)",
    "sma": "lambda a, w, m: sma(a, w, m, codes)",
    "ts_corr": "lambda a, b, w: ts_corr(a, b, w, codes)",
    "ts_cov": "lambda a, b, w: ts_cov(a, b, w, codes)",
    "ts_covariance": "lambda a, b, w: ts_covariance(a, b, w, codes)",
    "ts_beta": "lambda a, b, w: ts_beta(a, b, w, codes)",
    "ts_residual": "lambda a, b, w: ts_residual(a, b, w, codes)",
    "sequence": "lambda w: sequence(w, codes)",
    "regbeta": "lambda a, b, w: regbeta(a, b, w, codes)",
    "regresi": "lambda a, b, w: regresi(a, b, w, codes)",
    "rsi": "lambda a, w: rsi(a, w, codes)",
    "macd": "lambda a, short, long: macd(a, short, long, codes)",
    "bb_middle": "lambda a, w: bb_middle(a, w, codes)",
    "bb_upper": "lambda a, w: bb_upper(a, w, codes)",
    "bb_lower": "lambda a, w: bb_lower(a, w, codes)",
    "gt": "lambda a, b: gt(a, b)",
    "ge": "lambda a, b: ge(a, b)",
    "lt": "lambda a, b: lt(a, b)",
    "le": "lambda a, b: le(a, b)",
    "eq": "lambda a, b: eq(a, b)",
    "ne": "lambda a, b: ne(a, b)",
    "logical_and": "lambda a, b: logical_and(a, b)",
    "logical_or": "lambda a, b: logical_or(a, b)",
    "where": "lambda c, a, b: where(c, a, b)",
    "filter": "lambda a, c: filter(a, c)",
    "count": "lambda c, w: count(c, w, codes)",
    "sum_if": "lambda a, w, c: sum_if(a, w, c, codes)",
    "sumif": "lambda a, w, c: sumif(a, w, c, codes)",
    "neutralize_size": "lambda a: neutralize_size(a, features, dates)",
    "neutralize_styles": "lambda a: neutralize_styles(a, features, dates)",
}


def runtime_source_for_operators(operator_names: set[str]) -> str:
    """Return the runtime helper dependency closure for one expression."""
    unknown = set(operator_names) - set(_RUNTIME_OPERATOR_BINDINGS)
    if unknown:
        raise ValueError(f"Unknown symbolic runtime operators: {sorted(unknown)}")

    tree = ast.parse(RUNTIME_SOURCE)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    required = set()
    for operator in operator_names:
        binding = ast.parse(_RUNTIME_OPERATOR_BINDINGS[operator], mode="eval")
        required.update(
            node.id
            for node in ast.walk(binding)
            if isinstance(node, ast.Name) and node.id in functions
        )
    pending = list(required)
    while pending:
        function_name = pending.pop()
        for node in ast.walk(functions[function_name]):
            called_name = (
                node.func.id
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                else None
            )
            if called_name in functions and called_name not in required:
                required.add(called_name)
                pending.append(called_name)

    lines = RUNTIME_SOURCE.splitlines()
    blocks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in required:
            blocks.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n\n".join(blocks)


def runtime_namespace_source_for_operators(operator_names: set[str]) -> str:
    """Return a namespace factory containing only operators used by one expression."""
    unknown = set(operator_names) - set(_RUNTIME_OPERATOR_BINDINGS)
    if unknown:
        raise ValueError(f"Unknown symbolic runtime operators: {sorted(unknown)}")
    bindings = "\n".join(
        f"            {name!r}: {_RUNTIME_OPERATOR_BINDINGS[name]},"
        for name in sorted(operator_names)
    )
    return f"""def runtime_namespace(features, dates, codes):
    namespace = globals().copy()
    namespace.update(
        {{
            "features": features,
            "dates": dates,
            "codes": codes,
{bindings}
            "__builtins__": {{}},
        }}
    )
    return namespace"""


_RUNTIME_GLOBALS: dict = {}
exec(RUNTIME_SOURCE, _RUNTIME_GLOBALS)
exec(INTRADAY_AGGREGATION_SOURCE, _RUNTIME_GLOBALS)
exec(RUNTIME_NAMESPACE_SOURCE, _RUNTIME_GLOBALS)

aggregate_intraday_feature_series = _RUNTIME_GLOBALS[
    "_aggregate_intraday_feature_series"
]


def runtime_namespace(features: dict, dates, codes) -> dict:
    return _RUNTIME_GLOBALS["runtime_namespace"](features, dates, codes)
