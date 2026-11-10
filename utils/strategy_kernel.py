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
        return pd.Series(0.0, index=series.index) if zero_degenerate else values - mean
    return (values - mean) / std


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
    return values.reindex(columns=w.index).fillna(0.0).mul(w, axis=1).sum(axis=1)


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


def linear_signal(raw, model, parent_scores=None):
    """Frozen-model replay, using the local VectorBT cross-sectional semantics."""
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
    processed = pd.DataFrame(index=raw.index)
    for name in sorted(needed):
        method = methods.get(name) or methods.get("ALL", "zscore") or "zscore"
        if not model.get("standardize", True):
            values = raw[name]
        elif method == "binning":
            values = bin_series(raw[name], methods.get("_binning_n_bins", 5))
        else:
            # VectorBT's legacy `none` spelling means its default transform.
            method = "zscore" if method == "none" else method
            percentile = (
                methods.get("_winsorize_percentile", 0.025)
                if method == "winsorize_zscore"
                else 0.025
            )
            values = preprocess_series(raw[name], method, percentile)
        processed[name] = values.fillna(0.0)
    score = (
        weighted_score(processed, weights)
        if weights
        else pd.Series(0.0, index=raw.index)
    )
    if weights and model.get("postprocess_cs_standardize", True) and len(score) > 1:
        score = (score - score.mean()) / (score.std() + 1e-8)
    if ratio > 0 and parent_scores is not None:
        parent = parent_scores.reindex(score.index).dropna()
        if parent.empty:
            raise ValueError("parent baseline has no residual-score overlap")
        score = score.loc[parent.index] * (1 - ratio) + parent * ratio
    elif ratio > 0 and blend:
        score = score * (1 - ratio) + weighted_score(processed, blend) * ratio
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
