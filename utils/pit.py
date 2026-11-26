"""Point-in-time price adjustment shared by both execution environments."""

from numbers import Real
from typing import Optional, Tuple
import pandas as pd
import numpy as np


MONEY_NORMALIZATION_VERSION = "source-money-v3"
PRICE_ADJUSTED_ATTR = "price_adjusted"
PRICE_ADJUSTMENT_MODE_ATTR = "price_adjustment_mode"
ADJUSTABLE_PRICE_FIELDS = (
    "open", "high", "low", "close", "pre_close", "avg", "high_limit", "low_limit",
)


def _observed_mask(df: pd.DataFrame) -> Tuple[pd.Series, bool]:
    money_only = "money" in df and not any(
        field in df for field in (*ADJUSTABLE_PRICE_FIELDS, "volume")
    )
    price_or_volume = [
        f
        for f in ("open", "high", "low", "close", "volume", "pre_close", "avg")
        if f in df
    ]
    if money_only:
        observed = df["money"].notna()
    else:
        observed = (
            df[price_or_volume].notna().any(axis=1)
            if price_or_volume
            else pd.Series(True, index=df.index)
        )
    if "money" in df:
        observed = observed | (df["money"].notna() & df["money"].ne(0))
    return observed, money_only


def _coerce_positive_factors(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype):
        return pd.Series(np.nan, index=series.index, dtype=float), pd.Series(True, index=series.index)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = series.astype(float)
        return numeric, ~np.isfinite(numeric) | (numeric <= 0)
    valid_type = series.map(lambda v: isinstance(v, Real) and not isinstance(v, (bool, np.bool_)))
    numeric = pd.to_numeric(series.where(valid_type), errors="coerce").astype(float)
    return numeric, ~valid_type | ~np.isfinite(numeric) | (numeric <= 0)


def _prepare_factor_frame(df: pd.DataFrame, factor_column: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    result = df.copy()
    numeric_factors, invalid = _coerce_positive_factors(result[factor_column])
    money_only = "money" in result and not any(
        field in result for field in (*ADJUSTABLE_PRICE_FIELDS, "volume")
    )
    if "money" in result:
        # Infer an empty amount only from invalid factor + zero, regardless of projection.
        empty_amount = invalid & result["money"].eq(0)
        if money_only or empty_amount.any():
            result.loc[empty_amount, "money"] = np.nan
    observed, _ = _observed_mask(result)
    if (invalid & observed).any():
        raise ValueError("pit_factor_missing_or_invalid")
    # Empty bars cannot supply an anchor or adjustment scale, even if storage filled factor=1.
    result[factor_column] = numeric_factors.where(~(invalid | ~observed), np.nan)
    return result, invalid, observed


def normalize_price_money(df: pd.DataFrame, *, source: str, frequency: str) -> pd.DataFrame:
    """Normalize local storage placeholders to JoinQuant fill_paused=False semantics.

    A daily paused flag is authoritative. Minute zero-volume bars are not
    suspension evidence; only an explicitly supplied paused flag may erase zero.
    Native JoinQuant NaNs and genuine unpaused zero amounts pass through.
    """
    if source not in {"local", "joinquant"} or frequency not in {"daily", "5m", "1m"}:
        raise ValueError("Unsupported money normalization source or frequency")
    if df.empty or "money" not in df or "paused" not in df:
        return df
    empty = df["paused"].eq(1) & df["money"].eq(0)
    if not empty.any():
        return df
    result = df.copy()
    result.loc[empty, "money"] = np.nan
    return result


def apply_pit_adjustment(
    df: pd.DataFrame,
    factor_column: str = "factor",
    factor_end_overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """Adjust price/volume fields; nonempty adjustable inputs require a factor column.

    Empty frames, already-adjusted frames, and factor-free inputs without adjustable
    fields pass through. Money alone is not rescaled. Existing factor columns retain
    value validation.
    """
    price_fields = ADJUSTABLE_PRICE_FIELDS
    if df.empty or getattr(df, "attrs", {}).get(PRICE_ADJUSTED_ATTR):
        return df
    if factor_column not in df.columns:
        if any(field in df.columns for field in (*price_fields, "volume")):
            raise ValueError(f"pit_factor_column_missing: {factor_column!r}")
        return df
    stock_column = "code" if "code" in df.columns else "stock_code"
    result, _, _ = _prepare_factor_frame(df, factor_column)
    if not getattr(result, "attrs", {}).get("_sorted_by_code_date"):
        result = result.sort_values(
            [stock_column, "date"], kind="mergesort"
        ).reset_index(drop=True)
        if hasattr(result, "attrs"):
            result.attrs["_sorted_by_code_date"] = True
    factor_end = result.groupby(stock_column)[factor_column].transform("last")
    if factor_end_overrides:
        if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or v <= 0
               for v in factor_end_overrides.values()):
            raise ValueError("pit_anchor_missing_or_invalid")
        factor_end = result[stock_column].map(factor_end_overrides).fillna(factor_end)
    ratio = result[factor_column] / factor_end
    for field in price_fields:
        if field in result.columns:
            result[field] = result[field] * ratio
    if "volume" in result.columns:
        result["volume"] = result["volume"] / ratio
    if hasattr(result, "attrs"):
        result.attrs[PRICE_ADJUSTED_ATTR] = True
        result.attrs[PRICE_ADJUSTMENT_MODE_ATTR] = "pit_factor_end"
    return result
