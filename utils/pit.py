"""Point-in-time price adjustment shared by both execution environments."""

from typing import Optional
import pandas as pd
import numpy as np


MONEY_NORMALIZATION_VERSION = "source-money-v3"


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

    Empty frames and factor-free inputs without adjustable fields pass through.
    Money alone is not rescaled. Existing factor columns retain value validation.
    """
    price_fields = (
        "open", "high", "low", "close", "pre_close", "avg", "high_limit", "low_limit",
    )
    if df.empty:
        return df
    if factor_column not in df.columns:
        if any(field in df.columns for field in (*price_fields, "volume")):
            raise ValueError(f"pit_factor_column_missing: {factor_column!r}")
        return df
    stock_column = "code" if "code" in df.columns else "stock_code"
    result = df.copy()
    invalid = ~np.isfinite(result[factor_column]) | (result[factor_column] <= 0)
    money_only = "money" in result and not any(
        field in result for field in (*price_fields, "volume")
    )
    if "money" in result:
        # Infer an empty amount only from invalid factor + zero, regardless of projection.
        empty_amount = invalid & result["money"].eq(0)
        if money_only or empty_amount.any():
            result.loc[empty_amount, "money"] = np.nan
    price_or_volume = [
        f
        for f in ("open", "high", "low", "close", "volume", "pre_close", "avg")
        if f in result
    ]
    if money_only:
        observed = result["money"].notna()
    else:
        observed = (
            result[price_or_volume].notna().any(axis=1)
            if price_or_volume
            else pd.Series(True, index=result.index)
        )
    if "money" in result:
        observed |= result["money"].notna() & result["money"].ne(0)
    if (invalid & observed).any():
        raise ValueError("pit_factor_missing_or_invalid")
    # Empty bars cannot supply an anchor, even if storage filled their factor with 1.
    result.loc[invalid | ~observed, factor_column] = np.nan
    if not getattr(result, "attrs", {}).get("_sorted_by_code_date"):
        result = result.sort_values(
            [stock_column, "date"], kind="mergesort"
        ).reset_index(drop=True)
        if hasattr(result, "attrs"):
            result.attrs["_sorted_by_code_date"] = True
    factor_end = result.groupby(stock_column)[factor_column].transform("last")
    if factor_end_overrides:
        if any(not np.isfinite(v) or v <= 0 for v in factor_end_overrides.values()):
            raise ValueError("pit_anchor_missing_or_invalid")
        factor_end = result[stock_column].map(factor_end_overrides).fillna(factor_end)
    ratio = result[factor_column] / factor_end
    for field in price_fields:
        if field in result.columns:
            result[field] = result[field] * ratio
    if "volume" in result.columns:
        result["volume"] = result["volume"] / ratio
    return result
