"""Local factor-scaled price adjustment helpers (excluded from JoinQuant cloud bundles)."""

import pandas as pd

from utils.pit import (
    ADJUSTABLE_PRICE_FIELDS,
    PRICE_ADJUSTED_ATTR,
    PRICE_ADJUSTMENT_MODE_ATTR,
    _prepare_factor_frame,
    apply_pit_adjustment,
)


def adjust_prices_by_factor(
    df: pd.DataFrame,
    factor_column: str = "factor",
    *,
    require_factor: bool = False,
) -> pd.DataFrame:
    """Scale raw prices by ``price * factor`` (and ``volume / factor``) without future anchors.

    Idempotent when ``df.attrs['price_adjusted']`` is already True so pre-adjusted
    inputs are never multiplied by ``factor`` twice.
    """
    if df.empty or df.attrs.get(PRICE_ADJUSTED_ATTR):
        return df
    has_adjustable = any(field in df.columns for field in (*ADJUSTABLE_PRICE_FIELDS, "volume"))
    if factor_column not in df.columns:
        if require_factor and has_adjustable:
            raise ValueError(f"pit_factor_column_missing: {factor_column!r}")
        return df
    if not has_adjustable:
        return df
    result, _, _ = _prepare_factor_frame(df, factor_column)
    factors = result[factor_column]
    for field in ADJUSTABLE_PRICE_FIELDS:
        if field in result.columns:
            result[field] = result[field] * factors
    if "volume" in result.columns:
        result["volume"] = result["volume"] / factors
    result.attrs[PRICE_ADJUSTED_ATTR] = True
    result.attrs[PRICE_ADJUSTMENT_MODE_ATTR] = "factor_multiplied"
    return result


def adjust_execution_grid_prices(
    grid: pd.DataFrame,
    *,
    price_type: str,
    pit_enabled: bool,
    factor_column: str = "factor",
    ashare_panel: bool = True,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Adjust an A-share Bar grid once when ``factor_column`` is present and unapplied."""
    adjustable = price_type in (*ADJUSTABLE_PRICE_FIELDS, "volume")
    if (
        not adjusted
        or not adjustable
        or not ashare_panel
        or grid.attrs.get(PRICE_ADJUSTED_ATTR)
        or factor_column not in grid.columns
    ):
        return grid
    if pit_enabled:
        adjusted_grid = apply_pit_adjustment(
            grid.rename(columns={"instrument_id": "code", "timestamp_utc": "date"}),
            factor_column,
        )
        return adjusted_grid.rename(columns={"code": "instrument_id", "date": "timestamp_utc"})
    return adjust_prices_by_factor(grid, factor_column=factor_column, require_factor=False)
