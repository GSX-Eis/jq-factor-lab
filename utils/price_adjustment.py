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


def _no_adjustment_instruments(grid: pd.DataFrame) -> set[str]:
    """Return instrument IDs whose metadata explicitly declares ``price_adjustment='none'``."""
    metadata = grid.attrs.get("instrument_metadata")
    if metadata is None:
        return set()
    records = metadata.to_dict("records") if isinstance(metadata, pd.DataFrame) else metadata
    return {
        record["instrument_id"]
        for record in records
        if isinstance(record, dict) and record.get("price_adjustment") == "none" and "instrument_id" in record
    }


def _apply_grid_adjustment(
    frame: pd.DataFrame,
    *,
    pit_enabled: bool,
    factor_column: str,
) -> pd.DataFrame:
    if pit_enabled:
        sorted_index = (
            frame.sort_values(["instrument_id", "timestamp_utc"], kind="mergesort").index
            if {"instrument_id", "timestamp_utc"}.issubset(frame.columns)
            else frame.index
        )
        adjusted_grid = apply_pit_adjustment(
            frame.rename(columns={"instrument_id": "code", "timestamp_utc": "date"}),
            factor_column,
        )
        adjusted_grid = adjusted_grid.rename(columns={"code": "instrument_id", "date": "timestamp_utc"})
        adjusted_grid.index = sorted_index
        return adjusted_grid
    return adjust_prices_by_factor(frame, factor_column=factor_column, require_factor=False)


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
        or grid.empty
        or grid.attrs.get(PRICE_ADJUSTED_ATTR)
        or factor_column not in grid.columns
    ):
        return grid
    no_adjust_ids = _no_adjustment_instruments(grid)
    if not no_adjust_ids or "instrument_id" not in grid.columns:
        return _apply_grid_adjustment(grid, pit_enabled=pit_enabled, factor_column=factor_column)
    adjust_mask = ~grid["instrument_id"].isin(no_adjust_ids)
    if not adjust_mask.any():
        return grid
    if adjust_mask.all():
        return _apply_grid_adjustment(grid, pit_enabled=pit_enabled, factor_column=factor_column)
    sub = grid.loc[adjust_mask].copy()
    sub.attrs = {}
    adjusted_part = _apply_grid_adjustment(sub, pit_enabled=pit_enabled, factor_column=factor_column)
    result = grid.copy()
    adjusted_cols = [col for col in (*ADJUSTABLE_PRICE_FIELDS, "volume", factor_column) if col in result.columns]
    for col in adjusted_cols:
        if result[col].dtype != adjusted_part[col].dtype:
            result[col] = result[col].astype(adjusted_part[col].dtype)
    result.loc[adjusted_part.index, adjusted_cols] = adjusted_part[adjusted_cols]
    result.attrs[PRICE_ADJUSTED_ATTR] = bool(adjusted_part.attrs.get(PRICE_ADJUSTED_ATTR, False))
    if PRICE_ADJUSTMENT_MODE_ATTR in adjusted_part.attrs:
        result.attrs[PRICE_ADJUSTMENT_MODE_ATTR] = adjusted_part.attrs[PRICE_ADJUSTMENT_MODE_ATTR]
    return result


def collect_adjustment_attrs(parts: list[pd.DataFrame]) -> dict:
    """Propagate idempotent adjustment metadata only when all non-empty parts agree."""
    nonempty = [part for part in parts if not part.empty]
    if not nonempty or not all(bool(part.attrs.get(PRICE_ADJUSTED_ATTR, False)) for part in nonempty):
        return {}
    attrs: dict = {PRICE_ADJUSTED_ATTR: True}
    modes = {
        part.attrs.get(PRICE_ADJUSTMENT_MODE_ATTR)
        for part in nonempty
        if PRICE_ADJUSTMENT_MODE_ATTR in part.attrs
    }
    if len(modes) == 1:
        attrs[PRICE_ADJUSTMENT_MODE_ATTR] = next(iter(modes))
    return attrs
