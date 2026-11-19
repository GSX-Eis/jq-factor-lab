"""Explicit execution conventions for daily closes and exchange-scheduled Bars."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from utils.continuous_bars import ContinuousBarSource
    from utils.session_calendar import SessionCalendar, View
    from utils.trading_calendar import TradingCalendar


logger = logging.getLogger(__name__)
_Prices = TypeVar("_Prices", pd.DataFrame, np.ndarray)


@dataclass(frozen=True)
class ExecutionRules:
    entry_price_field: Literal["open", "close"]
    exit_price_field: Literal["close"]
    min_entry_offset: int
    exit_adjustment: int
    unit: Literal["decimal", "percent"]
    gap_rule: Literal["endpoints_only", "require_complete_bars"]

    def exit_offset(self, horizon: int) -> int:
        """Number of grid slots from the entry slot to the exit slot."""
        return horizon + self.exit_adjustment

    def returns(self, entry: _Prices, exit: _Prices) -> _Prices:
        result = exit / entry - 1
        return result * 100 if self.unit == "percent" else result


class ExecutionConvention(Enum):
    CLOSE_TO_CLOSE = ExecutionRules(
        entry_price_field="close", exit_price_field="close", min_entry_offset=0,
        exit_adjustment=0, unit="decimal", gap_rule="endpoints_only",
    )
    OPEN_TO_CLOSE = ExecutionRules(
        entry_price_field="open", exit_price_field="close", min_entry_offset=1,
        exit_adjustment=-1, unit="percent", gap_rule="require_complete_bars",
    )


_DAILY = ExecutionConvention.CLOSE_TO_CLOSE.value


class ForwardReturnsValidationError(ValueError):
    """Invalid prices or absent labels must not become successful empty analysis."""


@dataclass(frozen=True)
class DailyReturnLabels:
    """Signal-keyed returns and on-demand executed endpoints for each horizon."""

    returns: pd.DataFrame
    _closes: pd.DataFrame = field(repr=False)
    _entry_offset: int = field(repr=False)
    _calendar: TradingCalendar = field(repr=False)

    def scheduled_exit_dates(self, horizon: int) -> pd.Series:
        """Map signal dates to the same scheduled trade dates used by the labeler."""
        if f"{horizon}D" not in self.returns:
            raise ValueError(f"Unknown daily horizon: {horizon!r}")
        sessions = self._calendar.sessions()
        offsets = self._entry_offset + _DAILY.exit_offset(int(horizon))
        return pd.Series(sessions, index=sessions, name="exit_date").shift(-offsets)

    def endpoints(self, horizon: int) -> pd.DataFrame:
        """Trade dates and prices indexed by the signal keys with a valid return."""
        index = self.returns[f"{horizon}D"].dropna().index
        dates = self._closes.index
        signal_rows = dates.get_indexer(index.get_level_values("date"))
        columns = self._closes.columns.get_indexer(index.get_level_values("asset"))
        entry_rows = signal_rows + self._entry_offset
        exit_rows = entry_rows + _DAILY.exit_offset(int(horizon))
        prices = self._closes.to_numpy(copy=False)
        return pd.DataFrame({
            "entry_date": dates.take(entry_rows),
            "exit_date": dates.take(exit_rows),
            "entry_price": prices[entry_rows, columns],
            "exit_price": prices[exit_rows, columns],
        }, index=index)


def label_forward_returns(
    pricing: pd.Series | pd.DataFrame,
    *,
    convention: ExecutionConvention,
    calendar: TradingCalendar | SessionCalendar | Mapping[str, SessionCalendar] | None = None,
    entry_offset: int,
    horizons: Iterable[int] | int,
    factors: pd.DataFrame | None = None,
    signal_timestamps: Sequence | None = None,
    view: View | None = None,
    boundary_policy: str | None = None,
    bucket_strategy: str | None = None,
    continuous_source: ContinuousBarSource | None = None,
    include_scheduled_exit: bool = False,
) -> DailyReturnLabels | pd.DataFrame:
    """Dispatch by trade convention after materializing a scalar or iterable horizon input."""
    if not isinstance(convention, ExecutionConvention):
        raise ValueError("convention must be an ExecutionConvention")
    if isinstance(horizons, Integral):
        periods = (horizons,)
    else:
        if isinstance(horizons, (str, bytes)):
            raise ValueError("horizons must be an integer or an iterable of integers")
        try:
            periods = tuple(horizons)
        except TypeError as exc:
            raise ValueError("horizons must be an integer or an iterable of integers") from exc

    if convention is ExecutionConvention.CLOSE_TO_CLOSE:
        if any(option is not None for option in (
            factors, signal_timestamps, view, boundary_policy, bucket_strategy, continuous_source,
        )) or include_scheduled_exit:
            raise ValueError("Bar label options require the open_to_close convention")
        return label_daily_returns(pricing, calendar, entry_offset, periods)

    from backtest_and_analysis.time_series.labels import BarReturnLabeler

    labeler = BarReturnLabeler(
        calendar, entry_offset_bars=entry_offset, horizons=periods,
        view="ETH" if view is None else view,
        boundary_policy="same_session" if boundary_policy is None else boundary_policy,
        bucket_strategy=bucket_strategy, continuous_source=continuous_source,
        include_scheduled_exit=include_scheduled_exit,
    )
    return labeler.label(pricing, factors, signal_timestamps=signal_timestamps)


def resolve_daily_calendar(pricing: pd.Series, calendar: TradingCalendar | None) -> TradingCalendar:
    from utils.trading_calendar import load_trading_calendar

    calendar = calendar or pricing.attrs.get("trading_calendar")
    return calendar if calendar is not None else load_trading_calendar()


def validate_daily_horizons(
    periods: Iterable[int], execution_lag: int
) -> tuple[int, ...]:
    """Validate discrete trading-row horizons before applying ``shift``."""
    if isinstance(periods, (str, bytes)):
        raise ValueError("periods must be a non-empty iterable of positive integers")
    try:
        normalized_periods = tuple(periods)
    except TypeError as exc:
        raise ValueError("periods must be a non-empty iterable of positive integers") from exc
    if not normalized_periods or any(
        isinstance(period, bool) or not isinstance(period, (int, np.integer)) or period <= 0
        for period in normalized_periods
    ):
        raise ValueError("periods must be a non-empty iterable of positive integers")
    if (
        isinstance(execution_lag, bool)
        or not isinstance(execution_lag, (int, np.integer))
        or execution_lag < _DAILY.min_entry_offset
    ):
        raise ValueError("execution_lag must be a non-negative integer")
    return tuple(int(period) for period in normalized_periods)


def _daily_close_panel(pricing: pd.Series) -> pd.DataFrame:
    if not isinstance(pricing, pd.Series):
        raise ValueError("价格数据必须是Series，并使用MultiIndex")
    if not isinstance(pricing.index, pd.MultiIndex) or pricing.index.nlevels != 2:
        raise ValueError("价格数据必须是MultiIndex格式")

    if pricing.index.names in (["asset", "date"], [None, None]):
        asset_level, date_level = 0, 1
    elif pricing.index.names == ["date", "asset"]:
        asset_level, date_level = 1, 0
    else:
        raise ValueError(f"不支持的MultiIndex格式: {pricing.index.names}")

    dates = pd.to_datetime(pricing.index.get_level_values(date_level), errors="coerce")
    if dates.isna().any():
        raise ValueError("价格数据包含无法解析的日期")
    assets = pricing.index.get_level_values(asset_level)
    if pd.isna(assets).any():
        raise ValueError("价格数据包含空的 asset 键")
    canonical_index = pd.MultiIndex.from_arrays(
        [dates, assets], names=["date", "asset"]
    )
    canonical_prices = pd.Series(pricing.to_numpy(), index=canonical_index, name=pricing.name)
    if canonical_prices.index.has_duplicates:
        raise ValueError("价格数据包含重复的 (date, asset) 键")
    # Validate observed values before arithmetic; missing observations
    # remain missing and must never become zero or bridge a trading day.
    if pd.api.types.is_object_dtype(canonical_prices.dtype):
        real_values = canonical_prices.map(
            lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        )
    else:
        real_values = pd.Series(
            pd.api.types.is_numeric_dtype(canonical_prices.dtype)
            and not pd.api.types.is_bool_dtype(canonical_prices.dtype)
            and not pd.api.types.is_complex_dtype(canonical_prices.dtype),
            index=canonical_index,
        )
    numeric_prices = pd.to_numeric(canonical_prices.where(real_values), errors="coerce")
    numeric_prices = numeric_prices.where(canonical_prices.notna())
    invalid = canonical_prices.notna() & (
        ~real_values | ~np.isfinite(numeric_prices) | numeric_prices.le(0)
    )
    if invalid.any():
        key = invalid[invalid].index[0]
        raise ForwardReturnsValidationError(
            f"Invalid price at date={key[0]}, asset={key[1]!r}: "
            f"{canonical_prices.loc[key]!r}; expected a finite positive real number or missing value"
        )

    # All assets share the same sorted session rows, including missing prices.
    return numeric_prices.unstack("asset").sort_index()


def label_daily_returns(
    pricing: pd.Series, calendar: TradingCalendar | None, execution_lag: int, periods: Iterable[int],
) -> DailyReturnLabels:
    """Label daily closes using the calculator's iterable-horizon contract."""
    periods = validate_daily_horizons(periods, execution_lag)
    execution_lag = int(execution_lag)
    logger.info("开始计算前瞻收益率，期间: %s, 执行滞后: %s天", periods, execution_lag)
    prices_df = _daily_close_panel(pricing)
    calendar = resolve_daily_calendar(pricing, calendar)
    prices_df = calendar.align(prices_df)
    logger.info("价格数据形状: %s", prices_df.shape)

    entry_prices = prices_df.shift(-execution_lag) if execution_lag else prices_df
    forward_returns = {}
    for period in periods:
        exit_prices = prices_df.shift(-_DAILY.exit_offset(period) - execution_lag)
        returns = _DAILY.returns(entry_prices, exit_prices).replace([np.inf, -np.inf], np.nan)
        returns_series = returns.stack().dropna()
        returns_series.index.names = ["date", "asset"]
        forward_returns[f"{period}D"] = returns_series
        logger.info("%s 天前瞻收益率计算完成，有效数据点: %s", period, len(returns_series))

    result = pd.DataFrame(forward_returns)
    if result.empty:
        raise ForwardReturnsValidationError("未能计算出任何前瞻收益率：检查价格覆盖与请求的持有期/执行滞后")
    logger.info("前瞻收益率计算完成 (Lag=%s)，总形状: %s", execution_lag, result.shape)
    return DailyReturnLabels(result, prices_df, execution_lag, calendar)
