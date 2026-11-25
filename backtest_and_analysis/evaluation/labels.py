"""Explicit execution conventions for daily closes and exchange-scheduled Bars."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import pandas as pd

from utils.execution_schedule import ExecutionScheduler, compute_trade_rows

if TYPE_CHECKING:
    from utils.continuous_bars import ContinuousBarSource
    from utils.session_calendar import SessionCalendar, View
    from utils.slots.contracts import CalendarSlot, ExecutionProfile, MarketSlot
    from utils.time_axis import Frequency
    from utils.trading_calendar import TradingCalendar


logger = logging.getLogger(__name__)
_Prices = TypeVar("_Prices", pd.DataFrame, np.ndarray)
_Rows = TypeVar("_Rows", int, np.ndarray)


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
        return _price_returns(entry, exit, self.unit)


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


def _price_returns(entry: _Prices, exit: _Prices, unit: str) -> _Prices:
    result = exit / entry - 1
    return result * 100 if unit == "percent" else result


def _trade_rows(signal_rows: _Rows, lag: int, horizon: int, adjustment: int) -> tuple[_Rows, _Rows]:
    return compute_trade_rows(signal_rows, lag, horizon, adjustment)


def _window_counts(flags: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    prefix = np.concatenate(([0], np.cumsum(flags, dtype=np.int64)))
    return prefix[stops] - prefix[starts]


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
        _, offsets = _trade_rows(0, self._entry_offset, int(horizon), _DAILY.exit_adjustment)
        return pd.Series(sessions, index=sessions, name="exit_date").shift(-offsets)

    def endpoints(self, horizon: int) -> pd.DataFrame:
        """Trade dates and prices indexed by the signal keys with a valid return."""
        index = self.returns[f"{horizon}D"].dropna().index
        dates = self._closes.index
        signal_rows = dates.get_indexer(index.get_level_values("date"))
        columns = self._closes.columns.get_indexer(index.get_level_values("asset"))
        entry_rows, exit_rows = _trade_rows(
            signal_rows, self._entry_offset, int(horizon), _DAILY.exit_adjustment,
        )
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
    return UnifiedReturnLabeler._from_convention(
        ExecutionConvention.CLOSE_TO_CLOSE, "cross_session",
    )._label_daily(pricing, calendar, execution_lag, periods)


@dataclass(frozen=True)
class _LabelAxis:
    keys: pd.DatetimeIndex
    opens: pd.DatetimeIndex
    ends: pd.DatetimeIndex
    owners: np.ndarray

    def execution_times(self, price_field: str) -> pd.DatetimeIndex:
        return self.opens if price_field == "open" else self.ends

    @classmethod
    def from_slot_grid(
        cls, calendar: CalendarSlot, grid: pd.DataFrame, frequency: Frequency, view: View,
        bucket_strategy: str | None = None,
    ) -> _LabelAxis:
        from utils.session_calendar import SessionCalendar
        from utils.time_axis import BarSchedule

        if isinstance(calendar, SessionCalendar):
            schedule = BarSchedule(calendar, frequency, view=view, bucket_strategy=bucket_strategy)
            sessions = [schedule.for_session(day) for day in grid.session_id.unique()]
            slots = pd.concat(sessions, ignore_index=True)
            keys = pd.DatetimeIndex(grid.timestamp_utc)
            return cls(keys, pd.DatetimeIndex(slots.start_utc), pd.DatetimeIndex(slots.end_utc),
                       grid.session_id.to_numpy())
        starts, ends = [], []
        if frequency.kind == "week":
            daily = calendar.bar_grid("1d", view=view)
            weeks = pd.to_datetime(daily.session_id).dt.to_period("W-SUN")
            first_in_week = daily.groupby(weeks).timestamp_utc.first()
        for session_id, rows in grid.groupby("session_id", sort=False):
            keys = pd.DatetimeIndex(rows.timestamp_utc)
            session = calendar.session_at(keys[0], view, frequency=frequency)
            if session is None or session.session_date != session_id:
                raise ValueError("Slot calendar grid keys must belong to their declared session")
            close = session.segments[-1][1]
            if frequency.kind == "intraday":
                # The next open is the close unless it starts a new trading segment.
                # In that case the preceding Bar ends at the break, before that open.
                break_opens = [opening for opening, _ in session.segments[1:]]
                break_closes = [closing for _, closing in session.segments[:-1]]
                closes = keys[1:].append(pd.DatetimeIndex([close]))
                if break_opens:
                    breaks = pd.Series(break_closes, index=break_opens)
                    closes = pd.DatetimeIndex(breaks.reindex(closes).fillna(closes.to_series(index=closes)))
                starts.append(keys)
                ends.append(closes)
            else:
                opening_session = session
                if frequency.kind == "week":
                    day_key = first_in_week.loc[pd.Period(session_id, freq="W-SUN")]
                    opening_session = calendar.session_at(day_key, view, frequency="1d")
                    if opening_session is None:
                        raise ValueError("Slot calendar must provide the opening session of its weekly grid")
                starts.append(pd.DatetimeIndex([opening_session.segments[0][0]]))
                ends.append(pd.DatetimeIndex([close]))
        opens, closes = starts[0].append(starts[1:]), ends[0].append(ends[1:])
        keys = opens if frequency.kind == "intraday" else closes
        return cls(keys, opens, closes, grid.session_id.to_numpy())


def _slot_grid(
    calendar: CalendarSlot, frequency: Frequency, start: str | pd.Timestamp, end: str | pd.Timestamp,
    *, view: View, bucket_strategy: str | None, lookahead_bars: int,
) -> tuple[pd.DataFrame, int]:
    requested = calendar.bar_grid(frequency, start, end, view=view, bucket_strategy=bucket_strategy)
    if requested.empty or not lookahead_bars:
        return requested, len(requested)
    grids = [requested]
    later = calendar.sessions(start=end)
    later = later[later > pd.Timestamp(end)]
    # Extend the calendar in complete sessions, leaving the price bounds untouched.
    while lookahead_bars > 0 and len(later):
        if frequency.kind == "week":
            batch = lookahead_bars * 7
        elif frequency.kind == "intraday":
            batch = min(7, lookahead_bars)
        else:
            batch = lookahead_bars
        days, later = later[:batch], later[batch:]
        scheduled = calendar.bar_grid(frequency, days[0], days[-1], view=view, bucket_strategy=bucket_strategy)
        if not scheduled.empty:
            grids.append(scheduled)
            lookahead_bars -= len(scheduled)
    return pd.concat(grids, ignore_index=True), len(requested)



def _execution_bars_digest(bars: pd.DataFrame) -> str:
    """Hash ordered execution values independently of DataFrame pickle metadata."""
    digest = hashlib.sha256()

    class Sink:
        def write(self, value: str) -> int:
            digest.update(value.encode('utf-8'))
            return len(value)

    bars.sort_values(['instrument_id', 'timestamp_utc']).to_csv(
        Sink(), index=False, lineterminator='\n', float_format='%.17g',
    )
    return digest.hexdigest()


class UnifiedReturnLabeler:
    """Label execution windows using an ExecutionProfile and a scheduled bar grid."""

    def __init__(
        self, profile: ExecutionProfile | None = None, *, slot: MarketSlot | None = None,
        apply_tradability_mask: bool = False,
    ) -> None:
        from utils.slots.contracts import ExecutionProfile

        if profile is None and slot is not None:
            profile = slot.microstructure.execution_profile
        if not isinstance(profile, ExecutionProfile):
            raise ValueError("Supply an ExecutionProfile or a MarketSlot")
        if not isinstance(apply_tradability_mask, bool) or (apply_tradability_mask and slot is None):
            raise ValueError("apply_tradability_mask must be a boolean and requires a MarketSlot when enabled")
        self.profile = profile
        self.slot = slot
        self.apply_tradability_mask = apply_tradability_mask

    @classmethod
    def from_slot(
        cls, slot: MarketSlot, *, profile: ExecutionProfile | None = None,
        apply_tradability_mask: bool = False,
    ) -> UnifiedReturnLabeler:
        return cls(profile, slot=slot, apply_tradability_mask=apply_tradability_mask)

    @classmethod
    def _from_convention(cls, convention: ExecutionConvention, boundary_policy: str) -> UnifiedReturnLabeler:
        from utils.slots.contracts import ExecutionProfile

        rules = convention.value
        return cls(ExecutionProfile(
            rules.entry_price_field, rules.exit_price_field, rules.min_entry_offset,
            rules.exit_adjustment, rules.unit, rules.gap_rule, boundary_policy,
        ))

    def scheduler(self, entry_offset_bars: int | None = None) -> ExecutionScheduler:
        """Build an ``ExecutionScheduler`` enforcing this labeler's ``ExecutionProfile``."""
        from backtest_and_analysis.time_series.contracts import ContractValidationError

        try:
            return ExecutionScheduler(self.profile, entry_offset_bars=entry_offset_bars)
        except ValueError as exc:
            raise ContractValidationError(str(exc)) from exc

    def _label_daily(
        self, pricing: pd.Series, calendar: TradingCalendar | None,
        execution_lag: int, periods: Iterable[int],
    ) -> DailyReturnLabels:
        periods = validate_daily_horizons(periods, execution_lag)
        execution_lag = int(execution_lag)
        sched = self.scheduler(execution_lag)
        logger.info("开始计算前瞻收益率，期间: %s, 执行滞后: %s天", periods, execution_lag)
        prices_df = _daily_close_panel(pricing)
        calendar = resolve_daily_calendar(pricing, calendar)
        prices_df = calendar.align(prices_df)
        logger.info("价格数据形状: %s", prices_df.shape)

        entry_prices = prices_df.shift(-execution_lag) if execution_lag else prices_df
        forward_returns = {}
        for period in periods:
            _, exit_row = sched.trade_bar_indices(0, period)
            exit_prices = prices_df.shift(-exit_row)
            returns = _price_returns(entry_prices, exit_prices, self.profile.return_unit).replace(
                [np.inf, -np.inf], np.nan,
            )
            returns_series = returns.stack().dropna()
            returns_series.index.names = ["date", "asset"]
            forward_returns[f"{period}D"] = returns_series
            logger.info("%s 天前瞻收益率计算完成，有效数据点: %s", period, len(returns_series))

        result = pd.DataFrame(forward_returns)
        if result.empty:
            raise ForwardReturnsValidationError("未能计算出任何前瞻收益率：检查价格覆盖与请求的持有期/执行滞后")
        logger.info("前瞻收益率计算完成 (Lag=%s)，总形状: %s", execution_lag, result.shape)
        return DailyReturnLabels(result, prices_df, execution_lag, calendar)

    def label(
        self, instruments: Sequence[str], frequency: str | Frequency, *,
        start_date: str | pd.Timestamp, end_date: str | pd.Timestamp,
        horizons: Sequence[int] | int, entry_offset_bars: int | None = None,
        factors: pd.DataFrame | None = None, bucket_strategy: str | None = None,
        capture_data_identity: bool = False,
    ) -> pd.DataFrame:
        """Read Slot execution bars over inclusive session dates and return long labels.

        External factors use the Slot's public grid keys. Without factors, daily
        candidates use scheduled sessions and intraday candidates use observed bars.
        Returns use the profile's unit and the return-price endpoints. Entry
        and exit prices also record the unadjusted fills.
        """
        from backtest_and_analysis.time_series.contracts import _validate_aligned_factor_frame
        from backtest_and_analysis.time_series.labels import LABEL_COLUMNS, _normalize_bar_horizons
        from utils.time_axis import Frequency

        if self.slot is None:
            raise ValueError("label requires a MarketSlot")
        if isinstance(instruments, str) or not instruments or any(
            not isinstance(instrument, str) or not instrument.strip() for instrument in instruments
        ) or len(set(instruments)) != len(instruments):
            raise ValueError("instruments must be a nonempty sequence of distinct instrument IDs")
        periods = _normalize_bar_horizons(horizons)
        sched = self.scheduler(entry_offset_bars)
        lag = sched.resolved_entry_offset_bars
        factors = (None if factors is None else _validate_aligned_factor_frame(factors)
                   .set_index(["timestamp_utc", "instrument_id"]).factor_value)
        slot = self.slot
        view = getattr(slot.bar_feed, "view", "ETH")
        parsed = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
        calendar = slot.calendar
        strategy = self.resolve_bucket_strategy(parsed, instruments, bucket_strategy)
        provenance = {}
        lookahead = sched.lookahead_bars(max(periods))
        grid, requested_count = _slot_grid(
            calendar, parsed, start_date, end_date, view=view, bucket_strategy=strategy, lookahead_bars=lookahead,
        )
        tables = []
        if requested_count:
            axis = _LabelAxis.from_slot_grid(calendar, grid, parsed, view, strategy)
            keys = axis.keys
            price_keys = keys[:requested_count]
            bars = slot.bar_feed.read_execution_bars(
                instruments, parsed, start_utc=price_keys.min(), end_utc=price_keys.max(),
            )
            bars = self._validate_execution_bars(bars, instruments, parsed)
            scheduled_sessions = bars.timestamp_utc.map(pd.Series(axis.owners[:requested_count], index=price_keys))
            if not scheduled_sessions.eq(bars.session_id).all():
                raise ValueError("Execution bars must belong to their scheduled Slot session and grid")
            if capture_data_identity:
                provenance = {
                    "execution_bars_sha256": _execution_bars_digest(bars),
                    "sources": sorted(bars.source.dropna().unique()) if "source" in bars else bars.attrs.get("sources", []),
                    "data_versions": sorted(bars.data_version.dropna().unique()) if "data_version" in bars else [],
                    "raw_coverage_start_utc": bars.timestamp_utc.min(),
                    "raw_coverage_end_utc": bars.timestamp_utc.max(),
                }
            for instrument in sorted(instruments):
                group = bars.loc[bars.instrument_id.eq(instrument)].set_index("timestamp_utc")
                observed = group.reindex(keys)
                present = observed.instrument_id.iloc[:requested_count].notna().to_numpy()
                factor_missing = False
                if factors is not None:
                    lookup = pd.MultiIndex.from_arrays([price_keys, np.repeat(instrument, requested_count)])
                    candidates = lookup.isin(factors.index)
                    factor_missing = factors.reindex(lookup[candidates]).isna().to_numpy()
                else:
                    candidates = np.ones(requested_count, dtype=bool) if parsed.kind == "session" else present
                positions = np.flatnonzero(candidates)
                tables.extend(self._label_windows(
                    observed, axis, positions, keys, positions, group.index.max(),
                    instrument, parsed.canonical, lag, periods, factor_missing,
                    require_signal_bar=factors is None and parsed.kind == "intraday",
                    include_scheduled_exit=True,
                ))

        columns = ["forward_return" if name == "return_pct" else name for name in LABEL_COLUMNS]
        columns += ["scheduled_exit_timestamp", "signal_session_id", "return_unit",
                    "entry_return_price", "exit_return_price"]
        if not tables:
            result = pd.DataFrame(columns=columns)
            for name in ("signal_timestamp", "signal_available_at", "entry_timestamp", "exit_bar_timestamp",
                         "exit_timestamp", "scheduled_exit_timestamp"):
                result[name] = pd.Series(dtype="datetime64[ns, UTC]")
        else:
            result = pd.concat(tables, ignore_index=True).loc[:, columns].sort_values(
                ["instrument_id", "signal_timestamp", "horizon_bars"], kind="stable",
            ).reset_index(drop=True)
        result.attrs.update(provenance)
        return result

    def resolve_bucket_strategy(
        self, frequency: Frequency, instruments: Sequence[str], bucket_strategy: str | None,
    ) -> str | None:
        slot = self.slot
        if slot is None:
            raise ValueError("Resolving a bucket strategy requires a MarketSlot")
        strategy = bucket_strategy if bucket_strategy is not None else getattr(slot.bar_feed, "bucket_strategy", None)
        if strategy is None and frequency.kind == "session":
            strategy = "session"
        feed_calendar = getattr(slot.bar_feed, "calendar", slot.calendar)
        if strategy is None and getattr(feed_calendar, "is_domestic_futures", False) and frequency.kind == "intraday":
            from utils.factor_core.bar_bridge import _bucket_strategy

            strategies = {_bucket_strategy(slot.bar_feed, frequency, instrument) for instrument in instruments}
            if len(strategies) != 1:
                raise ValueError("Requested instruments must use the same bucket_strategy")
            strategy = strategies.pop()
        return strategy

    @staticmethod
    def _validate_execution_bars(
        bars: pd.DataFrame, instruments: Sequence[str], frequency: Frequency,
    ) -> pd.DataFrame:
        from backtest_and_analysis.time_series.contracts import _frame_keys, _numeric_column

        keys = ("timestamp_utc", "instrument_id")
        required = {*keys, "session_id", "open", "high", "low", "close"}
        if not isinstance(bars, pd.DataFrame) or not bars.columns.is_unique or required - set(bars):
            raise ValueError("Execution bars require unique columns with UTC keys, session_id and OHLC")
        bars = _frame_keys(bars, keys)
        if not bars.instrument_id.isin(instruments).all():
            raise ValueError("Execution bars contain an unrequested instrument")
        if "frequency" in bars and not bars.frequency.eq(frequency.canonical).all():
            raise ValueError("Execution bars do not match the requested frequency")
        if "concrete_instrument_id" in bars and not bars.concrete_instrument_id.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError("Execution bars require nonempty concrete_instrument_id values")
        numeric_fields = {"open", "high", "low", "close", "factor", "total_return_factor"}
        for field in numeric_fields.intersection(bars):
            bars[field] = _numeric_column(bars[field]) if not bars.empty else bars[field].astype(float)
        return bars

    def _label_windows(
        self, observed: pd.DataFrame, axis: _LabelAxis, positions: np.ndarray,
        coverage_keys: pd.DatetimeIndex, coverage_positions: np.ndarray, coverage_end: pd.Timestamp,
        instrument: str, frequency: str, lag: int, horizons: Sequence[int],
        factor_missing: bool | np.ndarray, *, incomplete_signal: bool | np.ndarray = False,
        require_signal_bar: bool = False,
        include_scheduled_exit: bool = False,
    ) -> list[pd.DataFrame]:
        from backtest_and_analysis.time_series.contracts import invalid_price_mask, primary_exclusion_reasons
        from backtest_and_analysis.time_series.labels import TIME_OUTPUTS, VALUE_OUTPUTS

        profile = self.profile
        is_slot = self.slot is not None
        entry_times = axis.execution_times(profile.entry_price_field)
        exit_times = axis.execution_times(profile.exit_price_field)
        last = len(axis.keys) - 1
        missing = observed.instrument_id.isna().to_numpy()
        if "is_complete" in observed:
            missing |= ~observed.is_complete.eq(True).to_numpy(dtype=bool, na_value=False)
        if require_signal_bar:
            incomplete_signal = incomplete_signal | missing[positions]
        opening = observed[profile.entry_price_field].to_numpy(dtype=float, na_value=np.nan)
        closing = observed[profile.exit_price_field].to_numpy(dtype=float, na_value=np.nan)
        complete_window = profile.gap_rule == "require_complete_bars"
        invalid = invalid_price_mask(observed).to_numpy() if complete_window else None
        adjusted_entry, adjusted_exit = opening, closing
        if is_slot:
            factor_field = "total_return_factor"
            # Native futures expose an optional storage factor, separate from execution returns.
            if factor_field not in observed and "concrete_instrument_id" not in observed:
                factor_field = "factor"
            if factor_field in observed:
                from utils.pit import apply_pit_adjustment

                adjusted = apply_pit_adjustment(observed.assign(code=instrument, date=axis.keys), factor_field)
                adjusted_entry = adjusted[profile.entry_price_field].to_numpy(dtype=float, na_value=np.nan)
                adjusted_exit = adjusted[profile.exit_price_field].to_numpy(dtype=float, na_value=np.nan)
        if not len(positions):
            return []
        invalid_entry = ~np.isfinite(adjusted_entry) | (adjusted_entry <= 0)
        invalid_exit = ~np.isfinite(adjusted_exit) | (adjusted_exit <= 0)
        changes = None
        if is_slot and "concrete_instrument_id" in observed:
            concrete = observed.concrete_instrument_id.ffill()
            changes = concrete.notna().to_numpy() & concrete.ne(concrete.shift()).to_numpy()

        sched = self.scheduler(lag)
        tables = []
        for horizon in horizons:
            entries, exits = sched.trade_bar_indices(positions, horizon)
            _, coverage_exits = sched.trade_bar_indices(coverage_positions, horizon)
            entry, exit = np.minimum(entries, last), np.minimum(exits, last)
            boundary = (exits > last) | sched.crosses_session_boundary(
                axis.owners[positions], axis.owners[exit],
            )
            insufficient = pd.isna(coverage_end) | np.where(
                boundary,
                (coverage_exits >= len(coverage_keys)) |
                (coverage_keys.take(np.minimum(coverage_exits, len(coverage_keys) - 1)) > coverage_end),
                axis.keys.take(exit) > coverage_end,
            )
            missing_window = missing[entry] | missing[exit]
            bad_prices = invalid_entry[entry] | invalid_exit[exit]
            if complete_window:
                missing_window |= _window_counts(missing, np.minimum(positions + 1, entry), exit + 1) > 0
                bad_prices |= _window_counts(invalid, entry, exit + 1) > 0
            reasons = primary_exclusion_reasons(pd.DataFrame({
                "factor_missing": factor_missing,
                "insufficient_history": insufficient,
                "session_boundary": boundary,
                "missing_bar": missing_window | incomplete_signal,
                "invalid_price": bad_prices,
            }))
            if is_slot:
                entry_before_signal = entry_times.take(entry) < axis.ends.take(positions)
                exit_before_entry = exit_times.take(exit) < entry_times.take(entry)
                reasons = reasons.mask(reasons.isna() & (entry_before_signal | exit_before_entry),
                                       "invalid_execution_order")
            if changes is not None:
                crossed = _window_counts(changes, entry + 1, exit + 1) > 0
                reasons = reasons.mask(reasons.isna() & crossed, "contract_roll")
            if self.apply_tradability_mask:
                eligible = np.flatnonzero(reasons.isna().to_numpy())
                rejected = self._untradable_endpoints(observed, axis, entry[eligible], exit[eligible], instrument)
                reasons.iloc[eligible[rejected == "entry_not_tradable"]] = "entry_not_tradable"
                reasons.iloc[eligible[rejected == "exit_not_tradable"]] = "exit_not_tradable"
            valid = reasons.isna().to_numpy()
            result = pd.DataFrame({
                "instrument_id": instrument, "frequency": frequency,
                "entry_offset_bars": lag, "horizon_bars": horizon,
                "signal_timestamp": axis.keys.take(positions), "signal_available_at": axis.ends.take(positions),
                "entry_timestamp": entry_times.take(entry), "exit_bar_timestamp": axis.keys.take(exit),
                "exit_timestamp": exit_times.take(exit),
                "status": np.where(valid, "valid", "excluded"), "exclusion_reason": reasons,
            })
            if include_scheduled_exit:
                result["scheduled_exit_timestamp"] = result.exit_timestamp.where(~boundary)
            result.loc[~valid, list(TIME_OUTPUTS)] = pd.NaT
            result.loc[:, list(VALUE_OUTPUTS)] = np.nan
            result.loc[valid, "entry_price"] = opening[entry[valid]]
            result.loc[valid, "exit_price"] = closing[exit[valid]]
            result.loc[valid, "price_change_points"] = closing[exit[valid]] - opening[entry[valid]]
            returns = _price_returns(adjusted_entry[entry[valid]], adjusted_exit[exit[valid]], profile.return_unit)
            result.loc[valid, "return_pct"] = returns
            if is_slot:
                result = result.rename(columns={"return_pct": "forward_return"})
                result["signal_session_id"] = axis.owners[positions]
                result["return_unit"] = profile.return_unit
                result["entry_return_price"] = np.where(valid, adjusted_entry[entry], np.nan)
                result["exit_return_price"] = np.where(valid, adjusted_exit[exit], np.nan)
            tables.append(result)
        return tables

    def _untradable_endpoints(
        self, observed: pd.DataFrame, axis: _LabelAxis, entries: np.ndarray, exits: np.ndarray,
        instrument: str,
    ) -> np.ndarray:
        rejected = np.full(len(entries), "", dtype=object)
        if not len(entries):
            return rejected
        for rows, field, side, reason in (
            (entries, self.profile.entry_price_field, "buy", "entry_not_tradable"),
            (exits, self.profile.exit_price_field, "sell", "exit_not_tradable"),
        ):
            endpoints = observed.iloc[rows].assign(
                timestamp_utc=axis.execution_times(field).take(rows), session_id=axis.owners[rows],
                side=side, execution_price_field=field, close=observed[field].iloc[rows].to_numpy(),
            ).reset_index(drop=True)
            if side == "sell":
                endpoints["entry_session_id"] = axis.owners[entries]
            mask = self.slot.microstructure.tradability_mask(endpoints, instrument_id=instrument)
            if (not isinstance(mask, pd.Series) or not mask.index.equals(endpoints.index)
                    or not pd.api.types.is_bool_dtype(mask) or mask.isna().any()):
                raise ValueError("tradability_mask must return a non-null boolean Series indexed by the endpoint grid")
            rejected[(rejected == "") & ~mask.to_numpy(dtype=bool)] = reason
        return rejected
