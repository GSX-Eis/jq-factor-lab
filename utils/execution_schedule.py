"""Unified signal-to-execution bar timing contract shared across evaluation, synthesis, and backtest."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
import re
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from utils.slots.contracts import ExecutionProfile, PortfolioExecutionProfile

_Rows = TypeVar("_Rows", int, np.ndarray)
_SESSION_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _execution_profile_cls() -> type[ExecutionProfile]:
    from utils.slots.contracts import ExecutionProfile

    return ExecutionProfile


def _portfolio_execution_profile_cls() -> type[PortfolioExecutionProfile]:
    from utils.slots.contracts import PortfolioExecutionProfile

    return PortfolioExecutionProfile


def resolve_slot_backtest_profile(slot: Any) -> ExecutionProfile | None:
    """Return the portfolio backtest ``ExecutionProfile`` declared by ``slot``."""
    if slot is None:
        return None
    micro = getattr(slot, "microstructure", None)
    if micro is None:
        return None
    portfolio_profile = getattr(micro, "portfolio_execution_profile", None)
    if portfolio_profile is not None:
        return portfolio_profile
    return getattr(micro, "execution_profile", None)


def validate_non_bool_int(
    value: Any,
    name: str,
    minimum: int,
    *,
    error_cls: type[Exception] = ValueError,
) -> int:
    """Validate an integer (excluding bool) is at least ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise error_cls(f"{name} must be an integer >= {minimum}")
    return int(value)


def compute_trade_rows(
    signal_rows: _Rows,
    entry_offset_bars: int,
    horizon: int,
    exit_adjustment: int,
) -> tuple[_Rows, _Rows]:
    """Map signal grid positions to ``(entry_rows, exit_rows)``."""
    entry_rows = signal_rows + entry_offset_bars
    return entry_rows, entry_rows + (horizon + exit_adjustment)


@dataclass(frozen=True)
class ExecutionScheduler:
    """Pure timing contract mapping signal bars, entry bars, and exit bars."""

    profile: ExecutionProfile
    entry_offset_bars: int | None = None
    resolved_entry_offset_bars: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile, _execution_profile_cls()):
            raise ValueError("profile must be an ExecutionProfile")
        if self.profile.boundary_policy not in ("same_session", "cross_session"):
            raise ValueError(f"Unsupported boundary_policy: {self.profile.boundary_policy!r}")
        if self.entry_offset_bars is None:
            resolved = int(self.profile.min_entry_offset_bars)
        else:
            resolved = validate_non_bool_int(
                self.entry_offset_bars,
                "entry_offset_bars",
                self.profile.min_entry_offset_bars,
            )
        object.__setattr__(self, "resolved_entry_offset_bars", resolved)

    @classmethod
    def from_slot(
        cls,
        slot: Any,
        *,
        entry_offset_bars: int | None = None,
    ) -> ExecutionScheduler:
        """Build an ``ExecutionScheduler`` from a ``MarketSlot``."""
        profile = getattr(getattr(slot, "microstructure", None), "execution_profile", None)
        if not isinstance(profile, _execution_profile_cls()):
            raise ValueError("slot must expose microstructure.execution_profile")
        return cls(profile, entry_offset_bars=entry_offset_bars)

    @classmethod
    def for_backtest(
        cls,
        profile: ExecutionProfile,
        *,
        entry_offset_bars: int | None = None,
        effective_entry_price_field: str | None = None,
        allow_entry_field_override: bool = False,
        holding_period: int = 1,
    ) -> ExecutionScheduler:
        """Validate ``ExecutionProfile`` timing for ``VectorBTBacktestRunner`` and resolve lag."""
        if not isinstance(profile, _execution_profile_cls()):
            raise ValueError("profile must be an ExecutionProfile")
        validate_non_bool_int(holding_period, "holding_period", 1)

        entry_field = effective_entry_price_field or profile.entry_price_field
        if entry_field not in ("open", "close"):
            raise ValueError(f"Unsupported entry_price_field for backtest: {entry_field!r}")
        if not allow_entry_field_override and entry_field != profile.entry_price_field:
            raise ValueError(
                f"Effective entry_price_field={entry_field!r} disagrees with "
                f"declared profile.entry_price_field={profile.entry_price_field!r}"
            )

        if profile.boundary_policy != "cross_session":
            raise ValueError(
                f"VectorBTBacktestRunner does not support boundary_policy={profile.boundary_policy!r}; "
                "continuous portfolio simulation requires 'cross_session'"
            )

        if profile.exit_adjustment != 0:
            raise ValueError(
                f"VectorBTBacktestRunner does not support exit_adjustment={profile.exit_adjustment}; "
                "continuous portfolio simulation only supports exit_adjustment=0"
            )

        if allow_entry_field_override:
            if profile.exit_price_field != profile.entry_price_field:
                raise ValueError(
                    "VectorBTBacktestRunner does not support mismatched entry/exit price fields "
                    f"(entry_price_field={profile.entry_price_field!r}, "
                    f"exit_price_field={profile.exit_price_field!r}); "
                    "continuous portfolio simulation requires exit_price_field == entry_price_field"
                )
        elif profile.exit_price_field != entry_field:
            raise ValueError(
                "VectorBTBacktestRunner does not support mismatched entry/exit price fields "
                f"(entry_price_field={entry_field!r}, "
                f"exit_price_field={profile.exit_price_field!r}); "
                "continuous portfolio simulation requires exit_price_field == entry_price_field"
            )

        if entry_offset_bars is None:
            resolved_offset = max(1, int(profile.min_entry_offset_bars))
        else:
            resolved_offset = validate_non_bool_int(
                entry_offset_bars,
                "entry_offset_bars",
                profile.min_entry_offset_bars,
            )
            if resolved_offset < 1:
                raise ValueError(
                    "VectorBTBacktestRunner requires entry_offset_bars >= 1; "
                    "same-bar entry_offset_bars=0 is not supported"
                )

        return cls(profile, entry_offset_bars=resolved_offset)

    def effective_profile(
        self,
        *,
        entry_price_field: str | None = None,
        exit_price_field: str | None = None,
        allow_short: bool | None = None,
        benchmark_instrument: str | None = None,
        update_benchmark: bool = False,
        commission_bps: float | None = None,
        sell_tax_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> ExecutionProfile:
        """Return an ``ExecutionProfile`` reflecting the resolved execution parameters."""
        overrides: dict[str, Any] = {
            "min_entry_offset_bars": self.resolved_entry_offset_bars,
        }
        if entry_price_field is not None:
            overrides["entry_price_field"] = entry_price_field
        if exit_price_field is not None:
            overrides["exit_price_field"] = exit_price_field
        if allow_short is not None:
            overrides["allow_short"] = bool(allow_short)
        if update_benchmark:
            overrides["benchmark_instrument"] = benchmark_instrument
        if commission_bps is not None:
            overrides["commission_bps"] = float(commission_bps)
        if sell_tax_bps is not None:
            overrides["sell_tax_bps"] = float(sell_tax_bps)
        if slippage_bps is not None:
            overrides["slippage_bps"] = float(slippage_bps)
        return replace(self.profile, **overrides)

    def entry_bar_index(self, signal_pos: _Rows) -> _Rows:
        """Map signal bar index (or array of indices) to entry bar index."""
        return signal_pos + self.resolved_entry_offset_bars

    def exit_offset_from_entry(self, horizon: int) -> int:
        """Return bar delta from entry bar to exit bar for ``horizon``."""
        validated = validate_non_bool_int(horizon, "horizon", 1)
        delta = validated + self.profile.exit_adjustment
        if delta < 0:
            raise ValueError(
                f"Invalid exit offset {delta} for horizon={validated} and "
                f"exit_adjustment={self.profile.exit_adjustment}"
            )
        return delta

    def exit_bar_index(self, entry_pos: _Rows, horizon: int) -> _Rows:
        """Map entry bar index (or array of indices) to exit bar index."""
        return entry_pos + self.exit_offset_from_entry(horizon)

    def trade_bar_indices(self, signal_pos: _Rows, horizon: int) -> tuple[_Rows, _Rows]:
        """Map signal bar index (or array) to ``(entry_pos, exit_pos)``."""
        validated = validate_non_bool_int(horizon, "horizon", 1)
        return compute_trade_rows(
            signal_pos,
            self.resolved_entry_offset_bars,
            validated,
            self.profile.exit_adjustment,
        )

    def lookahead_bars(self, max_horizon: int) -> int:
        """Total lookahead bars needed past signal bar for ``max_horizon``."""
        _, lookahead = self.trade_bar_indices(0, max_horizon)
        return int(lookahead)

    def crosses_session_boundary(
        self,
        signal_sessions: np.ndarray,
        exit_sessions: np.ndarray,
    ) -> np.ndarray:
        """Return boolean mask of windows that violate ``profile.boundary_policy``."""
        if self.profile.boundary_policy == "same_session":
            return np.asarray(signal_sessions) != np.asarray(exit_sessions)
        return np.zeros(len(signal_sessions), dtype=bool)

    def signal_bar_index(self, execution_pos: int) -> int | None:
        """Map an execution bar index back to its signal bar index, or ``None``."""
        pos = validate_non_bool_int(execution_pos, "execution_pos", 0)
        signal_pos = pos - self.resolved_entry_offset_bars
        return signal_pos if signal_pos >= 0 else None

    def signal_timestamp_for_position(
        self,
        index: Sequence[Any] | pd.Index,
        execution_pos: int,
    ) -> Any | None:
        """Return the signal bar key/timestamp for ``index[execution_pos]``, or ``None``."""
        signal_pos = self.signal_bar_index(execution_pos)
        if signal_pos is None or signal_pos >= len(index):
            return None
        return index[signal_pos]

    def signal_timestamp_map(
        self,
        index: Sequence[Any] | pd.Index,
        *,
        key_fn: Callable[[Any], Any] | None = None,
    ) -> dict[Any, Any]:
        """Build execution-bar -> signal-bar mapping across ``index``."""
        offset = self.resolved_entry_offset_bars
        if key_fn is None:
            return {index[idx]: index[idx - offset] for idx in range(offset, len(index))}
        return {
            key_fn(index[idx]): key_fn(index[idx - offset])
            for idx in range(offset, len(index))
        }


def resolve_calendar_signal_bar(
    calendar: Any,
    current_bar_key: str,
    *,
    frequency: str,
    entry_offset_bars: int,
) -> str | None:
    """Resolve the signal bar ``entry_offset_bars`` prior to ``current_bar_key`` via ``calendar``."""
    offset = validate_non_bool_int(entry_offset_bars, "entry_offset_bars", 1)
    if calendar is None:
        return None
    if _SESSION_DATE_RE.fullmatch(current_bar_key) and frequency == "1d":
        sessions = calendar.sessions()
        if len(sessions) < offset:
            return None
        try:
            prev = calendar.previous_sessions(current_bar_key, offset)
        except Exception:
            return None
        if len(prev) < offset:
            return None
        return pd.Timestamp(prev[0]).strftime("%Y-%m-%d")

    ts = pd.Timestamp(current_bar_key)
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    session = calendar.session_at(ts_utc, frequency=frequency) or calendar.session_at(ts_utc)
    if session is None:
        return None
    lookback_days = max(14, offset * 7)
    start_day = max(calendar.start, pd.Timestamp(session.session_date) - pd.Timedelta(days=lookback_days))
    grid = calendar.bar_grid(frequency, start_day, session.session_date)
    prior = grid.loc[grid["timestamp_utc"] < ts_utc, "timestamp_utc"]
    if len(prior) < offset and start_day > calendar.start:
        grid = calendar.bar_grid(frequency, calendar.start, session.session_date)
        prior = grid.loc[grid["timestamp_utc"] < ts_utc, "timestamp_utc"]
    if len(prior) < offset:
        return None
    return pd.Timestamp(prior.iloc[-offset]).isoformat()


def resolve_runner_execution(
    runner: Any,
    effective_entry_field: str,
) -> tuple[ExecutionScheduler, PortfolioExecutionProfile]:
    """Build and validate ``(ExecutionScheduler, effective_PortfolioExecutionProfile)`` for a backtest runner."""
    slot = getattr(runner, "slot", None)
    profile = resolve_slot_backtest_profile(slot)
    allow_override = slot is not None and slot.slot_id == "jq_ashare_local"
    if profile is None:
        profile = _portfolio_execution_profile_cls()(
            effective_entry_field, effective_entry_field, 1, 0, "decimal", "endpoints_only", "cross_session"
        )
        allow_override = True
    scheduler = ExecutionScheduler.for_backtest(
        profile,
        entry_offset_bars=getattr(runner, "entry_offset_bars", None),
        effective_entry_price_field=effective_entry_field,
        allow_entry_field_override=allow_override,
        holding_period=getattr(runner, "holding_period", 1),
    )
    buy_rate = float(getattr(runner, "buy_cost_rate", 0.0))
    sell_rate = float(getattr(runner, "sell_cost_rate", 0.0))
    slippage = float(getattr(runner, "slippage", 0.0))
    effective_exit_field = effective_entry_field if allow_override else profile.exit_price_field
    effective_raw = scheduler.effective_profile(
        entry_price_field=effective_entry_field,
        exit_price_field=effective_exit_field,
        allow_short=bool(getattr(runner, "allow_short", False)),
        benchmark_instrument=getattr(runner, "benchmark_instrument", None),
        update_benchmark=True,
        commission_bps=round(buy_rate * 10_000.0, 12),
        sell_tax_bps=round(max(0.0, sell_rate - buy_rate) * 10_000.0, 12),
        slippage_bps=round(slippage * 10_000.0, 12),
    )
    effective = _portfolio_execution_profile_cls()(**asdict(effective_raw))
    validate_effective_execution_profile(
        profile,
        effective,
        allow_entry_field_override=allow_override,
    )
    return scheduler, effective


def validate_effective_execution_profile(
    declared: ExecutionProfile,
    effective: ExecutionProfile,
    *,
    allow_entry_field_override: bool = False,
) -> None:
    """Fail fast if effective backtest timing diverges from the declared ``ExecutionProfile``."""
    profile_cls = _execution_profile_cls()
    if not isinstance(declared, profile_cls) or not isinstance(effective, profile_cls):
        raise ValueError("declared and effective profiles must be ExecutionProfile instances")
    if effective.min_entry_offset_bars < declared.min_entry_offset_bars:
        raise ValueError(
            f"Effective min_entry_offset_bars={effective.min_entry_offset_bars} violates "
            f"declared min_entry_offset_bars={declared.min_entry_offset_bars}"
        )
    if effective.exit_adjustment != declared.exit_adjustment:
        raise ValueError(
            f"Effective exit_adjustment={effective.exit_adjustment} disagrees with "
            f"declared exit_adjustment={declared.exit_adjustment}"
        )
    if effective.exit_adjustment != 0:
        raise ValueError(
            f"Effective exit_adjustment={effective.exit_adjustment} is unsupported; "
            "continuous portfolio simulation only supports exit_adjustment=0"
        )
    if effective.boundary_policy != declared.boundary_policy:
        raise ValueError(
            f"Effective boundary_policy={effective.boundary_policy!r} disagrees with "
            f"declared boundary_policy={declared.boundary_policy!r}"
        )
    if effective.exit_price_field != effective.entry_price_field:
        raise ValueError(
            f"Effective exit_price_field={effective.exit_price_field!r} disagrees with "
            f"effective entry_price_field={effective.entry_price_field!r}"
        )
    if not allow_entry_field_override and effective.exit_price_field != declared.exit_price_field:
        raise ValueError(
            f"Effective exit_price_field={effective.exit_price_field!r} disagrees with "
            f"declared exit_price_field={declared.exit_price_field!r}"
        )
    if effective.return_unit != declared.return_unit:
        raise ValueError(
            f"Effective return_unit={effective.return_unit!r} disagrees with "
            f"declared return_unit={declared.return_unit!r}"
        )
    if effective.gap_rule != declared.gap_rule:
        raise ValueError(
            f"Effective gap_rule={effective.gap_rule!r} disagrees with "
            f"declared gap_rule={declared.gap_rule!r}"
        )
    if not allow_entry_field_override and effective.entry_price_field != declared.entry_price_field:
        raise ValueError(
            f"Effective entry_price_field={effective.entry_price_field!r} disagrees with "
            f"declared entry_price_field={declared.entry_price_field!r}"
        )
