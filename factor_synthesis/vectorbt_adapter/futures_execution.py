"""Contract-level futures execution engine and nominal-mode roll guard.

Separates nominal-weight research simulation from contract-level futures
execution so continuous futures roots cannot silently accrue roll-gap PnL under
``execution_mode="nominal_weight"``, while ``execution_mode="futures_contract"``
models contract multipliers, integer lot rounding, margin/leverage constraints,
and close-old/open-new contract rolls on ``adjust='none'`` concrete prices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
from numbers import Real
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.slots.contracts import InstrumentSpec, MarketSlot
from utils.time_axis import Frequency

try:
    import vectorbt as vbt

    _VBT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when vectorbt is absent
    _VBT_AVAILABLE = False
    vbt = None


EXECUTION_MODES: tuple[str, ...] = ("nominal_weight", "futures_contract")
LOT_ROUNDING_MODES: tuple[str, ...] = ("floor", "nearest")
INSUFFICIENT_MARGIN_POLICIES: tuple[str, ...] = ("clip", "raise")
LIQUIDATION_POLICIES: tuple[str, ...] = ("liquidate", "record", "raise")
ROLL_SIZING_MODES: tuple[str, ...] = ("same_lots", "match_notional")
TARGET_WEIGHT_BASES: tuple[str, ...] = ("notional", "margin")
MISSING_EXECUTION_BAR_POLICIES: tuple[str, ...] = ("raise", "hold")
ROLL_CLOSE_FALLBACK_POLICIES: tuple[str, ...] = ("raise", "last_close")


class FuturesExecutionError(ValueError):
    """Base error for futures contract execution and nominal-mode guards."""


class ContinuousFuturesNominalModeError(FuturesExecutionError):
    """Raised when continuous rolling futures run under nominal_weight mode."""


class FuturesMarginError(FuturesExecutionError):
    """Raised when margin requirements or maintenance thresholds fail."""


def _require_positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return out


@dataclass(frozen=True)
class FuturesExecutionConfig:
    """Configuration for futures execution mode and margin/lot semantics."""

    execution_mode: Literal["nominal_weight", "futures_contract"] = "nominal_weight"
    margin_rate: float = 1.0
    margin_rates: Mapping[str, float] = field(default_factory=dict)
    maintenance_margin_rate: float | None = None
    lot_rounding: Literal["floor", "nearest"] = "floor"
    leverage: float = 1.0
    target_weight_basis: Literal["notional", "margin"] = "notional"
    insufficient_margin_policy: Literal["clip", "raise"] = "clip"
    liquidation_policy: Literal["liquidate", "record", "raise"] = "liquidate"
    roll_sizing: Literal["same_lots", "match_notional"] = "same_lots"
    missing_execution_bar_policy: Literal["raise", "hold"] = "raise"
    roll_close_fallback_policy: Literal["raise", "last_close"] = "raise"

    def __post_init__(self) -> None:
        if self.execution_mode not in EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {EXECUTION_MODES}, got {self.execution_mode!r}"
            )
        margin_rate = _require_positive_finite(self.margin_rate, "margin_rate")
        object.__setattr__(self, "margin_rate", margin_rate)

        if not isinstance(self.margin_rates, Mapping):
            raise ValueError("margin_rates must be a mapping")
        clean_rates: Dict[str, float] = {}
        for key, val in self.margin_rates.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("margin_rates keys must be non-empty strings")
            clean_rates[key] = _require_positive_finite(val, f"margin_rates[{key!r}]")
        object.__setattr__(self, "margin_rates", clean_rates)

        if self.maintenance_margin_rate is not None:
            maint = _require_positive_finite(
                self.maintenance_margin_rate, "maintenance_margin_rate"
            )
            min_initial = min([margin_rate, *clean_rates.values()])
            if maint > min_initial:
                raise ValueError(
                    f"maintenance_margin_rate ({maint}) must not exceed initial margin_rate ({min_initial})"
                )
            object.__setattr__(self, "maintenance_margin_rate", maint)

        if self.lot_rounding not in LOT_ROUNDING_MODES:
            raise ValueError(
                f"lot_rounding must be one of {LOT_ROUNDING_MODES}, got {self.lot_rounding!r}"
            )
        object.__setattr__(
            self, "leverage", _require_positive_finite(self.leverage, "leverage")
        )
        if self.target_weight_basis not in TARGET_WEIGHT_BASES:
            raise ValueError(
                f"target_weight_basis must be one of {TARGET_WEIGHT_BASES}, got {self.target_weight_basis!r}"
            )
        if self.insufficient_margin_policy not in INSUFFICIENT_MARGIN_POLICIES:
            raise ValueError(
                f"insufficient_margin_policy must be one of {INSUFFICIENT_MARGIN_POLICIES}, "
                f"got {self.insufficient_margin_policy!r}"
            )
        if self.liquidation_policy not in LIQUIDATION_POLICIES:
            raise ValueError(
                f"liquidation_policy must be one of {LIQUIDATION_POLICIES}, "
                f"got {self.liquidation_policy!r}"
            )
        if self.roll_sizing not in ROLL_SIZING_MODES:
            raise ValueError(
                f"roll_sizing must be one of {ROLL_SIZING_MODES}, got {self.roll_sizing!r}"
            )
        if self.missing_execution_bar_policy not in MISSING_EXECUTION_BAR_POLICIES:
            raise ValueError(
                f"missing_execution_bar_policy must be one of {MISSING_EXECUTION_BAR_POLICIES}, "
                f"got {self.missing_execution_bar_policy!r}"
            )
        if self.roll_close_fallback_policy not in ROLL_CLOSE_FALLBACK_POLICIES:
            raise ValueError(
                f"roll_close_fallback_policy must be one of {ROLL_CLOSE_FALLBACK_POLICIES}, "
                f"got {self.roll_close_fallback_policy!r}"
            )

    def margin_rate_for(self, instrument_id: str, concrete_contract: str | None = None) -> float:
        if concrete_contract is not None and concrete_contract in self.margin_rates:
            return float(self.margin_rates[concrete_contract])
        if instrument_id in self.margin_rates:
            return float(self.margin_rates[instrument_id])
        return float(self.margin_rate)


def parse_futures_execution_config(config: Mapping[str, Any] | None) -> FuturesExecutionConfig:
    """Parse ``FuturesExecutionConfig`` from runner config dicts."""
    cfg = config or {}
    top_k_cfg = cfg.get("top_k_strategy") or {}
    if not isinstance(top_k_cfg, Mapping):
        raise ValueError("top_k_strategy must be a mapping")
    top_fut = cfg.get("futures_execution") or {}
    if not isinstance(top_fut, Mapping):
        raise ValueError("futures_execution must be a mapping")
    sub_fut = top_k_cfg.get("futures_execution") or {}
    if not isinstance(sub_fut, Mapping):
        raise ValueError("top_k_strategy.futures_execution must be a mapping")

    merged: Dict[str, Any] = {}
    keys = (
        "execution_mode",
        "margin_rate",
        "margin_rates",
        "maintenance_margin_rate",
        "lot_rounding",
        "leverage",
        "target_weight_basis",
        "insufficient_margin_policy",
        "liquidation_policy",
        "roll_sizing",
        "missing_execution_bar_policy",
        "roll_close_fallback_policy",
    )
    for source in (top_fut, top_k_cfg, sub_fut):
        for key in keys:
            if key in source and source[key] is not None:
                merged[key] = source[key]
    # Explicit None for maintenance_margin_rate disables it even if inherited.
    for source in (top_fut, top_k_cfg, sub_fut):
        if "maintenance_margin_rate" in source and source["maintenance_margin_rate"] is None:
            merged["maintenance_margin_rate"] = None
    return FuturesExecutionConfig(**merged)


@dataclass(frozen=True)
class ContinuousFuturesDetectionResult:
    """Result of checking whether instruments undergo continuous futures rolls."""

    rolling_instruments: Dict[str, Tuple[str, ...]]
    bar_utc_index: pd.DatetimeIndex
    execution_bars: Optional[pd.DataFrame] = None


def _resolve_bar_utc_index(
    slot: MarketSlot,
    index: pd.DatetimeIndex,
    frequency: str | Frequency,
) -> Tuple[pd.DatetimeIndex, Optional[Dict[pd.Timestamp, pd.Timestamp]]]:
    """Map backtest bar index onto UTC bar timestamps and optional session->UTC lookup."""
    freq_obj = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
    idx = pd.DatetimeIndex(index)
    if idx.empty:
        return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC"), None
    daily_naive = bool(
        freq_obj.canonical == "1d"
        and idx.tz is None
        and idx.equals(idx.normalize())
    )
    if not daily_naive:
        utc_idx = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).as_unit("ns")
        return utc_idx, None

    grid = slot.calendar.bar_grid(
        freq_obj.canonical or "1d",
        idx.min().strftime("%Y-%m-%d"),
        idx.max().strftime("%Y-%m-%d"),
    )
    session_to_utc: Dict[pd.Timestamp, pd.Timestamp] = {}
    if not grid.empty:
        for row in grid.itertuples(index=False):
            session_to_utc[pd.Timestamp(row.session_id).normalize()] = (
                pd.Timestamp(row.timestamp_utc).tz_convert("UTC").as_unit("ns")
            )
    mapped: List[pd.Timestamp] = []
    for d in idx:
        norm = pd.Timestamp(d).normalize()
        ts = session_to_utc.get(norm)
        if ts is None:
            ts = norm.tz_localize("UTC").as_unit("ns")
        mapped.append(ts)
    return pd.DatetimeIndex(mapped, tz="UTC").as_unit("ns"), session_to_utc


def _try_resolve_active(slot: MarketSlot, instrument_id: str, as_of_utc: datetime) -> Optional[str]:
    try:
        return slot.universe.resolve_active_contract(instrument_id, as_of_utc=as_of_utc)
    except Exception:
        return None


def detect_continuous_futures_rolls(
    *,
    slot: Optional[MarketSlot],
    instruments: Sequence[str],
    index: pd.DatetimeIndex,
    frequency: str | Frequency,
) -> ContinuousFuturesDetectionResult:
    """Detect instruments that resolve to multiple concrete contracts across ``index``."""
    idx = pd.DatetimeIndex(index)
    if slot is None or not instruments or idx.empty:
        utc_empty = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return ContinuousFuturesDetectionResult({}, utc_empty, None)

    freq_obj = frequency if isinstance(frequency, Frequency) else Frequency.parse(frequency)
    bar_utc_index, _ = _resolve_bar_utc_index(slot, idx, freq_obj)

    sample_positions = sorted({0, len(bar_utc_index) // 4, len(bar_utc_index) // 2, (3 * len(bar_utc_index)) // 4, len(bar_utc_index) - 1})
    sample_times = [bar_utc_index[pos].to_pydatetime() for pos in sample_positions]

    needs_bar_inspection = slot.slot_id in ("cn_futures", "global_futures") or hasattr(
        slot.bar_feed, "root"
    )
    resolved_differs: Dict[str, List[str]] = {}
    for inst in instruments:
        seen: List[str] = []
        for ts in sample_times:
            c = _try_resolve_active(slot, inst, ts)
            if c is not None:
                if c != inst:
                    needs_bar_inspection = True
                if c not in seen:
                    seen.append(c)
        if seen:
            resolved_differs[inst] = seen

    if not needs_bar_inspection:
        # Avoid extra DuckDB/CSV bar reads for standard equity/spot slots.
        return ContinuousFuturesDetectionResult({}, bar_utc_index, None)

    exec_bars: Optional[pd.DataFrame] = None
    rolling: Dict[str, Tuple[str, ...]] = {}
    try:
        exec_bars = slot.bar_feed.read_execution_bars(
            list(instruments),
            freq_obj,
            start_utc=bar_utc_index.min().to_pydatetime(),
            end_utc=bar_utc_index.max().to_pydatetime(),
        )
    except Exception:
        exec_bars = None

    if (
        exec_bars is not None
        and not exec_bars.empty
        and "concrete_instrument_id" in exec_bars.columns
    ):
        ordered = exec_bars.sort_values(["timestamp_utc", "instrument_id"])
        for inst, grp in ordered.groupby("instrument_id", sort=False):
            contracts = tuple(dict.fromkeys(str(v) for v in grp["concrete_instrument_id"].dropna()))
            if len(contracts) > 1:
                rolling[str(inst)] = contracts

    # Fallback / supplement via resolve_active_contract across all bar timestamps
    for inst in instruments:
        if inst in rolling:
            continue
        sample_contracts = resolved_differs.get(inst, [])
        if any(c != inst for c in sample_contracts):
            contracts_all: List[str] = []
            for ts in bar_utc_index:
                c = _try_resolve_active(slot, inst, ts.to_pydatetime())
                if c is not None and c not in contracts_all:
                    contracts_all.append(c)
            if len(contracts_all) > 1:
                rolling[str(inst)] = tuple(contracts_all)

    return ContinuousFuturesDetectionResult(rolling, bar_utc_index, exec_bars)


def validate_nominal_execution_mode(
    *,
    execution_mode: str,
    slot: Optional[MarketSlot],
    instruments: Sequence[str],
    index: pd.DatetimeIndex,
    frequency: str | Frequency,
) -> None:
    """Fail fast when ``nominal_weight`` mode is used on rolling continuous futures."""
    if execution_mode != "nominal_weight":
        return
    detection = detect_continuous_futures_rolls(
        slot=slot,
        instruments=instruments,
        index=index,
        frequency=frequency,
    )
    if detection.rolling_instruments:
        detail = "; ".join(
            f"{inst} -> {list(contracts)}"
            for inst, contracts in sorted(detection.rolling_instruments.items())
        )
        raise ContinuousFuturesNominalModeError(
            "execution_mode='nominal_weight' cannot backtest rolling continuous futures "
            f"across contract rolls because spliced price jumps distort PnL ({detail}). "
            "Use execution_mode='futures_contract' for contract-level futures simulation "
            "or pass concrete non-rolling contracts."
        )


def round_contract_lots(
    target_notional: float,
    exec_price: float,
    contract_multiplier: float,
    lot_size: int = 1,
    lot_rounding: Literal["floor", "nearest"] = "floor",
) -> int:
    """Convert signed ``target_notional`` into integer lots aligned to ``lot_size``."""
    if (
        not math.isfinite(target_notional)
        or not math.isfinite(exec_price)
        or exec_price <= 0.0
        or not math.isfinite(contract_multiplier)
        or contract_multiplier <= 0.0
        or lot_size < 1
    ):
        return 0
    if abs(target_notional) <= 1e-12:
        return 0
    raw_units = target_notional / (exec_price * contract_multiplier)
    sign = 1 if raw_units >= 0.0 else -1
    abs_lot_units = abs(raw_units) / float(lot_size)
    if lot_rounding == "floor":
        n_lots = math.floor(abs_lot_units + 1e-12)
    elif lot_rounding == "nearest":
        n_lots = int(round(abs_lot_units))
    else:
        raise ValueError(f"Unsupported lot_rounding: {lot_rounding!r}")
    return int(sign * n_lots * lot_size)


@dataclass
class _OpenPosition:
    instrument_id: str
    concrete_instrument_id: str
    lots: int
    multiplier: float
    margin_rate: float
    avg_entry_price: float
    last_mark_price: float
    entry_cost: float = 0.0
    entry_timestamp: Any = None


class _FuturesOrdersAccessor:
    def __init__(self, records: pd.DataFrame) -> None:
        self.records_readable = records


class _FuturesTradesAccessor:
    def __init__(self, records: pd.DataFrame, total_count: int, win_rate: float) -> None:
        self.records_readable = records
        self._total_count = int(total_count)
        self._win_rate = float(win_rate)

    def count(self) -> int:
        return self._total_count

    def win_rate(self) -> float:
        return self._win_rate


class _FuturesLogsAccessor:
    def __init__(self, records: pd.DataFrame) -> None:
        self.records_readable = records


class _FuturesWrapper:
    def __init__(self, index: pd.DatetimeIndex, columns: pd.Index) -> None:
        self.index = index
        self.columns = columns


class FuturesPortfolio:
    """VectorBT-compatible Portfolio facade returned by contract-level futures simulation."""

    def __init__(
        self,
        *,
        equity: pd.Series,
        returns: pd.Series,
        asset_values: pd.DataFrame,
        orders: pd.DataFrame,
        trades: pd.DataFrame,
        trade_count: int,
        win_rate: float,
        logs: pd.DataFrame,
        init_cash: float,
        vbt_freq: str,
        year_freq: Union[str, pd.Timedelta],
        lots_history: pd.DataFrame,
        margin_used: pd.Series,
        margin_utilization: pd.Series,
        roll_events: List[Dict[str, Any]],
        liquidation_events: List[Dict[str, Any]],
        diagnostics: Dict[str, Any],
    ) -> None:
        self._equity = equity
        self._returns = returns
        self._asset_values = asset_values
        self._init_cash = float(init_cash)
        self._vbt_freq = vbt_freq
        self._default_year_freq = year_freq
        self.wrapper = _FuturesWrapper(
            pd.DatetimeIndex(equity.index), pd.Index(asset_values.columns)
        )
        self.orders = _FuturesOrdersAccessor(orders)
        self.trades = _FuturesTradesAccessor(trades, trade_count, win_rate)
        self.logs = _FuturesLogsAccessor(logs)
        self.lots_history = lots_history
        self.margin_used = margin_used
        self.margin_utilization = margin_utilization
        self.roll_events = roll_events
        self.liquidation_events = liquidation_events
        self.diagnostics = diagnostics

    def value(self) -> pd.Series:
        return self._equity.copy()

    def returns(self) -> pd.Series:
        return self._returns.copy()

    def asset_value(self, group_by: bool = False) -> Union[pd.DataFrame, pd.Series]:
        if group_by:
            return self._asset_values.sum(axis=1)
        return self._asset_values.copy()

    def _accessor(self, year_freq: Any = None) -> Any:
        if not _VBT_AVAILABLE:
            return None
        yf = self._default_year_freq if year_freq is None else year_freq
        return self._returns.vbt.returns(freq=self._vbt_freq, year_freq=yf)

    def _ann_factor(self, year_freq: Any = None) -> float:
        yf = self._default_year_freq if year_freq is None else year_freq
        if isinstance(yf, pd.Timedelta):
            bar_td = pd.to_timedelta(self._vbt_freq) if self._vbt_freq != "1M" else pd.Timedelta(days=30)
            return float(yf / bar_td) if bar_td.total_seconds() > 0 else 252.0
        if isinstance(yf, str) and yf.endswith("D"):
            return float(yf[:-1])
        if isinstance(yf, str) and yf.endswith("W"):
            return float(yf[:-1])
        return 252.0

    def total_return(self) -> float:
        if self._equity.empty or self._init_cash <= 0.0:
            return 0.0
        return float(self._equity.iloc[-1] / self._init_cash - 1.0)

    def annualized_return(self, year_freq: Any = None) -> float:
        acc = self._accessor(year_freq)
        if acc is not None:
            try:
                return float(acc.annualized())
            except Exception:
                pass
        n = max(len(self._returns), 1)
        ann = self._ann_factor(year_freq)
        tot = self.total_return()
        if 1.0 + tot <= 0.0:
            return -1.0
        return float((1.0 + tot) ** (ann / n) - 1.0)

    def sharpe_ratio(self, year_freq: Any = None) -> float:
        acc = self._accessor(year_freq)
        if acc is not None:
            try:
                return float(acc.sharpe_ratio())
            except Exception:
                pass
        std = float(self._returns.std(ddof=1)) if len(self._returns) > 1 else 0.0
        if std <= 0.0 or not math.isfinite(std):
            return float("nan")
        return float(self._returns.mean() / std * math.sqrt(self._ann_factor(year_freq)))

    def max_drawdown(self) -> float:
        acc = self._accessor()
        if acc is not None:
            try:
                return float(acc.max_drawdown())
            except Exception:
                pass
        if self._equity.empty:
            return 0.0
        peak = self._equity.cummax()
        dd = self._equity / peak.where(peak > 0, np.nan) - 1.0
        val = float(dd.min())
        return val if math.isfinite(val) else 0.0

    def calmar_ratio(self, year_freq: Any = None) -> float:
        acc = self._accessor(year_freq)
        if acc is not None:
            try:
                return float(acc.calmar_ratio())
            except Exception:
                pass
        mdd = abs(self.max_drawdown())
        if mdd <= 0.0:
            return float("nan")
        return float(self.annualized_return(year_freq) / mdd)

    def sortino_ratio(self, year_freq: Any = None) -> float:
        acc = self._accessor(year_freq)
        if acc is not None:
            try:
                return float(acc.sortino_ratio())
            except Exception:
                pass
        downside = np.minimum(self._returns.to_numpy(dtype=float), 0.0)
        dd_dev = float(np.sqrt(np.mean(downside ** 2)))
        if dd_dev <= 0.0:
            return float("nan")
        return float(self._returns.mean() / dd_dev * math.sqrt(self._ann_factor(year_freq)))

    def annual_returns(self) -> pd.Series:
        acc = self._accessor()
        if acc is not None:
            try:
                res = acc.annual()
                if isinstance(res, pd.Series) and not res.empty:
                    return res
            except Exception:
                pass
        if self._returns.empty:
            return pd.Series(dtype=float)
        years = pd.DatetimeIndex(self._returns.index).year
        grouped = (1.0 + self._returns).groupby(years).prod() - 1.0
        return grouped


def _valid_positive(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    val = float(value)
    return val if math.isfinite(val) and val > 0.0 else None


def _decode_runner_target_weights(
    *,
    target_weights: pd.DataFrame,
    price_aligned: pd.DataFrame,
    open_aligned: Optional[pd.DataFrame],
    use_open: bool,
    init_cash: float,
    buy_cost_rate: float,
    sell_cost_rate: float,
    slippage: float,
    allow_short: bool,
) -> pd.DataFrame:
    """Recover clean signed target weights from ``_compute_target_weights`` dollar matrix.

    Replays the nominal pre-sim NAV on the input ``price_aligned`` / ``open_aligned``
    so any spliced roll-price jump inside ``target_weights`` cancels out cleanly
    instead of inflating contract-level target notionals.
    """
    exec_frame = open_aligned if (use_open and open_aligned is not None) else price_aligned
    buy_mult = 1.0 + max(float(buy_cost_rate), 0.0) + max(float(slippage), 0.0)
    sell_mult = max(0.0, 1.0 - max(float(sell_cost_rate), 0.0) - max(float(slippage), 0.0))
    buy_fee_rate = max(float(buy_cost_rate), 0.0) + max(float(slippage), 0.0)
    sell_fee_rate = max(float(sell_cost_rate), 0.0) + max(float(slippage), 0.0)

    out = pd.DataFrame(
        np.nan,
        index=target_weights.index,
        columns=target_weights.columns,
        dtype=np.float64,
    )
    nom_cash = float(init_cash)
    nom_pos: Dict[str, float] = {}

    def _nom_mark(dt: Any, inst: str) -> Optional[float]:
        for frame in (exec_frame, price_aligned):
            if inst in frame.columns and dt in frame.index:
                px = _valid_positive(frame.loc[dt, inst])
                if px is not None:
                    return px
        if inst in price_aligned.columns:
            hist = pd.to_numeric(price_aligned.loc[:dt, inst], errors="coerce")
            hist = hist[np.isfinite(hist) & (hist > 0)]
            if not hist.empty:
                return float(hist.iloc[-1])
        return None

    def _nom_exec(dt: Any, inst: str) -> Optional[float]:
        if inst not in exec_frame.columns or dt not in exec_frame.index:
            return None
        return _valid_positive(exec_frame.loc[dt, inst])

    for dt in target_weights.index:
        row = target_weights.loc[dt]
        if not row.notna().any():
            continue

        # Leading flat rows before first rebalance have 0.0 and no open positions.
        if not nom_pos and bool((row.fillna(0.0).abs() <= 1e-12).all()):
            out.loc[dt, row.notna()] = 0.0
            continue

        nom_nav = nom_cash
        for s, shares in nom_pos.items():
            px = _nom_mark(dt, s)
            if px is not None:
                nom_nav += shares * px

        for inst in target_weights.columns:
            raw_val = row[inst]
            if pd.isna(raw_val):
                continue
            tgt_val = float(raw_val)
            if abs(tgt_val) <= 1e-12:
                out.loc[dt, inst] = 0.0
            elif nom_nav > 0.0:
                out.loc[dt, inst] = tgt_val * buy_mult / nom_nav
            else:
                out.loc[dt, inst] = 0.0

            px = _nom_exec(dt, inst)
            if px is None:
                continue
            old_shares = nom_pos.get(inst, 0.0)
            if allow_short:
                cur_val = old_shares * px
                delta = tgt_val - cur_val
                if abs(delta) > 1e-12:
                    fee_r = buy_fee_rate if delta > 0.0 else sell_fee_rate
                    nom_cash -= delta + abs(delta) * fee_r
                if abs(tgt_val) > 1e-12:
                    nom_pos[inst] = tgt_val / px
                else:
                    nom_pos.pop(inst, None)
            else:
                if abs(tgt_val) <= 1e-12 and old_shares > 0.0:
                    nom_cash += old_shares * px * sell_mult
                    nom_pos.pop(inst, None)
                elif tgt_val > 1e-12:
                    cur_val = old_shares * px
                    if old_shares > 0.0:
                        delta = tgt_val - cur_val
                        if delta > 0.0:
                            nom_cash -= delta * buy_mult
                        elif delta < 0.0:
                            nom_cash += (-delta) * sell_mult
                    else:
                        nom_cash -= tgt_val * buy_mult
                    nom_pos[inst] = tgt_val / px
    return out


def simulate_futures_contract_portfolio(
    *,
    target_weights: pd.DataFrame,
    price_aligned: pd.DataFrame,
    open_aligned: Optional[pd.DataFrame],
    use_open: bool,
    slot: Optional[MarketSlot],
    frequency: str,
    init_cash: float,
    buy_cost_rate: float,
    sell_cost_rate: float,
    slippage: float,
    config: FuturesExecutionConfig,
    allow_short: bool = True,
    target_value_mode: Literal["runner_target_value", "target_weight", "target_notional"] = "runner_target_value",
    instrument_specs: Optional[Mapping[str, InstrumentSpec]] = None,
    vbt_freq: str = "1D",
    year_freq: Union[str, pd.Timedelta] = "252D",
    logger: Optional[logging.Logger] = None,
) -> Tuple[FuturesPortfolio, Dict[str, Any]]:
    """Run contract-level futures portfolio simulation with roll close-old/open-new."""
    log = logger or logging.getLogger(__name__)
    all_dates = pd.DatetimeIndex(price_aligned.index)
    instruments = [str(c) for c in price_aligned.columns]
    freq_obj = Frequency.parse(frequency)
    if not freq_obj.is_valid or freq_obj.canonical is None:
        raise ValueError(f"Invalid frequency: {frequency!r}")

    if all_dates.empty or not instruments:
        raise ValueError("Futures contract simulation requires non-empty prices and instruments")

    # 1. Decode signed target weights / notionals.
    aligned_targets = target_weights.reindex(index=all_dates, columns=instruments)
    if target_value_mode == "runner_target_value":
        signed_weights = _decode_runner_target_weights(
            target_weights=aligned_targets,
            price_aligned=price_aligned,
            open_aligned=open_aligned,
            use_open=use_open,
            init_cash=init_cash,
            buy_cost_rate=buy_cost_rate,
            sell_cost_rate=sell_cost_rate,
            slippage=slippage,
            allow_short=allow_short,
        )
    elif target_value_mode in ("target_weight", "target_notional"):
        signed_weights = aligned_targets.astype(float)
    else:
        raise ValueError(f"Unsupported target_value_mode: {target_value_mode!r}")

    # 2. Build UTC bar mapping and load physical adjust='none' execution bars.
    if slot is not None:
        bar_utc_index, session_to_utc = _resolve_bar_utc_index(slot, all_dates, freq_obj)
        exec_bars = slot.bar_feed.read_execution_bars(
            instruments,
            freq_obj,
            start_utc=bar_utc_index.min().to_pydatetime(),
            end_utc=bar_utc_index.max().to_pydatetime(),
        )
    else:
        bar_utc_index = (
            all_dates.tz_localize("UTC") if all_dates.tz is None else all_dates.tz_convert("UTC")
        ).as_unit("ns")
        session_to_utc = None
        exec_bars = pd.DataFrame()

    # Map (bar_pos, inst) -> active concrete contract & physical open/close prices.
    active_contract: Dict[Tuple[int, str], str] = {}
    concrete_open: Dict[Tuple[int, str], Optional[float]] = {}
    concrete_close: Dict[Tuple[int, str], Optional[float]] = {}
    last_contract_close: Dict[str, float] = {}

    utc_to_pos = {ts: i for i, ts in enumerate(bar_utc_index)}
    bar_row_lookup: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if exec_bars is not None and not exec_bars.empty:
        eb = exec_bars.copy()
        if session_to_utc is not None and "session_id" in eb.columns and eb["session_id"].notna().all():
            eb["_utc"] = pd.to_datetime(eb["session_id"]).dt.normalize().map(session_to_utc)
        else:
            eb["_utc"] = pd.to_datetime(eb["timestamp_utc"], utc=True).dt.as_unit("ns")
        for rec in eb.to_dict(orient="records"):
            pos = utc_to_pos.get(rec.get("_utc"))
            inst_key = str(rec.get("instrument_id", ""))
            if pos is not None and inst_key:
                bar_row_lookup[(pos, inst_key)] = rec

    for pos, bar_ts in enumerate(bar_utc_index):
        dt = all_dates[pos]
        bar_py_dt = bar_ts.to_pydatetime()
        for inst in instruments:
            rec = bar_row_lookup.get((pos, inst))
            c_id: Optional[str] = None
            if rec is not None and pd.notna(rec.get("concrete_instrument_id")):
                c_id = str(rec["concrete_instrument_id"])
            elif slot is not None:
                c_id = _try_resolve_active(slot, inst, bar_py_dt)
            if not c_id:
                c_id = active_contract[(pos - 1, inst)] if pos > 0 else inst
            active_contract[(pos, inst)] = c_id

            if slot is not None:
                # Never fall back to spliced/adjusted price_aligned or open_aligned when a slot feed is present.
                c_px = _valid_positive(rec.get("close")) if rec is not None else None
                o_px = _valid_positive(rec.get("open")) if rec is not None else None
                concrete_close[(pos, inst)] = c_px
                concrete_open[(pos, inst)] = o_px
            else:
                c_px = (
                    _valid_positive(price_aligned.loc[dt, inst])
                    if inst in price_aligned.columns
                    else None
                )
                o_px = (
                    _valid_positive(open_aligned.loc[dt, inst])
                    if (open_aligned is not None and inst in open_aligned.columns)
                    else None
                )
                concrete_close[(pos, inst)] = c_px
                concrete_open[(pos, inst)] = o_px if o_px is not None else c_px

    spec_cache: Dict[str, InstrumentSpec] = {}

    def _get_spec(contract_id: str) -> InstrumentSpec:
        if contract_id in spec_cache:
            return spec_cache[contract_id]
        if instrument_specs and contract_id in instrument_specs:
            spec = instrument_specs[contract_id]
        elif slot is not None:
            spec = slot.universe.instrument_spec(contract_id)
        else:
            spec = InstrumentSpec(contract_id, "futures", "CNY")
        spec_cache[contract_id] = spec
        return spec

    roll_close_cache: Dict[Tuple[int, str], Optional[float]] = {}
    roll_close_errors: Dict[Tuple[int, str], Exception] = {}

    def _resolve_roll_close_price(pos: int, contract_id: str) -> Optional[float]:
        key = (pos, contract_id)
        if key in roll_close_cache:
            return roll_close_cache[key]
        price_val: Optional[float] = None
        if slot is not None:
            bar_dt = bar_utc_index[pos].to_pydatetime()
            try:
                old_bar = slot.bar_feed.read_execution_bars(
                    [contract_id],
                    freq_obj,
                    start_utc=bar_dt,
                    end_utc=bar_dt,
                )
                if old_bar is not None and not old_bar.empty:
                    if (
                        session_to_utc is not None
                        and "session_id" in old_bar.columns
                        and old_bar["session_id"].notna().all()
                    ):
                        matched_utc = (
                            pd.to_datetime(old_bar["session_id"]).dt.normalize().map(session_to_utc)
                        )
                        old_bar = old_bar.loc[matched_utc == bar_utc_index[pos]]
                    elif "timestamp_utc" in old_bar.columns:
                        matched_utc = pd.to_datetime(old_bar["timestamp_utc"], utc=True).dt.as_unit("ns")
                        old_bar = old_bar.loc[matched_utc == bar_utc_index[pos]]
                if old_bar is not None and not old_bar.empty:
                    field = "open" if use_open else "close"
                    if field in old_bar.columns:
                        price_val = _valid_positive(old_bar[field].iloc[-1])
            except Exception as exc:
                roll_close_errors[key] = exc
                log.debug(
                    f"[FuturesExec] Could not read roll exit bar for {contract_id} at {bar_dt}: {exc}"
                )
        roll_close_cache[key] = price_val
        return price_val

    # 3. Simulation state & output buffers.
    equity = float(init_cash)
    positions: Dict[str, _OpenPosition] = {}

    equity_vals = np.zeros(len(all_dates), dtype=float)
    returns_vals = np.zeros(len(all_dates), dtype=float)
    margin_used_vals = np.zeros(len(all_dates), dtype=float)
    margin_util_vals = np.zeros(len(all_dates), dtype=float)
    asset_val_df = pd.DataFrame(0.0, index=all_dates, columns=instruments, dtype=float)
    lots_df = pd.DataFrame(0, index=all_dates, columns=instruments, dtype=int)

    order_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    log_rows: List[Dict[str, Any]] = []
    roll_events: List[Dict[str, Any]] = []
    liquidation_events: List[Dict[str, Any]] = []

    order_counter = 0
    margin_clipped_events = 0
    lot_floored_to_zero_events = 0
    missing_execution_bar_count = 0
    roll_close_fallback_count = 0
    total_roll_cost = 0.0
    total_transaction_cost = 0.0

    def _exec_price(pos: int, inst: str) -> Optional[float]:
        return concrete_open[(pos, inst)] if use_open else concrete_close[(pos, inst)]

    def _record_order(
        *,
        dt: Any,
        inst: str,
        contract: str,
        delta_lots: int,
        multiplier: float,
        price: float,
        fee_cost: float,
        order_type: str,
        resulting_lots: int,
    ) -> None:
        nonlocal order_counter, total_transaction_cost
        side = "Buy" if delta_lots > 0 else "Sell"
        abs_lots = abs(int(delta_lots))
        units = float(abs_lots) * float(multiplier)
        total_transaction_cost += float(fee_cost)
        order_rows.append(
            {
                "Order Id": order_counter,
                "Column": inst,
                "Timestamp": dt,
                "Side": side,
                "Size": units,
                "Price": float(price),
                "Fees": float(fee_cost),
                "Lots": abs_lots,
                "Signed Lots": int(delta_lots),
                "Contract Multiplier": float(multiplier),
                "Concrete Instrument": contract,
                "Order Type": order_type,
            }
        )
        log_rows.append(
            {
                "Timestamp": dt,
                "Column": inst,
                "Result Status": "Filled",
                "Request Size": float(delta_lots) * float(multiplier),
                "Position": float(resulting_lots) * float(multiplier),
                "Val Price": float(price),
                "Reason": order_type,
            }
        )
        order_counter += 1

    def _close_open_position(
        *,
        dt: Any,
        pos_obj: _OpenPosition,
        close_lots: int,
        exit_price: float,
        order_type: str,
    ) -> float:
        """Close ``close_lots`` (signed opposite or full) of ``pos_obj`` and update equity."""
        nonlocal equity
        abs_close = abs(int(close_lots))
        abs_held = abs(int(pos_obj.lots))
        assert 0 < abs_close <= abs_held
        notional = float(abs_close) * exit_price * pos_obj.multiplier
        # Closing a long (lots > 0) sells; closing a short (lots < 0) buys.
        fee_rate = (
            max(float(sell_cost_rate), 0.0)
            if pos_obj.lots > 0
            else max(float(buy_cost_rate), 0.0)
        ) + max(float(slippage), 0.0)
        close_cost = notional * fee_rate
        equity -= close_cost

        frac = float(abs_close) / float(abs_held)
        alloc_entry_cost = pos_obj.entry_cost * frac
        sign_held = 1 if pos_obj.lots > 0 else -1
        gross_pnl = float(sign_held * abs_close) * (exit_price - pos_obj.avg_entry_price) * pos_obj.multiplier
        net_pnl = gross_pnl - alloc_entry_cost - close_cost

        delta_lots = -sign_held * abs_close
        remaining_lots = pos_obj.lots + delta_lots
        _record_order(
            dt=dt,
            inst=pos_obj.instrument_id,
            contract=pos_obj.concrete_instrument_id,
            delta_lots=delta_lots,
            multiplier=pos_obj.multiplier,
            price=exit_price,
            fee_cost=close_cost,
            order_type=order_type,
            resulting_lots=remaining_lots,
        )
        trade_rows.append(
            {
                "Column": pos_obj.instrument_id,
                "Concrete Instrument": pos_obj.concrete_instrument_id,
                "Entry Timestamp": pos_obj.entry_timestamp,
                "Exit Timestamp": dt,
                "Direction": "Long" if pos_obj.lots > 0 else "Short",
                "Lots": abs_close,
                "Contract Multiplier": pos_obj.multiplier,
                "Entry Price": pos_obj.avg_entry_price,
                "Exit Price": exit_price,
                "PnL": net_pnl,
                "Fees": alloc_entry_cost + close_cost,
                "Status": "Closed",
                "Reason": order_type,
            }
        )
        pos_obj.lots = remaining_lots
        pos_obj.entry_cost -= alloc_entry_cost
        return close_cost

    # 4. Main bar-by-bar execution loop.
    prev_equity = float(init_cash)
    for bar_idx, dt in enumerate(all_dates):
        w_row = signed_weights.loc[dt]
        missing_bar_insts: set[str] = set()
        if slot is not None:
            for inst in instruments:
                exec_px = _exec_price(bar_idx, inst)
                close_px = concrete_close[(bar_idx, inst)]
                if exec_px is not None and close_px is not None:
                    continue
                has_pos = inst in positions
                raw_w = w_row[inst]
                has_target = pd.notna(raw_w) and (abs(float(raw_w)) > 1e-12 or has_pos)
                has_signal_bar = (
                    inst in price_aligned.columns
                    and _valid_positive(price_aligned.loc[dt, inst]) is not None
                )
                if not (has_pos or has_target or has_signal_bar):
                    continue
                missing_execution_bar_count += 1
                c_id = active_contract[(bar_idx, inst)]
                if config.missing_execution_bar_policy == "raise":
                    raise FuturesExecutionError(
                        f"Missing physical execution bar for instrument={inst!r}, "
                        f"contract={c_id!r}, bar={dt} (use_open={use_open})"
                    )
                missing_bar_insts.add(inst)

        # Step 4.1: Mark open positions to bar execution prices (or roll exit prices).
        roll_exit_prices: Dict[str, Tuple[float, bool]] = {}
        for inst in sorted(positions):
            pos_obj = positions[inst]
            c_cur = active_contract[(bar_idx, inst)]
            if pos_obj.concrete_instrument_id == c_cur:
                if inst in missing_bar_insts:
                    continue
                px = _exec_price(bar_idx, inst)
                if px is not None:
                    equity += float(pos_obj.lots) * (px - pos_obj.last_mark_price) * pos_obj.multiplier
                    pos_obj.last_mark_price = px
            else:
                roll_px = _resolve_roll_close_price(bar_idx, pos_obj.concrete_instrument_id)
                used_fallback = False
                if roll_px is None:
                    if config.roll_close_fallback_policy == "raise":
                        err = roll_close_errors.get((bar_idx, pos_obj.concrete_instrument_id))
                        msg = (
                            f"Missing physical roll exit bar for instrument={inst!r}, "
                            f"from_contract={pos_obj.concrete_instrument_id!r}, "
                            f"to_contract={c_cur!r}, bar={dt}"
                        )
                        if err is not None:
                            raise FuturesExecutionError(msg) from err
                        raise FuturesExecutionError(msg)
                    roll_px = last_contract_close.get(
                        pos_obj.concrete_instrument_id, pos_obj.last_mark_price
                    )
                    used_fallback = True
                    roll_close_fallback_count += 1
                roll_exit_prices[inst] = (roll_px, used_fallback)
                equity += float(pos_obj.lots) * (roll_px - pos_obj.last_mark_price) * pos_obj.multiplier
                pos_obj.last_mark_price = roll_px

        # Step 4.2: Execute rolls (close C_old first so jump gaps never pollute PnL).
        rolled_from: Dict[str, Dict[str, Any]] = {}
        for inst in sorted(list(positions.keys())):
            pos_obj = positions[inst]
            c_new = active_contract[(bar_idx, inst)]
            if pos_obj.concrete_instrument_id != c_new:
                c_old = pos_obj.concrete_instrument_id
                old_lots = pos_obj.lots
                old_mult = pos_obj.multiplier
                p_old_exec, used_fallback = roll_exit_prices[inst]
                close_cost = _close_open_position(
                    dt=dt,
                    pos_obj=pos_obj,
                    close_lots=old_lots,
                    exit_price=p_old_exec,
                    order_type="ROLL_CLOSE",
                )
                total_roll_cost += close_cost
                del positions[inst]
                roll_rec: Dict[str, Any] = {
                    "timestamp": dt,
                    "instrument_id": inst,
                    "from_contract": c_old,
                    "to_contract": c_new,
                    "closed_lots": int(old_lots),
                    "close_price": float(p_old_exec),
                    "close_cost": float(close_cost),
                    "close_price_fallback": bool(used_fallback),
                    "opened_lots": 0,
                    "open_price": None,
                    "open_cost": 0.0,
                    "price_gap": None,
                }
                rolled_from[inst] = roll_rec
                roll_events.append(roll_rec)

        # Step 4.3: Compute desired target lots for each instrument on this bar.
        desired_lots: Dict[str, int] = {}
        for inst in instruments:
            if inst in missing_bar_insts:
                desired_lots[inst] = positions[inst].lots if inst in positions else 0
                continue
            raw_w = w_row[inst]
            if pd.notna(raw_w):
                w = float(raw_w)
                if not allow_short and w < 0.0:
                    w = 0.0
                if abs(w) <= 1e-12:
                    desired_lots[inst] = 0
                    continue
                c_cur = active_contract[(bar_idx, inst)]
                px = _exec_price(bar_idx, inst)
                if px is None:
                    desired_lots[inst] = positions[inst].lots if inst in positions else 0
                    continue
                spec = _get_spec(c_cur)
                mr = config.margin_rate_for(inst, c_cur)
                if target_value_mode == "target_notional":
                    target_notional = w * config.leverage
                elif config.target_weight_basis == "margin":
                    target_notional = (w * max(equity, 0.0) / mr) * config.leverage
                else:
                    target_notional = w * max(equity, 0.0) * config.leverage
                lots = round_contract_lots(
                    target_notional,
                    px,
                    spec.contract_multiplier,
                    spec.lot_size,
                    config.lot_rounding,
                )
                if lots == 0 and abs(target_notional) > 1e-12:
                    lot_floored_to_zero_events += 1
                desired_lots[inst] = lots
            else:
                if inst in rolled_from:
                    c_new = active_contract[(bar_idx, inst)]
                    px = _exec_price(bar_idx, inst)
                    if px is None:
                        desired_lots[inst] = 0
                    else:
                        spec = _get_spec(c_new)
                        roll_rec = rolled_from[inst]
                        old_lots = int(roll_rec["closed_lots"])
                        if config.roll_sizing == "same_lots":
                            sign = 1 if old_lots >= 0 else -1
                            aligned_lots = (abs(old_lots) // spec.lot_size) * spec.lot_size
                            desired_lots[inst] = sign * aligned_lots
                        else:
                            old_notional = (
                                float(old_lots)
                                * float(roll_rec["close_price"])
                                * _get_spec(roll_rec["from_contract"]).contract_multiplier
                            )
                            desired_lots[inst] = round_contract_lots(
                                old_notional,
                                px,
                                spec.contract_multiplier,
                                spec.lot_size,
                                config.lot_rounding,
                            )
                elif inst in positions:
                    desired_lots[inst] = positions[inst].lots
                else:
                    desired_lots[inst] = 0

        # Step 4.4: Execute reductions / closes / sign-flips first (frees margin).
        for inst in instruments:
            if inst not in positions:
                continue
            pos_obj = positions[inst]
            tgt = desired_lots[inst]
            cur = pos_obj.lots
            if tgt == cur:
                continue
            px = _exec_price(bar_idx, inst)
            if px is None:
                continue
            if tgt == 0 or (tgt * cur < 0):
                _close_open_position(
                    dt=dt,
                    pos_obj=pos_obj,
                    close_lots=cur,
                    exit_price=px,
                    order_type="REBALANCE",
                )
                del positions[inst]
            elif abs(tgt) < abs(cur):
                _close_open_position(
                    dt=dt,
                    pos_obj=pos_obj,
                    close_lots=abs(cur) - abs(tgt),
                    exit_price=px,
                    order_type="REBALANCE",
                )

        # Step 4.5: Margin validation & clipping for new openings / position increases.
        used_margin = sum(
            abs(p.lots) * p.last_mark_price * p.multiplier * p.margin_rate
            for p in positions.values()
        )
        increases: List[Dict[str, Any]] = []
        for inst in instruments:
            cur_lots = positions[inst].lots if inst in positions else 0
            tgt_lots = desired_lots[inst]
            if abs(tgt_lots) <= abs(cur_lots):
                continue
            c_cur = active_contract[(bar_idx, inst)]
            px = _exec_price(bar_idx, inst)
            if px is None:
                continue
            spec = _get_spec(c_cur)
            mr = config.margin_rate_for(inst, c_cur)
            delta_lots = tgt_lots - cur_lots
            fee_rate = (
                max(float(buy_cost_rate), 0.0)
                if delta_lots > 0
                else max(float(sell_cost_rate), 0.0)
            ) + max(float(slippage), 0.0)
            increases.append(
                {
                    "inst": inst,
                    "contract": c_cur,
                    "cur_lots": cur_lots,
                    "tgt_lots": tgt_lots,
                    "px": px,
                    "mult": float(spec.contract_multiplier),
                    "lot_size": int(spec.lot_size),
                    "mr": mr,
                    "fee_rate": fee_rate,
                }
            )

        if increases:
            def _totals(items: List[Dict[str, Any]]) -> Tuple[float, float]:
                add_margin = 0.0
                add_cost = 0.0
                for item in items:
                    added_abs = abs(item["tgt_lots"]) - abs(item["cur_lots"])
                    if added_abs <= 0:
                        continue
                    unit_notional = item["px"] * item["mult"]
                    add_margin += float(added_abs) * unit_notional * item["mr"]
                    add_cost += float(added_abs) * unit_notional * item["fee_rate"]
                return add_margin, add_cost

            add_margin, add_cost = _totals(increases)
            total_req_margin = used_margin + add_margin
            avail_equity = equity - add_cost
            if total_req_margin > avail_equity + 1e-9:
                if config.insufficient_margin_policy == "raise":
                    raise FuturesMarginError(
                        f"Insufficient margin at {dt}: required={total_req_margin:.2f}, "
                        f"available_after_costs={avail_equity:.2f}"
                    )
                margin_clipped_events += 1
                free_budget = max(0.0, equity - used_margin)
                denom = max(add_margin + add_cost, 1e-12)
                scale = free_budget / denom
                for item in increases:
                    sign = 1 if item["tgt_lots"] >= 0 else -1
                    cur_abs = abs(item["cur_lots"])
                    added_abs = abs(item["tgt_lots"]) - cur_abs
                    scaled_added = (
                        math.floor((added_abs * scale) / item["lot_size"] + 1e-12)
                        * item["lot_size"]
                    )
                    item["tgt_lots"] = sign * (cur_abs + max(0, scaled_added))

                # Trim one lot_size at a time if discrete fees/rounding still overshoot.
                while True:
                    add_margin, add_cost = _totals(increases)
                    if used_margin + add_margin <= (equity - add_cost) + 1e-9:
                        break
                    reducible = [
                        item
                        for item in increases
                        if abs(item["tgt_lots"]) - abs(item["cur_lots"]) >= item["lot_size"]
                    ]
                    if not reducible:
                        for item in increases:
                            item["tgt_lots"] = item["cur_lots"]
                        break
                    # Trim the largest marginal margin consumer first.
                    victim = max(
                        reducible,
                        key=lambda it: it["px"] * it["mult"] * it["mr"] * it["lot_size"],
                    )
                    sign = 1 if victim["tgt_lots"] >= 0 else -1
                    new_abs = max(
                        abs(victim["cur_lots"]),
                        abs(victim["tgt_lots"]) - victim["lot_size"],
                    )
                    victim["tgt_lots"] = sign * new_abs

            # Step 4.6: Execute accepted openings / increases.
            for item in increases:
                inst = item["inst"]
                tgt_lots = int(item["tgt_lots"])
                cur_lots = int(item["cur_lots"])
                delta_lots = tgt_lots - cur_lots
                if delta_lots == 0:
                    continue
                px = float(item["px"])
                mult = float(item["mult"])
                mr = float(item["mr"])
                cost = float(abs(delta_lots)) * px * mult * float(item["fee_rate"])
                equity -= cost
                order_type = "ROLL_OPEN" if inst in rolled_from else "REBALANCE"
                if inst in rolled_from:
                    total_roll_cost += cost
                    roll_rec = rolled_from[inst]
                    roll_rec["opened_lots"] = tgt_lots
                    roll_rec["open_price"] = px
                    roll_rec["open_cost"] = float(cost)
                    roll_rec["price_gap"] = float(px - float(roll_rec["close_price"]))

                _record_order(
                    dt=dt,
                    inst=inst,
                    contract=item["contract"],
                    delta_lots=delta_lots,
                    multiplier=mult,
                    price=px,
                    fee_cost=cost,
                    order_type=order_type,
                    resulting_lots=tgt_lots,
                )
                if inst in positions:
                    pos_obj = positions[inst]
                    old_abs = abs(pos_obj.lots)
                    new_abs = abs(tgt_lots)
                    pos_obj.avg_entry_price = (
                        pos_obj.avg_entry_price * old_abs + px * abs(delta_lots)
                    ) / float(new_abs)
                    pos_obj.lots = tgt_lots
                    pos_obj.entry_cost += cost
                    pos_obj.last_mark_price = px
                else:
                    positions[inst] = _OpenPosition(
                        instrument_id=inst,
                        concrete_instrument_id=item["contract"],
                        lots=tgt_lots,
                        multiplier=mult,
                        margin_rate=mr,
                        avg_entry_price=px,
                        last_mark_price=px,
                        entry_cost=cost,
                        entry_timestamp=dt,
                    )

        # Step 4.7: Mark open positions to bar close price & check maintenance margin.
        bar_margin_used = 0.0
        for inst in sorted(positions):
            pos_obj = positions[inst]
            close_px = None if inst in missing_bar_insts else concrete_close[(bar_idx, inst)]
            if close_px is not None:
                equity += float(pos_obj.lots) * (close_px - pos_obj.last_mark_price) * pos_obj.multiplier
                pos_obj.last_mark_price = close_px
                last_contract_close[pos_obj.concrete_instrument_id] = close_px
            pos_notional = float(pos_obj.lots) * pos_obj.last_mark_price * pos_obj.multiplier
            asset_val_df.loc[dt, inst] = pos_notional
            lots_df.loc[dt, inst] = int(pos_obj.lots)
            bar_margin_used += abs(pos_notional) * pos_obj.margin_rate

        # Track close prices even when flat (for roll exit fallback if needed).
        for inst in instruments:
            c_cur = active_contract[(bar_idx, inst)]
            c_px = concrete_close[(bar_idx, inst)]
            if c_px is not None:
                last_contract_close[c_cur] = c_px

        if config.maintenance_margin_rate is not None and positions:
            maint_req = sum(
                abs(p.lots) * p.last_mark_price * p.multiplier * float(config.maintenance_margin_rate)
                for p in positions.values()
            )
            if equity < maint_req:
                event: Dict[str, Any] = {
                    "timestamp": dt,
                    "equity_before_action": float(equity),
                    "maintenance_margin_required": float(maint_req),
                    "action": config.liquidation_policy,
                    "positions": {
                        inst: {
                            "contract": p.concrete_instrument_id,
                            "lots": int(p.lots),
                            "mark_price": float(p.last_mark_price),
                        }
                        for inst, p in sorted(positions.items())
                    },
                }
                if config.liquidation_policy == "raise":
                    raise FuturesMarginError(
                        f"Maintenance margin breached at {dt}: equity={equity:.2f} < "
                        f"maintenance_required={maint_req:.2f}"
                    )
                if config.liquidation_policy == "liquidate":
                    for inst in sorted(list(positions.keys())):
                        pos_obj = positions[inst]
                        _close_open_position(
                            dt=dt,
                            pos_obj=pos_obj,
                            close_lots=pos_obj.lots,
                            exit_price=pos_obj.last_mark_price,
                            order_type="LIQUIDATION",
                        )
                        del positions[inst]
                        asset_val_df.loc[dt, inst] = 0.0
                        lots_df.loc[dt, inst] = 0
                    bar_margin_used = 0.0
                event["equity_after_action"] = float(equity)
                liquidation_events.append(event)

        equity_vals[bar_idx] = equity
        if prev_equity > 0.0:
            returns_vals[bar_idx] = equity / prev_equity - 1.0
        else:
            returns_vals[bar_idx] = 0.0
        prev_equity = equity
        margin_used_vals[bar_idx] = bar_margin_used
        margin_util_vals[bar_idx] = bar_margin_used / equity if equity > 0.0 else float("nan")

    # 5. Include any still-open positions in trade records (with Status="Open").
    for inst in sorted(positions):
        pos_obj = positions[inst]
        unrealized = (
            float(pos_obj.lots) * (pos_obj.last_mark_price - pos_obj.avg_entry_price) * pos_obj.multiplier
            - pos_obj.entry_cost
        )
        trade_rows.append(
            {
                "Column": pos_obj.instrument_id,
                "Concrete Instrument": pos_obj.concrete_instrument_id,
                "Entry Timestamp": pos_obj.entry_timestamp,
                "Exit Timestamp": all_dates[-1],
                "Direction": "Long" if pos_obj.lots > 0 else "Short",
                "Lots": abs(int(pos_obj.lots)),
                "Contract Multiplier": pos_obj.multiplier,
                "Entry Price": pos_obj.avg_entry_price,
                "Exit Price": pos_obj.last_mark_price,
                "PnL": unrealized,
                "Fees": pos_obj.entry_cost,
                "Status": "Open",
                "Reason": "MARK_TO_MARKET",
            }
        )

    equity_series = pd.Series(equity_vals, index=all_dates, name="equity", dtype=float)
    returns_series = pd.Series(returns_vals, index=all_dates, name="returns", dtype=float)
    margin_used_series = pd.Series(margin_used_vals, index=all_dates, name="margin_used", dtype=float)
    margin_util_series = pd.Series(margin_util_vals, index=all_dates, name="margin_utilization", dtype=float)

    orders_columns = [
        "Order Id",
        "Column",
        "Timestamp",
        "Side",
        "Size",
        "Price",
        "Fees",
        "Lots",
        "Signed Lots",
        "Contract Multiplier",
        "Concrete Instrument",
        "Order Type",
    ]
    orders_df = pd.DataFrame(order_rows, columns=orders_columns)
    trades_df = pd.DataFrame(
        trade_rows,
        columns=[
            "Column",
            "Concrete Instrument",
            "Entry Timestamp",
            "Exit Timestamp",
            "Direction",
            "Lots",
            "Contract Multiplier",
            "Entry Price",
            "Exit Price",
            "PnL",
            "Fees",
            "Status",
            "Reason",
        ],
    )
    logs_df = pd.DataFrame(
        log_rows,
        columns=[
            "Timestamp",
            "Column",
            "Result Status",
            "Request Size",
            "Position",
            "Val Price",
            "Reason",
        ],
    )

    closed_df = trades_df.loc[trades_df["Status"] == "Closed"] if not trades_df.empty else trades_df
    win_rate = float((closed_df["PnL"] > 0.0).mean()) if not closed_df.empty else 0.0
    trade_count = int(len(trades_df))

    finite_util = margin_util_vals[np.isfinite(margin_util_vals)]
    contract_multipliers = {
        contract: float(spec.contract_multiplier) for contract, spec in sorted(spec_cache.items())
    }
    lot_sizes = {contract: int(spec.lot_size) for contract, spec in sorted(spec_cache.items())}

    diagnostics: Dict[str, Any] = {
        "execution_mode": "futures_contract",
        "margin_rate": float(config.margin_rate),
        "margin_rates": dict(config.margin_rates),
        "maintenance_margin_rate": config.maintenance_margin_rate,
        "lot_rounding": config.lot_rounding,
        "leverage": float(config.leverage),
        "target_weight_basis": config.target_weight_basis,
        "insufficient_margin_policy": config.insufficient_margin_policy,
        "liquidation_policy": config.liquidation_policy,
        "roll_sizing": config.roll_sizing,
        "missing_execution_bar_policy": config.missing_execution_bar_policy,
        "roll_close_fallback_policy": config.roll_close_fallback_policy,
        "contract_multipliers": contract_multipliers,
        "lot_sizes": lot_sizes,
        "roll_count": len(roll_events),
        "roll_cost_total": float(total_roll_cost),
        "missing_execution_bar_count": int(missing_execution_bar_count),
        "roll_close_fallback_count": int(roll_close_fallback_count),
        "roll_events": roll_events,
        "total_transaction_cost": float(total_transaction_cost),
        "max_margin_used": float(np.max(margin_used_vals)) if len(margin_used_vals) else 0.0,
        "avg_margin_used": float(np.mean(margin_used_vals)) if len(margin_used_vals) else 0.0,
        "max_margin_utilization": float(np.max(finite_util)) if len(finite_util) else 0.0,
        "avg_margin_utilization": float(np.mean(finite_util)) if len(finite_util) else 0.0,
        "min_equity": float(np.min(equity_vals)) if len(equity_vals) else float(init_cash),
        "margin_clipped_events": int(margin_clipped_events),
        "lot_floored_to_zero_events": int(lot_floored_to_zero_events),
        "liquidation_count": len(liquidation_events),
        "liquidation_events": liquidation_events,
        "max_abs_lots": int(lots_df.abs().to_numpy().max()) if not lots_df.empty else 0,
    }

    portfolio = FuturesPortfolio(
        equity=equity_series,
        returns=returns_series,
        asset_values=asset_val_df,
        orders=orders_df,
        trades=trades_df,
        trade_count=trade_count,
        win_rate=win_rate,
        logs=logs_df,
        init_cash=init_cash,
        vbt_freq=vbt_freq,
        year_freq=year_freq,
        lots_history=lots_df,
        margin_used=margin_used_series,
        margin_utilization=margin_util_series,
        roll_events=roll_events,
        liquidation_events=liquidation_events,
        diagnostics=diagnostics,
    )
    return portfolio, diagnostics


def execute_portfolio_simulation(
    *,
    execution_mode: str,
    futures_config: FuturesExecutionConfig,
    target_weights: pd.DataFrame,
    price_aligned: pd.DataFrame,
    open_aligned: Optional[pd.DataFrame],
    use_open: bool,
    slot: Optional[MarketSlot],
    frequency: str,
    init_cash: float,
    buy_cost_rate: float,
    sell_cost_rate: float,
    slippage: float,
    allow_short: bool,
    vbt_freq: str,
    year_freq: Union[str, pd.Timedelta],
    last_order_fees: Optional[pd.DataFrame],
    portfolio_audit_enabled: bool,
    sanitize_prices_fn: Any,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Dispatch simulation between VectorBT nominal target-value and futures contract mode."""
    validate_nominal_execution_mode(
        execution_mode=execution_mode,
        slot=slot,
        instruments=[str(c) for c in price_aligned.columns],
        index=pd.DatetimeIndex(price_aligned.index),
        frequency=frequency,
    )
    if execution_mode == "futures_contract":
        return simulate_futures_contract_portfolio(
            target_weights=target_weights,
            price_aligned=price_aligned,
            open_aligned=open_aligned,
            use_open=use_open,
            slot=slot,
            frequency=frequency,
            init_cash=float(init_cash),
            buy_cost_rate=float(buy_cost_rate),
            sell_cost_rate=float(sell_cost_rate),
            slippage=float(slippage),
            config=futures_config,
            allow_short=bool(allow_short),
            vbt_freq=vbt_freq,
            year_freq=year_freq,
            logger=logger,
        )

    fees_arg: Any = buy_cost_rate
    if (
        buy_cost_rate != sell_cost_rate
        and last_order_fees is not None
        and last_order_fees.shape == target_weights.shape
        and last_order_fees.index.equals(target_weights.index)
        and last_order_fees.columns.equals(target_weights.columns)
    ):
        fees_arg = last_order_fees
    kwargs: Dict[str, Any] = {
        "close": price_aligned,
        "size": target_weights,
        "size_type": "targetvalue",
        "fees": fees_arg,
        "slippage": slippage,
        "cash_sharing": True,
        "call_seq": "auto",
        "freq": vbt_freq,
        "init_cash": init_cash,
        "log": portfolio_audit_enabled,
    }
    if use_open:
        kwargs["price"] = sanitize_prices_fn(open_aligned)
    if allow_short:
        kwargs["direction"] = "both"
    return vbt.Portfolio.from_orders(**kwargs), {}
