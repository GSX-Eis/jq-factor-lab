"""Slot-backed data and microstructure provider for VectorBTBacktestRunner.

Centralizes calendar alignment, bar-feed reads, directional tradability,
auxiliary style/industry/regime-index reads, and long-short allocation helpers
so ``backtest_runner.py`` stays free of market-specific SQL and hardcodes.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import date as _date, datetime as _datetime
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import duckdb
import numpy as np
import pandas as pd

from slots.jq_ashare_local import build_jq_ashare_local
from slots.jq_ashare_local.aux import (
    query_legacy_stub,
    read_legacy_daily_execution_rows,
)
from utils.data_store import binding_contract
from utils.data_store import connect_read_only as _connect_data_read_only
from utils.market_db import MarketDbProfileError
from utils.market_db.validation import market_db_calendar_bounds, validate_market_db_calendar
from utils.market_factor import MarketFactorValueStore
from utils.market_db.identity import slot_market_db_identity
from utils.slots.contracts import ExecutionProfile, MarketSlot
from utils.time_axis import BarSchedule, Frequency

from .factor_loader import FactorLoader


_SESSION_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _is_duckdb_conn(conn: Any) -> bool:
    return isinstance(conn, duckdb.DuckDBPyConnection)


def _session_or_bar_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if "session_id" in frame.columns and frame["session_id"].notna().all():
        # Preserve daily date alignment when bars are uniquely identified by session.
        if not frame["session_id"].duplicated().any():
            return pd.DatetimeIndex(pd.to_datetime(frame["session_id"]))
    return pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))


class SlotDataProvider:
    """Market-agnostic data and microstructure adapter for ``VectorBTBacktestRunner``."""

    def __init__(
        self,
        *,
        slot: Optional[MarketSlot] = None,
        frequency: str = "1d",
        logger: Optional[logging.Logger] = None,
        tradeability_cache: Optional[Dict[tuple, Any]] = None,
        factor_values_store: Optional[Union[str, Path, MarketFactorValueStore]] = None,
        factor_identities: Optional[Mapping[str, str]] = None,
        slot_id_override: Optional[str] = None,
    ) -> None:
        parsed_freq = Frequency.parse(frequency)
        if not parsed_freq.is_valid or parsed_freq.canonical is None:
            raise ValueError(f"Invalid frequency: {frequency!r}")
        self.slot = slot
        self.frequency = parsed_freq.canonical
        self._freq_obj = parsed_freq
        self.logger = logger or logging.getLogger(__name__)
        self.tradeability_cache = tradeability_cache
        self.factor_values_store = factor_values_store
        self.factor_identities = dict(factor_identities or {})
        self.slot_id_override = slot_id_override

        self._owned_conn: Optional[duckdb.DuckDBPyConnection] = None
        self._auto_slot_key: Optional[int] = None
        self._auto_slot: Optional[MarketSlot] = None
        self._session_ts_cache: Dict[tuple[int, str], pd.Timestamp] = {}
        self._bar_schedule_cache: Dict[tuple[int, int, str], BarSchedule] = {}
        self._adv_cache: Dict[tuple[str, int, bool], Dict[str, Optional[float]]] = {}
        self._abv_cache: Dict[tuple[str, int], Dict[str, Optional[float]]] = {}
        self._style_exposure_cache: Dict[str, Optional[pd.DataFrame]] = {}
        self._industry_cache: Dict[str, Dict[str, str]] = {}
        self._regime_z_cache: Optional[tuple[tuple[Any, ...], Optional[pd.Series]]] = None

    @property
    def market_db_identity(self) -> dict[str, str]:
        return slot_market_db_identity(self.slot or self._auto_slot)

    @property
    def execution_profile(self) -> Optional[ExecutionProfile]:
        return self.slot.microstructure.execution_profile if self.slot is not None else None

    def close(self) -> Optional[duckdb.DuckDBPyConnection]:
        """Close any internally opened read-only DuckDB connection and return it."""
        owned = self._owned_conn
        if owned is not None:
            try:
                owned.close()
            except Exception:
                pass
            self._owned_conn = None
        self._auto_slot = None
        self._auto_slot_key = None
        return owned

    def resolve_cost_rates(self, top_k_cfg: Mapping[str, Any]) -> Tuple[float, float]:
        """Resolve ``(buy_cost_rate, sell_cost_rate)`` honoring config overrides and A-share defaults."""
        explicit = top_k_cfg.get("cost_rate")
        if explicit is not None:
            rate = float(explicit)
            return rate, rate
        profile = self.execution_profile
        if profile is not None:
            if (
                self.slot is not None
                and self.slot.slot_id == "jq_ashare_local"
                and not getattr(self.slot.microstructure, "has_explicit_fees", True)
            ):
                return 0.003, 0.003
            buy_rate = float(profile.commission_bps) / 10_000.0
            sell_rate = float(profile.commission_bps + profile.sell_tax_bps) / 10_000.0
            return buy_rate, sell_rate
        return 0.003, 0.003

    def resolve_cost_rate(self, top_k_cfg: Mapping[str, Any]) -> float:
        return self.resolve_cost_rates(top_k_cfg)[0]

    def resolve_slippage(self, top_k_cfg: Mapping[str, Any]) -> float:
        explicit = top_k_cfg.get("slippage")
        if explicit is not None:
            return float(explicit)
        profile = self.execution_profile
        if profile is not None:
            return float(profile.slippage_bps) / 10_000.0
        return 0.0

    def resolve_allow_short(self, top_k_cfg: Mapping[str, Any]) -> bool:
        explicit = top_k_cfg.get("allow_short")
        if explicit is not None:
            return bool(explicit)
        profile = self.execution_profile
        return bool(profile.allow_short) if profile is not None else False

    def resolve_benchmark_instrument(
        self, top_k_cfg: Optional[Mapping[str, Any]] = None
    ) -> Optional[str]:
        if top_k_cfg and top_k_cfg.get("benchmark_instrument"):
            return str(top_k_cfg["benchmark_instrument"])
        profile = self.execution_profile
        return profile.benchmark_instrument if profile is not None else None

    def resolve_calendar(
        self,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
    ) -> Any:
        if self.slot is not None:
            return self.slot.calendar
        from utils.trading_calendar import load_trading_calendar

        if price_db_conn is None and not price_db_path and self._owned_conn is None:
            bindings = binding_contract()
            if input_path := bindings.get("input_db"):
                with _connect_data_read_only(input_path, bindings=bindings) as conn:
                    return self._calendar_for_slot(build_jq_ashare_local(conn), conn, require_frozen=True)
        slot, conn, _ = self._resolve_slot_and_conn(price_db_conn, price_db_path)
        if slot is None:
            return load_trading_calendar(conn, duckdb_path=price_db_path, bindings={})
        return self._calendar_for_slot(slot, conn)

    def _calendar_for_slot(self, slot: MarketSlot, conn: duckdb.DuckDBPyConnection, *,
                           require_frozen: bool = False) -> Any:
        from utils.trading_calendar import TradingCalendar, load_trading_calendar

        profile = slot.market_db_profile
        if self._freq_obj.kind != "session" or (not require_frozen and profile.calendar_source != "local_crosschecked"):
            return slot.calendar
        try:
            verified = load_trading_calendar(conn, require_bound_view=require_frozen)
        except duckdb.CatalogException:
            if require_frozen:
                raise
            return slot.calendar
        validate_market_db_calendar(profile, verified.calendar_id, slot.calendar.timezone)
        start, end = market_db_calendar_bounds(profile)
        start, end = max(start, verified.start), min(end, verified.end)
        verified.validate_coverage(start, end)
        if (start, end) == (verified.start, verified.end):
            return verified
        return TradingCalendar(verified.sessions(start, end), start, end, verified.calendar_id)

    def is_daily_naive_index(self, index: pd.DatetimeIndex) -> bool:
        return bool(
            self.frequency == "1d"
            and index.tz is None
            and (index.empty or index.equals(index.normalize()))
        )

    @staticmethod
    def bar_key(ts: Any, *, daily_naive: bool) -> str:
        t = pd.Timestamp(ts)
        if daily_naive:
            return t.strftime("%Y-%m-%d")
        return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).isoformat()

    @staticmethod
    def norm_bar_ts(ts: Any, *, daily_naive: bool) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if daily_naive:
            return t.normalize()
        return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).as_unit("ns")

    @staticmethod
    def _is_date_only_key(key: Any) -> bool:
        if type(key) is _date:
            return True
        if isinstance(key, (_datetime, pd.Timestamp)):
            return False
        return bool(_SESSION_DATE_RE.fullmatch(str(key).strip()))

    def _session_bars_map(
        self,
        common_dates: pd.DatetimeIndex,
        cal: Any,
    ) -> Tuple[pd.DatetimeIndex, Dict[str, List[pd.Timestamp]]]:
        utc_dates = (
            common_dates.tz_localize("UTC")
            if common_dates.tz is None
            else common_dates.tz_convert("UTC")
        ).as_unit("ns")
        session_to_bars: Dict[str, List[pd.Timestamp]] = {}
        for ts in utc_dates:
            ts_ns = pd.Timestamp(ts).as_unit("ns")
            day_str = ts_ns.strftime("%Y-%m-%d")
            if cal is not None and hasattr(cal, "session_at"):
                try:
                    sess = cal.session_at(ts_ns, frequency=self.frequency) or cal.session_at(ts_ns)
                    if sess is not None:
                        day_str = str(sess.session_date)
                except Exception:
                    pass
            session_to_bars.setdefault(day_str, []).append(ts_ns)
        return utc_dates, session_to_bars

    def session_date_for_bar(
        self,
        ts: Any,
        *,
        daily_naive: bool,
        cal: Any = None,
    ) -> pd.Timestamp:
        if daily_naive:
            return pd.Timestamp(ts).normalize()
        ts_utc = self.norm_bar_ts(ts, daily_naive=False)
        if cal is not None and hasattr(cal, "session_at"):
            try:
                sess = cal.session_at(ts_utc, frequency=self.frequency) or cal.session_at(ts_utc)
                if sess is not None:
                    return pd.Timestamp(sess.session_date).normalize()
            except Exception:
                pass
        return ts_utc.tz_convert("UTC").tz_localize(None).normalize()

    def resolve_pool_key_timestamp(
        self,
        key: Any,
        *,
        daily_naive: bool,
        cal: Any = None,
        session_to_bars: Optional[Mapping[str, List[pd.Timestamp]]] = None,
    ) -> pd.Timestamp:
        """Resolve a ``dynamic_pools`` key onto the unified backtest Bar time axis."""
        ts = pd.Timestamp(key)
        is_date_only = self._is_date_only_key(key)
        if daily_naive:
            if is_date_only:
                return ts.normalize()
            return (ts.tz_localize(None) if ts.tzinfo is None else ts.tz_convert("UTC").tz_localize(None)).as_unit("ns")

        if is_date_only:
            day_str = ts.strftime("%Y-%m-%d")
            if cal is not None and hasattr(cal, "bar_grid"):
                try:
                    grid = cal.bar_grid(self.frequency, day_str, day_str)
                    if not grid.empty:
                        return pd.Timestamp(grid["timestamp_utc"].iloc[0]).tz_convert("UTC").as_unit("ns")
                except Exception:
                    pass
            if session_to_bars and day_str in session_to_bars:
                return pd.Timestamp(session_to_bars[day_str][0]).as_unit("ns")
            if cal is not None and hasattr(cal, "timezone"):
                try:
                    return pd.Timestamp(day_str).tz_localize(cal.timezone).tz_convert("UTC").as_unit("ns")
                except Exception:
                    pass
            return pd.Timestamp(day_str).tz_localize("UTC").as_unit("ns")

        return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).as_unit("ns")

    def resolve_dynamic_pools(
        self,
        common_dates: pd.DatetimeIndex,
        dynamic_pools: Mapping[Any, Iterable[str]],
        price_db_conn: Optional[Any] = None,
        price_db_path: Optional[str] = None,
    ) -> Tuple[List[pd.Timestamp], Dict[pd.Timestamp, set[str]]]:
        """Resolve ``dynamic_pools`` onto sorted Bar keys matching ``common_dates``."""
        if not dynamic_pools:
            return [], {}
        daily_naive = self.is_daily_naive_index(common_dates)
        cal = None
        session_to_bars: Dict[str, List[pd.Timestamp]] = {}
        if not daily_naive:
            try:
                cal = self.resolve_calendar(price_db_conn, price_db_path)
            except MarketDbProfileError:
                raise
            except Exception:
                cal = None
            _, session_to_bars = self._session_bars_map(common_dates, cal)

        resolved_pools: Dict[pd.Timestamp, set[str]] = {}
        for key, values in dynamic_pools.items():
            eff_ts = self.resolve_pool_key_timestamp(
                key, daily_naive=daily_naive, cal=cal, session_to_bars=session_to_bars
            )
            resolved_pools[eff_ts] = set(values)
        return sorted(resolved_pools), resolved_pools

    def resolve_pool_rebalance_dates(
        self,
        common_dates: pd.DatetimeIndex,
        pool_keys: Iterable[Any],
        price_db_conn: Optional[Any] = None,
        price_db_path: Optional[str] = None,
    ) -> List[Any]:
        """Map ``dynamic_pools`` keys onto actual backtest bar timestamps."""
        if common_dates.empty:
            return []
        daily_naive = self.is_daily_naive_index(common_dates)
        if daily_naive:
            sorted_keys = sorted(
                pool_keys,
                key=lambda d: self.resolve_pool_key_timestamp(d, daily_naive=True),
            )
            return [pd.to_datetime(d).to_pydatetime() for d in sorted_keys]

        try:
            cal = self.resolve_calendar(price_db_conn, price_db_path)
        except Exception:
            cal = None
        utc_dates, session_to_bars = self._session_bars_map(common_dates, cal)
        sorted_utc = sorted(pd.Timestamp(ts).as_unit("ns") for ts in utc_dates)
        utc_lookup = {ts: ts.to_pydatetime() for ts in sorted_utc}

        resolved: List[Any] = []
        seen: set[pd.Timestamp] = set()
        keyed = sorted(
            (
                (
                    self.resolve_pool_key_timestamp(
                        key, daily_naive=False, cal=cal, session_to_bars=session_to_bars
                    ),
                    key,
                )
                for key in pool_keys
            ),
            key=lambda item: item[0],
        )
        for eff_ts, key in keyed:
            if self._is_date_only_key(key):
                day_str = pd.Timestamp(key).strftime("%Y-%m-%d")
                bars = session_to_bars.get(day_str)
                if bars:
                    bar_ts = pd.Timestamp(bars[0]).as_unit("ns")
                    if bar_ts not in seen:
                        resolved.append(bars[0].to_pydatetime())
                        seen.add(bar_ts)
                continue

            bar_dt = utc_lookup.get(eff_ts)
            if bar_dt is not None:
                if eff_ts not in seen:
                    resolved.append(bar_dt)
                    seen.add(eff_ts)
                continue
            if sorted_utc and sorted_utc[0] <= eff_ts <= sorted_utc[-1]:
                idx = bisect_left(sorted_utc, eff_ts)
                if idx < len(sorted_utc):
                    bar_ts = sorted_utc[idx]
                    if bar_ts not in seen:
                        resolved.append(bar_ts.to_pydatetime())
                        seen.add(bar_ts)
        return resolved

    def read_execution_price_frame(
        self,
        all_dates: pd.DatetimeIndex,
        columns: Sequence[str],
    ) -> Optional[pd.DataFrame]:
        """Read the slot's ``entry_price_field`` execution panel when ``open_df`` is omitted."""
        if self.slot is None or all_dates.empty or not columns:
            return None
        field = self.slot.microstructure.execution_profile.entry_price_field
        if field != "open":
            return None
        daily_naive = self.is_daily_naive_index(all_dates)
        session_map: Optional[Dict[pd.Timestamp, pd.Timestamp]] = None
        if daily_naive:
            grid = self.slot.calendar.bar_grid(
                self.frequency,
                all_dates.min().strftime("%Y-%m-%d"),
                all_dates.max().strftime("%Y-%m-%d"),
            )
            if grid.empty:
                raise ValueError(
                    f"Slot requires entry_price_field={field!r}, but calendar bar_grid is empty"
                )
            start_utc = pd.Timestamp(grid["timestamp_utc"].min())
            end_utc = pd.Timestamp(grid["timestamp_utc"].max())
            session_map = {
                pd.Timestamp(row.timestamp_utc).tz_convert("UTC").as_unit("ns"): pd.Timestamp(row.session_id)
                for row in grid.itertuples(index=False)
            }
        else:
            utc_dates = (
                all_dates.tz_localize("UTC")
                if all_dates.tz is None
                else all_dates.tz_convert("UTC")
            ).as_unit("ns")
            start_utc = pd.Timestamp(utc_dates.min())
            end_utc = pd.Timestamp(utc_dates.max())
        try:
            bars = self.slot.bar_feed.read_execution_bars(
                list(columns),
                self._freq_obj,
                start_utc=start_utc.to_pydatetime(),
                end_utc=end_utc.to_pydatetime(),
            )
        except Exception as exc:
            raise ValueError(
                f"Slot requires entry_price_field={field!r}, but execution bar read failed: {exc}"
            ) from exc
        if bars is None or bars.empty or field not in bars.columns:
            raise ValueError(
                f"Slot requires entry_price_field={field!r}, but execution bars are missing {field!r}"
            )
        frame = bars.copy()
        if daily_naive:
            if "session_id" in frame.columns and frame["session_id"].notna().all():
                frame["_axis"] = pd.to_datetime(frame["session_id"]).dt.as_unit("ns")
            elif session_map is not None and "timestamp_utc" in frame.columns:
                ts_col = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.as_unit("ns")
                frame["_axis"] = ts_col.map(session_map)
            else:
                raise ValueError(
                    f"Slot requires entry_price_field={field!r}, but execution bars cannot align to session dates"
                )
        else:
            frame["_axis"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.as_unit("ns")
        frame = frame.dropna(subset=["_axis"])
        if frame.empty:
            raise ValueError(
                f"Slot requires entry_price_field={field!r}, but no execution bars align to backtest dates"
            )
        wide = frame.assign(**{field: pd.to_numeric(frame[field], errors="coerce")}).pivot_table(
            index="_axis",
            columns="instrument_id",
            values=field,
            aggfunc="last",
        )
        aligned = sanitize_execution_prices(
            wide.reindex(index=all_dates, columns=list(columns))
        )
        if not has_finite_positive_prices(aligned):
            raise ValueError(
                f"Slot requires entry_price_field={field!r}, but execution bars contain no finite positive {field!r} prices"
            )
        return aligned

    def annualization_spec(
        self,
        price_db_conn: Optional[Any] = None,
        price_db_path: Optional[str] = None,
        *,
        calendar: Optional[Any] = None,
    ) -> Tuple[str, Union[str, pd.Timedelta], float]:
        """Resolve ``(vbt_freq, year_freq, bars_per_year)`` solely from slot calendar & frequency."""
        freq_map = {
            "1d": "1D",
            "1w": "1W",
            "1h": "1h",
            "4h": "4h",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1min": "1min",
        }
        vbt_freq = freq_map.get(self.frequency, self.frequency)
        cal = calendar
        if cal is None:
            try:
                cal = self.resolve_calendar(price_db_conn, price_db_path)
            except MarketDbProfileError:
                raise
            except Exception:
                cal = None

        sessions_per_year = 252.0
        if cal is not None:
            padded = getattr(cal, "_padded_sessions", None)
            if isinstance(padded, dict) and "ETH" in padded and padded["ETH"]:
                active_days = pd.DatetimeIndex(
                    [k for k, v in padded["ETH"].items() if getattr(v, "segments", ())]
                )
                active_weekdays = set(active_days.weekday.tolist())
                if {5, 6} <= active_weekdays and len(active_weekdays) == 7:
                    sessions_per_year = 365.0
                elif len(active_weekdays) > 5:
                    sessions_per_year = 365.0 * len(active_weekdays) / 7.0
                else:
                    sessions_per_year = 252.0
            else:
                try:
                    sess = cal.sessions()
                    if any(pd.Timestamp(d).weekday() >= 5 for d in sess):
                        sessions_per_year = 365.0
                except Exception:
                    sessions_per_year = 252.0

        if self._freq_obj.kind == "session":
            bars_per_year = float(round(sessions_per_year))
            return vbt_freq, f"{int(bars_per_year)}D", bars_per_year
        if self._freq_obj.kind == "week":
            return vbt_freq, "52W", 52.0

        bars_per_session = self.bars_per_session(
            price_db_conn, price_db_path, calendar=cal
        )
        bars_per_year = float(sessions_per_year * bars_per_session)
        total_minutes = max(1, int(round(bars_per_year * float(self._freq_obj.minutes))))
        return vbt_freq, pd.Timedelta(minutes=total_minutes), bars_per_year

    @staticmethod
    def _week_session_count(cal: Any, session_id: str) -> Optional[int]:
        """Return the number of trading sessions covered by the weekly bar at ``session_id``."""
        if cal is None:
            return None
        if hasattr(cal, "_week_on") and hasattr(cal, "_padded_sessions"):
            try:
                week = cal._week_on(session_id, "ETH")
                if week:
                    return len(week)
            except Exception:
                pass
        if hasattr(cal, "sessions"):
            try:
                day = pd.Timestamp(session_id).normalize()
                monday = day - pd.Timedelta(days=int(day.weekday()))
                sunday = monday + pd.Timedelta(days=6)
                start_bound = getattr(cal, "start", monday)
                end_bound = getattr(cal, "end", sunday)
                s0 = max(start_bound, monday)
                s1 = min(end_bound, sunday)
                if s0 <= s1:
                    n_sess = len(cal.sessions(s0, s1))
                    if n_sess > 0:
                        return int(n_sess)
            except Exception:
                pass
        return None

    def bars_per_session(
        self,
        price_db_conn: Optional[Any] = None,
        price_db_path: Optional[str] = None,
        *,
        calendar: Optional[Any] = None,
    ) -> float:
        """Resolve expected bar count per trading session for the runner's frequency."""
        if self._freq_obj.kind == "session":
            return 1.0
        cal = calendar
        if cal is None:
            try:
                cal = self.resolve_calendar(price_db_conn, price_db_path)
            except MarketDbProfileError:
                raise
            except Exception:
                cal = None
        if self._freq_obj.kind == "week":
            if cal is not None and hasattr(cal, "bar_grid"):
                try:
                    sess = cal.sessions()
                    if len(sess) > 0:
                        grid = cal.bar_grid("1w", sess[0], sess[-1])
                        if not grid.empty:
                            covered_sessions = 0
                            for sid in grid["session_id"].astype(str):
                                n_week_sess = self._week_session_count(cal, sid)
                                if n_week_sess is not None and n_week_sess > 0:
                                    covered_sessions += n_week_sess
                            if covered_sessions > 0:
                                return float(len(grid)) / float(covered_sessions)
                            return float(len(grid)) / float(len(sess))
                except Exception:
                    pass
            if cal is not None and hasattr(cal, "sessions"):
                try:
                    sess = cal.sessions()
                    if any(pd.Timestamp(d).weekday() >= 5 for d in sess):
                        return 1.0 / 7.0
                except Exception:
                    pass
            return 0.2

        bars_per_session: Optional[float] = None
        slot = self.slot or self._auto_slot
        profile = slot.market_db_profile if slot is not None else None
        if profile is not None:
            for frequency, count in profile.market_rules.get("bars_per_session", {}).items():
                if Frequency.parse(frequency).canonical == self.frequency:
                    bars_per_session = float(count)
                    break
        if bars_per_session is None and cal is not None and hasattr(cal, "bar_grid"):
            try:
                sess = cal.sessions()
                if len(sess) > 0:
                    sample_end = sess[min(4, len(sess) - 1)]
                    grid = cal.bar_grid(self.frequency, sess[0], sample_end)
                    if not grid.empty:
                        n_sess = max(1, int(grid["session_id"].nunique()))
                        bars_per_session = len(grid) / float(n_sess)
            except Exception:
                bars_per_session = None
        if bars_per_session is None or bars_per_session <= 0:
            raise ValueError(
                f"Cannot determine bars per session for {self.frequency} from the market calendar"
            )
        return float(bars_per_session)

    def align_price_index(self, price_df: pd.DataFrame, calendar: Any) -> pd.DatetimeIndex:
        """Align ``price_df`` index to ``calendar`` for both daily and UTC/intraday bars."""
        if price_df.empty:
            return pd.DatetimeIndex(price_df.index)
        idx = pd.DatetimeIndex(price_df.index)
        if (
            self.frequency == "1d"
            and idx.tz is None
            and not idx.hasnans
            and idx.equals(idx.normalize())
        ):
            return calendar.align(price_df).index

        if idx.hasnans or idx.has_duplicates or not idx.is_monotonic_increasing:
            raise ValueError(
                "Backtest prices must have strictly increasing, unique, non-null timestamps"
            )
        utc_idx = (
            idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        ).as_unit("ns")
        first_session = calendar.session_at(utc_idx[0], frequency=self.frequency) or calendar.session_at(utc_idx[0])
        last_session = calendar.session_at(utc_idx[-1], frequency=self.frequency) or calendar.session_at(utc_idx[-1])
        start_day = first_session.session_date if first_session is not None else calendar.start
        end_day = last_session.session_date if last_session is not None else calendar.end
        grid = calendar.bar_grid(self.frequency, start_day, end_day)
        expected = pd.DatetimeIndex(
            grid.loc[grid["timestamp_utc"].between(utc_idx[0], utc_idx[-1]), "timestamp_utc"]
        ).as_unit("ns")
        if not utc_idx.equals(expected):
            raise ValueError(
                "Backtest prices must cover every market session in order; import missing dates"
            )
        return expected

    def previous_trading_day(
        self,
        current_date_str: str,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
    ) -> Optional[str]:
        cal = self.resolve_calendar(price_db_conn, price_db_path)
        if _SESSION_DATE_RE.fullmatch(current_date_str) and self.frequency == "1d":
            sessions = cal.sessions()
            if len(sessions) > 0 and current_date_str == sessions[0].strftime("%Y-%m-%d"):
                return None
            return cal.previous_sessions(current_date_str, 1)[0].strftime("%Y-%m-%d")
        ts = pd.Timestamp(current_date_str)
        ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        session = cal.session_at(ts_utc, frequency=self.frequency) or cal.session_at(ts_utc)
        if session is None:
            return None
        start_day = max(cal.start, pd.Timestamp(session.session_date) - pd.Timedelta(days=14))
        grid = cal.bar_grid(self.frequency, start_day, session.session_date)
        prior = grid.loc[grid["timestamp_utc"] < ts_utc, "timestamp_utc"]
        if prior.empty:
            return None
        return pd.Timestamp(prior.iloc[-1]).isoformat()

    def _ensure_duckdb_conn(
        self,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
    ) -> tuple[Optional[Any], bool]:
        if price_db_conn is not None:
            return price_db_conn, False
        if self._owned_conn is not None:
            return self._owned_conn, False
        if price_db_path:
            conn = _connect_data_read_only(price_db_path)
            self._owned_conn = conn
            return conn, True
        return None, False

    def _resolve_slot_and_conn(
        self,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
    ) -> tuple[Optional[MarketSlot], Optional[Any], bool]:
        if self.slot is not None:
            return self.slot, price_db_conn, False
        conn, newly_opened = self._ensure_duckdb_conn(price_db_conn, price_db_path)
        if conn is None or not _is_duckdb_conn(conn):
            return None, conn, newly_opened

        if self._auto_slot is not None and self._auto_slot_key == id(conn):
            return self._auto_slot, conn, newly_opened

        built = build_jq_ashare_local(conn)
        self._auto_slot = built
        self._auto_slot_key = id(conn)
        return built, conn, newly_opened

    def _resolve_bar_anchor(
        self, slot: MarketSlot, date_str: str
    ) -> Optional[tuple[pd.Timestamp, str]]:
        key = (id(slot.calendar), date_str)
        cached = self._session_ts_cache.get(key)
        if _SESSION_DATE_RE.fullmatch(date_str) and self.frequency == "1d":
            if cached is not None:
                return cached, date_str
            try:
                grid = slot.calendar.bar_grid("1d", date_str, date_str)
            except Exception:
                return None
            if grid.empty:
                return None
            ts = pd.Timestamp(grid.iloc[0]["timestamp_utc"])
            self._session_ts_cache[key] = ts
            return ts, date_str

        ts = pd.Timestamp(date_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        try:
            session = slot.calendar.session_at(ts, frequency=self.frequency) or slot.calendar.session_at(ts)
        except Exception:
            return None
        if session is None:
            return None
        return ts, session.session_date

    def _window_utc_bounds(
        self,
        slot: MarketSlot,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        cal = slot.calendar
        start_day = cal.start if start_date is None else max(pd.Timestamp(start_date).normalize(), cal.start)
        end_day = cal.end if end_date is None else min(pd.Timestamp(end_date).normalize(), cal.end)
        if start_day > end_day:
            return None
        grid = cal.bar_grid(self.frequency, start_day, end_day)
        if grid.empty:
            return None
        return pd.Timestamp(grid["timestamp_utc"].min()), pd.Timestamp(grid["timestamp_utc"].max())

    def filter_tradeable_stocks(
        self,
        date_str: str,
        stock_list: List[str],
        *,
        side: str = "both",
        require_non_st: bool = True,
        price_db_conn: Optional[Any] = None,
        price_db_path: Optional[str] = None,
        exec_price_df: Optional[pd.DataFrame] = None,
        on_conn_opened: Optional[Any] = None,
    ) -> List[str]:
        """Filter ``stock_list`` via ``slot.microstructure.tradability_mask``."""
        if not stock_list:
            return []
        if side not in {"buy", "sell", "both", "hold"}:
            raise ValueError(f"unsupported trade side: {side}")

        cache_key = (
            "directional",
            date_str,
            tuple(sorted(stock_list)),
            side,
            require_non_st,
        )
        if self.tradeability_cache is not None and cache_key in self.tradeability_cache:
            return list(self.tradeability_cache[cache_key])

        try:
            slot, conn, newly_opened = self._resolve_slot_and_conn(
                price_db_conn, price_db_path
            )
            if newly_opened and conn is not None and on_conn_opened is not None:
                on_conn_opened(conn)
        except MarketDbProfileError:
            raise
        except Exception as exc:
            self.logger.warning(f"[VectorBT] Cannot connect to market slot DB: {exc}")
            return []

        if slot is None:
            return stock_list

        anchor = self._resolve_bar_anchor(slot, date_str)
        if anchor is None:
            return []
        ts_utc, session_id = anchor

        row_cache = self.tradeability_cache
        unresolved = [
            stock
            for stock in stock_list
            if row_cache is None or ("directional_row", date_str, stock) not in row_cache
        ]
        local_states: Dict[str, Dict[str, bool]] = {}

        if unresolved:
            bars_by_id: Dict[str, Mapping[str, Any]] = {}
            has_limit_cols = False
            try:
                bars = slot.bar_feed.read_execution_bars(
                    unresolved, self._freq_obj, start_utc=ts_utc, end_utc=ts_utc
                )
                has_limit_cols = "high_limit" in bars.columns and "low_limit" in bars.columns
                if not bars.empty:
                    bars_by_id = {
                        str(row["instrument_id"]): row
                        for row in bars.to_dict(orient="records")
                    }
            except ValueError as exc:
                # Minimal legacy DuckDB fixtures may omit high/low/volume/factor.
                if self.slot is None and conn is not None and _is_duckdb_conn(conn):
                    legacy = read_legacy_daily_execution_rows(conn, unresolved, session_id)
                    has_limit_cols = "high_limit" in legacy.columns and "low_limit" in legacy.columns
                    if not legacy.empty:
                        bars_by_id = {
                            str(row["instrument_id"]): row
                            for row in legacy.to_dict(orient="records")
                        }
                else:
                    self.logger.warning(f"[VectorBT] Bar feed read failed at {date_str}: {exc}")

            # Optional ST flag via AuxFeatureSlot (silently absent on non-A-share slots).
            non_st_set = set(unresolved)
            try:
                st_grid = pd.DataFrame(
                    {
                        "instrument_id": unresolved,
                        "timestamp_utc": pd.DatetimeIndex([ts_utc] * len(unresolved), tz="UTC"),
                        "session_id": [session_id] * len(unresolved),
                    }
                )
                st_df = slot.read_aux_features(st_grid, features=["is_st"])
                if not st_df.empty and "is_st" in st_df.columns:
                    st_map = st_df.set_index("instrument_id")["is_st"].to_dict()
                    for stock in unresolved:
                        val = st_map.get(stock)
                        if pd.notna(val):
                            norm = str(val).strip().lower()
                            if norm not in {"0", "false", "f", "no", "n", ""}:
                                non_st_set.discard(stock)
            except (ValueError, KeyError, NotImplementedError, LookupError):
                pass
            except Exception as exc:
                self.logger.debug(f"[VectorBT] ST aux query skipped at {date_str}: {exc}")

            is_ashare = slot.slot_id == "jq_ashare_local"
            is_st_fn = getattr(slot.microstructure, "is_st_on", None)
            if callable(is_st_fn):
                for stock in unresolved:
                    if stock in non_st_set:
                        try:
                            if is_st_fn(session_id, stock):
                                non_st_set.discard(stock)
                        except Exception:
                            pass
            for stock in unresolved:
                bar = bars_by_id.get(stock)
                exec_px: Optional[float] = None
                if bar is not None:
                    entry_field = slot.microstructure.execution_profile.entry_price_field
                    raw_px = bar.get("open") if is_ashare else bar.get(entry_field)
                    if (raw_px is None or not pd.notna(raw_px)) and not is_ashare and entry_field != "open":
                        raw_px = bar.get("close")
                    if raw_px is not None and pd.notna(raw_px):
                        px_val = float(raw_px)
                        if np.isfinite(px_val) and px_val > 0:
                            exec_px = px_val
                if exec_px is None and exec_price_df is not None and stock in exec_price_df.columns:
                    key_idx = pd.Timestamp(date_str)
                    if exec_price_df.index.tz is not None and key_idx.tzinfo is None:
                        key_idx = key_idx.tz_localize(exec_price_df.index.tz)
                    elif exec_price_df.index.tz is None and key_idx.tzinfo is not None:
                        key_idx = key_idx.tz_convert("UTC").tz_localize(None)
                    if key_idx in exec_price_df.index:
                        val = exec_price_df.loc[key_idx, stock]
                        if pd.notna(val):
                            px_val = float(val)
                            if np.isfinite(px_val) and px_val > 0:
                                exec_px = px_val

                row_grid = pd.DataFrame(
                    {
                        "instrument_id": [stock],
                        "timestamp_utc": pd.DatetimeIndex([ts_utc], tz="UTC"),
                        "session_id": [session_id],
                    }
                )
                if exec_px is not None:
                    row_grid["close"] = exec_px

                if is_ashare:
                    limits_ok = True
                    paused_known = True
                    if bar is not None:
                        if has_limit_cols:
                            limits_ok = bool(
                                pd.notna(bar.get("high_limit"))
                                and pd.notna(bar.get("low_limit"))
                            )
                        if "paused" in bar:
                            paused_known = pd.notna(bar.get("paused"))
                    valid = bool(exec_px is not None and limits_ok)
                    if not valid or not paused_known:
                        buy_ok = False
                        sell_ok = False
                    else:
                        buy_ok = bool(
                            slot.microstructure.tradability_mask(
                                row_grid.assign(side="buy"), instrument_id=stock
                            ).iloc[0]
                        )
                        try:
                            sell_ok = bool(
                                slot.microstructure.tradability_mask(
                                    row_grid.assign(side="sell"),
                                    instrument_id=stock,
                                    exclude_st=False,
                                ).iloc[0]
                            )
                        except TypeError:
                            sell_ok = bool(
                                slot.microstructure.tradability_mask(
                                    row_grid.assign(side="sell"), instrument_id=stock
                                ).iloc[0]
                            )
                    hold_ok = valid
                else:
                    buy_ok = bool(
                        slot.microstructure.tradability_mask(
                            row_grid.assign(side="buy"), instrument_id=stock
                        ).iloc[0]
                    )
                    sell_ok = bool(
                        slot.microstructure.tradability_mask(
                            row_grid.assign(side="sell"), instrument_id=stock
                        ).iloc[0]
                    )
                    hold_ok = bool(
                        slot.microstructure.tradability_mask(
                            row_grid, instrument_id=stock
                        ).iloc[0]
                    )

                state = {
                    "buy": buy_ok,
                    "sell": sell_ok,
                    "both": bool(buy_ok and sell_ok),
                    "hold": hold_ok,
                    "non_st": stock in non_st_set,
                }
                local_states[stock] = state
                if row_cache is not None:
                    row_cache[("directional_row", date_str, stock)] = state

        allowed: List[str] = []
        for stock in stock_list:
            state = (
                row_cache.get(("directional_row", date_str, stock))
                if row_cache is not None
                else local_states.get(stock)
            )
            if state is None:
                continue
            if state[side] and (not require_non_st or state["non_st"]):
                allowed.append(stock)

        if row_cache is not None:
            row_cache[cache_key] = list(allowed)
        return allowed

    def _bar_schedule_for_slot(self, slot: MarketSlot) -> Optional[BarSchedule]:
        """Resolve a ``BarSchedule`` matching ``slot.calendar`` and ``self._freq_obj``."""
        feed_schedule = getattr(slot.bar_feed, "schedule", None)
        if (
            isinstance(feed_schedule, BarSchedule)
            and feed_schedule.frequency.canonical == self._freq_obj.canonical
        ):
            return feed_schedule
        cal = slot.calendar
        if cal is None or not hasattr(cal, "_intervals"):
            return None
        cache_key = (id(slot.bar_feed), id(cal), str(self.frequency))
        cached = self._bar_schedule_cache.get(cache_key)
        if cached is not None:
            return cached
        bucket_strategy = getattr(slot.bar_feed, "bucket_strategy", None)
        if (
            getattr(cal, "is_domestic_futures", False)
            and self._freq_obj.kind != "week"
            and bucket_strategy is None
        ):
            return None
        view = getattr(slot.bar_feed, "view", "ETH")
        schedule = BarSchedule(
            cal, self._freq_obj, view=view, bucket_strategy=bucket_strategy
        )
        self._bar_schedule_cache[cache_key] = schedule
        return schedule

    def _session_completed_at(
        self,
        slot: MarketSlot,
        session_id: str,
        schedule: Optional[BarSchedule],
    ) -> Optional[pd.Timestamp]:
        """Return the UTC completion timestamp (``available_at``) of ``session_id``."""
        if schedule is not None:
            slots = schedule.for_session(session_id)
            if not slots.empty:
                last_start = (
                    pd.Timestamp(slots["start_utc"].iloc[-1])
                    .tz_convert("UTC")
                    .as_unit("ns")
                )
                return schedule.available_at(last_start).tz_convert("UTC")
        cal = slot.calendar
        if cal is not None and hasattr(cal, "session_on") and hasattr(cal, "_intervals"):
            view = getattr(slot.bar_feed, "view", "ETH")
            sess = cal.session_on(session_id, view)
            if sess is not None and sess.segments:
                return pd.Timestamp(sess.segments[-1][1]).tz_convert("UTC")
        return None

    def _latest_started_session_day(
        self,
        slot: MarketSlot,
        as_of_utc: pd.Timestamp,
    ) -> Optional[pd.Timestamp]:
        """Return the latest session date in ``slot.calendar`` opened strictly before ``as_of_utc``."""
        cal = slot.calendar
        view = getattr(slot.bar_feed, "view", "ETH")
        opens = getattr(cal, "_opens", None)
        intervals = getattr(cal, "_intervals", None)
        if isinstance(opens, Mapping) and isinstance(intervals, Mapping) and view in opens:
            view_opens = opens[view]
            view_intervals = intervals[view]
            if not view_opens:
                return None
            pos = bisect_left(view_opens, as_of_utc) - 1
            if pos < 0:
                return None
            day = pd.Timestamp(view_intervals[pos][2].session_date).normalize()
            day = min(day, cal.end)
            return day if day >= cal.start else None
        return self.session_date_for_bar(as_of_utc, daily_naive=False, cal=cal)

    def liquidity_as_of_key(
        self,
        bar_key: str,
        price_db_conn: Optional[Any] = None,
        price_db_path: Optional[str] = None,
    ) -> str:
        """Resolve a signal bar key to its ``BarSchedule.available_at()`` publication key."""
        if self._is_date_only_key(bar_key):
            return bar_key
        slot, _, _ = self._resolve_slot_and_conn(price_db_conn, price_db_path)
        if slot is None:
            return bar_key
        ts = pd.Timestamp(bar_key)
        ts_utc = (
            ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        ).as_unit("ns")
        schedule = self._bar_schedule_for_slot(slot)
        if schedule is not None:
            try:
                return schedule.available_at(ts_utc).isoformat()
            except Exception:
                return bar_key
        cal = slot.calendar
        if (
            self._freq_obj.kind in ("session", "week")
            and cal is not None
            and hasattr(cal, "session_at")
            and hasattr(cal, "_intervals")
        ):
            view = getattr(slot.bar_feed, "view", "ETH")
            try:
                sess = cal.session_at(ts_utc, view, frequency=self._freq_obj)
                if sess is not None and sess.segments:
                    return pd.Timestamp(sess.segments[-1][1]).tz_convert("UTC").isoformat()
            except Exception:
                return bar_key
        return bar_key

    @staticmethod
    def _resolve_roll_schedule_contract(
        roll_schedule: Any,
        inst_id: str,
        session_id: Optional[str],
        ts_val: Any,
    ) -> Optional[str]:
        """Resolve a concrete contract ID from an optional ``slot.roll_schedule`` helper."""
        if roll_schedule is None:
            return None
        if callable(roll_schedule):
            try:
                res = roll_schedule(inst_id, session_id)
            except TypeError:
                if ts_val is None or pd.isna(ts_val):
                    return None
                res = roll_schedule(inst_id, ts_val)
            if res is not None and pd.notna(res):
                text = str(res).strip()
                return text or None
            return None
        if isinstance(roll_schedule, Mapping):
            if session_id is not None and (inst_id, session_id) in roll_schedule:
                val = roll_schedule[(inst_id, session_id)]
                if val is not None and pd.notna(val):
                    text = str(val).strip()
                    return text or None
            entry = roll_schedule.get(inst_id)
            if isinstance(entry, Mapping) and session_id is not None:
                val = entry.get(session_id)
                if val is not None and pd.notna(val):
                    text = str(val).strip()
                    return text or None
            elif isinstance(entry, str) and entry.strip():
                return entry.strip()
            return None
        if isinstance(roll_schedule, pd.DataFrame) and not roll_schedule.empty:
            mask = pd.Series(True, index=roll_schedule.index)
            if "product_root" in roll_schedule.columns:
                mask &= roll_schedule["product_root"].astype(str).eq(inst_id)
            elif "instrument_id" in roll_schedule.columns:
                mask &= roll_schedule["instrument_id"].astype(str).eq(inst_id)
            if session_id is not None:
                if "effective_session_id" in roll_schedule.columns:
                    mask &= roll_schedule["effective_session_id"].astype(str).eq(session_id)
                elif "session_id" in roll_schedule.columns:
                    mask &= roll_schedule["session_id"].astype(str).eq(session_id)
            matched = roll_schedule.loc[mask]
            if not matched.empty:
                for col in ("to_contract", "concrete_instrument_id", "contract_id"):
                    if col in matched.columns and pd.notna(matched[col].iloc[0]):
                        text = str(matched[col].iloc[0]).strip()
                        if text:
                            return text
        return None

    def _resolve_bar_contract_multipliers(
        self,
        slot: Optional[MarketSlot],
        bars: pd.DataFrame,
    ) -> pd.Series:
        """Resolve per-row ``contract_multiplier`` for OHLCV ``volume * close`` fallback."""
        if bars.empty:
            return pd.Series(dtype=float, index=bars.index)
        universe = getattr(slot, "universe", None) if slot is not None else None
        spec_fn = getattr(universe, "instrument_spec", None)
        if not callable(spec_fn):
            return pd.Series(1.0, index=bars.index, dtype=float)

        spec_mult_cache: Dict[str, Optional[float]] = {}

        def _spec_multiplier(target_id: str) -> Optional[float]:
            if target_id in spec_mult_cache:
                return spec_mult_cache[target_id]
            try:
                spec = spec_fn(target_id)
            except (KeyError, ValueError, LookupError):
                spec_mult_cache[target_id] = None
                return None
            raw_mult = getattr(spec, "contract_multiplier", 1.0)
            mult_val = float(raw_mult)
            if not np.isfinite(mult_val) or mult_val <= 0:
                raise ValueError(
                    f"Invalid contract_multiplier for {target_id!r}: {raw_mult!r}"
                )
            spec_mult_cache[target_id] = mult_val
            return mult_val

        roll_schedule = getattr(slot, "roll_schedule", None)
        resolve_active_fn = getattr(universe, "resolve_active_contract", None)
        concrete_col = (
            "concrete_instrument_id"
            if "concrete_instrument_id" in bars.columns
            else ("contract_id" if "contract_id" in bars.columns else None)
        )
        has_session_col = "session_id" in bars.columns
        has_ts_col = "timestamp_utc" in bars.columns

        inst_vals = (
            bars["instrument_id"].tolist()
            if "instrument_id" in bars.columns
            else [None] * len(bars)
        )
        concrete_vals = bars[concrete_col].tolist() if concrete_col else [None] * len(bars)
        session_vals = bars["session_id"].tolist() if has_session_col else [None] * len(bars)
        ts_vals = bars["timestamp_utc"].tolist() if has_ts_col else [None] * len(bars)

        row_cache: Dict[tuple[str, str, str], float] = {}
        multipliers: List[float] = []
        for inst_raw, concrete_raw, sid_raw, ts_raw in zip(
            inst_vals, concrete_vals, session_vals, ts_vals
        ):
            inst_id = (
                str(inst_raw).strip()
                if inst_raw is not None and pd.notna(inst_raw)
                else ""
            )
            concrete_id = (
                str(concrete_raw).strip()
                if concrete_raw is not None and pd.notna(concrete_raw)
                else ""
            )
            if not concrete_id and roll_schedule is None and inst_id:
                direct_mult = _spec_multiplier(inst_id)
                if direct_mult is not None:
                    multipliers.append(direct_mult)
                    continue

            sid_str: Optional[str] = (
                str(sid_raw).strip()
                if sid_raw is not None and pd.notna(sid_raw)
                else None
            )
            if (
                sid_str is None
                and roll_schedule is not None
                and ts_raw is not None
                and pd.notna(ts_raw)
                and slot is not None
                and slot.calendar is not None
            ):
                sid_str = self.session_date_for_bar(
                    ts_raw, daily_naive=False, cal=slot.calendar
                ).strftime("%Y-%m-%d")

            cache_key = (inst_id, concrete_id, sid_str or "")
            if (concrete_id or roll_schedule is not None) and cache_key in row_cache:
                multipliers.append(row_cache[cache_key])
                continue

            mult: Optional[float] = None
            if concrete_id:
                mult = _spec_multiplier(concrete_id)
            if mult is None and roll_schedule is not None and inst_id:
                sched_contract = self._resolve_roll_schedule_contract(
                    roll_schedule, inst_id, sid_str, ts_raw
                )
                if sched_contract:
                    mult = _spec_multiplier(sched_contract)
            if mult is None and inst_id:
                mult = _spec_multiplier(inst_id)
            if (
                mult is None
                and callable(resolve_active_fn)
                and inst_id
                and ts_raw is not None
                and pd.notna(ts_raw)
            ):
                ts_utc = pd.Timestamp(ts_raw)
                ts_utc = (
                    ts_utc.tz_localize("UTC")
                    if ts_utc.tzinfo is None
                    else ts_utc.tz_convert("UTC")
                )
                try:
                    active_id = resolve_active_fn(inst_id, as_of_utc=ts_utc.to_pydatetime())
                    if active_id is not None and pd.notna(active_id):
                        active_str = str(active_id).strip()
                        if active_str:
                            mult = _spec_multiplier(active_str)
                except (KeyError, ValueError, LookupError):
                    mult = None

            resolved = float(mult) if mult is not None else 1.0
            if concrete_id or roll_schedule is not None:
                row_cache[cache_key] = resolved
            multipliers.append(resolved)

        return pd.Series(multipliers, index=bars.index, dtype=float)

    def _resolve_traded_value_series(
        self,
        slot: Optional[MarketSlot],
        bars: pd.DataFrame,
    ) -> pd.Series:
        """Resolve per-bar traded value, falling back to ``volume * close * multiplier`` only when ``money`` is missing."""
        if bars.empty:
            return pd.Series(dtype=float)
        has_money_col = "money" in bars.columns
        has_vc = "volume" in bars.columns and "close" in bars.columns
        if not has_money_col and not has_vc:
            return pd.Series(dtype=float)

        if has_money_col:
            raw_money = bars["money"]
            money_num = pd.to_numeric(raw_money, errors="coerce").astype(float)
            money_missing = raw_money.isna() | (
                money_num.isna() & raw_money.astype(str).str.strip().eq("")
            )
            traded = money_num.copy()
        else:
            money_missing = pd.Series(True, index=bars.index, dtype=bool)
            traded = pd.Series(np.nan, index=bars.index, dtype=float)

        if bool(money_missing.any()) and has_vc:
            missing_bars = bars.loc[money_missing]
            vol = pd.to_numeric(missing_bars["volume"], errors="coerce").astype(float)
            close = pd.to_numeric(missing_bars["close"], errors="coerce").astype(float)
            valid_vc = (
                vol.notna()
                & np.isfinite(vol)
                & (vol >= 0)
                & close.notna()
                & np.isfinite(close)
                & (close > 0)
            )
            fallback = pd.Series(np.nan, index=missing_bars.index, dtype=float)
            if bool(valid_vc.any()):
                mult = self._resolve_bar_contract_multipliers(
                    slot, missing_bars.loc[valid_vc]
                )
                fallback.loc[valid_vc] = (
                    vol.loc[valid_vc] * close.loc[valid_vc] * mult
                )
            traded.loc[money_missing] = fallback

        return traded

    def _read_traded_value_bars(
        self,
        stocks: List[str],
        as_of_date_str: str,
        lookback_calendar_days: int,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
    ) -> tuple[pd.DataFrame, Optional[MarketSlot], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Read causally truncated execution bars with valid non-negative finite traded value."""
        slot, _, _ = self._resolve_slot_and_conn(price_db_conn, price_db_path)
        if slot is None:
            return pd.DataFrame(), None, None, None
        as_of_ts = pd.Timestamp(as_of_date_str)
        if not self._is_date_only_key(as_of_date_str):
            as_of_utc: Optional[pd.Timestamp] = (
                as_of_ts.tz_localize("UTC")
                if as_of_ts.tzinfo is None
                else as_of_ts.tz_convert("UTC")
            )
            end_day = self._latest_started_session_day(slot, as_of_utc)
            if end_day is None:
                return pd.DataFrame(), slot, as_of_utc, None
        else:
            as_of_utc = None
            end_day = as_of_ts.normalize()
        start_day = max(
            slot.calendar.start,
            end_day - pd.Timedelta(days=int(lookback_calendar_days)),
        )
        bounds = self._window_utc_bounds(
            slot,
            start_date=start_day.strftime("%Y-%m-%d"),
            end_date=end_day.strftime("%Y-%m-%d"),
        )
        if bounds is None:
            return pd.DataFrame(), slot, as_of_utc, end_day
        bars = slot.bar_feed.read_execution_bars(
            stocks, self._freq_obj, start_utc=bounds[0], end_utc=bounds[1]
        )
        schedule = self._bar_schedule_for_slot(slot)
        if as_of_utc is not None and not bars.empty:
            if "available_at_utc" in bars.columns and bars["available_at_utc"].notna().all():
                avail_utc = pd.to_datetime(bars["available_at_utc"], utc=True)
            elif schedule is not None and "timestamp_utc" in bars.columns:
                ts_utc = pd.to_datetime(bars["timestamp_utc"], utc=True)
                avail_map = {
                    ts: schedule.available_at(
                        pd.Timestamp(ts).tz_convert("UTC").as_unit("ns")
                    )
                    for ts in ts_utc.unique()
                }
                avail_utc = pd.to_datetime(ts_utc.map(avail_map), utc=True)
            elif "timestamp_utc" in bars.columns:
                avail_utc = pd.to_datetime(bars["timestamp_utc"], utc=True)
            else:
                avail_utc = None
            if avail_utc is not None:
                bars = bars.loc[avail_utc <= as_of_utc]
        if bars.empty:
            return pd.DataFrame(), slot, as_of_utc, end_day
        traded = self._resolve_traded_value_series(slot, bars)
        valid = bars.assign(_traded=traded)
        valid = valid.loc[
            valid["_traded"].notna()
            & np.isfinite(valid["_traded"])
            & (valid["_traded"] >= 0)
        ]
        if valid.empty:
            return pd.DataFrame(), slot, as_of_utc, end_day
        if "session_id" in valid.columns and valid["session_id"].notna().all():
            session_ids = valid["session_id"].astype(str)
        else:
            session_ids = pd.to_datetime(valid["timestamp_utc"], utc=True).map(
                lambda ts: self.session_date_for_bar(
                    ts, daily_naive=False, cal=slot.calendar
                ).strftime("%Y-%m-%d")
            )
        return valid.assign(_session_id=session_ids), slot, as_of_utc, end_day

    def average_daily_traded_value(
        self,
        stocks: List[str],
        as_of_date_str: str,
        lookback_days: int,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
        *,
        include_incomplete_session: bool = False,
    ) -> Dict[str, float]:
        """Trailing average daily traded value (ADV) per instrument up to ``as_of_date_str``.

        Execution bars are truncated causally at ``as_of_utc`` (when timestamped),
        summed by ``(instrument_id, session_id)`` to obtain each session's total
        traded value, and averaged over the trailing ``lookback_days`` sessions.

        Incomplete intraday session policy: when ``as_of_date_str`` specifies an
        intraday timestamp strictly earlier than ``end_day``'s final calendar bar,
        the in-progress session is excluded by default
        (``include_incomplete_session=False``) so partial-day volume does not
        understate ADV; pass ``include_incomplete_session=True`` to include the
        observed partial session sum up to ``as_of_utc``.

        Weekly bars (``1w``) carry the full week's traded value on the week's
        final session, so each weekly bar is normalized by the actual number of
        trading sessions in that week before averaging over the trailing
        ``lookback_days`` trading sessions.
        """
        if not stocks:
            return {}
        lookback = int(lookback_days)
        if lookback <= 0:
            raise ValueError(f"lookback_days must be positive, got {lookback_days!r}")
        if self.slot is None and price_db_conn is not None and not _is_duckdb_conn(price_db_conn):
            stocks_str = "', '".join(stocks)
            rows = query_legacy_stub(price_db_conn, "adv", stocks_str, as_of_date_str, lookback)
            return {str(stock_code): float(adv) for stock_code, adv in rows}

        cache_key = (as_of_date_str, lookback, bool(include_incomplete_session))
        cached = self._adv_cache.setdefault(cache_key, {})
        missing = [s for s in stocks if s not in cached]
        if missing:
            try:
                valid, slot, as_of_utc, end_day = self._read_traded_value_bars(
                    missing,
                    as_of_date_str,
                    max(lookback * 4, 90),
                    price_db_conn,
                    price_db_path,
                )
                if not valid.empty and slot is not None:
                    if (
                        not include_incomplete_session
                        and as_of_utc is not None
                        and self._freq_obj.kind == "intraday"
                    ):
                        schedule = self._bar_schedule_for_slot(slot)
                        incomplete_sessions = {
                            str(sid)
                            for sid in valid["_session_id"].unique()
                            if (
                                (
                                    completed_at := self._session_completed_at(
                                        slot, str(sid), schedule
                                    )
                                )
                                is not None
                                and as_of_utc < completed_at
                            )
                        }
                        if incomplete_sessions:
                            # Exclude unfinished sessions so partial-day sums are
                            # never averaged on the same footing as full days.
                            valid = valid.loc[
                                ~valid["_session_id"].astype(str).isin(incomplete_sessions)
                            ]
                    if not valid.empty:
                        daily_totals = (
                            valid.groupby(["instrument_id", "_session_id"], sort=False)[
                                "_traded"
                            ]
                            .sum()
                            .reset_index()
                            .sort_values("_session_id")
                        )
                        if self._freq_obj.kind == "week":
                            session_counts = {
                                str(sid): self._week_session_count(slot.calendar, str(sid))
                                for sid in daily_totals["_session_id"].unique()
                            }
                            for stock_code, grp in daily_totals.groupby(
                                "instrument_id", sort=False
                            ):
                                daily_series: List[float] = []
                                for sid, traded_val in zip(
                                    grp["_session_id"], grp["_traded"]
                                ):
                                    n_sess = session_counts.get(str(sid))
                                    if n_sess is None or n_sess <= 0:
                                        raise ValueError(
                                            f"Cannot determine session count for weekly bar {sid!r}"
                                        )
                                    daily_val = float(traded_val) / float(n_sess)
                                    daily_series.extend([daily_val] * int(n_sess))
                                if daily_series:
                                    cached[str(stock_code)] = float(
                                        np.mean(daily_series[-lookback:])
                                    )
                        else:
                            for stock_code, grp in daily_totals.groupby(
                                "instrument_id", sort=False
                            ):
                                cached[str(stock_code)] = float(
                                    grp["_traded"].tail(lookback).mean()
                                )
            except Exception as exc:
                self.logger.warning(f"[Overlay] ADV query failed at {as_of_date_str}: {exc}")
            for s in missing:
                cached.setdefault(s, None)
        return {s: cached[s] for s in stocks if cached.get(s) is not None}

    def average_daily_value(
        self,
        stocks: List[str],
        as_of_date_str: str,
        lookback_days: int,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
        *,
        include_incomplete_session: bool = False,
    ) -> Dict[str, float]:
        """Backward-compatible alias forwarding to :meth:`average_daily_traded_value`."""
        return self.average_daily_traded_value(
            stocks,
            as_of_date_str,
            lookback_days,
            price_db_conn,
            price_db_path,
            include_incomplete_session=include_incomplete_session,
        )

    def average_bar_traded_value(
        self,
        stocks: List[str],
        as_of_date_str: str,
        lookback_bars: int,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
    ) -> Dict[str, float]:
        """Trailing average single-bar traded value (ABV) per instrument up to ``as_of_date_str``."""
        if not stocks:
            return {}
        lookback = int(lookback_bars)
        if lookback <= 0:
            raise ValueError(f"lookback_bars must be positive, got {lookback_bars!r}")
        if self.slot is None and price_db_conn is not None and not _is_duckdb_conn(price_db_conn):
            if self._freq_obj.kind != "session":
                raise ValueError(
                    "Legacy stub connection only supports daily ('1d') bars for ABV"
                )
            stocks_str = "', '".join(stocks)
            rows = query_legacy_stub(price_db_conn, "adv", stocks_str, as_of_date_str, lookback)
            return {str(stock_code): float(abv) for stock_code, abv in rows}

        cache_key = (as_of_date_str, lookback)
        cached = self._abv_cache.setdefault(cache_key, {})
        missing = [s for s in stocks if s not in cached]
        if missing:
            try:
                if self._freq_obj.kind == "intraday":
                    bpd = self.bars_per_session(price_db_conn, price_db_path)
                    est_days = max(int(np.ceil(lookback / max(bpd, 1.0))) * 4, 90)
                elif self._freq_obj.kind == "week":
                    est_days = max(lookback * 14, 90)
                else:
                    est_days = max(lookback * 4, 90)
                valid, _, _, _ = self._read_traded_value_bars(
                    missing, as_of_date_str, est_days, price_db_conn, price_db_path
                )
                if not valid.empty:
                    valid = valid.sort_values("timestamp_utc")
                    for stock_code, grp in valid.groupby("instrument_id", sort=False):
                        cached[str(stock_code)] = float(
                            grp["_traded"].tail(lookback).mean()
                        )
            except Exception as exc:
                self.logger.warning(f"[Overlay] ABV query failed at {as_of_date_str}: {exc}")
            for s in missing:
                cached.setdefault(s, None)
        return {s: cached[s] for s in stocks if cached.get(s) is not None}

    def liquidity_weight_caps(
        self,
        stocks: List[str],
        as_of_date_str: str,
        *,
        liq_config: Mapping[str, Any],
        raw_liq_config: Optional[Mapping[str, Any]] = None,
        holding_period: Union[int, float],
        capital: float,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
        adv_fetcher: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Resolve liquidity basis/units and compute per-name absolute weight caps."""
        if not stocks or capital <= 0:
            return {}
        raw = raw_liq_config if isinstance(raw_liq_config, Mapping) else {}
        basis_raw = liq_config.get("liquidity_basis", raw.get("liquidity_basis", "adv"))
        unit_raw = liq_config.get(
            "holding_period_unit", raw.get("holding_period_unit", "bars")
        )
        basis = basis_raw.strip().lower() if isinstance(basis_raw, str) else ""
        if basis not in {"adv", "abv"}:
            raise ValueError(
                f"Unsupported liquidity_basis: {basis_raw!r}; expected 'adv' or 'abv'"
            )
        unit = unit_raw.strip().lower() if isinstance(unit_raw, str) else ""
        if unit not in {"bars", "days"}:
            raise ValueError(
                f"Unsupported holding_period_unit: {unit_raw!r}; expected 'bars' or 'days'"
            )

        if basis == "abv":
            raw_abv_lb = liq_config.get("abv_lookback_bars")
            if raw_abv_lb is None:
                raw_abv_lb = raw.get("abv_lookback_bars")
            if raw_abv_lb is not None:
                lookback = int(raw_abv_lb)
                if lookback <= 0:
                    raise ValueError(
                        f"abv_lookback_bars must be positive, got {raw_abv_lb!r}"
                    )
            else:
                lookback = int(liq_config["adv_lookback_days"])
            liq_values = self.average_bar_traded_value(
                stocks, as_of_date_str, lookback, price_db_conn, price_db_path
            )
        else:
            lookback = int(liq_config["adv_lookback_days"])
            if adv_fetcher is not None:
                liq_values = adv_fetcher(stocks, as_of_date_str, lookback)
            else:
                liq_values = self.average_daily_traded_value(
                    stocks, as_of_date_str, lookback, price_db_conn, price_db_path
                )

        bars_per_day = (
            self.bars_per_session(price_db_conn, price_db_path)
            if ((basis == "adv" and unit == "bars") or (basis == "abv" and unit == "days"))
            else None
        )
        return compute_liquidity_weight_caps(
            liq_values,
            stocks,
            max_participation=float(liq_config["max_participation"]),
            holding_period=holding_period,
            capital=capital,
            holding_period_unit=unit,
            bars_per_day=bars_per_day,
            liquidity_basis=basis,
        )

    def style_exposures_for(
        self,
        stocks: List[str],
        as_of_date_str: str,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
    ) -> Optional[pd.DataFrame]:
        """Latest style exposures (<= ``as_of_date_str``) via ``slot.read_aux_features``."""
        if not stocks:
            return None
        if self.slot is None and price_db_conn is not None and not _is_duckdb_conn(price_db_conn):
            try:
                stocks_str = "', '".join(sorted(set(stocks)))
                frame = query_legacy_stub(price_db_conn, "style", stocks_str, as_of_date_str)
            except Exception:
                return None
            if frame.empty:
                return None
            frame = frame.set_index("stock_code").drop(columns=["date"], errors="ignore")
            return frame.reindex(stocks).fillna(0.0)

        if as_of_date_str not in self._style_exposure_cache:
            slot, _, _ = self._resolve_slot_and_conn(price_db_conn, price_db_path)
            if slot is None:
                self._style_exposure_cache[as_of_date_str] = None
                return None
            list_fn = getattr(slot.aux_features, "list_table_features", None)
            features = list_fn("style_factor_exposure") if callable(list_fn) else []
            if not features:
                self._style_exposure_cache[as_of_date_str] = None
                return None
            try:
                end_day = pd.Timestamp(as_of_date_str).normalize()
                start_day = max(slot.calendar.start, end_day - pd.Timedelta(days=365))
                cal_grid = slot.calendar.bar_grid("1d", start_day, end_day)
                if cal_grid.empty:
                    self._style_exposure_cache[as_of_date_str] = None
                    return None
                unique_stocks = sorted(set(stocks))
                grid = (
                    pd.MultiIndex.from_product(
                        [unique_stocks, range(len(cal_grid))],
                        names=["instrument_id", "_pos"],
                    )
                    .to_frame(index=False)
                )
                grid["timestamp_utc"] = cal_grid["timestamp_utc"].to_numpy()[grid["_pos"].to_numpy()]
                grid["session_id"] = cal_grid["session_id"].to_numpy()[grid["_pos"].to_numpy()]
                grid = grid.drop(columns=["_pos"])
                aux = slot.read_aux_features(grid, features=features)
                if aux.empty:
                    self._style_exposure_cache[as_of_date_str] = None
                else:
                    latest_day = str(aux["session_id"].max())
                    latest = aux.loc[aux["session_id"].eq(latest_day)].copy()
                    latest = latest.set_index("instrument_id")
                    rename_map = {col: col.split(".", 1)[-1] for col in features if col in latest.columns}
                    self._style_exposure_cache[as_of_date_str] = latest[list(rename_map)].rename(columns=rename_map)
            except Exception as exc:
                self.logger.debug(f"[Overlay] style exposure query failed at {as_of_date_str}: {exc}")
                self._style_exposure_cache[as_of_date_str] = None

        base = self._style_exposure_cache.get(as_of_date_str)
        if base is None:
            return None
        return base.reindex(stocks).fillna(0.0)

    def industry_groups_for(
        self,
        stocks: List[str],
        as_of_date_str: str,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
    ) -> Dict[str, str]:
        """Latest industry membership (<= ``as_of_date_str``) via ``slot.read_aux_features``."""
        if not stocks:
            return {}
        if self.slot is None and price_db_conn is not None and not _is_duckdb_conn(price_db_conn):
            try:
                stocks_str = "', '".join(sorted(set(stocks)))
                rows = query_legacy_stub(price_db_conn, "industry", stocks_str, as_of_date_str)
                return {str(code): str(ind) for code, ind in rows if ind}
            except Exception:
                return {}

        if as_of_date_str not in self._industry_cache:
            slot, _, _ = self._resolve_slot_and_conn(price_db_conn, price_db_path)
            if slot is None:
                self._industry_cache[as_of_date_str] = {}
                return {}
            try:
                cal_grid = slot.calendar.bar_grid("1d", as_of_date_str, as_of_date_str)
                if cal_grid.empty:
                    self._industry_cache[as_of_date_str] = {}
                else:
                    ts = cal_grid.iloc[0]["timestamp_utc"]
                    sid = cal_grid.iloc[0]["session_id"]
                    unique_stocks = sorted(set(stocks))
                    grid = pd.DataFrame(
                        {
                            "instrument_id": unique_stocks,
                            "timestamp_utc": pd.DatetimeIndex([ts] * len(unique_stocks), tz="UTC"),
                            "session_id": [sid] * len(unique_stocks),
                        }
                    )
                    aux = slot.read_aux_features(grid, features=["industry_membership.sw_l1_code"])
                    col = "industry_membership.sw_l1_code"
                    if aux.empty or col not in aux.columns:
                        self._industry_cache[as_of_date_str] = {}
                    else:
                        self._industry_cache[as_of_date_str] = {
                            str(row.instrument_id): str(getattr(row, "sw_val"))
                            for row in aux.assign(sw_val=aux[col]).itertuples(index=False)
                            if getattr(row, "sw_val")
                        }
            except (ValueError, KeyError, NotImplementedError, LookupError):
                self._industry_cache[as_of_date_str] = {}
            except Exception as exc:
                self.logger.debug(f"[Overlay] industry query failed at {as_of_date_str}: {exc}")
                self._industry_cache[as_of_date_str] = {}
        cached = self._industry_cache[as_of_date_str]
        return {s: cached[s] for s in stocks if s in cached}

    def regime_spread_zscore(
        self,
        rf: Mapping[str, Any],
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
        *,
        benchmark_instrument: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Compute rolling z-score of micro-cap vs benchmark index spread via ``AuxFeatureSlot``."""
        micro_idx = rf.get("micro_index")
        bench_idx = (
            rf.get("benchmark_index")
            or benchmark_instrument
            or self.resolve_benchmark_instrument()
        )

        if self.slot is None and price_db_conn is not None and not _is_duckdb_conn(price_db_conn):
            default_fn = None
            from slots.jq_ashare_local.aux import AShareAuxFeatures

            default_micro, default_bench = AShareAuxFeatures.default_regime_indices()
            micro_idx = micro_idx or default_micro
            bench_idx = bench_idx or default_bench
            cache_key = (micro_idx, bench_idx, rf["spread_window_days"], rf["zscore_window_days"])
            if self._regime_z_cache is not None and self._regime_z_cache[0] == cache_key:
                return self._regime_z_cache[1]
            try:
                frame = query_legacy_stub(price_db_conn, "index", micro_idx, bench_idx)
            except Exception:
                self._regime_z_cache = (cache_key, None)
                return None
            return self._compute_spread_z(frame, micro_idx, bench_idx, rf, cache_key)

        slot, _, _ = self._resolve_slot_and_conn(price_db_conn, price_db_path)
        if slot is None:
            self.logger.warning("[Overlay] regime_filter enabled but no slot/connection; skipping")
            return None

        default_fn = getattr(slot.aux_features, "default_regime_indices", None)
        if callable(default_fn):
            default_micro, default_bench = default_fn()
            micro_idx = micro_idx or default_micro
            bench_idx = bench_idx or default_bench

        if not micro_idx or not bench_idx:
            self.logger.info(
                "[Overlay] regime_filter skipped: no benchmark/micro index configured for slot=%s",
                slot.slot_id,
            )
            return None

        cache_key = (micro_idx, bench_idx, rf["spread_window_days"], rf["zscore_window_days"])
        if self._regime_z_cache is not None and self._regime_z_cache[0] == cache_key:
            return self._regime_z_cache[1]

        try:
            cal_grid = slot.calendar.bar_grid("1d")
            if cal_grid.empty:
                self._regime_z_cache = (cache_key, None)
                return None
            instruments = [str(micro_idx), str(bench_idx)]
            grid = (
                pd.MultiIndex.from_product(
                    [instruments, range(len(cal_grid))],
                    names=["instrument_id", "_pos"],
                )
                .to_frame(index=False)
            )
            grid["timestamp_utc"] = cal_grid["timestamp_utc"].to_numpy()[grid["_pos"].to_numpy()]
            grid["session_id"] = cal_grid["session_id"].to_numpy()[grid["_pos"].to_numpy()]
            grid = grid.drop(columns=["_pos"])
            aux = slot.read_aux_features(grid, features=["index_price_daily.close"])
        except (ValueError, KeyError, NotImplementedError, LookupError):
            self._regime_z_cache = (cache_key, None)
            return None
        except Exception as exc:
            self.logger.warning(f"[Overlay] regime_filter index query failed ({exc}); skipping")
            self._regime_z_cache = (cache_key, None)
            return None

        if aux.empty or "index_price_daily.close" not in aux.columns:
            self._regime_z_cache = (cache_key, None)
            return None
        frame = aux.rename(
            columns={
                "session_id": "date",
                "instrument_id": "index_code",
                "index_price_daily.close": "close",
            }
        )[["date", "index_code", "close"]]
        return self._compute_spread_z(frame, str(micro_idx), str(bench_idx), rf, cache_key)

    def _compute_spread_z(
        self,
        frame: pd.DataFrame,
        micro_idx: str,
        bench_idx: str,
        rf: Mapping[str, Any],
        cache_key: tuple[Any, ...],
    ) -> Optional[pd.Series]:
        if frame.empty:
            self._regime_z_cache = (cache_key, None)
            return None
        wide = frame.pivot(index="date", columns="index_code", values="close").sort_index()
        wide.index = pd.to_datetime(wide.index)
        if micro_idx not in wide.columns or bench_idx not in wide.columns:
            self._regime_z_cache = (cache_key, None)
            return None
        window = int(rf["spread_window_days"])
        z_window = int(rf["zscore_window_days"])
        micro_ret = wide[micro_idx].pct_change(window)
        bench_ret = wide[bench_idx].pct_change(window)
        spread = (micro_ret - bench_ret).dropna()
        rolling_mean = spread.rolling(z_window, min_periods=max(20, z_window // 5)).mean()
        rolling_std = spread.rolling(z_window, min_periods=max(20, z_window // 5)).std()
        z = ((spread - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).dropna()
        self._regime_z_cache = (cache_key, z)
        return z

    def pool_limit_down_fraction(
        self,
        pool_stocks: List[str],
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Daily fraction of pool names closing at limit-down via ``slot.bar_feed``."""
        if not pool_stocks:
            return None
        slot, _, _ = self._resolve_slot_and_conn(price_db_conn, price_db_path)
        if slot is None:
            return None
        try:
            bounds = self._window_utc_bounds(slot, start_date=start_date, end_date=end_date)
            if bounds is None:
                return None
            bars = slot.bar_feed.read_execution_bars(
                pool_stocks, self._freq_obj, start_utc=bounds[0], end_utc=bounds[1]
            )
            if bars.empty or "low_limit" not in bars.columns or "close" not in bars.columns:
                return None
            if "paused" in bars.columns:
                bars = bars.loc[bars["paused"].isna() | (bars["paused"] == 0)]
            bars = bars.loc[bars["low_limit"].notna() & (bars["low_limit"] > 0)]
            if bars.empty:
                return None
            hit = (
                pd.to_numeric(bars["close"], errors="coerce")
                <= pd.to_numeric(bars["low_limit"], errors="coerce") * 1.0001
            ).astype(float)
            grouped = hit.groupby(_session_or_bar_index(bars)).mean()
            return grouped.sort_index()
        except Exception as exc:
            self.logger.warning(f"[Overlay] crisis_guard limit_down query failed ({exc}); skipping")
            return None

    def pool_turnover_ratio(
        self,
        pool_stocks: List[str],
        short_window: int,
        long_window: int,
        price_db_conn: Optional[Any],
        price_db_path: Optional[str],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.Series]:
        """Short-MA / long-MA ratio of pool aggregate traded value via ``slot.bar_feed``."""
        if not pool_stocks:
            return None
        slot, _, _ = self._resolve_slot_and_conn(price_db_conn, price_db_path)
        if slot is None:
            return None
        try:
            bounds = self._window_utc_bounds(slot, start_date=start_date, end_date=end_date)
            if bounds is None:
                return None
            bars = slot.bar_feed.read_execution_bars(
                pool_stocks, self._freq_obj, start_utc=bounds[0], end_utc=bounds[1]
            )
            if bars.empty:
                return None
            if "paused" in bars.columns:
                bars = bars.loc[bars["paused"].isna() | (bars["paused"] == 0)]
            if bars.empty:
                return None
            traded = self._resolve_traded_value_series(slot, bars)
            if traded.empty:
                return None
            valid_mask = traded.notna() & np.isfinite(traded) & (traded >= 0)
            if not bool(valid_mask.any()):
                return None
            bars = bars.loc[valid_mask]
            traded = traded.loc[valid_mask]
            money = traded.groupby(_session_or_bar_index(bars)).sum().sort_index()
            short_ma = money.rolling(short_window, min_periods=short_window).mean()
            long_ma = money.rolling(long_window, min_periods=long_window).mean()
            return (short_ma / long_ma).replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as exc:
            self.logger.warning(f"[Overlay] crisis_guard turnover query failed ({exc}); skipping")
            return None

    def create_factor_loader(
        self,
        *,
        factor_db_path: Optional[str],
        factor_db_conn: Optional[Any],
        shared_cache: Optional[Dict[tuple, pd.DataFrame]],
        price_db_path: Optional[str] = None,
        price_db_conn: Optional[Any] = None,
    ) -> Optional[FactorLoader]:
        """Instantiate ``FactorLoader`` bound to Scheme B store or explicit legacy DB."""
        if self.factor_values_store is None and factor_db_conn is None and not factor_db_path:
            return None
        resolved_slot: Optional[MarketSlot] = self.slot
        if resolved_slot is not None:
            slot_id: Optional[str] = resolved_slot.slot_id
            calendar: Optional[Any] = resolved_slot.calendar
        elif self.slot_id_override is not None:
            slot_id = self.slot_id_override
            calendar = None
        else:
            try:
                resolved_slot, _, _ = self._resolve_slot_and_conn(
                    price_db_conn, price_db_path
                )
            except MarketDbProfileError:
                raise
            except Exception:
                resolved_slot = None
            if resolved_slot is not None:
                slot_id = resolved_slot.slot_id
                calendar = resolved_slot.calendar
            else:
                slot_id = "jq_ashare_local"
                calendar = None
        return FactorLoader(
            duckdb_path=factor_db_path,
            existing_conn=factor_db_conn,
            logger=self.logger,
            shared_cache=shared_cache,
            factor_values_store=self.factor_values_store,
            slot_id=slot_id,
            frequency=self.frequency,
            factor_identities=self.factor_identities,
            calendar=calendar,
            market_db=slot_market_db_identity(resolved_slot),
        )


def compute_long_short_target_weights(
    ranked: pd.Series,
    effective_top_k: int,
    *,
    strategy_weight_map: Optional[Mapping[Any, float]] = None,
) -> Dict[str, float]:
    """Build signed dollar-neutral target weight fractions when ``allow_short=True``."""
    if strategy_weight_map is not None:
        return {str(k): float(v) for k, v in strategy_weight_map.items()}
    ordered = list(ranked.index)
    n_valid = len(ordered)
    if n_valid == 0:
        return {}
    if n_valid == 1:
        sym = str(ordered[0])
        val = float(ranked.iloc[0])
        return {sym: 1.0 if val >= 0 else -1.0}
    leg_k = max(1, min(int(effective_top_k), n_valid // 2))
    longs = ordered[:leg_k]
    shorts = ordered[-leg_k:]
    weights: Dict[str, float] = {}
    for sym in longs:
        weights[str(sym)] = 0.5 / len(longs)
    for sym in shorts:
        weights[str(sym)] = -0.5 / len(shorts)
    return weights


def sanitize_execution_prices(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Coerce ``frame`` to float and mask non-finite or non-positive prices as ``NaN``."""
    if frame is None:
        return None
    numeric = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    vals = numeric.to_numpy(dtype=float, copy=True)
    invalid = ~(np.isfinite(vals) & (vals > 0))
    if invalid.any():
        vals[invalid] = np.nan
    return pd.DataFrame(vals, index=frame.index, columns=frame.columns, dtype=float)


def has_finite_positive_prices(frame: Optional[pd.DataFrame]) -> bool:
    """Return True when ``frame`` contains at least one finite strictly positive price."""
    if frame is None or frame.empty:
        return False
    vals = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=False)
    return bool(np.any(np.isfinite(vals) & (vals > 0)))


def apply_magnitude_caps(
    values: Mapping[str, float],
    caps: Mapping[str, float],
    *,
    tol: float = 1e-12,
) -> Dict[str, float]:
    """Cap signed values at per-name magnitude limits, redistributing same-sign excess."""
    result = {stock: float(val) for stock, val in values.items()}
    if not result or not caps:
        return result
    for _ in range(100):
        over = {
            s: v
            for s, v in result.items()
            if s in caps
            and (
                abs(v) > max(0.0, float(caps[s])) + tol
                or (float(caps[s]) <= 0.0 and abs(v) > 0.0)
            )
        }
        if not over:
            break
        excess_by_sign = {1: 0.0, -1: 0.0}
        for s, v in over.items():
            sign = 1 if v >= 0 else -1
            cap = max(0.0, float(caps[s]))
            excess_by_sign[sign] += max(0.0, abs(v) - cap)
            result[s] = 0.0 if cap == 0.0 else sign * cap
        for sign, excess in excess_by_sign.items():
            if excess <= tol:
                continue
            headroom = {
                s: (caps[s] - abs(result[s])) if s in caps else float("inf")
                for s in result
                if s not in over
                and result[s] * sign > tol
                and (s not in caps or abs(result[s]) < caps[s] - tol)
            }
            free_total = sum(abs(result[s]) for s in headroom)
            if not headroom or free_total <= tol:
                continue
            for s in list(headroom):
                add = excess * (abs(result[s]) / free_total)
                result[s] += sign * min(add, headroom[s])
    return result


def compute_liquidity_weight_caps(
    adv: Mapping[str, float],
    stocks: Iterable[str],
    *,
    max_participation: float,
    holding_period: Union[int, float],
    capital: float,
    holding_period_unit: str = "bars",
    bars_per_day: Optional[float] = None,
    liquidity_basis: str = "adv",
) -> Dict[str, float]:
    """Compute per-name absolute weight caps with explicit ADV/ABV time-unit alignment.

    When ``liquidity_basis == "adv"`` (daily traded value), ``holding_period`` is
    converted to trading days (divided by ``bars_per_day`` when
    ``holding_period_unit == "bars"``). When ``liquidity_basis == "abv"``
    (per-bar traded value), ``holding_period`` is expressed in bars (multiplied by
    ``bars_per_day`` when ``holding_period_unit == "days"``).
    """
    basis = liquidity_basis.strip().lower() if isinstance(liquidity_basis, str) else ""
    if basis not in {"adv", "abv"}:
        raise ValueError(
            f"Unsupported liquidity_basis: {liquidity_basis!r}; expected 'adv' or 'abv'"
        )
    unit = holding_period_unit.strip().lower() if isinstance(holding_period_unit, str) else ""
    if unit not in {"bars", "days"}:
        raise ValueError(
            f"Unsupported holding_period_unit: {holding_period_unit!r}; expected 'bars' or 'days'"
        )
    try:
        hp = float(holding_period)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid holding_period: {holding_period!r}") from exc
    if not np.isfinite(hp) or hp <= 0:
        raise ValueError(f"holding_period must be finite and positive, got {holding_period!r}")
    try:
        participation = float(max_participation)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid max_participation: {max_participation!r}") from exc
    if not np.isfinite(participation) or participation <= 0:
        raise ValueError(
            f"max_participation must be finite and positive, got {max_participation!r}"
        )
    try:
        cap_val = float(capital)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid capital: {capital!r}") from exc
    if not np.isfinite(cap_val):
        raise ValueError(f"capital must be finite, got {capital!r}")
    if cap_val <= 0:
        return {}

    bpd: Optional[float] = None
    if bars_per_day is not None:
        try:
            bpd = float(bars_per_day)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bars_per_day: {bars_per_day!r}") from exc
        if not np.isfinite(bpd) or bpd <= 0:
            raise ValueError(
                f"bars_per_day must be finite and positive, got {bars_per_day!r}"
            )

    if basis == "adv":
        if unit == "days":
            effective_horizon = hp
        else:
            if bpd is None:
                raise ValueError(
                    "bars_per_day is required when liquidity_basis='adv' and holding_period_unit='bars'"
                )
            effective_horizon = hp / bpd
    else:
        if unit == "bars":
            effective_horizon = hp
        else:
            if bpd is None:
                raise ValueError(
                    "bars_per_day is required when liquidity_basis='abv' and holding_period_unit='days'"
                )
            effective_horizon = hp * bpd

    caps: Dict[str, float] = {}
    for stock in sorted(set(stocks)):
        stock_val = adv.get(stock)
        if stock_val is not None:
            val = float(stock_val)
            if np.isfinite(val) and val >= 0:
                caps[stock] = participation * val * effective_horizon / cap_val
    return caps
