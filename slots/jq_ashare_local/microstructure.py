"""A-share execution costs, trading status and sell eligibility."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from utils.bars.catalog import identifier, table_exists
from utils.session_calendar import SessionCalendar
from utils.slots import ExecutionProfile

from .feed import PRICE_TABLES, table_columns
from .panel import TIMEZONE


class AShareMicrostructure:
    def __init__(self, connection: duckdb.DuckDBPyConnection, calendar: SessionCalendar, *,
                 commission_bps: float | None = None, sell_tax_bps: float | None = None,
                 slippage_bps: float = 0.0,
                 benchmark_instrument: str | None = None) -> None:
        self.has_explicit_fees = commission_bps is not None or sell_tax_bps is not None
        self.connection = connection
        self.calendar = calendar
        self.execution_profile = ExecutionProfile(
            'close', 'close', 0, 0, 'decimal', 'endpoints_only', 'cross_session',
            allow_short=False,
            commission_bps=0.0 if commission_bps is None else commission_bps,
            sell_tax_bps=0.0 if sell_tax_bps is None else sell_tax_bps,
            slippage_bps=slippage_bps, benchmark_instrument=benchmark_instrument)

    def _daily(self, table: str, fields: tuple[str, ...], instrument: str,
               days: list[str]) -> pd.DataFrame:
        if not table_exists(self.connection, table):
            return pd.DataFrame(columns=list(fields))
        present = table_columns(self.connection, table)
        selected = [field for field in fields if field in present]
        if not selected:
            return pd.DataFrame(columns=list(fields))
        columns = ', '.join(
            f"CAST({identifier(field)} AS {'BOOLEAN' if field in ('paused', 'is_st') else 'DOUBLE'}) "
            f"AS {identifier(field)}" for field in selected)

        rows = self.connection.execute(
            f'SELECT CAST(date AS DATE)::VARCHAR AS session_id, {columns} FROM {identifier(table)} '
            'WHERE stock_code=? AND CAST(date AS DATE) IN (SELECT unnest(?)::DATE)',
            [instrument, days]).fetchdf()
        return rows.set_index('session_id', verify_integrity=True).reindex(columns=list(fields))

    def _minute_close(self, timestamps: pd.Series, instrument: str) -> pd.Series:
        table = PRICE_TABLES['5min']
        if timestamps.isna().all() or not table_exists(self.connection, table):
            return pd.Series(float('nan'), index=timestamps.index)
        labels = (timestamps.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
                  + pd.Timedelta(minutes=5))
        rows = self.connection.execute(
            f'SELECT CAST(date AS TIMESTAMP) AS date, CAST("close" AS DOUBLE) AS "close" '
            f'FROM {identifier(table)} WHERE stock_code=? AND date IN (SELECT unnest(?)::TIMESTAMP)',
            [instrument, pd.DatetimeIndex(labels.dropna().drop_duplicates()).to_pydatetime().tolist()]).fetchdf()
        return labels.map(rows.set_index('date', verify_integrity=True).close)

    def is_st_on(self, session_id: str, instrument_id: str) -> bool:
        """Return True if ``historical_st_daily`` marks ``instrument_id`` as ST on ``session_id``."""
        st = self._daily('historical_st_daily', ('is_st',), instrument_id, [session_id]).is_st
        return bool(st.eq(True).any()) if not st.empty else False

    def tradability_mask(self, grid: pd.DataFrame, *, instrument_id: str,
                         exclude_st: bool = True) -> pd.Series:
        """Use optional side, observed close and entry_session_id for direction and T+1."""
        if grid.empty:
            return pd.Series(index=grid.index, dtype=bool)
        days = list(grid.session_id.unique())
        prices = self._daily(PRICE_TABLES['1d'], ('close', 'high_limit', 'low_limit', 'paused'),
                             instrument_id, days)
        paused = prices.paused
        for table in ('all_a_paused_daily', 'regime_all_a_paused_daily_v1'):
            flags = self._daily(table, ('paused',), instrument_id, days).paused
            if not flags.empty:
                paused = flags.combine_first(paused)
        st = self._daily('historical_st_daily', ('is_st',), instrument_id, days).is_st if exclude_st else pd.Series(dtype=object)

        halted = grid.session_id.map(paused).eq(True).fillna(False)
        special_treatment = grid.session_id.map(st).eq(True).fillna(False) if exclude_st else False
        if 'close' in grid:
            close = grid.close
        else:
            closes = self.calendar.bar_grid('1d', min(days), max(days)).set_index('session_id').timestamp_utc
            daily = grid.timestamp_utc.eq(grid.session_id.map(closes))
            close = grid.session_id.map(prices.close).where(daily)
            minutes = self._minute_close(grid.timestamp_utc.where(~daily), instrument_id)
            close = close.fillna(minutes)
        close = pd.to_numeric(close, errors='coerce')
        valid_price = (np.isfinite(close) & close.gt(0)).fillna(False)
        mask = valid_price & ~(halted | special_treatment)
        if 'instrument_id' in grid:
            mask &= grid.instrument_id.eq(instrument_id)
        side = grid.get('side', pd.Series('', index=grid.index))
        if not side.isin(('', 'buy', 'sell')).all():
            raise ValueError("side must be 'buy' or 'sell'")
        if 'entry_session_id' in grid:
            entry = grid.entry_session_id
            mask &= side.eq('buy') | entry.isna() | entry.lt(grid.session_id)
        mask &= side.eq('sell') | ~close.ge(grid.session_id.map(prices.high_limit))
        mask &= side.eq('buy') | ~close.le(grid.session_id.map(prices.low_limit))
        return mask.astype(bool)
