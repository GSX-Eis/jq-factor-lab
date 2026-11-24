"""A-share instrument metadata and date-effective stock pools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import duckdb
import pandas as pd

from backtest_and_analysis.time_series.contracts import _utc
from utils.bars.catalog import table_exists
from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter
from utils.session_calendar import SessionCalendar
from utils.slots import InstrumentSpec

from .feed import PRICE_TABLES, local_day


@dataclass(frozen=True)
class UniverseDefinition:
    filters: tuple[StockFilter, ...]
    max_stocks: int | None = None
    sort_by: str | None = None
    ascending: bool = False


def _index_pool(index_code: str) -> tuple[StockFilter, ...]:
    return (StockFilter('paused', '==', 0), StockFilter('is_st', '==', 0),
            StockFilter('index_code', '==', index_code))


_BUILTIN_UNIVERSES = {
    'wind_micro': UniverseDefinition(
        (*_index_pool('8841431.WI'), StockFilter('is_new_stock', '==', 0),
         StockFilter('val_market_cap', 'percentile_between', (20, 80))), max_stocks=500),
    'csi300': UniverseDefinition(_index_pool('000300.XSHG')),
    'csi500': UniverseDefinition(_index_pool('000905.XSHG')),
}


class AShareUniverse:
    def __init__(self, connection: duckdb.DuckDBPyConnection, calendar: SessionCalendar,
                 universes: Mapping[str, UniverseDefinition] | None = None) -> None:
        self.connection = connection
        self.calendar = calendar
        self.universes = {**_BUILTIN_UNIVERSES, **(universes or {})}
        self.manager = DynamicStockPoolManager(connection, price_table=PRICE_TABLES['1d'], data_bindings={})

    def resolve_universe(self, universe_id: str, *, as_of_utc: datetime) -> list[str]:
        _utc(as_of_utc, 'as_of_utc')
        anchor = pd.Timestamp(as_of_utc)
        day = local_day(anchor)
        self.calendar.validate_coverage(day, day)
        # The existing selector can read same-day state and price fields.
        session = self.calendar.session_on(day)
        if session is None or anchor < session.segments[-1][1]:
            prior = self.calendar.sessions(end=day)
            prior = prior[prior < day]
            if prior.empty:
                return []
            day = prior[-1]
        definition = self.universes.get(universe_id)
        if definition is None:
            # Explicit index lookups are date-effective memberships without additional filters.
            return self.manager._read_index_constituents(universe_id, day.strftime('%Y-%m-%d'))
        return self.manager._select_stock_pool(
            day.to_pydatetime(), list(definition.filters), max_stocks=definition.max_stocks,
            sort_by=definition.sort_by, ascending=definition.ascending)

    def instrument_spec(self, instrument_id: str) -> InstrumentSpec:
        defaults = dict(asset_class='equity', currency='CNY', lot_size=100, contract_multiplier=1.0)
        if table_exists(self.connection, 'security_master'):
            rows = self.connection.execute(
                'SELECT * FROM security_master WHERE stock_code = ?', [instrument_id]).fetchdf()
            if len(rows) > 1:
                raise ValueError(f'Duplicate security_master rows for {instrument_id}')
            if not rows.empty:
                row = rows.iloc[0]
                for field in defaults.keys() & rows.columns:
                    if pd.notna(row[field]):
                        defaults[field] = row[field]
                for field, column in [('listing_date', 'start_date'), ('expiry_date', 'end_date')]:
                    value = row.get(field, row.get(column))
                    if pd.notna(value):
                        defaults[field] = pd.Timestamp(value).date()
        return InstrumentSpec(instrument_id=instrument_id, **defaults)

    def resolve_active_contract(self, instrument_id: str, *, as_of_utc: datetime) -> str:
        _utc(as_of_utc, 'as_of_utc')
        return instrument_id
