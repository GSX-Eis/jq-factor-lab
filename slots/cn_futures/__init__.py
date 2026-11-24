"""Domestic futures slot with product calendars and CNY execution."""

from typing import Unpack

import duckdb

from slots.futures.slot import FuturesOptions, build_futures_slot
from utils.slots.contracts import MarketSlot


def build_cn_futures_slot(connection: duckdb.DuckDBPyConnection, *, product_root: str,
                          start: str | None = None, end: str | None = None, slot_id: str = 'cn_futures', currency: str = 'CNY',
                          **options: Unpack[FuturesOptions]) -> MarketSlot:
    return build_futures_slot(connection, provider='cn_futures', slot_id=slot_id, product_root=product_root,
                              start=start, end=end, currency=currency, **options)
