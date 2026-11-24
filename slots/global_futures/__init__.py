"""Global futures slot with explicitly declared product calendars."""

from typing import Unpack

import duckdb

from slots.futures.slot import FuturesOptions, build_futures_slot
from utils.slots.contracts import MarketSlot


def build_global_futures_slot(connection: duckdb.DuckDBPyConnection, *, product_root: str,
                              start: str | None = None, end: str | None = None, slot_id: str = 'global_futures', currency: str = 'USD',
                              **options: Unpack[FuturesOptions]) -> MarketSlot:
    return build_futures_slot(connection, provider='global_futures', slot_id=slot_id, product_root=product_root,
                              start=start, end=end, currency=currency, **options)
