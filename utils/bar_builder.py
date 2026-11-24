"""Compatibility imports for offline Bar builds and audits."""
from .bars.builder import (
    BarCatalog, BarSchedule, CatalogEntry, Frequency, Path, Sequence,
    SessionCalendar, _aggregate, _instrument_ids, _metadata, annotations, audit_bars as _audit_bars,
    build_bars as _build_bars, calendar_name, code_version, create_bar_table, duckdb, hashlib,
    identifier, import_transaction, instrument_calendar, pd, register_entry, replace,
    scan_quality, slot_mapping, store_instrument_metadata, table_exists, uuid, write_quality,
)
from .bars.source import MarketBarReader


build_bars = _build_bars


def audit_bars(connection: duckdb.DuckDBPyConnection, **kwargs) -> dict:
    from slots.jq_ashare_local.panel import ASharePanelReader

    return _audit_bars(connection, reader=ASharePanelReader(connection), **kwargs)
