"""Compatibility imports for the catalog-backed Bar reader."""
from .bars.source import (
    TYPE_CHECKING, BarCatalog, BarSource, DataAccessConfig, Frequency, Mapping,
    MarketBarReader, MissingBarDataError, Path, Sequence, _instrument_ids, _utc,
    annotations, duckdb, identifier, pd, table_exists, warn_quality,
)
