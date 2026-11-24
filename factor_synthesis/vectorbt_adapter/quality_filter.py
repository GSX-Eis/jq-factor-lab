"""Slot-delegated point-in-time universe quality eligibility for top-k selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from slots.jq_ashare_local import build_jq_ashare_local
from utils.data_store import connect_read_only as _connect_data_read_only
from utils.slots.contracts import MarketSlot

from .universe_filter import (
    UniverseFilterError,
    UniverseFilterPlugin,
    apply_universe_filter,
)


class QualityFilterError(UniverseFilterError):
    """Raised when an enabled quality filter cannot produce a valid mask."""


def apply_pit_quality_filter(
    dynamic_pools: Mapping[str, list[str]],
    *,
    db_path: str | None = None,
    slot: MarketSlot | None = None,
    plugin: UniverseFilterPlugin | None = None,
    config: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Filter each rebalance pool via ``slot`` / ``UniverseFilterPlugin``.

    Legacy callers supplying only ``db_path`` automatically construct a read-only
    ``jq_ashare_local`` ``MarketSlot`` and delegate to its attached plugin.
    """
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return {str(k): list(v or []) for k, v in dynamic_pools.items()}, {
            "enabled": False,
            "qualified_counts": {},
        }

    if plugin is not None or slot is not None:
        try:
            return apply_universe_filter(
                dynamic_pools,
                config=cfg,
                slot=slot,
                plugin=plugin,
            )
        except UniverseFilterError as exc:
            raise QualityFilterError(str(exc), code=exc.code) from exc

    if not db_path:
        raise QualityFilterError(
            "UNIVERSE_FILTER_INPUT_MISSING: quality filter requires slot, plugin, or db_path",
            code="UNIVERSE_FILTER_INPUT_MISSING",
        )

    conn = _connect_data_read_only(str(db_path))
    try:
        ashare_slot = build_jq_ashare_local(conn)
        return apply_universe_filter(
            dynamic_pools,
            config=cfg,
            slot=ashare_slot,
        )
    except UniverseFilterError as exc:
        raise QualityFilterError(str(exc), code=exc.code) from exc
    finally:
        conn.close()


__all__ = ["QualityFilterError", "apply_pit_quality_filter"]
