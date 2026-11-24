"""Market-agnostic universe filter plugin protocol and slot dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from utils.slots.contracts import MarketSlot


class UniverseFilterError(RuntimeError):
    """Raised when an enabled universe filter fails or a required plugin is absent."""

    def __init__(self, message: str, *, code: str = "UNIVERSE_FILTER_ERROR") -> None:
        super().__init__(message)
        self.code = code


@runtime_checkable
class UniverseFilterPlugin(Protocol):
    """Optional market-specific universe/quality filter plugin mounted on a slot."""

    def filter_universe(
        self,
        dynamic_pools: Mapping[str, list[str]],
        *,
        config: Mapping[str, Any],
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        """Filter rebalance candidate pools and return (filtered_pools, diagnostics)."""
        ...


def resolve_universe_filter_plugin(
    slot: MarketSlot | None = None,
    *,
    plugin: UniverseFilterPlugin | None = None,
) -> UniverseFilterPlugin | None:
    """Resolve an explicit UniverseFilterPlugin or one mounted on ``slot.aux_features``."""
    if plugin is not None:
        if not isinstance(plugin, UniverseFilterPlugin) or not callable(
            getattr(plugin, "filter_universe", None)
        ):
            raise TypeError("plugin must implement UniverseFilterPlugin.filter_universe")
        return plugin

    if slot is None:
        return None

    aux = slot.aux_features
    candidate = getattr(aux, "universe_filter_plugin", None)
    if candidate is None:
        factory = getattr(aux, "get_universe_filter_plugin", None)
        if callable(factory):
            candidate = factory()
    if candidate is None:
        return None
    if not isinstance(candidate, UniverseFilterPlugin) or not callable(
        getattr(candidate, "filter_universe", None)
    ):
        raise TypeError(
            "slot.aux_features.universe_filter_plugin must implement UniverseFilterPlugin"
        )
    return candidate


def apply_universe_filter(
    dynamic_pools: Mapping[str, list[str]],
    *,
    config: Mapping[str, Any],
    slot: MarketSlot | None = None,
    plugin: UniverseFilterPlugin | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Dispatch universe filtering to an attached plugin or pass through on non-plugin slots."""
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return {str(k): list(v or []) for k, v in dynamic_pools.items()}, {
            "enabled": False,
            "qualified_counts": {},
        }

    resolved = resolve_universe_filter_plugin(slot, plugin=plugin)
    if resolved is None:
        if bool(cfg.get("required", False)) or bool(cfg.get("strict_plugin", False)):
            slot_id = slot.slot_id if slot is not None else "none"
            raise UniverseFilterError(
                f"UNIVERSE_FILTER_PLUGIN_MISSING: quality filter is required for slot={slot_id!r}, "
                "but no UniverseFilterPlugin is attached",
                code="UNIVERSE_FILTER_PLUGIN_MISSING",
            )
        passthrough = {str(k): list(v or []) for k, v in dynamic_pools.items()}
        return passthrough, {
            "enabled": True,
            "plugin_present": False,
            "skipped_no_plugin": True,
            "qualified_counts": {k: len(v) for k, v in passthrough.items()},
        }

    try:
        filtered, diagnostics = resolved.filter_universe(dynamic_pools, config=cfg)
    except UniverseFilterError:
        raise
    except RuntimeError as exc:
        raise UniverseFilterError(str(exc), code="UNIVERSE_FILTER_FAILED") from exc
    diag_out = dict(diagnostics or {})
    diag_out.setdefault("plugin_present", True)
    return filtered, diag_out


__all__ = [
    "UniverseFilterError",
    "UniverseFilterPlugin",
    "apply_universe_filter",
    "resolve_universe_filter_plugin",
]
