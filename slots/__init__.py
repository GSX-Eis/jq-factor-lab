"""Construct and register explicitly configured built-in market slots."""
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import duckdb

from utils.market_db.errors import MARKET_DB_DOMAIN_MISMATCH, MarketDbProfileError
from utils.slots import MarketSlot, SlotConfig, SlotRegistry, load_slot_config

_NATIVE_DOMAINS = {'jq_ashare_local': 'cn_equity', 'cn_futures': 'cn_futures', 'global_futures': 'global_futures'}


def build_slot_from_config(config: SlotConfig | str | Path, *,
                           connection: duckdb.DuckDBPyConnection | None = None,
                           base_dir: str | Path | None = None) -> MarketSlot:
    """Build a slot from a config or a YAML/JSON path. Native providers borrow connection."""
    if not isinstance(config, SlotConfig):
        path = Path(config).expanduser().resolve()
        config = load_slot_config(path)
        if base_dir is None:
            base_dir = path.parent
    provider = config.bar_feed.provider
    if provider not in _NATIVE_DOMAINS:
        from .declarative import build_declarative_slot

        return build_declarative_slot(config, base_dir=Path(base_dir or '.').expanduser().resolve())

    if connection is None:
        raise ValueError(f'{provider} requires a caller-owned DuckDB connection')
    if config.market_domain is not None and config.market_domain != _NATIVE_DOMAINS[provider]:
        raise MarketDbProfileError(MARKET_DB_DOMAIN_MISMATCH,
                                   f'{provider!r} cannot represent {config.market_domain!r}')
    for name in ('calendar', 'universe', 'aux_features', 'microstructure'):
        component = getattr(config, name)
        if component.provider != provider or (name != 'calendar' and component.options):
            raise ValueError(f'{name} must select the native {provider} component')
    if config.calendar.options.keys() - {'start', 'end', 'calendar_id', 'timezone'}:
        raise ValueError('Native calendar options may only declare start, end, calendar_id and timezone')
    options = dict(config.bar_feed.options)
    if options.keys() & {'start', 'end', 'slot_id', 'commission_bps', 'slippage_bps', 'sell_tax_bps',
                         'calendar', 'calendar_id', 'timezone'}:
        raise ValueError('Declare calendar coverage, slot ID and execution fees in their own sections')
    options.update(config.calendar.options)
    profile = config.execution_profile
    options.update(commission_bps=profile.commission_bps, slippage_bps=profile.slippage_bps)
    if provider == 'jq_ashare_local':
        from .jq_ashare_local import build_jq_ashare_local

        slot = replace(build_jq_ashare_local(connection, sell_tax_bps=profile.sell_tax_bps, **options),
                       slot_id=config.slot_id)
    else:
        from .cn_futures import build_cn_futures_slot
        from .global_futures import build_global_futures_slot

        builder = build_cn_futures_slot if provider == 'cn_futures' else build_global_futures_slot
        slot = builder(connection, slot_id=config.slot_id, **options)
    if slot.microstructure.execution_profile != profile:
        raise ValueError(f'execution_profile must match the {provider} native convention')
    return replace(slot, slot_config_hash=config.slot_config_hash)


def register_builtin_slots(registry: SlotRegistry, configs: Iterable[SlotConfig | str | Path], *,
                           connection: duckdb.DuckDBPyConnection | None = None,
                           base_dir: str | Path | None = None) -> None:
    """Register explicitly supplied instances of any built-in provider."""
    for config in configs:
        registry.register(build_slot_from_config(config, connection=connection, base_dir=base_dir))


__all__ = ['build_slot_from_config', 'register_builtin_slots']
