"""Read-only JoinQuant A-share MarketSlot."""

from __future__ import annotations

from collections.abc import Mapping

import duckdb

from utils.market_db.validation import (
    calendar_key, market_db_calendar_bounds, require_market_db_profile, validate_market_db_calendar,
    validate_market_db_execution_rules, validate_market_db_provider,
)
from utils.session_calendar import SessionCalendar, VERIFIED_XSHG
from utils.slots import ExecutionProfile, MarketSlot, ProviderConfig, SlotConfig

from .aux import AShareAuxFeatures
from .feed import AShareBarFeed
from .microstructure import AShareMicrostructure
from .universe import AShareUniverse, UniverseDefinition



def daily_slot_config(*, start: str | None = None, end: str | None = None) -> SlotConfig:
    provider = 'jq_ashare_local'
    component = ProviderConfig(provider)
    bounds = {'calendar_id': VERIFIED_XSHG, 'timezone': 'Asia/Shanghai'}
    bounds.update({key: value for key, value in (('start', start), ('end', end)) if value is not None})
    return SlotConfig(
        1, provider, ProviderConfig(provider, bounds), component, component, component, component,
        ExecutionProfile('close', 'close', 0, 0, 'decimal', 'endpoints_only', 'cross_session'),
    )

def build_jq_ashare_local(
    connection: duckdb.DuckDBPyConnection, *, start: str | None = None, end: str | None = None,
    calendar_id: str = 'XSHG', timezone: str = 'Asia/Shanghai',
    universes: Mapping[str, UniverseDefinition] | None = None,
    commission_bps: float | None = None, sell_tax_bps: float | None = None, slippage_bps: float = 0.0,
    benchmark_instrument: str | None = None,
    calendar: SessionCalendar | None = None,
) -> MarketSlot:
    """Use a caller-owned DuckDB connection and the cross-checked XSHG calendar."""
    profile = require_market_db_profile(connection)
    validate_market_db_provider(profile, 'jq_ashare_local', domain='cn_equity')
    validate_market_db_execution_rules(profile, 'jq_ashare_local', st_filter=True, price_limit=True, t_plus=1)
    validate_market_db_calendar(profile, 'XSHG', 'Asia/Shanghai')
    validate_market_db_calendar(profile, calendar_id, timezone)
    left, right = market_db_calendar_bounds(profile, start, end)
    if calendar is not None:
        validate_market_db_calendar(profile, calendar.calendar_id, calendar.timezone)
        market_db_calendar_bounds(profile, calendar.start, calendar.end)
        calendar.validate_coverage(left, right)
        resolved_calendar = calendar
    else:
        resolved_calendar = SessionCalendar.from_pmc(
            left, right, template_calendar=calendar_key(profile.calendar_id, market_domain=profile.market_domain),
        )
    return MarketSlot(
        'jq_ashare_local', resolved_calendar, AShareBarFeed(connection, resolved_calendar),
        AShareUniverse(connection, resolved_calendar, universes), AShareAuxFeatures(connection, resolved_calendar, market_db_profile=profile),
        AShareMicrostructure(connection, resolved_calendar, commission_bps=commission_bps,
                             sell_tax_bps=sell_tax_bps, slippage_bps=slippage_bps,
                             benchmark_instrument=benchmark_instrument),
        market_db_profile=profile,
    )


__all__ = ['UniverseDefinition', 'build_jq_ashare_local']
