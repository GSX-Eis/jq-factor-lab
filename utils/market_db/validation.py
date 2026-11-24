"""Validate the agreement between physical storage and its callers."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import re

import duckdb
import pandas as pd

from utils.session_calendar import VERIFIED_XSHG
from utils.time_axis import Frequency
from utils.time_axis.futures import futures_calendar

from .errors import (
    MARKET_DB_DOMAIN_MISMATCH, MARKET_DB_PROFILE_INVALID, MARKET_DB_PROVIDER_MISMATCH,
    MarketDbProfileError,
)
from .profile import MarketDbProfile
from .storage import read_market_db_profile


def require_market_db_profile(connection: duckdb.DuckDBPyConnection, *, write: bool = False) -> MarketDbProfile:
    profile, _ = read_market_db_profile(connection, write=write)
    return profile


def validate_market_db_provider(profile: MarketDbProfile, provider: str, *, domain: str | None = None) -> None:
    if provider not in profile.allowed_slot_providers:
        raise MarketDbProfileError(MARKET_DB_PROVIDER_MISMATCH,
                                   f'{provider!r} is not allowed by database {profile.db_id!r}')
    if domain is not None and domain != profile.market_domain:
        raise MarketDbProfileError(MARKET_DB_DOMAIN_MISMATCH,
                                   f'Slot domain {domain!r} differs from {profile.market_domain!r}')


def validate_market_db_execution_rules(
    profile: MarketDbProfile, provider: str, *, st_filter: bool, price_limit: bool, t_plus: int,
) -> None:
    for name, implemented in (('st_filter', st_filter), ('price_limit', price_limit), ('t_plus', t_plus)):
        declared = profile.market_rules[name]
        if declared != implemented:
            raise MarketDbProfileError(
                MARKET_DB_PROFILE_INVALID,
                f'{provider!r} does not support market_rules.{name}: profile={declared!r}, implemented={implemented!r}',
            )


def calendar_key(calendar_id: str, *, market_domain: str) -> str:
    if market_domain == 'cn_equity' and calendar_id == VERIFIED_XSHG:
        return 'XSHG'
    if calendar_id == 'CMEGlobex_Equity':
        return 'CME Globex Equity'
    return calendar_id


def validate_market_db_calendar(
    profile: MarketDbProfile, calendar_id: str, timezone: str, *, expected_calendar: str | None = None,
) -> None:
    expected = expected_calendar or profile.calendar_id
    domain = profile.market_domain
    if calendar_key(calendar_id, market_domain=domain) != calendar_key(expected, market_domain=domain):
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                   f'calendar_id {calendar_id!r} does not match {expected!r} for {profile.db_id}')
    if timezone != profile.timezone:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                   f'timezone {timezone!r} does not match {profile.timezone!r} for {profile.db_id}')


def market_db_calendar_bounds(
    profile: MarketDbProfile, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = map(pd.Timestamp, profile.calendar_coverage.split('..'))
    try:
        left = first if start is None else pd.Timestamp(start)
        right = last if end is None else pd.Timestamp(end)
    except (TypeError, ValueError) as exc:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, 'Calendar coverage requires local session dates') from exc
    if any(pd.isna(day) or day.tzinfo is not None or day != day.normalize() for day in (left, right)):
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID, 'Calendar coverage requires local session dates')
    if not first <= left <= right <= last:
        raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                   f'Session range {left.date()}..{right.date()} exceeds {profile.calendar_coverage}')
    return left, right


def validate_market_db_instruments(profile: MarketDbProfile, instruments: Iterable[str]) -> None:
    pattern = re.compile(profile.instrument_id_pattern)
    for instrument in instruments:
        code, _, suffix = instrument.rpartition('.')
        if not code or suffix not in profile.exchanges or pattern.fullmatch(instrument) is None:
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                       f'instrument_id {instrument!r} is not allowed by {profile.db_id}')


def validate_market_db_frequencies(profile: MarketDbProfile, frequencies: Iterable[str]) -> None:
    allowed = {Frequency.parse(value).canonical for value in profile.frequencies}
    for frequency in frequencies:
        if Frequency.parse(frequency).canonical not in allowed:
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                       f'frequency {frequency!r} is not allowed by {profile.db_id}')


def validate_market_db_metadata(
    profile: MarketDbProfile, metadata: pd.DataFrame, *, calendar_name: Callable[[Mapping], str],
) -> None:
    validate_market_db_instruments(profile, metadata.instrument_id)
    for row in metadata.to_dict('records'):
        instrument = row['instrument_id']
        suffix = instrument.rsplit('.', 1)[1]
        exchange = profile.exchange_aliases.get(row['exchange'], row['exchange'])
        if suffix != exchange:
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                       f'exchange {row["exchange"]!r} disagrees with {instrument!r}')
        asset = row['asset_class']
        if asset not in profile.asset_classes:
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                       f'asset_class {asset!r} is not allowed by {profile.db_id}')
        expected = profile.calendar_id
        if profile.market_domain == 'global_futures' and not row.get('calendar_id'):
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                       f'Explicit calendar_id is required for {instrument}')
        if profile.market_domain == 'cn_futures' and expected == 'CN_FUTURES':
            product = futures_calendar(row.get('product_root'))
            expected = product['calendar_id']
            if not expected.startswith(exchange + '_'):
                raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                           f'product calendar {expected!r} disagrees with {instrument!r}')
            if row['rth_reference_calendar'] != product['rth_reference_calendar']:
                raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                           f'RTH calendar disagrees with product calendar for {instrument!r}')
        if row.get('custom_sessions') and profile.calendar_source != 'custom_sessions':
            raise MarketDbProfileError(MARKET_DB_PROFILE_INVALID,
                                       f'custom_sessions is not allowed by {profile.db_id}')
        validate_market_db_calendar(profile, calendar_name(row), row['timezone'], expected_calendar=expected)
