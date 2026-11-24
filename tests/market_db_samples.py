"""Explicit database declarations for synthetic DuckDB fixtures."""
from dataclasses import replace
from pathlib import Path

from utils.market_db import load_market_db_profile_yaml
from utils.market_db.storage import _create_profile_table


def market_profile(domain, **overrides):
    path = Path(__file__).resolve().parents[1] / 'config' / 'market_databases' / f'{domain}.yaml'
    profile = load_market_db_profile_yaml(path)
    return replace(profile, profile_hash='', **overrides) if overrides else profile


def declare_market_db(connection, domain, **overrides):
    profile = market_profile(domain, **overrides)
    _create_profile_table(connection, profile)
    return profile


def crypto_profile(**overrides):
    return market_profile('us_equity', **{
        'market_domain': 'custom:crypto',
        'db_id': 'crypto_fixture',
        'allowed_slot_providers': ('csv_files', 'duckdb_table'),
        'asset_classes': ('crypto',),
        'timezone': 'UTC',
        'calendar_id': 'crypto_utc',
        'calendar_source': 'custom_sessions',
        'calendar_coverage': '2026-01-01..2026-01-10',
        'exchanges': ('SPOT',),
        'exchange_aliases': {},
        'instrument_id_pattern': r'^[A-Z]+-USD\.SPOT$',
        'frequencies': ('30min', '1h', '2h'),
        **overrides,
    })


def declare_equity_bars(connection):
    return declare_market_db(connection, 'cn_equity',
                             frequencies=('1min', '5min', '15min', '30min', '60min', '1d', '1w'))


def declare_futures_bars(connection, exchange, **overrides):
    domain = 'cn_futures' if exchange in ('SHFE', 'INE', 'DCE', 'CZCE', 'CFFEX', 'GFEX') else 'global_futures'
    return declare_market_db(connection, domain, **{
        'frequencies': ('1min', '5min', '15min', '30min', '60min', '1d', '1w'),
        **overrides,
    })
