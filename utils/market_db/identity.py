"""Canonical, path-free provenance for a physical market database."""
from __future__ import annotations

import re
from collections.abc import Mapping

from .profile import MarketDbProfile, is_market_db_id

MARKET_DB_FIELDS = ("db_id", "profile_hash", "market_domain")


def market_db_identity(source: object) -> dict[str, str]:
    if isinstance(source, MarketDbProfile):
        return {key: getattr(source, key) for key in MARKET_DB_FIELDS}
    if not isinstance(source, Mapping) or not any(key in source for key in MARKET_DB_FIELDS):
        return {}
    db_id, digest, domain = (source.get(key) for key in MARKET_DB_FIELDS)
    if not is_market_db_id(db_id):
        raise ValueError("Market database identity requires a valid db_id")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Market database identity requires a lowercase SHA-256 profile_hash")
    if not isinstance(domain, str) or not domain or domain != domain.strip():
        raise ValueError("Market database identity requires a market_domain")
    return dict(zip(MARKET_DB_FIELDS, (db_id, digest, domain)))


def bind_market_db_fingerprint(
    fingerprint: str | Mapping[str, object] | None, identity: Mapping[str, str],
) -> str | Mapping[str, object] | None:
    if not identity:
        return fingerprint
    existing = market_db_identity(fingerprint)
    if existing and existing != identity:
        raise ValueError("slot_data_fingerprint disagrees with the market database identity")
    fields = (dict(fingerprint) if isinstance(fingerprint, Mapping)
              else {"revision": fingerprint} if fingerprint is not None else {})
    return {**fields, **identity}


def slot_market_db_identity(slot: object) -> dict[str, str]:
    fingerprint = market_db_identity(getattr(slot, "slot_data_fingerprint", None))
    profile = market_db_identity(getattr(slot, "market_db_profile", None))
    if profile and fingerprint and profile != fingerprint:
        raise ValueError("MarketSlot profile disagrees with slot_data_fingerprint")
    return profile or fingerprint
