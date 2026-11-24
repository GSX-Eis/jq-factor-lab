"""Versioned market database declarations and their content identity."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from utils.time_axis.frequency import Frequency

from .errors import MARKET_DB_PROFILE_HASH_MISMATCH, MARKET_DB_PROFILE_INVALID, MarketDbProfileError

_MARKET_DOMAINS = {"cn_equity", "cn_futures", "global_futures", "us_equity"}
_DB_ID = re.compile(r"[a-z][a-z0-9_-]*")
_DATE_RANGE = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})\.\.([0-9]{4}-[0-9]{2}-[0-9]{2})")
_UTC_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|\+00:00)")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LIST_FIELDS = ("allowed_slot_providers", "asset_classes", "exchanges", "frequencies", "session_views")

RuleValue = bool | int | float | Mapping[str, int]


def is_market_db_id(value: object) -> bool:
    return isinstance(value, str) and _DB_ID.fullmatch(value) is not None


def _invalid(message: str) -> MarketDbProfileError:
    return MarketDbProfileError(MARKET_DB_PROFILE_INVALID, message)


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _invalid(f"{label} must be a nonempty string without surrounding whitespace")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise _invalid(f"{label} must be valid Unicode") from exc
    if "\x00" in value:
        raise _invalid(f"{label} must not contain NUL")


def _mapping(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise _invalid(f"{label} must be an object")
    for key in value:
        _text(key, f"{label} key")


def _validated_market_rules(value: Mapping[str, RuleValue], frequencies: set[str]) -> Mapping[str, RuleValue]:
    _mapping(value, "market_rules")
    for name in ("st_filter", "price_limit"):
        if type(value.get(name)) is not bool:
            raise _invalid(f"market_rules.{name} must be a boolean")
    t_plus = value.get("t_plus")
    if type(t_plus) is not int or t_plus < 0:
        raise _invalid("market_rules.t_plus must be a nonnegative integer")
    rules = dict(value)
    for name, setting in rules.items():
        if name == "bars_per_session":
            _mapping(setting, "market_rules.bars_per_session")
            seen = set()
            for frequency, bars in setting.items():
                canonical = Frequency.parse(frequency).canonical
                if canonical not in frequencies or canonical in seen or type(bars) is not int or bars < 1:
                    raise _invalid("bars_per_session requires distinct declared frequencies with positive counts")
                seen.add(canonical)
            rules[name] = MappingProxyType(dict(setting))
        elif not (type(setting) in (bool, int) or (type(setting) is float and isfinite(setting))):
            raise _invalid(f"market_rules.{name} must be a boolean or a finite number")
    return MappingProxyType(rules)


@dataclass(frozen=True)
class MarketDbProfile:
    schema_version: int
    db_id: str
    market_domain: str
    allowed_slot_providers: tuple[str, ...]
    asset_classes: tuple[str, ...]
    timezone: str
    calendar_id: str
    calendar_source: str
    calendar_coverage: str
    exchanges: tuple[str, ...]
    exchange_aliases: Mapping[str, str]
    instrument_id_pattern: str
    currency: str
    frequencies: tuple[str, ...]
    session_views: tuple[str, ...]
    market_rules: Mapping[str, RuleValue]
    default_execution_profile: str
    created_at_utc: str
    profile_hash: str = ""

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise _invalid("schema_version must be 1")
        if not is_market_db_id(self.db_id):
            raise _invalid(f"Invalid db_id: {self.db_id!r}")
        for name in ("market_domain", "timezone", "calendar_id", "calendar_source", "calendar_coverage",
                     "instrument_id_pattern", "currency", "default_execution_profile", "created_at_utc"):
            _text(getattr(self, name), name)
        if self.market_domain not in _MARKET_DOMAINS and not (
            self.market_domain.startswith("custom:") and is_market_db_id(self.market_domain[7:])
        ):
            raise _invalid(f"Unsupported market_domain: {self.market_domain!r}")
        try:
            ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise _invalid(f"Invalid IANA timezone: {self.timezone!r}") from exc
        coverage = _DATE_RANGE.fullmatch(self.calendar_coverage)
        try:
            if not coverage or date.fromisoformat(coverage[1]) > date.fromisoformat(coverage[2]):
                raise ValueError
        except ValueError as exc:
            raise _invalid("calendar_coverage must be YYYY-MM-DD..YYYY-MM-DD with start <= end") from exc
        try:
            re.compile(self.instrument_id_pattern)
        except (re.error, OverflowError, RecursionError) as exc:
            raise _invalid(f"Invalid instrument_id_pattern: {exc}") from exc
        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise _invalid("currency must be a three-letter uppercase code")
        try:
            if (not _UTC_TIMESTAMP.fullmatch(self.created_at_utc)
                    or datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00")).utcoffset() != timedelta(0)):
                raise ValueError
        except ValueError as exc:
            raise _invalid("created_at_utc must be an ISO 8601 timestamp in UTC") from exc

        for name in _LIST_FIELDS:
            items = getattr(self, name)
            if not isinstance(items, (list, tuple)) or not items:
                raise _invalid(f"{name} must be a nonempty list")
            for item in items:
                _text(item, name)
            if len(set(items)) != len(items):
                raise _invalid(f"{name} must not contain duplicates")
            object.__setattr__(self, name, tuple(items))

        frequencies = {Frequency.parse(item).canonical for item in self.frequencies}
        if None in frequencies or len(frequencies) != len(self.frequencies):
            raise _invalid("frequencies must contain distinct supported frequencies")
        if set(self.session_views) - {"ETH", "RTH"}:
            raise _invalid("session_views must contain only ETH or RTH")
        _mapping(self.exchange_aliases, "exchange_aliases")
        for alias, exchange in self.exchange_aliases.items():
            if exchange not in self.exchanges or (alias in self.exchanges and alias != exchange):
                raise _invalid(f"Exchange alias {alias!r} must map unambiguously to a declared exchange")
        object.__setattr__(self, "exchange_aliases", MappingProxyType(dict(self.exchange_aliases)))
        object.__setattr__(self, "market_rules", _validated_market_rules(self.market_rules, frequencies))

        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        if self.profile_hash != "" and (
            not isinstance(self.profile_hash, str) or not _SHA256.fullmatch(self.profile_hash)
            or self.profile_hash != digest
        ):
            raise MarketDbProfileError(MARKET_DB_PROFILE_HASH_MISMATCH, "Profile content does not match profile_hash")
        object.__setattr__(self, "profile_hash", digest)

    @classmethod
    def from_mapping(cls, document: object, *, require_hash: bool = False) -> MarketDbProfile:
        _mapping(document, "profile")
        expected = {item.name for item in fields(cls)} - {"profile_hash"}
        actual = document.keys()
        missing = expected - actual
        extra = actual - expected - {"profile_hash"}
        if missing or extra:
            raise _invalid(f"Unexpected profile fields (missing={sorted(missing)}, unknown={sorted(extra)})")
        if (require_hash or "profile_hash" in actual) and document.get("profile_hash") in (None, ""):
            raise MarketDbProfileError(MARKET_DB_PROFILE_HASH_MISMATCH, "Stored profile_hash is missing")
        return cls(**document)

    def as_dict(self) -> dict[str, object]:
        document = {item.name: getattr(self, item.name) for item in fields(self)}
        for name in _LIST_FIELDS:
            document[name] = list(document[name])
        document["exchange_aliases"] = dict(self.exchange_aliases)
        document["market_rules"] = {
            name: dict(setting) if isinstance(setting, Mapping) else setting
            for name, setting in self.market_rules.items()
        }
        return document

    def canonical_json(self) -> str:
        """Encode identity fields. The declaration timestamp is audit metadata."""
        document = self.as_dict()
        del document["created_at_utc"], document["profile_hash"]
        return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        _text(key, "object key")
        if key in result:
            raise _invalid(f"Duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _invalid(f"Non-finite JSON value: {value}")


def parse_profile_json(content: str) -> object:
    try:
        return json.loads(content, object_pairs_hook=_unique_keys, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, MarketDbProfileError):
            raise
        raise _invalid(f"Invalid profile JSON: {exc}") from exc


def load_market_db_profile_yaml(path: str | Path) -> MarketDbProfile:
    source = Path(path).expanduser()
    try:
        content = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _invalid(f"Cannot read profile declaration {source}: {exc}") from exc

    if source.suffix.lower() == ".json":
        document = parse_profile_json(content)
    elif source.suffix.lower() in (".yaml", ".yml"):
        import yaml

        class ProfileLoader(yaml.SafeLoader):
            pass

        ProfileLoader.add_constructor("tag:yaml.org,2002:map",
                                      lambda loader, node: _unique_keys(loader.construct_pairs(node)))
        ProfileLoader.add_constructor("tag:yaml.org,2002:timestamp", ProfileLoader.construct_scalar)
        try:
            document = yaml.load(content, Loader=ProfileLoader)
        except (yaml.YAMLError, ValueError, OverflowError, RecursionError) as exc:
            if isinstance(exc, MarketDbProfileError):
                raise
            raise _invalid(f"Invalid profile YAML: {exc}") from exc
    else:
        raise _invalid("Profile declaration must be a .yaml, .yml or .json file")
    return MarketDbProfile.from_mapping(document)
