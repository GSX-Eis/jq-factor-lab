"""Versioned YAML/JSON market declarations and a canonical config digest."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from .contracts import ExecutionProfile, _name

JsonValue: TypeAlias = (
    str | int | float | bool | None | tuple['JsonValue', ...] | Mapping[str, 'JsonValue']
)
_COMPONENTS = ('calendar', 'bar_feed', 'universe', 'aux_features', 'microstructure')


def _config_value(value: object, *, freeze: bool, ancestors: frozenset[int] = frozenset()) -> JsonValue:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and isfinite(value):
        return value
    if id(value) in ancestors:
        raise ValueError('Slot config options cannot contain cycles')
    ancestors = ancestors | {id(value)}
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        mapping = {key: _config_value(item, freeze=freeze, ancestors=ancestors) for key, item in value.items()}
        return MappingProxyType(mapping) if freeze else mapping
    if isinstance(value, (tuple, list)):
        return tuple(_config_value(item, freeze=freeze, ancestors=ancestors) for item in value)
    raise ValueError('Slot config options must contain finite JSON values and string keys')


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.provider, 'provider')
        if not isinstance(self.options, Mapping):
            raise ValueError('Provider options must be a mapping')
        object.__setattr__(self, 'options', _config_value(self.options, freeze=True))


@dataclass(frozen=True)
class SlotConfig:
    schema_version: int
    slot_id: str
    calendar: ProviderConfig
    bar_feed: ProviderConfig
    universe: ProviderConfig
    aux_features: ProviderConfig
    microstructure: ProviderConfig
    execution_profile: ExecutionProfile
    market_domain: str | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('Slot config requires schema_version 1')
        _name(self.slot_id, 'slot_id')
        if self.market_domain is not None:
            _name(self.market_domain, 'market_domain')
        for field_name in _COMPONENTS:
            if not isinstance(getattr(self, field_name), ProviderConfig):
                raise ValueError(f'{field_name} must be a ProviderConfig')
        if not isinstance(self.execution_profile, ExecutionProfile):
            raise ValueError('execution_profile must be an ExecutionProfile')

    @classmethod
    def from_mapping(cls, document: Mapping[str, object]) -> SlotConfig:
        expected = {'schema_version', 'slot_id', 'execution_profile', *_COMPONENTS}
        if not isinstance(document, Mapping) or document.keys() - {'market_domain'} != expected:
            raise ValueError(f'Slot config requires exactly these fields: {", ".join(sorted(expected))}')
        components = {}
        for name in _COMPONENTS:
            component = document[name]
            if (not isinstance(component, Mapping) or 'provider' not in component
                    or component.keys() - {'provider', 'options'}):
                raise ValueError(f'{name} requires a provider and optional options')
            try:
                components[name] = ProviderConfig(**component)
            except ValueError as exc:
                raise ValueError(f'{name}: {exc}') from exc
        profile = document['execution_profile']
        if not isinstance(profile, Mapping):
            raise ValueError('execution_profile must be a mapping')
        try:
            execution = ExecutionProfile(**profile)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'execution_profile: {exc}') from exc
        return cls(document['schema_version'], document['slot_id'], **components, execution_profile=execution,
                   market_domain=document.get('market_domain'))

    def as_dict(self) -> dict[str, object]:
        document = {
            'schema_version': self.schema_version, 'slot_id': self.slot_id,
            'execution_profile': asdict(self.execution_profile),
        }
        if self.market_domain is not None:
            document['market_domain'] = self.market_domain
        for name in _COMPONENTS:
            component = getattr(self, name)
            document[name] = {'provider': component.provider,
                              'options': _config_value(component.options, freeze=False)}
        return document

    @property
    def slot_config_hash(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def slot_config_hash(config: SlotConfig | Mapping[str, object]) -> str:
    parsed = config if isinstance(config, SlotConfig) else SlotConfig.from_mapping(config)
    return parsed.slot_config_hash


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError('Slot config keys must be strings')
        if key in result:
            raise ValueError(f'Duplicate slot config key: {key!r}')
        result[key] = value
    return result


def _load_yaml(content: str) -> object:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        if exc.name != 'yaml':
            raise
        raise ValueError('YAML syntax requires installed PyYAML. Use JSON syntax or a .json file.') from exc

    class SlotLoader(yaml.SafeLoader):
        pass

    SlotLoader.add_constructor('tag:yaml.org,2002:map',
                               lambda loader, node: _unique_keys(loader.construct_pairs(node)))
    SlotLoader.add_constructor('tag:yaml.org,2002:timestamp', SlotLoader.construct_scalar)
    try:
        return yaml.load(content, Loader=SlotLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid slot YAML: {exc}') from exc


def load_slot_config(path: str | Path) -> SlotConfig:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ('.json', '.yaml', '.yml'):
        raise ValueError('Slot config must be a .json, .yaml or .yml file')
    content = path.read_text(encoding='utf-8')
    try:
        document = json.loads(content, object_pairs_hook=_unique_keys)
    except json.JSONDecodeError:
        if suffix == '.json':
            raise
        document = _load_yaml(content)
    return SlotConfig.from_mapping(document)
