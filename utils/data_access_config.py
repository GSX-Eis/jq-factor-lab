"""Immutable dataset definitions shared by local panel readers."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'data_access.json'


def _name(value, label):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError(f'{label} must be a nonempty name')
    return value


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{label} must contain exactly {sorted(expected)}')


@dataclass(frozen=True)
class Dataset:
    table: str
    date_column: str = 'date'
    stock_column: str = 'stock_code'
    parquet_subdir: str = ''
    default_fields: Optional[Tuple[str, ...]] = None

    def fields(self, requested):
        values = self.default_fields if requested is None else requested
        if values is None:
            raise ValueError(f'{self.table}: supply fields or configure default_fields')
        if not isinstance(values, (list, tuple)):
            raise ValueError('fields must be a list or tuple of column names')
        result = []
        for field in values:
            _name(field, 'field')
            if field in (self.date_column, self.stock_column):
                continue
            if field in ('date', 'code'):
                raise ValueError(f'field {field!r} conflicts with normalized date/code keys')
            if field not in result:
                result.append(field)
        return tuple(result)


@dataclass(frozen=True)
class IndexLookup:
    table: str
    index_column: str


@dataclass(frozen=True)
class DataAccessConfig:
    price_tables: Mapping[str, str]
    factor_table: str
    tables: Mapping[str, Dataset]
    index_constituents: Optional[IndexLookup] = None
    price_aliases: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def as_dict(self):
        """Effective config for reproducibility and persistent factor cache identity."""
        result = {
            'schema_version': 1, 'price_tables': dict(self.price_tables),
            'factor_table': self.factor_table,
            'tables': {name: {
                'date_column': spec.date_column, 'stock_column': spec.stock_column,
                'parquet_subdir': spec.parquet_subdir,
                'default_fields': None if spec.default_fields is None else list(spec.default_fields),
            } for name, spec in self.tables.items()},
        }
        if self.index_constituents is not None:
            result['index_constituents'] = {
                'table': self.index_constituents.table,
                'index_column': self.index_constituents.index_column,
            }
        result['price_aliases'] = dict(self.price_aliases)
        return result

    def index_dataset(self, table=None):
        if self.index_constituents is None:
            raise ValueError('Configure index_constituents before querying index membership')
        dataset = self.dataset(self.index_constituents.table if table is None else table)
        column = self.index_constituents.index_column
        if column in (dataset.date_column, dataset.stock_column):
            raise ValueError('index_column must differ from date_column and stock_column')
        return dataset, column

    def dataset(self, table):
        _name(table, 'table')
        return self.tables.get(table) or Dataset(table, parquet_subdir=_subdir(table))

    def price(self, frequency):
        if not isinstance(frequency, str) or frequency not in self.price_tables:
            raise ValueError(f'Unsupported local price frequency {frequency!r}; configured: {list(self.price_tables)}')
        return self.dataset(self.price_tables[frequency])

    def resolve_price(self, table):
        """Resolve configured aliases first, then literal tables and their frequency."""
        _name(table, 'price table')
        if table in self.price_aliases:
            frequency = self.price_aliases[table]
            return self.price(frequency), frequency
        frequency = next((freq for freq, name in self.price_tables.items() if name == table), None)
        return self.dataset(table), frequency


def _subdir(value):
    _name(value, 'parquet_subdir')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or any(c in value for c in '*?[]') or value == '.':
        raise ValueError('parquet_subdir must be a relative dataset directory without traversal or glob characters')
    return value


def load_data_access_config(path=None):
    """Read once; reject malformed configs instead of silently using defaults."""
    with Path(path if path is not None else DEFAULT_CONFIG_PATH).open(encoding='utf-8') as handle:
        value = json.load(handle)
    expected = {'schema_version', 'price_tables', 'price_aliases', 'factor_table', 'tables'}
    if isinstance(value, dict) and 'index_constituents' in value:
        expected.add('index_constituents')
    _keys(value, expected, 'data config')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Unsupported data config schema_version')
    if not isinstance(value['tables'], dict) or not value['tables']:
        raise ValueError('tables must be a nonempty mapping')
    tables = {}
    for table, spec in value['tables'].items():
        _name(table, 'table')
        _keys(spec, {'date_column', 'stock_column', 'parquet_subdir', 'default_fields'}, table)
        date = _name(spec['date_column'], 'date_column')
        stock = _name(spec['stock_column'], 'stock_column')
        if date == stock:
            raise ValueError(f'{table}: date_column and stock_column must differ')
        dataset = Dataset(table, date, stock, _subdir(spec['parquet_subdir']))
        defaults = None if spec['default_fields'] is None else dataset.fields(spec['default_fields'])
        tables[table] = Dataset(table, date, stock, dataset.parquet_subdir, defaults)
    prices = value['price_tables']
    if not isinstance(prices, dict) or not prices:
        raise ValueError('price_tables must be a nonempty mapping')
    for frequency, table in prices.items():
        _name(frequency, 'frequency')
        if _name(table, 'price table') not in tables:
            raise ValueError(f'price_tables references unregistered table {table!r}')
    if len(set(prices.values())) != len(prices):
        raise ValueError('price_tables cannot assign one table to multiple frequencies')
    aliases = value['price_aliases']
    if not isinstance(aliases, dict):
        raise ValueError('price_aliases must be a mapping')
    for alias, frequency in aliases.items():
        _name(alias, 'price alias')
        if _name(frequency, 'price alias frequency') not in prices:
            raise ValueError(f'price alias {alias!r} references unconfigured frequency {frequency!r}')
        if alias in prices.values() and alias != prices[frequency]:
            raise ValueError(f'price alias {alias!r} conflicts with a configured price table frequency')
    factor_table = _name(value['factor_table'], 'factor_table')
    if factor_table not in tables:
        raise ValueError(f'factor_table references unregistered table {factor_table!r}')
    index_lookup = None
    if 'index_constituents' in value:
        spec = value['index_constituents']
        _keys(spec, {'table', 'index_column'}, 'index_constituents')
        table = _name(spec['table'], 'index_constituents table')
        if table not in tables:
            raise ValueError(f'index_constituents references unregistered table {table!r}')
        index_lookup = IndexLookup(table, _name(spec['index_column'], 'index_column'))
    config = DataAccessConfig(MappingProxyType(dict(prices)), factor_table, MappingProxyType(tables),
                              index_lookup, MappingProxyType(dict(aliases)))
    if index_lookup is not None:
        config.index_dataset()
    return config
