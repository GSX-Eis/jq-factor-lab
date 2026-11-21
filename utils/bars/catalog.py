"""Read-only catalog resolution; writers explicitly register physical Bar tables."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import duckdb

from utils.time_axis import Frequency


def identifier(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', name):
        raise ValueError(f'Invalid SQL table name: {name!r}')
    return '"' + name + '"'


def table_exists(connection: duckdb.DuckDBPyConnection, name: str) -> bool:
    return connection.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',
                              [name]).fetchone()[0] > 0


@dataclass(frozen=True)
class CatalogEntry:
    frequency: str
    table_name: str
    kind: str = 'native'
    dataset: str = 'market'
    instrument_id: str = '*'
    source_frequency: str | None = None
    source_version: str | None = None
    bucket_strategy: str | None = None
    calendar_hash: str | None = None
    build_id: str | None = None
    code_version: str | None = None
    coverage_start: str | None = None
    coverage_end: str | None = None
    timestamp_semantics: str = 'UTC_bar_start'
    quality_status: str = 'unscanned'

    def __post_init__(self) -> None:
        parsed = Frequency.parse(self.frequency)
        if not parsed.is_valid:
            raise ValueError(f'Invalid frequency: {self.frequency!r}')
        object.__setattr__(self, 'frequency', parsed.canonical)
        identifier(self.table_name)
        if self.kind not in ('native', 'derived'):
            raise ValueError('Catalog kind must be native or derived')


class BarCatalog:
    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None, *,
                 config: Mapping | str | Path | None = None,
                 entries: Sequence[CatalogEntry] = ()) -> None:
        self.connection = connection
        if config is None:
            config = Path(__file__).resolve().parents[2] / 'config/data_access.json'
        if not isinstance(config, Mapping):
            config = json.loads(Path(config).read_text(encoding='utf-8'))
        self.entries = list(entries)
        for frequency, name in config.get('price_tables', {}).items():
            parsed = Frequency.parse(frequency)
            if not parsed.is_valid:
                raise ValueError(f'Invalid configured frequency: {frequency!r}')
            semantics = ('local_midnight_day_identifier_pit' if parsed.kind == 'session'
                         else 'Asia/Shanghai_bar_end')
            self.entries.append(CatalogEntry(parsed.canonical, name, dataset='equity',
                                             timestamp_semantics=semantics))
        if connection is not None and table_exists(connection, 'bar_catalog'):
            self.entries.extend(CatalogEntry(**row) for row in
                                connection.execute('SELECT * FROM bar_catalog').fetchdf().where(
                                    lambda x: x.notna(), None).to_dict('records'))

    def available(self, instrument: str, *, asset_class: str = 'equity',
                  dataset: str | None = None) -> list[CatalogEntry]:
        return [entry for entry in self.entries
                if entry.instrument_id in ('*', instrument)
                and (entry.dataset != 'equity' or asset_class == 'equity')
                and (dataset is None or entry.dataset == dataset)
                and (self.connection is None or table_exists(self.connection, entry.table_name))]

    def resolve(self, frequency: str, instrument: str, *, asset_class: str = 'equity',
                dataset: str | None = None) -> CatalogEntry:
        parsed = Frequency.parse(frequency)
        if not parsed.is_valid:
            raise ValueError(f'Invalid frequency: {frequency!r}')
        matches = [entry for entry in self.available(instrument, asset_class=asset_class, dataset=dataset)
                   if entry.frequency == parsed.canonical]
        natives = [entry for entry in matches if entry.kind == 'native']
        matches = natives or matches
        names = {entry.table_name for entry in matches}
        if len(names) > 1:
            raise ValueError(f'Ambiguous frequency {parsed.canonical}: {sorted(names)}')
        if not matches:
            raise LookupError(f'Missing frequency {parsed.canonical} for {instrument}')
        # A persisted audit overrides the config's unscanned declaration.
        return next((entry for entry in matches if entry.instrument_id == instrument), matches[0])


def register_entry(connection: duckdb.DuckDBPyConnection, entry: CatalogEntry) -> None:
    """Called only inside a writer transaction; registration never touches prices."""
    connection.execute('''CREATE TABLE IF NOT EXISTS bar_catalog (
        frequency VARCHAR, table_name VARCHAR, kind VARCHAR, dataset VARCHAR,
        instrument_id VARCHAR, source_frequency VARCHAR, source_version VARCHAR,
        bucket_strategy VARCHAR, calendar_hash VARCHAR, build_id VARCHAR, code_version VARCHAR,
        coverage_start VARCHAR, coverage_end VARCHAR, timestamp_semantics VARCHAR,
        quality_status VARCHAR,
        PRIMARY KEY (dataset, frequency, table_name, instrument_id))''')
    old = connection.execute('SELECT coverage_start, coverage_end FROM bar_catalog '
        'WHERE dataset=? AND frequency=? AND table_name=? AND instrument_id=?',
        [entry.dataset, entry.frequency, entry.table_name, entry.instrument_id]).fetchone()
    if old:
        starts = [value for value in (old[0], entry.coverage_start) if value]
        ends = [value for value in (old[1], entry.coverage_end) if value]
        entry = replace(entry, coverage_start=min(starts) if starts else None,
                        coverage_end=max(ends) if ends else None)
    values = asdict(entry)
    connection.execute('INSERT OR REPLACE INTO bar_catalog VALUES (' +
                       ','.join('?' for _ in values) + ')', list(values.values()))


def create_bar_table(connection: duckdb.DuckDBPyConnection, frequency: str) -> str:
    name = Frequency.parse(frequency).table_name
    connection.execute(f'''CREATE TABLE IF NOT EXISTS {identifier(name)} (
        instrument_id VARCHAR NOT NULL, timestamp_utc TIMESTAMPTZ NOT NULL,
        frequency VARCHAR NOT NULL, open DOUBLE NOT NULL, high DOUBLE NOT NULL,
        low DOUBLE NOT NULL, close DOUBLE NOT NULL, volume DOUBLE NOT NULL,
        source VARCHAR NOT NULL, data_version VARCHAR NOT NULL, money DOUBLE, factor DOUBLE,
        n_source_bars BIGINT, n_expected_bars BIGINT, is_complete BOOLEAN,
        PRIMARY KEY (instrument_id, timestamp_utc, frequency))''')
    connection.execute(f'ALTER TABLE {identifier(name)} ADD COLUMN IF NOT EXISTS open_interest DOUBLE')
    return name
