"""Local table and file drivers shared by declarative slot components."""
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from glob import glob
from pathlib import Path
import re

import pandas as pd

from utils.market_db import MarketDbProfile, open_market_db
from utils.market_db.storage import _require_same_hash
from utils.slots import ProviderConfig


def _path(base_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Data source paths must be nonempty strings')
    return base_dir / Path(value).expanduser()


@dataclass(frozen=True)
class FrameSource:
    provider: ProviderConfig
    base_dir: Path
    text_fields: Sequence[str]
    market_db_profile: MarketDbProfile | None = None

    def __post_init__(self) -> None:
        if self.provider.provider not in ('duckdb_table', 'parquet_glob', 'csv_files'):
            raise ValueError(f'Unknown declarative data driver: {self.provider.provider!r}')

    @property
    def database_path(self) -> Path:
        return _path(self.base_dir, self.provider.options.get('path')).resolve()

    def read(self, *, schema_only: bool = False) -> pd.DataFrame:
        """Read source data, or return an empty frame with the same columns."""
        options = self.provider.options
        driver = self.provider.provider
        mapping = options.get('column_mapping', {})
        if (not isinstance(mapping, Mapping)
                or any(not isinstance(value, str) or not value for value in mapping.values())):
            raise ValueError('column_mapping must map source names to column names')
        if driver == 'duckdb_table':
            table = options.get('table')
            if not isinstance(table, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*(\.[A-Za-z_][A-Za-z_0-9]*)?', table):
                raise ValueError('duckdb_table requires a table name or schema.table')
            table_sql = '.'.join(f'"{part}"' for part in table.split('.'))
            with open_market_db(self.database_path) as (connection, profile, _):
                if self.market_db_profile is not None:
                    _require_same_hash(profile, self.market_db_profile, 'Declarative database changed')
                connection.execute("SET TimeZone='UTC'")
                limit = ' LIMIT 0' if schema_only else ''
                frame = connection.execute(f'SELECT * FROM {table_sql}{limit}').fetchdf()
            timestamp = next((column for column, target in mapping.items() if target == 'timestamp'), None)
            source_timezone = options.get('source_timezone')
            if (source_timezone and timestamp in frame
                    and isinstance(frame[timestamp].dtype, pd.DatetimeTZDtype)):
                # DuckDB projects TIMESTAMPTZ in the connection timezone, regardless of the input offset.
                frame[timestamp] = frame[timestamp].dt.tz_convert(source_timezone)
        elif driver == 'parquet_glob':
            paths = sorted(glob(str(_path(self.base_dir, options.get('path'))), recursive=True))
            if not paths:
                raise ValueError('parquet_glob did not match any files')
            if schema_only:
                from pyarrow.parquet import ParquetDataset

                frame = self._combine([ParquetDataset(path).schema.empty_table().to_pandas() for path in paths])
            else:
                frame = self._combine([pd.read_parquet(path) for path in paths])
        else:
            paths = options.get('paths')
            if not isinstance(paths, tuple) or not paths:
                raise ValueError('csv_files requires a nonempty paths list')
            reverse = {target: source for source, target in mapping.items()}
            text_columns = {reverse.get(field, field): 'string' for field in self.text_fields}
            frames = []
            for value in paths:
                path = _path(self.base_dir, value)
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    header = next(csv.reader(handle), [])
                    if len(set(header)) != len(header):
                        raise ValueError(f'Duplicate CSV columns in {path}')
                    handle.seek(0)
                    frames.append(pd.read_csv(handle, dtype=text_columns, keep_default_na=False, na_values=[''],
                                              nrows=0 if schema_only else None))
            frame = self._combine(frames)
        if not frame.columns.is_unique or set(mapping) - set(frame):
            raise ValueError('column_mapping references missing or duplicate source columns')
        return frame

    @staticmethod
    def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if any(set(frame.columns) != set(frames[0].columns) for frame in frames):
            raise ValueError('All data source files must have the same columns')
        return pd.concat(frames, ignore_index=True)

    def read_mapped(self, *, schema_only: bool = False) -> pd.DataFrame:
        frame = self.read(schema_only=schema_only).rename(columns=dict(self.provider.options.get('column_mapping', {})))
        if not frame.columns.is_unique:
            raise ValueError('column_mapping produces duplicate names')
        return frame
