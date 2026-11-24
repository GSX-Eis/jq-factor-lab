"""Freeze raw inputs and load isolated public market databases through the Harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

from utils.market_db import load_market_db_profile_yaml
from utils.market_db.research import open_research_slot
from utils.slots.schema import load_slot_config

ROOT = Path(__file__).resolve().parents[3]
PRICES = ['open', 'high', 'low', 'close', 'volume']
COLUMNS = ['instrument_id', 'timestamp_utc', 'frequency', *PRICES, 'source', 'data_version']


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path: Path, data: object) -> None:
    pending = path.with_suffix(path.suffix + '.part')
    pending.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        with destination.open('wb') as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                handle.write(chunk)


class Snapshot:
    """Cache completed source files with immutable checksums and acquisition metadata."""

    def __init__(self, root: Path) -> None:
        self.root = root / 'raw'
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / 'sha256.json'
        self.entries = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}

    def capture(self, name: str, source: str, fetch: Callable[[Path], None]) -> Path:
        path = self.root / name
        if name in self.entries:
            record = self.entries[name]
            if source != record['source'] or not path.is_file() or sha256(path) != record['sha256']:
                raise ValueError(f'Frozen snapshot changed or is missing: {name}')
            return path
        pending = path.with_suffix(path.suffix + '.part')
        # A completed atomic rename can survive an interruption before manifest publication.
        if not path.exists():
            fetch(pending)
            pending.replace(path)
        self.entries[name] = {
            'source': source, 'sha256': sha256(path),
            'downloaded_at_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        atomic_json(self.manifest, self.entries)
        return path

    def http(self, name: str, url: str) -> Path:
        return self.capture(name, url, lambda path: download(url, path))

    def version(self, label: str) -> str:
        # Acquisition times remain in the manifest. Copies of identical bytes have one identity.
        digests = {name: entry['sha256'] for name, entry in self.entries.items()}
        return f'{label}:sha256={hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()}'


def arguments(db_id: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f'Build the {db_id} public market snapshot')
    parser.add_argument('--work-dir', type=Path, required=True, help='Owned, durable directory outside tracked inputs')
    parser.add_argument('--db', type=Path, required=True, help='New DuckDB file inside the work directory')
    parser.add_argument('--prepare-only', action='store_true', help='Freeze and normalize source inputs without importing')
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.db = args.db.resolve()
    if not args.db.is_relative_to(args.work_dir) or args.db.suffix != '.duckdb':
        parser.error('--db must be a .duckdb file within --work-dir')
    return args


def instrument(instrument_id: str, asset_class: str, timezone_name: str, calendar_id: str,
               custom_sessions: dict | None = None) -> dict:
    metadata = {
        'instrument_id': instrument_id, 'asset_class': asset_class, 'exchange': instrument_id.rsplit('.', 1)[1],
        'timezone': timezone_name, 'calendar_id': calendar_id, 'rth_reference_calendar': calendar_id,
        'price_adjustment': 'none', 'point_value': 1.0,
    }
    if custom_sessions is not None:
        metadata['custom_sessions'] = json.dumps(custom_sessions, sort_keys=True, separators=(',', ':'))
    return metadata


def daily_opens(calendar_id: str, start: str, end: str) -> pd.Series:
    from utils.session_calendar import SessionCalendar

    calendar = SessionCalendar(calendar_id, start, end)
    dates = calendar.sessions()
    return pd.Series([calendar.session_on(day).segments[0][0] for day in dates], index=dates)


def daily_summary(dates: pd.DatetimeIndex, opens: pd.Series) -> dict:
    expected = opens.loc[dates.min():dates.max()].index
    missing = expected.difference(dates)
    off_grid = dates.difference(expected)
    return {
        'first_date': dates.min().date().isoformat(), 'last_date': dates.max().date().isoformat(),
        'rows': len(dates), 'missing_bars': len(missing),
        'calendar_only_dates': missing.strftime('%Y-%m-%d').tolist(),
        'source_only_dates': off_grid.strftime('%Y-%m-%d').tolist(),
    }


def canonical(frame: pd.DataFrame, instrument_id: str, frequency: str, source: str, version: str) -> pd.DataFrame:
    return frame.assign(instrument_id=instrument_id, frequency=frequency, source=source,
                        data_version=version).loc[:, COLUMNS]


@dataclass
class PublicMarket:
    db_id: str
    workspace: Path
    db: Path
    snapshot: Snapshot
    data_version: str
    report: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.profile_path = ROOT / 'config/market_databases' / f'{self.db_id}.yaml'
        self.profile = load_market_db_profile_yaml(self.profile_path)
        self.report.update(db_id=self.db_id, profile_hash=self.profile.profile_hash,
                           data_version=self.data_version, snapshot=self.snapshot.entries,
                           instruments={}, rejected_bars=0, status='preparing')
        self.write_report()

    def write_report(self) -> None:
        atomic_json(self.workspace / 'import_report.json', self.report)

    def prepared(self) -> None:
        stats = self.report['instruments']
        self.report.update(status='prepared', instrument_count=len(stats),
                           rows=sum(item['rows'] for item in stats.values()),
                           missing_bars=sum(item['missing_bars'] for item in stats.values()))
        self.write_report()

    def harness(self, command: str, *args: object) -> dict:
        if command in ('import-bars', 'build-bars') and '--dry-run' not in args and '--check-only' not in args:
            with duckdb.connect(str(self.db)) as connection:
                # Commit invalidation before the Harness can commit any physical Bar changes.
                connection.begin()
                for view in self._published_views(connection):
                    connection.execute(f'DROP VIEW {view}')
                connection.commit()
        result = subprocess.run(
            [sys.executable, '-m', 'tools.agent_harness.cli', command, *map(str, args),
             '--output-root', str(self.workspace / 'harness')],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        try:
            payload = json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError(result.stderr.strip() or 'Harness did not emit JSON') from exc
        if result.returncode or payload['exit_code']:
            message = payload.get('metrics', {}).get('message') or payload.get('reasons') or 'Harness failed'
            raise ValueError(str(message).replace(str(self.workspace), '<work-dir>'))
        return payload['metrics']

    def initialize(self) -> None:
        if not self.db.exists():
            args = ('market-db', 'init', '--profile', self.profile_path, '--db', self.db)
            self.harness(*args, '--dry-run')
            self.harness(*args)
        identity = self.harness('market-db', 'show', '--db', self.db)
        if (identity['db_id'], identity['profile_hash']) != (self.db_id, self.profile.profile_hash):
            raise ValueError(f'Unexpected database identity for {self.db_id}')

    def prepare(self, key: str, frames: Iterable[pd.DataFrame], metadata: list[dict]) -> tuple[Path, Path]:
        parts = self.workspace / 'normalized'
        parts.mkdir(exist_ok=True)
        bars = parts / f'{key}.csv'
        pending = bars.with_suffix('.csv.part')
        try:
            with pending.open('w', encoding='utf-8') as handle:
                for index, frame in enumerate(frames):
                    frame.to_csv(handle, index=False, header=index == 0, date_format='%Y-%m-%dT%H:%M:%SZ')
            pending.replace(bars)
        finally:
            pending.unlink(missing_ok=True)
        meta = parts / f'{key}.json'
        atomic_json(meta, metadata)
        return bars, meta

    def import_bars(self, paths: tuple[Path, Path], ids: list[str], rows: int, frequency: str) -> None:
        # An import is atomic. DuckDB itself is the resume checkpoint if the process dies after commit.
        table = f'market_bars_{frequency}'
        with duckdb.connect(str(self.db), read_only=True) as connection:
            exists = connection.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',
                                        [table]).fetchone()[0]
            if exists:
                stored, same_version = connection.execute(
                    f'SELECT count(*), count(*) FILTER (WHERE starts_with(data_version, ?)) '
                    f'FROM {table} WHERE instrument_id IN (SELECT unnest(?))',
                    [self.data_version, ids],
                ).fetchone()
                if stored == same_version == rows:
                    return
                if stored:
                    raise ValueError(f'Existing {frequency} rows conflict with frozen snapshot for {ids}')
        args = ('import-bars', '--db', self.db, '--bars', paths[0], '--metadata', paths[1])
        try:
            self.harness(*args, '--dry-run')
            result = self.harness(*args)
            if result['imported_bars'] != rows:
                raise RuntimeError('Harness import row count disagrees with normalized input')
        except ValueError as exc:
            self.report.update(status='import_rejected', rejected_bars=rows, rejection_reason=str(exc))
            self.write_report()
            raise

    def _database_snapshot_id(self, connection: duckdb.DuckDBPyConnection) -> str:
        digest = hashlib.sha256(json.dumps(self.data_version).encode())
        for frequency in sorted(self.profile.frequencies):
            digest.update(f'\n{frequency}\n'.encode())
            rows = connection.execute(
                f'SELECT instrument_id, epoch_us(timestamp_utc), {", ".join(PRICES)}, is_complete '
                f'FROM market_bars_{frequency} ORDER BY instrument_id, timestamp_utc'
            )
            while batch := rows.fetchmany(8192):
                digest.update(('\n'.join(json.dumps(row, separators=(',', ':')) for row in batch) + '\n').encode())
        return digest.hexdigest()

    @staticmethod
    def _published_views(connection: duckdb.DuckDBPyConnection) -> list[str]:
        rows = connection.execute(
            "SELECT table_name FROM information_schema.views WHERE table_catalog = current_database() "
            "AND table_schema = 'main' AND regexp_full_match(table_name, 'public_bars_(1d|1min)(_[a-f0-9]{64})?') "
            "ORDER BY length(table_name), table_name"
        ).fetchall()
        return [view for (view,) in rows]

    def finish(self, *, derived: dict | None = None) -> None:
        binding = json.loads(os.environ.get('QUANT_DATA_BINDINGS', '{}'))
        binding.setdefault('market_dbs', {})[self.db_id] = str(self.db)
        previous = os.environ.get('QUANT_DATA_BINDINGS')
        os.environ['QUANT_DATA_BINDINGS'] = json.dumps(binding)
        versioned_views = {}
        try:
            with duckdb.connect(str(self.db)) as connection:
                connection.begin()
                snapshot_id = self._database_snapshot_id(connection)
                previous_views = self._published_views(connection)
                for frequency in self.profile.frequencies:
                    view = f'public_bars_{frequency}'
                    frozen_view = f'{view}_{snapshot_id}'
                    versioned_views[view] = frozen_view
                    # Keep partial rollups in storage for auditing, without treating them as research bars.
                    connection.execute(f'CREATE OR REPLACE VIEW {frozen_view} AS '
                                       f'SELECT instrument_id, timestamp_utc, {", ".join(PRICES)} '
                                       f'FROM market_bars_{frequency} WHERE is_complete')
                    connection.execute(f'CREATE OR REPLACE VIEW {view} AS SELECT * FROM {frozen_view}')
                for view in previous_views:
                    if view not in versioned_views and view not in versioned_views.values():
                        connection.execute(f'DROP VIEW {view}')
                connection.commit()
            shown = self.harness('market-db', 'show', '--market-db', self.db_id)
            if (shown['db_id'], shown['profile_hash']) != (self.db_id, self.profile.profile_hash):
                raise ValueError('Bound database identity differs from repository profile')
            slots_dir = self.workspace / 'slots'
            slots_dir.mkdir(exist_ok=True)
            for path in sorted((ROOT / 'config/market_slots' / self.db_id).glob('*.yaml')):
                config = load_slot_config(path)
                with open_research_slot(self.db_id, config, path.parent):
                    pass
                # Declarative adapters attach the Slot version when reading rows from the source view.
                document = config.as_dict()
                options = document['bar_feed']['options']
                options['data_version'] = self.data_version
                options['table'] = versioned_views[options['table']]
                atomic_json(slots_dir / f'{path.stem}.json', document)
        finally:
            if previous is None:
                os.environ.pop('QUANT_DATA_BINDINGS')
            else:
                os.environ['QUANT_DATA_BINDINGS'] = previous
        stats = self.report['instruments']
        self.report.update(status='imported', database_snapshot_id=snapshot_id, instrument_count=len(stats),
                           rows=sum(item['rows'] for item in stats.values()),
                           missing_bars=sum(item['missing_bars'] for item in stats.values()),
                           snapshot=self.snapshot.entries, derived=derived or {})
        self.write_report()
