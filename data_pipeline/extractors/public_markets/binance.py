"""Build Binance Vision spot minutes and derive daily bars from the same observations."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd

from .common import PRICES, PublicMarket, Snapshot, arguments, canonical, download, instrument, sha256

DB_ID = 'crypto_binance_spot_v1'
BASE = 'https://data.binance.vision/data/spot/monthly/klines'
CALENDAR_ID = 'BINANCE_UTC'
SESSIONS = {'timezone': 'UTC', 'weekdays': list(range(7)), 'segments': [
    {'open': '00:00', 'close': '00:00', 'open_day_offset': 0, 'close_day_offset': 1},
]}
PAIRS = ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT',
         'LINKUSDT', 'LTCUSDT', 'AVAXUSDT')
START, LAST_MONTH = '2024-11', '2026-10'
BUDGET = 5_000_000_000


def verify_checksum(path: Path, checksum: Path, name: str) -> None:
    record = checksum.read_text(encoding='ascii').strip().split()
    if (len(record) != 2 or not re.fullmatch(r'[0-9a-fA-F]{64}', record[0])
            or record[1].lstrip('*') != name or sha256(path) != record[0].lower()):
        raise ValueError(f'Binance checksum failed for {name}')


def read_klines(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1 or not members[0].endswith('.csv'):
            raise ValueError(f'Expected one Binance K-line CSV in {path.name}')
        with archive.open(members[0]) as source:
            rows = pd.read_csv(source, header=None, dtype={0: str})
    if rows.shape[1] != 12:
        raise ValueError(f'Expected 12 Binance K-line columns in {path.name}')
    if rows.iloc[0, 0] == 'open_time':
        rows = rows.iloc[1:]
    if rows.empty or not rows[0].str.fullmatch(r'\d{13}|\d{16}').fillna(False).all():
        raise ValueError(f'Invalid Binance open timestamps in {path.name}')
    epochs = rows[0].astype('int64')
    micros = epochs.where(epochs >= 10**14, epochs * 1000)
    fields = rows.iloc[:, 1:6].apply(pd.to_numeric)
    fields.columns = PRICES
    fields.insert(0, 'timestamp_utc', pd.to_datetime(micros, unit='us', utc=True))
    if fields.timestamp_utc.duplicated().any() or not np.isfinite(fields[PRICES]).all().all():
        raise ValueError(f'Duplicate or non-finite Binance data in {path.name}')
    return fields


def capture_archive(snapshot: Snapshot, pair: str, interval: str, month: pd.Period) -> Path:
    name = f'{pair}-{interval}-{month}.zip'
    url = f'{BASE}/{pair}/{interval}/{name}'
    checksum = snapshot.http(name + '.CHECKSUM', url + '.CHECKSUM')

    def fetch(path: Path) -> None:
        download(url, path)
        verify_checksum(path, checksum, name)

    path = snapshot.capture(name, url, fetch)
    verify_checksum(path, checksum, name)
    return path


def read_month(path: Path, month: pd.Period, interval: str) -> pd.DataFrame:
    frame = read_klines(path)
    start = month.start_time.tz_localize('UTC')
    end = (month + 1).start_time.tz_localize('UTC')
    ts = frame.timestamp_utc
    if not (ts.ge(start) & ts.lt(end) & ts.eq(ts.dt.floor(interval))).all():
        raise ValueError(f'Binance timestamps are outside the declared month or grid in {path.name}')
    return frame


def reconcile_daily(db: Path, symbol: str, official: pd.DataFrame) -> dict:
    expected = official.set_index('timestamp_utc').sort_index()
    with duckdb.connect(str(db), read_only=True) as connection:
        connection.execute("SET TimeZone='UTC'")
        actual = connection.execute('''
            SELECT timestamp_utc, open, high, low, close, volume, is_complete
            FROM market_bars_1d WHERE instrument_id=? AND timestamp_utc BETWEEN ? AND ?
        ''', [symbol, expected.index.min(), expected.index.max()]).fetchdf().set_index('timestamp_utc')
    missing = expected.index.difference(actual.index)
    partial = actual.index[~actual.is_complete]
    common = expected.index.intersection(actual.index[actual.is_complete])
    matches = np.isclose(actual.loc[common, PRICES], expected.loc[common, PRICES], rtol=1e-10, atol=1e-8).all(axis=1)
    return {
        'official_rows': len(expected), 'complete_days_compared': len(common), 'matched_days': int(matches.sum()),
        'different_days': common[~matches].strftime('%Y-%m-%d').tolist(),
        'missing_derived_days': missing.strftime('%Y-%m-%d').tolist(),
        'incomplete_derived_days': partial.strftime('%Y-%m-%d').tolist(),
    }


def run(workspace: Path, database: Path, *, prepare_only: bool = False) -> None:
    month_ranges = {pair: pd.period_range(START if pair in PAIRS[:2] else LAST_MONTH, LAST_MONTH, freq='M')
                    for pair in PAIRS}
    estimated_rows = sum(month.days_in_month * 1440 for months in month_ranges.values() for month in months)
    estimate_bytes = estimated_rows * 1200
    if estimate_bytes > BUDGET:
        raise ValueError('Binance coverage estimate exceeds the 5 GB disk limit')
    snapshot = Snapshot(workspace)
    if shutil.disk_usage(workspace).free < estimate_bytes + 15 * 1024**3:
        raise ValueError('Insufficient free disk for Binance estimate and 15 GiB reserve')
    archives = {pair: {month: capture_archive(snapshot, pair, '1m', month) for month in months}
                for pair, months in month_ranges.items()}
    last_month = pd.Period(LAST_MONTH, freq='M')
    daily = {pair: capture_archive(snapshot, pair, '1d', last_month) for pair in PAIRS}
    market = PublicMarket(DB_ID, workspace, database, snapshot, snapshot.version('binance-spot-v1'))
    market.report.update(estimated_disk_bytes=estimate_bytes, disk_budget_bytes=BUDGET,
                         official_daily_sample_month=LAST_MONTH, snapshot=snapshot.entries)
    imports = []
    for pair, files in archives.items():
        symbol = f'{pair}.BINANCE'
        months = month_ranges[pair]
        fields = pd.concat([read_month(path, month, 'min') for month, path in files.items()], ignore_index=True)
        start = months[0].start_time.date().isoformat()
        end = months[-1].end_time.date().isoformat()
        rows = len(fields)
        market.report['instruments'][symbol] = {
            'requested_first_date': start, 'requested_last_date': end,
            'first_date': fields.timestamp_utc.min().date().isoformat(),
            'last_date': fields.timestamp_utc.max().date().isoformat(), 'rows': rows,
            'missing_bars': sum(month.days_in_month * 1440 for month in months) - rows,
        }
        metadata = [instrument(symbol, 'crypto', 'UTC', CALENDAR_ID, SESSIONS)]
        paths = market.prepare(pair, [canonical(fields, symbol, '1min', 'binance_vision', market.data_version)], metadata)
        imports.append((paths, symbol, rows))
    market.prepared()
    if prepare_only:
        return
    market.initialize()
    for paths, symbol, rows in imports:
        market.import_bars(paths, [symbol], rows, '1min')

    reconciliation = {}
    for pair in PAIRS:
        symbol = f'{pair}.BINANCE'
        stats = market.report['instruments'][symbol]
        args = ('build-bars', '--db', database, '--source', '1min', '--targets', '1d', '--instruments', symbol,
                '--start', stats['requested_first_date'], '--end', stats['requested_last_date'])
        market.harness(*args, '--dry-run')
        market.harness(*args)
        reconciliation[symbol] = reconcile_daily(database, symbol, read_month(daily[pair], last_month, 'D'))
    mismatches = sum(len(item['different_days']) for item in reconciliation.values())
    if mismatches:
        market.report.update(status='daily_reconciliation_failed', derived=reconciliation)
        market.write_report()
        raise ValueError(f'Binance daily reconciliation found {mismatches} complete days with mismatches')
    with duckdb.connect(str(database), read_only=True) as connection:
        daily_stats = connection.execute('''
            SELECT instrument_id, count(*), count(*) FILTER (WHERE NOT is_complete),
                   min(timezone('UTC', timestamp_utc))::DATE, max(timezone('UTC', timestamp_utc))::DATE
            FROM market_bars_1d GROUP BY instrument_id
        ''').fetchall()
    for symbol, rows, incomplete, first, last in daily_stats:
        reconciliation[symbol].update(rows=rows, incomplete_rows=incomplete,
                                      first_date=first.isoformat(), last_date=last.isoformat())
    market.finish(derived=reconciliation)


if __name__ == '__main__':
    args = arguments(DB_ID)
    run(args.work_dir, args.db, prepare_only=args.prepare_only)
