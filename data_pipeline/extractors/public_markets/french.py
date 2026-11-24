"""Freeze French daily return files and encode returns as close-only indices."""
from __future__ import annotations

import csv
from itertools import chain
import re
from collections.abc import Iterator
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .common import PublicMarket, Snapshot, arguments, canonical, daily_opens, daily_summary, instrument

DB_ID = 'kf_us_portfolios_v1'
START, END = '1963-07-01', '2026-10-31'
BASE = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
INDUSTRY = '49_Industry_Portfolios_daily_CSV.zip'
FACTORS = 'F-F_Research_Data_Factors_daily_CSV.zip'


def read_daily_returns(path: Path) -> list[pd.DataFrame]:
    """Find daily sections by their empty date-column heading and 8-digit date rows."""
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected one French CSV in the archive')
        rows = csv.reader(archive.read(members[0]).decode('latin1').splitlines())
        header, records, sections = [], [], []
        for cells in chain(rows, [[]]):
            cells = [cell.strip() for cell in cells]
            if cells and re.fullmatch(r'\d{8}', cells[0]):
                if not header or len(cells) != len(header) + 1:
                    raise ValueError('French daily row does not match its header')
                records.append(cells)
                continue
            if records:
                dates, *columns = zip(*records)
                daily = pd.DataFrame(dict(zip(header, columns)), index=pd.to_datetime(list(dates), format='%Y%m%d'))
                if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
                    raise ValueError('French dates must be distinct and increasing')
                sections.append(daily.astype(float).replace([-99.99, -999], float('nan')) / 100)
                records = []
            header = cells[1:] if len(cells) > 1 and not cells[0] else []
            if len(set(header)) != len(header):
                raise ValueError('French daily header contains duplicate series')
    return sections


def close_index(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod().dropna()


def run(workspace: Path, database: Path, *, prepare_only: bool = False) -> None:
    snapshot = Snapshot(workspace)
    archives = {name: snapshot.http(name, BASE + name) for name in (INDUSTRY, FACTORS)}
    market = PublicMarket(DB_ID, workspace, database, snapshot, snapshot.version('french-daily-v1'))
    market.report['missing_policy'] = 'Omit sentinel rows; resume the index from the last observed level, applying only the next reported return.'
    opens = daily_opens('XNYS', START, END)
    groups = []
    for name, suffixes, asset in ((INDUSTRY, ('_VW', '_EW'), 'portfolio'), (FACTORS, ('',), 'factor')):
        sections = read_daily_returns(archives[name])
        if (len(sections) != len(suffixes)
                or any(section.shape[1] != (49 if asset == 'portfolio' else 4) for section in sections)):
            raise ValueError(f'Unexpected French daily sections in {name}')
        release = snapshot.entries[name]
        version = (f'{market.data_version};downloaded={release["downloaded_at_utc"][:10]};'
                   f'archive_sha256={release["sha256"]}')
        for section, suffix in zip(sections, suffixes, strict=True):
            series = {f'{column.upper().replace("-", "_")}{suffix}.KFDL': section[column].loc[START:END]
                      for column in section}
            metadata = [instrument(symbol, asset, 'America/New_York', 'XNYS') for symbol in series]
            groups.append((suffix.removeprefix('_').lower() or 'ff3', series, metadata, version))

    def frames(series: dict[str, pd.Series], version: str) -> Iterator[pd.DataFrame]:
        for symbol, returns in series.items():
            levels = close_index(returns)
            stats = daily_summary(levels.index, opens)
            stats['missing_bars'] = len(opens.loc[START:returns.index.max()]) - len(levels)
            stats['calendar_only_dates'] = opens.loc[START:returns.index.max()].index.difference(
                levels.index).strftime('%Y-%m-%d').tolist()
            stats['sentinel_values'] = int(returns.isna().sum())
            market.report['instruments'][symbol] = stats
            if stats['source_only_dates']:
                market.report.update(status='calendar_mismatch', rejected_bars=len(stats['source_only_dates']),
                                     rejection_reason=f'{symbol}: source dates outside XNYS')
                market.write_report()
                raise ValueError(market.report['rejection_reason'])
            values = levels.to_numpy()
            fields = pd.DataFrame(dict(timestamp_utc=opens.loc[levels.index].to_numpy(), open=values,
                                       high=values, low=values, close=values, volume=0.0))
            yield canonical(fields, symbol, '1d', 'kenneth_french', version)

    imports = []
    for key, series, metadata, version in groups:
        paths = market.prepare(key, frames(series, version), metadata)
        count = sum(market.report['instruments'][symbol]['rows'] for symbol in series)
        imports.append((paths, list(series), count))
    market.prepared()
    if prepare_only:
        return
    market.initialize()
    for paths, symbols, count in imports:
        market.import_bars(paths, symbols, count, '1d')
    market.finish()


if __name__ == '__main__':
    args = arguments(DB_ID)
    run(args.work_dir, args.db, prepare_only=args.prepare_only)
