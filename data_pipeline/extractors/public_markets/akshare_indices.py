"""Build the mainland China price-index database from frozen AKShare responses."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .common import PRICES, ROOT, PublicMarket, Snapshot, arguments, canonical, daily_opens, daily_summary, instrument
from utils.slots.schema import load_slot_config

DB_ID = 'cn_index_akshare_v1'
START, END = '1990-12-19', '2026-10-31'
BROAD = {
    '000001.XSHG': ('sh000001', '1991-07-15'),
    '000016.XSHG': ('sh000016', '2004-01-02'),
    '000300.XSHG': ('sh000300', '2005-04-08'),
    '000905.XSHG': ('sh000905', '2007-01-15'),
    '000852.XSHG': ('sh000852', '2014-10-17'),
    '399006.XSHE': ('sz399006', '2010-06-01'),
}
SW_RELEASE = '2021-12-13'
SW_FIELDS = {'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}


def read_index(path: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'代码': str})
    code, exchange = symbol.split('.')
    if exchange == 'SW':
        if '代码' not in frame or not frame['代码'].eq(code).all():
            raise ValueError(f'Shenwan response code disagrees with {symbol}')
        frame = frame.rename(columns=SW_FIELDS)
    frame = frame.loc[:, ['date', *PRICES]].copy()
    frame['date'] = pd.to_datetime(frame.date, format='%Y-%m-%d')
    frame = frame.set_index('date').sort_index()
    if frame.empty or frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError(f'AKShare response has missing or duplicate index dates for {symbol}')
    return frame.apply(pd.to_numeric)


def run(workspace: Path, database: Path, *, prepare_only: bool = False) -> None:
    import akshare as ak

    snapshot = Snapshot(workspace)
    slots = ROOT / 'config/market_slots' / DB_ID
    slot = load_slot_config(slots / f'{DB_ID}_1d.yaml')
    shenwan = sorted(key.split('.')[0] for key in slot.universe.options['instruments'] if key.endswith('.SW'))
    # An existing snapshot keeps its recorded library version across offline rebuilds.
    ak_version = next(iter(snapshot.entries.values()))['source'].split(':', 1)[0] if snapshot.entries else f'akshare={ak.__version__}'

    def capture(name: str, endpoint: str, fetch: Callable[[], pd.DataFrame]) -> Path:
        def download(path: Path) -> None:
            if ak_version != f'akshare={ak.__version__}':
                raise ValueError(f'Use {ak_version} to resume fetching this snapshot')
            time.sleep(1)
            result = fetch()
            if result.empty:
                raise ValueError(f'Empty AKShare response for {name}')
            result.to_csv(path, index=False)

        return snapshot.capture(name, f'{ak_version}:{endpoint}', download)

    directory_path = capture('shenwan-level1.csv', 'index_realtime_sw(一级行业)',
                             lambda: ak.index_realtime_sw('一级行业'))
    discovered = pd.read_csv(directory_path, dtype={'指数代码': str})['指数代码']
    if sorted(discovered) != shenwan:
        raise ValueError('Shenwan level-1 directory differs from the frozen Slot universe')
    paths = {}
    for symbol, (code, _) in BROAD.items():
        paths[symbol] = capture(f'{code}.csv', f'stock_zh_index_daily({code})',
                                lambda code=code: ak.stock_zh_index_daily(code))
    for code in shenwan:
        paths[f'{code}.SW'] = capture(f'sw-{code}.csv', f'index_hist_sw({code},day)',
                                     lambda code=code: ak.index_hist_sw(code, period='day'))

    market = PublicMarket(DB_ID, workspace, database, snapshot, snapshot.version('akshare-price-index-v1'))
    market.report.update(price_index_kind='price_index', library_version=ak_version, snapshot=snapshot.entries)
    opens = daily_opens('XSHG', START, END)
    frames = []
    for symbol, path in paths.items():
        series = read_index(path, symbol)
        release = BROAD[symbol][1] if symbol in BROAD else SW_RELEASE
        scoped = series.loc[release:END].copy()
        if scoped.empty:
            raise ValueError(f'No eligible index observations for {symbol}')
        stats = daily_summary(scoped.index, opens)
        expected = opens.loc[release:END].index
        missing = expected.difference(scoped.index)
        stats.update(release_date=release, missing_bars=len(missing),
                     calendar_only_dates=missing.strftime('%Y-%m-%d').tolist(),
                     pre_release_rows=int((series.index < release).sum()),
                     full_snapshot_calendar_check=daily_summary(series.loc[START:END].index, opens))
        market.report['instruments'][symbol] = stats
        scoped['timestamp_utc'] = scoped.index.map(opens)
        frames.append(canonical(scoped, symbol, '1d', 'akshare_price_index', market.data_version))
    outside = {symbol: stats['source_only_dates'] for symbol, stats in market.report['instruments'].items()
               if stats['source_only_dates']}
    if outside:
        market.report.update(status='calendar_mismatch', rejected_bars=sum(map(len, outside.values())),
                             rejection_reason='AKShare source dates outside the PMC XSHG grid')
        market.write_report()
        raise ValueError(market.report['rejection_reason'])
    metadata = [instrument(symbol, 'index', 'Asia/Shanghai', 'XSHG') for symbol in paths]
    outputs = market.prepare('price_indices', frames, metadata)
    market.prepared()
    if not prepare_only:
        market.initialize()
        market.import_bars(outputs, list(paths), sum(map(len, frames)), '1d')
        market.finish()


if __name__ == '__main__':
    args = arguments(DB_ID)
    run(args.work_dir, args.db, prepare_only=args.prepare_only)
