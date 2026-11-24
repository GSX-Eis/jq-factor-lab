"""Replay a bounded real-market snapshot under the archived master and candidate."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from data_pipeline.loaders.market_bar_importer import import_market_bars
from utils.bars.continuous import ContinuousBarSource
from utils.bars.futures_roll import build_roll_schedule
from utils.duckdb_adapter import DuckDBAdapter
from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter
from utils.pit import apply_pit_adjustment, normalize_price_money
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule, Frequency
from utils.time_axis.futures import futures_calendar


A_SHARE_DATES = ('2022-06-01', '2022-07-08')
FUTURES_DATES = ('2022-07-18', '2022-07-19', '2022-09-02')
SYMBOLS = ['000001.XSHE', '000002.XSHE', '600000.XSHG', '600519.XSHG', '601318.XSHG']
BAR_KEYS = ['instrument_id', 'timestamp_utc', 'session_id']
PRICES = ['open', 'high', 'low', 'close']
COMMON = [*PRICES, 'volume', 'money', 'factor']
FINANCIAL = 'raw_income_statement_events.operating_revenue'
ROLL_COLUMNS = ['effective_session_id', 'decision_session_id', 'from_contract', 'to_contract',
                'from_open_interest', 'to_open_interest']


def _ordered(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.sort_values(keys, kind='stable').reset_index(drop=True)


def _a_share(connection: duckdb.DuckDBPyConnection, path: Path,
             candidate: bool) -> dict[str, pd.DataFrame]:
    start, end = A_SHARE_DATES
    close = pd.Timestamp(f'{end} 07:00Z')
    anchor_day = '2022-07-06'
    if candidate:
        from slots.jq_ashare_local import build_jq_ashare_local

        slot = build_jq_ashare_local(connection, start=start, end=end)
    else:
        adapter = DuckDBAdapter(duckdb_path=str(path), data_bindings={})
        adapter._conn = connection
        manager = DynamicStockPoolManager(connection, price_table='price_data_daily', data_bindings={})
    evidence = {}
    for frequency in ('daily', '5m'):
        intraday = frequency == '5m'
        parsed = Frequency.parse(frequency)
        fields = COMMON if intraday else [*COMMON, 'paused', 'high_limit', 'low_limit', 'pre_close']
        anchor = pd.Timestamp(f'{anchor_day} {"02:00" if intraday else "07:00"}Z')
        opening = pd.Timestamp(f'{start} 01:30Z')
        if candidate:
            execution = slot.bar_feed.read_execution_bars(SYMBOLS, parsed, start_utc=opening, end_utc=close)
            signal = slot.bar_feed.read_signal_bars(SYMBOLS, parsed, start_utc=opening, end_utc=close,
                                                    as_of_utc=anchor)
        else:
            raw = adapter.fetch_price_data(SYMBOLS, start, end, fields=fields, frequency=frequency)
            labels = raw.date.dt.strftime('%Y-%m-%d')
            times = (raw.date.dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC') - parsed.duration) if intraday else (
                pd.to_datetime(labels + ' 07:00Z', utc=True))
            normalized = raw.assign(date=times, session_id=labels)
            available = times + (parsed.duration if intraday else pd.Timedelta(0))
            execution = normalized.rename(columns={'code': 'instrument_id', 'date': 'timestamp_utc'})
            visible = normalize_price_money(normalized.loc[available.le(anchor)], source='local', frequency=frequency)
            signal = apply_pit_adjustment(visible).rename(columns={'code': 'instrument_id', 'date': 'timestamp_utc'})
        for name, frame in [('execution', execution), ('signal', signal)]:
            assert not frame.empty, (name, frequency)
            evidence[f'ashare.{frequency}.{name}'] = _ordered(frame[[*BAR_KEYS, *fields]], BAR_KEYS)
        if not intraday:
            dates = sorted(execution.session_id.unique())
            assert execution.groupby('instrument_id').factor.nunique().gt(1).any(), 'Expected a real corporate action'

    for day in ('2022-06-01', '2022-06-24', '2022-07-08'):
        for pool, index in [('csi300', '000300.XSHG'), ('csi500', '000905.XSHG'), ('wind_micro', '8841431.WI')]:
            if candidate:
                members = slot.universe.resolve_universe(pool, as_of_utc=pd.Timestamp(f'{day} 07:00Z'))
            else:
                filters = [StockFilter('paused', '==', 0), StockFilter('is_st', '==', 0),
                           StockFilter('index_code', '==', index)]
                limit = None
                if pool == 'wind_micro':
                    filters += [StockFilter('is_new_stock', '==', 0),
                                StockFilter('val_market_cap', 'percentile_between', (20, 80))]
                    limit = 500
                members = manager._select_stock_pool(pd.Timestamp(day).to_pydatetime(), filters, max_stocks=limit)
            evidence[f'ashare.universe.{pool}.{day}'] = pd.DataFrame({'instrument_id': members})

    grid = pd.DataFrame({'session_id': dates, 'timestamp_utc': pd.to_datetime(
        [f'{day} 07:00Z' for day in dates], utc=True)}).merge(pd.DataFrame({'instrument_id': SYMBOLS}), how='cross')
    if candidate:
        factor = slot.read_aux_features(grid, features=['val_pe_ratio'])
        financial = slot.read_aux_features(grid, features=[FINANCIAL])
    else:
        factor = adapter.fetch_factor_data(SYMBOLS, start, end, fields=['val_pe_ratio'])
        factor = factor.assign(session_id=factor.date.dt.strftime('%Y-%m-%d')).rename(columns={'code': 'instrument_id'})
        factor = grid.merge(factor, on=['instrument_id', 'session_id']).assign(available_at_utc=lambda x: x.timestamp_utc)
        financial = connection.execute(f'''SELECT instrument_id, session_id, timestamp_utc, pub_date,
                operating_revenue AS "{FINANCIAL}"
            FROM grid JOIN raw_income_statement_events ON code = instrument_id
              AND pub_date = (SELECT max(pub_date) FROM raw_income_statement_events
                              WHERE code = instrument_id AND pub_date < session_id::DATE)''').fetchdf()
        financial['available_at_utc'] = (
            pd.to_datetime(financial.pub_date).dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC')
            + pd.Timedelta(days=1))
    keys = [*BAR_KEYS, 'available_at_utc']
    assert not factor.empty and not financial.empty
    evidence['ashare.factor'] = _ordered(factor[[*keys, 'val_pe_ratio']], keys)
    evidence['ashare.financial'] = _ordered(financial[[*keys, FINANCIAL]], [*keys, FINANCIAL])
    return evidence


def _futures(sample: duckdb.DuckDBPyConnection, candidate: bool) -> dict[str, pd.DataFrame]:
    from contextlib import closing

    start, first, end = FUTURES_DATES
    evidence = {}
    raw = sample.table('futures_quotes').fetchdf()
    for root, quotes in raw.groupby('product_root', sort=True):
        spec = futures_calendar(root)
        reference = spec['rth_reference_calendar']
        calendar = SessionCalendar(spec['calendar_id'], start, end, rth_reference_calendar=reference)
        schedule = BarSchedule(calendar, '1d', bucket_strategy='segment_anchored')
        sessions = {day.strftime('%Y-%m-%d'): schedule.for_session(day).iloc[0] for day in calendar.sessions()}
        times = quotes.trading_day.dt.strftime('%Y-%m-%d')
        bars = quotes.assign(timestamp_utc=[sessions[day].start_utc for day in times],
                             source='tq_snapshot', data_version='2022-bounded-sample', frequency='1d')
        metadata = quotes[['instrument_id', 'product_root', 'exchange', 'expiry']].drop_duplicates()
        metadata = metadata.rename(columns={'expiry': 'contract_expiry'}).assign(
            asset_class='futures', timezone=calendar.timezone, price_adjustment='none',
            point_value=5 if root == 'TA' else 10, calendar_id=spec['calendar_id'],
            rth_reference_calendar=reference, contract_month=lambda x: x.contract_expiry.str[:7])
        with closing(duckdb.connect(config={'threads': 1})) as connection:
            import_market_bars(connection, bars, metadata, bucket_strategy='segment_anchored', legacy_compatibility=False)
            build_roll_schedule(connection, product_roots=[root], start=first, end=end, rule='max_oi')
            columns = ', '.join(ROLL_COLUMNS)
            evidence[f'futures.{root}.roll'] = connection.execute(
                f'SELECT {columns} FROM futures_roll_schedule ORDER BY effective_session_id').fetchdf()
            grid = {day: session.end_utc for day, session in sessions.items()}
            lo, hi = grid[first], grid[end]
            fields = [*PRICES, 'volume', 'open_interest']
            for adjustment, legacy in [('none', 'none'), ('ratio_back', 'ratio_back_adjust'),
                                       ('panama_canal', 'diff_back_adjust')]:
                if candidate:
                    from slots.cn_futures import build_cn_futures_slot

                    slot = build_cn_futures_slot(connection, product_root=root, start=start, end=end,
                                                 adjustment=adjustment)
                    frame = slot.bar_feed.read_signal_bars([root], Frequency.parse('1d'),
                                                           start_utc=lo, end_utc=hi, as_of_utc=hi)
                else:
                    frame = ContinuousBarSource(connection).read(root, '1d', first, end, adjust=legacy)
                    frame = frame.assign(instrument_id=root, timestamp_utc=frame.session_id.map(grid))
                    frame = frame.rename(columns={'contract_id': 'concrete_instrument_id'})
                assert not frame.empty
                evidence[f'futures.{root}.{adjustment}'] = frame[[*BAR_KEYS, 'concrete_instrument_id', *fields]]
    return evidence


def _compare(actual: Mapping[str, pd.DataFrame], expected: Mapping[str, pd.DataFrame]) -> dict:
    assert actual.keys() == expected.keys()
    evidence = {}
    for name, frame in actual.items():
        reference = expected[name]
        try:
            pd.testing.assert_index_equal(frame.columns, reference.columns)
            assert len(frame) == len(reference) > 0
            fields = frame.select_dtypes(include='number').columns
            keys = frame.columns.difference(fields, sort=False)
            for key in keys:
                pd.testing.assert_series_equal(frame[key].reset_index(drop=True), reference[key].reset_index(drop=True),
                                               check_dtype=False, check_exact=True)
            left, right = frame[fields].to_numpy(float), reference[fields].to_numpy(float)
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-12, equal_nan=True)
        except AssertionError as exc:
            raise AssertionError(f'{name}: {exc}') from exc
        finite = np.isfinite(left) & np.isfinite(right)
        evidence[name] = {'rows': len(frame), 'compared_values': left.size,
                          'matching_nan_values': int(np.isnan(left).sum()),
                          'max_abs_error': float(np.max(np.abs(left[finite] - right[finite]), initial=0.))}
        if name.endswith('.roll'):
            changes = frame.from_contract.notna() & frame.from_contract.ne(frame.to_contract)
            evidence[name]['contract_switches'] = int(changes.sum())
    assert sum(case.get('contract_switches', 0) for case in evidence.values()) > 0
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reference', 'candidate', 'compare'))
    parser.add_argument('sample', type=Path)
    parser.add_argument('reference', type=Path, help='Directory for the Parquet comparison results')
    parser.add_argument('--actual', type=Path, help='Captured candidate results for comparison')
    args = parser.parse_args()
    if args.mode == 'compare':
        if args.actual is None:
            parser.error('compare requires --actual')
        reference = {file.stem: pd.read_parquet(file) for file in args.reference.glob('*.parquet')}
        actual = {file.stem: pd.read_parquet(file) for file in args.actual.glob('*.parquet')}
        evidence = _compare(actual, reference)
        with args.sample.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        print(json.dumps({'sample_sha256': digest, 'compared': evidence}, indent=2, sort_keys=True))
        return
    candidate = args.mode == 'candidate'
    with duckdb.connect(str(args.sample), read_only=True, config={'threads': 2, 'TimeZone': 'UTC'}) as connection:
        snapshot = {**_a_share(connection, args.sample, candidate), **_futures(connection, candidate)}
    args.reference.mkdir()
    for name, frame in snapshot.items():
        frame.attrs.clear()
        frame.to_parquet(args.reference / f'{name}.parquet', index=False)


if __name__ == '__main__':
    main()
