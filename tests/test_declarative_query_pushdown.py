"""Regression and pushdown verification tests for declarative FrameSource and DeclarativeBarFeed."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import shutil

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backtest_and_analysis.time_series.contracts import ContractValidationError
from market_db_samples import crypto_profile, market_profile
from slots import build_slot_from_config
from slots.declarative.source import FrameSource, ReadRequest
from utils.market_db import init_market_db
from utils.market_db.storage import _create_profile_table
from utils.slots import ProviderConfig, load_slot_config
from utils.time_axis import Frequency

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'declarative_crypto'
BASE_CONFIG = load_slot_config(EXAMPLE / 'slot.yaml')


def _multi_instrument_config(driver: str, source_opts: dict, *, frequency: str = '1h',
                             timestamp_anchor: str = 'bar_start', volume_mode: str = 'interval',
                             source_timezone: str = 'UTC', calendar: ProviderConfig | None = None,
                             instrument_id: str | None = None,
                             column_mapping: dict[str, str] | None = None,
                             bucket_strategy: str | None = None):
    universe_opts = dict(BASE_CONFIG.universe.options)
    base_spec = dict(universe_opts['instruments']['BTC-USD'])
    universe_opts['instruments'] = {
        'BTC-USD': dict(base_spec),
        'ETH-USD': dict(base_spec),
        'SOL-USD': dict(base_spec),
    }
    universe_opts['universes'] = {'spot': ['BTC-USD', 'ETH-USD', 'SOL-USD']}
    bar_opts = dict(BASE_CONFIG.bar_feed.options)
    bar_opts.pop('paths', None)
    bar_opts.update(
        frequency=frequency,
        timestamp_anchor=timestamp_anchor,
        volume_mode=volume_mode,
        source_timezone=source_timezone,
        **source_opts,
    )
    if column_mapping is not None:
        bar_opts['column_mapping'] = column_mapping
    if instrument_id is not None:
        bar_opts['instrument_id'] = instrument_id
    if bucket_strategy is not None:
        bar_opts['bucket_strategy'] = bucket_strategy
    return replace(
        BASE_CONFIG,
        calendar=calendar or BASE_CONFIG.calendar,
        bar_feed=ProviderConfig(driver, bar_opts),
        universe=ProviderConfig('static', universe_opts),
    )


def _sample_multi_bars(*, include_extras: bool = True, include_wide: bool = True) -> pd.DataFrame:
    rows = []
    for inst_idx, inst in enumerate(('BTC-USD', 'ETH-USD', 'SOL-USD')):
        base_px = 100.0 * (inst_idx + 1)
        for day_offset in range(3):
            for hour in range(4):
                ts = pd.Timestamp('2026-01-03 00:00Z') + pd.Timedelta(days=day_offset, hours=hour)
                row = {
                    'symbol': inst,
                    'time': ts,
                    'open': base_px + hour,
                    'high': base_px + hour + 2.0,
                    'low': base_px + hour - 1.0,
                    'close': base_px + hour + 1.0,
                    'volume': 10.0 * (hour + 1),
                }
                if include_extras:
                    row['money'] = row['close'] * row['volume']
                    row['factor'] = 1.0
                    row['open_interest'] = 50.0 + hour
                if include_wide:
                    for col_i in range(12):
                        row[f'wide_{col_i:02d}'] = f'unused_{inst}_{day_offset}_{hour}_{col_i}'
                rows.append(row)
    return pd.DataFrame(rows)


def _dev_unpushed_read(source: FrameSource) -> pd.DataFrame:
    """Independent unpushed full-scan baseline mirroring dev FrameSource.read()."""
    from glob import glob
    import csv as _csv

    options = source.provider.options
    driver = source.provider.provider
    mapping = options.get('column_mapping', {})
    if driver == 'duckdb_table':
        table_sql = '.'.join(f'"{part}"' for part in options['table'].split('.'))
        with duckdb.connect(str(source.database_path), read_only=True) as conn:
            conn.execute("SET TimeZone='UTC'")
            frame = conn.execute(f'SELECT * FROM {table_sql}').fetchdf()
        timestamp = next((col for col, dst in mapping.items() if dst == 'timestamp'), None)
        source_timezone = options.get('source_timezone')
        if (source_timezone and timestamp in frame
                and isinstance(frame[timestamp].dtype, pd.DatetimeTZDtype)):
            frame[timestamp] = frame[timestamp].dt.tz_convert(source_timezone)
        return frame
    if driver == 'parquet_glob':
        paths = sorted(glob(str(source.base_dir / Path(options['path']).expanduser()), recursive=True))
        return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    reverse = {dst: src for src, dst in mapping.items()}
    text_columns = {reverse.get(field, field): 'string' for field in source.text_fields}
    frames = []
    for value in options['paths']:
        path = source.base_dir / Path(value).expanduser()
        with path.open(encoding='utf-8-sig', newline='') as handle:
            header = next(_csv.reader(handle), [])
            assert len(set(header)) == len(header)
            handle.seek(0)
            frames.append(pd.read_csv(handle, dtype=text_columns, keep_default_na=False, na_values=['']))
    return pd.concat(frames, ignore_index=True)


def _baseline_read_bar_feed(feed, instruments, frequency, *, start_utc, end_utc, as_of_utc=None):
    """Unpushed full-scan baseline mirroring original dev DeclarativeBarFeed behavior."""
    raw = _dev_unpushed_read(feed.source)
    inst_col = next((src for src, dst in feed.spec.column_mapping.items() if dst == 'instrument_id'), None)
    if inst_col is not None:
        raw = raw.loc[raw[inst_col].isin(instruments)]
    elif feed.instrument_id not in instruments:
        raw = raw.iloc[:0]
    bars = feed._adapt_bars(raw, as_of_utc=as_of_utc)
    if feed.frequency.kind == 'intraday':
        bars = bars.loc[bars.timestamp_utc.between(start_utc, end_utc)].copy()
    ends = {ts: feed.schedule.end_at(ts) for ts in bars.timestamp_utc.unique()}
    end_times = pd.to_datetime(bars.timestamp_utc.map(ends), utc=True)
    days = bars.timestamp_utc if feed.frequency.kind == 'intraday' else end_times - pd.Timedelta(1, 'ns')
    view = getattr(feed, 'view', 'ETH')
    owners = [feed.calendar.session_at(ts, view).session_date for ts in days]
    bars = bars.assign(session_id=pd.Series(owners, index=bars.index, dtype=object))
    if feed.frequency.kind != 'intraday':
        keys = feed.calendar.bar_grid(feed.frequency, view=view).set_index('session_id').timestamp_utc
        bars = bars.assign(timestamp_utc=bars.session_id.map(keys))
    mask = bars.timestamp_utc.between(start_utc, end_utc)
    extras = [field for field in ('money', 'factor', 'open_interest') if field in bars]
    from backtest_and_analysis.time_series.contracts import BAR_COLUMNS
    return bars.loc[mask, [*BAR_COLUMNS, 'session_id', *extras]].reset_index(drop=True)


def _write_driver_dataset(tmp_path: Path, driver: str, df: pd.DataFrame, *,
                          as_iso_string: bool = False, profile=None) -> dict:
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    out = df.copy()
    if as_iso_string and 'time' in out:
        out['time'] = out['time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    if driver == 'csv_files':
        out.iloc[:len(out) // 2].to_csv(tmp_path / 'part_1.csv', index=False)
        out.iloc[len(out) // 2:].to_csv(tmp_path / 'part_2.csv', index=False)
        return {'paths': ['part_1.csv', 'part_2.csv']}
    if driver == 'parquet_glob':
        out.iloc[:len(out) // 2].to_parquet(tmp_path / 'part_1.parquet', index=False)
        out.iloc[len(out) // 2:].to_parquet(tmp_path / 'part_2.parquet', index=False)
        return {'path': 'part_*.parquet'}
    db_path = tmp_path / 'bars.duckdb'
    with duckdb.connect(str(db_path)) as conn:
        _create_profile_table(conn, profile or crypto_profile())
        conn.register('src_df', out)
        conn.execute('CREATE TABLE bars AS SELECT * FROM src_df')
    return {'path': 'bars.duckdb', 'table': 'main.bars'}


@pytest.mark.parametrize('driver', ['duckdb_table', 'parquet_glob', 'csv_files'])
@pytest.mark.parametrize('volume_mode', ['interval', 'cumulative'])
def test_intraday_pushdown_matches_full_baseline_across_drivers(tmp_path, driver, volume_mode):
    raw = _sample_multi_bars(include_extras=True, include_wide=True)
    if volume_mode == 'cumulative':
        # Cumulative volume resets each session and must be monotonic per instrument/day.
        raw = raw.copy()
        day_key = raw['time'].dt.strftime('%Y-%m-%d')
        raw['volume'] = raw.groupby(['symbol', day_key])['volume'].cumsum()
    src_opts = _write_driver_dataset(tmp_path, driver, raw, as_iso_string=(driver == 'csv_files'))
    cfg = _multi_instrument_config(driver, src_opts, frequency='1h', timestamp_anchor='bar_start',
                                   volume_mode=volume_mode)
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    start = pd.Timestamp('2026-01-04 01:00Z')
    end = pd.Timestamp('2026-01-05 02:00Z')
    as_of = pd.Timestamp('2026-01-04 03:00Z')

    expected_exec = _baseline_read_bar_feed(feed, ['BTC-USD', 'SOL-USD'], Frequency.parse('1h'),
                                            start_utc=start, end_utc=end)
    actual_exec = feed.read_execution_bars(['BTC-USD', 'SOL-USD'], Frequency.parse('1h'),
                                           start_utc=start, end_utc=end)
    pd.testing.assert_frame_equal(actual_exec, expected_exec)

    expected_sig = _baseline_read_bar_feed(feed, ['BTC-USD', 'SOL-USD'], Frequency.parse('1h'),
                                           start_utc=start, end_utc=end, as_of_utc=as_of)
    actual_sig = feed.read_signal_bars(['BTC-USD', 'SOL-USD'], Frequency.parse('1h'),
                                         start_utc=start, end_utc=end, as_of_utc=as_of)
    pd.testing.assert_frame_equal(actual_sig, expected_sig)
    assert set(actual_exec.columns) >= {'money', 'factor', 'open_interest'}


@pytest.mark.parametrize('driver', ['duckdb_table', 'parquet_glob', 'csv_files'])
@pytest.mark.parametrize('anchor', ['bar_start', 'bar_end'])
def test_daily_dst_and_timezone_pushdown_matches_baseline(tmp_path, driver, anchor):
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    # NYSE 2026-03-06 (EST, -05:00 -> close 21:00Z) and 2026-03-09 (EDT, -04:00 -> close 20:00Z)
    times = (
        ['2026-03-06 09:30', '2026-03-09 09:30'] if anchor == 'bar_start'
        else ['2026-03-06 16:00', '2026-03-09 16:00']
    )
    raw = pd.DataFrame({
        'time': times if driver != 'parquet_glob' else pd.to_datetime(times),
        'open': [100.0, 102.0],
        'high': [105.0, 106.0],
        'low': [99.0, 101.0],
        'close': [103.0, 104.0],
        'volume': [50.0, 60.0],
        'wide_unused': ['x', 'y'],
    })
    if driver == 'csv_files':
        raw.to_csv(tmp_path / 'nyse.csv', index=False)
        src_opts = {'paths': ['nyse.csv']}
    elif driver == 'parquet_glob':
        raw.to_parquet(tmp_path / 'nyse.parquet', index=False)
        src_opts = {'path': 'nyse.parquet'}
    else:
        db_path = tmp_path / 'nyse.duckdb'
        profile = market_profile('us_equity', calendar_coverage='2026-03-06..2026-03-09')
        init_market_db(db_path, profile)
        with duckdb.connect(str(db_path)) as conn:
            conn.register('raw_df', raw)
            conn.execute('CREATE TABLE nyse AS SELECT * FROM raw_df')
        src_opts = {'path': 'nyse.duckdb', 'table': 'nyse'}

    mapping = {name: name for name in ('open', 'high', 'low', 'close', 'volume')}
    mapping['time'] = 'timestamp'
    cal = ProviderConfig('pmc', {'calendar_id': 'XNYS', 'start': '2026-03-06', 'end': '2026-03-09'})
    universe_opts = {
        'instruments': {'AAPL.XNAS': {'asset_class': 'equity', 'currency': 'USD'}},
        'universes': {'spot': ['AAPL.XNAS']},
    }
    cfg = replace(
        BASE_CONFIG,
        market_domain='us_equity',
        calendar=cal,
        bar_feed=ProviderConfig(driver, {
            'frequency': '1d',
            'timestamp_anchor': anchor,
            'source_timezone': 'America/New_York',
            'volume_mode': 'interval',
            'data_version': 'v1',
            'instrument_id': 'AAPL.XNAS',
            'column_mapping': mapping,
            **src_opts,
        }),
        universe=ProviderConfig('static', universe_opts),
    )
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    # Query exact daily grid key for 2026-03-09 (20:00Z); raw bar_start is 13:30Z (earlier than start!).
    start = pd.Timestamp('2026-03-09 20:00Z')
    end = pd.Timestamp('2026-03-09 20:00Z')
    expected = _baseline_read_bar_feed(feed, ['AAPL.XNAS'], Frequency.parse('1d'), start_utc=start, end_utc=end)
    actual = feed.read_execution_bars(['AAPL.XNAS'], Frequency.parse('1d'), start_utc=start, end_utc=end)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.session_id.tolist() == ['2026-03-09']
    assert feed.read_execution_bars(['MSFT.XNAS'], Frequency.parse('1d'), start_utc=start, end_utc=end).empty


def test_intraday_bar_end_session_clock_lunch_break_buffering(tmp_path):
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    # Session has two segments with a 2h lunch break (01:00..02:00 and 04:00..05:00 UTC).
    # Under session_clock 90min: bar 0 is 01:00..04:30 (crosses break!), bar 1 is 04:30..05:00.
    session = {
        'timezone': 'UTC',
        'weekdays': list(range(7)),
        'segments': [{'open': '01:00', 'close': '02:00'}, {'open': '04:00', 'close': '05:00'}],
    }
    raw = pd.DataFrame({
        'symbol': ['BTC-USD', 'BTC-USD'],
        'time': ['2026-01-03T04:30:00Z', '2026-01-03T05:00:00Z'],
        'open': [100.0, 101.0],
        'high': [102.0, 103.0],
        'low': [99.0, 100.0],
        'close': [101.0, 102.0],
        'volume': [10.0, 20.0],
    })
    raw.to_csv(tmp_path / 'bars.csv', index=False)
    cal = ProviderConfig('custom_sessions', {
        'calendar_id': 'lunch_test', 'start': '2026-01-03', 'end': '2026-01-03', 'custom_sessions': session,
    })
    cfg = _multi_instrument_config(
        'csv_files', {'paths': ['bars.csv']}, frequency='90min', timestamp_anchor='bar_end',
        calendar=cal, bucket_strategy='session_clock',
    )
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed
    start, end = pd.Timestamp('2026-01-03 01:00Z'), pd.Timestamp('2026-01-03 01:00Z')
    expected = _baseline_read_bar_feed(feed, ['BTC-USD'], Frequency.parse('90min'), start_utc=start, end_utc=end)
    actual = feed.read_execution_bars(['BTC-USD'], Frequency.parse('90min'), start_utc=start, end_utc=end)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.timestamp_utc.tolist() == [start]
    assert actual.close.tolist() == [101.0]


def test_duckdb_pushdown_sql_projection_and_parameters(tmp_path, monkeypatch):
    raw = _sample_multi_bars(include_extras=False, include_wide=True)
    src_opts = _write_driver_dataset(tmp_path, 'duckdb_table', raw)
    cfg = _multi_instrument_config('duckdb_table', src_opts)
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    executed: list[tuple[str, list]] = []
    real_open = duckdb.connect

    class SpyConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, query, parameters=None):
            executed.append((query, list(parameters) if parameters is not None else []))
            return self._inner.execute(query, parameters) if parameters is not None else self._inner.execute(query)

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

    monkeypatch.setattr('utils.market_db.storage.duckdb.connect',
                        lambda *a, **kw: SpyConnection(real_open(*a, **kw)))

    start = pd.Timestamp('2026-01-04 01:00Z')
    end = pd.Timestamp('2026-01-04 02:00Z')
    bars = feed.read_signal_bars(['ETH-USD'], Frequency.parse('1h'),
                                 start_utc=start, end_utc=end, as_of_utc=end)
    assert len(bars) == 1
    select_calls = [(q, p) for q, p in executed if q.lstrip().upper().startswith('SELECT') and '"main"."bars"' in q]
    assert len(select_calls) == 1
    sql, params = select_calls[0]
    assert 'WHERE' in sql
    assert 'IN (?)' in sql
    assert 'wide_00' not in sql
    assert 'ETH-USD' in params and 'ETH-USD' not in sql
    assert isinstance(params[-1], datetime)

    # Direct FrameSource column pruning only returns requested source columns.
    pruned = feed.source.read(ReadRequest(instruments=['BTC-USD'], start_utc=start, end_utc=end,
                                          columns=('timestamp', 'close')))
    assert list(pruned.columns) == ['time', 'close']
    assert len(pruned) == 2


def test_parquet_pushdown_column_pruning_and_row_group_skipping(tmp_path, monkeypatch):
    raw = _sample_multi_bars(include_extras=False, include_wide=True)
    # Write a single Parquet file with 3 distinct row groups (one per day) sorted by time.
    raw = raw.sort_values(['time', 'symbol']).reset_index(drop=True)
    pq_path = tmp_path / 'grouped.parquet'
    table = pa.Table.from_pandas(raw, preserve_index=False)
    with pq.ParquetWriter(pq_path, table.schema) as writer:
        for i in range(0, len(table), 12):
            writer.write_table(table.slice(i, 12))

    cfg = _multi_instrument_config('parquet_glob', {'path': 'grouped.parquet'})
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    calls: list[dict] = []
    real_dataset = pq.ParquetDataset

    class SpyParquetDataset:
        def __init__(self, path, filters=None, **kwargs):
            self._inner = real_dataset(path, filters=filters, **kwargs)
            calls.append({'filters': filters})

        @property
        def schema(self):
            return self._inner.schema

        def read(self, columns=None, **kwargs):
            calls[-1]['columns'] = columns
            return self._inner.read(columns=columns, **kwargs)

    monkeypatch.setattr(pq, 'ParquetDataset', SpyParquetDataset)
    start = pd.Timestamp('2026-01-05 00:00Z')
    end = pd.Timestamp('2026-01-05 02:00Z')
    result = feed.read_execution_bars(['SOL-USD'], Frequency.parse('1h'), start_utc=start, end_utc=end)
    assert len(result) == 3
    read_calls = [c for c in calls if 'columns' in c]
    assert len(read_calls) == 1
    assert read_calls[0]['filters'] is not None
    assert 'wide_00' not in read_calls[0]['columns']
    assert set(read_calls[0]['columns']) == {'symbol', 'time', 'open', 'high', 'low', 'close', 'volume'}


def test_csv_pushdown_usecols_cache_and_invalidation(tmp_path, monkeypatch):
    # Create two multi-chunk CSV files (> 512 rows each) with BOM + wide columns.
    rows_a = []
    rows_b = []
    for i in range(600):
        ts_a = (pd.Timestamp('2026-01-03 00:00Z') + pd.Timedelta(minutes=i)).strftime('%Y-%m-%dT%H:%M:%SZ')
        ts_b = (pd.Timestamp('2026-01-10 00:00Z') + pd.Timedelta(minutes=i)).strftime('%Y-%m-%dT%H:%M:%SZ')
        rows_a.append({'symbol': 'BTC-USD', 'time': ts_a, 'open': 100.0, 'high': 101.0,
                       'low': 99.0, 'close': 100.5, 'volume': 1.0, 'wide_01': 'skip'})
        rows_b.append({'symbol': 'ETH-USD', 'time': ts_b, 'open': 200.0, 'high': 201.0,
                       'low': 199.0, 'close': 200.5, 'volume': 2.0, 'wide_01': 'skip'})
    (tmp_path / 'a.csv').write_text(pd.DataFrame(rows_a).to_csv(index=False), encoding='utf-8-sig')
    (tmp_path / 'b.csv').write_text(pd.DataFrame(rows_b).to_csv(index=False), encoding='utf-8-sig')

    source = FrameSource(
        ProviderConfig('csv_files', {
            'paths': ['a.csv', 'b.csv'],
            'column_mapping': {'symbol': 'instrument_id', 'time': 'timestamp', 'close': 'close'},
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    req = ReadRequest(
        instruments=['ETH-USD'],
        start_utc=pd.Timestamp('2026-01-10 09:00Z'),
        end_utc=pd.Timestamp('2026-01-10 09:10Z'),
        columns=('instrument_id', 'timestamp', 'close'),
    )

    read_csv_calls: list[dict] = []
    real_read_csv = pd.read_csv

    def spy_read_csv(*args, **kwargs):
        read_csv_calls.append({'usecols': kwargs.get('usecols')})
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', spy_read_csv)
    first = source.read(req)
    assert list(first.columns) == ['symbol', 'time', 'close']
    assert str(first['symbol'].dtype) == 'string'
    assert str(first['time'].dtype) == 'string'
    assert len(first) == 11
    assert all(call['usecols'] is not None and 'wide_01' not in call['usecols'] for call in read_csv_calls)

    # Warm cache hit: a.csv is skipped with 0 pd.read_csv calls, and only 1 of 2 chunks in b.csv is read.
    read_csv_calls.clear()
    second = source.read(req)
    pd.testing.assert_frame_equal(second, first)
    assert len(read_csv_calls) == 1
    assert set(read_csv_calls[0]['usecols']) == {'symbol', 'time', 'close'}

    # Modify b.csv -> mtime/size changes and cache invalidates cleanly.
    rows_b.append({'symbol': 'ETH-USD', 'time': '2026-01-10T09:05:30Z', 'open': 200.0, 'high': 201.0,
                   'low': 199.0, 'close': 205.0, 'volume': 3.0, 'wide_01': 'skip'})
    (tmp_path / 'b.csv').write_text(pd.DataFrame(rows_b).to_csv(index=False), encoding='utf-8-sig')
    third = source.read(req)
    assert len(third) == 12
    assert 205.0 in third['close'].tolist()


def test_membership_aux_and_microstructure_pushdown(tmp_path):
    shutil.copy(EXAMPLE / 'bars.csv', tmp_path)
    pd.DataFrame({
        'instrument_id': ['BTC-USD', 'ETH-USD'],
        'available_at_utc': ['2026-01-03T00:00:00Z', '2026-01-03T05:00:00Z'],
        'funding_rate': [0.0001, 0.0009],
        'unused_feature': [99.0, 88.0],
        'wide_aux': ['a', 'b'],
    }).to_csv(tmp_path / 'funding.csv', index=False)
    pd.DataFrame({
        'pool': ['spot', 'spot'],
        'symbol': ['BTC-USD', 'ETH-USD'],
        'seen': ['2026-01-03T01:00:00Z', '2026-01-03T02:00:00Z'],
        'wide_member': [1, 2],
    }).to_csv(tmp_path / 'membership.csv', index=False)
    pd.DataFrame({
        'instrument_id': ['BTC-USD', 'BTC-USD', 'BTC-USD', 'ETH-USD'],
        'available_at_utc': ['2026-01-03T02:00:00Z', '2026-01-03T03:00:00Z',
                             '2026-01-03T10:00:00Z', '2026-01-03T01:00:00Z'],
        'tradable': [False, True, False, False],
        'wide_micro': ['m1', 'm2', 'm3', 'm4'],
    }).to_csv(tmp_path / 'halts.csv', index=False)

    option = BASE_CONFIG.universe.options['instruments']['BTC-USD']
    cfg = replace(
        BASE_CONFIG,
        universe=ProviderConfig('csv_files', {
            'paths': ['membership.csv'],
            'column_mapping': {'pool': 'universe_id', 'symbol': 'instrument_id', 'seen': 'available_at_utc'},
            'instruments': {'BTC-USD': dict(option), 'ETH-USD': dict(option)},
        }),
        microstructure=ProviderConfig('csv_files', {'paths': ['halts.csv']}),
    )
    slot = build_slot_from_config(cfg, base_dir=tmp_path)
    assert slot.universe.resolve_universe('spot', as_of_utc=pd.Timestamp('2026-01-03 00:00Z')) == ()
    assert slot.universe.resolve_universe('spot', as_of_utc=pd.Timestamp('2026-01-03 01:00Z')) == ('BTC-USD',)
    grid = slot.calendar.bar_grid('1h', '2026-01-03', '2026-01-03').iloc[:4]
    keyed = grid.assign(instrument_id='BTC-USD')
    aux = slot.read_aux_features(keyed, features=['funding_rate'])
    assert aux.funding_rate.tolist() == [0.0001] * 4
    assert 'wide_aux' not in aux.columns
    assert slot.microstructure.tradability_mask(grid, instrument_id='BTC-USD').tolist() == [True, True, False, True]
    with pytest.raises(ValueError, match='Unknown auxiliary features'):
        slot.read_aux_features(keyed, features=['missing_feat'])


def test_read_request_validation_and_security_rejections(tmp_path):
    with pytest.raises(ValueError, match='start_utc must not follow end_utc'):
        ReadRequest(
            start_utc=datetime(2026, 1, 4, tzinfo=timezone.utc),
            end_utc=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
    with pytest.raises(ValueError, match='columns must not be empty'):
        ReadRequest(columns=())
    with pytest.raises(ValueError, match='Invalid column name'):
        ReadRequest(columns=('',))
    with pytest.raises(ValueError, match='Invalid column name'):
        ReadRequest(optional_columns=('bad\x00col',))
    with pytest.raises(ValueError, match='instruments must be a sequence'):
        ReadRequest(instruments='BTC-USD')
    with pytest.raises(ValueError, match='UTC'):
        ReadRequest(start_utc=datetime(2026, 1, 3))

    # Spaces and hyphens are valid column / feature names in ReadRequest.
    req_with_spaces = ReadRequest(columns=('Close Price',), optional_columns=('funding-rate',))
    assert req_with_spaces.columns == ('Close Price',)
    assert req_with_spaces.optional_columns == ('funding-rate',)

    db_path = tmp_path / 'bad.duckdb'
    with duckdb.connect(str(db_path)) as conn:
        _create_profile_table(conn, crypto_profile())
        conn.execute('CREATE TABLE bars (symbol VARCHAR, time TIMESTAMPTZ, close DOUBLE, "a""; DROP TABLE x; --" DOUBLE)')
        conn.execute('CREATE TABLE x (val INTEGER)')
        conn.execute(
            """INSERT INTO bars VALUES ('BTC-USD', TIMESTAMPTZ '2026-01-03 00:00:00Z', 100.0, 42.5)"""
        )
        conn.execute('INSERT INTO x VALUES (7)')

    bad_table = FrameSource(
        ProviderConfig('duckdb_table', {'path': 'bad.duckdb', 'table': 'bars; DROP TABLE bars'}),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    with pytest.raises(ValueError, match='table name'):
        bad_table.read()

    bad_mapping = FrameSource(
        ProviderConfig('duckdb_table', {
            'path': 'bad.duckdb', 'table': 'bars',
            'column_mapping': {'close" FROM bars --': 'close'},
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    with pytest.raises(ValueError, match='missing or duplicate source columns'):
        bad_mapping.read()

    valid_source = FrameSource(
        ProviderConfig('duckdb_table', {
            'path': 'bad.duckdb', 'table': 'bars',
            'column_mapping': {'symbol': 'instrument_id', 'time': 'timestamp', 'close': 'close'},
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    with pytest.raises(ValueError, match='Unknown requested columns'):
        valid_source.read(ReadRequest(columns=('close"; DROP TABLE x; --',)))

    # Real column containing quotes/semicolons is safely double-quote escaped without executing SQL.
    tricky_col = 'a"; DROP TABLE x; --'
    quoted_df = valid_source.read(ReadRequest(
        instruments=['BTC-USD'],
        columns=('instrument_id', tricky_col),
    ))
    assert quoted_df[tricky_col].tolist() == [42.5]
    with duckdb.connect(str(db_path), read_only=True) as conn:
        assert conn.execute('SELECT val FROM x').fetchall() == [(7,)]

    no_inst_source = FrameSource(
        ProviderConfig('duckdb_table', {
            'path': 'bad.duckdb', 'table': 'bars',
            'column_mapping': {'time': 'timestamp', 'close': 'close'},
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    with pytest.raises(ValueError, match='instrument_id'):
        no_inst_source.read(ReadRequest(instruments=['BTC-USD']))

    no_time_source = FrameSource(
        ProviderConfig('duckdb_table', {
            'path': 'bad.duckdb', 'table': 'bars',
            'column_mapping': {'symbol': 'instrument_id', 'close': 'close'},
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    with pytest.raises(ValueError, match='Time filter'):
        no_time_source.read(ReadRequest(start_utc=datetime(2026, 1, 3, tzinfo=timezone.utc)))


def test_unparseable_varchar_timestamp_still_raises_downstream_error(tmp_path):
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    db_path = tmp_path / 'bad_ts.duckdb'
    with duckdb.connect(str(db_path)) as conn:
        _create_profile_table(conn, crypto_profile())
        conn.register('df', pd.DataFrame({
            'symbol': ['BTC-USD'],
            'time': ['not-a-timestamp'],
            'open': [100.0], 'high': [101.0], 'low': [99.0], 'close': [100.0], 'volume': [10.0],
        }))
        conn.execute('CREATE TABLE bars AS SELECT * FROM df')
    cfg = _multi_instrument_config('duckdb_table', {'path': 'bad_ts.duckdb', 'table': 'bars'})
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed
    with pytest.raises(ContractValidationError):
        feed.read_execution_bars(['BTC-USD'], Frequency.parse('1h'),
                                 start_utc=pd.Timestamp('2026-01-03 00:00Z'),
                                 end_utc=pd.Timestamp('2026-01-03 03:00Z'))


@pytest.mark.parametrize('driver', ['duckdb_table', 'parquet_glob', 'csv_files'])
def test_declarative_bar_feed_load_pushdown_matches_baseline(tmp_path, driver):
    raw = _sample_multi_bars(include_extras=True, include_wide=True)
    src_opts = _write_driver_dataset(tmp_path, driver, raw, as_iso_string=(driver == 'csv_files'))
    cfg = _multi_instrument_config(driver, src_opts, frequency='1h', timestamp_anchor='bar_start',
                                   volume_mode='interval')
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed
    as_of = pd.Timestamp('2026-01-04 03:00Z')
    unpushed_raw = _dev_unpushed_read(feed.source)
    unpushed_raw = unpushed_raw.loc[unpushed_raw['symbol'].isin(['BTC-USD', 'ETH-USD'])]
    expected = feed._adapt_bars(unpushed_raw, as_of_utc=as_of)
    actual = feed.load(instruments=['BTC-USD', 'ETH-USD'], as_of_utc=as_of)
    assert 'wide_00' not in actual.columns
    assert set(actual.columns) >= {'money', 'factor', 'open_interest'}
    pd.testing.assert_frame_equal(actual, expected[actual.columns])


def test_csv_files_mismatched_columns_rejected_with_column_pruning(tmp_path):
    (tmp_path / 'one.csv').write_text('symbol,time,close,extra_a\nBTC-USD,2026-01-03T00:00:00Z,100.0,1\n',
                                      encoding='utf-8')
    (tmp_path / 'two.csv').write_text('symbol,time,close,extra_b\nBTC-USD,2026-01-03T01:00:00Z,101.0,2\n',
                                      encoding='utf-8')
    source = FrameSource(
        ProviderConfig('csv_files', {
            'paths': ['one.csv', 'two.csv'],
            'column_mapping': {'symbol': 'instrument_id', 'time': 'timestamp', 'close': 'close'},
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    with pytest.raises(ValueError, match='same columns'):
        source.read(ReadRequest(instruments=['BTC-USD'], columns=('instrument_id', 'close')))


@pytest.mark.parametrize('driver', ['duckdb_table', 'parquet_glob', 'csv_files'])
def test_non_ascii_and_space_source_columns_match_baseline(tmp_path, driver):
    base = _sample_multi_bars(include_extras=False, include_wide=False)
    renamed = base.rename(columns={
        'symbol': 'trade-symbol',
        'time': '收盘时间',
        'open': 'Open Price',
        'high': 'High Price',
        'low': 'Low Price',
        'close': 'Close Price',
        'volume': '成交量',
    })
    if driver == 'csv_files':
        out = renamed.copy()
        out['收盘时间'] = out['收盘时间'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        out.to_csv(tmp_path / 'cn_bars.csv', index=False)
        src_opts = {'paths': ['cn_bars.csv']}
    elif driver == 'parquet_glob':
        renamed.to_parquet(tmp_path / 'cn_bars.parquet', index=False)
        src_opts = {'path': 'cn_bars.parquet'}
    else:
        db_path = tmp_path / 'cn_bars.duckdb'
        with duckdb.connect(str(db_path)) as conn:
            _create_profile_table(conn, crypto_profile())
            conn.register('cn_df', renamed)
            conn.execute('CREATE TABLE bars AS SELECT * FROM cn_df')
        src_opts = {'path': 'cn_bars.duckdb', 'table': 'main.bars'}
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    mapping = {
        'trade-symbol': 'instrument_id',
        '收盘时间': 'timestamp',
        'Open Price': 'open',
        'High Price': 'high',
        'Low Price': 'low',
        'Close Price': 'close',
        '成交量': 'volume',
    }
    cfg = _multi_instrument_config(driver, src_opts, column_mapping=mapping)
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    # Unpushed read() and pushed read_execution_bars() both work and match dev baseline.
    pd.testing.assert_frame_equal(feed.source.read(), _dev_unpushed_read(feed.source), check_dtype=True)
    start = pd.Timestamp('2026-01-04 00:00Z')
    end = pd.Timestamp('2026-01-05 02:00Z')
    expected = _baseline_read_bar_feed(feed, ['BTC-USD', 'SOL-USD'], Frequency.parse('1h'),
                                       start_utc=start, end_utc=end)
    actual = feed.read_execution_bars(['BTC-USD', 'SOL-USD'], Frequency.parse('1h'),
                                      start_utc=start, end_utc=end)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True)


def test_aux_features_with_hyphen_and_spaces(tmp_path):
    shutil.copy(EXAMPLE / 'bars.csv', tmp_path)
    pd.DataFrame({
        'instrument_id': ['BTC-USD', 'ETH-USD'],
        'available_at_utc': ['2026-01-03T00:00:00Z', '2026-01-03T00:00:00Z'],
        'funding-rate': [0.0005, 0.0012],
        'basis spread': [1.5, 2.5],
    }).to_csv(tmp_path / 'funding.csv', index=False)
    slot = build_slot_from_config(BASE_CONFIG, base_dir=tmp_path)
    grid = slot.calendar.bar_grid('1h', '2026-01-03', '2026-01-03').iloc[:2]
    keyed = grid.assign(instrument_id='BTC-USD')
    aux = slot.read_aux_features(keyed, features=['funding-rate', 'basis spread'])
    assert aux['funding-rate'].tolist() == [0.0005, 0.0005]
    assert aux['basis spread'].tolist() == [1.5, 1.5]


def test_csv_multichunk_dtype_consistency_with_nulls_and_decimals_in_unselected_chunk(tmp_path):
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    # 1200 1min rows (> 2 * 512 chunks). Null open_interest and decimal factor/money appear ONLY at row 1100.
    rows = []
    start_ts = pd.Timestamp('2026-01-03 00:00Z')
    for idx in range(1200):
        ts = (start_ts + pd.Timedelta(minutes=idx)).strftime('%Y-%m-%dT%H:%M:%SZ')
        rows.append({
            'symbol': 'BTC-USD',
            'time': ts,
            'open': 100.0,
            'high': 102.0,
            'low': 99.0,
            'close': 101.0,
            'volume': 10.0,
            'open_interest': '' if idx == 1100 else 50,
            'factor': 1.5 if idx == 1100 else 1,
            'money': 1010.25 if idx == 1100 else 1010,
            'wide_unused': f'w_{idx}',
        })
    csv_path = tmp_path / 'bars.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    source = FrameSource(
        ProviderConfig('csv_files', {
            'paths': ['bars.csv'],
            'column_mapping': {'symbol': 'instrument_id', 'time': 'timestamp', 'close': 'close'},
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    full_raw = _dev_unpushed_read(source)
    win_start = start_ts + pd.Timedelta(minutes=10)
    win_end = start_ts + pd.Timedelta(minutes=20)
    mask = pd.to_datetime(full_raw['time'], utc=True).between(win_start, win_end)

    req = ReadRequest(
        instruments=['BTC-USD'],
        start_utc=win_start,
        end_utc=win_end,
        columns=('instrument_id', 'timestamp', 'close'),
        optional_columns=('open_interest', 'factor'),
    )
    expected_subset = full_raw.loc[mask, ['symbol', 'time', 'close', 'open_interest', 'factor']].reset_index(drop=True)
    # Cold cache miss and warm cache hit must both preserve file-level float64 dtypes.
    cold = source.read(req)
    warm = source.read(req)
    assert str(cold['open_interest'].dtype) == 'float64'
    assert str(cold['factor'].dtype) == 'float64'
    pd.testing.assert_frame_equal(cold, expected_subset, check_dtype=True)
    pd.testing.assert_frame_equal(warm, expected_subset, check_dtype=True)

    # Cache hit requesting an additional column ('money') must also inherit file-level float64 dtype.
    req_with_money = replace(req, optional_columns=('open_interest', 'factor', 'money'))
    expected_money = full_raw.loc[
        mask, ['symbol', 'time', 'close', 'open_interest', 'factor', 'money'],
    ].reset_index(drop=True)
    warm_money = source.read(req_with_money)
    assert str(warm_money['money'].dtype) == 'float64'
    pd.testing.assert_frame_equal(warm_money, expected_money, check_dtype=True)

    # Full DeclarativeBarFeed read_execution_bars must match baseline with exact dtypes.
    cfg = _multi_instrument_config(
        'csv_files', {'paths': ['bars.csv']}, frequency='1min',
        calendar=ProviderConfig('custom_sessions', {
            **BASE_CONFIG.calendar.options, 'start': '2026-01-03', 'end': '2026-01-04',
        }),
    )
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed
    expected_bars = _baseline_read_bar_feed(
        feed, ['BTC-USD'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    actual_bars = feed.read_execution_bars(
        ['BTC-USD'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    pd.testing.assert_frame_equal(actual_bars, expected_bars, check_dtype=True)


@pytest.mark.parametrize('driver', ['duckdb_table', 'parquet_glob', 'csv_files'])
def test_weekly_1w_grid_key_pushdown_matches_baseline(tmp_path, driver):
    cal = ProviderConfig('custom_sessions', {
        'calendar_id': 'crypto_weekly',
        'start': '2025-12-29',
        'end': '2026-01-25',
        'custom_sessions': dict(BASE_CONFIG.calendar.options['custom_sessions']),
    })
    rows = []
    # Three weekly bars at Monday 00:00Z bar_start (grid keys are end-of-week 2026-01-05, 01-12, 01-19).
    for idx, ts_str in enumerate(['2025-12-29 00:00Z', '2026-01-05 00:00Z', '2026-01-12 00:00Z']):
        rows.append({
            'symbol': 'BTC-USD',
            'time': pd.Timestamp(ts_str),
            'open': 100.0 + idx,
            'high': 105.0 + idx,
            'low': 99.0 + idx,
            'close': 103.0 + idx,
            'volume': 50.0 + idx,
        })
    profile = crypto_profile(
        calendar_id='crypto_weekly',
        calendar_coverage='2025-12-29..2026-01-25',
        frequencies=('1w',),
    )
    src_opts = _write_driver_dataset(
        tmp_path, driver, pd.DataFrame(rows), as_iso_string=(driver == 'csv_files'), profile=profile,
    )
    cfg = _multi_instrument_config(driver, src_opts, frequency='1w', timestamp_anchor='bar_start', calendar=cal)
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    grid = feed.calendar.bar_grid('1w')
    start = grid.timestamp_utc.iloc[1]
    end = grid.timestamp_utc.iloc[2]
    as_of = end - pd.Timedelta(hours=1)

    expected_exec = _baseline_read_bar_feed(feed, ['BTC-USD'], Frequency.parse('1w'), start_utc=start, end_utc=end)
    actual_exec = feed.read_execution_bars(['BTC-USD'], Frequency.parse('1w'), start_utc=start, end_utc=end)
    pd.testing.assert_frame_equal(actual_exec, expected_exec, check_dtype=True)
    assert len(actual_exec) == 2

    expected_sig = _baseline_read_bar_feed(
        feed, ['BTC-USD'], Frequency.parse('1w'), start_utc=start, end_utc=end, as_of_utc=as_of,
    )
    actual_sig = feed.read_signal_bars(
        ['BTC-USD'], Frequency.parse('1w'), start_utc=start, end_utc=end, as_of_utc=as_of,
    )
    pd.testing.assert_frame_equal(actual_sig, expected_sig, check_dtype=True)


@pytest.mark.parametrize('driver', ['duckdb_table', 'parquet_glob', 'csv_files'])
def test_cumulative_session_midday_window_pushdown_matches_baseline(tmp_path, driver):
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    # Asia/Shanghai session with lunch break: 09:30-11:30 and 13:00-15:00 local naive time.
    session = {
        'timezone': 'Asia/Shanghai',
        'weekdays': list(range(5)),
        'segments': [{'open': '09:30', 'close': '11:30'}, {'open': '13:00', 'close': '15:00'}],
    }
    cal = ProviderConfig('custom_sessions', {
        'calendar_id': 'cn_lunch', 'start': '2026-01-05', 'end': '2026-01-06', 'custom_sessions': session,
    })
    local_times = [
        '2026-01-05 09:30', '2026-01-05 10:00', '2026-01-05 10:30', '2026-01-05 11:00',
        '2026-01-05 13:00', '2026-01-05 13:30', '2026-01-05 14:00', '2026-01-05 14:30',
    ]
    raw = pd.DataFrame({
        'symbol': ['BTC-USD'] * len(local_times),
        'time': local_times if driver != 'parquet_glob' else pd.to_datetime(local_times),
        'open': [100.0 + i for i in range(len(local_times))],
        'high': [102.0 + i for i in range(len(local_times))],
        'low': [99.0 + i for i in range(len(local_times))],
        'close': [101.0 + i for i in range(len(local_times))],
        'volume': [10.0 * (i + 1) for i in range(len(local_times))],
    })
    profile = crypto_profile(
        calendar_id='cn_lunch',
        timezone='Asia/Shanghai',
        calendar_coverage='2026-01-05..2026-01-06',
        frequencies=('30min',),
    )
    src_opts = _write_driver_dataset(tmp_path, driver, raw, as_iso_string=False, profile=profile)
    cfg = _multi_instrument_config(
        driver, src_opts, frequency='30min', timestamp_anchor='bar_start',
        volume_mode='cumulative_session', source_timezone='Asia/Shanghai', calendar=cal,
    )
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    # Query strictly afternoon window (13:30..14:30 CST = 05:30..06:30Z); pushdown must include session open.
    start = pd.Timestamp('2026-01-05 05:30Z')
    end = pd.Timestamp('2026-01-05 06:30Z')
    as_of = pd.Timestamp('2026-01-05 06:30Z')
    expected_exec = _baseline_read_bar_feed(
        feed, ['BTC-USD'], Frequency.parse('30min'), start_utc=start, end_utc=end,
    )
    actual_exec = feed.read_execution_bars(
        ['BTC-USD'], Frequency.parse('30min'), start_utc=start, end_utc=end,
    )
    pd.testing.assert_frame_equal(actual_exec, expected_exec, check_dtype=True)
    assert actual_exec['volume'].tolist() == [10.0, 10.0, 10.0]

    expected_sig = _baseline_read_bar_feed(
        feed, ['BTC-USD'], Frequency.parse('30min'), start_utc=start, end_utc=end, as_of_utc=as_of,
    )
    actual_sig = feed.read_signal_bars(
        ['BTC-USD'], Frequency.parse('30min'), start_utc=start, end_utc=end, as_of_utc=as_of,
    )
    pd.testing.assert_frame_equal(actual_sig, expected_sig, check_dtype=True)


def test_csv_empty_shortcut_and_impossible_time_preserve_numeric_dtypes(tmp_path):
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    start_ts = pd.Timestamp('2026-01-03 00:00Z')
    rows = []
    for idx in range(1200):
        ts = (start_ts + pd.Timedelta(minutes=idx)).strftime('%Y-%m-%dT%H:%M:%SZ')
        rows.append({
            'time': ts,
            'open': 100.0,
            'high': 102.0,
            'low': 99.0,
            'close': 101.0,
            'volume': 10.0,
            'open_interest': '' if idx == 1100 else 50,
            'factor': 1.5 if idx == 1100 else 1,
        })
    csv_a = tmp_path / 'aapl_a.csv'
    csv_b = tmp_path / 'aapl_b.csv'
    pd.DataFrame(rows).to_csv(csv_a, index=False)
    pd.DataFrame(rows).to_csv(csv_b, index=False)

    mapping = {name: name for name in ('open', 'high', 'low', 'close', 'volume')}
    mapping['time'] = 'timestamp'
    cal = ProviderConfig('custom_sessions', {
        **BASE_CONFIG.calendar.options, 'start': '2026-01-03', 'end': '2026-01-04',
    })

    def make_feed(filename: str):
        cfg = replace(
            BASE_CONFIG,
            calendar=cal,
            bar_feed=ProviderConfig('csv_files', {
                'paths': [filename], 'frequency': '1min', 'timestamp_anchor': 'bar_start',
                'source_timezone': 'UTC', 'volume_mode': 'interval',
                'data_version': 'v1', 'instrument_id': 'AAPL.XNAS', 'column_mapping': mapping,
            }),
            universe=ProviderConfig('static', {
                'instruments': {'AAPL.XNAS': {'asset_class': 'equity', 'currency': 'USD'}},
                'universes': {'spot': ['AAPL.XNAS']},
            }),
        )
        return build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    win_start = start_ts + pd.Timedelta(minutes=10)
    win_end = start_ts + pd.Timedelta(minutes=20)

    # Cold-cache empty_shortcut (different instrument on static single-instrument feed).
    feed_a = make_feed('aapl_a.csv')
    expected_empty_inst = _baseline_read_bar_feed(
        feed_a, ['MSFT.XNAS'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    cold_empty_inst = feed_a.read_execution_bars(
        ['MSFT.XNAS'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    assert cold_empty_inst.empty
    assert str(cold_empty_inst['open_interest'].dtype) == 'float64'
    assert str(cold_empty_inst['factor'].dtype) == 'float64'
    pd.testing.assert_frame_equal(cold_empty_inst, expected_empty_inst, check_dtype=True)
    warm_empty_inst = feed_a.read_execution_bars(
        ['MSFT.XNAS'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    pd.testing.assert_frame_equal(warm_empty_inst, expected_empty_inst, check_dtype=True)

    # Cold-cache impossible_time (as_of_utc earlier than start_utc) on an independent uncached file.
    feed_b = make_feed('aapl_b.csv')
    early_as_of = start_ts - pd.Timedelta(hours=1)
    expected_impossible = _baseline_read_bar_feed(
        feed_b, ['AAPL.XNAS'], Frequency.parse('1min'),
        start_utc=win_start, end_utc=win_end, as_of_utc=early_as_of,
    )
    cold_impossible = feed_b.read_signal_bars(
        ['AAPL.XNAS'], Frequency.parse('1min'),
        start_utc=win_start, end_utc=win_end, as_of_utc=early_as_of,
    )
    assert cold_impossible.empty
    assert str(cold_impossible['open_interest'].dtype) == 'float64'
    assert str(cold_impossible['factor'].dtype) == 'float64'
    pd.testing.assert_frame_equal(cold_impossible, expected_impossible, check_dtype=True)
    warm_impossible = feed_b.read_signal_bars(
        ['AAPL.XNAS'], Frequency.parse('1min'),
        start_utc=win_start, end_utc=win_end, as_of_utc=early_as_of,
    )
    pd.testing.assert_frame_equal(warm_impossible, expected_impossible, check_dtype=True)

    # Warm cache with a newly requested optional column also resolves file-level dtype on empty_shortcut.
    src_c = FrameSource(
        ProviderConfig('csv_files', {
            'paths': ['aapl_a.csv'],
            'instrument_id': 'AAPL.XNAS',
            'column_mapping': mapping,
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    src_c.read(ReadRequest(instruments=['AAPL.XNAS'], start_utc=win_start, end_utc=win_end,
                           columns=('timestamp', 'close')))
    extra_empty = src_c.read(ReadRequest(
        instruments=['MSFT.XNAS'], start_utc=win_start, end_utc=win_end,
        columns=('timestamp', 'close'), optional_columns=('open_interest', 'factor'),
    ))
    assert extra_empty.empty
    assert str(extra_empty['open_interest'].dtype) == 'float64'
    assert str(extra_empty['factor'].dtype) == 'float64'


@pytest.mark.parametrize('mode', ['duckdb', 'parquet_arrow', 'parquet_pandas_nullable'])
def test_duckdb_and_parquet_nulls_outside_window_preserve_dtypes(tmp_path, mode):
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    start_ts = pd.Timestamp('2026-01-03 00:00Z')
    times = [start_ts + pd.Timedelta(minutes=i) for i in range(60)]
    oi_vals = [100 + i if i < 50 else None for i in range(60)]
    flag_vals = [(i % 2 == 0) if i < 50 else None for i in range(60)]

    if mode == 'duckdb':
        db_path = tmp_path / 'bars.duckdb'
        with duckdb.connect(str(db_path)) as conn:
            _create_profile_table(conn, crypto_profile(frequencies=('1min',)))
            conn.execute('''
                CREATE TABLE bars (
                    symbol VARCHAR, time TIMESTAMPTZ,
                    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
                    open_interest BIGINT, extra_flag BOOLEAN
                )
            ''')
            for ts, oi, flg in zip(times, oi_vals, flag_vals):
                conn.execute(
                    'INSERT INTO bars VALUES (?, ?, 100.0, 102.0, 99.0, 101.0, 10.0, ?, ?)',
                    ['BTC-USD', ts.to_pydatetime(), oi, flg],
                )
        src_opts = {'path': 'bars.duckdb', 'table': 'main.bars'}
        driver = 'duckdb_table'
    elif mode == 'parquet_arrow':
        tbl = pa.table({
            'symbol': pa.array(['BTC-USD'] * 60, type=pa.string()),
            'time': pa.array([ts.to_pydatetime() for ts in times], type=pa.timestamp('us', tz='UTC')),
            'open': pa.array([100.0] * 60, type=pa.float64()),
            'high': pa.array([102.0] * 60, type=pa.float64()),
            'low': pa.array([99.0] * 60, type=pa.float64()),
            'close': pa.array([101.0] * 60, type=pa.float64()),
            'volume': pa.array([10.0] * 60, type=pa.float64()),
            'open_interest': pa.array(oi_vals, type=pa.int64()),
            'extra_flag': pa.array(flag_vals, type=pa.bool_()),
        })
        pq.write_table(tbl, tmp_path / 'bars.parquet')
        src_opts = {'path': 'bars.parquet'}
        driver = 'parquet_glob'
    else:
        df = pd.DataFrame({
            'symbol': ['BTC-USD'] * 60,
            'time': times,
            'open': [100.0] * 60,
            'high': [102.0] * 60,
            'low': [99.0] * 60,
            'close': [101.0] * 60,
            'volume': [10.0] * 60,
            'open_interest': pd.Series(oi_vals, dtype='Int64'),
            'extra_flag': pd.Series(flag_vals, dtype='boolean'),
        })
        df.to_parquet(tmp_path / 'bars.parquet', index=False)
        src_opts = {'path': 'bars.parquet'}
        driver = 'parquet_glob'

    cal = ProviderConfig('custom_sessions', {
        **BASE_CONFIG.calendar.options, 'start': '2026-01-03', 'end': '2026-01-04',
    })
    cfg = _multi_instrument_config(driver, src_opts, frequency='1min', calendar=cal)
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed

    win_start = start_ts + pd.Timedelta(minutes=10)
    win_end = start_ts + pd.Timedelta(minutes=20)

    # BarFeed window read (NULLs exist only at rows 50..59 outside window).
    expected_bars = _baseline_read_bar_feed(
        feed, ['BTC-USD'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    actual_bars = feed.read_execution_bars(
        ['BTC-USD'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    pd.testing.assert_frame_equal(actual_bars, expected_bars, check_dtype=True)

    # BarFeed empty result for missing symbol also matches dev dtypes.
    expected_empty = _baseline_read_bar_feed(
        feed, ['ETH-USD'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    actual_empty = feed.read_execution_bars(
        ['ETH-USD'], Frequency.parse('1min'), start_utc=win_start, end_utc=win_end,
    )
    assert actual_empty.empty
    pd.testing.assert_frame_equal(actual_empty, expected_empty, check_dtype=True)

    # Direct FrameSource read including boolean column matches dev dtypes (window and empty).
    full_raw = _dev_unpushed_read(feed.source)
    req_cols = ('instrument_id', 'timestamp', 'close', 'open_interest', 'extra_flag')
    sel_cols = ['symbol', 'time', 'close', 'open_interest', 'extra_flag']
    mask = full_raw['time'].between(win_start, win_end)
    expected_src = full_raw.loc[mask, sel_cols].reset_index(drop=True)
    actual_src = feed.source.read(ReadRequest(
        instruments=['BTC-USD'], start_utc=win_start, end_utc=win_end, columns=req_cols,
    ))
    pd.testing.assert_frame_equal(actual_src, expected_src, check_dtype=True)

    actual_src_empty = feed.source.read(ReadRequest(
        instruments=['ETH-USD'], start_utc=win_start, end_utc=win_end, columns=req_cols,
    ))
    expected_src_empty = full_raw.iloc[:0][sel_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_src_empty, expected_src_empty, check_dtype=True)


def test_csv_whitespace_lines_fallback_reparse_and_multifile_combine(tmp_path):
    start_ts = pd.Timestamp('2026-01-03 00:00Z')
    lines = ['symbol,time,close,open_interest,wide_unused']
    for idx in range(700):
        ts = (start_ts + pd.Timedelta(minutes=idx)).strftime('%Y-%m-%dT%H:%M:%SZ')
        oi = '' if idx == 650 else '50'
        lines.append(f'BTC-USD,{ts},100.0,{oi},skip_{idx}')
        if idx in (10, 550):
            # Whitespace-only line counted as non-empty by byte scanner, skipped by pandas.read_csv.
            lines.append('   ')
    (tmp_path / 'ws.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # Second file has only integer open_interest for ETH-USD while ws.csv has float64 (due to row 650).
    (tmp_path / 'eth.csv').write_text(
        'symbol,time,close,open_interest,wide_unused\n'
        'ETH-USD,2026-01-03T00:15:00Z,200.0,75,skip\n',
        encoding='utf-8',
    )
    source = FrameSource(
        ProviderConfig('csv_files', {
            'paths': ['ws.csv', 'eth.csv'],
            'column_mapping': {'symbol': 'instrument_id', 'time': 'timestamp', 'close': 'close'},
        }),
        tmp_path,
        ('instrument_id', 'timestamp'),
    )
    full_raw = _dev_unpushed_read(source)
    win_start = start_ts + pd.Timedelta(minutes=10)
    win_end = start_ts + pd.Timedelta(minutes=20)

    req_btc = ReadRequest(
        instruments=['BTC-USD'], start_utc=win_start, end_utc=win_end,
        columns=('instrument_id', 'timestamp', 'close', 'open_interest'),
    )
    mask_btc = (full_raw['symbol'] == 'BTC-USD') & pd.to_datetime(full_raw['time'], utc=True).between(win_start, win_end)
    expected_btc = full_raw.loc[mask_btc, ['symbol', 'time', 'close', 'open_interest']].reset_index(drop=True)
    cold_btc = source.read(req_btc)
    warm_btc = source.read(req_btc)
    pd.testing.assert_frame_equal(cold_btc, expected_btc, check_dtype=True)
    pd.testing.assert_frame_equal(warm_btc, expected_btc, check_dtype=True)

    # Querying only ETH-USD (ws.csv yields empty float64 frame) must still combine to float64 like dev.
    req_eth = ReadRequest(
        instruments=['ETH-USD'], start_utc=win_start, end_utc=win_end,
        columns=('instrument_id', 'timestamp', 'close', 'open_interest'),
    )
    mask_eth = (full_raw['symbol'] == 'ETH-USD') & pd.to_datetime(full_raw['time'], utc=True).between(win_start, win_end)
    expected_eth = full_raw.loc[mask_eth, ['symbol', 'time', 'close', 'open_interest']].reset_index(drop=True)
    actual_eth = source.read(req_eth)
    assert str(actual_eth['open_interest'].dtype) == 'float64'
    pd.testing.assert_frame_equal(actual_eth, expected_eth, check_dtype=True)


@pytest.mark.parametrize('driver', ['duckdb_table', 'parquet_glob', 'csv_files'])
@pytest.mark.parametrize('volume_mode,anchor', [
    ('interval', 'bar_start'),
    ('interval', 'bar_end'),
    ('cumulative', 'bar_start'),
    ('cumulative', 'bar_end'),
])
def test_cme_rth_intraday_pushdown_matches_full_baseline(tmp_path, driver, volume_mode, anchor):
    from utils.session_calendar import SessionCalendar

    cal = SessionCalendar('CMEGlobex_Equity', '2026-11-25', '2026-11-27', rth_reference_calendar='NYSE')
    # Include overnight, regular RTH, Thanksgiving (RTH closed), and Black Friday (18:00Z RTH close).
    eth_1m = pd.DatetimeIndex(cal.bar_grid('1min', view='ETH').timestamp_utc)
    # Keep a representative contiguous slice per session so cumulative mode has contiguous ETH bars from open.
    selected = pd.DatetimeIndex(sorted({
        *pd.date_range('2026-11-24 23:00Z', '2026-11-25 15:00Z', freq='1min'),
        *pd.date_range('2026-11-25 23:00Z', '2026-11-26 02:00Z', freq='1min'),
        *pd.date_range('2026-11-26 23:00Z', '2026-11-27 18:14Z', freq='1min'),
    }))
    assert selected.isin(eth_1m).all()
    if volume_mode == 'interval':
        # Interval mode can also use a sparse window around RTH open/close.
        selected = selected[(selected >= pd.Timestamp('2026-11-25 14:25Z'))]
    rows = []
    for idx, ts in enumerate(selected):
        stamp = ts + pd.Timedelta(minutes=1) if anchor == 'bar_end' else ts
        rows.append({
            'symbol': 'NQZ26.CME',
            'time': stamp,
            'open': 20000.0 + idx,
            'high': 20002.0 + idx,
            'low': 19999.0 + idx,
            'close': 20001.0 + idx,
            'volume': float(idx + 1),
            'wide_unused': f'skip_{idx}',
        })
    raw = pd.DataFrame(rows)
    if volume_mode == 'cumulative':
        owners = [cal.session_at(ts, 'ETH').session_date for ts in selected]
        raw['volume'] = raw.groupby(owners)['volume'].cumsum()

    global_profile = market_profile(
        'global_futures',
        allowed_slot_providers=('global_futures', 'duckdb_table'),
        calendar_coverage='2026-11-01..2026-12-31',
    )
    src_opts = _write_driver_dataset(
        tmp_path, driver, raw, as_iso_string=(driver == 'csv_files'), profile=global_profile,
    )
    cfg = replace(
        _multi_instrument_config(
            driver, src_opts, frequency='1m', timestamp_anchor=anchor, volume_mode=volume_mode,
            calendar=ProviderConfig('pmc', {
                'calendar_id': 'CMEGlobex_Equity', 'rth_reference_calendar': 'NYSE',
                'start': '2026-11-25', 'end': '2026-11-27',
            }),
        ),
        market_domain='global_futures',
        bar_feed=ProviderConfig(driver, {
            **{k: v for k, v in BASE_CONFIG.bar_feed.options.items() if k != 'paths'},
            'frequency': '1m',
            'timestamp_anchor': anchor,
            'volume_mode': volume_mode,
            'source_timezone': 'UTC',
            'view': 'RTH',
            **src_opts,
        }),
        universe=ProviderConfig('static', {
            'instruments': {
                'NQZ26.CME': {
                    'asset_class': 'futures', 'currency': 'USD',
                    'contract_multiplier': 20.0, 'expiry_date': '2026-12-18',
                },
            },
            'universes': {'nq': ['NQZ26.CME']},
        }),
    )
    feed = build_slot_from_config(cfg, base_dir=tmp_path).bar_feed
    assert feed.view == 'RTH'

    start = pd.Timestamp('2026-11-25 14:30Z')
    end = pd.Timestamp('2026-11-27 18:10Z')
    as_of = pd.Timestamp('2026-11-27 18:00Z')

    expected_exec = _baseline_read_bar_feed(feed, ['NQZ26.CME'], Frequency.parse('1m'), start_utc=start, end_utc=end)
    actual_exec = feed.read_execution_bars(['NQZ26.CME'], Frequency.parse('1m'), start_utc=start, end_utc=end)
    pd.testing.assert_frame_equal(actual_exec, expected_exec)
    assert actual_exec.session_id.unique().tolist() == ['2026-11-25', '2026-11-27']
    assert actual_exec.timestamp_utc.max() == pd.Timestamp('2026-11-27 17:59Z')

    expected_sig = _baseline_read_bar_feed(
        feed, ['NQZ26.CME'], Frequency.parse('1m'), start_utc=start, end_utc=end, as_of_utc=as_of,
    )
    actual_sig = feed.read_signal_bars(
        ['NQZ26.CME'], Frequency.parse('1m'), start_utc=start, end_utc=end, as_of_utc=as_of,
    )
    pd.testing.assert_frame_equal(actual_sig, expected_sig)
