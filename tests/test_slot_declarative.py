"""Declarative sources, point-in-time components and the public import entry point."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import pytest

from market_db_samples import crypto_profile
from utils.market_db import init_market_db
from utils.market_db.storage import _create_profile_table

from slots import build_slot_from_config, register_builtin_slots
from slots.declarative.source import FrameSource, ReadRequest
from tools.agent_harness.cli import main
from utils.bars.source import BarSource
from utils.slots import ProviderConfig, SlotRegistry, load_slot_config
from utils.time_axis import Frequency


EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'declarative_crypto'
CONFIG = load_slot_config(EXAMPLE / 'slot.yaml')
MAPPING = dict(CONFIG.bar_feed.options['column_mapping'])
SYMBOL = 'BTC-USD'
START = pd.Timestamp('2026-01-03 00:00Z')
END = pd.Timestamp('2026-01-03 03:00Z')
FREQ = Frequency.parse('1h')


@pytest.mark.parametrize('driver', ['csv_files', 'parquet_glob', 'duckdb_table'])
def test_bar_drivers_normalize_session_cumulative_volume_and_cut_at_completion(tmp_path, driver):
    raw = pd.read_csv(EXAMPLE / 'bars.csv').assign(volume=[12, 26])
    raw = pd.concat([raw, raw.iloc[:1].assign(
        time='2026-01-04T01:00:00Z', volume=9)], ignore_index=True)
    if driver == 'csv_files':
        raw.iloc[:1].to_csv(tmp_path / 'a.csv', index=False)
        raw.iloc[1:].to_csv(tmp_path / 'b.csv', index=False)
        source = dict(paths=['b.csv', 'a.csv'])
    elif driver == 'parquet_glob':
        raw.iloc[:1].to_parquet(tmp_path / 'a.parquet', index=False)
        raw.iloc[1:].to_parquet(tmp_path / 'b.parquet', index=False)
        source = dict(path='*.parquet')
    else:
        shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
        with duckdb.connect(str(tmp_path / 'in.duckdb')) as connection:
            _create_profile_table(connection, crypto_profile())
            connection.from_df(raw.assign(time=pd.to_datetime(raw.time, utc=True))).create('bars')
        source = dict(path='in.duckdb', table='main.bars')
    options = dict(CONFIG.bar_feed.options)
    options.pop('paths')
    config = replace(CONFIG, bar_feed=ProviderConfig(driver, {**options, **source, 'volume_mode': 'cumulative'}))
    feed = build_slot_from_config(config, base_dir=tmp_path).bar_feed
    exec_bars = feed.read_execution_bars([SYMBOL], FREQ, start_utc=START, end_utc=END + pd.Timedelta(days=1))
    assert exec_bars.timestamp_utc.tolist() == [START, START + pd.Timedelta(hours=1), START + pd.Timedelta(days=1)]
    assert exec_bars.session_id.tolist() == ['2026-01-03', '2026-01-03', '2026-01-04']
    assert exec_bars.volume.tolist() == [12, 14, 9]
    assert exec_bars.close.tolist() == [103, 106, 103]
    assert feed.read_signal_bars([SYMBOL], FREQ, start_utc=START, end_utc=END, as_of_utc=START).empty
    assert feed.read_signal_bars([SYMBOL], FREQ, start_utc=START, end_utc=END,
                                 as_of_utc=START + pd.Timedelta(hours=1)).close.tolist() == [103]
    assert feed.read_execution_bars(['UNKNOWN'], FREQ, start_utc=START, end_utc=END).empty


def test_duckdb_timestamptz_matches_csv_in_declared_source_timezone(tmp_path):
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    label = '2026-01-02 20:00:00-05:00'
    raw = pd.read_csv(EXAMPLE / 'bars.csv').iloc[:1].assign(time=label)
    raw.to_csv(tmp_path / 'bars.csv', index=False)
    with duckdb.connect(str(tmp_path / 'source.duckdb')) as connection:
        _create_profile_table(connection, crypto_profile())
        connection.execute('CREATE TABLE aware_bars AS SELECT * REPLACE (time::TIMESTAMPTZ AS time) FROM raw')
        conflicting_text = raw.assign(time='2026-01-03 01:00:00+00:00')
        connection.execute('CREATE TABLE text_bars AS SELECT * FROM conflicting_text')
    options = {**CONFIG.bar_feed.options, 'source_timezone': 'America/New_York'}
    configs = (
        ProviderConfig('csv_files', options),
        ProviderConfig('duckdb_table', {**options, 'path': 'source.duckdb', 'table': 'aware_bars'}),
    )

    def read(source):
        slot = build_slot_from_config(replace(CONFIG, bar_feed=source), base_dir=tmp_path)
        return slot.bar_feed.read_execution_bars([SYMBOL], FREQ, start_utc=START, end_utc=END)

    expected, actual = map(read, configs)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.timestamp_utc.tolist() == [START]
    assert actual.close.tolist() == [103]
    text = ProviderConfig('duckdb_table', {**options, 'path': 'source.duckdb', 'table': 'text_bars'})
    with pytest.raises(ValueError, match='timestamp offset conflicts with source_timezone'):
        read(text)


@pytest.mark.parametrize('provider,name,days,hours', [
    ('pmc', 'NYSE', ['2026-01-05'], ['14:30']),
    ('pmc', 'XHKG', ['2026-01-05'], ['01:30']),
    ('custom_sessions', 'fx_weekdays', ['2026-01-05'], ['00:00']),
])
def test_explicit_market_calendars(provider, name, days, hours):
    options = dict(calendar_id=name, start='2026-01-03', end='2026-01-05')
    if provider == 'custom_sessions':
        options['custom_sessions'] = {**CONFIG.calendar.options['custom_sessions'], 'weekdays': list(range(5))}
    config = replace(CONFIG, calendar=ProviderConfig(provider, options))
    slot = build_slot_from_config(config)
    assert slot.calendar.sessions().strftime('%Y-%m-%d').tolist() == days
    grid = slot.calendar.bar_grid('30min')
    for day, hour in zip(days, hours):
        assert grid.loc[grid.session_id.eq(day), 'timestamp_utc'].min() == pd.Timestamp(f'{day} {hour}Z')


def test_daily_bar_end_is_scheduled_close_including_dst(tmp_path):
    raw = pd.read_csv(EXAMPLE / 'bars.csv')
    raw['time'] = ['2026-03-06 16:00', '2026-03-09 16:00']
    raw.to_csv(tmp_path / 'nyse.csv', index=False)
    config = replace(CONFIG, calendar=ProviderConfig('pmc', dict(
        calendar_id='NYSE', start='2026-03-06', end='2026-03-09')),
        bar_feed=ProviderConfig('csv_files', {**CONFIG.bar_feed.options, 'paths': ['nyse.csv'],
                                             'frequency': '1d', 'source_timezone': 'America/New_York'}))
    slot = build_slot_from_config(config, base_dir=tmp_path)
    start, end = pd.to_datetime(['2026-03-06 00:00Z', '2026-03-10 00:00Z'])
    expected = slot.calendar.bar_grid('1d')
    bars = slot.bar_feed.read_execution_bars([SYMBOL], Frequency.parse('1d'), start_utc=start, end_utc=end)
    assert bars.timestamp_utc.tolist() == pd.to_datetime(['2026-03-06 21:00Z', '2026-03-09 20:00Z']).tolist()
    pd.testing.assert_frame_equal(bars[expected.columns], expected)
    visible = slot.bar_feed.read_signal_bars([SYMBOL], Frequency.parse('1d'), start_utc=start,
                                             end_utc=end, as_of_utc=end - pd.Timedelta(hours=5))
    assert visible.session_id.tolist() == ['2026-03-06']


def test_universe_masks_and_funding_use_latest_visible_release(tmp_path):
    shutil.copy(EXAMPLE / 'bars.csv', tmp_path)
    shutil.copy(EXAMPLE / 'funding.csv', tmp_path)
    (tmp_path / 'membership.csv').write_text(
        'pool,symbol,seen\nspot,BTC-USD,2026-01-03T01:00:00Z\n'
        'spot,ETH-USD,2026-01-03T02:00:00Z\n', encoding='utf-8')
    (tmp_path / 'halts.csv').write_text(
        'instrument_id,available_at_utc,tradable\nBTC-USD,2026-01-03T02:00:00Z,false\n'
        'BTC-USD,2026-01-03T03:00:00Z,true\n', encoding='utf-8')
    option = CONFIG.universe.options['instruments'][SYMBOL]
    config = replace(CONFIG, universe=ProviderConfig('csv_files', dict(
        paths=['membership.csv'], column_mapping={'pool': 'universe_id', 'symbol': 'instrument_id',
                                                  'seen': 'available_at_utc'},
        instruments={SYMBOL: dict(option), 'ETH-USD': dict(option)})),
        microstructure=ProviderConfig('csv_files', dict(paths=['halts.csv'])))
    slot = build_slot_from_config(config, base_dir=tmp_path)
    assert slot.universe.instrument_spec(SYMBOL).currency == 'USD'
    assert slot.universe.resolve_universe('spot', as_of_utc=START) == ()
    assert slot.universe.resolve_universe('spot', as_of_utc=START + pd.Timedelta(hours=1)) == (SYMBOL,)
    assert slot.universe.resolve_universe('spot', as_of_utc=START + pd.Timedelta(hours=2)) == ('ETH-USD',)
    grid = slot.calendar.bar_grid('1h', '2026-01-03', '2026-01-03').iloc[:4]
    keyed = grid.assign(instrument_id=pd.Categorical([SYMBOL] * len(grid)))
    result = slot.read_aux_features(keyed, features=['funding_rate'])
    assert result.funding_rate.tolist() == [0.0001, 0.0003, 0.0003]
    assert (result.available_at_utc <= result.timestamp_utc).all()
    assert slot.microstructure.execution_profile == CONFIG.execution_profile
    assert slot.microstructure.tradability_mask(grid.iloc[::-1], instrument_id=SYMBOL).tolist() == [
        True, False, True, True]
    registry = SlotRegistry()
    register_builtin_slots(registry, [EXAMPLE / 'slot.yaml'])
    default = registry.get(CONFIG.slot_id)
    assert default.microstructure.tradability_mask(grid, instrument_id=SYMBOL).all()
    assert default.universe.resolve_universe('spot', as_of_utc=START) == (SYMBOL,)


def test_tradability_uses_declared_session_at_daily_close_and_shared_boundary():
    slot = build_slot_from_config(EXAMPLE / 'slot.yaml')
    daily = slot.calendar.bar_grid('1d')
    assert daily.iloc[-1].timestamp_utc == pd.Timestamp('2026-01-06 00:00Z')
    assert daily.session_id.tolist() == ['2026-01-03', '2026-01-04', '2026-01-05']
    opening = slot.calendar.bar_grid(FREQ, '2026-01-04', '2026-01-04').iloc[:1]
    assert opening.iloc[0].timestamp_utc == daily.iloc[0].timestamp_utc
    wrong_session = opening.assign(session_id='2026-01-05')
    grid = pd.concat([daily, opening, wrong_session], ignore_index=True)
    assert slot.microstructure.tradability_mask(grid, instrument_id=SYMBOL).tolist() == [
        True, True, True, True, False]


def import_manifest(tmp_path, suffix='json'):
    source = pd.read_csv(EXAMPLE / 'bars.csv').assign(symbol='BTC-USD.SPOT')
    source.to_csv(tmp_path / 'bars.csv', index=False)
    shutil.copyfile(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')
    document = CONFIG.as_dict()
    document['market_domain'] = 'custom:crypto'
    document['bar_feed']['options']['exchange'] = 'SPOT'
    universe = document['universe']['options']
    universe['instruments']['BTC-USD.SPOT'] = universe['instruments'].pop(SYMBOL)
    universe['universes']['spot'] = ['BTC-USD.SPOT']
    manifest = tmp_path / f'slot.{suffix}'
    return document, manifest


@pytest.mark.parametrize('suffix', ['yaml', 'yml', 'json'])
def test_import_manifest_preserves_calendar_and_audits_derived_bars(tmp_path, capsys, suffix):
    import yaml

    document, manifest = import_manifest(tmp_path, suffix)
    source = json.loads(json.dumps(document))
    manifest.write_text(json.dumps(source) if suffix == 'json' else yaml.safe_dump(source), encoding='utf-8')
    destination = tmp_path / 'import.duckdb'
    init_market_db(destination, crypto_profile())
    common = ['--db', str(destination), '--output-root', str(tmp_path / 'reports')]
    assert main(['import-bars', *common, '--manifest', str(manifest), '--derive', '2h']) == 0
    metrics = json.loads(capsys.readouterr().out)['metrics']
    assert metrics['imported_bars'] == 2 and metrics['audited_sessions'] == 2
    assert {(row['frequency'], row['status'], row['expected_bars'], row['actual_bars'])
            for row in metrics['quality']} == {('60min', 'partial', 24, 2), ('120min', 'partial', 12, 1)}
    with duckdb.connect(str(destination), read_only=True) as connection:
        with pytest.warns(UserWarning, match='incomplete_buckets'):
            bars = BarSource(connection).read(['BTC-USD.SPOT'], '120min', '2026-01-03', '2026-01-03', 'ETH')
        assert bars[['open', 'close', 'volume']].values.tolist() == [[100., 106., 26.]]
    assert main(['audit-bars', *common]) == 0
    assert json.loads(capsys.readouterr().out)['metrics']['audited_sessions'] == 2


def test_import_manifest_rolls_back_native_rows_if_derive_fails(tmp_path, capsys):
    document, manifest = import_manifest(tmp_path)
    manifest.write_text(json.dumps(document), encoding='utf-8')
    destination = tmp_path / 'invalid.duckdb'
    init_market_db(destination, crypto_profile())
    assert main(['import-bars', '--db', str(destination), '--manifest', str(manifest),
                 '--derive', '30min', '--output-root', str(tmp_path / 'reports')]) != 0
    assert 'coarser' in json.loads(capsys.readouterr().out)['metrics']['message']
    with duckdb.connect(str(destination), read_only=True) as connection:
        assert connection.execute('SHOW TABLES').fetchall() == [('market_db_profile',)]


def test_import_manifest_unknown_pmc_calendar_returns_precondition_json(tmp_path, capsys):
    document, manifest = import_manifest(tmp_path)
    document['calendar'] = {'provider': 'pmc', 'options': {
        'calendar_id': 'NYES', 'start': '2026-01-03', 'end': '2026-01-05',
    }}
    manifest.write_text(json.dumps(document), encoding='utf-8')
    destination = tmp_path / 'import.duckdb'
    init_market_db(destination, crypto_profile())
    assert main(['import-bars', '--db', str(destination), '--manifest', str(manifest),
                 '--output-root', str(tmp_path / 'reports')]) != 0
    result = json.loads(capsys.readouterr().out)
    assert not result['ok']
    assert result['summary']['reason'] == 'bar_storage_precondition_failed'
    assert 'NYES' in result['metrics']['message']
    with duckdb.connect(str(destination), read_only=True) as connection:
        assert connection.execute('SHOW TABLES').fetchall() == [('market_db_profile',)]


def test_import_manifest_can_read_a_source_table_in_its_destination(tmp_path, capsys):
    destination = tmp_path / 'source-and-destination.duckdb'
    document, manifest = import_manifest(tmp_path)
    source = pd.read_csv(tmp_path / 'bars.csv')
    with duckdb.connect(str(destination)) as connection:
        _create_profile_table(connection, crypto_profile())
        connection.from_df(source).create('landing_bars')
    source_config = document['bar_feed']
    source_config['provider'] = 'duckdb_table'
    source_config['options'].pop('paths')
    source_config['options'].update(path=destination.name, table='landing_bars')
    manifest = tmp_path / 'slot.json'
    manifest.write_text(json.dumps(document), encoding='utf-8')

    assert main(['import-bars', '--db', str(destination), '--manifest', str(manifest),
                 '--output-root', str(tmp_path / 'reports')]) == 0
    assert json.loads(capsys.readouterr().out)['metrics']['imported_bars'] == 2
    with duckdb.connect(str(destination), read_only=True) as connection:
        pd.testing.assert_frame_equal(connection.table('landing_bars').fetchdf(), source)


def test_invalid_calendar_and_missing_cumulative_session_prefix(tmp_path):
    config = replace(CONFIG, calendar=ProviderConfig('pmc', dict(start='2026-01-03', end='2026-01-05')))
    with pytest.raises(ValueError, match='calendar_id'):
        build_slot_from_config(config)
    pd.read_csv(EXAMPLE / 'bars.csv').iloc[1:].to_csv(tmp_path / 'bars.csv', index=False)
    config = replace(CONFIG, bar_feed=ProviderConfig(
        'csv_files', {**CONFIG.bar_feed.options, 'volume_mode': 'cumulative'}))
    slot = build_slot_from_config(config, base_dir=tmp_path)
    with pytest.raises(ValueError, match='missing bars'):
        slot.bar_feed.read_execution_bars([SYMBOL], FREQ, start_utc=START, end_utc=END)


@pytest.mark.parametrize('anchor,first,second', [
    ('bar_start', '00:00', '01:00'), ('bar_end', '01:00', '01:30'),
])
def test_partial_session_bucket_uses_scheduled_completion(tmp_path, anchor, first, second):
    bars = pd.read_csv(EXAMPLE / 'bars.csv').assign(
        time=[f'2026-01-03T{time}:00Z' for time in (first, second)])
    bars.to_csv(tmp_path / 'bars.csv', index=False)
    session = {'timezone': 'UTC', 'weekdays': list(range(7)),
               'segments': [{'open': '00:00', 'close': '01:30'}]}
    config = replace(CONFIG,
        calendar=ProviderConfig('custom_sessions', {**CONFIG.calendar.options, 'custom_sessions': session}),
        bar_feed=ProviderConfig('csv_files', {**CONFIG.bar_feed.options, 'timestamp_anchor': anchor}))
    feed = build_slot_from_config(config, base_dir=tmp_path).bar_feed
    close = pd.Timestamp('2026-01-03 01:30Z')
    before = feed.read_signal_bars([SYMBOL], FREQ, start_utc=START, end_utc=END,
                                   as_of_utc=close - pd.Timedelta(nanoseconds=1))
    assert before.close.tolist() == [103]
    actual = feed.read_signal_bars([SYMBOL], FREQ, start_utc=START, end_utc=END, as_of_utc=close)
    assert actual.timestamp_utc.tolist() == [START, START + pd.Timedelta(hours=1)]
    assert actual.close.tolist() == [103, 106]


@pytest.mark.parametrize('unit,delta,sub_delta,gap_low,gap_high', [
    ('s', pd.Timedelta(seconds=1), pd.Timedelta(milliseconds=500),
     pd.Timedelta(milliseconds=200), pd.Timedelta(milliseconds=500)),
    ('ms', pd.Timedelta(milliseconds=1), pd.Timedelta(microseconds=500),
     pd.Timedelta(microseconds=200), pd.Timedelta(microseconds=500)),
    ('us', pd.Timedelta(microseconds=1), pd.Timedelta(nanoseconds=500),
     pd.Timedelta(nanoseconds=200), pd.Timedelta(nanoseconds=500)),
    ('ns', pd.Timedelta(nanoseconds=2), pd.Timedelta(nanoseconds=1),
     pd.Timedelta(nanoseconds=1), pd.Timedelta(nanoseconds=1)),
])
@pytest.mark.parametrize('aware_tz', [True, False])
def test_parquet_asof_pushdown_precision_bounds(
    tmp_path, monkeypatch, unit, delta, sub_delta, gap_low, gap_high, aware_tz,
):
    base_utc = pd.Timestamp('2026-01-03 01:30:00Z')
    t0 = base_utc
    t1 = base_utc + delta
    source_tz = None if aware_tz else 'America/New_York'
    if aware_tz:
        ts_values = [t0, t1]
        pa_type = pa.timestamp(unit, tz='UTC')
    else:
        ts_values = [
            t0.tz_convert(source_tz).tz_localize(None),
            t1.tz_convert(source_tz).tz_localize(None),
        ]
        pa_type = pa.timestamp(unit)

    arrow_vals = [
        ts.as_unit('ns').value if unit == 'ns' else ts.to_pydatetime()
        for ts in ts_values
    ]
    table = pa.table({
        'sym': pa.array([SYMBOL, SYMBOL], type=pa.string()),
        'ts': pa.array(arrow_vals, type=pa_type),
        'feat': pa.array([1.0, 999.0], type=pa.float64()),
    })

    pq_path = tmp_path / 'events.parquet'
    pq.write_table(table, pq_path)

    # PyArrow Parquet writer coerces timestamp('s') to 'ms' on disk; for unit='s', also wrap
    # ParquetDataset with an in-memory 's' table so the 's' branch in _read_parquet is exercised.
    pushed_reads: list[pd.DataFrame] = []
    real_dataset = pq.ParquetDataset

    class SpyDataset:
        def __init__(self, path, filters=None, **kwargs):
            self._filters = filters
            self._inner = real_dataset(path, filters=filters, **kwargs) if unit != 's' else None

        @property
        def schema(self):
            return table.schema if unit == 's' else self._inner.schema

        def read(self, columns=None, **kwargs):
            if unit == 's':
                out = pds.dataset(table).to_table(filter=self._filters, columns=columns)
            else:
                out = self._inner.read(columns=columns, **kwargs)
            pushed_reads.append(out.to_pandas())
            return out

    monkeypatch.setattr(pq, 'ParquetDataset', SpyDataset)

    options = {
        'path': 'events.parquet',
        'column_mapping': {'sym': 'instrument_id', 'ts': 'available_at_utc'},
    }
    if source_tz is not None:
        options['source_timezone'] = source_tz
    source = FrameSource(
        ProviderConfig('parquet_glob', options),
        tmp_path,
        ('instrument_id', 'available_at_utc'),
    )

    # 1. Sub-unit as_of_utc between t0 and t1 must return only t0 (never leak t1).
    as_of_mid = base_utc + sub_delta
    res_asof = source.read(ReadRequest(instruments=[SYMBOL], as_of_utc=as_of_mid))
    assert res_asof['feat'].tolist() == [1.0]
    if aware_tz:
        assert pushed_reads[-1]['feat'].tolist() == [1.0]

    # 2. read_mapped with sub-unit lower_utc and upper_utc returns only t1.
    mapped_upper = source.read_mapped(ReadRequest(
        instruments=[SYMBOL],
        start_utc=base_utc + sub_delta,
        end_utc=t1,
    ))
    assert mapped_upper['feat'].tolist() == [999.0]
    assert list(mapped_upper.columns) == ['instrument_id', 'available_at_utc', 'feat']
    if aware_tz:
        assert pushed_reads[-1]['feat'].tolist() == [999.0]

    # 3. Exact unit-boundary hits include both t0 and t1.
    exact_both = source.read(ReadRequest(
        instruments=[SYMBOL], start_utc=t0, end_utc=t1, as_of_utc=t1,
    ))
    assert exact_both['feat'].tolist() == [1.0, 999.0]

    # 4. lower_utc and upper_utc inside the same unit gap return an empty frame.
    if unit != 'ns':
        empty_gap = source.read_mapped(ReadRequest(
            instruments=[SYMBOL],
            start_utc=base_utc + gap_low,
            end_utc=base_utc + gap_high,
            columns=('instrument_id', 'feat'),
        ))
        assert empty_gap.empty
        assert list(empty_gap.columns) == ['instrument_id', 'feat']
        if aware_tz:
            assert pushed_reads[-1].empty

    # 5. Unselected timestamp column (need_temp_ts pruning) still filters accurately.
    pruned = source.read(ReadRequest(
        instruments=[SYMBOL],
        as_of_utc=as_of_mid,
        columns=('instrument_id', 'feat'),
    ))
    assert list(pruned.columns) == ['sym', 'feat']
    assert pruned['feat'].tolist() == [1.0]


@pytest.mark.parametrize('sql_type,unit,delta,sub_delta,gap_low,gap_high', [
    ('TIMESTAMPTZ', 'us', pd.Timedelta(microseconds=1), pd.Timedelta(nanoseconds=500),
     pd.Timedelta(nanoseconds=200), pd.Timedelta(nanoseconds=500)),
    ('TIMESTAMP_S', 's', pd.Timedelta(seconds=1), pd.Timedelta(milliseconds=500),
     pd.Timedelta(milliseconds=200), pd.Timedelta(milliseconds=500)),
    ('TIMESTAMP_MS', 'ms', pd.Timedelta(milliseconds=1), pd.Timedelta(microseconds=500),
     pd.Timedelta(microseconds=200), pd.Timedelta(microseconds=500)),
    ('TIMESTAMP', 'us', pd.Timedelta(microseconds=1), pd.Timedelta(nanoseconds=500),
     pd.Timedelta(nanoseconds=200), pd.Timedelta(nanoseconds=500)),
    ('TIMESTAMP_NS', 'ns', pd.Timedelta(nanoseconds=2), pd.Timedelta(nanoseconds=1),
     pd.Timedelta(nanoseconds=1), pd.Timedelta(nanoseconds=1)),
])
def test_duckdb_asof_pushdown_precision_bounds(
    tmp_path, monkeypatch, sql_type, unit, delta, sub_delta, gap_low, gap_high,
):
    base_utc = pd.Timestamp('2026-01-03 01:30:00Z')
    t0 = base_utc
    t1 = base_utc + delta
    is_aware = sql_type == 'TIMESTAMPTZ'
    source_tz = None if is_aware else 'America/New_York'
    if is_aware:
        ts_lit = [t0.strftime('%Y-%m-%d %H:%M:%S.%f+00'), t1.strftime('%Y-%m-%d %H:%M:%S.%f+00')]
    else:
        ts_lit = [
            t0.tz_convert(source_tz).tz_localize(None).strftime('%Y-%m-%d %H:%M:%S.%f'),
            t1.tz_convert(source_tz).tz_localize(None).strftime('%Y-%m-%d %H:%M:%S.%f'),
        ]
        if unit == 'ns':
            ts_lit[1] = f'{ts_lit[1]}002'

    db_path = tmp_path / 'events.duckdb'
    with duckdb.connect(str(db_path)) as conn:
        _create_profile_table(conn, crypto_profile())
        conn.execute(f'CREATE TABLE events (sym VARCHAR, ts {sql_type}, feat DOUBLE)')
        conn.execute(f"INSERT INTO events VALUES ('{SYMBOL}', '{ts_lit[0]}'::{sql_type}, 1.0)")
        conn.execute(f"INSERT INTO events VALUES ('{SYMBOL}', '{ts_lit[1]}'::{sql_type}, 999.0)")

    executed: list[tuple[str, list]] = []
    real_connect = duckdb.connect

    class SpyConn:
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

    monkeypatch.setattr('utils.market_db.storage.duckdb.connect', lambda *a, **kw: SpyConn(real_connect(*a, **kw)))

    options = {
        'path': 'events.duckdb',
        'table': 'main.events',
        'column_mapping': {'sym': 'instrument_id', 'ts': 'available_at_utc'},
    }
    if source_tz is not None:
        options['source_timezone'] = source_tz
    source = FrameSource(
        ProviderConfig('duckdb_table', options),
        tmp_path,
        ('instrument_id', 'available_at_utc'),
    )

    as_of_mid = base_utc + sub_delta
    res_asof = source.read(ReadRequest(
        instruments=[SYMBOL], as_of_utc=as_of_mid, columns=('instrument_id', 'feat'),
    ))
    assert list(res_asof.columns) == ['sym', 'feat']
    assert res_asof['feat'].tolist() == [1.0]
    if is_aware:
        select_calls = [p for q, p in executed if '"main"."events"' in q and 'WHERE' in q]
        assert select_calls[-1][-1] == t0.to_pydatetime()

    res_upper = source.read_mapped(ReadRequest(
        instruments=[SYMBOL], start_utc=as_of_mid, end_utc=t1,
    ))
    assert res_upper['feat'].tolist() == [999.0]

    res_both = source.read(ReadRequest(
        instruments=[SYMBOL], start_utc=t0, end_utc=t1, as_of_utc=t1,
    ))
    assert res_both['feat'].tolist() == [1.0, 999.0]

    if unit != 'ns':
        res_gap = source.read(ReadRequest(
            instruments=[SYMBOL],
            start_utc=base_utc + gap_low,
            end_utc=base_utc + gap_high,
            columns=('instrument_id', 'feat'),
        ))
        assert res_gap.empty
        assert list(res_gap.columns) == ['sym', 'feat']


def test_p206_parquet_ms_asof_cutoff_and_causal_event_validation(tmp_path, monkeypatch):
    t0 = datetime(2026, 1, 3, 1, 30, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 3, 1, 30, 0, 1000, tzinfo=timezone.utc)
    as_of = pd.Timestamp('2026-01-03 01:30:00.000500Z')

    # Counterexample 1: direct FrameSource.read() on timestamp[ms, tz=UTC] must not leak t1.
    bars_tbl = pa.table({
        'instrument_id': pa.array([SYMBOL, SYMBOL], type=pa.string()),
        'available_at_utc': pa.array([t0, t1], type=pa.timestamp('ms', tz='UTC')),
        'feat': pa.array([1.0, 999.0], type=pa.float64()),
    })
    pq.write_table(bars_tbl, tmp_path / 'ce1.parquet')

    pushed_counts: list[int] = []
    real_dataset = pq.ParquetDataset

    class CountingDataset:
        def __init__(self, path, filters=None, **kwargs):
            self._inner = real_dataset(path, filters=filters, **kwargs)

        @property
        def schema(self):
            return self._inner.schema

        def read(self, columns=None, **kwargs):
            out = self._inner.read(columns=columns, **kwargs)
            pushed_counts.append(out.num_rows)
            return out

    monkeypatch.setattr(pq, 'ParquetDataset', CountingDataset)

    ce1_source = FrameSource(
        ProviderConfig('parquet_glob', {'path': 'ce1.parquet'}),
        tmp_path,
        ('instrument_id', 'available_at_utc'),
    )
    ce1 = ce1_source.read(ReadRequest(instruments=[SYMBOL], as_of_utc=as_of))
    assert ce1['feat'].tolist() == [1.0]
    assert pushed_counts == [1]

    # Counterexample 2: duplicate event keys at future t1=01:30:00.001 must not trigger
    # ValueError: Duplicate event keys in DeclarativeAuxFeatures or DeclarativeMicrostructure.
    aux_tbl = pa.table({
        'instrument_id': pa.array([SYMBOL, SYMBOL, SYMBOL], type=pa.string()),
        'available_at_utc': pa.array([t0, t1, t1], type=pa.timestamp('ms', tz='UTC')),
        'feat': pa.array([1.0, 999.0, 888.0], type=pa.float64()),
    })
    pq.write_table(aux_tbl, tmp_path / 'aux.parquet')

    micro_tbl = pa.table({
        'instrument_id': pa.array([SYMBOL, SYMBOL, SYMBOL], type=pa.string()),
        'available_at_utc': pa.array([t0, t1, t1], type=pa.timestamp('ms', tz='UTC')),
        'tradable': pa.array([True, False, True], type=pa.bool_()),
    })
    pq.write_table(micro_tbl, tmp_path / 'micro.parquet')

    bars_df = pd.read_csv(EXAMPLE / 'bars.csv')
    bars_df.to_csv(tmp_path / 'bars.csv', index=False)
    config = replace(
        CONFIG,
        aux_features=ProviderConfig('parquet_glob', {'path': 'aux.parquet'}),
        microstructure=ProviderConfig('parquet_glob', {'path': 'micro.parquet'}),
    )
    slot = build_slot_from_config(config, base_dir=tmp_path)
    grid = pd.DataFrame({
        'instrument_id': [SYMBOL],
        'timestamp_utc': [as_of],
        'session_id': ['2026-01-03'],
    })

    aux_out = slot.read_aux_features(grid, features=['feat'])
    assert aux_out['feat'].tolist() == [1.0]

    mask_out = slot.microstructure.tradability_mask(grid, instrument_id=SYMBOL)
    assert mask_out.tolist() == [True]


def _nq_declarative_config(
    tmp_path: Path,
    *,
    frequency: str = '1m',
    view: str | None = 'ETH',
    include_view: bool = True,
    timestamp_anchor: str = 'bar_start',
    volume_mode: str = 'interval',
    start: str = '2026-11-25',
    end: str = '2026-11-30',
    rth_reference_calendar: str | None = 'NYSE',
    include_rth_ref: bool = True,
    csv_name: str = 'nq.csv',
    driver: str = 'csv_files',
    driver_options: dict | None = None,
):
    from utils.slots import ExecutionProfile

    cal_opts: dict = {'calendar_id': 'CMEGlobex_Equity', 'start': start, 'end': end}
    if include_rth_ref and rth_reference_calendar is not None:
        cal_opts['rth_reference_calendar'] = rth_reference_calendar
    bar_opts: dict = {
        'frequency': frequency,
        'timestamp_anchor': timestamp_anchor,
        'source_timezone': 'UTC',
        'volume_mode': volume_mode,
        'source': 'synthetic',
        'data_version': 'v1',
        'column_mapping': {
            'timestamp': 'timestamp',
            'symbol': 'instrument_id',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        },
        **(driver_options or {'paths': [csv_name]}),
    }
    if include_view:
        bar_opts['view'] = view
    return replace(
        CONFIG,
        slot_id='nq_declarative_test',
        market_domain='global_futures',
        calendar=ProviderConfig('pmc', cal_opts),
        bar_feed=ProviderConfig(driver, bar_opts),
        universe=ProviderConfig('static', {
            'instruments': {
                'NQZ26.CME': {
                    'asset_class': 'futures',
                    'currency': 'USD',
                    'contract_multiplier': 20.0,
                    'expiry_date': '2026-12-18',
                },
            },
            'universes': {'nq': ['NQZ26.CME']},
        }),
        aux_features=ProviderConfig('none'),
        microstructure=ProviderConfig('open'),
        execution_profile=ExecutionProfile(
            entry_price_field='open',
            exit_price_field='close',
            min_entry_offset_bars=1,
            exit_adjustment=0,
            return_unit='decimal',
            gap_rule='endpoints_only',
            boundary_policy='same_session',
            allow_short=True,
        ),
    )


def _write_nq_csv(path: Path, timestamps: pd.DatetimeIndex, *, cumulative_per_eth_session: bool = False,
                  calendar=None) -> pd.DataFrame:
    rows = []
    for idx, ts in enumerate(timestamps):
        px = 20000.0 + idx * 0.25
        rows.append({
            'timestamp': ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'symbol': 'NQZ26.CME',
            'open': px,
            'high': px + 1.0,
            'low': px - 1.0,
            'close': px + 0.5,
            'volume': float(idx + 1),
        })
    df = pd.DataFrame(rows)
    if cumulative_per_eth_session:
        assert calendar is not None
        df = df.copy()
        owners = [calendar.session_at(pd.Timestamp(ts), 'ETH').session_date for ts in timestamps]
        df['volume'] = df.groupby(owners)['volume'].cumsum()
    df.to_csv(path, index=False)
    return df


def test_declarative_view_defaults_backward_compat_and_fail_closed(tmp_path):
    from utils.market_db import MarketDbProfileError
    from utils.market_db.errors import MARKET_DB_PROFILE_INVALID
    from market_db_samples import market_profile

    shutil.copy(EXAMPLE / 'bars.csv', tmp_path / 'bars.csv')
    shutil.copy(EXAMPLE / 'funding.csv', tmp_path / 'funding.csv')

    default_slot = build_slot_from_config(CONFIG, base_dir=tmp_path)
    assert default_slot.bar_feed.view == 'ETH'
    assert default_slot.microstructure.view == 'ETH'
    assert default_slot.slot_config_hash == CONFIG.slot_config_hash

    for bad_view in (None, 1, True, 'rth', 'eth', 'unknown', ''):
        bad_cfg = replace(
            CONFIG,
            bar_feed=ProviderConfig(CONFIG.bar_feed.provider, {**CONFIG.bar_feed.options, 'view': bad_view}),
        )
        with pytest.raises(ValueError, match='view must be ETH or RTH'):
            build_slot_from_config(bad_cfg, base_dir=tmp_path)

    # Profile session_views=['ETH'] rejects view='RTH' with MarketDbProfileError.
    us_profile = market_profile(
        'us_equity',
        instrument_id_pattern=r'^AAPL\.XNAS$',
        frequencies=('1d', '1h'),
        session_views=('ETH',),
    )
    db_path = tmp_path / 'us.duckdb'
    init_market_db(db_path, us_profile)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE bars AS SELECT 'AAPL.XNAS' AS symbol, TIMESTAMP '2026-01-05 09:30:00' AS time, "
            "10.0 AS open, 11.0 AS high, 9.0 AS low, 10.5 AS close, 100.0 AS volume"
        )
    db_cfg = replace(
        CONFIG,
        market_domain='us_equity',
        calendar=ProviderConfig('pmc', {'calendar_id': 'XNYS', 'start': '2026-01-05', 'end': '2026-01-05'}),
        bar_feed=ProviderConfig('duckdb_table', {
            'path': 'us.duckdb',
            'table': 'main.bars',
            'frequency': '1h',
            'timestamp_anchor': 'bar_start',
            'source_timezone': 'America/New_York',
            'volume_mode': 'interval',
            'data_version': 'v1',
            'view': 'RTH',
            'column_mapping': {
                'symbol': 'instrument_id', 'time': 'timestamp',
                'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume',
            },
        }),
        universe=ProviderConfig('static', {
            'instruments': {'AAPL.XNAS': {'asset_class': 'equity', 'currency': 'USD'}},
            'universes': {'us': ['AAPL.XNAS']},
        }),
        aux_features=ProviderConfig('none'),
        microstructure=ProviderConfig('open'),
    )
    with pytest.raises(MarketDbProfileError, match=MARKET_DB_PROFILE_INVALID):
        build_slot_from_config(db_cfg, base_dir=tmp_path)


def test_declarative_cme_globex_nyse_eth_rth_projection_holiday_and_half_day(tmp_path):
    from backtest_and_analysis.evaluation.labels import UnifiedReturnLabeler
    from utils.session_calendar import SessionCalendar

    # Cover Wed normal (2026-11-25), Thu Thanksgiving (2026-11-26, RTH closed),
    # Fri Black Friday (2026-11-27, RTH 13:00 ET / 18:00Z early close), and Mon (2026-11-30) for session_boundary.
    cal = SessionCalendar('CMEGlobex_Equity', '2026-11-25', '2026-11-30', rth_reference_calendar='NYSE')
    eth_grid = cal.bar_grid('1min', view='ETH')
    _write_nq_csv(tmp_path / 'nq_eth.csv', pd.DatetimeIndex(eth_grid.timestamp_utc))

    # Also test omitting rth_reference_calendar defaults CMEGlobex_Equity to NYSE.
    eth_cfg = _nq_declarative_config(
        tmp_path, view='ETH', start='2026-11-25', end='2026-11-30',
        include_rth_ref=False, csv_name='nq_eth.csv',
    )
    rth_cfg = _nq_declarative_config(
        tmp_path, view='RTH', start='2026-11-25', end='2026-11-30',
        rth_reference_calendar='NYSE', csv_name='nq_eth.csv',
    )
    eth_slot = build_slot_from_config(eth_cfg, base_dir=tmp_path)
    rth_slot = build_slot_from_config(rth_cfg, base_dir=tmp_path)
    assert eth_slot.calendar.rth_reference_calendar == 'NYSE'
    assert eth_slot.bar_feed.view == 'ETH' and eth_slot.microstructure.view == 'ETH'
    assert rth_slot.bar_feed.view == 'RTH' and rth_slot.microstructure.view == 'RTH'

    lo = pd.Timestamp('2026-11-24 23:00Z')
    hi = pd.Timestamp('2026-11-30 22:00Z')
    eth_bars = eth_slot.bar_feed.read_execution_bars(['NQZ26.CME'], Frequency.parse('1m'), start_utc=lo, end_utc=hi)
    rth_bars = rth_slot.bar_feed.read_execution_bars(['NQZ26.CME'], Frequency.parse('1m'), start_utc=lo, end_utc=hi)
    rth_grid = cal.bar_grid('1min', view='RTH')
    assert len(eth_bars) == len(eth_grid)
    assert rth_bars.timestamp_utc.tolist() == rth_grid.timestamp_utc.tolist()
    assert rth_bars.session_id.tolist() == rth_grid.session_id.tolist()
    assert '2026-11-26' not in set(rth_bars.session_id)
    assert rth_bars.loc[rth_bars.session_id.eq('2026-11-27'), 'timestamp_utc'].max() == pd.Timestamp('2026-11-27 17:59Z')

    # load() and read_signal_bars() match the RTH projection without lookahead.
    loaded_rth = rth_slot.bar_feed.load(instruments=['NQZ26.CME'])
    assert loaded_rth.timestamp_utc.tolist() == rth_grid.timestamp_utc.tolist()
    sig_at_half_close = rth_slot.bar_feed.read_signal_bars(
        ['NQZ26.CME'], Frequency.parse('1m'),
        start_utc=pd.Timestamp('2026-11-27 17:55Z'),
        end_utc=pd.Timestamp('2026-11-27 18:15Z'),
        as_of_utc=pd.Timestamp('2026-11-27 18:00Z'),
    )
    assert sig_at_half_close.timestamp_utc.tolist() == list(
        pd.date_range('2026-11-27 17:55Z', '2026-11-27 17:59Z', freq='1min')
    )

    # DeclarativeMicrostructure.tradability_mask handles overnight, half-day post-13:00 ET, and Thanksgiving empty RTH.
    probe_grid = pd.DataFrame({
        'instrument_id': ['NQZ26.CME'] * 5,
        'timestamp_utc': pd.to_datetime([
            '2026-11-25 14:30Z',  # Regular RTH open
            '2026-11-25 04:00Z',  # Overnight ETH only
            '2026-11-26 15:00Z',  # Thanksgiving: ETH open, RTH empty (no IndexError!)
            '2026-11-27 17:59Z',  # Black Friday last RTH minute
            '2026-11-27 18:05Z',  # Black Friday post-13:00 ET (ETH only)
        ]),
        'session_id': ['2026-11-25', '2026-11-25', '2026-11-26', '2026-11-27', '2026-11-27'],
        'execution_price_field': ['open'] * 5,
    })
    assert eth_slot.microstructure.tradability_mask(probe_grid, instrument_id='NQZ26.CME').tolist() == [
        True, True, True, True, True,
    ]
    assert rth_slot.microstructure.tradability_mask(probe_grid, instrument_id='NQZ26.CME').tolist() == [
        True, False, False, True, False,
    ]

    # same_session boundary checks on Black Friday 13:00 ET early close: RTH stops at 17:59Z, ETH continues.
    rth_labels = UnifiedReturnLabeler.from_slot(rth_slot, apply_tradability_mask=True).label(
        ['NQZ26.CME'], '1m', start_date='2026-11-27', end_date='2026-11-30',
        horizons=[1], entry_offset_bars=1,
    )
    eth_labels = UnifiedReturnLabeler.from_slot(eth_slot, apply_tradability_mask=True).label(
        ['NQZ26.CME'], '1m', start_date='2026-11-27', end_date='2026-11-30',
        horizons=[1], entry_offset_bars=1,
    )
    rth_by_sig = rth_labels.set_index('signal_timestamp')
    eth_by_sig = eth_labels.set_index('signal_timestamp')
    assert rth_by_sig.loc[pd.Timestamp('2026-11-27 17:57Z'), 'status'] == 'valid'
    assert rth_by_sig.loc[pd.Timestamp('2026-11-27 17:58Z'), 'exclusion_reason'] == 'session_boundary'
    assert rth_by_sig.loc[pd.Timestamp('2026-11-27 17:59Z'), 'exclusion_reason'] == 'session_boundary'
    assert eth_by_sig.loc[pd.Timestamp('2026-11-27 17:58Z'), 'status'] == 'valid'
    assert eth_by_sig.loc[pd.Timestamp('2026-11-27 18:13Z'), 'exclusion_reason'] == 'session_boundary'


@pytest.mark.parametrize('volume_mode', ['interval', 'cumulative'])
def test_declarative_rth_pure_60m_and_cumulative_eth_projection(tmp_path, volume_mode):
    from utils.session_calendar import SessionCalendar

    cal = SessionCalendar('CMEGlobex_Equity', '2026-11-25', '2026-11-25', rth_reference_calendar='NYSE')
    # 1. Pure RTH 60m source (starts at 09:30 ET = 14:30Z) works for both interval and cumulative volume.
    rth_60m = cal.bar_grid('60m', view='RTH').timestamp_utc
    raw_60m = _write_nq_csv(tmp_path / 'rth_60m.csv', pd.DatetimeIndex(rth_60m))
    if volume_mode == 'cumulative':
        raw_60m['volume'] = raw_60m['volume'].cumsum()
        raw_60m.to_csv(tmp_path / 'rth_60m.csv', index=False)
    slot_60m = build_slot_from_config(
        _nq_declarative_config(
            tmp_path, frequency='60m', view='RTH', volume_mode=volume_mode,
            start='2026-11-25', end='2026-11-25', csv_name='rth_60m.csv',
        ),
        base_dir=tmp_path,
    )
    bars_60m = slot_60m.bar_feed.read_execution_bars(
        ['NQZ26.CME'], Frequency.parse('60m'),
        start_utc=pd.Timestamp('2026-11-25 14:30Z'),
        end_utc=pd.Timestamp('2026-11-25 21:00Z'),
    )
    assert bars_60m.timestamp_utc.tolist() == list(rth_60m)
    assert bars_60m.volume.tolist() == [float(i + 1) for i in range(len(rth_60m))]

    # 2. Full ETH 1m CSV with cumulative volume per ETH session read under view='RTH'
    #    must diff against 14:29Z (09:29 ET), not treat 14:30Z as cumulative zero.
    if volume_mode == 'cumulative':
        eth_1m = pd.DatetimeIndex(cal.bar_grid('1m', view='ETH').timestamp_utc)
        _write_nq_csv(tmp_path / 'eth_cum_1m.csv', eth_1m, cumulative_per_eth_session=True, calendar=cal)
        slot_rth = build_slot_from_config(
            _nq_declarative_config(
                tmp_path, frequency='1m', view='RTH', volume_mode='cumulative',
                start='2026-11-25', end='2026-11-25', csv_name='eth_cum_1m.csv',
            ),
            base_dir=tmp_path,
        )
        bars_1m = slot_rth.bar_feed.read_execution_bars(
            ['NQZ26.CME'], Frequency.parse('1m'),
            start_utc=pd.Timestamp('2026-11-25 14:30Z'),
            end_utc=pd.Timestamp('2026-11-25 14:34Z'),
        )
        first_rth_idx = eth_1m.get_loc(pd.Timestamp('2026-11-25 14:30Z'))
        assert bars_1m.volume.tolist() == [float(first_rth_idx + 1 + k) for k in range(5)]


def test_declarative_rth_rejects_off_grid_and_misaligned_eth_bars(tmp_path):
    from utils.session_calendar import SessionCalendar

    cal = SessionCalendar('CMEGlobex_Equity', '2026-11-25', '2026-11-25', rth_reference_calendar='NYSE')
    # 1. Off-grid dirty timestamp inside RTH fails closed.
    _write_nq_csv(tmp_path / 'dirty.csv', pd.DatetimeIndex([pd.Timestamp('2026-11-25 14:30:30Z')]))
    dirty_slot = build_slot_from_config(
        _nq_declarative_config(
            tmp_path, frequency='1m', view='RTH', start='2026-11-25', end='2026-11-25', csv_name='dirty.csv',
        ),
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError, match='off the Bar schedule grid'):
        dirty_slot.bar_feed.read_execution_bars(
            ['NQZ26.CME'], Frequency.parse('1m'),
            start_utc=pd.Timestamp('2026-11-25 14:00Z'), end_utc=pd.Timestamp('2026-11-25 15:00Z'),
        )

    # 2. Maintenance-closed timestamp (outside both RTH and ETH) fails closed.
    _write_nq_csv(tmp_path / 'closed.csv', pd.DatetimeIndex([pd.Timestamp('2026-11-25 22:30Z')]))
    closed_slot = build_slot_from_config(
        _nq_declarative_config(
            tmp_path, frequency='1m', view='RTH', start='2026-11-25', end='2026-11-25', csv_name='closed.csv',
        ),
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError, match='outside the Bar schedule'):
        closed_slot.bar_feed.load(instruments=['NQZ26.CME'])

    # 3. ETH 60m bars overlapping RTH with misaligned boundaries fail closed under view='RTH'.
    eth_60m = cal.bar_grid('60m', view='ETH').timestamp_utc
    _write_nq_csv(tmp_path / 'eth_60m.csv', pd.DatetimeIndex(eth_60m))
    misaligned_slot = build_slot_from_config(
        _nq_declarative_config(
            tmp_path, frequency='60m', view='RTH', start='2026-11-25', end='2026-11-25', csv_name='eth_60m.csv',
        ),
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError, match='misaligned|off the Bar schedule grid'):
        misaligned_slot.bar_feed.load(instruments=['NQZ26.CME'])


@pytest.mark.parametrize('volume_mode', ['interval', 'cumulative'])
def test_declarative_rth_pure_60m_bar_end_retains_closing_anchor(tmp_path, volume_mode):
    from utils.session_calendar import SessionCalendar

    cal = SessionCalendar('CMEGlobex_Equity', '2026-11-25', '2026-11-25', rth_reference_calendar='NYSE')
    rth_schedule = cal.bar_grid('60m', view='RTH')
    rth_slots = cal.session_on('2026-11-25', 'RTH')
    assert rth_slots is not None
    # RTH 60m bar_end anchors: 15:30Z..20:30Z plus closing partial bar ending at 21:00Z.
    from utils.time_axis import BarSchedule
    rth_end_times = pd.DatetimeIndex(BarSchedule(cal, '60m', view='RTH').for_session('2026-11-25').end_utc)
    assert rth_end_times[-1] == pd.Timestamp('2026-11-25 21:00Z')
    raw = _write_nq_csv(tmp_path / 'rth_60m_end.csv', rth_end_times)
    if volume_mode == 'cumulative':
        raw['volume'] = raw['volume'].cumsum()
        raw.to_csv(tmp_path / 'rth_60m_end.csv', index=False)

    slot = build_slot_from_config(
        _nq_declarative_config(
            tmp_path, frequency='60m', view='RTH', timestamp_anchor='bar_end',
            volume_mode=volume_mode, start='2026-11-25', end='2026-11-25', csv_name='rth_60m_end.csv',
        ),
        base_dir=tmp_path,
    )
    loaded = slot.bar_feed.load(instruments=['NQZ26.CME'])
    assert loaded.timestamp_utc.tolist() == list(rth_schedule.timestamp_utc)
    assert loaded.volume.tolist() == [float(i + 1) for i in range(len(rth_end_times))]

    # Narrow query at closing bar [20:30Z, 21:00Z) must retain 20:30Z across execution and signal paths.
    close_start = pd.Timestamp('2026-11-25 20:30Z')
    exec_close = slot.bar_feed.read_execution_bars(
        ['NQZ26.CME'], Frequency.parse('60m'), start_utc=close_start, end_utc=close_start,
    )
    sig_close = slot.bar_feed.read_signal_bars(
        ['NQZ26.CME'], Frequency.parse('60m'),
        start_utc=close_start, end_utc=close_start, as_of_utc=pd.Timestamp('2026-11-25 21:00Z'),
    )
    assert exec_close.timestamp_utc.tolist() == [close_start]
    assert sig_close.timestamp_utc.tolist() == [close_start]
    assert exec_close.volume.tolist() == [float(len(rth_end_times))]
    assert sig_close.volume.tolist() == [float(len(rth_end_times))]


def test_declarative_rth_60m_bar_end_rejects_full_and_sparse_eth_on_window_and_load(tmp_path):
    from backtest_and_analysis.time_series.contracts import ContractValidationError
    from utils.session_calendar import SessionCalendar
    from utils.time_axis import BarSchedule

    cal = SessionCalendar('CMEGlobex_Equity', '2026-11-25', '2026-11-25', rth_reference_calendar='NYSE')
    eth_end_times = pd.DatetimeIndex(BarSchedule(cal, '60m', view='ETH').for_session('2026-11-25').end_utc)
    assert pd.Timestamp('2026-11-25 21:00Z') in eth_end_times

    close_start = pd.Timestamp('2026-11-25 20:30Z')
    as_of_close = pd.Timestamp('2026-11-25 21:00Z')

    # Case 1: Full ETH 60m bar_end CSV must fail-closed across load, read_execution_bars(20:30Z..20:30Z),
    # and read_signal_bars(as_of=21:00Z), never mislabeling ETH [20:00Z, 21:00Z) as RTH [20:30Z, 21:00Z).
    _write_nq_csv(tmp_path / 'eth_60m_end.csv', eth_end_times)
    full_eth_slot = build_slot_from_config(
        _nq_declarative_config(
            tmp_path, frequency='60m', view='RTH', timestamp_anchor='bar_end',
            start='2026-11-25', end='2026-11-25', csv_name='eth_60m_end.csv',
        ),
        base_dir=tmp_path,
    )
    for call in (
        lambda: full_eth_slot.bar_feed.load(instruments=['NQZ26.CME']),
        lambda: full_eth_slot.bar_feed.read_execution_bars(
            ['NQZ26.CME'], Frequency.parse('60m'), start_utc=close_start, end_utc=close_start,
        ),
        lambda: full_eth_slot.bar_feed.read_signal_bars(
            ['NQZ26.CME'], Frequency.parse('60m'),
            start_utc=close_start, end_utc=close_start, as_of_utc=as_of_close,
        ),
    ):
        with pytest.raises(ContractValidationError, match='misaligned|off the Bar schedule grid'):
            call()

    # Case 2: Sparse ETH 60m bar_end CSV containing only 00:00Z (ETH) and 21:00Z (shared anchor)
    # must fail-closed across load, read_execution_bars, and read_signal_bars instead of dropping 21:00Z.
    sparse_eth_end = pd.DatetimeIndex(['2026-11-25 00:00Z', '2026-11-25 21:00Z'])
    _write_nq_csv(tmp_path / 'sparse_eth_60m_end.csv', sparse_eth_end)
    sparse_eth_slot = build_slot_from_config(
        _nq_declarative_config(
            tmp_path, frequency='60m', view='RTH', timestamp_anchor='bar_end',
            start='2026-11-25', end='2026-11-25', csv_name='sparse_eth_60m_end.csv',
        ),
        base_dir=tmp_path,
    )
    for call in (
        lambda: sparse_eth_slot.bar_feed.load(instruments=['NQZ26.CME']),
        lambda: sparse_eth_slot.bar_feed.read_execution_bars(
            ['NQZ26.CME'], Frequency.parse('60m'), start_utc=close_start, end_utc=close_start,
        ),
        lambda: sparse_eth_slot.bar_feed.read_signal_bars(
            ['NQZ26.CME'], Frequency.parse('60m'),
            start_utc=close_start, end_utc=close_start, as_of_utc=as_of_close,
        ),
    ):
        with pytest.raises(ContractValidationError, match='misaligned'):
            call()

    # Case 3: Single shared-anchor 21:00Z bar is mathematically ambiguous without source_view -> fail-closed;
    # explicit source_view='RTH' accepts [20:30Z, 21:00Z), while source_view='ETH' rejects misaligned [20:00Z, 21:00Z).
    _write_nq_csv(tmp_path / 'ambiguous_2100.csv', pd.DatetimeIndex(['2026-11-25 21:00Z']))
    ambig_cfg = _nq_declarative_config(
        tmp_path, frequency='60m', view='RTH', timestamp_anchor='bar_end',
        start='2026-11-25', end='2026-11-25', csv_name='ambiguous_2100.csv',
    )
    ambig_slot = build_slot_from_config(ambig_cfg, base_dir=tmp_path)
    with pytest.raises(ContractValidationError, match='Ambiguous bar timestamps'):
        ambig_slot.bar_feed.load(instruments=['NQZ26.CME'])
    with pytest.raises(ContractValidationError, match='Ambiguous bar timestamps'):
        ambig_slot.bar_feed.read_execution_bars(
            ['NQZ26.CME'], Frequency.parse('60m'), start_utc=close_start, end_utc=close_start,
        )

    rth_declared = build_slot_from_config(
        replace(ambig_cfg, bar_feed=ProviderConfig('csv_files', {**ambig_cfg.bar_feed.options, 'source_view': 'RTH'})),
        base_dir=tmp_path,
    )
    assert rth_declared.bar_feed.read_execution_bars(
        ['NQZ26.CME'], Frequency.parse('60m'), start_utc=close_start, end_utc=close_start,
    ).timestamp_utc.tolist() == [close_start]

    eth_declared = build_slot_from_config(
        replace(ambig_cfg, bar_feed=ProviderConfig('csv_files', {**ambig_cfg.bar_feed.options, 'source_view': 'ETH'})),
        base_dir=tmp_path,
    )
    with pytest.raises(ContractValidationError, match='misaligned'):
        eth_declared.bar_feed.read_execution_bars(
            ['NQZ26.CME'], Frequency.parse('60m'), start_utc=close_start, end_utc=close_start,
        )

    # Invalid source_view or incompatible view='ETH' + source_view='RTH' fail-closed.
    for bad_sv in (None, 1, 'rth', 'unknown'):
        with pytest.raises(ValueError, match='source_view must be ETH or RTH'):
            build_slot_from_config(
                replace(ambig_cfg, bar_feed=ProviderConfig('csv_files', {**ambig_cfg.bar_feed.options, 'source_view': bad_sv})),
                base_dir=tmp_path,
            )
    with pytest.raises(ValueError, match="view='ETH' cannot read"):
        build_slot_from_config(
            replace(ambig_cfg, bar_feed=ProviderConfig('csv_files', {
                **ambig_cfg.bar_feed.options, 'view': 'ETH', 'source_view': 'RTH',
            })),
            base_dir=tmp_path,
        )


def test_declarative_rth_60m_bar_start_narrow_window_rejects_eth_and_retains_rth(tmp_path):
    from utils.session_calendar import SessionCalendar

    cal = SessionCalendar('CMEGlobex_Equity', '2026-11-25', '2026-11-25', rth_reference_calendar='NYSE')
    eth_starts = pd.DatetimeIndex(cal.bar_grid('60m', view='ETH').timestamp_utc)
    rth_starts = pd.DatetimeIndex(cal.bar_grid('60m', view='RTH').timestamp_utc)
    _write_nq_csv(tmp_path / 'eth_60m_start.csv', eth_starts)
    _write_nq_csv(tmp_path / 'rth_60m_start.csv', rth_starts)

    rth_open = pd.Timestamp('2026-11-25 14:30Z')
    rth_close_open = pd.Timestamp('2026-11-25 20:30Z')

    # Full ETH 60m bar_start CSV under view='RTH' (source_view omitted or explicit 'ETH'):
    # narrow window [14:30Z, 14:30Z] and [20:30Z, 20:30Z] must fail-closed across load, execution, and signal.
    for extra_opts in ({}, {'source_view': 'ETH'}):
        eth_cfg = _nq_declarative_config(
            tmp_path, frequency='60m', view='RTH', timestamp_anchor='bar_start',
            start='2026-11-25', end='2026-11-25', csv_name='eth_60m_start.csv',
        )
        eth_cfg = replace(
            eth_cfg,
            bar_feed=ProviderConfig('csv_files', {**eth_cfg.bar_feed.options, **extra_opts}),
        )
        feed = build_slot_from_config(eth_cfg, base_dir=tmp_path).bar_feed
        for call in (
            lambda: feed.load(instruments=['NQZ26.CME']),
            lambda: feed.read_execution_bars(
                ['NQZ26.CME'], Frequency.parse('60m'), start_utc=rth_open, end_utc=rth_open,
            ),
            lambda: feed.read_signal_bars(
                ['NQZ26.CME'], Frequency.parse('60m'),
                start_utc=rth_open, end_utc=rth_open, as_of_utc=pd.Timestamp('2026-11-25 15:30Z'),
            ),
            lambda: feed.read_execution_bars(
                ['NQZ26.CME'], Frequency.parse('60m'), start_utc=rth_close_open, end_utc=rth_close_open,
            ),
            lambda: feed.read_signal_bars(
                ['NQZ26.CME'], Frequency.parse('60m'),
                start_utc=rth_close_open, end_utc=rth_close_open, as_of_utc=pd.Timestamp('2026-11-25 21:00Z'),
            ),
        ):
            with pytest.raises(ValueError, match='misaligned'):
                call()

    # Pure RTH 60m bar_start CSV retains narrow-window bars and preserves causal as_of cutoff.
    rth_cfg = _nq_declarative_config(
        tmp_path, frequency='60m', view='RTH', timestamp_anchor='bar_start',
        start='2026-11-25', end='2026-11-25', csv_name='rth_60m_start.csv',
    )
    rth_feed = build_slot_from_config(rth_cfg, base_dir=tmp_path).bar_feed
    exec_first = rth_feed.read_execution_bars(
        ['NQZ26.CME'], Frequency.parse('60m'), start_utc=rth_open, end_utc=rth_open,
    )
    sig_first = rth_feed.read_signal_bars(
        ['NQZ26.CME'], Frequency.parse('60m'),
        start_utc=rth_open, end_utc=rth_open, as_of_utc=pd.Timestamp('2026-11-25 15:30Z'),
    )
    sig_before_complete = rth_feed.read_signal_bars(
        ['NQZ26.CME'], Frequency.parse('60m'),
        start_utc=rth_open, end_utc=rth_open, as_of_utc=pd.Timestamp('2026-11-25 15:29Z'),
    )
    exec_last = rth_feed.read_execution_bars(
        ['NQZ26.CME'], Frequency.parse('60m'), start_utc=rth_close_open, end_utc=rth_close_open,
    )
    assert exec_first.timestamp_utc.tolist() == [rth_open]
    assert sig_first.timestamp_utc.tolist() == [rth_open]
    assert sig_before_complete.empty
    assert exec_last.timestamp_utc.tolist() == [rth_close_open]

    # Aligned 1m RTH interval bar_start pushdown remains tight (no unnecessary overnight expansion).
    _write_nq_csv(tmp_path / 'eth_1m_start.csv', pd.DatetimeIndex(cal.bar_grid('1m', view='ETH').timestamp_utc))
    aligned_feed = build_slot_from_config(
        _nq_declarative_config(
            tmp_path, frequency='1m', view='RTH', timestamp_anchor='bar_start',
            start='2026-11-25', end='2026-11-25', csv_name='eth_1m_start.csv',
        ),
        base_dir=tmp_path,
    ).bar_feed
    push_lo, push_hi = aligned_feed._pushdown_window(
        pd.Timestamp('2026-11-25 14:30Z'), pd.Timestamp('2026-11-25 15:00Z'),
    )
    assert (push_lo, push_hi) == (pd.Timestamp('2026-11-25 14:30Z'), pd.Timestamp('2026-11-25 15:00Z'))
