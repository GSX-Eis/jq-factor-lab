"""Declarative sources, point-in-time components and the public import entry point."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import duckdb
import pandas as pd
import pytest

from market_db_samples import crypto_profile
from utils.market_db import init_market_db
from utils.market_db.storage import _create_profile_table

from slots import build_slot_from_config, register_builtin_slots
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
