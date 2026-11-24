"""Offline source parsing and S7 Harness/Slot contract checks for public markets."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd
import pytest

from data_pipeline.extractors.public_markets import akshare_indices, binance, french
from data_pipeline.extractors.public_markets.common import (
    ROOT, PublicMarket, Snapshot, canonical, daily_opens, instrument, sha256,
)
from utils.market_db.research import open_research_slot
from utils.slots.schema import load_slot_config
from utils.time_axis import Frequency

FIXTURES = Path(__file__).parent / 'fixtures/public_markets'


def zip_fixture(tmp_path, name):
    path = tmp_path / f'{name}.zip'
    with ZipFile(path, 'w') as archive:
        archive.write(FIXTURES / f'{name}.csv', f'{name}.csv')
    return path


def test_french_daily_sections_and_sentinels(tmp_path):
    value, equal = french.read_daily_returns(zip_fixture(tmp_path, 'french_daily'))
    assert list(value) == list(equal) == ['Agric', 'Food']
    assert value.Food.isna().tolist() == [True, False, True]
    assert equal.Agric.isna().tolist() == [False, True, False]
    index = french.close_index(value.Agric)
    assert index.index.strftime('%Y-%m-%d').tolist() == ['1963-07-01', '1963-07-03']
    assert index.tolist() == pytest.approx([1.1, 1.122])


def test_binance_timestamp_units_and_checksum_failure(tmp_path):
    path = zip_fixture(tmp_path, 'binance_klines')
    checksum = tmp_path / f'{path.name}.CHECKSUM'
    checksum.write_text(f'{sha256(path)}  {path.name}\n')
    binance.verify_checksum(path, checksum, path.name)
    parsed = binance.read_month(path, pd.Period('2025-01', 'M'), 'min')
    assert parsed.timestamp_utc.tolist() == [pd.Timestamp(f'2025-01-01 00:0{minute}Z') for minute in range(2)]
    assert list(parsed.close) == [100, 102]
    assert list(parsed.volume) == [1.5, 1.1]
    checksum.write_text(f'{"0" * 64}  {path.name}\n')
    with pytest.raises(ValueError, match='checksum failed'):
        binance.verify_checksum(path, checksum, path.name)


def test_frozen_snapshot_resume_rejects_mutated_input(tmp_path):
    snapshot = Snapshot(tmp_path)
    source = 'source description'
    original = snapshot.capture('raw.txt', source, lambda path: path.write_text('frozen data'))
    resumed = Snapshot(tmp_path)
    assert resumed.capture('raw.txt', source, lambda _: pytest.fail('fetched again')) == original
    original.write_text('changed')
    with pytest.raises(ValueError, match='Frozen snapshot changed'):
        resumed.capture('raw.txt', source, lambda _: pytest.fail('overwrote original'))


def test_akshare_field_mapping():
    broad = akshare_indices.read_index(FIXTURES / 'akshare_broad.csv', '000300.XSHG')
    industry = akshare_indices.read_index(FIXTURES / 'akshare_shenwan.csv', '801010.SW')
    pd.testing.assert_frame_equal(broad, industry, check_dtype=False)
    assert industry.loc['2024-11-01'].to_dict() == dict(open=100, high=108, low=99, close=105, volume=200)
    with pytest.raises(ValueError, match='disagrees'):
        akshare_indices.read_index(FIXTURES / 'akshare_shenwan.csv', '801030.SW')


def bars(symbol, timestamp, frequency='1d', *, version='frozen-fixture'):
    prices = pd.DataFrame(dict(timestamp_utc=[pd.Timestamp(timestamp)], open=[100.], high=[102.],
                               low=[99.], close=[101.], volume=[4.]))
    return canonical(prices, symbol, frequency, 'fixture', version)


def new_market(tmp_path, db_id, version='frozen-fixture'):
    snapshot = Snapshot(tmp_path)
    market = PublicMarket(db_id, tmp_path, tmp_path / 'test.duckdb', snapshot, version)
    market.initialize()
    return market


@pytest.mark.parametrize('case', ['cross_market', 'missing_point', 'off_grid', 'bad_suffix'])
def test_public_market_harness_rejects_invalid_bars(tmp_path, monkeypatch, case):
    monkeypatch.delenv('QUANT_OUTPUT_CONTEXT', raising=False)
    market = new_market(tmp_path, akshare_indices.DB_ID)
    symbol, ts, freq = '000300.XSHG', '2024-11-01T01:30:00Z', '1d'
    meta = instrument(symbol, 'index', 'Asia/Shanghai', 'XSHG')
    error = 'point_value'
    if case == 'cross_market':
        symbol, ts, freq = 'BTCUSDT.BINANCE', '2024-11-01T00:00:00Z', '1min'
        meta = instrument(symbol, 'crypto', 'UTC', binance.CALENDAR_ID, binance.SESSIONS)
        error = 'instrument_id|exchange|asset_class'
    if case == 'missing_point':
        del meta['point_value']
    if case == 'off_grid':
        ts, error = '2024-11-01T01:31:00Z', 'off the Bar schedule grid'
    if case == 'bad_suffix':
        symbol, error = '000300.BINANCE', 'instrument_id|exchange'
        meta['instrument_id'] = symbol
    paths = market.prepare('negative', [bars(symbol, ts, freq)], [meta])
    with pytest.raises(ValueError, match=error):
        market.import_bars(paths, [symbol], 1, freq)
    assert market.report['rejected_bars'] == 1


@pytest.mark.parametrize('db_id,symbol,asset,cal,zone', [
    (french.DB_ID, 'AGRIC_VW.KFDL', 'portfolio', 'XNYS', 'America/New_York'),
    (akshare_indices.DB_ID, '000300.XSHG', 'index', 'XSHG', 'Asia/Shanghai'),
    (binance.DB_ID, 'BTCUSDT.BINANCE', 'crypto', binance.CALENDAR_ID, 'UTC'),
])
def test_public_market_init_import_bind_and_check_hash(tmp_path, monkeypatch, db_id, symbol, asset, cal, zone):
    monkeypatch.delenv('QUANT_OUTPUT_CONTEXT', raising=False)
    market = new_market(tmp_path, db_id)
    day = '2024-11-01'
    is_crypto = db_id == binance.DB_ID
    meta = instrument(symbol, asset, zone, cal, binance.SESSIONS if is_crypto else None)
    ts = f'{day}T00:00:00Z' if is_crypto else daily_opens(cal, day, day).iloc[0]
    freq = '1min' if is_crypto else '1d'
    frame = bars(symbol, ts, freq)
    if is_crypto:
        # Two full UTC days, followed by a day with just the opening minute.
        times = pd.date_range(ts, periods=2881, freq='min')
        frame = frame.iloc[[0] * len(times)].assign(timestamp_utc=times)
    paths = market.prepare('valid', [frame], [meta])
    market.report['instruments'][symbol] = dict(rows=len(frame), missing_bars=1439 if is_crypto else 0)
    market.import_bars(paths, [symbol], len(frame), freq)
    market.import_bars(paths, [symbol], len(frame), freq)
    if is_crypto:
        args = ('build-bars', '--db', market.db, '--source', '1min', '--targets', '1d',
                '--instruments', symbol, '--start', day, '--end')
        market.harness(*args, day, '--dry-run')
        market.harness(*args, day)
        market.finish()
        published = load_slot_config(tmp_path / 'slots' / f'{db_id}_1d.json')
        monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'market_dbs': {db_id: str(market.db)}}))
        market.harness(*args, '2024-11-03', '--check-only')
        with open_research_slot(db_id, published, tmp_path) as (slot, _):
            assert slot.bar_feed.load(instruments=[symbol]).close.tolist() == [101.]
        market.harness(*args, '2024-11-03')
        with open_research_slot(db_id, published, tmp_path) as (slot, _):
            with pytest.raises(duckdb.CatalogException, match=published.bar_feed.options['table']):
                slot.bar_feed.load(instruments=[symbol])
        with duckdb.connect(str(market.db), read_only=True) as connection:
            complete = connection.execute('SELECT is_complete FROM market_bars_1d ORDER BY timestamp_utc').fetchall()
        assert complete == [(True,), (True,), (False,)]
    market.finish()
    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'market_dbs': {db_id: str(market.db)}}))
    for config_path in sorted((tmp_path / 'slots').glob('*.json')):
        config = load_slot_config(config_path)
        with open_research_slot(db_id, config, tmp_path) as (slot, _):
            assert slot.market_db_profile.profile_hash == market.profile.profile_hash
            loaded = slot.bar_feed.load(instruments=[symbol])
            assert loaded.close.eq(101.).all()
            if config.bar_feed.options['frequency'] == '1d':
                expected_days = [pd.Timestamp(ts)]
                if is_crypto:
                    expected_days.append(pd.Timestamp('2024-11-02T00:00:00Z'))
                    execution = slot.bar_feed.read_execution_bars(
                        [symbol], Frequency.parse('1d'), start_utc=pd.Timestamp(ts),
                        end_utc=pd.Timestamp('2024-11-03T00:00:00Z'),
                    )
                    assert execution.close.tolist() == [101., 101.]
                assert loaded.timestamp_utc.tolist() == expected_days
            else:
                assert len(loaded) == 2881
    other_id = akshare_indices.DB_ID if is_crypto else binance.DB_ID
    other = ROOT / 'config/market_slots' / other_id / f'{other_id}_1d.yaml'
    with pytest.raises(ValueError, match='market_domain'):
        with open_research_slot(db_id, load_slot_config(other), other.parent):
            pytest.fail('opened wrong Slot')


@pytest.mark.parametrize('full_version', ['frozen-A', 'frozen-B'], ids=['same-source', 'different-source'])
def test_frozen_slot_refuses_different_database_snapshot(tmp_path, monkeypatch, full_version):
    monkeypatch.delenv('QUANT_OUTPUT_CONTEXT', raising=False)
    db_id = akshare_indices.DB_ID
    symbol = '000300.XSHG'
    first, second = daily_opens('XSHG', '2024-11-01', '2024-11-04')
    meta = instrument(symbol, 'index', 'Asia/Shanghai', 'XSHG')
    first_bar = bars(symbol, first)
    second_bar = bars(symbol, second)
    second_bar[['open', 'high', 'low', 'close']] *= 2
    full_bars = pd.concat([first_bar, second_bar], ignore_index=True)

    def snapshot(name, version, rows):
        market = new_market(tmp_path / name, db_id, version=version)
        frame = full_bars.iloc[:rows].assign(data_version=version)
        paths = market.prepare('valid', [frame], [meta])
        market.report['instruments'][symbol] = dict(rows=len(frame), missing_bars=0)
        market.import_bars(paths, [symbol], len(frame), '1d')
        legacy_view = f'public_bars_1d_{hashlib.sha256(version.encode()).hexdigest()}'
        with duckdb.connect(str(market.db)) as connection:
            connection.execute(f'CREATE VIEW {legacy_view} AS SELECT * FROM market_bars_1d')
        market.finish()
        with duckdb.connect(str(market.db), read_only=True) as connection:
            assert connection.execute('SELECT view_name FROM duckdb_views() WHERE view_name = ?',
                                      [legacy_view]).fetchall() == []
        return market, load_slot_config(market.workspace / 'slots' / f'{db_id}_1d.json')

    original, original_config = snapshot('sample', 'frozen-A', 1)
    changed, changed_config = snapshot('full', full_version, 2)
    assert original.profile.profile_hash == changed.profile.profile_hash
    assert original.report['database_snapshot_id'] != changed.report['database_snapshot_id']

    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'market_dbs': {db_id: str(original.db)}}))
    with open_research_slot(db_id, original_config, tmp_path) as (slot, _):
        loaded = slot.bar_feed.load(instruments=[symbol])
        assert loaded.close.tolist() == [101.]
        assert loaded.data_version.tolist() == ['frozen-A']

    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'market_dbs': {db_id: str(changed.db)}}))
    with open_research_slot(db_id, original_config, tmp_path) as (slot, _):
        with pytest.raises(duckdb.CatalogException, match=original_config.bar_feed.options['table']):
            slot.bar_feed.load(instruments=[symbol])
    with open_research_slot(db_id, changed_config, tmp_path) as (slot, _):
        loaded = slot.bar_feed.load(instruments=[symbol])
        assert loaded.close.tolist() == [101., 202.]
        assert loaded.data_version.tolist() == [full_version, full_version]

    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'market_dbs': {db_id: str(original.db)}}))
    with open_research_slot(db_id, changed_config, tmp_path) as (slot, _):
        with pytest.raises(duckdb.CatalogException, match=changed_config.bar_feed.options['table']):
            slot.bar_feed.load(instruments=[symbol])


def test_published_slot_stays_invalid_until_interrupted_import_is_published(tmp_path, monkeypatch):
    monkeypatch.delenv('QUANT_OUTPUT_CONTEXT', raising=False)
    db_id, version = akshare_indices.DB_ID, 'frozen-A'
    market = new_market(tmp_path, db_id, version)
    first, second = '000300.XSHG', '000905.XSHG'
    symbols = [first, second]
    timestamp = daily_opens('XSHG', '2024-11-01', '2024-11-01').iloc[0]
    paths = {}
    for symbol, price_scale in zip(symbols, (1, 2), strict=True):
        frame = bars(symbol, timestamp, version=version)
        frame[['open', 'high', 'low', 'close']] *= price_scale
        meta = instrument(symbol, 'index', 'Asia/Shanghai', 'XSHG')
        paths[symbol] = market.prepare(symbol, [frame], [meta])

    market.import_bars(paths[first], [first], 1, '1d')
    market.report['instruments'][first] = dict(rows=1, missing_bars=0)
    market.finish()
    config_path = market.workspace / 'slots' / f'{db_id}_1d.json'
    old_config = load_slot_config(config_path)
    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({'market_dbs': {db_id: str(market.db)}}))

    market.harness('import-bars', '--db', market.db, '--bars', paths[second][0],
                   '--metadata', paths[second][1], '--dry-run')
    market.import_bars(paths[first], [first], 1, '1d')  # The committed batch is skipped.
    with open_research_slot(db_id, old_config, tmp_path) as (slot, _):
        assert slot.bar_feed.load(instruments=symbols).close.tolist() == [101.]

    market.import_bars(paths[second], [second], 1, '1d')
    # Stop after the physical commit and reopen the database before publishing.
    resumed = new_market(tmp_path, db_id, version)
    with duckdb.connect(str(resumed.db), read_only=True) as connection:
        assert connection.execute('SELECT count(*) FROM market_bars_1d').fetchone() == (2,)
    with open_research_slot(db_id, old_config, tmp_path) as (slot, _):
        with pytest.raises(duckdb.CatalogException, match=old_config.bar_feed.options['table']):
            slot.bar_feed.load(instruments=symbols)

    for symbol in symbols:
        resumed.import_bars(paths[symbol], [symbol], 1, '1d')
        resumed.report['instruments'][symbol] = dict(rows=1, missing_bars=0)
    resumed.finish()
    new_config = load_slot_config(config_path)
    assert new_config.bar_feed.options['table'] != old_config.bar_feed.options['table']
    with open_research_slot(db_id, new_config, tmp_path) as (slot, _):
        loaded = slot.bar_feed.load(instruments=symbols)
        assert loaded.set_index('instrument_id').close.to_dict() == {first: 101., second: 202.}
        assert loaded.data_version.tolist() == [version, version]
