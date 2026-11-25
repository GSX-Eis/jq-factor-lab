"""Market profile enforcement at the public storage and Slot boundaries."""
from dataclasses import replace
import json

import duckdb
import pandas as pd
import pytest

from data_pipeline.loaders.market_bar_importer import import_market_bars
from market_db_samples import declare_market_db, market_profile
from slots import build_slot_from_config
from slots.jq_ashare_local import daily_slot_config
from tools.agent_harness.cli import main
from utils.bars.builder import build_bars
from utils.market_db import MarketDbProfileError, adopt_market_db_sidecar, init_market_db
from utils.slots import ProviderConfig
from utils.time_axis import Frequency


def bar_inputs(instrument='000001.XSHG', timestamp='2026-01-05 01:30Z', **metadata):
    bars = pd.DataFrame(dict(instrument_id=[instrument], timestamp_utc=pd.to_datetime([timestamp], utc=True),
                            frequency='5m', open=100., high=101., low=99., close=100., volume=10.,
                            source='fixture', data_version='v1'))
    meta = pd.DataFrame([{**dict(instrument_id=instrument, asset_class='equity', exchange='XSHG',
                                timezone='Asia/Shanghai', calendar_id='XSHG', rth_reference_calendar='XSHG',
                                price_adjustment='none'),
                          **metadata}])
    return bars, meta


@pytest.mark.parametrize('instrument,timestamp,metadata,error', [
    ('AAPL.XNAS', '2026-01-05 14:30Z', {'exchange': 'XNAS'}, 'instrument_id'),
    ('000001', '2026-01-05 01:30Z', {}, 'instrument_id'),
    ('AAPL', '2026-01-05 01:30Z', {}, 'instrument_id'),
    ('000001.XSHG', '2026-01-05 04:00Z', {}, 'open Session'),
    ('000001.XSHG', '2026-01-05 01:30Z', {'exchange': 'SZSE'}, 'exchange'),
    ('000001.XSHG', '2026-01-05 01:30Z', {'asset_class': 'crypto', 'point_value': 1.}, 'asset_class'),
    ('000001.XSHG', '2026-01-05 01:30Z', {'timezone': 'UTC'}, 'timezone'),
    ('000001.XSHG', '2026-01-05 01:30Z', {'calendar_id': 'XNYS'}, 'calendar_id'),
    ('000001.XSHG', '2027-01-05 01:30Z', {}, 'exceeds'),
])
def test_import_rejects_market_mismatch_before_writes(instrument, timestamp, metadata, error):
    with duckdb.connect() as connection:
        declare_market_db(connection, 'cn_equity')
        with pytest.raises(ValueError, match=error):
            import_market_bars(connection, *bar_inputs(instrument, timestamp, **metadata))
        assert connection.execute('SHOW TABLES').fetchall() == [('market_db_profile',)]


def test_cme_import_requires_explicit_calendar_id():
    bars, metadata = bar_inputs(
        'NQM26.CME', '2026-01-05 14:30Z', asset_class='futures', exchange='CME',
        timezone='America/Chicago', calendar_id=None, rth_reference_calendar='NYSE', point_value=20., contract_expiry='2026-06',
    )
    with duckdb.connect() as connection:
        declare_market_db(connection, 'global_futures')
        with pytest.raises(MarketDbProfileError, match='Explicit calendar_id'):
            import_market_bars(connection, bars, metadata)
        assert connection.execute('SHOW TABLES').fetchall() == [('market_db_profile',)]


def test_missing_profile_refuses_all_three_entry_points(tmp_path, capsys):
    path = tmp_path / 'unprofiled.duckdb'
    bars, metadata = bar_inputs()
    csv, meta = tmp_path / 'bars.csv', tmp_path / 'instruments.json'
    bars.to_csv(csv, index=False)
    metadata.to_json(meta, orient='records')
    common = ['--db', str(path), '--output-root', str(tmp_path / 'out')]
    with duckdb.connect(str(path)) as connection:
        pass
    for dry in ([], ['--dry-run']):
        for command in (
            ['import-bars', '--bars', str(csv), '--metadata', str(meta)],
            ['build-bars', '--source', '5m', '--targets', '1d', '--instruments', '000001.XSHG',
             '--start', '2026-01-05', '--end', '2026-01-05'],
        ):
            assert main([*command, *common, *dry]) == 2
            assert json.loads(capsys.readouterr().out)['summary']['reason'] == 'MARKET_DB_PROFILE_MISSING'
    assert main(['import-bars', '--bars', str(csv), '--metadata', str(meta), *common,
                 '--db', str(tmp_path / 'absent.duckdb')]) == 2
    assert not (tmp_path / 'absent.duckdb').exists()
    with duckdb.connect(str(path)) as connection:
        with pytest.raises(MarketDbProfileError, match='MARKET_DB_PROFILE_MISSING'):
            build_slot_from_config(daily_slot_config(), connection=connection)
        assert connection.execute('SHOW TABLES').fetchall() == []


def test_build_checks_frequency_instrument_and_coverage_before_writes():
    bars, meta = bar_inputs(exchange='SSE')
    with duckdb.connect() as connection:
        declare_market_db(connection, 'cn_equity')
        with pytest.raises(MarketDbProfileError, match='frequency'):
            import_market_bars(connection, bars, meta, derive=('15m',))
        assert connection.execute('SHOW TABLES').fetchall() == [('market_db_profile',)]
        import_market_bars(connection, bars, meta)
        tables = connection.execute('SHOW TABLES').fetchall()
        args = dict(source='5m', targets=['1d'], instruments=['000001.XSHG'], start='2026-01-05', end='2026-01-05')
        for override, error in (({'targets': ['15m']}, 'frequency'),
                                ({'instruments': ['000001']}, 'instrument_id'),
                                ({'end': '2027-01-05'}, 'exceeds')):
            with pytest.raises(MarketDbProfileError, match=error):
                build_bars(connection, **{**args, **override})
        assert connection.execute('SHOW TABLES').fetchall() == tables


def test_native_slot_rejects_another_market_even_if_provider_is_allowed():
    with duckdb.connect() as connection:
        declare_market_db(connection, 'us_equity', allowed_slot_providers=('jq_ashare_local',))
        with pytest.raises(MarketDbProfileError, match='DOMAIN_MISMATCH'):
            build_slot_from_config(daily_slot_config(), connection=connection)


def test_tampered_hash_blocks_import_and_slot():
    with duckdb.connect() as connection:
        declare_market_db(connection, 'cn_equity')
        connection.execute("UPDATE market_db_profile SET calendar_id='XNYS'")
        with pytest.raises(MarketDbProfileError, match='HASH_MISMATCH'):
            import_market_bars(connection, *bar_inputs())
        with pytest.raises(MarketDbProfileError, match='HASH_MISMATCH'):
            build_slot_from_config(daily_slot_config(), connection=connection)


def test_sidecar_allows_verified_reads_only_and_detects_database_change(tmp_path):
    path = tmp_path / 'adopted.duckdb'
    with duckdb.connect(str(path)) as connection:
        connection.execute('CREATE TABLE legacy (a INT)')
    profile = market_profile('cn_equity', calendar_coverage='2026-01-05..2026-01-09')
    adopt_market_db_sidecar(path, profile)
    with duckdb.connect(str(path), read_only=True) as connection:
        slot = build_slot_from_config(daily_slot_config(), connection=connection)
        assert slot.market_db_profile == profile
        assert slot.calendar.end == pd.Timestamp('2026-01-09')
        with pytest.raises(MarketDbProfileError, match='read-only'):
            import_market_bars(connection, *bar_inputs())
    with duckdb.connect(str(path)) as connection:
        connection.execute('INSERT INTO legacy VALUES (7)')
    with duckdb.connect(str(path), read_only=True) as connection:
        with pytest.raises(MarketDbProfileError, match='SIDECAR_FINGERPRINT_MISMATCH'):
            build_slot_from_config(daily_slot_config(), connection=connection)


def us_declaration(path):
    return replace(
        daily_slot_config(), slot_id='us_fixture', market_domain='us_equity',
        calendar=ProviderConfig('pmc', {'calendar_id': 'XNYS'}),
        bar_feed=ProviderConfig('duckdb_table', {
            'path': str(path), 'table': 'landing', 'frequency': '1d',
            'source_timezone': 'America/New_York', 'timestamp_anchor': 'bar_end',
            'data_version': 'v1', 'volume_mode': 'interval',
            'column_mapping': {'time': 'timestamp', **{c: c for c in ('open', 'high', 'low', 'close', 'volume')}},
            'instrument_id': 'AAPL.XNAS',
        }),
        universe=ProviderConfig('static', {'instruments': {'AAPL.XNAS': {'asset_class': 'equity', 'currency': 'USD'}}}),
        aux_features=ProviderConfig('none'), microstructure=ProviderConfig('open'),
    )


def test_declarative_us_slot_uses_profile_coverage_and_timezone(tmp_path):
    path = tmp_path / 'us.duckdb'
    profile = market_profile('us_equity', calendar_coverage='2026-03-06..2026-03-09')
    init_market_db(path, profile)
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE landing AS SELECT TIMESTAMP '2026-03-09 16:00:00' AS time,
                100.0 AS open, 101.0 AS high, 99.0 AS low, 100.0 AS "close", 10.0 AS volume
        """)
    slot = build_slot_from_config(us_declaration(path))
    assert slot.market_db_profile == profile
    assert slot.calendar.calendar_id == 'XNYS'
    assert slot.calendar.timezone == 'America/New_York'
    assert slot.calendar.start == pd.Timestamp('2026-03-06')
    bars = slot.bar_feed.read_execution_bars(
        ['AAPL.XNAS'], Frequency.parse('1d'),
        start_utc=pd.Timestamp('2026-03-09 00:00Z'), end_utc=pd.Timestamp('2026-03-10 00:00Z'),
    )
    assert bars.timestamp_utc.tolist() == [pd.Timestamp('2026-03-09 20:00Z')]
    assert slot.bar_feed.metadata.exchange.tolist() == ['XNAS']
    with duckdb.connect(str(path)) as connection:
        connection.execute("UPDATE market_db_profile SET market_domain='cn_equity'")
    with pytest.raises(MarketDbProfileError, match='HASH_MISMATCH'):
        slot.bar_feed.load()


@pytest.mark.parametrize(('driver', 'column', 'mapping', 'has_st'), [
    ('duckdb_table', 'is_st', {}, True),
    ('duckdb_table', 'is_st', {'is_st': 'other'}, False),
    ('csv_files', 'status_flag', {'status_flag': 'is_st'}, True),
    ('parquet_glob', 'status_flag', {'status_flag': 'is_st'}, True),
], ids=['raw-table', 'renamed-away', 'mapped-csv', 'mapped-parquet'])
@pytest.mark.parametrize('declared', [False, True])
def test_declarative_st_capability_matches_vectorbt(tmp_path, driver, column, mapping, has_st, declared):
    from factor_synthesis.vectorbt_adapter.slot_provider import SlotDataProvider

    path = tmp_path / 'us.duckdb'
    profile = market_profile('us_equity', calendar_coverage='2026-03-06..2026-03-09')
    profile = replace(profile, profile_hash='', market_rules={**profile.market_rules, 'st_filter': declared})
    init_market_db(path, profile)
    events = pd.DataFrame({
        'instrument_id': ['AAPL.XNAS'], 'available_at_utc': ['2026-03-09T19:55:00Z'], column: [True],
    })
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE landing AS SELECT TIMESTAMP '2026-03-09 16:00:00' AS time,
                100.0 AS open, 101.0 AS high, 99.0 AS low, 100.0 AS "close", 10.0 AS volume
        """)
        if driver == 'duckdb_table':
            connection.register('fixture_events', events)
            connection.execute('CREATE TABLE aux AS SELECT * FROM fixture_events')
    options = {'column_mapping': mapping}
    if driver == 'duckdb_table':
        options.update(path=str(path), table='aux')
    elif driver == 'csv_files':
        events.to_csv(tmp_path / 'aux.csv', index=False)
        options['paths'] = [str(tmp_path / 'aux.csv')]
    else:
        events.to_parquet(tmp_path / 'aux.parquet', index=False)
        options['path'] = str(tmp_path / 'aux.parquet')
    config = replace(us_declaration(path), aux_features=ProviderConfig(driver, options))

    if declared != has_st:
        with pytest.raises(MarketDbProfileError) as exc:
            build_slot_from_config(config)
        assert exc.value.code == 'MARKET_DB_PROFILE_INVALID'
        assert f'market_rules.st_filter: profile={declared}, implemented={has_st}' in str(exc.value)
    else:
        slot = build_slot_from_config(config)
        provider = SlotDataProvider(slot=slot, frequency='1d')
        instrument = ['AAPL.XNAS']
        assert provider.filter_tradeable_stocks('2026-03-09', instrument, side='buy') == (
            [] if has_st else instrument
        )
        assert provider.filter_tradeable_stocks(
            '2026-03-09', instrument, side='buy', require_non_st=False,
        ) == instrument


def test_declarative_slot_binds_partitioned_parquet_aux(tmp_path):
    path = tmp_path / 'us.duckdb'
    init_market_db(path, market_profile('us_equity', calendar_coverage='2026-03-06..2026-03-09'))
    dataset = tmp_path / 'aux.parquet'
    pd.DataFrame({
        'instrument_id': ['AAPL.XNAS'],
        'available_at_utc': ['2026-03-09T19:55:00Z'],
        'funding': [0.25],
        'year': [2026],
    }).to_parquet(dataset, index=False, partition_cols=['year'])
    config = replace(us_declaration(path), aux_features=ProviderConfig('parquet_glob', {
        'path': str(dataset), 'column_mapping': {'year': 'dataset_year'},
    }))

    slot = build_slot_from_config(config)
    grid = slot.calendar.bar_grid('1d', '2026-03-09', '2026-03-09').assign(instrument_id='AAPL.XNAS')
    result = slot.read_aux_features(grid, features=['funding', 'dataset_year'])
    assert result.funding.tolist() == [0.25]
    assert result.dataset_year.tolist() == [2026]


def test_declarative_duckdb_source_requires_market_profile(tmp_path):
    path = tmp_path / 'unprofiled.duckdb'
    duckdb.connect(str(path)).close()
    with pytest.raises(MarketDbProfileError, match='MARKET_DB_PROFILE_MISSING'):
        build_slot_from_config(us_declaration(path))


@pytest.mark.parametrize('overrides,error', [
    ({'allowed_slot_providers': ('csv_files',)}, 'PROVIDER_MISMATCH'),
    ({'calendar_id': 'XSHG'}, 'calendar_id'),
    ({'timezone': 'Asia/Shanghai'}, 'timezone'),
    ({'market_domain': 'custom:other'}, 'DOMAIN_MISMATCH'),
])
def test_declarative_slot_rejects_profile_mismatches(tmp_path, overrides, error):
    path = tmp_path / 'incompatible.duckdb'
    init_market_db(path, market_profile('us_equity', **overrides))
    with pytest.raises(MarketDbProfileError, match=error):
        build_slot_from_config(us_declaration(path))


def test_import_rolls_back_when_an_allowed_derive_cannot_align():
    with duckdb.connect() as connection:
        declare_market_db(connection, 'cn_equity', frequencies=('1d', '5m', '7min'))
        with pytest.raises(ValueError, match='boundaries'):
            import_market_bars(connection, *bar_inputs(), derive=('7min',))
        assert connection.execute('SHOW TABLES').fetchall() == [('market_db_profile',)]


def test_manifest_provider_is_checked_before_opening_source(tmp_path, capsys):
    path = tmp_path / 'destination.duckdb'
    init_market_db(path, market_profile('us_equity'))
    config = us_declaration(tmp_path / 'absent.duckdb')
    config = replace(config, bar_feed=ProviderConfig('csv_files', {'paths': ['absent.csv']}))
    manifest = tmp_path / 'slot.json'
    manifest.write_text(json.dumps(config.as_dict()), encoding='utf-8')
    assert main(['import-bars', '--db', str(path), '--manifest', str(manifest),
                 '--output-root', str(tmp_path / 'out')]) == 2
    assert json.loads(capsys.readouterr().out)['summary']['reason'] == 'MARKET_DB_PROVIDER_MISMATCH'
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute('SHOW TABLES').fetchall() == [('market_db_profile',)]


def test_vectorbt_annualization_uses_the_bound_profile():
    from factor_synthesis.vectorbt_adapter.slot_provider import SlotDataProvider

    with duckdb.connect() as connection:
        declare_market_db(connection, 'cn_equity', market_rules={
            'st_filter': True, 'price_limit': True, 't_plus': 1, 'bars_per_session': {'5m': 47},
        })
        config = daily_slot_config(start='2026-01-05', end='2026-01-09')
        slot = build_slot_from_config(config, connection=connection)
        _, _, bars_per_year = SlotDataProvider(slot=slot, frequency='5min').annualization_spec()
        assert bars_per_year == 252 * 47


@pytest.mark.parametrize('profile_source', ['table', 'sidecar'])
def test_native_slot_reads_frozen_input_views_with_physical_profile(tmp_path, profile_source):
    from utils.data_store import connect_input, migrate_table

    source, target = tmp_path / 'source.duckdb', tmp_path / 'input.duckdb'
    with duckdb.connect(str(source)) as connection:
        connection.execute("""
            CREATE TABLE raw_prices AS SELECT '000001.XSHG' AS stock_code, DATE '2026-03-09' AS date,
                10. AS open, 12. AS high, 9. AS low, 11. AS "close", 10. AS volume, 1. AS factor
        """)
    version = migrate_table(source, target, 'raw_prices', role='input', execute=True)['ref']
    profile = market_profile('cn_equity', calendar_coverage='2026-03-06..2026-03-09')
    if profile_source == 'sidecar':
        adopt_market_db_sidecar(target, profile)
    else:
        with duckdb.connect(str(target)) as connection:
            declare_market_db(connection, 'cn_equity', calendar_coverage=profile.calendar_coverage)
    bindings = {'input_db': str(target), 'inputs': {'price_data_daily': version}}
    with connect_input(target, bindings=bindings) as connection:
        slot = build_slot_from_config(daily_slot_config(), connection=connection)
        assert slot.market_db_profile == profile
        bars = slot.bar_feed.read_execution_bars(
            ['000001.XSHG'], Frequency.parse('1d'),
            start_utc=pd.Timestamp('2026-03-09 00:00Z'), end_utc=pd.Timestamp('2026-03-09 12:00Z'),
        )
        assert bars.close.tolist() == [11.]
        with pytest.raises(MarketDbProfileError, match='read-only'):
            import_market_bars(connection, *bar_inputs())


def test_manifest_uses_bound_source_domain_without_explicit_domain(tmp_path, capsys):
    source, target = tmp_path / 'us.duckdb', tmp_path / 'other.duckdb'
    profile = market_profile('us_equity', calendar_coverage='2026-03-06..2026-03-09')
    init_market_db(source, profile)
    init_market_db(target, replace(profile, profile_hash='', db_id='other', market_domain='custom:other'))
    with duckdb.connect(str(source)) as connection:
        connection.execute("""
            CREATE TABLE landing AS SELECT TIMESTAMP '2026-03-09 16:00:00' AS time,
                100. AS open, 101. AS high, 99. AS low, 100. AS "close", 10. AS volume
        """)
    config = replace(us_declaration(source), market_domain=None)
    manifest = tmp_path / 'slot.json'
    manifest.write_text(json.dumps(config.as_dict()), encoding='utf-8')
    for dry_run in ([], ['--dry-run']):
        assert main(['import-bars', '--db', str(target), '--manifest', str(manifest),
                     '--output-root', str(tmp_path / 'out'), *dry_run]) == 2
        assert json.loads(capsys.readouterr().out)['summary']['reason'] == 'MARKET_DB_DOMAIN_MISMATCH'
    with duckdb.connect(str(target), read_only=True) as connection:
        assert connection.execute('SHOW TABLES').fetchall() == [('market_db_profile',)]


@pytest.mark.parametrize('profile_domain,error', [
    (None, 'MARKET_DB_PROFILE_MISSING'),
    ('us_equity', 'MARKET_DB_DOMAIN_MISMATCH'),
])
def test_factor_base_checks_profile_before_first_path_panel_read(tmp_path, profile_domain, error):
    from utils.factor_base import FactorBase

    path = tmp_path / 'factor-input.duckdb'
    with duckdb.connect(str(path)) as connection:
        if profile_domain:
            declare_market_db(connection, profile_domain, allowed_slot_providers=('jq_ashare_local',))
        connection.execute("""
            CREATE TABLE price_data_daily AS SELECT 'AAPL.XNAS' AS stock_code, DATE '2026-03-09' AS date,
                100. AS "close"
        """)
    factor = FactorBase(None, 'fixture', '2026-03-09', '2026-03-09', stock_pool=['AAPL.XNAS'],
                        duckdb_path=str(path), data_bindings={}, pit_enabled=False)
    try:
        assert factor._duckdb_adapter._conn is None
        with pytest.raises(MarketDbProfileError, match=error):
            factor.get_price_data()
    finally:
        factor._duckdb_adapter.close()


def test_factor_base_rechecks_profile_after_path_adapter_reopens(tmp_path):
    from utils.factor_base import FactorBase

    path = tmp_path / 'factor-input.duckdb'
    with duckdb.connect(str(path)) as connection:
        declare_market_db(connection, 'cn_equity')
        connection.execute("""
            CREATE TABLE price_data_daily AS SELECT '000001.XSHG' AS stock_code,
                DATE '2026-03-09' AS date, 11. AS "close"
        """)
    factor = FactorBase(None, 'fixture', '2026-03-09', '2026-03-09', stock_pool=['000001.XSHG'],
                        duckdb_path=str(path), data_bindings={}, pit_enabled=False)
    try:
        assert factor.get_price_data()['000001.XSHG'].tolist() == [11.]
        factor._duckdb_adapter.close()
        with duckdb.connect(str(path)) as connection:
            connection.execute('DROP TABLE market_db_profile')
        with pytest.raises(MarketDbProfileError, match='MARKET_DB_PROFILE_MISSING'):
            factor.get_price_data()
    finally:
        factor.close()


@pytest.mark.parametrize('domain,error', [
    (None, 'MARKET_DB_PROFILE_MISSING'),
    ('us_equity', 'MARKET_DB_DOMAIN_MISMATCH'),
])
def test_factor_base_checks_file_profile_before_generating_dynamic_pool(tmp_path, monkeypatch, domain, error):
    from calendar_test_support import seed_calendar
    from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter
    from utils.factor_base import FactorBase

    path = tmp_path / 'pool.duckdb'
    with duckdb.connect(str(path)) as connection:
        if domain:
            declare_market_db(connection, domain, allowed_slot_providers=('jq_ashare_local',))
        seed_calendar(connection, start='2026-03-09', end='2026-03-10')
        connection.execute("""
            CREATE TABLE factor_data AS SELECT DATE '2026-03-09' AS date,
                'AAPL.XNAS' AS stock_code, 100. AS val_market_cap
        """)
    monkeypatch.setattr(DynamicStockPoolManager, '_save_filter_log', lambda *args: None)
    with pytest.raises(MarketDbProfileError, match=error):
        factor = FactorBase(None, 'pool', '2026-03-09', '2026-03-10', duckdb_path=str(path),
                            data_bindings={}, use_dynamic_pool=True, rebalance_schedule='weekly',
                            stock_filters=[StockFilter('val_market_cap', '>', 0)])
        factor.close()


@pytest.mark.parametrize('frozen', [False, True], ids=['file', 'frozen'])
@pytest.mark.parametrize('domain,error', [
    (None, 'MARKET_DB_PROFILE_MISSING'),
    ('us_equity', 'MARKET_DB_DOMAIN_MISMATCH'),
    ('cn_equity', None),
])
def test_factor_base_checks_borrowed_pool_connection_before_reading(tmp_path, monkeypatch, frozen, domain, error):
    from calendar_test_support import seed_calendar
    from utils.data_store import connect_input, migrate_table
    from utils.dynamic_stock_pool import DynamicStockPoolManager, StockFilter
    from utils.factor_base import FactorBase
    from utils.session_calendar import SessionCalendar

    start, end = '2026-03-09', '2026-03-10'
    instrument = '000001.XSHG' if domain == 'cn_equity' else 'AAPL.XNAS'
    source = tmp_path / 'pool.duckdb'
    with duckdb.connect(str(source)) as connection:
        seed_calendar(connection, start=start, end=end)
        connection.execute("""
            CREATE TABLE factor_data AS SELECT DATE '2026-03-09' AS date,
                ? AS stock_code, 100. AS val_market_cap
        """, [instrument])
    path, bindings = source, {}
    if frozen:
        path = tmp_path / 'input.duckdb'
        bindings = {'input_db': str(path), 'inputs': {
            table: migrate_table(source, path, table, role='input', execute=True)['ref']
            for table in ('trading_calendar', 'factor_data')
        }}
    if domain:
        with duckdb.connect(str(path)) as connection:
            declare_market_db(connection, domain, allowed_slot_providers=('jq_ashare_local',))

    calls = []
    validate_filters = DynamicStockPoolManager.validate_filters

    def record_validation(manager, filters):
        calls.append(filters)
        return validate_filters(manager, filters)

    monkeypatch.setattr(DynamicStockPoolManager, 'validate_filters', record_validation)
    monkeypatch.setattr(DynamicStockPoolManager, '_save_filter_log', lambda *args: None)
    options = dict(data_bindings={}, use_dynamic_pool=True, rebalance_schedule='weekly',
                   stock_filters=[StockFilter('val_market_cap', '>', 0)],
                   session_calendar=SessionCalendar('XSHG', start, end))
    borrowed = connect_input(path, bindings=bindings) if frozen else duckdb.connect(str(path), read_only=True)
    with borrowed:
        if error:
            with pytest.raises(MarketDbProfileError, match=error):
                FactorBase(borrowed, 'pool', start, end, **options).close()
            assert calls == []
        else:
            factor = FactorBase(borrowed, 'pool', start, end, **options)
            try:
                assert factor.stock_pools_by_period == {end: [instrument]}
            finally:
                factor.close()
        assert borrowed.execute('SELECT stock_code FROM factor_data').fetchone() == (instrument,)


@pytest.mark.parametrize('domain', ['cn_equity', 'us_equity'])
def test_vectorbt_checks_implicit_frozen_input_before_reading_calendar(tmp_path, monkeypatch, domain):
    from factor_synthesis.vectorbt_adapter.slot_provider import SlotDataProvider
    from utils.data_store import migrate_table
    from utils.session_calendar import VERIFIED_XSHG

    source, target = tmp_path / 'calendar.duckdb', tmp_path / 'input.duckdb'
    with duckdb.connect(str(source)) as connection:
        connection.execute("""
            CREATE TABLE trading_calendar AS SELECT ? AS calendar_id, day::DATE AS date, TRUE AS is_open
            FROM generate_series(DATE '2026-03-09', DATE '2026-03-10', INTERVAL 1 DAY) AS dates(day)
        """, [VERIFIED_XSHG])
    version = migrate_table(source, target, 'trading_calendar', role='input', execute=True)['ref']
    with duckdb.connect(str(target)) as connection:
        declare_market_db(connection, domain, allowed_slot_providers=('jq_ashare_local',))
    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({
        'input_db': str(target), 'inputs': {'trading_calendar': version},
    }))
    provider = SlotDataProvider()
    try:
        if domain == 'us_equity':
            with pytest.raises(MarketDbProfileError, match='MARKET_DB_DOMAIN_MISMATCH'):
                provider.resolve_calendar(None, None)
        else:
            assert provider.resolve_calendar(None, None).sessions().strftime('%Y-%m-%d').tolist() == [
                '2026-03-09', '2026-03-10',
            ]
        assert provider._owned_conn is None
    finally:
        provider.close()


@pytest.mark.parametrize('has_live_calendar', [False, True])
def test_vectorbt_implicit_frozen_input_requires_exact_calendar_view(tmp_path, monkeypatch, has_live_calendar):
    from calendar_test_support import seed_calendar
    from factor_synthesis.vectorbt_adapter.slot_provider import SlotDataProvider
    from utils.data_store import migrate_table

    source, target = tmp_path / 'source.duckdb', tmp_path / 'input.duckdb'
    with duckdb.connect(str(source)) as connection:
        connection.execute('CREATE TABLE samples AS SELECT 1 AS value')
        if has_live_calendar:
            seed_calendar(connection, start='2026-03-09', end='2026-03-10')
    sample = migrate_table(source, target, 'samples', role='input', execute=True)['ref']
    if has_live_calendar:
        migrate_table(source, target, 'trading_calendar', role='input', execute=True)
    with duckdb.connect(str(target)) as connection:
        declare_market_db(connection, 'cn_equity', calendar_coverage='2026-03-09..2026-03-10')
        if has_live_calendar:
            assert connection.execute('SELECT count(*) FROM trading_calendar').fetchone() == (2,)
    monkeypatch.setenv('QUANT_DATA_BINDINGS', json.dumps({
        'input_db': str(target), 'inputs': {'samples': sample},
    }))
    provider = SlotDataProvider()
    try:
        with pytest.raises(duckdb.CatalogException, match='trading_calendar'):
            provider.resolve_calendar(None, None)
    finally:
        provider.close()


def test_factor_base_init_does_not_evaluate_calendar_hash_or_load_profile(tmp_path, monkeypatch):
    from utils.factor_base import FactorBase
    from utils.factor_core.slot_binding import AShareFactorCalendar, AShareFactorFeed

    path = tmp_path / 'factor-input.duckdb'
    with duckdb.connect(str(path)) as connection:
        connection.execute("""
            CREATE TABLE price_data_daily AS SELECT '000001.XSHG' AS stock_code, DATE '2026-03-09' AS date,
                10. AS "close"
        """)

    calls = {'calendar_hash': 0, 'calendar_profile': 0}

    @property
    def fail_calendar_hash(self) -> str:
        calls['calendar_hash'] += 1
        raise AssertionError('calendar_hash must not be evaluated during FactorBase.__init__')

    def fail_calendar_profile(self):
        calls['calendar_profile'] += 1
        raise AssertionError('calendar_profile must not be evaluated during FactorBase.__init__')

    monkeypatch.setattr(AShareFactorCalendar, 'calendar_hash', fail_calendar_hash)
    monkeypatch.setattr(AShareFactorFeed, 'calendar_profile', fail_calendar_profile)

    factor = FactorBase(None, 'fixture', '2026-03-09', '2026-03-09', stock_pool=['000001.XSHG'],
                        duckdb_path=str(path), data_bindings={}, pit_enabled=False)
    try:
        assert factor._duckdb_adapter._conn is None
        assert calls == {'calendar_hash': 0, 'calendar_profile': 0}
    finally:
        factor.close()
