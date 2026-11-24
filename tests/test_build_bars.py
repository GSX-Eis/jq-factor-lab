import duckdb

from market_db_samples import declare_equity_bars, declare_futures_bars
import numpy as np
import pandas as pd
import pytest

from slots.jq_ashare_local.panel import ASharePanelReader
from data_pipeline.loaders.market_bar_importer import import_market_bars
from utils.bar_builder import build_bars
from utils.bar_source import BarSource
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule


def sample(calendar_id='XSHG', strategy='segment_anchored', days=('2026-01-06', '2026-01-08'), frequency='5min'):
    future = calendar_id != 'XSHG'
    root = calendar_id.split('_')[-1] if future else None
    instrument = f'{root}2602.{calendar_id.split("_")[0]}' if future else '000001.XSHG'
    exchange = calendar_id.split('_')[0]
    reference = exchange + '_Day' if future else 'XSHG'
    cal = SessionCalendar(calendar_id, days[0], days[-1], rth_reference_calendar=reference)
    schedule = BarSchedule(cal, frequency, bucket_strategy=strategy)
    grids = [schedule.for_session(d.strftime('%Y-%m-%d'))
             for d in cal.sessions()]
    times = pd.concat(grids).start_utc.reset_index(drop=True)
    x = np.arange(len(times), dtype=float) + 100
    bars = pd.DataFrame(dict(instrument_id=instrument, timestamp_utc=times, frequency=frequency,
        open=x, high=x+2, low=x-1, close=x+1, volume=2., money=20., factor=1.,
        source='synthetic', data_version='v1'))
    meta = pd.DataFrame([dict(instrument_id=instrument, asset_class='futures' if future else 'equity',
        exchange=exchange, timezone='Asia/Shanghai', contract_expiry='2026-02' if future else None,
        price_adjustment='none', point_value=1., calendar_id=calendar_id,
        product_root=root, rth_reference_calendar=reference)])
    return bars, meta, cal


@pytest.mark.parametrize('calendar_id,strategy', [('XSHG','segment_anchored'),
    ('SHFE_CU','segment_anchored'), ('SHFE_CU','session_clock'),
    ('DCE_M','segment_anchored'), ('DCE_M','session_clock')])
@pytest.mark.slow
def test_sql_ohlcv_and_incremental_build(tmp_path, calendar_id, strategy):
    bars, meta, cal = sample(calendar_id, strategy)
    bars.loc[0, 'factor'] = np.nan
    instrument = meta.instrument_id.iloc[0]
    with duckdb.connect(str(tmp_path / 'bars.duckdb'), config={'threads': 1}) as conn:
        if calendar_id == 'XSHG':
            declare_equity_bars(conn)
        else:
            declare_futures_bars(conn, meta.exchange.item())
        import_market_bars(conn, bars, meta, bucket_strategy=strategy)
        args = dict(source='5min', targets=['15min', '60min', '1d'], instruments=[instrument],
                    start='2026-01-06', end='2026-01-08', bucket_strategy=strategy)
        first = build_bars(conn, **args)
        assert first['rebuilt_sessions'] == 9
        for target in args['targets']:
            result = BarSource(conn).read([instrument], target, args['start'], args['end'], 'ETH')
            expected = []
            bar_schedule = BarSchedule(cal, target, bucket_strategy=strategy)
            for day in cal.sessions():
                slots = bar_schedule.for_session(day.strftime('%Y-%m-%d'))
                for slot in slots.itertuples():
                    group = bars.loc[bars.timestamp_utc.ge(slot.start_utc) & bars.timestamp_utc.lt(slot.end_utc)]
                    expected.append((group.open.iloc[0], group.high.max(), group.low.min(),
                                     group.close.iloc[-1], group.volume.sum(), group.money.sum(), len(group)))
            assert result[['open','high','low','close','volume','money','n_source_bars']].to_numpy() == pytest.approx(np.array(expected))
            assert result.is_complete.all()
            assert result.factor.eq(1.).all()
        logs = conn.execute('SELECT * FROM bar_build_log ORDER BY frequency, session_id').fetchall()
        assert build_bars(conn, **args)['rebuilt_sessions'] == 0
        assert conn.execute('SELECT * FROM bar_build_log ORDER BY frequency, session_id').fetchall() == logs
        conn.execute('UPDATE market_bars_5min SET close=close+0.5 WHERE timestamp_utc=?', [bars.timestamp_utc.iloc[0]])
        assert build_bars(conn, **args)['rebuilt_sessions'] == 3
        versions = conn.execute('SELECT session_id, version FROM bar_build_log').fetchall()
        assert all(v == (2 if day == '2026-01-06' else 1) for day,v in versions)


def test_native_daily_refused_and_check_only_never_writes(tmp_path):
    bars, meta, _ = sample(days=('2026-01-06',))
    with duckdb.connect(str(tmp_path / 'native.duckdb')) as conn:
        declare_equity_bars(conn)
        import_market_bars(conn, bars, meta)
        # Synthetic fixture only; the product code must never write these tables.
        conn.execute("CREATE TABLE price_data_daily AS SELECT '000001.XSHG' stock_code, DATE '2026-01-06' date, 100.0 open, 200.0 high, 90.0 low, 148.0 AS \"close\", 96.0 volume, 960.0 money, 1.0 factor")
        args = dict(source='5min', targets=['1d'], instruments=['000001.XSHG'], start='2026-01-06', end='2026-01-06')
        before = conn.execute('SHOW TABLES').fetchall()
        with pytest.raises(ValueError, match='native'):
            build_bars(conn, **args)
        report = build_bars(conn, **args, check_only=True)
        assert report['comparisons'][0]['close_difference'] == 0
        assert report['comparisons'][0]['volume_difference'] == 0
        assert report['comparisons'][0]['missing_days'] == []
        assert conn.execute('SHOW TABLES').fetchall() == before


def test_unaligned_boundaries_refused(tmp_path):
    bars, meta, _ = sample('SHFE_CU', 'session_clock', days=('2026-01-06',), frequency='30min')
    with duckdb.connect(str(tmp_path / 'bad.duckdb')) as conn:
        declare_futures_bars(conn, 'SHFE')
        import_market_bars(conn, bars, meta, bucket_strategy='session_clock')
        with pytest.raises(ValueError, match='boundar'):
            build_bars(conn, source='30min', targets=['60min'], instruments=list(meta.instrument_id),
                       start='2026-01-06', end='2026-01-06', bucket_strategy='segment_anchored')


def test_domestic_strategy_required_on_import_and_build(tmp_path):
    bars, meta, _ = sample('DCE_M', days=('2026-01-06',))
    with duckdb.connect(str(tmp_path / 'futures.duckdb')) as conn:
        declare_futures_bars(conn, 'DCE')
        with pytest.raises(ValueError, match='bucket_strategy'):
            import_market_bars(conn, bars, meta, legacy_compatibility=False)
        import_market_bars(conn, bars, meta, bucket_strategy='segment_anchored')
        with pytest.raises(ValueError, match='bucket_strategy'):
            build_bars(conn, source='5min', targets=['60min'], instruments=list(meta.instrument_id),
                       start='2026-01-06', end='2026-01-06')


def test_disjoint_builds_do_not_claim_unbuilt_sessions(tmp_path):
    from utils.bar_source import MissingBarDataError
    bars, meta, _ = sample()
    with duckdb.connect(str(tmp_path / 'disjoint.duckdb')) as conn:
        declare_equity_bars(conn)
        import_market_bars(conn, bars, meta)
        for day in ['2026-01-06', '2026-01-08']:
            build_bars(conn, source='5min', targets=['15min'], instruments=['000001.XSHG'], start=day, end=day)
        with pytest.raises(MissingBarDataError, match='Session coverage'):
            BarSource(conn).read(['000001.XSHG'], '15min', '2026-01-06', '2026-01-08', 'ETH')


def test_harness_build_audit_and_import_derive(tmp_path, capsys):
    import json
    from tools.agent_harness.cli import main
    bars, meta, _ = sample(days=('2026-01-06',))
    csv, metadata = tmp_path / 'bars.csv', tmp_path / 'instruments.json'
    bars.to_csv(csv, index=False)
    meta.to_json(metadata, orient='records')
    path = tmp_path / 'cli.duckdb'
    with duckdb.connect(str(path)) as connection:
        declare_equity_bars(connection)
    common = ['--db', str(path), '--output-root', str(tmp_path / 'reports')]
    assert main(['import-bars', *common, '--bars', str(csv), '--metadata', str(metadata),
                 '--derive', '15min']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['metrics']['imported_bars'] == 48
    assert main(['build-bars', *common, '--source', '5min', '--targets', '60min',
                 '--instruments', '000001.XSHG', '--start', '2026-01-06', '--end', '2026-01-06']) == 0
    assert json.loads(capsys.readouterr().out)['metrics']['rebuilt_sessions'] == 1
    assert main(['audit-bars', *common]) == 0
    assert json.loads(capsys.readouterr().out)['metrics']['audited_sessions'] == 3


def test_legacy_source_in_database_with_other_instrument_metadata(tmp_path):
    with duckdb.connect(str(tmp_path / 'mixed.duckdb')) as conn:
        declare_equity_bars(conn)
        conn.from_df(ASharePanelReader._legacy_metadata(['000002.XSHG'], 'Asia/Shanghai')).create('instrument_metadata')
        legacy = pd.DataFrame(dict(stock_code=['000001.XSHG'], date=pd.to_datetime(['2026-01-06 09:35']),
            open=[100.], high=[102.], low=[99.], close=[101.], volume=[2.]))
        conn.from_df(legacy).create('price_data_5m')
        result = build_bars(conn, source='5min', targets=['15min'], instruments=['000001.XSHG'],
                            start='2026-01-06', end='2026-01-06')
        assert result['rebuilt_sessions'] == 1
        assert conn.execute('SELECT n_source_bars, n_expected_bars FROM market_bars_15min').fetchone() == (1,3)
        from utils.bar_quality import BarDataQualityWarning
        with pytest.warns(BarDataQualityWarning):
            assert len(BarSource(conn).read(['000001.XSHG'], '15min', '2026-01-06', '2026-01-06', 'ETH')) == 1
