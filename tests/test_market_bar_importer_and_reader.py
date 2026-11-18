"""Synthetic, isolated minute data integration; never opens the real data store."""
from dataclasses import replace

import duckdb
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series.contracts import (
    ContractValidationError, invalid_price_mask, validate_bar_frame,
)
from backtest_and_analysis.time_series.csv_reader import BarAvailability
from backtest_and_analysis.time_series.labels import BarReturnLabeler
from data_pipeline.loaders.market_bar_importer import (
    CONVERSION_MARKER, ExternalBarSourceSpec, adapt_external_bars,
    import_market_bars, validate_instrument_metadata,
)
from utils.data_revision import connection_revision
from utils.market_bar_reader import MarketBarReader, read_market_bars
from utils.session_calendar import SessionCalendar


def metadata(instrument='NQZ26', **overrides):
    future = instrument.startswith('NQ')
    row = dict(instrument_id=instrument, asset_class='futures' if future else 'equity',
               exchange='CME' if future else 'XSHG',
               timezone='America/Chicago' if future else 'Asia/Shanghai',
               contract_expiry='2026-12' if future else None, price_adjustment='none',
               point_value=20.0 if future else 1.0,
               rth_reference_calendar='NYSE' if future else 'XSHG')
    return pd.DataFrame([{**row, **overrides}])


def bars(instrument='NQZ26', times=('2026-12-01 14:30Z',), frequency='1min', **extra):
    return pd.DataFrame(dict(instrument_id=instrument, timestamp_utc=pd.to_datetime(list(times), utc=True),
                             frequency=frequency, open=100., high=102., low=99., close=101.,
                             volume=10., source='synthetic', data_version='v1', **extra))


def external(times=('2026-12-01 09:31', '2026-12-01 09:32'), volumes=(10., 25.)):
    return pd.DataFrame(dict(Date_Time=list(times), Open=100., High=102., Low=99., Close=101.,
                             TotalVolume=list(volumes)))


def spec(**overrides):
    return ExternalBarSourceSpec(**{**dict(
        timestamp_anchor='bar_end', source_timezone='America/New_York', volume_mode='interval',
        column_mapping={'Date_Time': 'timestamp', 'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'TotalVolume': 'volume'}), **overrides})


def adapt(data, source_spec=None, **kwargs):
    return adapt_external_bars(data, source_spec or spec(), metadata=metadata(), frequency='1min',
                               source='IQFeed', data_version='test-v1', instrument_id='NQZ26', **kwargs)


@pytest.fixture
def conn(tmp_path):
    with duckdb.connect(str(tmp_path / 'isolated.duckdb')) as connection:
        yield connection


def test_atomic_import_schema_revision_and_multimarket_frequencies(conn):
    meta = pd.concat([metadata(), metadata('600001.XSHG'), metadata('600002.XSHG')], ignore_index=True)
    frames = [bars(instrument, times, frequency) for instrument, times in [
        ('NQZ26', ['2026-09-30 13:30Z', '2026-09-30 13:35Z']),
        ('600001.XSHG', ['2026-09-30 01:30Z', '2026-09-30 01:35Z']),
        ('600002.XSHG', ['2026-09-30 01:30Z', '2026-09-30 01:35Z'])]
        for frequency in ('1min', '5min')]
    before = connection_revision(conn)
    assert import_market_bars(conn, pd.concat(frames, ignore_index=True), meta) == 12
    revision = connection_revision(conn)
    assert revision != before
    assert conn.execute('SELECT count(*) FROM instrument_metadata').fetchone()[0] == 3
    reader = MarketBarReader(conn)
    ids = list(meta.instrument_id)
    for frequency in ('1min', '5min'):
        result = reader.read_bars(ids, frequency)
        assert len(result) == 6
        assert str(result.timestamp_utc.dtype) == 'datetime64[ns, UTC]'
        assert set(result.frequency) == {frequency}
        assert len(result.attrs['calendar_hash']) == 3
        assert set(reader.get_calendars(result)) == set(ids)
        factors = result[['instrument_id', 'timestamp_utc']].drop_duplicates().assign(factor_value=1.)
        labels = BarReturnLabeler(reader.get_calendars(result), horizons=[1]).label(result, factors)
        assert len(labels) == 6
        snapshots = reader.build_availability(result)
        for row in result.itertuples():
            assert snapshots[row.instrument_id].status(row.instrument_id, row.timestamp_utc, frequency) == 'observed'
    import_market_bars(conn, bars(times=['2026-12-01 14:31Z']), metadata())
    assert connection_revision(conn) != revision


def test_transaction_failure_rolls_back_metadata_bars_schema_and_revision(conn, monkeypatch):
    # Fail during commit after both data inserts, including first-time DDL.
    import data_pipeline.loaders._transactions as transactions
    original = transactions.advance_revision
    def fail(connection):
        original(connection)
        raise RuntimeError('synthetic revision failure')
    before = connection_revision(conn)
    with monkeypatch.context() as patch:
        patch.setattr(transactions, 'advance_revision', fail)
        with pytest.raises(RuntimeError, match='revision failure'):
            import_market_bars(conn, bars(), metadata())
    assert conn.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name IN "
                        "('market_bars', 'instrument_metadata')").fetchone()[0] == 0
    assert connection_revision(conn) == before
    import_market_bars(conn, bars(), metadata())
    revision = connection_revision(conn)
    incoming = pd.concat([bars(), bars('600001.XSHG', ['2026-09-30 01:30Z'])], ignore_index=True)
    with pytest.raises(duckdb.ConstraintException):
        import_market_bars(conn, incoming, pd.concat([metadata(), metadata('600001.XSHG')]))
    assert conn.execute('SELECT count(*) FROM market_bars').fetchone()[0] == 1
    assert conn.execute('SELECT count(*) FROM instrument_metadata').fetchone()[0] == 1
    assert connection_revision(conn) == revision


def test_import_preserves_invalid_price_evidence(conn):
    frame = bars(times=['2026-12-01 14:30Z', '2026-12-01 14:31Z'])
    frame.loc[0, 'open'] = np.nan
    frame.loc[1, 'close'] = -1.
    import_market_bars(conn, frame, metadata())
    result = MarketBarReader(conn).read_bars(['NQZ26'], '1min')
    assert len(result) == 2
    assert invalid_price_mask(result).tolist() == [True, True]
    assert conn.execute('SELECT count(*) FROM market_bars WHERE open IS NULL').fetchone()[0] == 0


def test_optional_null_factor_and_money_are_accepted(conn):
    frame = bars(factor=None, money=None)
    assert import_market_bars(conn, frame, metadata()) == 1
    result = MarketBarReader(conn).read_bars(['NQZ26'], '1min')
    assert result.factor.isna().all() and result.money.isna().all()


@pytest.mark.parametrize('change', ['duplicate', 'naive', 'non_utc', 'futures_factor', 'conflict'])
def test_invalid_import_leaves_existing_data_and_revision_unchanged(conn, change):
    import_market_bars(conn, bars(), metadata())
    revision = connection_revision(conn)
    frame = bars(times=['2026-12-01 14:31Z'])
    meta = metadata()
    if change == 'duplicate':
        frame = pd.concat([frame, frame])
    elif change == 'naive':
        frame['timestamp_utc'] = frame.timestamp_utc.dt.tz_localize(None)
    elif change == 'non_utc':
        frame['timestamp_utc'] = frame.timestamp_utc.dt.tz_convert('America/New_York')
    elif change == 'futures_factor':
        frame['factor'] = 2.
    else:
        meta['point_value'] = 50.
    with pytest.raises(ContractValidationError):
        import_market_bars(conn, frame, meta)
    assert conn.execute('SELECT count(*) FROM market_bars').fetchone()[0] == 1
    assert connection_revision(conn) == revision


@pytest.mark.parametrize('override', [dict(contract_expiry=None), dict(contract_expiry='CONT'),
    dict(contract_expiry='2026-13'), dict(instrument_id='NQ_CONT'), dict(instrument_id='NQ_MAIN'),
    dict(instrument_id='NQ888'), dict(instrument_id='NQ000'), dict(price_adjustment='adjusted'),
    dict(point_value=0), dict(point_value=-1), dict(point_value=np.inf), dict(point_value=True),
    dict(point_value='20'), dict(timezone='bad/timezone'), dict(rth_reference_calendar='unknown'),
    dict(exchange='unknown'), dict(asset_class=None), dict(contract_expiry=202612)])
def test_invalid_metadata_rejected(override):
    with pytest.raises(ContractValidationError):
        validate_instrument_metadata(metadata(**override))


@pytest.mark.parametrize('column', ['instrument_id', 'asset_class', 'exchange', 'timezone',
                                   'price_adjustment', 'rth_reference_calendar'])
def test_required_metadata_and_equity_point_value_default(column):
    with pytest.raises(ContractValidationError):
        validate_instrument_metadata(metadata().drop(columns=column))
    equity = validate_instrument_metadata(metadata('000001.XSHE').drop(columns=['point_value']))
    assert equity.point_value.iloc[0] == 1.


@pytest.mark.parametrize('field', ['timestamp_anchor', 'source_timezone', 'volume_mode', 'column_mapping'])
def test_missing_external_semantics_rejected(field):
    with pytest.raises(ContractValidationError, match=field):
        spec(**{field: None})


def test_iqfeed_timezone_end_anchor_start_anchor_csv_and_no_repeat(tmp_path):
    data = external()
    original = data.copy(deep=True)
    result = adapt(data)
    assert result.timestamp_utc.tolist() == list(pd.to_datetime(['2026-12-01 14:30Z', '2026-12-01 14:31Z']))
    assert result.volume.tolist() == [10., 25.]
    assert result.attrs['timestamp_anchor'] == 'bar_start'
    pd.testing.assert_frame_equal(data, original)
    with pytest.raises(ContractValidationError, match='already converted'):
        adapt(result)
    marked = data.copy()
    marked.attrs[CONVERSION_MARKER] = True
    with pytest.raises(ContractValidationError, match='already converted'):
        adapt(marked)
    with pytest.raises(ContractValidationError, match='already converted'):
        replace(spec(), conversion_applied=True)
    path = tmp_path / 'iqfeed.csv'
    data.to_csv(path, index=False)
    pd.testing.assert_frame_equal(adapt(path), result)
    utc = external(['2026-12-01 14:30Z', '2026-12-01 14:31+00:00'])
    converted = adapt(utc, spec(timestamp_anchor='bar_start', source_timezone='UTC'))
    assert converted.timestamp_utc.tolist() == result.timestamp_utc.tolist()


@pytest.mark.parametrize('timezone,timestamp', [
    ('UTC', '2026-12-01 14:30'), ('UTC', '2026-12-01 09:30-05:00'),
    ('America/New_York', '2026-12-01 09:30+00:00'),
    ('America/New_York', '2026-03-08 02:30'),
    ('America/New_York', '2026-11-01 01:30'), ('America/New_York', None)])
def test_incompatible_naive_dst_or_missing_timestamp_rejected(timezone, timestamp):
    with pytest.raises(ContractValidationError, match='timestamp'):
        adapt(external([timestamp], [1.]), spec(source_timezone=timezone))


def test_aware_offsets_follow_dst():
    result = adapt(external(['2026-07-07 09:31-04:00', '2026-12-01 09:31-05:00']))
    assert result.timestamp_utc.tolist() == list(pd.to_datetime(['2026-07-07 13:30Z', '2026-12-01 14:30Z']))


def test_csv_mapping_preserves_leading_zero_instrument_ids(tmp_path):
    path = tmp_path / 'equity.csv'
    external(['2026-09-30 09:31'], [10.]).assign(Symbol='000001').to_csv(path, index=False)
    mapped = spec(source_timezone='Asia/Shanghai', column_mapping={**spec().column_mapping,
                                                                 'Symbol': 'instrument_id'})
    result = adapt_external_bars(path, mapped, metadata=metadata('000001'), frequency='1min',
                                 source='external', data_version='v1')
    assert result.instrument_id.tolist() == ['000001']


@pytest.mark.parametrize('mapping', [
    {'Date_Time': 'timestamp'},
    {'Date_Time': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low',
     'Close': 'close', 'TotalVolume': 'volume', 'Missing': 'money'},
    {'Date_Time': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low',
     'Close': 'close', 'TotalVolume': 'volume', 'other': 'close'}])
def test_incomplete_or_duplicate_column_mapping_rejected(mapping):
    with pytest.raises(ContractValidationError, match='column_mapping'):
        adapt(external().assign(other=1.), spec(column_mapping=mapping))


def test_cumulative_volume_resets_by_calendar_session_and_instrument():
    times = ['2026-11-30 18:01', '2026-11-30 18:02', '2026-12-01 09:31',
             '2026-12-01 18:01', '2026-12-01 18:02']
    # A long missing interval within the first session cannot be called one minute volume.
    with pytest.raises(ContractValidationError, match='missing bars'):
        adapt(external(times, [10., 15., 25., 3., 7.]), spec(volume_mode='cumulative_session'))
    data = external([times[i] for i in (0, 1, 3, 4)], [10., 15., 3., 7.])
    result = adapt(data.iloc[::-1], spec(volume_mode='cumulative_session'))
    assert result.volume.tolist() == [10., 5., 3., 4.]
    data['Symbol'] = 'NQZ26'
    other = data.assign(Symbol='NQH27', TotalVolume=[100., 150., 30., 70.])
    mapped = spec(volume_mode='cumulative_session', column_mapping={**spec().column_mapping,
                                                                   'Symbol': 'instrument_id'})
    result = adapt_external_bars(pd.concat([data, other]), mapped,
        metadata=pd.concat([metadata(), metadata('NQH27', contract_expiry='2027-03')]),
        frequency='1min', source='IQFeed', data_version='test')
    assert result.groupby('instrument_id').volume.apply(list).to_dict() == {
        'NQZ26': [10., 5., 3., 4.], 'NQH27': [100., 50., 30., 40.]}


@pytest.mark.parametrize('volumes', [(10., 9.), (10., np.nan), (10., np.inf), (10., -1.)])
def test_bad_cumulative_volume_rejected(volumes):
    with pytest.raises(ContractValidationError, match='volume'):
        adapt(external(volumes=volumes), spec(volume_mode='cumulative_session'))


def test_rth_raw_coverage_labels_availability_and_bounds(conn):
    times = ['2026-11-30 23:00Z', '2026-12-01 14:30Z', '2026-12-01 20:58Z',
             '2026-12-01 20:59Z', '2026-12-01 21:59Z', '2026-12-01 23:00Z',
             '2026-12-02 14:30Z']
    import_market_bars(conn, bars(times=times), metadata())
    reader = MarketBarReader(conn)
    eth = reader.read_bars(['NQZ26'], '1min')
    rth = reader.read_bars(['NQZ26'], '1min', view='RTH')
    assert len(eth) == 7 and len(rth) == 4
    assert rth.attrs['raw_coverage_by_key'] == eth.attrs['raw_coverage_by_key']
    assert rth.attrs['raw_coverage_end_utc'] == pd.Timestamp(times[-1])
    assert rth.attrs['raw_coverage_start_utc'] == pd.Timestamp(times[0])
    calendars = reader.get_calendars(rth)
    labels = BarReturnLabeler(calendars, view='RTH', horizons=[1]).label(rth.assign(factor_value=1.))
    reasons = labels.set_index('signal_timestamp').exclusion_reason.to_dict()
    assert reasons[pd.Timestamp('2026-12-01 20:59Z')] == 'session_boundary'
    assert reasons[pd.Timestamp('2026-12-02 14:30Z')] == 'insufficient_history'
    availability = reader.build_availability(rth, view='RTH')
    assert isinstance(availability, BarAvailability)
    assert availability.status('NQZ26', pd.Timestamp('2026-12-01 21:00Z'), '1min') == 'market_closed'
    assert availability.status('NQZ26', pd.Timestamp('2026-12-01 14:31Z'), '1min') == 'missing_bar'
    bounded = reader.read_bars(['NQZ26'], '1min', start_utc='2026-12-01 14:30Z', end_utc='2026-12-01 21:59Z')
    assert len(bounded) == 4
    empty = reader.read_bars(['NQZ26'], '5min')
    assert empty.empty
    assert empty.attrs['raw_coverage_by_key'] == {}
    with pytest.raises(ContractValidationError, match='UTC'):
        reader.read_bars(['NQZ26'], '1min', start_utc='2026-12-01')
    with pytest.raises(ContractValidationError, match='calendar'):
        wrong = SessionCalendar('NYSE', '2026-11-29', '2026-12-10')
        BarReturnLabeler(wrong, view='RTH', horizons=[1]).label(rth.assign(factor_value=1.))


def test_reader_file_is_read_only_and_borrowed_connection_survives(tmp_path):
    path = tmp_path / 'isolated.duckdb'
    with duckdb.connect(str(path)) as writer:
        import_market_bars(writer, bars(), metadata())
        with MarketBarReader(writer) as reader:
            assert len(reader.read_bars(['NQZ26'], '1min')) == 1
        assert writer.execute('SELECT 1').fetchone()[0] == 1
    result = read_market_bars(['NQZ26'], '1min', duckdb_path=path)
    assert len(result) == 1
    with MarketBarReader(duckdb_path=path) as reader:
        with pytest.raises(duckdb.InvalidInputException, match='read-only'):
            reader.connection.execute('CREATE TABLE forbidden (x INT)')
    with pytest.raises(ValueError, match='exactly one'):
        MarketBarReader()


def test_legacy_5m_and_daily_read_only_schema_data_revision_unchanged(conn):
    for table, date_type, clock in [('price_data_5m', 'TIMESTAMP', '2026-09-30 09:35:00'),
                                   ('price_data_daily', 'DATE', '2026-09-30')]:
        conn.execute(f'''CREATE TABLE {table} (stock_code VARCHAR, date {date_type},
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, money DOUBLE, factor DOUBLE)''')
        conn.execute(f'INSERT INTO {table} VALUES (?, ?, 100, 102, 99, 101, 10, 1000, 2)',
                     ['600001.XSHG', clock])
    before_tables = conn.execute('SHOW TABLES').fetchall()
    before_schema = {table: conn.execute(f'DESCRIBE {table}').fetchall() for table, in before_tables}
    before_data = {table: conn.execute(f'SELECT * FROM {table}').fetchall() for table, in before_tables}
    revision = connection_revision(conn)
    reader = MarketBarReader(conn)
    minute = reader.read_bars(['600001.XSHG'], '5min', source_table='price_data_5m')
    assert reader.build_availability(minute).status(
        '600001.XSHG', pd.Timestamp('2026-09-30 01:30Z'), '5min') == 'observed'
    daily = reader.read_bars(['600001.XSHG'], '1d', source_table='price_data_daily')
    assert minute.timestamp_utc.iloc[0] == pd.Timestamp('2026-09-30 01:30Z')
    assert daily.timestamp_utc.iloc[0] == pd.Timestamp('2026-09-29 16:00Z')
    assert daily.attrs['daily_timestamp_semantics'] == 'local_midnight_day_identifier'
    for table, frame in [('price_data_5m', minute), ('price_data_daily', daily)]:
        assert len(validate_bar_frame(frame)) == 1
        assert frame.money.iloc[0] == 1000 and frame.factor.iloc[0] == 2
        assert frame.source.iloc[0] == f'legacy_{table}'
        assert conn.execute(f'DESCRIBE {table}').fetchall() == before_schema[table]
        assert conn.execute(f'SELECT * FROM {table}').fetchall() == before_data[table]
    shifted = reader.read_legacy_price_table('price_data_5m', ['600001.XSHG'],
        legacy_timestamp_anchor='bar_start', start_utc='2026-09-30 01:35Z', end_utc='2026-09-30 01:35Z')
    assert shifted.timestamp_utc.iloc[0] == pd.Timestamp('2026-09-30 01:35Z')
    assert reader.read_legacy_price_table('price_data_daily', ['missing']).empty
    assert conn.execute('SHOW TABLES').fetchall() == before_tables
    assert connection_revision(conn) == revision
