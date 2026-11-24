"""Compare S7 import and quality evidence with the scalar pre-optimization rules."""
import json
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from backtest_and_analysis.time_series.contracts import ContractValidationError
from data_pipeline.loaders.market_bar_importer import import_market_bars
from market_db_samples import crypto_profile, declare_market_db
from utils import bar_builder, bar_quality
from utils.bars import builder, quality
from utils.market_db.storage import _create_profile_table
from utils.session_calendar import CalendarCoverageError, SessionCalendar
from utils.time_axis import BarSchedule


def _reference_scan(connection, instrument, schedule, bars, **_):
    for day in schedule.calendar.sessions().strftime('%Y-%m-%d'):
        slots = schedule.for_session(day)
        if not slots.empty:
            quality.write_quality(connection, schedule.frequency.canonical, instrument, day, slots, bars)
    return len(schedule.calendar.sessions())


def _scalar_dates(calendar, timestamps):
    owners = [calendar.session_at(ts) for ts in timestamps]
    return pd.Index([owner.session_date if owner else None for owner in owners], dtype=object)


def _scalar_reference(patch):
    patch.setattr(SessionCalendar, 'session_dates_at', _scalar_dates, raising=False)

    def scan_full_history(connection, instrument, schedule, _bars, **kwargs):
        observed = connection.execute(
            f'SELECT * FROM {schedule.frequency.table_name} WHERE instrument_id=?', [instrument],
        ).fetchdf()
        observed['timestamp_utc'] = pd.to_datetime(observed.timestamp_utc, utc=True)
        return _reference_scan(connection, instrument, schedule, observed, **kwargs)

    patch.setattr(bar_quality, 'scan_quality', scan_full_history)
    # The old builder issues one quality upsert per Session.
    if hasattr(quality, 'write_quality_rows'):
        def write_one_at_a_time(connection, rows):
            for row in rows:
                connection.execute('INSERT OR REPLACE INTO bar_quality VALUES (?,?,?,?,?,?,?,?,?)', row)

        patch.setattr(builder, 'write_quality_rows', write_one_at_a_time)


def _snapshot(connection, *frequencies):
    tables = ('bar_quality', 'bar_catalog', 'instrument_metadata',
              *(f'market_bars_{freq}' for freq in frequencies))
    if len(frequencies) > 1:
        tables += ('bar_build_log',)
    return {name: connection.execute(f'SELECT * FROM {name} ORDER BY ALL').fetchdf() for name in tables}


def _assert_snapshot_equal(left, right):
    assert left.keys() == right.keys()
    for name, frame in left.items():
        pd.testing.assert_frame_equal(frame, right[name], check_exact=True, obj=name)


def _bars(instrument, timestamps, frequency):
    return pd.DataFrame(dict(
        instrument_id=instrument, timestamp_utc=timestamps, frequency=frequency,
        open=10., high=11., low=9., close=10., volume=1., source='synthetic', data_version='v1',
    ))


def _meta(instrument, **overrides):
    return dict(instrument_id=instrument, asset_class='equity', exchange='XNYS',
                timezone='America/New_York', calendar_id='XNYS',
                rth_reference_calendar='XNYS', price_adjustment='none', **overrides)


def test_xnys_missing_session_and_call_local_grid_reuse(monkeypatch):
    schedule = BarSchedule(SessionCalendar('XNYS', '2026-03-05', '2026-03-09'), '1d')
    opens = [schedule.for_session(day).start_utc.item() for day in ('2026-03-05', '2026-03-09')]
    symbols = ['AAA.XNYS', 'BBB.XNYS']
    bars = pd.concat([_bars(symbol, opens, '1d') for symbol in symbols], ignore_index=True)
    meta = pd.DataFrame([_meta(symbol) for symbol in symbols])
    monkeypatch.setattr(bar_builder, 'code_version', lambda: 'test-revision')

    def run(reference):
        with duckdb.connect() as connection, monkeypatch.context() as patch:
            declare_market_db(connection, 'us_equity')
            if reference:
                _scalar_reference(patch)
            else:
                visit = BarSchedule.for_session
                visits = []

                def count_grid(self, day):
                    visits.append(day)
                    return visit(self, day)

                patch.setattr(BarSchedule, 'for_session', count_grid)
                assert import_market_bars(connection, bars, meta, check_only=True,
                                          legacy_compatibility=False) == 4
                assert visits == ['2026-03-05', '2026-03-06', '2026-03-09']
                visits.clear()
            assert import_market_bars(connection, bars, meta, legacy_compatibility=False) == 4
            if not reference:
                assert visits == ['2026-03-05', '2026-03-06', '2026-03-09']
            result = _snapshot(connection, '1d')
            gaps = result['bar_quality'].query("status == 'missing_session'")
            assert gaps.session_id.tolist() == ['2026-03-06'] * 2
            assert gaps.incomplete_buckets.tolist() == [0, 0]
            return result

    _assert_snapshot_equal(run(False), run(True))


def test_crypto_minute_derive_and_incremental_quality_parity(monkeypatch):
    instrument = 'BTC-USD.SPOT'
    template = dict(timezone='UTC', weekdays=list(range(7)),
                    segments=[dict(open='00:00', close='00:00', open_day_offset=0, close_day_offset=1)])
    meta = pd.DataFrame([_meta(instrument, custom_sessions=json.dumps(template), point_value=1.)]).assign(
        asset_class='crypto', exchange='SPOT', timezone='UTC',
        calendar_id='crypto_utc', rth_reference_calendar='crypto_utc',
    )
    times = pd.date_range('2026-01-03', periods=1441, freq='min', tz='UTC')
    bars = _bars(instrument, times, '1min')
    gap = 10
    monkeypatch.setattr(bar_builder, 'code_version', lambda: 'test-revision')
    monkeypatch.setattr(builder, 'code_version', lambda: 'test-revision')

    def run(reference):
        build_ids = iter(('day-one', 'day-two', 'day-one-filled'))
        with duckdb.connect() as connection, monkeypatch.context() as patch:
            _create_profile_table(connection, crypto_profile(frequencies=('1min', '1d')))
            patch.setattr(builder, 'uuid', SimpleNamespace(
                uuid4=lambda: SimpleNamespace(hex=next(build_ids))))
            if reference:
                _scalar_reference(patch)
            options = dict(derive=('1d',), legacy_compatibility=False)
            assert import_market_bars(connection, bars.drop(gap), meta, **options) == 1440
            initial = _snapshot(connection, '1min', '1d')
            assert initial['market_bars_1d'].is_complete.tolist() == [False, False]
            assert initial['bar_quality'].query("frequency == '1d'").incomplete_buckets.tolist() == [1, 1]
            assert import_market_bars(connection, bars.iloc[[gap]], meta, **options) == 1
            filled = _snapshot(connection, '1min', '1d')
            first_day = filled['bar_quality'].query("session_id == '2026-01-03'")
            assert first_day.status.tolist() == ['complete', 'complete']
            assert first_day.set_index('frequency').n_actual_bars.to_dict() == {'1min': 1440, '1d': 1}
            return initial, filled

    for optimized, reference in zip(run(False), run(True)):
        _assert_snapshot_equal(optimized, reference)


@pytest.mark.parametrize(('timestamp', 'message'), [
    ('2026-03-06 14:31Z', 'Imported timestamp is off the Bar schedule grid'),
    ('2026-03-06 21:00Z', 'Imported bars must belong to an open Session'),
    ('2026-03-07 14:30Z', 'Imported bars must belong to an open Session'),
])
def test_xnys_rejections_match_scalar(monkeypatch, timestamp, message):
    instrument = 'AAA.XNYS'
    bars = _bars(instrument, pd.DatetimeIndex([timestamp]), '1d')
    for reference in (False, True):
        with duckdb.connect() as connection, monkeypatch.context() as patch:
            declare_market_db(connection, 'us_equity')
            if reference:
                _scalar_reference(patch)
            for check_only in (True, False):
                with pytest.raises(ContractValidationError) as caught:
                    import_market_bars(connection, bars, pd.DataFrame([_meta(instrument)]),
                                       legacy_compatibility=False, check_only=check_only)
                assert str(caught.value) == message
            assert not connection.execute("SHOW TABLES").fetchdf().name.str.startswith('bar_').any()


def test_vector_ownership_matches_scalar_at_pauses_and_padded_boundaries():
    template = dict(timezone='UTC', weekdays=list(range(7)), segments=[
        dict(open='20:00', close='02:00', open_day_offset=-1),
        dict(open='03:00', close='16:00'),
    ])
    calendar = SessionCalendar('overnight', '2026-01-03', '2026-01-04', custom_sessions=template)
    timestamps = pd.to_datetime([
        '2026-01-02 20:00Z', '2026-01-03 01:59:59.999999999Z', '2026-01-03 02:00Z',
        '2026-01-03 03:00Z', '2026-01-03 16:00Z', '2026-01-03 20:00Z',
    ], format='ISO8601', utc=True)
    assert calendar.session_dates_at(timestamps).equals(_scalar_dates(calendar, timestamps))
    for outside in ('2026-01-04 20:00Z', '2026-01-01 02:30Z'):
        padded = timestamps.append(pd.DatetimeIndex([outside]))
        with pytest.raises(CalendarCoverageError) as expected:
            _scalar_dates(calendar, padded)
        with pytest.raises(CalendarCoverageError) as actual:
            calendar.session_dates_at(padded)
        assert str(expected.value) == str(actual.value)


@pytest.mark.parametrize('times, flags', [
    (['2026-03-05 14:30Z', '2026-03-05 14:30Z', '2026-03-06 14:30Z'], [True, False, None]),
    (['2026-03-05 14:30Z'], None),
    ([], None),
])
def test_quality_grouping_preserves_scalar_counts_for_raw_audit(times, flags):
    schedule = BarSchedule(SessionCalendar('XNYS', '2026-03-05', '2026-03-06'), '1d')
    bars = pd.DataFrame({'timestamp_utc': pd.DatetimeIndex(times)}) if times else pd.DataFrame(
        columns=['timestamp_utc'])
    if flags is not None:
        bars['is_complete'] = pd.array(flags, dtype='boolean')
    with duckdb.connect() as optimized, duckdb.connect() as reference:
        assert quality.scan_quality(optimized, 'AAA.XNYS', schedule, bars) == 2
        assert _reference_scan(reference, 'AAA.XNYS', schedule, bars) == 2
        query = 'SELECT * FROM bar_quality ORDER BY session_id'
        assert optimized.execute(query).fetchall() == reference.execute(query).fetchall()


def test_quality_batch_preserves_pre_epoch_submicrosecond_bounds(tmp_path):
    day = '1969-12-31'
    start = pd.Timestamp(f'{day}T14:30:00.000000999Z')
    end = pd.Timestamp(f'{day}T21:00:00.000000999Z')
    overrides = tmp_path / 'overrides.json'
    overrides.write_text(json.dumps({'schema_version': 1, 'calendars': {'XNYS': {
        'special_sessions': [{'session_date': day, 'source': 'synthetic nanosecond boundaries',
                              'segments': [[start.isoformat(), end.isoformat()]]}],
    }}}))
    schedule = BarSchedule(SessionCalendar('XNYS', day, day, overrides_path=overrides), '1d')
    bars = pd.DataFrame({'timestamp_utc': [start], 'is_complete': [True]})
    query = 'SELECT * FROM bar_quality'
    with duckdb.connect() as batch, duckdb.connect() as scalar:
        assert _reference_scan(scalar, 'AAA.XNYS', schedule, bars) == 1
        expected = scalar.execute(query).fetchall()
        assert expected[0][5] == 'complete'
        assert expected[0][-2:] == (pd.Timestamp(f'{day}T14:30Z'), pd.Timestamp(f'{day}T21:00Z'))
        assert quality.scan_quality(batch, 'AAA.XNYS', schedule, bars) == 1
        assert batch.execute(query).fetchall() == expected
        quality.write_quality_rows(batch, [
            quality.quality_row('1d', 'AAA.XNYS', day, schedule.for_session(day), bars),
        ])
        assert batch.execute(query).fetchall() == expected
