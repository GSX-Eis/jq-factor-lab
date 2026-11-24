"""Synthetic futures slot coverage with explicitly owned, in-memory databases."""
import duckdb

from market_db_samples import declare_futures_bars
import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.time_series.contracts import PRICE_COLUMNS
from data_pipeline.loaders.market_bar_importer import import_market_bars, validate_instrument_metadata
from futures_roll_samples import futures_sample, warm_market_calendars  # noqa: F401
from utils.bars.catalog import table_exists
from utils.bars.continuous import ContinuousBarSource
from utils.bars.futures_roll import build_roll_schedule
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule, Frequency


@pytest.mark.parametrize('root,calendar', [
    ('NQ', 'CME Globex Equity'), ('ES', 'CME Globex Equity'),
    ('GC', 'CMEGlobex_Gold'), ('CL', 'CMEGlobex_CL'),
])
def test_named_overseas_product_requires_explicit_calendars(root, calendar):
    row = dict(instrument_id=root+'M26', asset_class='futures', exchange='CME',
               timezone='America/Chicago', contract_expiry='2026-06-18', contract_month='2026-06',
               price_adjustment='none', point_value=20., product_root=root,
               calendar_id=calendar, rth_reference_calendar='NYSE')
    assert validate_instrument_metadata(pd.DataFrame([row])).product_root.item() == root
    for name in ('calendar_id', 'rth_reference_calendar'):
        with pytest.raises(ValueError, match=name):
            validate_instrument_metadata(pd.DataFrame([{**row, name: None}]))


def _populate(connection, *, root, exchange, start, end, expiries,
              calendar_id=None, reference=None, frequency='1d'):
    from utils.time_axis.futures import futures_calendar

    if calendar_id is None:
        mapping = futures_calendar(root)
        calendar_id, reference = mapping['calendar_id'], mapping['rth_reference_calendar']
    if not table_exists(connection, 'market_db_profile'):
        overrides = {'calendar_id': calendar_id} if exchange == 'CME' else {}
        declare_futures_bars(connection, exchange, **overrides)
    cal = SessionCalendar(calendar_id, start, end, rth_reference_calendar=reference)
    schedule = BarSchedule(cal, frequency, bucket_strategy='segment_anchored')
    contracts, bars = [], []
    for index, expiry in enumerate(expiries):
        instrument = f'{root}{expiry[2:7].replace("-", "")}.{exchange}'
        contracts.append(dict(instrument_id=instrument, asset_class='futures', exchange=exchange,
            timezone=cal.timezone, contract_expiry=expiry, contract_month=expiry[:7],
            price_adjustment='none', point_value=20., calendar_id=calendar_id if exchange == 'CME' else None,
            product_root=root, rth_reference_calendar=reference))
        for position, day in enumerate(cal.sessions()):
            slots = schedule.for_session(day)
            price = 100. + 20 * index + position * 2
            bars.append(slots[['session_id']].assign(timestamp_utc=slots.start_utc, instrument_id=instrument,
                frequency=frequency, open=price, high=price+2, low=price-1, close=price+1,
                open_interest=100. if position == 1-index else 300., volume=500. if index else 100.,
                source='synthetic', data_version='v1'))
    raw = pd.concat(bars, ignore_index=True)
    import_market_bars(connection, raw.drop(columns='session_id'), pd.DataFrame(contracts),
                       bucket_strategy='segment_anchored', legacy_compatibility=False)
    return raw, cal


def _prices(actual, expected):
    np.testing.assert_allclose(actual[list(PRICE_COLUMNS)], expected[list(PRICE_COLUMNS)],
                               rtol=0, atol=1e-12, equal_nan=True)


@pytest.mark.usefixtures('warm_market_calendars')
@pytest.mark.parametrize('root,exchange', [
    ('CU', 'SHFE'), ('SC', 'INE'), ('M', 'DCE'), ('SR', 'CZCE'), ('IF', 'CFFEX'), ('SI', 'GFEX'),
])
def test_domestic_max_oi_matches_legacy_across_six_exchanges(root, exchange):
    from slots.cn_futures import build_cn_futures_slot

    start, end = '2026-01-05', '2026-01-09'
    with duckdb.connect(config={'threads': 1}) as conn:
        raw, calendar = _populate(conn, root=root, exchange=exchange, start=start, end=end,
                                  expiries=['2026-02', '2026-03'])
        build_roll_schedule(conn, product_roots=[root], start='2026-01-06', end=end)
        rows = conn.execute('SELECT effective_session_id, decision_session_id, from_contract, to_contract, '
            'from_open_interest, to_open_interest FROM futures_roll_schedule ORDER BY 1').fetchall()
        early, late = f'{root}2602.{exchange}', f'{root}2603.{exchange}'
        assert rows == [
            ('2026-01-06', '2026-01-05', None, early, None, 300.),
            ('2026-01-07', '2026-01-06', early, late, 100., 300.),
            ('2026-01-08', '2026-01-07', late, late, 300., 300.),
            ('2026-01-09', '2026-01-08', late, late, 300., 300.),
        ]
        lo = calendar.session_on('2026-01-06').segments[0][0]
        hi = calendar.session_on(end).segments[-1][1]
        reader = ContinuousBarSource(conn)
        for mode, legacy in [('none', 'none'), ('ratio_back', 'ratio_back_adjust'),
                             ('panama_canal', 'diff_back_adjust')]:
            slot = build_cn_futures_slot(conn, product_root=root, start=start, end=end, adjustment=mode)
            signal = slot.bar_feed.read_signal_bars([root], Frequency.parse('1d'),
                                                    start_utc=lo, end_utc=hi, as_of_utc=hi)
            reference = reader.read(root, '1d', '2026-01-06', end, adjust=legacy)
            grid = calendar.bar_grid('1d', '2026-01-06', end)
            pd.testing.assert_frame_equal(signal[list(grid.columns)], grid)
            assert signal.session_id.tolist() == reference.session_id.tolist()
            assert signal.concrete_instrument_id.tolist() == reference.contract_id.tolist()
            _prices(signal, reference)

            execution = slot.bar_feed.read_execution_bars([root], '1d', start_utc=lo, end_utc=hi)
            pd.testing.assert_frame_equal(execution[list(grid.columns)], grid)
            expected = execution[['session_id', 'concrete_instrument_id']].merge(
                raw, left_on=['session_id', 'concrete_instrument_id'],
                right_on=['session_id', 'instrument_id'])
            _prices(execution, expected)
            assert execution.concrete_instrument_id.tolist() == [early, late, late, late]
            if mode == 'ratio_back':
                expected.loc[0, list(PRICE_COLUMNS)] *= 123. / 103.
            elif mode == 'panama_canal':
                expected.loc[0, list(PRICE_COLUMNS)] += 20.
            _prices(signal, expected)


@pytest.mark.usefixtures('warm_market_calendars')
@pytest.mark.parametrize('adjustment', ['none', 'ratio_back', 'panama_canal'])
def test_as_of_uses_closed_bars_and_effective_rolls_beyond_output_end(adjustment):
    from slots.cn_futures import build_cn_futures_slot

    with duckdb.connect(config={'threads': 1}) as conn:
        interests = [(300,200,100)] * 4 + [(100,400,200)] * 6
        futures_sample(conn, oi=interests)
        build_roll_schedule(conn, product_roots=['CU'], start='2026-01-06', end='2026-01-16')
        slot = build_cn_futures_slot(conn, product_root='CU', start='2026-01-05',
                                     end='2026-01-16', adjustment=adjustment)
        lo, end = pd.Timestamp('2026-01-05 13:00Z'), pd.Timestamp('2026-01-09 07:00Z')
        open_next = pd.Timestamp('2026-01-09 13:00Z')
        before_open = open_next - pd.Timedelta(minutes=1)

        def read(anchor, right=end):
            return slot.bar_feed.read_signal_bars(['CU'], '1d', start_utc=lo,
                                                  end_utc=right, as_of_utc=anchor)

        before = read(before_open)
        assert before.close.tolist() == [103., 105., 107., 109.]
        assert slot.universe.resolve_active_contract('CU', as_of_utc=before_open) == 'CU2602.SHFE'
        assert slot.universe.resolve_active_contract('CU', as_of_utc=open_next) == 'CU2603.SHFE'
        after = read(open_next)
        expected = {'none': before.close, 'ratio_back': before.close * (129. / 109.),
                    'panama_canal': before.close + 20.}[adjustment]
        np.testing.assert_allclose(after.close, expected, rtol=0, atol=1e-12, equal_nan=True)
        # Monday's daily bar is stored at Friday's open and publishes on Monday.
        pd.testing.assert_frame_equal(after, read(open_next, right=pd.Timestamp('2026-01-13 07:00Z')))
        conn.execute('UPDATE market_bars_1d SET close=777, open_interest=9999 WHERE timestamp_utc>=?',
                     [open_next])
        conn.execute("DELETE FROM futures_roll_schedule WHERE effective_session_id>='2026-01-12'")
        pd.testing.assert_frame_equal(before, read(before_open), check_exact=True)


def test_overseas_volume_roll_reads_eth_and_rth_across_dst():
    from slots.global_futures import build_global_futures_slot

    start, end = '2026-03-05', '2026-03-10'
    with duckdb.connect(config={'threads': 1}) as conn:
        _, calendar = _populate(conn, root='NQ', exchange='CME', start=start, end=end,
            expiries=['2026-03-20', '2026-06-18'], calendar_id='CME Globex Equity', reference='NYSE',
            frequency='30min')
        conn.execute("UPDATE market_bars_30min SET volume=600 WHERE instrument_id='NQ2603.CME' "
                     "AND timestamp_utc < '2026-03-06 22:00:00+00'")
        build_roll_schedule(conn, product_roots=['NQ'], start='2026-03-06', end=end, rule='max_volume')
        roll = conn.execute('SELECT decision_session_id, effective_session_id, to_contract, rule '
                            'FROM futures_roll_schedule ORDER BY effective_session_id').fetchall()
        assert roll == [('2026-03-05', '2026-03-06', 'NQ2603.CME', 'max_volume'),
                        ('2026-03-06', '2026-03-09', 'NQ2603.CME', 'max_volume'),
                        ('2026-03-09', '2026-03-10', 'NQ2606.CME', 'max_volume')]
        lo = calendar.session_on('2026-03-06').segments[0][0]
        hi = calendar.session_on(end).segments[-1][1]
        reads = {}
        for view in ('ETH', 'RTH'):
            slot = build_global_futures_slot(conn, product_root='NQ', start=start, end=end,
                                            view=view, adjustment='panama_canal', roll_policy='max_volume')
            dates = slot.calendar.bar_grid('30min', '2026-03-06', '2026-03-09', view=view)
            assert dates.session_id.nunique() == 2
            signal = slot.bar_feed.read_signal_bars(['NQ'], '30min', start_utc=lo, end_utc=hi, as_of_utc=hi)
            execution = slot.bar_feed.read_execution_bars(['NQ'], '30min', start_utc=lo, end_utc=hi)
            assert execution.groupby('session_id').concrete_instrument_id.first().to_dict() == {
                '2026-03-06': 'NQ2603.CME', '2026-03-09': 'NQ2603.CME', '2026-03-10': 'NQ2606.CME'}
            assert execution.loc[execution.session_id.eq('2026-03-09'), 'close'].eq(105.).all()
            assert signal.loc[signal.session_id.eq('2026-03-09'), 'close'].eq(125.).all()
            reads[view] = signal.groupby('session_id').timestamp_utc.first()
        assert reads['ETH']['2026-03-06'] == pd.Timestamp('2026-03-05 23:00Z')
        assert reads['ETH']['2026-03-09'] == pd.Timestamp('2026-03-08 22:00Z')
        assert reads['RTH']['2026-03-06'] == pd.Timestamp('2026-03-06 14:30Z')
        assert reads['RTH']['2026-03-09'] == pd.Timestamp('2026-03-09 13:30Z')

        future_open = calendar.session_on('2026-03-10').segments[0][0]
        anchor = future_open - pd.Timedelta(seconds=1)
        before = slot.bar_feed.read_signal_bars(['NQ'], '30min', start_utc=lo, end_utc=hi, as_of_utc=anchor)
        conn.execute('UPDATE market_bars_30min SET close=999 WHERE timestamp_utc>=?', [future_open])
        conn.execute("DELETE FROM futures_roll_schedule WHERE effective_session_id='2026-03-10'")
        after = slot.bar_feed.read_signal_bars(['NQ'], '30min', start_utc=lo, end_utc=hi, as_of_utc=anchor)
        pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_expiry_policy_and_contract_auxiliary_features_are_point_in_time():
    from slots.futures import DaysBeforeExpiry
    from slots.global_futures import build_global_futures_slot

    start, end = '2026-03-04', '2026-03-12'
    with duckdb.connect(config={'threads': 1}) as conn:
        raw, calendar = _populate(conn, root='ES', exchange='CME', start=start, end=end,
            expiries=['2026-03-10', '2026-06-18'], calendar_id='CME Globex Equity', reference='NYSE')
        policy = DaysBeforeExpiry(2)
        build_roll_schedule(conn, product_roots=['ES'], start='2026-03-05', end=end, rule=policy)
        rows = conn.execute('SELECT decision_session_id, effective_session_id, to_contract '
                            'FROM futures_roll_schedule ORDER BY effective_session_id').fetchall()
        assert rows[:2] == [('2026-03-04', '2026-03-05', 'ES2603.CME'),
                            ('2026-03-05', '2026-03-06', 'ES2606.CME')]
        assert all(decision < effective and contract == 'ES2606.CME' for decision, effective, contract in rows[1:])
        slot = build_global_futures_slot(conn, product_root='ES', start=start, end=end, roll_policy=policy,
                                         commission_bps=1.5, slippage_bps=2., exclude_delivery_month=True)
        spec = slot.universe.instrument_spec('ES2603.CME')
        assert (spec.currency, spec.contract_multiplier, str(spec.expiry_date)) == ('USD', 20., '2026-03-10')
        assert slot.universe.resolve_universe('ES', as_of_utc=calendar.session_on(end).segments[-1][1]) == (
            'ES2606.CME',)
        day = '2026-03-06'
        opened, closed = calendar.session_on(day).segments[0][0], calendar.session_on(day).segments[-1][1]
        grid = pd.DataFrame(dict(instrument_id='ES2603.CME', timestamp_utc=[opened, closed], session_id=day),
                            index=[5, 7])
        aux = slot.read_aux_features(grid, features=['open_interest', 'days_to_expiry'])
        assert aux.open_interest.tolist() == [100., 300.]
        assert aux.days_to_expiry.tolist() == [4, 4]
        assert (aux.available_at_utc <= aux.timestamp_utc).all()
        only_oi = slot.read_aux_features(grid, features=['open_interest'])
        assert only_oi.available_at_utc.tolist() == [calendar.session_on('2026-03-05').segments[-1][1], closed]
        assert slot.microstructure.tradability_mask(grid, instrument_id='ES2603.CME').tolist() == [False, False]
        assert slot.microstructure.tradability_mask(grid, instrument_id='ES2606.CME').tolist() == [True, True]
        profile = slot.microstructure.execution_profile
        assert (profile.entry_price_field, profile.exit_price_field, profile.allow_short,
                profile.commission_bps, profile.slippage_bps) == ('open', 'close', True, 1.5, 2.)
        expiry_day = '2026-03-10'
        grid = raw.loc[raw.session_id.eq(expiry_day) & raw.instrument_id.eq('ES2603.CME')]
        other = build_global_futures_slot(conn, product_root='ES', start=start, end=end)
        assert not other.microstructure.tradability_mask(grid, instrument_id='ES2603.CME').any()


def test_custom_policy_cannot_observe_an_effective_session():
    class FutureVolume:
        name = 'invalid'

        def select(self, inputs):
            return inputs.observe(inputs.effective_session, 'CU2602.SHFE', 'volume')

    with duckdb.connect(config={'threads': 1}) as conn:
        futures_sample(conn, start='2026-01-05', end='2026-01-06')
        with pytest.raises(ValueError, match='decision window'):
            build_roll_schedule(conn, product_roots=['CU'], start='2026-01-06', end='2026-01-06',
                                rule=FutureVolume())


@pytest.mark.usefixtures('warm_market_calendars')
def test_max_volume_confirmation_tie_and_no_backward_roll():
    with duckdb.connect(config={'threads': 1}) as conn:
        bars, _, _ = futures_sample(conn, start='2026-01-05', end='2026-01-12')
        timestamps = sorted(bars.timestamp_utc.unique())
        for timestamp, volumes in zip(timestamps, [
                (100, 100, 50), (50, 100, 25), (70, 100, 40), (90, 80, 30), (90, 80, 30)]):
            for contract, volume in zip(('CU2602.SHFE', 'CU2603.SHFE', 'CU2604.SHFE'), volumes):
                conn.execute('UPDATE market_bars_1d SET volume=? '
                             'WHERE timestamp_utc=? AND instrument_id=?', [volume, timestamp, contract])

        for forward_only, last in [(True, 'CU2603.SHFE'), (False, 'CU2602.SHFE')]:
            build_roll_schedule(conn, product_roots=['CU'], start='2026-01-07', end='2026-01-12',
                                rule='max_volume', confirm_days=2, forward_only=forward_only)
            rows = conn.execute('SELECT to_contract FROM futures_roll_schedule '
                                'ORDER BY effective_session_id').fetchall()
            assert rows == [('CU2602.SHFE',), ('CU2603.SHFE',), ('CU2603.SHFE',), (last,)]


def test_delivery_month_mask_uses_contract_month_not_expiry_month():
    from data_pipeline.loaders.market_bar_importer import store_instrument_metadata
    from slots.global_futures import build_global_futures_slot

    calendar = 'CMEGlobex_CL'
    contract = 'CL2604.CME'
    start, end = '2026-03-05', '2026-03-06'
    with duckdb.connect(config={'threads': 1}) as conn:
        declare_futures_bars(conn, 'CME', calendar_id=calendar)
        store_instrument_metadata(conn, pd.DataFrame([dict(
            instrument_id=contract, asset_class='futures', exchange='CME', timezone='America/Chicago',
            contract_expiry='2026-03-20', contract_month='2026-04', price_adjustment='none',
            point_value=1000., calendar_id=calendar, product_root='CL', rth_reference_calendar=calendar,
        )]))
        slot = build_global_futures_slot(conn, product_root='CL', start=start, end=end,
                                         exclude_delivery_month=True)
        grid = slot.calendar.bar_grid('1d')
        assert slot.microstructure.tradability_mask(grid, instrument_id=contract).all()


def test_review_max_volume_compares_whole_session_across_storage_frequencies():
    from utils.bars.source import BarSource

    with duckdb.connect(config={'threads': 1}) as conn:
        raw, _ = _populate(conn, root='NQ', exchange='CME', start='2026-03-05', end='2026-03-06',
            expiries=['2026-03-20', '2026-06-18'], calendar_id='CME Globex Equity', reference='NYSE',
            frequency='30min')
        conn.execute("UPDATE market_bars_30min SET volume=CASE WHEN instrument_id='NQ2603.CME' THEN 10 ELSE 9 END")
        raw['volume'] = raw.instrument_id.map({'NQ2603.CME': 10., 'NQ2606.CME': 9.})
        grouped = raw.groupby(['instrument_id', 'session_id'], sort=False)
        daily = grouped.first().assign(volume=grouped.volume.sum(), frequency='1d').reset_index()
        assert daily.loc[daily.session_id.eq('2026-03-05'), 'volume'].tolist() == [460., 414.]
        metadata = BarSource(conn).get_instrument_metadata(['NQ2603.CME', 'NQ2606.CME'])

        def leader():
            build_roll_schedule(conn, product_roots=['NQ'], start='2026-03-06', end='2026-03-06',
                                rule='max_volume')
            return conn.execute('SELECT to_contract FROM futures_roll_schedule').fetchone()[0]

        assert leader() == 'NQ2603.CME'
        # Redistribute the same near-month volume so its final bucket is smaller.
        near = raw.loc[raw.instrument_id.eq('NQ2603.CME') & raw.session_id.eq('2026-03-05'), 'timestamp_utc']
        for stamp, volume in [(near.iloc[0], 19.), (near.iloc[-1], 1.)]:
            conn.execute("UPDATE market_bars_30min SET volume=? WHERE instrument_id='NQ2603.CME' AND timestamp_utc=?",
                         [volume, stamp])
        assert leader() == 'NQ2603.CME'
        for contract in ('NQ2606.CME', 'NQ2603.CME'):
            import_market_bars(conn, daily.loc[daily.instrument_id.eq(contract)].drop(columns='session_id'),
                               metadata.loc[metadata.instrument_id.eq(contract)], legacy_compatibility=False)
            assert leader() == 'NQ2603.CME'
        conn.execute('DROP TABLE market_bars_30min')
        assert leader() == 'NQ2603.CME'


def test_review_max_volume_replaces_expiring_incumbent_without_waiting_for_confirmation():
    from slots.global_futures import build_global_futures_slot

    start, end = '2026-03-03', '2026-03-09'
    with duckdb.connect(config={'threads': 1}) as conn:
        bars, calendar = _populate(conn, root='NQ', exchange='CME', start=start, end=end,
            expiries=['2026-03-06', '2026-06-18', '2026-09-18'],
            calendar_id='CME Globex Equity', reference='NYSE')
        conn.execute("UPDATE market_bars_1d SET volume=1000 WHERE instrument_id='NQ2603.CME'")
        for day, volume in [('2026-03-04', 700), ('2026-03-05', 100), ('2026-03-06', 700)]:
            stamp = bars.loc[bars.session_id.eq(day), 'timestamp_utc'].iloc[0]
            conn.execute("UPDATE market_bars_1d SET volume=? WHERE instrument_id='NQ2609.CME' AND timestamp_utc=?",
                         [volume, stamp])
        build_roll_schedule(conn, product_roots=['NQ'], start='2026-03-05', end=end,
                            rule='max_volume', confirm_days=2)
        rows = conn.execute('SELECT effective_session_id, to_contract FROM futures_roll_schedule '
                            'ORDER BY effective_session_id').fetchall()
        assert rows == [('2026-03-05', 'NQ2603.CME'), ('2026-03-06', 'NQ2606.CME'), ('2026-03-09', 'NQ2606.CME')]
        slot = build_global_futures_slot(conn, product_root='NQ', start=start, end=end,
                                        roll_policy='max_volume')
        monday_open = calendar.session_on(end).segments[0][0]
        assert slot.universe.resolve_active_contract('NQ', as_of_utc=monday_open) == 'NQ2606.CME'


def test_review_forced_volume_roll_uses_latest_close_for_newly_listed_contract():
    import json

    decision, effective = '2026-03-05', '2026-03-06'
    with duckdb.connect(config={'threads': 1}) as conn:
        _populate(conn, root='NQ', exchange='CME', start='2026-03-03', end=decision,
                  expiries=['2026-03-06', '2026-06-18'],
                  calendar_id='CME Globex Equity', reference='NYSE')
        conn.execute("UPDATE market_bars_1d SET volume=1000 WHERE instrument_id='NQ2603.CME'")
        build_roll_schedule(conn, product_roots=['NQ'], start=decision, end=decision,
                            rule='max_volume', confirm_days=2)
        assert conn.execute('SELECT to_contract FROM futures_roll_schedule').fetchone() == ('NQ2603.CME',)

        _populate(conn, root='NQ', exchange='CME', start=decision, end=decision,
                  expiries=['2026-09-18'], calendar_id='CME Globex Equity', reference='NYSE')
        conn.execute("UPDATE market_bars_1d SET volume=800 WHERE instrument_id='NQ2609.CME'")
        build_roll_schedule(conn, product_roots=['NQ'], start=effective, end=effective,
                            rule='max_volume', confirm_days=2)
        from_contract, to_contract, decided_on, params = conn.execute(
            'SELECT from_contract, to_contract, decision_session_id, rule_params '
            'FROM futures_roll_schedule WHERE effective_session_id=?', [effective]).fetchone()
        assert (from_contract, to_contract, decided_on) == ('NQ2603.CME', 'NQ2609.CME', decision)
        assert set(json.loads(params)['volume_sources']) == {f'{decision}/NQ2606.CME', f'{decision}/NQ2609.CME'}


def test_review_overnight_daily_bars_use_the_calendar_grid_close_key():
    from slots.global_futures import build_global_futures_slot

    start, day = '2026-03-04', '2026-03-05'
    with duckdb.connect(config={'threads': 1}) as conn:
        _populate(conn, root='ES', exchange='CME', start=start, end=day,
                  expiries=['2026-03-20'], calendar_id='CME Globex Equity', reference='NYSE')
        build_roll_schedule(conn, product_roots=['ES'], start=day, end=day)
        slot = build_global_futures_slot(conn, product_root='ES', start=day, end=day)
        grid = slot.calendar.bar_grid('1d')
        stamp = grid.timestamp_utc.item()
        assert stamp == pd.Timestamp('2026-03-05 22:00Z')
        assert slot.bar_feed.read_signal_bars(
            ['ES'], '1d', start_utc=stamp, end_utc=stamp, as_of_utc=stamp-pd.Timedelta(seconds=1)).empty
        signal = slot.bar_feed.read_signal_bars(['ES'], '1d', start_utc=stamp, end_utc=stamp, as_of_utc=stamp)
        execution = slot.bar_feed.read_execution_bars(['ES'], '1d', start_utc=stamp, end_utc=stamp)
        for frame in (signal, execution):
            pd.testing.assert_frame_equal(frame[list(grid.columns)], grid)
            assert frame.available_at_utc.item() == stamp
            assert frame.close.item() == 103.
            assert frame.concrete_instrument_id.item() == 'ES2603.CME'


def test_review_open_interest_at_first_session_matches_wider_slot():
    from slots.global_futures import build_global_futures_slot

    first, second = '2026-03-05', '2026-03-06'
    with duckdb.connect(config={'threads': 1}) as conn:
        _, calendar = _populate(conn, root='NQ', exchange='CME', start=first, end=second,
            expiries=['2026-03-20'], calendar_id='CME Globex Equity', reference='NYSE')
        wide = build_global_futures_slot(conn, product_root='NQ', start=first, end=second)
        narrow = build_global_futures_slot(conn, product_root='NQ', start=second, end=second)
        grid = narrow.calendar.bar_grid('30min').head(1).assign(instrument_id='NQ2603.CME')
        expected = wide.read_aux_features(grid, features=['open_interest'])
        assert expected.open_interest.item() == 300.
        assert expected.available_at_utc.item() == calendar.session_on(first).segments[-1][1]
        pd.testing.assert_frame_equal(narrow.read_aux_features(grid, features=['open_interest']), expected)


@pytest.mark.parametrize('stored_days', [2, 5])
def test_review_expiry_policy_validates_resolved_default_and_explicit_days(stored_days):
    from slots.futures import DaysBeforeExpiry
    from slots.global_futures import build_global_futures_slot

    start, day = '2026-03-04', '2026-03-05'
    with duckdb.connect(config={'threads': 1}) as conn:
        _, calendar = _populate(conn, root='ES', exchange='CME', start=start, end=day,
            expiries=['2026-03-10', '2026-06-18'], calendar_id='CME Globex Equity', reference='NYSE')
        build_roll_schedule(conn, product_roots=['ES'], start=day, end=day, rule=DaysBeforeExpiry(stored_days))
        open_utc = calendar.session_on(day).segments[0][0]
        for policy, days in [('days_before_expiry', 5), (DaysBeforeExpiry(5), 5), (DaysBeforeExpiry(2), 2)]:
            slot = build_global_futures_slot(conn, product_root='ES', start=start, end=day, roll_policy=policy)
            if days == stored_days:
                contract = slot.universe.resolve_active_contract('ES', as_of_utc=open_utc)
                assert contract == ('ES2603.CME' if days == 2 else 'ES2606.CME')
            else:
                with pytest.raises(ValueError, match='Roll schedule must use days_before_expiry'):
                    slot.universe.resolve_active_contract('ES', as_of_utc=open_utc)
