"""Contract validation for Stage 1, using synthetic adapters and clocks."""
from dataclasses import asdict, replace
from datetime import date, datetime
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from backtest_and_analysis.evaluation.labels import ExecutionConvention
from utils.session_calendar import SessionCalendar
from utils.slots import (
    ExecutionProfile, InstrumentSpec, MarketSlot, SlotConfig, SlotRegistry, load_slot_config,
    slot_config_hash,
)
from utils.time_axis import FactorFrequencyRoute


@pytest.fixture
def daily_profile():
    return ExecutionProfile('close', 'close', 0, 0, 'decimal', 'endpoints_only', 'cross_session')


@pytest.fixture
def config(daily_profile):
    return dict(schema_version=1, slot_id='sample', calendar=dict(provider='pmc', options={'id': 'XSHG'}),
                bar_feed=dict(provider='inputs'), universe=dict(provider='stocks'),
                aux_features=dict(provider='features'), microstructure=dict(provider='exchange'),
                execution_profile=asdict(daily_profile))


@pytest.fixture
def slot(daily_profile):
    return MarketSlot(
        slot_id='sample',
        calendar=SessionCalendar('XSHG', '2026-01-05', '2026-01-05'),
        bar_feed=SimpleNamespace(read_signal_bars=lambda *a, **kw: pd.DataFrame(),
                                 read_execution_bars=lambda *a, **kw: pd.DataFrame()),
        universe=SimpleNamespace(instrument_spec=lambda i: InstrumentSpec(i, 'equity', 'CNY', 100),
                                 resolve_universe=lambda i, **kw: ['A'],
                                 resolve_active_contract=lambda i, **kw: i),
        aux_features=SimpleNamespace(read_aux_features=lambda grid, **kw: grid.assign(
            available_at_utc=grid.timestamp_utc, value=1.0)),
        microstructure=SimpleNamespace(execution_profile=daily_profile, tradability_mask=lambda *a, **kw: None),
    )


@pytest.mark.parametrize('convention,profile,boundary', [
    (ExecutionConvention.CLOSE_TO_CLOSE,
     ExecutionProfile('close', 'close', 0, 0, 'decimal', 'endpoints_only', 'cross_session'),
     'cross_session'),
    (ExecutionConvention.OPEN_TO_CLOSE,
     ExecutionProfile('open', 'close', 1, -1, 'percent', 'require_complete_bars', 'same_session'),
     'same_session'),
])
def test_execution_profile_preserves_both_label_conventions(convention, profile, boundary):
    rules = convention.value
    assert (profile.entry_price_field, profile.exit_price_field, profile.min_entry_offset_bars,
            profile.exit_adjustment, profile.return_unit, profile.gap_rule, profile.boundary_policy) == (
                rules.entry_price_field, rules.exit_price_field, rules.min_entry_offset,
                rules.exit_adjustment, rules.unit, rules.gap_rule, boundary)
    assert profile.exit_adjustment + 1 == rules.exit_offset(1)
    assert (profile.allow_short, profile.commission_bps, profile.sell_tax_bps,
            profile.slippage_bps, profile.benchmark_instrument) == (False, 0, 0, 0, None)


@pytest.mark.parametrize('field,value', [
    ('entry_price_field', 'vwap'), ('exit_price_field', 'high'), ('return_unit', 'bps'),
    ('gap_rule', None), ('boundary_policy', 'auto'), ('min_entry_offset_bars', True),
    ('exit_adjustment', -2), ('allow_short', 1), ('commission_bps', -1),
    ('sell_tax_bps', float('nan')), ('slippage_bps', float('inf')), ('benchmark_instrument', []),
])
def test_execution_profile_rejects_invalid_fields(daily_profile, field, value):
    with pytest.raises(ValueError, match=field):
        replace(daily_profile, **{field: value})


@pytest.mark.parametrize('field', ['calendar', 'bar_feed', 'universe', 'aux_features', 'microstructure'])
def test_market_slot_requires_all_five_protocols(slot, field):
    with pytest.raises(TypeError, match=field):
        replace(slot, **{field: object()})


def test_market_slot_registry_and_grid_routing(slot):
    registry = SlotRegistry()
    registry.register(slot)
    assert registry.get('sample') is slot
    assert FactorFrequencyRoute.resolve('1d', 'time_series', slot=slot).bar_grid
    assert slot.universe.instrument_spec('A').lot_size == 100
    with pytest.raises(ValueError, match='already registered'):
        registry.register(slot)
    with pytest.raises(KeyError, match='Unknown MarketSlot'):
        registry.get('missing')
    with pytest.raises(TypeError, match='BarFeedSlot'):
        replace(slot, bar_feed=SimpleNamespace(read_signal_bars=1, read_execution_bars=lambda: None))


def test_market_slot_enforces_pit_and_grid_membership(slot):
    grid = slot.calendar.bar_grid('1d').assign(instrument_id='A')
    expected = grid.assign(available_at_utc=grid.timestamp_utc, value=1.0)
    pd.testing.assert_frame_equal(slot.read_aux_features(grid, features=['value']), expected, check_exact=True)
    for malformed, message in [
        (expected.assign(available_at_utc=grid.timestamp_utc + pd.Timedelta(1, 'ns')), '<='),
        (expected.assign(available_at_utc=grid.timestamp_utc.dt.tz_localize(None)), 'non-null UTC'),
        (expected.assign(session_id='wrong-session'), 'requested grid'),
        (expected.assign(timestamp_utc=grid.timestamp_utc - pd.Timedelta(1, 'min')), 'requested grid'),
        (expected.assign(instrument_id='B'), 'requested grid'),
        (expected.drop(columns='instrument_id'), 'instrument_id'),
        (expected.drop(columns='available_at_utc'), 'requires available_at_utc'),
    ]:
        adapter = SimpleNamespace(read_aux_features=lambda *a, **kw: malformed)
        with pytest.raises(ValueError, match=message):
            replace(slot, aux_features=adapter).read_aux_features(grid, features=['value'])


@pytest.mark.parametrize('categorical_side', ['request', 'result'])
def test_aux_features_accept_single_instrument_categorical(slot, categorical_side):
    calendar = SessionCalendar('XSHG', '2026-03-02', '2026-03-03')
    grid = calendar.bar_grid('1d').assign(instrument_id='A')
    expected = grid.assign(available_at_utc=grid.timestamp_utc, value=1.0)
    if categorical_side == 'request':
        grid = grid.astype({'instrument_id': 'category'})
    else:
        expected = expected.astype({'instrument_id': 'category'})
    adapter = SimpleNamespace(read_aux_features=lambda *a, **kw: expected)
    slot = replace(slot, calendar=calendar, aux_features=adapter)
    pd.testing.assert_frame_equal(slot.read_aux_features(grid, features=['value']), expected, check_exact=True)


def test_slot_config_hash_normalizes_order_defaults_and_file_format(config, tmp_path):
    config['calendar']['options']['templates'] = [{'end_date': '2026-01-06', 'start_date': '2026-01-05'}]
    parsed = SlotConfig.from_mapping(config)
    expected = slot_config_hash(parsed)
    assert len(expected) == 64
    path = tmp_path / 'slot.json'
    path.write_text(json.dumps(dict(reversed(list(config.items()))), sort_keys=True))
    assert load_slot_config(path) == parsed
    assert load_slot_config(path).slot_config_hash == expected
    config['execution_profile'].pop('allow_short')
    config['execution_profile']['commission_bps'] = 0
    config['universe']['options'] = {}
    yaml_path = tmp_path / 'slot.yaml'
    yaml_path.write_text(json.dumps(config, indent=4))
    assert load_slot_config(yaml_path).slot_config_hash == expected
    config['calendar']['options']['templates'][0]['end_date'] = '2026-01-07'
    assert slot_config_hash(config) != expected
    assert parsed.slot_config_hash == expected


def test_slot_config_reads_safe_yaml_and_rejects_duplicate_keys(config, tmp_path):
    yaml = pytest.importorskip('yaml')
    path = tmp_path / 'slot.yaml'
    path.write_text(yaml.safe_dump(config))
    assert load_slot_config(path).slot_config_hash == slot_config_hash(config)
    path.write_text(path.read_text() + 'slot_id: replacement\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_slot_config(path)
    path.write_text('!!python/object/apply:builtins.str [10]')
    with pytest.raises(ValueError, match='Invalid slot YAML'):
        load_slot_config(path)


@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('slot_id', ''), ('calendar', {'options': {}}),
    ('bar_feed', {'provider': 'fixture', 'options': {'bad': float('nan')}}),
    ('universe', {'provider': 'fixture', 'unknown': True}), ('aux_features', None),
    ('microstructure', {'provider': 'fixture', 'options': []}), ('execution_profile', {}),
])
def test_slot_config_rejects_invalid_contract(config, field, value):
    config[field] = value
    with pytest.raises(ValueError, match=field):
        SlotConfig.from_mapping(config)


def test_aux_features_accept_same_timestamp_for_distinct_instruments(slot):
    clock_grid = slot.calendar.bar_grid('1d')
    grid = pd.concat([clock_grid.assign(instrument_id='A'), clock_grid.assign(instrument_id='B')],
                     ignore_index=True)
    expected = grid.assign(available_at_utc=grid.timestamp_utc, value=1.0)
    pd.testing.assert_frame_equal(slot.read_aux_features(grid, features=['value']), expected, check_exact=True)
    with pytest.raises(ValueError, match='distinct'):
        slot.read_aux_features(grid.assign(instrument_id='A'), features=['value'])
    late = expected.copy()
    late.loc[1, 'available_at_utc'] += pd.Timedelta(1, 'ns')
    adapter = SimpleNamespace(read_aux_features=lambda *a, **kw: late)
    with pytest.raises(ValueError, match='<='):
        replace(slot, aux_features=adapter).read_aux_features(grid, features=['value'])


@pytest.mark.parametrize('instrument', [
    None, pd.NA, '', ' A ', 1,
    pytest.param(pd.Categorical([None], categories=['A']), id='categorical-null'),
    pytest.param(pd.Categorical(['']), id='categorical-empty'),
    pytest.param(pd.Categorical([' A ']), id='categorical-whitespace'),
    pytest.param(pd.Categorical([1]), id='categorical-number'),
])
def test_aux_features_require_instrument_ids_in_request(slot, instrument):
    grid = slot.calendar.bar_grid('1d')
    if instrument is not None:
        grid = grid.assign(instrument_id=instrument)
    with pytest.raises(ValueError, match='instrument_id'):
        slot.read_aux_features(grid, features=['value'])


def test_instrument_spec_supports_dated_and_perpetual_instruments():
    listing = date(2025, 9, 22)
    expiry = date(2026, 3, 20)
    contract = InstrumentSpec('ES.H26', 'future', 'USD', contract_multiplier=50,
                              listing_date=listing, expiry_date=expiry)
    equity = InstrumentSpec('A', 'equity', 'CNY', listing_date=listing)
    assert (contract.listing_date, contract.expiry_date) == (listing, expiry)
    assert equity.listing_date == listing and equity.expiry_date is None


@pytest.mark.parametrize('listing,expiry,field', [
    ('2026-03-19', date(2026, 3, 20), 'listing_date'),
    (date(2026, 3, 19), datetime(2026, 3, 20), 'expiry_date'),
    (date(2026, 3, 21), date(2026, 3, 20), 'expiry_date'),
])
def test_instrument_spec_rejects_invalid_lifecycle_dates(listing, expiry, field):
    with pytest.raises(ValueError, match=field):
        InstrumentSpec('ES.H26', 'future', 'USD', listing_date=listing, expiry_date=expiry)
