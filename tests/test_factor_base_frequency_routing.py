"""Frequency spelling is a routing contract, independent of execution backend."""
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

import duckdb
import pandas as pd
import pytest

from utils.factor_base import FactorBase


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('frequency', ['1m', '5m', '15m', '30m', '60m', '1h', 'daily', '5M', None, ''])
def test_legacy_and_invalid_construct_without_frequency_validation(tmp_path, local, frequency):
    connection = duckdb.connect() if local else None
    try:
        item = FactorBase(connection, 'route', '2026-01-05', '2026-01-05',
                          stock_pool=['A'], data_bindings={}, data_frequency=frequency,
                          data_config_path=str(tmp_path / 'missing.json'))
        assert item.data_frequency == frequency
        assert item._local_data_config is None
    finally:
        if connection is not None:
            connection.close()


@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('frequency', ['1min', '5min', '15min', '30min', '60min', '1d'])
def test_canonical_requires_explicit_mode(local, frequency):
    connection = duckdb.connect() if local else None
    try:
        with pytest.raises(ValueError, match='analysis_mode'):
            FactorBase(connection, 'route', '2026-01-05', '2026-01-05',
                       stock_pool=['A'], data_bindings={}, data_frequency=frequency)
    finally:
        if connection is not None:
            connection.close()


@pytest.mark.parametrize('frequency', ['1m', '5M', None, ''])
def test_missing_frequency_fails_at_local_read(frequency):
    with duckdb.connect() as connection:
        item = FactorBase(connection, 'route', '2026-01-05', '2026-01-05',
                          stock_pool=['A'], data_bindings={}, data_frequency=frequency,
                          pit_enabled=False)
        with pytest.raises(ValueError, match='frequency'):
            item.get_price_data()


def baseline_base():
    source = Path(__file__).resolve().parents[1] / 'scripts/build/cloud_runtime/factor_base.py.txt'
    name = 'utils._stage62_factor_base'
    loader = importlib.machinery.SourceFileLoader(name, str(source))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module.FactorBase


def probe(base, connection=None, start='2026-01-05', end='2026-01-06', **options):
    class DailyProbe(base):
        def _get_cache_inputs(self):
            return {'window': 3}
    return DailyProbe(connection, 'daily', start, end,
                      stock_pool=['A'], data_bindings={}, **options)


def test_daily_cache_key_is_byte_identical_to_stage62():
    raw = pd.DataFrame({'date': pd.to_datetime(['2026-01-05']), 'code': ['A'], 'close': [10.]})
    expected = probe(baseline_base())._get_cache_key(raw)
    assert expected is not None
    assert probe(FactorBase)._get_cache_key(raw).encode() == expected.encode()


@pytest.mark.parametrize('backing', ['borrowed', 'path'])
@pytest.mark.parametrize('pit', [False, True])
def test_daily_sql_pit_and_cache_identity_match_stage62(tmp_path, backing, pit):
    from test_factor_cache_identity import make_store
    path = tmp_path / 'daily.duckdb'
    make_store(path, 1.)
    connection = duckdb.connect(str(path)) if backing == 'borrowed' else None
    options = dict(pit_enabled=pit)
    if backing == 'path':
        options['duckdb_path'] = str(path)
    items = [probe(base, connection, start='2024-01-02', end='2024-01-03', **options)
             for base in (baseline_base(), FactorBase)]
    try:
        raw = [item._fetch_raw_data(['close'], table_name='price_data') for item in items]
        pd.testing.assert_frame_equal(raw[0], raw[1], check_exact=True)
        assert raw[0].attrs == raw[1].attrs
        assert items[0]._get_cache_key(raw[0]) == items[1]._get_cache_key(raw[1])
        if pit:
            pd.testing.assert_frame_equal(items[0]._apply_pit_adjustment(raw[0]),
                                          items[1]._apply_pit_adjustment(raw[1]), check_exact=True)
    finally:
        for item in items:
            if item._duckdb_adapter:
                item._duckdb_adapter.close()
        if connection is not None:
            connection.close()


def test_five_minute_explicit_mode_uses_bar_interface():
    item = FactorBase(None, 'route', '2026-01-05', '2026-01-05', stock_pool=['A'],
                      data_bindings={}, data_frequency='5m', analysis_mode='time_series')
    assert item._frequency_route.bar_grid


@pytest.mark.parametrize('frequency', ['1min', '5min', '15min', '30min', '60min', '1d'])
def test_canonical_explicit_mode_constructs_without_local_config(tmp_path, frequency):
    item = FactorBase(None, 'route', '2026-01-05', '2026-01-05', stock_pool=['A'],
                      data_bindings={}, data_frequency=frequency, analysis_mode='cross_sectional',
                      data_config_path=str(tmp_path / 'absent.json'))
    assert item._frequency_route.explicit_interface
    assert item._local_data_config is None


def test_future_source_edits_invalidate_legacy_identity(tmp_path):
    from utils.legacy_cache_identity import factor_base_digest
    import hashlib
    source = tmp_path / 'factor_base.py'
    source.write_text('future implementation')
    assert factor_base_digest(str(source)) == hashlib.sha256(source.read_bytes()).hexdigest()


def test_extracted_reader_edits_invalidate_legacy_identity(tmp_path):
    import hashlib
    import json
    from utils.legacy_cache_identity import factor_base_digest
    original = Path(__file__).resolve().parents[1] / 'utils'
    mapping = json.loads((original / 'legacy_cache_identity.json').read_text())
    source = tmp_path / 'factor_base.py'
    source.write_bytes((original / 'factor_base.py').read_bytes())
    for name in mapping['dependencies']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    assert factor_base_digest(str(source)) == mapping['baseline_sha256']
    (tmp_path / 'bar_source.py').write_text('changed SQL')
    assert factor_base_digest(str(source)) != mapping['baseline_sha256']


@pytest.mark.parametrize('frequency', ['1min', '5min', '15min', '30min', '60min'])
def test_new_tables_flow_through_bar_source(frequency):
    from test_build_bars import sample
    from data_pipeline.loaders.market_bar_importer import import_market_bars
    bars, metadata, _ = sample(days=('2026-01-06',), frequency=frequency)
    with duckdb.connect() as connection:
        import_market_bars(connection, bars, metadata, legacy_compatibility=False)
        item = FactorBase(connection, 'new-table', '2026-01-06', '2026-01-06', stock_pool=['A'],
                          data_bindings={}, data_frequency=frequency, analysis_mode='time_series')
        grid = item._fetch_market_bars()
        assert len(grid) == len(bars)
        assert grid.observed.all()
        assert item.market_bars.close.tolist() == bars.close.tolist()


def test_legacy_spelling_reads_registered_new_frequency():
    from test_build_bars import sample
    from data_pipeline.loaders.market_bar_importer import import_market_bars
    bars, metadata, _ = sample(days=('2026-01-06',), frequency='15min')
    with duckdb.connect() as connection:
        import_market_bars(connection, bars, metadata, legacy_compatibility=False)
        item = FactorBase(connection, 'panel', '2026-01-06', '2026-01-06', stock_pool=['A'],
                          data_bindings={}, data_frequency='15m', pit_enabled=False)
        assert item.get_price_data()['A'].tolist() == bars.close.tolist()


def test_compatibility_snapshot_warns_on_each_quality_marked_read():
    from test_build_bars import sample
    from data_pipeline.loaders.market_bar_importer import import_market_bars
    from utils.bar_quality import BarDataQualityWarning
    bars, metadata, _ = sample(days=('2026-01-06',), frequency='1min')
    with duckdb.connect() as connection:
        import_market_bars(connection, bars.iloc[1:], metadata)
        item = FactorBase(connection, 'quality', '2026-01-06', '2026-01-06', stock_pool=['A'],
                          data_bindings={}, data_frequency='1min', analysis_mode='time_series')
        for _ in range(2):
            with pytest.warns(BarDataQualityWarning, match='frequency=1min'):
                item._fetch_market_bars()
