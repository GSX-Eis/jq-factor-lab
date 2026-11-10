"""Full-calculation fallback and configured PIT anchor queries across sources."""
import json
from calendar_test_support import seed_calendar
import duckdb
import pandas as pd
import pytest
from utils.factor_base import FactorBase
from utils.data_access_config import load_data_access_config


class CrossSection(FactorBase):
    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(['close', 'factor'], table_name='price_data')

    def calculate_factor(self, data):
        data = data.copy()
        data['value'] = data.close - data.groupby('date').close.transform('mean')
        return data.pivot(index='date', columns='code', values='value')


@pytest.fixture
def configured_prices(tmp_path, monkeypatch):
    monkeypatch.delenv('QUANT_DATA_BINDINGS', raising=False)
    # A changing factor across blocks catches lost full-window anchors.
    frame = pd.DataFrame([
        (pd.Timestamp(day), code, float((i + 1) * 100), adjustment)
        for day, adjustment in [('2024-01-02', 2.), ('2024-07-02', 4.)]
        for i, code in enumerate('ABCD')
    ], columns=['trade date', 'security', 'close', 'factor'])
    db = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(db)) as conn:
        seed_calendar(conn)
        conn.execute('CREATE TABLE custom_prices AS SELECT * FROM frame')
        conn.execute('CREATE TABLE price_data_5m AS SELECT "trade date" AS date, security AS stock_code, close, 10.0::DOUBLE AS factor FROM frame')
    config = load_data_access_config().as_dict()
    config['price_tables']['5m'] = 'custom_prices'
    config['tables']['custom_prices'] = {'date_column': 'trade date', 'stock_column': 'security',
        'parquet_subdir': 'custom_prices', 'default_fields': ['close']}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return db, path, frame


@pytest.mark.parametrize('source', ['duckdb', 'no_legacy', 'parquet', 'fallback'])
def test_configured_chunked_values_equal_full(configured_prices, tmp_path, source):
    db, config, frame = configured_prices
    options = {'duckdb_path': str(db)}
    if source == 'no_legacy':
        with duckdb.connect(str(db)) as conn:
            conn.execute('DROP TABLE price_data_5m')
    if source in ('parquet', 'fallback'):
        parquet = tmp_path / 'parquet'
        for month, rows in frame.groupby(frame['trade date'].dt.month):
            folder = parquet / 'custom_prices' / 'year=2024' / f'month={month}'
            folder.mkdir(parents=True)
            rows.to_parquet(folder / 'part.parquet', index=False)
        options = {'parquet_dir': str(parquet), 'calendar_duckdb_path': str(db)}
        if source == 'fallback':
            with duckdb.connect(str(db)) as conn:
                conn.execute('DELETE FROM custom_prices')
            options.update(duckdb_path=str(db), source_policy='duckdb_then_parquet_on_empty')
    factor = CrossSection(None, 'ChunkProbe', '2024-01-02', '2024-07-02', list('ABCD'),
        data_frequency='5m', data_config_path=str(config), data_bindings={}, **options)
    try:
        full = factor.get_factor_values_unfiltered()
        chunked = factor.get_factor_values_unfiltered_chunked(2)
        pd.testing.assert_frame_equal(full, chunked, check_exact=True)
        assert factor.last_performance_metrics['stock_chunk_mode'] == 'fallback_full'
    finally:
        factor._duckdb_adapter.close()

@pytest.mark.parametrize('source', ['duckdb', 'parquet', 'fallback', 'bound'])
def test_pit_anchors_match_requested_observations(configured_prices, tmp_path, source):
    from utils.duckdb_adapter import DuckDBAdapter
    from utils.data_store import migrate_table
    from utils.pit import apply_pit_adjustment
    db, config, frame = configured_prices
    # The unused volume column must not make a missing close into an anchor.
    with duckdb.connect(str(db)) as conn:
        conn.execute('ALTER TABLE custom_prices ADD COLUMN volume DOUBLE DEFAULT 0')
        conn.execute("INSERT INTO custom_prices VALUES ('2024-07-03', 'A', NULL, 1., 0.)")
    options = {'duckdb_path': str(db), 'data_bindings': {}}
    if source in ('parquet', 'fallback'):
        folder = tmp_path / 'parquet' / 'custom_prices' / 'year=2024' / 'month=7'
        folder.mkdir(parents=True)
        with duckdb.connect(str(db)) as conn:
            conn.execute('COPY (SELECT * FROM custom_prices WHERE "trade date" >= DATE \'2024-07-01\') TO ? (FORMAT PARQUET)', [str(folder / 'part.parquet')])
            conn.execute('DELETE FROM custom_prices')
        options['parquet_dir'] = str(tmp_path / 'parquet')
        if source == 'parquet':
            options.pop('duckdb_path')
        else:
            options['source_policy'] = 'duckdb_then_parquet_on_empty'
    if source == 'bound':
        store = tmp_path / 'canonical.duckdb'
        ref = migrate_table(db, store, 'custom_prices', role='input', execute=True)['ref']
        options = {'duckdb_path': str(store), 'data_bindings': {'input_db': str(store),
            'inputs': {'custom_prices': ref}}}
    reader = DuckDBAdapter(data_config_path=str(config), **options)
    try:
        for fields, expected in [(['close', 'factor'], 4.), (['close', 'volume', 'factor'], 1.)]:
            anchors = reader.fetch_price_factor_ends(['A'], '2024-07-01', '2024-07-03',
                frequency='5m', fields=fields)
            assert anchors == {'A': expected}
            raw = reader.fetch_price_data(['A'], '2024-07-01', '2024-07-03', fields, frequency='5m')
            pd.testing.assert_frame_equal(apply_pit_adjustment(raw),
                apply_pit_adjustment(raw, factor_end_overrides=anchors))
    finally:
        reader.close()


def test_anchor_query_errors_do_not_use_parquet(configured_prices, tmp_path):
    from utils.duckdb_adapter import DuckDBAdapter
    db, config, frame = configured_prices
    folder = tmp_path / 'parquet' / 'custom_prices' / 'year=2024' / 'month=1'
    folder.mkdir(parents=True)
    frame.to_parquet(folder / 'part.parquet', index=False)
    with duckdb.connect(str(db)) as conn:
        conn.execute('DROP TABLE custom_prices')
    reader = DuckDBAdapter(str(db), str(tmp_path / 'parquet'), data_config_path=str(config),
        data_bindings={}, source_policy='duckdb_then_parquet_on_empty')
    try:
        with pytest.raises(duckdb.CatalogException):
            reader.fetch_price_factor_ends(['A'], '2024-01-01', '2024-07-03', frequency='5m')
    finally:
        reader.close()


@pytest.mark.parametrize('invalid', [0., -1., float('nan'), float('inf')])
def test_invalid_observed_anchor_fails(configured_prices, invalid):
    from utils.duckdb_adapter import DuckDBAdapter
    db, config, frame = configured_prices
    with duckdb.connect(str(db)) as conn:
        conn.execute('UPDATE custom_prices SET factor = ? WHERE security = ?', [invalid, 'A'])
    reader = DuckDBAdapter(str(db), data_config_path=str(config), data_bindings={})
    try:
        with pytest.raises(ValueError, match='pit_factor_missing_or_invalid'):
            reader.fetch_price_factor_ends(['A'], '2024-01-01', '2024-07-03', frequency='5m')
    finally:
        reader.close()


def test_probe_fallback_keeps_full_window_without_diagnostic_metadata(configured_prices):
    db, config, _ = configured_prices
    with duckdb.connect(str(db)) as conn:
        conn.execute('ALTER TABLE custom_prices ADD COLUMN volume DOUBLE DEFAULT 0')
        for code in 'ABCD':
            conn.execute("INSERT INTO custom_prices VALUES ('2024-07-03', ?, NULL, 1., 0.)", [code])
    class Probe(CrossSection):
        STOCK_CHUNK_INDEPENDENT = True  # Deliberately contradicted by the live probe.
        def calculate_factor(self, data):
            self.last_performance_metrics.clear()
            return super().calculate_factor(data)
    factor = Probe(None, 'fields', '2024-01-02', '2024-07-03', list('ABCD'),
                data_frequency='5m', duckdb_path=str(db), data_config_path=str(config), data_bindings={})
    try:
        expected = factor.get_factor_values_unfiltered()
        actual = factor.get_factor_values_unfiltered_chunked(2)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert factor.last_performance_metrics['stock_chunk_mode'] == 'fallback_full'
    finally:
        factor._duckdb_adapter.close()
