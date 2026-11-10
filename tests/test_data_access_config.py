"""Configured panel reads must keep source, field and empty-result semantics explicit."""
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from utils.duckdb_adapter import DuckDBAdapter, get_data_adapter
from utils.factor_base import FactorBase


def config_value():
    return {
        'schema_version': 1,
        'price_aliases': {},
        'price_tables': {'daily': 'custom_prices'},
        'factor_table': 'custom_prices',
        'tables': {'custom_prices': {
            'date_column': 'trade_date', 'stock_column': 'symbol',
            'parquet_subdir': 'prices', 'default_fields': ['value'],
        }},
    }


def save_config(tmp_path, value=None):
    path = tmp_path / 'reader.json'
    path.write_text(json.dumps(config_value() if value is None else value))
    return str(path)


def test_price_alias_resolution_and_snapshot(tmp_path):
    from utils.data_access_config import load_data_access_config
    value = config_value()
    value['price_aliases'] = {'legacy_quotes': 'daily'}
    config = load_data_access_config(save_config(tmp_path, value))
    for name in ('legacy_quotes', 'custom_prices'):
        dataset, frequency = config.resolve_price(name)
        assert dataset.table == 'custom_prices'
        assert dataset.date_column == 'trade_date'
        assert frequency == 'daily'
    dataset, frequency = config.resolve_price('other_prices')
    assert dataset.table == 'other_prices' and frequency is None
    assert config.as_dict() == value
    with pytest.raises(TypeError):
        config.price_aliases['legacy_quotes'] = '5m'
    value['price_aliases']['legacy_quotes'] = '5m'
    assert config.resolve_price('legacy_quotes')[1] == 'daily'


def test_explicit_empty_aliases_do_not_inherit_default_aliases(tmp_path):
    from utils.data_access_config import load_data_access_config
    config = load_data_access_config(save_config(tmp_path))
    dataset, frequency = config.resolve_price('price_data')
    assert dataset.table == 'price_data' and frequency is None
    assert config.as_dict() == config_value()


def test_missing_alias_declaration_cannot_silently_change_table_selection(tmp_path):
    from utils.data_access_config import load_data_access_config
    value = config_value()
    del value['price_aliases']
    with pytest.raises(ValueError, match='price_aliases'):
        load_data_access_config(save_config(tmp_path, value))


@pytest.mark.parametrize('change,match', [
    ('unknown_frequency', 'unconfigured frequency'),
    ('invalid_aliases', 'price_aliases must'),
    ('invalid_name', 'price alias'),
    ('invalid_frequency', 'frequency'),
    ('duplicate_table', 'multiple frequencies'),
    ('conflicting_alias', 'conflicts'),
])
def test_invalid_price_mappings_fail_when_loading(tmp_path, change, match):
    from utils.data_access_config import load_data_access_config
    value = config_value()
    value['price_aliases'] = {'legacy_quotes': 'daily'}
    if change == 'unknown_frequency': value['price_aliases']['legacy_quotes'] = '5m'
    if change == 'invalid_aliases': value['price_aliases'] = []
    if change == 'invalid_name': value['price_aliases'] = {'': 'daily'}
    if change == 'invalid_frequency': value['price_aliases'] = {'legacy_quotes': []}
    if change == 'duplicate_table': value['price_tables']['5m'] = 'custom_prices'
    if change == 'conflicting_alias':
        value['tables']['intraday'] = dict(value['tables']['custom_prices'])
        value['price_tables']['5m'] = 'intraday'
        value['price_aliases']['intraday'] = 'daily'
    with pytest.raises(ValueError, match=match):
        load_data_access_config(save_config(tmp_path, value))


@pytest.fixture(params=['duckdb', 'parquet'])
def configured(tmp_path, request):
    frame = pd.DataFrame({'trade_date': pd.to_datetime(['2026-01-05 15:00', '2026-01-05 10:00']),
                          'symbol': ['B', 'A'], 'value': [2., 1.], 'other': [20., 10.]})
    config = save_config(tmp_path)
    if request.param == 'duckdb':
        db = tmp_path / 'input.duckdb'
        with duckdb.connect(str(db)) as conn:
            conn.execute('CREATE TABLE custom_prices AS SELECT * FROM frame')
        args = {'duckdb_path': str(db)}
    else:
        root = tmp_path / 'parquet'
        folder = root / 'prices' / 'year=2026' / 'month=1'
        folder.mkdir(parents=True)
        frame.to_parquet(folder / 'part.parquet', index=False)
        args = {'parquet_dir': str(root)}
    reader = get_data_adapter(**args, data_config_path=config, data_bindings={})
    yield reader, config
    reader.close()


def fetch(reader, method, stocks, fields=None, start='2026-01-05', end='2026-01-05'):
    args = (stocks, start, end, fields)
    if method == 'table':
        return reader.fetch_table_data('custom_prices', *args)
    return getattr(reader, f'fetch_{method}_data')(*args)


@pytest.mark.parametrize('method', ['price', 'factor', 'table'])
@pytest.mark.parametrize('fields,expected', [(None, ['value']), ([], []),
    (['other', 'value', 'other', 'symbol', 'trade_date'], ['other', 'value'])])
def test_configured_reads_share_schema_and_values(configured, method, fields, expected):
    reader, _ = configured
    result = fetch(reader, method, ['B', 'A'], fields)
    assert result.columns.tolist() == ['date', 'code', *expected]
    assert result.code.tolist() == ['A', 'B']
    if 'value' in expected:
        assert result.value.tolist() == [1., 2.]
    for stocks in ([], ['ABSENT']):
        empty = fetch(reader, method, stocks, fields)
        assert empty.columns.tolist() == result.columns.tolist()
        assert empty.empty and pd.api.types.is_datetime64_any_dtype(empty.date)
        assert isinstance(empty.index, pd.RangeIndex)


@pytest.mark.parametrize('stocks', [[], ['A']])
@pytest.mark.parametrize('fields', ['value', [None], [''], ['code']])
def test_invalid_fields_cannot_hide_behind_empty_stocks_or_cache(configured, stocks, fields):
    reader, _ = configured
    fetch(reader, 'price', ['A'])
    with pytest.raises(ValueError, match='field'):
        fetch(reader, 'price', stocks, fields)


@pytest.mark.parametrize('stocks', [[], ['A']])
@pytest.mark.parametrize('start,end', [('2026-01-06', '2026-01-05'), ('bad', '2026-01-05')])
def test_invalid_dates_never_read_or_hit_cache(configured, stocks, start, end):
    reader, _ = configured
    fetch(reader, 'price', ['A'])
    with pytest.raises(ValueError):
        fetch(reader, 'price', stocks, start=start, end=end)


def test_explicit_missing_field_errors(configured):
    reader, _ = configured
    with pytest.raises(duckdb.Error):
        fetch(reader, 'factor', ['A'], ['missing'])


def test_config_is_snapshot_and_cache_does_not_follow_file_changes(configured):
    reader, path = configured
    first = fetch(reader, 'price', ['A'])
    value = config_value()
    value['tables']['custom_prices']['default_fields'] = ['other']
    Path(path).write_text(json.dumps(value))
    pd.testing.assert_frame_equal(fetch(reader, 'price', ['A']), first)
    reader.close()
    second = DuckDBAdapter(duckdb_path=reader.duckdb_path, parquet_dir=reader.parquet_dir,
                          data_config_path=path, data_bindings={})
    try:
        assert fetch(second, 'price', ['A']).columns.tolist() == ['date', 'code', 'other']
    finally:
        second.close()


@pytest.mark.parametrize('change', ['version', 'unknown_table', 'same_keys', 'escape', 'fields_string', 'unknown_option'])
def test_bad_config_fails_at_construction(tmp_path, change):
    value = config_value()
    table = value['tables']['custom_prices']
    if change == 'version': value['schema_version'] = 2
    if change == 'unknown_table': value['factor_table'] = 'missing'
    if change == 'same_keys': table['stock_column'] = 'trade_date'
    if change == 'escape': table['parquet_subdir'] = '../outside'
    if change == 'fields_string': table['default_fields'] = 'value'
    if change == 'unknown_option': table['default_field'] = ['value']
    with pytest.raises(ValueError):
        DuckDBAdapter(parquet_dir=str(tmp_path), data_config_path=save_config(tmp_path, value), data_bindings={})


def test_explicit_missing_and_malformed_config_are_not_ignored(tmp_path):
    path = tmp_path / 'missing.json'
    with pytest.raises(FileNotFoundError):
        DuckDBAdapter(parquet_dir=str(tmp_path), data_config_path=str(path), data_bindings={})
    path.write_text('{bad')
    with pytest.raises(ValueError):
        DuckDBAdapter(parquet_dir=str(tmp_path), data_config_path=str(path), data_bindings={})


def test_default_resource_independent_of_cwd_and_factor_requires_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reader = DuckDBAdapter(parquet_dir=str(tmp_path), data_bindings={})
    try:
        assert reader.fetch_price_data([], '2026-01-05', '2026-01-05').columns.tolist() == [
            'date', 'code', 'open', 'close', 'high', 'low', 'volume', 'money']
        with pytest.raises(ValueError, match='fields'):
            reader.fetch_factor_data([], '2026-01-05', '2026-01-05')
    finally:
        reader.close()


def test_both_sources_require_explicit_policy(tmp_path):
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)):
        pass
    with pytest.raises(ValueError, match='source_policy'):
        DuckDBAdapter(duckdb_path=str(db), parquet_dir=str(tmp_path), data_bindings={})
    with pytest.raises(ValueError, match='source_policy'):
        DuckDBAdapter(duckdb_path=str(db), source_policy='typo', data_bindings={})


@pytest.mark.parametrize('kind', ['missing', 'empty'])
def test_missing_parquet_dataset_is_not_empty_data(tmp_path, kind):
    if kind == 'empty': (tmp_path / 'factor_data').mkdir()
    reader = DuckDBAdapter(parquet_dir=str(tmp_path), data_bindings={})
    try:
        with pytest.raises((FileNotFoundError, duckdb.Error)):
            reader.fetch_factor_data(['A'], '2026-01-05', '2026-01-05', ['value'])
    finally:
        reader.close()


def test_missing_month_is_empty_but_requested_partition_columns_are_preserved(configured):
    reader, _ = configured
    assert fetch(reader, 'factor', ['A'], start='2026-02-01', end='2026-02-02').empty
    if reader.parquet_dir:
        frame = fetch(reader, 'factor', ['A'], ['year', 'month', 'value'])
        assert frame.columns.tolist() == ['date', 'code', 'year', 'month', 'value']
        assert frame.year.tolist() == [2026] and frame.month.tolist() == [1]


@pytest.mark.parametrize('mode', ['path', 'borrowed'])
def test_factor_base_uses_same_custom_table_and_keys(tmp_path, mode):
    config = save_config(tmp_path)
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE custom_prices AS SELECT TIMESTAMP '2026-01-05 15:00:00' AS trade_date, 'A' AS symbol, 7 AS value")
    conn = duckdb.connect(str(db)) if mode == 'borrowed' else None
    factor = FactorBase(conn, 'Configured', '2026-01-05', '2026-01-05', stock_pool=['A'],
        pit_enabled=False, data_bindings={}, data_config_path=config,
        duckdb_path=str(db) if mode == 'path' else None)
    try:
        for table in ['price_data', 'factor_data']:
            result = factor._fetch_raw_data(['value'], table_name=table)
            assert result.columns.tolist() == ['date', 'code', 'value']
            assert result.value.tolist() == [7]
    finally:
        if factor._duckdb_adapter: factor._duckdb_adapter.close()
        if conn is not None: conn.close()


def test_cloud_does_not_load_local_config(tmp_path):
    factor = FactorBase(None, 'CloudConfig', '2026-01-05', '2026-01-05', stock_pool=['A'],
        pit_enabled=False, data_bindings={}, data_frequency='15m', data_config_path=str(tmp_path / 'absent.json'))
    assert factor.data_frequency == '15m'


def test_config_is_immutable_and_cache_identity_includes_key_mapping(tmp_path):
    from dataclasses import FrozenInstanceError
    from utils.data_access_config import load_data_access_config
    path = save_config(tmp_path)
    first = DuckDBAdapter(parquet_dir=str(tmp_path), data_config_path=path, data_bindings={})
    try:
        with pytest.raises(TypeError):
            first.data_config.price_tables['daily'] = 'elsewhere'
        with pytest.raises(FrozenInstanceError):
            first.data_config.price('daily').date_column = 'other_date'
        value = config_value()
        value['tables']['custom_prices']['date_column'] = 'other_date'
        second_config = load_data_access_config(save_config(tmp_path, value))
        second = DuckDBAdapter(parquet_dir=str(tmp_path), data_config_path=path, data_bindings={})
        try:
            args = ('custom_prices', ['A'], '2026-01-05', '2026-01-05', ['value'])
            assert first._build_cache_key(*args) != second._build_cache_key(*args)
            assert second_config.as_dict() == value
        finally:
            second.close()
    finally:
        first.close()


def test_dual_source_does_not_merge_partial_results(tmp_path):
    db = tmp_path / 'input.duckdb'
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE factor_data AS SELECT DATE '2026-01-05' AS date, 'A' AS stock_code, 1 AS value")
    folder = tmp_path / 'parquet' / 'factor_data' / 'year=2026' / 'month=1'
    folder.mkdir(parents=True)
    pd.DataFrame({'date': pd.to_datetime(['2026-01-05']), 'stock_code': ['B'], 'value': [2]}).to_parquet(folder / 'part.parquet')
    reader = DuckDBAdapter(duckdb_path=str(db), parquet_dir=str(tmp_path / 'parquet'),
        source_policy='duckdb_then_parquet_on_empty', data_bindings={})
    try:
        result = reader.fetch_factor_data(['A', 'B'], '2026-01-05', '2026-01-05', ['value'])
        assert result.code.tolist() == ['A']
        assert reader.fetch_factor_data(['B'], '2026-01-05', '2026-01-05', ['value']).value.tolist() == [2]
    finally:
        reader.close()


def test_schema_qualified_table_uses_explicit_parquet_directory(tmp_path):
    value = config_value()
    value['tables']['main.custom_prices'] = value['tables'].pop('custom_prices')
    value['price_tables'] = {'daily': 'main.custom_prices'}
    value['factor_table'] = 'main.custom_prices'
    root = tmp_path / 'parquet'
    folder = root / 'prices' / 'year=2026' / 'month=1'
    folder.mkdir(parents=True)
    pd.DataFrame({'trade_date': pd.to_datetime(['2026-01-05']), 'symbol': ['A'], 'value': [3]}).to_parquet(folder / 'part.parquet')
    reader = DuckDBAdapter(parquet_dir=str(root), data_config_path=save_config(tmp_path, value), data_bindings={})
    try:
        assert reader.fetch_price_data(['A'], '2026-01-05', '2026-01-05').value.tolist() == [3]
    finally:
        reader.close()


def test_mispartitioned_rows_require_upstream_validation(tmp_path):
    # Query pruning trusts year/month; it does not scan other partitions to repair bad data.
    folder = tmp_path / 'factor_data' / 'year=2026' / 'month=1'
    folder.mkdir(parents=True)
    pd.DataFrame({'date': pd.to_datetime(['2026-02-05']), 'stock_code': ['A'], 'value': [3]}).to_parquet(folder / 'part.parquet')
    reader = DuckDBAdapter(parquet_dir=str(tmp_path), data_bindings={})
    try:
        assert reader.fetch_factor_data(['A'], '2026-02-05', '2026-02-05', ['value']).empty
    finally:
        reader.close()
