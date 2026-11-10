"""Cache correctness across sources, adjustment and user factor parameters."""
import inspect
import logging
import pickle

import duckdb
import numpy as np
import pytest
import pandas as pd

from utils.factor_base import FactorBase
from utils.dynamic_stock_pool import StockFilter

CALLS = []


@pytest.mark.parametrize('kind', [
    'dict', 'series', 'empty', 'row_numbers', 'date_strings', 'unnamed_dates',
    'duplicate_dates', 'missing_date', 'multiindex', 'duplicate_stocks',
    'numeric_stock', 'empty_stock',
])
def test_invalid_cache_payload_is_recomputed_and_replaced(tmp_path, caplog, kind):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = factor(path, tmp_path/'cache', pit_enabled=False)
    CALLS.clear()
    try:
        expected = item.get_factor_values()
        key = item._get_cache_key(item._get_raw_data_for_factor())
        payload = expected.copy()
        if kind == 'dict': payload = {'A': [999., 999.]}
        elif kind == 'series': payload = expected['A']
        elif kind == 'empty': payload = expected.iloc[:0]
        elif kind == 'row_numbers': payload.index = pd.Index([0, 1], name='date')
        elif kind == 'date_strings': payload.index = expected.index.strftime('%Y-%m-%d')
        elif kind == 'unnamed_dates': payload.index = expected.index.rename(None)
        elif kind == 'duplicate_dates': payload.index = pd.DatetimeIndex([expected.index[0]]*2, name='date')
        elif kind == 'missing_date': payload.index = pd.DatetimeIndex([expected.index[0], pd.NaT], name='date')
        elif kind == 'multiindex': payload.index = pd.MultiIndex.from_arrays([expected.index, ['A', 'A']])
        elif kind == 'duplicate_stocks': payload = pd.concat([expected, expected], axis=1)
        elif kind == 'numeric_stock': payload.columns = [1]
        elif kind == 'empty_stock': payload.columns = ['']
        cache_path = tmp_path/'cache'/(key+'.pkl')
        with cache_path.open('wb') as stream: pickle.dump(payload, stream)
        with caplog.at_level(logging.WARNING):
            actual = item.get_factor_values()
        pd.testing.assert_frame_equal(actual, expected)
        assert len(CALLS) == 2
        assert '加载缓存失败' in caplog.text
        pd.testing.assert_frame_equal(item.get_factor_values(), expected)
        assert len(CALLS) == 2
    finally:
        item._duckdb_adapter.close()


@pytest.mark.parametrize('kind', ['all_nan', 'unsorted', 'intraday_timezone', 'object_values'])
def test_cache_validation_preserves_existing_normalized_output_contract(tmp_path, kind):
    item = CachedClose(None, 'valid', '2024-01-02', '2024-01-03',
                       stock_pool=['A'], cache_dir=str(tmp_path), data_bindings={})
    frame = pd.DataFrame({'A': [1., 2.]}, index=pd.date_range('2024-01-02', periods=2, name='date'))
    if kind == 'all_nan': frame[:] = np.nan
    elif kind == 'unsorted': frame = frame.iloc[::-1]
    elif kind == 'intraday_timezone': frame.index = frame.index.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
    elif kind == 'object_values': frame = frame.astype(object)
    frame.attrs['source'] = 'validation-probe'
    pd.testing.assert_frame_equal(FactorBase._normalize_factor_dataframe(frame, pd.DataFrame()), frame)
    item._save_to_cache(frame, 'factor-v3-valid')
    pd.testing.assert_frame_equal(item._load_from_cache('factor-v3-valid'), frame)


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('failure', ['write', 'replace', 'interrupt'])
def test_failed_cache_save_preserves_previous_file_and_cleans_scratch(tmp_path, monkeypatch, existing, failure):
    item = CachedClose(None, 'atomic', '2024-01-02', '2024-01-03',
                       stock_pool=['A'], cache_dir=str(tmp_path), data_bindings={})
    key = 'factor-v3-' + 'a' * 64
    target = tmp_path / (key + '.pkl')
    previous = pd.DataFrame({'A': [1., 2.]}, index=pd.date_range('2024-01-02', periods=2, name='date'))
    replacement = pd.DataFrame({'A': [3., 4.]}, index=previous.index)
    if existing:
        item._save_to_cache(previous, key)
    before = target.read_bytes() if existing else None

    def fail_write(data, stream):
        stream.write(b'partial pickle')
        stream.flush()
        if failure == 'interrupt':
            raise KeyboardInterrupt('simulated interruption')
        raise OSError('simulated write failure')

    def fail_replace(source, destination):
        raise OSError('simulated publication failure')

    if failure == 'replace':
        monkeypatch.setattr('utils.factor_base.os.replace', fail_replace)
    else:
        monkeypatch.setattr('utils.factor_base.pickle.dump', fail_write)
    if failure == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            item._save_to_cache(replacement, key)
    else:
        item._save_to_cache(replacement, key)
    assert (target.read_bytes() if target.exists() else None) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ([target.name] if existing else [])
    if existing:
        pd.testing.assert_frame_equal(item._load_from_cache(key), previous)


def test_cache_reader_sees_old_file_until_new_file_is_published(tmp_path, monkeypatch):
    item = CachedClose(None, 'atomic', '2024-01-02', '2024-01-03',
                       stock_pool=['A'], cache_dir=str(tmp_path), data_bindings={})
    key = 'factor-v3-' + 'b' * 64
    previous = pd.DataFrame({'A': [1., 2.]}, index=pd.date_range('2024-01-02', periods=2, name='date'))
    replacement = pd.DataFrame({'A': [3., 4.]}, index=previous.index)
    replacement.attrs['unit'] = 'factor'
    item._save_to_cache(previous, key)
    original_dump = pickle.dump

    def inspect_write(data, stream):
        pd.testing.assert_frame_equal(item._load_from_cache(key), previous)
        original_dump(data, stream)
        stream.flush()
        pd.testing.assert_frame_equal(item._load_from_cache(key), previous)

    monkeypatch.setattr('utils.factor_base.pickle.dump', inspect_write)
    item._save_to_cache(replacement, key)
    pd.testing.assert_frame_equal(item._load_from_cache(key), replacement)
    assert [p.name for p in tmp_path.iterdir()] == [key + '.pkl']


@pytest.mark.parametrize('operator', ['in', 'not_in'])
@pytest.mark.parametrize('container', [set, frozenset])
@pytest.mark.parametrize('members', [['A', 'B'], ['A', 2, None]])
def test_set_filters_reuse_cache_and_member_changes_invalidate(tmp_path, operator, container, members):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    original = container(members)
    condition = StockFilter('stock_code', operator, original)
    item = factor(path, tmp_path/'cache', pit_enabled=False, stock_filters=[condition])
    CALLS.clear()
    try:
        raw = item._get_raw_data_for_factor()
        first_key = item._get_cache_key(raw)
        assert first_key is not None
        first = item.get_factor_values()
        assert condition.value is original
        condition.value = container(reversed(members))
        assert item._get_cache_key(raw) == first_key
        pd.testing.assert_frame_equal(item.get_factor_values(), first)
        assert len(CALLS) == 1
        assert original == container(members)
        assert isinstance(condition.value, container)
        condition.value = container([*members, 'C'])
        assert item._get_cache_key(raw) != first_key
        item.get_factor_values()
        assert len(CALLS) == 2
        item.stock_filters = [StockFilter('stock_code', operator, ['A', 'B'])]
        ordered_key = item._get_cache_key(raw)
        item.stock_filters[0].value = ['B', 'A']
        assert item._get_cache_key(raw) != ordered_key
    finally:
        item._duckdb_adapter.close()


class CachedClose(FactorBase):
    scale = 1.0

    def _get_cache_inputs(self):
        return {'scale': self.scale, 'custom_parameter': getattr(self, 'custom_parameter', None)}

    def _get_raw_data_for_factor(self):
        return self._fetch_raw_data(['close'], table_name='price_data')

    def calculate_factor(self, data):
        CALLS.append(1)
        return data.pivot(index='date', columns='code', values='close') * self.scale


def make_store(path, scale):
    with duckdb.connect(str(path)) as conn:
        conn.execute('CREATE TABLE price_data_daily(date DATE, stock_code VARCHAR, close DOUBLE, factor DOUBLE)')
        conn.execute("INSERT INTO price_data_daily VALUES ('2024-01-02','A',?,1), ('2024-01-03','A',?,2)", [10*scale,12*scale])


def factor(path, cache, **options):
    return CachedClose(None, 'same_name', '2024-01-02', '2024-01-03', stock_pool=['A'],
                       cache_dir=str(cache), duckdb_path=str(path), **options)


def test_cache_reuses_identical_calculation_but_separates_sources_pit_and_parameters(tmp_path):
    a, b = tmp_path/'a.duckdb', tmp_path/'b.duckdb'
    make_store(a, 1); make_store(b, 10)
    CALLS.clear()
    factors = []
    try:
        first = factor(a, tmp_path/'cache', pit_enabled=False); factors.append(first)
        assert first.get_factor_values()['A'].tolist() == [10,12]
        assert first.get_factor_values()['A'].tolist() == [10,12]
        same = factor(a, tmp_path/'cache', pit_enabled=False); factors.append(same)
        same.get_factor_values()
        assert len(CALLS) == 1
        second = factor(b, tmp_path/'cache', pit_enabled=False); factors.append(second)
        assert second.get_factor_values()['A'].tolist() == [100,120]
        adjusted = factor(a, tmp_path/'cache', pit_enabled=True); factors.append(adjusted)
        assert adjusted.get_factor_values()['A'].tolist() == [5,12]
        adjusted.scale = 2
        assert adjusted.get_factor_values()['A'].tolist() == [10,24]
        assert len(CALLS) == 4
    finally:
        for item in factors:
            item._duckdb_adapter.close()


def test_borrowed_connection_mutation_invalidates_calculation_cache(tmp_path):
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE price_data_daily AS SELECT DATE '2024-01-02' date, 'A' stock_code, 10.0 AS close")
        item = CachedClose(conn, 'same_name', '2024-01-02', '2024-01-03', stock_pool=['A'],
                           cache_dir=str(tmp_path), pit_enabled=False)
        assert item.get_factor_values().iloc[0,0] == 10
        conn.execute('UPDATE price_data_daily SET close=20')
        assert item.get_factor_values().iloc[0,0] == 20


def test_unidentifiable_declared_parameter_disables_cache(tmp_path):
    path = tmp_path/'a.duckdb'; make_store(path, 1)
    item = factor(path, tmp_path/'cache', pit_enabled=False)
    try:
        item.custom_parameter = object()
        assert item._get_cache_key(item._get_raw_data_for_factor()) is None
        item.get_factor_values()
        assert not list((tmp_path/'cache').glob('*.pkl'))
    finally:
        item._duckdb_adapter.close()


class StatefulClose(CachedClose):
    def _get_cache_inputs(self):
        return super()._get_cache_inputs()

    def calculate_factor(self, data):
        self.diagnostic_count = getattr(self, 'diagnostic_count', 0) + 1
        self._frame_cache = data.copy()
        self.runtime_handle = object()
        return super().calculate_factor(data)


class UndeclaredClose(CachedClose):
    """An inherited declaration cannot describe arbitrary new child behavior."""


class DependencyClose(CachedClose):
    dependency = None

    def _get_cache_inputs(self):
        if isinstance(self.dependency, Exception):
            raise self.dependency
        return {'scale': self.scale, 'events': self.dependency}

    def calculate_factor(self, data):
        return super().calculate_factor(data) + self.dependency['value'].sum()


def test_runtime_state_does_not_prevent_same_or_fresh_instance_hits(tmp_path):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    CALLS.clear()
    for _ in range(2):
        item = StatefulClose(None, 'stateful', '2024-01-02', '2024-01-03',
                             stock_pool=['A'], duckdb_path=str(path), cache_dir=str(tmp_path/'cache'), pit_enabled=False)
        try:
            assert item.get_factor_values()['A'].tolist() == [10, 12]
            assert item.get_factor_values()['A'].tolist() == [10, 12]
        finally:
            item._duckdb_adapter.close()
    assert len(CALLS) == 1


def test_undeclared_subclass_computes_without_cache_and_explains_why(tmp_path, caplog):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = UndeclaredClose(None, 'undeclared', '2024-01-02', '2024-01-03',
                           stock_pool=['A'], duckdb_path=str(path), cache_dir=str(tmp_path/'cache'), pit_enabled=False)
    CALLS.clear()
    try:
        with caplog.at_level(logging.WARNING):
            item.get_factor_values(); item.get_factor_values()
        assert len(CALLS) == 2
        assert not list((tmp_path/'cache').glob('*.pkl'))
        assert '_get_cache_inputs' in caplog.text
    finally:
        item._duckdb_adapter.close()


def test_stock_filters_have_stable_identity_and_changes_invalidate(tmp_path):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    CALLS.clear()
    for cutoff in [100, 100, 200]:
        item = factor(path, tmp_path/'cache', pit_enabled=False, use_dynamic_pool=True,
                      stock_pools_by_period={'2024-01': ['A']},
                      stock_filters=[StockFilter('market_cap', '>', cutoff)])
        try:
            item.get_factor_values()
        finally:
            item._duckdb_adapter.close()
    assert len(CALLS) == 2


def test_extra_data_updates_invalidate_same_and_fresh_instances(tmp_path):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    CALLS.clear()
    for values in [[1, 1, 2], [2, 3]]:
        item = DependencyClose(None, 'events', '2024-01-02', '2024-01-03',
                               stock_pool=['A'], duckdb_path=str(path), cache_dir=str(tmp_path/'cache'), pit_enabled=False)
        try:
            for value in values:
                item.dependency = pd.DataFrame({'value': [value]})
                assert item.get_factor_values().iloc[0,0] == 10 + value
        finally:
            item._duckdb_adapter.close()
    assert len(CALLS) == 3


def test_dependency_read_error_is_not_hidden_by_cache(tmp_path):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = DependencyClose(None, 'events', '2024-01-02', '2024-01-03',
                           stock_pool=['A'], duckdb_path=str(path), cache_dir=str(tmp_path/'cache'), pit_enabled=False)
    try:
        item.dependency = pd.DataFrame({'value': [1]})
        item.get_factor_values()
        item.dependency = ValueError('dependency unavailable')
        with pytest.raises(ValueError, match='dependency unavailable'):
            item.get_factor_values()
    finally:
        item._duckdb_adapter.close()


@pytest.mark.parametrize('value', [object(), float('nan'), float('inf'), np.array([1, 2])])
def test_unsupported_declared_inputs_disable_cache_visibly(tmp_path, caplog, value):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = factor(path, tmp_path/'cache', pit_enabled=False)
    try:
        item.custom_parameter = value
        with caplog.at_level(logging.WARNING):
            assert item.get_factor_values().iloc[0,0] == 10
        assert not list((tmp_path/'cache').glob('*.pkl'))
        assert 'Cache disabled' in caplog.text
    finally:
        item._duckdb_adapter.close()


def test_class_defaults_force_recalculate_and_broken_cache(tmp_path, monkeypatch):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = factor(path, tmp_path/'cache', pit_enabled=False)
    CALLS.clear()
    try:
        item.get_factor_values(); item.get_factor_values(force_recalculate=True)
        assert len(CALLS) == 2
        monkeypatch.setattr(CachedClose, 'scale', 2.)
        assert item.get_factor_values().iloc[0,0] == 20
        assert len(CALLS) == 3
        key = item._get_cache_key(item._get_raw_data_for_factor())
        (tmp_path/'cache'/f'{key}.pkl').write_bytes(b'broken pickle')
        assert item.get_factor_values().iloc[0,0] == 20
        assert len(CALLS) == 4
    finally:
        item._duckdb_adapter.close()


def test_source_changes_invalidate_and_new_keys_do_not_use_v2(tmp_path, monkeypatch):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = factor(path, tmp_path/'cache', pit_enabled=False)
    CALLS.clear()
    original = inspect.getsourcefile
    source = tmp_path/'factor_source.py'; source.write_text('version one')
    monkeypatch.setattr(inspect, 'getsourcefile', lambda cls: str(source) if cls is CachedClose else original(cls))
    try:
        key = item._get_cache_key(item._get_raw_data_for_factor())
        assert key.startswith('factor-v3-')
        with (tmp_path/'cache'/(key.replace('factor-v3-', 'factor-v2-')+'.pkl')).open('wb') as out:
            pickle.dump(pd.DataFrame({'A': [999]}), out)
        assert item.get_factor_values().iloc[0,0] == 10
        source.write_text('version two')
        assert item.get_factor_values().iloc[0,0] == 10
        assert len(CALLS) == 2
    finally:
        item._duckdb_adapter.close()


@pytest.mark.parametrize('change', ['index', 'dtype', 'column_order', 'value'])
def test_dataframe_identity_covers_structure_and_contents(tmp_path, change):
    item = CachedClose(None, 'frame', '2024-01-02', '2024-01-03', stock_pool=['A'], pit_enabled=False)
    first = pd.DataFrame({'date': pd.to_datetime(['2024-01-02']), 'code': ['A'], 'close': [10.]})
    second = first.copy()
    if change == 'index': second.index = pd.Index([1], name='row')
    elif change == 'dtype': second['close'] = second['close'].astype('float32')
    elif change == 'column_order': second = second[['close', 'code', 'date']]
    else: second['close'] = 20.
    assert item._get_cache_key(first) != item._get_cache_key(second)


@pytest.mark.parametrize('value', [(1, 2), {1: 'numeric key'}])
def test_non_json_parameter_shapes_are_not_silently_coerced(tmp_path, value, caplog):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = factor(path, tmp_path/'cache', pit_enabled=False)
    try:
        item.custom_parameter = value
        with caplog.at_level(logging.WARNING):
            item.get_factor_values()
        assert not list((tmp_path/'cache').glob('*.pkl'))
        assert 'Cache disabled' in caplog.text
    finally:
        item._duckdb_adapter.close()


def test_pool_membership_updates_invalidate(tmp_path):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = factor(path, tmp_path/'cache', pit_enabled=True, use_dynamic_pool=True,
                  stock_pools_by_period={'2024-01': ['A']})
    try:
        assert item.get_factor_values()['A'].tolist() == [5, 12]
        item.stock_pools_by_period = {'2024-01-02': ['A'], '2024-01-03': []}
        actual = item.get_factor_values()['A']
        assert actual.iloc[0] == 5 and pd.isna(actual.iloc[1])
    finally:
        item._duckdb_adapter.close()


def test_frame_metadata_and_categorical_order_are_inputs():
    item = CachedClose(None, 'frame', '2024-01-02', '2024-01-03', stock_pool=['A'], pit_enabled=False)
    first = pd.DataFrame({'code': pd.Categorical(['A'], categories=['A', 'B'], ordered=True)})
    different_order = pd.DataFrame({'code': pd.Categorical(['A'], categories=['B', 'A'], ordered=True)})
    assert item._get_cache_key(first) != item._get_cache_key(different_order)
    different_attrs = first.copy(); different_attrs.attrs['adjusted'] = True
    assert item._get_cache_key(first) != item._get_cache_key(different_attrs)
    assert item._get_cache_key(first) == item._get_cache_key(first.copy())


class IncompleteClose(CachedClose):
    declaration = None

    def _get_cache_inputs(self):
        return self.declaration


@pytest.mark.parametrize('declaration', [None, [], {1: 'not a string key'}])
def test_incomplete_declaration_disables_only_the_cache(tmp_path, caplog, declaration):
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = IncompleteClose(None, 'incomplete', '2024-01-02', '2024-01-03',
                           stock_pool=['A'], duckdb_path=str(path), cache_dir=str(tmp_path/'cache'), pit_enabled=False)
    item.declaration = declaration
    CALLS.clear()
    try:
        with caplog.at_level(logging.WARNING):
            assert item.get_factor_values().iloc[0,0] == 10
            assert item.get_factor_values().iloc[0,0] == 10
        assert len(CALLS) == 2
        assert not list((tmp_path/'cache').glob('*.pkl'))
        assert 'Cache disabled' in caplog.text
    finally:
        item._duckdb_adapter.close()


def test_shared_pit_source_change_invalidates_cache(tmp_path, monkeypatch):
    from utils.factor_base import apply_pit_adjustment
    path = tmp_path/'prices.duckdb'; make_store(path, 1)
    item = factor(path, tmp_path/'cache', pit_enabled=True)
    CALLS.clear()
    original = inspect.getsourcefile
    source = tmp_path/'pit_source.py'; source.write_text('pit version one')
    monkeypatch.setattr(inspect, 'getsourcefile', lambda obj: str(source) if obj is apply_pit_adjustment else original(obj))
    try:
        assert item.get_factor_values().iloc[0,0] == 5
        source.write_text('pit version two')
        assert item.get_factor_values().iloc[0,0] == 5
        assert len(CALLS) == 2
    finally:
        item._duckdb_adapter.close()


def test_factor_cache_identity_uses_effective_config_not_mutable_path(tmp_path):
    import json
    from utils.data_access_config import load_data_access_config

    db = tmp_path / 'prices.duckdb'
    make_store(db, 1)
    config = load_data_access_config().as_dict()
    path = tmp_path / 'reader.json'
    path.write_text(json.dumps(config))
    first = factor(db, tmp_path / 'cache', pit_enabled=False, data_config_path=str(path))
    second = None
    try:
        raw = first._get_raw_data_for_factor()
        first_key = first._get_cache_key(raw)
        assert first_key is not None
        config['tables']['price_data_daily']['parquet_subdir'] = 'different_directory'
        path.write_text(json.dumps(config))
        assert first._get_cache_key(raw) == first_key
        second = factor(db, tmp_path / 'cache', pit_enabled=False, data_config_path=str(path))
        assert second._get_cache_key(raw) != first_key
    finally:
        first._duckdb_adapter.close()
        if second is not None:
            second._duckdb_adapter.close()
