"""Approved A260 contracts: physical columns, read guards and synthesis state."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import duckdb
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from factor_synthesis.composite_factor_base import CompositeFactorBase
from factor_synthesis.factor_reader import FactorReader
from utils.analysis_results_guard import assert_no_legacy_pooled_results_duckdb
from utils.duckdb_results_writer import (
    CommittedParquetExportError, DuckDBResultsWriter, connect_read_only_with_retry,
)


def test_monthly_pool_reordered_columns(tmp_path):
    path = tmp_path / 'pool.duckdb'
    writer = DuckDBResultsWriter(str(path), export_parquet=False)
    ic = pd.DataFrame({'1D': [.1, .2, .4]}, index=pd.bdate_range('2020-01-02', periods=3))
    expected = writer._calculate_monthly_metrics(ic)
    writer.write_monthly_pool_results('f', {'ic': ic})
    with duckdb.connect(str(path)) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info(monthly_pool_results)').fetchall()]
        cols[4], cols[5] = cols[5], cols[4]
        conn.execute('CREATE TABLE reordered AS SELECT ' + ','.join(cols) + ' FROM monthly_pool_results WHERE false')
        conn.execute('DROP TABLE monthly_pool_results')
        conn.execute('ALTER TABLE reordered RENAME TO monthly_pool_results')
    writer.write_monthly_pool_results('f', {'ic': ic})
    with duckdb.connect(str(path), read_only=True) as conn:
        got = conn.execute('SELECT monthly_ic,monthly_ir FROM monthly_pool_results').fetchdf()
    assert_frame_equal(got, expected[['monthly_ic', 'monthly_ir']], check_exact=True)


def test_guard_scans_beyond_first_batch():
    with duckdb.connect(':memory:') as conn:
        conn.execute('CREATE TABLE factor_values AS SELECT \'clean\'::VARCHAR AS factor_name FROM range(10001)')
        conn.execute("INSERT INTO factor_values VALUES ('alpha_pool_rules')")
        assert all(r[0] == 'clean' for r in conn.execute('SELECT * FROM factor_values LIMIT 10000').fetchall())
        with pytest.raises(RuntimeError, match='legacy pooled'):
            assert_no_legacy_pooled_results_duckdb(conn)


@pytest.mark.parametrize('ddl', [
    'CREATE TABLE MONTHLY_POOL_RESULTS(x INTEGER)',
    'CREATE TABLE FACTOR_VALUES(factor_name VARCHAR, POOL_TYPE VARCHAR)',
    'CREATE TABLE FACTOR_ANALYSIS_RESULTS(factor_name VARCHAR)',
])
def test_guard_ascii_names(ddl):
    with duckdb.connect(':memory:') as conn:
        conn.execute(ddl)
        if 'ANALYSIS' in ddl:
            conn.execute("INSERT INTO FACTOR_ANALYSIS_RESULTS VALUES ('alpha_pool_rules')")
        with pytest.raises(RuntimeError, match='legacy pooled'):
            assert_no_legacy_pooled_results_duckdb(conn)


@pytest.mark.parametrize('ddl', [
    'CREATE SCHEMA other; CREATE TABLE other.MONTHLY_POOL_RESULTS(x INTEGER)',
    'CREATE TABLE "factor_valueſ"(factor_name VARCHAR)',
    'CREATE TABLE factor_values(factor_name VARCHAR, "pool_typK" VARCHAR)',
])
def test_guard_keeps_schema_and_unicode_boundaries(ddl):
    with duckdb.connect(':memory:') as conn:
        conn.execute(ddl)
        assert assert_no_legacy_pooled_results_duckdb(conn) is None


def create_values(path, table='factor_values'):
    with duckdb.connect(str(path)) as conn:
        conn.execute(f'CREATE TABLE {table}(date DATE, stock_code VARCHAR, factor_name VARCHAR, factor_value DOUBLE)')
        conn.execute(f"INSERT INTO {table} VALUES ('2020-01-02','A','f',42)")


def test_reader_case_equivalent_table(tmp_path):
    path = tmp_path / 'upper.duckdb'
    create_values(path, 'FACTOR_VALUES')
    reader = FactorReader(str(path), data_bindings={})
    assert reader.get_available_factors() == ['f']


def test_reader_frozen_empty_bindings(tmp_path, monkeypatch):
    path = tmp_path / 'values.duckdb'
    create_values(path)
    monkeypatch.setenv('QUANT_DATA_BINDINGS', '{}')
    reader = FactorReader(str(path), data_bindings={})
    monkeypatch.setenv('QUANT_DATA_BINDINGS', 'invalid-json')
    assert reader.get_available_factors() == ['f']
    assert FactorReader(str(path), data_bindings={}).get_available_factors() == ['f']


def test_retry_forwards_empty_bindings_and_preserves_wait(monkeypatch):
    import utils.data_store as store
    import utils.duckdb_results_writer as writer
    expected = object()
    connect = Mock(side_effect=[RuntimeError('Could not set lock on file'), expected])
    wait = Mock()
    monkeypatch.setattr(store, 'connect_read_only', connect)
    monkeypatch.setattr(writer.time, 'sleep', wait)
    assert connect_read_only_with_retry('unused', .125, bindings={}) is expected
    assert connect.call_count == 2
    connect.assert_called_with('unused', bindings={})
    wait.assert_called_once_with(.125)


def synthesis_state(old=False):
    candidate = pd.DataFrame({'A': [2.]}, index=pd.to_datetime(['2020-01-02']))
    state = SimpleNamespace(
        logger=logging.getLogger(__name__), factor_name='blend',
        input_factor_names=['f'], start_date='2020-01-02', end_date='2020-01-02', all_stocks=['A'],
        _synthesized_factor_df=candidate / 2 if old else None, _synthesis_completed=old,
        _get_target_returns=Mock(return_value=candidate / 4),
        _target_available_at=pd.DataFrame({col:candidate.index for col in candidate},index=candidate.index),
        _trading_calendar=candidate.index,
        factor_composer=SimpleNamespace(synthesize_factors=Mock(return_value=candidate)),
        factor_reader=SimpleNamespace(save_composite_factor=Mock(return_value=True)),
    )
    state._perform_synthesis = lambda: CompositeFactorBase._perform_synthesis(state)
    return state, candidate


@pytest.mark.parametrize('old', [False, True])
def test_failed_save_preserves_candidate_and_raw_behavior(old):
    state, candidate = synthesis_state(old)
    previous = state._synthesized_factor_df
    state.factor_reader.save_composite_factor.side_effect = [RuntimeError('save failed'), True]
    assert state._perform_synthesis() is False
    assert state._synthesis_completed is old
    assert state._synthesized_factor_df is previous
    raw = CompositeFactorBase._get_raw_data_for_factor(state)
    assert raw.composite_factor.tolist() == ([1.] if old else [2.])
    assert state.factor_composer.synthesize_factors.call_count == (1 if old else 2)


def test_committed_export_error_installs_candidate_and_propagates():
    state, candidate = synthesis_state(True)
    error = CommittedParquetExportError('blend')
    state.factor_reader.save_composite_factor.side_effect = error
    with pytest.raises(CommittedParquetExportError) as caught:
        state._perform_synthesis()
    assert caught.value is error
    assert state._synthesized_factor_df is candidate and state._synthesis_completed
    assert CompositeFactorBase._get_raw_data_for_factor(state).composite_factor.tolist() == [2.]
    assert state.factor_composer.synthesize_factors.call_count == 1


@pytest.mark.parametrize('success', [True, False])
def test_normal_save_result_keeps_success_contract(success):
    state, candidate = synthesis_state()
    state.factor_reader.save_composite_factor.return_value = success
    assert state._perform_synthesis() is True
    assert state._synthesized_factor_df is candidate and state._synthesis_completed
    state.factor_reader.save_composite_factor.assert_called_once_with(
        composite_factor_df=candidate, factor_name='blend', overwrite=True)


def test_interrupt_propagates_without_retry():
    state, _ = synthesis_state(True)
    previous = state._synthesized_factor_df
    state.factor_reader.save_composite_factor.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        state._perform_synthesis()
    assert state._synthesized_factor_df is previous
    assert state.factor_reader.save_composite_factor.call_count == 1
