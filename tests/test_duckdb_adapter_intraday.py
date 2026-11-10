from __future__ import annotations

from pathlib import Path
import logging
import sys
from types import MethodType

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.duckdb_adapter import DataCache, DuckDBAdapter
from utils.factor_base import FactorBase


def test_5m_end_date_is_inclusive_without_reading_next_day(tmp_path: Path) -> None:
    db_path = tmp_path / "prices.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE price_data_5m (
            date TIMESTAMP, stock_code VARCHAR, close DOUBLE, factor DOUBLE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO price_data_5m VALUES
          ('2025-12-31 14:55:00', 'B', 2, 1),
          ('2025-12-31 09:35:00', 'A', 1, 1),
          ('2026-01-01 09:35:00', 'A', 3, 1)
        """
    )
    conn.close()

    adapter = DuckDBAdapter(str(db_path), use_cache=False)
    result = adapter.fetch_price_data(
        ["B", "A"], "2025-12-31", "2025-12-31", ["close", "factor"], "5m"
    )
    adapter.close()

    assert result[["code", "date"]].to_records(index=False).tolist() == [
        ("A", pd.Timestamp("2025-12-31 09:35:00")),
        ("B", pd.Timestamp("2025-12-31 14:55:00")),
    ]
    assert result.attrs["_sorted_by_code_date"] is True


def test_factor_base_5m_read_keeps_full_end_day(tmp_path) -> None:
    db = tmp_path / 'prices.duckdb'
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE price_data_5m(date TIMESTAMP, stock_code VARCHAR, close DOUBLE)")
        conn.execute("INSERT INTO price_data_5m VALUES ('2025-12-30 14:55', 'A', 0), "
                     "('2025-12-31 14:55', 'A', 1), ('2026-01-01 09:35', 'A', 2)")
    factor = FactorBase(None, 'IntradayBoundary', '2025-12-31', '2025-12-31',
                        stock_pool=['A'], data_frequency='5m', pit_enabled=False,
                        duckdb_path=str(db), data_bindings={})
    try:
        result = factor._fetch_raw_data_duckdb(['close'], 'price_data')
        assert result['date'].tolist() == [pd.Timestamp('2025-12-31 14:55')]
        assert result['close'].tolist() == [1.0]
    finally:
        factor._duckdb_adapter.close()


def test_factor_base_honors_non_factor_duckdb_table() -> None:
    class Adapter:
        def fetch_table_data(self, **kwargs):
            assert kwargs["table_name"] == "valuation_daily"
            return pd.DataFrame(
                {"date": pd.to_datetime(["2025-12-31"]), "code": ["A"], "capitalization": [1.0]}
            )

    factor = FactorBase.__new__(FactorBase)
    factor._duckdb_adapter = Adapter()
    factor.pit_enabled = False
    factor.pit_factor_column = "factor"
    factor.data_frequency = "daily"
    factor.all_stocks = ["A"]
    factor.start_date = factor.end_date = "2025-12-31"
    factor.use_dynamic_pool = False
    factor.dynamic_pool_mode = "factor"

    result = factor._fetch_raw_data_duckdb(["capitalization"], "valuation_daily")

    assert result["capitalization"].tolist() == [1.0]


def test_oversize_dataframe_is_not_copied_into_cache() -> None:
    cache = DataCache(max_size_mb=0.0001)
    key = ("binding", "table", (), "2020-01-01", "2020-01-02", ("value",))
    cache.put(key, pd.DataFrame({"value": range(100)}))
    assert cache.get(key) is None


def test_declared_stock_chunks_are_enabled_only_after_exact_equivalence() -> None:
    class Independent(FactorBase):
        STOCK_CHUNK_INDEPENDENT = True

    factor = Independent.__new__(Independent)
    factor.data_frequency = "5m"
    factor.all_stocks = ["E", "D", "C", "B", "A"]
    factor.logger = logging.getLogger("chunk-test")

    def independent(self, force_recalculate=False):
        del force_recalculate
        self.last_performance_metrics = {}
        return pd.DataFrame(
            {code: [float(ord(code))] for code in self.all_stocks},
            index=pd.to_datetime(["2024-01-02"]),
        )

    factor.get_factor_values_unfiltered = MethodType(independent, factor)
    result = factor.get_factor_values_unfiltered_chunked(2)

    assert result.columns.tolist() == ["A", "B", "C", "D", "E"]
    assert factor.last_performance_metrics["stock_chunk_mode"] == "verified"
    assert factor.all_stocks == ["E", "D", "C", "B", "A"]


def test_cross_sectional_factor_falls_back_to_full_universe() -> None:
    factor = FactorBase.__new__(FactorBase)
    factor.data_frequency = "5m"
    factor.all_stocks = ["A", "B", "C", "D", "E"]
    factor.logger = logging.getLogger("chunk-fallback-test")

    def cross_sectional(self, force_recalculate=False):
        del force_recalculate
        self.last_performance_metrics = {}
        return pd.DataFrame(
            {code: [float(len(self.all_stocks))] for code in self.all_stocks},
            index=pd.to_datetime(["2024-01-02"]),
        )

    factor.get_factor_values_unfiltered = MethodType(cross_sectional, factor)
    result = factor.get_factor_values_unfiltered_chunked(2)

    assert result.iloc[0].tolist() == [5.0] * 5
    assert factor.last_performance_metrics["stock_chunk_mode"] == "fallback_full"
    assert factor.all_stocks == ["A", "B", "C", "D", "E"]


class _StockChunkProbe(FactorBase):
    def _get_raw_data_for_factor(self):
        self.fetch_universes.append(tuple(self.all_stocks))
        return pd.DataFrame({'date': pd.to_datetime(['2024-01-02'] * len(self.all_stocks)),
                             'code': self.all_stocks,
                             'close': [20. if code == 'E' else 10. for code in self.all_stocks]})

    def calculate_factor(self, data):
        return data.pivot(index='date', columns='code', values='close')


class _IndependentStockChunkProbe(_StockChunkProbe):
    STOCK_CHUNK_INDEPENDENT = True


@pytest.mark.parametrize('parent', [_StockChunkProbe, _IndependentStockChunkProbe])
def test_equal_probe_stocks_cannot_authorize_cross_section_chunks(parent):
    class CrossSection(parent):
        # An inherited declaration must not authorize this changed formula.
        def calculate_factor(self, data):
            wide = super().calculate_factor(data)
            return wide.sub(wide.mean(axis=1), axis=0)

    factor = CrossSection(None, 'review', '2024-01-02', '2024-01-02',
                          stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    expected = factor.get_factor_values_unfiltered()
    factor.fetch_universes.clear()
    actual = factor.get_factor_values_unfiltered_chunked(2)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.iloc[0].tolist() == [-2., -2., -2., -2., 8.]
    assert factor.fetch_universes == [tuple('ABCDE')]
    assert factor.last_performance_metrics['stock_chunk_mode'] == 'fallback_full'
    assert factor.all_stocks == list('ABCDE')


@pytest.mark.parametrize('declaration', [False, 1, 'true', None])
def test_stock_chunk_opt_in_requires_explicit_true(declaration):
    class Probe(_StockChunkProbe):
        STOCK_CHUNK_INDEPENDENT = declaration
    factor = Probe(None, 'review', '2024-01-02', '2024-01-02',
                   stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    factor.get_factor_values_unfiltered_chunked(2)
    assert factor.fetch_universes == [tuple('ABCDE')]
    assert factor.last_performance_metrics['stock_chunk_mode'] == 'fallback_full'


def test_explicit_independent_factor_keeps_exact_stock_chunks():
    factor = _IndependentStockChunkProbe(None, 'review', '2024-01-02', '2024-01-02',
                stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    expected = factor.get_factor_values_unfiltered()
    factor.fetch_universes.clear()
    actual = factor.get_factor_values_unfiltered_chunked(2)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert [len(stocks) for stocks in factor.fetch_universes] == [4, 2, 2, 1]
    assert factor.last_performance_metrics['stock_chunk_mode'] == 'verified'
    assert factor.all_stocks == list('ABCDE')


@pytest.mark.parametrize('records_metrics', [True, False])
def test_overridden_unfiltered_entry_is_not_implicitly_time_chunked(records_metrics):
    class Override(_StockChunkProbe):
        STOCK_CHUNK_INDEPENDENT = True
        def get_factor_values_unfiltered(self, force_recalculate=False):
            if records_metrics:
                return super().get_factor_values_unfiltered(force_recalculate)
            return self.calculate_factor(self._get_raw_data_for_factor())
    factor = Override(None, 'review', '2024-01-02', '2024-01-02',
                      stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    factor.get_factor_values_unfiltered_chunked(2)
    assert factor.fetch_universes == [tuple('ABCDE')]
    assert factor.last_performance_metrics['stock_chunk_mode'] == 'fallback_full'


@pytest.mark.parametrize('stage', ['probe_full', 'probe_chunk', 'comparison', 'remaining'])
@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt, SystemExit])
def test_stock_chunk_failure_restores_original_universe(stage, failure_type, monkeypatch):
    factor = _IndependentStockChunkProbe(None, 'review', '2024-01-02', '2024-01-02',
                 stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    original_stocks = factor.all_stocks
    failure = failure_type('simulated interruption')
    fetch = factor._get_raw_data_for_factor
    target = {'probe_full': tuple('ABCD'), 'probe_chunk': tuple('AB'), 'remaining': ('E',)}
    def interrupted_fetch():
        if tuple(factor.all_stocks) == target.get(stage):
            raise failure
        return fetch()
    monkeypatch.setattr(factor, '_get_raw_data_for_factor', interrupted_fetch)
    if stage == 'comparison':
        def interrupted_comparison(*args, **kwargs):
            raise failure
        monkeypatch.setattr(pd.testing, 'assert_frame_equal', interrupted_comparison)
    with pytest.raises(failure_type) as caught:
        factor.get_factor_values_unfiltered_chunked(2)
    assert caught.value is failure
    assert factor.all_stocks is original_stocks
    assert factor.all_stocks == list('ABCDE')


@pytest.mark.parametrize('failure_type', [None, RuntimeError, KeyboardInterrupt])
def test_stock_probe_fallback_receives_full_universe_and_preserves_errors(failure_type, monkeypatch):
    factor = _IndependentStockChunkProbe(None, 'review', '2024-01-02', '2024-01-02',
                 stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    original_stocks = factor.all_stocks
    expected = factor.get_factor_values_unfiltered()
    def mismatch(*args, **kwargs):
        raise AssertionError('simulated mismatch')
    monkeypatch.setattr(pd.testing, 'assert_frame_equal', mismatch)
    failure = failure_type('fallback failed') if failure_type else None
    original_compute = factor.get_factor_values_unfiltered
    def compute(force_recalculate=False):
        if factor.all_stocks is original_stocks:
            assert force_recalculate is True
            if failure is not None: raise failure
            return expected
        return original_compute(force_recalculate=force_recalculate)
    monkeypatch.setattr(factor, 'get_factor_values_unfiltered', compute)
    if failure_type:
        with pytest.raises(failure_type) as caught:
            factor.get_factor_values_unfiltered_chunked(2)
        assert caught.value is failure
    else:
        assert factor.get_factor_values_unfiltered_chunked(2) is expected
        assert factor.last_performance_metrics['stock_chunk_mode'] == 'fallback_full'
    assert factor.all_stocks is original_stocks


@pytest.mark.parametrize("minimum", [0, 4, 5, 6])
def test_prepared_chunks_apply_minimum_to_total_input(minimum):
    factor = _StockChunkProbe(None, 'minimum', '2024-01-02', '2024-01-02',
        stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    expected = factor.get_factor_values_unfiltered()
    original_stocks = factor.all_stocks
    calls = []
    def calculate(data):
        calls.append(len(data))
        return factor.calculate_factor(data)
    actual = factor._get_factor_values_from_prepared_stock_chunks(
        2, lambda data: data, calculate, min_total_rows=minimum)
    if minimum <= 5:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert calls == [5]
    else:
        assert actual.empty and calls == []
    assert factor.all_stocks is original_stocks


def test_prepared_calculation_sees_complete_original_universe():
    factor = _StockChunkProbe(None, 'universe', '2024-01-02', '2024-01-02',
        stock_pool=list('EDCBA'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    original_stocks = factor.all_stocks
    def calculate(data):
        assert factor.all_stocks is original_stocks
        assert not factor.use_dynamic_pool
        return factor.calculate_factor(data).reindex(columns=factor.all_stocks)
    actual = factor._get_factor_values_from_prepared_stock_chunks(2, lambda data: data, calculate)
    assert actual.columns.tolist() == original_stocks
    assert actual.notna().all().all()
    assert factor.all_stocks is original_stocks


@pytest.mark.parametrize('empty_mode', ['raw', 'prepared', 'minimum'])
def test_prepared_empty_run_replaces_previous_metrics(empty_mode):
    factor = _StockChunkProbe(None, 'metrics', '2024-01-02', '2024-01-02',
        stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=False, data_bindings={})
    factor.fetch_universes = []
    factor._get_factor_values_from_prepared_stock_chunks(2, lambda data: data, factor.calculate_factor)
    factor.last_performance_metrics['previous_run_only'] = True
    prepare = lambda data: data
    if empty_mode == 'raw': factor._get_raw_data_for_factor = lambda: pd.DataFrame()
    if empty_mode == 'prepared': prepare = lambda data: pd.DataFrame()
    def unexpected(data): raise AssertionError('empty run must not calculate')
    actual = factor._get_factor_values_from_prepared_stock_chunks(2, prepare, unexpected,
        min_total_rows=6 if empty_mode == 'minimum' else 0)
    metrics = factor.last_performance_metrics
    assert actual.empty
    assert 'previous_run_only' not in metrics
    assert metrics['raw_rows'] == (0 if empty_mode == 'raw' else 5)
    assert metrics['prepared_rows'] == (5 if empty_mode == 'minimum' else 0)
    assert metrics['calculate_seconds'] == 0
    assert metrics['total_seconds'] > 0


@pytest.mark.parametrize('stage', ['raw_fetch', 'pit', 'prepare', 'calculate'])
@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
def test_prepared_failure_keeps_current_metrics_and_restores_state(stage, failure_type):
    factor = _StockChunkProbe(None, 'failure', '2024-01-02', '2024-01-02',
        stock_pool=list('ABCDE'), data_frequency='5m', pit_enabled=True, data_bindings={})
    factor.fetch_universes = []
    factor.use_dynamic_pool = True
    factor.dynamic_pool_mode = 'both'
    original_stocks = factor.all_stocks
    factor.last_performance_metrics = {'previous_run_only': True, 'raw_rows': 999}
    failure = failure_type('stage failed')
    fetch = factor._get_raw_data_for_factor
    def fail(*args): raise failure
    if stage == 'raw_fetch': factor._get_raw_data_for_factor = fail
    factor._apply_pit_adjustment = fail if stage == 'pit' else lambda data: data
    prepare = fail if stage == 'prepare' else lambda data: data
    calculate = fail if stage == 'calculate' else factor.calculate_factor
    with pytest.raises(failure_type) as caught:
        factor._get_factor_values_from_prepared_stock_chunks(2, prepare, calculate)
    assert caught.value is failure
    assert factor.all_stocks is original_stocks
    assert factor.use_dynamic_pool is True and factor.dynamic_pool_mode == 'both'
    metrics = factor.last_performance_metrics
    assert 'previous_run_only' not in metrics
    assert metrics['raw_rows'] == (0 if stage == 'raw_fetch' else 5 if stage == 'calculate' else 2)
    assert metrics['prepared_rows'] == (5 if stage == 'calculate' else 0)
    assert metrics[f'{stage}_seconds'] > 0
    assert metrics['total_seconds'] >= metrics[f'{stage}_seconds']
