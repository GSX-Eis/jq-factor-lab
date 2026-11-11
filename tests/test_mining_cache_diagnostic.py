import json
import logging
from pathlib import Path

import pandas as pd
import pytest

import run_analysis_main as runner
from utils.data_access_config import load_data_access_config


@pytest.mark.parametrize("cache_mode", ["hit", "miss", "fallback_after_hit"])
@pytest.mark.parametrize("persisted", [True, False])
def test_mining_cache_diagnostic_matches_persisted_values(
    tmp_path, monkeypatch, caplog, cache_mode, persisted
):
    # Legacy receipts may only be reused without a database binding.
    monkeypatch.delenv("QUANT_DATA_BINDINGS", raising=False)
    values = pd.DataFrame(
        {code: [1.0, 2.0, 3.0] for code in "ABCDEF"},
        index=pd.date_range("2018-01-02", periods=3),
    )
    calls = []

    class Factor:
        DATA_FREQUENCY = "5m"
        FACTOR_VALUE_CACHE_DIR = str(tmp_path)

        def __init__(self, **kwargs):
            self.monthly_stock_pools = kwargs["monthly_stock_pools"]
            self.last_performance_metrics = {"factor_value_cache_hit": "stale"}

        def get_factor_values_unfiltered_chunked(self, chunk_size):
            calls.append("raw_chunk")
            return values.copy()

        def get_factor_values(self):
            calls.append("raw_fallback")
            return values.copy()

    if cache_mode != "miss":
        source_hash = runner._sha256_file(Path(runner.inspect.getsourcefile(Factor)))
        data_path = tmp_path / f"{source_hash}.parquet"
        values.to_parquet(data_path)
        (tmp_path / f"{source_hash}.json").write_text(json.dumps({
            "schema_version": "quanta_alpha.factor_value_cache.v1",
            "data_access": load_data_access_config().as_dict(),
            "source_sha256": source_hash,
            "data_file": data_path.name,
            "data_sha256": runner._sha256_file(data_path),
            "start_date": "2018-01-01",
            "end_date": "2018-01-04",
        }))

    # The same receipt must be bypassed when input identity is database-bound.
    window = {"start_date": "2018-01-01", "end_date": "2018-01-04"}
    assert runner._load_factor_value_cache(Factor, {**window, "duckdb_path": "unused.duckdb"}) is None
    with monkeypatch.context() as scoped:
        scoped.setenv("QUANT_DATA_BINDINGS", json.dumps({"input_db": "unused.duckdb"}))
        assert runner._load_factor_value_cache(Factor, window) is None

    def pool_filter(frame, pools):
        if cache_mode == "fallback_after_hit":
            raise RuntimeError("pool-filter toy failure after cache load")
        return frame

    class PriceFetcher:
        def fetch(self, stocks, start_date, end_date, **kwargs):
            assert kwargs["future_periods"] == 0
            return values.stack().reorder_levels([1, 0]).sort_index()

    def persist(**kwargs):
        calls.append("persist")
        assert kwargs["monthly_writer"] is None
        pd.testing.assert_frame_equal(kwargs["factor_df_unfiltered"], values, check_freq=False)
        metrics = kwargs["combined_results"]["performance_metrics"]
        assert metrics["factor_value_cache_hit"] is (cache_mode == "hit")
        assert metrics["mining_only"] is True
        return persisted

    monkeypatch.setattr("utils.dynamic_stock_pool.apply_date_level_pool_filter_to_wide_df", pool_filter)
    monkeypatch.setattr(runner, "PriceFetcher", lambda *args, **kwargs: PriceFetcher())
    monkeypatch.setattr(runner, "_compute_mining_only_ic", lambda *args: pd.DataFrame({"5D": [0.1]}))
    monkeypatch.setattr(runner, "_persist_analysis_outputs", persist)
    caplog.set_level(logging.INFO, logger=runner.logger.name)
    ok = runner.analyze_factor_dynamic(
        Factor, {"name": "wind_micro", "filters": []},
        {"start_date": "2018-01-01", "end_date": "2018-01-04",
         "mining_only": True, "ic_periods": [5], "duckdb_path": None,
         "price_table": "unused"},
        object(), object(), None, {"2018-01-01": list("ABCDEF")},
    )
    assert ok is persisted
    expected = {"hit": [], "miss": ["raw_chunk"], "fallback_after_hit": ["raw_fallback"]}
    assert calls == expected[cache_mode] + ["persist"]
    diagnostics = [json.loads(record.getMessage().split("PERF_METRICS ", 1)[1])
                   for record in caplog.records if record.getMessage().startswith("PERF_METRICS ")]
    assert len(diagnostics) == int(persisted)
    if persisted:
        assert diagnostics[0]["factor_value_cache_hit"] is (cache_mode == "hit")
        assert diagnostics[0]["mining_only"] is True
