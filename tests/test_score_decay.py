from __future__ import annotations

import pandas as pd
import pytest

from factor_synthesis.vectorbt_adapter import benchmark_suite
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner


def _runner(score_decay=None):
    return VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 10, "score_decay": score_decay or {}}}
    )


def test_score_decay_is_identity_when_disabled():
    runner = _runner()
    scores = pd.Series({"A": 1.0, "B": -1.0})
    pd.testing.assert_series_equal(
        runner._apply_score_decay(scores, "2024-01-01"), scores
    )


def test_score_decay_uses_calendar_half_life_and_current_universe():
    runner = _runner({"enabled": True, "half_life_days": 7})
    runner._apply_score_decay(pd.Series({"A": 1.0, "B": 0.0}), "2024-01-01")
    actual = runner._apply_score_decay(
        pd.Series({"A": 0.0, "C": 2.0}), "2024-01-08"
    )
    pd.testing.assert_series_equal(actual, pd.Series({"A": 0.5, "C": 2.0}))


def test_score_decay_rejects_invalid_half_life():
    with pytest.raises(ValueError, match="half_life_days"):
        _runner({"enabled": True, "half_life_days": 0})


def test_score_decay_defaults_to_disabled_and_seven_days_when_enabled():
    disabled = _runner({"enabled": False, "half_life_days": "ignored"})
    assert disabled.score_decay_enabled is False

    enabled = _runner({"enabled": True})
    assert enabled.score_decay_half_life_days == 7.0


def test_benchmarks_do_not_inherit_candidate_score_decay(monkeypatch, tmp_path):
    captured = {}

    class StubRunner:
        def __init__(self, config, **_kwargs):
            captured.update(config)

        def run_backtest(self, **_kwargs):
            return {}

        def close(self):
            return None

    monkeypatch.setattr(benchmark_suite, "VectorBTBacktestRunner", StubRunner)
    config = {
        "top_k_strategy": {
            "score_decay": {"enabled": True, "half_life_days": 7}
        }
    }
    panel = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-01"]))
    benchmark_suite._run_one(
        config, panel, panel, None, None, {"2024-01-01": ["A"]}
    )
    assert "score_decay" not in captured["top_k_strategy"]

    common = {
        "cache_dir": tmp_path,
        "price_df": panel,
        "open_df": None,
        "factor_ratio_df": None,
        "dynamic_pools": {"2024-01-01": ["A"]},
        "factor1_panel": panel,
        "rank_average_score": None,
        "quality_score": None,
        "candidate_is_rank_average": False,
        "random_seeds": 1,
        "rebalance_dates": list(panel.index),
    }
    with_decay = benchmark_suite.fixed_benchmark_cache_path(
        base_config=config, **common
    )
    without_decay = benchmark_suite.fixed_benchmark_cache_path(
        base_config={"top_k_strategy": {}}, **common
    )
    assert with_decay == without_decay
