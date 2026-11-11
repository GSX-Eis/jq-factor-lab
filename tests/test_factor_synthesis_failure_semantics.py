from unittest.mock import Mock

import pytest

from run_factor_synthesis import FactorSynthesizer


def test_run_synthesis_raises_when_every_method_fails() -> None:
    synthesizer = object.__new__(FactorSynthesizer)
    synthesizer.config = {"synthesis_methods": {"demo": {"method": "lasso"}}}
    synthesizer.logger = Mock()
    synthesizer.rolling_window_synthesis = Mock(side_effect=RuntimeError("duckdb lock conflict"))
    synthesizer._run_local_reranker_profile = Mock()
    synthesizer._vectorbt_result_summaries = []

    with pytest.raises(RuntimeError, match="duckdb lock conflict"):
        synthesizer.run_synthesis()


def test_run_synthesis_fails_when_enabled_backtest_produces_no_metrics() -> None:
    synthesizer = object.__new__(FactorSynthesizer)
    synthesizer.config = {
        "synthesis_methods": {"demo": {"method": "lasso"}},
        "performance_evaluation": {"top_k_strategy": {"enabled": True}},
    }
    synthesizer.logger = Mock()
    synthesizer.rolling_window_synthesis = Mock()
    synthesizer._run_local_reranker_profile = Mock()
    synthesizer._vectorbt_result_summaries = []

    with pytest.raises(RuntimeError, match="backtest_metrics_missing"):
        synthesizer.run_synthesis()
