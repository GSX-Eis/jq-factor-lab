import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from factor_synthesis.vectorbt_adapter import benchmark_suite
from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner
from factor_synthesis.vectorbt_adapter.factor_loader import (
    cross_sectional_raw_zscore,
    cross_sectional_zscore,
)
from run_factor_synthesis import (
    FactorSynthesizer,
    _lasso_bundle_blend_ratio,
    _pure_blend_fallback_factor,
)


class _Portfolio:
    def __init__(self, values):
        self._values = pd.Series(
            values, index=pd.date_range("2023-01-01", periods=len(values))
        )

    def returns(self):
        return self._values


def test_serving_raw_zscore_does_not_winsorize():
    values = pd.Series([*range(98), 500.0, 1000.0])

    raw = cross_sectional_raw_zscore(values)
    winsorized = cross_sectional_zscore(values)

    assert winsorized.iloc[-1] == pytest.approx(winsorized.iloc[-2])
    assert raw.equals((values - values.mean()) / values.std())


def test_pure_blend_bundle_resolves_exact_parent(tmp_path):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        """
        {"index":{"models":[
          {"blend_only":true,"weights":{},"blend_with_original_ratio":1,
           "blend_weights":{"Factor1":1}},
          {"blend_only":true,"weights":{},"blend_with_original_ratio":1,
           "blend_weights":{"Factor1":1}}
        ]}}
        """,
        encoding="utf-8",
    )

    assert _pure_blend_fallback_factor(str(bundle)) == "Factor1"


def test_lasso_bundle_uses_window_selected_static_blend_ratio():
    ratio, blend_info = _lasso_bundle_blend_ratio(
        {
            "selected_blend_ratio": 2 / 3,
            "blend_info": {
                "dynamic_blend_enabled": False,
                "blend_ratio_used": 2 / 3,
            },
        },
        configured_ratio=1.0,
    )

    assert ratio == pytest.approx(2 / 3)
    assert blend_info["blend_ratio_used"] == pytest.approx(2 / 3)


def test_fixed_local_reranker_profile_appends_harness_summary(tmp_path, monkeypatch):
    synth = FactorSynthesizer.__new__(FactorSynthesizer)
    synth.config = {
        "performance_evaluation": {
            "top_k_strategy": {
                "top_k": 10,
                "local_reranker_profile": "en_top30_binary_lambdarank_v1",
            }
        },
        "synthesis_methods": {
            "w11": {"method": "lasso", "model_type": "elasticnet"}
        },
        "output_settings": {
            "diagnostics_output_dir": str(tmp_path / "run" / "execution")
        },
    }
    synth._vectorbt_result_summaries = []
    synth.logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "harness_summary": {
                        "sharpe_ratio": 1.81,
                        "annual_return": 0.69,
                    },
                    "design": {"prediction_start": "2023-03-15"},
                }
            ),
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("run_factor_synthesis.subprocess.run", fake_run)
    synth._run_local_reranker_profile()

    assert synth._vectorbt_result_summaries[-1]["local_reranker_profile"] == (
        "en_top30_binary_lambdarank_v1"
    )
    assert synth._vectorbt_result_summaries[-1]["sharpe_ratio"] == 1.81


def test_input_factor_reserves_cash_for_fees_and_slippage():
    dates = pd.date_range("2023-03-01", periods=3, freq="B")
    stocks = ["A", "B"]
    scores = pd.DataFrame([[2.0, 1.0]] * 3, index=dates, columns=stocks)
    prices = pd.DataFrame(10.0, index=dates, columns=stocks)
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "enable_tday_filter": False,
                "cost_rate": 0.01,
                "slippage": 0.02,
                "init_cash": 1_000.0,
            }
        }
    )

    weights = runner._compute_target_weights(
        scores, prices, {}, [dates[1]], open_df=prices
    )

    assert weights.loc[dates[1]].sum() == pytest.approx(1_000.0 / 1.03)


def test_pit_adjusted_prices_do_not_apply_corporate_action_ratio_twice():
    dates = pd.date_range("2023-03-01", periods=4, freq="B")
    scores = pd.DataFrame(
        [[2.0, 1.0], [2.0, 1.0], [1.0, 2.0], [1.0, 2.0]],
        index=dates,
        columns=["A", "B"],
    )
    prices = pd.DataFrame(10.0, index=dates, columns=["A", "B"])
    ratios = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    ratios.loc[dates[2], "A"] = 2.0
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "score_source": "input_factor",
                            "enable_tday_filter": False, "init_cash": 1_000.0}}
    )

    baseline = runner._compute_target_weights(
        scores, prices, {}, [dates[1], dates[3]], open_df=prices
    )
    with_ratio = runner._compute_target_weights(
        scores, prices, {}, [dates[1], dates[3]], open_df=prices,
        factor_ratio_df=ratios,
    )

    pd.testing.assert_frame_equal(with_ratio, baseline)
    assert with_ratio.loc[dates[3], "B"] > 0


def test_next_open_tradeability_uses_open_not_later_prices():
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE price_data_daily "
        "(stock_code VARCHAR, date DATE, open DOUBLE, close DOUBLE, "
        "high_limit DOUBLE, low_limit DOUBLE, paused INTEGER)"
    )
    conn.execute(
        "INSERT INTO price_data_daily VALUES "
        "('OPEN_OK', '2023-03-01', 9.0, 10.0, 10.0, 8.0, 0), "
        "('OPEN_LIMIT', '2023-03-01', 10.0, 9.0, 10.0, 8.0, 0)"
    )
    conn.execute(
        "CREATE TABLE factor_data (stock_code VARCHAR, date DATE, is_st VARCHAR)"
    )
    conn.execute(
        "INSERT INTO factor_data VALUES "
        "('OPEN_OK', '2023-03-01', 'false'), "
        "('OPEN_LIMIT', '2023-03-01', 'false')"
    )
    runner = VectorBTBacktestRunner({}, price_db_conn=conn)

    assert runner._filter_tradeable_stocks(
        "2023-03-01", ["OPEN_OK", "OPEN_LIMIT"]
    ) == ["OPEN_OK"]
    conn.close()


def test_exact_rank_average_reuses_candidate_result(monkeypatch):
    calls = []

    def fake_run_one(*args, **kwargs):
        calls.append((args, kwargs))
        result = {
            "annual_return": 0.1,
            "sharpe_ratio": 1.0,
            "portfolio": _Portfolio([0.0, 0.01]),
        }
        return result, pd.Series([0.0, 0.01])

    monkeypatch.setattr(benchmark_suite, "_run_one", fake_run_one)
    candidate = {
        "annual_return": 0.2,
        "sharpe_ratio": 1.5,
        "portfolio": _Portfolio([0.0, 0.02]),
        "selection_diagnostics": {
            "rows": [
                {"date": "2023-01-02"},
                {"date": "2023-01-09"},
            ]
        },
    }
    gross = {
        "annual_return": 0.3,
        "sharpe_ratio": 1.7,
        "portfolio": _Portfolio([0.0, 0.03]),
    }
    panel = pd.DataFrame(
        1.0, index=pd.date_range("2023-01-01", periods=2), columns=["A"]
    )

    report = benchmark_suite.run_benchmark_suite(
        base_config={"top_k_strategy": {"top_pct": 0.1}},
        price_df=panel,
        open_df=panel,
        factor_ratio_df=None,
        dynamic_pools={},
        rank_average_score=panel,
        candidate_result=candidate,
        candidate_gross_result=gross,
        candidate_is_rank_average=True,
        random_seeds=0,
    )

    references = report["benchmarks"]
    assert references["rank_average"]["net"] == references["candidate"]["net"]
    assert references["rank_average"]["gross"] == references["candidate"]["gross"]
    assert len(calls) == 2
    assert all(
        call[1]["rebalance_dates"]
        == [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-09")]
        for call in calls
    )


def test_active_metrics_are_recorded_against_declared_parent():
    dates = pd.date_range("2023-01-02", periods=4, freq="B")
    candidate = pd.Series([0.0, 0.02, -0.01, 0.03], index=dates)
    parent = pd.Series([0.0, 0.01, -0.01, 0.01], index=dates)

    active = benchmark_suite._active_metrics(candidate, parent, "factor1_top")

    assert active["benchmark"] == "factor1_top"
    assert active["comparison_start"] == "2023-01-02"
    assert active["comparison_end"] == "2023-01-05"
    assert active["observations"] == 4
    assert active["annual_active_return"] > 0
    assert active["information_ratio"] > 0


def test_equal_rank_average_config_detection():
    factors = ["A", "B", "C"]
    method = {
        "method": "weighted_sum",
        "weights": {name: 1 / 3 for name in factors},
        "preprocess": {name: "rank" for name in factors},
    }

    assert benchmark_suite.is_equal_rank_average_config(method, factors, factors)
    method["weights"]["A"] = 0.5
    assert not benchmark_suite.is_equal_rank_average_config(method, factors, factors)
