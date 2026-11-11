import pandas as pd

from factor_synthesis.vectorbt_adapter import benchmark_suite


class _Portfolio:
    def __init__(self, values):
        self._values = pd.Series(
            values, index=pd.date_range("2023-01-01", periods=len(values))
        )

    def returns(self):
        return self._values


def test_exact_rank_average_reuses_candidate_result(monkeypatch):
    calls = []

    def fake_run_one(*args, **kwargs):
        calls.append((args, kwargs))
        returns = pd.Series([0.0, 0.01], index=pd.date_range("2023-01-01", periods=2))
        return {
            "annual_return": 0.1,
            "sharpe_ratio": 1.0,
            "portfolio": _Portfolio([0.0, 0.01]),
        }, returns

    monkeypatch.setattr(benchmark_suite, "_run_one", fake_run_one)
    candidate = {
        "annual_return": 0.2,
        "sharpe_ratio": 1.5,
        "portfolio": _Portfolio([0.0, 0.02]),
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

    assert (
        report["benchmarks"]["rank_average"]["net"]
        == report["benchmarks"]["candidate"]["net"]
    )
    assert (
        report["benchmarks"]["rank_average"]["gross"]
        == report["benchmarks"]["candidate"]["gross"]
    )
    assert len(calls) == 2  # pool EW net + gross; rank-average is not rerun


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


def test_fixed_benchmark_cache_reuses_references_but_not_candidate(
    tmp_path, monkeypatch
):
    calls = []

    def fake_run_one(*args, **kwargs):
        calls.append((args, kwargs))
        returns = pd.Series([0.0, 0.01], index=pd.date_range("2023-01-01", periods=2))
        return {
            "annual_return": 0.1,
            "sharpe_ratio": 1.0,
            "portfolio": _Portfolio([0.0, 0.01]),
        }, returns

    monkeypatch.setattr(benchmark_suite, "_run_one", fake_run_one)
    panel = pd.DataFrame(
        1.0, index=pd.date_range("2023-01-01", periods=2), columns=["A"]
    )
    cache_path = tmp_path / "fixed.pkl"
    first_candidate = {
        "annual_return": 0.2,
        "sharpe_ratio": 1.5,
        "portfolio": _Portfolio([0.0, 0.02]),
    }
    first = benchmark_suite.run_benchmark_suite(
        base_config={"top_k_strategy": {"top_k": 10}},
        price_df=panel,
        open_df=panel,
        factor_ratio_df=None,
        dynamic_pools={},
        candidate_result=first_candidate,
        random_seeds=3,
        fixed_cache_path=cache_path,
    )
    assert first["fixed_benchmark_cache"]["hit"] is False
    assert len(calls) == 5  # pool EW net/gross + three random references

    calls.clear()
    second_candidate = {
        "annual_return": 0.3,
        "sharpe_ratio": 2.0,
        "portfolio": _Portfolio([0.0, 0.03]),
    }
    second = benchmark_suite.run_benchmark_suite(
        base_config={"top_k_strategy": {"top_k": 10}},
        price_df=panel,
        open_df=panel,
        factor_ratio_df=None,
        dynamic_pools={},
        candidate_result=second_candidate,
        random_seeds=3,
        fixed_cache_path=cache_path,
    )
    assert second["fixed_benchmark_cache"]["hit"] is True
    assert calls == []
    assert second["benchmarks"]["candidate"]["net"]["sharpe_ratio"] == 2.0
    assert "active" in second["benchmarks"]["candidate"]


def test_fixed_benchmark_cache_key_ignores_pool_member_order(tmp_path):
    panel = pd.DataFrame(
        1.0, index=pd.date_range("2023-01-01", periods=2), columns=["A", "B"]
    )
    common = dict(
        cache_dir=tmp_path,
        base_config={"top_k_strategy": {"top_k": 10}},
        price_df=panel,
        open_df=panel,
        factor_ratio_df=None,
        factor1_panel=None,
        rank_average_score=None,
        quality_score=None,
        candidate_is_rank_average=False,
        random_seeds=100,
        rebalance_dates=pd.DatetimeIndex(["2023-01-04"]),
    )
    first = benchmark_suite.fixed_benchmark_cache_path(
        dynamic_pools={"2023-01-04": ["B", "A"]}, **common
    )
    second = benchmark_suite.fixed_benchmark_cache_path(
        dynamic_pools={"2023-01-04": ["A", "B"]}, **common
    )
    assert first == second
