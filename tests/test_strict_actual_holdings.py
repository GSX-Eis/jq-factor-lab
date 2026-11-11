from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner


def _weights_with_one_blocked_exit(
    *, strict: bool, monkeypatch, tmp_path
) -> tuple[pd.DataFrame, VectorBTBacktestRunner]:
    monkeypatch.chdir(tmp_path)
    dates = pd.date_range("2024-03-04", periods=4, freq="B")
    scores = pd.DataFrame(
        [
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 1.0, 4.0, 3.0],
            [2.0, 1.0, 4.0, 3.0],
            [2.0, 1.0, 4.0, 3.0],
        ],
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    prices = pd.DataFrame(10.0, index=dates, columns=scores.columns)
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "enable_tday_filter": False,
                "cost_rate": 0.0,
                "slippage": 0.0,
                "init_cash": 1_000.0,
                "strict_max_actual_holdings": strict,
            }
        }
    )

    blocked_date = dates[2].strftime("%Y-%m-%d")

    def tradeable(
        date_str: str,
        stocks: list[str],
        side: str = "buy",
        require_non_st: bool = True,
    ) -> list[str]:
        del require_non_st
        if side == "sell" and date_str == blocked_date:
            return [stock for stock in stocks if stock != "A"]
        return list(stocks)

    filter_tradeable: Callable[..., list[str]] = tradeable
    monkeypatch.setattr(runner, "_filter_tradeable_stocks", filter_tradeable)
    weights = runner._compute_target_weights(
        scores,
        prices,
        {},
        [dates[1], dates[2], dates[3]],
        open_df=prices,
    )
    return weights, runner


def test_strict_limit_defaults_off() -> None:
    runner = VectorBTBacktestRunner({"top_k_strategy": {"top_k": 2}})
    assert runner.strict_max_actual_holdings is False


def test_blocked_exit_consumes_slot_until_later_rebalance(
    monkeypatch, tmp_path
) -> None:
    weights, runner = _weights_with_one_blocked_exit(
        strict=True, monkeypatch=monkeypatch, tmp_path=tmp_path
    )
    dates = weights.index

    # A cannot exit on the second rebalance. B exits, so only the top-ranked
    # replacement C may enter; D is deferred instead of creating 3 holdings.
    assert pd.isna(weights.loc[dates[2], "A"])
    assert weights.loc[dates[2], "B"] == 0.0
    assert weights.loc[dates[2], "C"] > 0.0
    assert pd.isna(weights.loc[dates[2], "D"])

    # Once A becomes sellable, its slot is released and D enters on the next
    # scheduled rebalance. The skip counter records the deferred entry.
    assert weights.loc[dates[3], "A"] == 0.0
    assert weights.loc[dates[3], "D"] > 0.0
    assert runner._execution_diagnostics["strict_position_buys_skipped"] == 1
    assert runner._execution_diagnostics["strict_max_actual_holdings"] is True


def test_legacy_mode_still_buys_all_targets_after_blocked_exit(
    monkeypatch, tmp_path
) -> None:
    weights, runner = _weights_with_one_blocked_exit(
        strict=False, monkeypatch=monkeypatch, tmp_path=tmp_path
    )
    blocked_rebalance = weights.index[2]

    assert weights.loc[blocked_rebalance, "C"] > 0.0
    assert weights.loc[blocked_rebalance, "D"] > 0.0
    assert runner._execution_diagnostics["strict_position_buys_skipped"] == 0


def test_strict_limit_does_not_change_skip_liquidation(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    dates = pd.to_datetime(["2024-01-12", "2024-01-15", "2024-01-16", "2024-01-17"])
    scores = pd.DataFrame([[2.0, 1.0]] * len(dates), index=dates, columns=["A", "B"])
    prices = pd.DataFrame(10.0, index=dates, columns=scores.columns)
    open_prices = prices.copy()
    open_prices.loc[dates[2], "A"] = float("nan")
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "enable_tday_filter": False,
                "cost_rate": 0.0,
                "strict_max_actual_holdings": True,
            },
            "calendar_filters": {"skip_months": ["01-16:01-31"]},
        }
    )
    monkeypatch.setattr(
        runner,
        "_filter_tradeable_stocks",
        lambda date_str, stocks, side="buy", require_non_st=True: list(stocks),
    )

    weights = runner._compute_target_weights(
        scores,
        prices,
        {},
        [dates[1], dates[2], dates[3]],
        open_df=open_prices,
    )

    assert pd.isna(weights.loc[dates[2], "A"])
    assert weights.loc[dates[2], "B"] == 0.0
    assert (weights.loc[dates[3], ["A", "B"]] == 0.0).all()


def test_missing_next_open_does_not_release_strict_slot(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    dates = pd.date_range("2024-03-04", periods=4, freq="B")
    scores = pd.DataFrame(
        [
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 1.0, 4.0, 3.0],
            [2.0, 1.0, 4.0, 3.0],
            [2.0, 1.0, 4.0, 3.0],
        ],
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    close_prices = pd.DataFrame(10.0, index=dates, columns=scores.columns)
    open_prices = close_prices.copy()
    open_prices.loc[dates[2], "A"] = float("nan")
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "enable_tday_filter": False,
                "cost_rate": 0.0,
                "strict_max_actual_holdings": True,
            }
        }
    )
    monkeypatch.setattr(
        runner,
        "_filter_tradeable_stocks",
        lambda date_str, stocks, side="buy", require_non_st=True: list(stocks),
    )

    weights = runner._compute_target_weights(
        scores,
        close_prices,
        {},
        [dates[1], dates[2], dates[3]],
        open_df=open_prices,
    )

    assert pd.isna(weights.loc[dates[2], "A"])
    assert weights.loc[dates[2], "B"] == 0.0
    assert weights.loc[dates[2], "C"] > 0.0
    assert pd.isna(weights.loc[dates[2], "D"])
    assert weights.loc[dates[3], "A"] == 0.0
    assert weights.loc[dates[3], "D"] > 0.0
