from __future__ import annotations

import duckdb
from calendar_test_support import seed_calendar
import pandas as pd
import pytest

import run_factor_synthesis
from factor_synthesis.core.factor_data_loader import process_raw_factor_df
from factor_synthesis.core.factor_ops import (
    LABEL_MODE_CLOSE_TO_CLOSE,
    LABEL_MODE_NEXT_REBALANCE_CLOSE,
    LABEL_MODE_NEXT_REBALANCE_OPEN,
    calculate_scheduled_price_returns,
    resolve_label_definition,
)
from factor_synthesis.core.calendar_skip import parse_skip_rules
from factor_synthesis.pipeline.lasso_method import (
    _filter_index_by_label_exit,
    _label_dates_exiting_before_boundary,
    _resolve_embargo_trading_days,
)




@pytest.mark.parametrize('stage', ['close', 'open'])
def test_vectorbt_does_not_swallow_pit_failure(stage):
    from unittest.mock import Mock
    from factor_synthesis.core.factor_data_loader import PITValidationError

    synth = object.__new__(run_factor_synthesis.FactorSynthesizer)
    synth.config = {'analysis_period': {'start_date': '2023-01-02', 'end_date': '2023-01-03'}}
    synth.factor_duckdb_path = synth.price_duckdb_path = 'unused.duckdb'
    synth.logger = Mock()
    synth._get_price_conn = lambda: None
    wide = pd.DataFrame({'A': [100., 50.]}, index=pd.to_datetime(['2023-01-02', '2023-01-03']))
    failure = PITValidationError('PIT invalid factor')
    synth._get_complete_price_data_for_vectorbt = Mock(return_value=(wide, None))
    synth._get_open_price_data = Mock(return_value=wide)
    if stage == 'close':
        synth._get_complete_price_data_for_vectorbt.side_effect = failure
    else:
        synth._get_open_price_data.side_effect = failure
    pricing = wide.rename_axis(index='date', columns='asset').stack().swaplevel()
    with pytest.raises(PITValidationError, match='invalid factor'):
        synth._run_vectorbt_backtest(wide, 'signal', pricing)


def test_rebalance_label_contract_is_executable_and_fail_closed() -> None:
    assert resolve_label_definition({"mode": "next_rebalance_open"}) == {
        "mode": LABEL_MODE_NEXT_REBALANCE_OPEN,
        "feature_lag_trading_days": 1,
        "entry_price": "open",
        "exit_price": "next_rebalance_open",
        "embargo_trading_days": 5,
    }
    assert (
        resolve_label_definition({"mode": "next_rebalance_close"})["mode"]
        == LABEL_MODE_NEXT_REBALANCE_CLOSE
    )
    with pytest.raises(ValueError, match="feature_lag_trading_days=1"):
        resolve_label_definition(
            {"mode": "next_rebalance_open", "feature_lag_trading_days": 2}
        )
    with pytest.raises(ValueError, match="requires entry_price='open'"):
        resolve_label_definition(
            {"mode": "next_rebalance_open", "entry_price": "close"}
        )


def test_factor_panel_row_t_contains_raw_t_minus_one_value() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "stock_code": ["A", "A", "A"],
            "factor_name": ["signal", "signal", "signal"],
            "factor_value": [1.0, 2.0, 3.0],
        }
    )
    conn = duckdb.connect()
    seed_calendar(conn)
    shifted = process_raw_factor_df(
        raw,
        ["signal"],
        "2025-01-02",
        "2025-01-06",
        config={"missing_data_strategy": "drop"},
        price_conn=conn,
    )
    conn.close()
    assert shifted.loc[(pd.Timestamp("2025-01-03"), "A"), "signal"] == 1.0
    assert shifted.loc[(pd.Timestamp("2025-01-06"), "A"), "signal"] == 2.0




def test_scheduled_label_can_exit_on_forced_liquidation_date() -> None:
    dates = pd.to_datetime(["2025-01-15", "2025-01-16", "2025-01-22"])
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "stock_code"])
    factors = pd.DataFrame({"signal": [1.0, 2.0, 3.0]}, index=index)
    prices = pd.DataFrame({"A": [10.0, 9.0, 12.0]}, index=dates)
    result = calculate_scheduled_price_returns(
        factors,
        prices,
        [(dates[0], dates[1])],
    )
    assert result.loc[(dates[0], "A")] == pytest.approx(-0.1)
    assert result.attrs["label_exit_dates"] == {"2025-01-15": "2025-01-16"}


def test_label_schedule_reuses_production_skip_window_liquidation() -> None:
    synth = object.__new__(run_factor_synthesis.FactorSynthesizer)
    synth.skip_months_backtest = True
    synth.skip_months = parse_skip_rules(["01-16:02-05"])
    schedule = list(pd.to_datetime(["2025-01-15", "2025-01-22", "2025-02-12"]))
    calendar = pd.bdate_range("2025-01-15", "2025-02-12")
    pairs = synth._label_entry_exit_dates(schedule, calendar)
    assert pairs == [(pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-16"))]


def test_execution_aligned_label_can_use_a_longer_embargo_than_target_period() -> None:
    assert _resolve_embargo_trading_days({"target_period": 3}) == 3
    assert _resolve_embargo_trading_days({"target_period": 3, "_label_embargo_days": 5}) == 5


def test_explicit_exit_dates_purge_only_labels_crossing_boundary() -> None:
    dates = pd.to_datetime(["2025-01-08", "2025-01-15", "2025-01-22"])
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "stock_code"])
    exits = {
        dates[0]: dates[1],
        dates[1]: dates[2],
        dates[2]: pd.Timestamp("2025-01-29"),
    }
    kept = _filter_index_by_label_exit(index, exits, dates[1])
    assert list(kept.get_level_values("date")) == [dates[0]]

    inner_dates = _label_dates_exiting_before_boundary(dates[:2], exits, dates[1])
    assert inner_dates == []


def test_factor_synthesizer_keeps_legacy_path_unless_label_is_explicit(monkeypatch) -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-08"])
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "stock_code"])
    factors = pd.DataFrame({"signal": [1.0, 2.0]}, index=index)
    legacy = pd.Series([0.1], index=index[:1], name="forward_return")
    synth = object.__new__(run_factor_synthesis.FactorSynthesizer)
    synth.config = {}
    synth.price_table = "prices"
    synth.fundamental_table = "fundamentals"
    synth.pit_config = {"enabled": True}
    synth.logger = type("Logger", (), {"info": lambda *_args, **_kwargs: None})()
    synth._get_price_conn = lambda: object()
    monkeypatch.setattr(
        run_factor_synthesis,
        "core_calculate_forward_returns",
        lambda **_kwargs: legacy,
    )
    assert synth.calculate_forward_returns(factors, 5) is legacy

    synth.config = {"label_definition": {"mode": "next_rebalance_open"}}
    synth._label_rebalance_dates = lambda _frame: list(dates)
    synth._get_open_price_data = lambda *_args, for_label: pd.DataFrame({"A": [10.0, 12.0]}, index=dates)
    opened = synth.calculate_forward_returns(factors, 5)
    assert opened.iloc[0] == pytest.approx(0.2)
