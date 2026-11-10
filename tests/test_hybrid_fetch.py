"""Unit tests for utils.hybrid_fetch.

These tests use a lightweight fake that mimics the ``FactorBase`` fetch surface
(``start_date``/``end_date``/``data_frequency``/``_fetch_raw_data``) so they run
without any database, while still verifying that:
  * the recent fetch uses 5m frequency and a short backward-shifted start,
  * the long fetch uses daily frequency and a wide backward-shifted start,
  * the factor's original scope is always restored (even on error),
  * the trading-day -> calendar-day conversion is monotonic and widened.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from types import SimpleNamespace
from calendar_test_support import synthetic_calendar
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_fetch import (  # noqa: E402
    fetch_hybrid_price_data,
    override_fetch_scope,
    trading_days_to_calendar_days,
)


class _FakeFactor:
    """Records the fetch scope active at each ``_fetch_raw_data`` call."""

    def __init__(self) -> None:
        self.start_date = "2023-06-01"
        self.end_date = "2023-12-31"
        self.data_frequency = "daily"
        self.calls: List[dict] = []
        self._duckdb_adapter = SimpleNamespace(trading_calendar=synthetic_calendar)

    def _fetch_raw_data(self, fields: List[str], table_name: str = "price_data") -> pd.DataFrame:
        self.calls.append(
            {
                "fields": list(fields),
                "table_name": table_name,
                "start_date": self.start_date,
                "data_frequency": self.data_frequency,
            }
        )
        return pd.DataFrame({"date": [], "code": [], **{f: [] for f in fields}})


def test_trading_days_to_calendar_days_widens_and_is_monotonic() -> None:
    assert trading_days_to_calendar_days(0) == 0
    assert trading_days_to_calendar_days(-5) == 0
    # 20 trading days -> at least 20 calendar days (weekends/holidays + slack).
    small = trading_days_to_calendar_days(20)
    assert small >= 20
    # Monotonic: a longer trading window must not shrink the calendar buffer.
    assert trading_days_to_calendar_days(120) > small
    # Slack increases the buffer.
    assert trading_days_to_calendar_days(120, slack=0.5) > trading_days_to_calendar_days(120, slack=0.0)


def test_override_fetch_scope_restores_after_success() -> None:
    factor = _FakeFactor()
    with override_fetch_scope(factor, start_date="2020-01-01", data_frequency="5m"):
        assert factor.start_date == "2020-01-01"
        assert factor.data_frequency == "5m"
    assert factor.start_date == "2023-06-01"
    assert factor.data_frequency == "daily"


def test_override_fetch_scope_restores_after_error() -> None:
    factor = _FakeFactor()
    with pytest.raises(RuntimeError):
        with override_fetch_scope(factor, start_date="2020-01-01", data_frequency="5m"):
            raise RuntimeError("boom")
    # Scope must be restored even though the body raised.
    assert factor.start_date == "2023-06-01"
    assert factor.data_frequency == "daily"


def test_fetch_hybrid_price_data_splits_frequency_and_window() -> None:
    factor = _FakeFactor()
    anchor = "2023-06-01"

    intraday_df, daily_df = fetch_hybrid_price_data(
        factor,
        anchor_start_date=anchor,
        recent_trading_days=20,
        long_trading_days=120,
        intraday_fields=["close", "volume", "factor"],
        daily_fields=["close", "factor"],
    )

    assert isinstance(intraday_df, pd.DataFrame)
    assert isinstance(daily_df, pd.DataFrame)
    assert len(factor.calls) == 2

    recent_call, daily_call = factor.calls
    # First call: recent 5m window.
    assert recent_call["data_frequency"] == "5m"
    assert recent_call["fields"] == ["close", "volume", "factor"]
    # Second call: wide daily window.
    assert daily_call["data_frequency"] == "daily"
    assert daily_call["fields"] == ["close", "factor"]

    anchor_ts = pd.to_datetime(anchor)
    recent_start = pd.to_datetime(recent_call["start_date"])
    long_start = pd.to_datetime(daily_call["start_date"])

    # Both windows shift strictly backward from the anchor.
    assert recent_start < anchor_ts
    assert long_start < anchor_ts
    # The daily baseline window must reach further back than the recent 5m window.
    assert long_start < recent_start

    # Original scope restored after the helper returns.
    assert factor.start_date == "2023-06-01"
    assert factor.data_frequency == "daily"


def test_fetch_hybrid_price_data_respects_calendar_floors() -> None:
    factor = _FakeFactor()
    # Tiny windows should still honor the minimum calendar-day floors.
    fetch_hybrid_price_data(
        factor,
        anchor_start_date="2023-06-01",
        recent_trading_days=1,
        long_trading_days=1,
        intraday_fields=["close"],
        daily_fields=["close"],
        min_recent_calendar_days=10,
        min_long_calendar_days=30,
    )
    recent_call, daily_call = factor.calls
    anchor_ts = pd.to_datetime("2023-06-01")
    recent_days = (anchor_ts - pd.to_datetime(recent_call["start_date"])).days
    long_days = (anchor_ts - pd.to_datetime(daily_call["start_date"])).days
    assert recent_days >= 10
    assert long_days >= 30


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
