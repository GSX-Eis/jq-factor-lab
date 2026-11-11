from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_analysis_main import _compute_mining_only_ic, main


def test_mining_only_ic_returns_daily_rank_ic_without_future_prices() -> None:
    dates = pd.bdate_range("2023-01-02", periods=15)
    assets = ["A", "B", "C"]
    factor = pd.DataFrame(
        [[day + offset for offset in range(3)] for day in range(len(dates))],
        index=dates,
        columns=assets,
        dtype=float,
    )
    pricing = pd.Series(
        np.concatenate([100 + np.arange(len(dates)) * slope for slope in (1.0, 2.0, 3.0)]),
        index=pd.MultiIndex.from_product([assets, dates], names=["asset", "date"]),
    )

    from calendar_test_support import synthetic_calendar
    pricing.attrs["trading_calendar"] = synthetic_calendar()
    ic = _compute_mining_only_ic(factor, pricing, (5,))

    assert list(ic.columns) == ["5D"]
    assert not ic.dropna().empty
    assert ic.index.max() <= dates.max()


def test_mining_only_rejects_post_2023_window() -> None:
    with pytest.raises(RuntimeError, match="post-2023"):
        main({"mining_only": True, "end_date": "2024-01-01"})
