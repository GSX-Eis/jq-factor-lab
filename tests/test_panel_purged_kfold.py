import numpy as np
import pandas as pd

from factor_synthesis.qlib_adapter.panel_purged_kfold import PanelPurgedKFold


def test_weekly_panel_embargo_counts_trading_days_not_rows():
    dates = pd.date_range("2024-01-01", periods=36, freq="W-MON")
    splits = list(
        PanelPurgedKFold(n_splits=5, embargo_td=5).split(
            np.zeros((len(dates), 1)),
            groups=dates.to_numpy(),
            label_exit_dates=(dates + pd.Timedelta(days=7)).to_numpy(),
        )
    )

    assert len(splits) == 5
    assert all(len(validation) >= 4 for _, validation in splits)
    for train, validation in splits:
        assert np.busday_count(
            np.datetime64(dates[train[-1]].date()),
            np.datetime64(dates[validation[0]].date()),
        ) > 5
