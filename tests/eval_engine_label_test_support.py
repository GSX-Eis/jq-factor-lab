"""Shared synthetic bar frame builders for evaluation-engine label tests."""
from __future__ import annotations

import pandas as pd


def _bars(times: pd.DatetimeIndex, frequency: str = '1min') -> pd.DataFrame:
    return pd.DataFrame(dict(
        timestamp_utc=times, instrument_id='SYNTHETIC', frequency=frequency,
        open=100., high=110., low=90., close=105., volume=1.,
        source='SYNTHETIC', data_version='NERE-792', factor_value=1.,
    ))
