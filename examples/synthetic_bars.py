"""Create a deterministic, one-instrument minute panel for offline evaluation."""

from pathlib import Path
from typing import Any


def write_demo_bars(path: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    steps = np.arange(40)
    close = 100 + steps / 10 + np.sin(steps / 3)
    pd.DataFrame({
        "timestamp_utc": pd.date_range("2026-12-01T14:30Z", periods=len(steps), freq="1min"),
        "instrument_id": "DEMO", "frequency": "1min",
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 100, "source": "SYNTHETIC", "data_version": "demo-v1",
        "factor_value": np.cos(steps / 3),
    }).to_csv(path, index=False)
    return {
        "bars_csv": str(path.resolve()), "analysis_mode": "time_series", "calendar": "NYSE",
        "start_date": "2026-12-01", "end_date": "2026-12-01", "horizons": [1, 2, 5],
    }
