from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VectorBTBacktestRunner = importlib.import_module(
    "factor_synthesis.vectorbt_adapter.backtest_runner"
).VectorBTBacktestRunner


class _Orders:
    records_readable = pd.DataFrame(
        [
            {
                "Order Id": 1,
                "Column": ("AAA", "AAA"),
                "Timestamp": pd.Timestamp("2024-01-02"),
                "Size": 10.0,
                "Price": 10.0,
                "Fees": 0.2,
                "Side": "Buy",
            }
        ]
    )


class _Logs:
    records_readable = pd.DataFrame(
        [
            {
                "Result Status": "Filled",
                "Request Size": 100.0,
                "Position": 10.0,
                "Val Price": 10.0,
            },
            {
                "Result Status": "Rejected",
                "Request Size": 120.0,
                "Position": 10.0,
                "Val Price": 10.0,
            },
        ]
    )


class _Portfolio:
    orders = _Orders()
    logs = _Logs()

    def value(self) -> pd.Series:
        return pd.Series([100.0, 105.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    def asset_value(self, group_by: bool = False) -> pd.DataFrame:
        assert group_by is False
        return pd.DataFrame(
            [[100.0], [105.0]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            columns=pd.MultiIndex.from_tuples([("AAA", "AAA")]),
        )


def test_vectorbt_runner_writes_portfolio_audit_artifacts(tmp_path: Path) -> None:
    runner = VectorBTBacktestRunner(
        {"top_k_strategy": {"top_k": 1, "slippage": 0.001}},
        diagnostics_output_dir=str(tmp_path),
        return_series_trial_id="trial/a",
    )
    targets = pd.DataFrame(
        [[100.0], [float("nan")]],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=["AAA"],
    )

    artifact = runner._write_portfolio_audit_artifacts(
        portfolio=_Portfolio(),
        target_values=targets,
    )

    assert artifact is not None
    manifest = json.loads(Path(artifact["manifest_path"]).read_text(encoding="utf-8"))
    orders = pd.read_csv(artifact["orders_path"])
    actual = pd.read_csv(artifact["actual_weights_path"])
    assert manifest["order_count"] == 1
    assert manifest["unfilled_order_count"] == 1
    assert manifest["unfilled_notional"] == 20.0
    assert orders.loc[0, "gross_notional"] == 100.0
    assert orders.loc[0, "modeled_price_impact"] == 0.1
    assert actual["actual_weight"].tolist() == [1.0, 1.0]
