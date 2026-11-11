from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from factor_synthesis.parent_factor_screen import (
    ParentFactorScreenError,
    run_parent_factor_screen,
)


def _screen_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    stocks = [f"{index:06d}.XSHE" for index in range(40)]
    price_dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(18)]
    base_dates = price_dates[:12]
    signal = np.linspace(-0.04, 0.04, len(stocks))
    parent_rows = []
    factor_rows = []
    for day_index, day in enumerate(base_dates):
        parent = rng.permutation(len(stocks)).astype(float)
        for stock_index, stock in enumerate(stocks):
            parent_rows.append((day, stock, parent[stock_index]))
            factor_rows.append(
                (day, stock, "incremental", signal[stock_index] + rng.normal(0.0, 0.001))
            )
            factor_rows.append((day, stock, "spanned", parent[stock_index]))

    parent_path = tmp_path / "parent.parquet"
    pd.DataFrame(
        parent_rows, columns=["date", "stock_code", "composite_value"]
    ).set_index(["date", "stock_code"]).to_parquet(parent_path)
    factor_db = tmp_path / "factors.duckdb"
    with duckdb.connect(str(factor_db)) as connection:
        connection.execute(
            "CREATE TABLE factor_values "
            "(date DATE, stock_code VARCHAR, factor_name VARCHAR, factor_value DOUBLE)"
        )
        connection.executemany("INSERT INTO factor_values VALUES (?, ?, ?, ?)", factor_rows)
    price_db = tmp_path / "prices.duckdb"
    with duckdb.connect(str(price_db)) as connection:
        connection.execute(
            "CREATE TABLE price_data_daily "
            "(date DATE, stock_code VARCHAR, close DOUBLE, factor DOUBLE)"
        )
        connection.executemany(
            "INSERT INTO price_data_daily VALUES (?, ?, ?, ?)",
            [
                (day, stock, float(np.exp(signal[stock_index] * day_index)), 1.0)
                for day_index, day in enumerate(price_dates)
                for stock_index, stock in enumerate(stocks)
            ],
        )

    job = {
        "schema_version": "algorithm_parent_factor_screen.v1",
        "pool_id": "wind_micro",
        "parent": {
            "parent_winner_id": "parent",
            "parent_run_id": "parent-run",
            "path": str(parent_path),
            "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        },
        "development": {"start_date": "2020-01-01", "end_date": "2020-01-17"},
        "target_period": 2,
        "min_stocks": 20,
        "database": {
            "factor_db_path": str(factor_db),
            "factor_table": "factor_values",
            "price_db_path": str(price_db),
            "price_table": "price_data_daily",
        },
        "candidates": [
            {"factor_id": "incremental", "source_hash": "a", "direction_by_year": {"2020": 1}},
            {"factor_id": "spanned", "source_hash": "b", "direction_by_year": {"2020": 1}},
        ],
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    return job_path, tmp_path / "evidence.json"


def test_parent_screen_finds_increment_and_exact_redundancy(tmp_path: Path) -> None:
    job_path, output_path = _screen_fixture(tmp_path)

    evidence = run_parent_factor_screen(job_path, output_path)

    factors = {item["factor_id"]: item for item in evidence["ranking"]}
    assert factors["incremental"]["classification"] == "incremental_candidate"
    assert factors["incremental"]["direction_adjusted_residual_rank_ic_mean"] > 0.9
    assert factors["spanned"]["classification"] == "redundant"
    assert factors["spanned"]["residual_rank_ic_mean"] == pytest.approx(0.0)
    assert json.loads(output_path.read_text()) == evidence


def test_parent_screen_rejects_changed_parent_score(tmp_path: Path) -> None:
    job_path, output_path = _screen_fixture(tmp_path)
    job = json.loads(job_path.read_text())
    job["parent"]["sha256"] = "changed"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    with pytest.raises(ParentFactorScreenError, match="parent score hash mismatch"):
        run_parent_factor_screen(job_path, output_path)
