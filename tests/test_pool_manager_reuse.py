from datetime import datetime
import logging
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factor_synthesis.core import pool_manager  # noqa: E402
from utils.dynamic_stock_pool import DynamicStockPoolManager  # noqa: E402


def test_pool_manager_is_reused_for_the_same_price_connection(monkeypatch) -> None:
    created = []

    class FakeManager:
        def __init__(self, *args, **kwargs) -> None:
            created.append((args, kwargs))

    monkeypatch.setattr(pool_manager, "DynamicStockPoolManager", FakeManager)
    pool_manager._create_pool_manager.cache_clear()
    connection = object()

    first = pool_manager._create_pool_manager(
        connection, "factor_data", "price_data_daily", "duckdb", "/tmp/test.duckdb"
    )
    second = pool_manager._create_pool_manager(
        connection, "factor_data", "price_data_daily", "duckdb", "/tmp/test.duckdb"
    )

    assert first is second
    assert len(created) == 1
    pool_manager._create_pool_manager.cache_clear()


def test_weekly_pool_reuses_the_same_date_and_filters() -> None:
    manager = DynamicStockPoolManager.__new__(DynamicStockPoolManager)
    manager.logger = logging.getLogger("test_weekly_pool_cache")
    manager.fundamental_table = "factor_data"
    manager.price_table = "price_data_daily"
    manager._pool_cache = {}
    from utils.stock_pool_schedule import load_pool_policy
    manager.pool_policy = load_pool_policy()
    manager.cache_revision = lambda: ("fixed-test-source",)
    manager._get_available_columns = lambda: ["date", "stock_code"]
    manager._get_price_columns = lambda: []
    manager._get_previous_data_date = lambda _date: "2024-01-03"
    manager._save_filter_log = lambda _log: None
    calls = []

    def execute_sql(sql: str) -> pd.DataFrame:
        calls.append(sql)
        return pd.DataFrame({"stock_code": ["A", "B"]})

    manager._execute_sql = execute_sql
    trading_date = datetime(2024, 1, 4)

    first = manager._select_stock_pool(trading_date, [])
    second = manager._select_stock_pool(trading_date, [])

    assert first == second == ["A", "B"]
    assert len(calls) == 1
