import duckdb
import pandas as pd

from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner


def test_pretrade_fixed_targets_do_not_replace_or_redistribute_blocked_buy():
    dates = pd.date_range("2023-03-01", periods=3, freq="B")
    scores = pd.DataFrame(
        [
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 1.0, 4.0, 3.0],
            [2.0, 1.0, 4.0, 3.0],
        ],
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    prices = pd.DataFrame(10.0, index=dates, columns=scores.columns)
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE price_data_daily "
        "(stock_code VARCHAR, date DATE, open DOUBLE, close DOUBLE, "
        "high_limit DOUBLE, low_limit DOUBLE, paused INTEGER)"
    )
    conn.execute(
        "INSERT INTO price_data_daily VALUES "
        "('A', '2023-03-02', 9.0, 9.0, 10.0, 8.0, 0), "
        "('B', '2023-03-02', 10.0, 10.0, 10.0, 8.0, 0), "
        "('C', '2023-03-02', 9.0, 9.0, 10.0, 8.0, 0), "
        "('D', '2023-03-02', 9.0, 9.0, 10.0, 8.0, 0), "
        "('A', '2023-03-03', 8.0, 8.0, 10.0, 8.0, 0), "
        "('B', '2023-03-03', 9.0, 9.0, 10.0, 8.0, 0), "
        "('C', '2023-03-03', 9.0, 9.0, 10.0, 8.0, 0), "
        "('D', '2023-03-03', 9.0, 9.0, 10.0, 8.0, 0)"
    )
    conn.execute(
        "CREATE TABLE factor_data (stock_code VARCHAR, date DATE, is_st VARCHAR)"
    )
    conn.execute(
        "INSERT INTO factor_data VALUES "
        "('A', '2023-03-02', 'false'), "
        "('B', '2023-03-02', 'false'), "
        "('C', '2023-03-02', 'false'), "
        "('D', '2023-03-02', 'false'), "
        "('A', '2023-03-03', 'false'), "
        "('B', '2023-03-03', 'false'), "
        "('C', '2023-03-03', 'false'), "
        "('D', '2023-03-03', 'false')"
    )
    runner = VectorBTBacktestRunner(
        {
            "top_k_strategy": {
                "top_k": 2,
                "score_source": "input_factor",
                "pretrade_fixed_targets": True,
                "strict_max_actual_holdings": True,
                "cost_rate": 0.0,
                "init_cash": 1_000.0,
            }
        },
        price_db_conn=conn,
    )

    weights = runner._compute_target_weights(
        scores, prices, {}, [dates[1], dates[2]], open_df=prices
    )

    assert weights.loc[dates[1], "A"] == 500.0
    assert pd.isna(weights.loc[dates[1], "B"])
    assert pd.isna(weights.loc[dates[1], "C"])
    assert weights.loc[dates[1]].sum() == 500.0
    assert pd.isna(weights.loc[dates[2], "A"])
    assert weights.loc[dates[2], "C"] == 500.0
    assert pd.isna(weights.loc[dates[2], "D"])
    conn.close()
