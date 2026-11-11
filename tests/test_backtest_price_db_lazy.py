import duckdb

from factor_synthesis.vectorbt_adapter.backtest_runner import VectorBTBacktestRunner


def test_tradeability_opens_configured_price_db_lazily(tmp_path):
    path = tmp_path / "prices.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute(
        "CREATE TABLE price_data_daily "
        "(stock_code VARCHAR, date DATE, open DOUBLE, close DOUBLE, "
        "high_limit DOUBLE, low_limit DOUBLE, paused INTEGER)"
    )
    conn.execute(
        "INSERT INTO price_data_daily VALUES "
        "('LIMIT_DOWN', '2023-03-01', 8.0, 8.0, 10.0, 8.0, 0)"
    )
    conn.execute(
        "CREATE TABLE factor_data (stock_code VARCHAR, date DATE, is_st VARCHAR)"
    )
    conn.execute(
        "INSERT INTO factor_data VALUES ('LIMIT_DOWN', '2023-03-01', 'false')"
    )
    conn.close()
    runner = VectorBTBacktestRunner({}, price_db_path=str(path))

    assert runner._filter_tradeable_stocks(
        "2023-03-01", ["LIMIT_DOWN"], side="sell", require_non_st=False
    ) == []
    runner.close()
