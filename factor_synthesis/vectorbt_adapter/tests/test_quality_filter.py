import duckdb
import pandas as pd

from factor_synthesis.vectorbt_adapter.quality_filter import apply_pit_quality_filter


def test_pit_quality_filter_requires_only_disclosed_history(tmp_path):
    db = tmp_path / "quality.duckdb"
    conn = duckdb.connect(str(db))
    conn.execute(
        """
        create table dividend_quality_financial_events (
          statement_type varchar, stock_code varchar, pub_date date,
          start_date date, end_date date, net_profit double,
          net_operate_cash_flow double, total_liability double, total_assets double
        )
        """
    )
    conn.execute(
        """
        create table dividend_events (
          stock_code varchar, report_date date, implementation_pub_date date,
          bonus_type varchar, bonus_amount_rmb double, bonus_ratio_rmb double
        )
        """
    )
    conn.execute(
        """
        create table valuation_daily (
          stock_code varchar, date date, market_cap double
        )
        """
    )
    conn.execute(
        """
        create table industry_membership (
          stock_code varchar, date date, sw_l1_name varchar
        )
        """
    )
    rows = []
    for period in pd.period_range("2018Q1", "2022Q1", freq="Q"):
        end = period.end_time.date()
        pub = end + pd.Timedelta(days=30)
        rows.append(("income", "000001.XSHE", pub, period.start_time.date(), end, 100.0 * period.quarter, 40.0 * period.quarter, None, None))
        rows.append(("balance", "000001.XSHE", pub, period.start_time.date(), end, None, None, 500.0, 1000.0))
    conn.executemany("insert into dividend_quality_financial_events values (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.executemany(
        "insert into dividend_events values (?, ?, ?, ?, ?, ?)",
        [("000001.XSHE", f"{year}-12-31", f"{year + 1}-06-30", "年度分红", 0.02, 1.0) for year in (2019, 2020, 2021)],
    )
    conn.execute("insert into valuation_daily values ('000001.XSHE', '2022-08-01', 1000.0)")
    conn.execute("insert into industry_membership values ('000001.XSHE', '2022-08-01', '公用事业I')")
    conn.close()

    filtered, diagnostics = apply_pit_quality_filter(
        {"2022-08-01": ["000001.XSHE"]},
        db_path=str(db),
        config={"enabled": True, "min_qualified_count": 1},
    )
    assert filtered["2022-08-01"] == ["000001.XSHE"]
    assert diagnostics["qualified_counts"]["2022-08-01"] == 1

    conn = duckdb.connect(str(db))
    conn.execute(
        "update dividend_events set bonus_ratio_rmb = 0.5 where report_date = '2021-12-31'"
    )
    conn.close()
    filtered, diagnostics = apply_pit_quality_filter(
        {"2022-08-01": ["000001.XSHE"]},
        db_path=str(db),
        config={
            "enabled": True,
            "mode": "dividend_cut_veto",
            "min_comparable_dps_growth": 0.0,
            "min_qualified_count": 0,
        },
    )
    assert filtered["2022-08-01"] == []
    assert diagnostics["veto_counts"]["2022-08-01"] == 1
