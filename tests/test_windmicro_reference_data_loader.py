from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import duckdb


_LOADER_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline/loaders/portfolio_data_to_duckdb.py"
)
_SPEC = importlib.util.spec_from_file_location("portfolio_data_to_duckdb", _LOADER_PATH)
assert _SPEC and _SPEC.loader
_LOADER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LOADER
_SPEC.loader.exec_module(_LOADER)
DATASETS = _LOADER.DATASETS
import_dataset = _LOADER.import_dataset


def test_reference_tables_import_idempotently(tmp_path) -> None:
    samples = {
        "security_master": (
            "security_master.csv",
            "stock_code,display_name,name,start_date,end_date,security_type\n"
            "000001.XSHE,平安银行,PAYH,1991-04-03,2200-01-01,stock\n",
        ),
        "historical_st": (
            "historical_st_2024.csv",
            "stock_code,date,is_st\n000001.XSHE,2024-01-02,false\n",
        ),
        "shareholder_top10": (
            "shareholder_top10_2024.csv",
            "stock_code,pub_date,end_date,report_type,holder_scope,holder_rank,shareholder_name,"
            "shareholder_class_id,share_number,share_ratio\n"
            "000001.XSHE,2024-04-01,2024-03-31,3,total,1,示例股东,9,1000,10.0\n",
        ),
        "dividend_quality_financials": (
            "financial_events_2024.csv",
                "statement_type,source_record_id,stock_code,pub_date,start_date,end_date,report_date,report_type,source_id,net_profit,net_operate_cash_flow,total_liability,total_assets,operating_revenue,total_operating_revenue,operating_profit,total_profit,income_net_profit\n"
                "income,1,000001.XSHE,2024-04-01,2024-01-01,2024-03-31,2024-03-31,0,source-1,100,1,1,1,200,200,20,18,16\n"
                "cashflow,2,000001.XSHE,2024-04-01,2024-01-01,2024-03-31,2024-03-31,0,source-2,1,90,1,1,1,1,1,1,1\n",
        ),
    }
    conn = duckdb.connect(str(tmp_path / "test.duckdb"))
    try:
        for name, (filename, content) in samples.items():
            source = tmp_path / name
            source.mkdir()
            (source / filename).write_text(content, encoding="utf-8")
            import_dataset(conn, DATASETS[name], source, replace=False)
            import_dataset(conn, DATASETS[name], source, replace=False)

        assert conn.execute("SELECT COUNT(*) FROM security_master").fetchone()[0] == 1
        assert (
            conn.execute("SELECT COUNT(*) FROM historical_st_daily").fetchone()[0] == 1
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM shareholder_top10_events").fetchone()[0]
            == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM dividend_quality_financial_events").fetchone()[0] == 2
    finally:
        conn.close()
