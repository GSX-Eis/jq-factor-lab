"""Diagnose the worst date using the shared frozen alignment reader and metrics."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.alignment.compare_factors import (
    compare_single_factor, load_cloud_data, load_local_data,
)


def find_max_diff_date(factor_name, cloud_csv_path, db_path=None, *,
                       local_csv=None, bindings=None, local_factor_name=None):
    if local_csv and (db_path or bindings is not None or local_factor_name):
        raise ValueError("Choose a frozen CSV or database bindings, not both")
    local = pd.read_csv(local_csv) if local_csv else load_local_data(
        db_path, factor_name, bindings=bindings, local_factor_name=local_factor_name,
    )
    result = compare_single_factor(load_cloud_data(cloud_csv_path, factor_name), local, factor_name)
    diagnosis = result.get("worst_date_diagnosis")
    print(json.dumps({"status": result["status"], "worst_date_diagnosis": diagnosis},
                     ensure_ascii=False, indent=2))
    return diagnosis["date"] if diagnosis else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", required=True)
    parser.add_argument("--cloud-csv", required=True)
    parser.add_argument("--local-csv", help="Frozen local CSV with date,code,value columns")
    parser.add_argument("--data-bindings", help="Harness frozen bindings JSON; otherwise use QUANT_DATA_BINDINGS")
    parser.add_argument("--db", help="Optional output path; must match the binding")
    parser.add_argument("--local-factor", help="Exact local factor name including pool suffix")
    args = parser.parse_args()
    if args.local_csv and (args.db or args.data_bindings or args.local_factor):
        parser.error("--local-csv cannot be combined with database selection options")
    bindings = json.loads(Path(args.data_bindings).read_text()) if args.data_bindings else None
    find_max_diff_date(args.factor, args.cloud_csv, args.db, local_csv=args.local_csv,
                       bindings=bindings, local_factor_name=args.local_factor)


if __name__ == "__main__":
    main()
