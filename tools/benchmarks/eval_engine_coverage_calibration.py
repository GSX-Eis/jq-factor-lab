"""Run the frozen NERE-791 synthetic confidence-interval coverage calibration."""

import argparse
from datetime import date
from pathlib import Path

from backtest_and_analysis.evaluation.calibration import calibrate_coverage
from utils.local_outputs import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/NERE-791") / f"coverage_calibration_{date.today():%Y%m%d}.json",
    )
    args = parser.parse_args()
    report = calibrate_coverage()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    failures = sum(not cell["passes"] for cell in report["cells"])
    print(f"{args.output} ({failures}/{len(report['cells'])} cells outside the coverage gate)")


if __name__ == "__main__":
    main()
