"""Materialize a target-only dumper from the canonical runtime and requirements."""

import argparse
import ast
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", required=True)
    parser.add_argument("--source", default="jq_strategies/cloud_factor_dumper.py")
    parser.add_argument(
        "--output", default="outputs/alignment/prepared/cloud_factor_dumper.py"
    )
    parser.add_argument("--target-date", default="")
    parser.add_argument("--fingerprint", default="")
    args = parser.parse_args()
    requirements = json.loads(Path("config/factor_requirements.json").read_text())
    if args.factor not in requirements["factors"]:
        raise ValueError("factor_requirement_missing: " + args.factor)
    requirements["factors"] = {args.factor: requirements["factors"][args.factor]}
    period = json.loads(Path("config/small_mkt_config.json").read_text()).get(
        "analysis_period", {}
    )
    start, end = (
        period.get("start_date", "2022-01-01"),
        period.get("end_date", "2023-12-31"),
    )
    if args.target_date:
        start = end = args.target_date
    if not "2022-01-01" <= start <= end <= "2023-12-31":
        raise ValueError("alignment_development_window_required")
    source = Path(args.source).read_text()
    source = source.replace("start_date = '2022-01-01'", "start_date = " + repr(start))
    source = source.replace("end_date = '2023-12-31'", "end_date = " + repr(end))
    lines = source.splitlines(keepends=True)
    for node in reversed(ast.parse(source).body):
        if not isinstance(node, ast.Assign):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "ALIGNMENT_FINGERPRINT" in names:
            lines[node.lineno - 1 : node.end_lineno] = [
                "ALIGNMENT_FINGERPRINT = " + repr(args.fingerprint) + "\n"
            ]
        elif "REQ" in names:
            lines[node.lineno - 1 : node.end_lineno] = [
                "REQ = " + repr(requirements) + "\n"
            ]
        elif "TARGET_FACTORS" in names:
            lines[node.lineno - 1 : node.end_lineno] = [
                "TARGET_FACTORS = " + repr([args.factor]) + "\n"
            ]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    print(path)


if __name__ == "__main__":
    main()
