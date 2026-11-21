"""Materialize a target-only dumper from the canonical runtime and requirements."""

import argparse
import ast
import json
from pathlib import Path


def _replace_dates(source: str, start: str, end: str) -> str:
    dates = {"start_date": start, "end_date": end}
    tree = ast.parse(source)
    scopes = [tree.body]
    mains = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    if len(mains) == 1:
        scopes.append(mains[0].body)

    bindings = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id in dates
        and not isinstance(node.ctx, ast.Load)
    ]
    for body in scopes:
        assignments = {
            node.targets[0].id: node.value
            for node in body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and node.targets[0] in bindings
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        if assignments.keys() == dates.keys() and len(bindings) == len(dates):
            break
    else:
        raise ValueError(
            "alignment_dumper_date_assignments_required: assign start_date and end_date "
            "once using string literals, at module level or directly inside main()"
        )

    # AST columns are UTF-8 byte offsets. Change only the literal spans.
    lines = source.encode("utf-8").splitlines(keepends=True)
    for name, value in sorted(
        assignments.items(),
        key=lambda item: (item[1].lineno, item[1].col_offset),
        reverse=True,
    ):
        first, last = value.lineno - 1, value.end_lineno - 1
        lines[first : last + 1] = [
            lines[first][:value.col_offset]
            + repr(dates[name]).encode("utf-8")
            + lines[last][value.end_col_offset:]
        ]
    return b"".join(lines).decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", required=True)
    parser.add_argument("--source", required=True)
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
    start, end = "2022-01-01", "2023-12-31"
    if args.target_date:
        start = end = args.target_date
    if not "2022-01-01" <= start <= end <= "2023-12-31":
        raise ValueError("alignment_development_window_required")
    source = Path(args.source).read_text()
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
    prepared = _replace_dates("".join(lines), start, end)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prepared)
    print(path)


if __name__ == "__main__":
    main()
