"""Rewrite generated symbolic factors to avoid runtime eval().

The symbolic compiler historically emitted factor files that evaluated the
stored EXPRESSION_CODE string with eval(). This script mechanically rewrites
the generated value assignment into direct calls through the local
runtime_namespace dictionary.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


ALLOWED_CALLS = frozenset(
    {
        "abs",
        "add",
        "clip",
        "cs_demean",
        "cs_quantile",
        "cs_rank",
        "cs_scale",
        "cs_zscore",
        "decay_linear",
        "delay",
        "delta",
        "div",
        "ema",
        "log",
        "mul",
        "neg",
        "neutralize_size",
        "neutralize_styles",
        "pct_change",
        "power",
        "sign",
        "sqrt",
        "sub",
        "ts_argmax",
        "ts_argmin",
        "ts_autocorr",
        "ts_beta",
        "ts_corr",
        "ts_cov",
        "ts_cv",
        "ts_iqr",
        "ts_kurt",
        "ts_mad",
        "ts_max",
        "ts_mean",
        "ts_median",
        "ts_min",
        "ts_rank",
        "ts_residual",
        "ts_skew",
        "ts_slope",
        "ts_std",
        "ts_sum",
        "ts_zscore",
        "winsorize",
    }
)


@dataclass
class RewriteResult:
    source: str
    changed: bool
    skipped_reason: str = ""


def _render_expression_node(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            raise ValueError("bare string constants are not supported")
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        rendered = _render_expression_node(node.operand)
        if rendered.startswith("-"):
            raise ValueError("nested unary minus is not supported")
        return f"-{rendered}"
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id != "features":
            raise ValueError("only features[...] subscripts are supported")
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            return f"features[{node.slice.value!r}]"
        raise ValueError("feature subscript must be a string literal")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in ALLOWED_CALLS:
            raise ValueError(f"unsupported call: {node.func.id}")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        args = ", ".join(_render_expression_node(arg) for arg in node.args)
        return f"namespace[{node.func.id!r}]({args})"
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def render_namespace_expression(expression_code: str) -> str:
    tree = ast.parse(expression_code, mode="eval")
    return _render_expression_node(tree.body)


def _expression_code_from_module(tree: ast.Module) -> str | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "EXPRESSION_CODE"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def _is_expression_code_ref(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "EXPRESSION_CODE"
    if isinstance(node, ast.Attribute) and node.attr == "EXPRESSION_CODE":
        return isinstance(node.value, ast.Name) and node.value.id == "self"
    return False


def _is_values_target(node: ast.Assign) -> bool:
    return any(
        isinstance(target, ast.Name) and target.id == "values"
        for target in node.targets
    )


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    total = 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _offset(offsets: list[int], lineno: int, col: int) -> int:
    return offsets[lineno - 1] + col


def _indent_continuation(source_segment: str, indent: str) -> str:
    lines = source_segment.splitlines()
    if len(lines) <= 1:
        return source_segment
    continuation_indent = f"{indent}    "
    return "\n".join(
        [lines[0], *(f"{continuation_indent}{line.strip()}" for line in lines[1:])]
    )


def rewrite_source(source: str) -> RewriteResult:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return RewriteResult(
            source=source, changed=False, skipped_reason=f"syntax_error: {exc}"
        )

    expression_code = _expression_code_from_module(tree)
    if expression_code is None:
        return RewriteResult(
            source=source, changed=False, skipped_reason="missing_expression_code"
        )

    try:
        rendered_expression = render_namespace_expression(expression_code)
    except ValueError as exc:
        return RewriteResult(source=source, changed=False, skipped_reason=str(exc))

    replacements: list[tuple[int, int, str]] = []
    offsets = _line_offsets(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not _is_values_target(node):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name) or call.func.id != "eval":
            continue
        if len(call.args) != 3 or not _is_expression_code_ref(call.args[0]):
            continue

        indent = " " * node.col_offset
        namespace_arg = call.args[1]
        if isinstance(namespace_arg, ast.Name) and namespace_arg.id == "namespace":
            replacement = f"{indent}values = {rendered_expression}"
        elif isinstance(namespace_arg, ast.Call):
            namespace_source = ast.get_source_segment(source, namespace_arg)
            if not namespace_source:
                continue
            namespace_source = _indent_continuation(namespace_source, indent)
            replacement = f"{indent}namespace = {namespace_source}\n{indent}values = {rendered_expression}"
        else:
            continue

        if node.end_lineno is None or node.end_col_offset is None:
            continue
        start = _offset(offsets, node.lineno, 0)
        end = _offset(offsets, node.end_lineno, node.end_col_offset)
        replacements.append((start, end, replacement))

    if not replacements:
        return RewriteResult(
            source=source, changed=False, skipped_reason="no_matching_eval_assignment"
        )

    rewritten = source
    for start, end, replacement in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return RewriteResult(source=rewritten, changed=rewritten != source)


def rewrite_path(path: Path, *, apply: bool) -> RewriteResult:
    source = path.read_text(encoding="utf-8")
    result = rewrite_source(source)
    if apply and result.changed:
        path.write_text(result.source, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("factors/dso_mining"))
    parser.add_argument("--apply", action="store_true", help="write rewritten files")
    args = parser.parse_args()

    changed = 0
    skipped = 0
    for path in sorted(args.root.rglob("*.py")):
        result = rewrite_path(path, apply=args.apply)
        if result.changed:
            changed += 1
            print(f"{'rewrote' if args.apply else 'would_rewrite'} {path}")
        elif result.skipped_reason:
            skipped += 1
            print(f"skipped {path}: {result.skipped_reason}")

    mode = "applied" if args.apply else "dry_run"
    print(f"{mode}: changed={changed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
