"""Compatibility entrypoints for the shared factor Numba runtime."""

import ast
from pathlib import Path

from utils.numba_compat import normalize_numba_imports


def gen_conditional_jit_snippet() -> str:
    source = Path(__file__).resolve().parents[2] / "utils/numba_compat.py"
    return "NUMBA_SERIAL = True\n" + source.read_text(encoding="utf-8")


def replace_decorators_with_conditional_jit(source_code: str) -> str:
    # Legacy name; preserve decorators and normalize their imports instead.
    return ast.unparse(normalize_numba_imports(source_code, inline=True))
