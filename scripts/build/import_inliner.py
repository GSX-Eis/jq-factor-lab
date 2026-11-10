import ast
import re
from pathlib import Path
from typing import List, Set, Optional

from utils.numba_compat import normalize_numba_imports


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def _remove_main_blocks(text: str) -> str:
    lines = text.split('\n')
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*$", line):
            base_indent = len(line) - len(line.lstrip())
            i += 1
            # Skip indented block
            while i < len(lines):
                curr = lines[i]
                if curr.strip() == '':
                    i += 1
                    continue
                curr_indent = len(curr) - len(curr.lstrip())
                if curr_indent > base_indent:
                    i += 1
                    continue
                break
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)


def strip_package_imports(text: str) -> str:
    """Remove imports that refer to project packages; rely on inlined code instead.
    - Remove utils.* and factors.* imports
    - Route Numba imports through the shared cloud runtime
    - Remove relative imports
    - Keep standard/numpy/pandas
    """
    text = ast.unparse(normalize_numba_imports(text, inline=True))
    lines = text.split('\n')
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Multi-line import blocks
        if re.match(r"^\s*from\s+(?:utils\.|factors\.)[\w\.]+\s+import\s*\(\s*$", line):
            i += 1
            while i < len(lines):
                if ')' in lines[i]:
                    i += 1
                    break
                i += 1
            out.append('# (removed package import block)')
            continue
        # Single line project imports
        if re.match(r"^\s*from\s+(?:utils\.|factors\.)[\w\.]+\s+import\s+", line):
            out.append('# (removed project import)')
            i += 1
            continue
        if re.match(r"^\s*import\s+(?:utils|factors)(?:\s+as\s+\w+)?\s*$", line):
            out.append('# (removed project import)')
            i += 1
            continue
        if re.match(r"^\s*from\s+\.[\w\.]+\s+import\s+", line):
            out.append('# (removed relative import)')
            i += 1
            continue
        out.append(line)
        i += 1
    return '\n'.join(out)


def _resolve_factors_imports(code: str) -> List[str]:
    results: List[str] = []
    for m in re.finditer(r"^\s*from\s+factors\.[\w\.]+\s+import\b", code, flags=re.MULTILINE):
        # extract module path before ' import'
        line = m.group(0)
        mod = line.split('import')[0].strip().split()[1]
        results.append(mod)
    return results


def _dependency_module_bindings(source: str) -> Set[str]:
    """Collect module bindings without Python 3.12's inlined comprehension locals."""
    names: Set[str] = set()

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            for decorator in node.decorator_list:
                visit(decorator)
            if isinstance(node, ast.ClassDef):
                for expression in [*node.bases, *node.keywords]:
                    visit(expression)
            else:
                visit(node.args)
                if node.returns is not None:
                    visit(node.returns)
            return
        if isinstance(node, ast.Lambda):
            visit(node.args)
            return
        if isinstance(node, ast.comprehension):
            visit(node.iter)
            for condition in node.ifs:
                visit(condition)
            return
        if isinstance(node, ast.AnnAssign) and node.value is None:
            visit(node.annotation)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        if isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    return names


def inline_factors_imports(original: str, repo_root: Path, included_paths: Set[str], allowed_paths: Optional[Set[str]] = None, *, _bindings=None) -> str:
    """Recursively inline factors.* modules referenced by 'from factors.x import ...'.
    This is a coarse inliner that includes the entire target module (after cleaning),
    which is acceptable for Phase 1.
    """
    if _bindings is None:
        _bindings = {}
    chunks: List[str] = []
    imports = _resolve_factors_imports(original)
    for mod in imports:
        rel = Path(*mod.split('.')).with_suffix('.py')
        abs_path = repo_root / rel
        if not abs_path.is_file():
            raise FileNotFoundError("factor_dependency_missing: " + str(abs_path))
        key = str(abs_path.resolve())
        if allowed_paths is not None and key not in allowed_paths:
            raise ValueError('factor_dependency_not_explicit: ' + mod)
        if key in included_paths:
            continue
        try:
            src = _read_text(abs_path)
            for name in sorted(_dependency_module_bindings(src)):
                previous = _bindings.get(name)
                if previous is not None and previous != key:
                    raise ValueError(f'factor_dependency_binding_conflict: {name}: {previous} vs {key}')
                _bindings[name] = key
            # Recursive inline for its inner dependencies first
            nested = inline_factors_imports(src, repo_root, included_paths, allowed_paths=allowed_paths, _bindings=_bindings)
            nested = strip_package_imports(nested)
            nested = _remove_main_blocks(nested)
            chunks.append(f"# [inlined] {rel}")
            chunks.append(nested)
            included_paths.add(key)
        except Exception as exc:
            raise ValueError("factor_dependency_inline_failed: " + str(rel)) from exc
    # After collecting inlined chunks, remove the import lines from original
    cleaned = strip_package_imports(original)
    cleaned = _remove_main_blocks(cleaned)
    return ('\n'.join(chunks + [cleaned])).strip()


