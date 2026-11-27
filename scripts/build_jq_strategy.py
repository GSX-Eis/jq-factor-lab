#!/usr/bin/env python3
# ruff: noqa: E402
import argparse
import ast
import json
import math
import re
import symtable
import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

# Ensure repository root on sys.path so 'tools' package is importable when run as script
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
PUBLIC_RELEASE = True  # The public exporter enables explicit-input defaults.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build.ast_dependency_analyzer import (
    parse_python_to_ast,
    extract_required_fields_decl,
    extract_fields_from_calculate_factor,
    extract_fields_from_fetch_calls,
    extract_frequency_hints,
    extract_symbolic_declared_fields,
    extract_symbolic_frequency,
    extract_symbolic_required_window,
    merge_dependency_sources,
)
from scripts.build.numba_compat import (
    gen_conditional_jit_snippet,
    replace_decorators_with_conditional_jit,
)
from scripts.build.py36_compat import apply_py36_fixes
from scripts.build.assembler import (
    header_banner,
    compose_imports_block,
    compose_dependencies_block,
    compose_adapter_block,
    compose_runtime_shims_block,
    compose_strategy_entry_block,
    compose_mapping_block,
)
from scripts.build.import_inliner import inline_factors_imports, strip_package_imports
from scripts.build.ast_rewriter import rename_top_level_functions


def validate_no_external_imports(code: str):
    """
    Performs a post-build check to ensure no forbidden external imports remain.
    """
    lines = code.split('\n')
    forbidden_patterns = [
        r'^\s*from\s+\.factors',
        r'^\s*from\s+factors',
        r'^\s*import\s+factors',
        r'^\s*from\s+\.utils',
        r'^\s*from\s+utils',
        r'^\s*import\s+utils',
        # Allow `from .dynamic_stock_pool ...` inside FactorBase's try/except block
        r'^\s*from\s+\.dynamic_stock_pool'
    ]
    
    errors = []
    for i, line in enumerate(lines):
        # Skip the one relative import we allow inside FactorBase
        if 'from .dynamic_stock_pool' in line:
            continue
            
        for pattern in forbidden_patterns:
            if re.search(pattern, line):
                errors.append(f"L{i+1}: {line.strip()}")
                break # Move to next line once a match is found
    
    if errors:
        error_msg = "Build failed: The following external project imports were found in the bundled code.\n"
        error_msg += "This indicates the import inliner or stripper failed.\n"
        error_msg += "\n".join(errors)
        raise RuntimeError(error_msg)


def strip_test_helpers(source_code: str) -> str:
    """Remove top-level demo/test helper functions from inlined modules.
    This avoids bundling heavy demo routines with prints that can break on py36 quoting.
    Criteria: test-prefixed functions or names containing 'synthetic'/'demo'.
    """
    try:
        tree = ast.parse(source_code)
        new_body = []
        for node in getattr(tree, 'body', []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lname = node.name.lower()
                if (
                    lname.startswith(("test_", "_test_"))
                    or 'synthetic' in lname
                    or 'demo' in lname
                ):
                    continue
            new_body.append(node)
        tree.body = new_body
        ast.fix_missing_locations(tree)
        try:
            return ast.unparse(tree)
        except Exception:
            # If unparse unavailable, fall back to original
            return source_code
    except Exception:
        return source_code


def strip_prints_and_logger_calls(source_code: str) -> str:
    """Remove verbose diagnostics while retaining error evidence in the bundle.
    Targets:
      - print("...")
      - logger.info(...), self.logger.info(...), logger.debug/warning
    """
    try:
        class PrintLoggerStripper(ast.NodeTransformer):
            _LOGGER_METHODS = {"debug", "info", "warning", "warn"}

            def _is_print_call(self, call: ast.AST) -> bool:
                return isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'print'

            def _is_logger_method_call(self, call: ast.AST) -> bool:
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
                    return False
                if call.func.attr not in self._LOGGER_METHODS:
                    return False
                # Resolve base object chain and see if it ends with a '.logger' or Name('logger')
                val = call.func.value
                while isinstance(val, ast.Attribute):
                    if val.attr == 'logger':
                        return True
                    val = val.value
                return isinstance(val, ast.Name) and val.id == 'logger'

            def visit_Expr(self, node: ast.Expr):
                self.generic_visit(node)
                value = getattr(node, 'value', None)
                if isinstance(value, ast.Call):
                    if self._is_print_call(value) or self._is_logger_method_call(value):
                        return None
                return node

        tree = ast.parse(source_code)
        new_tree = PrintLoggerStripper().visit(tree)
        ast.fix_missing_locations(new_tree)
        try:
            return ast.unparse(new_tree)
        except Exception:
            return source_code
    except Exception:
        return source_code


def ensure_nonempty_blocks(source_code: str) -> str:
    """Ensure that all block bodies are non-empty by inserting 'pass' where needed.
    Handles: FunctionDef, AsyncFunctionDef, ClassDef, If, For, AsyncFor, While, With, AsyncWith, Try and its handlers/finally.
    """
    try:
        class NonEmptyBlockFiller(ast.NodeTransformer):
            def _fill(self, body):
                if not body:
                    return [ast.Pass()]
                return body

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                return node

            def visit_ClassDef(self, node: ast.ClassDef):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                return node

            def visit_If(self, node: ast.If):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                # orelse can be empty; if it's empty, unparser omits else:
                return node

            def visit_For(self, node: ast.For):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                return node

            def visit_AsyncFor(self, node: ast.AsyncFor):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                return node

            def visit_While(self, node: ast.While):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                return node

            def visit_With(self, node: ast.With):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                return node

            def visit_AsyncWith(self, node: ast.AsyncWith):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                return node

            def visit_Try(self, node: ast.Try):
                self.generic_visit(node)
                node.body = self._fill(node.body)
                for h in node.handlers:
                    h.body = self._fill(h.body)
                # finally body
                node.finalbody = self._fill(node.finalbody)
                # orelse can be empty safely
                return node

        tree = ast.parse(source_code)
        new_tree = NonEmptyBlockFiller().visit(tree)
        ast.fix_missing_locations(new_tree)
        try:
            return ast.unparse(new_tree)
        except Exception:
            return source_code
    except Exception:
        return source_code


def inject_missing_passes(source_code: str) -> str:
    """Line-based safety net: insert 'pass' after empty block starters when next line is not indented.
    Handles common starters: if/for/while/try/with/except/else/elif/finally.
    """
    try:
        lines = source_code.splitlines()
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            out.append(line)
            # Match block starter
            m = re.match(r'^(\s*)(if|for|while|try|with|except|else|elif|finally)\b.*:\s*$', line)
            if m:
                cur_indent = m.group(1)
                next_line = lines[i+1] if i+1 < len(lines) else ''
                # Determine leading spaces of next line
                next_lead = len(next_line) - len(next_line.lstrip(' ')) if next_line else 0
                cur_lead = len(cur_indent)
                if i+1 >= len(lines) or next_lead <= cur_lead:
                    # Insert a pass with one indentation level deeper
                    out.append(cur_indent + '    ' + 'pass')
            i += 1
        return '\n'.join(out)
    except Exception:
        return source_code


def parse_cli_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build JoinQuant single-file strategy from FactorPilot modules')
    p.add_argument('--factors', nargs='*', default=None, help='Factor class names to include; default: auto-discover all')
    p.add_argument(
        '--factor-files',
        nargs='*',
        default=None,
        help='Explicit factor file paths to include in discovery, even when they are outside repo_root/factors',
    )
    p.add_argument('--public', action='store_true', default=PUBLIC_RELEASE,
                   help='Only build explicit factor files with explicit requirements')
    p.add_argument('--requirements', type=Path, help='Factor requirements JSON path')
    p.add_argument('--cloud-api-mapping', type=Path, help='Cloud field mapping JSON path')
    p.add_argument('--index', default='000300.XSHG', help='Index universe for the example entrypoint')
    p.add_argument('--out', default='outputs/jq_strategy.py', help='Output single-file path')
    # --freq is optional now; when omitted, we auto-group by JSON requirements
    p.add_argument('--freq', default=None, choices=['daily', '5m', '1m'], help='Adapter frequency (optional). If omitted, group by config')
    p.add_argument('--keep-logs', action='store_true', help='Keep print/logger calls for debugging')
    p.add_argument('--single-file', action='store_true', help='When no --freq provided, merge all frequencies into one output file')
    args = p.parse_args(argv)
    if args.public:
        if not args.factor_files or args.requirements is None:
            p.error('public builds require --factor-files and --requirements')
        for raw in args.factor_files:
            path = Path(raw).expanduser()
            if not path.is_file() or path.suffix.lower() != '.py':
                p.error('invalid factor file: ' + raw)
    return args


def discover_factor_files(factors_root: str) -> List[str]:
    root = Path(factors_root)
    files: List[str] = []
    # 排除开发/归档/实验目录，避免类名冲突
    EXCLUDE_DIRS = {'todo', '__pycache__', 'test', 'backup','useful'}
    for py in root.rglob('*.py'):
        if py.name.startswith('__'):
            continue
        # 检查路径中是否包含排除目录
        if any(excluded in py.parts for excluded in EXCLUDE_DIRS):
            continue
        files.append(str(py))
    return files


def resolve_explicit_factor_files(raw_paths: Optional[List[str]]) -> List[str]:
    files: List[str] = []
    for raw in raw_paths or []:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file() or path.suffix.lower() != '.py':
            continue
        files.append(str(path))
    return files


def read_source(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_factor_requirements(repo_root: Path, path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser() if path is not None else repo_root / 'config' / 'factor_requirements.json'
    import json
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing required configuration: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    factors = data.get('factors')
    if not isinstance(factors, dict) or not factors:
        raise ValueError("factor_requirements.json is missing a non-empty 'factors' mapping")
    return factors


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_cloud_api_mapping(mapping: Any) -> Dict[str, Any]:
    """Validate mapping shapes that the generated JQ adapter dereferences.

    The mapping is copied verbatim into the generated strategy. Validate the
    portions the adapter uses so a correctable JSON error fails during the
    local build rather than as an AttributeError or KeyError in cloud runtime.
    Other top-level metadata, such as ``field_contracts``, is left untouched.
    """
    if not isinstance(mapping, dict):
        raise ValueError("cloud_api_mapping must be a JSON object")

    if "get_price" in mapping:
        get_price = mapping["get_price"]
        if not isinstance(get_price, list) or not all(_is_non_empty_string(field) for field in get_price):
            raise ValueError("cloud_api_mapping.get_price must be a list of non-empty strings")

    if "get_valuation" in mapping:
        valuation = mapping["get_valuation"]
        if not isinstance(valuation, dict):
            raise ValueError("cloud_api_mapping.get_valuation must be an object")
        for name, field in valuation.items():
            if not _is_non_empty_string(name) or not _is_non_empty_string(field):
                raise ValueError(
                    "cloud_api_mapping.get_valuation entries must map non-empty strings to non-empty strings"
                )

    if "get_factor_values" in mapping:
        factor_values = mapping["get_factor_values"]
        if not isinstance(factor_values, list) or not all(_is_non_empty_string(field) for field in factor_values):
            raise ValueError("cloud_api_mapping.get_factor_values must be a list of non-empty strings")

    if "get_money_flow" in mapping:
        money_flow = mapping["get_money_flow"]
        if not isinstance(money_flow, dict):
            raise ValueError("cloud_api_mapping.get_money_flow must be an object")
        for name, spec in money_flow.items():
            entry = f"cloud_api_mapping.get_money_flow.{name}"
            if not _is_non_empty_string(name) or not isinstance(spec, dict):
                raise ValueError(f"{entry} must be an object with a non-empty string field")
            if not _is_non_empty_string(spec.get("field")):
                raise ValueError(f"{entry}.field must be a non-empty string")
            rolling_days = spec.get("rolling_days")
            min_periods = spec.get("min_periods")
            if rolling_days is not None:
                if not _is_positive_int(rolling_days):
                    raise ValueError(f"{entry}.rolling_days must be a positive integer")
                if not _is_positive_int(min_periods):
                    raise ValueError(f"{entry}.min_periods must be a positive integer when rolling_days is set")
                if min_periods > rolling_days:
                    raise ValueError(f"{entry}.min_periods must not exceed rolling_days")
            elif min_periods is not None and not _is_positive_int(min_periods):
                raise ValueError(f"{entry}.min_periods must be a positive integer")

    return mapping


def load_cloud_api_mapping(repo_root: Path, path: Optional[Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser() if path is not None else repo_root / 'config' / 'cloud_api_mapping.json'
    import json
    if not cfg_path.exists():
        if path is not None:
            raise FileNotFoundError(f"Missing cloud API mapping: {cfg_path}")
        return {}
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return validate_cloud_api_mapping(json.load(f))


def compute_required_window(prepared: List[Dict[str, Any]], req_map: Dict[str, Any], freq: str) -> int:
    """
    Compute the maximum required history window across selected factors using
    factor_requirements.json. Enforce presence and presence of the chosen freq.
    Uses min_bars (for intraday) or min_days (for daily) plus slack_ratio.
    """
    if not prepared:
        return 0
    max_needed = 0
    missing: List[str] = []
    wrong_freq: List[str] = []
    for item in prepared:
        name = item.get('class_name')
        req = req_map.get(name)
        if not req:
            item_freq = item.get('frequency') or 'daily'
            if item_freq != freq:
                wrong_freq.append(name)
                continue
            fallback_window = int(item.get('required_window') or 0)
            if fallback_window > 0:
                max_needed = max(max_needed, fallback_window)
                continue
            missing.append(name)
            continue
        reqs = req.get('requirements') if isinstance(req, dict) else None
        if not isinstance(reqs, dict) or not reqs:
            item_freq = item.get('frequency') or 'daily'
            if item_freq != freq:
                wrong_freq.append(name)
                continue
            fallback_window = int(item.get('required_window') or 0)
            if fallback_window > 0:
                max_needed = max(max_needed, fallback_window)
                continue
            missing.append(name)
            continue
        freq_req = reqs.get(freq)
        if not isinstance(freq_req, dict):
            item_freq = item.get('frequency') or 'daily'
            fallback_window = int(item.get('required_window') or 0)
            if item_freq == freq and fallback_window > 0:
                max_needed = max(max_needed, fallback_window)
                continue
            wrong_freq.append(name)
            continue
        slack = float(freq_req.get('slack_ratio', 0.0) or 0.0)
        base = None
        if 'min_bars' in freq_req and freq != 'daily':
            base = int(freq_req.get('min_bars') or 0)
        elif 'min_days' in freq_req and freq == 'daily':
            base = int(freq_req.get('min_days') or 0)
        else:
            # If keys don't match the chosen freq semantics, still try either
            if 'min_bars' in freq_req:
                base = int(freq_req.get('min_bars') or 0)
            elif 'min_days' in freq_req:
                base = int(freq_req.get('min_days') or 0)
        if not base or base <= 0:
            missing.append(name)
            continue
        needed = int(math.ceil(base * (1.0 + slack)))
        if needed > max_needed:
            max_needed = needed
    if missing:
        raise RuntimeError(
            "Missing factor requirements for: " + ", ".join(sorted(set(missing)))
        )
    if wrong_freq:
        raise RuntimeError(
            f"Chosen freq '{freq}' not defined in requirements for: " + ", ".join(sorted(set(wrong_freq)))
        )
    if max_needed <= 0:
        raise RuntimeError("Computed required window is non-positive; check factor_requirements.json")
    return max_needed


def build_required_windows_map(prepared: List[Dict[str, Any]], req_map: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Build a mapping: factor_name -> {freq -> required_window}, using requirements json.
    For intraday, use min_bars; for daily, min_days; apply slack_ratio.
    Unknown/missing entries are omitted.
    """
    out: Dict[str, Dict[str, int]] = {}
    for item in prepared:
        name = item.get('class_name')
        req = req_map.get(name)
        fallback_freq = item.get('frequency')
        fallback_window = item.get('required_window')
        if not isinstance(req, dict):
            if fallback_freq and fallback_window:
                out[name] = {str(fallback_freq): int(fallback_window)}
            continue
        reqs = req.get('requirements')
        if not isinstance(reqs, dict):
            if fallback_freq and fallback_window:
                out[name] = {str(fallback_freq): int(fallback_window)}
            continue
        freq_map: Dict[str, int] = {}
        for k, v in reqs.items():
            if not isinstance(v, dict):
                continue
            slack = float(v.get('slack_ratio', 0.0) or 0.0)
            base = None
            if 'min_bars' in v:
                base = int(v.get('min_bars') or 0)
            elif 'min_days' in v:
                base = int(v.get('min_days') or 0)
            if base and base > 0:
                freq_map[k] = int(math.ceil(base * (1.0 + slack)))
        if freq_map:
            out[name] = freq_map
        elif fallback_freq and fallback_window:
            out[name] = {str(fallback_freq): int(fallback_window)}
    return out


def extract_factor_specs(py_file: str) -> List[Dict[str, Any]]:
    src = read_source(py_file)
    tree = parse_python_to_ast(src)
    specs: List[Dict[str, Any]] = []
    import ast
    factor_bases = {'FactorBase'}
    for item in tree.body:
        if isinstance(item, ast.ImportFrom) and (item.module or '').startswith('factors.'):
            factor_bases.update(alias.asname or alias.name for alias in item.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            inherits_factor_base = False
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in factor_bases:
                    inherits_factor_base = True
                elif isinstance(base, ast.Attribute) and base.attr == 'FactorBase':
                    inherits_factor_base = True
            if not inherits_factor_base:
                continue
            symbolic_declared = {
                field for field in extract_symbolic_declared_fields(tree, node) if field != 'factor'
            }
            declared = extract_required_fields_decl(node)
            inferred_calc = extract_fields_from_calculate_factor(node)
            inferred_fetch = extract_fields_from_fetch_calls(node)
            deps = symbolic_declared or merge_dependency_sources(declared, inferred_calc | inferred_fetch)
            freq = extract_symbolic_frequency(tree, node) or extract_frequency_hints(node)
            required_window = extract_symbolic_required_window(tree, node)
            specs.append({
                'class_name': node.name,
                'module_file': py_file,
                'declared_fields': declared,
                'symbolic_declared_fields': symbolic_declared,
                'inferred_fields': inferred_calc | inferred_fetch,
                'dependencies': deps,
                'frequency': freq,
                'required_window': required_window,
            })
    return specs


def neutralize_numba_decorators(source_code: str) -> str:
    """Legacy forwarding entrypoint for shared Numba import normalization."""
    return replace_decorators_with_conditional_jit(source_code)


def ensure_py36_compatibility(source_code: str) -> str:
    text, issues = apply_py36_fixes(source_code)
    # We only log issues to stderr; do not fail build
    for msg in issues:
        print(f"[py36-compat] {msg}", file=sys.stderr)
    return text


def normalize_indexers_in_fstring_expressions(source_code: str) -> str:
    lines = source_code.splitlines(keepends=True)
    return ''.join(
        re.sub(r"\['([^'\\\n]+)'\]", r'["\1"]', line)
        if "f'" in line and '{' in line
        else line
        for line in lines
    )


def lift_minimal_factor_base() -> str:
    # Cloud runtime is frozen at the reviewed stage-5 boundary.
    repo = Path(__file__).resolve().parent.parent
    fb_path = repo / "scripts/build/cloud_runtime/factor_base.py.txt"
    source = neutralize_numba_decorators(read_source(str(fb_path)))
    # The cloud adapter replaces local reads; never ship their config/DB dependencies.
    tree = ast.parse(source)
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FactorBase')
    local_names = {'_fetch_raw_data', '_get_local_data_config', '_get_cache_key',
                   '_fetch_market_bars', 'session_rolling', '_get_market_factor_values',
                   'get_bar_factor_frame', 'get_session_daily_pricing', 'evaluate_bar_factor'}
    local_methods = [node for node in base.body if isinstance(node, ast.FunctionDef)
                     and node.name in local_names]
    if len(local_methods) != len(local_names):
        raise ValueError('FactorBase local read boundary changed; review cloud bundling')
    lines = source.splitlines(keepends=True)
    for node in sorted(local_methods, key=lambda item: item.lineno, reverse=True):
        body_start = node.body[0].lineno - 1
        # Local disk caches depend on database import generations; cloud runs
        # calculate through their cloud adapter and do not use those caches.
        lines[body_start:node.end_lineno] = [
            '        return None\n' if node.name == '_get_cache_key' else
            '        raise RuntimeError("Local reads are unavailable in a JoinQuant bundle; initialize the cloud adapter")\n'
        ]
    return (''.join(lines)
            .replace("from utils.pit import apply_pit_adjustment", "")
            .replace("from utils.pit import normalize_price_money", ""))


_LEGACY_SYMBOLIC_CLASS_CONSTANTS = frozenset(
    {
        "EXPRESSION_CODE",
        "EXPRESSION_TEXT",
        "PRICE_FIELDS",
        "FACTOR_DATA_FIELDS",
        "DAILY_FEATURES",
        "INTRADAY_SPECS",
        "FACTOR_FEATURES",
        "DATA_FREQUENCY",
        "REQUIRED_HISTORY_WINDOW",
        "FACTOR_VALUE_CACHE_DIR",
        "SYMBOLIC_FACTOR_METADATA",
    }
)
_READONLY_CONTAINER_ITER_METHODS = frozenset({"items", "values", "keys"})
_READONLY_BUILTINS = frozenset({"len", "bool"})
_READONLY_COMPARE_OPS = (ast.In, ast.NotIn, ast.Eq, ast.NotEq, ast.Is, ast.IsNot)


def _is_immutable_scalar_expr(node: ast.AST, immutable_names: set[str]) -> bool:
    """Return True iff *node* evaluates to a deeply immutable value."""
    if isinstance(node, ast.Constant):
        if type(node.value) not in (str, bytes, int, float, bool, type(None)):
            return False
        return not isinstance(node.value, float) or math.isfinite(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)
    if isinstance(node, ast.Tuple):
        return all(_is_immutable_scalar_expr(elt, immutable_names) for elt in node.elts)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return node.id in immutable_names
    return False


def _is_flat_readonly_container_expr(node: ast.AST, immutable_names: set[str]) -> bool:
    """Return True iff *node* is a 1-level container whose elements/values are immutable scalars."""
    if isinstance(node, (ast.List, ast.Set)):
        return all(_is_immutable_scalar_expr(elt, immutable_names) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None
            and _is_immutable_scalar_expr(key, immutable_names)
            and _is_immutable_scalar_expr(val, immutable_names)
            for key, val in zip(node.keys, node.values)
        )
    return False


def _is_static_readonly_expr(node: ast.AST, verified_names: set[str]) -> bool:
    """Return True iff *node* is a pure literal/container tree referencing only verified constants."""
    if isinstance(node, ast.Constant):
        if type(node.value) not in (str, bytes, int, float, bool, type(None)):
            return False
        return not isinstance(node.value, float) or math.isfinite(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_static_readonly_expr(elt, verified_names) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None
            and _is_static_readonly_expr(key, verified_names)
            and _is_static_readonly_expr(val, verified_names)
            for key, val in zip(node.keys, node.values)
        )
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return node.id in verified_names
    return False


def _referenced_load_names(node: ast.AST) -> set[str]:
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }


def _collect_bound_names(tree: ast.Module) -> set[str]:
    """Collect all identifier names bound or imported anywhere in *tree*."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


def _is_valid_fetch_raw_data_arg(
    node: ast.Name,
    parent_map: Dict[int, ast.AST],
    fetch_method_shadowed: bool,
) -> bool:
    """Allow ``self._fetch_raw_data(fields=NAME, ...)`` inside a FactorBase method."""
    if fetch_method_shadowed:
        return False
    kw = parent_map.get(id(node))
    if not isinstance(kw, ast.keyword) or kw.arg != "fields" or kw.value is not node:
        return False
    call = parent_map.get(id(kw))
    if not isinstance(call, ast.Call):
        return False
    if not (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == "_fetch_raw_data"
    ):
        return False
    for arg in call.args:
        if not _is_immutable_scalar_expr(arg, set()):
            return False
    for other_kw in call.keywords:
        if other_kw is kw:
            continue
        if not _is_immutable_scalar_expr(other_kw.value, set()):
            return False

    # Require the call to reside directly inside a method of a top-level ClassDef with first arg 'self'.
    curr: ast.AST = call
    enclosing_method: Optional[ast.AST] = None
    while id(curr) in parent_map:
        par = parent_map[id(curr)]
        if isinstance(par, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not any(curr is item for item in par.body):
                return False
            enclosing_method = par
            break
        if isinstance(par, ast.Lambda):
            return False
        curr = par
    if enclosing_method is None or id(enclosing_method) not in parent_map:
        return False
    owner_cls = parent_map[id(enclosing_method)]
    if not isinstance(owner_cls, ast.ClassDef) or not isinstance(parent_map.get(id(owner_cls)), ast.Module):
        return False
    pos_args = getattr(enclosing_method.args, "posonlyargs", []) + enclosing_method.args.args
    return bool(pos_args) and pos_args[0].arg == "self"


_INTRADAY_SCALAR_SPEC_KEYS = frozenset({"name", "field", "aggregator"})


def _is_verified_intraday_specs_shape(node: ast.AST, immutable_names: set[str]) -> bool:
    """Verify ``INTRADAY_SPECS`` is a dict whose spec dicts bind scalar keys to immutable scalars."""
    if not isinstance(node, ast.Dict):
        return False
    for key_node, val_node in zip(node.keys, node.values):
        if key_node is None or not _is_immutable_scalar_expr(key_node, immutable_names):
            return False
        if not isinstance(val_node, ast.Dict):
            return False
        for sk_node, sv_node in zip(val_node.keys, val_node.values):
            if not (isinstance(sk_node, ast.Constant) and isinstance(sk_node.value, str)):
                return False
            if sk_node.value in _INTRADAY_SCALAR_SPEC_KEYS:
                if not _is_immutable_scalar_expr(sv_node, immutable_names):
                    return False
            elif sk_node.value == "dependencies":
                if not isinstance(sv_node, (ast.List, ast.Tuple)) or not all(
                    _is_immutable_scalar_expr(elt, immutable_names) for elt in sv_node.elts
                ):
                    return False
            else:
                return False
    return True


def _is_exact_legacy_spec_get_fn(sg_fn: ast.FunctionDef) -> bool:
    """Match only the compiler-emitted ``_spec_get(spec, key, default=None)`` template."""
    if sg_fn.decorator_list:
        return False
    if sg_fn.args.vararg or sg_fn.args.kwarg or sg_fn.args.kwonlyargs:
        return False
    sg_pos = getattr(sg_fn.args, "posonlyargs", []) + sg_fn.args.args
    if len(sg_pos) not in (2, 3):
        return False
    p_spec = sg_pos[0].arg
    p_key = sg_pos[1].arg
    p_default = sg_pos[2].arg if len(sg_pos) == 3 else None
    if len({p_spec, p_key, *([p_default] if p_default is not None else [])}) != len(sg_pos):
        return False
    if len(sg_pos) == 2:
        if sg_fn.args.defaults:
            return False
    elif len(sg_fn.args.defaults) > 1 or any(
        not _is_immutable_scalar_expr(d, set()) for d in sg_fn.args.defaults
    ):
        return False

    if len(sg_fn.body) != 2:
        return False
    if_stmt, tail_ret = sg_fn.body
    if not isinstance(if_stmt, ast.If) or if_stmt.orelse or len(if_stmt.body) != 1:
        return False

    test = if_stmt.test
    if not (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Name)
        and test.func.id == "isinstance"
        and isinstance(test.func.ctx, ast.Load)
        and not test.keywords
        and len(test.args) == 2
        and isinstance(test.args[0], ast.Name)
        and test.args[0].id == p_spec
        and isinstance(test.args[0].ctx, ast.Load)
        and isinstance(test.args[1], ast.Name)
        and test.args[1].id == "dict"
        and isinstance(test.args[1].ctx, ast.Load)
    ):
        return False

    def _is_scalar_default_expr(expr: ast.AST) -> bool:
        if (
            p_default is not None
            and isinstance(expr, ast.Name)
            and expr.id == p_default
            and isinstance(expr.ctx, ast.Load)
        ):
            return True
        return _is_immutable_scalar_expr(expr, set())

    dict_ret = if_stmt.body[0]
    if not isinstance(dict_ret, ast.Return) or not isinstance(dict_ret.value, ast.Call):
        return False
    get_call = dict_ret.value
    if not (
        isinstance(get_call.func, ast.Attribute)
        and isinstance(get_call.func.value, ast.Name)
        and get_call.func.value.id == p_spec
        and isinstance(get_call.func.value.ctx, ast.Load)
        and get_call.func.attr == "get"
        and isinstance(get_call.func.ctx, ast.Load)
        and not get_call.keywords
        and len(get_call.args) in (1, 2)
        and isinstance(get_call.args[0], ast.Name)
        and get_call.args[0].id == p_key
        and isinstance(get_call.args[0].ctx, ast.Load)
        and (len(get_call.args) == 1 or _is_scalar_default_expr(get_call.args[1]))
    ):
        return False

    if not isinstance(tail_ret, ast.Return) or not isinstance(tail_ret.value, ast.Call):
        return False
    getattr_call = tail_ret.value
    if not (
        isinstance(getattr_call.func, ast.Name)
        and getattr_call.func.id == "getattr"
        and isinstance(getattr_call.func.ctx, ast.Load)
        and not getattr_call.keywords
        and len(getattr_call.args) in (2, 3)
        and isinstance(getattr_call.args[0], ast.Name)
        and getattr_call.args[0].id == p_spec
        and isinstance(getattr_call.args[0].ctx, ast.Load)
        and isinstance(getattr_call.args[1], ast.Name)
        and getattr_call.args[1].id == p_key
        and isinstance(getattr_call.args[1].ctx, ast.Load)
        and (len(getattr_call.args) == 2 or _is_scalar_default_expr(getattr_call.args[2]))
    ):
        return False
    return True


def _is_verified_intraday_aggregator_tree(tree: ast.Module, bound_names: set[str]) -> bool:
    """Verify the compiler-emitted ``_aggregate_intraday_feature_series``/``_spec_get`` helpers
    only perform read-only lookups on their ``spec`` parameter."""
    agg_fns = [
        stmt
        for stmt in tree.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "_aggregate_intraday_feature_series"
    ]
    if len(agg_fns) != 1:
        return False
    agg_fn = agg_fns[0]
    if agg_fn.decorator_list:
        return False
    agg_pos = getattr(agg_fn.args, "posonlyargs", []) + agg_fn.args.args
    if len(agg_pos) < 3:
        return False
    spec_param = agg_pos[2].arg

    # Ensure neither helper name is rebound/shadowed/imported elsewhere.
    fn_defs_count = {"_aggregate_intraday_feature_series": 0, "_spec_get": 0}
    spec_get_fns: List[ast.FunctionDef] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "_spec_get":
            spec_get_fns.append(stmt)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in fn_defs_count:
                fn_defs_count[node.name] += 1
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id in fn_defs_count:
                return False
        elif isinstance(node, ast.arg) and node.arg in fn_defs_count:
            return False
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) in fn_defs_count:
                    return False
    if fn_defs_count["_aggregate_intraday_feature_series"] != 1:
        return False

    agg_parents: Dict[int, ast.AST] = {}
    for parent in ast.walk(agg_fn):
        for child in ast.iter_child_nodes(parent):
            agg_parents[id(child)] = parent

    uses_spec_get = False
    for node in ast.walk(agg_fn):
        if node is agg_fn:
            continue
        if isinstance(node, ast.arg) and node.arg == spec_param and node is not agg_pos[2]:
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == spec_param:
            return False
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == spec_param:
            return False
        if isinstance(node, ast.MatchMapping) and node.rest == spec_param:
            return False
        if isinstance(node, ast.Name) and node.id == spec_param:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                return False
            par = agg_parents.get(id(node))
            if (
                isinstance(par, ast.Call)
                and isinstance(par.func, ast.Name)
                and par.func.id == "_spec_get"
                and not par.keywords
                and len(par.args) in (2, 3)
                and par.args[0] is node
                and isinstance(par.args[1], ast.Constant)
                and par.args[1].value in _INTRADAY_SCALAR_SPEC_KEYS
                and (len(par.args) == 2 or _is_immutable_scalar_expr(par.args[2], set()))
            ):
                uses_spec_get = True
                continue
            return False

    if not uses_spec_get:
        return True
    if len(spec_get_fns) != 1 or fn_defs_count["_spec_get"] != 1:
        return False
    if {"isinstance", "dict", "getattr"} & bound_names:
        return False

    return _is_exact_legacy_spec_get_fn(spec_get_fns[0])


def _is_verified_intraday_specs_use(
    node: ast.Name,
    parent_map: Dict[int, ast.AST],
    has_verified_aggregator: bool,
    has_verified_intraday_shape: bool,
) -> bool:
    """Allow the exact compiler-emitted ``INTRADAY_SPECS.values()`` comprehension and
    ``INTRADAY_SPECS.items()`` aggregation loop."""
    if node.id != "INTRADAY_SPECS" or not has_verified_intraday_shape:
        return False
    parent = parent_map.get(id(node))
    if not (
        isinstance(parent, ast.Attribute)
        and parent.value is node
        and isinstance(parent.ctx, ast.Load)
        and parent.attr in {"values", "items"}
    ):
        return False
    call = parent_map.get(id(parent))
    if not (isinstance(call, ast.Call) and call.func is parent and not call.args and not call.keywords):
        return False

    gpar = parent_map.get(id(call))
    if parent.attr == "values":
        if not (isinstance(gpar, ast.comprehension) and gpar.iter is call and not gpar.ifs):
            return False
        if not isinstance(gpar.target, ast.Name):
            return False
        spec_var = gpar.target.id
        comp_expr = parent_map.get(id(gpar))
        if not isinstance(comp_expr, (ast.SetComp, ast.ListComp, ast.GeneratorExp)):
            return False
        comp_parents: Dict[int, ast.AST] = {}
        for p in ast.walk(comp_expr):
            for ch in ast.iter_child_nodes(p):
                comp_parents[id(ch)] = p
        for sub in ast.walk(comp_expr):
            if sub is gpar.target:
                continue
            if isinstance(sub, ast.Name) and sub.id == spec_var:
                if isinstance(sub.ctx, (ast.Store, ast.Del)):
                    return False
                sub_par = comp_parents.get(id(sub))
                if (
                    isinstance(sub_par, ast.Subscript)
                    and sub_par.value is sub
                    and isinstance(sub_par.ctx, ast.Load)
                    and isinstance(sub_par.slice, ast.Constant)
                    and sub_par.slice.value in _INTRADAY_SCALAR_SPEC_KEYS
                ):
                    seq_par = comp_parents.get(id(sub_par))
                    if isinstance(seq_par, (ast.List, ast.Tuple)) and isinstance(
                        comp_parents.get(id(seq_par)), ast.comprehension
                    ):
                        continue
                if (
                    isinstance(sub_par, ast.Attribute)
                    and sub_par.value is sub
                    and sub_par.attr == "get"
                    and isinstance(sub_par.ctx, ast.Load)
                ):
                    get_call = comp_parents.get(id(sub_par))
                    if (
                        isinstance(get_call, ast.Call)
                        and get_call.func is sub_par
                        and not get_call.keywords
                        and len(get_call.args) == 2
                        and isinstance(get_call.args[0], ast.Constant)
                        and isinstance(get_call.args[0].value, str)
                        and isinstance(get_call.args[1], (ast.List, ast.Tuple))
                        and not get_call.args[1].elts
                    ):
                        starred_par = comp_parents.get(id(get_call))
                        if isinstance(starred_par, ast.Starred):
                            seq_par = comp_parents.get(id(starred_par))
                            if isinstance(seq_par, (ast.List, ast.Tuple)) and isinstance(
                                comp_parents.get(id(seq_par)), ast.comprehension
                            ):
                                continue
                return False
        return True

    # parent.attr == "items"
    if not has_verified_aggregator:
        return False
    if not (isinstance(gpar, ast.For) and gpar.iter is call and not gpar.orelse):
        return False
    if not (
        isinstance(gpar.target, ast.Tuple)
        and len(gpar.target.elts) == 2
        and all(isinstance(elt, ast.Name) for elt in gpar.target.elts)
        and gpar.target.elts[0].id != gpar.target.elts[1].id
    ):
        return False
    spec_var = gpar.target.elts[1].id
    # Locate the enclosing function scope and verify *spec_var* never leaks or is
    # referenced anywhere outside the loop's _aggregate_intraday_feature_series call.
    curr_scope: ast.AST = gpar
    enclosing_fn: Optional[ast.AST] = None
    while id(curr_scope) in parent_map:
        par = parent_map[id(curr_scope)]
        if isinstance(par, (ast.FunctionDef, ast.AsyncFunctionDef)):
            enclosing_fn = par
            break
        curr_scope = par
    if enclosing_fn is None:
        return False

    allowed_spec_loads: set[int] = set()
    loop_parents: Dict[int, ast.AST] = {}
    for stmt in gpar.body:
        for p in ast.walk(stmt):
            for ch in ast.iter_child_nodes(p):
                loop_parents[id(ch)] = p
    for stmt in gpar.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Name) and sub.id == spec_var and isinstance(sub.ctx, ast.Load):
                sub_par = loop_parents.get(id(sub))
                if (
                    isinstance(sub_par, ast.Call)
                    and isinstance(sub_par.func, ast.Name)
                    and sub_par.func.id == "_aggregate_intraday_feature_series"
                    and not sub_par.keywords
                    and len(sub_par.args) == 3
                    and sub_par.args[2] is sub
                ):
                    allowed_spec_loads.add(id(sub))

    def _walk_enclosing_fn_for_spec(node_cur: ast.AST) -> bool:
        if node_cur is gpar.target.elts[1] or id(node_cur) in allowed_spec_loads:
            return True
        if isinstance(node_cur, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_targets = {
                t.id
                for gen in node_cur.generators
                for t in ast.walk(gen.target)
                if isinstance(t, ast.Name)
            }
            if spec_var in comp_targets:
                # Only generators[0].iter evaluates in enclosing_fn; the comprehension body
                # binds its own isolated *spec_var*.
                return not node_cur.generators or _walk_enclosing_fn_for_spec(node_cur.generators[0].iter)
        if isinstance(node_cur, ast.arg) and node_cur.arg == spec_var:
            return False
        if isinstance(node_cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node_cur.name == spec_var:
            return False
        if isinstance(node_cur, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node_cur.name == spec_var:
            return False
        if isinstance(node_cur, ast.MatchMapping) and node_cur.rest == spec_var:
            return False
        if isinstance(node_cur, (ast.Global, ast.Nonlocal)) and spec_var in node_cur.names:
            return False
        if isinstance(node_cur, ast.Name) and node_cur.id == spec_var:
            return False
        return all(_walk_enclosing_fn_for_spec(child) for child in ast.iter_child_nodes(node_cur))

    return all(_walk_enclosing_fn_for_spec(stmt) for stmt in enclosing_fn.body)


def _is_provably_readonly_container_load(
    node: ast.Name,
    *,
    is_flat: bool,
    is_dict: bool,
    parent_map: Dict[int, ast.AST],
    unshadowed_builtins: set[str],
    fetch_method_shadowed: bool,
    has_verified_intraday_aggregator: bool,
    has_verified_intraday_shape: bool,
) -> bool:
    """Positive allowlist for provably read-only container uses inside callable bodies."""
    parent = parent_map.get(id(node))
    if isinstance(parent, (ast.If, ast.IfExp, ast.While)) and parent.test is node:
        return True
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not) and parent.operand is node:
        return True
    if isinstance(parent, ast.BoolOp):
        curr: ast.AST = parent
        while isinstance(parent_map.get(id(curr)), ast.BoolOp):
            curr = parent_map[id(curr)]
        top_par = parent_map.get(id(curr))
        if isinstance(top_par, (ast.If, ast.IfExp, ast.While)) and top_par.test is curr:
            return True
        if isinstance(top_par, ast.UnaryOp) and isinstance(top_par.op, ast.Not) and top_par.operand is curr:
            return True
        return False
    if isinstance(parent, ast.Compare):
        return all(isinstance(op, _READONLY_COMPARE_OPS) for op in parent.ops)
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id in unshadowed_builtins
        and parent.args == [node]
        and not parent.keywords
    ):
        return True
    if is_flat and isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node:
        return True
    if is_dict and isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node:
        return True
    if is_flat and isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load):
        return True
    if (
        isinstance(parent, ast.Attribute)
        and parent.value is node
        and isinstance(parent.ctx, ast.Load)
        and (
            (is_flat and parent.attr in _READONLY_CONTAINER_ITER_METHODS)
            or (is_dict and parent.attr == "keys")
        )
    ):
        call = parent_map.get(id(parent))
        if not (isinstance(call, ast.Call) and call.func is parent and not call.args and not call.keywords):
            return False
        gpar = parent_map.get(id(call))
        if isinstance(gpar, (ast.For, ast.comprehension)) and gpar.iter is call:
            return True
        if (
            isinstance(gpar, ast.Call)
            and isinstance(gpar.func, ast.Name)
            and gpar.func.id in unshadowed_builtins
            and gpar.args == [call]
            and not gpar.keywords
        ):
            return True
        if isinstance(gpar, ast.Compare) and all(isinstance(op, (ast.In, ast.NotIn)) for op in gpar.ops):
            return True
        return False
    if _is_valid_fetch_raw_data_arg(node, parent_map, fetch_method_shadowed):
        return True
    return _is_verified_intraday_specs_use(
        node,
        parent_map,
        has_verified_intraday_aggregator,
        has_verified_intraday_shape,
    )


def _iter_symtables(table: symtable.SymbolTable):
    yield table
    for child in table.get_children():
        yield from _iter_symtables(child)


def _collect_disqualified_symbolic_names(tree: ast.Module, table: symtable.SymbolTable) -> set[str]:
    """Identify symbolic constant names that are not strictly single-bound read-only literals."""
    disqualified: set[str] = set()
    module_def_nodes: Dict[str, ast.stmt] = {}

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name in _LEGACY_SYMBOLIC_CLASS_CONSTANTS:
                if name in module_def_nodes:
                    disqualified.add(name)
                else:
                    module_def_nodes[name] = stmt
                continue
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
        ):
            name = stmt.target.id
            if name in _LEGACY_SYMBOLIC_CLASS_CONSTANTS:
                if name in module_def_nodes:
                    disqualified.add(name)
                else:
                    module_def_nodes[name] = stmt
                continue

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if stmt.name in _LEGACY_SYMBOLIC_CLASS_CONSTANTS:
                disqualified.add(stmt.name)
            continue

        # Any other top-level statement (including multi-target/unpacking Assign or control flow)
        # that binds/imports/deletes a candidate name disqualifies it.
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                if sub.id in _LEGACY_SYMBOLIC_CLASS_CONSTANTS:
                    disqualified.add(sub.id)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for alias in sub.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if bound in _LEGACY_SYMBOLIC_CLASS_CONSTANTS:
                        disqualified.add(bound)

    top_class_tables = {
        (child.get_name(), child.get_lineno()): child
        for child in table.get_children()
        if child.get_type() == "class"
    }
    top_class_table_ids = {id(child) for child in top_class_tables.values()}

    for child_table in _iter_symtables(table):
        if child_table is table:
            continue
        is_top_class = id(child_table) in top_class_table_ids
        for name in _LEGACY_SYMBOLIC_CLASS_CONSTANTS:
            try:
                sym = child_table.lookup(name)
            except KeyError:
                continue
            if sym.is_declared_global() or sym.is_parameter():
                disqualified.add(name)
            elif not is_top_class and (sym.is_local() or sym.is_free()):
                disqualified.add(name)

    return disqualified


def _collect_class_scope_name_refs(cls_node: ast.ClassDef, skip_stmts: set[int]) -> set[str]:
    """Collect all identifier loads/stores in class scope outside method/nested-class bodies."""
    refs: set[str] = set()

    def walk_class_scope(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            refs.add(node.name)
            for dec in node.decorator_list:
                walk_class_scope(dec)
            for arg in [*getattr(node.args, "posonlyargs", []), *node.args.args, *node.args.kwonlyargs]:
                refs.add(arg.arg)
                if arg.annotation is not None:
                    walk_class_scope(arg.annotation)
            if node.args.vararg is not None:
                refs.add(node.args.vararg.arg)
                if node.args.vararg.annotation is not None:
                    walk_class_scope(node.args.vararg.annotation)
            if node.args.kwarg is not None:
                refs.add(node.args.kwarg.arg)
                if node.args.kwarg.annotation is not None:
                    walk_class_scope(node.args.kwarg.annotation)
            for default in node.args.defaults:
                walk_class_scope(default)
            for default in node.args.kw_defaults:
                if default is not None:
                    walk_class_scope(default)
            if node.returns is not None:
                walk_class_scope(node.returns)
            return
        if isinstance(node, ast.ClassDef):
            refs.add(node.name)
            for expr in [*node.decorator_list, *node.bases]:
                walk_class_scope(expr)
            for kw in node.keywords:
                walk_class_scope(kw.value)
            return
        if isinstance(node, ast.Name):
            refs.add(node.id)
        for child in ast.iter_child_nodes(node):
            walk_class_scope(child)

    for expr in [*cls_node.decorator_list, *cls_node.bases]:
        walk_class_scope(expr)
    for kw in cls_node.keywords:
        walk_class_scope(kw.value)
    for stmt in cls_node.body:
        if id(stmt) in skip_stmts:
            continue
        walk_class_scope(stmt)
    return refs


def _is_inside_callable_body(node: ast.AST, parent_map: Dict[int, ast.AST]) -> bool:
    """Return True only when *node* executes inside a function/method/lambda body."""
    curr = node
    while id(curr) in parent_map:
        par = parent_map[id(curr)]
        if isinstance(par, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return any(curr is item for item in par.body)
        if isinstance(par, ast.Lambda):
            return curr is par.body
        curr = par
    return False


def normalize_legacy_symbolic_constants(source_code: str) -> str:
    """Strictly decouple legacy symbolic ``NAME = NAME`` class pass-throughs.

    Historical symbolic factors bound read-only constants at module scope and
    forwarded them into the factor class via ``EXPRESSION_CODE = EXPRESSION_CODE``.
    ASTPrefixer rejects class bindings that shadow module bindings due to
    class-body execution-order ambiguity. When a known symbolic constant is
    proven to be a single-bound static module literal/container tree with
    strictly read-only usage across the entire module (no mutation, aliasing,
    or extra class-scope reference) and is forwarded once as ``NAME = NAME``,
    alias the module constant to ``_LEGACY_MODULE_<NAME>`` and point the class
    attribute at that alias—preserving shared object identity before prefixing.
    """
    tree = ast.parse(source_code)
    table = symtable.symtable(source_code, "<factor>", "exec")
    disqualified = _collect_disqualified_symbolic_names(tree, table)

    verified_names: set[str] = set()
    immutable_names: set[str] = set()
    flat_container_names: set[str] = set()
    container_names: set[str] = set()
    dict_names: set[str] = set()
    module_assign_nodes: Dict[str, ast.stmt] = {}
    module_value_nodes: Dict[str, ast.AST] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            value_node = stmt.value
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
        ):
            name = stmt.target.id
            value_node = stmt.value
        else:
            continue
        if name not in _LEGACY_SYMBOLIC_CLASS_CONSTANTS or name in disqualified:
            continue
        if not _is_static_readonly_expr(value_node, verified_names):
            disqualified.add(name)
            continue
        verified_names.add(name)
        module_assign_nodes[name] = stmt
        module_value_nodes[name] = value_node
        if _is_immutable_scalar_expr(value_node, immutable_names):
            immutable_names.add(name)
        else:
            container_names.add(name)
            if _is_flat_readonly_container_expr(value_node, immutable_names):
                flat_container_names.add(name)
            if isinstance(value_node, ast.Dict):
                dict_names.add(name)

    if not verified_names:
        return source_code

    valid_passthroughs: List[tuple[ast.ClassDef, ast.Assign, str]] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef):
            continue
        seen_in_class: Dict[str, int] = {}
        cls_candidates: List[tuple[ast.Assign, str]] = []
        cls_skip_ids: set[int] = set()
        for member in stmt.body:
            if (
                isinstance(member, ast.Assign)
                and len(member.targets) == 1
                and isinstance(member.targets[0], ast.Name)
                and isinstance(member.value, ast.Name)
                and member.targets[0].id == member.value.id
                and member.targets[0].id in verified_names
            ):
                name = member.targets[0].id
                seen_in_class[name] = seen_in_class.get(name, 0) + 1
                cls_candidates.append((member, name))
                cls_skip_ids.add(id(member))
        for name, count in seen_in_class.items():
            if count > 1:
                disqualified.add(name)
        other_class_refs = _collect_class_scope_name_refs(stmt, cls_skip_ids)
        for name in other_class_refs:
            if name in verified_names:
                disqualified.add(name)
        for member, name in cls_candidates:
            if name not in disqualified:
                valid_passthroughs.append((stmt, member, name))

    allowed_store_ids = {
        id(mod_stmt.targets[0] if isinstance(mod_stmt, ast.Assign) else mod_stmt.target)
        for name, mod_stmt in module_assign_nodes.items()
        if name not in disqualified
    } | {
        id(member.targets[0])
        for _, member, name in valid_passthroughs
        if name not in disqualified
    }
    allowed_passthrough_load_ids = {
        id(member.value)
        for _, member, name in valid_passthroughs
        if name not in disqualified
    }
    allowed_top_level_load_ids = set(allowed_passthrough_load_ids)
    for name, value_node in module_value_nodes.items():
        if name in disqualified:
            continue
        for sub in ast.walk(value_node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                allowed_top_level_load_ids.add(id(sub))

    parent_map: Dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent

    bound_names = _collect_bound_names(tree)
    unshadowed_builtins = _READONLY_BUILTINS - bound_names
    fetch_method_shadowed = "_fetch_raw_data" in bound_names
    has_verified_intraday_aggregator = _is_verified_intraday_aggregator_tree(tree, bound_names)
    has_verified_intraday_shape = (
        "INTRADAY_SPECS" in module_value_nodes
        and _is_verified_intraday_specs_shape(module_value_nodes["INTRADAY_SPECS"], immutable_names)
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in verified_names:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                if id(node) not in allowed_store_ids:
                    disqualified.add(node.id)
                continue
            if isinstance(node.ctx, ast.Load):
                if not _is_inside_callable_body(node, parent_map):
                    if id(node) not in allowed_top_level_load_ids:
                        disqualified.add(node.id)
                elif node.id in container_names:
                    if not _is_provably_readonly_container_load(
                        node,
                        is_flat=node.id in flat_container_names,
                        is_dict=node.id in dict_names,
                        parent_map=parent_map,
                        unshadowed_builtins=unshadowed_builtins,
                        fetch_method_shadowed=fetch_method_shadowed,
                        has_verified_intraday_aggregator=has_verified_intraday_aggregator,
                        has_verified_intraday_shape=has_verified_intraday_shape,
                    ):
                        disqualified.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in verified_names:
                    disqualified.add(bound)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name in verified_names:
            disqualified.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest in verified_names:
            disqualified.add(node.rest)

    # Propagate disqualification bidirectionally across container-embedding edges:
    # if an outer container embeds a disqualified constant, or an outer container itself
    # is disqualified (e.g. via nested element mutation), any embedded mutable container
    # cannot be proven read-only either.
    changed = True
    while changed:
        changed = False
        for name, value_node in module_value_nodes.items():
            deps = _referenced_load_names(value_node)
            if name in disqualified:
                for dep in deps:
                    if dep in container_names and dep not in disqualified:
                        disqualified.add(dep)
                        changed = True
            elif deps & disqualified:
                disqualified.add(name)
                changed = True

    decouple_names = {
        name for (_, _, name) in valid_passthroughs if name not in disqualified
    }
    if not decouple_names:
        return source_code

    existing_ids = (
        {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        | {node.name for node in ast.walk(tree) if hasattr(node, "name") and isinstance(node.name, str)}
        | {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    )
    alias_map: Dict[str, str] = {}
    for name in sorted(decouple_names):
        candidate = f"_LEGACY_MODULE_{name}"
        suffix = 1
        while candidate in existing_ids:
            candidate = f"_LEGACY_MODULE_{name}_{suffix}"
            suffix += 1
        existing_ids.add(candidate)
        alias_map[name] = candidate

    for _, member, name in valid_passthroughs:
        if name in alias_map:
            member.value = ast.Name(id=alias_map[name], ctx=ast.Load())

    for name, alias in alias_map.items():
        mod_stmt = module_assign_nodes[name]
        if isinstance(mod_stmt, ast.Assign):
            assert isinstance(mod_stmt.targets[0], ast.Name)
            mod_stmt.targets[0].id = alias
        elif isinstance(mod_stmt, ast.AnnAssign):
            assert isinstance(mod_stmt.target, ast.Name)
            mod_stmt.target.id = alias

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in alias_map:
            node.id = alias_map[node.id]

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def select_and_prepare_factors(specs: List[Dict[str, Any]], selected: Optional[List[str]], allowed_paths: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    chosen = []
    want: Set[str] = set(selected) if selected else set(s['class_name'] for s in specs)
    for sp in specs:
        if sp['class_name'] not in want:
            continue
        code = read_source(sp['module_file'])
        # Phase 1: inline factors.* dependencies and strip utils.* imports
        repo_root = Path(__file__).resolve().parent.parent
        code = inline_factors_imports(code, repo_root, included_paths=set(), allowed_paths=allowed_paths)
        # Drop demo/test helper functions before any further transforms
        code = strip_test_helpers(code)
        # Decouple verified read-only legacy symbolic class pass-through constants before prefix isolation.
        code = normalize_legacy_symbolic_constants(code)
        # Prefix-rename module bindings to avoid cross-file collisions in a single-file bundle.
        # A failed transform must not silently emit source with shared globals.
        try:
            code = rename_top_level_functions(code, sp['class_name'])
        except Exception as exc:
            raise ValueError(
                f"factor_namespace_isolation_failed for {sp['class_name']}: {exc}"
            ) from exc
        # Normalize bracket indexer quotes to double quotes to avoid f-string outer-quote conflicts after AST unparse
        # Example: f'... {df["code"]} ...' stays valid even if outer string uses single quotes
        # Avoid naive global quote replacements that can corrupt f-strings
        code = ensure_py36_compatibility(code)
        chosen.append({
            'class_name': sp['class_name'],
            'code': code,
            'dependencies': sp['dependencies'],
            'frequency': sp.get('frequency'),
            'required_window': sp.get('required_window'),
            'symbolic_declared_fields': sp.get('symbolic_declared_fields', set()),
        })
    return chosen


def build_dependency_map(prepared: List[Dict[str, Any]], req_map: Dict[str, Any]) -> Dict[str, Dict[str, Set[str]]]:
    dep: Dict[str, Dict[str, Set[str]]] = {}
    STD_FIELDS = {
        'daily': {'open','close','high','low','volume','money'},
        '1m': {'open','close','high','low','volume','money'},
        '5m': {'open','close','high','low','volume','money'},
        '15m': {'open','close','high','low','volume','money'},
        '30m': {'open','close','high','low','volume','money'},
        '60m': {'open','close','high','low','volume','money'},
    }
    for item in prepared:
        name = item['class_name']
        # Prefer frequency from config requirements when available; fallback to AST hint or 'daily'
        cfg = req_map.get(name) or {}
        cfg_reqs = cfg.get('requirements', {}) if isinstance(cfg, dict) else {}
        cfg_freq = None
        if cfg_reqs:
            cfg_keys = list(cfg_reqs.keys())
            if len(cfg_keys) == 1:
                cfg_freq = cfg_keys[0]
            else:
                # Deterministic priority: intraday first, then daily
                _priority = ['1m', '5m', '15m', '30m', '60m', 'daily']
                for _p in _priority:
                    if _p in cfg_keys:
                        cfg_freq = _p
                        break
                if cfg_freq is None and cfg_keys:
                    cfg_freq = cfg_keys[0]
        freq = cfg_freq or item.get('frequency') or 'daily'
        # Overlay order: config.fields > extracted > standard
        cfg_fields = set(cfg_reqs.get(freq, {}).get('fields') or [])
        fields = set(item['dependencies'])
        if cfg_fields:
            fields = cfg_fields
        if not fields:
            fields = set(STD_FIELDS.get(freq, STD_FIELDS['daily']))
        dep[name] = {freq: fields}
        if freq != 'daily':
            dep[name].setdefault('daily', set())
    return dep


def _choose_freq_for_factor(name: str, req_map: Dict[str, Any]) -> Optional[str]:
    cfg = req_map.get(name) or {}
    cfg_reqs = cfg.get('requirements', {}) if isinstance(cfg, dict) else {}
    if not cfg_reqs:
        return None
    # Deterministic priority: intraday first, then daily
    _priority = ['1m', '5m', '15m', '30m', '60m', 'daily']
    for _p in _priority:
        if _p in cfg_reqs:
            # We only support '1m', '5m', 'daily' adapters; treat 15m/30m/60m as 5m bucket for fetching,
            # but here we still return the concrete key to compute window later if needed.
            return _p
    # Fallback to first key
    return next(iter(cfg_reqs.keys()))


def group_factors_by_config_frequency(prepared: List[Dict[str, Any]], req_map: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Partition prepared factors into frequency groups based on factor_requirements.json.
    Returns mapping of adapter frequency -> list of factor specs.
    Adapter frequencies supported: 'daily', '5m', '1m'.
    15m/30m/60m requirements will be grouped under '5m' adapter.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {'daily': [], '5m': [], '1m': []}
    missing: List[str] = []
    for item in prepared:
        name = item.get('class_name')
        freq_key = _choose_freq_for_factor(name, req_map) or item.get('frequency')
        if not freq_key:
            missing.append(name)
            continue
        # Map non-standard intraday keys to 5m adapter bucket
        if freq_key in ('1m', '5m', 'daily'):
            adapter_freq = freq_key
        elif freq_key in ('15m', '30m', '60m'):
            adapter_freq = '5m'
        else:
            # Unknown -> default to 5m bucket
            adapter_freq = '5m'
        groups.setdefault(adapter_freq, [])
        groups[adapter_freq].append(item)
    if missing:
        raise RuntimeError("Missing factor requirements for: " + ", ".join(sorted(set(missing))))
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


def render_single_file_strategy(context: Dict[str, Any], *, entry_block: Optional[str] = None) -> str:
    parts: List[str] = []
    parts.append(header_banner(context.get("version", "1.0")))
    parts.append("")
    parts.append(compose_imports_block(needs_typing=True))
    parts.append("")
    parts.append("# ---- conditional_jit ----")
    parts.append(gen_conditional_jit_snippet())
    parts.append("")
    parts.append(compose_mapping_block(context.get("cloud_mapping", {})))
    parts.append("")
    parts.append((_REPO_ROOT / "utils/pit.py").read_text(encoding="utf-8"))
    parts.append("# ---- FactorBase (inlined) ----")
    parts.append(context["factor_base_code"])
    parts.append("")
    parts.append("# ---- runtime shims ----")
    parts.append(compose_runtime_shims_block(install_modules=entry_block is None))
    parts.append("")
    for blk in context["factor_blocks"]:
        parts.append("# ---- factor implementation ----")
        parts.append(blk)
    parts.append("")
    parts.append("# ---- dependencies ----")
    parts.append(compose_dependencies_block(context["dependencies"]))
    parts.append("")
    parts.append("# ---- adapter ----")
    parts.append(compose_adapter_block())
    for module in ("factor_runtime.py", "strategy_kernel.py"):
        parts.append((_REPO_ROOT / "utils" / module).read_text(encoding="utf-8"))
    parts.append("")
    parts.append("")
    parts.append("# ---- strategy entry ----")
    parts.append(
        entry_block if entry_block is not None else compose_strategy_entry_block(
            context["default_factor"], context["freq"], context["window"], context.get("index", "000300.XSHG")
        )
    )
    return "\n\n".join(parts)


def compile_single_file_strategy(
    context: Dict[str, Any],
    prepared: List[Dict[str, Any]],
    requirements: Dict[str, Any],
    *,
    entry_block: Optional[str] = None,
    keep_logs: bool = False,
) -> str:
    """Render and finalize a self-contained JoinQuant candidate."""
    code = render_single_file_strategy(context, entry_block=entry_block)
    code = inject_factor_required_windows(code, prepared, requirements)
    code = normalize_indexers_in_fstring_expressions(code)
    if not keep_logs:
        code = strip_prints_and_logger_calls(code)
    code = ensure_nonempty_blocks(code)
    code = inject_missing_passes(code)
    validate_no_external_imports(code)
    return code


def inject_factor_required_windows(
    code: str, prepared: List[Dict[str, Any]], req_map: Dict[str, Any]
) -> str:
    factor_windows_map = build_required_windows_map(prepared, req_map)
    if not factor_windows_map:
        return code
    import json as _json

    code = (
        "FACTOR_REQUIREMENTS = "
        + repr(
            {
                "version": "2",
                "factors": {
                    item["class_name"]: req_map.get(
                        item["class_name"],
                        {
                            "requirements": {
                                item.get("frequency") or "daily": {
                                    "min_days": item.get("required_window", 1),
                                    "fields": sorted(item.get("dependencies", [])),
                                }
                            }
                        },
                    )
                    for item in prepared
                },
            }
        )
        + "\n"
        + code
    )
    return code.replace(
        "\n# ---- dependencies ----",
        "\n# ---- dependencies ----\nFACTOR_REQUIRED_WINDOWS = "
        + _json.dumps(factor_windows_map, ensure_ascii=False, indent=2)
        + "\n",
    )


def write_output_file(code: str, out_path: str) -> None:
    code = normalize_indexers_in_fstring_expressions(code)
    compile(code, out_path, 'exec')
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(code)


def build_jq_strategy_main(argv: List[str]) -> int:
    args = parse_cli_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    factor_files = [] if args.public else discover_factor_files(str(repo_root / 'factors'))
    for py in resolve_explicit_factor_files(args.factor_files):
        if py not in factor_files:
            factor_files.append(py)
    # Prefer non-archived implementations on name collisions: keep first occurrence per class
    # Enforce presence of factor_requirements.json
    try:
        req_map = load_factor_requirements(repo_root, args.requirements)
    except Exception as e:
        print(f"Error loading factor requirements: {e}")
        return 1
    
    # Load Cloud API Mapping
    try:
        cloud_mapping = ({} if args.public and args.cloud_api_mapping is None
                         else load_cloud_api_mapping(repo_root, args.cloud_api_mapping))
    except Exception as e:
        print(f"Error loading cloud API mapping: {e}", file=sys.stderr)
        return 1

    specs: List[Dict[str, Any]] = []
    for py in factor_files:
        try:
            specs.extend(extract_factor_specs(py))
        except Exception:
            if args.public:
                raise
            continue
    if args.public:
        names = [sp["class_name"] for sp in specs]
        if len(names) != len(set(names)):
            print(json.dumps({"ok": False, "error": {
                "code": "duplicate_factor_class_in_explicit_files",
                "message": "Factor class names must be unique across explicit files",
                "details": {"classes": sorted({name for name in names if names.count(name) > 1})},
            }}), file=sys.stderr)
            return 1
        if args.factors and set(args.factors) - set(names):
            print(json.dumps({"ok": False, "error": {
                "code": "requested_factor_class_missing",
                "message": "Requested factor classes were not found in the explicit files",
                "details": {"classes": sorted(set(args.factors) - set(names))},
            }}), file=sys.stderr)
            return 1
    # Deduplicate by class_name: choose non-archived file when duplicates exist
    seen: Dict[str, Dict[str, Any]] = {}
    def _is_archived(path: str) -> bool:
        try:
            from pathlib import Path as _P
            return 'archived' in set(map(str.lower, _P(path).parts))
        except Exception:
            return False
    for sp in specs:
        name = sp.get('class_name')
        if not name:
            continue
        prev = seen.get(name)
        if prev is None:
            seen[name] = sp
        else:
            prev_arch = _is_archived(prev.get('module_file', ''))
            cur_arch = _is_archived(sp.get('module_file', ''))
            if prev_arch and not cur_arch:
                seen[name] = sp
            # if both archived or both non-archived, keep first
    specs_dedup = list(seen.values())

    allowed_paths = {str(Path(p).resolve()) for p in factor_files} if args.public else None
    try:
        prepared = select_and_prepare_factors(specs_dedup, args.factors, allowed_paths=allowed_paths)
    except ValueError as e:
        print(f"Error preparing factor source: {e}", file=sys.stderr)
        return 1
    if not prepared:
        print('No factors prepared.')
        return 1
    import os as _os
    keep_logs = bool(args.keep_logs or _os.getenv('FP_KEEP_LOGS'))
    # If --freq is provided, build single bundle for that frequency (strict)
    if args.freq is not None:
        dep_map = build_dependency_map(prepared, req_map)
        try:
            computed_window = compute_required_window(prepared, req_map, args.freq)
        except Exception as e:
            print(f"Error computing required window: {e}")
            return 1
        fb_code = lift_minimal_factor_base()
        code = compile_single_file_strategy({
            'version': '1.0',
            'factor_base_code': fb_code,
            'factor_blocks': [p['code'] for p in prepared],
            'dependencies': dep_map,
            'default_factor': prepared[0]['class_name'],
            'freq': args.freq,
            'window': computed_window,
            'cloud_mapping': cloud_mapping,
            'index': args.index,
        }, prepared, req_map, keep_logs=keep_logs)
        write_output_file(code, args.out)
        print(f"Generated: {args.out} with {len(prepared)} factors")
        return 0

    # Auto-mode: group factors by frequency from config and build one bundle per adapter freq
    try:
        groups = group_factors_by_config_frequency(prepared, req_map)
    except Exception as e:
        print(f"Error grouping factors by frequency: {e}")
        return 1

    # Optional: merge all groups into a single output file
    if getattr(args, 'single_file', False):
        # Compute an entry window as the max across group windows
        max_window = 0
        for adapter_freq, items in groups.items():
            try:
                w = compute_required_window(items, req_map, adapter_freq)
                if w > max_window:
                    max_window = w
            except Exception as e:
                print(f"Error computing required window for {adapter_freq}: {e}")
                return 1
        if max_window <= 0:
            print('Failed to compute a positive combined window')
            return 1
        # Build combined dependencies and factor blocks
        all_items: List[Dict[str, Any]] = []
        for items in groups.values():
            all_items.extend(items)
        dep_map_all = build_dependency_map(all_items, req_map)
        fb_code = lift_minimal_factor_base()
        code = compile_single_file_strategy({
            'version': '1.0',
            'factor_base_code': fb_code,
            'factor_blocks': [p['code'] for p in all_items],
            'dependencies': dep_map_all,
            'default_factor': all_items[0]['class_name'],
            'freq': 'daily',  # initial; each factor switches at runtime
            'window': max_window,
            'cloud_mapping': cloud_mapping,
            'index': args.index,
        }, all_items, req_map, keep_logs=keep_logs)
        write_output_file(code, args.out)
        print(f"Generated: {args.out} with {len(all_items)} factors (merged, window={max_window})")
        return 0

    # Derive output base and extension
    out_path = Path(args.out)
    base_name = out_path.stem
    ext = out_path.suffix or '.py'
    out_dir = out_path.parent
    fb_code = lift_minimal_factor_base()

    total_factors = 0
    for adapter_freq, items in groups.items():
        if not items:
            continue
        # Compute per-group dependencies and window
        dep_map_group = build_dependency_map(items, req_map)
        try:
            computed_window = compute_required_window(items, req_map, adapter_freq)
        except Exception as e:
            print(f"Error computing required window for {adapter_freq}: {e}")
            return 1
        code = compile_single_file_strategy({
            'version': '1.0',
            'factor_base_code': fb_code,
            'factor_blocks': [p['code'] for p in items],
            'dependencies': dep_map_group,
            'default_factor': items[0]['class_name'],
            'freq': adapter_freq,
            'window': computed_window,
            'cloud_mapping': cloud_mapping,
            'index': args.index,
        }, items, req_map, keep_logs=keep_logs)
        # Output file name with suffix
        out_file = out_dir / f"{base_name}_{adapter_freq}{ext}"
        write_output_file(code, str(out_file))
        print(f"Generated: {out_file} with {len(items)} factors (freq={adapter_freq}, window={computed_window})")
        total_factors += len(items)

    if total_factors == 0:
        print("No factors to build after grouping.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(build_jq_strategy_main(sys.argv[1:]))
