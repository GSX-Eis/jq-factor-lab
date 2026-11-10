#!/usr/bin/env python3
# ruff: noqa: E402
import argparse
import ast
import json
import math
import re
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
    # Reuse the canonical FactorBase and fail the build if it cannot be normalized.
    repo = Path(__file__).resolve().parent.parent
    fb_path = repo / "utils" / "factor_base.py"
    source = neutralize_numba_decorators(read_source(str(fb_path)))
    # The cloud adapter replaces local reads; never ship their config/DB dependencies.
    tree = ast.parse(source)
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FactorBase')
    local_methods = [node for node in base.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'_fetch_raw_data', '_get_local_data_config', '_get_cache_key'}]
    if len(local_methods) != 3:
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


def render_single_file_strategy(context: Dict[str, Any]) -> str:
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
    parts.append(compose_runtime_shims_block())
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
        compose_strategy_entry_block(
            context["default_factor"], context["freq"], context["window"], context.get("index", "000300.XSHG")
        )
    )
    return "\n\n".join(parts)


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
    # If --freq is provided, build single bundle for that frequency (strict)
    if args.freq is not None:
        dep_map = build_dependency_map(prepared, req_map)
        try:
            computed_window = compute_required_window(prepared, req_map, args.freq)
        except Exception as e:
            print(f"Error computing required window: {e}")
            return 1
        fb_code = lift_minimal_factor_base()
        code = render_single_file_strategy({
            'version': '1.0',
            'factor_base_code': fb_code,
            'factor_blocks': [p['code'] for p in prepared],
            'dependencies': dep_map,
            'default_factor': prepared[0]['class_name'],
            'freq': args.freq,
            'window': computed_window,
            'cloud_mapping': cloud_mapping,
            'index': args.index,
        })
        code = inject_factor_required_windows(code, prepared, req_map)
        code = normalize_indexers_in_fstring_expressions(code)
        import os as _os
        _keep_logs = bool(getattr(args, 'keep_logs', False) or _os.getenv('FP_KEEP_LOGS'))
        if not _keep_logs:
            code = strip_prints_and_logger_calls(code)
        code = ensure_nonempty_blocks(code)
        code = inject_missing_passes(code)
        validate_no_external_imports(code)
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
        code = render_single_file_strategy({
            'version': '1.0',
            'factor_base_code': fb_code,
            'factor_blocks': [p['code'] for p in all_items],
            'dependencies': dep_map_all,
            'default_factor': all_items[0]['class_name'],
            'freq': 'daily',  # initial; each factor switches at runtime
            'window': max_window,
            'cloud_mapping': cloud_mapping,
            'index': args.index,
        })
        code = inject_factor_required_windows(code, all_items, req_map)
        code = normalize_indexers_in_fstring_expressions(code)
        import os as _os
        _keep_logs = bool(getattr(args, 'keep_logs', False) or _os.getenv('FP_KEEP_LOGS'))
        if not _keep_logs:
            code = strip_prints_and_logger_calls(code)
        code = ensure_nonempty_blocks(code)
        code = inject_missing_passes(code)
        validate_no_external_imports(code)
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
        code = render_single_file_strategy({
            'version': '1.0',
            'factor_base_code': fb_code,
            'factor_blocks': [p['code'] for p in items],
            'dependencies': dep_map_group,
            'default_factor': items[0]['class_name'],
            'freq': adapter_freq,
            'window': computed_window,
            'cloud_mapping': cloud_mapping,
            'index': args.index,
        })
        code = inject_factor_required_windows(code, items, req_map)
        code = normalize_indexers_in_fstring_expressions(code)
        import os as _os
        _keep_logs = bool(getattr(args, 'keep_logs', False) or _os.getenv('FP_KEEP_LOGS'))
        if not _keep_logs:
            code = strip_prints_and_logger_calls(code)
        code = ensure_nonempty_blocks(code)
        code = inject_missing_passes(code)
        validate_no_external_imports(code)
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
