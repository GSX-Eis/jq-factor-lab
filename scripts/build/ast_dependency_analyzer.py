import ast
from typing import Any, Set, Dict, Optional


def parse_python_to_ast(source_code: str) -> ast.AST:
    """Parse Python source to AST and return the root node."""
    return ast.parse(source_code)


def _string_literal(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def extract_required_fields_decl(cls_node: ast.ClassDef) -> Set[str]:
    """Extract REQUIRED_FIELDS set from class body if declared as simple literal."""
    fields: Set[str] = set()
    for node in cls_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'REQUIRED_FIELDS':
                    value = node.value
                    if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
                        for elt in value.elts:
                            literal = _string_literal(elt)
                            if literal is not None:
                                fields.add(literal)
                    elif isinstance(value, ast.Dict):
                        for key in value.keys:
                            literal = _string_literal(key)
                            if literal is not None:
                                fields.add(literal)
    return fields


def _string_sequence(value: ast.AST) -> Set[str]:
    values: Set[str] = set()
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        for item in value.elts:
            literal = _string_literal(item)
            if literal is not None:
                values.add(literal)
    return values


def _literal_value(value: ast.AST) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def _module_assignments(tree: ast.AST) -> Dict[str, ast.AST]:
    assignments: Dict[str, ast.AST] = {}
    for node in getattr(tree, 'body', []):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value
    return assignments


def _class_assignments(cls_node: ast.ClassDef) -> Dict[str, ast.AST]:
    assignments: Dict[str, ast.AST] = {}
    for node in cls_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value
    return assignments


def _resolve_name(value: ast.AST, module_assignments: Dict[str, ast.AST]) -> ast.AST:
    if isinstance(value, ast.Name) and value.id in module_assignments:
        return module_assignments[value.id]
    return value


def _dict_literal_has_items(value: ast.AST) -> bool:
    resolved = _literal_value(value)
    return isinstance(resolved, dict) and bool(resolved)


def _expression_code_literal(module_assigns: Dict[str, ast.AST], class_assigns: Dict[str, ast.AST]) -> Optional[str]:
    value = class_assigns.get('EXPRESSION_CODE', module_assigns.get('EXPRESSION_CODE'))
    if value is not None:
        literal = _literal_value(_resolve_name(value, module_assigns))
        if isinstance(literal, str):
            return literal
    return None


def _window_from_expression_code(expression_code: str) -> Optional[int]:
    try:
        parsed = ast.parse(expression_code, mode='eval')
    except Exception:
        return None
    # Same lag/rolling recurrence as symbolic/compiler._required_history_periods.
    # Keep this builder independent of the optional mining package.
    lag = ['delay', 'delta', 'pct_change', 'ts_pctchange']
    rolling = ['bb_lower', 'bb_middle', 'bb_upper', 'count', 'decay_linear', 'decaylinear', 'ema', 'highday', 'lowday', 'macd', 'percentile', 'prod', 'regbeta', 'regresi', 'rsi', 'sma', 'sum_if', 'sumif', 'ts_argmax', 'ts_argmin', 'ts_autocorr', 'ts_beta', 'ts_corr', 'ts_cov', 'ts_covariance', 'ts_cumsum', 'ts_cv', 'ts_iqr', 'ts_kurt', 'ts_mad', 'ts_max', 'ts_mean', 'ts_median', 'ts_min', 'ts_product', 'ts_quantile', 'ts_rank', 'ts_residual', 'ts_skew', 'ts_slope', 'ts_std', 'ts_sum', 'ts_var', 'ts_zscore', 'wma']
    known = ['abs', 'add', 'bb_lower', 'bb_middle', 'bb_upper', 'clip', 'count', 'cs_demean', 'cs_kurt', 'cs_max', 'cs_mean', 'cs_median', 'cs_min', 'cs_quantile', 'cs_rank', 'cs_scale', 'cs_scale_to', 'cs_skew', 'cs_std', 'cs_zscore', 'decay_linear', 'decaylinear', 'delay', 'delta', 'div', 'element_max', 'element_min', 'ema', 'eq', 'exp', 'filter', 'floor', 'ge', 'gt', 'highday', 'inv', 'le', 'log', 'logical_and', 'logical_or', 'lowday', 'lt', 'macd', 'mul', 'ne', 'neg', 'neutralize_size', 'neutralize_styles', 'pct_change', 'percentile', 'pow', 'power', 'prod', 'regbeta', 'regresi', 'rsi', 'sequence', 'sign', 'sma', 'sqrt', 'sub', 'sum_if', 'sumif', 'ts_argmax', 'ts_argmin', 'ts_autocorr', 'ts_beta', 'ts_corr', 'ts_cov', 'ts_covariance', 'ts_cumsum', 'ts_cv', 'ts_iqr', 'ts_kurt', 'ts_mad', 'ts_max', 'ts_mean', 'ts_median', 'ts_min', 'ts_pctchange', 'ts_product', 'ts_quantile', 'ts_rank', 'ts_residual', 'ts_skew', 'ts_slope', 'ts_std', 'ts_sum', 'ts_var', 'ts_zscore', 'where', 'winsorize', 'wma']

    def history(node):
        if isinstance(node, ast.Name):
            return 1
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return 0
        if isinstance(node, ast.Subscript):
            return 1
        if isinstance(node, ast.UnaryOp):
            return history(node.operand)
        if isinstance(node, ast.BinOp):
            return max(history(node.left), history(node.right))
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError('unsupported expression')
        name = node.func.id
        if name not in known or node.keywords:
            raise ValueError('unsupported operator')
        if name in lag or name in rolling:
            positions = {'macd': (1, 2), 'sum_if': (1,), 'sumif': (1,),
                         'ts_quantile': (1,), 'sma': (1,)}.get(name, (len(node.args) - 1,))
            if not node.args or any(i >= len(node.args) for i in positions):
                raise ValueError('missing window')
            windows = [node.args[i] for i in positions]
            if any(not isinstance(w, ast.Constant) or type(w.value) is not int or w.value <= 0
                   for w in windows):
                raise ValueError('unsupported window')
            child = max((history(arg) for i, arg in enumerate(node.args) if i not in positions), default=1)
            return child + max(w.value for w in windows) - (name in rolling)
        if name == 'sequence':
            return 1
        return max((history(arg) for arg in node.args), default=1)

    try:
        return max(1, history(parsed.body))
    except ValueError:
        # Unsupported legacy syntax keeps its existing inference contract.
        windows = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None)
                if not name or not str(name).startswith('ts_') or not node.args:
                    continue
                try:
                    window = int(float(_literal_value(node.args[-1])))
                except Exception:
                    continue
                if window > 0:
                    windows.append(window)
        return max(windows or [1])

def extract_symbolic_declared_fields(tree: ast.AST, cls_node: ast.ClassDef) -> Set[str]:
    """Extract raw generated symbolic data fields from PRICE_FIELDS/FACTOR_DATA_FIELDS."""
    fields: Set[str] = set()
    module_assigns = _module_assignments(tree)
    class_assigns = _class_assignments(cls_node)
    for name in ('PRICE_FIELDS', 'FACTOR_DATA_FIELDS'):
        if name in module_assigns:
            fields.update(_string_sequence(module_assigns[name]))
        if name in class_assigns:
            fields.update(_string_sequence(_resolve_name(class_assigns[name], module_assigns)))
    return fields


def extract_symbolic_frequency(tree: ast.AST, cls_node: ast.ClassDef) -> Optional[str]:
    module_assigns = _module_assignments(tree)
    class_assigns = _class_assignments(cls_node)
    for name in ('DATA_FREQUENCY', 'FREQUENCY'):
        value = class_assigns.get(name, module_assigns.get(name))
        if value is not None:
            value = _resolve_name(value, module_assigns)
            literal = _literal_value(value)
            if isinstance(literal, str):
                freq = _normalize_freq(literal)
                if freq:
                    return freq
    metadata = class_assigns.get('SYMBOLIC_FACTOR_METADATA', module_assigns.get('SYMBOLIC_FACTOR_METADATA'))
    if metadata is not None:
        metadata_value = _literal_value(_resolve_name(metadata, module_assigns))
        if isinstance(metadata_value, dict):
            freq = _normalize_freq(str(metadata_value.get('data_frequency') or ''))
            if freq:
                return freq
    intraday_specs = class_assigns.get('INTRADAY_SPECS', module_assigns.get('INTRADAY_SPECS'))
    if intraday_specs is not None and _dict_literal_has_items(_resolve_name(intraday_specs, module_assigns)):
        return '5m'
    daily_features = class_assigns.get('DAILY_FEATURES', module_assigns.get('DAILY_FEATURES'))
    factor_features = class_assigns.get('FACTOR_FEATURES', module_assigns.get('FACTOR_FEATURES'))
    if daily_features is not None or factor_features is not None:
        return 'daily'
    for sub in ast.walk(cls_node):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        if sub.func.attr != 'setdefault' or len(sub.args) < 2:
            continue
        key = _string_literal(sub.args[0])
        if key != 'data_frequency':
            continue
        literal = _string_literal(sub.args[1])
        if literal is not None:
            freq = _normalize_freq(literal)
            if freq:
                return freq
    return None


def extract_symbolic_required_window(tree: ast.AST, cls_node: ast.ClassDef) -> Optional[int]:
    module_assigns = _module_assignments(tree)
    class_assigns = _class_assignments(cls_node)
    value = class_assigns.get('REQUIRED_HISTORY_WINDOW', module_assigns.get('REQUIRED_HISTORY_WINDOW'))
    if value is not None:
        value = _resolve_name(value, module_assigns)
        literal = _literal_value(value)
        try:
            window = int(literal)
            return window if window > 0 else None
        except Exception:
            pass
    metadata = class_assigns.get('SYMBOLIC_FACTOR_METADATA', module_assigns.get('SYMBOLIC_FACTOR_METADATA'))
    if metadata is not None:
        metadata_value = _literal_value(_resolve_name(metadata, module_assigns))
        if isinstance(metadata_value, dict):
            try:
                window = int(metadata_value.get('required_history_window') or 0)
                return window if window > 0 else None
            except Exception:
                return None
    expression_code = _expression_code_literal(module_assigns, class_assigns)
    if expression_code:
        window = _window_from_expression_code(expression_code)
        if window and window > 0:
            freq = extract_symbolic_frequency(tree, cls_node)
            return window * 48 if freq and freq != 'daily' else window
    return None


def extract_fields_from_calculate_factor(cls_node: ast.ClassDef) -> Set[str]:
    """Infer field names from calculate_factor method body by scanning for string-key access patterns."""
    fields: Set[str] = set()
    calc_node: Optional[ast.FunctionDef] = None
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'calculate_factor':
            calc_node = node
            break
    if calc_node is None:
        return fields

    for sub in ast.walk(calc_node):
        # df['open'] or df["close"]
        if isinstance(sub, ast.Subscript):
            key = sub.slice
            if isinstance(key, ast.Index):
                key = key.value
            literal = _string_literal(key)
            if literal is not None:
                fields.add(literal)

        # df.get('volume')
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr == 'get' and len(sub.args) >= 1:
                arg0 = sub.args[0]
                literal = _string_literal(arg0)
                if literal is not None:
                    fields.add(literal)

        # .loc[:, ['open','high']]
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr in ('loc', 'iloc') and len(sub.args) >= 2:
                cols_arg = sub.args[1]
                if isinstance(cols_arg, (ast.List, ast.Tuple)):
                    for elt in cols_arg.elts:
                        literal = _string_literal(elt)
                        if literal is not None:
                            fields.add(literal)

    # Only add literal selections from the unchanged input parameter.  Keep the
    # existing scalar inference contract; do not infer aliases or derived columns.
    args = calc_node.args.posonlyargs + calc_node.args.args
    if len(args) < 2:
        return fields
    input_name = args[1].arg
    nodes = list(ast.walk(calc_node))
    if any(isinstance(n, ast.Name) and n.id == input_name
           and isinstance(n.ctx, (ast.Store, ast.Del)) for n in nodes):
        return fields
    if sum(isinstance(n, ast.arg) and n.arg == input_name for n in nodes) != 1:
        return fields
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
           and n.name == input_name for n in nodes):
        return fields
    for n in nodes:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split('.')[0]) == input_name for alias in n.names):
                return fields
        if isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and n.name == input_name:
            return fields
        if isinstance(n, ast.MatchMapping) and n.rest == input_name:
            return fields
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
            if isinstance(n.value, ast.Name) and n.value.id == input_name:
                return fields
        if isinstance(n, ast.Subscript) and not isinstance(n.ctx, ast.Load):
            if (isinstance(n.value, ast.Attribute) and n.value.attr != 'loc'
                    and isinstance(n.value.value, ast.Name) and n.value.value.id == input_name):
                return fields  # Positional/other indexers can also create derived columns.
    # Mutation calls can create or rename columns without a subscript Store.
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and isinstance(n.func.value, ast.Name) and n.func.value.id == input_name
           and n.func.attr in {'insert', 'update', 'pop', 'rename', 'drop', 'eval'}
           for n in nodes):
        return fields
    selected = set()
    written = set()
    for n in nodes:
        if not isinstance(n, ast.Subscript):
            continue
        owner, key = n.value, n.slice
        if isinstance(owner, ast.Attribute) and owner.attr == 'loc':
            owner = owner.value
            if not isinstance(key, ast.Tuple) or len(key.elts) != 2:
                continue
            key = key.elts[1]
        if not isinstance(owner, ast.Name) or owner.id != input_name:
            continue
        if isinstance(key, (ast.List, ast.Tuple)):
            names = [_string_literal(item) for item in key.elts]
        else:
            names = [_string_literal(key)]
        if any(name is None for name in names):
            if not isinstance(n.ctx, ast.Load):
                return fields  # Unknown write target can shadow any input column.
            continue
        if isinstance(n.ctx, ast.Load):
            selected.update(names)
        else:
            written.update(names)
    return fields | (selected - written)


def extract_fields_from_fetch_calls(cls_node: ast.ClassDef) -> Set[str]:
    """Capture fields from calls like self._fetch_* (fields=[...]) or _get_raw_data(fields=[...]).
    Only collects literal lists/tuples of string constants.
    """
    fields: Set[str] = set()
    for sub in ast.walk(cls_node):
        if isinstance(sub, ast.Call):
            # Match any call that has a keyword 'fields'
            for kw in sub.keywords or []:
                if kw.arg != 'fields':
                    continue
                val = kw.value
                seq = None
                if isinstance(val, (ast.List, ast.Tuple)):
                    seq = val.elts
                # Support simple Name assignment is complex; skip for Phase 1
                if not seq:
                    continue
                for elt in seq:
                    literal = _string_literal(elt)
                    if literal is not None:
                        fields.add(literal)
            # Additionally, capture positional list/tuple of fields for common raw-data fetch helpers
            # e.g., self._fetch_raw_data(["MAC120", "low"]) without keyword
            try:
                fn_name = None
                if isinstance(sub.func, ast.Attribute):
                    fn_name = sub.func.attr
                elif isinstance(sub.func, ast.Name):
                    fn_name = sub.func.id
                # Heuristic: treat functions containing 'fetch' or 'raw_data' as data fetchers
                if fn_name and any(tok in fn_name.lower() for tok in ('fetch', 'raw_data')):
                    if getattr(sub, 'args', None):
                        first_arg = sub.args[0]
                        if isinstance(first_arg, (ast.List, ast.Tuple)):
                            for elt in first_arg.elts:
                                literal = _string_literal(elt)
                                if literal is not None:
                                    fields.add(literal)
            except Exception:
                # Best-effort; ignore analyzer failures
                pass
    return fields


def merge_dependency_sources(declared: Set[str], inferred: Set[str]) -> Set[str]:
    return set(sorted(set(declared) | set(inferred)))


def extract_frequency_hints(cls_node: ast.ClassDef) -> Optional[str]:
    """Try to infer data frequency for a factor class.
    Priority:
      1) Class attribute DATA_FREQUENCY/FREQUENCY
      2) Calls with keyword 'frequency' having a string literal (e.g., '5m', '1m', '60m', 'daily')
    Returns a normalized frequency or None.
    """
    # 1) Class attribute
    for node in cls_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('DATA_FREQUENCY', 'FREQUENCY'):
                    val = node.value
                    literal = _string_literal(val)
                    if literal is not None:
                        return _normalize_freq(literal)

    # 2) Keyword 'frequency' / 'data_frequency' in calls
    counts: Dict[str, int] = {}
    for sub in ast.walk(cls_node):
        if isinstance(sub, ast.Call):
            for kw in sub.keywords or []:
                if kw.arg in ('frequency', 'data_frequency'):
                    v = kw.value
                    literal = _string_literal(v)
                    if literal is not None:
                        f = _normalize_freq(literal)
                        if f:
                            counts[f] = counts.get(f, 0) + 1
    if counts:
        # return the most frequent
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return None


def _normalize_freq(val: str) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip().lower()
    aliases = {
        'd': 'daily', '1d': 'daily', 'day': 'daily', 'daily': 'daily',
        '1m': '1m', 'min': '1m',
        '5m': '5m', '5min': '5m',
        '15m': '15m', '15min': '15m',
        '30m': '30m', '30min': '30m',
        '60m': '60m', '60min': '60m', '1h': '60m', 'hour': '60m', '1hour': '60m'
    }
    return aliases.get(s, s if s in {'1m','5m','15m','30m','60m','daily'} else None)
