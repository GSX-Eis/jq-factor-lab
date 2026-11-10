import ast
import symtable

class ASTPrefixer(ast.NodeTransformer):
    """
    Traverses a Python AST to add a unique prefix to top-level symbols and
    their corresponding references within the same AST.

    This prevents name collisions when multiple factor files containing helper
    functions or generated constants with identical names are bundled into a
    single script.
    """
    def __init__(self, prefix: str, source_code: str | None = None):
        if not prefix or not prefix.isidentifier():
            raise ValueError("Prefix must be a valid Python identifier.")
        self.prefix = prefix
        self._top_level_symbols = set()
        self._scope_stack = []
        self._comprehension_locals = []
        if source_code is not None:
            self._scope_stack.append(symtable.symtable(source_code, "<factor>", "exec"))

    def _prefixed(self, name: str) -> str:
        return f"{self.prefix}_{name}"

    @staticmethod
    def _target_names(node: ast.AST):
        """Yield module names bound by an assignment target.

        Module-level assignments execute when a bundled strategy is imported.
        They therefore share a namespace just like helper functions do.  Limit
        this to ``Name`` (and unpacking) targets: attributes and subscriptions
        are side effects on an existing object, not new module bindings.
        """
        if isinstance(node, ast.Name):
            if not (node.id.startswith("__") and node.id.endswith("__")):
                yield node.id
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                yield from ASTPrefixer._target_names(item)

    def _rename_target(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if node.id in self._top_level_symbols:
                node.id = self._prefixed(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                self._rename_target(item)

    @staticmethod
    def _scope_name(node: ast.AST) -> str | None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return getattr(node, "_namespace_original_name", node.name)
        if isinstance(node, ast.ClassDef):
            return node.name
        if isinstance(node, ast.Lambda):
            return "lambda"
        if isinstance(node, ast.ListComp):
            return "listcomp"
        if isinstance(node, ast.SetComp):
            return "setcomp"
        if isinstance(node, ast.DictComp):
            return "dictcomp"
        if isinstance(node, ast.GeneratorExp):
            return "genexpr"
        return None

    def _child_scope(self, node: ast.AST):
        if not self._scope_stack:
            return None
        expected_type = "class" if isinstance(node, ast.ClassDef) else "function"
        expected_name = self._scope_name(node)
        candidates = [
            child for child in self._scope_stack[-1].get_children()
            if child.get_type() == expected_type and child.get_lineno() == node.lineno
        ]
        for child in candidates:
            if child.get_name() == expected_name:
                return child
        return None

    def _push_scope(self, node: ast.AST):
        child_scope = self._child_scope(node)
        if self._scope_stack and child_scope is None:
            raise ValueError(
                "unable to determine lexical scope for namespace isolation at "
                f"line {getattr(node, 'lineno', '?')}"
            )
        if child_scope is not None:
            self._scope_stack.append(child_scope)
        return child_scope

    def _pop_scope(self, child_scope) -> None:
        if child_scope is not None:
            self._scope_stack.pop()

    def _visit_arguments_in_outer_scope(self, arguments: ast.arguments) -> None:
        """Visit defaults/annotations where Python evaluates them: outer scope."""
        all_args = [*getattr(arguments, "posonlyargs", []), *arguments.args, *arguments.kwonlyargs]
        if arguments.vararg is not None:
            all_args.append(arguments.vararg)
        if arguments.kwarg is not None:
            all_args.append(arguments.kwarg)
        for argument in all_args:
            if argument.annotation is not None:
                argument.annotation = self.visit(argument.annotation)
        arguments.defaults = [self.visit(default) for default in arguments.defaults]
        arguments.kw_defaults = [
            self.visit(default) if default is not None else None
            for default in arguments.kw_defaults
        ]

    def _visit_function_definition(self, node):
        # Decorators, annotations, and default values are evaluated before the
        # function enters its lexical scope. Visiting them under that scope
        # would corrupt ``def f(offset=offset)`` when ``offset`` is a module
        # binding that gets prefixed.
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        self._visit_arguments_in_outer_scope(node.args)
        if node.returns is not None:
            node.returns = self.visit(node.returns)
        child_scope = self._push_scope(node)
        try:
            node.body = [self.visit(statement) for statement in node.body]
        finally:
            self._pop_scope(child_scope)
        return node

    def _visit_lambda(self, node: ast.Lambda) -> ast.Lambda:
        self._visit_arguments_in_outer_scope(node.args)
        child_scope = self._push_scope(node)
        try:
            node.body = self.visit(node.body)
        finally:
            self._pop_scope(child_scope)
        return node

    def _visit_comprehension_scope(self, node):
        # The first iterable is evaluated by the outer scope. Every target,
        # filter, and subsequent iterable belongs to the comprehension scope.
        if not node.generators:
            return node
        first = node.generators[0]
        first.iter = self.visit(first.iter)
        # Python 3.12+ inlines list/set/dict comprehensions in the symbol table,
        # while their iteration variables still have an isolated lexical scope.
        inlined = (not isinstance(node, ast.GeneratorExp)
                   and self._scope_stack and self._child_scope(node) is None)
        child_scope = None if inlined else self._push_scope(node)
        self._comprehension_locals.append({
            name for generator in node.generators
            for name in self._target_names(generator.target)
        })
        try:
            first.target = self.visit(first.target)
            first.ifs = [self.visit(condition) for condition in first.ifs]
            for generator in node.generators[1:]:
                generator.target = self.visit(generator.target)
                generator.iter = self.visit(generator.iter)
                generator.ifs = [self.visit(condition) for condition in generator.ifs]
            if isinstance(node, ast.DictComp):
                node.key = self.visit(node.key)
                node.value = self.visit(node.value)
            else:
                node.elt = self.visit(node.elt)
        finally:
            self._comprehension_locals.pop()
            self._pop_scope(child_scope)
        return node

    def _resolves_to_module_binding(self, name: str) -> bool:
        if any(name in names for names in self._comprehension_locals):
            return False
        if not self._scope_stack:
            return True
        scope = self._scope_stack[-1]
        if scope.get_type() == "module":
            return True
        try:
            symbol = scope.lookup(name)
        except KeyError:
            return True
        return not (symbol.is_local() or symbol.is_free())

    def _is_declared_global(self, name: str) -> bool:
        if any(name in names for names in self._comprehension_locals):
            return False
        if not self._scope_stack or self._scope_stack[-1].get_type() == "module":
            return False
        try:
            return self._scope_stack[-1].lookup(name).is_declared_global()
        except KeyError:
            return False

    def _unsupported_control_flow_binding(self, node: ast.AST) -> tuple[str, str] | None:
        """Find bindings in module control flow that this rewriter cannot isolate."""
        names = []

        def add_targets(target: ast.AST) -> None:
            names.extend(self._target_names(target))

        class Finder(ast.NodeVisitor):
            def visit_FunctionDef(self, child):
                names.append(child.name)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, child):
                names.append(child.name)

            def visit_Lambda(self, child):
                return

            def visit_Assign(self, child):
                for target in child.targets:
                    add_targets(target)
                self.generic_visit(child.value)

            def visit_AnnAssign(self, child):
                add_targets(child.target)
                if child.value is not None:
                    self.visit(child.value)

            def visit_AugAssign(self, child):
                add_targets(child.target)
                self.visit(child.value)

            def visit_NamedExpr(self, child):
                add_targets(child.target)
                self.visit(child.value)

            def visit_For(self, child):
                add_targets(child.target)
                self.generic_visit(child)

            visit_AsyncFor = visit_For

            def visit_With(self, child):
                for item in child.items:
                    if item.optional_vars is not None:
                        add_targets(item.optional_vars)
                self.generic_visit(child)

            visit_AsyncWith = visit_With

            def visit_ExceptHandler(self, child):
                if child.name:
                    names.append(child.name)
                self.generic_visit(child)

            def visit_Import(self, child):
                names.extend(alias.asname or alias.name.split(".")[0] for alias in child.names)

            def visit_ImportFrom(self, child):
                names.extend(alias.asname or alias.name for alias in child.names)

        Finder().visit(node)
        if names:
            return type(node).__name__, sorted(set(names))[0]
        return None

    def _unsupported_class_binding(self, node: ast.ClassDef) -> str | None:
        """Reject class attributes that shadow a renamed module binding.

        Class-body lookup is execution-order dependent: both ``offset =
        offset`` and ``value = offset; offset = 20`` can read the module
        binding before the class attribute exists.  A symbol table alone does
        not preserve that distinction, so reject the ambiguous construct.
        """
        for statement in node.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                for name in self._target_names(target):
                    if name in self._top_level_symbols:
                        return name
        return None

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        First pass: identify top-level functions and every module binding.

        Values are deliberately not restricted to literals.  ``helper =
        lambda ...``, ``helper = partial(...)``, and ``left, right = ...``
        all execute during module import and otherwise overwrite an earlier
        factor module's binding in a merged strategy.
        """
        for sub_node in node.body:
            if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._top_level_symbols.add(sub_node.name)
            elif isinstance(sub_node, ast.Assign):
                for target in sub_node.targets:
                    for name in self._target_names(target):
                        self._top_level_symbols.add(name)
            elif isinstance(sub_node, ast.AnnAssign) and sub_node.value is not None:
                for name in self._target_names(sub_node.target):
                    self._top_level_symbols.add(name)
            elif isinstance(sub_node, (ast.If, ast.Try, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Match)):
                unsupported = self._unsupported_control_flow_binding(sub_node)
                if unsupported is not None:
                    kind, name = unsupported
                    raise ValueError(
                        f"unsupported_module_binding_in_control_flow: {name} ({kind})"
                    )

        for sub_node in node.body:
            if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub_node.name in self._top_level_symbols:
                sub_node._namespace_original_name = sub_node.name
                sub_node.name = self._prefixed(sub_node.name)
            elif isinstance(sub_node, ast.Assign):
                for target in sub_node.targets:
                    self._rename_target(target)
            elif isinstance(sub_node, ast.AnnAssign):
                self._rename_target(sub_node.target)

        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._visit_function_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._visit_function_definition(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        unsupported = self._unsupported_class_binding(node)
        if unsupported is not None:
            raise ValueError(
                f"unsupported_class_binding_for_namespace_isolation: {unsupported}"
            )
        node.decorator_list = [self.visit(decorator) for decorator in node.decorator_list]
        node.bases = [self.visit(base) for base in node.bases]
        for keyword in node.keywords:
            keyword.value = self.visit(keyword.value)
        child_scope = self._push_scope(node)
        try:
            node.body = [self.visit(statement) for statement in node.body]
        finally:
            self._pop_scope(child_scope)
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        return self._visit_lambda(node)

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        return self._visit_comprehension_scope(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        return self._visit_comprehension_scope(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        return self._visit_comprehension_scope(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        return self._visit_comprehension_scope(node)

    def visit_Global(self, node: ast.Global) -> ast.Global:
        node.names = [
            self._prefixed(name) if name in self._top_level_symbols else name
            for name in node.names
        ]
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if (
            node.id in self._top_level_symbols
            and (
                isinstance(node.ctx, ast.Load) and self._resolves_to_module_binding(node.id)
                or self._is_declared_global(node.id)
            )
        ):
            node.id = self._prefixed(node.id)
        return node

def rename_top_level_functions(source_code: str, prefix: str) -> str:
    """
    Parses Python source code, renames top-level functions/module bindings and
    their internal references with a given prefix, and returns the modified
    code.

    Args:
        source_code: The Python source code to transform.
        prefix: The unique prefix to add to function names.

    Returns:
        The transformed Python source code.
    """
    if not prefix:
        return source_code
        
    tree = ast.parse(source_code)
    transformer = ASTPrefixer(prefix, source_code)
    transformed_tree = transformer.visit(tree)
    ast.fix_missing_locations(transformed_tree)
    # The unparse function was added in Python 3.9. Generated source is
    # normalized for the cloud's Python version by the caller afterwards.
    return ast.unparse(transformed_tree)
