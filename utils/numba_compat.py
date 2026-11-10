"""One Numba contract: native locally, explicit serial execution in cloud bundles."""

import ast
import types


def _serial_jit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return lambda function: function


SERIAL_NUMBA = types.SimpleNamespace(jit=_serial_jit, njit=_serial_jit, prange=range)
NUMBA_SERIAL = globals().get("NUMBA_SERIAL", False)
if NUMBA_SERIAL:
    numba = SERIAL_NUMBA
else:
    try:
        import numba
    except ImportError:
        numba = SERIAL_NUMBA

jit = numba.jit
njit = numba.njit
prange = numba.prange
conditional_jit = jit
IS_JOINQUANT = NUMBA_SERIAL


def normalize_numba_imports(source, inline=False):
    """Normalize imports before loading; preserve locations for local JIT caching.

    Cloud assembly binds aliases to the inlined serial backend without changing
    process-wide modules or removing decorators from individual factors.
    """

    class Imports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.level or node.module not in (
                "numba",
                "jq_adapter",
                "utils.jq_adapter",
                "utils.numba_compat",
            ):
                return node
            for alias in node.names:
                if alias.name not in ("jit", "njit", "prange", "IS_JOINQUANT"):
                    raise ValueError("unsupported_numba_import: " + alias.name)
            if not inline:
                node.module = "utils.numba_compat"
                return node
            return [
                ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=a.asname or a.name, ctx=ast.Store())],
                        value=(
                            ast.Name(id="IS_JOINQUANT", ctx=ast.Load())
                            if a.name == "IS_JOINQUANT"
                            else ast.Attribute(
                                value=ast.Name(id="numba", ctx=ast.Load()),
                                attr=a.name,
                                ctx=ast.Load(),
                            )
                        ),
                    ),
                    node,
                )
                for a in node.names
            ]

        def visit_Import(self, node):
            result = []
            for alias in node.names:
                if alias.name != "numba":
                    result.append(ast.copy_location(ast.Import(names=[alias]), node))
                elif inline:
                    result.append(
                        ast.copy_location(
                            ast.Assign(
                                targets=[
                                    ast.Name(
                                        id=alias.asname or "numba", ctx=ast.Store()
                                    )
                                ],
                                value=ast.Name(id="numba", ctx=ast.Load()),
                            ),
                            node,
                        )
                    )
                else:
                    result.append(
                        ast.copy_location(
                            ast.ImportFrom(
                                module="utils.numba_compat",
                                level=0,
                                names=[ast.alias(name="numba", asname=alias.asname)],
                            ),
                            node,
                        )
                    )
            return result

    return ast.fix_missing_locations(Imports().visit(ast.parse(source)))
