"""Resolve original CLI bindings without importing the compatibility façade."""

import sys
from types import ModuleType
from typing import Any

_CLI_MODULE = "tools.agent_harness.cli"


class _ModuleDefaults:
    def __init__(self, modules: tuple[ModuleType, ...]) -> None:
        self._modules = modules

    def __getattr__(self, name: str) -> Any:
        for module in self._modules:
            if name in vars(module):
                return vars(module)[name]
        raise AttributeError(name)


def _cli_bindings(default_module: str, *dependencies: ModuleType) -> ModuleType | _ModuleDefaults:
    """Resolve legacy patches at call time, with local defaults for direct imports."""
    main = sys.modules.get("__main__")
    if main is not None and getattr(getattr(main, "__spec__", None), "name", None) == _CLI_MODULE:
        return main
    cli = sys.modules.get(_CLI_MODULE)
    if cli is not None:
        return cli
    return _ModuleDefaults((sys.modules[default_module], *dependencies))
