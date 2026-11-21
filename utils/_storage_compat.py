"""Resolve runtime overrides on the original storage modules."""
import sys
from typing import TypeVar


_Binding = TypeVar("_Binding")


def _legacy_binding(module_name: str, name: str, fallback: _Binding) -> _Binding:
    """Use the legacy binding when ready, without importing or waiting on the facade."""
    return getattr(sys.modules.get(module_name), name, fallback)
