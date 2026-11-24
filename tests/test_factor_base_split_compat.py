"""Compatibility tests for the FactorBase facade and its extracted implementations."""
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from test_factor_base_frequency_routing import probe
from utils import factor_base
from utils.factor_base import (
    FactorBase,
    _get_pit_window_from_config,
    _load_factor_requirements,
    apply_pit_adjustment,
)
from utils.pit import apply_pit_adjustment as shared_pit_adjustment


UTILS = Path(__file__).resolve().parents[1] / 'utils'
CORE_FILES = ('__init__.py', 'bar_bridge.py', 'legacy_storage.py', 'cache.py', 'execution.py', 'slot_binding.py', 'slot_replay.py')


def test_factor_base_keeps_legacy_exports_and_source_location() -> None:
    assert {name for name in vars(factor_base) if not name.startswith('_')} == {
        'Any', 'Callable', 'Dict', 'List', 'Optional', 'Union',
        'DUCKDB_ADAPTER_AVAILABLE', 'DuckDBAdapter', 'FactorBase',
        'DynamicStockPoolManager', 'StockFilter', 'get_data_adapter',
        'apply_pit_adjustment', 'datetime', 'gc', 'json', 'logging', 'np',
        'os', 'pd', 'pickle', 'tempfile', 'time',
    }
    assert apply_pit_adjustment is shared_pit_adjustment
    assert FactorBase.__mro__ == (FactorBase, object)
    source = UTILS / 'factor_base.py'
    for definition in (FactorBase, FactorBase._is_cloud_environment,
                       _get_pit_window_from_config, _load_factor_requirements):
        assert inspect.getsourcefile(definition) == str(source)


def test_factor_core_module_sizes_and_import_boundaries() -> None:
    core = UTILS / 'factor_core'
    assert {path.name for path in core.glob('*.py')} == set(CORE_FILES)
    assert 280 <= len((UTILS / 'factor_base.py').read_text().splitlines()) <= 435
    for name in CORE_FILES[1:]:
        source = (core / name).read_text()
        assert len(source.splitlines()) <= 500, name
        # The implementation modules are independent leaves. The package collects
        # them and the facade passes any bindings they need from FactorBase.
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, name
                imports = [node.module, *(f'{node.module}.{alias.name}' for alias in node.names)]
            elif isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            else:
                continue
            assert not any(module.startswith(('utils.factor_base', 'utils.factor_core'))
                           for module in imports), name


@pytest.mark.parametrize('first', ['facade', 'core'])
def test_factor_base_imports_in_fresh_process(first: str) -> None:
    command = """
from importlib import import_module
import sys

if sys.argv[1] == 'core':
    import_module('utils.factor_core')
    assert 'utils.factor_base' not in sys.modules
base = import_module('utils.factor_base')
core = import_module('utils.factor_core')
assert base.FactorBase.__mro__ == (base.FactorBase, object)
assert core.bar_bridge and core.cache and core.execution and core.legacy_storage
"""
    subprocess.run([sys.executable, '-c', command, first], cwd=UTILS.parent, check=True, timeout=30)


@pytest.fixture
def copied_source(tmp_path: Path) -> tuple[Path, dict]:
    mapping = json.loads((UTILS / 'legacy_cache_identity.json').read_text())
    names = ('factor_base.py', *mapping['dependencies'], 'factor_core/slot_binding.py')
    for name in names:
        target = tmp_path / 'utils' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((UTILS / name).read_bytes())
    return tmp_path / 'utils' / 'factor_base.py', mapping


@pytest.mark.parametrize('route', ['bar', 'edited-daily-facade'])
def test_factor_core_edit_changes_nonbaseline_cache_key(
    copied_source: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    source, _ = copied_source
    original_getsourcefile = inspect.getsourcefile

    def getsourcefile(obj: object) -> str | None:
        return str(source) if obj is FactorBase else original_getsourcefile(obj)

    monkeypatch.setattr(inspect, 'getsourcefile', getsourcefile)
    if route == 'bar':
        factor = probe(FactorBase, data_frequency='5min', analysis_mode='time_series')
    else:
        source.write_bytes(source.read_bytes() + b'\n# Future facade edit\n')
        factor = probe(FactorBase)

    raw = pd.DataFrame({'date': pd.to_datetime(['2026-01-05']), 'code': ['A'], 'close': [10.]})
    before = factor._get_cache_key(raw)
    assert before is not None and before.startswith('factor-v3-')
    changed = source.parent / 'factor_core' / 'bar_bridge.py'
    changed.write_bytes(changed.read_bytes() + b'\n# Future implementation edit\n')
    after = factor._get_cache_key(raw)
    assert after is not None and after != before
