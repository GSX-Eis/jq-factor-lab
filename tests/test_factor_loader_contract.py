"""Explicit loading must preserve identity and report construction failures."""
import sys
from pathlib import Path

import pytest

from backtest_and_analysis.factor_loader import FactorLoader


@pytest.fixture(autouse=True)
def restore_test_modules():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith('_factor_loader_'):
            sys.modules.pop(name, None)


def write_factor(tmp_path, source):
    path = tmp_path / 'factor.py'
    path.write_text('from utils.factor_base import FactorBase\n' + source)
    return path


def test_aliases_do_not_duplicate_classes_or_break_explicit_selection(tmp_path):
    path = write_factor(tmp_path, 'class Current(FactorBase):\n    pass\nOld = Current\n')
    classes = FactorLoader.load_factor_class_from_file(str(path))
    assert len(classes) == 1
    assert classes[0].__name__ == 'Current'
    selected = FactorLoader.load_factor_class_from_file(str(path), 'Current')
    assert len(selected) == 1
    assert selected[0].__name__ == 'Current'
    assert len(FactorLoader.load_factor_classes(str(tmp_path))) == 1


def test_distinct_subclasses_remain_distinct(tmp_path):
    path = write_factor(tmp_path, 'class A(FactorBase):\n    pass\nclass B(A):\n    pass\nAlias = A\n')
    assert [cls.__name__ for cls in FactorLoader.load_factor_class_from_file(str(path))] == ['A', 'B']


@pytest.mark.parametrize('directory', [False, True])
@pytest.mark.parametrize('good_first', [False, True])
def test_constructor_failure_is_raised_with_context(tmp_path, directory, good_first):
    source = '''class AGood(FactorBase):
    def __init__(self, *args, **kwargs): pass
class ZBad(FactorBase):
    def __init__(self, *args, **kwargs): raise ValueError("bad configuration")
'''
    if not good_first:
        source = source.replace('AGood', 'ZGood').replace('ZBad', 'ABad')
    path = write_factor(tmp_path, source)
    loader = FactorLoader.load_all_factors if directory else FactorLoader.load_factor_from_file
    with pytest.raises(RuntimeError) as error:
        loader(str(tmp_path if directory else path), None, '2024-01-01', '2024-01-31', ['A'], None)
    assert str(path) in str(error.value)
    assert ('ZBad' if good_first else 'ABad') in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == 'bad configuration'


def test_constructor_arguments_and_single_instance_are_preserved(tmp_path):
    path = write_factor(tmp_path, '''class Current(FactorBase):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
Alias = Current
''')
    results = FactorLoader.load_factor_from_file(str(path), None, '2024-01-01', '2024-01-31', ['A'], None, duckdb_path='input.duckdb')
    assert len(results) == 1
    assert results[0].args == (None, 'Current', '2024-01-01', '2024-01-31', ['A'])
    assert results[0].kwargs == {'cache_dir': None, 'duckdb_path': 'input.duckdb'}


def test_explicit_missing_class_still_errors(tmp_path):
    path = write_factor(tmp_path, 'class Current(FactorBase):\n    pass\n')
    with pytest.raises(ValueError, match='Expected one factor class'):
        FactorLoader.load_factor_class_from_file(str(path), 'Missing')


def test_module_failure_is_propagated_and_registration_removed(tmp_path):
    path = write_factor(tmp_path, 'raise ValueError("bad module")\n')
    with pytest.raises(ValueError, match='bad module'):
        FactorLoader.load_factor_class_from_file(str(path))
    assert FactorLoader._module_name_for_path(Path(path)) not in sys.modules


@pytest.mark.parametrize('good_first', [False, True])
def test_directory_class_import_failure_is_not_partial_success(tmp_path, good_first):
    good = tmp_path / ('a_good.py' if good_first else 'z_good.py')
    bad = tmp_path / ('z_bad.py' if good_first else 'a_bad.py')
    good.write_text('from utils.factor_base import FactorBase\nclass Good(FactorBase): pass\n')
    bad.write_text('raise ImportError("missing factor dependency")\n')

    with pytest.raises(ImportError, match='Failed to load factor module') as error:
        FactorLoader.load_factor_classes(str(tmp_path))
    assert str(bad) in str(error.value)
    assert isinstance(error.value.__cause__, ImportError)
    assert str(error.value.__cause__) == 'missing factor dependency'
    assert FactorLoader._module_name_for_path(bad) not in sys.modules


@pytest.mark.parametrize('instances', [False, True])
def test_directory_loaders_use_sorted_python_files_only(tmp_path, monkeypatch, instances):
    for filename, name in [('z.py', 'Z'), ('a.py', 'A')]:
        (tmp_path / filename).write_text(
            f'from utils.factor_base import FactorBase\nclass {name}(FactorBase):\n'
            '    def __init__(self, *args, **kwargs): pass\n'
        )
    (tmp_path / 'nested.py').mkdir()
    (tmp_path / 'nested.py' / 'child.py').write_text('raise AssertionError("do not recurse")\n')
    (tmp_path / '__init__.py').write_text('raise AssertionError("skip package entry")\n')
    (tmp_path / 'notes.txt').write_text('not Python')
    monkeypatch.setattr(
        'backtest_and_analysis.factor_loader.os.listdir',
        lambda _: ['z.py', 'nested.py', 'notes.txt', '__init__.py', 'a.py'],
    )
    if instances:
        loaded = FactorLoader.load_all_factors(str(tmp_path), None, '2024-01-01', '2024-01-31', ['A'], None)
        names = [type(obj).__name__ for obj in loaded]
    else:
        names = [cls.__name__ for cls in FactorLoader.load_factor_classes(str(tmp_path))]
    assert names == ['A', 'Z']
