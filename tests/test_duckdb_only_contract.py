"""Guard against reintroducing a second local database runtime."""
import ast
from pathlib import Path
import subprocess


def test_runtime_has_no_legacy_database_imports():
    root = Path(__file__).resolve().parents[1]
    if (root / '.git').exists():
        names = subprocess.check_output(['git', 'ls-files', '*.py'], cwd=root, text=True).splitlines()
        paths = [root / name for name in names]
    else:
        paths = list(root.rglob('*.py'))
    violations = []
    for path in paths:
        if not path.exists() or 'tests' in path.relative_to(root).parts or path.name.startswith('test_'):
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            modules = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            if any(module and module.split('.')[0] in {'sqlite3', 'sqlalchemy'} for module in modules):
                violations.append(f'{path.relative_to(root)}:{node.lineno}')
    assert not violations, 'Non-DuckDB runtime imports: ' + ', '.join(violations)
