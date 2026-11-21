"""Version lookup must not discover an unrelated enclosing repository."""
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('nested', [False, True])
def test_non_repository_uses_explicit_environment(tmp_path, monkeypatch, nested):
    from utils.code_version import resolve_code_version
    if nested:
        subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
        subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c',
            'user.email=test@example.org', 'commit', '--allow-empty', '-m', 'parent fixture'],
            check=True, capture_output=True)
    root = tmp_path / 'export'
    root.mkdir()
    monkeypatch.setenv('QUANTATION_CODE_VERSION', 'export-123')
    assert resolve_code_version(root) == ('export-123', 'env')


def test_package_and_unknown_fallbacks(tmp_path, monkeypatch):
    from importlib import metadata
    from utils.code_version import resolve_code_version
    monkeypatch.delenv('QUANTATION_CODE_VERSION', raising=False)
    monkeypatch.setattr(metadata, 'version', lambda name: '1.2.3')
    assert resolve_code_version(tmp_path) == ('1.2.3', 'package')
    def missing(name):
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata, 'version', missing)
    assert resolve_code_version(tmp_path) == ('unknown', 'unknown')


def test_git_is_bound_to_code_root_and_reports_dirty(tmp_path, monkeypatch):
    from utils.code_version import resolve_code_version
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    file = tmp_path / 'code.py'
    file.write_text('original')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'code.py'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.org', 'commit', '-m', 'fixture'], check=True, capture_output=True)
    expected = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.chdir(tmp_path.parent)
    assert resolve_code_version(tmp_path) == (expected, 'git')
    file.write_text('changed')
    assert resolve_code_version(tmp_path) == (expected + '-dirty', 'git')


@pytest.mark.parametrize(('relative_path', 'dirty'), [
    ('backtest_and_analysis/new.py', True),
    ('factor_synthesis/nested/new.py', True),
    ('factors/nested/新 因子.py', True),
    ('data_pipeline/new.py', True),
    ('utils/nested/new.py', True),
    ('tools/new.py', True),
    ('utils/new.md', False),
    ('tools/new.json', False),
    ('utils/ignored/new.py', False),
    ('work/new.py', False),
    ('reports/new.py', False),
    ('data/new.py', False),
    ('outputs/new.py', False),
    ('utils_extra/new.py', False),
    ('new.py', False),
])
def test_git_detects_only_unignored_core_python_files(
    tmp_path: Path, relative_path: str, dirty: bool,
) -> None:
    from utils.code_version import resolve_code_version

    git = ['git', '-C', str(tmp_path)]
    subprocess.run([*git, 'init'], check=True, capture_output=True)
    (tmp_path / '.gitignore').write_text('/utils/ignored/\n', encoding='utf-8')
    subprocess.run([*git, 'add', '.gitignore'], check=True)
    subprocess.run([*git, '-c', 'user.name=Test', '-c', 'user.email=test@example.org',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    revision = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()

    file = tmp_path / relative_path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()

    expected = revision + ('-dirty' if dirty else '')
    assert resolve_code_version(tmp_path) == (expected, 'git')


def test_worker_records_version_source_in_artifacts(tmp_path, monkeypatch):
    import json
    from test_time_series_cli_end_to_end import nq_bars, base_config
    from tools.agent_harness.time_series_analysis import evaluate
    import utils.code_version as versions
    monkeypatch.setattr(versions, 'resolve_code_version', lambda root: ('export-abc', 'env'))
    path, _ = nq_bars(tmp_path)
    output = tmp_path / 'evaluation'
    payload = evaluate(base_config(path), output)
    identity = json.loads((output / 'summary.json').read_text())['result_identity']
    assert identity['code_version'] == 'export-abc'
    assert identity['code_version_source'] == 'env'
    assert payload['metrics']['result_identity'] == identity
