"""Version lookup must not discover an unrelated enclosing repository."""
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
