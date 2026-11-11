import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

WORK = Path(__file__).resolve().parents[1]
FINGERPRINT = 'a' * 64


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def producer():
    path = WORK / 'jq_strategies/cloud_factor_dumper.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'durable_alignment_output')
    module = SimpleNamespace()
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), module.__dict__)
    module.ALIGNMENT_FINGERPRINT = FINGERPRINT
    return module


def downloader():
    return load('durable_downloader', WORK / 'scripts/alignment/download_cloud_evidence.py')


def test_journal_is_flushed_during_execution_and_survives_output_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = producer()
    path = tmp_path / ('alignment_cloud_' + FINGERPRINT + '.log')

    @module.durable_alignment_output
    def operation():
        print('中文 progress', flush=True)
        assert '中文 progress' in path.read_text()
        print('Total Rows: 12')
        return 42

    assert operation() == 42
    text = path.read_text()
    downloader().validate_durable_log(text, FINGERPRINT)
    assert 'Total Rows: 12' in text


def test_exception_preserves_traceback_without_a_success_marker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = producer()
    original = (sys.stdout, sys.stderr)

    @module.durable_alignment_output
    def operation():
        print('started calculation')
        raise RuntimeError('actual producer failure')

    with pytest.raises(RuntimeError, match='actual producer failure'):
        operation()
    text = (tmp_path / ('alignment_cloud_' + FINGERPRINT + '.log')).read_text()
    assert 'Traceback (most recent call last)' in text
    assert 'ALIGNMENT_PRODUCER_DONE' not in text
    assert (sys.stdout, sys.stderr) == original
    with pytest.raises(ValueError, match='terminal_missing'):
        downloader().validate_durable_log(text, FINGERPRINT)


def test_non_alignment_calls_preserve_legacy_behavior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = producer()
    module.ALIGNMENT_FINGERPRINT = ''
    assert module.durable_alignment_output(lambda: 'legacy')() == 'legacy'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('shadowed_value', [True, False])
def test_jq_wildcard_exports_cannot_change_fingerprint_validation(tmp_path, monkeypatch, shadowed_value):
    monkeypatch.chdir(tmp_path)
    module = producer()
    module.any = lambda values: shadowed_value
    assert module.durable_alignment_output(lambda: 'correct')() == 'correct'
    module.ALIGNMENT_FINGERPRINT = 'x' * 64
    with pytest.raises(ValueError, match='invalid alignment fingerprint'):
        module.durable_alignment_output(lambda: 'must not run')()


@pytest.mark.parametrize('text', [
    'ALIGNMENT_PRODUCER_STARTED|' + FINGERPRINT,
    'ALIGNMENT_PRODUCER_STARTED|' + 'b' * 64 + '\nALIGNMENT_PRODUCER_DONE|' + 'b' * 64,
    'ALIGNMENT_PRODUCER_STARTED|' + FINGERPRINT + '\nTraceback (most recent call last)\nALIGNMENT_PRODUCER_DONE|' + FINGERPRINT,
])
def test_incomplete_stale_or_failed_producers_are_rejected(text):
    with pytest.raises(ValueError):
        downloader().validate_durable_log(text, FINGERPRINT)


@pytest.mark.parametrize('complete', [False, True])
def test_download_uses_persisted_log_before_fetching_values(tmp_path, monkeypatch, complete):
    calls = []

    def download(name, directory):
        calls.append(name)
        text = 'ALIGNMENT_PRODUCER_STARTED|' + FINGERPRINT + '\nTotal Rows: 12\n'
        if complete:
            text += 'ALIGNMENT_PRODUCER_DONE|' + FINGERPRINT + '\n'
        (Path(directory) / name).write_text(text if name.endswith('.log') else 'date,code,factor,value\n')
        return True

    monkeypatch.setitem(sys.modules, 'extract_and_download', SimpleNamespace(download_file=download))
    monkeypatch.setattr(sys, 'argv', ['download', '--fingerprint', FINGERPRINT,
                                    '--output-dir', str(tmp_path)])
    if complete:
        assert downloader().main() == 0
        assert calls[-1] == 'cloud_factor_dump.csv'
        assert (tmp_path / 'cloud_data_dumper_log.txt').is_file()
    else:
        with pytest.raises(ValueError, match='terminal_missing'):
            downloader().main()
        assert 'cloud_factor_dump.csv' not in calls
