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




def downloader():
    return load('durable_downloader', WORK / 'scripts/alignment/download_cloud_evidence.py')










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
