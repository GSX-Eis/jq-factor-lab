"""Download complete, fingerprint-bound producer evidence without DOM logs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def validate_durable_log(text: str, fingerprint: str) -> None:
    if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
        raise ValueError('invalid alignment fingerprint')
    lines = text.strip().splitlines()
    if not lines or lines[0] != 'ALIGNMENT_PRODUCER_STARTED|' + fingerprint:
        raise ValueError('cloud_producer_start_missing_or_stale')
    if lines[-1] != 'ALIGNMENT_PRODUCER_DONE|' + fingerprint:
        raise ValueError('cloud_producer_terminal_missing')
    if 'Traceback (most recent call last)' in text:
        raise ValueError('cloud_producer_traceback')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fingerprint', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--log-filename', default='cloud_data_dumper_log.txt')
    args = parser.parse_args()
    validate_durable_log('ALIGNMENT_PRODUCER_STARTED|' + args.fingerprint + '\n'
                         'ALIGNMENT_PRODUCER_DONE|' + args.fingerprint, args.fingerprint)
    if Path(args.log_filename).name != args.log_filename:
        raise ValueError('log filename must be a basename')
    helpers = Path.cwd() / '.claude/skills/quantation-alignment/scripts'
    sys.path.insert(0, str(helpers))
    from extract_and_download import download_file

    remote_log = 'alignment_cloud_' + args.fingerprint + '.log'
    if not download_file(remote_log, str(args.output_dir)):
        return 1
    downloaded = args.output_dir / remote_log
    validate_durable_log(downloaded.read_text(encoding='utf-8'), args.fingerprint)
    downloaded.replace(args.output_dir / args.log_filename)
    if not download_file('cloud_factor_dump.csv', str(args.output_dir)):
        return 1
    print('Complete producer journal and factor CSV downloaded for ' + args.fingerprint)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
