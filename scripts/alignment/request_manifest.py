"""Fetch a frozen cloud request schedule before factor computation finishes."""
import argparse
import base64
import csv
import contextlib
import gzip
import io
import json
from pathlib import Path
import re
import sys
import time
from datetime import date

FIELDS = ('date', 'rebalance_date', 'pool_date', 'code', 'factor')

def request_keys(rows, factor):
    keys = []
    for row in rows:
        values = tuple(row[key] for key in FIELDS)
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError('invalid_request_fields')
        asof, rebalance, pool, code, name = values
        if name != factor or not '2022-01-01' <= asof <= rebalance <= '2023-12-31':
            raise ValueError('request_contract_mismatch')
        if pool > rebalance:
            raise ValueError('future_pool_request')
        for value in (asof, rebalance, pool):
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError('invalid_request_date')
        keys.append(values)
    if not keys or len({(r[1], r[3], r[4]) for r in keys}) != len(keys):
        raise ValueError('empty_or_duplicate_requests')
    return sorted(keys)

def materialize(payload, fingerprint, factor, target):
    if payload.get('fingerprint') != fingerprint:
        raise ValueError('request_fingerprint_mismatch')
    if payload.get('encoding') == 'gzip+base64':
        if 'rows' in payload:
            raise ValueError('ambiguous_request_encoding')
        raw = gzip.decompress(base64.b64decode(payload['rows_gzip_base64'], validate=True))
        decoded = json.loads(raw.decode('utf-8'))
    elif 'encoding' in payload:
        raise ValueError('unsupported_request_encoding')
    else:
        decoded = payload['rows']
    rows = request_keys(decoded, factor)
    temp = target.with_suffix('.tmp')
    with temp.open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    temp.replace(target)

def compare_requests(request_csv, cloud_csv, factor):
    with request_csv.open(newline='') as stream:
        expected = request_keys(csv.DictReader(stream), factor)
    with cloud_csv.open(newline='') as stream:
        actual = request_keys(csv.DictReader(stream), factor)
    if expected != actual:
        raise ValueError('cloud_request_schedule_changed')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fingerprint', required=True)
    parser.add_argument('--factor', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cloud-csv', type=Path)
    parser.add_argument('--timeout', type=float, default=5400)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.fingerprint):
        raise ValueError('invalid_request_fingerprint')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cloud_csv:
        compare_requests(args.output, args.cloud_csv, args.factor)
        print('request_schedule_match')
        return
    sys.path.insert(0, str(Path.cwd() / '.claude/skills/alignment-pipeline/scripts'))
    from extract_and_download import download_file
    filename = 'alignment_requests_' + args.fingerprint + '.json'
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        # The generic downloader previews an entire one-line JSON; keep only errors.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            downloaded = download_file(filename, str(args.output.parent))
        if downloaded:
            payload = json.loads((args.output.parent / filename).read_text())
            materialize(payload, args.fingerprint, args.factor, args.output)
            print('request_manifest_ready', flush=True)
            return
        print(captured.getvalue()[-2000:], flush=True)
        print('waiting_for_request_manifest', flush=True)
        time.sleep(min(15, max(0, deadline - time.monotonic())))
    raise TimeoutError('request_manifest_pending')

if __name__ == '__main__':
    main()
