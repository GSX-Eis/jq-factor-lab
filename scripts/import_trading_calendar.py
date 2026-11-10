"""Publish a locally verified daily calendar using the versioned input-store protocol."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from utils.local_outputs import managed_output

import duckdb
import pandas as pd

from utils.trading_calendar import TradingCalendar, DEFAULT_CONFIG
from utils.input_store import publish_input_batch
from utils.data_store import table_name


def import_calendar(database, csv_path, verification_path, *, config_path=None):
    csv_path, verification_path = Path(csv_path), Path(verification_path)
    receipt = json.loads(verification_path.read_text())
    spec = json.loads(Path(config_path or DEFAULT_CONFIG).read_text())
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if receipt.get('status') != 'verified' or receipt.get('differences') != {'local_only': [], 'online_only': []}:
        raise ValueError('Calendar requires successful local/external verification with zero differences')
    if receipt.get('files', {}).get(csv_path.name) != digest:
        raise ValueError('Calendar CSV hash differs from verification receipt')
    if receipt.get('calendar_id') != spec['calendar_id']:
        raise ValueError('Calendar ID differs from configuration')
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != ['date'] or len(frame) != receipt['trading_days']:
        raise ValueError('Calendar CSV schema/count differs from receipt')
    calendar = TradingCalendar(frame.date, receipt['start_date'], receipt['end_date'], spec['calendar_id'])
    dates = pd.date_range(calendar.start, calendar.end)
    dense = pd.DataFrame({'calendar_id': spec['calendar_id'], 'date': dates,
                          'is_open': dates.isin(calendar.sessions())})
    target = spec['table'] if '.' in spec['table'] else 'main.' + spec['table']
    parents = {}
    if Path(database).exists():
        with duckdb.connect(str(database), read_only=True) as conn:
            if conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='meta' AND table_name='input_heads'").fetchone()[0]:
                head = conn.execute('SELECT version FROM meta.input_heads WHERE target_table=?', [target]).fetchone()
                if head:
                    parents[spec['table']] = {'role': 'input', 'table': target, 'version': head[0]}
    metadata = {'source': 'local_database_crosschecked_external_calendar', 'source_hash': digest,
                'loader_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'verification': receipt,
                'verification_hash': hashlib.sha256(verification_path.read_bytes()).hexdigest()}
    # Disposable staging, never the only copy of either source or verification.
    with managed_output(None, 'verified-calendar-import') as (directory, _lease):
        staging = Path(directory) / 'calendar.duckdb'
        with duckdb.connect(str(staging)) as conn:
            conn.execute(f'CREATE TABLE {table_name(spec["table"])} '
                         '(calendar_id VARCHAR NOT NULL, date DATE NOT NULL, is_open BOOLEAN NOT NULL, '
                         'PRIMARY KEY(calendar_id,date))')
            conn.register('calendar_rows', dense)
            conn.execute(f'INSERT INTO {table_name(spec["table"])} SELECT * FROM calendar_rows')
        return publish_input_batch(database, staging, tables=[spec['table']], metadata=metadata, parents=parents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--config')
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    result = import_calendar(args.database, args.csv, args.verification, config_path=args.config)
    Path(args.receipt).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
