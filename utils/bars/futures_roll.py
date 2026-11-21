"""Offline, Session-bound futures selection using only closed prior Sessions."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Sequence
import uuid

import duckdb
import numpy as np
import pandas as pd

from data_pipeline.loaders._transactions import import_transaction
from .builder import instrument_calendar
from .catalog import BarCatalog, identifier, table_exists
from utils.code_version import resolve_code_version
from .source import BarSource, _instrument_ids
from utils.time_axis import BarSchedule, Frequency


def product_metadata(connection: duckdb.DuckDBPyConnection, product_root: str) -> pd.DataFrame:
    if not isinstance(product_root, str) or not product_root.strip():
        raise ValueError('product_root must be a nonempty string')
    if not table_exists(connection, 'instrument_metadata'):
        raise ValueError(f'Missing instrument metadata for product_root={product_root}')
    ids = [row[0] for row in connection.execute(
        "SELECT instrument_id FROM instrument_metadata WHERE product_root=? AND asset_class='futures'",
        [product_root]).fetchall()]
    if not ids:
        raise ValueError(f'Missing futures metadata for product_root={product_root}')
    metadata = BarSource(connection).get_instrument_metadata(ids)
    if metadata.contract_month.isna().any():
        raise ValueError(f'contract_month is required for product_root={product_root}')
    identities = {(row['calendar_id'], row['exchange'], row['rth_reference_calendar'])
                  for row in metadata.to_dict('records')}
    if len(identities) != 1:
        raise ValueError(f'Contracts must share a calendar for product_root={product_root}')
    return metadata.sort_values(['contract_month', 'instrument_id']).reset_index(drop=True)


def session_close(connection: duckdb.DuckDBPyConnection, metadata: dict,
                  session_id: str, field: str) -> dict:
    """Prefer a daily dataset; otherwise use the finest stored minute dataset.

    An existing daily dataset with a hole/NULL fails instead of hiding that hole.
    The last minute must be the scheduled close slot, never an earlier quote.
    """
    if field not in ('open_interest', 'close'):
        raise ValueError('session_close field must be open_interest or close')
    root, instrument = metadata['product_root'], metadata['instrument_id']
    catalog = BarCatalog(connection)
    entries = [entry for entry in catalog.available(instrument, asset_class='futures')
               if entry.timestamp_semantics == 'UTC_bar_start']
    frequencies = sorted({entry.frequency for entry in entries},
                         key=lambda freq: (freq != '1d', Frequency.parse(freq).duration))
    message = f'Missing {field}: product_root={root} Session={session_id} contract={instrument}'
    if not frequencies:
        raise ValueError(message)
    entry = catalog.resolve(frequencies[0], instrument, asset_class='futures')
    cal = instrument_calendar(metadata, session_id, session_id)
    slots = BarSchedule(cal, entry.frequency,
        bucket_strategy='session' if entry.frequency == '1d' else entry.bucket_strategy).for_session(session_id)
    columns = {row[0] for row in connection.execute(f'DESCRIBE {identifier(entry.table_name)}').fetchall()}
    if field not in columns or slots.empty:
        raise ValueError(message)
    complete = 'is_complete' if 'is_complete' in columns else 'true'
    row = connection.execute(f'SELECT {field}, data_version, {complete} FROM {identifier(entry.table_name)} '
        'WHERE instrument_id=? AND timestamp_utc=? AND frequency=?',
        [instrument, slots.start_utc.iloc[-1], entry.frequency]).fetchone()
    if (row is None or row[0] is None or not np.isfinite(row[0]) or not row[2]
            or (field == 'open_interest' and row[0] < 0)):
        raise ValueError(message)
    return dict(value=float(row[0]), table=entry.table_name, data_version=row[1],
                timestamp_utc=slots.start_utc.iloc[-1].isoformat(),
                calendar_hash=cal.calendar_hash, bucket_strategy=entry.bucket_strategy)


def build_roll_schedule(connection: duckdb.DuckDBPyConnection, *, product_roots: Sequence[str],
                        start: str, end: str, rule: str = 'max_oi', confirm_days: int = 1,
                        forward_only: bool = True) -> dict:
    roots = _instrument_ids(product_roots)
    if rule != 'max_oi':
        raise ValueError('rule must be max_oi')
    if isinstance(confirm_days, bool) or not isinstance(confirm_days, Integral) or confirm_days < 1:
        raise ValueError('confirm_days must be an integer >= 1')
    if not isinstance(forward_only, bool):
        raise ValueError('forward_only must be a boolean')
    revision, revision_source = resolve_code_version()
    jobs = []
    for root in roots:
        metadata = product_metadata(connection, root)
        meta = metadata.iloc[0].to_dict()
        # Validate the public date bounds before constructing the lookback calendar.
        requested = instrument_calendar(meta, start, end)
        if not requested.is_domestic_futures:
            raise ValueError(f'Only domestic futures supported: product_root={root}')
        existing = {}
        if table_exists(connection, 'futures_roll_schedule'):
            existing = {row['effective_session_id']: row for row in connection.execute(
                'SELECT * FROM futures_roll_schedule WHERE product_root=? ORDER BY effective_session_id',
                [root]).fetchdf().to_dict('records')}
        # Recompute persisted successors too: an earlier OI edit can change their state.
        effective_end = max([end, *[day for day in existing if day >= start]])
        prior = [day for day in existing if day < start]
        effective_start = min(start, (pd.Timestamp(max(prior))+pd.Timedelta(days=1)).strftime('%Y-%m-%d')) if prior else start
        span = max(45, int(confirm_days)*4)
        while True:
            padded_start = (pd.Timestamp(effective_start)-pd.Timedelta(days=span)).strftime('%Y-%m-%d')
            cal = instrument_calendar(meta, padded_start, effective_end)
            days = [day.strftime('%Y-%m-%d') for day in cal.sessions()]
            if sum(day < effective_start for day in days) >= confirm_days:
                break
            span *= 2
        records = metadata.to_dict('records')
        by_id = {row['instrument_id']: row for row in records}
        current = existing[max(prior)]['to_contract'] if prior else None
        observations = {}
        for pos, effective in enumerate(days):
            if effective < effective_start:
                continue
            decision_days = days[pos-int(confirm_days):pos]
            winners, inputs, sources = [], [], {}
            for decision in decision_days:
                # The declared universe includes every unexpired registered contract.
                active = [row for row in records if (
                    pd.Timestamp(row['contract_expiry']) + (pd.offsets.MonthEnd(0)
                        if len(row['contract_expiry']) == 7 else pd.Timedelta(0))) >= pd.Timestamp(decision)]
                if not active:
                    raise ValueError(f'Missing open_interest: product_root={root} Session={decision}')
                values = {}
                for row in active:
                    key = (decision, row['instrument_id'])
                    if key not in observations:
                        observations[key] = session_close(connection, row, decision, 'open_interest')
                    observation = observations[key]
                    values[row['instrument_id']] = observation['value']
                    sources[f'{decision}/{row["instrument_id"]}'] = observation['table']
                    inputs.append([decision, row['instrument_id'], row['contract_month'], observation])
                winner = min(values, key=lambda contract: (
                    -values[contract], by_id[contract]['contract_month'], contract))
                winners.append(winner)
            decision = decision_days[-1]
            previous = current
            if current is None:
                # Bootstrap from the first observed leader, then apply the same streak rule.
                current = winners[0]
            candidate = winners[-1]
            confirmed = len(set(winners)) == 1
            if confirmed and (not forward_only or by_id[candidate]['contract_month'] > by_id[current]['contract_month']):
                current = candidate
            for contract in (previous, current):
                if contract and not any(item[:2] == [decision, contract] for item in inputs):
                    key = (decision, contract)
                    if key not in observations:
                        observations[key] = session_close(connection, by_id[contract], decision, 'open_interest')
                    observation = observations[key]
                    inputs.append([decision, contract, by_id[contract]['contract_month'], observation])
                    sources[f'{decision}/{contract}'] = observation['table']
            params = dict(confirm_days=int(confirm_days), forward_only=forward_only,
                          tie_break='contract_month_ascending_then_instrument_id', oi_sources=sources)
            payload = dict(inputs=inputs, previous_contract=previous, to_contract=current,
                           rule=rule, rule_params=params)
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode()).hexdigest()
            old = existing.get(effective)
            if old and old['source_hash'] == digest:
                continue
            # Both decision values must exist even when the incumbent is no longer active.
            def oi(contract):
                if contract is None:
                    return None
                key = (decision, contract)
                if key not in observations:
                    observations[key] = session_close(connection, by_id[contract], decision, 'open_interest')
                return observations[key]['value']
            jobs.append((root, effective, decision, previous, current, rule,
                json.dumps(params, sort_keys=True), oi(previous), oi(current), digest,
                int(old['version'])+1 if old else 1, uuid.uuid4().hex, revision, revision_source))
    if jobs:
        with import_transaction(connection):
            connection.execute('''CREATE TABLE IF NOT EXISTS futures_roll_schedule (
                product_root VARCHAR, effective_session_id VARCHAR, decision_session_id VARCHAR,
                from_contract VARCHAR, to_contract VARCHAR NOT NULL, rule VARCHAR, rule_params JSON,
                from_open_interest DOUBLE, to_open_interest DOUBLE, source_hash VARCHAR,
                version BIGINT, build_id VARCHAR, code_version VARCHAR, code_version_source VARCHAR,
                PRIMARY KEY (product_root, effective_session_id),
                CHECK (decision_session_id < effective_session_id))''')
            for row in jobs:
                connection.execute('INSERT OR REPLACE INTO futures_roll_schedule VALUES ('
                                   + ','.join('?' for _ in row) + ')', row)
    return dict(rebuilt_sessions=len(jobs), product_roots=roots, rule=rule)
