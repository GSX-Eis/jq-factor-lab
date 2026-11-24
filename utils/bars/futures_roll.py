"""Offline, Session-bound futures selection using only closed prior Sessions."""
from __future__ import annotations

import hashlib
import json
from math import fsum
from numbers import Integral
from typing import Sequence
import uuid

import duckdb
import numpy as np
import pandas as pd

from data_pipeline.loaders._transactions import import_transaction
from data_pipeline.loaders.market_bar_importer import calendar_name
from slots.futures.policy import (
    DaysBeforeExpiry, RollContract, RollInputs, RollPolicy, expiry_date, resolve_roll_policy,
)
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
    identities = {(calendar_name(row), row['exchange'], row['rth_reference_calendar'], row['timezone'])
                  for row in metadata.to_dict('records')}
    if len(identities) != 1:
        raise ValueError(f'Contracts must share a calendar for product_root={product_root}')
    return metadata.sort_values(['contract_month', 'instrument_id']).reset_index(drop=True)


class MissingSessionCloseError(ValueError):
    """A scheduled close observation is absent or invalid."""


def session_close(connection: duckdb.DuckDBPyConnection, metadata: dict,
                  session_id: str, field: str) -> dict:
    """Read the close snapshot or the volume of the entire closed Session.

    Close and OI retain daily precedence and require the scheduled close slot.
    Volume can use any complete representation of the same Session.
    """
    if field not in ('open_interest', 'volume', 'close'):
        raise ValueError('session_close field must be open_interest, volume or close')
    root, instrument = metadata['product_root'], metadata['instrument_id']
    catalog = BarCatalog(connection)
    entries = [entry for entry in catalog.available(instrument, asset_class='futures')
               if entry.timestamp_semantics == 'UTC_bar_start' and Frequency.parse(entry.frequency).kind != 'week']
    frequencies = sorted({entry.frequency for entry in entries},
                         key=lambda freq: (freq != '1d', Frequency.parse(freq).duration))
    message = f'Missing {field}: product_root={root} Session={session_id} contract={instrument}'
    if not frequencies:
        raise MissingSessionCloseError(message)
    cal = instrument_calendar(metadata, session_id, session_id)
    whole_session = field == 'volume'
    for frequency in frequencies if whole_session else frequencies[:1]:
        entry = catalog.resolve(frequency, instrument, asset_class='futures')
        slots = BarSchedule(cal, entry.frequency,
            bucket_strategy='session' if frequency == '1d' else entry.bucket_strategy).for_session(session_id)
        table = identifier(entry.table_name)
        columns = {row[0] for row in connection.execute(f'DESCRIBE {table}').fetchall()}
        if field not in columns or slots.empty:
            continue
        starts = slots.start_utc.tolist() if whole_session else [slots.start_utc.iloc[-1]]
        complete = 'is_complete' if 'is_complete' in columns else 'true'
        rows = connection.execute(f'SELECT {field}, data_version, {complete} FROM {table} '
            'WHERE instrument_id=? AND timestamp_utc=ANY(?) AND frequency=?',
            [instrument, starts, frequency]).fetchall()
        if len(rows) != len(starts) or any(
            value is None or not np.isfinite(value) or not valid
            or (field in ('open_interest', 'volume') and value < 0)
            for value, _, valid in rows
        ):
            continue
        value = fsum(row[0] for row in rows) if whole_session else float(rows[0][0])
        version = ','.join(sorted({row[1] for row in rows})) if whole_session else rows[0][1]
        return dict(value=value, table=entry.table_name, data_version=version,
                    timestamp_utc=starts[0].isoformat(), calendar_hash=cal.calendar_hash,
                    bucket_strategy=entry.bucket_strategy)
    raise MissingSessionCloseError(message)


def build_roll_schedule(connection: duckdb.DuckDBPyConnection, *, product_roots: Sequence[str],
                        start: str, end: str, rule: str | RollPolicy = 'max_oi',
                        confirm_days: int = 1, forward_only: bool = True) -> dict:
    """Materialize policy decisions in the existing per-product session table."""
    roots = _instrument_ids(product_roots)
    policy = resolve_roll_policy(rule)
    rule_name = policy.name
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
        instrument_calendar(meta, start, end)
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
        by_id = metadata.set_index('instrument_id', drop=False).to_dict('index')
        contracts = tuple(RollContract(row['instrument_id'], row['contract_month'],
                                       expiry_date(row['contract_expiry'])) for row in by_id.values())
        prior_row = existing[max(prior)] if prior else None
        if prior_row and prior_row['rule'] != rule_name:
            raise ValueError(f'Rebuild {root} from its first effective Session to change the roll policy')
        current = prior_row['to_contract'] if prior_row else None
        if current is not None and current not in by_id:
            raise ValueError(f'Missing incumbent contract={current} for product_root={root}')
        observations = {}
        expiry_calendars = {}

        def sessions_to_expiry(effective: str, expiry: pd.Timestamp) -> int:
            if expiry not in expiry_calendars:
                expiry_calendars[expiry] = (cal if expiry <= cal.end else
                    instrument_calendar(meta, effective_start, expiry.strftime('%Y-%m-%d')))
            return len(expiry_calendars[expiry].sessions(effective, expiry)) - 1

        for pos, effective in enumerate(days):
            if effective < effective_start:
                continue
            decision_days = tuple(days[pos-int(confirm_days):pos])
            decision = decision_days[-1]
            if cal.session_on(decision).segments[-1][1] > cal.session_on(effective).segments[0][0]:
                raise ValueError(f'Decision Session must close before effective open: {root} {effective}')
            inputs, sources, recorded = [], {}, set()

            def observe(day: str, contract: str, field: str) -> float:
                if day not in decision_days or contract not in by_id:
                    raise ValueError(f'Roll observation must belong to the decision window for {root}: {day}/{contract}')
                key = (day, contract, field)
                if key not in observations:
                    observations[key] = session_close(connection, by_id[contract], day, field)
                observation = observations[key]
                if key not in recorded:
                    evidence = observation if rule_name == 'max_oi' else dict(observation, field=field)
                    inputs.append([day, contract, by_id[contract]['contract_month'], evidence])
                    sources[f'{day}/{contract}'] = observation['table']
                    recorded.add(key)
                return observation['value']

            previous = current
            try:
                current = policy.select(RollInputs(contracts, current, decision_days, effective,
                                                   forward_only, observe, sessions_to_expiry))
            except MissingSessionCloseError:
                raise
            except ValueError as exc:
                raise ValueError(f'Roll policy {rule_name} failed for product_root={root}: {exc}') from exc
            if not isinstance(current, str) or current not in by_id:
                raise ValueError(f'Roll policy {rule_name} selected an unknown contract for {root}: {current!r}')
            previous_oi = current_oi = None
            if rule_name == 'max_oi':
                # Existing rows retain their evidence order and source hash.
                previous_oi = observe(decision, previous, 'open_interest') if previous else None
                current_oi = observe(decision, current, 'open_interest')
            params = dict(confirm_days=int(confirm_days), forward_only=forward_only,
                          tie_break='contract_month_ascending_then_instrument_id')
            if rule_name == 'max_oi':
                params['oi_sources'] = sources
            elif rule_name == 'max_volume':
                params['volume_sources'] = sources
            elif isinstance(policy, DaysBeforeExpiry):
                params = {'days_before_expiry': int(policy.days),
                          'tie_break': 'expiry_then_contract_month_then_instrument_id',
                          'expiries': {contract.instrument_id: contract.expiry.strftime('%Y-%m-%d')
                                       for contract in contracts}}
            payload = dict(inputs=inputs, previous_contract=previous, to_contract=current,
                           rule=rule_name, rule_params=params)
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode()).hexdigest()
            old = existing.get(effective)
            if old and old['source_hash'] == digest:
                continue
            jobs.append((root, effective, decision, previous, current, rule_name,
                json.dumps(params, sort_keys=True), previous_oi, current_oi, digest,
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
    return dict(rebuilt_sessions=len(jobs), product_roots=roots, rule=rule_name)
