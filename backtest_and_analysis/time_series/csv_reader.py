"""Standard CSV ingestion and calendar-backed Bar availability; no imputation."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .contracts import (
    ContractValidationError, _duration, _text, validate_bar_frame, validate_factor_frame,
)

if TYPE_CHECKING:
    from utils.session_calendar import SessionCalendar, View


BarStatus = Literal['observed', 'market_closed', 'missing_bar', 'insufficient_history']


def _read_csv(path: str | Path, kind: Literal['bars', 'factors']) -> pd.DataFrame:
    text = ['timestamp_utc', 'instrument_id', 'timestamp', 'symbol', 'source', 'data_version']
    numeric = ['factor_value']
    if kind == 'bars':
        text.append('frequency')
        numeric = ['open', 'high', 'low', 'close', 'volume', 'factor_value']
    dtypes = {name: 'string' for name in text}
    try:
        # pandas renames duplicate headers, which would bypass the frame contract.
        with Path(path).open(encoding='utf-8-sig', newline='') as source:
            header = next((fields for fields in csv.reader(source)
                           if any(value.strip() for value in fields)), [])
        if len(set(header)) != len(header):
            raise ContractValidationError('column names must be unique')
        aliases = {'timestamp': 'timestamp_utc', 'symbol': 'instrument_id'}
        for alias, canonical in aliases.items():
            if alias in header and canonical in header:
                raise ContractValidationError(f'ambiguous columns: {canonical} and {alias}')
        frame = pd.read_csv(path, dtype=dtypes)
        frame = frame.rename(columns={a: c for a, c in aliases.items() if a in header})
    except (ValueError, TypeError, csv.Error, pd.errors.ParserError) as exc:
        raise ContractValidationError(f'Invalid {kind} CSV {path}: {exc}') from exc
    if 'timestamp_utc' not in frame:
        raise ContractValidationError('missing columns: timestamp_utc')
    # Infer populated numeric columns so booleans are not coerced to 0/1 floats.
    if frame.empty:
        for name in numeric:
            if name in frame:
                frame[name] = frame[name].astype('float64')
    times = frame['timestamp_utc']
    # Check original text before utc=True can silently convert an offset or naive time.
    pattern = r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|\+00:00|\+0000)'
    if not times.str.fullmatch(pattern).fillna(False).all():
        raise ContractValidationError('timestamp_utc must explicitly specify UTC (Z, +00:00 or +0000)')
    try:
        frame['timestamp_utc'] = pd.to_datetime(times, format='ISO8601', utc=True).astype('datetime64[ns, UTC]')
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractValidationError(f'Invalid timestamp_utc: {exc}') from exc
    validate = validate_bar_frame if kind == 'bars' else validate_factor_frame
    return validate(frame).reset_index(drop=True)


def _record_raw_coverage(bars: pd.DataFrame) -> None:
    if 'raw_coverage_end_utc' in bars.attrs:
        return
    bars.attrs['raw_coverage_start_utc'] = bars.timestamp_utc.min()
    bars.attrs['raw_coverage_end_utc'] = bars.timestamp_utc.max()
    for name, keys in [('raw_coverage_by_instrument', ['instrument_id']),
                       ('raw_coverage_by_key', ['instrument_id', 'frequency'])]:
        bounds = bars.groupby(keys).timestamp_utc.agg(['min', 'max'])
        bars.attrs[name] = {key: (start, end) for key, start, end in bounds.itertuples(name=None)}


def read_bar_csv(path: str | Path) -> pd.DataFrame:
    """Read explicit UTC Bar starts, validate, sort/reset index and retain bad prices.

    Raw coverage is the chronological minimum/maximum before any view selection;
    it is also recorded separately per instrument and (instrument, frequency).
    """
    frame = _read_csv(path, 'bars')
    _record_raw_coverage(frame)
    return frame


def read_factor_csv(path: str | Path) -> pd.DataFrame:
    """Read explicit UTC factor timestamps; validate unique keys and sort/reset."""
    return _read_csv(path, 'factors')


def _check_view(bars: pd.DataFrame, calendar: SessionCalendar, view: View) -> None:
    if view not in ('ETH', 'RTH'):
        raise ContractValidationError('view must be ETH or RTH')
    if bars.attrs.get('session_view', view) != view:
        raise ContractValidationError('Cannot change view after filtering; use the raw bars')
    hashes = bars.attrs.get('calendar_hash', calendar.calendar_hash)
    if isinstance(hashes, dict):
        hashes = {hashes.get(instrument) for instrument in bars.instrument_id.unique()}
        matches = hashes <= {calendar.calendar_hash}
    else:
        matches = hashes == calendar.calendar_hash
    if not matches:
        raise ContractValidationError('Cannot change calendar after filtering; use the raw bars')


def filter_bars_by_view(bars: pd.DataFrame, calendar: SessionCalendar,
                         view: View = 'ETH') -> pd.DataFrame:
    """Keep starts in calendar segments, preserving raw coverage and caller attrs.

    The supplied calendar must apply to every instrument in this frame. Group
    instruments by calendar before calling when their markets differ.
    """
    frame = validate_bar_frame(bars).reset_index(drop=True)
    _check_view(frame, calendar, view)
    _record_raw_coverage(frame)
    membership = {ts: calendar.session_at(ts, view) is not None
                  for ts in frame.timestamp_utc.unique()}
    selected = frame.loc[frame.timestamp_utc.map(membership).astype(bool)].reset_index(drop=True)
    selected.attrs.update(session_view=view, calendar_hash=calendar.calendar_hash)
    return selected


class BarAvailability:
    """Snapshot observed keys and raw bounds for repeated calendar-backed queries.

    Accept raw bars or the same calendar/view's filtered bars. A market-closed
    timestamp never counts as missing, even beyond the data tail. Unknown/empty
    instrument-frequency series have insufficient_history at scheduled starts.
    """

    def __init__(self, bars: pd.DataFrame, calendar: SessionCalendar,
                 view: View = 'ETH', *, bucket_strategy: str | None = None) -> None:
        frame = validate_bar_frame(bars)
        _check_view(frame, calendar, view)
        _record_raw_coverage(frame)
        self._calendar = calendar
        self._view = view
        self._bucket_strategy = bucket_strategy
        self._coverage = dict(frame.attrs['raw_coverage_by_key'])
        complete = frame.loc[frame.is_complete.fillna(False)] if 'is_complete' in frame else frame
        self._observed = set(zip(complete.instrument_id, complete.timestamp_utc, complete.frequency))

    def status(self, instrument_id: str, timestamp_utc: datetime,
               frequency: str) -> BarStatus:
        """Classify one scheduled start; off-grid intrasegment queries are invalid.

        Missing scheduled starts at/before the raw tail are missing_bar; starts
        after it are insufficient_history. Prices do not affect this presence
        check; their validity remains a separate downstream exclusion.
        """
        _text(instrument_id, 'instrument_id')
        _duration(frequency)
        session = self._calendar.session_at(timestamp_utc, self._view)
        if session is None:
            return 'market_closed'
        ts = pd.Timestamp(timestamp_utc)
        if ts not in self._calendar.expected_bar_opens(session, frequency, self._view,
                                                       bucket_strategy=self._bucket_strategy):
            raise ContractValidationError('timestamp_utc is not on the expected Bar start grid')
        bounds = self._coverage.get((instrument_id, frequency))
        if bounds is None or ts > bounds[1]:
            return 'insufficient_history'
        if (instrument_id, ts, frequency) in self._observed:
            return 'observed'
        return 'missing_bar'
