"""Frozen fixture decoding and the only seams for stage 2–4 implementations.

Stages 2–4 are wired. Never rewrite oracles.
"""
from pathlib import Path
from functools import lru_cache
import json
from typing import Any, Sequence

import pandas as pd

from backtest_and_analysis.time_series import (
    BarReturnLabeler, ExclusionReason, LabelRecord, filter_bars_by_view, read_bar_csv, read_factor_csv,
    TimeSeriesEvaluator,
)
from utils.session_calendar import SessionCalendar


FIXTURES = Path(__file__).parent / 'fixtures' / 'time_series'
DATA_VERSION = 'SYNTHETIC_NERE704_v1'
TIME_FIELDS = ('signal_timestamp', 'signal_available_at', 'entry_timestamp',
               'exit_bar_timestamp', 'exit_timestamp')


def load_expected(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f'expected_{name}.json').read_text(encoding='utf-8'))


def decode_record(raw: dict[str, Any]) -> LabelRecord:
    values = dict(raw)
    for field in TIME_FIELDS:
        if values[field] is not None:
            values[field] = pd.Timestamp(values[field])
    if values['exclusion_reason'] is not None:
        values['exclusion_reason'] = ExclusionReason(values['exclusion_reason'])
    return LabelRecord(**values)


@lru_cache(maxsize=1)
def _calendar() -> SessionCalendar:
    return SessionCalendar('NQ', '2026-12-01', '2026-12-02')


def session_date(timestamp: pd.Timestamp, view: str) -> str | None:
    """Return CME trading date (YYYY-MM-DD), or None outside the selected view."""
    session = _calendar().session_at(timestamp, view)
    return None if session is None else session.session_date


def expected_bar_times(start: pd.Timestamp, end: pd.Timestamp, view: str) -> list[pd.Timestamp]:
    """Return scheduled 1min starts in the half-open [start, end) interval."""
    return list(_calendar().expected_bar_opens_in_range(start, end, '1min', view))


def select_view(bars_csv: Path, view: str) -> pd.DataFrame:
    """Read CSV and retain only bars whose start belongs to the selected view."""
    return filter_bars_by_view(read_bar_csv(bars_csv), _calendar(), view)


def read_csv(path: Path, kind: str) -> pd.DataFrame:
    """Read kind=bars/factors, parse UTC, reject duplicate keys, sort stably.

    Return a RangeIndex so file row order is not part of the data identity.
    """
    if kind == 'bars':
        return read_bar_csv(path)
    if kind == 'factors':
        return read_factor_csv(path)
    raise ValueError('kind must be bars or factors')


def label_case(case: dict[str, Any], view: str) -> Sequence[LabelRecord]:
    """Read case paths under FIXTURES; label only case signal_timestamps.

    Filter non-view signals before producing records, sort by signal timestamp,
    and return exactly the LabelRecord fields. Excluded labels use None for
    entry/exit times and numeric outputs, retaining signal/availability times.
    Determine dataset coverage from the raw CSV before selecting the view.
    """
    labeler = BarReturnLabeler(_calendar(), entry_offset_bars=case['entry_offset_bars'],
                              horizons=[case['horizon_bars']], view=view)
    return labeler.label_records(read_bar_csv(FIXTURES / case['bars']),
                                 read_factor_csv(FIXTURES / case['factors']),
                                 signal_timestamps=case['signal_timestamps'])


def evaluate_case(case: dict[str, Any], view: str) -> dict[str, Any]:
    """Run reader → calendar/view → labels → evaluator on every input signal.

    Return {valid: int, excluded: int, exclusion_counts: {reason: int}}.
    All signals for the smoke fixture are specified in case signal_timestamps.
    """
    factors = read_factor_csv(FIXTURES / case['factors'])
    labeler = BarReturnLabeler(_calendar(), entry_offset_bars=case['entry_offset_bars'],
                              horizons=[case['horizon_bars']], view=view)
    labels = labeler.label(read_bar_csv(FIXTURES / case['bars']), factors,
                           signal_timestamps=case['signal_timestamps'])
    result = TimeSeriesEvaluator(analysis_mode='time_series').evaluate(labels, factors)
    counts = result.exclusion_summary
    return {'valid': counts['valid_count'], 'excluded': counts['excluded_count'],
            'exclusion_counts': counts['nonzero_exclusion_counts']}
