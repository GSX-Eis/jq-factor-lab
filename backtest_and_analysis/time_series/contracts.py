"""Pure time-series research contracts; no calendars, IO readers or evaluation."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from utils.time_axis import Frequency
if TYPE_CHECKING:
    from utils.time_axis import BarSchedule
from pandas.api.types import infer_dtype, is_bool_dtype, is_numeric_dtype, is_complex_dtype


class ContractValidationError(ValueError):
    """A structural input or record invariant is invalid."""


class DuplicateKeyError(ContractValidationError):
    """A table contains repeated research observation keys."""


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f'{name} must be a nonempty string')


def _utc(value: datetime, name: str) -> None:
    if (not isinstance(value, datetime) or pd.isna(value)
            or str(value.tzinfo) not in ('UTC', 'Etc/UTC')
            or value.utcoffset() != timedelta(0)):
        raise ContractValidationError(f'{name} must be a timezone-aware UTC datetime')


def _duration(frequency: str) -> pd.Timedelta:
    try:
        return Frequency.contract_duration(frequency)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractValidationError(f'Invalid frequency: {frequency!r}') from exc


def _number(value: float, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ContractValidationError(f'{name} must be a real number')


@dataclass(frozen=True)
class Bar:
    instrument_id: str
    timestamp_utc: datetime
    frequency: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    data_version: str

    def __post_init__(self) -> None:
        for name in ('instrument_id', 'source', 'data_version'):
            _text(getattr(self, name), name)
        _utc(self.timestamp_utc, 'timestamp_utc')
        _duration(self.frequency)
        for name in ('open', 'high', 'low', 'close', 'volume'):
            _number(getattr(self, name), name)
        if not math.isfinite(self.volume) or self.volume < 0:
            raise ContractValidationError('volume must be finite and nonnegative')

    @property
    def bar_end_utc(self) -> pd.Timestamp:
        return pd.Timestamp(self.timestamp_utc) + _duration(self.frequency)

    @property
    def has_valid_prices(self) -> bool:
        return (all(math.isfinite(p) and p > 0 for p in
                    (self.open, self.high, self.low, self.close))
                and self.low <= min(self.open, self.close)
                and self.high >= max(self.open, self.close)
                and self.low <= self.high)


@dataclass(frozen=True)
class FactorValue:
    timestamp_utc: datetime
    instrument_id: str
    factor_value: float

    def __post_init__(self) -> None:
        _utc(self.timestamp_utc, 'timestamp_utc')
        _text(self.instrument_id, 'instrument_id')
        _number(self.factor_value, 'factor_value')
        if math.isinf(self.factor_value):
            raise ContractValidationError('factor_value must not be infinite (NaN is allowed)')


BAR_COLUMNS = tuple(Bar.__dataclass_fields__)
FACTOR_COLUMNS = tuple(FactorValue.__dataclass_fields__)
BAR_KEY = ('instrument_id', 'timestamp_utc', 'frequency')
FACTOR_KEY = ('timestamp_utc', 'instrument_id')
PRICE_COLUMNS = ('open', 'high', 'low', 'close')


def _frame_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ContractValidationError('input must be a pandas DataFrame')
    if not frame.columns.is_unique:
        raise ContractValidationError('column names must be unique')
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ContractValidationError(f'missing columns: {sorted(missing)}')


def _text_column(series: pd.Series) -> None:
    if (infer_dtype(series, skipna=True) not in ('string', 'unicode', 'empty')
            or any(not isinstance(value, str) or not value.strip() for value in series.unique())):
        raise ContractValidationError(f'{series.name} must contain nonempty strings')


def _numeric_column(series: pd.Series) -> np.ndarray:
    if (not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)
            or is_complex_dtype(series.dtype)):
        raise ContractValidationError(f'{series.name} must contain real numbers')
    return series.to_numpy(dtype=float, na_value=np.nan)


def _mask_unavailable_factor_values(values: pd.Series, unavailable: np.ndarray) -> pd.Series:
    if not unavailable.any():
        return values
    # A float conversion could round valid integer ranks when adding a missing value.
    if isinstance(values.dtype, np.dtype) and values.dtype.kind in 'iu':
        masked = pd.arrays.IntegerArray(values.to_numpy(), unavailable)
        return pd.Series(masked, index=values.index, name=values.name)
    return values.mask(unavailable)


def _frame_keys(frame: pd.DataFrame, key: tuple[str, ...]) -> pd.DataFrame:
    times = frame['timestamp_utc']
    if (not isinstance(times.dtype, pd.DatetimeTZDtype)
            or str(times.dt.tz) not in ('UTC', 'Etc/UTC')
            or times.dt.tz.utcoffset(datetime(2000, 1, 1)) != timedelta(0)
            or times.isna().any()):
        raise ContractValidationError('timestamp_utc must have a UTC datetime dtype and no NaT')
    _text_column(frame['instrument_id'])
    duplicates = frame.duplicated(subset=list(key), keep=False)
    if duplicates.any():
        repeated = frame.loc[duplicates, list(key)].drop_duplicates().to_dict('records')
        raise DuplicateKeyError(f'duplicate keys {key}: {repeated}')
    return frame.sort_values(list(key), kind='stable').copy()


def invalid_price_mask(df: pd.DataFrame) -> pd.Series:
    """True for nonfinite, nonpositive or inconsistent OHLC; keep the row index."""
    _frame_columns(df, PRICE_COLUMNS)
    prices = np.column_stack([_numeric_column(df[name]) for name in PRICE_COLUMNS])
    opening, high, low, close = prices.T
    valid = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
             & (high >= np.maximum(opening, close))
             & (low <= np.minimum(opening, close)) & (low <= high))
    return pd.Series(~valid, index=df.index, name='invalid_price')


def validate_bar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate structural fields and return a sorted copy, retaining bad prices."""
    _frame_columns(df, BAR_COLUMNS)
    for name in ('source', 'data_version', 'frequency'):
        _text_column(df[name])
    for frequency in df.frequency.unique():
        _duration(frequency)
    for name in PRICE_COLUMNS:
        _numeric_column(df[name])
    volume = _numeric_column(df['volume'])
    if not (np.isfinite(volume) & (volume >= 0)).all():
        raise ContractValidationError('volume must be finite and nonnegative')
    return _frame_keys(df, BAR_KEY)


def validate_factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate factor keys/values and return a sorted copy; NaN is permitted."""
    _frame_columns(df, FACTOR_COLUMNS)
    if np.isinf(_numeric_column(df['factor_value'])).any():
        raise ContractValidationError('factor_value must not be infinite (NaN is allowed)')
    return _frame_keys(df, FACTOR_KEY)


def _validate_aligned_factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Require external values already aligned to the signal Bar clock."""
    frame = validate_factor_frame(df)
    release_columns = sorted({
        'signal_available_at', 'signal_available_at_utc', 'available_at', 'available_at_utc',
        'published_at', 'published_at_utc', 'high_tf_bar_end',
    }.intersection(frame.columns))
    if release_columns:
        raise ContractValidationError(
            f'External factor timing columns {release_columns} require prior alignment. '
            'Use backtest_and_analysis.evaluation.split.align_released_features per instrument '
            'at the actual signal Bar ends, then key the aligned values by signal Bar start.',
        )
    return frame


class ExclusionReason(str, Enum):
    FACTOR_MISSING = 'factor_missing'
    INSUFFICIENT_HISTORY = 'insufficient_history'
    SESSION_BOUNDARY = 'session_boundary'
    MISSING_BAR = 'missing_bar'
    INVALID_PRICE = 'invalid_price'


EXCLUSION_PRIORITY = tuple(ExclusionReason)


def primary_exclusion_reason(flags: Mapping[ExclusionReason, bool]) -> ExclusionReason | None:
    """Absent flags are False; return the first active reason in fixed priority."""
    if not isinstance(flags, Mapping) or set(flags) - set(EXCLUSION_PRIORITY):
        raise ContractValidationError('flags must map known exclusion reasons to booleans')
    if any(not isinstance(value, (bool, np.bool_)) for value in flags.values()):
        raise ContractValidationError('exclusion flags must be non-null booleans')
    return next((reason for reason in EXCLUSION_PRIORITY if flags.get(reason, False)), None)


def primary_exclusion_reasons(flags_df: pd.DataFrame) -> pd.Series:
    """Require all five boolean columns; return enum values or None, index intact."""
    columns = tuple(reason.value for reason in EXCLUSION_PRIORITY)
    _frame_columns(flags_df, columns)
    flags = flags_df.loc[:, list(columns)]
    if flags.isna().any().any() or any(not is_bool_dtype(t) for t in flags.dtypes):
        raise ContractValidationError('exclusion flags must be non-null booleans')
    values = flags.to_numpy(dtype=bool)
    reasons = np.asarray(EXCLUSION_PRIORITY, dtype=object)[values.argmax(axis=1)]
    reasons[~values.any(axis=1)] = None
    return pd.Series(reasons, index=flags_df.index, name='exclusion_reason')


@dataclass(frozen=True)
class LabelRecord:
    instrument_id: str
    frequency: str
    entry_offset_bars: int
    horizon_bars: int
    signal_timestamp: datetime
    signal_available_at: datetime
    entry_timestamp: datetime | None
    exit_bar_timestamp: datetime | None
    exit_timestamp: datetime | None
    entry_price: float | None
    exit_price: float | None
    price_change_points: float | None
    return_pct: float | None
    status: Literal['valid', 'excluded']
    exclusion_reason: ExclusionReason | None

    @classmethod
    def from_schedule(cls, schedule: BarSchedule, **values: Any) -> 'LabelRecord':
        """Validate actual ends without changing the frozen serialized field set."""
        if set(values) != {field.name for field in fields(cls)}:
            raise ContractValidationError('Supply exactly the LabelRecord fields')
        record = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(record, name, value)
        record.__post_init__(schedule)
        return record

    def __post_init__(self, schedule: BarSchedule | None = None) -> None:
        _text(self.instrument_id, 'instrument_id')
        duration = _duration(self.frequency)
        for name in ('entry_offset_bars', 'horizon_bars'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ContractValidationError(f'{name} must be an integer >= 1')
        if self.status not in ('valid', 'excluded'):
            raise ContractValidationError('status must be valid or excluded')
        if ((self.status == 'valid') != (self.exclusion_reason is None)
                or (self.exclusion_reason is not None
                    and not isinstance(self.exclusion_reason, ExclusionReason))):
            raise ContractValidationError('valid requires no exclusion_reason; excluded requires an ExclusionReason')
        for name in ('signal_timestamp', 'signal_available_at'):
            _utc(getattr(self, name), name)

        def expected_end(timestamp: datetime) -> pd.Timestamp:
            if schedule is None:
                return pd.Timestamp(timestamp) + duration
            if schedule.frequency.canonical != Frequency.parse(self.frequency).canonical:
                raise ContractValidationError('LabelRecord schedule frequency must match')
            try:
                return schedule.end_at(timestamp)
            except ValueError as exc:
                raise ContractValidationError(str(exc)) from exc

        if self.signal_available_at != expected_end(self.signal_timestamp):
            raise ContractValidationError('signal_available_at must equal signal_timestamp + frequency')
        for name in ('entry_timestamp', 'exit_bar_timestamp', 'exit_timestamp'):
            value = getattr(self, name)
            if value is None:
                if self.status == 'valid':
                    raise ContractValidationError(f'valid records require {name}')
            else:
                _utc(value, name)
                if value < self.signal_available_at:
                    raise ContractValidationError(f'{name} must not precede signal_available_at')
        if (self.entry_timestamp is not None and self.exit_bar_timestamp is not None
                and self.entry_timestamp > self.exit_bar_timestamp):
            raise ContractValidationError('entry_timestamp must not follow exit_bar_timestamp')
        if (self.exit_bar_timestamp is not None and self.exit_timestamp is not None
                and self.exit_timestamp != expected_end(self.exit_bar_timestamp)):
            raise ContractValidationError('exit_timestamp must equal exit_bar_timestamp + frequency')
        if (self.entry_timestamp is not None and self.exit_timestamp is not None
                and self.entry_timestamp >= self.exit_timestamp):
            raise ContractValidationError('entry_timestamp must precede exit_timestamp')
        for name in ('entry_price', 'exit_price', 'price_change_points', 'return_pct'):
            value = getattr(self, name)
            if value is None:
                if self.status == 'valid':
                    raise ContractValidationError(f'valid records require {name}')
            else:
                _number(value, name)
        if self.status == 'valid':
            for name in ('entry_price', 'exit_price', 'price_change_points', 'return_pct'):
                value = getattr(self, name)
                if not math.isfinite(value) or (name.endswith('price') and value <= 0):
                    raise ContractValidationError(f'{name} must be finite; prices must be positive')
            if not math.isclose(self.price_change_points, self.exit_price - self.entry_price,
                                rel_tol=0, abs_tol=1e-9):
                raise ContractValidationError('price_change_points must equal exit_price - entry_price')
            if not math.isclose(self.return_pct, (self.exit_price / self.entry_price - 1) * 100,
                                rel_tol=0, abs_tol=1e-9):
                raise ContractValidationError('return_pct must equal (exit_price / entry_price - 1) * 100')


def sha256_file(path: str | Path) -> str:
    """Hash file bytes in bounded chunks, without modifying the input."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResultIdentity:
    config_hash: str
    input_file_hashes: Mapping[str, str]
    calendar_hash: str
    code_version: str
    result_schema_version: str

    def __post_init__(self) -> None:
        for name in ('config_hash', 'calendar_hash'):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
                raise ContractValidationError(f'{name} must be a lowercase sha256 hex digest')
        for name in ('code_version', 'result_schema_version'):
            _text(getattr(self, name), name)
        if not isinstance(self.input_file_hashes, Mapping):
            raise ContractValidationError('input_file_hashes must be a file-to-sha256 mapping')
        for path, digest in self.input_file_hashes.items():
            _text(path, 'input file path')
            if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
                raise ContractValidationError(f'input hash for {path} must be a lowercase sha256 hex digest')
        # Frozen identity must not drift if the caller later mutates its mapping.
        object.__setattr__(self, 'input_file_hashes', MappingProxyType(dict(self.input_file_hashes)))

    def identity_hash(self) -> str:
        payload = {
            'config_hash': self.config_hash,
            'input_file_hashes': dict(self.input_file_hashes),
            'calendar_hash': self.calendar_hash,
            'code_version': self.code_version,
            'result_schema_version': self.result_schema_version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
