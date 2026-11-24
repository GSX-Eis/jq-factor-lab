"""Trading grid frequencies; invalid inputs remain data until rejected."""
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from utils.slots import MarketSlot

_INTRADAY = re.compile(r'([1-9][0-9]{0,8})(min|m|h)')
_MAX_MINUTES = pd.Timedelta.max // pd.Timedelta(minutes=1)
_LEGACY_MINUTES = (1, 5, 15, 30, 60)


@dataclass(frozen=True)
class Frequency:
    raw: Any
    kind: str | None = None
    minutes: int | None = None
    spelling: str | None = None

    @classmethod
    def parse(cls, value: Any) -> 'Frequency':
        if isinstance(value, str):
            if value in ('daily', '1d'):
                return cls(value, 'session', None, 'legacy' if value == 'daily' else 'canonical')
            if value == '1w':
                return cls(value, 'week', None, 'canonical')
            match = _INTRADAY.fullmatch(value)
            if match:
                count, suffix = match.groups()
                minutes = int(count) * (60 if suffix == 'h' else 1)
                if minutes <= _MAX_MINUTES:
                    return cls(value, 'intraday', minutes, 'canonical' if suffix == 'min' else 'legacy')
        return cls(value)

    @property
    def is_valid(self) -> bool:
        return self.kind is not None

    @property
    def canonical(self) -> str | None:
        if self.kind == 'intraday':
            return f'{self.minutes}min'
        return {'session': '1d', 'week': '1w'}.get(self.kind)

    @property
    def joinquant(self) -> str | None:
        return f'{self.minutes}m' if self.kind == 'intraday' else self.canonical

    @property
    def table_name(self) -> str:
        if not self.is_valid:
            raise ValueError(f'Invalid frequency: {self.raw!r}')
        return f'market_bars_{self.canonical}'

    @property
    def duration(self) -> pd.Timedelta:
        """Nominal width only; actual availability comes from BarSchedule."""
        if not self.is_valid:
            raise ValueError(f'Invalid frequency: {self.raw!r}')
        if self.kind == 'intraday':
            return pd.Timedelta(minutes=self.minutes)
        return pd.Timedelta(days=7 if self.kind == 'week' else 1)

    def aggregation_ratio(self, source: 'Frequency | str') -> int:
        """Check nominal widths; scheduled sessions determine actual bucket membership."""
        base = source if isinstance(source, Frequency) else self.parse(source)
        ratio, remainder = divmod(self.duration, base.duration)
        if ratio < 1 or remainder:
            raise ValueError(f'{self.raw!r} must be an integer multiple of {base.raw!r}')
        return ratio

    @classmethod
    def contract_duration(cls, value: Any) -> pd.Timedelta:
        """Preserve the stage-2 structural Bar contract, including nanoseconds.

        These arbitrary positive widths do not imply a supported trading grid.
        """
        parsed = cls.parse(value)
        if parsed.is_valid:
            return parsed.duration
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Invalid frequency: {value!r}')
        duration = pd.Timedelta(value)
        if pd.isna(duration) or duration <= pd.Timedelta(0):
            raise ValueError(f'Invalid frequency: {value!r}')
        return duration


@dataclass(frozen=True)
class FactorFrequencyRoute:
    frequency: Frequency
    explicit_interface: bool
    bar_grid: bool

    @classmethod
    def resolve(cls, raw: Any, analysis_mode: str | None, *, slot: 'MarketSlot | None' = None,
                bar_grid: bool = False) -> 'FactorFrequencyRoute':
        frequency = Frequency.parse(raw)
        if slot is not None or bar_grid:
            if not frequency.is_valid:
                raise ValueError(f'Invalid frequency: {raw!r}')
            return cls(frequency, True, True)
        # Legacy callers rely on the original parser's invalid result for other granularities.
        if frequency.kind == 'week' or (frequency.kind == 'intraday' and
                (frequency.minutes not in _LEGACY_MINUTES or (raw.endswith('h') and raw != '1h'))):
            frequency = Frequency(raw)
        explicit = frequency.spelling == 'canonical' or (raw == '5m' and analysis_mode is not None)
        # Session panels retain the stage-5 daily/PIT calculation contract.
        return cls(frequency, explicit, explicit and frequency.kind == 'intraday')
