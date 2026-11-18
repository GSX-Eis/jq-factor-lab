"""Exact aliases; invalid inputs remain data until the caller rejects them."""
from dataclasses import dataclass
from typing import Any

import pandas as pd


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
            for minutes in (1, 5, 15, 30, 60):
                if value == f'{minutes}min':
                    return cls(value, 'intraday', minutes, 'canonical')
                if value == f'{minutes}m' or (minutes == 60 and value == '1h'):
                    return cls(value, 'intraday', minutes, 'legacy')
        return cls(value)

    @property
    def is_valid(self) -> bool:
        return self.kind is not None

    @property
    def canonical(self) -> str | None:
        return ('1d' if self.kind == 'session' else f'{self.minutes}min') if self.is_valid else None

    @property
    def joinquant(self) -> str | None:
        return ('1d' if self.kind == 'session' else f'{self.minutes}m') if self.is_valid else None

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
        return pd.Timedelta(days=1) if self.kind == 'session' else pd.Timedelta(minutes=self.minutes)

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
    def resolve(cls, raw: Any, analysis_mode: str | None) -> 'FactorFrequencyRoute':
        frequency = Frequency.parse(raw)
        explicit = frequency.spelling == 'canonical' or (raw == '5m' and analysis_mode is not None)
        # Session panels retain the stage-5 daily/PIT calculation contract.
        return cls(frequency, explicit, explicit and frequency.kind == 'intraday')
