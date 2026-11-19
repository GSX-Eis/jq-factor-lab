"""Results of cross-sectional and time-series factor evaluation."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .kernels import CorrelationResult, QuantileResult

FloatArray = npt.NDArray[np.float64]
ReturnMode = Literal["raw", "cross_section_excess"]
ICMethod = Literal["spearman", "pearson"]


@dataclass(frozen=True)
class ICSummary:
    n_valid: int
    mean: float
    median: float
    positive_fraction: float
    sample_std: float
    icir: float
    reason: str | None


@dataclass(frozen=True)
class HACInterval:
    n_valid: int
    grid_size: int
    lag: int
    alpha: float
    pair_counts: npt.NDArray[np.intp]
    variance: float
    lower: float
    upper: float
    reason: str | None


@dataclass(frozen=True)
class TimeMean:
    n_times: int
    mean: float
    standard_error: float
    reason: str | None


@dataclass(frozen=True)
class DirectionalAccuracy:
    hits: int
    denominator: int
    up_count: int
    accuracy: float
    up_baseline: float
    lift: float
    reason: str | None


@dataclass(frozen=True)
class CrossSectionHorizon:
    ic: CorrelationResult
    ic_summary: ICSummary
    hac: HACInterval
    bucket_returns: FloatArray
    bucket_counts: npt.NDArray[np.intp]
    bucket_summary: dict[int, TimeMean]
    spread: FloatArray
    spread_summary: TimeMean
    direction: DirectionalAccuracy
    direction_by_bucket: dict[int, DirectionalAccuracy]
    bucket_standard_errors: FloatArray | None = None


@dataclass(frozen=True)
class CrossSectionResult:
    datetimes: pd.DatetimeIndex
    assets: pd.Index
    min_samples: int
    ic_method: ICMethod
    return_mode: ReturnMode
    direction: int
    low_quantile: int
    high_quantile: int
    diagnostic_period: int
    quantiles: QuantileResult
    quantile_reasons: npt.NDArray[np.object_]
    rank_autocorrelation: FloatArray
    turnover: FloatArray
    horizons: dict[Hashable, CrossSectionHorizon]


@dataclass(frozen=True)
class BootstrapInterval:
    grid_size: int
    block_length: int
    seed: int
    n_boot: int
    valid_replicates: int
    min_valid_replicates: int
    alpha: float
    lower: float
    upper: float
    reason: str | None

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha


@dataclass(frozen=True)
class ReturnDistribution:
    n_samples: int
    mean: float = np.nan
    standard_error: float = np.nan
    p5: float = np.nan
    p25: float = np.nan
    p50: float = np.nan
    p75: float = np.nan
    p95: float = np.nan
    mean_le_p5: float = np.nan
    mean_le_p25: float = np.nan
    mean_ge_p75: float = np.nan
    mean_ge_p95: float = np.nan
    reason: str | None = None


@dataclass(frozen=True)
class TimeSeriesInstrument:
    ic: CorrelationResult
    spearman_ic: CorrelationResult
    bootstrap: BootstrapInterval
    returns: ReturnDistribution
    returns_by_bucket: dict[int, ReturnDistribution]
    direction: DirectionalAccuracy
    direction_by_bucket: dict[int, DirectionalAccuracy]
    non_overlapping_ic: dict[int, CorrelationResult]


@dataclass(frozen=True)
class InstrumentICSummary:
    n_instruments: int
    n_valid: int
    mean: float
    median: float
    reason: str | None


@dataclass(frozen=True)
class TimeSeriesHorizon:
    instruments: dict[Hashable, TimeSeriesInstrument]
    ic_summary: InstrumentICSummary
    non_overlapping_ic_summary: dict[int, InstrumentICSummary]


@dataclass(frozen=True)
class TimeSeriesResult:
    datetimes: pd.DatetimeIndex
    assets: pd.Index
    min_samples: int
    ic_method: ICMethod
    direction: int
    quantiles: QuantileResult
    quantile_reasons: npt.NDArray[np.object_]
    horizons: dict[int, TimeSeriesHorizon]
    intervals_verified: bool = True
