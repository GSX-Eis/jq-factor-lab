"""Results of cross-sectional and time-series factor evaluation."""

from collections.abc import Hashable
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .kernels import CorrelationResult, QuantileResult

FloatArray = npt.NDArray[np.float64]
ReturnMode = Literal["raw", "cross_section_excess"]
ICMethod = Literal["spearman", "pearson"]

DEFAULT_BOOTSTRAP_REPLICATES = 999
DEFAULT_MIN_VALID_REPLICATES = 900
P_VALUE_BOOTSTRAP_REPLICATES = 9999

BOOTSTRAP_FIXED = (None, "fixed", "moving_block", "percentile")
BOOTSTRAP_V1 = ("ts_boot_adaptive_v1", "ppw2009", "circular_block", "fisher_z_bootstrap_t")
BOOTSTRAP_V2 = ("ts_boot_adaptive_v2", "ppw2009", "circular_block", "fisher_z_bootstrap_t_flat_top_block")

BOOTSTRAP_V3_P = ("ts_boot_adaptive_v3_p", *BOOTSTRAP_V2[1:])

CalibrationStatus = Literal["uncalibrated", "calibrated_pass", "calibrated_fail"]
CalibrationMethod = Literal[
    "cross_section_hac", "time_series_mbb", "cross_section_hac_adaptive", "time_series_bootstrap_adaptive",
]


@dataclass(frozen=True)
class AdaptiveCalibrationObservation:
    method_version: str
    scenario: str
    grid_size: int
    n_valid: int
    horizon: int
    window: int
    structural_minimum: int
    missing_pattern: str
    alpha: float
    cross_section_width_min: int | None
    cross_section_width_max: int | None
    effective_df: float | None
    hac_variance: float | None
    effective_blocks: float | None
    n_boot: int | None
    min_valid_replicates: int | None
    valid_replicates: int | None
    min_pairs_per_replicate: int | None
    fisher_z_standard_error: float | None = None


@dataclass(frozen=True)
class CalibrationEvidence:
    scenario: str
    method: CalibrationMethod
    window: int
    sample_size: int
    nominal_coverage: float
    observed_coverage: float
    tolerance: float
    passes: bool
    report_ref: str
    bootstrap_replicates: int | None = None
    method_version: str | None = None
    domain_id: str | None = None
    report_sha256: str | None = None


@dataclass(frozen=True)
class IntervalUsability:
    reason: str | None
    calibration_status: CalibrationStatus
    calibration_evidence: CalibrationEvidence | None
    descriptive_stats_usable: bool
    significance_interpretation_usable: bool
    verified_horizon: int | None = None
    adaptive_observation: AdaptiveCalibrationObservation | None = None


def interval_usability_payload(use: IntervalUsability) -> dict:
    payload = asdict(use)
    observation = payload["adaptive_observation"]
    if observation is not None and observation["method_version"] != BOOTSTRAP_V2[0]:
        del observation["fisher_z_standard_error"]
    return payload


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
    calibration_status: CalibrationStatus = "uncalibrated"
    calibration_evidence: CalibrationEvidence | None = None
    method_version: str | None = None
    missing_pattern: str | None = None
    calibration_scenario: str | None = None
    bandwidth_method: Literal["fixed", "nw94_prewhitened"] = "fixed"
    min_lag: int | None = None
    prewhitening_ar1: float | None = None
    effective_df: float | None = None
    critical_value: float | None = None


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
    calibration_status: CalibrationStatus = "uncalibrated"
    calibration_evidence: CalibrationEvidence | None = None
    n_valid_pairs: int | None = None
    method_version: str | None = None
    missing_pattern: str | None = None
    calibration_scenario: str | None = None
    block_length_method: Literal["fixed", "ppw2009"] = "fixed"
    resample_method: Literal["moving_block", "circular_block"] = "moving_block"
    interval_method: Literal["percentile", "fisher_z_bootstrap_t", "fisher_z_bootstrap_t_flat_top_block"] = "percentile"
    min_block_length: int | None = None
    min_pairs_per_replicate: int | None = None
    effective_blocks: float | None = None
    fisher_z_standard_error: float | None = None

    p_value: float | None = None
    p_value_reason: str | None = None

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha


def bootstrap_interval_payload(interval: BootstrapInterval) -> dict:
    payload = asdict(interval)
    if interval.method_version != BOOTSTRAP_V3_P[0]:
        del payload["p_value"]
        del payload["p_value_reason"]
    return payload


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
    p_value: float | None = None
    p_value_reason: str | None = None


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
