"""Synthetic interval coverage and selection across a registered family of trials."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from math import asin, ceil, isclose, isfinite, pi, sqrt
from numbers import Real
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.stats import t as student_t

from .kernels import _positive_int, spearman
from .types import (
    BOOTSTRAP_V1,
    BOOTSTRAP_V2,
    AdaptiveCalibrationObservation,
    BootstrapInterval,
    CalibrationEvidence,
    CalibrationMethod,
    CalibrationStatus,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_MIN_VALID_REPLICATES,
    P_VALUE_BOOTSTRAP_REPLICATES,
    FloatArray,
    HACInterval,
    ICMethod,
    IntervalUsability,
)

_ALPHA = 0.05
_NOMINAL_COVERAGE = 1 - _ALPHA
_TOLERANCE_MCSE = 3
_REGISTERED_BOOTSTRAP_REPLICATES = 399
_REGISTERED_CROSS_SECTION_ASSETS = 32


@dataclass(frozen=True)
class _Scenario:
    horizon: int = 1
    factor_ar: float = 0.0
    return_ar: float = 0.0
    pearson_rho: float = 0.0


_SCENARIOS = {
    "independent_null": _Scenario(),
    "ar1_null": _Scenario(factor_ar=0.8, return_ar=0.8),
    "overlap_null": _Scenario(horizon=5, factor_ar=0.8),
    "positive_effect": _Scenario(pearson_rho=0.35),
    "negative_effect": _Scenario(pearson_rho=-0.35),
}


def bootstrap_pvalue_reason(interval: BootstrapInterval) -> str | None:
    """Check the preregistered diagnostic design and finite-replicate budget."""
    if interval.reason == "insufficient_bootstrap_replicates":
        return "insufficient_replicates_for_pvalue"
    if interval.reason is not None:
        return "bootstrap_interval_unestimable"
    if (interval.n_boot < P_VALUE_BOOTSTRAP_REPLICATES
            or 10 * interval.valid_replicates < 9 * interval.n_boot):
        return "insufficient_replicates_for_pvalue"
    if (
        not isinstance(interval.calibration_scenario, str) or interval.calibration_scenario not in _SCENARIOS
        or interval.grid_size not in (512, 1024)
        or interval.min_block_length not in (1, 5, 20) or interval.missing_pattern != "none"
        or interval.min_pairs_per_replicate != 4 or interval.alpha != _ALPHA
        or not _finite_real(interval.effective_blocks) or interval.effective_blocks < 4
    ):
        return "outside_pvalue_calibration_domain"
    return None


def _gaussian_series(
    rng: np.random.Generator, n_times: int, n_assets: int, ar: float,
) -> FloatArray:
    values = rng.standard_normal((n_times, n_assets))
    if ar:
        values[1:] *= sqrt(1 - ar**2)
        for i in range(1, n_times):
            values[i] += ar * values[i - 1]
    return values


def synthetic_sample(
    scenario: str, rng: np.random.Generator, *, n_times: int, n_assets: int,
) -> tuple[FloatArray, FloatArray]:
    """Generate paired Gaussian factors and forward returns on a dense signal grid."""
    if scenario not in _SCENARIOS:
        raise ValueError(f"Unknown calibration scenario {scenario!r}")
    n_times = _positive_int(n_times, "n_times")
    n_assets = _positive_int(n_assets, "n_assets")
    spec = _SCENARIOS[scenario]
    factor = _gaussian_series(rng, n_times, n_assets, spec.factor_ar)
    returns = _gaussian_series(rng, n_times + spec.horizon - 1, n_assets, spec.return_ar)
    if spec.horizon > 1:
        returns = np.lib.stride_tricks.sliding_window_view(returns, spec.horizon, axis=0).sum(axis=-1)
    if spec.pearson_rho:
        rho = spec.pearson_rho
        returns = rho * factor + sqrt(1 - rho**2) * returns
    return factor, returns


def _coverage_mcse(repetitions: int) -> float:
    return sqrt(_NOMINAL_COVERAGE * _ALPHA / repetitions)


def coverage_passes(covered: int, repetitions: int) -> bool:
    """Apply the predeclared per-cell tolerance to the count of covering intervals."""
    repetitions = _positive_int(repetitions, "repetitions")
    covered = _positive_int(covered, "covered", minimum=0)
    if covered > repetitions:
        raise ValueError("covered must be <= repetitions")
    # Square the frozen 95% ± 3 MCSE inequality to keep both boundaries exact.
    return (20 * covered - 19 * repetitions)**2 <= 171 * repetitions


def _registered_evidence(
    method: CalibrationMethod, scenario: str, window: int, covered: int,
) -> CalibrationEvidence:
    repetitions = 2000 if method == "cross_section_hac" else 500
    return CalibrationEvidence(
        scenario=scenario, method=method, window=window, sample_size=512,
        nominal_coverage=_NOMINAL_COVERAGE, observed_coverage=covered / repetitions,
        tolerance=_TOLERANCE_MCSE * _coverage_mcse(repetitions),
        passes=coverage_passes(covered, repetitions),
        report_ref="docs/EVALUATION_ENGINE.md#61-一次性合成校准",
        bootstrap_replicates=_REGISTERED_BOOTSTRAP_REPLICATES if method == "time_series_mbb" else None,
    )


# Frozen coverage counts from §6.1, seed 791823. All simulated intervals were estimable.
REGISTERED_CALIBRATIONS = (
    _registered_evidence("cross_section_hac", "independent_null", 0, 1883),
    _registered_evidence("cross_section_hac", "ar1_null", 0, 1305),
    _registered_evidence("cross_section_hac", "ar1_null", 16, 1849),
    _registered_evidence("cross_section_hac", "overlap_null", 4, 1839),
    _registered_evidence("cross_section_hac", "overlap_null", 16, 1875),
    _registered_evidence("cross_section_hac", "positive_effect", 0, 1897),
    _registered_evidence("cross_section_hac", "negative_effect", 0, 1904),
    _registered_evidence("time_series_mbb", "independent_null", 1, 470),
    _registered_evidence("time_series_mbb", "ar1_null", 1, 310),
    _registered_evidence("time_series_mbb", "ar1_null", 17, 457),
    _registered_evidence("time_series_mbb", "overlap_null", 5, 458),
    _registered_evidence("time_series_mbb", "overlap_null", 17, 466),
    _registered_evidence("time_series_mbb", "positive_effect", 1, 469),
    _registered_evidence("time_series_mbb", "negative_effect", 1, 484),
)
_CALIBRATION_BY_KEY = {
    (row.scenario, row.method, row.window, row.sample_size, row.nominal_coverage): row
    for row in REGISTERED_CALIBRATIONS
}


def resolve_interval_calibration(
    *, scenario: str | None, method: CalibrationMethod, window: int, sample_size: int,
    nominal_coverage: float,
) -> tuple[CalibrationStatus, CalibrationEvidence | None]:
    """Look up the explicitly declared scenario and the exact interval settings."""
    evidence = _CALIBRATION_BY_KEY.get((scenario, method, window, sample_size, nominal_coverage))
    if evidence is None:
        return "uncalibrated", None
    return ("calibrated_pass" if evidence.passes else "calibrated_fail"), evidence


@dataclass(frozen=True)
class AdaptiveCalibrationDomain:
    domain_id: str
    method_version: str
    scenario: str
    grid_size_min: int
    grid_size_max: int
    min_n_valid: int
    max_missing_fraction: float
    missing_patterns: tuple[str, ...]
    horizons: tuple[int, ...]
    cross_section_width_min: int | None
    cross_section_width_max: int | None
    min_effective_df: float | None
    min_effective_blocks: float | None
    min_n_boot: int | None
    min_valid_replicates: int | None
    min_valid_fraction: float | None
    min_pairs_per_replicate: int | None
    nominal_coverage: float
    report_ref: str
    report_sha256: str
    coverage_cell_ids: tuple[str, ...]
    covered: int
    repetitions: int
    grid_sizes: tuple[int, ...] | None = None
    alpha: float = _ALPHA


_PATTERN_GAPS = {"none": 0, "periodic_17_3": 3, "periodic_17_5": 5}


def _expected_mask(size: int, pattern: str) -> npt.NDArray[np.bool_]:
    return np.arange(size) % 17 < 17 - _PATTERN_GAPS[pattern]


def _finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and isfinite(value)


def _bootstrap_observation_shape(observation: AdaptiveCalibrationObservation) -> bool:
    o = observation
    counts = (
        o.grid_size, o.n_valid, o.horizon, o.window, o.structural_minimum, o.n_boot,
        o.min_valid_replicates, o.valid_replicates, o.min_pairs_per_replicate,
    )
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in counts):
        return False
    se = o.fisher_z_standard_error
    return (
        isinstance(o.scenario, str) and bool(o.scenario.strip())
        and isinstance(o.missing_pattern, str) and o.missing_pattern in _PATTERN_GAPS
        and 4 <= o.n_valid <= o.grid_size and 1 <= o.horizon == o.structural_minimum <= o.window
        and 4 <= o.min_pairs_per_replicate <= o.n_valid
        and 2 <= o.min_valid_replicates <= o.valid_replicates <= o.n_boot
        and _finite_real(o.alpha) and 0 < o.alpha < 1 and o.alpha / 2 > 0
        and _finite_real(o.effective_blocks) and o.effective_blocks > 0
        and o.cross_section_width_min is None and o.cross_section_width_max is None
        and o.effective_df is None and o.hac_variance is None
        and (_finite_real(se) and se > 0 if o.method_version == BOOTSTRAP_V2[0] else se is None)
    )


def _adaptive_calibration_observation(
    interval: HACInterval | BootstrapInterval, *, point_estimate: float, horizon: int,
    ic_method: ICMethod | None, point_reason: str | None = None,
    cross_section_sizes: npt.ArrayLike | None = None, intervals_verified: bool = False,
) -> AdaptiveCalibrationObservation | None:
    if (
        ic_method != "spearman" or not isinstance(interval.calibration_scenario, str)
        or not interval.calibration_scenario or not isinstance(interval.missing_pattern, str)
        or interval.missing_pattern not in _PATTERN_GAPS or point_reason is not None
        or interval.reason is not None or not isfinite(point_estimate)
        or not isfinite(interval.lower) or not isfinite(interval.upper) or interval.lower > interval.upper
        or isinstance(interval.grid_size, (bool, np.bool_))
        or not isinstance(interval.grid_size, (int, np.integer)) or interval.grid_size <= 0
    ):
        return None
    common = dict(
        scenario=interval.calibration_scenario, grid_size=int(interval.grid_size),
        horizon=horizon, missing_pattern=interval.missing_pattern, alpha=interval.alpha,
    )
    if isinstance(interval, HACInterval):
        if (
            interval.method_version != "cs_hac_adaptive_v1" or interval.bandwidth_method != "nw94_prewhitened"
            or interval.min_lag is None or interval.effective_df is None
            or cross_section_sizes is None or np.ma.is_masked(cross_section_sizes)
        ):
            return None
        sizes = np.asarray(cross_section_sizes)
        if sizes.shape != (interval.grid_size,) or sizes.dtype.kind not in "iu" or np.any(sizes < 0):
            return None
        valid = sizes > 0
        if not valid.any() or not np.array_equal(valid, _expected_mask(interval.grid_size, interval.missing_pattern)):
            return None
        return AdaptiveCalibrationObservation(
            method_version=interval.method_version, n_valid=interval.n_valid, window=interval.lag,
            structural_minimum=interval.min_lag,
            cross_section_width_min=int(sizes[valid].min()), cross_section_width_max=int(sizes[valid].max()),
            effective_df=interval.effective_df, hac_variance=interval.variance, effective_blocks=None,
            n_boot=None, min_valid_replicates=None, valid_replicates=None, min_pairs_per_replicate=None,
            **common,
        )
    tags = (interval.method_version, interval.block_length_method, interval.resample_method, interval.interval_method)
    se = interval.fisher_z_standard_error
    if (
        intervals_verified is not True or tags not in (BOOTSTRAP_V1, BOOTSTRAP_V2)
        or not all(_finite_real(v) for v in (point_estimate, interval.lower, interval.upper, se))
        or not -1 <= point_estimate <= 1 or not -1 <= interval.lower <= interval.upper <= 1 or se <= 0
    ):
        return None
    observation = AdaptiveCalibrationObservation(
        method_version=interval.method_version, n_valid=interval.n_valid_pairs, window=interval.block_length,
        structural_minimum=interval.min_block_length, cross_section_width_min=None, cross_section_width_max=None,
        effective_df=None, hac_variance=None, effective_blocks=interval.effective_blocks,
        n_boot=interval.n_boot, min_valid_replicates=interval.min_valid_replicates,
        valid_replicates=interval.valid_replicates, min_pairs_per_replicate=interval.min_pairs_per_replicate,
        fisher_z_standard_error=se if tags == BOOTSTRAP_V2 else None,
        **common,
    )
    return observation if _bootstrap_observation_shape(observation) else None


def _adaptive_domain_mismatches(
    domain: AdaptiveCalibrationDomain, observation: AdaptiveCalibrationObservation,
) -> tuple[str, ...]:
    o = observation
    integers = (o.grid_size, o.n_valid, o.horizon, o.window, o.structural_minimum)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in integers):
        return ("invalid_observation_counts",)
    size, n = int(o.grid_size), int(o.n_valid)
    if not 0 < n <= size:
        return ("invalid_observation_counts",)
    if domain.method_version == BOOTSTRAP_V2[0] and not _bootstrap_observation_shape(o):
        return ("invalid_bootstrap_observation",)
    if not isinstance(o.missing_pattern, str):
        return ("missing_pattern",)
    gaps = _PATTERN_GAPS.get(o.missing_pattern)
    periods, remainder = divmod(size, 17)
    expected_n = None if gaps is None else periods * (17 - gaps) + min(remainder, 17 - gaps)
    checks = {
        "method_version": o.method_version == domain.method_version,
        "scenario": o.scenario == domain.scenario,
        "grid_size": (
            size in domain.grid_sizes if domain.grid_sizes is not None
            else domain.grid_size_min <= size <= domain.grid_size_max
        ),
        "n_valid": n == expected_n and n >= domain.min_n_valid,
        "missing_fraction": 1 - n / size <= domain.max_missing_fraction,
        "missing_pattern": o.missing_pattern in domain.missing_patterns,
        "horizon": o.horizon in domain.horizons,
        "nominal_coverage": o.alpha == domain.alpha and 1 - o.alpha == domain.nominal_coverage,
    }
    if domain.min_effective_df is not None:
        checks.update(
            structural_minimum=o.structural_minimum == o.horizon - 1,
            window=o.horizon - 1 <= o.window <= (size - 1) // 4,
            cross_section_width=(
                isinstance(o.cross_section_width_min, (int, np.integer))
                and isinstance(o.cross_section_width_max, (int, np.integer))
                and domain.cross_section_width_min
                <= o.cross_section_width_min == o.cross_section_width_max <= domain.cross_section_width_max
            ),
            effective_df=isinstance(o.effective_df, Real) and domain.min_effective_df <= o.effective_df <= n - 1,
            hac_variance=isinstance(o.hac_variance, Real) and isfinite(o.hac_variance) and o.hac_variance > 0,
        )
    else:
        boot_counts = (o.n_boot, o.min_valid_replicates, o.valid_replicates, o.min_pairs_per_replicate)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in boot_counts):
            return ("invalid_bootstrap_counts",)
        requested, minimum, valid, pairs = map(int, boot_counts)
        fraction = Fraction(str(domain.min_valid_fraction))
        checks.update(
            structural_minimum=o.structural_minimum == o.horizon,
            window=o.horizon <= o.window <= min(n // 4, ceil(min(3 * sqrt(size), size / 3))),
            effective_blocks=(
                o.window > 0 and isinstance(o.effective_blocks, Real)
                and o.effective_blocks >= domain.min_effective_blocks
                and isclose(o.effective_blocks, n / o.window, rel_tol=1e-12, abs_tol=0)
            ),
            n_boot=requested >= domain.min_n_boot,
            min_valid_replicates=minimum >= domain.min_valid_replicates,
            valid_replicates=minimum <= valid <= requested,
            valid_fraction=requested > 0 and valid * fraction.denominator >= requested * fraction.numerator,
            min_pairs_per_replicate=pairs == domain.min_pairs_per_replicate,
        )
    return tuple(key for key, passed in checks.items() if not passed)


def _candidate_adaptive_domain(
    *, cross_section: bool, scenario: str, horizon: int, gap: int, s5: bool = False, v2_boot: bool = False,
) -> AdaptiveCalibrationDomain:
    generalized = scenario == "generalized_ar_heteroskedastic"
    mode = "none" if gap == 0 else f"periodic_17_{gap}"
    core_valid = int(_expected_mask(512, mode).sum())
    if cross_section:
        version = "cs_hac_adaptive_v1"
    else:
        version = BOOTSTRAP_V2[0] if v2_boot else BOOTSTRAP_V1[0]
    grid_min, valid_min, width_min = (128, 90, 16)
    if s5 or v2_boot:
        grid_min, valid_min, width_min = (256, 180, 32) if cross_section else (512, 360, 32)
    return AdaptiveCalibrationDomain(
        domain_id=f"{version}:{scenario}", method_version=version, scenario=scenario,
        grid_size_min=grid_min if generalized else 512, grid_size_max=1024 if generalized else 512,
        min_n_valid=valid_min if generalized else core_valid,
        max_missing_fraction=.30 if generalized else 1 - core_valid / 512,
        missing_patterns=tuple(_PATTERN_GAPS) if generalized else (mode,),
        horizons=(1, 5, 20) if generalized else (horizon,),
        cross_section_width_min=(width_min if generalized else 32) if cross_section else None,
        cross_section_width_max=(64 if generalized else 32) if cross_section else None,
        min_effective_df=4 if cross_section else None, min_effective_blocks=None if cross_section else 4,
        min_n_boot=None if cross_section else DEFAULT_BOOTSTRAP_REPLICATES,
        min_valid_replicates=None if cross_section else DEFAULT_MIN_VALID_REPLICATES,
        min_valid_fraction=None if cross_section else .9,
        min_pairs_per_replicate=None if cross_section else 4,
        nominal_coverage=_NOMINAL_COVERAGE, report_ref="", report_sha256="",
        coverage_cell_ids=(), covered=0, repetitions=0,
        grid_sizes=(512, 1024) if v2_boot and generalized else None,
    )


_CS_HAC_SIZE_REPORT = "backtest_and_analysis/evaluation/calibration_data/cs_hac_adaptive_v1_size.json"
_CS_HAC_SIZE_ALPHAS = (.01, .05)

# Retained NERE-1280 null cells at T=1024. Counts are ordered by _CS_HAC_SIZE_ALPHAS.
_CS_HAC_SIZE_COUNTS = (
    ("independent_null", 256, 1, (217, 1042)),
    ("independent_null", 256, 20, (225, 1036)),
    ("independent_null", 1024, 1, (180, 842)),
    ("independent_null", 1024, 20, (178, 864)),
    ("ar1_null", 256, 1, (223, 1083)),
    ("ar1_null", 256, 20, (208, 1005)),
    ("ar1_null", 1024, 1, (180, 856)),
    ("ar1_null", 1024, 20, (161, 803)),
    ("overlap_null", 256, 1, (202, 972)),
    ("overlap_null", 256, 20, (224, 1029)),
    ("overlap_null", 1024, 1, (182, 829)),
    ("overlap_null", 1024, 20, (148, 826)),
)
_CS_HAC_SIZE_REPETITIONS = {256: 20000, 1024: 16000}


# S5 released cross-section HAC. Bootstrap v2 was separately released.
REGISTERED_ADAPTIVE_CALIBRATIONS: tuple[AdaptiveCalibrationDomain, ...] = tuple(
    replace(
        _candidate_adaptive_domain(cross_section=True, scenario=scenario, horizon=horizon, gap=gap, s5=True),
        report_ref="backtest_and_analysis/evaluation/calibration_data/cs_hac_adaptive_v1_coverage.json",
        report_sha256="d0828711fcee509b781cd240057b9f0ab36fc539b01acc3d2c6be004e4122f88",
        coverage_cell_ids=cell_ids,
        covered=covered,
        repetitions=500,
    )
    for scenario, horizon, gap, cell_ids, covered in (
        ("iid", 1, 0, ("0:000",), 465),
        ("strong_ar", 1, 0, ("0:001",), 479),
        ("overlap", 5, 0, ("0:002",), 478),
        ("gaps", 1, 3, ("0:003",), 472),
        ("heteroskedastic", 1, 0, ("0:004",), 470),
        ("combined", 5, 3, ("0:005",), 478),
        ("positive_effect", 1, 0, ("0:006",), 468),
        ("negative_effect", 1, 0, ("0:007",), 480),
        ("generalized_ar_heteroskedastic", 1, 0, tuple(f"2:{index:03d}" for index in range(54)), 461),
    )
) + tuple(
    replace(
        _candidate_adaptive_domain(cross_section=False, scenario=scenario, horizon=horizon, gap=gap, v2_boot=True),
        report_ref="backtest_and_analysis/evaluation/calibration_data/ts_bootstrap_v2_coverage.json",
        report_sha256="9d63d9272b172b1aa3efa8368f0207df2f0b42b95f38f226dd6c5355b50c50b7",
        coverage_cell_ids=cell_ids,
        covered=covered,
        repetitions=500,
    )
    for scenario, horizon, gap, cell_ids, covered in (
        ("iid", 1, 0, ("1:000",), 477),
        ("strong_ar", 1, 0, ("1:001",), 468),
        ("overlap", 5, 0, ("1:002",), 467),
        ("gaps", 1, 3, ("1:003",), 477),
        ("heteroskedastic", 1, 0, ("1:004",), 479),
        ("combined", 5, 3, ("1:005",), 471),
        ("positive_effect", 1, 0, ("1:006",), 477),
        ("negative_effect", 1, 0, ("1:007",), 471),
        ("generalized_ar_heteroskedastic", 1, 0, tuple(f"3:{index:03d}" for index in range(18)), 462),
    )
) + tuple(
    replace(
        _candidate_adaptive_domain(cross_section=True, scenario=scenario, horizon=horizon, gap=0),
        domain_id=f"cs_hac_adaptive_v1:{scenario}:nere1280:{width}:1024:{horizon}:{alpha}",
        grid_size_min=1024,
        grid_size_max=1024,
        min_n_valid=1024,
        cross_section_width_min=width,
        cross_section_width_max=width,
        min_effective_df=1,
        nominal_coverage=1 - alpha,
        report_ref=_CS_HAC_SIZE_REPORT,
        report_sha256="d6529bfe5cb92aa737b6c1e75c42f95a6458072564863d535ec679c5054f8234",
        coverage_cell_ids=(f"{scenario}:{width}:1024:{horizon}:{alpha}",),
        covered=_CS_HAC_SIZE_REPETITIONS[width] - rejected,
        repetitions=_CS_HAC_SIZE_REPETITIONS[width],
        alpha=alpha,
    )
    for scenario, width, horizon, counts in _CS_HAC_SIZE_COUNTS
    for alpha, rejected in zip(_CS_HAC_SIZE_ALPHAS, counts, strict=True)
)


def _registered_adaptive_evidence(
    observation: AdaptiveCalibrationObservation | None,
) -> CalibrationEvidence | None:
    if observation is None:
        return None
    for domain in REGISTERED_ADAPTIVE_CALIBRATIONS:
        if _adaptive_domain_mismatches(domain, observation):
            continue
        cross_section = domain.method_version == "cs_hac_adaptive_v1"
        method: CalibrationMethod = "cross_section_hac_adaptive" if cross_section else "time_series_bootstrap_adaptive"
        return CalibrationEvidence(
            scenario=observation.scenario, method=method, window=observation.window,
            sample_size=observation.grid_size, bootstrap_replicates=observation.n_boot,
            nominal_coverage=domain.nominal_coverage, observed_coverage=domain.covered / domain.repetitions,
            tolerance=(.5 * domain.alpha if domain.report_ref == _CS_HAC_SIZE_REPORT
                       else _TOLERANCE_MCSE * _coverage_mcse(domain.repetitions)),
            passes=True, report_ref=domain.report_ref, method_version=domain.method_version,
            domain_id=domain.domain_id, report_sha256=domain.report_sha256,
        )
    return None


def evaluate_interval_usability(
    interval: HACInterval | BootstrapInterval, *, point_estimate: float, point_reason: str | None = None,
    horizon: int | None = None, cross_section_sizes: npt.ArrayLike | None = None,
    ic_method: ICMethod | None = None, intervals_verified: bool = False,
) -> IntervalUsability:
    """Assess usability and bind a horizon when the full registered design was checked."""
    if horizon is not None:
        horizon = _positive_int(horizon, "horizon")
    point_usable = point_reason is None and isfinite(point_estimate)
    adaptive = interval.method_version is not None or (
        interval.bandwidth_method != "fixed" if isinstance(interval, HACInterval) else (
            interval.block_length_method != "fixed" or interval.resample_method != "moving_block"
            or interval.interval_method != "percentile"
        )
    )
    if adaptive:
        observation = None if horizon is None else _adaptive_calibration_observation(
            interval, point_estimate=point_estimate, point_reason=point_reason, horizon=horizon,
            ic_method=ic_method, cross_section_sizes=cross_section_sizes, intervals_verified=intervals_verified,
        )
        evidence = _registered_adaptive_evidence(observation)
        return IntervalUsability(
            reason=interval.reason,
            calibration_status="uncalibrated" if evidence is None else "calibrated_pass",
            calibration_evidence=evidence,
            descriptive_stats_usable=point_usable,
            significance_interpretation_usable=evidence is not None,
            verified_horizon=horizon if evidence is not None else None,
            adaptive_observation=observation,
        )
    design_supplied = horizon is not None
    verified_horizon = None
    status: CalibrationStatus = "uncalibrated"
    evidence = None
    applicable = False
    candidate = interval.calibration_evidence
    if isinstance(candidate, CalibrationEvidence):
        method: CalibrationMethod
        if isinstance(interval, HACInterval):
            method, window = "cross_section_hac", interval.lag
            design_supplied = design_supplied and cross_section_sizes is not None
            applicable = interval.n_valid == interval.grid_size and (
                cross_section_sizes is None
                or (
                    np.shape(cross_section_sizes) == (interval.grid_size,)
                    and not np.ma.is_masked(cross_section_sizes)
                    and bool(np.all(np.asarray(cross_section_sizes) == _REGISTERED_CROSS_SECTION_ASSETS))
                )
            )
        else:
            method, window = "time_series_mbb", interval.block_length
            applicable = (
                interval.n_valid_pairs == interval.grid_size
                and interval.n_boot == interval.valid_replicates == candidate.bootstrap_replicates
            )
        registered_status, registered_evidence = resolve_interval_calibration(
            scenario=candidate.scenario, method=method, window=window,
            sample_size=interval.grid_size, nominal_coverage=1 - interval.alpha,
        )
        if candidate == registered_evidence and interval.calibration_status == registered_status:
            status, evidence = registered_status, registered_evidence
            applicable = applicable and (horizon is None or horizon == _SCENARIOS[candidate.scenario].horizon)
            if applicable and design_supplied:
                verified_horizon = horizon

    return IntervalUsability(
        reason=interval.reason,
        calibration_status=status,
        calibration_evidence=evidence,
        descriptive_stats_usable=point_usable,
        significance_interpretation_usable=(
            point_usable and interval.reason is None and status == "calibrated_pass" and applicable
        ),
        verified_horizon=verified_horizon,
    )


@dataclass
class _CoverageCounts:
    covered: int = 0
    point_sign_correct: int = 0
    correct_direction: int = 0
    rejects_zero: int = 0

    def add(self, point: float, lower: float, upper: float, truth: float) -> None:
        self.covered += lower <= truth <= upper
        self.point_sign_correct += (truth > 0 and point > 0) or (truth < 0 and point < 0)
        self.correct_direction += (truth > 0 and lower > 0) or (truth < 0 and upper < 0)
        self.rejects_zero += lower > 0 or upper < 0


def calibrate_coverage(
    *, seed: int = 791823, cross_section_repetitions: int = 2000,
    time_series_repetitions: int = 500, n_times: int = 512, n_assets: int = _REGISTERED_CROSS_SECTION_ASSETS,
    n_boot: int = _REGISTERED_BOOTSTRAP_REPLICATES,
) -> dict:
    """Run the fixed synthetic scenarios through the S3-A and S3-B interval estimators."""
    from .cross_section import bartlett_hac
    from .time_series import _bootstrap_spearman

    seed = _positive_int(seed, "seed", minimum=0)
    n_times = _positive_int(n_times, "n_times", minimum=18)
    n_assets = _positive_int(n_assets, "n_assets", minimum=3)
    n_boot = _positive_int(n_boot, "n_boot", minimum=2)
    repetitions = {
        "cross_section_hac": _positive_int(cross_section_repetitions, "cross_section_repetitions"),
        "time_series_mbb": _positive_int(time_series_repetitions, "time_series_repetitions"),
    }
    cells = []
    seed_streams = iter(np.random.SeedSequence(seed).spawn(len(repetitions) * len(_SCENARIOS)))
    for method, m in repetitions.items():
        cross_section = method == "cross_section_hac"
        mcse = _coverage_mcse(m)
        for scenario, spec in _SCENARIOS.items():
            child = next(seed_streams)
            rng = np.random.default_rng(child)
            minimum = spec.horizon - int(cross_section)
            dependent = bool(spec.factor_ar or spec.return_ar or spec.horizon > 1)
            recommended = max(minimum, 16 if cross_section else 17) if dependent else minimum
            windows = tuple(dict.fromkeys((minimum, recommended)))
            rho = spec.pearson_rho
            half_angle = asin(rho / 2)
            if cross_section:
                # The HAC target is the expected finite-cross-section sample Spearman.
                truth = 6 * (asin(rho) + (n_assets - 2) * half_angle) / (pi * (n_assets + 1))
            else:
                truth = 6 * half_angle / pi

            counts = {window: _CoverageCounts() for window in windows}
            unavailable = {window: Counter() for window in windows}
            valid = np.ones(n_times, dtype=bool)
            for _ in range(m):
                x, y = synthetic_sample(
                    scenario, rng, n_times=n_times, n_assets=n_assets if cross_section else 1,
                )
                ic = spearman(x, y, axis=1 if cross_section else 0, min_samples=3)
                point = float(ic.values.mean())
                draw_seed = 0 if cross_section else int(rng.integers(2**63))
                for window in windows:
                    if cross_section:
                        interval = bartlett_hac(ic.values, lag=window, alpha=_ALPHA)
                    else:
                        interval = _bootstrap_spearman(
                            x[:, 0], y[:, 0], valid, point=ic, min_samples=3,
                            block_length=window, seed=draw_seed, n_boot=n_boot,
                            min_valid_replicates=n_boot, alpha=_ALPHA,
                        )
                    if interval.reason is not None:
                        unavailable[window][interval.reason] += 1
                    counts[window].add(point, interval.lower, interval.upper, truth)

            for window, count in counts.items():
                within_tolerance = coverage_passes(count.covered, m)
                cells.append({
                    "method": method,
                    "scenario": scenario,
                    **asdict(spec),
                    "seed_spawn_key": list(child.spawn_key),
                    "M": m,
                    "window": window,
                    "minimum_window": minimum,
                    "recommended_window": recommended,
                    "true_ic": truth,
                    "covered": count.covered,
                    "unavailable_intervals": dict(unavailable[window]),
                    "coverage": count.covered / m,
                    "mcse_at_nominal": mcse,
                    "tolerance": _TOLERANCE_MCSE * mcse,
                    "within_tolerance": within_tolerance,
                    "passes": within_tolerance and not unavailable[window],
                    "point_sign_correct_rate": count.point_sign_correct / m if truth else None,
                    "correct_direction_rate": count.correct_direction / m if truth else None,
                    "reject_zero_rate": count.rejects_zero / m,
                })
    return {
        "seed": seed,
        "n_times": n_times,
        "n_cross_section_assets": n_assets,
        "n_boot": n_boot,
        "nominal_coverage": _NOMINAL_COVERAGE,
        "tolerance_mcse_multiple": _TOLERANCE_MCSE,
        "cells": cells,
    }


Correction = Literal["holm", "bh"]
TrialKey = tuple[str, int]


def adjust_pvalues(p_values: npt.ArrayLike, *, method: Correction) -> FloatArray:
    """Return adjusted p-values in input order for Holm FWER or BH FDR control."""
    if method not in ("holm", "bh"):
        raise ValueError("method must be 'holm' or 'bh'")
    if np.ma.is_masked(p_values):
        raise ValueError("p_values must not contain masked values")
    values = np.asarray(p_values)
    if (values.ndim != 1 or not values.size or values.dtype.kind not in "iuf"
            or not np.all((0 <= values) & (values <= 1))):
        raise ValueError("p_values must be a nonempty vector of finite values in [0, 1]")
    values = values.astype(float, copy=False)
    order = np.argsort(values, kind="stable")
    sorted_p = values[order]
    if method == "holm":
        adjusted = np.maximum.accumulate(sorted_p * np.arange(values.size, 0, -1))
    else:
        # Scale the input's exact integer ratio so only the final division rounds.
        scaled = np.empty_like(sorted_p)
        for index, p_value in enumerate(sorted_p):
            numerator, denominator = p_value.as_integer_ratio()
            scaled[index] = (numerator * values.size) / (denominator * (index + 1))
        adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(adjusted, 1)
    return result


def _hac_test_at_alpha(
    statistic: float, observation: AdaptiveCalibrationObservation, alpha: float,
) -> tuple[float, float, float]:
    from .cross_section import student_t_hac_p_value

    df, variance = observation.effective_df, observation.hac_variance
    assert df is not None and variance is not None
    half_width = float(student_t.isf(alpha / 2, df)) * sqrt(variance)
    lower, upper = statistic - half_width, statistic + half_width
    return lower, upper, student_t_hac_p_value(statistic, variance, df, alpha, lower, upper)


@dataclass(frozen=True)
class Trial:
    """One attempt; the legacy n_effective field counts observed pairs or sections."""

    statistic: float | None
    p_value: float | None
    n_effective: int
    interval: tuple[float, float] | None = None
    interval_usability: IntervalUsability | None = None
    p_value_method: str | None = None

    @property
    def n_observations(self) -> int:
        return self.n_effective

    def inference_eligible(self, *, horizon: int, alpha: float) -> bool:
        use = self.interval_usability
        if (
            self.p_value is None or self.interval is None or use is None or use.reason is not None
            or not use.descriptive_stats_usable or not use.significance_interpretation_usable
            or use.calibration_status != "calibrated_pass" or use.verified_horizon != horizon
            or not isinstance(use.calibration_evidence, CalibrationEvidence) or not use.calibration_evidence.passes
        ):
            return False
        evidence = use.calibration_evidence
        observation = use.adaptive_observation
        if observation is not None or self.p_value_method == "two_sided_student_t_hac":
            levels = _CS_HAC_SIZE_ALPHAS if evidence.report_ref == _CS_HAC_SIZE_REPORT else (_ALPHA,)
            if (
                not isinstance(observation, AdaptiveCalibrationObservation)
                or observation.horizon != horizon or observation.n_valid != self.n_observations
                or alpha not in levels or self.p_value_method != "two_sided_student_t_hac"
                or evidence.method != "cross_section_hac_adaptive"
                or evidence != _registered_adaptive_evidence(observation)
                or self.statistic is None
            ):
                return False
            interval_alpha = observation.alpha
            lower, upper, expected = _hac_test_at_alpha(self.statistic, observation, interval_alpha)
            actual = self.p_value
            return (
                isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)
                and (actual < interval_alpha) == (expected < interval_alpha)
                and (actual > interval_alpha) == (expected > interval_alpha)
                and all((actual <= level) == (expected <= level) for level in levels)
                and isclose(self.interval[0], lower, rel_tol=1e-12, abs_tol=0.0)
                and isclose(self.interval[1], upper, rel_tol=1e-12, abs_tol=0.0)
            )
        return (
            evidence.sample_size == self.n_observations
            and 1 - alpha >= evidence.nominal_coverage and evidence in REGISTERED_CALIBRATIONS
            and horizon == _SCENARIOS[evidence.scenario].horizon
        )

    def _p_value_for_adjustment(self) -> float:
        assert self.p_value is not None and self.interval_usability is not None
        evidence = self.interval_usability.calibration_evidence
        assert evidence is not None
        if evidence.report_ref == _CS_HAC_SIZE_REPORT:
            observation = self.interval_usability.adaptive_observation
            assert observation is not None and self.statistic is not None
            # The registered size study used the 95% interval to evaluate all thresholds.
            _, _, reference_p = _hac_test_at_alpha(self.statistic, observation, _ALPHA)
            return next((level for level in _CS_HAC_SIZE_ALPHAS if reference_p <= level), 1.)
        return self.p_value

    def __post_init__(self) -> None:
        _positive_int(self.n_effective, "n_effective", minimum=0)
        if self.interval_usability is not None and not isinstance(self.interval_usability, IntervalUsability):
            raise TypeError("interval_usability must be an IntervalUsability")
        if self.statistic is not None and not np.isfinite(self.statistic):
            raise ValueError("statistic must be finite or None")
        if self.p_value is not None and not 0 <= self.p_value <= 1:
            raise ValueError("p_value must be in [0, 1] or None")
        if self.interval is not None:
            if (not isinstance(self.interval, tuple) or len(self.interval) != 2
                    or not np.isfinite(self.interval).all() or self.interval[0] > self.interval[1]):
                raise ValueError("interval must be a pair of ordered finite bounds")
        if self.p_value is not None or self.interval is not None:
            if self.statistic is None or not self.n_effective:
                raise ValueError("inference requires a statistic and at least one observation")


class AdjustedTrial(NamedTuple):
    """Audit adjustment and gated significance decision."""

    key: TrialKey
    trial: Trial
    adjusted_p_value: float
    significant: bool


@dataclass(frozen=True)
class TrialSelection:
    best: TrialKey | None
    method: Correction
    alpha: float
    trials: tuple[AdjustedTrial, ...]
    inference_adjusted_p_values: dict[TrialKey, float] = field(default_factory=dict)

    @property
    def attempt_count(self) -> int:
        return len(self.trials)


class TrialLedger:
    """Register the complete candidate family before recording its evaluated trials."""

    def __init__(self, candidates: Iterable[TrialKey]) -> None:
        self._candidates = tuple(candidates)
        if not self._candidates:
            raise ValueError("Register at least one candidate")
        for key in self._candidates:
            if (not isinstance(key, tuple) or len(key) != 2
                    or not isinstance(key[0], str) or not key[0].strip()):
                raise ValueError("Each candidate must be a (factor, horizon) pair")
            _positive_int(key[1], "horizon")
        self._expected = frozenset(self._candidates)
        if len(self._expected) != len(self._candidates):
            raise ValueError("Candidate keys must be unique")
        self._trials: dict[TrialKey, Trial] = {}

    def record(self, key: TrialKey, trial: Trial) -> None:
        if key not in self._expected:
            raise ValueError(f"Unregistered candidate {key!r}")
        if key in self._trials:
            raise ValueError(f"Trial already recorded for {key!r}")
        if not isinstance(trial, Trial):
            raise TypeError("trial must be a Trial")
        self._trials[key] = trial

    @property
    def records(self) -> dict[TrialKey, Trial]:
        """Snapshot recorded attempts, including trials with unavailable inference."""
        return self._trials.copy()

    def select_best(self, *, method: Correction, alpha: float = 0.05) -> TrialSelection:
        """Return audit adjustments and select among eligible, significant candidates."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        missing = self._expected - self._trials.keys()
        if missing:
            raise ValueError(f"Missing trial records for {sorted(missing)!r}")

        p_values = []
        for key in self._candidates:
            p_value = self._trials[key].p_value
            if p_value is None:
                raise ValueError(f"Missing raw p-value for {key!r}")
            p_values.append(p_value)
        adjusted = adjust_pvalues(p_values, method=method)
        eligible = [
            self._trials[key].inference_eligible(horizon=key[1], alpha=alpha) for key in self._candidates
        ]
        # Keep every registered hypothesis in the denominator, without borrowing evidence
        # from the raw p-values of ineligible candidates.
        inference = adjust_pvalues([
            self._trials[key]._p_value_for_adjustment() if usable else 1.
            for key, usable in zip(self._candidates, eligible, strict=True)
        ], method=method)
        trials = tuple(
            AdjustedTrial(key, self._trials[key], float(p), bool(usable and q <= alpha))
            for key, p, q, usable in zip(self._candidates, adjusted, inference, eligible, strict=True)
        )
        winners = [(p, index) for index, (p, row) in enumerate(zip(p_values, trials, strict=True))
                   if row.significant]
        winner = self._candidates[min(winners)[1]] if winners else None
        inference_adjusted = {
            key: float(q) for key, q, usable in zip(self._candidates, inference, eligible, strict=True) if usable
        }
        return TrialSelection(winner, method, alpha, trials, inference_adjusted)
