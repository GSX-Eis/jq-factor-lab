"""Verify published interval fields, p values and applicable calibration evidence."""
from __future__ import annotations

from dataclasses import MISSING, asdict, replace
from math import ceil, isclose, sqrt
from pathlib import Path
import re
from statistics import NormalDist
import subprocess
from typing import Any

import numpy as np
from scipy.stats import t

from backtest_and_analysis.evaluation.calibration import (
    REGISTERED_CALIBRATIONS, _CS_HAC_SIZE_REPORT, _registered_adaptive_evidence, bootstrap_pvalue_reason,
    evaluate_interval_usability, resolve_interval_calibration,
)
from backtest_and_analysis.evaluation.cross_section import bartlett_hac, hac_p_value
from backtest_and_analysis.time_series.inference import _bootstrap_observation_matches
from backtest_and_analysis.evaluation.types import (
    BOOTSTRAP_FIXED, BOOTSTRAP_V1, BOOTSTRAP_V2, BOOTSTRAP_V3_P,
    AdaptiveCalibrationObservation, BootstrapInterval, CalibrationEvidence, HACInterval, IntervalUsability,
)

from .normalization import _array, _count, _metric, _nullable_text, _object, _require
from .schema import _CALIBRATION_STATUSES


# Populate at first v2 registration with the reviewed 0-domain source SHA.
_PRE_REGISTRATION_BOOTSTRAP_V2_CODE_VERSIONS: frozenset[str] = frozenset((
    "1f062ecd1a1bcc6518e22c97779e704674487be7",
    "8f84c483abec9a117af50c3dfff355157875b74a",
))


# Reviewed source snapshots from before NERE-1280 entered the registry.
_PRE_REGISTRATION_HAC_BASE = "8ff43272ca14df94c52f2d0a872059a49d43ca05"
_PRE_REGISTRATION_HAC_TIPS = (
    "e77b42e2300a71a361d3d78c9e0539cb5726d9ac",  # Parent of the registration commit.
    "1dea3770556898d52e6028ed355f7a64ff3c91ea",  # EV-1b integration.
)


def _pre_registration_hac_source(code_version: str, source: str | None) -> bool:
    if source not in (None, "git") or re.fullmatch(r"[0-9a-f]{40}", code_version) is None:
        return False
    if code_version == _PRE_REGISTRATION_HAC_BASE or code_version in _PRE_REGISTRATION_HAC_TIPS:
        return True
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return False
    git = ["git", "--no-replace-objects", "-C", str(root)]
    try:
        kind = subprocess.run([*git, "cat-file", "-t", code_version], capture_output=True, timeout=5)
        return kind.returncode == 0 and kind.stdout.strip() == b"commit" and any(
            subprocess.run([*git, "merge-base", "--is-ancestor", code_version, tip],
                           capture_output=True, timeout=5).returncode == 0
            for tip in _PRE_REGISTRATION_HAC_TIPS
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _interval_fields(value: Any, field: str, interval_type: type) -> dict[str, Any]:
    # Earlier publishers omitted fields that now have dataclass defaults.
    fields = interval_type.__dataclass_fields__
    required = [name for name, member in fields.items() if member.default is MISSING]
    payload = _object(value, field, *required)
    return {name: payload.get(name, member.default) for name, member in fields.items()}


def _usability(value: Any, field: str, horizon: int) -> IntervalUsability:
    payload = _object(value, field, *(k for k in IntervalUsability.__dataclass_fields__
                                      if k != "adaptive_observation"))
    status = payload["calibration_status"]
    _require(status in _CALIBRATION_STATUSES, field, "a valid calibration_status")
    _nullable_text(payload["reason"], f"{field}.reason")
    for flag in ("descriptive_stats_usable", "significance_interpretation_usable"):
        _require(type(payload[flag]) is bool, f"{field}.{flag}", "a boolean")
    verified = payload["verified_horizon"]
    if verified is not None:
        _require(_count(verified, field, minimum=1) == horizon, field, "matching verified_horizon")
    evidence = payload["calibration_evidence"]
    observation = payload.get("adaptive_observation")
    if observation is not None:
        observation = _interval_fields(observation, f"{field}.adaptive_observation", AdaptiveCalibrationObservation)
        se = _metric(observation["fisher_z_standard_error"], f"{field}.adaptive_observation.fisher_z_standard_error")
        _require((se is not None and se > 0) if observation["method_version"] == BOOTSTRAP_V2[0] else se is None,
                 field, "positive Fisher-z standard error for v2 observations only")
        for name in ("grid_size", "n_valid", "horizon", "window", "structural_minimum",
                     "cross_section_width_min", "cross_section_width_max", "n_boot",
                     "min_valid_replicates", "valid_replicates", "min_pairs_per_replicate"):
            if observation[name] is not None or name in ("grid_size", "n_valid", "horizon", "window",
                                                         "structural_minimum"):
                _count(observation[name], f"{field}.adaptive_observation.{name}")
        for name in ("effective_df", "hac_variance", "effective_blocks", "alpha"):
            _metric(observation[name], f"{field}.adaptive_observation.{name}")
        for name in ("method_version", "scenario", "missing_pattern"):
            _require(isinstance(observation[name], str), field, f"a string adaptive observation {name}")
        observation = AdaptiveCalibrationObservation(**{
            k: observation[k] for k in AdaptiveCalibrationObservation.__dataclass_fields__
        })
        _require(observation.horizon == horizon, field, "matching adaptive observation horizon")
    if status == "uncalibrated":
        _require(evidence is None and verified is None, field, "no calibration evidence for uncalibrated status")
    else:
        evidence = _interval_fields(evidence, f"{field}.calibration_evidence", CalibrationEvidence)
        _require(evidence["passes"] is (status == "calibrated_pass"), field, "evidence consistent with status")
        if observation is not None and observation.method_version == BOOTSTRAP_V2[0]:
            for name in ("window", "sample_size", "bootstrap_replicates"):
                _count(evidence[name], f"{field}.calibration_evidence.{name}", minimum=1)
        evidence = CalibrationEvidence(**{name: evidence[name] for name in CalibrationEvidence.__dataclass_fields__})
        _require(
            evidence == _registered_adaptive_evidence(observation) if observation is not None
            else evidence in REGISTERED_CALIBRATIONS,
            field, "registered calibration evidence",
        )
    if payload["significance_interpretation_usable"]:
        _require(status == "calibrated_pass" and payload["descriptive_stats_usable"] and payload["reason"] is None,
                 field, "an estimable calibrated interval for significance interpretation")
    return IntervalUsability(
        payload["reason"], status, evidence, payload["descriptive_stats_usable"],
        payload["significance_interpretation_usable"], verified, observation,
    )


def _verified_interval(
    interval: dict[str, Any], use: IntervalUsability, *, time_series: bool, horizon: int,
    statistic: float, point_reason: str | None, n: int, cross_section_sizes: np.ndarray | None,
    section_ics: np.ndarray | None, pair_pattern: str | None, bounds: list[float] | None, p_value: float | None,
    p_method: str | None, alpha: float, config: dict[str, Any], field: str, legacy: bool,
    raw_interval: dict[str, Any], code_version: str, code_version_source: str | None,
) -> None:
    grid_size = interval["grid_size"]
    _require((interval["reason"] is None) == (bounds is not None), field, "interval reason matching bounds")
    method = "time_series_mbb" if time_series else "cross_section_hac"
    interval_type = BootstrapInterval if time_series else HACInterval
    selector = interval["block_length_method" if time_series else "bandwidth_method"]
    adaptive = selector != "fixed"
    scenario = interval["calibration_scenario"] if adaptive else (
        use.calibration_evidence.scenario if use.calibration_evidence is not None else None
    )
    if "calibration_scenario" in config or "analysis_mode" in config:
        _require(not adaptive or scenario == config.get("calibration_scenario"),
                 field, "the adaptive calibration scenario matching the published configuration")
        scenario = config.get("calibration_scenario")
    _nullable_text(scenario, field)

    if time_series:
        tags = (interval["method_version"], selector, interval["resample_method"], interval["interval_method"])
        _require(tags in (BOOTSTRAP_FIXED, BOOTSTRAP_V1, BOOTSTRAP_V2, BOOTSTRAP_V3_P), field, "a recognized Bootstrap method")
        _require(-1 <= statistic <= 1 if point_reason is None else bounds is None, field, "a valid Spearman IC")
        _require(bounds is None or -1 <= bounds[0] <= bounds[1] <= 1, field, "bounds in [-1, 1]")
        options = _object(config.get("bootstrap") or {}, "config.bootstrap")
        request_pvalue = tags == BOOTSTRAP_V3_P
        if "bootstrap" in config or "analysis_mode" in config:
            configured_pvalue = options.get("p_value", False)
            _require(type(configured_pvalue) is bool and configured_pvalue == request_pvalue,
                     field, "bootstrap p-value method matching the published configuration")
        explicit = options.get("block_length", "auto")
        if explicit != "auto":
            explicit = _count(explicit, "config.bootstrap.block_length", minimum=horizon)
        window = _count(interval["block_length"], field)
        _require((not adaptive and window >= horizon) or (adaptive and (
            (window == 0 and bounds is None) or horizon <= window < grid_size
        )), field, "a valid Bootstrap block length")
        if adaptive:
            _require(
                _count(interval["min_block_length"], field) == horizon
                and _count(interval["min_pairs_per_replicate"], field, minimum=4) >= 4,
                field, "Bootstrap structural minimum matching the interval",
            )
            if "min_ic_samples" in config:
                minimum_pairs = max(4, _count(config["min_ic_samples"], "config.min_ic_samples", minimum=2))
                _require(interval["min_pairs_per_replicate"] == minimum_pairs,
                         field, "Bootstrap pair minimum matching the published configuration")
            if tags in (BOOTSTRAP_V2, BOOTSTRAP_V3_P):
                _require(window <= min(n // 4, ceil(min(3 * sqrt(grid_size), grid_size / 3))),
                         field, "v2 Bootstrap block length within the PPW cap")
                pattern = interval["missing_pattern"]
                _require(pattern in ("none", "periodic_17_3", "periodic_17_5", "other"),
                         field, "a v2 Bootstrap missing pattern")
                _require((pattern == "none") == (n == grid_size) and (pair_pattern is None or pattern == pair_pattern),
                         field, "a v2 Bootstrap missing pattern matching the split label pairs")
                unverified_periodic = pair_pattern is None and pattern in ("periodic_17_3", "periodic_17_5")
                _require(not unverified_periodic or use.calibration_status == "uncalibrated",
                         field, "a verified split grid for the v2 Bootstrap missing pattern")
            if use.adaptive_observation is not None:
                _require(
                    _bootstrap_observation_matches(
                        raw_interval, use.adaptive_observation, horizon=horizon, count=n, intervals_verified=True,
                    ), field, "Bootstrap observation matching the interval fields",
                )
        if "block_length" in options or ("analysis_mode" in config and not legacy):
            _require(adaptive if explicit == "auto" else (not adaptive and window == explicit),
                     field, "Bootstrap block length matching the published configuration")
        _count(interval["seed"], field)
        requested = _count(interval["n_boot"], field, minimum=1)
        valid = _count(interval["valid_replicates"], field)
        minimum = _count(interval["min_valid_replicates"], field, minimum=2)
        _require(all(interval[name] == options[name] for name in ("seed", "n_boot", "min_valid_replicates", "alpha")
                     if name in options), field, "Bootstrap parameters matching the published configuration")
        _require(_count(interval["n_valid_pairs"], field) == n and valid <= requested,
                 field, "observed pairs and valid replicate counts")
        _require(bounds is None or (window < grid_size and valid >= minimum),
                 field, "sufficient Bootstrap grid and replicas for finite bounds")
        if adaptive:
            _nullable_text(interval["missing_pattern"], field)
            effective_blocks = _metric(interval["effective_blocks"], field)
            standard_error = _metric(interval["fisher_z_standard_error"], field)
            _require(effective_blocks == (n / window if window else None),
                     field, "effective Bootstrap blocks matching observed pairs and block length")
            _require(bounds is None or (standard_error is not None and standard_error > 0),
                     field, "positive Fisher-z standard error for finite bounds")
        _require(p_method is None and p_value is None, field, "Bootstrap without a ledger p value")
        if request_pvalue:
            diagnostic = _metric(interval["p_value"], f"{field}.p_value")
            diagnostic_reason = bootstrap_pvalue_reason(BootstrapInterval(**interval))
            _require({"p_value", "p_value_reason"} <= raw_interval.keys()
                     and interval["p_value_reason"] == diagnostic_reason,
                     field, "the bootstrap p-value reason matching the design")
            on_lattice = (
                diagnostic is not None and 0 < diagnostic <= 1
                and diagnostic == round(diagnostic * (1 + valid)) / (1 + valid)
            )
            _require(diagnostic is None if diagnostic_reason is not None else on_lattice,
                     field, "a bootstrap p value matching the valid replicate count")
        else:
            _require(interval["p_value"] is None and interval["p_value_reason"] is None,
                     field, "legacy Bootstrap without a diagnostic p value")
    else:
        requested = config.get("hac_lag")
        if requested not in (None, "auto"):
            requested = _count(requested, "config.hac_lag")
        if requested is not None or ("analysis_mode" in config and not legacy):
            _require(adaptive == (requested in (None, "auto")),
                     field, "HAC method matching the published configuration")
        window = _count(interval["lag"], field)
        _require(
            (not adaptive and interval["method_version"] is None and window >= horizon - 1)
            or (selector == "nw94_prewhitened" and interval["method_version"] == "cs_hac_adaptive_v1"
                and _count(interval["min_lag"], field) == horizon - 1
                and (bounds is None or window >= horizon - 1)),
            field, "HAC method and structural minimum matching the interval",
        )
        if isinstance(requested, int):
            _require(window == max(horizon - 1, requested), field, "HAC lag matching the published configuration")
        _require(_count(interval["n_valid"], field) == n, field, "HAC count matching observed sections")
        pair_counts = _array(interval["pair_counts"], f"{field}.pair_counts")
        _require(len(pair_counts) == window and all(_count(x, field) <= n for x in pair_counts),
                 field, "HAC pair counts matching the lag and sample")
        variance = _metric(interval["variance"], f"{field}.variance")
        df = _metric(interval["effective_df"], f"{field}.effective_df")
        if adaptive:
            _nullable_text(interval["missing_pattern"], field)
            _metric(interval["prewhitening_ar1"], field)
            _metric(interval["critical_value"], field)
        _require(bounds is None or (n >= 2 and window < grid_size and variance is not None and variance > 0),
                 field, "estimable HAC variance on the reported grid")
        if bounds is not None:
            _require(0 < alpha / 2 < 0.5, field, "representable two-sided HAC alpha")
            if adaptive:
                _require(df is not None and 1 <= df <= n - 1, field, "valid effective HAC degrees of freedom")
                critical = float(t.isf(alpha / 2, df))
                recorded = interval["critical_value"]
                _require(recorded is not None and isclose(recorded, critical, rel_tol=1e-12),
                         field, "HAC critical value matching effective df")
            else:
                critical = -NormalDist().inv_cdf(alpha / 2)
            half_width = critical * sqrt(variance)
            expected_bounds = (statistic - half_width, statistic + half_width)
            _require(all(isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
                         for value, expected in zip(bounds, expected_bounds, strict=True)),
                     field, "HAC bounds matching mean IC, variance and alpha")
        if adaptive and section_ics is not None:
            recomputed = bartlett_hac(section_ics, lag="auto", min_lag=horizon - 1, alpha=alpha)
            _require(window == recomputed.lag and pair_counts == recomputed.pair_counts.tolist()
                     and interval["reason"] == recomputed.reason
                     and interval["missing_pattern"] == recomputed.missing_pattern,
                     field, "HAC bandwidth and missing pattern matching the label grid")
            for name in ("variance", "effective_df", "prewhitening_ar1"):
                actual, expected = interval[name], getattr(recomputed, name)
                abs_tol = 1e-14 if name == "prewhitening_ar1" else 0.0
                _require(
                    actual is None if expected is None or not np.isfinite(expected)
                    else actual is not None and isclose(actual, expected, rel_tol=1e-12, abs_tol=abs_tol),
                    field, f"HAC {name} matching the label grid",
                )

    status, registered = ("uncalibrated", None) if adaptive else resolve_interval_calibration(
        scenario=scenario, method=method, window=window, sample_size=grid_size, nominal_coverage=1 - alpha,
    )
    payload_evidence = interval["calibration_evidence"]
    if payload_evidence is not None:
        payload_evidence = _interval_fields(payload_evidence, f"{field}.calibration_evidence", CalibrationEvidence)
    _require(interval["calibration_status"] == status
             and payload_evidence == (asdict(registered) if registered is not None else None),
             field, "interval calibration evidence matching its method, settings and declared scenario")
    typed = interval_type(**(interval | {"calibration_evidence": registered}))
    if not time_series:
        expected_p, expected_method = hac_p_value(typed, statistic)
        _require(
            p_method == expected_method and (
                p_value is None if expected_p is None else (
                    p_value is not None and isclose(p_value, expected_p, rel_tol=1e-12)
                    and (p_value < alpha) == (expected_p < alpha)
                    and (p_value > alpha) == (expected_p > alpha)
                )
            ), field, "HAC p value matching mean IC, variance and its method",
        )
        if adaptive:
            cross_section_sizes = (
                np.where(np.isfinite(section_ics), cross_section_sizes, 0) if section_ics is not None else None
            )
    expected = evaluate_interval_usability(
        typed, point_estimate=statistic, point_reason=point_reason, horizon=horizon,
        cross_section_sizes=cross_section_sizes, ic_method="spearman", intervals_verified=time_series,
    )
    evidence = expected.calibration_evidence
    historical_hac = (
        not time_series and evidence is not None and evidence.report_ref == _CS_HAC_SIZE_REPORT
        and _pre_registration_hac_source(code_version, code_version_source)
    )
    if historical_hac or (
        time_series and tags == BOOTSTRAP_V2 and expected.calibration_status == "calibrated_pass"
        and use.calibration_status == "uncalibrated"
        and re.fullmatch(r"[0-9a-f]{40}", code_version)
        and code_version in _PRE_REGISTRATION_BOOTSTRAP_V2_CODE_VERSIONS
    ):
        expected = replace(
            expected, calibration_status="uncalibrated", calibration_evidence=None,
            verified_horizon=None, significance_interpretation_usable=False,
        )
    _require(use == expected, field, "interval usability matching the verified interval and observed design")
