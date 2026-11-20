"""Frozen adaptive HAC formulas, entry points, and floating-point decisions."""

from dataclasses import replace
from math import erfc, inf, nextafter, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
from scipy.stats import t

from backtest_and_analysis.evaluation import EvaluationEngine
from backtest_and_analysis.evaluation.cross_section import (
    bartlett_hac, evaluate_cross_section, hac_p_value, student_t_hac_p_value,
)
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.types import HACInterval
from backtest_and_analysis.time_series.inference import _cross_section_rows


def _panel(*horizons: str | int) -> EvaluationPanel:
    index = pd.MultiIndex.from_product([pd.bdate_range("2024-01-02", periods=24), range(4)])
    factor = np.tile([1, 2, 3, 4], 24)
    returns = np.tile([1, 2, 3, 4, 4, 2, 3, 1], 12)
    return EvaluationPanel.from_long(pd.DataFrame(
        {"factor": factor, **dict.fromkeys(horizons, returns)}, index=index,
    ))


@pytest.mark.parametrize("sign", [1, -1])
def test_adaptive_hac_matches_dense_working_model_with_gaps(sign):
    rng = np.random.default_rng(93620261121)
    values = sign ** np.arange(68) * lfilter([1], [1, -.8], rng.normal(size=68))
    values[np.arange(68) % 17 >= 14] = np.nan
    values[0] = np.nan
    valid = np.isfinite(values)
    slots = np.flatnonzero(valid)
    observed = values[valid]
    n, grid_size = len(observed), len(values)
    residuals = observed - observed.mean()
    adjacent = np.diff(slots) == 1
    prev, curr = residuals[:-1][adjacent], residuals[1:][adjacent]
    phi = np.clip((prev @ curr) / (prev @ prev), -.97, .97)
    transition = phi ** np.diff(slots)
    scales = np.sqrt(1 - transition**2)
    whitening = np.diag(np.r_[1., 1 / scales]) + np.diag(-transition / scales, k=-1)
    weights = np.linalg.solve(whitening.T, np.ones(n))
    whitened = whitening @ residuals

    scores = np.zeros(grid_size)
    scores[valid] = weights * whitened
    pilot = min(grid_size - 1, max(1, 4, int(np.ceil(4 * (grid_size / 100) ** (2 / 9)))))
    gamma = np.correlate(scores, scores, "full")[grid_size - 1:grid_size + pilot] / grid_size
    s0 = gamma[0] + 2 * gamma[1:].sum()
    s1 = 2 * np.arange(1, pilot + 1) @ gamma[1:]
    cap = (grid_size - 1) // 4
    lag = int(np.clip(np.ceil(1.1447 * np.cbrt(grid_size * (s1 / s0)**2)) - 1, 4, cap)
              if s0 > 1e-12 * gamma[0] else cap)
    distance = np.abs(slots[:, None] - slots[None, :])
    h = np.outer(weights, weights) * np.maximum(1 - distance / (lag + 1), 0)
    projection = np.eye(n) - np.outer(whitening @ np.ones(n), weights) / n
    b = projection.T @ h @ projection
    variance = (whitened @ h @ whitened) * (weights @ weights) / (n**2 * np.trace(b))
    df = min(n - 1, np.trace(b)**2 / np.trace(b @ b))
    critical = t.isf(.025, df)

    result = bartlett_hac(values, min_lag=4, calibration_scenario="combined")
    assert result.reason is None
    assert (result.n_valid, result.grid_size, result.lag, result.min_lag) == (n, grid_size, lag, 4)
    assert result.prewhitening_ar1 == pytest.approx(phi)
    assert result.variance == pytest.approx(variance, rel=1e-12)
    assert result.effective_df == pytest.approx(df, rel=1e-12)
    assert result.critical_value == pytest.approx(critical, rel=1e-12)
    assert [result.lower, result.upper] == pytest.approx(observed.mean() + np.array([-1, 1]) * critical * sqrt(variance))
    assert result.pair_counts.tolist() == [np.count_nonzero(distance == k) // 2 for k in range(1, lag + 1)]
    assert (result.method_version, result.bandwidth_method) == ("cs_hac_adaptive_v1", "nw94_prewhitened")
    assert result.missing_pattern == "other"
    assert result.calibration_scenario == "combined"
    assert result.calibration_status == "uncalibrated"
    p_value, method = hac_p_value(result, observed.mean())
    assert method == "two_sided_student_t_hac"
    assert p_value == pytest.approx(2 * t.sf(abs(observed.mean()) / sqrt(variance), df))


def test_zero_lag_adaptive_hac_matches_sample_variance_when_phi_is_zero():
    values = np.array([-1., 0., 1., 0.])
    result = bartlett_hac(values)
    assert result.reason is None
    assert result.lag == result.prewhitening_ar1 == 0
    assert result.effective_df == len(values) - 1
    assert result.variance == pytest.approx(values.var(ddof=1) / len(values))
    assert result.missing_pattern == "none"


@pytest.mark.parametrize("values", [
    [1., np.nan, -.5, np.nan, 1.5, np.nan, -2.],
    [0., 0., 0., np.nan, 1., np.nan, -1.],
])
def test_adaptive_hac_falls_back_to_no_prewhitening_when_ar_fit_is_unidentified(values):
    result = bartlett_hac(values, min_lag=1)
    assert result.reason is None
    assert result.prewhitening_ar1 == 0
    assert result.variance > 0


@pytest.mark.parametrize("gap", [3, 5])
def test_adaptive_hac_records_periodic_mask_and_horizon_floor(gap):
    values = np.random.default_rng(93620261121).normal(size=128)
    values[np.arange(128) % 17 >= 17 - gap] = np.nan
    result = bartlett_hac(values, min_lag=19)
    assert result.reason is None
    assert result.missing_pattern == f"periodic_17_{gap}"
    assert 19 <= result.lag <= (len(values) - 1) // 4
    assert result.variance > 0


def test_adaptive_hac_at_grid_cap_counts_pairs_on_original_slots():
    values = np.random.default_rng(93620261121).normal(size=289)
    values[[0, 2, 13, 288]] = np.nan
    result = bartlett_hac(values, min_lag=72)
    assert result.reason is None and result.lag == 72
    valid = np.isfinite(values)
    assert result.pair_counts.tolist() == [
        np.count_nonzero(valid[k:] & valid[:-k]) for k in range(1, 73)
    ]


@pytest.mark.parametrize(("values", "min_lag", "reason"), [
    ([np.nan, .2, .3], 3, "insufficient_valid_cross_sections"),
    ([.1, .2, .3], 1, "insufficient_hac_grid"),
    ([.3, np.nan, .3, .3], 0, "degenerate_hac_variance"),
    ([-1e308, 1e308, .5], 0, "degenerate_hac_variance"),
])
def test_adaptive_hac_failures_retain_audit_without_a_p_value(values, min_lag, reason):
    result = bartlett_hac(values, min_lag=min_lag)
    assert result.reason == reason
    assert result.min_lag == min_lag
    assert result.method_version == "cs_hac_adaptive_v1"
    assert result.lag == 0 and not result.pair_counts.size
    assert result.prewhitening_ar1 is result.effective_df is result.critical_value is None
    assert np.isnan([result.variance, result.lower, result.upper]).all()
    assert hac_p_value(result, .3) == (None, None)


@pytest.mark.parametrize("min_lag", [True, -1, 1.5])
def test_auto_rejects_invalid_structure_but_fixed_ignores_it(min_lag):
    with pytest.raises(ValueError, match="min_lag"):
        bartlett_hac([.1, .2, .3], min_lag=min_lag)
    assert bartlett_hac([.1, .2, .3], lag=0, min_lag=min_lag).reason is None


@pytest.mark.parametrize("lag", [0, 1, 5])
def test_fixed_hac_retains_original_variance_and_normal_tail(lag):
    values = np.array([.3, .1, np.nan, .2, .6, -.2, .2, .5])
    observed = values[np.isfinite(values)]
    residuals = np.where(np.isfinite(values), values - observed.mean(), 0)
    products = np.correlate(residuals, residuals, "full")[len(values) - 1:]
    variance = (products[0] + 2 * sum(
        (1 - k / (lag + 1)) * products[k] for k in range(1, lag + 1)
    )) / len(observed)**2
    result = bartlett_hac(values, lag=lag, min_lag=9)
    half_width = -NormalDist().inv_cdf(.025) * sqrt(variance)
    assert [result.variance, result.lower, result.upper] == pytest.approx(
        [variance, observed.mean() - half_width, observed.mean() + half_width],
    )
    assert result.bandwidth_method == "fixed" and result.method_version is None
    p_value, method = hac_p_value(result, observed.mean())
    assert p_value == pytest.approx(erfc(abs(observed.mean()) / sqrt(variance) / sqrt(2)))
    assert method == "two_sided_normal_hac"
    assert hac_p_value(result, np.nan) == (None, None)


@pytest.mark.parametrize("df", [4., 12.])
@pytest.mark.parametrize("sign", [-1, 1])
def test_student_t_hac_p_value_agrees_with_bar_v2_at_both_float_endpoints(df, sign):
    panel = _panel(1)
    stats = evaluate_cross_section(panel, hac_lag=0, frequency="bar")
    base = stats.horizons[1]
    alpha, se = .05, 1 / 8
    critical = t.isf(alpha / 2, df)
    edge = critical * se
    for magnitude in (nextafter(edge, 0), edge, nextafter(edge, inf)):
        point = sign * magnitude
        lower, upper = point - edge, point + edge
        interval = HACInterval(
            len(panel.datetimes), len(panel.datetimes), 1, alpha, np.array([23]),
            se**2, lower, upper, None, bandwidth_method="nw94_prewhitened",
            effective_df=df, critical_value=critical, method_version="cs_hac_adaptive_v1",
        )
        p_value = student_t_hac_p_value(point, se**2, df, alpha, lower, upper)
        assert (p_value > alpha, p_value == alpha, p_value < alpha) == (
            magnitude < edge, magnitude == edge, magnitude > edge,
        )
        assert (p_value <= alpha) == (not lower < 0 < upper)
        assert (p_value < alpha) == (not lower <= 0 <= upper)
        assert hac_p_value(interval, point) == (p_value, "two_sided_student_t_hac")
        horizon = replace(base, hac=interval, ic_summary=replace(base.ic_summary, mean=point))
        row, = _cross_section_rows(replace(stats, horizons={1: horizon}), "1min")
        assert (row["p_value"], row["p_value_method"]) == (p_value, "two_sided_student_t_hac")


def test_direct_and_engine_default_hac_resolve_horizons_and_legacy_rounding():
    panel = _panel("1D", "5d")
    engine = EvaluationEngine("cross_sectional")
    direct = evaluate_cross_section(panel)
    via_engine = engine.evaluate(panel).splits["full"].statistics
    assert via_engine.horizons["5d"].hac.variance == direct.horizons["5d"].hac.variance
    assert direct.horizons["5d"].hac.min_lag == 4
    assert direct.horizons["5d"].hac.lag >= 4
    assert direct.horizons["5d"].hac.prewhitening_ar1 == -.97
    pearson = evaluate_cross_section(panel, ic_method="pearson", calibration_scenario="combined")
    assert pearson.horizons["5d"].hac.calibration_scenario == "combined"
    assert pearson.horizons["5d"].hac.calibration_status == "uncalibrated"

    options = dict(hac_lag={"1D": 1, "5d": "auto"}, hac_min_lag={"5d": 5})
    mixed = evaluate_cross_section(panel, **options).horizons
    assert mixed["1D"].hac.lag == 1 and mixed["1D"].hac.bandwidth_method == "fixed"
    assert mixed["5d"].hac.lag == mixed["5d"].hac.min_lag == 5
    assert evaluate_cross_section(panel, _legacy_ic_rounding=True).horizons["5d"].hac.lag == 0
    legacy = engine.evaluate(panel, _legacy_ic_rounding=True).splits["full"].statistics
    assert legacy.horizons["5d"].hac.lag == 4
    with pytest.raises(ValueError, match="auto"):
        engine.evaluate(panel, hac_lag="auto", _legacy_ic_rounding=True)
    for hac_lag in ("auto", options["hac_lag"]):
        with pytest.raises(ValueError, match="auto"):
            evaluate_cross_section(panel, hac_lag=hac_lag, _legacy_ic_rounding=True)

    unknown = _panel("other")
    with pytest.raises(ValueError, match="horizon 'other'"):
        evaluate_cross_section(unknown)
    assert evaluate_cross_section(unknown, hac_min_lag={"other": 2}).horizons["other"].hac.min_lag == 2
    legacy_unknown = engine.evaluate(unknown, _legacy_ic_rounding=True).splits["full"].statistics
    assert legacy_unknown.horizons["other"].hac.lag == 0


def test_development_seed_ar_dependence_widens_adaptive_intervals():
    rng = np.random.default_rng(93620261121)
    iid = rng.normal(size=(12, 256))
    ar = lfilter([.6], [1, -.8], iid, axis=1)
    iid_intervals = [bartlett_hac(row) for row in iid]
    ar_intervals = [bartlett_hac(row) for row in ar]
    assert all(interval.reason is None for interval in iid_intervals + ar_intervals)
    assert np.median([interval.prewhitening_ar1 for interval in ar_intervals]) > .7
    assert np.median([interval.variance for interval in ar_intervals]) > 4 * np.median(
        [interval.variance for interval in iid_intervals],
    )
