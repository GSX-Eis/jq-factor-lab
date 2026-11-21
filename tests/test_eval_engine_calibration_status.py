"""Calibration evidence and interpretation gates across the evaluation adapters."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_and_analysis.alphalens_runner import AlphalensRunner
from backtest_and_analysis.evaluation import (
    REGISTERED_CALIBRATIONS,
    CalibrationMethod,
    CalibrationStatus,
    EvaluationEngine,
    evaluate_interval_usability,
    resolve_interval_calibration,
)
from backtest_and_analysis.evaluation.cross_section import bartlett_hac
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.types import BootstrapInterval, HACInterval
from backtest_and_analysis.forward_returns_calculator import newey_west_mean_test
from backtest_and_analysis.time_series import (
    BarCrossSectionalEvaluator, BarReturnLabeler, evaluate_bars, read_bar_csv,
)
from backtest_and_analysis.time_series.diagnostics_report import (
    DIAGNOSTICS_CSV_COLUMNS, diagnostics_rows, evaluate_diagnostics, render_diagnostics_markdown,
)
from test_time_series_evaluator import labels
from utils.session_calendar import SessionCalendar


@pytest.fixture(params=["hac", "mbb"])
def interval(request: pytest.FixtureRequest) -> HACInterval | BootstrapInterval:
    if request.param == "hac":
        return HACInterval(512, 512, 0, .05, np.array([]), .01, -.1, .1, None)
    return BootstrapInterval(512, 1, 7, 399, 399, 399, .05, -.1, .1, None)


@pytest.mark.parametrize("reason", [None, "interval_unavailable"])
@pytest.mark.parametrize("status", ["uncalibrated", "calibrated_pass", "calibrated_fail"])
def test_interval_calibration_and_estimability_are_independent(
    interval: HACInterval | BootstrapInterval, reason: str | None, status: CalibrationStatus,
) -> None:
    assert interval.calibration_status == "uncalibrated"
    assert interval.calibration_evidence is None
    method = "cross_section_hac" if isinstance(interval, HACInterval) else "time_series_mbb"
    window = interval.lag if isinstance(interval, HACInterval) else interval.block_length
    if status == "uncalibrated":
        evidence = None
    else:
        scenario = "independent_null" if status == "calibrated_pass" else "ar1_null"
        _, evidence = resolve_interval_calibration(
            scenario=scenario, method=method, window=window, sample_size=512, nominal_coverage=.95,
        )
    if isinstance(interval, BootstrapInterval):
        interval = replace(interval, n_valid_pairs=512)
    result = replace(interval, reason=reason, calibration_status=status, calibration_evidence=evidence)
    usability = evaluate_interval_usability(result, point_estimate=0.)
    assert usability.reason == reason
    assert usability.calibration_status == status
    assert usability.calibration_evidence is evidence
    assert usability.descriptive_stats_usable is True
    assert usability.significance_interpretation_usable is (reason is None and status == "calibrated_pass")
    assert not evaluate_interval_usability(result, point_estimate=np.nan).descriptive_stats_usable
    assert not evaluate_interval_usability(
        result, point_estimate=0., point_reason="not_estimable",
    ).descriptive_stats_usable


@pytest.mark.parametrize("invalid_evidence", ["missing", "altered", "wrong_grid", "failed_cell"])
def test_usability_downgrades_inconsistent_calibration(
    interval: HACInterval | BootstrapInterval, invalid_evidence: str,
) -> None:
    method = "cross_section_hac" if isinstance(interval, HACInterval) else "time_series_mbb"
    window = interval.lag if isinstance(interval, HACInterval) else interval.block_length
    scenario = "ar1_null" if invalid_evidence == "failed_cell" else "independent_null"
    _, evidence = resolve_interval_calibration(
        scenario=scenario, method=method, window=window, sample_size=512, nominal_coverage=.95,
    )
    assert evidence is not None
    if invalid_evidence == "missing":
        evidence = None
    elif invalid_evidence == "altered":
        evidence = replace(evidence, observed_coverage=.99)
    elif invalid_evidence == "wrong_grid":
        interval = replace(interval, grid_size=2)

    result = replace(interval, calibration_status="calibrated_pass", calibration_evidence=evidence)
    usability = evaluate_interval_usability(result, point_estimate=0.)
    assert usability.reason is None
    assert usability.calibration_status == "uncalibrated" and usability.calibration_evidence is None
    assert usability.descriptive_stats_usable and not usability.significance_interpretation_usable


def test_hac_with_missing_grid_slots_keeps_interval_without_significance() -> None:
    values = np.full(512, np.nan)
    values[:2] = [-1., 1.]
    interval = bartlett_hac(values, lag=0, calibration_scenario="independent_null")
    assert interval.reason is None
    assert interval.n_valid == 2 and interval.grid_size == 512
    assert interval.lower < 0 < interval.upper
    usability = evaluate_interval_usability(interval, point_estimate=0.)
    assert usability.calibration_status == "calibrated_pass"
    assert usability.calibration_evidence is not None
    assert usability.descriptive_stats_usable and not usability.significance_interpretation_usable


@pytest.mark.parametrize(("scenario", "method", "window", "coverage", "passes"), [
    ("independent_null", "cross_section_hac", 0, .9415, True),
    ("ar1_null", "cross_section_hac", 0, .6525, False),
    ("ar1_null", "cross_section_hac", 16, .9245, False),
    ("overlap_null", "cross_section_hac", 4, .9195, False),
    ("overlap_null", "cross_section_hac", 16, .9375, True),
    ("positive_effect", "cross_section_hac", 0, .9485, True),
    ("negative_effect", "cross_section_hac", 0, .9520, True),
    ("independent_null", "time_series_mbb", 1, .9400, True),
    ("ar1_null", "time_series_mbb", 1, .6200, False),
    ("ar1_null", "time_series_mbb", 17, .9140, False),
    ("overlap_null", "time_series_mbb", 5, .9160, False),
    ("overlap_null", "time_series_mbb", 17, .9320, True),
    ("positive_effect", "time_series_mbb", 1, .9380, True),
    ("negative_effect", "time_series_mbb", 1, .9680, True),
])
def test_registry_matches_all_frozen_coverage_cells(
    scenario: str, method: CalibrationMethod, window: int, coverage: float, passes: bool,
) -> None:
    assert len(REGISTERED_CALIBRATIONS) == 14
    status, evidence = resolve_interval_calibration(
        scenario=scenario, method=method, window=window, sample_size=512, nominal_coverage=.95,
    )
    assert status == ("calibrated_pass" if passes else "calibrated_fail")
    assert evidence is not None and evidence in REGISTERED_CALIBRATIONS
    assert (evidence.scenario, evidence.method, evidence.window, evidence.sample_size) == (
        scenario, method, window, 512,
    )
    assert evidence.nominal_coverage == .95
    assert evidence.bootstrap_replicates == (399 if method == "time_series_mbb" else None)
    assert evidence.observed_coverage == coverage
    repetitions = 2000 if method == "cross_section_hac" else 500
    assert evidence.tolerance == pytest.approx(3 * np.sqrt(.95 * .05 / repetitions))
    assert evidence.passes is passes
    assert evidence.report_ref.startswith("docs/EVALUATION_ENGINE.md#61-")
    with pytest.raises(FrozenInstanceError):
        setattr(evidence, "passes", not passes)


@pytest.mark.parametrize(("scenario", "method", "window", "size", "nominal_coverage"), [
    (None, "cross_section_hac", 16, 512, .95),
    ("unknown", "cross_section_hac", 16, 512, .95),
    ("ar1_null", "cross_section_hac", 17, 512, .95),
    ("ar1_null", "time_series_mbb", 16, 512, .95),
    ("ar1_null", "time_series_mbb", 17, 511, .95),
    ("ar1_null", "time_series_mbb", 17, 512, .90),
])
def test_registry_has_no_fallback_matches(
    scenario: str | None, method: CalibrationMethod, window: int, size: int, nominal_coverage: float,
) -> None:
    assert resolve_interval_calibration(
        scenario=scenario, method=method, window=window, sample_size=size, nominal_coverage=nominal_coverage,
    ) == ("uncalibrated", None)


def _panel(n_assets: int, horizon: int = 5) -> EvaluationPanel:
    times = pd.date_range("2024-01-01", periods=512)
    index = pd.MultiIndex.from_product([times, range(n_assets)], names=["date", "asset"])
    rng = np.random.default_rng(71)
    return EvaluationPanel.from_long(pd.DataFrame({
        "factor": rng.normal(size=len(index)), horizon: rng.normal(size=len(index)),
    }, index=index))


def test_engine_hac_uses_effective_lag_and_spearman_method() -> None:
    panel = _panel(32)
    engine = EvaluationEngine("cross_sectional")
    options = {"calibration_scenario": "overlap_null"}
    adaptive = engine.evaluate(panel, **options).splits["full"].statistics.horizons[5]
    failed = engine.evaluate(panel, hac_lag=0, **options).splits["full"].statistics.horizons[5]
    passed = engine.evaluate(panel, hac_lag=16, **options).splits["full"].statistics.horizons[5]
    pearson = engine.evaluate(
        panel, hac_lag=16, ic_method="pearson", **options,
    ).splits["full"].statistics.horizons[5]
    undeclared = engine.evaluate(panel, hac_lag=16).splits["full"].statistics.horizons[5]
    assert (failed.hac.lag, failed.hac.calibration_status, failed.hac.reason) == (4, "calibrated_fail", None)
    assert adaptive.hac.calibration_status == "uncalibrated" and adaptive.hac.calibration_evidence is None
    assert (passed.hac.lag, passed.hac.calibration_status, passed.hac.reason) == (16, "calibrated_pass", None)
    assert evaluate_interval_usability(passed.hac, point_estimate=passed.ic_summary.mean).significance_interpretation_usable
    assert pearson.hac.calibration_status == undeclared.hac.calibration_status == "uncalibrated"
    assert (passed.hac.lower, passed.hac.upper) == (undeclared.hac.lower, undeclared.hac.upper)


def test_engine_mbb_uses_expected_grid_and_keeps_unavailable_reason() -> None:
    panel = _panel(2)
    engine = EvaluationEngine("time_series")
    grid = np.ones(panel.factor.shape, dtype=bool)
    grid[-1, 1] = False
    entries = np.broadcast_to(panel.datetimes.to_numpy()[:, None], panel.factor.shape)
    verified = engine.evaluate(
        panel, block_length=17, calibration_scenario="overlap_null",
        n_boot=399, min_valid_replicates=399, expected_grid=grid,
        entry_timestamps={5: entries}, exit_timestamps={5: entries + np.timedelta64(5, "D")},
    ).splits["full"].statistics.horizons[5].instruments
    passed, unmatched = verified[0].bootstrap, verified[1].bootstrap
    assert (passed.grid_size, passed.reason, passed.calibration_status) == (512, None, "calibrated_pass")
    assert (unmatched.grid_size, unmatched.reason, unmatched.calibration_status) == (511, None, "uncalibrated")
    missing = engine.evaluate(
        panel, block_length=17, calibration_scenario="overlap_null",
    ).splits["full"].statistics.horizons[5].instruments[0]
    assert (missing.bootstrap.reason, missing.bootstrap.calibration_status) == (
        "holding_intervals_unavailable", "calibrated_pass",
    )
    use = evaluate_interval_usability(missing.bootstrap, point_estimate=missing.spearman_ic.values[0])
    assert use.descriptive_stats_usable and not use.significance_interpretation_usable


@pytest.mark.parametrize(("n_boot", "missing_pair", "significance_usable"), [
    (2, False, False), (399, True, False), (399, False, True),
])
def test_engine_mbb_requires_replicates_and_complete_pairs_from_registered_design(
    n_boot: int, missing_pair: bool, significance_usable: bool,
) -> None:
    panel = _panel(1, horizon=1)
    if missing_pair:
        panel.valid[1][-1, 0] = False
    entries = panel.datetimes.to_numpy()[:, None]
    engine = EvaluationEngine("time_series")
    options = dict(
        block_length=1, n_boot=n_boot, min_valid_replicates=n_boot,
        entry_timestamps={1: entries}, exit_timestamps={1: entries + np.timedelta64(1, "D")},
    )
    tested = engine.evaluate(
        panel, calibration_scenario="independent_null", **options,
    ).splits["full"].statistics.horizons[1].instruments[0]
    reference = engine.evaluate(panel, **options).splits["full"].statistics.horizons[1].instruments[0]
    interval = tested.bootstrap
    assert interval.reason is None and interval.valid_replicates == n_boot
    assert interval.n_valid_pairs == 512 - int(missing_pair)
    assert (interval.lower, interval.upper) == (reference.bootstrap.lower, reference.bootstrap.upper)
    assert tested.spearman_ic.values[0] == reference.spearman_ic.values[0]
    usability = evaluate_interval_usability(interval, point_estimate=tested.spearman_ic.values[0])
    assert usability.calibration_status == "calibrated_pass"
    assert usability.calibration_evidence is not None
    assert usability.descriptive_stats_usable
    assert usability.significance_interpretation_usable is significance_usable


def test_bar_inference_and_published_diagnostics_report_usability() -> None:
    day = "2026-12-01"
    calendar = SessionCalendar("CME Globex Equity", day, day, rth_reference_calendar="NYSE")
    labeler = BarReturnLabeler(calendar, view="RTH", horizons=[1])
    bars = read_bar_csv(Path(__file__).parent / "fixtures/eval_engine/ts_diagnostics_contract/bars.csv")
    result = evaluate_bars(bars, labeler=labeler, analysis_mode="time_series", n_quantiles=2, diagnostics=True)
    inference = result.ic_inference["1min", 1, "NQZ26"]
    assert inference.calibration_status == "uncalibrated"
    assert inference.reason == "holding_intervals_unavailable"
    assert inference.descriptive_stats_usable and not inference.significance_interpretation_usable

    report = evaluate_diagnostics(result)
    horizon, = report["horizons"]
    assert horizon["bootstrap"]["reason"] == "insufficient_bootstrap_replicates"
    assert horizon["interval_usability"] == {
        "reason": "insufficient_bootstrap_replicates",
        "calibration_status": "uncalibrated", "calibration_evidence": None,
        "descriptive_stats_usable": True, "significance_interpretation_usable": False,
    }
    rows = diagnostics_rows(report)
    assert all(set(row) <= set(DIAGNOSTICS_CSV_COLUMNS) for row in rows)
    markdown = render_diagnostics_markdown(report, rows)
    assert "Calibration **uncalibrated**" in markdown
    assert "Descriptive statistics usable **yes**" in markdown
    assert "Significance interpretation usable **no**" in markdown


def test_reversed_full_diagnostics_use_one_bootstrap_and_requested_direction(monkeypatch) -> None:
    from backtest_and_analysis.evaluation import time_series

    day = "2026-12-01"
    calendar = SessionCalendar("CME Globex Equity", day, day, rth_reference_calendar="NYSE")
    labeler = BarReturnLabeler(calendar, view="RTH", horizons=[1])
    bars = read_bar_csv(Path(__file__).parent / "fixtures/eval_engine/ts_diagnostics_contract/bars.csv")
    calls = []
    original = time_series._bootstrap_spearman

    def record(*args, **kwargs):
        interval = original(*args, **kwargs)
        calls.append(interval)
        return interval

    monkeypatch.setattr(time_series, "_bootstrap_spearman", record)
    result = evaluate_bars(
        bars, labeler=labeler, analysis_mode="time_series", n_quantiles=2, direction=-1, diagnostics=True,
        bootstrap={"block_length": 1},
    )
    inference, = evaluate_diagnostics(result)["horizons"]
    assert (inference["direction"]["hits"], inference["direction"]["denominator"]) == (1, 6)
    assert len(calls) == 1
    assert (calls[0].lower, calls[0].upper) == pytest.approx((.18786728732554484, 1.0))


def test_full_diagnostics_reuse_configured_bootstrap_without_split(monkeypatch) -> None:
    day = "2026-12-01"
    calendar = SessionCalendar("CME Globex Equity", day, day, rth_reference_calendar="NYSE")
    labeler = BarReturnLabeler(calendar, view="RTH", horizons=[1])
    bars = read_bar_csv(Path(__file__).parent / "fixtures/eval_engine/ts_diagnostics_contract/bars.csv")
    options = dict(labeler=labeler, analysis_mode="time_series", n_quantiles=2, diagnostics=True)
    result = evaluate_bars(
        bars, bootstrap=dict(block_length=1, seed=73, n_boot=21, min_valid_replicates=15), **options,
    )

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("Rendering diagnostics must reuse statistics")

    monkeypatch.setattr(EvaluationEngine, "evaluate", unexpected_evaluation)
    inference, = evaluate_diagnostics(result)["horizons"]
    assert (inference["bootstrap"]["seed"], inference["bootstrap"]["n_boot"]) == (73, 21)
    assert inference["bootstrap"]["lower"] == pytest.approx(.322322, abs=1e-6)


def test_cross_section_bar_reports_a_point_even_with_degenerate_hac() -> None:
    frame = pd.concat([labels([i, i], instrument=str(i)) for i in range(1, 4)])
    result = BarCrossSectionalEvaluator(analysis_mode="cross_sectional").evaluate(frame)
    inference = result.ic_inference["1min", 1, None]
    assert inference.reason == "degenerate_hac_variance"
    assert inference.calibration_status == "uncalibrated"
    assert inference.descriptive_stats_usable and not inference.significance_interpretation_usable


def test_daily_adapters_keep_legacy_values_and_expose_uncalibrated_hac() -> None:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), list("ABC")], names=["date", "asset"],
    )
    data = pd.DataFrame({
        "factor": [1., 2., 3.] * 2, "1D": [1., 2., 3., 3., 2., 1.],
        "5D": [1., 2., 3., np.nan, np.nan, np.nan], "factor_quantile": [1, 1, 2] * 2,
    }, index=index)
    result = AlphalensRunner(n_quantiles=2)._process_alphalens_results(data)
    assert set(result) == {"ic", "ir", "quantile_returns", "std_err"}
    frame = result["ic"]
    np.testing.assert_allclose(frame["1D"], [1., -1.])
    assert result["ir"].loc["5D", "ir"] == pytest.approx(1.)
    metadata = frame.attrs["hac_inference"]
    assert metadata["1D"]["reason"] is None
    assert metadata["5D"]["reason"] == "insufficient_valid_cross_sections"
    for stats in metadata.values():
        assert stats["calibration_status"] == "uncalibrated"
        assert stats["descriptive_stats_usable"] and not stats["significance_interpretation_usable"]

    t_stat, p_value = newey_west_mean_test([.1, .2, .3], max_lag=0)
    assert t_stat == pytest.approx(4.242640687119286)
    assert p_value == pytest.approx(2.20904969985853e-5)
    assert "uncalibrated" in newey_west_mean_test.__doc__
