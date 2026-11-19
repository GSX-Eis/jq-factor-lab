import math

import numpy as np
import pytest

from backtest_and_analysis.evaluation.calibration import (
    Trial,
    TrialLedger,
    adjust_pvalues,
    calibrate_coverage,
    coverage_passes,
    synthetic_sample,
)


def test_synthetic_overlap_shares_exactly_four_single_step_returns():
    seed = 205
    kwargs = {"n_times": 64, "n_assets": 3}
    x, y = synthetic_sample("overlap_null", np.random.default_rng(seed), **kwargs)
    same_x, same_y = synthetic_sample("overlap_null", np.random.default_rng(seed), **kwargs)
    np.testing.assert_array_equal(same_x, x)
    np.testing.assert_array_equal(same_y, y)

    rng = np.random.default_rng(seed)
    rng.standard_normal((64, 3))  # Factor innovations precede the independent return draws.
    steps = rng.standard_normal((68, 3))
    np.testing.assert_allclose(y[1:] - y[:-1], steps[5:] - steps[:-5], atol=1e-15)


def test_synthetic_ar1_null_has_autocorrelation_and_no_cross_effect():
    x, y = synthetic_sample("ar1_null", np.random.default_rng(71), n_times=4096, n_assets=4)
    for values in (x, y):
        assert np.corrcoef(values[1:].ravel(), values[:-1].ravel())[0, 1] == pytest.approx(0.8, abs=0.03)
    assert abs(np.corrcoef(x.ravel(), y.ravel())[0, 1]) < 0.06


@pytest.mark.parametrize(
    "covered,repetitions,expected",
    [(1870, 2000, False), (1871, 2000, True), (1929, 2000, True), (1930, 2000, False),
     (460, 500, False), (461, 500, True), (489, 500, True), (490, 500, False)],
)
def test_three_mcse_coverage_gate_uses_prespecified_counts(covered, repetitions, expected):
    assert coverage_passes(covered, repetitions) is expected


def test_calibration_is_reproducible_and_checks_both_window_rules():
    kwargs = {
        "seed": 412, "cross_section_repetitions": 9, "time_series_repetitions": 8,
        "n_times": 64, "n_assets": 8, "n_boot": 39,
    }
    report = calibrate_coverage(**kwargs)
    assert report == calibrate_coverage(**kwargs)
    assert report["seed"] == kwargs["seed"]
    assert report["nominal_coverage"] == 0.95
    assert len(report["cells"]) == 14

    for method, minimum_overlap, recommended, repetitions in [
        ("cross_section_hac", 4, 16, 9), ("time_series_mbb", 5, 17, 8),
    ]:
        cells = {(c["scenario"], c["window"]): c for c in report["cells"] if c["method"] == method}
        floor = minimum_overlap - 4
        assert cells.keys() == {
            ("independent_null", floor), ("ar1_null", floor), ("ar1_null", recommended),
            ("overlap_null", minimum_overlap), ("overlap_null", recommended),
            ("positive_effect", floor), ("negative_effect", floor),
        }
        for (scenario, _), cell in cells.items():
            assert cell["M"] == repetitions
            assert not cell["unavailable_intervals"]
            assert cell["coverage"] == cell["covered"] / repetitions
            assert cell["mcse_at_nominal"] == pytest.approx(math.sqrt(0.95 * 0.05 / repetitions))
            assert cell["passes"] == coverage_passes(cell["covered"], repetitions)
            if scenario.endswith("_null"):
                assert cell["true_ic"] == 0
                assert cell["correct_direction_rate"] is None
                assert cell["coverage"] + cell["reject_zero_rate"] == pytest.approx(1)

        positive = cells["positive_effect", floor]
        negative = cells["negative_effect", floor]
        assert positive["true_ic"] == pytest.approx(-negative["true_ic"])
        assert positive["true_ic"] > 0
        for cell in (positive, negative):
            assert cell["point_sign_correct_rate"] >= 0.75
            assert 0 < cell["correct_direction_rate"] <= cell["reject_zero_rate"]

    cs = next(c for c in report["cells"] if c["scenario"] == "positive_effect")
    ts = next(c for c in report["cells"] if c["method"] == "time_series_mbb" and c["scenario"] == "positive_effect")
    assert ts["true_ic"] == pytest.approx(6 * math.asin(0.35 / 2) / math.pi)
    assert 0 < cs["true_ic"] < ts["true_ic"]


@pytest.mark.parametrize(
    "method,expected",
    [("holm", [0.09, 0.005, 0.09, 0.04, 0.09]), ("bh", [0.05, 0.005, 0.05, 0.025, 0.07])],
)
def test_adjust_pvalues_matches_hand_computed_monotonic_adjustments(method, expected):
    np.testing.assert_allclose(adjust_pvalues([0.04, 0.001, 0.03, 0.01, 0.07], method=method), expected)


@pytest.mark.parametrize("values", [[], [[0.02]], [0.02 + 0.03j], [float("nan")], [float("inf")], [1.01], [-0.01]])
def test_adjust_pvalues_rejects_invalid_family(values):
    with pytest.raises(ValueError, match="nonempty vector"):
        adjust_pvalues(values, method="bh")


@pytest.mark.parametrize("method,expected", [("holm", [1, 0.08, 0.08, 1]), ("bh", [1, 0.04, 0.04, 1])])
def test_adjust_pvalues_handles_ties_and_caps(method, expected):
    np.testing.assert_allclose(adjust_pvalues([0.9, 0.02, 0.02, 1], method=method), expected)


def test_ledger_selects_only_after_all_factors_and_horizons_are_recorded():
    family = [("first", 1), ("first", 5), ("second", 1)]
    first, winner, last = family
    ledger = TrialLedger(family)
    winning_trial = Trial(0.18, 0.02, 99, (0.01, 0.35))
    ledger.record(winner, winning_trial)
    with pytest.raises(ValueError, match="Missing trial records"):
        ledger.select_best(method="holm")
    with pytest.raises(ValueError, match="Unregistered candidate"):
        ledger.record(("second", 5), winning_trial)
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record(winner, winning_trial)
    ledger.record(last, Trial(0.10, 0.03, 150))
    ledger.record(first, Trial(0.08, 0.04, 150))

    holm = ledger.select_best(method="holm")
    bh = ledger.select_best(method="bh")
    assert holm.best == bh.best == winner
    assert holm.attempt_count == bh.attempt_count == 3
    assert [row.key for row in holm.trials] == family
    winner_row = holm.trials[1]
    assert winner_row.trial == winning_trial
    assert winner_row.adjusted_p_value == pytest.approx(0.06)
    assert not winner_row.significant
    assert bh.trials[1].adjusted_p_value == pytest.approx(0.04)
    assert bh.trials[1].significant


def test_ledger_rejects_duplicate_manifest_and_blocks_missing_pvalue():
    key = ("factor", 1)
    with pytest.raises(ValueError, match="unique"):
        TrialLedger([key, key])
    ledger = TrialLedger([key])
    interval_only = Trial(0.02, None, 100, (-0.01, 0.05))
    ledger.record(key, interval_only)
    snapshot = ledger.records
    assert snapshot.pop(key) == interval_only
    assert ledger.records == {key: interval_only}
    with pytest.raises(ValueError, match="Missing raw p-value"):
        ledger.select_best(method="bh")


def test_coverage_gate_includes_exact_upper_boundary():
    assert coverage_passes(1178, 1216)
    assert not coverage_passes(1179, 1216)


@pytest.mark.parametrize("method", ["holm", "bh"])
def test_adjust_pvalues_rejects_masked_values(method):
    values = np.ma.array([0.001, 0.03], mask=[True, False])
    with pytest.raises(ValueError, match="masked"):
        adjust_pvalues(values, method=method)


def test_ledger_bh_includes_ties_at_significance_boundary():
    keys = [("first", 1), ("first", 5), ("second", 1)]
    ledger = TrialLedger(keys)
    for key in keys:
        ledger.record(key, Trial(0.18, 0.05, 99))
    result = ledger.select_best(method="bh", alpha=0.05)
    assert [(row.adjusted_p_value, row.significant) for row in result.trials] == [(0.05, True)] * 3


@pytest.mark.parametrize(
    "p_value,adjusted_p,significant",
    [(0.0415, 0.05, True),
     (np.nextafter(0.0415, 1.0), np.nextafter(0.05, 1.0), False)],
)
def test_ledger_bh_significance_at_interior_rank(p_value, adjusted_p, significant):
    keys = [(f"factor_{i}", 1) for i in range(100)]
    ledger = TrialLedger(keys)
    for key, raw_p in zip(keys, [p_value] * 83 + [1.0] * 17):
        ledger.record(key, Trial(0.18, raw_p, 99))

    result = ledger.select_best(method="bh", alpha=0.05)
    assert [(row.adjusted_p_value, row.significant) for row in result.trials] == (
        [(adjusted_p, significant)] * 83 + [(1.0, False)] * 17
    )
