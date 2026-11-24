"""Frozen adaptive bootstrap formula, sampling and public entry point checks."""

from dataclasses import asdict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
from scipy.stats import rankdata

from backtest_and_analysis.evaluation import EvaluationEngine, evaluate_interval_usability, kernels, time_series
from backtest_and_analysis.evaluation.calibration import Trial, TrialLedger
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.time_series import BarReturnLabeler, evaluate_bars, read_bar_csv
from utils.session_calendar import SessionCalendar


def _bootstrap(x, y, valid=None, *, block_length="auto", **options):
    mask = np.ones(len(x), dtype=bool) if valid is None else valid
    point = kernels.spearman(x[:, None], y[:, None], valid=mask[:, None], axis=0, min_samples=3)
    settings = dict(min_samples=3, seed=53, n_boot=49, min_valid_replicates=35, alpha=0.10)
    return time_series._bootstrap_spearman(
        x, y, mask, point=point, block_length=block_length, **(settings | options),
    )


def test_default_bootstrap_is_adaptive_with_shared_replicate_counts():
    x, y = np.random.default_rng(64).normal(size=(2, 120))
    dates = pd.date_range("2026-01-05", periods=len(x), freq="D")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "asset"])
    panel = EvaluationPanel.from_long(pd.DataFrame({"factor": x, "5D": y}, index=index))
    entries = dates.to_numpy()[:, None]
    exits = entries + np.timedelta64(5, "D")
    engine = EvaluationEngine("time_series", "daily")
    full = engine.evaluate(panel, entry_timestamps={5: entries}, exit_timestamps={5: exits})
    interval = full.splits["full"].statistics.horizons[5].instruments["A"].bootstrap
    assert interval.reason is None
    assert interval.method_version == "ts_boot_adaptive_v2"
    assert 5 == interval.min_block_length <= interval.block_length
    assert interval.n_boot == interval.valid_replicates == 999
    assert interval.min_valid_replicates == 900
    assert interval.min_pairs_per_replicate == 4

    direct = time_series.evaluate_time_series(
        EvaluationPanel.from_long(panel.to_long().rename(columns={"5D": 5})), {5: (entries, exits)},
    ).horizons[5].instruments["A"].bootstrap
    assert direct == interval

    fixed = engine.evaluate(
        panel, entry_timestamps={5: entries}, exit_timestamps={5: exits},
        block_length=5, n_boot=49, min_valid_replicates=35,
    ).splits["full"].statistics.horizons[5].instruments["A"].bootstrap
    assert (fixed.block_length, fixed.block_length_method, fixed.resample_method, fixed.interval_method) == (
        5, "fixed", "moving_block", "percentile",
    )
    assert fixed.method_version is fixed.min_block_length is fixed.effective_blocks is None

    unavailable = engine.evaluate(panel).splits["full"].statistics.horizons[5].instruments["A"].bootstrap
    assert unavailable.reason == "holding_intervals_unavailable"
    assert unavailable.method_version == interval.method_version
    assert unavailable.min_block_length == 5
    assert unavailable.block_length == unavailable.n_boot == 0


def test_ppw_block_length_responds_to_dependence_and_respects_bounds():
    iid = np.random.default_rng(53).standard_normal((2, 512))
    ar = lfilter([1], [1, -0.92], iid, axis=-1)
    independent = _bootstrap(*iid)
    dependent = _bootstrap(*ar)
    assert 1 <= independent.block_length < dependent.block_length <= 68
    assert dependent.effective_blocks == 512 / dependent.block_length
    floor = _bootstrap(*iid, min_block_length=32)
    assert floor.block_length == floor.min_block_length == 32
    cap = _bootstrap(*ar, valid=np.arange(512) < 16)
    assert cap.block_length == 4
    impossible = _bootstrap(*iid, min_block_length=69)
    assert impossible.reason == "insufficient_bootstrap_grid"
    assert impossible.block_length == 0 and impossible.effective_blocks is None


def test_circular_blocks_use_uniform_starts_and_keep_missing_slots():
    size, length = 11, 4
    rng = np.random.Generator(np.random.PCG64(72))
    reference = np.random.Generator(np.random.PCG64(72))
    mask = np.arange(size) % 3 != 1
    counts = np.zeros((size, size), dtype=int)
    for _ in range(2200):
        starts = reference.integers(0, size, size=3, dtype=np.int64)
        expected = [(start + offset) % size for start in starts for offset in range(length)][:size]
        rows = time_series._circular_block_indices(rng, size, np.arange(length))
        np.testing.assert_array_equal(mask[rows], mask[expected])
        np.testing.assert_array_equal(rows, expected)
        counts[np.arange(size), rows] += 1
    assert np.all(np.abs(counts - 200) < 55)
    assert np.all(np.abs(counts[:, [0, size - 1]] - counts[:, [5]]) < 70)


def _reference_fisher(x, y, valid, length, *, historical_v1, residual_ratio=None):
    """Compute pairwise H matrices and enumerate circular windows directly."""
    x, y = x[valid], y[valid]
    n, size = len(x), len(valid)
    if n < 4 or min(len(set(x)), len(set(y))) < 2:
        return np.nan, np.nan, np.nan
    fx, fy = (rankdata(x) - 0.5) / n, (rankdata(y) - 0.5) / n
    sx, sy = np.sqrt(np.mean((fx - 0.5)**2)), np.sqrt(np.mean((fy - 0.5)**2))
    u, v = (fx - 0.5) / sx, (fy - 0.5) / sy
    rho = np.clip(np.mean(u * v), -1, 1)
    hx = (x[:, None] > x) + 0.5 * (x[:, None] == x)
    hy = (y[:, None] > y) + 0.5 * (y[:, None] == y)
    psi = u * v - rho * (u**2 + v**2) / 2
    psi += np.mean((v - rho * u)[:, None] * (hx - fx[:, None]), axis=0) / sx
    psi += np.mean((u - rho * v)[:, None] * (hy - fy[:, None]), axis=0) / sy
    clipped = np.clip(rho, -1 + 1e-6, 1 - 1e-6)
    scores = np.zeros(size)
    scores[valid] = (psi - psi.mean()) / (1 - clipped**2)
    q, r = divmod(size, length)
    def sums(width):
        return np.array([np.roll(scores, -j)[:width].sum() for j in range(size)])

    v0 = np.sum(scores**2) / n**2
    if v0 <= 1e-24 / n:
        return np.nan, np.nan, np.nan
    full = np.mean(sums(length)**2)
    if historical_v1:
        var = (q * full + np.mean(sums(r)**2)) / n**2
    elif residual_ratio is None:
        residual_ratio = np.mean(sums(r)**2) / full if r else 0
        k = length // 2
        spectrum = (np.mean(sums(2 * k)**2) - np.mean(sums(k)**2)) / k / (1 - 3 * k / size) if k else 0
        if spectrum <= 1e-12 * np.mean(scores**2):
            spectrum = full / length / (1 - length / size)
        var = size * spectrum / n**2
    else:
        blocks = np.array([scores[j * length:(j + 1) * length].sum() for j in range(q)])
        var = (q + residual_ratio) * np.sum((blocks - blocks.mean())**2) / (n**2 * (q - 1))
    return np.arctanh(clipped), np.sqrt(var) if var > 1e-12 * v0 else np.nan, residual_ratio


@pytest.mark.parametrize("historical_v1", [False, True], ids=["v2", "historical-v1"])
def test_fisher_z_studentization_matches_reference_and_reflects_under_sign_change(historical_v1):
    x, noise = np.random.default_rng(123).normal(size=(2, 67))
    y = np.round(0.5 * x + noise, 1)
    x = np.round(x, 1)
    valid = np.arange(len(x)) % 17 < 14
    # A masked observation and a non-finite observation yield the same grid gap.
    missing_y = y.copy()
    missing_y[~valid] = np.inf
    interval = _bootstrap(x, missing_y, min_block_length=5, calibration_scenario="ar1_null", historical_v1=historical_v1)
    assert interval == _bootstrap(x, y, valid, min_block_length=5, calibration_scenario="ar1_null", historical_v1=historical_v1)
    assert interval.reason is None and interval.calibration_status == "uncalibrated"
    assert interval.calibration_evidence is None and interval.calibration_scenario == "ar1_null"
    assert interval.missing_pattern == "periodic_17_3"
    assert interval.n_valid_pairs == valid.sum()
    assert interval.interval_method == (
        "fisher_z_bootstrap_t" if historical_v1 else "fisher_z_bootstrap_t_flat_top_block"
    )
    assert interval.resample_method == "circular_block"
    assert -1 <= interval.lower < interval.upper <= 1

    size, length = len(x), interval.block_length
    rng = np.random.Generator(np.random.PCG64(interval.seed))
    f, se, ratio = _reference_fisher(x, y, valid, length, historical_v1=historical_v1)
    assert size % length != 0
    valid_pair_counts = set()
    pivots = []
    for _ in range(interval.n_boot):
        starts = rng.integers(0, size, size=(size + length - 1) // length, dtype=np.int64)
        rows = [(start + j) % size for start in starts for j in range(length)][:size]
        valid_pair_counts.add(valid[rows].sum())
        f_star, se_star, _ = _reference_fisher(
            x[rows], y[rows], valid[rows], length, historical_v1=historical_v1, residual_ratio=ratio,
        )
        pivots.append((f_star - f) / se_star)
    assert len(valid_pair_counts) > 1
    finite = np.asarray(pivots)[np.isfinite(pivots)]
    upper_q, lower_q = np.quantile(finite, [0.95, 0.05])
    assert interval.valid_replicates == len(finite)
    assert interval.fisher_z_standard_error == pytest.approx(se, abs=1e-14)
    assert (interval.lower, interval.upper) == pytest.approx(
        np.tanh(f - np.array([upper_q, lower_q]) * se), abs=1e-13,
    )
    mirrored = _bootstrap(x, -y, valid, min_block_length=5, historical_v1=historical_v1)
    assert mirrored.block_length == length
    assert (mirrored.lower, mirrored.upper) == pytest.approx((-interval.upper, -interval.lower), abs=1e-13)
    changed_seed = _bootstrap(x, y, valid, min_block_length=5, seed=54, historical_v1=historical_v1)
    assert changed_seed.block_length == length
    assert (changed_seed.lower, changed_seed.upper) != (interval.lower, interval.upper)


@pytest.mark.parametrize(("x", "y", "minimum", "reason", "block"), [
    ([0, 1], [1, 0], 3, "insufficient_samples", 0),
    ([0, 1, 2], [1, 0, 2], 3, "insufficient_samples", 0),
    ([0, 0, 0, 0], [1, 0, 2, 3], 3, "constant_factor", 0),
    ([0, 1, 2, 3], [1, 1, 1, 1], 3, "constant_return", 0),
    ([0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0], 3, "degenerate_bootstrap_standard_error", 1),
    ([0, 1, 2, 3], [0, 3, 2, 1], 5, "insufficient_samples", 0),
])
def test_adaptive_bootstrap_reason_codes(x, y, minimum, reason, block):
    interval = _bootstrap(np.array(x), np.array(y), min_samples=minimum)
    assert interval.method_version == "ts_boot_adaptive_v2"
    assert interval.missing_pattern == "none"
    assert interval.min_pairs_per_replicate == max(4, minimum)
    assert interval.reason == reason
    assert interval.block_length == block
    assert interval.valid_replicates == 0 and interval.fisher_z_standard_error is None
    assert np.isnan(interval.lower) and np.isnan(interval.upper)


def test_adaptive_bootstrap_keeps_audit_when_too_few_replicates_are_valid():
    x, y = np.random.default_rng(53).normal(size=(2, 40))
    valid = np.arange(40) % 17 < 12
    interval = _bootstrap(x, y, valid, min_samples=valid.sum(), min_valid_replicates=49)
    assert interval.missing_pattern == "periodic_17_5"
    assert interval.reason == "insufficient_bootstrap_replicates"
    assert 0 < interval.valid_replicates < 49
    assert interval.fisher_z_standard_error > 0
    valid[0] = False
    assert _bootstrap(x, y, valid).missing_pattern == "other"


@pytest.mark.parametrize(("options", "error"), [
    ({"min_block_length": True}, "min_block_length"),
    ({"min_block_length": 0}, "min_block_length"),
    ({"alpha": np.nextafter(0.0, 1.0)}, "alpha"),
])
def test_adaptive_bootstrap_rejects_invalid_options(options, error):
    x, y = np.random.default_rng(53).normal(size=(2, 40))
    with pytest.raises(ValueError, match=error):
        _bootstrap(x, y, **options)


@pytest.mark.parametrize("block_options", [{}, {"block_length": "auto"}])
def test_bar_bootstrap_config_forwards_auto_and_tuning_options(block_options):
    day = "2026-12-01"
    calendar = SessionCalendar("CME Globex Equity", day, day, rth_reference_calendar="NYSE")
    labeler = BarReturnLabeler(calendar, view="RTH", horizons=[1])
    bars = read_bar_csv(Path(__file__).parent / "fixtures/eval_engine/ts_diagnostics_contract/bars.csv")
    result = evaluate_bars(
        bars, labeler=labeler, analysis_mode="time_series", n_quantiles=2,
        bootstrap=dict(seed=72, n_boot=39, min_valid_replicates=20, alpha=0.1, **block_options),
    )
    row, = result.splits["full"]["horizons"]
    interval = row["bootstrap"]
    assert interval["block_length_method"] == "ppw2009"
    assert interval["min_block_length"] == 1
    assert (interval["seed"], interval["n_boot"], interval["min_valid_replicates"], interval["alpha"]) == (
        72, 39, 20, 0.1,
    )


def test_flat_top_hand_calculation_and_bartlett_fallback():
    # Integer scores make the circular window energies directly enumerable.
    scores = np.array([1., 1., 0., 0., -1., -1., 0., 0., 0., 0., 0., 0.])
    # T A_1=4, T A_2=12, T A_3=18, b=3 and r=0.
    v0 = 4 / 12**2
    variance, ratio = time_series._flat_top_variance(scores, 12, 3, 0, v0)
    assert variance == pytest.approx((12 - 4) / (1 - 3 / 12) / 12**2)
    assert ratio == 0
    # A_2=0 with alternating scores triggers the residual guard if r>0.
    alternating = np.tile([1., -1.], 4)
    assert np.isnan(time_series._flat_top_variance(alternating, 8, 2, 1, 1 / 8)[0])
    assert time_series._flat_top_variance(alternating, 8, 2, 0, 1 / 8)[0] == 0
    # Negative flat-top spectrum and a strictly positive Bartlett spectrum.
    scores = np.tile([2., -1., 1., -2.], 2)
    variance, ratio = time_series._flat_top_variance(scores, 8, 2, 0, 20 / 64)
    # T A_2=4 and T A_1=20.
    assert variance == pytest.approx((4 / 2) / (1 - 2 / 8) / 8**2)
    assert ratio == 0
    variance, _ = time_series._flat_top_variance(scores, 8, 1, 0, 20 / 64)
    assert variance == pytest.approx(20 / (8 * 7))


def test_fixed_blocks_keep_the_original_seed_draws_and_percentiles():
    x, y = np.random.default_rng(166).normal(size=(2, 37))
    n, length = len(x), 5
    interval = _bootstrap(x, y, block_length=length)
    rng = np.random.default_rng(interval.seed)
    estimates = []
    for _ in range(interval.n_boot):
        starts = rng.integers(n - length + 1, size=8)
        rows = [start + offset for start in starts for offset in range(length)][:n]
        estimates.append(kernels.spearman(x[rows, None], y[rows, None], axis=0, min_samples=3).values.item())
    np.testing.assert_array_equal([interval.lower, interval.upper], np.quantile(estimates, [.05, .95]))
    assert interval.method_version is None and interval.fisher_z_standard_error is None


def test_bootstrap_pvalue_matches_seeded_draws_with_independent_observed_statistic(monkeypatch):
    x, noise = np.random.default_rng(91).normal(size=(2, 512))
    y = noise + .13 * x
    valid = np.ones(512, dtype=bool)
    generated = time_series._adaptive_pivots
    pivots = []

    def capture(*args, **kwargs):
        pivots.extend(generated(*args, **kwargs))
        return iter(pivots)

    monkeypatch.setattr(time_series, "_adaptive_pivots", capture)
    actual = _bootstrap(
        x, y, p_value=True, n_boot=9999, min_valid_replicates=9000,
        alpha=.05, calibration_scenario="independent_null",
    )
    z, se, _ = _reference_fisher(x, y, valid, actual.block_length, historical_v1=False)
    observed = abs(z / se)
    exceedances = sum(not -observed < value < observed for value in pivots)
    assert 0 < exceedances < len(pivots)
    assert actual.valid_replicates == len(pivots)
    assert actual.p_value == (1 + exceedances) / (1 + len(pivots))


def test_bootstrap_pvalue_matches_independent_tail_count_and_keeps_interval(monkeypatch):
    x, y = np.random.default_rng(64).normal(size=(2, 512))
    valid = np.ones(512, dtype=bool)
    # Exercise the exact tail tie, both signs, a finite-valid denominator and the 90% boundary.
    pivots = [-4., -2., 0., 2., 3.] + [0.] * 8995
    legacy = _bootstrap(x, y, alpha=.05, calibration_scenario="independent_null")
    fisher_z, se, _ = _reference_fisher(
        x, y, valid, legacy.block_length, historical_v1=False,
    )

    tied_se = abs(fisher_z) / 2

    def studentization(*args, **kwargs):
        return fisher_z, tied_se, 0.

    monkeypatch.setattr(time_series, "_fisher_v2_studentization", studentization)
    monkeypatch.setattr(time_series, "_adaptive_pivots", lambda *args, **kwargs: iter(pivots))
    options = dict(alpha=.05, n_boot=9999, min_valid_replicates=9000, calibration_scenario="independent_null")
    before = _bootstrap(x, y, **options)
    estimate = _bootstrap(x, y, p_value=True, **options)
    observed = abs(fisher_z / tied_se)
    exceedances = sum(value <= -observed or value >= observed for value in pivots)
    assert se > 0 and observed == 2 and exceedances == 4
    assert estimate.p_value == (1 + exceedances) / (len(pivots) + 1)
    assert estimate.p_value_reason is None
    assert estimate.method_version == "ts_boot_adaptive_v3_p"
    assert asdict(estimate) | {
        "p_value": None, "p_value_reason": None, "method_version": before.method_version,
    } == asdict(before)

    use = evaluate_interval_usability(
        estimate, point_estimate=.1, horizon=1, ic_method="spearman", intervals_verified=True,
    )
    assert use.calibration_status == "uncalibrated"
    assert not use.significance_interpretation_usable
    ledger = TrialLedger([("ts", 1)])
    ledger.record(("ts", 1), Trial(.1, estimate.p_value, 512, (estimate.lower, estimate.upper), use))
    selection = ledger.select_best(method="holm", alpha=.0025)
    assert selection.best is None
    assert selection.inference_adjusted_p_values == {}
    assert selection.trials[0].adjusted_p_value == estimate.p_value


@pytest.mark.parametrize(("grid", "requested", "valid_draws", "missing_intervals", "expected"), [
    (512, 999, 999, False, "insufficient_replicates_for_pvalue"),
    (512, 9999, 8999, False, "insufficient_replicates_for_pvalue"),
    (512, 9999, 9000, True, "bootstrap_interval_unestimable"),
    (256, 9999, 9000, False, "outside_pvalue_calibration_domain"),
])
def test_bootstrap_pvalue_unavailable_reasons(monkeypatch, grid, requested, valid_draws, missing_intervals, expected):
    x, y = np.random.default_rng(64).normal(size=(2, grid))
    monkeypatch.setattr(time_series, "_adaptive_pivots", lambda *args, **kwargs: repeat(0., valid_draws))
    interval = _bootstrap(
        x, y, p_value=True, n_boot=requested, min_valid_replicates=2, alpha=.05,
        calibration_scenario="independent_null", intervals_verified=not missing_intervals,
    )
    assert interval.p_value is None
    assert interval.p_value_reason == expected


def test_pvalue_opt_in_defaults_and_multi_asset_aggregation(monkeypatch):
    x, y = np.random.default_rng(64).normal(size=(2, 1024))
    dates = pd.date_range("2026-01-05", periods=512, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "asset"])
    panel = EvaluationPanel.from_long(pd.DataFrame({"factor": x, "1D": y}, index=index))
    entries = np.repeat(dates.to_numpy()[:, None], 2, axis=1)
    monkeypatch.setattr(time_series, "_adaptive_pivots", lambda *args, **kwargs: repeat(0., kwargs["n_boot"]))
    full = EvaluationEngine("time_series", "daily").evaluate(
        panel, entry_timestamps={1: entries}, exit_timestamps={1: entries + np.timedelta64(1, "D")},
        p_value=True, calibration_scenario="independent_null",
    )
    horizon = full.splits["full"].statistics.horizons[1]
    assert horizon.p_value is None
    assert horizon.p_value_reason == "multi_asset_pvalue_undefined"
    for asset in panel.assets:
        interval = horizon.instruments[asset].bootstrap
        assert interval.n_boot == interval.valid_replicates == 9999
        assert interval.min_valid_replicates == 9000
        assert interval.p_value == 1 / 10000
