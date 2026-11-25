"""Frozen adaptive calibration benchmark, applicability and trial binding."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import cycle, product
import json
from math import inf, nextafter, sqrt
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from backtest_and_analysis.evaluation import EvaluationEngine, evaluate_interval_usability
from backtest_and_analysis.evaluation import calibration
from backtest_and_analysis.evaluation.calibration import Trial, TrialLedger
from backtest_and_analysis.evaluation.cross_section import hac_p_value
from backtest_and_analysis.evaluation.panel import EvaluationPanel
from backtest_and_analysis.evaluation.types import (
    BOOTSTRAP_V1, BOOTSTRAP_V2, BootstrapInterval, HACInterval, interval_usability_payload,
)
from backtest_and_analysis.time_series.inference import (
    BarTrialFamily, _cross_section_rows, _read_trial_usability, _record_trials, _time_series_rows,
)
from backtest_and_analysis.time_series.artifacts import _json_value
from tools.benchmarks import adaptive_inference_calibration as benchmark


def test_registered_adaptive_calibration_reports_match_frozen_hashes() -> None:
    for domain in calibration.REGISTERED_ADAPTIVE_CALIBRATIONS:
        report = Path(__file__).resolve().parents[1] / domain.report_ref
        assert report.is_file(), f"{domain.domain_id}: missing calibration report {domain.report_ref}"
        assert sha256(report.read_bytes()).hexdigest() == domain.report_sha256, domain.domain_id


@pytest.mark.slow
def test_quick_cli_emits_all_preregistered_cells(tmp_path: Path) -> None:
    output = tmp_path / "quick.json"
    subprocess.run(
        [sys.executable, "-m", "tools.benchmarks.adaptive_inference_calibration", "--quick",
         "--output", str(output), "--workers", "1"],
        check=True, timeout=180,
    )
    report = json.loads(output.read_text())
    assert (report["seed"], report["repetitions"], report["release_passes"]) == (93673, 3, None)
    assert (report["fixed_boot"], report["adaptive_boot"], report["adaptive_min_valid"]) == (19, 19, 18)
    ids = [f"{section}:{i:03d}" for section, count in enumerate((8, 8, 108, 36)) for i in range(count)]
    assert [row["cell_id"] for row in report["coverage"]] == ids
    assert [row["cell_id"] for row in report["multiplicity"]] == [f"4:{i:03d}" for i in range(4)]
    assert [row["n_alternatives"] for row in report["multiplicity"]] == [0, 2, 0, 2]
    for row in report["coverage"]:
        auto = row["adaptive"]
        assert auto["unavailable_count"] == 0
        assert sum(auto["selected_windows"].values()) == 3
        if row["method"] == "hac":
            assert auto["in_domain_count"] == 3
        else:
            assert auto["in_domain_count"] == 0
            assert set(auto["out_of_domain_reasons"]) == {"n_boot", "min_valid_replicates"}


@pytest.mark.slow
def test_s5_quick_cli_preserves_matrix_and_resumes_by_cell(tmp_path: Path) -> None:
    output = tmp_path / "adaptive-calibration-s5-quick.json"
    command = [
        sys.executable, "-m", "tools.benchmarks.adaptive_inference_calibration",
        "--profile", "s5", "--quick", "--output", str(output), "--workers", "2",
    ]
    subprocess.run(command, check=True, timeout=180)
    original = output.read_bytes()
    report = json.loads(original)
    assert (report["benchmark_profile"], report["seed"], report["repetitions"]) == ("s5", 93679, 3)
    assert (report["fixed_boot"], report["adaptive_boot"], report["adaptive_min_valid"]) == (19, 19, 18)
    assert all(report[key] is None for key in ("release_passes", "hac_release_passes", "bootstrap_release_passes"))
    rows = report["coverage"] + report["multiplicity"]
    counts = (8, 8, 54, 18, 4)
    assert [row["cell_id"] for row in rows] == [
        f"{section}:{index:03d}" for section, count in enumerate(counts) for index in range(count)
    ]
    for row in rows:
        assert row["seed_spawn_prefix"] == list(map(int, row["cell_id"].split(":")))
    design_fields = ("n_times", "n_assets", "horizon", "gap")
    assert [tuple(row[field] for field in design_fields) for row in rows[16:70]] == list(
        product((256, 512, 1024), (32, 64), (1, 5, 20), (0, 3, 5))
    )
    assert [tuple(row[field] for field in design_fields) for row in rows[70:88]] == list(
        product((512, 1024), (1,), (1, 5, 20), (0, 3, 5))
    )
    assert [(row["scenario"], row["n_alternatives"]) for row in rows[88:]] == list(
        product(("iid", "combined"), (0, 2))
    )
    for row in rows[88:]:
        auto = row["adaptive"]
        assert auto["in_domain_count"] == 3 and auto["unavailable_families"] == 0
        assert "fwer" in auto["methods"]["bh"]

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([*command, "--resume"], check=True, timeout=30, capture_output=True)
    output.unlink()
    checkpoints = output.with_suffix(".cells")
    manifest = checkpoints / "profile.json"
    saved_manifest = manifest.read_bytes()
    manifest.write_text(json.dumps(json.loads(saved_manifest) | {"seed": 93673}))
    mismatch = subprocess.run([*command, "--resume"], timeout=30, capture_output=True, text=True)
    assert mismatch.returncode != 0 and "Checkpoint profile does not match" in mismatch.stderr
    manifest.write_bytes(saved_manifest)
    retained = checkpoints / "0:000.json"
    retained_mtime = retained.stat().st_mtime_ns
    pending = checkpoints / "2:053.json"
    pending.unlink()
    swapped = checkpoints / "2:000.json"
    saved_cell = swapped.read_bytes()
    s4_cell = next(cell for cell in benchmark._matrix() if cell.cell_id == swapped.stem)
    s4_result = benchmark._run_cell(s4_cell, benchmark.QUICK, False)
    assert (s4_result["n_times"], s4_result["n_assets"]) == (128, 16)
    saved_row = json.loads(saved_cell)
    for foreign_cell in (
        s4_result,
        saved_row | {"n_assets": 16},
        saved_row | {"run_fingerprint": "0" * 64},
    ):
        benchmark.write_json(swapped, foreign_cell)
        mismatch = subprocess.run([*command, "--resume"], timeout=60, capture_output=True, text=True)
        assert mismatch.returncode != 0 and "Checkpoint data does not match frozen cell 2:000" in mismatch.stderr
        assert not pending.exists() and not output.exists()
    swapped.write_bytes(saved_cell)
    subprocess.run([*command, "--resume"], check=True, timeout=60)
    assert retained.stat().st_mtime_ns == retained_mtime
    assert output.read_bytes() == original


def test_s5_mixed_family_only_gates_bh_fdr(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two true discoveries and a BH-only false discovery per family.
    for s5 in (False, True):
        p_values = cycle((.001, .001, .02, 1., 1., 1.))
        monkeypatch.setattr(benchmark, "hac_p_value", lambda *_: (next(p_values), "test"))
        row = benchmark._family(
            benchmark.Cell(4, 1, benchmark.CORE[0], n_alternatives=2),
            replace(benchmark.S5_QUICK, seed=42, repetitions=4), s5=s5,
        )
        bh = row["adaptive"]["methods"]["bh"]
        assert bh["fwer"] == 1. and bh["fdr"] == 1 / 3
        assert not bh["fwer_passes"] and bh["fdr_passes"]
        assert row["passes"] is s5


@pytest.mark.parametrize(("failed_section", "gates"), [(4, (False, True)), (3, (True, False))])
def test_s5_release_gate_is_independent_per_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_section: int, gates: tuple[bool, bool],
) -> None:
    def stub_cell(cell: benchmark.Cell, profile: benchmark.Profile, s5: bool) -> dict:
        assert s5 and profile.seed == 43
        return dict(
            **asdict(cell.design), n_alternatives=cell.n_alternatives,
            cell_id=cell.cell_id, seed_spawn_prefix=[cell.section, cell.index],
            repetitions=profile.repetitions, passes=(cell.section, cell.index) != (failed_section, 0),
        )

    # Exercise report aggregation without drawing from the acceptance seed.
    monkeypatch.setattr(benchmark, "S5_FORMAL", replace(benchmark.S5_FORMAL, seed=43))
    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(benchmark, "_run_cell", stub_cell)
    report = benchmark.run_benchmark(quick=False, output=tmp_path / "formal.json", workers=2, profile_name="s5")
    assert (report["hac_release_passes"], report["bootstrap_release_passes"]) == gates
    assert report["release_passes"] is False


@pytest.fixture
def evidence_registry(monkeypatch: pytest.MonkeyPatch) -> tuple[calibration.AdaptiveCalibrationDomain, ...]:
    # Fixture observations exercise the gate without asserting any empirical release.
    domains = tuple(replace(
        calibration._candidate_adaptive_domain(
            cross_section=cross_section, scenario="generalized_ar_heteroskedastic", horizon=1, gap=0,
            v2_boot=not cross_section,
        ),
        report_ref="fixture.json", report_sha256="a" * 64, coverage_cell_ids=("2:000" if cross_section else "3:000",),
        covered=475, repetitions=500,
    ) for cross_section in (True, False))
    monkeypatch.setattr(calibration, "REGISTERED_ADAPTIVE_CALIBRATIONS", domains)
    return domains


@pytest.fixture
def hac() -> HACInterval:
    point, variance, df = .4, .01, 12
    width = float(t.isf(.025, df)) * sqrt(variance)
    return HACInterval(
        512, 512, 5, .05, np.array([511, 510, 509, 508, 507]), variance,
        point - width, point + width, None,
        method_version="cs_hac_adaptive_v1", missing_pattern="none",
        calibration_scenario="generalized_ar_heteroskedastic", bandwidth_method="nw94_prewhitened",
        min_lag=4, effective_df=df,
    )


@pytest.fixture
def bootstrap() -> BootstrapInterval:
    return BootstrapInterval(
        512, 5, 0, 1000, 900, 900, .05, -.1, .2, None, n_valid_pairs=512,
        method_version=BOOTSTRAP_V2[0], missing_pattern="none",
        calibration_scenario="generalized_ar_heteroskedastic", block_length_method="ppw2009",
        resample_method="circular_block", interval_method=BOOTSTRAP_V2[-1], min_block_length=5,
        min_pairs_per_replicate=4, effective_blocks=512 / 5, fisher_z_standard_error=.02,
    )


def test_candidate_domains_use_full_audit_observations(evidence_registry, hac, bootstrap) -> None:
    for interval, domain in zip((hac, bootstrap), evidence_registry, strict=True):
        use = evaluate_interval_usability(
            interval, point_estimate=.4, horizon=5, cross_section_sizes=np.full(512, 32),
            ic_method="spearman", intervals_verified=True,
        )
        evidence = use.calibration_evidence
        assert use.calibration_status == "calibrated_pass"
        assert use.verified_horizon == 5 and use.significance_interpretation_usable
        assert evidence.domain_id == domain.domain_id and evidence.report_sha256 == domain.report_sha256
        assert evidence.sample_size == 512 and evidence.window == 5
        assert use.adaptive_observation.n_valid == 512


@pytest.mark.parametrize(("patch", "width", "horizon", "ic_method"), [
    ({"method_version": "cs_hac_adaptive_v2"}, 32, 5, "spearman"),
    ({"bandwidth_method": "fixed"}, 32, 5, "spearman"),
    ({"calibration_scenario": None}, 32, 5, "spearman"),
    ({"grid_size": 127, "n_valid": 127}, 32, 5, "spearman"),
    ({"grid_size": 1025, "n_valid": 1025}, 32, 5, "spearman"),
    ({"n_valid": 89}, 32, 5, "spearman"),
    ({"missing_pattern": "other"}, 32, 5, "spearman"),
    ({"missing_pattern": "periodic_17_3"}, 32, 5, "spearman"),
    ({"effective_df": nextafter(4., 0.)}, 32, 5, "spearman"),
    ({"effective_df": 512}, 32, 5, "spearman"),
    ({"variance": float("nan")}, 32, 5, "spearman"),
    ({"alpha": .04}, 32, 5, "spearman"),
    ({"upper": -1.}, 32, 5, "spearman"),
    ({"min_lag": 3}, 32, 5, "spearman"),
    ({"lag": 3}, 32, 5, "spearman"),
    ({"lag": 128}, 32, 5, "spearman"),
    ({}, 15, 5, "spearman"),
    ({}, 65, 5, "spearman"),
    ({}, None, 5, "spearman"),
    ({}, 32, 2, "spearman"),
    ({}, 32, None, "spearman"),
    ({}, 32, 5, "pearson"),
])
def test_hac_outside_domain_fails_closed(evidence_registry, hac, patch, width, horizon, ic_method) -> None:
    interval = replace(hac, **patch)
    sizes = None if width is None else np.full(interval.grid_size, width)
    use = evaluate_interval_usability(
        interval, point_estimate=.4, horizon=horizon, cross_section_sizes=sizes, ic_method=ic_method,
    )
    assert use.calibration_status == "uncalibrated"
    assert use.calibration_evidence is None and use.verified_horizon is None
    assert not use.significance_interpretation_usable


def test_hac_width_must_be_constant_across_effective_grid(evidence_registry, hac) -> None:
    widths = np.full(hac.grid_size, 32)
    widths[0] = 31
    use = evaluate_interval_usability(
        hac, point_estimate=.4, horizon=5, cross_section_sizes=widths, ic_method="spearman",
    )
    assert use.adaptive_observation.cross_section_width_min == 31
    assert use.adaptive_observation.cross_section_width_max == 32
    assert not use.significance_interpretation_usable


@pytest.mark.parametrize(("patch", "verified"), [
    ({"method_version": BOOTSTRAP_V1[0]}, True),
    ({"fisher_z_standard_error": None}, True),
    ({"fisher_z_standard_error": True}, True),
    ({"fisher_z_standard_error": inf}, True),
    ({"grid_size": 768, "n_valid_pairs": 768, "effective_blocks": 768 / 5}, True),
    ({"resample_method": "moving_block"}, True),
    ({"interval_method": "percentile"}, True),
    ({"n_valid_pairs": None}, True),
    ({"fisher_z_standard_error": 0.}, True),
    ({"effective_blocks": 3.99}, True),
    ({"effective_blocks": 6.}, True),
    ({"block_length": 4, "effective_blocks": 128}, True),
    ({"block_length": 69, "effective_blocks": 512 / 69}, True),
    ({"min_block_length": 4}, True),
    ({"n_boot": 998}, True),
    ({"min_valid_replicates": 899}, True),
    ({"min_valid_replicates": 901}, True),
    ({"valid_replicates": 1001}, True),
    ({"n_boot": 1001}, True),
    ({"min_pairs_per_replicate": 5}, True),
    ({}, False),
])
def test_bootstrap_outside_domain_fails_closed(evidence_registry, bootstrap, patch, verified) -> None:
    use = evaluate_interval_usability(
        replace(bootstrap, **patch), point_estimate=.1, horizon=5,
        ic_method="spearman", intervals_verified=verified,
    )
    assert use.calibration_status == "uncalibrated"
    assert use.calibration_evidence is None and not use.significance_interpretation_usable


def test_generalized_domain_rejects_high_missing_fraction(evidence_registry, hac) -> None:
    size = 135
    valid = np.arange(size) % 17 < 12
    assert 1 - valid.sum() / size < .30
    # A synthetic stricter boundary exercises the independent missing-fraction check.
    domain = replace(evidence_registry[0], max_missing_fraction=.25)
    interval = replace(hac, grid_size=size, n_valid=int(valid.sum()), missing_pattern="periodic_17_5")
    use = evaluate_interval_usability(
        interval, point_estimate=.4, horizon=5, ic_method="spearman",
        cross_section_sizes=np.where(valid, 32, 0),
    )
    assert calibration._adaptive_domain_mismatches(domain, use.adaptive_observation) == ("missing_fraction",)


def test_unregistered_interval_cannot_reuse_fixed_evidence(monkeypatch, hac) -> None:
    monkeypatch.setattr(calibration, "REGISTERED_ADAPTIVE_CALIBRATIONS", ())
    fixed = calibration.REGISTERED_CALIBRATIONS[0]
    forged = replace(hac, lag=0, min_lag=0, calibration_status="calibrated_pass", calibration_evidence=fixed)
    use = evaluate_interval_usability(
        forged, point_estimate=.4, horizon=1, cross_section_sizes=np.full(512, 32), ic_method="spearman",
    )
    assert use.calibration_status == "uncalibrated"
    assert use.calibration_evidence is None and use.adaptive_observation is not None
    p, method = hac_p_value(forged, .4)
    trial = Trial(.4, p, 512, (forged.lower, forged.upper), use, method)
    key = ("factor", 1)
    ledger = TrialLedger([key])
    ledger.record(key, trial)
    result = ledger.select_best(method="holm")
    assert result.best is None and key not in result.inference_adjusted_p_values
    assert result.trials[0].adjusted_p_value == p


@pytest.mark.parametrize(("direction", "offset"), [(d, i) for d in (-1, 1) for i in (-1, 0, 1)])
def test_adaptive_trial_binds_p_value_at_and_around_exact_endpoint(evidence_registry, hac, direction, offset) -> None:
    half = float(t.isf(.025, hac.effective_df)) * sqrt(hac.variance)
    point = direction * (half if not offset else nextafter(half, 0 if offset < 0 else inf))
    interval = replace(hac, lower=point - half, upper=point + half)
    use = evaluate_interval_usability(
        interval, point_estimate=point, horizon=5, cross_section_sizes=np.full(512, 32), ic_method="spearman",
    )
    p, method = hac_p_value(interval, point)
    trial = Trial(point, p, 512, (interval.lower, interval.upper), use, method)
    assert (p > .05, p == .05, p < .05) == (offset < 0, offset == 0, offset > 0)
    assert trial.inference_eligible(horizon=5, alpha=.05)
    if not offset:
        for target in (0., 1.):
            assert not replace(trial, p_value=nextafter(.05, target)).inference_eligible(horizon=5, alpha=.05)


def _v2_hac_row(
    *, horizon: int = 1, legacy: bool = False, grid: tuple[int, int] | None = None, perfect_rank: bool = False,
) -> dict:
    rng = np.random.default_rng(11)
    periods, assets = grid if grid is not None else ((512, 32) if legacy else (128, 16))
    index = pd.MultiIndex.from_product([pd.date_range("2025-01-06", periods=periods, freq="min"), range(assets)])
    x = rng.normal(size=len(index))
    returns = x if perfect_rank else .35 * x + rng.normal(size=len(index))
    panel = EvaluationPanel.from_long(pd.DataFrame({"factor": x, horizon: returns}, index=index))
    stats = EvaluationEngine("cross_sectional").evaluate(
        panel, calibration_scenario="positive_effect" if legacy else "generalized_ar_heteroskedastic",
        hac_lag=0 if legacy else "auto",
    ).splits["full"].statistics
    row, = json.loads(json.dumps(_json_value(_cross_section_rows(stats, "1min")), allow_nan=False))
    return row


def test_s5_production_registry_is_bound_to_formal_hac_evidence() -> None:
    report_ref = "backtest_and_analysis/evaluation/calibration_data/cs_hac_adaptive_v1_coverage.json"
    *core, generalized = (
        domain for domain in calibration.REGISTERED_ADAPTIVE_CALIBRATIONS
        if domain.method_version == "cs_hac_adaptive_v1"
        and domain.report_ref == report_ref
    )
    report_bytes = (Path(__file__).resolve().parents[1] / report_ref).read_bytes()
    digest = sha256(report_bytes).hexdigest()
    report = json.loads(report_bytes)
    assert report["manifest_sha256"] == sha256(
        json.dumps({k: v for k, v in report.items() if k != "manifest_sha256"},
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert report["source_report_sha256"] == "b56486d13112787bff507143f56bbc82413112d05bc73dac4c79889321242d62"
    assert report["total_cells"] == 88 and report["registered_cells"] == len(report["coverage"]) == 62
    assert report["hac_release_passes"] is True and report["repetitions"] == 500
    rows = {row["cell_id"]: row for row in report["coverage"]}
    assert [domain.scenario for domain in core] == [design.scenario for design in benchmark.CORE]
    assert generalized.scenario == "generalized_ar_heteroskedastic"
    for index, domain in enumerate(core):
        design = benchmark.CORE[index]
        assert domain.coverage_cell_ids == (f"0:{index:03d}",)
        assert domain.grid_size_min == domain.grid_size_max == 512
        assert domain.cross_section_width_min == domain.cross_section_width_max == 32
        assert domain.horizons == (design.horizon,)
        assert domain.min_n_valid == (422 if design.gap else 512)
        assert domain.max_missing_fraction == 1 - domain.min_n_valid / 512
    assert (generalized.grid_size_min, generalized.grid_size_max) == (256, 1024)
    assert (generalized.cross_section_width_min, generalized.cross_section_width_max) == (32, 64)
    assert generalized.min_n_valid == 180 and generalized.max_missing_fraction == .30
    assert generalized.horizons == (1, 5, 20)
    assert generalized.missing_patterns == ("none", "periodic_17_3", "periodic_17_5")
    assert generalized.coverage_cell_ids == tuple(f"2:{index:03d}" for index in range(54))
    assert [domain.covered for domain in (*core, generalized)] == [465, 479, 478, 472, 470, 478, 468, 480, 461]
    for domain in (*core, generalized):
        assert domain.method_version == "cs_hac_adaptive_v1"
        assert domain.min_effective_df == 4.0 and domain.repetitions == report["repetitions"]
        assert domain.report_ref == report_ref
        assert domain.report_sha256 == digest
        support = [rows[cell_id] for cell_id in domain.coverage_cell_ids]
        assert all(row["passes"] is True and row["scenario"] == domain.scenario for row in support)
        weakest = min(support, key=lambda row: (-abs(row["covered"] - 475), row["cell_id"]))
        assert domain.covered == weakest["covered"]
    assert len(calibration.REGISTERED_CALIBRATIONS) == 14


@pytest.mark.parametrize(("grid", "mismatches"), [
    ((256, 32), ()), ((128, 32), ("grid_size", "n_valid")), ((512, 16), ("cross_section_width",)),
])
def test_s5_production_hac_domain_through_v2_and_trial_ledger(grid, mismatches) -> None:
    row = _v2_hac_row(grid=grid)
    use = row["interval_usability"]
    assert row["hac"]["reason"] is None and row["p_value"] < .05
    assert use["adaptive_observation"]["grid_size"] == grid[0]
    assert use["adaptive_observation"]["cross_section_width_min"] == grid[1]
    domain = calibration._candidate_adaptive_domain(
        cross_section=True, scenario="generalized_ar_heteroskedastic", horizon=1, gap=0, s5=True,
    )
    observation = calibration.AdaptiveCalibrationObservation(**use["adaptive_observation"])
    reasons = calibration._adaptive_domain_mismatches(domain, observation)
    assert reasons == mismatches
    eligible = not mismatches
    assert use["calibration_status"] == ("calibrated_pass" if eligible else "uncalibrated")
    assert use["significance_interpretation_usable"] is eligible
    if eligible:
        assert use["calibration_evidence"]["domain_id"] == domain.domain_id
        assert use["calibration_evidence"]["observed_coverage"] == 461 / 500
    else:
        assert use["calibration_evidence"] is None
    key = ("candidate", 1)
    family = BarTrialFamily([key])
    _record_trials(family, [row], keys=(key,), mode="cross_sectional", window=("full", None))
    report = family.report()
    candidate, = report["candidates"]
    assert report["best"] == (key if eligible else None)
    assert candidate["inference_adjusted_p_value"] == (row["p_value"] if eligible else None)


def test_s3_production_registry_is_bound_to_formal_bootstrap_evidence() -> None:
    report_ref = "backtest_and_analysis/evaluation/calibration_data/ts_bootstrap_v2_coverage.json"
    *core, generalized = (
        domain for domain in calibration.REGISTERED_ADAPTIVE_CALIBRATIONS
        if domain.method_version == BOOTSTRAP_V2[0]
    )
    assert [domain.scenario for domain in core] == [design.scenario for design in benchmark.CORE]
    assert generalized.scenario == "generalized_ar_heteroskedastic"
    assert generalized.grid_sizes == (512, 1024)
    assert generalized.horizons == (1, 5, 20)
    assert generalized.missing_patterns == ("none", "periodic_17_3", "periodic_17_5")
    assert generalized.coverage_cell_ids == tuple(f"3:{index:03d}" for index in range(18))
    assert all(domain.report_ref == report_ref for domain in (*core, generalized))
    report_bytes = (Path(__file__).resolve().parents[1] / generalized.report_ref).read_bytes()
    digest = sha256(report_bytes).hexdigest()
    report = json.loads(report_bytes)
    assert report["manifest_sha256"] == sha256(
        json.dumps({k: v for k, v in report.items() if k != "manifest_sha256"},
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert report["source_report_sha256"] == "f6ffd98efb29cd155589b24cb8396214ddab505d119139ac4a5c5b2b97370dc7"
    assert report["run_fingerprint"] == "52d85917443f9b134e121e636d71be144eb8f5c40960e4080161df04c669ca84"
    assert len(report["coverage"]) == report["registered_cells"] == report["total_cells"] == 26
    assert report["bootstrap_release_passes"] is True and report["repetitions"] == 500
    rows = {row["cell_id"]: row for row in report["coverage"]}
    for index, domain in enumerate(core):
        row = rows[f"1:{index:03d}"]
        assert domain.coverage_cell_ids == (row["cell_id"],)
        assert domain.grid_size_min == domain.grid_size_max == row["n_times"]
        assert domain.horizons == (row["horizon"],)
        assert domain.missing_patterns == (("none" if row["gap"] == 0 else f"periodic_17_{row['gap']}"),)
    for domain in (*core, generalized):
        assert domain.report_sha256 == digest and domain.repetitions == report["repetitions"]
        support = [rows[cell_id] for cell_id in domain.coverage_cell_ids]
        assert all(row["passes"] is True and row["scenario"] == domain.scenario for row in support)
        weakest = min(support, key=lambda row: (-abs(row["covered"] - 475), row["cell_id"]))
        assert domain.covered == weakest["covered"]
    assert not any(domain.method_version == BOOTSTRAP_V1[0] for domain in calibration.REGISTERED_ADAPTIVE_CALIBRATIONS)


@pytest.mark.parametrize("size", [128, 256, 512, 768, 1024])
def test_v2_production_bootstrap_domain_on_verified_grid(size: int) -> None:
    x, noise = np.random.default_rng(64).normal(size=(2, size))
    dates = pd.date_range("2025-01-06", periods=size, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["datetime", "asset"])
    panel = EvaluationPanel.from_long(pd.DataFrame({"factor": x, "5D": .35 * x + noise}, index=index))
    entries = dates.to_numpy()[:, None]
    scenario = "generalized_ar_heteroskedastic"
    result = EvaluationEngine("time_series", "daily").evaluate(
        panel, entry_timestamps={5: entries}, exit_timestamps={5: entries + np.timedelta64(5, "D")},
        calibration_scenario=scenario,
    ).splits["full"].statistics
    row, = json.loads(json.dumps(_json_value(_time_series_rows(result, "A", "1D")), allow_nan=False))
    raw = row["bootstrap"]
    assert raw["reason"] is None and raw["n_valid_pairs"] == size
    declared = row["interval_usability"]

    def read(interval, usability):
        return _read_trial_usability(
            row, interval, usability, mode="time_series", count=size, has_bounds=True, point_usable=True,
        )

    eligible = size in (512, 1024)
    use = read(raw, declared)
    assert use.calibration_status == ("calibrated_pass" if eligible else "uncalibrated")
    assert use.significance_interpretation_usable is eligible
    assert use.verified_horizon == (5 if eligible else None)
    if eligible:
        assert use.calibration_evidence.domain_id == f"{BOOTSTRAP_V2[0]}:{scenario}"
        assert use.calibration_evidence.observed_coverage == 462 / 500
        for changed in (
            raw | {"fisher_z_standard_error": raw["fisher_z_standard_error"] * 2},
            {key: value for key, value in raw.items() if key != "min_pairs_per_replicate"},
        ):
            assert read(changed, declared).calibration_status == "uncalibrated"
        for changes in ({"min_pairs_per_replicate": 5}, {"missing_pattern": "other"}):
            outside = replace(BootstrapInterval(**raw), **changes)
            outside_use = evaluate_interval_usability(
                outside, point_estimate=row["ic"]["spearman_ic"], horizon=5,
                ic_method="spearman", intervals_verified=True,
            )
            assert outside_use.calibration_status == "uncalibrated"
            assert read(asdict(outside), interval_usability_payload(outside_use)).calibration_status == "uncalibrated"

    key = ("candidate", 5)
    family = BarTrialFamily([key])
    _record_trials(family, [row], keys=(key,), mode="time_series", window=("full", None))
    assert family.report()["best"] is None
    trial = Trial(row["ic"]["spearman_ic"], None, size, (raw["lower"], raw["upper"]), use)
    ledger = TrialLedger([key])
    ledger.record(key, trial)
    with pytest.raises(ValueError, match="Missing raw p-value"):
        ledger.select_best(method="holm")


def test_v2_evidence_round_trip_selects_and_rejects_tampering(evidence_registry) -> None:
    row = _v2_hac_row()
    assert row["interval_usability"]["calibration_status"] == "calibrated_pass"
    keys = (("matching", 1), ("wrong_sha", 1), ("wrong_p", 1), ("fixed_p_method", 1), ("wrong_variance", 1))
    family = BarTrialFamily(keys)
    for key in keys:
        candidate = json.loads(json.dumps(row))
        use = candidate["interval_usability"]
        if key[0] == "wrong_sha":
            use["calibration_evidence"]["report_sha256"] = "b" * 64
        if key[0] == "wrong_p":
            candidate["p_value"] /= 2
        if key[0] == "fixed_p_method":
            candidate["p_value_method"] = "two_sided_normal_hac"
        if key[0] == "wrong_variance":
            use["adaptive_observation"]["hac_variance"] *= 2
        _record_trials(family, [candidate], keys=(key,), mode="cross_sectional", window=("full", None))
    report = family.report()
    assert report["best"] == keys[0]
    assert report["candidates"][0]["interval_usability"] == row["interval_usability"]
    assert [c["significant"] for c in report["candidates"]] == [True, False, False, False, False]
    assert all(c["inference_adjusted_p_value"] is None for c in report["candidates"][1:])


@pytest.mark.parametrize(("section", "field", "value"), [
    ("hac", "method_version", "unknown"),
    ("hac", "calibration_scenario", "iid"),
    ("hac", "grid_size", 129),
    ("hac", "n_valid", 127),
    ("hac", "lag", -1),
    ("hac", "min_lag", 1),
    ("hac", "variance", 1.),
    ("hac", "variance", None),
    ("hac", "effective_df", 3.),
    ("hac", "alpha", .1),
    ("hac", "missing_pattern", "periodic_17_3"),
    ("hac", "bandwidth_method", "fixed"),
    ("hac", "reason", "variance_unavailable"),
    ("ic", "n_valid", 127),
])
def test_v2_hac_conflicting_fields_disqualify_candidate(evidence_registry, section, field, value) -> None:
    row = _v2_hac_row()
    assert row["interval_usability"]["calibration_status"] == "calibrated_pass"
    suspect = json.loads(json.dumps(row))
    assert suspect[section][field] != value
    suspect[section][field] = value
    suspect_key, valid_key = ("suspect", 1), ("valid", 1)
    family = BarTrialFamily([suspect_key, valid_key])
    for key, candidate in ((suspect_key, suspect), (valid_key, row)):
        _record_trials(family, [candidate], keys=(key,), mode="cross_sectional", window=("full", None))

    report = family.report()
    assert report["best"] == valid_key
    rejected, accepted = report["candidates"]
    assert rejected["p_value"] == row["p_value"]
    assert rejected["statistic"] == row["ic"]["mean"]
    assert rejected["interval_usability"]["descriptive_stats_usable"]
    assert rejected["interval_usability"]["calibration_status"] == "uncalibrated"
    assert rejected["inference_adjusted_p_value"] is None
    assert accepted["inference_adjusted_p_value"] == 2 * row["p_value"]


@pytest.mark.parametrize(("section", "field"), [("hac", "effective_df"), ("adaptive_observation", "n_valid")])
def test_v2_missing_hac_field_clears_calibration_claim(evidence_registry, section, field) -> None:
    row = _v2_hac_row()
    assert row["interval_usability"]["calibration_status"] == "calibrated_pass"
    payload = row["hac"] if section == "hac" else row["interval_usability"][section]
    del payload[field]
    key = ("missing", 1)
    family = BarTrialFamily([key])
    _record_trials(family, [row], keys=(key,), mode="cross_sectional", window=("full", None))
    report = family.report()
    candidate, = report["candidates"]
    assert report["selection_complete"] and report["best"] is None
    assert candidate["interval_usability"]["calibration_status"] == "uncalibrated"
    assert candidate["interval_usability"]["descriptive_stats_usable"]
    assert candidate["inference_adjusted_p_value"] is None


@pytest.mark.parametrize(("path", "replacement"), [
    (("interval_usability", "calibration_evidence", "report_ref"), None),
    (("interval_usability", "calibration_evidence", "report_sha256"), None),
    (("interval_usability", "calibration_evidence"), None),
    (("hac", "lower"), None),
    (("hac", "upper"), None),
    (("hac", "reason"), None),
    (("interval_usability", "calibration_status"), None),
    (("interval_usability", "reason"), None),
    (("interval_usability", "significance_interpretation_usable"), None),
    (("interval_usability", "descriptive_stats_usable"), None),
    (("interval_usability", "verified_horizon"), None),
    (("interval_usability", "verified_horizon"), 2),
    (("interval_usability", "verified_horizon"), True),
    pytest.param(("horizon_bars",), True, id="boolean-row-horizon"),
    pytest.param(("horizon_bars",), 1.0, id="float-row-horizon"),
    (("interval_usability",), None),
])
def test_v2_incomplete_production_evidence_preserves_family(path: tuple[str, ...], replacement: int | float | None) -> None:
    complete = _v2_hac_row(grid=(256, 32))
    assert complete["interval_usability"]["calibration_status"] == "calibrated_pass"
    incomplete = json.loads(json.dumps(complete))
    target = incomplete
    for part in path[:-1]:
        target = target[part]
    if replacement is None:
        del target[path[-1]]
    else:
        target[path[-1]] = replacement
    bad_key, good_key = ("incomplete", 1), ("complete", 1)
    family = BarTrialFamily([bad_key, good_key])
    for key, row in ((bad_key, incomplete), (good_key, complete)):
        _record_trials(family, [row], keys=(key,), mode="cross_sectional", window=("full", None))

    report = family.report()
    assert report["selection_complete"] and report["attempt_count"] == 2
    assert report["best"] == good_key
    rejected, accepted = report["candidates"]
    assert rejected["statistic"] == complete["ic"]["mean"]
    assert rejected["n_observations"] == complete["ic"]["n_valid"]
    assert rejected["p_value"] == complete["p_value"]
    assert rejected["inference_adjusted_p_value"] is None and not rejected["significant"]
    use = rejected["interval_usability"]
    assert use["descriptive_stats_usable"]
    assert use["calibration_status"] == "uncalibrated" and use["calibration_evidence"] is None
    assert not use["significance_interpretation_usable"] and use["verified_horizon"] is None
    assert accepted["inference_adjusted_p_value"] == 2 * complete["p_value"] and accepted["significant"]


def test_v2_missing_declaration_preserves_zero_std_ic_mean() -> None:
    row = _v2_hac_row(grid=(256, 32), perfect_rank=True)
    ic = row["ic"]
    assert (ic["mean"], ic["n_valid"], ic["icir"], ic["reason"]) == (1.0, 256, None, "zero_ic_std")
    assert row["interval_usability"]["descriptive_stats_usable"]
    del row["interval_usability"]

    key = ("zero-std", 1)
    family = BarTrialFamily([key])
    _record_trials(family, [row], keys=(key,), mode="cross_sectional", window=("full", None))
    report = family.report()
    candidate, = report["candidates"]
    assert candidate["statistic"] == 1.0 and candidate["n_observations"] == 256
    use = candidate["interval_usability"]
    assert use["descriptive_stats_usable"] and use["calibration_status"] == "uncalibrated"
    assert not use["significance_interpretation_usable"] and report["best"] is None


@pytest.mark.parametrize("damage", ["missing_df", "zero_ic_count"])
def test_v2_incomplete_observation_preserves_family_with_legacy_report(damage: str) -> None:
    incomplete = _v2_hac_row(horizon=5)
    use = incomplete["interval_usability"]
    assert use["calibration_status"] == "uncalibrated"
    count = incomplete["hac"]["n_valid"]
    assert count == use["adaptive_observation"]["n_valid"] == incomplete["ic"]["n_valid"] == 128
    if damage == "missing_df":
        del use["adaptive_observation"]["effective_df"]
    else:
        incomplete["ic"]["n_valid"] = 0

    legacy = _v2_hac_row(legacy=True)
    del legacy["interval_usability"]["adaptive_observation"]
    for field in ("method_version", "calibration_scenario", "bandwidth_method", "min_lag",
                  "missing_pattern", "effective_df", "prewhitening_ar1", "critical_value"):
        del legacy["hac"][field]
    del legacy["p_value_method"]
    assert legacy["interval_usability"]["calibration_status"] == "calibrated_pass"

    incomplete_key, legacy_key = ("incomplete", 5), ("legacy", 1)
    keys = (incomplete_key, legacy_key)
    family = BarTrialFamily(keys)
    _record_trials(family, [incomplete, legacy], keys=keys, mode="cross_sectional", window=("full", None))

    report = family.report()
    assert report["selection_complete"] and report["attempt_count"] == 2
    assert report["pending_candidates"] == []
    assert report["best"] == legacy_key
    rejected, accepted = report["candidates"]
    assert rejected["p_value"] == incomplete["p_value"]
    assert rejected["statistic"] == incomplete["ic"]["mean"]
    assert rejected["n_observations"] == count
    assert rejected["interval_usability"]["descriptive_stats_usable"]
    assert rejected["interval_usability"]["calibration_status"] == "uncalibrated"
    assert rejected["interval_usability"]["adaptive_observation"] is None
    assert rejected["inference_adjusted_p_value"] is None
    assert accepted["inference_adjusted_p_value"] == 2 * legacy["p_value"]


@pytest.mark.parametrize("field", ["cross_section_width_min", "cross_section_width_max"])
def test_v2_null_hac_width_clears_calibration_claim(evidence_registry, field: str) -> None:
    row = _v2_hac_row()
    use = row["interval_usability"]
    assert use["calibration_status"] == "calibrated_pass"
    assert use["adaptive_observation"][field] == 16
    use["adaptive_observation"][field] = None

    key = ("incomplete_width", 1)
    family = BarTrialFamily([key])
    _record_trials(family, [row], keys=(key,), mode="cross_sectional", window=("full", None))
    report = family.report()
    candidate, = report["candidates"]
    assert report["selection_complete"] and report["attempt_count"] == 1
    assert candidate["p_value"] == row["p_value"]
    assert candidate["statistic"] == row["ic"]["mean"]
    assert candidate["n_observations"] == row["ic"]["n_valid"]
    assert candidate["inference_adjusted_p_value"] is None and report["best"] is None
    readback = candidate["interval_usability"]
    assert readback["descriptive_stats_usable"]
    assert readback["calibration_status"] == "uncalibrated"
    assert readback["calibration_evidence"] is None
    assert not readback["significance_interpretation_usable"]


def test_time_series_interval_pass_does_not_supply_a_trial_p_value(evidence_registry, bootstrap) -> None:
    use = evaluate_interval_usability(
        bootstrap, point_estimate=.1, horizon=5, ic_method="spearman", intervals_verified=True,
    )
    assert use.significance_interpretation_usable
    assert not Trial(.1, None, 512, (-.1, .2), use).inference_eligible(horizon=5, alpha=.05)


def test_bootstrap_read_requires_exact_v2_observation_and_evidence(evidence_registry, bootstrap):
    use = evaluate_interval_usability(
        bootstrap, point_estimate=.1, horizon=5, ic_method="spearman", intervals_verified=True,
    )
    raw_use = interval_usability_payload(use)
    interval = asdict(bootstrap)
    row = {"horizon_bars": 5, "intervals_verified": True,
           "ic": {"sample_count": 512, "spearman_ic": .1, "reason": None}}

    def read(row, interval, payload):
        return _read_trial_usability(
            row, interval, payload, mode="time_series", count=row["ic"]["sample_count"],
            has_bounds=True, point_usable=True,
        )

    assert read(row, interval, raw_use) == use
    for field in (
        "method_version", "block_length_method", "resample_method", "interval_method", "calibration_scenario",
        "grid_size", "n_valid_pairs", "block_length", "min_block_length", "min_pairs_per_replicate",
        "n_boot", "min_valid_replicates", "valid_replicates", "alpha", "effective_blocks", "missing_pattern",
        "fisher_z_standard_error", "reason",
    ):
        incomplete = {key: value for key, value in interval.items() if key != field}
        assert read(row, incomplete, raw_use).calibration_status == "uncalibrated"
    for changed_row, changed_interval, changed_use in (
        (row | {"intervals_verified": 1}, interval, raw_use),
        (row | {"horizon_bars": 5.0}, interval, raw_use),
        (row, interval | {"grid_size": 512.0}, raw_use),
        (row, interval | {"min_block_length": True}, raw_use),
        (row, interval | {"alpha": np.nan}, raw_use),
        (row, interval | {"fisher_z_standard_error": .03}, raw_use),
        (row, interval | {"lower": -1.01}, raw_use),
        (row, interval, raw_use | {"adaptive_observation": None}),
        (row, interval, raw_use | {"calibration_evidence": raw_use["calibration_evidence"] | {"report_sha256": "b" * 64}}),
        (row, interval, raw_use | {"adaptive_observation": raw_use["adaptive_observation"] | {"n_valid": 512.0}}),
        (row, interval, raw_use | {"significance_interpretation_usable": False}),
    ):
        downgraded = read(changed_row, changed_interval, changed_use)
        assert downgraded.calibration_status == "uncalibrated"
        assert downgraded.calibration_evidence is downgraded.verified_horizon is None
        assert downgraded.descriptive_stats_usable and not downgraded.significance_interpretation_usable
    assert use.adaptive_observation.fisher_z_standard_error == .02


@pytest.mark.parametrize(("recorded_count", "corroborated"), [(0, True), (512.0, True), (511, True), (0, False)])
def test_trial_family_reads_conflicting_bootstrap_sample_count(
    evidence_registry, bootstrap, recorded_count, corroborated,
):
    keys = (("candidate", 1), ("candidate", 5))
    rows = []
    for _, horizon in keys:
        interval = replace(bootstrap, min_block_length=horizon)
        use = evaluate_interval_usability(
            interval, point_estimate=.1, horizon=horizon, ic_method="spearman", intervals_verified=True,
        )
        assert use.calibration_status == "calibrated_pass"
        rows.append({
            "horizon_bars": horizon, "intervals_verified": True, "bootstrap": asdict(interval),
            "ic": {"sample_count": 512 if horizon == 1 else recorded_count, "spearman_ic": .1, "reason": None},
            "interval_usability": interval_usability_payload(use),
        })
    if not corroborated:
        rows[1]["interval_usability"]["adaptive_observation"]["n_valid"] = 510
    family = BarTrialFamily(keys)
    _record_trials(family, rows, keys=keys, mode="time_series", window=("full", None))
    valid, conflicted = family.report()["candidates"]
    assert valid["interval_usability"]["calibration_status"] == "calibrated_pass"
    assert valid["n_observations"] == 512 and valid["interval"] == (-.1, .2)
    assert conflicted["n_observations"] == (512 if corroborated else 0)
    assert conflicted["statistic"] == .1 and conflicted["reason"] == "inconsistent_sample_count"
    assert conflicted["interval"] is None
    use = conflicted["interval_usability"]
    assert use["descriptive_stats_usable"] is True and use["calibration_status"] == "uncalibrated"
    assert use["significance_interpretation_usable"] is False
    assert use["calibration_evidence"] is use["verified_horizon"] is None


@pytest.mark.slow
@pytest.mark.parametrize(
    "corruption", ["zero_coverage", "fixed_zero_coverage", "in_domain", "budget_reasons", "valid_fraction"],
)
def test_v2_quick_cli_resume_rejects_impossible_counts(tmp_path, monkeypatch, corruption):
    output = tmp_path / "quick.json"
    monkeypatch.setenv("QUANTATION_CODE_VERSION", "a" * 40)
    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", ThreadPoolExecutor)
    benchmark.run_benchmark(quick=True, output=output, workers=2, profile_name="v2_boot")
    checkpoints = output.with_suffix(".cells")
    path = checkpoints / "1:000.json"
    pending = checkpoints / "3:017.json"
    pending.unlink()
    row = json.loads(path.read_text())
    adaptive = row["adaptive"]
    if corruption in ("zero_coverage", "fixed_zero_coverage"):
        estimate = adaptive if corruption == "zero_coverage" else row["fixed"][0]
        assert estimate["unavailable_count"] == 0
        estimate["rejects_zero"] = (estimate["rejects_zero"] + 1) % 4
    elif corruption == "in_domain":
        adaptive.update(in_domain_count=3, in_domain_covered=adaptive["covered"], out_of_domain_count=0,
                        domain_passes=True, out_of_domain_reasons={})
    elif corruption == "budget_reasons":
        adaptive["out_of_domain_reasons"].pop("n_boot")
    else:
        adaptive["out_of_domain_reasons"]["valid_fraction"] = row["repetitions"]
    benchmark.write_json(path, row)
    result = subprocess.run(
        [sys.executable, "-m", "tools.benchmarks.adaptive_inference_calibration", "--profile", "v2_boot",
         "--quick", "--resume", "--output", str(output), "--workers", "1"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0 and "does not match frozen cell 1:000" in result.stderr
    assert not pending.exists() and not output.exists()


def test_v2_cli_writes_report_to_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path.resolve()
    report_path = Path("reports/quick.json")
    target = repo / report_path
    alias = repo / "alias.json"
    alias.symlink_to(report_path)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(benchmark, "REPO_ROOT", repo)
    monkeypatch.setattr(benchmark, "_matrix", lambda **_: [])
    monkeypatch.setenv("QUANTATION_CODE_VERSION", "a" * 40)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--profile", "v2_boot", "--quick", "--output", alias.name])

    benchmark.main()

    assert alias.is_symlink() and alias.resolve() == target
    assert target.is_file()
    report = json.loads(target.read_text())
    checkpoint = json.loads((target.with_suffix(".cells") / "profile.json").read_text())
    assert report["output"] == checkpoint["output"] == report_path.as_posix()


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_v2_repo_output_is_normalized_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool,
) -> None:
    repo = tmp_path.resolve()
    cwd = repo / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(benchmark, "REPO_ROOT", repo)
    monkeypatch.setenv("QUANTATION_CODE_VERSION", "a" * 40)
    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", ThreadPoolExecutor)
    report_path = Path("reports/quick.json")
    relative_output = Path("..") / report_path
    absolute_output = repo / report_path
    output, resume_output = (absolute_output, relative_output) if absolute else (relative_output, absolute_output)
    options = dict(quick=True, workers=2, profile_name="v2_boot")

    report = benchmark.run_benchmark(output=output, **options)
    manifest = absolute_output.with_suffix(".cells") / "profile.json"
    assert json.loads(manifest.read_text())["output"] == report["output"] == report_path.as_posix()
    assert benchmark.run_benchmark(output=resume_output, resume=True, **options) == report


def test_v2_quick_profile_checkpoint_binds_matrix_and_budget(tmp_path, monkeypatch):
    output = tmp_path / "quick.json"
    monkeypatch.setenv("QUANTATION_CODE_VERSION", "a" * 40)
    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", ThreadPoolExecutor)
    options = dict(quick=True, output=output, workers=2, profile_name="v2_boot")
    report = benchmark.run_benchmark(**options)
    assert report["release_passes"] is report["bootstrap_release_passes"] is None
    assert report["multiplicity"] == []
    rows = report["coverage"]
    assert [row["cell_id"] for row in rows] == [
        *(f"1:{i:03}" for i in range(8)), *(f"3:{i:03}" for i in range(18)),
    ]
    assert [tuple(row[key] for key in ("n_times", "horizon", "gap")) for row in rows[8:]] == list(
        product((512, 1024), (1, 5, 20), (0, 3, 5))
    )
    assert all(row["passes"] is None and row["n_assets"] == 1 for row in rows)
    assert all(row["adaptive"]["unavailable_count"] == 0 for row in rows)
    assert all(row["adaptive"]["out_of_domain_reasons"] == {"n_boot": 3, "min_valid_replicates": 3} for row in rows)
    assert [[r["window"] for r in row["fixed"]] for row in rows[:8]] == [
        [1], [1, 17], [5, 17], [1], [1], [17], [1], [1],
    ]
    checkpoints = output.with_suffix(".cells")
    manifest = checkpoints / "profile.json"
    metadata = json.loads(manifest.read_text())
    fingerprint = sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    assert report["run_fingerprint"] == fingerprint
    assert metadata["output"] == report["output"] == str(output.resolve())
    assert metadata["source_revision"] == "a" * 40
    assert (metadata["seed"], metadata["repetitions"], metadata["adaptive_boot"],
            metadata["adaptive_min_valid"], metadata["fixed_boot"], metadata["fixed_min_valid"]) == (
        1538543278954444760, 3, 19, 18, 19, 19,
    )
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(**options)
    cell = checkpoints / "1:000.json"
    saved = json.loads(cell.read_text())
    other = checkpoints / "3:017.json"
    other.unlink()
    for patch in (
        {"run_fingerprint": "b" * 64}, {"seed_spawn_prefix": [1, 1]}, {"n_assets": True}, {"passes": False},
        {"adaptive_boot": 20}, {"adaptive": saved["adaptive"] | {"covered": True}},
        {"adaptive": saved["adaptive"] | {"in_domain_covered": 4}},
    ):
        benchmark.write_json(cell, saved | patch)
        with pytest.raises(ValueError, match="does not match frozen cell"):
            benchmark.run_benchmark(**options, resume=True)
        assert not other.exists()
    cell.write_text("{")
    with pytest.raises(ValueError, match="does not match frozen cell"):
        benchmark.run_benchmark(**options, resume=True)
    benchmark.write_json(cell, saved)
    foreign = checkpoints / "0:000.json"
    benchmark.write_json(foreign, saved)
    with pytest.raises(ValueError, match="outside the frozen matrix"):
        benchmark.run_benchmark(**options, resume=True)
    foreign.unlink()
    benchmark.write_json(manifest, metadata | {"source_revision": "b" * 40})
    with pytest.raises(ValueError, match="Checkpoint profile does not match"):
        benchmark.run_benchmark(**options, resume=True)
    benchmark.write_json(manifest, metadata)
    mtime = cell.stat().st_mtime_ns
    assert benchmark.run_benchmark(**options, resume=True) == report
    assert cell.stat().st_mtime_ns == mtime and other.is_file()
    benchmark.write_json(output, report)
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(**options, resume=True)
