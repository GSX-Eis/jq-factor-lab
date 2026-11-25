"""Show that the synthetic study preserves the frozen statistics and legacy samples."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from itertools import product
from math import sqrt
import os
from pathlib import Path
from shutil import copyfile
from statistics import NormalDist
import subprocess
import sys
from textwrap import dedent

import numpy as np
import pytest

from backtest_and_analysis.evaluation import kernels, time_series
from backtest_and_analysis.evaluation.calibration import _SCENARIOS
from tools.benchmarks import ts_pvalue_confirmatory, ts_pvalue_recalibration
from tools.benchmarks.ts_bootstrap_pvalue_calibration import _sample
from tools.benchmarks.ts_pvalue_dense_draws import legacy_sample, warp_statistic
from tools.benchmarks.ts_pvalue_features import feature_sample, innovations
from tools.benchmarks.ts_pvalue_recalibration_rules import apply_holm, check_scope, size_summary


@pytest.mark.parametrize("scenario,size,horizon", tuple(product(_SCENARIOS, (512, 1024, 2048), (1, 5, 20))))
def test_legacy_draws_are_bitwise_unchanged(scenario, size, horizon):
    expected = _sample(scenario, size, horizon, np.random.default_rng(725))
    actual = legacy_sample(scenario, size, horizon, np.random.default_rng(725))
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))


@pytest.mark.parametrize("feature,params", [
    ("student_t", {"df": 4}),
    ("garch", {"omega": 0.05, "alpha": 0.05, "beta": 0.90, "burn_in": 256}),
    ("jumps", {"intensity": 0.02, "jump_std": 6.0}),
])
def test_new_features_have_unit_variance_and_expected_dependence(feature, params):
    values = innovations(feature, np.random.default_rng(206), 300_000, params)
    assert 0.94 < values.var() < 1.06
    assert abs(values.mean()) < 0.01
    if feature != "garch":
        assert np.mean(np.abs(values) > 4) > 0.001
    squared = values**2
    lag = np.corrcoef(squared[1:], squared[:-1])[0, 1]
    assert lag > 0.04 if feature == "garch" else abs(lag) < 0.02
    for rho in (-0.35, 0.0, 0.35):
        x, y = feature_sample("test", feature, rho, params, 40_000, 20, np.random.default_rng(33))
        assert y.var() == pytest.approx(1.0, abs=0.11)
        assert np.corrcoef(x, y)[0, 1] == pytest.approx(rho, abs=0.035)


@pytest.mark.parametrize("scenario,horizon", tuple(product(_SCENARIOS, (1, 5, 20))))
def test_dense_warp_uses_the_reference_draw_and_studentization(scenario, horizon):
    x, y = _sample(scenario, 512, horizon, np.random.default_rng(502))
    mask = np.ones(512, dtype=bool)
    point = kernels.spearman(x[:, None], y[:, None], valid=mask[:, None], axis=0, min_samples=4)
    kernel = time_series._adaptive_bootstrap_inputs
    fit, z, pivots = kernel(x, y, mask, point=point, min_samples=4, min_block_length=horizon,
                            seed=364, n_boot=1, min_valid_replicates=2, alpha=0.05,
                            calibration_scenario=scenario, intervals_verified=True,
                            historical_v1=False, p_value=True)
    assert warp_statistic(x, y, horizon, 364, pivot=False) == pytest.approx(z / fit.fisher_z_standard_error, abs=1e-12)
    assert warp_statistic(x, y, horizon, 364, pivot=True) == pytest.approx(abs(next(pivots)), abs=1e-12)


def test_size_delta_and_holm_against_independent_formulas():
    critical = np.arange(1, 4001) / 4000
    observed = np.arange(1, 5001) / 5010
    alpha = 0.05
    row = size_summary(critical, observed, alpha)
    p = sum(v > 0.95 for v in observed) / len(observed)
    slope = sum(0.94 < v <= 0.96 for v in observed) / (len(observed) * 0.02)
    se = sqrt(p * (1 - p) / len(observed) + slope**2 * alpha * (1 - alpha) / len(critical))
    standard = NormalDist()
    assert row["ci99"] == pytest.approx([p - standard.inv_cdf(0.995) * se, p + standard.inv_cdf(0.995) * se])
    assert row["upper99"] == pytest.approx(p + standard.inv_cdf(0.99) * se)
    assert row["critical_variance"] > 0
    assert row["deviation_p"] == pytest.approx(2 * standard.cdf(-abs(p - alpha) / se))

    raw = [Fraction(1, 10), Fraction(1, 1000), Fraction(2, 1000), Fraction(1, 1000)]
    rows = [dict(deviation_p=float(value), hard_cap_pass=True) for value in raw]
    apply_holm([dict(sign=0, thresholds=rows)])
    for value, row in zip(raw, rows, strict=True):
        adjusted = max(p * sum(x >= p for x in raw) for p in raw if p <= value)
        assert row["holm_p"] == pytest.approx(float(adjusted))
        assert row["passes"] == (adjusted > Fraction(1, 100))


def _cell(feature, size, thresholds):
    return dict(feature=feature, grid_size=size, sign=0, thresholds=[
        dict(alpha=alpha, passes=passes, rejected=rejects, repetitions=10_000)
        for alpha, passes, rejects in thresholds
    ])


def test_scope_excludes_feature_and_only_the_registered_t_alpha_pair():
    cells = [_cell("gaussian", 512, [(0.05, False, 500), (0.01, True, 100)]),
             _cell("gaussian", 1024, [(0.05, True, 500), (0.01, True, 100)]),
             _cell("jumps", 1024, [(0.05, False, 600), (0.01, True, 100)])]
    narrowed = check_scope(cells, nested_pass=True)
    assert narrowed["passes"]
    assert narrowed["excluded_features"] == ["jumps"]
    assert narrowed["exclude_512_alpha_005"]
    assert not check_scope(cells, nested_pass=False)["passes"]
    cells[1]["thresholds"][1]["rejected"] = 111
    assert not check_scope(cells, nested_pass=True)["passes"]


@pytest.mark.parametrize("scenario,horizon", [
    ("independent_null", 1), ("ar1_null", 5), ("overlap_null", 20),
])
@pytest.mark.parametrize("size", [512, 1024])
def test_nested_p_value_matches_production_exactly(scenario, horizon, size):
    from tools.benchmarks.ts_pvalue_nested_draws import nested_draws, nested_p_value

    x, y = _sample(scenario, size, horizon, np.random.default_rng(76868))
    valid = np.ones(size, dtype=bool)
    point = kernels.spearman(x[:, None], y[:, None], valid=valid[:, None], axis=0, min_samples=4)
    settings = dict(point=point, min_samples=4, min_block_length=horizon, seed=488, n_boot=9999,
                    min_valid_replicates=9000, alpha=0.05, calibration_scenario=scenario, p_value=True)
    fit, z, production_pivots = time_series._adaptive_bootstrap_inputs(
        x, y, valid, intervals_verified=True, historical_v1=False, **settings,
    )
    obs, actual = nested_draws(x, y, horizon, 488, 9999)
    assert obs == z / fit.fisher_z_standard_error
    expected = np.fromiter(production_pivots, dtype=float)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    production = time_series._bootstrap_spearman(x, y, valid, **settings)
    assert nested_p_value(x, y, horizon, 488, 9999, 9000) == production.p_value


@pytest.mark.parametrize("mode", ["diagnostic", "warp", "nested"])
@pytest.mark.parametrize("changed_dependency", ["scenario", "source", "environment"])
def test_checkpoint_rejects_changed_calculation_dependencies(tmp_path, monkeypatch, mode, changed_dependency):
    # Exercise real chunk calculations in one interpreter so the scenario mutation is
    # observable independently of the platform's process start method.
    checkpoint = tmp_path / "cells"
    binding = {"preregistration_commit": "fixture", "design_sha256": "fixture"}
    if mode == "diagnostic":
        monkeypatch.setattr(ts_pvalue_recalibration, "ProcessPoolExecutor", ThreadPoolExecutor)
        chunks = [
            ts_pvalue_recalibration.Chunk("ar1_null", 512, 5, 127900, 0, 0, i, i + 1)
            for i in range(2)
        ]

        def run():
            return ts_pvalue_recalibration._run_chunks(chunks, checkpoint, 1, binding)
    else:
        monkeypatch.setattr(ts_pvalue_confirmatory, "ProcessPoolExecutor", ThreadPoolExecutor)
        design = {
            "grid_sizes": [512], "horizons": [5], "features": {},
            "scenarios": {"ar1_null": {"name": "ar1_null", "feature": "gaussian", "rho": 0}},
            "warp": {"seed": 127901, "null": {"critical": 1, "evaluation": 1}, "chunk_size": 1},
            "nested": {"seed": 127902, "cells": [("ar1_null", 512, 5)], "repetitions_per_cell": 2,
                       "chunk_size": 1, "n_boot": 31, "min_valid": 20},
        }

        def run():
            return ts_pvalue_confirmatory._run(design, mode, 1, checkpoint, binding)[1]

    reference = run()
    kept, missing = sorted(checkpoint.glob("*.npy"))
    kept_payload = kept.read_bytes()
    kept_inode = kept.stat().st_ino
    missing.unlink()
    resumed = run()
    assert kept.read_bytes() == kept_payload
    assert kept.stat().st_ino == kept_inode
    for key in reference:
        np.testing.assert_array_equal(reference[key].view(np.uint64), resumed[key].view(np.uint64))

    missing.unlink()
    if changed_dependency == "scenario":
        monkeypatch.setitem(_SCENARIOS, "ar1_null", replace(_SCENARIOS["ar1_null"], return_ar=0.4))
        if mode == "diagnostic":
            assert ts_pvalue_recalibration._warp_chunk(chunks[0])[0] != reference[0, 0][0]
    elif changed_dependency == "source":
        source = tmp_path / "time_series.py"
        source.write_bytes(Path(time_series.__file__).read_bytes() + b"\n# source drift\n")
        monkeypatch.setattr(time_series, "__file__", str(source))
    else:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "11")

    with pytest.raises(ValueError, match="Checkpoint.*changed"):
        run()
    assert kept.read_bytes() == kept_payload
    assert not missing.exists()

    (checkpoint / "binding.json").unlink()
    with pytest.raises(ValueError, match="Checkpoint binding missing"):
        run()
    assert kept.read_bytes() == kept_payload


@pytest.mark.slow
def test_nested_resume_recompiles_after_scoring_source_change(tmp_path):
    pytest.importorskip("numba")
    root = Path(__file__).resolve().parents[1]
    relative = Path("tools/benchmarks")
    sources = tmp_path / relative
    sources.mkdir(parents=True)
    (sources.parent / "__init__.py").write_text((root / "tools/__init__.py").read_text())
    for name in ("ts_pvalue_nested_draws.py", "ts_pvalue_dense_draws.py"):
        copyfile(root / relative / name, sources / name)

    driver = tmp_path / "run_nested.py"
    driver.write_text(dedent("""
        import multiprocessing
        from pathlib import Path
        import sys

        from tools.benchmarks.ts_pvalue_confirmatory import _run

        if __name__ == "__main__":
            multiprocessing.set_start_method("spawn")
            design = {
                "features": {},
                "scenarios": {"ar1_null": {"name": "ar1_null", "feature": "gaussian", "rho": 0}},
                "nested": {"seed": 127902, "cells": [("ar1_null", 512, 5)], "repetitions_per_cell": 2,
                           "chunk_size": 1, "n_boot": 999, "min_valid": 900},
            }
            binding = {"preregistration_commit": "fixture", "design_sha256": "fixture"}
            _run(design, "nested", 1, Path(sys.argv[1]), binding)
    """))

    def run(directory: str, cache: str) -> np.ndarray:
        env = os.environ | {
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(root))),
            "NUMBA_CACHE_DIR": str(tmp_path / cache),
            "NUMBA_DISABLE_JIT": "0",
        }
        result = subprocess.run([sys.executable, str(driver), str(tmp_path / directory)],
                                env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        chunks = sorted((tmp_path / directory).glob("*.npy"))
        assert len(chunks) == 2
        return np.concatenate([np.load(chunk, allow_pickle=False) for chunk in chunks])

    old = run("old", "warm-cache")
    assert any((tmp_path / "warm-cache").rglob("*_scores*.nbc"))
    scores_source = sources / "ts_pvalue_dense_draws.py"
    text = scores_source.read_text()
    needle = "return rho, score - rho, influence\n"
    assert text.count(needle) == 1
    scores_source.write_text(text.replace(needle, "return rho, score - rho, influence * 2\n"))

    first = run("resume", "warm-cache")
    kept, missing = sorted((tmp_path / "resume").glob("*.npy"))
    kept_inode = kept.stat().st_ino
    missing.unlink()
    resumed = run("resume", "empty-resume-cache")
    clean = run("clean", "empty-clean-cache")

    assert kept.stat().st_ino == kept_inode
    assert (tmp_path / "resume/binding.json").read_bytes() == (tmp_path / "clean/binding.json").read_bytes()
    assert np.isfinite(clean).all()
    assert np.all(old != clean)
    np.testing.assert_array_equal(
        np.stack([first, resumed]).view(np.uint64), np.stack([clean, clean]).view(np.uint64),
        err_msg=f"old={old}, warm-cache new={first}, resumed={resumed}, clean={clean}",
    )
