from tools.benchmarks.trial_ledger_calibration import run_benchmark


def test_quick_benchmark_preserves_each_cell_and_replays_seed():
    report = run_benchmark(quick=True)
    assert report == run_benchmark(quick=True)
    assert report["synthetic"] and report["profile"] == "quick"
    assert (report["seed"], report["repetitions"], report["n_boot"]) == (91773, 3, 19)
    coverage, families = report["coverage"], report["multiplicity"]
    assert len(coverage) == 16 and len(families) == 4
    assert len({tuple(row["seed_spawn_key"]) for row in coverage + families}) == 20
    assert {row["scenario"] for row in coverage} == {
        "iid", "strong_ar", "overlap", "gaps", "heteroskedastic", "combined",
    }
    for row in coverage:
        assert row["covered"] + row["unavailable_count"] <= report["repetitions"]
        assert row["coverage"] == row["covered"] / report["repetitions"]
        assert row["passes"] is (row["within_tolerance"] and not row["unavailable_count"])
    for row in families:
        assert row["n_alternatives"] + row["n_null_candidates"] == 6
        assert set(row["methods"]) == {"raw", "holm", "bh"}
        for values in row["methods"].values():
            assert 0 <= values["fdr"] <= values["fwer"] <= 1
            if not row["n_alternatives"]:
                assert values["fdr"] == values["fwer"]
