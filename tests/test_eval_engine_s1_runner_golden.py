"""Replay the current daily runner against the S1-A frozen outputs."""

import json

import numpy as np
import pytest

from fixtures.eval_engine.baseline.regenerate import (
    BASELINE_DIR,
    MODES,
    run_snapshot,
    synthetic_inputs,
)


@pytest.fixture(scope="module", params=["synthetic"])
def baseline(request: pytest.FixtureRequest) -> tuple:
    sample = request.param
    expected = json.loads((BASELINE_DIR / f"{sample}.json").read_text(encoding="utf-8"))
    identity, *inputs = synthetic_inputs()
    assert identity == expected["input"]
    return inputs, expected["paths"]


@pytest.mark.parametrize("mode", MODES)
def test_eval_engine_s1_runner_golden(baseline: tuple, mode: str) -> None:
    inputs, paths = baseline
    expected = paths[mode]
    actual = run_snapshot(*inputs, mode=mode)

    assert actual["max_loss"] == expected["max_loss"]
    assert actual["results"].keys() == expected["results"].keys()
    for metric, frame in actual["results"].items():
        frozen = expected["results"][metric]
        for axis in ("index", "index_names", "columns", "column_names"):
            assert frame[axis] == frozen[axis], (metric, axis)
        np.testing.assert_allclose(
            np.asarray(frame["values"], dtype=float),
            np.asarray(frozen["values"], dtype=float),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"{mode} {metric}",
        )
