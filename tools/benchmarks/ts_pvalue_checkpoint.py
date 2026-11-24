"""Bind calibration chunks to the code and runtime that produced their values."""
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from threadpoolctl import threadpool_info

from tools.benchmarks import ts_pvalue_dense_draws
from utils import numba_compat


BINDING_KEYS = (
    "preregistration_commit", "design_sha256", "source_sha256", "legacy_scenarios",
    "python", "numpy", "scipy", "environment",
)
_SHARED_MODULES = (
    "tools.benchmarks.ts_pvalue_dense_draws",
    "backtest_and_analysis.evaluation.calibration",
    "backtest_and_analysis.evaluation.time_series",
    "backtest_and_analysis.evaluation.kernels",
    "backtest_and_analysis.evaluation.types",
    "utils.numba_compat",
)
_RUNTIME_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OMP_DYNAMIC",
    "MKL_NUM_THREADS", "MKL_DYNAMIC", "MKL_CBWR", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NPY_DISABLE_CPU_FEATURES", "NPY_ENABLE_CPU_FEATURES",
    "NUMBA_DISABLE_JIT", "NUMBA_NUM_THREADS", "NUMBA_CPU_NAME", "NUMBA_CPU_FEATURES",
    "NUMBA_THREADING_LAYER", "NUMBA_OPT", "NUMBA_ENABLE_AVX",
)


def _environment() -> dict:
    blas = [
        {key: value for key, value in library.items() if key != "filepath"}
        for library in threadpool_info() if library["user_api"] == "blas"
    ]
    numba_version = getattr(numba_compat.numba, "__version__", None)
    jit = None
    if numba_version is not None:
        import llvmlite
        from llvmlite import binding as llvm

        config = numba_compat.numba.config
        jit = {
            "numba": numba_version,
            "llvmlite": llvmlite.__version__,
            "llvm": list(llvm.llvm_version_info),
            "host_cpu": llvm.get_host_cpu_name(),
            "host_features": llvm.get_host_cpu_features().flatten(),
            "config": {name: str(getattr(config, name)) for name in (
                "DISABLE_JIT", "NUMBA_NUM_THREADS", "CPU_NAME", "CPU_FEATURES",
                "THREADING_LAYER", "OPT", "ENABLE_AVX",
            )},
        }
    return {
        "python_implementation": platform.python_implementation(),
        "python_build": list(platform.python_build()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "blas": sorted(blas, key=lambda library: json.dumps(library, sort_keys=True)),
        "numba_jit": jit,
        "variables": {name: os.environ[name] for name in _RUNTIME_VARIABLES if name in os.environ},
    }


def calculation_binding(identity: dict, runner: str, extra_modules: tuple[str, ...] = ()) -> dict:
    sources = {Path(runner).name: Path(runner), Path(__file__).name: Path(__file__)}
    sources.update({name: Path(sys.modules[name].__file__) for name in (*_SHARED_MODULES, *extra_modules)})
    binding = identity | {
        "source_sha256": {name: sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "legacy_scenarios": {name: asdict(scenario) for name, scenario in ts_pvalue_dense_draws._SCENARIOS.items()},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "environment": _environment(),
    }
    if "environment" in identity and binding != identity:
        raise ValueError("Checkpoint calculation inputs changed after startup")
    return binding


def bind_checkpoint(directory: Path, binding: dict) -> None:
    """Reject legacy or incompatible chunks before any chunk can be loaded."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "binding.json"
    if not manifest.exists():
        if next(directory.glob("*.npy"), None) is not None:
            raise ValueError(f"Checkpoint binding missing at {manifest}. Use a new checkpoint directory.")
        try:
            with manifest.open("x") as stream:
                json.dump(binding, stream, allow_nan=False)
            return
        except FileExistsError:
            pass
    try:
        previous = json.loads(manifest.read_text())
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid checkpoint binding at {manifest}. Use a new checkpoint directory.") from exc
    if previous != binding:
        raise ValueError(f"Checkpoint binding changed at {manifest}. Use a new checkpoint directory.")
