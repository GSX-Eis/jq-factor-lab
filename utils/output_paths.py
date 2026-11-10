"""Execution output bindings. Explicit paths win; defaults are attempt-local."""

import copy
import json
import logging
import os
import shutil
import subprocess
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from utils.local_outputs import (
    managed_output, output_directory, output_lock, output_root, read_output_marker,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Compatibility constants; core callers use the runtime helpers below.
OUTPUTS_ROOT = Path(os.environ.get("QUANTATION_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
CACHE_DIR = PROJECT_ROOT / "cache"
FACTOR_CACHE_DIR = CACHE_DIR / "factor_data"
FORWARD_RETURNS_CACHE_DIR = CACHE_DIR / "forward_returns"




def execution_directory() -> Path:
    supplied = os.environ.get("QUANT_EXECUTION_DIR")
    if supplied:
        return Path(supplied).expanduser().resolve()
    root = (
        Path(os.environ.get("QUANTATION_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
        .expanduser()
        .resolve()
    )
    return root / "agent_harness/standalone" / uuid.uuid4().hex




def diagnostic_file(name: str, *, directory=None) -> Path | None:
    directory = directory or os.environ.get("QUANT_DIAGNOSTICS_DIR")
    if not directory:
        return None
    path = Path(directory).expanduser().resolve() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def execution_environment(directory: Path):
    """Lease subprocess scratch/default cache; inherited env binds all children."""
    directory = directory.resolve()
    with ExitStack() as stack:
        scratch, scratch_fd = stack.enter_context(managed_output(None, "execution"))
        cache, cache_fd = stack.enter_context(
            managed_output(None, "execution-inputs", kind="cache")
        )
        env = os.environ | {
            "QUANT_EXECUTION_DIR": str(directory),
            "QUANT_CACHE_DIR": str(cache),
            "QUANT_DIAGNOSTICS_DIR": str(directory / "diagnostics"),
            "TMPDIR": str(scratch),
            "TMP": str(scratch),
            "TEMP": str(scratch),
        }
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "output_bindings.json",
            {
                "schema_version": "execution_outputs.v1",
                "execution_dir": str(directory),
                "scratch": str(scratch),
                "cache": str(cache),
                "retention": {
                    "execution_dir": "evidence",
                    "scratch": "temporary",
                    "cache": "rebuildable",
                },
            },
        )
        yield env, (scratch_fd, cache_fd)






def open_diagnostic(name: str, mode: str, *, directory=None, enabled=False):
    enabled = enabled or os.environ.get("QUANT_DEBUG_OUTPUTS") == "1"
    path = diagnostic_file(name, directory=directory) if enabled else None
    return open(path if path is not None else os.devnull, mode, encoding="utf-8")


def qlib_tracking_config():
    root = execution_directory() / "mlruns"
    return {
        "class": "LocalMLflowExpManager",
        "module_path": "factor_synthesis.qlib_adapter.experiment_manager",
        "kwargs": {"uri": root.as_uri(), "default_exp_name": "Experiment"},
    }


@contextmanager
def process_output_environment(directory):
    """CLI-only binding; library calls must not mutate the process environment."""
    directory = Path(directory).resolve()
    keys = (
        "QUANT_EXECUTION_DIR",
        "QUANT_CACHE_DIR",
        "QUANT_DIAGNOSTICS_DIR",
        "TMPDIR",
        "TMP",
        "TEMP",
    )
    if os.environ.get("QUANT_EXECUTION_DIR") == str(directory):
        with ExitStack() as leases:
            try:
                if not all(os.environ.get(key) for key in keys):
                    raise ValueError("Incomplete output binding; enter through execution_environment")
                bindings = json.loads((directory / "output_bindings.json").read_text())
                if (
                    not isinstance(bindings, dict)
                    or bindings.get("schema_version") != "execution_outputs.v1"
                    or bindings.get("execution_dir") != str(directory)
                    or bindings.get("cache") != os.environ["QUANT_CACHE_DIR"]
                    or bindings.get("scratch") != os.environ["TMPDIR"]
                    or os.environ["TMP"] != os.environ["TMPDIR"]
                    or os.environ["TEMP"] != os.environ["TMPDIR"]
                    or os.environ["QUANT_DIAGNOSTICS_DIR"] != str(directory / "diagnostics")
                ):
                    raise ValueError("Invalid execution output binding")
                for field, kind, category_name in (
                    ("scratch", "temporary", "tmp"), ("cache", "cache", "cache")
                ):
                    path = Path(bindings[field])
                    if (
                        not path.is_absolute() or path != path.resolve()
                        or path.parent.name != category_name
                        or output_directory(path.parent.parent, category_name, path.name, create=False) != path
                    ):
                        raise ValueError("Invalid bound output path")
                    category = ExitStack()
                    leases.callback(category.close)
                    if category.enter_context(output_lock(path.parent, create=False)) is None:
                        raise ValueError("Missing output category lease")
                    marker = read_output_marker(path, kind)
                    if marker.get("cleanup"):
                        raise ValueError("Output cleanup is incomplete")
                    if marker["schema_version"] == "local_output.v2":
                        if leases.enter_context(output_lock(path, create=False)) is None:
                            raise ValueError("Missing output entry lease")
                        category.close()
            except OSError as exc:
                raise ValueError("Invalid output binding; enter through execution_environment") from exc
            yield
        return
    import tempfile

    before = {key: os.environ.get(key) for key in keys}
    previous_temp = tempfile.tempdir
    with execution_environment(directory) as (environment, _leases):
        try:
            os.environ.update({key: environment[key] for key in keys})
            tempfile.tempdir = None
            yield
        finally:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            tempfile.tempdir = previous_temp
