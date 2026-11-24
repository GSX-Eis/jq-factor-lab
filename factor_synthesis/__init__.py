"""Reusable factor synthesis algorithms; no experiment presets or dashboards."""
from .factor_reader import FactorReader
from .strategy_bundle import (
    BundleArtifactRef,
    MANIFEST_FILENAME,
    STRATEGY_BUNDLE_SCHEMA_VERSION,
    StrategyBundle,
    StrategyBundleError,
    StrategyFactorSpec,
    compute_bundle_hash,
    load_strategy_bundle,
    verify_strategy_bundle,
    write_strategy_bundle,
)

__all__ = [
    "FactorComposer",
    "SynthesisConfig",
    "FactorReader",
    "CompositeFactorBase",
    "BundleArtifactRef",
    "MANIFEST_FILENAME",
    "STRATEGY_BUNDLE_SCHEMA_VERSION",
    "StrategyBundle",
    "StrategyBundleError",
    "StrategyFactorSpec",
    "compute_bundle_hash",
    "load_strategy_bundle",
    "verify_strategy_bundle",
    "write_strategy_bundle",
]


def __getattr__(name: str):
    from importlib import import_module

    modules = {
        "FactorComposer": ".factor_composer",
        "SynthesisConfig": ".factor_composer",
        "CompositeFactorBase": ".composite_factor_base",
    }
    if name not in modules:
        raise AttributeError(name)
    try:
        value = getattr(import_module(modules[name], __name__), name)
    except ModuleNotFoundError as exc:
        if exc.name not in {"sklearn", "dcor", "joblib", "threadpoolctl", "lightgbm"}:
            raise
        raise ImportError("Install synthesis dependencies with python3 -m pip install 'jq-factor-lab[synthesis]'") from exc
    globals()[name] = value
    return value
