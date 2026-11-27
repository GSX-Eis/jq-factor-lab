"""
Pipeline-style composite factor synthesis methods.

This subpackage contains the concrete implementation of different synthesis
algorithms (LASSO, PCA, LightGBM, weighted sum).  The main script
`run_factor_synthesis.py` imports and delegates to these functions so that
the orchestration code stays lightweight.

Preprocessing utilities are now based on Qlib Processors for consistent
behavior across all synthesis methods.
"""

import importlib

_EXPORTS = {
    "run_lasso_synthesis": ".lasso_method",
    "run_pca_synthesis": ".pca_method",
    "run_lightgbm_synthesis": ".lightgbm_method",
    "run_weighted_sum_synthesis": ".weighted_sum_method",
    "run_size_conditional_synthesis": ".size_conditional_method",
    "write_size_conditional_diagnostics": ".size_conditional_method",
    "apply_target_cs_preprocessing": ".preprocessing",
    "compute_sample_weights": ".preprocessing",
    "DynamicBlendConfig": ".dynamic_blend",
    "DynamicBlendResult": ".dynamic_blend",
    "compute_rolling_ic_stats": ".dynamic_blend",
    "compute_prior_composite": ".dynamic_blend",
    "compute_dynamic_blend_ratio": ".dynamic_blend",
    "blend_predictions": ".dynamic_blend",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "run_lasso_synthesis",
    "run_pca_synthesis",
    "run_lightgbm_synthesis",
    "run_weighted_sum_synthesis",
    "run_size_conditional_synthesis",
    "write_size_conditional_diagnostics",
    # Preprocessing utilities (Qlib-based)
    "apply_target_cs_preprocessing",
    "compute_sample_weights",
    # Dynamic blend utilities
    "DynamicBlendConfig",
    "DynamicBlendResult",
    "compute_rolling_ic_stats",
    "compute_prior_composite",
    "compute_dynamic_blend_ratio",
    "blend_predictions",
]

