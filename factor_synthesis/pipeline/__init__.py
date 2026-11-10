"""
Pipeline-style composite factor synthesis methods.

This subpackage contains the concrete implementation of different synthesis
algorithms (LASSO, PCA, LightGBM, weighted sum).  The main script
`run_factor_synthesis.py` imports and delegates to these functions so that
the orchestration code stays lightweight.

Preprocessing utilities are now based on Qlib Processors for consistent
behavior across all synthesis methods.
"""

from .lasso_method import run_lasso_synthesis
from .pca_method import run_pca_synthesis
from .lightgbm_method import run_lightgbm_synthesis
from .weighted_sum_method import run_weighted_sum_synthesis
from .size_conditional_method import (
    run_size_conditional_synthesis,
    write_size_conditional_diagnostics,
)
from .preprocessing import (
    apply_target_cs_preprocessing,
    compute_sample_weights,
)
from .dynamic_blend import (
    DynamicBlendConfig,
    DynamicBlendResult,
    compute_rolling_ic_stats,
    compute_prior_composite,
    compute_dynamic_blend_ratio,
    blend_predictions,
)

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

