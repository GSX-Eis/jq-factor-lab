"""
Core utilities for factor synthesis.

This module provides extracted, reusable utilities from the main FactorSynthesizer class.
"""

from .feature_preprocessor import (
    cross_sectional_standardize,
    cross_sectional_binning,
    cross_sectional_winsorize_zscore,
)
from .result_persistence import check_table_exists, initialize_local_tables
from .factor_data_loader import (
    load_factor_data,
    process_raw_factor_df,
    get_pricing_data,
    get_open_price_data,
    get_complete_price_data,
    get_factor_ratio_data,
    load_market_cap_data,
    # PIT adjustment helpers
    apply_pit_adjustment_factor_end,
    apply_pit_adjustment_factor_end_df,
)
from .pool_manager import (
    build_stock_filters,
    apply_dynamic_stock_pool_filter,
    get_all_pool_stocks,
    compute_predict_period_dynamic_pools,
    compute_dynamic_pools_for_topk,
    apply_skip_months,
)
from .calendar_skip import (
    SkipRules,
    parse_skip_rules,
    skip_mask,
    date_in_skip,
    describe_skip_rules,
)
from .factor_ops import (
    LABEL_MODE_CLOSE_TO_CLOSE,
    LABEL_MODE_NEXT_REBALANCE_CLOSE,
    LABEL_MODE_NEXT_REBALANCE_OPEN,
    calculate_forward_returns,
    calculate_scheduled_price_returns,
    resolve_label_definition,
    apply_feature_preprocessing,
)

__all__ = [
    # Feature preprocessor
    "cross_sectional_standardize",
    "cross_sectional_binning",
    "cross_sectional_winsorize_zscore",
    # Result persistence
    "check_table_exists",
    "initialize_local_tables",
    # Factor data loader
    "load_factor_data",
    "process_raw_factor_df",
    "get_pricing_data",
    "get_open_price_data",
    "get_complete_price_data",
    "get_factor_ratio_data",
    "load_market_cap_data",
    # PIT adjustment helpers
    "apply_pit_adjustment_factor_end",
    "apply_pit_adjustment_factor_end_df",
    # Pool manager
    "build_stock_filters",
    "apply_dynamic_stock_pool_filter",
    "get_all_pool_stocks",
    "compute_predict_period_dynamic_pools",
    "compute_dynamic_pools_for_topk",
    "apply_skip_months",
    # Calendar skip rules
    "SkipRules",
    "parse_skip_rules",
    "skip_mask",
    "date_in_skip",
    "describe_skip_rules",
    # Factor ops
    "LABEL_MODE_CLOSE_TO_CLOSE",
    "LABEL_MODE_NEXT_REBALANCE_CLOSE",
    "LABEL_MODE_NEXT_REBALANCE_OPEN",
    "calculate_forward_returns",
    "calculate_scheduled_price_returns",
    "resolve_label_definition",
    "apply_feature_preprocessing",
]
