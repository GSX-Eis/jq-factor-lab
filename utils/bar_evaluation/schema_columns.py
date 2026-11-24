"""Column order and value types for evaluation metric and artifact rows."""

_CSV_STRING_DTYPES: dict[str, str] = {
    "scope": "string",
    "instrument_id": "string",
    "frequency": "string",
    "ic_status": "string",
    "non_overlapping_status": "string",
    "quantile_status": "string",
    "quantile_scope": "string",
    "exclusion_counts": "string",
    "nonzero_exclusion_counts": "string",
}

_IC_TABLE_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "row_index",
    "analysis_mode",
    "scope",
    "instrument_id",
    "frequency",
    "horizon_bars",
    "total_count",
    "valid_count",
    "excluded_count",
    "exclusion_counts",
    "nonzero_exclusion_counts",
    "spearman_ic",
    "ic_status",
    "non_overlapping_ic",
    "non_overlapping_count",
    "non_overlapping_status",
    "requested_quantiles",
    "actual_quantiles",
    "quantile_status",
    "quantile_scope",
    "descriptive_only",
    "mean_spearman_ic",
    "median_spearman_ic",
    "mean_non_overlapping_ic",
    "median_non_overlapping_ic",
    "instrument_count",
    "valid_instrument_count",
    "valid_non_overlapping_instrument_count",
    "mean_ic",
    "median_ic",
    "std_ic",
    "icir",
    "positive_ic_ratio",
    "total_cross_sections",
    "valid_cross_sections",
    "excluded_cross_sections",
    "insufficient_cross_section_count",
    "constant_factor_cross_section_count",
    "constant_return_cross_section_count",
    "mean_valid_instruments",
    "median_valid_instruments",
)

_QUANTILE_TABLE_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "row_index",
    "analysis_mode",
    "instrument_id",
    "frequency",
    "horizon_bars",
    "quantile",
    "requested_quantiles",
    "actual_quantiles",
    "min_actual_quantiles",
    "max_actual_quantiles",
    "quantile_status",
    "quantile_scope",
    "descriptive_only",
    "sample_count",
    "cross_section_count",
    "total_sample_count",
    "mean_return_pct",
    "median_return_pct",
    "positive_ratio",
)

_CROSS_SECTION_TABLE_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "row_index",
    "frequency",
    "horizon_bars",
    "signal_timestamp",
    "total_count",
    "valid_count",
    "excluded_count",
    "exclusion_counts",
    "nonzero_exclusion_counts",
    "valid_instruments",
    "spearman_ic",
    "cs_status",
    "requested_quantiles",
    "actual_quantiles",
    "quantile_status",
    "quantile_scope",
    "descriptive_only",
)

_IC_INT_COLUMNS = frozenset({
    "row_index", "horizon_bars", "total_count", "valid_count", "excluded_count",
    "non_overlapping_count", "requested_quantiles", "actual_quantiles",
    "instrument_count", "valid_instrument_count", "valid_non_overlapping_instrument_count",
    "total_cross_sections", "valid_cross_sections", "excluded_cross_sections",
    "insufficient_cross_section_count", "constant_factor_cross_section_count",
    "constant_return_cross_section_count",
})
_IC_FLOAT_COLUMNS = frozenset({
    "spearman_ic", "non_overlapping_ic", "mean_spearman_ic", "median_spearman_ic",
    "mean_non_overlapping_ic", "median_non_overlapping_ic", "mean_ic", "median_ic",
    "std_ic", "icir", "positive_ic_ratio", "mean_valid_instruments", "median_valid_instruments",
})
_IC_BOOL_COLUMNS = frozenset({"descriptive_only"})
_IC_JSON_COLUMNS = frozenset({"exclusion_counts", "nonzero_exclusion_counts"})

_QUANTILE_INT_COLUMNS = frozenset({
    "row_index", "horizon_bars", "quantile", "requested_quantiles", "actual_quantiles",
    "min_actual_quantiles", "max_actual_quantiles", "sample_count",
    "cross_section_count", "total_sample_count",
})
_QUANTILE_FLOAT_COLUMNS = frozenset({"mean_return_pct", "median_return_pct", "positive_ratio"})
_QUANTILE_BOOL_COLUMNS = frozenset({"descriptive_only"})

_CS_INT_COLUMNS = frozenset({
    "row_index", "horizon_bars", "total_count", "valid_count", "excluded_count",
    "valid_instruments", "requested_quantiles", "actual_quantiles",
})
_CS_FLOAT_COLUMNS = frozenset({"spearman_ic"})
_CS_BOOL_COLUMNS = frozenset({"descriptive_only"})
_CS_JSON_COLUMNS = frozenset({"exclusion_counts", "nonzero_exclusion_counts"})
