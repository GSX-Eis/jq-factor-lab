"""DuckDB evaluation tables, column types, exceptions and ingest receipt."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

from backtest_and_analysis.time_series import RESULT_SCHEMA_VERSION

_BASE_TABLES: tuple[str, ...] = (
    "bar_evaluation_runs",
    "bar_evaluation_ic",
    "bar_evaluation_quantiles",
    "bar_evaluation_cross_sections",
)
_SPLIT_TABLE = "bar_evaluation_split_metrics"
BAR_EVALUATION_TABLES = (*_BASE_TABLES, _SPLIT_TABLE)
_SPLIT_NAMES = ("full", "train", "oos")
_CALIBRATION_STATUSES = ("uncalibrated", "calibrated_pass", "calibrated_fail")
SUPPORTED_RESULT_SCHEMA_VERSIONS: tuple[str, ...] = (RESULT_SCHEMA_VERSION,)
VALID_ANALYSIS_MODES: tuple[str, ...] = ("time_series", "cross_sectional")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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

_RUN_V2_COLUMN_TYPES = {
    "has_split": "BOOLEAN",
    "split_config_json": "VARCHAR",
    "trial_ledger_json": "VARCHAR",
    "calibration_summary_json": "VARCHAR",
}
_RUN_OPTIONAL_COLUMN_TYPES = _RUN_V2_COLUMN_TYPES | {"factor_id": "VARCHAR"}
_RUN_BOOL_COLUMNS = frozenset({"has_split"})
_RUN_JSON_COLUMNS = frozenset({
    "input_file_hashes", "data_identity", "exclusion_summary",
    "split_config_json", "trial_ledger_json", "calibration_summary_json",
})
_SPLIT_COLUMN_TYPES = {
    "identity_hash": "VARCHAR NOT NULL",
    "row_index": "BIGINT NOT NULL",
    "split": "VARCHAR NOT NULL",
    "frequency": "VARCHAR NOT NULL",
    "instrument_id": "VARCHAR",
    "horizon_bars": "BIGINT NOT NULL",
    "split_purged_count": "BIGINT NOT NULL",
    "ic": "DOUBLE",
    "ic_reason": "VARCHAR",
    "n_observations": "BIGINT NOT NULL",
    "n_dependence_adjusted": "DOUBLE",
    "interval_method": "VARCHAR NOT NULL",
    "interval_lower": "DOUBLE",
    "interval_upper": "DOUBLE",
    "calibration_status": "VARCHAR NOT NULL",
    "descriptive_stats_usable": "BOOLEAN NOT NULL",
    "significance_interpretation_usable": "BOOLEAN NOT NULL",
    "p_value": "DOUBLE",
    "p_value_method": "VARCHAR",
    "adjusted_p_value": "DOUBLE",
    "inference_adjusted_p_value": "DOUBLE",
    "significant": "BOOLEAN",
    "fit_identity_json": "VARCHAR",
    "metrics_json": "VARCHAR NOT NULL",
}
_SPLIT_TABLE_COLUMNS = tuple(_SPLIT_COLUMN_TYPES)
_SPLIT_INT_COLUMNS = frozenset(k for k, t in _SPLIT_COLUMN_TYPES.items() if t.startswith("BIGINT"))
_SPLIT_FLOAT_COLUMNS = frozenset(k for k, t in _SPLIT_COLUMN_TYPES.items() if t == "DOUBLE")
_SPLIT_BOOL_COLUMNS = frozenset(k for k, t in _SPLIT_COLUMN_TYPES.items() if t.startswith("BOOLEAN"))
_SPLIT_JSON_COLUMNS = frozenset({"fit_identity_json", "metrics_json"})

_RUN_SEMANTIC_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "analysis_mode",
    "frequency",
    "view",
    "boundary_policy",
    "bucket_strategy",
    "calendar_id",
    "calendar_hash",
    "config_hash",
    "input_file_hashes",
    "data_identity",
    "code_version",
    "code_version_source",
    "result_schema_version",
    "exclusion_summary",
    *_RUN_V2_COLUMN_TYPES,
)

_RUN_ARTIFACT_HASH_COLUMNS: tuple[str, ...] = (
    "summary_sha256",
    "per_bar_labels_sha256",
    "ic_by_horizon_sha256",
    "quantile_stats_sha256",
)

_RUN_ALL_COLUMNS: tuple[str, ...] = (
    "identity_hash",
    "analysis_mode",
    "frequency",
    "view",
    "boundary_policy",
    "bucket_strategy",
    "calendar_id",
    "calendar_hash",
    "config_hash",
    "input_file_hashes",
    "data_identity",
    "code_version",
    "code_version_source",
    "result_schema_version",
    "exclusion_summary",
    "summary_path",
    "summary_sha256",
    "per_bar_labels_path",
    "per_bar_labels_sha256",
    "ic_by_horizon_path",
    "ic_by_horizon_sha256",
    "quantile_stats_path",
    "quantile_stats_sha256",
    "run_id",
    "ingested_at_utc",
    *_RUN_OPTIONAL_COLUMN_TYPES,
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


class BarEvaluationPreconditionError(ValueError):
    """Raised when results DB path, schema prerequisites, or inputs are invalid."""


class BarEvaluationIntegrityError(ValueError):
    """Raised when published artifact files fail completeness or SHA256 verification."""


class BarEvaluationConflictError(ValueError):
    """Raised when re-ingesting an existing identity_hash with different content."""

    def __init__(self, identity_hash: str, message: str) -> None:
        self.identity_hash = identity_hash
        super().__init__(message)


@dataclass(frozen=True)
class BarEvaluationIngestReceipt:
    db_path: Path
    identity_hash: str
    status: str  # "inserted" or "noop"
    analysis_mode: str
    ic_rows: int
    quantile_rows: int
    cross_section_rows: int
    ingested_at_utc: str
    split_rows: int = 0


def _create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    optional_columns = ", ".join(f"{name} {kind}" for name, kind in _RUN_OPTIONAL_COLUMN_TYPES.items())
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS bar_evaluation_runs (
            identity_hash VARCHAR PRIMARY KEY,
            analysis_mode VARCHAR NOT NULL,
            frequency VARCHAR NOT NULL,
            view VARCHAR,
            boundary_policy VARCHAR,
            bucket_strategy VARCHAR,
            calendar_id VARCHAR,
            calendar_hash VARCHAR NOT NULL,
            config_hash VARCHAR NOT NULL,
            input_file_hashes VARCHAR NOT NULL,
            data_identity VARCHAR NOT NULL,
            code_version VARCHAR NOT NULL,
            code_version_source VARCHAR,
            result_schema_version VARCHAR NOT NULL,
            exclusion_summary VARCHAR NOT NULL,
            summary_path VARCHAR NOT NULL,
            summary_sha256 VARCHAR NOT NULL,
            per_bar_labels_path VARCHAR NOT NULL,
            per_bar_labels_sha256 VARCHAR NOT NULL,
            ic_by_horizon_path VARCHAR NOT NULL,
            ic_by_horizon_sha256 VARCHAR NOT NULL,
            quantile_stats_path VARCHAR NOT NULL,
            quantile_stats_sha256 VARCHAR NOT NULL,
            run_id VARCHAR,
            ingested_at_utc TIMESTAMPTZ NOT NULL,
            {optional_columns}
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bar_evaluation_ic (
            identity_hash VARCHAR NOT NULL,
            row_index BIGINT NOT NULL,
            analysis_mode VARCHAR NOT NULL,
            scope VARCHAR,
            instrument_id VARCHAR,
            frequency VARCHAR NOT NULL,
            horizon_bars BIGINT NOT NULL,
            total_count BIGINT,
            valid_count BIGINT,
            excluded_count BIGINT,
            exclusion_counts VARCHAR,
            nonzero_exclusion_counts VARCHAR,
            spearman_ic DOUBLE,
            ic_status VARCHAR,
            non_overlapping_ic DOUBLE,
            non_overlapping_count BIGINT,
            non_overlapping_status VARCHAR,
            requested_quantiles BIGINT,
            actual_quantiles BIGINT,
            quantile_status VARCHAR,
            quantile_scope VARCHAR,
            descriptive_only BOOLEAN,
            mean_spearman_ic DOUBLE,
            median_spearman_ic DOUBLE,
            mean_non_overlapping_ic DOUBLE,
            median_non_overlapping_ic DOUBLE,
            instrument_count BIGINT,
            valid_instrument_count BIGINT,
            valid_non_overlapping_instrument_count BIGINT,
            mean_ic DOUBLE,
            median_ic DOUBLE,
            std_ic DOUBLE,
            icir DOUBLE,
            positive_ic_ratio DOUBLE,
            total_cross_sections BIGINT,
            valid_cross_sections BIGINT,
            excluded_cross_sections BIGINT,
            insufficient_cross_section_count BIGINT,
            constant_factor_cross_section_count BIGINT,
            constant_return_cross_section_count BIGINT,
            mean_valid_instruments DOUBLE,
            median_valid_instruments DOUBLE,
            PRIMARY KEY (identity_hash, row_index)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bar_evaluation_quantiles (
            identity_hash VARCHAR NOT NULL,
            row_index BIGINT NOT NULL,
            analysis_mode VARCHAR NOT NULL,
            instrument_id VARCHAR,
            frequency VARCHAR NOT NULL,
            horizon_bars BIGINT NOT NULL,
            quantile BIGINT NOT NULL,
            requested_quantiles BIGINT,
            actual_quantiles BIGINT,
            min_actual_quantiles BIGINT,
            max_actual_quantiles BIGINT,
            quantile_status VARCHAR,
            quantile_scope VARCHAR,
            descriptive_only BOOLEAN,
            sample_count BIGINT,
            cross_section_count BIGINT,
            total_sample_count BIGINT,
            mean_return_pct DOUBLE,
            median_return_pct DOUBLE,
            positive_ratio DOUBLE,
            PRIMARY KEY (identity_hash, row_index)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS bar_evaluation_cross_sections (
            identity_hash VARCHAR NOT NULL,
            row_index BIGINT NOT NULL,
            frequency VARCHAR NOT NULL,
            horizon_bars BIGINT NOT NULL,
            signal_timestamp TIMESTAMPTZ NOT NULL,
            total_count BIGINT,
            valid_count BIGINT,
            excluded_count BIGINT,
            exclusion_counts VARCHAR,
            nonzero_exclusion_counts VARCHAR,
            valid_instruments BIGINT,
            spearman_ic DOUBLE,
            cs_status VARCHAR,
            requested_quantiles BIGINT,
            actual_quantiles BIGINT,
            quantile_status VARCHAR,
            quantile_scope VARCHAR,
            descriptive_only BOOLEAN,
            PRIMARY KEY (identity_hash, row_index)
        )"""
    )

    _create_split_table(conn)


def _create_split_table(conn: duckdb.DuckDBPyConnection) -> None:
    columns = ", ".join(f"{name} {kind}" for name, kind in _SPLIT_COLUMN_TYPES.items())
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {_SPLIT_TABLE} (
        {columns}, PRIMARY KEY (identity_hash, row_index)
    )""")


def _catalog_columns(conn: duckdb.DuckDBPyConnection) -> dict[str, dict[str, str]]:
    rows = conn.execute("""
        SELECT t.table_name, c.column_name, c.data_type
        FROM duckdb_tables() AS t JOIN duckdb_columns() AS c USING (database_name, schema_name, table_name)
        WHERE t.database_name = current_database() AND t.schema_name = 'main'
          AND NOT t.temporary AND t.table_name IN (SELECT unnest(?))
    """, [list(BAR_EVALUATION_TABLES)]).fetchall()
    result: dict[str, dict[str, str]] = {}
    for table, column, data_type in rows:
        result.setdefault(table, {})[column] = data_type
    return result


def _require_write_schema(
    conn: duckdb.DuckDBPyConnection, catalog: dict[str, dict[str, str]], *, legacy_only: bool = False,
) -> None:
    types = {
        "str": "VARCHAR", "int": "BIGINT", "float": "DOUBLE", "bool": "BOOLEAN",
        "timestamp": "TIMESTAMP WITH TIME ZONE",
    }
    runs = {col: "VARCHAR" for col in _RUN_ALL_COLUMNS if col not in _RUN_OPTIONAL_COLUMN_TYPES}
    runs["ingested_at_utc"] = types["timestamp"]
    required = {"bar_evaluation_runs": runs}
    for table, columns, ints, floats, bools in (
        ("bar_evaluation_ic", _IC_TABLE_COLUMNS, _IC_INT_COLUMNS, _IC_FLOAT_COLUMNS, _IC_BOOL_COLUMNS),
        ("bar_evaluation_quantiles", _QUANTILE_TABLE_COLUMNS, _QUANTILE_INT_COLUMNS,
         _QUANTILE_FLOAT_COLUMNS, _QUANTILE_BOOL_COLUMNS),
        ("bar_evaluation_cross_sections", _CROSS_SECTION_TABLE_COLUMNS, _CS_INT_COLUMNS,
         _CS_FLOAT_COLUMNS, _CS_BOOL_COLUMNS),
    ):
        required[table] = {col: types[_column_kind(col, ints, floats, bools, frozenset())] for col in columns}
    if not legacy_only:
        runs.update(_RUN_OPTIONAL_COLUMN_TYPES)
        required[_SPLIT_TABLE] = {col: kind.removesuffix(" NOT NULL") for col, kind in _SPLIT_COLUMN_TYPES.items()}
    primary_keys = dict(conn.execute("""
        SELECT table_name, constraint_column_names FROM duckdb_constraints()
        WHERE database_name = current_database() AND schema_name = 'main' AND constraint_type = 'PRIMARY KEY'
    """).fetchall())
    missing = []
    for table, columns in required.items():
        if table not in catalog:
            missing.append(table)
            continue
        expected_key = ["identity_hash"] if table == "bar_evaluation_runs" else ["identity_hash", "row_index"]
        if primary_keys.get(table) != expected_key:
            missing.append(f"{table} PRIMARY KEY ({', '.join(expected_key)})")
        for column, expected in columns.items():
            actual = catalog[table].get(column)
            if actual is None:
                missing.append(f"{table}.{column}")
            elif actual != expected:
                missing.append(f"{table}.{column} (expected {expected}, found {actual})")
    if missing:
        hint = "" if legacy_only else "; upgrade an older database explicitly with migrate_bar_evaluation_db(db_path)"
        raise BarEvaluationPreconditionError(
            f"Bar evaluation schema missing required tables/columns or has incompatible types: {', '.join(missing)}{hint}"
        )


def _column_kind(col: str, int_cols: frozenset[str], float_cols: frozenset[str],
                 bool_cols: frozenset[str], json_cols: frozenset[str]) -> str:
    if col in int_cols:
        return "int"
    if col in float_cols:
        return "float"
    if col in bool_cols:
        return "bool"
    if col in json_cols:
        return "json"
    if col == "signal_timestamp":
        return "timestamp"
    return "str"
