"""DuckDB persistence and read-only query store for Bar evaluation results (Stage 6.6).

Stores published `time-series-analysis` evaluation artifacts in an explicitly
specified DuckDB database:
  - bar_evaluation_runs
  - bar_evaluation_ic
  - bar_evaluation_quantiles
  - bar_evaluation_cross_sections
  - bar_evaluation_split_metrics

`per_bar_labels.csv` is NOT stored in DuckDB; only its path and SHA256 digest
are recorded in `bar_evaluation_runs`.

Write locking reuses `utils.duckdb_results_writer.acquire_exclusive_write_conn`
so this module never creates `meta.*` tables or modifies any existing table
such as `factor_analysis_results` or `price_data_*`.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import pandas as pd

from backtest_and_analysis.time_series.contracts import ResultIdentity, sha256_file
from backtest_and_analysis.time_series.evaluator import (
    CS_COLUMNS,
    CS_IC_COLUMNS,
    CS_QUANTILE_COLUMNS,
    RESULT_SCHEMA_VERSION,
    TS_IC_COLUMNS,
    TS_QUANTILE_COLUMNS,
)
from utils.duckdb_results_writer import acquire_exclusive_write_conn
from utils.time_axis import Frequency

from .ingest import (
    ingest_evaluation,
    migrate_bar_evaluation_db,
)
from .query import (
    get_bar_evaluation_filter_options,
    query_bar_evaluation_detail,
    query_bar_evaluation_runs,
    query_bar_evaluations,
    summarize_bar_evaluation_connection,
    summarize_bar_evaluation_db,
)
from .schema import (
    BAR_EVALUATION_TABLES,
    SUPPORTED_RESULT_SCHEMA_VERSIONS,
    VALID_ANALYSIS_MODES,
    BarEvaluationConflictError,
    BarEvaluationIngestReceipt,
    BarEvaluationIntegrityError,
    BarEvaluationPreconditionError,
)

__all__ = [
    "Any",
    "BAR_EVALUATION_TABLES",
    "BarEvaluationConflictError",
    "BarEvaluationIngestReceipt",
    "BarEvaluationIntegrityError",
    "BarEvaluationPreconditionError",
    "CS_COLUMNS",
    "CS_IC_COLUMNS",
    "CS_QUANTILE_COLUMNS",
    "Frequency",
    "Iterator",
    "Mapping",
    "Path",
    "RESULT_SCHEMA_VERSION",
    "ResultIdentity",
    "SUPPORTED_RESULT_SCHEMA_VERSIONS",
    "Sequence",
    "TS_IC_COLUMNS",
    "TS_QUANTILE_COLUMNS",
    "VALID_ANALYSIS_MODES",
    "acquire_exclusive_write_conn",
    "annotations",
    "contextmanager",
    "dataclass",
    "dt",
    "duckdb",
    "get_bar_evaluation_filter_options",
    "ingest_evaluation",
    "json",
    "math",
    "migrate_bar_evaluation_db",
    "np",
    "pd",
    "query_bar_evaluation_detail",
    "query_bar_evaluation_runs",
    "query_bar_evaluations",
    "re",
    "sha256_file",
    "struct",
    "summarize_bar_evaluation_connection",
    "summarize_bar_evaluation_db",
]
