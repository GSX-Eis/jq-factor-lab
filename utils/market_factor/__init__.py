"""Append-only, versioned minute/bar factor value store (Scheme B).

Storage layout under an explicit store root:
  - parquet_data/factor_id=<fid>/frequency=<freq>/year_month=<YYYY-MM>/*.parquet (ZSTD)
  - identities.jsonl: append-only identity registry (factor_value_identities)
  - batches.jsonl: append-only committed shard manifest
  - .ledger.lock: POSIX fcntl.flock advisory lock protecting writers only

Readers never acquire the writer lock and never glob uncommitted Parquet files;
all reads and DuckDB views resolve strictly through committed batch records in
batches.jsonl. Note: fcntl.flock requires a POSIX host (macOS/Linux).
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import inspect
import json
import os
import pickle
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from backtest_and_analysis.time_series.contracts import DuplicateKeyError
from utils.bar_catalog import BarCatalog, table_exists
from utils.code_version import resolve_code_version
from utils.session_calendar import SessionCalendar
from utils.time_axis import BarSchedule, Frequency

from .bridge import (
    save_bar_factor_values,
)
from .identity import (
    build_identity_record,
    compute_factor_identity_hash,
)
from .schema import (
    DEFAULT_ROW_GROUP_SIZE,
    IDENTITY_COLUMNS,
    LEAF_ARROW_SCHEMA,
    LEAF_COLUMNS,
    MARKET_FACTOR_IDENTITY_SCHEMA_VERSION,
    NAN_CANONICAL_BITS,
    TABLE_COLUMNS,
    VALID_ADJUSTMENTS,
    VALID_ANALYSIS_MODES,
    VALID_BUCKET_STRATEGIES,
    VALID_VIEWS,
    VIEW_COLUMNS,
    FactorReadPlan,
    FactorValueConflictError,
    FactorWriteReceipt,
    canonical_float_bits,
    sql_canonical_float,
    validate_factor_id,
    validate_sha256_hex,
)
from .store import (
    MarketFactorValueStore,
)

__all__ = [
    "Any",
    "BarCatalog",
    "BarSchedule",
    "DEFAULT_ROW_GROUP_SIZE",
    "DuplicateKeyError",
    "FactorReadPlan",
    "FactorValueConflictError",
    "FactorWriteReceipt",
    "Frequency",
    "IDENTITY_COLUMNS",
    "Iterator",
    "LEAF_ARROW_SCHEMA",
    "LEAF_COLUMNS",
    "MARKET_FACTOR_IDENTITY_SCHEMA_VERSION",
    "Mapping",
    "MarketFactorValueStore",
    "NAN_CANONICAL_BITS",
    "Path",
    "Sequence",
    "SessionCalendar",
    "TABLE_COLUMNS",
    "VALID_ADJUSTMENTS",
    "VALID_ANALYSIS_MODES",
    "VALID_BUCKET_STRATEGIES",
    "VALID_VIEWS",
    "VIEW_COLUMNS",
    "annotations",
    "asdict",
    "build_identity_record",
    "canonical_float_bits",
    "compute_factor_identity_hash",
    "contextmanager",
    "dataclass",
    "dt",
    "duckdb",
    "fcntl",
    "hashlib",
    "inspect",
    "is_bool_dtype",
    "is_complex_dtype",
    "is_numeric_dtype",
    "json",
    "np",
    "os",
    "pa",
    "pd",
    "pickle",
    "pq",
    "re",
    "resolve_code_version",
    "save_bar_factor_values",
    "shutil",
    "sql_canonical_float",
    "table_exists",
    "uuid",
    "validate_factor_id",
    "validate_sha256_hex",
]
