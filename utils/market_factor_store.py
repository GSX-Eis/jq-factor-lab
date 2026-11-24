"""Compatibility imports for `utils.market_factor`."""
import os as os

from backtest_and_analysis.time_series.contracts import _utc as _utc

from .market_factor import *  # noqa: F403
from .market_factor import __all__ as __all__
from .market_factor.bridge import (
    _factor_source_hashes as _factor_source_hashes,
)
from .market_factor.identity import (
    _REQUIRED_IDENTITY_PAYLOAD_KEYS as _REQUIRED_IDENTITY_PAYLOAD_KEYS,
    _normalize_calendar_mapping as _normalize_calendar_mapping,
    _normalize_params as _normalize_params,
    _normalize_rth_reference as _normalize_rth_reference,
    _normalize_source_build_id as _normalize_source_build_id,
    _reject_json_constant as _reject_json_constant,
    _validate_identity_record as _validate_identity_record,
)
from .market_factor.ledger import (
    _append_jsonl_line as _append_jsonl_line,
    _read_jsonl as _read_jsonl,
    _trim_unterminated_trailing_line as _trim_unterminated_trailing_line,
)
from .market_factor.schema import (
    _FACTOR_ID_RE as _FACTOR_ID_RE,
    _HIVE_TYPES as _HIVE_TYPES,
    _SESSION_DATE_RE as _SESSION_DATE_RE,
    _SHA256_RE as _SHA256_RE,
    _canonical_json as _canonical_json,
    _frame_digest as _frame_digest,
    _require_nonempty_str as _require_nonempty_str,
    _validate_session_date_str as _validate_session_date_str,
)
from .market_factor.writer import (
    _validate_and_prepare_batch as _validate_and_prepare_batch,
    _validate_calendars_for_batch as _validate_calendars_for_batch,
    _validate_text_series as _validate_text_series,
    _validate_utc_series as _validate_utc_series,
)
